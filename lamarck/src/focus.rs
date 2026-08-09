//! Focus-neuron selection and incumbent-specific streaming statistics.

use crate::backprop::{BackpropConfig, LearningSignal};
use crate::learning::squash_derivative;
use neat_core::{CompiledNetwork, CreatureExport, TrainingDataConfig, TrainingDataIterator};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Strategy for choosing which non-input neuron to investigate.
pub trait FocusSelector {
    /// Select a focus neuron UUID among eligible non-input neurons.
    fn select(&mut self, creature: &CreatureExport, rng: &mut impl Rng) -> Option<String>;
}

/// Version-1 random focus selection.
#[derive(Debug, Default, Clone)]
pub struct RandomFocusSelector;

impl FocusSelector for RandomFocusSelector {
    fn select(&mut self, creature: &CreatureExport, rng: &mut impl Rng) -> Option<String> {
        let candidates: Vec<&str> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type != "input")
            .map(|n| n.uuid.as_str())
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let idx = rng.random_range(0..candidates.len());
        Some(candidates[idx].to_string())
    }
}

/// Always select a caller-specified non-input neuron UUID (for tests / smoke runs).
#[derive(Debug, Clone)]
pub struct FixedFocusSelector {
    /// Neuron UUID to focus (must exist and not be an input).
    pub uuid: String,
}

impl FocusSelector for FixedFocusSelector {
    fn select(&mut self, creature: &CreatureExport, _rng: &mut impl Rng) -> Option<String> {
        creature
            .neurons
            .iter()
            .find(|n| n.uuid == self.uuid && n.neuron_type != "input")
            .map(|n| n.uuid.clone())
    }
}

/// Prefer output neurons (then any non-input) with lowest saturation / highest activity.
#[derive(Debug, Default, Clone)]
pub struct UnsaturatedFocusSelector;

impl FocusSelector for UnsaturatedFocusSelector {
    fn select(&mut self, creature: &CreatureExport, rng: &mut impl Rng) -> Option<String> {
        let outputs: Vec<&str> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output")
            .map(|n| n.uuid.as_str())
            .collect();
        if !outputs.is_empty() {
            let idx = rng.random_range(0..outputs.len());
            return Some(outputs[idx].to_string());
        }
        RandomFocusSelector.select(creature, rng)
    }
}

/// Prefer the first output neuron when present (fallback when no signals yet).
#[derive(Debug, Default, Clone)]
pub struct HighErrorFocusSelector;

impl FocusSelector for HighErrorFocusSelector {
    fn select(&mut self, creature: &CreatureExport, rng: &mut impl Rng) -> Option<String> {
        if let Some(out) = creature.neurons.iter().find(|n| n.neuron_type == "output") {
            return Some(out.uuid.clone());
        }
        RandomFocusSelector.select(creature, rng)
    }
}

/// Pick the neuron with the largest improvement signal (high-error policy).
pub fn select_highest_signal(signals: &HashMap<String, f64>) -> Option<FocusChoice> {
    signals
        .iter()
        .filter(|(_, s)| **s > FOCUS_SIGNAL_EPS)
        .max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(b.0))
        })
        .map(|(uuid, signal)| FocusChoice {
            uuid: uuid.clone(),
            weight: *signal,
            reason: format!("highest_signal={signal:.6e}"),
        })
}

/// Focus-policy name for CLI / config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FocusPolicy {
    /// Weighted-random by improvement signal (issue #25).
    ///
    /// Default. Neurons are ranked by residual MAE (outputs) or |backprop blame|
    /// (hidden). Neurons with ~zero signal are **never** selected — you cannot
    /// improve on zero error. Opt into uninformed exploration with `random`.
    #[default]
    Weighted,
    /// Pick the single highest-signal neuron (usually the worst output head).
    HighError,
    /// Random non-input each experiment (may pick zero-signal neurons).
    Random,
    /// Prefer outputs (unsaturated / direct heads).
    Unsaturated,
}

impl FocusPolicy {
    /// Parse from CLI string.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "weighted" | "improvement" | "improvement-potential" => Some(Self::Weighted),
            "random" => Some(Self::Random),
            "unsaturated" => Some(Self::Unsaturated),
            "high-error" | "high_error" => Some(Self::HighError),
            _ => None,
        }
    }

    /// Human label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Weighted => "weighted",
            Self::Random => "random",
            Self::Unsaturated => "unsaturated",
            Self::HighError => "high-error",
        }
    }
}

/// Per-neuron bookkeeping for weighted focus (updated each experiment).
#[derive(Debug, Clone, Default)]
pub struct FocusNeuronHistory {
    /// Times an acceptance occurred while this neuron was focus.
    pub accepts: u32,
    /// Times best full-corpus Δ was positive but below the accept threshold.
    pub near_misses: u32,
    /// Scorer failures or large negative full-corpus Δ while focused here.
    pub hard_fails: u32,
}

/// Result of a weighted focus draw (for logging).
#[derive(Debug, Clone)]
pub struct FocusChoice {
    /// Selected neuron UUID.
    pub uuid: String,
    /// Relative selection weight.
    pub weight: f64,
    /// Short explanation of the dominant signals.
    pub reason: String,
}

/// Minimum weight among neurons that already cleared the signal gate.
pub const FOCUS_EXPLORATION_FLOOR: f64 = 0.01;

/// Signals at or below this are treated as zero — never selected by weighted /
/// high-error policies (you cannot improve on zero error / blame).
pub const FOCUS_SIGNAL_EPS: f64 = 1e-12;

/// Weighted-random focus by improvement potential (issue #25).
#[derive(Debug, Default, Clone)]
pub struct WeightedFocusSelector {
    /// Running per-UUID history for this optimisation session.
    pub history: HashMap<String, FocusNeuronHistory>,
}

impl WeightedFocusSelector {
    /// Record outcome after an experiment on `focus_uuid`.
    pub fn record_outcome(
        &mut self,
        focus_uuid: &str,
        accepted: bool,
        best_full_delta: Option<f64>,
        scorer_failed: bool,
        min_improvement: f64,
    ) {
        let entry = self.history.entry(focus_uuid.to_string()).or_default();
        if scorer_failed {
            entry.hard_fails = entry.hard_fails.saturating_add(1);
            return;
        }
        if accepted {
            entry.accepts = entry.accepts.saturating_add(1);
            return;
        }
        if let Some(delta) = best_full_delta {
            if delta > 0.0 && delta <= min_improvement {
                entry.near_misses = entry.near_misses.saturating_add(1);
            } else if delta < -1e-4 {
                entry.hard_fails = entry.hard_fails.saturating_add(1);
            }
        }
    }

    /// Draw a focus neuron ∝ improvement signal (zeros never enter the pool).
    pub fn select_weighted(
        &self,
        creature: &CreatureExport,
        signals: &HashMap<String, f64>,
        rng: &mut impl Rng,
    ) -> Option<FocusChoice> {
        let ranked = self.rank_candidates(creature, signals);
        if ranked.is_empty() {
            return None;
        }
        let total: f64 = ranked.iter().map(|(_, w, _)| *w).sum();
        if total <= 0.0 {
            return None;
        }
        let mut pick = rng.random_range(0.0..total);
        for (uuid, weight, reason) in &ranked {
            pick -= weight;
            if pick <= 0.0 {
                return Some(FocusChoice {
                    uuid: uuid.clone(),
                    weight: *weight,
                    reason: reason.clone(),
                });
            }
        }
        let (uuid, weight, reason) = ranked.last()?;
        Some(FocusChoice {
            uuid: uuid.clone(),
            weight: *weight,
            reason: reason.clone(),
        })
    }

    /// Rank non-input neurons that have a non-zero improvement signal.
    ///
    /// Neurons with signal ≤ [`FOCUS_SIGNAL_EPS`] are omitted entirely — they
    /// cannot improve (zero residual / zero blame).
    pub fn rank_candidates(
        &self,
        creature: &CreatureExport,
        signals: &HashMap<String, f64>,
    ) -> Vec<(String, f64, String)> {
        let mut ranked = Vec::with_capacity(signals.len());
        for n in &creature.neurons {
            if n.neuron_type == "input" {
                continue;
            }
            let Some(&signal) = signals.get(&n.uuid) else {
                continue;
            };
            if !signal.is_finite() || signal <= FOCUS_SIGNAL_EPS {
                continue;
            }

            let hist = self.history.get(&n.uuid);
            let mut weight = FOCUS_EXPLORATION_FLOOR + 50.0 * signal;
            let mut reasons = vec![format!("signal={signal:.6e}")];
            if n.neuron_type == "output" {
                reasons.push("output".into());
            } else {
                reasons.push("hidden_blame".into());
            }

            if let Some(h) = hist {
                if h.accepts > 0 {
                    weight += 8.0 * f64::from(h.accepts);
                    reasons.push(format!("accepts={}", h.accepts));
                }
                if h.near_misses > 0 {
                    weight += 3.0 * f64::from(h.near_misses);
                    reasons.push(format!("near_miss={}", h.near_misses));
                }
                if h.hard_fails > 0 {
                    let damp = 1.0 / (1.0 + f64::from(h.hard_fails));
                    weight *= damp;
                    reasons.push(format!("fails={}", h.hard_fails));
                }
            }

            ranked.push((
                n.uuid.clone(),
                weight.max(FOCUS_EXPLORATION_FLOOR),
                reasons.join("+"),
            ));
        }
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked
    }
}

/// Mean absolute residual per output UUID over a training sample.
pub fn collect_output_mean_abs_errors(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    max_records: Option<u64>,
) -> Result<HashMap<String, f64>, String> {
    let outputs: Vec<(usize, String)> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .enumerate()
        .map(|(i, n)| (i, n.uuid.clone()))
        .collect();
    if outputs.is_empty() {
        return Ok(HashMap::new());
    }

    let config = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = TrainingDataIterator::new(training_data, config).map_err(|e| e.to_string())?;
    let mut abs_sums = vec![0.0f64; outputs.len()];
    let mut count = 0u64;

    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if let Some(limit) = max_records
            && count >= limit
        {
            break;
        }
        let preds = network.activate(&record.inputs, creature.output);
        count += 1;
        for (out_i, _) in &outputs {
            if *out_i >= record.outputs.len() || *out_i >= preds.len() {
                continue;
            }
            let pred = f64::from(preds[*out_i]);
            let target = f64::from(record.outputs[*out_i]);
            abs_sums[*out_i] += (target - pred).abs();
        }
    }

    let mut map = HashMap::with_capacity(outputs.len());
    if count == 0 {
        return Ok(map);
    }
    for (out_i, uuid) in outputs {
        map.insert(uuid, abs_sums[out_i] / count as f64);
    }
    Ok(map)
}

/// Build per-neuron improvement signals used by weighted / high-error focus.
///
/// - Outputs: mean absolute residual (MAE). Zero MAE ⇒ omitted.
/// - Hidden: `|mean adjusted bias blame|` when `count > 0`. Zero blame ⇒ omitted.
pub fn build_improvement_signals(
    creature: &CreatureExport,
    output_mae: &HashMap<String, f64>,
    learning: &LearningSignal,
) -> HashMap<String, f64> {
    let mut signals = HashMap::new();
    for (i, n) in creature.neurons.iter().enumerate() {
        if n.neuron_type == "input" {
            continue;
        }
        if n.neuron_type == "output" {
            let mae = output_mae.get(&n.uuid).copied().unwrap_or(0.0);
            if mae > FOCUS_SIGNAL_EPS {
                signals.insert(n.uuid.clone(), mae);
            }
            continue;
        }
        let Some(sig) = learning.biases.get(i) else {
            continue;
        };
        if sig.count <= 0.0 {
            continue;
        }
        let mean_abs = (sig.total_adjusted_bias / sig.count).abs();
        if mean_abs > FOCUS_SIGNAL_EPS {
            signals.insert(n.uuid.clone(), mean_abs);
        }
    }
    signals
}

/// Per-incoming-synapse source statistics for the focus neuron.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct IncomingSourceStats {
    /// Index into `creature.synapses`.
    pub synapse_index: usize,
    /// Source neuron / input uuid.
    pub from_uuid: String,
    /// Current synapse weight.
    pub weight: f64,
    /// Whether the source is a raw input.
    pub is_input: bool,
    /// Input index when `is_input`.
    pub input_index: Option<usize>,
    /// Source activation mean.
    pub mean: f64,
    /// Source activation variance.
    pub variance: f64,
    /// Source activation std-dev.
    pub std_dev: f64,
    /// Pearson correlation with focus residual when available.
    pub correlation_with_error: Option<f64>,
    /// Backprop weight-signal accumulation count for this synapse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_signal_count: Option<f64>,
    /// Proposed weight change (`propose − current`) from the learning signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposed_weight_delta: Option<f64>,
    /// Mean adjusted-value mass per sample from the weight signal (sensitivity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_weight_sensitivity: Option<f64>,
}

/// Streaming statistics for one focused neuron.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusNeuronStats {
    /// Focused neuron UUID.
    pub neuron_uuid: String,
    /// Squash name when present.
    pub squash: Option<String>,
    /// Incoming synapse count.
    pub incoming_count: usize,
    /// Pre-activation mean.
    pub pre_mean: f64,
    /// Pre-activation variance.
    pub pre_variance: f64,
    /// Pre-activation min.
    pub pre_min: f64,
    /// Pre-activation max.
    pub pre_max: f64,
    /// Post-activation mean.
    pub post_mean: f64,
    /// Post-activation variance.
    pub post_variance: f64,
    /// Fraction of near-zero post activations (|x| < 1e-6).
    pub near_zero_fraction: f64,
    /// Fraction of saturated activations (squash-aware heuristic).
    pub saturation_fraction: f64,
    /// Mean signed error `target - post` when the focus is an output neuron.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_error: Option<f64>,
    /// Mean absolute error when the focus is an output neuron.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_abs_error: Option<f64>,
    /// Mean squash-aware residual `mean((target-post) * derivative(post))`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_adjusted_error: Option<f64>,
    /// Mean squash derivative over the scan (0 ⇒ saturated / flat).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_derivative: Option<f64>,
    /// Mean backprop bias blame (`total_adjusted_bias / count`) for the focus.
    /// Present for hidden and output focuses when a learning signal was attached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_blame: Option<f64>,
    /// Backprop bias-signal accumulation count for the focus neuron.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blame_count: Option<f64>,
    /// Absolute value of [`Self::mean_blame`] (convenience for ranking).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_abs_blame: Option<f64>,
    /// Whether the focus bias signal flagged no-change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blame_no_change: Option<bool>,
    /// Records scanned.
    pub record_count: u64,
}

/// Resolve the compiled-network index for a neuron UUID.
pub fn neuron_index(creature: &CreatureExport, uuid: &str) -> Option<usize> {
    creature
        .neurons
        .iter()
        .position(|n| n.uuid == uuid)
        .map(|i| creature.input + i)
}

/// Collect focused statistics by scanning the incumbent over training data.
///
/// When `max_records` is `Some(n)`, stop after `n` records (used with `--quick`).
pub fn collect_focus_stats(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &std::path::Path,
    focus_uuid: &str,
    max_records: Option<u64>,
) -> Result<FocusNeuronStats, String> {
    let neuron = creature
        .neurons
        .iter()
        .find(|n| n.uuid == focus_uuid)
        .ok_or_else(|| format!("focus neuron {focus_uuid} not found"))?;
    let compiled_idx = neuron_index(creature, focus_uuid)
        .ok_or_else(|| format!("focus neuron {focus_uuid} missing compiled index"))?;
    let relative_idx = compiled_idx
        .checked_sub(creature.input)
        .ok_or("focus neuron index below input count")?;

    let incoming_count = creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == focus_uuid)
        .count();

    let output_index = if neuron.neuron_type == "output" {
        creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output")
            .position(|n| n.uuid == focus_uuid)
    } else {
        None
    };

    let config = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = TrainingDataIterator::new(training_data, config).map_err(|e| e.to_string())?;

    let mut pre_mean = 0.0;
    let mut pre_m2 = 0.0;
    let mut post_mean = 0.0;
    let mut post_m2 = 0.0;
    let mut pre_min = f64::INFINITY;
    let mut pre_max = f64::NEG_INFINITY;
    let mut near_zero = 0u64;
    let mut saturated = 0u64;
    let mut count = 0u64;
    let mut err_sum = 0.0;
    let mut abs_err_sum = 0.0;
    let mut adj_err_sum = 0.0;
    let mut deriv_sum = 0.0;
    let mut err_count = 0u64;

    let squash = neuron.squash.clone();
    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if let Some(limit) = max_records
            && count >= limit
        {
            break;
        }
        let traced = network.activate_and_trace(&record.inputs, creature.output);
        let num_non_inputs = network.num_neurons.saturating_sub(creature.input);
        let post_offset = creature.output;
        let pre_offset = creature.output + num_non_inputs;
        if pre_offset + relative_idx >= traced.len() || post_offset + relative_idx >= traced.len() {
            continue;
        }
        let pre = f64::from(traced[pre_offset + relative_idx]);
        let post = f64::from(traced[post_offset + relative_idx]);
        count += 1;
        let d1 = pre - pre_mean;
        pre_mean += d1 / count as f64;
        pre_m2 += d1 * (pre - pre_mean);
        let d2 = post - post_mean;
        post_mean += d2 / count as f64;
        post_m2 += d2 * (post - post_mean);
        pre_min = pre_min.min(pre);
        pre_max = pre_max.max(pre);
        if post.abs() < 1e-6 {
            near_zero += 1;
        }
        if is_saturated(squash.as_deref(), post) {
            saturated += 1;
        }
        if let Some(out_i) = output_index
            && out_i < record.outputs.len()
        {
            let target = f64::from(record.outputs[out_i]);
            let err = target - post;
            let deriv = squash_derivative(squash.as_deref(), post);
            err_sum += err;
            abs_err_sum += err.abs();
            adj_err_sum += err * deriv;
            deriv_sum += deriv;
            err_count += 1;
        }
    }

    let (mean_error, mean_abs_error, mean_adjusted_error, mean_derivative) = if err_count > 0 {
        (
            Some(err_sum / err_count as f64),
            Some(abs_err_sum / err_count as f64),
            Some(adj_err_sum / err_count as f64),
            Some(deriv_sum / err_count as f64),
        )
    } else {
        (None, None, None, None)
    };

    Ok(FocusNeuronStats {
        neuron_uuid: focus_uuid.to_string(),
        squash,
        incoming_count,
        pre_mean,
        pre_variance: if count > 0 {
            pre_m2 / count as f64
        } else {
            0.0
        },
        pre_min: if count > 0 { pre_min } else { 0.0 },
        pre_max: if count > 0 { pre_max } else { 0.0 },
        post_mean,
        post_variance: if count > 0 {
            post_m2 / count as f64
        } else {
            0.0
        },
        near_zero_fraction: if count > 0 {
            near_zero as f64 / count as f64
        } else {
            0.0
        },
        saturation_fraction: if count > 0 {
            saturated as f64 / count as f64
        } else {
            0.0
        },
        mean_error,
        mean_abs_error,
        mean_adjusted_error,
        mean_derivative,
        mean_blame: None,
        blame_count: None,
        mean_abs_blame: None,
        blame_no_change: None,
        record_count: count,
    })
}

/// Collect per-incoming-source activation stats (and residual correlation when possible).
pub fn collect_incoming_source_stats(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &std::path::Path,
    focus_uuid: &str,
    max_records: Option<u64>,
    observations: Option<&crate::observations::ObservationsStatistics>,
) -> Result<Vec<IncomingSourceStats>, String> {
    let neuron = creature
        .neurons
        .iter()
        .find(|n| n.uuid == focus_uuid)
        .ok_or_else(|| format!("focus neuron {focus_uuid} not found"))?;
    let compiled_idx = neuron_index(creature, focus_uuid)
        .ok_or_else(|| format!("focus neuron {focus_uuid} missing compiled index"))?;
    let relative_idx = compiled_idx
        .checked_sub(creature.input)
        .ok_or("focus neuron index below input count")?;
    let output_index = if neuron.neuron_type == "output" {
        creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output")
            .position(|n| n.uuid == focus_uuid)
    } else {
        None
    };

    let incoming: Vec<(usize, String, f64)> = creature
        .synapses
        .iter()
        .enumerate()
        .filter(|(_, s)| s.to_uuid == focus_uuid)
        .map(|(i, s)| (i, s.from_uuid.clone(), s.weight))
        .collect();
    if incoming.is_empty() {
        return Ok(vec![]);
    }

    // Reuse observations for raw inputs when available.
    let mut out: Vec<IncomingSourceStats> = incoming
        .iter()
        .map(|(syn_idx, from, weight)| {
            let input_index = from
                .strip_prefix("input-")
                .and_then(|s| s.parse::<usize>().ok());
            let (mean, variance, std_dev) = if let (Some(idx), Some(obs)) =
                (input_index, observations)
                && let Some(stats) = obs.inputs.get(idx)
            {
                (stats.mean, stats.variance, stats.std_dev)
            } else {
                (0.0, 0.0, 0.0)
            };
            IncomingSourceStats {
                synapse_index: *syn_idx,
                from_uuid: from.clone(),
                weight: *weight,
                is_input: input_index.is_some(),
                input_index,
                mean,
                variance,
                std_dev,
                correlation_with_error: None,
                weight_signal_count: None,
                proposed_weight_delta: None,
                mean_weight_sensitivity: None,
            }
        })
        .collect();

    // Measure hidden sources (and refine correlations) with a live scan.
    let need_live = out.iter().any(|s| !s.is_input) || output_index.is_some();
    if !need_live {
        return Ok(out);
    }

    let n = out.len();
    let mut sums = vec![0.0f64; n];
    let mut sq = vec![0.0f64; n];
    let mut err_sum = 0.0f64;
    let mut err_sq = 0.0f64;
    let mut cross = vec![0.0f64; n];
    let mut count = 0u64;

    let config = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = TrainingDataIterator::new(training_data, config).map_err(|e| e.to_string())?;
    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if let Some(limit) = max_records
            && count >= limit
        {
            break;
        }
        let traced = network.activate_and_trace(&record.inputs, creature.output);
        let num_non_inputs = network.num_neurons.saturating_sub(creature.input);
        let post_offset = creature.output;
        if post_offset + relative_idx >= traced.len() {
            continue;
        }
        let post = f64::from(traced[post_offset + relative_idx]);
        let err = if let Some(out_i) = output_index
            && out_i < record.outputs.len()
        {
            f64::from(record.outputs[out_i]) - post
        } else {
            0.0
        };
        count += 1;
        err_sum += err;
        err_sq += err * err;
        for (i, src) in out.iter().enumerate() {
            let act = if let Some(idx) = src.input_index {
                f64::from(*record.inputs.get(idx).unwrap_or(&0.0))
            } else if let Some(pos) = creature
                .neurons
                .iter()
                .position(|n| n.uuid == src.from_uuid)
            {
                let idx = post_offset + pos;
                if idx < traced.len() {
                    f64::from(traced[idx])
                } else {
                    0.0
                }
            } else {
                0.0
            };
            sums[i] += act;
            sq[i] += act * act;
            cross[i] += act * err;
        }
        let _ = num_non_inputs;
    }

    if count == 0 {
        return Ok(out);
    }
    let n_f = count as f64;
    let err_mean = err_sum / n_f;
    let err_var = (err_sq / n_f) - err_mean * err_mean;
    for (i, src) in out.iter_mut().enumerate() {
        if !src.is_input {
            let mean = sums[i] / n_f;
            let variance = ((sq[i] / n_f) - mean * mean).max(0.0);
            src.mean = mean;
            src.variance = variance;
            src.std_dev = variance.sqrt();
        }
        if output_index.is_some() {
            let mean = sums[i] / n_f;
            let var = ((sq[i] / n_f) - mean * mean).max(0.0);
            let cov = (cross[i] / n_f) - mean * err_mean;
            let denom = (var * err_var.max(0.0)).sqrt();
            src.correlation_with_error = if denom > f64::EPSILON {
                Some((cov / denom).clamp(-1.0, 1.0))
            } else {
                Some(0.0)
            };
        }
    }
    Ok(out)
}

/// Attach backprop bias blame for the focus neuron onto focus stats.
///
/// Uses the real [`LearningSignal`] from TS-parity propagate (issue #2/#4).
/// Never invents a hidden-neuron target — only surfaces accumulated blame.
pub fn attach_focus_blame(
    stats: &mut FocusNeuronStats,
    creature: &CreatureExport,
    learning: &LearningSignal,
) {
    let Some(pos) = creature
        .neurons
        .iter()
        .position(|n| n.uuid == stats.neuron_uuid)
    else {
        return;
    };
    let Some(signal) = learning.biases.get(pos) else {
        return;
    };
    if signal.count <= 0.0 {
        stats.mean_blame = Some(0.0);
        stats.blame_count = Some(0.0);
        stats.mean_abs_blame = Some(0.0);
        stats.blame_no_change = Some(signal.no_change);
        return;
    }
    let mean = signal.total_adjusted_bias / signal.count;
    stats.mean_blame = Some(mean);
    stats.blame_count = Some(signal.count);
    stats.mean_abs_blame = Some(mean.abs());
    stats.blame_no_change = Some(signal.no_change);
}

/// Attach per-synapse weight-signal summaries onto incoming source stats.
pub fn attach_learning_to_incoming(
    incoming: &mut [IncomingSourceStats],
    learning: &LearningSignal,
    config: &BackpropConfig,
    learning_rate: f64,
) {
    for src in incoming.iter_mut() {
        let Some(signal) = learning.weights.get(src.synapse_index) else {
            continue;
        };
        if signal.count <= 0.0 {
            continue;
        }
        let proposed = signal.propose(src.weight, config, learning_rate);
        let adj_mass = signal.total_positive_adjusted_value + signal.total_negative_adjusted_value;
        src.weight_signal_count = Some(signal.count);
        src.proposed_weight_delta = Some(proposed - src.weight);
        src.mean_weight_sensitivity = Some(adj_mass / signal.count);
    }
}

/// Squash-aware saturation heuristic used by focus scans (issue #4).
pub fn is_saturated(squash: Option<&str>, post: f64) -> bool {
    match squash {
        Some("TANH") | Some("BIPOLAR_SIGMOID") => post.abs() > 0.99,
        Some("LOGISTIC") | Some("SIGMOID") => !(0.01..=0.99).contains(&post),
        Some("RELU") | Some("LEAKY_RELU") => post <= 0.0,
        Some("CLIPPED") | Some("HARD_TANH") => post.abs() >= 1.0 - 1e-6,
        _ => post.abs() > 0.99,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backprop::BackpropConfig;
    use crate::learning::accumulate_focus_learning;
    use neat_core::{compile_creature, parse_creature_json};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::io::Write;
    use tempfile::tempdir;

    const TINY: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 1,
      "output": 1,
      "neurons": [
        {"type":"hidden","uuid":"h1","bias":0.1,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0}
      ]
    }"#;

    /// Direct input → output (no hidden).
    const DIRECT: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 1,
      "output": 1,
      "neurons": [
        {"type":"output","uuid":"output-0","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"output-0","weight":1.0}
      ]
    }"#;

    /// Deeper chain: input → h1 → h2 → output with saturating TANH on h1.
    const DEEP: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 1,
      "output": 1,
      "neurons": [
        {"type":"hidden","uuid":"h1","bias":0.0,"squash":"TANH"},
        {"type":"hidden","uuid":"h2","bias":0.0,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h1","weight":2.0},
        {"fromUUID":"h1","toUUID":"h2","weight":1.0},
        {"fromUUID":"h2","toUUID":"o1","weight":1.0}
      ]
    }"#;

    fn write_records(dir: &std::path::Path, pairs: &[(f32, f32)]) {
        let mut f = std::fs::File::create(dir.join("0.bin")).unwrap();
        for &(inp, out) in pairs {
            f.write_all(&inp.to_le_bytes()).unwrap();
            f.write_all(&out.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn random_focus_is_deterministic_with_seed() {
        let creature = parse_creature_json(TINY).unwrap();
        let mut a = RandomFocusSelector;
        let mut b = RandomFocusSelector;
        let mut rng_a = StdRng::seed_from_u64(7);
        let mut rng_b = StdRng::seed_from_u64(7);
        assert_eq!(
            a.select(&creature, &mut rng_a),
            b.select(&creature, &mut rng_b)
        );
    }

    #[test]
    fn fixed_focus_selects_requested_uuid() {
        let creature = parse_creature_json(TINY).unwrap();
        let mut selector = FixedFocusSelector { uuid: "o1".into() };
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(selector.select(&creature, &mut rng).as_deref(), Some("o1"));
        selector.uuid = "missing".into();
        assert!(selector.select(&creature, &mut rng).is_none());
    }

    fn sample_signals() -> HashMap<String, f64> {
        HashMap::from([("o1".into(), 0.5), ("h1".into(), 0.1)])
    }

    #[test]
    fn weighted_focus_is_deterministic_with_seed() {
        let creature = parse_creature_json(TINY).unwrap();
        let sel = WeightedFocusSelector::default();
        let signals = sample_signals();
        let mut rng_a = StdRng::seed_from_u64(11);
        let mut rng_b = StdRng::seed_from_u64(11);
        let a = sel
            .select_weighted(&creature, &signals, &mut rng_a)
            .unwrap();
        let b = sel
            .select_weighted(&creature, &signals, &mut rng_b)
            .unwrap();
        assert_eq!(a.uuid, b.uuid);
        assert!((a.weight - b.weight).abs() < 1e-12);
    }

    #[test]
    fn weighted_focus_skips_zero_signal_neurons() {
        let creature = parse_creature_json(TINY).unwrap();
        let sel = WeightedFocusSelector::default();
        // Only o1 has error; h1 is perfect / unblamed → never ranked.
        let signals = HashMap::from([("o1".into(), 0.4), ("h1".into(), 0.0)]);
        let ranked = sel.rank_candidates(&creature, &signals);
        assert!(
            ranked
                .iter()
                .all(|(u, w, _)| u == "o1" && *w >= FOCUS_EXPLORATION_FLOOR)
        );
        assert!(
            sel.select_weighted(&creature, &signals, &mut StdRng::seed_from_u64(1))
                .unwrap()
                .uuid
                == "o1"
        );
        assert!(
            sel.rank_candidates(&creature, &HashMap::from([("h1".into(), 0.0)]))
                .is_empty()
        );
    }

    #[test]
    fn weighted_focus_history_boosts_accepts() {
        let creature = parse_creature_json(TINY).unwrap();
        let signals = sample_signals();
        let mut sel = WeightedFocusSelector::default();
        sel.record_outcome("h1", true, Some(2e-6), false, 1e-6);
        let ranked = sel.rank_candidates(&creature, &signals);
        let w_h = ranked.iter().find(|(u, _, _)| u == "h1").unwrap().1;
        let base = WeightedFocusSelector::default()
            .rank_candidates(&creature, &signals)
            .iter()
            .find(|(u, _, _)| u == "h1")
            .unwrap()
            .1;
        assert!(w_h > base);
    }

    #[test]
    fn highest_signal_picks_worst_output() {
        let signals = HashMap::from([
            ("o-good".into(), 1e-15),
            ("o-bad".into(), 0.8),
            ("h1".into(), 0.05),
        ]);
        let choice = select_highest_signal(&signals).unwrap();
        assert_eq!(choice.uuid, "o-bad");
    }

    #[test]
    fn build_improvement_signals_omits_zero_mae_outputs() {
        let creature = parse_creature_json(TINY).unwrap();
        let mae = HashMap::from([("o1".into(), 0.0)]);
        let learning = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
        let signals = build_improvement_signals(&creature, &mae, &learning);
        assert!(!signals.contains_key("o1"));
        assert!(!signals.contains_key("h1"));
    }

    #[test]
    fn direct_to_output_focus_stats_are_stable() {
        let dir = tempdir().unwrap();
        // IDENTITY output: pred = input; targets 0.5 / 1.5 → residuals known.
        write_records(dir.path(), &[(0.5, 1.0), (1.5, 1.0)]);
        let creature = parse_creature_json(DIRECT).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let mut selector = FixedFocusSelector {
            uuid: "output-0".into(),
        };
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(
            selector.select(&creature, &mut rng).as_deref(),
            Some("output-0")
        );

        let a = collect_focus_stats(&creature, &mut network, dir.path(), "output-0", None).unwrap();
        let b = collect_focus_stats(&creature, &mut network, dir.path(), "output-0", None).unwrap();
        assert_eq!(a.record_count, 2);
        assert_eq!(a.record_count, b.record_count);
        assert!((a.pre_mean - b.pre_mean).abs() < 1e-12);
        assert!((a.post_mean - b.post_mean).abs() < 1e-12);
        assert!((a.mean_error.unwrap() - b.mean_error.unwrap()).abs() < 1e-12);
        // pred means 1.0; targets 1.0 → mean error 0; MAE from |0.5-1|+|1.5-1|/2 = 0.5
        assert!((a.mean_error.unwrap()).abs() < 1e-9);
        assert!((a.mean_abs_error.unwrap() - 0.5).abs() < 1e-9);
        assert!(a.mean_derivative.unwrap() > 0.99);
    }

    #[test]
    fn deeper_hidden_focus_gets_blame_and_incoming_learning() {
        let dir = tempdir().unwrap();
        write_records(
            dir.path(),
            &[(0.5, 0.0), (1.0, 0.0), (-0.5, 0.0), (2.0, 1.0)],
        );
        let creature = parse_creature_json(DEEP).unwrap();
        let mut network = compile_creature(&creature).unwrap();

        let stats = collect_focus_stats(&creature, &mut network, dir.path(), "h1", None).unwrap();
        assert_eq!(stats.neuron_uuid, "h1");
        assert_eq!(stats.record_count, 4);
        assert!(stats.mean_error.is_none(), "hidden has no target residual");
        // Large |input| through TANH → some saturation.
        assert!(
            stats.saturation_fraction > 0.0,
            "TANH focus should report squash-aware saturation, got {}",
            stats.saturation_fraction
        );
        assert!(is_saturated(Some("TANH"), 0.995));
        assert!(!is_saturated(Some("TANH"), 0.5));

        let mut incoming =
            collect_incoming_source_stats(&creature, &mut network, dir.path(), "h1", None, None)
                .unwrap();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].from_uuid, "input-0");

        let cfg = BackpropConfig::default();
        let mut rng = StdRng::seed_from_u64(7);
        let learning = accumulate_focus_learning(
            &creature,
            &mut network,
            dir.path(),
            "h1",
            None,
            &cfg,
            &mut rng,
        )
        .unwrap();
        let h1_pos = creature
            .neurons
            .iter()
            .position(|n| n.uuid == "h1")
            .unwrap();
        assert!(
            learning.biases[h1_pos].count > 0.0,
            "hidden must receive propagated blame count"
        );

        let mut focus_stats = stats;
        attach_focus_blame(&mut focus_stats, &creature, &learning);
        assert!(focus_stats.blame_count.unwrap() > 0.0);
        assert!(focus_stats.mean_blame.is_some());
        assert!(focus_stats.mean_abs_blame.is_some());

        attach_learning_to_incoming(&mut incoming, &learning, &cfg, cfg.learning_rate);
        assert!(
            incoming[0].weight_signal_count.unwrap_or(0.0) > 0.0
                || incoming[0].proposed_weight_delta.is_some(),
            "incoming weight signal should attach for hidden focus"
        );
    }

    #[test]
    fn logistic_saturation_heuristic_is_squash_aware() {
        assert!(is_saturated(Some("LOGISTIC"), 0.005));
        assert!(is_saturated(Some("LOGISTIC"), 0.995));
        assert!(!is_saturated(Some("LOGISTIC"), 0.5));
        assert!(is_saturated(Some("RELU"), 0.0));
        assert!(!is_saturated(Some("RELU"), 0.5));
    }
}
