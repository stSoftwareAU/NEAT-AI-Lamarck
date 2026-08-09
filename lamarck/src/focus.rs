//! Focus-neuron selection and incumbent-specific streaming statistics.

use crate::backprop::{BackpropConfig, LearningSignal};
use crate::learning::squash_derivative;
use crate::observations::ObservationsStatistics;
use neat_core::{CompiledNetwork, CreatureExport, TrainingDataConfig, TrainingDataIterator};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

/// Prefer the first output neuron when present (high-error proxy before a scan).
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

/// Focus-policy name for CLI / config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FocusPolicy {
    /// Prefer first output neuron (error-bearing head).
    ///
    /// Default: only outputs have training targets, so residual / mean-error
    /// candidates need an output focus. Hidden exploration remains available
    /// via `--focus-policy weighted|random`.
    #[default]
    HighError,
    /// Weighted-random by estimated improvement potential (issue #25).
    Weighted,
    /// Random non-input each experiment.
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

/// Minimum weight so every eligible neuron retains a non-zero draw chance.
pub const FOCUS_EXPLORATION_FLOOR: f64 = 1.0;

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

    /// Draw a focus neuron ∝ weight, with exploration floor.
    pub fn select_weighted(
        &self,
        creature: &CreatureExport,
        observations: Option<&ObservationsStatistics>,
        rng: &mut impl Rng,
    ) -> Option<FocusChoice> {
        let ranked = self.rank_candidates(creature, observations);
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

    /// Compute (uuid, weight, reason) for every eligible non-input neuron.
    pub fn rank_candidates(
        &self,
        creature: &CreatureExport,
        observations: Option<&ObservationsStatistics>,
    ) -> Vec<(String, f64, String)> {
        let incoming_counts = incoming_counts(creature);
        let output_index: HashMap<&str, usize> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output")
            .enumerate()
            .map(|(i, n)| (n.uuid.as_str(), i))
            .collect();

        let mut ranked = Vec::with_capacity(creature.neurons.len());
        for n in &creature.neurons {
            if n.neuron_type == "input" {
                continue;
            }
            let hist = self.history.get(&n.uuid);
            let incoming = *incoming_counts.get(&n.uuid).unwrap_or(&0);
            let mut weight = FOCUS_EXPLORATION_FLOOR;
            let mut reasons = Vec::new();

            if n.neuron_type == "output" {
                weight += 20.0;
                reasons.push("output".into());
                if let Some(obs) = observations
                    && let Some(&out_i) = output_index.get(n.uuid.as_str())
                    && let Some(stats) = obs.outputs.get(out_i)
                {
                    // Harder targets → more room for focus work.
                    let signal = stats.mean_abs.max(stats.std_dev).max(0.0);
                    let bump = 10.0 * signal.min(5.0);
                    weight += bump;
                    reasons.push(format!("target_scale={signal:.3}"));
                }
            } else {
                let deg = (incoming as f64).min(40.0);
                weight += 0.05 * deg;
                if incoming > 0 {
                    reasons.push(format!("in={incoming}"));
                }
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

            if reasons.is_empty() {
                reasons.push("explore".into());
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

fn incoming_counts(creature: &CreatureExport) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for s in &creature.synapses {
        *map.entry(s.to_uuid.clone()).or_default() += 1;
    }
    map
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

    #[test]
    fn weighted_focus_is_deterministic_with_seed() {
        let creature = parse_creature_json(TINY).unwrap();
        let sel = WeightedFocusSelector::default();
        let mut rng_a = StdRng::seed_from_u64(11);
        let mut rng_b = StdRng::seed_from_u64(11);
        let a = sel.select_weighted(&creature, None, &mut rng_a).unwrap();
        let b = sel.select_weighted(&creature, None, &mut rng_b).unwrap();
        assert_eq!(a.uuid, b.uuid);
        assert!((a.weight - b.weight).abs() < 1e-12);
    }

    #[test]
    fn weighted_focus_keeps_exploration_floor() {
        let creature = parse_creature_json(TINY).unwrap();
        let sel = WeightedFocusSelector::default();
        let ranked = sel.rank_candidates(&creature, None);
        assert!(ranked.iter().all(|(_, w, _)| *w >= FOCUS_EXPLORATION_FLOOR));
        // Output should outrank a plain hidden with no history.
        let w_out = ranked.iter().find(|(u, _, _)| u == "o1").unwrap().1;
        let w_h = ranked.iter().find(|(u, _, _)| u == "h1").unwrap().1;
        assert!(w_out > w_h);
    }

    #[test]
    fn weighted_focus_history_boosts_accepts() {
        let creature = parse_creature_json(TINY).unwrap();
        let mut sel = WeightedFocusSelector::default();
        sel.record_outcome("h1", true, Some(2e-6), false, 1e-6);
        let ranked = sel.rank_candidates(&creature, None);
        let w_h = ranked.iter().find(|(u, _, _)| u == "h1").unwrap().1;
        let base = WeightedFocusSelector::default()
            .rank_candidates(&creature, None)
            .iter()
            .find(|(u, _, _)| u == "h1")
            .unwrap()
            .1;
        assert!(w_h > base);
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
