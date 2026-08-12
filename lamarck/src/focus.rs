//! Focus-neuron selection and incumbent-specific streaming statistics.

use crate::backprop::{BackpropConfig, LearningSignal};
use crate::learning::squash_derivative;
use neat_core::{CompiledNetwork, CreatureExport, TrainingDataConfig};
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
        select_random_excluding(creature, &[], rng)
    }
}

/// Draw a random non-input neuron that is not already in `excluded` (issue #109).
///
/// With an empty exclusion list this is exactly [`RandomFocusSelector::select`]
/// — one `random_range` draw — so a single-focus experiment keeps its rng
/// stream.
pub fn select_random_excluding(
    creature: &CreatureExport,
    excluded: &[String],
    rng: &mut impl Rng,
) -> Option<String> {
    let candidates: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type != "input")
        .map(|n| n.uuid.as_str())
        .filter(|uuid| !excluded.iter().any(|e| e == uuid))
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let idx = rng.random_range(0..candidates.len());
    Some(candidates[idx].to_string())
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
        select_unsaturated_excluding(creature, &[], rng)
    }
}

/// Prefer an unselected output, else any unselected non-input (issue #109).
pub fn select_unsaturated_excluding(
    creature: &CreatureExport,
    excluded: &[String],
    rng: &mut impl Rng,
) -> Option<String> {
    let outputs: Vec<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .map(|n| n.uuid.as_str())
        .filter(|uuid| !excluded.iter().any(|e| e == uuid))
        .collect();
    if !outputs.is_empty() {
        let idx = rng.random_range(0..outputs.len());
        return Some(outputs[idx].to_string());
    }
    select_random_excluding(creature, excluded, rng)
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
    select_highest_signal_excluding(signals, &[])
}

/// Highest-signal neuron that is not already in `excluded` (issue #109).
pub fn select_highest_signal_excluding(
    signals: &HashMap<String, f64>,
    excluded: &[String],
) -> Option<FocusChoice> {
    signals
        .iter()
        .filter(|(uuid, _)| !excluded.iter().any(|e| e == *uuid))
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
    /// Weighted-random by influence on creature error (issue #25).
    ///
    /// Default. Draw ∝ error-influence mass: output residual L1, or hidden
    /// `|total adjusted-bias blame|` decayed by synapse distance to the nearest
    /// output (deep/diluted neurons rarely win the lottery). Outputs are usually
    /// strongest but not chosen every time. Zero-signal neurons are never
    /// selected. Prefer this over `high-error`, which sticks on one neuron.
    #[default]
    Weighted,
    /// Always pick the single highest-influence neuron (debug / A/B only).
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

/// Per-hop decay when converting hidden blame mass → error influence.
///
/// A neuron `d` synapses upstream of the nearest output keeps roughly
/// `FOCUS_DEPTH_DECAY^d` of its raw blame mass. Deep, diluted units therefore
/// lose the lottery to heads closer to the residual.
pub const FOCUS_DEPTH_DECAY: f64 = 0.5;

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
            } else if delta <= 0.0 {
                // Screen-empty / no positive promote Δ — treat as a soft fail so
                // weighted focus does not stick on the same sterile neuron.
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
        self.select_weighted_excluding(creature, signals, &[], rng)
    }

    /// Draw a focus neuron ∝ improvement signal, skipping `excluded` (#109).
    ///
    /// With an empty exclusion list this is exactly [`Self::select_weighted`],
    /// so a single-focus experiment keeps its rng stream unchanged.
    pub fn select_weighted_excluding(
        &self,
        creature: &CreatureExport,
        signals: &HashMap<String, f64>,
        excluded: &[String],
        rng: &mut impl Rng,
    ) -> Option<FocusChoice> {
        let mut ranked = self.rank_candidates(creature, signals);
        ranked.retain(|(uuid, _, _)| !excluded.iter().any(|e| e == uuid));
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
                reasons.push("influence".into());
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

/// Per-output residual summary over a training sample.
#[derive(Debug, Clone, Default)]
pub struct OutputErrorInfluence {
    /// Mean absolute residual (MAE).
    pub mean_abs_error: f64,
    /// Total L1 residual mass `sum |target − pred|` (influence on creature error).
    pub abs_error_mass: f64,
    /// Records contributing to the sums.
    pub record_count: u64,
}

/// Streaming accumulator behind [`collect_output_mean_abs_errors`].
///
/// Fed one already-predicted record at a time so the fused pre-focus scan can
/// share an activation with the learning pass (issue #105).
pub(crate) struct OutputErrorScan {
    outputs: Vec<(usize, String)>,
    abs_sums: Vec<f64>,
    count: u64,
}

impl OutputErrorScan {
    /// Build the accumulator for a creature's output neurons.
    pub(crate) fn new(creature: &CreatureExport) -> Self {
        let outputs: Vec<(usize, String)> = creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output")
            .enumerate()
            .map(|(i, n)| (i, n.uuid.clone()))
            .collect();
        let abs_sums = vec![0.0f64; outputs.len()];
        Self {
            outputs,
            abs_sums,
            count: 0,
        }
    }

    /// Whether the creature has any output neurons to measure.
    pub(crate) fn has_outputs(&self) -> bool {
        !self.outputs.is_empty()
    }

    /// Fold one record's predictions against its targets.
    pub(crate) fn observe(&mut self, preds: &[f32], targets: &[f32]) {
        self.count += 1;
        for (out_i, _) in &self.outputs {
            if *out_i >= targets.len() || *out_i >= preds.len() {
                continue;
            }
            let pred = f64::from(preds[*out_i]);
            let target = f64::from(targets[*out_i]);
            self.abs_sums[*out_i] += (target - pred).abs();
        }
    }

    /// Fold another chunk's accumulator into this one (issue #107).
    ///
    /// Both sides describe the same creature, so the residual sums simply add.
    pub(crate) fn merge(&mut self, other: &Self) {
        self.count += other.count;
        for (mine, theirs) in self.abs_sums.iter_mut().zip(&other.abs_sums) {
            *mine += *theirs;
        }
    }

    /// Consume the accumulator and hand back per-output residual summaries.
    pub(crate) fn finish(self) -> HashMap<String, OutputErrorInfluence> {
        let mut map = HashMap::with_capacity(self.outputs.len());
        if self.count == 0 {
            return map;
        }
        for (out_i, uuid) in self.outputs {
            let mass = self.abs_sums[out_i];
            map.insert(
                uuid,
                OutputErrorInfluence {
                    mean_abs_error: mass / self.count as f64,
                    abs_error_mass: mass,
                    record_count: self.count,
                },
            );
        }
        map
    }
}

/// Collect per-output MAE and total L1 residual mass over a training sample.
pub fn collect_output_mean_abs_errors(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    max_records: Option<u64>,
) -> Result<HashMap<String, OutputErrorInfluence>, String> {
    let mut scan = OutputErrorScan::new(creature);
    if !scan.has_outputs() {
        return Ok(HashMap::new());
    }

    let config = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = crate::analysis::open_training_scan(training_data, config)?;
    let mut count = 0u64;

    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if let Some(limit) = max_records
            && count >= limit
        {
            break;
        }
        let preds = network.activate(&record.inputs, creature.output);
        count += 1;
        scan.observe(&preds, &record.outputs);
    }

    Ok(scan.finish())
}

/// Shortest synapse-path length from each non-input neuron to any output.
///
/// Outputs are distance `0`. Unreachable neurons are omitted.
pub fn distance_to_nearest_output(creature: &CreatureExport) -> HashMap<String, usize> {
    let mut reverse: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in &creature.synapses {
        reverse
            .entry(s.to_uuid.as_str())
            .or_default()
            .push(s.from_uuid.as_str());
    }
    let mut dist: HashMap<String, usize> = HashMap::new();
    let mut queue: std::collections::VecDeque<(String, usize)> = std::collections::VecDeque::new();
    for n in &creature.neurons {
        if n.neuron_type == "output" {
            dist.insert(n.uuid.clone(), 0);
            queue.push_back((n.uuid.clone(), 0));
        }
    }
    while let Some((uuid, d)) = queue.pop_front() {
        let Some(sources) = reverse.get(uuid.as_str()) else {
            continue;
        };
        for &src in sources {
            if src.starts_with("input-") {
                continue;
            }
            let next = d + 1;
            match dist.get(src) {
                Some(&existing) if existing <= next => {}
                _ => {
                    dist.insert(src.to_string(), next);
                    queue.push_back((src.to_string(), next));
                }
            }
        }
    }
    dist
}

/// Build per-neuron error-influence signals for weighted / high-error focus.
///
/// - Outputs: total L1 residual mass (`sum |error|`). Zero ⇒ omitted.
/// - Hidden: `|total adjusted-bias blame| × FOCUS_DEPTH_DECAY^depth`, where
///   `depth` is the shortest synapse distance to an output. Mean blame alone
///   is **not** used — a deep saturated unit with a huge local mean can have
///   almost no leverage on creature error. `no_change` biases are omitted.
pub fn build_improvement_signals(
    creature: &CreatureExport,
    output_errors: &HashMap<String, OutputErrorInfluence>,
    learning: &LearningSignal,
) -> HashMap<String, f64> {
    let depth = distance_to_nearest_output(creature);
    let mut signals = HashMap::new();
    for (i, n) in creature.neurons.iter().enumerate() {
        if n.neuron_type == "input" {
            continue;
        }
        if n.neuron_type == "output" {
            let Some(err) = output_errors.get(&n.uuid) else {
                continue;
            };
            if err.abs_error_mass > FOCUS_SIGNAL_EPS {
                signals.insert(n.uuid.clone(), err.abs_error_mass);
            }
            continue;
        }
        let Some(sig) = learning.biases.get(i) else {
            continue;
        };
        if sig.count <= 0.0 || sig.no_change {
            continue;
        }
        let blame_mass = sig.total_adjusted_bias.abs();
        if !blame_mass.is_finite() || blame_mass <= FOCUS_SIGNAL_EPS {
            continue;
        }
        let hops = depth.get(&n.uuid).copied().unwrap_or(usize::MAX);
        if hops == usize::MAX {
            continue;
        }
        let decay = FOCUS_DEPTH_DECAY.powi(hops as i32);
        let influence = blame_mass * decay;
        if influence > FOCUS_SIGNAL_EPS {
            signals.insert(n.uuid.clone(), influence);
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

/// Streaming accumulator behind [`collect_focus_stats`].
///
/// Fed one already-traced record at a time so the fused post-focus scan can
/// share its activation with the incoming-source and residual passes (#105).
pub(crate) struct FocusStatsScan {
    focus_uuid: String,
    squash: Option<String>,
    incoming_count: usize,
    output_index: Option<usize>,
    relative_idx: usize,
    post_offset: usize,
    pre_offset: usize,
    pre_mean: f64,
    pre_m2: f64,
    post_mean: f64,
    post_m2: f64,
    pre_min: f64,
    pre_max: f64,
    near_zero: u64,
    saturated: u64,
    count: u64,
    err_sum: f64,
    abs_err_sum: f64,
    adj_err_sum: f64,
    deriv_sum: f64,
    err_count: u64,
}

impl FocusStatsScan {
    /// Resolve the focus neuron and size the trace offsets.
    pub(crate) fn new(
        creature: &CreatureExport,
        network: &CompiledNetwork,
        focus_uuid: &str,
    ) -> Result<Self, String> {
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

        let num_non_inputs = network.num_neurons.saturating_sub(creature.input);
        Ok(Self {
            focus_uuid: focus_uuid.to_string(),
            squash: neuron.squash.clone(),
            incoming_count,
            output_index,
            relative_idx,
            post_offset: creature.output,
            pre_offset: creature.output + num_non_inputs,
            pre_mean: 0.0,
            pre_m2: 0.0,
            post_mean: 0.0,
            post_m2: 0.0,
            pre_min: f64::INFINITY,
            pre_max: f64::NEG_INFINITY,
            near_zero: 0,
            saturated: 0,
            count: 0,
            err_sum: 0.0,
            abs_err_sum: 0.0,
            adj_err_sum: 0.0,
            deriv_sum: 0.0,
            err_count: 0,
        })
    }

    /// Fold one traced record (`activate_and_trace` output) plus its targets.
    pub(crate) fn observe(&mut self, traced: &[f32], targets: &[f32]) {
        if self.pre_offset + self.relative_idx >= traced.len()
            || self.post_offset + self.relative_idx >= traced.len()
        {
            return;
        }
        let pre = f64::from(traced[self.pre_offset + self.relative_idx]);
        let post = f64::from(traced[self.post_offset + self.relative_idx]);
        self.count += 1;
        let d1 = pre - self.pre_mean;
        self.pre_mean += d1 / self.count as f64;
        self.pre_m2 += d1 * (pre - self.pre_mean);
        let d2 = post - self.post_mean;
        self.post_mean += d2 / self.count as f64;
        self.post_m2 += d2 * (post - self.post_mean);
        self.pre_min = self.pre_min.min(pre);
        self.pre_max = self.pre_max.max(pre);
        if post.abs() < 1e-6 {
            self.near_zero += 1;
        }
        if is_saturated(self.squash.as_deref(), post) {
            self.saturated += 1;
        }
        if let Some(out_i) = self.output_index
            && out_i < targets.len()
        {
            let target = f64::from(targets[out_i]);
            let err = target - post;
            let deriv = squash_derivative(self.squash.as_deref(), post);
            self.err_sum += err;
            self.abs_err_sum += err.abs();
            self.adj_err_sum += err * deriv;
            self.deriv_sum += deriv;
            self.err_count += 1;
        }
    }

    /// Fold another chunk's accumulator into this one (issue #107).
    ///
    /// The pre/post activation moments are Welford accumulators, so they merge
    /// by Chan's parallel formula rather than by adding means; every other
    /// field is a plain count or sum. Callers merge in chunk order, which is
    /// what keeps the result independent of how the chunks were scheduled.
    pub(crate) fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            self.pre_mean = other.pre_mean;
            self.pre_m2 = other.pre_m2;
            self.post_mean = other.post_mean;
            self.post_m2 = other.post_m2;
        } else {
            let na = self.count as f64;
            let nb = other.count as f64;
            let n = na + nb;
            let pre_delta = other.pre_mean - self.pre_mean;
            self.pre_mean += pre_delta * nb / n;
            self.pre_m2 += other.pre_m2 + pre_delta * pre_delta * na * nb / n;
            let post_delta = other.post_mean - self.post_mean;
            self.post_mean += post_delta * nb / n;
            self.post_m2 += other.post_m2 + post_delta * post_delta * na * nb / n;
        }
        self.count += other.count;
        self.pre_min = self.pre_min.min(other.pre_min);
        self.pre_max = self.pre_max.max(other.pre_max);
        self.near_zero += other.near_zero;
        self.saturated += other.saturated;
        self.err_sum += other.err_sum;
        self.abs_err_sum += other.abs_err_sum;
        self.adj_err_sum += other.adj_err_sum;
        self.deriv_sum += other.deriv_sum;
        self.err_count += other.err_count;
    }

    /// Consume the accumulator and hand back the focus statistics.
    pub(crate) fn finish(self) -> FocusNeuronStats {
        let count = self.count;
        let (mean_error, mean_abs_error, mean_adjusted_error, mean_derivative) =
            if self.err_count > 0 {
                (
                    Some(self.err_sum / self.err_count as f64),
                    Some(self.abs_err_sum / self.err_count as f64),
                    Some(self.adj_err_sum / self.err_count as f64),
                    Some(self.deriv_sum / self.err_count as f64),
                )
            } else {
                (None, None, None, None)
            };

        FocusNeuronStats {
            neuron_uuid: self.focus_uuid,
            squash: self.squash,
            incoming_count: self.incoming_count,
            pre_mean: self.pre_mean,
            pre_variance: if count > 0 {
                self.pre_m2 / count as f64
            } else {
                0.0
            },
            pre_min: if count > 0 { self.pre_min } else { 0.0 },
            pre_max: if count > 0 { self.pre_max } else { 0.0 },
            post_mean: self.post_mean,
            post_variance: if count > 0 {
                self.post_m2 / count as f64
            } else {
                0.0
            },
            near_zero_fraction: if count > 0 {
                self.near_zero as f64 / count as f64
            } else {
                0.0
            },
            saturation_fraction: if count > 0 {
                self.saturated as f64 / count as f64
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
        }
    }
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
    let mut scan = FocusStatsScan::new(creature, network, focus_uuid)?;

    let config = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = crate::analysis::open_training_scan(training_data, config)?;

    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if let Some(limit) = max_records
            && scan.count >= limit
        {
            break;
        }
        let traced = network.activate_and_trace(&record.inputs, creature.output);
        scan.observe(&traced, &record.outputs);
    }

    Ok(scan.finish())
}

/// Where one incoming source's per-record activation comes from.
#[derive(Debug, Clone, Copy)]
enum SourceActivation {
    /// Raw training-record input at this index.
    Input(usize),
    /// Post-activation of the creature neuron at this position.
    Neuron(usize),
    /// Source not resolvable in the creature — contributes zero.
    Missing,
}

/// Streaming accumulator behind [`collect_incoming_source_stats`].
///
/// Fed one already-traced record at a time so the fused post-focus scan can
/// share its activation with the focus-stats and residual passes (issue #105).
pub(crate) struct IncomingSourceScan {
    out: Vec<IncomingSourceStats>,
    sources: Vec<SourceActivation>,
    needs_scan: bool,
    output_index: Option<usize>,
    relative_idx: usize,
    post_offset: usize,
    sums: Vec<f64>,
    sq: Vec<f64>,
    cross: Vec<f64>,
    err_sum: f64,
    err_sq: f64,
    count: u64,
}

impl IncomingSourceScan {
    /// Resolve the focus neuron's incoming sources, seeding input stats from
    /// `observations` where available.
    pub(crate) fn new(
        creature: &CreatureExport,
        focus_uuid: &str,
        observations: Option<&crate::observations::ObservationsStatistics>,
    ) -> Result<Self, String> {
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

        // Reuse observations for raw inputs when available.
        let out: Vec<IncomingSourceStats> = incoming
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

        // Per-record source lookup, resolved once instead of per record.
        let sources: Vec<SourceActivation> = out
            .iter()
            .map(|src| {
                if let Some(idx) = src.input_index {
                    SourceActivation::Input(idx)
                } else if let Some(pos) = creature
                    .neurons
                    .iter()
                    .position(|n| n.uuid == src.from_uuid)
                {
                    SourceActivation::Neuron(pos)
                } else {
                    SourceActivation::Missing
                }
            })
            .collect();

        let n = out.len();
        // No incoming synapses, or nothing a live scan could refine.
        let needs_scan =
            !out.is_empty() && (out.iter().any(|s| !s.is_input) || output_index.is_some());
        Ok(Self {
            out,
            sources,
            needs_scan,
            output_index,
            relative_idx,
            post_offset: creature.output,
            sums: vec![0.0f64; n],
            sq: vec![0.0f64; n],
            cross: vec![0.0f64; n],
            err_sum: 0.0,
            err_sq: 0.0,
            count: 0,
        })
    }

    /// Whether folding records in can change the result at all.
    pub(crate) fn needs_scan(&self) -> bool {
        self.needs_scan
    }

    /// Fold one traced record (`activate_and_trace` output) plus its inputs and targets.
    pub(crate) fn observe(&mut self, inputs: &[f32], targets: &[f32], traced: &[f32]) {
        if self.post_offset + self.relative_idx >= traced.len() {
            return;
        }
        let post = f64::from(traced[self.post_offset + self.relative_idx]);
        let err = if let Some(out_i) = self.output_index
            && out_i < targets.len()
        {
            f64::from(targets[out_i]) - post
        } else {
            0.0
        };
        self.count += 1;
        self.err_sum += err;
        self.err_sq += err * err;
        for (i, source) in self.sources.iter().enumerate() {
            let act = match *source {
                SourceActivation::Input(idx) => f64::from(*inputs.get(idx).unwrap_or(&0.0)),
                SourceActivation::Neuron(pos) => {
                    let idx = self.post_offset + pos;
                    if idx < traced.len() {
                        f64::from(traced[idx])
                    } else {
                        0.0
                    }
                }
                SourceActivation::Missing => 0.0,
            };
            self.sums[i] += act;
            self.sq[i] += act * act;
            self.cross[i] += act * err;
        }
    }

    /// Fold another chunk's accumulator into this one (issue #107).
    ///
    /// Every accumulated field is a plain sum; the per-source descriptions are
    /// identical on both sides because both were built from the same creature.
    pub(crate) fn merge(&mut self, other: &Self) {
        for (mine, theirs) in self.sums.iter_mut().zip(&other.sums) {
            *mine += *theirs;
        }
        for (mine, theirs) in self.sq.iter_mut().zip(&other.sq) {
            *mine += *theirs;
        }
        for (mine, theirs) in self.cross.iter_mut().zip(&other.cross) {
            *mine += *theirs;
        }
        self.err_sum += other.err_sum;
        self.err_sq += other.err_sq;
        self.count += other.count;
    }

    /// Consume the accumulator and hand back the per-source statistics.
    pub(crate) fn finish(mut self) -> Vec<IncomingSourceStats> {
        if self.count == 0 {
            return self.out;
        }
        let n_f = self.count as f64;
        let err_mean = self.err_sum / n_f;
        let err_var = (self.err_sq / n_f) - err_mean * err_mean;
        for (i, src) in self.out.iter_mut().enumerate() {
            if !src.is_input {
                let mean = self.sums[i] / n_f;
                let variance = ((self.sq[i] / n_f) - mean * mean).max(0.0);
                src.mean = mean;
                src.variance = variance;
                src.std_dev = variance.sqrt();
            }
            if self.output_index.is_some() {
                let mean = self.sums[i] / n_f;
                let var = ((self.sq[i] / n_f) - mean * mean).max(0.0);
                let cov = (self.cross[i] / n_f) - mean * err_mean;
                let denom = (var * err_var.max(0.0)).sqrt();
                src.correlation_with_error = if denom > f64::EPSILON {
                    Some((cov / denom).clamp(-1.0, 1.0))
                } else {
                    Some(0.0)
                };
            }
        }
        self.out
    }
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
    let mut scan = IncomingSourceScan::new(creature, focus_uuid, observations)?;
    if !scan.needs_scan() {
        return Ok(scan.finish());
    }

    let config = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = crate::analysis::open_training_scan(training_data, config)?;
    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if let Some(limit) = max_records
            && scan.count >= limit
        {
            break;
        }
        let traced = network.activate_and_trace(&record.inputs, creature.output);
        scan.observe(&record.inputs, &record.outputs, &traced);
    }

    Ok(scan.finish())
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
        let errs = HashMap::from([(
            "o1".into(),
            OutputErrorInfluence {
                mean_abs_error: 0.0,
                abs_error_mass: 0.0,
                record_count: 10,
            },
        )]);
        let learning = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
        let signals = build_improvement_signals(&creature, &errs, &learning);
        assert!(!signals.contains_key("o1"));
        assert!(!signals.contains_key("h1"));
    }

    #[test]
    fn error_influence_prefers_output_mass_over_deep_mean_blame() {
        // output <- h1 <- h2  (h2 is two hops from the residual)
        let deep = r#"{
          "semanticVersion": "4.0.0",
          "forwardOnly": true,
          "input": 1,
          "output": 1,
          "neurons": [
            {"type":"hidden","uuid":"h2","bias":0.0,"squash":"IDENTITY"},
            {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
            {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
          ],
          "synapses": [
            {"fromUUID":"input-0","toUUID":"h2","weight":1.0},
            {"fromUUID":"h2","toUUID":"h1","weight":1.0},
            {"fromUUID":"h1","toUUID":"o1","weight":1.0}
          ]
        }"#;
        let creature = parse_creature_json(deep).unwrap();
        let dist = distance_to_nearest_output(&creature);
        assert_eq!(dist.get("o1"), Some(&0));
        assert_eq!(dist.get("h1"), Some(&1));
        assert_eq!(dist.get("h2"), Some(&2));

        let mut learning = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
        // Huge local mean blame on deep h2, tiny total mass; modest mass on h1.
        let h2 = creature
            .neurons
            .iter()
            .position(|n| n.uuid == "h2")
            .unwrap();
        let h1 = creature
            .neurons
            .iter()
            .position(|n| n.uuid == "h1")
            .unwrap();
        learning.biases[h2].count = 10.0;
        learning.biases[h2].total_adjusted_bias = -160.0; // mean 16, mass 160
        learning.biases[h1].count = 100.0;
        learning.biases[h1].total_adjusted_bias = -50.0; // mean 0.5, mass 50

        let errs = HashMap::from([(
            "o1".into(),
            OutputErrorInfluence {
                mean_abs_error: 0.65,
                abs_error_mass: 16_250.0, // 0.65 × 25000
                record_count: 25_000,
            },
        )]);
        let signals = build_improvement_signals(&creature, &errs, &learning);
        let o = *signals.get("o1").unwrap();
        let s_h1 = *signals.get("h1").unwrap();
        let s_h2 = *signals.get("h2").unwrap();
        // Depth decay: h2 mass 160 × 0.25 = 40; h1 mass 50 × 0.5 = 25.
        assert!((s_h2 - 40.0).abs() < 1e-9, "h2 influence={s_h2}");
        assert!((s_h1 - 25.0).abs() < 1e-9, "h1 influence={s_h1}");
        assert!(o > s_h2 && o > s_h1, "output residual mass should dominate");
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

    /// Issue #109: an excluded focus is never drawn again, whatever the policy.
    #[test]
    fn exclusion_skips_already_chosen_focuses() {
        let creature = parse_creature_json(TINY).unwrap();
        let mut rng = StdRng::seed_from_u64(3);

        // Random: excluding one of two non-inputs leaves exactly the other.
        for _ in 0..8 {
            assert_eq!(
                select_random_excluding(&creature, &["h1".into()], &mut rng),
                Some("o1".to_string())
            );
        }
        assert_eq!(
            select_random_excluding(&creature, &["h1".into(), "o1".into()], &mut rng),
            None,
            "a fully excluded creature yields no focus rather than repeating one"
        );

        // Unsaturated prefers outputs, so excluding the output falls back.
        assert_eq!(
            select_unsaturated_excluding(&creature, &["o1".into()], &mut rng),
            Some("h1".to_string())
        );

        // High-error picks the next strongest signal once the top is taken.
        let signals: HashMap<String, f64> =
            [("o1".to_string(), 1.0), ("h1".to_string(), 0.5)].into();
        assert_eq!(
            select_highest_signal(&signals).map(|c| c.uuid),
            Some("o1".to_string())
        );
        assert_eq!(
            select_highest_signal_excluding(&signals, &["o1".into()]).map(|c| c.uuid),
            Some("h1".to_string())
        );
        assert!(
            select_highest_signal_excluding(&signals, &["o1".into(), "h1".into()]).is_none(),
            "no unexcluded signal means no focus"
        );

        // Weighted: with the only other neuron excluded the draw is forced.
        let selector = WeightedFocusSelector::default();
        for _ in 0..8 {
            let choice = selector
                .select_weighted_excluding(&creature, &signals, &["o1".into()], &mut rng)
                .expect("one candidate remains");
            assert_eq!(choice.uuid, "h1");
        }
        assert!(
            selector
                .select_weighted_excluding(
                    &creature,
                    &signals,
                    &["o1".into(), "h1".into()],
                    &mut rng
                )
                .is_none()
        );
    }

    /// Issue #109: excluding nothing must consume the rng exactly as the
    /// unfiltered draw did, or a K=1 run would drift from its recorded seed.
    #[test]
    fn an_empty_exclusion_draws_exactly_as_before() {
        let creature = parse_creature_json(TINY).unwrap();
        let signals: HashMap<String, f64> =
            [("o1".to_string(), 1.0), ("h1".to_string(), 0.5)].into();
        let selector = WeightedFocusSelector::default();

        let mut a = StdRng::seed_from_u64(11);
        let mut b = StdRng::seed_from_u64(11);
        for _ in 0..16 {
            let plain = selector.select_weighted(&creature, &signals, &mut a);
            let excluding = selector.select_weighted_excluding(&creature, &signals, &[], &mut b);
            assert_eq!(plain.map(|c| c.uuid), excluding.map(|c| c.uuid));
        }

        let mut a = StdRng::seed_from_u64(5);
        let mut b = StdRng::seed_from_u64(5);
        for _ in 0..16 {
            assert_eq!(
                RandomFocusSelector.select(&creature, &mut a),
                select_random_excluding(&creature, &[], &mut b)
            );
        }
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
