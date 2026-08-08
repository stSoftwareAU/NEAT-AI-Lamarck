//! Focus-neuron selection and incumbent-specific streaming statistics.

use crate::learning::squash_derivative;
use neat_core::{CompiledNetwork, CreatureExport, TrainingDataConfig, TrainingDataIterator};
use rand::Rng;
use serde::{Deserialize, Serialize};

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
    /// Random non-input each experiment.
    #[default]
    Random,
    /// Prefer outputs (unsaturated / direct heads).
    Unsaturated,
    /// Prefer first output neuron (error-bearing head).
    HighError,
}

impl FocusPolicy {
    /// Parse from CLI string.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "random" => Some(Self::Random),
            "unsaturated" => Some(Self::Unsaturated),
            "high-error" | "high_error" => Some(Self::HighError),
            _ => None,
        }
    }

    /// Human label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Unsaturated => "unsaturated",
            Self::HighError => "high-error",
        }
    }
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

fn is_saturated(squash: Option<&str>, post: f64) -> bool {
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
    use neat_core::parse_creature_json;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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
}
