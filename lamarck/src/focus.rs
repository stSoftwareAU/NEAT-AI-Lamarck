//! Focus-neuron selection and incumbent-specific streaming statistics.

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
    }

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
        record_count: count,
    })
}

fn is_saturated(squash: Option<&str>, post: f64) -> bool {
    match squash {
        Some("TANH") | Some("BIPOLAR_SIGMOID") => post.abs() > 0.99,
        Some("LOGISTIC") | Some("SIGMOID") => !(0.01..=0.99).contains(&post),
        Some("RELU") | Some("LEAKY_RELU") => post <= 0.0,
        Some("CLIPPED") => post.abs() >= 1.0,
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
