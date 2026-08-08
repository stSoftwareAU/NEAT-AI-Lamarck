//! Analyse-without-apply learning-signal accumulation for a focus neuron.
//!
//! This is a pragmatic GRQ-oriented path: forward-trace the incumbent, seed the
//! focus (and its incoming synapses) from output residual × squash derivative,
//! and accumulate [`LearningSignal`] entries indexed the same way candidates
//! already expect (`creature.neurons` / `creature.synapses` positions).

use crate::backprop::{BiasSignal, LearningSignal, WeightSignal};
use neat_core::{CompiledNetwork, CreatureExport, TrainingDataConfig, TrainingDataIterator};
use std::path::Path;

/// Squash derivative at a post-activation value (heuristic, production-oriented).
pub fn squash_derivative(squash: Option<&str>, post: f64) -> f64 {
    match squash {
        Some("IDENTITY") => 1.0,
        Some("HARD_TANH") | Some("CLIPPED") => {
            if post.abs() < 1.0 - 1e-6 {
                1.0
            } else {
                0.0
            }
        }
        Some("TANH") | Some("BIPOLAR_SIGMOID") => (1.0 - post * post).max(0.0),
        Some("LOGISTIC") | Some("SIGMOID") => (post * (1.0 - post)).max(0.0),
        Some("RELU") => {
            if post > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Some("LEAKY_RELU") => {
            if post > 0.0 {
                1.0
            } else {
                0.01
            }
        }
        _ => {
            if post.abs() < 0.99 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Resolve source activation for a synapse `from` uuid after `activate_and_trace`.
fn source_activation(
    creature: &CreatureExport,
    from_uuid: &str,
    inputs: &[f32],
    traced: &[f32],
) -> Option<f64> {
    if let Some(idx) = from_uuid
        .strip_prefix("input-")
        .and_then(|s| s.parse::<usize>().ok())
    {
        return inputs.get(idx).map(|v| f64::from(*v));
    }
    let neuron_pos = creature.neurons.iter().position(|n| n.uuid == from_uuid)?;
    let relative_idx = neuron_pos;
    let num_non_inputs = creature.neurons.len();
    let post_offset = creature.output;
    let idx = post_offset + relative_idx;
    if idx < traced.len() && relative_idx < num_non_inputs {
        Some(f64::from(traced[idx]))
    } else {
        None
    }
}

/// Accumulate a focus-local [`LearningSignal`] over training records.
///
/// Biases/weights are indexed by position in `creature.neurons` /
/// `creature.synapses` (matching [`crate::candidates`]).
pub fn accumulate_focus_learning(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    focus_uuid: &str,
    max_records: Option<u64>,
) -> Result<LearningSignal, String> {
    let neuron_pos = creature
        .neurons
        .iter()
        .position(|n| n.uuid == focus_uuid)
        .ok_or_else(|| format!("focus neuron {focus_uuid} not found"))?;
    let neuron = &creature.neurons[neuron_pos];
    let old_bias = neuron.bias;
    let squash = neuron.squash.as_deref();
    let output_index = if neuron.neuron_type == "output" {
        creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output")
            .position(|n| n.uuid == focus_uuid)
    } else {
        None
    };

    let relative_idx = neuron_pos;
    let compiled_ok = crate::focus::neuron_index(creature, focus_uuid).is_some();
    if !compiled_ok {
        return Err(format!("focus neuron {focus_uuid} missing compiled index"));
    }

    let mut learning = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
    let incoming: Vec<usize> = creature
        .synapses
        .iter()
        .enumerate()
        .filter(|(_, s)| s.to_uuid == focus_uuid)
        .map(|(i, _)| i)
        .collect();

    let config = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = TrainingDataIterator::new(training_data, config).map_err(|e| e.to_string())?;
    let mut count = 0u64;

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
        let post = f64::from(traced[post_offset + relative_idx]);
        count += 1;

        // Hidden neurons have no natural target — only accumulate when focus is
        // an output (or when a residual can be derived from the output head).
        let Some(out_i) = output_index else {
            continue;
        };
        if out_i >= record.outputs.len() {
            continue;
        }
        let target = f64::from(record.outputs[out_i]);
        let err = target - post;
        let deriv = squash_derivative(squash, post);
        let adjusted = err * deriv;

        {
            let signal: &mut BiasSignal = &mut learning.biases[neuron_pos];
            signal.count += 1.0;
            signal.total_adjusted_bias += old_bias + adjusted;
            if deriv <= f64::EPSILON {
                signal.no_change = true;
            }
        }

        for &syn_idx in &incoming {
            let syn = &creature.synapses[syn_idx];
            let Some(activation) =
                source_activation(creature, &syn.from_uuid, &record.inputs, &traced)
            else {
                continue;
            };
            let weight = syn.weight;
            // Propose toward weight + (error*deriv)/activation when usable.
            let adj = if activation.abs() > 1e-6 {
                weight + adjusted / activation
            } else {
                weight
            };
            let signal: &mut WeightSignal = &mut learning.weights[syn_idx];
            signal.count += 1.0;
            if activation >= 0.0 {
                signal.total_positive_activation += activation;
                signal.count_positive += 1.0;
                signal.total_positive_adjusted_value += adj;
            } else {
                signal.total_negative_activation += activation;
                signal.count_negative += 1.0;
                signal.total_negative_adjusted_value += adj;
            }
        }
    }

    Ok(learning)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::{compile_creature, parse_creature_json};
    use std::io::Write;
    use tempfile::tempdir;

    const TINY: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 1,
      "output": 1,
      "neurons": [
        {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0}
      ]
    }"#;

    #[test]
    fn identity_output_accumulates_nonzero_bias_signal() {
        let dir = tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join("0.bin")).unwrap();
        // input=1, target=1 → with weights 1, bias 0: post≈1, err≈0 for h1→o1 path
        // Use target=2 so residual is clearly positive.
        for v in [1.0f32, 2.0f32] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        let creature = parse_creature_json(TINY).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let learning =
            accumulate_focus_learning(&creature, &mut network, dir.path(), "o1", Some(1)).unwrap();
        assert!(learning.biases[1].count >= 1.0);
        assert!(learning.biases[1].total_adjusted_bias != 0.0);
    }

    #[test]
    fn hard_tanh_saturated_derivative_is_zero() {
        assert_eq!(squash_derivative(Some("HARD_TANH"), 1.0), 0.0);
        assert_eq!(squash_derivative(Some("HARD_TANH"), 0.0), 1.0);
    }
}
