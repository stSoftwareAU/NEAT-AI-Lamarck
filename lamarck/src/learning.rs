//! Analyse-without-apply learning-signal accumulation for a focus neuron.
//!
//! Issue #2: accumulation goes through neat-core
//! [`propagate_topological_loop`](neat_core::propagate_topological_loop) via
//! [`crate::propagate_layout`]. The focus UUID only affects which signals
//! candidates consume; hidden neurons receive real propagated blame.

use crate::backprop::{BackpropConfig, LearningSignal};
use crate::propagate_layout::accumulate_creature_learning;
use neat_core::{CompiledNetwork, CreatureExport};
use rand::Rng;
use std::path::Path;

/// Squash derivative at a post-activation value (heuristic, used by focus stats).
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

/// Accumulate a creature-wide [`LearningSignal`] (analyse-without-apply).
///
/// `focus_uuid` is retained for API compatibility with the run loop; the
/// reverse-topo pass trains the full sparse selection (default: all
/// hidden/output neurons). Candidates still propose only on the focus.
pub fn accumulate_focus_learning(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    focus_uuid: &str,
    max_records: Option<u64>,
    config: &BackpropConfig,
    rng: &mut impl Rng,
) -> Result<LearningSignal, String> {
    if creature.neurons.iter().all(|n| n.uuid != focus_uuid) {
        return Err(format!("focus neuron {focus_uuid} not found"));
    }
    accumulate_creature_learning(creature, network, training_data, config, max_records, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
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
        for v in [1.0f32, 2.0f32] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        let creature = parse_creature_json(TINY).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let cfg = BackpropConfig::default();
        let mut rng = StdRng::seed_from_u64(1);
        let learning = accumulate_focus_learning(
            &creature,
            &mut network,
            dir.path(),
            "o1",
            Some(1),
            &cfg,
            &mut rng,
        )
        .unwrap();
        assert!(learning.biases[1].count >= 1.0);
        assert!(learning.biases[1].total_adjusted_bias != 0.0);
    }

    #[test]
    fn hard_tanh_saturated_derivative_is_zero() {
        assert_eq!(squash_derivative(Some("HARD_TANH"), 1.0), 0.0);
        assert_eq!(squash_derivative(Some("HARD_TANH"), 0.0), 1.0);
    }
}
