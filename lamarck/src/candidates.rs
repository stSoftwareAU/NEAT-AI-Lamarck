//! Candidate population generation from an incumbent creature.

use crate::backprop::{BackpropConfig, BiasSignal, LearningSignal};
use crate::focus::{FocusNeuronStats, IncomingSourceStats};
use crate::observations::ObservationsStatistics;
use neat_core::{CreatureExport, creature_to_json_pretty};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Strategy label recorded in the experiment journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStrategy {
    /// Conventional backprop-derived bias/weight change.
    Backprop,
    /// Output-neuron bias += squash-aware mean adjusted error.
    MeanErrorBias,
    /// Statistics-guided weight change.
    StatsWeight,
    /// Statistics-guided bias change.
    StatsBias,
    /// Add a plausible upstream connection into the focus.
    StructuralAdd,
    /// Weaken an apparently weak/useless incoming connection.
    StructuralWeaken,
    /// Random exploratory mutation.
    Random,
}

/// Provenance for one candidate creature.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateProvenance {
    /// Strategy that produced the candidate.
    pub strategy: CandidateStrategy,
    /// Focus neuron UUID.
    pub focus_neuron: String,
    /// Human-readable mutation description.
    pub mutation: String,
    /// Old value when a scalar field changed.
    pub old_value: Option<f64>,
    /// New value when a scalar field changed.
    pub new_value: Option<f64>,
}

/// One generated candidate plus provenance.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Candidate creature JSON export.
    pub creature: CreatureExport,
    /// Provenance metadata.
    pub provenance: CandidateProvenance,
}

/// Inputs shared by candidate generation for one focus-neuron experiment.
pub struct CandidateGenContext<'a> {
    /// Incumbent creature (never mutated).
    pub incumbent: &'a CreatureExport,
    /// Focus neuron UUID.
    pub focus_uuid: &'a str,
    /// Focused neuron statistics.
    pub focus_stats: &'a FocusNeuronStats,
    /// Incoming source statistics for the focus.
    pub incoming: &'a [IncomingSourceStats],
    /// Dataset statistics.
    pub observations: &'a ObservationsStatistics,
    /// Optional accumulated learning signal.
    pub learning: Option<&'a LearningSignal>,
    /// Backprop configuration.
    pub backprop: &'a BackpropConfig,
}

/// Generate a candidate population without mutating the incumbent.
pub fn generate_candidates(
    ctx: &CandidateGenContext<'_>,
    count: usize,
    rng: &mut impl Rng,
) -> Vec<Candidate> {
    let mut out = Vec::with_capacity(count);
    if count == 0 {
        return out;
    }

    let strategies = [
        CandidateStrategy::Backprop,
        CandidateStrategy::StatsWeight,
        CandidateStrategy::StatsBias,
        CandidateStrategy::StructuralAdd,
        CandidateStrategy::StructuralWeaken,
        CandidateStrategy::Random,
    ];

    let mut start = 0usize;
    let has_error =
        ctx.focus_stats.mean_adjusted_error.is_some() || ctx.focus_stats.mean_error.is_some();
    if has_error
        && let Some(candidate) = build_candidate(ctx, CandidateStrategy::MeanErrorBias, rng)
    {
        out.push(candidate);
        start = 1;
    }

    for i in start..count {
        let strategy = strategies[(i - start) % strategies.len()];
        if let Some(candidate) = build_candidate(ctx, strategy, rng) {
            out.push(candidate);
        }
    }
    out
}

fn build_candidate(
    ctx: &CandidateGenContext<'_>,
    strategy: CandidateStrategy,
    rng: &mut impl Rng,
) -> Option<Candidate> {
    let incumbent = ctx.incumbent;
    let focus_uuid = ctx.focus_uuid;
    let focus_stats = ctx.focus_stats;
    let learning = ctx.learning;
    let backprop = ctx.backprop;

    let mut creature = incumbent.clone();
    let neuron_pos = creature.neurons.iter().position(|n| n.uuid == focus_uuid)?;
    let old_bias = creature.neurons[neuron_pos].bias;

    let (mutation, old_value, new_value) = match strategy {
        CandidateStrategy::MeanErrorBias => {
            let mean_adj = focus_stats.mean_adjusted_error.or(focus_stats.mean_error)?;
            let mean_deriv = focus_stats.mean_derivative.unwrap_or(1.0);
            if mean_deriv <= 1e-6 {
                return None; // saturated — skip blunt bias nudge
            }
            let new_bias =
                (old_bias + mean_adj).clamp(-backprop.limit_bias_scale, backprop.limit_bias_scale);
            creature.neurons[neuron_pos].bias = new_bias;
            (
                format!(
                    "mean-error bias Δ {mean_adj} (deriv={mean_deriv:.4}, {old_bias} -> {new_bias})"
                ),
                Some(old_bias),
                Some(new_bias),
            )
        }
        CandidateStrategy::Backprop => {
            let lr = backprop.learning_rate;
            if let Some(signal) = learning.and_then(|l| l.biases.get(neuron_pos))
                && signal.count > 0.0
            {
                let new_bias = signal.propose(old_bias, backprop, lr);
                // Also try a weight propose on the strongest incoming synapse.
                if let Some(src) = pick_best_incoming(ctx.incoming, rng)
                    && let Some(w_signal) = learning.and_then(|l| l.weights.get(src.synapse_index))
                    && w_signal.count > 0.0
                    && rng.random_bool(0.5)
                {
                    let old_w = creature.synapses[src.synapse_index].weight;
                    let new_w = w_signal.propose(old_w, backprop, lr);
                    creature.synapses[src.synapse_index].weight = new_w;
                    return Some(Candidate {
                        creature,
                        provenance: CandidateProvenance {
                            strategy,
                            focus_neuron: focus_uuid.to_string(),
                            mutation: format!(
                                "backprop weight {} {old_w} -> {new_w} (count={})",
                                src.from_uuid, w_signal.count
                            ),
                            old_value: Some(old_w),
                            new_value: Some(new_w),
                        },
                    });
                }
                creature.neurons[neuron_pos].bias = new_bias;
                (
                    format!(
                        "backprop bias {old_bias} -> {new_bias} (count={})",
                        signal.count
                    ),
                    Some(old_bias),
                    Some(new_bias),
                )
            } else if let Some(mean_adj) =
                focus_stats.mean_adjusted_error.or(focus_stats.mean_error)
            {
                let fallback = BiasSignal {
                    count: 1.0,
                    total_adjusted_bias: old_bias + mean_adj,
                    no_change: focus_stats.mean_derivative.is_some_and(|d| d <= 1e-6),
                };
                let new_bias = fallback.propose(old_bias, backprop, lr);
                creature.neurons[neuron_pos].bias = new_bias;
                (
                    format!("backprop-fallback bias {old_bias} -> {new_bias}"),
                    Some(old_bias),
                    Some(new_bias),
                )
            } else {
                return None;
            }
        }
        CandidateStrategy::StatsBias => {
            let scale = if focus_stats.pre_variance > 0.0 {
                focus_stats.pre_variance.sqrt()
            } else {
                0.01
            };
            // Prefer moving away from saturation when saturated.
            let direction = if focus_stats.saturation_fraction > 0.5 {
                if focus_stats.post_mean >= 0.0 {
                    -1.0
                } else {
                    1.0
                }
            } else {
                rng.random_range(-1.0..1.0)
            };
            let delta = scale * 0.05 * direction;
            let new_bias =
                (old_bias + delta).clamp(-backprop.limit_bias_scale, backprop.limit_bias_scale);
            creature.neurons[neuron_pos].bias = new_bias;
            (
                format!(
                    "stats bias Δ {delta} (pre_std={scale:.4}, sat={:.3})",
                    focus_stats.saturation_fraction
                ),
                Some(old_bias),
                Some(new_bias),
            )
        }
        CandidateStrategy::StatsWeight => {
            let src = pick_best_incoming(ctx.incoming, rng)?;
            let syn_idx = src.synapse_index;
            let old_w = creature.synapses[syn_idx].weight;
            let source_scale = src.std_dev.max(1e-3);
            let preferred = src.correlation_with_error.unwrap_or(0.0);
            let direction = if preferred.abs() > 0.05 {
                preferred.signum()
            } else {
                rng.random_range(-1.0..1.0)
            };
            let delta = (0.01 / source_scale) * direction * rng.random_range(0.25..1.0);
            let new_w =
                (old_w + delta).clamp(-backprop.limit_weight_scale, backprop.limit_weight_scale);
            creature.synapses[syn_idx].weight = new_w;
            (
                format!(
                    "stats weight {} Δ {delta} (std={source_scale:.4}, corr={:?})",
                    src.from_uuid, src.correlation_with_error
                ),
                Some(old_w),
                Some(new_w),
            )
        }
        CandidateStrategy::StructuralWeaken => {
            let src = ctx
                .incoming
                .iter()
                .min_by(|a, b| {
                    a.weight
                        .abs()
                        .partial_cmp(&b.weight.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .or_else(|| pick_best_incoming(ctx.incoming, rng))?;
            let syn_idx = src.synapse_index;
            let old_w = creature.synapses[syn_idx].weight;
            let new_w = old_w * 0.5;
            if (new_w - old_w).abs() < backprop.plank_constant {
                return None;
            }
            creature.synapses[syn_idx].weight = new_w;
            (
                format!("structural weaken {} {old_w} -> {new_w}", src.from_uuid),
                Some(old_w),
                Some(new_w),
            )
        }
        CandidateStrategy::StructuralAdd => {
            // Pick a raw input not already connected to the focus.
            let existing: std::collections::BTreeSet<&str> = creature
                .synapses
                .iter()
                .filter(|s| s.to_uuid == focus_uuid)
                .map(|s| s.from_uuid.as_str())
                .collect();
            let mut candidates: Vec<String> = (0..creature.input)
                .map(|i| format!("input-{i}"))
                .filter(|u| !existing.contains(u.as_str()))
                .collect();
            if candidates.is_empty() {
                // Fall back to a hidden source not already connected.
                for n in &creature.neurons {
                    if n.uuid != focus_uuid && !existing.contains(n.uuid.as_str()) {
                        candidates.push(n.uuid.clone());
                    }
                }
            }
            if candidates.is_empty() {
                return None;
            }
            let from = candidates[rng.random_range(0..candidates.len())].clone();
            let weight = rng.random_range(-0.1..0.1);
            // Prefer observation scale when source is an input.
            let weight = if let Some(idx) = from
                .strip_prefix("input-")
                .and_then(|s| s.parse::<usize>().ok())
            {
                let scale = ctx
                    .observations
                    .inputs
                    .get(idx)
                    .map(|s| 0.01 / s.std_dev.max(1e-3))
                    .unwrap_or(0.01);
                scale * rng.random_range(-1.0..1.0)
            } else {
                weight
            };
            creature.synapses.push(neat_core::SynapseExport {
                from_uuid: from.clone(),
                to_uuid: focus_uuid.to_string(),
                weight,
                synapse_type: None,
            });
            (
                format!("structural add {from} -> {focus_uuid} w={weight}"),
                None,
                Some(weight),
            )
        }
        CandidateStrategy::Random => {
            if rng.random_bool(0.5) {
                let delta = rng.random_range(-0.05..0.05);
                let new_bias = old_bias + delta;
                creature.neurons[neuron_pos].bias = new_bias;
                (
                    format!("random bias delta {delta}"),
                    Some(old_bias),
                    Some(new_bias),
                )
            } else {
                let src = pick_best_incoming(ctx.incoming, rng)?;
                let syn_idx = src.synapse_index;
                let old_w = creature.synapses[syn_idx].weight;
                let delta = rng.random_range(-0.05..0.05);
                let new_w = old_w + delta;
                creature.synapses[syn_idx].weight = new_w;
                (
                    format!("random weight delta {delta}"),
                    Some(old_w),
                    Some(new_w),
                )
            }
        }
    };

    Some(Candidate {
        creature,
        provenance: CandidateProvenance {
            strategy,
            focus_neuron: focus_uuid.to_string(),
            mutation,
            old_value,
            new_value,
        },
    })
}

fn pick_best_incoming<'a>(
    incoming: &'a [IncomingSourceStats],
    rng: &mut impl Rng,
) -> Option<&'a IncomingSourceStats> {
    if incoming.is_empty() {
        return None;
    }
    // Prefer highest |correlation_with_error|, else random.
    if let Some(best) = incoming.iter().max_by(|a, b| {
        let aa = a.correlation_with_error.unwrap_or(0.0).abs();
        let bb = b.correlation_with_error.unwrap_or(0.0).abs();
        aa.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
    }) && best.correlation_with_error.unwrap_or(0.0).abs() > 0.05
    {
        return Some(best);
    }
    Some(&incoming[rng.random_range(0..incoming.len())])
}

/// Write baseline + candidates into a temporary scoring directory.
pub fn write_candidate_batch(
    dir: &Path,
    incumbent: &CreatureExport,
    candidates: &[Candidate],
) -> Result<Vec<String>, String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let baseline = creature_to_json_pretty(incumbent).map_err(|e| e.to_string())?;
    fs::write(dir.join("baseline.json"), baseline).map_err(|e| e.to_string())?;

    let mut stems = vec!["baseline".to_string()];
    for (i, candidate) in candidates.iter().enumerate() {
        let stem = format!("candidate-{i:03}");
        let json = creature_to_json_pretty(&candidate.creature).map_err(|e| e.to_string())?;
        fs::write(dir.join(format!("{stem}.json")), json).map_err(|e| e.to_string())?;
        stems.push(stem);
    }
    Ok(stems)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;
    use rand::{SeedableRng, rngs::StdRng};

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

    fn empty_obs() -> ObservationsStatistics {
        ObservationsStatistics {
            format_version: "1.0.0".into(),
            algorithm_version: "1.0.0".into(),
            mode: crate::observations::StatsMode::Full,
            sample_record_limit: None,
            input_count: 1,
            output_count: 1,
            record_count: 0,
            corpus_identity: "x".into(),
            created_at_unix: 0,
            inputs: vec![crate::observations::ScalarStats {
                count: 1,
                mean: 0.0,
                variance: 1.0,
                std_dev: 1.0,
                min: 0.0,
                max: 0.0,
                zero_count: 0,
                non_zero_count: 0,
                non_finite_count: 0,
                mean_abs: 0.0,
                rms: 0.0,
                quantiles: [0.0; 7],
            }],
            outputs: vec![],
            input_correlations: vec![],
            input_target_correlations: vec![],
        }
    }

    #[test]
    fn generation_is_deterministic_and_non_mutating() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let original = incumbent.clone();
        let focus = FocusNeuronStats {
            neuron_uuid: "h1".into(),
            pre_variance: 0.04,
            ..FocusNeuronStats::default()
        };
        let incoming = vec![IncomingSourceStats {
            synapse_index: 0,
            from_uuid: "input-0".into(),
            weight: 1.0,
            is_input: true,
            input_index: Some(0),
            mean: 0.0,
            variance: 1.0,
            std_dev: 1.0,
            correlation_with_error: None,
        }];
        let observations = empty_obs();
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "h1",
            focus_stats: &focus,
            incoming: &incoming,
            observations: &observations,
            learning: None,
            backprop: &cfg,
        };
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        let a = generate_candidates(&ctx, 8, &mut rng_a);
        let b = generate_candidates(&ctx, 8, &mut rng_b);
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].provenance.mutation, b[0].provenance.mutation);
        assert_eq!(incumbent, original);
        assert!(incumbent.forward_only);
    }

    #[test]
    fn mean_error_bias_nudge_is_first_when_error_known() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_error: Some(0.25),
            mean_abs_error: Some(0.25),
            mean_adjusted_error: Some(0.25),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        };
        let incoming = [IncomingSourceStats {
            synapse_index: 1,
            from_uuid: "h1".into(),
            weight: 1.0,
            is_input: false,
            input_index: None,
            mean: 0.0,
            variance: 1.0,
            std_dev: 1.0,
            correlation_with_error: Some(0.5),
        }];
        let observations = empty_obs();
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
            incoming: &incoming,
            observations: &observations,
            learning: None,
            backprop: &cfg,
        };
        let mut rng = StdRng::seed_from_u64(1);
        let candidates = generate_candidates(&ctx, 4, &mut rng);
        assert_eq!(
            candidates[0].provenance.strategy,
            CandidateStrategy::MeanErrorBias
        );
        let out_bias = candidates[0]
            .creature
            .neurons
            .iter()
            .find(|n| n.uuid == "o1")
            .unwrap()
            .bias;
        assert!((out_bias - 0.25).abs() < 1e-12);
    }

    #[test]
    fn saturated_mean_error_bias_is_skipped() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_error: Some(0.5),
            mean_adjusted_error: Some(0.0),
            mean_derivative: Some(0.0),
            saturation_fraction: 1.0,
            ..FocusNeuronStats::default()
        };
        let observations = empty_obs();
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
            incoming: &[],
            observations: &observations,
            learning: None,
            backprop: &cfg,
        };
        let mut rng = StdRng::seed_from_u64(1);
        assert!(build_candidate(&ctx, CandidateStrategy::MeanErrorBias, &mut rng).is_none());
    }
}
