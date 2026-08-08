//! Candidate population generation from an incumbent creature.

use crate::backprop::{BackpropConfig, BiasSignal, LearningSignal};
use crate::focus::FocusNeuronStats;
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
    /// Output-neuron bias += measured mean error (`target - post`).
    MeanErrorBias,
    /// Statistics-guided weight change.
    StatsWeight,
    /// Statistics-guided bias change.
    StatsBias,
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
        CandidateStrategy::Random,
    ];

    // Always try mean-error bias first when the focus is an output with a
    // measured residual — this is the classic "nudge bias by average error".
    let mut start = 0usize;
    if ctx.focus_stats.mean_error.is_some()
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
    let observations = ctx.observations;
    let learning = ctx.learning;
    let backprop = ctx.backprop;

    let mut creature = incumbent.clone();
    let neuron_pos = creature.neurons.iter().position(|n| n.uuid == focus_uuid)?;
    let old_bias = creature.neurons[neuron_pos].bias;

    let (mutation, old_value, new_value) = match strategy {
        CandidateStrategy::MeanErrorBias => {
            let mean_err = focus_stats.mean_error?;
            let new_bias =
                (old_bias + mean_err).clamp(-backprop.limit_bias_scale, backprop.limit_bias_scale);
            creature.neurons[neuron_pos].bias = new_bias;
            (
                format!("mean-error bias Δ {mean_err} ({old_bias} -> {new_bias})"),
                Some(old_bias),
                Some(new_bias),
            )
        }
        CandidateStrategy::Backprop => {
            let lr = backprop.learning_rate;
            // Prefer a real accumulated learning signal. Without one, fall back
            // to mean-error (outputs) or a tiny pre-activation nudge (hidden).
            let fallback = if let Some(mean_err) = focus_stats.mean_error {
                BiasSignal {
                    count: 1.0,
                    total_adjusted_bias: old_bias + mean_err,
                    no_change: false,
                }
            } else {
                BiasSignal {
                    count: 1.0,
                    total_adjusted_bias: old_bias + focus_stats.pre_mean * 0.01,
                    no_change: false,
                }
            };
            let signal = learning
                .and_then(|l| l.biases.get(neuron_pos))
                .cloned()
                .unwrap_or(fallback);
            let new_bias = signal.propose(old_bias, backprop, lr);
            creature.neurons[neuron_pos].bias = new_bias;
            (
                format!("backprop bias {old_bias} -> {new_bias}"),
                Some(old_bias),
                Some(new_bias),
            )
        }
        CandidateStrategy::StatsBias => {
            let scale = if focus_stats.pre_variance > 0.0 {
                focus_stats.pre_variance.sqrt()
            } else {
                0.01
            };
            let delta = scale * rng.random_range(-0.05..0.05);
            let new_bias =
                (old_bias + delta).clamp(-backprop.limit_bias_scale, backprop.limit_bias_scale);
            creature.neurons[neuron_pos].bias = new_bias;
            (
                format!("stats bias delta {delta}"),
                Some(old_bias),
                Some(new_bias),
            )
        }
        CandidateStrategy::StatsWeight => {
            let syn_idx = creature
                .synapses
                .iter()
                .position(|s| s.to_uuid == focus_uuid)?;
            let old_w = creature.synapses[syn_idx].weight;
            let source_scale = observations
                .inputs
                .first()
                .map(|s| s.std_dev.max(1e-3))
                .unwrap_or(1.0);
            let delta = (0.01 / source_scale) * rng.random_range(-1.0..1.0);
            let new_w =
                (old_w + delta).clamp(-backprop.limit_weight_scale, backprop.limit_weight_scale);
            creature.synapses[syn_idx].weight = new_w;
            (
                format!("stats weight delta {delta}"),
                Some(old_w),
                Some(new_w),
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
                let syn_idx = creature
                    .synapses
                    .iter()
                    .position(|s| s.to_uuid == focus_uuid)?;
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

    #[test]
    fn generation_is_deterministic_and_non_mutating() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let original = incumbent.clone();
        let focus = FocusNeuronStats {
            neuron_uuid: "h1".into(),
            pre_variance: 0.04,
            ..FocusNeuronStats::default()
        };
        let observations = ObservationsStatistics {
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
        };
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "h1",
            focus_stats: &focus,
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
            ..FocusNeuronStats::default()
        };
        let observations = ObservationsStatistics {
            format_version: "1.0.0".into(),
            algorithm_version: "1.0.0".into(),
            mode: crate::observations::StatsMode::Full,
            sample_record_limit: None,
            input_count: 1,
            output_count: 1,
            record_count: 0,
            corpus_identity: "x".into(),
            created_at_unix: 0,
            inputs: vec![],
            outputs: vec![],
            input_correlations: vec![],
            input_target_correlations: vec![],
        };
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
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
}
