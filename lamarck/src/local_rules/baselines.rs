//! Baseline update arms: Lamarck's current sparse backprop refinement, and a
//! derivative-free search built on the crate's mirrored signed-perturbation
//! helpers (`crate::mirror`, issue #203). Both clone the parent and charge cost in
//! the same record units as the symbolic arm.

use std::f64::consts::TAU;

use neat_core::CreatureExport;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::apply::{Cost, compile, mse};
use super::tasks::TaskRecord;
use crate::backprop::{BackpropConfig, apply_learnings, calculate_learning_rate};
use crate::candidates::{Candidate, CandidateProvenance, CandidateStrategy};
use crate::mirror::{PerturbedScalar, mirror_candidate, signed_perturbation};
use crate::propagate_layout::LearningPlan;

/// Mixed into the seed so the sparse selection stream differs from other streams.
const PLAN_STREAM: u64 = 0x9E37_79B9_7F4A_7C15;

/// Lamarck's current refinement, run as the issue-267 `refine` loop runs it:
/// every training record in every step, one `LearningPlan` per step.
///
/// Each step costs one forward pass plus one backprop update per record (2 units).
pub fn sparse_backprop(
    parent: &CreatureExport,
    records: &[TaskRecord],
    steps: u32,
    learning_rate: f64,
    sparse_ratio: f64,
    seed: u64,
) -> Result<(CreatureExport, Cost), String> {
    let config = BackpropConfig {
        learning_rate,
        initial_learning_rate: learning_rate,
        sparse_ratio,
        ..BackpropConfig::default()
    };
    let mut creature = parent.clone();
    let mut rng = StdRng::seed_from_u64(seed ^ PLAN_STREAM);
    for step in 0..steps {
        let plan = LearningPlan::new(&creature, &config, &mut rng)?;
        let mut network = compile(&creature)?;
        let mut scan = plan.scan();
        for record in records {
            network.activate_and_trace(&record.inputs, creature.output);
            scan.observe(&network, &record.expected);
        }
        let signal = scan.finish();
        let rate = calculate_learning_rate(&config, u64::from(step), None);
        creature = apply_learnings(&creature, &signal, &config, rate);
    }
    let units = 2 * u64::from(steps) * records.len() as u64;
    Ok((
        creature,
        Cost {
            record_units: units,
        },
    ))
}

/// Standard normal draw by Box–Muller from `rng`.
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(f64::EPSILON..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

/// Every synapse weight, then every hidden/output neuron bias.
fn eligible_scalars(creature: &CreatureExport) -> Vec<PerturbedScalar> {
    let weights = (0..creature.synapses.len()).map(|synapse| PerturbedScalar::Weight { synapse });
    let biases = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n.neuron_type.as_str(), "hidden" | "output"))
        .map(|(neuron, _)| PerturbedScalar::Bias { neuron });
    weights.chain(biases).collect()
}

/// The `+δ` candidate: `best` with one scalar stepped by `delta`.
fn step_one(best: &CreatureExport, scalar: PerturbedScalar, delta: f64) -> Candidate {
    let mut creature = best.clone();
    let (focus, old, new, what) = match scalar {
        PerturbedScalar::Weight { synapse } => {
            let synapse = &mut creature.synapses[synapse];
            let old = synapse.weight;
            synapse.weight += delta;
            (synapse.to_uuid.clone(), old, synapse.weight, "weight")
        }
        PerturbedScalar::Bias { neuron } => {
            let neuron = &mut creature.neurons[neuron];
            let old = neuron.bias;
            neuron.bias += delta;
            (neuron.uuid.clone(), old, neuron.bias, "bias")
        }
    };
    Candidate {
        creature,
        provenance: CandidateProvenance {
            strategy: CandidateStrategy::Random,
            focus_neuron: focus,
            mutation: format!("local-rules derivative-free {what} delta {delta}"),
            old_value: Some(old),
            new_value: Some(new),
            mirror: None,
            follow_up: None,
            neighbourhood: None,
        },
    }
}

/// Derivative-free search built on Lamarck's mirror helpers (`crate::mirror`).
///
/// Each round steps one scalar of the incumbent (a synapse weight or a
/// hidden/output bias, drawn uniformly) by `sigma` times a standard normal,
/// confirms the step with [`crate::mirror::signed_perturbation`], and scores it
/// beside its `−δ` twin from [`crate::mirror::mirror_candidate`] — the same
/// antithetic pair the candidate generator emits (issue #203). The lower-MSE
/// half replaces the incumbent only if its training MSE is strictly lower, so
/// the returned creature never has a higher training MSE than the parent.
///
/// The parent's own evaluation counts as the first of `evaluations`. When one
/// evaluation is left, or the twin is refused (a clamped or sub-plank `−δ`),
/// only the `+δ` half is scored. Each evaluation costs one forward pass per
/// record (1 unit).
pub fn derivative_free(
    parent: &CreatureExport,
    records: &[TaskRecord],
    evaluations: u64,
    sigma: f64,
    seed: u64,
) -> Result<(CreatureExport, Cost), String> {
    let scalars = eligible_scalars(parent);
    if scalars.is_empty() {
        return Err("derivative-free search found no weight or bias to step".to_string());
    }
    let limits = BackpropConfig::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best = parent.clone();
    let mut best_mse = mse(&best, records)?;
    let mut done = 1u64;
    while done < evaluations {
        let scalar = scalars[rng.random_range(0..scalars.len())];
        let original = step_one(&best, scalar, sigma * gaussian(&mut rng));
        let twin = if evaluations - done >= 2 {
            signed_perturbation(&best, &original.creature)
                .and_then(|p| mirror_candidate(&original, &p, &limits))
        } else {
            None
        };
        for half in std::iter::once(original).chain(twin) {
            let half_mse = mse(&half.creature, records)?;
            done += 1;
            if half_mse < best_mse {
                best = half.creature;
                best_mse = half_mse;
            }
        }
    }
    Ok((
        best,
        Cost {
            record_units: done * records.len() as u64,
        },
    ))
}

/// Whole steps (or evaluations) that fit in `budget_units`, at
/// `units_per_record` per record across `records` records.
pub fn matched_count(budget_units: u64, units_per_record: u64, records: usize) -> u64 {
    let per_step = units_per_record * records as u64;
    budget_units.checked_div(per_step).unwrap_or(0)
}
