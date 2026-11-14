//! Baseline update arms: Lamarck's current sparse backprop refinement, and a
//! derivative-free (1+1) random search. Both clone the parent and charge cost
//! in the same record units as the symbolic arm.

use std::f64::consts::TAU;

use neat_core::CreatureExport;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::apply::{Cost, compile, mse};
use super::tasks::TaskRecord;
use crate::backprop::{BackpropConfig, apply_learnings, calculate_learning_rate};
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

/// Derivative-free (1+1) random search over every weight and hidden/output bias.
///
/// The parent's own evaluation counts as the first of `evaluations`. A candidate
/// is accepted only if its training MSE strictly decreases, so the returned
/// creature never has a higher training MSE than the parent. Each evaluation
/// costs one forward pass per record (1 unit).
pub fn derivative_free(
    parent: &CreatureExport,
    records: &[TaskRecord],
    evaluations: u64,
    sigma: f64,
    seed: u64,
) -> Result<(CreatureExport, Cost), String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best = parent.clone();
    let mut best_mse = mse(&best, records)?;
    let mut done = 1u64;
    while done < evaluations {
        let mut candidate = best.clone();
        for synapse in &mut candidate.synapses {
            synapse.weight += sigma * gaussian(&mut rng);
        }
        for neuron in &mut candidate.neurons {
            if matches!(neuron.neuron_type.as_str(), "hidden" | "output") {
                neuron.bias += sigma * gaussian(&mut rng);
            }
        }
        let candidate_mse = mse(&candidate, records)?;
        done += 1;
        if candidate_mse < best_mse {
            best = candidate;
            best_mse = candidate_mse;
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
