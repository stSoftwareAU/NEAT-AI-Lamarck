//! Meta-search: discovers one sparse rule per activation family.
//!
//! Calibration only. Validation tasks are never passed to this module. A
//! candidate is scored on TRAINING records alone, and every evaluation is
//! charged to the meta cost.

use std::collections::BTreeSet;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::Serialize;

use super::apply::{Cost, apply_rule_steps, mse};
use super::manifest::Manifest;
use super::rule::{ActivationFamily, Feature, RuleSet, SparseRule};
use super::tasks::Task;

/// One candidate evaluated during discovery.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchStep {
    /// Discovery round, from zero.
    pub round: u32,
    /// Family whose rule the candidate replaces.
    pub family: ActivationFamily,
    /// The candidate rule, in readable form.
    pub candidate: String,
    /// Mean ratio of after-to-parent training MSE over calibration tasks.
    pub score: f64,
    /// Whether the candidate lowered the score and was kept.
    pub accepted: bool,
}

/// Activation families of the hidden and output neurons across `tasks`, sorted.
fn calibration_families(tasks: &[Task]) -> Vec<ActivationFamily> {
    let mut families = BTreeSet::new();
    for task in tasks {
        for neuron in &task.parent.neurons {
            if matches!(neuron.neuron_type.as_str(), "hidden" | "output") {
                families.insert(ActivationFamily::from_squash(neuron.squash.as_deref()));
            }
        }
    }
    families.into_iter().collect()
}

/// Mean over `tasks` of training MSE after the rules, divided by the parent's.
///
/// Charges the rule steps plus the two MSE forward passes per task.
fn score(tasks: &[Task], rules: &RuleSet, manifest: &Manifest) -> Result<(f64, Cost), String> {
    let mut total = 0.0;
    let mut cost = Cost::default();
    for task in tasks {
        let (after, steps_cost) = apply_rule_steps(
            &task.parent,
            &task.train,
            rules,
            manifest.calibration_steps,
            manifest.eta,
            manifest.max_step,
        )?;
        let before = mse(&task.parent, &task.train)?;
        let now = mse(&after, &task.train)?;
        let forward = Cost {
            record_units: 2 * task.train.len() as u64,
        };
        cost = cost + steps_cost + forward;
        total += now / before;
    }
    Ok((total / tasks.len() as f64, cost))
}

/// Draw one sparse rule: support uniform in `1..=max_support`, distinct features
/// from the library, coefficients from the grid.
fn sample_rule(rng: &mut StdRng, manifest: &Manifest) -> Result<SparseRule, String> {
    let max_support = manifest.max_support.min(Feature::ALL.len());
    let support = rng.random_range(1..=max_support);
    let mut terms: Vec<(Feature, f64)> = Vec::new();
    while terms.len() < support {
        let feature = Feature::ALL[rng.random_range(0..Feature::ALL.len())];
        if terms.iter().any(|(f, _)| *f == feature) {
            continue;
        }
        let grid = &manifest.coefficient_grid;
        terms.push((feature, grid[rng.random_range(0..grid.len())]));
    }
    SparseRule::new(terms, max_support)
}

/// Discover a rule set on `calibration` tasks.
///
/// Starts from an empty rule set (score 1.0, the parent itself). For each round,
/// each family present in the calibration neurons, and each of
/// `candidates_per_round` samples, a candidate replaces that family's rule and is
/// kept only if the score strictly drops.
pub fn discover(
    manifest: &Manifest,
    calibration: &[Task],
) -> Result<(RuleSet, Cost, Vec<SearchStep>), String> {
    if calibration.is_empty() {
        return Err("discover needs at least one calibration task".to_string());
    }
    let families = calibration_families(calibration);
    let mut rng = StdRng::seed_from_u64(manifest.meta_seed);
    let mut rules = RuleSet::new();
    let mut best_score = 1.0;
    let mut meta_cost = Cost::default();
    let mut trace = Vec::new();
    for round in 0..manifest.meta_rounds {
        for family in &families {
            for _ in 0..manifest.candidates_per_round {
                let rule = sample_rule(&mut rng, manifest)?;
                let mut candidate = rules.clone();
                candidate.insert(*family, rule.clone());
                let (candidate_score, cost) = score(calibration, &candidate, manifest)?;
                meta_cost = meta_cost + cost;
                let accepted = candidate_score < best_score;
                trace.push(SearchStep {
                    round,
                    family: *family,
                    candidate: rule.to_string(),
                    score: candidate_score,
                    accepted,
                });
                if accepted {
                    rules = candidate;
                    best_score = candidate_score;
                }
            }
        }
    }
    Ok((rules, meta_cost, trace))
}
