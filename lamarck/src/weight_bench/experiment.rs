//! The predeclared experiment: compare the arms on paired validation seeds.
//!
//! Neither arm has a tuned parameter, so there is no calibration phase. The
//! router that needed one was removed after an earlier run (crate 0.1.46),
//! where it never beat CMA-ES alone.

use std::collections::HashSet;

use neat_core::CreatureExport;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::arms::{Arm, run_arm};
use super::objective::{BenchError, BudgetedObjective, WeightObjective};
use super::stats::{bootstrap_ci_mean, mean, median};
use super::suite::{
    CorpusSpec, Family, MAX_DIMENSION, MIN_DIMENSION, Problem, ProblemSpec, build_problem,
    is_extrapolation, load_corpus, problem_from_corpus,
};

/// Budget multiples (of the dimension) the experiment may use.
pub const ALLOWED_BUDGET_MULTIPLES: [usize; 3] = [10, 30, 100];
/// Confirmation evaluations per successful run: the start on the screening
/// records, then the start and the best point on the full records, then the
/// start and the best point on the held-out records. The row records the
/// number of calls actually made; this constant states the expectation.
pub const CONFIRMATION_EVALUATIONS: usize = 5;

/// What to run. Declared before any result is seen.
///
/// Unknown fields are refused, so a stale router-era manifest (with
/// `calibrationSeeds`) cannot run while silently ignoring part of what it
/// declared.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Manifest {
    /// Squash families.
    pub families: Vec<Family>,
    /// Free-weight counts (`20..=100`).
    pub dimensions: Vec<usize>,
    /// Budgets as multiples of the dimension (10, 30 or 100).
    pub budget_multiples: Vec<usize>,
    /// Seeds the comparison is judged on; each runs every arm on one problem.
    pub validation_seeds: Vec<u64>,
    /// Bootstrap resamples for the confidence intervals.
    pub bootstrap_resamples: usize,
    /// Seed of the bootstrap RNG.
    pub bootstrap_seed: u64,
    /// A real creature and records; when set, `families` must be `[Corpus]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus: Option<CorpusSpec>,
}

impl Manifest {
    /// Reject a manifest that could not support a clean comparison, including
    /// repeated cells or seeds that would count a trial more than once.
    pub fn validate(&self) -> Result<(), BenchError> {
        let bad = |m: String| Err(BenchError::InvalidConfig(m));
        if self.families.is_empty()
            || self.dimensions.is_empty()
            || self.budget_multiples.is_empty()
        {
            return bad("families, dimensions and budgetMultiples must be non-empty".into());
        }
        unique("families", &self.families)?;
        unique("dimensions", &self.dimensions)?;
        unique("budgetMultiples", &self.budget_multiples)?;
        unique("validationSeeds", &self.validation_seeds)?;
        match &self.corpus {
            Some(_) if self.families != [Family::Corpus] => {
                return bad("a corpus manifest must declare families exactly [corpus]".into());
            }
            None if self.families.contains(&Family::Corpus) => {
                return bad("family corpus needs a corpus specification in the manifest".into());
            }
            _ => {}
        }
        if let Some(d) = self
            .dimensions
            .iter()
            .find(|d| !(MIN_DIMENSION..=MAX_DIMENSION).contains(d))
        {
            return bad(format!(
                "dimension {d} outside {MIN_DIMENSION}..={MAX_DIMENSION}"
            ));
        }
        if let Some(b) = self
            .budget_multiples
            .iter()
            .find(|b| !ALLOWED_BUDGET_MULTIPLES.contains(b))
        {
            return bad(format!(
                "budget multiple {b} is not one of {ALLOWED_BUDGET_MULTIPLES:?}"
            ));
        }
        if self.validation_seeds.is_empty() {
            return bad("validationSeeds must be non-empty".into());
        }
        if self.bootstrap_resamples == 0 {
            return bad("bootstrapResamples must be positive".into());
        }
        Ok(())
    }
}

fn unique<T: Eq + std::hash::Hash>(field: &str, values: &[T]) -> Result<(), BenchError> {
    if values.iter().collect::<HashSet<_>>().len() != values.len() {
        return Err(BenchError::InvalidConfig(format!(
            "duplicate entry in {field}"
        )));
    }
    Ok(())
}

/// The manifest declared for issue #266.
pub fn predeclared_manifest() -> Manifest {
    Manifest {
        families: vec![Family::Tanh, Family::Relu, Family::HardTanh],
        dimensions: vec![20, 40, 100],
        budget_multiples: vec![10, 30, 100],
        validation_seeds: (1001..=1012).collect(),
        bootstrap_resamples: 2000,
        bootstrap_seed: 266,
        corpus: None,
    }
}

/// One optimiser run on one problem.
///
/// Unknown fields are tolerated: the committed AMM rows
/// (`docs/evidence/weight-router/amm-runs.jsonl`) were written while the router
/// existed and still carry `routedTo` and `probeEvaluations`. They are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRow {
    /// Squash family.
    pub family: Family,
    /// Free weights.
    pub dimension: usize,
    /// Budget as a multiple of the dimension.
    pub budget_multiple: usize,
    /// Budget in evaluations.
    pub budget: usize,
    /// Problem seed.
    pub seed: u64,
    /// Always `"validation"`; router-era `runs.jsonl` rows (not committed) also carried `"calibration"`.
    pub phase: String,
    /// The arm run.
    pub arm: Arm,
    /// Screening loss at the start.
    pub start_screening_loss: Option<f64>,
    /// Best screening loss found.
    pub final_screening_loss: Option<f64>,
    /// Final / start screening loss.
    pub normalised_screening_loss: Option<f64>,
    /// Budgeted evaluations used.
    pub evaluations_used: usize,
    /// Evaluations spent choosing the free weights. This is the cost of
    /// building the row's problem, so it is repeated on every row of that
    /// problem; totals count it once per problem built.
    pub preparation_evaluations: usize,
    /// Full-record loss at the start.
    pub full_loss_start: Option<f64>,
    /// Full-record loss at the best point.
    pub full_loss_final: Option<f64>,
    /// Held-out loss at the start.
    pub held_out_loss_start: Option<f64>,
    /// Held-out loss at the best point.
    pub held_out_loss_final: Option<f64>,
    /// Final / start held-out loss.
    pub normalised_held_out_loss: Option<f64>,
    /// Unbudgeted confirmation attempts, including any failed call (reported separately).
    pub confirmation_evaluations: usize,
    /// Whether the dimension is beyond the calibrated range.
    pub extrapolation: bool,
    /// Why the run failed, if it did.
    pub error: Option<String>,
}

/// Planned versus finished runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Coverage {
    /// Runs the manifest asked for.
    pub planned_runs: usize,
    /// Runs that finished without error.
    pub completed_runs: usize,
    /// Runs that recorded an error.
    pub failed_runs: usize,
}

/// Paired comparison of one loss metric (first arm minus second arm).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetricSummary {
    /// Per-seed differences, in validation-seed order.
    pub paired_differences: Vec<f64>,
    /// Mean difference.
    pub mean: Option<f64>,
    /// Median difference.
    pub median: Option<f64>,
    /// Lower end of the 95% bootstrap interval of the mean.
    pub ci_low: Option<f64>,
    /// Upper end of the 95% bootstrap interval of the mean.
    pub ci_high: Option<f64>,
    /// Seeds where the first arm had the lower loss.
    pub wins: usize,
    /// Seeds where the first arm had the higher loss.
    pub losses: usize,
    /// Seeds with equal loss.
    pub ties: usize,
    /// Seeds dropped because either arm failed.
    pub failures: usize,
}

/// One arm pair compared in one cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Comparison {
    /// The arm whose loss is on the left of the difference.
    pub first: Arm,
    /// The arm subtracted.
    pub second: Arm,
    /// On normalised screening loss.
    pub screening: MetricSummary,
    /// On normalised held-out loss.
    pub held_out: MetricSummary,
    /// `"<first>-better"`, `"<first>-worse"` or `"inconclusive"`.
    pub verdict: String,
}

/// All comparisons for one (family, dimension, budget) validation cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CellSummary {
    /// Squash family.
    pub family: Family,
    /// Free weights.
    pub dimension: usize,
    /// Budget multiple.
    pub budget_multiple: usize,
    /// Whether the dimension is beyond the calibrated range.
    pub extrapolation: bool,
    /// One paired comparison per arm pair requested of the summary, in the
    /// order requested (the main experiment: CMA-ES vs refine).
    pub comparisons: Vec<Comparison>,
}

/// Everything the experiment produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentResults {
    /// Output format version.
    pub format_version: u32,
    /// The manifest that was run.
    pub manifest: Manifest,
    /// Budgeted + confirmation evaluations of every row, plus preparation
    /// counted once per problem built (not once per row).
    pub total_evaluations: usize,
    /// Planned versus completed runs.
    pub coverage: Coverage,
    /// Every run, failures included.
    pub runs: Vec<RunRow>,
    /// Validation summaries.
    pub cells: Vec<CellSummary>,
}

fn blank_row(problem: &Problem, multiple: usize, phase: &str, arm: Arm) -> RunRow {
    let dimension = problem.spec.dimension;
    RunRow {
        family: problem.spec.family,
        dimension,
        budget_multiple: multiple,
        budget: multiple * dimension,
        seed: problem.spec.seed,
        phase: phase.to_string(),
        arm,
        start_screening_loss: None,
        final_screening_loss: None,
        normalised_screening_loss: None,
        evaluations_used: 0,
        preparation_evaluations: problem.preparation_evaluations,
        full_loss_start: None,
        full_loss_final: None,
        held_out_loss_start: None,
        held_out_loss_final: None,
        normalised_held_out_loss: None,
        confirmation_evaluations: 0,
        extrapolation: is_extrapolation(dimension),
        error: None,
    }
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    numerator / denominator.max(1e-300)
}

/// Run one arm on a fresh budget and record the outcome; a failure is recorded
/// on the row, never propagated.
fn run_row(
    problem: &Problem,
    objective: &WeightObjective,
    multiple: usize,
    phase: &str,
    arm: Arm,
) -> RunRow {
    finish_row(problem, objective, multiple, phase, arm, |budgeted| {
        run_arm(budgeted, arm, problem.spec.seed)
    })
}

/// Run `optimise` on a fresh budget, then do the confirmation passes and build
/// the row. A failure is recorded on the row, never propagated.
pub(super) fn finish_row(
    problem: &Problem,
    objective: &WeightObjective,
    multiple: usize,
    phase: &str,
    arm: Arm,
    optimise: impl FnOnce(&mut BudgetedObjective<'_>) -> Result<(), BenchError>,
) -> RunRow {
    let mut row = blank_row(problem, multiple, phase, arm);
    let mut budgeted = BudgetedObjective::new(objective, row.budget);
    let confirmations = std::cell::Cell::new(0usize);
    let outcome = optimise(&mut budgeted);
    row.evaluations_used = budgeted.used();
    let result = outcome.and_then(|()| {
        let (best_loss, best_x) = budgeted
            .best()
            .ok_or_else(|| BenchError::Evaluation("no evaluation completed".into()))?;
        let start = budgeted.start();
        let confirm = |x: &[f64], records: &[f32]| {
            confirmations.set(confirmations.get() + 1);
            objective
                .loss_on(x, records)
                .map_err(BenchError::Evaluation)
        };
        let start_loss = confirm(&start, &problem.screening)?;
        let (full_s, full_f) = (
            confirm(&start, &problem.full)?,
            confirm(best_x, &problem.full)?,
        );
        let (held_s, held_f) = (
            confirm(&start, &problem.held_out)?,
            confirm(best_x, &problem.held_out)?,
        );
        Ok((start_loss, best_loss, full_s, full_f, held_s, held_f))
    });
    match result {
        Ok((start, best, full_s, full_f, held_s, held_f)) => {
            row.start_screening_loss = Some(start);
            row.final_screening_loss = Some(best);
            row.normalised_screening_loss = Some(ratio(best, start));
            row.full_loss_start = Some(full_s);
            row.full_loss_final = Some(full_f);
            row.held_out_loss_start = Some(held_s);
            row.held_out_loss_final = Some(held_f);
            row.normalised_held_out_loss = Some(ratio(held_f, held_s));
            debug_assert_eq!(confirmations.get(), CONFIRMATION_EVALUATIONS);
        }
        Err(e) => row.error = Some(e.to_string()),
    }
    row.confirmation_evaluations = confirmations.get();
    row
}

/// A manifest's corpus, loaded once (`None` for the synthetic suite).
pub(super) type LoadedCorpus<'a> = Option<(&'a CorpusSpec, (CreatureExport, Vec<f32>))>;

/// Load the manifest's corpus, if it declares one.
pub(super) fn load_manifest_corpus(m: &Manifest) -> Result<LoadedCorpus<'_>, BenchError> {
    match &m.corpus {
        Some(spec) => Ok(Some((spec, load_corpus(spec)?))),
        None => Ok(None),
    }
}

/// Build one (problem, objective) pair exactly as the experiment does.
pub(super) fn build_one(
    corpus: &LoadedCorpus<'_>,
    family: Family,
    dimension: usize,
    seed: u64,
) -> Result<(Problem, WeightObjective), BenchError> {
    let problem = match corpus {
        Some((spec, (creature, packed))) => {
            problem_from_corpus(creature, packed, spec, dimension, seed)?
        }
        None => build_problem(&ProblemSpec {
            family,
            dimension,
            seed,
        })?,
    };
    let objective = WeightObjective::new(
        problem.creature.clone(),
        problem.synapse_indices.clone(),
        &problem.screening,
    )?;
    Ok((problem, objective))
}

/// Every arm on every validation problem and budget, in manifest order.
/// Returns the rows and the preparation evaluations, counted once per problem.
fn validation_phase(m: &Manifest) -> Result<(Vec<RunRow>, usize), BenchError> {
    let corpus = load_manifest_corpus(m)?;
    let mut rows = Vec::new();
    let mut preparation = 0;
    for &family in &m.families {
        for &dimension in &m.dimensions {
            for &seed in &m.validation_seeds {
                let (problem, objective) = build_one(&corpus, family, dimension, seed)?;
                preparation += problem.preparation_evaluations;
                for &multiple in &m.budget_multiples {
                    for arm in [Arm::LamarckRefine, Arm::CmaEs] {
                        rows.push(run_row(&problem, &objective, multiple, "validation", arm));
                    }
                }
            }
        }
    }
    Ok((rows, preparation))
}

fn summarise_metric(
    first: &[Option<f64>],
    second: &[Option<f64>],
    resamples: usize,
    rng: &mut StdRng,
) -> MetricSummary {
    let mut diffs = Vec::new();
    let mut failures = 0;
    for (a, b) in first.iter().zip(second) {
        match (a, b) {
            (Some(a), Some(b)) => diffs.push(a - b),
            _ => failures += 1,
        }
    }
    let ci = bootstrap_ci_mean(&diffs, resamples, rng);
    MetricSummary {
        mean: mean(&diffs),
        median: median(&diffs),
        ci_low: ci.map(|c| c.0),
        ci_high: ci.map(|c| c.1),
        wins: diffs.iter().filter(|d| **d < 0.0).count(),
        losses: diffs.iter().filter(|d| **d > 0.0).count(),
        ties: diffs.iter().filter(|d| **d == 0.0).count(),
        failures,
        paired_differences: diffs,
    }
}

fn arm_name(arm: Arm) -> &'static str {
    match arm {
        Arm::LamarckRefine => "lamarck-refine",
        Arm::CmaEs => "cma-es",
        Arm::AmmReference => "amm-reference",
    }
}

/// "better" iff the screening interval is wholly below zero and the held-out
/// mean is not above zero; "worse" iff the screening interval is wholly above.
fn verdict(first: Arm, screening: &MetricSummary, held_out: &MetricSummary) -> String {
    let name = arm_name(first);
    match (screening.ci_low, screening.ci_high, held_out.mean) {
        (_, Some(hi), Some(held)) if hi < 0.0 && held <= 0.0 => format!("{name}-better"),
        (Some(lo), _, _) if lo > 0.0 => format!("{name}-worse"),
        _ => "inconclusive".to_string(),
    }
}

/// One paired comparison per `pairs` entry, for every validation cell.
pub(super) fn summarise_cells(
    m: &Manifest,
    runs: &[RunRow],
    pairs: &[(Arm, Arm)],
    rng: &mut StdRng,
) -> Vec<CellSummary> {
    let mut cells = Vec::new();
    for &family in &m.families {
        for &dimension in &m.dimensions {
            for &multiple in &m.budget_multiples {
                // Per validation seed, in manifest order: (screening, held-out).
                let pick = |arm: Arm| -> Vec<(Option<f64>, Option<f64>)> {
                    m.validation_seeds
                        .iter()
                        .map(|&seed| {
                            runs.iter()
                                .find(|r| {
                                    r.phase == "validation"
                                        && r.family == family
                                        && r.dimension == dimension
                                        && r.budget_multiple == multiple
                                        && r.arm == arm
                                        && r.seed == seed
                                })
                                .filter(|r| r.error.is_none())
                                .map_or((None, None), |r| {
                                    (r.normalised_screening_loss, r.normalised_held_out_loss)
                                })
                        })
                        .collect()
                };
                let comparisons = pairs
                    .iter()
                    .map(|&(first, second)| {
                        let (a, b) = (pick(first), pick(second));
                        let col =
                            |v: &[(Option<f64>, Option<f64>)], held: bool| -> Vec<Option<f64>> {
                                v.iter().map(|p| if held { p.1 } else { p.0 }).collect()
                            };
                        let screening = summarise_metric(
                            &col(&a, false),
                            &col(&b, false),
                            m.bootstrap_resamples,
                            rng,
                        );
                        let held_out = summarise_metric(
                            &col(&a, true),
                            &col(&b, true),
                            m.bootstrap_resamples,
                            rng,
                        );
                        Comparison {
                            first,
                            second,
                            verdict: verdict(first, &screening, &held_out),
                            screening,
                            held_out,
                        }
                    })
                    .collect();
                cells.push(CellSummary {
                    family,
                    dimension,
                    budget_multiple: multiple,
                    extrapolation: is_extrapolation(dimension),
                    comparisons,
                });
            }
        }
    }
    cells
}

/// Run the whole manifest: validate, then summarise the paired comparisons.
///
/// A failed run is recorded as a row with `error` set and counted in the
/// coverage; only a configuration or problem-construction error aborts.
pub fn run_manifest(m: &Manifest) -> Result<ExperimentResults, BenchError> {
    m.validate()?;
    let (runs, preparation) = validation_phase(m)?;
    let planned = m.families.len()
        * m.dimensions.len()
        * m.budget_multiples.len()
        * m.validation_seeds.len()
        * 2;
    let failed = runs.iter().filter(|r| r.error.is_some()).count();
    // Preparation is a per-problem cost repeated on each of the problem's
    // rows, so it is counted once per problem built, not summed over rows.
    let total_evaluations = preparation
        + runs
            .iter()
            .map(|r| r.evaluations_used + r.confirmation_evaluations)
            .sum::<usize>();
    let mut rng = StdRng::seed_from_u64(m.bootstrap_seed);
    let cells = summarise_cells(m, &runs, &[(Arm::CmaEs, Arm::LamarckRefine)], &mut rng);
    Ok(ExperimentResults {
        format_version: 2,
        manifest: m.clone(),
        total_evaluations,
        coverage: Coverage {
            planned_runs: planned,
            completed_runs: runs.len() - failed,
            failed_runs: failed,
        },
        runs,
        cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_confirmations_keep_the_calls_already_spent() {
        let (problem, objective) = build_one(&None, Family::Tanh, 20, 1001).unwrap();
        for (invalid_full, expected_calls) in [(true, 2), (false, 4)] {
            let mut failed = problem.clone();
            if invalid_full {
                failed.full.clear();
            } else {
                failed.held_out.clear();
            }
            let row = finish_row(
                &failed,
                &objective,
                10,
                "validation",
                Arm::AmmReference,
                |obj| {
                    obj.eval(&obj.start())?;
                    Ok(())
                },
            );
            assert_eq!(row.evaluations_used, 1);
            assert_eq!(
                row.error.as_deref(),
                Some("evaluation failed: records are empty or not a multiple of the record width")
            );
            assert_eq!(row.confirmation_evaluations, expected_calls);
        }
        let control = finish_row(
            &problem,
            &objective,
            10,
            "validation",
            Arm::AmmReference,
            |obj| {
                obj.eval(&obj.start())?;
                Ok(())
            },
        );
        assert_eq!(control.error, None);
        assert_eq!(control.confirmation_evaluations, 5);
        assert_eq!(control.normalised_screening_loss, Some(1.0));
    }
}
