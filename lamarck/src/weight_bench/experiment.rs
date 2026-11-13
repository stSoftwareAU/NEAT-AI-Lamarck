//! The predeclared experiment: calibrate the router on calibration seeds, then
//! compare the arms on disjoint validation seeds.

use std::collections::HashSet;

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::objective::{BenchError, BudgetedObjective, WeightObjective};
use super::probes::probe_landscape;
use super::router::{Arm, CalibrationRow, ROUTER_PROBE_AXES, RouterRule, calibrate, run_arm};
use super::stats::{bootstrap_ci_mean, mean, median};
use super::suite::{
    CorpusSpec, Family, MAX_DIMENSION, MIN_DIMENSION, Problem, ProblemSpec, build_problem,
    is_extrapolation, load_corpus, problem_from_corpus,
};

/// Budget multiples (of the dimension) the experiment may use.
pub const ALLOWED_BUDGET_MULTIPLES: [usize; 3] = [10, 30, 100];
/// Confirmation evaluations per successful run (full and held-out records, at
/// the start and at the best point).
pub const CONFIRMATION_EVALUATIONS: usize = 4;

/// What to run. Declared before any result is seen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    /// Squash families.
    pub families: Vec<Family>,
    /// Free-weight counts (`20..=100`).
    pub dimensions: Vec<usize>,
    /// Budgets as multiples of the dimension (10, 30 or 100).
    pub budget_multiples: Vec<usize>,
    /// Seeds the router is calibrated on.
    pub calibration_seeds: Vec<u64>,
    /// Seeds the comparison is judged on; disjoint from calibration.
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
    /// Reject a manifest that could not support a clean comparison.
    pub fn validate(&self) -> Result<(), BenchError> {
        let bad = |m: String| Err(BenchError::InvalidConfig(m));
        if self.families.is_empty()
            || self.dimensions.is_empty()
            || self.budget_multiples.is_empty()
        {
            return bad("families, dimensions and budgetMultiples must be non-empty".into());
        }
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
        if self.calibration_seeds.is_empty() {
            return bad("calibrationSeeds must be non-empty".into());
        }
        if self.validation_seeds.is_empty() {
            return bad("validationSeeds must be non-empty".into());
        }
        let calibration: HashSet<u64> = self.calibration_seeds.iter().copied().collect();
        if let Some(s) = self
            .validation_seeds
            .iter()
            .find(|s| calibration.contains(s))
        {
            return bad(format!("seed {s} is in both calibration and validation"));
        }
        if self.bootstrap_resamples == 0 {
            return bad("bootstrapResamples must be positive".into());
        }
        Ok(())
    }
}

/// The manifest declared for issue #266.
pub fn predeclared_manifest() -> Manifest {
    Manifest {
        families: vec![Family::Tanh, Family::Relu, Family::HardTanh],
        dimensions: vec![20, 40, 100],
        budget_multiples: vec![10, 30, 100],
        calibration_seeds: (1..=8).collect(),
        validation_seeds: (1001..=1012).collect(),
        bootstrap_resamples: 2000,
        bootstrap_seed: 266,
        corpus: None,
    }
}

/// One optimiser run on one problem.
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
    /// `"calibration"` or `"validation"`.
    pub phase: String,
    /// The arm run.
    pub arm: Arm,
    /// For the router, the arm it chose.
    pub routed_to: Option<Arm>,
    /// Screening loss at the start.
    pub start_screening_loss: Option<f64>,
    /// Best screening loss found.
    pub final_screening_loss: Option<f64>,
    /// Final / start screening loss.
    pub normalised_screening_loss: Option<f64>,
    /// Budgeted evaluations used (router: includes the probe).
    pub evaluations_used: usize,
    /// Router only: evaluations the landscape probe spent.
    pub probe_evaluations: usize,
    /// Evaluations spent choosing the free weights.
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
    /// Unbudgeted confirmation evaluations (reported separately).
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
    /// Router vs refine, router vs CMA-ES, CMA-ES vs refine.
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
    /// The rule chosen on calibration seeds.
    pub calibrated_rule: RouterRule,
    /// Evaluations spent by the probe-only calibration passes.
    pub calibration_probe_evaluations: usize,
    /// Budgeted + preparation + confirmation + calibration-probe evaluations.
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
        routed_to: None,
        start_screening_loss: None,
        final_screening_loss: None,
        normalised_screening_loss: None,
        evaluations_used: 0,
        probe_evaluations: 0,
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
    rule: &RouterRule,
) -> RunRow {
    let mut row = blank_row(problem, multiple, phase, arm);
    let mut budgeted = BudgetedObjective::new(objective, row.budget);
    let outcome = run_arm(&mut budgeted, arm, rule, problem.spec.seed);
    row.evaluations_used = budgeted.used();
    let result = outcome.and_then(|routed| {
        let (best_loss, best_x) = budgeted
            .best()
            .ok_or_else(|| BenchError::Evaluation("no evaluation completed".into()))?;
        let start = budgeted.start();
        let confirm = |x: &[f64], records: &[f32]| {
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
        Ok((
            routed, start_loss, best_loss, full_s, full_f, held_s, held_f,
        ))
    });
    match result {
        Ok((routed, start, best, full_s, full_f, held_s, held_f)) => {
            if arm == Arm::Router {
                row.routed_to = Some(routed);
                row.probe_evaluations = 2 * problem.spec.dimension.min(rule.probe_axes) + 1;
            }
            row.start_screening_loss = Some(start);
            row.final_screening_loss = Some(best);
            row.normalised_screening_loss = Some(ratio(best, start));
            row.full_loss_start = Some(full_s);
            row.full_loss_final = Some(full_f);
            row.held_out_loss_start = Some(held_s);
            row.held_out_loss_final = Some(held_f);
            row.normalised_held_out_loss = Some(ratio(held_f, held_s));
            row.confirmation_evaluations = CONFIRMATION_EVALUATIONS;
        }
        Err(e) => row.error = Some(e.to_string()),
    }
    row
}

/// Visit every (family, dimension, seed) problem once, in manifest order.
fn problems(
    m: &Manifest,
    seeds: &[u64],
    mut visit: impl FnMut(&Problem, &WeightObjective) -> Result<(), BenchError>,
) -> Result<(), BenchError> {
    let corpus = match &m.corpus {
        Some(spec) => Some((spec, load_corpus(spec)?)),
        None => None,
    };
    for &family in &m.families {
        for &dimension in &m.dimensions {
            for &seed in seeds {
                let problem = match &corpus {
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
                visit(&problem, &objective)?;
            }
        }
    }
    Ok(())
}

struct Calibration {
    rows: Vec<RunRow>,
    table: Vec<CalibrationRow>,
    probe_evaluations: usize,
}

/// Phase 1: both fixed arms plus a probe-only pass on the calibration seeds.
fn calibration_phase(m: &Manifest) -> Result<Calibration, BenchError> {
    let default_rule = RouterRule {
        min_budget_multiple: None,
        max_non_smooth_share: 0.0,
        probe_axes: ROUTER_PROBE_AXES,
    };
    let mut out = Calibration {
        rows: Vec::new(),
        table: Vec::new(),
        probe_evaluations: 0,
    };
    problems(m, &m.calibration_seeds, |problem, objective| {
        let dimension = problem.spec.dimension;
        let mut probe_obj =
            BudgetedObjective::new(objective, 2 * ROUTER_PROBE_AXES.min(dimension) + 1);
        let probe = probe_landscape(&mut probe_obj, ROUTER_PROBE_AXES);
        out.probe_evaluations += probe_obj.used();
        for &multiple in &m.budget_multiples {
            let refine = run_row(
                problem,
                objective,
                multiple,
                "calibration",
                Arm::LamarckRefine,
                &default_rule,
            );
            let cma = run_row(
                problem,
                objective,
                multiple,
                "calibration",
                Arm::CmaEs,
                &default_rule,
            );
            // A failed arm, or a failed probe, cannot be scored: count a failed
            // arm as "no improvement" (1.0) and skip a row whose probe failed.
            if let Ok(probe) = &probe {
                out.table.push(CalibrationRow {
                    dimension,
                    budget: multiple * dimension,
                    probe: probe.clone(),
                    refine_normalised: refine.normalised_screening_loss.unwrap_or(1.0),
                    cmaes_normalised: cma.normalised_screening_loss.unwrap_or(1.0),
                });
            }
            out.rows.push(refine);
            out.rows.push(cma);
        }
        Ok(())
    })?;
    Ok(out)
}

/// Phase 2: all three arms on the validation seeds.
fn validation_phase(m: &Manifest, rule: &RouterRule) -> Result<Vec<RunRow>, BenchError> {
    let mut rows = Vec::new();
    problems(m, &m.validation_seeds, |problem, objective| {
        for &multiple in &m.budget_multiples {
            for arm in [Arm::LamarckRefine, Arm::CmaEs, Arm::Router] {
                rows.push(run_row(
                    problem,
                    objective,
                    multiple,
                    "validation",
                    arm,
                    rule,
                ));
            }
        }
        Ok(())
    })?;
    Ok(rows)
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
        Arm::Router => "router",
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

fn summarise_cells(m: &Manifest, runs: &[RunRow], rng: &mut StdRng) -> Vec<CellSummary> {
    let pairs = [
        (Arm::Router, Arm::LamarckRefine),
        (Arm::Router, Arm::CmaEs),
        (Arm::CmaEs, Arm::LamarckRefine),
    ];
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

/// Run the whole manifest: calibrate, validate, summarise.
///
/// A failed run is recorded as a row with `error` set and counted in the
/// coverage; only a configuration or problem-construction error aborts.
pub fn run_manifest(m: &Manifest) -> Result<ExperimentResults, BenchError> {
    m.validate()?;
    let calibration = calibration_phase(m)?;
    let calibrated_rule = calibrate(&calibration.table);
    let validation = validation_phase(m, &calibrated_rule)?;

    let mut runs = calibration.rows;
    runs.extend(validation);
    let planned = m.families.len()
        * m.dimensions.len()
        * m.budget_multiples.len()
        * (m.calibration_seeds.len() * 2 + m.validation_seeds.len() * 3);
    let failed = runs.iter().filter(|r| r.error.is_some()).count();
    let total_evaluations = calibration.probe_evaluations
        + runs
            .iter()
            .map(|r| r.evaluations_used + r.preparation_evaluations + r.confirmation_evaluations)
            .sum::<usize>();
    let mut rng = StdRng::seed_from_u64(m.bootstrap_seed);
    let cells = summarise_cells(m, &runs, &mut rng);
    Ok(ExperimentResults {
        format_version: 1,
        manifest: m.clone(),
        calibrated_rule,
        calibration_probe_evaluations: calibration.probe_evaluations,
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
