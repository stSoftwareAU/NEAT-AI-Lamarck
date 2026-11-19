//! Refinement driver and manifest runner for the record-skip experiment.
//!
//! Every arm refines a clone of the same immutable parent through the real
//! refinement update path. Audit measurements are taken after the loop; they
//! are not charged to the arm's `record_passes` but are counted in
//! `audit_passes`. Full training MSE is audited in every phase; held-out
//! (validation) records are read only by validation-phase runs, so preparation
//! and calibration choices never depend on them.

use std::path::Path;

use neat_core::{CompiledNetwork, CreatureExport, compile_creature};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;

use crate::backprop::{BackpropConfig, apply_learnings, calculate_learning_rate};
use crate::memo::creature_fingerprint;
use crate::propagate_layout::LearningPlan;

pub use super::corpus::{BenchRecord, Corpus, load_corpus, synthetic_corpus, synthetic_parent};
pub use super::manifest::{GridPoint, Manifest};
use super::schedule::{
    DifficultyScope, DifficultyTracker, ScheduleConfig, SkipPolicy, StratumEdges, rare_flags,
};
use super::stats::{Verdict, bootstrap_mean_ci, mean, verdict};

/// The manifest this crate was built from. The neat-core pin is read from it, so
/// the reported tag follows the pin when the automated refresh moves it.
const CARGO_TOML: &str = include_str!("../../Cargo.toml");

/// neat-core tag this crate is built against, read from `lamarck/Cargo.toml`.
pub fn neat_core_tag() -> String {
    neat_core_tag_from(CARGO_TOML)
}

/// Extract the `tag` of the `neat-core` dependency from manifest text, or
/// `unknown` when the dependency carries no tag.
pub fn neat_core_tag_from(manifest: &str) -> String {
    manifest
        .lines()
        .map(str::trim_start)
        .find(|l| l.starts_with("neat-core") && l[9..].trim_start().starts_with('='))
        .and_then(|l| {
            let rest = &l[l.find("tag")? + 3..];
            let rest = rest.trim_start().strip_prefix('=')?.trim_start();
            let rest = rest.strip_prefix('"')?;
            Some(rest[..rest.find('"')?].to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Identify the checkout in `dir` for `versions.gitCommit`: the commit hash,
/// with `-dirty` appended when `git status --porcelain` reports any change, or
/// `unavailable: <cause>` when git cannot say. The probe itself is the shared
/// [`crate::weight_bench::provenance::git_state`].
pub fn git_commit(dir: &Path) -> String {
    match crate::weight_bench::provenance::git_state(dir) {
        Ok(state) if state.dirty => format!("{}-dirty", state.commit),
        Ok(state) => state.commit,
        Err(cause) => format!("unavailable: {cause}"),
    }
}

/// Result of one refinement run.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunOutcome {
    /// Arm label, for example `easy-50`.
    pub arm: String,
    /// `preparation`, `calibration` or `validation` (set by the runner).
    pub phase: String,
    /// Seed driving the random and revisit draws.
    pub seed: u64,
    /// EWMA smoothing factor (null for the full arm).
    pub ewma_alpha: Option<f64>,
    /// Revisit share (null for full and random arms).
    pub revisit_share: Option<f64>,
    /// Training MSE of the parent.
    pub initial_train_mse: f64,
    /// Training MSE of the refined creature over all training records.
    pub final_train_mse: f64,
    /// Held-out validation MSE (null on preparation and calibration runs,
    /// which never read held-out records).
    pub final_validation_mse: Option<f64>,
    /// Validation MSE over records in protected (rare) strata (null on
    /// preparation and calibration runs, which never read held-out records).
    pub rare_stratum_validation_mse: Option<f64>,
    /// Validation MSE per stratum (empty on preparation and calibration runs,
    /// which never read held-out records).
    pub strata_validation_mse: Vec<Option<f64>>,
    /// Forward and backward record passes charged to the arm.
    pub record_passes: u64,
    /// Forward-only measurement passes (initial and final training MSE, plus
    /// held-out records on validation runs); a search cost counted apart from
    /// `record_passes`.
    pub audit_passes: u64,
    /// Record passes as a share of epochs times training records.
    pub processed_share: f64,
    /// Passes spent on easy-pool revisits.
    pub revisit_passes: u64,
    /// Scheduling operations.
    pub schedule_ops: u64,
    /// Share of observed records whose error rose after their last visit.
    pub forgotten_share: f64,
    /// Parent creature fingerprint.
    pub parent_fingerprint: String,
    /// Corpus key.
    pub corpus_key: String,
}

/// Arm label for a policy.
fn arm_label(policy: &SkipPolicy) -> String {
    match policy {
        SkipPolicy::Full => "full".to_string(),
        SkipPolicy::EasySkip { reduction } => {
            format!("easy-{}", (reduction * 100.0).round() as u32)
        }
        SkipPolicy::RandomSkip { reduction } => {
            format!("random-{}", (reduction * 100.0).round() as u32)
        }
    }
}

fn compile(creature: &CreatureExport) -> Result<CompiledNetwork, String> {
    compile_creature(creature).map_err(|e| e.to_string())
}

/// Per-record mean squared error of `creature` over `records`.
fn record_errors(creature: &CreatureExport, records: &[BenchRecord]) -> Result<Vec<f64>, String> {
    let mut network = compile(creature)?;
    Ok(records
        .iter()
        .map(|r| {
            squared_error(
                &network.activate_and_trace(&r.inputs, creature.output),
                &r.outputs,
            )
        })
        .collect())
}

fn squared_error(out: &[f32], expected: &[f32]) -> f64 {
    let n = expected.len().max(1);
    out.iter()
        .zip(expected)
        .map(|(o, e)| {
            let d = f64::from(*o) - f64::from(*e);
            d * d
        })
        .sum::<f64>()
        / n as f64
}

/// Passes and operations a run has spent so far. `refine_inner` keeps it
/// current, so a run that fails after processing records still reports them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Spend {
    record_passes: u64,
    audit_passes: u64,
    revisit_passes: u64,
    schedule_ops: u64,
}

/// Refine a clone of `parent` over `corpus.train` under `policy` at `learning_rate`.
// The arms differ only in these inputs; a parameter struct would add noise.
#[allow(clippy::too_many_arguments)]
pub fn refine(
    parent: &CreatureExport,
    corpus: &Corpus,
    scope: &DifficultyScope,
    policy: &SkipPolicy,
    schedule: &ScheduleConfig,
    manifest: &Manifest,
    learning_rate: f64,
    rng_seed: u64,
) -> Result<RunOutcome, String> {
    refine_inner(
        parent,
        corpus,
        scope,
        policy,
        schedule,
        manifest,
        learning_rate,
        rng_seed,
        &mut Spend::default(),
        true,
    )
}

/// [`refine`], recording what was spent in `spent` even when it returns an error.
/// Held-out records are read only when `audit_held_out` is set.
#[allow(clippy::too_many_arguments)]
fn refine_inner(
    parent: &CreatureExport,
    corpus: &Corpus,
    scope: &DifficultyScope,
    policy: &SkipPolicy,
    schedule: &ScheduleConfig,
    manifest: &Manifest,
    learning_rate: f64,
    rng_seed: u64,
    spent: &mut Spend,
    audit_held_out: bool,
) -> Result<RunOutcome, String> {
    let n = corpus.train.len();
    let first: Vec<f32> = corpus.train.iter().map(|r| r.outputs[0]).collect();
    let edges = StratumEdges::from_values(&first, manifest.strata_bins);
    let strata: Vec<u16> = first.iter().map(|&v| edges.assign(v)).collect();
    let own_scope = DifficultyScope {
        parent_fingerprint: creature_fingerprint(parent),
        corpus_key: corpus.key.clone(),
        record_count: n,
    };
    let mut tracker = DifficultyTracker::new(own_scope.clone(), strata.clone(), schedule.clone())
        .map_err(|e| e.to_string())?;

    let config = BackpropConfig {
        learning_rate,
        initial_learning_rate: learning_rate,
        ..BackpropConfig::default()
    };

    let initial = mean(&record_errors(parent, &corpus.train)?);
    spent.audit_passes += corpus.train.len() as u64;
    let mut creature = parent.clone();
    let mut sched_rng = StdRng::seed_from_u64(rng_seed);
    let mut plan_rng = StdRng::seed_from_u64(rng_seed ^ 0x9E37_79B9_7F4A_7C15);

    for epoch in 0..manifest.epochs {
        tracker.ensure_scope(scope).map_err(|e| e.to_string())?;
        let sel = tracker
            .select(epoch, policy, &mut sched_rng)
            .map_err(|e| e.to_string())?;
        let plan = LearningPlan::new(&creature, &config, &mut plan_rng)?;
        let mut network = compile(&creature)?;
        let mut scan = plan.scan();
        for (i, record) in corpus.train.iter().enumerate() {
            if !sel.keep[i] {
                continue;
            }
            let out = network.activate_and_trace(&record.inputs, creature.output);
            tracker.observe(i, squared_error(&out, &record.outputs));
            scan.observe(&network, &record.outputs);
        }
        let signal = scan.finish();
        let lr = calculate_learning_rate(&config, u64::from(epoch), None);
        creature = apply_learnings(&creature, &signal, &config, lr);
        spent.record_passes += sel.processed as u64;
        spent.revisit_passes += sel.revisits as u64;
        spent.schedule_ops += sel.schedule_ops;
    }

    // Audit: measurement only. Not charged to the arm's `record_passes`, but
    // counted in `audit_passes` because selection depends on it.
    let train_errors = record_errors(&creature, &corpus.train)?;
    spent.audit_passes += corpus.train.len() as u64;
    let final_train = mean(&train_errors);
    if !final_train.is_finite() {
        return Err("refinement diverged: non-finite training MSE".to_string());
    }

    let (final_val, rare_mse, strata_mse) = if audit_held_out {
        let val_errors = record_errors(&creature, &corpus.validation)?;
        spent.audit_passes += corpus.validation.len() as u64;
        let final_val = mean(&val_errors);
        if !final_val.is_finite() {
            return Err("refinement diverged: non-finite validation MSE".to_string());
        }
        let val_strata: Vec<u16> = corpus
            .validation
            .iter()
            .map(|r| edges.assign(r.outputs[0]))
            .collect();
        let mut sums = vec![(0.0f64, 0usize); manifest.strata_bins.max(1)];
        for (s, e) in val_strata.iter().zip(&val_errors) {
            let slot = &mut sums[usize::from(*s)];
            slot.0 += e;
            slot.1 += 1;
        }
        let strata_mse: Vec<Option<f64>> = sums
            .iter()
            .map(|&(s, c)| (c > 0).then(|| s / c as f64))
            .collect();
        let flags = rare_flags(&strata, schedule.rare_stratum_share);
        let rare: Vec<f64> = val_strata
            .iter()
            .zip(&val_errors)
            .filter(|(s, _)| flags.get(usize::from(**s)).copied().unwrap_or(false))
            .map(|(_, e)| *e)
            .collect();
        let rare_mse = (!rare.is_empty()).then(|| mean(&rare));
        (Some(final_val), rare_mse, strata_mse)
    } else {
        (None, None, Vec::new())
    };

    let (mut observed, mut forgotten) = (0usize, 0usize);
    for (i, final_err) in train_errors.iter().enumerate() {
        if let Some(last) = tracker.last_seen(i) {
            observed += 1;
            if *final_err > last + manifest.forgetting_tolerance {
                forgotten += 1;
            }
        }
    }
    let forgotten_share = if observed == 0 {
        0.0
    } else {
        forgotten as f64 / observed as f64
    };

    let full = matches!(policy, SkipPolicy::Full);
    let easy = matches!(policy, SkipPolicy::EasySkip { .. });
    Ok(RunOutcome {
        arm: arm_label(policy),
        phase: String::new(),
        seed: rng_seed,
        ewma_alpha: (!full).then_some(schedule.ewma_alpha),
        revisit_share: easy.then_some(schedule.revisit_share),
        initial_train_mse: initial,
        final_train_mse: final_train,
        final_validation_mse: final_val,
        rare_stratum_validation_mse: rare_mse,
        strata_validation_mse: strata_mse,
        record_passes: spent.record_passes,
        audit_passes: spent.audit_passes,
        processed_share: spent.record_passes as f64 / (f64::from(manifest.epochs) * n as f64),
        revisit_passes: spent.revisit_passes,
        schedule_ops: spent.schedule_ops,
        forgotten_share,
        parent_fingerprint: format!("{:016x}", own_scope.parent_fingerprint),
        corpus_key: corpus.key.clone(),
    })
}

/// Where the corpus and parent come from.
// Built once per run; boxing would only complicate the public shape.
#[allow(clippy::large_enum_variant)]
pub enum CorpusSource {
    /// Generated per seed from the manifest.
    Synthetic,
    /// One fixed corpus and parent; the seed drives only the random draws.
    Files {
        /// The loaded corpus.
        corpus: Corpus,
        /// The parent creature.
        parent: CreatureExport,
    },
}

/// A run that errored.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FailureRow {
    /// Arm label.
    pub arm: String,
    /// Seed.
    pub seed: u64,
    /// Phase.
    pub phase: String,
    /// Error text.
    pub error: String,
    /// Record passes spent before the failure; a diverged run spends them all.
    pub record_passes: u64,
    /// Measurement passes spent before the failure.
    pub audit_passes: u64,
    /// Easy-pool revisit passes spent before the failure.
    pub revisit_passes: u64,
    /// Scheduling operations spent before the failure.
    pub schedule_ops: u64,
}

/// A row of `runs.jsonl`.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum Row {
    /// Completed run.
    Ok(Box<RunOutcome>),
    /// Failed run.
    Failed(FailureRow),
}

/// Crate and neat-core versions.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Versions {
    /// This crate's version.
    #[serde(rename = "crate")]
    pub crate_version: String,
    /// neat-core tag.
    pub neat_core_tag: String,
    /// Commit of the checkout that built the run; the library leaves it empty
    /// and the example fills it in.
    pub git_commit: Option<String>,
}

/// Chosen schedule parameters.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChosenSchedule {
    /// EWMA smoothing factor.
    pub ewma_alpha: f64,
    /// Revisit share.
    pub revisit_share: f64,
}

/// Calibration result for one grid point.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationEntry {
    /// EWMA smoothing factor.
    pub ewma_alpha: f64,
    /// Revisit share.
    pub revisit_share: f64,
    /// Matched pairs contributing.
    pub n_pairs: usize,
    /// Mean of (easy - random) final training MSE; null when no pairs.
    pub mean_train_diff: Option<f64>,
}

/// Learning-rate sweep result for one rate.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LearningRateEntry {
    /// Learning rate tried.
    pub learning_rate: f64,
    /// Completed runs at this rate.
    pub n_runs: usize,
    /// Mean final training MSE; null when no run completed.
    pub mean_train_mse: Option<f64>,
}

/// Predeclared compute limit against planned and actual cost.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComputeBudget {
    /// The manifest's `maxRecordPasses`.
    pub limit_record_passes: u64,
    /// Upper bound planned before any run: runs * epochs * training records.
    /// `u64::MAX` when the bound overflowed and the run was refused.
    pub planned_bound_record_passes: u64,
    /// Record passes charged across every run, all phases, including the
    /// passes a failed run spent before it failed.
    pub actual_record_passes: u64,
    /// Forward-only measurement passes across every run, all phases; not
    /// governed by `maxRecordPasses`.
    pub actual_audit_passes: u64,
}

/// Validation-phase summary of one arm (means over its runs).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArmSummary {
    /// Arm label.
    pub arm: String,
    /// Completed validation runs.
    pub n_runs: usize,
    /// Mean record passes.
    pub record_passes: Option<f64>,
    /// Mean processed share.
    pub processed_share: Option<f64>,
    /// Mean final training MSE.
    pub final_train_mse: Option<f64>,
    /// Mean final validation MSE.
    pub final_validation_mse: Option<f64>,
    /// Mean rare-stratum validation MSE over runs that have one.
    pub rare_stratum_validation_mse: Option<f64>,
    /// Mean forgotten share.
    pub forgotten_share: Option<f64>,
    /// Mean revisit passes.
    pub revisit_passes: Option<f64>,
    /// Mean scheduling operations.
    pub schedule_ops: Option<f64>,
    /// Per-stratum mean validation MSE over runs where the stratum is present.
    pub strata_validation_mse: Vec<Option<f64>>,
}

fn arm_summary(arm: String, runs: &[&RunOutcome]) -> ArmSummary {
    let m = |f: &dyn Fn(&RunOutcome) -> f64| {
        let v: Vec<f64> = runs.iter().map(|o| f(o)).collect();
        opt(mean(&v))
    };
    let val: Vec<f64> = runs.iter().filter_map(|o| o.final_validation_mse).collect();
    let rare: Vec<f64> = runs
        .iter()
        .filter_map(|o| o.rare_stratum_validation_mse)
        .collect();
    let bins = runs
        .iter()
        .map(|o| o.strata_validation_mse.len())
        .max()
        .unwrap_or(0);
    let strata = (0..bins)
        .map(|b| {
            let v: Vec<f64> = runs
                .iter()
                .filter_map(|o| o.strata_validation_mse.get(b).copied().flatten())
                .collect();
            opt(mean(&v))
        })
        .collect();
    ArmSummary {
        arm,
        n_runs: runs.len(),
        record_passes: m(&|o| o.record_passes as f64),
        processed_share: m(&|o| o.processed_share),
        final_train_mse: m(&|o| o.final_train_mse),
        final_validation_mse: opt(mean(&val)),
        rare_stratum_validation_mse: opt(mean(&rare)),
        forgotten_share: m(&|o| o.forgotten_share),
        revisit_passes: m(&|o| o.revisit_passes as f64),
        schedule_ops: m(&|o| o.schedule_ops as f64),
        strata_validation_mse: strata,
    }
}

/// Run coverage.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Coverage {
    /// Runs planned.
    pub planned: usize,
    /// Runs completed.
    pub completed: usize,
    /// Runs failed.
    pub failed: usize,
}

/// A paired comparison of two arms at one reduction.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Comparison {
    /// Comparison label, for example `easy-50 vs random-50`.
    pub name: String,
    /// `primary` or `secondary`.
    pub role: String,
    /// Treatment arm.
    pub treatment: String,
    /// Control arm.
    pub control: String,
    /// Paired seeds.
    pub n_pairs: usize,
    /// Seeds in pair order.
    pub seeds: Vec<u64>,
    /// Per-seed validation MSE differences (treatment - control).
    pub validation_diffs: Vec<f64>,
    /// Mean validation difference.
    pub mean_validation_diff: Option<f64>,
    /// Lower bound of the 95% bootstrap interval.
    pub ci_low: Option<f64>,
    /// Upper bound of the 95% bootstrap interval.
    pub ci_high: Option<f64>,
    /// Mean rare-stratum validation difference.
    pub rare_mean_diff: Option<f64>,
    /// Mean final training MSE difference.
    pub train_mean_diff: Option<f64>,
    /// Mean record passes, treatment.
    pub treatment_record_passes: Option<f64>,
    /// Mean record passes, control.
    pub control_record_passes: Option<f64>,
    /// Mean forgotten share, treatment.
    pub treatment_forgotten_share: Option<f64>,
    /// Mean forgotten share, control.
    pub control_forgotten_share: Option<f64>,
    /// Verdict.
    pub verdict: Verdict,
}

/// Summed cost of one phase. The passes and operations include those spent by
/// runs that failed; `runs` counts completed runs only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseCost {
    /// Completed runs in the phase.
    pub runs: usize,
    /// Forward and backward record passes.
    pub record_passes: u64,
    /// Forward-only measurement passes.
    pub audit_passes: u64,
    /// Passes spent on easy-pool revisits.
    pub revisit_passes: u64,
    /// Scheduling operations.
    pub schedule_ops: u64,
}

fn phase_cost(rows: &[Row], phase: &str) -> PhaseCost {
    let mut c = PhaseCost::default();
    for row in rows {
        match row {
            Row::Ok(o) if o.phase == phase => {
                c.runs += 1;
                c.record_passes += o.record_passes;
                c.audit_passes += o.audit_passes;
                c.revisit_passes += o.revisit_passes;
                c.schedule_ops += o.schedule_ops;
            }
            Row::Failed(f) if f.phase == phase => {
                c.record_passes += f.record_passes;
                c.audit_passes += f.audit_passes;
                c.revisit_passes += f.revisit_passes;
                c.schedule_ops += f.schedule_ops;
            }
            _ => {}
        }
    }
    c
}

/// Record passes spent by every run, including those a failed run spent.
fn total_record_passes(rows: &[Row]) -> u64 {
    rows.iter()
        .map(|r| match r {
            Row::Ok(o) => o.record_passes,
            Row::Failed(f) => f.record_passes,
        })
        .sum()
}

/// Measurement passes spent by every run, including those a failed run spent.
fn total_audit_passes(rows: &[Row]) -> u64 {
    rows.iter()
        .map(|r| match r {
            Row::Ok(o) => o.audit_passes,
            Row::Failed(f) => f.audit_passes,
        })
        .sum()
}

/// Full report of a manifest run.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchReport {
    /// Manifest echo.
    pub manifest: Manifest,
    /// Versions.
    pub versions: Versions,
    /// Corpus source description.
    pub source: String,
    /// Learning rate chosen by the preparation sweep (training MSE only).
    pub chosen_learning_rate: f64,
    /// Preparation sweep table.
    pub learning_rate_sweep: Vec<LearningRateEntry>,
    /// Schedule chosen by calibration.
    pub chosen_schedule: ChosenSchedule,
    /// Calibration table.
    pub calibration: Vec<CalibrationEntry>,
    /// Coverage.
    pub coverage: Coverage,
    /// Cost of the preparation phase (learning-rate sweep).
    pub preparation_cost: PhaseCost,
    /// Predeclared compute limit, planned bound and actual cost.
    pub compute: ComputeBudget,
    /// Per-arm validation summary.
    pub arms: Vec<ArmSummary>,
    /// Cost of the calibration phase (schedule search).
    pub calibration_cost: PhaseCost,
    /// Cost of the validation phase.
    pub validation_cost: PhaseCost,
    /// Per-reduction comparisons.
    pub comparisons: Vec<Comparison>,
    /// Every run (goes to `runs.jsonl`, not the summary).
    pub rows: Vec<Row>,
}

impl BenchReport {
    /// Summary as pretty JSON, without the per-run rows.
    pub fn summary_json(&self) -> Result<String, String> {
        summary_without_rows(self)
    }
}

/// A manifest run that stopped before it could report, with what it spent.
///
/// A run can stop part-way (no learning rate completed every seed, or
/// calibration has no matched pair). The runs already made still cost
/// compute and still explain the stop, so they travel with the error and
/// [`write_aborted_outputs`] writes them. A refusal made before any run
/// (an invalid manifest, a compute bound over the limit) carries no rows.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AbortedRun {
    /// Manifest echo.
    pub manifest: Manifest,
    /// Versions.
    pub versions: Versions,
    /// Why the run stopped.
    pub error: String,
    /// Planned, completed and failed runs at the stop.
    pub coverage: Coverage,
    /// Cost of the preparation phase, failed runs included.
    pub preparation_cost: PhaseCost,
    /// Cost of the calibration phase, failed runs included.
    pub calibration_cost: PhaseCost,
    /// Predeclared compute limit, planned bound and the passes actually spent.
    pub compute: ComputeBudget,
    /// Every run made before the stop (goes to `runs.jsonl`, not the summary).
    pub rows: Vec<Row>,
}

impl std::fmt::Display for AbortedRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.error)
    }
}

impl std::error::Error for AbortedRun {}

impl AbortedRun {
    fn new(manifest: &Manifest, planned: usize, bound: u64, rows: Vec<Row>, error: String) -> Self {
        let failed = rows.iter().filter(|r| matches!(r, Row::Failed(_))).count();
        Self {
            manifest: manifest.clone(),
            versions: versions(),
            error,
            coverage: Coverage {
                planned,
                completed: rows.len() - failed,
                failed,
            },
            preparation_cost: phase_cost(&rows, "preparation"),
            calibration_cost: phase_cost(&rows, "calibration"),
            compute: ComputeBudget {
                limit_record_passes: manifest.max_record_passes,
                planned_bound_record_passes: bound,
                actual_record_passes: total_record_passes(&rows),
                actual_audit_passes: total_audit_passes(&rows),
            },
            rows,
        }
    }

    /// Summary as pretty JSON, without the per-run rows.
    pub fn summary_json(&self) -> Result<String, String> {
        summary_without_rows(self)
    }
}

fn versions() -> Versions {
    Versions {
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        neat_core_tag: neat_core_tag(),
        git_commit: None,
    }
}

fn summary_without_rows<T: Serialize>(report: &T) -> Result<String, String> {
    let mut v = serde_json::to_value(report).map_err(|e| e.to_string())?;
    if let Some(o) = v.as_object_mut() {
        o.remove("rows");
    }
    serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
}

fn opt(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}

struct Runner<'a> {
    manifest: &'a Manifest,
    source: &'a CorpusSource,
    rows: Vec<Row>,
}

impl Runner<'_> {
    fn run_one(
        &mut self,
        phase: &str,
        seed: u64,
        policy: SkipPolicy,
        sched: &ScheduleConfig,
        learning_rate: f64,
    ) -> Option<RunOutcome> {
        let (corpus, parent) = match self.source {
            CorpusSource::Synthetic => (
                synthetic_corpus(
                    seed,
                    self.manifest.train_records,
                    self.manifest.validation_records,
                ),
                synthetic_parent(self.manifest.parent_seed),
            ),
            CorpusSource::Files { corpus, parent } => (corpus.clone(), parent.clone()),
        };
        let scope = DifficultyScope {
            parent_fingerprint: creature_fingerprint(&parent),
            corpus_key: corpus.key.clone(),
            record_count: corpus.train.len(),
        };
        let mut spent = Spend::default();
        let result = refine_inner(
            &parent,
            &corpus,
            &scope,
            &policy,
            sched,
            self.manifest,
            learning_rate,
            seed,
            &mut spent,
            phase == "validation",
        );
        match result {
            Ok(mut o) => {
                o.phase = phase.to_string();
                self.rows.push(Row::Ok(Box::new(o.clone())));
                Some(o)
            }
            Err(error) => {
                self.rows.push(Row::Failed(FailureRow {
                    arm: arm_label(&policy),
                    seed,
                    phase: phase.to_string(),
                    error,
                    record_passes: spent.record_passes,
                    audit_passes: spent.audit_passes,
                    revisit_passes: spent.revisit_passes,
                    schedule_ops: spent.schedule_ops,
                }));
                None
            }
        }
    }
}

fn schedule_for(m: &Manifest, g: &GridPoint) -> ScheduleConfig {
    ScheduleConfig {
        ewma_alpha: g.ewma_alpha,
        warmup_epochs: m.warmup_epochs,
        revisit_share: g.revisit_share,
        rare_stratum_share: m.rare_stratum_share,
    }
}

/// Index of the rate with the lowest mean training MSE among the rates that
/// completed every one of `seeds` runs. A rate scored on only the seeds that
/// survived is not comparable with one scored on all of them (each synthetic
/// seed is a different corpus), so it is never eligible. No input reaches a
/// partly completed rate today (the update clips, and a file corpus is shared
/// by every seed), so this guard is defensive and no test reaches that case.
fn choose_learning_rate(sweep: &[LearningRateEntry], seeds: usize) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, e) in sweep.iter().enumerate() {
        if e.n_runs != seeds {
            continue;
        }
        if let Some(d) = e.mean_train_mse
            && best.is_none_or(|b| d < sweep[b].mean_train_mse.unwrap_or(f64::INFINITY))
        {
            best = Some(i);
        }
    }
    best
}

/// Run preparation, calibration then validation per the manifest.
///
/// An error carries every run made before the stop and what they spent; see
/// [`AbortedRun`].
pub fn run_manifest(
    manifest: &Manifest,
    source: &CorpusSource,
) -> Result<BenchReport, Box<AbortedRun>> {
    manifest
        .validate()
        .map_err(|e| Box::new(AbortedRun::new(manifest, 0, 0, Vec::new(), e)))?;
    run_validated(manifest, source)
}

/// Number of runs the manifest plans, or `None` when the count overflows.
fn planned_runs(m: &Manifest) -> Option<usize> {
    let nred = m.reductions.len();
    let prep = m
        .calibration_seeds
        .len()
        .checked_mul(m.learning_rate_grid.len())?;
    let calibration = m
        .calibration_seeds
        .len()
        .checked_mul(nred)?
        .checked_mul(m.calibration_grid.len())?
        .checked_mul(2)?;
    let validation = m
        .validation_seeds
        .len()
        .checked_mul(nred.checked_mul(2)?.checked_add(1)?)?;
    prep.checked_add(calibration)?.checked_add(validation)
}

/// Run an already validated manifest. Split out so tests can drive the
/// run-failure path with values `Manifest::validate` refuses up front.
fn run_validated(
    manifest: &Manifest,
    source: &CorpusSource,
) -> Result<BenchReport, Box<AbortedRun>> {
    let train_count = match source {
        CorpusSource::Synthetic => manifest.train_records,
        CorpusSource::Files { corpus, .. } => corpus.train.len(),
    } as u64;
    let planned_count = planned_runs(manifest);
    let bound = planned_count.and_then(|p| {
        (p as u64)
            .checked_mul(u64::from(manifest.epochs))?
            .checked_mul(train_count)
    });
    let planned = planned_count.unwrap_or(usize::MAX);
    let Some(bound) = bound else {
        return Err(Box::new(AbortedRun::new(
            manifest,
            planned,
            u64::MAX,
            Vec::new(),
            format!(
                "planned compute bound overflows u64 record passes; exceeds maxRecordPasses {}",
                manifest.max_record_passes
            ),
        )));
    };
    if bound > manifest.max_record_passes {
        return Err(Box::new(AbortedRun::new(
            manifest,
            planned,
            bound,
            Vec::new(),
            format!(
                "planned compute bound {bound} record passes exceeds maxRecordPasses {}",
                manifest.max_record_passes
            ),
        )));
    }
    let mut runner = Runner {
        manifest,
        source,
        rows: Vec::new(),
    };
    let first_failure = |rows: &[Row]| {
        rows.iter()
            .find_map(|r| match r {
                Row::Failed(f) => Some(format!(
                    "; first failure: {} seed {} {}: {}",
                    f.arm, f.seed, f.phase, f.error
                )),
                Row::Ok(_) => None,
            })
            .unwrap_or_default()
    };

    // Phase 0: learning-rate preparation on training MSE only.
    let prep_sched = schedule_for(manifest, &manifest.calibration_grid[0]);
    let mut sweep = Vec::new();
    for &lr in &manifest.learning_rate_grid {
        let mut finals = Vec::new();
        for &seed in &manifest.calibration_seeds {
            if let Some(o) = runner.run_one("preparation", seed, SkipPolicy::Full, &prep_sched, lr)
            {
                finals.push(o.final_train_mse);
            }
        }
        sweep.push(LearningRateEntry {
            learning_rate: lr,
            n_runs: finals.len(),
            mean_train_mse: opt(mean(&finals)),
        });
    }
    let Some(best_lr) = choose_learning_rate(&sweep, manifest.calibration_seeds.len()) else {
        let error = format!(
            "preparation: no learning rate completed all {} calibration seed(s); cannot choose a learning rate{}",
            manifest.calibration_seeds.len(),
            first_failure(&runner.rows)
        );
        return Err(Box::new(AbortedRun::new(
            manifest,
            planned,
            bound,
            runner.rows,
            error,
        )));
    };
    let learning_rate = sweep[best_lr].learning_rate;

    // Phase 1: calibration on training MSE only.
    let mut calibration = Vec::new();
    for g in &manifest.calibration_grid {
        let sched = schedule_for(manifest, g);
        let mut diffs = Vec::new();
        for &seed in &manifest.calibration_seeds {
            for &r in &manifest.reductions {
                let rnd = runner.run_one(
                    "calibration",
                    seed,
                    SkipPolicy::RandomSkip { reduction: r },
                    &sched,
                    learning_rate,
                );
                let easy = runner.run_one(
                    "calibration",
                    seed,
                    SkipPolicy::EasySkip { reduction: r },
                    &sched,
                    learning_rate,
                );
                if let (Some(rnd), Some(easy)) = (rnd, easy) {
                    diffs.push(easy.final_train_mse - rnd.final_train_mse);
                }
            }
        }
        calibration.push(CalibrationEntry {
            ewma_alpha: g.ewma_alpha,
            revisit_share: g.revisit_share,
            n_pairs: diffs.len(),
            mean_train_diff: opt(mean(&diffs)),
        });
    }
    let mut best: Option<usize> = None;
    for (i, c) in calibration.iter().enumerate() {
        if let Some(d) = c.mean_train_diff
            && best.is_none_or(|b| d < calibration[b].mean_train_diff.unwrap_or(f64::INFINITY))
        {
            best = Some(i);
        }
    }
    let Some(best) = best else {
        let error = format!(
            "calibration produced no matched pairs; cannot choose a schedule{}",
            first_failure(&runner.rows)
        );
        return Err(Box::new(AbortedRun::new(
            manifest,
            planned,
            bound,
            runner.rows,
            error,
        )));
    };
    let chosen = manifest.calibration_grid[best];
    let sched = schedule_for(manifest, &chosen);

    // Phase 2: validation with the chosen point.
    let mut done: Vec<RunOutcome> = Vec::new();
    for &seed in &manifest.validation_seeds {
        let mut policies = vec![SkipPolicy::Full];
        for &r in &manifest.reductions {
            policies.push(SkipPolicy::RandomSkip { reduction: r });
            policies.push(SkipPolicy::EasySkip { reduction: r });
        }
        for p in policies {
            if let Some(o) = runner.run_one("validation", seed, p, &sched, learning_rate) {
                done.push(o);
            }
        }
    }

    let mut arm_labels = vec!["full".to_string()];
    for &r in &manifest.reductions {
        arm_labels.push(arm_label(&SkipPolicy::RandomSkip { reduction: r }));
        arm_labels.push(arm_label(&SkipPolicy::EasySkip { reduction: r }));
    }
    let arms: Vec<ArmSummary> = arm_labels
        .into_iter()
        .map(|label| {
            let runs: Vec<&RunOutcome> = done.iter().filter(|o| o.arm == label).collect();
            arm_summary(label, &runs)
        })
        .collect();

    let mut comparisons = Vec::new();
    for &r in &manifest.reductions {
        let easy = arm_label(&SkipPolicy::EasySkip { reduction: r });
        let rnd = arm_label(&SkipPolicy::RandomSkip { reduction: r });
        comparisons.push(compare(manifest, &done, &easy, &rnd, "primary"));
        comparisons.push(compare(manifest, &done, &easy, "full", "secondary"));
        comparisons.push(compare(manifest, &done, &rnd, "full", "secondary"));
    }

    let failed = runner
        .rows
        .iter()
        .filter(|r| matches!(r, Row::Failed(_)))
        .count();
    let rows = runner.rows;
    let source_desc = match source {
        CorpusSource::Synthetic => "synthetic".to_string(),
        CorpusSource::Files { corpus, parent } => format!(
            "files: corpus {} parent {:016x}",
            corpus.key,
            creature_fingerprint(parent)
        ),
    };
    Ok(BenchReport {
        manifest: manifest.clone(),
        versions: versions(),
        source: source_desc,
        chosen_learning_rate: learning_rate,
        learning_rate_sweep: sweep,
        chosen_schedule: ChosenSchedule {
            ewma_alpha: chosen.ewma_alpha,
            revisit_share: chosen.revisit_share,
        },
        calibration,
        coverage: Coverage {
            planned,
            completed: rows.len() - failed,
            failed,
        },
        comparisons,
        preparation_cost: phase_cost(&rows, "preparation"),
        compute: ComputeBudget {
            limit_record_passes: manifest.max_record_passes,
            planned_bound_record_passes: bound,
            actual_record_passes: total_record_passes(&rows),
            actual_audit_passes: total_audit_passes(&rows),
        },
        arms,
        calibration_cost: phase_cost(&rows, "calibration"),
        validation_cost: phase_cost(&rows, "validation"),
        rows,
    })
}

fn compare(
    m: &Manifest,
    done: &[RunOutcome],
    treatment: &str,
    control: &str,
    role: &str,
) -> Comparison {
    let find = |arm: &str, seed: u64| done.iter().find(|o| o.arm == arm && o.seed == seed);
    let mut seeds = Vec::new();
    let (mut diffs, mut rare, mut train) = (Vec::new(), Vec::new(), Vec::new());
    let (mut tp, mut cp, mut tf, mut cf) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &seed in &m.validation_seeds {
        let (Some(t), Some(c)) = (find(treatment, seed), find(control, seed)) else {
            continue;
        };
        let (Some(tv), Some(cv)) = (t.final_validation_mse, c.final_validation_mse) else {
            continue;
        };
        seeds.push(seed);
        diffs.push(tv - cv);
        train.push(t.final_train_mse - c.final_train_mse);
        if let (Some(a), Some(b)) = (t.rare_stratum_validation_mse, c.rare_stratum_validation_mse) {
            rare.push(a - b);
        }
        tp.push(t.record_passes as f64);
        cp.push(c.record_passes as f64);
        tf.push(t.forgotten_share);
        cf.push(c.forgotten_share);
    }
    let ci = bootstrap_mean_ci(&diffs, m.bootstrap_resamples, m.bootstrap_seed);
    Comparison {
        name: format!("{treatment} vs {control}"),
        role: role.to_string(),
        treatment: treatment.to_string(),
        control: control.to_string(),
        n_pairs: seeds.len(),
        seeds,
        mean_validation_diff: opt(mean(&diffs)),
        ci_low: opt(ci.0),
        ci_high: opt(ci.1),
        rare_mean_diff: opt(mean(&rare)),
        train_mean_diff: opt(mean(&train)),
        treatment_record_passes: opt(mean(&tp)),
        control_record_passes: opt(mean(&cp)),
        treatment_forgotten_share: opt(mean(&tf)),
        control_forgotten_share: opt(mean(&cf)),
        verdict: verdict(ci, &rare),
        validation_diffs: diffs,
    }
}

/// Write `manifest.json`, `runs.jsonl` and `summary.json` into `dir`.
pub fn write_outputs(dir: &Path, report: &BenchReport) -> Result<(), String> {
    write_files(dir, &report.manifest, &report.rows, report.summary_json()?)
}

/// Write the same three files for a run that stopped early. `summary.json`
/// holds the error, the coverage and the cost of the runs that were made;
/// `runs.jsonl` holds their rows, failed ones with the passes they spent.
pub fn write_aborted_outputs(dir: &Path, aborted: &AbortedRun) -> Result<(), String> {
    write_files(
        dir,
        &aborted.manifest,
        &aborted.rows,
        aborted.summary_json()?,
    )
}

fn write_files(
    dir: &Path,
    manifest: &Manifest,
    rows: &[Row],
    summary: String,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let write = |name: &str, body: String| {
        std::fs::write(dir.join(name), body).map_err(|e| format!("write {name}: {e}"))
    };
    write(
        "manifest.json",
        serde_json::to_string_pretty(manifest).map_err(|e| e.to_string())?,
    )?;
    let mut lines = String::new();
    for row in rows {
        lines.push_str(&serde_json::to_string(row).map_err(|e| e.to_string())?);
        lines.push('\n');
    }
    write("runs.jsonl", lines)?;
    write("summary.json", summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Manifest {
        Manifest {
            train_records: 100,
            validation_records: 50,
            reductions: vec![0.5],
            calibration_seeds: vec![1],
            validation_seeds: vec![1001, 1002],
            ..Manifest::smoke()
        }
    }

    fn failed(rows: &[Row]) -> Vec<&FailureRow> {
        rows.iter()
            .filter_map(|r| match r {
                Row::Failed(f) => Some(f),
                Row::Ok(_) => None,
            })
            .collect()
    }

    #[test]
    fn refused_grid_point_is_reported_beside_a_valid_one() {
        let mut m = tiny();
        // The valid point comes first: preparation runs the full arm on it.
        m.calibration_grid = vec![
            GridPoint {
                ewma_alpha: 0.5,
                revisit_share: 0.1,
            },
            GridPoint {
                ewma_alpha: 0.0,
                revisit_share: 0.1,
            },
        ];
        // Control: the public entry point refuses the bad grid point up front.
        assert!(m.validate().unwrap_err().contains("ewmaAlpha"));

        let report = run_validated(&m, &CorpusSource::Synthetic).unwrap();
        // 4 rates * 1 seed + 1 seed * 1 reduction * 2 grid * 2 arms + 2 seeds * 3 arms.
        assert_eq!(report.coverage.planned, 14);
        assert_eq!(report.coverage.failed, 2);
        assert_eq!(report.coverage.completed, 12);

        let fails = failed(&report.rows);
        assert_eq!(fails.len(), 2);
        let arms: Vec<&str> = fails.iter().map(|f| f.arm.as_str()).collect();
        assert_eq!(arms, ["random-50", "easy-50"]);
        for f in &fails {
            assert_eq!((f.seed, f.phase.as_str()), (1, "calibration"));
            assert!(f.error.contains("ewma_alpha"), "{}", f.error);
        }

        assert_eq!(report.calibration[1].n_pairs, 0);
        assert!(report.calibration[1].mean_train_diff.is_none());
        assert_eq!(report.calibration[0].n_pairs, 1);
        assert!(report.calibration[0].mean_train_diff.is_some());
        assert_eq!(report.chosen_schedule.ewma_alpha, 0.5);

        let dir = tempfile::tempdir().unwrap();
        write_outputs(dir.path(), &report).unwrap();
        let runs = std::fs::read_to_string(dir.path().join("runs.jsonl")).unwrap();
        assert_eq!(runs.lines().count(), 14);
        let error_lines = runs.lines().filter(|l| l.contains("\"error\"")).count();
        assert_eq!(error_lines, 2);
    }

    #[test]
    fn schedule_choice_skips_a_refused_point_and_takes_the_lowest_diff() {
        let mut m = tiny();
        // The refused point sits between two valid ones. The third point must
        // win on calibration diff, so a selector that always takes index 0
        // fails here.
        m.calibration_grid = vec![
            GridPoint {
                ewma_alpha: 0.5,
                revisit_share: 0.1,
            },
            GridPoint {
                ewma_alpha: 0.0,
                revisit_share: 0.1,
            },
            GridPoint {
                ewma_alpha: 0.8,
                revisit_share: 0.3,
            },
        ];

        let report = run_validated(&m, &CorpusSource::Synthetic).unwrap();
        assert!(report.calibration[1].mean_train_diff.is_none());
        // Premise: the third point beats the first on calibration diff.
        assert!(
            report.calibration[2].mean_train_diff.unwrap()
                < report.calibration[0].mean_train_diff.unwrap()
        );
        assert_eq!(report.chosen_schedule.ewma_alpha, 0.8);
        assert_eq!(report.chosen_schedule.revisit_share, 0.3);
    }

    #[test]
    fn all_failed_preparation_names_the_first_failure() {
        let mut m = tiny();
        m.warmup_epochs = 0;
        assert!(m.validate().unwrap_err().contains("warmupEpochs"));

        let err = run_validated(&m, &CorpusSource::Synthetic)
            .unwrap_err()
            .error;
        assert!(err.contains("no learning rate completed"), "{err}");
        assert!(
            err.contains("first failure: full seed 1 preparation"),
            "{err}"
        );
        assert!(err.contains("warmup_epochs"), "{err}");
    }

    #[test]
    fn all_failed_calibration_names_the_first_failure() {
        let mut m = tiny();
        // Full runs validate any reduction, so preparation passes; every
        // random and easy calibration run refuses it in `select`.
        m.reductions = vec![1.5];
        assert!(m.validate().unwrap_err().contains("reduction"));

        let aborted = run_validated(&m, &CorpusSource::Synthetic).unwrap_err();
        let err = &aborted.error;
        assert!(err.contains("no matched pairs"), "{err}");
        assert!(
            err.contains("first failure: random-150 seed 1 calibration"),
            "{err}"
        );
        assert!(err.contains("invalid reduction"), "{err}");
    }

    fn nan_training_source(m: &Manifest) -> CorpusSource {
        // A NaN training target makes every run diverge, but only in the
        // audit after all epochs ran. (A NaN or infinite training input does
        // not: the network sanitises inputs, so the error stays finite. The
        // strata edges skip a NaN target, so they are unaffected.)
        let mut corpus = synthetic_corpus(1, m.train_records, m.validation_records);
        corpus.train[0].outputs[0] = f32::NAN;
        CorpusSource::Files {
            corpus,
            parent: synthetic_parent(m.parent_seed),
        }
    }

    fn full_passes(m: &Manifest) -> u64 {
        u64::from(m.epochs) * m.train_records as u64
    }

    #[test]
    fn an_aborted_preparation_keeps_the_passes_it_spent() {
        let m = tiny();
        m.validate().unwrap();
        let source = nan_training_source(&m);

        // Through the public entry point: every rate diverges.
        let aborted = run_manifest(&m, &source).unwrap_err();
        assert!(
            aborted.error.contains("no learning rate completed"),
            "{aborted}"
        );
        assert!(aborted.error.contains("diverged"), "{aborted}");
        assert!(aborted.error.contains("training"), "{aborted}");
        let rates = m.learning_rate_grid.len() as u64 * m.calibration_seeds.len() as u64;
        assert_eq!(aborted.rows.len() as u64, rates);
        let fails = failed(&aborted.rows);
        assert_eq!(fails.len() as u64, rates);
        for f in &fails {
            assert!(f.error.contains("diverged"), "{}", f.error);
            assert!(f.error.contains("training"), "{}", f.error);
            assert_eq!(f.record_passes, full_passes(&m));
            assert_eq!(f.schedule_ops, 0);
            // Initial and final training audits; no held-out read in preparation.
            assert_eq!(f.audit_passes, 2 * m.train_records as u64);
            assert_eq!(f.phase, "preparation");
        }
        assert_eq!(
            aborted.preparation_cost.audit_passes,
            rates * 2 * m.train_records as u64
        );
        assert_eq!(aborted.coverage.planned, 14);
        assert_eq!(
            (aborted.coverage.completed, aborted.coverage.failed),
            (0, rates as usize)
        );
        assert_eq!(aborted.preparation_cost.runs, 0);
        assert_eq!(
            aborted.preparation_cost.record_passes,
            rates * full_passes(&m)
        );
        assert_eq!(aborted.calibration_cost, PhaseCost::default());
        assert_eq!(
            aborted.compute.actual_audit_passes,
            rates * 2 * m.train_records as u64
        );
        assert_eq!(
            aborted.compute.actual_record_passes,
            rates * full_passes(&m)
        );
        assert_eq!(
            aborted.compute.planned_bound_record_passes,
            14 * full_passes(&m)
        );

        // The outputs carry the cost and the failed rows' passes.
        let dir = tempfile::tempdir().unwrap();
        write_aborted_outputs(dir.path(), &aborted).unwrap();
        let summary: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            summary["compute"]["actualRecordPasses"],
            rates * full_passes(&m)
        );
        assert_eq!(
            summary["preparationCost"]["recordPasses"],
            rates * full_passes(&m)
        );
        assert!(
            summary["error"]
                .as_str()
                .unwrap()
                .contains("no learning rate")
        );
        assert!(summary.get("rows").is_none());
        assert!(dir.path().join("manifest.json").exists());
        let runs = std::fs::read_to_string(dir.path().join("runs.jsonl")).unwrap();
        assert_eq!(runs.lines().count() as u64, rates);
        for line in runs.lines() {
            assert!(
                line.contains(&format!("\"recordPasses\":{}", full_passes(&m))),
                "{line}"
            );
            assert!(
                line.contains(&format!("\"auditPasses\":{}", 2 * m.train_records)),
                "{line}"
            );
        }
    }

    #[test]
    fn an_aborted_calibration_keeps_the_preparation_cost() {
        let mut m = tiny();
        m.reductions = vec![1.5];
        let aborted = run_validated(&m, &CorpusSource::Synthetic).unwrap_err();
        assert!(aborted.error.contains("no matched pairs"), "{aborted}");
        // Preparation completed 4 runs; calibration failed 2 grid points x 2 arms.
        assert_eq!(aborted.preparation_cost.runs, 4);
        assert_eq!(aborted.preparation_cost.record_passes, 4 * full_passes(&m));
        assert_eq!(aborted.calibration_cost.runs, 0);
        assert_eq!(
            (aborted.coverage.completed, aborted.coverage.failed),
            (4, 4)
        );
        assert_eq!(aborted.compute.actual_record_passes, 4 * full_passes(&m));
        assert_eq!(aborted.rows.len(), 8);
    }

    #[test]
    fn a_compute_bound_that_overflows_is_refused_before_any_run() {
        let mut m = tiny();
        m.epochs = 1 << 31;
        m.train_records = 1 << 32;
        assert!(m.validate().is_ok(), "premise: the manifest is accepted");

        let aborted = run_manifest(&m, &CorpusSource::Synthetic).unwrap_err();
        assert!(aborted.error.contains("overflows"), "{}", aborted.error);
        assert!(
            aborted.error.contains("maxRecordPasses"),
            "{}",
            aborted.error
        );
        assert_eq!(aborted.compute.planned_bound_record_passes, u64::MAX);
        assert!(aborted.rows.is_empty());
    }

    #[test]
    fn a_representable_bound_over_the_limit_is_still_refused_exactly() {
        let mut m = tiny();
        m.max_record_passes = 1000;
        assert!(m.validate().is_ok(), "premise: the manifest is accepted");
        let aborted = run_manifest(&m, &CorpusSource::Synthetic).unwrap_err();
        let bound = 14 * full_passes(&m);
        assert!(
            aborted.error.contains("exceeds maxRecordPasses"),
            "{aborted}"
        );
        assert_eq!(aborted.compute.planned_bound_record_passes, bound);
        assert!(aborted.error.contains(&bound.to_string()), "{aborted}");
        assert!(aborted.rows.is_empty());
    }

    fn files_source(m: &Manifest, nan_held_out: bool) -> CorpusSource {
        let mut corpus = synthetic_corpus(1, m.train_records, m.validation_records);
        if nan_held_out {
            corpus.validation[0].outputs[0] = f32::NAN;
        }
        CorpusSource::Files {
            corpus,
            parent: synthetic_parent(m.parent_seed),
        }
    }

    #[test]
    fn a_held_out_record_cannot_steer_preparation_or_calibration() {
        let m = tiny();
        let clean = run_manifest(&m, &files_source(&m, false)).unwrap();
        let dirty = run_manifest(&m, &files_source(&m, true)).unwrap();

        // Positive control: the clean run fails nothing.
        assert!(failed(&clean.rows).is_empty());

        assert_eq!(clean.chosen_learning_rate, dirty.chosen_learning_rate);
        assert_eq!(
            clean.chosen_schedule.ewma_alpha,
            dirty.chosen_schedule.ewma_alpha
        );
        assert_eq!(
            clean.chosen_schedule.revisit_share,
            dirty.chosen_schedule.revisit_share
        );
        assert_eq!(
            clean.learning_rate_sweep.len(),
            dirty.learning_rate_sweep.len()
        );
        for (a, b) in clean
            .learning_rate_sweep
            .iter()
            .zip(&dirty.learning_rate_sweep)
        {
            assert_eq!(a.learning_rate, b.learning_rate);
            assert_eq!(a.n_runs, b.n_runs);
            assert_eq!(a.mean_train_mse, b.mean_train_mse);
        }
        assert_eq!(clean.calibration.len(), dirty.calibration.len());
        for (a, b) in clean.calibration.iter().zip(&dirty.calibration) {
            assert_eq!(a.ewma_alpha, b.ewma_alpha);
            assert_eq!(a.revisit_share, b.revisit_share);
            assert_eq!(a.n_pairs, b.n_pairs);
            assert_eq!(a.mean_train_diff, b.mean_train_diff);
        }

        let mut validation_rows = 0;
        for row in &dirty.rows {
            match row {
                Row::Ok(o) => assert_ne!(o.phase, "validation", "{o:?}"),
                Row::Failed(f) if f.phase == "validation" => {
                    validation_rows += 1;
                    assert!(f.error.contains("validation"), "{}", f.error);
                }
                Row::Failed(f) => panic!("{} run failed on a held-out NaN: {}", f.phase, f.error),
            }
        }
        assert_eq!(validation_rows, m.validation_seeds.len() * 3);
    }
}
