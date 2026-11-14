//! Refinement driver and manifest runner for the record-skip experiment.
//!
//! Every arm refines a clone of the same immutable parent through the real
//! refinement update path. Audit measurements (full training MSE, validation
//! MSE) are taken after the loop and are not charged to the arm.

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

/// neat-core tag this crate is built against. Must match `lamarck/Cargo.toml`
/// (asserted by a test).
pub const NEAT_CORE_TAG: &str = "v0.23.4";

/// Result of one refinement run.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunOutcome {
    /// Arm label, for example `easy-50`.
    pub arm: String,
    /// `calibration` or `validation` (set by the runner).
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
    /// Held-out validation MSE.
    pub final_validation_mse: f64,
    /// Validation MSE over records in protected (rare) strata.
    pub rare_stratum_validation_mse: Option<f64>,
    /// Validation MSE per stratum.
    pub strata_validation_mse: Vec<Option<f64>>,
    /// Forward and backward record passes charged to the arm.
    pub record_passes: u64,
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

/// Refine a clone of `parent` over `corpus.train` under `policy`.
pub fn refine(
    parent: &CreatureExport,
    corpus: &Corpus,
    scope: &DifficultyScope,
    policy: &SkipPolicy,
    schedule: &ScheduleConfig,
    manifest: &Manifest,
    rng_seed: u64,
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
        learning_rate: manifest.learning_rate,
        initial_learning_rate: manifest.learning_rate,
        ..BackpropConfig::default()
    };

    let initial = mean(&record_errors(parent, &corpus.train)?);
    let mut creature = parent.clone();
    let mut sched_rng = StdRng::seed_from_u64(rng_seed);
    let mut plan_rng = StdRng::seed_from_u64(rng_seed ^ 0x9E37_79B9_7F4A_7C15);
    let (mut passes, mut revisit_passes, mut ops) = (0u64, 0u64, 0u64);

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
        passes += sel.processed as u64;
        revisit_passes += sel.revisits as u64;
        ops += sel.schedule_ops;
    }

    // Audit: measurement only, not charged to the arm.
    let train_errors = record_errors(&creature, &corpus.train)?;
    let val_errors = record_errors(&creature, &corpus.validation)?;
    let final_train = mean(&train_errors);
    let final_val = mean(&val_errors);
    if !final_train.is_finite() || !final_val.is_finite() {
        return Err("refinement diverged: non-finite MSE".to_string());
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
        record_passes: passes,
        processed_share: passes as f64 / (f64::from(manifest.epochs) * n as f64),
        revisit_passes,
        schedule_ops: ops,
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

/// Summed cost of one phase, counting every run (including failed ones in
/// `runs` only through completed rows).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseCost {
    /// Completed runs in the phase.
    pub runs: usize,
    /// Forward and backward record passes.
    pub record_passes: u64,
    /// Passes spent on easy-pool revisits.
    pub revisit_passes: u64,
    /// Scheduling operations.
    pub schedule_ops: u64,
}

fn phase_cost(rows: &[Row], phase: &str) -> PhaseCost {
    let mut c = PhaseCost::default();
    for row in rows {
        if let Row::Ok(o) = row
            && o.phase == phase
        {
            c.runs += 1;
            c.record_passes += o.record_passes;
            c.revisit_passes += o.revisit_passes;
            c.schedule_ops += o.schedule_ops;
        }
    }
    c
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
    /// Schedule chosen by calibration.
    pub chosen_schedule: ChosenSchedule,
    /// Calibration table.
    pub calibration: Vec<CalibrationEntry>,
    /// Coverage.
    pub coverage: Coverage,
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
        let mut v = serde_json::to_value(self).map_err(|e| e.to_string())?;
        if let Some(o) = v.as_object_mut() {
            o.remove("rows");
        }
        serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
    }
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
        match refine(
            &parent,
            &corpus,
            &scope,
            &policy,
            sched,
            self.manifest,
            seed,
        ) {
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

/// Run calibration then validation per the manifest.
pub fn run_manifest(manifest: &Manifest, source: &CorpusSource) -> Result<BenchReport, String> {
    manifest.validate()?;
    let mut runner = Runner {
        manifest,
        source,
        rows: Vec::new(),
    };
    let nred = manifest.reductions.len();
    let planned = manifest.calibration_seeds.len() * nred * manifest.calibration_grid.len() * 2
        + manifest.validation_seeds.len() * (1 + 2 * nred);

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
                );
                let easy = runner.run_one(
                    "calibration",
                    seed,
                    SkipPolicy::EasySkip { reduction: r },
                    &sched,
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
    let best = best.ok_or("calibration produced no matched pairs; cannot choose a schedule")?;
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
            if let Some(o) = runner.run_one("validation", seed, p, &sched) {
                done.push(o);
            }
        }
    }

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
        versions: Versions {
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            neat_core_tag: NEAT_CORE_TAG.to_string(),
        },
        source: source_desc,
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
        seeds.push(seed);
        diffs.push(t.final_validation_mse - c.final_validation_mse);
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
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let write = |name: &str, body: String| {
        std::fs::write(dir.join(name), body).map_err(|e| format!("write {name}: {e}"))
    };
    write(
        "manifest.json",
        serde_json::to_string_pretty(&report.manifest).map_err(|e| e.to_string())?,
    )?;
    let mut lines = String::new();
    for row in &report.rows {
        lines.push_str(&serde_json::to_string(row).map_err(|e| e.to_string())?);
        lines.push('\n');
    }
    write("runs.jsonl", lines)?;
    write("summary.json", report.summary_json()?)
}
