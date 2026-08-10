//! End-to-end Lamarck optimisation loop and experiment journal.

use crate::backprop::BackpropConfig;
use crate::candidates::{
    Candidate, CandidateGenContext, CandidateProvenance, CandidateStrategy, generate_candidates,
    write_candidate_batch,
};
use crate::combos::{
    ComboSelectRequest, ComboSelection, StackDampenReport, select_best_with_combinations,
};
use crate::config::LamarckConfig;
use crate::focus::{
    FixedFocusSelector, FocusChoice, FocusNeuronStats, FocusPolicy, FocusSelector,
    HighErrorFocusSelector, RandomFocusSelector, UnsaturatedFocusSelector, WeightedFocusSelector,
    attach_focus_blame, attach_learning_to_incoming, build_improvement_signals,
    collect_focus_stats, collect_incoming_source_stats, collect_output_mean_abs_errors,
    select_highest_signal,
};
use crate::grafts::{
    GraftReplayRequest, GraftStore, default_graft_replay_budget, record_structural_acceptance,
    replay_grafts,
};
use crate::log;
use crate::observations::ensure_statistics;
use crate::parity::{check_phase0_parity, compute_local_mse};
use crate::propagate_layout::accumulate_creature_learning;
use crate::scorer::improvement;
use crate::scorer::{
    DirectoryScorer, ScoreResult, ScoreSample, accepts_improvement, log_scorer_batch_stats_labeled,
    screen_promote_stems, write_promote_batch,
};
use crate::structural::{
    is_input_source, rank_unused_sources, refine_sources_by_residual_with_observations,
};
use crate::tags::{CreatureMeta, LamarckProgress, serialize_creature_with_meta};
use neat_core::{
    TrainingDataConfig, compile_creature, creature_to_json_pretty, parse_creature_json,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// One journal line written to `experiments.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentRecord {
    /// Monotonic experiment number.
    pub experiment_number: u64,
    /// Unix timestamp.
    pub timestamp_unix: u64,
    /// RNG seed for this run (if any).
    pub seed: Option<u64>,
    /// Incumbent creature checksum (uuid or hash).
    pub incumbent_id: String,
    /// Baseline authoritative score.
    pub baseline_score: f64,
    /// Focus neuron UUID.
    pub focus_neuron: String,
    /// Focus-neuron structure, activation statistics and backprop blame (#70).
    ///
    /// Omitted only when the focus scan did not run for this experiment (and on
    /// journals written before the field existed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus_stats: Option<FocusNeuronStats>,
    /// Candidate provenances.
    pub candidates: Vec<CandidateProvenance>,
    /// Authoritative (full-corpus) scores by stem when a promote/full score ran.
    pub scores: std::collections::BTreeMap<String, f64>,
    /// Screen-phase (subsample) scores by stem when two-phase scoring is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screen_scores: Option<std::collections::BTreeMap<String, f64>>,
    /// Winning stem if accepted.
    pub winner: Option<String>,
    /// Absolute score improvement when accepted.
    pub improvement: Option<f64>,
    /// Whether a candidate was accepted.
    pub accepted: bool,
    /// Analysis elapsed milliseconds.
    pub analysis_ms: u128,
    /// Scorer elapsed milliseconds (screen + promote when both ran).
    pub scorer_ms: u128,
    /// Scorer error message when the batch failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scorer_error: Option<String>,
    /// Member count of the selected combo (`None` / omitted for pure singles).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combo_members: Option<usize>,
    /// Combination creatures scored during selection (for dampen tuning).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combos_scored: Option<usize>,
    /// How many scored combos applied stacked-synapse dampening.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combos_dampened: Option<usize>,
    /// Per-target dampen detail for the accepted winner (when any).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combo_dampen: Option<StackDampenReport>,
}

/// Marks a journal line as the one-off run header (issue #71).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RunHeaderKind {
    /// This line is a run header, not an experiment.
    RunHeader,
}

/// Where the effective RNG seed came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SeedSource {
    /// The seed was supplied by the caller (`--seed`).
    Supplied,
    /// The seed was drawn from OS entropy and recorded for replay.
    Drawn,
}

/// Run knobs recorded in the journal header so a run can be replayed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunConfigRecord {
    /// Supplied incumbent creature path.
    pub creature: PathBuf,
    /// Training-data directory.
    pub training_data: PathBuf,
    /// Scorer binary path.
    pub scorer_path: PathBuf,
    /// Wall-clock budget in seconds.
    pub timeout_seconds: u64,
    /// Candidates generated per experiment.
    pub candidates: usize,
    /// Absolute score delta required for acceptance.
    pub min_improvement: f64,
    /// Screen subsample rate (`None` = full-corpus scoring only).
    pub screen_sample_rate: Option<f64>,
    /// Minimum sample-score delta to promote to full-corpus scoring.
    pub screen_promote_threshold: f64,
    /// Pinned focus neuron UUID when set.
    pub focus_neuron: Option<String>,
    /// Focus selection policy label.
    pub focus_policy: String,
    /// Observations mode label (`full` / `quick`).
    pub stats_mode: String,
    /// Record cap for quick-mode analysis sampling.
    pub quick_sample_records: u64,
    /// Whether input×input correlations were computed.
    pub compute_correlations: bool,
    /// Whether only structural growth candidates were generated.
    pub structural_only: bool,
    /// Whether the Phase-0 parity gate ran.
    pub phase0_parity: bool,
    /// Whether rejected candidate directories were kept.
    pub preserve_losers: bool,
    /// Consecutive scorer failures tolerated before aborting.
    pub max_consecutive_scorer_failures: u32,
    /// Structural graft store path when phase-G was enabled.
    pub grafts_path: Option<PathBuf>,
    /// Explicit phase-G replay budget in seconds when set.
    pub graft_replay_budget_seconds: Option<u64>,
}

impl RunConfigRecord {
    /// Capture the reproducible knobs of a [`LamarckConfig`].
    pub fn from_config(config: &LamarckConfig) -> Self {
        Self {
            creature: config.creature.clone(),
            training_data: config.training_data.clone(),
            scorer_path: config.scorer_path.clone(),
            timeout_seconds: config.timeout.as_secs(),
            candidates: config.candidates,
            min_improvement: config.min_improvement,
            screen_sample_rate: config.screen_sample_rate,
            screen_promote_threshold: config.screen_promote_threshold,
            focus_neuron: config.focus_neuron.clone(),
            focus_policy: config.focus_policy.label().to_string(),
            stats_mode: config.stats_mode.label().to_string(),
            quick_sample_records: config.quick_sample_records,
            compute_correlations: config.compute_correlations,
            structural_only: config.structural_only,
            phase0_parity: config.phase0_parity,
            preserve_losers: config.preserve_losers,
            max_consecutive_scorer_failures: config.max_consecutive_scorer_failures,
            grafts_path: config.grafts_path.clone(),
            graft_replay_budget_seconds: config.graft_replay_budget.map(|d| d.as_secs()),
        }
    }
}

/// One-off header line written to `experiments.jsonl` at the start of a run.
///
/// The header pins the reproducibility contract: the effective seed (drawn from
/// OS entropy when `--seed` was omitted) plus the run configuration. Replaying
/// with the recorded seed reproduces the RNG stream; because the experiment
/// count is wall-clock bounded and the screen phase is derived from the
/// experiment index, a differently timed replay may still run a different number
/// of experiments.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunHeaderRecord {
    /// Discriminates this line from an [`ExperimentRecord`].
    pub record: RunHeaderKind,
    /// Unix timestamp at run start.
    pub timestamp_unix: u64,
    /// Effective RNG seed for the run.
    pub seed: u64,
    /// Whether the seed was supplied or drawn.
    pub seed_source: SeedSource,
    /// Lamarck version that wrote the journal.
    pub version: String,
    /// Run configuration knobs.
    pub config: RunConfigRecord,
}

impl RunHeaderRecord {
    /// Build a header for the effective seed and captured configuration.
    pub fn new(
        seed: u64,
        seed_source: SeedSource,
        config: RunConfigRecord,
        timestamp_unix: u64,
    ) -> Self {
        Self {
            record: RunHeaderKind::RunHeader,
            timestamp_unix,
            seed,
            seed_source,
            version: env!("CARGO_PKG_VERSION").to_string(),
            config,
        }
    }
}

/// One line of `experiments.jsonl`: the run header or an experiment record.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum JournalLine {
    /// Run header written once at the start of a run.
    Header(Box<RunHeaderRecord>),
    /// One experiment outcome.
    Experiment(Box<ExperimentRecord>),
}

impl JournalLine {
    /// Parse one journal line, dispatching on the `record` discriminator.
    ///
    /// A line that is neither a valid header nor a valid experiment is an
    /// error — a malformed journal must never be read as an empty run.
    pub fn parse(line: &str) -> Result<Self, String> {
        #[derive(Deserialize)]
        struct Probe {
            #[serde(default)]
            record: Option<RunHeaderKind>,
        }
        let probe: Probe = serde_json::from_str(line).map_err(|e| e.to_string())?;
        if probe.record.is_some() {
            let header = serde_json::from_str(line).map_err(|e| e.to_string())?;
            Ok(Self::Header(Box::new(header)))
        } else {
            let record = serde_json::from_str(line).map_err(|e| e.to_string())?;
            Ok(Self::Experiment(Box::new(record)))
        }
    }
}

/// Result of a completed Lamarck run.
#[derive(Debug)]
pub struct RunResult {
    /// Path to `best.json`.
    pub best_path: PathBuf,
    /// Path to `experiments.jsonl`.
    pub journal_path: PathBuf,
    /// Final best score.
    pub best_score: f64,
    /// Experiments attempted.
    pub experiments: u64,
    /// Accepted improvements.
    pub acceptances: u64,
    /// Scorer batch failures.
    pub scorer_failures: u64,
    /// Successful scorer batches.
    pub scorer_successes: u64,
    /// Opening Phase-0 baseline score when recorded.
    pub opening_baseline_score: Option<f64>,
    /// Effective RNG seed used by the run (drawn when `--seed` was omitted).
    pub seed: u64,
}

/// Run the Lamarck optimisation loop until the wall-clock budget expires.
pub fn run_optimisation(
    config: &LamarckConfig,
    scorer: &impl DirectoryScorer,
) -> Result<RunResult, String> {
    fs::create_dir_all(&config.output_dir).map_err(|e| e.to_string())?;
    let journal_path = config.output_dir.join("experiments.jsonl");
    let best_path = config.output_dir.join("best.json");
    let winners_dir = config.output_dir.join("winners");

    // Draw an explicit seed when none was supplied so the run is replayable
    // (issue #71) — an unrecorded OS-entropy seed cannot be replayed.
    let (seed, seed_source) = match config.seed {
        Some(seed) => (seed, SeedSource::Supplied),
        None => (rand::random::<u64>(), SeedSource::Drawn),
    };
    match seed_source {
        SeedSource::Supplied => log::info(&format!("seed {seed} (supplied)")),
        SeedSource::Drawn => log::info(&format!(
            "seed {seed} (drawn from OS entropy; replay this run with --seed {seed})"
        )),
    }
    append_journal_line(
        &journal_path,
        &RunHeaderRecord::new(
            seed,
            seed_source,
            RunConfigRecord::from_config(config),
            unix_now(),
        ),
    )?;

    let original_text = fs::read_to_string(&config.creature).map_err(|e| e.to_string())?;
    let mut incumbent = parse_creature_json(&original_text).map_err(|e| e.to_string())?;
    // Tags/uuid are stripped by CreatureExport — keep them for check-in writes.
    let mut creature_meta = CreatureMeta::from_creature_json(&original_text);
    // Never modify the supplied file — work from in-memory / output copies.
    fs::write(&best_path, &original_text).map_err(|e| e.to_string())?;

    let train_cfg = TrainingDataConfig::new(incumbent.input, incumbent.output);
    log::info(&format!(
        "ensuring observations-{} (inputs={} outputs={})",
        config.stats_mode.label(),
        incumbent.input,
        incumbent.output
    ));
    if matches!(config.stats_mode, crate::observations::StatsMode::Quick) {
        log::warn(&format!(
            "quick mode: analysis uses first {} records; scorer still evaluates the full corpus",
            config.quick_sample_records
        ));
    }
    let sample_limit = match config.stats_mode {
        crate::observations::StatsMode::Quick => Some(config.quick_sample_records),
        crate::observations::StatsMode::Full => None,
    };
    let observations = ensure_statistics(
        &config.training_data,
        &train_cfg,
        config.stats_mode,
        sample_limit,
        config.compute_correlations,
    )
    .map_err(|e| e.to_string())?;

    let mut opening_baseline_score = None;
    if config.phase0_parity {
        log::info("Phase-0: scoring incumbent baseline via authoritative scorer");
        let phase0_dir = config.output_dir.join("phase0-baseline");
        match score_single_creature_dir(&incumbent, &config.training_data, scorer, &phase0_dir) {
            Ok(baseline) => {
                if !baseline.score.is_finite() || !baseline.error.is_finite() {
                    return Err(format!(
                        "Phase-0 parity failed: non-finite score/error (score={} error={})",
                        baseline.score, baseline.error
                    ));
                }
                log::ok(&format!(
                    "Phase-0 scorer baseline score={:.12} error={:.12} complexity={:.12}",
                    baseline.score, baseline.error, baseline.complexity_penalty
                ));
                log::detail("Phase-0: computing Lamarck local MSE for parity check...");
                let mut phase0_net = compile_creature(&incumbent).map_err(|e| e.to_string())?;
                let (local_error, local_count) =
                    compute_local_mse(&incumbent, &mut phase0_net, &config.training_data)?;
                log::detail(&format!(
                    "Phase-0 local MSE={local_error:.12} over {local_count} records"
                ));
                check_phase0_parity(
                    local_error,
                    baseline.error,
                    baseline.score,
                    baseline.complexity_penalty,
                )?;
                log::ok("Phase-0 Lamarck ↔ scorer parity within documented epsilon");
                opening_baseline_score = Some(baseline.score);
                creature_meta.upsert("score", format!("{}", baseline.score));
                creature_meta.upsert("error", format!("{}", baseline.error));
                if !config.preserve_losers {
                    let _ = fs::remove_dir_all(&phase0_dir);
                }
            }
            Err(e) => {
                return Err(format!("Phase-0 parity gate failed (scorer): {e}"));
            }
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut random_focus = RandomFocusSelector;
    let mut unsaturated_focus = UnsaturatedFocusSelector;
    let mut high_error_focus = HighErrorFocusSelector;
    let mut weighted_focus = WeightedFocusSelector::default();
    let mut fixed_focus = config
        .focus_neuron
        .as_ref()
        .map(|uuid| FixedFocusSelector { uuid: uuid.clone() });
    let backprop = BackpropConfig::default();
    let focus_sample_limit = match config.stats_mode {
        crate::observations::StatsMode::Quick => Some(config.quick_sample_records),
        crate::observations::StatsMode::Full => None,
    };

    let deadline = Instant::now() + config.timeout;
    let mut experiments = 0u64;
    let mut acceptances = 0u64;
    let mut scorer_failures = 0u64;
    let mut scorer_successes = 0u64;
    let mut consecutive_scorer_failures = 0u32;
    let mut best_score = opening_baseline_score.unwrap_or(f64::NEG_INFINITY);
    // Last-accept details for the final run-summary stamp (Issue #35).
    let mut last_accept_focus = String::new();
    let mut last_accept_strategy = CandidateStrategy::Random;
    let mut last_accept_error = f64::NAN;

    // Phase-G: replay local structural grafts onto the opening fittest.
    let mut graft_store = if let Some(path) = &config.grafts_path {
        match GraftStore::load(path) {
            Ok(store) => Some((path.clone(), store)),
            Err(e) => {
                log::warn(&format!(
                    "grafts: failed to load {}: {e}; starting empty",
                    path.display()
                ));
                Some((path.clone(), GraftStore::new()))
            }
        }
    } else {
        None
    };

    if let Some((grafts_path, store)) = graft_store.take() {
        let budget = config
            .graft_replay_budget
            .unwrap_or_else(|| default_graft_replay_budget(config.timeout));
        let remaining = deadline.saturating_duration_since(Instant::now());
        let graft_deadline = Instant::now() + budget.min(remaining);
        log::info(&format!(
            "Phase-G: graft replay (budget={}ms, store={})",
            budget.min(remaining).as_millis(),
            grafts_path.display()
        ));
        let baseline_hint = opening_baseline_score.map(|score| {
            let error = creature_meta
                .tags
                .iter()
                .find(|t| t.name == "error")
                .and_then(|t| t.value.parse().ok())
                .unwrap_or(f64::NAN);
            ScoreResult {
                score,
                error,
                complexity_penalty: 0.0,
            }
        });
        let work = config.output_dir.join("graft-replay");
        match replay_grafts(
            scorer,
            GraftReplayRequest {
                host: &incumbent,
                store,
                training_data: &config.training_data,
                work_dir: &work,
                deadline: graft_deadline,
                min_improvement: config.min_improvement,
                baseline_hint: baseline_hint.as_ref(),
            },
        ) {
            Ok(replay) => {
                scorer_successes += replay.scorer_successes;
                scorer_failures += replay.scorer_failures;
                if replay.grafts_applied > 0 {
                    incumbent = replay.creature;
                    if let (Some(score), Some(error)) = (replay.score, replay.error) {
                        best_score = score;
                        last_accept_error = error;
                        last_accept_focus = "(graft-replay)".into();
                        last_accept_strategy = CandidateStrategy::StructuralAdd;
                        acceptances += 1;
                        creature_meta.upsert("score", format!("{score}"));
                        creature_meta.upsert("error", format!("{error}"));
                        let tagged = serialize_creature_with_meta(&incumbent, &creature_meta)?;
                        fs::write(&best_path, &tagged).map_err(|e| e.to_string())?;
                        log::ok(&format!(
                            "Phase-G: applied {} graft(s); incumbent score={score:.12}",
                            replay.grafts_applied
                        ));
                    }
                } else if let Some(score) = replay.score
                    && best_score.is_infinite()
                {
                    best_score = score;
                }
                if let Err(e) = replay.store.save(&grafts_path) {
                    log::warn(&format!(
                        "grafts: failed to save {}: {e}",
                        grafts_path.display()
                    ));
                }
                graft_store = Some((grafts_path, replay.store));
            }
            Err(e) => {
                log::warn(&format!("Phase-G: graft replay failed: {e}"));
                if let Err(save_e) = e.store.save(&grafts_path) {
                    log::warn(&format!(
                        "grafts: failed to save {}: {save_e}",
                        grafts_path.display()
                    ));
                }
                graft_store = Some((grafts_path, e.store));
            }
        }
        if !config.preserve_losers {
            let _ = fs::remove_dir_all(config.output_dir.join("graft-replay"));
        }
    }

    log::info(&format!(
        "starting optimisation loop (timeout={}s, candidates={})",
        config.timeout.as_secs(),
        config.candidates
    ));
    if let Some(uuid) = &config.focus_neuron {
        log::detail(&format!("focus locked to {uuid}"));
    } else {
        log::detail(&format!("focus policy: {}", config.focus_policy.label()));
    }

    while Instant::now() < deadline {
        experiments += 1;
        let remaining = deadline.saturating_duration_since(Instant::now());
        log::info(&format!(
            "experiment {experiments} ({}s remaining, acceptances={acceptances})",
            remaining.as_secs()
        ));
        let analysis_start = Instant::now();
        let mut network = compile_creature(&incumbent).map_err(|e| e.to_string())?;

        // Learning + output MAE first so weighted/high-error can rank by
        // improvement chance and skip zero-error / zero-blame neurons.
        log::detail("accumulating creature learning signal (propagate_topological_loop)...");
        let learn_start = Instant::now();
        let mut learn_rng = StdRng::seed_from_u64(
            seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(experiments),
        );
        let learning = accumulate_creature_learning(
            &incumbent,
            &mut network,
            &config.training_data,
            &backprop,
            focus_sample_limit,
            &mut learn_rng,
        )?;
        log::ok(&format!(
            "learning signal in {}ms (bias_count={:.0}, weight_count={:.0})",
            learn_start.elapsed().as_millis(),
            learning.biases.iter().map(|b| b.count).sum::<f64>(),
            learning.weights.iter().map(|w| w.count).sum::<f64>()
        ));

        let needs_signals = fixed_focus.is_none()
            && matches!(
                config.focus_policy,
                FocusPolicy::Weighted | FocusPolicy::HighError
            );
        let improvement_signals = if needs_signals {
            log::detail("scanning output residuals for focus ranking...");
            let mae_start = Instant::now();
            let output_errors = collect_output_mean_abs_errors(
                &incumbent,
                &mut network,
                &config.training_data,
                focus_sample_limit,
            )?;
            let signals = build_improvement_signals(&incumbent, &output_errors, &learning);
            log::detail(&format!(
                "error-influence signals: {} eligible neurons in {}ms",
                signals.len(),
                mae_start.elapsed().as_millis()
            ));
            signals
        } else {
            std::collections::HashMap::new()
        };

        let focus = if let Some(selector) = fixed_focus.as_mut() {
            let uuid = selector.select(&incumbent, &mut rng).ok_or_else(|| {
                format!(
                    "focus neuron '{}' not found (or is an input)",
                    selector.uuid
                )
            })?;
            log::detail(&format!("focus neuron: {uuid} (pinned)"));
            uuid
        } else {
            match config.focus_policy {
                FocusPolicy::Weighted => {
                    let choice = weighted_focus
                        .select_weighted(&incumbent, &improvement_signals, &mut rng)
                        .or_else(|| {
                            log::warn(
                                "no non-zero improvement signal; falling back to first output",
                            );
                            high_error_focus
                                .select(&incumbent, &mut rng)
                                .map(|uuid| FocusChoice {
                                    uuid,
                                    weight: 0.0,
                                    reason: "fallback_first_output".into(),
                                })
                        })
                        .ok_or_else(|| "no focus neuron available".to_string())?;
                    log::detail(&format!(
                        "focus neuron: {} (weight={:.2}, {})",
                        choice.uuid, choice.weight, choice.reason
                    ));
                    choice.uuid
                }
                FocusPolicy::HighError => {
                    let choice = select_highest_signal(&improvement_signals).or_else(|| {
                        log::warn("no non-zero improvement signal; falling back to first output");
                        high_error_focus
                            .select(&incumbent, &mut rng)
                            .map(|uuid| FocusChoice {
                                uuid,
                                weight: 0.0,
                                reason: "fallback_first_output".into(),
                            })
                    });
                    let choice = choice.ok_or_else(|| "no focus neuron available".to_string())?;
                    log::detail(&format!(
                        "focus neuron: {} ({})",
                        choice.uuid, choice.reason
                    ));
                    choice.uuid
                }
                FocusPolicy::Random => {
                    let uuid = random_focus
                        .select(&incumbent, &mut rng)
                        .ok_or_else(|| "no focus neuron available".to_string())?;
                    log::detail(&format!("focus neuron: {uuid}"));
                    uuid
                }
                FocusPolicy::Unsaturated => {
                    let uuid = unsaturated_focus
                        .select(&incumbent, &mut rng)
                        .ok_or_else(|| "no focus neuron available".to_string())?;
                    log::detail(&format!("focus neuron: {uuid}"));
                    uuid
                }
            }
        };
        log::detail("scanning incumbent for focus stats...");
        let focus_scan_start = Instant::now();
        let mut focus_stats = collect_focus_stats(
            &incumbent,
            &mut network,
            &config.training_data,
            &focus,
            focus_sample_limit,
        )?;
        let focus_scan_ms = focus_scan_start.elapsed().as_millis();
        log::ok(&format!(
            "focus scan: {} records in {focus_scan_ms}ms",
            focus_stats.record_count
        ));
        match (
            focus_stats.mean_error,
            focus_stats.mean_abs_error,
            focus_stats.mean_adjusted_error,
            focus_stats.mean_derivative,
        ) {
            (Some(me), Some(mae), Some(madj), Some(md)) => log::detail(&format!(
                "focus error: mean={me:.6e}  mae={mae:.6e}  adj={madj:.6e}  deriv={md:.4}  sat={:.3}",
                focus_stats.saturation_fraction
            )),
            _ => log::detail(&format!(
                "focus (no target error): post_mean={:.6e}  pre_mean={:.6e}",
                focus_stats.post_mean, focus_stats.pre_mean
            )),
        }
        if matches!(config.stats_mode, crate::observations::StatsMode::Quick) {
            log::detail(&format!(
                "analysis sample={} records; acceptance uses full-corpus scorer",
                focus_stats.record_count
            ));
        }

        let mut incoming = collect_incoming_source_stats(
            &incumbent,
            &mut network,
            &config.training_data,
            &focus,
            focus_sample_limit,
            Some(&observations),
        )?;
        log::detail(&format!("incoming sources: {}", incoming.len()));

        // Surface focus blame + incoming weight signals from real backprop (#4).
        attach_focus_blame(&mut focus_stats, &incumbent, &learning);
        let lr = crate::backprop::calculate_learning_rate(&backprop, experiments, None);
        attach_learning_to_incoming(&mut incoming, &learning, &backprop, lr);
        if let Some(blame) = focus_stats.mean_blame {
            log::detail(&format!(
                "focus blame: mean={blame:.6e}  count={:.0}  no_change={}",
                focus_stats.blame_count.unwrap_or(0.0),
                focus_stats.blame_no_change.unwrap_or(false)
            ));
        }
        let linked = incoming
            .iter()
            .filter(|s| s.proposed_weight_delta.is_some())
            .count();
        if linked > 0 {
            log::detail(&format!("incoming weight-signals attached: {linked}"));
        }

        let prior_sources = rank_unused_sources(&incumbent, &focus, &observations);
        let ranked_sources = refine_sources_by_residual_with_observations(
            &incumbent,
            &mut network,
            &config.training_data,
            &focus,
            &prior_sources,
            focus_sample_limit,
            Some(&observations),
        )?;
        if let Some(best) = ranked_sources.first() {
            log::detail(&format!(
                "best unused source {} residual|corr|={:.4}",
                best.from_uuid, best.score
            ));
        }
        if let Some(best_hidden) = ranked_sources
            .iter()
            .find(|s| !is_input_source(&s.from_uuid))
        {
            log::detail(&format!(
                "best unused hidden {} residual|corr|={:.4}",
                best_hidden.from_uuid, best_hidden.score
            ));
        }

        if config.structural_only {
            log::detail("structural-only: synapse/neuron growth candidates only");
        }
        let gen_ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: &focus,
            focus_stats: &focus_stats,
            incoming: &incoming,
            observations: &observations,
            ranked_sources: Some(&ranked_sources),
            learning: Some(&learning),
            backprop: &backprop,
            structural_only: config.structural_only,
        };
        let gen_start = Instant::now();
        let candidates = generate_candidates(&gen_ctx, config.candidates, &mut rng);
        let generate_ms = gen_start.elapsed().as_millis();
        let analysis_ms = analysis_start.elapsed().as_millis();
        log::ok(&format!(
            "generated {} candidates in {generate_ms}ms (analysis total {analysis_ms}ms)",
            candidates.len()
        ));

        let batch_dir = config
            .output_dir
            .join(format!("candidates-exp-{experiments}"));
        write_candidate_batch(&batch_dir, &incumbent, &candidates, Some(&creature_meta))?;

        let screen_rate = config.screen_sample_rate.filter(|r| *r > 0.0 && *r < 1.0);
        let scorer_start = Instant::now();
        let mut screen_score_map: Option<std::collections::BTreeMap<String, f64>> = None;
        let mut promote_dir: Option<PathBuf> = None;

        let scores = if let Some(rate) = screen_rate {
            // --- Screen phase (cheap subsample) ---
            let sample = ScoreSample {
                rate,
                phase: experiments.saturating_sub(1),
            };
            log::detail(&format!(
                "screen: scoring baseline + {} candidates at sample-rate={rate} phase={} via {}",
                candidates.len(),
                sample.phase,
                config.scorer_path.display()
            ));
            let screen_scores = match scorer.score_directory_sampled(
                &batch_dir,
                &config.training_data,
                sample,
            ) {
                Ok(s) => s,
                Err(e) => {
                    scorer_failures += 1;
                    consecutive_scorer_failures += 1;
                    log::warn(&format!("screen scorer failed: {e}"));
                    weighted_focus.record_outcome(
                        &focus,
                        false,
                        None,
                        true,
                        config.min_improvement,
                    );
                    append_journal(
                        &journal_path,
                        &ExperimentRecord {
                            experiment_number: experiments,
                            timestamp_unix: unix_now(),
                            seed: Some(seed),
                            incumbent_id: incumbent_id(&incumbent),
                            baseline_score: best_score,
                            focus_neuron: focus,
                            focus_stats: Some(focus_stats.clone()),
                            candidates: candidates.iter().map(|c| c.provenance.clone()).collect(),
                            scores: Default::default(),
                            screen_scores: None,
                            winner: None,
                            improvement: None,
                            accepted: false,
                            analysis_ms,
                            scorer_ms: scorer_start.elapsed().as_millis(),
                            scorer_error: Some(e.to_string()),
                            combo_members: None,
                            combos_scored: None,
                            combos_dampened: None,
                            combo_dampen: None,
                        },
                    )?;
                    if !config.preserve_losers {
                        let _ = fs::remove_dir_all(&batch_dir);
                    }
                    if consecutive_scorer_failures >= config.max_consecutive_scorer_failures {
                        return Err(format!(
                            "aborting after {consecutive_scorer_failures} consecutive scorer failures; last error: {e}"
                        ));
                    }
                    continue;
                }
            };
            let screen_ms = scorer_start.elapsed().as_millis();
            log_scorer_batch_stats_labeled(
                &screen_scores,
                screen_ms,
                config.screen_promote_threshold,
                "screen",
            );
            consecutive_scorer_failures = 0;
            scorer_successes += 1;
            screen_score_map = Some(
                screen_scores
                    .iter()
                    .map(|(k, v)| (k.clone(), v.score))
                    .collect(),
            );

            let promote_stems =
                screen_promote_stems(&screen_scores, config.screen_promote_threshold)
                    .map_err(|e| e.to_string())?;
            if promote_stems.is_empty() {
                log::detail("screen empty: no sample improvers → skipping full-corpus score");
                // Sterile focus: dampen so weighted draw explores elsewhere.
                weighted_focus.record_outcome(
                    &focus,
                    false,
                    Some(0.0),
                    false,
                    config.min_improvement,
                );
                append_journal(
                    &journal_path,
                    &ExperimentRecord {
                        experiment_number: experiments,
                        timestamp_unix: unix_now(),
                        seed: Some(seed),
                        incumbent_id: incumbent_id(&incumbent),
                        baseline_score: screen_scores
                            .get("baseline")
                            .map(|b| b.score)
                            .unwrap_or(best_score),
                        focus_neuron: focus,
                        focus_stats: Some(focus_stats.clone()),
                        candidates: candidates.iter().map(|c| c.provenance.clone()).collect(),
                        scores: Default::default(),
                        screen_scores: screen_score_map,
                        winner: None,
                        improvement: None,
                        accepted: false,
                        analysis_ms,
                        scorer_ms: screen_ms,
                        scorer_error: None,
                        combo_members: None,
                        combos_scored: None,
                        combos_dampened: None,
                        combo_dampen: None,
                    },
                )?;
                if !config.preserve_losers {
                    let _ = fs::remove_dir_all(&batch_dir);
                }
                continue;
            }

            log::detail(&format!(
                "promote: {} candidate(s) cleared screen (threshold Δ > {}) → full corpus",
                promote_stems.len(),
                config.screen_promote_threshold
            ));
            let pdir = config.output_dir.join(format!("promote-exp-{experiments}"));
            write_promote_batch(&pdir, &batch_dir, &promote_stems)?;
            promote_dir = Some(pdir.clone());

            let promote_start = Instant::now();
            match scorer.score_directory(&pdir, &config.training_data) {
                Ok(s) => {
                    let promote_ms = promote_start.elapsed().as_millis();
                    log_scorer_batch_stats_labeled(
                        &s,
                        promote_ms,
                        config.min_improvement,
                        "promote",
                    );
                    scorer_successes += 1;
                    s
                }
                Err(e) => {
                    scorer_failures += 1;
                    consecutive_scorer_failures += 1;
                    log::warn(&format!("promote scorer failed: {e}"));
                    weighted_focus.record_outcome(
                        &focus,
                        false,
                        None,
                        true,
                        config.min_improvement,
                    );
                    append_journal(
                        &journal_path,
                        &ExperimentRecord {
                            experiment_number: experiments,
                            timestamp_unix: unix_now(),
                            seed: Some(seed),
                            incumbent_id: incumbent_id(&incumbent),
                            baseline_score: best_score,
                            focus_neuron: focus,
                            focus_stats: Some(focus_stats.clone()),
                            candidates: candidates.iter().map(|c| c.provenance.clone()).collect(),
                            scores: Default::default(),
                            screen_scores: screen_score_map,
                            winner: None,
                            improvement: None,
                            accepted: false,
                            analysis_ms,
                            scorer_ms: scorer_start.elapsed().as_millis(),
                            scorer_error: Some(e.to_string()),
                            combo_members: None,
                            combos_scored: None,
                            combos_dampened: None,
                            combo_dampen: None,
                        },
                    )?;
                    if !config.preserve_losers {
                        let _ = fs::remove_dir_all(&batch_dir);
                        let _ = fs::remove_dir_all(&pdir);
                    }
                    if consecutive_scorer_failures >= config.max_consecutive_scorer_failures {
                        return Err(format!(
                            "aborting after {consecutive_scorer_failures} consecutive scorer failures; last error: {e}"
                        ));
                    }
                    continue;
                }
            }
        } else {
            // --- Single full-corpus score (legacy path) ---
            log::detail(&format!(
                "scoring baseline + {} candidates via {}",
                candidates.len(),
                config.scorer_path.display()
            ));
            match scorer.score_directory(&batch_dir, &config.training_data) {
                Ok(s) => {
                    scorer_successes += 1;
                    let full_ms = scorer_start.elapsed().as_millis();
                    log_scorer_batch_stats_labeled(&s, full_ms, config.min_improvement, "scorer");
                    s
                }
                Err(e) => {
                    scorer_failures += 1;
                    consecutive_scorer_failures += 1;
                    log::warn(&format!("scorer failed: {e}"));
                    weighted_focus.record_outcome(
                        &focus,
                        false,
                        None,
                        true,
                        config.min_improvement,
                    );
                    append_journal(
                        &journal_path,
                        &ExperimentRecord {
                            experiment_number: experiments,
                            timestamp_unix: unix_now(),
                            seed: Some(seed),
                            incumbent_id: incumbent_id(&incumbent),
                            baseline_score: best_score,
                            focus_neuron: focus,
                            focus_stats: Some(focus_stats.clone()),
                            candidates: candidates.iter().map(|c| c.provenance.clone()).collect(),
                            scores: Default::default(),
                            screen_scores: None,
                            winner: None,
                            improvement: None,
                            accepted: false,
                            analysis_ms,
                            scorer_ms: scorer_start.elapsed().as_millis(),
                            scorer_error: Some(e.to_string()),
                            combo_members: None,
                            combos_scored: None,
                            combos_dampened: None,
                            combo_dampen: None,
                        },
                    )?;
                    if !config.preserve_losers {
                        let _ = fs::remove_dir_all(&batch_dir);
                    }
                    if consecutive_scorer_failures >= config.max_consecutive_scorer_failures {
                        return Err(format!(
                            "aborting after {consecutive_scorer_failures} consecutive scorer failures; last error: {e}"
                        ));
                    }
                    continue;
                }
            }
        };

        consecutive_scorer_failures = 0;
        let scorer_ms = scorer_start.elapsed().as_millis();

        let baseline = scores
            .get("baseline")
            .ok_or_else(|| "baseline missing from scorer results".to_string())?;
        if best_score.is_infinite() {
            best_score = baseline.score;
        }
        if opening_baseline_score.is_none() {
            opening_baseline_score = Some(baseline.score);
        }

        let source_dir = promote_dir.as_ref().unwrap_or(&batch_dir);
        let combo_dir = config.output_dir.join(format!("combos-exp-{experiments}"));
        let selection = match select_best_with_combinations(
            scorer,
            ComboSelectRequest {
                training_data: &config.training_data,
                incumbent: &incumbent,
                candidates: &candidates,
                scores: &scores,
                min_improvement: config.min_improvement,
                source_dir,
                combo_work_dir: &combo_dir,
            },
        ) {
            Ok(s) => s,
            Err(e) => {
                log::warn(&format!(
                    "combo selection failed: {e}; using best improving single if any"
                ));
                crate::combos::collect_improvers(&scores, config.min_improvement)
                    .ok()
                    .and_then(|improvers| {
                        let best = improvers.into_iter().next()?;
                        Some(ComboSelection {
                            creature_path: source_dir.join(format!("{}.json", best.stem)),
                            stem: best.stem,
                            result: best.result,
                            delta: best.delta,
                            member_indices: vec![best.index],
                            dampen: StackDampenReport::default(),
                            combos_scored: 0,
                            combos_dampened: 0,
                        })
                    })
            }
        };

        let (combo_members, combos_scored, combos_dampened, combo_dampen) = match &selection {
            Some(sel) if sel.combos_scored > 0 || sel.member_indices.len() > 1 => (
                (sel.member_indices.len() > 1).then_some(sel.member_indices.len()),
                Some(sel.combos_scored),
                Some(sel.combos_dampened),
                (!sel.dampen.is_empty()).then(|| sel.dampen.clone()),
            ),
            _ => (None, None, None, None),
        };

        let mut accepted = false;
        let mut improvement = None;
        let mut winner_stem = None;
        if let Some(sel) = selection
            && accepts_improvement(sel.result.score, baseline.score, config.min_improvement)
        {
            if screen_rate.is_some() {
                log::detail(
                    "accepting on full-corpus promote score (screen used a scorer subsample)",
                );
            } else if matches!(config.stats_mode, crate::observations::StatsMode::Quick) {
                log::detail(
                    "accepting on full-corpus scorer (analysis used a quick sample; directions may differ)",
                );
            }
            let strategy = sel
                .member_indices
                .first()
                .and_then(|i| candidates.get(*i).map(|c| c.provenance.strategy))
                .unwrap_or(CandidateStrategy::Random);
            let delta = sel.delta;
            log::ok(&format!(
                "🏆 accepted {}: score={} (+{delta:.3e}, {} member(s))",
                sel.stem,
                sel.result.score,
                sel.member_indices.len()
            ));
            let winner_json = fs::read_to_string(&sel.creature_path).map_err(|e| e.to_string())?;
            let previous = incumbent.clone();
            incumbent = parse_creature_json(&winner_json).map_err(|e| e.to_string())?;
            let opening = opening_baseline_score.unwrap_or(baseline.score);
            last_accept_focus = focus.clone();
            last_accept_strategy = strategy;
            last_accept_error = sel.result.error;
            creature_meta.stamp_acceptance(&LamarckProgress {
                acceptances: acceptances + 1,
                score: sel.result.score,
                error: sel.result.error,
                opening_score: opening,
                focus_neuron: &focus,
                strategy,
                experiments,
            });
            let tagged = serialize_creature_with_meta(&incumbent, &creature_meta)?;
            log::detail(&format!(
                "🏷️  {}",
                creature_meta
                    .tags
                    .iter()
                    .find(|t| t.name == "lamarck")
                    .map(|t| t.value.as_str())
                    .unwrap_or("(no lamarck tag)")
            ));
            fs::write(&best_path, &tagged).map_err(|e| e.to_string())?;
            fs::create_dir_all(&winners_dir).map_err(|e| e.to_string())?;
            fs::write(
                winners_dir.join(format!("winner-{experiments:04}.json")),
                &tagged,
            )
            .map_err(|e| e.to_string())?;
            best_score = sel.result.score;
            accepted = true;
            acceptances += 1;
            improvement = Some(delta);
            winner_stem = Some(sel.stem.clone());

            // Record structural improvements into the local graft store.
            // Combo accepts must persist each member's solo-sized delta — never
            // the dampened merged creature (replay would then double-dampen).
            if let Some((grafts_path, store)) = graft_store.as_mut() {
                let mut recorded_any = false;
                let structural_members: Vec<&Candidate> = if sel.member_indices.len() <= 1 {
                    sel.member_indices
                        .first()
                        .and_then(|i| candidates.get(*i))
                        .filter(|c| {
                            matches!(
                                c.provenance.strategy,
                                CandidateStrategy::StructuralAdd
                                    | CandidateStrategy::StructuralAddNeuron
                            )
                        })
                        .into_iter()
                        .collect()
                } else {
                    sel.member_indices
                        .iter()
                        .filter_map(|i| candidates.get(*i))
                        .filter(|c| {
                            matches!(
                                c.provenance.strategy,
                                CandidateStrategy::StructuralAdd
                                    | CandidateStrategy::StructuralAddNeuron
                            )
                        })
                        .collect()
                };
                for member in structural_members {
                    // Solo weights from the member candidate (not dampened incumbent).
                    if let Some(id) =
                        record_structural_acceptance(store, &previous, &member.creature)
                    {
                        log::detail(&format!("grafts: recorded structural accept {id}"));
                        recorded_any = true;
                    }
                }
                if recorded_any && let Err(e) = store.save(grafts_path) {
                    log::warn(&format!(
                        "grafts: failed to save {}: {e}",
                        grafts_path.display()
                    ));
                }
            }
        } else {
            log::detail("no candidate met the acceptance threshold");
        }

        let best_full_delta = best_candidate_delta(&scores);
        weighted_focus.record_outcome(
            &focus,
            accepted,
            best_full_delta,
            false,
            config.min_improvement,
        );

        let score_map = scores.iter().map(|(k, v)| (k.clone(), v.score)).collect();
        append_journal(
            &journal_path,
            &ExperimentRecord {
                experiment_number: experiments,
                timestamp_unix: unix_now(),
                seed: Some(seed),
                incumbent_id: incumbent_id(&incumbent),
                baseline_score: baseline.score,
                focus_neuron: focus,
                focus_stats: Some(focus_stats.clone()),
                candidates: candidates.iter().map(|c| c.provenance.clone()).collect(),
                scores: score_map,
                screen_scores: screen_score_map,
                winner: winner_stem,
                improvement,
                accepted,
                analysis_ms,
                scorer_ms,
                scorer_error: None,
                combo_members,
                combos_scored,
                combos_dampened,
                combo_dampen,
            },
        )?;

        if !config.preserve_losers {
            let _ = fs::remove_dir_all(&batch_dir);
            if let Some(pdir) = &promote_dir {
                let _ = fs::remove_dir_all(pdir);
            }
            let _ = fs::remove_dir_all(&combo_dir);
        }
    }

    if experiments > 0 && scorer_successes == 0 {
        return Err(format!(
            "no successful scorer batches ({scorer_failures} failures); check rust_scorer path/binary"
        ));
    }

    // Final check-in tag: full run experiment count (may exceed last-accept exp).
    if acceptances > 0 {
        let opening = opening_baseline_score.unwrap_or(best_score);
        let error = if last_accept_error.is_finite() {
            last_accept_error
        } else {
            creature_meta
                .tags
                .iter()
                .find(|t| t.name == "error")
                .and_then(|t| t.value.parse().ok())
                .unwrap_or(f64::NAN)
        };
        creature_meta.stamp_acceptance(&LamarckProgress {
            acceptances,
            score: best_score,
            error,
            opening_score: opening,
            focus_neuron: &last_accept_focus,
            strategy: last_accept_strategy,
            experiments,
        });
        let tagged = serialize_creature_with_meta(&incumbent, &creature_meta)?;
        log::detail(&format!(
            "🏷️  final {}",
            creature_meta
                .tags
                .iter()
                .find(|t| t.name == "lamarck")
                .map(|t| t.value.as_str())
                .unwrap_or("(no lamarck tag)")
        ));
        fs::write(&best_path, &tagged).map_err(|e| e.to_string())?;
    }

    Ok(RunResult {
        best_path,
        journal_path,
        best_score,
        experiments,
        acceptances,
        scorer_failures,
        scorer_successes,
        opening_baseline_score,
        seed,
    })
}

/// Best full-corpus Δ vs baseline among non-baseline stems (for focus history).
fn best_candidate_delta(scores: &std::collections::BTreeMap<String, ScoreResult>) -> Option<f64> {
    let baseline = scores.get("baseline")?;
    scores
        .iter()
        .filter(|(stem, _)| stem.as_str() != "baseline")
        .map(|(_, r)| improvement(r.score, baseline.score))
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn incumbent_id(creature: &neat_core::CreatureExport) -> String {
    format!(
        "in{}-out{}-n{}-s{}",
        creature.input,
        creature.output,
        creature.neurons.len(),
        creature.synapses.len()
    )
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn append_journal(path: &Path, record: &ExperimentRecord) -> Result<(), String> {
    append_journal_line(path, record)
}

/// Append one serialisable journal line (header or experiment) to the journal.
fn append_journal_line(path: &Path, line: &impl Serialize) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    let encoded = serde_json::to_string(line).map_err(|e| e.to_string())?;
    writeln!(file, "{encoded}").map_err(|e| e.to_string())?;
    Ok(())
}

/// Score the supplied creature once for Phase-0 baseline recording.
pub fn score_single_creature_dir(
    creature: &neat_core::CreatureExport,
    training_data: &Path,
    scorer: &impl DirectoryScorer,
    work_dir: &Path,
) -> Result<ScoreResult, String> {
    fs::create_dir_all(work_dir).map_err(|e| e.to_string())?;
    let json = creature_to_json_pretty(creature).map_err(|e| e.to_string())?;
    fs::write(work_dir.join("baseline.json"), json).map_err(|e| e.to_string())?;
    let results = scorer
        .score_directory(work_dir, training_data)
        .map_err(|e| e.to_string())?;
    results
        .get("baseline")
        .cloned()
        .ok_or_else(|| "baseline missing".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorer::ScoreResult;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tempfile::tempdir;

    struct ScriptedScorer {
        calls: Arc<Mutex<usize>>,
    }

    impl DirectoryScorer for ScriptedScorer {
        fn score_directory_sampled(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
            _sample: crate::scorer::ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            // Matches tiny_setup local MSE: pred=1.1, target=0.5 → error=0.36.
            const BASE_ERROR: f64 = 0.36;
            const BASE_SCORE: f64 = 1.0 - BASE_ERROR;
            let mut map = BTreeMap::new();
            map.insert(
                "baseline".into(),
                ScoreResult {
                    score: BASE_SCORE,
                    error: BASE_ERROR,
                    complexity_penalty: 0.0,
                },
            );
            if let Ok(rd) = fs::read_dir(candidates_dir) {
                for entry in rd.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if let Some(stem) = name.strip_suffix(".json") {
                        if stem == "baseline" {
                            continue;
                        }
                        let score = if *calls <= 2 && stem == "candidate-000" {
                            BASE_SCORE + 2e-6
                        } else {
                            BASE_SCORE
                        };
                        map.insert(
                            stem.to_string(),
                            ScoreResult {
                                score,
                                error: 1.0 - score,
                                complexity_penalty: 0.0,
                            },
                        );
                    }
                }
            }
            Ok(map)
        }
    }

    struct FailingScorer;

    impl DirectoryScorer for FailingScorer {
        fn score_directory_sampled(
            &self,
            _candidates_dir: &Path,
            _training_data: &Path,
            _sample: crate::scorer::ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            Err(crate::scorer::ScorerError::Process("boom".into()))
        }
    }

    fn tiny_setup(dir: &Path) -> (PathBuf, PathBuf) {
        let creature_path = dir.join("creature.json");
        let training = dir.join("data");
        fs::create_dir_all(&training).unwrap();
        fs::write(
            training.join("0.bin"),
            [1.0f32, 0.5f32]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        fs::write(
            &creature_path,
            r#"{
              "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
              "neurons":[
                {"type":"hidden","uuid":"h1","bias":0.1,"squash":"IDENTITY"},
                {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
              ],
              "synapses":[
                {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
                {"fromUUID":"h1","toUUID":"o1","weight":1.0}
              ]
            }"#,
        )
        .unwrap();
        (creature_path, training)
    }

    #[test]
    fn loop_accepts_winner_and_writes_journal() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let out = dir.path().join("out");
        let config = LamarckConfig {
            creature: creature_path,
            training_data: training,
            timeout: Duration::from_millis(500),
            candidates: 4,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: out.clone(),
            preserve_losers: true,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            phase0_parity: true,
            structural_only: false,
            screen_sample_rate: None,
            screen_promote_threshold: 0.0,
            grafts_path: None,
            graft_replay_budget: None,
        };
        let scorer = ScriptedScorer {
            calls: Arc::new(Mutex::new(0)),
        };
        let result = run_optimisation(&config, &scorer).unwrap();
        assert!(result.journal_path.is_file());
        assert!(result.best_path.is_file());
        assert!(result.experiments >= 1);
        assert!(result.scorer_successes >= 1);
        let journal = fs::read_to_string(&result.journal_path).unwrap();
        let mut lines = journal.lines();
        // Line 1 is the run header (issue #71); experiments follow.
        assert!(lines.next().unwrap().contains("\"record\":\"runHeader\""));
        assert!(lines.next().unwrap().contains("experimentNumber"));
        if result.acceptances > 0 {
            let best = fs::read_to_string(&result.best_path).unwrap();
            let value: serde_json::Value = serde_json::from_str(&best).unwrap();
            let tags = value["tags"].as_array().expect("tags");
            let lamarck = tags
                .iter()
                .find(|t| t["name"] == "lamarck")
                .expect("lamarck tag")["value"]
                .as_str()
                .unwrap();
            assert!(
                lamarck.contains("accept") && lamarck.contains("score:"),
                "run-summary tag missing accepts/score: {lamarck}"
            );
            assert!(
                !lamarck.contains("accept #") && !lamarck.contains("🏆"),
                "tag should not use legacy last-accept wording: {lamarck}"
            );
        }
    }

    /// Screen subsample finds no improvers → skip full score (issue #24).
    struct NegativeScreenScorer;

    impl DirectoryScorer for NegativeScreenScorer {
        fn score_directory_sampled(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
            sample: crate::scorer::ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            // Subsample screen: every candidate worse than baseline.
            // Full corpus must not be reached in this test.
            assert!(
                sample.is_subsample(),
                "empty-screen test should only invoke subsample scoring"
            );
            let mut map = BTreeMap::new();
            map.insert(
                "baseline".into(),
                ScoreResult {
                    score: 0.5,
                    error: 0.5,
                    complexity_penalty: 0.0,
                },
            );
            if let Ok(rd) = fs::read_dir(candidates_dir) {
                for entry in rd.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if let Some(stem) = name.strip_suffix(".json")
                        && stem != "baseline"
                    {
                        map.insert(
                            stem.to_string(),
                            ScoreResult {
                                score: 0.5 - 1e-4,
                                error: 0.5 + 1e-4,
                                complexity_penalty: 0.0,
                            },
                        );
                    }
                }
            }
            Ok(map)
        }
    }

    #[test]
    fn screen_empty_skips_full_corpus_score() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let out = dir.path().join("out");
        let config = LamarckConfig {
            creature: creature_path,
            training_data: training,
            timeout: Duration::from_millis(800),
            candidates: 4,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: out.clone(),
            preserve_losers: true,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            phase0_parity: false,
            structural_only: false,
            screen_sample_rate: Some(0.1),
            screen_promote_threshold: 0.0,
            grafts_path: None,
            graft_replay_budget: None,
        };
        let result = run_optimisation(&config, &NegativeScreenScorer).unwrap();
        assert!(result.experiments >= 1);
        assert_eq!(result.acceptances, 0);
        // Skip the run header (issue #71) — the first experiment follows it.
        let first = journal_lines(&result.journal_path)
            .into_iter()
            .find_map(|l| match l {
                JournalLine::Experiment(record) => Some(*record),
                JournalLine::Header(_) => None,
            })
            .expect("at least one experiment");
        assert!(first.screen_scores.is_some());
        assert!(first.scores.is_empty());
        assert!(!first.accepted);
        assert!(!out.join("promote-exp-1").exists());
    }

    /// Phase-G replays a stored synapse graft onto the opening fittest.
    struct GraftAwareScorer;

    impl DirectoryScorer for GraftAwareScorer {
        fn score_directory_sampled(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
            _sample: crate::scorer::ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            let base = 0.64; // 1 - 0.36 for phase0-ish parity if needed
            let mut map = BTreeMap::new();
            for entry in fs::read_dir(candidates_dir).into_iter().flatten().flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                let Some(stem) = name.strip_suffix(".json") else {
                    continue;
                };
                let text = fs::read_to_string(entry.path()).unwrap_or_default();
                let has_edge = parse_creature_json(&text)
                    .map(|c| {
                        c.synapses
                            .iter()
                            .any(|s| s.from_uuid == "input-1" && s.to_uuid == "h1")
                    })
                    .unwrap_or(false);
                let score = if stem == "baseline" {
                    base
                } else if has_edge {
                    base + 2e-6
                } else {
                    base
                };
                map.insert(
                    stem.to_string(),
                    ScoreResult {
                        score,
                        error: 1.0 - score,
                        complexity_penalty: 0.0,
                    },
                );
            }
            Ok(map)
        }
    }

    #[test]
    fn phase_g_replays_stored_graft_before_loop() {
        use crate::grafts::{GraftStore, graft_from_add_synapse};

        let dir = tempdir().unwrap();
        let creature_path = dir.path().join("creature.json");
        let training = dir.path().join("data");
        fs::create_dir_all(&training).unwrap();
        fs::write(
            training.join("0.bin"),
            [1.0f32, 0.0f32, 0.5f32]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        fs::write(
            &creature_path,
            r#"{
              "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
              "neurons":[
                {"type":"hidden","uuid":"h1","bias":0.1,"squash":"IDENTITY"},
                {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
              ],
              "synapses":[
                {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
                {"fromUUID":"h1","toUUID":"o1","weight":1.0}
              ]
            }"#,
        )
        .unwrap();

        let grafts_path = dir.path().join("grafts.json");
        let mut store = GraftStore::new();
        store.upsert(graft_from_add_synapse("input-1", "h1", 0.05));
        store.save(&grafts_path).unwrap();

        let out = dir.path().join("out");
        let config = LamarckConfig {
            creature: creature_path,
            training_data: training,
            timeout: Duration::from_millis(500),
            candidates: 2,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: out,
            preserve_losers: true,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            phase0_parity: false,
            structural_only: false,
            screen_sample_rate: None,
            screen_promote_threshold: 0.0,
            grafts_path: Some(grafts_path.clone()),
            // Explicit budget so phase-G is not starved by a sub-second run timeout.
            graft_replay_budget: Some(Duration::from_secs(5)),
        };
        let result = run_optimisation(&config, &GraftAwareScorer).unwrap();
        assert!(result.acceptances >= 1, "phase-G should accept the graft");
        let best = parse_creature_json(&fs::read_to_string(result.best_path).unwrap()).unwrap();
        assert!(
            best.synapses
                .iter()
                .any(|s| s.from_uuid == "input-1" && s.to_uuid == "h1"),
            "best creature should carry replayed graft edge"
        );
        let stored = GraftStore::load(&grafts_path).unwrap();
        assert!(
            stored.get("edge:input-1->h1").is_some(),
            "helpful graft remains in store"
        );
    }

    fn reproducibility_config(creature: PathBuf, training: PathBuf, out: PathBuf) -> LamarckConfig {
        LamarckConfig {
            creature,
            training_data: training,
            timeout: Duration::from_millis(400),
            candidates: 4,
            min_improvement: 1e-6,
            seed: None,
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: out,
            preserve_losers: false,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            phase0_parity: false,
            structural_only: false,
            screen_sample_rate: None,
            screen_promote_threshold: 0.0,
            grafts_path: None,
            graft_replay_budget: None,
        }
    }

    fn journal_lines(path: &Path) -> Vec<JournalLine> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| JournalLine::parse(l).expect("journal line parses"))
            .collect()
    }

    /// Issue #71: an unseeded run must draw, log and record an effective seed.
    #[test]
    fn unseeded_run_records_effective_seed() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let config = reproducibility_config(creature_path, training, dir.path().join("out"));
        let scorer = ScriptedScorer {
            calls: Arc::new(Mutex::new(0)),
        };
        let result = run_optimisation(&config, &scorer).unwrap();

        let lines = journal_lines(&result.journal_path);
        let JournalLine::Header(header) = &lines[0] else {
            panic!("first journal line must be the run header");
        };
        assert_eq!(
            header.seed, result.seed,
            "header seed is the effective seed"
        );
        assert_eq!(header.seed_source, SeedSource::Drawn);
        let experiments: Vec<&ExperimentRecord> = lines
            .iter()
            .filter_map(|l| match l {
                JournalLine::Experiment(record) => Some(record.as_ref()),
                JournalLine::Header(_) => None,
            })
            .collect();
        assert!(!experiments.is_empty(), "run wrote at least one experiment");
        for record in experiments {
            assert_eq!(
                record.seed,
                Some(result.seed),
                "every record carries the effective seed"
            );
        }
    }

    /// Issue #70: every experiment journals the focus scan that drove it.
    #[test]
    fn experiment_records_the_focus_structure_statistics_and_blame() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let config = reproducibility_config(creature_path, training, dir.path().join("out"));
        let scorer = ScriptedScorer {
            calls: Arc::new(Mutex::new(0)),
        };
        let result = run_optimisation(&config, &scorer).unwrap();

        let experiments: Vec<ExperimentRecord> = journal_lines(&result.journal_path)
            .into_iter()
            .filter_map(|l| match l {
                JournalLine::Experiment(record) => Some(*record),
                JournalLine::Header(_) => None,
            })
            .collect();
        assert!(!experiments.is_empty(), "run wrote at least one experiment");
        for record in &experiments {
            let stats = record
                .focus_stats
                .as_ref()
                .expect("every experiment carries the focus scan");
            // Structure: the scanned neuron is the journalled focus.
            assert_eq!(stats.neuron_uuid, record.focus_neuron);
            assert_eq!(stats.neuron_uuid, "o1");
            assert_eq!(stats.squash.as_deref(), Some("IDENTITY"));
            assert_eq!(stats.incoming_count, 1, "o1 has one incoming synapse");
            // Statistics: the scan actually ran over training records.
            assert!(stats.record_count > 0, "focus scan saw records");
            assert!(stats.mean_abs_error.is_some(), "output focus has an error");
            assert!((0.0..=1.0).contains(&stats.saturation_fraction));
            // Blame: backprop attached a bias signal for the focus.
            assert!(stats.mean_blame.is_some(), "focus blame is journalled");
        }

        // The field must survive the journal encoding, under its camelCase name.
        let encoded = fs::read_to_string(&result.journal_path).unwrap();
        assert!(encoded.contains("\"focusStats\""));
    }

    /// Issue #71: the run header carries the knobs needed to replay the run.
    #[test]
    fn run_header_records_run_configuration() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let mut config = reproducibility_config(
            creature_path.clone(),
            training.clone(),
            dir.path().join("out"),
        );
        config.seed = Some(7);
        config.structural_only = true;
        config.screen_sample_rate = Some(0.25);
        config.screen_promote_threshold = 1e-7;
        let scorer = ScriptedScorer {
            calls: Arc::new(Mutex::new(0)),
        };
        let result = run_optimisation(&config, &scorer).unwrap();

        let lines = journal_lines(&result.journal_path);
        let JournalLine::Header(header) = &lines[0] else {
            panic!("first journal line must be the run header");
        };
        assert_eq!(header.seed, 7);
        assert_eq!(header.seed_source, SeedSource::Supplied);
        assert_eq!(header.version, env!("CARGO_PKG_VERSION"));
        let cfg = &header.config;
        assert_eq!(cfg.creature, creature_path);
        assert_eq!(cfg.training_data, training);
        assert_eq!(cfg.timeout_seconds, config.timeout.as_secs());
        assert_eq!(cfg.candidates, 4);
        assert_eq!(cfg.min_improvement, 1e-6);
        assert_eq!(cfg.screen_sample_rate, Some(0.25));
        assert_eq!(cfg.screen_promote_threshold, 1e-7);
        assert_eq!(cfg.focus_neuron.as_deref(), Some("o1"));
        assert_eq!(cfg.focus_policy, "random");
        assert_eq!(cfg.stats_mode, "quick");
        assert_eq!(cfg.quick_sample_records, 8);
        assert!(cfg.structural_only);
        assert!(!cfg.phase0_parity);
        assert_eq!(cfg.grafts_path, None);
        assert_eq!(cfg.max_consecutive_scorer_failures, 3);

        // The header must survive a round-trip through the journal encoding.
        let encoded = serde_json::to_string(header.as_ref()).unwrap();
        assert!(encoded.contains("\"record\":\"runHeader\""));
    }

    /// Issue #71: replaying the recorded seed reproduces the candidate stream.
    #[test]
    fn recorded_seed_replays_the_candidate_stream() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let first_config = reproducibility_config(
            creature_path.clone(),
            training.clone(),
            dir.path().join("out-1"),
        );
        let first = run_optimisation(
            &first_config,
            &ScriptedScorer {
                calls: Arc::new(Mutex::new(0)),
            },
        )
        .unwrap();

        let mut replay_config =
            reproducibility_config(creature_path, training, dir.path().join("out-2"));
        replay_config.seed = Some(first.seed);
        let replay = run_optimisation(
            &replay_config,
            &ScriptedScorer {
                calls: Arc::new(Mutex::new(0)),
            },
        )
        .unwrap();

        let first_experiment = |result: &RunResult| -> ExperimentRecord {
            journal_lines(&result.journal_path)
                .into_iter()
                .find_map(|l| match l {
                    JournalLine::Experiment(record) => Some(*record),
                    JournalLine::Header(_) => None,
                })
                .expect("at least one experiment")
        };
        let a = first_experiment(&first);
        let b = first_experiment(&replay);
        assert_eq!(a.focus_neuron, b.focus_neuron);
        assert_eq!(
            serde_json::to_string(&a.candidates).unwrap(),
            serde_json::to_string(&b.candidates).unwrap(),
            "same seed must regenerate the same candidate stream"
        );
    }

    #[test]
    fn consecutive_scorer_failures_abort() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let config = LamarckConfig {
            creature: creature_path,
            training_data: training,
            timeout: Duration::from_secs(30),
            candidates: 2,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: dir.path().join("out"),
            preserve_losers: true,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            compute_correlations: false,
            max_consecutive_scorer_failures: 2,
            phase0_parity: false,
            structural_only: false,
            screen_sample_rate: None,
            screen_promote_threshold: 0.0,
            grafts_path: None,
            graft_replay_budget: None,
        };
        let err = run_optimisation(&config, &FailingScorer).unwrap_err();
        assert!(err.contains("consecutive scorer failures") || err.contains("no successful"));
    }
}
