//! End-to-end Lamarck optimisation loop and experiment journal.

use crate::analysis::{ScanBudget, scan_post_focus, scan_pre_focus};
use crate::baseline::{
    BaselineKey, BaselineSource, RememberedBaseline, estimate_corpus_records, training_data_key,
};
use crate::cancel::CancelToken;
use crate::candidates::{
    BatchLimit, Candidate, CandidateBudget, CandidateGenContext, CandidateProvenance,
    CandidateStrategy, generate_candidate_batch, strategy_mix_summary, write_candidate_batch,
};
use crate::combos::{
    ComboSelectRequest, ComboSelection, StackDampenReport, select_best_with_combinations,
};
use crate::config::LamarckConfig;
use crate::focus::{
    FixedFocusSelector, FocusChoice, FocusNeuronStats, FocusPolicy, FocusSelector,
    HighErrorFocusSelector, RandomFocusSelector, UnsaturatedFocusSelector, WeightedFocusSelector,
    attach_focus_blame, attach_learning_to_incoming, build_improvement_signals,
    select_highest_signal, select_highest_signal_excluding, select_random_excluding,
    select_unsaturated_excluding,
};
use crate::grafts::{
    GraftReplayRequest, GraftStore, default_graft_replay_budget, record_structural_acceptance,
    replay_grafts,
};
use crate::log;
use crate::memo::{AnalysisMemo, MemoScope};
use crate::observations::ensure_statistics;
use crate::parity::{check_phase0_parity, compute_local_mse};
#[cfg(test)]
use crate::promote_gate::DEFAULT_SCREEN_PROMOTE_SIGMA_K;
use crate::promote_gate::PromoteGateMode;
use crate::scorer::improvement;
use crate::scorer::{
    DirectoryScorer, RecordingScorer, ScoreResult, ScoreSample, accepts_improvement,
    log_scorer_batch_stats_against, log_scorer_batch_stats_labeled, screen_promote_decision,
    write_promote_batch, write_promote_batch_without_baseline,
};
use crate::scorer_cost::{ScorerCallPhase, ScorerCallRecord};
use crate::structural::{is_input_source, rank_unused_sources};
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
    /// Primary focus neuron UUID (the first of [`Self::focus_neurons`]).
    pub focus_neuron: String,
    /// Every focus neuron this experiment proposed against (issue #109).
    ///
    /// Omitted for a single-focus experiment — the pre-#109 shape, where
    /// [`Self::focus_neuron`] already says everything — and absent from
    /// journals written before the field existed. Each candidate's provenance
    /// names the focus it was proposed for, so a winner is attributable to one
    /// member of this set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus_neurons: Option<Vec<String>>,
    /// Focus-neuron structure, activation statistics and backprop blame (#70).
    ///
    /// Omitted only when the focus scan did not run for this experiment (and on
    /// journals written before the field existed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus_stats: Option<FocusNeuronStats>,
    /// Candidate provenances.
    pub candidates: Vec<CandidateProvenance>,
    /// Candidates the run asked the generator for (`--candidates`).
    ///
    /// Recorded with [`Self::batch_limit`] so `report` can show the achieved
    /// batch size against the budget and say which limit bound it (issue #108).
    /// Absent from journals written before the fields existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates_requested: Option<usize>,
    /// Why the batch stopped growing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_limit: Option<BatchLimit>,
    /// Authoritative (full-corpus) scores by stem when a promote/full score ran.
    pub scores: std::collections::BTreeMap<String, f64>,
    /// Screen-phase (subsample) scores by stem when two-phase scoring is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screen_scores: Option<std::collections::BTreeMap<String, f64>>,
    /// What the screen tier and the promote gate did this experiment (#111).
    ///
    /// Omitted when no screen phase ran, and absent from journals written
    /// before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screen_tiers: Option<ScreenTierRecord>,
    /// Which baseline decided this experiment's promote call (issue #113).
    ///
    /// `fresh` when the call carried the incumbent and scored it, `remembered`
    /// when it reused the run's carried full-corpus score. Omitted when no
    /// promote call ran, and absent from journals written before the field
    /// existed — so any accept can be traced to the baseline that decided it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_source: Option<BaselineSource>,
    /// Winning stem if accepted.
    pub winner: Option<String>,
    /// Absolute score improvement when accepted.
    pub improvement: Option<f64>,
    /// Whether a candidate was accepted.
    pub accepted: bool,
    /// Analysis elapsed milliseconds.
    pub analysis_ms: u128,
    /// Analysis-memo lookups served from the memo this experiment (issue #106).
    #[serde(default)]
    pub memo_hits: u64,
    /// Analysis-memo lookups that had to be recomputed this experiment.
    #[serde(default)]
    pub memo_misses: u64,
    /// Training-scan milliseconds avoided by memo hits this experiment.
    ///
    /// Counts whole scans skipped, measured on the miss that stored the entry.
    /// A cached output-MAE map saves only the error accumulation inside a scan
    /// that still has to run for the (uncached) learning signal, so it is
    /// deliberately not counted here — this number never over-claims.
    #[serde(default)]
    pub memo_ms_saved: u128,
    /// Scorer elapsed milliseconds (screen + promote when both ran).
    pub scorer_ms: u128,
    /// Every scorer call this experiment made, with its creature count (#112).
    ///
    /// `scorerMs` alone cannot be regressed into a fixed per-call cost and a
    /// marginal per-creature cost, because it sums calls of different sizes.
    /// The per-call breakdown makes that decomposition reproducible from any
    /// journal. Absent from journals written before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scorer_calls: Option<Vec<ScorerCallRecord>>,
    /// Scorer error message when the batch failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scorer_error: Option<String>,
    /// Member count of the selected combo (`None` / omitted for pure singles).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combo_members: Option<usize>,
    /// Indices into `candidates` of the accepted winner's members (issue #74).
    ///
    /// Recorded for every acceptance — a single is a one-member list — so the
    /// report can attribute a merged `combo-NNN-kM` win to each member's
    /// strategy. Omitted when nothing was accepted, and absent from journals
    /// written before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combo_member_indices: Option<Vec<usize>>,
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

/// Per-experiment admission counts for the screen tier (issue #111).
///
/// Recorded so `report` can price a gate change from the journal alone: an
/// over- or under-promoting gate is visible in the first few experiments of a
/// run rather than at the end of a 45-minute budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScreenTierRecord {
    /// Promote gate in force (`absolute` / `noise-aware`).
    pub gate: String,
    /// Candidates the screen tier scored (baseline excluded).
    pub screened: u64,
    /// Candidates the gate admitted to full-corpus scoring.
    pub promoted: u64,
    /// Screen Δ a candidate had to clear in this batch.
    pub threshold: f64,
    /// σ̂ estimated for this batch; omitted under the absolute gate and when
    /// the batch was too degenerate to price its own noise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma: Option<f64>,
}

/// Marks a journal line as the one-off run header (issue #71).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RunHeaderKind {
    /// This line is a run header, not an experiment.
    RunHeader,
}

/// Marks a journal line as a Phase-G graft-replay outcome (issue #74).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GraftReplayKind {
    /// This line is a graft-replay outcome, not an experiment.
    GraftReplay,
}

/// One journal line recording the Phase-G graft-replay outcome (issue #74).
///
/// Phase-G runs before the experiment loop and can improve the incumbent
/// without any candidate stem, so its accepts have no experiment record to live
/// on. Journalling them keeps `report` from silently dropping the improvement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraftReplayRecord {
    /// Discriminates this line from an [`ExperimentRecord`].
    pub record: GraftReplayKind,
    /// Unix timestamp when the phase finished.
    pub timestamp_unix: u64,
    /// Grafts accepted into the incumbent (0 when the replay changed nothing).
    pub grafts_applied: usize,
    /// Whether the replay improved the incumbent.
    pub accepted: bool,
    /// Incumbent score before the replay, when it was scored.
    pub baseline_score: Option<f64>,
    /// Incumbent score after the replay, when it was scored.
    pub score: Option<f64>,
    /// Absolute score improvement when accepted.
    pub improvement: Option<f64>,
    /// Wall-clock milliseconds spent in the phase.
    pub elapsed_ms: u128,
    /// Scorer batches that succeeded during the phase.
    pub scorer_successes: u64,
    /// Scorer batches that failed during the phase.
    pub scorer_failures: u64,
    /// Every scorer call the phase made, with its creature count (#112).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scorer_calls: Option<Vec<ScorerCallRecord>>,
    /// Failure message when the phase aborted instead of completing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_error: Option<String>,
}

/// Marks a journal line as scorer calls made outside any experiment (#112).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ScorerCallsKind {
    /// This line carries scorer calls, not an experiment.
    ScorerCalls,
}

/// One journal line carrying scorer calls that belong to no experiment (#112).
///
/// The Phase-0 baseline call is the standing example: it runs before the first
/// experiment, so it has no experiment record to live on. Journalling it on its
/// own line is what lets the fixed-cost regression be fitted to *every* call a
/// run made rather than to the subset the experiment loop happened to own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScorerCallsRecord {
    /// Discriminates this line from an [`ExperimentRecord`].
    pub record: ScorerCallsKind,
    /// Unix timestamp when the line was written.
    pub timestamp_unix: u64,
    /// Which part of the run made these calls (`phase0` / `trailing`).
    pub stage: String,
    /// The calls themselves, in the order they were made.
    pub calls: Vec<ScorerCallRecord>,
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
    /// Experiment cap when one was configured (`--max-experiments`).
    #[serde(default)]
    pub max_experiments: Option<u64>,
    /// Candidates generated per experiment.
    pub candidates: usize,
    /// Absolute score delta required for acceptance.
    pub min_improvement: f64,
    /// Screen subsample rate (`None` = full-corpus scoring only).
    pub screen_sample_rate: Option<f64>,
    /// Minimum sample-score delta to promote to full-corpus scoring.
    ///
    /// Under the noise-aware gate this is the gate's absolute floor.
    pub screen_promote_threshold: f64,
    /// Promote gate in force (`absolute` / `noise-aware`, issue #111).
    ///
    /// `None` in journals written before the gate was a knob — those ran the
    /// absolute gate, which is still the default.
    #[serde(default)]
    pub screen_promote_gate: Option<String>,
    /// σ̂ multiplier the noise-aware gate used; `None` under the absolute gate.
    #[serde(default)]
    pub screen_promote_sigma_k: Option<f64>,
    /// Promote calls served from a remembered baseline before one is scored
    /// fresh (`--baseline-reverify-interval`, issue #113).
    ///
    /// `0` — the default, and what a journal written before the knob existed
    /// reports — means every promote call carried the incumbent.
    #[serde(default)]
    pub baseline_reverify_interval: u64,
    /// Baseline score drift that aborts the run (effective value after auto-tune).
    #[serde(default)]
    pub baseline_drift_epsilon: f64,
    /// True when the drift epsilon was auto-tuned (flag omitted).
    #[serde(default)]
    pub baseline_drift_epsilon_auto: bool,
    /// Pinned focus neuron UUID when set.
    pub focus_neuron: Option<String>,
    /// Focus selection policy label.
    pub focus_policy: String,
    /// Focus neurons proposed against per experiment (`--focus-count`, #109).
    ///
    /// `0` in journals written before the knob existed; a real run always
    /// records at least 1.
    #[serde(default)]
    pub focus_count: usize,
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
    /// Backprop learning rate in force for this run (issue #75 A/B arms).
    #[serde(default)]
    pub backprop_learning_rate: Option<f64>,
    /// Backprop bias-step cap in force for this run (issue #96 A/B arm).
    #[serde(default)]
    pub backprop_max_bias_adjustment_scale: Option<f64>,
    /// Analysis-memo entry cap in force for this run (`0` = memo off, #106).
    #[serde(default)]
    pub analysis_memo_entries: usize,
    /// Analysis worker threads in force for this run (issue #107).
    ///
    /// The scans are bit-identical at every thread count, so this identifies a
    /// wall-clock arm — a parallel run that is *slower* than serial is visible
    /// from the journal alone, without guessing what the host chose.
    #[serde(default)]
    pub analysis_threads: usize,
}

impl RunConfigRecord {
    /// Capture the reproducible knobs of a [`LamarckConfig`].
    ///
    /// `drift_epsilon` is the effective tolerance after auto-tune (or the
    /// explicit override). Journal readers always see the number the run used.
    pub fn from_config(config: &LamarckConfig, drift_epsilon: f64) -> Self {
        Self {
            creature: config.creature.clone(),
            training_data: config.training_data.clone(),
            scorer_path: config.scorer_path.clone(),
            timeout_seconds: config.timeout.as_secs(),
            max_experiments: config.max_experiments,
            candidates: config.candidates,
            min_improvement: config.min_improvement,
            screen_sample_rate: config.screen_sample_rate,
            screen_promote_threshold: config.screen_promote_threshold,
            screen_promote_gate: Some(config.screen_promote_gate.label().to_string()),
            // Recorded only when it is actually in force, so a journal never
            // implies a knob the run did not use.
            screen_promote_sigma_k: match config.screen_promote_gate {
                PromoteGateMode::Absolute => None,
                PromoteGateMode::NoiseAware => Some(config.screen_promote_sigma_k),
            },
            baseline_reverify_interval: config.baseline_reverify_interval,
            baseline_drift_epsilon: drift_epsilon,
            baseline_drift_epsilon_auto: config.baseline_drift_epsilon.is_none(),
            focus_neuron: config.focus_neuron.clone(),
            focus_policy: config.focus_policy.label().to_string(),
            focus_count: config.focus_count,
            stats_mode: config.stats_mode.label().to_string(),
            quick_sample_records: config.quick_sample_records,
            compute_correlations: config.compute_correlations,
            structural_only: config.structural_only,
            phase0_parity: config.phase0_parity,
            preserve_losers: config.preserve_losers,
            max_consecutive_scorer_failures: config.max_consecutive_scorer_failures,
            grafts_path: config.grafts_path.clone(),
            graft_replay_budget_seconds: config.graft_replay_budget.map(|d| d.as_secs()),
            backprop_learning_rate: config.backprop_learning_rate,
            backprop_max_bias_adjustment_scale: config.backprop_max_bias_adjustment_scale,
            analysis_memo_entries: config.analysis_memo_entries,
            analysis_threads: config.analysis_threads,
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

/// One line of `experiments.jsonl`: run header, graft replay or experiment.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum JournalLine {
    /// Run header written once at the start of a run.
    Header(Box<RunHeaderRecord>),
    /// Phase-G graft-replay outcome written once before the loop (issue #74).
    GraftReplay(Box<GraftReplayRecord>),
    /// Scorer calls made outside any experiment (issue #112).
    ScorerCalls(Box<ScorerCallsRecord>),
    /// One experiment outcome.
    Experiment(Box<ExperimentRecord>),
}

impl JournalLine {
    /// Parse one journal line, dispatching on the `record` discriminator.
    ///
    /// A line that is none of a valid header, graft replay or experiment is an
    /// error — a malformed journal must never be read as an empty run.
    pub fn parse(line: &str) -> Result<Self, String> {
        #[derive(Deserialize)]
        struct Probe {
            #[serde(default)]
            record: Option<String>,
        }
        let probe: Probe = serde_json::from_str(line).map_err(|e| e.to_string())?;
        match probe.record.as_deref() {
            Some("runHeader") => {
                let header = serde_json::from_str(line).map_err(|e| e.to_string())?;
                Ok(Self::Header(Box::new(header)))
            }
            Some("graftReplay") => {
                let replay = serde_json::from_str(line).map_err(|e| e.to_string())?;
                Ok(Self::GraftReplay(Box::new(replay)))
            }
            Some("scorerCalls") => {
                let calls = serde_json::from_str(line).map_err(|e| e.to_string())?;
                Ok(Self::ScorerCalls(Box::new(calls)))
            }
            Some(other) => Err(format!("unknown journal record kind: {other}")),
            None => {
                let record = serde_json::from_str(line).map_err(|e| e.to_string())?;
                Ok(Self::Experiment(Box::new(record)))
            }
        }
    }
}

/// Why the optimisation loop stopped (issue #72).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    /// The wall-clock budget expired.
    Timeout,
    /// The configured experiment cap was reached.
    MaxExperiments,
    /// `SIGINT`/`SIGTERM` (or a programmatic cancel) asked the run to stop.
    Cancelled,
}

impl StopReason {
    /// Stable lower-case label for logs and journals.
    pub fn label(self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::MaxExperiments => "max-experiments",
            Self::Cancelled => "cancelled",
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
    /// Which stopping rule ended the loop.
    pub stop_reason: StopReason,
}

/// The focus selectors an optimisation run keeps across experiments.
///
/// Grouped so the multi-focus draw (issue #109) is one call rather than a
/// policy `match` repeated per focus.
struct FocusSelectors {
    random: RandomFocusSelector,
    unsaturated: UnsaturatedFocusSelector,
    high_error: HighErrorFocusSelector,
    weighted: WeightedFocusSelector,
    fixed: Option<FixedFocusSelector>,
}

impl FocusSelectors {
    /// Select up to `focus_count` distinct focus neurons for one experiment.
    ///
    /// The first focus is drawn exactly as the single-focus path always drew
    /// it, so `focus_count == 1` consumes the same rng stream as before #109.
    /// A pinned `--focus-neuron` yields exactly that neuron. Fewer than
    /// `focus_count` focuses is legitimate — a creature can run out of neurons
    /// carrying a non-zero improvement signal.
    fn select_focus_set(
        &mut self,
        creature: &neat_core::CreatureExport,
        policy: FocusPolicy,
        focus_count: usize,
        signals: &std::collections::HashMap<String, f64>,
        rng: &mut StdRng,
    ) -> Result<Vec<String>, String> {
        if let Some(selector) = self.fixed.as_mut() {
            let uuid = selector.select(creature, rng).ok_or_else(|| {
                format!(
                    "focus neuron '{}' not found (or is an input)",
                    selector.uuid
                )
            })?;
            log::detail(&format!("focus neuron: {uuid} (pinned)"));
            return Ok(vec![uuid]);
        }

        let mut chosen = vec![self.select_primary(creature, policy, signals, rng)?];
        while chosen.len() < focus_count {
            let Some(choice) = self.select_additional(creature, policy, signals, &chosen, rng)
            else {
                log::detail(&format!(
                    "focus set: {} of {focus_count} focuses available on this creature",
                    chosen.len()
                ));
                break;
            };
            log::detail(&format!(
                "focus neuron {}: {} ({})",
                chosen.len() + 1,
                choice.uuid,
                choice.reason
            ));
            chosen.push(choice.uuid);
        }
        Ok(chosen)
    }

    /// Draw the experiment's primary focus (the pre-#109 selection).
    fn select_primary(
        &mut self,
        creature: &neat_core::CreatureExport,
        policy: FocusPolicy,
        signals: &std::collections::HashMap<String, f64>,
        rng: &mut StdRng,
    ) -> Result<String, String> {
        match policy {
            FocusPolicy::Weighted => {
                let choice = self
                    .weighted
                    .select_weighted(creature, signals, rng)
                    .or_else(|| {
                        log::warn("no non-zero improvement signal; falling back to first output");
                        self.high_error
                            .select(creature, rng)
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
                Ok(choice.uuid)
            }
            FocusPolicy::HighError => {
                let choice = select_highest_signal(signals).or_else(|| {
                    log::warn("no non-zero improvement signal; falling back to first output");
                    self.high_error
                        .select(creature, rng)
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
                Ok(choice.uuid)
            }
            FocusPolicy::Random => {
                let uuid = self
                    .random
                    .select(creature, rng)
                    .ok_or_else(|| "no focus neuron available".to_string())?;
                log::detail(&format!("focus neuron: {uuid}"));
                Ok(uuid)
            }
            FocusPolicy::Unsaturated => {
                let uuid = self
                    .unsaturated
                    .select(creature, rng)
                    .ok_or_else(|| "no focus neuron available".to_string())?;
                log::detail(&format!("focus neuron: {uuid}"));
                Ok(uuid)
            }
        }
    }

    /// Draw one further focus under `policy`, skipping the already-chosen set.
    fn select_additional(
        &mut self,
        creature: &neat_core::CreatureExport,
        policy: FocusPolicy,
        signals: &std::collections::HashMap<String, f64>,
        chosen: &[String],
        rng: &mut StdRng,
    ) -> Option<FocusChoice> {
        match policy {
            FocusPolicy::Weighted => self
                .weighted
                .select_weighted_excluding(creature, signals, chosen, rng),
            FocusPolicy::HighError => select_highest_signal_excluding(signals, chosen),
            FocusPolicy::Random => {
                select_random_excluding(creature, chosen, rng).map(|uuid| FocusChoice {
                    uuid,
                    weight: 0.0,
                    reason: "random".into(),
                })
            }
            FocusPolicy::Unsaturated => {
                select_unsaturated_excluding(creature, chosen, rng).map(|uuid| FocusChoice {
                    uuid,
                    weight: 0.0,
                    reason: "unsaturated".into(),
                })
            }
        }
    }
}

/// Split a candidate budget across `focuses`, largest shares first.
///
/// A single focus keeps the whole budget, so `K = 1` proposes exactly the batch
/// it did before #109. Any remainder goes to the earliest (highest-ranked)
/// focuses rather than being dropped.
fn split_candidate_budget(count: usize, focuses: usize) -> Vec<usize> {
    if focuses == 0 {
        return Vec::new();
    }
    let base = count / focuses;
    let remainder = count % focuses;
    (0..focuses)
        .map(|i| base + usize::from(i < remainder))
        .collect()
}

/// Why a merged multi-focus batch stopped growing (issue #109).
///
/// The batch is one journal line, so the per-focus limits collapse to the one
/// that actually bound it: budget only when every focus filled its share, and
/// otherwise the strictest ceiling any focus hit. A generator that ran dry for
/// one focus must not be reported as a satisfied budget.
fn merge_batch_limits(limits: &[BatchLimit]) -> BatchLimit {
    if limits.contains(&BatchLimit::QuotaCeiling) {
        BatchLimit::QuotaCeiling
    } else if limits.contains(&BatchLimit::Exhausted) {
        BatchLimit::Exhausted
    } else {
        BatchLimit::Budget
    }
}

/// Index of a `candidate-NNN` stem, or `None` for any other stem.
fn candidate_stem_index(stem: &str) -> Option<usize> {
    stem.strip_prefix("candidate-")?.parse().ok()
}

/// Run the Lamarck optimisation loop with no external cancellation.
///
/// Equivalent to [`run_optimisation_cancellable`] with a token that is never
/// cancelled: the run stops on the wall-clock budget or the experiment cap.
pub fn run_optimisation(
    config: &LamarckConfig,
    scorer: &impl DirectoryScorer,
) -> Result<RunResult, String> {
    run_optimisation_cancellable(config, scorer, &CancelToken::new())
}

/// Run the Lamarck optimisation loop until a stopping rule fires (issue #72).
///
/// The loop ends on the first of: `cancel` being set (`SIGINT`/`SIGTERM` when
/// the caller installed the handlers), the configured experiment cap, or the
/// wall-clock budget. Cancellation abandons the in-flight experiment before its
/// scorer batch rather than killing the process, so `best.json` is still
/// re-stamped with the run summary and the run summary is still returned.
pub fn run_optimisation_cancellable(
    config: &LamarckConfig,
    scorer: &impl DirectoryScorer,
    cancel: &CancelToken,
) -> Result<RunResult, String> {
    // Validate before the expensive Phase-0 gate: a bad learning rate must stop
    // the run, never be silently replaced by the default mid-A/B.
    let backprop = config.backprop_config()?;
    let analysis_threads = config.analysis_threads()?;
    let focus_count = config.focus_count()?;
    let promote_gate = config.promote_gate()?;
    // Measure every scorer call at the boundary (issue #112): the wrapper sees
    // Phase-0, Phase-G, screen, promote and combo batches alike, so the journal
    // can never be fitted to a subset of the calls a run actually made.
    let recorder = RecordingScorer::new(scorer);
    let scorer = &recorder;
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

    let original_text = fs::read_to_string(&config.creature).map_err(|e| e.to_string())?;
    let mut incumbent = parse_creature_json(&original_text).map_err(|e| e.to_string())?;
    // Tags/uuid are stripped by CreatureExport — keep them for check-in writes.
    let mut creature_meta = CreatureMeta::from_creature_json(&original_text);
    // Never modify the supplied file — work from in-memory / output copies.
    fs::write(&best_path, &original_text).map_err(|e| e.to_string())?;

    let train_cfg = TrainingDataConfig::new(incumbent.input, incumbent.output);
    // Auto-tune the drift canary from the corpus before the journal header so
    // the recorded epsilon is the one the run will use (refined after Phase-0).
    let approx_records =
        estimate_corpus_records(&config.training_data, train_cfg.bytes_per_record())?;
    let mut baseline_policy = config.baseline_reuse_policy(approx_records, 1.0)?;
    if config.baseline_drift_epsilon.is_none() {
        log::info(&format!(
            "baseline-drift-epsilon={:.6e} (auto from ~{approx_records} records; refined after Phase-0)",
            baseline_policy.drift_epsilon
        ));
    } else {
        log::info(&format!(
            "baseline-drift-epsilon={:.6e} (explicit override)",
            baseline_policy.drift_epsilon
        ));
    }
    append_journal_line(
        &journal_path,
        &RunHeaderRecord::new(
            seed,
            seed_source,
            RunConfigRecord::from_config(config, baseline_policy.drift_epsilon),
            unix_now(),
        ),
    )?;
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
    // The authoritative full-corpus baseline the run carries (issue #113).
    // Established here by Phase-0 and re-established by every fresh promote
    // call; only ever handed to a promote call whose creature and corpus still
    // match the key it was measured under.
    let mut remembered_baseline: Option<RememberedBaseline> = None;
    if config.phase0_parity {
        log::info("Phase-0: scoring incumbent baseline via authoritative scorer");
        scorer.set_phase(ScorerCallPhase::Phase0);
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
                // Refine the auto-tuned canary with the true record count and
                // the scorer's error scale — still owned by Lamarck, never GRQ.
                if config.baseline_drift_epsilon.is_none() {
                    let refined = config.baseline_reuse_policy(local_count, baseline.error)?;
                    if (refined.drift_epsilon - baseline_policy.drift_epsilon).abs() > 0.0 {
                        log::info(&format!(
                            "baseline-drift-epsilon={:.6e} (auto refined: {local_count} records, error={:.6})",
                            refined.drift_epsilon, baseline.error
                        ));
                    }
                    baseline_policy = refined;
                }
                // Phase-0 scored the incumbent on the full corpus, so the run
                // starts already knowing the number every promote call would
                // otherwise re-derive (issue #113). Only carry it when reuse is
                // enabled — with the default interval=0 each promote is already
                // self-paired, and retaining a canary across promotes false-
                // aborted healthy GRQ runs on directory-scorer association noise.
                if baseline_policy.is_enabled()
                    && let Some(key) = baseline_key(&incumbent, &config.training_data)
                {
                    remembered_baseline = Some(RememberedBaseline::new(key, baseline.clone()));
                }
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
        // The Phase-0 call belongs to no experiment, so it gets its own journal
        // line — otherwise the fixed-cost regression misses it (issue #112).
        journal_scorer_calls(&journal_path, "phase0", scorer.drain())?;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut selectors = FocusSelectors {
        random: RandomFocusSelector,
        unsaturated: UnsaturatedFocusSelector,
        high_error: HighErrorFocusSelector,
        weighted: WeightedFocusSelector::default(),
        fixed: config
            .focus_neuron
            .as_ref()
            .map(|uuid| FixedFocusSelector { uuid: uuid.clone() }),
    };
    let focus_sample_limit = match config.stats_mode {
        crate::observations::StatsMode::Quick => Some(config.quick_sample_records),
        crate::observations::StatsMode::Full => None,
    };

    // Cross-experiment analysis memo (issue #106). The sample key pins every
    // input the cached scans read besides the creature itself, so a run that
    // changed `--quick-sample-records` mid-flight could never hit a stale entry.
    let analysis_sample_key = format!(
        "{}:{}:{}:{}",
        config.stats_mode.label(),
        focus_sample_limit.map_or_else(|| "all".to_string(), |n| n.to_string()),
        config.training_data.display(),
        config.compute_correlations
    );
    let mut analysis_memo = AnalysisMemo::new(config.analysis_memo_entries);
    if analysis_memo.is_enabled() {
        log::detail(&format!(
            "analysis memo: up to {} focus entries per incumbent",
            analysis_memo.capacity()
        ));
    } else {
        log::detail("analysis memo: disabled (--analysis-memo-entries 0)");
    }
    log::detail(&format!(
        "analysis scans: {analysis_threads} worker thread(s), {} records per chunk",
        crate::chunks::ANALYSIS_CHUNK_RECORDS
    ));

    let deadline = Instant::now() + config.timeout;
    let mut experiments = 0u64;
    let mut acceptances = 0u64;
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
        let graft_start = Instant::now();
        scorer.set_phase(ScorerCallPhase::GraftReplay);
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
                let mut graft_accepted = false;
                if replay.grafts_applied > 0 {
                    incumbent = replay.creature;
                    // Phase-G rewrites the incumbent before the loop starts —
                    // anything cached against the pre-graft creature is stale.
                    analysis_memo.invalidate();
                    if let (Some(score), Some(error)) = (replay.score, replay.error) {
                        best_score = score;
                        last_accept_error = error;
                        last_accept_focus = "(graft-replay)".into();
                        last_accept_strategy = CandidateStrategy::StructuralAdd;
                        acceptances += 1;
                        graft_accepted = true;
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
                // Phase-G accepts have no candidate stem, so they need their own
                // journal line or the report drops them entirely (issue #74).
                let improvement = match (graft_accepted, replay.score, replay.baseline_score) {
                    (true, Some(score), Some(base)) => Some(score - base),
                    _ => None,
                };
                append_journal_line(
                    &journal_path,
                    &GraftReplayRecord {
                        record: GraftReplayKind::GraftReplay,
                        timestamp_unix: unix_now(),
                        grafts_applied: replay.grafts_applied,
                        accepted: graft_accepted,
                        baseline_score: replay.baseline_score,
                        score: replay.score,
                        improvement,
                        elapsed_ms: graft_start.elapsed().as_millis(),
                        scorer_successes: replay.scorer_successes,
                        scorer_failures: replay.scorer_failures,
                        scorer_calls: journal_calls(scorer.drain()),
                        replay_error: None,
                    },
                )?;
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
                append_journal_line(
                    &journal_path,
                    &GraftReplayRecord {
                        record: GraftReplayKind::GraftReplay,
                        timestamp_unix: unix_now(),
                        grafts_applied: 0,
                        accepted: false,
                        baseline_score: opening_baseline_score,
                        score: None,
                        improvement: None,
                        elapsed_ms: graft_start.elapsed().as_millis(),
                        scorer_successes: 0,
                        scorer_failures: 0,
                        scorer_calls: journal_calls(scorer.drain()),
                        replay_error: Some(e.message.clone()),
                    },
                )?;
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
        "starting optimisation loop (timeout={}s, candidates={}, max_experiments={})",
        config.timeout.as_secs(),
        config.candidates,
        config
            .max_experiments
            .map_or_else(|| "unlimited".to_string(), |max| max.to_string())
    ));
    if let Some(uuid) = &config.focus_neuron {
        log::detail(&format!("focus locked to {uuid}"));
        if focus_count > 1 {
            log::warn(&format!(
                "--focus-count {focus_count} ignored: --focus-neuron pins the experiment to one focus"
            ));
        }
    } else {
        log::detail(&format!(
            "focus policy: {} ({focus_count} focus neuron(s) per experiment)",
            config.focus_policy.label()
        ));
    }

    let stop_reason = loop {
        // Stopping rules, cheapest and most urgent first.
        if cancel.is_cancelled() {
            log::warn("cancellation requested — stopping before the next experiment");
            break StopReason::Cancelled;
        }
        if let Some(max) = config.max_experiments
            && experiments >= max
        {
            log::info(&format!("experiment cap reached ({max}) — stopping"));
            break StopReason::MaxExperiments;
        }
        if Instant::now() >= deadline {
            break StopReason::Timeout;
        }
        experiments += 1;
        let remaining = deadline.saturating_duration_since(Instant::now());
        log::info(&format!(
            "experiment {experiments} ({}s remaining, acceptances={acceptances})",
            remaining.as_secs()
        ));
        let analysis_start = Instant::now();
        let mut network = compile_creature(&incumbent).map_err(|e| e.to_string())?;
        // Memo scope for this experiment's incumbent (issue #106). Building it
        // fresh from the creature every experiment is what makes a missed
        // invalidation impossible: a changed creature is a changed scope, and a
        // changed scope drops every entry before the first lookup is answered.
        let memo_scope = MemoScope::new(incumbent_id(&incumbent), &incumbent, &analysis_sample_key);
        let memo_before = analysis_memo.stats();

        // Learning + output MAE first so weighted/high-error can rank by
        // improvement chance and skip zero-error / zero-blame neurons. Both are
        // focus-independent, so one scan feeds them (issue #105).
        let needs_signals = selectors.fixed.is_none()
            && matches!(
                config.focus_policy,
                FocusPolicy::Weighted | FocusPolicy::HighError
            );
        // The output MAE is a pure function of `(incumbent, sample)`; the
        // learning signal is rng-driven and deliberately never cached, so the
        // scan still runs — the hit skips the per-record error accumulation.
        let cached_output_errors = if needs_signals {
            analysis_memo.output_errors(&memo_scope)
        } else {
            None
        };
        log::detail("pre-focus scan: creature learning signal + output residuals...");
        let learn_start = Instant::now();
        let mut learn_rng = StdRng::seed_from_u64(
            seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(experiments),
        );
        let pre_focus = scan_pre_focus(
            &incumbent,
            &mut network,
            &config.training_data,
            &backprop,
            ScanBudget::new(focus_sample_limit, analysis_threads),
            &mut learn_rng,
            needs_signals && cached_output_errors.is_none(),
        )?;
        let learning = pre_focus.learning;
        log::ok(&format!(
            "learning signal in {}ms (bias_count={:.0}, weight_count={:.0})",
            learn_start.elapsed().as_millis(),
            learning.biases.iter().map(|b| b.count).sum::<f64>(),
            learning.weights.iter().map(|w| w.count).sum::<f64>()
        ));
        let output_errors = match cached_output_errors {
            Some(cached) => {
                log::detail("memo hit: output residuals reused from the last experiment");
                cached
            }
            None => {
                if needs_signals {
                    analysis_memo.store_output_errors(
                        &memo_scope,
                        pre_focus.output_errors.clone(),
                        0,
                    );
                }
                pre_focus.output_errors
            }
        };

        let improvement_signals = if needs_signals {
            let signals = build_improvement_signals(&incumbent, &output_errors, &learning);
            log::detail(&format!(
                "error-influence signals: {} eligible neurons",
                signals.len()
            ));
            signals
        } else {
            std::collections::HashMap::new()
        };

        // Focus set for this experiment (issue #109). The creature-wide passes
        // above ran once; each focus below pays only its own focus-specific
        // scan, so K focuses amortise the expensive analysis over K proposals
        // instead of one.
        let focus_set = selectors.select_focus_set(
            &incumbent,
            config.focus_policy,
            focus_count,
            &improvement_signals,
            &mut rng,
        )?;
        let focus = focus_set
            .first()
            .cloned()
            .ok_or_else(|| "no focus neuron available".to_string())?;
        // Runtime key check: whatever the memo still holds must describe the
        // creature about to be analysed. In a debug/test build a missed
        // invalidation panics here instead of quietly proposing against the
        // wrong creature.
        if let Some(scope) = analysis_memo.scope() {
            debug_assert_eq!(
                scope.incumbent_id,
                incumbent_id(&incumbent),
                "analysis memo is keyed to a different incumbent"
            );
            debug_assert_eq!(
                scope.fingerprint,
                crate::memo::creature_fingerprint(&incumbent),
                "analysis memo is keyed to a different creature revision"
            );
        }
        let lr = crate::backprop::calculate_learning_rate(&backprop, experiments, None);
        if config.structural_only {
            log::detail("structural-only: synapse/neuron growth candidates only");
        }
        let budgets = split_candidate_budget(config.candidates, focus_set.len());
        let mut candidates: Vec<Candidate> = Vec::with_capacity(config.candidates);
        let mut focus_limits: Vec<BatchLimit> = Vec::with_capacity(focus_set.len());
        let mut primary_focus_stats: Option<FocusNeuronStats> = None;
        let mut generate_ms = 0u128;
        for (index, focus_uuid) in focus_set.iter().enumerate() {
            // Focus stats, incoming-source stats and the residual source ranking
            // all need the focus uuid and nothing else — one scan feeds all
            // three (#105). All three are pure functions of
            // `(incumbent, focus, sample)`, so a repeated focus on an unchanged
            // incumbent skips this scan outright (issue #106).
            let prior_sources = rank_unused_sources(&incumbent, focus_uuid, &observations);
            let focus_scan_start = Instant::now();
            let (post_focus, post_focus_from_memo) =
                match analysis_memo.post_focus(&memo_scope, focus_uuid) {
                    Some(cached) => (cached, true),
                    None => {
                        log::detail(
                            "post-focus scan: focus stats + incoming sources + residual ranking...",
                        );
                        let scan = scan_post_focus(
                            &incumbent,
                            &mut network,
                            &config.training_data,
                            focus_uuid,
                            ScanBudget::new(focus_sample_limit, analysis_threads),
                            Some(&observations),
                            &prior_sources,
                        )?;
                        analysis_memo.store_post_focus(
                            &memo_scope,
                            focus_uuid,
                            scan.clone(),
                            focus_scan_start.elapsed().as_millis(),
                        );
                        (scan, false)
                    }
                };
            let mut focus_stats = post_focus.focus_stats;
            let mut incoming = post_focus.incoming;
            let ranked_sources = post_focus.ranked_sources;
            let focus_scan_ms = focus_scan_start.elapsed().as_millis();
            if post_focus_from_memo {
                log::ok(&format!(
                    "memo hit: focus scan for {focus_uuid} reused ({} records, no training scan)",
                    focus_stats.record_count
                ));
            } else {
                log::ok(&format!(
                    "focus scan {focus_uuid}: {} records in {focus_scan_ms}ms",
                    focus_stats.record_count
                ));
            }
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

            log::detail(&format!("incoming sources: {}", incoming.len()));

            // Surface focus blame + incoming weight signals from real backprop (#4).
            attach_focus_blame(&mut focus_stats, &incumbent, &learning);
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

            let gen_ctx = CandidateGenContext {
                incumbent: &incumbent,
                focus_uuid,
                focus_stats: &focus_stats,
                incoming: &incoming,
                observations: &observations,
                ranked_sources: Some(&ranked_sources),
                learning: Some(&learning),
                backprop: &backprop,
                structural_only: config.structural_only,
            };
            let gen_start = Instant::now();
            let batch = generate_candidate_batch(
                &gen_ctx,
                CandidateBudget {
                    count: budgets[index],
                    scale_quotas: config.scale_candidate_quotas,
                },
                &mut rng,
            );
            generate_ms += gen_start.elapsed().as_millis();
            if focus_set.len() > 1 {
                log::detail(&format!(
                    "focus {focus_uuid}: {} of {} candidates — {}",
                    batch.candidates.len(),
                    budgets[index],
                    batch.limit.label()
                ));
            }
            focus_limits.push(batch.limit);
            candidates.extend(batch.candidates);
            if index == 0 {
                primary_focus_stats = Some(focus_stats);
            }
        }
        let batch_limit = merge_batch_limits(&focus_limits);
        let focus_stats = primary_focus_stats
            .ok_or_else(|| "no focus neuron produced an analysis".to_string())?;
        let strategy_mix = strategy_mix_summary(&candidates);
        let analysis_ms = analysis_start.elapsed().as_millis();
        // Journalled so the memo's value is auditable from experiments.jsonl,
        // the same way the scan and scorer economics already are (issue #106).
        let memo_delta = analysis_memo.stats().since(memo_before);
        if analysis_memo.is_enabled() {
            log::detail(&format!(
                "analysis memo: {} hit(s), {} miss(es), {}ms saved ({} entries)",
                memo_delta.hits,
                memo_delta.misses,
                memo_delta.ms_saved,
                analysis_memo.len()
            ));
        }
        log::ok(&format!(
            "generated {} of {} requested candidates across {} focus neuron(s) in {generate_ms}ms — {} (analysis total {analysis_ms}ms)",
            candidates.len(),
            config.candidates,
            focus_set.len(),
            batch_limit.label()
        ));
        log::detail(&format!("batch strategy mix: {strategy_mix}"));
        // An under-filled batch says which limit bound it, so a generator that
        // quietly runs dry is visible in the log rather than inferred (#108).
        if candidates.len() < config.candidates {
            match batch_limit {
                BatchLimit::QuotaCeiling => log::warn(&format!(
                    "candidate budget {} unmet: the fixed per-phase quotas topped out at {} \
                     — drop --fixed-candidate-quotas to let the budget bind",
                    config.candidates,
                    candidates.len()
                )),
                BatchLimit::Exhausted => log::warn(&format!(
                    "candidate budget {} unmet: the generator is exhausted at {} \
                     (every ranked source and squash proposed)",
                    config.candidates,
                    candidates.len()
                )),
                BatchLimit::Budget => {}
            }
        }

        // Scoring dominates an experiment, so poll here: a signal arriving
        // during analysis abandons this experiment before any working
        // directory is written, instead of waiting out a full scorer batch.
        if cancel.is_cancelled() {
            log::warn(&format!(
                "cancellation requested — abandoning experiment {experiments} before scoring"
            ));
            experiments = experiments.saturating_sub(1);
            break StopReason::Cancelled;
        }

        let batch_dir = config
            .output_dir
            .join(format!("candidates-exp-{experiments}"));
        write_candidate_batch(&batch_dir, &incumbent, &candidates, Some(&creature_meta))?;

        let screen_rate = config.screen_sample_rate.filter(|r| *r > 0.0 && *r < 1.0);
        let scorer_start = Instant::now();
        let mut screen_score_map: Option<std::collections::BTreeMap<String, f64>> = None;
        let mut screen_tiers: Option<ScreenTierRecord> = None;
        let mut promote_dir: Option<PathBuf> = None;
        // Identity any remembered baseline must still match this experiment
        // (issue #113): a changed creature or a changed corpus invalidates it.
        let baseline_scope = baseline_key(&incumbent, &config.training_data);
        let mut promote_reuse: Option<ScoreResult> = None;
        let mut experiment_scorer_error: Option<String> = None;

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
            scorer.set_phase(ScorerCallPhase::Screen);
            let screen_scores = match scorer.score_directory_sampled(
                &batch_dir,
                &config.training_data,
                sample,
            ) {
                Ok(s) => s,
                Err(e) => {
                    consecutive_scorer_failures += 1;
                    log::warn(&format!("screen scorer failed: {e}"));
                    record_focus_failure(
                        &mut selectors.weighted,
                        &focus_set,
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
                            focus_neuron: focus.clone(),
                            focus_neurons: journal_focus_set(&focus_set),
                            focus_stats: Some(focus_stats.clone()),
                            candidates: candidates.iter().map(|c| c.provenance.clone()).collect(),
                            candidates_requested: Some(config.candidates),
                            batch_limit: Some(batch_limit),
                            scores: Default::default(),
                            screen_scores: None,
                            screen_tiers: None,
                            baseline_source: None,
                            winner: None,
                            improvement: None,
                            accepted: false,
                            analysis_ms,
                            memo_hits: memo_delta.hits,
                            memo_misses: memo_delta.misses,
                            memo_ms_saved: memo_delta.ms_saved,
                            scorer_ms: scorer_start.elapsed().as_millis(),
                            scorer_calls: journal_calls(scorer.drain()),
                            scorer_error: Some(e.to_string()),
                            combo_members: None,
                            combo_member_indices: None,
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
            let decision = screen_promote_decision(&screen_scores, &promote_gate)
                .map_err(|e| e.to_string())?;
            // Report the batch against the threshold the gate actually applied,
            // so the ">threshold" count cannot disagree with what is promoted.
            log_scorer_batch_stats_labeled(&screen_scores, screen_ms, decision.threshold, "screen");
            consecutive_scorer_failures = 0;
            screen_score_map = Some(
                screen_scores
                    .iter()
                    .map(|(k, v)| (k.clone(), v.score))
                    .collect(),
            );

            let promote_stems = decision.stems.clone();
            screen_tiers = Some(ScreenTierRecord {
                gate: promote_gate.label().to_string(),
                screened: decision.screened as u64,
                promoted: promote_stems.len() as u64,
                threshold: decision.threshold,
                sigma: decision.sigma,
            });
            if promote_stems.is_empty() {
                log::detail("screen empty: no sample improvers → skipping full-corpus score");
                // Sterile focus: dampen so weighted draw explores elsewhere.
                for focus_uuid in &focus_set {
                    selectors.weighted.record_outcome(
                        focus_uuid,
                        false,
                        Some(0.0),
                        false,
                        config.min_improvement,
                    );
                }
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
                        focus_neuron: focus.clone(),
                        focus_neurons: journal_focus_set(&focus_set),
                        focus_stats: Some(focus_stats.clone()),
                        candidates: candidates.iter().map(|c| c.provenance.clone()).collect(),
                        candidates_requested: Some(config.candidates),
                        batch_limit: Some(batch_limit),
                        scores: Default::default(),
                        screen_scores: screen_score_map,
                        screen_tiers,
                        baseline_source: None,
                        winner: None,
                        improvement: None,
                        accepted: false,
                        analysis_ms,
                        memo_hits: memo_delta.hits,
                        memo_misses: memo_delta.misses,
                        memo_ms_saved: memo_delta.ms_saved,
                        scorer_ms: screen_ms,
                        scorer_calls: journal_calls(scorer.drain()),
                        scorer_error: None,
                        combo_members: None,
                        combo_member_indices: None,
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
                "promote: {} of {} candidate(s) cleared the {} screen gate (Δ > {:.6e}{}) → full corpus",
                promote_stems.len(),
                decision.screened,
                promote_gate.label(),
                decision.threshold,
                match decision.sigma {
                    Some(sigma) => format!(", σ̂ {sigma:.6e}"),
                    None => String::new(),
                }
            ));
            let pdir = config.output_dir.join(format!("promote-exp-{experiments}"));
            // Issue #113: between accepts the incumbent's full-corpus score is
            // a constant the run already holds, so a promote call with a valid
            // remembered score spends the whole call on candidates.
            promote_reuse = match (&remembered_baseline, &baseline_scope) {
                (Some(remembered), Some(scope)) if remembered.may_serve(scope, baseline_policy) => {
                    Some(remembered.result().clone())
                }
                _ => None,
            };
            match &promote_reuse {
                Some(remembered) => {
                    log::detail(&format!(
                        "promote: reusing the remembered full-corpus baseline score={:.12} \
                         (not re-scored this call)",
                        remembered.score
                    ));
                    write_promote_batch_without_baseline(&pdir, &batch_dir, &promote_stems)?;
                }
                None => write_promote_batch(&pdir, &batch_dir, &promote_stems)?,
            }
            promote_dir = Some(pdir.clone());

            let promote_start = Instant::now();
            scorer.set_phase(ScorerCallPhase::Promote);
            match scorer.score_directory(&pdir, &config.training_data) {
                Ok(s) => {
                    let promote_ms = promote_start.elapsed().as_millis();
                    log_scorer_batch_stats_against(
                        &s,
                        s.get("baseline").or(promote_reuse.as_ref()),
                        promote_ms,
                        config.min_improvement,
                        "promote",
                    );
                    s
                }
                Err(e) => {
                    consecutive_scorer_failures += 1;
                    log::warn(&format!("promote scorer failed: {e}"));
                    record_focus_failure(
                        &mut selectors.weighted,
                        &focus_set,
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
                            focus_neuron: focus.clone(),
                            focus_neurons: journal_focus_set(&focus_set),
                            focus_stats: Some(focus_stats.clone()),
                            candidates: candidates.iter().map(|c| c.provenance.clone()).collect(),
                            candidates_requested: Some(config.candidates),
                            batch_limit: Some(batch_limit),
                            scores: Default::default(),
                            screen_scores: screen_score_map,
                            screen_tiers,
                            baseline_source: None,
                            winner: None,
                            improvement: None,
                            accepted: false,
                            analysis_ms,
                            memo_hits: memo_delta.hits,
                            memo_misses: memo_delta.misses,
                            memo_ms_saved: memo_delta.ms_saved,
                            scorer_ms: scorer_start.elapsed().as_millis(),
                            scorer_calls: journal_calls(scorer.drain()),
                            scorer_error: Some(e.to_string()),
                            combo_members: None,
                            combo_member_indices: None,
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
            scorer.set_phase(ScorerCallPhase::Promote);
            match scorer.score_directory(&batch_dir, &config.training_data) {
                Ok(s) => {
                    let full_ms = scorer_start.elapsed().as_millis();
                    log_scorer_batch_stats_labeled(&s, full_ms, config.min_improvement, "scorer");
                    s
                }
                Err(e) => {
                    consecutive_scorer_failures += 1;
                    log::warn(&format!("scorer failed: {e}"));
                    record_focus_failure(
                        &mut selectors.weighted,
                        &focus_set,
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
                            focus_neuron: focus.clone(),
                            focus_neurons: journal_focus_set(&focus_set),
                            focus_stats: Some(focus_stats.clone()),
                            candidates: candidates.iter().map(|c| c.provenance.clone()).collect(),
                            candidates_requested: Some(config.candidates),
                            batch_limit: Some(batch_limit),
                            scores: Default::default(),
                            screen_scores: None,
                            screen_tiers: None,
                            baseline_source: None,
                            winner: None,
                            improvement: None,
                            accepted: false,
                            analysis_ms,
                            memo_hits: memo_delta.hits,
                            memo_misses: memo_delta.misses,
                            memo_ms_saved: memo_delta.ms_saved,
                            scorer_ms: scorer_start.elapsed().as_millis(),
                            scorer_calls: journal_calls(scorer.drain()),
                            scorer_error: Some(e.to_string()),
                            combo_members: None,
                            combo_member_indices: None,
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
        // Screen + promote, as before #113; the accept-path verification call
        // is added to it below when it runs.
        let mut scorer_ms = scorer_start.elapsed().as_millis();

        // Resolve the full-corpus baseline this experiment is judged against
        // (issue #113). A fresh score is also the re-verification of any score
        // the run was carrying: two different numbers for the same creature on
        // the same corpus is the state that lands a false accept, so it aborts
        // the run rather than deciding anything.
        let (mut baseline, mut baseline_source) = match scores.get("baseline") {
            Some(fresh) => {
                // Drift canary + remembered update only when reuse is enabled.
                // Default interval=0 scores the incumbent in every promote batch
                // (self-verifying); carrying a cross-promote canary there is a
                // footgun against directory-scorer f32 association noise.
                if baseline_policy.is_enabled() {
                    if let (Some(remembered), Some(scope)) = (&remembered_baseline, &baseline_scope)
                        && remembered.is_valid_for(scope)
                    {
                        let drift = remembered.drift(fresh);
                        if drift > baseline_policy.drift_epsilon {
                            return Err(baseline_drift_error(
                                drift,
                                remembered.result().score,
                                fresh.score,
                                baseline_policy.drift_epsilon,
                                "promote re-verification",
                            ));
                        }
                    }
                    remembered_baseline = baseline_scope
                        .clone()
                        .map(|scope| RememberedBaseline::new(scope, fresh.clone()));
                } else {
                    remembered_baseline = None;
                }
                (fresh.clone(), BaselineSource::Fresh)
            }
            None => {
                // Reached only when this run deliberately omitted the baseline;
                // a scorer that silently dropped it must never be read as a
                // score of zero, which would promote everything.
                let remembered = promote_reuse.clone().ok_or_else(|| {
                    "baseline missing from scorer results and no remembered score is valid"
                        .to_string()
                })?;
                if let Some(carried) = remembered_baseline.as_mut() {
                    carried.note_reuse();
                }
                (remembered, BaselineSource::Remembered)
            }
        };
        if best_score.is_infinite() {
            best_score = baseline.score;
        }
        if opening_baseline_score.is_none() {
            opening_baseline_score = Some(baseline.score);
        }

        let source_dir = promote_dir.as_ref().unwrap_or(&batch_dir);
        let combo_dir = config.output_dir.join(format!("combos-exp-{experiments}"));
        scorer.set_phase(ScorerCallPhase::Combo);
        let mut selection = match select_best_with_combinations(
            scorer,
            ComboSelectRequest {
                training_data: &config.training_data,
                incumbent: &incumbent,
                candidates: &candidates,
                scores: &scores,
                baseline: &baseline,
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
                crate::combos::collect_improvers_against(&scores, &baseline, config.min_improvement)
                    .into_iter()
                    .next()
                    .map(|best| ComboSelection {
                        creature_path: source_dir.join(format!("{}.json", best.stem)),
                        stem: best.stem,
                        result: best.result,
                        delta: best.delta,
                        member_indices: vec![best.index],
                        dampen: StackDampenReport::default(),
                        combos_scored: 0,
                        combos_dampened: 0,
                    })
            }
        };

        // Issue #113: an accept proposed against a remembered baseline is never
        // swapped in on that number. The winner and the incumbent are re-scored
        // together — same binary, same corpus, same call — so the decision that
        // changes the incumbent is always made from a self-consistent pair.
        if baseline_source == BaselineSource::Remembered
            && selection
                .as_ref()
                .is_some_and(|sel| sel.accepts(config.min_improvement))
        {
            let verify_dir = config.output_dir.join(format!("verify-exp-{experiments}"));
            let winner_path = selection
                .as_ref()
                .map(|sel| sel.creature_path.clone())
                .expect("checked above");
            let verify_start = Instant::now();
            scorer.set_phase(ScorerCallPhase::Promote);
            let verified = verify_accept_pair(
                scorer,
                &incumbent,
                &winner_path,
                &config.training_data,
                &verify_dir,
            );
            scorer_ms = scorer_ms.saturating_add(verify_start.elapsed().as_millis());
            if !config.preserve_losers {
                let _ = fs::remove_dir_all(&verify_dir);
            }
            match verified {
                Ok(pair) => {
                    if let Some(remembered) = &remembered_baseline {
                        let drift = remembered.drift(&pair.baseline);
                        if drift > baseline_policy.drift_epsilon {
                            return Err(baseline_drift_error(
                                drift,
                                remembered.result().score,
                                pair.baseline.score,
                                baseline_policy.drift_epsilon,
                                "accept verification",
                            ));
                        }
                    }
                    let fresh_delta = improvement(pair.winner.score, pair.baseline.score);
                    if accepts_improvement(
                        pair.winner.score,
                        pair.baseline.score,
                        config.min_improvement,
                    ) {
                        log::ok(&format!(
                            "accept verified against a freshly scored baseline: Δ {fresh_delta:+.6e}"
                        ));
                        if let Some(sel) = selection.as_mut() {
                            sel.result = pair.winner.clone();
                            sel.delta = fresh_delta;
                        }
                    } else {
                        log::warn(&format!(
                            "accept withdrawn: the freshly scored pair rejects {} (Δ {fresh_delta:+.6e} ≤ {:.6e})",
                            selection
                                .as_ref()
                                .map(|sel| sel.stem.clone())
                                .unwrap_or_default(),
                            config.min_improvement
                        ));
                        selection = None;
                    }
                    baseline = pair.baseline.clone();
                    remembered_baseline = if baseline_policy.is_enabled() {
                        baseline_scope
                            .clone()
                            .map(|scope| RememberedBaseline::new(scope, pair.baseline))
                    } else {
                        None
                    };
                    baseline_source = BaselineSource::RememberedVerified;
                }
                Err(e) => {
                    // No fresh pair, no swap: a verification that could not run
                    // is a rejected accept, never an unverified one.
                    consecutive_scorer_failures += 1;
                    log::warn(&format!("accept verification failed: {e}; not accepting"));
                    experiment_scorer_error = Some(e.clone());
                    selection = None;
                    if consecutive_scorer_failures >= config.max_consecutive_scorer_failures {
                        return Err(format!(
                            "aborting after {consecutive_scorer_failures} consecutive scorer failures; last error: {e}"
                        ));
                    }
                }
            }
        }

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
        let mut winner_member_indices = None;
        // Focuses credited with the accept. Only these are boosted; the rest of
        // the focus set is dampened as sterile, exactly as a losing single-focus
        // experiment always was (issue #109, following the #74 member rule).
        let mut accepted_focuses: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        // Issue #130: the gate reads the selection's own Δ, which was formed
        // inside the call that scored the winner. Re-subtracting the promote
        // call's baseline would compare a combo winner against a number
        // measured beside a different set of creatures.
        if let Some(sel) = selection
            && sel.accepts(config.min_improvement)
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
            // Every member of the winner names the focus it was proposed for,
            // so a K > 1 accept credits the focus that earned it rather than
            // whichever focus happened to be drawn first (issue #109).
            accepted_focuses = winning_focuses(&sel.member_indices, &candidates);
            let winner_focus = accepted_focuses
                .iter()
                .next()
                .cloned()
                .unwrap_or_else(|| focus.clone());
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
            // The analysis the memo holds describes the creature we just
            // replaced — every entry is stale from here (issue #106). So is the
            // remembered baseline: it scores the old incumbent, and the next
            // promote call must establish the new one's score fresh (#113).
            analysis_memo.invalidate();
            remembered_baseline = None;
            let opening = opening_baseline_score.unwrap_or(baseline.score);
            last_accept_focus = winner_focus.clone();
            last_accept_strategy = strategy;
            last_accept_error = sel.result.error;
            creature_meta.stamp_acceptance(&LamarckProgress {
                acceptances: acceptances + 1,
                score: sel.result.score,
                error: sel.result.error,
                opening_score: opening,
                focus_neuron: &winner_focus,
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
            winner_member_indices = Some(sel.member_indices.clone());

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

        record_focus_outcomes(
            &mut selectors.weighted,
            &focus_set,
            &accepted_focuses,
            &scores,
            &baseline,
            &candidates,
            config.min_improvement,
        );

        // A promote call that omitted the baseline still journals the score the
        // experiment was judged against, so `scores.baseline` means the same
        // thing to every reader; `baselineSource` says where it came from.
        let mut score_map: std::collections::BTreeMap<String, f64> =
            scores.iter().map(|(k, v)| (k.clone(), v.score)).collect();
        score_map.insert("baseline".to_string(), baseline.score);
        append_journal(
            &journal_path,
            &ExperimentRecord {
                experiment_number: experiments,
                timestamp_unix: unix_now(),
                seed: Some(seed),
                incumbent_id: incumbent_id(&incumbent),
                baseline_score: baseline.score,
                focus_neuron: focus.clone(),
                focus_neurons: journal_focus_set(&focus_set),
                focus_stats: Some(focus_stats.clone()),
                candidates: candidates.iter().map(|c| c.provenance.clone()).collect(),
                candidates_requested: Some(config.candidates),
                batch_limit: Some(batch_limit),
                scores: score_map,
                screen_scores: screen_score_map,
                screen_tiers,
                baseline_source: Some(baseline_source),
                winner: winner_stem,
                improvement,
                accepted,
                analysis_ms,
                memo_hits: memo_delta.hits,
                memo_misses: memo_delta.misses,
                memo_ms_saved: memo_delta.ms_saved,
                scorer_ms,
                scorer_calls: journal_calls(scorer.drain()),
                scorer_error: experiment_scorer_error,
                combo_members,
                combo_member_indices: winner_member_indices,
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
    };

    // Any call not already journalled on an experiment or replay line — a run
    // cancelled mid-experiment leaves one behind — still has to reach the
    // journal, or the fixed-cost regression is fitted to a subset (issue #112).
    journal_scorer_calls(&journal_path, "trailing", scorer.drain())?;
    let scorer_successes = scorer.successes();
    let scorer_failures = scorer.failures();

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
        stop_reason,
    })
}

/// Best full-corpus Δ vs baseline among the candidates proposed for `focus`.
///
/// With one focus every scored stem belongs to it, so this is the whole-batch
/// best Δ the focus history has always been fed. With several, each focus is
/// judged only on what it actually proposed (issue #109).
fn best_focus_delta(
    scores: &std::collections::BTreeMap<String, ScoreResult>,
    baseline: &ScoreResult,
    candidates: &[Candidate],
    focus: &str,
) -> Option<f64> {
    scores
        .iter()
        .filter(|(stem, _)| stem.as_str() != "baseline")
        .filter(|(stem, _)| {
            candidate_stem_index(stem)
                .and_then(|i| candidates.get(i))
                .is_some_and(|c| c.provenance.focus_neuron == focus)
        })
        .map(|(_, r)| improvement(r.score, baseline.score))
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

/// Focuses that produced the accepted winner's members (issue #109).
///
/// A merged combo can span more than one focus, so every member's focus is
/// credited — the same rule issue #74 applied to member strategies.
fn winning_focuses(
    member_indices: &[usize],
    candidates: &[Candidate],
) -> std::collections::BTreeSet<String> {
    member_indices
        .iter()
        .filter_map(|i| candidates.get(*i))
        .map(|c| c.provenance.focus_neuron.clone())
        .collect()
}

/// Feed a scored experiment back into the weighted focus history (issue #109).
///
/// Each focus is judged on its **own** candidates: the focus that produced the
/// winner is boosted, and a focus whose proposals went nowhere is dampened as
/// sterile even when the experiment as a whole accepted. With one focus this is
/// exactly the pre-#109 single call — the whole batch is that focus's.
fn record_focus_outcomes(
    selector: &mut WeightedFocusSelector,
    focus_set: &[String],
    accepted_focuses: &std::collections::BTreeSet<String>,
    scores: &std::collections::BTreeMap<String, ScoreResult>,
    baseline: &ScoreResult,
    candidates: &[Candidate],
    min_improvement: f64,
) {
    for focus_uuid in focus_set {
        selector.record_outcome(
            focus_uuid,
            accepted_focuses.contains(focus_uuid),
            best_focus_delta(scores, baseline, candidates, focus_uuid),
            false,
            min_improvement,
        );
    }
}

/// Record a scorer failure against every focus the experiment proposed against.
fn record_focus_failure(
    selector: &mut WeightedFocusSelector,
    focus_set: &[String],
    min_improvement: f64,
) {
    for focus_uuid in focus_set {
        selector.record_outcome(focus_uuid, false, None, true, min_improvement);
    }
}

/// The focus set a journal line records.
///
/// `None` for a single focus, so a `--focus-count 1` journal keeps its
/// pre-#109 shape and every existing reader stays correct.
fn journal_focus_set(focus_set: &[String]) -> Option<Vec<String>> {
    (focus_set.len() > 1).then(|| focus_set.to_vec())
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

/// Journal field for a drained set of scorer calls (issue #112).
///
/// An empty set is omitted rather than written as `[]`, so a journal line only
/// carries the field when the phase it describes actually called the scorer.
fn journal_calls(calls: Vec<ScorerCallRecord>) -> Option<Vec<ScorerCallRecord>> {
    (!calls.is_empty()).then_some(calls)
}

/// Write scorer calls that belong to no experiment on their own line (#112).
///
/// A no-op when nothing was recorded; a blank line would say the run made calls
/// it did not.
fn journal_scorer_calls(
    path: &Path,
    stage: &str,
    calls: Vec<ScorerCallRecord>,
) -> Result<(), String> {
    if calls.is_empty() {
        return Ok(());
    }
    append_journal_line(
        path,
        &ScorerCallsRecord {
            record: ScorerCallsKind::ScorerCalls,
            timestamp_unix: unix_now(),
            stage: stage.to_string(),
            calls,
        },
    )
}

/// Identity a remembered baseline must still match to be reused (issue #113).
///
/// `None` when the training corpus cannot be fingerprinted: the run then scores
/// a fresh baseline in every promote call, which costs work rather than
/// correctness, and says so in the log rather than reusing a score it cannot
/// key.
fn baseline_key(
    incumbent: &neat_core::CreatureExport,
    training_data: &Path,
) -> Option<BaselineKey> {
    match training_data_key(training_data) {
        Ok(training) => Some(BaselineKey {
            // The coarse shape id alone would survive a weight-only accept, so
            // the content fingerprint is what actually guards reuse.
            incumbent_id: incumbent_id(incumbent),
            fingerprint: crate::memo::creature_fingerprint(incumbent),
            training,
        }),
        Err(e) => {
            log::warn(&format!(
                "baseline reuse disabled for this experiment: {e}; scoring a fresh baseline"
            ));
            None
        }
    }
}

/// The abort message for a baseline that moved beyond the documented epsilon.
fn baseline_drift_error(
    drift: f64,
    remembered: f64,
    fresh: f64,
    epsilon: f64,
    stage: &str,
) -> String {
    format!(
        "baseline drift at {stage}: the incumbent scored {fresh:.12} now against the \
         remembered {remembered:.12} (|Δ| {drift:.6e} > --baseline-drift-epsilon {epsilon:.6e}). \
         Beyond known directory-scorer association noise this usually means the \
         training data, the scorer binary, or the incumbent changed under the run — \
         aborting rather than deciding an acceptance against a stale score."
    )
}

/// A winner and the incumbent, scored together in one full-corpus call (#113).
struct VerifiedPair {
    /// Freshly scored incumbent.
    baseline: ScoreResult,
    /// Freshly scored proposed winner.
    winner: ScoreResult,
}

/// Re-score a proposed winner beside the incumbent, full corpus, in one call.
///
/// This is what makes an accept off a remembered baseline safe: both creatures
/// are scored by the same binary, on the same corpus, in the same process, at
/// the same moment — the pairing the promote call gave up.
fn verify_accept_pair(
    scorer: &impl DirectoryScorer,
    incumbent: &neat_core::CreatureExport,
    winner_path: &Path,
    training_data: &Path,
    work_dir: &Path,
) -> Result<VerifiedPair, String> {
    if work_dir.exists() {
        fs::remove_dir_all(work_dir).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(work_dir).map_err(|e| e.to_string())?;
    let json = creature_to_json_pretty(incumbent).map_err(|e| e.to_string())?;
    fs::write(work_dir.join("baseline.json"), json).map_err(|e| e.to_string())?;
    fs::copy(winner_path, work_dir.join("winner.json"))
        .map_err(|e| format!("copy winner from {} failed: {e}", winner_path.display()))?;
    let scores = scorer
        .score_directory(work_dir, training_data)
        .map_err(|e| e.to_string())?;
    let baseline = scores
        .get("baseline")
        .cloned()
        .ok_or_else(|| "accept verification returned no baseline".to_string())?;
    let winner = scores
        .get("winner")
        .cloned()
        .ok_or_else(|| "accept verification returned no winner".to_string())?;
    Ok(VerifiedPair { baseline, winner })
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

    /// Default run config over `tiny_setup`, for tests that only vary a knob.
    fn base_config(
        creature: PathBuf,
        training_data: PathBuf,
        output_dir: PathBuf,
    ) -> LamarckConfig {
        LamarckConfig {
            creature,
            training_data,
            timeout: Duration::from_millis(500),
            max_experiments: None,
            candidates: 4,
            scale_candidate_quotas: false,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir,
            preserve_losers: true,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            focus_count: crate::config::DEFAULT_FOCUS_COUNT,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            phase0_parity: true,
            structural_only: false,
            screen_sample_rate: None,
            screen_promote_threshold: 0.0,
            screen_promote_gate: PromoteGateMode::Absolute,
            screen_promote_sigma_k: DEFAULT_SCREEN_PROMOTE_SIGMA_K,
            baseline_reverify_interval: 0,
            baseline_drift_epsilon: Some(crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON),
            grafts_path: None,
            graft_replay_budget: None,
            backprop_learning_rate: None,
            backprop_max_bias_adjustment_scale: None,
            analysis_memo_entries: crate::memo::DEFAULT_ANALYSIS_MEMO_ENTRIES,
            analysis_threads: crate::chunks::DEFAULT_ANALYSIS_THREADS,
        }
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
            max_experiments: None,
            candidates: 4,
            scale_candidate_quotas: false,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: out.clone(),
            preserve_losers: true,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            focus_count: crate::config::DEFAULT_FOCUS_COUNT,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            phase0_parity: true,
            structural_only: false,
            screen_sample_rate: None,
            screen_promote_threshold: 0.0,
            screen_promote_gate: PromoteGateMode::Absolute,
            screen_promote_sigma_k: DEFAULT_SCREEN_PROMOTE_SIGMA_K,
            baseline_reverify_interval: 0,
            baseline_drift_epsilon: Some(crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON),
            grafts_path: None,
            graft_replay_budget: None,
            backprop_learning_rate: None,
            backprop_max_bias_adjustment_scale: None,
            analysis_memo_entries: crate::memo::DEFAULT_ANALYSIS_MEMO_ENTRIES,
            analysis_threads: crate::chunks::DEFAULT_ANALYSIS_THREADS,
        };
        let scorer = ScriptedScorer {
            calls: Arc::new(Mutex::new(0)),
        };
        let result = run_optimisation(&config, &scorer).unwrap();
        assert!(result.journal_path.is_file());
        assert!(result.best_path.is_file());
        assert!(result.experiments >= 1);
        assert!(result.scorer_successes >= 1);
        assert_eq!(
            result.stop_reason,
            StopReason::Timeout,
            "an uncapped, uncancelled run stops on the wall clock"
        );
        let journal = fs::read_to_string(&result.journal_path).unwrap();
        let mut lines = journal.lines();
        // Line 1 is the run header (issue #71); line 2 carries the Phase-0
        // scorer call, which belongs to no experiment (issue #112); experiments
        // follow.
        assert!(lines.next().unwrap().contains("\"record\":\"runHeader\""));
        assert!(lines.next().unwrap().contains("\"record\":\"scorerCalls\""));
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

    /// Issue #114: the scorer's files are compact, the human's stay pretty, and
    /// a promote file is a hard link to the screen file rather than a copy.
    #[test]
    fn batch_files_are_compact_while_best_and_winners_stay_pretty() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let out = dir.path().join("out");
        let config = LamarckConfig {
            max_experiments: Some(1),
            // Screen then promote, so the promote directory is written from the
            // screen batch — the copy this issue replaces with a link.
            screen_sample_rate: Some(0.5),
            phase0_parity: false,
            ..base_config(creature_path, training, out.clone())
        };
        let result = run_optimisation(
            &config,
            &ScriptedScorer {
                calls: Arc::new(Mutex::new(0)),
            },
        )
        .unwrap();
        assert_eq!(
            result.acceptances, 1,
            "the scripted winner must be accepted"
        );

        // Scorer-facing batch files: one line each, no indentation.
        let batch = out.join("candidates-exp-1");
        for name in ["baseline.json", "candidate-000.json"] {
            let text = fs::read_to_string(batch.join(name)).unwrap();
            assert!(
                !text.contains('\n'),
                "{name} must be compact for the scorer, got:\n{text}"
            );
            // Still a creature the scorer can parse.
            neat_core::parse_creature_json(&text).unwrap();
        }

        // Human-facing artefacts: still pretty-printed.
        for path in [
            out.join("best.json"),
            out.join("winners").join("winner-0001.json"),
        ] {
            let text = fs::read_to_string(&path).unwrap();
            assert!(
                text.contains("\n  \""),
                "{} must stay pretty-printed for a human reader",
                path.display()
            );
        }

        // Promote directory: the same bytes at a second path, not a second copy.
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let promote = out.join("promote-exp-1");
            let promoted = fs::read_dir(&promote)
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.ends_with(".json"))
                .collect::<Vec<_>>();
            assert!(!promoted.is_empty(), "expected a promote batch");
            for name in promoted {
                let source = fs::metadata(batch.join(&name)).unwrap();
                let linked = fs::metadata(promote.join(&name)).unwrap();
                assert_eq!(
                    (source.dev(), source.ino()),
                    (linked.dev(), linked.ino()),
                    "{name} must be hard-linked from the screen batch"
                );
            }
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
            max_experiments: None,
            candidates: 4,
            scale_candidate_quotas: false,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: out.clone(),
            preserve_losers: true,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            focus_count: crate::config::DEFAULT_FOCUS_COUNT,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            phase0_parity: false,
            structural_only: false,
            screen_sample_rate: Some(0.1),
            screen_promote_threshold: 0.0,
            screen_promote_gate: PromoteGateMode::Absolute,
            screen_promote_sigma_k: DEFAULT_SCREEN_PROMOTE_SIGMA_K,
            baseline_reverify_interval: 0,
            baseline_drift_epsilon: Some(crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON),
            grafts_path: None,
            graft_replay_budget: None,
            backprop_learning_rate: None,
            backprop_max_bias_adjustment_scale: None,
            analysis_memo_entries: crate::memo::DEFAULT_ANALYSIS_MEMO_ENTRIES,
            analysis_threads: crate::chunks::DEFAULT_ANALYSIS_THREADS,
        };
        let result = run_optimisation(&config, &NegativeScreenScorer).unwrap();
        assert!(result.experiments >= 1);
        assert_eq!(result.acceptances, 0);
        // Skip the run header (issue #71) — the first experiment follows it.
        let first = journal_lines(&result.journal_path)
            .into_iter()
            .find_map(|l| match l {
                JournalLine::Experiment(record) => Some(*record),
                JournalLine::Header(_)
                | JournalLine::GraftReplay(_)
                | JournalLine::ScorerCalls(_) => None,
            })
            .expect("at least one experiment");
        assert!(first.screen_scores.is_some());
        assert!(first.scores.is_empty());
        assert!(!first.accepted);
        assert!(!out.join("promote-exp-1").exists());
    }

    /// Screen scorer producing one clear improver in a sea of sampling wobble
    /// (issue #111): the shape the noise-aware gate exists to separate.
    struct BimodalScreenScorer;

    impl DirectoryScorer for BimodalScreenScorer {
        fn score_directory_sampled(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
            sample: crate::scorer::ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            let mut stems: Vec<String> = fs::read_dir(candidates_dir)
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.strip_suffix(".json")
                        .filter(|stem| *stem != "baseline")
                        .map(str::to_string)
                })
                .collect();
            stems.sort();
            let mut map = BTreeMap::new();
            let score = |delta: f64| ScoreResult {
                score: 0.5 + delta,
                error: 0.5 - delta,
                complexity_penalty: 0.0,
            };
            map.insert("baseline".to_string(), score(0.0));
            for (index, stem) in stems.iter().enumerate() {
                let delta = match (sample.is_subsample(), index) {
                    // One real improver, far above the batch's own wobble.
                    (true, 0) => 5e-5,
                    // Wobble: alternating, all comfortably over the 1e-6 floor.
                    (true, _) if index % 2 == 0 => 1.4e-6,
                    (true, _) => -1.3e-6,
                    // The full corpus contradicts every promotion, so the run
                    // never accepts and keeps screening.
                    (false, _) => -1e-4,
                };
                map.insert(stem.clone(), score(delta));
            }
            Ok(map)
        }
    }

    fn bimodal_screen_config(dir: &Path, gate: PromoteGateMode) -> (LamarckConfig, PathBuf) {
        let (creature_path, training) = tiny_setup(dir);
        let out = dir.join(match gate {
            PromoteGateMode::Absolute => "out-absolute",
            PromoteGateMode::NoiseAware => "out-noise-aware",
        });
        let config = LamarckConfig {
            creature: creature_path,
            training_data: training,
            timeout: Duration::from_millis(800),
            max_experiments: Some(1),
            candidates: 12,
            scale_candidate_quotas: true,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: out.clone(),
            preserve_losers: true,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            focus_count: crate::config::DEFAULT_FOCUS_COUNT,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            phase0_parity: false,
            structural_only: false,
            screen_sample_rate: Some(0.1),
            screen_promote_threshold: 1e-6,
            screen_promote_gate: gate,
            screen_promote_sigma_k: DEFAULT_SCREEN_PROMOTE_SIGMA_K,
            baseline_reverify_interval: 0,
            baseline_drift_epsilon: Some(crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON),
            grafts_path: None,
            graft_replay_budget: None,
            backprop_learning_rate: None,
            backprop_max_bias_adjustment_scale: None,
            analysis_memo_entries: crate::memo::DEFAULT_ANALYSIS_MEMO_ENTRIES,
            analysis_threads: crate::chunks::DEFAULT_ANALYSIS_THREADS,
        };
        (config, out)
    }

    /// Every scorer call the journal recorded, whichever line kind carries it.
    fn journalled_scorer_calls(path: &Path) -> Vec<ScorerCallRecord> {
        journal_lines(path)
            .into_iter()
            .flat_map(|line| match line {
                JournalLine::Experiment(record) => record.scorer_calls.unwrap_or_default(),
                JournalLine::GraftReplay(replay) => replay.scorer_calls.unwrap_or_default(),
                JournalLine::ScorerCalls(calls) => calls.calls,
                JournalLine::Header(_) => Vec::new(),
            })
            .collect()
    }

    /// Issue #112: the journal accounts for **every** scorer invocation a run
    /// made — Phase-0, screen, promote and combo alike. An intercept fitted to a
    /// subset of the calls is the mis-measurement a go/no-go must not rest on,
    /// so the count is pinned against the run's own success/failure totals.
    #[test]
    fn every_scorer_call_reaches_the_journal_with_its_creature_count() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let out = dir.path().join("out-call-cost");
        let config = LamarckConfig {
            creature: creature_path,
            training_data: training,
            timeout: Duration::from_millis(2000),
            max_experiments: Some(2),
            candidates: 6,
            scale_candidate_quotas: false,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: out.clone(),
            preserve_losers: true,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            focus_count: crate::config::DEFAULT_FOCUS_COUNT,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            // Phase-0 runs, so a call is made before the first experiment.
            phase0_parity: true,
            structural_only: false,
            screen_sample_rate: Some(0.1),
            screen_promote_threshold: 0.0,
            screen_promote_gate: PromoteGateMode::Absolute,
            screen_promote_sigma_k: DEFAULT_SCREEN_PROMOTE_SIGMA_K,
            baseline_reverify_interval: 0,
            baseline_drift_epsilon: Some(crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON),
            grafts_path: None,
            graft_replay_budget: None,
            backprop_learning_rate: None,
            backprop_max_bias_adjustment_scale: None,
            analysis_memo_entries: crate::memo::DEFAULT_ANALYSIS_MEMO_ENTRIES,
            analysis_threads: crate::chunks::DEFAULT_ANALYSIS_THREADS,
        };
        let scorer = ScriptedScorer {
            calls: Arc::new(Mutex::new(0)),
        };
        let result = run_optimisation(&config, &scorer).unwrap();

        let calls = journalled_scorer_calls(&result.journal_path);
        assert_eq!(
            calls.len() as u64,
            result.scorer_successes + result.scorer_failures,
            "journalled calls must account for every scorer invocation: {calls:?}"
        );
        assert!(
            calls.iter().all(|call| call.creatures >= 1),
            "every call names the creatures it scored: {calls:?}"
        );

        let phases: std::collections::BTreeSet<ScorerCallPhase> =
            calls.iter().map(|call| call.phase).collect();
        assert!(
            phases.contains(&ScorerCallPhase::Phase0),
            "the Phase-0 baseline call is journalled on its own line: {phases:?}"
        );
        assert!(phases.contains(&ScorerCallPhase::Screen), "{phases:?}");
        assert!(phases.contains(&ScorerCallPhase::Promote), "{phases:?}");

        // A screen call is sampled and scores the whole batch; a promote call is
        // full corpus over the survivors — the distinction the fit needs.
        let screen = calls
            .iter()
            .find(|call| call.phase == ScorerCallPhase::Screen)
            .expect("a screen call");
        assert_eq!(screen.sample_rate, Some(0.1));
        assert!(
            screen.creatures > 1,
            "the screen call scores baseline + candidates: {screen:?}"
        );
        let promote = calls
            .iter()
            .find(|call| call.phase == ScorerCallPhase::Promote)
            .expect("a promote call");
        assert_eq!(promote.sample_rate, None, "promote is full corpus");

        // The same journal the run wrote is what `report` regresses (issue #112).
        let report = crate::report::report_from_journal(&result.journal_path).unwrap();
        assert_eq!(report.scorer_call_cost.calls as usize, calls.len());
        assert!(report.scorer_call_cost.by_phase.contains_key("screen"));
    }

    fn first_experiment(path: &Path) -> ExperimentRecord {
        journal_lines(path)
            .into_iter()
            .find_map(|l| match l {
                JournalLine::Experiment(record) => Some(*record),
                JournalLine::Header(_)
                | JournalLine::GraftReplay(_)
                | JournalLine::ScorerCalls(_) => None,
            })
            .expect("at least one experiment")
    }

    fn run_header(path: &Path) -> RunHeaderRecord {
        journal_lines(path)
            .into_iter()
            .find_map(|l| match l {
                JournalLine::Header(header) => Some(*header),
                JournalLine::Experiment(_)
                | JournalLine::GraftReplay(_)
                | JournalLine::ScorerCalls(_) => None,
            })
            .expect("a run header")
    }

    /// Issue #111 end to end: the noise-aware gate promotes the improver alone,
    /// journals its tier counts, and records itself in the run header.
    #[test]
    fn the_noise_aware_gate_promotes_only_the_improver_and_journals_the_tiers() {
        let dir = tempdir().unwrap();
        let (config, _out) = bimodal_screen_config(dir.path(), PromoteGateMode::NoiseAware);
        let result = run_optimisation(&config, &BimodalScreenScorer).unwrap();

        let header = run_header(&result.journal_path);
        assert_eq!(
            header.config.screen_promote_gate.as_deref(),
            Some("noise-aware")
        );
        assert_eq!(
            header.config.screen_promote_sigma_k,
            Some(DEFAULT_SCREEN_PROMOTE_SIGMA_K)
        );

        let record = first_experiment(&result.journal_path);
        let tiers = record.screen_tiers.expect("the screen tier is journalled");
        assert_eq!(tiers.gate, "noise-aware");
        assert!(tiers.screened >= 4, "batch too small to price: {tiers:?}");
        assert_eq!(tiers.promoted, 1, "only the improver should be bought");
        assert!(tiers.sigma.is_some_and(|s| s > 0.0));
        assert!(tiers.threshold > 1e-6, "the gate rose above the floor");
        // The full corpus scored exactly what the gate admitted, plus baseline.
        assert_eq!(record.scores.len(), 2);
    }

    /// The opt-in property, end to end: with no new flag set the same batch
    /// promotes everything over the bare threshold, exactly as before #111.
    #[test]
    fn the_default_gate_still_promotes_every_candidate_over_the_threshold() {
        let dir = tempdir().unwrap();
        let (config, _out) = bimodal_screen_config(dir.path(), PromoteGateMode::Absolute);
        let result = run_optimisation(&config, &BimodalScreenScorer).unwrap();

        let header = run_header(&result.journal_path);
        assert_eq!(
            header.config.screen_promote_gate.as_deref(),
            Some("absolute")
        );
        assert_eq!(header.config.screen_promote_sigma_k, None);

        let record = first_experiment(&result.journal_path);
        let tiers = record.screen_tiers.expect("the screen tier is journalled");
        assert_eq!(tiers.gate, "absolute");
        assert_eq!(tiers.threshold, 1e-6);
        assert_eq!(tiers.sigma, None);
        // Every wobble above the floor is bought, which is the cost #111 measures.
        assert!(
            tiers.promoted > 1,
            "the absolute gate should buy the wobble too: {tiers:?}"
        );
        assert_eq!(record.scores.len() as u64, tiers.promoted + 1);
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

    /// Run with a graft store holding one helpful synapse graft for phase G.
    fn graft_replay_run(dir: &Path) -> (RunResult, PathBuf) {
        use crate::grafts::{GraftStore, graft_from_add_synapse};

        let creature_path = dir.join("creature.json");
        let training = dir.join("data");
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

        let grafts_path = dir.join("grafts.json");
        let mut store = GraftStore::new();
        store.upsert(graft_from_add_synapse("input-1", "h1", 0.05));
        store.save(&grafts_path).unwrap();

        let out = dir.join("out");
        let config = LamarckConfig {
            creature: creature_path,
            training_data: training,
            timeout: Duration::from_millis(500),
            max_experiments: None,
            candidates: 2,
            scale_candidate_quotas: false,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: out,
            preserve_losers: true,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            focus_count: crate::config::DEFAULT_FOCUS_COUNT,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            phase0_parity: false,
            structural_only: false,
            screen_sample_rate: None,
            screen_promote_threshold: 0.0,
            screen_promote_gate: PromoteGateMode::Absolute,
            screen_promote_sigma_k: DEFAULT_SCREEN_PROMOTE_SIGMA_K,
            baseline_reverify_interval: 0,
            baseline_drift_epsilon: Some(crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON),
            grafts_path: Some(grafts_path.clone()),
            // Explicit budget so phase-G is not starved by a sub-second run timeout.
            graft_replay_budget: Some(Duration::from_secs(5)),
            backprop_learning_rate: None,
            backprop_max_bias_adjustment_scale: None,
            analysis_memo_entries: crate::memo::DEFAULT_ANALYSIS_MEMO_ENTRIES,
            analysis_threads: crate::chunks::DEFAULT_ANALYSIS_THREADS,
        };
        let result = run_optimisation(&config, &GraftAwareScorer).unwrap();
        (result, grafts_path)
    }

    #[test]
    fn phase_g_replays_stored_graft_before_loop() {
        use crate::grafts::GraftStore;

        let dir = tempdir().unwrap();
        let (result, grafts_path) = graft_replay_run(dir.path());
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

    /// Issue #74: a phase-G accept is journalled so `report` can bucket it.
    #[test]
    fn phase_g_accept_is_journalled_as_a_graft_replay_record() {
        let dir = tempdir().unwrap();
        let (result, _) = graft_replay_run(dir.path());

        let replay = journal_lines(&result.journal_path)
            .into_iter()
            .find_map(|l| match l {
                JournalLine::GraftReplay(record) => Some(*record),
                _ => None,
            })
            .expect("phase-G writes a graftReplay journal line");
        assert!(replay.accepted, "the helpful graft was accepted");
        assert!(replay.grafts_applied >= 1);
        assert!(
            replay.improvement.is_some_and(|d| d > 0.0),
            "an accepted replay records its score Δ"
        );
        assert!(replay.baseline_score.is_some() && replay.score.is_some());

        // The report must bucket it rather than drop it (issue #74).
        let grafts = crate::report::report_from_journal(&result.journal_path)
            .unwrap()
            .graft_replay
            .expect("report carries the graft-replay bucket");
        assert_eq!(grafts.accepts, 1);
        assert!(grafts.cumulative_improvement > 0.0);

        // The record must survive the journal encoding under its camelCase name.
        let encoded = fs::read_to_string(&result.journal_path).unwrap();
        assert!(encoded.contains("\"record\":\"graftReplay\""));
        assert!(encoded.contains("\"graftsApplied\""));
    }

    /// Issue #74: an accepted winner journals the candidate indices behind it.
    #[test]
    fn an_accepted_winner_journals_its_member_indices() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let config = LamarckConfig {
            creature: creature_path,
            training_data: training,
            timeout: Duration::from_millis(500),
            max_experiments: Some(1),
            candidates: 4,
            scale_candidate_quotas: false,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: dir.path().join("out"),
            preserve_losers: false,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            focus_count: crate::config::DEFAULT_FOCUS_COUNT,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            phase0_parity: false,
            structural_only: false,
            screen_sample_rate: None,
            screen_promote_threshold: 0.0,
            screen_promote_gate: PromoteGateMode::Absolute,
            screen_promote_sigma_k: DEFAULT_SCREEN_PROMOTE_SIGMA_K,
            baseline_reverify_interval: 0,
            baseline_drift_epsilon: Some(crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON),
            grafts_path: None,
            graft_replay_budget: None,
            backprop_learning_rate: None,
            backprop_max_bias_adjustment_scale: None,
            analysis_memo_entries: crate::memo::DEFAULT_ANALYSIS_MEMO_ENTRIES,
            analysis_threads: crate::chunks::DEFAULT_ANALYSIS_THREADS,
        };
        let scorer = ScriptedScorer {
            calls: Arc::new(Mutex::new(0)),
        };
        let result = run_optimisation(&config, &scorer).unwrap();
        let record = journal_lines(&result.journal_path)
            .into_iter()
            .find_map(|l| match l {
                JournalLine::Experiment(record) if record.accepted => Some(*record),
                _ => None,
            })
            .expect("at least one accepted experiment");
        let indices = record
            .combo_member_indices
            .expect("an accepted winner records its member indices");
        assert!(!indices.is_empty(), "a winner has at least one member");
        assert!(
            indices.iter().all(|i| *i < record.candidates.len()),
            "member indices address the journalled candidates"
        );
    }

    fn reproducibility_config(creature: PathBuf, training: PathBuf, out: PathBuf) -> LamarckConfig {
        LamarckConfig {
            creature,
            training_data: training,
            timeout: Duration::from_millis(400),
            max_experiments: None,
            candidates: 4,
            scale_candidate_quotas: false,
            min_improvement: 1e-6,
            seed: None,
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: out,
            preserve_losers: false,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            focus_count: crate::config::DEFAULT_FOCUS_COUNT,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            phase0_parity: false,
            structural_only: false,
            screen_sample_rate: None,
            screen_promote_threshold: 0.0,
            screen_promote_gate: PromoteGateMode::Absolute,
            screen_promote_sigma_k: DEFAULT_SCREEN_PROMOTE_SIGMA_K,
            baseline_reverify_interval: 0,
            baseline_drift_epsilon: Some(crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON),
            grafts_path: None,
            graft_replay_budget: None,
            backprop_learning_rate: None,
            backprop_max_bias_adjustment_scale: None,
            analysis_memo_entries: crate::memo::DEFAULT_ANALYSIS_MEMO_ENTRIES,
            analysis_threads: crate::chunks::DEFAULT_ANALYSIS_THREADS,
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
                JournalLine::Header(_)
                | JournalLine::GraftReplay(_)
                | JournalLine::ScorerCalls(_) => None,
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
                JournalLine::Header(_)
                | JournalLine::GraftReplay(_)
                | JournalLine::ScorerCalls(_) => None,
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

    /// Issue #108: every experiment records the budget it asked for and why the
    /// batch stopped, so an under-filled generator is visible in `report`.
    #[test]
    fn experiment_records_the_requested_budget_and_batch_limit() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let mut config = reproducibility_config(creature_path, training, dir.path().join("out"));
        // Far past what a two-neuron creature can propose.
        config.candidates = 64;
        config.scale_candidate_quotas = true;
        let scorer = ScriptedScorer {
            calls: Arc::new(Mutex::new(0)),
        };
        let result = run_optimisation(&config, &scorer).unwrap();

        let experiments: Vec<ExperimentRecord> = journal_lines(&result.journal_path)
            .into_iter()
            .filter_map(|l| match l {
                JournalLine::Experiment(record) => Some(*record),
                JournalLine::Header(_)
                | JournalLine::GraftReplay(_)
                | JournalLine::ScorerCalls(_) => None,
            })
            .collect();
        assert!(!experiments.is_empty(), "run wrote at least one experiment");
        for record in &experiments {
            assert_eq!(record.candidates_requested, Some(64));
            let limit = record.batch_limit.expect("batch limit is journalled");
            if record.candidates.len() < 64 {
                assert_eq!(
                    limit,
                    BatchLimit::Exhausted,
                    "an under-filled scaled batch must name exhaustion"
                );
            } else {
                assert_eq!(limit, BatchLimit::Budget);
            }
        }

        // The fields must survive the journal encoding, under their camelCase names.
        let encoded = fs::read_to_string(&result.journal_path).unwrap();
        assert!(encoded.contains("\"candidatesRequested\""));
        assert!(encoded.contains("\"batchLimit\""));
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

    /// Issue #96: the backprop cap A/B is only readable from its journal if the
    /// cap in force is recorded, exactly as the learning rate already is.
    #[test]
    fn run_header_records_the_analysis_thread_count() {
        let config = LamarckConfig {
            analysis_threads: 8,
            ..LamarckConfig::default()
        };
        let record =
            RunConfigRecord::from_config(&config, crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON);
        assert_eq!(record.analysis_threads, 8);

        // Without this field a parallel arm and a serial arm are
        // indistinguishable from their journals, so a run that was *slower*
        // than serial could not be diagnosed after the fact (issue #107).
        let encoded = serde_json::to_string(&record).unwrap();
        assert!(
            encoded.contains("\"analysisThreads\":8"),
            "thread count missing from the encoded header: {encoded}"
        );
        let decoded: RunConfigRecord = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.analysis_threads, 8);

        // Journals written before the knob existed must still parse.
        let legacy = encoded.replace(",\"analysisThreads\":8", "");
        let legacy: RunConfigRecord = serde_json::from_str(&legacy).unwrap();
        assert_eq!(legacy.analysis_threads, 0);
    }

    /// Seed replay must survive parallel analysis at every thread count (#107).
    #[test]
    fn recorded_seed_replays_the_candidate_stream_at_every_thread_count() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let candidates_at = |threads: usize, label: &str, seed: Option<u64>| -> (u64, String) {
            let mut config = reproducibility_config(
                creature_path.clone(),
                training.clone(),
                dir.path().join(label),
            );
            config.analysis_threads = threads;
            config.seed = seed;
            let result = run_optimisation(
                &config,
                &ScriptedScorer {
                    calls: Arc::new(Mutex::new(0)),
                },
            )
            .unwrap();
            let experiment = journal_lines(&result.journal_path)
                .into_iter()
                .find_map(|l| match l {
                    JournalLine::Experiment(record) => Some(*record),
                    JournalLine::Header(_)
                    | JournalLine::GraftReplay(_)
                    | JournalLine::ScorerCalls(_) => None,
                })
                .expect("at least one experiment");
            (
                result.seed,
                format!(
                    "{}|{}",
                    experiment.focus_neuron,
                    serde_json::to_string(&experiment.candidates).unwrap()
                ),
            )
        };

        let (seed, baseline) = candidates_at(1, "threads-1", None);
        for threads in [2, 8] {
            let (replay_seed, replayed) =
                candidates_at(threads, &format!("threads-{threads}"), Some(seed));
            assert_eq!(replay_seed, seed);
            assert_eq!(
                baseline, replayed,
                "the recorded seed must regenerate the same candidate stream at {threads} threads"
            );
        }
    }

    #[test]
    fn run_header_records_the_backprop_bias_cap() {
        let config = LamarckConfig {
            backprop_learning_rate: Some(0.001),
            backprop_max_bias_adjustment_scale: Some(1e-6),
            ..LamarckConfig::default()
        };
        let record =
            RunConfigRecord::from_config(&config, crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON);
        assert_eq!(record.backprop_max_bias_adjustment_scale, Some(1e-6));
        assert_eq!(record.backprop_learning_rate, Some(0.001));

        // Round-trip through the journal encoding: an arm is identified from
        // its journal alone, so the field has to survive read-back.
        let encoded = serde_json::to_string(&record).unwrap();
        assert!(
            encoded.contains("\"backpropMaxBiasAdjustmentScale\":1e-6"),
            "cap missing from the encoded header: {encoded}"
        );
        let decoded: RunConfigRecord = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.backprop_max_bias_adjustment_scale, Some(1e-6));

        // Older journals predate the field and must still parse (fail loud on
        // a real fault, not on a header written before the knob existed).
        let legacy = encoded.replace(",\"backpropMaxBiasAdjustmentScale\":1e-6", "");
        let legacy: RunConfigRecord = serde_json::from_str(&legacy).unwrap();
        assert_eq!(legacy.backprop_max_bias_adjustment_scale, None);
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
                    JournalLine::Header(_)
                    | JournalLine::GraftReplay(_)
                    | JournalLine::ScorerCalls(_) => None,
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

    /// Issue #105: an experiment fuses its five analysis passes into two scans.
    #[test]
    fn each_experiment_opens_at_most_two_training_scans() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let mut config = reproducibility_config(creature_path, training, dir.path().join("out"));
        config.max_experiments = Some(2);
        config.timeout = Duration::from_secs(30);
        // Weighted focus exercises the output-MAE pass fused into scan one.
        config.focus_neuron = None;
        config.focus_policy = FocusPolicy::Weighted;
        // The #105 contract is "two scans per *computed* experiment"; issue #106
        // then removes scan two on a memo hit, so the memo is off here and the
        // count below stays the pre-memo one.
        config.analysis_memo_entries = 0;

        crate::analysis::reset_training_scan_count();
        let result = run_optimisation(
            &config,
            &ScriptedScorer {
                calls: Arc::new(Mutex::new(0)),
            },
        )
        .unwrap();

        assert!(result.experiments >= 1, "expected at least one experiment");
        assert_eq!(
            crate::analysis::training_scans_opened(),
            2 * result.experiments,
            "each experiment must open exactly two analysis training scans"
        );
        for record in experiment_records(&result.journal_path) {
            assert_eq!(
                (record.memo_hits, record.memo_misses, record.memo_ms_saved),
                (0, 0, 0),
                "a disabled memo journals zeros, never a phantom saving"
            );
        }
    }

    /// Every `ExperimentRecord` in a journal, in order.
    fn experiment_records(path: &Path) -> Vec<ExperimentRecord> {
        journal_lines(path)
            .into_iter()
            .filter_map(|line| match line {
                JournalLine::Experiment(record) => Some(*record),
                JournalLine::Header(_)
                | JournalLine::GraftReplay(_)
                | JournalLine::ScorerCalls(_) => None,
            })
            .collect()
    }

    /// Never improves anything, so the incumbent survives the whole run.
    struct FlatScorer;

    impl DirectoryScorer for FlatScorer {
        fn score_directory_sampled(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
            _sample: crate::scorer::ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            const BASE_SCORE: f64 = 0.64;
            let mut map = BTreeMap::new();
            map.insert(
                "baseline".into(),
                ScoreResult {
                    score: BASE_SCORE,
                    error: 1.0 - BASE_SCORE,
                    complexity_penalty: 0.0,
                },
            );
            for entry in fs::read_dir(candidates_dir).into_iter().flatten().flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(stem) = name.strip_suffix(".json")
                    && stem != "baseline"
                {
                    map.insert(
                        stem.to_string(),
                        ScoreResult {
                            score: BASE_SCORE,
                            error: 1.0 - BASE_SCORE,
                            complexity_penalty: 0.0,
                        },
                    );
                }
            }
            Ok(map)
        }
    }

    /// Improves `candidate-000` on exactly one scorer batch, so the run has one
    /// accept in a known place.
    struct AcceptOnceScorer {
        accept_on_call: usize,
        calls: Arc<Mutex<usize>>,
    }

    impl DirectoryScorer for AcceptOnceScorer {
        fn score_directory_sampled(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
            _sample: crate::scorer::ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            let accepting = *calls == self.accept_on_call;
            const BASE_SCORE: f64 = 0.64;
            let mut map = BTreeMap::new();
            map.insert(
                "baseline".into(),
                ScoreResult {
                    score: BASE_SCORE,
                    error: 1.0 - BASE_SCORE,
                    complexity_penalty: 0.0,
                },
            );
            for entry in fs::read_dir(candidates_dir).into_iter().flatten().flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(stem) = name.strip_suffix(".json")
                    && stem != "baseline"
                {
                    let score = if accepting && stem == "candidate-000" {
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
            Ok(map)
        }
    }

    /// Issue #106: an unchanged incumbent on a repeated focus reuses the whole
    /// post-focus scan — experiment 2 opens no scan for it and reports the same
    /// numbers experiment 1 measured.
    #[test]
    fn an_unchanged_incumbent_reuses_the_focus_scan() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let mut config = reproducibility_config(creature_path, training, dir.path().join("out"));
        config.max_experiments = Some(3);
        config.timeout = Duration::from_secs(30);
        config.seed = Some(1);
        config.focus_neuron = Some("o1".into());

        crate::analysis::reset_training_scan_count();
        let result = run_optimisation(&config, &FlatScorer).unwrap();
        let records = experiment_records(&result.journal_path);
        assert_eq!(records.len() as u64, result.experiments);
        assert!(result.experiments >= 2, "need a second experiment to reuse");
        assert_eq!(result.acceptances, 0, "FlatScorer never improves anything");

        assert_eq!(
            crate::analysis::training_scans_opened(),
            result.experiments + 1,
            "only the first experiment scans twice; later ones reuse the focus scan \
             and open just the (uncacheable) learning scan"
        );

        let first = &records[0];
        assert_eq!(first.memo_hits, 0, "the first experiment cannot hit");
        assert!(first.memo_misses >= 1);
        for record in &records[1..] {
            assert_eq!(
                record.memo_hits, 1,
                "experiment {} reuses the focus scan",
                record.experiment_number
            );
            assert_eq!(record.memo_misses, 0);
        }

        // The reused numbers are the measured ones, not a fresh approximation.
        let scan_fields = |record: &ExperimentRecord| {
            let stats = record.focus_stats.clone().expect("focus stats journalled");
            format!(
                "{}|{}|{}|{:?}|{:?}|{}",
                stats.record_count,
                stats.pre_mean,
                stats.post_mean,
                stats.mean_abs_error,
                stats.mean_derivative,
                stats.saturation_fraction
            )
        };
        assert_eq!(
            scan_fields(&records[0]),
            scan_fields(&records[1]),
            "a memo hit must reproduce the scan it replaced"
        );
    }

    /// Issue #106: memoisation is an optimisation, not a behaviour change — the
    /// same seed must produce the same experiments with the memo off and on.
    #[test]
    fn the_memo_does_not_move_the_candidate_stream() {
        let run = |entries: usize, out: PathBuf| {
            let dir = tempdir().unwrap();
            let (creature_path, training) = tiny_setup(dir.path());
            let mut config = reproducibility_config(creature_path, training, out);
            config.max_experiments = Some(3);
            config.timeout = Duration::from_secs(30);
            config.seed = Some(9);
            config.focus_neuron = None;
            config.focus_policy = FocusPolicy::Weighted;
            config.analysis_memo_entries = entries;
            let result = run_optimisation(&config, &FlatScorer).unwrap();
            // The tempdir must outlive the run, so return the journal contents.
            experiment_records(&result.journal_path)
        };

        let out = tempdir().unwrap();
        let without = run(0, out.path().join("off"));
        let with = run(
            crate::memo::DEFAULT_ANALYSIS_MEMO_ENTRIES,
            out.path().join("on"),
        );

        assert_eq!(without.len(), with.len(), "same cap, same experiment count");
        for (a, b) in without.iter().zip(with.iter()) {
            assert_eq!(a.focus_neuron, b.focus_neuron, "focus choice must not move");
            assert_eq!(
                serde_json::to_string(&a.candidates).unwrap(),
                serde_json::to_string(&b.candidates).unwrap(),
                "experiment {} proposed a different candidate stream with the memo on",
                a.experiment_number
            );
            assert_eq!(
                serde_json::to_string(&a.focus_stats).unwrap(),
                serde_json::to_string(&b.focus_stats).unwrap(),
                "experiment {} journalled different focus statistics",
                a.experiment_number
            );
        }
        assert!(
            with.iter().any(|r| r.memo_hits > 0),
            "the memo arm must actually have hit, or this proves nothing"
        );
    }

    /// Issue #106: accepting a winner invalidates the memo, so the experiment
    /// after an accept recomputes everything against the new incumbent.
    #[test]
    fn an_accept_invalidates_the_memo_for_the_next_experiment() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let mut config = reproducibility_config(creature_path, training, dir.path().join("out"));
        config.max_experiments = Some(4);
        config.timeout = Duration::from_secs(30);
        config.seed = Some(1);
        config.focus_neuron = Some("o1".into());

        // Accept on the second batch: experiment 2 changes the incumbent, so
        // experiment 3 must miss even though it uses the same focus.
        let result = run_optimisation(
            &config,
            &AcceptOnceScorer {
                accept_on_call: 2,
                calls: Arc::new(Mutex::new(0)),
            },
        )
        .unwrap();
        let records = experiment_records(&result.journal_path);
        assert_eq!(result.acceptances, 1, "exactly one accept");
        let accept_at = records
            .iter()
            .position(|r| r.accepted)
            .expect("one experiment accepted");
        assert!(
            records.len() > accept_at + 1,
            "need an experiment after the accept"
        );

        let after = &records[accept_at + 1];
        assert_eq!(
            after.memo_hits, 0,
            "the experiment after an accept analyses a new creature — no hits"
        );
        assert!(after.memo_misses >= 1, "it recomputes the focus scan");
        assert_eq!(after.memo_ms_saved, 0);
        // The winner here is a weight/bias nudge, so neuron and synapse counts
        // are untouched and the coarse journal id cannot see the change. Only
        // the content fingerprint invalidated the memo — which is the whole
        // reason the scope is not keyed on `incumbentId` alone.
        assert_eq!(
            after.incumbent_id, records[accept_at].incumbent_id,
            "a weight-only accept leaves the coarse incumbentId unchanged"
        );
    }

    /// Issue #106: a Phase-G graft rewrites the incumbent before the loop, so
    /// the first experiment must analyse the grafted creature from scratch.
    #[test]
    fn phase_g_graft_leaves_the_memo_cold_for_the_first_experiment() {
        let dir = tempdir().unwrap();
        let (result, _) = graft_replay_run(dir.path());
        assert!(result.acceptances >= 1, "phase-G applied the graft");

        let records = experiment_records(&result.journal_path);
        let first = records.first().expect("at least one experiment ran");
        assert_eq!(
            first.memo_hits, 0,
            "nothing cached against the pre-graft creature may be served"
        );
        assert!(first.memo_misses >= 1);
        assert!(
            first.incumbent_id.ends_with("-s3"),
            "experiments analyse the grafted creature (3 synapses), got {}",
            first.incumbent_id
        );
    }

    /// Cancels the run from inside the first scorer batch, as `SIGINT` would.
    struct CancellingScorer {
        token: CancelToken,
    }

    impl DirectoryScorer for CancellingScorer {
        fn score_directory_sampled(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
            _sample: crate::scorer::ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            self.token.cancel();
            const BASE_SCORE: f64 = 0.64;
            let mut map = BTreeMap::new();
            map.insert(
                "baseline".into(),
                ScoreResult {
                    score: BASE_SCORE,
                    error: 1.0 - BASE_SCORE,
                    complexity_penalty: 0.0,
                },
            );
            for entry in fs::read_dir(candidates_dir).into_iter().flatten().flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(stem) = name.strip_suffix(".json")
                    && stem != "baseline"
                {
                    let score = if stem == "candidate-000" {
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
            Ok(map)
        }
    }

    /// Working directories the loop creates per experiment.
    fn experiment_work_dirs(output_dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(output_dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| {
                n.starts_with("candidates-exp-")
                    || n.starts_with("promote-exp-")
                    || n.starts_with("combos-exp-")
            })
            .collect();
        names.sort();
        names
    }

    /// Issue #72: `--max-experiments` stops the loop inside the time budget.
    #[test]
    fn max_experiments_caps_the_loop() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let out = dir.path().join("out");
        let mut config = reproducibility_config(creature_path, training, out.clone());
        // Generous budget: only the cap can end this run.
        config.timeout = Duration::from_secs(300);
        config.seed = Some(1);
        config.max_experiments = Some(2);

        let result = run_optimisation(
            &config,
            &ScriptedScorer {
                calls: Arc::new(Mutex::new(0)),
            },
        )
        .unwrap();

        assert_eq!(result.experiments, 2, "the cap bounds the experiment count");
        assert_eq!(result.stop_reason, StopReason::MaxExperiments);
        let lines = journal_lines(&result.journal_path);
        let experiments: Vec<&ExperimentRecord> = lines
            .iter()
            .filter_map(|l| match l {
                JournalLine::Experiment(record) => Some(record.as_ref()),
                JournalLine::Header(_)
                | JournalLine::GraftReplay(_)
                | JournalLine::ScorerCalls(_) => None,
            })
            .collect();
        assert_eq!(
            experiments.len(),
            2,
            "every capped experiment is journalled"
        );
        let JournalLine::Header(header) = &lines[0] else {
            panic!("first journal line must be the run header");
        };
        assert_eq!(
            header.config.max_experiments,
            Some(2),
            "the cap is part of the replay contract"
        );
        assert!(result.best_path.is_file());
    }

    /// Issue #72: a cap of zero is a legitimate no-op, not an underflow.
    #[test]
    fn a_zero_experiment_cap_runs_nothing() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let out = dir.path().join("out");
        let mut config = reproducibility_config(creature_path, training, out.clone());
        config.timeout = Duration::from_secs(300);
        config.max_experiments = Some(0);

        let result = run_optimisation(
            &config,
            &ScriptedScorer {
                calls: Arc::new(Mutex::new(0)),
            },
        )
        .unwrap();

        assert_eq!(result.experiments, 0);
        assert_eq!(result.acceptances, 0);
        assert_eq!(result.stop_reason, StopReason::MaxExperiments);
        assert!(experiment_work_dirs(&out).is_empty());
    }

    /// Issue #72: cancellation ends the run through the normal exit path — the
    /// journal, the `best.json` run-summary stamp and the cleanup all survive.
    #[test]
    fn cancellation_stops_the_loop_and_still_stamps_best() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let out = dir.path().join("out");
        let mut config = reproducibility_config(creature_path, training, out.clone());
        // Generous budget: only the cancellation can end this run promptly.
        config.timeout = Duration::from_secs(300);
        config.seed = Some(1);

        let cancel = CancelToken::new();
        let result = run_optimisation_cancellable(
            &config,
            &CancellingScorer {
                token: cancel.clone(),
            },
            &cancel,
        )
        .unwrap();

        assert_eq!(result.stop_reason, StopReason::Cancelled);
        assert_eq!(
            result.experiments, 1,
            "the in-flight experiment finishes, and no further one starts"
        );
        assert_eq!(
            journal_lines(&result.journal_path)
                .into_iter()
                .filter(|l| matches!(l, JournalLine::Experiment(_)))
                .count(),
            1,
            "the finished experiment is journalled before exiting"
        );
        assert!(result.acceptances >= 1, "the winner was accepted");
        let best: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&result.best_path).unwrap()).unwrap();
        let lamarck = best["tags"]
            .as_array()
            .expect("tags")
            .iter()
            .find(|t| t["name"] == "lamarck")
            .expect("lamarck tag")["value"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(
            lamarck.contains("accept") && lamarck.contains("score:"),
            "cancelled run must still re-stamp best.json: {lamarck}"
        );
        assert!(
            experiment_work_dirs(&out).is_empty(),
            "cancellation must not leave working directories behind: {:?}",
            experiment_work_dirs(&out)
        );
    }

    /// Issue #72: a token already set stops the run before any experiment.
    #[test]
    fn cancellation_before_the_first_experiment_runs_none() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let out = dir.path().join("out");
        let mut config = reproducibility_config(creature_path.clone(), training, out.clone());
        config.timeout = Duration::from_secs(300);

        let cancel = CancelToken::new();
        cancel.cancel();
        let result = run_optimisation_cancellable(
            &config,
            &ScriptedScorer {
                calls: Arc::new(Mutex::new(0)),
            },
            &cancel,
        )
        .unwrap();

        assert_eq!(result.experiments, 0);
        assert_eq!(result.stop_reason, StopReason::Cancelled);
        assert!(experiment_work_dirs(&out).is_empty());
        // best.json is still the verbatim copy of the supplied creature.
        assert_eq!(
            fs::read_to_string(&result.best_path).unwrap(),
            fs::read_to_string(&creature_path).unwrap()
        );
    }

    /// Issue #109: one focus keeps the whole budget; several split it with the
    /// remainder going to the earliest (highest-ranked) focuses.
    #[test]
    fn a_candidate_budget_splits_across_the_focus_set() {
        assert_eq!(split_candidate_budget(100, 1), vec![100]);
        assert_eq!(split_candidate_budget(100, 3), vec![34, 33, 33]);
        assert_eq!(split_candidate_budget(6, 3), vec![2, 2, 2]);
        assert_eq!(split_candidate_budget(2, 3), vec![1, 1, 0]);
        assert_eq!(split_candidate_budget(0, 2), vec![0, 0]);
        assert!(split_candidate_budget(10, 0).is_empty());
        // Nothing is dropped: the shares always add back up to the budget.
        for count in [0usize, 1, 7, 29, 100] {
            for focuses in 1..=5 {
                assert_eq!(
                    split_candidate_budget(count, focuses).iter().sum::<usize>(),
                    count,
                    "budget {count} over {focuses} focuses lost candidates"
                );
            }
        }
    }

    /// Issue #109: a merged batch reports the limit that actually bound it — a
    /// focus whose generator ran dry must not be reported as budget-satisfied.
    #[test]
    fn a_merged_batch_reports_the_strictest_limit() {
        assert_eq!(
            merge_batch_limits(&[BatchLimit::Budget, BatchLimit::Budget]),
            BatchLimit::Budget
        );
        assert_eq!(
            merge_batch_limits(&[BatchLimit::Budget, BatchLimit::Exhausted]),
            BatchLimit::Exhausted
        );
        assert_eq!(
            merge_batch_limits(&[BatchLimit::Exhausted, BatchLimit::QuotaCeiling]),
            BatchLimit::QuotaCeiling
        );
        // An empty focus set never reaches here; an empty list is "nothing
        // stopped it short", which is the budget.
        assert_eq!(merge_batch_limits(&[]), BatchLimit::Budget);
    }

    /// Issue #109: a single-focus journal keeps its pre-change shape.
    #[test]
    fn only_a_multi_focus_experiment_journals_a_focus_set() {
        assert_eq!(journal_focus_set(&["o1".to_string()]), None);
        assert_eq!(
            journal_focus_set(&["o1".to_string(), "h1".to_string()]),
            Some(vec!["o1".to_string(), "h1".to_string()])
        );
    }

    fn scored(entries: &[(&str, f64)]) -> BTreeMap<String, ScoreResult> {
        entries
            .iter()
            .map(|(stem, score)| {
                (
                    (*stem).to_string(),
                    ScoreResult {
                        score: *score,
                        error: 1.0 - *score,
                        complexity_penalty: 0.0,
                    },
                )
            })
            .collect()
    }

    /// Candidates whose provenance names `focus`, in stem order.
    fn candidates_for(focuses: &[&str]) -> Vec<Candidate> {
        let creature = parse_creature_json(
            r#"{"semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
                "neurons":[{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
                "synapses":[{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]}"#,
        )
        .unwrap();
        focuses
            .iter()
            .map(|focus| Candidate {
                creature: creature.clone(),
                provenance: CandidateProvenance {
                    strategy: CandidateStrategy::Random,
                    focus_neuron: (*focus).to_string(),
                    mutation: "test".into(),
                    old_value: None,
                    new_value: None,
                },
            })
            .collect()
    }

    /// Issue #109: each focus is judged on its own candidates, never on the
    /// best of the whole batch.
    #[test]
    fn a_focus_is_scored_on_its_own_candidates() {
        let candidates = candidates_for(&["a", "a", "b"]);
        let scores = scored(&[
            ("baseline", 0.5),
            ("candidate-000", 0.5 + 1e-3),
            ("candidate-001", 0.5),
            ("candidate-002", 0.5 - 1e-3),
        ]);

        let baseline = scores
            .get("baseline")
            .expect("the fixture carries one")
            .clone();
        let a = best_focus_delta(&scores, &baseline, &candidates, "a").expect("focus a scored");
        let b = best_focus_delta(&scores, &baseline, &candidates, "b").expect("focus b scored");
        assert!((a - 1e-3).abs() < 1e-12, "focus a takes its own best: {a}");
        assert!(b < 0.0, "focus b must not inherit focus a's winner: {b}");
        assert_eq!(
            best_focus_delta(&scores, &baseline, &candidates, "unseen"),
            None,
            "a focus with no scored candidate has no delta"
        );
    }

    /// Issue #109: an accept in a K=3 batch boosts only the winner's focus; the
    /// other two are dampened as sterile, exactly as a losing experiment is.
    #[test]
    fn an_accept_boosts_only_the_winning_focus() {
        let focus_set = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let candidates = candidates_for(&["a", "b", "c"]);
        let scores = scored(&[
            ("baseline", 0.5),
            ("candidate-000", 0.5),
            ("candidate-001", 0.5 + 1e-3),
            ("candidate-002", 0.5),
        ]);
        let winners = winning_focuses(&[1], &candidates);
        assert_eq!(winners, ["b".to_string()].into_iter().collect());

        let mut selector = WeightedFocusSelector::default();
        record_focus_outcomes(
            &mut selector,
            &focus_set,
            &winners,
            &scores,
            scores.get("baseline").expect("the fixture carries one"),
            &candidates,
            1e-6,
        );

        let history = |uuid: &str| selector.history.get(uuid).cloned().unwrap_or_default();
        assert_eq!(history("b").accepts, 1, "the winner's focus is boosted");
        assert_eq!(history("b").hard_fails, 0);
        for sterile in ["a", "c"] {
            assert_eq!(
                history(sterile).accepts,
                0,
                "focus {sterile} produced no winner"
            );
            assert_eq!(
                history(sterile).hard_fails,
                1,
                "a focus whose candidates went nowhere is dampened"
            );
        }
    }

    /// Issue #109: a combo winner spanning two focuses credits both.
    #[test]
    fn a_combo_winner_credits_every_member_focus() {
        let candidates = candidates_for(&["a", "b", "c"]);
        let winners = winning_focuses(&[0, 2], &candidates);
        assert_eq!(
            winners,
            ["a".to_string(), "c".to_string()].into_iter().collect()
        );
        assert!(
            winning_focuses(&[9], &candidates).is_empty(),
            "an out-of-range member credits nothing rather than guessing"
        );
    }

    /// Issue #109: a scorer failure dampens every focus the batch served, not
    /// just the primary — none of them produced a scored candidate.
    #[test]
    fn a_scorer_failure_dampens_the_whole_focus_set() {
        let mut selector = WeightedFocusSelector::default();
        let focus_set = vec!["a".to_string(), "b".to_string()];
        record_focus_failure(&mut selector, &focus_set, 1e-6);
        for uuid in &focus_set {
            assert_eq!(
                selector.history.get(uuid).map(|h| h.hard_fails),
                Some(1),
                "focus {uuid} carried the scorer failure"
            );
        }
    }

    #[test]
    fn consecutive_scorer_failures_abort() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let config = LamarckConfig {
            creature: creature_path,
            training_data: training,
            timeout: Duration::from_secs(30),
            max_experiments: None,
            candidates: 2,
            scale_candidate_quotas: false,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: dir.path().join("out"),
            preserve_losers: true,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            focus_count: crate::config::DEFAULT_FOCUS_COUNT,
            compute_correlations: false,
            max_consecutive_scorer_failures: 2,
            phase0_parity: false,
            structural_only: false,
            screen_sample_rate: None,
            screen_promote_threshold: 0.0,
            screen_promote_gate: PromoteGateMode::Absolute,
            screen_promote_sigma_k: DEFAULT_SCREEN_PROMOTE_SIGMA_K,
            baseline_reverify_interval: 0,
            baseline_drift_epsilon: Some(crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON),
            grafts_path: None,
            graft_replay_budget: None,
            backprop_learning_rate: None,
            backprop_max_bias_adjustment_scale: None,
            analysis_memo_entries: crate::memo::DEFAULT_ANALYSIS_MEMO_ENTRIES,
            analysis_threads: crate::chunks::DEFAULT_ANALYSIS_THREADS,
        };
        let err = run_optimisation(&config, &FailingScorer).unwrap_err();
        assert!(err.contains("consecutive scorer failures") || err.contains("no successful"));
    }

    // ---------------------------------------------------------------------
    // Issue #113 — remembered full-corpus baseline
    // ---------------------------------------------------------------------

    /// One batch a probe scorer was handed: its stems, and whether the call
    /// sampled (screen) or scored the full corpus (Phase-0 / promote / verify).
    type ProbedBatch = (Vec<String>, bool);

    /// Scores tiny_setup batches while recording exactly which creatures each
    /// call was handed, so a test can assert what the promote directory held.
    ///
    /// Baseline error/score match `tiny_setup`'s local MSE so the Phase-0
    /// parity gate passes; the second and later full-corpus calls can report a
    /// moved baseline, which is the drift these tests exist to catch.
    struct BaselineProbeScorer {
        batches: Arc<Mutex<Vec<ProbedBatch>>>,
        /// Baseline score for the first full-corpus call (Phase-0).
        baseline_first: f64,
        /// Baseline score for every full-corpus call after it.
        baseline_after: f64,
        /// Candidate Δ vs `baseline_first` on a sampled (screen) call.
        screen_delta: f64,
        /// Candidate Δ vs `baseline_first` on a full-corpus call.
        promote_delta: f64,
    }

    impl BaselineProbeScorer {
        fn rejecting() -> Self {
            Self {
                batches: Arc::new(Mutex::new(Vec::new())),
                baseline_first: 0.64,
                baseline_after: 0.64,
                // Promoted by the screen, rejected on the full corpus: the
                // overwhelmingly common shape, and the one the saving is for.
                screen_delta: 1e-4,
                promote_delta: -1e-4,
            }
        }

        /// Full-corpus batches, in call order (Phase-0 first).
        fn full_batches(&self) -> Vec<Vec<String>> {
            self.batches
                .lock()
                .unwrap()
                .iter()
                .filter(|(_, sampled)| !sampled)
                .map(|(stems, _)| stems.clone())
                .collect()
        }
    }

    impl DirectoryScorer for BaselineProbeScorer {
        fn score_directory_sampled(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
            sample: crate::scorer::ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            let mut stems: Vec<String> = fs::read_dir(candidates_dir)
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .strip_suffix(".json")
                        .map(str::to_string)
                })
                .collect();
            stems.sort();
            let sampled = sample.is_subsample();
            let full_calls_before = self
                .batches
                .lock()
                .unwrap()
                .iter()
                .filter(|(_, s)| !s)
                .count();
            self.batches.lock().unwrap().push((stems.clone(), sampled));

            let baseline = if sampled || full_calls_before == 0 {
                self.baseline_first
            } else {
                self.baseline_after
            };
            let delta = if sampled {
                self.screen_delta
            } else {
                self.promote_delta
            };
            let mut map = BTreeMap::new();
            for stem in stems {
                let score = if stem == "baseline" {
                    baseline
                } else {
                    self.baseline_first + delta
                };
                map.insert(
                    stem,
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

    /// Config for the #113 probes: screen + promote, one pinned focus, and the
    /// reuse knobs under test.
    fn reuse_config(
        creature: PathBuf,
        training: PathBuf,
        out: PathBuf,
        interval: u64,
        epsilon: f64,
        max_experiments: u64,
    ) -> LamarckConfig {
        LamarckConfig {
            creature,
            training_data: training,
            timeout: Duration::from_secs(20),
            max_experiments: Some(max_experiments),
            candidates: 3,
            scale_candidate_quotas: false,
            min_improvement: 1e-6,
            seed: Some(7),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: out,
            preserve_losers: false,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("o1".into()),
            focus_policy: FocusPolicy::Random,
            focus_count: crate::config::DEFAULT_FOCUS_COUNT,
            compute_correlations: false,
            max_consecutive_scorer_failures: 3,
            phase0_parity: true,
            structural_only: false,
            screen_sample_rate: Some(0.1),
            screen_promote_threshold: 0.0,
            screen_promote_gate: PromoteGateMode::Absolute,
            screen_promote_sigma_k: DEFAULT_SCREEN_PROMOTE_SIGMA_K,
            baseline_reverify_interval: interval,
            baseline_drift_epsilon: Some(epsilon),
            grafts_path: None,
            graft_replay_budget: None,
            backprop_learning_rate: None,
            backprop_max_bias_adjustment_scale: None,
            analysis_memo_entries: crate::memo::DEFAULT_ANALYSIS_MEMO_ENTRIES,
            analysis_threads: crate::chunks::DEFAULT_ANALYSIS_THREADS,
        }
    }

    /// The saving itself: with a valid remembered score, promote calls carry
    /// candidates only — and the run still rejects them correctly.
    #[test]
    fn promote_calls_omit_the_baseline_when_a_remembered_score_is_valid() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let scorer = BaselineProbeScorer::rejecting();
        let config = reuse_config(
            creature_path,
            training,
            dir.path().join("out"),
            10,
            crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON,
            3,
        );
        let result = run_optimisation(&config, &scorer).unwrap();
        assert_eq!(
            result.acceptances, 0,
            "the probe promotes but never accepts"
        );

        let full = scorer.full_batches();
        assert!(
            full.len() >= 3,
            "expected Phase-0 + promote calls: {full:?}"
        );
        assert_eq!(full[0], vec!["baseline".to_string()], "Phase-0 scores it");
        for batch in &full[1..] {
            assert!(
                !batch.contains(&"baseline".to_string()),
                "a promote call still carried the incumbent: {batch:?}"
            );
            assert!(!batch.is_empty(), "the promote call scored nothing");
        }

        let records = experiment_records(&result.journal_path);
        assert!(!records.is_empty());
        for record in &records {
            assert_eq!(
                record.baseline_source,
                Some(BaselineSource::Remembered),
                "experiment {} did not journal the remembered baseline",
                record.experiment_number
            );
            // The journal still names the score the experiment was judged
            // against, so every existing reader stays correct.
            assert!((record.scores["baseline"] - 0.64).abs() < 1e-12);
        }
    }

    /// The pre-#113 default: every promote call carries the incumbent.
    #[test]
    fn the_default_run_still_pairs_every_promote_call_with_a_fresh_baseline() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let scorer = BaselineProbeScorer::rejecting();
        let config = reuse_config(
            creature_path,
            training,
            dir.path().join("out"),
            0,
            crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON,
            2,
        );
        let result = run_optimisation(&config, &scorer).unwrap();
        for batch in scorer.full_batches() {
            assert!(
                batch.contains(&"baseline".to_string()),
                "the default run dropped the paired baseline: {batch:?}"
            );
        }
        for record in experiment_records(&result.journal_path) {
            assert_eq!(record.baseline_source, Some(BaselineSource::Fresh));
        }
    }

    /// The guard the reuse removes, put back: a baseline that has moved beyond
    /// the epsilon by the time it is re-scored aborts the run.
    #[test]
    fn a_drifted_baseline_aborts_the_run() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let mut scorer = BaselineProbeScorer::rejecting();
        // Every full-corpus call after Phase-0 reports a moved incumbent — the
        // training data or the scorer changed under the run.
        scorer.baseline_after = 0.64 + 1e-3;
        let config = reuse_config(
            creature_path,
            training,
            dir.path().join("out"),
            // One reuse, then a fresh baseline: the re-verification.
            1,
            crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON,
            4,
        );
        let err = run_optimisation(&config, &scorer)
            .expect_err("a moved baseline must stop the run, not be scored against");
        assert!(err.contains("baseline drift"), "{err}");
        assert!(err.contains("--baseline-drift-epsilon"), "{err}");
    }

    /// The worst outcome this issue could cause, made impossible: a margin that
    /// exists only against the remembered score is withdrawn by the fresh pair.
    #[test]
    fn an_accept_whose_margin_needs_the_stale_baseline_is_withdrawn() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let mut scorer = BaselineProbeScorer::rejecting();
        // The candidate beats the remembered 0.64 by 2e-6 …
        scorer.promote_delta = 2e-6;
        // … but the incumbent is really worth 5e-6 more than that now, so the
        // freshly scored pair rejects it. The movement is inside this run's
        // documented epsilon, so it is a withdrawal, not an abort.
        scorer.baseline_after = 0.64 + 5e-6;
        let config = reuse_config(creature_path, training, dir.path().join("out"), 10, 1e-3, 1);
        let result = run_optimisation(&config, &scorer).unwrap();
        assert_eq!(
            result.acceptances, 0,
            "the stale margin must never reach best.json"
        );
        let records = experiment_records(&result.journal_path);
        let promoted = records
            .iter()
            .find(|r| r.baseline_source.is_some_and(|s| s.omitted_baseline()))
            .expect("an experiment promoted off the remembered baseline");
        assert!(!promoted.accepted);
        assert_eq!(
            promoted.baseline_source,
            Some(BaselineSource::RememberedVerified),
            "an accept off a remembered baseline is always verified"
        );
        // The verification call is the pair, scored together.
        let verify = scorer
            .full_batches()
            .into_iter()
            .find(|batch| batch.contains(&"winner".to_string()))
            .expect("the accept was verified against a freshly scored pair");
        assert!(verify.contains(&"baseline".to_string()), "{verify:?}");
    }

    /// The positive control: a real improver survives verification, is swapped
    /// in, and invalidates the remembered score for the new incumbent.
    #[test]
    fn a_real_improver_survives_verification_and_invalidates_the_remembered_score() {
        let dir = tempdir().unwrap();
        let (creature_path, training) = tiny_setup(dir.path());
        let mut scorer = BaselineProbeScorer::rejecting();
        scorer.promote_delta = 2e-5;
        let config = reuse_config(
            creature_path,
            training,
            dir.path().join("out"),
            10,
            crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON,
            2,
        );
        let result = run_optimisation(&config, &scorer).unwrap();
        assert!(
            result.acceptances >= 1,
            "a genuine improver must be accepted"
        );

        let records = experiment_records(&result.journal_path);
        let accepted = records
            .iter()
            .find(|r| r.accepted)
            .expect("an accepted experiment");
        assert_eq!(
            accepted.baseline_source,
            Some(BaselineSource::RememberedVerified)
        );
        assert!(accepted.improvement.is_some_and(|d| d > 1e-6));

        // The accept changed the incumbent, so the promote call after it can no
        // longer reuse the score — it carries the baseline again.
        let full = scorer.full_batches();
        let verify_at = full
            .iter()
            .position(|batch| batch.contains(&"winner".to_string()))
            .expect("the accept was verified");
        if let Some(next) = full.get(verify_at + 1) {
            assert!(
                next.contains(&"baseline".to_string()),
                "the promote call after an accept reused a score of the old incumbent: {next:?}"
            );
        }
    }
}
