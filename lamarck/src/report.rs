//! Benchmark / strategy economics reporting from `experiments.jsonl`.

use crate::baseline::BaselineSource;
use crate::candidates::{BatchLimit, CandidateStrategy};
use crate::log;
use crate::promote_gate::{PromoteGateReplay, PromoteGateReplayAccumulator};
use crate::run::{JournalLine, RunResult};
use crate::scorer_cost::{ScorerCallCost, ScorerCallCostAccumulator};
use crate::screen_calibration::{ScreenCalibration, ScreenCalibrationAccumulator};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Aggregated strategy hit-rate row.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyStats {
    /// Strategy name.
    pub strategy: String,
    /// Times this strategy appeared on an accepted winner provenance.
    ///
    /// A merged combo win counts once for *every* member strategy (issue #74),
    /// so the column sums to more than [`JournalReport::acceptances`] whenever
    /// combos win.
    pub wins: u64,
    /// Subset of [`Self::wins`] earned as a member of a merged combo winner.
    pub combo_wins: u64,
    /// Times this strategy appeared among candidates across all experiments.
    pub appearances_total: u64,
    /// Times this strategy appeared among candidates in accepted experiments.
    pub appearances_in_accepted: u64,
    /// `wins / appearances_total` (0 when no appearances).
    pub acceptance_rate: f64,
}

/// Phase-G graft-replay bucket (issue #74).
///
/// Graft replay runs before the experiment loop and accepts without any
/// candidate stem, so its outcomes are reported separately from experiment
/// acceptances rather than being dropped.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraftReplayStats {
    /// Graft-replay phases recorded in the journal.
    pub replays: u64,
    /// Replays that improved the incumbent.
    pub accepts: u64,
    /// Grafts applied across accepting replays.
    pub grafts_applied: u64,
    /// Cumulative accepted score Δ from graft replay.
    pub cumulative_improvement: f64,
    /// Scorer batches that failed during replay.
    pub scorer_failures: u64,
    /// Replays that aborted with an error instead of completing.
    pub replay_errors: u64,
}

/// Per-focus hit / failure history.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusHistory {
    /// Focus neuron UUID.
    pub focus_neuron: String,
    /// Experiments that selected this focus.
    pub experiments: u64,
    /// Accepted improvements while focused here.
    pub accepts: u64,
    /// Cumulative accepted score Δ while focused here.
    pub cumulative_improvement: f64,
}

/// Aggregated focus-neuron analysis over a bucket of experiments (issue #70).
///
/// Averages are over the experiments in the bucket that carried `focusStats`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusStatsAggregate {
    /// Experiments in this bucket carrying focus statistics.
    pub experiments: u64,
    /// Mean incoming-connection count of the focus neuron.
    pub mean_incoming_count: f64,
    /// Mean saturated-activation fraction.
    pub mean_saturation_fraction: f64,
    /// Mean near-zero ("dead") activation fraction.
    pub mean_near_zero_fraction: f64,
    /// Mean post-activation variance.
    pub mean_post_variance: f64,
    /// Mean absolute backprop blame over the experiments that recorded one.
    pub mean_abs_blame: Option<f64>,
    /// Experiment count per focus squash name.
    pub squash_counts: BTreeMap<String, u64>,
}

/// Focus-neuron aggregates split by experiment outcome (issue #70).
///
/// Comparing `accepted` against `rejected` is what answers the journal's
/// experimental questions 4 (are saturated/dead neurons good targets?) and 6
/// (does propagated blame predict a successful direction?).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusStatsSummary {
    /// Every experiment carrying focus statistics.
    pub all: FocusStatsAggregate,
    /// Experiments that accepted a candidate.
    pub accepted: FocusStatsAggregate,
    /// Experiments that accepted nothing.
    pub rejected: FocusStatsAggregate,
}

/// Streaming accumulator behind one [`CacheReport`] (issue #93).
///
/// Tracks whether the journal said anything about the cache at all, so a
/// pre-cache journal ends with no section rather than an empty one.
#[derive(Debug, Default)]
struct CacheAccumulator {
    header_seen: bool,
    enabled: bool,
    max_entries: Option<usize>,
    max_age_seconds: Option<u64>,
    tolerance_abs: Option<f64>,
    tolerance_rel: Option<f64>,
    max_bytes: Option<usize>,
    standdown_window: Option<usize>,
    standdown_margin: Option<f64>,
    experiments_with_cache: u64,
    proposals_examined: u64,
    cache_hits: u64,
    deduplicated: u64,
    backfilled: u64,
    batch_candidates: u64,
    final_cache_size: Option<usize>,
    peak_cache_size: Option<usize>,
    estimated_saved_ms: f64,
    experiment_spent_ms: f64,
    rebuild_ms: u128,
    stood_down_at_experiment: Option<u64>,
    stood_down_reason: Option<String>,
}

impl CacheAccumulator {
    /// Record the cache knobs the #71 run header declares.
    fn push_header(&mut self, config: &crate::run::RunConfigRecord) {
        self.header_seen = true;
        self.enabled = config.failed_cache;
        self.max_entries = config.failed_cache_max_entries;
        self.max_age_seconds = config.failed_cache_max_age_seconds;
        self.tolerance_abs = config.failed_cache_tolerance_abs;
        self.tolerance_rel = config.failed_cache_tolerance_rel;
        self.max_bytes = config.failed_cache_max_bytes;
        self.standdown_window = config.failed_cache_stand_down_window;
        self.standdown_margin = config.failed_cache_stand_down_margin_ms;
    }

    /// Fold in one experiment's cache activity.
    ///
    /// An experiment with no `cacheSkipped` never consulted the cache — either
    /// the run had it off, or the #92 guardrail had already stood it down — so
    /// it contributes nothing but is still counted out of
    /// [`CacheReport::experiments_with_cache`].
    fn push_experiment(&mut self, record: &crate::run::ExperimentRecord) {
        self.rebuild_ms = self
            .rebuild_ms
            .saturating_add(record.cache_rebuild_ms.unwrap_or(0));

        let Some(hits) = record.cache_skipped else {
            return;
        };
        self.experiments_with_cache += 1;
        self.cache_hits += hits as u64;
        self.deduplicated += record.cache_deduplicated.unwrap_or(0) as u64;
        self.backfilled += record.cache_backfilled.unwrap_or(0) as u64;
        self.batch_candidates += record.candidates.len() as u64;
        // Proposals are not journalled directly, and do not need to be: the
        // filter examined everything it kept plus everything it rejected, and
        // `candidates[]` is the batch it kept.
        self.proposals_examined +=
            (record.candidates.len() + hits + record.cache_deduplicated.unwrap_or(0)) as u64;
        self.estimated_saved_ms += record.cache_saved_ms.unwrap_or(0.0);
        self.experiment_spent_ms += record.cache_spent_ms.unwrap_or_else(|| {
            // A journal from before #92 priced nothing, but still recorded
            // the timings the price is made of.
            (record.cache_lookup_ms.unwrap_or(0) + record.cache_maintenance_ms.unwrap_or(0)) as f64
        });
        if let Some(size) = record.cache_size {
            self.final_cache_size = Some(size);
            self.peak_cache_size = Some(self.peak_cache_size.map_or(size, |peak| peak.max(size)));
        }
    }

    /// Record a #92 stand-down journal line.
    fn push_stand_down(&mut self, record: &crate::run::CacheStandDownRecord) {
        self.stood_down_at_experiment = Some(record.experiment_number);
        self.stood_down_reason = Some(record.message.clone());
    }

    /// The finished section, or `None` when the journal says nothing about a
    /// cache at all.
    fn finish(self) -> Option<CacheReport> {
        let mentioned = self.enabled
            || self.experiments_with_cache > 0
            || self.stood_down_at_experiment.is_some();
        if !mentioned {
            return None;
        }
        let spent_ms = self.experiment_spent_ms + self.rebuild_ms as f64;
        let hit_rate = if self.proposals_examined > 0 {
            self.cache_hits as f64 / self.proposals_examined as f64
        } else {
            0.0
        };
        Some(CacheReport {
            enabled: self.enabled,
            max_entries: self.max_entries,
            max_age_seconds: self.max_age_seconds,
            tolerance_abs: self.tolerance_abs,
            tolerance_rel: self.tolerance_rel,
            max_bytes: self.max_bytes,
            standdown_window: self.standdown_window,
            standdown_margin: self.standdown_margin,
            experiments_with_cache: self.experiments_with_cache,
            proposals_examined: self.proposals_examined,
            cache_hits: self.cache_hits,
            deduplicated: self.deduplicated,
            backfilled: self.backfilled,
            batch_candidates: self.batch_candidates,
            hit_rate,
            final_cache_size: self.final_cache_size,
            peak_cache_size: self.peak_cache_size,
            estimated_saved_ms: self.estimated_saved_ms,
            spent_ms,
            rebuild_ms: self.rebuild_ms,
            net_ms: self.estimated_saved_ms - spent_ms,
            stood_down_at_experiment: self.stood_down_at_experiment,
            stood_down_reason: self.stood_down_reason,
        })
    }
}

/// Streaming accumulator behind one [`FocusStatsAggregate`].
#[derive(Debug, Default)]
struct FocusStatsAccumulator {
    experiments: u64,
    incoming_count: f64,
    saturation_fraction: f64,
    near_zero_fraction: f64,
    post_variance: f64,
    abs_blame: f64,
    blame_records: u64,
    squash_counts: BTreeMap<String, u64>,
}

impl FocusStatsAccumulator {
    fn push(&mut self, stats: &crate::focus::FocusNeuronStats) {
        self.experiments += 1;
        self.incoming_count += stats.incoming_count as f64;
        self.saturation_fraction += stats.saturation_fraction;
        self.near_zero_fraction += stats.near_zero_fraction;
        self.post_variance += stats.post_variance;
        if let Some(blame) = stats.mean_abs_blame.or(stats.mean_blame.map(f64::abs)) {
            self.abs_blame += blame;
            self.blame_records += 1;
        }
        let squash = stats.squash.clone().unwrap_or_else(|| "unknown".into());
        *self.squash_counts.entry(squash).or_default() += 1;
    }

    fn finish(self) -> FocusStatsAggregate {
        let n = self.experiments as f64;
        let mean = |total: f64| if self.experiments > 0 { total / n } else { 0.0 };
        FocusStatsAggregate {
            experiments: self.experiments,
            mean_incoming_count: mean(self.incoming_count),
            mean_saturation_fraction: mean(self.saturation_fraction),
            mean_near_zero_fraction: mean(self.near_zero_fraction),
            mean_post_variance: mean(self.post_variance),
            mean_abs_blame: (self.blame_records > 0)
                .then(|| self.abs_blame / self.blame_records as f64),
            squash_counts: self.squash_counts,
        }
    }
}

/// One point on the improvement-vs-fitness series.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImprovementPoint {
    /// Experiment number.
    pub experiment_number: u64,
    /// Incumbent baseline score at experiment start.
    pub baseline_score: f64,
    /// Absolute improvement when accepted.
    pub improvement: Option<f64>,
    /// Whether a candidate was accepted.
    pub accepted: bool,
}

/// Failed-candidate cache economics for one run journal (issue #93).
///
/// #69's go/no-go is decided from journals, so this states what the cache cost
/// against what it saved, without needing the run that produced it.
///
/// # Attribution
///
/// Three mechanisms keep a proposal out of a scorer batch, and this section
/// credits the cache with exactly one of them:
///
/// * [`Self::cache_hits`] — the cache recognised a known-failed candidate.
///   This, and only this, is the cache's saving.
/// * [`Self::deduplicated`] — the generator proposed the same mutation twice in
///   one batch and the filter dropped the repeat. Real avoided work, reported
///   separately so it cannot inflate the cache's account.
/// * Generation-time gating, such as `backprop` declining to propose on a focus
///   with no accumulated blame (issue #83). Those candidates never become
///   proposals, so they are absent from [`Self::proposals_examined`] entirely
///   and cannot be double-counted here. What they do change is batch size,
///   which is why [`Self::batch_candidates`] is reported rather than assumed
///   to be `--candidates`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheReport {
    /// Whether the run header says the cache was on.
    ///
    /// `false` with cache activity present means the journal is inconsistent;
    /// `false` with no activity is simply a cache-off arm.
    pub enabled: bool,
    /// Size cap in force, from the run header.
    pub max_entries: Option<usize>,
    /// Age bound in seconds, from the run header.
    pub max_age_seconds: Option<u64>,
    /// Near-duplicate absolute tolerance, from the run header.
    pub tolerance_abs: Option<f64>,
    /// Near-duplicate relative tolerance, from the run header.
    pub tolerance_rel: Option<f64>,
    /// Resident byte ceiling, from the run header (issue #92).
    pub max_bytes: Option<usize>,
    /// Stand-down window in experiments, from the run header (issue #92).
    pub standdown_window: Option<usize>,
    /// Stand-down margin, from the run header (issue #92).
    pub standdown_margin: Option<f64>,
    /// Experiments that ran with the cache in service.
    ///
    /// Below the run's experiment count when the cache stood down mid-run.
    pub experiments_with_cache: u64,
    /// Proposals the filter examined, kept and rejected alike.
    pub proposals_examined: u64,
    /// Proposals rejected because the cache knew they fail.
    pub cache_hits: u64,
    /// Proposals dropped as near-duplicates within one batch.
    pub deduplicated: u64,
    /// Replacement proposals accepted to refill a batch.
    pub backfilled: u64,
    /// Candidates that reached a scorer batch across cache-on experiments.
    pub batch_candidates: u64,
    /// `cache_hits / proposals_examined`, `0.0` when nothing was examined.
    pub hit_rate: f64,
    /// Live cache entries after the last cache-on experiment.
    pub final_cache_size: Option<usize>,
    /// Largest live entry count seen across the run.
    pub peak_cache_size: Option<usize>,
    /// Estimated scorer milliseconds the cache's hits avoided.
    ///
    /// Summed from the per-experiment `cacheSavedMs` the run itself computed;
    /// see [`crate::failed_cache::economics`] for the estimator and its
    /// deliberate conservatism.
    pub estimated_saved_ms: f64,
    /// Milliseconds the cache cost: lookups, maintenance and the one-off
    /// startup rebuild.
    pub spent_ms: f64,
    /// Milliseconds of that spend attributable to the startup rebuild.
    pub rebuild_ms: u128,
    /// [`Self::estimated_saved_ms`] less [`Self::spent_ms`]; negative means the
    /// cache lost the run time.
    pub net_ms: f64,
    /// Experiment number at which the #92 guardrail stood the cache down.
    pub stood_down_at_experiment: Option<u64>,
    /// Why the guardrail fired.
    pub stood_down_reason: Option<String>,
}

/// Summary report for a Lamarck run journal.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JournalReport {
    /// Experiments attempted.
    pub experiments: u64,
    /// Accepted improvements.
    pub acceptances: u64,
    /// Scorer batch failures recorded in the journal.
    pub scorer_failures: u64,
    /// Opening incumbent score, anchored on a **full-corpus** baseline (#84).
    ///
    /// This is the `scores.baseline` of the first experiment that actually
    /// promoted to full-corpus scoring — the same number Phase-0 measured when
    /// it ran, because the incumbent cannot change before the first acceptance.
    /// It is `None` until such an experiment exists: an experiment whose
    /// candidate batch screened empty only ever recorded a subsample baseline,
    /// which is not comparable with an authoritative score.
    pub opening_baseline_score: Option<f64>,
    /// Time to first acceptance (sum of analysis+scorer ms until first accept).
    pub time_to_first_acceptance_ms: Option<u128>,
    /// Total analysis milliseconds.
    pub total_analysis_ms: u128,
    /// Total scorer milliseconds.
    pub total_scorer_ms: u128,
    /// Wall duration from first→last `timestampUnix` when available (ms).
    pub wall_duration_ms: Option<u128>,
    /// Promote-phase candidates scored (excludes baseline stems; excludes failures).
    pub candidates_scored: u64,
    /// Screen-phase candidate stems counted when `screenScores` present.
    pub screen_candidates_scored: u64,
    /// Promote candidates scored per minute of scorer wall time.
    pub candidates_per_scorer_minute: f64,
    /// Screen + promote candidate stems per minute of scorer wall time.
    pub candidates_per_screen_minute: f64,
    /// Promote candidates per wall-clock minute (when timestamps span > 0).
    pub candidates_per_wall_minute: Option<f64>,
    /// Analysis share of (analysis+scorer) time.
    pub analysis_time_fraction: f64,
    /// Absolute improvement from [`Self::opening_baseline_score`] to the last
    /// accepted score; `None` until both anchors are full-corpus scores (#84).
    pub total_score_improvement: Option<f64>,
    /// Relative improvement `total / opening` when opening > 0.
    pub relative_score_improvement: Option<f64>,
    /// #69's gate metric: full-corpus score improvement per wall-clock hour.
    ///
    /// `None` whenever the journal cannot support it — no full-corpus anchor
    /// (issue #84), or no wall-clock span between the first and last
    /// experiment. A cache-on and a cache-off journal are compared on this
    /// number, so reporting `0.0` for "unknown" would decide the go/no-go on a
    /// value nobody measured.
    pub score_improvement_per_wall_hour: Option<f64>,
    /// Mean experiment duration (analysis+scorer ms).
    pub mean_experiment_ms: f64,
    /// Projected batches completable in a 45-minute budget from mean duration.
    pub projected_batches_per_45_min: f64,
    /// Focus neurons seen (uuid → experiment count).
    pub focus_counts: BTreeMap<String, u64>,
    /// Focus hit/failure history.
    pub focus_history: Vec<FocusHistory>,
    /// Focus structure / statistics / blame aggregates, split by outcome.
    ///
    /// `None` when no experiment in the journal recorded `focusStats`.
    pub focus_stats: Option<FocusStatsSummary>,
    /// Improvement-vs-fitness series (one point per experiment).
    pub improvement_series: Vec<ImprovementPoint>,
    /// Per-strategy win / appearance / acceptance-rate rows.
    pub strategies: Vec<StrategyStats>,
    /// Sum of `combosScored` across experiments (combo batch size for #63 tuning).
    pub combos_scored_total: u64,
    /// Sum of `combosDampened` across experiments.
    pub combos_dampened_total: u64,
    /// Acceptances whose winner was a multi-member combo.
    pub combo_acceptances: u64,
    /// Combo acceptances that also recorded a non-empty `comboDampen`.
    pub combo_acceptances_with_dampen: u64,
    /// Combo acceptances whose members could not be attributed to a strategy.
    ///
    /// A journal written before `comboMemberIndices` existed (issue #74) names
    /// only the merged `combo-NNN-kM` stem, so its members are unknowable.
    pub combo_acceptances_unattributed: u64,
    /// Phase-G graft-replay bucket; `None` when the journal has no replay line.
    pub graft_replay: Option<GraftReplayStats>,
    /// Cross-experiment analysis-memo economics (issue #106).
    pub analysis_memo: AnalysisMemoStats,
    /// Achieved candidate batch size per experiment (issue #108).
    pub candidate_batch: CandidateBatchStats,
    /// How well the 5% screen predicted the full-corpus score (issue #110).
    pub screen_calibration: ScreenCalibration,
    /// Promote-call baseline economics (issue #113).
    ///
    /// How many promote calls reused the run's remembered full-corpus baseline
    /// instead of re-scoring the incumbent, what that saved in creature-scores,
    /// and how many accepts were re-decided against a freshly scored pair.
    pub baseline_reuse: BaselineReuseStats,
    /// Fixed vs marginal scorer cost per call, fitted per phase (issue #112).
    ///
    /// The intercept of call time against creature count is what a persistent
    /// scoring session could save; the slope is what it could not. Fitted from
    /// the journal's own `scorerCalls`, so any run reproduces the measurement.
    pub scorer_call_cost: ScorerCallCost,
    /// Failed-candidate cache economics (issue #93). `None` on a cache-off journal.
    pub cache: Option<CacheReport>,
    /// What the noise-aware promote gate would have done to this journal (#111).
    ///
    /// Replayed offline from the journal's own `screenScores`, at the default
    /// σ̂ multiplier, so a gate change can be priced — and its effect on the
    /// accepts that were actually earned checked — without any box time.
    pub promote_gate_replay: PromoteGateReplay,
}

/// Achieved candidate batch size across a journal (issue #108).
///
/// A journal written before the generator recorded its budget reports the
/// achieved sizes with `requested` `None`: the batches are real, only the
/// budget they were measured against is unknown.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateBatchStats {
    /// Mean candidates actually generated per experiment.
    pub mean_generated: f64,
    /// Smallest batch generated.
    pub min_generated: usize,
    /// Largest batch generated.
    pub max_generated: usize,
    /// `--candidates` recorded by the experiments, when they agree on one value.
    pub requested: Option<usize>,
    /// Experiments whose batch stopped at the fixed pre-#108 quota ceiling.
    pub quota_ceiling_experiments: u64,
    /// Experiments whose generator ran genuinely dry.
    pub exhausted_experiments: u64,
}

/// Promote-call baseline economics summed over a journal (issue #113).
///
/// A journal written before the field existed — or a run with reuse off —
/// reports every promote call as `fresh`, which is exactly what it was.
#[derive(Debug, Clone, Copy, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BaselineReuseStats {
    /// Promote calls that carried the incumbent and scored it.
    pub fresh_promote_calls: u64,
    /// Promote calls that omitted the incumbent and reused a known score.
    pub remembered_promote_calls: u64,
    /// Accepts re-decided against a freshly scored baseline + winner pair.
    pub verified_accepts: u64,
    /// Full-corpus creature-scores the omitted baselines saved (one each).
    pub baseline_scores_saved: u64,
    /// Creature-scores the accept verifications cost (baseline + winner each).
    pub verification_creature_scores: u64,
    /// Saved minus verification cost — never an over-claim.
    pub net_creature_scores_saved: i64,
    /// Share of promote calls that reused a remembered baseline.
    pub remembered_fraction: f64,
}

/// Cross-experiment analysis-memo economics summed over a journal (issue #106).
///
/// A journal written before the memo existed reports zeros: its experiments
/// recomputed everything, which is exactly what zero hits and zero saved
/// milliseconds mean.
#[derive(Debug, Clone, Copy, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisMemoStats {
    /// Analysis lookups served from the memo.
    pub hits: u64,
    /// Analysis lookups that had to be recomputed.
    pub misses: u64,
    /// Training-scan milliseconds avoided by the hits.
    pub ms_saved: u128,
    /// `hits / (hits + misses)`; `0.0` when nothing was looked up.
    pub hit_rate: f64,
    /// Saved milliseconds as a share of analysis + saved time.
    pub analysis_ms_saved_fraction: f64,
}

/// Consume `experiments.jsonl` and produce an economics report.
pub fn report_from_journal(path: &Path) -> Result<JournalReport, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut experiments = 0u64;
    let mut acceptances = 0u64;
    let mut scorer_failures = 0u64;
    let mut total_analysis_ms = 0u128;
    let mut total_scorer_ms = 0u128;
    let mut memo_hits = 0u64;
    let mut memo_misses = 0u64;
    let mut memo_ms_saved = 0u128;
    let mut candidates_scored = 0u64;
    let mut screen_candidates_scored = 0u64;
    let mut time_to_first = None;
    let mut elapsed = 0u128;
    let mut first_baseline = None;
    let mut last_best = None;
    let mut first_ts: Option<u64> = None;
    let mut last_ts: Option<u64> = None;
    let mut wins: BTreeMap<String, u64> = BTreeMap::new();
    let mut combo_wins: BTreeMap<String, u64> = BTreeMap::new();
    let mut appearances_total: BTreeMap<String, u64> = BTreeMap::new();
    let mut appearances_in_accepted: BTreeMap<String, u64> = BTreeMap::new();
    let mut focus_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut focus_accepts: BTreeMap<String, u64> = BTreeMap::new();
    let mut focus_delta: BTreeMap<String, f64> = BTreeMap::new();
    let mut improvement_series = Vec::new();
    let mut combos_scored_total = 0u64;
    let mut combos_dampened_total = 0u64;
    let mut combo_acceptances = 0u64;
    let mut combo_acceptances_with_dampen = 0u64;
    let mut combo_acceptances_unattributed = 0u64;
    let mut graft_replay: Option<GraftReplayStats> = None;
    let mut batch_generated_total = 0u64;
    let mut batch_min: Option<usize> = None;
    let mut batch_max = 0usize;
    let mut batch_requested: Option<usize> = None;
    let mut batch_requested_mixed = false;
    let mut batch_quota_ceiling = 0u64;
    let mut batch_exhausted = 0u64;
    let mut baseline_reuse = BaselineReuseStats::default();
    let mut scorer_call_cost = ScorerCallCostAccumulator::default();
    let mut screen_calibration = ScreenCalibrationAccumulator::default();
    let mut promote_gate_replay = PromoteGateReplayAccumulator::default();
    let mut focus_all = FocusStatsAccumulator::default();
    let mut focus_accepted = FocusStatsAccumulator::default();
    let mut focus_rejected = FocusStatsAccumulator::default();
    let mut cache = CacheAccumulator::default();

    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        // The run header (issue #71) is run metadata, not an experiment; the
        // graft-replay line (issue #74) is its own bucket.
        let record = match JournalLine::parse(&line)? {
            JournalLine::Header(header) => {
                screen_calibration.push_header(&header.config);
                promote_gate_replay.push_header(&header.config);
                cache.push_header(&header.config);
                continue;
            }
            JournalLine::ScorerCalls(calls) => {
                // Calls made outside any experiment — the Phase-0 baseline —
                // are part of the cost model too (issue #112).
                scorer_call_cost.push_all(&calls.calls);
                continue;
            }
            JournalLine::GraftReplay(replay) => {
                if let Some(calls) = &replay.scorer_calls {
                    scorer_call_cost.push_all(calls);
                }
                let bucket = graft_replay.get_or_insert_with(GraftReplayStats::default);
                bucket.replays += 1;
                bucket.scorer_failures += replay.scorer_failures;
                if replay.replay_error.is_some() {
                    bucket.replay_errors += 1;
                }
                if replay.accepted {
                    bucket.accepts += 1;
                    bucket.grafts_applied += replay.grafts_applied as u64;
                    bucket.cumulative_improvement += replay.improvement.unwrap_or(0.0);
                }
                continue;
            }
            JournalLine::CacheStandDown(stand_down) => {
                cache.push_stand_down(&stand_down);
                continue;
            }
            JournalLine::Experiment(record) => *record,
        };
        experiments += 1;
        screen_calibration.push_experiment(&record)?;
        promote_gate_replay.push_experiment(&record)?;
        memo_hits += record.memo_hits;
        memo_misses += record.memo_misses;
        memo_ms_saved = memo_ms_saved.saturating_add(record.memo_ms_saved);
        total_analysis_ms += record.analysis_ms;
        total_scorer_ms += record.scorer_ms;
        if let Some(calls) = &record.scorer_calls {
            scorer_call_cost.push_all(calls);
        }
        match record.baseline_source {
            Some(BaselineSource::Fresh) => baseline_reuse.fresh_promote_calls += 1,
            Some(source) => {
                baseline_reuse.remembered_promote_calls += 1;
                if source == BaselineSource::RememberedVerified {
                    baseline_reuse.verified_accepts += 1;
                }
            }
            // No promote call ran (the screen was empty, or the batch failed).
            None => {}
        }
        elapsed += record.analysis_ms + record.scorer_ms;
        first_ts = Some(first_ts.map_or(record.timestamp_unix, |t| t.min(record.timestamp_unix)));
        last_ts = Some(last_ts.map_or(record.timestamp_unix, |t| t.max(record.timestamp_unix)));
        // An experiment serves every focus in its set (issue #109); a
        // pre-#109 journal has no set and names exactly one focus.
        for focus in record_focus_set(&record) {
            *focus_counts.entry(focus).or_default() += 1;
        }

        if let Some(stats) = &record.focus_stats {
            focus_all.push(stats);
            if record.accepted {
                focus_accepted.push(stats);
            } else {
                focus_rejected.push(stats);
            }
        }

        for prov in &record.candidates {
            let key = strategy_name(prov.strategy);
            *appearances_total.entry(key).or_default() += 1;
        }
        cache.push_experiment(&record);

        batch_generated_total += record.candidates.len() as u64;
        batch_min = Some(batch_min.map_or(record.candidates.len(), |n: usize| {
            n.min(record.candidates.len())
        }));
        batch_max = batch_max.max(record.candidates.len());
        if let Some(requested) = record.candidates_requested {
            match batch_requested {
                None => batch_requested = Some(requested),
                // Journals from an A/B with several budgets have no single one.
                Some(seen) if seen != requested => batch_requested_mixed = true,
                Some(_) => {}
            }
        }
        match record.batch_limit {
            Some(BatchLimit::QuotaCeiling) => batch_quota_ceiling += 1,
            Some(BatchLimit::Exhausted) => batch_exhausted += 1,
            Some(BatchLimit::Budget) | None => {}
        }

        if record.scorer_error.is_some() {
            scorer_failures += 1;
        } else {
            candidates_scored += record.scores.len().saturating_sub(1) as u64;
            if let Some(screen) = &record.screen_scores {
                screen_candidates_scored += screen.len().saturating_sub(1) as u64;
            }
        }
        // Anchor the opening baseline on a full-corpus score only (issue #84).
        // An experiment whose batch screened empty records the subsample
        // baseline, which swings by ~5e-3 between experiments — thousands of
        // times the accept threshold — so anchoring on it makes
        // `totalScoreImprovement` subtract two different quantities and can
        // report a negative total for a run that only ever improved.
        if first_baseline.is_none()
            && let Some(full_corpus_baseline) = record.scores.get("baseline")
        {
            first_baseline = Some(*full_corpus_baseline);
        }

        improvement_series.push(ImprovementPoint {
            experiment_number: record.experiment_number,
            baseline_score: record.baseline_score,
            improvement: record.improvement,
            accepted: record.accepted,
        });

        if let Some(n) = record.combos_scored {
            combos_scored_total += n as u64;
        }
        if let Some(n) = record.combos_dampened {
            combos_dampened_total += n as u64;
        }

        if record.accepted {
            acceptances += 1;
            if time_to_first.is_none() {
                time_to_first = Some(elapsed);
            }
            if record.combo_members.is_some_and(|m| m > 1) {
                combo_acceptances += 1;
                if record.combo_dampen.as_ref().is_some_and(|d| !d.is_empty()) {
                    combo_acceptances_with_dampen += 1;
                }
            }
            if let Some(delta) = record.improvement {
                let base = last_best.unwrap_or(record.baseline_score);
                last_best = Some(base + delta);
                // Credit the focus the winner was actually proposed against,
                // not whichever focus the experiment drew first (issue #109).
                let credited = accepted_focuses(&record);
                let share = delta / credited.len() as f64;
                for focus in credited {
                    *focus_accepts.entry(focus.clone()).or_default() += 1;
                    *focus_delta.entry(focus).or_default() += share;
                }
            }
            for prov in &record.candidates {
                let key = strategy_name(prov.strategy);
                *appearances_in_accepted.entry(key).or_default() += 1;
            }
            // Attribute the win to every member of the winner: a merged combo
            // (`combo-NNN-kM`) has no single owning strategy (issue #74).
            let members = winner_member_indices(&record);
            let is_combo = members.len() > 1;
            let mut attributed = 0u64;
            for idx in members {
                let Some(prov) = record.candidates.get(idx) else {
                    continue;
                };
                let key = strategy_name(prov.strategy);
                *wins.entry(key.clone()).or_default() += 1;
                if is_combo {
                    *combo_wins.entry(key).or_default() += 1;
                }
                attributed += 1;
            }
            // A pre-#74 journal names only the merged stem, so its member
            // strategies are unknowable — count the gap rather than hide it.
            if attributed == 0 && record.winner.is_some() {
                combo_acceptances_unattributed += 1;
            }
        }
    }

    let mut strategies: Vec<StrategyStats> = appearances_total
        .iter()
        .map(|(strategy, &total)| {
            let w = *wins.get(strategy).unwrap_or(&0);
            let in_acc = *appearances_in_accepted.get(strategy).unwrap_or(&0);
            StrategyStats {
                strategy: strategy.clone(),
                wins: w,
                combo_wins: *combo_wins.get(strategy).unwrap_or(&0),
                appearances_total: total,
                appearances_in_accepted: in_acc,
                acceptance_rate: if total > 0 {
                    w as f64 / total as f64
                } else {
                    0.0
                },
            }
        })
        .collect();
    strategies.sort_by(|a, b| {
        b.wins
            .cmp(&a.wins)
            .then_with(|| a.strategy.cmp(&b.strategy))
    });

    let focus_history: Vec<FocusHistory> = focus_counts
        .iter()
        .map(|(uuid, &exps)| FocusHistory {
            focus_neuron: uuid.clone(),
            experiments: exps,
            accepts: *focus_accepts.get(uuid).unwrap_or(&0),
            cumulative_improvement: *focus_delta.get(uuid).unwrap_or(&0.0),
        })
        .collect();

    let focus_stats = (focus_all.experiments > 0).then(|| FocusStatsSummary {
        all: focus_all.finish(),
        accepted: focus_accepted.finish(),
        rejected: focus_rejected.finish(),
    });

    let total_score_improvement = match (first_baseline, last_best) {
        (Some(first), Some(last)) => Some(last - first),
        _ => None,
    };
    let relative_score_improvement = match (first_baseline, total_score_improvement) {
        (Some(open), Some(delta)) if open.abs() > f64::EPSILON => Some(delta / open),
        _ => None,
    };
    let total_ms = (total_analysis_ms + total_scorer_ms) as f64;
    let analysis_time_fraction = if total_ms > 0.0 {
        total_analysis_ms as f64 / total_ms
    } else {
        0.0
    };
    let candidates_per_scorer_minute = if total_scorer_ms > 0 {
        candidates_scored as f64 / (total_scorer_ms as f64 / 60_000.0)
    } else {
        0.0
    };
    let screen_total = screen_candidates_scored + candidates_scored;
    let candidates_per_screen_minute = if total_scorer_ms > 0 {
        screen_total as f64 / (total_scorer_ms as f64 / 60_000.0)
    } else {
        0.0
    };
    let wall_duration_ms = match (first_ts, last_ts) {
        (Some(a), Some(b)) if b > a => Some(((b - a) as u128).saturating_mul(1000)),
        _ => None,
    };
    let candidates_per_wall_minute = wall_duration_ms.and_then(|wall| {
        if wall > 0 {
            Some(candidates_scored as f64 / (wall as f64 / 60_000.0))
        } else {
            None
        }
    });
    let memo_lookups = memo_hits + memo_misses;
    let analysis_memo = AnalysisMemoStats {
        hits: memo_hits,
        misses: memo_misses,
        ms_saved: memo_ms_saved,
        hit_rate: if memo_lookups > 0 {
            memo_hits as f64 / memo_lookups as f64
        } else {
            0.0
        },
        // What the analysis phase would have cost without the memo is its
        // measured cost plus what the memo saved.
        analysis_ms_saved_fraction: if total_analysis_ms + memo_ms_saved > 0 {
            memo_ms_saved as f64 / (total_analysis_ms + memo_ms_saved) as f64
        } else {
            0.0
        },
    };
    let mean_experiment_ms = if experiments > 0 {
        total_ms / experiments as f64
    } else {
        0.0
    };
    let projected_batches_per_45_min = if mean_experiment_ms > 0.0 {
        (45.0 * 60_000.0) / mean_experiment_ms
    } else {
        0.0
    };
    // #69's gate metric. Both anchors have to be real: the improvement is only
    // computed from full-corpus scores (#84), and a journal with no wall-clock
    // span cannot support a per-hour rate at all.
    let score_improvement_per_wall_hour = match (total_score_improvement, wall_duration_ms) {
        (Some(delta), Some(wall)) if wall > 0 => Some(delta / (wall as f64 / 3_600_000.0)),
        _ => None,
    };

    Ok(JournalReport {
        experiments,
        acceptances,
        scorer_failures,
        opening_baseline_score: first_baseline,
        time_to_first_acceptance_ms: time_to_first,
        total_analysis_ms,
        total_scorer_ms,
        wall_duration_ms,
        candidates_scored,
        screen_candidates_scored,
        candidates_per_scorer_minute,
        candidates_per_screen_minute,
        candidates_per_wall_minute,
        analysis_time_fraction,
        total_score_improvement,
        relative_score_improvement,
        score_improvement_per_wall_hour,
        mean_experiment_ms,
        projected_batches_per_45_min,
        focus_counts,
        focus_history,
        focus_stats,
        improvement_series,
        strategies,
        combos_scored_total,
        combos_dampened_total,
        combo_acceptances,
        combo_acceptances_with_dampen,
        combo_acceptances_unattributed,
        graft_replay,
        analysis_memo,
        candidate_batch: CandidateBatchStats {
            mean_generated: if experiments > 0 {
                batch_generated_total as f64 / experiments as f64
            } else {
                0.0
            },
            min_generated: batch_min.unwrap_or(0),
            max_generated: batch_max,
            requested: batch_requested.filter(|_| !batch_requested_mixed),
            quota_ceiling_experiments: batch_quota_ceiling,
            exhausted_experiments: batch_exhausted,
        },
        baseline_reuse: {
            let promote_calls =
                baseline_reuse.fresh_promote_calls + baseline_reuse.remembered_promote_calls;
            baseline_reuse.baseline_scores_saved = baseline_reuse.remembered_promote_calls;
            baseline_reuse.verification_creature_scores = baseline_reuse.verified_accepts * 2;
            baseline_reuse.net_creature_scores_saved = baseline_reuse.baseline_scores_saved as i64
                - baseline_reuse.verification_creature_scores as i64;
            baseline_reuse.remembered_fraction = if promote_calls > 0 {
                baseline_reuse.remembered_promote_calls as f64 / promote_calls as f64
            } else {
                0.0
            };
            baseline_reuse
        },
        screen_calibration: screen_calibration.finish(),
        scorer_call_cost: scorer_call_cost.finish(),
        promote_gate_replay: promote_gate_replay.finish(),
        cache: cache.finish(),
    })
}

/// Candidate indices behind an accepted winner (issue #74).
///
/// `comboMemberIndices` is authoritative when present — it is the only thing
/// that names the members of a merged `combo-NNN-kM` winner. Journals written
/// before that field fall back to parsing a `candidate-NNN` stem, which is all
/// a single-candidate win ever needed.
fn winner_member_indices(record: &crate::run::ExperimentRecord) -> Vec<usize> {
    if let Some(indices) = &record.combo_member_indices
        && !indices.is_empty()
    {
        return indices.clone();
    }
    record
        .winner
        .as_deref()
        .and_then(|stem| stem.strip_prefix("candidate-"))
        .and_then(|s| s.parse::<usize>().ok())
        .map(|idx| vec![idx])
        .unwrap_or_default()
}

/// Every focus neuron an experiment proposed against (issue #109).
///
/// A journal written before the focus set existed names one focus, which is
/// exactly the set it served.
fn record_focus_set(record: &crate::run::ExperimentRecord) -> Vec<String> {
    match &record.focus_neurons {
        Some(set) if !set.is_empty() => set.clone(),
        _ => vec![record.focus_neuron.clone()],
    }
}

/// Focuses credited with an acceptance (issue #109).
///
/// Derived from the winner's members, each of which names the focus it was
/// proposed for. Falls back to the record's primary focus when the members
/// cannot be resolved — a pre-#74 journal names only a merged combo stem.
fn accepted_focuses(record: &crate::run::ExperimentRecord) -> Vec<String> {
    let mut focuses: Vec<String> = winner_member_indices(record)
        .into_iter()
        .filter_map(|idx| record.candidates.get(idx))
        .map(|prov| prov.focus_neuron.clone())
        .collect();
    focuses.sort();
    focuses.dedup();
    if focuses.is_empty() {
        focuses.push(record.focus_neuron.clone());
    }
    focuses
}

fn strategy_name(strategy: CandidateStrategy) -> String {
    strategy.label().to_string()
}

fn format_ms(ms: u128) -> String {
    if ms >= 60_000 {
        let secs = ms as f64 / 1000.0;
        format!("{secs:.1}s")
    } else if ms >= 1000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{ms}ms")
    }
}

/// One-line rendering of a focus aggregate for the human run summary.
fn format_focus(aggregate: &FocusStatsAggregate) -> String {
    let squashes = aggregate
        .squash_counts
        .iter()
        .map(|(squash, n)| format!("{squash}×{n}"))
        .collect::<Vec<_>>()
        .join(" ");
    let blame = match aggregate.mean_abs_blame {
        Some(blame) => format!("{blame:.3e}"),
        None => "n/a".to_string(),
    };
    format!(
        "{} exp  incoming {:.1}  sat {:.3}  dead {:.3}  postVar {:.3e}  |blame| {blame}  [{squashes}]",
        aggregate.experiments,
        aggregate.mean_incoming_count,
        aggregate.mean_saturation_fraction,
        aggregate.mean_near_zero_fraction,
        aggregate.mean_post_variance,
    )
}

/// Print a coloured human summary of a completed optimisation run to stderr.
pub fn print_run_summary(result: &RunResult) {
    log::info("run summary");
    log::detail(&format!(
        "experiments:  {}  (accepted {}  scorer_ok {}  scorer_fail {})",
        result.experiments, result.acceptances, result.scorer_successes, result.scorer_failures
    ));
    log::detail(&format!(
        "seed:          {}  (replay with --seed {})",
        result.seed, result.seed
    ));
    log::detail(&format!("stopped on:    {}", result.stop_reason.label()));

    if let Ok(report) = report_from_journal(&result.journal_path) {
        let open = result
            .opening_baseline_score
            .or(report.opening_baseline_score);
        if let Some(open) = open {
            let delta = result.best_score - open;
            if result.acceptances == 0 {
                log::detail(&format!(
                    "best score:    {}  (no improvement over opening {})",
                    result.best_score, open
                ));
            } else {
                log::detail(&format!(
                    "best score:    {}  (opening {}  Δ {delta:+.6e})",
                    result.best_score, open
                ));
            }
        } else {
            log::detail(&format!("best score:    {}", result.best_score));
        }

        let total_ms = report.total_analysis_ms + report.total_scorer_ms;
        log::detail(&format!(
            "time:          analysis {}  + scorer {}  = {}  (analysis {:.0}%)",
            format_ms(report.total_analysis_ms),
            format_ms(report.total_scorer_ms),
            format_ms(total_ms),
            report.analysis_time_fraction * 100.0
        ));
        log::detail(&format!(
            "throughput:    {:.1} promote/scorer-min ({} scored); ~{:.1} batches/45min",
            report.candidates_per_scorer_minute,
            report.candidates_scored,
            report.projected_batches_per_45_min
        ));
        let batch = &report.candidate_batch;
        if report.experiments > 0 {
            let requested = match batch.requested {
                Some(n) => format!("{n} requested"),
                None => "budget unrecorded".to_string(),
            };
            log::detail(&format!(
                "batch size:    mean {:.1} generated ({}–{} range, {requested}); \
                 quota-ceiling {}  exhausted {}",
                batch.mean_generated,
                batch.min_generated,
                batch.max_generated,
                batch.quota_ceiling_experiments,
                batch.exhausted_experiments
            ));
        }
        let memo = &report.analysis_memo;
        if memo.hits + memo.misses > 0 {
            log::detail(&format!(
                "analysis memo: {} hit / {} miss ({:.0}% hit rate)  saved {}  ({:.0}% of analysis)",
                memo.hits,
                memo.misses,
                memo.hit_rate * 100.0,
                format_ms(memo.ms_saved),
                memo.analysis_ms_saved_fraction * 100.0
            ));
        }
        let reuse = &report.baseline_reuse;
        if reuse.remembered_promote_calls > 0 {
            log::detail(&format!(
                "baseline:      {} remembered / {} fresh promote call(s) ({:.0}%); \
                 {} creature-score(s) saved net, {} accept(s) verified",
                reuse.remembered_promote_calls,
                reuse.fresh_promote_calls,
                reuse.remembered_fraction * 100.0,
                reuse.net_creature_scores_saved,
                reuse.verified_accepts
            ));
        }
        let cost = &report.scorer_call_cost;
        for (phase, fit) in &cost.by_phase {
            match (fit.fixed_ms, fit.marginal_ms_per_creature) {
                (Some(fixed), Some(marginal)) => log::detail(&format!(
                    "scorer {phase:<8}{} call(s)  fixed {}/call  marginal {}/creature  ({:.0}% fixed at {:.1} creatures)",
                    fit.calls,
                    format_ms(fixed.max(0.0) as u128),
                    format_ms(marginal.max(0.0) as u128),
                    fit.fixed_ms_share_at_mean.unwrap_or(0.0) * 100.0,
                    fit.mean_creatures
                )),
                // One batch size cannot separate fixed from marginal cost;
                // report the means rather than invent an intercept (#112).
                _ => log::detail(&format!(
                    "scorer {phase:<8}{} call(s)  mean {} over {:.1} creatures  (one batch size — no split)",
                    fit.calls,
                    format_ms(fit.mean_ms as u128),
                    fit.mean_creatures
                )),
            }
        }
        let screen = &report.screen_calibration;
        if screen.paired_candidates > 0 {
            let rho = match screen.spearman {
                Some(rho) => format!("{rho:+.2}"),
                None => "n/a".to_string(),
            };
            log::detail(&format!(
                "screen gate:   {} paired (of {} screened)  rank ρ {rho}  precision {}",
                screen.paired_candidates,
                screen.paired_candidates + screen.screen_only_candidates,
                match screen.promotion_precision {
                    Some(p) => format!("{:.0}%", p * 100.0),
                    None => "n/a".to_string(),
                }
            ));
        }
        if let Some(ms) = report.time_to_first_acceptance_ms {
            log::detail(&format!("first accept:  {}", format_ms(ms)));
        }
        if report.combos_scored_total > 0 || report.combo_acceptances > 0 {
            log::detail(&format!(
                "combos:        scored {}  dampened {}  accepts {}  (with dampen {})  exponent={}",
                report.combos_scored_total,
                report.combos_dampened_total,
                report.combo_acceptances,
                report.combo_acceptances_with_dampen,
                crate::combos::STACK_DAMPEN_EXPONENT
            ));
        }
        if let Some(cache) = &report.cache {
            log::detail(&format!(
                "failed cache:  hits {}/{} ({:.1}%)  backfilled {}  dedup {}  entries {} (peak {})",
                cache.cache_hits,
                cache.proposals_examined,
                cache.hit_rate * 100.0,
                cache.backfilled,
                cache.deduplicated,
                cache.final_cache_size.unwrap_or(0),
                cache.peak_cache_size.unwrap_or(0)
            ));
            log::detail(&format!(
                "cache economy: saved ~{}  spent {} (rebuild {})  net {:+.0}ms{}",
                format_ms(cache.estimated_saved_ms.round() as u128),
                format_ms(cache.spent_ms.round() as u128),
                format_ms(cache.rebuild_ms),
                cache.net_ms,
                match cache.stood_down_at_experiment {
                    Some(n) => format!("  STOOD DOWN at experiment {n}"),
                    None => String::new(),
                }
            ));
        }
        if let Some(grafts) = &report.graft_replay {
            log::detail(&format!(
                "graft replay:  {} phase(s)  accepts {}  grafts {}  Δ {:+.3e}  (scorer_fail {}  errors {})",
                grafts.replays,
                grafts.accepts,
                grafts.grafts_applied,
                grafts.cumulative_improvement,
                grafts.scorer_failures,
                grafts.replay_errors
            ));
        }
        if !report.focus_history.is_empty() {
            let focuses = report
                .focus_history
                .iter()
                .map(|h| {
                    format!(
                        "{}×{} (accepts={} Δ={:+.3e})",
                        h.focus_neuron, h.experiments, h.accepts, h.cumulative_improvement
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            log::detail(&format!("focus history: {focuses}"));
        }
        if let Some(focus) = &report.focus_stats {
            log::detail(&format!("focus stats:   {}", format_focus(&focus.all)));
            if focus.accepted.experiments > 0 {
                log::detail(&format!("  accepted:    {}", format_focus(&focus.accepted)));
                log::detail(&format!("  rejected:    {}", format_focus(&focus.rejected)));
            }
        }
        if !report.strategies.is_empty() {
            let wins = report
                .strategies
                .iter()
                .filter(|s| s.wins > 0 || s.appearances_total > 0)
                .map(|s| {
                    let combo = if s.combo_wins > 0 {
                        format!(" [{} in combo]", s.combo_wins)
                    } else {
                        String::new()
                    };
                    format!(
                        "{}×{}{combo}/{} ({:.0}%)",
                        s.strategy,
                        s.wins,
                        s.appearances_total,
                        s.acceptance_rate * 100.0
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            log::detail(&format!("strategies:    {wins}"));
        }
    } else {
        log::detail(&format!("best score:    {}", result.best_score));
    }

    log::detail(&format!("best.json:     {}", result.best_path.display()));
    log::detail(&format!("journal:       {}", result.journal_path.display()));
    if result.scorer_failures > 0 {
        log::warn(&format!(
            "scorer failures during run: {}",
            result.scorer_failures
        ));
    }
    if result.acceptances > 0 {
        log::ok(&format!(
            "finished with {} acceptance(s)",
            result.acceptances
        ));
    } else {
        log::warn("finished with no acceptances");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::CandidateProvenance;
    use crate::run::ExperimentRecord;
    use crate::scorer_cost::{ScorerCallPhase, ScorerCallRecord};
    use std::collections::BTreeMap;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn prov(strategy: CandidateStrategy) -> CandidateProvenance {
        CandidateProvenance {
            strategy,
            focus_neuron: "h1".into(),
            mutation: "x".into(),
            old_value: Some(0.0),
            new_value: Some(0.1),
            mirror: None,
        }
    }

    /// Minimal focus statistics for the aggregate tests (issue #70).
    fn focus_stats(
        squash: &str,
        incoming: usize,
        saturation: f64,
        blame: Option<f64>,
    ) -> crate::focus::FocusNeuronStats {
        crate::focus::FocusNeuronStats {
            neuron_uuid: "h1".into(),
            squash: Some(squash.into()),
            incoming_count: incoming,
            post_variance: 0.25,
            near_zero_fraction: 0.5 - saturation / 2.0,
            saturation_fraction: saturation,
            mean_blame: blame,
            mean_abs_blame: blame.map(f64::abs),
            record_count: 10,
            ..Default::default()
        }
    }

    fn experiment(number: u64, accepted: bool) -> ExperimentRecord {
        ExperimentRecord {
            experiment_number: number,
            timestamp_unix: 1000 + number,
            seed: Some(1),
            incumbent_id: "x".into(),
            baseline_score: 0.4,
            focus_neuron: "h1".into(),
            focus_neurons: None,
            focus_stats: None,
            candidates: vec![prov(CandidateStrategy::Random)],
            candidates_requested: None,
            batch_limit: None,
            scores: BTreeMap::new(),
            mirror_axis_failures: None,
            screen_scores: None,
            screen_tiers: None,
            baseline_source: None,
            winner: accepted.then(|| "candidate-000".to_string()),
            improvement: accepted.then_some(1e-6),
            accepted,
            analysis_ms: 1,
            memo_hits: 0,
            memo_misses: 0,
            memo_ms_saved: 0,
            scorer_ms: 2,
            scorer_calls: None,
            scorer_error: None,
            combo_members: None,
            combo_member_indices: accepted.then(|| vec![0]),
            combos_scored: None,
            combos_dampened: None,
            combo_dampen: None,
            cache_skipped: None,
            cache_deduplicated: None,
            cache_backfilled: None,
            cache_size: None,
            cache_lookup_ms: None,
            cache_maintenance_ms: None,
            cache_saved_ms: None,
            cache_spent_ms: None,
            cache_net_cumulative_ms: None,
            cache_resident_bytes: None,
            cache_rebuild_ms: None,
        }
    }

    fn journal_of(records: &[ExperimentRecord]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for record in records {
            writeln!(file, "{}", serde_json::to_string(record).unwrap()).unwrap();
        }
        file
    }

    /// One journalled scorer call (issue #112).
    fn call(phase: ScorerCallPhase, creatures: u64, elapsed_ms: u128) -> ScorerCallRecord {
        ScorerCallRecord {
            phase,
            creatures,
            sample_rate: matches!(phase, ScorerCallPhase::Screen).then_some(0.05),
            elapsed_ms,
            failed: false,
        }
    }

    /// Issue #112: the per-call creature counts a run journals recover the fixed
    /// and marginal scorer cost exactly, and screen and promote are fitted apart
    /// — pooling them would fit one line through two different populations.
    #[test]
    fn report_recovers_the_fixed_and_marginal_scorer_cost_per_phase() {
        // Screen: 800 ms fixed + 40 ms/creature. Promote: 9000 + 11 000 ms.
        let mut first = experiment(1, false);
        first.scorer_calls = Some(vec![
            call(ScorerCallPhase::Screen, 1, 840),
            call(ScorerCallPhase::Promote, 2, 31_000),
        ]);
        let mut second = experiment(2, false);
        second.scorer_calls = Some(vec![
            call(ScorerCallPhase::Screen, 30, 2000),
            call(ScorerCallPhase::Promote, 4, 53_000),
        ]);
        let file = journal_of(&[first, second]);

        let report = report_from_journal(file.path()).unwrap();
        let cost = &report.scorer_call_cost;
        assert_eq!(cost.calls, 4);
        assert_eq!(cost.failed_calls, 0);
        assert_eq!(cost.creatures_scored, 37);

        let screen = &cost.by_phase["screen"];
        assert_eq!(screen.calls, 2);
        assert!((screen.fixed_ms.expect("screen fixed") - 800.0).abs() < 1e-6);
        assert!((screen.marginal_ms_per_creature.expect("screen slope") - 40.0).abs() < 1e-9);

        let promote = &cost.by_phase["promote"];
        assert_eq!(promote.calls, 2);
        assert!((promote.fixed_ms.expect("promote fixed") - 9000.0).abs() < 1e-6);
        assert!((promote.marginal_ms_per_creature.expect("promote slope") - 11_000.0).abs() < 1e-9);
    }

    /// Issue #112: Phase-0 and Phase-G calls live on their own journal lines,
    /// and both belong in the cost model — an intercept fitted to the experiment
    /// loop alone is fitted to a subset of the run.
    #[test]
    fn report_folds_phase0_and_graft_replay_calls_into_the_cost_model() {
        let mut file = NamedTempFile::new().unwrap();
        let phase0 = crate::run::ScorerCallsRecord {
            record: crate::run::ScorerCallsKind::ScorerCalls,
            timestamp_unix: 900,
            stage: "phase0".into(),
            calls: vec![call(ScorerCallPhase::Phase0, 1, 10_000)],
        };
        writeln!(file, "{}", serde_json::to_string(&phase0).unwrap()).unwrap();
        let replay = crate::run::GraftReplayRecord {
            record: crate::run::GraftReplayKind::GraftReplay,
            timestamp_unix: 950,
            grafts_applied: 0,
            accepted: false,
            baseline_score: Some(0.4),
            score: None,
            improvement: None,
            elapsed_ms: 5,
            scorer_successes: 2,
            scorer_failures: 0,
            scorer_calls: Some(vec![
                call(ScorerCallPhase::GraftReplay, 2, 12_000),
                call(ScorerCallPhase::GraftReplay, 5, 15_000),
            ]),
            replay_error: None,
        };
        writeln!(file, "{}", serde_json::to_string(&replay).unwrap()).unwrap();
        let mut experiment = experiment(1, false);
        experiment.scorer_calls = Some(vec![call(ScorerCallPhase::Screen, 30, 2000)]);
        writeln!(file, "{}", serde_json::to_string(&experiment).unwrap()).unwrap();

        let report = report_from_journal(file.path()).unwrap();
        let cost = &report.scorer_call_cost;
        assert_eq!(cost.calls, 4, "every call in the journal is counted");
        // One Phase-0 call is one batch size: reported, but not decomposed.
        let phase0_fit = &cost.by_phase["phase0"];
        assert_eq!(phase0_fit.calls, 1);
        assert_eq!(phase0_fit.fixed_ms, None);
        assert!((phase0_fit.mean_ms - 10_000.0).abs() < 1e-9);
        let graft_fit = &cost.by_phase["graftReplay"];
        assert!((graft_fit.fixed_ms.expect("graft fixed") - 10_000.0).abs() < 1e-6);
        assert!((graft_fit.marginal_ms_per_creature.expect("graft slope") - 1000.0).abs() < 1e-9);
    }

    /// Issue #112: a journal written before per-call records existed reports no
    /// cost model rather than a fabricated one.
    #[test]
    fn report_reports_no_call_cost_for_a_pre_change_journal() {
        let file = journal_of(&[experiment(1, false), experiment(2, true)]);
        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.scorer_call_cost, ScorerCallCost::default());
        assert!(report.scorer_call_cost.by_phase.is_empty());
    }

    /// Provenance naming a specific focus (issue #109).
    fn prov_for(focus: &str) -> CandidateProvenance {
        CandidateProvenance {
            focus_neuron: focus.into(),
            ..prov(CandidateStrategy::Random)
        }
    }

    /// Issue #109: `focusHistory` counts every focus an experiment served, and
    /// credits the accept to the focus the winner was proposed against.
    #[test]
    fn report_renders_focus_history_for_a_multi_focus_journal() {
        let mut record = experiment(1, true);
        record.focus_neuron = "a".into();
        record.focus_neurons = Some(vec!["a".into(), "b".into(), "c".into()]);
        record.candidates = vec![prov_for("a"), prov_for("b"), prov_for("c")];
        // The winner is candidate 1 — focus `b`, not the primary.
        record.winner = Some("candidate-001".into());
        record.combo_member_indices = Some(vec![1]);
        record.improvement = Some(2e-6);
        let file = journal_of(&[record]);

        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.focus_counts.get("a"), Some(&1));
        assert_eq!(report.focus_counts.get("b"), Some(&1));
        assert_eq!(report.focus_counts.get("c"), Some(&1));
        let credited: Vec<&str> = report
            .focus_history
            .iter()
            .filter(|h| h.accepts > 0)
            .map(|h| h.focus_neuron.as_str())
            .collect();
        assert_eq!(credited, vec!["b"], "only the winner's focus is credited");
        let b = report
            .focus_history
            .iter()
            .find(|h| h.focus_neuron == "b")
            .expect("focus b in history");
        assert!((b.cumulative_improvement - 2e-6).abs() < 1e-15);
    }

    /// Issue #109: a single-focus journal reports exactly what it always did.
    #[test]
    fn report_renders_focus_history_for_a_single_focus_journal() {
        let file = journal_of(&[experiment(1, true), experiment(2, false)]);
        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.focus_counts, [("h1".to_string(), 2)].into());
        assert_eq!(report.focus_history.len(), 1);
        assert_eq!(report.focus_history[0].focus_neuron, "h1");
        assert_eq!(report.focus_history[0].accepts, 1);
    }

    /// Issue #109: a journal written before the focus set existed — and before
    /// `comboMemberIndices` did — still attributes to the one focus it names.
    #[test]
    fn report_reads_a_pre_change_journal_with_no_focus_set() {
        let mut record = experiment(1, true);
        record.focus_neurons = None;
        // Pre-#74: a merged combo stem with no member indices.
        record.winner = Some("combo-001-k2".into());
        record.combo_member_indices = None;
        let file = journal_of(&[record]);

        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.focus_counts, [("h1".to_string(), 1)].into());
        assert_eq!(report.focus_history[0].focus_neuron, "h1");
        assert_eq!(
            report.focus_history[0].accepts, 1,
            "an unattributable winner still credits the experiment's focus"
        );
    }

    /// Issue #108: the achieved batch size per experiment is reportable, and
    /// an under-filled batch says which limit bound it.
    #[test]
    fn report_summarises_the_achieved_candidate_batch_size() {
        let mut small = experiment(1, false);
        small.candidates = vec![prov(CandidateStrategy::Random); 29];
        small.candidates_requested = Some(100);
        small.batch_limit = Some(BatchLimit::QuotaCeiling);
        let mut large = experiment(2, false);
        large.candidates = vec![prov(CandidateStrategy::Random); 61];
        large.candidates_requested = Some(100);
        large.batch_limit = Some(BatchLimit::Exhausted);
        let file = journal_of(&[small, large]);

        let batch = report_from_journal(file.path()).unwrap().candidate_batch;
        assert!((batch.mean_generated - 45.0).abs() < 1e-12);
        assert_eq!(batch.min_generated, 29);
        assert_eq!(batch.max_generated, 61);
        assert_eq!(batch.requested, Some(100));
        assert_eq!(batch.quota_ceiling_experiments, 1);
        assert_eq!(batch.exhausted_experiments, 1);
    }

    /// A journal written before the generator recorded its budget still reports
    /// the achieved sizes — only the budget behind them is unknown.
    #[test]
    fn a_journal_without_batch_fields_still_reports_the_sizes() {
        let file = journal_of(&[experiment(1, false), experiment(2, false)]);
        let batch = report_from_journal(file.path()).unwrap().candidate_batch;
        assert!((batch.mean_generated - 1.0).abs() < 1e-12);
        assert_eq!(batch.requested, None);
        assert_eq!(batch.quota_ceiling_experiments, 0);
        assert_eq!(batch.exhausted_experiments, 0);
    }

    /// Issue #70: focus structure/statistics/blame aggregate, split by outcome.
    #[test]
    fn report_aggregates_focus_stats_by_outcome() {
        let mut accepted = experiment(1, true);
        accepted.focus_stats = Some(focus_stats("TANH", 4, 0.8, Some(-2e-3)));
        let mut rejected = experiment(2, false);
        rejected.focus_stats = Some(focus_stats("IDENTITY", 2, 0.2, Some(1e-4)));
        let file = journal_of(&[accepted, rejected]);

        let summary = report_from_journal(file.path())
            .unwrap()
            .focus_stats
            .expect("journal carries focus statistics");

        assert_eq!(summary.all.experiments, 2);
        assert!((summary.all.mean_incoming_count - 3.0).abs() < 1e-12);
        assert!((summary.all.mean_saturation_fraction - 0.5).abs() < 1e-12);
        assert!((summary.all.mean_near_zero_fraction - 0.25).abs() < 1e-12);
        assert!((summary.all.mean_post_variance - 0.25).abs() < 1e-12);
        assert_eq!(summary.all.squash_counts.get("TANH"), Some(&1));
        assert_eq!(summary.all.squash_counts.get("IDENTITY"), Some(&1));

        assert_eq!(summary.accepted.experiments, 1);
        assert!((summary.accepted.mean_saturation_fraction - 0.8).abs() < 1e-12);
        assert!((summary.accepted.mean_incoming_count - 4.0).abs() < 1e-12);
        // Blame is aggregated as a magnitude so signs cannot cancel out.
        assert!((summary.accepted.mean_abs_blame.unwrap() - 2e-3).abs() < 1e-12);

        assert_eq!(summary.rejected.experiments, 1);
        assert!((summary.rejected.mean_saturation_fraction - 0.2).abs() < 1e-12);
        assert!((summary.rejected.mean_abs_blame.unwrap() - 1e-4).abs() < 1e-12);
    }

    /// A focus without a learning signal must not fabricate a blame average.
    #[test]
    fn report_reports_no_blame_when_none_was_recorded() {
        let mut record = experiment(1, false);
        record.focus_stats = Some(focus_stats("LOGISTIC", 1, 0.0, None));
        let file = journal_of(&[record]);

        let summary = report_from_journal(file.path())
            .unwrap()
            .focus_stats
            .unwrap();
        assert_eq!(summary.all.experiments, 1);
        assert_eq!(summary.all.mean_abs_blame, None);
        assert_eq!(summary.accepted, FocusStatsAggregate::default());
    }

    /// Journals written before issue #70 have no focus statistics to summarise.
    #[test]
    fn report_omits_focus_stats_for_a_legacy_journal() {
        let mut file = NamedTempFile::new().unwrap();
        let legacy = serde_json::to_value(experiment(1, false)).unwrap();
        assert!(
            legacy.get("focusStats").is_none(),
            "an absent focus scan must not be serialised"
        );
        writeln!(file, "{legacy}").unwrap();

        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.experiments, 1);
        assert_eq!(report.focus_stats, None);
    }

    /// Issue #71: the run-header line is metadata, not an experiment.
    #[test]
    fn report_skips_the_run_header_line() {
        use crate::run::{RunConfigRecord, RunHeaderRecord, SeedSource};

        let mut file = NamedTempFile::new().unwrap();
        let header = RunHeaderRecord::new(
            42,
            SeedSource::Drawn,
            RunConfigRecord::from_config(
                &crate::config::LamarckConfig::default(),
                crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON,
            ),
            1000,
        );
        let experiment = ExperimentRecord {
            experiment_number: 1,
            timestamp_unix: 1000,
            seed: Some(42),
            incumbent_id: "x".into(),
            baseline_score: 0.4,
            focus_neuron: "h1".into(),
            focus_neurons: None,
            focus_stats: None,
            candidates: vec![prov(CandidateStrategy::Random)],
            candidates_requested: None,
            batch_limit: None,
            scores: BTreeMap::new(),
            mirror_axis_failures: None,
            screen_scores: None,
            screen_tiers: None,
            baseline_source: None,
            winner: None,
            improvement: None,
            accepted: false,
            analysis_ms: 1,
            memo_hits: 0,
            memo_misses: 0,
            memo_ms_saved: 0,
            scorer_ms: 2,
            scorer_calls: None,
            scorer_error: None,
            combo_members: None,
            combo_member_indices: None,
            combos_scored: None,
            combos_dampened: None,
            combo_dampen: None,
            cache_skipped: None,
            cache_deduplicated: None,
            cache_backfilled: None,
            cache_size: None,
            cache_lookup_ms: None,
            cache_maintenance_ms: None,
            cache_saved_ms: None,
            cache_spent_ms: None,
            cache_net_cumulative_ms: None,
            cache_resident_bytes: None,
            cache_rebuild_ms: None,
        };
        writeln!(file, "{}", serde_json::to_string(&header).unwrap()).unwrap();
        writeln!(file, "{}", serde_json::to_string(&experiment).unwrap()).unwrap();
        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.experiments, 1, "header must not count as experiment");
    }

    /// Issue #74: a merged combo win belongs to every member strategy.
    #[test]
    fn report_attributes_a_combo_win_to_every_member_strategy() {
        let mut record = experiment(1, true);
        record.candidates = vec![
            prov(CandidateStrategy::Random),
            prov(CandidateStrategy::Backprop),
            prov(CandidateStrategy::StatsBias),
        ];
        record.winner = Some("combo-000-k2".into());
        record.combo_members = Some(2);
        record.combo_member_indices = Some(vec![0, 1]);
        record.combos_scored = Some(3);
        let file = journal_of(&[record]);

        let report = report_from_journal(file.path()).unwrap();
        let row = |name: &str| {
            report
                .strategies
                .iter()
                .find(|s| s.strategy == name)
                .unwrap_or_else(|| panic!("{name} row"))
                .clone()
        };
        assert_eq!(report.combo_acceptances, 1);
        assert_eq!(report.combo_acceptances_unattributed, 0);
        assert_eq!(row("random").wins, 1, "member strategy wins with the combo");
        assert_eq!(row("random").combo_wins, 1);
        assert_eq!(row("backprop").wins, 1);
        assert_eq!(row("backprop").combo_wins, 1);
        assert_eq!(
            row("stats_bias").wins,
            0,
            "a non-member must not inherit the win"
        );
        assert!((row("random").acceptance_rate - 1.0).abs() < 1e-12);
    }

    /// A single-candidate win is attributed to exactly one strategy, once.
    #[test]
    fn report_attributes_a_single_win_to_one_strategy() {
        let mut record = experiment(1, true);
        record.candidates = vec![
            prov(CandidateStrategy::Random),
            prov(CandidateStrategy::Backprop),
        ];
        record.winner = Some("candidate-001".into());
        record.combo_member_indices = Some(vec![1]);
        let file = journal_of(&[record]);

        let report = report_from_journal(file.path()).unwrap();
        let backprop = report
            .strategies
            .iter()
            .find(|s| s.strategy == "backprop")
            .unwrap();
        assert_eq!(backprop.wins, 1);
        assert_eq!(backprop.combo_wins, 0, "a single is not a combo win");
        assert_eq!(
            report
                .strategies
                .iter()
                .find(|s| s.strategy == "random")
                .unwrap()
                .wins,
            0
        );
    }

    /// A pre-#74 combo win names no members, so the gap is reported, not hidden.
    #[test]
    fn report_counts_an_unattributable_legacy_combo_win() {
        let mut record = experiment(1, true);
        record.candidates = vec![
            prov(CandidateStrategy::Random),
            prov(CandidateStrategy::Backprop),
        ];
        record.winner = Some("combo-000-k2".into());
        record.combo_members = Some(2);
        record.combo_member_indices = None;
        let file = journal_of(&[record]);

        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.acceptances, 1);
        assert_eq!(report.combo_acceptances, 1);
        assert_eq!(report.combo_acceptances_unattributed, 1);
        assert!(
            report.strategies.iter().all(|s| s.wins == 0),
            "unknown members must not be guessed at"
        );
    }

    /// Issue #74: Phase-G accepts get their own bucket instead of being dropped.
    #[test]
    fn report_buckets_graft_replay_accepts() {
        use crate::run::{GraftReplayKind, GraftReplayRecord};

        let replay = GraftReplayRecord {
            record: GraftReplayKind::GraftReplay,
            timestamp_unix: 999,
            grafts_applied: 2,
            accepted: true,
            baseline_score: Some(0.40),
            score: Some(0.4000030),
            improvement: Some(3e-6),
            elapsed_ms: 120,
            scorer_successes: 2,
            scorer_failures: 1,
            scorer_calls: None,
            replay_error: None,
        };
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{}", serde_json::to_string(&replay).unwrap()).unwrap();
        writeln!(
            file,
            "{}",
            serde_json::to_string(&experiment(1, false)).unwrap()
        )
        .unwrap();

        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.experiments, 1, "a replay line is not an experiment");
        let grafts = report.graft_replay.expect("graft replay bucket");
        assert_eq!(grafts.replays, 1);
        assert_eq!(grafts.accepts, 1);
        assert_eq!(grafts.grafts_applied, 2);
        assert!((grafts.cumulative_improvement - 3e-6).abs() < 1e-15);
        assert_eq!(grafts.scorer_failures, 1);
        assert_eq!(grafts.replay_errors, 0);
    }

    /// A failed replay is surfaced in the bucket rather than passing as clean.
    #[test]
    fn report_surfaces_a_failed_graft_replay() {
        use crate::run::{GraftReplayKind, GraftReplayRecord};

        let replay = GraftReplayRecord {
            record: GraftReplayKind::GraftReplay,
            timestamp_unix: 999,
            grafts_applied: 0,
            accepted: false,
            baseline_score: Some(0.4),
            score: None,
            improvement: None,
            elapsed_ms: 5,
            scorer_successes: 0,
            scorer_failures: 0,
            scorer_calls: None,
            replay_error: Some("graft singles: baseline missing".into()),
        };
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{}", serde_json::to_string(&replay).unwrap()).unwrap();

        let grafts = report_from_journal(file.path())
            .unwrap()
            .graft_replay
            .expect("graft replay bucket");
        assert_eq!(grafts.replays, 1);
        assert_eq!(grafts.accepts, 0);
        assert_eq!(grafts.replay_errors, 1);
        assert_eq!(grafts.cumulative_improvement, 0.0);
    }

    /// A journal with no Phase-G line reports no bucket at all.
    #[test]
    fn report_omits_the_graft_bucket_without_a_replay_line() {
        let file = journal_of(&[experiment(1, false)]);
        assert_eq!(report_from_journal(file.path()).unwrap().graft_replay, None);
    }

    /// A screen-empty experiment: subsample baseline only, no full-corpus score.
    fn screened_empty(number: u64, screen_baseline: f64) -> ExperimentRecord {
        let mut record = experiment(number, false);
        record.scores = BTreeMap::new();
        record.screen_scores = Some({
            let mut m = BTreeMap::new();
            m.insert("baseline".into(), screen_baseline);
            m.insert("candidate-000".into(), screen_baseline - 1e-3);
            m
        });
        record
    }

    /// An experiment that promoted to full-corpus scoring.
    fn promoted(number: u64, baseline: f64, improvement: Option<f64>) -> ExperimentRecord {
        let mut record = experiment(number, improvement.is_some());
        record.baseline_score = baseline;
        record.improvement = improvement;
        record.scores = {
            let mut m = BTreeMap::new();
            m.insert("baseline".into(), baseline);
            m.insert(
                "candidate-000".into(),
                baseline + improvement.unwrap_or(-1e-6),
            );
            m
        };
        record
    }

    /// Issue #113: the report counts which baseline decided each promote call
    /// and never over-claims the saving — the verification call is subtracted.
    #[test]
    fn report_counts_remembered_promote_calls_and_nets_off_verification() {
        let mut fresh = promoted(1, 0.3470, None);
        fresh.baseline_source = Some(BaselineSource::Fresh);
        let mut remembered = promoted(2, 0.3470, None);
        remembered.baseline_source = Some(BaselineSource::Remembered);
        let mut verified = promoted(3, 0.3470, Some(2e-6));
        verified.baseline_source = Some(BaselineSource::RememberedVerified);
        // An experiment whose screen was empty made no promote call at all.
        let empty = screened_empty(4, 0.3475);
        let file = journal_of(&[fresh, remembered, verified, empty]);

        let reuse = report_from_journal(file.path()).unwrap().baseline_reuse;
        assert_eq!(reuse.fresh_promote_calls, 1);
        assert_eq!(reuse.remembered_promote_calls, 2);
        assert_eq!(reuse.verified_accepts, 1);
        assert_eq!(reuse.baseline_scores_saved, 2);
        assert_eq!(reuse.verification_creature_scores, 2);
        assert_eq!(reuse.net_creature_scores_saved, 0);
        assert!((reuse.remembered_fraction - 2.0 / 3.0).abs() < 1e-12);
    }

    /// A journal written before the field existed reports every promote call as
    /// what it was: paired with a freshly scored baseline.
    #[test]
    fn a_pre_113_journal_reports_no_remembered_baselines() {
        let file = journal_of(&[promoted(1, 0.3470, None), promoted(2, 0.3470, None)]);
        let reuse = report_from_journal(file.path()).unwrap().baseline_reuse;
        assert_eq!(reuse.remembered_promote_calls, 0);
        assert_eq!(reuse.net_creature_scores_saved, 0);
        assert_eq!(reuse.remembered_fraction, 0.0);
    }

    /// Issue #84: the opening anchor must never be a 5% screen-sample baseline.
    ///
    /// With `--skip-phase0` the first experiments can all screen empty, and each
    /// records the subsample baseline. Anchoring there subtracts two different
    /// quantities and reports a negative total for a run that only improved.
    #[test]
    fn report_anchors_the_opening_baseline_on_a_full_corpus_score() {
        let file = journal_of(&[
            screened_empty(1, 0.3475),
            promoted(2, 0.3470, None),
            promoted(3, 0.3470, Some(1.322e-6)),
            promoted(4, 0.3470 + 1.322e-6, Some(1.724e-6)),
        ]);

        let report = report_from_journal(file.path()).unwrap();
        let opening = report.opening_baseline_score.expect("full-corpus anchor");
        assert!(
            (opening - 0.3470).abs() < 1e-12,
            "anchored on {opening}, expected the first promoted full-corpus baseline"
        );
        let total = report.total_score_improvement.expect("total improvement");
        assert!(
            (total - 3.046e-6).abs() < 1e-12,
            "total {total} must be the accepted gain, not a sampling artefact"
        );
        assert!(
            total > 0.0,
            "two accepted improvements cannot total negative"
        );
        assert!(report.relative_score_improvement.unwrap() > 0.0);
    }

    /// No full-corpus score anywhere: report `null` rather than a sampled score.
    #[test]
    fn report_leaves_the_opening_baseline_null_without_a_full_corpus_score() {
        let file = journal_of(&[screened_empty(1, 0.3475), screened_empty(2, 0.3480)]);

        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.experiments, 2);
        assert_eq!(
            report.opening_baseline_score, None,
            "a screen-sample baseline is not an opening baseline"
        );
        assert_eq!(report.total_score_improvement, None);
        assert_eq!(report.relative_score_improvement, None);
    }

    /// A scorer failure records the incumbent score but no full-corpus batch.
    #[test]
    fn report_skips_a_scorer_failure_when_anchoring_the_opening_baseline() {
        let mut failed = experiment(1, false);
        failed.scorer_error = Some("scorer exited 1".into());
        let file = journal_of(&[failed, promoted(2, 0.3470, Some(2e-6))]);

        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.scorer_failures, 1);
        assert!((report.opening_baseline_score.unwrap() - 0.3470).abs() < 1e-12);
        assert!((report.total_score_improvement.unwrap() - 2e-6).abs() < 1e-12);
    }

    #[test]
    fn report_counts_acceptances_and_strategy_rates() {
        let mut file = NamedTempFile::new().unwrap();
        let accepted = ExperimentRecord {
            experiment_number: 1,
            timestamp_unix: 1000,
            seed: Some(1),
            incumbent_id: "x".into(),
            baseline_score: 0.4,
            focus_neuron: "h1".into(),
            focus_neurons: None,
            focus_stats: None,
            candidates: vec![
                prov(CandidateStrategy::Random),
                prov(CandidateStrategy::Backprop),
            ],
            candidates_requested: None,
            batch_limit: None,
            scores: {
                let mut m = BTreeMap::new();
                m.insert("baseline".into(), 0.4);
                m.insert("candidate-000".into(), 0.400002);
                m
            },
            mirror_axis_failures: None,
            screen_scores: Some({
                let mut m = BTreeMap::new();
                m.insert("baseline".into(), 0.4);
                m.insert("candidate-000".into(), 0.41);
                m.insert("candidate-001".into(), 0.39);
                m
            }),
            screen_tiers: None,
            baseline_source: None,
            winner: Some("candidate-000".into()),
            improvement: Some(2e-6),
            accepted: true,
            analysis_ms: 10,
            memo_hits: 2,
            memo_misses: 0,
            memo_ms_saved: 40,
            scorer_ms: 20,
            scorer_calls: None,
            scorer_error: None,
            combo_members: None,
            combo_member_indices: None,
            combos_scored: None,
            combos_dampened: None,
            combo_dampen: None,
            cache_skipped: None,
            cache_deduplicated: None,
            cache_backfilled: None,
            cache_size: None,
            cache_lookup_ms: None,
            cache_maintenance_ms: None,
            cache_saved_ms: None,
            cache_spent_ms: None,
            cache_net_cumulative_ms: None,
            cache_resident_bytes: None,
            cache_rebuild_ms: None,
        };
        let rejected = ExperimentRecord {
            experiment_number: 2,
            timestamp_unix: 1060,
            seed: Some(1),
            incumbent_id: "x".into(),
            baseline_score: 0.400002,
            focus_neuron: "h1".into(),
            focus_neurons: None,
            focus_stats: None,
            candidates: vec![prov(CandidateStrategy::Random)],
            candidates_requested: None,
            batch_limit: None,
            scores: {
                let mut m = BTreeMap::new();
                m.insert("baseline".into(), 0.400002);
                m.insert("candidate-000".into(), 0.400001);
                m
            },
            mirror_axis_failures: None,
            screen_scores: None,
            screen_tiers: None,
            baseline_source: None,
            winner: None,
            improvement: None,
            accepted: false,
            analysis_ms: 5,
            memo_hits: 0,
            memo_misses: 2,
            memo_ms_saved: 0,
            scorer_ms: 15,
            scorer_calls: None,
            scorer_error: None,
            combo_members: None,
            combo_member_indices: None,
            combos_scored: None,
            combos_dampened: None,
            combo_dampen: None,
            cache_skipped: None,
            cache_deduplicated: None,
            cache_backfilled: None,
            cache_size: None,
            cache_lookup_ms: None,
            cache_maintenance_ms: None,
            cache_saved_ms: None,
            cache_spent_ms: None,
            cache_net_cumulative_ms: None,
            cache_resident_bytes: None,
            cache_rebuild_ms: None,
        };
        writeln!(file, "{}", serde_json::to_string(&accepted).unwrap()).unwrap();
        writeln!(file, "{}", serde_json::to_string(&rejected).unwrap()).unwrap();
        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.experiments, 2);
        assert_eq!(report.acceptances, 1);
        assert_eq!(report.screen_candidates_scored, 2);
        assert_eq!(report.candidates_scored, 2); // 1 + 1 promote candidates
        assert!(report.projected_batches_per_45_min > 0.0);
        assert_eq!(report.improvement_series.len(), 2);
        assert_eq!(report.focus_history.len(), 1);
        assert_eq!(report.focus_history[0].accepts, 1);
        let random = report
            .strategies
            .iter()
            .find(|s| s.strategy == "random")
            .unwrap();
        assert_eq!(random.appearances_total, 2);
        assert_eq!(random.wins, 1);
        assert!((random.acceptance_rate - 0.5).abs() < 1e-12);
        let backprop = report
            .strategies
            .iter()
            .find(|s| s.strategy == "backprop")
            .unwrap();
        assert_eq!(backprop.wins, 0);
        assert_eq!(backprop.appearances_total, 1);
        assert_eq!(backprop.appearances_in_accepted, 1);
    }

    /// Issue #110: `report` carries the screen-versus-full-corpus calibration,
    /// paired from the two score maps the journal already holds.
    #[test]
    fn report_calibrates_the_screen_against_the_full_corpus() {
        let mut promoted = experiment(1, false);
        promoted.screen_scores = Some({
            let mut m = BTreeMap::new();
            m.insert("baseline".into(), 0.40);
            m.insert("candidate-000".into(), 0.40 + 1e-5);
            m.insert("candidate-001".into(), 0.40 + 2e-5);
            // Screened, never promoted: no full score to pair with.
            m.insert("candidate-002".into(), 0.40 - 1e-5);
            m
        });
        promoted.scores = {
            let mut m = BTreeMap::new();
            m.insert("baseline".into(), 0.50);
            m.insert("candidate-000".into(), 0.50 + 2e-6);
            m.insert("candidate-001".into(), 0.50 - 3e-6);
            m
        };
        let file = journal_of(&[promoted, screened_empty(2, 0.3475)]);

        let calibration = report_from_journal(file.path()).unwrap().screen_calibration;
        assert!(calibration.screen_enabled);
        assert_eq!(calibration.experiments_screened, 2);
        assert_eq!(calibration.paired_candidates, 2);
        // candidate-002 plus the screen-empty experiment's own candidate.
        assert_eq!(calibration.screen_only_candidates, 2);
        assert_eq!(calibration.full_only_candidates, 0);
        assert_eq!(calibration.promoted_improved, 1);
        assert_eq!(calibration.promoted_worse, 1);
        assert!((calibration.promotion_precision.unwrap() - 0.5).abs() < 1e-12);
        // Two pairs is below the three the coefficient needs.
        assert_eq!(calibration.spearman, None);
        assert_eq!(calibration.pairs.len(), 2);
        assert!((calibration.pairs[0].screen_delta - 1e-5).abs() < 1e-15);
        assert!((calibration.pairs[0].full_delta - 2e-6).abs() < 1e-15);
    }

    /// A journal with screening disabled reports "not applicable", not a
    /// fabricated correlation — and `report` still produces every other field.
    #[test]
    fn report_reads_a_journal_with_no_screen_phase() {
        let file = journal_of(&[promoted(1, 0.3470, Some(2e-6))]);

        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.experiments, 1);
        let calibration = report.screen_calibration;
        assert!(!calibration.screen_enabled);
        assert_eq!(calibration.experiments_without_screen, 1);
        assert_eq!(calibration.spearman, None);
        assert_eq!(calibration.promotion_precision, None);
        assert!(calibration.pairs.is_empty());
    }

    /// Issue #110: the run header's knobs are quoted beside the calibration, so
    /// a threshold recommendation names the gate it was measured under.
    #[test]
    fn report_quotes_the_screen_knobs_from_the_run_header() {
        use crate::run::{RunConfigRecord, RunHeaderRecord, SeedSource};

        let mut file = NamedTempFile::new().unwrap();
        let header = RunHeaderRecord::new(
            42,
            SeedSource::Drawn,
            RunConfigRecord::from_config(
                &crate::config::LamarckConfig::default(),
                crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON,
            ),
            1000,
        );
        writeln!(file, "{}", serde_json::to_string(&header).unwrap()).unwrap();
        writeln!(
            file,
            "{}",
            serde_json::to_string(&experiment(1, false)).unwrap()
        )
        .unwrap();

        let calibration = report_from_journal(file.path()).unwrap().screen_calibration;
        assert_eq!(
            calibration.screen_sample_rate,
            Some(crate::config::DEFAULT_SCREEN_SAMPLE_RATE)
        );
        assert_eq!(
            calibration.promote_threshold,
            Some(crate::config::DEFAULT_SCREEN_PROMOTE_THRESHOLD)
        );
        assert_eq!(
            calibration.accept_bar,
            Some(crate::config::DEFAULT_MIN_IMPROVEMENT)
        );
    }

    /// Issue #106: the memo's value is auditable from the journal.
    #[test]
    fn report_totals_the_analysis_memo_economics() {
        // 2 hits saving 40ms in experiment one, 2 misses in experiment two.
        let mut hitting = experiment(1, false);
        hitting.analysis_ms = 60;
        hitting.memo_hits = 2;
        hitting.memo_ms_saved = 40;
        let mut missing = experiment(2, false);
        missing.analysis_ms = 100;
        missing.memo_misses = 2;
        let file = journal_of(&[hitting, missing]);

        let memo = report_from_journal(file.path()).unwrap().analysis_memo;
        assert_eq!(memo.hits, 2);
        assert_eq!(memo.misses, 2);
        assert_eq!(memo.ms_saved, 40);
        assert!((memo.hit_rate - 0.5).abs() < 1e-12);
        // 40 saved against 160 measured + 40 saved.
        assert!((memo.analysis_ms_saved_fraction - 0.2).abs() < 1e-12);
    }

    /// A journal written before the memo existed must report zeros, not fail.
    #[test]
    fn report_reads_a_pre_memo_journal_as_zero_savings() {
        let mut line = serde_json::to_value(experiment(1, false)).unwrap();
        let map = line.as_object_mut().unwrap();
        map.remove("memoHits");
        map.remove("memoMisses");
        map.remove("memoMsSaved");
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{line}").unwrap();

        let memo = report_from_journal(file.path()).unwrap().analysis_memo;
        assert_eq!(memo.hits, 0);
        assert_eq!(memo.misses, 0);
        assert_eq!(memo.ms_saved, 0);
        assert_eq!(memo.hit_rate, 0.0);
    }
}
