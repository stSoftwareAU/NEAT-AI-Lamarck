//! Benchmark / strategy economics reporting from `experiments.jsonl`.

use crate::candidates::CandidateStrategy;
use crate::log;
use crate::run::{JournalLine, RunResult};
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
    /// Failed-candidate cache economics (issue #93).
    ///
    /// `None` for a journal that neither declares the cache in its run header
    /// nor carries a single cache field — a pre-cache journal, or a cache-off
    /// arm written before the header recorded the knob. `report` runs over
    /// historical journals, so an absent section is reported as absent rather
    /// than as a row of zeroes that reads like a cache that did nothing.
    pub cache: Option<CacheReport>,
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
                cache.push_header(&header.config);
                continue;
            }
            JournalLine::GraftReplay(replay) => {
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
        total_analysis_ms += record.analysis_ms;
        total_scorer_ms += record.scorer_ms;
        elapsed += record.analysis_ms + record.scorer_ms;
        first_ts = Some(first_ts.map_or(record.timestamp_unix, |t| t.min(record.timestamp_unix)));
        last_ts = Some(last_ts.map_or(record.timestamp_unix, |t| t.max(record.timestamp_unix)));
        *focus_counts.entry(record.focus_neuron.clone()).or_default() += 1;

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
                *focus_accepts
                    .entry(record.focus_neuron.clone())
                    .or_default() += 1;
                *focus_delta.entry(record.focus_neuron.clone()).or_default() += delta;
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

fn strategy_name(strategy: CandidateStrategy) -> String {
    match strategy {
        CandidateStrategy::Backprop => "backprop".into(),
        CandidateStrategy::MeanErrorBias => "mean_error_bias".into(),
        CandidateStrategy::StatsWeight => "stats_weight".into(),
        CandidateStrategy::StatsBias => "stats_bias".into(),
        CandidateStrategy::StatsSkewBias => "stats_skew_bias".into(),
        CandidateStrategy::StructuralAdd => "structural_add".into(),
        CandidateStrategy::StructuralAddNeuron => "structural_add_neuron".into(),
        CandidateStrategy::StructuralWeaken => "structural_weaken".into(),
        CandidateStrategy::Random => "random".into(),
    }
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
            focus_stats: None,
            candidates: vec![prov(CandidateStrategy::Random)],
            scores: BTreeMap::new(),
            screen_scores: None,
            winner: accepted.then(|| "candidate-000".to_string()),
            improvement: accepted.then_some(1e-6),
            accepted,
            analysis_ms: 1,
            scorer_ms: 2,
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

    /// A #71 run header declaring whether the cache was on (issue #93).
    fn header(failed_cache: bool) -> crate::run::RunHeaderRecord {
        let config = crate::config::LamarckConfig {
            failed_cache,
            failed_cache_max_entries: 4096,
            failed_cache_max_age_seconds: 3600,
            ..Default::default()
        };
        crate::run::RunHeaderRecord::new(
            7,
            crate::run::SeedSource::Supplied,
            crate::run::RunConfigRecord::from_config(&config),
            900,
        )
    }

    /// A journal that opens with a run header, as every post-#71 run does.
    fn journal_with_header(
        header: &crate::run::RunHeaderRecord,
        records: &[ExperimentRecord],
    ) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{}", serde_json::to_string(header).unwrap()).unwrap();
        for record in records {
            writeln!(file, "{}", serde_json::to_string(record).unwrap()).unwrap();
        }
        file
    }

    /// An experiment that consulted the cache, with the batch it went on to
    /// score and the filter counts behind it.
    fn cache_experiment(
        number: u64,
        kept: usize,
        hits: usize,
        deduplicated: usize,
        saved_ms: u128,
        spent_ms: u128,
    ) -> ExperimentRecord {
        let mut record = experiment(number, false);
        record.candidates = vec![prov(CandidateStrategy::Random); kept];
        record.cache_skipped = Some(hits);
        record.cache_deduplicated = Some(deduplicated);
        record.cache_backfilled = Some(hits);
        record.cache_size = Some(10 * number as usize);
        record.cache_lookup_ms = Some(spent_ms);
        record.cache_maintenance_ms = Some(0);
        record.cache_saved_ms = Some(saved_ms as f64);
        record.cache_spent_ms = Some(spent_ms as f64);
        record
    }

    /// A promoting experiment, i.e. one that recorded a full-corpus baseline
    /// the #84 anchor can trust.
    fn full_corpus_experiment(number: u64, baseline: f64, best: Option<f64>) -> ExperimentRecord {
        let mut record = experiment(number, best.is_some());
        record.baseline_score = baseline;
        record.scores.insert("baseline".into(), baseline);
        if let Some(best) = best {
            record.scores.insert("candidate-000".into(), best);
            record.improvement = Some(best - baseline);
        }
        record
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
            RunConfigRecord::from_config(&crate::config::LamarckConfig::default()),
            1000,
        );
        let experiment = ExperimentRecord {
            experiment_number: 1,
            timestamp_unix: 1000,
            seed: Some(42),
            incumbent_id: "x".into(),
            baseline_score: 0.4,
            focus_neuron: "h1".into(),
            focus_stats: None,
            candidates: vec![prov(CandidateStrategy::Random)],
            scores: BTreeMap::new(),
            screen_scores: None,
            winner: None,
            improvement: None,
            accepted: false,
            analysis_ms: 1,
            scorer_ms: 2,
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
            focus_stats: None,
            candidates: vec![
                prov(CandidateStrategy::Random),
                prov(CandidateStrategy::Backprop),
            ],
            scores: {
                let mut m = BTreeMap::new();
                m.insert("baseline".into(), 0.4);
                m.insert("candidate-000".into(), 0.400002);
                m
            },
            screen_scores: Some({
                let mut m = BTreeMap::new();
                m.insert("baseline".into(), 0.4);
                m.insert("candidate-000".into(), 0.41);
                m.insert("candidate-001".into(), 0.39);
                m
            }),
            winner: Some("candidate-000".into()),
            improvement: Some(2e-6),
            accepted: true,
            analysis_ms: 10,
            scorer_ms: 20,
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
            focus_stats: None,
            candidates: vec![prov(CandidateStrategy::Random)],
            scores: {
                let mut m = BTreeMap::new();
                m.insert("baseline".into(), 0.400002);
                m.insert("candidate-000".into(), 0.400001);
                m
            },
            screen_scores: None,
            winner: None,
            improvement: None,
            accepted: false,
            analysis_ms: 5,
            scorer_ms: 15,
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

    /// Issue #93: a cache-on journal reports its economics arithmetic exactly.
    ///
    /// The figures are hand-computed here rather than recomputed by the test,
    /// because #69's go/no-go is read off them.
    #[test]
    fn report_cache_on_journal_states_the_economics() {
        let head = header(true);
        let file = journal_with_header(
            &head,
            &[
                cache_experiment(1, 6, 2, 0, 400, 5),
                cache_experiment(2, 5, 3, 0, 600, 7),
            ],
        );

        let cache = report_from_journal(file.path())
            .unwrap()
            .cache
            .expect("cache-on journal carries a cache section");

        assert!(cache.enabled);
        assert_eq!(cache.max_entries, Some(4096));
        assert_eq!(cache.max_age_seconds, Some(3600));
        assert_eq!(cache.experiments_with_cache, 2);
        assert_eq!(cache.cache_hits, 5);
        assert_eq!(cache.backfilled, 5);
        assert_eq!(cache.batch_candidates, 11);
        // 6 kept + 2 hits, then 5 kept + 3 hits.
        assert_eq!(cache.proposals_examined, 16);
        assert!((cache.hit_rate - 5.0 / 16.0).abs() < 1e-12);
        assert_eq!(cache.final_cache_size, Some(20));
        assert_eq!(cache.peak_cache_size, Some(20));
        assert_eq!(cache.estimated_saved_ms, 1000.0);
        assert_eq!(cache.spent_ms, 12.0);
        assert_eq!(cache.net_ms, 988.0);
        assert_eq!(cache.stood_down_at_experiment, None);
    }

    /// The startup rebuild is a cost the run paid, so it is in the spend.
    #[test]
    fn report_charges_the_startup_rebuild_to_the_cache() {
        let head = header(true);
        let mut first = cache_experiment(1, 4, 1, 0, 100, 5);
        first.cache_rebuild_ms = Some(900);
        let file = journal_with_header(&head, &[first, cache_experiment(2, 4, 1, 0, 100, 5)]);

        let cache = report_from_journal(file.path()).unwrap().cache.unwrap();

        assert_eq!(cache.rebuild_ms, 900);
        assert_eq!(cache.spent_ms, 910.0);
        // A rebuild this expensive turns a nominally winning cache into a loss,
        // which is exactly the number the go/no-go needs to see.
        assert_eq!(cache.net_ms, -710.0);
    }

    /// A cache-off arm reports the section as present but idle, so it can be
    /// compared against a cache-on arm rather than looking like a broken run.
    #[test]
    fn report_cache_off_journal_reports_an_idle_cache() {
        let head = header(false);
        let file = journal_with_header(&head, &[experiment(1, false), experiment(2, true)]);

        let report = report_from_journal(file.path()).unwrap();

        assert!(
            report.cache.is_none(),
            "cache never ran and never journalled"
        );
        assert_eq!(report.experiments, 2);
    }

    /// A pre-cache journal has neither header nor cache fields, and must not
    /// grow a fabricated section.
    #[test]
    fn report_pre_cache_journal_omits_cache_section() {
        let file = journal_of(&[experiment(1, false), experiment(2, true)]);

        let report = report_from_journal(file.path()).unwrap();

        assert!(report.cache.is_none());
        assert_eq!(report.experiments, 2);
        assert_eq!(report.acceptances, 1);
    }

    /// A mixed journal: some experiments consulted the cache, some ran before
    /// it was warm enough to be consulted at all.
    #[test]
    fn report_mixed_journal_counts_only_the_cache_on_experiments() {
        let head = header(true);
        let file = journal_with_header(
            &head,
            &[
                experiment(1, false),
                cache_experiment(2, 4, 2, 0, 250, 3),
                experiment(3, false),
                cache_experiment(4, 4, 1, 0, 125, 2),
            ],
        );

        let report = report_from_journal(file.path()).unwrap();
        let cache = report.cache.unwrap();

        assert_eq!(report.experiments, 4);
        assert_eq!(cache.experiments_with_cache, 2);
        assert_eq!(cache.cache_hits, 3);
        assert_eq!(cache.proposals_examined, 11);
        assert_eq!(cache.estimated_saved_ms, 375.0);
        assert_eq!(cache.spent_ms, 5.0);
    }

    /// Issue #93 / #83: a skip belongs to exactly one mechanism, and the
    /// proposal count reconciles against the batch that was actually scored.
    #[test]
    fn report_attributes_each_suppressed_proposal_to_one_mechanism() {
        let head = header(true);
        // Experiment 2 also lost candidates to generation-time gating (#83):
        // `backprop` proposed nothing, so the batch is short. Those candidates
        // never reached the filter and must not appear anywhere in the cache's
        // account.
        let file = journal_with_header(
            &head,
            &[
                cache_experiment(1, 6, 2, 1, 300, 4),
                cache_experiment(2, 3, 1, 2, 150, 2),
            ],
        );

        let cache = report_from_journal(file.path()).unwrap().cache.unwrap();

        assert_eq!(cache.cache_hits, 3);
        assert_eq!(cache.deduplicated, 3);
        assert_eq!(cache.batch_candidates, 9);
        // Everything the filter saw is either scored, a cache hit, or a
        // duplicate — nothing is counted twice and nothing is unexplained.
        assert_eq!(
            cache.proposals_examined,
            cache.batch_candidates + cache.cache_hits + cache.deduplicated
        );
        // The short batch in experiment 2 is #83's doing, so the cache is not
        // credited with the three candidates that were never proposed.
        assert_eq!(cache.proposals_examined, 15);
    }

    /// Issue #92/#93: the report names the experiment where the guardrail fired
    /// and stops counting the cache after it.
    #[test]
    fn report_names_the_experiment_where_the_cache_stood_down() {
        use crate::run::{CacheStandDownKind, CacheStandDownRecord};

        let head = header(true);
        let stand_down = CacheStandDownRecord {
            record: CacheStandDownKind::CacheStandDown,
            timestamp_unix: 1002,
            experiment_number: 2,
            saved_ms: 40.0,
            spent_ms: 600.0,
            net_ms: -560.0,
            window_experiments: 2,
            margin_ms: 1.5,
            entries: 20,
            message: "spend 600ms exceeds estimated saving 40ms by more than 1.5x".into(),
        };
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{}", serde_json::to_string(&head).unwrap()).unwrap();
        for record in [
            cache_experiment(1, 4, 1, 0, 20, 300),
            cache_experiment(2, 4, 1, 0, 20, 300),
        ] {
            writeln!(file, "{}", serde_json::to_string(&record).unwrap()).unwrap();
        }
        writeln!(file, "{}", serde_json::to_string(&stand_down).unwrap()).unwrap();
        // After standing down the run journals no cache fields at all.
        for record in [experiment(3, false), experiment(4, true)] {
            writeln!(file, "{}", serde_json::to_string(&record).unwrap()).unwrap();
        }

        let report = report_from_journal(file.path()).unwrap();
        let cache = report.cache.unwrap();

        assert_eq!(cache.stood_down_at_experiment, Some(2));
        assert!(cache.stood_down_reason.unwrap().contains("exceeds"));
        assert_eq!(cache.experiments_with_cache, 2);
        assert_eq!(report.experiments, 4);
        assert_eq!(cache.net_ms, -560.0);
    }

    /// Issue #69's gate metric, from a journal that can support it.
    #[test]
    fn report_gate_metric_uses_full_corpus_scores() {
        let mut opening = full_corpus_experiment(1, 0.40, None);
        opening.timestamp_unix = 0;
        let mut win = full_corpus_experiment(2, 0.40, Some(0.44));
        // Half an hour apart, so the per-hour rate is twice the improvement.
        win.timestamp_unix = 1800;
        let file = journal_of(&[opening, win]);

        let report = report_from_journal(file.path()).unwrap();

        assert_eq!(report.opening_baseline_score, Some(0.40));
        let rate = report
            .score_improvement_per_wall_hour
            .expect("full-corpus anchor and a wall-clock span");
        assert!((rate - 0.08).abs() < 1e-9, "unexpected rate {rate}");
    }

    /// Issue #84: under `--skip-phase0` a journal whose batches all screened
    /// empty has only sampled baselines, so the gate metric is unavailable
    /// rather than a number that looks authoritative.
    #[test]
    fn report_gate_metric_unavailable_without_a_full_corpus_anchor() {
        // No `scores.baseline`: every batch screened empty, so the only
        // baseline recorded is the 5% sample.
        let mut first = experiment(1, false);
        first.baseline_score = 0.347_028_969_693;
        first.timestamp_unix = 0;
        let mut second = experiment(2, false);
        second.baseline_score = 0.352_1;
        second.timestamp_unix = 1800;
        let file = journal_of(&[first, second]);

        let report = report_from_journal(file.path()).unwrap();

        assert_eq!(report.opening_baseline_score, None);
        assert_eq!(report.total_score_improvement, None);
        assert_eq!(report.score_improvement_per_wall_hour, None);
    }

    /// A journal with no wall-clock span cannot support a per-hour rate either.
    #[test]
    fn report_gate_metric_unavailable_without_a_wall_clock_span() {
        let mut only = full_corpus_experiment(1, 0.40, Some(0.44));
        only.timestamp_unix = 500;
        let file = journal_of(&[only]);

        let report = report_from_journal(file.path()).unwrap();

        assert!((report.total_score_improvement.unwrap() - 0.04).abs() < 1e-12);
        assert_eq!(report.wall_duration_ms, None);
        assert_eq!(report.score_improvement_per_wall_hour, None);
    }

    /// A `--skip-phase0` cache-on run still reports its cache economics; only
    /// the gate metric is withheld.
    #[test]
    fn report_skip_phase0_cache_run_reports_cache_without_the_gate_metric() {
        let head = header(true);
        let file = journal_with_header(
            &head,
            &[
                cache_experiment(1, 4, 2, 0, 200, 3),
                cache_experiment(2, 4, 2, 0, 200, 3),
            ],
        );

        let report = report_from_journal(file.path()).unwrap();

        assert_eq!(report.score_improvement_per_wall_hour, None);
        let cache = report.cache.expect("cache economics do not need an anchor");
        assert_eq!(cache.cache_hits, 4);
        assert_eq!(cache.estimated_saved_ms, 400.0);
    }
}
