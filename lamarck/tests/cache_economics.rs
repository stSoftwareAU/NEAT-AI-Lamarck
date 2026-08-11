//! Failed-cache economics: accounting, stand-down and the byte ceiling (#92).
//!
//! These are the earliest detection points for the guardrail regressing: the
//! arithmetic is pinned against known inputs, a synthetic net-negative run has
//! to disable the cache (and a net-positive one must *not*), and the ceiling has
//! to evict loudly rather than truncating the cache in silence.

use neat_ai_lamarck::candidates::{CandidateProvenance, CandidateStrategy};
use neat_ai_lamarck::failed_cache::economics::{
    CacheEconomics, CacheEconomicsConfig, ExperimentCost,
};
use neat_ai_lamarck::failed_cache::store::FAILED_CACHE_BYTES_PER_ENTRY;
use neat_ai_lamarck::focus::FocusPolicy;
use neat_ai_lamarck::observations::StatsMode;
use neat_ai_lamarck::run::{JournalLine, RunResult};
use neat_ai_lamarck::scorer::{DirectoryScorer, ScoreResult, ScoreSample, ScorerError};
use neat_ai_lamarck::{
    CandidateFingerprint, DEFAULT_FAILED_CACHE_TOLERANCE_ABS, DEFAULT_FAILED_CACHE_TOLERANCE_REL,
    FailedCandidateCache, LamarckConfig, Tolerance, run_optimisation,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::tempdir;

/// A scorer on which every candidate loses on the sample, so each experiment
/// fills the cache and the next one keeps re-proposing the same knobs.
///
/// `batch_delay` is the scorer time one batch costs; it is what the ledger
/// measures a skip's saving from, so a zero delay makes the cache worthless by
/// construction and a real delay makes it pay.
struct LosingScorer {
    batch_delay: Duration,
}

impl DirectoryScorer for LosingScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        _training_data: &Path,
        _sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        std::thread::sleep(self.batch_delay);
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
        for entry in fs::read_dir(candidates_dir).into_iter().flatten().flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(stem) = name.strip_suffix(".json") else {
                continue;
            };
            if stem == "baseline" {
                continue;
            }
            map.insert(
                stem.to_string(),
                ScoreResult {
                    // Strictly worse than the baseline: nothing is ever
                    // promoted or accepted, so every candidate is cached.
                    score: BASE_SCORE - 1e-3,
                    error: BASE_ERROR + 1e-3,
                    complexity_penalty: 0.0,
                },
            );
        }
        Ok(map)
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

fn cache_run_config(dir: &Path, candidates: usize, experiments: u64) -> LamarckConfig {
    let (creature, training_data) = tiny_setup(dir);
    LamarckConfig {
        creature,
        training_data,
        timeout: Duration::from_secs(300),
        max_experiments: Some(experiments),
        candidates,
        min_improvement: 1e-6,
        seed: Some(1),
        scorer_path: PathBuf::from("rust_scorer"),
        output_dir: dir.join("out"),
        preserve_losers: false,
        stats_mode: StatsMode::Quick,
        quick_sample_records: 8,
        focus_neuron: Some("o1".into()),
        focus_policy: FocusPolicy::Random,
        compute_correlations: false,
        max_consecutive_scorer_failures: 3,
        phase0_parity: false,
        structural_only: false,
        screen_sample_rate: Some(0.05),
        screen_promote_threshold: 0.0,
        grafts_path: None,
        graft_replay_budget: None,
        backprop_learning_rate: None,
        backprop_max_bias_adjustment_scale: None,
        failed_cache: true,
        failed_cache_max_entries: 1_000,
        failed_cache_max_age_seconds: 0,
        failed_cache_tolerance_abs: DEFAULT_FAILED_CACHE_TOLERANCE_ABS,
        failed_cache_tolerance_rel: DEFAULT_FAILED_CACHE_TOLERANCE_REL,
        failed_cache_stand_down_margin_ms: 0.0,
        failed_cache_stand_down_window: 1,
        failed_cache_max_bytes: 0,
    }
}

fn journal_lines(result: &RunResult) -> Vec<JournalLine> {
    fs::read_to_string(&result.journal_path)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| JournalLine::parse(line).expect("journal line parses"))
        .collect()
}

fn fingerprint(mutation: &str) -> CandidateFingerprint {
    CandidateFingerprint::from_provenance(
        "inc",
        &CandidateProvenance {
            strategy: CandidateStrategy::StatsWeight,
            focus_neuron: "o1".into(),
            mutation: mutation.into(),
            old_value: Some(0.0),
            new_value: Some(1.0),
        },
    )
}

/// Any drift in the estimation formula fails here first: fixed skip counts,
/// screen/promote samples and lookup, maintenance, rebuild and snapshot costs
/// produce exactly these saved/spent/net and footprint figures.
#[test]
fn accounting_matches_known_inputs() {
    let mut economics = CacheEconomics::new(CacheEconomicsConfig {
        stand_down_margin_ms: 0.0,
        stand_down_window: 0,
        max_resident_bytes: 0,
    });

    // One-off costs the cache has to earn back: 12ms rebuild, 3ms snapshot.
    economics.record_startup_rebuild(12);
    economics.record_snapshot(3, 8_192);

    // Measured scorer cost: 220ms for 11 screened creatures (baseline + 10) is
    // 20ms each; 300ms for 3 promoted creatures is 100ms each.
    economics.observe_screen(220, 11);
    economics.observe_promote(300, 3);
    assert_eq!(economics.mean_screen_ms(), 20.0);
    assert_eq!(economics.mean_promote_ms(), 100.0);

    // Experiment 1: 4 skips, none previously promoted → screen cost only. Three
    // of the four were backfilled, so only one shortened the scorer's work.
    let first = economics.record_experiment(
        1,
        ExperimentCost {
            proposals: 12,
            skipped: 4,
            skipped_previously_promoted: 0,
            backfilled: 3,
            lookup_micros: 1_500,
            maintenance_micros: 500,
            entries: 20,
        },
    );
    assert_eq!(first.saved_ms, 80.0, "4 × 20ms of avoided screen time");
    assert_eq!(first.spent_ms, 2.0, "1500µs + 500µs");
    assert_eq!(first.net_cumulative_ms, 80.0 - (12.0 + 3.0 + 2.0));
    assert_eq!(first.resident_bytes, 20 * FAILED_CACHE_BYTES_PER_ENTRY);

    // Experiment 2: 3 skips, one of which the cache had recorded as promoted;
    // the batch was refilled completely, so no scorer work was shortened.
    let second = economics.record_experiment(
        2,
        ExperimentCost {
            proposals: 8,
            skipped: 3,
            skipped_previously_promoted: 1,
            backfilled: 3,
            lookup_micros: 1_000,
            maintenance_micros: 0,
            entries: 25,
        },
    );
    assert_eq!(
        second.saved_ms,
        3.0 * 20.0 + 100.0,
        "only the promoted skip may claim promote time back"
    );

    let summary = economics.summary();
    assert_eq!(summary.proposals, 20);
    assert_eq!(summary.skipped, 7);
    assert_eq!(summary.hit_rate, 0.35);
    assert_eq!(summary.saved_ms, 7.0 * 20.0 + 1.0 * 100.0);
    assert_eq!(
        summary.wall_clock_saved_ms, 20.0,
        "only the one skip the backfill could not replace shortened the batch"
    );
    assert_eq!(summary.spent_ms, 12.0 + 3.0 + 2.0 + 1.0);
    assert_eq!(summary.net_ms, 240.0 - 18.0);
    assert_eq!(
        summary.peak_resident_bytes,
        25 * FAILED_CACHE_BYTES_PER_ENTRY
    );
    assert_eq!(summary.disk_bytes, 8_192);
    assert!(!summary.stood_down);
}

/// A run whose overhead exceeds its savings must warn, journal the stand-down,
/// stay disabled for the rest of the run — and still finish normally.
#[test]
fn net_negative_run_disables_cache() {
    let dir = tempdir().unwrap();
    // A scorer that costs nothing makes every skip worth nothing, so the
    // cache's own lookup time is pure loss: the ledger has to notice.
    let config = cache_run_config(dir.path(), 32, 4);
    let result = run_optimisation(
        &config,
        &LosingScorer {
            batch_delay: Duration::ZERO,
        },
    )
    .expect("a stand-down must not fail the run");

    assert_eq!(result.experiments, 4, "the run completes normally");
    let summary = result.cache_economics.expect("a cache-on run has a ledger");
    assert!(summary.stood_down, "the guardrail took the cache out");
    assert!(
        summary.spent_ms > 0.0 && summary.net_ms < 0.0,
        "the cache spent time and saved none: {summary:?}"
    );

    // (b) The event is journalled, with the warning that was logged.
    let lines = journal_lines(&result);
    let stand_downs: Vec<_> = lines
        .iter()
        .filter_map(|line| match line {
            JournalLine::CacheStandDown(record) => Some(record.as_ref()),
            _ => None,
        })
        .collect();
    assert_eq!(stand_downs.len(), 1, "journalled exactly once");
    let event = stand_downs[0];
    assert!(event.net_ms < 0.0);
    assert_eq!(event.margin_ms, config.failed_cache_stand_down_margin_ms);
    assert!(
        event.message.contains("STANDING DOWN"),
        "the journalled event carries the logged warning: {}",
        event.message
    );

    // (c) Every experiment after the stand-down runs without the cache.
    let experiments: Vec<_> = lines
        .iter()
        .filter_map(|line| match line {
            JournalLine::Experiment(record) => Some(record.as_ref()),
            _ => None,
        })
        .collect();
    let after: Vec<_> = experiments
        .iter()
        .filter(|record| record.experiment_number > event.experiment_number)
        .collect();
    assert!(!after.is_empty(), "the run continued after standing down");
    for record in after {
        assert_eq!(
            record.cache_skipped, None,
            "experiment {} still consulted a stood-down cache",
            record.experiment_number
        );
    }
    assert!(
        !neat_ai_lamarck::failed_cache::snapshot_path(&config.output_dir).exists(),
        "a cache judged uneconomic must not be persisted for the next run"
    );
}

/// The inverse: a cache that pays for itself must survive the same guardrail —
/// an over-eager stand-down would silently revert the feature on every run.
#[test]
fn net_positive_run_keeps_cache_enabled() {
    let dir = tempdir().unwrap();
    let mut config = cache_run_config(dir.path(), 8, 4);
    // Two consecutive losing experiments are tolerated, so the cold first
    // experiment cannot on its own end the run's cache.
    config.failed_cache_stand_down_window = 2;
    let result = run_optimisation(
        &config,
        &LosingScorer {
            batch_delay: Duration::from_millis(30),
        },
    )
    .expect("run completes");

    let summary = result.cache_economics.expect("a cache-on run has a ledger");
    assert!(
        summary.skipped > 0,
        "the same knobs are re-proposed, so there are skips to price: {summary:?}"
    );
    assert!(
        summary.saved_ms > summary.spent_ms,
        "a 30ms batch dwarfs the lookup cost: {summary:?}"
    );
    assert!(!summary.stood_down, "a paying cache must not be stood down");
    assert!(
        !journal_lines(&result)
            .iter()
            .any(|line| matches!(line, JournalLine::CacheStandDown(_))),
        "no stand-down event should be journalled"
    );
    assert!(
        neat_ai_lamarck::failed_cache::snapshot_path(&config.output_dir).is_file(),
        "a paying cache is still snapshotted for the next run"
    );
}

/// The byte ceiling is a bound, not a target: past it the cache evicts harder,
/// and the bite is reported so a truncated cache cannot pass for a working one.
#[test]
fn ceiling_evicts_and_logs() {
    let ceiling = 8 * FAILED_CACHE_BYTES_PER_ENTRY;
    let mut economics = CacheEconomics::new(CacheEconomicsConfig {
        stand_down_margin_ms: 0.0,
        stand_down_window: 0,
        max_resident_bytes: ceiling,
    });
    let mut cache = FailedCandidateCache::new(Tolerance::default(), 10_000, 0);
    for i in 0..50u64 {
        assert!(cache.insert(fingerprint(&format!("knob-{i}")), 1_000 + i));
    }
    assert!(cache.resident_bytes() > ceiling, "the ceiling is exceeded");

    let bite = economics
        .enforce_ceiling(&mut cache)
        .expect("the ceiling has to bite");
    assert_eq!(bite.evicted, 42);
    assert_eq!(bite.ceiling_bytes, ceiling);
    assert!(bite.bytes_after <= ceiling);
    assert_eq!(cache.resident_bytes(), bite.bytes_after);
    assert!(
        cache.resident_bytes() <= ceiling,
        "the resident footprint stays under the ceiling"
    );
    let message = bite.message();
    assert!(
        message.contains("CEILING") && message.contains(&ceiling.to_string()),
        "the bite has to be reportable: {message}"
    );
    assert_eq!(economics.summary().ceiling_bites, 1);
}

/// No spurious eviction: a cache inside its ceiling keeps everything it learnt.
#[test]
fn under_ceiling_run_does_not_evict() {
    let ceiling = 100 * FAILED_CACHE_BYTES_PER_ENTRY;
    let mut economics = CacheEconomics::new(CacheEconomicsConfig {
        stand_down_margin_ms: 0.0,
        stand_down_window: 0,
        max_resident_bytes: ceiling,
    });
    let mut cache = FailedCandidateCache::new(Tolerance::default(), 10_000, 0);
    for i in 0..50u64 {
        cache.insert(fingerprint(&format!("knob-{i}")), 1_000 + i);
    }

    assert!(economics.enforce_ceiling(&mut cache).is_none());
    assert_eq!(cache.len(), 50);
    assert_eq!(economics.summary().ceiling_bites, 0);
    assert_eq!(
        economics.summary().peak_resident_bytes,
        50 * FAILED_CACHE_BYTES_PER_ENTRY,
        "the footprint is still measured when the ceiling does not bite"
    );
}

/// Downstream tooling parses the end-of-run line, so a missing or renamed field
/// is caught here rather than when the benchmark run comes up empty.
#[test]
fn end_of_run_summary_line_carries_every_field() {
    let dir = tempdir().unwrap();
    let mut config = cache_run_config(dir.path(), 8, 2);
    config.failed_cache_stand_down_window = 0; // the summary is not about standing down
    let result = run_optimisation(
        &config,
        &LosingScorer {
            batch_delay: Duration::from_millis(10),
        },
    )
    .expect("run completes");

    let summary = result.cache_economics.expect("a cache-on run has a ledger");
    let line = summary.summary_line();
    for field in [
        "entries=",
        "hitRate=",
        "savedMs=",
        "spentMs=",
        "netMs=",
        "peakMemoryBytes=",
        "diskBytes=",
    ] {
        assert!(line.contains(field), "summary line missing {field}: {line}");
    }
    assert!(summary.entries > 0, "the run cached its rejections");
    assert!(
        summary.peak_resident_bytes >= summary.entries * FAILED_CACHE_BYTES_PER_ENTRY,
        "peak footprint cannot be below the final footprint: {summary:?}"
    );
    assert!(summary.disk_bytes > 0, "the snapshot is on disk");
}

/// A cache restored from a run with a larger ceiling must be brought under the
/// current one before the loop starts, not after the first insert notices.
#[test]
fn a_warm_cache_above_the_ceiling_is_evicted_at_startup() {
    let dir = tempdir().unwrap();
    let mut warm = cache_run_config(dir.path(), 8, 2);
    warm.failed_cache_stand_down_window = 0;
    let first = run_optimisation(
        &warm,
        &LosingScorer {
            batch_delay: Duration::from_millis(5),
        },
    )
    .expect("run completes");
    let cached = first.cache_economics.expect("a ledger").entries;
    assert!(cached > 4, "the first run has to fill the cache: {cached}");

    // Second run over the same output directory, under a four-entry ceiling.
    let ceiling = 4 * FAILED_CACHE_BYTES_PER_ENTRY;
    let mut tight = cache_run_config(dir.path(), 8, 1);
    tight.failed_cache_stand_down_window = 0;
    tight.failed_cache_max_bytes = ceiling;
    let second = run_optimisation(
        &tight,
        &LosingScorer {
            batch_delay: Duration::from_millis(5),
        },
    )
    .expect("run completes");

    let summary = second.cache_economics.expect("a ledger");
    assert!(
        summary.ceiling_bites > 0,
        "the ceiling has to bite on the restored cache: {summary:?}"
    );
    assert!(
        summary.entries * FAILED_CACHE_BYTES_PER_ENTRY <= ceiling,
        "the resident footprint must end at or under the ceiling: {summary:?}"
    );
}

/// With the cache off there is no ledger at all — and no economics fields in the
/// journal, so a cache-off arm journals exactly what it always did.
#[test]
fn cache_off_run_has_no_ledger() {
    let dir = tempdir().unwrap();
    let mut config = cache_run_config(dir.path(), 4, 2);
    config.failed_cache = false;
    let result = run_optimisation(
        &config,
        &LosingScorer {
            batch_delay: Duration::ZERO,
        },
    )
    .expect("run completes");

    assert!(result.cache_economics.is_none());
    let encoded = fs::read_to_string(&result.journal_path).unwrap();
    assert!(
        !encoded.contains("cacheSavedMs")
            && !encoded.contains("cacheSpentMs")
            && !encoded.contains("cacheNetCumulativeMs")
            && !encoded.contains("cacheResidentBytes")
            && !encoded.contains("cacheStandDown"),
        "a cache-off run must not journal cache economics: {encoded}"
    );
}
