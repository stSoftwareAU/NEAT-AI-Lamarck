//! Per-strategy screen threshold calibration end to end (issue #220).
//!
//! These tests drive real runs and read the journal back: that a calibrated run
//! records the threshold and model version every candidate faced, that it keeps
//! promoting a control sample below the threshold so its false negatives stay
//! measurable, that the shared threshold is what a thin window falls back to,
//! and that `report` prices the calibrated gate against the shared one.

use neat_ai_lamarck::focus::FocusPolicy;
use neat_ai_lamarck::observations::StatsMode;
use neat_ai_lamarck::report::report_from_journal;
use neat_ai_lamarck::run::{ExperimentRecord, JournalLine, RunResult};
use neat_ai_lamarck::scorer::{DirectoryScorer, ScoreResult, ScoreSample, ScorerError};
use neat_ai_lamarck::screen_thresholds::{
    DEFAULT_SCREEN_CONTROL_RATE, SCREEN_THRESHOLD_MODEL_VERSION, ScreenThresholdMode,
};
use neat_ai_lamarck::{LamarckConfig, run_optimisation};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::tempdir;

/// Baseline score of the fixture incumbent.
const BASE_SCORE: f64 = 0.64;

/// The shared promote threshold every test run starts from.
const SHARED_THRESHOLD: f64 = 1e-6;

/// A scorer on which every candidate is worse than the incumbent.
///
/// Nothing clears the screen, so the only full-corpus score a run can buy is a
/// control promotion — which is exactly what the guardrail is for.
struct LosingScorer;

impl DirectoryScorer for LosingScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        _training_data: &Path,
        _sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        Ok(score_batch(candidates_dir, |stem| {
            if stem == "baseline" {
                BASE_SCORE
            } else {
                BASE_SCORE - 1e-3
            }
        }))
    }
}

/// A scorer on which every candidate clears the screen by a wide margin.
struct ImprovingScorer;

impl DirectoryScorer for ImprovingScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        _training_data: &Path,
        _sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        Ok(score_batch(candidates_dir, |stem| {
            if stem == "baseline" {
                BASE_SCORE
            } else {
                BASE_SCORE + 1e-4
            }
        }))
    }
}

/// A scorer whose **sample** loves every candidate and whose **full corpus**
/// hates them: the screen is as wrong as it can be, in the direction that
/// would matter if the screen could accept anything.
struct MisleadingScreenScorer;

impl DirectoryScorer for MisleadingScreenScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        _training_data: &Path,
        sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let sampled = sample.rate < 1.0;
        Ok(score_batch(candidates_dir, |stem| match (stem, sampled) {
            ("baseline", _) => BASE_SCORE,
            (_, true) => BASE_SCORE + 1e-2,
            (_, false) => BASE_SCORE - 1e-2,
        }))
    }
}

fn score_batch(
    candidates_dir: &Path,
    score: impl Fn(&str) -> f64,
) -> BTreeMap<String, ScoreResult> {
    let mut map = BTreeMap::new();
    for entry in fs::read_dir(candidates_dir).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(stem) = name.strip_suffix(".json") else {
            continue;
        };
        let value = score(stem);
        map.insert(
            stem.to_string(),
            ScoreResult {
                score: value,
                error: 1.0 - value,
                complexity_penalty: 0.0,
            },
        );
    }
    map
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

fn run_config(dir: &Path, mode: ScreenThresholdMode) -> LamarckConfig {
    let (creature, training_data) = tiny_setup(dir);
    LamarckConfig {
        creature,
        training_data,
        timeout: Duration::from_secs(300),
        max_experiments: Some(3),
        candidates: 8,
        min_improvement: 1e-6,
        seed: Some(1),
        scorer_path: PathBuf::from("rust_scorer"),
        output_dir: dir.join("out"),
        stats_mode: StatsMode::Quick,
        quick_sample_records: 8,
        focus_neuron: Some("o1".into()),
        focus_policy: FocusPolicy::Random,
        compute_correlations: false,
        phase0_parity: false,
        screen_sample_rate: Some(0.05),
        screen_promote_threshold: SHARED_THRESHOLD,
        screen_threshold_mode: mode,
        screen_control_rate: DEFAULT_SCREEN_CONTROL_RATE,
        failed_cache: false,
        ..LamarckConfig::default()
    }
}

fn experiments(result: &RunResult) -> Vec<ExperimentRecord> {
    fs::read_to_string(&result.journal_path)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(
            |line| match JournalLine::parse(line).expect("journal parses") {
                JournalLine::Experiment(record) => Some(*record),
                _ => None,
            },
        )
        .collect()
}

/// Acceptance: every candidate is journalled with the threshold it faced and
/// the calibration model version that produced it.
#[test]
fn every_candidate_is_journalled_with_its_threshold_and_model_version() {
    let dir = tempdir().unwrap();
    let result = run_optimisation(
        &run_config(dir.path(), ScreenThresholdMode::PerStrategy),
        &ImprovingScorer,
    )
    .unwrap();

    let records = experiments(&result);
    assert!(!records.is_empty(), "the run journalled no experiment");
    for record in &records {
        let screened = record
            .screen_scores
            .as_ref()
            .expect("a screened run records subsample scores");
        let thresholds = record
            .screen_thresholds
            .as_ref()
            .expect("a calibrated run records its thresholds");
        assert_eq!(thresholds.mode, "per-strategy");
        assert_eq!(thresholds.model_version, SCREEN_THRESHOLD_MODEL_VERSION);
        assert!((thresholds.control_rate - DEFAULT_SCREEN_CONTROL_RATE).abs() < 1e-15);

        // Every scored candidate — the baseline is the anchor, not a candidate.
        let candidates: Vec<&str> = screened
            .keys()
            .filter(|stem| stem.as_str() != "baseline")
            .map(String::as_str)
            .collect();
        assert_eq!(thresholds.candidates.len(), candidates.len());
        for stem in candidates {
            let decision = thresholds
                .candidates
                .iter()
                .find(|candidate| candidate.stem == stem)
                .unwrap_or_else(|| panic!("{stem} has no journalled threshold"));
            assert_eq!(decision.model_version, SCREEN_THRESHOLD_MODEL_VERSION);
            assert!(
                decision.threshold.is_finite(),
                "{stem} was judged against a threshold that is not a number"
            );
            assert_eq!(
                decision.threshold, thresholds.thresholds[&decision.strategy],
                "{stem} did not face its own strategy's threshold"
            );
        }
    }
}

/// Acceptance: a thin window falls back to the shared threshold exactly, so
/// the opening experiments of a calibrated run gate as the shared arm does.
#[test]
fn an_uncalibrated_strategy_falls_back_to_the_shared_threshold() {
    let dir = tempdir().unwrap();
    let result = run_optimisation(
        &run_config(dir.path(), ScreenThresholdMode::PerStrategy),
        &ImprovingScorer,
    )
    .unwrap();

    let first = experiments(&result)
        .into_iter()
        .next()
        .expect("the run journalled an experiment");
    let thresholds = first.screen_thresholds.expect("calibrated run");
    for (strategy, threshold) in &thresholds.thresholds {
        assert!(
            (threshold - SHARED_THRESHOLD).abs() < 1e-15,
            "{strategy} moved its threshold on the first batch, with no evidence"
        );
        assert_eq!(
            thresholds.basis.get(strategy).map(String::as_str),
            Some("shared"),
            "{strategy} claimed a basis it had no evidence for"
        );
    }
}

/// Acceptance: the control sample keeps false negatives measurable — a batch
/// nothing clears still buys full-corpus scores, and the journal names them.
#[test]
fn a_calibrated_run_promotes_controls_below_the_threshold() {
    let dir = tempdir().unwrap();
    let result = run_optimisation(
        &run_config(dir.path(), ScreenThresholdMode::PerStrategy),
        &LosingScorer,
    )
    .unwrap();

    let mut controls = 0;
    for record in experiments(&result) {
        let thresholds = record
            .screen_thresholds
            .as_ref()
            .expect("a calibrated run records its thresholds");
        assert!(
            thresholds.controls > 0,
            "experiment {} rejected everything and measured nothing",
            record.experiment_number
        );
        let screen = record.screen_scores.as_ref().expect("screened");
        let baseline = screen["baseline"];
        for candidate in thresholds.candidates.iter().filter(|c| c.control) {
            controls += 1;
            assert!(
                screen[&candidate.stem] - baseline <= candidate.threshold,
                "{} was above its threshold, so it is not a control",
                candidate.stem
            );
            assert!(
                record.scores.contains_key(&candidate.stem),
                "{} was journalled as a control but never full-corpus scored",
                candidate.stem
            );
        }
    }
    assert!(controls > 0, "no control promotion was journalled");
}

/// The shared threshold is untouched: same scorer, no controls, no full-corpus
/// call, and no `screenThresholds` record at all.
#[test]
fn the_shared_threshold_run_is_unchanged() {
    let dir = tempdir().unwrap();
    let result = run_optimisation(
        &run_config(dir.path(), ScreenThresholdMode::Shared),
        &LosingScorer,
    )
    .unwrap();

    for record in experiments(&result) {
        assert!(
            record.screen_thresholds.is_none(),
            "the shared arm journalled a calibration record it never used"
        );
        assert!(
            record.scores.is_empty(),
            "the shared arm bought a full-corpus score for a batch it rejected"
        );
    }
}

/// The critical guardrail: calibration decides what is *scored*, never what is
/// *accepted*. A screen that loves every candidate — including the controls it
/// promotes below threshold — accepts nothing once the full corpus disagrees.
#[test]
fn calibration_never_makes_the_screen_authoritative() {
    let dir = tempdir().unwrap();
    let result = run_optimisation(
        &run_config(dir.path(), ScreenThresholdMode::PerStrategy),
        &MisleadingScreenScorer,
    )
    .unwrap();

    let records = experiments(&result);
    assert!(!records.is_empty(), "the run journalled no experiment");
    let mut promoted = 0;
    for record in &records {
        assert!(
            !record.accepted,
            "experiment {} accepted on sampled evidence the full corpus contradicts",
            record.experiment_number
        );
        assert!(record.winner.is_none());
        promoted += record.scores.keys().filter(|s| *s != "baseline").count();
        // Whatever the screen said, every full-corpus Δ here is negative.
        let baseline = record.scores.get("baseline").copied().unwrap_or_default();
        for (stem, score) in &record.scores {
            if stem != "baseline" {
                assert!(*score < baseline, "{stem} did not lose on the full corpus");
            }
        }
    }
    assert!(
        promoted > 0,
        "the misleading screen must have promoted something for this to prove anything"
    );
    assert_eq!(result.acceptances, 0, "the run accepted on screen evidence");
}

/// Acceptance: `report` breaks the calibration down by strategy and prices the
/// calibrated gate against the shared one the run applied.
#[test]
fn report_states_the_calibration_by_strategy_and_prices_it() {
    let dir = tempdir().unwrap();
    let result = run_optimisation(
        &run_config(dir.path(), ScreenThresholdMode::PerStrategy),
        &ImprovingScorer,
    )
    .unwrap();

    let report = report_from_journal(&result.journal_path).expect("the journal reports");
    let calibration = report.screen_calibration;
    assert!(
        !calibration.by_strategy.is_empty(),
        "the report gives no per-strategy calibration"
    );
    for row in &calibration.by_strategy {
        assert!(row.screened > 0, "{} screened nothing", row.strategy);
        assert!(
            row.paired <= row.screened,
            "{} paired more candidates than it screened",
            row.strategy
        );
        if row.paired > 0 {
            assert!(row.promotion_precision.is_some());
            assert!(row.full_delta.is_some());
        }
    }

    let replay = report.screen_threshold_replay;
    assert_eq!(replay.mode_as_run.as_deref(), Some("per-strategy"));
    assert_eq!(replay.model_version, SCREEN_THRESHOLD_MODEL_VERSION);
    assert!(replay.screened > 0, "the replay saw no screened candidate");
    assert_eq!(
        replay.accepts_dropped + replay.accepts_kept,
        report.acceptances,
        "every accept is either kept or dropped by the replayed gate"
    );
}

/// The false-negative guardrail is a hard configuration rule: calibration
/// without controls cannot measure what it throws away, so it is refused.
#[test]
fn calibration_without_controls_is_refused() {
    let dir = tempdir().unwrap();
    let mut config = run_config(dir.path(), ScreenThresholdMode::PerStrategy);
    config.screen_control_rate = 0.0;
    let error = config
        .screen_threshold_policy()
        .expect_err("a calibrated run with no control sample is a fault");
    assert!(error.contains("--screen-control-rate"), "{error}");
    assert!(error.contains("false negatives"), "{error}");

    // The shared arm never calibrates, so it needs no control sample.
    let mut shared = run_config(dir.path(), ScreenThresholdMode::Shared);
    shared.screen_control_rate = 0.0;
    assert!(shared.screen_threshold_policy().is_ok());

    // A rate outside `[0, 1)` is a fault under either mode.
    let mut broken = run_config(dir.path(), ScreenThresholdMode::Shared);
    broken.screen_control_rate = 1.0;
    let error = broken
        .screen_threshold_policy()
        .expect_err("a control rate of 1 promotes everything");
    assert!(error.contains("--screen-control-rate"), "{error}");
}
