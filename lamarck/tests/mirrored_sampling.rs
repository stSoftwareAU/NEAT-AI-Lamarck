//! Mirrored (antithetic) sampling end to end (issue #203).
//!
//! Salimans et al. 2017 evaluates every perturbation `ε` beside its negation
//! `−ε`, and the whole variance-reduction argument rests on the two being
//! priced by **one** scorer call against identical records. These tests drive a
//! real run and read its journal back: that both halves of every pair reach the
//! same score map, that a pair losing in both directions is journalled as an
//! axis-level failure, and that `report` states how often the mirror rescued a
//! batch its original lost.

use neat_ai_lamarck::focus::FocusPolicy;
use neat_ai_lamarck::mirror::MirrorRole;
use neat_ai_lamarck::observations::StatsMode;
use neat_ai_lamarck::report::report_from_journal;
use neat_ai_lamarck::run::{ExperimentRecord, JournalLine, RunResult};
use neat_ai_lamarck::scorer::{DirectoryScorer, ScoreResult, ScoreSample, ScorerError};
use neat_ai_lamarck::{LamarckConfig, run_optimisation};
use neat_core::parse_creature_json;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::tempdir;

/// Baseline score of the fixture incumbent, before any scalar term.
const BASE_SCORE: f64 = 0.64;

/// A scorer on which every candidate loses, so every pair loses twice and every
/// axis it straddles is retired.
struct LosingScorer;

impl DirectoryScorer for LosingScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        _training_data: &Path,
        _sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        Ok(score_batch(candidates_dir, |stem, _| {
            if stem == "baseline" {
                BASE_SCORE
            } else {
                BASE_SCORE - 1e-3
            }
        }))
    }
}

/// A scorer that is strictly monotone in the sum of the creature's scalars.
///
/// Exactly one half of a `±δ` pair moves that sum the favoured way, so every
/// scored pair has one winner and one loser — whichever direction the strategy
/// happened to propose first.
struct MonotoneScorer {
    /// `+1.0` rewards larger scalars, `-1.0` rewards smaller ones.
    direction: f64,
}

impl DirectoryScorer for MonotoneScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        _training_data: &Path,
        _sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        Ok(score_batch(candidates_dir, |_, scalars| {
            BASE_SCORE + self.direction * 1e-3 * scalars
        }))
    }
}

/// Sum of every bias and weight in a creature file — the scalar the monotone
/// scorer prices, and the only thing a `±δ` pair moves.
fn scalar_sum(path: &Path) -> f64 {
    let creature = parse_creature_json(&fs::read_to_string(path).unwrap()).unwrap();
    creature.neurons.iter().map(|n| n.bias).sum::<f64>()
        + creature.synapses.iter().map(|s| s.weight).sum::<f64>()
}

/// Score every `*.json` in a batch directory with `score(stem, scalar_sum)`.
fn score_batch(
    candidates_dir: &Path,
    score: impl Fn(&str, f64) -> f64,
) -> BTreeMap<String, ScoreResult> {
    let mut map = BTreeMap::new();
    // The promote call may omit the incumbent (baseline reuse), so the
    // baseline is only scored when the run actually wrote it.
    for entry in fs::read_dir(candidates_dir).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(stem) = name.strip_suffix(".json") else {
            continue;
        };
        let value = score(stem, scalar_sum(&entry.path()));
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

fn run_config(dir: &Path, mirrored_sampling: bool) -> LamarckConfig {
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
        mirrored_sampling,
        screen_sample_rate: Some(0.05),
        screen_promote_threshold: 0.0,
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

/// Acceptance 1: both halves are generated together and land in the same score
/// map, which is what "scored on identical records" means — one scorer call,
/// one baseline, one sample.
#[test]
fn both_halves_of_a_pair_are_scored_in_one_call() {
    let dir = tempdir().unwrap();
    let result = run_optimisation(&run_config(dir.path(), true), &LosingScorer).unwrap();

    let mut pairs = 0;
    for record in experiments(&result) {
        // The screen map covers the whole batch; a promote map only covers
        // what screened through.
        let scores = record
            .screen_scores
            .clone()
            .unwrap_or(record.scores.clone());
        for (index, provenance) in record.candidates.iter().enumerate() {
            let Some(pair) = &provenance.mirror else {
                continue;
            };
            if pair.role != MirrorRole::Original {
                continue;
            }
            pairs += 1;
            let twin = record
                .candidates
                .iter()
                .position(|c| {
                    c.mirror.as_ref().is_some_and(|m| {
                        m.role == MirrorRole::Mirror
                            && m.axis == pair.axis
                            && (m.delta + pair.delta).abs() < 1e-12
                    })
                })
                .unwrap_or_else(|| panic!("{} has no −δ twin journalled", pair.axis));
            for half in [index, twin] {
                let stem = format!("candidate-{half:03}");
                assert!(
                    scores.contains_key(&stem),
                    "{stem} on axis {} was not scored beside its twin",
                    pair.axis
                );
            }
            assert!(
                scores.contains_key("baseline"),
                "a pair without a shared baseline supports no comparison"
            );
        }
    }
    assert!(
        pairs > 0,
        "a mixed batch proposes signed perturbations, so the run must journal pairs"
    );
}

/// Acceptance 2: when neither direction improves, the axis is journalled as a
/// failure — the run has measured a local optimum, and that is worth recording.
#[test]
fn a_pair_that_loses_twice_is_journalled_as_an_axis_failure() {
    let dir = tempdir().unwrap();
    let result = run_optimisation(&run_config(dir.path(), true), &LosingScorer).unwrap();
    let records = experiments(&result);

    let retired: Vec<String> = records
        .iter()
        .filter_map(|r| r.mirror_axis_failures.clone())
        .flatten()
        .collect();
    assert!(
        !retired.is_empty(),
        "every candidate lost, so at least one axis must be journalled as failed"
    );
    for axis in &retired {
        assert!(
            axis.starts_with("bias:") || axis.starts_with("weight:"),
            "a retired axis names the scalar it straddles, got {axis}"
        );
    }

    let report = report_from_journal(&result.journal_path).unwrap();
    assert!(report.mirror.pairs_scored > 0);
    assert_eq!(
        report.mirror.both_lost, report.mirror.original_lost,
        "no mirror can win on a scorer where everything loses"
    );
    assert_eq!(report.mirror.mirror_win_rate, 0.0);
    assert_eq!(report.mirror.axes_retired as usize, retired.len());
}

/// Acceptance 3: `report` states how often the mirror won a batch its original
/// lost. On a monotone scorer exactly one half of every pair wins, so one of
/// the two directions must produce rescues — and the rate has to see them.
#[test]
fn report_measures_the_mirror_rescues_a_monotone_scorer_produces() {
    let mut rescues = 0u64;
    for direction in [1.0, -1.0] {
        let dir = tempdir().unwrap();
        let result =
            run_optimisation(&run_config(dir.path(), true), &MonotoneScorer { direction }).unwrap();
        let mirror = report_from_journal(&result.journal_path).unwrap().mirror;
        assert!(mirror.pairs_scored > 0, "direction {direction}: no pairs");
        assert_eq!(
            mirror.both_lost, 0,
            "direction {direction}: a monotone scorer cannot lose in both directions"
        );
        if mirror.original_lost > 0 {
            assert_eq!(
                mirror.mirror_win_rate, 1.0,
                "direction {direction}: every losing original had a winning twin"
            );
        }
        rescues += mirror.mirror_won_when_original_lost;
    }
    assert!(
        rescues > 0,
        "one of the two directions must reward the mirror, or the rate measures nothing"
    );
}

/// `--no-mirrored-sampling` is the A/B arm the win rate is read against: the
/// run must behave exactly as it did before #203.
#[test]
fn mirroring_off_journals_no_pairs_and_reports_zeros() {
    let dir = tempdir().unwrap();
    let result = run_optimisation(&run_config(dir.path(), false), &LosingScorer).unwrap();

    for record in experiments(&result) {
        assert!(
            record.candidates.iter().all(|c| c.mirror.is_none()),
            "experiment {} journalled a pair with mirroring off",
            record.experiment_number
        );
        assert_eq!(record.mirror_axis_failures, None);
    }
    let mirror = report_from_journal(&result.journal_path).unwrap().mirror;
    assert_eq!(mirror.pairs_scored, 0);
    assert_eq!(mirror.axes_retired, 0);
    assert_eq!(mirror.mirror_win_rate, 0.0);
}
