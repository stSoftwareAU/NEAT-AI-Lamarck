//! Transferable operator priors across runs (Issue #221).
//!
//! The acceptance criteria are behavioural, so they are tested through the API
//! a run and a report actually use: a priors file written from one run's
//! ledger, loaded by the next, discounted for age and source/corpus drift, and
//! folded into the allocation the generator is handed. Experiments are built as
//! journal JSON and parsed back, so the evidence a prior competes against is
//! exactly the evidence a real `experiments.jsonl` carries.

use neat_ai_lamarck::candidates::CandidateStrategy;
use neat_ai_lamarck::config::LamarckConfig;
use neat_ai_lamarck::focus::FocusPolicy;
use neat_ai_lamarck::observations::StatsMode;
use neat_ai_lamarck::report::report_from_journal;
use neat_ai_lamarck::run::{ExperimentRecord, StrategyPriorsRecord};
use neat_ai_lamarck::run_optimisation;
use neat_ai_lamarck::scorer::{DirectoryScorer, ScoreResult, ScoreSample, ScorerError};
use neat_ai_lamarck::strategy_allocation::{
    DEFAULT_STRATEGY_EVIDENCE_DECAY, DEFAULT_STRATEGY_EXPLORATION_FLOOR, StrategyLedger,
    adaptive_strategies,
};
use neat_ai_lamarck::strategy_priors::{
    CORPUS_MISMATCH_CONFIDENCE, MAX_PRIOR_TRIALS, STRATEGY_PRIORS_FORMAT_VERSION,
    STRATEGY_PRIORS_MAX_AGE_HOURS, StrategyPriors, StrategyPriorsMode, TOPOLOGY_DRIFT_CONFIDENCE,
    TOPOLOGY_MATCH_CONFIDENCE, load_priors, priors_path, write_priors,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

const MIN_IMPROVEMENT: f64 = 1e-6;
const HALF_LIFE: f64 = 24.0;
const NOW: u64 = 1_800_000_000;
const HOUR: u64 = 3_600;

/// One journalled experiment: `mix` candidates in proposal order, the listed
/// indices promoted to full-corpus scoring, and an optional accepted winner.
struct Experiment {
    number: u64,
    mix: Vec<CandidateStrategy>,
    promoted: Vec<usize>,
    winner: Option<(usize, f64)>,
}

impl Experiment {
    fn json(&self) -> serde_json::Value {
        let candidates: Vec<serde_json::Value> = self
            .mix
            .iter()
            .map(|strategy| {
                json!({
                    "strategy": strategy.label(),
                    "focusNeuron": "neuron-1",
                    "mutation": format!("{} proposal", strategy.label()),
                    "oldValue": null,
                    "newValue": null,
                })
            })
            .collect();
        let mut scores = serde_json::Map::new();
        scores.insert("baseline".to_string(), json!(0.5));
        let mut screen = serde_json::Map::new();
        screen.insert("baseline".to_string(), json!(0.5));
        for (index, _) in self.mix.iter().enumerate() {
            screen.insert(format!("candidate-{index:03}"), json!(0.5));
        }
        for index in &self.promoted {
            scores.insert(format!("candidate-{index:03}"), json!(0.5));
            screen.insert(format!("candidate-{index:03}"), json!(0.500_001));
        }
        json!({
            "experimentNumber": self.number,
            "timestampUnix": 1_700_000_000u64 + self.number,
            "seed": 7,
            "incumbentId": "in2-out1-n3-s4",
            "baselineScore": 0.5,
            "focusNeuron": "neuron-1",
            "candidates": candidates,
            "scores": serde_json::Value::Object(scores),
            "screenScores": serde_json::Value::Object(screen),
            "winner": self.winner.map(|(index, _)| format!("candidate-{index:03}")),
            "improvement": self.winner.map(|(_, delta)| delta),
            "accepted": self.winner.is_some(),
            "comboMemberIndices": self.winner.map(|(index, _)| vec![index]),
            "analysisMs": 1_000,
            "scorerMs": 20_000,
            "scorerCalls": [
                { "phase": "screen", "creatures": self.mix.len() as u64 + 1, "sampleRate": 0.05, "elapsedMs": 9_000 },
                { "phase": "promote", "creatures": self.promoted.len() as u64 + 1, "elapsedMs": 11_000 },
            ],
        })
    }

    fn record(&self) -> ExperimentRecord {
        serde_json::from_value(self.json()).expect("journal shape parses as an experiment")
    }
}

fn round_robin_mix() -> Vec<CandidateStrategy> {
    adaptive_strategies(false).to_vec()
}

fn slot_of(strategy: CandidateStrategy) -> usize {
    adaptive_strategies(false)
        .iter()
        .position(|s| *s == strategy)
        .expect("strategy is an adaptive arm")
}

/// `count` experiments in which `winner` earns every accept.
fn winning_run(count: u64, winner: CandidateStrategy) -> Vec<ExperimentRecord> {
    let index = slot_of(winner);
    (1..=count)
        .map(|number| {
            Experiment {
                number,
                mix: round_robin_mix(),
                promoted: vec![index],
                winner: Some((index, 4e-6)),
            }
            .record()
        })
        .collect()
}

fn ledger_with(records: &[ExperimentRecord]) -> StrategyLedger {
    let mut ledger = StrategyLedger::new(DEFAULT_STRATEGY_EVIDENCE_DECAY, MIN_IMPROVEMENT);
    for record in records {
        ledger.observe(record);
    }
    ledger
}

/// Priors written by a run in which `winner` earned everything.
fn priors_from_winning_run(winner: CandidateStrategy, written_unix: u64) -> StrategyPriors {
    StrategyPriors::from_ledger(
        &ledger_with(&winning_run(6, winner)),
        source("in2-out1-n3-s4", 11, Some(22)),
        MIN_IMPROVEMENT,
        written_unix,
    )
}

fn source(
    incumbent_id: &str,
    fingerprint: u64,
    training: Option<u64>,
) -> neat_ai_lamarck::strategy_priors::PriorSource {
    neat_ai_lamarck::strategy_priors::PriorSource {
        incumbent_id: incumbent_id.to_string(),
        creature_fingerprint: fingerprint,
        training_key: training,
    }
}

fn slots(ledger: &StrategyLedger, strategy: CandidateStrategy) -> usize {
    ledger
        .allocate(
            adaptive_strategies(false),
            100,
            DEFAULT_STRATEGY_EXPLORATION_FLOOR,
        )
        .slots_for(strategy)
        .expect("every adaptive arm is allocated")
}

/// A scorer on which nothing improves, so the runs below measure trials and
/// cost without an accept moving the incumbent under them.
struct FlatScorer;

impl DirectoryScorer for FlatScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        _training_data: &Path,
        _sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let mut scores = BTreeMap::new();
        for entry in std::fs::read_dir(candidates_dir)
            .into_iter()
            .flatten()
            .flatten()
        {
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(stem) = name.strip_suffix(".json") else {
                continue;
            };
            let score = if stem == "baseline" {
                0.64
            } else {
                0.64 - 1e-3
            };
            scores.insert(
                stem.to_string(),
                ScoreResult {
                    score,
                    error: 1.0 - score,
                    complexity_penalty: 0.0,
                },
            );
        }
        Ok(scores)
    }
}

/// A tiny creature and one training record — enough to drive a real run.
fn tiny_setup(dir: &Path) -> (PathBuf, PathBuf) {
    let creature_path = dir.join("creature.json");
    let training = dir.join("data");
    std::fs::create_dir_all(&training).expect("training dir");
    std::fs::write(
        training.join("0.bin"),
        [1.0f32, 0.5f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .expect("training record");
    std::fs::write(
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
    .expect("creature");
    (creature_path, training)
}

/// A short real run writing into `dir/<name>`, with priors on when a path is
/// supplied and at the shipped default (`off`) when it is not.
fn run_config(dir: &Path, name: &str, priors: Option<PathBuf>) -> LamarckConfig {
    let (creature, training_data) = tiny_setup(dir);
    LamarckConfig {
        creature,
        training_data,
        timeout: Duration::from_secs(300),
        max_experiments: Some(2),
        candidates: 8,
        min_improvement: MIN_IMPROVEMENT,
        seed: Some(1),
        output_dir: dir.join(name),
        stats_mode: StatsMode::Quick,
        quick_sample_records: 8,
        focus_neuron: Some("o1".into()),
        focus_policy: FocusPolicy::Random,
        phase0_parity: false,
        screen_sample_rate: Some(0.05),
        screen_promote_threshold: 0.0,
        failed_cache: false,
        strategy_priors: match priors {
            Some(_) => StrategyPriorsMode::Seed,
            None => StrategyPriorsMode::Off,
        },
        strategy_priors_path: priors,
        ..LamarckConfig::default()
    }
}

/// Acceptance criterion: a versioned, portable prior format that round-trips.
#[test]
fn the_prior_format_is_versioned_and_round_trips_through_disk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = priors_path(dir.path());
    let priors = priors_from_winning_run(CandidateStrategy::StructuralAdd, NOW);
    assert_eq!(priors.format_version, STRATEGY_PRIORS_FORMAT_VERSION);

    assert_eq!(
        load_priors(&path).expect("a missing file is not an error"),
        None,
        "no priors yet is not a failure — the run simply starts cold"
    );
    write_priors(&path, &priors).expect("priors are written");
    let loaded = load_priors(&path)
        .expect("a written file loads")
        .expect("the file is present");
    assert_eq!(loaded, priors, "the format round-trips byte for byte");

    // A future format version is refused loudly rather than misread.
    let mut future = serde_json::to_value(&priors).expect("serialises");
    future["formatVersion"] = json!("9.9.9");
    std::fs::write(&path, future.to_string()).expect("write");
    let err = load_priors(&path).expect_err("a version mismatch is reported");
    assert!(
        err.contains("formatVersion"),
        "the reason must name the version: {err}"
    );

    // So is a corrupt file: an unreadable prior must never read as "no history".
    std::fs::write(&path, "{not json").expect("write");
    assert!(
        load_priors(&path).is_err(),
        "a corrupt prior file fails loud"
    );
}

/// Acceptance criterion: no exact historical candidate can be replayed —
/// the format carries operator-level aggregates and nothing else.
#[test]
fn the_prior_format_carries_no_candidate_or_creature_payload() {
    let priors = priors_from_winning_run(CandidateStrategy::StructuralAdd, NOW);
    let encoded = serde_json::to_string(&priors).expect("serialises");

    for forbidden in [
        "mutation",
        "newValue",
        "oldValue",
        "focusNeuron",
        "neurons",
        "synapses",
        "candidate-",
        "winner",
    ] {
        assert!(
            !encoded.contains(forbidden),
            "the portable prior must not carry `{forbidden}`: {encoded}"
        );
    }
    let value: serde_json::Value = serde_json::from_str(&encoded).expect("parses");
    let arms = value["arms"].as_object().expect("arms is a map");
    assert!(
        arms.contains_key(CandidateStrategy::StructuralAdd.label()),
        "priors are keyed by strategy label: {encoded}"
    );
    let arm = arms[CandidateStrategy::StructuralAdd.label()]
        .as_object()
        .expect("an arm is an evidence object");
    for field in ["trials", "promotions", "accepts", "scoreGain", "costMs"] {
        assert!(
            arm.contains_key(field),
            "the arm drops `{field}`: {encoded}"
        );
    }
    // The source is fingerprints only — never a path, creature or corpus name.
    let source = value["source"].as_object().expect("source object");
    assert!(
        source["creatureFingerprint"].is_u64() && source["trainingKey"].is_u64(),
        "source identity is opaque fingerprints: {encoded}"
    );
}

/// Acceptance criterion: prior confidence decays with age, and dies at the
/// maximum age rather than funding an operator forever.
#[test]
fn confidence_decays_with_age_and_expires_at_the_maximum() {
    let priors = priors_from_winning_run(CandidateStrategy::StructuralAdd, NOW);
    let same = source("in2-out1-n3-s4", 11, Some(22));

    let fresh = priors.confidence(&same, NOW, HALF_LIFE).confidence;
    let one_half_life = priors
        .confidence(&same, NOW + 24 * HOUR, HALF_LIFE)
        .confidence;
    let two_half_lives = priors
        .confidence(&same, NOW + 48 * HOUR, HALF_LIFE)
        .confidence;

    assert!(
        (fresh - 1.0).abs() < 1e-12,
        "a fresh, matching prior is 1.0"
    );
    assert!((one_half_life - 0.5).abs() < 1e-12, "{one_half_life}");
    assert!((two_half_lives - 0.25).abs() < 1e-12, "{two_half_lives}");

    let expired = priors.confidence(
        &same,
        NOW + (STRATEGY_PRIORS_MAX_AGE_HOURS as u64 + 1) * HOUR,
        HALF_LIFE,
    );
    assert_eq!(
        expired.confidence, 0.0,
        "beyond the maximum age a prior carries no confidence at all"
    );
}

/// Acceptance criterion: source and corpus mismatch reduce confidence.
#[test]
fn changed_training_data_or_a_drifted_incumbent_reduce_confidence() {
    let priors = priors_from_winning_run(CandidateStrategy::StructuralAdd, NOW);

    let matched = priors.confidence(&source("in2-out1-n3-s4", 11, Some(22)), NOW, HALF_LIFE);
    let new_corpus = priors.confidence(&source("in2-out1-n3-s4", 11, Some(99)), NOW, HALF_LIFE);
    let same_shape = priors.confidence(&source("in2-out1-n3-s4", 77, Some(22)), NOW, HALF_LIFE);
    let new_shape = priors.confidence(&source("in2-out1-n9-s20", 77, Some(22)), NOW, HALF_LIFE);
    let unknown_corpus = priors.confidence(&source("in2-out1-n3-s4", 11, None), NOW, HALF_LIFE);

    assert_eq!(matched.confidence, 1.0);
    assert_eq!(new_corpus.corpus, CORPUS_MISMATCH_CONFIDENCE);
    assert_eq!(new_corpus.confidence, CORPUS_MISMATCH_CONFIDENCE);
    assert_eq!(same_shape.source, TOPOLOGY_MATCH_CONFIDENCE);
    assert_eq!(new_shape.source, TOPOLOGY_DRIFT_CONFIDENCE);
    assert_eq!(
        unknown_corpus.corpus, CORPUS_MISMATCH_CONFIDENCE,
        "a corpus that cannot be fingerprinted is not a corpus that matches"
    );
    assert!(
        new_shape.confidence < same_shape.confidence,
        "a changed topology must be trusted less than a changed weight: {new_shape:?} vs {same_shape:?}"
    );
}

/// Acceptance criterion: priors initialise the next run's allocation.
#[test]
fn a_seeded_prior_moves_the_opening_allocation() {
    let priors = priors_from_winning_run(CandidateStrategy::StructuralAdd, NOW);
    let cold = StrategyLedger::new(DEFAULT_STRATEGY_EVIDENCE_DECAY, MIN_IMPROVEMENT);
    let even = slots(&cold, CandidateStrategy::StructuralAdd);

    let mut seeded = StrategyLedger::new(DEFAULT_STRATEGY_EVIDENCE_DECAY, MIN_IMPROVEMENT);
    let seed = priors.seed(1.0);
    assert!(seed.unknown.is_empty(), "every arm label is known");
    seeded.seed_priors(&seed.arms);

    let warmed = slots(&seeded, CandidateStrategy::StructuralAdd);
    assert!(
        warmed > even,
        "a prior that earned must open with more slots than a cold start: {warmed} vs {even}"
    );
    // …and the exploration floor still binds for every other arm.
    let allocation = seeded.allocate(
        adaptive_strategies(false),
        100,
        DEFAULT_STRATEGY_EXPLORATION_FLOOR,
    );
    for arm in adaptive_strategies(false) {
        assert!(
            allocation.slots_for(*arm).unwrap_or(0) >= 2,
            "the exploration floor is mandatory: {arm:?} got {:?}",
            allocation.slots_for(*arm)
        );
    }
    assert_eq!(allocation.total_slots(), 100);
}

/// Acceptance criterion: current-run evidence can dominate a stale prior fast.
#[test]
fn current_run_evidence_overrides_a_stale_prior_within_a_few_experiments() {
    let priors = priors_from_winning_run(CandidateStrategy::Random, NOW);
    let mut ledger = StrategyLedger::new(DEFAULT_STRATEGY_EVIDENCE_DECAY, MIN_IMPROVEMENT);
    ledger.seed_priors(&priors.seed(1.0).arms);
    assert!(
        slots(&ledger, CandidateStrategy::Random)
            > slots(&ledger, CandidateStrategy::StructuralAdd),
        "the prior opens in favour of the operator it measured"
    );

    // This run says otherwise: structural_add earns every accept.
    for record in winning_run(5, CandidateStrategy::StructuralAdd) {
        ledger.observe(&record);
    }
    assert!(
        slots(&ledger, CandidateStrategy::StructuralAdd)
            > slots(&ledger, CandidateStrategy::Random),
        "five experiments of fresh evidence must outrank the prior: structural_add={} random={}",
        slots(&ledger, CandidateStrategy::StructuralAdd),
        slots(&ledger, CandidateStrategy::Random)
    );
    assert!(
        ledger.prior_share(CandidateStrategy::Random) < 0.5,
        "the prior's share of the ledger falls as the run measures its own: {}",
        ledger.prior_share(CandidateStrategy::Random)
    );
}

/// A prior is capped in absolute mass, so no amount of history can swamp a run.
#[test]
fn a_prior_cannot_carry_more_than_the_capped_trial_mass() {
    let index = slot_of(CandidateStrategy::StructuralAdd);
    let heavy: Vec<ExperimentRecord> = (1..=30)
        .map(|number| {
            let mut mix = Vec::new();
            for _ in 0..20 {
                mix.extend(round_robin_mix());
            }
            Experiment {
                number,
                mix,
                promoted: vec![index],
                winner: Some((index, 9e-6)),
            }
            .record()
        })
        .collect();
    let priors = StrategyPriors::from_ledger(
        &ledger_with(&heavy),
        source("in2-out1-n3-s4", 11, Some(22)),
        MIN_IMPROVEMENT,
        NOW,
    );

    let seeded = priors.seed(1.0);
    for (strategy, evidence) in &seeded.arms {
        assert!(
            evidence.trials <= MAX_PRIOR_TRIALS + 1e-9,
            "{strategy:?} seeds {} trials, over the {MAX_PRIOR_TRIALS} cap",
            evidence.trials
        );
    }
    // Halving the confidence halves what is carried.
    let half = priors.seed(0.5);
    assert!(
        half.arms[&CandidateStrategy::StructuralAdd].trials
            < seeded.arms[&CandidateStrategy::StructuralAdd].trials,
        "confidence scales the mass a prior carries"
    );
}

/// Acceptance criterion: the report separates prior-driven from fresh evidence.
#[test]
fn the_report_separates_prior_driven_from_fresh_evidence() {
    let dir = tempfile::tempdir().expect("tempdir");
    let journal = dir.path().join("experiments.jsonl");
    let priors = priors_from_winning_run(CandidateStrategy::Random, NOW);
    let confidence = priors.confidence(&source("in2-out1-n3-s4", 11, Some(22)), NOW, HALF_LIFE);
    let record = StrategyPriorsRecord::new(&priors, &confidence, &priors.seed(1.0), NOW);

    let mut lines = vec![serde_json::to_string(&record).expect("record serialises")];
    for experiment in winning_run(3, CandidateStrategy::StructuralAdd) {
        lines.push(serde_json::to_string(&experiment).expect("experiment serialises"));
    }
    std::fs::write(&journal, format!("{}\n", lines.join("\n"))).expect("write journal");

    let report = report_from_journal(&journal).expect("report");
    let priors_bucket = report
        .strategy_allocation
        .priors
        .as_ref()
        .expect("the report carries a priors bucket when the run seeded any");
    assert_eq!(
        priors_bucket.format_version.as_deref(),
        Some(STRATEGY_PRIORS_FORMAT_VERSION)
    );
    assert_eq!(priors_bucket.confidence, 1.0);
    assert!(priors_bucket.seeded_trials > 0.0);
    assert!(priors_bucket.unknown_arms.is_empty());

    let seeded_arm = report
        .strategy_allocation
        .strategies
        .iter()
        .find(|row| row.strategy == CandidateStrategy::Random.label())
        .expect("random row");
    assert!(
        seeded_arm.prior_trials > 0.0,
        "the seeded arm's prior evidence is reported apart from its fresh trials"
    );
    assert!(
        seeded_arm.prior_share > 0.0 && seeded_arm.prior_share <= 1.0,
        "prior share is a fraction: {}",
        seeded_arm.prior_share
    );

    // The measured columns are exactly what the three journalled experiments
    // did — the prior seeded `random` with accepts and gain, and none of it may
    // leak into what this journal is reported to have earned.
    assert_eq!(
        seeded_arm.trials, 3,
        "one random candidate per journalled experiment, and not one inherited"
    );
    assert_eq!(
        seeded_arm.accepts, 0,
        "the prior's accepts belong to another run: {seeded_arm:?}"
    );
    assert_eq!(
        seeded_arm.score_gain, 0.0,
        "the prior's score gain belongs to another run: {seeded_arm:?}"
    );
    let earner = report
        .strategy_allocation
        .strategies
        .iter()
        .find(|row| row.strategy == CandidateStrategy::StructuralAdd.label())
        .expect("structural_add row");
    assert_eq!(
        earner.accepts, 3,
        "this journal's own accepts are still counted"
    );
    // The prior credited its accepts to `random`, so `structural_add` inherited
    // trials but no wins — and the row must show exactly that split.
    assert!(earner.prior_trials > 0.0, "it inherited trials");
    assert!(
        priors.seed(1.0).arms[&CandidateStrategy::Random].accepts > 0.0,
        "the prior really did carry accepts for the seeded arm — otherwise the \
         assertions above would pass on an empty prior"
    );
}

/// An arm label this build cannot name is reported, never quietly dropped.
#[test]
fn the_report_names_prior_arms_this_build_does_not_recognise() {
    let dir = tempfile::tempdir().expect("tempdir");
    let journal = dir.path().join("experiments.jsonl");
    let priors = priors_from_winning_run(CandidateStrategy::Random, NOW);
    let confidence = priors.confidence(&source("in2-out1-n3-s4", 11, Some(22)), NOW, HALF_LIFE);
    let mut line = serde_json::to_value(StrategyPriorsRecord::new(
        &priors,
        &confidence,
        &priors.seed(1.0),
        NOW,
    ))
    .expect("record serialises");
    // A newer Lamarck's operator, journalled by a run this build is reading.
    line["seeded"]["quantum_tunnel"] = json!({
        "trials": 4.0, "promotions": 0.0, "accepts": 0.0, "scoreGain": 0.0, "costMs": 10.0
    });
    std::fs::write(&journal, format!("{line}\n")).expect("write journal");

    let report = report_from_journal(&journal).expect("report");
    let priors_bucket = report
        .strategy_allocation
        .priors
        .as_ref()
        .expect("priors bucket");
    assert_eq!(
        priors_bucket.unknown_arms,
        vec!["quantum_tunnel".to_string()],
        "an unrecognised arm is named rather than silently reducing the seed"
    );
}

/// A priors file this build cannot read is refused **and left alone**: the run
/// starts cold rather than overwriting a chain it could not see.
#[test]
fn an_unreadable_priors_file_is_refused_and_never_overwritten() {
    let dir = tempfile::tempdir().expect("tempdir");
    let priors_file = dir.path().join("newer-priors.json");
    let corrupt = r#"{"formatVersion":"9.9.9","writtenUnix":1,"minImprovement":1e-6,"source":{"incumbentId":"x","creatureFingerprint":1,"trainingKey":2},"arms":{}}"#;
    std::fs::write(&priors_file, corrupt).expect("write");

    let result = run_optimisation(
        &run_config(dir.path(), "refused", Some(priors_file.clone())),
        &FlatScorer,
    )
    .expect("the run completes on an unusable priors file");

    assert_eq!(
        std::fs::read_to_string(&priors_file).expect("the file survives"),
        corrupt,
        "a file this build cannot read must not be replaced by less history"
    );
    let report = report_from_journal(&result.journal_path).expect("report");
    let priors = report
        .strategy_allocation
        .priors
        .as_ref()
        .expect("the refusal is journalled, not left looking like a cold start");
    let reason = priors.rejected.as_deref().expect("the reason is recorded");
    assert!(
        reason.contains("formatVersion"),
        "the reason names what was wrong: {reason}"
    );
    assert_eq!(
        priors.format_version, None,
        "nothing was read, so no version is claimed"
    );
    assert_eq!(priors.seeded_trials, 0.0);
}

/// A journal with no priors line reports no priors bucket — a cold-start arm
/// must read as cold, not as a prior-seeded run with zero confidence.
#[test]
fn a_cold_start_journal_reports_no_priors_bucket() {
    let dir = tempfile::tempdir().expect("tempdir");
    let journal = dir.path().join("experiments.jsonl");
    let lines: Vec<String> = winning_run(2, CandidateStrategy::StructuralAdd)
        .iter()
        .map(|record| serde_json::to_string(record).expect("serialises"))
        .collect();
    std::fs::write(&journal, format!("{}\n", lines.join("\n"))).expect("write journal");

    let report = report_from_journal(&journal).expect("report");
    assert!(report.strategy_allocation.priors.is_none());
    for row in &report.strategy_allocation.strategies {
        assert_eq!(row.prior_trials, 0.0, "{} carries no prior", row.strategy);
        assert_eq!(row.prior_share, 0.0, "{} carries no prior", row.strategy);
    }
}

/// End to end: a seeded run writes priors, and the next one reads them.
///
/// The unit-level tests above pin the model; this pins the wiring — that a real
/// run persists what it measured, that the next run folds it in and journals
/// what it applied, and that a default run touches neither.
#[test]
fn a_run_persists_its_evidence_and_the_next_run_opens_with_it() {
    let dir = tempfile::tempdir().expect("tempdir");
    let priors_file = dir.path().join("carried-priors.json");

    let cold = run_optimisation(&run_config(dir.path(), "cold", None), &FlatScorer)
        .expect("the cold-start run completes");
    assert!(
        !priors_path(cold.journal_path.parent().expect("output dir")).exists(),
        "a default run must neither read nor write priors"
    );

    run_optimisation(
        &run_config(dir.path(), "first", Some(priors_file.clone())),
        &FlatScorer,
    )
    .expect("the first seeded run completes");
    let written = load_priors(&priors_file)
        .expect("the priors file loads")
        .expect("a seeded run writes its evidence");
    assert!(
        written.arms.values().any(|evidence| evidence.trials > 0.0),
        "the run persists the arms it actually tried: {written:?}"
    );

    let second = run_optimisation(
        &run_config(dir.path(), "second", Some(priors_file.clone())),
        &FlatScorer,
    )
    .expect("the second seeded run completes");
    let report = report_from_journal(&second.journal_path).expect("report");
    let priors = report
        .strategy_allocation
        .priors
        .as_ref()
        .expect("the second run journals what it seeded");
    assert_eq!(
        priors.rejected, None,
        "the freshly written priors are usable"
    );
    assert!(
        priors.confidence > 0.0 && priors.seeded_trials > 0.0,
        "the second run opens on the first run's evidence: {priors:?}"
    );
    assert_eq!(
        report.strategy_allocation.priors_mode.as_deref(),
        Some("seed"),
        "the run header records the arm the journal belongs to"
    );
}

/// The mode is a documented A/B knob, and a bad one stops the run.
#[test]
fn the_mode_parses_and_a_bad_half_life_is_a_configuration_fault() {
    assert_eq!(
        StrategyPriorsMode::parse("off"),
        Some(StrategyPriorsMode::Off)
    );
    assert_eq!(
        StrategyPriorsMode::parse(" Seed "),
        Some(StrategyPriorsMode::Seed)
    );
    assert_eq!(StrategyPriorsMode::parse("replay"), None);
    assert_eq!(StrategyPriorsMode::default(), StrategyPriorsMode::Off);
    assert!(!StrategyPriorsMode::Off.is_enabled());
    assert!(StrategyPriorsMode::Seed.is_enabled());

    let config = LamarckConfig {
        strategy_priors: StrategyPriorsMode::Seed,
        strategy_priors_half_life_hours: 0.0,
        ..LamarckConfig::default()
    };
    let err = config
        .strategy_priors_policy()
        .expect_err("a zero half-life must not fall back to the default");
    assert!(
        err.contains("--strategy-priors-half-life-hours"),
        "the error should name the flag: {err}"
    );
    assert!(
        LamarckConfig::default().strategy_priors_policy().is_ok(),
        "the default configuration is valid"
    );
}
