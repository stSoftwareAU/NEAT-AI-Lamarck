//! Creature-scale budgets across materially different creatures (issue #223).
//!
//! Lamarck is pointed at a creature that keeps evolving, so the guards here are
//! about *size*, not about one shape:
//!
//! 1. Every run journals the dimensions it was handed and the budgets it
//!    resolved from them — on both arms, so a `fixed` run is as legible as a
//!    `derived` one.
//! 2. `--scale-budgets fixed` resolves the same budgets whatever the creature,
//!    which is what makes it the A/B arm `derived` is measured against.
//! 3. `--scale-budgets derived` examines materially more of a materially bigger
//!    creature — measured on the scan's own output, not on the arithmetic.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use neat_ai_lamarck::analysis::{ScanBudget, scan_post_focus};
use neat_ai_lamarck::observations::StatsMode;
use neat_ai_lamarck::scale::{CreatureScale, ResidualLimits, ResolvedBudgets, ScaleBudgetMode};
use neat_ai_lamarck::scorer::{DirectoryScorer, ScoreResult, ScoreSample, ScorerError};
use neat_ai_lamarck::structural::RankedSource;
use neat_ai_lamarck::{JournalLine, LamarckConfig, RunHeaderRecord, RunResult, run_optimisation};
use neat_core::{CreatureExport, compile_creature, parse_creature_json};

const SAMPLE_RECORDS: usize = 48;

/// Synthetic creature: `inputs` inputs, `hidden` TANH hiddens, one output.
///
/// Every input is unused, so the residual pass has the whole input width to
/// rank — the property the shortlist budget bounds.
fn creature_json(inputs: usize, hidden: usize) -> String {
    let mut neurons = String::new();
    let mut synapses = String::new();
    for h in 0..hidden {
        neurons.push_str(&format!(
            r#"{{"type":"hidden","uuid":"h{h}","bias":0.01,"squash":"TANH"}},"#
        ));
        synapses.push_str(&format!(
            r#"{{"fromUUID":"input-{}","toUUID":"h{h}","weight":0.3}},"#,
            h % inputs
        ));
        synapses.push_str(&format!(
            r#"{{"fromUUID":"h{h}","toUUID":"o1","weight":0.2}},"#
        ));
    }
    neurons.push_str(r#"{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}"#);
    synapses.push_str(r#"{"fromUUID":"input-0","toUUID":"o1","weight":0.1}"#);
    format!(
        r#"{{"semanticVersion":"4.0.0","forwardOnly":true,"input":{inputs},"output":1,
           "neurons":[{neurons}],"synapses":[{synapses}]}}"#
    )
}

/// Deterministic xorshift sample: `inputs` inputs and a target per record.
fn write_sample(dir: &Path, inputs: usize, records: usize) {
    let mut bytes = Vec::new();
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..records {
        for _ in 0..=inputs {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(dir.join("0.bin"), &bytes).unwrap();
}

/// Scores everything flat, so the incumbent survives the whole run.
struct FlatScorer;

impl DirectoryScorer for FlatScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        _training_data: &Path,
        _sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let mut map = BTreeMap::new();
        for entry in std::fs::read_dir(candidates_dir).unwrap().flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(stem) = name.strip_suffix(".json") else {
                continue;
            };
            map.insert(
                stem.to_string(),
                ScoreResult {
                    score: 0.5,
                    error: 0.5,
                    complexity_penalty: 0.0,
                },
            );
        }
        Ok(map)
    }
}

/// One short run over a creature of the given shape.
fn run(dir: &Path, out: &str, inputs: usize, hidden: usize, mode: ScaleBudgetMode) -> RunResult {
    let creature = dir.join(format!("{out}-creature.json"));
    let training = dir.join(format!("{out}-data"));
    std::fs::create_dir_all(&training).unwrap();
    write_sample(&training, inputs, SAMPLE_RECORDS);
    std::fs::write(&creature, creature_json(inputs, hidden)).unwrap();
    let config = LamarckConfig {
        creature,
        training_data: training,
        timeout: Duration::from_secs(120),
        max_experiments: Some(1),
        candidates: 4,
        scale_budgets: mode,
        min_improvement: 1e-6,
        seed: Some(7),
        scorer_path: PathBuf::from("rust_scorer"),
        output_dir: dir.join(out),
        stats_mode: StatsMode::Quick,
        quick_sample_records: SAMPLE_RECORDS as u64,
        phase0_parity: false,
        screen_sample_rate: None,
        screen_promote_threshold: 0.0,
        ..LamarckConfig::default()
    };
    run_optimisation(&config, &FlatScorer).expect("run completes")
}

/// The run header of a completed run.
fn header(result: &RunResult) -> RunHeaderRecord {
    let text = std::fs::read_to_string(&result.journal_path).unwrap();
    let first = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .expect("journal");
    match JournalLine::parse(first).expect("header parses") {
        JournalLine::Header(header) => *header,
        other => panic!("first journal line is not a header: {other:?}"),
    }
}

/// Rank every input as an unused source for the focus.
fn prior_inputs(creature: &CreatureExport) -> Vec<RankedSource> {
    (0..creature.input)
        .map(|i| RankedSource {
            from_uuid: format!("input-{i}"),
            score: 0.0,
            direction: 0.0,
            weight_scale: 1.0,
            ols_weight: None,
        })
        .collect()
}

/// Sources the scan actually re-scored, identified by a score it assigned.
fn rescored(ranked: &[RankedSource]) -> usize {
    ranked.iter().filter(|s| s.score > 0.0).count()
}

#[test]
fn every_run_journals_the_creature_it_was_handed_and_the_budgets_it_resolved() {
    let dir = tempfile::tempdir().unwrap();
    let result = run(dir.path(), "fixed", 6, 3, ScaleBudgetMode::Fixed);
    let header = header(&result);

    let dims = header.creature.expect("header records creature dimensions");
    assert_eq!(dims.input, 6);
    assert_eq!(dims.output, 1);
    // Three hiddens plus the output.
    assert_eq!(dims.non_input_neurons, 4);
    assert_eq!(dims.neurons, 10);
    assert_eq!(dims.synapses, 7);
    assert!(dims.forward_only);

    let budgets = header.budgets.expect("header records resolved budgets");
    assert_eq!(budgets.scale_budgets, "fixed");
    assert_eq!(budgets.candidates, 4);
    assert_eq!(budgets.focus_count, 1);
    assert_eq!(budgets.residual_shortlist, ResidualLimits::FIXED.shortlist);
    assert_eq!(
        budgets.residual_hidden_extra,
        ResidualLimits::FIXED.hidden_extra
    );
    assert_eq!(
        budgets.synthetic_probe_rows,
        ResidualLimits::FIXED.synthetic_probes
    );
    assert_eq!(budgets.timeout_seconds, 120);
    assert!(
        budgets.graft_replay_ms > 0,
        "the resolved Phase-G budget must be recorded, not left implicit"
    );
    assert_eq!(header.config.scale_budgets.as_deref(), Some("fixed"));
}

#[test]
fn a_derived_run_journals_the_budgets_its_own_creature_earned() {
    let dir = tempfile::tempdir().unwrap();
    let result = run(dir.path(), "derived", 6, 3, ScaleBudgetMode::Derived);
    let header = header(&result);
    let budgets = header.budgets.expect("header records resolved budgets");

    assert_eq!(budgets.scale_budgets, "derived");
    // This creature is far smaller than the literals were chosen for, so the
    // derived budgets floor at them — the derived arm must never shrink.
    assert_eq!(budgets.residual_shortlist, ResidualLimits::FIXED.shortlist);
    assert_eq!(budgets.focus_count, 1);
}

#[test]
fn fixed_budgets_are_identical_across_materially_different_creatures() {
    let config = LamarckConfig {
        candidates: 100,
        timeout: Duration::from_secs(2_700),
        ..LamarckConfig::default()
    };
    let shapes = [(4usize, 2usize), (2_511, 1_590), (2_511, 4_852)];
    let resolved: Vec<ResolvedBudgets> = shapes
        .iter()
        .map(|(inputs, hidden)| {
            let creature = parse_creature_json(&creature_json(*inputs, *hidden)).unwrap();
            ResolvedBudgets::resolve(&config, CreatureScale::from_creature(&creature))
        })
        .collect();
    for budgets in &resolved {
        assert_eq!(budgets.residual, ResidualLimits::FIXED);
        assert_eq!(budgets.focus_count, resolved[0].focus_count);
    }
}

#[test]
fn derived_budgets_grow_with_the_creature_they_are_handed() {
    let config = LamarckConfig {
        candidates: 100,
        timeout: Duration::from_secs(2_700),
        scale_budgets: ScaleBudgetMode::Derived,
        ..LamarckConfig::default()
    };

    let resolve = |inputs: usize, hidden: usize| {
        let creature = parse_creature_json(&creature_json(inputs, hidden)).unwrap();
        ResolvedBudgets::resolve(&config, CreatureScale::from_creature(&creature))
    };
    let tiny = resolve(4, 2);
    let historical = resolve(2_511, 1_590);
    let current = resolve(2_511, 4_852);

    assert_eq!(
        tiny.residual,
        ResidualLimits::FIXED,
        "a creature smaller than the literals keeps them"
    );
    assert!(historical.residual.shortlist > tiny.residual.shortlist);
    assert!(current.residual.shortlist > historical.residual.shortlist);
    assert!(current.residual.hidden_extra > historical.residual.hidden_extra);
    assert!(current.focus_count > tiny.focus_count);
    // Same input width, so the synthetic-probe budget is the same: the probe
    // count follows the space it has to span, not the creature's depth.
    assert_eq!(
        current.residual.synthetic_probes,
        historical.residual.synthetic_probes
    );
}

#[test]
fn a_derived_scan_examines_more_of_a_wide_creature_than_a_fixed_one() {
    // Wide enough that the derived shortlist clears the fixed literal: 2 000
    // ranked sources buys a head of about 2 × √2 000 ≈ 90.
    let inputs = 2_000;
    let dir = tempfile::tempdir().unwrap();
    write_sample(dir.path(), inputs, SAMPLE_RECORDS);
    let creature = parse_creature_json(&creature_json(inputs, 4)).unwrap();
    let scale = CreatureScale::from_creature(&creature);
    let prior = prior_inputs(&creature);

    let fixed = ResidualLimits::resolve(ScaleBudgetMode::Fixed, scale);
    let derived = ResidualLimits::resolve(ScaleBudgetMode::Derived, scale);
    assert!(
        derived.shortlist > fixed.shortlist,
        "fixture must be wide enough to separate the arms"
    );

    let scan = |limits| {
        let mut network = compile_creature(&creature).unwrap();
        scan_post_focus(
            &creature,
            &mut network,
            dir.path(),
            "o1",
            ScanBudget::serial(Some(SAMPLE_RECORDS as u64)).with_residual(limits),
            None,
            &prior,
        )
        .expect("scan completes")
    };

    let fixed_ranked = rescored(&scan(fixed).ranked_sources);
    let derived_ranked = rescored(&scan(derived).ranked_sources);

    assert!(
        fixed_ranked <= fixed.shortlist,
        "the fixed arm cannot re-score more than its own head: {fixed_ranked}"
    );
    assert!(
        derived_ranked > fixed_ranked,
        "the derived arm must examine more of a wide creature: {derived_ranked} vs {fixed_ranked}"
    );
    assert!(
        derived_ranked <= derived.shortlist,
        "the derived arm still honours its own budget: {derived_ranked}"
    );
    // Both arms return the whole prior — only the re-scored head differs.
    assert_eq!(scan(fixed).ranked_sources.len(), prior.len());
}
