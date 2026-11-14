//! Behaviour tests for the offline weight bench (issue #266).

use neat_ai_lamarck::weight_bench::experiment::ALLOWED_BUDGET_MULTIPLES;
use neat_ai_lamarck::weight_bench::suite::{
    CorpusSpec, Family, Problem, ProblemSpec, load_corpus, problem_from_corpus,
};
use neat_ai_lamarck::weight_bench::{
    Arm, BenchError, BudgetedObjective, CMA_SIGMA0, Manifest, WeightObjective, build_problem,
    cma_es, is_extrapolation, lamarck_refine, run_arm, run_manifest, select_influential,
};

fn problem(family: Family, seed: u64) -> (Problem, WeightObjective) {
    let p = build_problem(&ProblemSpec {
        family,
        dimension: 20,
        seed,
    })
    .unwrap();
    let o =
        WeightObjective::new(p.creature.clone(), p.synapse_indices.clone(), &p.screening).unwrap();
    (p, o)
}

fn tiny_manifest() -> Manifest {
    Manifest {
        families: vec![Family::Tanh],
        dimensions: vec![20],
        budget_multiples: vec![10],
        validation_seeds: vec![1001],
        bootstrap_resamples: 200,
        bootstrap_seed: 266,
        corpus: None,
    }
}

#[test]
fn budget_refuses_evaluation_past_limit() {
    let (_, o) = problem(Family::Tanh, 1);
    let x = o.start();
    let mut b = BudgetedObjective::new(&o, 2);
    assert!(b.eval(&x).is_ok());
    assert!(b.eval(&x).is_ok());
    assert_eq!(b.eval(&x), Err(BenchError::BudgetExhausted { limit: 2 }));
    assert_eq!(b.used(), 2);
    assert_eq!(b.remaining(), 0);
}

#[test]
fn arms_respect_budget_and_improve() {
    let (_, o) = problem(Family::Tanh, 1);
    let dim = o.dimension();
    let start = o.loss(&o.start()).unwrap();
    for multiple in [10, 30] {
        let limit = multiple * dim;
        let mut a = BudgetedObjective::new(&o, limit);
        lamarck_refine(&mut a, 7).unwrap();
        assert!(a.used() <= limit);
        let mut c = BudgetedObjective::new(&o, limit);
        cma_es(&mut c, 7, CMA_SIGMA0).unwrap();
        assert!(c.used() <= limit);
        if multiple == 30 {
            assert!(a.best().unwrap().0 < start);
            assert!(c.best().unwrap().0 < start);
        }
    }
}

fn best_of(o: &WeightObjective, limit: usize, seed: u64, cma: bool) -> f64 {
    let mut b = BudgetedObjective::new(o, limit);
    if cma {
        cma_es(&mut b, seed, CMA_SIGMA0).unwrap();
    } else {
        lamarck_refine(&mut b, seed).unwrap();
    }
    b.best().unwrap().0
}

#[test]
fn arms_are_deterministic_and_seed_sensitive() {
    let (_, o) = problem(Family::Tanh, 2);
    let limit = 10 * o.dimension();
    for cma in [false, true] {
        assert_eq!(best_of(&o, limit, 5, cma), best_of(&o, limit, 5, cma));
    }
    assert_ne!(best_of(&o, limit, 5, true), best_of(&o, limit, 6, true));
}

#[test]
fn arms_share_the_start_without_mutating_the_parent() {
    let (p, objective) = problem(Family::Tanh, 2);
    let parent = serde_json::to_value(&p.creature).unwrap();
    let start = objective.start();
    let loss = objective.loss(&start).unwrap();
    for arm in [Arm::LamarckRefine, Arm::CmaEs] {
        let mut budgeted = BudgetedObjective::new(&objective, 200);
        assert_eq!(budgeted.start(), start);
        assert_eq!(budgeted.eval(&start).unwrap(), loss);
        run_arm(&mut budgeted, arm, 2).unwrap();
        assert!(budgeted.best().unwrap().0 <= loss);
        assert_eq!(objective.start(), start);
        assert_eq!(objective.loss(&start).unwrap(), loss);
        assert_eq!(serde_json::to_value(&p.creature).unwrap(), parent);
    }
}

#[test]
fn cma_es_improves_at_large_budget() {
    let (_, o) = problem(Family::Tanh, 3);
    let start = o.loss(&o.start()).unwrap();
    let best = best_of(&o, 100 * o.dimension(), 11, true);
    assert!(best / start < 1.0, "normalised {}", best / start);
}

#[test]
fn manifest_validation() {
    assert!(tiny_manifest().validate().is_ok());
    let mut m = tiny_manifest();
    m.budget_multiples = vec![50];
    let err = m.validate().unwrap_err().to_string();
    assert!(err.contains("50"), "{err}");
    for d in [19, 101] {
        let mut m = tiny_manifest();
        m.dimensions = vec![d];
        assert!(matches!(m.validate(), Err(BenchError::InvalidConfig(_))));
    }
    for d in [20, 100] {
        let mut m = tiny_manifest();
        m.dimensions = vec![d];
        assert!(m.validate().is_ok());
    }
    let mut m = tiny_manifest();
    m.validation_seeds = vec![];
    assert!(matches!(m.validate(), Err(BenchError::InvalidConfig(_))));
    for b in ALLOWED_BUDGET_MULTIPLES {
        let mut m = tiny_manifest();
        m.budget_multiples = vec![b];
        assert!(m.validate().is_ok());
    }
}

#[test]
fn manifest_rejects_duplicate_trial_identifiers() {
    let control = tiny_manifest();
    assert!(control.validate().is_ok());
    for field in [
        "families",
        "dimensions",
        "budgetMultiples",
        "validationSeeds",
    ] {
        let mut value = serde_json::to_value(&control).unwrap();
        let items = value[field].as_array_mut().unwrap();
        items.push(items[0].clone());
        let manifest: Manifest = serde_json::from_value(value).unwrap();
        assert_eq!(
            manifest.validate(),
            Err(BenchError::InvalidConfig(format!(
                "duplicate entry in {field}"
            )))
        );
    }
}

#[test]
fn manifest_rejects_unknown_fields() {
    let control = tiny_manifest();
    let value = serde_json::to_value(&control).unwrap();
    assert!(serde_json::from_value::<Manifest>(value.clone()).is_ok());
    let mut stale = value;
    stale["calibrationSeeds"] = serde_json::json!([1, 2]);
    let err = serde_json::from_value::<Manifest>(stale).unwrap_err();
    assert!(err.to_string().contains("calibrationSeeds"), "{err}");
}

#[test]
fn objective_requires_distinct_weights() {
    let (p, _) = problem(Family::Tanh, 1);
    let selected = p.synapse_indices[0];
    assert!(WeightObjective::new(p.creature.clone(), vec![selected], &p.screening).is_ok());
    assert_eq!(
        WeightObjective::new(p.creature.clone(), vec![selected, selected], &p.screening).err(),
        Some(BenchError::InvalidConfig(
            "synapse indices must be distinct".into()
        ))
    );
}

#[test]
fn objective_requires_nonzero_record_dimensions() {
    let (p, _) = problem(Family::Tanh, 1);
    let selected = p.synapse_indices[0];
    assert!(WeightObjective::new(p.creature.clone(), vec![selected], &p.screening).is_ok());
    for (input, output) in [(0, 1), (1, 0), (0, 0)] {
        let mut creature = p.creature.clone();
        creature.input = input;
        creature.output = output;
        assert_eq!(
            WeightObjective::new(creature, vec![selected], &[1.0]).err(),
            Some(BenchError::InvalidConfig(
                "creature input and output must both be positive".into()
            ))
        );
    }
}

#[test]
fn extrapolation_boundary() {
    assert!(!is_extrapolation(40));
    assert!(is_extrapolation(41));
    assert!(is_extrapolation(100));
}

#[test]
fn smoke_manifest_end_to_end() {
    let m = tiny_manifest();
    let r = run_manifest(&m).unwrap();
    assert_eq!(r.coverage.planned_runs, r.coverage.completed_runs);
    assert_eq!(r.coverage.failed_runs, 0);
    assert!(r.total_evaluations > 0);
    assert_eq!(r.cells.len(), 1);
    for cell in &r.cells {
        assert_eq!(cell.comparisons.len(), 1);
        assert_eq!(
            (cell.comparisons[0].first, cell.comparisons[0].second),
            (Arm::CmaEs, Arm::LamarckRefine)
        );
        for c in &cell.comparisons {
            assert_eq!(c.screening.paired_differences.len(), 1);
            assert_eq!(c.held_out.paired_differences.len(), 1);
        }
    }
    for row in &r.runs {
        assert!(row.evaluations_used <= row.budget, "{row:?}");
        assert!(!row.extrapolation);
        assert_eq!(row.phase, "validation");
    }
}

/// The mark is carried on every row and cell just past the boundary, and is
/// absent at the boundary itself. D=41 is cheap enough for the routine gate.
#[test]
fn extrapolation_flag_marks_rows_and_cells_just_past_the_boundary() {
    for (dimension, expected) in [(40, false), (41, true)] {
        let mut m = tiny_manifest();
        m.dimensions = vec![dimension];
        let r = run_manifest(&m).unwrap();
        assert!(!r.runs.is_empty());
        assert!(
            r.runs.iter().all(|row| row.extrapolation == expected),
            "D={dimension}"
        );
        assert_eq!(r.cells.len(), 1);
        assert_eq!(r.cells[0].extrapolation, expected, "D={dimension}");
    }
}

#[test]
#[ignore = "CMA-ES at D=100 takes ~40s in a debug build; run with --release -- --ignored"]
fn extrapolation_flag_on_rows_for_large_dimension() {
    let mut m = tiny_manifest();
    m.dimensions = vec![100];
    m.budget_multiples = vec![10];
    let r = run_manifest(&m).unwrap();
    assert!(r.runs.iter().all(|row| row.extrapolation));
    assert!(r.cells.iter().all(|cell| cell.extrapolation));
}

#[test]
fn select_influential_contract() {
    let (p, _) = problem(Family::HardTanh, 1);
    let n = p.creature.synapses.len();
    assert_eq!(n, 144);
    assert!(matches!(
        select_influential(&p.creature, &p.screening, n + 1),
        Err(BenchError::InvalidConfig(_))
    ));
    assert!(matches!(
        select_influential(&p.creature, &p.screening, 0),
        Err(BenchError::InvalidConfig(_))
    ));
    let (idx, evals) = select_influential(&p.creature, &p.screening, 20).unwrap();
    assert_eq!(idx.len(), 20);
    assert!(idx.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(evals, n + 1);
}

/// Writes a creature (8 inputs, 4 TANH hiddens reading every input, one
/// IDENTITY output: 36 synapses) and 600 deterministic records.
fn corpus_fixture(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
    use std::io::Write;
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for h in 0..4 {
        neurons.push(format!(
            r#"{{"type":"hidden","uuid":"h{h}","bias":0.0,"squash":"TANH"}}"#
        ));
        for i in 0..8 {
            let w = ((h * 8 + i) % 11) as f64 * 0.1 - 0.5;
            synapses.push(format!(
                r#"{{"fromUUID":"input-{i}","toUUID":"h{h}","weight":{w}}}"#
            ));
        }
        let w = 0.3 + h as f64 * 0.1;
        synapses.push(format!(
            r#"{{"fromUUID":"h{h}","toUUID":"o1","weight":{w}}}"#
        ));
    }
    neurons.push(r#"{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}"#.to_string());
    let json = format!(
        r#"{{"semanticVersion":"4.0.0","forwardOnly":true,"input":8,"output":1,"neurons":[{}],"synapses":[{}]}}"#,
        neurons.join(","),
        synapses.join(",")
    );
    let creature = dir.join("creature.json");
    std::fs::write(&creature, json).unwrap();
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    let mut file = std::io::BufWriter::new(std::fs::File::create(data.join("0.bin")).unwrap());
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..600 {
        for _ in 0..=8 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
            file.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    file.flush().unwrap();
    (creature, data)
}

fn corpus_spec(dir: &std::path::Path, screening: usize, held_out: usize) -> CorpusSpec {
    let (creature, training_data) = corpus_fixture(dir);
    CorpusSpec {
        creature,
        training_data,
        screening_records: screening,
        held_out_records: held_out,
        max_records: 600,
    }
}

fn corpus_manifest(spec: CorpusSpec) -> Manifest {
    Manifest {
        families: vec![Family::Corpus],
        corpus: Some(spec),
        bootstrap_resamples: 50,
        ..tiny_manifest()
    }
}

#[test]
fn corpus_manifest_runs_within_budget() {
    let dir = tempfile::tempdir().unwrap();
    let results = run_manifest(&corpus_manifest(corpus_spec(dir.path(), 128, 128))).unwrap();
    assert_eq!(
        results.coverage.planned_runs,
        results.coverage.completed_runs
    );
    assert_eq!(results.coverage.failed_runs, 0);
    assert!(results.runs.iter().all(|r| r.family == Family::Corpus));
    assert!(results.runs.iter().all(|r| r.evaluations_used <= r.budget));
    assert_eq!(results.runs.len(), 2);
    // Wiring check: each row starts from the creature's own weights on the
    // window's first 128 screening and last 128 held-out records.
    let spec = corpus_spec(dir.path(), 128, 128);
    let (creature, packed) = load_corpus(&spec).unwrap();
    let p = problem_from_corpus(&creature, &packed, &spec, 20, 1001).unwrap();
    let o =
        WeightObjective::new(p.creature.clone(), p.synapse_indices.clone(), &p.screening).unwrap();
    let start = o.start();
    for row in &results.runs {
        assert_eq!(row.start_screening_loss, Some(o.loss(&start).unwrap()));
        assert_eq!(
            row.held_out_loss_start,
            Some(o.loss_on(&start, &p.held_out).unwrap())
        );
    }
}

#[test]
fn corpus_with_too_few_records_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    // Positive control: the same files are accepted with a feasible split.
    assert!(run_manifest(&corpus_manifest(corpus_spec(dir.path(), 128, 128))).is_ok());
    let err = run_manifest(&corpus_manifest(corpus_spec(dir.path(), 400, 400))).unwrap_err();
    match err {
        BenchError::InvalidConfig(m) => assert!(m.contains("800"), "{m}"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
    // Boundary: 300 + 300 needs exactly the 600 records on offer.
    let spec = corpus_spec(dir.path(), 300, 300);
    let (creature, packed) = load_corpus(&spec).unwrap();
    assert_eq!(packed.len(), 600 * 9);
    assert!(problem_from_corpus(&creature, &packed, &spec, 20, 7).is_ok());
    match load_corpus(&corpus_spec(dir.path(), 301, 300)).unwrap_err() {
        BenchError::InvalidConfig(m) => assert!(m.contains("601"), "{m}"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
    // problem_from_corpus refuses a window too short for the split on its own.
    let short = CorpusSpec {
        screening_records: 301,
        ..spec
    };
    match problem_from_corpus(&creature, &packed, &short, 20, 7).unwrap_err() {
        BenchError::InvalidConfig(m) => assert!(m.contains("601"), "{m}"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn corpus_and_families_must_agree() {
    let dir = tempfile::tempdir().unwrap();
    let spec = corpus_spec(dir.path(), 128, 128);
    let mixed = Manifest {
        corpus: Some(spec),
        ..tiny_manifest()
    };
    let no_spec = Manifest {
        families: vec![Family::Corpus],
        ..tiny_manifest()
    };
    for m in [mixed, no_spec] {
        match run_manifest(&m).unwrap_err() {
            BenchError::InvalidConfig(msg) => assert!(msg.contains("corpus"), "{msg}"),
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }
}

#[test]
fn corpus_problem_splits_records() {
    let dir = tempfile::tempdir().unwrap();
    let spec = corpus_spec(dir.path(), 128, 128);
    let (creature, packed) = load_corpus(&spec).unwrap();
    assert_eq!(packed.len(), 600 * 9);
    let p = problem_from_corpus(&creature, &packed, &spec, 20, 7).unwrap();
    assert_eq!(p.screening[..], packed[..128 * 9]);
    assert_eq!(p.held_out[..], packed[(600 - 128) * 9..]);
    assert_eq!(p.full.len(), (600 - 128) * 9);
    // Full and held-out are disjoint and together cover the window exactly.
    let all = [&p.full[..], &p.held_out[..]].concat();
    assert_eq!(all[..], packed[..]);
    assert!(
        build_problem(&ProblemSpec {
            family: Family::Corpus,
            dimension: 20,
            seed: 1
        })
        .is_err()
    );
}

#[test]
fn total_evaluations_count_confirmations_and_preparation_once_per_problem() {
    let m = tiny_manifest();
    let r = run_manifest(&m).unwrap();
    // One validation problem, built once and shared by both arms.
    let prep = |seed| {
        build_problem(&ProblemSpec {
            family: Family::Tanh,
            dimension: 20,
            seed,
        })
        .unwrap()
        .preparation_evaluations
    };
    let preparation = prep(1001);
    assert!(preparation > 0);
    // Two validation rows, five confirmation evaluations each (screening
    // start; full start and best; held-out start and best).
    assert_eq!(r.runs.len(), 2);
    assert!(r.runs.iter().all(|row| row.error.is_none()));
    for row in &r.runs {
        assert_eq!(row.confirmation_evaluations, 5);
        assert_eq!(row.preparation_evaluations, prep(row.seed));
    }
    let used: usize = r.runs.iter().map(|row| row.evaluations_used).sum();
    assert_eq!(r.total_evaluations, used + 2 * 5 + preparation);
}
