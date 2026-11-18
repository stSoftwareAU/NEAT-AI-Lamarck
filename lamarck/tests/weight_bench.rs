//! Behaviour tests for the offline weight bench (issue #266).

use neat_ai_lamarck::weight_bench::experiment::ALLOWED_BUDGET_MULTIPLES;
use neat_ai_lamarck::weight_bench::objective::{
    PLATEAU_REL_TOL, count_plateau_axes, scan_sensitivities,
};
use neat_ai_lamarck::weight_bench::suite::{
    CorpusSpec, Family, Problem, ProblemSpec, load_corpus, problem_from_corpus,
};
use neat_ai_lamarck::weight_bench::{
    Arm, BenchError, BudgetedObjective, CMA_SIGMA0, Manifest, RunRow, WeightObjective,
    build_problem, cma_es, is_extrapolation, lamarck_refine, predeclared_manifest, run_arm,
    run_manifest, select_influential,
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

/// Two inputs, three TANH hiddens and one IDENTITY output. `h2` reaches the
/// output through a zero weight, so its two input synapses (indices 2 and 3)
/// cannot move the loss at all: both have sensitivity exactly zero, a tie.
fn sensitivity_fixture() -> (neat_core::CreatureExport, Vec<f32>) {
    let json = r#"{"semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
      "neurons":[
        {"type":"hidden","uuid":"h0","bias":0.0,"squash":"TANH"},
        {"type":"hidden","uuid":"h1","bias":0.0,"squash":"TANH"},
        {"type":"hidden","uuid":"h2","bias":0.0,"squash":"TANH"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"h0","weight":0.5},
        {"fromUUID":"input-1","toUUID":"h0","weight":-0.3},
        {"fromUUID":"input-0","toUUID":"h2","weight":0.4},
        {"fromUUID":"input-1","toUUID":"h2","weight":0.2},
        {"fromUUID":"input-0","toUUID":"h1","weight":0.5},
        {"fromUUID":"input-1","toUUID":"h1","weight":-0.3},
        {"fromUUID":"h0","toUUID":"o1","weight":0.7},
        {"fromUUID":"h1","toUUID":"o1","weight":0.7},
        {"fromUUID":"h2","toUUID":"o1","weight":0.0}]}"#;
    let creature: neat_core::CreatureExport = serde_json::from_str(json).unwrap();
    // Per record: two inputs, then the target.
    let records = vec![
        0.9, -0.4, 1.5, //
        -0.7, 0.8, -1.2, //
        0.3, 0.6, 0.9, //
        -0.5, -0.9, 0.4, //
    ];
    (creature, records)
}

/// Sensitivity of each synapse, measured the way `select_influential`
/// documents it, so the expected choice is derived independently.
fn sensitivities(creature: &neat_core::CreatureExport, records: &[f32]) -> Vec<f64> {
    let all: Vec<usize> = (0..creature.synapses.len()).collect();
    let o = WeightObjective::new(creature.clone(), all, records).unwrap();
    let x0 = o.start();
    let base = o.loss(&x0).unwrap();
    (0..x0.len())
        .map(|i| {
            let mut x = x0.clone();
            x[i] += neat_ai_lamarck::candidates::RANDOM_DELTA_HALF_WIDTH;
            (o.loss(&x).unwrap() - base).abs()
        })
        .collect()
}

#[test]
fn select_influential_picks_the_most_sensitive_synapses() {
    let (creature, records) = sensitivity_fixture();
    let s = sensitivities(&creature, &records);
    // The fixture's premises: h2's input synapses are dead and every other
    // synapse moves the loss.
    assert_eq!(s[2], 0.0);
    assert_eq!(s[3], 0.0);
    for i in [0, 1, 4, 5, 6, 7, 8] {
        assert!(s[i] > 0.0, "synapse {i} has sensitivity {}", s[i]);
    }

    // Every live synapse, and no dead one, when d is the live count.
    let (idx, evals) = select_influential(&creature, &records, 7).unwrap();
    assert_eq!(idx, vec![0, 1, 4, 5, 6, 7, 8]);
    assert_eq!(evals, 10);

    // For each d, the choice is exactly the d largest sensitivities with
    // ties broken towards the lower index.
    let mut order: Vec<usize> = (0..s.len()).collect();
    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));
    for d in 1..=s.len() {
        let mut expected = order[..d].to_vec();
        expected.sort_unstable();
        let (idx, _) = select_influential(&creature, &records, d).unwrap();
        assert_eq!(idx, expected, "d = {d}");
        if d < 7 {
            assert!(!idx.contains(&2) && !idx.contains(&3), "d = {d}: {idx:?}");
        }
    }

    // Tie-break: with room for one dead synapse, the tied pair (2, 3)
    // keeps the lower index.
    let (idx, _) = select_influential(&creature, &records, 8).unwrap();
    assert_eq!(idx, vec![0, 1, 2, 4, 5, 6, 7, 8]);
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
    let results = run_manifest(&corpus_manifest(corpus_spec(dir.path(), 128, 96))).unwrap();
    assert_eq!(
        results.coverage.planned_runs,
        results.coverage.completed_runs
    );
    assert_eq!(results.coverage.failed_runs, 0);
    assert!(results.runs.iter().all(|r| r.family == Family::Corpus));
    assert!(results.runs.iter().all(|r| r.evaluations_used <= r.budget));
    assert_eq!(results.runs.len(), 2);
    // Wiring check: each row starts from the creature's own weights on the
    // window's first 128 screening and last 96 held-out records.
    let spec = corpus_spec(dir.path(), 128, 96);
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

/// Each row's confirmation losses are taken at the best point its arm found:
/// replaying the arm deterministically gives that point, and the row's final
/// screening, full and held-out losses are that point's losses.
#[test]
fn confirmation_losses_are_measured_at_the_arms_best_point() {
    let mut m = tiny_manifest();
    m.budget_multiples = vec![30];
    let results = run_manifest(&m).unwrap();
    assert_eq!(results.runs.len(), 2);
    for row in &results.runs {
        assert!(row.error.is_none(), "{:?}", row.error);
        let (p, o) = problem(row.family, row.seed);
        assert_eq!(p.synapse_indices.len(), row.dimension);
        let start = o.start();
        let mut replay = BudgetedObjective::new(&o, row.budget);
        run_arm(&mut replay, row.arm, row.seed).unwrap();
        assert_eq!(replay.used(), row.evaluations_used);
        let (best_loss, best_x) = replay.best().unwrap();
        // The fixture moves: the best point is not the start on either
        // window, so confirming the start instead would change these columns.
        let held_start = o.loss_on(&start, &p.held_out).unwrap();
        let held_best = o.loss_on(best_x, &p.held_out).unwrap();
        let full_start = o.loss_on(&start, &p.full).unwrap();
        let full_best = o.loss_on(best_x, &p.full).unwrap();
        assert_ne!(held_best, held_start, "{:?}", row.arm);
        assert_ne!(full_best, full_start, "{:?}", row.arm);

        assert_eq!(row.final_screening_loss, Some(best_loss));
        assert_eq!(best_loss, o.loss(best_x).unwrap());
        assert_eq!(row.full_loss_start, Some(full_start));
        assert_eq!(row.full_loss_final, Some(full_best));
        assert_eq!(row.held_out_loss_start, Some(held_start));
        assert_eq!(row.held_out_loss_final, Some(held_best));
        assert_eq!(
            row.normalised_screening_loss,
            Some(best_loss / o.loss(&start).unwrap())
        );
        assert_eq!(row.normalised_held_out_loss, Some(held_best / held_start));
    }
}

/// A corpus run names the exact model and records it used; the synthetic
/// suite carries no fingerprint and serialises without the field.
#[test]
fn corpus_runs_fingerprint_the_model_and_records() {
    let dir = tempfile::tempdir().unwrap();
    let spec = corpus_spec(dir.path(), 128, 96);
    let first = run_manifest(&corpus_manifest(spec.clone())).unwrap();
    let fp = first.corpus_fingerprint.clone().unwrap();
    assert_eq!(fp.records, 600);
    assert_eq!(fp.creature_fnv1a64.len(), 16);
    assert_ne!(fp.creature_fnv1a64, fp.records_fnv1a64);
    // Independent FNV-1a 64 of the creature file's bytes.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in std::fs::read(&spec.creature).unwrap() {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    assert_eq!(fp.creature_fnv1a64, format!("{h:016x}"));
    let text = serde_json::to_string(&first).unwrap();
    assert!(text.contains("corpusFingerprint"), "{text:.200}");
    // Same files, same fingerprint.
    let again = run_manifest(&corpus_manifest(spec.clone())).unwrap();
    assert_eq!(again.corpus_fingerprint, Some(fp.clone()));

    // One changed record value moves only the records hash.
    let data = spec.training_data.join("0.bin");
    let mut bytes = std::fs::read(&data).unwrap();
    bytes[0] ^= 0x01;
    std::fs::write(&data, &bytes).unwrap();
    let records_changed = run_manifest(&corpus_manifest(spec.clone()))
        .unwrap()
        .corpus_fingerprint
        .unwrap();
    assert_eq!(records_changed.creature_fnv1a64, fp.creature_fnv1a64);
    assert_ne!(records_changed.records_fnv1a64, fp.records_fnv1a64);
    assert_eq!(records_changed.records, 600);

    // A changed creature weight moves only the creature hash.
    let json = std::fs::read_to_string(&spec.creature).unwrap();
    let edited = json.replacen(r#""weight":0.3}"#, r#""weight":0.31}"#, 1);
    assert_ne!(edited, json);
    std::fs::write(&spec.creature, edited).unwrap();
    let creature_changed = run_manifest(&corpus_manifest(spec))
        .unwrap()
        .corpus_fingerprint
        .unwrap();
    assert_ne!(creature_changed.creature_fnv1a64, fp.creature_fnv1a64);
    assert_eq!(
        creature_changed.records_fnv1a64,
        records_changed.records_fnv1a64
    );

    // Fewer records loaded (a smaller window) changes the count and hash.
    let dir2 = tempfile::tempdir().unwrap();
    let small = CorpusSpec {
        max_records: 300,
        ..corpus_spec(dir2.path(), 128, 96)
    };
    let fp_small = run_manifest(&corpus_manifest(small))
        .unwrap()
        .corpus_fingerprint
        .unwrap();
    assert_eq!(fp_small.records, 300);
    assert_ne!(fp_small.records_fnv1a64, fp.records_fnv1a64);
    assert_eq!(fp_small.creature_fnv1a64, fp.creature_fnv1a64);

    let synthetic = run_manifest(&tiny_manifest()).unwrap();
    assert_eq!(synthetic.corpus_fingerprint, None);
    let text = serde_json::to_string(&synthetic).unwrap();
    assert!(!text.contains("corpusFingerprint"));
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
    // Boundary: 350 + 250 needs exactly the 600 records on offer.
    let spec = corpus_spec(dir.path(), 350, 250);
    let (creature, packed) = load_corpus(&spec).unwrap();
    assert_eq!(packed.len(), 600 * 9);
    assert!(problem_from_corpus(&creature, &packed, &spec, 20, 7).is_ok());
    match load_corpus(&corpus_spec(dir.path(), 351, 250)).unwrap_err() {
        BenchError::InvalidConfig(m) => assert!(m.contains("601"), "{m}"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
    // problem_from_corpus refuses a window too short for the split on its own.
    let short = CorpusSpec {
        screening_records: 351,
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
    // Unequal counts, so reading the wrong one for either slice changes a result.
    let spec = corpus_spec(dir.path(), 128, 96);
    let (creature, packed) = load_corpus(&spec).unwrap();
    assert_eq!(packed.len(), 600 * 9);
    let p = problem_from_corpus(&creature, &packed, &spec, 20, 7).unwrap();
    assert_eq!(p.screening[..], packed[..128 * 9]);
    assert_eq!(p.held_out[..], packed[(600 - 96) * 9..]);
    assert_eq!(p.full.len(), (600 - 96) * 9);
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

/// The plateau count is the number of chosen axes whose one-sided step moves
/// the loss by at most `PLATEAU_REL_TOL` times the base loss. In the fixture,
/// synapses 2 and 3 (the dead `h2` inputs) are flat and the rest are not.
#[test]
fn plateau_free_axes_count_only_the_flat_synapses_chosen() {
    let (creature, records) = sensitivity_fixture();
    let (base, changes, evaluations) = scan_sensitivities(&creature, &records).unwrap();
    assert_eq!(evaluations, 10);
    assert_eq!(changes, sensitivities(&creature, &records));
    assert_eq!(changes[2], 0.0);
    assert_eq!(changes[3], 0.0);
    assert!(changes[0] > PLATEAU_REL_TOL * base, "premise: live axis");

    // All nine synapses chosen: the two flat ones are counted.
    let all: Vec<usize> = (0..9).collect();
    assert_eq!(count_plateau_axes(&all, &changes, base), 2);
    // Only the live synapses chosen: none is flat.
    assert_eq!(
        count_plateau_axes(&[0, 1, 4, 5, 6, 7, 8], &changes, base),
        0
    );
    // A flat synapse and a live one: one flat axis.
    assert_eq!(count_plateau_axes(&[2, 0], &changes, base), 1);
}

/// A corpus creature whose hidden neuron `h0` reaches the output through a
/// zero weight, so its eight input synapses are flat (36 synapses in all).
/// Choosing all 36 records eight plateau axes on every row of the problem;
/// choosing 20 (all of them live) records none.
#[test]
fn corpus_rows_record_the_plateau_count_of_their_problem() {
    let dir = tempfile::tempdir().unwrap();
    let spec = corpus_spec(dir.path(), 128, 96);
    let json = std::fs::read_to_string(&spec.creature).unwrap();
    let dead = json.replacen(
        r#""fromUUID":"h0","toUUID":"o1","weight":0.3}"#,
        r#""fromUUID":"h0","toUUID":"o1","weight":0}"#,
        1,
    );
    assert_ne!(dead, json);
    std::fs::write(&spec.creature, dead).unwrap();

    for (dimension, expected) in [(20, 0), (36, 8)] {
        let m = Manifest {
            dimensions: vec![dimension],
            ..corpus_manifest(spec.clone())
        };
        let results = run_manifest(&m).unwrap();
        assert_eq!(results.runs.len(), 2);
        for row in &results.runs {
            assert_eq!(row.plateau_free_axes, Some(expected), "D={dimension}");
        }
        let (creature, packed) = load_corpus(&spec).unwrap();
        let p = problem_from_corpus(&creature, &packed, &spec, dimension, 1001).unwrap();
        assert_eq!(p.plateau_free_axes, expected, "D={dimension}");
    }
}

/// A synthetic problem's rows carry its plateau count (`Some`, not absent),
/// and the cell reports that count for its one validation seed.
#[test]
fn synthetic_rows_carry_their_problem_plateau_count() {
    let (p, _) = problem(Family::Tanh, 1001);
    let r = run_manifest(&tiny_manifest()).unwrap();
    assert_eq!(r.runs.len(), 2);
    for row in &r.runs {
        assert_eq!(row.plateau_free_axes, Some(p.plateau_free_axes));
    }
    assert_eq!(
        r.cells[0].plateau_free_axes,
        vec![Some(p.plateau_free_axes)]
    );
}

/// Rows written before the field existed have no `plateauFreeAxes` key and
/// load as `None`.
#[test]
fn rows_without_the_plateau_field_load_as_none() {
    let r = run_manifest(&tiny_manifest()).unwrap();
    let mut value = serde_json::to_value(&r.runs[0]).unwrap();
    assert!(value.get("plateauFreeAxes").is_some());
    value.as_object_mut().unwrap().remove("plateauFreeAxes");
    let older: RunRow = serde_json::from_value(value).unwrap();
    assert_eq!(older.plateau_free_axes, None);
}

/// The refine arm routes its twin and follow-up probes through
/// `mirror_candidate` and `probe_candidate`; that must not move a single
/// committed row. This is a pin, expected green before and after the helper
/// change. It covers 8 of the 324 committed refine rows (tanh and hard-tanh,
/// D=20, 10D and 30D, two seeds); the full 648/216 reproduction was a manual
/// re-run.
#[test]
fn the_refine_arm_reproduces_its_committed_rows() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/evidence/weight-router");
    let text = std::fs::read_to_string(dir.join("runs.jsonl")).unwrap();
    let committed: Vec<RunRow> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();

    let full = predeclared_manifest();
    let m = Manifest {
        families: vec![Family::Tanh, Family::HardTanh],
        dimensions: vec![20],
        budget_multiples: vec![10, 30],
        validation_seeds: full.validation_seeds.iter().take(2).copied().collect(),
        ..full
    };
    assert!(m.validate().is_ok());

    let results = run_manifest(&m).unwrap();
    let mut compared = 0;
    for row in results.runs.iter().filter(|r| r.arm == Arm::LamarckRefine) {
        let expected = committed
            .iter()
            .find(|c| {
                c.arm == row.arm
                    && c.family == row.family
                    && c.dimension == row.dimension
                    && c.budget_multiple == row.budget_multiple
                    && c.seed == row.seed
            })
            .expect("a committed row for every refine run");
        assert_eq!(row, expected);
        compared += 1;
    }
    assert!(compared > 0);
    assert_eq!(
        compared,
        m.families.len() * m.dimensions.len() * m.budget_multiples.len() * m.validation_seeds.len()
    );
}

/// One IDENTITY output fed by one synapse `input-0 -> o1` at `weight`, scored
/// on a single record (input 1, target 200000): the loss falls as the weight
/// rises, so a step towards the hard weight limit (100000) always improves.
fn one_weight(weight: f64) -> WeightObjective {
    let json = format!(
        r#"{{
          "semanticVersion": "4.0.0",
          "forwardOnly": true,
          "input": 1,
          "output": 1,
          "neurons": [{{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}}],
          "synapses": [{{"fromUUID":"input-0","toUUID":"o1","weight":{weight:?}}}]
        }}"#
    );
    let creature = neat_core::parse_creature_json(&json).unwrap();
    WeightObjective::new(creature, vec![0], &[1.0, 200_000.0]).unwrap()
}

/// A twin `mirror_candidate` refuses is not evaluated: the round spends one
/// evaluation instead of two. At the hard weight limit (100000), seed 4 draws a
/// negative delta, so `+delta` lowers the weight (worse) and the twin would
/// land past the limit. Had the twin been evaluated it would have won (the loss
/// falls as the weight rises) and the budget of 3 would be spent.
#[test]
fn the_refine_arm_skips_a_refused_twin() {
    let o = one_weight(100_000.0);
    let mut b = BudgetedObjective::new(&o, 3);
    lamarck_refine(&mut b, 4).unwrap();
    assert_eq!(
        b.used(),
        2,
        "seed 4 must draw a negative delta, so the twin lands past the limit"
    );
    assert_eq!(b.best().unwrap().1, &[100_000.0]);

    // Control: away from the limit the same seed evaluates both twins and
    // accepts the upward one.
    let o = one_weight(5.0);
    let mut b = BudgetedObjective::new(&o, 3);
    lamarck_refine(&mut b, 4).unwrap();
    assert_eq!(b.used(), 3);
    assert!(b.best().unwrap().1[0] > 5.0);
}

/// A follow-up probe `probe_candidate` refuses is not evaluated. A budget of 6
/// covers the start plus one round: plus, twin and three probes (+0.5, -0.5
/// and +1.0 of the accepted move `m`). The delta draw does not depend on the
/// weight, so both runs below make the same move `m`. Seed 1 accepts an upward move; from
/// 99999.9 every probe stays inside the limit and the best is `w + 2m`, but
/// from 99999.95 the `+1.0` probe would pass 100000, so it is dropped, one
/// evaluation is left unspent and the best is the `+0.5` probe, `w + 1.5m`.
#[test]
fn the_refine_arm_skips_a_refused_follow_up_probe() {
    let control_w = 99_999.9;
    let o = one_weight(control_w);
    let mut b = BudgetedObjective::new(&o, 6);
    lamarck_refine(&mut b, 1).unwrap();
    assert_eq!(b.used(), 6);
    let m = (b.best().unwrap().1[0] - control_w) / 2.0;
    assert!(m > 0.0);

    let w = 99_999.95;
    assert!(w + 1.5 * m <= 100_000.0 && w + 2.0 * m > 100_000.0);
    let o = one_weight(w);
    let mut b = BudgetedObjective::new(&o, 6);
    lamarck_refine(&mut b, 1).unwrap();
    assert_eq!(b.used(), 5);
    let best = b.best().unwrap().1[0];
    assert!((best - (w + 1.5 * m)).abs() < 1e-6, "best {best}");
}

/// An `IF` output fed by `input-0` as its condition (0.7) and, from `source`,
/// as its positive branch (-0.4), plus a negative branch from a hidden neuron.
/// With `source` = `input-0` the pair `input-0 -> o1` repeats on two synapses
/// (different roles); otherwise every pair is unique. All four synapses are free.
fn if_creature(source: &str) -> WeightObjective {
    let json = format!(
        r#"{{
          "semanticVersion": "4.0.0",
          "forwardOnly": true,
          "input": 2,
          "output": 1,
          "neurons": [
            {{"type":"hidden","uuid":"neg","bias":0.0,"squash":"IDENTITY"}},
            {{"type":"output","uuid":"o1","bias":0.0,"squash":"IF"}}
          ],
          "synapses": [
            {{"fromUUID":"input-0","toUUID":"o1","weight":0.7,"type":"condition"}},
            {{"fromUUID":"{source}","toUUID":"o1","weight":-0.4,"type":"positive"}},
            {{"fromUUID":"input-0","toUUID":"neg","weight":1.5}},
            {{"fromUUID":"neg","toUUID":"o1","weight":0.9,"type":"negative"}}
          ]
        }}"#
    );
    let creature = neat_core::parse_creature_json(&json).unwrap();
    let records = [
        1.0, 0.5, 2.0, -1.0, 0.25, 0.5, 0.3, -0.7, 1.0, 1.5, 0.2, 0.9,
    ];
    WeightObjective::new(creature, vec![0, 1, 2, 3], &records).unwrap()
}

/// The production follow-up helpers address a weight by `(from, to)` and take
/// the first match, so a creature repeating a pair would make the refine arm
/// take steps the protocol does not describe. It must refuse before any
/// evaluation.
#[test]
fn the_refine_arm_refuses_a_creature_with_a_repeated_synapse_pair() {
    let o = if_creature("input-0");
    let mut b = BudgetedObjective::new(&o, 60);
    match run_arm(&mut b, Arm::LamarckRefine, 1001) {
        Err(BenchError::InvalidConfig(msg)) => {
            assert!(msg.contains("input-0 -> o1"), "message was: {msg}");
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
    assert_eq!(b.used(), 0, "refusal must come before any evaluation");

    // Control: the same objective is valid and evaluable under another arm.
    let mut b = BudgetedObjective::new(&o, 40);
    run_arm(&mut b, Arm::CmaEs, 1001).unwrap();
    assert!(b.used() > 0);

    // Control: with unique pairs the refine arm runs.
    let unique = if_creature("input-1");
    let mut b = BudgetedObjective::new(&unique, 60);
    run_arm(&mut b, Arm::LamarckRefine, 1001).unwrap();
    assert!(b.used() > 0);
}
