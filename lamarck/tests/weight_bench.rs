//! Behaviour tests for the offline weight bench (issue #266).

use neat_ai_lamarck::weight_bench::experiment::ALLOWED_BUDGET_MULTIPLES;
use neat_ai_lamarck::weight_bench::probes::LandscapeProbe;
use neat_ai_lamarck::weight_bench::suite::{
    CorpusRole, CorpusSpec, Family, Problem, ProblemSpec, load_corpus, problem_from_corpus,
};
use neat_ai_lamarck::weight_bench::{
    Arm, BenchError, BudgetedObjective, CMA_SIGMA0, CalibrationRow, Manifest, ROUTER_PROBE_AXES,
    RouterRule, WeightObjective, build_problem, calibrate, cma_es, is_extrapolation,
    lamarck_refine, probe_landscape, route, run_manifest, select_influential,
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

fn probe(plateau: f64, kink: f64) -> LandscapeProbe {
    LandscapeProbe {
        axes: 8,
        plateau_share: plateau,
        kink_share: kink,
        evaluations: 17,
    }
}

fn rule(multiple: Option<f64>, share: f64) -> RouterRule {
    RouterRule {
        min_budget_multiple: multiple,
        max_non_smooth_share: share,
        probe_axes: 8,
    }
}

fn tiny_manifest() -> Manifest {
    Manifest {
        families: vec![Family::Tanh],
        dimensions: vec![20],
        budget_multiples: vec![10],
        calibration_seeds: vec![1],
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
fn cma_es_improves_at_large_budget() {
    let (_, o) = problem(Family::Tanh, 3);
    let start = o.loss(&o.start()).unwrap();
    let best = best_of(&o, 100 * o.dimension(), 11, true);
    assert!(best / start < 1.0, "normalised {}", best / start);
}

#[test]
fn probe_counts_evaluations_against_the_budget() {
    let (_, o) = problem(Family::Relu, 1);
    let mut b = BudgetedObjective::new(&o, 100);
    let p = probe_landscape(&mut b, ROUTER_PROBE_AXES).unwrap();
    assert_eq!(p.axes, 8);
    assert_eq!(p.evaluations, 17);
    assert_eq!(b.used(), 17);
    assert!(p.plateau_share + p.kink_share <= 1.0);
    // A budget too small for the probe fails loudly.
    let mut tiny = BudgetedObjective::new(&o, 5);
    assert!(matches!(
        probe_landscape(&mut tiny, 8),
        Err(BenchError::BudgetExhausted { limit: 5 })
    ));
}

#[test]
fn route_follows_the_rule() {
    let r = rule(Some(30.0), 0.5);
    assert_eq!(route(&r, 20, 600, &probe(0.0, 0.25)), Arm::CmaEs);
    assert_eq!(route(&r, 20, 599, &probe(0.0, 0.25)), Arm::LamarckRefine);
    assert_eq!(route(&r, 20, 6000, &probe(0.5, 0.25)), Arm::LamarckRefine);
    assert_eq!(
        route(&rule(None, 1.01), 20, 100_000, &probe(0.0, 0.0)),
        Arm::LamarckRefine
    );
}

fn row(budget: usize, kink: f64, refine: f64, cma: f64) -> CalibrationRow {
    CalibrationRow {
        dimension: 20,
        budget,
        probe: probe(0.0, kink),
        refine_normalised: refine,
        cmaes_normalised: cma,
    }
}

#[test]
fn calibrate_picks_the_favoured_rule() {
    // CMA-ES wins only at the 100x budget on smooth probes.
    let rows = vec![
        row(200, 0.0, 0.5, 0.9),
        row(600, 0.0, 0.5, 0.9),
        row(2000, 0.0, 0.9, 0.1),
        row(2000, 0.0, 0.9, 0.1),
        row(2000, 1.0, 0.1, 0.9),
    ];
    let r = calibrate(&rows);
    assert_eq!(r.min_budget_multiple, Some(100.0));
    assert!(r.max_non_smooth_share < 1.0);
    // Refinement always wins: never route to CMA-ES.
    let never: Vec<_> = rows.iter().map(|r| row(r.budget, 0.0, 0.1, 0.9)).collect();
    assert_eq!(calibrate(&never).min_budget_multiple, None);
}

#[test]
fn manifest_validation() {
    assert!(tiny_manifest().validate().is_ok());
    let mut m = tiny_manifest();
    m.budget_multiples = vec![50];
    let err = m.validate().unwrap_err().to_string();
    assert!(err.contains("50"), "{err}");
    let mut m = tiny_manifest();
    m.validation_seeds = vec![1];
    assert!(matches!(m.validate(), Err(BenchError::InvalidConfig(_))));
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
        assert_eq!(cell.comparisons.len(), 3);
        for c in &cell.comparisons {
            assert_eq!(c.screening.paired_differences.len(), 1);
            assert_eq!(c.held_out.paired_differences.len(), 1);
        }
    }
    for row in &r.runs {
        assert!(row.evaluations_used <= row.budget, "{row:?}");
        assert!(!row.extrapolation);
        if row.phase == "validation" && row.arm == Arm::Router {
            assert_eq!(row.probe_evaluations, 2 * 8 + 1);
            assert!(row.routed_to.is_some());
        }
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
    let routers: Vec<_> = results
        .runs
        .iter()
        .filter(|r| r.phase == "validation" && r.arm == Arm::Router)
        .collect();
    assert_eq!(routers.len(), 1);
    assert_eq!(routers[0].probe_evaluations, 17);
    // Wiring check: both phases start from the creature's own weights, so the
    // starting losses differ only if the phases read different records.
    let multiple = results.runs[0].budget_multiple;
    let pick = |phase: &str| {
        results
            .runs
            .iter()
            .find(|r| {
                r.phase == phase && r.arm == Arm::LamarckRefine && r.budget_multiple == multiple
            })
            .unwrap()
    };
    let (cal, val) = (pick("calibration"), pick("validation"));
    assert!(cal.held_out_loss_start.is_some() && val.held_out_loss_start.is_some());
    assert_ne!(cal.held_out_loss_start, val.held_out_loss_start);
    assert_ne!(cal.start_screening_loss, val.start_screening_loss);
}

#[test]
fn corpus_with_too_few_records_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    // Positive control: the same files are accepted with a feasible split.
    assert!(run_manifest(&corpus_manifest(corpus_spec(dir.path(), 128, 128))).is_ok());
    let err = run_manifest(&corpus_manifest(corpus_spec(dir.path(), 400, 400))).unwrap_err();
    match err {
        BenchError::InvalidConfig(m) => assert!(m.contains("1600"), "{m}"),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
    // Boundary: 150 + 150 needs exactly the 600 records on offer.
    let (creature, packed) = load_corpus(&corpus_spec(dir.path(), 150, 150)).unwrap();
    assert_eq!(packed.len(), 600 * 9);
    for role in [CorpusRole::Calibration, CorpusRole::Validation] {
        let spec = corpus_spec(dir.path(), 150, 150);
        assert!(problem_from_corpus(&creature, &packed, &spec, role, 20, 7).is_ok());
    }
    match load_corpus(&corpus_spec(dir.path(), 151, 150)).unwrap_err() {
        BenchError::InvalidConfig(m) => assert!(m.contains("602"), "{m}"),
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
    let cal =
        problem_from_corpus(&creature, &packed, &spec, CorpusRole::Calibration, 20, 7).unwrap();
    let val =
        problem_from_corpus(&creature, &packed, &spec, CorpusRole::Validation, 20, 7).unwrap();
    assert_eq!(cal.screening[..], packed[..128 * 9]);
    assert_eq!(cal.held_out[..], packed[(300 - 128) * 9..300 * 9]);
    assert_eq!(cal.full.len(), (300 - 128) * 9);
    assert_eq!(val.screening[..], packed[300 * 9..(300 + 128) * 9]);
    assert_eq!(val.held_out[..], packed[(600 - 128) * 9..]);
    assert_eq!(val.full.len(), (300 - 128) * 9);
    // The halves are disjoint and together cover each role's own records only.
    let cal_all = [&cal.full[..], &cal.held_out[..]].concat();
    let val_all = [&val.full[..], &val.held_out[..]].concat();
    assert_eq!(cal_all[..], packed[..300 * 9]);
    assert_eq!(val_all[..], packed[300 * 9..]);
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
    // One calibration problem and one validation problem, each built once.
    let prep = |seed| {
        build_problem(&ProblemSpec {
            family: Family::Tanh,
            dimension: 20,
            seed,
        })
        .unwrap()
        .preparation_evaluations
    };
    let preparation = prep(1) + prep(1001);
    assert!(preparation > 0);
    // Two calibration rows and three validation rows, five confirmation
    // evaluations each (screening start; full start and best; held-out start
    // and best).
    assert_eq!(r.runs.len(), 5);
    assert!(r.runs.iter().all(|row| row.error.is_none()));
    for row in &r.runs {
        assert_eq!(row.confirmation_evaluations, 5);
        assert_eq!(row.preparation_evaluations, prep(row.seed));
    }
    let used: usize = r.runs.iter().map(|row| row.evaluations_used).sum();
    let probe = 2 * ROUTER_PROBE_AXES.min(20) + 1;
    assert_eq!(r.calibration_probe_evaluations, probe);
    assert_eq!(r.total_evaluations, used + 5 * 5 + preparation + probe);
}
