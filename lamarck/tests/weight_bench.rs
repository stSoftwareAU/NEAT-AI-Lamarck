//! Behaviour tests for the offline weight bench (issue #266).

use neat_ai_lamarck::weight_bench::experiment::ALLOWED_BUDGET_MULTIPLES;
use neat_ai_lamarck::weight_bench::probes::LandscapeProbe;
use neat_ai_lamarck::weight_bench::suite::{Family, Problem, ProblemSpec};
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
