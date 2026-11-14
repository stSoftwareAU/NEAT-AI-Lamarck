//! Behaviour tests for the offline AMM reference arm (issue #266).

use std::io::Cursor;

use neat_ai_lamarck::weight_bench::suite::Family;
use neat_ai_lamarck::weight_bench::suite::ProblemSpec;
use neat_ai_lamarck::weight_bench::{
    Arm, BenchError, BudgetedObjective, Manifest, REFERENCE_BOX_HALF_WIDTH, ReferenceRequest,
    RouterRule, RunRow, WeightObjective, build_problem, run_arm, serve_reference,
    summarise_reference,
};
use serde_json::Value;

fn manifest() -> Manifest {
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

fn request(seed: u64) -> ReferenceRequest {
    ReferenceRequest {
        family: Family::Tanh,
        dimension: 20,
        seed,
        budget_multiple: 10,
    }
}

struct Served {
    lines: Vec<Value>,
    row: RunRow,
}

fn serve(m: &Manifest, req: &ReferenceRequest, input: &str) -> Result<Served, BenchError> {
    let mut out = Vec::new();
    let row = serve_reference(m, req, Cursor::new(input.to_string()), &mut out)?;
    let lines = String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    Ok(Served { lines, row })
}

fn start_of(m: &Manifest) -> Vec<f64> {
    let out = serve(m, &request(1001), "done\n").unwrap();
    serde_json::from_value(out.lines[0]["start"].clone()).unwrap()
}

fn point_line(x: &[f64]) -> String {
    format!("{}\n", serde_json::to_string(x).unwrap())
}

#[test]
fn serve_answers_each_point_and_builds_the_row() {
    let m = manifest();
    let start = start_of(&m);
    let mut nudged = start.clone();
    nudged[0] += 0.01;
    let input = format!(
        "{}{}\n{}done\n",
        point_line(&start),
        point_line(&nudged).trim_end(),
        point_line(&start)
    );
    let s = serve(&m, &request(1001), &input).unwrap();

    let header = &s.lines[0];
    assert_eq!(header["dimension"], 20);
    assert_eq!(header["budget"], 200);
    assert_eq!(header["start"].as_array().unwrap().len(), 20);
    assert_eq!(header["boxHalfWidth"], REFERENCE_BOX_HALF_WIDTH);
    assert_eq!(header["boxHalfWidth"], 0.5);

    assert_eq!(s.lines.len(), 1 + 3 + 1, "header, three losses, row");
    let losses: Vec<f64> = s.lines[1..4]
        .iter()
        .map(|l| l["loss"].as_f64().expect("a loss reply"))
        .collect();
    assert_eq!(losses[0], losses[2], "the same point has the same loss");
    assert_ne!(losses[0], losses[1], "a moved point changes the loss");

    let row: RunRow = serde_json::from_value(s.lines[4]["row"].clone()).unwrap();
    assert_eq!(row, s.row);
    assert_eq!(row.arm, Arm::AmmReference);
    assert_eq!(row.phase, "validation");
    assert_eq!(row.evaluations_used, 3);
    assert_eq!(row.budget, 200);
    assert!(row.normalised_screening_loss.is_some());
    assert!(row.normalised_held_out_loss.is_some());
    assert!(row.confirmation_evaluations > 0);
    assert_eq!(row.error, None);
}

#[test]
fn serve_header_seed_is_the_arm_seed_and_blank_lines_and_eof_are_fine() {
    let m = manifest();
    let start = start_of(&m);
    // Blank lines are skipped and EOF (no `done`) ends the run.
    let s = serve(&m, &request(1001), &format!("\n  \n{}", point_line(&start))).unwrap();
    assert_eq!(
        s.lines[0]["seed"].as_u64().unwrap(),
        neat_ai_lamarck::weight_bench::arm_seed(Arm::AmmReference, 1001)
    );
    assert_eq!(s.row.evaluations_used, 1);
    assert_eq!(s.row.error, None);
}

#[test]
fn exceeding_the_budget_is_an_error_but_using_it_all_is_not() {
    let m = manifest();
    let start = start_of(&m);
    let line = point_line(&start);

    let exact = serve(&m, &request(1001), &line.repeat(200)).unwrap();
    assert_eq!(exact.row.evaluations_used, 200);
    assert_eq!(exact.row.error, None);

    let over = serve(&m, &request(1001), &line.repeat(201)).unwrap();
    let reply = &over.lines[over.lines.len() - 2];
    assert!(
        reply["error"].as_str().unwrap().contains("budget"),
        "{reply}"
    );
    assert!(over.row.error.as_ref().unwrap().contains("budget"));
    assert_eq!(over.row.evaluations_used, 200);
    let last: RunRow = serde_json::from_value(over.lines.last().unwrap()["row"].clone()).unwrap();
    assert_eq!(last, over.row);
}

#[test]
fn malformed_points_are_recorded_not_swallowed() {
    let m = manifest();
    let start = start_of(&m);
    let good = point_line(&start);

    for bad in ["[1,2]\n", "nope\n"] {
        let s = serve(&m, &request(1001), &format!("{good}{bad}{good}")).unwrap();
        let reply = &s.lines[s.lines.len() - 2];
        assert!(reply["error"].is_string(), "{bad:?}: {reply}");
        assert!(s.row.error.is_some(), "{bad:?}");
        // The run stops at the error: the trailing good point is not served.
        assert_eq!(
            s.lines.iter().filter(|l| l.get("loss").is_some()).count(),
            1
        );
    }
    let control = serve(&m, &request(1001), &format!("{good}{good}")).unwrap();
    assert_eq!(control.row.error, None);
}

#[test]
fn only_validation_seeds_and_manifest_cells_are_served() {
    let m = manifest();
    let refused = serve(&m, &request(1), "done\n").err().unwrap();
    match refused {
        BenchError::InvalidConfig(msg) => assert!(msg.contains("seed"), "{msg}"),
        other => panic!("unexpected {other:?}"),
    }
    assert!(serve(&m, &request(1001), "done\n").is_ok());

    let mut wrong_dimension = request(1001);
    wrong_dimension.dimension = 40;
    match serve(&m, &wrong_dimension, "done\n").err().unwrap() {
        BenchError::InvalidConfig(msg) => assert!(msg.contains("dimension"), "{msg}"),
        other => panic!("unexpected {other:?}"),
    }
    let mut wrong_budget = request(1001);
    wrong_budget.budget_multiple = 30;
    match serve(&m, &wrong_budget, "done\n").err().unwrap() {
        BenchError::InvalidConfig(msg) => assert!(msg.contains("budgetMultiple"), "{msg}"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn run_arm_refuses_the_reference_arm() {
    let p = build_problem(&ProblemSpec {
        family: Family::Tanh,
        dimension: 20,
        seed: 1001,
    })
    .unwrap();
    let o =
        WeightObjective::new(p.creature.clone(), p.synapse_indices.clone(), &p.screening).unwrap();
    let rule = RouterRule {
        min_budget_multiple: None,
        max_non_smooth_share: 0.0,
        probe_axes: 8,
    };
    let mut b = BudgetedObjective::new(&o, 50);
    match run_arm(&mut b, Arm::AmmReference, &rule, 1001) {
        Err(BenchError::InvalidConfig(msg)) => assert!(msg.contains("offline"), "{msg}"),
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(b.used(), 0);
    assert!(run_arm(&mut b, Arm::LamarckRefine, &rule, 1001).is_ok());
}

// ---- summarise_reference -------------------------------------------------

const SEEDS: [u64; 6] = [1001, 1002, 1003, 1004, 1005, 1006];

fn many_seeds() -> Manifest {
    let mut m = manifest();
    m.validation_seeds = SEEDS.to_vec();
    m
}

fn fixture(arm: Arm, seed: u64, normalised: f64) -> RunRow {
    RunRow {
        family: Family::Tanh,
        dimension: 20,
        budget_multiple: 10,
        budget: 200,
        seed,
        phase: "validation".into(),
        arm,
        routed_to: None,
        start_screening_loss: Some(1.0),
        final_screening_loss: Some(normalised),
        normalised_screening_loss: Some(normalised),
        evaluations_used: 200,
        probe_evaluations: 0,
        preparation_evaluations: 21,
        full_loss_start: Some(1.0),
        full_loss_final: Some(normalised),
        held_out_loss_start: Some(1.0),
        held_out_loss_final: Some(normalised),
        normalised_held_out_loss: Some(normalised),
        confirmation_evaluations: 4,
        extrapolation: false,
        error: None,
    }
}

/// Refine, CMA-ES and router rows at about 0.8, with seed-to-seed variation.
fn in_process_rows() -> Vec<RunRow> {
    let mut rows = Vec::new();
    for (i, &seed) in SEEDS.iter().enumerate() {
        for arm in [Arm::LamarckRefine, Arm::CmaEs, Arm::Router] {
            rows.push(fixture(arm, seed, 0.8 + 0.01 * i as f64));
        }
    }
    rows
}

fn reference_rows(offset: f64) -> Vec<RunRow> {
    SEEDS
        .iter()
        .enumerate()
        .map(|(i, &seed)| fixture(Arm::AmmReference, seed, 0.8 + 0.01 * i as f64 + offset))
        .collect()
}

#[test]
fn clearly_better_and_worse_references_get_matching_verdicts() {
    let m = many_seeds();
    let runs = in_process_rows();

    let better = summarise_reference(&m, &runs, &reference_rows(-0.3)).unwrap();
    assert_eq!(better.format_version, 1);
    assert_eq!(better.cells.len(), 1);
    let comparisons = &better.cells[0].comparisons;
    let pairs: Vec<(Arm, Arm)> = comparisons.iter().map(|c| (c.first, c.second)).collect();
    assert_eq!(
        pairs,
        vec![
            (Arm::AmmReference, Arm::LamarckRefine),
            (Arm::AmmReference, Arm::CmaEs),
            (Arm::AmmReference, Arm::Router),
        ]
    );
    for c in comparisons {
        assert_eq!(c.verdict, "amm-reference-better");
        assert_eq!(c.screening.wins, 6);
        assert!((c.screening.mean.unwrap() + 0.3).abs() < 1e-9);
    }

    let worse = summarise_reference(&m, &runs, &reference_rows(0.3)).unwrap();
    for c in &worse.cells[0].comparisons {
        assert_eq!(c.verdict, "amm-reference-worse");
        assert_eq!(c.screening.losses, 6);
    }

    let same = summarise_reference(&m, &runs, &reference_rows(0.0)).unwrap();
    for c in &same.cells[0].comparisons {
        assert_eq!(c.verdict, "inconclusive");
        assert_eq!(c.screening.ties, 6);
    }
}

#[test]
fn coverage_counts_failed_rows_and_missing_runs() {
    let m = many_seeds();
    let runs = in_process_rows();
    let mut reference = reference_rows(-0.3);
    reference.pop(); // a planned run with no row
    reference[0].error = Some("evaluation budget of 200 exhausted".into());
    reference[0].normalised_screening_loss = None;
    reference[0].normalised_held_out_loss = None;

    let r = summarise_reference(&m, &runs, &reference).unwrap();
    assert_eq!(r.coverage.planned_runs, 6);
    assert_eq!(r.coverage.completed_runs, 4);
    assert_eq!(r.coverage.failed_runs, 1);
    assert_eq!(r.total_evaluations, 5 * (200 + 21 + 4));
    // The failed seed and the missing seed are dropped from the pairing.
    let c = &r.cells[0].comparisons[0];
    assert_eq!(c.screening.paired_differences.len(), 4);
    assert_eq!(c.screening.failures, 2);
}

#[test]
fn failed_reference_with_completed_metrics_is_excluded_from_pairs() {
    let m = many_seeds();
    let runs = in_process_rows();
    let mut reference = reference_rows(-0.3);
    let control = summarise_reference(&m, &runs, &reference).unwrap();
    assert_eq!(control.cells[0].comparisons[0].screening.wins, 6);
    // The driver can receive a complete server row, then record an AMM error.
    reference[0].error = Some("amm: RuntimeError: reference failed after evaluation".into());
    let result = summarise_reference(&m, &runs, &reference).unwrap();
    assert_eq!(result.coverage.failed_runs, 1);
    assert_eq!(result.total_evaluations, control.total_evaluations);
    for comparison in &result.cells[0].comparisons {
        assert_eq!(comparison.screening.wins, 5);
        assert_eq!(comparison.screening.failures, 1);
        assert_eq!(comparison.held_out.wins, 5);
        assert_eq!(comparison.held_out.failures, 1);
    }
}

#[test]
fn malformed_reference_sets_are_refused() {
    let m = many_seeds();
    let runs = in_process_rows();
    assert!(summarise_reference(&m, &runs, &reference_rows(-0.3)).is_ok());

    let mut wrong_arm = reference_rows(-0.3);
    wrong_arm[2].arm = Arm::CmaEs;
    assert!(matches!(
        summarise_reference(&m, &runs, &wrong_arm),
        Err(BenchError::InvalidConfig(_))
    ));

    let mut duplicate = reference_rows(-0.3);
    duplicate.push(duplicate[1].clone());
    match summarise_reference(&m, &runs, &duplicate) {
        Err(BenchError::InvalidConfig(msg)) => assert!(msg.contains("duplicate"), "{msg}"),
        other => panic!("unexpected {other:?}"),
    }

    let mut calibration = reference_rows(-0.3);
    calibration[0].phase = "calibration".into();
    assert!(matches!(
        summarise_reference(&m, &runs, &calibration),
        Err(BenchError::InvalidConfig(_))
    ));

    let mut stray = reference_rows(-0.3);
    stray[0].seed = 999;
    assert!(matches!(
        summarise_reference(&m, &runs, &stray),
        Err(BenchError::InvalidConfig(_))
    ));
}

#[test]
fn hand_built_error_row_with_null_fields_counts_as_failed() {
    let m = many_seeds();
    let runs = in_process_rows();
    let mut reference = reference_rows(-0.3);
    let mut failed = fixture(Arm::AmmReference, SEEDS[0], 0.0);
    failed.start_screening_loss = None;
    failed.final_screening_loss = None;
    failed.normalised_screening_loss = None;
    failed.evaluations_used = 0;
    failed.preparation_evaluations = 0;
    failed.full_loss_start = None;
    failed.full_loss_final = None;
    failed.held_out_loss_start = None;
    failed.held_out_loss_final = None;
    failed.normalised_held_out_loss = None;
    failed.confirmation_evaluations = 0;
    failed.error = Some("server gave no row".into());
    // It must survive the JSON round trip the script and the example use.
    let text = serde_json::to_string(&failed).unwrap();
    let failed: RunRow = serde_json::from_str(&text).unwrap();
    assert!(failed.routed_to.is_none());
    reference[0] = failed;

    let r = summarise_reference(&m, &runs, &reference).unwrap();
    assert_eq!(r.coverage.completed_runs, 5);
    assert_eq!(r.coverage.failed_runs, 1);
    assert_eq!(r.total_evaluations, 5 * (200 + 21 + 4));
    let c = &r.cells[0].comparisons[0];
    assert_eq!(c.screening.failures, 1);
    assert_eq!(c.held_out.failures, 1);
    assert_eq!(c.screening.paired_differences.len(), 5);
}
