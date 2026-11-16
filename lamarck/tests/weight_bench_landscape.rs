//! Plateau, kink and activation-family tests for the weight bench (issue
//! #266): the saturated hard-tanh family and the two-sided landscape
//! diagnosis.

use neat_ai_lamarck::weight_bench::landscape::{
    AxisShape, CURVATURE_REL_TOL, KINK_RATIO_REL_TOL, LandscapeCell, LandscapeRow, ShapeCounts,
    axis_shapes, classify_axis, diagnose, diagnose_manifest, summarise_landscape,
};
use neat_ai_lamarck::weight_bench::suite::{
    CorpusSpec, Family, LIVE_HIDDEN_UNITS, Problem, ProblemSpec, SATURATION_BIAS,
};
use neat_ai_lamarck::weight_bench::{BenchError, Manifest, RunRow, build_problem, run_manifest};
use serde_json::Value;

const H: f64 = 0.05;

fn build(family: Family, dimension: usize, seed: u64) -> Problem {
    build_problem(&ProblemSpec {
        family,
        dimension,
        seed,
    })
    .unwrap()
}

fn manifest(families: Vec<Family>, dimensions: Vec<usize>, seeds: Vec<u64>) -> Manifest {
    Manifest {
        families,
        dimensions,
        budget_multiples: vec![10],
        validation_seeds: seeds,
        bootstrap_resamples: 200,
        bootstrap_seed: 266,
        corpus: None,
    }
}

/// Classify `f` around `x` with the diagnosis steps.
fn shape_of(f: impl Fn(f64) -> f64, x: f64) -> AxisShape {
    classify_axis(f(x), f(x + H), f(x - H), f(x + H / 2.0), f(x - H / 2.0))
}

#[test]
fn classify_axis_on_hand_values() {
    // Quadratic: D2(h) / D2(h/2) is exactly 4.
    assert_eq!(shape_of(|x| 1.0 + x * x, 0.3), AxisShape::Smooth);
    assert_eq!(
        shape_of(|x| 2.0 + (x - 1.0).powi(2), 0.0),
        AxisShape::Smooth
    );
    // |x|-shaped kink at the start: the ratio is 2.
    assert_eq!(shape_of(|x| 1.0 + x.abs(), 0.0), AxisShape::Kinked);
    // A kink inside the half step but off-centre is still caught.
    assert_eq!(shape_of(|x| 1.0 + (x - 0.01).abs(), 0.0), AxisShape::Kinked);
    // A slope jump between h/2 and h: D2(h/2) is zero.
    assert_eq!(
        shape_of(|x| 1.0 + x + (x - 0.04).max(0.0), 0.0),
        AxisShape::Kinked
    );
    // Both sides flat; exactly one side flat (ReLU-like plateau boundary).
    assert_eq!(shape_of(|_| 1.0, 0.0), AxisShape::Flat);
    assert_eq!(shape_of(|x| 1.0 + x.max(0.0), 0.0), AxisShape::Edge);
    assert_eq!(shape_of(|x| 1.0 + (-x).max(0.0), 0.0), AxisShape::Edge);
    // Linear: neither side flat, no curvature.
    assert_eq!(shape_of(|x| 1.0 + x, 0.0), AxisShape::Indeterminate);
    // Hand values with D2(h/2) = 0.01 and D2(h) = 4 * r * 0.01: a ratio
    // inside 4 * (1 ± KINK_RATIO_REL_TOL) is smooth, outside it is kinked.
    let with_ratio = |r: f64| {
        let d2_h = 4.0 * r * 0.01;
        classify_axis(1.0, 1.1 + d2_h / 2.0, 0.9 + d2_h / 2.0, 1.055, 0.955)
    };
    let tol = KINK_RATIO_REL_TOL;
    assert_eq!(with_ratio(1.0), AxisShape::Smooth);
    assert_eq!(with_ratio(1.0 + tol - 0.01), AxisShape::Smooth);
    assert_eq!(with_ratio(1.0 - tol + 0.01), AxisShape::Smooth);
    assert_eq!(with_ratio(1.0 + tol + 0.01), AxisShape::Kinked);
    assert_eq!(with_ratio(1.0 - tol - 0.01), AxisShape::Kinked);
    // Curvature of the other sign at h/2 is kinked.
    assert_eq!(
        classify_axis(1.0, 1.15, 0.95, 1.045, 0.945),
        AxisShape::Kinked
    );
    // Curvature below the tolerance is indeterminate, well above it is not.
    let tiny = CURVATURE_REL_TOL * 1.0;
    assert_eq!(
        classify_axis(1.0, 1.1 + tiny / 2.0, 0.9, 1.05, 0.95),
        AxisShape::Indeterminate
    );
    assert_eq!(
        classify_axis(1.0, 1.1 + 400.0 * tiny, 0.9, 1.05 + 100.0 * tiny, 0.95),
        AxisShape::Smooth
    );
}

#[test]
fn shape_counts_tally_every_shape() {
    let counts = ShapeCounts::of(&[
        AxisShape::Flat,
        AxisShape::Flat,
        AxisShape::Edge,
        AxisShape::Kinked,
        AxisShape::Smooth,
        AxisShape::Indeterminate,
    ]);
    assert_eq!(
        counts,
        ShapeCounts {
            flat: 2,
            edge: 1,
            kinked: 1,
            smooth: 1,
            indeterminate: 1,
        }
    );
    assert_eq!(counts.total(), 6);
}

#[test]
fn saturated_hard_tanh_has_flat_axes_only_beyond_its_live_synapses() {
    let live = LIVE_HIDDEN_UNITS * 8 + 16;
    assert_eq!(live, 32, "2 live units x 8 inputs + 16 outputs");
    const { assert!(SATURATION_BIAS > 8.0 * 1.3 + H) };
    // D=20 fits inside the 32 live axes: nothing flat is chosen.
    assert_eq!(
        build(Family::SaturatedHardTanh, 20, 1001).plateau_free_axes,
        0
    );
    for (dimension, seed) in [(40, 1001), (100, 1001), (100, 1002)] {
        let p = build(Family::SaturatedHardTanh, dimension, seed);
        assert!(p.plateau_free_axes > 0, "D={dimension} seed={seed}");
        assert!(
            p.plateau_free_axes >= dimension - live,
            "D={dimension} seed={seed}: {} flat",
            p.plateau_free_axes
        );
    }
    // The ordinary hard-tanh family has no such plateau.
    assert_eq!(build(Family::HardTanh, 100, 1001).plateau_free_axes, 0);
}

#[test]
fn saturated_start_creature_carries_the_teacher_biases() {
    let p = build(Family::SaturatedHardTanh, 20, 1001);
    let biases: Vec<f64> = p
        .creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| n.bias)
        .collect();
    assert_eq!(biases.len(), 16);
    for (h, &b) in biases.iter().enumerate() {
        let expected = match h {
            h if h < LIVE_HIDDEN_UNITS => 0.0,
            h if h.is_multiple_of(2) => SATURATION_BIAS,
            _ => -SATURATION_BIAS,
        };
        assert_eq!(b, expected, "hidden {h}");
    }
    // The other families keep unbiased hidden units.
    let plain = build(Family::HardTanh, 20, 1001);
    assert!(
        plain
            .creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "hidden")
            .all(|n| n.bias == 0.0)
    );
}

#[test]
fn diagnosis_finds_the_saturated_plateau() {
    let p = build(Family::SaturatedHardTanh, 100, 1001);
    let row = diagnose(&p).unwrap();
    assert_eq!(row.family, Family::SaturatedHardTanh);
    assert_eq!((row.dimension, row.seed), (100, 1001));
    assert_eq!(row.shapes.total(), 100);
    assert_eq!(row.evaluations, 1 + 4 * 100);
    assert!(row.shapes.flat > 0);
    // Only 32 synapses feed live units, so at most 100 - 32 chosen axes can be
    // flat, and here the one-sided and two-sided counts agree exactly.
    assert!(row.shapes.flat <= 100 - 32);
    assert_eq!(row.shapes.flat, row.plateau_free_axes);
}

#[test]
fn tanh_axes_are_smooth_and_relu_and_hard_tanh_axes_kink() {
    for seed in [1001, 1002] {
        for dimension in [20, 40] {
            let row = diagnose(&build(Family::Tanh, dimension, seed)).unwrap();
            assert_eq!(row.shapes.kinked, 0, "tanh D={dimension} seed={seed}");
            assert_eq!(row.shapes.flat + row.shapes.edge, 0);
        }
    }
    for family in [Family::Relu, Family::HardTanh] {
        let row = diagnose(&build(family, 20, 1001)).unwrap();
        assert!(row.shapes.kinked > 0, "{family:?}: {:?}", row.shapes);
    }
}

#[test]
fn axis_shapes_are_deterministic() {
    let p = build(Family::Relu, 20, 1003);
    assert_eq!(axis_shapes(&p).unwrap(), axis_shapes(&p).unwrap());
}

#[test]
fn landscape_rows_and_cells_serialise_in_camel_case() {
    let m = manifest(vec![Family::SaturatedHardTanh], vec![20, 40], vec![1001]);
    let rows = diagnose_manifest(&m).unwrap();
    assert_eq!(rows.len(), 2);
    let value = serde_json::to_value(&rows[1]).unwrap();
    for key in [
        "family",
        "dimension",
        "seed",
        "flat",
        "edge",
        "kinked",
        "smooth",
        "indeterminate",
        "evaluations",
        "plateauFreeAxes",
    ] {
        assert!(value.get(key).is_some(), "row lacks {key}: {value}");
    }
    assert_eq!(value["family"], "saturated-hard-tanh");
    let back: LandscapeRow = serde_json::from_value(value).unwrap();
    assert_eq!(back, rows[1]);
    let cells: Vec<LandscapeCell> = summarise_landscape(&m, &rows);
    assert_eq!(cells.len(), 2);
    assert_eq!(cells[1].problems, 1);
    assert_eq!(cells[1].shapes, rows[1].shapes);
    assert_eq!(cells[1].evaluations, 1 + 4 * 40);
    assert!(cells[1].shapes.flat > 0);
}

#[test]
fn landscape_refuses_corpus_and_invalid_manifests() {
    let mut m = manifest(vec![Family::Corpus], vec![20], vec![1001]);
    m.corpus = Some(CorpusSpec {
        creature: "missing.json".into(),
        training_data: "missing".into(),
        screening_records: 1,
        held_out_records: 1,
        max_records: 2,
    });
    assert!(matches!(
        diagnose_manifest(&m),
        Err(BenchError::InvalidConfig(_))
    ));
    let empty = manifest(vec![Family::Tanh], vec![20], vec![]);
    assert!(matches!(
        diagnose_manifest(&empty),
        Err(BenchError::InvalidConfig(_))
    ));
}

#[test]
fn saturated_family_runs_through_the_manifest_with_confirmation() {
    let m = manifest(vec![Family::SaturatedHardTanh], vec![20, 40], vec![1001]);
    let results = run_manifest(&m).unwrap();
    assert_eq!(results.runs.len(), 4);
    assert_eq!(results.coverage.failed_runs, 0);
    for row in &results.runs {
        assert_eq!(row.family, Family::SaturatedHardTanh);
        assert!(row.error.is_none());
        for loss in [
            row.full_loss_start,
            row.full_loss_final,
            row.held_out_loss_start,
            row.held_out_loss_final,
        ] {
            assert!(loss.is_some_and(f64::is_finite), "{row:?}");
        }
    }
    let at_40: Vec<&RunRow> = results.runs.iter().filter(|r| r.dimension == 40).collect();
    assert!(at_40.iter().all(|r| r.plateau_free_axes == Some(8)));
    assert_eq!(results.cells.len(), 2);
}

#[test]
fn existing_families_rebuild_the_committed_rows_exactly() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/evidence/weight-router");
    let committed: Vec<RunRow> = std::fs::read_to_string(dir.join("runs.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let m = manifest(
        vec![Family::Tanh, Family::Relu, Family::HardTanh],
        vec![20],
        vec![1001],
    );
    let results = run_manifest(&m).unwrap();
    assert_eq!(results.runs.len(), 6);
    for row in &results.runs {
        let old = committed
            .iter()
            .find(|r| {
                r.family == row.family
                    && r.dimension == row.dimension
                    && r.seed == row.seed
                    && r.budget_multiple == row.budget_multiple
                    && r.arm == row.arm
            })
            .unwrap();
        assert_eq!(row, old);
    }
}

#[test]
fn committed_landscape_evidence_is_reproduced_and_shows_the_saturated_plateau() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/evidence/weight-router-landscape");
    let summary: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("landscape-summary.json")).unwrap())
            .unwrap();
    let m: Manifest = serde_json::from_value(summary["manifest"].clone()).unwrap();
    let declared: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m, declared);
    let cells: Vec<LandscapeCell> = serde_json::from_value(summary["cells"].clone()).unwrap();
    let rows: Vec<LandscapeRow> = std::fs::read_to_string(dir.join("landscape.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        rows.len(),
        m.families.len() * m.dimensions.len() * m.validation_seeds.len()
    );
    assert_eq!(summarise_landscape(&m, &rows), cells);
    assert_eq!(
        summary["totalEvaluations"],
        serde_json::json!(rows.iter().map(|r| r.evaluations).sum::<usize>())
    );
    for cell in &cells {
        let axes = cell.dimension * cell.problems;
        assert_eq!(cell.shapes.total(), axes);
        match (cell.family, cell.dimension) {
            (Family::SaturatedHardTanh, 100) => {
                assert!(cell.shapes.flat > 0);
                assert_eq!(cell.shapes.flat, cell.plateau_free_axes);
            }
            (Family::SaturatedHardTanh, _) => {}
            _ => assert_eq!(cell.shapes.flat + cell.shapes.edge, 0, "{cell:?}"),
        }
        match cell.family {
            // Smooth activation: at most a rounding-level stray kink.
            Family::Tanh => assert!(100 * cell.shapes.kinked <= axes, "{cell:?}"),
            Family::Relu | Family::HardTanh if cell.dimension == 100 => {
                assert!(10 * cell.shapes.kinked > axes, "{cell:?}")
            }
            _ => {}
        }
    }
    // The rows are what the current code computes (first seed only, to keep
    // the test cheap; the full run is reproduced by the example).
    let mut first_seed = m.clone();
    first_seed.validation_seeds.truncate(1);
    let expected: Vec<LandscapeRow> = rows
        .iter()
        .filter(|r| r.seed == m.validation_seeds[0])
        .cloned()
        .collect();
    assert_eq!(diagnose_manifest(&first_seed).unwrap(), expected);
}
