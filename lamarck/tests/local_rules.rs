//! Tests for the opt-in sparse symbolic local-rules harness (issue #268).
//! Public API only; every test is deterministic.

use std::collections::BTreeMap;

use neat_ai_lamarck::local_rules::{
    ActivationFamily, Comparison, DEFAULT_OUT_DIR, DataSource, Feature, Manifest, NEAT_CORE_TAG,
    OverallVerdict, RuleSet, SparseRule, SuccessClaim, TASK_GENERATOR_VERSION, Task, TaskGroup,
    TaskSpec, Verdict, apply_rule_steps, build_task, derivative_free, local_residual,
    matched_count, mse, paired_verdict, predeclared_verdict, resolve_out_dir, run_files,
    run_manifest, sparse_backprop, success_claim, verdict,
};

/// The shared corpus writer: `write_sample` lays out the training-data format
/// that `open_training_scan` reads, so files-mode tests reuse it.
#[path = "../examples/support/mod.rs"]
mod support;

fn task(seed: u64) -> Task {
    let spec = TaskSpec {
        seed,
        hidden_width: 4,
        hidden_squash: "TANH".to_string(),
    };
    build_task(spec, 3, 64, 64).expect("task builds")
}

fn rule(terms: Vec<(Feature, f64)>) -> SparseRule {
    SparseRule::new(terms, 3).expect("rule is valid")
}

fn only(family: ActivationFamily, rule: SparseRule) -> RuleSet {
    BTreeMap::from([(family, rule)])
}

#[test]
fn delta_rule_lowers_training_mse() {
    let t = task(1);
    let before = mse(&t.parent, &t.train).expect("parent mse");
    let rules = only(ActivationFamily::Identity, rule(vec![(Feature::XR, 1.0)]));
    let (after, cost) =
        apply_rule_steps(&t.parent, &t.train, &rules, 10, 0.05, 1.0).expect("rule steps run");
    let now = mse(&after, &t.train).expect("refined mse");
    assert!(
        now < before,
        "XR*1 should lower training MSE: {before} -> {now}"
    );
    assert_eq!(cost.record_units, 2 * 10 * 64);
}

#[test]
fn residual_only_rule_leaves_hidden_synapses_and_biases_alone() {
    let t = task(2);
    let families = [
        ActivationFamily::Identity,
        ActivationFamily::Tanh,
        ActivationFamily::Logistic,
        ActivationFamily::Relu,
        ActivationFamily::Other,
    ];
    let rules: RuleSet = families
        .into_iter()
        .map(|f| (f, rule(vec![(Feature::R, 1.0)])))
        .collect();
    let (after, _) =
        apply_rule_steps(&t.parent, &t.train, &rules, 5, 0.05, 1.0).expect("rule steps run");
    for (p, a) in t.parent.synapses.iter().zip(&after.synapses) {
        if p.from_uuid.starts_with("input-") {
            assert_eq!(
                p.weight, a.weight,
                "input->hidden weight {} moved",
                p.from_uuid
            );
        }
    }
    for (p, a) in t.parent.neurons.iter().zip(&after.neurons) {
        if p.neuron_type == "hidden" {
            assert_eq!(p.bias, a.bias, "hidden bias {} moved", p.uuid);
        }
    }
    let output_changed = t
        .parent
        .synapses
        .iter()
        .zip(&after.synapses)
        .filter(|(p, _)| !p.from_uuid.starts_with("input-"))
        .any(|(p, a)| p.weight != a.weight);
    assert!(
        output_changed,
        "hidden->output weights should change under R*1"
    );
}

#[test]
fn parent_is_unchanged_by_every_arm() {
    let t = task(3);
    let before = serde_json::to_string(&t.parent).expect("parent serialises");
    let rules = only(ActivationFamily::Identity, rule(vec![(Feature::XR, 1.0)]));
    apply_rule_steps(&t.parent, &t.train, &rules, 5, 0.05, 1.0).expect("symbolic arm runs");
    sparse_backprop(&t.parent, &t.train, 3, 0.01, 0.5, 3).expect("sparse backprop runs");
    derivative_free(&t.parent, &t.train, 20, 0.05, 3).expect("derivative-free runs");
    let after = serde_json::to_string(&t.parent).expect("parent serialises");
    assert_eq!(before, after, "the parent was mutated");
}

#[test]
fn sparse_rule_refusals_name_their_cause() {
    assert_eq!(
        SparseRule::new(vec![], 3).unwrap_err(),
        "sparse rule has no terms"
    );
    assert_eq!(
        SparseRule::new(
            vec![
                (Feature::X, 1.0),
                (Feature::Y, 1.0),
                (Feature::W, 1.0),
                (Feature::R, 1.0)
            ],
            3
        )
        .unwrap_err(),
        "sparse rule has 4 terms, above max_support 3"
    );
    assert_eq!(
        SparseRule::new(vec![(Feature::XR, 1.0), (Feature::XR, 2.0)], 3).unwrap_err(),
        "duplicate feature XR in sparse rule"
    );
    assert_eq!(
        SparseRule::new(vec![(Feature::XR, f64::NAN)], 3).unwrap_err(),
        "non-finite coefficient for XR"
    );
    assert!(SparseRule::new(vec![(Feature::XR, 1.0), (Feature::WYY, -0.5)], 3).is_ok());
}

fn refusal(edit: impl FnOnce(&mut Manifest)) -> String {
    let mut manifest = Manifest::predeclared();
    edit(&mut manifest);
    manifest.validate().unwrap_err()
}

#[test]
fn manifest_refusals_name_their_cause() {
    Manifest::predeclared()
        .validate()
        .expect("predeclared protocol validates");
    assert_eq!(
        refusal(|m| m.steps = vec![4, 10, 20]),
        "step count 4 outside 5..=20"
    );
    assert_eq!(
        refusal(|m| m.steps = vec![5, 10, 21]),
        "step count 21 outside 5..=20"
    );
    assert_eq!(
        refusal(|m| m.validation_groups[0].seeds[0] = 1),
        "validation seed 1 overlaps a calibration seed"
    );
    assert_eq!(
        refusal(|m| m.sparse_ratio = 0.0),
        "sparse_ratio must be in (0, 1]"
    );
}

#[test]
fn derivative_free_never_raises_training_mse() {
    let t = task(4);
    let before = mse(&t.parent, &t.train).expect("parent mse");
    let (after, cost) = derivative_free(&t.parent, &t.train, 40, 0.05, 9).expect("search runs");
    let now = mse(&after, &t.train).expect("result mse");
    assert!(now <= before, "training MSE rose: {before} -> {now}");
    assert_eq!(cost.record_units, 40 * 64);
}

#[test]
fn matched_count_arithmetic() {
    assert_eq!(matched_count(1000, 2, 64), 7);
    assert_eq!(matched_count(100, 2, 64), 0);
}

#[test]
fn verdict_from_confidence_interval() {
    assert_eq!(verdict((-2.0, -1.0)), Verdict::Better);
    assert_eq!(verdict((1.0, 2.0)), Verdict::Worse);
    assert_eq!(verdict((-1.0, 1.0)), Verdict::Inconclusive);
}

#[test]
fn local_residual_is_output_only() {
    assert_eq!(local_residual(true, 0, 0.25, &[1.0]), Some(0.75));
    assert_eq!(local_residual(false, 0, 0.25, &[1.0]), None);
}

fn reduced_manifest() -> Manifest {
    let mut m = Manifest::predeclared();
    m.calibration_seeds = vec![1, 2];
    m.validation_groups = vec![
        TaskGroup {
            group: "same-width".to_string(),
            seeds: vec![101],
            hidden_width: 4,
        },
        TaskGroup {
            group: "new-width".to_string(),
            seeds: vec![201],
            hidden_width: 2,
        },
    ];
    m.steps = vec![5];
    m.meta_rounds = 1;
    m.candidates_per_round = 4;
    m.bootstrap_resamples = 200;
    m.min_paired_tasks = 2;
    m
}

#[test]
fn reduced_manifest_is_reproducible_and_fully_covered() {
    let first = run_manifest(&reduced_manifest()).expect("reduced run");
    let second = run_manifest(&reduced_manifest()).expect("reduced run again");
    let a = first.summary_json().expect("summary serialises");
    let b = second.summary_json().expect("summary serialises");
    assert_eq!(a, b, "summary JSON differs between identical runs");
    assert_eq!(first.coverage.completed, first.coverage.planned);
    assert_eq!(first.coverage.failed, 0);
    assert!(
        first.failures.is_empty(),
        "unexpected failures: {:?}",
        first.failures
    );
}

#[test]
fn neat_core_tag_matches_cargo_manifest() {
    let cargo = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .expect("Cargo.toml readable");
    let line = cargo
        .lines()
        .find(|l| l.starts_with("neat-core"))
        .expect("neat-core dependency line");
    let tag = line
        .split("tag = \"")
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .expect("tag attribute");
    assert_eq!(NEAT_CORE_TAG, tag);
}

/// Reduced files-mode protocol: small record counts keep the corpus tiny.
fn reduced_files_manifest() -> Manifest {
    let mut m = reduced_manifest();
    m.train_records = 16;
    m.held_out_records = 16;
    m
}

/// Records a files run reads: calibration training, validation training, held-out.
fn files_records(manifest: &Manifest) -> usize {
    2 * manifest.train_records + manifest.held_out_records
}

#[test]
fn files_run_completes_and_names_its_source() {
    let dir = tempfile::tempdir().expect("temp dir");
    let manifest = reduced_files_manifest();
    support::write_sample(dir.path(), files_records(&manifest), manifest.inputs);
    let calibration = task(1).parent;
    let validation = [task(2).parent, task(3).parent];
    let report =
        run_files(&manifest, &calibration, &validation, dir.path()).expect("files run completes");
    assert_eq!(report.coverage.completed, report.coverage.planned);
    assert_eq!(report.coverage.failed, 0);
    assert!(
        report.comparisons.iter().any(|c| c.group == "files"),
        "no comparison for group files"
    );
    let summary = report.summary_json().expect("summary serialises");
    let json: serde_json::Value = serde_json::from_str(&summary).expect("summary parses");
    assert_eq!(json["dataSource"]["kind"], "files");
    assert!(json["dataSource"]["corpusKey"].is_string());
    assert!(json["dataSource"]["calibrationCreature"].is_string());
    assert_eq!(
        json["dataSource"]["validationCreatures"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    let temp = dir.path().display().to_string();
    assert!(
        !summary.contains(&temp),
        "summary leaks the temp dir path {temp}"
    );
}

#[test]
fn files_run_refuses_a_short_corpus_and_accepts_a_full_one() {
    let dir = tempfile::tempdir().expect("temp dir");
    let manifest = reduced_files_manifest();
    let calibration = task(1).parent;
    let validation = [task(2).parent, task(3).parent];
    support::write_sample(dir.path(), files_records(&manifest) - 1, manifest.inputs);
    let err = run_files(&manifest, &calibration, &validation, dir.path())
        .expect_err("a short corpus is refused");
    assert_eq!(
        err,
        "training data holds 47 records but the protocol needs 48: 16 calibration, 16 validation training and 16 held-out"
    );
    support::write_sample(dir.path(), files_records(&manifest), manifest.inputs);
    run_files(&manifest, &calibration, &validation, dir.path())
        .expect("the same call succeeds with enough records");
}

#[test]
fn files_run_refuses_the_calibration_creature_as_validation() {
    let dir = tempfile::tempdir().expect("temp dir");
    let manifest = reduced_files_manifest();
    let calibration = task(1).parent;
    let validation = [calibration.clone()];
    let err = run_files(&manifest, &calibration, &validation, dir.path())
        .expect_err("transfer to the calibration creature is refused");
    assert_eq!(
        err,
        "validation creature 0 is the calibration creature (same fingerprint); validation must use a different creature"
    );
}

#[test]
fn files_run_refuses_no_validation_creatures_and_mismatched_widths() {
    let dir = tempfile::tempdir().expect("temp dir");
    let manifest = reduced_files_manifest();
    let calibration = task(1).parent;
    let err = run_files(&manifest, &calibration, &[], dir.path())
        .expect_err("an empty validation list is refused");
    assert_eq!(
        err,
        "files run has 0 validation creatures, below the predeclared minimum of 2 paired tasks"
    );
    let spec = TaskSpec {
        seed: 2,
        hidden_width: 4,
        hidden_squash: "TANH".to_string(),
    };
    let narrow = build_task(spec, 2, 64, 64)
        .expect("two-input task builds")
        .parent;
    let err = run_files(&manifest, &calibration, &[narrow], dir.path())
        .expect_err("a width mismatch is refused");
    assert_eq!(
        err,
        "validation creature 0 has input/output 2/1, but the calibration creature has 3/1"
    );
}

#[test]
fn synthetic_summary_carries_the_search_trace_and_source() {
    let report = run_manifest(&reduced_manifest()).expect("reduced run");
    let summary = report.summary_json().expect("summary serialises");
    let json: serde_json::Value = serde_json::from_str(&summary).expect("summary parses");
    let trace = json["searchTrace"]
        .as_array()
        .expect("searchTrace is an array");
    assert!(!trace.is_empty(), "searchTrace is empty");
    assert_eq!(json["dataSource"]["kind"], "synthetic");
    assert_eq!(json["dataSource"]["generator"], TASK_GENERATOR_VERSION);
}

const VERDICTS: [OverallVerdict; 3] = [
    OverallVerdict::EarnsItsPlace,
    OverallVerdict::Negative,
    OverallVerdict::Inconclusive,
];

#[test]
fn synthetic_source_never_supports_a_success_claim() {
    let synthetic = DataSource::Synthetic {
        generator: TASK_GENERATOR_VERSION.to_string(),
    };
    for v in VERDICTS {
        assert_eq!(
            success_claim(&synthetic, v),
            SuccessClaim::NotClaimableSynthetic,
            "synthetic source with verdict {v:?}"
        );
    }
    let files = DataSource::Files {
        corpus_key: "corpus".to_string(),
        calibration_creature: "aa".to_string(),
        validation_creatures: vec!["bb".to_string()],
    };
    assert_eq!(
        success_claim(&files, OverallVerdict::EarnsItsPlace),
        SuccessClaim::PendingOwnerReview
    );
    assert_eq!(
        success_claim(&files, OverallVerdict::Negative),
        SuccessClaim::NoSuccess
    );
    assert_eq!(
        success_claim(&files, OverallVerdict::Inconclusive),
        SuccessClaim::NoSuccess
    );
}

#[test]
fn synthetic_report_is_not_claimable_as_a_success() {
    let report = run_manifest(&reduced_manifest()).expect("reduced run");
    assert_eq!(report.success_claim, SuccessClaim::NotClaimableSynthetic);
    let summary = report.summary_json().expect("summary serialises");
    assert!(
        summary.contains(r#""successClaim": "not-claimable-synthetic""#),
        "summary lacks the synthetic success claim"
    );
}

#[test]
fn files_report_claim_follows_its_source_and_verdict() {
    let dir = tempfile::tempdir().expect("temp dir");
    let manifest = reduced_files_manifest();
    support::write_sample(dir.path(), files_records(&manifest), manifest.inputs);
    let calibration = task(1).parent;
    let validation = [task(2).parent, task(3).parent];
    let report =
        run_files(&manifest, &calibration, &validation, dir.path()).expect("files run completes");
    assert_eq!(
        report.success_claim,
        success_claim(&report.data_source, report.overall_verdict)
    );
    assert_ne!(report.success_claim, SuccessClaim::NotClaimableSynthetic);
}

#[test]
fn out_dir_defaults_for_synthetic_mode_only() {
    assert_eq!(
        resolve_out_dir(false, None).expect("synthetic default"),
        DEFAULT_OUT_DIR
    );
    let err = resolve_out_dir(true, None).expect_err("files mode without OUT_DIR is refused");
    assert!(
        err.starts_with("files mode needs an explicit OUT_DIR") && err.contains(DEFAULT_OUT_DIR),
        "unhelpful refusal: {err}"
    );
    assert_eq!(
        resolve_out_dir(true, Some("private")).expect("explicit files OUT_DIR"),
        "private"
    );
    assert_eq!(
        resolve_out_dir(false, Some("elsewhere")).expect("explicit synthetic OUT_DIR"),
        "elsewhere"
    );
}

#[test]
fn a_single_paired_task_cannot_decide_a_comparison() {
    let one_sided = (-1.0, -1.0);
    assert_eq!(verdict(one_sided), Verdict::Better);
    assert_eq!(paired_verdict(one_sided, 1, 2, 0), Verdict::Inconclusive);
    assert_eq!(paired_verdict(one_sided, 2, 2, 0), Verdict::Better);
    assert_eq!(paired_verdict((1.0, 1.0), 1, 2, 0), Verdict::Inconclusive);
    assert_eq!(paired_verdict((1.0, 1.0), 2, 2, 0), Verdict::Worse);
}

#[test]
fn a_lost_run_makes_its_comparison_inconclusive() {
    assert_eq!(paired_verdict((-2.0, -1.0), 10, 2, 0), Verdict::Better);
    assert_eq!(
        paired_verdict((-2.0, -1.0), 10, 2, 1),
        Verdict::Inconclusive
    );
    assert_eq!(paired_verdict((1.0, 2.0), 10, 2, 1), Verdict::Inconclusive);
}

#[test]
fn manifest_predeclares_the_minimum_paired_sample() {
    assert_eq!(Manifest::predeclared().min_paired_tasks, 8);
    assert_eq!(
        refusal(|m| m.min_paired_tasks = 1),
        "min_paired_tasks must be at least 2"
    );
    assert_eq!(
        refusal(|m| m.min_paired_tasks = 25),
        "24 validation tasks, below min_paired_tasks 25"
    );
    let criterion = Manifest::predeclared().success_criterion;
    assert!(
        criterion.contains("min_paired_tasks") && criterion.contains("any run behind it failed")
    );
}

#[test]
fn files_run_refuses_fewer_creatures_than_the_minimum_paired_sample() {
    let dir = tempfile::tempdir().expect("temp dir");
    let manifest = reduced_files_manifest();
    support::write_sample(dir.path(), files_records(&manifest), manifest.inputs);
    let calibration = task(1).parent;
    let err = run_files(&manifest, &calibration, &[task(2).parent], dir.path())
        .expect_err("one validation creature is below the minimum of two");
    assert_eq!(
        err,
        "files run has 1 validation creatures, below the predeclared minimum of 2 paired tasks"
    );
    let report = run_files(
        &manifest,
        &calibration,
        &[task(2).parent, task(3).parent],
        dir.path(),
    )
    .expect("a run at the minimum completes");
    assert!(
        report
            .comparisons
            .iter()
            .filter(|c| c.group == "files")
            .all(|c| c.n == 2 && c.missing == 0)
    );
}

fn comparison(arm: &str, k: u32, group: &str, verdict: Verdict) -> Comparison {
    Comparison {
        arm: arm.to_string(),
        k,
        group: group.to_string(),
        primary: !arm.ends_with("app-only"),
        n: 24,
        missing: 0,
        mean_difference: 0.0,
        ci_low: 0.0,
        ci_high: 0.0,
        verdict,
        paired_differences: Vec::new(),
    }
}

/// `all`-group comparisons for K = 5, 10, 20 given each primary baseline's verdicts.
fn all_group(backprop: [Verdict; 3], free: [Verdict; 3]) -> Vec<Comparison> {
    let steps = [5, 10, 20];
    let mut comparisons = Vec::new();
    for (i, k) in steps.into_iter().enumerate() {
        comparisons.push(comparison("sparse-backprop", k, "all", backprop[i]));
        comparisons.push(comparison("derivative-free", k, "all", free[i]));
    }
    comparisons
}

#[test]
fn predeclared_verdict_follows_the_success_criterion() {
    use Verdict::{Better as B, Inconclusive as I, Worse as W};
    let steps = [5, 10, 20];
    let verdict_of = |backprop, free| predeclared_verdict(&all_group(backprop, free), &steps);
    assert_eq!(
        verdict_of([I, B, I], [I, B, I]),
        OverallVerdict::EarnsItsPlace
    );
    assert_eq!(
        verdict_of([B, B, B], [B, I, B]),
        OverallVerdict::EarnsItsPlace
    );
    assert_eq!(verdict_of([W, W, W], [W, W, W]), OverallVerdict::Negative);
    assert_eq!(verdict_of([W, W, W], [I, I, I]), OverallVerdict::Negative);
    assert_eq!(
        verdict_of([I, I, I], [I, I, I]),
        OverallVerdict::Inconclusive
    );
    // Better than one baseline only is not enough.
    assert_eq!(
        verdict_of([B, B, B], [I, I, I]),
        OverallVerdict::Inconclusive
    );
    // Better than both at one K, but Worse than one at another K.
    assert_eq!(
        verdict_of([B, I, W], [B, I, I]),
        OverallVerdict::Inconclusive
    );
    assert_eq!(
        verdict_of([B, I, I], [B, W, I]),
        OverallVerdict::Inconclusive
    );
    // Worse than sparse backprop at only some K is not negative.
    assert_eq!(
        verdict_of([W, W, I], [W, W, W]),
        OverallVerdict::Inconclusive
    );
}

#[test]
fn predeclared_verdict_ignores_other_groups_and_app_only_arms() {
    use Verdict::{Better as B, Inconclusive as I};
    let steps = [5];
    let mut comparisons = vec![
        comparison("sparse-backprop", 5, "all", I),
        comparison("derivative-free", 5, "all", I),
        comparison("sparse-backprop", 5, "same-width", B),
        comparison("derivative-free", 5, "same-width", B),
        comparison("sparse-backprop-app-only", 5, "all", B),
        comparison("derivative-free-app-only", 5, "all", B),
    ];
    assert_eq!(
        predeclared_verdict(&comparisons, &steps),
        OverallVerdict::Inconclusive
    );
    comparisons[0].verdict = B;
    comparisons[1].verdict = B;
    assert_eq!(
        predeclared_verdict(&comparisons, &steps),
        OverallVerdict::EarnsItsPlace
    );
}

#[test]
fn baselines_get_the_symbolic_budget_plus_the_meta_share() {
    let manifest = reduced_manifest();
    let report = run_manifest(&manifest).expect("reduced run");
    let n = manifest.train_records;
    let unit = n as u64;
    let k = u64::from(manifest.steps[0]);
    let symbolic_units = 2 * k * unit;
    // 2 squashes x 2 validation seeds.
    let validation_tasks = 4u64;

    // Meta cost: every candidate scores each calibration task with
    // `calibration_steps` rule steps at 2 units plus two MSE passes at 1 unit.
    let families: std::collections::BTreeSet<_> =
        report.search_trace.iter().map(|s| s.family).collect();
    let candidates = u64::from(manifest.meta_rounds)
        * manifest.candidates_per_round as u64
        * families.len() as u64;
    assert_eq!(report.search_trace.len() as u64, candidates);
    let calibration_tasks =
        (manifest.calibration_seeds.len() * manifest.hidden_squashes.len()) as u64;
    let per_task = 2 * u64::from(manifest.calibration_steps) * unit + 2 * unit;
    assert_eq!(
        report.meta_cost.record_units,
        candidates * calibration_tasks * per_task
    );
    let share = report.meta_cost.record_units.div_ceil(validation_tasks);
    assert!(share > 0);

    let rows = |arm: &str| {
        let rows: Vec<_> = report.runs.iter().filter(|r| r.arm == arm).collect();
        assert_eq!(rows.len() as u64, validation_tasks, "rows for {arm}");
        rows
    };
    for row in rows("symbolic") {
        assert_eq!((row.steps, row.record_units), (k, symbolic_units));
    }
    for row in rows("sparse-backprop") {
        let steps = (symbolic_units + share) / (2 * unit);
        assert_eq!((row.steps, row.record_units), (steps, steps * 2 * unit));
    }
    for row in rows("derivative-free") {
        let evals = (symbolic_units + share) / unit;
        assert_eq!((row.steps, row.record_units), (evals, evals * unit));
    }
    for row in rows("sparse-backprop-app-only") {
        let steps = symbolic_units / (2 * unit);
        assert_eq!((row.steps, row.record_units), (steps, steps * 2 * unit));
    }
    for row in rows("derivative-free-app-only") {
        let evals = symbolic_units / unit;
        assert_eq!((row.steps, row.record_units), (evals, evals * unit));
    }
    for c in &report.comparisons {
        assert_eq!((c.missing, c.n > 0), (0, true), "{} {}", c.arm, c.group);
    }
}

#[test]
fn a_failed_validation_run_is_recorded_and_never_decides_the_verdict() {
    let dir = tempfile::tempdir().expect("temp dir");
    let manifest = reduced_files_manifest();
    support::write_sample(dir.path(), files_records(&manifest), manifest.inputs);
    let calibration = task(1).parent;
    let mut broken = task(3).parent;
    broken.synapses[0].to_uuid = "nowhere".to_string();
    let report = run_files(
        &manifest,
        &calibration,
        &[task(2).parent, broken, task(4).parent],
        dir.path(),
    )
    .expect("a run with one failing creature still reports");
    let arms = 5 * manifest.steps.len() as u64;
    assert_eq!(report.coverage.planned, 3 * arms);
    assert_eq!(
        report.coverage.failed, arms,
        "every run of the broken creature fails"
    );
    assert_eq!(report.coverage.failed, report.failures.len() as u64);
    assert_eq!(
        report.coverage.completed + report.coverage.failed,
        report.coverage.planned
    );
    assert!(
        report
            .failures
            .iter()
            .all(|f| f.task == "files:creature-1" && f.reason.contains("nowhere")),
        "unexpected failures: {:?}",
        report.failures
    );
    // Two sound creatures meet `min_paired_tasks` (2), so the sample alone
    // would not make a comparison inconclusive: only the failed-run rule does.
    assert_eq!(manifest.min_paired_tasks, 2);
    let primary: Vec<_> = report.comparisons.iter().filter(|c| c.primary).collect();
    assert!(!primary.is_empty());
    for c in primary {
        assert_eq!(
            (c.n, c.missing),
            (2, 1),
            "{} at K {}: the broken creature is the one missing pair",
            c.arm,
            c.k
        );
        assert_eq!(c.verdict, Verdict::Inconclusive, "{} at K {}", c.arm, c.k);
    }
    assert_ne!(report.overall_verdict, OverallVerdict::EarnsItsPlace);
}
