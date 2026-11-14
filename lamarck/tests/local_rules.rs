//! Tests for the opt-in sparse symbolic local-rules harness (issue #268).
//! Public API only; every test is deterministic.

use std::collections::BTreeMap;

use neat_ai_lamarck::local_rules::{
    ActivationFamily, Feature, Manifest, NEAT_CORE_TAG, RuleSet, SparseRule, Task, TaskGroup,
    TaskSpec, Verdict, apply_rule_steps, build_task, derivative_free, local_residual,
    matched_count, mse, run_manifest, sparse_backprop, verdict,
};

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
