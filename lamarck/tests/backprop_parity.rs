//! Golden parity fixtures for issue #2 (TS-behavioural backprop via neat-core).
//!
//! Regenerate expected.json files:
//! ```text
//! LAMARCK_REGEN_BACKPROP_FIXTURES=1 cargo test -p neat_ai_lamarck --test backprop_parity
//! ```
//! Optional Deno regenerator against sibling NEAT-AI:
//! `deno run -A scripts/generate_backprop_parity_fixtures.ts`

use neat_ai_lamarck::{
    BackpropConfig, accumulate_creature_learning, apply_learnings,
    backprop::{FLOAT_ABS_TOL, FLOAT_REL_TOL, calculate_learning_rate, nearly_equal},
};
use neat_core::{compile_creature, parse_creature_json};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FixtureMeta {
    seed: u64,
    max_records: u64,
    config: FixtureConfig,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FixtureConfig {
    sparse_ratio: f64,
    generations: f64,
    learning_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExpectedFixture {
    bias_counts: Vec<f64>,
    bias_totals: Vec<f64>,
    weight_counts: Vec<f64>,
    proposed_biases: Vec<f64>,
    proposed_weights: Vec<f64>,
    applied_biases: Vec<f64>,
    applied_weights: Vec<f64>,
}

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backprop")
}

fn load_and_run(dir: &Path) -> (ExpectedFixture, neat_core::CreatureExport) {
    let creature_json = fs::read_to_string(dir.join("creature.json")).unwrap();
    let creature = parse_creature_json(&creature_json).unwrap();
    let meta: FixtureMeta =
        serde_json::from_str(&fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let cfg = BackpropConfig {
        sparse_ratio: meta.config.sparse_ratio,
        generations: meta.config.generations,
        learning_rate: meta.config.learning_rate,
        initial_learning_rate: meta.config.learning_rate,
        ..Default::default()
    };

    let mut network = compile_creature(&creature).unwrap();
    let mut rng = StdRng::seed_from_u64(meta.seed);
    let learning = accumulate_creature_learning(
        &creature,
        &mut network,
        dir,
        &cfg,
        Some(meta.max_records),
        &mut rng,
    )
    .unwrap();
    let lr = calculate_learning_rate(&cfg, 0, None);
    let proposed_biases: Vec<f64> = creature
        .neurons
        .iter()
        .enumerate()
        .map(|(i, n)| learning.biases[i].propose(n.bias, &cfg, lr))
        .collect();
    let proposed_weights: Vec<f64> = creature
        .synapses
        .iter()
        .enumerate()
        .map(|(i, s)| learning.weights[i].propose(s.weight, &cfg, lr))
        .collect();
    let applied = apply_learnings(&creature, &learning, &cfg, lr);
    let expected = ExpectedFixture {
        bias_counts: learning.biases.iter().map(|b| b.count).collect(),
        bias_totals: learning
            .biases
            .iter()
            .map(|b| b.total_adjusted_bias)
            .collect(),
        weight_counts: learning.weights.iter().map(|w| w.count).collect(),
        proposed_biases,
        proposed_weights,
        applied_biases: applied.neurons.iter().map(|n| n.bias).collect(),
        applied_weights: applied.synapses.iter().map(|s| s.weight).collect(),
    };
    (expected, creature)
}

fn assert_close(label: &str, a: f64, b: f64) {
    let ok = nearly_equal(a, b) || {
        let diff = (a - b).abs();
        diff <= FLOAT_ABS_TOL.max(FLOAT_REL_TOL * a.abs().max(b.abs()).max(1.0))
    };
    assert!(ok, "{label}: got {a} expected {b}");
}

fn check_fixture(name: &str) {
    let dir = fixtures_root().join(name);
    let (got, _) = load_and_run(&dir);
    let expected_path = dir.join("expected.json");
    if std::env::var_os("LAMARCK_REGEN_BACKPROP_FIXTURES").is_some() {
        let text = serde_json::to_string_pretty(&got).unwrap() + "\n";
        fs::write(&expected_path, text).unwrap();
        eprintln!("regenerated {}", expected_path.display());
        return;
    }
    let expected: ExpectedFixture =
        serde_json::from_str(&fs::read_to_string(&expected_path).unwrap()).unwrap();
    assert_eq!(got.bias_counts.len(), expected.bias_counts.len());
    for (i, (a, b)) in got
        .bias_counts
        .iter()
        .zip(&expected.bias_counts)
        .enumerate()
    {
        assert_close(&format!("{name} bias_count[{i}]"), *a, *b);
    }
    for (i, (a, b)) in got
        .bias_totals
        .iter()
        .zip(&expected.bias_totals)
        .enumerate()
    {
        assert_close(&format!("{name} bias_total[{i}]"), *a, *b);
    }
    for (i, (a, b)) in got
        .weight_counts
        .iter()
        .zip(&expected.weight_counts)
        .enumerate()
    {
        assert_close(&format!("{name} weight_count[{i}]"), *a, *b);
    }
    for (i, (a, b)) in got
        .proposed_biases
        .iter()
        .zip(&expected.proposed_biases)
        .enumerate()
    {
        assert_close(&format!("{name} proposed_bias[{i}]"), *a, *b);
    }
    for (i, (a, b)) in got
        .proposed_weights
        .iter()
        .zip(&expected.proposed_weights)
        .enumerate()
    {
        assert_close(&format!("{name} proposed_weight[{i}]"), *a, *b);
    }
    for (i, (a, b)) in got
        .applied_biases
        .iter()
        .zip(&expected.applied_biases)
        .enumerate()
    {
        assert_close(&format!("{name} applied_bias[{i}]"), *a, *b);
    }
    for (i, (a, b)) in got
        .applied_weights
        .iter()
        .zip(&expected.applied_weights)
        .enumerate()
    {
        assert_close(&format!("{name} applied_weight[{i}]"), *a, *b);
    }
}

#[test]
fn tiny_identity_chain_parity() {
    check_fixture("tiny_identity_chain");
}

#[test]
fn squash_tanh_logistic_parity() {
    check_fixture("squash_tanh_logistic");
}

#[test]
fn weight_and_bias_apply_parity() {
    check_fixture("weight_and_bias_apply");
}

#[test]
fn hidden_chain_bias_count_positive() {
    let dir = fixtures_root().join("tiny_identity_chain");
    let (got, creature) = load_and_run(&dir);
    assert_eq!(creature.neurons[0].uuid, "h1");
    assert!(
        got.bias_counts[0] > 0.0,
        "hidden h1 must receive propagated blame"
    );
    assert!(got.bias_counts[1] > 0.0, "output o1 must accumulate");
}
