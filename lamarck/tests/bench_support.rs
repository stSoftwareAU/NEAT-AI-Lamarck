//! Contract tests for the shared bench fixture (issue #138).
//!
//! The example benches are only comparable when every one of them builds the
//! *same* synthetic creature over the *same* deterministic corpus. Since the
//! fixture now lives in one place, these tests pin what that one place emits:
//! a drift in the creature shape or in the xorshift stream fails here rather
//! than silently invalidating the numbers in `docs/followup-economics.md`,
//! `docs/baseline-reuse.md` and the README's thread-scaling table.

use std::collections::BTreeMap;
use std::path::Path;

use neat_ai_lamarck::scorer::{DirectoryScorer, ScoreResult, ScoreSample};
use neat_core::parse_creature_json;

#[path = "../examples/support/mod.rs"]
mod support;

use support::{LocalMseScorer, creature_json, write_sample};

/// The corpus writer, re-implemented here from the documented rule so the test
/// fails if the shared writer's stream ever changes.
fn expected_corpus_bytes(records: usize, inputs: usize) -> Vec<u8> {
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
    bytes
}

#[test]
fn creature_json_builds_the_documented_shape() {
    let creature = parse_creature_json(&creature_json(16, 3)).expect("fixture parses");

    assert_eq!(creature.input, 16);
    assert_eq!(creature.output, 1);
    assert_eq!(creature.semantic_version.as_deref(), Some("4.0.0"));
    assert!(creature.forward_only);

    // Three TANH hiddens plus the single IDENTITY output `o1`.
    assert_eq!(creature.neurons.len(), 4);
    for h in 0..3 {
        let neuron = &creature.neurons[h];
        assert_eq!(neuron.uuid, format!("h{h}"));
        assert_eq!(neuron.neuron_type, "hidden");
        assert_eq!(neuron.squash.as_deref(), Some("TANH"));
        assert!((neuron.bias - ((h as f64 % 7.0) * 0.01 - 0.03)).abs() < 1e-12);
    }
    let output = &creature.neurons[3];
    assert_eq!(output.uuid, "o1");
    assert_eq!(output.neuron_type, "output");
    assert_eq!(output.squash.as_deref(), Some("IDENTITY"));

    // Each hidden reads a deterministic four-input fan-in slice and feeds `o1`.
    assert_eq!(creature.synapses.len(), 3 * 5);
    for h in 0..3usize {
        for k in 0..4usize {
            let synapse = &creature.synapses[h * 5 + k];
            assert_eq!(synapse.from_uuid, format!("input-{}", (h * 4 + k) % 16));
            assert_eq!(synapse.to_uuid, format!("h{h}"));
        }
        let to_output = &creature.synapses[h * 5 + 4];
        assert_eq!(to_output.from_uuid, format!("h{h}"));
        assert_eq!(to_output.to_uuid, "o1");
    }
}

#[test]
fn creature_json_wraps_the_fan_in_when_inputs_are_scarce() {
    // Edge case: fewer inputs than the fan-in slice needs, so indices wrap.
    let creature = parse_creature_json(&creature_json(2, 2)).expect("fixture parses");
    assert_eq!(creature.input, 2);
    for synapse in creature.synapses.iter().filter(|s| s.to_uuid != "o1") {
        let index: usize = synapse
            .from_uuid
            .strip_prefix("input-")
            .expect("input source")
            .parse()
            .expect("numeric input index");
        assert!(index < 2, "fan-in index {index} escaped the input width");
    }
}

#[test]
fn creature_json_is_deterministic() {
    assert_eq!(creature_json(32, 5), creature_json(32, 5));
}

#[test]
fn write_sample_emits_the_deterministic_xorshift_corpus() {
    let dir = tempfile::tempdir().unwrap();
    write_sample(dir.path(), 7, 5);

    let written = std::fs::read(dir.path().join("0.bin")).expect("corpus written to 0.bin");
    // `inputs + 1` little-endian f32s per record: the inputs plus the target.
    assert_eq!(written.len(), 7 * (5 + 1) * 4);
    assert_eq!(written, expected_corpus_bytes(7, 5));
}

#[test]
fn write_sample_is_reproducible_across_directories() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    write_sample(first.path(), 4, 3);
    write_sample(second.path(), 4, 3);

    assert_eq!(
        std::fs::read(first.path().join("0.bin")).unwrap(),
        std::fs::read(second.path().join("0.bin")).unwrap()
    );
}

#[test]
fn write_sample_handles_an_empty_corpus() {
    let dir = tempfile::tempdir().unwrap();
    write_sample(dir.path(), 0, 4);
    assert_eq!(std::fs::read(dir.path().join("0.bin")).unwrap().len(), 0);
}

#[test]
fn local_mse_scorer_scores_every_candidate_in_the_directory() {
    let dir = tempfile::tempdir().unwrap();
    let training = dir.path().join("data");
    std::fs::create_dir_all(&training).unwrap();
    write_sample(&training, 64, 8);

    let candidates = dir.path().join("candidates");
    std::fs::create_dir_all(&candidates).unwrap();
    std::fs::write(candidates.join("a.json"), creature_json(8, 3)).unwrap();
    std::fs::write(candidates.join("b.json"), creature_json(8, 4)).unwrap();
    // Non-JSON entries are skipped rather than failing the batch.
    std::fs::write(candidates.join("notes.txt"), "ignored").unwrap();

    let scores: BTreeMap<String, ScoreResult> = LocalMseScorer
        .score_directory_sampled(&candidates, &training, ScoreSample::full())
        .expect("scoring succeeds");

    assert_eq!(scores.keys().collect::<Vec<_>>(), vec!["a", "b"]);
    for result in scores.values() {
        assert!(result.error.is_finite(), "MSE must be finite");
        assert!((result.score - (1.0 - result.error)).abs() < 1e-12);
        assert_eq!(result.complexity_penalty, 0.0);
    }
}

#[test]
fn local_mse_scorer_reports_an_unparseable_candidate() {
    let dir = tempfile::tempdir().unwrap();
    let training = dir.path().join("data");
    std::fs::create_dir_all(&training).unwrap();
    write_sample(&training, 8, 4);

    let candidates = dir.path().join("candidates");
    std::fs::create_dir_all(&candidates).unwrap();
    std::fs::write(candidates.join("broken.json"), "{ not json").unwrap();

    let err = LocalMseScorer
        .score_directory_sampled(&candidates, &training, ScoreSample::full())
        .expect_err("a malformed candidate must fail loud");
    assert!(
        matches!(err, neat_ai_lamarck::scorer::ScorerError::Json(_)),
        "expected a JSON error, got {err:?}"
    );
}

/// Guards the point of the extraction: the corpus a bench writes is the corpus
/// the shared fixture defines, so paired arms in different benches compare.
#[test]
fn every_bench_shares_one_corpus_definition() {
    let dir = tempfile::tempdir().unwrap();
    write_sample(dir.path(), 3, 2);
    let corpus: &Path = &dir.path().join("0.bin");
    assert_eq!(std::fs::read(corpus).unwrap(), expected_corpus_bytes(3, 2));
}
