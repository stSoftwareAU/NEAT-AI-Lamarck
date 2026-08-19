//! Shared fixture for the example benches (issue #138).
//!
//! Every paired benchmark in `lamarck/examples/` is only meaningful when it
//! measures the *same* synthetic creature over the *same* deterministic
//! corpus — the numbers in `docs/followup-economics.md`,
//! `docs/baseline-reuse.md` and the README's thread-scaling table are compared
//! across benches. This module is the single definition of that fixture, so a
//! change to it (a `semanticVersion` bump, a different fan-in, a widened bias
//! spread) lands on every bench at once instead of drifting copy by copy.
//!
//! Pull it in from an example with:
//!
//! ```ignore
//! mod support;
//! use support::{creature_json, write_sample};
//! ```
//!
//! `lamarck/tests/bench_support.rs` pins the fixture's output.

// Each example is its own crate and uses only the parts of the fixture it
// needs, so the unused remainder is expected rather than dead code.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use neat_ai_lamarck::compute_local_mse;
use neat_ai_lamarck::scorer::{DirectoryScorer, ScoreResult, ScoreSample, ScorerError};
use neat_core::{compile_creature, parse_creature_json};

/// Synthetic creature: `inputs` inputs, `hidden` TANH hiddens, one output.
///
/// Only the hiddens feed the output, so every input is an unused ranked source
/// the candidate generator may propose — the production shape the benchmarks
/// were measured on.
pub fn creature_json(inputs: usize, hidden: usize) -> String {
    let mut neurons = String::new();
    let mut synapses = String::new();
    for h in 0..hidden {
        if h > 0 {
            neurons.push(',');
        }
        let bias = (h as f64 % 7.0) * 0.01 - 0.03;
        neurons.push_str(&format!(
            r#"{{"type":"hidden","uuid":"h{h}","bias":{bias},"squash":"TANH"}}"#
        ));
        // Each hidden reads a deterministic slice of the inputs.
        for k in 0..4 {
            let i = (h * 4 + k) % inputs;
            let weight = 0.05 + ((h + k) as f64 % 11.0) * 0.01;
            synapses.push_str(&format!(
                r#"{{"fromUUID":"input-{i}","toUUID":"h{h}","weight":{weight}}},"#
            ));
        }
        synapses.push_str(&format!(
            r#"{{"fromUUID":"h{h}","toUUID":"o1","weight":{}}},"#,
            0.02 + (h as f64 % 5.0) * 0.01
        ));
    }
    let synapses = synapses.trim_end_matches(',');
    format!(
        r#"{{"semanticVersion":"4.0.0","forwardOnly":true,"input":{inputs},"output":1,
           "neurons":[{neurons},{{"type":"output","uuid":"o1","bias":0.01,"squash":"IDENTITY"}}],
           "synapses":[{synapses}]}}"#
    )
}

/// Deterministic xorshift sample: `inputs` inputs + one target per record,
/// written as little-endian `f32`s into `0.bin` under `dir`.
pub fn write_sample(dir: &Path, records: usize, inputs: usize) {
    let mut file = std::io::BufWriter::new(std::fs::File::create(dir.join("0.bin")).unwrap());
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..records {
        for _ in 0..=inputs {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
            file.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    file.flush().unwrap();
}

/// Scores every creature in a directory by local MSE over the training corpus.
///
/// In-process, so a benchmark using it needs no `rust_scorer` binary and every
/// arm pays exactly the same per-creature scoring cost.
pub struct LocalMseScorer;

impl DirectoryScorer for LocalMseScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        training_data: &Path,
        _sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let mut scores = BTreeMap::new();
        let entries =
            std::fs::read_dir(candidates_dir).map_err(|e| ScorerError::Process(e.to_string()))?;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(stem) = name.strip_suffix(".json") else {
                continue;
            };
            let text = std::fs::read_to_string(entry.path())
                .map_err(|e| ScorerError::Json(e.to_string()))?;
            let creature =
                parse_creature_json(&text).map_err(|e| ScorerError::Json(e.to_string()))?;
            let mut network =
                compile_creature(&creature).map_err(|e| ScorerError::Invalid(e.to_string()))?;
            let (mse, _) = compute_local_mse(&creature, &mut network, training_data)
                .map_err(ScorerError::Invalid)?;
            scores.insert(
                stem.to_string(),
                ScoreResult {
                    score: 1.0 - mse,
                    error: mse,
                    complexity_penalty: 0.0,
                },
            );
        }
        Ok(scores)
    }
}
