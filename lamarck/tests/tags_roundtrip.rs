//! NEAT-AI#3750: a real Lamarck accept-and-write cycle must not lose metadata.
//!
//! An optimisation run reads a creature, mutates biases / weights / structure,
//! and writes `best.json` (and `winners/`) back for GRQ check-in. Everything it
//! did not optimise — the top-level `uuid` / `tags` / `memetic` block, the
//! per-neuron `intelligentDesign` pedigree and the per-synapse tags — must come
//! out the other side byte for byte, while the tags this optimiser owns
//! (`score`, `error`, `lamarck`) are stamped (GRQ #3952).
//!
//! The assertions are byte-level, not "key exists": tag values and tag order
//! are compared verbatim, the stamped numbers are compared as exact strings at
//! full precision (`worker/Lamarck/run.sh` parses them), and the memetic block
//! is compared as raw text so a re-ordered key or a re-formatted number fails.

use neat_ai_lamarck::focus::FocusPolicy;
use neat_ai_lamarck::observations::StatsMode;
use neat_ai_lamarck::scorer::{DirectoryScorer, ScoreResult, ScoreSample, ScorerError};
use neat_ai_lamarck::{LamarckConfig, RunResult, run_optimisation};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::TempDir;

/// Identity chain carrying every metadata surface the Rust path used to drop.
///
/// `memetic` deliberately lists its keys out of alphabetical order and holds
/// numbers whose formatting is easy to lose (`1e-7`, a full-precision double),
/// so any re-serialisation through a sorted map shows up as a failure.
const TAGGED_CHAIN: &str = r#"{
  "semanticVersion": "4.0.0",
  "uuid": "creature-3750",
  "input": 1,
  "output": 1,
  "forwardOnly": true,
  "neurons": [
    {
      "type": "hidden",
      "uuid": "h1",
      "bias": 0.1,
      "squash": "IDENTITY",
      "tags": [
        {"name": "intelligentDesign", "value": "Swish -> SOFTSIGN"},
        {"name": "CRISPR", "value": "h1-remapped"}
      ]
    },
    {
      "type": "output",
      "uuid": "o1",
      "bias": 0.0,
      "squash": "IDENTITY",
      "tags": [{"name": "intelligentDesign", "value": "IDENTITY -> IDENTITY"}]
    }
  ],
  "synapses": [
    {
      "fromUUID": "input-0",
      "toUUID": "h1",
      "weight": 1.0,
      "tags": [{"name": "backpropagation", "value": "🌀 leave me alone"}]
    },
    {"fromUUID": "h1", "toUUID": "o1", "weight": 1.0}
  ],
  "tags": [
    {"name": "name", "value": "Tiny"},
    {"name": "intelligentDesign", "value": "Swish -> SOFTSIGN"},
    {"name": "backpropagation", "value": "🌀 leave me alone"},
    {"name": "score", "value": "0.1"}
  ],
  "memetic": {
    "generation": 7,
    "score": 0.30000000000000004,
    "biases": {"h1": 1e-7, "o1": -0.5},
    "weights": {"input-0->h1": 2.5}
  }
}"#;

/// Awkward numbers on purpose: a `{:.6}`-style stamp would truncate both, so
/// the exact-string assertions below pin full precision for GRQ's parser.
const BASELINE_SCORE: f64 = 0.3451502296337825;
const WINNER_SCORE: f64 = 0.3451532296337825;
const WINNER_ERROR: f64 = 0.6548467703662175;

/// Scores `candidate-000` just over the acceptance bar, everything else flat.
struct WinningCandidateScorer;

impl DirectoryScorer for WinningCandidateScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        _training_data: &Path,
        _sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let mut map = BTreeMap::new();
        for entry in std::fs::read_dir(candidates_dir)
            .into_iter()
            .flatten()
            .flatten()
        {
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(stem) = name.strip_suffix(".json") else {
                continue;
            };
            let (score, error) = if stem == "candidate-000" {
                (WINNER_SCORE, WINNER_ERROR)
            } else {
                (BASELINE_SCORE, 1.0 - BASELINE_SCORE)
            };
            map.insert(
                stem.to_string(),
                ScoreResult {
                    score,
                    error,
                    complexity_penalty: 0.0,
                },
            );
        }
        Ok(map)
    }
}

/// One experiment over the tagged fixture, pinned to the `o1` focus.
fn accept_once() -> (TempDir, RunResult) {
    let dir = tempfile::tempdir().expect("tempdir");
    let creature = dir.path().join("creature.json");
    std::fs::write(&creature, TAGGED_CHAIN).expect("write creature");
    let training = dir.path().join("data");
    std::fs::create_dir_all(&training).expect("create data dir");
    // One record: input 1.0 → target 0.5, leaving a gap to optimise.
    std::fs::write(
        training.join("0.bin"),
        [1.0f32, 0.5f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .expect("write record");
    let config = LamarckConfig {
        creature,
        training_data: training,
        timeout: Duration::from_secs(30),
        max_experiments: Some(1),
        candidates: 4,
        scale_candidate_quotas: false,
        min_improvement: 1e-6,
        seed: Some(1),
        scorer_path: PathBuf::from("rust_scorer"),
        output_dir: dir.path().join("out"),
        preserve_losers: true,
        stats_mode: StatsMode::Quick,
        quick_sample_records: 8,
        focus_neuron: Some("o1".into()),
        focus_policy: FocusPolicy::Random,
        phase0_parity: false,
        screen_sample_rate: None,
        screen_promote_threshold: 0.0,
        ..LamarckConfig::default()
    };
    let result = run_optimisation(&config, &WinningCandidateScorer).expect("run the optimiser");
    assert_eq!(
        result.acceptances, 1,
        "the scripted winner must be accepted"
    );
    (dir, result)
}

/// Raw text of the top-level `memetic` block with all whitespace removed.
///
/// The fixture's memetic block holds only numbers and plain keys — no string
/// values — so dropping whitespace is a safe way to compare the source against
/// the pretty-printed rewrite byte for byte, key order and number formatting
/// included.
fn memetic_text(json: &str) -> String {
    let key = json.find("\"memetic\"").expect("memetic key present");
    let start = json[key..].find('{').expect("memetic object opens") + key;
    let mut depth = 0usize;
    for (offset, ch) in json[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let block = &json[start..start + offset + 1];
                    return block.chars().filter(|c| !c.is_whitespace()).collect();
                }
            }
            _ => {}
        }
    }
    panic!("memetic object never closed");
}

/// The value of `name` in a tag array, or `None` when absent.
fn tag_value<'a>(tags: &'a Value, name: &str) -> Option<&'a str> {
    tags.as_array()?
        .iter()
        .find(|t| t["name"] == name)?
        .get("value")?
        .as_str()
}

/// Every metadata surface #3746 reported as lost, through a real accept.
#[test]
fn full_metadata_survives_a_lamarck_accept() {
    let (_dir, result) = accept_once();
    let best_text = std::fs::read_to_string(&result.best_path).expect("read best.json");
    let source: Value = serde_json::from_str(TAGGED_CHAIN).expect("parse fixture");
    let best: Value = serde_json::from_str(&best_text).expect("parse best.json");

    // The rewrite must be real: the accepted candidate changed a gene.
    assert!(
        best["neurons"] != source["neurons"] || best["synapses"] != source["synapses"],
        "the accept must have changed the creature — otherwise this proves nothing"
    );

    assert_eq!(best["uuid"], source["uuid"], "top-level uuid survives");

    // Per-neuron tags, verbatim and in order (the intelligentDesign pedigree).
    for (index, neuron) in source["neurons"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            best["neurons"][index]["tags"], neuron["tags"],
            "neuron {index} tags survive byte for byte"
        );
    }

    // Per-synapse tags, verbatim — including the synapse that carries none,
    // which must not gain a `null` or an empty array.
    for (index, synapse) in source["synapses"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            best["synapses"][index].get("tags"),
            synapse.get("tags"),
            "synapse {index} tags survive byte for byte"
        );
    }

    // Untouched top-level tags keep their value and their relative order.
    let best_tags = best["tags"].as_array().expect("top-level tags array");
    let untouched: Vec<(&str, &str)> = best_tags
        .iter()
        .take(3)
        .map(|t| {
            (
                t["name"].as_str().expect("tag name"),
                t["value"].as_str().expect("tag value"),
            )
        })
        .collect();
    assert_eq!(
        untouched,
        vec![
            ("name", "Tiny"),
            ("intelligentDesign", "Swish -> SOFTSIGN"),
            ("backpropagation", "🌀 leave me alone"),
        ],
        "tags owned by other programs are untouched, and keep their order"
    );

    // The memetic block is preserved verbatim: key order and number
    // formatting included.
    assert_eq!(
        memetic_text(&best_text),
        memetic_text(TAGGED_CHAIN),
        "memetic block survives verbatim, key order and all"
    );

    assert!(
        best_text.ends_with('\n'),
        "best.json stays newline-terminated for check-in"
    );
}

/// The check-in contract `worker/Lamarck/run.sh` reads: exact strings, full
/// numeric precision, and nobody else's tag disturbed.
#[test]
fn the_check_in_stamp_keeps_its_exact_strings_and_precision() {
    let (_dir, result) = accept_once();
    let best_text = std::fs::read_to_string(&result.best_path).expect("read best.json");
    let best: Value = serde_json::from_str(&best_text).expect("parse best.json");
    let tags = &best["tags"];

    assert_eq!(
        tag_value(tags, "score"),
        Some("0.3451532296337825"),
        "score is stamped at full precision"
    );
    assert_eq!(
        tag_value(tags, "error"),
        Some("0.6548467703662175"),
        "error is stamped at full precision"
    );
    let message = tag_value(tags, "lamarck").expect("lamarck tag");
    assert!(
        message.starts_with("🦒 Lamarck · 1 accept / 1 exp · last: "),
        "unexpected check-in subject: {message}"
    );
    assert!(
        message.ends_with("· 🎯 o1 · score: 0.345153 improved by 3e-06"),
        "unexpected score clause: {message}"
    );

    // `score` was already present, so it is replaced in place — the pedigree
    // tags around it neither move nor change.
    let names: Vec<&str> = tags
        .as_array()
        .expect("tags array")
        .iter()
        .map(|t| t["name"].as_str().expect("tag name"))
        .collect();
    assert_eq!(
        names,
        vec![
            "name",
            "intelligentDesign",
            "backpropagation",
            "score",
            "error",
            "lamarck"
        ],
        "stamping appends; it never re-orders the existing tags"
    );
    assert_eq!(
        tag_value(tags, "intelligentDesign"),
        Some("Swish -> SOFTSIGN"),
        "intelligentDesign is never mutated (GRQ #3952)"
    );
    assert_eq!(
        tag_value(tags, "backpropagation"),
        Some("🌀 leave me alone"),
        "another program's tag is never mutated (GRQ #3952)"
    );
}

/// `winners/` is a check-in artefact too — it carries the same metadata.
#[test]
fn the_winner_snapshot_carries_the_same_metadata() {
    let (_dir, result) = accept_once();
    let winners = result
        .best_path
        .parent()
        .expect("output dir")
        .join("winners");
    let winner_path = std::fs::read_dir(&winners)
        .expect("winners dir")
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "json"))
        .expect("a winner snapshot was written");
    let text = std::fs::read_to_string(&winner_path).expect("read winner");
    let winner: Value = serde_json::from_str(&text).expect("parse winner");
    let source: Value = serde_json::from_str(TAGGED_CHAIN).expect("parse fixture");

    assert_eq!(winner["uuid"], source["uuid"], "winner keeps the uuid");
    assert_eq!(
        winner["neurons"][0]["tags"], source["neurons"][0]["tags"],
        "winner keeps the per-neuron pedigree"
    );
    assert_eq!(
        winner["synapses"][0]["tags"], source["synapses"][0]["tags"],
        "winner keeps the per-synapse tags"
    );
    assert_eq!(
        memetic_text(&text),
        memetic_text(TAGGED_CHAIN),
        "winner keeps the memetic block verbatim"
    );
    assert_eq!(
        tag_value(&winner["tags"], "score"),
        Some("0.3451532296337825"),
        "winner carries the accept's stamp"
    );
}
