//! A removal prunes the creature's `memetic` record (issue #197).
//!
//! Every pass that removes a neuron or a synapse must drop the memetic keys
//! that named it: the record holds per-neuron bias deltas and per-synapse
//! weight deltas keyed to a specific structure, so a removal leaves those keys
//! dangling and `neat_core::creature_validate` rule 31 (`MEMETIC`) refuses the
//! result. Before this was fixed, `split_incoming_synapse` returned `Err` on
//! any creature carrying a memetic row for the edge it split — so
//! `structural_add_neuron` silently produced no candidate at all on exactly the
//! creatures a Backprop or fine-tuning stage had worked hardest on.
//!
//! These tests call the real entry points and assert on the creature they hand
//! back — never on how the prune is implemented.

use neat_ai_lamarck::focus::IncomingSourceStats;
use neat_ai_lamarck::grafts::{apply_graft, graft_from_add_synapse};
use neat_ai_lamarck::structural::{add_synapse, split_incoming_synapse};
use neat_ai_lamarck::tags::{CreatureMeta, serialize_creature_with_meta};
use neat_ai_lamarck::validate::validate_creature;
use neat_ai_lamarck::width::checked_creature_json;
use neat_core::{CreatureExport, MemeticWeights, parse_creature_json};

/// 2 inputs, `input-0 -> h1`, `input-1 -> h1`, `h1 -> o1`, with a memetic
/// record in the **row** (wire) form naming the `input-0 -> h1` edge.
const ROWS: &str = r#"{
  "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
  "neurons":[
    {"type":"hidden","uuid":"h1","bias":0.25,"squash":"IDENTITY"},
    {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
  ],
  "synapses":[
    {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
    {"fromUUID":"input-1","toUUID":"h1","weight":0.5},
    {"fromUUID":"h1","toUUID":"o1","weight":1.0}
  ],
  "memetic":{
    "biases":{"h1":0.01},
    "weights":[
      {"fromUUID":"input-0","toUUID":"h1","weight":0.9},
      {"fromUUID":"h1","toUUID":"o1","weight":1.1}
    ],
    "generation":7,
    "score":0.42,
    "ancestry":["seed","backprop"]
  }
}"#;

/// The same topology with runtime ids, and a memetic record in the **id-keyed
/// map** form naming the `h1 -> o1` edge (`1 -> -1`).
const BY_ID: &str = r#"{
  "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
  "neurons":[
    {"id":2,"type":"hidden","uuid":"h1","bias":0.25,"squash":"IDENTITY"},
    {"id":-1,"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
  ],
  "synapses":[
    {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
    {"fromUUID":"input-1","toUUID":"h1","weight":0.5},
    {"fromUUID":"h1","toUUID":"o1","weight":1.0}
  ],
  "memetic":{
    "biases":{"2":0.01},
    "weights":{"2":[{"toId":-1,"weight":1.1}]},
    "generation":7
  }
}"#;

/// The stats `split_incoming_synapse` reads: which synapse, from where, at what
/// weight. Every other field is a statistic the split does not consult.
fn incoming(creature: &CreatureExport, synapse_index: usize) -> IncomingSourceStats {
    let synapse = &creature.synapses[synapse_index];
    IncomingSourceStats {
        synapse_index,
        from_uuid: synapse.from_uuid.clone(),
        weight: synapse.weight,
        is_input: synapse.from_uuid.starts_with("input-"),
        input_index: None,
        mean: 0.0,
        variance: 0.0,
        std_dev: 0.0,
        correlation_with_error: None,
        weight_signal_count: None,
        proposed_weight_delta: None,
        mean_weight_sensitivity: None,
    }
}

/// The `(from, to)` pairs a memetic record names, in whichever form it uses.
/// The id-keyed form is reported as `(fromKey, toId)` text so one helper covers
/// both.
fn memetic_pairs(creature: &CreatureExport) -> Vec<(String, String)> {
    let memetic = creature.memetic.as_ref().expect("creature carries memetic");
    match &memetic.weights {
        MemeticWeights::Rows(rows) => rows
            .iter()
            .map(|row| {
                (
                    row.from_uuid.clone().unwrap_or_default(),
                    row.to_uuid.clone().unwrap_or_default(),
                )
            })
            .collect(),
        MemeticWeights::ById(by_id) => by_id
            .iter()
            .flat_map(|(from, entries)| {
                entries.iter().map(move |entry| {
                    (
                        from.clone(),
                        entry.to_id.map(|id| id.to_string()).unwrap_or_default(),
                    )
                })
            })
            .collect(),
    }
}

#[test]
fn split_prunes_the_row_naming_the_removed_edge() {
    let mut creature = parse_creature_json(ROWS).unwrap();
    let src = incoming(&creature, 0);

    let uuid = split_incoming_synapse(&mut creature, &src, "h1", "grown-1".into(), "TANH")
        .expect("the split must produce a candidate, not vanish");
    assert_eq!(uuid, "grown-1");

    // The removed edge is gone from the record; the untouched one survives.
    assert_eq!(
        memetic_pairs(&creature),
        vec![("h1".to_string(), "o1".to_string())],
        "the row naming the removed input-0 -> h1 edge must be pruned"
    );
    validate_creature(&creature, "split").expect("the split output must satisfy rule 31");
}

#[test]
fn split_prunes_the_id_keyed_entry_naming_the_removed_edge() {
    let mut creature = parse_creature_json(BY_ID).unwrap();
    // Split `h1 -> o1` — the edge the id-keyed record names.
    let src = incoming(&creature, 2);

    split_incoming_synapse(&mut creature, &src, "o1", "grown-2".into(), "TANH")
        .expect("the split must produce a candidate, not vanish");

    assert!(
        memetic_pairs(&creature).is_empty(),
        "the entry naming the removed h1 -> o1 edge must be pruned: {:?}",
        memetic_pairs(&creature)
    );
    validate_creature(&creature, "split").expect("the split output must satisfy rule 31");
}

#[test]
fn split_keeps_a_bias_whose_neuron_survives_and_every_extra_key() {
    let mut creature = parse_creature_json(ROWS).unwrap();
    let src = incoming(&creature, 0);
    split_incoming_synapse(&mut creature, &src, "h1", "grown-1".into(), "TANH").unwrap();

    let memetic = creature
        .memetic
        .as_ref()
        .expect("the record itself survives");
    assert_eq!(
        memetic.biases.get("h1"),
        Some(&0.01),
        "h1 is still in the creature, so its bias delta is still meaningful"
    );
    // `generation` / `score` / `ancestry` are flattened into `extra`: the whole
    // fine-tuning history must survive a prune, which is what rules out
    // `memetic = None`.
    assert_eq!(
        memetic.extra.get("generation").and_then(|v| v.as_i64()),
        Some(7)
    );
    assert_eq!(
        memetic.extra.get("score").and_then(|v| v.as_f64()),
        Some(0.42)
    );
    assert_eq!(
        memetic
            .extra
            .get("ancestry")
            .and_then(|v| v.as_array())
            .map(Vec::len),
        Some(2)
    );
}

#[test]
fn a_refused_split_restores_the_memetic_it_pruned() {
    let mut creature = parse_creature_json(ROWS).unwrap();
    // A dangling edge makes every certification fail, so the bridge is refused.
    add_synapse(&mut creature, "ghost".into(), "o1", 0.1);
    let before = creature.clone();
    let src = incoming(&creature, 0);

    split_incoming_synapse(&mut creature, &src, "h1", "grown-1".into(), "TANH")
        .expect_err("an uncertifiable creature must not be returned");

    assert_eq!(
        creature.synapses, before.synapses,
        "the removed edge is restored"
    );
    assert_eq!(
        creature.memetic, before.memetic,
        "a rolled-back split must hand back the memetic it started with"
    );
}

#[test]
fn appending_structure_keeps_the_whole_memetic_record() {
    // Adding structure leaves every existing memetic key referentially valid —
    // pruning here would throw away valid fine-tuning history for no reason.
    let source = parse_creature_json(ROWS).unwrap();

    let mut added = source.clone();
    add_synapse(&mut added, "input-0".into(), "o1", 0.05);
    assert_eq!(added.memetic, source.memetic, "add_synapse must not prune");

    let grafted = apply_graft(&source, &graft_from_add_synapse("input-1", "o1", 0.05))
        .expect("valid graft applies");
    assert_eq!(
        grafted.memetic, source.memetic,
        "apply_graft must not prune"
    );
}

#[test]
fn a_tags_only_write_leaves_memetic_and_uuid_untouched() {
    let tagged = ROWS.replacen('{', r#"{"uuid":"creature-1","#, 1);
    let creature = parse_creature_json(&tagged).unwrap();
    let meta = CreatureMeta::from_creature_json(&tagged);

    let written = serialize_creature_with_meta(&creature, &meta).unwrap();
    let value: serde_json::Value = serde_json::from_str(&written).unwrap();
    let source: serde_json::Value = serde_json::from_str(&tagged).unwrap();

    assert_eq!(
        value["memetic"], source["memetic"],
        "a tags-only pass is not structural: memetic stays byte-identical"
    );
    assert_eq!(
        value["uuid"], source["uuid"],
        "the creature uuid is preserved"
    );
}

#[test]
fn the_write_boundary_refuses_a_dangling_memetic() {
    let mut creature = parse_creature_json(ROWS).unwrap();
    // A removal that skipped the prune — exactly what any future removal path
    // would produce.
    creature.synapses.remove(0);

    let err = checked_creature_json(&creature)
        .expect_err("a creature with a dangling memetic must never reach disk");
    assert!(
        err.to_lowercase().contains("memetic"),
        "the refusal must name the invariant that failed: {err}"
    );
    assert!(
        err.contains("input-0"),
        "the refusal must name the dangling reference: {err}"
    );
}

#[test]
fn the_write_boundary_accepts_a_pruned_creature() {
    let mut creature = parse_creature_json(ROWS).unwrap();
    let src = incoming(&creature, 0);
    split_incoming_synapse(&mut creature, &src, "h1", "grown-1".into(), "TANH").unwrap();

    let json = checked_creature_json(&creature).expect("a pruned creature writes cleanly");
    let reloaded = parse_creature_json(&json).unwrap();
    validate_creature(&reloaded, "written").expect("the written bytes satisfy the shared rules");
}
