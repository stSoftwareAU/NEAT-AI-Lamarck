//! Output gate: every creature Lamarck builds is certified by the shared
//! `neat_core::creature_validate` before it escapes (issues #189 / #192).
//!
//! These tests call the real graft and structural entry points and assert on
//! what they return — a certified creature, or a loud error naming the graft
//! that broke a rule.

use neat_ai_lamarck::grafts::{
    Graft, GraftKind, GraftNeuron, GraftStats, GraftSynapse, apply_graft, apply_grafts,
    graft_from_add_synapse,
};
use neat_ai_lamarck::structural::{NeuronBridgeSpec, add_neuron_bridge, add_synapse};
use neat_ai_lamarck::validate::{LAMARCK_VALIDATE_OPTIONS, validate_creature};
use neat_core::{CreatureExport, creature_validate, parse_creature_json};

/// The `grafts.rs` fixture: 2 inputs, `input-0 -> h1 -> o1`, feed-forward.
fn tiny_creature() -> CreatureExport {
    parse_creature_json(
        r#"{
          "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
          "neurons":[
            {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
            {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
          ],
          "synapses":[
            {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
            {"fromUUID":"h1","toUUID":"o1","weight":1.0}
          ]
        }"#,
    )
    .unwrap()
}

fn stats_now() -> GraftStats {
    // `GraftStats` has no public constructor; round-trip a known-good graft.
    graft_from_add_synapse("input-1", "h1", 0.0).stats
}

#[test]
fn tiny_fixture_is_certified_by_the_shared_validator() {
    validate_creature(&tiny_creature(), "fixture").expect("fixture host must be valid");
}

#[test]
fn applied_graft_output_passes_the_shared_validator() {
    let host = tiny_creature();
    let graft = graft_from_add_synapse("input-1", "h1", 0.05);
    let out = apply_graft(&host, &graft).expect("valid graft applies");

    let stats = creature_validate(&out, &LAMARCK_VALIDATE_OPTIONS)
        .expect("graft output must satisfy the shared rules");
    assert_eq!(stats.connections, 3);
    assert_eq!(stats.hidden, 1);
}

#[test]
fn applied_graft_keeps_synapses_in_canonical_order() {
    // Rule 25 — synapses sorted by `(from index, to index)`. Appending a new
    // edge at the end of the list breaks it; the insert must be ordered.
    let host = tiny_creature();
    let graft = graft_from_add_synapse("input-1", "h1", 0.05);
    let out = apply_graft(&host, &graft).expect("valid graft applies");

    let order: Vec<(&str, &str)> = out
        .synapses
        .iter()
        .map(|s| (s.from_uuid.as_str(), s.to_uuid.as_str()))
        .collect();
    assert_eq!(
        order,
        vec![("input-0", "h1"), ("input-1", "h1"), ("h1", "o1")],
        "new edge must land in (from, to) index order, not at the end"
    );
}

#[test]
fn graft_growing_an_unreachable_hidden_is_refused() {
    // `b1` gains an outward edge into `h1` but nothing feeds it: rule 17,
    // `NO_INWARD_CONNECTIONS`. Before the gate this returned `Ok`.
    let host = tiny_creature();
    let graft = Graft {
        id: "neuron:b1".into(),
        kind: GraftKind::AddNeuronBridge,
        neurons: vec![GraftNeuron {
            uuid: "b1".into(),
            bias: 0.0,
            squash: Some("IDENTITY".into()),
        }],
        synapses: vec![GraftSynapse {
            from_uuid: "b1".into(),
            to_uuid: "h1".into(),
            weight: 0.02,
            synapse_type: None,
        }],
        requires: vec!["h1".into()],
        stats: stats_now(),
    };

    let err = apply_graft(&host, &graft).expect_err("an unreachable hidden must not escape");
    assert!(err.contains("neuron:b1"), "error names the graft: {err}");
    assert!(
        err.contains("NO_INWARD_CONNECTIONS"),
        "error carries the validator reason: {err}"
    );
    assert!(
        err.contains("b1"),
        "error carries the validator message: {err}"
    );
}

#[test]
fn apply_grafts_attributes_the_failure_to_the_offending_graft() {
    let host = tiny_creature();
    let good = graft_from_add_synapse("input-1", "h1", 0.05);
    let bad = Graft {
        id: "neuron:orphan".into(),
        kind: GraftKind::AddNeuronBridge,
        neurons: vec![GraftNeuron {
            uuid: "orphan".into(),
            bias: 0.0,
            squash: Some("IDENTITY".into()),
        }],
        synapses: vec![GraftSynapse {
            from_uuid: "orphan".into(),
            to_uuid: "o1".into(),
            weight: 0.02,
            synapse_type: None,
        }],
        requires: vec!["o1".into()],
        stats: stats_now(),
    };

    let err = apply_grafts(&host, &[&good, &bad]).expect_err("the second graft is invalid");
    assert!(
        err.contains("neuron:orphan"),
        "the graft that caused it is named, not the earlier one: {err}"
    );
    assert!(
        !err.contains("edge:input-1->h1"),
        "wrong attribution: {err}"
    );
}

#[test]
fn valid_multi_graft_stack_still_applies() {
    let host = tiny_creature();
    let a = graft_from_add_synapse("input-1", "h1", 0.05);
    let mut b = graft_from_add_synapse("input-0", "o1", 0.01);
    b.id = "edge:input-0->o1".into();

    let out = apply_grafts(&host, &[&a, &b]).expect("both grafts are valid");
    creature_validate(&out, &LAMARCK_VALIDATE_OPTIONS).expect("stacked output stays valid");
    assert_eq!(out.synapses.len(), 4);
}

#[test]
fn neuron_bridge_output_is_certified() {
    let mut creature = tiny_creature();
    add_neuron_bridge(
        &mut creature,
        NeuronBridgeSpec {
            from_uuid: "input-1",
            focus_uuid: "h1",
            new_uuid: "bridge-1".into(),
            squash: "IDENTITY",
            bias: 0.0,
            w_in: 0.02,
            w_out: 0.03,
        },
    )
    .expect("a legal bridge applies");

    creature_validate(&creature, &LAMARCK_VALIDATE_OPTIONS)
        .expect("bridge output must satisfy the shared rules");
}

#[test]
fn add_synapse_inserts_in_canonical_order() {
    let mut creature = tiny_creature();
    // `input-1 -> h1` sorts before the existing `h1 -> o1`.
    add_synapse(&mut creature, "input-1".into(), "h1", 0.05);
    creature_validate(&creature, &LAMARCK_VALIDATE_OPTIONS).expect("ordered insert stays valid");

    // A second edge into the output sorts after both.
    add_synapse(&mut creature, "input-0".into(), "o1", 0.01);
    creature_validate(&creature, &LAMARCK_VALIDATE_OPTIONS).expect("ordered insert stays valid");
}

#[test]
fn validation_failure_names_reason_message_and_index() {
    // A hidden neuron nothing reads — rule 17 stops on the neuron itself.
    let broken = parse_creature_json(
        r#"{
          "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
          "neurons":[
            {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
            {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
          ],
          "synapses":[
            {"fromUUID":"input-0","toUUID":"o1","weight":1.0},
            {"fromUUID":"h1","toUUID":"o1","weight":1.0}
          ]
        }"#,
    )
    .unwrap();

    let err = validate_creature(&broken, "graft neuron:x").expect_err("h1 has no inward edge");
    assert!(err.contains("graft neuron:x"), "context is carried: {err}");
    assert!(err.contains("NO_INWARD_CONNECTIONS"), "reason: {err}");
    // Compiled index — the single input occupies 0, so `h1` is 1.
    assert!(err.contains("[neuron 1]"), "neuron index is carried: {err}");
}

#[test]
fn chosen_options_reject_a_recursive_edge() {
    // `forward_only: true` is the deliberate choice — prove it bites rather
    // than asserting on the constant's fields.
    let mut creature = tiny_creature();
    creature.synapses.clear();
    creature.synapses.push(neat_core::SynapseExport {
        from_uuid: "input-0".into(),
        to_uuid: "h1".into(),
        weight: 1.0,
        synapse_type: None,
    });
    creature.synapses.push(neat_core::SynapseExport {
        from_uuid: "h1".into(),
        to_uuid: "o1".into(),
        weight: 1.0,
        synapse_type: None,
    });
    // `o1` (index 3) feeding `h1` (index 2) is a backward edge.
    creature.synapses.push(neat_core::SynapseExport {
        from_uuid: "o1".into(),
        to_uuid: "h1".into(),
        weight: 0.1,
        synapse_type: None,
    });

    let err = validate_creature(&creature, "options").expect_err("recursion must be refused");
    assert!(err.contains("RECURSIVE_SYNAPSE"), "reason: {err}");
}

#[test]
fn chosen_options_pin_no_expected_counts() {
    // Lamarck grows structure, so it has no fixed expected counts: a creature
    // that gained a neuron and two synapses is still certified.
    let mut grown = tiny_creature();
    add_neuron_bridge(
        &mut grown,
        NeuronBridgeSpec {
            from_uuid: "input-1",
            focus_uuid: "h1",
            new_uuid: "bridge-1".into(),
            squash: "IDENTITY",
            bias: 0.0,
            w_in: 0.02,
            w_out: 0.03,
        },
    )
    .expect("a legal bridge applies");

    let before = creature_validate(&tiny_creature(), &LAMARCK_VALIDATE_OPTIONS).unwrap();
    let after = creature_validate(&grown, &LAMARCK_VALIDATE_OPTIONS)
        .expect("a grown creature is not held to the host's counts");
    assert_eq!(after.neurons(), before.neurons() + 1);
    assert_eq!(after.connections, before.connections + 2);
}
