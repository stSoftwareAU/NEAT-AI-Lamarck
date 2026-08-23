//! Keep a creature's `memetic` record pointing at structure it still has
//! (issues #197, #199).
//!
//! `memetic` holds per-neuron bias deltas and per-synapse weight deltas keyed
//! to a **specific** structure, so the rule is: any pass that removes a neuron
//! or a synapse must prune the record with it. A key naming structure that is
//! gone is a dangling reference, and `neat_core::creature_validate` rule 31
//! (`MEMETIC`) refuses the whole creature over it — which is how one unpruned
//! removal in [`crate::structural::split_incoming_synapse`] made
//! `structural_add_neuron` yield nothing at all on fine-tuned creatures.
//!
//! Two boundaries, both here:
//!
//! * [`prune_memetic`] — called at the point of removal, before the edit is
//!   certified. It drops **only** dangling keys; `MemeticExport::extra`
//!   (`generation` / `score` / `ancestry`) and every still-resolving delta are
//!   kept, which is why the blunt `memetic = None` is wrong.
//! * [`assert_memetic_resolves`] — called on every write path, so a removal
//!   that forgets to prune cannot reach disk. A dangling reference costs the
//!   fleet days when it surfaces downstream, so it fails loud here instead.
//!
//! Adding structure is *not* a removal: appending a synapse or a neuron leaves
//! every existing key resolvable, and pruning there would throw away valid
//! fine-tuning history. A tags-only pass is not structural either — it must
//! leave both `memetic` and `uuid` untouched.
//!
//! **Resolution is `neat-core`'s, not Lamarck's.** Both boundaries run
//! `CreatureExport::prune_memetic` / `MemeticExport::prune_to`, which live
//! beside rule 31 in `neat_core::creature_validate`, so what Lamarck prunes and
//! what Lamarck refuses cannot drift from what the rule accepts. That matters
//! because a key resolves by runtime id first — implicit inputs by index,
//! outputs forced to `-(outputIndex + 1)`, everything else by its declared id
//! **or a deterministic hash of its uuid** — and only then by wire uuid. The
//! hash half is NEAT-AI's `deterministicIdFromUuid`; Lamarck used to treat any
//! numeric key in its `[1_000_000, 2_000_000_000)` range as unverifiable, which
//! left the original silent-candidate-loss reachable for a memetic keyed by a
//! derived id. Delegating removes that last gap.

use neat_core::{CreatureExport, MemeticWeights};

/// Drop every memetic key that no longer resolves against `creature`.
///
/// Call immediately after removing a neuron or a synapse, before the edit is
/// certified. Idempotent, and a no-op on a creature with no memetic record or
/// with a record that still resolves in full — including on every pure-append
/// edit, which is why the append paths do not call it.
///
/// Kept: `extra` verbatim, every bias whose neuron survives, every weight whose
/// edge survives, and the record itself (an emptied record is still the fact
/// that the creature was fine-tuned).
pub fn prune_memetic(creature: &mut CreatureExport) {
    creature.prune_memetic();
}

/// Refuse a creature whose memetic record names structure it no longer carries.
///
/// The write-path half of the rule: [`prune_memetic`] stops the dangling record
/// being *built*, this stops one being *written*. Called from every creature
/// write path (`width::checked_creature_json*`, the `tags` check-in
/// serialisers), so the failure points at the writer, not at a later reader.
///
/// The check *is* the prune: whatever `MemeticExport::prune_to` would drop is
/// exactly what rule 31 would refuse, so the guard asks the shared resolution
/// what it would remove and names the first casualty rather than re-deriving
/// the vocabulary itself.
///
/// # Errors
///
/// Returns the first dangling reference, naming the key or edge that no longer
/// resolves. Never degraded to a log line: a dangling memetic on disk is the
/// same class of wire-format fault that has already cost the fleet whole days
/// of Backprop and Lamarck outages, and it surfaces far from its cause.
pub fn assert_memetic_resolves(creature: &CreatureExport) -> Result<(), String> {
    let Some(memetic) = creature.memetic.as_ref() else {
        return Ok(());
    };
    let mut pruned = memetic.clone();
    pruned.prune_to(creature);

    if let Some(key) = memetic
        .biases
        .keys()
        .find(|key| !pruned.biases.contains_key(key.as_str()))
    {
        return Err(format!(
            "refusing to write creature: memetic bias {key} names no neuron in the creature \
             — a removal did not prune the memetic record"
        ));
    }

    let dangling_edge = |from: String, to: String| {
        format!(
            "refusing to write creature: memetic weight {from} -> {to} names no synapse in the \
             creature — a removal did not prune the memetic record"
        )
    };
    match (&memetic.weights, &pruned.weights) {
        (MemeticWeights::Rows(rows), MemeticWeights::Rows(kept)) => {
            if let Some(row) = first_dropped(rows, kept) {
                return Err(dangling_edge(
                    row.from_uuid.clone().unwrap_or_default(),
                    row.to_uuid.clone().unwrap_or_default(),
                ));
            }
        }
        (MemeticWeights::ById(by_id), MemeticWeights::ById(kept)) => {
            for (key, entries) in by_id {
                let Some(kept_entries) = kept.get(key) else {
                    return Err(format!(
                        "refusing to write creature: memetic weight key {key} names no neuron in \
                         the creature — a removal did not prune the memetic record"
                    ));
                };
                if let Some(entry) = first_dropped(entries, kept_entries) {
                    return Err(dangling_edge(
                        key.clone(),
                        entry.to_id.map(|id| id.to_string()).unwrap_or_default(),
                    ));
                }
            }
        }
        // `prune_to` never rewrites one wire form as the other. Reaching here
        // means the shared contract changed under us — say so rather than
        // reporting a record that was never checked as sound.
        _ => {
            return Err(
                "refusing to write creature: neat_core::MemeticExport::prune_to changed the \
                 memetic weights wire form — the memetic guard cannot be trusted"
                    .to_string(),
            );
        }
    }
    Ok(())
}

/// The first element of `before` that the prune dropped, or `None` when it kept
/// them all.
///
/// `retain` preserves order, so `after` is a subsequence of `before` and one
/// forward walk finds the first gap.
fn first_dropped<'a, T: PartialEq>(before: &'a [T], after: &[T]) -> Option<&'a T> {
    let mut kept = after.iter();
    let mut next = kept.next();
    for item in before {
        match next {
            Some(candidate) if candidate == item => next = kept.next(),
            _ => return Some(item),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;

    const ROWS: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
      "neurons":[
        {"type":"hidden","uuid":"h1","bias":0.25,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0}
      ],
      "memetic":{
        "biases":{"h1":0.01,"ghost":0.02},
        "weights":[
          {"fromUUID":"input-0","toUUID":"h1","weight":0.9},
          {"fromUUID":"ghost","toUUID":"o1","weight":0.5}
        ],
        "generation":3
      }
    }"#;

    const BY_ID: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
      "neurons":[
        {"id":2,"type":"hidden","uuid":"h1","bias":0.25,"squash":"IDENTITY"},
        {"id":-1,"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0}
      ],
      "memetic":{
        "biases":{"2":0.01},
        "weights":{"2":[{"toId":-1,"weight":1.1},{"toId":99,"weight":0.3}],"77":[]}
      }
    }"#;

    /// Outputs are keyed by the id the shared rules force on them
    /// (`-(outputIndex + 1)`), not by anything the file declares, and `h1`
    /// declares none at all.
    const DERIVED_IDS: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
      "neurons":[
        {"id":2,"type":"hidden","uuid":"h1","bias":0.25,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0}
      ],
      "memetic":{
        "biases":{"0":0.05,"2":0.01},
        "weights":{"2":[{"toId":-1,"weight":1.1}]}
      }
    }"#;

    /// `h1` declares no `id`, so its runtime id is NEAT-AI's deterministic hash
    /// of its uuid (`deterministicIdFromUuid`, folded into
    /// `[1_000_000, 2_000_000_000)`). `1500000` is in the same range and names
    /// nothing — the distinction Lamarck could not draw before issue #199.
    const HASHED_IDS: &str = r#"{
      "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
      "neurons":[
        {"type":"hidden","uuid":"h1","bias":0.25,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses":[
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0}
      ],
      "memetic":{
        "biases":{"1003273":0.01,"1500000":0.02},
        "weights":{"1003273":[{"toId":-1,"weight":1.1}],"1500000":[{"toId":-1,"weight":0.7}]}
      }
    }"#;

    /// `deterministicIdFromUuid("h1")` — the runtime id rule 31 gives the
    /// `HASHED_IDS` hidden neuron.
    const H1_DERIVED_ID: &str = "1003273";

    fn by_id(
        creature: &CreatureExport,
    ) -> &std::collections::BTreeMap<String, Vec<neat_core::MemeticWeightExport>> {
        match &creature.memetic.as_ref().unwrap().weights {
            MemeticWeights::ById(by_id) => by_id,
            MemeticWeights::Rows(_) => panic!("fixture is the map form"),
        }
    }

    /// A bias keyed by an input's runtime id (an input's id *is* its index) is
    /// a live reference — dropping it would throw away fine-tuning history the
    /// shared rules accept.
    #[test]
    fn prune_keeps_a_bias_keyed_by_an_input_index() {
        let mut creature = parse_creature_json(DERIVED_IDS).unwrap();
        prune_memetic(&mut creature);
        let biases = &creature.memetic.as_ref().unwrap().biases;
        assert_eq!(biases.get("0"), Some(&0.05), "input-0's bias delta");
        assert_eq!(biases.get("2"), Some(&0.01), "h1's bias delta");
    }

    /// `toId: -1` names the first output whatever the file declares, so the
    /// entry lives while `h1 -> o1` does and dies with it.
    #[test]
    fn prune_follows_the_forced_output_id() {
        let mut kept = parse_creature_json(DERIVED_IDS).unwrap();
        prune_memetic(&mut kept);
        assert_memetic_resolves(&kept).expect("h1 -> o1 is still there");
        assert_eq!(by_id(&kept)["2"].len(), 1);

        let mut removed = parse_creature_json(DERIVED_IDS).unwrap();
        removed.synapses.retain(|s| s.to_uuid != "o1");
        prune_memetic(&mut removed);
        assert!(
            by_id(&removed)["2"].is_empty(),
            "the removed edge's entry is pruned"
        );
    }

    /// A neuron declaring no `id` still has one — the hash of its uuid — so a
    /// key naming it is live. Lamarck used to call this "unverifiable" and
    /// guess; the shared resolution knows.
    #[test]
    fn prune_keeps_a_key_that_is_a_neurons_uuid_derived_id() {
        let mut creature = parse_creature_json(HASHED_IDS).unwrap();
        prune_memetic(&mut creature);
        let memetic = creature.memetic.as_ref().unwrap();
        assert_eq!(
            memetic.biases.get(H1_DERIVED_ID),
            Some(&0.01),
            "h1's bias delta, keyed by its uuid-derived id: {memetic:?}"
        );
        assert_eq!(
            by_id(&creature)[H1_DERIVED_ID].len(),
            1,
            "h1 -> o1 is still there"
        );
        assert_memetic_resolves(&creature).expect("a hash-keyed live reference must be writable");
    }

    /// The other half of the same distinction: a numeric key in the derived-id
    /// range that hashes to no neuron is dangling, and is now both pruned and
    /// refused rather than waved through.
    #[test]
    fn prune_drops_a_derived_range_key_that_names_nothing() {
        let unpruned = parse_creature_json(HASHED_IDS).unwrap();
        let err = assert_memetic_resolves(&unpruned).expect_err("1500000 names no neuron");
        assert!(err.contains("memetic bias 1500000"), "{err}");

        let mut creature = parse_creature_json(HASHED_IDS).unwrap();
        prune_memetic(&mut creature);
        let memetic = creature.memetic.as_ref().unwrap();
        assert!(
            !memetic.biases.contains_key("1500000"),
            "the dangling bias is dropped: {memetic:?}"
        );
        assert!(
            !by_id(&creature).contains_key("1500000"),
            "the dangling weight key is dropped: {memetic:?}"
        );
    }

    #[test]
    fn prune_drops_only_the_dangling_bias() {
        let mut creature = parse_creature_json(ROWS).unwrap();
        prune_memetic(&mut creature);
        let biases = &creature.memetic.as_ref().unwrap().biases;
        assert_eq!(biases.len(), 1);
        assert_eq!(biases.get("h1"), Some(&0.01));
    }

    #[test]
    fn prune_drops_only_the_dangling_row_and_keeps_extra() {
        let mut creature = parse_creature_json(ROWS).unwrap();
        prune_memetic(&mut creature);
        let memetic = creature.memetic.as_ref().unwrap();
        let rows = match &memetic.weights {
            MemeticWeights::Rows(rows) => rows,
            MemeticWeights::ById(_) => panic!("the row form must be written back as rows"),
        };
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].from_uuid.as_deref(), Some("input-0"));
        assert_eq!(
            memetic.extra.get("generation").and_then(|v| v.as_i64()),
            Some(3)
        );
    }

    #[test]
    fn prune_drops_an_unresolvable_key_and_a_dangling_entry() {
        let mut creature = parse_creature_json(BY_ID).unwrap();
        prune_memetic(&mut creature);
        // Key "77" names no neuron; entry `toId: 99` names no neuron.
        assert_eq!(by_id(&creature).len(), 1);
        assert_eq!(by_id(&creature)["2"].len(), 1);
        assert_eq!(by_id(&creature)["2"][0].to_id, Some(-1));
    }

    #[test]
    fn prune_is_idempotent_and_leaves_a_sound_record_alone() {
        let mut once = parse_creature_json(ROWS).unwrap();
        prune_memetic(&mut once);
        let mut twice = once.clone();
        prune_memetic(&mut twice);
        assert_eq!(once, twice);
    }

    #[test]
    fn prune_is_a_no_op_without_a_memetic_record() {
        let plain = ROWS.replacen(r#""memetic""#, r#""ignoredMemetic""#, 1);
        let mut creature = parse_creature_json(&plain).unwrap();
        let before = creature.clone();
        prune_memetic(&mut creature);
        assert_eq!(creature, before);
        assert!(creature.memetic.is_none());
    }

    #[test]
    fn prune_keeps_the_record_when_everything_dangles() {
        let mut creature = parse_creature_json(ROWS).unwrap();
        creature.synapses.clear();
        creature.neurons.clear();
        prune_memetic(&mut creature);
        let memetic = creature
            .memetic
            .as_ref()
            .expect("the record itself survives");
        assert!(memetic.biases.is_empty());
        assert_eq!(
            memetic.extra.get("generation").and_then(|v| v.as_i64()),
            Some(3)
        );
    }

    #[test]
    fn assert_names_the_first_dangling_reference() {
        let creature = parse_creature_json(ROWS).unwrap();
        let err = assert_memetic_resolves(&creature).expect_err("`ghost` resolves to nothing");
        assert!(err.contains("memetic bias ghost"), "{err}");

        let mut only_weights = parse_creature_json(ROWS).unwrap();
        only_weights.memetic.as_mut().unwrap().biases.clear();
        let err = assert_memetic_resolves(&only_weights).expect_err("the ghost row still dangles");
        assert!(err.contains("ghost -> o1"), "{err}");
    }

    #[test]
    fn assert_passes_once_pruned() {
        let mut creature = parse_creature_json(ROWS).unwrap();
        prune_memetic(&mut creature);
        assert_memetic_resolves(&creature).expect("a pruned record resolves in full");

        let mut by_id = parse_creature_json(BY_ID).unwrap();
        prune_memetic(&mut by_id);
        assert_memetic_resolves(&by_id).expect("a pruned map record resolves in full");
    }

    #[test]
    fn assert_reports_a_dangling_id_keyed_entry_then_an_unresolvable_key() {
        let creature = parse_creature_json(BY_ID).unwrap();
        let err = assert_memetic_resolves(&creature).expect_err("`toId: 99` names no neuron");
        assert!(err.contains("2 -> 99"), "{err}");

        // With the dangling entry gone, the unresolvable key is next.
        let mut creature = creature;
        match &mut creature.memetic.as_mut().unwrap().weights {
            MemeticWeights::ById(by_id) => {
                by_id.get_mut("2").unwrap().retain(|e| e.to_id != Some(99))
            }
            MemeticWeights::Rows(_) => panic!("fixture is the map form"),
        }
        let err = assert_memetic_resolves(&creature).expect_err("key 77 names no neuron");
        assert!(err.contains("key 77"), "{err}");
    }

    /// A malformed row is a defect in the record as supplied, not something a
    /// removal caused — the shared prune leaves it for rule 31 to report, so
    /// the guard must not claim it dangles either.
    #[test]
    fn assert_leaves_a_malformed_row_to_rule_31() {
        let malformed = ROWS.replacen(
            r#"{"fromUUID":"ghost","toUUID":"o1","weight":0.5}"#,
            r#"{"toUUID":"o1","weight":0.5}"#,
            1,
        );
        let mut creature = parse_creature_json(&malformed).unwrap();
        creature.memetic.as_mut().unwrap().biases.clear();
        assert_memetic_resolves(&creature).expect("malformed is not dangling");
    }
}
