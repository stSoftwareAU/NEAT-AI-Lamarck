//! Keep a creature's `memetic` record pointing at structure it still has
//! (issue #197).
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
//! Resolution follows rule 31: a key is a runtime neuron id first — implicit
//! inputs by their index, outputs by `-(outputIndex + 1)` whatever the file
//! declares, everything else by its declared id — then a wire uuid, where
//! implicit inputs are `input-N`. An id-keyed entry's `toId` resolves by
//! runtime id only.
//!
//! That mirrors the *derivable* half of the rule; rule 31 additionally accepts
//! a deterministic hash of a neuron's uuid as its id, and that hash lives in
//! `neat-core`. This module is therefore deliberately conservative about the
//! keys it cannot verify: a numeric key inside that hash's range which matches
//! no declared id is neither pruned nor refused. The shared
//! home for the whole rule is `neat-core`'s `CreatureExport::prune_memetic` /
//! `MemeticExport::prune_to`, raised alongside this fix; issue #199 tracks
//! replacing this module's body with a call to it once that release lands.

use neat_core::{CreatureExport, MemeticExport, MemeticWeightExport, MemeticWeightRowExport};
use std::collections::{HashMap, HashSet};

/// The half-open range NEAT-AI folds a uuid-derived runtime id into
/// (`deterministicIdFromUuid`), chosen so a derived id can collide with neither
/// an input id (`0..input`) nor an output id (negative).
///
/// Lamarck cannot reproduce the hash itself — it is `neat-core`'s to own — so a
/// numeric key in this range that matches no declared id is treated as
/// *unverifiable* rather than dangling — neither pruned nor refused.
const DERIVED_ID_RANGE: std::ops::Range<i64> = 1_000_000..2_000_000_000;

/// What a memetic key resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyTarget {
    /// The compiled walk index of the neuron it names.
    Index(usize),
    /// A numeric key that could be a uuid-derived runtime id. Neither pruned
    /// nor refused: this half of rule 31's resolution needs the hash that lives
    /// in `neat-core`, and guessing wrong would throw away valid fine-tuning
    /// history or refuse a sound creature. Erring here costs at worst the old
    /// behaviour for that one key; erring the other way loses data.
    Unverifiable,
    /// The key names nothing this creature carries.
    Missing,
}

/// Everything a memetic key can name, resolved once per creature.
struct StructureIndex<'a> {
    /// Listed neurons by wire uuid, at their compiled walk index.
    uuid_to_index: HashMap<&'a str, usize>,
    /// Neurons by the runtime id the shared rules give them, at their compiled
    /// walk index — implicit inputs by their own index, outputs by
    /// `-(outputIndex + 1)` whatever the file declares, everything else by its
    /// declared id.
    id_to_index: HashMap<i64, usize>,
    /// Declared observation width — the implicit `input-N` range.
    input: usize,
    /// Live `(from, to)` compiled index pairs.
    pairs: HashSet<(usize, usize)>,
}

impl<'a> StructureIndex<'a> {
    fn build(creature: &'a CreatureExport) -> Self {
        let mut uuid_to_index = HashMap::with_capacity(creature.neurons.len());
        let mut id_to_index = HashMap::with_capacity(creature.input + creature.neurons.len());
        for index in 0..creature.input {
            id_to_index.insert(index as i64, index);
        }
        let mut output_index: i64 = 0;
        for (i, neuron) in creature.neurons.iter().enumerate() {
            let index = creature.input + i;
            uuid_to_index.insert(neuron.uuid.as_str(), index);
            if neuron.neuron_type == "output" {
                id_to_index.insert(-(output_index + 1), index);
                output_index += 1;
            } else if let Some(id) = neuron.id {
                id_to_index.insert(id, index);
            }
        }
        let mut index = Self {
            uuid_to_index,
            id_to_index,
            input: creature.input,
            pairs: HashSet::with_capacity(creature.synapses.len()),
        };
        for synapse in &creature.synapses {
            if let (Some(from), Some(to)) = (
                index.resolve_wire(&synapse.from_uuid),
                index.resolve_wire(&synapse.to_uuid),
            ) {
                index.pairs.insert((from, to));
            }
        }
        index
    }

    /// The neuron a wire uuid names — a listed neuron, or an implicit input.
    fn resolve_wire(&self, uuid: &str) -> Option<usize> {
        if let Some(index) = self.uuid_to_index.get(uuid) {
            return Some(*index);
        }
        let index: usize = uuid.strip_prefix("input-")?.parse().ok()?;
        (index < self.input).then_some(index)
    }

    /// The neuron a memetic key names — runtime id first, then wire uuid.
    fn resolve_key(&self, key: &str) -> KeyTarget {
        if let Ok(id) = key.parse::<i64>() {
            if let Some(index) = self.id_to_index.get(&id) {
                return KeyTarget::Index(*index);
            }
            if let Some(index) = self.resolve_wire(key) {
                return KeyTarget::Index(index);
            }
            return self.unmatched_id(id);
        }
        match self.resolve_wire(key) {
            Some(index) => KeyTarget::Index(index),
            None => KeyTarget::Missing,
        }
    }

    /// A runtime id no neuron declares: dangling, unless it could be one of the
    /// uuid-derived ids only `neat-core` can compute.
    fn unmatched_id(&self, id: i64) -> KeyTarget {
        if DERIVED_ID_RANGE.contains(&id) {
            KeyTarget::Unverifiable
        } else {
            KeyTarget::Missing
        }
    }

    /// Whether the edge between two resolved endpoints is gone. Unverifiable at
    /// either end means the question cannot be answered — treated as live.
    fn edge_dangles(&self, from: KeyTarget, to: KeyTarget) -> bool {
        match (from, to) {
            (KeyTarget::Missing, _) | (_, KeyTarget::Missing) => true,
            (KeyTarget::Index(from), KeyTarget::Index(to)) => !self.pairs.contains(&(from, to)),
            _ => false,
        }
    }

    /// Whether a bias key names a neuron the creature no longer carries.
    fn bias_dangles(&self, key: &str) -> bool {
        self.resolve_key(key) == KeyTarget::Missing
    }

    /// Whether a row names an endpoint or an edge the creature no longer
    /// carries. A row missing a field is malformed rather than dangling — rule
    /// 31 reports that as the input defect it is, and pruning it would hide it.
    fn row_dangles(&self, row: &MemeticWeightRowExport) -> bool {
        let (Some(from_uuid), Some(to_uuid)) = (row.from_uuid.as_deref(), row.to_uuid.as_deref())
        else {
            return false;
        };
        self.edge_dangles(self.resolve_key(from_uuid), self.resolve_key(to_uuid))
    }

    /// Whether an id-keyed entry names an edge the creature no longer carries.
    /// `toId` resolves by runtime id only, exactly as rule 31 reads it.
    fn entry_dangles(&self, from: KeyTarget, entry: &MemeticWeightExport) -> bool {
        let Some(to_id) = entry.to_id else {
            return false;
        };
        let to = match self.id_to_index.get(&to_id) {
            Some(&index) => KeyTarget::Index(index),
            None => self.unmatched_id(to_id),
        };
        self.edge_dangles(from, to)
    }
}

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
    let Some(mut memetic) = creature.memetic.take() else {
        return;
    };
    let index = StructureIndex::build(creature);
    prune_record(&mut memetic, &index);
    creature.memetic = Some(memetic);
}

/// [`prune_memetic`]'s body, against an already-built index.
fn prune_record(memetic: &mut MemeticExport, index: &StructureIndex<'_>) {
    memetic.biases.retain(|key, _| !index.bias_dangles(key));
    match &mut memetic.weights {
        neat_core::MemeticWeights::Rows(rows) => rows.retain(|row| !index.row_dangles(row)),
        neat_core::MemeticWeights::ById(by_id) => {
            by_id.retain(|key, entries| {
                let from = index.resolve_key(key);
                if from == KeyTarget::Missing {
                    return false;
                }
                entries.retain(|entry| !index.entry_dangles(from, entry));
                true
            });
        }
    }
}

/// Refuse a creature whose memetic record names structure it no longer carries.
///
/// The write-path half of the rule: [`prune_memetic`] stops the dangling record
/// being *built*, this stops one being *written*. Called from every creature
/// write path (`width::checked_creature_json*`, the `tags` check-in
/// serialisers), so the failure points at the writer, not at a later reader.
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
    let index = StructureIndex::build(creature);

    if let Some(key) = memetic.biases.keys().find(|key| index.bias_dangles(key)) {
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
    match &memetic.weights {
        neat_core::MemeticWeights::Rows(rows) => {
            if let Some(row) = rows.iter().find(|row| index.row_dangles(row)) {
                return Err(dangling_edge(
                    row.from_uuid.clone().unwrap_or_default(),
                    row.to_uuid.clone().unwrap_or_default(),
                ));
            }
        }
        neat_core::MemeticWeights::ById(by_id) => {
            for (key, entries) in by_id {
                let from = index.resolve_key(key);
                if from == KeyTarget::Missing {
                    return Err(format!(
                        "refusing to write creature: memetic weight key {key} names no neuron in \
                         the creature — a removal did not prune the memetic record"
                    ));
                }
                if let Some(entry) = entries
                    .iter()
                    .find(|entry| index.entry_dangles(from, entry))
                {
                    return Err(dangling_edge(
                        key.clone(),
                        entry.to_id.map(|id| id.to_string()).unwrap_or_default(),
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::{MemeticWeights, parse_creature_json};

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
        "weights":{"2":[{"toId":-1,"weight":1.1}],"1500000":[{"toId":-1,"weight":0.7}]}
      }
    }"#;

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
        match &kept.memetic.as_ref().unwrap().weights {
            MemeticWeights::ById(by_id) => assert_eq!(by_id["2"].len(), 1),
            MemeticWeights::Rows(_) => panic!("fixture is the map form"),
        }

        let mut removed = parse_creature_json(DERIVED_IDS).unwrap();
        removed.synapses.retain(|s| s.to_uuid != "o1");
        prune_memetic(&mut removed);
        match &removed.memetic.as_ref().unwrap().weights {
            MemeticWeights::ById(by_id) => {
                assert!(by_id["2"].is_empty(), "the removed edge's entry is pruned")
            }
            MemeticWeights::Rows(_) => panic!("fixture is the map form"),
        }
    }

    /// A numeric key in NEAT-AI's uuid-derived id range names a neuron only
    /// `neat-core` can identify. Lamarck neither prunes it nor refuses it: a
    /// wrong guess would destroy valid history or block a sound write.
    #[test]
    fn prune_keeps_a_key_that_could_be_a_uuid_derived_id() {
        let mut creature = parse_creature_json(DERIVED_IDS).unwrap();
        prune_memetic(&mut creature);
        match &creature.memetic.as_ref().unwrap().weights {
            MemeticWeights::ById(by_id) => assert!(
                by_id.contains_key("1500000"),
                "an unverifiable key must survive: {by_id:?}"
            ),
            MemeticWeights::Rows(_) => panic!("fixture is the map form"),
        }
        assert_memetic_resolves(&creature).expect("an unverifiable key must not block a write");
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
        let by_id = match &creature.memetic.as_ref().unwrap().weights {
            MemeticWeights::ById(by_id) => by_id,
            MemeticWeights::Rows(_) => panic!("the map form must be written back as a map"),
        };
        // Key "77" names no neuron; entry `toId: 99` names no neuron.
        assert_eq!(by_id.len(), 1);
        assert_eq!(by_id["2"].len(), 1);
        assert_eq!(by_id["2"][0].to_id, Some(-1));
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
}
