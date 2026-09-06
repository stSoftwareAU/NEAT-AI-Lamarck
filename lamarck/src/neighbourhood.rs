//! Bounded focus-neighbourhood expansion (issue #222).
//!
//! A focus neuron that keeps producing accepted candidates is evidence that
//! something useful lives *there* — but the useful thing may be a small local
//! subgraph rather than one neuron in isolation. This module turns a repeatedly
//! successful focus into a small, bounded region of the creature — the focus
//! itself and the neurons reached from it along the highest-impact edges, with
//! any neuron an accepted structural mutation recently grew taken first — whose
//! members the ordinary candidate generator can then target in turn. Growth is
//! a ranking preference, not an exemption: a grown neuron outside the region's
//! radius is as far away as any other.
//!
//! Four properties keep the expansion honest:
//!
//! * **Evidence-driven.** A region is derived only once the root focus has
//!   earned [`NeighbourhoodLimits::accepts`] measured acceptances, or under the
//!   explicit `0` policy arm. Nothing here encodes domain knowledge about which
//!   neurons matter.
//! * **Bounded.** [`NeighbourhoodLimits`] caps the members, the edges the region
//!   spans and the graph radius it reaches. A region cannot grow into a broad
//!   section of the network.
//! * **Never monopolising.** Each root may steer only
//!   [`NeighbourhoodLimits::experiments`] experiments per accept it earned, and
//!   the root itself is still drawn by the ordinary focus policy every
//!   experiment — so random / control focus selection is untouched.
//! * **Attributable.** Every candidate proposed against a member carries a
//!   [`NeighbourhoodLink`] naming the root focus and the member actually
//!   targeted, so a win is attributable to the original focus or to adjacent
//!   structure rather than to the region as a whole.

use neat_core::CreatureExport;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hard limits on one focus-neighbourhood expansion.
///
/// All four bind. [`Self::neurons`] is the master switch: `0` is the pre-#222
/// run, where a focus is always treated as an isolated scalar target. A zero
/// [`Self::edges`] or [`Self::radius`] can reach nothing either, so it derives
/// no region — which is why [`crate::config::LamarckConfig`] rejects those two
/// outright rather than letting a run silently lose the arm it was set for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighbourhoodLimits {
    /// Adjacent neurons one region may hold, over and above the root focus.
    pub neurons: usize,
    /// Edges one expansion may traverse into the region.
    pub edges: usize,
    /// Graph hops the region may reach from the root focus.
    pub radius: usize,
    /// Measured acceptances the root must have earned before it expands.
    ///
    /// `0` is the explicit-policy arm: expand every drawn focus, whatever it
    /// has earned. Any higher value is the evidence trigger.
    pub accepts: u32,
    /// Experiments one root may steer per accept it has earned.
    pub experiments: usize,
}

/// Where a region member sits relative to the root focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeighbourhoodRole {
    /// The root focus itself.
    Root,
    /// Upstream of the neuron it was reached from.
    Predecessor,
    /// Downstream of the neuron it was reached from.
    Successor,
}

/// One neuron of a derived region.
#[derive(Debug, Clone, PartialEq)]
pub struct NeighbourhoodMember {
    /// Neuron uuid.
    pub uuid: String,
    /// Graph hops from the root focus (`0` only for the root).
    pub hops: usize,
    /// Direction the member was reached in, relative to the neuron it was
    /// reached *from* — which is the root at one hop, and an inner member
    /// beyond that.
    pub role: NeighbourhoodRole,
    /// True when an accepted structural mutation grew this neuron.
    pub grown: bool,
}

/// One edge the region spans, in the order it was traversed.
#[derive(Debug, Clone, PartialEq)]
pub struct NeighbourhoodEdge {
    /// Source neuron uuid.
    pub from_uuid: String,
    /// Destination neuron uuid.
    pub to_uuid: String,
    /// Weight of the synapse, the impact the ranking is by.
    pub weight: f64,
}

/// Provenance link from a candidate back to the region it was proposed in.
///
/// Stamped on every candidate of an expanded experiment — the root's own
/// candidates included, with [`NeighbourhoodRole::Root`] — so the journal says
/// whether a win landed on the original focus or on adjacent structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeighbourhoodLink {
    /// The focus the region was derived from.
    pub root_focus: String,
    /// The member this candidate actually targeted.
    pub member: String,
    /// Graph hops from the root to the member.
    pub hops: usize,
    /// Direction the member was reached in, relative to the neuron it was
    /// reached *from*: the root itself at one hop, an inner member beyond that.
    pub role: NeighbourhoodRole,
}

/// One experiment's expansion, as journalled (issue #222).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeighbourhoodExpansion {
    /// The focus the region was derived from.
    pub root_focus: String,
    /// Measured acceptances the root had earned when it expanded.
    pub accepts: u32,
    /// Adjacent members the region holds, in traversal order.
    pub members: Vec<String>,
    /// The members an accepted structural mutation grew.
    ///
    /// Growth is the one signal the ranking treats as decisive, so a reader can
    /// see whether a region was built around structure the scorer had already
    /// paid for or around the creature's existing wiring. Omitted when the
    /// region holds no grown member.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grown: Vec<String>,
    /// Edges the region spans.
    pub edges: usize,
    /// Greatest hop count any member sits at.
    pub radius: usize,
    /// Experiments this root may still steer on the evidence it has now.
    pub remaining: usize,
}

/// A bounded local region derived from one root focus.
#[derive(Debug, Clone, PartialEq)]
pub struct FocusNeighbourhood {
    root: String,
    accepts: u32,
    members: Vec<NeighbourhoodMember>,
    edges: Vec<NeighbourhoodEdge>,
    remaining: usize,
}

impl FocusNeighbourhood {
    /// The focus this region was derived from.
    pub fn root(&self) -> &str {
        &self.root
    }

    /// The adjacent members, in traversal order (the root is not among them).
    pub fn members(&self) -> &[NeighbourhoodMember] {
        &self.members
    }

    /// The edges the region spans.
    pub fn edges(&self) -> &[NeighbourhoodEdge] {
        &self.edges
    }

    /// Every neuron the generator may target, root first.
    pub fn targets(&self) -> Vec<String> {
        let mut targets = Vec::with_capacity(self.members.len() + 1);
        targets.push(self.root.clone());
        targets.extend(self.members.iter().map(|member| member.uuid.clone()));
        targets
    }

    /// Provenance for a candidate proposed against `uuid`, or `None` when the
    /// uuid is not in the region at all.
    pub fn link(&self, uuid: &str) -> Option<NeighbourhoodLink> {
        if uuid == self.root {
            return Some(NeighbourhoodLink {
                root_focus: self.root.clone(),
                member: self.root.clone(),
                hops: 0,
                role: NeighbourhoodRole::Root,
            });
        }
        self.members
            .iter()
            .find(|member| member.uuid == uuid)
            .map(|member| NeighbourhoodLink {
                root_focus: self.root.clone(),
                member: member.uuid.clone(),
                hops: member.hops,
                role: member.role,
            })
    }

    /// The journal record for this expansion.
    pub fn record(&self) -> NeighbourhoodExpansion {
        NeighbourhoodExpansion {
            root_focus: self.root.clone(),
            accepts: self.accepts,
            members: self.members.iter().map(|m| m.uuid.clone()).collect(),
            grown: self
                .members
                .iter()
                .filter(|m| m.grown)
                .map(|m| m.uuid.clone())
                .collect(),
            edges: self.edges.len(),
            radius: self.members.iter().map(|m| m.hops).max().unwrap_or(0),
            remaining: self.remaining,
        }
    }
}

/// What one root has spent, and on what evidence it was granted.
#[derive(Debug, Clone, Copy, Default)]
struct RootAllowance {
    /// Accept count the current allowance was granted against.
    credited_accepts: u32,
    /// Expansions already spent against that evidence.
    spent: usize,
}

/// Per-run bookkeeping that keeps expansion from monopolising the run.
///
/// A root's allowance is renewed only by *fresh* evidence: once it has spent
/// [`NeighbourhoodLimits::experiments`] expansions it stops expanding until it
/// earns another acceptance, at which point the run may explore its region
/// again. The ordinary focus policy keeps drawing roots meanwhile, so a run can
/// never be pinned to one region.
#[derive(Debug, Clone)]
pub struct NeighbourhoodLedger {
    limits: NeighbourhoodLimits,
    allowances: HashMap<String, RootAllowance>,
}

impl NeighbourhoodLedger {
    /// A ledger enforcing `limits`.
    pub fn new(limits: NeighbourhoodLimits) -> Self {
        Self {
            limits,
            allowances: HashMap::new(),
        }
    }

    /// The limits in force.
    pub fn limits(&self) -> NeighbourhoodLimits {
        self.limits
    }

    /// Expand around `root` when its measured evidence and allowance permit.
    ///
    /// `accepts` is the acceptances the run has credited to `root` so far and
    /// `grown` the neurons accepted structural mutations recently inserted. A
    /// grown neighbour is taken ahead of every plain one, however heavy that
    /// one's edge: the scorer has already paid for the grown structure, which
    /// is the strongest local evidence available. It still has to lie inside
    /// the region's radius to join at all. Returns
    /// `None` when the trigger is unmet, when the root has spent its allowance,
    /// or when the creature offers no adjacent neuron to target.
    ///
    /// An expansion that returns a region is counted against the allowance, so
    /// this must be called once per experiment.
    pub fn expand(
        &mut self,
        creature: &CreatureExport,
        root: &str,
        accepts: u32,
        grown: &[String],
    ) -> Option<FocusNeighbourhood> {
        if self.limits.neurons == 0 || self.limits.edges == 0 || self.limits.radius == 0 {
            return None;
        }
        if accepts < self.limits.accepts {
            return None;
        }
        let allowance = self.allowances.entry(root.to_string()).or_default();
        if accepts > allowance.credited_accepts {
            // Fresh evidence renews the allowance; nothing else does.
            allowance.credited_accepts = accepts;
            allowance.spent = 0;
        }
        if allowance.spent >= self.limits.experiments {
            return None;
        }

        let (members, edges) = derive_region(creature, root, grown, &self.limits);
        if members.is_empty() {
            return None;
        }
        allowance.spent += 1;
        let remaining = self.limits.experiments.saturating_sub(allowance.spent);
        Some(FocusNeighbourhood {
            root: root.to_string(),
            accepts,
            members,
            edges,
            remaining,
        })
    }
}

/// Breadth-first region growth, bounded by every limit at once.
///
/// Edges are considered in impact order: a neuron an accepted structural
/// mutation grew first — the scorer has already paid for it, so it outranks a
/// heavier plain neighbour outright — then descending `|weight|`, then by uuid
/// so the walk is deterministic. Each admitted edge spends the edge budget, and
/// admits its endpoint as a member while the neuron budget allows.
///
/// An edge whose far endpoint cannot be a focus — an input neuron, or a
/// dangling uuid no neuron declares — is skipped without spending the edge
/// budget. Spending it there would let a focus fed by several strong input
/// edges exhaust the budget before reaching a single targetable neighbour, and
/// silently never expand at all.
fn derive_region(
    creature: &CreatureExport,
    root: &str,
    grown: &[String],
    limits: &NeighbourhoodLimits,
) -> (Vec<NeighbourhoodMember>, Vec<NeighbourhoodEdge>) {
    let inputs: std::collections::HashSet<&str> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "input")
        .map(|n| n.uuid.as_str())
        .collect();
    let known: std::collections::HashSet<&str> =
        creature.neurons.iter().map(|n| n.uuid.as_str()).collect();

    let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
    visited.insert(root.to_string());
    let mut members: Vec<NeighbourhoodMember> = Vec::new();
    let mut edges: Vec<NeighbourhoodEdge> = Vec::new();
    let mut frontier = vec![root.to_string()];

    for hops in 1..=limits.radius {
        if frontier.is_empty() || members.len() >= limits.neurons || edges.len() >= limits.edges {
            break;
        }
        let mut ranked: Vec<(NeighbourhoodEdge, String, NeighbourhoodRole)> = Vec::new();
        let targetable = |uuid: &str| known.contains(uuid) && !inputs.contains(uuid);
        for synapse in &creature.synapses {
            let edge = NeighbourhoodEdge {
                from_uuid: synapse.from_uuid.clone(),
                to_uuid: synapse.to_uuid.clone(),
                weight: synapse.weight,
            };
            if frontier.iter().any(|f| f == &synapse.to_uuid)
                && !visited.contains(&synapse.from_uuid)
                && targetable(&synapse.from_uuid)
            {
                ranked.push((
                    edge.clone(),
                    synapse.from_uuid.clone(),
                    NeighbourhoodRole::Predecessor,
                ));
            }
            if frontier.iter().any(|f| f == &synapse.from_uuid)
                && !visited.contains(&synapse.to_uuid)
                && targetable(&synapse.to_uuid)
            {
                ranked.push((edge, synapse.to_uuid.clone(), NeighbourhoodRole::Successor));
            }
        }
        ranked.sort_by(|a, b| {
            let a_grown = grown.iter().any(|uuid| uuid == &a.1);
            let b_grown = grown.iter().any(|uuid| uuid == &b.1);
            b_grown
                .cmp(&a_grown)
                .then_with(|| {
                    b.0.weight
                        .abs()
                        .partial_cmp(&a.0.weight.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.1.cmp(&b.1))
        });

        let mut next = Vec::new();
        for (edge, other, role) in ranked {
            if edges.len() >= limits.edges || members.len() >= limits.neurons {
                break;
            }
            if visited.contains(&other) {
                continue;
            }
            edges.push(edge);
            visited.insert(other.clone());
            members.push(NeighbourhoodMember {
                uuid: other.clone(),
                hops,
                role,
                grown: grown.iter().any(|uuid| uuid == &other),
            });
            next.push(other);
        }
        frontier = next;
    }

    (members, edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;

    /// `in0 → h1 → h2 → o1`, with `in1 → h2` and a strong `h1 → o1` shortcut.
    const CREATURE: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 2,
      "output": 1,
      "neurons": [
        {"type":"hidden","uuid":"h1","bias":0.1,"squash":"IDENTITY"},
        {"type":"hidden","uuid":"h2","bias":0.2,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h1","weight":0.9},
        {"fromUUID":"input-1","toUUID":"h2","weight":0.05},
        {"fromUUID":"h1","toUUID":"h2","weight":0.4},
        {"fromUUID":"h2","toUUID":"o1","weight":0.3},
        {"fromUUID":"h1","toUUID":"o1","weight":0.8}
      ]
    }"#;

    fn creature() -> CreatureExport {
        parse_creature_json(CREATURE).expect("the test creature parses")
    }

    fn limits() -> NeighbourhoodLimits {
        NeighbourhoodLimits {
            neurons: 2,
            edges: 4,
            radius: 1,
            accepts: 1,
            experiments: 1,
        }
    }

    /// Expansion is off by default: a zero neuron cap derives nothing.
    #[test]
    fn a_zero_neuron_cap_never_expands() {
        let mut ledger = NeighbourhoodLedger::new(NeighbourhoodLimits {
            neurons: 0,
            ..limits()
        });
        assert!(ledger.expand(&creature(), "o1", 9, &[]).is_none());
    }

    /// A focus with no measured acceptance is not expanded around.
    #[test]
    fn a_focus_without_measured_success_is_not_expanded() {
        let mut ledger = NeighbourhoodLedger::new(limits());
        assert!(
            ledger.expand(&creature(), "o1", 0, &[]).is_none(),
            "no accepts, no evidence, no region"
        );
        assert!(
            ledger.expand(&creature(), "o1", 1, &[]).is_some(),
            "one accept meets the trigger"
        );
    }

    /// The explicit-policy arm expands whatever the focus has earned.
    #[test]
    fn a_zero_accept_trigger_is_the_explicit_policy_arm() {
        let mut ledger = NeighbourhoodLedger::new(NeighbourhoodLimits {
            accepts: 0,
            ..limits()
        });
        assert!(ledger.expand(&creature(), "o1", 0, &[]).is_some());
    }

    /// The region is the focus plus its highest-impact direct neighbours.
    #[test]
    fn the_region_holds_the_root_and_its_direct_neighbours() {
        let mut ledger = NeighbourhoodLedger::new(limits());
        let region = ledger.expand(&creature(), "o1", 1, &[]).expect("a region");
        assert_eq!(region.root(), "o1");
        let members: Vec<&str> = region
            .members()
            .iter()
            .map(|member| member.uuid.as_str())
            .collect();
        // Both feed `o1`; the stronger edge (`h1`, 0.8) ranks first.
        assert_eq!(members, vec!["h1", "h2"]);
        assert!(
            region
                .members()
                .iter()
                .all(|member| member.role == NeighbourhoodRole::Predecessor && member.hops == 1)
        );
        assert_eq!(region.targets(), vec!["o1", "h1", "h2"]);
    }

    /// Successors are region members too, not just upstream sources.
    #[test]
    fn successors_join_the_region() {
        let mut ledger = NeighbourhoodLedger::new(limits());
        let region = ledger.expand(&creature(), "h1", 1, &[]).expect("a region");
        let roles: Vec<(&str, NeighbourhoodRole)> = region
            .members()
            .iter()
            .map(|member| (member.uuid.as_str(), member.role))
            .collect();
        assert_eq!(
            roles,
            vec![
                ("o1", NeighbourhoodRole::Successor),
                ("h2", NeighbourhoodRole::Successor),
            ],
            "both of h1's downstream neurons, strongest edge first"
        );
    }

    /// The neuron cap is hard: a larger creature cannot widen the region.
    #[test]
    fn the_neuron_cap_bounds_the_region() {
        let mut ledger = NeighbourhoodLedger::new(NeighbourhoodLimits {
            neurons: 1,
            ..limits()
        });
        let region = ledger.expand(&creature(), "o1", 1, &[]).expect("a region");
        assert_eq!(region.members().len(), 1);
        assert_eq!(region.members()[0].uuid, "h1", "the strongest edge wins");
    }

    /// The edge cap is hard: it bounds the edges the region spans.
    #[test]
    fn the_edge_cap_bounds_the_region() {
        let mut ledger = NeighbourhoodLedger::new(NeighbourhoodLimits {
            neurons: 4,
            edges: 1,
            radius: 2,
            ..limits()
        });
        let region = ledger.expand(&creature(), "o1", 1, &[]).expect("a region");
        assert_eq!(region.edges().len(), 1);
        assert_eq!(region.members().len(), 1);
    }

    /// A focus fed by heavy input edges still expands (issue #222).
    ///
    /// An input can never be a focus, so an edge into one buys the region
    /// nothing. Spending the edge budget there let a first-layer focus — the
    /// very shape this feature was built for — exhaust its budget before
    /// reaching a single targetable neighbour, and silently never expand.
    #[test]
    fn heavy_input_edges_do_not_consume_the_edge_budget() {
        let fed: CreatureExport = parse_creature_json(
            r#"{
              "semanticVersion": "4.0.0",
              "forwardOnly": true,
              "input": 2,
              "output": 1,
              "neurons": [
                {"type":"hidden","uuid":"h1","bias":0.1,"squash":"IDENTITY"},
                {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
              ],
              "synapses": [
                {"fromUUID":"input-0","toUUID":"h1","weight":0.95},
                {"fromUUID":"input-1","toUUID":"h1","weight":0.9},
                {"fromUUID":"h1","toUUID":"o1","weight":0.4}
              ]
            }"#,
        )
        .expect("the input-fed creature parses");
        let mut ledger = NeighbourhoodLedger::new(NeighbourhoodLimits {
            neurons: 4,
            edges: 1,
            radius: 1,
            ..limits()
        });
        let region = ledger
            .expand(&fed, "h1", 1, &[])
            .expect("the two heavier input edges must not starve the region");
        let members: Vec<&str> = region
            .members()
            .iter()
            .map(|member| member.uuid.as_str())
            .collect();
        assert_eq!(members, vec!["o1"], "the one targetable neighbour joins");
        assert_eq!(
            region.edges().len(),
            1,
            "only the edge that admitted a member spends the budget"
        );
    }

    /// The radius is hard: a two-hop neuron never joins a radius-1 region.
    #[test]
    fn the_radius_bounds_the_region() {
        // input-0 → a → b → c → o1: one targetable neuron per hop from `o1`.
        let chain: CreatureExport = parse_creature_json(
            r#"{
              "semanticVersion": "4.0.0",
              "forwardOnly": true,
              "input": 1,
              "output": 1,
              "neurons": [
                {"type":"hidden","uuid":"a","bias":0.0,"squash":"IDENTITY"},
                {"type":"hidden","uuid":"b","bias":0.0,"squash":"IDENTITY"},
                {"type":"hidden","uuid":"c","bias":0.0,"squash":"IDENTITY"},
                {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
              ],
              "synapses": [
                {"fromUUID":"input-0","toUUID":"a","weight":1.0},
                {"fromUUID":"a","toUUID":"b","weight":0.7},
                {"fromUUID":"b","toUUID":"c","weight":0.6},
                {"fromUUID":"c","toUUID":"o1","weight":0.5}
              ]
            }"#,
        )
        .expect("the chain creature parses");
        let region_at = |radius: usize| {
            NeighbourhoodLedger::new(NeighbourhoodLimits {
                neurons: 4,
                edges: 8,
                radius,
                accepts: 1,
                experiments: 1,
            })
            .expand(&chain, "o1", 1, &[])
            .expect("a region")
        };

        let one_hop = region_at(1);
        let members: Vec<&str> = one_hop
            .members()
            .iter()
            .map(|member| member.uuid.as_str())
            .collect();
        assert_eq!(members, vec!["c"], "only the direct predecessor");

        let two_hop = region_at(2);
        let members: Vec<(&str, usize)> = two_hop
            .members()
            .iter()
            .map(|member| (member.uuid.as_str(), member.hops))
            .collect();
        assert_eq!(
            members,
            vec![("c", 1), ("b", 2)],
            "the second hop reaches exactly one neuron further"
        );
        assert!(
            two_hop.edges().len() > one_hop.edges().len(),
            "the second hop spans one more edge"
        );
        assert!(
            !two_hop.members().iter().any(|member| member.uuid == "a"),
            "the three-hop neuron stays outside a radius-2 region"
        );
    }

    /// A neuron an accepted mutation grew outranks a heavier plain edge.
    #[test]
    fn a_grown_neuron_is_preferred_over_a_heavier_neighbour() {
        let mut ledger = NeighbourhoodLedger::new(NeighbourhoodLimits {
            neurons: 1,
            ..limits()
        });
        let grown = vec!["h2".to_string()];
        let region = ledger
            .expand(&creature(), "o1", 1, &grown)
            .expect("a region");
        assert_eq!(region.members()[0].uuid, "h2");
        assert!(region.members()[0].grown, "the member is marked as grown");
        assert_eq!(
            region.record().grown,
            vec!["h2".to_string()],
            "the journal says the region was built around grown structure"
        );
    }

    /// The ledger reports the limits it enforces, so a caller can bound its own
    /// bookkeeping by them rather than by a second copy of the numbers.
    #[test]
    fn the_ledger_reports_its_limits() {
        let ledger = NeighbourhoodLedger::new(limits());
        assert_eq!(ledger.limits(), limits());
    }

    /// Every candidate target resolves to provenance naming root and member.
    #[test]
    fn provenance_names_the_root_and_the_targeted_member() {
        let mut ledger = NeighbourhoodLedger::new(limits());
        let region = ledger.expand(&creature(), "o1", 2, &[]).expect("a region");

        let root = region.link("o1").expect("the root is in its own region");
        assert_eq!(root.root_focus, "o1");
        assert_eq!(root.member, "o1");
        assert_eq!(root.hops, 0);
        assert_eq!(root.role, NeighbourhoodRole::Root);

        let member = region.link("h1").expect("h1 is a member");
        assert_eq!(member.root_focus, "o1");
        assert_eq!(member.member, "h1");
        assert_eq!(member.hops, 1);
        assert_eq!(member.role, NeighbourhoodRole::Predecessor);

        assert!(region.link("nobody").is_none(), "a stranger has no link");
    }

    /// The journal record carries the region and the evidence behind it.
    #[test]
    fn the_journal_record_carries_the_region() {
        let mut ledger = NeighbourhoodLedger::new(NeighbourhoodLimits {
            experiments: 3,
            ..limits()
        });
        let record = ledger
            .expand(&creature(), "o1", 4, &[])
            .expect("a region")
            .record();
        assert_eq!(record.root_focus, "o1");
        assert_eq!(record.accepts, 4);
        assert_eq!(record.members, vec!["h1".to_string(), "h2".to_string()]);
        assert!(record.grown.is_empty(), "nothing here was grown");
        assert_eq!(record.edges, 2);
        assert_eq!(record.radius, 1);
        assert_eq!(record.remaining, 2, "one of three expansions spent");
    }

    /// One root cannot monopolise the run: its allowance runs out.
    #[test]
    fn a_root_cannot_expand_beyond_its_allowance() {
        let mut ledger = NeighbourhoodLedger::new(NeighbourhoodLimits {
            experiments: 2,
            ..limits()
        });
        let creature = creature();
        assert!(ledger.expand(&creature, "o1", 1, &[]).is_some());
        assert!(ledger.expand(&creature, "o1", 1, &[]).is_some());
        assert!(
            ledger.expand(&creature, "o1", 1, &[]).is_none(),
            "the allowance is spent until fresh evidence arrives"
        );
        // Another focus is unaffected — expansion is per root.
        assert!(ledger.expand(&creature, "h1", 1, &[]).is_some());
    }

    /// A further acceptance renews the exhausted allowance, nothing else does.
    #[test]
    fn a_fresh_accept_renews_the_allowance() {
        let mut ledger = NeighbourhoodLedger::new(limits());
        let creature = creature();
        assert!(ledger.expand(&creature, "o1", 1, &[]).is_some());
        assert!(ledger.expand(&creature, "o1", 1, &[]).is_none());
        assert!(
            ledger.expand(&creature, "o1", 2, &[]).is_some(),
            "a second accept buys a second expansion"
        );
        assert!(ledger.expand(&creature, "o1", 2, &[]).is_none());
    }

    /// A focus with no adjacent neuron to target yields no region at all.
    #[test]
    fn an_isolated_focus_yields_no_region() {
        let isolated: CreatureExport = parse_creature_json(
            r#"{
              "semanticVersion": "4.0.0",
              "forwardOnly": true,
              "input": 1,
              "output": 1,
              "neurons": [{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
              "synapses": [{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
            }"#,
        )
        .expect("the isolated creature parses");
        let mut ledger = NeighbourhoodLedger::new(limits());
        assert!(
            ledger.expand(&isolated, "o1", 3, &[]).is_none(),
            "only an input neighbour, which can never be a focus"
        );
    }
}
