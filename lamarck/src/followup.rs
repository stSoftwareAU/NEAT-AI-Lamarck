//! Bounded local follow-up search around an accepted mutation (issue #219).
//!
//! Unrelated to the #75 *follow-up economics campaign* (`docs/followup-economics.md`,
//! `scripts/run-followup-economics.sh`), which is a set of shell-driven
//! measurement arms. This module is the in-run search.
//!
//! An accepted candidate is stronger evidence than a merely useful focus: the
//! scorer has just confirmed real gradient or structure at one place in the
//! creature. This module turns that win into a small, finite set of neighbouring
//! hypotheses — nearby weight scales, an alternate squash for a grown neuron,
//! a partial back-off of the winning step — and hands them to the ordinary
//! candidate batch.
//!
//! Three properties keep the burst honest:
//!
//! * **Bounded.** [`FollowUpBudget`] caps both the candidates one win may spend
//!   and the experiments its burst may span. Nothing here can run long.
//! * **Exploitation, never acceptance.** A follow-up candidate is an ordinary
//!   member of the batch. It is written into the same scoring directory, passes
//!   the same screen, and is accepted only by the same full-corpus scorer gate.
//!   There is deliberately no path from this module to an acceptance.
//! * **Additive, not displacing.** Follow-ups join the batch beside the
//!   ordinary strategy mix rather than replacing it, so the random controls the
//!   generator would have proposed are still proposed. A winner's neighbourhood
//!   is not assumed smooth.

use crate::backprop::BackpropConfig;
use crate::candidates::{Candidate, CandidateProvenance, CandidateStrategy, candidate_fingerprint};
use crate::structural::NEURON_GROWTH_SQUASHES;
use neat_core::CreatureExport;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};

/// Alternate squashes offered for one neuron the winner grew.
const SQUASH_ALTERNATIVES: usize = 3;

/// Step fractions applied to a scalar the winner introduced or moved.
///
/// `+0.5` doubles down half a step further along the winning direction, `-0.5`
/// backs half of it out (the weakening control), and `+1.0` takes a full second
/// step. All three are relative to the winning move, so a tiny accepted nudge
/// produces tiny probes and a large one produces large probes.
const STEP_FRACTIONS: [f64; 3] = [0.5, -0.5, 1.0];

/// The accepted winner a follow-up burst is exploring around.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FollowUpParent {
    /// Experiment number that accepted the winner.
    pub experiment: u64,
    /// Scoring stem of the accepted winner (`candidate-003`, `combo-002-k2`).
    pub winner_stem: String,
    /// Strategy of the accepted winner's first member.
    pub strategy: CandidateStrategy,
    /// Focus neuron credited with the accept.
    pub focus_neuron: String,
}

/// Hard caps on what one accepted win may spend on follow-ups.
///
/// Both caps bind: the burst stops at [`Self::candidates`] proposals whatever
/// happens, and expires after [`Self::experiments`] experiments even when
/// probes remain. The experiment cap is what bounds follow-up *time* per win —
/// an experiment is the run's unit of wall clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FollowUpBudget {
    /// Follow-up candidates one accepted win may emit in total.
    pub candidates: usize,
    /// Experiments the burst may span before the plan is dropped.
    pub experiments: usize,
}

/// Which scalar of the incumbent a probe steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeTarget {
    /// Bias of the neuron with this uuid.
    Bias {
        /// Neuron uuid.
        uuid: String,
    },
    /// Weight of the synapse between these uuids.
    Weight {
        /// Source neuron uuid.
        from_uuid: String,
        /// Destination neuron uuid.
        to_uuid: String,
    },
}

impl ProbeTarget {
    /// Stable axis name, matching the mirrored-sampling spelling (issue #203).
    pub fn axis(&self) -> String {
        match self {
            ProbeTarget::Bias { uuid } => format!("bias:{uuid}"),
            ProbeTarget::Weight { from_uuid, to_uuid } => {
                format!("weight:{from_uuid}->{to_uuid}")
            }
        }
    }
}

/// One neighbouring hypothesis around the accepted winner.
#[derive(Debug, Clone, PartialEq)]
pub enum FollowUpProbe {
    /// Step a scalar the winner introduced or moved.
    Scalar {
        /// Scalar to step.
        target: ProbeTarget,
        /// Additive step applied to the incumbent's current value.
        step: f64,
    },
    /// Re-squash a neuron the winner grew or re-squashed.
    Squash {
        /// Neuron uuid.
        uuid: String,
        /// Squash to try instead.
        squash: String,
    },
}

impl FollowUpProbe {
    /// Deduplication key — one probe per axis and step, per burst.
    fn key(&self) -> String {
        match self {
            FollowUpProbe::Scalar { target, step } => {
                format!("{} step {}", target.axis(), step.to_bits())
            }
            FollowUpProbe::Squash { uuid, squash } => format!("squash:{uuid} -> {squash}"),
        }
    }
}

/// Provenance link from a follow-up candidate back to the win it explores.
///
/// Stamped on the candidate, so the journal attributes every follow-up trial to
/// its parent winner without depending on batch order or experiment adjacency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FollowUpLink {
    /// Experiment number that accepted the parent winner.
    pub parent_experiment: u64,
    /// Scoring stem of the parent winner.
    pub parent_winner: String,
    /// Strategy of the parent winner.
    pub parent_strategy: CandidateStrategy,
    /// The neighbouring hypothesis this candidate tests.
    pub probe: String,
}

/// One experiment's slice of a follow-up burst, as journalled (issue #219).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FollowUpBurst {
    /// Experiment number that accepted the parent winner.
    pub parent_experiment: u64,
    /// Scoring stem of the parent winner.
    pub parent_winner: String,
    /// Follow-up candidates this experiment added to the batch.
    pub candidates: usize,
    /// Follow-up candidates the burst may still spend after this experiment.
    pub remaining: usize,
}

/// What the ordinary generator already put in this experiment's batch.
///
/// Held by reference so a probe is deduplicated against the real batch rather
/// than against a copy of it: [`Self::seen`] is updated as probes join, so two
/// probes cannot collide with each other either.
pub struct BatchContext<'a> {
    /// Neuron uuids of the creature this batch was proposed against.
    pub incumbent_uuids: &'a HashSet<String>,
    /// Structural fingerprints of the candidates already in the batch.
    pub seen: &'a mut HashSet<u64>,
    /// Perturbation axes retired against this incumbent (issue #203).
    pub dead_axes: &'a [String],
}

/// A bounded local search plan emitted by one accepted candidate.
#[derive(Debug, Clone)]
pub struct FollowUpPlan {
    parent: FollowUpParent,
    probes: Vec<FollowUpProbe>,
    next_probe: usize,
    spent: usize,
    experiments_used: usize,
    budget: FollowUpBudget,
    /// Probe keys already emitted, so a burst never re-tests its own hypothesis.
    tested: BTreeSet<String>,
}

impl FollowUpPlan {
    /// Plan a local search around the mutation that just won.
    ///
    /// `previous` is the creature the winner replaced and `winner` the new
    /// incumbent; the difference between them is the local region worth
    /// exploring. Returns `None` when the budget is off, when the pair differs
    /// in no way this module can probe, or when every probe was filtered out.
    pub fn from_accept(
        previous: &CreatureExport,
        winner: &CreatureExport,
        parent: FollowUpParent,
        budget: FollowUpBudget,
    ) -> Option<Self> {
        if budget.candidates == 0 || budget.experiments == 0 {
            return None;
        }
        let probes = plan_probes(previous, winner);
        if probes.is_empty() {
            return None;
        }
        Some(Self {
            parent,
            probes,
            next_probe: 0,
            spent: 0,
            experiments_used: 0,
            budget,
            tested: BTreeSet::new(),
        })
    }

    /// The winner this burst is exploring around.
    pub fn parent(&self) -> &FollowUpParent {
        &self.parent
    }

    /// Follow-up candidates the burst may still spend.
    pub fn remaining(&self) -> usize {
        self.budget.candidates.saturating_sub(self.spent)
    }

    /// True once no further follow-up candidate may be emitted.
    pub fn is_exhausted(&self) -> bool {
        self.remaining() == 0
            || self.experiments_used >= self.budget.experiments
            || self.next_probe >= self.probes.len()
    }

    /// Candidates this experiment may take, spreading the cap over the burst.
    ///
    /// Read *before* the experiment is counted, so the first of a two-experiment
    /// burst takes half the cap rather than all of it.
    fn slots(&self) -> usize {
        let experiments_left = self
            .budget
            .experiments
            .saturating_sub(self.experiments_used)
            .max(1);
        self.remaining().div_ceil(experiments_left)
    }

    /// Emit this experiment's follow-up candidates against `incumbent`.
    ///
    /// `batch` carries what the ordinary generator already put in this
    /// experiment's batch — the neuron-uuid set the fingerprints are keyed on,
    /// the fingerprints themselves, and the axes issue #203 has retired. A probe
    /// that reproduces a candidate the batch already holds, or that re-opens a
    /// retired axis, is dropped rather than spending a scorer slot on a question
    /// already asked or already answered.
    ///
    /// A probe that no longer applies — the uuid or edge is gone, the step
    /// would breach the hard bias/weight limit, or it lands within the plank
    /// constant of the current value — is consumed and skipped. Consuming it is
    /// deliberate: a probe that cannot apply must not be retried every
    /// experiment of the burst.
    pub fn emit(
        &mut self,
        incumbent: &CreatureExport,
        backprop: &BackpropConfig,
        batch: &mut BatchContext<'_>,
    ) -> Vec<Candidate> {
        let mut out = Vec::new();
        if self.is_exhausted() {
            return out;
        }
        let slots = self.slots();
        self.experiments_used += 1;
        while out.len() < slots && self.next_probe < self.probes.len() {
            let probe = self.probes[self.next_probe].clone();
            self.next_probe += 1;
            if !self.tested.insert(probe.key()) {
                continue;
            }
            // A retired axis lost in both directions against this incumbent
            // (issue #203); a probe is no more entitled to re-open it than the
            // ordinary generator is.
            if let FollowUpProbe::Scalar { target, .. } = &probe
                && batch.dead_axes.contains(&target.axis())
            {
                continue;
            }
            let Some(candidate) = probe_candidate(incumbent, &probe, backprop, &self.parent) else {
                continue;
            };
            if !batch.seen.insert(candidate_fingerprint(
                batch.incumbent_uuids,
                &candidate.creature,
            )) {
                continue;
            }
            out.push(candidate);
        }
        self.spent += out.len();
        out
    }

    /// The journal record for the slice [`Self::emit`] just produced.
    pub fn burst(&self, candidates: usize) -> FollowUpBurst {
        FollowUpBurst {
            parent_experiment: self.parent.experiment,
            parent_winner: self.parent.winner_stem.clone(),
            candidates,
            remaining: self.remaining(),
        }
    }
}

/// The neighbouring hypotheses around the difference `previous` → `winner`.
///
/// Families are interleaved rather than concatenated, so a cap that bites part
/// way through still spends its slots across weights, biases and squashes
/// instead of exhausting one family first.
pub fn plan_probes(previous: &CreatureExport, winner: &CreatureExport) -> Vec<FollowUpProbe> {
    let mut weights: Vec<FollowUpProbe> = Vec::new();
    let mut biases: Vec<FollowUpProbe> = Vec::new();
    let mut squashes: Vec<FollowUpProbe> = Vec::new();

    for neuron in &winner.neurons {
        let target = ProbeTarget::Bias {
            uuid: neuron.uuid.clone(),
        };
        match previous.neurons.iter().find(|n| n.uuid == neuron.uuid) {
            // A neuron the winner grew: try other squashes for it, and refine
            // the bias it was born with — which for a grown bridge is `0.0`, a
            // move of no size and therefore no probe. Its scale is refined
            // through the new synapses either side of it instead.
            None => {
                squashes.extend(alternate_squashes(&neuron.uuid, neuron.squash.as_deref()));
                biases.extend(scalar_probes(&target, neuron.bias));
            }
            Some(before) => {
                if before.squash != neuron.squash {
                    squashes.extend(alternate_squashes(&neuron.uuid, neuron.squash.as_deref()));
                }
                if before.bias != neuron.bias {
                    biases.extend(scalar_probes(&target, neuron.bias - before.bias));
                }
            }
        }
    }

    for synapse in &winner.synapses {
        let target = ProbeTarget::Weight {
            from_uuid: synapse.from_uuid.clone(),
            to_uuid: synapse.to_uuid.clone(),
        };
        let before = previous
            .synapses
            .iter()
            .find(|s| s.from_uuid == synapse.from_uuid && s.to_uuid == synapse.to_uuid);
        match before {
            // A synapse the winner added: the whole weight is the winning move,
            // so the probes scale around it.
            None => weights.extend(scalar_probes(&target, synapse.weight)),
            Some(before) if before.weight != synapse.weight => {
                weights.extend(scalar_probes(&target, synapse.weight - before.weight));
            }
            Some(_) => {}
        }
    }

    interleave(vec![weights, squashes, biases])
}

/// Round-robin the family lists into one proposal order.
fn interleave(mut families: Vec<Vec<FollowUpProbe>>) -> Vec<FollowUpProbe> {
    let mut out = Vec::new();
    let mut index = 0;
    loop {
        let mut pushed = false;
        for family in &mut families {
            if let Some(probe) = family.get(index) {
                out.push(probe.clone());
                pushed = true;
            }
        }
        if !pushed {
            return out;
        }
        index += 1;
    }
}

/// Step probes around a winning move of size `delta`.
fn scalar_probes(target: &ProbeTarget, delta: f64) -> Vec<FollowUpProbe> {
    if !delta.is_finite() || delta == 0.0 {
        return Vec::new();
    }
    STEP_FRACTIONS
        .iter()
        .map(|fraction| FollowUpProbe::Scalar {
            target: target.clone(),
            step: delta * fraction,
        })
        .collect()
}

/// Alternate squashes for a neuron the winner grew or re-squashed.
///
/// `current` is `None` for a neuron carrying no explicit squash, in which case
/// nothing is excluded — every alternative is a genuinely different hypothesis.
fn alternate_squashes(uuid: &str, current: Option<&str>) -> Vec<FollowUpProbe> {
    NEURON_GROWTH_SQUASHES
        .iter()
        .filter(|squash| Some(**squash) != current)
        .take(SQUASH_ALTERNATIVES)
        .map(|squash| FollowUpProbe::Squash {
            uuid: uuid.to_string(),
            squash: (*squash).to_string(),
        })
        .collect()
}

/// Apply one probe to the incumbent, or `None` when it no longer applies.
pub fn probe_candidate(
    incumbent: &CreatureExport,
    probe: &FollowUpProbe,
    backprop: &BackpropConfig,
    parent: &FollowUpParent,
) -> Option<Candidate> {
    let mut creature = incumbent.clone();
    let (mutation, old_value, new_value) = match probe {
        FollowUpProbe::Scalar { target, step } => {
            if !step.is_finite() {
                return None;
            }
            let (old, limit) = match target {
                ProbeTarget::Bias { uuid } => {
                    let neuron = creature.neurons.iter().find(|n| &n.uuid == uuid)?;
                    (neuron.bias, backprop.limit_bias_scale)
                }
                ProbeTarget::Weight { from_uuid, to_uuid } => {
                    let synapse = creature
                        .synapses
                        .iter()
                        .find(|s| &s.from_uuid == from_uuid && &s.to_uuid == to_uuid)?;
                    (synapse.weight, backprop.limit_weight_scale)
                }
            };
            let new = old + step;
            // A clamped probe is no longer the hypothesis it was planned as, and
            // a step below the plank constant is not a hypothesis at all.
            if !new.is_finite()
                || (limit.is_finite() && new.abs() > limit)
                || (new - old).abs() < backprop.plank_constant
            {
                return None;
            }
            match target {
                ProbeTarget::Bias { uuid } => {
                    creature.neurons.iter_mut().find(|n| &n.uuid == uuid)?.bias = new;
                }
                ProbeTarget::Weight { from_uuid, to_uuid } => {
                    creature
                        .synapses
                        .iter_mut()
                        .find(|s| &s.from_uuid == from_uuid && &s.to_uuid == to_uuid)?
                        .weight = new;
                }
            }
            (
                format!("follow-up {} {old} -> {new}", target.axis()),
                Some(old),
                Some(new),
            )
        }
        FollowUpProbe::Squash { uuid, squash } => {
            let neuron = creature.neurons.iter_mut().find(|n| &n.uuid == uuid)?;
            if neuron.squash.as_deref() == Some(squash.as_str()) {
                return None;
            }
            let mutation = format!(
                "follow-up squash {uuid} {} -> {squash}",
                neuron.squash.as_deref().unwrap_or("(none)")
            );
            neuron.squash = Some(squash.clone());
            (mutation, None, None)
        }
    };
    Some(Candidate {
        creature,
        provenance: CandidateProvenance {
            strategy: CandidateStrategy::FollowUp,
            focus_neuron: parent.focus_neuron.clone(),
            mutation,
            old_value,
            new_value,
            mirror: None,
            follow_up: Some(FollowUpLink {
                parent_experiment: parent.experiment,
                parent_winner: parent.winner_stem.clone(),
                parent_strategy: parent.strategy,
                probe: probe.key(),
            }),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::{NeuronExport, SynapseExport, parse_creature_json};

    const TINY: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 1,
      "output": 1,
      "neurons": [
        {"type":"hidden","uuid":"h1","bias":0.1,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0}
      ]
    }"#;

    fn neuron(uuid: &str, neuron_type: &str, bias: f64, squash: &str) -> NeuronExport {
        NeuronExport {
            id: None,
            neuron_type: neuron_type.to_string(),
            uuid: uuid.to_string(),
            bias,
            squash: Some(squash.to_string()),
        }
    }

    fn synapse(from: &str, to: &str, weight: f64) -> SynapseExport {
        SynapseExport {
            from_uuid: from.to_string(),
            to_uuid: to.to_string(),
            weight,
            synapse_type: None,
        }
    }

    fn incumbent() -> CreatureExport {
        parse_creature_json(TINY).expect("the tiny creature parses")
    }

    fn parent() -> FollowUpParent {
        FollowUpParent {
            experiment: 7,
            winner_stem: "candidate-003".to_string(),
            strategy: CandidateStrategy::StructuralAdd,
            focus_neuron: "o1".to_string(),
        }
    }

    /// Emit against a batch that holds `seen` and has retired `dead_axes`.
    fn emit_with(
        plan: &mut FollowUpPlan,
        incumbent: &CreatureExport,
        backprop: &BackpropConfig,
        seen: &mut HashSet<u64>,
        dead_axes: &[String],
    ) -> Vec<Candidate> {
        let incumbent_uuids: HashSet<String> = incumbent
            .neurons
            .iter()
            .map(|neuron| neuron.uuid.clone())
            .collect();
        plan.emit(
            incumbent,
            backprop,
            &mut BatchContext {
                incumbent_uuids: &incumbent_uuids,
                seen,
                dead_axes,
            },
        )
    }

    /// The same, at the default backprop limits.
    fn emit_into(
        plan: &mut FollowUpPlan,
        incumbent: &CreatureExport,
        seen: &mut HashSet<u64>,
        dead_axes: &[String],
    ) -> Vec<Candidate> {
        emit_with(plan, incumbent, &BackpropConfig::default(), seen, dead_axes)
    }

    fn budget(candidates: usize, experiments: usize) -> FollowUpBudget {
        FollowUpBudget {
            candidates,
            experiments,
        }
    }

    /// A winner that added a synapse is probed at nearby weight scales.
    #[test]
    fn a_new_synapse_is_probed_at_nearby_weights() {
        let previous = incumbent();
        let mut winner = previous.clone();
        winner.synapses.push(synapse("input-0", "o1", 0.2));

        let probes = plan_probes(&previous, &winner);
        let steps: Vec<f64> = probes
            .iter()
            .filter_map(|probe| match probe {
                FollowUpProbe::Scalar { target, step } if target.axis() == "weight:input-0->o1" => {
                    Some(*step)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            steps,
            vec![0.1, -0.1, 0.2],
            "probes scale the winning weight"
        );
    }

    /// A winner that grew a neuron is probed with alternate squashes for it.
    #[test]
    fn a_grown_neuron_is_probed_with_alternate_squashes() {
        let previous = incumbent();
        let mut winner = previous.clone();
        winner
            .neurons
            .insert(1, neuron("grown", "hidden", 0.3, "TANH"));
        winner.synapses.push(synapse("input-0", "grown", 0.4));
        winner.synapses.push(synapse("grown", "o1", 0.5));

        let squashes: Vec<String> = plan_probes(&previous, &winner)
            .into_iter()
            .filter_map(|probe| match probe {
                FollowUpProbe::Squash { uuid, squash } if uuid == "grown" => Some(squash),
                _ => None,
            })
            .collect();
        assert_eq!(squashes.len(), SQUASH_ALTERNATIVES);
        assert!(
            !squashes.iter().any(|squash| squash == "TANH"),
            "the winner's own squash is not re-proposed: {squashes:?}"
        );
    }

    /// A moved scalar is followed further along, and half-way back.
    #[test]
    fn a_moved_weight_is_probed_further_and_backed_off() {
        let previous = incumbent();
        let mut winner = previous.clone();
        winner.synapses[1].weight = 1.2;

        let mut plan =
            FollowUpPlan::from_accept(&previous, &winner, parent(), budget(3, 1)).expect("a plan");
        let mut seen = HashSet::new();
        let emitted = emit_into(&mut plan, &winner, &mut seen, &[]);
        let weights: Vec<f64> = emitted
            .iter()
            .map(|candidate| candidate.creature.synapses[1].weight)
            .collect();
        // The incumbent sits at 1.2 after a +0.2 win: +0.1 further, 0.1 back,
        // and a full second step.
        assert_eq!(weights.len(), 3);
        assert!((weights[0] - 1.3).abs() < 1e-12, "{weights:?}");
        assert!((weights[1] - 1.1).abs() < 1e-12, "{weights:?}");
        assert!((weights[2] - 1.4).abs() < 1e-12, "{weights:?}");
    }

    /// Every follow-up candidate names the winner it came from.
    #[test]
    fn follow_up_provenance_links_back_to_the_parent_winner() {
        let previous = incumbent();
        let mut winner = previous.clone();
        winner.synapses[1].weight = 1.2;

        let mut plan =
            FollowUpPlan::from_accept(&previous, &winner, parent(), budget(2, 1)).expect("a plan");
        let emitted = emit_into(&mut plan, &winner, &mut HashSet::new(), &[]);
        assert!(!emitted.is_empty());
        for candidate in &emitted {
            let link = candidate
                .provenance
                .follow_up
                .as_ref()
                .expect("a follow-up candidate carries its link");
            assert_eq!(link.parent_experiment, 7);
            assert_eq!(link.parent_winner, "candidate-003");
            assert_eq!(link.parent_strategy, CandidateStrategy::StructuralAdd);
            assert_eq!(candidate.provenance.strategy, CandidateStrategy::FollowUp);
            assert_eq!(candidate.provenance.focus_neuron, "o1");
        }
    }

    /// The candidate cap is a hard cap across the whole burst.
    #[test]
    fn the_candidate_cap_bounds_the_whole_burst() {
        let previous = incumbent();
        let mut winner = previous.clone();
        winner.synapses[0].weight = 1.1;
        winner.synapses[1].weight = 1.2;
        winner.neurons[0].bias = 0.2;

        let mut plan =
            FollowUpPlan::from_accept(&previous, &winner, parent(), budget(4, 4)).expect("a plan");
        let mut emitted = 0;
        for _ in 0..10 {
            emitted += emit_into(&mut plan, &winner, &mut HashSet::new(), &[]).len();
        }
        assert_eq!(emitted, 4, "the burst never exceeds its candidate cap");
        assert!(plan.is_exhausted());
    }

    /// The experiment cap expires the burst even with probes left over.
    #[test]
    fn the_experiment_cap_expires_the_burst() {
        let previous = incumbent();
        let mut winner = previous.clone();
        winner.synapses[0].weight = 1.1;
        winner.synapses[1].weight = 1.2;
        winner.neurons[0].bias = 0.2;

        let mut plan =
            FollowUpPlan::from_accept(&previous, &winner, parent(), budget(9, 2)).expect("a plan");
        let first = emit_into(&mut plan, &winner, &mut HashSet::new(), &[]);
        assert_eq!(
            first.len(),
            5,
            "half the cap, rounded up, in one experiment"
        );
        let second = emit_into(&mut plan, &winner, &mut HashSet::new(), &[]);
        assert_eq!(second.len(), 4);
        assert!(
            plan.is_exhausted(),
            "the burst expires after two experiments"
        );
        assert!(emit_into(&mut plan, &winner, &mut HashSet::new(), &[]).is_empty());
    }

    /// A probe the batch is already asking is dropped, not scored twice.
    #[test]
    fn a_probe_the_batch_already_holds_is_deduplicated() {
        let previous = incumbent();
        let mut winner = previous.clone();
        winner.synapses[1].weight = 1.2;

        // The batch already holds every hypothesis this plan would propose.
        let mut seen = HashSet::new();
        let mut already_batched =
            FollowUpPlan::from_accept(&previous, &winner, parent(), budget(3, 1)).expect("a plan");
        let batched = emit_into(&mut already_batched, &winner, &mut seen, &[]);
        assert_eq!(batched.len(), 3, "the batch holds three hypotheses");

        let mut plan =
            FollowUpPlan::from_accept(&previous, &winner, parent(), budget(3, 1)).expect("a plan");
        let emitted = emit_into(&mut plan, &winner, &mut seen, &[]);
        assert!(
            emitted.is_empty(),
            "a hypothesis the batch already asks is not re-proposed: {:?}",
            emitted
                .iter()
                .map(|candidate| candidate.provenance.mutation.clone())
                .collect::<Vec<_>>()
        );
    }

    /// A retired axis is not re-opened by a probe (issues #203, #219).
    #[test]
    fn a_probe_on_a_retired_axis_is_not_proposed() {
        let previous = incumbent();
        let mut winner = previous.clone();
        winner.synapses[1].weight = 1.2;
        let mut plan =
            FollowUpPlan::from_accept(&previous, &winner, parent(), budget(3, 1)).expect("a plan");

        let dead = vec!["weight:h1->o1".to_string()];
        let emitted = emit_into(&mut plan, &winner, &mut HashSet::new(), &dead);
        assert!(
            emitted.is_empty(),
            "both directions of this axis already lost against this incumbent"
        );
    }

    /// A neuron carrying no explicit squash excludes nothing (issue #219).
    #[test]
    fn a_neuron_without_a_squash_is_offered_every_alternative() {
        let previous = incumbent();
        let mut winner = previous.clone();
        let mut grown = neuron("grown", "hidden", 0.0, "TANH");
        grown.squash = None;
        winner.neurons.insert(1, grown);
        winner.synapses.push(synapse("input-0", "grown", 0.4));
        winner.synapses.push(synapse("grown", "o1", 0.5));

        let squashes: Vec<String> = plan_probes(&previous, &winner)
            .into_iter()
            .filter_map(|probe| match probe {
                FollowUpProbe::Squash { uuid, squash } if uuid == "grown" => Some(squash),
                _ => None,
            })
            .collect();
        assert_eq!(
            squashes,
            NEURON_GROWTH_SQUASHES[..SQUASH_ALTERNATIVES]
                .iter()
                .map(|squash| (*squash).to_string())
                .collect::<Vec<_>>(),
            "with no current squash, nothing is excluded"
        );
    }

    /// A zero budget plans nothing at all — the off arm of the A/B.
    #[test]
    fn a_zero_budget_plans_nothing() {
        let previous = incumbent();
        let mut winner = previous.clone();
        winner.synapses[1].weight = 1.2;
        assert!(FollowUpPlan::from_accept(&previous, &winner, parent(), budget(0, 2)).is_none());
        assert!(FollowUpPlan::from_accept(&previous, &winner, parent(), budget(4, 0)).is_none());
    }

    /// An accept that changed nothing probeable plans nothing.
    #[test]
    fn an_unchanged_winner_plans_nothing() {
        let previous = incumbent();
        assert!(
            FollowUpPlan::from_accept(&previous, &previous.clone(), parent(), budget(4, 2))
                .is_none()
        );
    }

    /// A probe whose target has gone is skipped rather than faked.
    #[test]
    fn a_probe_whose_target_vanished_is_skipped() {
        let previous = incumbent();
        let mut winner = previous.clone();
        winner.synapses[1].weight = 1.2;
        let mut plan =
            FollowUpPlan::from_accept(&previous, &winner, parent(), budget(3, 1)).expect("a plan");

        // The run moved on: the probed edge no longer exists.
        let mut moved_on = winner.clone();
        moved_on.synapses.remove(1);
        let emitted = emit_into(&mut plan, &moved_on, &mut HashSet::new(), &[]);
        assert!(
            emitted.is_empty(),
            "no candidate is invented for a lost edge"
        );
    }

    /// A step past the hard weight limit is dropped, not clamped.
    #[test]
    fn a_probe_past_the_weight_limit_is_dropped() {
        let previous = incumbent();
        let mut winner = previous.clone();
        winner.synapses[1].weight = 1.2;
        let mut plan =
            FollowUpPlan::from_accept(&previous, &winner, parent(), budget(3, 1)).expect("a plan");
        let backprop = BackpropConfig {
            limit_weight_scale: 1.25,
            ..BackpropConfig::default()
        };
        let weights: Vec<f64> = emit_with(&mut plan, &winner, &backprop, &mut HashSet::new(), &[])
            .iter()
            .map(|candidate| candidate.creature.synapses[1].weight)
            .collect();
        assert_eq!(weights.len(), 1, "only the back-off fits the limit");
        assert!((weights[0] - 1.1).abs() < 1e-12, "{weights:?}");
    }
}
