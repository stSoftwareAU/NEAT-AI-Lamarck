//! Mirrored (antithetic) sampling for signed perturbation candidates (#203).
//!
//! Salimans et al. 2017, *Evolution Strategies as a Scalable Alternative to
//! Reinforcement Learning* (<https://arxiv.org/abs/1703.03864>), evaluates every
//! perturbation `ε` together with its negation `−ε`. The two evaluations are
//! negatively correlated, so their difference prices the local slope with far
//! less noise than two independent draws — which matters here, where a fleet win
//! lands around `1e-04` and an unpaired estimate at that scale is as likely to
//! discard a win as to find one.
//!
//! A candidate qualifies when it moves **exactly one scalar** of the incumbent —
//! one neuron bias or one synapse weight — and changes nothing else. That is the
//! whole "signed perturbation" family: weight nudges, bias shifts and scaled
//! rescales. Structural candidates (a grown neuron, an added synapse, a grafted
//! subtree) change the shape of the creature, have no meaningful negation, and
//! are never mirrored.
//!
//! Both halves join the same batch, so they are written into the same scoring
//! directory and scored in the same scorer call against identical records — the
//! only comparison `docs/scorer-batch-composition.md` permits.

use crate::backprop::BackpropConfig;
use crate::candidates::{Candidate, CandidateProvenance};
use neat_core::CreatureExport;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Which half of a mirrored (antithetic) pair a candidate is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MirrorRole {
    /// The `+δ` proposal a strategy offered.
    Original,
    /// The `−δ` twin built from it.
    Mirror,
}

/// Which scalar of the incumbent a candidate moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerturbedScalar {
    /// Bias of the neuron at this position.
    Bias {
        /// Index into `CreatureExport::neurons`.
        neuron: usize,
    },
    /// Weight of the synapse at this position.
    Weight {
        /// Index into `CreatureExport::synapses`.
        synapse: usize,
    },
}

/// The single signed scalar step a candidate applies to the incumbent.
#[derive(Debug, Clone, PartialEq)]
pub struct SignedPerturbation {
    /// Which scalar moved, by position in the incumbent.
    pub scalar: PerturbedScalar,
    /// Stable name of the axis: `bias:<uuid>` or `weight:<from>-><to>`.
    pub axis: String,
    /// Incumbent value before the step.
    pub old_value: f64,
    /// Candidate value after the step.
    pub new_value: f64,
}

impl SignedPerturbation {
    /// The signed step itself.
    pub fn delta(&self) -> f64 {
        self.new_value - self.old_value
    }
}

/// Pair membership recorded on a candidate's provenance.
///
/// Both halves carry the same [`Self::axis`] and opposite [`Self::delta`], which
/// is what lets `report` re-pair them from a journal alone — no batch order, no
/// index arithmetic, and nothing that a merged multi-focus population or a
/// cache-filtered batch can invalidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MirrorPair {
    /// Axis both halves move along.
    pub axis: String,
    /// Signed step this half applies along the axis.
    pub delta: f64,
    /// Which half of the pair this candidate is.
    pub role: MirrorRole,
}

/// Mirroring policy handed to candidate generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct MirrorPolicy<'a> {
    /// Emit the `−δ` twin of every signed perturbation candidate.
    pub enabled: bool,
    /// Axes whose mirrored pair already lost on both sides for this incumbent.
    ///
    /// Both directions of the axis have been measured and neither improved, so
    /// the incumbent is at a local optimum along it: re-proposing there spends a
    /// batch slot on a question already answered (issue #203).
    pub dead_axes: &'a [String],
}

impl MirrorPolicy<'_> {
    /// True when the policy makes generation look at perturbation axes at all.
    pub fn is_active(&self) -> bool {
        self.enabled || !self.dead_axes.is_empty()
    }

    /// True when this axis has already lost in both directions.
    pub fn is_dead(&self, axis: &str) -> bool {
        self.dead_axes.iter().any(|dead| dead == axis)
    }
}

/// Axis name for a neuron bias.
fn bias_axis(uuid: &str) -> String {
    format!("bias:{uuid}")
}

/// Axis name for a synapse weight.
fn weight_axis(from_uuid: &str, to_uuid: &str) -> String {
    format!("weight:{from_uuid}->{to_uuid}")
}

/// The signed perturbation a candidate applies, when it applies exactly one.
///
/// Returns `None` for anything structural (a differing neuron or synapse count,
/// a changed squash, a re-pointed edge), for a candidate that changed nothing,
/// and for one that moved more than a single scalar — none of those has a
/// well-defined negation.
pub fn signed_perturbation(
    incumbent: &CreatureExport,
    candidate: &CreatureExport,
) -> Option<SignedPerturbation> {
    if incumbent.neurons.len() != candidate.neurons.len()
        || incumbent.synapses.len() != candidate.synapses.len()
    {
        return None;
    }
    let mut found: Option<SignedPerturbation> = None;
    for (index, (before, after)) in incumbent
        .neurons
        .iter()
        .zip(candidate.neurons.iter())
        .enumerate()
    {
        if before.uuid != after.uuid
            || before.neuron_type != after.neuron_type
            || before.squash != after.squash
        {
            return None;
        }
        if before.bias != after.bias {
            if found.is_some() {
                return None;
            }
            found = Some(SignedPerturbation {
                scalar: PerturbedScalar::Bias { neuron: index },
                axis: bias_axis(&before.uuid),
                old_value: before.bias,
                new_value: after.bias,
            });
        }
    }
    for (index, (before, after)) in incumbent
        .synapses
        .iter()
        .zip(candidate.synapses.iter())
        .enumerate()
    {
        if before.from_uuid != after.from_uuid || before.to_uuid != after.to_uuid {
            return None;
        }
        if before.weight != after.weight {
            if found.is_some() {
                return None;
            }
            found = Some(SignedPerturbation {
                scalar: PerturbedScalar::Weight { synapse: index },
                axis: weight_axis(&before.from_uuid, &before.to_uuid),
                old_value: before.weight,
                new_value: after.weight,
            });
        }
    }
    found.filter(|p| p.old_value.is_finite() && p.new_value.is_finite())
}

/// Build the antithetic (`−δ`) twin of a signed perturbation candidate.
///
/// The twin steps the same scalar the same distance the other way from the
/// incumbent value, so the pair straddles the incumbent exactly. A step the hard
/// bias/weight limit would clamp produces `None`: a clamped `−δ` is no longer the
/// antithesis of `+δ`, and a pair that is not exactly opposite cannot cancel the
/// noise it exists to cancel.
pub fn mirror_candidate(
    candidate: &Candidate,
    perturbation: &SignedPerturbation,
    backprop: &BackpropConfig,
) -> Option<Candidate> {
    let delta = perturbation.delta();
    if !delta.is_finite() || delta == 0.0 {
        return None;
    }
    let mirrored = perturbation.old_value - delta;
    if !mirrored.is_finite() {
        return None;
    }
    let limit = match perturbation.scalar {
        PerturbedScalar::Bias { .. } => backprop.limit_bias_scale,
        PerturbedScalar::Weight { .. } => backprop.limit_weight_scale,
    };
    if limit.is_finite() && mirrored.abs() > limit {
        return None;
    }
    if (mirrored - perturbation.old_value).abs() < backprop.plank_constant {
        return None;
    }
    let mut creature = candidate.creature.clone();
    match perturbation.scalar {
        PerturbedScalar::Bias { neuron } => creature.neurons.get_mut(neuron)?.bias = mirrored,
        PerturbedScalar::Weight { synapse } => {
            creature.synapses.get_mut(synapse)?.weight = mirrored
        }
    }
    Some(Candidate {
        creature,
        provenance: CandidateProvenance {
            strategy: candidate.provenance.strategy,
            focus_neuron: candidate.provenance.focus_neuron.clone(),
            mutation: format!(
                "mirror -delta on {} {} -> {mirrored} (of: {})",
                perturbation.axis, perturbation.old_value, candidate.provenance.mutation
            ),
            old_value: Some(perturbation.old_value),
            new_value: Some(mirrored),
            mirror: Some(MirrorPair {
                axis: perturbation.axis.clone(),
                delta: -delta,
                role: MirrorRole::Mirror,
            }),
            follow_up: None,
        },
    })
}

/// Pair membership to stamp on the `+δ` half once its twin has joined the batch.
pub fn original_pair(perturbation: &SignedPerturbation) -> MirrorPair {
    MirrorPair {
        axis: perturbation.axis.clone(),
        delta: perturbation.delta(),
        role: MirrorRole::Original,
    }
}

/// Scoring stem of the candidate at `index`, as `write_candidate_batch` names it.
pub fn candidate_stem(index: usize) -> String {
    format!("candidate-{index:03}")
}

/// What a scored mirrored pair measured.
///
/// Both deltas are against the `baseline` stem of the **same** score map, so
/// they were formed inside one scorer call against identical records.
#[derive(Debug, Clone, PartialEq)]
pub struct PairOutcome {
    /// Axis the pair straddles.
    pub axis: String,
    /// Index of the `+δ` half in the experiment's candidate list.
    pub original_index: usize,
    /// Index of the `−δ` half.
    pub mirror_index: usize,
    /// Score improvement the `+δ` half measured.
    pub original_delta: f64,
    /// Score improvement the `−δ` half measured.
    pub mirror_delta: f64,
}

impl PairOutcome {
    /// True when the `+δ` half beat the baseline it was scored beside.
    pub fn original_won(&self) -> bool {
        self.original_delta > 0.0
    }

    /// True when the `−δ` half beat the baseline it was scored beside.
    pub fn mirror_won(&self) -> bool {
        self.mirror_delta > 0.0
    }

    /// True when neither direction improved — an axis-level failure.
    ///
    /// The incumbent is at a local optimum along this axis: both directions were
    /// measured against the same records and neither is worth revisiting.
    pub fn both_lost(&self) -> bool {
        !self.original_won() && !self.mirror_won()
    }
}

/// Axis plus step magnitude — what both halves of one pair share.
type PairKey<'a> = (&'a str, u64);

/// The `(index, score − baseline)` of each half of a pair, once scored.
type PairHalves = (Option<(usize, f64)>, Option<(usize, f64)>);

/// Re-pair the mirrored candidates of one experiment and score both halves.
///
/// A pair is reported only when **both** halves appear in `scores`, so a
/// promote map that carried one half and dropped the other never fabricates a
/// comparison across two scorer calls.
pub fn pair_outcomes(
    candidates: &[CandidateProvenance],
    scores: &BTreeMap<String, f64>,
) -> Vec<PairOutcome> {
    let Some(baseline) = scores.get("baseline").copied() else {
        return Vec::new();
    };
    // Key on the axis and the step magnitude: the two halves of a pair are the
    // only candidates that share both, whichever focus proposed them.
    let mut halves: BTreeMap<PairKey<'_>, PairHalves> = BTreeMap::new();
    for (index, provenance) in candidates.iter().enumerate() {
        let Some(pair) = &provenance.mirror else {
            continue;
        };
        let Some(score) = scores.get(&candidate_stem(index)) else {
            continue;
        };
        let entry = halves
            .entry((pair.axis.as_str(), pair.delta.abs().to_bits()))
            .or_default();
        let half = match pair.role {
            MirrorRole::Original => &mut entry.0,
            MirrorRole::Mirror => &mut entry.1,
        };
        if half.is_none() {
            *half = Some((index, score - baseline));
        }
    }
    let mut outcomes: Vec<PairOutcome> = halves
        .into_iter()
        .filter_map(|((axis, _), (original, mirror))| {
            let (original_index, original_delta) = original?;
            let (mirror_index, mirror_delta) = mirror?;
            Some(PairOutcome {
                axis: axis.to_string(),
                original_index,
                mirror_index,
                original_delta,
                mirror_delta,
            })
        })
        .collect();
    outcomes.sort_by_key(|outcome| outcome.original_index);
    outcomes
}

/// Axes whose mirrored pair lost in **both** directions, deduplicated.
pub fn axis_failures(
    candidates: &[CandidateProvenance],
    scores: &BTreeMap<String, f64>,
) -> Vec<String> {
    let mut axes: Vec<String> = pair_outcomes(candidates, scores)
        .into_iter()
        .filter(PairOutcome::both_lost)
        .map(|outcome| outcome.axis)
        .collect();
    axes.sort();
    axes.dedup();
    axes
}

/// What mirrored sampling bought, summed over a journal (issue #203).
///
/// [`Self::mirror_win_rate`] is the number the change is judged on: how often
/// the `−δ` twin improved on a batch where the `+δ` proposal did not.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MirrorStats {
    /// Pairs where both halves were scored in one call.
    pub pairs_scored: u64,
    /// Pairs whose `+δ` half did not beat the baseline.
    pub original_lost: u64,
    /// Subset of [`Self::original_lost`] whose `−δ` half did.
    pub mirror_won_when_original_lost: u64,
    /// Pairs where neither direction improved — the axis-level failures.
    pub both_lost: u64,
    /// `mirror_won_when_original_lost / original_lost` (0 with no losing pairs).
    pub mirror_win_rate: f64,
    /// Axis retirements journalled as `mirrorAxisFailures` (issue #203).
    ///
    /// Counted from the journal field rather than recomputed, so it reports
    /// what the run actually acted on. It exceeds [`Self::both_lost`] whenever
    /// one axis was retired by several experiments in a row.
    pub axes_retired: u64,
}

impl MirrorStats {
    /// Fold one experiment's journalled axis retirements in.
    pub fn push_axis_failures(&mut self, axes: &[String]) {
        self.axes_retired += axes.len() as u64;
    }

    /// Fold one scored pair in.
    pub fn push(&mut self, outcome: &PairOutcome) {
        self.pairs_scored += 1;
        if !outcome.original_won() {
            self.original_lost += 1;
            if outcome.mirror_won() {
                self.mirror_won_when_original_lost += 1;
            } else {
                self.both_lost += 1;
            }
        }
        self.mirror_win_rate = if self.original_lost > 0 {
            self.mirror_won_when_original_lost as f64 / self.original_lost as f64
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::CandidateStrategy;
    use neat_core::parse_creature_json;

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

    fn candidate(creature: CreatureExport, mutation: &str) -> Candidate {
        Candidate {
            creature,
            provenance: CandidateProvenance {
                strategy: CandidateStrategy::StatsBias,
                focus_neuron: "h1".into(),
                mutation: mutation.into(),
                old_value: None,
                new_value: None,
                mirror: None,
                follow_up: None,
            },
        }
    }

    fn provenance(mirror: Option<MirrorPair>) -> CandidateProvenance {
        CandidateProvenance {
            strategy: CandidateStrategy::StatsWeight,
            focus_neuron: "h1".into(),
            mutation: "m".into(),
            old_value: None,
            new_value: None,
            mirror,
            follow_up: None,
        }
    }

    #[test]
    fn bias_nudge_is_a_signed_perturbation() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let mut moved = incumbent.clone();
        moved.neurons[0].bias = 0.3;
        let perturbation = signed_perturbation(&incumbent, &moved).expect("one scalar moved");
        assert_eq!(perturbation.axis, "bias:h1");
        assert_eq!(perturbation.scalar, PerturbedScalar::Bias { neuron: 0 });
        assert!((perturbation.delta() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn weight_nudge_is_a_signed_perturbation() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let mut moved = incumbent.clone();
        moved.synapses[1].weight = 0.75;
        let perturbation = signed_perturbation(&incumbent, &moved).expect("one scalar moved");
        assert_eq!(perturbation.axis, "weight:h1->o1");
        assert_eq!(perturbation.scalar, PerturbedScalar::Weight { synapse: 1 });
        assert!((perturbation.delta() + 0.25).abs() < 1e-12);
    }

    #[test]
    fn structural_and_multi_scalar_changes_have_no_mirror() {
        let incumbent = parse_creature_json(TINY).unwrap();

        let mut grown = incumbent.clone();
        grown.synapses.push(neat_core::SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "o1".into(),
            weight: 0.2,
            synapse_type: None,
        });
        assert!(signed_perturbation(&incumbent, &grown).is_none());

        let mut two = incumbent.clone();
        two.neurons[0].bias = 0.2;
        two.synapses[0].weight = 1.2;
        assert!(signed_perturbation(&incumbent, &two).is_none());

        let mut squashed = incumbent.clone();
        squashed.neurons[0].squash = Some("TANH".into());
        squashed.neurons[0].bias = 0.2;
        assert!(signed_perturbation(&incumbent, &squashed).is_none());

        assert!(signed_perturbation(&incumbent, &incumbent).is_none());
    }

    #[test]
    fn mirror_straddles_the_incumbent_value() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let mut moved = incumbent.clone();
        moved.neurons[0].bias = 0.1 + 0.05;
        let original = candidate(moved, "stats bias");
        let perturbation = signed_perturbation(&incumbent, &original.creature).unwrap();
        let backprop = BackpropConfig::default();
        let mirror = mirror_candidate(&original, &perturbation, &backprop).expect("mirror");

        assert!((mirror.creature.neurons[0].bias - (0.1 - 0.05)).abs() < 1e-12);
        // Everything else is untouched: the pair differs only along the axis.
        assert_eq!(mirror.creature.synapses, incumbent.synapses);
        assert_eq!(mirror.provenance.strategy, original.provenance.strategy);
        let pair = mirror.provenance.mirror.expect("pair metadata");
        assert_eq!(pair.role, MirrorRole::Mirror);
        assert_eq!(pair.axis, "bias:h1");
        assert!((pair.delta + 0.05).abs() < 1e-12);
        assert!((original_pair(&perturbation).delta - 0.05).abs() < 1e-12);
    }

    #[test]
    fn mirror_refuses_a_step_the_hard_limit_would_clamp() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let backprop = BackpropConfig::default();
        let mut moved = incumbent.clone();
        // Sit the incumbent bias just inside the negative limit, then step
        // towards zero: the mirror would land outside the limit.
        moved.neurons[0].bias = -backprop.limit_bias_scale;
        let mut nudged = moved.clone();
        nudged.neurons[0].bias = -backprop.limit_bias_scale + 0.5;
        let perturbation = signed_perturbation(&moved, &nudged).unwrap();
        let original = candidate(nudged, "stats bias");
        assert!(mirror_candidate(&original, &perturbation, &backprop).is_none());
    }

    #[test]
    fn mirror_refuses_a_step_below_the_plank_constant() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let backprop = BackpropConfig::default();
        let mut nudged = incumbent.clone();
        nudged.neurons[0].bias = 0.1 + backprop.plank_constant / 4.0;
        let perturbation = signed_perturbation(&incumbent, &nudged).unwrap();
        let original = candidate(nudged, "tiny");
        assert!(mirror_candidate(&original, &perturbation, &backprop).is_none());
    }

    fn pair_provenances() -> Vec<CandidateProvenance> {
        vec![
            provenance(Some(MirrorPair {
                axis: "bias:h1".into(),
                delta: 0.05,
                role: MirrorRole::Original,
            })),
            provenance(Some(MirrorPair {
                axis: "bias:h1".into(),
                delta: -0.05,
                role: MirrorRole::Mirror,
            })),
            provenance(None),
        ]
    }

    #[test]
    fn pair_outcomes_measure_both_halves_against_one_baseline() {
        let scores = BTreeMap::from([
            ("baseline".to_string(), 0.5),
            ("candidate-000".to_string(), 0.4),
            ("candidate-001".to_string(), 0.7),
            ("candidate-002".to_string(), 0.9),
        ]);
        let outcomes = pair_outcomes(&pair_provenances(), &scores);
        assert_eq!(outcomes.len(), 1, "the unpaired candidate is not a pair");
        let outcome = &outcomes[0];
        assert_eq!(outcome.axis, "bias:h1");
        assert_eq!((outcome.original_index, outcome.mirror_index), (0, 1));
        assert!(!outcome.original_won());
        assert!(outcome.mirror_won());
        assert!(!outcome.both_lost());
        assert!(axis_failures(&pair_provenances(), &scores).is_empty());
    }

    #[test]
    fn both_sides_losing_is_an_axis_failure() {
        let scores = BTreeMap::from([
            ("baseline".to_string(), 0.5),
            ("candidate-000".to_string(), 0.4),
            ("candidate-001".to_string(), 0.45),
        ]);
        let outcomes = pair_outcomes(&pair_provenances(), &scores);
        assert!(outcomes[0].both_lost());
        assert_eq!(axis_failures(&pair_provenances(), &scores), vec!["bias:h1"]);
    }

    #[test]
    fn a_half_scored_in_another_call_is_not_a_pair() {
        // Only the promoted half reached the full-corpus map; comparing it with
        // a screen-phase score would subtract two different measurements.
        let scores = BTreeMap::from([
            ("baseline".to_string(), 0.5),
            ("candidate-001".to_string(), 0.7),
        ]);
        assert!(pair_outcomes(&pair_provenances(), &scores).is_empty());
        // A map with no baseline supports no comparison at all.
        let no_baseline = BTreeMap::from([
            ("candidate-000".to_string(), 0.4),
            ("candidate-001".to_string(), 0.7),
        ]);
        assert!(pair_outcomes(&pair_provenances(), &no_baseline).is_empty());
    }

    #[test]
    fn mirror_stats_rate_is_wins_over_losing_originals() {
        let mut stats = MirrorStats::default();
        stats.push(&PairOutcome {
            axis: "bias:h1".into(),
            original_index: 0,
            mirror_index: 1,
            original_delta: -0.1,
            mirror_delta: 0.2,
        });
        stats.push(&PairOutcome {
            axis: "weight:h1->o1".into(),
            original_index: 2,
            mirror_index: 3,
            original_delta: -0.1,
            mirror_delta: -0.2,
        });
        stats.push(&PairOutcome {
            axis: "bias:o1".into(),
            original_index: 4,
            mirror_index: 5,
            original_delta: 0.3,
            mirror_delta: -0.3,
        });
        assert_eq!(stats.pairs_scored, 3);
        assert_eq!(stats.original_lost, 2);
        assert_eq!(stats.mirror_won_when_original_lost, 1);
        assert_eq!(stats.both_lost, 1);
        assert!((stats.mirror_win_rate - 0.5).abs() < 1e-12);
    }

    #[test]
    fn dead_axis_policy_matches_by_name() {
        let dead = vec!["bias:h1".to_string()];
        let policy = MirrorPolicy {
            enabled: false,
            dead_axes: &dead,
        };
        assert!(policy.is_active());
        assert!(policy.is_dead("bias:h1"));
        assert!(!policy.is_dead("bias:o1"));
        assert!(!MirrorPolicy::default().is_active());
    }
}
