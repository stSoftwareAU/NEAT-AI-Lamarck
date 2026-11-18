//! A simplified random-axis control on a fixed weight set.
//!
//! This is not the production candidate pipeline (`generate_candidate_batch`)
//! or its statistical/backprop proposals: the axis and step are drawn at
//! random. The antithetic twin and the follow-up probes, however, now come from
//! [`crate::mirror::mirror_candidate`] and [`crate::followup::probe_candidate`],
//! so the arm inherits their refusals (the hard weight limit and the plank
//! constant).
//!
//! The follow-up helpers (`plan_probes` and `probe_candidate`) address a weight by `(from_uuid, to_uuid)` and take the first
//! match, whereas the bench's axes are synapse indices. The arm therefore
//! refuses, before any evaluation, a creature in which one `(from, to)` pair
//! appears on more than one synapse (for example one source feeding an `IF`
//! neuron in two roles).

use neat_core::CreatureExport;
use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::objective::{BenchError, BudgetedObjective, WeightObjective};
use crate::backprop::BackpropConfig;
use crate::candidates::{
    Candidate, CandidateProvenance, CandidateStrategy, RANDOM_DELTA_HALF_WIDTH,
};
use crate::followup::{FollowUpParent, plan_probes, probe_candidate};
use crate::mirror::{candidate_stem, mirror_candidate, signed_perturbation};

/// Weights of the selected synapses, read back from a creature.
fn weights_of(objective: &WeightObjective, creature: &CreatureExport) -> Vec<f64> {
    objective
        .synapse_indices()
        .iter()
        .map(|&i| creature.synapses[i].weight)
        .collect()
}

/// Wrap a random-axis creature as a candidate so the production helpers accept it.
fn random_candidate(creature: CreatureExport) -> Candidate {
    Candidate {
        creature,
        provenance: CandidateProvenance {
            strategy: CandidateStrategy::Random,
            focus_neuron: String::new(),
            mutation: "weight bench random-axis step".into(),
            old_value: None,
            new_value: None,
            mirror: None,
            follow_up: None,
            neighbourhood: None,
        },
    }
}

/// The `+delta` point on `axis` and its `-delta` twin from
/// [`mirror_candidate`], or `None` for the twin when that helper refuses (the
/// hard weight limit would clamp it, or the step is below the plank constant)
/// or when `signed_perturbation` finds no single-weight change.
fn mirrored_pair(
    objective: &WeightObjective,
    x: &[f64],
    axis: usize,
    delta: f64,
    backprop: &BackpropConfig,
) -> (Vec<f64>, Option<Vec<f64>>) {
    let incumbent = objective.creature_at(x);
    let mut plus_creature = incumbent.clone();
    plus_creature.synapses[objective.synapse_indices()[axis]].weight += delta;
    let plus = weights_of(objective, &plus_creature);
    let twin = signed_perturbation(&incumbent, &plus_creature).and_then(|perturbation| {
        mirror_candidate(&random_candidate(plus_creature), &perturbation, backprop)
    });
    (plus, twin.map(|c| weights_of(objective, &c.creature)))
}

/// The follow-up points [`plan_probes`] proposes around a move from
/// `previous_x` to `winner_x`, in proposal order; probes
/// [`probe_candidate`] refuses (past the weight limit, below plank) are dropped.
fn follow_up_points(
    objective: &WeightObjective,
    previous_x: &[f64],
    winner_x: &[f64],
    backprop: &BackpropConfig,
) -> Vec<Vec<f64>> {
    let previous = objective.creature_at(previous_x);
    let winner = objective.creature_at(winner_x);
    let parent = FollowUpParent {
        experiment: 0,
        winner_stem: candidate_stem(0),
        strategy: CandidateStrategy::Random,
        focus_neuron: String::new(),
    };
    plan_probes(&previous, &winner)
        .iter()
        .filter_map(|probe| probe_candidate(&winner, probe, backprop, &parent))
        .map(|c| weights_of(objective, &c.creature))
        .collect()
}

/// Refuse a creature where a `(from_uuid, to_uuid)` pair repeats on any synapse
/// (free or not); `plan_probes` compares whole creatures by that pair.
fn require_unique_synapse_pairs(creature: &CreatureExport) -> Result<(), BenchError> {
    let mut seen = HashSet::new();
    for synapse in &creature.synapses {
        if !seen.insert((synapse.from_uuid.as_str(), synapse.to_uuid.as_str())) {
            return Err(BenchError::InvalidConfig(format!(
                "synapse pair {} -> {} appears on more than one synapse; the production \
                 follow-up helpers address a weight by (from, to), so the refine arm \
                 refuses this creature",
                synapse.from_uuid, synapse.to_uuid
            )));
        }
    }
    Ok(())
}

/// Random-axis refinement with antithetic pairs and follow-up probes.
///
/// Each round picks an axis and a delta drawn like the Random strategy, tries
/// `x + delta` and the twin from [`mirror_candidate`], and accepts the better
/// one if it strictly improves. After an accept it evaluates the probes from
/// [`plan_probes`] / [`probe_candidate`] (the `STEP_FRACTIONS` along the
/// winning move) and takes the best strict improvement (one greedy round). The
/// start is evaluated first and counts. A twin or probe those helpers refuse is
/// simply not evaluated. This remains a simplified control, not the production
/// pipeline.
///
/// # Errors
///
/// Returns [`BenchError::InvalidConfig`], before any evaluation, if any
/// `(from_uuid, to_uuid)` pair appears on more than one synapse of the
/// creature: the follow-up helpers address a weight by that pair, so they
/// would act on the wrong synapse.
pub fn lamarck_refine(obj: &mut BudgetedObjective<'_>, seed: u64) -> Result<(), BenchError> {
    let backprop = BackpropConfig::default();
    let objective = obj.objective();
    require_unique_synapse_pairs(&objective.creature_at(&obj.start()))?;
    let mut rng = StdRng::seed_from_u64(seed);
    let dim = obj.dimension();
    let mut x = obj.start();
    let mut f = obj.eval(&x)?;
    while obj.remaining() >= 2 {
        let axis = rng.random_range(0..dim);
        let delta = loop {
            let d = rng.random_range(-RANDOM_DELTA_HALF_WIDTH..RANDOM_DELTA_HALF_WIDTH);
            if d != 0.0 {
                break d;
            }
        };
        let (plus, twin) = mirrored_pair(objective, &x, axis, delta, &backprop);
        let f_plus = obj.eval(&plus)?;
        let mut best = (f_plus, plus);
        if let Some(minus) = twin {
            let f_minus = obj.eval(&minus)?;
            if f_minus < f_plus {
                best = (f_minus, minus);
            }
        }
        let (f_best, winner) = best;
        if f_best >= f {
            continue;
        }
        let previous = std::mem::replace(&mut x, winner);
        f = f_best;
        // Follow-up probes along the accepted move, all from the accepted point.
        let mut best_probe: Option<(f64, Vec<f64>)> = None;
        for probe in follow_up_points(objective, &previous, &x, &backprop) {
            if obj.remaining() == 0 {
                break;
            }
            let fp = obj.eval(&probe)?;
            if fp < f && best_probe.as_ref().is_none_or(|(b, _)| fp < *b) {
                best_probe = Some((fp, probe));
            }
        }
        if let Some((fp, probe)) = best_probe {
            x = probe;
            f = fp;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;

    fn objective(w0: f64, w1: f64) -> WeightObjective {
        let json = format!(
            r#"{{
              "semanticVersion": "4.0.0",
              "forwardOnly": true,
              "input": 1,
              "output": 1,
              "neurons": [
                {{"type":"hidden","uuid":"h1","bias":0.1,"squash":"IDENTITY"}},
                {{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}}
              ],
              "synapses": [
                {{"fromUUID":"input-0","toUUID":"h1","weight":{w0:?}}},
                {{"fromUUID":"h1","toUUID":"o1","weight":{w1:?}}}
              ]
            }}"#
        );
        let creature = parse_creature_json(&json).unwrap();
        WeightObjective::new(creature, vec![0, 1], &[1.0, 2.0, 2.0, 3.0]).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn twin_straddles_x_on_the_chosen_axis() {
        let obj = objective(1.0, -2.0);
        let x = obj.start();
        let (plus, minus) = mirrored_pair(&obj, &x, 1, 0.03, &BackpropConfig::default());
        let minus = minus.expect("a normal weight has a twin");
        assert!(close(plus[1], -1.97));
        assert!(close(minus[1], -2.03));
        assert_eq!(plus[0], 1.0);
        assert_eq!(minus[0], 1.0);
    }

    #[test]
    fn twin_is_refused_past_the_hard_weight_limit() {
        let obj = objective(1.0, -99_999.99);
        let x = obj.start();
        let config = BackpropConfig::default();
        // delta -0.02 puts the twin at -99_999.97 (inside); delta +0.02 puts it at
        // -100_000.01 (past the limit).
        let (_, inside) = mirrored_pair(&obj, &x, 1, -0.02, &config);
        assert!(inside.is_some(), "positive control: mirror stays inside");
        let (plus, outside) = mirrored_pair(&obj, &x, 1, 0.02, &config);
        assert!(outside.is_none(), "twin at -100_000.01 must be refused");
        assert!(close(plus[1], -99_999.97));
    }

    #[test]
    fn twin_is_refused_below_the_plank_constant() {
        let obj = objective(1.0, -2.0);
        let x = obj.start();
        let config = BackpropConfig {
            plank_constant: 0.05,
            ..BackpropConfig::default()
        };
        let (_, below) = mirrored_pair(&obj, &x, 1, 0.03, &config);
        assert!(below.is_none(), "a 0.03 step is below the 0.05 plank");
        let (_, above) = mirrored_pair(&obj, &x, 1, 0.06, &config);
        assert!(above.is_some(), "a 0.06 step clears the 0.05 plank");
    }

    #[test]
    fn follow_up_steps_around_the_winning_move() {
        let obj = objective(1.0, -2.0);
        let previous = vec![1.0, -2.0];
        let winner = vec![1.0, -1.96];
        let points = follow_up_points(&obj, &previous, &winner, &BackpropConfig::default());
        assert_eq!(points.len(), 3);
        assert!(close(points[0][1], -1.94));
        assert!(close(points[1][1], -1.98));
        assert!(close(points[2][1], -1.92));
        for p in &points {
            assert_eq!(p[0], 1.0);
        }
    }

    #[test]
    fn follow_up_drops_a_probe_past_the_hard_limit() {
        let obj = objective(1.0, 99_999.95);
        let previous = vec![1.0, 99_999.91];
        let winner = vec![1.0, 99_999.95];
        let points = follow_up_points(&obj, &previous, &winner, &BackpropConfig::default());
        // +0.02 -> 99_999.97 ok, -0.02 -> 99_999.93 ok, +0.04 -> 99_999.99 ok.
        assert_eq!(points.len(), 3);
        let previous = vec![1.0, 99_999.89];
        let winner = vec![1.0, 99_999.97];
        let points = follow_up_points(&obj, &previous, &winner, &BackpropConfig::default());
        // Move of +0.08: the +0.04 probe (to 100_000.01) and the +0.08 probe (to
        // 100_000.05) are dropped; only the -0.04 probe (to 99_999.93) survives.
        assert_eq!(points.len(), 1);
        assert!(close(points[0][1], 99_999.93));
    }
}
