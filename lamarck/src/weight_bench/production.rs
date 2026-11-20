//! An opt-in arm that drives the real production candidate generator.
//!
//! [`super::refine::lamarck_refine`] is a random-axis surrogate. This arm
//! instead calls [`crate::candidates::generate_candidate_batch`] — the same
//! generator the optimisation loop uses — on the incumbent, once per focus
//! neuron in round-robin order, with statistics gathered by the production
//! helpers ([`crate::focus::collect_focus_stats`],
//! [`crate::focus::collect_incoming_source_stats`] and
//! [`crate::observations::generate_statistics`]) over the objective's own
//! screening records.
//!
//! The bench optimises exactly the objective's selected weights, so only an
//! *admissible* candidate is evaluated (see [`admissible_point`]): one whose
//! creature differs from the incumbent in selected weights alone. Structural,
//! bias and out-of-set weight proposals are skipped, and the evaluations they
//! would have cost are spent on admissible ones instead. When a full cycle over
//! every focus neuron yields nothing admissible the arm stops and leaves the
//! rest of its budget unused; the row's `evaluationsUsed` records that.
//!
//! The screening records are written once to a fresh temporary directory as
//! `0.bin` (little-endian `f32`, inputs then outputs per record), which the
//! production helpers scan. A drop guard removes exactly that directory.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use neat_core::{CreatureExport, TrainingDataConfig, compile_creature};
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::objective::{BenchError, BudgetedObjective, WeightObjective};
use crate::backprop::BackpropConfig;
use crate::candidates::{CandidateBudget, CandidateGenContext, generate_candidate_batch};
use crate::focus::{collect_focus_stats, collect_incoming_source_stats};
use crate::mirror::MirrorPolicy;
use crate::observations::{StatsMode, generate_statistics};

/// Candidates requested from the production generator per round.
///
/// Sixteen is a modest batch: large enough to span the generator's opening
/// phases for one focus neuron, small enough that a tight bench budget is not
/// swallowed by a single round. The fixed pre-#108 quotas apply
/// (`scale_quotas: false`), so the generator may return fewer.
pub const PRODUCTION_BATCH: usize = 16;

/// Distinguishes temporary directories made within one process and instant.
static DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A fresh temporary directory holding the records as `0.bin`; removed on drop.
struct TempCorpus {
    path: PathBuf,
}

impl TempCorpus {
    /// Create a new directory (never reusing one) and write `records` into it.
    fn create(records: &[f32]) -> Result<Self, BenchError> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let name = format!(
            "lamarck-weight-bench-{}-{}-{nanos}",
            std::process::id(),
            DIR_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir(&path).map_err(|e| {
            BenchError::Evaluation(format!("create temp corpus {}: {e}", path.display()))
        })?;
        // From here the guard owns the directory, so a failed write still cleans up.
        let corpus = Self { path };
        let bytes: Vec<u8> = records.iter().flat_map(|v| v.to_le_bytes()).collect();
        let file = corpus.path.join("0.bin");
        std::fs::write(&file, bytes)
            .map_err(|e| BenchError::Evaluation(format!("write {}: {e}", file.display())))?;
        Ok(corpus)
    }
}

impl Drop for TempCorpus {
    fn drop(&mut self) {
        // A destructor cannot return an error, so report a failed clean-up as a warning.
        if let Err(e) = std::fs::remove_dir_all(&self.path) {
            crate::log::warn(&format!(
                "could not remove temporary corpus {}: {e}",
                self.path.display()
            ));
        }
    }
}

/// The weight vector of `candidate` if it is admissible against `incumbent`.
///
/// Admissible means: the same neurons (uuid, type, squash and bias) and the
/// same synapses in the same order with the same endpoints and type; every
/// weight that differs sits at an index in `objective.synapse_indices()`; and at
/// least one selected weight differs. Anything else returns `None`.
pub(super) fn admissible_point(
    objective: &WeightObjective,
    incumbent: &CreatureExport,
    candidate: &CreatureExport,
) -> Option<Vec<f64>> {
    if incumbent.neurons.len() != candidate.neurons.len()
        || incumbent.synapses.len() != candidate.synapses.len()
    {
        return None;
    }
    let same_neurons = incumbent
        .neurons
        .iter()
        .zip(&candidate.neurons)
        .all(|(a, b)| {
            a.uuid == b.uuid
                && a.neuron_type == b.neuron_type
                && a.squash == b.squash
                && a.bias == b.bias
        });
    if !same_neurons {
        return None;
    }
    let selected = objective.synapse_indices();
    let mut changed = false;
    for (i, (a, b)) in incumbent
        .synapses
        .iter()
        .zip(&candidate.synapses)
        .enumerate()
    {
        if a.from_uuid != b.from_uuid || a.to_uuid != b.to_uuid || a.synapse_type != b.synapse_type
        {
            return None;
        }
        if a.weight != b.weight {
            if !selected.contains(&i) {
                return None;
            }
            changed = true;
        }
    }
    changed.then(|| {
        selected
            .iter()
            .map(|&i| candidate.synapses[i].weight)
            .collect()
    })
}

/// Distinct target neurons of the selected synapses, in first-appearance order.
fn focus_list(objective: &WeightObjective) -> Vec<String> {
    let creature = objective.creature_at(&objective.start());
    let mut focus: Vec<String> = Vec::new();
    for &i in objective.synapse_indices() {
        let to = &creature.synapses[i].to_uuid;
        if !focus.contains(to) {
            focus.push(to.clone());
        }
    }
    focus
}

/// Refine with the production candidate generator under `obj`'s budget.
///
/// Deterministic for a given `seed`. The start is evaluated first and counts.
/// Each round takes the next focus neuron, generates a batch of
/// [`PRODUCTION_BATCH`] candidates from the incumbent, evaluates the admissible
/// ones in batch order until the budget runs out, and accepts the best if it
/// strictly beats the incumbent. Returns early, leaving budget unused, when a
/// whole cycle over the focus neurons produces nothing admissible.
///
/// # Errors
///
/// Returns a [`BenchError`] if the temporary corpus cannot be written, a
/// production helper fails, or `obj.eval` reports an error.
pub(super) fn production_refine(
    obj: &mut BudgetedObjective<'_>,
    seed: u64,
) -> Result<(), BenchError> {
    run(obj, seed, &mut None)
}

/// [`production_refine`], also reporting the temporary directory it used.
fn run(
    obj: &mut BudgetedObjective<'_>,
    seed: u64,
    used_dir: &mut Option<PathBuf>,
) -> Result<(), BenchError> {
    let objective = obj.objective();
    let corpus = TempCorpus::create(objective.records())?;
    *used_dir = Some(corpus.path.clone());
    let dir: &Path = &corpus.path;

    let base = objective.creature_at(&objective.start());
    let observations = generate_statistics(
        dir,
        &TrainingDataConfig::new(base.input, base.output),
        StatsMode::Full,
        None,
        false,
    )
    .map_err(|e| BenchError::Evaluation(format!("production observations: {e}")))?;
    let focus = focus_list(objective);
    let backprop = BackpropConfig::default();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut x = obj.start();
    let mut loss = obj.eval(&x)?;
    let (mut cursor, mut idle) = (0usize, 0usize);
    while obj.remaining() > 0 && idle < focus.len() {
        let focus_uuid = &focus[cursor % focus.len()];
        cursor += 1;
        let incumbent = objective.creature_at(&x);
        let mut net = compile_creature(&incumbent)
            .map_err(|e| BenchError::Evaluation(format!("production compile: {e}")))?;
        let focus_stats = collect_focus_stats(&incumbent, &mut net, dir, focus_uuid, None)
            .map_err(|e| BenchError::Evaluation(format!("production focus stats: {e}")))?;
        let incoming = collect_incoming_source_stats(
            &incumbent,
            &mut net,
            dir,
            focus_uuid,
            None,
            Some(&observations),
        )
        .map_err(|e| BenchError::Evaluation(format!("production incoming stats: {e}")))?;
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid,
            focus_stats: &focus_stats,
            incoming: &incoming,
            observations: &observations,
            ranked_sources: None,
            learning: None,
            backprop: &backprop,
            structural_only: false,
            mirror: MirrorPolicy::default(),
        };
        let batch = generate_candidate_batch(
            &ctx,
            CandidateBudget {
                count: PRODUCTION_BATCH,
                scale_quotas: false,
                allocation: None,
            },
            &mut rng,
        );
        let points: Vec<Vec<f64>> = batch
            .candidates
            .iter()
            .filter_map(|c| admissible_point(objective, &incumbent, &c.creature))
            .collect();
        if points.is_empty() {
            idle += 1;
            continue;
        }
        idle = 0;
        let mut best: Option<(f64, Vec<f64>)> = None;
        for point in points {
            if obj.remaining() == 0 {
                break;
            }
            let l = obj.eval(&point)?;
            if best.as_ref().is_none_or(|(b, _)| l < *b) {
                best = Some((l, point));
            }
        }
        if let Some((l, point)) = best
            && l < loss
        {
            x = point;
            loss = l;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_bench::suite::{Family, ProblemSpec, build_problem};
    use neat_core::parse_creature_json;

    fn objective() -> (WeightObjective, CreatureExport) {
        let json = r#"{
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
            {"fromUUID":"h1","toUUID":"o1","weight":0.5},
            {"fromUUID":"input-0","toUUID":"o1","weight":0.25}
          ]
        }"#;
        let creature = parse_creature_json(json).unwrap();
        let obj =
            WeightObjective::new(creature.clone(), vec![0, 1], &[1.0, 2.0, 2.0, 3.0]).unwrap();
        (obj, creature)
    }

    #[test]
    fn accepts_a_selected_weight_change_and_returns_x() {
        let (obj, base) = objective();
        let mut c = base.clone();
        c.synapses[1].weight = 0.7;
        assert_eq!(admissible_point(&obj, &base, &c), Some(vec![1.0, 0.7]));
    }

    #[test]
    fn rejects_bias_out_of_set_added_synapse_and_identical() {
        let (obj, base) = objective();
        let mut bias = base.clone();
        bias.neurons[0].bias = 0.2;
        bias.synapses[0].weight = 1.1;
        assert_eq!(admissible_point(&obj, &base, &bias), None);

        let mut outside = base.clone();
        outside.synapses[0].weight = 1.1;
        outside.synapses[2].weight = 0.3;
        assert_eq!(admissible_point(&obj, &base, &outside), None);
        let mut only_outside = base.clone();
        only_outside.synapses[2].weight = 0.3;
        assert_eq!(admissible_point(&obj, &base, &only_outside), None);

        let mut added = base.clone();
        added.synapses[0].weight = 1.1;
        let extra = added.synapses[2].clone();
        added.synapses.push(extra);
        assert_eq!(admissible_point(&obj, &base, &added), None);

        assert_eq!(admissible_point(&obj, &base, &base), None);
    }

    fn synthetic() -> WeightObjective {
        let p = build_problem(&ProblemSpec {
            family: Family::Tanh,
            dimension: 20,
            seed: 1001,
        })
        .unwrap();
        WeightObjective::new(p.creature.clone(), p.synapse_indices.clone(), &p.screening).unwrap()
    }

    #[test]
    fn production_refine_is_bounded_deterministic_and_cleans_up() {
        let o = synthetic();
        let start = o.loss(&o.start()).unwrap();
        let go = || {
            let mut b = BudgetedObjective::new(&o, 200);
            let mut dir = None;
            run(&mut b, 7, &mut dir).unwrap();
            (b.best().unwrap().0, b.used(), dir.unwrap())
        };
        let (loss_a, used_a, dir_a) = go();
        let (loss_b, used_b, dir_b) = go();
        assert!(used_a <= 200);
        assert!(used_a >= 1);
        assert_eq!((loss_a, used_a), (loss_b, used_b));
        assert!(loss_a <= start);
        assert!(!dir_a.exists(), "temp dir {} left behind", dir_a.display());
        assert_ne!(dir_a, dir_b);
        assert!(!dir_b.exists());
    }
}
