//! Fused per-experiment analysis scans (issue #105).
//!
//! An experiment used to walk the training sample five times — learning
//! signal, output MAE, focus stats, incoming-source stats and residual source
//! ranking — each opening its own [`TrainingDataIterator`] and re-activating
//! the incumbent on every record. Only two of those groups are genuinely
//! ordered: the first two passes choose the focus neuron, the last three need
//! the focus that choice produced.
//!
//! This module fuses them into exactly two scans:
//!
//! ```text
//! scan 1 (pre-focus)   learning signal + output MAE      → focus choice
//! scan 2 (post-focus)  focus stats + incoming + residual → candidate inputs
//! ```
//!
//! Each pass keeps its own accumulator (`LearningScan`, `OutputErrorScan`,
//! `FocusStatsScan`, `IncomingSourceScan`, `ResidualScan`) and the standalone
//! `collect_*` / `refine_*` functions drive the very same accumulators over
//! their own scan. The arithmetic is shared, so the fused and per-pass paths
//! cannot drift apart — `analysis::tests` asserts they agree record for record.
//!
//! Both scans are read-only reductions over records, so they fold record chunks
//! on up to `--analysis-threads` workers (issue #107). Determinism is kept by
//! the partition, not the schedule: the sample is cut into fixed
//! [`ANALYSIS_CHUNK_RECORDS`]-record chunks and the per-chunk partials are
//! merged in **chunk order**, so 1, 2 and 8 threads produce bit-identical
//! accumulators. Every RNG draw (`select_sparse`) happens on the calling thread
//! before the parallel region opens.
//!
//! A creature that is not `forwardOnly` carries activation state between
//! records, so chunking it would change its activations. Such a creature is
//! folded as a single chunk instead — correctness first, speed second.

use std::cell::Cell;
use std::collections::HashMap;
use std::path::Path;

use neat_core::{
    CompiledNetwork, CreatureExport, TrainingDataConfig, TrainingDataIterator, TrainingRecord,
};
use rand::Rng;

use crate::backprop::{BackpropConfig, LearningSignal};
use crate::chunks::{ANALYSIS_CHUNK_RECORDS, SamplePlan, map_chunks};
use crate::focus::{
    FocusNeuronStats, FocusStatsScan, IncomingSourceScan, IncomingSourceStats,
    OutputErrorInfluence, OutputErrorScan,
};
use crate::observations::ObservationsStatistics;
use crate::propagate_layout::LearningPlan;
use crate::structural::{RankedSource, ResidualScan, refine_sources_from_synthetic};

thread_local! {
    /// Training scans opened on this thread (see [`training_scans_opened`]).
    static TRAINING_SCANS_OPENED: Cell<u64> = const { Cell::new(0) };
}

/// Open a training-data scan, counting it against this thread's total.
///
/// Every analysis pass opens its iterator here so the per-experiment scan count
/// is observable — the fused path must never exceed two scans per experiment.
pub(crate) fn open_training_scan(
    training_data: &Path,
    config: TrainingDataConfig,
) -> Result<TrainingDataIterator, String> {
    note_training_scan();
    TrainingDataIterator::new(training_data, config).map_err(|e| e.to_string())
}

/// Count one logical pass over the training sample against this thread.
///
/// The chunked scans read the sample through many per-chunk readers, but they
/// still make exactly one pass over it — so the pass is counted once, by the
/// thread that starts the scan, and the two-scans-per-experiment budget from
/// issue #105 keeps meaning what it says.
pub(crate) fn note_training_scan() {
    TRAINING_SCANS_OPENED.with(|c| c.set(c.get().saturating_add(1)));
}

/// Number of analysis training scans opened on this thread so far.
pub fn training_scans_opened() -> u64 {
    TRAINING_SCANS_OPENED.with(Cell::get)
}

/// Reset this thread's analysis scan counter.
pub fn reset_training_scan_count() {
    TRAINING_SCANS_OPENED.with(|c| c.set(0));
}

/// Result of the pre-focus scan: everything needed to choose a focus neuron.
#[derive(Debug, Clone)]
pub struct PreFocusScan {
    /// Creature-wide backprop learning signal.
    pub learning: LearningSignal,
    /// Per-output residual summaries — empty unless requested.
    pub output_errors: HashMap<String, OutputErrorInfluence>,
    /// Records folded in.
    pub record_count: u64,
}

/// Result of the post-focus scan: everything the candidate generator consumes.
#[derive(Debug, Clone)]
pub struct PostFocusScan {
    /// Focus-neuron activation / residual statistics.
    pub focus_stats: FocusNeuronStats,
    /// Per-incoming-source statistics for the focus.
    pub incoming: Vec<IncomingSourceStats>,
    /// Unused sources re-ranked against the focus residual.
    pub ranked_sources: Vec<RankedSource>,
}

/// How much of the training sample a scan folds, and on how many workers.
///
/// Bundled so the scan entry points keep a readable signature: the cap and the
/// worker count are always chosen together, at the same call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanBudget {
    /// Cap on records folded — `None` folds the whole sample.
    pub max_records: Option<u64>,
    /// Worker threads folding record chunks (issue #107). Must be at least 1.
    pub threads: usize,
}

impl ScanBudget {
    /// Fold at most `max_records` records on `threads` workers.
    pub fn new(max_records: Option<u64>, threads: usize) -> Self {
        Self {
            max_records,
            threads,
        }
    }

    /// Fold at most `max_records` records on the calling thread alone.
    pub fn serial(max_records: Option<u64>) -> Self {
        Self::new(max_records, 1)
    }
}

/// Records per chunk for this creature.
///
/// A creature that is not `forwardOnly` may read a neuron that is activated
/// later in the pass, so its activation depends on the previous record — one
/// chunk, folded in record order, is the only correct partition for it.
fn chunk_records_for(creature: &CreatureExport) -> u64 {
    if creature.forward_only {
        ANALYSIS_CHUNK_RECORDS
    } else {
        u64::MAX
    }
}

/// An empty record buffer to refill per record, reusing its capacity.
fn empty_record() -> TrainingRecord {
    TrainingRecord {
        inputs: Vec::new(),
        outputs: Vec::new(),
    }
}

/// Fuse the two focus-independent passes into one training scan.
///
/// Accumulates the learning signal and — when `collect_output_errors` is set —
/// the per-output MAE from the same activation. Equivalent to calling
/// [`crate::propagate_layout::accumulate_creature_learning`] and
/// [`crate::focus::collect_output_mean_abs_errors`] in sequence, and
/// bit-identical to them whenever the sample fits one chunk.
///
/// `budget.threads` workers fold record chunks concurrently (issue #107); the
/// partials merge in chunk order, so the result does not depend on the worker
/// count. The sparse selection is drawn from `rng` here, on the calling thread,
/// before any worker starts.
pub fn scan_pre_focus(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    config: &BackpropConfig,
    budget: ScanBudget,
    rng: &mut impl Rng,
    collect_output_errors: bool,
) -> Result<PreFocusScan, String> {
    let learning_plan = LearningPlan::new(creature, config, rng)?;
    let want_errors = collect_output_errors && OutputErrorScan::new(creature).has_outputs();

    note_training_scan();
    let td_cfg = TrainingDataConfig::new(creature.input, creature.output);
    let sample = SamplePlan::new(training_data, td_cfg, budget.max_records)?;
    let chunks = sample.chunks(chunk_records_for(creature));
    let template = network.clone();

    let partials = map_chunks(
        budget.threads,
        &chunks,
        || Ok(template.clone()),
        |net, chunk| {
            let mut learning_scan = learning_plan.scan();
            let mut error_scan = OutputErrorScan::new(creature);
            let mut reader = sample.reader(chunk);
            let mut record = empty_record();
            let mut count = 0u64;
            while reader.next_record_into(&mut record)? {
                // `activate_and_trace` leaves the same outputs `activate`
                // returns in the leading `creature.output` slots, so both
                // passes read one activation.
                let traced = net.activate_and_trace(&record.inputs, creature.output);
                count += 1;
                learning_scan.observe(net, &record.outputs);
                if want_errors {
                    error_scan.observe(
                        &traced[..creature.output.min(traced.len())],
                        &record.outputs,
                    );
                }
            }
            Ok((learning_scan.finish(), error_scan, count))
        },
    )?;

    let mut learning = LearningSignal::new(creature.neurons.len(), creature.synapses.len());
    let mut error_scan = OutputErrorScan::new(creature);
    let mut count = 0u64;
    for (chunk_learning, chunk_errors, chunk_count) in &partials {
        learning.merge(chunk_learning)?;
        error_scan.merge(chunk_errors);
        count += chunk_count;
    }

    Ok(PreFocusScan {
        learning,
        output_errors: if want_errors {
            error_scan.finish()
        } else {
            HashMap::new()
        },
        record_count: count,
    })
}

/// Fuse the three focus-dependent passes into one training scan.
///
/// Accumulates focus statistics, incoming-source statistics and the residual
/// source ranking from a single activation per record. Equivalent to calling
/// [`crate::focus::collect_focus_stats`],
/// [`crate::focus::collect_incoming_source_stats`] and
/// [`crate::structural::refine_sources_by_residual_with_observations`] in
/// sequence, never materialises the sample as activation probes, and is
/// bit-identical to them whenever the sample fits one chunk.
///
/// `budget.threads` workers fold record chunks concurrently (issue #107); the
/// partials merge in chunk order, so the result does not depend on the worker
/// count.
pub fn scan_post_focus(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    focus_uuid: &str,
    budget: ScanBudget,
    observations: Option<&ObservationsStatistics>,
    prior_sources: &[RankedSource],
) -> Result<PostFocusScan, String> {
    let mut focus_scan = FocusStatsScan::new(creature, network, focus_uuid)?;
    let mut incoming_scan = IncomingSourceScan::new(creature, focus_uuid, observations)?;
    let mut residual_scan = ResidualScan::new(creature, focus_uuid, prior_sources)?;
    let scan_incoming = incoming_scan.needs_scan();

    note_training_scan();
    let td_cfg = TrainingDataConfig::new(creature.input, creature.output);
    let sample = SamplePlan::new(training_data, td_cfg, budget.max_records)?;
    let chunks = sample.chunks(chunk_records_for(creature));
    let template = network.clone();

    let partials = map_chunks(
        budget.threads,
        &chunks,
        || Ok(template.clone()),
        |net, chunk| {
            let mut focus = FocusStatsScan::new(creature, net, focus_uuid)?;
            let mut incoming = IncomingSourceScan::new(creature, focus_uuid, observations)?;
            let mut residual = ResidualScan::new(creature, focus_uuid, prior_sources)?;
            let mut reader = sample.reader(chunk);
            let mut record = empty_record();
            let mut count = 0u64;
            while reader.next_record_into(&mut record)? {
                count += 1;
                let traced = net.activate_and_trace(&record.inputs, creature.output);
                focus.observe(&traced, &record.outputs);
                if scan_incoming {
                    incoming.observe(&record.inputs, &record.outputs, &traced);
                }
                if residual.wants_probe(&record.inputs, &record.outputs) {
                    residual.observe(&record.inputs, &record.outputs, &traced);
                }
            }
            Ok((focus, incoming, residual, count))
        },
    )?;

    let mut count = 0u64;
    for (chunk_focus, chunk_incoming, chunk_residual, chunk_count) in &partials {
        focus_scan.merge(chunk_focus);
        incoming_scan.merge(chunk_incoming);
        residual_scan.merge(chunk_residual);
        count += chunk_count;
    }

    // Fewer than two rows cannot carry a residual statistic — fall back to
    // synthetic probes exactly as the standalone refine does.
    let ranked_sources = if count < 2 {
        refine_sources_from_synthetic(creature, network, focus_uuid, prior_sources, observations)?
    } else {
        residual_scan.finish(prior_sources)
    };

    Ok(PostFocusScan {
        focus_stats: focus_scan.finish(),
        incoming: incoming_scan.finish(),
        ranked_sources,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::focus::{collect_focus_stats, collect_incoming_source_stats};
    use crate::observations::{StatsMode, generate_statistics};
    use crate::propagate_layout::accumulate_creature_learning;
    use crate::structural::{rank_unused_sources, refine_sources_by_residual_with_observations};
    use neat_core::{compile_creature, parse_creature_json};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::io::Write;
    use tempfile::{TempDir, tempdir};

    /// Two inputs, two hidden neurons (one saturating), one output — enough to
    /// exercise hidden sources, residual correlation and unused-source ranking.
    const CREATURE: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 3,
      "output": 1,
      "neurons": [
        {"type":"hidden","uuid":"h1","bias":0.1,"squash":"TANH"},
        {"type":"hidden","uuid":"h2","bias":-0.2,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.05,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h1","weight":0.7},
        {"fromUUID":"input-1","toUUID":"h2","weight":-0.4},
        {"fromUUID":"h1","toUUID":"o1","weight":0.9},
        {"fromUUID":"h2","toUUID":"o1","weight":0.3}
      ]
    }"#;

    /// Deterministic pseudo-random sample: 3 inputs + 1 target per record.
    fn write_sample(records: usize) -> TempDir {
        let dir = tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join("0.bin")).unwrap();
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..records {
            for _ in 0..4 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let v = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        f.flush().unwrap();
        dir
    }

    #[test]
    fn fused_pre_focus_scan_matches_the_two_separate_passes() {
        let dir = write_sample(64);
        let creature = parse_creature_json(CREATURE).unwrap();
        let cfg = BackpropConfig::default();

        let mut net_a = compile_creature(&creature).unwrap();
        let mut rng_a = StdRng::seed_from_u64(42);
        let learning = accumulate_creature_learning(
            &creature,
            &mut net_a,
            dir.path(),
            &cfg,
            Some(50),
            &mut rng_a,
        )
        .unwrap();
        let output_errors = crate::focus::collect_output_mean_abs_errors(
            &creature,
            &mut net_a,
            dir.path(),
            Some(50),
        )
        .unwrap();

        let mut net_b = compile_creature(&creature).unwrap();
        let mut rng_b = StdRng::seed_from_u64(42);
        let fused = scan_pre_focus(
            &creature,
            &mut net_b,
            dir.path(),
            &cfg,
            ScanBudget::serial(Some(50)),
            &mut rng_b,
            true,
        )
        .unwrap();

        assert_eq!(
            format!("{learning:?}"),
            format!("{:?}", fused.learning),
            "fused learning signal must be bit-identical"
        );
        let mut expected: Vec<_> = output_errors.iter().collect();
        expected.sort_by_key(|(k, _)| k.as_str());
        let mut actual: Vec<_> = fused.output_errors.iter().collect();
        actual.sort_by_key(|(k, _)| k.as_str());
        assert_eq!(
            format!("{expected:?}"),
            format!("{actual:?}"),
            "fused output MAE must be bit-identical"
        );
        assert_eq!(fused.record_count, 50);
    }

    #[test]
    fn fused_pre_focus_scan_skips_output_errors_when_not_requested() {
        let dir = write_sample(8);
        let creature = parse_creature_json(CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        let fused = scan_pre_focus(
            &creature,
            &mut network,
            dir.path(),
            &BackpropConfig::default(),
            ScanBudget::serial(None),
            &mut rng,
            false,
        )
        .unwrap();
        assert!(fused.output_errors.is_empty());
        assert_eq!(fused.record_count, 8);
    }

    /// Run the three post-focus passes separately, the fused way, and compare.
    fn assert_post_focus_matches(focus: &str, records: usize, limit: Option<u64>) {
        let dir = write_sample(records);
        let creature = parse_creature_json(CREATURE).unwrap();
        let observations = generate_statistics(
            dir.path(),
            &TrainingDataConfig::new(creature.input, creature.output),
            StatsMode::Quick,
            limit,
            true,
        )
        .unwrap();
        let prior = rank_unused_sources(&creature, focus, &observations);

        let mut net_a = compile_creature(&creature).unwrap();
        let focus_stats =
            collect_focus_stats(&creature, &mut net_a, dir.path(), focus, limit).unwrap();
        let incoming = collect_incoming_source_stats(
            &creature,
            &mut net_a,
            dir.path(),
            focus,
            limit,
            Some(&observations),
        )
        .unwrap();
        let ranked = refine_sources_by_residual_with_observations(
            &creature,
            &mut net_a,
            dir.path(),
            focus,
            &prior,
            limit,
            Some(&observations),
        )
        .unwrap();

        let mut net_b = compile_creature(&creature).unwrap();
        let fused = scan_post_focus(
            &creature,
            &mut net_b,
            dir.path(),
            focus,
            ScanBudget::serial(limit),
            Some(&observations),
            &prior,
        )
        .unwrap();

        assert_eq!(
            format!("{focus_stats:?}"),
            format!("{:?}", fused.focus_stats),
            "fused focus stats must be bit-identical ({focus})"
        );
        assert_eq!(
            format!("{incoming:?}"),
            format!("{:?}", fused.incoming),
            "fused incoming stats must be bit-identical ({focus})"
        );
        assert_eq!(
            format!("{ranked:?}"),
            format!("{:?}", fused.ranked_sources),
            "fused ranked sources must be bit-identical ({focus})"
        );
    }

    #[test]
    fn fused_post_focus_scan_matches_the_three_separate_passes_on_an_output() {
        assert_post_focus_matches("o1", 64, Some(40));
    }

    #[test]
    fn fused_post_focus_scan_matches_the_three_separate_passes_on_a_hidden() {
        assert_post_focus_matches("h1", 64, None);
    }

    #[test]
    fn fused_post_focus_scan_matches_the_synthetic_probe_fallback() {
        // A single record forces the residual fallback in both paths.
        assert_post_focus_matches("o1", 1, None);
    }

    #[test]
    fn fused_post_focus_scan_honours_the_record_limit() {
        let dir = write_sample(64);
        let creature = parse_creature_json(CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let fused = scan_post_focus(
            &creature,
            &mut network,
            dir.path(),
            "o1",
            ScanBudget::serial(Some(25)),
            None,
            &[],
        )
        .unwrap();
        assert_eq!(fused.focus_stats.record_count, 25);
    }

    #[test]
    fn the_two_fused_scans_open_exactly_two_training_iterators() {
        let dir = write_sample(32);
        let creature = parse_creature_json(CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let mut rng = StdRng::seed_from_u64(7);

        reset_training_scan_count();
        let pre = scan_pre_focus(
            &creature,
            &mut network,
            dir.path(),
            &BackpropConfig::default(),
            ScanBudget::serial(Some(20)),
            &mut rng,
            true,
        )
        .unwrap();
        assert_eq!(pre.record_count, 20);
        let prior = Vec::new();
        scan_post_focus(
            &creature,
            &mut network,
            dir.path(),
            "o1",
            ScanBudget::serial(Some(20)),
            None,
            &prior,
        )
        .unwrap();

        assert_eq!(
            training_scans_opened(),
            2,
            "an experiment must open at most two training scans"
        );
    }

    /// Records enough to span several chunks, so the merge is exercised.
    const MULTI_CHUNK_RECORDS: usize = (ANALYSIS_CHUNK_RECORDS as usize) * 2 + 137;

    /// Deterministic sample whose values span twelve orders of magnitude.
    ///
    /// Float addition is only non-associative when the terms differ in scale,
    /// so an even sample would let a wrongly-ordered merge pass by luck. Here a
    /// changed fold order moves the low-order bits and the equality assertions
    /// below fail — which is the whole point of running them at three thread
    /// counts.
    fn write_wide_magnitude_sample(records: usize) -> TempDir {
        let dir = tempdir().unwrap();
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(dir.path().join("0.bin")).unwrap());
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let scales = [1e-6f32, 1.0, 1e6];
        for r in 0..records {
            for k in 0..4 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
                let v = unit * scales[(r + k) % scales.len()];
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        f.flush().unwrap();
        dir
    }

    #[test]
    fn the_equality_fixture_is_order_sensitive() {
        // Guards the equality tests: on a sample whose sums are associative by
        // luck they would pass whatever the merge did. Folding the same values
        // in two different groupings must give two different totals.
        let dir = write_wide_magnitude_sample(3 * 64);
        let creature = parse_creature_json(CREATURE).unwrap();
        let plan = SamplePlan::new(
            dir.path(),
            TrainingDataConfig::new(creature.input, creature.output),
            None,
        )
        .unwrap();
        let total = |stride: u64| -> f64 {
            let mut sum = 0.0f64;
            for chunk in plan.chunks(stride) {
                let mut part = 0.0f64;
                let mut reader = plan.reader(&chunk);
                let mut record = empty_record();
                while reader.next_record_into(&mut record).unwrap() {
                    part += f64::from(record.inputs[0]) + f64::from(record.outputs[0]);
                }
                sum += part;
            }
            sum
        };
        assert_ne!(
            total(u64::MAX).to_bits(),
            total(7).to_bits(),
            "the fixture must expose float non-associativity"
        );
    }

    #[test]
    fn the_test_fixture_really_spans_several_chunks() {
        // Guards the two equality tests below: on a single-chunk sample they
        // would pass without ever merging anything.
        let dir = write_wide_magnitude_sample(MULTI_CHUNK_RECORDS);
        let creature = parse_creature_json(CREATURE).unwrap();
        let plan = SamplePlan::new(
            dir.path(),
            TrainingDataConfig::new(creature.input, creature.output),
            None,
        )
        .unwrap();
        assert!(
            plan.chunks(chunk_records_for(&creature)).len() >= 3,
            "the equality fixture must cross at least two chunk boundaries"
        );
    }

    #[test]
    fn the_pre_focus_scan_is_bit_identical_at_one_two_and_eight_threads() {
        let dir = write_wide_magnitude_sample(MULTI_CHUNK_RECORDS);
        let creature = parse_creature_json(CREATURE).unwrap();
        let cfg = BackpropConfig::default();

        let scan_at = |threads: usize| {
            let mut network = compile_creature(&creature).unwrap();
            let mut rng = StdRng::seed_from_u64(42);
            scan_pre_focus(
                &creature,
                &mut network,
                dir.path(),
                &cfg,
                ScanBudget::new(None, threads),
                &mut rng,
                true,
            )
            .unwrap()
        };

        let one = scan_at(1);
        assert_eq!(one.record_count, MULTI_CHUNK_RECORDS as u64);
        for threads in [2, 8] {
            let many = scan_at(threads);
            assert_eq!(
                format!("{:?}", one.learning),
                format!("{:?}", many.learning),
                "learning signal must be bit-identical at {threads} threads"
            );
            let sorted = |scan: &PreFocusScan| {
                let mut pairs: Vec<_> = scan.output_errors.iter().collect();
                pairs.sort_by_key(|(k, _)| k.as_str());
                format!("{pairs:?}")
            };
            assert_eq!(
                sorted(&one),
                sorted(&many),
                "output MAE must be bit-identical at {threads} threads"
            );
            assert_eq!(one.record_count, many.record_count);
        }
    }

    #[test]
    fn the_post_focus_scan_is_bit_identical_at_one_two_and_eight_threads() {
        let dir = write_wide_magnitude_sample(MULTI_CHUNK_RECORDS);
        let creature = parse_creature_json(CREATURE).unwrap();
        let observations = generate_statistics(
            dir.path(),
            &TrainingDataConfig::new(creature.input, creature.output),
            StatsMode::Quick,
            None,
            true,
        )
        .unwrap();

        // Both focus kinds: an output drives the residual/correlation branch,
        // a hidden drives the activation-std branch.
        for focus in ["o1", "h1"] {
            let prior = rank_unused_sources(&creature, focus, &observations);
            let scan_at = |threads: usize| {
                let mut network = compile_creature(&creature).unwrap();
                scan_post_focus(
                    &creature,
                    &mut network,
                    dir.path(),
                    focus,
                    ScanBudget::new(None, threads),
                    Some(&observations),
                    &prior,
                )
                .unwrap()
            };
            let one = scan_at(1);
            assert_eq!(one.focus_stats.record_count, MULTI_CHUNK_RECORDS as u64);
            for threads in [2, 8] {
                let many = scan_at(threads);
                assert_eq!(
                    format!("{:?}", one.focus_stats),
                    format!("{:?}", many.focus_stats),
                    "focus stats must be bit-identical at {threads} threads ({focus})"
                );
                assert_eq!(
                    format!("{:?}", one.incoming),
                    format!("{:?}", many.incoming),
                    "incoming stats must be bit-identical at {threads} threads ({focus})"
                );
                assert_eq!(
                    format!("{:?}", one.ranked_sources),
                    format!("{:?}", many.ranked_sources),
                    "ranked sources must be bit-identical at {threads} threads ({focus})"
                );
            }
        }
    }

    #[test]
    fn a_capped_sample_folds_the_same_records_at_every_thread_count() {
        // The cap must be applied to the sample, not per chunk.
        let dir = write_wide_magnitude_sample(MULTI_CHUNK_RECORDS);
        let creature = parse_creature_json(CREATURE).unwrap();
        let cfg = BackpropConfig::default();
        let limit = ANALYSIS_CHUNK_RECORDS + 11;
        let scan_at = |threads: usize| {
            let mut network = compile_creature(&creature).unwrap();
            let mut rng = StdRng::seed_from_u64(9);
            scan_pre_focus(
                &creature,
                &mut network,
                dir.path(),
                &cfg,
                ScanBudget::new(Some(limit), threads),
                &mut rng,
                true,
            )
            .unwrap()
        };
        let one = scan_at(1);
        let four = scan_at(4);
        assert_eq!(one.record_count, limit);
        assert_eq!(four.record_count, limit);
        assert_eq!(
            format!("{:?}", one.learning),
            format!("{:?}", four.learning)
        );
    }

    #[test]
    fn a_multi_threaded_scan_still_counts_as_one_training_pass() {
        let dir = write_wide_magnitude_sample(MULTI_CHUNK_RECORDS);
        let creature = parse_creature_json(CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let mut rng = StdRng::seed_from_u64(3);

        reset_training_scan_count();
        scan_pre_focus(
            &creature,
            &mut network,
            dir.path(),
            &BackpropConfig::default(),
            ScanBudget::new(None, 8),
            &mut rng,
            true,
        )
        .unwrap();
        scan_post_focus(
            &creature,
            &mut network,
            dir.path(),
            "o1",
            ScanBudget::new(None, 8),
            None,
            &[],
        )
        .unwrap();
        assert_eq!(
            training_scans_opened(),
            2,
            "chunking must not multiply the per-experiment scan budget"
        );
    }

    #[test]
    fn a_scan_rejects_a_zero_thread_count() {
        let dir = write_sample(8);
        let creature = parse_creature_json(CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        let err = scan_pre_focus(
            &creature,
            &mut network,
            dir.path(),
            &BackpropConfig::default(),
            ScanBudget::new(None, 0),
            &mut rng,
            true,
        )
        .expect_err("zero workers must not be read as serial");
        assert!(err.contains("at least 1"), "{err}");
    }

    #[test]
    fn a_recurrent_creature_is_folded_as_one_chunk() {
        // A creature that is not forwardOnly reads last record's activation, so
        // it must never be split across chunks.
        let recurrent = CREATURE.replace("\"forwardOnly\": true", "\"forwardOnly\": false");
        let creature = parse_creature_json(&recurrent).unwrap();
        assert_eq!(chunk_records_for(&creature), u64::MAX);
        let forward = parse_creature_json(CREATURE).unwrap();
        assert_eq!(chunk_records_for(&forward), ANALYSIS_CHUNK_RECORDS);
    }

    #[test]
    fn the_five_separate_passes_open_five_training_iterators() {
        // Guards the counter itself: it must see the pre-fusion scan count.
        let dir = write_sample(32);
        let creature = parse_creature_json(CREATURE).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let mut rng = StdRng::seed_from_u64(7);

        reset_training_scan_count();
        accumulate_creature_learning(
            &creature,
            &mut network,
            dir.path(),
            &BackpropConfig::default(),
            Some(20),
            &mut rng,
        )
        .unwrap();
        crate::focus::collect_output_mean_abs_errors(&creature, &mut network, dir.path(), Some(20))
            .unwrap();
        collect_focus_stats(&creature, &mut network, dir.path(), "o1", Some(20)).unwrap();
        collect_incoming_source_stats(&creature, &mut network, dir.path(), "o1", Some(20), None)
            .unwrap();
        refine_sources_by_residual_with_observations(
            &creature,
            &mut network,
            dir.path(),
            "o1",
            &[],
            Some(20),
            None,
        )
        .unwrap();

        assert_eq!(training_scans_opened(), 5);
    }
}
