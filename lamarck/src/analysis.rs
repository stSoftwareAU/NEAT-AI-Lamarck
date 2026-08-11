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

use std::cell::Cell;
use std::collections::HashMap;
use std::path::Path;

use neat_core::{CompiledNetwork, CreatureExport, TrainingDataConfig, TrainingDataIterator};
use rand::Rng;

use crate::backprop::{BackpropConfig, LearningSignal};
use crate::focus::{
    FocusNeuronStats, FocusStatsScan, IncomingSourceScan, IncomingSourceStats, OutputErrorInfluence,
    OutputErrorScan,
};
use crate::observations::ObservationsStatistics;
use crate::propagate_layout::LearningScan;
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
    TRAINING_SCANS_OPENED.with(|c| c.set(c.get().saturating_add(1)));
    TrainingDataIterator::new(training_data, config).map_err(|e| e.to_string())
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

/// Fuse the two focus-independent passes into one training scan.
///
/// Accumulates the learning signal and — when `collect_output_errors` is set —
/// the per-output MAE from the same activation. Bit-identical to calling
/// [`crate::propagate_layout::accumulate_creature_learning`] and
/// [`crate::focus::collect_output_mean_abs_errors`] in sequence.
pub fn scan_pre_focus(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    config: &BackpropConfig,
    max_records: Option<u64>,
    rng: &mut impl Rng,
    collect_output_errors: bool,
) -> Result<PreFocusScan, String> {
    let mut learning_scan = LearningScan::new(creature, config, rng)?;
    let mut error_scan = OutputErrorScan::new(creature);
    let want_errors = collect_output_errors && error_scan.has_outputs();

    let td_cfg = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = open_training_scan(training_data, td_cfg)?;
    let mut count = 0u64;

    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if let Some(limit) = max_records
            && count >= limit
        {
            break;
        }
        // `activate_and_trace` leaves the same outputs `activate` returns in the
        // leading `creature.output` slots, so both passes read one activation.
        let traced = network.activate_and_trace(&record.inputs, creature.output);
        count += 1;
        learning_scan.observe(network, &record.outputs);
        if want_errors {
            error_scan.observe(&traced[..creature.output.min(traced.len())], &record.outputs);
        }
    }

    Ok(PreFocusScan {
        learning: learning_scan.finish(),
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
/// source ranking from a single activation per record. Bit-identical to calling
/// [`crate::focus::collect_focus_stats`],
/// [`crate::focus::collect_incoming_source_stats`] and
/// [`crate::structural::refine_sources_by_residual_with_observations`] in
/// sequence, and never materialises the sample as activation probes.
pub fn scan_post_focus(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    focus_uuid: &str,
    max_records: Option<u64>,
    observations: Option<&ObservationsStatistics>,
    prior_sources: &[RankedSource],
) -> Result<PostFocusScan, String> {
    let mut focus_scan = FocusStatsScan::new(creature, network, focus_uuid)?;
    let mut incoming_scan = IncomingSourceScan::new(creature, focus_uuid, observations)?;
    let mut residual_scan = ResidualScan::new(creature, focus_uuid, prior_sources)?;
    let scan_incoming = incoming_scan.needs_scan();

    let td_cfg = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = open_training_scan(training_data, td_cfg)?;
    let mut count = 0u64;

    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if let Some(limit) = max_records
            && count >= limit
        {
            break;
        }
        count += 1;
        let traced = network.activate_and_trace(&record.inputs, creature.output);
        focus_scan.observe(&traced, &record.outputs);
        if scan_incoming {
            incoming_scan.observe(&record.inputs, &record.outputs, &traced);
        }
        if residual_scan.wants_probe(&record.inputs, &record.outputs) {
            residual_scan.observe(&record.inputs, &record.outputs, &traced);
        }
    }

    // Fewer than two rows cannot carry a residual statistic — fall back to
    // synthetic probes exactly as the standalone refine does.
    let ranked_sources = if count < 2 {
        refine_sources_from_synthetic(
            creature,
            network,
            focus_uuid,
            prior_sources,
            observations,
        )?
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
        let output_errors =
            crate::focus::collect_output_mean_abs_errors(&creature, &mut net_a, dir.path(), Some(50))
                .unwrap();

        let mut net_b = compile_creature(&creature).unwrap();
        let mut rng_b = StdRng::seed_from_u64(42);
        let fused = scan_pre_focus(
            &creature,
            &mut net_b,
            dir.path(),
            &cfg,
            Some(50),
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
            None,
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
            limit,
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
        let fused =
            scan_post_focus(&creature, &mut network, dir.path(), "o1", Some(25), None, &[]).unwrap();
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
            Some(20),
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
            Some(20),
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
