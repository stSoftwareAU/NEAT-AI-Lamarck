//! Paired benchmark for the scaled candidate-generator quotas (issue #108).
//!
//! Builds a production-shaped synthetic creature, then generates a batch at
//! each requested `--candidates` budget under the fixed pre-#108 quotas and
//! under the scaled ones, reporting the achieved batch size, why the batch
//! stopped, the generation cost and the strategy mix.
//!
//! It measures the **generator**, not the run economics: what a bigger batch is
//! worth in promote-scores per scorer-minute needs the production creature and
//! the scorer, and is the `candidate-quotas` arm of
//! `scripts/run-followup-economics.sh`.
//!
//! Usage (release build — debug timings are meaningless):
//!
//! ```text
//! cargo run --release --example candidate_quota_bench -- [INPUTS] [HIDDEN] [REPEATS] [COUNTS,...]
//! ```

use std::time::Instant;

use neat_ai_lamarck::backprop::BackpropConfig;
use neat_ai_lamarck::candidates::generate_candidate_batch;
use neat_ai_lamarck::candidates::{CandidateBatch, CandidateBudget, CandidateGenContext};
use neat_ai_lamarck::focus::{FocusNeuronStats, IncomingSourceStats};
use neat_ai_lamarck::mirror::MirrorPolicy;
use neat_ai_lamarck::observations::{ObservationsStatistics, ScalarStats, StatsMode};
use neat_core::{CreatureExport, parse_creature_json};
use rand::SeedableRng;
use rand::rngs::StdRng;

mod support;
use support::creature_json;

const FOCUS: &str = "o1";

fn scalar() -> ScalarStats {
    ScalarStats {
        count: 100,
        mean: 0.0,
        variance: 1.0,
        std_dev: 1.0,
        min: -1.0,
        max: 1.0,
        zero_count: 0,
        non_zero_count: 100,
        non_finite_count: 0,
        mean_abs: 0.5,
        rms: 1.0,
        skewness: 0.0,
        excess_kurtosis: 0.0,
        quantiles: [0.0; 7],
    }
}

/// Observations giving every input a distinct, comfortably-scoring correlation.
fn observations(inputs: usize) -> ObservationsStatistics {
    ObservationsStatistics {
        format_version: "1.0.0".into(),
        algorithm_version: "1.0.0".into(),
        mode: StatsMode::Full,
        sample_record_limit: None,
        input_count: inputs,
        output_count: 1,
        record_count: 100,
        corpus_identity: "bench".into(),
        created_at_unix: 0,
        inputs: (0..inputs).map(|_| scalar()).collect(),
        outputs: vec![scalar()],
        input_correlations: Vec::new(),
        input_target_correlations: (0..inputs)
            .map(|i| 0.9 - (i as f64 % 160.0) * 0.005)
            .collect(),
    }
}

fn generate(
    creature: &CreatureExport,
    focus: &FocusNeuronStats,
    incoming: &[IncomingSourceStats],
    obs: &ObservationsStatistics,
    count: usize,
    scale_quotas: bool,
) -> (CandidateBatch, u128) {
    let cfg = BackpropConfig::default();
    let ctx = CandidateGenContext {
        incumbent: creature,
        focus_uuid: FOCUS,
        focus_stats: focus,
        incoming,
        observations: obs,
        ranked_sources: None,
        learning: None,
        backprop: &cfg,
        structural_only: false,
        mirror: MirrorPolicy::default(),
    };
    let mut rng = StdRng::seed_from_u64(5);
    let start = Instant::now();
    let batch = generate_candidate_batch(
        &ctx,
        CandidateBudget {
            count,
            scale_quotas,
            allocation: None,
        },
        &mut rng,
    );
    (batch, start.elapsed().as_micros())
}

/// Distinct mutations in a batch, ignoring the random UUID a grown neuron
/// carries: two identical bridges are one hypothesis, scored twice.
fn distinct_mutations(batch: &CandidateBatch) -> usize {
    batch
        .candidates
        .iter()
        .map(|c| {
            let body = c
                .provenance
                .mutation
                .split_whitespace()
                .filter(|t| t.len() != 36 || t.chars().filter(|c| *c == '-').count() != 4)
                .collect::<Vec<_>>()
                .join(" ");
            format!("{}|{body}", c.provenance.strategy.label())
        })
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, default: &str| -> String {
        args.get(i).cloned().unwrap_or_else(|| default.to_string())
    };
    let inputs: usize = arg(0, "2511").parse().expect("INPUTS must be a number");
    let hidden: usize = arg(1, "12").parse().expect("HIDDEN must be a number");
    let repeats: usize = arg(2, "5").parse().expect("REPEATS must be a number");
    let counts: Vec<usize> = arg(3, "12,29,40,60,100,120,240")
        .split(',')
        .map(|c| c.trim().parse().expect("COUNTS must be numbers"))
        .collect();

    let creature = parse_creature_json(&creature_json(inputs, hidden)).unwrap();
    let obs = observations(inputs);
    let focus = FocusNeuronStats {
        neuron_uuid: FOCUS.into(),
        mean_error: Some(0.1),
        mean_abs_error: Some(0.1),
        mean_adjusted_error: Some(0.1),
        mean_derivative: Some(1.0),
        ..FocusNeuronStats::default()
    };
    let incoming = vec![IncomingSourceStats {
        synapse_index: 4,
        from_uuid: "h0".into(),
        weight: 0.02,
        is_input: false,
        input_index: None,
        mean: 0.0,
        variance: 1.0,
        std_dev: 1.0,
        correlation_with_error: Some(0.4),
        weight_signal_count: None,
        proposed_weight_delta: None,
        mean_weight_sensitivity: None,
    }];

    println!("inputs={inputs} hidden={hidden} repeats={repeats}");
    println!("\nminimum of {repeats} repeats (least-noise estimator):");
    println!(
        "| --candidates | fixed quotas | distinct | fixed generation \
         | scaled quotas | distinct | scaled generation |"
    );
    println!("|---|---|---|---|---|---|---|");
    let mut mixes = Vec::new();
    for &count in &counts {
        let mut fixed_us = u128::MAX;
        let mut scaled_us = u128::MAX;
        let mut fixed = None;
        let mut scaled = None;
        // Interleave the arms so machine drift hits them both.
        for _ in 0..repeats {
            let (batch, us) = generate(&creature, &focus, &incoming, &obs, count, false);
            fixed_us = fixed_us.min(us);
            fixed = Some(batch);
            let (batch, us) = generate(&creature, &focus, &incoming, &obs, count, true);
            scaled_us = scaled_us.min(us);
            scaled = Some(batch);
        }
        let fixed = fixed.expect("at least one repeat");
        let scaled = scaled.expect("at least one repeat");
        println!(
            "| {count} | {} ({}) | {} | {:.2} ms | {} ({}) | {} | {:.2} ms |",
            fixed.candidates.len(),
            fixed.limit.label(),
            distinct_mutations(&fixed),
            fixed_us as f64 / 1000.0,
            scaled.candidates.len(),
            scaled.limit.label(),
            distinct_mutations(&scaled),
            scaled_us as f64 / 1000.0
        );
        mixes.push((count, fixed, scaled));
    }

    println!("\nstrategy mix (fixed → scaled):");
    for (count, fixed, scaled) in &mixes {
        println!("  --candidates {count}");
        println!("    fixed:  {}", fixed.strategy_mix_summary());
        println!("    scaled: {}", scaled.strategy_mix_summary());
    }
}
