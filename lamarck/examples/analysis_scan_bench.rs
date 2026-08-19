//! Paired benchmark for the per-experiment analysis scans (issue #105).
//!
//! Builds a synthetic creature and training sample, then times the analysis
//! phase two ways over identical inputs:
//!
//! * `legacy` — the five separate passes the run loop used to make
//!   (learning, output MAE, focus stats, incoming sources, residual refine).
//! * `fused`  — the two scans [`scan_pre_focus`] / [`scan_post_focus`] make.
//!
//! Usage (release build — debug timings are meaningless):
//!
//! ```text
//! cargo run --release --example analysis_scan_bench -- [MODE] [RECORDS] [INPUTS] [HIDDEN]
//! ```
//!
//! `MODE` is `legacy`, `fused` or `both` (default). Run each mode under
//! `/usr/bin/time -l` (macOS) or `/usr/bin/time -v` (Linux) for peak RSS.

use std::path::Path;
use std::time::Instant;

use neat_ai_lamarck::analysis::{ScanBudget, scan_post_focus, scan_pre_focus};
use neat_ai_lamarck::backprop::BackpropConfig;
use neat_ai_lamarck::focus::{
    collect_focus_stats, collect_incoming_source_stats, collect_output_mean_abs_errors,
};
use neat_ai_lamarck::propagate_layout::accumulate_creature_learning;
use neat_ai_lamarck::structural::{RankedSource, refine_sources_by_residual_with_observations};
use neat_core::{CreatureExport, compile_creature, parse_creature_json};
use rand::SeedableRng;
use rand::rngs::StdRng;

mod support;
use support::{creature_json, write_sample};

const FOCUS: &str = "o1";

/// The five separate passes, as the run loop made them before issue #105.
fn legacy(creature: &CreatureExport, data: &Path, limit: Option<u64>, prior: &[RankedSource]) {
    let mut network = compile_creature(creature).unwrap();
    let cfg = BackpropConfig::default();
    let mut rng = StdRng::seed_from_u64(11);
    let mut lap = Instant::now();
    let report = |name: &str, lap: &mut Instant| {
        println!("    {name}: {} ms", lap.elapsed().as_millis());
        *lap = Instant::now();
    };
    let learning =
        accumulate_creature_learning(creature, &mut network, data, &cfg, limit, &mut rng).unwrap();
    report("1 learning       ", &mut lap);
    let errors = collect_output_mean_abs_errors(creature, &mut network, data, limit).unwrap();
    report("2 output MAE     ", &mut lap);
    let stats = collect_focus_stats(creature, &mut network, data, FOCUS, limit).unwrap();
    report("3 focus stats    ", &mut lap);
    let incoming =
        collect_incoming_source_stats(creature, &mut network, data, FOCUS, limit, None).unwrap();
    report("4 incoming stats ", &mut lap);
    let ranked = refine_sources_by_residual_with_observations(
        creature,
        &mut network,
        data,
        FOCUS,
        prior,
        limit,
        None,
    )
    .unwrap();
    report("5 residual refine", &mut lap);
    std::hint::black_box((learning, errors, stats, incoming, ranked));
}

/// The two fused scans, folded serially (one worker).
fn fused(creature: &CreatureExport, data: &Path, limit: Option<u64>, prior: &[RankedSource]) {
    let mut network = compile_creature(creature).unwrap();
    let cfg = BackpropConfig::default();
    let mut rng = StdRng::seed_from_u64(11);
    let mut lap = Instant::now();
    let pre = scan_pre_focus(
        creature,
        &mut network,
        data,
        &cfg,
        ScanBudget::serial(limit),
        &mut rng,
        true,
    )
    .unwrap();
    println!("    A pre-focus     : {} ms", lap.elapsed().as_millis());
    lap = Instant::now();
    let post = scan_post_focus(
        creature,
        &mut network,
        data,
        FOCUS,
        ScanBudget::serial(limit),
        None,
        prior,
    )
    .unwrap();
    println!("    B post-focus    : {} ms", lap.elapsed().as_millis());
    std::hint::black_box((pre, post));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, default: &str| -> String {
        args.get(i).cloned().unwrap_or_else(|| default.to_string())
    };
    let mode = arg(0, "both");
    let records: usize = arg(1, "25000").parse().expect("RECORDS must be a number");
    let inputs: usize = arg(2, "256").parse().expect("INPUTS must be a number");
    let hidden: usize = arg(3, "32").parse().expect("HIDDEN must be a number");

    let dir = tempfile::tempdir().unwrap();
    write_sample(dir.path(), records, inputs);
    let creature = parse_creature_json(&creature_json(inputs, hidden)).unwrap();
    let limit = Some(records as u64);
    // Rank every input as an unused source so the residual pass has work to do.
    let prior: Vec<RankedSource> = (0..inputs.min(48))
        .map(|i| RankedSource {
            from_uuid: format!("input-{i}"),
            score: 0.0,
            direction: 0.0,
            weight_scale: 1.0,
            ols_weight: None,
        })
        .collect();

    let repeats: usize = arg(4, "1").parse().expect("REPEATS must be a number");

    println!("records={records} inputs={inputs} hidden={hidden} repeats={repeats}");
    // Alternate the two modes so page-cache state and machine drift hit both.
    for _ in 0..repeats {
        if mode == "legacy" || mode == "both" {
            let start = Instant::now();
            legacy(&creature, dir.path(), limit, &prior);
            println!("legacy (5 scans): {} ms", start.elapsed().as_millis());
        }
        if mode == "fused" || mode == "both" {
            let start = Instant::now();
            fused(&creature, dir.path(), limit, &prior);
            println!("fused  (2 scans): {} ms", start.elapsed().as_millis());
        }
    }
}
