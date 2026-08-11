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

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use neat_ai_lamarck::analysis::{scan_post_focus, scan_pre_focus};
use neat_ai_lamarck::backprop::BackpropConfig;
use neat_ai_lamarck::focus::{
    collect_focus_stats, collect_incoming_source_stats, collect_output_mean_abs_errors,
};
use neat_ai_lamarck::propagate_layout::accumulate_creature_learning;
use neat_ai_lamarck::structural::{
    RankedSource, refine_sources_by_residual_with_observations,
};
use neat_core::{CreatureExport, compile_creature, parse_creature_json};
use rand::SeedableRng;
use rand::rngs::StdRng;

const FOCUS: &str = "o1";

/// Synthetic creature: `inputs` inputs, `hidden` TANH hiddens, one output.
fn creature_json(inputs: usize, hidden: usize) -> String {
    let mut neurons = String::new();
    let mut synapses = String::new();
    for h in 0..hidden {
        if h > 0 {
            neurons.push(',');
        }
        let bias = (h as f64 % 7.0) * 0.01 - 0.03;
        neurons.push_str(&format!(
            r#"{{"type":"hidden","uuid":"h{h}","bias":{bias},"squash":"TANH"}}"#
        ));
        // Each hidden reads a deterministic slice of the inputs.
        for k in 0..4 {
            let i = (h * 4 + k) % inputs;
            let weight = 0.05 + ((h + k) as f64 % 11.0) * 0.01;
            synapses.push_str(&format!(
                r#"{{"fromUUID":"input-{i}","toUUID":"h{h}","weight":{weight}}},"#
            ));
        }
        synapses.push_str(&format!(
            r#"{{"fromUUID":"h{h}","toUUID":"o1","weight":{}}},"#,
            0.02 + (h as f64 % 5.0) * 0.01
        ));
    }
    let synapses = synapses.trim_end_matches(',');
    format!(
        r#"{{"semanticVersion":"4.0.0","forwardOnly":true,"input":{inputs},"output":1,
           "neurons":[{neurons},{{"type":"output","uuid":"o1","bias":0.01,"squash":"IDENTITY"}}],
           "synapses":[{synapses}]}}"#
    )
}

/// Deterministic xorshift sample: `inputs` inputs + one target per record.
fn write_sample(dir: &Path, records: usize, inputs: usize) {
    let mut file = std::io::BufWriter::new(std::fs::File::create(dir.join("0.bin")).unwrap());
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..records {
        for _ in 0..=inputs {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
            file.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    file.flush().unwrap();
}

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

/// The two fused scans.
fn fused(creature: &CreatureExport, data: &Path, limit: Option<u64>, prior: &[RankedSource]) {
    let mut network = compile_creature(creature).unwrap();
    let cfg = BackpropConfig::default();
    let mut rng = StdRng::seed_from_u64(11);
    let pre = scan_pre_focus(creature, &mut network, data, &cfg, limit, &mut rng, true).unwrap();
    let post = scan_post_focus(creature, &mut network, data, FOCUS, limit, None, prior).unwrap();
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

    println!("records={records} inputs={inputs} hidden={hidden}");
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
