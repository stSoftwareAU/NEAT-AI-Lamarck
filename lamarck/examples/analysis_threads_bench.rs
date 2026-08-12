//! Paired benchmark for the parallel analysis scans (issue #107).
//!
//! Builds a production-shaped synthetic creature and training sample, then
//! times the two fused analysis scans at each requested worker count over
//! identical inputs — and asserts the accumulators are bit-identical, so a
//! faster arm that quietly changed the numbers cannot be reported as a win.
//!
//! Usage (release build — debug timings are meaningless):
//!
//! ```text
//! cargo run --release --example analysis_threads_bench -- [RECORDS] [INPUTS] [HIDDEN] [REPEATS] [THREADS,...]
//! ```

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use neat_ai_lamarck::analysis::{ScanBudget, scan_post_focus, scan_pre_focus};
use neat_ai_lamarck::backprop::BackpropConfig;
use neat_ai_lamarck::structural::RankedSource;
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

/// One timed analysis phase: both fused scans at `threads` workers.
fn analysis(
    creature: &CreatureExport,
    data: &Path,
    limit: Option<u64>,
    prior: &[RankedSource],
    threads: usize,
) -> (u128, u128, String) {
    let mut network = compile_creature(creature).unwrap();
    let cfg = BackpropConfig::default();
    let mut rng = StdRng::seed_from_u64(11);

    let start = Instant::now();
    let pre = scan_pre_focus(
        creature,
        &mut network,
        data,
        &cfg,
        ScanBudget::new(limit, threads),
        &mut rng,
        true,
    )
    .unwrap();
    let pre_ms = start.elapsed().as_millis();

    let start = Instant::now();
    let post = scan_post_focus(
        creature,
        &mut network,
        data,
        FOCUS,
        ScanBudget::new(limit, threads),
        None,
        prior,
    )
    .unwrap();
    let post_ms = start.elapsed().as_millis();

    // Fingerprint every accumulator: a thread count that changed a number is a
    // failed benchmark, not a faster one.
    let fingerprint = format!(
        "{:?}|{:?}|{:?}|{:?}",
        pre.learning, post.focus_stats, post.incoming, post.ranked_sources
    );
    (pre_ms, post_ms, fingerprint)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, default: &str| -> String {
        args.get(i).cloned().unwrap_or_else(|| default.to_string())
    };
    let records: usize = arg(0, "25000").parse().expect("RECORDS must be a number");
    let inputs: usize = arg(1, "2511").parse().expect("INPUTS must be a number");
    let hidden: usize = arg(2, "12").parse().expect("HIDDEN must be a number");
    let repeats: usize = arg(3, "5").parse().expect("REPEATS must be a number");
    let thread_counts: Vec<usize> = arg(4, "1,2,4,8")
        .split(',')
        .map(|t| t.trim().parse().expect("THREADS must be numbers"))
        .collect();

    let dir = tempfile::tempdir().unwrap();
    write_sample(dir.path(), records, inputs);
    let creature = parse_creature_json(&creature_json(inputs, hidden)).unwrap();
    let limit = Some(records as u64);
    let prior: Vec<RankedSource> = (0..inputs.min(48))
        .map(|i| RankedSource {
            from_uuid: format!("input-{i}"),
            score: 0.0,
            direction: 0.0,
            weight_scale: 1.0,
            ols_weight: None,
        })
        .collect();

    println!("records={records} inputs={inputs} hidden={hidden} repeats={repeats}");
    let mut best: Vec<(usize, u128, u128)> = Vec::new();
    let mut reference: Option<String> = None;
    // Interleave the arms so page-cache state and machine drift hit them all.
    for _ in 0..repeats {
        for &threads in &thread_counts {
            let (pre_ms, post_ms, fingerprint) =
                analysis(&creature, dir.path(), limit, &prior, threads);
            match &reference {
                None => reference = Some(fingerprint),
                Some(expected) => assert_eq!(
                    *expected, fingerprint,
                    "analysis at {threads} threads changed the accumulators"
                ),
            }
            match best.iter_mut().find(|(t, _, _)| *t == threads) {
                Some(entry) => {
                    entry.1 = entry.1.min(pre_ms);
                    entry.2 = entry.2.min(post_ms);
                }
                None => best.push((threads, pre_ms, post_ms)),
            }
        }
    }

    println!("\nminimum of {repeats} repeats (least-noise estimator):");
    println!("| threads | pre-focus ms | post-focus ms | analysis ms | speed-up |");
    println!("|---|---|---|---|---|");
    let serial = best
        .iter()
        .find(|(t, _, _)| *t == 1)
        .map(|(_, pre, post)| pre + post);
    for (threads, pre_ms, post_ms) in &best {
        let total = pre_ms + post_ms;
        let speedup = serial.map_or_else(
            || "—".to_string(),
            |s| format!("{:.2}×", s as f64 / total.max(1) as f64),
        );
        println!("| {threads} | {pre_ms} | {post_ms} | {total} | {speedup} |");
    }
    println!("\naccumulators identical at every thread count: yes");
}
