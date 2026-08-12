//! Fixed vs marginal scorer call cost on a real creature and corpus (#112).
//!
//! Lamarck spawns the scorer once per batch, so every call pays a fixed cost —
//! process start, corpus open, per-run setup — before its first creature is
//! scored. This harness measures that cost from Lamarck's side, with **no
//! scorer changes**: it scores a directory of `baseline + N` creatures at a
//! given sample rate for several `N`, then regresses milliseconds against
//! creature count. The intercept is the fixed per-call cost, the slope the
//! marginal per-creature cost.
//!
//! Candidates are the supplied creature with one bias nudged, so every file
//! differs in content — an identical-file batch could be collapsed by a
//! content-addressed cache and would measure nothing.
//!
//! Usage (release build — debug timings are meaningless):
//!
//! ```text
//! cargo run --release --example scorer_call_cost_bench -- \
//!     <creature.json> <training-data-dir> <rust_scorer> [SIZES] [RATE] [REPEATS]
//! ```
//!
//! `SIZES` is a comma-separated list of **candidate** counts (creature count is
//! one more, the baseline); default `0,1,29`. `RATE` is the scorer sample rate
//! (`1` = full corpus); default `1`. `REPEATS` runs the whole sweep again;
//! default `1`.

use std::path::{Path, PathBuf};
use std::process::exit;

use neat_ai_lamarck::scorer::{DirectoryScorer, ExternalScorer, RecordingScorer, ScoreSample};
use neat_ai_lamarck::scorer_cost::{ScorerCallPhase, fit_calls};
use neat_core::{CreatureExport, creature_to_json_pretty, parse_creature_json};

/// The supplied creature with neuron `index % neurons` biased by `delta`.
fn perturbed(creature: &CreatureExport, index: usize, delta: f64) -> CreatureExport {
    let mut candidate = creature.clone();
    if !candidate.neurons.is_empty() {
        let slot = index % candidate.neurons.len();
        candidate.neurons[slot].bias += delta;
    }
    candidate
}

/// Write `baseline.json` plus `candidates` perturbed candidate files.
fn write_batch(dir: &Path, creature: &CreatureExport, candidates: usize) -> Result<(), String> {
    if dir.exists() {
        std::fs::remove_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let baseline = creature_to_json_pretty(creature).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("baseline.json"), &baseline).map_err(|e| e.to_string())?;
    for index in 0..candidates {
        // A distinct, tiny bias per candidate: different content, same work.
        let delta = 1e-6 * (index as f64 + 1.0);
        let json = creature_to_json_pretty(&perturbed(creature, index, delta))
            .map_err(|e| e.to_string())?;
        std::fs::write(dir.join(format!("candidate-{index:03}.json")), json)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn parse_sizes(arg: Option<&String>) -> Vec<usize> {
    match arg {
        Some(list) => list
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect(),
        None => vec![0, 1, 29],
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <creature.json> <training-data-dir> <rust_scorer> [SIZES] [RATE] [REPEATS]",
            args[0]
        );
        exit(2);
    }
    let creature_path = PathBuf::from(&args[1]);
    let training_data = PathBuf::from(&args[2]);
    let binary = PathBuf::from(&args[3]);
    let sizes = parse_sizes(args.get(4));
    let rate: f64 = args.get(5).map_or(1.0, |s| s.trim().parse().unwrap_or(1.0));
    let repeats: usize = args.get(6).map_or(1, |s| s.trim().parse().unwrap_or(1));
    if sizes.is_empty() {
        eprintln!("no batch sizes to measure");
        exit(2);
    }
    if sizes.len() < 2 {
        eprintln!("at least two distinct batch sizes are needed to separate fixed from marginal");
        exit(2);
    }

    let text = std::fs::read_to_string(&creature_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", creature_path.display()));
    let creature = parse_creature_json(&text)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", creature_path.display()));
    println!(
        "creature {} : inputs={} outputs={} neurons={} synapses={}",
        creature_path.display(),
        creature.input,
        creature.output,
        creature.neurons.len(),
        creature.synapses.len()
    );
    println!(
        "corpus {} : sample-rate={rate} repeats={repeats} sizes={sizes:?} (candidates; +1 baseline)",
        training_data.display()
    );

    let work = std::env::temp_dir().join("lamarck-scorer-call-cost");
    let external = ExternalScorer {
        binary: binary.clone(),
    };
    let recorder = RecordingScorer::new(&external);
    recorder.set_phase(if rate > 0.0 && rate < 1.0 {
        ScorerCallPhase::Screen
    } else {
        ScorerCallPhase::Promote
    });
    let sample = ScoreSample { rate, phase: 0 };

    for repeat in 1..=repeats {
        for candidates in &sizes {
            let batch = work.join(format!("batch-{candidates}"));
            write_batch(&batch, &creature, *candidates)
                .unwrap_or_else(|e| panic!("failed to write batch: {e}"));
            match recorder.score_directory_sampled(&batch, &training_data, sample) {
                Ok(scores) => {
                    let baseline = scores.get("baseline").map(|r| r.score).unwrap_or(f64::NAN);
                    eprintln!(
                        "repeat {repeat}: {} creature(s) scored, baseline={baseline:.12}",
                        candidates + 1
                    );
                }
                Err(e) => {
                    eprintln!("scorer call failed at {candidates} candidates: {e}");
                    exit(1);
                }
            }
            let _ = std::fs::remove_dir_all(&batch);
        }
    }
    let _ = std::fs::remove_dir_all(&work);

    // The recorder is the single source of truth for the timings, so the
    // printed rows and the fit can never disagree.
    let calls = recorder.drain();
    println!("\nphase,creatures,ms,sampleRate,failed");
    for call in &calls {
        println!(
            "{},{},{},{},{}",
            call.phase.label(),
            call.creatures,
            call.elapsed_ms,
            call.sample_rate.map_or(1.0, |r| r),
            call.failed
        );
    }
    let points: Vec<(u64, u128)> = calls
        .iter()
        .filter(|call| !call.failed)
        .map(|call| (call.creatures, call.elapsed_ms))
        .collect();
    let fit = fit_calls(&points);
    println!("\nfit over {} call(s) at sample-rate={rate}", fit.calls);
    println!("  mean creatures : {:.2}", fit.mean_creatures);
    println!("  mean ms        : {:.0}", fit.mean_ms);
    match (fit.fixed_ms, fit.marginal_ms_per_creature) {
        (Some(fixed), Some(marginal)) => {
            println!("  fixed ms/call  : {fixed:.0}");
            println!("  marginal ms/cr : {marginal:.0}");
            println!(
                "  r^2            : {}",
                fit.r_squared
                    .map_or_else(|| "n/a".to_string(), |r| format!("{r:.4}"))
            );
            println!(
                "  fixed share    : {}",
                fit.fixed_ms_share_at_mean
                    .map_or_else(|| "n/a".to_string(), |s| format!("{:.1}%", s * 100.0))
            );
        }
        _ => println!("  (not enough distinct batch sizes to decompose)"),
    }
}
