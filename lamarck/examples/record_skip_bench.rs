//! Offline controlled experiment: temporary easy-record skipping during
//! refinement (issue #267). Opt-in; nothing in the optimisation loop uses it.
//!
//! ```text
//! cargo run --release --example record_skip_bench -- OUT_DIR [--smoke] [--manifest FILE]
//!     [--creature FILE --training-data DIR [--validation-records N] [--max-records N]]
//! ```
//!
//! Writes `manifest.json`, `runs.jsonl` and `summary.json` into `OUT_DIR`. The
//! creature and training-data paths are only ever read.

use std::path::PathBuf;
use std::process::ExitCode;

use neat_ai_lamarck::record_skip::experiment::{
    CorpusSource, Manifest, load_corpus, run_manifest, write_outputs,
};
use neat_core::parse_creature_json;

struct Args {
    out: PathBuf,
    smoke: bool,
    manifest: Option<PathBuf>,
    creature: Option<PathBuf>,
    training: Option<PathBuf>,
    validation_records: usize,
    max_records: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut out = None;
    let mut a = Args {
        out: PathBuf::new(),
        smoke: false,
        manifest: None,
        creature: None,
        training: None,
        validation_records: 2000,
        max_records: 20000,
    };
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--smoke" => a.smoke = true,
            "--manifest" => a.manifest = Some(value("--manifest")?.into()),
            "--creature" => a.creature = Some(value("--creature")?.into()),
            "--training-data" => a.training = Some(value("--training-data")?.into()),
            "--validation-records" => {
                a.validation_records = value("--validation-records")?
                    .parse()
                    .map_err(|e| format!("--validation-records: {e}"))?;
            }
            "--max-records" => {
                a.max_records = value("--max-records")?
                    .parse()
                    .map_err(|e| format!("--max-records: {e}"))?;
            }
            s if s.starts_with("--") => return Err(format!("unknown option {s}")),
            _ if out.is_none() => out = Some(PathBuf::from(arg)),
            _ => return Err(format!("unexpected argument {arg}")),
        }
    }
    a.out = out.ok_or("usage: record_skip_bench OUT_DIR [--smoke] [--manifest FILE] [--creature FILE --training-data DIR ...]")?;
    if a.creature.is_some() != a.training.is_some() {
        return Err("--creature and --training-data must be given together".to_string());
    }
    Ok(a)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let manifest = match &args.manifest {
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?
        }
        None if args.smoke => Manifest::smoke(),
        None => Manifest::predeclared(),
    };
    let source = match (&args.creature, &args.training) {
        (Some(c), Some(t)) => {
            let text = std::fs::read_to_string(c).map_err(|e| format!("{}: {e}", c.display()))?;
            let parent = parse_creature_json(&text).map_err(|e| e.to_string())?;
            let corpus = load_corpus(t, &parent, args.validation_records, args.max_records)?;
            CorpusSource::Files { corpus, parent }
        }
        _ => CorpusSource::Synthetic,
    };
    let report = run_manifest(&manifest, &source)?;
    write_outputs(&args.out, &report)?;

    println!("source: {}", report.source);
    println!(
        "chosen schedule: ewmaAlpha={} revisitShare={}",
        report.chosen_schedule.ewma_alpha, report.chosen_schedule.revisit_share
    );
    println!(
        "coverage: planned={} completed={} failed={}",
        report.coverage.planned, report.coverage.completed, report.coverage.failed
    );
    for (name, c) in [
        ("calibration", &report.calibration_cost),
        ("validation", &report.validation_cost),
    ] {
        println!(
            "{name} cost: runs={} recordPasses={} revisitPasses={} scheduleOps={}",
            c.runs, c.record_passes, c.revisit_passes, c.schedule_ops
        );
    }
    println!(
        "{:<28} {:>3} {:>12} {:>12} {:>12} {:>9} {:>9}  verdict",
        "comparison", "n", "meanDiff", "ciLow", "ciHigh", "passesT", "passesC"
    );
    let f = |x: Option<f64>| x.map_or("n/a".to_string(), |v| format!("{v:.6}"));
    for c in &report.comparisons {
        println!(
            "{:<28} {:>3} {:>12} {:>12} {:>12} {:>9} {:>9}  {:?}",
            c.name,
            c.n_pairs,
            f(c.mean_validation_diff),
            f(c.ci_low),
            f(c.ci_high),
            c.treatment_record_passes
                .map_or("n/a".to_string(), |v| format!("{v:.0}")),
            c.control_record_passes
                .map_or("n/a".to_string(), |v| format!("{v:.0}")),
            c.verdict
        );
    }
    println!("outputs written to {}", args.out.display());
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
