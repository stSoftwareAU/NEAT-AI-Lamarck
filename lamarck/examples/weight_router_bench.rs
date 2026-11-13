//! Offline benchmark of local weight optimisers and a budget-aware router
//! (issue #266). Opt-in experiment; nothing in the optimisation loop uses it.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example weight_router_bench -- [OUT_DIR] [--manifest FILE] [--smoke]
//!     [--creature FILE --training-data DIR [--screening-records N] [--held-out-records N] [--max-records N]]
//! ```
//!
//! Writes `OUT_DIR/manifest.json`, `OUT_DIR/runs.jsonl` (one run per line) and
//! `OUT_DIR/summary.json`. `OUT_DIR` defaults to `weight-router-bench`. The
//! default manifest is the predeclared one; `--manifest` reads a JSON manifest;
//! `--smoke` runs one tiny cell.
//!
//! Corpus mode: `--creature` and `--training-data` (both required together)
//! run the same protocol on a real creature and its records instead of the
//! synthetic suite. The first `--screening-records` (default 256) records
//! screen, the last `--held-out-records` (default 512) of the first
//! `--max-records` (default 4096) are held out, and the rest are the full set.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use neat_ai_lamarck::weight_bench::suite::{CorpusSpec, Family};
use neat_ai_lamarck::weight_bench::{Manifest, predeclared_manifest, run_manifest};
use serde_json::json;

fn smoke_manifest() -> Manifest {
    Manifest {
        families: vec![Family::Tanh],
        dimensions: vec![20],
        budget_multiples: vec![10],
        calibration_seeds: vec![1],
        validation_seeds: vec![1001],
        bootstrap_resamples: 200,
        bootstrap_seed: 266,
        corpus: None,
    }
}

fn count(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    let text = args.next().ok_or(format!("{flag} needs a number"))?;
    text.parse().map_err(|e| format!("{flag} {text}: {e}"))
}

fn run() -> Result<(), String> {
    let mut out_dir = PathBuf::from("weight-router-bench");
    let mut manifest: Option<Manifest> = None;
    let mut smoke = false;
    let mut creature: Option<PathBuf> = None;
    let mut training_data: Option<PathBuf> = None;
    let (mut screening, mut held_out, mut max_records) = (256usize, 512usize, 4096usize);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => smoke = true,
            "--manifest" => {
                let path = args.next().ok_or("--manifest needs a file")?;
                let text = fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
                manifest = Some(serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?);
            }
            "--creature" => {
                creature = Some(PathBuf::from(args.next().ok_or("--creature needs a file")?));
            }
            "--training-data" => {
                training_data = Some(PathBuf::from(
                    args.next().ok_or("--training-data needs a directory")?,
                ));
            }
            "--screening-records" => screening = count(&mut args, "--screening-records")?,
            "--held-out-records" => held_out = count(&mut args, "--held-out-records")?,
            "--max-records" => max_records = count(&mut args, "--max-records")?,
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => out_dir = PathBuf::from(other),
        }
    }
    let mut manifest = match (manifest, smoke) {
        (Some(_), true) => return Err("--manifest and --smoke are mutually exclusive".into()),
        (Some(m), false) => m,
        (None, true) => smoke_manifest(),
        (None, false) => predeclared_manifest(),
    };
    match (creature, training_data) {
        (Some(creature), Some(training_data)) => {
            manifest.families = vec![Family::Corpus];
            manifest.corpus = Some(CorpusSpec {
                creature,
                training_data,
                screening_records: screening,
                held_out_records: held_out,
                max_records,
            });
        }
        (None, None) => {}
        _ => return Err("--creature and --training-data must be given together".into()),
    }

    let results = run_manifest(&manifest).map_err(|e| e.to_string())?;

    fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let pretty = |v: &serde_json::Value| serde_json::to_string_pretty(v).map_err(|e| e.to_string());
    let write = |name: &str, text: String| {
        fs::write(out_dir.join(name), text).map_err(|e| format!("{name}: {e}"))
    };
    write("manifest.json", pretty(&json!(results.manifest))?)?;
    let mut lines = String::new();
    for row in &results.runs {
        lines.push_str(&serde_json::to_string(row).map_err(|e| e.to_string())?);
        lines.push('\n');
    }
    write("runs.jsonl", lines)?;
    let summary = json!({
        "formatVersion": results.format_version,
        "manifest": results.manifest,
        "calibratedRule": results.calibrated_rule,
        "totalEvaluations": results.total_evaluations,
        "coverage": results.coverage,
        "cells": results.cells,
    });
    write("summary.json", pretty(&summary)?)?;

    for cell in &results.cells {
        let verdicts: Vec<String> = cell
            .comparisons
            .iter()
            .map(|c| {
                format!(
                    "{:?}-{:?}: {} (screening mean {:?})",
                    c.first, c.second, c.verdict, c.screening.mean
                )
            })
            .collect();
        println!(
            "{:?} D={} x{}{}: {}",
            cell.family,
            cell.dimension,
            cell.budget_multiple,
            if cell.extrapolation {
                " [extrapolation]"
            } else {
                ""
            },
            verdicts.join("; ")
        );
    }
    println!(
        "rule {:?}; runs {}/{} completed ({} failed); {} evaluations; wrote {}",
        results.calibrated_rule,
        results.coverage.completed_runs,
        results.coverage.planned_runs,
        results.coverage.failed_runs,
        results.total_evaluations,
        out_dir.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("weight_router_bench: {e}");
            ExitCode::FAILURE
        }
    }
}
