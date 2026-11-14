//! Predeclared sparse symbolic local-rules experiment (issue #268).
//!
//! Runs `Manifest::predeclared()`: discovers sparse rules on calibration tasks,
//! then compares the rules with sparse backprop and a derivative-free search on
//! untouched validation tasks, at equal record budgets. Writes `manifest.json`
//! and `summary.json` to OUT_DIR and prints the verdict and coverage.
//!
//! Research only: nothing in the library calls this harness.
//!
//! Usage (release build):
//!
//! ```text
//! cargo run --release --example local_rules_bench -- [OUT_DIR]
//! ```
//!
//! OUT_DIR defaults to `docs/evidence/local-rules`.

use std::path::Path;
use std::time::Instant;

use neat_ai_lamarck::local_rules::{Manifest, run_manifest, write_outputs};

fn main() -> Result<(), String> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "docs/evidence/local-rules".to_string());
    let started = Instant::now();
    let manifest = Manifest::predeclared();
    let report = run_manifest(&manifest)?;
    write_outputs(Path::new(&out), &report)?;
    let coverage = report.coverage;
    println!("overall verdict: {:?}", report.overall_verdict);
    println!(
        "coverage: planned {} completed {} failed {}",
        coverage.planned, coverage.completed, coverage.failed
    );
    println!("meta record units: {}", report.meta_cost.record_units);
    println!("output: {out}");
    println!("runtime: {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}
