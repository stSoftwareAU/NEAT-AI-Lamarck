//! Predeclared sparse symbolic local-rules experiment (issue #268).
//!
//! Runs `Manifest::predeclared()`: discovers sparse rules on calibration tasks,
//! then compares the rules with sparse backprop and a derivative-free search on
//! untouched validation tasks, at equal record budgets. Writes `manifest.json`
//! and `summary.json` to OUT_DIR and prints the verdict and coverage.
//!
//! Research only: nothing in the library calls this harness.
//!
//! Usage (release build), synthetic tasks:
//!
//! ```text
//! cargo run --release --example local_rules_bench -- [OUT_DIR]
//! ```
//!
//! OUT_DIR defaults to `docs/evidence/local-rules`.
//!
//! Usage, a real creature and corpus (files mode). The three flags go together:
//! one calibration creature, one or more validation creatures (each a different
//! creature), and the training-data directory. Records are never written into
//! the report, only a corpus key and creature fingerprints:
//!
//! ```text
//! cargo run --release --example local_rules_bench -- OUT_DIR \
//!     --calibration-creature PATH --creature PATH [--creature PATH ...] \
//!     --training-data DIR
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use neat_ai_lamarck::local_rules::{Manifest, Report, run_files, run_manifest, write_outputs};
use neat_ai_lamarck::width::load_creature;
use neat_core::CreatureExport;

/// Parsed command line. Every value is a path or a directory.
#[derive(Debug, Default)]
struct Args {
    out: Option<String>,
    calibration: Option<PathBuf>,
    creatures: Vec<PathBuf>,
    training_data: Option<PathBuf>,
}

/// Files-mode inputs, all required together.
struct FilesInputs {
    calibration: CreatureExport,
    validation: Vec<CreatureExport>,
    training_data: PathBuf,
}

fn set_once(slot: &mut Option<PathBuf>, flag: &str, value: String) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("{flag} given more than once"));
    }
    *slot = Some(PathBuf::from(value));
    Ok(())
}

/// Plain `std` argument parsing: a positional OUT_DIR and four flags, each flag
/// taking exactly one value.
fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--calibration-creature" | "--creature" | "--training-data" => {
                let value = match args.next() {
                    Some(value) if !value.starts_with("--") => value,
                    _ => return Err(format!("{arg} needs a value")),
                };
                match arg.as_str() {
                    "--calibration-creature" => set_once(&mut parsed.calibration, &arg, value)?,
                    "--creature" => parsed.creatures.push(PathBuf::from(value)),
                    _ => set_once(&mut parsed.training_data, &arg, value)?,
                }
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            _ => {
                if parsed.out.is_some() {
                    return Err(format!("unexpected argument {arg}"));
                }
                parsed.out = Some(arg);
            }
        }
    }
    Ok(parsed)
}

fn load(path: &Path) -> Result<CreatureExport, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read creature {}: {e}", path.display()))?;
    load_creature(&text).map_err(|e| format!("creature {}: {e}", path.display()))
}

/// `None` for synthetic mode. Files mode needs all three flags, so a partial set
/// names every flag still missing before any file is read.
fn files_inputs(args: &Args) -> Result<Option<FilesInputs>, String> {
    let given =
        args.calibration.is_some() || !args.creatures.is_empty() || args.training_data.is_some();
    if !given {
        return Ok(None);
    }
    let mut missing = Vec::new();
    if args.calibration.is_none() {
        missing.push("--calibration-creature");
    }
    if args.creatures.is_empty() {
        missing.push("--creature");
    }
    if args.training_data.is_none() {
        missing.push("--training-data");
    }
    if !missing.is_empty() {
        return Err(format!(
            "files mode needs --calibration-creature, --creature and --training-data; missing {}",
            missing.join(", ")
        ));
    }
    let (Some(calibration), Some(training_data)) = (&args.calibration, &args.training_data) else {
        unreachable!("missing flags were refused above");
    };
    Ok(Some(FilesInputs {
        calibration: load(calibration)?,
        validation: args
            .creatures
            .iter()
            .map(|p| load(p))
            .collect::<Result<_, _>>()?,
        training_data: training_data.clone(),
    }))
}

fn run(report_inputs: Option<FilesInputs>, manifest: &Manifest) -> Result<Report, String> {
    match report_inputs {
        None => run_manifest(manifest),
        Some(files) => run_files(
            manifest,
            &files.calibration,
            &files.validation,
            &files.training_data,
        ),
    }
}

fn main() -> Result<(), String> {
    let args = parse_args(std::env::args().skip(1))?;
    let inputs = files_inputs(&args)?;
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| "docs/evidence/local-rules".to_string());
    let started = Instant::now();
    let manifest = Manifest::predeclared();
    let report = run(inputs, &manifest)?;
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
