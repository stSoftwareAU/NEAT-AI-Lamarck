//! Offline benchmark of local weight optimisers (issue #266). Opt-in
//! experiment; nothing in the optimisation loop uses it. The binary keeps its
//! historical name; the budget-aware router it once measured was removed after
//! an earlier run (crate 0.1.46) found no advantage over CMA-ES alone.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example weight_router_bench -- [OUT_DIR] [--manifest FILE] [--smoke]
//!     [--creature FILE --training-data DIR [--screening-records N] [--held-out-records N] [--max-records N]]
//! ```
//!
//! Writes `OUT_DIR/manifest.json`, `OUT_DIR/runs.jsonl` (one run per line) and
//! `OUT_DIR/summary.json`; `--compare-reference` writes
//! `OUT_DIR/reference-summary.json` as well.
//!
//! Both summaries carry the provenance of the run: `crateVersion` (the crate
//! that produced it), `gitCommit` (the checkout's `HEAD`, or null), `gitDirty`
//! (whether that checkout had uncommitted changes, or null) and
//! `neatCoreSource` (the `neat-core` source pinned in `Cargo.lock`). When git
//! cannot be run, a one-line warning goes to stderr and the two git fields are
//! null. `summary.json` also carries `wallClockSeconds`, the time
//! `run_manifest` took, to three decimals.
//!
//! A corpus run's `summary.json` also carries `corpusFingerprint`: FNV-1a
//! hashes of the creature file and of the records loaded, and the record count.
//! `OUT_DIR` defaults to `weight-router-bench`. The default manifest is the predeclared one; `--manifest` reads a JSON manifest;
//! `--smoke` runs one tiny cell.
//!
//! Corpus mode: `--creature` and `--training-data` (both required together)
//! run the same protocol on a real creature and its records instead of the
//! synthetic suite. Of the first `--max-records` (default 4096) records, the
//! first `--screening-records` (default 256) screen, the last
//! `--held-out-records` (default 512) are held out, and the rest of the window
//! is the full set. The window needs screening + held-out records.
//!
//! Landscape diagnosis (synthetic manifests only, i.e. the default or
//! `--manifest`; refused with `--smoke`, corpus flags or the reference flags):
//!
//! ```text
//! weight_router_bench OUT_DIR [--manifest FILE] --landscape
//! ```
//!
//! builds every (family, dimension, validation seed) problem of the manifest,
//! classifies each free axis from the start weights with two-sided steps of
//! `±h` and `±h/2` (`h = 0.05`; see `weight_bench::landscape`) as flat, edge,
//! kinked, smooth or indeterminate, and writes `OUT_DIR/landscape.jsonl` (one
//! row per problem) and `OUT_DIR/landscape-summary.json` (totals per family and
//! dimension, the manifest, the classifier tolerances and the same provenance
//! fields as `summary.json`, including `wallClockSeconds`). It runs no arm and
//! does not write `OUT_DIR/manifest.json`.
//!
//! AMM reference arm (offline; the optimiser itself is `scripts/amm-reference.py`):
//!
//! ```text
//! weight_router_bench --serve-reference --family F --dimension D --seed S --budget-multiple K
//!     [--manifest FILE | --smoke | --creature ... --training-data ...]
//! weight_router_bench OUT_DIR --compare-reference AMM_ROWS_JSONL
//! ```
//!
//! `--serve-reference` serves one validation run over stdin/stdout (a JSON
//! header line, then one `{"loss":v}` reply per point sent; see
//! `weight_bench::reference`), using the manifest the other flags select, and
//! exits non-zero if the run recorded an error. `--compare-reference` reads
//! `OUT_DIR/manifest.json` and `OUT_DIR/runs.jsonl` from an earlier run plus
//! the AMM rows (one `RunRow` JSON per line), writes
//! `OUT_DIR/reference-summary.json` and prints one verdict line per cell.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use neat_ai_lamarck::weight_bench::landscape::{CURVATURE_REL_TOL, SMOOTH_RATIO};
use neat_ai_lamarck::weight_bench::objective::PLATEAU_REL_TOL;
use neat_ai_lamarck::weight_bench::provenance;
use neat_ai_lamarck::weight_bench::suite::{CorpusSpec, Family};
use neat_ai_lamarck::weight_bench::{
    KINK_RATIO_REL_TOL, Manifest, ReferenceRequest, RunRow, diagnose_manifest,
    predeclared_manifest, run_manifest, serve_reference, summarise_landscape, summarise_reference,
};
use serde_json::json;

/// The workspace lock file, which pins the `neat-core` source.
const NEAT_CORE_LOCK: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../Cargo.lock"));

/// Where a run's outputs came from, probed before any work or output is written.
struct Provenance {
    git_commit: serde_json::Value,
    git_dirty: serde_json::Value,
    neat_core_source: String,
}

impl Provenance {
    /// Probe git and the lock. A git failure is warned about on stderr and
    /// recorded as null; a missing `neat-core` pin is an error, so no summary is
    /// written without it.
    fn probe() -> Result<Self, String> {
        let (git_commit, git_dirty) =
            match provenance::git_state(Path::new(env!("CARGO_MANIFEST_DIR"))) {
                Ok(state) => (json!(state.commit), json!(state.dirty)),
                Err(cause) => {
                    eprintln!("weight_router_bench: warning: git fields are null: {cause}");
                    (json!(null), json!(null))
                }
            };
        Ok(Self {
            git_commit,
            git_dirty,
            neat_core_source: provenance::neat_core_source(NEAT_CORE_LOCK)?,
        })
    }

    /// Add the provenance fields to a summary object.
    fn insert_into(&self, object: &mut serde_json::Map<String, serde_json::Value>) {
        object.insert("gitCommit".into(), self.git_commit.clone());
        object.insert("gitDirty".into(), self.git_dirty.clone());
        object.insert("neatCoreSource".into(), json!(self.neat_core_source));
    }
}

fn smoke_manifest() -> Manifest {
    Manifest {
        families: vec![Family::Tanh],
        dimensions: vec![20],
        budget_multiples: vec![10],
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

fn read_rows(path: &Path) -> Result<Vec<RunRow>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row =
            serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

fn print_cells(cells: &[neat_ai_lamarck::weight_bench::CellSummary]) {
    for cell in cells {
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
}

fn compare_reference(out_dir: &Path, amm_rows: &Path) -> Result<(), String> {
    let provenance = Provenance::probe()?;
    let manifest_path = out_dir.join("manifest.json");
    let text = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest: Manifest =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let runs = read_rows(&out_dir.join("runs.jsonl"))?;
    let reference = read_rows(amm_rows)?;
    let results = summarise_reference(&manifest, &runs, &reference).map_err(|e| e.to_string())?;
    let path = out_dir.join("reference-summary.json");
    let mut value = serde_json::to_value(&results).map_err(|e| e.to_string())?;
    if let Some(object) = value.as_object_mut() {
        object.insert("crateVersion".into(), json!(env!("CARGO_PKG_VERSION")));
        provenance.insert_into(object);
    }
    let pretty = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    fs::write(&path, pretty).map_err(|e| format!("{}: {e}", path.display()))?;
    print_cells(&results.cells);
    println!(
        "reference runs {}/{} completed ({} failed); {} evaluations; wrote {}",
        results.coverage.completed_runs,
        results.coverage.planned_runs,
        results.coverage.failed_runs,
        results.total_evaluations,
        path.display()
    );
    Ok(())
}

fn landscape(out_dir: &Path, manifest: &Manifest) -> Result<(), String> {
    let provenance = Provenance::probe()?;
    let started = Instant::now();
    let rows = diagnose_manifest(manifest).map_err(|e| e.to_string())?;
    let wall_clock_seconds = (started.elapsed().as_secs_f64() * 1000.0).round() / 1000.0;
    let cells = summarise_landscape(manifest, &rows);
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let mut lines = String::new();
    for row in &rows {
        lines.push_str(&serde_json::to_string(row).map_err(|e| e.to_string())?);
        lines.push('\n');
    }
    let rows_path = out_dir.join("landscape.jsonl");
    fs::write(&rows_path, lines).map_err(|e| format!("{}: {e}", rows_path.display()))?;
    let mut summary = json!({
        "formatVersion": 1,
        "crateVersion": env!("CARGO_PKG_VERSION"),
        "wallClockSeconds": wall_clock_seconds,
        "manifest": manifest,
        "stepHalfWidth": neat_ai_lamarck::candidates::RANDOM_DELTA_HALF_WIDTH,
        "plateauRelTol": PLATEAU_REL_TOL,
        "curvatureRelTol": CURVATURE_REL_TOL,
        "smoothRatio": SMOOTH_RATIO,
        "kinkRatioRelTol": KINK_RATIO_REL_TOL,
        "totalEvaluations": rows.iter().map(|r| r.evaluations).sum::<usize>(),
        "cells": cells,
    });
    if let Some(object) = summary.as_object_mut() {
        provenance.insert_into(object);
    }
    let summary_path = out_dir.join("landscape-summary.json");
    let pretty = serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?;
    fs::write(&summary_path, pretty).map_err(|e| format!("{}: {e}", summary_path.display()))?;
    for cell in &cells {
        println!(
            "{:?} D={} ({} problems): flat {} edge {} kinked {} smooth {} indeterminate {}; one-sided plateau {}; {} evaluations",
            cell.family,
            cell.dimension,
            cell.problems,
            cell.shapes.flat,
            cell.shapes.edge,
            cell.shapes.kinked,
            cell.shapes.smooth,
            cell.shapes.indeterminate,
            cell.plateau_free_axes,
            cell.evaluations
        );
    }
    println!(
        "wrote {} and {}",
        rows_path.display(),
        summary_path.display()
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let mut out_dir = PathBuf::from("weight-router-bench");
    let mut manifest: Option<Manifest> = None;
    let mut smoke = false;
    let mut creature: Option<PathBuf> = None;
    let mut training_data: Option<PathBuf> = None;
    let (mut screening, mut held_out, mut max_records) = (256usize, 512usize, 4096usize);
    let mut serve = false;
    let mut family: Option<Family> = None;
    let mut dimension: Option<usize> = None;
    let mut seed: Option<u64> = None;
    let mut budget_multiple: Option<usize> = None;
    let mut compare: Option<PathBuf> = None;
    let mut landscape_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => smoke = true,
            "--landscape" => landscape_only = true,
            "--serve-reference" => serve = true,
            "--family" => {
                let text = args.next().ok_or("--family needs a name")?;
                family = Some(
                    serde_json::from_value(serde_json::Value::String(text.clone()))
                        .map_err(|e| format!("--family {text}: {e}"))?,
                );
            }
            "--dimension" => dimension = Some(count(&mut args, "--dimension")?),
            "--seed" => {
                let text = args.next().ok_or("--seed needs a number")?;
                seed = Some(text.parse().map_err(|e| format!("--seed {text}: {e}"))?);
            }
            "--budget-multiple" => budget_multiple = Some(count(&mut args, "--budget-multiple")?),
            "--compare-reference" => {
                compare = Some(PathBuf::from(
                    args.next().ok_or("--compare-reference needs a file")?,
                ));
            }
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
    if landscape_only
        && (smoke || serve || compare.is_some() || creature.is_some() || training_data.is_some())
    {
        return Err("--landscape needs a synthetic manifest (the default or --manifest) and no --smoke, corpus or reference flags".into());
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

    if landscape_only {
        return landscape(&out_dir, &manifest);
    }
    if serve && compare.is_some() {
        return Err("--serve-reference and --compare-reference are mutually exclusive".into());
    }
    if let Some(amm_rows) = compare {
        return compare_reference(&out_dir, &amm_rows);
    }
    if serve {
        let request = ReferenceRequest {
            family: family.ok_or("--serve-reference needs --family")?,
            dimension: dimension.ok_or("--serve-reference needs --dimension")?,
            seed: seed.ok_or("--serve-reference needs --seed")?,
            budget_multiple: budget_multiple.ok_or("--serve-reference needs --budget-multiple")?,
        };
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        let row = serve_reference(&manifest, &request, stdin.lock(), stdout.lock())
            .map_err(|e| e.to_string())?;
        std::io::stdout().flush().map_err(|e| e.to_string())?;
        return match row.error {
            Some(e) => Err(format!("reference run failed: {e}")),
            None => Ok(()),
        };
    }

    let provenance = Provenance::probe()?;
    let started = Instant::now();
    let results = run_manifest(&manifest).map_err(|e| e.to_string())?;
    let wall_clock_seconds = (started.elapsed().as_secs_f64() * 1000.0).round() / 1000.0;

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
    let mut summary = json!({
        "formatVersion": results.format_version,
        "crateVersion": env!("CARGO_PKG_VERSION"),
        "wallClockSeconds": wall_clock_seconds,
        "manifest": results.manifest,
        "totalEvaluations": results.total_evaluations,
        "coverage": results.coverage,
        "cells": results.cells,
    });
    if let Some(object) = summary.as_object_mut() {
        provenance.insert_into(object);
    }
    // Corpus runs name the exact model and records; the synthetic suite is
    // fixed by the crate version and the manifest seeds.
    if let (Some(fp), Some(object)) = (&results.corpus_fingerprint, summary.as_object_mut()) {
        object.insert("corpusFingerprint".into(), json!(fp));
    }
    write("summary.json", pretty(&summary)?)?;

    print_cells(&results.cells);
    println!(
        "runs {}/{} completed ({} failed); {} evaluations; wrote {}",
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
