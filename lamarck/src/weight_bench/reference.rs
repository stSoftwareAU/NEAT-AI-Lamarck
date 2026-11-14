//! The published AMM optimiser as an offline reference arm.
//!
//! AMM runs in a separate process (`scripts/amm-reference.py`). This module
//! serves it the same budgeted objective the in-process arms see, one JSON
//! line per evaluation, and later compares its rows with theirs.
//!
//! Protocol (one JSON value per line, every reply flushed):
//!
//! 1. the server writes a header
//!    `{"dimension":D,"budget":B,"seed":S,"start":[...],"boxHalfWidth":0.5}`;
//! 2. the client sends a JSON array of `D` numbers per evaluation and the
//!    server replies `{"loss":v}`;
//! 3. the client sends `done` (or closes its output) to finish;
//! 4. the server writes `{"row":{...}}` with the finished run row.
//!
//! Any parse or evaluation failure (including exceeding the budget) is
//! answered with `{"error":"..."}`, recorded on the row and ends the run.

use std::io::{BufRead, Write};

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::experiment::{
    CellSummary, Coverage, Manifest, RunRow, build_one, finish_row, load_manifest_corpus,
    summarise_cells,
};
use super::objective::BenchError;
use super::router::{Arm, arm_seed};
use super::suite::Family;

/// Half-width of the box handed to AMM, which needs a finite search region:
/// the start weight plus or minus this, per weight. Predeclared, not tuned.
pub const REFERENCE_BOX_HALF_WIDTH: f64 = 0.5;

/// Which validation run to serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceRequest {
    /// Squash family.
    pub family: Family,
    /// Free weights.
    pub dimension: usize,
    /// Problem seed (must be a validation seed).
    pub seed: u64,
    /// Budget as a multiple of the dimension.
    pub budget_multiple: usize,
}

fn invalid<T>(message: String) -> Result<T, BenchError> {
    Err(BenchError::InvalidConfig(message))
}

fn io_error(e: std::io::Error) -> BenchError {
    BenchError::Evaluation(format!("reference I/O failed: {e}"))
}

fn send(output: &mut impl Write, value: &serde_json::Value) -> Result<(), BenchError> {
    writeln!(output, "{value}").map_err(io_error)?;
    output.flush().map_err(io_error)
}

/// Serve one validation run to an external optimiser; see the module docs for
/// the protocol. The returned row has also been written as the final line.
///
/// A failed run is recorded on the row (`error`), not returned as `Err`; `Err`
/// means the request was refused or the pipes broke.
pub fn serve_reference(
    m: &Manifest,
    req: &ReferenceRequest,
    input: impl BufRead,
    mut output: impl Write,
) -> Result<RunRow, BenchError> {
    m.validate()?;
    if !m.families.contains(&req.family) {
        return invalid(format!("family {:?} is not in the manifest", req.family));
    }
    if !m.dimensions.contains(&req.dimension) {
        return invalid(format!(
            "dimension {} is not in the manifest",
            req.dimension
        ));
    }
    if !m.budget_multiples.contains(&req.budget_multiple) {
        return invalid(format!(
            "budgetMultiple {} is not in the manifest",
            req.budget_multiple
        ));
    }
    if !m.validation_seeds.contains(&req.seed) {
        return invalid(format!(
            "seed {} is not a validation seed; the reference arm only runs on validation seeds",
            req.seed
        ));
    }
    let corpus = load_manifest_corpus(m)?;
    let (problem, objective) = build_one(&corpus, req.family, req.dimension, req.seed)?;
    let budget = req.budget_multiple * req.dimension;
    send(
        &mut output,
        &serde_json::json!({
            "dimension": req.dimension,
            "budget": budget,
            "seed": arm_seed(Arm::AmmReference, req.seed),
            "start": objective.start(),
            "boxHalfWidth": REFERENCE_BOX_HALF_WIDTH,
        }),
    )?;

    let mut input = input;
    let row = finish_row(
        &problem,
        &objective,
        req.budget_multiple,
        "validation",
        Arm::AmmReference,
        0,
        |budgeted| {
            let mut line = String::new();
            loop {
                line.clear();
                if input.read_line(&mut line).map_err(io_error)? == 0 {
                    break;
                }
                let text = line.trim();
                if text.is_empty() {
                    continue;
                }
                if text == "done" {
                    break;
                }
                let step = serde_json::from_str::<Vec<f64>>(text)
                    .map_err(|e| BenchError::Evaluation(format!("cannot parse point: {e}")))
                    .and_then(|x| budgeted.eval(&x));
                match step {
                    Ok(loss) => send(&mut output, &serde_json::json!({ "loss": loss }))?,
                    Err(e) => {
                        send(&mut output, &serde_json::json!({ "error": e.to_string() }))?;
                        return Err(e);
                    }
                }
            }
            Ok(Arm::AmmReference)
        },
    );
    send(&mut output, &serde_json::json!({ "row": row }))?;
    Ok(row)
}

/// The AMM reference compared with the in-process arms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReferenceResults {
    /// Output format version.
    pub format_version: u32,
    /// The manifest the reference rows belong to.
    pub manifest: Manifest,
    /// Budgeted + preparation + confirmation evaluations of the reference
    /// rows. Each AMM run rebuilds its problem in its own serve process, so
    /// preparation is correctly summed per row here (unlike the main
    /// experiment, which builds each problem once).
    pub total_evaluations: usize,
    /// Planned versus finished reference runs; a planned run with no row is
    /// counted as neither completed nor failed.
    pub coverage: Coverage,
    /// Per-cell comparisons: AMM vs refine, AMM vs CMA-ES, AMM vs router.
    pub cells: Vec<CellSummary>,
}

/// Compare the AMM reference rows with the in-process `runs` using the same
/// paired bootstrap and verdict rule as the main experiment.
pub fn summarise_reference(
    m: &Manifest,
    runs: &[RunRow],
    reference: &[RunRow],
) -> Result<ReferenceResults, BenchError> {
    m.validate()?;
    let mut seen = Vec::new();
    for r in reference {
        let key = (r.family, r.dimension, r.budget_multiple, r.seed);
        if r.arm != Arm::AmmReference {
            return invalid(format!(
                "reference row for seed {} has arm {:?}, expected AmmReference",
                r.seed, r.arm
            ));
        }
        if r.phase != "validation" {
            return invalid(format!(
                "reference row for seed {} has phase {:?}, expected \"validation\"",
                r.seed, r.phase
            ));
        }
        if !m.families.contains(&r.family)
            || !m.dimensions.contains(&r.dimension)
            || !m.budget_multiples.contains(&r.budget_multiple)
            || !m.validation_seeds.contains(&r.seed)
        {
            return invalid(format!(
                "reference row ({:?}, dimension {}, budget multiple {}, seed {}) is not in the manifest's validation plan",
                r.family, r.dimension, r.budget_multiple, r.seed
            ));
        }
        if seen.contains(&key) {
            return invalid(format!(
                "duplicate reference row ({:?}, dimension {}, budget multiple {}, seed {})",
                r.family, r.dimension, r.budget_multiple, r.seed
            ));
        }
        seen.push(key);
    }
    let planned =
        m.families.len() * m.dimensions.len() * m.budget_multiples.len() * m.validation_seeds.len();
    let failed = reference.iter().filter(|r| r.error.is_some()).count();
    let total_evaluations = reference
        .iter()
        .map(|r| r.evaluations_used + r.preparation_evaluations + r.confirmation_evaluations)
        .sum();
    let mut all = runs.to_vec();
    all.extend_from_slice(reference);
    let pairs = [
        (Arm::AmmReference, Arm::LamarckRefine),
        (Arm::AmmReference, Arm::CmaEs),
        (Arm::AmmReference, Arm::Router),
    ];
    let mut rng = StdRng::seed_from_u64(m.bootstrap_seed);
    let cells = summarise_cells(m, &all, &pairs, &mut rng);
    Ok(ReferenceResults {
        format_version: 1,
        manifest: m.clone(),
        total_evaluations,
        coverage: Coverage {
            planned_runs: planned,
            completed_runs: reference.len() - failed,
            failed_runs: failed,
        },
        cells,
    })
}
