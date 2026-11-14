//! Budget-aware routing between the refinement and CMA-ES arms.

use serde::{Deserialize, Serialize};

use super::cmaes::{CMA_SIGMA0, cma_es};
use super::objective::{BenchError, BudgetedObjective};
use super::probes::{LandscapeProbe, probe_landscape};
use super::refine::lamarck_refine;

/// Axes the router's landscape probe samples.
pub const ROUTER_PROBE_AXES: usize = 8;

/// Calibration grid for the budget threshold, in multiples of the dimension.
/// `None` means "never route to CMA-ES".
pub const CALIBRATION_GRID_MULTIPLES: [Option<f64>; 4] =
    [Some(10.0), Some(30.0), Some(100.0), None];

/// Calibration grid for the largest tolerated plateau-plus-kink share. The
/// last value exceeds 1, so it never blocks CMA-ES.
pub const CALIBRATION_GRID_NON_SMOOTH: [f64; 4] = [0.25, 0.5, 0.75, 1.01];

/// An optimiser arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Arm {
    /// The simplified random-axis control; not the production candidate pipeline.
    LamarckRefine,
    /// Full-covariance CMA-ES.
    CmaEs,
    /// Probe, then route to one of the other two.
    Router,
    /// The published AMM optimiser, run offline through `scripts/amm-reference.py`; never run in-process.
    AmmReference,
}

/// Thresholds deciding when the router picks CMA-ES.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RouterRule {
    /// CMA-ES needs a budget of at least this many times the dimension;
    /// `None` never routes to CMA-ES.
    pub min_budget_multiple: Option<f64>,
    /// CMA-ES is only chosen when plateau plus kink share is at most this.
    pub max_non_smooth_share: f64,
    /// Axes the probe samples.
    pub probe_axes: usize,
}

/// Choose an arm (never [`Arm::Router`]) from the budget and a landscape probe.
pub fn route(rule: &RouterRule, dimension: usize, budget: usize, probe: &LandscapeProbe) -> Arm {
    let budget_ok = rule
        .min_budget_multiple
        .is_some_and(|m| budget as f64 >= m * dimension as f64);
    let smooth = probe.plateau_share + probe.kink_share <= rule.max_non_smooth_share;
    if budget_ok && smooth {
        Arm::CmaEs
    } else {
        Arm::LamarckRefine
    }
}

/// Per-arm seed: the problem seed XOR a fixed constant for the arm.
pub fn arm_seed(arm: Arm, problem_seed: u64) -> u64 {
    problem_seed
        ^ match arm {
            Arm::LamarckRefine => 0x5245_4649_4e45,
            Arm::CmaEs => 0x434d_4145_5300,
            Arm::Router => 0x524f_5554_4552,
            Arm::AmmReference => 0x414d_4d52_4546,
        }
}

/// Probe, route using the objective's total limit, then run the chosen arm on
/// the remaining budget. The chosen arm re-evaluates the start (a known cost of
/// one evaluation, counted).
pub fn run_router(
    obj: &mut BudgetedObjective<'_>,
    rule: &RouterRule,
    problem_seed: u64,
) -> Result<Arm, BenchError> {
    let probe = probe_landscape(obj, rule.probe_axes)?;
    let chosen = route(rule, obj.dimension(), obj.limit(), &probe);
    run_arm(obj, chosen, rule, problem_seed)
}

/// Run `arm` and return the arm that actually did the work (differs from
/// `arm` only for [`Arm::Router`]).
pub fn run_arm(
    obj: &mut BudgetedObjective<'_>,
    arm: Arm,
    rule: &RouterRule,
    problem_seed: u64,
) -> Result<Arm, BenchError> {
    match arm {
        Arm::LamarckRefine => {
            lamarck_refine(obj, arm_seed(arm, problem_seed))?;
            Ok(arm)
        }
        Arm::CmaEs => {
            cma_es(obj, arm_seed(arm, problem_seed), CMA_SIGMA0)?;
            Ok(arm)
        }
        Arm::Router => run_router(obj, rule, problem_seed),
        Arm::AmmReference => Err(BenchError::InvalidConfig(
            "the AMM reference arm runs offline via the serve mode (--serve-reference), not in-process"
                .into(),
        )),
    }
}

/// One calibration observation: a problem at one budget.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationRow {
    /// Free weights.
    pub dimension: usize,
    /// Evaluation budget.
    pub budget: usize,
    /// The landscape probe at the start.
    pub probe: LandscapeProbe,
    /// Final / start screening loss of the refinement arm (a failed run counts
    /// as 1.0, no improvement).
    pub refine_normalised: f64,
    /// Final / start screening loss of the CMA-ES arm (same convention).
    pub cmaes_normalised: f64,
}

/// Pick the grid rule minimising the mean normalised loss of the arm it would
/// route each row to. Ties go to the first pair (multiples outer, non-smooth
/// inner).
pub fn calibrate(rows: &[CalibrationRow]) -> RouterRule {
    let mut best: Option<(f64, RouterRule)> = None;
    for multiple in CALIBRATION_GRID_MULTIPLES {
        for share in CALIBRATION_GRID_NON_SMOOTH {
            let rule = RouterRule {
                min_budget_multiple: multiple,
                max_non_smooth_share: share,
                probe_axes: ROUTER_PROBE_AXES,
            };
            let total: f64 = rows
                .iter()
                .map(|r| match route(&rule, r.dimension, r.budget, &r.probe) {
                    Arm::CmaEs => r.cmaes_normalised,
                    _ => r.refine_normalised,
                })
                .sum();
            let mean = if rows.is_empty() {
                0.0
            } else {
                total / rows.len() as f64
            };
            if best.as_ref().is_none_or(|(b, _)| mean < *b) {
                best = Some((mean, rule));
            }
        }
    }
    best.map(|(_, r)| r).expect("grid is non-empty")
}
