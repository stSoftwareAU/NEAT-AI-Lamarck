//! Cheap landscape probes: how flat or kinked is the loss around the start?

use serde::Serialize;

use super::objective::{BenchError, BudgetedObjective};
use crate::candidates::RANDOM_DELTA_HALF_WIDTH;

/// Relative loss change below which an axis counts as a plateau.
pub const PLATEAU_REL_TOL: f64 = 1e-6;
/// Slope asymmetry (relative to the slope magnitudes) above which an axis
/// counts as a kink.
pub const KINK_ASYMMETRY: f64 = 0.5;

/// What the probe saw along the sampled axes.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LandscapeProbe {
    /// Axes probed.
    pub axes: usize,
    /// Share of probed axes where the loss did not move.
    pub plateau_share: f64,
    /// Share of probed axes where the left and right slopes disagree.
    pub kink_share: f64,
    /// Evaluations spent by the probe (`2 * axes + 1`).
    pub evaluations: usize,
}

/// Probe up to `max_axes` evenly spaced axes with `x0 +/- h e_i`.
///
/// The probe is not free: its `2k + 1` evaluations come out of the same budget.
pub fn probe_landscape(
    obj: &mut BudgetedObjective<'_>,
    max_axes: usize,
) -> Result<LandscapeProbe, BenchError> {
    let dim = obj.dimension();
    let k = dim.min(max_axes);
    if k == 0 {
        return Err(BenchError::InvalidConfig(
            "probe needs at least one axis".into(),
        ));
    }
    let h = RANDOM_DELTA_HALF_WIDTH;
    let x0 = obj.start();
    let f0 = obj.eval(&x0)?;
    let tol = PLATEAU_REL_TOL * f0.max(1e-12);
    let (mut plateaus, mut kinks) = (0usize, 0usize);
    for j in 0..k {
        let axis = j * dim / k;
        let mut plus = x0.clone();
        plus[axis] += h;
        let mut minus = x0.clone();
        minus[axis] -= h;
        let fp = obj.eval(&plus)?;
        let fm = obj.eval(&minus)?;
        if (fp - f0).abs() <= tol && (fm - f0).abs() <= tol {
            plateaus += 1;
            continue;
        }
        let s_plus = (fp - f0) / h;
        let s_minus = (f0 - fm) / h;
        if (s_plus - s_minus).abs() > KINK_ASYMMETRY * (s_plus.abs() + s_minus.abs()) {
            kinks += 1;
        }
    }
    Ok(LandscapeProbe {
        axes: k,
        plateau_share: plateaus as f64 / k as f64,
        kink_share: kinks as f64 / k as f64,
        evaluations: 2 * k + 1,
    })
}
