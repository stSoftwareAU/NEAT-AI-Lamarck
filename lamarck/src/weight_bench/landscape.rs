//! Two-sided landscape diagnosis of a built problem's free axes.
//!
//! The sensitivity scan that chooses the free weights
//! ([`scan_sensitivities`](super::objective::scan_sensitivities)) takes one
//! one-sided step per synapse, so it can count flat axes but cannot see the
//! other side of a plateau, or tell a kink from smooth curvature. This module
//! takes, per chosen axis and from the start weights over the screening
//! records, the four steps `±h` and `±h/2` with `h =`
//! [`RANDOM_DELTA_HALF_WIDTH`], and classifies the axis (see [`classify_axis`]).
//!
//! It is a measurement only: nothing in the experiment or the optimisers reads
//! it, and its evaluations are counted separately (`evaluations`), never
//! against an arm's budget.

use serde::{Deserialize, Serialize};

use super::experiment::Manifest;
use super::objective::{BenchError, PLATEAU_REL_TOL, WeightObjective};
use super::suite::{Family, Problem, ProblemSpec, build_problem};
use crate::candidates::RANDOM_DELTA_HALF_WIDTH;

/// Second differences no larger than this fraction of the base loss (floored
/// at `1e-12`) are too small to classify: the axis is
/// [`AxisShape::Indeterminate`]. `1e-9` (declared before the evidence run)
/// only catches axes with essentially no curvature, for example exactly
/// linear ones. It is *below* the rounding floor of the forward pass: an axis
/// whose second difference is only about `1e-7` of the loss can have its ratio
/// set by rounding rather than by shape. In the committed evidence
/// (`docs/evidence/weight-router-landscape/`) that misclassifies exactly one
/// of the 2160 smooth `TANH` axes as kinked (D=100, seed 1006:
/// `D2(h) / base ≈ 1.5e-7`, ratio ≈ 13).
pub const CURVATURE_REL_TOL: f64 = 1e-9;

/// The ratio `D2(h) / D2(h/2)` of a `C²` function's second differences
/// tends to 4, because each scales with the step squared. A slope jump inside
/// the step makes them scale with the step instead, so the ratio falls towards
/// 2.
pub const SMOOTH_RATIO: f64 = 4.0;

/// Largest relative deviation of `D2(h) / D2(h/2)` from [`SMOOTH_RATIO`] still
/// called smooth.
pub const KINK_RATIO_REL_TOL: f64 = 0.25;

/// Shape of the loss along one axis around the start weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AxisShape {
    /// Both one-sided steps of `h` leave the loss unchanged within
    /// [`PLATEAU_REL_TOL`] of the base loss.
    Flat,
    /// Exactly one side is flat: the start sits on a plateau boundary.
    Edge,
    /// The second differences do not scale like a smooth function's.
    Kinked,
    /// The second differences scale like a smooth function's.
    Smooth,
    /// Neither side flat, but too little curvature to tell (see
    /// [`CURVATURE_REL_TOL`]).
    Indeterminate,
}

/// Classify one axis from the base loss `base = f(x)` and the losses at
/// `x + h`, `x - h`, `x + h/2` and `x - h/2`.
///
/// - [`AxisShape::Flat`]: `|f(x±h) - f(x)| <= PLATEAU_REL_TOL * base` on both
///   sides; [`AxisShape::Edge`]: on exactly one side.
/// - Otherwise, with `D2(s) = f(x+s) + f(x-s) - 2 f(x)`:
///   [`AxisShape::Indeterminate`] if `|D2(h)| <= CURVATURE_REL_TOL *
///   max(base, 1e-12)`; [`AxisShape::Kinked`] if `D2(h/2)` is zero or of the
///   other sign, or `D2(h) / D2(h/2)` deviates from [`SMOOTH_RATIO`] by more
///   than [`KINK_RATIO_REL_TOL`] relative; else [`AxisShape::Smooth`].
pub fn classify_axis(
    base: f64,
    f_plus_h: f64,
    f_minus_h: f64,
    f_plus_half: f64,
    f_minus_half: f64,
) -> AxisShape {
    let flat_tolerance = PLATEAU_REL_TOL * base;
    let plus_flat = (f_plus_h - base).abs() <= flat_tolerance;
    let minus_flat = (f_minus_h - base).abs() <= flat_tolerance;
    match (plus_flat, minus_flat) {
        (true, true) => return AxisShape::Flat,
        (true, false) | (false, true) => return AxisShape::Edge,
        (false, false) => {}
    }
    let d2_h = f_plus_h + f_minus_h - 2.0 * base;
    let d2_half = f_plus_half + f_minus_half - 2.0 * base;
    if d2_h.abs() <= CURVATURE_REL_TOL * base.max(1e-12) {
        return AxisShape::Indeterminate;
    }
    if d2_half == 0.0 || d2_half.signum() != d2_h.signum() {
        return AxisShape::Kinked;
    }
    let ratio = d2_h / d2_half;
    if (ratio / SMOOTH_RATIO - 1.0).abs() > KINK_RATIO_REL_TOL {
        AxisShape::Kinked
    } else {
        AxisShape::Smooth
    }
}

/// Shape counts over a list of classified axes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShapeCounts {
    /// [`AxisShape::Flat`] axes.
    pub flat: usize,
    /// [`AxisShape::Edge`] axes.
    pub edge: usize,
    /// [`AxisShape::Kinked`] axes.
    pub kinked: usize,
    /// [`AxisShape::Smooth`] axes.
    pub smooth: usize,
    /// [`AxisShape::Indeterminate`] axes.
    pub indeterminate: usize,
}

impl ShapeCounts {
    /// Count `shapes`.
    pub fn of(shapes: &[AxisShape]) -> Self {
        let mut counts = Self::default();
        for shape in shapes {
            match shape {
                AxisShape::Flat => counts.flat += 1,
                AxisShape::Edge => counts.edge += 1,
                AxisShape::Kinked => counts.kinked += 1,
                AxisShape::Smooth => counts.smooth += 1,
                AxisShape::Indeterminate => counts.indeterminate += 1,
            }
        }
        counts
    }

    fn add(&mut self, other: &Self) {
        self.flat += other.flat;
        self.edge += other.edge;
        self.kinked += other.kinked;
        self.smooth += other.smooth;
        self.indeterminate += other.indeterminate;
    }

    /// Axes counted.
    pub fn total(&self) -> usize {
        self.flat + self.edge + self.kinked + self.smooth + self.indeterminate
    }
}

/// The diagnosis of one problem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LandscapeRow {
    /// Problem family.
    pub family: Family,
    /// Free weights.
    pub dimension: usize,
    /// Problem seed.
    pub seed: u64,
    /// Shape counts over the free axes.
    #[serde(flatten)]
    pub shapes: ShapeCounts,
    /// Loss evaluations the diagnosis spent: one base plus four per axis.
    pub evaluations: usize,
    /// The one-sided plateau count of the same axes from the selection scan
    /// ([`Problem::plateau_free_axes`]), for comparison.
    pub plateau_free_axes: usize,
}

/// Shape of every free axis of `problem`, in `synapse_indices` order, and the
/// evaluations spent (`1 + 4 * dimension`).
pub fn axis_shapes(problem: &Problem) -> Result<(Vec<AxisShape>, usize), BenchError> {
    let objective = WeightObjective::new(
        problem.creature.clone(),
        problem.synapse_indices.clone(),
        &problem.screening,
    )?;
    let x0 = objective.start();
    let mut evaluations = 0usize;
    let mut eval = |x: &[f64]| {
        evaluations += 1;
        objective.loss(x).map_err(BenchError::Evaluation)
    };
    let base = eval(&x0)?;
    let h = RANDOM_DELTA_HALF_WIDTH;
    let mut shapes = Vec::with_capacity(x0.len());
    for i in 0..x0.len() {
        let mut at = |step: f64| {
            let mut x = x0.clone();
            x[i] += step;
            eval(&x)
        };
        let (plus_h, minus_h, plus_half, minus_half) =
            (at(h)?, at(-h)?, at(h / 2.0)?, at(-h / 2.0)?);
        shapes.push(classify_axis(base, plus_h, minus_h, plus_half, minus_half));
    }
    Ok((shapes, evaluations))
}

/// Diagnose one built problem.
pub fn diagnose(problem: &Problem) -> Result<LandscapeRow, BenchError> {
    let (shapes, evaluations) = axis_shapes(problem)?;
    Ok(LandscapeRow {
        family: problem.spec.family,
        dimension: problem.spec.dimension,
        seed: problem.spec.seed,
        shapes: ShapeCounts::of(&shapes),
        evaluations,
        plateau_free_axes: problem.plateau_free_axes,
    })
}

/// Diagnose every (family, dimension, validation seed) problem of a synthetic
/// manifest, in manifest order. A corpus manifest is refused.
pub fn diagnose_manifest(m: &Manifest) -> Result<Vec<LandscapeRow>, BenchError> {
    m.validate()?;
    if m.corpus.is_some() {
        return Err(BenchError::InvalidConfig(
            "the landscape diagnosis needs a synthetic manifest, not a corpus".into(),
        ));
    }
    let mut rows = Vec::new();
    for &family in &m.families {
        for &dimension in &m.dimensions {
            for &seed in &m.validation_seeds {
                let problem = build_problem(&ProblemSpec {
                    family,
                    dimension,
                    seed,
                })?;
                rows.push(diagnose(&problem)?);
            }
        }
    }
    Ok(rows)
}

/// Totals of one (family, dimension) cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LandscapeCell {
    /// Problem family.
    pub family: Family,
    /// Free weights.
    pub dimension: usize,
    /// Problems diagnosed.
    pub problems: usize,
    /// Shape totals over every axis of every problem.
    #[serde(flatten)]
    pub shapes: ShapeCounts,
    /// Total evaluations spent.
    pub evaluations: usize,
    /// Total one-sided plateau count from the selection scans.
    pub plateau_free_axes: usize,
}

/// Per (family, dimension) totals of `rows`, in manifest order.
pub fn summarise_landscape(m: &Manifest, rows: &[LandscapeRow]) -> Vec<LandscapeCell> {
    let mut cells = Vec::new();
    for &family in &m.families {
        for &dimension in &m.dimensions {
            let mut cell = LandscapeCell {
                family,
                dimension,
                problems: 0,
                shapes: ShapeCounts::default(),
                evaluations: 0,
                plateau_free_axes: 0,
            };
            for row in rows
                .iter()
                .filter(|r| r.family == family && r.dimension == dimension)
            {
                cell.problems += 1;
                cell.shapes.add(&row.shapes);
                cell.evaluations += row.evaluations;
                cell.plateau_free_axes += row.plateau_free_axes;
            }
            cells.push(cell);
        }
    }
    cells
}
