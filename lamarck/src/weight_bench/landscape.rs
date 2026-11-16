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
//! it, and its evaluations are counted separately, never against an arm's
//! budget. The diagnosis's own evaluations are `evaluations`; the selection scan
//! that built each problem is `preparation_evaluations`, counted once per
//! problem. [`landscape_total_evaluations`] sums both.
//!
//! A problem that cannot be built or diagnosed does not stop the run: it is
//! recorded as a row with `error` set (see [`diagnose_specs`]) and counted in
//! [`landscape_coverage`].

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
/// of the 1920 `TANH` axes as kinked (D=100, seed 1006:
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
///
/// A failed problem has `error` set, zero shapes and a zero
/// `plateau_free_axes`, and keeps the evaluations it actually spent. One whose
/// [`build_problem`] failed reports `preparation_evaluations` 0, because a
/// failed build does not expose how much of its selection scan ran.
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
    /// Diagnosis evaluations: one base plus four per axis. The selection scan
    /// is in `preparation_evaluations`. On failure, the evaluations made
    /// before the error.
    pub evaluations: usize,
    /// Loss evaluations of the selection scan that built the problem
    /// ([`Problem::preparation_evaluations`]), counted once per problem.
    pub preparation_evaluations: usize,
    /// The one-sided plateau count of the same axes from the selection scan
    /// ([`Problem::plateau_free_axes`]), for comparison.
    pub plateau_free_axes: usize,
    /// Why building or diagnosing the problem failed, if it did.
    pub error: Option<String>,
}

/// Classify every axis of `x0` under `loss`, stopping at the first failed
/// evaluation. Returns the shapes (or that error) with the evaluations made
/// either way, so a failure partway through still reports its cost.
fn classify_axes(
    x0: &[f64],
    mut loss: impl FnMut(&[f64]) -> Result<f64, BenchError>,
) -> (Result<Vec<AxisShape>, BenchError>, usize) {
    let mut evaluations = 0usize;
    let shapes = classify_each_axis(x0, &mut |x: &[f64]| {
        evaluations += 1;
        loss(x)
    });
    (shapes, evaluations)
}

fn classify_each_axis(
    x0: &[f64],
    eval: &mut impl FnMut(&[f64]) -> Result<f64, BenchError>,
) -> Result<Vec<AxisShape>, BenchError> {
    let base = eval(x0)?;
    let h = RANDOM_DELTA_HALF_WIDTH;
    let mut shapes = Vec::with_capacity(x0.len());
    for i in 0..x0.len() {
        let mut at = |step: f64| {
            let mut x = x0.to_vec();
            x[i] += step;
            eval(&x)
        };
        let (plus_h, minus_h, plus_half, minus_half) =
            (at(h)?, at(-h)?, at(h / 2.0)?, at(-h / 2.0)?);
        shapes.push(classify_axis(base, plus_h, minus_h, plus_half, minus_half));
    }
    Ok(shapes)
}

/// [`axis_shapes`] with the evaluations reported on failure too (0 when the
/// objective cannot be built).
fn measure_axes(problem: &Problem) -> (Result<Vec<AxisShape>, BenchError>, usize) {
    match WeightObjective::new(
        problem.creature.clone(),
        problem.synapse_indices.clone(),
        &problem.screening,
    ) {
        Ok(objective) => classify_axes(&objective.start(), |x| {
            objective.loss(x).map_err(BenchError::Evaluation)
        }),
        Err(e) => (Err(e), 0),
    }
}

/// Shape of every free axis of `problem`, in `synapse_indices` order, and the
/// evaluations spent (`1 + 4 * dimension`).
pub fn axis_shapes(problem: &Problem) -> Result<(Vec<AxisShape>, usize), BenchError> {
    let (shapes, evaluations) = measure_axes(problem);
    Ok((shapes?, evaluations))
}

fn completed_row(problem: &Problem, shapes: &[AxisShape], evaluations: usize) -> LandscapeRow {
    LandscapeRow {
        family: problem.spec.family,
        dimension: problem.spec.dimension,
        seed: problem.spec.seed,
        shapes: ShapeCounts::of(shapes),
        evaluations,
        preparation_evaluations: problem.preparation_evaluations,
        plateau_free_axes: problem.plateau_free_axes,
        error: None,
    }
}

fn failed_row(
    spec: &ProblemSpec,
    evaluations: usize,
    preparation_evaluations: usize,
    error: &BenchError,
) -> LandscapeRow {
    LandscapeRow {
        family: spec.family,
        dimension: spec.dimension,
        seed: spec.seed,
        shapes: ShapeCounts::default(),
        evaluations,
        preparation_evaluations,
        plateau_free_axes: 0,
        error: Some(error.to_string()),
    }
}

/// Diagnose one built problem.
pub fn diagnose(problem: &Problem) -> Result<LandscapeRow, BenchError> {
    let (shapes, evaluations) = axis_shapes(problem)?;
    Ok(completed_row(problem, &shapes, evaluations))
}

/// Build each of `specs` with `build` and diagnose it, in order. A failure to
/// build or diagnose is recorded on that problem's row (see [`LandscapeRow`]),
/// never propagated, so every spec yields exactly one row.
pub fn diagnose_specs(
    specs: &[ProblemSpec],
    mut build: impl FnMut(&ProblemSpec) -> Result<Problem, BenchError>,
) -> Vec<LandscapeRow> {
    specs
        .iter()
        .map(|spec| match build(spec) {
            Ok(problem) => match measure_axes(&problem) {
                (Ok(shapes), evaluations) => completed_row(&problem, &shapes, evaluations),
                (Err(e), evaluations) => {
                    failed_row(spec, evaluations, problem.preparation_evaluations, &e)
                }
            },
            Err(e) => failed_row(spec, 0, 0, &e),
        })
        .collect()
}

/// Every loss evaluation the diagnosis of `rows` executes: each row's
/// `evaluations` plus its `preparation_evaluations`.
pub fn landscape_total_evaluations(rows: &[LandscapeRow]) -> usize {
    rows.iter()
        .map(|r| r.evaluations + r.preparation_evaluations)
        .sum()
}

/// Every (family, dimension, validation seed) problem of `m`, in manifest
/// order.
fn manifest_specs(m: &Manifest) -> Vec<ProblemSpec> {
    let mut specs = Vec::new();
    for &family in &m.families {
        for &dimension in &m.dimensions {
            for &seed in &m.validation_seeds {
                specs.push(ProblemSpec {
                    family,
                    dimension,
                    seed,
                });
            }
        }
    }
    specs
}

/// Diagnose every (family, dimension, validation seed) problem of a synthetic
/// manifest, in manifest order, with [`build_problem`] (see
/// [`diagnose_specs`]). Only an invalid or corpus manifest is an error; a
/// problem that fails is recorded on its row.
pub fn diagnose_manifest(m: &Manifest) -> Result<Vec<LandscapeRow>, BenchError> {
    m.validate()?;
    if m.corpus.is_some() {
        return Err(BenchError::InvalidConfig(
            "the landscape diagnosis needs a synthetic manifest, not a corpus".into(),
        ));
    }
    Ok(diagnose_specs(&manifest_specs(m), build_problem))
}

/// Planned versus diagnosed problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LandscapeCoverage {
    /// Problems the manifest asked for: families x dimensions x validation
    /// seeds.
    pub planned_problems: usize,
    /// Rows diagnosed without error.
    pub completed_problems: usize,
    /// Rows that recorded an error.
    pub failed_problems: usize,
}

/// Coverage of `rows` against what `m` plans.
pub fn landscape_coverage(m: &Manifest, rows: &[LandscapeRow]) -> LandscapeCoverage {
    let failed = rows.iter().filter(|r| r.error.is_some()).count();
    LandscapeCoverage {
        planned_problems: m.families.len() * m.dimensions.len() * m.validation_seeds.len(),
        completed_problems: rows.len() - failed,
        failed_problems: failed,
    }
}

/// Totals of one (family, dimension) cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LandscapeCell {
    /// Problem family.
    pub family: Family,
    /// Free weights.
    pub dimension: usize,
    /// Rows in the cell, failed ones included.
    pub problems: usize,
    /// Rows in the cell that recorded an error. Absent (0) in format-2
    /// summaries, whose runs stopped at the first error.
    #[serde(default)]
    pub failures: usize,
    /// Shape totals over every axis of every completed problem.
    #[serde(flatten)]
    pub shapes: ShapeCounts,
    /// Total diagnosis evaluations (see [`LandscapeRow::evaluations`]).
    pub evaluations: usize,
    /// Total selection-scan evaluations (see
    /// [`LandscapeRow::preparation_evaluations`]).
    pub preparation_evaluations: usize,
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
                failures: 0,
                shapes: ShapeCounts::default(),
                evaluations: 0,
                preparation_evaluations: 0,
                plateau_free_axes: 0,
            };
            for row in rows
                .iter()
                .filter(|r| r.family == family && r.dimension == dimension)
            {
                cell.problems += 1;
                cell.failures += usize::from(row.error.is_some());
                cell.shapes.add(&row.shapes);
                cell.evaluations += row.evaluations;
                cell.preparation_evaluations += row.preparation_evaluations;
                cell.plateau_free_axes += row.plateau_free_axes;
            }
            cells.push(cell);
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failure_partway_through_the_axes_keeps_the_evaluations_spent() {
        let x0 = [0.0, 0.5, 1.0];
        // The base and the first axis succeed; the 7th call (axis 1, second
        // step) fails.
        let mut calls = 0usize;
        let (result, evaluations) = classify_axes(&x0, |x| {
            calls += 1;
            if calls == 7 {
                Err(BenchError::Evaluation("boom".into()))
            } else {
                Ok(1.0 + x.iter().map(|v| v * v).sum::<f64>())
            }
        });
        assert_eq!(result, Err(BenchError::Evaluation("boom".into())));
        assert_eq!(evaluations, 7);
        // Without the failure every axis is classified for 1 + 4 * 3 calls.
        let (result, evaluations) =
            classify_axes(&x0, |x| Ok(1.0 + x.iter().map(|v| v * v).sum::<f64>()));
        assert_eq!(result, Ok(vec![AxisShape::Smooth; 3]));
        assert_eq!(evaluations, 13);
    }
}
