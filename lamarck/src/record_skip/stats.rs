//! Paired-difference statistics for the record-skip experiment.

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;

/// Mean of `xs` via the shared `weight_bench::stats::mean`; NaN when empty.
pub(super) fn mean(xs: &[f64]) -> f64 {
    crate::weight_bench::stats::mean(xs).unwrap_or(f64::NAN)
}

/// 95% percentile bootstrap interval of the mean of `diffs`.
///
/// Seeds the shared `weight_bench::stats::bootstrap_ci_mean`, so it is
/// deterministic for a given `seed`. Returns `(NaN, NaN)` for empty input or
/// zero resamples.
pub(super) fn bootstrap_mean_ci(diffs: &[f64], resamples: usize, seed: u64) -> (f64, f64) {
    crate::weight_bench::stats::bootstrap_ci_mean(
        diffs,
        resamples,
        &mut StdRng::seed_from_u64(seed),
    )
    .unwrap_or((f64::NAN, f64::NAN))
}

/// Outcome of a paired comparison (lower error is better).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    /// Treatment is credibly better with no rare-stratum regression.
    Better,
    /// Treatment is credibly worse.
    Worse,
    /// Neither.
    Inconclusive,
}

/// Verdict on a paired comparison, given the 95% CI of its treatment-minus-control
/// error differences and the rare-stratum validation differences used as a
/// regression guard.
pub fn verdict(ci: (f64, f64), guard_diffs: &[f64]) -> Verdict {
    if ci.1 < 0.0 && !guard_diffs.is_empty() && mean(guard_diffs) <= 0.0 {
        Verdict::Better
    } else if ci.0 > 0.0 {
        Verdict::Worse
    } else {
        Verdict::Inconclusive
    }
}
