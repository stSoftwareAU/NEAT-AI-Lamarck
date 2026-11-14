//! Paired-difference statistics for the record-skip experiment.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::Serialize;

/// Mean of `xs`; NaN when empty.
pub(super) fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// 95% percentile bootstrap interval of the mean of `diffs`.
///
/// Deterministic for a given `seed`. Returns `(NaN, NaN)` for empty input.
pub(super) fn bootstrap_mean_ci(diffs: &[f64], resamples: usize, seed: u64) -> (f64, f64) {
    if diffs.is_empty() || resamples == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let n = diffs.len();
    let mut means: Vec<f64> = (0..resamples)
        .map(|_| {
            let mut sum = 0.0;
            for _ in 0..n {
                sum += diffs[rng.random_range(0..n)];
            }
            sum / n as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    let lo = ((resamples as f64) * 0.025).floor() as usize;
    let hi = (((resamples as f64) * 0.975).ceil() as usize).clamp(1, resamples) - 1;
    (means[lo.min(resamples - 1)], means[hi])
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
