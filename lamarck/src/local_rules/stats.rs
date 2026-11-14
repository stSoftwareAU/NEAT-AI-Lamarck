//! Summary statistics and the paired-difference verdict.
//!
//! Differences are symbolic minus baseline held-out MSE, so a negative
//! difference means the symbolic arm is better.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

/// Arithmetic mean. An empty slice gives `NaN`.
pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Percentile bootstrap 95% confidence interval of the mean, deterministic in `seed`.
///
/// Returns `(NaN, NaN)` when there are no values or no resamples.
pub fn bootstrap_mean_ci(values: &[f64], resamples: usize, seed: u64) -> (f64, f64) {
    if values.is_empty() || resamples == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let n = values.len();
    let mut means: Vec<f64> = (0..resamples)
        .map(|_| {
            let total: f64 = (0..n).map(|_| values[rng.random_range(0..n)]).sum();
            total / n as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    let at = |p: f64| means[((means.len() - 1) as f64 * p).round() as usize];
    (at(0.025), at(0.975))
}

/// Verdict on a paired difference from its confidence interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// The whole interval is below zero: the symbolic arm is better.
    Better,
    /// The whole interval is above zero: the symbolic arm is worse.
    Worse,
    /// The interval straddles zero.
    Inconclusive,
}

/// `Better` iff the upper bound is below 0, `Worse` iff the lower bound is above 0.
pub fn verdict(ci: (f64, f64)) -> Verdict {
    if ci.1 < 0.0 {
        Verdict::Better
    } else if ci.0 > 0.0 {
        Verdict::Worse
    } else {
        Verdict::Inconclusive
    }
}
