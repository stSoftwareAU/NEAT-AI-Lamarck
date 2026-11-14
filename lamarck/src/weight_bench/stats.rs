//! Small statistics helpers: mean, median and a percentile bootstrap.

use rand::RngExt;
use rand::rngs::StdRng;

/// Arithmetic mean; `None` for an empty slice.
pub fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// Median; `None` for an empty slice.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    Some(if !sorted.len().is_multiple_of(2) {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    })
}

/// 95% percentile-bootstrap confidence interval of the mean; `None` for an
/// empty slice or zero resamples. Deterministic given `rng`.
pub fn bootstrap_ci_mean(values: &[f64], resamples: usize, rng: &mut StdRng) -> Option<(f64, f64)> {
    if values.is_empty() || resamples == 0 {
        return None;
    }
    let n = values.len();
    let mut means: Vec<f64> = (0..resamples)
        .map(|_| (0..n).map(|_| values[rng.random_range(0..n)]).sum::<f64>() / n as f64)
        .collect();
    means.sort_by(f64::total_cmp);
    let at = |q: f64| means[((q * (resamples - 1) as f64).round() as usize).min(resamples - 1)];
    Some((at(0.025), at(0.975)))
}
