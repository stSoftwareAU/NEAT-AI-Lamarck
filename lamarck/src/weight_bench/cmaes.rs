//! Full-covariance (mu/mu_w, lambda)-CMA-ES with default strategy parameters
//! from Hansen's tutorial (arXiv:1604.00772).

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::objective::{BenchError, BudgetedObjective};

/// Initial step size. It matches the scale of the refinement arm's step
/// ([`crate::candidates::RANDOM_DELTA_HALF_WIDTH`]) so neither arm starts with
/// a larger search radius by construction.
pub const CMA_SIGMA0: f64 = 0.05;

/// One standard-normal draw by Box-Muller.
fn normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = 1.0 - rng.random::<f64>(); // in (0, 1]
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Eigendecomposition of the symmetric `n x n` row-major matrix `c` by the
/// cyclic Jacobi method. Returns `(eigenvalues, eigenvectors)` where the
/// eigenvector for `values[j]` is column `j` of the row-major matrix.
fn jacobi_eigen(c: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = c.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j])
            .sum();
        let diag: f64 = (0..n).map(|i| a[i * n + i] * a[i * n + i]).sum();
        if off <= 1e-26 * diag.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let cs = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * cs;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = cs * akp - sn * akq;
                    a[k * n + q] = sn * akp + cs * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = cs * apk - sn * aqk;
                    a[q * n + k] = sn * apk + cs * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = cs * vkp - sn * vkq;
                    v[k * n + q] = sn * vkp + cs * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// Strategy parameters (Hansen, Table 1).
struct Params {
    lambda: usize,
    mu: usize,
    weights: Vec<f64>,
    mueff: f64,
    cc: f64,
    cs: f64,
    c1: f64,
    cmu: f64,
    damps: f64,
    chi_n: f64,
}

impl Params {
    fn new(n: usize) -> Self {
        let nf = n as f64;
        let lambda = 4 + (3.0 * nf.ln()).floor() as usize;
        let mu = lambda / 2;
        let raw: Vec<f64> = (1..=mu)
            .map(|i| ((mu as f64) + 0.5).ln() - (i as f64).ln())
            .collect();
        let total: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
        let mueff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
        let cs = (mueff + 2.0) / (nf + mueff + 5.0);
        let c1 = 2.0 / ((nf + 1.3).powi(2) + mueff);
        let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((nf + 2.0).powi(2) + mueff));
        let damps = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));
        Self {
            lambda,
            mu,
            weights,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            chi_n,
        }
    }
}

/// `m * v` for a row-major `n x n` matrix.
fn mat_vec(m: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| m[i * n + j] * v[j]).sum())
        .collect()
}

/// Run CMA-ES from the objective's start until the budget is spent.
///
/// The start is evaluated once first. A generation only runs when `lambda`
/// evaluations remain; any smaller remainder is spent on plain samples around
/// the current mean (the objective tracks the best) without updating state.
pub fn cma_es(obj: &mut BudgetedObjective<'_>, seed: u64, sigma0: f64) -> Result<(), BenchError> {
    if !(sigma0.is_finite() && sigma0 > 0.0) {
        return Err(BenchError::InvalidConfig(format!(
            "sigma0 must be positive and finite, got {sigma0}"
        )));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let n = obj.dimension();
    let p = Params::new(n);
    let mut mean = obj.start();
    obj.eval(&mean)?;
    let mut sigma = sigma0;
    let mut cov = vec![0.0; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0;
    }
    let (mut pc, mut ps) = (vec![0.0; n], vec![0.0; n]);
    let mut generation = 0usize;

    while obj.remaining() >= p.lambda {
        let (values, b) = jacobi_eigen(&cov, n);
        let d: Vec<f64> = values.iter().map(|v| v.max(1e-30).sqrt()).collect();
        let mut sampled: Vec<(f64, Vec<f64>)> = Vec::with_capacity(p.lambda);
        for _ in 0..p.lambda {
            let z: Vec<f64> = (0..n).map(|j| d[j] * normal(&mut rng)).collect();
            let y = mat_vec(&b, &z, n);
            let x: Vec<f64> = (0..n).map(|i| mean[i] + sigma * y[i]).collect();
            sampled.push((obj.eval(&x)?, y));
        }
        sampled.sort_by(|a, b| a.0.total_cmp(&b.0));
        let top = &sampled[..p.mu];

        let mut y_w = vec![0.0; n];
        for (w, (_, y)) in p.weights.iter().zip(top) {
            for i in 0..n {
                y_w[i] += w * y[i];
            }
        }
        for i in 0..n {
            mean[i] += sigma * y_w[i];
        }

        // C^(-1/2) y_w = B D^-1 B^T y_w
        let bt_y: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|i| b[i * n + j] * y_w[i]).sum::<f64>() / d[j])
            .collect();
        let inv_sqrt_y = mat_vec(&b, &bt_y, n);
        let ps_gain = (p.cs * (2.0 - p.cs) * p.mueff).sqrt();
        for i in 0..n {
            ps[i] = (1.0 - p.cs) * ps[i] + ps_gain * inv_sqrt_y[i];
        }
        let ps_norm = ps.iter().map(|v| v * v).sum::<f64>().sqrt();
        generation += 1;
        let decay = (1.0 - (1.0 - p.cs).powi(2 * generation as i32)).sqrt();
        let h_sigma = ps_norm / decay / p.chi_n < 1.4 + 2.0 / (n as f64 + 1.0);
        let h = if h_sigma { 1.0 } else { 0.0 };
        let pc_gain = h * (p.cc * (2.0 - p.cc) * p.mueff).sqrt();
        for i in 0..n {
            pc[i] = (1.0 - p.cc) * pc[i] + pc_gain * y_w[i];
        }

        let stall = (1.0 - h) * p.cc * (2.0 - p.cc);
        let keep = 1.0 - p.c1 - p.cmu;
        for i in 0..n {
            for j in 0..n {
                let mut rank_mu = 0.0;
                for (w, (_, y)) in p.weights.iter().zip(top) {
                    rank_mu += w * y[i] * y[j];
                }
                cov[i * n + j] = keep * cov[i * n + j]
                    + p.c1 * (pc[i] * pc[j] + stall * cov[i * n + j])
                    + p.cmu * rank_mu;
            }
        }
        sigma *= (p.cs / p.damps * (ps_norm / p.chi_n - 1.0)).exp();
        if !sigma.is_finite() || sigma <= 0.0 || mean.iter().any(|v| !v.is_finite()) {
            return Err(BenchError::Evaluation("CMA-ES state diverged".into()));
        }
    }

    // Fewer than lambda evaluations remain: spend them sampling, no update.
    let (values, b) = jacobi_eigen(&cov, n);
    let d: Vec<f64> = values.iter().map(|v| v.max(1e-30).sqrt()).collect();
    while obj.remaining() > 0 {
        let z: Vec<f64> = (0..n).map(|j| d[j] * normal(&mut rng)).collect();
        let y = mat_vec(&b, &z, n);
        let x: Vec<f64> = (0..n).map(|i| mean[i] + sigma * y[i]).collect();
        obj.eval(&x)?;
    }
    Ok(())
}
