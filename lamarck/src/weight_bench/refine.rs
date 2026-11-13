//! The current Lamarck local refinement, replayed on a fixed weight set.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::objective::{BenchError, BudgetedObjective};
use crate::candidates::RANDOM_DELTA_HALF_WIDTH;
use crate::followup::STEP_FRACTIONS;

/// Random-axis refinement with antithetic pairs and follow-up probes.
///
/// Each round picks an axis and a delta drawn like the Random strategy, tries
/// the mirrored pair `x +/- delta` (as `mirror.rs` does), and accepts the
/// better one if it strictly improves. After an accept it probes the
/// [`STEP_FRACTIONS`] along the winning move and takes the best strict
/// improvement (one greedy round). The start is evaluated first and counts.
pub fn lamarck_refine(obj: &mut BudgetedObjective<'_>, seed: u64) -> Result<(), BenchError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dim = obj.dimension();
    let mut x = obj.start();
    let mut f = obj.eval(&x)?;
    while obj.remaining() >= 2 {
        let axis = rng.random_range(0..dim);
        let delta = loop {
            let d = rng.random_range(-RANDOM_DELTA_HALF_WIDTH..RANDOM_DELTA_HALF_WIDTH);
            if d != 0.0 {
                break d;
            }
        };
        let mut plus = x.clone();
        plus[axis] += delta;
        let mut minus = x.clone();
        minus[axis] -= delta;
        let f_plus = obj.eval(&plus)?;
        let f_minus = obj.eval(&minus)?;
        let (f_best, signed, winner) = if f_plus <= f_minus {
            (f_plus, delta, plus)
        } else {
            (f_minus, -delta, minus)
        };
        if f_best >= f {
            continue;
        }
        x = winner;
        f = f_best;
        // Follow-up probes along the accepted move, all from the accepted point.
        let mut best_probe: Option<(f64, Vec<f64>)> = None;
        for fraction in STEP_FRACTIONS {
            if obj.remaining() == 0 {
                break;
            }
            let mut probe = x.clone();
            probe[axis] += fraction * signed;
            let fp = obj.eval(&probe)?;
            if fp < f && best_probe.as_ref().is_none_or(|(b, _)| fp < *b) {
                best_probe = Some((fp, probe));
            }
        }
        if let Some((fp, probe)) = best_probe {
            x = probe;
            f = fp;
        }
    }
    Ok(())
}
