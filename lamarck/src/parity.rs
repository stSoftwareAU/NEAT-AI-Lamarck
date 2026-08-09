//! Phase-0 Lamarck ↔ authoritative-scorer parity helpers (issue #4).
//!
//! Before optimisation starts, Lamarck computes a whole-creature MSE over the
//! training corpus with the same compiled network used for analysis, then
//! compares it to the scorer's `error` / unpenalized score. Unexplained
//! disagreement aborts the run.

use neat_core::{
    CompiledNetwork, CreatureExport, TrainingDataConfig, TrainingDataIterator, mse_sum_batch_packed,
};
use std::path::Path;

/// Absolute tolerance on mean squared error (Phase-0).
pub const PHASE0_ERROR_ABS_TOL: f64 = 1e-6;

/// Relative tolerance on mean squared error (Phase-0).
pub const PHASE0_ERROR_REL_TOL: f64 = 1e-5;

/// Absolute tolerance on unpenalized score `1 - error` (Phase-0).
pub const PHASE0_SCORE_ABS_TOL: f64 = 1e-5;

/// Relative tolerance on unpenalized score (Phase-0).
pub const PHASE0_SCORE_REL_TOL: f64 = 1e-4;

/// Records packed per `mse_sum_batch_packed` call (SIMD-friendly chunk).
const PHASE0_MSE_BATCH_RECORDS: usize = 1024;

/// True when `|local - authoritative|` is within abs or relative tolerance.
pub fn values_agree(local: f64, authoritative: f64, abs_tol: f64, rel_tol: f64) -> bool {
    if !local.is_finite() || !authoritative.is_finite() {
        return false;
    }
    let diff = (local - authoritative).abs();
    diff <= abs_tol || diff <= rel_tol * authoritative.abs().max(1.0)
}

/// Whole-creature mean squared error via neat-core packed batch MSE.
///
/// Returns `(mean_mse, record_count)`. Matches the scorer's average-error
/// finalisation (`mse_sum / records`) for forward-only creatures.
pub fn compute_local_mse(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
) -> Result<(f64, u64), String> {
    let config = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = TrainingDataIterator::new(training_data, config).map_err(|e| e.to_string())?;
    let width = creature.input + creature.output;
    let mut packed: Vec<f32> = Vec::with_capacity(PHASE0_MSE_BATCH_RECORDS * width);
    let mut sum = 0.0f64;
    let mut count = 0u64;
    let forward_only = creature.forward_only;

    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if record.inputs.len() < creature.input || record.outputs.len() < creature.output {
            continue;
        }
        packed.extend_from_slice(&record.inputs[..creature.input]);
        packed.extend_from_slice(&record.outputs[..creature.output]);
        if packed.len() / width >= PHASE0_MSE_BATCH_RECORDS {
            let n = packed.len() / width;
            sum += mse_sum_batch_packed(
                network,
                &packed,
                creature.input,
                creature.output,
                forward_only,
            );
            count += n as u64;
            packed.clear();
        }
    }
    if !packed.is_empty() {
        let n = packed.len() / width;
        sum += mse_sum_batch_packed(
            network,
            &packed,
            creature.input,
            creature.output,
            forward_only,
        );
        count += n as u64;
    }
    if count == 0 {
        return Err("Phase-0 local MSE: no training records scored".into());
    }
    Ok((sum / count as f64, count))
}

/// Unpenalized fitness proxy used for Phase-0 score overlap: `1.0 - error`.
///
/// The authoritative scorer subtracts complexity/version penalties; compare
/// this proxy to `scorer.score + scorer.complexity_penalty`.
pub fn local_unpenalized_score(error: f64) -> f64 {
    1.0 - error
}

/// Compare Lamarck local MSE/score proxies to an authoritative [`crate::scorer::ScoreResult`].
///
/// Returns `Ok(())` on agreement, or `Err` with a loud reason.
pub fn check_phase0_parity(
    local_error: f64,
    authoritative_error: f64,
    authoritative_score: f64,
    authoritative_complexity_penalty: f64,
) -> Result<(), String> {
    if !values_agree(
        local_error,
        authoritative_error,
        PHASE0_ERROR_ABS_TOL,
        PHASE0_ERROR_REL_TOL,
    ) {
        return Err(format!(
            "Phase-0 parity failed: local error {local_error:.12} disagrees with scorer error {authoritative_error:.12} (abs_tol={PHASE0_ERROR_ABS_TOL}, rel_tol={PHASE0_ERROR_REL_TOL})"
        ));
    }
    let local_score = local_unpenalized_score(local_error);
    let scorer_unpenalized = authoritative_score + authoritative_complexity_penalty;
    if !values_agree(
        local_score,
        scorer_unpenalized,
        PHASE0_SCORE_ABS_TOL,
        PHASE0_SCORE_REL_TOL,
    ) {
        return Err(format!(
            "Phase-0 parity failed: local unpenalized score {local_score:.12} disagrees with scorer score+complexity {scorer_unpenalized:.12} (score={authoritative_score:.12}, complexity={authoritative_complexity_penalty:.12}; abs_tol={PHASE0_SCORE_ABS_TOL}, rel_tol={PHASE0_SCORE_REL_TOL})"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_agree_within_abs() {
        assert!(values_agree(0.5, 0.5 + 1e-7, 1e-6, 1e-9));
        assert!(!values_agree(0.5, 0.5 + 1e-4, 1e-6, 1e-9));
    }

    #[test]
    fn phase0_parity_passes_on_match() {
        check_phase0_parity(0.25, 0.25, 0.74, 0.01).unwrap();
    }

    #[test]
    fn phase0_parity_aborts_on_error_mismatch() {
        let err = check_phase0_parity(0.25, 0.30, 0.70, 0.0).unwrap_err();
        assert!(err.contains("local error"));
    }

    #[test]
    fn phase0_parity_aborts_on_score_mismatch() {
        // Error agrees, but score+complexity does not match 1-error.
        let err = check_phase0_parity(0.25, 0.25, 0.50, 0.0).unwrap_err();
        assert!(err.contains("unpenalized score"));
    }
}
