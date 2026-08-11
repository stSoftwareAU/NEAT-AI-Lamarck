//! Shared Lamarck configuration defaults and run options.

use crate::backprop::BackpropConfig;
#[cfg(test)]
use crate::backprop::BiasSignal;
use crate::focus::FocusPolicy;
use crate::observations::{DEFAULT_QUICK_SAMPLE_RECORDS, StatsMode};
use std::path::PathBuf;
use std::time::Duration;

/// Default wall-clock budget for one Lamarck run, in seconds.
pub const DEFAULT_TIMEOUT_SECONDS: u64 = 45 * 60;

/// Default number of candidate creatures generated per experiment.
///
/// Tuned for **score improvement per hour** on a ~10-core GRQ box that can stay
/// at full CPU for the run budget. Production creature ≈11s/creature full-corpus
/// directory score, ≈1s/creature at 5% sample. Prefer a large screen batch so the
/// scorer stays saturated; promote cost is bounded by
/// [`DEFAULT_SCREEN_PROMOTE_THRESHOLD`].
pub const DEFAULT_CANDIDATE_COUNT: usize = 100;

/// Default absolute score improvement required for acceptance (strict `>`).
///
/// GRQ `costOfGrowth` is `1e-7`; this threshold is deliberately larger.
pub const DEFAULT_MIN_IMPROVEMENT: f64 = 1e-6;

/// Default scorer subsample rate for the screen phase (issue #24).
///
/// On GRQ-scale directory batches, `0.05` ≈ 0.7–1s/creature vs ≈11s full.
/// Slightly stabler ranking than `0.02` at similar IO-bound cost for small N;
/// use `1.0` to disable screening.
pub const DEFAULT_SCREEN_SAMPLE_RATE: f64 = 0.05;

/// Default minimum sample-score Δ to promote to full-corpus scoring.
///
/// Matched to [`DEFAULT_MIN_IMPROVEMENT`]: only burn ~11s/creature full scores
/// on candidates that already look acceptable on the sample. Sub-threshold
/// sample noise (~1e-7–1e-6) previously dominated promote time without
/// producing accepts — bad for improvement/hour.
pub const DEFAULT_SCREEN_PROMOTE_THRESHOLD: f64 = DEFAULT_MIN_IMPROVEMENT;

/// Abort after this many consecutive scorer failures.
pub const DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES: u32 = 3;

/// Run-time knobs for a Lamarck optimisation session.
#[derive(Debug, Clone)]
pub struct LamarckConfig {
    /// Path to the supplied incumbent creature JSON (never modified in place).
    pub creature: PathBuf,
    /// Training-data directory containing `.bin` files.
    pub training_data: PathBuf,
    /// Wall-clock budget.
    pub timeout: Duration,
    /// Stop after this many experiments. `None` = wall-clock bounded only.
    pub max_experiments: Option<u64>,
    /// Candidates per focus-neuron experiment.
    pub candidates: usize,
    /// Absolute score delta required for acceptance (`candidate - baseline`).
    pub min_improvement: f64,
    /// Optional deterministic RNG seed.
    pub seed: Option<u64>,
    /// Optional path to the `rust_scorer` binary.
    pub scorer_path: PathBuf,
    /// Directory for `best.json`, `experiments.jsonl`, and optional winners.
    pub output_dir: PathBuf,
    /// When true, keep rejected candidate JSON files.
    pub preserve_losers: bool,
    /// Observations cache mode (`full` or `quick` sample).
    pub stats_mode: StatsMode,
    /// Max records for quick-mode observations (ignored in full mode).
    pub quick_sample_records: u64,
    /// When set, always focus this neuron UUID instead of policy selection.
    pub focus_neuron: Option<String>,
    /// Focus selection policy when `focus_neuron` is unset.
    pub focus_policy: FocusPolicy,
    /// Compute expensive input×input correlations in observations.
    pub compute_correlations: bool,
    /// Abort after this many consecutive scorer failures.
    pub max_consecutive_scorer_failures: u32,
    /// Run Phase-0 baseline score gate before optimising.
    pub phase0_parity: bool,
    /// When true, only generate synapse/neuron growth candidates (no weight/bias nudges).
    pub structural_only: bool,
    /// When set in `(0, 1)`, screen the candidate batch on a scorer subsample first
    /// (issue #24). `None` or `Some(1.0)` = full-corpus score only.
    pub screen_sample_rate: Option<f64>,
    /// Minimum sample-score Δ to promote a candidate to full-corpus scoring.
    pub screen_promote_threshold: f64,
    /// Local structural graft store path. `None` disables phase-G replay / recording.
    pub grafts_path: Option<PathBuf>,
    /// Wall-clock budget for phase-G graft replay. `None` → 10% of [`Self::timeout`].
    pub graft_replay_budget: Option<Duration>,
    /// Backprop learning-rate override for candidate proposal (issue #75 A/B).
    /// `None` keeps the [`BackpropConfig::default`] rate.
    pub backprop_learning_rate: Option<f64>,
    /// Override for [`BackpropConfig::maximum_bias_adjustment_scale`], the ±cap
    /// on one `backprop` bias proposal (issue #96 A/B).
    ///
    /// On a creature whose focus carries a saturating blame mass the proposal
    /// hits this cap at every learning rate, so the cap — not the rate — is the
    /// knob that moves the step. `None` keeps the [`BackpropConfig::default`]
    /// cap of `10.0`.
    pub backprop_max_bias_adjustment_scale: Option<f64>,
}

impl LamarckConfig {
    /// Backprop configuration for this run, with the CLI overrides applied.
    ///
    /// An override that is not finite and strictly positive is a configuration
    /// fault, reported as an error rather than silently reverting to the
    /// default — a run that quietly ignored the flag would invalidate the A/B
    /// it was set for.
    pub fn backprop_config(&self) -> Result<BackpropConfig, String> {
        let mut backprop = BackpropConfig::default();
        if let Some(rate) = self.backprop_learning_rate {
            if !rate.is_finite() || rate <= 0.0 {
                return Err(format!(
                    "--backprop-learning-rate must be finite and greater than 0 (got {rate})"
                ));
            }
            backprop.learning_rate = rate;
            backprop.initial_learning_rate = rate;
        }
        if let Some(cap) = self.backprop_max_bias_adjustment_scale {
            if !cap.is_finite() || cap <= 0.0 {
                return Err(format!(
                    "--backprop-max-bias-adjustment-scale must be finite and greater than 0 (got {cap})"
                ));
            }
            backprop.maximum_bias_adjustment_scale = cap;
        }
        Ok(backprop)
    }
}

impl Default for LamarckConfig {
    fn default() -> Self {
        Self {
            creature: PathBuf::from("creature.json"),
            training_data: PathBuf::from("training"),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECONDS),
            max_experiments: None,
            candidates: DEFAULT_CANDIDATE_COUNT,
            min_improvement: DEFAULT_MIN_IMPROVEMENT,
            seed: None,
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: PathBuf::from("."),
            preserve_losers: false,
            // Full observations once/day is fine for GRQ; `--quick` is smoke-only.
            stats_mode: StatsMode::Full,
            quick_sample_records: DEFAULT_QUICK_SAMPLE_RECORDS,
            focus_neuron: None,
            focus_policy: FocusPolicy::Weighted,
            compute_correlations: false,
            max_consecutive_scorer_failures: DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES,
            phase0_parity: true,
            structural_only: false,
            screen_sample_rate: Some(DEFAULT_SCREEN_SAMPLE_RATE),
            screen_promote_threshold: DEFAULT_SCREEN_PROMOTE_THRESHOLD,
            grafts_path: None,
            graft_replay_budget: None,
            backprop_learning_rate: None,
            backprop_max_bias_adjustment_scale: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backprop_config_keeps_the_port_default_rate_when_unset() {
        let config = LamarckConfig::default();
        let backprop = config.backprop_config().expect("default rate is valid");
        assert_eq!(
            backprop.learning_rate,
            BackpropConfig::default().learning_rate
        );
    }

    #[test]
    fn backprop_config_applies_the_override_to_both_rate_fields() {
        let config = LamarckConfig {
            backprop_learning_rate: Some(0.001),
            ..LamarckConfig::default()
        };
        let backprop = config.backprop_config().expect("0.001 is valid");
        assert_eq!(backprop.learning_rate, 0.001);
        // Decay/adaptive schedules start from `initial_learning_rate`, so an
        // override that moved only `learning_rate` would silently be undone.
        assert_eq!(backprop.initial_learning_rate, 0.001);
    }

    #[test]
    fn a_larger_rate_proposes_a_proportionally_larger_bias_step() {
        let signal = BiasSignal {
            count: 1.0,
            total_adjusted_bias: 0.5,
            ..BiasSignal::default()
        };
        let small = LamarckConfig {
            backprop_learning_rate: Some(0.001),
            ..LamarckConfig::default()
        }
        .backprop_config()
        .unwrap();
        let large = LamarckConfig {
            backprop_learning_rate: Some(0.01),
            ..LamarckConfig::default()
        }
        .backprop_config()
        .unwrap();
        let small_step = signal.propose(0.0, &small, small.learning_rate) - 0.0;
        let large_step = signal.propose(0.0, &large, large.learning_rate) - 0.0;
        assert!(
            large_step.abs() > small_step.abs() * 5.0,
            "10x rate should move the bias far further: {small_step} vs {large_step}"
        );
    }

    #[test]
    fn backprop_config_keeps_the_port_default_bias_cap_when_unset() {
        let config = LamarckConfig::default();
        let backprop = config.backprop_config().expect("default cap is valid");
        assert_eq!(
            backprop.maximum_bias_adjustment_scale,
            BackpropConfig::default().maximum_bias_adjustment_scale
        );
    }

    #[test]
    fn backprop_config_applies_the_bias_cap_override() {
        let config = LamarckConfig {
            backprop_max_bias_adjustment_scale: Some(0.001),
            ..LamarckConfig::default()
        };
        let backprop = config.backprop_config().expect("0.001 is valid");
        assert_eq!(backprop.maximum_bias_adjustment_scale, 0.001);
        // The cap override must not disturb the weight cap: the #96 A/B varies
        // the bias step alone, and a moved weight cap would confound it.
        assert_eq!(
            backprop.maximum_weight_adjustment_scale,
            BackpropConfig::default().maximum_weight_adjustment_scale
        );
    }

    #[test]
    fn a_smaller_cap_shrinks_a_blame_saturated_bias_step() {
        // Blame mass far above the cap: the step is cap-bound, not rate-bound
        // (issue #96 item 3 — the knob the #75 A/B actually needed).
        let signal = BiasSignal {
            count: 6.0,
            total_adjusted_bias: 6.0 * 2.3e13,
            ..BiasSignal::default()
        };
        let step_at = |cap: f64| {
            let backprop = LamarckConfig {
                backprop_max_bias_adjustment_scale: Some(cap),
                ..LamarckConfig::default()
            }
            .backprop_config()
            .expect("positive cap is valid");
            (signal.propose(0.0, &backprop, backprop.learning_rate) - 0.0).abs()
        };
        for cap in [10.0, 0.01, 1e-6] {
            let step = step_at(cap);
            assert!(
                (step - cap).abs() < cap * 1e-9,
                "saturating blame should pin the step at the ±{cap} cap, got {step}"
            );
        }
    }

    #[test]
    fn a_non_positive_or_non_finite_cap_is_rejected_loudly() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let config = LamarckConfig {
                backprop_max_bias_adjustment_scale: Some(bad),
                ..LamarckConfig::default()
            };
            let err = config
                .backprop_config()
                .expect_err("invalid cap must not fall back to the default");
            assert!(
                err.contains("--backprop-max-bias-adjustment-scale"),
                "error should name the flag: {err}"
            );
        }
    }

    #[test]
    fn a_non_positive_or_non_finite_rate_is_rejected_loudly() {
        for bad in [0.0, -0.01, f64::NAN, f64::INFINITY] {
            let config = LamarckConfig {
                backprop_learning_rate: Some(bad),
                ..LamarckConfig::default()
            };
            let err = config
                .backprop_config()
                .expect_err("invalid rate must not fall back to the default");
            assert!(
                err.contains("--backprop-learning-rate"),
                "error should name the flag: {err}"
            );
        }
    }
}
