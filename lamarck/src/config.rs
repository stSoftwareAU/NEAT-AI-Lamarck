//! Shared Lamarck configuration defaults and run options.

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
}

impl Default for LamarckConfig {
    fn default() -> Self {
        Self {
            creature: PathBuf::from("creature.json"),
            training_data: PathBuf::from("training"),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECONDS),
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
        }
    }
}
