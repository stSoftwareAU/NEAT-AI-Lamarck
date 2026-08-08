//! Shared Lamarck configuration defaults and run options.

use crate::observations::{DEFAULT_QUICK_SAMPLE_RECORDS, StatsMode};
use std::path::PathBuf;
use std::time::Duration;

/// Default wall-clock budget for one Lamarck run, in seconds.
pub const DEFAULT_TIMEOUT_SECONDS: u64 = 45 * 60;

/// Default number of candidate creatures generated per experiment.
pub const DEFAULT_CANDIDATE_COUNT: usize = 50;

/// Default absolute score improvement required for acceptance (strict `>`).
///
/// GRQ `costOfGrowth` is `1e-7`; this threshold is deliberately larger.
pub const DEFAULT_MIN_IMPROVEMENT: f64 = 1e-6;

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
    /// When set, always focus this neuron UUID instead of random selection.
    pub focus_neuron: Option<String>,
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
            stats_mode: StatsMode::Full,
            quick_sample_records: DEFAULT_QUICK_SAMPLE_RECORDS,
            focus_neuron: None,
        }
    }
}
