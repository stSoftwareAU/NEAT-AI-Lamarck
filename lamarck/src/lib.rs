//! NEAT-AI-Lamarck experimental optimiser library.
//!
//! Behaviour is intentionally introduced through small, independently tested
//! issues. See the repository README for the experiment architecture.

/// Default wall-clock budget for one Lamarck run, in seconds.
pub const DEFAULT_TIMEOUT_SECONDS: u64 = 45 * 60;

/// Default number of candidate creatures generated per experiment.
pub const DEFAULT_CANDIDATE_COUNT: usize = 50;
