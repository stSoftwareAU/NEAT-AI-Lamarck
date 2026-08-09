//! NEAT-AI-Lamarck experimental optimiser library.
//!
//! Behaviour is introduced through independently tested modules. See the
//! repository README for the experiment architecture and locked contracts.

#![warn(missing_docs)]

pub mod backprop;
pub mod candidates;
pub mod config;
pub mod focus;
pub mod learning;
pub mod log;
pub mod observations;
pub mod report;
pub mod run;
pub mod scorer;
pub mod structural;

pub use config::{
    DEFAULT_CANDIDATE_COUNT, DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES, DEFAULT_MIN_IMPROVEMENT,
    DEFAULT_SCREEN_PROMOTE_THRESHOLD, DEFAULT_SCREEN_SAMPLE_RATE, DEFAULT_TIMEOUT_SECONDS,
    LamarckConfig,
};
pub use focus::{FocusChoice, FocusPolicy, WeightedFocusSelector};
pub use report::{JournalReport, print_run_summary, report_from_journal};
pub use run::{ExperimentRecord, RunResult, run_optimisation};
pub use scorer::{
    ExternalScorer, ScoreResult, ScoreSample, accepts_improvement, log_scorer_batch_stats,
    screen_promote_stems, write_promote_batch,
};
