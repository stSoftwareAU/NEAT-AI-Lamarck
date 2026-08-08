//! NEAT-AI-Lamarck experimental optimiser library.
//!
//! Behaviour is introduced through independently tested modules. See the
//! repository README for the experiment architecture and locked contracts.

#![warn(missing_docs)]

pub mod backprop;
pub mod candidates;
pub mod config;
pub mod focus;
pub mod observations;
pub mod report;
pub mod run;
pub mod scorer;

pub use config::{
    DEFAULT_CANDIDATE_COUNT, DEFAULT_MIN_IMPROVEMENT, DEFAULT_TIMEOUT_SECONDS, LamarckConfig,
};
pub use report::{JournalReport, report_from_journal};
pub use run::{ExperimentRecord, RunResult, run_optimisation};
pub use scorer::{ExternalScorer, ScoreResult, accepts_improvement};
