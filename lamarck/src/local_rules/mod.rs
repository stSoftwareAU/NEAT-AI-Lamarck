//! Opt-in offline research: do sparse symbolic local update rules beat, at an
//! equal record budget, Lamarck's current refinement?
//!
//! Issue #268. A small rule language (activation family, local features, sparse
//! linear rules) is discovered on calibration tasks only, then scored on
//! untouched validation tasks against sparse backprop and a derivative-free
//! search on the crate's mirrored signed-perturbation path (`crate::mirror`).
//! Parents are never mutated: every arm clones its parent.
//!
//! This module is research only. It is never wired into `run_optimisation`, and
//! nothing in the crate calls it, so default behaviour is unchanged.

mod apply;
mod baselines;
mod experiment;
mod files;
mod manifest;
mod rule;
mod search;
mod stats;
mod tasks;

pub use apply::{Cost, apply_rule_steps, mse};
pub use baselines::{derivative_free, matched_count, sparse_backprop};
pub use experiment::{
    Comparison, Coverage, DEFAULT_OUT_DIR, DataSource, Failure, OverallVerdict, PairedDifference,
    Report, RunRow, SuccessClaim, predeclared_verdict, resolve_out_dir, run_manifest,
    success_claim, write_outputs,
};
pub use files::run_files;
pub use manifest::{Manifest, NEAT_CORE_TAG, TaskGroup, Versions};
pub use rule::{ActivationFamily, Feature, LocalObservation, RuleSet, SparseRule, local_residual};
pub use search::SearchStep;
pub use stats::{Verdict, paired_verdict, verdict};
pub use tasks::{TASK_GENERATOR_VERSION, Task, TaskRecord, TaskSpec, build_task};
