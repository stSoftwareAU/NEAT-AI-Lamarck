//! NEAT-AI-Lamarck experimental optimiser library.
//!
//! Behaviour is introduced through independently tested modules. See the
//! repository README for the experiment architecture and locked contracts.

#![warn(missing_docs)]

pub mod backprop;
pub mod candidates;
pub mod combos;
pub mod config;
pub mod focus;
pub mod grafts;
pub mod learning;
pub mod log;
pub mod observations;
pub mod parity;
pub mod propagate_layout;
pub mod report;
pub mod run;
pub mod scorer;
pub mod structural;
pub mod tags;

pub use backprop::{BackpropConfig, LearningSignal, apply_learnings};
pub use combos::{
    ComboSelectRequest, ComboSelection, Improver, MAX_COMBO_CANDIDATES, collect_improvers,
    combination_index_sets, merge_candidate_deltas, select_best_with_combinations,
};
pub use config::{
    DEFAULT_CANDIDATE_COUNT, DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES, DEFAULT_MIN_IMPROVEMENT,
    DEFAULT_SCREEN_PROMOTE_THRESHOLD, DEFAULT_SCREEN_SAMPLE_RATE, DEFAULT_TIMEOUT_SECONDS,
    LamarckConfig,
};
pub use focus::{FocusChoice, FocusPolicy, WeightedFocusSelector};
pub use grafts::{
    Graft, GraftKind, GraftReplayError, GraftReplayRequest, GraftStore, MAX_GRAFT_COMBO_CANDIDATES,
    classify_graft, extract_structural_graft, is_present, replay_grafts,
};
pub use parity::{
    PHASE0_ERROR_ABS_TOL, PHASE0_ERROR_REL_TOL, PHASE0_SCORE_ABS_TOL, PHASE0_SCORE_REL_TOL,
    check_phase0_parity, compute_local_mse,
};
pub use propagate_layout::{PropagateLayout, accumulate_creature_learning};
pub use report::{JournalReport, print_run_summary, report_from_journal};
pub use run::{ExperimentRecord, RunResult, run_optimisation};
pub use scorer::{
    ExternalScorer, ScoreResult, ScoreSample, accepts_improvement, log_scorer_batch_stats,
    screen_promote_stems, write_promote_batch,
};
pub use tags::{CreatureMeta, CreatureTag, LamarckProgress, serialize_creature_with_meta};
