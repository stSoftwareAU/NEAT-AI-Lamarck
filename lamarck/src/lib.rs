//! NEAT-AI-Lamarck experimental optimiser library.
//!
//! Behaviour is introduced through independently tested modules. See the
//! repository README for the experiment architecture and locked contracts.

#![warn(missing_docs)]

pub mod analysis;
pub mod backprop;
pub mod baseline;
pub mod cancel;
pub mod candidates;
pub mod chunks;
pub mod combos;
pub mod config;
pub mod failed_cache;
pub mod focus;
pub mod followup;
pub mod grafts;
pub mod learning;
pub mod log;
pub mod memetic;
pub mod memo;
pub mod mirror;
pub mod observations;
pub mod parity;
pub mod promote_gate;
pub mod propagate_layout;
pub mod report;
pub mod run;
pub mod scorer;
pub mod scorer_cost;
pub mod screen_calibration;
pub mod structural;
pub mod tags;
pub mod validate;
pub mod width;

pub use analysis::{
    PostFocusScan, PreFocusScan, reset_training_scan_count, scan_post_focus, scan_pre_focus,
    training_scans_opened,
};
pub use backprop::{BackpropConfig, LearningSignal, apply_learnings};
pub use baseline::{
    BaselineKey, BaselineReusePolicy, BaselineSource, DEFAULT_BASELINE_DRIFT_EPSILON,
    MAX_AUTO_BASELINE_DRIFT_EPSILON, RememberedBaseline, estimate_corpus_records,
    resolve_baseline_drift_epsilon, training_data_key, tuned_baseline_drift_epsilon,
};
pub use cancel::CancelToken;
pub use chunks::{ANALYSIS_CHUNK_RECORDS, DEFAULT_ANALYSIS_THREADS};
pub use combos::{
    ComboSelectRequest, ComboSelection, Improver, MAX_COMBO_CANDIDATES, STACK_DAMPEN_EXPONENT,
    StackDampenReport, StackDampenTarget, collect_improvers, collect_improvers_against,
    combination_index_sets, dampen_stacked_new_synapses, merge_candidate_deltas,
    new_synapse_contributor_counts, select_best_with_combinations, stack_dampen_scale,
};
pub use config::{
    DEFAULT_CANDIDATE_COUNT, DEFAULT_FOCUS_COUNT, DEFAULT_FOLLOWUP_CANDIDATES,
    DEFAULT_FOLLOWUP_EXPERIMENTS, DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES, DEFAULT_MIN_IMPROVEMENT,
    DEFAULT_SCREEN_PROMOTE_THRESHOLD, DEFAULT_SCREEN_SAMPLE_RATE, DEFAULT_TIMEOUT_SECONDS,
    LamarckConfig,
};
pub use failed_cache::{
    CacheEconomics, CacheEconomicsConfig, CacheHit, CandidateFingerprint, CeilingBite,
    DEFAULT_CACHE_MAX_RESIDENT_BYTES, DEFAULT_CACHE_STAND_DOWN_MARGIN_MS,
    DEFAULT_CACHE_STAND_DOWN_WINDOW, DEFAULT_FAILED_CACHE_MAX_AGE_SECONDS,
    DEFAULT_FAILED_CACHE_MAX_ENTRIES, DEFAULT_FAILED_CACHE_TOLERANCE_ABS,
    DEFAULT_FAILED_CACHE_TOLERANCE_REL, EconomicsSummary, ExperimentCost, ExperimentEconomics,
    FailedCacheStats, FailedCandidateCache, FilteredBatch, FingerprintBucketKey, RebuildReport,
    RebuildSource, StandDown, Tolerance, filter_and_backfill, insert_failures, load_or_rebuild,
    rebuild_from_journal, scored_candidate_indices, write_snapshot,
};
pub use focus::{FocusChoice, FocusPolicy, WeightedFocusSelector};
pub use followup::{
    FollowUpBudget, FollowUpBurst, FollowUpLink, FollowUpParent, FollowUpPlan, FollowUpProbe,
    ProbeTarget, plan_probes, probe_candidate,
};
pub use grafts::{
    Graft, GraftKind, GraftReplayError, GraftReplayRequest, GraftStore, MAX_GRAFT_COMBO_CANDIDATES,
    classify_graft, extract_structural_graft, is_present, replay_grafts,
};
pub use memetic::{assert_memetic_resolves, prune_memetic};
pub use memo::{
    AnalysisMemo, DEFAULT_ANALYSIS_MEMO_ENTRIES, MemoScope, MemoStats, creature_fingerprint,
};
pub use mirror::{
    MirrorPair, MirrorPolicy, MirrorRole, MirrorStats, PairOutcome, PerturbedScalar,
    SignedPerturbation, axis_failures, mirror_candidate, pair_outcomes, signed_perturbation,
};
pub use parity::{
    PHASE0_ERROR_ABS_TOL, PHASE0_ERROR_REL_TOL, PHASE0_SCORE_ABS_TOL, PHASE0_SCORE_REL_TOL,
    check_phase0_parity, compute_local_mse,
};
pub use promote_gate::{
    DEFAULT_SCREEN_PROMOTE_SIGMA_K, GateThreshold, PromoteGate, PromoteGateMode, PromoteGateReplay,
    PromoteGateReplayAccumulator, ReplayedAccept, estimate_screen_sigma,
};
pub use propagate_layout::{PropagateLayout, accumulate_creature_learning};
pub use report::{
    FocusStatsAggregate, FocusStatsSummary, JournalReport, print_run_summary, report_from_journal,
};
pub use run::{
    ExperimentRecord, JournalLine, RunConfigRecord, RunHeaderKind, RunHeaderRecord, RunResult,
    SeedSource, StopReason, run_optimisation, run_optimisation_cancellable,
};
pub use scorer::{
    ExternalScorer, PromoteDecision, RecordingScorer, ScoreResult, ScoreSample,
    accepts_improvement, accepts_improvement_delta, log_scorer_batch_stats,
    screen_promote_decision, screen_promote_decision_against, screen_promote_stems, select_winner,
    select_winner_against, write_promote_batch, write_promote_batch_without_baseline,
};
pub use scorer_cost::{
    CallCostFit, ScorerCallCost, ScorerCallCostAccumulator, ScorerCallPhase, ScorerCallRecord,
    fit_calls,
};
pub use screen_calibration::{
    AcceptedScreenPoint, BaselineSampleGap, DeltaDistribution, ScreenCalibration,
    ScreenCalibrationAccumulator, ScreenNoise, ScreenPair, spearman_rank_correlation,
};
pub use tags::{
    CreatureMeta, CreatureTag, LamarckProgress, NeuronOrigin, lamarck_neuron_origin_message,
    serialize_creature_with_meta, serialize_creature_with_meta_compact,
};
pub use width::{
    assert_same_width, assert_written_width, checked_creature_json, checked_creature_json_pretty,
    load_creature, validate_creature_width, written_width,
};
