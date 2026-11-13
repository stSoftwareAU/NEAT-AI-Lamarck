//! Offline benchmark of local weight optimisers under evaluation budgets.
//!
//! This is an opt-in experiment for issue #266. Nothing in the optimisation
//! loop calls it: it only runs when the `weight_router_bench` example (or a
//! test) drives it, so ordinary behaviour is unchanged.
//!
//! It compares three arms on fixed, deterministic screening records:
//!
//! - **lamarck-refine** — the current local refinement (random axis deltas,
//!   antithetic pairs, follow-up probes) replayed on a fixed weight set;
//! - **cma-es** — a full-covariance (mu/mu_w, lambda)-CMA-ES;
//! - **router** — a budget-aware rule that spends a few evaluations probing
//!   the landscape, then hands the remaining budget to one of the two arms.
//!
//! AMM, the method of the paper arXiv 2611.06879, is deliberately NOT
//! implemented: its method could not be retrieved and pinned, so comparing
//! against a guess would be misleading.
//!
//! Every evaluation is counted against an explicit [`Budget`]; exceeding it is
//! an error ([`BenchError::BudgetExhausted`]), never a silent value.

pub mod cmaes;
pub mod experiment;
pub mod objective;
pub mod probes;
pub mod refine;
pub mod router;
pub mod stats;
pub mod suite;

pub use cmaes::{CMA_SIGMA0, cma_es};
pub use experiment::{
    CellSummary, Comparison, Coverage, ExperimentResults, Manifest, MetricSummary, RunRow,
    predeclared_manifest, run_manifest,
};
pub use objective::{BenchError, Budget, BudgetedObjective, WeightObjective, select_influential};
pub use probes::{LandscapeProbe, probe_landscape};
pub use refine::lamarck_refine;
pub use router::{
    Arm, CALIBRATION_GRID_MULTIPLES, CALIBRATION_GRID_NON_SMOOTH, CalibrationRow,
    ROUTER_PROBE_AXES, RouterRule, arm_seed, calibrate, route, run_arm, run_router,
};
pub use suite::{Family, Problem, ProblemSpec, build_problem, is_extrapolation};
