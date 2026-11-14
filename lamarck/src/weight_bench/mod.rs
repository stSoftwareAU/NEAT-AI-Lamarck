//! Offline benchmark of local weight optimisers under evaluation budgets.
//!
//! This is an opt-in experiment for issue #266. Nothing in the optimisation
//! loop calls it: it only runs when the `weight_router_bench` example (or a
//! test) drives it, so ordinary behaviour is unchanged.
//!
//! It compares two in-process arms on fixed, deterministic screening records:
//!
//! - **lamarck-refine** — a simplified random-axis control (antithetic pairs
//!   and follow-up fractions), not the production candidate pipeline;
//! - **cma-es** — a full-covariance (mu/mu_w, lambda)-CMA-ES.
//!
//! A budget-aware router and its landscape probes were measured in an earlier
//! run (crate 0.1.46) and removed: the router never beat CMA-ES alone, so it
//! did not earn its maintenance cost.
//!
//! AMM, the method of the paper arXiv 2611.06879, is deliberately NOT
//! reimplemented in Rust. It is run offline as a reference arm, pinned to its
//! authors' implementation (`BjBodner/amm-optimizer` commit
//! `74d39b4f7988a1f1aa8b40a2a0c3ee547f251107`, preset `c46`) via
//! `scripts/amm-reference.py`, as a third **amm-reference** arm. [`reference::serve_reference`] hands it the same
//! budgeted objective over a line protocol, and
//! [`reference::summarise_reference`] compares its rows with the in-process
//! arms using the same paired bootstrap. It only ever runs on validation seeds.
//!
//! Every evaluation is counted against an explicit [`Budget`]; exceeding it is
//! an error ([`BenchError::BudgetExhausted`]), never a silent value.

pub mod arms;
pub mod cmaes;
pub mod experiment;
pub mod objective;
pub mod reference;
pub mod refine;
pub mod stats;
pub mod suite;

pub use arms::{Arm, arm_seed, run_arm};
pub use cmaes::{CMA_SIGMA0, cma_es};
pub use experiment::{
    CellSummary, Comparison, Coverage, ExperimentResults, Manifest, MetricSummary, RunRow,
    predeclared_manifest, run_manifest,
};
pub use objective::{BenchError, Budget, BudgetedObjective, WeightObjective, select_influential};
pub use reference::{
    REFERENCE_BOX_HALF_WIDTH, ReferenceRequest, ReferenceResults, serve_reference,
    summarise_reference,
};
pub use refine::lamarck_refine;
pub use suite::{Family, Problem, ProblemSpec, build_problem, is_extrapolation};
