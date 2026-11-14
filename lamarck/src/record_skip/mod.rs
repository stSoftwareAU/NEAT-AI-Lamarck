//! Opt-in, offline experiment: temporary easy-record skipping during refinement
//! (issue #267).
//!
//! Nothing in the optimisation loop calls this module. It exists so a
//! predeclared, paired benchmark can ask whether skipping records the creature
//! already fits well, for the refinement phase only, saves record passes
//! without costing held-out accuracy. Every arm reuses the real refinement
//! update path (`LearningPlan` / `LearningScan` / `apply_learnings`); the
//! parent creature is never mutated.
//!
//! * [`schedule`] — difficulty tracking and the skip policies (pure).
//! * [`stats`] — paired bootstrap interval and the verdict rule.
//! * [`experiment`] — corpora, the refinement driver and the manifest runner.

pub mod corpus;
pub mod experiment;
pub mod manifest;
pub mod schedule;
pub mod stats;
