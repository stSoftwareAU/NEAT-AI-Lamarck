//! The predeclared protocol for the local-rules experiment.
//!
//! Every number that decides the verdict lives here, so it is fixed before any
//! validation run. Calibration seeds drive rule discovery; validation seeds are
//! disjoint and are only ever measured.

use serde::{Deserialize, Serialize};

use super::tasks::TASK_GENERATOR_VERSION;

/// neat-core tag this crate is built against. Must match `lamarck/Cargo.toml`
/// (asserted by a test).
pub const NEAT_CORE_TAG: &str = "v0.23.4";

/// Inclusive bounds on the step count of any arm.
const STEP_RANGE: std::ops::RangeInclusive<u32> = 5..=20;

/// Versions recorded in every report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Versions {
    /// Crate version of the harness.
    pub crate_version: String,
    /// neat-core tag the crate was built against.
    pub neat_core_tag: String,
    /// Synthetic task generator version.
    pub task_generator_version: String,
}

impl Versions {
    /// Versions of the running build.
    pub fn current() -> Self {
        Self {
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            neat_core_tag: NEAT_CORE_TAG.to_string(),
            task_generator_version: TASK_GENERATOR_VERSION.to_string(),
        }
    }
}

/// A named group of synthetic tasks sharing one hidden width.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskGroup {
    /// Group label: `same-width` or `new-width`.
    pub group: String,
    /// Task seeds in this group.
    pub seeds: Vec<u64>,
    /// Hidden width of every task in this group.
    pub hidden_width: usize,
}

/// The full predeclared protocol.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    /// Input count of every task.
    pub inputs: usize,
    /// Training records per task (the only records update arms may see).
    pub train_records: usize,
    /// Held-out records per task (measurement only).
    pub held_out_records: usize,
    /// Hidden squash names, one task set per name.
    pub hidden_squashes: Vec<String>,
    /// Seeds for calibration tasks (rule discovery only).
    pub calibration_seeds: Vec<u64>,
    /// Hidden width of calibration tasks.
    pub calibration_width: usize,
    /// Untouched validation groups.
    pub validation_groups: Vec<TaskGroup>,
    /// Step counts measured for every validation task.
    pub steps: Vec<u32>,
    /// Steps used while scoring candidate rules during discovery.
    pub calibration_steps: u32,
    /// Learning rate applied to the mean symbolic update.
    pub eta: f64,
    /// Clamp on the mean symbolic update per step.
    pub max_step: f64,
    /// Maximum terms in one sparse rule.
    pub max_support: usize,
    /// Discovery rounds.
    pub meta_rounds: u32,
    /// Candidate rules sampled per family per round.
    pub candidates_per_round: usize,
    /// Coefficients a candidate may take.
    pub coefficient_grid: Vec<f64>,
    /// Learning rate of the sparse-backprop baseline (Lamarck's `BackpropConfig` default).
    pub backprop_learning_rate: f64,
    /// Fraction of neurons selected by the sparse-backprop baseline.
    pub sparse_ratio: f64,
    /// Gaussian standard deviation of the derivative-free baseline.
    pub derivative_free_sigma: f64,
    /// Resamples for the bootstrap confidence interval.
    pub bootstrap_resamples: usize,
    /// Seed for the bootstrap.
    pub bootstrap_seed: u64,
    /// Seed for rule discovery.
    pub meta_seed: u64,
    /// Fewest paired tasks a comparison needs before its interval can give a
    /// verdict. Fewer tasks, in a synthetic scope or a files run, are refused or
    /// reported `Inconclusive`: a single difference has no uncertainty.
    pub min_paired_tasks: usize,
    /// The predeclared success rule, applied in code as `overallVerdict`.
    pub success_criterion: String,
}

impl Manifest {
    /// The predeclared protocol for issue #268.
    pub fn predeclared() -> Self {
        let validation = |group: &str, seeds: std::ops::RangeInclusive<u64>, width| TaskGroup {
            group: group.to_string(),
            seeds: seeds.collect(),
            hidden_width: width,
        };
        Self {
            inputs: 3,
            train_records: 64,
            held_out_records: 64,
            hidden_squashes: vec!["TANH".to_string(), "LOGISTIC".to_string()],
            calibration_seeds: (1..=4).collect(),
            calibration_width: 4,
            validation_groups: vec![
                validation("same-width", 101..=104, 4),
                validation("new-width", 201..=204, 2),
                validation("new-width", 301..=304, 8),
            ],
            steps: vec![5, 10, 20],
            calibration_steps: 10,
            eta: 0.05,
            max_step: 1.0,
            max_support: 3,
            meta_rounds: 3,
            candidates_per_round: 16,
            coefficient_grid: vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
            backprop_learning_rate: 0.01,
            sparse_ratio: 0.5,
            derivative_free_sigma: 0.05,
            bootstrap_resamples: 2000,
            bootstrap_seed: 7,
            meta_seed: 11,
            min_paired_tasks: 8,
            success_criterion: "earns-its-place iff the symbolic arm is Better than BOTH primary \
                baselines (sparse-backprop, derivative-free) at at least one K and Worse than \
                neither at any K; negative iff Worse than sparse-backprop at every K; otherwise \
                inconclusive. A comparison is Inconclusive when it has fewer than \
                min_paired_tasks paired tasks, or when any run behind it failed (a failed run \
                is never dropped silently and never counts as a win). A files run with fewer \
                than min_paired_tasks validation creatures is refused."
                .to_string(),
        }
    }

    /// Refuse a protocol that is outside the predeclared bounds.
    pub fn validate(&self) -> Result<(), String> {
        if self.steps.is_empty() {
            return Err("steps must not be empty".to_string());
        }
        for &k in self
            .steps
            .iter()
            .chain(std::iter::once(&self.calibration_steps))
        {
            if !STEP_RANGE.contains(&k) {
                return Err(format!("step count {k} outside 5..=20"));
            }
        }
        for group in &self.validation_groups {
            for seed in &group.seeds {
                if self.calibration_seeds.contains(seed) {
                    return Err(format!(
                        "validation seed {seed} overlaps a calibration seed"
                    ));
                }
            }
        }
        if self.train_records == 0 {
            return Err("train_records must be positive".to_string());
        }
        if self.held_out_records == 0 {
            return Err("held_out_records must be positive".to_string());
        }
        for (name, value) in [
            ("eta", self.eta),
            ("derivative_free_sigma", self.derivative_free_sigma),
            ("backprop_learning_rate", self.backprop_learning_rate),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        if !(self.sparse_ratio > 0.0 && self.sparse_ratio <= 1.0) {
            return Err("sparse_ratio must be in (0, 1]".to_string());
        }
        if self.min_paired_tasks < 2 {
            return Err("min_paired_tasks must be at least 2".to_string());
        }
        let tasks = self.hidden_squashes.len()
            * self
                .validation_groups
                .iter()
                .map(|group| group.seeds.len())
                .sum::<usize>();
        if tasks < self.min_paired_tasks {
            return Err(format!(
                "{tasks} validation tasks, below min_paired_tasks {}",
                self.min_paired_tasks
            ));
        }
        if self.max_support == 0 {
            return Err("max_support must be at least 1".to_string());
        }
        if self.coefficient_grid.is_empty() {
            return Err("coefficient_grid must not be empty".to_string());
        }
        Ok(())
    }
}
