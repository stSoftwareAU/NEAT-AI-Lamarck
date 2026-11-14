//! The weight objective, its evaluation budget, and influential-synapse
//! selection.

use std::fmt;
use std::collections::HashSet;

use neat_core::{CreatureExport, compile_creature, mse_sum_batch_packed};

use crate::candidates::RANDOM_DELTA_HALF_WIDTH;

/// Errors raised by the weight bench.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchError {
    /// An evaluation was requested past the budget limit.
    BudgetExhausted {
        /// The budget limit that was exceeded.
        limit: usize,
    },
    /// A loss evaluation failed (compile error, wrong length, non-finite loss).
    Evaluation(String),
    /// A configuration was rejected before any work ran.
    InvalidConfig(String),
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BudgetExhausted { limit } => {
                write!(f, "evaluation budget of {limit} exhausted")
            }
            Self::Evaluation(m) => write!(f, "evaluation failed: {m}"),
            Self::InvalidConfig(m) => write!(f, "invalid configuration: {m}"),
        }
    }
}

impl std::error::Error for BenchError {}

/// Mean squared error of a creature as a function of a chosen set of synapse
/// weights; every other parameter stays fixed.
#[derive(Debug, Clone)]
pub struct WeightObjective {
    creature: CreatureExport,
    synapse_indices: Vec<usize>,
    records: Vec<f32>,
}

impl WeightObjective {
    /// Build an objective over `synapse_indices` of `creature`, scored on the
    /// packed `records` (per record: inputs then outputs). Both dimensions must
    /// be positive and each selected synapse must be a distinct axis.
    pub fn new(
        creature: CreatureExport,
        synapse_indices: Vec<usize>,
        records: &[f32],
    ) -> Result<Self, BenchError> {
        if creature.input == 0 || creature.output == 0 {
            return Err(BenchError::InvalidConfig(
                "creature input and output must both be positive".into(),
            ));
        }
        let width = creature.input + creature.output;
        if synapse_indices.is_empty() {
            return Err(BenchError::InvalidConfig(
                "no synapses selected for the objective".into(),
            ));
        }
        if let Some(bad) = synapse_indices
            .iter()
            .find(|&&i| i >= creature.synapses.len())
        {
            return Err(BenchError::InvalidConfig(format!(
                "synapse index {bad} out of range (creature has {})",
                creature.synapses.len()
            )));
        }
        if synapse_indices.iter().collect::<HashSet<_>>().len() != synapse_indices.len() {
            return Err(BenchError::InvalidConfig(
                "synapse indices must be distinct".into(),
            ));
        }
        if records.is_empty() || !records.len().is_multiple_of(width) {
            return Err(BenchError::InvalidConfig(format!(
                "records must be a non-empty multiple of the record width {width}"
            )));
        }
        Ok(Self {
            creature,
            synapse_indices,
            records: records.to_vec(),
        })
    }

    /// Number of free weights.
    pub fn dimension(&self) -> usize {
        self.synapse_indices.len()
    }

    /// Current weights of the chosen synapses.
    pub fn start(&self) -> Vec<f64> {
        self.synapse_indices
            .iter()
            .map(|&i| self.creature.synapses[i].weight)
            .collect()
    }

    /// Mean squared error over the objective's own (screening) records.
    pub fn loss(&self, x: &[f64]) -> Result<f64, String> {
        self.loss_on(x, &self.records)
    }

    /// Mean squared error of the creature with weights `x` over other packed
    /// `records` (used for the confirmation passes).
    pub fn loss_on(&self, x: &[f64], records: &[f32]) -> Result<f64, String> {
        if x.len() != self.dimension() {
            return Err(format!(
                "expected {} weights, got {}",
                self.dimension(),
                x.len()
            ));
        }
        let mut creature = self.creature.clone();
        for (&i, &w) in self.synapse_indices.iter().zip(x) {
            creature.synapses[i].weight = w;
        }
        let width = creature.input + creature.output;
        if records.is_empty() || !records.len().is_multiple_of(width) {
            return Err("records are empty or not a multiple of the record width".into());
        }
        let mut net = compile_creature(&creature).map_err(|e| e.to_string())?;
        let sum = mse_sum_batch_packed(
            &mut net,
            records,
            creature.input,
            creature.output,
            creature.forward_only,
        );
        let loss = sum / (records.len() / width) as f64;
        if !loss.is_finite() {
            return Err(format!("non-finite loss ({loss})"));
        }
        Ok(loss)
    }
}

/// An evaluation budget: a fixed limit and a running count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Maximum number of evaluations allowed.
    pub limit: usize,
    /// Evaluations spent so far.
    pub used: usize,
}

/// A [`WeightObjective`] behind an evaluation budget, tracking the best point.
#[derive(Debug)]
pub struct BudgetedObjective<'a> {
    objective: &'a WeightObjective,
    budget: Budget,
    best: Option<(f64, Vec<f64>)>,
}

impl<'a> BudgetedObjective<'a> {
    /// Wrap `objective` with a budget of `limit` evaluations.
    pub fn new(objective: &'a WeightObjective, limit: usize) -> Self {
        Self {
            objective,
            budget: Budget { limit, used: 0 },
            best: None,
        }
    }

    /// Number of free weights.
    pub fn dimension(&self) -> usize {
        self.objective.dimension()
    }

    /// Starting weights.
    pub fn start(&self) -> Vec<f64> {
        self.objective.start()
    }

    /// The total budget limit.
    pub fn limit(&self) -> usize {
        self.budget.limit
    }

    /// Evaluations spent so far.
    pub fn used(&self) -> usize {
        self.budget.used
    }

    /// Evaluations still available.
    pub fn remaining(&self) -> usize {
        self.budget.limit - self.budget.used
    }

    /// Best `(loss, weights)` seen so far, if anything has been evaluated.
    pub fn best(&self) -> Option<(f64, &[f64])> {
        self.best.as_ref().map(|(l, x)| (*l, x.as_slice()))
    }

    /// Evaluate `x`, spending one evaluation.
    ///
    /// Fails with [`BenchError::BudgetExhausted`] when the budget is spent; the
    /// count never exceeds the limit. An evaluation that fails still counts.
    pub fn eval(&mut self, x: &[f64]) -> Result<f64, BenchError> {
        if self.budget.used >= self.budget.limit {
            return Err(BenchError::BudgetExhausted {
                limit: self.budget.limit,
            });
        }
        self.budget.used += 1;
        let loss = self.objective.loss(x).map_err(BenchError::Evaluation)?;
        if self.best.as_ref().is_none_or(|(b, _)| loss < *b) {
            self.best = Some((loss, x.to_vec()));
        }
        Ok(loss)
    }
}

/// Preparation step: pick the `d` synapses whose weight the loss is most
/// sensitive to.
///
/// Sensitivity is the absolute change in loss from a one-sided finite
/// difference of [`RANDOM_DELTA_HALF_WIDTH`]. Returns the chosen indices
/// (ascending; ties go to the lower index) and the evaluations spent
/// (`synapses + 1`).
pub fn select_influential(
    creature: &CreatureExport,
    records: &[f32],
    d: usize,
) -> Result<(Vec<usize>, usize), BenchError> {
    let n = creature.synapses.len();
    if d == 0 || d > n {
        return Err(BenchError::InvalidConfig(format!(
            "cannot select {d} synapses from {n}"
        )));
    }
    let all: Vec<usize> = (0..n).collect();
    let objective = WeightObjective::new(creature.clone(), all, records)?;
    let x0 = objective.start();
    let eval = |x: &[f64]| objective.loss(x).map_err(BenchError::Evaluation);
    let base = eval(&x0)?;
    let mut scored = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = x0.clone();
        x[i] += RANDOM_DELTA_HALF_WIDTH;
        scored.push((i, (eval(&x)? - base).abs()));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut picked: Vec<usize> = scored.iter().take(d).map(|s| s.0).collect();
    picked.sort_unstable();
    Ok((picked, n + 1))
}
