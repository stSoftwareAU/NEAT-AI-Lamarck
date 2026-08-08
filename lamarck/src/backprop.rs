//! Rust port surface for NEAT-AI backpropagation configuration and learning
//! application helpers.
//!
//! The heavy reverse-topological loop lives in `neat-core`
//! ([`neat_core::propagate_topological_loop`]). This module ports the
//! TypeScript `BackPropagation` configuration / learning-rate semantics and
//! exposes analyse-without-apply accumulation helpers that finalise bias and
//! weight proposals via [`neat_core::calculate_bias`] /
//! [`neat_core::calculate_weight`].

use neat_core::{
    PropagateOutcome, PropagateOutput, StandardOutcome, SynapseDelta, calculate_bias,
    calculate_weight,
};
use serde::{Deserialize, Serialize};

/// Floating-point absolute tolerance used by parity-style unit tests.
pub const FLOAT_ABS_TOL: f64 = 1e-9;

/// Floating-point relative tolerance used by parity-style unit tests.
pub const FLOAT_REL_TOL: f64 = 1e-6;

/// Learning-rate schedule strategies matching the TypeScript port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LearningRateStrategy {
    /// Constant learning rate.
    #[default]
    Fixed,
    /// Multiplicative decay each iteration.
    Decay,
    /// Boost when error improves / stagnates.
    Adaptive,
    /// Decay with periodic warm restart.
    WarmRestart,
}

/// Backpropagation configuration — behavioural port of NEAT-AI defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackpropConfig {
    /// Generational dampening weight.
    pub generations: f64,
    /// Active learning rate (may be overridden by strategy).
    pub learning_rate: f64,
    /// Maximum bias adjustment magnitude per apply.
    pub maximum_bias_adjustment_scale: f64,
    /// Maximum weight adjustment magnitude per apply.
    pub maximum_weight_adjustment_scale: f64,
    /// Hard bias magnitude limit.
    pub limit_bias_scale: f64,
    /// Hard weight magnitude limit.
    pub limit_weight_scale: f64,
    /// Minimum meaningful magnitude (plank constant).
    pub plank_constant: f64,
    /// Disable bias updates.
    pub disable_bias_adjustment: bool,
    /// Disable weight updates.
    pub disable_weight_adjustment: bool,
    /// Learning-rate strategy.
    pub learning_rate_strategy: LearningRateStrategy,
    /// Initial learning rate for decay/adaptive/warm_restart.
    pub initial_learning_rate: f64,
    /// Decay factor for decay / warm_restart.
    pub learning_rate_decay: f64,
    /// L1 bias decay.
    pub l1_bias_decay: f64,
    /// L2 bias decay.
    pub l2_bias_decay: f64,
    /// L1 weight decay.
    pub l1_weight_decay: f64,
    /// L2 weight decay.
    pub l2_weight_decay: f64,
}

impl Default for BackpropConfig {
    fn default() -> Self {
        Self {
            generations: 1.0,
            learning_rate: 0.01,
            maximum_bias_adjustment_scale: 10.0,
            maximum_weight_adjustment_scale: 10.0,
            limit_bias_scale: 10_000.0,
            limit_weight_scale: 100_000.0,
            plank_constant: 1e-7,
            disable_bias_adjustment: false,
            disable_weight_adjustment: false,
            learning_rate_strategy: LearningRateStrategy::Fixed,
            initial_learning_rate: 0.01,
            learning_rate_decay: 0.95,
            l1_bias_decay: 0.0,
            l2_bias_decay: 0.0,
            l1_weight_decay: 0.0,
            l2_weight_decay: 0.0,
        }
    }
}

/// Compute the learning rate for an iteration using the configured strategy.
pub fn calculate_learning_rate(
    config: &BackpropConfig,
    iteration: u64,
    error_feedback: Option<(f64, f64)>,
) -> f64 {
    match config.learning_rate_strategy {
        LearningRateStrategy::Fixed => config.learning_rate,
        LearningRateStrategy::Decay => {
            config.initial_learning_rate * config.learning_rate_decay.powi(iteration as i32)
        }
        LearningRateStrategy::WarmRestart => {
            let cycle = 10u64;
            let pos = iteration % cycle;
            config.initial_learning_rate * config.learning_rate_decay.powi(pos as i32)
        }
        LearningRateStrategy::Adaptive => {
            let mut lr = config.initial_learning_rate;
            if let Some((prev, curr)) = error_feedback
                && prev > 0.0
            {
                let ratio = curr / prev;
                if ratio < 0.95 {
                    lr *= 1.1;
                } else if ratio >= 1.0 {
                    lr *= 1.3;
                } else {
                    lr *= ratio.max(0.5);
                }
            }
            lr.clamp(1e-8, 1.0)
        }
    }
}

/// Accumulated bias learning signal for one neuron (analyse-without-apply).
#[derive(Debug, Clone, Default)]
pub struct BiasSignal {
    /// Accumulation count.
    pub count: f64,
    /// Sum of adjusted bias contributions.
    pub total_adjusted_bias: f64,
    /// Whether the neuron flagged no-change.
    pub no_change: bool,
}

impl BiasSignal {
    /// Fold a standard backprop outcome into this accumulator.
    pub fn accumulate_standard(&mut self, outcome: &StandardOutcome) {
        self.count += f64::from(outcome.bias_count_delta);
        self.total_adjusted_bias += f64::from(outcome.total_adjusted_bias_delta);
        self.no_change |= outcome.no_change;
    }

    /// Propose a new bias without mutating the creature.
    pub fn propose(&self, current_bias: f64, config: &BackpropConfig, learning_rate: f64) -> f64 {
        if config.disable_bias_adjustment {
            return current_bias;
        }
        calculate_bias(
            self.count,
            self.total_adjusted_bias,
            current_bias,
            self.no_change,
            config.generations,
            config.plank_constant,
            learning_rate,
            config.maximum_bias_adjustment_scale,
            config.limit_bias_scale,
            config.l1_bias_decay,
            config.l2_bias_decay,
        )
    }
}

/// Accumulated weight learning signal for one synapse (analyse-without-apply).
#[derive(Debug, Clone, Default)]
pub struct WeightSignal {
    /// Accumulation count.
    pub count: f64,
    /// Positive activation mass.
    pub total_positive_activation: f64,
    /// Negative activation mass.
    pub total_negative_activation: f64,
    /// Positive activation count.
    pub count_positive: f64,
    /// Negative activation count.
    pub count_negative: f64,
    /// Positive adjusted-value mass.
    pub total_positive_adjusted_value: f64,
    /// Negative adjusted-value mass.
    pub total_negative_adjusted_value: f64,
}

impl WeightSignal {
    /// Fold a synapse delta into this accumulator.
    pub fn accumulate_delta(&mut self, delta: &SynapseDelta) {
        self.count += f64::from(delta.count);
        self.total_positive_activation += f64::from(delta.total_positive_activation);
        self.total_negative_activation += f64::from(delta.total_negative_activation);
        self.count_positive += f64::from(delta.count_positive);
        self.count_negative += f64::from(delta.count_negative);
        self.total_positive_adjusted_value += f64::from(delta.total_positive_adjusted_value);
        self.total_negative_adjusted_value += f64::from(delta.total_negative_adjusted_value);
    }

    /// Propose a new weight without mutating the creature.
    pub fn propose(&self, current_weight: f64, config: &BackpropConfig, learning_rate: f64) -> f64 {
        if config.disable_weight_adjustment {
            return current_weight;
        }
        calculate_weight(
            self.count,
            self.total_positive_activation,
            self.total_negative_activation,
            self.count_positive,
            self.count_negative,
            self.total_positive_adjusted_value,
            self.total_negative_adjusted_value,
            current_weight,
            config.generations,
            config.plank_constant,
            learning_rate,
            config.maximum_weight_adjustment_scale,
            config.limit_weight_scale,
            config.l1_weight_decay,
            config.l2_weight_decay,
        )
    }
}

/// Aggregated learning signal for a creature — never applied automatically.
#[derive(Debug, Clone, Default)]
pub struct LearningSignal {
    /// Per-neuron bias signals (indexed by neuron id in the propagate layout).
    pub biases: Vec<BiasSignal>,
    /// Per-synapse weight signals.
    pub weights: Vec<WeightSignal>,
}

impl LearningSignal {
    /// Create empty accumulators sized for a network.
    pub fn new(neuron_count: usize, synapse_count: usize) -> Self {
        Self {
            biases: vec![BiasSignal::default(); neuron_count],
            weights: vec![WeightSignal::default(); synapse_count],
        }
    }

    /// Fold one [`PropagateOutput`] into this signal (analyse-without-apply).
    pub fn accumulate_propagate_output(&mut self, output: &PropagateOutput) {
        for (idx, outcome) in output.neurons.iter().enumerate() {
            if let Some(signal) = self.biases.get_mut(idx)
                && let PropagateOutcome::Standard(standard) = outcome
            {
                signal.accumulate_standard(standard);
            }
        }
        for (idx, delta) in output.synapses.iter().enumerate() {
            if let Some(signal) = self.weights.get_mut(idx) {
                signal.accumulate_delta(delta);
            }
        }
    }
}

/// Compare two floats with ordinary absolute/relative tolerances.
pub fn nearly_equal(a: f64, b: f64) -> bool {
    let diff = (a - b).abs();
    if diff <= FLOAT_ABS_TOL {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    diff <= FLOAT_REL_TOL * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_learning_rate_is_constant() {
        let cfg = BackpropConfig::default();
        assert!(nearly_equal(calculate_learning_rate(&cfg, 0, None), 0.01));
        assert!(nearly_equal(calculate_learning_rate(&cfg, 100, None), 0.01));
    }

    #[test]
    fn decay_learning_rate_shrinks() {
        let cfg = BackpropConfig {
            learning_rate_strategy: LearningRateStrategy::Decay,
            initial_learning_rate: 0.1,
            learning_rate_decay: 0.5,
            ..Default::default()
        };
        let lr0 = calculate_learning_rate(&cfg, 0, None);
        let lr2 = calculate_learning_rate(&cfg, 2, None);
        assert!(nearly_equal(lr0, 0.1));
        assert!(nearly_equal(lr2, 0.025));
    }

    #[test]
    fn bias_propose_uses_neat_core_calculate_bias() {
        let cfg = BackpropConfig::default();
        let signal = BiasSignal {
            count: 10.0,
            total_adjusted_bias: 5.0,
            no_change: false,
        };
        let proposed = signal.propose(0.0, &cfg, 0.01);
        let expected = calculate_bias(
            10.0,
            5.0,
            0.0,
            false,
            cfg.generations,
            cfg.plank_constant,
            0.01,
            cfg.maximum_bias_adjustment_scale,
            cfg.limit_bias_scale,
            0.0,
            0.0,
        );
        assert!(nearly_equal(proposed, expected));
    }

    #[test]
    fn weight_propose_uses_neat_core_calculate_weight() {
        let cfg = BackpropConfig::default();
        let signal = WeightSignal {
            count: 4.0,
            total_positive_activation: 2.0,
            total_negative_activation: 0.0,
            count_positive: 4.0,
            count_negative: 0.0,
            total_positive_adjusted_value: 1.0,
            total_negative_adjusted_value: 0.0,
        };
        let proposed = signal.propose(0.5, &cfg, 0.01);
        let expected = calculate_weight(
            4.0,
            2.0,
            0.0,
            4.0,
            0.0,
            1.0,
            0.0,
            0.5,
            cfg.generations,
            cfg.plank_constant,
            0.01,
            cfg.maximum_weight_adjustment_scale,
            cfg.limit_weight_scale,
            0.0,
            0.0,
        );
        assert!(nearly_equal(proposed, expected));
    }

    #[test]
    fn learning_signal_does_not_mutate_creature_on_accumulate() {
        let mut signal = LearningSignal::new(2, 1);
        let output = PropagateOutput {
            neurons: vec![
                PropagateOutcome::Skipped,
                PropagateOutcome::Standard(StandardOutcome {
                    bias_count_delta: 1,
                    total_adjusted_bias_delta: 0.25,
                    ..StandardOutcome::default()
                }),
            ],
            synapses: vec![SynapseDelta {
                count: 1.0,
                total_positive_activation: 1.0,
                count_positive: 1.0,
                total_positive_adjusted_value: 0.5,
                ..SynapseDelta::default()
            }],
        };
        signal.accumulate_propagate_output(&output);
        assert!(nearly_equal(signal.biases[1].count, 1.0));
        assert!(nearly_equal(signal.weights[0].count, 1.0));
    }
}
