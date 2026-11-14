//! The sparse symbolic rule language: activation families, local features and
//! sparse linear rules over them.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Activation family of a neuron, from its squash name (case-insensitive).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActivationFamily {
    /// `IDENTITY`, or no squash at all.
    Identity,
    /// `TANH`.
    Tanh,
    /// `LOGISTIC` or `SIGMOID`.
    Logistic,
    /// Any name containing `RELU`.
    Relu,
    /// Every other squash.
    Other,
}

impl ActivationFamily {
    /// Family of a neuron's squash name. `None` means the neat-core default, IDENTITY.
    pub fn from_squash(squash: Option<&str>) -> Self {
        let Some(name) = squash else {
            return Self::Identity;
        };
        let name = name.to_ascii_uppercase();
        match name.as_str() {
            "IDENTITY" => Self::Identity,
            "TANH" => Self::Tanh,
            "LOGISTIC" | "SIGMOID" => Self::Logistic,
            n if n.contains("RELU") => Self::Relu,
            _ => Self::Other,
        }
    }
}

/// Local residual of a neuron, or `None` when the neuron has no local residual.
///
/// Residual definition (issue #268): for an OUTPUT neuron `k` the residual is
/// `expected[k] - activation`, because the target is legitimately presented to
/// the output neuron. For a HIDDEN neuron it is `None`: the output error is not
/// treated as local hidden-neuron information. Features that use the residual
/// evaluate to `0.0` when it is `None`.
pub fn local_residual(
    is_output: bool,
    output_index: usize,
    activation: f64,
    expected: &[f32],
) -> Option<f64> {
    if !is_output {
        return None;
    }
    expected
        .get(output_index)
        .map(|target| f64::from(*target) - activation)
}

/// Local quantities visible to one synapse or bias update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalObservation {
    /// Presynaptic activation (`1.0` for a bias update).
    pub x: f64,
    /// Postsynaptic activation.
    pub y: f64,
    /// Current weight (or current bias for a bias update).
    pub w: f64,
    /// Local residual of the postsynaptic neuron, `None` for hidden neurons.
    pub r: Option<f64>,
}

/// The fixed feature library, in library order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Feature {
    /// Presynaptic activation.
    X,
    /// Postsynaptic activation.
    Y,
    /// Current weight.
    W,
    /// Local residual (zero when absent).
    R,
    /// `x * y`.
    XY,
    /// `x * r` (zero when the residual is absent).
    XR,
    /// `y * r` (zero when the residual is absent).
    YR,
    /// `w * y * y`.
    WYY,
    /// `x * y * y`.
    XYY,
}

impl Feature {
    /// Every feature, in library order.
    pub const ALL: [Feature; 9] = [
        Feature::X,
        Feature::Y,
        Feature::W,
        Feature::R,
        Feature::XY,
        Feature::XR,
        Feature::YR,
        Feature::WYY,
        Feature::XYY,
    ];

    /// Value of this feature for one observation.
    pub fn eval(&self, obs: &LocalObservation) -> f64 {
        let r = obs.r.unwrap_or(0.0);
        match self {
            Feature::X => obs.x,
            Feature::Y => obs.y,
            Feature::W => obs.w,
            Feature::R => r,
            Feature::XY => obs.x * obs.y,
            Feature::XR => obs.x * r,
            Feature::YR => obs.y * r,
            Feature::WYY => obs.w * obs.y * obs.y,
            Feature::XYY => obs.x * obs.y * obs.y,
        }
    }
}

/// A sparse linear rule: `delta = sum of c * feature`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseRule {
    /// Feature and coefficient pairs, each feature at most once.
    pub terms: Vec<(Feature, f64)>,
}

impl SparseRule {
    /// Build a rule, refusing empty, oversized, duplicated or non-finite terms.
    pub fn new(terms: Vec<(Feature, f64)>, max_support: usize) -> Result<Self, String> {
        if terms.is_empty() {
            return Err("sparse rule has no terms".to_string());
        }
        if terms.len() > max_support {
            return Err(format!(
                "sparse rule has {} terms, above max_support {max_support}",
                terms.len()
            ));
        }
        let mut seen = BTreeSet::new();
        for (feature, coefficient) in &terms {
            if !seen.insert(*feature) {
                return Err(format!("duplicate feature {feature:?} in sparse rule"));
            }
            if !coefficient.is_finite() {
                return Err(format!("non-finite coefficient for {feature:?}"));
            }
        }
        Ok(Self { terms })
    }

    /// Update for one observation: `sum of c * phi`.
    pub fn delta(&self, obs: &LocalObservation) -> f64 {
        self.terms
            .iter()
            .map(|(feature, coefficient)| coefficient * feature.eval(obs))
            .sum()
    }
}

impl std::fmt::Display for SparseRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts: Vec<String> = self
            .terms
            .iter()
            .map(|(feature, coefficient)| format!("{feature:?}*{coefficient}"))
            .collect();
        write!(f, "{}", parts.join(" + "))
    }
}

/// One sparse rule per activation family. A family with no rule gets a zero update.
pub type RuleSet = BTreeMap<ActivationFamily, SparseRule>;

/// Update for a neuron of `family` under `rules` (zero when the family has no rule).
pub fn rule_delta(rules: &RuleSet, family: ActivationFamily, obs: &LocalObservation) -> f64 {
    rules.get(&family).map_or(0.0, |rule| rule.delta(obs))
}

/// Human-readable form of a rule set, such as `{"tanh": "XR*1 + WYY*-0.5"}`.
pub fn describe(rules: &RuleSet) -> BTreeMap<String, String> {
    rules
        .iter()
        .map(|(family, rule)| (format!("{family:?}").to_lowercase(), rule.to_string()))
        .collect()
}
