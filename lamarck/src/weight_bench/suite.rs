//! Synthetic, application-agnostic problem generator.
//!
//! A random teacher network (16 inputs, 16 hidden neurons of one squash
//! family, fan-in 8, one identity output) generates the records; the start
//! creature is the teacher with every weight nudged, i.e. an already-fit
//! creature that is slightly off.

use neat_core::{CreatureExport, parse_creature_json};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

use super::objective::{BenchError, select_influential};

const INPUTS: usize = 16;
const HIDDEN: usize = 16;
const FAN_IN: usize = 8;
const SCREENING_RECORDS: usize = 256;
const FULL_RECORDS: usize = 1024;
const HELD_OUT_RECORDS: usize = 512;
const TARGET_NOISE: f64 = 0.01;
const START_PERTURBATION: f64 = 0.3;
/// Smallest allowed dimension.
pub const MIN_DIMENSION: usize = 20;
/// Largest allowed dimension.
pub const MAX_DIMENSION: usize = 100;
/// Dimensions above this are extrapolation beyond the calibrated range.
pub const EXTRAPOLATION_ABOVE: usize = 40;

/// Hidden-neuron squash family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Family {
    /// `TANH`.
    Tanh,
    /// `RELU`.
    Relu,
    /// `HARD_TANH`.
    HardTanh,
}

impl Family {
    fn squash_name(self) -> &'static str {
        match self {
            Self::Tanh => "TANH",
            Self::Relu => "RELU",
            Self::HardTanh => "HARD_TANH",
        }
    }

    fn apply(self, x: f64) -> f64 {
        match self {
            Self::Tanh => x.tanh(),
            Self::Relu => x.max(0.0),
            Self::HardTanh => x.clamp(-1.0, 1.0),
        }
    }
}

/// Which problem to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProblemSpec {
    /// Squash family.
    pub family: Family,
    /// Free weights to optimise (`20..=100`).
    pub dimension: usize,
    /// Seed for every random stream of the problem.
    pub seed: u64,
}

/// A built problem.
#[derive(Debug, Clone)]
pub struct Problem {
    /// The spec it came from.
    pub spec: ProblemSpec,
    /// The start creature (teacher with perturbed weights).
    pub creature: CreatureExport,
    /// The `dimension` synapses chosen as free weights (ascending).
    pub synapse_indices: Vec<usize>,
    /// Screening records (packed inputs then output).
    pub screening: Vec<f32>,
    /// Full records; the first screening-count records equal `screening`.
    pub full: Vec<f32>,
    /// Held-out records from a disjoint stream.
    pub held_out: Vec<f32>,
    /// Evaluations spent selecting the free weights.
    pub preparation_evaluations: usize,
}

/// Whether `dimension` lies beyond the calibrated range.
pub fn is_extrapolation(dimension: usize) -> bool {
    dimension > EXTRAPOLATION_ABOVE
}

/// `n` distinct values from `0..m`, by partial Fisher-Yates.
fn choose(rng: &mut StdRng, m: usize, n: usize) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..m).collect();
    for i in 0..n {
        let j = rng.random_range(i..m);
        pool.swap(i, j);
    }
    pool.truncate(n);
    pool
}

fn records(
    rng: &mut StdRng,
    count: usize,
    family: Family,
    taps: &[Vec<usize>],
    w_in: &[Vec<f64>],
    w_out: &[f64],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(count * (INPUTS + 1));
    for _ in 0..count {
        let x: Vec<f32> = (0..INPUTS)
            .map(|_| rng.random_range(-1.0..1.0f64) as f32)
            .collect();
        let mut y = 0.0;
        for h in 0..HIDDEN {
            let pre: f64 = taps[h]
                .iter()
                .zip(&w_in[h])
                .map(|(&i, &w)| w * f64::from(x[i]))
                .sum();
            y += w_out[h] * family.apply(pre);
        }
        y += rng.random_range(-TARGET_NOISE..TARGET_NOISE);
        out.extend_from_slice(&x);
        out.push(y as f32);
    }
    out
}

/// Build the problem for `spec`.
pub fn build_problem(spec: &ProblemSpec) -> Result<Problem, BenchError> {
    if !(MIN_DIMENSION..=MAX_DIMENSION).contains(&spec.dimension) {
        return Err(BenchError::InvalidConfig(format!(
            "dimension {} outside {MIN_DIMENSION}..={MAX_DIMENSION}",
            spec.dimension
        )));
    }
    let mut rng = StdRng::seed_from_u64(spec.seed);
    let taps: Vec<Vec<usize>> = (0..HIDDEN)
        .map(|_| choose(&mut rng, INPUTS, FAN_IN))
        .collect();
    let w_in: Vec<Vec<f64>> = (0..HIDDEN)
        .map(|_| (0..FAN_IN).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect();
    let w_out: Vec<f64> = (0..HIDDEN).map(|_| rng.random_range(-1.0..1.0)).collect();

    let mut data_rng = StdRng::seed_from_u64(spec.seed ^ 0x5352_4545_4e49);
    let full = records(
        &mut data_rng,
        FULL_RECORDS,
        spec.family,
        &taps,
        &w_in,
        &w_out,
    );
    let screening = full[..SCREENING_RECORDS * (INPUTS + 1)].to_vec();
    let mut held_rng = StdRng::seed_from_u64(spec.seed ^ 0x4845_4c44_4f55);
    let held_out = records(
        &mut held_rng,
        HELD_OUT_RECORDS,
        spec.family,
        &taps,
        &w_in,
        &w_out,
    );

    // Start creature: teacher weights, each nudged. Synapse order is, per
    // hidden, its FAN_IN input synapses then its output synapse.
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for h in 0..HIDDEN {
        neurons.push(format!(
            r#"{{"type":"hidden","uuid":"h{h}","bias":0.0,"squash":"{}"}}"#,
            spec.family.squash_name()
        ));
        for (k, &i) in taps[h].iter().enumerate() {
            let w = w_in[h][k] + rng.random_range(-START_PERTURBATION..START_PERTURBATION);
            synapses.push(format!(
                r#"{{"fromUUID":"input-{i}","toUUID":"h{h}","weight":{w}}}"#
            ));
        }
        let w = w_out[h] + rng.random_range(-START_PERTURBATION..START_PERTURBATION);
        synapses.push(format!(
            r#"{{"fromUUID":"h{h}","toUUID":"o1","weight":{w}}}"#
        ));
    }
    neurons.push(r#"{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}"#.to_string());
    let json = format!(
        r#"{{"semanticVersion":"4.0.0","forwardOnly":true,"input":{INPUTS},"output":1,"neurons":[{}],"synapses":[{}]}}"#,
        neurons.join(","),
        synapses.join(",")
    );
    let creature = parse_creature_json(&json)
        .map_err(|e| BenchError::Evaluation(format!("building creature: {e}")))?;

    let (synapse_indices, preparation_evaluations) =
        select_influential(&creature, &screening, spec.dimension)?;
    Ok(Problem {
        spec: *spec,
        creature,
        synapse_indices,
        screening,
        full,
        held_out,
        preparation_evaluations,
    })
}
