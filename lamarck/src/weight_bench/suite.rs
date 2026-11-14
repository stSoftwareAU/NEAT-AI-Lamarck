//! Synthetic, application-agnostic problem generator.
//!
//! A random teacher network (16 inputs, 16 hidden neurons of one squash
//! family, fan-in 8, one identity output) generates the records; the start
//! creature is the teacher with every weight nudged, i.e. an already-fit
//! creature that is slightly off.

use std::path::PathBuf;

use neat_core::{CreatureExport, TrainingDataConfig, TrainingDataIterator, parse_creature_json};
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Family {
    /// `TANH`.
    Tanh,
    /// `RELU`.
    Relu,
    /// `HARD_TANH`.
    HardTanh,
    /// A real creature and fixed records supplied by the caller (see
    /// [`problem_from_corpus`]); not a synthetic family.
    Corpus,
}

impl Family {
    /// Squash name of a synthetic family; `None` for [`Family::Corpus`].
    fn squash_name(self) -> Option<&'static str> {
        match self {
            Self::Tanh => Some("TANH"),
            Self::Relu => Some("RELU"),
            Self::HardTanh => Some("HARD_TANH"),
            Self::Corpus => None,
        }
    }

    fn apply(self, x: f64) -> f64 {
        match self {
            Self::Tanh => x.tanh(),
            Self::Relu => x.max(0.0),
            Self::HardTanh => x.clamp(-1.0, 1.0),
            // Never reached for a built problem; identity is harmless.
            Self::Corpus => x,
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
    let Some(squash) = spec.family.squash_name() else {
        return Err(BenchError::InvalidConfig(
            "corpus problems come from problem_from_corpus, not build_problem".into(),
        ));
    };
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
            squash
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

/// A real creature plus fixed training records, for representative runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusSpec {
    /// Creature JSON file.
    pub creature: PathBuf,
    /// Training-data directory.
    pub training_data: PathBuf,
    /// Records (from the start of the window) used for screening.
    pub screening_records: usize,
    /// Records (from the end of the window) held out.
    pub held_out_records: usize,
    /// Most records read from the corpus, in file order.
    pub max_records: usize,
}

/// Load the creature and up to `max_records` packed records (inputs then
/// outputs), in file order.
pub fn load_corpus(spec: &CorpusSpec) -> Result<(CreatureExport, Vec<f32>), BenchError> {
    let bad = |m: String| BenchError::InvalidConfig(m);
    if spec.screening_records == 0 || spec.held_out_records == 0 {
        return Err(bad(format!(
            "corpus screeningRecords ({}) and heldOutRecords ({}) must both be positive",
            spec.screening_records, spec.held_out_records
        )));
    }
    let text = std::fs::read_to_string(&spec.creature)
        .map_err(|e| bad(format!("corpus creature {}: {e}", spec.creature.display())))?;
    let creature = crate::width::load_creature(&text)
        .map_err(|e| bad(format!("corpus creature {}: {e}", spec.creature.display())))?;
    let config = TrainingDataConfig::new(creature.input, creature.output);
    let training = |e: String| {
        bad(format!(
            "corpus training data {}: {e}",
            spec.training_data.display()
        ))
    };
    let mut iter = TrainingDataIterator::new(&spec.training_data, config)
        .map_err(|e| training(e.to_string()))?;
    let width = creature.input + creature.output;
    let mut packed: Vec<f32> = Vec::new();
    let mut count = 0usize;
    while count < spec.max_records {
        let Some(record) = iter.next_record().map_err(|e| training(e.to_string()))? else {
            break;
        };
        if record.inputs.len() < creature.input || record.outputs.len() < creature.output {
            continue;
        }
        packed.extend_from_slice(&record.inputs[..creature.input]);
        packed.extend_from_slice(&record.outputs[..creature.output]);
        count += 1;
    }
    debug_assert_eq!(packed.len(), count * width);
    let needed = spec.screening_records + spec.held_out_records;
    if count < needed {
        return Err(bad(format!(
            "corpus has {count} usable records but needs screeningRecords + heldOutRecords ({} + {}), {needed} in all",
            spec.screening_records, spec.held_out_records
        )));
    }
    Ok((creature, packed))
}

/// Build a problem from a loaded corpus. Within the window of `T` records,
/// screening is the first `screening_records`, held-out is the last
/// `held_out_records`, and full is the window minus its held-out tail. Nothing
/// is calibrated (the router that needed a calibration half was removed), so
/// every record is untouched by tuning.
pub fn problem_from_corpus(
    creature: &CreatureExport,
    packed: &[f32],
    spec: &CorpusSpec,
    dimension: usize,
    seed: u64,
) -> Result<Problem, BenchError> {
    if !(MIN_DIMENSION..=MAX_DIMENSION).contains(&dimension) {
        return Err(BenchError::InvalidConfig(format!(
            "dimension {dimension} outside {MIN_DIMENSION}..={MAX_DIMENSION}"
        )));
    }
    if dimension > creature.synapses.len() {
        return Err(BenchError::InvalidConfig(format!(
            "dimension {dimension} exceeds the creature's {} synapses",
            creature.synapses.len()
        )));
    }
    let width = creature.input + creature.output;
    let total = packed.len() / width;
    let needed = spec.screening_records + spec.held_out_records;
    if spec.screening_records == 0 || spec.held_out_records == 0 || total < needed {
        return Err(BenchError::InvalidConfig(format!(
            "corpus has {total} records but needs screeningRecords + heldOutRecords ({} + {}), {needed} in all",
            spec.screening_records, spec.held_out_records
        )));
    }
    let window = &packed[..total * width];
    let held_start = (total - spec.held_out_records) * width;
    let screening = window[..spec.screening_records * width].to_vec();
    let full = window[..held_start].to_vec();
    let held_out = window[held_start..].to_vec();
    let (synapse_indices, preparation_evaluations) =
        select_influential(creature, &screening, dimension)?;
    Ok(Problem {
        spec: ProblemSpec {
            family: Family::Corpus,
            dimension,
            seed,
        },
        creature: creature.clone(),
        synapse_indices,
        screening,
        full,
        held_out,
        preparation_evaluations,
    })
}
