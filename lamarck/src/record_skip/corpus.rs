//! Corpora and parent creatures for the record-skip experiment.

use std::path::Path;

use neat_core::{CreatureExport, TrainingDataConfig, parse_creature_json};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// One training or validation record.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchRecord {
    /// Input values.
    pub inputs: Vec<f32>,
    /// Expected output values.
    pub outputs: Vec<f32>,
}

/// A train/validation split. The two parts are disjoint; validation is never
/// refined on and never used for calibration.
#[derive(Clone, Debug)]
pub struct Corpus {
    /// Records the arms refine on.
    pub train: Vec<BenchRecord>,
    /// Held-out records used only for measurement.
    pub validation: Vec<BenchRecord>,
    /// Identity of the corpus.
    pub key: String,
}

/// Synthetic corpus: a smooth common region and a rarer, distinct high-target
/// region (about 5% of records, `x3 > 0.9`).
pub fn synthetic_corpus(seed: u64, train: usize, validation: usize) -> Corpus {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut draw = |count: usize| -> Vec<BenchRecord> {
        (0..count)
            .map(|_| {
                let x: [f32; 4] = std::array::from_fn(|_| rng.random_range(-1.0f32..1.0));
                let y = if x[3] > 0.9 {
                    1.2 - 0.5 * x[0]
                } else {
                    0.5 * (1.5 * x[0] - x[1]).tanh() + 0.1 * x[2]
                };
                BenchRecord {
                    inputs: x.to_vec(),
                    outputs: vec![y],
                }
            })
            .collect()
    };
    let train_records = draw(train);
    let validation_records = draw(validation);
    Corpus {
        train: train_records,
        validation: validation_records,
        key: format!("synthetic:{seed}:{train}:{validation}"),
    }
}

/// Synthetic parent: 4 inputs, 6 hidden TANH neurons, 1 IDENTITY output, with
/// weights and biases uniform in [-0.5, 0.5].
pub fn synthetic_parent(seed: u64) -> CreatureExport {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut w = || rng.random_range(-0.5f64..0.5);
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for h in 0..6 {
        neurons.push(format!(
            r#"{{"type":"hidden","uuid":"h{h}","bias":{},"squash":"TANH"}}"#,
            w()
        ));
        for i in 0..4 {
            synapses.push(format!(
                r#"{{"fromUUID":"input-{i}","toUUID":"h{h}","weight":{}}}"#,
                w()
            ));
        }
    }
    neurons.push(format!(
        r#"{{"type":"output","uuid":"o0","bias":{},"squash":"IDENTITY"}}"#,
        w()
    ));
    for h in 0..6 {
        synapses.push(format!(
            r#"{{"fromUUID":"h{h}","toUUID":"o0","weight":{}}}"#,
            w()
        ));
    }
    let json = format!(
        r#"{{"semanticVersion":"4.0.0","forwardOnly":true,"input":4,"output":1,"neurons":[{}],"synapses":[{}]}}"#,
        neurons.join(","),
        synapses.join(",")
    );
    parse_creature_json(&json).expect("synthetic parent JSON is well formed")
}

/// Load up to `max_records` records from `dir`; the last `validation_records`
/// of the window are validation and the rest are train.
pub fn load_corpus(
    dir: &Path,
    creature: &CreatureExport,
    validation_records: usize,
    max_records: usize,
) -> Result<Corpus, String> {
    let key = format!(
        "corpus:{:016x}:{max_records}:{validation_records}",
        crate::baseline::training_data_key(dir)?
    );
    let cfg = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = crate::analysis::open_training_scan(dir, cfg)?;
    let mut all = Vec::new();
    while all.len() < max_records {
        let Some(record) = iter.next_record().map_err(|e| e.to_string())? else {
            break;
        };
        all.push(BenchRecord {
            inputs: record.inputs.to_vec(),
            outputs: record.outputs.to_vec(),
        });
    }
    let loaded = all.len();
    if loaded <= validation_records || loaded - validation_records < 10 {
        return Err(format!(
            "corpus too small: loaded {loaded} records, validation_records {validation_records} \
             leaves {} train records (need more than validation_records and at least 10 train)",
            loaded.saturating_sub(validation_records)
        ));
    }
    let validation = all.split_off(loaded - validation_records);
    Ok(Corpus {
        train: all,
        validation,
        key,
    })
}
