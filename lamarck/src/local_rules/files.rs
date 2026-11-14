//! Files mode: the same predeclared protocol, on a private creature and corpus.
//!
//! One calibration creature supplies the rule-discovery records. Each validation
//! creature is a different creature (transfer) measured on the same validation
//! records. Records are read from a training-data directory with the crate's own
//! reader, split by position, and never written into a report.

use std::path::Path;

use neat_core::{CreatureExport, TrainingDataConfig};

use crate::analysis::open_training_scan;
use crate::baseline::training_data_key;
use crate::memo::creature_fingerprint;

use super::experiment::{DataSource, Report, run_protocol};
use super::manifest::Manifest;
use super::tasks::{Task, TaskRecord};

/// Group label of every validation task in files mode.
const FILES_GROUP: &str = "files";

/// Run `manifest` on `calibration_creature` and `validation_creatures`, reading
/// records from `training_data`.
///
/// Records are split by position: the first `train_records` train the
/// calibration task, the next `train_records` are the validation training
/// records, and the next `held_out_records` are the held-out records that every
/// validation creature is measured on.
pub fn run_files(
    manifest: &Manifest,
    calibration_creature: &CreatureExport,
    validation_creatures: &[CreatureExport],
    training_data: &Path,
) -> Result<Report, String> {
    manifest.validate()?;
    let (inputs, outputs) = (calibration_creature.input, calibration_creature.output);
    for (index, creature) in validation_creatures.iter().enumerate() {
        if (creature.input, creature.output) != (inputs, outputs) {
            return Err(format!(
                "validation creature {index} has input/output {}/{}, but the calibration creature has {inputs}/{outputs}",
                creature.input, creature.output
            ));
        }
    }
    let calibration_key = creature_fingerprint(calibration_creature);
    for (index, creature) in validation_creatures.iter().enumerate() {
        if creature_fingerprint(creature) == calibration_key {
            return Err(format!(
                "validation creature {index} is the calibration creature (same fingerprint); validation must use a different creature"
            ));
        }
    }

    if validation_creatures.len() < manifest.min_paired_tasks {
        return Err(format!(
            "files run has {} validation creatures, below the predeclared minimum of {} paired tasks",
            validation_creatures.len(),
            manifest.min_paired_tasks
        ));
    }

    let train = manifest.train_records;
    let held_out = manifest.held_out_records;
    let records = read_records(training_data, calibration_creature, train, held_out)?;
    let (calibration_records, rest) = records.split_at(train);
    let (validation_records, held_out_records) = rest.split_at(train);

    let calibration = vec![Task {
        name: "calibration".to_string(),
        spec: None,
        parent: calibration_creature.clone(),
        train: calibration_records.to_vec(),
        held_out: Vec::new(),
    }];
    let validation: Vec<(String, Task)> = validation_creatures
        .iter()
        .enumerate()
        .map(|(index, creature)| {
            let task = Task {
                name: format!("creature-{index}"),
                spec: None,
                parent: creature.clone(),
                train: validation_records.to_vec(),
                held_out: held_out_records.to_vec(),
            };
            (FILES_GROUP.to_string(), task)
        })
        .collect();

    let data_source = DataSource::Files {
        corpus_key: format!("{:016x}", training_data_key(training_data)?),
        calibration_creature: format!("{calibration_key:016x}"),
        validation_creatures: validation_creatures
            .iter()
            .map(|creature| format!("{:016x}", creature_fingerprint(creature)))
            .collect(),
    };
    run_protocol(manifest, &calibration, &validation, data_source)
}

/// The first `2 * train + held_out` records of `training_data`, read with the
/// calibration creature's widths. Refuses when the corpus holds fewer.
fn read_records(
    training_data: &Path,
    creature: &CreatureExport,
    train: usize,
    held_out: usize,
) -> Result<Vec<TaskRecord>, String> {
    let needed = 2 * train + held_out;
    let config = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = open_training_scan(training_data, config)?;
    let mut records = Vec::with_capacity(needed);
    while records.len() < needed {
        let Some(record) = iter.next_record().map_err(|e| e.to_string())? else {
            break;
        };
        records.push(TaskRecord {
            inputs: record.inputs,
            expected: record.outputs,
        });
    }
    if records.len() < needed {
        return Err(format!(
            "training data holds {} records but the protocol needs {needed}: {train} calibration, {train} validation training and {held_out} held-out",
            records.len()
        ));
    }
    Ok(records)
}
