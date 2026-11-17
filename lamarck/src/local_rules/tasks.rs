//! Application-agnostic synthetic regression tasks.
//!
//! A random TEACHER creature generates targets for uniformly drawn inputs. The
//! STUDENT parent is a different random creature of the same shape with small
//! weights. Nothing here depends on any application's data.

use neat_core::{CreatureExport, compile_creature, parse_creature_json, parse_squash_name};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

/// Version of the synthetic task generator, recorded in every report.
pub const TASK_GENERATOR_VERSION: &str = "local-rules-synthetic-v1";

/// Seed offsets keep the teacher, input and student streams independent.
const INPUT_STREAM: u64 = 0xA5A5_5A5A_1234_5678;
const STUDENT_STREAM: u64 = 0x51ED_2701_9E37_79B9;

/// One regression record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskRecord {
    /// Input values.
    pub inputs: Vec<f32>,
    /// Target output values.
    pub expected: Vec<f32>,
}

/// Parameters that identify one synthetic task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskSpec {
    /// Seed for the teacher, the inputs and the student.
    pub seed: u64,
    /// Hidden neurons in both teacher and student.
    pub hidden_width: usize,
    /// Squash name of every hidden neuron.
    pub hidden_squash: String,
}

/// A task: the parent creature plus its train and held-out records.
#[derive(Clone, Debug)]
pub struct Task {
    /// Label of the task within its group: `seed:width:squash` for a synthetic
    /// task, a creature index for a task built from files.
    pub name: String,
    /// The parameters that generated a synthetic task; `None` for a file task.
    pub spec: Option<TaskSpec>,
    /// The student creature that every arm starts from.
    pub parent: CreatureExport,
    /// Records the update arms may train on.
    pub train: Vec<TaskRecord>,
    /// Records used only for measurement.
    pub held_out: Vec<TaskRecord>,
}

/// Build a task from `spec`, with `inputs` inputs and the given record counts.
pub fn build_task(
    spec: TaskSpec,
    inputs: usize,
    train_records: usize,
    held_out_records: usize,
) -> Result<Task, String> {
    if spec.hidden_width == 0 {
        return Err("hidden_width must be positive".to_string());
    }
    parse_squash_name(&spec.hidden_squash).map_err(|e| e.to_string())?;
    let mut teacher_rng = StdRng::seed_from_u64(spec.seed);
    let teacher = random_creature(&mut teacher_rng, inputs, &spec, 1.0)?;
    let mut student_rng = StdRng::seed_from_u64(spec.seed ^ STUDENT_STREAM);
    let parent = random_creature(&mut student_rng, inputs, &spec, 0.5)?;

    let mut network = compile_creature(&teacher).map_err(|e| e.to_string())?;
    let mut input_rng = StdRng::seed_from_u64(spec.seed ^ INPUT_STREAM);
    let mut records = Vec::with_capacity(train_records + held_out_records);
    for _ in 0..train_records + held_out_records {
        let x: Vec<f32> = (0..inputs)
            .map(|_| input_rng.random_range(-1.0f32..1.0))
            .collect();
        // The trace returns outputs first; keep only the output values as targets.
        let expected: Vec<f32> = network
            .activate_and_trace(&x, teacher.output)
            .into_iter()
            .take(teacher.output)
            .collect();
        records.push(TaskRecord {
            inputs: x,
            expected,
        });
    }
    let held_out = records.split_off(train_records);
    Ok(Task {
        name: format!("{}:{}:{}", spec.seed, spec.hidden_width, spec.hidden_squash),
        spec: Some(spec),
        parent,
        train: records,
        held_out,
    })
}

/// Random fully connected input → hidden → one IDENTITY output creature, with
/// weights and biases uniform in `[-range, range]`.
fn random_creature(
    rng: &mut StdRng,
    inputs: usize,
    spec: &TaskSpec,
    range: f64,
) -> Result<CreatureExport, String> {
    let mut draw = || rng.random_range(-range..range);
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for h in 0..spec.hidden_width {
        neurons.push(format!(
            r#"{{"type":"hidden","uuid":"h{h}","bias":{},"squash":"{}"}}"#,
            draw(),
            spec.hidden_squash
        ));
        for i in 0..inputs {
            synapses.push(format!(
                r#"{{"fromUUID":"input-{i}","toUUID":"h{h}","weight":{}}}"#,
                draw()
            ));
        }
    }
    neurons.push(format!(
        r#"{{"type":"output","uuid":"o0","bias":{},"squash":"IDENTITY"}}"#,
        draw()
    ));
    for h in 0..spec.hidden_width {
        synapses.push(format!(
            r#"{{"fromUUID":"h{h}","toUUID":"o0","weight":{}}}"#,
            draw()
        ));
    }
    let json = format!(
        r#"{{"semanticVersion":"4.0.0","forwardOnly":true,"input":{inputs},"output":1,"neurons":[{}],"synapses":[{}]}}"#,
        neurons.join(","),
        synapses.join(",")
    );
    parse_creature_json(&json).map_err(|e| e.to_string())
}
