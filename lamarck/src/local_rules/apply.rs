//! Applying a rule set to a creature, and measuring a creature's error.
//!
//! Every arm clones its parent: nothing here mutates the parent.
//!
//! Cost unit: one forward pass over one record is 1 unit; a forward pass plus
//! one update (local rule or backprop) is 2 units. Held-out measurement is never
//! charged.

use std::ops::Add;

use neat_core::{CompiledNetwork, CreatureExport, compile_creature};
use serde::Serialize;

use super::rule::{ActivationFamily, LocalObservation, RuleSet, local_residual, rule_delta};
use super::tasks::TaskRecord;
use crate::propagate_layout::PropagateLayout;

/// Record-unit cost of an arm (see the module docs for the unit definition).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cost {
    /// Forward passes plus updates, each counted as above.
    pub record_units: u64,
}

impl Add for Cost {
    type Output = Cost;

    fn add(self, other: Cost) -> Cost {
        Cost {
            record_units: self.record_units + other.record_units,
        }
    }
}

pub(super) fn compile(creature: &CreatureExport) -> Result<CompiledNetwork, String> {
    compile_creature(creature).map_err(|e| e.to_string())
}

fn check_record(creature: &CreatureExport, record: &TaskRecord) -> Result<(), String> {
    if record.inputs.len() != creature.input || record.expected.len() != creature.output {
        return Err(format!(
            "record shape {}/{} does not match creature {}/{}",
            record.inputs.len(),
            record.expected.len(),
            creature.input,
            creature.output
        ));
    }
    Ok(())
}

fn squared_error(out: &[f32], expected: &[f32]) -> f64 {
    let n = expected.len().max(1);
    out.iter()
        .zip(expected)
        .map(|(o, e)| {
            let d = f64::from(*o) - f64::from(*e);
            d * d
        })
        .sum::<f64>()
        / n as f64
}

/// Mean over records of the per-record mean squared error.
pub fn mse(creature: &CreatureExport, records: &[TaskRecord]) -> Result<f64, String> {
    if records.is_empty() {
        return Err("mse needs at least one record".to_string());
    }
    let mut network = compile(creature)?;
    let mut total = 0.0;
    for record in records {
        check_record(creature, record)?;
        let out = network.activate_and_trace(&record.inputs, creature.output);
        total += squared_error(&out, &record.expected);
    }
    Ok(total / records.len() as f64)
}

/// Sum the rule updates of every synapse and bias over `records`.
///
/// Returns `(per-synapse sums, per-neuron bias sums)`. Only hidden and output
/// neurons take bias updates; constant neurons keep a zero sum.
fn accumulate(
    creature: &CreatureExport,
    layout: &PropagateLayout,
    records: &[TaskRecord],
    rules: &RuleSet,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let input = creature.input;
    // neat-core reads the output neurons as the last `creature.output` entries.
    let first_output = (input + creature.neurons.len()).saturating_sub(creature.output);
    let families: Vec<ActivationFamily> = creature
        .neurons
        .iter()
        .map(|n| ActivationFamily::from_squash(n.squash.as_deref()))
        .collect();
    let mut synapse_sum = vec![0.0; layout.synapse_templates.len()];
    let mut bias_sum = vec![0.0; creature.neurons.len()];
    let mut network = compile(creature)?;
    for record in records {
        check_record(creature, record)?;
        network.activate_and_trace(&record.inputs, creature.output);
        let acts: Vec<f64> = network
            .activations()
            .iter()
            .map(|a| f64::from(*a))
            .collect();
        let residual = |prop: usize| {
            prop.checked_sub(first_output)
                .and_then(|slot| local_residual(true, slot, acts[prop], &record.expected))
        };
        for (s, template) in layout.synapse_templates.iter().enumerate() {
            let (from, to) = (template.from as usize, template.to as usize);
            let Some(family) = to.checked_sub(input).and_then(|i| families.get(i)) else {
                continue;
            };
            let obs = LocalObservation {
                x: acts[from],
                y: acts[to],
                w: creature.synapses[s].weight,
                r: residual(to),
            };
            synapse_sum[s] += rule_delta(rules, *family, &obs);
        }
        for (n, neuron) in creature.neurons.iter().enumerate() {
            if !matches!(neuron.neuron_type.as_str(), "hidden" | "output") {
                continue;
            }
            let prop = input + n;
            let obs = LocalObservation {
                x: 1.0,
                y: acts[prop],
                w: neuron.bias,
                r: residual(prop),
            };
            bias_sum[n] += rule_delta(rules, families[n], &obs);
        }
    }
    Ok((synapse_sum, bias_sum))
}

/// Apply `rules` to a clone of `parent` for `steps` steps.
///
/// Each step averages every synapse and bias update over `records`, clamps the
/// mean to `[-max_step, max_step]` and moves the parameter by `eta` times it.
pub fn apply_rule_steps(
    parent: &CreatureExport,
    records: &[TaskRecord],
    rules: &RuleSet,
    steps: u32,
    eta: f64,
    max_step: f64,
) -> Result<(CreatureExport, Cost), String> {
    if records.is_empty() {
        return Err("apply_rule_steps needs at least one record".to_string());
    }
    let mut creature = parent.clone();
    let mut cost = Cost::default();
    let n = records.len() as f64;
    for _ in 0..steps {
        let layout = PropagateLayout::from_creature(&creature)?;
        let (synapse_sum, bias_sum) = accumulate(&creature, &layout, records, rules)?;
        for (synapse, sum) in creature.synapses.iter_mut().zip(&synapse_sum) {
            synapse.weight += eta * (sum / n).clamp(-max_step, max_step);
        }
        for (neuron, sum) in creature.neurons.iter_mut().zip(&bias_sum) {
            neuron.bias += eta * (sum / n).clamp(-max_step, max_step);
        }
        // One forward pass and one local update per record.
        cost = cost
            + Cost {
                record_units: 2 * records.len() as u64,
            };
    }
    Ok((creature, cost))
}
