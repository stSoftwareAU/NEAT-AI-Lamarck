//! Build [`PropagateInput`] from a [`CreatureExport`] and fold reverse-topo
//! backprop into an export-indexed [`LearningSignal`].
//!
//! The reverse-topological loop itself lives in neat-core
//! ([`neat_core::propagate_topological_loop`]); this module is the Lamarck
//! creature/training-data bridge (issue #2).

use crate::backprop::{BackpropConfig, LearningSignal};
use neat_core::{
    CompiledNetwork, CreatureExport, NEURON_TYPE_CONSTANT, NEURON_TYPE_HIDDEN, NEURON_TYPE_INPUT,
    NEURON_TYPE_OUTPUT, NeuronInput, PropagateInput, SquashType, SynapseInput, TrainingDataConfig,
    TrainingDataIterator, apply_get_range, parse_squash_name, propagate_topological_loop,
};
use rand::Rng;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

/// Topology caches reused across training records for one creature.
#[derive(Debug, Clone)]
pub struct PropagateLayout {
    /// Virtual inputs + export neurons.
    pub neuron_count: usize,
    /// `creature.input`.
    pub input_count: usize,
    /// `creature.output`.
    pub output_count: usize,
    /// Squash / type / range templates (activations filled per record).
    pub neuron_templates: Vec<NeuronTemplate>,
    /// Synapses in export order.
    pub synapse_templates: Vec<SynapseTemplate>,
    /// Inward adjacency starts (per propagate neuron index).
    pub inward_starts: Vec<u32>,
    /// Inward adjacency counts.
    pub inward_counts: Vec<u32>,
    /// Flat inward synapse indices (export synapse ids).
    pub inward_synapse_indices: Vec<u32>,
    /// Reverse topological order (non-input indices, outputs first for forward-only).
    pub reverse_topo_order: Vec<u32>,
    /// UUID → propagate neuron index.
    pub uuid_to_prop: HashMap<String, usize>,
}

/// Static per-neuron fields (activation filled per record).
#[derive(Debug, Clone)]
pub struct NeuronTemplate {
    /// Squash discriminant.
    pub squash_type: u8,
    /// Neuron category code.
    pub neuron_type: u8,
    /// Bias snapshot from the creature (inputs use 0).
    pub bias: f64,
    /// Range low for target clamping.
    pub range_low: f32,
    /// Range high for target clamping.
    pub range_high: f32,
}

/// Static per-synapse fields (weights filled per record from creature).
#[derive(Debug, Clone)]
pub struct SynapseTemplate {
    /// Source propagate index.
    pub from: u32,
    /// Destination propagate index.
    pub to: u32,
    /// Export weight.
    pub weight: f64,
}

/// Sparse selection flags for one accumulate pass.
#[derive(Debug, Clone)]
pub struct SparseSelection {
    /// Neurons that accumulate weight/bias updates (`updateNeeded`).
    pub update_needed: HashSet<usize>,
    /// Neurons that participate in error distribution (`propagateNeeded`).
    pub propagate_needed: HashSet<usize>,
}

impl PropagateLayout {
    /// Build layout caches from a creature export.
    pub fn from_creature(creature: &CreatureExport) -> Result<Self, String> {
        let input_count = creature.input;
        let output_count = creature.output;
        let neuron_count = input_count + creature.neurons.len();

        let mut uuid_to_prop: HashMap<String, usize> = HashMap::new();
        for i in 0..input_count {
            uuid_to_prop.insert(format!("input-{i}"), i);
        }
        for (i, n) in creature.neurons.iter().enumerate() {
            uuid_to_prop.insert(n.uuid.clone(), input_count + i);
        }

        let mut neuron_templates = Vec::with_capacity(neuron_count);
        for i in 0..input_count {
            let _ = i;
            let (lo, hi) = apply_get_range(SquashType::Identity);
            neuron_templates.push(NeuronTemplate {
                squash_type: SquashType::Identity as u8,
                neuron_type: NEURON_TYPE_INPUT,
                bias: 0.0,
                range_low: lo,
                range_high: hi,
            });
        }
        for n in &creature.neurons {
            let squash = match n.squash.as_deref() {
                None => SquashType::Identity,
                Some(name) => parse_squash_name(name).map_err(|e| e.to_string())?,
            };
            let (lo, hi) = apply_get_range(squash);
            let neuron_type = match n.neuron_type.as_str() {
                "output" => NEURON_TYPE_OUTPUT,
                "constant" => NEURON_TYPE_CONSTANT,
                _ => NEURON_TYPE_HIDDEN,
            };
            neuron_templates.push(NeuronTemplate {
                squash_type: squash as u8,
                neuron_type,
                bias: n.bias,
                range_low: lo,
                range_high: hi,
            });
        }

        let mut synapse_templates = Vec::with_capacity(creature.synapses.len());
        for s in &creature.synapses {
            let from = *uuid_to_prop
                .get(&s.from_uuid)
                .ok_or_else(|| format!("unknown synapse from {}", s.from_uuid))?;
            let to = *uuid_to_prop
                .get(&s.to_uuid)
                .ok_or_else(|| format!("unknown synapse to {}", s.to_uuid))?;
            synapse_templates.push(SynapseTemplate {
                from: from as u32,
                to: to as u32,
                weight: s.weight,
            });
        }

        let mut inward_lists: Vec<Vec<u32>> = vec![Vec::new(); neuron_count];
        for (syn_idx, syn) in synapse_templates.iter().enumerate() {
            inward_lists[syn.to as usize].push(syn_idx as u32);
        }
        let mut inward_starts = Vec::with_capacity(neuron_count);
        let mut inward_counts = Vec::with_capacity(neuron_count);
        let mut inward_synapse_indices = Vec::new();
        for list in &inward_lists {
            inward_starts.push(inward_synapse_indices.len() as u32);
            inward_counts.push(list.len() as u32);
            inward_synapse_indices.extend_from_slice(list);
        }

        // Forward-only creatures keep export order ≈ activation order; reverse
        // non-input indices so outputs (at the end) are visited first.
        let reverse_topo_order: Vec<u32> = (input_count..neuron_count)
            .rev()
            .map(|i| i as u32)
            .collect();

        Ok(Self {
            neuron_count,
            input_count,
            output_count,
            neuron_templates,
            synapse_templates,
            inward_starts,
            inward_counts,
            inward_synapse_indices,
            reverse_topo_order,
            uuid_to_prop,
        })
    }
}

/// Select sparse neurons with injectable RNG (TS `SparseConfig` / `chooseNeurons`).
///
/// When `sparse_ratio >= 1.0`, every hidden/output neuron is selected.
pub fn select_sparse(
    creature: &CreatureExport,
    layout: &PropagateLayout,
    config: &BackpropConfig,
    rng: &mut impl Rng,
) -> SparseSelection {
    let eligible: Vec<usize> = creature
        .neurons
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type == "hidden" || n.neuron_type == "output")
        .map(|(i, _)| layout.input_count + i)
        .collect();

    let mut update_needed = HashSet::new();
    if config.sparse_ratio >= 1.0 - f64::EPSILON || eligible.is_empty() {
        update_needed.extend(eligible.iter().copied());
    } else {
        let target = ((eligible.len() as f64) * config.sparse_ratio.clamp(0.0, 1.0))
            .ceil()
            .max(1.0) as usize;
        let mut pool = eligible.clone();
        // Fisher–Yates with injected RNG.
        for i in (1..pool.len()).rev() {
            let j = rng.random_range(0..=i);
            pool.swap(i, j);
        }
        update_needed.extend(pool.into_iter().take(target));
    }

    // Paths-to-output: BFS forward from selected neurons along outgoing edges.
    let mut outgoing: HashMap<usize, Vec<usize>> = HashMap::new();
    for syn in &layout.synapse_templates {
        outgoing
            .entry(syn.from as usize)
            .or_default()
            .push(syn.to as usize);
    }
    let mut propagate_needed = update_needed.clone();
    let mut queue: VecDeque<usize> = update_needed.iter().copied().collect();
    while let Some(cur) = queue.pop_front() {
        if let Some(nexts) = outgoing.get(&cur) {
            for &n in nexts {
                if propagate_needed.insert(n) {
                    queue.push_back(n);
                }
            }
        }
    }

    SparseSelection {
        update_needed,
        propagate_needed,
    }
}

/// Accumulate a full-creature [`LearningSignal`] over training records.
///
/// Analyse-without-apply: the creature and network weights are not mutated.
pub fn accumulate_creature_learning(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    config: &BackpropConfig,
    max_records: Option<u64>,
    rng: &mut impl Rng,
) -> Result<LearningSignal, String> {
    let layout = PropagateLayout::from_creature(creature)?;
    let sparse = select_sparse(creature, &layout, config, rng);
    let mut learning = LearningSignal::new(creature.neurons.len(), creature.synapses.len());

    let td_cfg = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = TrainingDataIterator::new(training_data, td_cfg).map_err(|e| e.to_string())?;
    let mut count = 0u64;

    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if let Some(limit) = max_records
            && count >= limit
        {
            break;
        }
        let _traced = network.activate_and_trace(&record.inputs, creature.output);
        count += 1;

        let mut neurons: Vec<NeuronInput> = Vec::with_capacity(layout.neuron_count);
        for (prop_idx, tmpl) in layout.neuron_templates.iter().enumerate() {
            let activation = network.activations.get(prop_idx).copied().unwrap_or(0.0);
            let hint = if prop_idx < layout.input_count {
                activation
            } else {
                let rel = prop_idx - layout.input_count;
                network
                    .hint_values_buffer
                    .get(rel)
                    .copied()
                    .unwrap_or(activation)
            };
            neurons.push(NeuronInput {
                squash_type: tmpl.squash_type,
                neuron_type: tmpl.neuron_type,
                propagate_needed: sparse.propagate_needed.contains(&prop_idx),
                update_needed: sparse.update_needed.contains(&prop_idx),
                hint_value: hint,
                range_low: tmpl.range_low,
                range_high: tmpl.range_high,
                adjusted_activation: activation,
                adjusted_bias: tmpl.bias as f32,
            });
        }

        let synapses: Vec<SynapseInput> = layout
            .synapse_templates
            .iter()
            .map(|s| SynapseInput {
                from: s.from,
                to: s.to,
                original_weight: s.weight as f32,
                adjusted_weight: s.weight as f32,
                is_self_loop: s.from == s.to,
            })
            .collect();

        let expected = record.outputs.to_vec();
        let input = PropagateInput {
            neurons: &neurons,
            synapses: &synapses,
            inward_starts: &layout.inward_starts,
            inward_counts: &layout.inward_counts,
            inward_synapse_indices: &layout.inward_synapse_indices,
            reverse_topo_order: &layout.reverse_topo_order,
            expected: &expected,
            input_count: layout.input_count as u32,
            output_count: layout.output_count as u32,
            plank_constant: config.plank_constant as f32,
            normalise_gradients: config.normalise_gradients,
        };

        let output = propagate_topological_loop(&input);
        learning.accumulate_propagate_output(&output, layout.input_count);
    }

    Ok(learning)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backprop::{apply_learnings, calculate_learning_rate};
    use neat_core::{compile_creature, parse_creature_json};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::io::Write;
    use tempfile::tempdir;

    const TINY_CHAIN: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 1,
      "output": 1,
      "neurons": [
        {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0}
      ]
    }"#;

    fn write_records(dir: &Path, pairs: &[(f32, f32)]) {
        let mut f = std::fs::File::create(dir.join("0.bin")).unwrap();
        for &(x, y) in pairs {
            f.write_all(&x.to_le_bytes()).unwrap();
            f.write_all(&y.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn hidden_neuron_receives_propagated_blame() {
        let dir = tempdir().unwrap();
        // activation path: input=1 → h1=1 → o1=1; target=2 ⇒ error at output.
        write_records(dir.path(), &[(1.0, 2.0)]);
        let creature = parse_creature_json(TINY_CHAIN).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let cfg = BackpropConfig::default();
        let mut rng = StdRng::seed_from_u64(1);
        let learning = accumulate_creature_learning(
            &creature,
            &mut network,
            dir.path(),
            &cfg,
            Some(1),
            &mut rng,
        )
        .unwrap();
        // export index 0 = h1, 1 = o1
        assert!(
            learning.biases[0].count > 0.0,
            "hidden must receive propagated bias signal"
        );
        assert!(
            learning.biases[1].count > 0.0,
            "output must accumulate bias signal"
        );
        assert!(learning.weights.iter().any(|w| w.count > 0.0));
    }

    #[test]
    fn sparse_ratio_one_selects_all_eligible() {
        let creature = parse_creature_json(TINY_CHAIN).unwrap();
        let layout = PropagateLayout::from_creature(&creature).unwrap();
        let cfg = BackpropConfig::default();
        let mut rng = StdRng::seed_from_u64(2);
        let sparse = select_sparse(&creature, &layout, &cfg, &mut rng);
        assert!(sparse.update_needed.contains(&1)); // h1
        assert!(sparse.update_needed.contains(&2)); // o1
        assert!(!sparse.update_needed.contains(&0)); // input
    }

    #[test]
    fn apply_from_accumulated_signal_moves_output_bias() {
        let dir = tempdir().unwrap();
        write_records(dir.path(), &[(1.0, 2.0)]);
        let creature = parse_creature_json(TINY_CHAIN).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let cfg = BackpropConfig::default();
        let mut rng = StdRng::seed_from_u64(3);
        let learning = accumulate_creature_learning(
            &creature,
            &mut network,
            dir.path(),
            &cfg,
            Some(1),
            &mut rng,
        )
        .unwrap();
        let lr = calculate_learning_rate(&cfg, 0, None);
        let applied = apply_learnings(&creature, &learning, &cfg, lr);
        assert_ne!(applied.neurons[1].bias, creature.neurons[1].bias);
    }
}
