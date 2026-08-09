//! Focus-neuron structural mutations: smart upstream synapses and neuron growth.

use crate::focus::{FocusNeuronStats, IncomingSourceStats, neuron_index};
use crate::observations::ObservationsStatistics;
use neat_core::{
    CompiledNetwork, CreatureExport, NeuronExport, SynapseExport, TrainingDataConfig,
    TrainingDataIterator,
};
use rand::Rng;
use std::path::Path;

/// Target |Δpre| ≈ this when the source sits at one standard deviation.
const TARGET_PRE_DELTA: f64 = 1e-3;
/// Hard cap on a newly added synapse weight (sparse/low-std OLS can explode).
const MAX_NEW_WEIGHT: f64 = 0.08;
/// Apply this fraction of the residual OLS coefficient (full OLS overshoots).
pub const OLS_WEIGHT_FRACTION: f64 = 0.05;
/// How many target-corr shortlist inputs to re-rank by residual correlation.
const RESIDUAL_SHORTLIST: usize = 48;

/// Scored candidate source for a new edge into the focus.
#[derive(Debug, Clone)]
pub struct RankedSource {
    /// Source UUID (`input-N` or neuron uuid).
    pub from_uuid: String,
    /// Ranking score (higher is better); typically `|correlation|`.
    pub score: f64,
    /// Signed correlation / direction hint used for weight sign.
    pub direction: f64,
    /// Suggested absolute weight scale before sign (fallback when OLS unknown).
    pub weight_scale: f64,
    /// Residual OLS coefficient `cov(x,e) / var(x)` when measured.
    pub ols_weight: Option<f64>,
}

/// Compiled-network index for inputs and exported neurons.
pub fn compiled_index(creature: &CreatureExport, uuid: &str) -> Option<usize> {
    if let Some(i) = uuid
        .strip_prefix("input-")
        .and_then(|s| s.parse::<usize>().ok())
    {
        return (i < creature.input).then_some(i);
    }
    neuron_index(creature, uuid)
}

/// Whether `from -> to` is legal under the creature's `forwardOnly` flag.
pub fn is_forward_edge(creature: &CreatureExport, from_uuid: &str, to_uuid: &str) -> bool {
    let Some(from_idx) = compiled_index(creature, from_uuid) else {
        return false;
    };
    let Some(to_idx) = compiled_index(creature, to_uuid) else {
        return false;
    };
    if creature.forward_only {
        from_idx < to_idx
    } else {
        from_uuid != to_uuid
    }
}

/// Output index of `focus_uuid` among outputs, when the focus is an output neuron.
pub fn focus_output_index(creature: &CreatureExport, focus_uuid: &str) -> Option<usize> {
    let neuron = creature.neurons.iter().find(|n| n.uuid == focus_uuid)?;
    if neuron.neuron_type != "output" {
        return None;
    }
    creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output")
        .position(|n| n.uuid == focus_uuid)
}

/// Insertion index for a new hidden neuron that must feed `focus_uuid`.
///
/// Hidden targets: insert immediately before the focus. Output targets: insert
/// before the first output so outputs stay contiguous at the end.
pub fn insert_index_for_hidden(creature: &CreatureExport, focus_uuid: &str) -> Option<usize> {
    let focus = creature.neurons.iter().find(|n| n.uuid == focus_uuid)?;
    let first_output = creature
        .neurons
        .iter()
        .position(|n| n.neuron_type == "output");
    if focus.neuron_type == "output" {
        first_output.or(Some(creature.neurons.len()))
    } else {
        creature.neurons.iter().position(|n| n.uuid == focus_uuid)
    }
}

fn existing_sources_into<'a>(creature: &'a CreatureExport, focus_uuid: &str) -> std::collections::BTreeSet<&'a str> {
    creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == focus_uuid)
        .map(|s| s.from_uuid.as_str())
        .collect()
}

/// Rank unused, forward-legal sources that could connect into the focus.
///
/// Prefer unused raw inputs scored by `|input↔target|` correlation when the
/// focus is an output; otherwise fall back to source scale. Hidden/constant
/// sources without a residual correlation use a weak std-based prior.
pub fn rank_unused_sources(
    creature: &CreatureExport,
    focus_uuid: &str,
    observations: &ObservationsStatistics,
) -> Vec<RankedSource> {
    let existing = existing_sources_into(creature, focus_uuid);
    let out_idx = focus_output_index(creature, focus_uuid);
    let n_out = observations.output_count.max(creature.output);

    let mut ranked = Vec::new();

    for i in 0..creature.input {
        let uuid = format!("input-{i}");
        if existing.contains(uuid.as_str()) || !is_forward_edge(creature, &uuid, focus_uuid) {
            continue;
        }
        let std_dev = observations
            .inputs
            .get(i)
            .map(|s| s.std_dev)
            .unwrap_or(1.0)
            .max(1e-3);
        let corr = out_idx.and_then(|j| {
            let flat = i * n_out + j;
            observations.input_target_correlations.get(flat).copied()
        });
        let direction = corr.unwrap_or(0.0);
        let score = corr.map(|c| c.abs()).unwrap_or_else(|| {
            // Weak prior: prefer informative (non-constant) inputs.
            (std_dev / (1.0 + std_dev)).clamp(0.0, 1.0) * 0.01
        });
        ranked.push(RankedSource {
            from_uuid: uuid,
            score,
            direction,
            // Size weight so |w| * std ≈ TARGET_PRE_DELTA.
            weight_scale: TARGET_PRE_DELTA / std_dev,
            ols_weight: None,
        });
    }

    for n in &creature.neurons {
        if n.uuid == focus_uuid || n.neuron_type == "output" {
            continue;
        }
        if existing.contains(n.uuid.as_str()) || !is_forward_edge(creature, &n.uuid, focus_uuid) {
            continue;
        }
        // No residual correlation for unconnected hiddens yet — keep a tiny
        // exploratory score so they remain eligible after inputs.
        ranked.push(RankedSource {
            from_uuid: n.uuid.clone(),
            score: 1e-4,
            direction: 0.0,
            weight_scale: TARGET_PRE_DELTA,
            ols_weight: None,
        });
    }

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.from_uuid.cmp(&b.from_uuid))
    });
    ranked
}

/// Pick the best unused source, with light stochastic exploration among top ranks.
pub fn pick_smart_source<'a>(
    ranked: &'a [RankedSource],
    rng: &mut impl Rng,
) -> Option<&'a RankedSource> {
    if ranked.is_empty() {
        return None;
    }
    let top_score = ranked[0].score;
    let threshold = (top_score * 0.85).max(top_score - 0.05);
    let elite: Vec<&RankedSource> = ranked
        .iter()
        .take_while(|r| r.score + f64::EPSILON >= threshold)
        .collect();
    Some(elite[rng.random_range(0..elite.len())])
}

/// Suggested synapse weight from a ranked source and focus residual sign.
///
/// Prefer a fraction of the residual OLS coefficient when available; otherwise
/// fall back to a small std-scaled step with correlation sign.
pub fn suggested_weight(source: &RankedSource, focus_stats: &FocusNeuronStats) -> f64 {
    suggested_weight_scaled(source, focus_stats, OLS_WEIGHT_FRACTION)
}

/// Like [`suggested_weight`] but with an explicit OLS-fraction / scale factor.
pub fn suggested_weight_scaled(
    source: &RankedSource,
    focus_stats: &FocusNeuronStats,
    scale: f64,
) -> f64 {
    if let Some(ols) = source.ols_weight.filter(|w| w.is_finite() && w.abs() > 1e-12) {
        return (ols * scale).clamp(-MAX_NEW_WEIGHT, MAX_NEW_WEIGHT);
    }
    let mag = (source.weight_scale * scale / OLS_WEIGHT_FRACTION).clamp(1e-5, MAX_NEW_WEIGHT);
    let dir = if source.direction.abs() > 1e-3 {
        source.direction.signum()
    } else if let Some(err) = focus_stats
        .mean_adjusted_error
        .or(focus_stats.mean_error)
        .filter(|e| e.abs() > 1e-9)
    {
        err.signum()
    } else {
        1.0
    };
    dir * mag
}

fn pearson(n: f64, sum_x: f64, sum_y: f64, sum_xx: f64, sum_yy: f64, sum_xy: f64) -> f64 {
    if n < 2.0 {
        return 0.0;
    }
    let cov = sum_xy - (sum_x * sum_y) / n;
    let vx = sum_xx - (sum_x * sum_x) / n;
    let vy = sum_yy - (sum_y * sum_y) / n;
    if vx <= 0.0 || vy <= 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// Re-rank a shortlist of unused inputs by Pearson(input, focus residual).
///
/// Target correlation alone double-counts signal the net already captured; residual
/// correlation is the structural prior that can still cut focus error.
pub fn refine_sources_by_residual(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    focus_uuid: &str,
    prior: &[RankedSource],
    max_records: Option<u64>,
) -> Result<Vec<RankedSource>, String> {
    let Some(out_idx) = focus_output_index(creature, focus_uuid) else {
        return Ok(prior.to_vec());
    };
    let relative_idx = neuron_index(creature, focus_uuid)
        .and_then(|i| i.checked_sub(creature.input))
        .ok_or_else(|| format!("focus neuron {focus_uuid} missing compiled index"))?;

    let mut shortlist: Vec<RankedSource> = prior
        .iter()
        .filter(|s| s.from_uuid.starts_with("input-"))
        .take(RESIDUAL_SHORTLIST)
        .cloned()
        .collect();
    if shortlist.is_empty() {
        return Ok(prior.to_vec());
    }

    let indices: Vec<usize> = shortlist
        .iter()
        .filter_map(|s| {
            s.from_uuid
                .strip_prefix("input-")
                .and_then(|x| x.parse::<usize>().ok())
        })
        .collect();
    if indices.len() != shortlist.len() {
        return Ok(prior.to_vec());
    }

    let k = indices.len();
    let mut sum_x = vec![0.0f64; k];
    let mut sum_xx = vec![0.0f64; k];
    let mut sum_xy = vec![0.0f64; k];
    let mut sum_e = 0.0f64;
    let mut sum_ee = 0.0f64;
    let mut n = 0.0f64;

    let config = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = TrainingDataIterator::new(training_data, config).map_err(|e| e.to_string())?;
    let mut count = 0u64;
    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if let Some(limit) = max_records
            && count >= limit
        {
            break;
        }
        let traced = network.activate_and_trace(&record.inputs, creature.output);
        let post_offset = creature.output;
        if post_offset + relative_idx >= traced.len() || out_idx >= record.outputs.len() {
            continue;
        }
        let post = f64::from(traced[post_offset + relative_idx]);
        let err = f64::from(record.outputs[out_idx]) - post;
        count += 1;
        n += 1.0;
        sum_e += err;
        sum_ee += err * err;
        for (j, &idx) in indices.iter().enumerate() {
            let x = f64::from(*record.inputs.get(idx).unwrap_or(&0.0));
            sum_x[j] += x;
            sum_xx[j] += x * x;
            sum_xy[j] += x * err;
        }
    }

    for (j, src) in shortlist.iter_mut().enumerate() {
        let corr = pearson(n, sum_x[j], sum_e, sum_xx[j], sum_ee, sum_xy[j]);
        src.direction = corr;
        src.score = corr.abs();
        // Univariate OLS of residual on source: cov(x,e) / var(x).
        if n >= 2.0 {
            let cov = sum_xy[j] - (sum_x[j] * sum_e) / n;
            let var_x = sum_xx[j] - (sum_x[j] * sum_x[j]) / n;
            if var_x > 1e-12 {
                src.ols_weight = Some(cov / var_x);
            }
        }
    }
    shortlist.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.from_uuid.cmp(&b.from_uuid))
    });

    // Append any non-input priors (hidden exploratory) after residual-ranked inputs.
    let mut out = shortlist;
    for src in prior {
        if !src.from_uuid.starts_with("input-") {
            out.push(src.clone());
        }
    }
    Ok(out)
}

/// Deterministic UUID v4 from the RNG (seeded runs stay reproducible).
pub fn random_uuid_v4(rng: &mut impl Rng) -> String {
    let mut b = [0u8; 16];
    for byte in &mut b {
        *byte = rng.random();
    }
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    )
}

/// Add a direct synapse `from -> focus` with the given weight.
pub fn add_synapse(
    creature: &mut CreatureExport,
    from_uuid: String,
    focus_uuid: &str,
    weight: f64,
) {
    creature.synapses.push(SynapseExport {
        from_uuid,
        to_uuid: focus_uuid.to_string(),
        weight,
        synapse_type: None,
    });
}

/// Insert a hidden neuron on a path `from -> new -> focus`.
///
/// Returns the new neuron UUID. Fails closed when insertion would break
/// forward-only ordering or the focus cannot host a hidden predecessor.
pub fn add_neuron_bridge(
    creature: &mut CreatureExport,
    from_uuid: &str,
    focus_uuid: &str,
    new_uuid: String,
    squash: &str,
    bias: f64,
    w_in: f64,
    w_out: f64,
) -> Result<String, String> {
    if !is_forward_edge(creature, from_uuid, focus_uuid) {
        return Err(format!(
            "bridge {from_uuid} -> {focus_uuid} is not forward-legal before insert"
        ));
    }
    let insert_at = insert_index_for_hidden(creature, focus_uuid)
        .ok_or_else(|| format!("cannot locate insert index for focus {focus_uuid}"))?;

    creature.neurons.insert(
        insert_at,
        NeuronExport {
            neuron_type: "hidden".into(),
            uuid: new_uuid.clone(),
            bias,
            squash: Some(squash.to_string()),
        },
    );

    // After insert, verify both edges remain forward-legal.
    if !is_forward_edge(creature, from_uuid, &new_uuid)
        || !is_forward_edge(creature, &new_uuid, focus_uuid)
    {
        // Roll back neuron insert; synapses not yet added.
        creature.neurons.remove(insert_at);
        return Err(format!(
            "bridge {from_uuid} -> {new_uuid} -> {focus_uuid} violates forward-only after insert"
        ));
    }

    creature.synapses.push(SynapseExport {
        from_uuid: from_uuid.to_string(),
        to_uuid: new_uuid.clone(),
        weight: w_in,
        synapse_type: None,
    });
    creature.synapses.push(SynapseExport {
        from_uuid: new_uuid.clone(),
        to_uuid: focus_uuid.to_string(),
        weight: w_out,
        synapse_type: None,
    });
    Ok(new_uuid)
}

/// Split an existing incoming synapse through a new hidden neuron.
///
/// Removes the direct `from -> focus` edge and replaces it with
/// `from -> new (w=1)` and `new -> focus (w=old)`.
pub fn split_incoming_synapse(
    creature: &mut CreatureExport,
    incoming: &IncomingSourceStats,
    focus_uuid: &str,
    new_uuid: String,
    squash: &str,
) -> Result<String, String> {
    let syn_idx = incoming.synapse_index;
    if syn_idx >= creature.synapses.len() {
        return Err("incoming synapse index out of range".into());
    }
    let old = creature.synapses[syn_idx].clone();
    if old.to_uuid != focus_uuid || old.from_uuid != incoming.from_uuid {
        return Err("incoming synapse does not match focus/source".into());
    }
    let old_w = old.weight;
    creature.synapses.remove(syn_idx);
    add_neuron_bridge(
        creature,
        &incoming.from_uuid,
        focus_uuid,
        new_uuid,
        squash,
        0.0,
        1.0,
        old_w,
    )
}

/// Squashes tried when growing a hidden into the focus.
///
/// Different residual shapes need different nonlinearities (e.g. ABSOLUTE for
/// sign-insensitive error, ReLU for one-sided corrections, TANH for bounded).
pub const NEURON_GROWTH_SQUASHES: &[&str] = &[
    "TANH",
    "ReLU",
    "ABSOLUTE",
    "IDENTITY",
    "LeakyReLU",
    "Softplus",
    "GELU",
    "HARD_TANH",
];

/// Default squash when a single bridge is requested (first growth candidate).
pub fn bridge_squash(_creature: &CreatureExport, _focus_uuid: &str) -> &'static str {
    NEURON_GROWTH_SQUASHES[0]
}

/// Pick a growth squash by index (wraps) — used to emit multi-squash batches.
pub fn growth_squash_at(index: usize) -> &'static str {
    NEURON_GROWTH_SQUASHES[index % NEURON_GROWTH_SQUASHES.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;
    use rand::{SeedableRng, rngs::StdRng};

    const TINY: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 2,
      "output": 1,
      "neurons": [
        {"type":"hidden","uuid":"h1","bias":0.1,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0}
      ]
    }"#;

    fn obs_with_corr(c0: f64, c1: f64) -> ObservationsStatistics {
        ObservationsStatistics {
            format_version: "1.0.0".into(),
            algorithm_version: "1.0.0".into(),
            mode: crate::observations::StatsMode::Full,
            sample_record_limit: None,
            input_count: 2,
            output_count: 1,
            record_count: 10,
            corpus_identity: "x".into(),
            created_at_unix: 0,
            inputs: vec![
                crate::observations::ScalarStats {
                    count: 10,
                    mean: 0.0,
                    variance: 1.0,
                    std_dev: 1.0,
                    min: -1.0,
                    max: 1.0,
                    zero_count: 0,
                    non_zero_count: 10,
                    non_finite_count: 0,
                    mean_abs: 0.5,
                    rms: 1.0,
                    quantiles: [0.0; 7],
                },
                crate::observations::ScalarStats {
                    count: 10,
                    mean: 0.0,
                    variance: 1.0,
                    std_dev: 1.0,
                    min: -1.0,
                    max: 1.0,
                    zero_count: 0,
                    non_zero_count: 10,
                    non_finite_count: 0,
                    mean_abs: 0.5,
                    rms: 1.0,
                    quantiles: [0.0; 7],
                },
            ],
            outputs: vec![],
            input_correlations: vec![],
            input_target_correlations: vec![c0, c1],
        }
    }

    #[test]
    fn ranks_unused_inputs_by_target_correlation() {
        let creature = parse_creature_json(TINY).unwrap();
        let obs = obs_with_corr(0.1, 0.9);
        let ranked = rank_unused_sources(&creature, "o1", &obs);
        assert_eq!(ranked[0].from_uuid, "input-1");
        assert!((ranked[0].score - 0.9).abs() < 1e-12);
        // input-0 feeds h1 only; still unused into o1.
        assert!(ranked.iter().any(|r| r.from_uuid == "input-0"));
    }

    #[test]
    fn forward_only_rejects_later_hidden_into_earlier() {
        let creature = parse_creature_json(TINY).unwrap();
        assert!(!is_forward_edge(&creature, "o1", "h1"));
        assert!(is_forward_edge(&creature, "input-1", "o1"));
    }

    #[test]
    fn add_neuron_bridge_inserts_before_outputs_and_compiles() {
        use neat_core::compile_creature;
        let mut creature = parse_creature_json(TINY).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let uuid = random_uuid_v4(&mut rng);
        add_neuron_bridge(
            &mut creature,
            "input-1",
            "o1",
            uuid.clone(),
            "TANH",
            0.0,
            0.001,
            0.05,
        )
        .unwrap();
        assert!(creature.neurons.iter().any(|n| n.uuid == uuid));
        let pos_new = creature.neurons.iter().position(|n| n.uuid == uuid).unwrap();
        let pos_out = creature.neurons.iter().position(|n| n.uuid == "o1").unwrap();
        assert!(pos_new < pos_out);
        compile_creature(&creature).expect("bridged creature must compile");
    }
}
