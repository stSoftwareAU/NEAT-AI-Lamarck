//! Focus-neuron structural mutations: smart upstream synapses and neuron growth.

use crate::focus::{FocusNeuronStats, IncomingSourceStats, neuron_index};
use crate::observations::ObservationsStatistics;
use neat_core::{
    CompiledNetwork, CreatureExport, NeuronExport, SynapseExport, TrainingDataConfig,
    TrainingRecord,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::path::Path;

/// Target |Δpre| ≈ this when the source sits at one standard deviation.
const TARGET_PRE_DELTA: f64 = 1e-3;
/// Hard cap on a newly added synapse weight (sparse/low-std OLS can explode).
const MAX_NEW_WEIGHT: f64 = 0.08;
/// Apply this fraction of the residual OLS coefficient (full OLS overshoots).
pub const OLS_WEIGHT_FRACTION: f64 = 0.05;
/// How many target-corr shortlist sources to re-rank by residual correlation.
const RESIDUAL_SHORTLIST: usize = 48;
/// Extra unused hiddens always included in residual refine (beyond the shortlist head).
const RESIDUAL_HIDDEN_EXTRA: usize = 16;
/// Floor std used when converting measured activation scale → weight_scale.
const MIN_ACT_STD: f64 = 1e-3;

/// One observation vector used to probe unused-source activations.
#[derive(Debug, Clone)]
pub struct ActivationProbe {
    /// Input observation values (`creature.input` long).
    pub inputs: Vec<f32>,
    /// Target outputs when the focus is an output neuron.
    pub outputs: Vec<f32>,
}

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

fn existing_sources_into<'a>(
    creature: &'a CreatureExport,
    focus_uuid: &str,
) -> std::collections::BTreeSet<&'a str> {
    creature
        .synapses
        .iter()
        .filter(|s| s.to_uuid == focus_uuid)
        .map(|s| s.from_uuid.as_str())
        .collect()
}

/// Whether `uuid` names a raw input (`input-N`).
pub fn is_input_source(uuid: &str) -> bool {
    uuid.starts_with("input-")
}

/// Rank unused, forward-legal sources that could connect into the focus.
///
/// Prefer unused raw inputs scored by `|input↔target|` correlation when the
/// focus is an output; otherwise fall back to source scale. Unused previous
/// hiddens are listed with `score = 0` until
/// [`refine_sources_from_probes`] / [`refine_sources_by_residual`] measures
/// their post-activation residual correlation (never a flat exploratory constant).
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
            .max(MIN_ACT_STD);
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
        // Identity only — activation probes fill score / std / OLS.
        ranked.push(RankedSource {
            from_uuid: n.uuid.clone(),
            score: 0.0,
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

/// Draw seeded synthetic observation probes from per-input / per-output stats.
///
/// Uses `mean ± z·std` (clamped to observed `[min, max]`) with small noise so
/// nonlinear squashes see a spread of values — not a single mean-forward pass.
pub fn synthetic_observation_probes(
    observations: &ObservationsStatistics,
    input_count: usize,
    output_count: usize,
    k: usize,
    rng: &mut impl Rng,
) -> Vec<ActivationProbe> {
    let mut probes = Vec::with_capacity(k);
    for i in 0..k {
        let mut inputs = Vec::with_capacity(input_count);
        for j in 0..input_count {
            let stats = observations.inputs.get(j);
            let mean = stats.map(|s| s.mean).unwrap_or(0.0);
            let std = stats.map(|s| s.std_dev).unwrap_or(1.0).max(MIN_ACT_STD);
            let lo = stats.map(|s| s.min).unwrap_or(-1.0);
            let hi = stats.map(|s| s.max).unwrap_or(1.0);
            // Cycle z ∈ {-1, 0, +1} then add small noise for joint diversity.
            let z = match i % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            };
            let noise: f64 = rng.random_range(-0.05..0.05);
            let mut v = mean + z * std + noise;
            if lo.is_finite() && hi.is_finite() && lo <= hi {
                v = v.clamp(lo, hi);
            }
            inputs.push(v as f32);
        }
        let mut outputs = Vec::with_capacity(output_count);
        for j in 0..output_count {
            let stats = observations.outputs.get(j);
            let mean = stats.map(|s| s.mean).unwrap_or(0.0);
            let std = stats.map(|s| s.std_dev).unwrap_or(1.0).max(MIN_ACT_STD);
            let lo = stats.map(|s| s.min).unwrap_or(-1.0);
            let hi = stats.map(|s| s.max).unwrap_or(1.0);
            let z = match (i + j) % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            };
            let noise: f64 = rng.random_range(-0.05..0.05);
            let mut v = mean + z * std + noise;
            if lo.is_finite() && hi.is_finite() && lo <= hi {
                v = v.clamp(lo, hi);
            }
            outputs.push(v as f32);
        }
        probes.push(ActivationProbe { inputs, outputs });
    }
    probes
}

/// Index of the first unused previous hidden in a ranked source list.
pub fn first_previous_hidden_index(ranked: &[RankedSource]) -> Option<usize> {
    ranked.iter().position(|s| !is_input_source(&s.from_uuid))
}

/// Reorder ranked sources so a previous hidden leads (for dedicated growth tries).
pub fn with_previous_hidden_first(ranked: &[RankedSource]) -> Option<Vec<RankedSource>> {
    let i = first_previous_hidden_index(ranked)?;
    if i == 0 {
        return Some(ranked.to_vec());
    }
    let mut out = Vec::with_capacity(ranked.len());
    out.push(ranked[i].clone());
    out.extend(
        ranked
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, s)| s.clone()),
    );
    Some(out)
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
    if let Some(ols) = source
        .ols_weight
        .filter(|w| w.is_finite() && w.abs() > 1e-12)
    {
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

/// How to read a candidate source activation for residual correlation.
#[derive(Clone, Copy)]
enum ResidualAct {
    /// Raw observation index.
    Input(usize),
    /// Index into `creature.neurons` (post-activation slot = `output + pos`).
    Neuron(usize),
}

fn residual_act_for(creature: &CreatureExport, from_uuid: &str) -> Option<ResidualAct> {
    if let Some(i) = from_uuid
        .strip_prefix("input-")
        .and_then(|s| s.parse::<usize>().ok())
    {
        return (i < creature.input).then_some(ResidualAct::Input(i));
    }
    creature
        .neurons
        .iter()
        .position(|n| n.uuid == from_uuid)
        .map(ResidualAct::Neuron)
}

fn residual_activation(
    act: ResidualAct,
    record_inputs: &[f32],
    traced: &[f32],
    post_offset: usize,
) -> f64 {
    match act {
        ResidualAct::Input(i) => f64::from(*record_inputs.get(i).unwrap_or(&0.0)),
        ResidualAct::Neuron(pos) => {
            let idx = post_offset + pos;
            if idx < traced.len() {
                f64::from(traced[idx])
            } else {
                0.0
            }
        }
    }
}

/// Build the residual-refine shortlist: top priors, plus unused previous hiddens.
fn residual_shortlist(prior: &[RankedSource]) -> Vec<RankedSource> {
    let mut shortlist: Vec<RankedSource> = prior.iter().take(RESIDUAL_SHORTLIST).cloned().collect();
    let mut hidden_extra = 0usize;
    for src in prior.iter().filter(|s| !is_input_source(&s.from_uuid)) {
        if shortlist.iter().any(|s| s.from_uuid == src.from_uuid) {
            continue;
        }
        shortlist.push(src.clone());
        hidden_extra += 1;
        if hidden_extra >= RESIDUAL_HIDDEN_EXTRA {
            break;
        }
    }
    shortlist
}

/// Which residual statistic the scan is accumulating.
#[derive(Clone, Copy)]
enum ResidualMode {
    /// Focus is an output: correlate source activation with the focus residual.
    Residual {
        /// Index of the focus within the creature's outputs.
        out_idx: usize,
        /// Focus offset within the traced post-activation block.
        relative_idx: usize,
    },
    /// Focus is hidden: rank sources by measured activation std-dev.
    ActivationStd,
}

/// Streaming accumulator behind [`refine_sources_from_probes`].
///
/// Fed one already-traced probe at a time so the fused post-focus scan can
/// share its activation with the focus-stats and incoming-source passes, and
/// never materialise the sample as `Vec<ActivationProbe>` (issue #105).
pub(crate) struct ResidualScan {
    mode: ResidualMode,
    /// `false` when no probe can change the outcome — `finish` returns `prior`.
    active: bool,
    input_count: usize,
    post_offset: usize,
    shortlist: Vec<RankedSource>,
    acts: Vec<Option<ResidualAct>>,
    sum_x: Vec<f64>,
    sum_xx: Vec<f64>,
    sum_xy: Vec<f64>,
    sum_e: f64,
    sum_ee: f64,
    n: f64,
}

impl ResidualScan {
    /// Build the accumulator for a focus neuron and its prior ranking.
    pub(crate) fn new(
        creature: &CreatureExport,
        focus_uuid: &str,
        prior: &[RankedSource],
    ) -> Result<Self, String> {
        let mode = match focus_output_index(creature, focus_uuid) {
            Some(out_idx) => {
                let relative_idx = neuron_index(creature, focus_uuid)
                    .and_then(|i| i.checked_sub(creature.input))
                    .ok_or_else(|| format!("focus neuron {focus_uuid} missing compiled index"))?;
                ResidualMode::Residual {
                    out_idx,
                    relative_idx,
                }
            }
            None => ResidualMode::ActivationStd,
        };
        let shortlist = residual_shortlist(prior);
        let acts: Vec<Option<ResidualAct>> = shortlist
            .iter()
            .map(|s| residual_act_for(creature, &s.from_uuid))
            .collect();
        // The residual path bails out before scanning when nothing is rankable;
        // the activation-std path scores an empty shortlist to `prior` anyway.
        let active = match mode {
            ResidualMode::Residual { .. } => {
                !shortlist.is_empty() && acts.iter().any(Option::is_some)
            }
            ResidualMode::ActivationStd => true,
        };
        let k = shortlist.len();
        Ok(Self {
            mode,
            active,
            input_count: creature.input,
            post_offset: creature.output,
            shortlist,
            acts,
            sum_x: vec![0.0f64; k],
            sum_xx: vec![0.0f64; k],
            sum_xy: vec![0.0f64; k],
            sum_e: 0.0,
            sum_ee: 0.0,
            n: 0.0,
        })
    }

    /// Whether this probe is usable — callers skip activation when it is not.
    pub(crate) fn wants_probe(&self, inputs: &[f32], outputs: &[f32]) -> bool {
        if !self.active || inputs.len() < self.input_count {
            return false;
        }
        match self.mode {
            ResidualMode::Residual { out_idx, .. } => out_idx < outputs.len(),
            ResidualMode::ActivationStd => true,
        }
    }

    /// Fold one traced probe (`activate_and_trace` output) plus its inputs and targets.
    ///
    /// # Panics
    ///
    /// Panics when the probe was not cleared by [`Self::wants_probe`] first —
    /// that check is what guarantees the target index is in range.
    pub(crate) fn observe(&mut self, inputs: &[f32], outputs: &[f32], traced: &[f32]) {
        match self.mode {
            ResidualMode::Residual {
                out_idx,
                relative_idx,
            } => {
                if self.post_offset + relative_idx >= traced.len() {
                    return;
                }
                let post = f64::from(traced[self.post_offset + relative_idx]);
                let err = f64::from(outputs[out_idx]) - post;
                self.n += 1.0;
                self.sum_e += err;
                self.sum_ee += err * err;
                for (j, act) in self.acts.iter().enumerate() {
                    let Some(act) = *act else {
                        continue;
                    };
                    let x = residual_activation(act, inputs, traced, self.post_offset);
                    self.sum_x[j] += x;
                    self.sum_xx[j] += x * x;
                    self.sum_xy[j] += x * err;
                }
            }
            ResidualMode::ActivationStd => {
                self.n += 1.0;
                for (j, act) in self.acts.iter().enumerate() {
                    let Some(act) = *act else {
                        continue;
                    };
                    let x = residual_activation(act, inputs, traced, self.post_offset);
                    self.sum_x[j] += x;
                    self.sum_xx[j] += x * x;
                }
            }
        }
    }

    /// Consume the accumulator and hand back the re-ranked sources.
    ///
    /// Falls back to `prior` when too few probes were seen to rank anything.
    pub(crate) fn finish(mut self, prior: &[RankedSource]) -> Vec<RankedSource> {
        if !self.active {
            return prior.to_vec();
        }
        match self.mode {
            ResidualMode::Residual { .. } => {
                if self.n < 2.0 {
                    return prior.to_vec();
                }
                for (j, src) in self.shortlist.iter_mut().enumerate() {
                    if self.acts[j].is_none() {
                        continue;
                    }
                    let corr = pearson(
                        self.n,
                        self.sum_x[j],
                        self.sum_e,
                        self.sum_xx[j],
                        self.sum_ee,
                        self.sum_xy[j],
                    );
                    let var_x =
                        ((self.sum_xx[j] / self.n) - (self.sum_x[j] / self.n).powi(2)).max(0.0);
                    let std_x = var_x.sqrt().max(MIN_ACT_STD);
                    src.direction = corr;
                    src.score = corr.abs();
                    src.weight_scale = TARGET_PRE_DELTA / std_x;
                    // Univariate OLS of residual on source: cov(x,e) / var(x).
                    let cov = self.sum_xy[j] - (self.sum_x[j] * self.sum_e) / self.n;
                    let var_x_n = self.sum_xx[j] - (self.sum_x[j] * self.sum_x[j]) / self.n;
                    if var_x_n > 1e-12 {
                        src.ols_weight = Some(cov / var_x_n);
                    }
                    // Dead / flat units stay near zero score even if floating noise correlates weakly.
                    if std_x <= MIN_ACT_STD * 1.5 && !is_input_source(&src.from_uuid) {
                        src.score = 0.0;
                        src.ols_weight = None;
                    }
                }
            }
            ResidualMode::ActivationStd => {
                if self.n < 1.0 {
                    return prior.to_vec();
                }
                for (j, src) in self.shortlist.iter_mut().enumerate() {
                    if self.acts[j].is_none() {
                        continue;
                    }
                    let mean = self.sum_x[j] / self.n;
                    let var = ((self.sum_xx[j] / self.n) - mean * mean).max(0.0);
                    let std = var.sqrt().max(MIN_ACT_STD);
                    src.weight_scale = TARGET_PRE_DELTA / std;
                    // Calculated usefulness prior — dead units rank last.
                    src.score = (std / (1.0 + std)) * 0.05;
                    if std <= MIN_ACT_STD * 1.5 && !is_input_source(&src.from_uuid) {
                        src.score = 0.0;
                    }
                }
            }
        }
        self.shortlist.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.from_uuid.cmp(&b.from_uuid))
        });
        let mut out = self.shortlist;
        for src in prior {
            if !out.iter().any(|s| s.from_uuid == src.from_uuid) {
                out.push(src.clone());
            }
        }
        out
    }
}

/// Activation residual refine: re-rank unused inputs **and previous hiddens** by
/// flowing observation probes through the compiled network and measuring
/// Pearson(post-activation, focus residual).
///
/// This is the authoritative calculated score for unused hiddens — not
/// mean-forward through squashes (`E[f(Wx)] ≠ f(W E[x])`).
pub fn refine_sources_from_probes(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    focus_uuid: &str,
    prior: &[RankedSource],
    probes: &[ActivationProbe],
) -> Result<Vec<RankedSource>, String> {
    let mut scan = ResidualScan::new(creature, focus_uuid, prior)?;
    for probe in probes {
        if !scan.wants_probe(&probe.inputs, &probe.outputs) {
            continue;
        }
        let traced = network.activate_and_trace(&probe.inputs, creature.output);
        scan.observe(&probe.inputs, &probe.outputs, &traced);
    }
    Ok(scan.finish(prior))
}

/// Load training records as activation probes, then
/// [`refine_sources_from_probes`].
///
/// When the corpus yields no usable rows and `observations` is provided, falls
/// back to [`synthetic_observation_probes`].
pub fn refine_sources_by_residual(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    focus_uuid: &str,
    prior: &[RankedSource],
    max_records: Option<u64>,
) -> Result<Vec<RankedSource>, String> {
    refine_sources_by_residual_with_observations(
        creature,
        network,
        training_data,
        focus_uuid,
        prior,
        max_records,
        None,
    )
}

/// Build the synthetic-probe fallback ranking used when the corpus yields
/// fewer than two rows.
///
/// Returns `prior` unchanged when there are no observations to synthesise from.
pub(crate) fn refine_sources_from_synthetic(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    focus_uuid: &str,
    prior: &[RankedSource],
    observations: Option<&ObservationsStatistics>,
) -> Result<Vec<RankedSource>, String> {
    let Some(obs) = observations else {
        return Ok(prior.to_vec());
    };
    let mut rng = StdRng::seed_from_u64(0xA07E_u64);
    let probes = synthetic_observation_probes(obs, creature.input, creature.output, 64, &mut rng);
    refine_sources_from_probes(creature, network, focus_uuid, prior, &probes)
}

/// Like [`refine_sources_by_residual`], with optional synthetic-probe fallback.
///
/// Streams the sample: at most two records are held at once (the corpus needs
/// two rows before the residual statistics mean anything), instead of
/// materialising every record as an [`ActivationProbe`].
pub fn refine_sources_by_residual_with_observations(
    creature: &CreatureExport,
    network: &mut CompiledNetwork,
    training_data: &Path,
    focus_uuid: &str,
    prior: &[RankedSource],
    max_records: Option<u64>,
    observations: Option<&ObservationsStatistics>,
) -> Result<Vec<RankedSource>, String> {
    let mut scan = ResidualScan::new(creature, focus_uuid, prior)?;
    let config = TrainingDataConfig::new(creature.input, creature.output);
    let mut iter = crate::analysis::open_training_scan(training_data, config)?;
    // Hold the first two records back: with fewer than two the sample is
    // discarded for the synthetic fallback, and nothing must be activated.
    let mut held: Vec<TrainingRecord> = Vec::with_capacity(2);
    let mut count = 0u64;
    while let Some(record) = iter.next_record().map_err(|e| e.to_string())? {
        if let Some(limit) = max_records
            && count >= limit
        {
            break;
        }
        count += 1;
        if count <= 2 {
            held.push(record);
            if count == 2 {
                for held_record in std::mem::take(&mut held) {
                    activate_and_observe(&mut scan, network, creature, &held_record);
                }
            }
            continue;
        }
        activate_and_observe(&mut scan, network, creature, &record);
    }

    if count < 2 {
        return refine_sources_from_synthetic(creature, network, focus_uuid, prior, observations);
    }

    Ok(scan.finish(prior))
}

/// Activate one training record and fold it into the residual scan.
fn activate_and_observe(
    scan: &mut ResidualScan,
    network: &mut CompiledNetwork,
    creature: &CreatureExport,
    record: &TrainingRecord,
) {
    if !scan.wants_probe(&record.inputs, &record.outputs) {
        return;
    }
    let traced = network.activate_and_trace(&record.inputs, creature.output);
    scan.observe(&record.inputs, &record.outputs, &traced);
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
        b[0],
        b[1],
        b[2],
        b[3],
        b[4],
        b[5],
        b[6],
        b[7],
        b[8],
        b[9],
        b[10],
        b[11],
        b[12],
        b[13],
        b[14],
        b[15]
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

/// Parameters for inserting a hidden neuron on `from -> new -> focus`.
pub struct NeuronBridgeSpec<'a> {
    /// Upstream source UUID (input or earlier neuron).
    pub from_uuid: &'a str,
    /// Downstream focus neuron UUID.
    pub focus_uuid: &'a str,
    /// UUID for the new hidden neuron.
    pub new_uuid: String,
    /// Squash function name for the new neuron.
    pub squash: &'a str,
    /// Bias for the new neuron.
    pub bias: f64,
    /// Weight on `from -> new`.
    pub w_in: f64,
    /// Weight on `new -> focus`.
    pub w_out: f64,
}

/// Insert a hidden neuron on a path `from -> new -> focus`.
///
/// Returns the new neuron UUID. Fails closed when insertion would break
/// forward-only ordering or the focus cannot host a hidden predecessor.
pub fn add_neuron_bridge(
    creature: &mut CreatureExport,
    spec: NeuronBridgeSpec<'_>,
) -> Result<String, String> {
    let NeuronBridgeSpec {
        from_uuid,
        focus_uuid,
        new_uuid,
        squash,
        bias,
        w_in,
        w_out,
    } = spec;
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
        NeuronBridgeSpec {
            from_uuid: &incoming.from_uuid,
            focus_uuid,
            new_uuid,
            squash,
            bias: 0.0,
            w_in: 1.0,
            w_out: old_w,
        },
    )
}

/// Squashes tried when growing a hidden into the focus.
///
/// Aligned with NEAT-AI Intelligent Design Tier‑1 workhorses, plus a few
/// structural probes (`ABSOLUTE`, `HARD_TANH`, `Softplus`). `TANH` sits early
/// because it maps into a known (−1…1) range; [`growth_squashes_for`] elevates
/// it further when observation stats show values outside that band, and
/// elevates `ABSOLUTE`/`Softplus` when the focus residual is strongly signed.
pub const NEURON_GROWTH_SQUASHES: &[&str] = &[
    "GELU",
    "Swish",
    "TANH",
    "LeakyReLU",
    "Mish",
    "SELU",
    "ELU",
    "ABSOLUTE",
    "HARD_TANH",
    "Softplus",
];

/// Slack around [−1, 1] when deciding whether observations are already unit-scaled.
const UNIT_SCALE_EPS: f64 = 0.05;

/// Whether every observed input already lies (approximately) in [−1, 1].
///
/// When true, a range-bounding squash like `TANH` is less urgent for raw
/// observation bridges. Missing / empty stats → `false` (prefer bounding).
pub fn observations_appear_unit_scaled(observations: &ObservationsStatistics) -> bool {
    let lo = -1.0 - UNIT_SCALE_EPS;
    let hi = 1.0 + UNIT_SCALE_EPS;
    let mut saw = false;
    for s in &observations.inputs {
        if s.count == 0 {
            continue;
        }
        if !(s.min.is_finite() && s.max.is_finite()) {
            return false;
        }
        saw = true;
        if s.min < lo || s.max > hi {
            return false;
        }
    }
    saw
}

/// Squashes whose post-activation is (practically) non-negative.
///
/// For these, outbound weight sign should follow the focus residual: a negative
/// mean error (prediction too high) needs a **negative** weight on an ABSOLUTE
/// (or ReLU/Softplus) unit so the always-≥0 activation pulls the focus down.
pub fn is_nonnegative_squash(squash: &str) -> bool {
    matches!(squash, "ABSOLUTE" | "ReLU" | "RELU" | "ReLU6" | "Softplus")
}

/// Sign of the focus residual used for correction (`target − post`).
///
/// Negative ⇒ prediction too high ⇒ prefer subtracting from the focus.
pub fn residual_correction_sign(focus_stats: &FocusNeuronStats) -> f64 {
    focus_stats
        .mean_adjusted_error
        .or(focus_stats.mean_error)
        .filter(|e| e.is_finite() && e.abs() > 1e-9)
        .map(f64::signum)
        .unwrap_or(1.0)
}

/// Outbound weight for a new hidden → focus edge.
///
/// Non-negative squashes take the residual correction sign; symmetric squashes
/// keep a positive probe magnitude (inbound OLS already carries direction).
pub fn suggested_outbound_weight(
    squash: &str,
    focus_stats: &FocusNeuronStats,
    magnitude: f64,
) -> f64 {
    let mag = magnitude.abs().clamp(1e-5, MAX_NEW_WEIGHT);
    if is_nonnegative_squash(squash) {
        residual_correction_sign(focus_stats) * mag
    } else {
        mag
    }
}

/// Growth squash order for this focus residual / observation range.
///
/// - Signed residual → one-sided probes (`ABSOLUTE`, `Softplus`) first so
///   observation-scaled ABSOLUTE × negative weight can correct average-negative error.
/// - Observations outside ≈[−1, 1] → elevate `TANH` (known-range squash).
/// - Unit-scaled observations → leave `TANH` in its default Tier‑1 slot.
pub fn growth_squashes_for(
    focus_stats: &FocusNeuronStats,
    observations: Option<&ObservationsStatistics>,
) -> Vec<&'static str> {
    let signed = focus_stats
        .mean_adjusted_error
        .or(focus_stats.mean_error)
        .filter(|e| e.is_finite() && e.abs() > 1e-6)
        .is_some();
    let unit = observations
        .map(observations_appear_unit_scaled)
        .unwrap_or(false);

    let mut preferred: Vec<&'static str> = Vec::new();
    if signed {
        preferred.extend(["ABSOLUTE", "Softplus"]);
    }
    if !unit {
        preferred.push("TANH");
    }
    for &s in NEURON_GROWTH_SQUASHES {
        if !preferred.contains(&s) {
            preferred.push(s);
        }
    }
    preferred
}

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
                    skewness: 0.0,
                    excess_kurtosis: 0.0,
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
                    skewness: 0.0,
                    excess_kurtosis: 0.0,
                    quantiles: [0.0; 7],
                },
            ],
            outputs: vec![],
            input_correlations: vec![],
            input_target_correlations: vec![c0, c1],
        }
    }

    const WITH_ORPHAN_HIDDEN: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 2,
      "output": 1,
      "neurons": [
        {"type":"hidden","uuid":"h1","bias":0.1,"squash":"IDENTITY"},
        {"type":"hidden","uuid":"h2","bias":0.0,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"input-1","toUUID":"h2","weight":1.0},
        {"fromUUID":"h2","toUUID":"o1","weight":1.0}
      ]
    }"#;

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
    fn ranks_unused_previous_hiddens_with_zero_prior_score() {
        let creature = parse_creature_json(WITH_ORPHAN_HIDDEN).unwrap();
        let obs = obs_with_corr(0.01, 0.02);
        let ranked = rank_unused_sources(&creature, "o1", &obs);
        let h1 = ranked
            .iter()
            .find(|r| r.from_uuid == "h1")
            .expect("unused previous hidden h1 must be listed");
        assert_eq!(
            h1.score, 0.0,
            "unprobed hiddens must not invent a constant score"
        );
        // h2 already feeds o1 — not unused.
        assert!(!ranked.iter().any(|r| r.from_uuid == "h2"));
        assert_eq!(
            first_previous_hidden_index(&ranked),
            Some(ranked.iter().position(|r| r.from_uuid == "h1").unwrap())
        );
        let rotated = with_previous_hidden_first(&ranked).unwrap();
        assert_eq!(rotated[0].from_uuid, "h1");
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
            NeuronBridgeSpec {
                from_uuid: "input-1",
                focus_uuid: "o1",
                new_uuid: uuid.clone(),
                squash: "TANH",
                bias: 0.0,
                w_in: 0.001,
                w_out: 0.05,
            },
        )
        .unwrap();
        assert!(creature.neurons.iter().any(|n| n.uuid == uuid));
        let pos_new = creature
            .neurons
            .iter()
            .position(|n| n.uuid == uuid)
            .unwrap();
        let pos_out = creature
            .neurons
            .iter()
            .position(|n| n.uuid == "o1")
            .unwrap();
        assert!(pos_new < pos_out);
        compile_creature(&creature).expect("bridged creature must compile");
    }

    fn focus_with_mean_error(mean_error: f64) -> FocusNeuronStats {
        FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_error: Some(mean_error),
            mean_adjusted_error: Some(mean_error),
            ..FocusNeuronStats::default()
        }
    }

    fn focus_unsigned() -> FocusNeuronStats {
        FocusNeuronStats {
            neuron_uuid: "o1".into(),
            ..FocusNeuronStats::default()
        }
    }

    fn scalar(min: f64, max: f64) -> crate::observations::ScalarStats {
        crate::observations::ScalarStats {
            count: 100,
            mean: (min + max) * 0.5,
            variance: 1.0,
            std_dev: 1.0,
            min,
            max,
            zero_count: 0,
            non_zero_count: 100,
            non_finite_count: 0,
            mean_abs: max.abs().max(min.abs()),
            rms: max.abs().max(min.abs()),
            skewness: 0.0,
            excess_kurtosis: 0.0,
            quantiles: [0.0; 7],
        }
    }

    fn obs_with_input_range(min: f64, max: f64) -> ObservationsStatistics {
        ObservationsStatistics {
            format_version: "1.0.0".into(),
            algorithm_version: "1.0.0".into(),
            mode: crate::observations::StatsMode::Full,
            sample_record_limit: None,
            input_count: 1,
            output_count: 1,
            record_count: 100,
            corpus_identity: "range".into(),
            created_at_unix: 0,
            inputs: vec![scalar(min, max)],
            outputs: vec![],
            input_correlations: vec![],
            input_target_correlations: vec![],
        }
    }

    #[test]
    fn negative_residual_prioritises_absolute_growth_squash() {
        let unit = obs_with_input_range(-1.0, 1.0);
        let ordered = growth_squashes_for(&focus_with_mean_error(-0.4), Some(&unit));
        assert_eq!(
            ordered.first().copied(),
            Some("ABSOLUTE"),
            "signed negative residual should try ABSOLUTE first, got {ordered:?}"
        );
    }

    #[test]
    fn unit_scaled_observations_do_not_force_tanh_first() {
        let unit = obs_with_input_range(-0.9, 0.9);
        assert!(observations_appear_unit_scaled(&unit));
        let ordered = growth_squashes_for(&focus_unsigned(), Some(&unit));
        assert_ne!(
            ordered.first().copied(),
            Some("TANH"),
            "already unit-scaled inputs do not need TANH first, got {ordered:?}"
        );
        assert!(
            ordered.contains(&"TANH"),
            "TANH should remain in the growth set"
        );
    }

    #[test]
    fn wide_observations_prioritise_tanh_for_known_range() {
        let wide = obs_with_input_range(-12.0, 40.0);
        assert!(!observations_appear_unit_scaled(&wide));
        let ordered = growth_squashes_for(&focus_unsigned(), Some(&wide));
        assert_eq!(
            ordered.first().copied(),
            Some("TANH"),
            "wide observations should try TANH first for known-range squash, got {ordered:?}"
        );
    }

    #[test]
    fn signed_wide_residual_keeps_absolute_then_tanh() {
        let wide = obs_with_input_range(0.0, 250.0);
        let ordered = growth_squashes_for(&focus_with_mean_error(-0.4), Some(&wide));
        assert_eq!(ordered.first().copied(), Some("ABSOLUTE"));
        assert!(
            ordered.iter().position(|&s| s == "TANH") < ordered.iter().position(|&s| s == "GELU"),
            "wide + signed should elevate TANH ahead of Tier‑1 fillers, got {ordered:?}"
        );
    }

    #[test]
    fn absolute_outbound_weight_matches_residual_sign() {
        // pred too high → negative error → ABSOLUTE (always ≥0) needs negative w_out
        // to pull the focus activation down.
        let w_neg = suggested_outbound_weight("ABSOLUTE", &focus_with_mean_error(-0.25), 0.05);
        assert!(w_neg < 0.0, "expected negative outbound, got {w_neg}");
        let w_pos = suggested_outbound_weight("ABSOLUTE", &focus_with_mean_error(0.25), 0.05);
        assert!(w_pos > 0.0, "expected positive outbound, got {w_pos}");
        // Symmetric squashes keep a modest positive probe (inbound OLS carries sign).
        let w_tanh = suggested_outbound_weight("TANH", &focus_with_mean_error(-0.25), 0.05);
        assert!(
            w_tanh > 0.0,
            "TANH outbound should stay positive, got {w_tanh}"
        );
    }

    /// Symmetric ± inputs: mean-forward through ABSOLUTE looks dead (`abs(0)=0`),
    /// but sample-forward `|x|` tracks the residual when the target is `|x|`.
    const ABSOLUTE_TRAP: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 1,
      "output": 1,
      "neurons": [
        {"type":"hidden","uuid":"h_abs","bias":0.0,"squash":"ABSOLUTE"},
        {"type":"hidden","uuid":"h_signed","bias":0.0,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h_abs","weight":1.0},
        {"fromUUID":"input-0","toUUID":"h_signed","weight":1.0}
      ]
    }"#;

    const DEAD_VS_LIVE: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 1,
      "output": 1,
      "neurons": [
        {"type":"hidden","uuid":"h_live","bias":0.0,"squash":"IDENTITY"},
        {"type":"hidden","uuid":"h_dead","bias":1.0,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h_live","weight":1.0}
      ]
    }"#;

    fn probes_abs_target(xs: &[f32]) -> Vec<ActivationProbe> {
        xs.iter()
            .map(|&x| ActivationProbe {
                inputs: vec![x],
                outputs: vec![x.abs()],
            })
            .collect()
    }

    #[test]
    fn activation_probes_rank_absolute_not_mean_forward() {
        use neat_core::compile_creature;
        let creature = parse_creature_json(ABSOLUTE_TRAP).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let obs = ObservationsStatistics {
            format_version: "1.0.0".into(),
            algorithm_version: "1.0.0".into(),
            mode: crate::observations::StatsMode::Full,
            sample_record_limit: None,
            input_count: 1,
            output_count: 1,
            record_count: 0,
            corpus_identity: "trap".into(),
            created_at_unix: 0,
            inputs: vec![scalar(-1.0, 1.0)],
            outputs: vec![],
            input_correlations: vec![],
            input_target_correlations: vec![0.0],
        };
        let prior = rank_unused_sources(&creature, "o1", &obs);
        // Symmetric ± probes: mean input ≈ 0 ⇒ mean-forward ABSOLUTE is useless.
        let probes = probes_abs_target(&[-1.0, -0.5, 0.5, 1.0, -0.75, 0.75]);
        let ranked =
            refine_sources_from_probes(&creature, &mut network, "o1", &prior, &probes).unwrap();
        let h_abs = ranked.iter().find(|r| r.from_uuid == "h_abs").unwrap();
        let h_signed = ranked.iter().find(|r| r.from_uuid == "h_signed").unwrap();
        assert!(
            h_abs.score > 0.9,
            "ABSOLUTE post should track |x| residual, score={}",
            h_abs.score
        );
        assert!(
            h_abs.score > h_signed.score + 0.2,
            "sample-forward must prefer ABSOLUTE over signed IDENTITY for |x| target; scores abs={} signed={}",
            h_abs.score,
            h_signed.score
        );
        assert_ne!(h_abs.score, 0.05);
        assert_ne!(h_signed.score, 0.05);
    }

    #[test]
    fn activation_probes_rank_dead_hidden_below_live() {
        use neat_core::compile_creature;
        let creature = parse_creature_json(DEAD_VS_LIVE).unwrap();
        let mut network = compile_creature(&creature).unwrap();
        let obs = ObservationsStatistics {
            format_version: "1.0.0".into(),
            algorithm_version: "1.0.0".into(),
            mode: crate::observations::StatsMode::Full,
            sample_record_limit: None,
            input_count: 1,
            output_count: 1,
            record_count: 0,
            corpus_identity: "dead".into(),
            created_at_unix: 0,
            inputs: vec![scalar(-1.0, 1.0)],
            outputs: vec![],
            input_correlations: vec![],
            input_target_correlations: vec![0.5],
        };
        let prior = rank_unused_sources(&creature, "o1", &obs);
        // Target tracks input; live hidden posts = x; dead is constant bias.
        let probes: Vec<ActivationProbe> = [-1.0f32, -0.5, 0.0, 0.5, 1.0]
            .into_iter()
            .map(|x| ActivationProbe {
                inputs: vec![x],
                outputs: vec![x],
            })
            .collect();
        let ranked =
            refine_sources_from_probes(&creature, &mut network, "o1", &prior, &probes).unwrap();
        let live = ranked.iter().find(|r| r.from_uuid == "h_live").unwrap();
        let dead = ranked.iter().find(|r| r.from_uuid == "h_dead").unwrap();
        assert!(live.score > 0.9, "live score={}", live.score);
        assert!(
            dead.score < 0.05,
            "constant dead hidden must not rank, score={}",
            dead.score
        );
        assert!(live.score > dead.score);
    }

    #[test]
    fn synthetic_probes_are_seeded_and_clamped() {
        let obs = obs_with_input_range(-0.5, 0.5);
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        let a = synthetic_observation_probes(&obs, 1, 1, 6, &mut rng_a);
        let b = synthetic_observation_probes(&obs, 1, 1, 6, &mut rng_b);
        assert_eq!(a.len(), 6);
        assert_eq!(a[0].inputs, b[0].inputs);
        for p in &a {
            assert!(p.inputs[0] >= -0.5 - 1e-5 && p.inputs[0] <= 0.5 + 1e-5);
        }
    }
}
