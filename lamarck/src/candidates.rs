//! Candidate population generation from an incumbent creature.

use crate::backprop::{BackpropConfig, LearningSignal};
use crate::focus::{FocusNeuronStats, IncomingSourceStats};
use crate::observations::ObservationsStatistics;
use crate::structural::{
    NeuronBridgeSpec, OLS_WEIGHT_FRACTION, RankedSource, add_neuron_bridge, add_synapse,
    first_previous_hidden_index, growth_squashes_for, pick_smart_source, random_uuid_v4,
    rank_unused_sources, split_incoming_synapse, suggested_outbound_weight, suggested_weight,
    suggested_weight_scaled, with_previous_hidden_first,
};
use crate::tags::{CreatureMeta, serialize_creature_with_meta};
use neat_core::{CreatureExport, creature_to_json_pretty};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Strategy label recorded in the experiment journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStrategy {
    /// Conventional backprop-derived bias/weight change.
    Backprop,
    /// Output-neuron bias += squash-aware mean adjusted error.
    MeanErrorBias,
    /// Statistics-guided weight change.
    StatsWeight,
    /// Statistics-guided bias change.
    StatsBias,
    /// Output bias stepped towards a skewed target's median.
    StatsSkewBias,
    /// Add a correlation-ranked upstream connection into the focus.
    StructuralAdd,
    /// Grow a hidden neuron on a path into the focus.
    StructuralAddNeuron,
    /// Weaken an apparently weak/useless incoming connection.
    StructuralWeaken,
    /// Random exploratory mutation.
    Random,
}

/// Provenance for one candidate creature.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateProvenance {
    /// Strategy that produced the candidate.
    pub strategy: CandidateStrategy,
    /// Focus neuron UUID.
    pub focus_neuron: String,
    /// Human-readable mutation description.
    pub mutation: String,
    /// Old value when a scalar field changed.
    pub old_value: Option<f64>,
    /// New value when a scalar field changed.
    pub new_value: Option<f64>,
}

/// One generated candidate plus provenance.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Candidate creature JSON export.
    pub creature: CreatureExport,
    /// Provenance metadata.
    pub provenance: CandidateProvenance,
}

/// Fraction of mean adjusted error applied as a bias step (full mean overshoots).
const MEAN_ERROR_STEP_FRACTION: f64 = 0.1;
/// Cap on |Δw| from a single backprop weight propose (fine-tuned nets).
const MAX_BACKPROP_WEIGHT_DELTA: f64 = 0.01;
/// Minimum |residual corr| before growing a neuron bridge.
const MIN_NEURON_BRIDGE_SCORE: f64 = 0.02;
/// Minimum |target skewness| before a median-ward bias step is worth proposing.
const MIN_TARGET_SKEW: f64 = 0.25;
/// Fraction of the target's mean→median gap applied as a bias step.
const SKEW_BIAS_STEP_FRACTION: f64 = 0.25;
/// Excess kurtosis at which the skew-bias step is halved (heavy tails make the
/// sampled median gap noisy, so trust it less).
const SKEW_BIAS_KURTOSIS_REFERENCE: f64 = 3.0;

/// Inputs shared by candidate generation for one focus-neuron experiment.
pub struct CandidateGenContext<'a> {
    /// Incumbent creature (never mutated).
    pub incumbent: &'a CreatureExport,
    /// Focus neuron UUID.
    pub focus_uuid: &'a str,
    /// Focused neuron statistics.
    pub focus_stats: &'a FocusNeuronStats,
    /// Incoming source statistics for the focus.
    pub incoming: &'a [IncomingSourceStats],
    /// Dataset statistics.
    pub observations: &'a ObservationsStatistics,
    /// Optional residual-ranked unused sources (preferred over raw target corr).
    pub ranked_sources: Option<&'a [RankedSource]>,
    /// Optional accumulated learning signal.
    pub learning: Option<&'a LearningSignal>,
    /// Backprop configuration.
    pub backprop: &'a BackpropConfig,
    /// When true, only emit synapse/neuron growth candidates.
    pub structural_only: bool,
}

/// Generate a candidate population without mutating the incumbent.
pub fn generate_candidates(
    ctx: &CandidateGenContext<'_>,
    count: usize,
    rng: &mut impl Rng,
) -> Vec<Candidate> {
    let mut out = Vec::with_capacity(count);
    if count == 0 {
        return out;
    }

    let ranked = ranked_for(ctx);
    let hidden_first = with_previous_hidden_first(&ranked);

    // Synapse add, then one neuron per squash (order follows residual shape),
    // then additional synapse scales to fill the budget.
    if let Some(cand) = build_structural_add_scaled(ctx, &ranked, 0, OLS_WEIGHT_FRACTION) {
        out.push(cand);
    }
    // Explicitly try hooking an unused previous hidden into the focus even when
    // its probe score is still zero (unmeasured prior).
    if let Some(hid_i) = first_previous_hidden_index(&ranked)
        && out.len() < count
    {
        let already_added = out.iter().any(|c| {
            c.provenance.strategy == CandidateStrategy::StructuralAdd
                && c.provenance.mutation.contains(&ranked[hid_i].from_uuid)
        });
        if !already_added
            && let Some(cand) =
                build_structural_add_scaled_gated(ctx, &ranked, hid_i, OLS_WEIGHT_FRACTION, false)
        {
            out.push(cand);
        }
    }
    let growth_squashes = growth_squashes_for(ctx.focus_stats, Some(ctx.observations));
    for (squash_i, &squash) in growth_squashes.iter().enumerate() {
        if out.len() >= count {
            break;
        }
        // Under mixed mode, only emit a few squashes so weight strategies still fit.
        if !ctx.structural_only && squash_i >= 3 {
            break;
        }
        if let Some(cand) = build_structural_add_neuron_combo(ctx, &ranked, rng, squash) {
            out.push(cand);
        }
    }
    // Also grow via a previous hidden when the top residual source is an input.
    if let Some(ref hid_ranked) = hidden_first
        && hid_ranked
            .first()
            .is_some_and(|s| !crate::structural::is_input_source(&s.from_uuid))
        && ranked
            .first()
            .is_some_and(|s| crate::structural::is_input_source(&s.from_uuid))
    {
        for (squash_i, &squash) in growth_squashes.iter().enumerate() {
            if out.len() >= count {
                break;
            }
            if !ctx.structural_only && squash_i >= 2 {
                break;
            }
            if let Some(cand) = build_structural_add_neuron_combo(ctx, hid_ranked, rng, squash) {
                out.push(cand);
            }
        }
    }
    for (idx, scale) in [
        (0usize, OLS_WEIGHT_FRACTION * 2.0),
        (1, OLS_WEIGHT_FRACTION),
        (2, OLS_WEIGHT_FRACTION),
        (3, OLS_WEIGHT_FRACTION),
    ] {
        if out.len() >= count {
            break;
        }
        if let Some(cand) = build_structural_add_scaled(ctx, &ranked, idx, scale) {
            out.push(cand);
        }
    }
    if ctx.structural_only {
        // Keep filling with remaining residual-ordered squashes / synapse adds.
        let mut squash_i = 0usize;
        let mut syn_i = 0usize;
        let n_squash = growth_squashes.len().max(1);
        while out.len() < count && squash_i + syn_i < n_squash * 3 {
            if squash_i <= syn_i {
                let squash = growth_squashes[squash_i % n_squash];
                squash_i += 1;
                if let Some(cand) = build_structural_add_neuron_combo(ctx, &ranked, rng, squash) {
                    out.push(cand);
                }
            } else {
                syn_i += 1;
                if let Some(cand) =
                    build_structural_add_scaled(ctx, &ranked, syn_i, OLS_WEIGHT_FRACTION)
                {
                    out.push(cand);
                }
            }
        }
        return out;
    }

    let has_error =
        ctx.focus_stats.mean_adjusted_error.is_some() || ctx.focus_stats.mean_error.is_some();
    if has_error
        && out.len() < count
        && let Some(candidate) = build_candidate(ctx, CandidateStrategy::MeanErrorBias, rng)
    {
        out.push(candidate);
    }

    let fill = [
        CandidateStrategy::Backprop,
        CandidateStrategy::StatsWeight,
        CandidateStrategy::StructuralAdd,
        CandidateStrategy::StructuralAddNeuron,
        CandidateStrategy::StatsBias,
        CandidateStrategy::StatsSkewBias,
        CandidateStrategy::StructuralWeaken,
        CandidateStrategy::Random,
    ];
    let mut strategy_i = 0usize;
    while out.len() < count && strategy_i < fill.len() * 3 {
        let strategy = fill[strategy_i % fill.len()];
        strategy_i += 1;
        if let Some(candidate) = build_candidate(ctx, strategy, rng) {
            out.push(candidate);
        }
    }
    out
}

fn ranked_for(ctx: &CandidateGenContext<'_>) -> Vec<RankedSource> {
    if let Some(ranked) = ctx.ranked_sources {
        return ranked.to_vec();
    }
    rank_unused_sources(ctx.incumbent, ctx.focus_uuid, ctx.observations)
}

fn build_structural_add_scaled(
    ctx: &CandidateGenContext<'_>,
    ranked: &[RankedSource],
    source_index: usize,
    scale: f64,
) -> Option<Candidate> {
    build_structural_add_scaled_gated(ctx, ranked, source_index, scale, true)
}

/// Like [`build_structural_add_scaled`], optionally skipping the residual-score floor
/// so candidate gen can still force-try an unused previous hidden before probes run.
fn build_structural_add_scaled_gated(
    ctx: &CandidateGenContext<'_>,
    ranked: &[RankedSource],
    source_index: usize,
    scale: f64,
    require_score: bool,
) -> Option<Candidate> {
    let source = ranked.get(source_index)?;
    if require_score && source.score < 1e-4 {
        return None;
    }
    let focus_uuid = ctx.focus_uuid;
    let weight = suggested_weight_scaled(source, ctx.focus_stats, scale);
    if weight.abs() < ctx.backprop.plank_constant {
        return None;
    }
    let mut creature = ctx.incumbent.clone();
    // Skip if this edge already exists (can happen when emitting multiple scales).
    if creature
        .synapses
        .iter()
        .any(|s| s.to_uuid == focus_uuid && s.from_uuid == source.from_uuid)
    {
        return None;
    }
    let from = source.from_uuid.clone();
    let score = source.score;
    let ols = source.ols_weight;
    add_synapse(&mut creature, from.clone(), focus_uuid, weight);
    Some(Candidate {
        creature,
        provenance: CandidateProvenance {
            strategy: CandidateStrategy::StructuralAdd,
            focus_neuron: focus_uuid.to_string(),
            mutation: format!(
                "structural add {from} -> {focus_uuid} w={weight} \
                 (score={score:.4}, ols={ols:?}, scale={scale:.3})"
            ),
            old_value: None,
            new_value: Some(weight),
        },
    })
}

/// Grow a hidden that combines the top two residual sources into the focus.
fn build_structural_add_neuron_combo(
    ctx: &CandidateGenContext<'_>,
    ranked: &[RankedSource],
    rng: &mut impl Rng,
    squash: &str,
) -> Option<Candidate> {
    let focus_uuid = ctx.focus_uuid;
    let a = ranked.first()?;
    if a.score < MIN_NEURON_BRIDGE_SCORE {
        return None;
    }
    let mut creature = ctx.incumbent.clone();
    let new_uuid = random_uuid_v4(rng);
    let w_a = suggested_weight_scaled(a, ctx.focus_stats, OLS_WEIGHT_FRACTION);
    // Keep outbound modest; for ABSOLUTE/ReLU/Softplus the sign follows residual
    // (negative error → negative weight on a non-negative activation).
    let w_out = suggested_outbound_weight(squash, ctx.focus_stats, 0.05);

    let uuid = add_neuron_bridge(
        &mut creature,
        NeuronBridgeSpec {
            from_uuid: &a.from_uuid,
            focus_uuid,
            new_uuid: new_uuid.clone(),
            squash,
            bias: 0.0,
            w_in: w_a,
            w_out,
        },
    )
    .ok()?;

    // Optional second residual source into the new neuron.
    let mut second = None;
    if let Some(b) = ranked
        .get(1)
        .filter(|b| b.score >= MIN_NEURON_BRIDGE_SCORE * 0.5)
    {
        let w_b = suggested_weight_scaled(b, ctx.focus_stats, OLS_WEIGHT_FRACTION * 0.5);
        if crate::structural::is_forward_edge(&creature, &b.from_uuid, &uuid) {
            add_synapse(&mut creature, b.from_uuid.clone(), &uuid, w_b);
            second = Some((b.from_uuid.clone(), w_b));
        }
    }

    let mutation = match second {
        Some((from_b, w_b)) => format!(
            "structural add-neuron {a_from}+{from_b} -> {uuid} -> {focus_uuid} \
             squash={squash} wa={w_a} wb={w_b} wout={w_out} (scores={:.4}/{:.4})",
            a.score,
            ranked.get(1).map(|s| s.score).unwrap_or(0.0),
            a_from = a.from_uuid,
        ),
        None => format!(
            "structural add-neuron {} -> {uuid} -> {focus_uuid} \
             squash={squash} win={w_a} wout={w_out} (score={:.4})",
            a.from_uuid, a.score
        ),
    };

    Some(Candidate {
        creature,
        provenance: CandidateProvenance {
            strategy: CandidateStrategy::StructuralAddNeuron,
            focus_neuron: focus_uuid.to_string(),
            mutation,
            old_value: None,
            new_value: Some(w_a),
        },
    })
}

fn build_mean_error_bias(
    ctx: &CandidateGenContext<'_>,
    _rng: &mut impl Rng,
    fraction: f64,
) -> Option<Candidate> {
    let focus_uuid = ctx.focus_uuid;
    let focus_stats = ctx.focus_stats;
    let backprop = ctx.backprop;
    let mean_adj = focus_stats.mean_adjusted_error.or(focus_stats.mean_error)?;
    let mean_deriv = focus_stats.mean_derivative.unwrap_or(1.0);
    if mean_deriv <= 1e-6 {
        return None;
    }
    let mut creature = ctx.incumbent.clone();
    let neuron_pos = creature.neurons.iter().position(|n| n.uuid == focus_uuid)?;
    let old_bias = creature.neurons[neuron_pos].bias;
    let step = mean_adj * fraction;
    if step.abs() < backprop.plank_constant {
        return None;
    }
    let new_bias = (old_bias + step).clamp(-backprop.limit_bias_scale, backprop.limit_bias_scale);
    creature.neurons[neuron_pos].bias = new_bias;
    Some(Candidate {
        creature,
        provenance: CandidateProvenance {
            strategy: CandidateStrategy::MeanErrorBias,
            focus_neuron: focus_uuid.to_string(),
            mutation: format!(
                "mean-error bias Δ {step} (frac={fraction:.3}, deriv={mean_deriv:.4}, {old_bias} -> {new_bias})"
            ),
            old_value: Some(old_bias),
            new_value: Some(new_bias),
        },
    })
}

/// Step an output neuron's bias towards its target's median when that target is
/// skewed (Issue #73).
///
/// A squared-error fit centres a prediction on the target **mean**; under a
/// skewed target the mean sits away from the typical value, so a step towards
/// the median is a distinct hypothesis worth scoring. The observation cache's
/// skewness gates the proposal and its excess kurtosis damps it — heavy tails
/// make the sampled median gap unreliable. Hidden focuses (no target), symmetric
/// targets and saturated neurons produce nothing.
fn build_stats_skew_bias(ctx: &CandidateGenContext<'_>) -> Option<Candidate> {
    let focus_uuid = ctx.focus_uuid;
    let out_idx = crate::structural::focus_output_index(ctx.incumbent, focus_uuid)?;
    let target = ctx.observations.outputs.get(out_idx)?;
    if target.count == 0 || !target.skewness.is_finite() {
        return None;
    }
    let skewness = target.skewness;
    if skewness.abs() < MIN_TARGET_SKEW {
        return None;
    }
    // Median of the seven stored quantiles (1/5/25/50/75/95/99%).
    let median = target.quantiles[3];
    if !median.is_finite() || !target.mean.is_finite() {
        return None;
    }
    let mean_deriv = ctx.focus_stats.mean_derivative.unwrap_or(1.0);
    if mean_deriv <= 1e-6 {
        return None;
    }
    let excess_kurtosis = if target.excess_kurtosis.is_finite() {
        target.excess_kurtosis.max(0.0)
    } else {
        0.0
    };
    let damping = 1.0 / (1.0 + excess_kurtosis / SKEW_BIAS_KURTOSIS_REFERENCE);
    let step = (median - target.mean) * SKEW_BIAS_STEP_FRACTION * damping * mean_deriv;
    if !step.is_finite() || step.abs() < ctx.backprop.plank_constant {
        return None;
    }
    let mut creature = ctx.incumbent.clone();
    let neuron_pos = creature.neurons.iter().position(|n| n.uuid == focus_uuid)?;
    let old_bias = creature.neurons[neuron_pos].bias;
    let new_bias = (old_bias + step).clamp(
        -ctx.backprop.limit_bias_scale,
        ctx.backprop.limit_bias_scale,
    );
    creature.neurons[neuron_pos].bias = new_bias;
    Some(Candidate {
        creature,
        provenance: CandidateProvenance {
            strategy: CandidateStrategy::StatsSkewBias,
            focus_neuron: focus_uuid.to_string(),
            mutation: format!(
                "skew bias Δ {step} towards target-{out_idx} median {median} \
                 (mean={}, skew={skewness:.4}, excessKurtosis={:.4}, damp={damping:.4}, \
                 deriv={mean_deriv:.4})",
                target.mean, target.excess_kurtosis
            ),
            old_value: Some(old_bias),
            new_value: Some(new_bias),
        },
    })
}

fn build_candidate(
    ctx: &CandidateGenContext<'_>,
    strategy: CandidateStrategy,
    rng: &mut impl Rng,
) -> Option<Candidate> {
    let incumbent = ctx.incumbent;
    let focus_uuid = ctx.focus_uuid;
    let focus_stats = ctx.focus_stats;
    let learning = ctx.learning;
    let backprop = ctx.backprop;

    let mut creature = incumbent.clone();
    let neuron_pos = creature.neurons.iter().position(|n| n.uuid == focus_uuid)?;
    let old_bias = creature.neurons[neuron_pos].bias;

    let (mutation, old_value, new_value) = match strategy {
        CandidateStrategy::MeanErrorBias => {
            return build_mean_error_bias(ctx, rng, MEAN_ERROR_STEP_FRACTION);
        }
        CandidateStrategy::StatsSkewBias => return build_stats_skew_bias(ctx),
        CandidateStrategy::Backprop => {
            let lr = backprop.learning_rate;
            if let Some(signal) = learning.and_then(|l| l.biases.get(neuron_pos))
                && signal.count > 0.0
            {
                let new_bias = signal.propose(old_bias, backprop, lr);
                // Also try a weight propose on the strongest incoming synapse.
                if let Some(src) = pick_best_incoming(ctx.incoming, rng)
                    && let Some(w_signal) = learning.and_then(|l| l.weights.get(src.synapse_index))
                    && w_signal.count > 0.0
                    && rng.random_bool(0.5)
                {
                    let old_w = creature.synapses[src.synapse_index].weight;
                    let proposed = w_signal.propose(old_w, backprop, lr);
                    let delta = (proposed - old_w)
                        .clamp(-MAX_BACKPROP_WEIGHT_DELTA, MAX_BACKPROP_WEIGHT_DELTA);
                    let new_w = old_w + delta;
                    if (new_w - old_w).abs() < backprop.plank_constant {
                        // Fall through to bias propose.
                    } else {
                        creature.synapses[src.synapse_index].weight = new_w;
                        return Some(Candidate {
                            creature,
                            provenance: CandidateProvenance {
                                strategy,
                                focus_neuron: focus_uuid.to_string(),
                                mutation: format!(
                                    "backprop weight {} {old_w} -> {new_w} (count={}, capped)",
                                    src.from_uuid, w_signal.count
                                ),
                                old_value: Some(old_w),
                                new_value: Some(new_w),
                            },
                        });
                    }
                }
                if (new_bias - old_bias).abs() < backprop.plank_constant {
                    return None;
                }
                creature.neurons[neuron_pos].bias = new_bias;
                (
                    format!(
                        "backprop bias {old_bias} -> {new_bias} (count={})",
                        signal.count
                    ),
                    Some(old_bias),
                    Some(new_bias),
                )
            } else {
                // No accumulated blame on the focus — issue #83. The old
                // residual fallback ran the mean adjusted error through both
                // the 0.1 step fraction and the learning rate, landing ~200x
                // below the scale accepted candidates move at, and duplicated
                // `mean_error_bias` at a strictly worse size. Skip instead so
                // the batch slot goes to a strategy that can clear
                // `--min-improvement`.
                return None;
            }
        }
        CandidateStrategy::StatsBias => {
            let scale = if focus_stats.pre_variance > 0.0 {
                focus_stats.pre_variance.sqrt()
            } else {
                0.01
            };
            // Prefer moving away from saturation when saturated.
            let direction = if focus_stats.saturation_fraction > 0.5 {
                if focus_stats.post_mean >= 0.0 {
                    -1.0
                } else {
                    1.0
                }
            } else {
                rng.random_range(-1.0..1.0)
            };
            let delta = scale * 0.05 * direction;
            let new_bias =
                (old_bias + delta).clamp(-backprop.limit_bias_scale, backprop.limit_bias_scale);
            creature.neurons[neuron_pos].bias = new_bias;
            (
                format!(
                    "stats bias Δ {delta} (pre_std={scale:.4}, sat={:.3})",
                    focus_stats.saturation_fraction
                ),
                Some(old_bias),
                Some(new_bias),
            )
        }
        CandidateStrategy::StatsWeight => {
            let src = pick_best_incoming(ctx.incoming, rng)?;
            let syn_idx = src.synapse_index;
            let old_w = creature.synapses[syn_idx].weight;
            let source_scale = src.std_dev.max(1e-3);
            let preferred = src.correlation_with_error.unwrap_or(0.0);
            let direction = if preferred.abs() > 0.05 {
                preferred.signum()
            } else {
                rng.random_range(-1.0..1.0)
            };
            let delta = ((0.01 / source_scale) * direction * rng.random_range(0.25..1.0))
                .clamp(-MAX_BACKPROP_WEIGHT_DELTA, MAX_BACKPROP_WEIGHT_DELTA);
            let new_w =
                (old_w + delta).clamp(-backprop.limit_weight_scale, backprop.limit_weight_scale);
            creature.synapses[syn_idx].weight = new_w;
            (
                format!(
                    "stats weight {} Δ {delta} (std={source_scale:.4}, corr={:?})",
                    src.from_uuid, src.correlation_with_error
                ),
                Some(old_w),
                Some(new_w),
            )
        }
        CandidateStrategy::StructuralWeaken => {
            let src = ctx
                .incoming
                .iter()
                .min_by(|a, b| {
                    a.weight
                        .abs()
                        .partial_cmp(&b.weight.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .or_else(|| pick_best_incoming(ctx.incoming, rng))?;
            let syn_idx = src.synapse_index;
            let old_w = creature.synapses[syn_idx].weight;
            let new_w = old_w * 0.5;
            if (new_w - old_w).abs() < backprop.plank_constant {
                return None;
            }
            creature.synapses[syn_idx].weight = new_w;
            (
                format!("structural weaken {} {old_w} -> {new_w}", src.from_uuid),
                Some(old_w),
                Some(new_w),
            )
        }
        CandidateStrategy::StructuralAdd => {
            let ranked = ranked_for(ctx);
            let source = pick_smart_source(&ranked, rng)?;
            if source.score < 1e-4 {
                return None;
            }
            let weight = suggested_weight(source, focus_stats);
            let from = source.from_uuid.clone();
            let score = source.score;
            add_synapse(&mut creature, from.clone(), focus_uuid, weight);
            (
                format!("structural add {from} -> {focus_uuid} w={weight} (score={score:.4})"),
                None,
                Some(weight),
            )
        }
        CandidateStrategy::StructuralAddNeuron => {
            let ranked = ranked_for(ctx);
            let squashes = growth_squashes_for(ctx.focus_stats, Some(ctx.observations));
            // Prefer the residual-fronted squash (e.g. ABSOLUTE on negative mean error).
            let squash = squashes.first().copied().unwrap_or("LeakyReLU");
            if let Some(cand) = build_structural_add_neuron_combo(ctx, &ranked, rng, squash) {
                return Some(cand);
            }
            // Prefer reusing a previous hidden when the top residual source is an input.
            if let Some(hid_ranked) = with_previous_hidden_first(&ranked)
                && let Some(cand) = build_structural_add_neuron_combo(ctx, &hid_ranked, rng, squash)
            {
                return Some(cand);
            }
            // Fall back: split the strongest error-correlated incoming edge.
            let new_uuid = random_uuid_v4(rng);
            let src = pick_best_incoming(ctx.incoming, rng)?;
            let old_w = src.weight;
            let from = src.from_uuid.clone();
            let uuid =
                split_incoming_synapse(&mut creature, src, focus_uuid, new_uuid, squash).ok()?;
            (
                format!(
                    "structural split-neuron {from} -> {uuid} -> {focus_uuid} \
                     (old_w={old_w}, squash={squash})"
                ),
                None,
                Some(old_w),
            )
        }
        CandidateStrategy::Random => {
            if rng.random_bool(0.5) {
                let delta = rng.random_range(-0.05..0.05);
                let new_bias = old_bias + delta;
                creature.neurons[neuron_pos].bias = new_bias;
                (
                    format!("random bias delta {delta}"),
                    Some(old_bias),
                    Some(new_bias),
                )
            } else {
                let src = pick_best_incoming(ctx.incoming, rng)?;
                let syn_idx = src.synapse_index;
                let old_w = creature.synapses[syn_idx].weight;
                let delta = rng.random_range(-0.05..0.05);
                let new_w = old_w + delta;
                creature.synapses[syn_idx].weight = new_w;
                (
                    format!("random weight delta {delta}"),
                    Some(old_w),
                    Some(new_w),
                )
            }
        }
    };

    Some(Candidate {
        creature,
        provenance: CandidateProvenance {
            strategy,
            focus_neuron: focus_uuid.to_string(),
            mutation,
            old_value,
            new_value,
        },
    })
}

fn pick_best_incoming<'a>(
    incoming: &'a [IncomingSourceStats],
    rng: &mut impl Rng,
) -> Option<&'a IncomingSourceStats> {
    if incoming.is_empty() {
        return None;
    }
    // Prefer highest |correlation_with_error| when residual corr is meaningful.
    if let Some(best) = incoming.iter().max_by(|a, b| {
        let aa = a.correlation_with_error.unwrap_or(0.0).abs();
        let bb = b.correlation_with_error.unwrap_or(0.0).abs();
        aa.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
    }) && best.correlation_with_error.unwrap_or(0.0).abs() > 0.05
    {
        return Some(best);
    }
    // Hidden focus (or weak residual): prefer largest |proposed weight Δ| from
    // the backprop learning signal (issue #4).
    if let Some(best) = incoming.iter().max_by(|a, b| {
        let aa = a.proposed_weight_delta.unwrap_or(0.0).abs();
        let bb = b.proposed_weight_delta.unwrap_or(0.0).abs();
        aa.partial_cmp(&bb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let ac = a.weight_signal_count.unwrap_or(0.0);
                let bc = b.weight_signal_count.unwrap_or(0.0);
                ac.partial_cmp(&bc).unwrap_or(std::cmp::Ordering::Equal)
            })
    }) && best.proposed_weight_delta.unwrap_or(0.0).abs() > 1e-12
    {
        return Some(best);
    }
    Some(&incoming[rng.random_range(0..incoming.len())])
}

/// Write baseline + candidates into a temporary scoring directory.
///
/// Recreates `dir` so leftover JSON from a prior larger batch cannot be scored.
///
/// When `meta` is provided, `baseline.json` keeps original `uuid` / `tags`
/// (candidates stay untagged — acceptance stamps the winner on write).
pub fn write_candidate_batch(
    dir: &Path,
    incumbent: &CreatureExport,
    candidates: &[Candidate],
    meta: Option<&CreatureMeta>,
) -> Result<Vec<String>, String> {
    if dir.exists() {
        fs::remove_dir_all(dir).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let baseline = if let Some(meta) = meta {
        serialize_creature_with_meta(incumbent, meta)?
    } else {
        creature_to_json_pretty(incumbent).map_err(|e| e.to_string())?
    };
    fs::write(dir.join("baseline.json"), baseline).map_err(|e| e.to_string())?;

    let mut stems = vec!["baseline".to_string()];
    for (i, candidate) in candidates.iter().enumerate() {
        let stem = format!("candidate-{i:03}");
        let json = creature_to_json_pretty(&candidate.creature).map_err(|e| e.to_string())?;
        fs::write(dir.join(format!("{stem}.json")), json).map_err(|e| e.to_string())?;
        stems.push(stem);
    }
    Ok(stems)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backprop::BiasSignal;
    use crate::structural::{refine_sources_from_probes, synthetic_observation_probes};
    use neat_core::parse_creature_json;
    use rand::{SeedableRng, rngs::StdRng};

    const TINY: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 1,
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

    fn empty_obs() -> ObservationsStatistics {
        ObservationsStatistics {
            format_version: "1.0.0".into(),
            algorithm_version: "1.0.0".into(),
            mode: crate::observations::StatsMode::Full,
            sample_record_limit: None,
            input_count: 1,
            output_count: 1,
            record_count: 0,
            corpus_identity: "x".into(),
            created_at_unix: 0,
            inputs: vec![crate::observations::ScalarStats {
                count: 1,
                mean: 0.0,
                variance: 1.0,
                std_dev: 1.0,
                min: 0.0,
                max: 0.0,
                zero_count: 0,
                non_zero_count: 0,
                non_finite_count: 0,
                mean_abs: 0.0,
                rms: 0.0,
                skewness: 0.0,
                excess_kurtosis: 0.0,
                quantiles: [0.0; 7],
            }],
            outputs: vec![],
            input_correlations: vec![],
            input_target_correlations: vec![],
        }
    }

    #[test]
    fn generation_is_deterministic_and_non_mutating() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let original = incumbent.clone();
        let focus = FocusNeuronStats {
            neuron_uuid: "h1".into(),
            pre_variance: 0.04,
            ..FocusNeuronStats::default()
        };
        let incoming = vec![IncomingSourceStats {
            synapse_index: 0,
            from_uuid: "input-0".into(),
            weight: 1.0,
            is_input: true,
            input_index: Some(0),
            mean: 0.0,
            variance: 1.0,
            std_dev: 1.0,
            correlation_with_error: None,
            weight_signal_count: None,
            proposed_weight_delta: None,
            mean_weight_sensitivity: None,
        }];
        let observations = empty_obs();
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "h1",
            focus_stats: &focus,
            incoming: &incoming,
            observations: &observations,
            ranked_sources: None,
            learning: None,
            backprop: &cfg,
            structural_only: false,
        };
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        let a = generate_candidates(&ctx, 8, &mut rng_a);
        let b = generate_candidates(&ctx, 8, &mut rng_b);
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].provenance.mutation, b[0].provenance.mutation);
        assert_eq!(incumbent, original);
        assert!(incumbent.forward_only);
    }

    #[test]
    fn mean_error_bias_nudge_is_first_when_error_known() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_error: Some(0.25),
            mean_abs_error: Some(0.25),
            mean_adjusted_error: Some(0.25),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        };
        let incoming = [IncomingSourceStats {
            synapse_index: 1,
            from_uuid: "h1".into(),
            weight: 1.0,
            is_input: false,
            input_index: None,
            mean: 0.0,
            variance: 1.0,
            std_dev: 1.0,
            correlation_with_error: Some(0.5),
            weight_signal_count: None,
            proposed_weight_delta: None,
            mean_weight_sensitivity: None,
        }];
        let observations = empty_obs();
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
            incoming: &incoming,
            observations: &observations,
            ranked_sources: None,
            learning: None,
            backprop: &cfg,
            structural_only: false,
        };
        let mut rng = StdRng::seed_from_u64(1);
        let candidates = generate_candidates(&ctx, 4, &mut rng);
        let mean_err = candidates
            .iter()
            .find(|c| c.provenance.strategy == CandidateStrategy::MeanErrorBias)
            .expect("mean_error_bias candidate");
        let out_bias = mean_err
            .creature
            .neurons
            .iter()
            .find(|n| n.uuid == "o1")
            .unwrap()
            .bias;
        assert!((out_bias - 0.025).abs() < 1e-12);
    }

    #[test]
    fn saturated_mean_error_bias_is_skipped() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_error: Some(0.5),
            mean_adjusted_error: Some(0.0),
            mean_derivative: Some(0.0),
            saturation_fraction: 1.0,
            ..FocusNeuronStats::default()
        };
        let observations = empty_obs();
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
            incoming: &[],
            observations: &observations,
            ranked_sources: None,
            learning: None,
            backprop: &cfg,
            structural_only: false,
        };
        let mut rng = StdRng::seed_from_u64(1);
        assert!(build_candidate(&ctx, CandidateStrategy::MeanErrorBias, &mut rng).is_none());
    }

    #[test]
    fn backprop_without_a_learning_signal_proposes_nothing() {
        // Issue #83: a focus with no accumulated blame used to emit a
        // fallback bias step ~200x below the accepted scale — a strictly
        // worse duplicate of mean_error_bias. It must now be skipped.
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_error: Some(0.25),
            mean_abs_error: Some(0.25),
            mean_adjusted_error: Some(0.25),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        };
        let observations = empty_obs();
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
            incoming: &[],
            observations: &observations,
            ranked_sources: None,
            learning: None,
            backprop: &cfg,
            structural_only: false,
        };
        let mut rng = StdRng::seed_from_u64(3);
        assert!(build_candidate(&ctx, CandidateStrategy::Backprop, &mut rng).is_none());

        // An all-zero signal (blame never reached the focus) is the same case.
        let learning = LearningSignal::new(incumbent.neurons.len(), incumbent.synapses.len());
        let ctx = CandidateGenContext {
            learning: Some(&learning),
            ..ctx
        };
        assert!(build_candidate(&ctx, CandidateStrategy::Backprop, &mut rng).is_none());
    }

    #[test]
    fn backprop_proposes_from_an_accumulated_bias_signal() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        };
        let observations = empty_obs();
        let cfg = BackpropConfig::default();
        let mut learning = LearningSignal::new(incumbent.neurons.len(), incumbent.synapses.len());
        let focus_pos = incumbent
            .neurons
            .iter()
            .position(|n| n.uuid == "o1")
            .unwrap();
        learning.biases[focus_pos] = BiasSignal {
            count: 4.0,
            total_adjusted_bias: 2.0,
            no_change: false,
        };
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
            incoming: &[],
            observations: &observations,
            ranked_sources: None,
            learning: Some(&learning),
            backprop: &cfg,
            structural_only: false,
        };
        let mut rng = StdRng::seed_from_u64(3);
        let candidate = build_candidate(&ctx, CandidateStrategy::Backprop, &mut rng)
            .expect("backprop candidate from real blame");
        assert_eq!(candidate.provenance.strategy, CandidateStrategy::Backprop);
        let old = candidate.provenance.old_value.unwrap();
        let new = candidate.provenance.new_value.unwrap();
        assert!(
            new > old,
            "expected an upward bias step, got {old} -> {new}"
        );
    }

    /// Context over [`TINY`] with a bias signal of the given blame mass on `o1`.
    fn backprop_step_for(total_adjusted_bias: f64, learning_rate: f64) -> f64 {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        };
        let observations = empty_obs();
        let cfg = BackpropConfig {
            learning_rate,
            initial_learning_rate: learning_rate,
            ..BackpropConfig::default()
        };
        let mut learning = LearningSignal::new(incumbent.neurons.len(), incumbent.synapses.len());
        let focus_pos = incumbent
            .neurons
            .iter()
            .position(|n| n.uuid == "o1")
            .unwrap();
        learning.biases[focus_pos] = BiasSignal {
            count: 6.0,
            total_adjusted_bias,
            no_change: false,
        };
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
            incoming: &[],
            observations: &observations,
            ranked_sources: None,
            learning: Some(&learning),
            backprop: &cfg,
            structural_only: false,
        };
        let mut rng = StdRng::seed_from_u64(3);
        let candidate = build_candidate(&ctx, CandidateStrategy::Backprop, &mut rng)
            .expect("backprop candidate from real blame");
        candidate.provenance.new_value.unwrap() - candidate.provenance.old_value.unwrap()
    }

    /// Issue #75 arm 2: on production blame masses the bias step saturates
    /// `maximum_bias_adjustment_scale`, so `--backprop-learning-rate` cannot
    /// move it. The A/B has to vary the cap instead.
    #[test]
    fn a_saturating_blame_mass_pins_the_bias_step_to_the_cap_whatever_the_rate() {
        let cap = BackpropConfig::default().maximum_bias_adjustment_scale;
        // The GRQ focus carried mean |blame| ≈ 2.3e13 over 6 accumulations.
        let big = 6.0 * 2.3e13;
        let fast = backprop_step_for(big, 0.01);
        let slow = backprop_step_for(big, 0.001);
        assert_eq!(
            fast, slow,
            "a 10x smaller rate still produced the same step: {fast} vs {slow}"
        );
        assert!(
            (fast.abs() - cap).abs() < 1e-9,
            "expected the step pinned at the ±{cap} cap, got {fast}"
        );
        // Small blame is still rate-sensitive, so the knob is not inert per se.
        let small_fast = backprop_step_for(0.5, 0.01);
        let small_slow = backprop_step_for(0.5, 0.001);
        assert!(
            small_fast.abs() > small_slow.abs(),
            "small blame should scale with the rate: {small_fast} vs {small_slow}"
        );
    }

    /// Issue #75 arm 3: candidate generation has a fixed per-phase ceiling, so
    /// `--candidates` only binds below it. Raising it buys no extra proposals.
    #[test]
    fn raising_the_candidate_budget_above_the_generator_ceiling_adds_nothing() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        };
        let observations = empty_obs();
        let cfg = BackpropConfig::default();
        let ctx = skew_bias_ctx(&incumbent, &focus, &observations, &cfg, "o1");

        let generated = |count: usize| {
            let mut rng = StdRng::seed_from_u64(5);
            generate_candidates(&ctx, count, &mut rng).len()
        };
        let at_40 = generated(40);
        assert_eq!(at_40, generated(100), "40 and 100 must fill the same batch");
        assert_eq!(
            at_40,
            generated(150),
            "100 and 150 must fill the same batch"
        );
        assert!(
            at_40 < 40,
            "the ceiling should sit below the budget, got {at_40}"
        );
        // Below the ceiling the budget does bind.
        let small = generated(2);
        assert_eq!(small, 2, "a budget under the ceiling must cap the batch");
    }

    /// Observations whose single target has the given shape and mean→median gap.
    fn obs_with_target(
        mean: f64,
        median: f64,
        skewness: f64,
        excess_kurtosis: f64,
    ) -> ObservationsStatistics {
        let mut obs = empty_obs();
        obs.outputs.push(crate::observations::ScalarStats {
            count: 100,
            mean,
            variance: 1.0,
            std_dev: 1.0,
            min: -1.0,
            max: 5.0,
            zero_count: 0,
            non_zero_count: 100,
            non_finite_count: 0,
            mean_abs: mean.abs(),
            rms: 1.0,
            skewness,
            excess_kurtosis,
            quantiles: [median; 7],
        });
        obs
    }

    fn skew_bias_ctx<'a>(
        incumbent: &'a CreatureExport,
        focus: &'a FocusNeuronStats,
        observations: &'a ObservationsStatistics,
        backprop: &'a BackpropConfig,
        focus_uuid: &'a str,
    ) -> CandidateGenContext<'a> {
        CandidateGenContext {
            incumbent,
            focus_uuid,
            focus_stats: focus,
            incoming: &[],
            observations,
            ranked_sources: None,
            learning: None,
            backprop,
            structural_only: false,
        }
    }

    fn output_focus() -> FocusNeuronStats {
        FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        }
    }

    #[test]
    fn skew_bias_steps_the_output_towards_the_target_median() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = output_focus();
        // Right-skewed target: median 0.5 sits below mean 1.0.
        let observations = obs_with_target(1.0, 0.5, 1.2, 0.0);
        let cfg = BackpropConfig::default();
        let ctx = skew_bias_ctx(&incumbent, &focus, &observations, &cfg, "o1");
        let mut rng = StdRng::seed_from_u64(7);
        let candidate = build_candidate(&ctx, CandidateStrategy::StatsSkewBias, &mut rng)
            .expect("skew-bias candidate");
        let old = candidate.provenance.old_value.unwrap();
        let new = candidate.provenance.new_value.unwrap();
        // gap (−0.5) × fraction (0.25) × derivative (1.0), undamped.
        assert!(
            (new - old + 0.125).abs() < 1e-12,
            "expected a −0.125 bias step, got {}",
            new - old
        );
        assert_eq!(
            candidate.provenance.strategy,
            CandidateStrategy::StatsSkewBias
        );
    }

    #[test]
    fn skew_bias_follows_a_left_skewed_target_upwards() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = output_focus();
        // Left-skewed target: median 1.5 sits above mean 1.0.
        let observations = obs_with_target(1.0, 1.5, -1.2, 0.0);
        let cfg = BackpropConfig::default();
        let ctx = skew_bias_ctx(&incumbent, &focus, &observations, &cfg, "o1");
        let mut rng = StdRng::seed_from_u64(7);
        let candidate = build_candidate(&ctx, CandidateStrategy::StatsSkewBias, &mut rng)
            .expect("skew-bias candidate");
        let old = candidate.provenance.old_value.unwrap();
        let new = candidate.provenance.new_value.unwrap();
        assert!(new > old, "expected an upward step, got {old} -> {new}");
    }

    #[test]
    fn heavy_tails_damp_the_skew_bias_step() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = output_focus();
        let cfg = BackpropConfig::default();
        let mut rng = StdRng::seed_from_u64(7);

        let light = obs_with_target(1.0, 0.5, 1.2, 0.0);
        let ctx = skew_bias_ctx(&incumbent, &focus, &light, &cfg, "o1");
        let light_step = build_candidate(&ctx, CandidateStrategy::StatsSkewBias, &mut rng)
            .expect("light-tailed candidate");

        // Excess kurtosis of 3 (the reference) halves the step.
        let heavy = obs_with_target(1.0, 0.5, 1.2, 3.0);
        let ctx = skew_bias_ctx(&incumbent, &focus, &heavy, &cfg, "o1");
        let heavy_step = build_candidate(&ctx, CandidateStrategy::StatsSkewBias, &mut rng)
            .expect("heavy-tailed candidate");

        let delta =
            |c: &Candidate| c.provenance.new_value.unwrap() - c.provenance.old_value.unwrap();
        assert!(
            (delta(&heavy_step) - delta(&light_step) / 2.0).abs() < 1e-12,
            "heavy tails should halve the step: {} vs {}",
            delta(&heavy_step),
            delta(&light_step)
        );
    }

    #[test]
    fn a_symmetric_target_produces_no_skew_bias_candidate() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = output_focus();
        let observations = obs_with_target(1.0, 0.5, 0.05, 0.0);
        let cfg = BackpropConfig::default();
        let ctx = skew_bias_ctx(&incumbent, &focus, &observations, &cfg, "o1");
        let mut rng = StdRng::seed_from_u64(7);
        assert!(build_candidate(&ctx, CandidateStrategy::StatsSkewBias, &mut rng).is_none());
    }

    #[test]
    fn a_hidden_focus_produces_no_skew_bias_candidate() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "h1".into(),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        };
        let observations = obs_with_target(1.0, 0.5, 1.2, 0.0);
        let cfg = BackpropConfig::default();
        let ctx = skew_bias_ctx(&incumbent, &focus, &observations, &cfg, "h1");
        let mut rng = StdRng::seed_from_u64(7);
        assert!(build_candidate(&ctx, CandidateStrategy::StatsSkewBias, &mut rng).is_none());
    }

    #[test]
    fn a_saturated_focus_produces_no_skew_bias_candidate() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_derivative: Some(0.0),
            saturation_fraction: 1.0,
            ..FocusNeuronStats::default()
        };
        let observations = obs_with_target(1.0, 0.5, 1.2, 0.0);
        let cfg = BackpropConfig::default();
        let ctx = skew_bias_ctx(&incumbent, &focus, &observations, &cfg, "o1");
        let mut rng = StdRng::seed_from_u64(7);
        assert!(build_candidate(&ctx, CandidateStrategy::StatsSkewBias, &mut rng).is_none());
    }

    #[test]
    fn skew_bias_is_offered_in_a_generated_population() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let focus = output_focus();
        let observations = obs_with_target(1.0, 0.5, 1.2, 0.0);
        let cfg = BackpropConfig::default();
        let ctx = skew_bias_ctx(&incumbent, &focus, &observations, &cfg, "o1");
        let mut rng = StdRng::seed_from_u64(11);
        let population = generate_candidates(&ctx, 12, &mut rng);
        assert!(
            population
                .iter()
                .any(|c| c.provenance.strategy == CandidateStrategy::StatsSkewBias),
            "generated population never offered a skew-aware bias candidate"
        );
    }

    const TWO_INPUT: &str = r#"{
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

    fn obs_two_input(c0: f64, c1: f64) -> ObservationsStatistics {
        let mut obs = empty_obs();
        obs.input_count = 2;
        obs.inputs.push(crate::observations::ScalarStats {
            count: 1,
            mean: 0.0,
            variance: 1.0,
            std_dev: 1.0,
            min: 0.0,
            max: 0.0,
            zero_count: 0,
            non_zero_count: 0,
            non_finite_count: 0,
            mean_abs: 0.0,
            rms: 0.0,
            skewness: 0.0,
            excess_kurtosis: 0.0,
            quantiles: [0.0; 7],
        });
        obs.input_target_correlations = vec![c0, c1];
        obs
    }

    #[test]
    fn structural_add_picks_highest_target_correlation() {
        let incumbent = parse_creature_json(TWO_INPUT).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_error: Some(0.1),
            mean_adjusted_error: Some(0.1),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        };
        let observations = obs_two_input(0.05, 0.8);
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
            incoming: &[],
            observations: &observations,
            ranked_sources: None,
            learning: None,
            backprop: &cfg,
            structural_only: false,
        };
        let mut rng = StdRng::seed_from_u64(3);
        let cand = build_candidate(&ctx, CandidateStrategy::StructuralAdd, &mut rng).unwrap();
        assert!(
            cand.provenance.mutation.contains("input-1"),
            "mutation={}",
            cand.provenance.mutation
        );
        assert!(
            cand.creature
                .synapses
                .iter()
                .any(|s| s.from_uuid == "input-1" && s.to_uuid == "o1")
        );
    }

    #[test]
    fn structural_add_neuron_grows_hidden_bridge() {
        use neat_core::compile_creature;
        let incumbent = parse_creature_json(TWO_INPUT).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_error: Some(-0.2),
            mean_adjusted_error: Some(-0.2),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        };
        let observations = obs_two_input(0.1, 0.7);
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
            incoming: &[],
            observations: &observations,
            ranked_sources: None,
            learning: None,
            backprop: &cfg,
            structural_only: false,
        };
        let mut rng = StdRng::seed_from_u64(9);
        let cand = build_candidate(&ctx, CandidateStrategy::StructuralAddNeuron, &mut rng).unwrap();
        assert_eq!(
            cand.provenance.strategy,
            CandidateStrategy::StructuralAddNeuron
        );
        assert!(cand.provenance.mutation.contains("add-neuron"));
        assert_eq!(cand.creature.neurons.len(), incumbent.neurons.len() + 1);
        let grown = cand
            .creature
            .neurons
            .iter()
            .find(|n| n.neuron_type == "hidden" && n.uuid != "h1")
            .expect("grown hidden");
        // Negative residual → ABSOLUTE first, with negative w_out into the focus.
        assert_eq!(grown.squash.as_deref(), Some("ABSOLUTE"));
        let w_out = cand
            .creature
            .synapses
            .iter()
            .find(|s| s.from_uuid == grown.uuid && s.to_uuid == "o1")
            .map(|s| s.weight)
            .expect("outbound synapse");
        assert!(
            w_out < 0.0,
            "ABSOLUTE correcting negative residual needs negative w_out, got {w_out}"
        );
        assert!(cand.provenance.mutation.contains("squash=ABSOLUTE"));
        compile_creature(&cand.creature).expect("grown creature must compile");
    }

    #[test]
    fn four_candidates_include_structural_strategies() {
        let incumbent = parse_creature_json(TWO_INPUT).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_error: Some(0.1),
            mean_adjusted_error: Some(0.1),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        };
        let incoming = [IncomingSourceStats {
            synapse_index: 1,
            from_uuid: "h1".into(),
            weight: 1.0,
            is_input: false,
            input_index: None,
            mean: 0.0,
            variance: 1.0,
            std_dev: 1.0,
            correlation_with_error: Some(0.4),
            weight_signal_count: None,
            proposed_weight_delta: None,
            mean_weight_sensitivity: None,
        }];
        let observations = obs_two_input(0.2, 0.6);
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
            incoming: &incoming,
            observations: &observations,
            ranked_sources: None,
            learning: None,
            backprop: &cfg,
            structural_only: false,
        };
        let mut rng = StdRng::seed_from_u64(1);
        let candidates = generate_candidates(&ctx, 6, &mut rng);
        let strategies: Vec<_> = candidates.iter().map(|c| c.provenance.strategy).collect();
        assert!(strategies.contains(&CandidateStrategy::StructuralAdd));
        assert!(strategies.contains(&CandidateStrategy::StructuralAddNeuron));

        let mut rng = StdRng::seed_from_u64(1);
        let mut ctx_struct = ctx;
        ctx_struct.structural_only = true;
        let structural = generate_candidates(&ctx_struct, 10, &mut rng);
        assert!(!structural.is_empty());
        assert!(structural.iter().all(|c| matches!(
            c.provenance.strategy,
            CandidateStrategy::StructuralAdd | CandidateStrategy::StructuralAddNeuron
        )));
        let neuron_squashes: std::collections::BTreeSet<_> = structural
            .iter()
            .filter(|c| c.provenance.strategy == CandidateStrategy::StructuralAddNeuron)
            .filter_map(|c| {
                c.creature
                    .neurons
                    .iter()
                    .find(|n| n.neuron_type == "hidden" && n.uuid != "h1")
                    .and_then(|n| n.squash.clone())
            })
            .collect();
        assert!(
            neuron_squashes.len() >= 3,
            "expected multiple squashes, got {neuron_squashes:?}"
        );
        // Signed residual reorders ABSOLUTE ahead of the Tier‑1 defaults.
        assert!(neuron_squashes.contains("ABSOLUTE"));
        assert!(
            neuron_squashes.contains("GELU")
                || neuron_squashes.contains("Swish")
                || neuron_squashes.contains("TANH")
        );
    }

    const ORPHAN_HIDDEN: &str = r#"{
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
    fn generate_candidates_hooks_previous_hidden() {
        use neat_core::compile_creature;
        let incumbent = parse_creature_json(ORPHAN_HIDDEN).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_error: Some(-0.2),
            mean_adjusted_error: Some(-0.2),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        };
        // Strong input corr so top prior is an input; activation probes score h1.
        let mut observations = obs_two_input(0.2, 0.85);
        // Give outputs a range so synthetic probes have targets.
        observations.output_count = 1;
        observations.outputs.push(crate::observations::ScalarStats {
            count: 10,
            mean: 0.0,
            variance: 0.25,
            std_dev: 0.5,
            min: -1.0,
            max: 1.0,
            zero_count: 0,
            non_zero_count: 10,
            non_finite_count: 0,
            mean_abs: 0.5,
            rms: 0.5,
            skewness: 0.0,
            excess_kurtosis: 0.0,
            quantiles: [0.0; 7],
        });
        let mut network = compile_creature(&incumbent).unwrap();
        let prior = rank_unused_sources(&incumbent, "o1", &observations);
        let mut probe_rng = StdRng::seed_from_u64(3);
        let probes = synthetic_observation_probes(&observations, 2, 1, 32, &mut probe_rng);
        let ranked =
            refine_sources_from_probes(&incumbent, &mut network, "o1", &prior, &probes).unwrap();
        let h1 = ranked
            .iter()
            .find(|r| r.from_uuid == "h1")
            .expect("h1 ranked");
        assert_ne!(
            h1.score, 0.05,
            "hidden score must be calculated, not a constant"
        );

        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
            incoming: &[],
            observations: &observations,
            ranked_sources: Some(&ranked),
            learning: None,
            backprop: &cfg,
            structural_only: true,
        };
        let mut rng = StdRng::seed_from_u64(11);
        let structural = generate_candidates(&ctx, 12, &mut rng);
        assert!(
            structural.iter().any(|c| {
                c.provenance.strategy == CandidateStrategy::StructuralAdd
                    && c.provenance.mutation.contains("h1")
            }),
            "expected a direct structural-add of h1 into the focus; mutations={:?}",
            structural
                .iter()
                .map(|c| c.provenance.mutation.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn generate_candidates_tries_absolute_first_on_negative_residual() {
        let incumbent = parse_creature_json(TWO_INPUT).unwrap();
        let focus = FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_error: Some(-0.3),
            mean_adjusted_error: Some(-0.3),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        };
        let observations = obs_two_input(0.2, 0.8);
        let cfg = BackpropConfig::default();
        let ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: "o1",
            focus_stats: &focus,
            incoming: &[],
            observations: &observations,
            ranked_sources: None,
            learning: None,
            backprop: &cfg,
            structural_only: true,
        };
        let mut rng = StdRng::seed_from_u64(2);
        let structural = generate_candidates(&ctx, 8, &mut rng);
        let first_neuron = structural
            .iter()
            .find(|c| c.provenance.strategy == CandidateStrategy::StructuralAddNeuron)
            .expect("expected a neuron-growth candidate");
        assert!(
            first_neuron.provenance.mutation.contains("squash=ABSOLUTE"),
            "mutation={}",
            first_neuron.provenance.mutation
        );
    }
}
