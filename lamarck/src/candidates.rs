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
use crate::tags::{CreatureMeta, serialize_creature_with_meta_compact};
use neat_core::{CreatureExport, creature_to_json};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

/// Strategy label recorded in the experiment journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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

impl CandidateStrategy {
    /// Journal / log name of the strategy (the `serde` snake-case spelling).
    pub fn label(self) -> &'static str {
        match self {
            CandidateStrategy::Backprop => "backprop",
            CandidateStrategy::MeanErrorBias => "mean_error_bias",
            CandidateStrategy::StatsWeight => "stats_weight",
            CandidateStrategy::StatsBias => "stats_bias",
            CandidateStrategy::StatsSkewBias => "stats_skew_bias",
            CandidateStrategy::StructuralAdd => "structural_add",
            CandidateStrategy::StructuralAddNeuron => "structural_add_neuron",
            CandidateStrategy::StructuralWeaken => "structural_weaken",
            CandidateStrategy::Random => "random",
        }
    }
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

/// Weight/bias strategies swept round-robin once the structural phases are done.
const FILL_STRATEGIES: [CandidateStrategy; 8] = [
    CandidateStrategy::Backprop,
    CandidateStrategy::StatsWeight,
    CandidateStrategy::StructuralAdd,
    CandidateStrategy::StructuralAddNeuron,
    CandidateStrategy::StatsBias,
    CandidateStrategy::StatsSkewBias,
    CandidateStrategy::StructuralWeaken,
    CandidateStrategy::Random,
];

/// Candidates the round-robin fill may contribute per strategy (issue #119).
///
/// The cap counts candidates that *joined* the batch, not proposals offered, so
/// a rejected duplicate frees its slot for the next strategy instead of
/// shrinking the batch.
const FILL_PER_STRATEGY: usize = 3;

/// Scaled-quota per-round quota: synapse adds swept from the ranked grid.
const ADDS_PER_ROUND: usize = 4;
/// Scaled-quota per-round quota: neuron growths swept from the ranked × squash grid.
const NEURONS_PER_ROUND: usize = 3;
/// Scaled-quota per-round quota: neuron growths from the hidden-first ordering.
const HIDDEN_NEURONS_PER_ROUND: usize = 2;
/// Weight scales (× [`OLS_WEIGHT_FRACTION`]) swept per ranked source.
///
/// The grid of ranked sources × these scales is what makes a scaled batch
/// finite: once it is consumed there is no further synapse add to propose.
const ADD_SCALE_STEPS: [f64; 4] = [1.0, 2.0, 0.5, 4.0];

/// Requested size of one candidate batch (issue #108).
#[derive(Debug, Clone, Copy)]
pub struct CandidateBudget {
    /// Requested candidate count.
    pub count: usize,
    /// When true the per-phase quotas scale with `count`, so the budget binds
    /// until the generator genuinely runs out of proposals.
    ///
    /// When false the pre-#108 fixed quotas apply and the batch tops out at
    /// their ceiling (~33 distinct candidates on the production creature)
    /// whatever `count` says.
    pub scale_quotas: bool,
}

/// Why a candidate batch stopped growing (issue #108).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchLimit {
    /// The requested candidate count was reached.
    Budget,
    /// Every ranked source and squash was consumed and no strategy proposed
    /// anything new — the generator has genuinely run out.
    Exhausted,
    /// The fixed pre-#108 per-phase quotas ran out below the budget.
    QuotaCeiling,
}

impl BatchLimit {
    /// Short label for the run log.
    pub fn label(self) -> &'static str {
        match self {
            BatchLimit::Budget => "budget reached",
            BatchLimit::Exhausted => "generator exhausted",
            BatchLimit::QuotaCeiling => "fixed quota ceiling",
        }
    }
}

/// One generated batch plus why it stopped.
#[derive(Debug, Clone)]
pub struct CandidateBatch {
    /// Generated candidates, in proposal order.
    pub candidates: Vec<Candidate>,
    /// Why generation stopped.
    pub limit: BatchLimit,
}

impl CandidateBatch {
    /// Strategy mix of the batch — how many candidates each family contributed.
    pub fn strategy_mix(&self) -> BTreeMap<CandidateStrategy, usize> {
        strategy_mix(&self.candidates)
    }

    /// Strategy mix rendered for the run log, e.g. `structural_add=6 random=3`.
    pub fn strategy_mix_summary(&self) -> String {
        strategy_mix_summary(&self.candidates)
    }
}

/// Strategy mix of any candidate slice — how many each family contributed.
///
/// Takes a slice rather than a batch so a run that merged several per-focus
/// batches into one population can still report its mix (issue #109).
pub fn strategy_mix(candidates: &[Candidate]) -> BTreeMap<CandidateStrategy, usize> {
    let mut mix = BTreeMap::new();
    for candidate in candidates {
        *mix.entry(candidate.provenance.strategy).or_insert(0) += 1;
    }
    mix
}

/// Strategy mix rendered for the run log, e.g. `structural_add=6 random=3`.
pub fn strategy_mix_summary(candidates: &[Candidate]) -> String {
    strategy_mix(candidates)
        .into_iter()
        .map(|(strategy, n)| format!("{}={n}", strategy.label()))
        .collect::<Vec<_>>()
        .join(" ")
}

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

/// Generate a candidate population under the pre-#108 fixed quotas.
///
/// Kept as the legacy entry point: the batch tops out at the fixed per-phase
/// ceiling whatever `count` says. Duplicate proposals are rejected here too
/// (issue #119), so the ceiling is a count of distinct hypotheses. Use
/// [`generate_candidate_batch`] with [`CandidateBudget::scale_quotas`] to let
/// the budget bind.
pub fn generate_candidates(
    ctx: &CandidateGenContext<'_>,
    count: usize,
    rng: &mut impl Rng,
) -> Vec<Candidate> {
    generate_candidate_batch(
        ctx,
        CandidateBudget {
            count,
            scale_quotas: false,
        },
        rng,
    )
    .candidates
}

/// Generate a candidate population without mutating the incumbent.
///
/// The opening phases are the pre-#108 fixed quotas, and the round-robin fill
/// after them contributes at most three *accepted* candidates
/// per strategy — a duplicate frees its slot for the next strategy rather than
/// shrinking the batch (issue #119). When [`CandidateBudget::scale_quotas`] is
/// set, generation then
/// keeps sweeping the ranked-source × weight-scale and ranked-source × squash
/// grids — a slice of every family per round, so the strategy mix stays
/// proportional — until the budget is met or nothing new can be proposed.
/// Each grid is visited in weighted-random order ([`weighted_slot_order`]):
/// best guesses almost surely first, every pairing a nonzero chance per batch.
pub fn generate_candidate_batch(
    ctx: &CandidateGenContext<'_>,
    budget: CandidateBudget,
    rng: &mut impl Rng,
) -> CandidateBatch {
    let mut batch = Batch::new(ctx.incumbent, budget.count);
    if budget.count == 0 {
        return batch.finish(BatchLimit::Budget);
    }

    let ranked = ranked_for(ctx);
    let hidden_first = with_previous_hidden_first(&ranked);
    let growth_squashes = growth_squashes_for(ctx.focus_stats, Some(ctx.observations));
    let hidden_growth = hidden_first.as_ref().filter(|hid_ranked| {
        // Also grow via a previous hidden when the top residual source is an input.
        hid_ranked
            .first()
            .is_some_and(|s| !crate::structural::is_input_source(&s.from_uuid))
            && ranked
                .first()
                .is_some_and(|s| crate::structural::is_input_source(&s.from_uuid))
    });

    fill_opening(
        ctx,
        &mut batch,
        &ranked,
        hidden_growth,
        &growth_squashes,
        rng,
    );

    if !budget.scale_quotas {
        let limit = if batch.is_full() {
            BatchLimit::Budget
        } else {
            BatchLimit::QuotaCeiling
        };
        return batch.finish(limit);
    }
    if batch.is_full() {
        // The opening filled the budget — don't spend rng draws ordering
        // grids no round will visit.
        return batch.finish(BatchLimit::Budget);
    }

    // --- Scaled quotas (issue #108) ---
    // Each grid is visited in weighted-random order rather than strict rank
    // order: the best guesses almost surely come first, but every pairing
    // keeps a nonzero chance of an early draw, so repeated experiments cover
    // the whole grid in expectation — with no cross-run cursor to invalidate
    // when the incumbent changes.
    let mut cursors = GridCursors {
        add: 0,
        neuron: 0,
        hidden: 0,
        add_order: weighted_slot_order(
            ranked.len() * ADD_SCALE_STEPS.len(),
            |slot| ranked[slot % ranked.len()].score / (1 + slot / ranked.len()) as f64,
            rng,
        ),
        neuron_order: weighted_slot_order(
            ranked.len() * growth_squashes.len(),
            |slot| ranked[slot % ranked.len()].score / (1 + slot / ranked.len()) as f64,
            rng,
        ),
        hidden_order: hidden_growth.map_or_else(Vec::new, |h| {
            weighted_slot_order(
                h.len() * growth_squashes.len(),
                |slot| h[slot % h.len()].score / (1 + slot / h.len()) as f64,
                rng,
            )
        }),
    };
    let limit = loop {
        if batch.is_full() {
            break BatchLimit::Budget;
        }
        let productive = fill_round(
            ctx,
            &mut batch,
            &mut cursors,
            &ranked,
            hidden_growth,
            &growth_squashes,
            rng,
        );
        // A barren round is only exhaustion once every grid is consumed too:
        // a round can propose nothing but duplicates while sources remain.
        if !productive && !cursors.structural_remaining() {
            break BatchLimit::Exhausted;
        }
    };
    batch.finish(limit)
}

/// Fixed opening phases — identical to the pre-#108 generator.
fn fill_opening(
    ctx: &CandidateGenContext<'_>,
    batch: &mut Batch,
    ranked: &[RankedSource],
    hidden_growth: Option<&Vec<RankedSource>>,
    growth_squashes: &[&str],
    rng: &mut impl Rng,
) {
    // Synapse add, then one neuron per squash (order follows residual shape),
    // then additional synapse scales to fill the budget.
    batch.push(build_structural_add_scaled(
        ctx,
        ranked,
        0,
        OLS_WEIGHT_FRACTION,
    ));
    // Explicitly try hooking an unused previous hidden into the focus even when
    // its probe score is still zero (unmeasured prior).
    if let Some(hid_i) = first_previous_hidden_index(ranked)
        && !batch.is_full()
    {
        let already_added = batch.candidates().iter().any(|c| {
            c.provenance.strategy == CandidateStrategy::StructuralAdd
                && c.provenance.mutation.contains(&ranked[hid_i].from_uuid)
        });
        if !already_added {
            batch.push(build_structural_add_scaled_gated(
                ctx,
                ranked,
                hid_i,
                OLS_WEIGHT_FRACTION,
                false,
            ));
        }
    }
    for (squash_i, &squash) in growth_squashes.iter().enumerate() {
        if batch.is_full() {
            break;
        }
        // Under mixed mode, only emit a few squashes so weight strategies still fit.
        if !ctx.structural_only && squash_i >= 3 {
            break;
        }
        batch.push(build_structural_add_neuron_combo(
            ctx, ranked, rng, squash, 0,
        ));
    }
    if let Some(hid_ranked) = hidden_growth {
        for (squash_i, &squash) in growth_squashes.iter().enumerate() {
            if batch.is_full() {
                break;
            }
            if !ctx.structural_only && squash_i >= 2 {
                break;
            }
            batch.push(build_structural_add_neuron_combo(
                ctx, hid_ranked, rng, squash, 0,
            ));
        }
    }
    for (idx, scale) in [
        (0usize, OLS_WEIGHT_FRACTION * 2.0),
        (1, OLS_WEIGHT_FRACTION),
        (2, OLS_WEIGHT_FRACTION),
        (3, OLS_WEIGHT_FRACTION),
    ] {
        if batch.is_full() {
            break;
        }
        batch.push(build_structural_add_scaled(ctx, ranked, idx, scale));
    }
    if ctx.structural_only {
        // Keep filling with remaining residual-ordered squashes / synapse adds.
        let mut squash_i = 0usize;
        let mut syn_i = 0usize;
        let n_squash = growth_squashes.len().max(1);
        while !batch.is_full() && squash_i + syn_i < n_squash * 3 {
            if squash_i <= syn_i {
                let squash = growth_squashes[squash_i % n_squash];
                squash_i += 1;
                batch.push(build_structural_add_neuron_combo(
                    ctx, ranked, rng, squash, 0,
                ));
            } else {
                syn_i += 1;
                batch.push(build_structural_add_scaled(
                    ctx,
                    ranked,
                    syn_i,
                    OLS_WEIGHT_FRACTION,
                ));
            }
        }
        return;
    }

    let has_error =
        ctx.focus_stats.mean_adjusted_error.is_some() || ctx.focus_stats.mean_error.is_some();
    if has_error && !batch.is_full() {
        batch.push(build_candidate(ctx, CandidateStrategy::MeanErrorBias, rng));
    }

    // Round-robin fill. The budget counts candidates that joined the batch, so
    // a duplicate (or a strategy with nothing to offer) passes its slot to the
    // next strategy rather than costing the batch a hypothesis (issue #119).
    // A whole sweep that adds nothing means every strategy is spent — stop
    // there rather than spinning on proposals the batch already holds.
    let fill_budget = FILL_STRATEGIES.len() * FILL_PER_STRATEGY;
    let mut strategy_i = 0usize;
    let mut filled = 0usize;
    let mut barren = 0usize;
    while !batch.is_full() && filled < fill_budget && barren < FILL_STRATEGIES.len() {
        let strategy = FILL_STRATEGIES[strategy_i % FILL_STRATEGIES.len()];
        strategy_i += 1;
        if batch.push(build_candidate(ctx, strategy, rng)).accepted() {
            filled += 1;
            barren = 0;
        } else {
            barren += 1;
        }
    }
}

/// Floor applied to grid-slot weights so an unmeasured source (probe score
/// still zero) keeps a nonzero chance of being drawn.
const GRID_WEIGHT_FLOOR: f64 = 1e-4;

/// Weighted random visiting order over `n` grid slots, without replacement.
///
/// An exponential race (Efraimidis–Spirakis): slot `i` finishes at
/// `-ln(U)/w_i`, and slots are visited by finish time. Heavier slots almost
/// surely finish early — so the batch still tries the obvious pairings first —
/// while light ones retain probability `w_i / Σw` per draw of jumping the
/// queue. Deterministic for a given seed.
fn weighted_slot_order(
    n: usize,
    weight: impl Fn(usize) -> f64,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let mut keyed: Vec<(f64, usize)> = (0..n)
        .map(|slot| {
            let w = weight(slot).max(GRID_WEIGHT_FLOOR);
            let u = rng.random::<f64>().max(f64::MIN_POSITIVE);
            (-u.ln() / w, slot)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
    keyed.into_iter().map(|(_, slot)| slot).collect()
}

/// How far each scaled-quota grid has been swept, in its weighted-random
/// visiting order.
struct GridCursors {
    add: usize,
    neuron: usize,
    hidden: usize,
    add_order: Vec<usize>,
    neuron_order: Vec<usize>,
    hidden_order: Vec<usize>,
}

impl GridCursors {
    /// True while a ranked source / squash pairing is still unproposed.
    fn structural_remaining(&self) -> bool {
        self.add < self.add_order.len()
            || self.neuron < self.neuron_order.len()
            || self.hidden < self.hidden_order.len()
    }
}

/// One scaled-quota round: a slice of every family, so no strategy monopolises
/// the extra budget. Returns whether the round added anything new.
fn fill_round(
    ctx: &CandidateGenContext<'_>,
    batch: &mut Batch,
    cursors: &mut GridCursors,
    ranked: &[RankedSource],
    hidden_growth: Option<&Vec<RankedSource>>,
    growth_squashes: &[&str],
    rng: &mut impl Rng,
) -> bool {
    let mut productive = false;
    for _ in 0..ADDS_PER_ROUND {
        if batch.is_full() || cursors.add >= cursors.add_order.len() {
            break;
        }
        let slot = cursors.add_order[cursors.add];
        cursors.add += 1;
        let source_index = slot % ranked.len();
        let scale = ADD_SCALE_STEPS[slot / ranked.len()] * OLS_WEIGHT_FRACTION;
        productive |= batch
            .push(build_structural_add_scaled(
                ctx,
                ranked,
                source_index,
                scale,
            ))
            .accepted();
    }
    for _ in 0..NEURONS_PER_ROUND {
        if batch.is_full() || cursors.neuron >= cursors.neuron_order.len() {
            break;
        }
        let slot = cursors.neuron_order[cursors.neuron];
        cursors.neuron += 1;
        let squash = growth_squashes[slot / ranked.len()];
        productive |= batch
            .push(build_structural_add_neuron_combo(
                ctx,
                ranked,
                rng,
                squash,
                slot % ranked.len(),
            ))
            .accepted();
    }
    if let Some(hid_ranked) = hidden_growth {
        for _ in 0..HIDDEN_NEURONS_PER_ROUND {
            if batch.is_full() || cursors.hidden >= cursors.hidden_order.len() {
                break;
            }
            let slot = cursors.hidden_order[cursors.hidden];
            cursors.hidden += 1;
            let squash = growth_squashes[slot / hid_ranked.len()];
            productive |= batch
                .push(build_structural_add_neuron_combo(
                    ctx,
                    hid_ranked,
                    rng,
                    squash,
                    slot % hid_ranked.len(),
                ))
                .accepted();
        }
    }
    if !ctx.structural_only {
        for strategy in FILL_STRATEGIES {
            if batch.is_full() {
                break;
            }
            productive |= batch.push(build_candidate(ctx, strategy, rng)).accepted();
        }
    }
    productive
}

/// What became of a proposal offered to the batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Proposal {
    /// Joined the batch.
    Accepted,
    /// Structurally identical to a candidate the batch already holds.
    Duplicate,
    /// The strategy had nothing to offer, or the batch was already full.
    Nothing,
}

impl Proposal {
    /// True when the proposal joined the batch.
    fn accepted(self) -> bool {
        self == Proposal::Accepted
    }
}

/// Accumulator that enforces the budget and rejects duplicate proposals, so a
/// "filled" batch is never padded with repeats (issues #108, #119).
struct Batch {
    out: Vec<Candidate>,
    seen: HashSet<u64>,
    incumbent_uuids: HashSet<String>,
    count: usize,
}

impl Batch {
    fn new(incumbent: &CreatureExport, count: usize) -> Self {
        Self {
            out: Vec::with_capacity(count.min(1024)),
            seen: HashSet::new(),
            incumbent_uuids: incumbent.neurons.iter().map(|n| n.uuid.clone()).collect(),
            count,
        }
    }

    fn is_full(&self) -> bool {
        self.out.len() >= self.count
    }

    fn candidates(&self) -> &[Candidate] {
        &self.out
    }

    /// Offer a proposal to the batch, reporting what became of it.
    fn push(&mut self, candidate: Option<Candidate>) -> Proposal {
        let Some(candidate) = candidate else {
            return Proposal::Nothing;
        };
        if self.is_full() {
            return Proposal::Nothing;
        }
        if !self.seen.insert(self.fingerprint(&candidate.creature)) {
            return Proposal::Duplicate;
        }
        self.out.push(candidate);
        Proposal::Accepted
    }

    /// Structural fingerprint of a candidate creature.
    ///
    /// Neurons a mutation grew carry a fresh random UUID, so they are keyed by
    /// position instead: two identical bridges must collide rather than pass as
    /// distinct proposals.
    fn fingerprint(&self, creature: &CreatureExport) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut grown: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for (i, neuron) in creature.neurons.iter().enumerate() {
            if !self.incumbent_uuids.contains(&neuron.uuid) {
                grown.insert(neuron.uuid.as_str(), i);
            }
        }
        let mut hasher = DefaultHasher::new();
        let hash_uuid = |uuid: &str, hasher: &mut DefaultHasher| match grown.get(uuid) {
            Some(position) => ("grown", position).hash(hasher),
            None => ("existing", uuid).hash(hasher),
        };
        for neuron in &creature.neurons {
            hash_uuid(&neuron.uuid, &mut hasher);
            neuron.neuron_type.hash(&mut hasher);
            neuron.squash.hash(&mut hasher);
            neuron.bias.to_bits().hash(&mut hasher);
        }
        for synapse in &creature.synapses {
            hash_uuid(&synapse.from_uuid, &mut hasher);
            hash_uuid(&synapse.to_uuid, &mut hasher);
            synapse.weight.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    fn finish(self, limit: BatchLimit) -> CandidateBatch {
        CandidateBatch {
            candidates: self.out,
            limit,
        }
    }
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

/// Grow a hidden that combines two adjacent residual sources into the focus.
///
/// `source_index` selects the leading source: `0` is the top-ranked pair, and
/// scaled quotas (#108) sweep further down the ranking for extra proposals.
fn build_structural_add_neuron_combo(
    ctx: &CandidateGenContext<'_>,
    ranked: &[RankedSource],
    rng: &mut impl Rng,
    squash: &str,
    source_index: usize,
) -> Option<Candidate> {
    let focus_uuid = ctx.focus_uuid;
    let a = ranked.get(source_index)?;
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
    let next = ranked.get(source_index + 1);
    let mut second = None;
    if let Some(b) = next.filter(|b| b.score >= MIN_NEURON_BRIDGE_SCORE * 0.5) {
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
            next.map(|s| s.score).unwrap_or(0.0),
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
            if let Some(cand) = build_structural_add_neuron_combo(ctx, &ranked, rng, squash, 0) {
                return Some(cand);
            }
            // Prefer reusing a previous hidden when the top residual source is an input.
            if let Some(hid_ranked) = with_previous_hidden_first(&ranked)
                && let Some(cand) =
                    build_structural_add_neuron_combo(ctx, &hid_ranked, rng, squash, 0)
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
///
/// Every file here is **compact** JSON (issue #114): `rust_scorer` is its only
/// consumer, and on the production creature the pretty-printer's indentation is
/// about a third of the ~90 MB a batch writes and the scorer then parses.
/// Human-facing artefacts — `best.json`, `winners/` — stay pretty.
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
        serialize_creature_with_meta_compact(incumbent, meta)?
    } else {
        creature_to_json(incumbent).map_err(|e| e.to_string())?
    };
    fs::write(dir.join("baseline.json"), baseline).map_err(|e| e.to_string())?;

    let mut stems = vec!["baseline".to_string()];
    for (i, candidate) in candidates.iter().enumerate() {
        let stem = format!("candidate-{i:03}");
        let json = creature_to_json(&candidate.creature).map_err(|e| e.to_string())?;
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
    use neat_core::{creature_to_json_pretty, parse_creature_json};
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

    /// Step a `backprop` candidate proposes at the port-default bias cap.
    fn backprop_step_for(total_adjusted_bias: f64, learning_rate: f64) -> f64 {
        backprop_step_capped(
            total_adjusted_bias,
            learning_rate,
            BackpropConfig::default().maximum_bias_adjustment_scale,
        )
    }

    /// Context over [`TINY`] with a bias signal of the given blame mass on `o1`.
    fn backprop_step_capped(total_adjusted_bias: f64, learning_rate: f64, cap: f64) -> f64 {
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
            maximum_bias_adjustment_scale: cap,
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

    /// Issue #96 item 3: the cap *is* the knob the #75 A/B needed. On the same
    /// saturating blame mass, lowering `maximum_bias_adjustment_scale` resizes
    /// the proposed bias step in step with it — down to the `1e-6` accept bar
    /// the ±10 default overshoots by ~7 orders of magnitude.
    #[test]
    fn lowering_the_bias_cap_resizes_the_saturated_backprop_step() {
        // The GRQ focus carried mean |blame| ≈ 2.3e13 over 6 accumulations.
        let big = 6.0 * 2.3e13;
        for cap in [10.0, 0.01, 1e-6] {
            let step = backprop_step_capped(big, 0.01, cap);
            assert!(
                (step.abs() - cap).abs() < cap * 1e-9,
                "expected the step pinned at the ±{cap} cap, got {step}"
            );
        }
        // Direction is a property of the blame, not of the cap.
        assert!(
            backprop_step_capped(big, 0.01, 1e-6) * backprop_step_capped(big, 0.01, 10.0) > 0.0,
            "shrinking the cap must not flip the step's direction"
        );
    }

    /// Issue #75 arm 3: candidate generation has a fixed per-phase ceiling, so
    /// `--candidates` only binds below it. Raising it buys no extra proposals.
    ///
    /// Issue #108 kept this as the **default** path: the scaled quotas that let
    /// the budget bind are opt-in until the paired production benchmark runs,
    /// so an unchanged production config still tops out here.
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

        // The under-filled batch names the fixed quotas, not exhaustion.
        let mut rng = StdRng::seed_from_u64(5);
        let batch = generate_candidate_batch(
            &ctx,
            CandidateBudget {
                count: 100,
                scale_quotas: false,
            },
            &mut rng,
        );
        assert_eq!(batch.limit, BatchLimit::QuotaCeiling);
    }

    /// Creature with `inputs` inputs, of which only `input-0` is wired — the
    /// rest are unused ranked sources for the generator to propose from.
    fn wide_creature(inputs: usize) -> CreatureExport {
        let json = format!(
            r#"{{
              "semanticVersion": "4.0.0",
              "forwardOnly": true,
              "input": {inputs},
              "output": 1,
              "neurons": [
                {{"type":"hidden","uuid":"h1","bias":0.1,"squash":"IDENTITY"}},
                {{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}}
              ],
              "synapses": [
                {{"fromUUID":"input-0","toUUID":"h1","weight":1.0}},
                {{"fromUUID":"h1","toUUID":"o1","weight":1.0}}
              ]
            }}"#
        );
        parse_creature_json(&json).unwrap()
    }

    /// Observations for [`wide_creature`]: every input carries a distinct,
    /// comfortably-scoring correlation with the single target.
    fn wide_obs(inputs: usize) -> ObservationsStatistics {
        let scalar = |mean: f64| crate::observations::ScalarStats {
            count: 100,
            mean,
            variance: 1.0,
            std_dev: 1.0,
            min: -1.0,
            max: 1.0,
            zero_count: 0,
            non_zero_count: 100,
            non_finite_count: 0,
            mean_abs: mean.abs(),
            rms: 1.0,
            skewness: 0.0,
            excess_kurtosis: 0.0,
            quantiles: [0.0; 7],
        };
        let mut obs = empty_obs();
        obs.input_count = inputs;
        obs.output_count = 1;
        obs.inputs = (0..inputs).map(|_| scalar(0.0)).collect();
        obs.outputs = vec![scalar(0.0)];
        obs.input_target_correlations = (0..inputs)
            .map(|i| 0.9 - (i as f64) * 0.005)
            .collect::<Vec<_>>();
        obs
    }

    fn wide_incoming() -> IncomingSourceStats {
        IncomingSourceStats {
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
        }
    }

    fn wide_focus() -> FocusNeuronStats {
        FocusNeuronStats {
            neuron_uuid: "o1".into(),
            mean_error: Some(0.1),
            mean_abs_error: Some(0.1),
            mean_adjusted_error: Some(0.1),
            mean_derivative: Some(1.0),
            ..FocusNeuronStats::default()
        }
    }

    fn scaled_batch(
        ctx: &CandidateGenContext<'_>,
        count: usize,
        seed: u64,
    ) -> super::CandidateBatch {
        let mut rng = StdRng::seed_from_u64(seed);
        generate_candidate_batch(
            ctx,
            CandidateBudget {
                count,
                scale_quotas: true,
            },
            &mut rng,
        )
    }

    /// The weighted-random grid order is a true permutation — no slot is lost
    /// or repeated — so `Exhausted` still means every pairing was proposed.
    #[test]
    fn weighted_slot_order_is_a_permutation() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut order = super::weighted_slot_order(97, |slot| (slot + 1) as f64, &mut rng);
        order.sort_unstable();
        assert_eq!(order, (0..97).collect::<Vec<_>>());
    }

    /// Heavy slots almost surely lead the order (the obvious guesses go
    /// first), while the draw stays deterministic for a given seed.
    #[test]
    fn weighted_slot_order_puts_heavy_slots_first_and_is_seed_stable() {
        let weight = |slot: usize| if slot == 7 { 100.0 } else { 0.001 };
        let mut heavy_first = 0;
        for seed in 0..20 {
            let mut rng = StdRng::seed_from_u64(seed);
            let order = super::weighted_slot_order(32, weight, &mut rng);
            if order[0] == 7 {
                heavy_first += 1;
            }
        }
        // P(another slot beating a 100000:1 weight) is negligible per draw.
        assert!(heavy_first >= 19, "heavy slot led only {heavy_first}/20 draws");

        let mut rng_a = StdRng::seed_from_u64(11);
        let mut rng_b = StdRng::seed_from_u64(11);
        assert_eq!(
            super::weighted_slot_order(64, |s| (s + 1) as f64, &mut rng_a),
            super::weighted_slot_order(64, |s| (s + 1) as f64, &mut rng_b),
        );
    }

    /// Different seeds visit the grid tail differently — the property that
    /// lets repeated experiments cover the whole grid with no cursor.
    #[test]
    fn weighted_slot_order_varies_across_seeds() {
        let mut rng_a = StdRng::seed_from_u64(1);
        let mut rng_b = StdRng::seed_from_u64(2);
        let a = super::weighted_slot_order(64, |_| 1.0, &mut rng_a);
        let b = super::weighted_slot_order(64, |_| 1.0, &mut rng_b);
        assert_ne!(a, b);
    }

    /// Issue #108: with the quotas scaled, `--candidates N` binds at every N a
    /// creature with ample ranked sources can support — not just below ~29.
    #[test]
    fn the_candidate_budget_binds_until_genuine_exhaustion() {
        let incumbent = wide_creature(24);
        let focus = wide_focus();
        let observations = wide_obs(24);
        let incoming = [wide_incoming()];
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
        for count in [12usize, 29, 60, 120] {
            let batch = scaled_batch(&ctx, count, 5);
            assert_eq!(
                batch.candidates.len(),
                count,
                "budget {count} did not bind (limit {:?})",
                batch.limit
            );
            assert_eq!(batch.limit, BatchLimit::Budget);
        }
    }

    /// A budget past what the creature can support reports exhaustion — the
    /// true ceiling is named, not silently returned.
    #[test]
    fn an_exhausted_generator_reports_exhaustion_rather_than_the_budget() {
        let incumbent = parse_creature_json(TWO_INPUT).unwrap();
        let focus = wide_focus();
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
        let batch = scaled_batch(&ctx, 500, 5);
        assert_eq!(batch.limit, BatchLimit::Exhausted);
        assert!(
            !batch.candidates.is_empty() && batch.candidates.len() < 500,
            "expected a partial batch, got {}",
            batch.candidates.len()
        );
        // Two ranked sources across four weight scales and ten squashes is a
        // small but real hypothesis space — well above the old two-source floor.
        assert!(
            batch.candidates.len() > 8,
            "exhaustion should come after the whole grid, got {}",
            batch.candidates.len()
        );
    }

    /// Scaling must not let one family eat the extra budget: every strategy
    /// present at the old ceiling is still present in a much larger batch.
    #[test]
    fn a_scaled_batch_starves_no_strategy_family() {
        let incumbent = wide_creature(24);
        let focus = wide_focus();
        let observations = wide_obs(24);
        let incoming = [wide_incoming()];
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
        let small = scaled_batch(&ctx, 29, 5).strategy_mix();
        let large = scaled_batch(&ctx, 120, 5).strategy_mix();
        for (strategy, n) in &small {
            assert!(
                large.contains_key(strategy),
                "{} vanished from the larger batch (had {n}); mix={large:?}",
                strategy.label()
            );
        }
        assert!(
            large.len() >= small.len(),
            "the larger batch narrowed the mix: {small:?} -> {large:?}"
        );
    }

    /// Mutation identity of a candidate, with any generated neuron UUID removed
    /// so two identical bridges collide instead of looking distinct.
    fn mutation_identity(candidate: &Candidate) -> String {
        let is_uuid =
            |token: &str| token.len() == 36 && token.chars().filter(|c| *c == '-').count() == 4;
        let body = candidate
            .provenance
            .mutation
            .split_whitespace()
            .filter(|token| !is_uuid(token))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{}|{body}", candidate.provenance.strategy.label())
    }

    /// Filling a big budget must propose distinct mutations — a batch padded
    /// with repeats would bill screen time for candidates already scored.
    #[test]
    fn a_scaled_batch_contains_no_duplicate_candidates() {
        let incumbent = wide_creature(24);
        let focus = wide_focus();
        let observations = wide_obs(24);
        let incoming = [wide_incoming()];
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
        let batch = scaled_batch(&ctx, 120, 5);
        let mut seen = std::collections::BTreeSet::new();
        for candidate in &batch.candidates {
            let identity = mutation_identity(candidate);
            assert!(
                seen.insert(identity.clone()),
                "duplicate candidate in the batch: {identity}"
            );
        }
        assert_eq!(seen.len(), 120);
    }

    /// Issue #119: the **default** batch must not bill screen time twice for
    /// the same hypothesis. The round-robin fill re-proposed what the opening
    /// structural phases had already emitted, so 27 candidates carried only 22
    /// distinct mutations.
    #[test]
    fn the_default_batch_contains_no_duplicate_candidates() {
        let incumbent = wide_creature(24);
        let focus = wide_focus();
        let observations = wide_obs(24);
        let incoming = [wide_incoming()];
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
        let mut rng = StdRng::seed_from_u64(5);
        let batch = generate_candidates(&ctx, 29, &mut rng);
        let mut seen = std::collections::BTreeSet::new();
        for candidate in &batch {
            let identity = mutation_identity(candidate);
            assert!(
                seen.insert(identity.clone()),
                "duplicate candidate in the default batch: {identity}"
            );
        }
    }

    /// Issue #119: rejecting a duplicate must free its slot for the next
    /// strategy, not shrink the batch — the default path still delivers the
    /// requested budget, now as distinct hypotheses.
    #[test]
    fn rejecting_a_duplicate_frees_its_slot_rather_than_shrinking_the_batch() {
        let incumbent = wide_creature(24);
        let focus = wide_focus();
        let observations = wide_obs(24);
        let incoming = [wide_incoming()];
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
        let mut rng = StdRng::seed_from_u64(5);
        let batch = generate_candidate_batch(
            &ctx,
            CandidateBudget {
                count: 29,
                scale_quotas: false,
            },
            &mut rng,
        );
        assert_eq!(
            batch.candidates.len(),
            29,
            "the default batch under-filled its budget (limit {:?})",
            batch.limit
        );
        assert_eq!(batch.limit, BatchLimit::Budget);
    }

    /// A production-width creature has hundreds of ranked sources, so the
    /// budget binds well past the old ~29 ceiling without running dry.
    #[test]
    fn a_wide_creature_fills_a_far_larger_budget() {
        let incumbent = wide_creature(512);
        let focus = wide_focus();
        let observations = wide_obs(512);
        let incoming = [wide_incoming()];
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
        let batch = scaled_batch(&ctx, 240, 5);
        assert_eq!(batch.candidates.len(), 240);
        assert_eq!(batch.limit, BatchLimit::Budget);
        // The old fixed quotas top out well below it on the same creature.
        let mut rng = StdRng::seed_from_u64(5);
        let legacy = generate_candidates(&ctx, 240, &mut rng);
        assert!(
            legacy.len() < 40,
            "the fixed quotas should still cap the legacy batch, got {}",
            legacy.len()
        );
    }

    /// The opening phases are untouched, so a batch under the old ceiling is
    /// identical with or without the scaled quotas.
    #[test]
    fn scaling_the_quotas_leaves_a_small_batch_unchanged() {
        let incumbent = wide_creature(24);
        let focus = wide_focus();
        let observations = wide_obs(24);
        let incoming = [wide_incoming()];
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
        let mut rng = StdRng::seed_from_u64(5);
        let legacy = generate_candidates(&ctx, 8, &mut rng);
        let scaled = scaled_batch(&ctx, 8, 5);
        let mutations = |candidates: &[Candidate]| {
            candidates
                .iter()
                .map(|c| c.provenance.mutation.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(mutations(&legacy), mutations(&scaled.candidates));
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

    fn one_candidate(incumbent: &CreatureExport) -> Vec<Candidate> {
        let mut creature = incumbent.clone();
        creature.neurons[0].bias += 0.25;
        vec![Candidate {
            creature,
            provenance: CandidateProvenance {
                strategy: CandidateStrategy::Random,
                focus_neuron: "h1".into(),
                mutation: "test bias nudge".into(),
                old_value: Some(0.1),
                new_value: Some(0.35),
            },
        }]
    }

    /// Issue #114: the scorer is the only reader of a batch file, so it carries
    /// no pretty-printing.
    #[test]
    fn batch_files_are_compact_and_parse_back_to_the_same_creatures() {
        let incumbent = parse_creature_json(TINY).unwrap();
        let candidates = one_candidate(&incumbent);
        let dir = tempfile::tempdir().unwrap();
        let batch = dir.path().join("candidates-exp-1");
        let stems = write_candidate_batch(&batch, &incumbent, &candidates, None).unwrap();
        assert_eq!(stems, ["baseline", "candidate-000"]);

        for stem in &stems {
            let text = fs::read_to_string(batch.join(format!("{stem}.json"))).unwrap();
            assert!(
                !text.contains('\n'),
                "{stem}.json must be compact, got:\n{text}"
            );
        }
        // Formatting never changes a parsed value.
        let baseline = fs::read_to_string(batch.join("baseline.json")).unwrap();
        assert_eq!(parse_creature_json(&baseline).unwrap(), incumbent);
        let written = fs::read_to_string(batch.join("candidate-000.json")).unwrap();
        assert_eq!(
            parse_creature_json(&written).unwrap(),
            candidates[0].creature
        );
        // …and it is genuinely smaller than the pretty form it replaced.
        assert!(
            written.len()
                < creature_to_json_pretty(&candidates[0].creature)
                    .unwrap()
                    .len()
        );
    }

    /// The compact baseline still carries the `uuid` / `tags` the scorer and the
    /// check-in path read (issue #114).
    #[test]
    fn compact_baseline_keeps_uuid_and_tags() {
        let tagged = r#"{
          "uuid": "creature-9",
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
          ],
          "tags": [{"name":"score","value":"0.1"}]
        }"#;
        let incumbent = parse_creature_json(tagged).unwrap();
        let meta = CreatureMeta::from_creature_json(tagged);
        let dir = tempfile::tempdir().unwrap();
        let batch = dir.path().join("candidates-exp-2");
        write_candidate_batch(&batch, &incumbent, &[], Some(&meta)).unwrap();

        let text = fs::read_to_string(batch.join("baseline.json")).unwrap();
        assert!(!text.contains('\n'), "baseline.json must be compact");
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["uuid"], "creature-9");
        assert_eq!(value["tags"][0]["name"], "score");
        assert_eq!(value["tags"][0]["value"], "0.1");
        assert_eq!(parse_creature_json(&text).unwrap(), incumbent);
    }
}
