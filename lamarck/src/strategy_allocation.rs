//! Adaptive candidate-budget allocation by measured strategy return (#218).
//!
//! Lamarck's generator has always split the candidate budget the same way: the
//! fixed opening quotas, then a round-robin over every enabled strategy. That
//! allocation is *measured* — the journal records which strategy earned each
//! accept — but it is not *adaptive*: nine strategies share the budget however
//! they perform.
//!
//! This module adds the adaptive half, and nothing else. It keeps a decayed
//! ledger of what each strategy actually returned, and turns that ledger into a
//! per-strategy slot allocation for the next batch.
//!
//! # What a strategy is worth
//!
//! The reward is **authoritative full-corpus improvement per unit measured
//! cost**, never screen score:
//!
//! * *reward* — the accepted full-corpus score Δ credited to the strategies of
//!   the winning candidate (a combo splits its Δ evenly across its members),
//!   expressed in multiples of `--min-improvement` so the numbers are readable
//!   at a `1e-6` accept bar, plus [`PROMOTION_REWARD_UNITS`] for each candidate
//!   that converted from screen to a full-corpus promote. The promote credit is
//!   deliberately small: it is the only signal available before the first
//!   accept, and it must never outweigh a real improvement.
//! * *cost* — the scorer wall time the strategy's own candidates caused. Screen
//!   time is shared across every candidate in the batch; promote and combo time
//!   is shared across the candidates that were actually promoted. Both come
//!   from the journal's own `scorerCalls`, so a report reproduces exactly what
//!   the run computed.
//!
//! `value = reward units / (cost seconds + `[`PRIOR_COST_SECONDS`]`)`. A
//! strategy that has cost time and returned nothing is worth zero — never
//! negative, because a rejection is evidence about one proposal, not a debt.
//! The prior prices a thin sample honestly and, as [`StrategyEvidence::value`]
//! explains, is what lets decay move an allocation at all.
//!
//! # Why it cannot become a monoculture
//!
//! Three things bound the reallocation, in the order they bind:
//!
//! 1. **The exploration floor.** Every enabled strategy is reserved a share of
//!    [`crate::config::LamarckConfig::strategy_exploration_floor`] of the
//!    budget before value is consulted at all — and where the budget is too
//!    small to seat them all at once, that reserve rotates through the coldest
//!    arms rather than stranding any of them.
//! 2. **A UCB bonus.** Arms that have been tried least are lifted by
//!    [`OPTIMISTIC_VALUE`], so a cold arm keeps a real (not merely nonzero)
//!    chance of slots.
//! 3. **Decay.** Evidence is multiplied by
//!    [`crate::config::LamarckConfig::strategy_evidence_decay`] each experiment
//!    and again by [`INCUMBENT_CHANGE_RETENTION`] whenever an accept replaces
//!    the incumbent — the creature the evidence was measured against is gone.

use crate::candidates::CandidateStrategy;
use crate::run::ExperimentRecord;
use crate::scorer_cost::ScorerCallPhase;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Default share of the candidate budget reserved for exploration.
///
/// Split evenly across every enabled strategy before measured value allocates
/// anything, so `random` and a temporarily cold operator stay reachable. The
/// top of the 10–20% band #218 suggested: the conservative end of a change that
/// is otherwise free to concentrate the batch.
pub const DEFAULT_STRATEGY_EXPLORATION_FLOOR: f64 = 0.2;

/// Default per-experiment decay applied to every strategy's evidence.
///
/// `0.9` gives a half-life of about seven experiments, so an operator that
/// stopped working stops being funded within a few batches rather than at the
/// end of the run.
pub const DEFAULT_STRATEGY_EVIDENCE_DECAY: f64 = 0.9;

/// Share of the evidence that survives an incumbent change.
///
/// An accept replaces the creature every measurement was taken against, so most
/// of the ledger is stale the instant it happens. It is discounted rather than
/// cleared: the operator that just won is still the best evidence available
/// about the operator that might win next.
pub const INCUMBENT_CHANGE_RETENTION: f64 = 0.25;

/// Reward units credited for one screen → full-corpus promote conversion.
///
/// Worth a twentieth of clearing the accept bar. Before the first accept it is
/// the only measured return a strategy can show; after it, it is noise beside a
/// real improvement, which is exactly the weighting intended.
pub const PROMOTION_REWARD_UNITS: f64 = 0.05;

/// Weight of the UCB exploration bonus, in units of [`OPTIMISTIC_VALUE`].
pub const EXPLORATION_BONUS_WEIGHT: f64 = 0.5;

/// Scorer seconds of assumed silence every arm's value is measured against.
///
/// Value is `reward / (cost + prior)` rather than `reward / cost`, for two
/// reasons. It prices a thin sample honestly — one accept on two seconds of
/// scorer time is not a rate anybody should act on — and, more importantly, it
/// is what makes decay bite at all: a bare ratio is scale-invariant, so
/// discounting an arm's whole ledger would leave its value, and its slots,
/// exactly where they were.
///
/// Ten seconds is about one full-corpus creature score on the production
/// creature (`docs/scorer-call-cost.md`): an arm must earn against roughly one
/// promote call's worth of assumed silence before it starts to outrank a cold
/// one.
pub const PRIOR_COST_SECONDS: f64 = 10.0;

/// What an untried arm is optimistically assumed to be worth, in the units of
/// [`StrategyEvidence::value`]: one improvement at the accept bar over the
/// prior window.
///
/// The exploration bonus is scaled by this **fixed** optimism rather than by
/// the pool's own mean value. A bonus proportional to the pool would shrink in
/// step with a decaying leader, leaving the split between them unchanged — the
/// same scale-invariance trap [`PRIOR_COST_SECONDS`] exists to close, one level
/// up. Against a fixed reference, a leader that stops earning really does fall
/// back towards the cold arms.
pub const OPTIMISTIC_VALUE: f64 = 1.0 / PRIOR_COST_SECONDS;

/// Strategies the adaptive allocator may fund (issue #218).
///
/// Under `--structural-only` the generator emits nothing but growth
/// candidates, so those are the only arms that exist; allocating slots to a
/// weight strategy that cannot propose would silently shrink the batch.
pub fn adaptive_strategies(structural_only: bool) -> &'static [CandidateStrategy] {
    const ALL: [CandidateStrategy; 9] = [
        CandidateStrategy::Backprop,
        CandidateStrategy::MeanErrorBias,
        CandidateStrategy::StatsWeight,
        CandidateStrategy::StatsBias,
        CandidateStrategy::StatsSkewBias,
        CandidateStrategy::StructuralAdd,
        CandidateStrategy::StructuralAddNeuron,
        CandidateStrategy::StructuralWeaken,
        CandidateStrategy::Random,
    ];
    const STRUCTURAL: [CandidateStrategy; 2] = [
        CandidateStrategy::StructuralAdd,
        CandidateStrategy::StructuralAddNeuron,
    ];
    if structural_only { &STRUCTURAL } else { &ALL }
}

/// How the candidate budget is split across strategies (issue #218).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StrategyAllocationMode {
    /// The pre-#218 split: fixed opening quotas, then round-robin.
    #[default]
    Fixed,
    /// Slots allocated from decayed, measured per-strategy return.
    Adaptive,
}

impl StrategyAllocationMode {
    /// Parse a CLI spelling (`fixed` / `adaptive`).
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "fixed" => Some(Self::Fixed),
            "adaptive" => Some(Self::Adaptive),
            _ => None,
        }
    }

    /// Stable label for logs, journals and reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Adaptive => "adaptive",
        }
    }

    /// True when slots are drawn from measured return.
    pub fn is_adaptive(self) -> bool {
        matches!(self, Self::Adaptive)
    }
}

/// Decayed evidence about one strategy (issue #218).
///
/// Every field is a decayed sum rather than a count, so an old experiment
/// contributes a fraction of a trial. [`StrategyLedger::totals`] builds a
/// ledger that never decays, which is what `report` sums for its per-strategy
/// row.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyEvidence {
    /// Candidates this strategy contributed to a scored batch.
    pub trials: f64,
    /// Candidates that converted from screen to a full-corpus promote.
    pub promotions: f64,
    /// Accepted full-corpus improvements credited to this strategy.
    pub accepts: f64,
    /// Accepted full-corpus score Δ credited to this strategy.
    pub score_gain: f64,
    /// Scorer milliseconds this strategy's candidates caused.
    pub cost_ms: f64,
}

impl StrategyEvidence {
    /// Reward in units of `min_improvement` — full-corpus gain plus the small
    /// screen→promote conversion credit.
    pub fn reward_units(&self, min_improvement: f64) -> f64 {
        let gain_units = if min_improvement > 0.0 {
            self.score_gain / min_improvement
        } else {
            self.score_gain
        };
        gain_units.max(0.0) + PROMOTION_REWARD_UNITS * self.promotions
    }

    /// Reward units per second of measured scorer cost, shrunk towards zero by
    /// [`PRIOR_COST_SECONDS`].
    ///
    /// The prior is what makes decay *bite*. A bare `reward / cost` ratio is
    /// scale-invariant: multiplying an arm's whole ledger by `0.25` after an
    /// incumbent change would leave its value — and therefore its slots —
    /// exactly where they were, so the discount would be a no-op on the very
    /// decision it exists to influence. Dividing by `cost + prior` instead
    /// makes a decayed arm converge on zero, which is where an arm with no
    /// evidence already sits, so stale evidence really does return the pool
    /// towards the even split.
    ///
    /// It is also the honest reading of a thin sample: one accept on two
    /// seconds of scorer time is a rate estimate nobody should act on, and the
    /// prior prices it as `gain / (2 + prior)` rather than `gain / 2`.
    pub fn value(&self, min_improvement: f64) -> f64 {
        let cost_seconds = self.cost_ms / 1_000.0;
        if !cost_seconds.is_finite() || cost_seconds < 0.0 {
            return 0.0;
        }
        let value = self.reward_units(min_improvement) / (cost_seconds + PRIOR_COST_SECONDS);
        if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        }
    }

    fn scale(&mut self, factor: f64) {
        self.trials *= factor;
        self.promotions *= factor;
        self.accepts *= factor;
        self.score_gain *= factor;
        self.cost_ms *= factor;
    }
}

/// Candidate slots allocated to each strategy for one experiment (issue #218).
///
/// Journalled verbatim as `strategyAllocation`, so a reader can see what each
/// strategy was given and what it was worth at the time — the allocation is a
/// decision the run made, not one a report has to reconstruct.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyAllocation {
    /// Exploration floor in force when the slots were drawn.
    pub exploration_floor: f64,
    /// Allocated slots per strategy label.
    pub slots: BTreeMap<String, usize>,
    /// Estimated value (reward units per scorer second) at allocation time.
    pub value: BTreeMap<String, f64>,
}

impl StrategyAllocation {
    /// Slots allocated to `strategy`, or `None` when it was not an arm.
    ///
    /// `None` means *uncapped*, not zero: a strategy the allocator never
    /// considered must not be silenced by an allocation it is absent from.
    pub fn slots_for(&self, strategy: CandidateStrategy) -> Option<usize> {
        self.slots.get(strategy.label()).copied()
    }

    /// Estimated value of `strategy` at allocation time.
    pub fn value_for(&self, strategy: CandidateStrategy) -> Option<f64> {
        self.value.get(strategy.label()).copied()
    }

    /// Total slots across every arm.
    pub fn total_slots(&self) -> usize {
        self.slots.values().sum()
    }

    /// Fold `other` into this allocation, summing slots per strategy.
    ///
    /// A multi-focus experiment allocates once per focus and journals one line,
    /// so the line reports the slots the whole experiment allocated. Values are
    /// identical across the focuses of one experiment — the ledger does not
    /// move mid-experiment — so the merged record keeps them as they are.
    pub fn merge(&mut self, other: &Self) {
        for (strategy, slots) in &other.slots {
            *self.slots.entry(strategy.clone()).or_insert(0) += slots;
        }
        for (strategy, value) in &other.value {
            self.value.entry(strategy.clone()).or_insert(*value);
        }
        self.exploration_floor = other.exploration_floor;
    }

    /// One-line rendering for the run log, e.g. `structural_add=41 random=6`.
    pub fn summary(&self) -> String {
        self.slots
            .iter()
            .map(|(strategy, slots)| format!("{strategy}={slots}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Decayed per-strategy evidence, and the allocation drawn from it (#218).
#[derive(Debug, Clone)]
pub struct StrategyLedger {
    decay: f64,
    incumbent_retention: f64,
    min_improvement: f64,
    arms: BTreeMap<CandidateStrategy, StrategyEvidence>,
}

impl StrategyLedger {
    /// A ledger that decays by `decay` each experiment.
    pub fn new(decay: f64, min_improvement: f64) -> Self {
        Self {
            decay,
            incumbent_retention: INCUMBENT_CHANGE_RETENTION,
            min_improvement,
            arms: BTreeMap::new(),
        }
    }

    /// A ledger that never decays — running totals for `report`.
    pub fn totals(min_improvement: f64) -> Self {
        Self {
            decay: 1.0,
            incumbent_retention: 1.0,
            min_improvement,
            arms: BTreeMap::new(),
        }
    }

    /// Evidence held for `strategy` (all zeros when it has never been tried).
    pub fn evidence(&self, strategy: CandidateStrategy) -> StrategyEvidence {
        self.arms.get(&strategy).copied().unwrap_or_default()
    }

    /// Estimated value of `strategy` — reward units per scorer second.
    pub fn value(&self, strategy: CandidateStrategy) -> f64 {
        self.evidence(strategy).value(self.min_improvement)
    }

    /// Every strategy the ledger holds evidence for.
    pub fn strategies(&self) -> impl Iterator<Item = CandidateStrategy> + '_ {
        self.arms.keys().copied()
    }

    /// Fold one journalled experiment into the ledger.
    ///
    /// Reading the journal record — rather than the run's in-memory state — is
    /// what makes a report reproduce the run: `report` replays the same records
    /// through the same accumulator and gets the same values.
    pub fn observe(&mut self, record: &ExperimentRecord) {
        self.decay_all(self.decay);

        let strategies: Vec<CandidateStrategy> = record
            .candidates
            .iter()
            .map(|candidate| candidate.strategy)
            .collect();
        if strategies.is_empty() {
            return;
        }

        let mut trials: BTreeMap<CandidateStrategy, f64> = BTreeMap::new();
        for strategy in &strategies {
            *trials.entry(*strategy).or_insert(0.0) += 1.0;
        }
        let mut promotions: BTreeMap<CandidateStrategy, f64> = BTreeMap::new();
        for stem in record.scores.keys() {
            let Some(index) = candidate_stem_index(stem) else {
                continue;
            };
            if let Some(strategy) = strategies.get(index) {
                *promotions.entry(*strategy).or_insert(0.0) += 1.0;
            }
        }

        for (strategy, count) in &trials {
            self.arms.entry(*strategy).or_default().trials += count;
        }
        // The credit is for *converting* a screen into a promote. An experiment
        // that ran no screen phase scored every candidate on the full corpus,
        // so its "conversions" are just its trials and would credit every arm
        // equally — a signal that says nothing. Its promote cost is still
        // charged below; only the reward is withheld.
        if record.screen_scores.is_some() {
            for (strategy, count) in &promotions {
                self.arms.entry(*strategy).or_default().promotions += count;
            }
        }

        let (screen_ms, promote_ms) = phase_costs(record);
        self.charge(screen_ms, &trials);
        if promotions.is_empty() {
            self.charge(promote_ms, &trials);
        } else {
            self.charge(promote_ms, &promotions);
        }

        if record.accepted {
            self.credit_accept(record, &strategies);
            // The accept replaced the incumbent every measurement above was
            // taken against, so the ledger is discounted immediately.
            self.decay_all(self.incumbent_retention);
        }
    }

    /// Slots for `budget` candidates across `arms`, reserving `floor` of the
    /// budget for exploration.
    ///
    /// The reserve is spread evenly, one whole slot at a time; what is left is
    /// apportioned by the UCB index (measured value plus an under-trial bonus)
    /// using largest remainders, so the slots always sum to `budget`.
    pub fn allocate(
        &self,
        arms: &[CandidateStrategy],
        budget: usize,
        floor: f64,
    ) -> StrategyAllocation {
        let mut allocation = StrategyAllocation {
            exploration_floor: floor,
            ..StrategyAllocation::default()
        };
        if arms.is_empty() || budget == 0 {
            return allocation;
        }
        for arm in arms {
            allocation
                .value
                .insert(arm.label().to_string(), self.value(*arm));
        }

        let trials: Vec<f64> = arms.iter().map(|arm| self.evidence(*arm).trials).collect();
        let reserve = reserved_slots(&trials, budget, floor);
        let remaining = budget.saturating_sub(reserve.iter().sum());
        let shares = apportion(remaining, &self.indices(arms));
        for ((arm, reserved), share) in arms.iter().zip(&reserve).zip(shares) {
            allocation
                .slots
                .insert(arm.label().to_string(), reserved + share);
        }
        allocation
    }

    /// UCB index per arm: measured value plus an under-trial bonus.
    ///
    /// With nothing tried at all the horizon is `ln(1) = 0`, so every index is
    /// zero and the apportionment falls back to the even split — the
    /// round-robin allocation adaptive mode has to beat.
    fn indices(&self, arms: &[CandidateStrategy]) -> Vec<f64> {
        let total_trials: f64 = arms.iter().map(|arm| self.evidence(*arm).trials).sum();
        let horizon = (1.0 + total_trials).ln().max(0.0);
        arms.iter()
            .map(|arm| {
                let trials = self.evidence(*arm).trials.max(1.0);
                self.value(*arm)
                    + EXPLORATION_BONUS_WEIGHT * OPTIMISTIC_VALUE * (horizon / trials).sqrt()
            })
            .collect()
    }

    fn decay_all(&mut self, factor: f64) {
        if factor >= 1.0 {
            return;
        }
        for evidence in self.arms.values_mut() {
            evidence.scale(factor);
        }
    }

    /// Charge `ms` across `weights`, pro-rata by weight.
    fn charge(&mut self, ms: f64, weights: &BTreeMap<CandidateStrategy, f64>) {
        let total: f64 = weights.values().sum();
        if !ms.is_finite() || ms <= 0.0 || total <= 0.0 {
            return;
        }
        for (strategy, weight) in weights {
            self.arms.entry(*strategy).or_default().cost_ms += ms * weight / total;
        }
    }

    /// Credit the accepted improvement to the winner's member strategies.
    ///
    /// A merged combo splits its Δ evenly across its members: crediting each
    /// member the whole Δ would inflate the ledger's total gain above the
    /// improvement the run actually earned.
    fn credit_accept(&mut self, record: &ExperimentRecord, strategies: &[CandidateStrategy]) {
        let Some(delta) = record.improvement else {
            return;
        };
        let members: Vec<usize> = match &record.combo_member_indices {
            Some(indices) if !indices.is_empty() => indices.clone(),
            // A journal written before `comboMemberIndices` existed names only
            // the winning stem; a single is still attributable from it.
            _ => record
                .winner
                .as_deref()
                .and_then(candidate_stem_index)
                .into_iter()
                .collect(),
        };
        if members.is_empty() {
            return;
        }
        let share = delta / members.len() as f64;
        for index in members {
            let Some(strategy) = strategies.get(index) else {
                continue;
            };
            let evidence = self.arms.entry(*strategy).or_default();
            evidence.accepts += 1.0;
            evidence.score_gain += share;
        }
    }
}

/// Screen and promote (plus combo) milliseconds of one experiment.
///
/// Falls back to the experiment's total `scorerMs` when the journal predates
/// per-call records: the whole cost is then attributed to the screen phase,
/// which every candidate shares.
fn phase_costs(record: &ExperimentRecord) -> (f64, f64) {
    let Some(calls) = record
        .scorer_calls
        .as_ref()
        .filter(|calls| !calls.is_empty())
    else {
        return (record.scorer_ms as f64, 0.0);
    };
    let mut screen_ms = 0.0;
    let mut promote_ms = 0.0;
    for call in calls {
        match call.phase {
            ScorerCallPhase::Screen => screen_ms += call.elapsed_ms as f64,
            ScorerCallPhase::Promote | ScorerCallPhase::Combo => {
                promote_ms += call.elapsed_ms as f64
            }
            // Phase-0 parity and graft replay score no candidate from this
            // batch, so charging them to a strategy would be an invention.
            ScorerCallPhase::Phase0 | ScorerCallPhase::GraftReplay => {}
        }
    }
    (screen_ms, promote_ms)
}

/// Index of a `candidate-NNN` stem, or `None` for any other stem.
fn candidate_stem_index(stem: &str) -> Option<usize> {
    stem.strip_prefix("candidate-")?.parse().ok()
}

/// Slots the exploration floor reserves for each arm, coldest arms first.
///
/// `floor × budget` whole slots are reserved and spread evenly; the remainder
/// of that division goes to the arms with the fewest decayed trials, breaking
/// ties by arm order. That is what keeps every enabled strategy reachable at
/// **any** budget:
///
/// * When the budget can seat every arm several times over — the production
///   case, 100 candidates over nine arms at `0.2` — every arm is reserved whole
///   slots in the same batch, and the reserve is the fraction asked for rather
///   than a rounded-up approximation of it.
/// * When the budget is small (a large `--focus-count` splits it, and a focus
///   share can be smaller than the arm count), no allocation can seat every arm
///   at once. The reserve then rotates: the coldest arms take it, their trial
///   counts rise, and the next batch reserves for the next coldest. Every arm
///   is reached within a few batches instead of one — which is the honest form
///   of the guarantee at that budget, and is still the property that stops an
///   arm going permanently unreachable.
fn reserved_slots(trials: &[f64], budget: usize, floor: f64) -> Vec<usize> {
    let arms = trials.len();
    if arms == 0 {
        return Vec::new();
    }
    if !floor.is_finite() || floor <= 0.0 {
        return vec![0; arms];
    }
    let total = ((floor.min(1.0) * budget as f64).round() as usize).min(budget);
    let mut reserve = vec![total / arms; arms];
    let mut coldest: Vec<usize> = (0..arms).collect();
    coldest.sort_by(|a, b| trials[*a].total_cmp(&trials[*b]).then_with(|| a.cmp(b)));
    for index in coldest.into_iter().take(total % arms) {
        reserve[index] += 1;
    }
    reserve
}

/// Apportion `total` slots across `weights` by largest remainders.
///
/// Non-positive or non-finite weights fall back to an even split, so a pool
/// with nothing measured yet is allocated exactly as round-robin would.
fn apportion(total: usize, weights: &[f64]) -> Vec<usize> {
    let n = weights.len();
    if n == 0 || total == 0 {
        return vec![0; n];
    }
    let clean: Vec<f64> = weights
        .iter()
        .map(|w| if w.is_finite() && *w > 0.0 { *w } else { 0.0 })
        .collect();
    let sum: f64 = clean.iter().sum();
    let clean = if sum > 0.0 { clean } else { vec![1.0; n] };
    let sum: f64 = clean.iter().sum();

    let mut slots = Vec::with_capacity(n);
    let mut remainders: Vec<(f64, usize)> = Vec::with_capacity(n);
    let mut assigned = 0usize;
    for (i, weight) in clean.iter().enumerate() {
        let exact = total as f64 * weight / sum;
        let whole = exact.floor();
        let whole_slots = whole as usize;
        slots.push(whole_slots);
        assigned += whole_slots;
        remainders.push((exact - whole, i));
    }
    // Largest remainder first; ties by arm order, so the split is deterministic.
    remainders.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut leftover = total.saturating_sub(assigned);
    for (_, i) in remainders {
        if leftover == 0 {
            break;
        }
        slots[i] += 1;
        leftover -= 1;
    }
    slots
}

/// Validated allocation policy for one run (issue #218).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AllocationPolicy {
    /// Allocation mode in force.
    pub mode: StrategyAllocationMode,
    /// Share of the budget reserved for exploration.
    pub exploration_floor: f64,
    /// Per-experiment evidence decay.
    pub evidence_decay: f64,
}

impl AllocationPolicy {
    /// True when the budget is allocated from measured return.
    pub fn is_adaptive(self) -> bool {
        self.mode.is_adaptive()
    }

    /// Slots for one batch, or `None` under the fixed (pre-#218) allocation.
    ///
    /// `None` is what keeps the A/B honest: the generator receives no
    /// allocation at all and proposes exactly the batch it did before #218.
    pub fn allocate(
        self,
        ledger: &StrategyLedger,
        structural_only: bool,
        budget: usize,
    ) -> Option<StrategyAllocation> {
        if !self.is_adaptive() {
            return None;
        }
        Some(ledger.allocate(
            adaptive_strategies(structural_only),
            budget,
            self.exploration_floor,
        ))
    }

    /// A ledger configured for this policy's decay.
    pub fn ledger(self, min_improvement: f64) -> StrategyLedger {
        StrategyLedger::new(self.evidence_decay, min_improvement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parses_and_labels_both_arms() {
        assert_eq!(
            StrategyAllocationMode::parse("fixed"),
            Some(StrategyAllocationMode::Fixed)
        );
        assert_eq!(
            StrategyAllocationMode::parse(" Adaptive "),
            Some(StrategyAllocationMode::Adaptive)
        );
        assert_eq!(StrategyAllocationMode::parse("bandit"), None);
        assert_eq!(StrategyAllocationMode::default().label(), "fixed");
        assert!(StrategyAllocationMode::Adaptive.is_adaptive());
        assert!(!StrategyAllocationMode::Fixed.is_adaptive());
    }

    #[test]
    fn structural_only_runs_allocate_only_growth_arms() {
        assert_eq!(adaptive_strategies(false).len(), 9);
        assert_eq!(
            adaptive_strategies(true),
            [
                CandidateStrategy::StructuralAdd,
                CandidateStrategy::StructuralAddNeuron
            ]
        );
    }

    #[test]
    fn apportionment_always_sums_to_the_budget() {
        for total in [0usize, 1, 7, 100, 997] {
            for weights in [
                vec![1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0],
                vec![5.0, 1.0, 0.0, f64::NAN],
                vec![1e-9, 1.0],
            ] {
                let slots = apportion(total, &weights);
                assert_eq!(slots.len(), weights.len());
                assert_eq!(
                    slots.iter().sum::<usize>(),
                    total,
                    "apportioning {total} across {weights:?} lost a slot"
                );
            }
        }
    }

    #[test]
    fn an_all_zero_weight_pool_is_split_evenly() {
        assert_eq!(apportion(9, &[0.0, 0.0, 0.0]), vec![3, 3, 3]);
    }

    /// The reserve is the fraction asked for, spread evenly, with the odd
    /// slots going to the arms that have been tried least.
    #[test]
    fn the_reserve_is_the_floor_fraction_and_favours_the_coldest_arms() {
        let cold_last = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 0.0];
        let reserve = reserved_slots(&cold_last, 100, 0.2);
        assert_eq!(reserve.iter().sum::<usize>(), 20, "20% of 100 is 20 slots");
        assert_eq!(reserve[8], 3, "the coldest arm takes an odd slot");
        assert_eq!(reserve[7], 3, "and so does the next coldest");
        assert_eq!(reserve[0], 2, "every other arm keeps the even share");

        // A floor of 1.0 reserves the whole budget evenly — round-robin.
        assert_eq!(reserved_slots(&[0.0; 9], 90, 1.0), vec![10; 9]);
        // A floor of 0 reserves nothing: pure exploitation, as asked for.
        assert_eq!(reserved_slots(&[0.0; 9], 100, 0.0), vec![0; 9]);
    }

    /// A budget too small to seat every arm reserves for the coldest instead,
    /// so the reserve rotates rather than stranding an arm (issue #218).
    #[test]
    fn a_budget_smaller_than_the_arm_count_reserves_for_the_coldest() {
        let trials = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 6.0, 7.0, 8.0];
        let reserve = reserved_slots(&trials, 4, 0.5);
        assert_eq!(reserve.iter().sum::<usize>(), 2, "50% of 4 is 2 slots");
        assert_eq!(reserve[5], 1, "the coldest arm is reserved for");
        assert_eq!(reserve[4], 1, "and the next coldest");
        assert_eq!(reserve[8], 0, "the most-tried arm waits its turn");
    }

    #[test]
    fn value_is_reward_units_per_scorer_second_against_the_prior() {
        let evidence = StrategyEvidence {
            trials: 10.0,
            promotions: 2.0,
            accepts: 1.0,
            score_gain: 4e-6,
            cost_ms: 2_000.0,
        };
        // 4 units of gain + 2 promotions × 0.05 = 4.1 units, over 2 measured
        // seconds plus the 10-second prior.
        assert!((evidence.value(1e-6) - 4.1 / 12.0).abs() < 1e-12);
        // No return, no value — whatever it cost.
        assert_eq!(
            StrategyEvidence {
                trials: 40.0,
                cost_ms: 60_000.0,
                ..StrategyEvidence::default()
            }
            .value(1e-6),
            0.0
        );
    }

    /// The prior is what makes decay bite: a scaled-down ledger must be worth
    /// less, or the incumbent-change discount could not move a single slot.
    #[test]
    fn decayed_evidence_is_worth_less_than_the_evidence_it_came_from() {
        let fresh = StrategyEvidence {
            trials: 10.0,
            promotions: 2.0,
            accepts: 1.0,
            score_gain: 4e-6,
            cost_ms: 2_000.0,
        };
        let mut stale = fresh;
        stale.scale(INCUMBENT_CHANGE_RETENTION);
        assert!(
            stale.value(1e-6) < fresh.value(1e-6) * 0.5,
            "a quartered ledger must lose most of its value: {} vs {}",
            stale.value(1e-6),
            fresh.value(1e-6)
        );
        // …and keep losing it, towards the zero an unmeasured arm sits at.
        let mut staler = stale;
        staler.scale(INCUMBENT_CHANGE_RETENTION);
        assert!(staler.value(1e-6) < stale.value(1e-6));
    }

    #[test]
    fn a_negative_gain_never_becomes_a_negative_value() {
        let evidence = StrategyEvidence {
            score_gain: -1e-3,
            cost_ms: 1_000.0,
            ..StrategyEvidence::default()
        };
        assert_eq!(evidence.value(1e-6), 0.0);
    }
}
