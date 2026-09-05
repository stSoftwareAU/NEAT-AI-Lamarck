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
//! `value = reward units / cost seconds`. A strategy that has cost time and
//! returned nothing is worth zero — never negative, because a rejection is
//! evidence about one proposal, not a debt.
//!
//! # Why it cannot become a monoculture
//!
//! Three things bound the reallocation, in the order they bind:
//!
//! 1. **The exploration floor.** Every enabled strategy is reserved an equal
//!    share of [`crate::config::LamarckConfig::strategy_exploration_floor`] of
//!    the budget before value is consulted at all.
//! 2. **A UCB bonus.** Arms that have been tried least are lifted towards the
//!    leader's value, scaled by the mean value of the pool, so a cold arm keeps
//!    a real (not merely nonzero) chance of slots.
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

/// Weight of the UCB exploration bonus, in units of the pool's mean value.
pub const EXPLORATION_BONUS_WEIGHT: f64 = 0.5;

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

    /// Reward units per second of measured scorer cost.
    ///
    /// Zero while the strategy has cost nothing: an arm with no measured cost
    /// has no measured return either, and the exploration floor — not an
    /// invented value — is what keeps it reachable.
    pub fn value(&self, min_improvement: f64) -> f64 {
        let cost_seconds = self.cost_ms / 1_000.0;
        if !cost_seconds.is_finite() || cost_seconds <= 0.0 {
            return 0.0;
        }
        let value = self.reward_units(min_improvement) / cost_seconds;
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
        for (strategy, count) in &promotions {
            self.arms.entry(*strategy).or_default().promotions += count;
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

        let reserved_each = reserved_slots_each(arms.len(), budget, floor);
        let reserved = reserved_each * arms.len();
        let remaining = budget.saturating_sub(reserved);
        let shares = apportion(remaining, &self.indices(arms));
        for (arm, share) in arms.iter().zip(shares) {
            allocation
                .slots
                .insert(arm.label().to_string(), reserved_each + share);
        }
        allocation
    }

    /// UCB index per arm: measured value plus an under-trial bonus.
    fn indices(&self, arms: &[CandidateStrategy]) -> Vec<f64> {
        let values: Vec<f64> = arms.iter().map(|arm| self.value(*arm)).collect();
        let mean_value = values.iter().sum::<f64>() / values.len() as f64;
        if mean_value <= 0.0 {
            // Nothing has returned anything yet, so there is nothing to be
            // confident about: an even split is the honest allocation.
            return values.iter().map(|_| 0.0).collect();
        }
        let total_trials: f64 = arms.iter().map(|arm| self.evidence(*arm).trials).sum();
        let horizon = (1.0 + total_trials).ln().max(0.0);
        arms.iter()
            .zip(&values)
            .map(|(arm, value)| {
                let trials = self.evidence(*arm).trials.max(1.0);
                value + EXPLORATION_BONUS_WEIGHT * mean_value * (horizon / trials).sqrt()
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

/// Whole slots reserved for **each** arm by the exploration floor.
///
/// Rounded up so the reserve is never quieter than the fraction asked for, and
/// capped so the reserve alone cannot consume the budget. A budget smaller than
/// the arm count cannot give everyone a slot, so it reserves nothing and lets
/// the apportionment spread what there is.
fn reserved_slots_each(arms: usize, budget: usize, floor: f64) -> usize {
    if arms == 0 || budget < arms || !floor.is_finite() || floor <= 0.0 {
        return 0;
    }
    let wanted = (floor.min(1.0) * budget as f64 / arms as f64).ceil() as usize;
    wanted.clamp(1, budget / arms)
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

    #[test]
    fn the_reserve_never_consumes_the_whole_budget() {
        // 9 arms, 100 slots, 20% floor → 3 each (ceil), leaving 73 to allocate.
        assert_eq!(reserved_slots_each(9, 100, 0.2), 3);
        // A floor of 1.0 reserves the whole budget evenly — round-robin.
        assert_eq!(reserved_slots_each(9, 90, 1.0), 10);
        // Fewer slots than arms cannot seat everyone; nothing is reserved.
        assert_eq!(reserved_slots_each(9, 4, 0.5), 0);
        assert_eq!(reserved_slots_each(9, 100, 0.0), 0);
    }

    #[test]
    fn value_is_reward_units_per_scorer_second() {
        let evidence = StrategyEvidence {
            trials: 10.0,
            promotions: 2.0,
            accepts: 1.0,
            score_gain: 4e-6,
            cost_ms: 2_000.0,
        };
        // 4 units of gain + 2 promotions × 0.05 = 4.1 units over 2 seconds.
        assert!((evidence.value(1e-6) - 2.05).abs() < 1e-12);
        // No measured cost, no value — the floor is what keeps it reachable.
        assert_eq!(
            StrategyEvidence {
                score_gain: 1.0,
                ..StrategyEvidence::default()
            }
            .value(1e-6),
            0.0
        );
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
