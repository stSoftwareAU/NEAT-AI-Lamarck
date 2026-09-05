//! Adaptive candidate-budget allocation (Issue #218).
//!
//! The acceptance criteria of #218 are behavioural, so they are tested through
//! the public API a run and a report actually use: a ledger fed journalled
//! experiments, an allocation drawn from it, and `report` exposing what each
//! strategy was allocated, tried, accepted, gained and cost. Experiments are
//! built as journal JSON and parsed back, so the evidence the ledger reads is
//! exactly the evidence a real `experiments.jsonl` carries.

use neat_ai_lamarck::candidates::CandidateStrategy;
use neat_ai_lamarck::config::LamarckConfig;
use neat_ai_lamarck::report::report_from_journal;
use neat_ai_lamarck::run::ExperimentRecord;
use neat_ai_lamarck::strategy_allocation::{
    DEFAULT_STRATEGY_EVIDENCE_DECAY, DEFAULT_STRATEGY_EXPLORATION_FLOOR, StrategyAllocationMode,
    StrategyLedger, adaptive_strategies,
};
use serde_json::json;

const MIN_IMPROVEMENT: f64 = 1e-6;

/// One journalled experiment: `mix` candidates in proposal order, the listed
/// indices promoted to full-corpus scoring, and an optional accepted winner.
struct Experiment {
    number: u64,
    mix: Vec<CandidateStrategy>,
    promoted: Vec<usize>,
    winner: Option<(usize, f64)>,
    screen_ms: u64,
    promote_ms: u64,
}

impl Experiment {
    fn json(&self) -> serde_json::Value {
        let candidates: Vec<serde_json::Value> = self
            .mix
            .iter()
            .map(|strategy| {
                json!({
                    "strategy": strategy.label(),
                    "focusNeuron": "neuron-1",
                    "mutation": format!("{} proposal", strategy.label()),
                    "oldValue": null,
                    "newValue": null,
                })
            })
            .collect();
        let mut scores = serde_json::Map::new();
        scores.insert("baseline".to_string(), json!(0.5));
        for index in &self.promoted {
            scores.insert(format!("candidate-{index:03}"), json!(0.5));
        }
        json!({
            "experimentNumber": self.number,
            "timestampUnix": 1_700_000_000u64 + self.number,
            "seed": 7,
            "incumbentId": "in2-out1-n3-s4",
            "baselineScore": 0.5,
            "focusNeuron": "neuron-1",
            "candidates": candidates,
            "scores": serde_json::Value::Object(scores),
            "winner": self.winner.map(|(index, _)| format!("candidate-{index:03}")),
            "improvement": self.winner.map(|(_, delta)| delta),
            "accepted": self.winner.is_some(),
            "comboMemberIndices": self.winner.map(|(index, _)| vec![index]),
            "analysisMs": 1_000,
            "scorerMs": self.screen_ms + self.promote_ms,
            "scorerCalls": [
                {
                    "phase": "screen",
                    "creatures": self.mix.len() as u64 + 1,
                    "sampleRate": 0.05,
                    "elapsedMs": self.screen_ms,
                },
                {
                    "phase": "promote",
                    "creatures": self.promoted.len() as u64 + 1,
                    "elapsedMs": self.promote_ms,
                },
            ],
        })
    }

    fn record(&self) -> ExperimentRecord {
        serde_json::from_value(self.json()).expect("journal shape parses as an experiment")
    }
}

/// A batch offering every adaptive strategy once, in round-robin order.
fn round_robin_mix() -> Vec<CandidateStrategy> {
    adaptive_strategies(false).to_vec()
}

/// Candidate index `strategy` occupies in [`round_robin_mix`].
fn slot_of(strategy: CandidateStrategy) -> usize {
    adaptive_strategies(false)
        .iter()
        .position(|s| *s == strategy)
        .expect("strategy is an adaptive arm")
}

fn ledger_with(records: &[ExperimentRecord]) -> StrategyLedger {
    let mut ledger = StrategyLedger::new(DEFAULT_STRATEGY_EVIDENCE_DECAY, MIN_IMPROVEMENT);
    for record in records {
        ledger.observe(record);
    }
    ledger
}

/// `count` experiments in which `winner` earns every accept.
fn winning_run(count: u64, winner: CandidateStrategy, delta: f64) -> Vec<ExperimentRecord> {
    let index = slot_of(winner);
    (1..=count)
        .map(|number| {
            Experiment {
                number,
                mix: round_robin_mix(),
                promoted: vec![index],
                winner: Some((index, delta)),
                screen_ms: 9_000,
                promote_ms: 11_000,
            }
            .record()
        })
        .collect()
}

/// Acceptance criterion: adaptive mode allocates the candidate budget from
/// journalled historical performance — the strategy that earned the accepts
/// gets more slots than one that only ever cost scorer time.
#[test]
fn measured_return_moves_slots_towards_the_earning_strategy() {
    let ledger = ledger_with(&winning_run(6, CandidateStrategy::StructuralAdd, 4e-6));

    let allocation = ledger.allocate(
        adaptive_strategies(false),
        100,
        DEFAULT_STRATEGY_EXPLORATION_FLOOR,
    );
    let earned = allocation
        .slots_for(CandidateStrategy::StructuralAdd)
        .expect("the earning strategy is allocated");
    let idle = allocation
        .slots_for(CandidateStrategy::StatsBias)
        .expect("an idle strategy is still allocated");
    assert!(
        earned > idle,
        "measured return must buy slots: structural_add={earned} stats_bias={idle}"
    );
    assert_eq!(
        allocation.total_slots(),
        100,
        "every slot of the budget is allocated"
    );
}

/// The allocator has to track return at the size it will actually run at.
///
/// Value is `reward / (cost + prior)`, so its scale falls as an arm accumulates
/// scorer time: a nine-candidate test batch and a production batch of 100 at
/// ~100s per screen call are two different regimes. An exploration bonus fixed
/// at the test scale swamps the measured value at the production scale and the
/// allocation quietly reverts to uniform — which is why this asserts on a
/// production-shaped ledger, not a toy one.
#[test]
fn measured_return_still_moves_slots_at_production_batch_sizes() {
    let index = slot_of(CandidateStrategy::StructuralAdd);
    // 99 candidates per experiment (11 per arm), a ~100s screen call and a
    // ~33s promote call, and one accept every fifth experiment.
    let records: Vec<ExperimentRecord> = (1..=20)
        .map(|number| {
            let mut mix = Vec::new();
            for _ in 0..11 {
                mix.extend(round_robin_mix());
            }
            Experiment {
                number,
                mix,
                promoted: vec![index, index + 9, index + 18],
                winner: (number % 5 == 0).then_some((index, 3e-6)),
                screen_ms: 100_000,
                promote_ms: 33_000,
            }
            .record()
        })
        .collect();
    let ledger = ledger_with(&records);

    let allocation = ledger.allocate(
        adaptive_strategies(false),
        100,
        DEFAULT_STRATEGY_EXPLORATION_FLOOR,
    );
    let earned = allocation
        .slots_for(CandidateStrategy::StructuralAdd)
        .expect("the earning strategy is allocated");
    let arms = adaptive_strategies(false);
    let even_share = 100 / arms.len();
    assert!(
        earned * 2 >= even_share * 3,
        "at production scale the earner must take at least half again the even \
         share ({even_share}), got {earned} — the exploration bonus is swamping \
         measured value"
    );
    let others: Vec<usize> = arms
        .iter()
        .filter(|arm| **arm != CandidateStrategy::StructuralAdd)
        .map(|arm| allocation.slots_for(*arm).unwrap_or(0))
        .collect();
    assert!(
        others.iter().all(|slots| *slots < earned),
        "the earner must lead the pool: {earned} against {others:?}"
    );
    assert!(
        others.iter().all(|slots| *slots >= 5),
        "…and the reallocation stays conservative — no arm is starved: {others:?}"
    );
    assert_eq!(allocation.total_slots(), 100);
}

/// Guardrail: the exploration floor keeps every enabled strategy reachable, so
/// one dominant operator cannot become a monoculture.
#[test]
fn the_exploration_floor_keeps_every_strategy_reachable() {
    let ledger = ledger_with(&winning_run(20, CandidateStrategy::StructuralAdd, 1e-4));
    let arms = adaptive_strategies(false);
    let allocation = ledger.allocate(arms, 100, 0.2);

    for arm in arms {
        let slots = allocation
            .slots_for(*arm)
            .unwrap_or_else(|| panic!("{} is missing from the allocation", arm.label()));
        assert!(
            slots >= 2,
            "{} must keep its floor share, got {slots}",
            arm.label()
        );
    }
    assert_eq!(allocation.total_slots(), 100);
}

/// A zero floor is honoured as written — the operator asked for pure
/// exploitation, and a silently re-imposed floor would invalidate the A/B.
#[test]
fn a_zero_exploration_floor_allocates_purely_on_measured_value() {
    let ledger = ledger_with(&winning_run(20, CandidateStrategy::StructuralAdd, 1e-3));
    let allocation = ledger.allocate(adaptive_strategies(false), 100, 0.0);
    let earned = allocation
        .slots_for(CandidateStrategy::StructuralAdd)
        .expect("the earning strategy is allocated");
    assert!(
        earned > 50,
        "with no floor the measured leader takes the pool, got {earned}"
    );
}

/// Acceptance criterion: reward uses full-corpus scorer improvements **and**
/// measured cost — the same gain bought more cheaply is worth more.
#[test]
fn the_cheaper_of_two_equally_improving_strategies_is_valued_higher() {
    // Two runs of identical shape and identical gain, differing only in the
    // scorer time the winner's candidates cost. Fed to separate ledgers so
    // recency decay cannot confound the comparison.
    let winning_at_cost = |screen_ms: u64, promote_ms: u64| {
        let index = slot_of(CandidateStrategy::StatsWeight);
        let records: Vec<ExperimentRecord> = (1..=4)
            .map(|number| {
                Experiment {
                    number,
                    mix: round_robin_mix(),
                    promoted: vec![index],
                    winner: Some((index, 2e-6)),
                    screen_ms,
                    promote_ms,
                }
                .record()
            })
            .collect();
        ledger_with(&records)
    };

    let cheap = winning_at_cost(900, 100);
    let dear = winning_at_cost(9_000, 51_000);
    let cheap_value = cheap.value(CandidateStrategy::StatsWeight);
    let dear_value = dear.value(CandidateStrategy::StatsWeight);
    assert!(
        (cheap.evidence(CandidateStrategy::StatsWeight).score_gain
            - dear.evidence(CandidateStrategy::StatsWeight).score_gain)
            .abs()
            < 1e-15,
        "the two arms must differ only in cost"
    );
    assert!(
        cheap_value > dear_value,
        "equal gain at lower measured cost must value higher: cheap={cheap_value} dear={dear_value}"
    );
}

/// Acceptance criterion: evidence decays after an incumbent change, so an
/// operator that worked on an older incumbent cannot dominate forever.
#[test]
fn evidence_decays_after_an_incumbent_change_and_over_time() {
    let winner = slot_of(CandidateStrategy::StructuralAdd);
    let accept = Experiment {
        number: 1,
        mix: round_robin_mix(),
        promoted: vec![winner],
        winner: Some((winner, 5e-6)),
        screen_ms: 9_000,
        promote_ms: 11_000,
    }
    .record();
    let quiet = Experiment {
        number: 2,
        mix: round_robin_mix(),
        promoted: vec![],
        winner: None,
        screen_ms: 9_000,
        promote_ms: 0,
    }
    .record();

    let mut ledger = StrategyLedger::new(DEFAULT_STRATEGY_EVIDENCE_DECAY, MIN_IMPROVEMENT);
    ledger.observe(&accept);
    let fresh_gain = ledger.evidence(CandidateStrategy::StructuralAdd).score_gain;
    assert!(fresh_gain > 0.0, "the accept is credited to its strategy");
    // The accept moved the incumbent, so the evidence it left is already
    // discounted the moment it is recorded — it describes the creature that
    // has just been replaced.
    assert!(
        fresh_gain < 5e-6,
        "an accept must discount its own evidence for the new incumbent: {fresh_gain}"
    );

    let mut decayed = fresh_gain;
    for _ in 0..3 {
        ledger.observe(&quiet);
        let now = ledger.evidence(CandidateStrategy::StructuralAdd).score_gain;
        assert!(now < decayed, "stale evidence must keep decaying: {now}");
        decayed = now;
    }
}

/// Decay has to reach the **decision**, not just the ledger. A discount that
/// scaled every field but left the allocation identical would satisfy the
/// letter of "evidence decays after an incumbent change" and none of its point.
#[test]
fn decayed_evidence_gives_back_slots_it_won() {
    let winner = slot_of(CandidateStrategy::StructuralAdd);
    let accept = Experiment {
        number: 1,
        mix: round_robin_mix(),
        promoted: vec![winner],
        winner: Some((winner, 5e-5)),
        screen_ms: 9_000,
        promote_ms: 11_000,
    }
    .record();
    let quiet = Experiment {
        number: 2,
        mix: round_robin_mix(),
        promoted: vec![],
        winner: None,
        screen_ms: 9_000,
        promote_ms: 0,
    }
    .record();

    let mut ledger = StrategyLedger::new(DEFAULT_STRATEGY_EVIDENCE_DECAY, MIN_IMPROVEMENT);
    ledger.observe(&accept);
    let slots_of = |ledger: &StrategyLedger| {
        ledger
            .allocate(
                adaptive_strategies(false),
                100,
                DEFAULT_STRATEGY_EXPLORATION_FLOOR,
            )
            .slots_for(CandidateStrategy::StructuralAdd)
            .expect("the winner is an arm")
    };
    let won = slots_of(&ledger);
    let value_after_accept = ledger.value(CandidateStrategy::StructuralAdd);

    for _ in 0..5 {
        ledger.observe(&quiet);
    }
    assert!(
        ledger.value(CandidateStrategy::StructuralAdd) < value_after_accept,
        "stale evidence must be worth less"
    );
    let kept = slots_of(&ledger);
    assert!(
        kept < won,
        "an operator that stopped earning must give slots back: {won} → {kept}"
    );
}

/// With no evidence at all, adaptive allocation is the even split — the
/// round-robin allocation it has to beat, not a random one.
#[test]
fn an_empty_ledger_allocates_evenly() {
    let ledger = StrategyLedger::new(DEFAULT_STRATEGY_EVIDENCE_DECAY, MIN_IMPROVEMENT);
    let arms = adaptive_strategies(false);
    let allocation = ledger.allocate(arms, 90, DEFAULT_STRATEGY_EXPLORATION_FLOOR);
    for arm in arms {
        assert_eq!(
            allocation.slots_for(*arm),
            Some(10),
            "{} should get an even share of an unmeasured budget",
            arm.label()
        );
    }
}

/// Acceptance criterion: the existing fixed allocation remains available, and
/// it is what an untouched configuration still runs.
#[test]
fn fixed_mode_is_the_default_and_allocates_nothing() {
    let config = LamarckConfig::default();
    assert_eq!(config.strategy_allocation, StrategyAllocationMode::Fixed);
    let policy = config
        .strategy_allocation_policy()
        .expect("the default is valid");
    assert!(!policy.is_adaptive());
    let ledger = ledger_with(&winning_run(6, CandidateStrategy::StructuralAdd, 4e-6));
    assert!(
        policy.allocate(&ledger, false, 100).is_none(),
        "fixed mode must hand the generator no allocation at all"
    );

    let adaptive = LamarckConfig {
        strategy_allocation: StrategyAllocationMode::Adaptive,
        ..LamarckConfig::default()
    }
    .strategy_allocation_policy()
    .expect("the adaptive defaults are valid");
    assert!(adaptive.is_adaptive());
    assert_eq!(
        adaptive
            .allocate(&ledger, false, 100)
            .map(|allocation| allocation.total_slots()),
        Some(100),
        "adaptive mode allocates the whole budget"
    );
}

/// A misconfigured knob stops the run rather than silently reverting to the
/// default — a run that ignored the flag would invalidate its own A/B.
#[test]
fn invalid_allocation_knobs_are_rejected_loudly() {
    for bad in [-0.1, 1.5, f64::NAN] {
        let config = LamarckConfig {
            strategy_allocation: StrategyAllocationMode::Adaptive,
            strategy_exploration_floor: bad,
            ..LamarckConfig::default()
        };
        let err = config
            .strategy_allocation_policy()
            .expect_err("an invalid floor must not fall back to the default");
        assert!(
            err.contains("--strategy-exploration-floor"),
            "error should name the flag: {err}"
        );
    }
    for bad in [0.0, -1.0, 1.5, f64::NAN] {
        let config = LamarckConfig {
            strategy_allocation: StrategyAllocationMode::Adaptive,
            strategy_evidence_decay: bad,
            ..LamarckConfig::default()
        };
        let err = config
            .strategy_allocation_policy()
            .expect_err("an invalid decay must not fall back to the default");
        assert!(
            err.contains("--strategy-evidence-decay"),
            "error should name the flag: {err}"
        );
    }
}

/// Acceptance criterion: journal / report expose allocated slots, trials,
/// accepts, score gain, cost and estimated strategy value.
#[test]
fn the_report_exposes_allocated_slots_trials_accepts_gain_cost_and_value() {
    let dir = tempfile::tempdir().expect("tempdir");
    let journal = dir.path().join("experiments.jsonl");
    let winner = slot_of(CandidateStrategy::StructuralAdd);
    let empty = StrategyLedger::new(DEFAULT_STRATEGY_EVIDENCE_DECAY, MIN_IMPROVEMENT);
    let allocation = empty.allocate(
        adaptive_strategies(false),
        90,
        DEFAULT_STRATEGY_EXPLORATION_FLOOR,
    );

    let mut lines = Vec::new();
    for number in 1..=3u64 {
        let mut line = Experiment {
            number,
            mix: round_robin_mix(),
            promoted: vec![winner],
            winner: Some((winner, 3e-6)),
            screen_ms: 9_000,
            promote_ms: 11_000,
        }
        .json();
        line["strategyAllocation"] =
            serde_json::to_value(&allocation).expect("allocation serialises");
        lines.push(line.to_string());
    }
    std::fs::write(&journal, format!("{}\n", lines.join("\n"))).expect("write journal");

    let report = report_from_journal(&journal).expect("report");
    let row = report
        .strategy_allocation
        .strategies
        .iter()
        .find(|row| row.strategy == CandidateStrategy::StructuralAdd.label())
        .expect("structural_add row");

    assert_eq!(row.allocated_slots, 30, "3 experiments x 10 even slots");
    assert_eq!(row.trials, 3, "one candidate per experiment");
    assert_eq!(row.accepts, 3);
    assert!(
        (row.score_gain - 9e-6).abs() < 1e-12,
        "score gain is the summed full-corpus Δ: {}",
        row.score_gain
    );
    assert!(
        row.cost_ms > 0.0,
        "the strategy is charged the scorer time its candidates caused"
    );
    assert!(
        row.estimated_value > 0.0,
        "an earning strategy has a positive estimated value"
    );

    let idle = report
        .strategy_allocation
        .strategies
        .iter()
        .find(|row| row.strategy == CandidateStrategy::Random.label())
        .expect("random row");
    assert_eq!(idle.accepts, 0);
    assert_eq!(idle.estimated_value, 0.0, "no return, no value");
    assert!(
        idle.cost_ms > 0.0,
        "an idle strategy is still charged for the screen time it consumed"
    );
}
