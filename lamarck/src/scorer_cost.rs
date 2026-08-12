//! Per-call scorer cost decomposition (issue #112).
//!
//! Lamarck spawns the scorer once per batch, so every call pays a **fixed**
//! cost — process start, training-corpus open, per-run setup — before it scores
//! its first creature, then a **marginal** cost per creature after that. Scorer
//! time is ~83% of a run's wall clock (`docs/baseline-economics.md`), so which
//! of the two dominates decides whether a persistent scoring session is worth
//! the cross-repo protocol change it would need.
//!
//! The decomposition is an ordinary least-squares fit of measured milliseconds
//! against the creature count of the call: the intercept is the fixed cost, the
//! slope the marginal cost per creature. Calls are fitted **per phase** —
//! a screen call scores a subsample and a promote call the whole corpus, so
//! their marginal costs differ by the sample rate and pooling them would fit a
//! line through two different populations.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Which run phase made a scorer call (issue #112).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ScorerCallPhase {
    /// Phase-0 authoritative baseline / parity gate.
    Phase0,
    /// Phase-G structural graft replay.
    GraftReplay,
    /// Screen tier — the candidate batch on a corpus subsample.
    Screen,
    /// Promote tier — the screened survivors on the full corpus.
    Promote,
    /// Combination scoring of promoted improvers.
    Combo,
}

impl ScorerCallPhase {
    /// Stable label used as the report/journal key.
    pub fn label(self) -> &'static str {
        match self {
            Self::Phase0 => "phase0",
            Self::GraftReplay => "graftReplay",
            Self::Screen => "screen",
            Self::Promote => "promote",
            Self::Combo => "combo",
        }
    }
}

/// One authoritative scorer invocation, as journalled (issue #112).
///
/// The creature count is what makes the run's own `scorerMs` regressable: with
/// it, any future `experiments.jsonl` reproduces the fixed/marginal split
/// without a bespoke benchmark.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScorerCallRecord {
    /// Run phase that made the call.
    pub phase: ScorerCallPhase,
    /// Creature files handed to the scorer (baseline included).
    pub creatures: u64,
    /// Corpus subsample rate when the call sampled; omitted for full corpus.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<f64>,
    /// Wall-clock milliseconds the call took, measured around the invocation.
    pub elapsed_ms: u128,
    /// Whether the call failed. A failed call exits early, so its time is
    /// counted but never fitted — an aborted call would drag the intercept down.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub failed: bool,
}

/// Fixed / marginal cost fit for one phase's calls (issue #112).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CallCostFit {
    /// Successful calls fitted.
    pub calls: u64,
    /// Distinct creature counts among them — the fit needs at least two.
    pub distinct_sizes: u64,
    /// Mean creatures per call.
    pub mean_creatures: f64,
    /// Mean milliseconds per call.
    pub mean_ms: f64,
    /// Fixed cost per call (the regression intercept), `None` below two sizes.
    pub fixed_ms: Option<f64>,
    /// Marginal cost per creature (the regression slope), `None` below two sizes.
    pub marginal_ms_per_creature: Option<f64>,
    /// Fraction of the variance in call time the fit explains.
    pub r_squared: Option<f64>,
    /// Fixed cost as a share of an average call — the number a persistent
    /// scoring session would save.
    pub fixed_ms_share_at_mean: Option<f64>,
}

/// Fit one phase's `(creatures, milliseconds)` pairs by least squares.
///
/// A single creature count carries no slope information, so the fit reports the
/// means and leaves the decomposition `None` rather than inventing an intercept
/// from one point.
pub fn fit_calls(calls: &[(u64, u128)]) -> CallCostFit {
    let n = calls.len();
    if n == 0 {
        return CallCostFit::default();
    }
    let sizes: BTreeSet<u64> = calls.iter().map(|(creatures, _)| *creatures).collect();
    let count = n as f64;
    let mean_x = calls.iter().map(|(c, _)| *c as f64).sum::<f64>() / count;
    let mean_y = calls.iter().map(|(_, ms)| *ms as f64).sum::<f64>() / count;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for (creatures, ms) in calls {
        let dx = *creatures as f64 - mean_x;
        let dy = *ms as f64 - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let mut fit = CallCostFit {
        calls: n as u64,
        distinct_sizes: sizes.len() as u64,
        mean_creatures: mean_x,
        mean_ms: mean_y,
        fixed_ms: None,
        marginal_ms_per_creature: None,
        r_squared: None,
        fixed_ms_share_at_mean: None,
    };
    if sizes.len() < 2 || sxx <= 0.0 {
        return fit;
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let residual: f64 = calls
        .iter()
        .map(|(creatures, ms)| {
            let predicted = intercept + slope * *creatures as f64;
            let error = *ms as f64 - predicted;
            error * error
        })
        .sum();
    fit.fixed_ms = Some(intercept);
    fit.marginal_ms_per_creature = Some(slope);
    fit.r_squared = (syy > 0.0).then(|| 1.0 - residual / syy);
    fit.fixed_ms_share_at_mean = (mean_y > 0.0).then_some(intercept / mean_y);
    fit
}

/// Per-call scorer cost across a whole journal (issue #112).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScorerCallCost {
    /// Scorer invocations the journal recorded, failures included.
    pub calls: u64,
    /// Invocations that failed, and are therefore excluded from every fit.
    pub failed_calls: u64,
    /// Creatures handed to the scorer across the successful calls.
    pub creatures_scored: u64,
    /// Fixed / marginal fit per phase, keyed by [`ScorerCallPhase::label`].
    ///
    /// Deliberately never pooled across phases: a sampled screen call and a
    /// full-corpus promote call have different marginal costs, so one line
    /// through both would report neither.
    pub by_phase: BTreeMap<String, CallCostFit>,
}

/// Streaming collector behind a [`ScorerCallCost`].
#[derive(Debug, Default)]
pub struct ScorerCallCostAccumulator {
    calls: u64,
    failed_calls: u64,
    creatures_scored: u64,
    per_phase: BTreeMap<String, Vec<(u64, u128)>>,
}

impl ScorerCallCostAccumulator {
    /// Fold one journalled call into the per-phase fits.
    pub fn push(&mut self, call: &ScorerCallRecord) {
        self.calls += 1;
        if call.failed {
            self.failed_calls += 1;
            return;
        }
        self.creatures_scored += call.creatures;
        self.per_phase
            .entry(call.phase.label().to_string())
            .or_default()
            .push((call.creatures, call.elapsed_ms));
    }

    /// Fold every call of one journal line.
    pub fn push_all(&mut self, calls: &[ScorerCallRecord]) {
        for call in calls {
            self.push(call);
        }
    }

    /// Finish the per-phase regressions.
    pub fn finish(self) -> ScorerCallCost {
        ScorerCallCost {
            calls: self.calls,
            failed_calls: self.failed_calls,
            creatures_scored: self.creatures_scored,
            by_phase: self
                .per_phase
                .into_iter()
                .map(|(phase, calls)| (phase, fit_calls(&calls)))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_perfect_line_recovers_its_intercept_and_slope() {
        // ms = 5000 + 300 * creatures, exactly.
        let calls = [(1u64, 5300u128), (2, 5600), (30, 14_000)];
        let fit = fit_calls(&calls);
        assert_eq!(fit.calls, 3);
        assert_eq!(fit.distinct_sizes, 3);
        let fixed = fit.fixed_ms.expect("intercept");
        let marginal = fit.marginal_ms_per_creature.expect("slope");
        assert!((fixed - 5000.0).abs() < 1e-6, "fixed={fixed}");
        assert!((marginal - 300.0).abs() < 1e-9, "marginal={marginal}");
        assert!((fit.r_squared.expect("r2") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_pure_per_creature_cost_has_no_fixed_component() {
        let calls = [(1u64, 400u128), (4, 1600), (10, 4000)];
        let fit = fit_calls(&calls);
        assert!(fit.fixed_ms.expect("intercept").abs() < 1e-9);
        assert!((fit.marginal_ms_per_creature.expect("slope") - 400.0).abs() < 1e-9);
        assert!(fit.fixed_ms_share_at_mean.expect("share").abs() < 1e-9);
    }

    #[test]
    fn the_fixed_share_is_reported_against_an_average_call() {
        // Fixed 4000 ms, marginal 100 ms; mean size 10 ⇒ mean 5000 ms ⇒ 80%.
        let calls = [(5u64, 4500u128), (10, 5000), (15, 5500)];
        let fit = fit_calls(&calls);
        assert!((fit.mean_ms - 5000.0).abs() < 1e-9);
        assert!((fit.fixed_ms_share_at_mean.expect("share") - 0.8).abs() < 1e-9);
    }

    /// One creature count cannot separate fixed from marginal cost, and a fit
    /// that invented an intercept from it would be the mis-measurement issue
    /// #112 warns a go/no-go must not be made on.
    #[test]
    fn a_single_creature_count_reports_no_decomposition() {
        let calls = [(30u64, 12_000u128), (30, 12_400), (30, 11_800)];
        let fit = fit_calls(&calls);
        assert_eq!(fit.calls, 3);
        assert_eq!(fit.distinct_sizes, 1);
        assert_eq!(fit.fixed_ms, None);
        assert_eq!(fit.marginal_ms_per_creature, None);
        assert!((fit.mean_ms - 12_066.666_666_666_666).abs() < 1e-6);
    }

    #[test]
    fn no_calls_is_an_empty_fit_not_a_zero_intercept() {
        let fit = fit_calls(&[]);
        assert_eq!(fit.calls, 0);
        assert_eq!(fit.fixed_ms, None);
        assert_eq!(fit.r_squared, None);
    }

    #[test]
    fn phases_are_fitted_separately() {
        let mut acc = ScorerCallCostAccumulator::default();
        // Screen: 500 fixed + 20/creature. Promote: 5000 fixed + 400/creature.
        for (creatures, ms) in [(2u64, 540u128), (30, 1100)] {
            acc.push(&ScorerCallRecord {
                phase: ScorerCallPhase::Screen,
                creatures,
                sample_rate: Some(0.05),
                elapsed_ms: ms,
                failed: false,
            });
        }
        for (creatures, ms) in [(2u64, 5800u128), (5, 7000)] {
            acc.push(&ScorerCallRecord {
                phase: ScorerCallPhase::Promote,
                creatures,
                sample_rate: None,
                elapsed_ms: ms,
                failed: false,
            });
        }
        let cost = acc.finish();
        assert_eq!(cost.calls, 4);
        assert_eq!(cost.creatures_scored, 39);
        let screen = &cost.by_phase["screen"];
        assert!((screen.fixed_ms.expect("screen fixed") - 500.0).abs() < 1e-6);
        assert!((screen.marginal_ms_per_creature.expect("screen slope") - 20.0).abs() < 1e-9);
        let promote = &cost.by_phase["promote"];
        assert!((promote.fixed_ms.expect("promote fixed") - 5000.0).abs() < 1e-6);
        assert!((promote.marginal_ms_per_creature.expect("promote slope") - 400.0).abs() < 1e-9);
    }

    /// A failed call exits before it scores anything, so fitting its time would
    /// pull the intercept towards zero and understate the fixed cost.
    #[test]
    fn failed_calls_are_counted_but_never_fitted() {
        let mut acc = ScorerCallCostAccumulator::default();
        acc.push(&ScorerCallRecord {
            phase: ScorerCallPhase::Promote,
            creatures: 3,
            sample_rate: None,
            elapsed_ms: 12,
            failed: true,
        });
        for (creatures, ms) in [(1u64, 5100u128), (11, 6000)] {
            acc.push(&ScorerCallRecord {
                phase: ScorerCallPhase::Promote,
                creatures,
                sample_rate: None,
                elapsed_ms: ms,
                failed: false,
            });
        }
        let cost = acc.finish();
        assert_eq!(cost.calls, 3);
        assert_eq!(cost.failed_calls, 1);
        assert_eq!(cost.creatures_scored, 12);
        let promote = &cost.by_phase["promote"];
        assert_eq!(promote.calls, 2);
        assert!((promote.fixed_ms.expect("fixed") - 5010.0).abs() < 1e-6);
    }

    #[test]
    fn phase_labels_round_trip_through_json() {
        for phase in [
            ScorerCallPhase::Phase0,
            ScorerCallPhase::GraftReplay,
            ScorerCallPhase::Screen,
            ScorerCallPhase::Promote,
            ScorerCallPhase::Combo,
        ] {
            let json = serde_json::to_string(&phase).expect("serialises");
            assert_eq!(json, format!("\"{}\"", phase.label()));
            let back: ScorerCallPhase = serde_json::from_str(&json).expect("parses");
            assert_eq!(back, phase);
        }
    }
}
