//! Noise-aware promote gate for the screen phase (issue #111).
//!
//! The pre-#111 gate promotes every candidate whose subsample Δ beats a fixed
//! `--screen-promote-threshold` of `1e-6`. `docs/screen-calibration.md` measured
//! what that buys: 244 promotions, 3 of which cleared the accept bar and 207 of
//! which made the creature worse. The threshold is about **one** standard
//! deviation of the screen's own noise, so the gate is a coin flip bought at
//! ~11 s of full-corpus scorer time each.
//!
//! This module expresses the gate in units of the batch's own measured spread
//! instead:
//!
//! - [`estimate_screen_sigma`] estimates σ̂ from the **lower quartile of the
//!   batch's absolute screen deltas**, rescaled as if that quartile came from a
//!   half-normal. A low quantile is used deliberately: a candidate batch is
//!   bimodal — structural proposals routinely move the score by 5e-2 while the
//!   weight/bias nudges move it by ~1e-8 — so the median or the standard
//!   deviation measure *proposal dispersion*, not the resolution floor the gate
//!   cares about. The quartile sits inside the near-zero core, where the deltas
//!   are the smallest effects the screen can still see.
//! - [`PromoteGate::threshold_for`] turns σ̂ into the Δ a candidate must clear:
//!   `max(k · σ̂, floor)`, where `floor` is the existing absolute threshold.
//!
//! Two properties are deliberate and pinned by tests below:
//!
//! - **The gate is never weaker than the absolute one.** Taking the max with
//!   the floor means the noise-aware gate can only ever promote a subset of
//!   what `--screen-promote-threshold` promotes, so nothing here can make a
//!   candidate acceptable on sampled evidence.
//! - **A degenerate batch falls back, loudly and safely.** Fewer than
//!   [`MIN_SIGMA_SAMPLES`] candidates, or a batch whose lower quartile is
//!   exactly zero, yields σ̂ = 0 and the gate reverts to the absolute floor
//!   rather than dividing by zero or promoting everything.
//!
//! The sample rate is **not** applied a second time: σ̂ is measured on scores
//! the run produced at its own `--screen-sample-rate`, so the rate is already
//! inside the estimate.

use crate::run::{ExperimentRecord, RunConfigRecord};
use serde::Serialize;
use std::collections::BTreeMap;

/// Default σ̂ multiplier for the noise-aware gate (issue #110's recommendation).
///
/// `docs/screen-calibration.md` measured the screen's noise floor at
/// σ ≈ 1.06e-6 and recommended a 3σ gate; it also states plainly that the
/// recommendation rests on two accepts, which is why the gate is opt-in.
pub const DEFAULT_SCREEN_PROMOTE_SIGMA_K: f64 = 3.0;

/// Candidates a batch needs before its spread is treated as an estimate.
///
/// Below this the quartile is one or two order statistics, which is a number,
/// not a scale. Such a batch falls back to the absolute floor.
pub const MIN_SIGMA_SAMPLES: usize = 4;

/// Quantile of |screen Δ| used as the scale statistic.
const SIGMA_QUANTILE: f64 = 0.25;

/// `z` such that `P(|Z| <= z) = 0.25` for a standard normal.
///
/// Dividing the observed lower quartile of |Δ| by this rescales it to a σ.
const HALF_NORMAL_QUARTILE_Z: f64 = 0.318_639_363_964_375_4;

/// Estimate the screen's per-batch noise scale σ̂ from its own deltas.
///
/// Returns `0.0` — "no usable estimate" — for a batch below
/// [`MIN_SIGMA_SAMPLES`], a batch carrying a non-finite delta, or a batch whose
/// lower quartile of |Δ| is zero (every proposal in the core scored exactly the
/// baseline). Callers treat `0.0` as "fall back to the absolute floor" rather
/// than as "everything clears the gate".
pub fn estimate_screen_sigma(deltas: &[f64]) -> f64 {
    if deltas.len() < MIN_SIGMA_SAMPLES || deltas.iter().any(|d| !d.is_finite()) {
        return 0.0;
    }
    let mut magnitudes: Vec<f64> = deltas.iter().map(|d| d.abs()).collect();
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quartile = interpolated_quantile(&magnitudes, SIGMA_QUANTILE);
    let sigma = quartile / HALF_NORMAL_QUARTILE_Z;
    if sigma.is_finite() && sigma > 0.0 {
        sigma
    } else {
        0.0
    }
}

/// Linear-interpolated quantile of an already-sorted, non-empty slice.
fn interpolated_quantile(sorted: &[f64], q: f64) -> f64 {
    let position = q * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(sorted.len() - 1);
    let fraction = position - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

/// Which promote gate a run is using.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromoteGateMode {
    /// Pre-#111 behaviour: a bare absolute screen Δ threshold. The default.
    Absolute,
    /// Promote on `Δ > max(k · σ̂, floor)` (issue #111). Opt-in.
    NoiseAware,
}

impl PromoteGateMode {
    /// Parse the CLI spelling; `None` for an unknown mode.
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "absolute" => Some(Self::Absolute),
            "noise-aware" | "noise_aware" => Some(Self::NoiseAware),
            _ => None,
        }
    }

    /// Stable label for journals, logs and reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::Absolute => "absolute",
            Self::NoiseAware => "noise-aware",
        }
    }
}

/// The promote gate in force for a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromoteGate {
    mode: PromoteGateMode,
    floor: f64,
    sigma_k: f64,
}

/// The Δ one batch of candidates has to clear, and how it was arrived at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateThreshold {
    /// Screen Δ a candidate must strictly exceed to be promoted.
    pub threshold: f64,
    /// σ̂ estimated for this batch; `None` under the absolute gate, and `None`
    /// when the batch was too degenerate to estimate one.
    pub sigma: Option<f64>,
}

impl PromoteGate {
    /// The pre-#111 gate: promote on `Δ > threshold`.
    pub fn absolute(threshold: f64) -> Self {
        Self {
            mode: PromoteGateMode::Absolute,
            floor: threshold,
            sigma_k: 0.0,
        }
    }

    /// The noise-aware gate: promote on `Δ > max(k · σ̂, floor)`.
    ///
    /// `floor` is `--screen-promote-threshold`, kept in force so the gate can
    /// never be weaker than the absolute one.
    pub fn noise_aware(floor: f64, sigma_k: f64) -> Self {
        Self {
            mode: PromoteGateMode::NoiseAware,
            floor,
            sigma_k,
        }
    }

    /// Which gate this is.
    pub fn mode(&self) -> PromoteGateMode {
        self.mode
    }

    /// Label for journals and logs.
    pub fn label(&self) -> &'static str {
        self.mode.label()
    }

    /// The absolute floor in force.
    pub fn floor(&self) -> f64 {
        self.floor
    }

    /// The σ̂ multiplier; `None` under the absolute gate.
    pub fn sigma_k(&self) -> Option<f64> {
        match self.mode {
            PromoteGateMode::Absolute => None,
            PromoteGateMode::NoiseAware => Some(self.sigma_k),
        }
    }

    /// Resolve the promote threshold for one batch of screen deltas.
    pub fn threshold_for(&self, deltas: &[f64]) -> GateThreshold {
        match self.mode {
            PromoteGateMode::Absolute => GateThreshold {
                threshold: self.floor,
                sigma: None,
            },
            PromoteGateMode::NoiseAware => {
                let sigma = estimate_screen_sigma(deltas);
                // σ̂ = 0 means "this batch could not price its own noise", so
                // the gate falls back to the floor rather than promoting all.
                let threshold = if sigma > 0.0 {
                    (self.sigma_k * sigma).max(self.floor)
                } else {
                    self.floor
                };
                GateThreshold {
                    threshold,
                    sigma: (sigma > 0.0).then_some(sigma),
                }
            }
        }
    }
}

/// One journal's answer to "what would this gate have promoted?" (issue #111).
///
/// Computed offline from journals that already carry `screenScores`, so the
/// replay costs no box time and can be asserted in `cargo test`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromoteGateReplay {
    /// Gate label the journal's run header recorded, when its headers agree.
    pub gate_as_run: Option<String>,
    /// σ̂ multiplier the replayed gate used.
    pub replay_sigma_k: f64,
    /// Absolute floor the replayed gate used (the journal's own threshold when
    /// its headers agree on one, else [`crate::config::DEFAULT_MIN_IMPROVEMENT`]).
    pub replay_floor: f64,
    /// Candidates that got a subsample score.
    pub screened: u64,
    /// Candidates the run actually promoted to full-corpus scoring.
    pub promoted_as_run: u64,
    /// Candidates the replayed gate would have promoted.
    pub promoted_under_gate: u64,
    /// Full-corpus scores the replayed gate would have avoided buying.
    pub promotions_avoided: u64,
    /// Accepted winners the replayed gate would still have promoted.
    pub accepts_kept: u64,
    /// Accepted winners the replayed gate would have discarded — the failure
    /// this replay exists to detect.
    pub accepts_dropped: u64,
    /// Every accepted winner, with the Δ it needed and the Δ the gate demanded.
    pub accepts: Vec<ReplayedAccept>,
}

/// One accepted winner, replayed against a candidate gate.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayedAccept {
    /// Experiment that accepted.
    pub experiment_number: u64,
    /// Winning stem.
    pub stem: String,
    /// Its screen Δ; `None` for a merged combo, which is assembled after the
    /// screen and so never had a subsample score of its own.
    pub screen_delta: Option<f64>,
    /// The Δ the replayed gate demanded of that batch.
    pub threshold: f64,
    /// σ̂ the replayed gate estimated for that batch.
    pub sigma: Option<f64>,
    /// Whether the gate would still have promoted this winner.
    ///
    /// A winner with no screen Δ (a combo) counts as kept: the screen never
    /// gated it, so no promote gate could have dropped it.
    pub would_promote: bool,
}

/// Streaming accumulator behind [`PromoteGateReplay`], driven by
/// `report_from_journal` so a journal is read once.
#[derive(Debug)]
pub struct PromoteGateReplayAccumulator {
    sigma_k: f64,
    floor: Option<f64>,
    floor_mixed: bool,
    gate_label: Option<String>,
    gate_mixed: bool,
    screened: u64,
    promoted_as_run: u64,
    promoted_under_gate: u64,
    accepts: Vec<ReplayedAccept>,
}

impl Default for PromoteGateReplayAccumulator {
    fn default() -> Self {
        Self::with_sigma_k(DEFAULT_SCREEN_PROMOTE_SIGMA_K)
    }
}

impl PromoteGateReplayAccumulator {
    /// Replay at a chosen σ̂ multiplier.
    pub fn with_sigma_k(sigma_k: f64) -> Self {
        Self {
            sigma_k,
            floor: None,
            floor_mixed: false,
            gate_label: None,
            gate_mixed: false,
            screened: 0,
            promoted_as_run: 0,
            promoted_under_gate: 0,
            accepts: Vec::new(),
        }
    }

    /// Record the knobs a run header states.
    pub fn push_header(&mut self, config: &RunConfigRecord) {
        match self.floor {
            Some(seen) if (seen - config.screen_promote_threshold).abs() >= f64::EPSILON => {
                self.floor_mixed = true;
            }
            _ => self.floor = Some(config.screen_promote_threshold),
        }
        // A journal written before #111 recorded no gate; it ran the absolute one.
        let label = config
            .screen_promote_gate
            .clone()
            .unwrap_or_else(|| PromoteGateMode::Absolute.label().to_string());
        match &self.gate_label {
            Some(seen) if *seen != label => self.gate_mixed = true,
            _ => self.gate_label = Some(label),
        }
    }

    /// Replay one experiment's screen batch against the candidate gate.
    ///
    /// Fails loudly on `screenScores` with no `baseline`: the deltas are
    /// measured against it, so a missing anchor cannot be silently skipped.
    pub fn push_experiment(&mut self, record: &ExperimentRecord) -> Result<(), String> {
        let Some(screen) = &record.screen_scores else {
            return Ok(());
        };
        let baseline = *screen.get("baseline").ok_or_else(|| {
            format!(
                "experiment {}: screenScores has no baseline to measure deltas against",
                record.experiment_number
            )
        })?;
        let deltas: Vec<f64> = screen
            .iter()
            .filter(|(stem, _)| stem.as_str() != "baseline")
            .map(|(_, score)| score - baseline)
            .collect();
        let gate = self.gate();
        let resolved = gate.threshold_for(&deltas);
        self.screened += deltas.len() as u64;
        self.promoted_under_gate +=
            deltas.iter().filter(|d| **d > resolved.threshold).count() as u64;
        // Only stems the screen also scored were *promoted*: a merged combo is
        // assembled after the screen, so counting it here would overstate what
        // the gate could have avoided.
        self.promoted_as_run += record
            .scores
            .keys()
            .filter(|stem| stem.as_str() != "baseline" && screen.contains_key(stem.as_str()))
            .count() as u64;

        if record.accepted
            && let Some(stem) = &record.winner
        {
            let screen_delta = screen.get(stem).map(|score| score - baseline);
            let would_promote = match screen_delta {
                Some(delta) => delta > resolved.threshold,
                // A merged combo has no screen score of its own; it survives
                // only if the gate would still have promoted every member it
                // was assembled from. Members that cannot be resolved — a
                // journal written before `comboMemberIndices` (issue #74) —
                // count as dropped, so an unknowable accept surfaces as a
                // failure rather than passing silently.
                None => combo_members_survive(record, screen, baseline, resolved.threshold),
            };
            self.accepts.push(ReplayedAccept {
                experiment_number: record.experiment_number,
                stem: stem.clone(),
                screen_delta,
                threshold: resolved.threshold,
                sigma: resolved.sigma,
                would_promote,
            });
        }
        Ok(())
    }

    /// The gate this replay applies, using the journal's own floor when its
    /// headers agree on one.
    fn gate(&self) -> PromoteGate {
        PromoteGate::noise_aware(self.replay_floor(), self.sigma_k)
    }

    fn replay_floor(&self) -> f64 {
        match (self.floor, self.floor_mixed) {
            (Some(floor), false) => floor,
            _ => crate::config::DEFAULT_MIN_IMPROVEMENT,
        }
    }

    /// Finish the accumulation into a report section.
    pub fn finish(self) -> PromoteGateReplay {
        let accepts_kept = self.accepts.iter().filter(|a| a.would_promote).count() as u64;
        PromoteGateReplay {
            gate_as_run: match self.gate_mixed {
                true => None,
                false => self.gate_label.clone(),
            },
            replay_sigma_k: self.sigma_k,
            replay_floor: self.replay_floor(),
            screened: self.screened,
            promoted_as_run: self.promoted_as_run,
            promoted_under_gate: self.promoted_under_gate,
            promotions_avoided: self
                .promoted_as_run
                .saturating_sub(self.promoted_under_gate),
            accepts_kept,
            accepts_dropped: self.accepts.len() as u64 - accepts_kept,
            accepts: self.accepts,
        }
    }
}

/// Whether every member a merged combo was assembled from clears the gate.
///
/// `false` when the members are unknown: a combo can only exist if its members
/// were promoted, so an unresolvable winner is treated as lost rather than kept.
fn combo_members_survive(
    record: &ExperimentRecord,
    screen: &BTreeMap<String, f64>,
    baseline: f64,
    threshold: f64,
) -> bool {
    let Some(indices) = &record.combo_member_indices else {
        return false;
    };
    if indices.is_empty() {
        return false;
    }
    indices.iter().all(|index| {
        screen
            .get(&format!("candidate-{index:03}"))
            .is_some_and(|score| score - baseline > threshold)
    })
}

/// Screen deltas of one score map, baseline excluded.
pub fn screen_deltas(scores: &BTreeMap<String, f64>) -> Option<Vec<f64>> {
    let baseline = *scores.get("baseline")?;
    Some(
        scores
            .iter()
            .filter(|(stem, _)| stem.as_str() != "baseline")
            .map(|(_, score)| score - baseline)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    /// Deterministic standard-normal sample (Box–Muller on a seeded stream).
    fn normal_sample(n: usize, sigma: f64, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let u1: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
                let u2: f64 = rng.random_range(0.0..1.0);
                sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            })
            .collect()
    }

    #[test]
    fn sigma_recovers_the_spread_of_a_synthetic_batch() {
        for sigma in [1e-6, 1e-3, 2.5] {
            let sample = normal_sample(4000, sigma, 7);
            let estimate = estimate_screen_sigma(&sample);
            assert!(
                (estimate - sigma).abs() < sigma * 0.1,
                "σ̂ {estimate:e} should recover σ {sigma:e} within 10%"
            );
        }
    }

    /// The reason a low quantile is used: a batch is bimodal, and a quarter of
    /// it can be catastrophic structural proposals without moving the estimate.
    #[test]
    fn sigma_ignores_a_contaminating_tail_of_catastrophic_candidates() {
        let mut sample = normal_sample(100, 1e-6, 11);
        let clean = estimate_screen_sigma(&sample);
        sample.extend(std::iter::repeat_n(-5e-2, 33));
        let contaminated = estimate_screen_sigma(&sample);
        assert!(
            (contaminated - clean).abs() < clean * 0.5,
            "33 catastrophic candidates moved σ̂ from {clean:e} to {contaminated:e}"
        );
    }

    /// Past the quartile's breakdown point the estimate inflates — the gate
    /// gets *stricter*, which is the safe direction — but it must never track
    /// the contaminating scale itself.
    #[test]
    fn heavier_contamination_inflates_sigma_without_tracking_the_tail() {
        let mut sample = normal_sample(100, 1e-6, 23);
        let clean = estimate_screen_sigma(&sample);
        sample.extend(std::iter::repeat_n(-5e-2, 100));
        let contaminated = estimate_screen_sigma(&sample);
        assert!(
            contaminated > clean,
            "50% contamination should not shrink σ̂: {clean:e} → {contaminated:e}"
        );
        assert!(
            contaminated < clean * 5.0 && contaminated < 1e-4,
            "σ̂ {contaminated:e} tracked the 5e-2 tail instead of the near-zero core"
        );
    }

    #[test]
    fn a_batch_of_identical_scores_yields_no_estimate_and_promotes_nothing() {
        let deltas = vec![0.0; 20];
        assert_eq!(estimate_screen_sigma(&deltas), 0.0);
        let gate = PromoteGate::noise_aware(1e-6, 3.0);
        let resolved = gate.threshold_for(&deltas);
        assert_eq!(resolved.sigma, None);
        assert_eq!(resolved.threshold, 1e-6, "degenerate batch falls back");
        assert!(
            !deltas.iter().any(|d| *d > resolved.threshold),
            "an identical batch must not promote everything"
        );
    }

    /// Identical *non-zero* deltas are still a degenerate batch: there is no
    /// spread to price, and the gate must not wave all of them through.
    #[test]
    fn a_batch_of_identical_positive_deltas_does_not_promote_everything() {
        let deltas = vec![5e-6; 20];
        let gate = PromoteGate::noise_aware(1e-6, 3.0);
        let resolved = gate.threshold_for(&deltas);
        assert!(
            !deltas.iter().any(|d| *d > resolved.threshold),
            "20 identical deltas cleared a {:e} gate",
            resolved.threshold
        );
    }

    #[test]
    fn a_single_candidate_falls_back_to_the_absolute_floor() {
        let gate = PromoteGate::noise_aware(1e-6, 3.0);
        for deltas in [vec![4e-6], vec![4e-6, 2e-6], vec![4e-6, 2e-6, 1e-7]] {
            let resolved = gate.threshold_for(&deltas);
            assert_eq!(
                resolved.sigma,
                None,
                "{} deltas is not a scale",
                deltas.len()
            );
            assert_eq!(resolved.threshold, 1e-6);
            assert!(
                deltas[0] > resolved.threshold,
                "a lone real improver must still be promoted"
            );
        }
    }

    #[test]
    fn an_all_negative_batch_promotes_nothing() {
        let deltas = vec![-1e-3, -5e-4, -2e-6, -1e-7, -9e-9, -1e-9];
        let gate = PromoteGate::noise_aware(1e-6, 3.0);
        let resolved = gate.threshold_for(&deltas);
        assert!(resolved.threshold > 0.0, "the gate stays strictly positive");
        assert_eq!(
            deltas.iter().filter(|d| **d > resolved.threshold).count(),
            0
        );
    }

    #[test]
    fn a_non_finite_delta_yields_no_estimate() {
        assert_eq!(
            estimate_screen_sigma(&[1e-6, f64::NAN, 2e-6, 3e-6, 4e-6]),
            0.0
        );
        assert_eq!(
            estimate_screen_sigma(&[1e-6, f64::INFINITY, 2e-6, 3e-6, 4e-6]),
            0.0
        );
    }

    /// The safety property: the noise-aware gate promotes a subset of what the
    /// absolute gate promotes, whatever the batch looks like.
    #[test]
    fn the_noise_aware_gate_is_never_weaker_than_the_absolute_one() {
        let batches = [
            normal_sample(60, 1e-6, 3),
            normal_sample(60, 1e-2, 5),
            vec![0.0, 0.0, 0.0, 1e-5, 2e-5],
            vec![-5e-2; 40],
        ];
        for batch in batches {
            let absolute = PromoteGate::absolute(1e-6).threshold_for(&batch);
            let noise_aware = PromoteGate::noise_aware(1e-6, 3.0).threshold_for(&batch);
            assert!(
                noise_aware.threshold >= absolute.threshold,
                "noise-aware {:e} must not undercut absolute {:e}",
                noise_aware.threshold,
                absolute.threshold
            );
        }
    }

    #[test]
    fn the_absolute_gate_reports_no_sigma_and_the_bare_threshold() {
        let gate = PromoteGate::absolute(1e-6);
        let resolved = gate.threshold_for(&normal_sample(50, 1e-3, 13));
        assert_eq!(resolved.threshold, 1e-6);
        assert_eq!(resolved.sigma, None);
        assert_eq!(gate.sigma_k(), None);
        assert_eq!(gate.label(), "absolute");
    }

    #[test]
    fn a_larger_k_never_promotes_more() {
        let batch = normal_sample(200, 2e-6, 17);
        let mut previous = f64::NEG_INFINITY;
        for k in [1.0, 2.0, 3.0, 5.0, 10.0] {
            let threshold = PromoteGate::noise_aware(1e-6, k)
                .threshold_for(&batch)
                .threshold;
            assert!(threshold >= previous, "k={k} loosened the gate");
            previous = threshold;
        }
    }

    #[test]
    fn mode_parsing_round_trips_and_rejects_unknown_spellings() {
        assert_eq!(
            PromoteGateMode::parse("absolute"),
            Some(PromoteGateMode::Absolute)
        );
        assert_eq!(
            PromoteGateMode::parse("Noise-Aware"),
            Some(PromoteGateMode::NoiseAware)
        );
        assert_eq!(PromoteGateMode::parse("sigma"), None);
        for mode in [PromoteGateMode::Absolute, PromoteGateMode::NoiseAware] {
            assert_eq!(PromoteGateMode::parse(mode.label()), Some(mode));
        }
    }

    /// A journal-shaped record with a screened batch, for the replay tests.
    fn experiment(number: u64, screen: &[(&str, f64)]) -> ExperimentRecord {
        use crate::candidates::{CandidateProvenance, CandidateStrategy};
        ExperimentRecord {
            experiment_number: number,
            timestamp_unix: 1000 + number,
            seed: Some(1),
            incumbent_id: "x".into(),
            baseline_score: 0.5,
            focus_neuron: "h1".into(),
            focus_neurons: None,
            focus_stats: None,
            candidates: vec![
                CandidateProvenance {
                    strategy: CandidateStrategy::Random,
                    focus_neuron: "h1".into(),
                    mutation: "x".into(),
                    old_value: None,
                    new_value: None,
                    mirror: None,
                };
                8
            ],
            candidates_requested: None,
            batch_limit: None,
            strategy_allocation: None,
            scores: BTreeMap::new(),
            mirror_axis_failures: None,
            screen_scores: Some(
                screen
                    .iter()
                    .map(|(stem, score)| ((*stem).to_string(), *score))
                    .collect(),
            ),
            screen_tiers: None,
            baseline_source: None,
            winner: None,
            improvement: None,
            accepted: false,
            analysis_ms: 1,
            memo_hits: 0,
            memo_misses: 0,
            memo_ms_saved: 0,
            scorer_ms: 2,
            scorer_calls: None,
            scorer_error: None,
            combo_members: None,
            combo_member_indices: None,
            combos_scored: None,
            combos_dampened: None,
            combo_dampen: None,
            cache_skipped: None,
            cache_deduplicated: None,
            cache_backfilled: None,
            cache_size: None,
            cache_lookup_ms: None,
            cache_maintenance_ms: None,
            cache_saved_ms: None,
            cache_spent_ms: None,
            cache_net_cumulative_ms: None,
            cache_resident_bytes: None,
            cache_rebuild_ms: None,
        }
    }

    /// A wobbly batch with one real improver: the gate keeps the improver.
    fn wobbly_batch() -> Vec<(&'static str, f64)> {
        vec![
            ("baseline", 0.5),
            ("candidate-000", 0.5 + 5e-5),
            ("candidate-001", 0.5 + 1.2e-6),
            ("candidate-002", 0.5 - 1.1e-6),
            ("candidate-003", 0.5 + 1.4e-6),
            ("candidate-004", 0.5 - 1.3e-6),
            ("candidate-005", 0.5 + 1.1e-6),
            ("candidate-006", 0.5 - 9e-7),
        ]
    }

    /// A merged combo has no screen score of its own, so it survives the replay
    /// only if every member it was assembled from would still be promoted.
    #[test]
    fn a_combo_accept_survives_only_when_all_its_members_do() {
        let promoted_members = vec![0usize];
        let wobble_member = vec![0usize, 1usize];
        for (members, expected) in [(promoted_members, true), (wobble_member, false)] {
            let mut record = experiment(1, &wobbly_batch());
            record.scores = BTreeMap::from([
                ("baseline".to_string(), 0.5),
                ("combo-000-k2".to_string(), 0.5 + 2e-6),
            ]);
            record.accepted = true;
            record.winner = Some("combo-000-k2".to_string());
            record.combo_member_indices = Some(members.clone());

            let mut acc = PromoteGateReplayAccumulator::with_sigma_k(3.0);
            acc.push_experiment(&record).expect("well-formed fixture");
            let replayed = acc.finish();
            assert_eq!(
                replayed.accepts[0].would_promote, expected,
                "members {members:?} should resolve to {expected}"
            );
        }
    }

    /// Pre-#74 journals name no members, so the combo cannot be shown to
    /// survive — and an unknowable accept must surface, not pass silently.
    #[test]
    fn a_combo_accept_with_unknowable_members_counts_as_dropped() {
        let mut record = experiment(1, &wobbly_batch());
        record.scores = BTreeMap::from([
            ("baseline".to_string(), 0.5),
            ("combo-000-k2".to_string(), 0.5 + 2e-6),
        ]);
        record.accepted = true;
        record.winner = Some("combo-000-k2".to_string());
        record.combo_member_indices = None;

        let mut acc = PromoteGateReplayAccumulator::with_sigma_k(3.0);
        acc.push_experiment(&record).expect("well-formed fixture");
        let replayed = acc.finish();
        assert_eq!(replayed.accepts_dropped, 1);
        assert_eq!(replayed.accepts_kept, 0);
    }

    /// A combo stem in `scores` was never screened, so counting it as a
    /// promotion would overstate what the gate could have avoided.
    #[test]
    fn only_screened_stems_count_as_promotions() {
        let mut record = experiment(1, &wobbly_batch());
        record.scores = BTreeMap::from([
            ("baseline".to_string(), 0.5),
            ("candidate-000".to_string(), 0.5 - 1e-4),
            ("candidate-001".to_string(), 0.5 - 1e-4),
            ("combo-000-k2".to_string(), 0.5 - 1e-4),
        ]);
        let mut acc = PromoteGateReplayAccumulator::with_sigma_k(3.0);
        acc.push_experiment(&record).expect("well-formed fixture");
        let replayed = acc.finish();
        assert_eq!(replayed.promoted_as_run, 2, "the combo is not a promotion");
        assert_eq!(replayed.screened, 7);
        assert_eq!(replayed.promoted_under_gate, 1, "only the real improver");
        assert_eq!(replayed.promotions_avoided, 1);
    }

    /// A screened batch with no baseline anchor cannot be replayed silently.
    #[test]
    fn a_screen_map_without_a_baseline_fails_loudly() {
        let record = experiment(1, &[("candidate-000", 0.5)]);
        let mut acc = PromoteGateReplayAccumulator::with_sigma_k(3.0);
        let err = acc.push_experiment(&record).expect_err("no anchor");
        assert!(err.contains("baseline"), "error names the anchor: {err}");
    }

    #[test]
    fn screen_deltas_excludes_the_baseline_anchor() {
        let scores = BTreeMap::from([
            ("baseline".to_string(), 0.5),
            ("candidate-000".to_string(), 0.5 + 1e-6),
            ("candidate-001".to_string(), 0.5 - 2e-6),
        ]);
        let mut deltas = screen_deltas(&scores).expect("baseline present");
        deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(deltas.len(), 2);
        assert!((deltas[1] - 1e-6).abs() < 1e-12);
        assert!(screen_deltas(&BTreeMap::new()).is_none());
    }
}
