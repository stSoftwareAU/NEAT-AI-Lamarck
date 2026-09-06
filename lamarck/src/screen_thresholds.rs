//! Per-strategy screen threshold calibration (issue #220).
//!
//! The screen phase gates every candidate family on one number. A tiny weight
//! perturbation, a structural add and a backprop step do not share a
//! relationship between sampled Δ and full-corpus Δ, so a shared gate spends
//! full-corpus calls on families that never convert while dropping families
//! whose sample signal is weak but whose promoted precision is good.
//!
//! This module learns that relationship **per strategy** from the journal the
//! run is already writing, and turns it into a bounded per-strategy adjustment
//! of the gate the run would otherwise have applied.
//!
//! # What is learned, and how it is applied
//!
//! Evidence is the journal's own paired observations: every candidate that was
//! screened **and** full-corpus scored contributes `(screen Δ, full Δ)` to the
//! window of the strategy that proposed it. From that window,
//! [`calibrated_multiplier`] derives a multiplier on the shared threshold:
//!
//! * **Win margin** — with at least one promoted candidate whose full Δ cleared
//!   the accept bar, the threshold is set to half the *smallest* winning screen
//!   Δ **in the window**, with a 2× margin underneath it. A window is 128
//!   observations, so a winner that ages out stops holding the bar down.
//! * **Loss quantile** — a family with enough promotions and no winner at all
//!   has only wasted calls to learn from, so the threshold moves to the median
//!   screen Δ of its losing promotions: half of what it bought would not have
//!   been bought. That sample is **censored** — it holds only what the gate in
//!   force promoted — so the branch is braked by the weakest *improving*
//!   candidate the family has shown, and declines entirely when the screen
//!   scored that improvement at or below zero. Without the brake the branch
//!   would ratchet on its own past tightening rather than on evidence.
//! * **Shared** — below [`MIN_CALIBRATION_PAIRS`] paired observations, or with
//!   no usable statistic, the strategy falls back to the run's shared threshold
//!   exactly. Insufficient evidence changes nothing.
//!
//! The multiplier is clamped to `[1/`[`MAX_THRESHOLD_ADJUSTMENT`]`,
//! `[`MAX_THRESHOLD_ADJUSTMENT`]`]`, and applied to the threshold the batch's
//! own gate resolved — so it composes with the absolute and noise-aware gates
//! rather than replacing either.
//!
//! # The guardrail
//!
//! **Calibration never makes the screen authoritative.** It only decides which
//! candidates are worth a full-corpus score; the full-corpus scorer remains the
//! only acceptance gate, and nothing here can accept a candidate.
//!
//! A screen that is only ever measured on what it promoted cannot see its own
//! false negatives — `docs/screen-calibration.md` says exactly that. So a
//! calibrated run promotes a **control sample** of below-threshold candidates,
//! drawn uniformly at random from the rejected set at a minimum rate of
//! [`ScreenThresholdPolicy::control_rate`]. Their full-corpus scores are the
//! only measurement of what the gate is throwing away, and they are the only
//! way a winner with a weak screen Δ can ever enter the evidence.

use crate::candidates::CandidateStrategy;
use crate::run::{ExperimentRecord, candidate_stem_index};
use crate::scorer_cost::ScorerCallPhase;
use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// Version stamped on every calibrated screen decision.
///
/// Journalled per candidate so a journal read months later says which
/// estimator produced the threshold that candidate faced. Bump it whenever
/// [`calibrated_multiplier`] changes shape.
pub const SCREEN_THRESHOLD_MODEL_VERSION: &str = "per-strategy-screen-v1";

/// Default share of below-threshold candidates promoted as controls.
///
/// `0.02` on a 100-candidate batch is one or two full-corpus scores an
/// uncalibrated run would not have bought — the price of being able to measure
/// a false-negative rate at all. It is a **minimum**: the count is rounded up,
/// so a batch with any rejection at all promotes at least one control.
pub const DEFAULT_SCREEN_CONTROL_RATE: f64 = 0.02;

/// Paired observations a strategy needs before its own threshold is used.
///
/// Below this the window is a handful of points from one incumbent, and a
/// threshold drawn from it would be noise dressed as calibration. Such a
/// strategy runs on the shared threshold.
pub const MIN_CALIBRATION_PAIRS: usize = 8;

/// Most recent paired observations retained per strategy.
///
/// Evidence ages out by recency rather than by a decay factor: the estimator is
/// a quantile, and a quantile of fractionally weighted points is not a number
/// anybody can check by hand. A window keeps it a plain order statistic over
/// the observations closest to the creature the run is optimising now.
pub const CALIBRATION_WINDOW: usize = 128;

/// Share of the smallest winning screen Δ the win-margin threshold keeps.
///
/// Half: the gate sits a factor of two below the weakest sampled signal that
/// has ever produced a full-corpus win for that family.
pub const WIN_MARGIN: f64 = 0.5;

/// Quantile of losing screen Δ used when a strategy has shown no winner.
pub const LOSS_QUANTILE: f64 = 0.5;

/// Hardest adjustment calibration may make in either direction.
///
/// The measured threshold is clamped to `[floor/8, floor×8]`. A bound is what
/// keeps a thin window from silencing a family outright, or from opening the
/// gate so far that the screen stops screening.
pub const MAX_THRESHOLD_ADJUSTMENT: f64 = 8.0;

/// Strategy label recorded for a stem whose provenance cannot be resolved.
///
/// A combo assembled after the screen, or a journal too old to carry
/// provenances, has no strategy to calibrate against; such candidates run on
/// the shared threshold and are pooled under this label rather than dropped.
pub const UNATTRIBUTED_STRATEGY: &str = "unattributed";

/// Which screen threshold a run applies (issue #220).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScreenThresholdMode {
    /// The pre-#220 run: one threshold for every strategy.
    #[default]
    Shared,
    /// Per-strategy thresholds calibrated from measured history.
    PerStrategy,
}

impl ScreenThresholdMode {
    /// Parse a CLI spelling (`shared` / `per-strategy`).
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "shared" => Some(Self::Shared),
            "per-strategy" | "per_strategy" => Some(Self::PerStrategy),
            _ => None,
        }
    }

    /// Stable label for logs, journals and reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::PerStrategy => "per-strategy",
        }
    }

    /// True when per-strategy thresholds are in force.
    pub fn is_per_strategy(self) -> bool {
        matches!(self, Self::PerStrategy)
    }
}

/// Validated screen-threshold knobs for a run (issue #220).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenThresholdPolicy {
    /// Threshold mode in force.
    pub mode: ScreenThresholdMode,
    /// Minimum share of below-threshold candidates promoted as controls.
    pub control_rate: f64,
}

impl Default for ScreenThresholdPolicy {
    fn default() -> Self {
        Self {
            mode: ScreenThresholdMode::Shared,
            control_rate: DEFAULT_SCREEN_CONTROL_RATE,
        }
    }
}

/// One candidate scored on both the subsample and the full corpus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedScreenObservation {
    /// Subsample score minus the subsample baseline.
    pub screen_delta: f64,
    /// Full-corpus score minus the full-corpus baseline.
    pub full_delta: f64,
    /// Whether the candidate was promoted as a below-threshold control.
    pub control: bool,
}

/// How a strategy's threshold was arrived at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdBasis {
    /// Insufficient evidence — the run's shared threshold, unchanged.
    Shared,
    /// Half the smallest screen Δ that produced a full-corpus win.
    WinMargin,
    /// The median screen Δ of promotions that never improved.
    LossQuantile,
}

impl ThresholdBasis {
    /// Stable label for journals and reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::WinMargin => "win-margin",
            Self::LossQuantile => "loss-quantile",
        }
    }
}

/// What calibration decided for one strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrategyThreshold {
    /// Multiplier applied to the batch's shared threshold.
    pub multiplier: f64,
    /// How that multiplier was arrived at.
    pub basis: ThresholdBasis,
    /// Paired observations behind it.
    pub pairs: usize,
}

impl StrategyThreshold {
    /// The unchanged shared threshold — what insufficient evidence yields.
    pub fn shared(pairs: usize) -> Self {
        Self {
            multiplier: 1.0,
            basis: ThresholdBasis::Shared,
            pairs,
        }
    }
}

/// Threshold and model version one candidate faced (issue #220).
///
/// One entry per screened candidate, so every candidate in the journal carries
/// the threshold it was judged against and the estimator version that produced
/// it — including the controls, which were promoted *despite* their threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateScreenDecision {
    /// Candidate stem (`candidate-NNN`).
    pub stem: String,
    /// Strategy that proposed it, or [`UNATTRIBUTED_STRATEGY`].
    pub strategy: String,
    /// Screen Δ the candidate had to clear.
    pub threshold: f64,
    /// Calibration model version that produced that threshold.
    pub model_version: String,
    /// Whether the candidate was promoted to full-corpus scoring.
    pub promoted: bool,
    /// Whether it was promoted as a below-threshold control.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub control: bool,
}

/// What the calibrated gate did to one screened batch (issue #220).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScreenThresholdRecord {
    /// Calibration model version in force.
    pub model_version: String,
    /// Threshold mode (`shared` / `per-strategy`).
    pub mode: String,
    /// Threshold the batch's own gate resolved, before calibration.
    pub shared_threshold: f64,
    /// Minimum control-promotion rate in force.
    pub control_rate: f64,
    /// Applied threshold per strategy label.
    pub thresholds: BTreeMap<String, f64>,
    /// How each strategy's threshold was arrived at.
    pub basis: BTreeMap<String, String>,
    /// Candidates promoted as below-threshold controls.
    pub controls: u64,
    /// Every screened candidate, with the threshold and version it faced.
    pub candidates: Vec<CandidateScreenDecision>,
}

/// The calibrated gate's verdict on one batch.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedScreen {
    /// Stems admitted to full-corpus scoring, best screen Δ first.
    pub stems: Vec<String>,
    /// What to journal about the decision.
    pub record: ScreenThresholdRecord,
}

/// One screened candidate, as the gate sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenedCandidate {
    /// Candidate stem.
    pub stem: String,
    /// Strategy that proposed it, when the stem indexes a provenance.
    pub strategy: Option<CandidateStrategy>,
    /// Subsample score minus the subsample baseline.
    pub screen_delta: f64,
}

/// Per-strategy screen evidence, and the thresholds drawn from it (#220).
///
/// Fed from journalled [`ExperimentRecord`]s — not from the run's in-memory
/// state — so `report` replaying a journal derives exactly the thresholds the
/// run applied.
#[derive(Debug, Clone)]
pub struct ScreenThresholdLedger {
    floor: f64,
    accept_bar: f64,
    window: usize,
    arms: BTreeMap<String, VecDeque<PairedScreenObservation>>,
}

impl ScreenThresholdLedger {
    /// A ledger calibrating against `floor` (the run's shared threshold) and
    /// `accept_bar` (`--min-improvement`, what counts as a win).
    pub fn new(floor: f64, accept_bar: f64) -> Self {
        Self {
            floor,
            accept_bar,
            window: CALIBRATION_WINDOW,
            arms: BTreeMap::new(),
        }
    }

    /// The shared threshold every multiplier is measured against.
    pub fn floor(&self) -> f64 {
        self.floor
    }

    /// Replace the shared threshold (a report learns it from the run header
    /// after the ledger is built).
    pub fn set_floor(&mut self, floor: f64) {
        self.floor = floor;
    }

    /// Replace the accept bar.
    pub fn set_accept_bar(&mut self, accept_bar: f64) {
        self.accept_bar = accept_bar;
    }

    /// Strategy labels the ledger holds evidence for.
    pub fn strategies(&self) -> impl Iterator<Item = &str> {
        self.arms.keys().map(String::as_str)
    }

    /// Push one observation straight into a strategy's window.
    ///
    /// Test scaffolding for the estimator: a run only ever fills the ledger
    /// through [`Self::observe`], from journalled records.
    #[cfg(test)]
    fn push_for_test(&mut self, strategy: &str, observation: PairedScreenObservation) {
        let window = self.arms.entry(strategy.to_string()).or_default();
        window.push_back(observation);
        while window.len() > self.window {
            window.pop_front();
        }
    }

    /// Every paired observation held for `strategy`, in arrival order.
    pub fn observations(&self, strategy: &str) -> Vec<PairedScreenObservation> {
        self.arms
            .get(strategy)
            .map(|window| window.iter().copied().collect())
            .unwrap_or_default()
    }

    /// The threshold calibration puts on `strategy` right now.
    pub fn threshold_for(&self, strategy: &str) -> StrategyThreshold {
        let observations = self.observations(strategy);
        if observations.len() < MIN_CALIBRATION_PAIRS {
            return StrategyThreshold::shared(observations.len());
        }
        match calibrated_multiplier(&observations, self.floor, self.accept_bar) {
            Some((multiplier, basis)) => StrategyThreshold {
                multiplier,
                basis,
                pairs: observations.len(),
            },
            None => StrategyThreshold::shared(observations.len()),
        }
    }

    /// Fold one journalled experiment's paired observations into the ledger.
    ///
    /// An experiment with no screen phase, or one that promoted nothing, pairs
    /// nothing — there is no full-corpus score to learn from.
    pub fn observe(&mut self, record: &ExperimentRecord) {
        let Some(screen) = &record.screen_scores else {
            return;
        };
        let (Some(screen_baseline), Some(full_baseline)) =
            (screen.get("baseline"), record.scores.get("baseline"))
        else {
            return;
        };
        let controls = control_stems(record);
        for (stem, screen_score) in screen {
            if stem == "baseline" {
                continue;
            }
            let Some(full_score) = record.scores.get(stem) else {
                continue;
            };
            let strategy = strategy_label(record, stem);
            let window = self.arms.entry(strategy).or_default();
            window.push_back(PairedScreenObservation {
                screen_delta: screen_score - screen_baseline,
                full_delta: full_score - full_baseline,
                control: controls.contains(stem.as_str()),
            });
            while window.len() > self.window {
                window.pop_front();
            }
        }
    }
}

/// Stems this experiment promoted as below-threshold controls.
///
/// Empty for a shared-threshold experiment and for a journal written before the
/// record existed — neither promoted a control.
pub fn control_stems(record: &ExperimentRecord) -> std::collections::BTreeSet<&str> {
    record
        .screen_thresholds
        .iter()
        .flat_map(|thresholds| thresholds.candidates.iter())
        .filter(|candidate| candidate.control)
        .map(|candidate| candidate.stem.as_str())
        .collect()
}

/// Strategy behind a stem, when the stem indexes a journalled provenance.
///
/// `None` for a merged combo — assembled after the screen, so no strategy
/// proposed it under that name — and for a journal too old to carry
/// provenances.
pub fn strategy_of(record: &ExperimentRecord, stem: &str) -> Option<String> {
    candidate_stem_index(stem)
        .and_then(|index| record.candidates.get(index))
        .map(|provenance| provenance.strategy.label().to_string())
}

/// Strategy label behind a stem, or [`UNATTRIBUTED_STRATEGY`].
pub fn strategy_label(record: &ExperimentRecord, stem: &str) -> String {
    strategy_of(record, stem).unwrap_or_else(|| UNATTRIBUTED_STRATEGY.to_string())
}

/// Derive a strategy's threshold multiplier from its paired observations.
///
/// `None` means "no usable statistic" — the caller keeps the shared threshold.
/// A non-positive or non-finite `floor` also yields `None`: a multiplier is
/// meaningless against a threshold that is not a positive scale.
pub fn calibrated_multiplier(
    pairs: &[PairedScreenObservation],
    floor: f64,
    accept_bar: f64,
) -> Option<(f64, ThresholdBasis)> {
    if !floor.is_finite() || floor <= 0.0 {
        return None;
    }
    let bar = if accept_bar.is_finite() && accept_bar > 0.0 {
        accept_bar
    } else {
        0.0
    };
    let wins: Vec<f64> = pairs
        .iter()
        .filter(|pair| pair.full_delta > bar && pair.screen_delta.is_finite())
        .map(|pair| pair.screen_delta)
        .collect();
    if !wins.is_empty() {
        let weakest = wins.iter().copied().fold(f64::INFINITY, f64::min);
        // A winner the screen scored at or below zero says the gate cannot see
        // this family at all; the threshold drops to its bounded minimum
        // rather than being computed from a number with no scale in it.
        let target = if weakest > 0.0 {
            weakest * WIN_MARGIN
        } else {
            floor / MAX_THRESHOLD_ADJUSTMENT
        };
        return Some((clamp_multiplier(target / floor), ThresholdBasis::WinMargin));
    }

    let mut losses: Vec<f64> = pairs
        .iter()
        .filter(|pair| pair.full_delta <= 0.0 && pair.screen_delta.is_finite())
        .map(|pair| pair.screen_delta)
        .collect();
    if losses.len() < MIN_CALIBRATION_PAIRS {
        return None;
    }
    losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quantile = interpolated_quantile(&losses, LOSS_QUANTILE);
    // The loss sample is **censored**: it holds only candidates the gate in
    // force already promoted, so its quantile always sits above that gate, and
    // an unbounded loss branch would ratchet a family up to the clamp on
    // nothing but its own past tightening. The weakest *improving* candidate is
    // the brake: whatever the losses say, the bar never rises past half the
    // smallest screen Δ that produced a real full-corpus improvement, so a
    // family that keeps improving — without yet clearing the accept bar — keeps
    // its signal. Only a family with no improvement at all is free to ratchet,
    // and the control sample is what can still rescue it.
    let target = match weakest_improving(pairs) {
        Some(weakest) if weakest > 0.0 => quantile.min(weakest * WIN_MARGIN),
        // An improvement the screen scored at or below zero says the gate
        // cannot see this family: it may not be tightened at all.
        Some(_) => return None,
        None => quantile,
    };
    if !target.is_finite() || target <= 0.0 {
        return None;
    }
    Some((
        clamp_multiplier(target / floor),
        ThresholdBasis::LossQuantile,
    ))
}

/// Smallest screen Δ among the candidates the full corpus scored **above zero**.
///
/// A weaker signal than a win — it need not clear the accept bar — which is
/// exactly why it is the brake on the loss branch rather than a threshold of
/// its own. `None` when the family has never improved on the full corpus.
fn weakest_improving(pairs: &[PairedScreenObservation]) -> Option<f64> {
    pairs
        .iter()
        .filter(|pair| pair.full_delta > 0.0 && pair.screen_delta.is_finite())
        .map(|pair| pair.screen_delta)
        .reduce(f64::min)
}

/// Bound a raw multiplier to what calibration is allowed to move.
fn clamp_multiplier(raw: f64) -> f64 {
    if !raw.is_finite() {
        return 1.0;
    }
    raw.clamp(1.0 / MAX_THRESHOLD_ADJUSTMENT, MAX_THRESHOLD_ADJUSTMENT)
}

/// Linear-interpolated quantile of an already-sorted, non-empty slice.
fn interpolated_quantile(sorted: &[f64], q: f64) -> f64 {
    let position = q * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(sorted.len() - 1);
    let fraction = position - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

/// Controls owed for a rejected set of `rejected` candidates at `rate`.
///
/// Rounded **up**, so the configured rate is a minimum rather than an average:
/// any rejection at all buys at least one measurement of what the gate threw
/// away. Zero when the rate is zero or nothing was rejected.
pub fn control_quota(rejected: usize, rate: f64) -> usize {
    if rejected == 0 || !rate.is_finite() || rate <= 0.0 {
        return 0;
    }
    ((rejected as f64) * rate.min(1.0)).ceil() as usize
}

/// Apply the calibrated gate to one screened batch (issue #220).
///
/// `shared_threshold` is what the batch's own promote gate resolved — the
/// absolute floor, or the noise-aware `max(k · σ̂, floor)` — so calibration
/// scales that decision rather than replacing it.
///
/// Controls are drawn uniformly at random from the candidates the thresholds
/// rejected, using the run's seeded rng, so a run replays identically.
pub fn calibrate_screen_batch(
    candidates: &[ScreenedCandidate],
    shared_threshold: f64,
    ledger: &ScreenThresholdLedger,
    policy: &ScreenThresholdPolicy,
    rng: &mut StdRng,
) -> CalibratedScreen {
    let mut thresholds: BTreeMap<String, f64> = BTreeMap::new();
    let mut basis: BTreeMap<String, String> = BTreeMap::new();
    let mut decisions: Vec<CandidateScreenDecision> = Vec::with_capacity(candidates.len());
    let mut rejected: Vec<usize> = Vec::new();

    for candidate in candidates {
        let label = candidate
            .strategy
            .map(|strategy| strategy.label().to_string())
            .unwrap_or_else(|| UNATTRIBUTED_STRATEGY.to_string());
        let threshold = match thresholds.get(&label) {
            Some(threshold) => *threshold,
            None => {
                let resolved = match policy.mode {
                    ScreenThresholdMode::Shared => StrategyThreshold::shared(0),
                    ScreenThresholdMode::PerStrategy => ledger.threshold_for(&label),
                };
                let threshold = shared_threshold * resolved.multiplier;
                thresholds.insert(label.clone(), threshold);
                basis.insert(label.clone(), resolved.basis.label().to_string());
                threshold
            }
        };
        let promoted = candidate.screen_delta > threshold;
        if !promoted {
            rejected.push(decisions.len());
        }
        decisions.push(CandidateScreenDecision {
            stem: candidate.stem.clone(),
            strategy: label,
            threshold,
            model_version: SCREEN_THRESHOLD_MODEL_VERSION.to_string(),
            promoted,
            control: false,
        });
    }

    let quota = control_quota(rejected.len(), policy.control_rate).min(rejected.len());
    for drawn in 0..quota {
        // Partial Fisher–Yates over the rejected indices: each draw is uniform
        // over what is left, so no candidate is drawn twice.
        let pick = drawn + rng.random_range(0..rejected.len() - drawn);
        rejected.swap(drawn, pick);
        let decision = &mut decisions[rejected[drawn]];
        decision.promoted = true;
        decision.control = true;
    }

    let mut promoted: Vec<(&str, f64)> = decisions
        .iter()
        .zip(candidates)
        .filter(|(decision, _)| decision.promoted)
        .map(|(decision, candidate)| (decision.stem.as_str(), candidate.screen_delta))
        .collect();
    promoted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    CalibratedScreen {
        stems: promoted
            .into_iter()
            .map(|(stem, _)| stem.to_string())
            .collect(),
        record: ScreenThresholdRecord {
            model_version: SCREEN_THRESHOLD_MODEL_VERSION.to_string(),
            mode: policy.mode.label().to_string(),
            shared_threshold,
            control_rate: policy.control_rate,
            thresholds,
            basis,
            controls: quota as u64,
            candidates: decisions,
        },
    }
}

/// One strategy's calibration, as the end of a journal leaves it (issue #220).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyThresholdRow {
    /// Strategy label.
    pub strategy: String,
    /// Paired observations in the calibration window.
    pub pairs: u64,
    /// Multiplier calibration puts on the shared threshold.
    pub multiplier: f64,
    /// The threshold that multiplier implies against the run's shared floor.
    pub threshold: f64,
    /// How the multiplier was arrived at (`shared` / `win-margin` /
    /// `loss-quantile`).
    pub basis: String,
}

/// What per-strategy calibration would have done to one journal (issue #220).
///
/// Replayed offline from the journal's own `screenScores`, so a gate change is
/// priced — and its effect on the accepts that were actually earned checked —
/// with no box time. Every count is a counterfactual over the recorded screen
/// deltas; the projected rate below says plainly what it assumes.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScreenThresholdReplay {
    /// Calibration model version the replay ran.
    pub model_version: String,
    /// Threshold mode from the run header, when its headers agree.
    pub mode_as_run: Option<String>,
    /// Control rate the replay charged the calibrated arm for.
    pub control_rate: f64,
    /// Candidates that got a subsample score.
    pub screened: u64,
    /// Candidates the run actually promoted to full-corpus scoring.
    pub promoted_as_run: u64,
    /// Candidates per-strategy calibration would have promoted, controls
    /// included.
    pub promoted_under_calibration: u64,
    /// Promotions the calibrated thresholds would have avoided buying.
    pub promotions_avoided: u64,
    /// Promotions the calibrated thresholds would have added.
    pub promotions_added: u64,
    /// Control promotions the calibrated arm would have paid for.
    pub control_promotions: u64,
    /// Accepted winners calibration would still have promoted.
    pub accepts_kept: u64,
    /// Accepted winners calibration would have discarded.
    ///
    /// Deliberately **pessimistic**: the replay models the control sample as a
    /// count, not as a draw of named stems, so a winner below the calibrated
    /// bar is counted dropped even where a control draw might have promoted it
    /// anyway. The error is in the safe direction — the replay never
    /// under-states the accepts a calibrated arm risks.
    pub accepts_dropped: u64,
    /// Full-corpus improvement those dropped accepts carried.
    pub improvement_dropped: f64,
    /// Mean measured promote-phase milliseconds per creature scored.
    pub promote_ms_per_creature: Option<f64>,
    /// Full-corpus scorer seconds the calibrated arm would have saved
    /// (negative when it buys more than it avoids).
    ///
    /// Priced **per creature**, at the journal's own measured promote cost. A
    /// scorer call also carries a fixed per-call cost (`docs/scorer-fixed-cost.md`),
    /// which this model does not move: avoiding candidates from a call that
    /// still happens saves only the marginal cost counted here, and avoiding a
    /// whole call saves more than this says.
    pub promote_seconds_saved: Option<f64>,
    /// The journal's measured score improvement per wall hour.
    pub score_improvement_per_wall_hour_as_run: Option<f64>,
    /// The same rate under calibration, assuming every kept accept still lands
    /// and the wall clock moves only by the promote time saved.
    ///
    /// `None` when the journal has no wall-clock span or no full-corpus
    /// improvement to rate. Read it beside [`Self::accepts_dropped`]: an arm
    /// that drops an accept has changed which creature the run was optimising,
    /// and no replay of the remaining experiments can price that.
    pub projected_score_improvement_per_wall_hour: Option<f64>,
    /// Where each strategy's threshold ended up.
    pub strategies: Vec<StrategyThresholdRow>,
}

/// Streaming accumulator behind [`ScreenThresholdReplay`], driven by
/// `report_from_journal` so a journal is read once.
#[derive(Debug)]
pub struct ScreenThresholdReplayAccumulator {
    ledger: ScreenThresholdLedger,
    policy: ScreenThresholdPolicy,
    mode: Option<String>,
    mode_mixed: bool,
    floor_seen: bool,
    screened: u64,
    promoted_as_run: u64,
    promoted_under_calibration: u64,
    promotions_avoided: u64,
    promotions_added: u64,
    control_promotions: u64,
    accepts_kept: u64,
    accepts_dropped: u64,
    improvement_dropped: f64,
    promote_ms: f64,
    promote_creatures: f64,
}

impl Default for ScreenThresholdReplayAccumulator {
    fn default() -> Self {
        Self {
            ledger: ScreenThresholdLedger::new(
                crate::config::DEFAULT_SCREEN_PROMOTE_THRESHOLD,
                crate::config::DEFAULT_MIN_IMPROVEMENT,
            ),
            policy: ScreenThresholdPolicy::default(),
            mode: None,
            mode_mixed: false,
            floor_seen: false,
            screened: 0,
            promoted_as_run: 0,
            promoted_under_calibration: 0,
            promotions_avoided: 0,
            promotions_added: 0,
            control_promotions: 0,
            accepts_kept: 0,
            accepts_dropped: 0,
            improvement_dropped: 0.0,
            promote_ms: 0.0,
            promote_creatures: 0.0,
        }
    }
}

impl ScreenThresholdReplayAccumulator {
    /// Record the knobs a run header states.
    pub fn push_header(&mut self, config: &crate::run::RunConfigRecord) {
        // The first header sets the floor the replay calibrates against; a
        // campaign whose arms disagree keeps the first, and says so through
        // `modeAsRun`.
        if !self.floor_seen {
            self.ledger.set_floor(config.screen_promote_threshold);
            self.ledger.set_accept_bar(config.min_improvement);
            self.floor_seen = true;
        }
        if let Some(rate) = config.screen_control_rate {
            self.policy.control_rate = rate;
        }
        let label = config
            .screen_threshold_mode
            .clone()
            .unwrap_or_else(|| ScreenThresholdMode::Shared.label().to_string());
        match &self.mode {
            Some(seen) if *seen != label => self.mode_mixed = true,
            _ => self.mode = Some(label),
        }
    }

    /// Replay one experiment's screen batch against the calibrated gate.
    ///
    /// Fails loudly on `screenScores` with no `baseline`: the deltas are
    /// measured against it, so a missing anchor cannot be silently skipped.
    pub fn push_experiment(&mut self, record: &ExperimentRecord) -> Result<(), String> {
        self.observe_promote_cost(record);
        let Some(screen) = &record.screen_scores else {
            return Ok(());
        };
        let baseline = *screen.get("baseline").ok_or_else(|| {
            format!(
                "experiment {}: screenScores has no baseline to measure deltas against",
                record.experiment_number
            )
        })?;
        // The threshold the run's own gate demanded of this batch — the shared
        // arm of the comparison. A journal too old to carry one falls back to
        // the header's floor, which is what such a run applied.
        let shared_threshold = record
            .screen_tiers
            .as_ref()
            .map(|tiers| tiers.threshold)
            .unwrap_or_else(|| self.ledger.floor());

        let mut rejected = 0usize;
        let mut promoted_under_calibration: std::collections::BTreeSet<&str> =
            std::collections::BTreeSet::new();
        for (stem, score) in screen {
            if stem == "baseline" {
                continue;
            }
            self.screened += 1;
            let promoted_as_run = record.scores.contains_key(stem);
            if promoted_as_run {
                self.promoted_as_run += 1;
            }
            let strategy = strategy_label(record, stem);
            let threshold = shared_threshold * self.ledger.threshold_for(&strategy).multiplier;
            if score - baseline > threshold {
                promoted_under_calibration.insert(stem.as_str());
            } else {
                rejected += 1;
            }
        }
        let controls = control_quota(rejected, self.policy.control_rate);
        self.control_promotions += controls as u64;
        let promoted = promoted_under_calibration.len() as u64 + controls as u64;
        self.promoted_under_calibration += promoted;

        for (stem, _) in screen.iter().filter(|(stem, _)| *stem != "baseline") {
            match (
                record.scores.contains_key(stem),
                promoted_under_calibration.contains(stem.as_str()),
            ) {
                (true, false) => self.promotions_avoided += 1,
                (false, true) => self.promotions_added += 1,
                _ => {}
            }
        }
        // A control is bought on top of whatever the thresholds admitted, so it
        // is an added promotion in the economics as well as in the count.
        self.promotions_added += controls as u64;

        if record.accepted
            && let Some(winner) = &record.winner
        {
            // A winner the screen never scored — a combo assembled afterwards —
            // could not have been dropped by any screen threshold.
            let kept = !screen.contains_key(winner.as_str())
                || promoted_under_calibration.contains(winner.as_str());
            if kept {
                self.accepts_kept += 1;
            } else {
                self.accepts_dropped += 1;
                self.improvement_dropped += record.improvement.unwrap_or(0.0);
            }
        }

        self.ledger.observe(record);
        Ok(())
    }

    /// Accumulate this experiment's measured promote-phase cost per creature.
    fn observe_promote_cost(&mut self, record: &ExperimentRecord) {
        let Some(calls) = &record.scorer_calls else {
            return;
        };
        for call in calls {
            if call.phase == ScorerCallPhase::Promote && !call.failed && call.creatures > 0 {
                self.promote_ms += call.elapsed_ms as f64;
                self.promote_creatures += call.creatures as f64;
            }
        }
    }

    /// Finish the replay against the journal's own measured totals.
    pub fn finish(
        self,
        wall_duration_ms: Option<u128>,
        total_score_improvement: Option<f64>,
    ) -> ScreenThresholdReplay {
        let promote_ms_per_creature =
            (self.promote_creatures > 0.0).then(|| self.promote_ms / self.promote_creatures);
        let net_avoided = self.promotions_avoided as f64 - self.promotions_added as f64;
        let promote_seconds_saved =
            promote_ms_per_creature.map(|per_creature| net_avoided * per_creature / 1_000.0);
        let as_run = match (total_score_improvement, wall_duration_ms) {
            (Some(delta), Some(wall)) if wall > 0 => Some(delta / (wall as f64 / 3_600_000.0)),
            _ => None,
        };
        let projected = match (
            total_score_improvement,
            wall_duration_ms,
            promote_seconds_saved,
        ) {
            (Some(delta), Some(wall), Some(saved)) => {
                let hours = (wall as f64 / 3_600_000.0) - (saved / 3_600.0);
                (hours > 0.0).then(|| (delta - self.improvement_dropped) / hours)
            }
            _ => None,
        };
        let strategies = self
            .ledger
            .strategies()
            .map(|strategy| {
                let resolved = self.ledger.threshold_for(strategy);
                StrategyThresholdRow {
                    strategy: strategy.to_string(),
                    pairs: self.ledger.observations(strategy).len() as u64,
                    multiplier: resolved.multiplier,
                    threshold: self.ledger.floor() * resolved.multiplier,
                    basis: resolved.basis.label().to_string(),
                }
            })
            .collect();

        ScreenThresholdReplay {
            model_version: SCREEN_THRESHOLD_MODEL_VERSION.to_string(),
            mode_as_run: (!self.mode_mixed).then_some(self.mode).flatten(),
            control_rate: self.policy.control_rate,
            screened: self.screened,
            promoted_as_run: self.promoted_as_run,
            promoted_under_calibration: self.promoted_under_calibration,
            promotions_avoided: self.promotions_avoided,
            promotions_added: self.promotions_added,
            control_promotions: self.control_promotions,
            accepts_kept: self.accepts_kept,
            accepts_dropped: self.accepts_dropped,
            improvement_dropped: self.improvement_dropped,
            promote_ms_per_creature,
            promote_seconds_saved,
            score_improvement_per_wall_hour_as_run: as_run,
            projected_score_improvement_per_wall_hour: projected,
            strategies,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::CandidateProvenance;
    use rand::SeedableRng;

    fn pair(screen_delta: f64, full_delta: f64) -> PairedScreenObservation {
        PairedScreenObservation {
            screen_delta,
            full_delta,
            control: false,
        }
    }

    fn prov(strategy: CandidateStrategy) -> CandidateProvenance {
        CandidateProvenance {
            strategy,
            focus_neuron: "h1".into(),
            mutation: "test".into(),
            old_value: None,
            new_value: None,
            mirror: None,
            follow_up: None,
        }
    }

    fn screened(batch: &[(&str, Option<CandidateStrategy>, f64)]) -> Vec<ScreenedCandidate> {
        batch
            .iter()
            .map(|(stem, strategy, delta)| ScreenedCandidate {
                stem: (*stem).to_string(),
                strategy: *strategy,
                screen_delta: *delta,
            })
            .collect()
    }

    fn scores(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
        entries
            .iter()
            .map(|(stem, score)| ((*stem).to_string(), *score))
            .collect()
    }

    /// A ledger holding `pairs` observations for `strategy`.
    fn ledger_with(strategy: &str, pairs: &[PairedScreenObservation]) -> ScreenThresholdLedger {
        let mut ledger = ScreenThresholdLedger::new(1e-6, 1e-6);
        for observation in pairs {
            ledger.push_for_test(strategy, *observation);
        }
        ledger
    }

    /// Insufficient evidence keeps the shared threshold, exactly.
    #[test]
    fn a_thin_window_keeps_the_shared_threshold() {
        let empty = ScreenThresholdLedger::new(1e-6, 1e-6);
        let resolved = empty.threshold_for("random");
        assert_eq!(resolved.multiplier, 1.0);
        assert_eq!(resolved.basis, ThresholdBasis::Shared);
        assert_eq!(resolved.pairs, 0);

        // One observation short of the minimum: the estimator has a statistic,
        // and the ledger withholds it anyway.
        let thin: Vec<_> = (0..MIN_CALIBRATION_PAIRS - 1)
            .map(|_| pair(4e-6, 2e-6))
            .collect();
        assert!(calibrated_multiplier(&thin, 1e-6, 1e-6).is_some());
        let ledger = ledger_with("random", &thin);
        assert_eq!(ledger.threshold_for("random").basis, ThresholdBasis::Shared);
        assert_eq!(ledger.threshold_for("random").multiplier, 1.0);
    }

    /// The threshold sits a factor of two below the weakest measured winner.
    #[test]
    fn the_win_margin_keeps_every_winner_the_family_has_shown() {
        let pairs = vec![
            pair(8e-6, 3e-6),  // a win
            pair(4e-6, 2e-6),  // the weakest win
            pair(2e-6, -1e-6), // a wasted promote call
            pair(1e-6, 0.0),
        ];
        let (multiplier, basis) = calibrated_multiplier(&pairs, 1e-6, 1e-6).expect("two wins");
        assert_eq!(basis, ThresholdBasis::WinMargin);
        // 4e-6 × 0.5 = 2e-6 against a 1e-6 floor.
        assert!((multiplier - 2.0).abs() < 1e-12, "multiplier {multiplier}");
    }

    /// A family whose winner screened flat gets the gate opened, not closed.
    #[test]
    fn a_winner_the_screen_could_not_see_lowers_the_threshold() {
        let pairs = vec![pair(-1e-7, 5e-6), pair(3e-6, -2e-6)];
        let (multiplier, basis) = calibrated_multiplier(&pairs, 1e-6, 1e-6).expect("one win");
        assert_eq!(basis, ThresholdBasis::WinMargin);
        assert!(
            (multiplier - 1.0 / MAX_THRESHOLD_ADJUSTMENT).abs() < 1e-12,
            "multiplier {multiplier}"
        );
    }

    /// No winner, only wasted calls: the bar rises to the median loss.
    #[test]
    fn a_family_that_never_converts_pays_the_median_loss() {
        let pairs: Vec<_> = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6]
            .iter()
            .map(|screen| pair(*screen, -1e-7))
            .collect();
        let (multiplier, basis) = calibrated_multiplier(&pairs, 1e-6, 1e-6).expect("eight losses");
        assert_eq!(basis, ThresholdBasis::LossQuantile);
        // Median of the eight is 4.5e-6 against a 1e-6 floor.
        assert!((multiplier - 4.5).abs() < 1e-12, "multiplier {multiplier}");
    }

    /// The loss branch is braked by the weakest improving candidate, so a
    /// censored loss sample cannot ratchet a family that is still improving.
    #[test]
    fn the_loss_quantile_never_rises_past_a_measured_improvement() {
        // Eight losing promotions at 4e-6, plus one candidate the full corpus
        // scored barely above zero on a screen Δ of 1e-6.
        let mut pairs: Vec<_> = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6]
            .iter()
            .map(|screen| pair(*screen, -1e-7))
            .collect();
        pairs.push(pair(1e-6, 5e-7));
        let (multiplier, basis) = calibrated_multiplier(&pairs, 1e-6, 1e-6).expect("eight losses");
        assert_eq!(basis, ThresholdBasis::LossQuantile);
        // The median loss alone would say 4.5e-6; the improvement caps it at
        // 0.5 × 1e-6, so the bar is not raised past what the family has shown.
        assert!((multiplier - 0.5).abs() < 1e-12, "multiplier {multiplier}");
    }

    /// An improvement the screen scored at or below zero blocks the loss
    /// branch outright: that family cannot be tightened on censored evidence.
    #[test]
    fn an_invisible_improvement_blocks_the_loss_branch() {
        let mut pairs: Vec<_> = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6]
            .iter()
            .map(|screen| pair(*screen, -1e-7))
            .collect();
        pairs.push(pair(-2e-7, 5e-7));
        assert_eq!(calibrated_multiplier(&pairs, 1e-6, 1e-6), None);
    }

    /// Too few losses to quantile, and no win: nothing is claimed.
    #[test]
    fn a_family_with_no_usable_statistic_falls_back() {
        let pairs = vec![pair(2e-6, -1e-6), pair(3e-6, -1e-6)];
        assert_eq!(calibrated_multiplier(&pairs, 1e-6, 1e-6), None);
        // A floor that is not a positive scale cannot be multiplied.
        let wins: Vec<_> = (0..MIN_CALIBRATION_PAIRS)
            .map(|_| pair(4e-6, 2e-6))
            .collect();
        assert_eq!(calibrated_multiplier(&wins, 0.0, 1e-6), None);
    }

    /// Calibration is bounded in both directions, whatever the evidence says.
    #[test]
    fn the_multiplier_is_clamped_both_ways() {
        let loud: Vec<_> = (0..MIN_CALIBRATION_PAIRS)
            .map(|_| pair(1.0, 2e-6))
            .collect();
        let (multiplier, _) = calibrated_multiplier(&loud, 1e-6, 1e-6).expect("a win");
        assert_eq!(multiplier, MAX_THRESHOLD_ADJUSTMENT);

        let quiet: Vec<_> = (0..MIN_CALIBRATION_PAIRS)
            .map(|_| pair(1e-9, 2e-6))
            .collect();
        let (multiplier, _) = calibrated_multiplier(&quiet, 1e-6, 1e-6).expect("a win");
        assert_eq!(multiplier, 1.0 / MAX_THRESHOLD_ADJUSTMENT);
    }

    /// The shared mode promotes exactly what the batch gate promoted.
    #[test]
    fn shared_mode_leaves_the_batch_gate_alone() {
        // Evidence that *would* move a per-strategy threshold is present and
        // deliberately ignored: shared mode is the unchanged pre-#220 run.
        let ledger = ledger_with(
            "random",
            &(0..MIN_CALIBRATION_PAIRS)
                .map(|_| pair(4e-6, -1e-6))
                .collect::<Vec<_>>(),
        );
        let policy = ScreenThresholdPolicy {
            mode: ScreenThresholdMode::Shared,
            control_rate: 0.0,
        };
        let batch = screened(&[
            ("candidate-000", Some(CandidateStrategy::Random), 4e-6),
            ("candidate-001", Some(CandidateStrategy::Random), 5e-7),
        ]);
        let mut rng = StdRng::seed_from_u64(1);
        let calibrated = calibrate_screen_batch(&batch, 1e-6, &ledger, &policy, &mut rng);
        assert_eq!(calibrated.stems, vec!["candidate-000".to_string()]);
        assert_eq!(calibrated.record.controls, 0);
        assert_eq!(calibrated.record.mode, "shared");
        assert!((calibrated.record.candidates[0].threshold - 1e-6).abs() < 1e-15);
    }

    /// Per-strategy thresholds bind per family, and every candidate is
    /// journalled with the threshold and model version it faced.
    #[test]
    fn per_strategy_thresholds_bind_per_family_and_are_journalled() {
        let mut ledger = ScreenThresholdLedger::new(1e-6, 1e-6);
        // `random` has only wasted calls at 4e-6, so its bar rises to 4e-6.
        for _ in 0..MIN_CALIBRATION_PAIRS {
            ledger.push_for_test("random", pair(4e-6, -1e-6));
        }
        // `backprop` won on a screen Δ the gate could barely see, so its bar
        // drops to the bounded minimum.
        for _ in 0..MIN_CALIBRATION_PAIRS {
            ledger.push_for_test("backprop", pair(1e-7, 3e-6));
        }
        let policy = ScreenThresholdPolicy {
            mode: ScreenThresholdMode::PerStrategy,
            control_rate: 0.0,
        };
        let batch = screened(&[
            // Clears the shared 1e-6 bar but not `random`'s calibrated 4e-6.
            ("candidate-000", Some(CandidateStrategy::Random), 2e-6),
            // Below the shared bar, but above `backprop`'s calibrated 1.25e-7.
            ("candidate-001", Some(CandidateStrategy::Backprop), 5e-7),
        ]);
        let mut rng = StdRng::seed_from_u64(1);
        let calibrated = calibrate_screen_batch(&batch, 1e-6, &ledger, &policy, &mut rng);
        assert_eq!(calibrated.stems, vec!["candidate-001".to_string()]);

        let random = &calibrated.record.candidates[0];
        assert_eq!(random.strategy, "random");
        assert!((random.threshold - 4e-6).abs() < 1e-15);
        assert!(!random.promoted);
        assert_eq!(random.model_version, SCREEN_THRESHOLD_MODEL_VERSION);
        let backprop = &calibrated.record.candidates[1];
        assert!((backprop.threshold - 1e-6 / MAX_THRESHOLD_ADJUSTMENT).abs() < 1e-15);
        assert!(backprop.promoted);
        assert_eq!(
            calibrated.record.basis.get("random").map(String::as_str),
            Some("loss-quantile")
        );
        assert_eq!(
            calibrated.record.basis.get("backprop").map(String::as_str),
            Some("win-margin")
        );
    }

    /// A stem with no resolvable provenance runs on the shared threshold.
    #[test]
    fn an_unattributed_candidate_keeps_the_shared_threshold() {
        let ledger = ScreenThresholdLedger::new(1e-6, 1e-6);
        let policy = ScreenThresholdPolicy {
            mode: ScreenThresholdMode::PerStrategy,
            control_rate: 0.0,
        };
        let batch = screened(&[("candidate-042", None, 2e-6)]);
        let mut rng = StdRng::seed_from_u64(1);
        let calibrated = calibrate_screen_batch(&batch, 1e-6, &ledger, &policy, &mut rng);
        let decision = &calibrated.record.candidates[0];
        assert_eq!(decision.strategy, UNATTRIBUTED_STRATEGY);
        assert!((decision.threshold - 1e-6).abs() < 1e-15);
        assert!(decision.promoted);
    }

    /// The control rate is a floor: any rejection buys at least one control.
    #[test]
    fn controls_are_rounded_up_so_the_rate_is_a_minimum() {
        assert_eq!(control_quota(0, DEFAULT_SCREEN_CONTROL_RATE), 0);
        assert_eq!(
            control_quota(1, DEFAULT_SCREEN_CONTROL_RATE),
            1,
            "one rejection is still measured"
        );
        assert_eq!(control_quota(100, DEFAULT_SCREEN_CONTROL_RATE), 2);
        assert_eq!(control_quota(101, DEFAULT_SCREEN_CONTROL_RATE), 3);
        assert_eq!(control_quota(100, 0.0), 0, "controls off");
    }

    /// A control promotes a candidate its own threshold rejected, and the
    /// journal says which candidate that was.
    #[test]
    fn a_control_promotes_a_below_threshold_candidate_and_says_so() {
        let ledger = ScreenThresholdLedger::new(1e-6, 1e-6);
        let policy = ScreenThresholdPolicy {
            mode: ScreenThresholdMode::PerStrategy,
            control_rate: 0.5,
        };
        let batch = screened(&[
            ("candidate-000", Some(CandidateStrategy::Random), -1e-6),
            ("candidate-001", Some(CandidateStrategy::Random), -2e-6),
        ]);
        let mut rng = StdRng::seed_from_u64(7);
        let calibrated = calibrate_screen_batch(&batch, 1e-6, &ledger, &policy, &mut rng);
        assert_eq!(calibrated.record.controls, 1);
        assert_eq!(calibrated.stems.len(), 1, "the control is promoted");
        let control = calibrated
            .record
            .candidates
            .iter()
            .find(|candidate| candidate.control)
            .expect("one control");
        assert!(control.promoted, "a control is scored on the full corpus");
        assert_eq!(calibrated.stems[0], control.stem);
    }

    /// Every rejected candidate is a possible control, so a run that promotes
    /// controls forever cannot systematically miss one family.
    #[test]
    fn controls_are_drawn_from_the_whole_rejected_set() {
        let ledger = ScreenThresholdLedger::new(1e-6, 1e-6);
        let policy = ScreenThresholdPolicy {
            mode: ScreenThresholdMode::PerStrategy,
            control_rate: 0.01,
        };
        let batch: Vec<ScreenedCandidate> = (0..8)
            .map(|i| ScreenedCandidate {
                stem: format!("candidate-{i:03}"),
                strategy: Some(CandidateStrategy::Random),
                screen_delta: -1e-6,
            })
            .collect();
        let mut drawn = std::collections::BTreeSet::new();
        for seed in 0..64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let calibrated = calibrate_screen_batch(&batch, 1e-6, &ledger, &policy, &mut rng);
            assert_eq!(calibrated.stems.len(), 1);
            drawn.insert(calibrated.stems[0].clone());
        }
        assert_eq!(drawn.len(), 8, "every rejected candidate can be a control");
    }

    /// The ledger learns from journalled pairs, per strategy.
    #[test]
    fn the_ledger_pairs_journalled_experiments_per_strategy() {
        let mut record = experiment(1);
        record.candidates = vec![
            prov(CandidateStrategy::Random),
            prov(CandidateStrategy::Backprop),
        ];
        record.screen_scores = Some(scores(&[
            ("baseline", 0.4),
            ("candidate-000", 0.4 + 3e-6),
            ("candidate-001", 0.4 + 5e-6),
        ]));
        record.scores = scores(&[
            ("baseline", 0.5),
            ("candidate-000", 0.5 - 1e-6),
            ("candidate-001", 0.5 + 2e-6),
        ]);

        let mut ledger = ScreenThresholdLedger::new(1e-6, 1e-6);
        ledger.observe(&record);
        let random = ledger.observations("random");
        assert_eq!(random.len(), 1);
        assert!((random[0].screen_delta - 3e-6).abs() < 1e-15);
        assert!((random[0].full_delta + 1e-6).abs() < 1e-15);
        let backprop = ledger.observations("backprop");
        assert_eq!(backprop.len(), 1);
        assert!((backprop[0].full_delta - 2e-6).abs() < 1e-15);
    }

    /// The window keeps the most recent observations, not the first ones.
    #[test]
    fn the_window_bounds_what_a_strategy_remembers() {
        let mut ledger = ScreenThresholdLedger::new(1e-6, 1e-6);
        for i in 0..CALIBRATION_WINDOW + 5 {
            ledger.push_for_test("random", pair(i as f64, 0.0));
        }
        let held = ledger.observations("random");
        assert_eq!(held.len(), CALIBRATION_WINDOW);
        assert_eq!(held[0].screen_delta, 5.0, "the oldest five aged out");
    }

    /// A control's full-corpus score is the false-negative evidence: it enters
    /// the window flagged, and a win among the controls moves the threshold.
    #[test]
    fn a_control_promotion_feeds_the_window_as_false_negative_evidence() {
        let mut record = experiment(1);
        record.candidates = vec![prov(CandidateStrategy::Backprop)];
        record.screen_scores = Some(scores(&[("baseline", 0.4), ("candidate-000", 0.4 + 1e-8)]));
        record.scores = scores(&[("baseline", 0.5), ("candidate-000", 0.5 + 4e-6)]);
        record.screen_thresholds = Some(ScreenThresholdRecord {
            model_version: SCREEN_THRESHOLD_MODEL_VERSION.to_string(),
            mode: ScreenThresholdMode::PerStrategy.label().to_string(),
            shared_threshold: 1e-6,
            control_rate: DEFAULT_SCREEN_CONTROL_RATE,
            thresholds: [("backprop".to_string(), 1e-6)].into_iter().collect(),
            basis: [("backprop".to_string(), "shared".to_string())]
                .into_iter()
                .collect(),
            controls: 1,
            candidates: vec![CandidateScreenDecision {
                stem: "candidate-000".to_string(),
                strategy: "backprop".to_string(),
                threshold: 1e-6,
                model_version: SCREEN_THRESHOLD_MODEL_VERSION.to_string(),
                promoted: true,
                control: true,
            }],
        });

        let mut ledger = ScreenThresholdLedger::new(1e-6, 1e-6);
        for _ in 0..MIN_CALIBRATION_PAIRS {
            ledger.observe(&record);
        }
        let held = ledger.observations("backprop");
        assert_eq!(held.len(), MIN_CALIBRATION_PAIRS);
        assert!(held.iter().all(|observation| observation.control));
        // A winner the screen scored at 1e-8 drops the bar to its bounded
        // minimum — evidence no uncalibrated run could ever have collected.
        let resolved = ledger.threshold_for("backprop");
        assert_eq!(resolved.basis, ThresholdBasis::WinMargin);
        assert_eq!(resolved.multiplier, 1.0 / MAX_THRESHOLD_ADJUSTMENT);
    }

    /// The replay prices the calibrated gate against the shared one the run
    /// actually applied.
    #[test]
    fn the_replay_counts_the_promotions_calibration_would_have_avoided() {
        let mut accumulator = ScreenThresholdReplayAccumulator::default();
        // Eight experiments of `random` promoting at 4e-6 and never improving
        // fill the window; the ninth is the one calibration acts on.
        for number in 1..=MIN_CALIBRATION_PAIRS as u64 {
            let mut record = experiment(number);
            record.candidates = vec![prov(CandidateStrategy::Random)];
            record.screen_scores =
                Some(scores(&[("baseline", 0.4), ("candidate-000", 0.4 + 4e-6)]));
            record.scores = scores(&[("baseline", 0.5), ("candidate-000", 0.5 - 1e-6)]);
            accumulator.push_experiment(&record).expect("well formed");
        }
        let mut last = experiment(99);
        last.candidates = vec![prov(CandidateStrategy::Random)];
        last.screen_scores = Some(scores(&[("baseline", 0.4), ("candidate-000", 0.4 + 2e-6)]));
        last.scores = scores(&[("baseline", 0.5), ("candidate-000", 0.5 - 1e-6)]);
        accumulator.push_experiment(&last).expect("well formed");

        let replay = accumulator.finish(Some(3_600_000), Some(2e-6));
        assert_eq!(replay.screened, MIN_CALIBRATION_PAIRS as u64 + 1);
        assert_eq!(replay.promoted_as_run, MIN_CALIBRATION_PAIRS as u64 + 1);
        // The last experiment's 2e-6 no longer clears `random`'s 4e-6 bar.
        assert_eq!(replay.promotions_avoided, 1);
        assert_eq!(replay.accepts_dropped, 0);
        assert_eq!(replay.model_version, SCREEN_THRESHOLD_MODEL_VERSION);
        let random = replay
            .strategies
            .iter()
            .find(|row| row.strategy == "random")
            .expect("the replayed ledger holds random");
        assert_eq!(random.basis, "loss-quantile");
        assert!((random.threshold - 4e-6).abs() < 1e-15);
    }

    /// An accept the calibrated gate would have dropped is reported, and its
    /// improvement is removed from the projected rate rather than kept.
    #[test]
    fn the_replay_reports_an_accept_calibration_would_have_dropped() {
        let mut accumulator = ScreenThresholdReplayAccumulator::default();
        for number in 1..=MIN_CALIBRATION_PAIRS as u64 {
            let mut record = experiment(number);
            record.candidates = vec![prov(CandidateStrategy::Random)];
            record.screen_scores =
                Some(scores(&[("baseline", 0.4), ("candidate-000", 0.4 + 4e-6)]));
            record.scores = scores(&[("baseline", 0.5), ("candidate-000", 0.5 - 1e-6)]);
            accumulator.push_experiment(&record).expect("well formed");
        }
        let mut win = experiment(99);
        win.candidates = vec![prov(CandidateStrategy::Random)];
        win.screen_scores = Some(scores(&[("baseline", 0.4), ("candidate-000", 0.4 + 2e-6)]));
        win.scores = scores(&[("baseline", 0.5), ("candidate-000", 0.5 + 3e-6)]);
        win.winner = Some("candidate-000".to_string());
        win.improvement = Some(3e-6);
        win.accepted = true;
        // A measured promote call is what lets the replay price the avoided
        // full-corpus scores in seconds rather than in counts.
        win.scorer_calls = Some(vec![crate::scorer_cost::ScorerCallRecord {
            phase: ScorerCallPhase::Promote,
            creatures: 2,
            sample_rate: None,
            elapsed_ms: 20_000,
            failed: false,
        }]);
        accumulator.push_experiment(&win).expect("well formed");

        let replay = accumulator.finish(Some(3_600_000), Some(3e-6));
        assert_eq!(replay.accepts_dropped, 1, "4e-6 bar, 2e-6 screen Δ");
        assert!((replay.improvement_dropped - 3e-6).abs() < 1e-15);
        assert_eq!(
            replay.projected_score_improvement_per_wall_hour,
            Some(0.0),
            "the only accept was the one calibration dropped"
        );
    }

    /// A malformed screen map is a fault, not an empty batch.
    #[test]
    fn a_screen_map_without_a_baseline_fails_loudly() {
        let mut record = experiment(7);
        record.screen_scores = Some(scores(&[("candidate-000", 0.41)]));
        let mut accumulator = ScreenThresholdReplayAccumulator::default();
        let error = accumulator.push_experiment(&record).expect_err("no anchor");
        assert!(error.contains("experiment 7"), "{error}");
        assert!(error.contains("baseline"), "{error}");
    }

    /// A journal fixture with only the fields these tests set.
    fn experiment(number: u64) -> ExperimentRecord {
        let json = format!(
            r#"{{"experimentNumber":{number},"timestampUnix":{number},"incumbentId":"x",
                 "baselineScore":0.4,"focusNeuron":"h1","candidates":[],"scores":{{}},
                 "winner":null,"improvement":null,"accepted":false,"analysisMs":1,"scorerMs":2}}"#
        );
        serde_json::from_str(&json).expect("fixture record")
    }
}
