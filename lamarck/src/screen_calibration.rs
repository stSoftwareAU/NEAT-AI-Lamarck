//! How well the screen phase predicts the full-corpus score (issue #110).
//!
//! Every journalled experiment carries both numbers already — `screenScores`
//! (the subsample) and `scores` (the full corpus) — so a journal is a paired
//! sample of (screen Δ, full Δ) that needs no box time to analyse. This module
//! pairs them per candidate stem and reports the rank correlation, the
//! precision of the promote gate, the full-corpus spread of what it promoted,
//! and an empirical estimate of the subsample's own noise.
//!
//! Pairing rules, each of which a fixture below pins:
//!
//! - only the **intersection** of the screen and promote stem sets is paired;
//!   the unpaired remainder is counted on both sides rather than dropped,
//! - `baseline` is excluded from both sides — it is the anchor the deltas are
//!   measured against, and pairing it would pin the correlation near 1,
//! - a journal with no screen phase reports "not applicable" rather than a
//!   fabricated correlation.

use crate::config::DEFAULT_MIN_IMPROVEMENT;
use crate::run::{ExperimentRecord, RunConfigRecord};
use serde::Serialize;
use std::collections::BTreeMap;

/// One paired observation: the same candidate scored twice.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScreenPair {
    /// Experiment the candidate was proposed in.
    pub experiment_number: u64,
    /// Candidate stem (`candidate-NNN`).
    pub stem: String,
    /// Strategy that proposed it, when the stem indexes a provenance.
    pub strategy: Option<String>,
    /// Subsample score minus the subsample baseline.
    pub screen_delta: f64,
    /// Full-corpus score minus the full-corpus baseline.
    pub full_delta: f64,
}

/// Spread of a sample, reported next to its own size.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeltaDistribution {
    /// Observations behind every figure in this bucket.
    pub count: u64,
    /// Smallest observation.
    pub min: f64,
    /// 25th percentile (linear interpolation between order statistics).
    pub p25: f64,
    /// Median.
    pub median: f64,
    /// 75th percentile.
    pub p75: f64,
    /// Largest observation.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
}

/// Empirical noise floor of the screen, from candidates the full corpus says
/// are worth ~nothing.
///
/// A candidate whose full-corpus Δ sits inside `band` moved the true score by
/// less than the accept bar, so whatever its screen Δ says is subsample noise.
/// The estimate is **one-sided**: only promoted candidates have a full-corpus
/// score at all, and promotion already required `screenΔ > threshold`, so the
/// spread below the threshold is unobservable here and the true noise floor is
/// at least this wide.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScreenNoise {
    /// Half-width of the "full corpus says ~zero" band.
    pub band: f64,
    /// Paired candidates inside the band.
    pub samples: u64,
    /// Mean screen Δ inside the band.
    pub mean: f64,
    /// Sample standard deviation (n−1) of screen Δ inside the band.
    pub std_dev: f64,
    /// Root-mean-square screen Δ inside the band.
    pub rms: f64,
    /// Largest absolute screen Δ inside the band.
    pub max_abs: f64,
}

/// Subsample-versus-full-corpus gap on the **same** creature.
///
/// Every promoting experiment scores `baseline.json` twice: once on the 5%
/// subsample and once on the full corpus. The difference is pure sampling
/// error, measured with no selection effect at all, so it is the cleanest
/// statement of what a subsample score is worth.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BaselineSampleGap {
    /// Experiments contributing a paired baseline.
    pub samples: u64,
    /// Mean `screen baseline − full baseline`.
    pub mean: f64,
    /// Sample standard deviation (n−1) of that gap.
    pub std_dev: f64,
    /// Largest absolute gap.
    pub max_abs: f64,
}

/// The screen Δ of a candidate that was ultimately accepted.
///
/// This is the false-negative evidence: how much headroom a lower promote
/// threshold would have needed to keep the accepts that were actually earned.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AcceptedScreenPoint {
    /// Experiment that accepted.
    pub experiment_number: u64,
    /// Winning stem.
    pub stem: String,
    /// Its screen Δ; `None` for a merged combo, which is assembled after the
    /// screen and so was never scored on the subsample.
    pub screen_delta: Option<f64>,
    /// Its full-corpus Δ when the stem was scored under that name.
    pub full_delta: Option<f64>,
}

/// Screen-versus-full-corpus calibration for one journal (issue #110).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScreenCalibration {
    /// Whether any experiment in the journal ran a screen phase.
    ///
    /// `false` is the "not applicable" answer: every other field is empty or
    /// `None` rather than fabricated from the promote phase alone.
    pub screen_enabled: bool,
    /// Experiments seen.
    pub experiments: u64,
    /// Experiments that recorded subsample scores.
    pub experiments_screened: u64,
    /// Experiments with no screen phase (screening disabled, or a pre-screen
    /// journal).
    pub experiments_without_screen: u64,
    /// `--screen-sample-rate` from the run header, when the journal has one and
    /// its headers agree.
    pub screen_sample_rate: Option<f64>,
    /// `--screen-promote-threshold` from the run header, same caveat.
    pub promote_threshold: Option<f64>,
    /// `--min-improvement` (the accept bar) from the run header, same caveat.
    pub accept_bar: Option<f64>,
    /// Candidates scored on both the subsample and the full corpus.
    pub paired_candidates: u64,
    /// Screened candidates the gate did not promote, so they have no full
    /// score to pair with.
    pub screen_only_candidates: u64,
    /// Full-corpus stems with no screen score of the same name.
    ///
    /// Expected to be `0` for a candidate journal; a merged combo would land
    /// here, as would a genuine pairing bug.
    pub full_only_candidates: u64,
    /// Spearman rank correlation between screen Δ and full Δ over the paired
    /// candidates; `None` below three pairs or when one side is constant.
    pub spearman: Option<f64>,
    /// Distinct (screen Δ, full Δ) points among the pairs.
    ///
    /// The generator re-proposes the same mutation whenever the incumbent and
    /// the focus repeat, and a campaign of arms sharing a seed replays much of
    /// one experiment stream. Quoting `pairedCandidates` as a sample size when
    /// this number is far smaller would overstate the evidence.
    pub distinct_pairs: u64,
    /// [`Self::spearman`] recomputed over the distinct points only.
    ///
    /// The honest sensitivity check on the headline coefficient: repeats carry
    /// no extra information about the screen, but they do carry weight in a
    /// rank correlation.
    pub spearman_distinct: Option<f64>,
    /// Paired candidates whose full-corpus Δ beat zero.
    pub promoted_improved: u64,
    /// Paired candidates whose full-corpus Δ was negative.
    pub promoted_worse: u64,
    /// Paired candidates clearing the accept bar on the full corpus.
    pub promoted_clearing_accept_bar: u64,
    /// Paired candidates the full corpus put *below* `−accept bar` — promotions
    /// that were not merely flat but materially wrong.
    pub promoted_materially_worse: u64,
    /// `promotedImproved / pairedCandidates`; `None` with no pairs.
    pub promotion_precision: Option<f64>,
    /// Full-corpus Δ spread across the promoted candidates.
    pub full_delta: Option<DeltaDistribution>,
    /// Screen Δ spread among candidates the full corpus scored at ~zero.
    pub screen_noise: Option<ScreenNoise>,
    /// Subsample-versus-full-corpus baseline gap.
    pub baseline_sample_gap: Option<BaselineSampleGap>,
    /// Screen Δ of every candidate that was ultimately accepted.
    pub accepted_candidates: Vec<AcceptedScreenPoint>,
    /// The paired points themselves, in journal order.
    ///
    /// Bounded by the number of *promotions*, not the number of candidates —
    /// the screen is what keeps this list short.
    pub pairs: Vec<ScreenPair>,
}

/// Streaming accumulator behind [`ScreenCalibration`].
///
/// Driven by `report_from_journal` so a journal is read once.
#[derive(Debug, Default)]
pub struct ScreenCalibrationAccumulator {
    experiments: u64,
    experiments_screened: u64,
    experiments_without_screen: u64,
    screen_sample_rate: HeaderValue,
    promote_threshold: HeaderValue,
    accept_bar: HeaderValue,
    screen_only: u64,
    full_only: u64,
    baseline_gaps: Vec<f64>,
    accepted: Vec<AcceptedScreenPoint>,
    pairs: Vec<ScreenPair>,
}

/// A header knob that is only reportable while every header agrees on it.
///
/// Concatenated journals from an A/B campaign do not share one value, and
/// quoting the first arm's threshold for all of them would be a fabrication.
#[derive(Debug, Default)]
enum HeaderValue {
    #[default]
    Unseen,
    Agreed(f64),
    Mixed,
}

impl HeaderValue {
    fn push(&mut self, value: f64) {
        *self = match self {
            Self::Unseen => Self::Agreed(value),
            Self::Agreed(seen) if (*seen - value).abs() < f64::EPSILON => Self::Agreed(value),
            _ => Self::Mixed,
        };
    }

    fn resolve(&self) -> Option<f64> {
        match self {
            Self::Agreed(value) => Some(*value),
            _ => None,
        }
    }
}

impl ScreenCalibrationAccumulator {
    /// Record the knobs a run header states.
    pub fn push_header(&mut self, config: &RunConfigRecord) {
        if let Some(rate) = config.screen_sample_rate {
            self.screen_sample_rate.push(rate);
        }
        self.promote_threshold.push(config.screen_promote_threshold);
        self.accept_bar.push(config.min_improvement);
    }

    /// Pair one experiment's screen and full-corpus scores.
    ///
    /// Fails loudly on a score map with no `baseline`: the deltas are measured
    /// against it, so a missing anchor cannot be silently skipped.
    pub fn push_experiment(&mut self, record: &ExperimentRecord) -> Result<(), String> {
        self.experiments += 1;
        let Some(screen) = &record.screen_scores else {
            self.experiments_without_screen += 1;
            return Ok(());
        };
        self.experiments_screened += 1;
        let screen_baseline = *screen.get("baseline").ok_or_else(|| {
            format!(
                "experiment {}: screenScores has no baseline to measure deltas against",
                record.experiment_number
            )
        })?;
        let full_baseline = match record.scores.is_empty() {
            // A batch that screened empty was never promoted: no full scores,
            // and therefore nothing to pair.
            true => None,
            false => Some(*record.scores.get("baseline").ok_or_else(|| {
                format!(
                    "experiment {}: scores has no baseline to measure deltas against",
                    record.experiment_number
                )
            })?),
        };

        for (stem, screen_score) in candidate_stems(screen) {
            match full_baseline.and_then(|base| record.scores.get(stem).map(|s| s - base)) {
                Some(full_delta) => self.pairs.push(ScreenPair {
                    experiment_number: record.experiment_number,
                    stem: stem.to_string(),
                    strategy: strategy_of(record, stem),
                    screen_delta: screen_score - screen_baseline,
                    full_delta,
                }),
                None => self.screen_only += 1,
            }
        }
        for (stem, _) in candidate_stems(&record.scores) {
            if !screen.contains_key(stem) {
                self.full_only += 1;
            }
        }

        if let Some(full_baseline) = full_baseline {
            self.baseline_gaps.push(screen_baseline - full_baseline);
        }

        if record.accepted
            && let Some(stem) = &record.winner
        {
            self.accepted.push(AcceptedScreenPoint {
                experiment_number: record.experiment_number,
                stem: stem.clone(),
                screen_delta: screen.get(stem).map(|s| s - screen_baseline),
                full_delta: full_baseline
                    .and_then(|base| record.scores.get(stem).map(|s| s - base)),
            });
        }
        Ok(())
    }

    /// Finish the accumulation into a report section.
    pub fn finish(self) -> ScreenCalibration {
        let screen_deltas: Vec<f64> = self.pairs.iter().map(|p| p.screen_delta).collect();
        let full_deltas: Vec<f64> = self.pairs.iter().map(|p| p.full_delta).collect();
        let accept_bar = self.accept_bar.resolve();
        let band = accept_bar.unwrap_or(DEFAULT_MIN_IMPROVEMENT).abs();
        let paired = self.pairs.len() as u64;
        let improved = full_deltas.iter().filter(|d| **d > 0.0).count() as u64;
        let (distinct_screen, distinct_full) = distinct_points(&self.pairs);

        ScreenCalibration {
            screen_enabled: self.experiments_screened > 0,
            experiments: self.experiments,
            experiments_screened: self.experiments_screened,
            experiments_without_screen: self.experiments_without_screen,
            screen_sample_rate: self.screen_sample_rate.resolve(),
            promote_threshold: self.promote_threshold.resolve(),
            accept_bar,
            paired_candidates: paired,
            screen_only_candidates: self.screen_only,
            full_only_candidates: self.full_only,
            spearman: spearman_rank_correlation(&screen_deltas, &full_deltas),
            distinct_pairs: distinct_screen.len() as u64,
            spearman_distinct: spearman_rank_correlation(&distinct_screen, &distinct_full),
            promoted_improved: improved,
            promoted_worse: full_deltas.iter().filter(|d| **d < 0.0).count() as u64,
            promoted_clearing_accept_bar: full_deltas.iter().filter(|d| **d > band).count() as u64,
            promoted_materially_worse: full_deltas.iter().filter(|d| **d < -band).count() as u64,
            promotion_precision: (paired > 0).then(|| improved as f64 / paired as f64),
            full_delta: distribution(&full_deltas),
            screen_noise: screen_noise(&self.pairs, band),
            baseline_sample_gap: baseline_gap(&self.baseline_gaps),
            accepted_candidates: self.accepted,
            pairs: self.pairs,
        }
    }
}

/// Every stem in a score map except the `baseline` anchor.
fn candidate_stems(scores: &BTreeMap<String, f64>) -> impl Iterator<Item = (&str, f64)> {
    scores
        .iter()
        .filter(|(stem, _)| stem.as_str() != "baseline")
        .map(|(stem, score)| (stem.as_str(), *score))
}

/// Strategy behind a `candidate-NNN` stem, when the index resolves.
fn strategy_of(record: &ExperimentRecord, stem: &str) -> Option<String> {
    let index: usize = stem.strip_prefix("candidate-")?.parse().ok()?;
    record
        .candidates
        .get(index)
        .map(|prov| prov.strategy.label().to_string())
}

/// The pairs with duplicate (screen Δ, full Δ) points removed, first occurrence
/// kept, split into the two parallel series a correlation needs.
fn distinct_points(pairs: &[ScreenPair]) -> (Vec<f64>, Vec<f64>) {
    let mut seen = std::collections::BTreeSet::new();
    let mut screen = Vec::new();
    let mut full = Vec::new();
    for pair in pairs {
        // Bit patterns, so an exactly repeated proposal is exactly one point.
        if seen.insert((pair.screen_delta.to_bits(), pair.full_delta.to_bits())) {
            screen.push(pair.screen_delta);
            full.push(pair.full_delta);
        }
    }
    (screen, full)
}

/// Spearman rank correlation with tie-averaged ranks.
///
/// `None` below three pairs — two points always correlate perfectly — and
/// `None` when either side is constant, where the coefficient is undefined
/// rather than zero.
pub fn spearman_rank_correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 3 {
        return None;
    }
    let rx = average_ranks(xs);
    let ry = average_ranks(ys);
    let n = rx.len() as f64;
    let mean_x = rx.iter().sum::<f64>() / n;
    let mean_y = ry.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in rx.iter().zip(ry.iter()) {
        cov += (x - mean_x) * (y - mean_y);
        var_x += (x - mean_x).powi(2);
        var_y += (y - mean_y).powi(2);
    }
    if var_x <= 0.0 || var_y <= 0.0 {
        return None;
    }
    Some(cov / (var_x * var_y).sqrt())
}

/// Ranks 1..n, ties sharing the average of the ranks they span.
fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| {
        values[*a]
            .partial_cmp(&values[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; values.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && values[order[j]] == values[order[i]] {
            j += 1;
        }
        // Ranks are 1-based, so the tie block spans i+1 ..= j.
        let shared = ((i + 1 + j) as f64) / 2.0;
        for slot in &order[i..j] {
            ranks[*slot] = shared;
        }
        i = j;
    }
    ranks
}

/// Order statistics of a sample; `None` when the sample is empty.
fn distribution(values: &[f64]) -> Option<DeltaDistribution> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(DeltaDistribution {
        count: sorted.len() as u64,
        min: sorted[0],
        p25: quantile(&sorted, 0.25),
        median: quantile(&sorted, 0.5),
        p75: quantile(&sorted, 0.75),
        max: sorted[sorted.len() - 1],
        mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
    })
}

/// Linear-interpolated quantile of an ascending sample.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * q;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

/// Screen-Δ spread among candidates the full corpus scored inside `±band`.
fn screen_noise(pairs: &[ScreenPair], band: f64) -> Option<ScreenNoise> {
    let flat: Vec<f64> = pairs
        .iter()
        .filter(|p| p.full_delta.abs() <= band)
        .map(|p| p.screen_delta)
        .collect();
    // One observation is a point, not a spread.
    if flat.len() < 2 {
        return None;
    }
    let n = flat.len() as f64;
    let mean = flat.iter().sum::<f64>() / n;
    let variance = flat.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Some(ScreenNoise {
        band,
        samples: flat.len() as u64,
        mean,
        std_dev: variance.sqrt(),
        rms: (flat.iter().map(|d| d * d).sum::<f64>() / n).sqrt(),
        max_abs: flat.iter().fold(0.0_f64, |acc, d| acc.max(d.abs())),
    })
}

/// Spread of the subsample-versus-full-corpus baseline gap.
fn baseline_gap(gaps: &[f64]) -> Option<BaselineSampleGap> {
    if gaps.len() < 2 {
        return None;
    }
    let n = gaps.len() as f64;
    let mean = gaps.iter().sum::<f64>() / n;
    let variance = gaps.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Some(BaselineSampleGap {
        samples: gaps.len() as u64,
        mean,
        std_dev: variance.sqrt(),
        max_abs: gaps.iter().fold(0.0_f64, |acc, g| acc.max(g.abs())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::{CandidateProvenance, CandidateStrategy};
    use std::collections::BTreeMap;

    fn scores(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs
            .iter()
            .map(|(stem, score)| ((*stem).to_string(), *score))
            .collect()
    }

    fn prov(strategy: CandidateStrategy) -> CandidateProvenance {
        CandidateProvenance {
            strategy,
            focus_neuron: "h1".into(),
            mutation: "x".into(),
            old_value: Some(0.0),
            new_value: Some(0.1),
        }
    }

    fn experiment(number: u64) -> ExperimentRecord {
        ExperimentRecord {
            experiment_number: number,
            timestamp_unix: 1000 + number,
            seed: Some(1),
            incumbent_id: "x".into(),
            baseline_score: 0.4,
            focus_neuron: "h1".into(),
            focus_neurons: None,
            focus_stats: None,
            candidates: vec![prov(CandidateStrategy::Random); 8],
            candidates_requested: None,
            batch_limit: None,
            scores: BTreeMap::new(),
            screen_scores: None,
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
        }
    }

    fn calibrate(records: &[ExperimentRecord]) -> ScreenCalibration {
        let mut acc = ScreenCalibrationAccumulator::default();
        for record in records {
            acc.push_experiment(record).expect("well-formed fixture");
        }
        acc.finish()
    }

    /// Only the intersection is paired; the rest is counted, never dropped.
    #[test]
    fn only_candidates_scored_on_both_sides_are_paired() {
        let mut record = experiment(1);
        record.screen_scores = Some(scores(&[
            ("baseline", 0.40),
            ("candidate-000", 0.41),
            ("candidate-001", 0.42),
            ("candidate-002", 0.43),
        ]));
        // The promote batch dropped candidate-002 and carries a stem the screen
        // never saw — the pairing must survive both.
        record.scores = scores(&[
            ("baseline", 0.50),
            ("candidate-000", 0.51),
            ("candidate-001", 0.52),
            ("combo-000-k2", 0.55),
        ]);

        let calibration = calibrate(&[record]);
        assert_eq!(calibration.paired_candidates, 2);
        assert_eq!(calibration.screen_only_candidates, 1, "candidate-002");
        assert_eq!(calibration.full_only_candidates, 1, "combo-000-k2");
        let stems: Vec<&str> = calibration.pairs.iter().map(|p| p.stem.as_str()).collect();
        assert_eq!(stems, vec!["candidate-000", "candidate-001"]);
    }

    /// The baseline anchors the deltas; pairing it would pin the correlation.
    #[test]
    fn the_baseline_stem_is_excluded_from_both_sides() {
        let mut record = experiment(1);
        record.screen_scores = Some(scores(&[
            ("baseline", 0.40),
            ("candidate-000", 0.40 + 3e-6),
            ("candidate-001", 0.40 + 2e-6),
            ("candidate-002", 0.40 + 1e-6),
        ]));
        record.scores = scores(&[
            ("baseline", 0.50),
            ("candidate-000", 0.50 - 3e-6),
            ("candidate-001", 0.50 - 2e-6),
            ("candidate-002", 0.50 - 1e-6),
        ]);

        let calibration = calibrate(&[record]);
        assert_eq!(calibration.paired_candidates, 3);
        assert!(
            calibration.pairs.iter().all(|p| p.stem != "baseline"),
            "the anchor is not an observation"
        );
        // Perfectly reversed ranks: the screen ordering is exactly wrong.
        assert!((calibration.spearman.unwrap() + 1.0).abs() < 1e-12);
        // Every delta is measured against its own phase's baseline, so the
        // 0.1 gap between the two baselines never leaks into a delta.
        assert!((calibration.pairs[0].screen_delta - 3e-6).abs() < 1e-15);
        assert!((calibration.pairs[0].full_delta + 3e-6).abs() < 1e-15);
    }

    /// Screening disabled: a clean "not applicable", not a fabricated number.
    #[test]
    fn a_journal_without_a_screen_phase_reports_not_applicable() {
        let mut record = experiment(1);
        record.screen_scores = None;
        record.scores = scores(&[("baseline", 0.40), ("candidate-000", 0.41)]);

        let calibration = calibrate(&[record]);
        assert!(!calibration.screen_enabled);
        assert_eq!(calibration.experiments, 1);
        assert_eq!(calibration.experiments_without_screen, 1);
        assert_eq!(calibration.paired_candidates, 0);
        assert_eq!(calibration.spearman, None);
        assert_eq!(calibration.promotion_precision, None);
        assert_eq!(calibration.full_delta, None);
        assert_eq!(calibration.screen_noise, None);
        assert!(calibration.pairs.is_empty());
    }

    /// A batch that screened empty pairs nothing but is still a screened
    /// experiment — its candidates are the screen's own rejections.
    #[test]
    fn a_screen_empty_experiment_contributes_screen_only_candidates() {
        let mut record = experiment(1);
        record.screen_scores = Some(scores(&[
            ("baseline", 0.40),
            ("candidate-000", 0.399),
            ("candidate-001", 0.398),
        ]));

        let calibration = calibrate(&[record]);
        assert!(calibration.screen_enabled);
        assert_eq!(calibration.experiments_screened, 1);
        assert_eq!(calibration.screen_only_candidates, 2);
        assert_eq!(calibration.paired_candidates, 0);
        assert_eq!(calibration.baseline_sample_gap, None);
    }

    /// A score map with no anchor is a malformed journal, not an empty one.
    #[test]
    fn a_screen_map_without_a_baseline_fails_loudly() {
        let mut record = experiment(7);
        record.screen_scores = Some(scores(&[("candidate-000", 0.41)]));

        let mut acc = ScreenCalibrationAccumulator::default();
        let error = acc.push_experiment(&record).expect_err("no anchor");
        assert!(error.contains("experiment 7"), "{error}");
        assert!(error.contains("baseline"), "{error}");
    }

    /// Promote-side anchor missing while candidates were promoted: same rule.
    #[test]
    fn a_promote_map_without_a_baseline_fails_loudly() {
        let mut record = experiment(9);
        record.screen_scores = Some(scores(&[("baseline", 0.4), ("candidate-000", 0.41)]));
        record.scores = scores(&[("candidate-000", 0.51)]);

        let mut acc = ScreenCalibrationAccumulator::default();
        let error = acc.push_experiment(&record).expect_err("no anchor");
        assert!(error.contains("experiment 9"), "{error}");
    }

    /// Precision, the accept-bar counts and the full-Δ spread, hand-computed.
    #[test]
    fn promotion_precision_and_spread_are_hand_computable() {
        let mut record = experiment(1);
        record.screen_scores = Some(scores(&[
            ("baseline", 0.40),
            ("candidate-000", 0.40 + 1e-5),
            ("candidate-001", 0.40 + 2e-5),
            ("candidate-002", 0.40 + 3e-5),
            ("candidate-003", 0.40 + 4e-5),
        ]));
        // Full-corpus deltas: +2e-6, +5e-7, -5e-7, -2e-6.
        record.scores = scores(&[
            ("baseline", 0.50),
            ("candidate-000", 0.50 + 2e-6),
            ("candidate-001", 0.50 + 5e-7),
            ("candidate-002", 0.50 - 5e-7),
            ("candidate-003", 0.50 - 2e-6),
        ]);

        let calibration = calibrate(&[record]);
        assert_eq!(calibration.paired_candidates, 4);
        assert_eq!(calibration.promoted_improved, 2);
        assert_eq!(calibration.promoted_worse, 2);
        // The default 1e-6 band applies when no header states an accept bar.
        assert_eq!(calibration.promoted_clearing_accept_bar, 1);
        assert_eq!(calibration.promoted_materially_worse, 1);
        assert!((calibration.promotion_precision.unwrap() - 0.5).abs() < 1e-12);

        let spread = calibration.full_delta.expect("promoted candidates");
        assert_eq!(spread.count, 4);
        assert!((spread.min + 2e-6).abs() < 1e-15);
        assert!((spread.max - 2e-6).abs() < 1e-15);
        assert!(spread.median.abs() < 1e-15, "median of ±5e-7 is 0");
        assert!(spread.mean.abs() < 1e-15);
        // p25 interpolates between -2e-6 and -5e-7 at 0.75 of the gap.
        assert!((spread.p25 + 8.75e-7).abs() < 1e-15);
        assert!((spread.p75 - 8.75e-7).abs() < 1e-15);
    }

    /// A repeated proposal is one hypothesis, however many times it is scored.
    #[test]
    fn repeated_points_are_counted_once_for_the_sensitivity_check() {
        // Two experiments replaying the same three proposals, plus one point
        // that only the second experiment saw.
        let mut first = experiment(1);
        first.screen_scores = Some(scores(&[
            ("baseline", 0.40),
            ("candidate-000", 0.40 + 1e-5),
            ("candidate-001", 0.40 + 2e-5),
            ("candidate-002", 0.40 + 3e-5),
        ]));
        first.scores = scores(&[
            ("baseline", 0.50),
            ("candidate-000", 0.50 + 3e-6),
            ("candidate-001", 0.50 + 2e-6),
            ("candidate-002", 0.50 + 1e-6),
        ]);
        let mut second = first.clone();
        second.experiment_number = 2;
        let mut screen = second.screen_scores.clone().unwrap();
        screen.insert("candidate-003".into(), 0.40 + 4e-5);
        second.screen_scores = Some(screen);
        second.scores.insert("candidate-003".into(), 0.50 + 4e-6);

        let calibration = calibrate(&[first, second]);
        assert_eq!(calibration.paired_candidates, 7);
        assert_eq!(calibration.distinct_pairs, 4, "three points were replayed");
        // The replayed points are perfectly reversed; the fourth agrees. Over
        // all seven the repeats dominate, over the four distinct they do not.
        assert!(calibration.spearman.unwrap() < calibration.spearman_distinct.unwrap());
    }

    /// Tied screen deltas share an averaged rank instead of an arbitrary one.
    #[test]
    fn tied_screen_deltas_share_an_averaged_rank() {
        assert_eq!(average_ranks(&[5.0, 5.0, 1.0]), vec![2.5, 2.5, 1.0]);
        // Two of three screen deltas tie, so the screen cannot order them; the
        // correlation is real but not perfect.
        let rho = spearman_rank_correlation(&[1.0, 1.0, 2.0], &[1.0, 2.0, 3.0]).unwrap();
        assert!((rho - 0.8660254037844387).abs() < 1e-12, "rho was {rho}");
    }

    /// A constant side has no ordering, so there is no coefficient to report.
    #[test]
    fn a_constant_side_reports_no_correlation() {
        assert_eq!(
            spearman_rank_correlation(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]),
            None
        );
        assert_eq!(spearman_rank_correlation(&[1.0, 2.0], &[1.0, 2.0]), None);
    }

    /// The noise floor is the screen-Δ spread where the full corpus says zero.
    #[test]
    fn the_noise_floor_comes_from_candidates_the_full_corpus_scores_flat() {
        let mut record = experiment(1);
        record.screen_scores = Some(scores(&[
            ("baseline", 0.40),
            ("candidate-000", 0.40 + 2e-5),
            ("candidate-001", 0.40 + 4e-5),
            // Excluded: the full corpus says this one genuinely moved.
            ("candidate-002", 0.40 + 9e-5),
        ]));
        record.scores = scores(&[
            ("baseline", 0.50),
            ("candidate-000", 0.50 + 1e-7),
            ("candidate-001", 0.50 - 1e-7),
            ("candidate-002", 0.50 + 5e-6),
        ]);

        let noise = calibrate(&[record])
            .screen_noise
            .expect("two flat candidates");
        assert_eq!(noise.samples, 2, "only the ±1e-6 band counts");
        assert!((noise.band - 1e-6).abs() < 1e-15);
        assert!((noise.mean - 3e-5).abs() < 1e-15);
        // Sample sd of {2e-5, 4e-5} is 1.414…e-5.
        assert!((noise.std_dev - 2e-5 / 2.0_f64.sqrt()).abs() < 1e-15);
        assert!((noise.max_abs - 4e-5).abs() < 1e-15);
    }

    /// The baseline is scored twice per promoting experiment; the gap between
    /// those two numbers is sampling error with no selection effect.
    #[test]
    fn the_baseline_sample_gap_measures_subsample_error_directly() {
        let mut first = experiment(1);
        first.screen_scores = Some(scores(&[("baseline", 0.402), ("candidate-000", 0.41)]));
        first.scores = scores(&[("baseline", 0.400), ("candidate-000", 0.4000001)]);
        let mut second = experiment(2);
        second.screen_scores = Some(scores(&[("baseline", 0.398), ("candidate-000", 0.41)]));
        second.scores = scores(&[("baseline", 0.400), ("candidate-000", 0.4000001)]);

        let gap = calibrate(&[first, second])
            .baseline_sample_gap
            .expect("two promoting experiments");
        assert_eq!(gap.samples, 2);
        assert!(gap.mean.abs() < 1e-15, "+2e-3 and -2e-3 cancel");
        // Sample sd of {+2e-3, -2e-3} is 2√2 e-3.
        assert!((gap.std_dev - 2e-3 * 2.0_f64.sqrt()).abs() < 1e-15);
        assert!((gap.max_abs - 2e-3).abs() < 1e-15);
    }

    /// The accepted candidates' screen Δ is the false-negative evidence.
    #[test]
    fn every_accepted_candidate_reports_the_screen_delta_that_promoted_it() {
        let mut single = experiment(1);
        single.screen_scores = Some(scores(&[
            ("baseline", 0.40),
            ("candidate-000", 0.40 + 4e-6),
            ("candidate-001", 0.40 + 9e-6),
        ]));
        single.scores = scores(&[
            ("baseline", 0.50),
            ("candidate-000", 0.50 + 2e-6),
            ("candidate-001", 0.50 - 1e-6),
        ]);
        single.winner = Some("candidate-000".into());
        single.improvement = Some(2e-6);
        single.accepted = true;

        // A merged combo is assembled after the screen, so it has no screen Δ.
        let mut combo = experiment(2);
        combo.screen_scores = Some(scores(&[
            ("baseline", 0.40),
            ("candidate-000", 0.40 + 5e-6),
        ]));
        combo.scores = scores(&[("baseline", 0.50), ("candidate-000", 0.50 + 3e-6)]);
        combo.winner = Some("combo-000-k2".into());
        combo.improvement = Some(3e-6);
        combo.accepted = true;

        let calibration = calibrate(&[single, combo]);
        assert_eq!(calibration.accepted_candidates.len(), 2);
        let first = &calibration.accepted_candidates[0];
        assert_eq!(first.stem, "candidate-000");
        assert!((first.screen_delta.unwrap() - 4e-6).abs() < 1e-15);
        assert!((first.full_delta.unwrap() - 2e-6).abs() < 1e-15);
        let second = &calibration.accepted_candidates[1];
        assert_eq!(second.screen_delta, None, "a combo was never screened");
        assert_eq!(second.full_delta, None);
    }

    /// Each pair names the strategy that proposed it, so the correlation can be
    /// read per strategy family.
    #[test]
    fn a_pair_names_the_strategy_behind_its_stem() {
        let mut record = experiment(1);
        record.candidates = vec![
            prov(CandidateStrategy::Random),
            prov(CandidateStrategy::Backprop),
        ];
        record.screen_scores = Some(scores(&[
            ("baseline", 0.4),
            ("candidate-000", 0.41),
            ("candidate-001", 0.42),
        ]));
        record.scores = scores(&[
            ("baseline", 0.5),
            ("candidate-000", 0.51),
            ("candidate-001", 0.52),
        ]);

        let calibration = calibrate(&[record]);
        assert_eq!(calibration.pairs[0].strategy.as_deref(), Some("random"));
        assert_eq!(calibration.pairs[1].strategy.as_deref(), Some("backprop"));
    }

    /// Headers from an A/B campaign disagree, so no single threshold is quoted.
    #[test]
    fn disagreeing_run_headers_report_no_single_knob() {
        let mut acc = ScreenCalibrationAccumulator::default();
        let mut config = RunConfigRecord::from_config(
            &crate::config::LamarckConfig::default(),
            crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON,
        );
        acc.push_header(&config);
        config.screen_promote_threshold = 5e-6;
        acc.push_header(&config);

        let calibration = acc.finish();
        assert_eq!(calibration.promote_threshold, None, "two arms, two gates");
        assert_eq!(
            calibration.screen_sample_rate,
            Some(crate::config::DEFAULT_SCREEN_SAMPLE_RATE),
            "the arms agreed on the rate"
        );
        assert_eq!(
            calibration.accept_bar,
            Some(crate::config::DEFAULT_MIN_IMPROVEMENT)
        );
    }

    /// The header's accept bar — not the default — sets the near-zero band.
    #[test]
    fn the_header_accept_bar_sets_the_near_zero_band() {
        let mut acc = ScreenCalibrationAccumulator::default();
        let mut config = RunConfigRecord::from_config(
            &crate::config::LamarckConfig::default(),
            crate::baseline::DEFAULT_BASELINE_DRIFT_EPSILON,
        );
        config.min_improvement = 1e-5;
        acc.push_header(&config);

        let mut record = experiment(1);
        record.screen_scores = Some(scores(&[
            ("baseline", 0.40),
            ("candidate-000", 0.40 + 2e-5),
            ("candidate-001", 0.40 + 4e-5),
        ]));
        // Both sit inside ±1e-5 but outside ±1e-6.
        record.scores = scores(&[
            ("baseline", 0.50),
            ("candidate-000", 0.50 + 5e-6),
            ("candidate-001", 0.50 - 5e-6),
        ]);
        acc.push_experiment(&record).unwrap();

        let calibration = acc.finish();
        let noise = calibration.screen_noise.expect("both inside the band");
        assert!((noise.band - 1e-5).abs() < 1e-15);
        assert_eq!(noise.samples, 2);
        assert_eq!(calibration.promoted_clearing_accept_bar, 0, "bar is 1e-5");
    }
}
