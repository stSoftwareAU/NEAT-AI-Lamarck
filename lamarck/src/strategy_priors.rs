//! Transferable operator priors carried across runs (issue #221).
//!
//! Lamarck runs are deliberately short, because the champion they are handed
//! goes stale. Every run therefore re-discovers which mutation families are
//! currently productive, even though the journals of earlier runs already
//! measured that — against a different incumbent, but with the same operators.
//!
//! This module carries the **operator-level** half of that knowledge forward
//! and nothing else. What is persisted is one decayed
//! [`StrategyEvidence`] row per strategy: trials, screen→promote conversions,
//! accepts, full-corpus score gain and measured scorer cost. From those five
//! numbers a reader derives the screen-to-promote precision and the cost per
//! useful candidate; from none of them can a historical candidate be
//! reconstructed. There is no creature, no mutation, no focus neuron and no
//! scalar in the file — the format cannot replay an old edit even in principle,
//! which is the guardrail #221 asks for, held by construction rather than by a
//! check somewhere downstream.
//!
//! # What a prior is worth
//!
//! A prior is folded into the next run's [`StrategyLedger`] scaled by a
//! **confidence** in `[0, 1]`, the product of three independent discounts:
//!
//! * **Age** — `0.5 ^ (age hours / half-life)`, and exactly zero beyond
//!   [`STRATEGY_PRIORS_MAX_AGE_HOURS`]. A week-old measurement of a creature
//!   that has since accepted a hundred times is not evidence.
//! * **Corpus** — the training-data fingerprint must match, or the evidence was
//!   measured over data this run is not scoring against
//!   ([`CORPUS_MISMATCH_CONFIDENCE`]). A corpus that could not be fingerprinted
//!   at either end is treated as a mismatch: unconfirmed is not confirmed.
//! * **Source creature** — the same creature keeps full confidence; the same
//!   topology with different weights keeps [`TOPOLOGY_MATCH_CONFIDENCE`]; a
//!   different topology keeps [`TOPOLOGY_DRIFT_CONFIDENCE`].
//!
//! The scaled mass is then capped at [`MAX_PRIOR_TRIALS`] trials per arm — about
//! a quarter of one production batch — so however long the history behind a
//! prior is, it is worth a fraction of the run's own measurement and no more.
//! The run's ledger decays every arm once per experiment, so the prior's share
//! of an arm's evidence falls geometrically as the run measures its own — which
//! is what lets fresh evidence dominate a stale prior within a handful of
//! experiments.
//!
//! What a run writes back is what **it** measured, never what it inherited:
//! re-stamping an inherited row with today's timestamp and today's incumbent
//! would launder exactly the age bound and source discount that bound it, so a
//! week-old measurement could ride a nightly chain forever. Evidence therefore
//! transfers one run at a time, each hop discounted on its own merits.
//!
//! Priors initialise the allocation and never bypass a gate: a candidate is
//! still generated, screened and scored exactly as it would be on a cold start.
//! They reach the batch through the adaptive allocator, so they move nothing
//! under the default `--strategy-allocation fixed` — the run says so rather
//! than seeding an allocation nobody consults.

use crate::candidates::CandidateStrategy;
use crate::strategy_allocation::{StrategyEvidence, StrategyLedger};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Priors file written beside the journal in the run's output directory.
pub const STRATEGY_PRIORS_FILE: &str = "strategy-priors.json";

/// Prior format version, following the `observations.rs` pattern.
///
/// Bump this whenever the on-disk shape changes; a mismatch is refused rather
/// than misread, and the run simply starts cold.
pub const STRATEGY_PRIORS_FORMAT_VERSION: &str = "1.0.0";

/// Default half-life of a prior's confidence, in hours.
///
/// A day: two consecutive nightly runs against the same creature and corpus
/// still trade half their evidence, and a prior left over a long weekend is
/// worth an eighth of one measured today.
pub const DEFAULT_STRATEGY_PRIORS_HALF_LIFE_HOURS: f64 = 24.0;

/// Age at which a prior carries no confidence at all, in hours (7 days).
///
/// The half-life alone only ever approaches zero. The hard bound is what makes
/// "maximum age" a number rather than an adjective, and it matches the
/// failed-candidate cache's own week
/// (`crate::failed_cache::DEFAULT_FAILED_CACHE_MAX_AGE_SECONDS`).
pub const STRATEGY_PRIORS_MAX_AGE_HOURS: f64 = 168.0;

/// Confidence retained when the training corpus does not match.
///
/// Not zero: the operators are properties of the creature and the generator as
/// much as of the data, so which families propose *anything* scorable survives
/// a corpus change. Not much more than zero either — the score gains behind the
/// evidence were measured over records this run will not see.
pub const CORPUS_MISMATCH_CONFIDENCE: f64 = 0.25;

/// Confidence retained when the source creature has the same topology but
/// different weights — the ordinary case for a run seeded by yesterday's
/// champion.
pub const TOPOLOGY_MATCH_CONFIDENCE: f64 = 0.5;

/// Confidence retained when the source creature's topology has drifted.
///
/// A structural accept changes which operators can even apply, so evidence from
/// a differently-shaped creature is the weakest transferable kind.
pub const TOPOLOGY_DRIFT_CONFIDENCE: f64 = 0.25;

/// Most trials one arm's prior may carry into a run, after confidence.
///
/// Twenty-five candidate trials is roughly a quarter of one production batch:
/// enough to move the opening allocation, far too little to hold it against
/// what the run goes on to measure. The cap is applied to the arm's whole
/// evidence row — trials, promotions, accepts, gain and cost scale together —
/// so a capped prior keeps its measured *rate* and loses only its weight.
pub const MAX_PRIOR_TRIALS: f64 = 25.0;

/// Whether a run seeds its allocation from persisted priors (issue #221).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StrategyPriorsMode {
    /// Cold start: priors are neither read nor written. The A/B control.
    #[default]
    Off,
    /// Read priors at startup, seed the ledger, and write them back at the end.
    Seed,
}

impl StrategyPriorsMode {
    /// Parse a CLI spelling (`off` / `seed`).
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "seed" => Some(Self::Seed),
            _ => None,
        }
    }

    /// Stable label for logs, journals and reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Seed => "seed",
        }
    }

    /// True when priors are read at startup and written at the end.
    pub fn is_enabled(self) -> bool {
        matches!(self, Self::Seed)
    }
}

/// Validated prior policy for one run (issue #221).
#[derive(Debug, Clone, PartialEq)]
pub struct PriorsPolicy {
    /// Mode in force.
    pub mode: StrategyPriorsMode,
    /// File the priors are read from and written to.
    pub path: PathBuf,
    /// Half-life of prior confidence, in hours.
    pub half_life_hours: f64,
}

impl PriorsPolicy {
    /// True when priors are read at startup and written at the end.
    pub fn is_enabled(&self) -> bool {
        self.mode.is_enabled()
    }
}

/// What a prior was measured against — fingerprints only, never names.
///
/// Every field is either a coarse shape id (`in2-out1-n3-s4`, which describes
/// no application) or an opaque hash. Nothing here identifies a corpus, a file
/// or a deployment, which is what makes the file portable between machines
/// without carrying private metadata off one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PriorSource {
    /// Coarse journal `incumbentId` — the creature's shape, not its identity.
    pub incumbent_id: String,
    /// Content fingerprint of the creature the evidence was measured against.
    pub creature_fingerprint: u64,
    /// Fingerprint of the training corpus, when one could be taken.
    #[serde(default)]
    pub training_key: Option<u64>,
}

/// How far a prior is trusted, and why (issue #221).
///
/// Every factor is reported separately: a run that discounted its history to a
/// quarter should say whether that was age, a new corpus or a reshaped
/// creature, because those call for different responses.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PriorConfidence {
    /// Age of the prior in hours at the time it was read.
    pub age_hours: f64,
    /// Age discount: `0.5 ^ (age / half-life)`, or `0` past the maximum age.
    pub age: f64,
    /// Corpus-fingerprint discount.
    pub corpus: f64,
    /// Source-creature discount.
    pub source: f64,
    /// The product actually applied to the persisted evidence.
    pub confidence: f64,
}

/// Prior evidence per arm, scaled and capped, ready to seed a ledger.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PriorSeed {
    /// Evidence to fold into the ledger, by strategy.
    pub arms: BTreeMap<CandidateStrategy, StrategyEvidence>,
    /// Labels in the file that this build does not recognise.
    ///
    /// Kept rather than dropped silently: a prior written by a newer Lamarck
    /// carrying an operator this binary has never heard of is a fact the run
    /// log should state, not a line to swallow.
    pub unknown: Vec<String>,
}

impl PriorSeed {
    /// Total trials carried across every arm.
    pub fn total_trials(&self) -> f64 {
        self.arms.values().map(|evidence| evidence.trials).sum()
    }
}

/// Portable, versioned operator priors (issue #221).
///
/// The whole file: a format version, when and against what it was measured, and
/// one decayed evidence row per strategy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyPriors {
    /// Prior format version.
    pub format_version: String,
    /// Unix timestamp the priors were written at, which fixes their age.
    pub written_unix: u64,
    /// `--min-improvement` the score gains were measured against.
    ///
    /// Recorded so a reader can price the gains in accept-bar units the way the
    /// run that wrote them did; the gains themselves are absolute, so a run at
    /// a different bar still reads them correctly.
    pub min_improvement: f64,
    /// What the evidence was measured against.
    pub source: PriorSource,
    /// Decayed evidence per strategy label.
    pub arms: BTreeMap<String, StrategyEvidence>,
}

impl StrategyPriors {
    /// Priors describing what `ledger` **measured**, keyed to `source`.
    ///
    /// The ledger handed in is the run's own decayed ledger, so the file
    /// carries what was recently productive rather than a lifetime total. Any
    /// evidence the ledger was itself seeded with is taken back out
    /// ([`StrategyLedger::measured_evidence`]): a file stamped `now` and keyed
    /// to this run's incumbent must contain only evidence that is actually that
    /// fresh and was actually measured against that creature, or the age bound
    /// and the source discount would both be laundered on every hop of a
    /// nightly chain.
    pub fn from_ledger(
        ledger: &StrategyLedger,
        source: PriorSource,
        min_improvement: f64,
        now_unix: u64,
    ) -> Self {
        let arms = ledger
            .strategies()
            .map(|strategy| (strategy, ledger.measured_evidence(strategy)))
            .filter(|(_, evidence)| evidence.trials > 0.0)
            .map(|(strategy, evidence)| (strategy.label().to_string(), evidence))
            .collect();
        Self {
            format_version: STRATEGY_PRIORS_FORMAT_VERSION.to_string(),
            written_unix: now_unix,
            min_improvement,
            source,
            arms,
        }
    }

    /// How far these priors are trusted against `against`, read at `now_unix`.
    pub fn confidence(
        &self,
        against: &PriorSource,
        now_unix: u64,
        half_life_hours: f64,
    ) -> PriorConfidence {
        let age_hours = now_unix.saturating_sub(self.written_unix) as f64 / 3_600.0;
        let age = if age_hours > STRATEGY_PRIORS_MAX_AGE_HOURS {
            0.0
        } else if half_life_hours.is_finite() && half_life_hours > 0.0 {
            0.5_f64.powf(age_hours / half_life_hours)
        } else {
            // An unusable half-life must not silently become "trust it all":
            // the caller validates the knob, and this is the safe reading.
            0.0
        };
        let corpus = match (self.source.training_key, against.training_key) {
            (Some(mine), Some(theirs)) if mine == theirs => 1.0,
            _ => CORPUS_MISMATCH_CONFIDENCE,
        };
        let source = if self.source.creature_fingerprint == against.creature_fingerprint {
            1.0
        } else if self.source.incumbent_id == against.incumbent_id {
            TOPOLOGY_MATCH_CONFIDENCE
        } else {
            TOPOLOGY_DRIFT_CONFIDENCE
        };
        PriorConfidence {
            age_hours,
            age,
            corpus,
            source,
            confidence: (age * corpus * source).clamp(0.0, 1.0),
        }
    }

    /// Evidence to seed a ledger with at `confidence`, capped per arm.
    pub fn seed(&self, confidence: f64) -> PriorSeed {
        let mut seed = PriorSeed::default();
        if !confidence.is_finite() || confidence <= 0.0 {
            return seed;
        }
        for (label, evidence) in &self.arms {
            let Some(strategy) = CandidateStrategy::parse(label) else {
                seed.unknown.push(label.clone());
                continue;
            };
            let mut scaled = *evidence;
            scaled.scale(confidence.min(1.0));
            if scaled.trials > MAX_PRIOR_TRIALS {
                scaled.scale(MAX_PRIOR_TRIALS / scaled.trials);
            }
            if scaled.trials > 0.0 {
                seed.arms.insert(strategy, scaled);
            }
        }
        seed
    }
}

/// Path of the priors file inside `output_dir`.
pub fn priors_path(output_dir: &Path) -> PathBuf {
    output_dir.join(STRATEGY_PRIORS_FILE)
}

/// Read priors from `path`.
///
/// `Ok(None)` means there are none to read — the first run of a chain starts
/// cold, which is not a failure. Anything else is reported: an unreadable,
/// malformed or version-mismatched file is a fault the run log names rather
/// than a silent cold start that looks identical to having no history.
pub fn load_priors(path: &Path) -> Result<Option<StrategyPriors>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path).map_err(|e| format!("unreadable priors: {e}"))?;
    let priors: StrategyPriors =
        serde_json::from_str(&text).map_err(|e| format!("unparsable priors: {e}"))?;
    if priors.format_version != STRATEGY_PRIORS_FORMAT_VERSION {
        return Err(format!(
            "unsupported priors formatVersion {} (want {STRATEGY_PRIORS_FORMAT_VERSION})",
            priors.format_version
        ));
    }
    Ok(Some(priors))
}

/// Write `priors` to `path`, returning the on-disk byte count.
pub fn write_priors(path: &Path, priors: &StrategyPriors) -> Result<u64, String> {
    let encoded = serde_json::to_string(priors).map_err(|e| e.to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(path, &encoded).map_err(|e| e.to_string())?;
    Ok(encoded.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> PriorSource {
        PriorSource {
            incumbent_id: "in2-out1-n3-s4".into(),
            creature_fingerprint: 11,
            training_key: Some(22),
        }
    }

    fn priors(trials: f64) -> StrategyPriors {
        StrategyPriors {
            format_version: STRATEGY_PRIORS_FORMAT_VERSION.to_string(),
            written_unix: 1_000,
            min_improvement: 1e-6,
            source: source(),
            arms: BTreeMap::from([(
                CandidateStrategy::StructuralAdd.label().to_string(),
                StrategyEvidence {
                    trials,
                    promotions: trials / 10.0,
                    accepts: 1.0,
                    score_gain: 4e-6,
                    cost_ms: 20_000.0,
                },
            )]),
        }
    }

    #[test]
    fn an_unknown_arm_label_is_reported_rather_than_dropped() {
        let mut priors = priors(4.0);
        priors
            .arms
            .insert("quantum_tunnel".to_string(), StrategyEvidence::default());
        let seed = priors.seed(1.0);
        assert_eq!(seed.unknown, vec!["quantum_tunnel".to_string()]);
        assert_eq!(seed.arms.len(), 1);
    }

    #[test]
    fn a_zero_or_non_finite_confidence_seeds_nothing() {
        assert!(priors(4.0).seed(0.0).arms.is_empty());
        assert!(priors(4.0).seed(f64::NAN).arms.is_empty());
        assert!(priors(4.0).seed(-1.0).arms.is_empty());
    }

    /// The cap keeps the arm's measured rate and takes only its weight.
    #[test]
    fn capping_scales_the_whole_evidence_row() {
        let seed = priors(100.0).seed(1.0);
        let arm = seed.arms[&CandidateStrategy::StructuralAdd];
        assert!((arm.trials - MAX_PRIOR_TRIALS).abs() < 1e-9);
        assert!((arm.promotions - 2.5).abs() < 1e-9, "{}", arm.promotions);
        assert!((arm.cost_ms - 5_000.0).abs() < 1e-9, "{}", arm.cost_ms);
    }

    #[test]
    fn a_future_timestamp_is_read_as_fresh_rather_than_negative_age() {
        let confidence = priors(4.0).confidence(&source(), 0, 24.0);
        assert_eq!(confidence.age_hours, 0.0);
        assert_eq!(confidence.confidence, 1.0);
    }

    #[test]
    fn an_unusable_half_life_trusts_nothing() {
        for half_life in [0.0, -1.0, f64::NAN] {
            let confidence = priors(4.0).confidence(&source(), 1_000, half_life);
            assert_eq!(confidence.confidence, 0.0, "half-life {half_life}");
        }
    }
}
