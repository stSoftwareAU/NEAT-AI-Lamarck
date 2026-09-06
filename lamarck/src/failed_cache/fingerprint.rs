//! Candidate identity and near-duplicate matching (issue #88).
//!
//! The failed-candidate cache needs a definition of "the same candidate". A
//! candidate is identified by its [`CandidateProvenance`] plus the incumbent it
//! was proposed against — never by its creature JSON, because hashing a
//! ~2500-input creature per candidate is exactly the overhead the cache exists
//! to avoid.
//!
//! Identity splits in two:
//!
//! * the **discrete part** ([`FingerprintBucketKey`]) — incumbent, strategy,
//!   focus neuron and mutation description — which must match exactly, since it
//!   names the knob being turned; and
//! * the **changed scalar** (`new_value`), which matches within a
//!   relative-or-absolute tolerance. A pure relative bound is wrong because
//!   production weight/bias deltas pass through zero; a pure absolute bound is
//!   wrong because weights span orders of magnitude.
//!
//! The discrete part is a hash key, so a lookup only tolerance-compares the
//! scalars inside one bucket rather than scanning the whole cache.

use crate::candidates::{CandidateProvenance, CandidateStrategy};
use serde::{Deserialize, Serialize};

/// Default absolute bound on a `new_value` near-duplicate match.
///
/// Deltas that cross zero have no meaningful relative scale, so the absolute
/// bound is what catches them. `1e-9` is two orders below
/// [`crate::config::DEFAULT_MIN_IMPROVEMENT`]-scale weight nudges: close enough
/// that two proposals cannot score differently, far enough that a genuinely new
/// step is never swallowed.
pub const DEFAULT_FAILED_CACHE_TOLERANCE_ABS: f64 = 1e-9;

/// Default relative bound on a `new_value` near-duplicate match.
///
/// Production weights span orders of magnitude, so large values need a
/// proportional bound. `1e-6` treats two weights agreeing to six significant
/// figures as the same proposal.
pub const DEFAULT_FAILED_CACHE_TOLERANCE_REL: f64 = 1e-6;

/// Near-duplicate bounds for the changed scalar of a candidate.
///
/// Two values match when `|a - b| <= max(abs, rel * max(|a|, |b|))`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tolerance {
    /// Absolute bound, which carries the match across zero.
    pub abs: f64,
    /// Relative bound, which carries the match at large magnitudes.
    pub rel: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            abs: DEFAULT_FAILED_CACHE_TOLERANCE_ABS,
            rel: DEFAULT_FAILED_CACHE_TOLERANCE_REL,
        }
    }
}

impl Tolerance {
    /// Bounds from an explicit absolute and relative pair.
    ///
    /// A non-finite or negative bound is clamped to `0`, which narrows matching
    /// rather than widening it: a misconfigured knob must never make the cache
    /// skip a candidate it has not actually seen.
    pub fn new(abs: f64, rel: f64) -> Self {
        Self {
            abs: sanitise_bound(abs),
            rel: sanitise_bound(rel),
        }
    }

    /// Whether `a` and `b` are the same proposal within these bounds.
    ///
    /// Non-finite values never match, not even themselves: a `NaN` weight is
    /// not evidence about any mutation, so it must never register as a
    /// known-failed candidate.
    pub fn matches(self, a: f64, b: f64) -> bool {
        if !a.is_finite() || !b.is_finite() {
            return false;
        }
        let bound = self.abs.max(self.rel * a.abs().max(b.abs()));
        (a - b).abs() <= bound
    }
}

fn sanitise_bound(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

/// Exact-match part of a candidate identity, used as the cache's hash key.
///
/// Bucketing on this key is what keeps lookup off a linear scan: only the
/// candidates that turned the *same* knob on the *same* incumbent are ever
/// tolerance-compared.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FingerprintBucketKey {
    /// Incumbent identity, as journalled in `ExperimentRecord::incumbent_id`.
    pub incumbent_id: String,
    /// Strategy that produced the candidate.
    pub strategy: CandidateStrategy,
    /// Focus neuron UUID.
    pub focus_neuron: String,
    /// Mutation description, which encodes *what* changed.
    pub mutation: String,
}

/// Identity of one candidate: an exact-match bucket plus a changed scalar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateFingerprint {
    /// Exact-match discrete part.
    pub bucket: FingerprintBucketKey,
    /// Changed scalar, absent for structural mutations that change no value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_value: Option<f64>,
}

impl CandidateFingerprint {
    /// Fingerprint a candidate proposed against the incumbent `incumbent_id`.
    pub fn from_provenance(incumbent_id: &str, provenance: &CandidateProvenance) -> Self {
        Self {
            bucket: FingerprintBucketKey {
                incumbent_id: incumbent_id.to_string(),
                strategy: provenance.strategy,
                focus_neuron: provenance.focus_neuron.clone(),
                mutation: provenance.mutation.clone(),
            },
            new_value: provenance.new_value,
        }
    }

    /// Whether this candidate may be recorded as known-failed.
    ///
    /// A non-finite `new_value` is never cacheable: it can match nothing (not
    /// even an identical fingerprint), so storing it would only consume an
    /// entry that can never produce a hit.
    pub fn is_cacheable(&self) -> bool {
        self.new_value.is_none_or(f64::is_finite)
    }

    /// Whether `other` is the same candidate within `tolerance`.
    ///
    /// The discrete part must match exactly. `None`/`None` matches on the
    /// discrete part alone; `None` against `Some` never matches.
    pub fn matches(&self, other: &Self, tolerance: Tolerance) -> bool {
        self.matches_with(other, |a, b| tolerance.matches(a, b))
    }

    /// [`Self::matches`] with an injected scalar comparator.
    ///
    /// Exists so a test can count how many tolerance comparisons a lookup
    /// performs and prove a non-matching bucket performs none.
    fn matches_with(&self, other: &Self, mut compare: impl FnMut(f64, f64) -> bool) -> bool {
        if self.bucket != other.bucket {
            return false;
        }
        match (self.new_value, other.new_value) {
            (None, None) => true,
            (Some(a), Some(b)) => compare(a, b),
            (None, Some(_)) | (Some(_), None) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn provenance(strategy: CandidateStrategy, new_value: Option<f64>) -> CandidateProvenance {
        CandidateProvenance {
            strategy,
            focus_neuron: "o1".into(),
            mutation: "weight input-0->h1 1.0 -> 1.5".into(),
            old_value: Some(1.0),
            new_value,
            mirror: None,
            follow_up: None,
        }
    }

    fn fingerprint(new_value: Option<f64>) -> CandidateFingerprint {
        CandidateFingerprint::from_provenance(
            "in1-out1-n2-s2",
            &provenance(CandidateStrategy::StatsWeight, new_value),
        )
    }

    #[test]
    fn matches_within_absolute_tolerance_across_zero() {
        let tolerance = Tolerance::default();
        // A delta straddling zero has no relative scale to lean on, so only the
        // absolute bound can hold the two proposals together.
        let a = fingerprint(Some(-4e-10));
        let b = fingerprint(Some(4e-10));
        assert!(a.matches(&b, tolerance));
        assert!(fingerprint(Some(0.0)).matches(&fingerprint(Some(-1e-10)), tolerance));
    }

    #[test]
    fn matches_within_relative_tolerance_large_values() {
        let tolerance = Tolerance::default();
        // 1e-7 apart absolutely — far outside `abs` — but agreeing to nine
        // significant figures, so the relative bound must match them.
        let a = fingerprint(Some(1234.5678));
        let b = fingerprint(Some(1234.5678 + 1e-7));
        assert!(a.matches(&b, tolerance));
    }

    #[test]
    fn rejects_outside_both_tolerances() {
        let tolerance = Tolerance::default();
        assert!(!fingerprint(Some(0.5)).matches(&fingerprint(Some(0.5001)), tolerance));
        assert!(!fingerprint(Some(1e-3)).matches(&fingerprint(Some(2e-3)), tolerance));
    }

    #[test]
    fn none_some_never_match() {
        let tolerance = Tolerance::default();
        assert!(!fingerprint(None).matches(&fingerprint(Some(0.0)), tolerance));
        assert!(!fingerprint(Some(0.0)).matches(&fingerprint(None), tolerance));
        // Two structural mutations with no scalar match on the discrete part.
        assert!(fingerprint(None).matches(&fingerprint(None), tolerance));
    }

    #[test]
    fn non_finite_never_matches_even_itself() {
        let tolerance = Tolerance::default();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let fp = fingerprint(Some(bad));
            assert!(
                !fp.matches(&fp, tolerance),
                "{bad} must not match itself — it is not evidence about the mutation"
            );
            assert!(!fp.is_cacheable(), "{bad} must never be cached");
        }
        assert!(fingerprint(Some(0.25)).is_cacheable());
        assert!(fingerprint(None).is_cacheable());
    }

    #[test]
    fn different_discrete_incumbent_id_never_matches() {
        let tolerance = Tolerance::default();
        let a = fingerprint(Some(1.5));
        let mut b = a.clone();
        b.bucket.incumbent_id = "in1-out1-n3-s4".into();
        assert!(!a.matches(&b, tolerance));
    }

    #[test]
    fn different_discrete_strategy_never_matches() {
        let tolerance = Tolerance::default();
        let a = fingerprint(Some(1.5));
        let mut b = a.clone();
        b.bucket.strategy = CandidateStrategy::Backprop;
        assert!(!a.matches(&b, tolerance));
    }

    #[test]
    fn different_discrete_focus_neuron_never_matches() {
        let tolerance = Tolerance::default();
        let a = fingerprint(Some(1.5));
        let mut b = a.clone();
        b.bucket.focus_neuron = "h1".into();
        assert!(!a.matches(&b, tolerance));
    }

    #[test]
    fn different_discrete_mutation_never_matches() {
        let tolerance = Tolerance::default();
        let a = fingerprint(Some(1.5));
        let mut b = a.clone();
        b.bucket.mutation = "bias o1 0.0 -> 0.5".into();
        assert!(!a.matches(&b, tolerance));
    }

    #[test]
    fn non_matching_bucket_performs_no_tolerance_comparisons() {
        let comparisons = Cell::new(0usize);
        let count = |a: f64, b: f64| {
            comparisons.set(comparisons.get() + 1);
            Tolerance::default().matches(a, b)
        };
        let a = fingerprint(Some(1.5));
        let mut b = a.clone();
        b.bucket.mutation = "a different knob".into();

        assert!(!a.matches_with(&b, count));
        assert_eq!(
            comparisons.get(),
            0,
            "a different discrete part must be rejected on the hash key alone"
        );
        assert!(a.matches_with(&a.clone(), count));
        assert_eq!(comparisons.get(), 1, "a matching bucket compares once");
    }

    #[test]
    fn fingerprint_serde_roundtrip() {
        let fp = fingerprint(Some(-1.25));
        let encoded = serde_json::to_string(&fp).unwrap();
        assert!(
            encoded.contains("\"incumbentId\"")
                && encoded.contains("\"focusNeuron\"")
                && encoded.contains("\"newValue\""),
            "on-disk field names are part of the snapshot format: {encoded}"
        );
        let decoded: CandidateFingerprint = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, fp);

        // A structural fingerprint omits the scalar and still round-trips.
        let structural = fingerprint(None);
        let encoded = serde_json::to_string(&structural).unwrap();
        assert!(!encoded.contains("newValue"));
        let decoded: CandidateFingerprint = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, structural);
    }

    #[test]
    fn a_negative_or_non_finite_bound_narrows_rather_than_widens() {
        let tolerance = Tolerance::new(-1.0, f64::NAN);
        assert_eq!(tolerance.abs, 0.0);
        assert_eq!(tolerance.rel, 0.0);
        assert!(tolerance.matches(1.5, 1.5), "exact equality still matches");
        assert!(!tolerance.matches(1.5, 1.5 + 1e-15));
    }
}
