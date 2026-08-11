//! Batch filtering and backfill against the failed-candidate cache (issue #91).
//!
//! This is where the cache actually saves scorer time: known-failed candidates
//! are dropped before the batch is written, and the batch is topped back up so
//! the scorer still runs at full width. A batch that shrinks from 100 to 60
//! saves nothing — the scorer batch is the unit of cost, so a short batch just
//! wastes the box.
//!
//! # Reproducibility
//!
//! Backfill **consumes extra RNG draws**: each regeneration round advances the
//! same generator stream the run's candidates come from. Two runs with the same
//! seed therefore only produce the same candidate stream when they have the
//! same cache contents. With `--failed-cache` off nothing here runs and no
//! extra draw is taken, so the #71 replay contract for existing runs is
//! untouched.

use super::rebuild::candidate_index_from_stem;
use super::store::FailedCandidateCache;
use crate::cancel::CancelToken;
use crate::candidates::Candidate;
use crate::failed_cache::fingerprint::CandidateFingerprint;
use std::collections::BTreeSet;
use std::time::Instant;

/// Multiple of the target batch size that bounds total generation attempts.
///
/// A generator that can only re-propose known-failed candidates must not spin:
/// the run proceeds with a short batch (loudly logged) instead.
pub const BACKFILL_PROPOSAL_BUDGET_MULTIPLE: usize = 2;

/// Outcome of filtering and backfilling one candidate batch.
#[derive(Debug)]
pub struct FilteredBatch {
    /// The batch to score: cache misses, topped up where possible.
    pub candidates: Vec<Candidate>,
    /// Proposals rejected by a cache hit.
    pub skipped: usize,
    /// Replacement proposals accepted into the batch.
    pub backfilled: usize,
    /// Proposals examined in total, including the original batch.
    pub proposals: usize,
    /// Whether the batch is still short of the target.
    pub short_batch: bool,
    /// Whether the backfill loop stopped because cancellation was requested.
    pub cancelled: bool,
    /// Milliseconds spent in cache lookups and backfill.
    pub lookup_ms: u128,
}

/// Drop known-failed candidates from `batch` and regenerate to refill it.
///
/// `regenerate` is called with the shortfall and may return fewer candidates
/// than asked for (or none, which ends the backfill). Proposals are
/// deduplicated against both the cache and the candidates already accepted into
/// this batch, and the total number examined is capped at
/// [`BACKFILL_PROPOSAL_BUDGET_MULTIPLE`] × `target`.
///
/// The loop polls `cancel` before every regeneration round so a long retry
/// cannot swallow a stop signal (issue #72).
pub fn filter_and_backfill(
    cache: &FailedCandidateCache,
    incumbent_id: &str,
    batch: Vec<Candidate>,
    target: usize,
    now: u64,
    cancel: &CancelToken,
    mut regenerate: impl FnMut(usize) -> Vec<Candidate>,
) -> FilteredBatch {
    let started = Instant::now();
    let mut kept: Vec<Candidate> = Vec::with_capacity(batch.len());
    let mut accepted: Vec<CandidateFingerprint> = Vec::with_capacity(batch.len());
    let mut skipped = 0usize;
    let mut proposals = 0usize;
    let mut backfilled = 0usize;
    let mut cancelled = false;

    let tolerance = cache.tolerance();
    let consider = |candidate: Candidate,
                    kept: &mut Vec<Candidate>,
                    accepted: &mut Vec<CandidateFingerprint>|
     -> bool {
        let fingerprint =
            CandidateFingerprint::from_provenance(incumbent_id, &candidate.provenance);
        if cache.contains(&fingerprint, now) {
            return false;
        }
        if accepted
            .iter()
            .any(|seen| seen.matches(&fingerprint, tolerance))
        {
            return false;
        }
        accepted.push(fingerprint);
        kept.push(candidate);
        true
    };

    for candidate in batch {
        proposals += 1;
        if !consider(candidate, &mut kept, &mut accepted) {
            skipped += 1;
        }
    }

    let budget = target.saturating_mul(BACKFILL_PROPOSAL_BUDGET_MULTIPLE);
    while kept.len() < target && proposals < budget {
        if cancel.is_cancelled() {
            cancelled = true;
            break;
        }
        let wanted = (target - kept.len()).min(budget - proposals);
        let replacements = regenerate(wanted);
        if replacements.is_empty() {
            break;
        }
        let mut fresh = 0usize;
        for candidate in replacements {
            if proposals >= budget || kept.len() >= target {
                break;
            }
            proposals += 1;
            if consider(candidate, &mut kept, &mut accepted) {
                backfilled += 1;
                fresh += 1;
            } else {
                skipped += 1;
            }
        }
        if fresh == 0 {
            // The generator can only re-propose candidates we already have or
            // already know fail; another round would spin.
            break;
        }
    }

    FilteredBatch {
        short_batch: kept.len() < target,
        candidates: kept,
        skipped,
        backfilled,
        proposals,
        cancelled,
        lookup_ms: started.elapsed().as_millis(),
    }
}

/// Indices into the candidate batch for every stem the scorer reported on.
///
/// Screen-phase and full-corpus stems can both be passed; `baseline` and merged
/// combo stems address no single candidate and are ignored.
pub fn scored_candidate_indices<'a, I>(stems: I) -> BTreeSet<usize>
where
    I: IntoIterator<Item = &'a str>,
{
    stems
        .into_iter()
        .filter_map(candidate_index_from_stem)
        .collect()
}

/// Record the named candidates as known-failed; returns how many were stored.
///
/// Callers must pass the **pre-acceptance** incumbent identity: the candidates
/// were proposed against the creature that was incumbent when the batch was
/// generated, not against the winner that replaced it.
pub fn insert_failures<I>(
    cache: &mut FailedCandidateCache,
    incumbent_id: &str,
    candidates: &[Candidate],
    failed: I,
    now: u64,
) -> usize
where
    I: IntoIterator<Item = usize>,
{
    let mut stored = 0usize;
    for index in failed {
        let Some(candidate) = candidates.get(index) else {
            continue;
        };
        let fingerprint =
            CandidateFingerprint::from_provenance(incumbent_id, &candidate.provenance);
        if cache.insert(fingerprint, now) {
            stored += 1;
        }
    }
    stored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::{CandidateProvenance, CandidateStrategy};
    use crate::failed_cache::fingerprint::Tolerance;
    use neat_core::{CreatureExport, parse_creature_json};

    const INCUMBENT: &str = "in1-out1-n1-s1";

    fn creature() -> CreatureExport {
        parse_creature_json(
            r#"{
              "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
              "neurons":[{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
              "synapses":[{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
            }"#,
        )
        .unwrap()
    }

    fn candidate(mutation: &str, new_value: f64) -> Candidate {
        Candidate {
            creature: creature(),
            provenance: CandidateProvenance {
                strategy: CandidateStrategy::StatsWeight,
                focus_neuron: "o1".into(),
                mutation: mutation.into(),
                old_value: Some(0.0),
                new_value: Some(new_value),
            },
        }
    }

    fn fingerprint_of(candidate: &Candidate) -> CandidateFingerprint {
        CandidateFingerprint::from_provenance(INCUMBENT, &candidate.provenance)
    }

    fn cache() -> FailedCandidateCache {
        FailedCandidateCache::new(Tolerance::default(), 1_000, 0)
    }

    fn no_regeneration(_: usize) -> Vec<Candidate> {
        Vec::new()
    }

    #[test]
    fn known_failed_candidates_are_skipped() {
        let mut cache = cache();
        let doomed = candidate("knob-a", 1.0);
        cache.insert(fingerprint_of(&doomed), 1_000);

        let batch = vec![doomed, candidate("knob-b", 2.0)];
        let filtered = filter_and_backfill(
            &cache,
            INCUMBENT,
            batch,
            2,
            1_000,
            &CancelToken::new(),
            no_regeneration,
        );

        assert_eq!(filtered.skipped, 1);
        assert_eq!(filtered.candidates.len(), 1);
        assert_eq!(filtered.candidates[0].provenance.mutation, "knob-b");
        assert!(filtered.short_batch, "nothing could replace the skip");
    }

    #[test]
    fn a_near_duplicate_within_tolerance_is_skipped() {
        let mut cache = cache();
        cache.insert(fingerprint_of(&candidate("knob-a", 1.0)), 1_000);

        let filtered = filter_and_backfill(
            &cache,
            INCUMBENT,
            vec![candidate("knob-a", 1.0 + 1e-10)],
            1,
            1_000,
            &CancelToken::new(),
            no_regeneration,
        );
        assert_eq!(filtered.skipped, 1);
        assert!(filtered.candidates.is_empty());
    }

    #[test]
    fn a_candidate_against_a_different_incumbent_is_not_skipped() {
        let mut cache = cache();
        cache.insert(fingerprint_of(&candidate("knob-a", 1.0)), 1_000);

        let filtered = filter_and_backfill(
            &cache,
            "a-different-incumbent",
            vec![candidate("knob-a", 1.0)],
            1,
            1_000,
            &CancelToken::new(),
            no_regeneration,
        );
        assert_eq!(filtered.skipped, 0);
        assert_eq!(filtered.candidates.len(), 1);
    }

    #[test]
    fn backfill_restores_batch_size() {
        let mut cache = cache();
        for i in 0..3 {
            cache.insert(
                fingerprint_of(&candidate(&format!("knob-{i}"), i as f64)),
                1_000,
            );
        }
        let batch: Vec<Candidate> = (0..4)
            .map(|i| candidate(&format!("knob-{i}"), i as f64))
            .collect();

        let mut next = 100;
        let filtered = filter_and_backfill(
            &cache,
            INCUMBENT,
            batch,
            4,
            1_000,
            &CancelToken::new(),
            |wanted| {
                (0..wanted)
                    .map(|_| {
                        next += 1;
                        candidate(&format!("fresh-{next}"), next as f64)
                    })
                    .collect()
            },
        );

        assert_eq!(filtered.skipped, 3);
        assert_eq!(filtered.backfilled, 3);
        assert_eq!(filtered.candidates.len(), 4, "the batch is topped back up");
        assert!(!filtered.short_batch);
    }

    #[test]
    fn backfill_is_bounded_and_reports_a_short_batch() {
        let mut cache = cache();
        cache.insert(fingerprint_of(&candidate("knob-0", 0.0)), 1_000);

        // A generator that can only re-propose the one known-failed candidate.
        let filtered = filter_and_backfill(
            &cache,
            INCUMBENT,
            vec![candidate("knob-0", 0.0), candidate("knob-1", 1.0)],
            2,
            1_000,
            &CancelToken::new(),
            |wanted| (0..wanted).map(|_| candidate("knob-0", 0.0)).collect(),
        );

        assert!(filtered.short_batch, "a degenerate generator must not spin");
        assert_eq!(filtered.candidates.len(), 1);
        assert!(
            filtered.proposals <= 2 * BACKFILL_PROPOSAL_BUDGET_MULTIPLE,
            "the proposal budget bounds the retry: {}",
            filtered.proposals
        );
    }

    #[test]
    fn backfill_deduplicates_against_the_batch_it_is_filling() {
        let cache = cache();
        let filtered = filter_and_backfill(
            &cache,
            INCUMBENT,
            vec![candidate("knob-a", 1.0)],
            2,
            1_000,
            &CancelToken::new(),
            |wanted| (0..wanted).map(|_| candidate("knob-a", 1.0)).collect(),
        );
        assert_eq!(
            filtered.candidates.len(),
            1,
            "a duplicate of a candidate already in the batch is not a top-up"
        );
        assert_eq!(filtered.backfilled, 0);
    }

    #[test]
    fn cancellation_during_backfill_stops_immediately() {
        let mut cache = cache();
        cache.insert(fingerprint_of(&candidate("knob-0", 0.0)), 1_000);
        let cancel = CancelToken::new();
        cancel.cancel();

        let mut regenerations = 0;
        let filtered = filter_and_backfill(
            &cache,
            INCUMBENT,
            vec![candidate("knob-0", 0.0)],
            4,
            1_000,
            &cancel,
            |wanted| {
                regenerations += 1;
                (0..wanted)
                    .map(|i| candidate(&format!("f-{i}"), i as f64))
                    .collect()
            },
        );

        assert!(filtered.cancelled);
        assert_eq!(regenerations, 0, "a cancelled run does not regenerate");
        assert!(filtered.candidates.is_empty());
    }

    #[test]
    fn an_empty_batch_with_no_generator_is_not_an_infinite_loop() {
        let cache = cache();
        let filtered = filter_and_backfill(
            &cache,
            INCUMBENT,
            Vec::new(),
            8,
            1_000,
            &CancelToken::new(),
            no_regeneration,
        );
        assert!(filtered.short_batch);
        assert_eq!(filtered.proposals, 0);
    }

    #[test]
    fn scored_candidate_indices_maps_only_candidate_stems() {
        let indices = scored_candidate_indices([
            "baseline",
            "candidate-000",
            "candidate-003",
            "combo-001-k2",
        ]);
        assert_eq!(indices.into_iter().collect::<Vec<_>>(), vec![0, 3]);
    }

    #[test]
    fn insert_failures_records_the_named_candidates_only() {
        let mut cache = cache();
        let batch: Vec<Candidate> = (0..3)
            .map(|i| candidate(&format!("knob-{i}"), i as f64))
            .collect();

        // An out-of-range index (a stem from a longer earlier batch) is ignored.
        let stored = insert_failures(&mut cache, INCUMBENT, &batch, [0, 2, 99], 1_000);
        assert_eq!(stored, 2);
        assert!(cache.contains(&fingerprint_of(&batch[0]), 1_000));
        assert!(!cache.contains(&fingerprint_of(&batch[1]), 1_000));
        assert!(cache.contains(&fingerprint_of(&batch[2]), 1_000));
    }

    #[test]
    fn a_filtered_batch_reports_its_lookup_cost() {
        let cache = cache();
        let filtered = filter_and_backfill(
            &cache,
            INCUMBENT,
            vec![candidate("knob-a", 1.0)],
            1,
            1_000,
            &CancelToken::new(),
            no_regeneration,
        );
        assert!(filtered.lookup_ms < 60_000);
        assert_eq!(filtered.proposals, 1);
    }
}
