//! Bounded in-memory store of known-failed candidates (issue #89).
//!
//! The store is bounded by **both** a size cap and an age bound, because the
//! feature's hard constraint is that it must not cause memory or disk issues:
//! "bounded" needs a number, not an adjective. See
//! [`FAILED_CACHE_BYTES_PER_ENTRY`] for the worst-case footprint at the default
//! cap.
//!
//! Entries survive an incumbent acceptance. The fingerprint carries the
//! incumbent identity, so a new incumbent's entries are naturally distinct —
//! the requirement is only that acceptance does not *clear* what was learned.

use super::fingerprint::{CandidateFingerprint, FingerprintBucketKey, Tolerance};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Default cap on live entries.
///
/// At [`FAILED_CACHE_BYTES_PER_ENTRY`] this is a worst case of ~25 MiB
/// resident, which is affordable beside a production creature and a scorer
/// batch. Treat it as provisional until the benchmark sub-issue of #69 settles
/// it. Use [`FailedCandidateCache::worst_case_bytes`] to state the bound for a
/// configured cap.
pub const DEFAULT_FAILED_CACHE_MAX_ENTRIES: usize = 50_000;

/// Default age bound in seconds (7 days).
///
/// A week spans the usual cadence of consecutive runs against one creature, so
/// a rejection stays useful across a weekend without pinning results from a
/// creature that has since moved on.
pub const DEFAULT_FAILED_CACHE_MAX_AGE_SECONDS: u64 = 7 * 24 * 3600;

/// Budgeted worst-case bytes for one live entry.
///
/// An entry holds a fingerprint (incumbent id, focus UUID and a human-readable
/// mutation description, each a heap `String`) plus the scalar and timestamps,
/// and the bucket key is held twice: once as the map key, once in the
/// insertion-order queue. `512` is the budget those strings are allowed to
/// occupy on average; the mutation descriptions Lamarck emits are well inside
/// it.
pub const FAILED_CACHE_BYTES_PER_ENTRY: usize = 512;

/// Inserts between amortised age sweeps.
///
/// Lookups never sweep — the age bound is enforced per entry at lookup time —
/// so the only scan is this one, costing `O(entries / 256)` amortised per
/// insert.
const SWEEP_EVERY_INSERTS: u64 = 256;

/// One cached rejection, as persisted by the snapshot format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FailedCacheEntry {
    /// Identity of the candidate that was scored and did not improve.
    pub fingerprint: CandidateFingerprint,
    /// Unix timestamp the entry was recorded at, which fixes its age.
    pub inserted_unix: u64,
    /// Whether this candidate reached the full-corpus promote phase before it
    /// failed (issue #92).
    ///
    /// Skipping it therefore avoids promote time as well as screen time, and
    /// the economics ledger is allowed to claim both. Defaulted to `false` so a
    /// snapshot written before the flag existed reads as screen-cost only,
    /// which under-claims rather than over-claims.
    #[serde(default)]
    pub promoted: bool,
}

/// Counters describing the cache's size and maintenance cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FailedCacheStats {
    /// Entries currently held (live or awaiting the next sweep).
    pub entries: usize,
    /// Distinct discrete buckets currently held.
    pub buckets: usize,
    /// Entries accepted over the cache's lifetime.
    pub inserted: u64,
    /// Entries dropped to stay under the size cap.
    pub evicted: u64,
    /// Entries dropped by the age bound.
    pub expired: u64,
    /// Worst-case resident bytes at the configured cap.
    pub worst_case_bytes: usize,
    /// Milliseconds spent in the most recent age sweep.
    pub last_maintenance_ms: u128,
}

/// What a lookup found (issue #92).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheHit {
    /// Whether the matched entry had reached the promote phase before failing.
    pub promoted: bool,
}

/// One entry as held inside its discrete bucket.
#[derive(Debug, Clone)]
struct BucketEntry {
    seq: u64,
    new_value: Option<f64>,
    inserted_unix: u64,
    promoted: bool,
}

/// One entry as held in insertion order, for oldest-first eviction.
#[derive(Debug, Clone)]
struct OrderEntry {
    seq: u64,
    key: FingerprintBucketKey,
    inserted_unix: u64,
}

/// Bounded store of candidates that were scored and did not improve.
///
/// Lookup is `O(bucket)`: the discrete part of the fingerprint is the map key,
/// so only candidates that turned the same knob on the same incumbent are ever
/// tolerance-compared.
#[derive(Debug)]
pub struct FailedCandidateCache {
    tolerance: Tolerance,
    max_entries: usize,
    max_age_seconds: u64,
    buckets: HashMap<FingerprintBucketKey, Vec<BucketEntry>>,
    order: VecDeque<OrderEntry>,
    next_seq: u64,
    inserts_since_sweep: u64,
    inserted: u64,
    evicted: u64,
    expired: u64,
    last_maintenance_ms: u128,
    maintenance_micros_owed: u128,
}

impl FailedCandidateCache {
    /// An empty cache with the supplied bounds.
    ///
    /// `max_entries` of `0` disables the cache entirely: nothing is stored and
    /// every lookup misses. `max_age_seconds` of `0` disables the *age bound*
    /// only — entries then live until the size cap evicts them.
    pub fn new(tolerance: Tolerance, max_entries: usize, max_age_seconds: u64) -> Self {
        Self {
            tolerance,
            max_entries,
            max_age_seconds,
            buckets: HashMap::new(),
            order: VecDeque::new(),
            next_seq: 0,
            inserts_since_sweep: 0,
            inserted: 0,
            evicted: 0,
            expired: 0,
            last_maintenance_ms: 0,
            maintenance_micros_owed: 0,
        }
    }

    /// Near-duplicate bounds this cache matches with.
    pub fn tolerance(&self) -> Tolerance {
        self.tolerance
    }

    /// Configured size cap.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Configured age bound in seconds (`0` = no age bound).
    pub fn max_age_seconds(&self) -> u64 {
        self.max_age_seconds
    }

    /// Entries currently held.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Worst-case resident bytes at the configured cap.
    pub fn worst_case_bytes(&self) -> usize {
        self.max_entries
            .saturating_mul(FAILED_CACHE_BYTES_PER_ENTRY)
    }

    /// Resident bytes the entries actually held occupy, at the budgeted
    /// [`FAILED_CACHE_BYTES_PER_ENTRY`] each (issue #92).
    pub fn resident_bytes(&self) -> usize {
        self.len().saturating_mul(FAILED_CACHE_BYTES_PER_ENTRY)
    }

    /// Evict oldest-first until at most `max_entries` remain; returns how many
    /// were dropped.
    ///
    /// This is how the byte ceiling is enforced: the caller converts its byte
    /// bound into an entry count and evicts harder rather than letting the
    /// cache grow (issue #92). The size cap itself is untouched.
    pub fn trim_to(&mut self, max_entries: usize) -> usize {
        let mut dropped = 0usize;
        while self.order.len() > max_entries {
            if !self.evict_oldest() {
                break;
            }
            dropped += 1;
        }
        dropped
    }

    /// Milliseconds spent in the most recent age sweep.
    pub fn last_maintenance_ms(&self) -> u128 {
        self.last_maintenance_ms
    }

    /// Microseconds of sweep maintenance since this was last called, and reset.
    ///
    /// The economics ledger charges maintenance once, so the counter has to be
    /// drained rather than read: [`Self::last_maintenance_ms`] is the *most
    /// recent* sweep and would be charged again by every experiment that
    /// happened not to sweep (issue #92).
    pub fn take_maintenance_micros(&mut self) -> u128 {
        std::mem::take(&mut self.maintenance_micros_owed)
    }

    /// Size and maintenance counters for instrumentation.
    pub fn stats(&self) -> FailedCacheStats {
        FailedCacheStats {
            entries: self.len(),
            buckets: self.buckets.len(),
            inserted: self.inserted,
            evicted: self.evicted,
            expired: self.expired,
            worst_case_bytes: self.worst_case_bytes(),
            last_maintenance_ms: self.last_maintenance_ms,
        }
    }

    /// Record a candidate that was scored and did not improve the incumbent.
    ///
    /// `inserted_unix` fixes the entry's age. Returns whether it was stored: a
    /// non-cacheable fingerprint (non-finite scalar), a disabled cache, an
    /// already-expired timestamp or an existing live match are all declined.
    pub fn insert(&mut self, fingerprint: CandidateFingerprint, inserted_unix: u64) -> bool {
        self.insert_aged(fingerprint, inserted_unix, inserted_unix, false)
    }

    /// [`Self::insert`] for a candidate that failed at the promote phase.
    ///
    /// A repeat insert is still declined, so a candidate first recorded as a
    /// screen failure keeps `promoted = false` even if a later experiment
    /// promotes it: the economics ledger then under-claims the saving rather
    /// than over-claiming it (issue #92).
    pub fn insert_promoted(
        &mut self,
        fingerprint: CandidateFingerprint,
        inserted_unix: u64,
    ) -> bool {
        self.insert_aged(fingerprint, inserted_unix, inserted_unix, true)
    }

    /// [`Self::insert`] for an entry recorded in the past.
    ///
    /// Journal replay inserts entries with their original timestamps, so the
    /// age bound has to be applied against the replay's `now` rather than the
    /// entry's own timestamp. Entries already past the bound are dropped as
    /// they arrive, so a long journal never has to be loaded and trimmed
    /// afterwards.
    pub fn insert_aged(
        &mut self,
        fingerprint: CandidateFingerprint,
        inserted_unix: u64,
        now: u64,
        promoted: bool,
    ) -> bool {
        if self.max_entries == 0 || !fingerprint.is_cacheable() {
            return false;
        }
        if !self.is_live(inserted_unix, now) {
            self.expired += 1;
            return false;
        }
        if self.hit(&fingerprint, now).is_some() {
            // A repeat rejection is already known; re-recording it would only
            // consume an entry, and must not refresh the original's age.
            return false;
        }

        let seq = self.next_seq;
        self.next_seq += 1;
        self.order.push_back(OrderEntry {
            seq,
            key: fingerprint.bucket.clone(),
            inserted_unix,
        });
        self.buckets
            .entry(fingerprint.bucket)
            .or_default()
            .push(BucketEntry {
                seq,
                new_value: fingerprint.new_value,
                inserted_unix,
                promoted,
            });
        self.inserted += 1;

        self.inserts_since_sweep += 1;
        if self.inserts_since_sweep >= SWEEP_EVERY_INSERTS {
            self.sweep_expired(now);
        }
        while self.order.len() > self.max_entries {
            if !self.evict_oldest() {
                break;
            }
        }
        true
    }

    /// Whether a live (non-expired) entry matches `fingerprint` within tolerance.
    pub fn contains(&self, fingerprint: &CandidateFingerprint, now: u64) -> bool {
        self.hit(fingerprint, now).is_some()
    }

    /// The live entry matching `fingerprint` within tolerance, when there is one.
    ///
    /// The hit carries whether that entry had been promoted, which is what
    /// lets the economics ledger claim promote-phase savings only for a skip
    /// that genuinely avoided a full-corpus score (issue #92).
    pub fn hit(&self, fingerprint: &CandidateFingerprint, now: u64) -> Option<CacheHit> {
        if self.max_entries == 0 || !fingerprint.is_cacheable() {
            return None;
        }
        let bucket = self.buckets.get(&fingerprint.bucket)?;
        bucket
            .iter()
            .find(|entry| {
                self.is_live(entry.inserted_unix, now)
                    && match (entry.new_value, fingerprint.new_value) {
                        (None, None) => true,
                        (Some(a), Some(b)) => self.tolerance.matches(a, b),
                        (None, Some(_)) | (Some(_), None) => false,
                    }
            })
            .map(|entry| CacheHit {
                promoted: entry.promoted,
            })
    }

    /// Explicit-name alias for [`Self::contains`], which only ever hits live
    /// entries.
    pub fn contains_live(&self, fingerprint: &CandidateFingerprint, now: u64) -> bool {
        self.contains(fingerprint, now)
    }

    /// Every held entry in insertion order, for snapshotting to disk.
    ///
    /// Entries past the age bound may still be present until the next sweep;
    /// the snapshot writer applies the bound again on load, so a stale entry
    /// can never come back as a hit.
    pub fn snapshot_entries(&self) -> Vec<FailedCacheEntry> {
        self.order
            .iter()
            .filter_map(|slot| {
                let bucket = self.buckets.get(&slot.key)?;
                let entry = bucket.iter().find(|e| e.seq == slot.seq)?;
                Some(FailedCacheEntry {
                    fingerprint: CandidateFingerprint {
                        bucket: slot.key.clone(),
                        new_value: entry.new_value,
                    },
                    inserted_unix: entry.inserted_unix,
                    promoted: entry.promoted,
                })
            })
            .collect()
    }

    /// Drop every entry past the age bound, recording the cost.
    ///
    /// Called automatically every 256 inserts; exposed so a
    /// caller that wants a precise live count (a journal field, say) can force
    /// one.
    pub fn sweep_expired(&mut self, now: u64) -> usize {
        self.inserts_since_sweep = 0;
        if self.max_age_seconds == 0 {
            self.last_maintenance_ms = 0;
            return 0;
        }
        let started = Instant::now();
        let mut dropped = 0usize;
        let mut kept = VecDeque::with_capacity(self.order.len());
        for slot in std::mem::take(&mut self.order) {
            if self.is_live(slot.inserted_unix, now) {
                kept.push_back(slot);
            } else {
                remove_from_bucket(&mut self.buckets, &slot.key, slot.seq);
                dropped += 1;
            }
        }
        self.order = kept;
        self.expired += dropped as u64;
        let elapsed = started.elapsed();
        self.last_maintenance_ms = elapsed.as_millis();
        self.maintenance_micros_owed += elapsed.as_micros();
        dropped
    }

    fn is_live(&self, inserted_unix: u64, now: u64) -> bool {
        if self.max_age_seconds == 0 {
            return true;
        }
        // A clock that moved backwards must not silently expire the cache.
        now.saturating_sub(inserted_unix) < self.max_age_seconds
    }

    fn evict_oldest(&mut self) -> bool {
        let Some(slot) = self.order.pop_front() else {
            return false;
        };
        remove_from_bucket(&mut self.buckets, &slot.key, slot.seq);
        self.evicted += 1;
        true
    }
}

fn remove_from_bucket(
    buckets: &mut HashMap<FingerprintBucketKey, Vec<BucketEntry>>,
    key: &FingerprintBucketKey,
    seq: u64,
) {
    let Some(bucket) = buckets.get_mut(key) else {
        return;
    };
    bucket.retain(|entry| entry.seq != seq);
    if bucket.is_empty() {
        buckets.remove(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::{CandidateProvenance, CandidateStrategy};

    fn fingerprint(
        incumbent: &str,
        mutation: &str,
        new_value: Option<f64>,
    ) -> CandidateFingerprint {
        CandidateFingerprint::from_provenance(
            incumbent,
            &CandidateProvenance {
                strategy: CandidateStrategy::StatsWeight,
                focus_neuron: "o1".into(),
                mutation: mutation.into(),
                old_value: Some(0.0),
                new_value,
            },
        )
    }

    fn new_cache(max_entries: usize, max_age_seconds: u64) -> FailedCandidateCache {
        FailedCandidateCache::new(Tolerance::default(), max_entries, max_age_seconds)
    }

    #[test]
    fn evicts_oldest_first_at_cap() {
        let mut cache = new_cache(3, 0);
        for i in 0..3 {
            assert!(cache.insert(
                fingerprint("inc", &format!("knob-{i}"), Some(i as f64)),
                100 + i
            ));
        }
        assert_eq!(cache.len(), 3);
        cache.insert(fingerprint("inc", "knob-3", Some(3.0)), 103);

        assert_eq!(cache.len(), 3, "the cap holds");
        assert!(
            !cache.contains(&fingerprint("inc", "knob-0", Some(0.0)), 103),
            "the oldest entry is the one evicted"
        );
        for i in 1..4 {
            assert!(
                cache.contains(
                    &fingerprint("inc", &format!("knob-{i}"), Some(i as f64)),
                    103
                ),
                "knob-{i} must survive"
            );
        }
        assert_eq!(cache.stats().evicted, 1);
    }

    #[test]
    fn inserting_past_cap_never_exceeds_max_entries() {
        let mut cache = new_cache(64, 0);
        for i in 0..5_000u64 {
            cache.insert(
                fingerprint("inc", &format!("knob-{i}"), Some(i as f64)),
                1_000 + i,
            );
            assert!(
                cache.len() <= 64,
                "cap breached at insert {i}: {} entries",
                cache.len()
            );
        }
        assert_eq!(cache.len(), 64);
        assert_eq!(
            cache.worst_case_bytes(),
            64 * FAILED_CACHE_BYTES_PER_ENTRY,
            "the memory bound is a stated number, not an adjective"
        );
    }

    #[test]
    fn expired_entry_is_not_a_hit() {
        let mut cache = new_cache(16, 60);
        let fp = fingerprint("inc", "knob", Some(1.0));
        cache.insert(fp.clone(), 1_000);

        assert!(cache.contains(&fp, 1_059), "inside the age bound");
        assert!(
            !cache.contains(&fp, 1_060),
            "at the bound the entry is dead"
        );
        assert!(!cache.contains(&fp, 5_000));

        // The sweep reclaims it rather than leaving it resident forever.
        assert_eq!(cache.sweep_expired(5_000), 1);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats().expired, 1);
    }

    #[test]
    fn hit_does_not_refresh_insertion_age() {
        // Entries expire on insertion age, not on last access: a candidate the
        // generator keeps re-proposing must not pin itself in the cache.
        let mut cache = new_cache(16, 100);
        let fp = fingerprint("inc", "knob", Some(1.0));
        cache.insert(fp.clone(), 1_000);

        assert!(cache.contains(&fp, 1_090), "a hit at t=1090");
        assert!(
            !cache.contains(&fp, 1_100),
            "the hit must not have moved the insertion timestamp"
        );

        // A repeat insert is also declined rather than refreshing the age.
        let mut cache = new_cache(16, 100);
        cache.insert(fp.clone(), 1_000);
        assert!(!cache.insert(fp.clone(), 1_090), "already known");
        assert!(!cache.contains(&fp, 1_100));
    }

    #[test]
    fn survives_incumbent_acceptance() {
        let mut cache = new_cache(16, 0);
        let before = fingerprint("in1-out1-n2-s2", "knob", Some(1.0));
        cache.insert(before.clone(), 1_000);

        // Simulate an acceptance: the incumbent changes identity, and the new
        // incumbent's candidates are naturally distinct fingerprints.
        let after = fingerprint("in1-out1-n3-s4", "knob", Some(1.0));
        assert!(
            !cache.contains(&after, 1_001),
            "a new incumbent is a new key"
        );
        assert!(
            cache.contains(&before, 1_001),
            "acceptance must not clear what the run already learned"
        );
        cache.insert(after.clone(), 1_001);
        assert!(cache.contains(&before, 1_002) && cache.contains(&after, 1_002));
    }

    #[test]
    fn zero_cap_and_zero_age_disable_cleanly() {
        // Cap 0 disables the cache: nothing is stored and nothing ever hits.
        let mut disabled = new_cache(0, 3_600);
        let fp = fingerprint("inc", "knob", Some(1.0));
        assert!(!disabled.insert(fp.clone(), 1_000));
        assert!(!disabled.contains(&fp, 1_000));
        assert!(disabled.is_empty());
        assert_eq!(disabled.worst_case_bytes(), 0);
        assert_eq!(disabled.sweep_expired(1_000), 0);

        // Age 0 disables the age bound; the size cap still applies.
        let mut ageless = new_cache(2, 0);
        ageless.insert(fp.clone(), 1_000);
        assert!(
            ageless.contains(&fp, u64::MAX),
            "no age bound means no expiry"
        );
        assert_eq!(ageless.sweep_expired(u64::MAX), 0);
        assert!(ageless.contains(&fp, u64::MAX));
    }

    #[test]
    fn a_near_duplicate_within_tolerance_is_a_hit() {
        let mut cache = new_cache(16, 0);
        cache.insert(fingerprint("inc", "knob", Some(1.0)), 1_000);
        assert!(cache.contains(&fingerprint("inc", "knob", Some(1.0 + 1e-9)), 1_000));
        assert!(!cache.contains(&fingerprint("inc", "knob", Some(1.5)), 1_000));
    }

    #[test]
    fn a_non_finite_candidate_is_never_stored_or_matched() {
        let mut cache = new_cache(16, 0);
        let bad = fingerprint("inc", "knob", Some(f64::NAN));
        assert!(!cache.insert(bad.clone(), 1_000));
        assert!(!cache.contains(&bad, 1_000));
        assert!(cache.is_empty());
    }

    #[test]
    fn the_amortised_sweep_runs_without_scanning_on_lookup() {
        let mut cache = new_cache(10_000, 10);
        for i in 0..SWEEP_EVERY_INSERTS {
            cache.insert(
                fingerprint("inc", &format!("old-{i}"), Some(i as f64)),
                1_000,
            );
        }
        // Everything above is long dead by now; the next sweep reclaims it.
        for i in 0..SWEEP_EVERY_INSERTS {
            cache.insert(
                fingerprint("inc", &format!("new-{i}"), Some(i as f64)),
                10_000,
            );
        }
        assert!(
            cache.len() <= SWEEP_EVERY_INSERTS as usize,
            "the amortised sweep dropped the expired generation: {} entries",
            cache.len()
        );
        assert!(cache.stats().expired > 0);
    }

    /// Issue #92: promote-phase savings may only be claimed for a skip that
    /// genuinely avoided a full-corpus score, so the flag has to survive both
    /// the lookup and the snapshot.
    #[test]
    fn a_promoted_failure_is_reported_by_the_hit_and_snapshotted() {
        let mut cache = new_cache(16, 0);
        let screened = fingerprint("inc", "screened", Some(1.0));
        let promoted = fingerprint("inc", "promoted", Some(2.0));
        cache.insert(screened.clone(), 1_000);
        cache.insert_promoted(promoted.clone(), 1_001);

        assert_eq!(
            cache.hit(&screened, 1_002),
            Some(CacheHit { promoted: false })
        );
        assert_eq!(
            cache.hit(&promoted, 1_002),
            Some(CacheHit { promoted: true })
        );
        assert_eq!(
            cache.hit(&fingerprint("inc", "unknown", Some(3.0)), 1_002),
            None
        );

        // A later promote of an already-known screen failure is declined, so the
        // ledger keeps under-claiming rather than over-claiming.
        assert!(!cache.insert_promoted(screened.clone(), 1_003));
        assert_eq!(
            cache.hit(&screened, 1_004),
            Some(CacheHit { promoted: false })
        );

        let entries = cache.snapshot_entries();
        let decoded: Vec<FailedCacheEntry> =
            serde_json::from_str(&serde_json::to_string(&entries).unwrap()).unwrap();
        assert_eq!(decoded, entries);
        assert!(!decoded[0].promoted && decoded[1].promoted);
    }

    #[test]
    fn snapshot_entries_round_trip_insertion_order() {
        let mut cache = new_cache(16, 0);
        for i in 0..3u64 {
            cache.insert(
                fingerprint("inc", &format!("knob-{i}"), Some(i as f64)),
                100 + i,
            );
        }
        let entries = cache.snapshot_entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].inserted_unix, 100);
        assert_eq!(entries[2].inserted_unix, 102);
        assert_eq!(entries[1].fingerprint.bucket.mutation, "knob-1");

        let encoded = serde_json::to_string(&entries).unwrap();
        let decoded: Vec<FailedCacheEntry> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, entries);
    }
}
