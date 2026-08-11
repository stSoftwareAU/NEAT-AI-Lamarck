//! Startup rebuild and on-disk persistence for the failed-candidate cache
//! (issue #90).
//!
//! The cache survives across runs by being **rebuilt from
//! `experiments.jsonl`**. The journal is already the source of truth for what
//! was scored and rejected, so a rebuilt cache can never disagree with the run
//! history, and a corrupt or missing snapshot is always recoverable.
//!
//! A snapshot ([`FAILED_CACHE_SNAPSHOT_FILE`]) short-circuits the replay on the
//! next startup. It is advisory only: a missing, unparsable, version-mismatched
//! or differently-tuned snapshot falls back to a journal rebuild rather than
//! failing the run.

use super::fingerprint::{CandidateFingerprint, Tolerance};
use super::store::{FailedCacheEntry, FailedCandidateCache};
use crate::run::{ExperimentRecord, JournalLine};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Journal file the cache is rebuilt from.
pub const FAILED_CACHE_JOURNAL_FILE: &str = "experiments.jsonl";

/// Snapshot file written beside the journal in the run's output directory.
pub const FAILED_CACHE_SNAPSHOT_FILE: &str = "failed-candidates.cache.json";

/// Snapshot format version, following the `observations.rs` pattern.
///
/// Bump this whenever the on-disk shape changes; a mismatch falls back to a
/// journal rebuild rather than misreading old bytes.
///
/// `1.1.0` added the per-entry `promoted` flag the economics ledger prices
/// promote-phase savings from (issue #92). A `1.0.0` snapshot is discarded and
/// replayed from the journal, which restores the flag rather than defaulting
/// every entry to screen-cost only.
pub const FAILED_CACHE_SNAPSHOT_FORMAT_VERSION: &str = "1.1.0";

/// Where a loaded cache came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RebuildSource {
    /// Restored from a valid snapshot.
    Snapshot,
    /// Replayed from `experiments.jsonl`.
    Journal,
    /// Neither was available; the cache started empty.
    #[default]
    Empty,
}

impl RebuildSource {
    /// Stable lower-case label for logs.
    pub fn label(self) -> &'static str {
        match self {
            Self::Snapshot => "snapshot",
            Self::Journal => "journal",
            Self::Empty => "empty",
        }
    }
}

/// What a startup rebuild read, kept and cost.
///
/// Rebuild cost is overhead the cache has to earn back, so it is measured and
/// logged rather than assumed negligible: a journal long enough that rebuilding
/// costs more than it saves is a finding for the go/no-go sub-issue of #69.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RebuildReport {
    /// Where the entries came from.
    #[serde(default)]
    pub source: RebuildSource,
    /// Live entries in the cache afterwards.
    pub entries: usize,
    /// Journal lines read.
    pub lines_read: u64,
    /// Journal lines skipped as unparsable or unrecognised.
    pub lines_skipped: u64,
    /// Experiment records that contributed at least one candidate.
    pub experiments_replayed: u64,
    /// Rejections offered to the cache (some are declined as duplicates/aged).
    pub rejections_seen: u64,
    /// Why a snapshot was not used, when one was present but unusable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_rejected: Option<String>,
    /// Wall-clock milliseconds the rebuild took.
    pub elapsed_ms: u128,
}

/// On-disk snapshot of a bounded cache.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FailedCacheSnapshot {
    /// Snapshot format version.
    pub format_version: String,
    /// Absolute tolerance the entries were bucketed under.
    pub tolerance_abs: f64,
    /// Relative tolerance the entries were bucketed under.
    pub tolerance_rel: f64,
    /// Size cap in force when the snapshot was written.
    pub max_entries: usize,
    /// Age bound in force when the snapshot was written.
    pub max_age_seconds: u64,
    /// Unix timestamp the snapshot was written at.
    pub written_unix: u64,
    /// The entries themselves, in insertion order.
    pub entries: Vec<FailedCacheEntry>,
}

/// Candidate index behind a scorer stem, or `None` for anything else.
///
/// The whole rebuild hinges on this mapping, so it is stated in one place:
/// [`crate::candidates::write_candidate_batch`] writes candidate `i` as
/// `candidate-{i:03}.json`, and the scorer keys its results by that stem.
/// `baseline` and the merged `combo-NNN-kM` stems address no single candidate
/// and are therefore not failures anyone can fingerprint.
pub fn candidate_index_from_stem(stem: &str) -> Option<usize> {
    let digits = stem.strip_prefix("candidate-")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Replay `experiments.jsonl` into `cache`, honouring its bounds as it goes.
///
/// Classification per record:
///
/// * `scorer_error` set → nothing. A scorer crash is not evidence about the
///   mutation.
/// * `accepted` → nothing. The record's `incumbent_id` is written *after* the
///   acceptance swap, so its candidates are keyed to the wrong incumbent and
///   replaying them would poison the cache with hits against a creature they
///   were never proposed for.
/// * otherwise → every scored candidate (full-corpus `scores` **and**
///   screen-dropped `screen_scores`) that is not the winner is a failure.
///
/// A line that is not a recognised journal record — an unknown future record
/// kind, or the truncated last line of a killed run — is skipped, never fatal.
pub fn rebuild_from_journal(
    path: &Path,
    cache: &mut FailedCandidateCache,
    now: u64,
) -> RebuildReport {
    let started = Instant::now();
    let mut report = RebuildReport {
        source: RebuildSource::Journal,
        ..RebuildReport::default()
    };
    let Ok(file) = File::open(path) else {
        report.source = RebuildSource::Empty;
        report.entries = cache.len();
        report.elapsed_ms = started.elapsed().as_millis();
        return report;
    };

    for line in BufReader::new(file).lines() {
        let Ok(line) = line else {
            report.lines_skipped += 1;
            continue;
        };
        if line.trim().is_empty() {
            continue;
        }
        report.lines_read += 1;
        let record = match JournalLine::parse(&line) {
            Ok(JournalLine::Experiment(record)) => *record,
            Ok(
                JournalLine::Header(_)
                | JournalLine::GraftReplay(_)
                | JournalLine::CacheStandDown(_),
            ) => continue,
            Err(_) => {
                report.lines_skipped += 1;
                continue;
            }
        };
        let rejections = replay_experiment(&record, cache, now);
        if rejections > 0 {
            report.experiments_replayed += 1;
            report.rejections_seen += rejections;
        }
    }

    report.entries = cache.len();
    report.elapsed_ms = started.elapsed().as_millis();
    report
}

/// Offer one experiment's failures to the cache; returns how many were offered.
fn replay_experiment(record: &ExperimentRecord, cache: &mut FailedCandidateCache, now: u64) -> u64 {
    if record.scorer_error.is_some() || record.accepted {
        return 0;
    }
    let promoted = promoted_candidate_indices(record);
    let mut offered = 0u64;
    for index in failed_candidate_indices(record) {
        let Some(provenance) = record.candidates.get(index) else {
            continue;
        };
        offered += 1;
        cache.insert_aged(
            CandidateFingerprint::from_provenance(&record.incumbent_id, provenance),
            record.timestamp_unix,
            now,
            promoted.contains(&index),
        );
    }
    offered
}

/// Indices of every scored, non-winning candidate in one experiment record.
fn failed_candidate_indices(record: &ExperimentRecord) -> BTreeSet<usize> {
    let screen = record.screen_scores.iter().flat_map(|m| m.keys());
    record
        .scores
        .keys()
        .chain(screen)
        .filter(|stem| record.winner.as_ref() != Some(*stem))
        .filter_map(|stem| candidate_index_from_stem(stem))
        .collect()
}

/// Indices that reached the full-corpus promote phase in one record (#92).
///
/// Only a two-phase experiment has a promote phase to have reached: when the
/// record has no `screen_scores`, its `scores` *are* the first and only scoring
/// phase, and treating them as promoted would let a later run claim a saving
/// that never existed.
fn promoted_candidate_indices(record: &ExperimentRecord) -> BTreeSet<usize> {
    if record.screen_scores.is_none() {
        return BTreeSet::new();
    }
    record
        .scores
        .keys()
        .filter(|stem| record.winner.as_ref() != Some(*stem))
        .filter_map(|stem| candidate_index_from_stem(stem))
        .collect()
}

/// Load the cache for a run: snapshot when usable, journal replay otherwise.
///
/// Never fails. A snapshot that is absent, unreadable, malformed, of a
/// different format version, or written under different tolerance/bound knobs
/// is discarded (with the reason recorded in the report) and the journal is
/// replayed instead — entries bucketed under different matching rules cannot be
/// trusted.
pub fn load_or_rebuild(
    output_dir: &Path,
    tolerance: Tolerance,
    max_entries: usize,
    max_age_seconds: u64,
    now: u64,
) -> (FailedCandidateCache, RebuildReport) {
    let started = Instant::now();
    let mut cache = FailedCandidateCache::new(tolerance, max_entries, max_age_seconds);

    match read_snapshot(
        &output_dir.join(FAILED_CACHE_SNAPSHOT_FILE),
        tolerance,
        max_entries,
        max_age_seconds,
    ) {
        Ok(Some(snapshot)) => {
            for entry in snapshot.entries {
                cache.insert_aged(entry.fingerprint, entry.inserted_unix, now, entry.promoted);
            }
            let report = RebuildReport {
                source: RebuildSource::Snapshot,
                entries: cache.len(),
                elapsed_ms: started.elapsed().as_millis(),
                ..RebuildReport::default()
            };
            (cache, report)
        }
        Ok(None) => {
            let mut report =
                rebuild_from_journal(&output_dir.join(FAILED_CACHE_JOURNAL_FILE), &mut cache, now);
            report.elapsed_ms = started.elapsed().as_millis();
            (cache, report)
        }
        Err(reason) => {
            let mut report =
                rebuild_from_journal(&output_dir.join(FAILED_CACHE_JOURNAL_FILE), &mut cache, now);
            report.snapshot_rejected = Some(reason);
            report.elapsed_ms = started.elapsed().as_millis();
            (cache, report)
        }
    }
}

/// `Ok(None)` when there is no snapshot, `Err(reason)` when there is an
/// unusable one.
fn read_snapshot(
    path: &Path,
    tolerance: Tolerance,
    max_entries: usize,
    max_age_seconds: u64,
) -> Result<Option<FailedCacheSnapshot>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path).map_err(|e| format!("unreadable snapshot: {e}"))?;
    let snapshot: FailedCacheSnapshot =
        serde_json::from_str(&text).map_err(|e| format!("unparsable snapshot: {e}"))?;
    if snapshot.format_version != FAILED_CACHE_SNAPSHOT_FORMAT_VERSION {
        return Err(format!(
            "unsupported snapshot formatVersion {} (want {FAILED_CACHE_SNAPSHOT_FORMAT_VERSION})",
            snapshot.format_version
        ));
    }
    if snapshot.tolerance_abs != tolerance.abs || snapshot.tolerance_rel != tolerance.rel {
        return Err(format!(
            "snapshot tolerance ({}, {}) does not match the configured ({}, {})",
            snapshot.tolerance_abs, snapshot.tolerance_rel, tolerance.abs, tolerance.rel
        ));
    }
    if snapshot.max_entries != max_entries || snapshot.max_age_seconds != max_age_seconds {
        return Err(format!(
            "snapshot bounds (entries={}, age={}s) do not match the configured (entries={max_entries}, age={max_age_seconds}s)",
            snapshot.max_entries, snapshot.max_age_seconds
        ));
    }
    Ok(Some(snapshot))
}

/// Write the cache to its snapshot file, returning the on-disk byte count.
///
/// The file is bounded by the same entry cap as the cache, so its size is
/// bounded by [`FailedCandidateCache::worst_case_bytes`] too.
pub fn write_snapshot(
    output_dir: &Path,
    cache: &FailedCandidateCache,
    now: u64,
) -> Result<u64, String> {
    let snapshot = FailedCacheSnapshot {
        format_version: FAILED_CACHE_SNAPSHOT_FORMAT_VERSION.to_string(),
        tolerance_abs: cache.tolerance().abs,
        tolerance_rel: cache.tolerance().rel,
        max_entries: cache.max_entries(),
        max_age_seconds: cache.max_age_seconds(),
        written_unix: now,
        entries: cache.snapshot_entries(),
    };
    let encoded = serde_json::to_string(&snapshot).map_err(|e| e.to_string())?;
    fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    let path = snapshot_path(output_dir);
    fs::write(&path, &encoded).map_err(|e| e.to_string())?;
    Ok(encoded.len() as u64)
}

/// Path of the snapshot file inside `output_dir`.
pub fn snapshot_path(output_dir: &Path) -> PathBuf {
    output_dir.join(FAILED_CACHE_SNAPSHOT_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::{CandidateProvenance, CandidateStrategy};
    use crate::failed_cache::fingerprint::CandidateFingerprint;
    use crate::run::{RunConfigRecord, RunHeaderRecord, SeedSource};
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    const INCUMBENT: &str = "in1-out1-n2-s2";

    fn provenance(mutation: &str, new_value: Option<f64>) -> CandidateProvenance {
        CandidateProvenance {
            strategy: CandidateStrategy::StatsWeight,
            focus_neuron: "o1".into(),
            mutation: mutation.into(),
            old_value: Some(0.0),
            new_value,
        }
    }

    fn fingerprint(mutation: &str, new_value: Option<f64>) -> CandidateFingerprint {
        CandidateFingerprint::from_provenance(INCUMBENT, &provenance(mutation, new_value))
    }

    /// A rejected experiment whose candidates were all fully scored.
    fn experiment(number: u64, timestamp_unix: u64, mutations: &[&str]) -> ExperimentRecord {
        let mut scores = BTreeMap::new();
        scores.insert("baseline".to_string(), 0.5);
        for (i, _) in mutations.iter().enumerate() {
            scores.insert(format!("candidate-{i:03}"), 0.4);
        }
        ExperimentRecord {
            experiment_number: number,
            timestamp_unix,
            seed: Some(1),
            incumbent_id: INCUMBENT.into(),
            baseline_score: 0.5,
            focus_neuron: "o1".into(),
            focus_stats: None,
            candidates: mutations
                .iter()
                .enumerate()
                .map(|(i, m)| provenance(m, Some(i as f64)))
                .collect(),
            scores,
            screen_scores: None,
            winner: None,
            improvement: None,
            accepted: false,
            analysis_ms: 1,
            scorer_ms: 2,
            scorer_error: None,
            combo_members: None,
            combo_member_indices: None,
            combos_scored: None,
            combos_dampened: None,
            combo_dampen: None,
            cache_skipped: None,
            cache_backfilled: None,
            cache_size: None,
            cache_lookup_ms: None,
            cache_maintenance_ms: None,
            cache_saved_ms: None,
            cache_spent_ms: None,
            cache_net_cumulative_ms: None,
            cache_resident_bytes: None,
        }
    }

    fn write_journal(dir: &Path, lines: &[String]) -> PathBuf {
        let path = dir.join(FAILED_CACHE_JOURNAL_FILE);
        fs::write(&path, lines.join("\n")).unwrap();
        path
    }

    fn journal_line(record: &ExperimentRecord) -> String {
        serde_json::to_string(record).unwrap()
    }

    fn fresh_cache() -> FailedCandidateCache {
        FailedCandidateCache::new(Tolerance::default(), 1_000, 0)
    }

    #[test]
    fn candidate_stems_map_to_their_batch_index() {
        assert_eq!(candidate_index_from_stem("candidate-000"), Some(0));
        assert_eq!(candidate_index_from_stem("candidate-007"), Some(7));
        assert_eq!(candidate_index_from_stem("candidate-123"), Some(123));
        assert_eq!(candidate_index_from_stem("baseline"), None);
        assert_eq!(candidate_index_from_stem("combo-001-k2"), None);
        assert_eq!(candidate_index_from_stem("candidate-"), None);
        assert_eq!(candidate_index_from_stem("candidate-abc"), None);
    }

    /// The stem mapping is the hinge of the whole rebuild, so it is asserted
    /// against the real writer rather than a restatement of it.
    #[test]
    fn the_stem_mapping_matches_write_candidate_batch() {
        use crate::candidates::{Candidate, write_candidate_batch};
        use neat_core::parse_creature_json;

        let dir = tempdir().unwrap();
        let creature = parse_creature_json(
            r#"{
              "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
              "neurons":[{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
              "synapses":[{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
            }"#,
        )
        .unwrap();
        let candidates: Vec<Candidate> = (0..3)
            .map(|i| Candidate {
                creature: creature.clone(),
                provenance: provenance(&format!("knob-{i}"), Some(i as f64)),
            })
            .collect();
        let stems =
            write_candidate_batch(&dir.path().join("batch"), &creature, &candidates, None).unwrap();

        for stem in stems.iter().filter(|s| s.as_str() != "baseline") {
            let index = candidate_index_from_stem(stem).expect("writer stems must be mappable");
            assert_eq!(
                candidates[index].provenance.mutation,
                format!("knob-{index}"),
                "stem {stem} must address the candidate the writer wrote"
            );
        }
        assert_eq!(candidate_index_from_stem("baseline"), None);
    }

    /// Issue #92: a replayed rejection may only claim promote-phase savings
    /// when the record shows it actually reached the promote phase. A run with
    /// screening off has no promote phase for it to have reached, so its
    /// `scores` must not be replayed as promoted.
    #[test]
    fn only_a_two_phase_record_replays_a_promoted_entry() {
        let dir = tempdir().unwrap();

        // Two-phase: candidate-000 was screened *and* fully scored (promoted),
        // candidate-001 died at the screen.
        let mut two_phase = experiment(1, 1_000, &["knob-a", "knob-b"]);
        two_phase.scores = [("baseline".to_string(), 0.5), ("candidate-000".into(), 0.4)]
            .into_iter()
            .collect();
        two_phase.screen_scores = Some(
            [
                ("baseline".to_string(), 0.5),
                ("candidate-000".into(), 0.45),
                ("candidate-001".into(), 0.4),
            ]
            .into_iter()
            .collect(),
        );
        let path = write_journal(dir.path(), &[journal_line(&two_phase)]);
        let mut cache = fresh_cache();
        rebuild_from_journal(&path, &mut cache, 1_000);
        assert_eq!(
            cache.hit(&fingerprint("knob-a", Some(0.0)), 1_000),
            Some(super::super::store::CacheHit { promoted: true })
        );
        assert_eq!(
            cache.hit(&fingerprint("knob-b", Some(1.0)), 1_000),
            Some(super::super::store::CacheHit { promoted: false })
        );

        // Single-phase (screening off): `scores` is the only phase there was.
        let single_phase = experiment(1, 1_000, &["knob-a"]);
        assert!(single_phase.screen_scores.is_none());
        let single_dir = tempdir().unwrap();
        let path = write_journal(single_dir.path(), &[journal_line(&single_phase)]);
        let mut cache = fresh_cache();
        rebuild_from_journal(&path, &mut cache, 1_000);
        assert_eq!(
            cache.hit(&fingerprint("knob-a", Some(0.0)), 1_000),
            Some(super::super::store::CacheHit { promoted: false }),
            "a run without a screen phase has no promote saving to claim"
        );
    }

    #[test]
    fn an_absent_journal_rebuilds_nothing() {
        let dir = tempdir().unwrap();
        let mut cache = fresh_cache();
        let report = rebuild_from_journal(&dir.path().join("nope.jsonl"), &mut cache, 1_000);
        assert_eq!(report.source, RebuildSource::Empty);
        assert_eq!(report.entries, 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn an_empty_journal_rebuilds_nothing() {
        let dir = tempdir().unwrap();
        let path = write_journal(dir.path(), &[]);
        let mut cache = fresh_cache();
        let report = rebuild_from_journal(&path, &mut cache, 1_000);
        assert_eq!(report.lines_read, 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn a_header_only_journal_rebuilds_nothing() {
        let dir = tempdir().unwrap();
        let header = RunHeaderRecord::new(
            7,
            SeedSource::Supplied,
            RunConfigRecord::from_config(&crate::config::LamarckConfig::default()),
            1_000,
        );
        let path = write_journal(dir.path(), &[serde_json::to_string(&header).unwrap()]);
        let mut cache = fresh_cache();
        let report = rebuild_from_journal(&path, &mut cache, 1_000);
        assert_eq!(report.lines_read, 1);
        assert_eq!(report.lines_skipped, 0, "a header is known, not garbage");
        assert!(cache.is_empty());
    }

    #[test]
    fn graft_replay_lines_are_skipped_not_fatal() {
        use crate::run::{GraftReplayKind, GraftReplayRecord};

        let dir = tempdir().unwrap();
        let replay = GraftReplayRecord {
            record: GraftReplayKind::GraftReplay,
            timestamp_unix: 1_000,
            grafts_applied: 1,
            accepted: true,
            baseline_score: Some(0.5),
            score: Some(0.6),
            improvement: Some(0.1),
            elapsed_ms: 5,
            scorer_successes: 1,
            scorer_failures: 0,
            replay_error: None,
        };
        let path = write_journal(
            dir.path(),
            &[
                serde_json::to_string(&replay).unwrap(),
                journal_line(&experiment(1, 1_000, &["knob-a"])),
            ],
        );
        let mut cache = fresh_cache();
        let report = rebuild_from_journal(&path, &mut cache, 1_000);
        assert_eq!(report.lines_skipped, 0);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&fingerprint("knob-a", Some(0.0)), 1_000));
    }

    #[test]
    fn a_truncated_final_line_is_tolerated() {
        let dir = tempdir().unwrap();
        let good = journal_line(&experiment(1, 1_000, &["knob-a"]));
        let truncated = &good[..good.len() / 2];
        let path = write_journal(dir.path(), &[good.clone(), truncated.to_string()]);

        let mut cache = fresh_cache();
        let report = rebuild_from_journal(&path, &mut cache, 1_000);
        assert_eq!(report.lines_skipped, 1, "the half-written line is skipped");
        assert_eq!(cache.len(), 1, "the complete line still replays");
    }

    #[test]
    fn unknown_record_kinds_and_future_fields_are_tolerated() {
        let dir = tempdir().unwrap();
        let mut with_future_field: serde_json::Value =
            serde_json::to_value(experiment(1, 1_000, &["knob-a"])).unwrap();
        with_future_field["somethingFromTheFuture"] = serde_json::json!({"nested": true});
        let path = write_journal(
            dir.path(),
            &[
                r#"{"record":"quantumTeleport","payload":42}"#.to_string(),
                with_future_field.to_string(),
            ],
        );

        let mut cache = fresh_cache();
        let report = rebuild_from_journal(&path, &mut cache, 1_000);
        assert_eq!(report.lines_skipped, 1, "the unknown kind is skipped");
        assert_eq!(cache.len(), 1, "the forward-compatible record replays");
    }

    #[test]
    fn scorer_error_experiments_contribute_nothing() {
        let dir = tempdir().unwrap();
        let mut record = experiment(1, 1_000, &["knob-a", "knob-b"]);
        record.scorer_error = Some("boom".into());
        let path = write_journal(dir.path(), &[journal_line(&record)]);

        let mut cache = fresh_cache();
        rebuild_from_journal(&path, &mut cache, 1_000);
        assert!(
            cache.is_empty(),
            "a scorer crash is not evidence about the mutation"
        );
    }

    #[test]
    fn accepted_experiments_contribute_nothing() {
        // The journalled `incumbent_id` of an accepted experiment is the
        // *post-accept* creature, so replaying its candidates would key them to
        // an incumbent they were never proposed against.
        let dir = tempdir().unwrap();
        let mut record = experiment(1, 1_000, &["knob-a", "knob-b"]);
        record.accepted = true;
        record.winner = Some("candidate-000".into());
        record.improvement = Some(1e-5);
        let path = write_journal(dir.path(), &[journal_line(&record)]);

        let mut cache = fresh_cache();
        rebuild_from_journal(&path, &mut cache, 1_000);
        assert!(cache.is_empty());
    }

    #[test]
    fn screen_dropped_candidates_are_cached_as_failed() {
        let dir = tempdir().unwrap();
        let mut record = experiment(1, 1_000, &["knob-a", "knob-b", "knob-c"]);
        // Screen-only experiment: nothing cleared the promote threshold, so
        // there are no full-corpus scores at all.
        record.scores = BTreeMap::new();
        let mut screen = BTreeMap::new();
        screen.insert("baseline".to_string(), 0.5);
        for i in 0..3 {
            screen.insert(format!("candidate-{i:03}"), 0.4);
        }
        record.screen_scores = Some(screen);
        let path = write_journal(dir.path(), &[journal_line(&record)]);

        let mut cache = fresh_cache();
        rebuild_from_journal(&path, &mut cache, 1_000);
        assert_eq!(cache.len(), 3, "screen-dropped candidates are failures");
        assert!(cache.contains(&fingerprint("knob-c", Some(2.0)), 1_000));
        assert!(
            !cache.contains(&fingerprint("baseline", None), 1_000),
            "the baseline is not a candidate"
        );
    }

    #[test]
    fn the_winner_of_a_rejected_batch_is_not_cached() {
        // A `winner` without `accepted` cannot normally happen, but the reader
        // must never cache the stem the journal names as the winner.
        let dir = tempdir().unwrap();
        let mut record = experiment(1, 1_000, &["knob-a", "knob-b"]);
        record.winner = Some("candidate-000".into());
        let path = write_journal(dir.path(), &[journal_line(&record)]);

        let mut cache = fresh_cache();
        rebuild_from_journal(&path, &mut cache, 1_000);
        assert!(!cache.contains(&fingerprint("knob-a", Some(0.0)), 1_000));
        assert!(cache.contains(&fingerprint("knob-b", Some(1.0)), 1_000));
    }

    #[test]
    fn entries_past_the_age_bound_are_dropped_during_replay() {
        let dir = tempdir().unwrap();
        let path = write_journal(
            dir.path(),
            &[
                journal_line(&experiment(1, 1_000, &["ancient"])),
                journal_line(&experiment(2, 9_000, &["recent"])),
            ],
        );
        let mut cache = FailedCandidateCache::new(Tolerance::default(), 1_000, 3_600);
        let report = rebuild_from_journal(&path, &mut cache, 10_000);

        assert_eq!(report.rejections_seen, 2, "both were offered");
        assert_eq!(cache.len(), 1, "only the recent one was kept");
        assert!(cache.contains(&fingerprint("recent", Some(0.0)), 10_000));
    }

    #[test]
    fn the_size_cap_holds_during_replay() {
        let dir = tempdir().unwrap();
        let mutations: Vec<String> = (0..50).map(|i| format!("knob-{i}")).collect();
        let refs: Vec<&str> = mutations.iter().map(String::as_str).collect();
        let path = write_journal(dir.path(), &[journal_line(&experiment(1, 1_000, &refs))]);

        let mut cache = FailedCandidateCache::new(Tolerance::default(), 8, 0);
        rebuild_from_journal(&path, &mut cache, 1_000);
        assert_eq!(cache.len(), 8);
    }

    #[test]
    fn load_or_rebuild_replays_the_journal_when_there_is_no_snapshot() {
        let dir = tempdir().unwrap();
        write_journal(
            dir.path(),
            &[journal_line(&experiment(1, 1_000, &["knob-a"]))],
        );
        let (cache, report) = load_or_rebuild(dir.path(), Tolerance::default(), 1_000, 0, 1_000);
        assert_eq!(report.source, RebuildSource::Journal);
        assert_eq!(cache.len(), 1);
        assert!(report.snapshot_rejected.is_none());
    }

    #[test]
    fn a_snapshot_short_circuits_the_journal_replay() {
        let dir = tempdir().unwrap();
        write_journal(
            dir.path(),
            &[journal_line(&experiment(1, 1_000, &["from-journal"]))],
        );
        let mut cache = fresh_cache();
        cache.insert(fingerprint("from-snapshot", Some(9.0)), 1_000);
        let bytes = write_snapshot(dir.path(), &cache, 1_000).unwrap();
        assert!(bytes > 0, "the on-disk size is reported");

        let (restored, report) = load_or_rebuild(dir.path(), Tolerance::default(), 1_000, 0, 1_000);
        assert_eq!(report.source, RebuildSource::Snapshot);
        assert!(restored.contains(&fingerprint("from-snapshot", Some(9.0)), 1_000));
        assert!(
            !restored.contains(&fingerprint("from-journal", Some(0.0)), 1_000),
            "a usable snapshot means the journal is not re-read"
        );
    }

    #[test]
    fn a_corrupt_snapshot_falls_back_to_the_journal() {
        let dir = tempdir().unwrap();
        write_journal(
            dir.path(),
            &[journal_line(&experiment(1, 1_000, &["from-journal"]))],
        );
        fs::write(snapshot_path(dir.path()), "{ this is not json").unwrap();

        let (cache, report) = load_or_rebuild(dir.path(), Tolerance::default(), 1_000, 0, 1_000);
        assert_eq!(report.source, RebuildSource::Journal);
        assert!(
            report
                .snapshot_rejected
                .as_deref()
                .is_some_and(|r| r.contains("unparsable")),
            "the reason is reported, not swallowed: {:?}",
            report.snapshot_rejected
        );
        assert!(cache.contains(&fingerprint("from-journal", Some(0.0)), 1_000));
    }

    #[test]
    fn a_version_mismatched_snapshot_falls_back_to_the_journal() {
        let dir = tempdir().unwrap();
        write_journal(
            dir.path(),
            &[journal_line(&experiment(1, 1_000, &["from-journal"]))],
        );
        let mut cache = fresh_cache();
        cache.insert(fingerprint("from-snapshot", Some(9.0)), 1_000);
        write_snapshot(dir.path(), &cache, 1_000).unwrap();
        let path = snapshot_path(dir.path());
        let bumped = fs::read_to_string(&path)
            .unwrap()
            .replace(FAILED_CACHE_SNAPSHOT_FORMAT_VERSION, "9.9.9");
        fs::write(&path, bumped).unwrap();

        let (cache, report) = load_or_rebuild(dir.path(), Tolerance::default(), 1_000, 0, 1_000);
        assert_eq!(report.source, RebuildSource::Journal);
        assert!(
            report
                .snapshot_rejected
                .as_deref()
                .is_some_and(|r| r.contains("formatVersion")),
            "{:?}",
            report.snapshot_rejected
        );
        assert!(cache.contains(&fingerprint("from-journal", Some(0.0)), 1_000));
    }

    #[test]
    fn changed_tolerance_knobs_invalidate_the_snapshot() {
        // Entries were bucketed under different matching rules, so they cannot
        // be trusted under the new ones.
        let dir = tempdir().unwrap();
        write_journal(
            dir.path(),
            &[journal_line(&experiment(1, 1_000, &["from-journal"]))],
        );
        let mut cache = fresh_cache();
        cache.insert(fingerprint("from-snapshot", Some(9.0)), 1_000);
        write_snapshot(dir.path(), &cache, 1_000).unwrap();

        let (cache, report) =
            load_or_rebuild(dir.path(), Tolerance::new(1e-3, 1e-2), 1_000, 0, 1_000);
        assert_eq!(report.source, RebuildSource::Journal);
        assert!(
            report
                .snapshot_rejected
                .as_deref()
                .is_some_and(|r| r.contains("tolerance")),
            "{:?}",
            report.snapshot_rejected
        );
        assert!(!cache.contains(&fingerprint("from-snapshot", Some(9.0)), 1_000));
    }

    #[test]
    fn changed_bounds_invalidate_the_snapshot() {
        let dir = tempdir().unwrap();
        let mut cache = fresh_cache();
        cache.insert(fingerprint("from-snapshot", Some(9.0)), 1_000);
        write_snapshot(dir.path(), &cache, 1_000).unwrap();

        let (_, report) = load_or_rebuild(dir.path(), Tolerance::default(), 17, 0, 1_000);
        assert!(
            report
                .snapshot_rejected
                .as_deref()
                .is_some_and(|r| r.contains("bounds")),
            "{:?}",
            report.snapshot_rejected
        );
    }

    #[test]
    fn a_rebuild_reports_its_cost() {
        let dir = tempdir().unwrap();
        write_journal(
            dir.path(),
            &[journal_line(&experiment(1, 1_000, &["knob-a"]))],
        );
        let (_, report) = load_or_rebuild(dir.path(), Tolerance::default(), 1_000, 0, 1_000);
        // Nothing to assert about the magnitude on a two-line fixture; the
        // contract is that the field exists and the run can log it.
        assert!(report.elapsed_ms < 60_000);
        assert_eq!(report.entries, 1);
    }
}
