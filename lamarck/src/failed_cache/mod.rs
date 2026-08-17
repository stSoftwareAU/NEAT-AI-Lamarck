//! Cache of candidates that were scored and did not improve (issue #69).
//!
//! Lamarck re-proposes the same mutations across experiments and across runs.
//! Re-scoring a candidate that is already known to fail costs full scorer time
//! and buys nothing, so this module remembers rejections and keeps them out of
//! the batch.
//!
//! The feature is **off by default** (`--failed-cache`) until the benchmark
//! sub-issue of #69 says it pays for itself. With it off, nothing here runs:
//! no RNG draw is taken, no journal field is written, and the run behaves
//! exactly as it did before.
//!
//! The four parts land as separate sub-issues:
//!
//! * [`fingerprint`] (#88) — what makes two candidates "the same", including
//!   the near-duplicate tolerance on the changed scalar.
//! * [`store`] (#89) — the bounded in-memory cache, with a size cap and an age
//!   bound so the footprint is a stated number.
//! * [`rebuild`] (#90) — startup rebuild from `experiments.jsonl` and the
//!   optional on-disk snapshot.
//! * [`filter`] (#91) — dropping known-failed candidates from a batch and
//!   backfilling it so the scorer still runs at full width.
//!
//! [`economics`] (#92) then measures both sides of the ledger — estimated
//! scorer time saved against measured overhead and footprint — and stands the
//! cache down when it stops paying, so a cache that does not earn its keep
//! degrades to the cache-off behaviour instead of degrading the run.

pub mod economics;
pub mod filter;
pub mod fingerprint;
pub mod rebuild;
pub mod store;

pub use economics::{
    CacheEconomics, CacheEconomicsConfig, CeilingBite, DEFAULT_CACHE_MAX_RESIDENT_BYTES,
    DEFAULT_CACHE_STAND_DOWN_MARGIN_MS, DEFAULT_CACHE_STAND_DOWN_WINDOW, EconomicsSummary,
    ExperimentCost, ExperimentEconomics, StandDown,
};
pub use filter::{
    BACKFILL_PROPOSAL_BUDGET_MULTIPLE, FilteredBatch, filter_and_backfill, insert_failures,
    scored_candidate_indices,
};
pub use fingerprint::{
    CandidateFingerprint, DEFAULT_FAILED_CACHE_TOLERANCE_ABS, DEFAULT_FAILED_CACHE_TOLERANCE_REL,
    FingerprintBucketKey, Tolerance,
};
pub use rebuild::{
    FAILED_CACHE_JOURNAL_FILE, FAILED_CACHE_SNAPSHOT_FILE, FAILED_CACHE_SNAPSHOT_FORMAT_VERSION,
    FailedCacheSnapshot, RebuildReport, RebuildSource, candidate_index_from_stem, load_or_rebuild,
    rebuild_from_journal, snapshot_path, write_snapshot,
};
pub use store::{
    CacheHit, DEFAULT_FAILED_CACHE_MAX_AGE_SECONDS, DEFAULT_FAILED_CACHE_MAX_ENTRIES,
    FAILED_CACHE_BYTES_PER_ENTRY, FailedCacheEntry, FailedCacheStats, FailedCandidateCache,
};
