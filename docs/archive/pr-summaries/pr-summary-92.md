# Failed-candidate cache: overhead accounting and stand-down (Issue #92)

## Summary

The hard constraint on #69 — the failed-candidate cache must not spend more time
than it saves, and must not grow without bound — is now enforced in the code
rather than asserted in a benchmark report. The cache keeps a ledger of both
sides, journals it per experiment, and stands itself down when it stops paying.
Closes #92.

- **New `lamarck/src/failed_cache/economics.rs`.** Savings are *estimated* from
  this run's own measured screen cost (`skipped × mean screen ms per scored
  creature`, dividing by every creature the batch scored so the baseline cannot
  inflate the figure), never from a constant. With screening off, the single
  full-corpus batch is that first phase and is priced as such. Promote-phase
  time is claimed only for a skip whose cache entry records the candidate as
  having actually been promoted; every other skip is screen-cost only, so the
  estimate under-claims rather than over-claims. Spend is *measured* — lookup,
  maintenance, the startup rebuild and the snapshot write — and accumulated in
  **microseconds**, because a whole-millisecond timer truncates a small batch's
  overhead to zero and an overhead that rounds to zero would let a losing cache
  look free.
- **Stand-down.** When the cumulative net stays worse than
  `--failed-cache-stand-down-margin-ms` (default `1000`) for
  `--failed-cache-stand-down-window` (default `20`) consecutive experiments,
  Lamarck logs a warning, writes a `cacheStandDown` journal line and disables the
  cache for the remainder of the run — which continues, and writes no snapshot.
  A cache that does not earn its keep degrades to today's cache-off behaviour
  instead of degrading the run.
- **Byte ceiling.** `--failed-cache-max-bytes` (default ~25 MiB, the entry cap's
  own worst case) bounds the resident footprint; past it the cache evicts
  oldest-first and **logs the bite**, because a silently truncated cache reads as
  a working cache.
- **Reporting.** Each experiment journals `cacheSavedMs`, `cacheSpentMs`,
  `cacheNetCumulativeMs` and `cacheResidentBytes`; every cache-on run ends with
  one parseable summary line (entries, hit rate, ms saved, ms spent, net, peak
  memory bytes, disk bytes). The `runHeader` records the guardrail knobs so an
  arm is identifiable from its journal alone.

Cross-run aggregation in `report` and the production A/B remain out of scope —
they are the last two sub-issues of #69.

## Evidence

Backend/CLI change with no web interface, so there is no screenshot to capture.
The evidence is the test suite plus `./quality.sh`, which passes cleanly
(fmt, clippy `-D warnings`, cargo-deny, 219 lib tests + 7 new integration tests
in `cache_economics` + the existing integration suites, rustdoc `-D warnings`).

```mermaid
flowchart TD
    START[Startup rebuild: journal or snapshot] -->|rebuild ms| SPENT
    FILTER[Filter batch against the cache] -->|lookup micros| SPENT
    SWEEP[Age sweep and ceiling eviction] -->|maintenance micros| SPENT
    SNAP[Snapshot write at end of run] -->|write ms + disk bytes| SPENT
    SCREEN[Measured screen cost per creature this run] --> SAVED
    SKIP[Candidates skipped by a cache hit] --> SAVED
    PROMO[Skips whose entry had reached promote] --> SAVED
    SAVED[Estimated ms saved] --> NET{cumulative net worse than -margin<br/>for the whole window?}
    SPENT[Measured ms spent] --> NET
    NET -->|no| KEEP[Cache stays on]
    NET -->|yes| DOWN[Warn, journal cacheStandDown,<br/>disable cache, run continues]
    KEEP --> SUMMARY[End-of-run summary line]
    DOWN --> SUMMARY
```

The end-of-run line downstream tooling parses:

```text
● failed-cache economics: entries=1240 hitRate=0.1832 savedMs=48210.5 spentMs=311.2 netMs=47899.3 peakMemoryBytes=634880 diskBytes=98304 standDown=false ceilingBites=0
```

## Test Plan

New integration suite `lamarck/tests/cache_economics.rs` — the failure-detection
points the issue names:

- `accounting_matches_known_inputs` — fixed skip counts, screen/promote samples
  and lookup/maintenance/rebuild/snapshot costs produce exact saved/spent/net,
  hit-rate and footprint figures, including that promote savings accrue *only*
  to previously-promoted skips. Any drift in the estimator fails here first.
- `net_negative_run_disables_cache` — a run whose scorer costs nothing (so every
  skip is worth nothing) warns, journals exactly one `cacheStandDown` carrying
  the logged message, leaves every later experiment without cache fields, writes
  no snapshot, and still completes normally.
- `net_positive_run_keeps_cache_enabled` — the inverse guard against an
  over-eager stand-down silently reverting the feature: with a 30 ms batch the
  savings dwarf the lookup cost, nothing stands down and the snapshot is written.
- `ceiling_evicts_and_logs` / `under_ceiling_run_does_not_evict` — 50 entries
  against an 8-entry ceiling evict 42, leave the footprint at or under the
  ceiling and produce a reportable bite; a cache inside its ceiling evicts
  nothing but is still measured.
- `a_warm_cache_above_the_ceiling_is_evicted_at_startup` — a second run over the
  same output directory, under a tighter ceiling, brings the restored snapshot
  under the bound before the loop starts rather than after the first insert.
- `end_of_run_summary_line_carries_every_field` — asserts every field name on the
  summary line of a real run.
- `cache_off_run_has_no_ledger` — a cache-off run journals no economics fields at
  all.

Unit tests added alongside the code:

- `failed_cache::economics::tests` — measured-not-constant screen pricing,
  promote-only-when-promoted, sub-millisecond spend not rounded away, the window
  needing *consecutive* losses (and resetting on a win), `stand_down_window = 0`
  opting out, ceiling behaviour, and the summary-line shape.
- `failed_cache::store::tests::a_promoted_failure_is_reported_by_the_hit_and_snapshotted`
  — the `promoted` flag survives lookup and snapshot, and a later promote of an
  already-known screen failure does not upgrade it (under-claim, not over-claim).
- `failed_cache::filter::tests::a_filtered_batch_separates_previously_promoted_skips`.
- `run::tests::cache_on_journals_per_experiment_economics` — per-experiment
  ledger fields are journalled, the footprint equals entries × entry size, and
  the guardrail knobs are in the `runHeader`.
- `config::tests::the_economics_guardrail_is_configured_by_default`.

Existing tests were updated for the new `insert_failures` promoted argument, the
new config fields and the new `JournalLine::CacheStandDown` variant; none were
removed or weakened.
