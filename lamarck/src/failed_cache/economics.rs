//! Overhead accounting and stand-down for the failed-candidate cache (#92).
//!
//! The hard constraint on #69 is that the cache must not cost more time than it
//! saves, and must not grow without bound. Neither is enforceable by a sentence
//! in a benchmark report, so the cache measures both sides of its own ledger
//! here and stands down when it stops paying.
//!
//! # How savings are estimated
//!
//! A skipped candidate is one the scorer never saw, so its cost has to be
//! *estimated* rather than measured. The estimate uses this run's own measured
//! scorer cost, never a constant:
//!
//! ```text
//! saved_ms = skipped × mean_screen_ms_per_creature
//!          + skipped_previously_promoted × mean_promote_ms_per_creature
//! ```
//!
//! * `mean_screen_ms_per_creature` is this run's screen-phase wall clock
//!   divided by the creatures scored in it (the baseline included, so the
//!   baseline's own cost cannot inflate the per-candidate figure). The screen
//!   phase is where a skipped candidate would have been scored, so it is the
//!   only saving every skip earns. In a run with screening disabled, the single
//!   full-corpus batch *is* the screen phase for accounting purposes and is
//!   observed as such.
//! * The promote term is **conservative by default**: promote-phase time only
//!   accrues to candidates that would have cleared the screen threshold, and
//!   that cannot be known for a candidate that was never scored. It is claimed
//!   only for a skipped candidate the cache recorded as having actually been
//!   promoted before ([`super::store::FailedCandidateCache::hit`] reports it),
//!   and it is stated separately in the accounting so the claim is auditable.
//!   With screen-cost only — the default for every candidate never promoted —
//!   the estimate understates rather than overstates.
//!
//! Only a **cache hit** counts as a skip. A proposal dropped because it repeats
//! one already in the same batch is the generator's doing, not the cache's, and
//! is reported separately by [`super::filter::FilteredBatch::duplicates`] so it
//! cannot inflate the hit rate or the savings.
//!
//! ## What "saved" does and does not mean
//!
//! `saved_ms` is **redundant scorer work avoided** — the scorer time this run
//! would have spent on candidates already known to fail. That is the quantity
//! #69's constraint is stated in ("we spend more time than it saves in
//! redundant scoring"), so it is what the guardrail compares overhead against.
//!
//! It is **not** the same as wall clock removed from the run. The batch is
//! backfilled to full width (#91: the scorer batch is the unit of cost), so a
//! skip that was replaced converts redundant scoring into fresh exploration
//! rather than shortening the batch. Only the skips the backfill could not
//! replace actually shrink the work the scorer does, and that part is reported
//! separately as [`EconomicsSummary::wall_clock_saved_ms`] so the two are never
//! confused.
//!
//! Spend, by contrast, is measured: lookup and backfill time, age-sweep and
//! ceiling-eviction maintenance, the one-off startup rebuild, and the one-off
//! snapshot write. Spend is accumulated in **microseconds** because a
//! whole-millisecond lookup timer truncates to zero on a small batch, and an
//! overhead that rounds to zero is exactly the bias that would let a
//! net-negative cache look free.
//!
//! # Standing down
//!
//! The guardrail looks at a **rolling window** of the most recent
//! [`CacheEconomicsConfig::stand_down_window`] experiments: when the spend
//! inside that window exceeds the savings inside it by
//! [`CacheEconomicsConfig::stand_down_margin_ms`], the cache stands down. The
//! caller logs the warning, journals the event and disables the cache for the
//! remainder of the run.
//!
//! A window rather than the whole run matters: a run-long cumulative test lets
//! a sunk one-off cost (a slow startup rebuild) condemn a cache that is
//! currently profitable, which is not "it stopped paying" — and disabling it
//! cannot un-spend that cost, it only removes the chance of earning it back.
//! One-off costs therefore count in the run's cumulative ledger, which the
//! summary reports, but not in the window the guardrail judges.

use super::store::{
    DEFAULT_FAILED_CACHE_MAX_ENTRIES, FAILED_CACHE_BYTES_PER_ENTRY, FailedCandidateCache,
};
use std::collections::VecDeque;
use std::time::Instant;

/// Default margin, in milliseconds, that a window's spend must exceed its
/// savings by before the cache is stood down.
///
/// One second of run time is the smallest overspend worth acting on: below it
/// the estimator's own noise (a screen batch that happened to be slow) is the
/// larger term, and standing down on noise would silently revert the feature.
pub const DEFAULT_CACHE_STAND_DOWN_MARGIN_MS: f64 = 1_000.0;

/// Default length of the rolling window the guardrail judges the cache over.
///
/// A cold cache is *always* net-negative for its first experiments — it pays
/// lookup and rebuild cost before it holds anything to hit on — so the window
/// has to be long enough to let it warm up. Twenty experiments is a few minutes
/// of a production run and several times the batch reuse distance at which the
/// first hits appear.
pub const DEFAULT_CACHE_STAND_DOWN_WINDOW: usize = 20;

/// Default resident-footprint ceiling in bytes (~25 MiB).
///
/// Matched to the entry cap's own worst case so the two bounds agree by
/// construction: it is a bound, not a target, and it bites only when the entry
/// cap alone would have allowed more bytes than this.
pub const DEFAULT_CACHE_MAX_RESIDENT_BYTES: usize =
    DEFAULT_FAILED_CACHE_MAX_ENTRIES * FAILED_CACHE_BYTES_PER_ENTRY;

/// Guardrail knobs for the cache's economics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheEconomicsConfig {
    /// Milliseconds a window's spend must exceed its savings by before the
    /// cache stands down.
    pub stand_down_margin_ms: f64,
    /// Experiments in the rolling window the guardrail judges. `0` disables the
    /// guardrail.
    pub stand_down_window: usize,
    /// Resident-footprint ceiling in bytes. `0` disables the ceiling; the entry
    /// cap still bounds the cache.
    pub max_resident_bytes: usize,
}

impl Default for CacheEconomicsConfig {
    fn default() -> Self {
        Self {
            stand_down_margin_ms: DEFAULT_CACHE_STAND_DOWN_MARGIN_MS,
            stand_down_window: DEFAULT_CACHE_STAND_DOWN_WINDOW,
            max_resident_bytes: DEFAULT_CACHE_MAX_RESIDENT_BYTES,
        }
    }
}

/// One experiment's cache cost, as measured by the run loop.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExperimentCost {
    /// Proposals the cache examined, including the original batch.
    pub proposals: usize,
    /// Proposals dropped by a cache hit (cache hits only — not the generator's
    /// own intra-batch duplicates).
    pub skipped: usize,
    /// Of those, the ones the cache had recorded as previously promoted.
    pub skipped_previously_promoted: usize,
    /// Replacement proposals accepted into the batch to refill it.
    ///
    /// A refilled slot means the scorer still ran at full width, so only
    /// `skipped - backfilled` shortened the work the scorer actually did.
    pub backfilled: usize,
    /// Microseconds spent filtering and backfilling the batch.
    pub lookup_micros: u128,
    /// Microseconds spent in cache maintenance (the age sweep) this experiment.
    pub maintenance_micros: u128,
    /// Live entries after the experiment.
    pub entries: usize,
}

/// What one experiment came to, and where the run's ledger stands after it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExperimentEconomics {
    /// Redundant scorer milliseconds this experiment's skips are estimated to
    /// have avoided.
    pub saved_ms: f64,
    /// Milliseconds this experiment spent on lookups and maintenance.
    ///
    /// Ceiling evictions, the startup rebuild and the snapshot write are run
    /// costs rather than one experiment's, so they appear only in the run
    /// totals: summing this column does not reproduce
    /// [`EconomicsSummary::spent_ms`].
    pub spent_ms: f64,
    /// Cumulative saved minus cumulative spent for the run so far.
    pub net_cumulative_ms: f64,
    /// Resident cache bytes after the experiment.
    pub resident_bytes: usize,
}

/// A cache that stopped paying and was taken out of play.
#[derive(Debug, Clone, PartialEq)]
pub struct StandDown {
    /// Experiment the guardrail fired on.
    pub experiment_number: u64,
    /// Estimated savings inside the window that triggered it.
    pub saved_ms: f64,
    /// Measured spend inside the window that triggered it.
    pub spent_ms: f64,
    /// `saved_ms - spent_ms` for that window.
    pub net_ms: f64,
    /// Experiments in the window.
    pub window_experiments: usize,
    /// Margin the window's net had to be worse than.
    pub margin_ms: f64,
    /// Live entries the cache is giving up.
    pub entries: usize,
}

impl StandDown {
    /// The warning line the caller logs and journals.
    pub fn message(&self) -> String {
        format!(
            "failed-cache: STANDING DOWN at experiment {} — over the last {} experiment(s) it spent {:.1}ms to avoid an estimated {:.1}ms of redundant scoring (net {:.1}ms, worse than the {:.1}ms margin); disabling the cache for the rest of the run ({} entr(ies) dropped from play)",
            self.experiment_number,
            self.window_experiments,
            self.spent_ms,
            self.saved_ms,
            self.net_ms,
            self.margin_ms,
            self.entries
        )
    }
}

/// A byte ceiling that bit, and what it cost the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CeilingBite {
    /// Configured ceiling in bytes.
    pub ceiling_bytes: usize,
    /// Resident bytes before eviction.
    pub bytes_before: usize,
    /// Resident bytes after eviction.
    pub bytes_after: usize,
    /// Entries evicted to get under the ceiling.
    pub evicted: usize,
}

impl CeilingBite {
    /// The warning line the caller logs; a bound that bites silently reads as a
    /// working cache.
    pub fn message(&self) -> String {
        format!(
            "failed-cache: BYTE CEILING BIT — evicted {} oldest entr(ies) to bring the resident footprint from {} to {} bytes, under the {} byte ceiling",
            self.evicted, self.bytes_before, self.bytes_after, self.ceiling_bytes
        )
    }
}

/// The run's cache ledger, as reported at the end of the run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EconomicsSummary {
    /// Live entries at the end of the run.
    pub entries: usize,
    /// Proposals the cache examined over the run.
    pub proposals: u64,
    /// Proposals dropped by a cache hit.
    pub skipped: u64,
    /// `skipped / proposals`, or `0` when nothing was examined.
    pub hit_rate: f64,
    /// Cumulative estimated milliseconds of **redundant scoring** avoided.
    ///
    /// See the [module documentation](self): this is the quantity #69's
    /// constraint is stated in, not wall clock removed from the run.
    pub saved_ms: f64,
    /// The part of [`Self::saved_ms`] that shortened the scorer's actual work.
    ///
    /// A skip the backfill replaced left the batch at full width, so it bought
    /// fresh exploration rather than time. Only skips that could not be
    /// replaced shrink the batch, and they are priced at screen cost alone.
    pub wall_clock_saved_ms: f64,
    /// Cumulative measured milliseconds of cache overhead.
    pub spent_ms: f64,
    /// `saved_ms - spent_ms`.
    pub net_ms: f64,
    /// Largest resident footprint the cache reached.
    ///
    /// Measured *before* any ceiling eviction, so it reports the bytes that
    /// were genuinely resident rather than the bound that reclaimed them: a
    /// peak above the ceiling is the honest number, and the matching
    /// [`Self::ceiling_bites`] says how often it was pulled back.
    pub peak_resident_bytes: usize,
    /// Bytes the snapshot occupies on disk.
    pub disk_bytes: u64,
    /// Whether the guardrail took the cache out of play.
    pub stood_down: bool,
    /// Times the byte ceiling had to evict.
    pub ceiling_bites: u64,
}

impl EconomicsSummary {
    /// The single end-of-run line downstream tooling parses.
    ///
    /// Every field the benchmark sub-issue of #69 needs is on this one line and
    /// is named here; renaming or dropping one breaks that tooling, so the
    /// shape is asserted by test.
    pub fn summary_line(&self) -> String {
        format!(
            "failed-cache economics: entries={} hitRate={:.4} savedMs={:.1} wallClockSavedMs={:.1} spentMs={:.1} netMs={:.1} peakMemoryBytes={} diskBytes={} standDown={} ceilingBites={}",
            self.entries,
            self.hit_rate,
            self.saved_ms,
            self.wall_clock_saved_ms,
            self.spent_ms,
            self.net_ms,
            self.peak_resident_bytes,
            self.disk_bytes,
            self.stood_down,
            self.ceiling_bites
        )
    }
}

/// Accountant for one run's failed-candidate cache.
///
/// See the [module documentation](self) for the estimation method and its
/// deliberate conservatism.
#[derive(Debug)]
pub struct CacheEconomics {
    config: CacheEconomicsConfig,
    screen_ms_total: f64,
    screen_creatures: u64,
    promote_ms_total: f64,
    promote_creatures: u64,
    proposals: u64,
    skipped: u64,
    skipped_previously_promoted: u64,
    skipped_not_backfilled: u64,
    spent_micros: u128,
    /// Per-experiment `(skips, promoted skips, spend micros)` for the most
    /// recent window, from which the guardrail's sums are taken.
    window: VecDeque<(u64, u64, u128)>,
    experiments: u64,
    entries: usize,
    peak_resident_bytes: usize,
    disk_bytes: u64,
    ceiling_bites: u64,
    stood_down: bool,
    pending_stand_down: Option<StandDown>,
}

impl CacheEconomics {
    /// A fresh ledger under the supplied guardrail knobs.
    pub fn new(config: CacheEconomicsConfig) -> Self {
        Self {
            config,
            screen_ms_total: 0.0,
            screen_creatures: 0,
            promote_ms_total: 0.0,
            promote_creatures: 0,
            proposals: 0,
            skipped: 0,
            skipped_previously_promoted: 0,
            skipped_not_backfilled: 0,
            spent_micros: 0,
            window: VecDeque::new(),
            experiments: 0,
            entries: 0,
            peak_resident_bytes: 0,
            disk_bytes: 0,
            ceiling_bites: 0,
            stood_down: false,
            pending_stand_down: None,
        }
    }

    /// The guardrail knobs in force.
    pub fn config(&self) -> CacheEconomicsConfig {
        self.config
    }

    /// Charge the one-off startup rebuild (journal replay or snapshot load).
    ///
    /// It counts against the run's cumulative ledger — the summary's `netMs`
    /// includes it — but deliberately **not** against the stand-down window.
    /// The rebuild is sunk before the first experiment: no later experiment can
    /// un-spend it, so standing the cache down over it would only remove the
    /// chance of earning it back. The window answers "is the cache paying its
    /// way now?"; the summary answers "did this run come out ahead?".
    pub fn record_startup_rebuild(&mut self, elapsed_ms: u128) {
        self.spent_micros += elapsed_ms.saturating_mul(1_000);
    }

    /// Charge the one-off snapshot write and record its size on disk.
    pub fn record_snapshot(&mut self, elapsed_ms: u128, bytes: u64) {
        self.spent_micros += elapsed_ms.saturating_mul(1_000);
        self.disk_bytes = bytes;
    }

    /// Observe a screen-phase batch: its wall clock and the creatures scored.
    ///
    /// `creatures` includes the baseline, because the baseline is one of the
    /// creatures the batch actually paid for; dividing by the candidates alone
    /// would inflate the per-candidate cost and over-claim every skip. In a run
    /// with screening disabled, the single full-corpus batch is observed here —
    /// it is the phase a skipped candidate would have been scored in.
    pub fn observe_screen(&mut self, elapsed_ms: u128, creatures: usize) {
        if creatures == 0 || self.stood_down {
            return;
        }
        self.screen_ms_total += elapsed_ms as f64;
        self.screen_creatures += creatures as u64;
    }

    /// Observe a promote-phase batch: its wall clock and the creatures scored.
    ///
    /// Like [`Self::observe_screen`], ignored once the cache has stood down:
    /// the ledger closes with the cache, so its means describe the run the
    /// cache was actually part of.
    pub fn observe_promote(&mut self, elapsed_ms: u128, creatures: usize) {
        if creatures == 0 || self.stood_down {
            return;
        }
        self.promote_ms_total += elapsed_ms as f64;
        self.promote_creatures += creatures as u64;
    }

    /// Measured mean screen-phase milliseconds per scored creature.
    pub fn mean_screen_ms(&self) -> f64 {
        if self.screen_creatures == 0 {
            return 0.0;
        }
        self.screen_ms_total / self.screen_creatures as f64
    }

    /// Measured mean promote-phase milliseconds per scored creature.
    pub fn mean_promote_ms(&self) -> f64 {
        if self.promote_creatures == 0 {
            return 0.0;
        }
        self.promote_ms_total / self.promote_creatures as f64
    }

    /// Cumulative estimated milliseconds of redundant scoring the skips avoided.
    ///
    /// Every skip recorded so far is re-valued at the run's current measured
    /// means, so a mean that firms up over the run corrects the whole ledger
    /// rather than only its tail.
    pub fn saved_ms(&self) -> f64 {
        self.skipped as f64 * self.mean_screen_ms()
            + self.skipped_previously_promoted as f64 * self.mean_promote_ms()
    }

    /// The part of [`Self::saved_ms`] that shortened the scorer's actual work.
    ///
    /// Screen cost only, and only for skips the backfill could not replace: a
    /// refilled batch still scores at full width, so it bought exploration
    /// rather than time.
    pub fn wall_clock_saved_ms(&self) -> f64 {
        self.skipped_not_backfilled as f64 * self.mean_screen_ms()
    }

    /// Cumulative measured milliseconds of cache overhead.
    pub fn spent_ms(&self) -> f64 {
        self.spent_micros as f64 / 1_000.0
    }

    /// `saved_ms() - spent_ms()`.
    pub fn net_ms(&self) -> f64 {
        self.saved_ms() - self.spent_ms()
    }

    /// Whether the guardrail has taken the cache out of play.
    pub fn stood_down(&self) -> bool {
        self.stood_down
    }

    /// Take the stand-down event the last recorded experiment triggered, if any.
    ///
    /// Returns `Some` exactly once per run, so the caller can journal the event
    /// without risking a duplicate line.
    pub fn take_stand_down(&mut self) -> Option<StandDown> {
        self.pending_stand_down.take()
    }

    /// Account for one experiment and re-test the guardrail.
    pub fn record_experiment(
        &mut self,
        experiment_number: u64,
        cost: ExperimentCost,
    ) -> ExperimentEconomics {
        let saved_ms = cost.skipped as f64 * self.mean_screen_ms()
            + cost.skipped_previously_promoted as f64 * self.mean_promote_ms();
        let spent_micros = cost.lookup_micros.saturating_add(cost.maintenance_micros);

        self.experiments += 1;
        self.proposals += cost.proposals as u64;
        self.skipped += cost.skipped as u64;
        self.skipped_previously_promoted += cost.skipped_previously_promoted as u64;
        self.skipped_not_backfilled += cost.skipped.saturating_sub(cost.backfilled) as u64;
        self.spent_micros += spent_micros;
        self.entries = cost.entries;
        let resident_bytes = cost.entries.saturating_mul(FAILED_CACHE_BYTES_PER_ENTRY);
        self.peak_resident_bytes = self.peak_resident_bytes.max(resident_bytes);

        self.push_window(
            cost.skipped as u64,
            cost.skipped_previously_promoted as u64,
            spent_micros,
        );

        let net_cumulative_ms = self.net_ms();
        self.test_guardrail(experiment_number);

        ExperimentEconomics {
            saved_ms,
            spent_ms: spent_micros as f64 / 1_000.0,
            net_cumulative_ms,
            resident_bytes,
        }
    }

    /// Evict down to the byte ceiling, charging the eviction as overhead.
    ///
    /// Returns the bite when the ceiling actually bound, so the caller can log
    /// it: a bound that truncates the cache silently reads as a working cache.
    pub fn enforce_ceiling(&mut self, cache: &mut FailedCandidateCache) -> Option<CeilingBite> {
        let ceiling = self.config.max_resident_bytes;
        let bytes_before = cache.resident_bytes();
        self.peak_resident_bytes = self.peak_resident_bytes.max(bytes_before);
        if ceiling == 0 || bytes_before <= ceiling {
            return None;
        }
        let started = Instant::now();
        let evicted = cache.trim_to(ceiling / FAILED_CACHE_BYTES_PER_ENTRY);
        self.spent_micros += started.elapsed().as_micros();
        self.ceiling_bites += 1;
        Some(CeilingBite {
            ceiling_bytes: ceiling,
            bytes_before,
            bytes_after: cache.resident_bytes(),
            evicted,
        })
    }

    /// The run's ledger, for the end-of-run summary line.
    pub fn summary(&self) -> EconomicsSummary {
        let saved_ms = self.saved_ms();
        let spent_ms = self.spent_ms();
        EconomicsSummary {
            entries: self.entries,
            proposals: self.proposals,
            skipped: self.skipped,
            hit_rate: if self.proposals == 0 {
                0.0
            } else {
                self.skipped as f64 / self.proposals as f64
            },
            saved_ms,
            wall_clock_saved_ms: self.wall_clock_saved_ms(),
            spent_ms,
            net_ms: saved_ms - spent_ms,
            peak_resident_bytes: self.peak_resident_bytes,
            disk_bytes: self.disk_bytes,
            stood_down: self.stood_down,
            ceiling_bites: self.ceiling_bites,
        }
    }

    /// Add one experiment to the rolling window, dropping the oldest.
    fn push_window(&mut self, skipped: u64, previously_promoted: u64, spent_micros: u128) {
        let size = self.config.stand_down_window;
        if size == 0 {
            return;
        }
        self.window
            .push_back((skipped, previously_promoted, spent_micros));
        while self.window.len() > size {
            self.window.pop_front();
        }
    }

    /// Stand the cache down when the whole window has lost beyond the margin.
    ///
    /// The window's savings are valued at the run's current measured means, the
    /// same figures the summary reports, so the decision and the report can
    /// never disagree.
    fn test_guardrail(&mut self, experiment_number: u64) {
        if self.stood_down || self.config.stand_down_window == 0 {
            return;
        }
        if self.window.len() < self.config.stand_down_window {
            return;
        }
        let (skipped, promoted, spent_micros) = self.window.iter().fold(
            (0u64, 0u64, 0u128),
            |(s, p, micros), (skip, promo, spend)| {
                (s + skip, p + promo, micros.saturating_add(*spend))
            },
        );
        let saved_ms =
            skipped as f64 * self.mean_screen_ms() + promoted as f64 * self.mean_promote_ms();
        let spent_ms = spent_micros as f64 / 1_000.0;
        let net_ms = saved_ms - spent_ms;
        if net_ms >= -self.config.stand_down_margin_ms {
            return;
        }
        self.stood_down = true;
        self.pending_stand_down = Some(StandDown {
            experiment_number,
            saved_ms,
            spent_ms,
            net_ms,
            window_experiments: self.window.len(),
            margin_ms: self.config.stand_down_margin_ms,
            entries: self.entries,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::{CandidateProvenance, CandidateStrategy};
    use crate::failed_cache::fingerprint::{CandidateFingerprint, Tolerance};

    fn economics(margin_ms: f64, window: usize, ceiling: usize) -> CacheEconomics {
        CacheEconomics::new(CacheEconomicsConfig {
            stand_down_margin_ms: margin_ms,
            stand_down_window: window,
            max_resident_bytes: ceiling,
        })
    }

    fn fingerprint(mutation: &str) -> CandidateFingerprint {
        CandidateFingerprint::from_provenance(
            "inc",
            &CandidateProvenance {
                strategy: CandidateStrategy::StatsWeight,
                focus_neuron: "o1".into(),
                mutation: mutation.into(),
                old_value: Some(0.0),
                new_value: Some(1.0),
            },
        )
    }

    #[test]
    fn savings_use_the_measured_screen_cost_not_a_constant() {
        let mut econ = economics(0.0, 0, 0);
        // 100ms for 10 creatures = 10ms each; a skip saves one creature's screen.
        econ.observe_screen(100, 10);
        assert_eq!(econ.mean_screen_ms(), 10.0);

        let outcome = econ.record_experiment(
            1,
            ExperimentCost {
                proposals: 10,
                skipped: 3,
                entries: 3,
                ..ExperimentCost::default()
            },
        );
        assert_eq!(outcome.saved_ms, 30.0);

        // A slower second batch moves the mean, and the whole ledger with it.
        econ.observe_screen(300, 10);
        assert_eq!(econ.mean_screen_ms(), 20.0);
        assert_eq!(econ.saved_ms(), 60.0, "past skips re-value at the new mean");
    }

    #[test]
    fn promote_savings_need_a_previously_promoted_skip() {
        let mut econ = economics(0.0, 0, 0);
        econ.observe_screen(100, 10); // 10ms/creature
        econ.observe_promote(400, 4); // 100ms/creature

        econ.record_experiment(
            1,
            ExperimentCost {
                skipped: 2,
                skipped_previously_promoted: 0,
                ..ExperimentCost::default()
            },
        );
        assert_eq!(econ.saved_ms(), 20.0, "screen cost only by default");

        econ.record_experiment(
            2,
            ExperimentCost {
                skipped: 1,
                skipped_previously_promoted: 1,
                ..ExperimentCost::default()
            },
        );
        assert_eq!(
            econ.saved_ms(),
            30.0 + 100.0,
            "only the promoted skip claims promote time"
        );
    }

    #[test]
    fn spend_below_a_millisecond_is_not_rounded_away() {
        let mut econ = economics(0.0, 0, 0);
        for _ in 0..10 {
            econ.record_experiment(
                1,
                ExperimentCost {
                    lookup_micros: 250,
                    maintenance_micros: 50,
                    ..ExperimentCost::default()
                },
            );
        }
        assert!(
            (econ.spent_ms() - 3.0).abs() < 1e-9,
            "sub-millisecond overhead must accumulate: {}",
            econ.spent_ms()
        );
    }

    #[test]
    fn the_whole_window_must_lose_before_standing_down() {
        let mut econ = economics(1.0, 3, 0);
        econ.observe_screen(100, 10); // 10ms/creature

        // Two losing experiments: 2.5ms each of pure overhead.
        for exp in 1..=2 {
            econ.record_experiment(
                exp,
                ExperimentCost {
                    lookup_micros: 2_500,
                    ..ExperimentCost::default()
                },
            );
        }
        assert!(!econ.stood_down(), "the window is not full yet");

        // A winning experiment inside the window carries the other two.
        econ.record_experiment(
            3,
            ExperimentCost {
                skipped: 5,
                ..ExperimentCost::default()
            },
        );
        assert!(!econ.stood_down(), "the window as a whole is ahead");

        // Once the win has rolled out of the window, three losses stand it down.
        for exp in 4..=6 {
            econ.record_experiment(
                exp,
                ExperimentCost {
                    lookup_micros: 2_500,
                    ..ExperimentCost::default()
                },
            );
        }
        let stand_down = econ.take_stand_down().expect("the guardrail fires");
        assert!(econ.stood_down());
        assert_eq!(stand_down.experiment_number, 6);
        assert_eq!(stand_down.window_experiments, 3);
        assert_eq!(stand_down.saved_ms, 0.0);
        assert_eq!(stand_down.spent_ms, 7.5, "only the window's own spend");
        assert!(
            stand_down.message().contains("STANDING DOWN"),
            "{}",
            stand_down.message()
        );
        assert!(
            econ.take_stand_down().is_none(),
            "the event is journalled exactly once"
        );
    }

    /// A sunk one-off must not condemn a cache that is currently paying: the
    /// rebuild is judged by the first window and then rolls out of view.
    #[test]
    fn a_sunk_startup_cost_does_not_condemn_a_profitable_cache() {
        let mut econ = economics(1_000.0, 3, 0);
        econ.observe_screen(100, 10); // 10ms/creature
        econ.record_startup_rebuild(3_000); // a slow journal replay

        // The cache earns 200ms an experiment — far more than it spends — but
        // has not yet repaid the 3s rebuild.
        for exp in 1..=6 {
            econ.record_experiment(
                exp,
                ExperimentCost {
                    skipped: 20,
                    lookup_micros: 1_000,
                    ..ExperimentCost::default()
                },
            );
        }
        assert!(
            econ.net_ms() < 0.0,
            "the run has still not repaid the rebuild: {}",
            econ.net_ms()
        );
        assert!(
            !econ.stood_down(),
            "a cache that is currently paying must not be stood down for a sunk cost"
        );
    }

    #[test]
    fn a_zero_window_disables_the_guardrail() {
        let mut econ = economics(0.0, 0, 0);
        for exp in 1..=100 {
            econ.record_experiment(
                exp,
                ExperimentCost {
                    lookup_micros: 10_000,
                    ..ExperimentCost::default()
                },
            );
        }
        assert!(econ.net_ms() < 0.0);
        assert!(!econ.stood_down(), "window 0 opts out of standing down");
    }

    #[test]
    fn the_ceiling_evicts_and_the_bite_is_reportable() {
        let ceiling = 4 * FAILED_CACHE_BYTES_PER_ENTRY;
        let mut econ = economics(0.0, 0, ceiling);
        let mut cache = FailedCandidateCache::new(Tolerance::default(), 1_000, 0);
        for i in 0..10 {
            cache.insert(fingerprint(&format!("knob-{i}")), 1_000 + i);
        }

        let bite = econ
            .enforce_ceiling(&mut cache)
            .expect("ten entries is past a four-entry ceiling");
        assert_eq!(bite.evicted, 6);
        assert_eq!(cache.len(), 4);
        assert!(cache.resident_bytes() <= ceiling);
        assert!(
            bite.message().contains("BYTE CEILING BIT"),
            "{}",
            bite.message()
        );
        assert_eq!(econ.summary().ceiling_bites, 1);

        // The oldest entries are the ones given up.
        assert!(!cache.contains(&fingerprint("knob-0"), 1_010));
        assert!(cache.contains(&fingerprint("knob-9"), 1_010));

        // Under the ceiling there is no bite and nothing is evicted.
        assert!(econ.enforce_ceiling(&mut cache).is_none());
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn a_zero_ceiling_never_evicts() {
        let mut econ = economics(0.0, 0, 0);
        let mut cache = FailedCandidateCache::new(Tolerance::default(), 1_000, 0);
        for i in 0..10 {
            cache.insert(fingerprint(&format!("knob-{i}")), 1_000 + i);
        }
        assert!(econ.enforce_ceiling(&mut cache).is_none());
        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn the_summary_line_carries_every_field() {
        let mut econ = economics(0.0, 0, 0);
        econ.observe_screen(100, 10);
        econ.record_snapshot(2, 4_096);
        econ.record_experiment(
            1,
            ExperimentCost {
                proposals: 8,
                skipped: 2,
                lookup_micros: 1_000,
                entries: 6,
                ..ExperimentCost::default()
            },
        );

        let summary = econ.summary();
        assert_eq!(summary.entries, 6);
        assert!((summary.hit_rate - 0.25).abs() < 1e-12);
        assert_eq!(summary.saved_ms, 20.0);
        assert_eq!(summary.spent_ms, 3.0);
        assert_eq!(summary.net_ms, 17.0);
        assert_eq!(
            summary.peak_resident_bytes,
            6 * FAILED_CACHE_BYTES_PER_ENTRY
        );
        assert_eq!(summary.disk_bytes, 4_096);

        let line = summary.summary_line();
        for field in [
            "entries=",
            "hitRate=",
            "savedMs=",
            "spentMs=",
            "netMs=",
            "peakMemoryBytes=",
            "diskBytes=",
        ] {
            assert!(line.contains(field), "summary line missing {field}: {line}");
        }
    }
}
