//! Cross-experiment memo for incumbent-invariant analysis (issue #106).
//!
//! The incumbent only changes when an experiment is **accepted**, and accepts
//! are rare — `docs/followup-economics.md` records 0 accepts in 118 experiments.
//! For every experiment in between, the creature the analysis phase describes is
//! byte-identical to the one the previous experiment described, so two of the
//! analysis results are pure functions of what has not changed:
//!
//! ```text
//! output MAE                              f(incumbent, sample)
//! focus stats + incoming + ranked sources  f(incumbent, focus, sample)
//! ```
//!
//! This module caches exactly those two, which lets the run loop skip the whole
//! post-focus training scan on a repeated focus. The learning signal is **not**
//! cached: it is driven by a per-experiment seeded rng (`select_sparse`) and is
//! deliberately different every experiment.
//!
//! # Invalidation
//!
//! A stale entry would analyse creature *N* while proposing against creature
//! *N+1* — silently worse candidates, no test failure. Two mechanisms guard it:
//!
//! * every lookup carries its [`MemoScope`], and a scope that differs from the
//!   one the entries were stored under drops all of them before the lookup is
//!   answered. Any incumbent mutation invalidates the memo, not just the ones
//!   with an explicit [`AnalysisMemo::invalidate`] call;
//! * the scope's fingerprint is a **content** hash of the creature. The coarse
//!   `incumbentId` in the journal counts neurons and synapses only, so a
//!   weight-only accept leaves it unchanged — keying on it alone would serve a
//!   stale entry to the very accept the run was looking for.
//!
//! # Memory
//!
//! Focus-dependent entries are capped at [`AnalysisMemo::capacity`] and evicted
//! least-recently-used; the focus-independent entry is a single map per scope.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use neat_core::CreatureExport;

use crate::analysis::PostFocusScan;
use crate::focus::OutputErrorInfluence;

/// Default cap on focus-dependent memo entries held at once.
///
/// A production creature repeats its focus heavily — `docs/baseline-economics.md`
/// records one neuron selected 19 times out of 75 experiments — so a small cap
/// captures nearly all the reuse. Each entry holds one `FocusNeuronStats`, the
/// focus's incoming-source rows and its ranked unused sources, so the whole memo
/// is bounded by the focus fan-in, never by the creature's neuron count.
pub const DEFAULT_ANALYSIS_MEMO_ENTRIES: usize = 16;

/// Content hash of a creature: everything an analysis scan can observe.
///
/// Deliberately finer than the journal's `incumbentId` (which counts neurons and
/// synapses): a weight- or bias-only accept must invalidate the memo.
pub fn creature_fingerprint(creature: &CreatureExport) -> u64 {
    let mut hasher = DefaultHasher::new();
    creature.input.hash(&mut hasher);
    creature.output.hash(&mut hasher);
    creature.forward_only.hash(&mut hasher);
    for neuron in &creature.neurons {
        neuron.neuron_type.hash(&mut hasher);
        neuron.uuid.hash(&mut hasher);
        neuron.bias.to_bits().hash(&mut hasher);
        neuron.squash.hash(&mut hasher);
    }
    for synapse in &creature.synapses {
        synapse.from_uuid.hash(&mut hasher);
        synapse.to_uuid.hash(&mut hasher);
        synapse.weight.to_bits().hash(&mut hasher);
        synapse.synapse_type.hash(&mut hasher);
    }
    hasher.finish()
}

/// Identity of the creature and analysis sample a memo entry describes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoScope {
    /// Journal `incumbentId` of the creature (coarse; for the runtime check).
    pub incumbent_id: String,
    /// Content fingerprint of the creature (the key that actually guards reuse).
    pub fingerprint: u64,
    /// Analysis sample configuration: stats mode, record cap, training path.
    pub sample: String,
}

impl MemoScope {
    /// Scope for `creature` analysed under the `sample` configuration.
    pub fn new(incumbent_id: impl Into<String>, creature: &CreatureExport, sample: &str) -> Self {
        Self {
            incumbent_id: incumbent_id.into(),
            fingerprint: creature_fingerprint(creature),
            sample: sample.to_string(),
        }
    }
}

/// Memo hit / miss accounting, journalled per experiment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoStats {
    /// Lookups answered from the memo.
    pub hits: u64,
    /// Lookups that had to be computed.
    pub misses: u64,
    /// Scan milliseconds avoided, measured on the miss that stored the entry.
    pub ms_saved: u128,
}

impl MemoStats {
    /// Counters accumulated since `earlier` (per-experiment deltas).
    pub fn since(self, earlier: Self) -> Self {
        Self {
            hits: self.hits.saturating_sub(earlier.hits),
            misses: self.misses.saturating_sub(earlier.misses),
            ms_saved: self.ms_saved.saturating_sub(earlier.ms_saved),
        }
    }
}

/// A cached value plus what it cost to produce.
#[derive(Debug, Clone)]
struct Entry<T> {
    value: T,
    compute_ms: u128,
}

/// Memo of the incumbent-invariant analysis results (issue #106).
#[derive(Debug)]
pub struct AnalysisMemo {
    capacity: usize,
    scope: Option<MemoScope>,
    output_errors: Option<Entry<HashMap<String, OutputErrorInfluence>>>,
    /// Focus-keyed entries, least-recently-used first.
    post_focus: Vec<(String, Entry<PostFocusScan>)>,
    stats: MemoStats,
    invalidations: u64,
}

impl AnalysisMemo {
    /// Memo holding at most `capacity` focus-dependent entries (`0` disables it).
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            scope: None,
            output_errors: None,
            post_focus: Vec::new(),
            stats: MemoStats::default(),
            invalidations: 0,
        }
    }

    /// Cap on focus-dependent entries.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Whether the memo stores anything at all.
    pub fn is_enabled(&self) -> bool {
        self.capacity > 0
    }

    /// Entries currently held (focus-dependent plus the focus-independent one).
    pub fn len(&self) -> usize {
        self.post_focus.len() + usize::from(self.output_errors.is_some())
    }

    /// Whether the memo currently holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Scope the held entries were stored under.
    pub fn scope(&self) -> Option<&MemoScope> {
        self.scope.as_ref()
    }

    /// Cumulative hit / miss / ms-saved counters.
    pub fn stats(&self) -> MemoStats {
        self.stats
    }

    /// How many times the held entries have been dropped.
    pub fn invalidations(&self) -> u64 {
        self.invalidations
    }

    /// Drop every entry — call on accept, on graft application, and on any other
    /// path that mutates the incumbent.
    ///
    /// Scope changes are caught automatically at lookup time; this is the
    /// explicit belt-and-braces call at the known mutation sites.
    pub fn invalidate(&mut self) {
        if self.scope.is_some() || !self.is_empty() {
            self.invalidations += 1;
        }
        self.scope = None;
        self.output_errors = None;
        self.post_focus.clear();
    }

    /// Re-key onto `scope`, dropping entries that describe a different creature
    /// or a different analysis sample.
    fn enter(&mut self, scope: &MemoScope) {
        if self.scope.as_ref() == Some(scope) {
            return;
        }
        if self.scope.is_some() {
            self.invalidate();
        }
        self.scope = Some(scope.clone());
    }

    /// Cached per-output MAE for this scope, if any.
    pub fn output_errors(
        &mut self,
        scope: &MemoScope,
    ) -> Option<HashMap<String, OutputErrorInfluence>> {
        if !self.is_enabled() {
            return None;
        }
        self.enter(scope);
        match &self.output_errors {
            Some(entry) => {
                self.stats.hits += 1;
                self.stats.ms_saved = self.stats.ms_saved.saturating_add(entry.compute_ms);
                Some(entry.value.clone())
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Store the per-output MAE computed for this scope in `compute_ms`.
    pub fn store_output_errors(
        &mut self,
        scope: &MemoScope,
        errors: HashMap<String, OutputErrorInfluence>,
        compute_ms: u128,
    ) {
        if !self.is_enabled() {
            return;
        }
        self.enter(scope);
        self.output_errors = Some(Entry {
            value: errors,
            compute_ms,
        });
    }

    /// Cached post-focus scan for this scope and focus neuron, if any.
    pub fn post_focus(&mut self, scope: &MemoScope, focus: &str) -> Option<PostFocusScan> {
        if !self.is_enabled() {
            return None;
        }
        self.enter(scope);
        let Some(index) = self.post_focus.iter().position(|(uuid, _)| uuid == focus) else {
            self.stats.misses += 1;
            return None;
        };
        // Touch: most-recently-used entries move to the back, so eviction takes
        // the front — the focus nobody has asked for in the longest.
        let entry = self.post_focus.remove(index);
        self.stats.hits += 1;
        self.stats.ms_saved = self.stats.ms_saved.saturating_add(entry.1.compute_ms);
        let value = entry.1.value.clone();
        self.post_focus.push(entry);
        Some(value)
    }

    /// Store the post-focus scan computed for this scope and focus in `compute_ms`.
    pub fn store_post_focus(
        &mut self,
        scope: &MemoScope,
        focus: &str,
        scan: PostFocusScan,
        compute_ms: u128,
    ) {
        if !self.is_enabled() {
            return;
        }
        self.enter(scope);
        self.post_focus.retain(|(uuid, _)| uuid != focus);
        self.post_focus.push((
            focus.to_string(),
            Entry {
                value: scan,
                compute_ms,
            },
        ));
        while self.post_focus.len() > self.capacity {
            self.post_focus.remove(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{ScanBudget, scan_post_focus};
    use neat_core::{compile_creature, parse_creature_json};
    use std::io::Write;
    use tempfile::{TempDir, tempdir};

    const CREATURE: &str = r#"{
      "semanticVersion": "4.0.0",
      "forwardOnly": true,
      "input": 3,
      "output": 1,
      "neurons": [
        {"type":"hidden","uuid":"h1","bias":0.1,"squash":"TANH"},
        {"type":"hidden","uuid":"h2","bias":-0.2,"squash":"IDENTITY"},
        {"type":"output","uuid":"o1","bias":0.05,"squash":"IDENTITY"}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h1","weight":0.7},
        {"fromUUID":"input-1","toUUID":"h2","weight":-0.4},
        {"fromUUID":"h1","toUUID":"o1","weight":0.9},
        {"fromUUID":"h2","toUUID":"o1","weight":0.3}
      ]
    }"#;

    fn write_sample(records: usize) -> TempDir {
        let dir = tempdir().unwrap();
        let mut f = std::fs::File::create(dir.path().join("0.bin")).unwrap();
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..records {
            for _ in 0..4 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let v = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        f.flush().unwrap();
        dir
    }

    fn scope_for(json: &str) -> (CreatureExport, MemoScope) {
        let creature = parse_creature_json(json).unwrap();
        let scope = MemoScope::new("in3-out1-n3-s4", &creature, "quick:64:/data");
        (creature, scope)
    }

    fn empty_scan() -> PostFocusScan {
        PostFocusScan {
            focus_stats: Default::default(),
            incoming: Vec::new(),
            ranked_sources: Vec::new(),
        }
    }

    /// Acceptance criterion: a hit returns exactly what a fresh scan produces.
    #[test]
    fn a_hit_returns_exactly_what_a_fresh_scan_produced() {
        let dir = write_sample(64);
        let (creature, scope) = scope_for(CREATURE);
        let mut network = compile_creature(&creature).unwrap();
        let fresh = scan_post_focus(
            &creature,
            &mut network,
            dir.path(),
            "o1",
            ScanBudget::serial(Some(40)),
            None,
            &[],
        )
        .unwrap();

        let mut memo = AnalysisMemo::new(DEFAULT_ANALYSIS_MEMO_ENTRIES);
        assert!(memo.post_focus(&scope, "o1").is_none(), "cold memo misses");
        memo.store_post_focus(&scope, "o1", fresh.clone(), 12);
        let hit = memo.post_focus(&scope, "o1").expect("warm memo hits");

        // Recomputing must agree with the memo, field for field.
        let mut network = compile_creature(&creature).unwrap();
        let recomputed = scan_post_focus(
            &creature,
            &mut network,
            dir.path(),
            "o1",
            ScanBudget::serial(Some(40)),
            None,
            &[],
        )
        .unwrap();
        assert_eq!(format!("{recomputed:?}"), format!("{hit:?}"));
        assert_eq!(format!("{fresh:?}"), format!("{hit:?}"));
        assert_eq!(memo.stats().hits, 1);
        assert_eq!(memo.stats().misses, 1);
        assert_eq!(memo.stats().ms_saved, 12, "a hit banks the miss's cost");
    }

    #[test]
    fn a_structural_change_invalidates_every_entry() {
        let (_, scope) = scope_for(CREATURE);
        let grown = CREATURE.replace(
            r#"{"fromUUID":"h2","toUUID":"o1","weight":0.3}"#,
            r#"{"fromUUID":"h2","toUUID":"o1","weight":0.3},
               {"fromUUID":"input-2","toUUID":"o1","weight":0.11}"#,
        );
        let (_, next) = scope_for(&grown);

        let mut memo = AnalysisMemo::new(4);
        memo.store_post_focus(&scope, "o1", empty_scan(), 5);
        memo.store_output_errors(&scope, HashMap::new(), 3);
        assert_eq!(memo.len(), 2);

        assert!(
            memo.post_focus(&next, "o1").is_none(),
            "new creature misses"
        );
        assert!(memo.output_errors(&next).is_none());
        assert_eq!(memo.invalidations(), 1);
    }

    /// The coarse `incumbentId` cannot see a weight-only accept — the content
    /// fingerprint must, or the memo serves stale analysis after every nudge.
    #[test]
    fn a_weight_only_change_invalidates_even_though_the_incumbent_id_matches() {
        let (_, scope) = scope_for(CREATURE);
        let nudged_json = CREATURE.replace(r#""weight":0.9"#, r#""weight":0.9000001"#);
        let (_, nudged) = scope_for(&nudged_json);
        assert_eq!(
            scope.incumbent_id, nudged.incumbent_id,
            "neuron/synapse counts are unchanged — the coarse id cannot help"
        );
        assert_ne!(scope.fingerprint, nudged.fingerprint);

        let mut memo = AnalysisMemo::new(4);
        memo.store_post_focus(&scope, "o1", empty_scan(), 5);
        assert!(memo.post_focus(&nudged, "o1").is_none());
        assert!(memo.is_empty());
    }

    #[test]
    fn a_changed_sample_configuration_invalidates() {
        let (creature, scope) = scope_for(CREATURE);
        let wider = MemoScope::new("in3-out1-n3-s4", &creature, "quick:5000:/data");

        let mut memo = AnalysisMemo::new(4);
        memo.store_post_focus(&scope, "o1", empty_scan(), 5);
        assert!(
            memo.post_focus(&wider, "o1").is_none(),
            "a different --quick-sample-records is a different scan"
        );
    }

    #[test]
    fn an_explicit_invalidate_drops_the_entries() {
        let (_, scope) = scope_for(CREATURE);
        let mut memo = AnalysisMemo::new(4);
        memo.store_post_focus(&scope, "o1", empty_scan(), 5);
        memo.store_output_errors(&scope, HashMap::new(), 5);

        memo.invalidate();

        assert!(memo.is_empty());
        assert!(memo.scope().is_none());
        assert_eq!(memo.invalidations(), 1);
        assert!(memo.post_focus(&scope, "o1").is_none());
    }

    #[test]
    fn the_entry_cap_bounds_growth_across_many_focus_neurons() {
        let (_, scope) = scope_for(CREATURE);
        let mut memo = AnalysisMemo::new(3);
        for i in 0..50 {
            memo.store_post_focus(&scope, &format!("neuron-{i}"), empty_scan(), 1);
            assert!(memo.post_focus.len() <= 3, "cap holds at iteration {i}");
        }
        assert_eq!(memo.post_focus.len(), 3);
        assert!(
            memo.post_focus(&scope, "neuron-49").is_some(),
            "the newest focus survives"
        );
        assert!(
            memo.post_focus(&scope, "neuron-0").is_none(),
            "the oldest focus was evicted"
        );
    }

    #[test]
    fn eviction_takes_the_least_recently_used_focus() {
        let (_, scope) = scope_for(CREATURE);
        let mut memo = AnalysisMemo::new(2);
        memo.store_post_focus(&scope, "a", empty_scan(), 1);
        memo.store_post_focus(&scope, "b", empty_scan(), 1);
        // Touch `a` so `b` becomes the least-recently-used entry.
        assert!(memo.post_focus(&scope, "a").is_some());
        memo.store_post_focus(&scope, "c", empty_scan(), 1);

        assert!(
            memo.post_focus(&scope, "a").is_some(),
            "recently used stays"
        );
        assert!(memo.post_focus(&scope, "c").is_some());
        assert!(memo.post_focus(&scope, "b").is_none(), "LRU entry evicted");
    }

    #[test]
    fn a_disabled_memo_never_stores_hits_or_counts() {
        let (_, scope) = scope_for(CREATURE);
        let mut memo = AnalysisMemo::new(0);
        memo.store_post_focus(&scope, "o1", empty_scan(), 5);
        memo.store_output_errors(&scope, HashMap::new(), 5);

        assert!(!memo.is_enabled());
        assert!(memo.is_empty());
        assert!(memo.post_focus(&scope, "o1").is_none());
        assert!(memo.output_errors(&scope).is_none());
        assert_eq!(
            memo.stats(),
            MemoStats::default(),
            "memo off journals zeros"
        );
    }

    #[test]
    fn per_experiment_deltas_subtract_the_previous_snapshot() {
        let (_, scope) = scope_for(CREATURE);
        let mut memo = AnalysisMemo::new(4);
        memo.store_post_focus(&scope, "o1", empty_scan(), 7);
        assert!(memo.post_focus(&scope, "o1").is_some());
        let after_first = memo.stats();

        assert!(memo.post_focus(&scope, "o1").is_some());
        let delta = memo.stats().since(after_first);
        assert_eq!(
            delta,
            MemoStats {
                hits: 1,
                misses: 0,
                ms_saved: 7,
            }
        );
    }
}
