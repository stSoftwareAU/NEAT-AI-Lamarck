//! Parallel combination scoring for improving singles (experiments + graft replay).
//!
//! When several candidates beat the baseline alone, merge their deltas onto the
//! incumbent in groups (pairs, then triples, …) and score those creatures in one
//! scorer directory batch — up to [`MAX_COMBO_CANDIDATES`] including the singles.
//!
//! Newly stacked synapses into the same target across **combo members** are
//! dampened by [`stack_dampen_scale`] so solo-sized weights do not overshoot
//! when combined (#63). The exponent is [`STACK_DAMPEN_EXPONENT`] — tune after
//! production evidence. `k` is the number of members that contribute ≥1 new
//! edge to a target (not the raw new-edge count inside one multi-synapse
//! candidate).

use crate::candidates::Candidate;
use crate::log;
use crate::scorer::{
    DirectoryScorer, ScoreResult, accepts_improvement, accepts_improvement_delta, improvement,
};
use crate::structural::{insert_index_for_hidden, insert_synapse_ordered};
use crate::width::checked_creature_json_pretty;
use neat_core::{CreatureExport, NeuronExport, SynapseExport};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// Max improving singles + combination creatures scored together.
pub const MAX_COMBO_CANDIDATES: usize = 50;

/// Exponent for stacked-new-synapse dampening: scale = `k.powf(-STACK_DAMPEN_EXPONENT)`.
///
/// - `0.5` → `1/√k` (default; partially independent sources)
/// - `1.0` → `1/k` (fully redundant residual correction)
///
/// Change after reviewing journal `comboDampen` / `combosDampened` fields from runs.
pub const STACK_DAMPEN_EXPONENT: f64 = 0.5;

/// Dampen multiplier for `k` new synapses into the same target.
///
/// Returns `1.0` when `k <= 1` (no stacking). Otherwise `k.powf(-STACK_DAMPEN_EXPONENT)`.
pub fn stack_dampen_scale(k: usize) -> f64 {
    if k <= 1 {
        1.0
    } else {
        (k as f64).powf(-STACK_DAMPEN_EXPONENT)
    }
}

/// One full-corpus improver from a scored candidate batch.
#[derive(Debug, Clone)]
pub struct Improver {
    /// `candidate-NNN` stem.
    pub stem: String,
    /// Candidate index into the generation batch.
    pub index: usize,
    /// Authoritative score result.
    pub result: ScoreResult,
    /// Score Δ vs baseline.
    pub delta: f64,
}

/// Best creature chosen among singles and scored combinations.
#[derive(Debug, Clone)]
pub struct ComboSelection {
    /// Winning creature JSON (already on disk under `creature_path`).
    pub creature_path: std::path::PathBuf,
    /// Stem used in the journal (`candidate-NNN` or `combo-NNN`).
    pub stem: String,
    /// Score of the winner.
    pub result: ScoreResult,
    /// Δ vs the full-corpus incumbent scored in the **same call** as
    /// [`Self::result`] — the promote call for a single, the combo call for a
    /// combo (issue #130). This is the number the accept gate reads;
    /// [`Self::result`] and a baseline from another call are not comparable.
    pub delta: f64,
    /// Candidate indices merged into this winner (length 1 for a pure single).
    pub member_indices: Vec<usize>,
    /// Per-target dampening applied to the winning creature (empty if none).
    pub dampen: StackDampenReport,
    /// Combination creatures scored in this selection (0 when only a single).
    pub combos_scored: usize,
    /// How many of those combos had at least one stacked-target dampen.
    pub combos_dampened: usize,
}

impl ComboSelection {
    /// True when this selection's Δ clears the accept bar (issue #130).
    ///
    /// The gate reads [`ComboSelection::delta`] rather than re-subtracting
    /// [`ComboSelection::result`] from a baseline, because a combo winner was
    /// scored in the combo call and the promote-call baseline is not a
    /// comparable number (`docs/scorer-batch-composition.md`).
    pub fn accepts(&self, min_improvement: f64) -> bool {
        accepts_improvement_delta(self.delta, min_improvement)
    }
}

/// Index sets for combinations of size `>= 2` over `0..n`, up to `max_combos`.
///
/// Smaller groups first (all pairs, then triples, …) so a tight budget still
/// explores pairwise interactions before larger sets.
pub fn combination_index_sets(n: usize, max_combos: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if n < 2 || max_combos == 0 {
        return out;
    }
    for k in 2..=n {
        let mut cur = Vec::with_capacity(k);
        choose_indices(n, k, 0, &mut cur, &mut out, max_combos);
        if out.len() >= max_combos {
            break;
        }
    }
    out
}

fn choose_indices(
    n: usize,
    k: usize,
    start: usize,
    cur: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
    max_combos: usize,
) {
    if out.len() >= max_combos {
        return;
    }
    if cur.len() == k {
        out.push(cur.clone());
        return;
    }
    let need = k - cur.len();
    for i in start..=n - need {
        cur.push(i);
        choose_indices(n, k, i + 1, cur, out, max_combos);
        cur.pop();
        if out.len() >= max_combos {
            return;
        }
    }
}

/// Collect `candidate-*` stems that beat baseline by more than `min_improvement`.
///
/// Sorted by descending Δ.
pub fn collect_improvers(
    scores: &BTreeMap<String, ScoreResult>,
    min_improvement: f64,
) -> Result<Vec<Improver>, String> {
    let baseline = scores
        .get("baseline")
        .ok_or_else(|| "baseline missing from scorer results".to_string())?;
    Ok(collect_improvers_against(scores, baseline, min_improvement))
}

/// Collect improvers against a baseline scored outside this batch (issue #113).
///
/// A promote call that reused a remembered baseline returns no `baseline` stem,
/// so the score to beat is supplied rather than looked up. Treating a missing
/// baseline as `0.0` would promote every candidate, so there is deliberately no
/// default here.
pub fn collect_improvers_against(
    scores: &BTreeMap<String, ScoreResult>,
    baseline: &ScoreResult,
    min_improvement: f64,
) -> Vec<Improver> {
    let mut out = Vec::new();
    for (stem, result) in scores {
        if stem == "baseline" || !stem.starts_with("candidate-") {
            continue;
        }
        let delta = improvement(result.score, baseline.score);
        if !accepts_improvement(result.score, baseline.score, min_improvement) {
            continue;
        }
        let Some(idx_str) = stem.strip_prefix("candidate-") else {
            continue;
        };
        let Ok(index) = idx_str.parse::<usize>() else {
            continue;
        };
        out.push(Improver {
            stem: stem.clone(),
            index,
            result: result.clone(),
            delta,
        });
    }
    out.sort_by(|a, b| {
        b.delta
            .partial_cmp(&a.delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn neuron_changed(base: &NeuronExport, other: &NeuronExport) -> bool {
    (base.bias - other.bias).abs() > 0.0
        || base.squash != other.squash
        || base.neuron_type != other.neuron_type
}

fn synapse_key(s: &SynapseExport) -> (String, String) {
    (s.from_uuid.clone(), s.to_uuid.clone())
}

/// Merge mutation deltas from `variants` (each a near-clone of `base`) onto `base`.
///
/// Conflicts (two variants changing the same neuron/edge to different values)
/// return `Err` so that combo is skipped.
pub fn merge_candidate_deltas(
    base: &CreatureExport,
    variants: &[&CreatureExport],
) -> Result<CreatureExport, String> {
    let mut out = base.clone();
    for (vi, variant) in variants.iter().enumerate() {
        // New or changed neurons.
        for vn in &variant.neurons {
            if let Some(bn) = base.neurons.iter().find(|n| n.uuid == vn.uuid) {
                if !neuron_changed(bn, vn) {
                    continue;
                }
                if let Some(existing) = out.neurons.iter_mut().find(|n| n.uuid == vn.uuid) {
                    if neuron_changed(bn, existing)
                        && (existing.bias != vn.bias || existing.squash != vn.squash)
                    {
                        return Err(format!(
                            "neuron {} conflict across combo members (at variant {vi})",
                            vn.uuid
                        ));
                    }
                    existing.bias = vn.bias;
                    existing.squash = vn.squash.clone();
                }
            } else if !out.neurons.iter().any(|n| n.uuid == vn.uuid) {
                // New hidden: insert before its focus (outgoing edge to an existing uuid).
                let focus = variant
                    .synapses
                    .iter()
                    .find(|s| s.from_uuid == vn.uuid)
                    .map(|s| s.to_uuid.as_str())
                    .ok_or_else(|| {
                        format!("new neuron {} has no outgoing synapse for insert", vn.uuid)
                    })?;
                let insert_at = insert_index_for_hidden(&out, focus).ok_or_else(|| {
                    format!("cannot insert new neuron {} before focus {focus}", vn.uuid)
                })?;
                out.neurons.insert(
                    insert_at,
                    NeuronExport {
                        // Faithful copy of the variant's neuron — keep whatever
                        // runtime id it arrived with (neat-core 0.9.9).
                        id: vn.id,
                        neuron_type: vn.neuron_type.clone(),
                        uuid: vn.uuid.clone(),
                        bias: vn.bias,
                        squash: vn.squash.clone(),
                    },
                );
            }
        }

        // New or changed synapses.
        for vs in &variant.synapses {
            let key = synapse_key(vs);
            let base_syn = base
                .synapses
                .iter()
                .find(|s| s.from_uuid == key.0 && s.to_uuid == key.1);
            if let Some(bs) = base_syn {
                if (bs.weight - vs.weight).abs() == 0.0 && bs.synapse_type == vs.synapse_type {
                    continue;
                }
                if let Some(existing) = out
                    .synapses
                    .iter_mut()
                    .find(|s| s.from_uuid == key.0 && s.to_uuid == key.1)
                {
                    if (existing.weight - bs.weight).abs() > 0.0
                        && (existing.weight - vs.weight).abs() > 0.0
                    {
                        return Err(format!(
                            "synapse {}->{} conflict across combo members (at variant {vi})",
                            key.0, key.1
                        ));
                    }
                    existing.weight = vs.weight;
                    existing.synapse_type = vs.synapse_type.clone();
                }
            } else if !out
                .synapses
                .iter()
                .any(|s| s.from_uuid == key.0 && s.to_uuid == key.1)
            {
                // Ordered insert, never a tail append: `creature_validate`
                // rule 25 requires the compiled `(from, to)` order, and a
                // merged creature that breaks it is refused by every Rust
                // consumer while still loading and scoring elsewhere — which
                // is how `GRQ-23-lamarck.json` was published with
                // `SORT_FAILURE: 28564) synapses not sorted` and nobody
                // noticed (GRQ #4428). The single-edit paths in
                // `structural.rs` have used `insert_synapse_ordered` since
                // #192; the combo merge was the one that still appended.
                insert_synapse_ordered(
                    &mut out,
                    SynapseExport {
                        from_uuid: vs.from_uuid.clone(),
                        to_uuid: vs.to_uuid.clone(),
                        weight: vs.weight,
                        synapse_type: vs.synapse_type.clone(),
                    },
                );
            }
        }
    }
    Ok(out)
}

/// One target neuron that received `k > 1` new synapses in a combo merge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StackDampenTarget {
    /// Destination neuron UUID for the stacked new edges.
    pub to_uuid: String,
    /// Number of combo members that contributed ≥1 new synapse into this target.
    pub k: usize,
    /// Multiplier applied to each new weight (`stack_dampen_scale(k)`).
    pub scale: f64,
    /// Sum of absolute new-synapse weights before dampening.
    pub weight_abs_sum_before: f64,
    /// Sum of absolute new-synapse weights after dampening.
    pub weight_abs_sum_after: f64,
}

/// Report of per-target new-synapse dampening after a combo merge.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StackDampenReport {
    /// Targets that were scaled (`k > 1` only).
    pub targets: Vec<StackDampenTarget>,
}

impl StackDampenReport {
    /// Whether any target was dampened.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

/// Count how many `variants` each contribute ≥1 new synapse into each target.
///
/// Used so [`dampen_stacked_new_synapses`] scales by combo-member stacking, not
/// by how many edges a single multi-synapse candidate already carries (e.g. a
/// two-source neuron bridge).
pub fn new_synapse_contributor_counts(
    base: &CreatureExport,
    variants: &[&CreatureExport],
) -> BTreeMap<String, usize> {
    let base_keys: BTreeSet<(String, String)> = base
        .synapses
        .iter()
        .map(|s| (s.from_uuid.clone(), s.to_uuid.clone()))
        .collect();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for variant in variants {
        let mut targets = BTreeSet::new();
        for s in &variant.synapses {
            let key = (s.from_uuid.clone(), s.to_uuid.clone());
            if base_keys.contains(&key) {
                continue;
            }
            targets.insert(s.to_uuid.clone());
        }
        for to_uuid in targets {
            *counts.entry(to_uuid).or_default() += 1;
        }
    }
    counts
}

/// Scale newly added synapses that stack across combo members by [`stack_dampen_scale`].
///
/// `contributor_counts` maps `to_uuid →` number of combo members that added at
/// least one new synapse into that target (see [`new_synapse_contributor_counts`]).
/// Only edges absent from `base` are scaled. Bias tweaks and existing-edge
/// weight changes are untouched. Targets with `k <= 1` (or missing from the
/// map) are left alone.
pub fn dampen_stacked_new_synapses(
    base: &CreatureExport,
    creature: &mut CreatureExport,
    contributor_counts: &BTreeMap<String, usize>,
) -> StackDampenReport {
    let base_keys: BTreeSet<(String, String)> = base
        .synapses
        .iter()
        .map(|s| (s.from_uuid.clone(), s.to_uuid.clone()))
        .collect();

    // to_uuid -> indices of new synapses in `creature.synapses`.
    let mut by_target: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, s) in creature.synapses.iter().enumerate() {
        let key = (s.from_uuid.clone(), s.to_uuid.clone());
        if base_keys.contains(&key) {
            continue;
        }
        by_target.entry(s.to_uuid.clone()).or_default().push(i);
    }

    let mut report = StackDampenReport::default();
    for (to_uuid, indices) in by_target {
        let k = contributor_counts.get(&to_uuid).copied().unwrap_or(1);
        let scale = stack_dampen_scale(k);
        if (scale - 1.0).abs() < f64::EPSILON {
            continue;
        }
        let mut weight_abs_sum_before = 0.0;
        let mut weight_abs_sum_after = 0.0;
        for i in indices {
            let before = creature.synapses[i].weight;
            weight_abs_sum_before += before.abs();
            let after = before * scale;
            creature.synapses[i].weight = after;
            weight_abs_sum_after += after.abs();
        }
        report.targets.push(StackDampenTarget {
            to_uuid,
            k,
            scale,
            weight_abs_sum_before,
            weight_abs_sum_after,
        });
    }
    report
}

/// Format a dampen report for experiment / graft combo logs.
pub fn format_dampen_report(prefix: &str, report: &StackDampenReport) -> Option<String> {
    if report.is_empty() {
        return None;
    }
    let parts: Vec<String> = report
        .targets
        .iter()
        .map(|t| {
            format!(
                "{} k={} scale={:.3} |w|{:.4}->{:.4}",
                t.to_uuid, t.k, t.scale, t.weight_abs_sum_before, t.weight_abs_sum_after
            )
        })
        .collect();
    Some(format!("{prefix}: {}", parts.join(", ")))
}

/// Write a creature for the scorer — refused unless the written text carries
/// a `>= 1` observation width (issue #165).
fn write_creature_json(path: &Path, creature: &CreatureExport) -> Result<(), String> {
    let json = checked_creature_json_pretty(creature)?;
    fs::write(path, json).map_err(|e| e.to_string())
}

/// Inputs for [`select_best_with_combinations`].
pub struct ComboSelectRequest<'a> {
    /// Training-data directory for the scorer.
    pub training_data: &'a Path,
    /// Current incumbent (merge base).
    pub incumbent: &'a CreatureExport,
    /// Generated candidates for this experiment.
    pub candidates: &'a [Candidate],
    /// Full-corpus `candidate-*` scores. A `baseline` stem is ignored — the
    /// score to beat is [`Self::baseline`], which may have been measured by an
    /// earlier call (issue #113).
    pub scores: &'a BTreeMap<String, ScoreResult>,
    /// Authoritative full-corpus score of the incumbent, measured in the same
    /// call as [`Self::scores`]. It shortlists the improving singles; the
    /// combos are judged against the incumbent re-scored inside the combo call
    /// itself (issue #130).
    pub baseline: &'a ScoreResult,
    /// Absolute score Δ required for acceptance.
    pub min_improvement: f64,
    /// Directory holding scored `candidate-*.json` files.
    pub source_dir: &'a Path,
    /// Working directory for combination JSON + scoring.
    pub combo_work_dir: &'a Path,
}

/// Among full-corpus improvers, score combinations in parallel and pick the best.
///
/// Returns `None` when there is nothing better than (or no) improving single.
pub fn select_best_with_combinations(
    scorer: &impl DirectoryScorer,
    request: ComboSelectRequest<'_>,
) -> Result<Option<ComboSelection>, String> {
    let ComboSelectRequest {
        training_data,
        incumbent,
        candidates,
        scores,
        baseline,
        min_improvement,
        source_dir,
        combo_work_dir,
    } = request;
    let improvers = collect_improvers_against(scores, baseline, min_improvement);
    if improvers.is_empty() {
        return Ok(None);
    }

    let best_single = &improvers[0];
    let mut best = ComboSelection {
        creature_path: source_dir.join(format!("{}.json", best_single.stem)),
        stem: best_single.stem.clone(),
        result: best_single.result.clone(),
        delta: best_single.delta,
        member_indices: vec![best_single.index],
        dampen: StackDampenReport::default(),
        combos_scored: 0,
        combos_dampened: 0,
    };

    if improvers.len() < 2 {
        return Ok(Some(best));
    }

    let combo_slots = MAX_COMBO_CANDIDATES.saturating_sub(improvers.len());
    let index_sets = combination_index_sets(improvers.len(), combo_slots);
    if index_sets.is_empty() {
        return Ok(Some(best));
    }

    if combo_work_dir.exists() {
        fs::remove_dir_all(combo_work_dir).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(combo_work_dir).map_err(|e| e.to_string())?;
    write_creature_json(&combo_work_dir.join("baseline.json"), incumbent)?;

    let mut stem_to_members: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut stem_to_dampen: BTreeMap<String, StackDampenReport> = BTreeMap::new();
    let mut combos_dampened = 0usize;
    for (ci, idxs) in index_sets.iter().enumerate() {
        let mut variants: Vec<&CreatureExport> = Vec::with_capacity(idxs.len());
        let mut member_indices = Vec::with_capacity(idxs.len());
        for &ii in idxs {
            let improver = &improvers[ii];
            let Some(cand) = candidates.get(improver.index) else {
                continue;
            };
            variants.push(&cand.creature);
            member_indices.push(improver.index);
        }
        if variants.len() != idxs.len() {
            continue;
        }
        let Ok(mut merged) = merge_candidate_deltas(incumbent, &variants) else {
            continue;
        };
        let stem = format!("combo-{ci:03}-k{}", idxs.len());
        let contributors = new_synapse_contributor_counts(incumbent, &variants);
        let dampen = dampen_stacked_new_synapses(incumbent, &mut merged, &contributors);
        if let Some(msg) = format_dampen_report(&format!("combo dampen {stem}"), &dampen) {
            log::detail(&msg);
        }
        if !dampen.is_empty() {
            combos_dampened += 1;
        }
        write_creature_json(&combo_work_dir.join(format!("{stem}.json")), &merged)?;
        stem_to_members.insert(stem.clone(), member_indices);
        stem_to_dampen.insert(stem, dampen);
    }

    if stem_to_members.is_empty() {
        let _ = fs::remove_dir_all(combo_work_dir);
        return Ok(Some(best));
    }

    let combos_scored = stem_to_members.len();
    best.combos_scored = combos_scored;
    best.combos_dampened = combos_dampened;

    log::info(&format!(
        "combo: scoring {} combination(s) in parallel (improvers={}, budget={}, dampened={})",
        combos_scored,
        improvers.len(),
        MAX_COMBO_CANDIDATES,
        combos_dampened
    ));

    let combo_scores = scorer
        .score_directory(combo_work_dir, training_data)
        .map_err(|e| e.to_string())?;

    // Issue #130: the combo call scores the incumbent alongside the combos, and
    // that is the only baseline its scores may be subtracted from. Directory
    // scoring is reproducible for a fixed batch composition but not across
    // compositions, so the promote call's baseline — measured beside a different
    // number of creatures — is not a comparable number here.
    let combo_baseline = combo_scores.get("baseline").ok_or_else(|| {
        format!(
            "combo call returned no 'baseline' score for the incumbent scored in {}",
            combo_work_dir.display()
        )
    })?;

    for (stem, members) in &stem_to_members {
        let Some(result) = combo_scores.get(stem) else {
            continue;
        };
        let delta = improvement(result.score, combo_baseline.score);
        if accepts_improvement(result.score, combo_baseline.score, min_improvement)
            && delta > best.delta
        {
            best = ComboSelection {
                creature_path: combo_work_dir.join(format!("{stem}.json")),
                stem: stem.clone(),
                result: result.clone(),
                delta,
                member_indices: members.clone(),
                dampen: stem_to_dampen.get(stem).cloned().unwrap_or_default(),
                combos_scored,
                combos_dampened,
            };
            log::ok(&format!(
                "combo {stem}: score={:.12} (+{:.3e}, {} members{})",
                best.result.score,
                best.delta,
                best.member_indices.len(),
                if best.dampen.is_empty() {
                    String::new()
                } else {
                    format!(", dampen targets={}", best.dampen.targets.len())
                }
            ));
        }
    }

    Ok(Some(best))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::{Candidate, CandidateProvenance, CandidateStrategy};
    use crate::scorer::ScoreSample;
    use neat_core::parse_creature_json;
    use tempfile::tempdir;

    fn tiny() -> CreatureExport {
        parse_creature_json(
            r#"{
              "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
              "neurons":[
                {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
                {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
              ],
              "synapses":[
                {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
                {"fromUUID":"h1","toUUID":"o1","weight":1.0}
              ]
            }"#,
        )
        .unwrap()
    }

    fn cand(creature: CreatureExport) -> Candidate {
        Candidate {
            creature,
            provenance: CandidateProvenance {
                strategy: CandidateStrategy::StructuralAdd,
                focus_neuron: "h1".into(),
                mutation: "test".into(),
                old_value: None,
                new_value: None,
            },
        }
    }

    #[test]
    fn stack_dampen_scale_matches_exponent_contract() {
        // Tunable contract: k<=1 → 1; else k^(-STACK_DAMPEN_EXPONENT).
        assert_eq!(stack_dampen_scale(0), 1.0);
        assert_eq!(stack_dampen_scale(1), 1.0);
        assert!((stack_dampen_scale(2) - 2f64.powf(-STACK_DAMPEN_EXPONENT)).abs() < 1e-12);
        assert!((stack_dampen_scale(3) - 3f64.powf(-STACK_DAMPEN_EXPONENT)).abs() < 1e-12);
        assert!((stack_dampen_scale(4) - 4f64.powf(-STACK_DAMPEN_EXPONENT)).abs() < 1e-12);
        // Default exponent 0.5 → 1/√k.
        assert!((STACK_DAMPEN_EXPONENT - 0.5).abs() < 1e-12);
        assert!((stack_dampen_scale(4) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn combination_index_sets_prefers_pairs_then_triples() {
        let sets = combination_index_sets(3, 10);
        assert_eq!(
            sets,
            vec![vec![0, 1], vec![0, 2], vec![1, 2], vec![0, 1, 2]]
        );
        let capped = combination_index_sets(5, 3);
        assert_eq!(capped.len(), 3);
        assert!(capped.iter().all(|s| s.len() == 2));
    }

    #[test]
    fn merged_creature_keeps_synapses_in_canonical_order() {
        // A combo merge tail-appended every new edge, so the merged creature
        // broke `neat_core::creature_validate` rule 25 while every other gate
        // stayed green. `GRQ-23-lamarck.json` was published that way and the
        // shared validator refused it with
        // `TopologyError(SORT_FAILURE): 28564) synapses not sorted`.
        let base = tiny();
        let mut a = base.clone();
        a.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "h1".into(),
            weight: 0.05,
            synapse_type: None,
        });
        let mut b = base.clone();
        b.synapses.push(SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "o1".into(),
            weight: 0.02,
            synapse_type: None,
        });

        let merged = merge_candidate_deltas(&base, &[&a, &b]).unwrap();

        // `input-0` = 0, `input-1` = 1, `h1` = 2, `o1` = 3.
        let keys: Vec<(String, String)> = merged
            .synapses
            .iter()
            .map(|s| (s.from_uuid.clone(), s.to_uuid.clone()))
            .collect();
        assert_eq!(
            keys,
            vec![
                ("input-0".to_string(), "h1".to_string()),
                ("input-0".to_string(), "o1".to_string()),
                ("input-1".to_string(), "h1".to_string()),
                ("h1".to_string(), "o1".to_string()),
            ],
            "new edges must land in canonical (from, to) order, not at the tail"
        );
        crate::validate::validate_creature(&merged, "combo merge").unwrap();
    }

    #[test]
    fn merge_applies_two_independent_edges() {
        let base = tiny();
        let mut a = base.clone();
        a.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "h1".into(),
            weight: 0.05,
            synapse_type: None,
        });
        let mut b = base.clone();
        b.synapses.push(SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "o1".into(),
            weight: 0.02,
            synapse_type: None,
        });
        let merged = merge_candidate_deltas(&base, &[&a, &b]).unwrap();
        assert!(
            merged
                .synapses
                .iter()
                .any(|s| s.from_uuid == "input-1" && s.to_uuid == "h1")
        );
        assert!(
            merged
                .synapses
                .iter()
                .any(|s| s.from_uuid == "input-0" && s.to_uuid == "o1")
        );
    }

    /// neat-core 0.9.9 added `NeuronExport::id`. Merging a variant's new
    /// neuron onto the base is a faithful copy, so whatever runtime id the
    /// variant carried survives the merge rather than being dropped.
    #[test]
    fn merge_preserves_the_runtime_id_of_a_copied_neuron() {
        let base = tiny();
        let mut a = base.clone();
        a.neurons.insert(
            0,
            NeuronExport {
                id: Some(42),
                neuron_type: "hidden".into(),
                uuid: "h-new".into(),
                bias: 0.25,
                squash: Some("TANH".into()),
            },
        );
        a.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "h-new".into(),
            weight: 0.05,
            synapse_type: None,
        });
        a.synapses.push(SynapseExport {
            from_uuid: "h-new".into(),
            to_uuid: "o1".into(),
            weight: 0.05,
            synapse_type: None,
        });

        let merged = merge_candidate_deltas(&base, &[&a]).unwrap();
        let copied = merged
            .neurons
            .iter()
            .find(|n| n.uuid == "h-new")
            .expect("the variant's new neuron is merged in");
        assert_eq!(copied.id, Some(42));
        assert_eq!(copied.bias, 0.25);
        assert_eq!(copied.squash.as_deref(), Some("TANH"));
    }

    #[test]
    fn merge_rejects_weight_conflict() {
        let base = tiny();
        let mut a = base.clone();
        a.synapses[0].weight = 0.5;
        let mut b = base.clone();
        b.synapses[0].weight = 0.9;
        assert!(merge_candidate_deltas(&base, &[&a, &b]).is_err());
    }

    #[test]
    fn dampen_scales_two_new_edges_into_same_target() {
        let base = tiny();
        let mut a = base.clone();
        a.synapses.push(SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "o1".into(),
            weight: 0.05,
            synapse_type: None,
        });
        let mut b = base.clone();
        b.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "o1".into(),
            weight: 0.08,
            synapse_type: None,
        });
        let mut merged = merge_candidate_deltas(&base, &[&a, &b]).unwrap();
        let contributors = new_synapse_contributor_counts(&base, &[&a, &b]);
        let report = dampen_stacked_new_synapses(&base, &mut merged, &contributors);
        assert_eq!(report.targets.len(), 1);
        assert_eq!(report.targets[0].to_uuid, "o1");
        assert_eq!(report.targets[0].k, 2);
        let scale = stack_dampen_scale(2);
        assert!((report.targets[0].scale - scale).abs() < 1e-12);
        assert!((report.targets[0].weight_abs_sum_before - 0.13).abs() < 1e-12);
        assert!((report.targets[0].weight_abs_sum_after - 0.13 * scale).abs() < 1e-12);
        let w0 = merged
            .synapses
            .iter()
            .find(|s| s.from_uuid == "input-0" && s.to_uuid == "o1")
            .unwrap()
            .weight;
        let w1 = merged
            .synapses
            .iter()
            .find(|s| s.from_uuid == "input-1" && s.to_uuid == "o1")
            .unwrap()
            .weight;
        assert!((w0 - 0.05 * scale).abs() < 1e-12);
        assert!((w1 - 0.08 * scale).abs() < 1e-12);
        // Existing edges unchanged.
        assert!(
            (merged
                .synapses
                .iter()
                .find(|s| s.from_uuid == "input-0" && s.to_uuid == "h1")
                .unwrap()
                .weight
                - 1.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn dampen_leaves_multi_edge_single_member_pack() {
        // One candidate already carries two new edges into the same target
        // (e.g. StructuralAddNeuron). Merging with an unrelated bias tweak must
        // not treat those as two stacked solo residual steps.
        let base = tiny();
        let mut bridge = base.clone();
        bridge.synapses.push(SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "o1".into(),
            weight: 0.05,
            synapse_type: None,
        });
        bridge.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "o1".into(),
            weight: 0.08,
            synapse_type: None,
        });
        let mut bias_only = base.clone();
        bias_only.neurons[0].bias = 0.1;
        let mut merged = merge_candidate_deltas(&base, &[&bridge, &bias_only]).unwrap();
        let contributors = new_synapse_contributor_counts(&base, &[&bridge, &bias_only]);
        assert_eq!(contributors.get("o1").copied(), Some(1));
        let report = dampen_stacked_new_synapses(&base, &mut merged, &contributors);
        assert!(report.is_empty());
        assert!(
            (merged
                .synapses
                .iter()
                .find(|s| s.from_uuid == "input-0" && s.to_uuid == "o1")
                .unwrap()
                .weight
                - 0.05)
                .abs()
                < 1e-12
        );
        assert!(
            (merged
                .synapses
                .iter()
                .find(|s| s.from_uuid == "input-1" && s.to_uuid == "o1")
                .unwrap()
                .weight
                - 0.08)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn dampen_leaves_new_edges_into_different_targets() {
        let base = tiny();
        let mut a = base.clone();
        a.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "h1".into(),
            weight: 0.05,
            synapse_type: None,
        });
        let mut b = base.clone();
        b.synapses.push(SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "o1".into(),
            weight: 0.02,
            synapse_type: None,
        });
        let mut merged = merge_candidate_deltas(&base, &[&a, &b]).unwrap();
        let contributors = new_synapse_contributor_counts(&base, &[&a, &b]);
        let report = dampen_stacked_new_synapses(&base, &mut merged, &contributors);
        assert!(report.is_empty());
        assert!(
            (merged
                .synapses
                .iter()
                .find(|s| s.from_uuid == "input-1" && s.to_uuid == "h1")
                .unwrap()
                .weight
                - 0.05)
                .abs()
                < 1e-12
        );
        assert!(
            (merged
                .synapses
                .iter()
                .find(|s| s.from_uuid == "input-0" && s.to_uuid == "o1")
                .unwrap()
                .weight
                - 0.02)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn dampen_ignores_bias_only_stacking_with_one_new_edge() {
        let base = tiny();
        let mut merged = base.clone();
        merged.neurons[0].bias = 0.1;
        merged.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "h1".into(),
            weight: 0.05,
            synapse_type: None,
        });
        let mut counts = BTreeMap::new();
        counts.insert("h1".into(), 1usize);
        let report = dampen_stacked_new_synapses(&base, &mut merged, &counts);
        assert!(report.is_empty());
        assert!(
            (merged
                .synapses
                .iter()
                .find(|s| s.from_uuid == "input-1" && s.to_uuid == "h1")
                .unwrap()
                .weight
                - 0.05)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn dampen_ignores_existing_edge_weight_change_when_counting_k() {
        let base = tiny();
        let mut weight_change = base.clone();
        weight_change.synapses[0].weight = 0.5;
        let mut add = base.clone();
        add.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "h1".into(),
            weight: 0.05,
            synapse_type: None,
        });
        let mut merged = merge_candidate_deltas(&base, &[&weight_change, &add]).unwrap();
        let contributors = new_synapse_contributor_counts(&base, &[&weight_change, &add]);
        let report = dampen_stacked_new_synapses(&base, &mut merged, &contributors);
        assert!(report.is_empty());
        assert!((merged.synapses[0].weight - 0.5).abs() < 1e-12);
        assert!(
            (merged
                .synapses
                .iter()
                .find(|s| s.from_uuid == "input-1" && s.to_uuid == "h1")
                .unwrap()
                .weight
                - 0.05)
                .abs()
                < 1e-12
        );
    }

    struct ComboBoostScorer;

    impl DirectoryScorer for ComboBoostScorer {
        fn score_directory_sampled(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
            _sample: ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            let base = 0.5;
            let mut map = BTreeMap::new();
            for entry in fs::read_dir(candidates_dir).into_iter().flatten().flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                let Some(stem) = name.strip_suffix(".json") else {
                    continue;
                };
                let text = fs::read_to_string(entry.path()).unwrap_or_default();
                let Ok(c) = parse_creature_json(&text) else {
                    continue;
                };
                let has_a = c
                    .synapses
                    .iter()
                    .any(|s| s.from_uuid == "input-1" && s.to_uuid == "h1");
                let has_b = c
                    .synapses
                    .iter()
                    .any(|s| s.from_uuid == "input-0" && s.to_uuid == "o1");
                let score = if stem == "baseline" {
                    base
                } else if has_a && has_b {
                    base + 5e-6
                } else if has_a || has_b {
                    base + 2e-6
                } else {
                    base
                };
                map.insert(
                    stem.to_string(),
                    ScoreResult {
                        score,
                        error: 1.0 - score,
                        complexity_penalty: 0.0,
                    },
                );
            }
            Ok(map)
        }
    }

    #[test]
    fn select_best_prefers_merged_combo() {
        let dir = tempdir().unwrap();
        let base = tiny();
        let mut a = base.clone();
        a.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "h1".into(),
            weight: 0.05,
            synapse_type: None,
        });
        let mut b = base.clone();
        b.synapses.push(SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "o1".into(),
            weight: 0.02,
            synapse_type: None,
        });
        let candidates = vec![cand(a), cand(b)];
        let source = dir.path().join("source");
        fs::create_dir_all(&source).unwrap();
        write_creature_json(&source.join("baseline.json"), &base).unwrap();
        write_creature_json(&source.join("candidate-000.json"), &candidates[0].creature).unwrap();
        write_creature_json(&source.join("candidate-001.json"), &candidates[1].creature).unwrap();

        let mut scores = BTreeMap::new();
        scores.insert(
            "baseline".into(),
            ScoreResult {
                score: 0.5,
                error: 0.5,
                complexity_penalty: 0.0,
            },
        );
        scores.insert(
            "candidate-000".into(),
            ScoreResult {
                score: 0.5 + 2e-6,
                error: 0.5 - 2e-6,
                complexity_penalty: 0.0,
            },
        );
        scores.insert(
            "candidate-001".into(),
            ScoreResult {
                score: 0.5 + 2e-6,
                error: 0.5 - 2e-6,
                complexity_penalty: 0.0,
            },
        );

        let best = select_best_with_combinations(
            &ComboBoostScorer,
            ComboSelectRequest {
                training_data: dir.path(),
                incumbent: &base,
                candidates: &candidates,
                scores: &scores,
                baseline: scores.get("baseline").expect("the test batch carries one"),
                min_improvement: 1e-6,
                source_dir: &source,
                combo_work_dir: &dir.path().join("combos"),
            },
        )
        .unwrap()
        .expect("selection");
        assert!(best.stem.starts_with("combo-"));
        assert_eq!(best.member_indices.len(), 2);
        assert!((best.result.score - (0.5 + 5e-6)).abs() < 1e-12);
        assert_eq!(best.combos_scored, 1);
        assert_eq!(best.combos_dampened, 0); // different targets → no stack dampen
        assert!(best.dampen.is_empty());
    }

    /// Scorer that rewards two new edges into the same target (stacked case).
    struct StackedTargetScorer;

    impl DirectoryScorer for StackedTargetScorer {
        fn score_directory_sampled(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
            _sample: ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            let base = 0.5;
            let mut map = BTreeMap::new();
            for entry in fs::read_dir(candidates_dir).into_iter().flatten().flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                let Some(stem) = name.strip_suffix(".json") else {
                    continue;
                };
                let text = fs::read_to_string(entry.path()).unwrap_or_default();
                let Ok(c) = parse_creature_json(&text) else {
                    continue;
                };
                let new_into_o1 = c
                    .synapses
                    .iter()
                    .filter(|s| s.to_uuid == "o1" && s.from_uuid.starts_with("input-"))
                    .count();
                // Prefer both edges present; score does not depend on weight magnitude
                // so we can assert dampening on the written creature separately.
                let score = if stem == "baseline" {
                    base
                } else if new_into_o1 >= 2 {
                    base + 5e-6
                } else if new_into_o1 == 1 {
                    base + 2e-6
                } else {
                    base
                };
                map.insert(
                    stem.to_string(),
                    ScoreResult {
                        score,
                        error: 1.0 - score,
                        complexity_penalty: 0.0,
                    },
                );
            }
            Ok(map)
        }
    }

    #[test]
    fn select_best_writes_dampened_weights_for_stacked_target() {
        let dir = tempdir().unwrap();
        let base = tiny();
        let mut a = base.clone();
        a.synapses.push(SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "o1".into(),
            weight: 0.06,
            synapse_type: None,
        });
        let mut b = base.clone();
        b.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "o1".into(),
            weight: 0.08,
            synapse_type: None,
        });
        let candidates = vec![cand(a), cand(b)];
        let source = dir.path().join("source");
        fs::create_dir_all(&source).unwrap();
        write_creature_json(&source.join("baseline.json"), &base).unwrap();
        write_creature_json(&source.join("candidate-000.json"), &candidates[0].creature).unwrap();
        write_creature_json(&source.join("candidate-001.json"), &candidates[1].creature).unwrap();

        let mut scores = BTreeMap::new();
        scores.insert(
            "baseline".into(),
            ScoreResult {
                score: 0.5,
                error: 0.5,
                complexity_penalty: 0.0,
            },
        );
        scores.insert(
            "candidate-000".into(),
            ScoreResult {
                score: 0.5 + 2e-6,
                error: 0.5 - 2e-6,
                complexity_penalty: 0.0,
            },
        );
        scores.insert(
            "candidate-001".into(),
            ScoreResult {
                score: 0.5 + 2e-6,
                error: 0.5 - 2e-6,
                complexity_penalty: 0.0,
            },
        );

        let combo_work = dir.path().join("combos");
        let best = select_best_with_combinations(
            &StackedTargetScorer,
            ComboSelectRequest {
                training_data: dir.path(),
                incumbent: &base,
                candidates: &candidates,
                scores: &scores,
                baseline: scores.get("baseline").expect("the test batch carries one"),
                min_improvement: 1e-6,
                source_dir: &source,
                combo_work_dir: &combo_work,
            },
        )
        .unwrap()
        .expect("selection");
        assert!(best.stem.starts_with("combo-"));
        assert_eq!(best.combos_scored, 1);
        assert_eq!(best.combos_dampened, 1);
        assert_eq!(best.dampen.targets.len(), 1);
        assert_eq!(best.dampen.targets[0].k, 2);

        let written =
            parse_creature_json(&fs::read_to_string(&best.creature_path).unwrap()).unwrap();
        let scale = stack_dampen_scale(2);
        let w0 = written
            .synapses
            .iter()
            .find(|s| s.from_uuid == "input-0" && s.to_uuid == "o1")
            .unwrap()
            .weight;
        let w1 = written
            .synapses
            .iter()
            .find(|s| s.from_uuid == "input-1" && s.to_uuid == "o1")
            .unwrap()
            .weight;
        assert!((w0 - 0.06 * scale).abs() < 1e-12);
        assert!((w1 - 0.08 * scale).abs() < 1e-12);
    }

    /// Scorer that reproduces the issue #130 artefact: the combo call holds a
    /// different number of creatures from the promote call, so *every* score it
    /// returns — the incumbent's included — is shifted by `offset`. The merged
    /// combo earns `combo_delta` over the incumbent **within that call**.
    struct DriftingComboScorer {
        offset: f64,
        combo_delta: f64,
    }

    impl DirectoryScorer for DriftingComboScorer {
        fn score_directory_sampled(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
            _sample: ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            let mut map = BTreeMap::new();
            for entry in fs::read_dir(candidates_dir).into_iter().flatten().flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                let Some(stem) = name.strip_suffix(".json") else {
                    continue;
                };
                let score = if stem.starts_with("combo-") {
                    0.5 + self.offset + self.combo_delta
                } else {
                    0.5 + self.offset
                };
                map.insert(
                    stem.to_string(),
                    ScoreResult {
                        score,
                        error: 1.0 - score,
                        complexity_penalty: 0.0,
                    },
                );
            }
            Ok(map)
        }
    }

    /// Two improving singles plus the promote-call scores that shortlist them.
    fn drift_fixture(
        dir: &Path,
    ) -> (
        CreatureExport,
        Vec<Candidate>,
        BTreeMap<String, ScoreResult>,
    ) {
        let base = tiny();
        let mut a = base.clone();
        a.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "h1".into(),
            weight: 0.05,
            synapse_type: None,
        });
        let mut b = base.clone();
        b.synapses.push(SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "o1".into(),
            weight: 0.02,
            synapse_type: None,
        });
        let candidates = vec![cand(a), cand(b)];
        let source = dir.join("source");
        fs::create_dir_all(&source).unwrap();
        write_creature_json(&source.join("baseline.json"), &base).unwrap();
        write_creature_json(&source.join("candidate-000.json"), &candidates[0].creature).unwrap();
        write_creature_json(&source.join("candidate-001.json"), &candidates[1].creature).unwrap();

        let mut scores = BTreeMap::new();
        for (stem, score) in [
            ("baseline", 0.5),
            ("candidate-000", 0.5 + 2e-6),
            ("candidate-001", 0.5 + 2e-6),
        ] {
            scores.insert(
                stem.to_string(),
                ScoreResult {
                    score,
                    error: 1.0 - score,
                    complexity_penalty: 0.0,
                },
            );
        }
        (base, candidates, scores)
    }

    /// Issue #130: a combo is judged against the incumbent scored in **its own**
    /// call. Here the combo call's scores all sit 3e-6 above the promote call's,
    /// so subtracting the promote-call baseline would show a 3.5e-6 "improvement"
    /// where the same-call evidence is only 0.5e-6 — below the 1e-6 bar.
    #[test]
    fn combo_is_rejected_when_only_a_cross_call_baseline_would_accept_it() {
        let dir = tempdir().unwrap();
        let (base, candidates, scores) = drift_fixture(dir.path());

        let best = select_best_with_combinations(
            &DriftingComboScorer {
                offset: 3e-6,
                combo_delta: 0.5e-6,
            },
            ComboSelectRequest {
                training_data: dir.path(),
                incumbent: &base,
                candidates: &candidates,
                scores: &scores,
                baseline: scores.get("baseline").expect("the test batch carries one"),
                min_improvement: 1e-6,
                source_dir: &dir.path().join("source"),
                combo_work_dir: &dir.path().join("combos"),
            },
        )
        .unwrap()
        .expect("the improving single is still a selection");
        assert!(
            best.stem.starts_with("candidate-"),
            "a combo that only clears the bar against another call's baseline must not win, got {}",
            best.stem
        );
        assert!(
            (best.delta - 2e-6).abs() < 1e-12,
            "the surviving single keeps its own same-call Δ, got {:e}",
            best.delta
        );
    }

    /// Issue #130: the mirror case — the combo call's scores sit 3e-6 *below* the
    /// promote call's, so the combo's raw score never beats the single's, but its
    /// same-call Δ (5e-6) is the larger real improvement.
    #[test]
    fn combo_wins_on_its_same_call_delta_even_when_its_raw_score_looks_worse() {
        let dir = tempdir().unwrap();
        let (base, candidates, scores) = drift_fixture(dir.path());

        let best = select_best_with_combinations(
            &DriftingComboScorer {
                offset: -3e-6,
                combo_delta: 5e-6,
            },
            ComboSelectRequest {
                training_data: dir.path(),
                incumbent: &base,
                candidates: &candidates,
                scores: &scores,
                baseline: scores.get("baseline").expect("the test batch carries one"),
                min_improvement: 1e-6,
                source_dir: &dir.path().join("source"),
                combo_work_dir: &dir.path().join("combos"),
            },
        )
        .unwrap()
        .expect("selection");
        assert!(
            best.stem.starts_with("combo-"),
            "the combo earns the larger same-call Δ and must win, got {}",
            best.stem
        );
        assert!(
            (best.delta - 5e-6).abs() < 1e-12,
            "the winner's Δ is measured against the incumbent in its own call, got {:e}",
            best.delta
        );
    }

    /// Scorer that drops the incumbent from its result map.
    struct BaselinelessScorer;

    impl DirectoryScorer for BaselinelessScorer {
        fn score_directory_sampled(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
            _sample: ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            let mut map = BTreeMap::new();
            for entry in fs::read_dir(candidates_dir).into_iter().flatten().flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                let Some(stem) = name.strip_suffix(".json") else {
                    continue;
                };
                if stem == "baseline" {
                    continue;
                }
                map.insert(
                    stem.to_string(),
                    ScoreResult {
                        score: 0.6,
                        error: 0.4,
                        complexity_penalty: 0.0,
                    },
                );
            }
            Ok(map)
        }
    }

    /// Issue #130: the combo call pays to score the incumbent, so a missing
    /// `baseline` stem is a broken call — never a licence to fall back to another
    /// call's number.
    #[test]
    fn combo_call_without_a_baseline_score_is_a_loud_error() {
        let dir = tempdir().unwrap();
        let (base, candidates, scores) = drift_fixture(dir.path());

        let err = select_best_with_combinations(
            &BaselinelessScorer,
            ComboSelectRequest {
                training_data: dir.path(),
                incumbent: &base,
                candidates: &candidates,
                scores: &scores,
                baseline: scores.get("baseline").expect("the test batch carries one"),
                min_improvement: 1e-6,
                source_dir: &dir.path().join("source"),
                combo_work_dir: &dir.path().join("combos"),
            },
        )
        .expect_err("a combo call that returns no baseline must fail loudly");
        assert!(
            err.contains("baseline"),
            "the error must name the missing baseline, got: {err}"
        );
    }

    /// Issue #130: the accept gate reads the selection's same-call Δ, and the bar
    /// is strict — a Δ exactly on it is not an improvement.
    #[test]
    fn selection_accepts_on_its_same_call_delta() {
        let mut sel = ComboSelection {
            creature_path: std::path::PathBuf::from("winner.json"),
            stem: "combo-000-k2".into(),
            // A raw score far below the promote-call baseline: irrelevant to the
            // gate, which reads `delta` alone.
            result: ScoreResult {
                score: 0.1,
                error: 0.9,
                complexity_penalty: 0.0,
            },
            delta: 2e-6,
            member_indices: vec![0, 1],
            dampen: StackDampenReport::default(),
            combos_scored: 1,
            combos_dampened: 0,
        };
        assert!(sel.accepts(1e-6));
        sel.delta = 1e-6;
        assert!(!sel.accepts(1e-6), "the bar is strict");
        sel.delta = -2e-6;
        assert!(!sel.accepts(1e-6));
    }

    #[test]
    fn stack_dampen_report_roundtrips_json_for_journal() {
        let report = StackDampenReport {
            targets: vec![StackDampenTarget {
                to_uuid: "o1".into(),
                k: 2,
                scale: stack_dampen_scale(2),
                weight_abs_sum_before: 0.14,
                weight_abs_sum_after: 0.14 * stack_dampen_scale(2),
            }],
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("weightAbsSumBefore"));
        assert!(json.contains("toUuid"));
        let back: StackDampenReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
    }
}
