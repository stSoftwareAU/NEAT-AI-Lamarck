//! Parallel combination scoring for improving singles (experiments + graft replay).
//!
//! When several candidates beat the baseline alone, merge their deltas onto the
//! incumbent in groups (pairs, then triples, …) and score those creatures in one
//! scorer directory batch — up to [`MAX_COMBO_CANDIDATES`] including the singles.
//!
//! Newly stacked synapses into the same target are dampened by
//! [`stack_dampen_scale`] so solo-sized weights do not overshoot when combined
//! (#63). The exponent is [`STACK_DAMPEN_EXPONENT`] — tune after production evidence.

use crate::candidates::Candidate;
use crate::log;
use crate::scorer::{DirectoryScorer, ScoreResult, accepts_improvement, improvement};
use crate::structural::insert_index_for_hidden;
use neat_core::{CreatureExport, NeuronExport, SynapseExport, creature_to_json_pretty};
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
    /// Δ vs the full-corpus baseline.
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
    Ok(out)
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
                out.synapses.push(SynapseExport {
                    from_uuid: vs.from_uuid.clone(),
                    to_uuid: vs.to_uuid.clone(),
                    weight: vs.weight,
                    synapse_type: vs.synapse_type.clone(),
                });
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
    /// Number of new synapses into [`Self::to_uuid`].
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

/// Scale newly added synapses that stack into the same target by [`stack_dampen_scale`].
///
/// Only edges absent from `base` are considered. Bias tweaks and weight changes
/// on existing edges are untouched. When a target receives a single new edge,
/// that weight is left alone (`k == 1`).
pub fn dampen_stacked_new_synapses(
    base: &CreatureExport,
    creature: &mut CreatureExport,
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
        let k = indices.len();
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

fn write_creature_json(path: &Path, creature: &CreatureExport) -> Result<(), String> {
    let json = creature_to_json_pretty(creature).map_err(|e| e.to_string())?;
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
    /// Full-corpus scores (must include `baseline` + `candidate-*`).
    pub scores: &'a BTreeMap<String, ScoreResult>,
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
        min_improvement,
        source_dir,
        combo_work_dir,
    } = request;
    let baseline = scores
        .get("baseline")
        .ok_or_else(|| "baseline missing".to_string())?;
    let improvers = collect_improvers(scores, min_improvement)?;
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
        let dampen = dampen_stacked_new_synapses(incumbent, &mut merged);
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

    for (stem, members) in &stem_to_members {
        let Some(result) = combo_scores.get(stem) else {
            continue;
        };
        if accepts_improvement(result.score, baseline.score, min_improvement)
            && result.score > best.result.score
        {
            best = ComboSelection {
                creature_path: combo_work_dir.join(format!("{stem}.json")),
                stem: stem.clone(),
                result: result.clone(),
                delta: improvement(result.score, baseline.score),
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
        let mut merged = base.clone();
        merged.synapses.push(SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "o1".into(),
            weight: 0.05,
            synapse_type: None,
        });
        merged.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "o1".into(),
            weight: 0.08,
            synapse_type: None,
        });
        let report = dampen_stacked_new_synapses(&base, &mut merged);
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
    fn dampen_leaves_new_edges_into_different_targets() {
        let base = tiny();
        let mut merged = base.clone();
        merged.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "h1".into(),
            weight: 0.05,
            synapse_type: None,
        });
        merged.synapses.push(SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "o1".into(),
            weight: 0.02,
            synapse_type: None,
        });
        let report = dampen_stacked_new_synapses(&base, &mut merged);
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
        let report = dampen_stacked_new_synapses(&base, &mut merged);
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
        let mut merged = base.clone();
        // Change existing input-0->h1 weight.
        merged.synapses[0].weight = 0.5;
        // One new edge into h1.
        merged.synapses.push(SynapseExport {
            from_uuid: "input-1".into(),
            to_uuid: "h1".into(),
            weight: 0.05,
            synapse_type: None,
        });
        let report = dampen_stacked_new_synapses(&base, &mut merged);
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

        let written = parse_creature_json(
            &fs::read_to_string(&best.creature_path).unwrap(),
        )
        .unwrap();
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
