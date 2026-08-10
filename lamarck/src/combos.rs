//! Parallel combination scoring for improving singles (experiments + graft replay).
//!
//! When several candidates beat the baseline alone, merge their deltas onto the
//! incumbent in groups (pairs, then triples, …) and score those creatures in one
//! scorer directory batch — up to [`MAX_COMBO_CANDIDATES`] including the singles.

use crate::candidates::Candidate;
use crate::log;
use crate::scorer::{DirectoryScorer, ScoreResult, accepts_improvement, improvement};
use crate::structural::insert_index_for_hidden;
use neat_core::{CreatureExport, NeuronExport, SynapseExport, creature_to_json_pretty};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Max improving singles + combination creatures scored together.
pub const MAX_COMBO_CANDIDATES: usize = 50;

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
        let Ok(merged) = merge_candidate_deltas(incumbent, &variants) else {
            continue;
        };
        let stem = format!("combo-{ci:03}-k{}", idxs.len());
        write_creature_json(&combo_work_dir.join(format!("{stem}.json")), &merged)?;
        stem_to_members.insert(stem, member_indices);
    }

    if stem_to_members.is_empty() {
        let _ = fs::remove_dir_all(combo_work_dir);
        return Ok(Some(best));
    }

    log::info(&format!(
        "combo: scoring {} combination(s) in parallel (improvers={}, budget={})",
        stem_to_members.len(),
        improvers.len(),
        MAX_COMBO_CANDIDATES
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
            };
            log::ok(&format!(
                "combo {stem}: score={:.12} (+{:.3e}, {} members)",
                best.result.score,
                best.delta,
                best.member_indices.len()
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
    }
}
