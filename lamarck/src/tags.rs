//! Creature JSON tags (`{ name, value }[]`) for check-in / GRQ compatibility.
//!
//! `neat_core::CreatureExport` does not round-trip `tags` / `uuid`, so Lamarck
//! keeps them in [`CreatureMeta`] and re-attaches on write — preserving the
//! original pedigree (name, version, …) while stamping a **run-level** Lamarck
//! summary for GRQ check-in (`worker/Lamarck/run.sh` reads the tag as-is).

use crate::candidates::CandidateStrategy;
use crate::width::{assert_written_width, validate_creature_width, value_width};
use neat_core::{CreatureExport, creature_to_json};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};

/// One creature tag (NEAT-AI / `@stsoftware/tags` shape).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreatureTag {
    /// Tag key (`score`, `error`, `name`, `lamarck`, …).
    pub name: String,
    /// Tag value (always a string in the export format).
    pub value: String,
}

/// Parse a `{ name, value }[]` tag array off a JSON node (missing → empty).
fn parse_tag_array(node: Option<&Value>) -> Vec<CreatureTag> {
    node.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    let name = t.get("name")?.as_str()?.to_string();
                    let value = match t.get("value")? {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    Some(CreatureTag { name, value })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Insert or replace a tag by name, preserving order of first insert.
fn upsert_tag(tags: &mut Vec<CreatureTag>, name: &str, value: String) {
    if let Some(existing) = tags.iter_mut().find(|t| t.name == name) {
        existing.value = value;
    } else {
        tags.push(CreatureTag {
            name: name.to_string(),
            value,
        });
    }
}

/// Top-level fields stripped by `parse_creature_json` that we must keep.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CreatureMeta {
    /// Optional creature UUID from the source JSON.
    pub uuid: Option<String>,
    /// Ordered tags (upserts replace by name, preserving order of first insert).
    pub tags: Vec<CreatureTag>,
    /// Per-neuron tags keyed by neuron UUID (issue #187).
    ///
    /// `NeuronExport` carries no `tags`, so the discovery / intelligentDesign
    /// provenance of each neuron is stripped on parse and would be lost from
    /// every check-in write. Keyed by uuid rather than position because
    /// structural growth inserts and reorders neurons between writes.
    pub neuron_tags: BTreeMap<String, Vec<CreatureTag>>,
}

impl CreatureMeta {
    /// Parse `uuid` + creature `tags` + `neurons[].tags` from raw creature JSON
    /// (missing → empty).
    pub fn from_creature_json(text: &str) -> Self {
        let Ok(value) = serde_json::from_str::<Value>(text) else {
            return Self::default();
        };
        let uuid = value
            .get("uuid")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let tags = parse_tag_array(value.get("tags"));
        let mut neuron_tags = BTreeMap::new();
        if let Some(neurons) = value.get("neurons").and_then(|v| v.as_array()) {
            for neuron in neurons {
                let Some(uuid) = neuron.get("uuid").and_then(|v| v.as_str()) else {
                    continue;
                };
                let tags = parse_tag_array(neuron.get("tags"));
                if !tags.is_empty() {
                    neuron_tags.insert(uuid.to_string(), tags);
                }
            }
        }
        Self {
            uuid,
            tags,
            neuron_tags,
        }
    }

    /// Insert or replace a tag by name.
    pub fn upsert(&mut self, name: &str, value: impl Into<String>) {
        upsert_tag(&mut self.tags, name, value.into());
    }

    /// Insert or replace one neuron's tag by name (issue #187).
    pub fn upsert_neuron(&mut self, neuron_uuid: &str, name: &str, value: impl Into<String>) {
        upsert_tag(
            self.neuron_tags.entry(neuron_uuid.to_string()).or_default(),
            name,
            value.into(),
        );
    }

    /// Tag every neuron `current` has that `previous` did not (issue #187).
    ///
    /// Structural growth invents neurons no source JSON described, so their
    /// provenance is Lamarck's to record: the strategy that grew them, the
    /// focus neuron they were grown for, and the experiment that accepted
    /// them. Neurons that were already there keep the tags they arrived with.
    pub fn stamp_new_neurons(
        &mut self,
        previous: &CreatureExport,
        current: &CreatureExport,
        origin: &NeuronOrigin<'_>,
    ) {
        let before: HashSet<&str> = previous.neurons.iter().map(|n| n.uuid.as_str()).collect();
        let message = lamarck_neuron_origin_message(origin);
        for neuron in &current.neurons {
            if !before.contains(neuron.uuid.as_str()) {
                self.upsert_neuron(&neuron.uuid, "lamarck", message.clone());
            }
        }
    }

    /// Update score/error and stamp a run-level Lamarck summary for check-in.
    pub fn stamp_acceptance(&mut self, progress: &LamarckProgress<'_>) {
        // Keep full-precision numeric tags for machine consumers.
        self.upsert("score", format!("{}", progress.score));
        self.upsert("error", format!("{}", progress.error));
        // Only the `lamarck` tag is ours. `intelligentDesign` belongs to the
        // Intelligent Design program — stamping it here overwrote another
        // program's provenance (GRQ #3952), so never touch it.
        self.upsert("lamarck", lamarck_progress_message(progress));
    }
}

/// Fields for the run-level Lamarck check-in summary (cumulative, not last-step).
#[derive(Debug, Clone, Copy)]
pub struct LamarckProgress<'a> {
    /// Acceptances so far in this run.
    pub acceptances: u64,
    /// Authoritative score after the latest accept (or final best).
    pub score: f64,
    /// Authoritative error after the latest accept (or final best).
    pub error: f64,
    /// Opening baseline score (Phase-0 / first baseline) for cumulative Δ.
    pub opening_score: f64,
    /// Focus neuron UUID for the latest accept.
    pub focus_neuron: &'a str,
    /// Winning candidate strategy for the latest accept.
    pub strategy: CandidateStrategy,
    /// Experiments attempted so far (full run count when stamped at end).
    pub experiments: u64,
}

/// Run-level check-in blurb: accepts / experiments / last strategy / score once.
///
/// Score wording matches GRQ `grq_format_score_message` (`%.6g` + `improved by %.3g`
/// vs opening). GRQ should use this tag as the commit subject without appending
/// another score clause.
pub fn lamarck_progress_message(progress: &LamarckProgress<'_>) -> String {
    let (strat_emoji, strat_label) = strategy_emoji(progress.strategy);
    let score_clause = format_score_improved(progress.score, progress.opening_score);
    let accept_word = if progress.acceptances == 1 {
        "accept"
    } else {
        "accepts"
    };
    let exp_word = if progress.experiments == 1 {
        "exp"
    } else {
        "exps"
    };
    format!(
        "🦒 Lamarck · {} {accept_word} / {} {exp_word} · last: {strat_emoji} {strat_label} · 🎯 {} · {score_clause}",
        progress.acceptances, progress.experiments, progress.focus_neuron,
    )
}

/// Where a neuron Lamarck grew came from (issue #187).
#[derive(Debug, Clone, Copy)]
pub struct NeuronOrigin<'a> {
    /// Candidate strategy that proposed the neuron.
    pub strategy: CandidateStrategy,
    /// Focus neuron the growth was proposed for.
    pub focus_neuron: &'a str,
    /// Experiment number that accepted it.
    pub experiment: u64,
}

/// Per-neuron provenance blurb for a neuron Lamarck grew.
///
/// Deliberately shaped like the creature-level run summary so a reader of a
/// check-in creature sees one voice, but scoped to the single neuron: it names
/// the strategy, the focus and the experiment that produced *this* neuron.
pub fn lamarck_neuron_origin_message(origin: &NeuronOrigin<'_>) -> String {
    let (strat_emoji, strat_label) = strategy_emoji(origin.strategy);
    format!(
        "🦒 Lamarck · grown by {strat_emoji} {strat_label} · 🎯 {} · exp {}",
        origin.focus_neuron, origin.experiment
    )
}

/// `score: <%.6g> improved by <%.3g>` (cumulative vs opening). Same spirit as
/// GRQ `worker/shared/score_message.sh`.
fn format_score_improved(score: f64, opening: f64) -> String {
    let formatted = format_g(score, 6);
    let delta = score - opening;
    if delta < 0.0 {
        format!("score: {formatted} declined by {}", format_g(-delta, 3))
    } else {
        format!("score: {formatted} improved by {}", format_g(delta, 3))
    }
}

/// Approximate C/awk `printf("%.*g", prec, v)` for commit-friendly scores.
fn format_g(v: f64, prec: usize) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let prec = prec.max(1);
    let abs = v.abs();
    let exp = abs.log10().floor() as i32;
    // %g: scientific when exp < -4 or exp >= precision.
    if exp < -4 || exp >= prec as i32 {
        let digits = prec.saturating_sub(1);
        let s = format!("{v:.digits$e}");
        return trim_g_scientific(&s);
    }
    let decimals = (prec as i32 - exp - 1).max(0) as usize;
    let s = format!("{v:.decimals$}");
    trim_trailing_zeros_and_dot(&s)
}

fn trim_g_scientific(s: &str) -> String {
    // "1.23000e-6" → "1.23e-06" (C/awk printf %g style, two-digit exponent).
    let Some((mant, exp)) = s.split_once('e') else {
        return s.to_string();
    };
    let mant = trim_trailing_zeros_and_dot(mant);
    let exp_i: i32 = exp.parse().unwrap_or(0);
    format!("{mant}e{exp_i:+03}")
}

fn trim_trailing_zeros_and_dot(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let mut out = s.to_string();
    while out.ends_with('0') {
        out.pop();
    }
    if out.ends_with('.') {
        out.pop();
    }
    out
}

fn strategy_emoji(strategy: CandidateStrategy) -> (&'static str, &'static str) {
    match strategy {
        CandidateStrategy::Backprop => ("🧠", "backprop"),
        CandidateStrategy::MeanErrorBias => ("📏", "mean_error_bias"),
        CandidateStrategy::StatsWeight => ("📊", "stats_weight"),
        CandidateStrategy::StatsBias => ("📐", "stats_bias"),
        CandidateStrategy::StatsSkewBias => ("📉", "stats_skew_bias"),
        CandidateStrategy::StructuralAdd => ("🌱", "structural_add"),
        CandidateStrategy::StructuralAddNeuron => ("🧩", "structural_add_neuron"),
        CandidateStrategy::StructuralWeaken => ("✂️", "structural_weaken"),
        CandidateStrategy::Random => ("🎲", "random"),
    }
}

/// Creature JSON with `uuid` / `tags` / `neurons[].tags` re-attached, before it
/// is printed.
///
/// Refused unless the document about to be written carries the source
/// creature's `input` / `output` and both are `>= 1` (issue #165): a check-in
/// creature without its observation width cannot be re-derived downstream. Also
/// refused when its `memetic` record names structure the creature no longer
/// carries (issue #197).
///
/// This is a tags pass, not a structural one: `uuid`, `tags` and `memetic` are
/// carried through exactly as they arrived.
fn creature_value_with_meta(
    creature: &CreatureExport,
    meta: &CreatureMeta,
) -> Result<Value, String> {
    validate_creature_width(creature)?;
    crate::memetic::assert_memetic_resolves(creature)?;
    let body = creature_to_json(creature).map_err(|e| e.to_string())?;
    let mut value: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    if let Some(uuid) = &meta.uuid {
        value["uuid"] = json!(uuid);
    }
    if !meta.tags.is_empty() {
        value["tags"] = serde_json::to_value(&meta.tags).map_err(|e| e.to_string())?;
    }
    reattach_neuron_tags(&mut value, meta)?;
    assert_written_width(creature, value_width(&value)?)?;
    Ok(value)
}

/// Re-attach `neurons[].tags` by neuron uuid (issue #187).
///
/// Keyed by uuid, never by position: growth inserts neurons mid-list, so a
/// positional re-attach would hand one neuron's discovery provenance to
/// another. A neuron with no remembered tags is left exactly as exported —
/// no empty `tags` key is invented for it.
fn reattach_neuron_tags(value: &mut Value, meta: &CreatureMeta) -> Result<(), String> {
    if meta.neuron_tags.is_empty() {
        return Ok(());
    }
    let Some(neurons) = value.get_mut("neurons").and_then(|v| v.as_array_mut()) else {
        return Ok(());
    };
    for neuron in neurons.iter_mut() {
        let uuid = neuron
            .get("uuid")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let Some(tags) = uuid.as_deref().and_then(|u| meta.neuron_tags.get(u)) else {
            continue;
        };
        if tags.is_empty() {
            continue;
        }
        neuron["tags"] = serde_json::to_value(tags).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Pretty-print a creature with `uuid` / `tags` re-attached for check-in.
///
/// This is the **human-facing** form — `best.json` and `winners/`. Scorer-facing
/// batch files use [`serialize_creature_with_meta_compact`] (issue #114).
pub fn serialize_creature_with_meta(
    creature: &CreatureExport,
    meta: &CreatureMeta,
) -> Result<String, String> {
    let value = creature_value_with_meta(creature, meta)?;
    // Match typical NEAT export: trailing newline after pretty JSON.
    let mut out = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    if !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

/// Compact-print a creature with `uuid` / `tags` re-attached (issue #114).
///
/// Same document as [`serialize_creature_with_meta`] — same fields, same
/// values, same tags — with the indentation and newlines dropped. Only
/// scorer-facing batch files use it: on the production creature the whitespace
/// is about a third of the bytes written, parsed and thrown away on every
/// experiment, and `rust_scorer` is the file's only reader.
pub fn serialize_creature_with_meta_compact(
    creature: &CreatureExport,
    meta: &CreatureMeta,
) -> Result<String, String> {
    let value = creature_value_with_meta(creature, meta)?;
    serde_json::to_string(&value).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;

    const TINY_TAGGED: &str = r#"{
      "uuid": "creature-1",
      "input": 1,
      "output": 1,
      "forwardOnly": true,
      "neurons": [{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
      "synapses": [{"fromUUID":"input-0","toUUID":"o1","weight":1.0}],
      "tags": [
        {"name":"name","value":"Yara Richardson"},
        {"name":"version","value":"116"},
        {"name":"score","value":"0.1"}
      ]
    }"#;

    /// Issue #165: the check-in document carries the source's `input` /
    /// `output` byte-identically, and a source without a width is refused.
    #[test]
    fn serialize_preserves_source_width_byte_identically() {
        let src = TINY_TAGGED
            .replacen("\"input\": 1", "\"input\": 2511", 1)
            .replacen("\"output\": 1", "\"output\": 3", 1);
        let creature = parse_creature_json(&src).unwrap();
        let meta = CreatureMeta::from_creature_json(&src);
        for text in [
            serialize_creature_with_meta(&creature, &meta).unwrap(),
            serialize_creature_with_meta_compact(&creature, &meta).unwrap(),
        ] {
            let value: Value = serde_json::from_str(&text).unwrap();
            assert_eq!(value["input"], 2511);
            assert_eq!(value["output"], 3);
            let round = parse_creature_json(&text).unwrap();
            assert_eq!(
                (round.input, round.output),
                (creature.input, creature.output)
            );
        }
    }

    #[test]
    fn serialize_refuses_zero_width_source() {
        let meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        let mut creature = parse_creature_json(TINY_TAGGED).unwrap();
        creature.input = 0;
        let err = serialize_creature_with_meta(&creature, &meta).unwrap_err();
        assert_eq!(err, "Must have at least one input neurons was: 0");
        assert!(serialize_creature_with_meta_compact(&creature, &meta).is_err());
        let mut creature = parse_creature_json(TINY_TAGGED).unwrap();
        creature.output = 0;
        let err = serialize_creature_with_meta(&creature, &meta).unwrap_err();
        assert_eq!(err, "Must have at least one output neurons was: 0");
    }

    #[test]
    fn extract_preserves_uuid_and_tags() {
        let meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        assert_eq!(meta.uuid.as_deref(), Some("creature-1"));
        assert_eq!(meta.tags[0].name, "name");
        assert_eq!(meta.tags[0].value, "Yara Richardson");
    }

    #[test]
    fn serialize_round_trips_original_tags_plus_lamarck() {
        let creature = parse_creature_json(TINY_TAGGED).unwrap();
        let mut meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        meta.stamp_acceptance(&LamarckProgress {
            acceptances: 1,
            score: 0.2,
            error: 0.8,
            opening_score: 0.1,
            focus_neuron: "o1",
            strategy: CandidateStrategy::Backprop,
            experiments: 3,
        });
        let json = serialize_creature_with_meta(&creature, &meta).unwrap();
        let value: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["uuid"], "creature-1");
        let tags = value["tags"].as_array().unwrap();
        let name = tags.iter().find(|t| t["name"] == "name").unwrap();
        assert_eq!(name["value"], "Yara Richardson");
        let score = tags.iter().find(|t| t["name"] == "score").unwrap();
        assert_eq!(score["value"], "0.2");
        let lamarck = tags.iter().find(|t| t["name"] == "lamarck").unwrap();
        let msg = lamarck["value"].as_str().unwrap();
        assert!(msg.contains("🦒"));
        assert!(msg.contains("🧠"));
        assert!(msg.contains("1 accept / 3 exps"));
        assert!(msg.contains("score: 0.2 improved by 0.1"));
        assert!(!msg.contains("🏆"));
        assert!(!msg.contains("accept #"));
    }

    /// Issue #114: the compact form is the same document, not a lesser one.
    #[test]
    fn compact_round_trips_to_the_same_creature_and_tags_as_pretty() {
        let creature = parse_creature_json(TINY_TAGGED).unwrap();
        let mut meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        meta.stamp_acceptance(&LamarckProgress {
            acceptances: 1,
            score: 0.2,
            error: 0.8,
            opening_score: 0.1,
            focus_neuron: "o1",
            strategy: CandidateStrategy::Backprop,
            experiments: 3,
        });
        let pretty = serialize_creature_with_meta(&creature, &meta).unwrap();
        let compact = serialize_creature_with_meta_compact(&creature, &meta).unwrap();

        assert!(
            !compact.contains('\n'),
            "compact output must be one line: {compact}"
        );
        assert!(
            compact.len() < pretty.len(),
            "compact ({}) must be smaller than pretty ({})",
            compact.len(),
            pretty.len()
        );

        // Same creature after parsing — formatting never changes a value.
        assert_eq!(
            parse_creature_json(&compact).unwrap(),
            parse_creature_json(&pretty).unwrap()
        );
        // Same meta: uuid and every tag, including the score/error/lamarck
        // stamps `stamp_acceptance` attaches.
        let compact_value: Value = serde_json::from_str(&compact).unwrap();
        let pretty_value: Value = serde_json::from_str(&pretty).unwrap();
        assert_eq!(compact_value, pretty_value);
        assert_eq!(compact_value["uuid"], "creature-1");
        for tag in ["name", "version", "score", "error", "lamarck"] {
            assert!(
                compact_value["tags"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|t| t["name"] == tag),
                "compact output dropped the {tag} tag"
            );
        }
    }

    const TINY_ID_TAGGED: &str = r#"{
      "uuid": "creature-2",
      "input": 1,
      "output": 1,
      "forwardOnly": true,
      "neurons": [{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
      "synapses": [{"fromUUID":"input-0","toUUID":"o1","weight":1.0}],
      "tags": [
        {"name":"score","value":"0.1"},
        {"name":"intelligentDesign","value":"💍  Tacit Knowledge, score: 0.1 improved by 1e-6"}
      ]
    }"#;

    fn progress() -> LamarckProgress<'static> {
        LamarckProgress {
            acceptances: 1,
            score: 0.2,
            error: 0.8,
            opening_score: 0.1,
            focus_neuron: "o1",
            strategy: CandidateStrategy::Backprop,
            experiments: 3,
        }
    }

    /// GRQ #3952: `intelligentDesign` belongs to another program — Lamarck must
    /// never overwrite an existing value with its own run summary.
    #[test]
    fn stamp_acceptance_preserves_existing_intelligent_design_tag() {
        let mut meta = CreatureMeta::from_creature_json(TINY_ID_TAGGED);
        meta.stamp_acceptance(&progress());
        let id = meta
            .tags
            .iter()
            .find(|t| t.name == "intelligentDesign")
            .expect("intelligentDesign tag must survive");
        assert_eq!(id.value, "💍  Tacit Knowledge, score: 0.1 improved by 1e-6");
        let lamarck = meta
            .tags
            .iter()
            .find(|t| t.name == "lamarck")
            .expect("lamarck tag must be stamped");
        assert!(lamarck.value.contains("🦒"));
    }

    /// GRQ #3952: and never invents the tag on a creature that has none.
    #[test]
    fn stamp_acceptance_does_not_add_intelligent_design_tag() {
        let mut meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        meta.stamp_acceptance(&progress());
        assert!(
            !meta.tags.iter().any(|t| t.name == "intelligentDesign"),
            "Lamarck must not create an intelligentDesign tag"
        );
        assert!(meta.tags.iter().any(|t| t.name == "lamarck"));
    }

    #[test]
    fn run_summary_uses_cumulative_delta_and_six_sigfigs() {
        let msg = lamarck_progress_message(&LamarckProgress {
            acceptances: 2,
            score: 0.3451532296337825,
            error: 0.5,
            opening_score: 0.3451500296337825,
            focus_neuron: "neuron-1343748843",
            strategy: CandidateStrategy::Random,
            experiments: 75,
        });
        assert_eq!(
            msg,
            "🦒 Lamarck · 2 accepts / 75 exps · last: 🎲 random · 🎯 neuron-1343748843 · score: 0.345153 improved by 3.2e-06"
        );
    }

    #[test]
    fn format_g_matches_awk_style_sigfigs() {
        assert_eq!(format_g(0.345153229634, 6), "0.345153");
        assert_eq!(format_g(3.2e-6, 3), "3.2e-06");
        assert_eq!(format_g(0.1, 6), "0.1");
    }

    /// Issue #187: a GRQ-sampler creature carrying per-neuron discovery /
    /// intelligentDesign provenance.
    const NEURON_TAGGED: &str = r#"{
      "uuid": "creature-3",
      "input": 2,
      "output": 1,
      "forwardOnly": true,
      "neurons": [
        {"type":"hidden","uuid":"h1","bias":0.5,"squash":"TANH","tags":[
          {"name":"discovered","value":"2026-07-01"},
          {"name":"discovery-comment","value":"split of input-0 → o1"}
        ]},
        {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY","tags":[
          {"name":"intelligentDesign","value":"💍  Tacit Knowledge"}
        ]}
      ],
      "synapses": [
        {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
        {"fromUUID":"h1","toUUID":"o1","weight":1.0},
        {"fromUUID":"input-1","toUUID":"o1","weight":0.25}
      ],
      "tags": [{"name":"name","value":"Yara Richardson"}]
    }"#;

    fn neuron_tags_of<'a>(value: &'a Value, uuid: &str) -> Option<&'a Vec<Value>> {
        value["neurons"]
            .as_array()?
            .iter()
            .find(|n| n["uuid"] == uuid)?
            .get("tags")?
            .as_array()
    }

    /// Issue #187: `neurons[].tags` are parsed off the source and keyed by uuid.
    #[test]
    fn extract_preserves_neuron_tags_keyed_by_uuid() {
        let meta = CreatureMeta::from_creature_json(NEURON_TAGGED);
        let h1 = meta.neuron_tags.get("h1").expect("h1 tags");
        assert_eq!(h1.len(), 2);
        assert_eq!(h1[0].name, "discovered");
        assert_eq!(h1[0].value, "2026-07-01");
        assert_eq!(h1[1].name, "discovery-comment");
        let o1 = meta.neuron_tags.get("o1").expect("o1 tags");
        assert_eq!(o1[0].name, "intelligentDesign");
        assert_eq!(o1[0].value, "💍  Tacit Knowledge");
        assert!(!meta.neuron_tags.contains_key("missing"));
    }

    /// Issue #187: the check-in document re-attaches them by uuid, in both forms.
    #[test]
    fn serialize_reattaches_neuron_tags_by_uuid() {
        let creature = parse_creature_json(NEURON_TAGGED).unwrap();
        let meta = CreatureMeta::from_creature_json(NEURON_TAGGED);
        for text in [
            serialize_creature_with_meta(&creature, &meta).unwrap(),
            serialize_creature_with_meta_compact(&creature, &meta).unwrap(),
        ] {
            let value: Value = serde_json::from_str(&text).unwrap();
            let h1 = neuron_tags_of(&value, "h1").expect("h1 keeps its tags");
            assert_eq!(h1.len(), 2);
            assert_eq!(h1[0]["name"], "discovered");
            assert_eq!(h1[1]["value"], "split of input-0 → o1");
            let o1 = neuron_tags_of(&value, "o1").expect("o1 keeps its tags");
            assert_eq!(o1[0]["name"], "intelligentDesign");
            // Creature-level tags are untouched by the per-neuron re-attach.
            assert_eq!(value["tags"][0]["name"], "name");
            // Still a valid creature after the round trip.
            assert_eq!(parse_creature_json(&text).unwrap(), creature);
        }
    }

    /// Issue #187: a neuron the source never tagged gains no empty `tags` key.
    #[test]
    fn serialize_leaves_untagged_neurons_alone() {
        let creature = parse_creature_json(TINY_TAGGED).unwrap();
        let meta = CreatureMeta::from_creature_json(TINY_TAGGED);
        let text = serialize_creature_with_meta(&creature, &meta).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert!(value["neurons"][0].get("tags").is_none());
    }

    /// Issue #187: weight/bias edits keep the surviving neurons' provenance —
    /// re-attachment is by uuid, not by position.
    #[test]
    fn neuron_tags_survive_edits_that_reorder_neurons() {
        let mut creature = parse_creature_json(NEURON_TAGGED).unwrap();
        let meta = CreatureMeta::from_creature_json(NEURON_TAGGED);
        creature.neurons.reverse();
        creature.neurons[1].bias = 9.5;
        let text = serialize_creature_with_meta(&creature, &meta).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["neurons"][0]["uuid"], "o1");
        assert_eq!(
            neuron_tags_of(&value, "o1").unwrap()[0]["name"],
            "intelligentDesign"
        );
        assert_eq!(neuron_tags_of(&value, "h1").unwrap().len(), 2);
    }

    /// Issue #187: neurons Lamarck grows carry their own origin tag, and only
    /// those — the neurons that were already there are not restamped.
    #[test]
    fn stamp_new_neurons_tags_only_the_neurons_lamarck_added() {
        let previous = parse_creature_json(NEURON_TAGGED).unwrap();
        let mut current = previous.clone();
        current.neurons.insert(
            1,
            neat_core::NeuronExport {
                id: None,
                neuron_type: "hidden".into(),
                uuid: "h-new".into(),
                bias: 0.0,
                squash: Some("TANH".into()),
            },
        );
        let mut meta = CreatureMeta::from_creature_json(NEURON_TAGGED);
        meta.stamp_new_neurons(
            &previous,
            &current,
            &NeuronOrigin {
                strategy: CandidateStrategy::StructuralAddNeuron,
                focus_neuron: "o1",
                experiment: 42,
            },
        );

        let grown = meta.neuron_tags.get("h-new").expect("new neuron is tagged");
        assert_eq!(grown.len(), 1);
        assert_eq!(grown[0].name, "lamarck");
        assert!(grown[0].value.contains("🦒"));
        assert!(grown[0].value.contains("structural_add_neuron"));
        assert!(grown[0].value.contains("o1"));
        assert!(grown[0].value.contains("exp 42"));

        // Pre-existing neurons keep exactly the provenance they arrived with.
        assert_eq!(meta.neuron_tags.get("h1").unwrap().len(), 2);
        assert_eq!(meta.neuron_tags.get("o1").unwrap().len(), 1);
        assert_eq!(
            meta.neuron_tags.get("o1").unwrap()[0].name,
            "intelligentDesign"
        );
    }

    /// Issue #187: a second accept re-stamps the same neuron rather than
    /// stacking duplicate `lamarck` tags on it.
    #[test]
    fn stamp_new_neurons_upserts_its_own_tag() {
        let previous = parse_creature_json(NEURON_TAGGED).unwrap();
        let mut current = previous.clone();
        current.neurons.push(neat_core::NeuronExport {
            id: None,
            neuron_type: "hidden".into(),
            uuid: "h-new".into(),
            bias: 0.0,
            squash: None,
        });
        let mut meta = CreatureMeta::from_creature_json(NEURON_TAGGED);
        for experiment in [7, 9] {
            meta.stamp_new_neurons(
                &previous,
                &current,
                &NeuronOrigin {
                    strategy: CandidateStrategy::StructuralAdd,
                    focus_neuron: "o1",
                    experiment,
                },
            );
        }
        let grown = meta.neuron_tags.get("h-new").unwrap();
        assert_eq!(grown.len(), 1);
        assert!(grown[0].value.contains("exp 9"));
    }

    /// Issue #187: the grown neuron's origin tag reaches the written document.
    #[test]
    fn serialize_writes_the_origin_tag_of_a_grown_neuron() {
        let previous = parse_creature_json(NEURON_TAGGED).unwrap();
        let mut current = previous.clone();
        current.neurons.insert(
            0,
            neat_core::NeuronExport {
                id: None,
                neuron_type: "hidden".into(),
                uuid: "h-new".into(),
                bias: 0.25,
                squash: Some("TANH".into()),
            },
        );
        current.synapses.push(neat_core::SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "h-new".into(),
            weight: 1.0,
            synapse_type: None,
        });
        current.synapses.push(neat_core::SynapseExport {
            from_uuid: "h-new".into(),
            to_uuid: "o1".into(),
            weight: 1.0,
            synapse_type: None,
        });
        let mut meta = CreatureMeta::from_creature_json(NEURON_TAGGED);
        meta.stamp_new_neurons(
            &previous,
            &current,
            &NeuronOrigin {
                strategy: CandidateStrategy::StructuralAddNeuron,
                focus_neuron: "o1",
                experiment: 11,
            },
        );
        let text = serialize_creature_with_meta(&current, &meta).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        let grown = neuron_tags_of(&value, "h-new").expect("grown neuron is tagged");
        assert_eq!(grown[0]["name"], "lamarck");
        assert!(grown[0]["value"].as_str().unwrap().contains("🧩"));
        assert_eq!(neuron_tags_of(&value, "h1").unwrap().len(), 2);
    }

    #[test]
    fn neuron_origin_message_names_strategy_focus_and_experiment() {
        assert_eq!(
            lamarck_neuron_origin_message(&NeuronOrigin {
                strategy: CandidateStrategy::StructuralAddNeuron,
                focus_neuron: "neuron-1343748843",
                experiment: 75,
            }),
            "🦒 Lamarck · grown by 🧩 structural_add_neuron · 🎯 neuron-1343748843 · exp 75"
        );
    }
}
