//! Creature JSON tags (`{ name, value }[]`) for check-in / GRQ compatibility.
//!
//! `neat_core::CreatureExport` does not round-trip `tags` / `uuid`, so Lamarck
//! keeps them in [`CreatureMeta`] and re-attaches on write — preserving the
//! original pedigree (name, version, …) while stamping Lamarck progress in the
//! same fun emoji style as GRQ `worker/IntelligentDesign/run.sh`.

use crate::candidates::CandidateStrategy;
use neat_core::{CreatureExport, creature_to_json_pretty};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// One creature tag (NEAT-AI / `@stsoftware/tags` shape).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreatureTag {
    /// Tag key (`score`, `error`, `name`, `lamarck`, …).
    pub name: String,
    /// Tag value (always a string in the export format).
    pub value: String,
}

/// Top-level fields stripped by `parse_creature_json` that we must keep.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CreatureMeta {
    /// Optional creature UUID from the source JSON.
    pub uuid: Option<String>,
    /// Ordered tags (upserts replace by name, preserving order of first insert).
    pub tags: Vec<CreatureTag>,
}

impl CreatureMeta {
    /// Parse `uuid` + `tags` from raw creature JSON (missing → empty).
    pub fn from_creature_json(text: &str) -> Self {
        let Ok(value) = serde_json::from_str::<Value>(text) else {
            return Self::default();
        };
        let uuid = value
            .get("uuid")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let tags = value
            .get("tags")
            .and_then(|v| v.as_array())
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
            .unwrap_or_default();
        Self { uuid, tags }
    }

    /// Insert or replace a tag by name.
    pub fn upsert(&mut self, name: &str, value: impl Into<String>) {
        let value = value.into();
        if let Some(existing) = self.tags.iter_mut().find(|t| t.name == name) {
            existing.value = value;
        } else {
            self.tags.push(CreatureTag {
                name: name.to_string(),
                value,
            });
        }
    }

    /// Update score/error and append a fun Lamarck progress tag after an accept.
    pub fn stamp_acceptance(&mut self, progress: &LamarckProgress<'_>) {
        self.upsert("score", format!("{}", progress.score));
        self.upsert("error", format!("{}", progress.error));
        let message = lamarck_progress_message(progress);
        self.upsert("lamarck", message.clone());
        // GRQ check-in scripts often read `intelligentDesign` for the commit blurb.
        self.upsert("intelligentDesign", message);
    }
}

/// Fields for one accepted Lamarck improvement (used for emoji progress tags).
#[derive(Debug, Clone, Copy)]
pub struct LamarckProgress<'a> {
    /// 1-based acceptance count for this run.
    pub accept_number: u64,
    /// Authoritative score after accept.
    pub score: f64,
    /// Authoritative error after accept.
    pub error: f64,
    /// Score delta vs prior baseline.
    pub delta: f64,
    /// Focus neuron UUID for this experiment.
    pub focus_neuron: &'a str,
    /// Winning candidate strategy.
    pub strategy: CandidateStrategy,
    /// Experiment number that produced the accept.
    pub experiments: u64,
}

/// Emoji-rich progress blurb (same spirit as GRQ IntelligentDesign `run.sh` logs).
pub fn lamarck_progress_message(progress: &LamarckProgress<'_>) -> String {
    let (strat_emoji, strat_label) = strategy_emoji(progress.strategy);
    format!(
        "🦒 Lamarck accept #{} · Δ{:+.3e} · {strat_emoji} {strat_label} · 🎯 {} · 🧪 exp {} · 🏆 score {:.12}",
        progress.accept_number,
        progress.delta,
        progress.focus_neuron,
        progress.experiments,
        progress.score,
    )
}

fn strategy_emoji(strategy: CandidateStrategy) -> (&'static str, &'static str) {
    match strategy {
        CandidateStrategy::Backprop => ("🧠", "backprop"),
        CandidateStrategy::MeanErrorBias => ("📏", "mean_error_bias"),
        CandidateStrategy::StatsWeight => ("📊", "stats_weight"),
        CandidateStrategy::StatsBias => ("📐", "stats_bias"),
        CandidateStrategy::StructuralAdd => ("🌱", "structural_add"),
        CandidateStrategy::StructuralAddNeuron => ("🧩", "structural_add_neuron"),
        CandidateStrategy::StructuralWeaken => ("✂️", "structural_weaken"),
        CandidateStrategy::Random => ("🎲", "random"),
    }
}

/// Pretty-print a creature with `uuid` / `tags` re-attached for check-in.
pub fn serialize_creature_with_meta(
    creature: &CreatureExport,
    meta: &CreatureMeta,
) -> Result<String, String> {
    let body = creature_to_json_pretty(creature).map_err(|e| e.to_string())?;
    let mut value: Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    if let Some(uuid) = &meta.uuid {
        value["uuid"] = json!(uuid);
    }
    if !meta.tags.is_empty() {
        value["tags"] = serde_json::to_value(&meta.tags).map_err(|e| e.to_string())?;
    }
    // Match typical NEAT export: trailing newline after pretty JSON.
    let mut out = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    if !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
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
            accept_number: 1,
            score: 0.2,
            error: 0.8,
            delta: 1e-5,
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
        assert!(msg.contains("accept #1"));
    }
}
