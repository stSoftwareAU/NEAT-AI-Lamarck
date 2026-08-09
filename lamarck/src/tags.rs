//! Creature JSON tags (`{ name, value }[]`) for check-in / GRQ compatibility.
//!
//! `neat_core::CreatureExport` does not round-trip `tags` / `uuid`, so Lamarck
//! keeps them in [`CreatureMeta`] and re-attaches on write — preserving the
//! original pedigree (name, version, …) while stamping a **run-level** Lamarck
//! summary for GRQ check-in (`worker/Lamarck/run.sh` reads the tag as-is).

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

    /// Update score/error and stamp a run-level Lamarck summary for check-in.
    pub fn stamp_acceptance(&mut self, progress: &LamarckProgress<'_>) {
        // Keep full-precision numeric tags for machine consumers.
        self.upsert("score", format!("{}", progress.score));
        self.upsert("error", format!("{}", progress.error));
        let message = lamarck_progress_message(progress);
        self.upsert("lamarck", message.clone());
        // GRQ check-in scripts often read `intelligentDesign` for the commit blurb.
        self.upsert("intelligentDesign", message);
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
}
