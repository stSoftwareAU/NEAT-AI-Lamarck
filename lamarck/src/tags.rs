//! Creature check-in tags (`{ name, value }[]`) for GRQ.
//!
//! `neat_core::CreatureExport` round-trips a creature's metadata natively —
//! top-level `uuid` / `tags` / `memetic`, per-neuron and per-synapse `tags`,
//! and any key the Rust structs do not declare (NEAT-AI#3747 / NEAT-AI#3748).
//! This module therefore no longer mirrors that metadata on the side: it only
//! stamps the tags this optimiser owns — `score` / `error` / `lamarck` — onto
//! the creature's own tag list, and writes the check-in form
//! (`worker/Lamarck/run.sh` reads those tags as-is).

use crate::candidates::CandidateStrategy;
use neat_core::{CreatureExport, CreatureTag, creature_to_json_pretty};

/// Insert or replace a tag by name, keeping the order of first insert.
///
/// A creature with no `tags` key gains one; every other tag is left alone.
pub fn upsert_tag(creature: &mut CreatureExport, name: &str, value: impl Into<String>) {
    let value = value.into();
    let tags = creature.tags.get_or_insert_with(Vec::new);
    if let Some(existing) = tags.iter_mut().find(|t| t.name == name) {
        existing.value = value;
    } else {
        tags.push(CreatureTag {
            name: name.to_string(),
            value,
        });
    }
}

/// The value of a creature-level tag, or `None` when it carries none.
pub fn tag_value<'a>(creature: &'a CreatureExport, name: &str) -> Option<&'a str> {
    creature
        .tags
        .as_ref()?
        .iter()
        .find(|t| t.name == name)
        .map(|t| t.value.as_str())
}

/// Update score/error and stamp a run-level Lamarck summary for check-in.
///
/// Only the `lamarck` tag is ours. `intelligentDesign` belongs to the
/// Intelligent Design program — stamping it here overwrote another program's
/// provenance (GRQ #3952), so never touch it.
pub fn stamp_acceptance(creature: &mut CreatureExport, progress: &LamarckProgress<'_>) {
    // Keep full-precision numeric tags for machine consumers.
    upsert_tag(creature, "score", format!("{}", progress.score));
    upsert_tag(creature, "error", format!("{}", progress.error));
    upsert_tag(creature, "lamarck", lamarck_progress_message(progress));
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
        CandidateStrategy::StatsSkewBias => ("📉", "stats_skew_bias"),
        CandidateStrategy::StructuralAdd => ("🌱", "structural_add"),
        CandidateStrategy::StructuralAddNeuron => ("🧩", "structural_add_neuron"),
        CandidateStrategy::StructuralWeaken => ("✂️", "structural_weaken"),
        CandidateStrategy::Random => ("🎲", "random"),
    }
}

/// Pretty-print a creature for check-in, newline-terminated.
///
/// neat-core round-trips the creature's metadata itself (NEAT-AI#3747 /
/// NEAT-AI#3748), so this writes the parsed creature straight out — nothing is
/// re-attached on the side, and nothing passes through a `serde_json::Value`
/// whose sorted map would re-order the `memetic` block. This is the
/// **human-facing** form (`best.json`, `winners/`); scorer-facing batch files
/// use `neat_core::creature_to_json` compact (issue #114).
pub fn serialize_creature_pretty(creature: &CreatureExport) -> Result<String, String> {
    // Match typical NEAT export: trailing newline after pretty JSON.
    let mut out = creature_to_json_pretty(creature).map_err(|e| e.to_string())?;
    if !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::{creature_to_json, parse_creature_json};
    use serde_json::Value;

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

    const UNTAGGED: &str = r#"{
      "input": 1,
      "output": 1,
      "forwardOnly": true,
      "neurons": [{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
      "synapses": [{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
    }"#;

    /// The parse side of the contract this module now leans on: neat-core
    /// hands back `uuid` and `tags` rather than dropping them.
    #[test]
    fn parse_keeps_uuid_and_tags() {
        let creature = parse_creature_json(TINY_TAGGED).unwrap();
        assert_eq!(creature.uuid.as_deref(), Some("creature-1"));
        let tags = creature.tags.expect("creature keeps its tags");
        assert_eq!(tags[0].name, "name");
        assert_eq!(tags[0].value, "Yara Richardson");
    }

    #[test]
    fn serialize_round_trips_original_tags_plus_lamarck() {
        let mut creature = parse_creature_json(TINY_TAGGED).unwrap();
        stamp_acceptance(
            &mut creature,
            &LamarckProgress {
                acceptances: 1,
                score: 0.2,
                error: 0.8,
                opening_score: 0.1,
                focus_neuron: "o1",
                strategy: CandidateStrategy::Backprop,
                experiments: 3,
            },
        );
        let json = serialize_creature_pretty(&creature).unwrap();
        assert!(json.ends_with('\n'), "check-in files end with a newline");
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
        let mut creature = parse_creature_json(TINY_TAGGED).unwrap();
        stamp_acceptance(
            &mut creature,
            &LamarckProgress {
                acceptances: 1,
                score: 0.2,
                error: 0.8,
                opening_score: 0.1,
                focus_neuron: "o1",
                strategy: CandidateStrategy::Backprop,
                experiments: 3,
            },
        );
        let pretty = serialize_creature_pretty(&creature).unwrap();
        let compact = creature_to_json(&creature).unwrap();

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
        let mut creature = parse_creature_json(TINY_ID_TAGGED).unwrap();
        stamp_acceptance(&mut creature, &progress());
        assert_eq!(
            tag_value(&creature, "intelligentDesign"),
            Some("💍  Tacit Knowledge, score: 0.1 improved by 1e-6"),
            "intelligentDesign tag must survive"
        );
        let lamarck = tag_value(&creature, "lamarck").expect("lamarck tag must be stamped");
        assert!(lamarck.contains("🦒"));
    }

    /// GRQ #3952: and never invents the tag on a creature that has none.
    #[test]
    fn stamp_acceptance_does_not_add_intelligent_design_tag() {
        let mut creature = parse_creature_json(TINY_TAGGED).unwrap();
        stamp_acceptance(&mut creature, &progress());
        assert_eq!(
            tag_value(&creature, "intelligentDesign"),
            None,
            "Lamarck must not create an intelligentDesign tag"
        );
        assert!(tag_value(&creature, "lamarck").is_some());
    }

    #[test]
    fn stamp_gives_an_untagged_creature_its_first_tags() {
        let mut creature = parse_creature_json(UNTAGGED).unwrap();
        assert!(creature.tags.is_none(), "fixture starts without tags");
        stamp_acceptance(&mut creature, &progress());
        let names: Vec<&str> = creature
            .tags
            .as_ref()
            .expect("stamping creates the tag list")
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(names, vec!["score", "error", "lamarck"]);
    }

    /// `worker/Lamarck/run.sh` parses these values: they are stamped verbatim,
    /// at full precision, never rounded for display.
    #[test]
    fn score_and_error_are_stamped_at_full_precision() {
        let mut creature = parse_creature_json(TINY_TAGGED).unwrap();
        stamp_acceptance(
            &mut creature,
            &LamarckProgress {
                acceptances: 2,
                score: 0.3451532296337825,
                error: 0.6548467703662175,
                opening_score: 0.3451500296337825,
                focus_neuron: "o1",
                strategy: CandidateStrategy::Random,
                experiments: 75,
            },
        );
        assert_eq!(tag_value(&creature, "score"), Some("0.3451532296337825"));
        assert_eq!(tag_value(&creature, "error"), Some("0.6548467703662175"));
        assert_eq!(
            tag_value(&creature, "lamarck"),
            Some(
                "🦒 Lamarck · 2 accepts / 75 exps · last: 🎲 random · 🎯 o1 · score: 0.345153 improved by 3.2e-06"
            )
        );
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
