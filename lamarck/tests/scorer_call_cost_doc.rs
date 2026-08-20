//! `docs/scorer-call-cost.md` ↔ code contract (Issue #112).
//!
//! The document's whole value is that its numbers can be re-measured and that
//! the conditions they were measured under are on the page. These tests guard
//! the ways that decays: the tooling it points at going away, the report fields
//! it quotes being renamed, the load conditions being dropped, and the go/no-go
//! it exists to deliver being softened into a suggestion.

use neat_ai_lamarck::report::JournalReport;
use std::path::{Path, PathBuf};

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn doc() -> String {
    let path = repo_path("docs/scorer-call-cost.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// The document tells the reader to run these, so they have to exist.
#[test]
fn the_document_names_tooling_that_exists() {
    let doc = doc();
    for tool in [
        "scripts/measure-scorer-call-cost.sh",
        "lamarck/examples/scorer_call_cost_bench.rs",
        "lamarck/src/scorer_cost.rs",
        "docs/evidence/scorer-call-cost/rate-1.log",
        "docs/evidence/scorer-call-cost/rate-0_05.log",
    ] {
        assert!(doc.contains(tool), "docs/scorer-call-cost.md drops {tool}");
        assert!(
            repo_path(tool).exists(),
            "docs/scorer-call-cost.md points at a missing {tool}"
        );
    }
}

/// Every JSON field the document quotes is one `report` actually emits.
#[test]
fn every_report_field_the_document_quotes_is_serialised() {
    let doc = doc();
    let report = serde_json::to_value(empty_report()).expect("report serialises");
    let cost = report
        .get("scorerCallCost")
        .expect("report carries the scorerCallCost section");
    assert!(
        doc.contains("scorerCallCost"),
        "docs/scorer-call-cost.md no longer says which report section it is built from"
    );
    for field in ["calls", "failedCalls", "creaturesScored", "byPhase"] {
        assert!(
            cost.get(field).is_some(),
            "`report` no longer emits `scorerCallCost.{field}`, which the document is built on"
        );
        assert!(
            doc.contains(field),
            "docs/scorer-call-cost.md drops the {field} field"
        );
    }
    // The per-phase fit is the measurement itself.
    for field in ["fixedMs", "marginalMsPerCreature", "fixedMsShareAtMean"] {
        assert!(
            doc.contains(field),
            "docs/scorer-call-cost.md drops the {field} field"
        );
    }
}

/// A fixed cost measured beside a live scorer run is meaningless, so the
/// conditions are part of the result — the review-time gate from issue #112.
#[test]
fn the_document_records_the_load_conditions() {
    let doc = doc();
    for marker in ["loadBefore", "loadAfter"] {
        assert!(
            doc.contains(marker),
            "docs/scorer-call-cost.md dropped {marker} — the measurement conditions are the result"
        );
    }
}

/// The deliverable is a decision, not a recommendation for somebody else.
#[test]
fn the_document_carries_an_explicit_go_no_go_with_an_estimated_saving() {
    let doc = doc();
    assert!(
        doc.contains("## Decision"),
        "docs/scorer-call-cost.md has no decision section"
    );
    let decided = doc.contains("**Go**") || doc.contains("**No-go**");
    assert!(
        decided,
        "docs/scorer-call-cost.md states no explicit go/no-go on a persistent scoring session"
    );
    assert!(
        doc.contains("NEAT-AI-scorer#536"),
        "the decision must be cross-referenced to the scorer-side survey"
    );
}

/// The over-claim gate: the document must keep saying what it cannot support.
#[test]
fn the_document_states_what_the_measurement_cannot_support() {
    let doc = doc();
    assert!(
        doc.contains("What this measurement cannot support"),
        "the limits section is the review gate against a decision drawn from a loaded box"
    );
    for claim in [
        // The box carried a competing production scorer throughout.
        "not an idle box",
        // Lamarck cannot see inside the scorer process.
        "cannot attribute the fixed cost",
    ] {
        assert!(
            doc.contains(claim),
            "docs/scorer-call-cost.md dropped the limit: {claim}"
        );
    }
}

/// Marginal per-creature costs (ms) the Result table measures, by phase.
///
/// Parsed from the table itself so the guard tracks a re-measurement instead of
/// a number frozen into a test.
fn measured_marginal_ms_per_creature() -> Vec<(String, f64)> {
    let doc = doc();
    let mut measured = Vec::new();
    for line in doc.lines() {
        let fields: Vec<&str> = line.split('|').map(str::trim).collect();
        // "| phase | sample rate | calls | fixedMs | marginalMsPerCreature | …"
        if fields.len() < 6 {
            continue;
        }
        let phase = fields[1];
        if phase != "screen" && phase != "promote" {
            continue;
        }
        let marginal = fields[5]
            .trim_end_matches("ms")
            .replace([' ', ','], "")
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("Result table row for {phase} has no marginal cost: {e}"));
        measured.push((phase.to_string(), marginal));
    }
    assert_eq!(
        measured.len(),
        2,
        "docs/scorer-call-cost.md no longer measures both phases in its Result table"
    );
    measured
}

fn readme_text() -> String {
    let path = repo_path("README.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Numbers written immediately before a unit, newest first, range-aware.
///
/// `"≈0.7–1"` yields `[1.0, 0.7]`; text with no trailing number yields `[]`.
fn trailing_numbers(prefix: &str) -> Vec<f64> {
    let mut chars: Vec<char> = prefix.chars().collect();
    let mut numbers = Vec::new();
    loop {
        while chars.last().is_some_and(|c| c.is_whitespace()) {
            chars.pop();
        }
        let mut digits = String::new();
        while chars
            .last()
            .is_some_and(|c| c.is_ascii_digit() || *c == '.')
        {
            digits.insert(0, chars.pop().expect("checked above"));
        }
        match digits.parse::<f64>() {
            Ok(value) => numbers.push(value),
            Err(_) => return numbers,
        }
        // Keep walking only across a range separator, e.g. "0.7–1s".
        if chars.last().is_some_and(|c| *c == '–' || *c == '-') {
            chars.pop();
        } else {
            return numbers;
        }
    }
}

/// Per-creature scorer cost claims in `text`, in milliseconds, with context.
fn per_creature_cost_claims_ms(text: &str) -> Vec<(f64, String)> {
    let mut claims = Vec::new();
    for suffix in ["s/creature", "s per creature"] {
        let mut from = 0;
        while let Some(hit) = text[from..].find(suffix) {
            let at = from + hit;
            from = at + suffix.len();
            // `at` indexes the unit's `s`; a `m` in front of it makes it ms.
            let prefix = &text[..at];
            let (prefix, scale) = match prefix.strip_suffix('m') {
                Some(shorter) => (shorter, 1.0),
                None => (prefix, 1000.0),
            };
            let context: String = prefix
                .chars()
                .rev()
                .take(40)
                .collect::<Vec<char>>()
                .into_iter()
                .rev()
                .collect();
            for value in trailing_numbers(prefix) {
                claims.push((value * scale, format!("{context}{suffix}")));
            }
        }
    }
    claims
}

/// The README must not contradict the measurement it cites (issue #172).
///
/// Every per-creature scorer cost the README states has to match a phase this
/// document actually measured — the README's Phase 5 walkthrough once claimed
/// figures roughly double the fitted ones.
#[test]
fn every_per_creature_cost_the_readme_states_matches_a_measured_phase() {
    let measured = measured_marginal_ms_per_creature();
    let claims = per_creature_cost_claims_ms(&readme_text());
    for (claimed_ms, context) in claims {
        let closest = measured
            .iter()
            .min_by(|a, b| {
                (a.1 - claimed_ms)
                    .abs()
                    .total_cmp(&(b.1 - claimed_ms).abs())
            })
            .expect("both phases measured above");
        let error = (closest.1 - claimed_ms).abs() / closest.1;
        assert!(
            error <= 0.2,
            "README states {claimed_ms} ms/creature ({context:?}), but the nearest phase \
             docs/scorer-call-cost.md measures is {} at {} ms/creature — link to the doc \
             rather than restating its numbers",
            closest.0,
            closest.1
        );
    }
}

/// Where the README dropped those numbers, it has to point at this document.
#[test]
fn the_readme_screen_step_points_at_this_document() {
    let readme = readme_text();
    let start = readme
        .find("Scoring runs in three steps:")
        .expect("README.md has no Phase 5 scoring walkthrough");
    let steps = &readme[start..];
    let end = steps.find("\n### ").unwrap_or(steps.len());
    let steps = &steps[..end];
    assert!(
        steps.contains("docs/scorer-call-cost.md"),
        "the Phase 5 scoring walkthrough no longer points at docs/scorer-call-cost.md \
         for what a screen call costs against a full-corpus one"
    );
}

/// A `report` over no experiments at all — the shape check above needs one.
fn empty_report() -> JournalReport {
    let journal = tempfile::NamedTempFile::new().expect("temp journal");
    neat_ai_lamarck::report_from_journal(journal.path()).expect("an empty journal reports")
}
