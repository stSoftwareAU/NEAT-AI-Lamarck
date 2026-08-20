//! `docs/promote-gate.md` ↔ code contract (Issue #111).
//!
//! The document's value is that every figure in it is reproducible from a
//! journal and carries the sample it came from. These tests guard the ways
//! that decays: the tooling it points at going away, the report fields it
//! quotes being renamed, and — the review-time gate against a confident
//! recommendation drawn from two accepts — its limits section being deleted.
//!
//! *The two gates* also used to price the two scorer calls inline, at figures
//! predating issue #112's fixed/marginal decomposition (Issue #183), so the
//! same "no timing the measurement contradicts" contract the README carries is
//! asserted here over that section.

mod common;

use common::{measured_scorer_costs_ms, read, repo_path, section, timings_contradicting};
use neat_ai_lamarck::report::JournalReport;

fn doc() -> String {
    read("docs/promote-gate.md")
}

/// The section that introduces the screen-versus-promote economics.
fn two_gates_section() -> String {
    section(&doc(), "\n## The two gates").to_string()
}

/// The document tells the reader to run these, so they have to exist.
#[test]
fn the_document_names_tooling_that_exists() {
    let doc = doc();
    for tool in [
        "scripts/summarise-promote-gate.sh",
        "scripts/run-followup-economics.sh",
        "lamarck/src/promote_gate.rs",
        "lamarck/tests/promote_gate_replay.rs",
    ] {
        assert!(doc.contains(tool), "docs/promote-gate.md drops {tool}");
        assert!(
            repo_path(tool).exists(),
            "docs/promote-gate.md points at a missing {tool}"
        );
    }
}

/// Every JSON field the document quotes is one `report` actually emits.
#[test]
fn every_report_field_the_document_quotes_is_serialised() {
    let doc = doc();
    let report = serde_json::to_value(empty_report()).expect("report serialises");
    let replay = report
        .get("promoteGateReplay")
        .expect("report carries the promoteGateReplay section");
    assert!(
        doc.contains("promoteGateReplay"),
        "docs/promote-gate.md no longer says which report section it is built from"
    );
    for field in [
        "gateAsRun",
        "screened",
        "promotedAsRun",
        "promotedUnderGate",
        "promotionsAvoided",
        "acceptsKept",
        "acceptsDropped",
        "accepts",
    ] {
        assert!(
            replay.get(field).is_some(),
            "`report` no longer emits `{field}`, which the document is built on"
        );
    }
}

/// The default is the whole safety argument: the document must keep saying it
/// is unchanged, and the CLI must keep it that way.
#[test]
fn the_document_states_that_no_default_changed() {
    let doc = doc();
    assert!(
        doc.contains("No default is changed by this issue"),
        "docs/promote-gate.md dropped the unchanged-default statement"
    );
    assert!(
        doc.contains("`absolute`"),
        "docs/promote-gate.md no longer names the default gate"
    );
}

/// The over-claim gate: the document must keep saying what it cannot support.
#[test]
fn the_document_states_what_the_replay_cannot_support() {
    let doc = doc();
    assert!(
        doc.contains("What this replay cannot support"),
        "the limits section is the review gate against a two-accept recommendation"
    );
    for claim in [
        // A rejected candidate is never full-scored.
        "cannot establish a false-negative rate",
        // Zero promotions was measured where nothing was ever accepted.
        "accepted nothing",
        // Six arms share one seed and replay one experiment stream.
        "not 222 independent samples",
    ] {
        assert!(
            doc.contains(claim),
            "docs/promote-gate.md dropped the limit: {claim}"
        );
    }
}

/// Every timing *The two gates* states must be one `docs/scorer-call-cost.md`
/// measured — prose and Mermaid labels alike (Issue #183).
#[test]
fn the_two_gates_section_states_no_timing_the_measurement_contradicts() {
    let supported = measured_scorer_costs_ms();
    let contradicted = timings_contradicting(&two_gates_section(), &supported);

    assert!(
        contradicted.is_empty(),
        "docs/promote-gate.md `## The two gates` states scorer timings \
         docs/scorer-call-cost.md does not measure: {contradicted:?} — measured ms: \
         {supported:?}. Cite the document rather than restating its numbers."
    );
}

/// The section exists to explain why a full-corpus score is worth screening
/// for, so it has to point at what that call actually costs.
#[test]
fn the_two_gates_section_cites_the_measured_scorer_call_cost_document() {
    assert!(
        two_gates_section().contains("scorer-call-cost.md"),
        "docs/promote-gate.md `## The two gates` does not link docs/scorer-call-cost.md \
         as the source of the screen-versus-promote cost"
    );
}

/// A `report` over no experiments at all — the shape check above needs one.
fn empty_report() -> JournalReport {
    let journal = tempfile::NamedTempFile::new().expect("temp journal");
    neat_ai_lamarck::report_from_journal(journal.path()).expect("an empty journal reports")
}
