//! `docs/strategy-allocation.md` ↔ code contract (Issue #218).
//!
//! The document describes a mechanism that decides how a run spends its
//! candidate budget, so a reader has to be able to trust every knob, field and
//! default it quotes. These tests guard the ways that decays: the tooling it
//! points at going away, the report fields it tabulates being renamed, the
//! defaults it states drifting from the code, and — the honesty gate — its
//! "not yet run" status outliving the absence of a production A/B.

mod common;

use common::{read, repo_path};
use neat_ai_lamarck::report::JournalReport;
use neat_ai_lamarck::strategy_allocation::{
    DEFAULT_STRATEGY_EVIDENCE_DECAY, DEFAULT_STRATEGY_EXPLORATION_FLOOR,
    INCUMBENT_CHANGE_RETENTION, PROMOTION_REWARD_UNITS,
};

fn doc() -> String {
    read("docs/strategy-allocation.md")
}

/// The document tells the reader to run these, so they have to exist.
#[test]
fn the_document_names_tooling_that_exists() {
    let doc = doc();
    for tool in [
        "scripts/run-strategy-allocation-ab.sh",
        "scripts/summarise-strategy-allocation.sh",
        "lamarck/src/strategy_allocation.rs",
        "lamarck/tests/strategy_allocation.rs",
    ] {
        assert!(
            doc.contains(tool),
            "docs/strategy-allocation.md drops {tool}"
        );
        assert!(
            repo_path(tool).exists(),
            "docs/strategy-allocation.md points at a missing {tool}"
        );
    }
}

/// Every report field the document tabulates is one `report` actually emits.
#[test]
fn every_report_field_the_document_quotes_is_serialised() {
    let doc = doc();
    let report = serde_json::to_value(JournalReport {
        strategy_allocation: neat_ai_lamarck::report::StrategyAllocationReport {
            strategies: vec![neat_ai_lamarck::report::StrategyAllocationRow::default()],
            ..Default::default()
        },
        ..empty_report()
    })
    .expect("report serialises");
    let bucket = report
        .get("strategyAllocation")
        .expect("report carries the strategyAllocation section");
    for field in [
        "mode",
        "explorationFloor",
        "evidenceDecay",
        "allocatedExperiments",
    ] {
        assert!(
            bucket.get(field).is_some(),
            "`report` no longer emits `strategyAllocation.{field}`"
        );
        assert!(
            doc.contains(field),
            "docs/strategy-allocation.md drops the `{field}` field"
        );
    }
    let row = bucket
        .get("strategies")
        .and_then(|rows| rows.get(0))
        .expect("the bucket carries per-strategy rows");
    for field in [
        "strategy",
        "allocatedSlots",
        "trials",
        "promotions",
        "accepts",
        "scoreGain",
        "costMs",
        "estimatedValue",
    ] {
        assert!(
            row.get(field).is_some(),
            "`report` no longer emits `strategyAllocation.strategies[].{field}`"
        );
        assert!(
            doc.contains(field),
            "docs/strategy-allocation.md drops the `{field}` column"
        );
    }
}

/// Every default the document states is the default the code ships.
#[test]
fn the_documented_defaults_are_the_shipped_defaults() {
    let doc = doc();
    for (value, what) in [
        (DEFAULT_STRATEGY_EXPLORATION_FLOOR, "exploration floor"),
        (DEFAULT_STRATEGY_EVIDENCE_DECAY, "evidence decay"),
        (INCUMBENT_CHANGE_RETENTION, "incumbent-change retention"),
        (PROMOTION_REWARD_UNITS, "promote-conversion credit"),
    ] {
        let rendered = format!("{value}");
        assert!(
            doc.contains(&rendered),
            "docs/strategy-allocation.md no longer states the {what} ({rendered})"
        );
    }
}

/// The whole safety argument is that nothing about an untouched run changed:
/// the document must keep saying so, and the CLI must keep it true.
#[test]
fn the_document_states_that_the_default_allocation_is_unchanged() {
    let doc = doc();
    assert!(
        doc.contains("`--strategy-allocation fixed` stays the default"),
        "docs/strategy-allocation.md no longer states that the default is unchanged"
    );
    assert!(
        !neat_ai_lamarck::LamarckConfig::default()
            .strategy_allocation
            .is_adaptive(),
        "the shipped default is no longer the fixed allocation the document promises"
    );
}

/// No production A/B has been run, and the document must say so rather than
/// implying a measured verdict it does not have.
#[test]
fn the_document_states_that_the_production_ab_is_unrun() {
    let doc = doc();
    assert!(
        doc.contains("### Status: not yet run"),
        "docs/strategy-allocation.md dropped its status section"
    );
    assert!(
        doc.contains("**No production A/B has been run for this feature.**"),
        "docs/strategy-allocation.md no longer states that the A/B is unrun"
    );
    assert!(
        doc.contains("scoreImprovementPerWallHour"),
        "docs/strategy-allocation.md no longer names the gate metric the A/B compares"
    );
}

/// A report with no journal behind it — every field at its serialised default.
fn empty_report() -> JournalReport {
    let journal = tempfile::NamedTempFile::new().expect("temp journal");
    neat_ai_lamarck::report_from_journal(journal.path()).expect("an empty journal reports")
}
