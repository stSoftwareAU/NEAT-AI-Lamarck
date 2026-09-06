//! `docs/strategy-priors.md` ↔ code contract (Issue #221).
//!
//! The document describes evidence a run inherits rather than earns, so a
//! reader has to be able to trust every knob, field and discount it quotes.
//! These tests guard the ways that decays: the tooling it points at going away,
//! the report fields it tabulates being renamed, the confidence factors it
//! states drifting from the code, and — the honesty gate — its "not yet run"
//! status outliving the absence of a production A/B.

mod common;

use common::{read, repo_path};
use neat_ai_lamarck::report::JournalReport;
use neat_ai_lamarck::strategy_priors::{
    CORPUS_MISMATCH_CONFIDENCE, DEFAULT_STRATEGY_PRIORS_HALF_LIFE_HOURS, MAX_PRIOR_TRIALS,
    STRATEGY_PRIORS_FORMAT_VERSION, STRATEGY_PRIORS_MAX_AGE_HOURS, TOPOLOGY_DRIFT_CONFIDENCE,
    TOPOLOGY_MATCH_CONFIDENCE,
};

fn doc() -> String {
    read("docs/strategy-priors.md")
}

/// The document tells the reader to run these, so they have to exist.
#[test]
fn the_document_names_tooling_that_exists() {
    let doc = doc();
    for tool in [
        "scripts/run-strategy-priors-ab.sh",
        "scripts/summarise-strategy-priors.sh",
        "lamarck/src/strategy_priors.rs",
        "lamarck/tests/strategy_priors.rs",
    ] {
        assert!(doc.contains(tool), "docs/strategy-priors.md drops {tool}");
        assert!(
            repo_path(tool).exists(),
            "docs/strategy-priors.md points at a missing {tool}"
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
            priors: Some(neat_ai_lamarck::report::StrategyPriorsReport::default()),
            ..Default::default()
        },
        ..empty_report()
    })
    .expect("report serialises");
    let bucket = report
        .get("strategyAllocation")
        .expect("report carries the strategyAllocation section");

    assert!(
        bucket.get("priorsMode").is_some(),
        "`report` no longer emits `strategyAllocation.priorsMode`"
    );
    assert!(
        doc.contains("priorsMode"),
        "docs/strategy-priors.md drops the `priorsMode` field"
    );

    let priors = bucket
        .get("priors")
        .expect("the bucket carries the priors sub-object");
    for field in [
        "formatVersion",
        "ageHours",
        "confidence",
        "ageConfidence",
        "corpusConfidence",
        "sourceConfidence",
        "seededTrials",
        "seededArms",
    ] {
        assert!(
            priors.get(field).is_some(),
            "`report` no longer emits `strategyAllocation.priors.{field}`"
        );
        assert!(
            doc.contains(field),
            "docs/strategy-priors.md drops the `{field}` field"
        );
    }

    let row = bucket
        .get("strategies")
        .and_then(|rows| rows.get(0))
        .expect("the bucket carries per-strategy rows");
    for field in ["priorTrials", "priorShare"] {
        assert!(
            row.get(field).is_some(),
            "`report` no longer emits `strategyAllocation.strategies[].{field}`"
        );
        assert!(
            doc.contains(field),
            "docs/strategy-priors.md drops the `{field}` column"
        );
    }
}

/// Every discount and bound the document states is the one the code applies.
#[test]
fn the_documented_discounts_are_the_shipped_discounts() {
    let doc = doc();
    for (value, what) in [
        (DEFAULT_STRATEGY_PRIORS_HALF_LIFE_HOURS, "default half-life"),
        (STRATEGY_PRIORS_MAX_AGE_HOURS, "maximum prior age"),
        (CORPUS_MISMATCH_CONFIDENCE, "corpus-mismatch confidence"),
        (TOPOLOGY_MATCH_CONFIDENCE, "same-topology confidence"),
        (TOPOLOGY_DRIFT_CONFIDENCE, "drifted-topology confidence"),
        (MAX_PRIOR_TRIALS, "per-arm trial cap"),
    ] {
        let rendered = format!("{value}");
        assert!(
            doc.contains(&rendered),
            "docs/strategy-priors.md no longer states the {what} ({rendered})"
        );
    }
    assert!(
        doc.contains(STRATEGY_PRIORS_FORMAT_VERSION),
        "docs/strategy-priors.md no longer quotes the shipped format version"
    );
}

/// The whole safety argument is that nothing about an untouched run changed:
/// the document must keep saying so, and the config must keep it true.
#[test]
fn the_document_states_that_a_default_run_reads_and_writes_nothing() {
    let doc = doc();
    assert!(
        doc.contains("`--strategy-priors off` stays the default"),
        "docs/strategy-priors.md no longer states that the default is unchanged"
    );
    assert!(
        !neat_ai_lamarck::LamarckConfig::default()
            .strategy_priors
            .is_enabled(),
        "the shipped default now reads and writes priors the document promises it does not"
    );
}

/// The guardrail #221 turns on: the format cannot replay a historical
/// candidate, and the document has to keep claiming exactly that.
#[test]
fn the_document_states_that_no_historical_candidate_can_be_replayed() {
    let doc = doc();
    assert!(
        doc.contains("no historical candidate can be replayed even in principle"),
        "docs/strategy-priors.md dropped the no-replay guarantee"
    );
}

/// No production A/B has been run, and the document must say so rather than
/// implying a measured verdict it does not have.
#[test]
fn the_document_states_that_the_production_ab_is_unrun() {
    let doc = doc();
    assert!(
        doc.contains("### Status: not yet run"),
        "docs/strategy-priors.md dropped its status section"
    );
    assert!(
        doc.contains("**No production A/B has been run for this feature.**"),
        "docs/strategy-priors.md no longer states that the A/B is unrun"
    );
    assert!(
        doc.contains("scoreImprovementPerWallHour"),
        "docs/strategy-priors.md no longer names the gate metric the A/B compares"
    );
}

/// A report with no journal behind it — every field at its serialised default.
fn empty_report() -> JournalReport {
    let journal = tempfile::NamedTempFile::new().expect("temp journal");
    neat_ai_lamarck::report_from_journal(journal.path()).expect("an empty journal reports")
}
