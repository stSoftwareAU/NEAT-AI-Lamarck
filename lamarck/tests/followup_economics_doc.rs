//! `docs/followup-economics.md` ↔ code contract (Issue #75).
//!
//! The follow-up campaign's whole point is a verdict on each strategy, so the
//! document has to keep naming strategies the binary actually produces. A
//! renamed or dropped variant must fail here rather than leave a report that
//! quietly discusses a strategy no run can generate.

use neat_ai_lamarck::candidates::CandidateStrategy;
use std::path::{Path, PathBuf};

/// Every strategy label as the journal (and therefore the report) writes it.
const ALL_STRATEGIES: &[CandidateStrategy] = &[
    CandidateStrategy::Backprop,
    CandidateStrategy::MeanErrorBias,
    CandidateStrategy::StatsWeight,
    CandidateStrategy::StatsBias,
    CandidateStrategy::StatsSkewBias,
    CandidateStrategy::StructuralAdd,
    CandidateStrategy::StructuralAddNeuron,
    CandidateStrategy::StructuralWeaken,
    CandidateStrategy::Random,
];

fn label(strategy: CandidateStrategy) -> String {
    serde_json::to_value(strategy)
        .expect("strategy serialises")
        .as_str()
        .expect("strategy serialises to a string")
        .to_string()
}

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(relative)
}

fn read(relative: &str) -> String {
    let path = repo_path(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn doc() -> String {
    read("docs/followup-economics.md")
}

#[test]
fn the_document_covers_all_four_recommended_arms() {
    let doc = doc();
    for arm in [
        "Output-focus slice",
        "Backprop step A/B",
        "Batch-size A/B",
        "Multi-seed repeat",
    ] {
        assert!(
            doc.contains(arm),
            "docs/followup-economics.md has no `{arm}` arm — #75 asked for all four"
        );
    }
}

#[test]
fn every_strategy_gets_an_explicit_keep_or_disable_verdict() {
    let doc = doc();
    let start = doc
        .find("## Verdict")
        .expect("docs/followup-economics.md needs a `## Verdict` section stating what to disable");
    let verdict = &doc[start..];
    for strategy in ALL_STRATEGIES {
        let name = label(*strategy);
        assert!(
            verdict.contains(&format!("`{name}`")),
            "the verdict section does not rule on `{name}`"
        );
    }
    assert!(
        verdict.contains("disable") || verdict.contains("disabled"),
        "the verdict section must say plainly whether anything is disabled"
    );
}

#[test]
fn the_document_names_only_real_strategies() {
    let doc = doc();
    let known: Vec<String> = ALL_STRATEGIES.iter().map(|s| label(*s)).collect();
    // Table cells quote strategies as `snake_case` in backticks; anything that
    // looks like one but is not a real variant is stale documentation.
    const FAMILIES: &[&str] = &["backprop", "mean", "stats", "structural", "random"];
    for token in doc.split('`') {
        let snake = !token.is_empty()
            && token.contains('_')
            && token.chars().all(|c| c.is_ascii_lowercase() || c == '_');
        if !snake {
            continue;
        }
        let family = token.split('_').next().unwrap_or_default();
        if FAMILIES.contains(&family) {
            assert!(
                known.iter().any(|k| k == token),
                "`{token}` is not a CandidateStrategy the binary can generate"
            );
        }
    }
}

#[test]
fn the_readme_points_at_the_follow_up_results() {
    let readme = read("README.md");
    assert!(
        readme.contains("docs/followup-economics.md"),
        "README.md must link the follow-up economics results"
    );
}

#[test]
fn the_runner_script_drives_every_arm() {
    let script = read("scripts/run-followup-economics.sh");
    for arm in ["output-focus", "backprop-step", "batch-size", "multi-seed"] {
        assert!(
            script.contains(arm),
            "scripts/run-followup-economics.sh cannot run the `{arm}` arm"
        );
    }
}
