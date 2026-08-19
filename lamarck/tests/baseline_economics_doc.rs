//! `docs/baseline-economics.md` ↔ repository contract (Issue #134).
//!
//! The document is a dated measurement record, but it still tells a reader how
//! to reproduce the run. These tests guard the ways that decays: a pointer at a
//! script that never made it into the repository (Issue #134), the reporting
//! tooling it names going away, and the one operational warning that keeps GRQ
//! `node.sh` from deleting the training data mid-run being lost.

use std::path::{Path, PathBuf};

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn doc() -> String {
    let path = repo_path("docs/baseline-economics.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Split prose into path-shaped tokens: the characters a repo-relative path is
/// spelt with, everything else is a separator.
fn tokens(text: &str) -> Vec<&str> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-')))
        .filter(|t| !t.is_empty())
        .collect()
}

/// A `.sh` reference claims to live in this repository when it is spelt as a
/// path (`scripts/foo.sh`) or as a repo-root dotfile (`.foo.sh`). A bare tool
/// name such as GRQ's `node.sh` names someone else's script, not ours.
fn claims_to_be_in_this_repo(token: &str) -> bool {
    token.ends_with(".sh") && (token.contains('/') || token.starts_with('.'))
}

/// Issue #134: the document pointed at `.run-baseline-economics.sh`, which
/// exists nowhere in the repository. Any script it spells as one of ours has to
/// be one a reader can actually open.
#[test]
fn every_repo_script_the_document_names_exists() {
    let doc = doc();
    for token in tokens(&doc) {
        if !claims_to_be_in_this_repo(token) {
            continue;
        }
        let relative = token.trim_start_matches("./");
        assert!(
            repo_path(relative).exists(),
            "docs/baseline-economics.md points at {token}, which is not in the repository"
        );
    }
}

/// The document tells the reader to run this, so it has to exist.
#[test]
fn the_document_names_reporting_tooling_that_exists() {
    let doc = doc();
    let tool = "scripts/report-experiments.sh";
    assert!(
        doc.contains(tool),
        "docs/baseline-economics.md drops {tool}"
    );
    assert!(
        repo_path(tool).exists(),
        "docs/baseline-economics.md points at a missing {tool}"
    );
}

/// The reproduction command has to carry the train-data warning itself — the
/// reader cannot chase it into a helper script that was never committed.
#[test]
fn the_command_block_states_the_private_copy_advice_inline() {
    let doc = doc();
    let block = doc
        .split("```bash")
        .find(|b| b.contains("--timeout-seconds 2700"))
        .expect("the baseline command fence is present");
    let advice: String = block
        .lines()
        .take_while(|line| !line.starts_with("neat_ai_lamarck"))
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    for phrase in ["private train-data copy", "node.sh"] {
        assert!(
            advice.contains(phrase),
            "the baseline command fence no longer states {phrase:?} — the advice \
             must stand on its own, not point at an uncommitted helper"
        );
    }
}

/// The same operational advice is the closing recommendation; losing it would
/// leave the run's one destructive failure mode undocumented.
#[test]
fn the_document_keeps_the_operational_recommendation() {
    let doc = doc().to_lowercase();
    for phrase in ["private train-data copy", ".in-use.lock", "node.sh"] {
        assert!(
            doc.contains(phrase),
            "docs/baseline-economics.md no longer mentions {phrase:?}"
        );
    }
}
