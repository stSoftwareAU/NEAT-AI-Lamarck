//! `docs/scorer-fixed-cost.md` ↔ evidence contract (issue #123).
//!
//! The fix lives in two dependency repositories, so this document is the only
//! place in this repo where the before/after can be checked. These tests guard
//! the ways that decays: the evidence logs it quotes going away, the numbers on
//! the page drifting from the logs they came from, the load conditions being
//! quietly dropped, and the honest caveats — a contended box, a per-call rather
//! than whole-run measurement — being softened.

use std::path::{Path, PathBuf};

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn read(relative: &str) -> String {
    let path = repo_path(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn doc() -> String {
    read("docs/scorer-fixed-cost.md")
}

/// Every artefact the document points at has to exist — a claim whose evidence
/// is missing is worse than no claim.
#[test]
fn the_document_names_evidence_that_exists() {
    let doc = doc();
    for artefact in [
        "docs/evidence/scorer-fixed-cost/before/rate-0_05.log",
        "docs/evidence/scorer-fixed-cost/after/rate-0_05.log",
        "docs/evidence/scorer-fixed-cost/interleaved-rate-0_05.log",
        "scripts/measure-scorer-call-cost.sh",
        "lamarck/examples/scorer_call_cost_bench.rs",
        "lamarck/src/scorer_cost.rs",
    ] {
        let quoted = artefact.strip_prefix("docs/").map_or(artefact, |rest| {
            rest.strip_prefix("evidence/").map_or(rest, |_| rest)
        });
        assert!(
            doc.contains(quoted) || doc.contains(artefact),
            "docs/scorer-fixed-cost.md drops {artefact}"
        );
        assert!(
            repo_path(artefact).exists(),
            "docs/scorer-fixed-cost.md points at a missing {artefact}"
        );
    }
}

/// The headline before/after numbers are the ones the committed logs recorded.
/// A number edited on the page but not re-measured fails here.
#[test]
fn the_headline_numbers_come_from_the_committed_logs() {
    let doc = doc();
    for (log, fixed_ms) in [
        (
            "docs/evidence/scorer-fixed-cost/before/rate-0_05.log",
            "10693",
        ),
        (
            "docs/evidence/scorer-fixed-cost/after/rate-0_05.log",
            "3423",
        ),
    ] {
        let text = read(log);
        assert!(
            text.contains(&format!("fixed ms/call  : {fixed_ms}")),
            "{log} no longer reports a fixed cost of {fixed_ms} ms"
        );
        // The document writes it with a thin space: `10 693 ms`.
        let spaced = format!(
            "{} {}",
            &fixed_ms[..fixed_ms.len() - 3],
            &fixed_ms[fixed_ms.len() - 3..]
        );
        assert!(
            doc.contains(&spaced),
            "docs/scorer-fixed-cost.md no longer quotes the measured {spaced} ms"
        );
    }

    // Every interleaved pair on the page is a row in the log.
    let interleaved = read("docs/evidence/scorer-fixed-cost/interleaved-rate-0_05.log");
    let mut pairs = 0;
    for line in interleaved.lines().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 6 || fields[2] != "screen" {
            continue;
        }
        let ms: u64 = fields[4].parse().expect("a millisecond field");
        let spaced = format!("{} {}", ms / 1000, ms % 1000);
        assert!(
            doc.contains(&spaced),
            "docs/scorer-fixed-cost.md drops the interleaved {ms} ms call"
        );
        pairs += 1;
    }
    assert_eq!(pairs, 8, "the interleaved log must hold four off/on pairs");
}

/// The measurement's own limits stay on the page: it was taken on a busy box,
/// and it is a per-call number, not a whole-run one.
#[test]
fn the_document_keeps_its_caveats() {
    let doc = doc();
    for caveat in [
        "not an idle box",
        "loadBefore",
        "loadAfter",
        "scorerCallCost",
        "NEAT_SCORER_SAMPLED_READ=off",
    ] {
        assert!(
            doc.contains(caveat),
            "docs/scorer-fixed-cost.md no longer records `{caveat}`"
        );
    }
    // The load figures the runs actually recorded, not a rounded retelling.
    for log in [
        "docs/evidence/scorer-fixed-cost/before/rate-0_05.log",
        "docs/evidence/scorer-fixed-cost/after/rate-0_05.log",
    ] {
        let text = read(log);
        for prefix in ["# loadBefore: ", "# loadAfter: "] {
            let value = text
                .lines()
                .find_map(|l| l.strip_prefix(prefix))
                .unwrap_or_else(|| panic!("{log} no longer records `{prefix}`"))
                .trim()
                .to_string();
            assert!(
                doc.contains(&value),
                "docs/scorer-fixed-cost.md drops the measured {prefix}{value}"
            );
        }
    }
}

/// The fix lives in two dependency repositories; the document has to say which
/// branches carry it, or the reader cannot reproduce anything.
#[test]
fn the_document_names_the_cross_repo_branches() {
    let doc = doc();
    for reference in [
        "NEAT-AI-core",
        "NEAT-AI-scorer",
        "issue-scorer-sampled-read",
        "issue-lamarck-123-sampled-read",
        "for_each_sampled_read_chunk",
        "sampled_read_parity",
    ] {
        assert!(
            doc.contains(reference),
            "docs/scorer-fixed-cost.md no longer names {reference}"
        );
    }
}

/// The go/no-go it answers stays a decision: shape (1) was built, shape (2)
/// (a persistent session) was not.
#[test]
fn the_document_records_which_shape_was_built() {
    let doc = doc().to_lowercase();
    assert!(
        doc.contains("persistent scoring session"),
        "docs/scorer-fixed-cost.md no longer says what it did *not* build"
    );
    assert!(
        doc.contains("was not built"),
        "docs/scorer-fixed-cost.md softened the decision not to build a session"
    );
}
