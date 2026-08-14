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
    read("docs/followup-economics.md")
}

/// Absolute size of the issue #130 batch-composition artefact, as measured in
/// [`docs/scorer-batch-composition.md`]. Guarded against drift by
/// [`the_scorer_caveat_quotes_the_measured_artefact`].
const ARTEFACT_ABSOLUTE: f64 = 6.7e-8;

/// The accept bar every full-corpus Δ in the campaign is read against.
const MIN_IMPROVEMENT: f64 = 1e-6;

/// The `## Environment` caveat that dates the campaign's deltas to the
/// pre-#130 scorer, as a single paragraph.
fn scorer_caveat(doc: &str) -> &str {
    let start = doc.find("**Scorer caveat (#130)").expect(
        "docs/followup-economics.md must date its full-corpus deltas to the pre-#130 scorer",
    );
    let rest = &doc[start..];
    let end = rest.find("\n\n").unwrap_or(rest.len());
    &rest[..end]
}

/// Backticked figures written in exponent form, e.g. `` `6.7e-8` ``.
fn backticked_exponents(text: &str) -> Vec<&str> {
    text.split('`')
        .skip(1)
        .step_by(2)
        .filter(|token| {
            token.contains("e-")
                && token
                    .chars()
                    .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | '-' | '+'))
        })
        .collect()
}

/// The `Best full-corpus Δ` the `## Verdict` table records for `strategy`.
fn best_full_corpus_delta(doc: &str, strategy: &str) -> f64 {
    let verdict = &doc[doc.find("## Verdict").expect("a `## Verdict` section")..];
    let cells = |row: &str| -> Vec<String> {
        row.trim()
            .trim_matches('|')
            .split('|')
            .map(|cell| {
                cell.trim()
                    .trim_matches('*')
                    .trim()
                    .trim_matches('`')
                    .into()
            })
            .collect()
    };
    let header = verdict
        .lines()
        .find(|line| line.contains("Best full-corpus"))
        .expect("the verdict table needs a `Best full-corpus Δ` column");
    let column = cells(header)
        .iter()
        .position(|cell| cell.starts_with("Best full-corpus"))
        .expect("locate the delta column");
    let row = verdict
        .lines()
        .find(|line| line.trim_start().starts_with(&format!("| `{strategy}`")))
        .unwrap_or_else(|| panic!("the verdict table has no `{strategy}` row"));
    let cell = cells(row)
        .get(column)
        .unwrap_or_else(|| panic!("the `{strategy}` row is missing its delta cell"))
        .clone();
    cell.trim_start_matches('+')
        .parse()
        .unwrap_or_else(|e| panic!("`{cell}` is not a number ({e})"))
}

/// Every full-corpus Δ in this document was measured on a scorer whose score
/// depended on the batch a creature was scored in (#130). A reader who reaches
/// a delta table without that provenance will take numbers within the artefact
/// as settled, so the caveat has to survive editing and has to come first.
#[test]
fn the_full_corpus_deltas_carry_their_scorer_provenance() {
    let doc = doc();
    let caveat_at = doc.find("**Scorer caveat (#130)").expect(
        "docs/followup-economics.md must date its full-corpus deltas to the pre-#130 scorer",
    );
    let first_table = doc
        .find("Best full-corpus")
        .expect("the document reports full-corpus deltas");
    assert!(
        caveat_at < first_table,
        "the #130 scorer caveat sits after the first full-corpus Δ table — a reader meets the \
         numbers before the reason not to trust their last digits"
    );

    let caveat = scorer_caveat(&doc);
    for reference in ["scorer-batch-composition.md", "/issues/143"] {
        assert!(
            caveat.contains(reference),
            "the #130 caveat drops {reference}, so the reader cannot reach the measured artefact \
             or the issue tracking the re-measurement"
        );
    }

    let verdict = &doc[doc.find("## Verdict").expect("a `## Verdict` section")..];
    assert!(
        verdict.contains("#130") && verdict.contains("#143"),
        "the `## Verdict` table is the decision surface — it must point at the #130 caveat and \
         the #143 re-measurement, not leave `0 accepts` reading as settled"
    );
}

/// The two documents tell one story: the magnitude the caveat prices the
/// campaign against must be a figure the artefact document actually measured.
#[test]
fn the_scorer_caveat_quotes_the_measured_artefact() {
    let doc = doc();
    let caveat = scorer_caveat(&doc);
    let artefact_doc = read("docs/scorer-batch-composition.md");
    let quoted = backticked_exponents(caveat);
    assert!(
        !quoted.is_empty(),
        "the #130 caveat quotes no figure, so it cannot say how big the artefact is"
    );
    assert!(
        quoted.iter().any(|figure| artefact_doc.contains(figure)),
        "the #130 caveat quotes {quoted:?}, none of which docs/scorer-batch-composition.md \
         measures — the two documents have drifted apart"
    );
    assert!(
        artefact_doc.contains(&format!("{ARTEFACT_ABSOLUTE:e}")),
        "ARTEFACT_ABSOLUTE no longer matches docs/scorer-batch-composition.md"
    );
}

/// The caveat's claim is arithmetic, not rhetoric: the artefact is bigger than
/// the margin by which the campaign's best candidate missed the accept bar, so
/// `0 accepts` is not a safe verdict. If a re-measurement moves that margin
/// clear of the artefact, this fails and the caveat must be revisited — which
/// is exactly the check issue #143 owes.
#[test]
fn the_scorer_caveat_holds_against_the_campaign_margin() {
    let doc = doc();
    let best = best_full_corpus_delta(&doc, "stats_weight");
    let margin = MIN_IMPROVEMENT - best;
    assert!(
        margin > 0.0,
        "the verdict table records an accept ({best:e} ≥ {MIN_IMPROVEMENT:e}) while the campaign \
         reports 0 accepts"
    );
    assert!(
        ARTEFACT_ABSOLUTE > margin,
        "the #130 artefact ({ARTEFACT_ABSOLUTE:e}) no longer exceeds the {margin:e} by which the \
         best full-corpus Δ missed the {MIN_IMPROVEMENT:e} bar — the caveat's claim that the \
         artefact could have decided the verdict is now false and must be rewritten"
    );
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
