//! README ↔ CLI contract (Issue #40).
//!
//! The README documents the tool as it is built today, so every long flag the
//! binary accepts must appear in it, and the README must not advertise a flag
//! the binary does not have. Drift in either direction fails here rather than
//! surviving to a reader.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

/// Long flags of *other* tools that the README legitimately quotes.
///
/// Lamarck spawns `rust_scorer` and documents the exact argv it uses, so these
/// names appear in the README without being Lamarck's own options.
const FOREIGN_FLAGS: &[&str] = &[
    // rust_scorer
    "--sample-rate",
    "--sample-phase",
    "--gpu",
    "--cost",
    // cargo / pip invocations quoted in the build instructions
    "--all",
    "--all-features",
    "--all-targets",
    "--check",
    "--locked",
    "--no-deps",
    "--test",
    "--user",
    "--workspace",
];

/// Text of the README section introduced by `heading`, up to the next `## `.
fn section<'a>(readme: &'a str, heading: &str) -> &'a str {
    let start = readme
        .find(heading)
        .unwrap_or_else(|| panic!("README.md has no `{heading}` section"))
        + heading.len();
    let rest = &readme[start..];
    match rest.find("\n## ") {
        Some(end) => &rest[..end],
        None => rest,
    }
}

/// Text of the README subsection introduced by `heading`, up to the next `### `.
fn subsection<'a>(readme: &'a str, heading: &str) -> &'a str {
    let start = readme
        .find(heading)
        .unwrap_or_else(|| panic!("README.md has no `{heading}` subsection"))
        + heading.len();
    let rest = &readme[start..];
    match rest.find("\n### ") {
        Some(end) => &rest[..end],
        None => rest,
    }
}

fn readme_text() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../README.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn help_text() -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_neat_ai_lamarck"))
        .arg("--help")
        .output()
        .expect("failed to run neat_ai_lamarck --help");
    assert!(output.status.success(), "--help exited non-zero");
    String::from_utf8(output.stdout).expect("--help emitted non-UTF-8")
}

/// Collect every `--long-flag` token in `text`.
fn long_flags(text: &str) -> BTreeSet<String> {
    let bytes: Vec<char> = text.chars().collect();
    let mut flags = BTreeSet::new();
    let mut i = 0;
    while i + 2 < bytes.len() {
        let starts_flag = bytes[i] == '-'
            && bytes[i + 1] == '-'
            && bytes[i + 2].is_ascii_alphabetic()
            && (i == 0 || !matches!(bytes[i - 1], '-' | '_') && !bytes[i - 1].is_alphanumeric());
        if !starts_flag {
            i += 1;
            continue;
        }
        let mut end = i + 2;
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == '-') {
            end += 1;
        }
        let flag: String = bytes[i..end].iter().collect();
        let flag = flag.trim_end_matches('-').to_string();
        flags.insert(flag);
        i = end;
    }
    flags
}

#[test]
fn readme_documents_every_cli_flag() {
    let readme = readme_text();
    let readme_flags = long_flags(&readme);
    let mut missing: Vec<String> = long_flags(&help_text())
        .into_iter()
        .filter(|flag| !matches!(flag.as_str(), "--help" | "--version"))
        .filter(|flag| !readme_flags.contains(flag))
        .collect();
    missing.sort();
    assert!(
        missing.is_empty(),
        "README.md does not document these CLI flags: {missing:?}"
    );
}

#[test]
fn readme_mentions_no_unknown_lamarck_flags() {
    let help_flags = long_flags(&help_text());
    let mut unknown: Vec<String> = long_flags(&readme_text())
        .into_iter()
        .filter(|flag| !help_flags.contains(flag))
        .filter(|flag| !FOREIGN_FLAGS.contains(&flag.as_str()))
        .collect();
    unknown.sort();
    assert!(
        unknown.is_empty(),
        "README.md documents flags the binary does not accept: {unknown:?}"
    );
}

#[test]
fn readme_documents_the_report_subcommand() {
    let readme = readme_text();
    assert!(
        readme.contains("neat_ai_lamarck report"),
        "README.md does not document the `report` subcommand"
    );
}

/// The core-principle diagram must show the two-phase screen/promote scoring the
/// optimiser actually runs, not the single scoring step it started with (#39).
#[test]
fn core_principle_diagram_shows_two_phase_screening() {
    let readme = readme_text();
    let core = section(&readme, "\n## Core principle").to_lowercase();
    for phrase in ["screen", "subsample", "full corpus"] {
        assert!(
            core.contains(phrase),
            "core-principle diagram omits {phrase:?} — the screening phase is undocumented"
        );
    }
    let screen = core.find("screen").expect("screen phrase located above");
    let full = core.find("full corpus").expect("full corpus located above");
    assert!(
        screen < full,
        "core-principle diagram must screen on a subsample before full-corpus scoring"
    );
}

/// Screening exists to skip full-corpus scoring for candidates the sample rejects,
/// so the diagram has to show that branch rather than scoring everything.
#[test]
fn core_principle_diagram_shows_screened_out_candidates_are_dropped() {
    let readme = readme_text();
    let core = section(&readme, "\n## Core principle").to_lowercase();
    assert!(
        core.contains("dropped") || core.contains("discard"),
        "core-principle diagram does not show candidates failing the screen being dropped"
    );
}

/// The gap this issue records is closed, so it must no longer be listed as outstanding.
#[test]
fn outstanding_work_no_longer_lists_the_core_principle_diagram() {
    let readme = readme_text();
    let outstanding = section(&readme, "\n## Outstanding work");
    assert!(
        !outstanding.contains("/issues/39"),
        "Outstanding work still lists issue #39 after the diagram was updated"
    );
}

/// Phase 1 writes distribution shape, so the README must say so (#73).
#[test]
fn phase_one_documents_the_distribution_moments() {
    let readme = readme_text();
    let phase_one = subsection(&readme, "\n### Phase 1").to_lowercase();
    for phrase in ["skewness", "kurtosis"] {
        assert!(
            phase_one.contains(phrase),
            "Phase 1 does not document {phrase:?}, which the cache now records"
        );
    }
}

/// Covariance is derived from the stored correlation, not written to disk (#73).
#[test]
fn phase_one_documents_covariance_as_derived() {
    let readme = readme_text();
    let phase_one = subsection(&readme, "\n### Phase 1");
    assert!(
        phase_one.contains("input_target_covariance") && phase_one.contains("input_covariance"),
        "Phase 1 does not name the derived covariance accessors"
    );
    let lowered = phase_one.to_lowercase();
    assert!(
        lowered.contains("derived"),
        "Phase 1 does not say covariances are derived rather than stored"
    );
}

/// The moments have a consumer, so its journal tag must be documented (#73).
#[test]
fn candidate_table_documents_the_skew_bias_strategy() {
    let readme = readme_text();
    let phase_four = subsection(&readme, "\n### Phase 4");
    assert!(
        phase_four.contains("stats_skew_bias"),
        "Phase 4 candidate table omits the `stats_skew_bias` strategy"
    );
}

/// The gap this issue records is closed, so it must no longer be listed as outstanding.
#[test]
fn outstanding_work_no_longer_lists_the_observation_moments() {
    let readme = readme_text();
    let outstanding = section(&readme, "\n## Outstanding work");
    assert!(
        !outstanding.contains("/issues/73"),
        "Outstanding work still lists issue #73 after the moments were added"
    );
}

/// The journal records the focus scan, so its fields must be documented (#70).
#[test]
fn experiment_journal_documents_the_focus_stats_record() {
    let readme = readme_text();
    let journal = section(&readme, "\n## Experiment journal");
    for field in [
        "focusStats",
        "squash",
        "incomingCount",
        "saturationFraction",
        "meanBlame",
    ] {
        assert!(
            journal.contains(field),
            "Experiment journal does not document {field:?}, which each record now carries"
        );
    }
}

/// `report` summarises the focus scan, so its aggregates must be documented (#70).
#[test]
fn report_documents_the_focus_stats_aggregates() {
    let readme = readme_text();
    let journal = section(&readme, "\n## Experiment journal");
    for field in [
        "meanSaturationFraction",
        "meanAbsBlame",
        "squashCounts",
        "accepted",
        "rejected",
    ] {
        assert!(
            journal.contains(field),
            "the `report` description omits the {field:?} focus aggregate"
        );
    }
}

/// The gap this issue records is closed, so it must no longer be listed as outstanding.
#[test]
fn outstanding_work_no_longer_lists_the_focus_journal_gap() {
    let readme = readme_text();
    let outstanding = section(&readme, "\n## Outstanding work");
    assert!(
        !outstanding.contains("/issues/70"),
        "Outstanding work still lists issue #70 after the focus scan was journalled"
    );
}

/// Combo and graft wins are attributable now, so the journal must document how (#74).
#[test]
fn experiment_journal_documents_combo_members_and_graft_replay() {
    let readme = readme_text();
    let journal = section(&readme, "\n## Experiment journal");
    for field in [
        "comboMemberIndices",
        "graftReplay",
        "graftsApplied",
        "replayError",
    ] {
        assert!(
            journal.contains(field),
            "Experiment journal does not document {field:?}, which the journal now carries"
        );
    }
}

/// `report` buckets combo and graft wins, so the attribution rule must be stated (#74).
#[test]
fn report_documents_combo_and_graft_win_attribution() {
    let readme = readme_text();
    let journal = section(&readme, "\n## Experiment journal");
    for field in [
        "comboWins",
        "comboAcceptancesUnattributed",
        "cumulativeImprovement",
        "every** member strategy",
    ] {
        assert!(
            journal.contains(field),
            "the `report` description omits {field:?} from the win attribution rule"
        );
    }
}

/// The gap this issue records is closed, so it must no longer be listed as outstanding.
#[test]
fn outstanding_work_no_longer_lists_the_win_attribution_gap() {
    let readme = readme_text();
    let outstanding = section(&readme, "\n## Outstanding work");
    assert!(
        !outstanding.contains("/issues/74"),
        "Outstanding work still lists issue #74 after combo and graft wins were attributed"
    );
}

/// All four stopping rules exist now, so Phase 6 must list them (#72).
#[test]
fn phase_six_documents_every_stopping_rule() {
    let readme = readme_text();
    let phase_six = subsection(&readme, "\n### Phase 6").to_lowercase();
    for phrase in [
        "--timeout-seconds",
        "--max-experiments",
        "sigint",
        "sigterm",
        "consecutive",
    ] {
        assert!(
            phase_six.contains(phrase),
            "Phase 6 omits the {phrase:?} stopping rule"
        );
    }
}

/// Cancellation is only useful if the run still finishes cleanly (#72).
#[test]
fn phase_six_documents_the_graceful_cancellation_contract() {
    let readme = readme_text();
    let phase_six = subsection(&readme, "\n### Phase 6").to_lowercase();
    for phrase in ["best.json", "run summary", "130"] {
        assert!(
            phase_six.contains(phrase),
            "Phase 6 does not document {phrase:?} in the cancellation contract"
        );
    }
}

/// The gap this issue records is closed, so it must no longer be listed as outstanding.
#[test]
fn outstanding_work_no_longer_lists_the_stopping_rules_gap() {
    let readme = readme_text();
    let outstanding = section(&readme, "\n## Outstanding work");
    assert!(
        !outstanding.contains("/issues/72"),
        "Outstanding work still lists issue #72 after the stopping rules were added"
    );
}

#[test]
fn section_returns_only_the_requested_section() {
    let readme = "# T\n\n## A\n\nalpha\n\n## B\n\nbeta\n";
    assert!(section(readme, "\n## A").contains("alpha"));
    assert!(!section(readme, "\n## A").contains("beta"));
    assert!(section(readme, "\n## B").contains("beta"));
}

#[test]
#[should_panic(expected = "no `\n## Missing` section")]
fn section_panics_on_a_missing_heading() {
    section("## A\n\nalpha\n", "\n## Missing");
}

#[test]
fn subsection_returns_only_the_requested_subsection() {
    let readme = "## Top\n\n### A\n\nalpha\n\n### B\n\nbeta\n";
    assert!(subsection(readme, "\n### A").contains("alpha"));
    assert!(!subsection(readme, "\n### A").contains("beta"));
    assert!(subsection(readme, "\n### B").contains("beta"));
}

#[test]
#[should_panic(expected = "no `\n### Missing` subsection")]
fn subsection_panics_on_a_missing_heading() {
    subsection("### A\n\nalpha\n", "\n### Missing");
}

#[test]
fn long_flags_extracts_flags_and_ignores_prose_dashes() {
    let flags = long_flags("run with `--focus-policy weighted` — see --seed, not em--dash or -x");
    assert!(flags.contains("--focus-policy"));
    assert!(flags.contains("--seed"));
    assert!(!flags.contains("--dash"));
    assert!(!flags.contains("-x"));
}

#[test]
fn long_flags_handles_empty_input() {
    assert!(long_flags("").is_empty());
}
