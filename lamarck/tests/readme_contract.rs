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
    "--example",
    "--locked",
    "--release",
    "--no-deps",
    "--test",
    "--user",
    "--workspace",
    // helper scripts under scripts/ (spell-check.sh, typescript-check.sh,
    // check-workflow-npm-pins.sh)
    "--root",
    "--dir",
    "--verbose",
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

/// Body of the first ```` ```<lang> ```` fenced block in `text`.
fn fenced_block<'a>(text: &'a str, lang: &str) -> Option<&'a str> {
    let fence = format!("```{lang}\n");
    let start = text.find(&fence)? + fence.len();
    let rest = &text[start..];
    let end = rest.find("\n```")?;
    Some(&rest[..end])
}

fn readme_text() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../README.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// The ```text tree under `## Repository layout`.
fn repository_layout_tree() -> String {
    let readme = readme_text();
    let layout = section(&readme, "\n## Repository layout");
    fenced_block(layout, "text")
        .expect("README.md `## Repository layout` has no ```text tree")
        .to_string()
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

/// The ordered list that follows `marker` — its items and their indented
/// continuation lines, up to the first line that is neither.
fn ordered_list_after(text: &str, marker: &str) -> String {
    let start = text
        .find(marker)
        .unwrap_or_else(|| panic!("no `{marker}` in the text"))
        + marker.len();
    let mut lines = Vec::new();
    let mut started = false;
    for line in text[start..].lines() {
        let numbered = matches!(
            line.split_once(". "),
            Some((number, _)) if !number.is_empty() && number.chars().all(|c| c.is_ascii_digit())
        );
        let continuation = line.starts_with("   ") && !line.trim().is_empty();
        if !started {
            if numbered {
                started = true;
            } else if line.trim().is_empty() {
                continue;
            } else {
                panic!("no ordered list follows `{marker}`");
            }
        } else if !(numbered || continuation) {
            break;
        }
        lines.push(line);
    }
    assert!(started, "no ordered list follows `{marker}`");
    lines.join("\n")
}

/// The three-step scoring list of `### Phase 5`.
fn phase_five_scoring_steps() -> String {
    let readme = readme_text();
    let phase_five = subsection(&readme, "\n### Phase 5 — authoritative candidate scoring");
    ordered_list_after(phase_five, "Scoring runs in three steps:")
}

/// Time figures — a number, an optional space, then `s` or `ms` — in `text`.
///
/// `0.05` carries no unit and the `e` of `5e-2` is not one, so neither is a
/// figure; `≈11s`, `452 ms` and `0.7–1s` all are.
fn time_figures(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut figures = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let starts_number = chars[i].is_ascii_digit()
            && (i == 0 || !(chars[i - 1].is_alphanumeric() || chars[i - 1] == '.'));
        if !starts_number {
            i += 1;
            continue;
        }
        let mut end = i;
        while end < chars.len()
            && (chars[end].is_ascii_digit() || matches!(chars[end], '.' | '-' | '–'))
        {
            end += 1;
        }
        // A trailing separator belongs to the prose, not to the number.
        while end > i && !chars[end - 1].is_ascii_digit() {
            end -= 1;
        }
        let unit_start = if chars.get(end) == Some(&' ') {
            end + 1
        } else {
            end
        };
        let mut unit_end = unit_start;
        while unit_end < chars.len() && chars[unit_end].is_ascii_alphabetic() {
            unit_end += 1;
        }
        let unit: String = chars[unit_start..unit_end].iter().collect();
        if unit == "s" || unit == "ms" {
            figures.push(chars[i..unit_end].iter().collect());
        }
        i = end.max(i + 1);
    }
    figures
}

/// The time figure `text` ends with, as seconds — `None` when it does not end
/// in one. A range (`0.7–1s`) yields one value per endpoint.
fn trailing_seconds(text: &str) -> Option<Vec<f64>> {
    let chars: Vec<char> = text.chars().collect();
    let mut unit_start = chars.len();
    while unit_start > 0 && chars[unit_start - 1].is_ascii_alphabetic() {
        unit_start -= 1;
    }
    let unit: String = chars[unit_start..].iter().collect();
    let multiplier = match unit.as_str() {
        "s" => 1.0,
        "ms" => 0.001,
        _ => return None,
    };
    let mut end = unit_start;
    if end > 0 && chars[end - 1] == ' ' {
        end -= 1;
    }
    if end == 0 || !chars[end - 1].is_ascii_digit() {
        return None;
    }
    let mut start = end;
    while start > 0
        && (chars[start - 1].is_ascii_digit() || matches!(chars[start - 1], '.' | '-' | '–'))
    {
        start -= 1;
    }
    while start < end && !chars[start].is_ascii_digit() {
        start += 1;
    }
    let number: String = chars[start..end].iter().collect();
    let seconds = number
        .split(['–', '-'])
        .filter_map(|part| part.trim().parse::<f64>().ok())
        .map(|value| value * multiplier)
        .collect::<Vec<_>>();
    (!seconds.is_empty()).then_some(seconds)
}

/// Per-creature cost claims in `text` — a time figure immediately followed by
/// `/creature` or ` per creature` — paired with the seconds they assert.
fn per_creature_cost_claims(text: &str) -> Vec<(String, Vec<f64>)> {
    let mut claims = Vec::new();
    for (index, _) in text.match_indices("creature") {
        let before = &text[..index];
        let Some(figure) = before
            .strip_suffix('/')
            .or_else(|| before.strip_suffix(" per "))
            .or_else(|| before.strip_suffix(" per-"))
        else {
            continue;
        };
        if let Some(seconds) = trailing_seconds(figure) {
            // Enough of the sentence to find the claim in the README by eye.
            let quoted: String = figure
                .chars()
                .rev()
                .take(24)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            claims.push((quoted, seconds));
        }
    }
    claims
}

/// The `marginalMsPerCreature` column of the Result table in
/// `docs/scorer-call-cost.md`, in seconds — the measured source of truth the
/// README must not contradict (Issue #172).
fn measured_marginal_seconds_per_creature() -> Vec<(String, f64)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/scorer-call-cost.md");
    let doc = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut column = None;
    let mut measured = Vec::new();
    for line in doc.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('|') {
            continue;
        }
        let cells: Vec<&str> = trimmed
            .trim_matches('|')
            .split('|')
            .map(|cell| cell.trim().trim_matches('*'))
            .collect();
        if let Some(index) = cells
            .iter()
            .position(|cell| cell.contains("marginalMsPerCreature"))
        {
            column = Some(index);
            continue;
        }
        let Some(index) = column else { continue };
        let (Some(phase), Some(cell)) = (cells.first(), cells.get(index)) else {
            continue;
        };
        if !matches!(*phase, "screen" | "promote") {
            continue;
        }
        let ms: String = cell.chars().filter(|c| c.is_ascii_digit()).collect();
        if let Ok(ms) = ms.parse::<f64>() {
            measured.push(((*phase).to_string(), ms / 1000.0));
        }
    }
    measured
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

/// Every `lamarck/src/*.rs` file and source subdirectory must appear in the
/// layout tree (Issue #133).
///
/// The tree named 21 of 26 files with no elision marker, so a reader would
/// conclude `baseline.rs`, `cancel.rs`, `chunks.rs`, `memo.rs` and
/// `promote_gate.rs` did not exist — three of which the README's own prose
/// already cites. A new module that is absent from the tree fails here.
#[test]
fn repository_layout_lists_every_lamarck_src_rs() {
    let tree = repository_layout_tree();
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut missing = Vec::new();
    for entry in
        std::fs::read_dir(&src_dir).unwrap_or_else(|e| panic!("read {}: {e}", src_dir.display()))
    {
        let path = entry.expect("src dir entry").path();
        let is_rs = path.extension().and_then(|ext| ext.to_str()) == Some("rs");
        if !is_rs && !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("utf-8 rs file name");
        if !tree.contains(name) {
            missing.push(name.to_string());
        }
    }
    missing.sort();
    assert!(
        missing.is_empty(),
        "README.md repository-layout tree omits lamarck/src entries: {missing:?}"
    );
}

/// `docs/` in the layout tree must either list every top-level `docs/*.md` or
/// carry an explicit elision marker (`…` / `...`) so a partial list cannot
/// read as complete (Issue #133).
#[test]
fn repository_layout_elides_or_lists_top_level_docs() {
    let tree = repository_layout_tree();
    let docs_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs");
    let mut on_disk = Vec::new();
    for entry in
        std::fs::read_dir(&docs_dir).unwrap_or_else(|e| panic!("read {}: {e}", docs_dir.display()))
    {
        let path = entry.expect("docs dir entry").path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .expect("utf-8 md file name");
            on_disk.push(name.to_string());
        }
    }
    on_disk.sort();

    let mut in_docs = false;
    let mut listed = Vec::new();
    let mut elided = false;
    for line in tree.lines() {
        if !in_docs {
            if line.contains("docs/") {
                in_docs = true;
                elided = line.contains('…') || line.contains("...");
            }
            continue;
        }
        if line.contains("lamarck/src/") {
            break;
        }
        if let Some(name) = line.split_whitespace().find(|tok| tok.ends_with(".md")) {
            listed.push(name.to_string());
        }
        if line.contains('…') || line.contains("...") {
            elided = true;
        }
    }
    assert!(
        in_docs,
        "README.md repository-layout tree has no docs/ entry"
    );
    if listed.is_empty() {
        assert!(
            elided,
            "README.md repository-layout docs/ lists no files and has no elision marker (`…`)"
        );
        return;
    }
    let missing: Vec<&String> = on_disk
        .iter()
        .filter(|name| !listed.iter().any(|listed| listed == *name))
        .collect();
    assert!(
        missing.is_empty() || elided,
        "README.md repository-layout docs/ lists some markdown files but omits {missing:?} \
         and has no elision marker (`…`)"
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

/// The loop is a flow diagram, so it renders as Mermaid rather than ASCII art (#86).
#[test]
fn core_principle_diagram_is_a_mermaid_flowchart() {
    let readme = readme_text();
    let core = section(&readme, "\n## Core principle");
    let diagram =
        fenced_block(core, "mermaid").expect("Core principle has no ```mermaid fenced block");
    assert!(
        diagram.trim_start().starts_with("flowchart"),
        "the core-principle Mermaid block is not a flowchart"
    );
    assert!(
        fenced_block(core, "text").is_none(),
        "the ASCII core-principle diagram is still present alongside the Mermaid one"
    );
}

/// Converting the diagram must not lose a candidate source (#86).
#[test]
fn core_principle_diagram_keeps_every_candidate_source() {
    let readme = readme_text();
    let core = section(&readme, "\n## Core principle");
    let diagram = fenced_block(core, "mermaid")
        .expect("Core principle has no ```mermaid fenced block")
        .to_lowercase();
    for source in ["statistical", "backprop", "structural", "random"] {
        assert!(
            diagram.contains(source),
            "the core-principle diagram omits the {source:?} candidate source"
        );
    }
}

/// Both loop outcomes and the repeat edge must survive the conversion (#86).
#[test]
fn core_principle_diagram_shows_both_outcomes_and_repeats() {
    let readme = readme_text();
    let core = section(&readme, "\n## Core principle");
    let diagram = fenced_block(core, "mermaid")
        .expect("Core principle has no ```mermaid fenced block")
        .to_lowercase();
    for phrase in ["new incumbent", "keep incumbent", "repeat"] {
        assert!(
            diagram.contains(phrase),
            "the core-principle diagram omits {phrase:?}"
        );
    }
}

/// The diagram is colour-coded, and every colour it defines must be applied and
/// carry an explicit text colour so it stays legible in both GitHub themes (#86).
#[test]
fn core_principle_diagram_colours_and_applies_every_class() {
    let readme = readme_text();
    let core = section(&readme, "\n## Core principle");
    let diagram =
        fenced_block(core, "mermaid").expect("Core principle has no ```mermaid fenced block");

    let defined: Vec<(&str, &str)> = diagram
        .lines()
        .filter_map(|line| line.trim().strip_prefix("classDef "))
        .filter_map(|rest| rest.split_once(char::is_whitespace))
        .collect();
    assert!(
        defined.len() >= 3,
        "the core-principle diagram defines {} colour classes; group the sources, stages and outcomes",
        defined.len()
    );

    let assignments: Vec<&str> = diagram
        .lines()
        .filter_map(|line| line.trim().strip_prefix("class "))
        .collect();
    for (name, style) in defined {
        for property in ["fill:#", "stroke:#", "color:#"] {
            assert!(
                style.contains(property),
                "classDef {name} omits {property} — it will not read in both light and dark themes"
            );
        }
        assert!(
            assignments
                .iter()
                .any(|assignment| assignment.split_whitespace().next_back() == Some(name)),
            "classDef {name} is defined but never applied to any node"
        );
    }
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

/// The README preamble — everything before the first `## ` heading.
fn preamble(readme: &str) -> &str {
    match readme.find("\n## ") {
        Some(end) => &readme[..end],
        None => readme,
    }
}

/// The blank-line-delimited paragraph holding the first occurrence of `needle`.
fn paragraph_containing<'a>(text: &'a str, needle: &str) -> &'a str {
    let at = text
        .find(needle)
        .unwrap_or_else(|| panic!("text has no occurrence of `{needle}`"));
    let start = text[..at].rfind("\n\n").map_or(0, |i| i + 2);
    let end = text[at..].find("\n\n").map_or(text.len(), |i| at + i);
    &text[start..end]
}

/// NEAT is the algorithm this optimiser builds on, so the acronym must be
/// expanded — and linked — before any section uses it (Issue #135).
#[test]
fn readme_expands_neat_before_the_first_section() {
    let readme = readme_text();
    let preamble = preamble(&readme).to_lowercase();
    assert!(
        preamble.contains("neuroevolution of augmenting topologies"),
        "README.md never expands the NEAT acronym in its opening section"
    );
    assert!(
        preamble.contains("en.wikipedia.org/wiki/neuroevolution_of_augmenting_topologies"),
        "the NEAT expansion is not linked to its reference page"
    );
}

/// GRQ is a private system named nowhere else in this repository, so its first
/// use has to say what it is (Issue #135).
#[test]
fn readme_glosses_grq_where_it_is_first_used() {
    let readme = readme_text();
    let paragraph = paragraph_containing(&readme, "GRQ").to_lowercase();
    for phrase in ["private", "production system"] {
        assert!(
            paragraph.contains(phrase),
            "the paragraph first using GRQ omits {phrase:?} — a reader is left to guess what GRQ is"
        );
    }
}

/// `squash` is neat-core's name for a neuron's activation function, so its
/// first use has to gloss it (Issue #135).
#[test]
fn readme_glosses_squash_where_it_is_first_used() {
    let readme = readme_text();
    let paragraph = paragraph_containing(&readme, "squash").to_lowercase();
    for phrase in ["activation", "squashing"] {
        assert!(
            paragraph.contains(phrase),
            "the paragraph first using `squash` omits {phrase:?} — the term is never glossed"
        );
    }
}

/// The Phase 5 scoring steps must point at the measurement rather than restate
/// it (Issue #172).
///
/// The screen step carried "≈0.7–1s/creature on GRQ against ≈11s full", written
/// in #76 before the #112 fixed/marginal decomposition existed and never
/// reconciled against it — roughly double the measured marginal costs the
/// README itself cites `docs/scorer-call-cost.md` for further down the page. A
/// restated number drifts; a link cannot.
#[test]
fn phase_five_scoring_steps_restate_no_scorer_timings() {
    let figures = time_figures(&phase_five_scoring_steps());
    assert!(
        figures.is_empty(),
        "the Phase 5 scoring steps restate scorer timings {figures:?} — \
         link `docs/scorer-call-cost.md` instead of copying its numbers"
    );
}

/// Dropping the inline numbers only helps if the reader is sent somewhere.
#[test]
fn phase_five_scoring_steps_link_the_measured_scorer_call_cost() {
    let steps = phase_five_scoring_steps();
    assert!(
        steps.contains("docs/scorer-call-cost.md"),
        "the Phase 5 scoring steps no longer point at the measured screen/promote costs"
    );
}

/// Wherever the README does quote a per-creature scorer cost, it must be the
/// measured one (Issue #172).
///
/// `docs/scorer-call-cost.md` is the source of truth, so its Result table is
/// read here rather than hard-coded: re-measuring the doc updates the bar this
/// test holds the README to.
#[test]
fn every_per_creature_cost_the_readme_quotes_matches_the_measurement() {
    let measured = measured_marginal_seconds_per_creature();
    assert_eq!(
        measured.len(),
        2,
        "docs/scorer-call-cost.md no longer reports a screen and a promote marginal cost: {measured:?}"
    );
    for (claim, seconds) in per_creature_cost_claims(&readme_text()) {
        for value in seconds {
            let matched = measured
                .iter()
                .any(|(_, m)| (value - m).abs() <= 0.25 * m.max(value));
            assert!(
                matched,
                "README quotes {claim:?} ({value} s/creature), which matches neither \
                 measured marginal cost in docs/scorer-call-cost.md: {measured:?}"
            );
        }
    }
}

#[test]
fn preamble_stops_at_the_first_section() {
    let readme = "# T\n\nintro\n\n## A\n\nalpha\n";
    assert!(preamble(readme).contains("intro"));
    assert!(!preamble(readme).contains("alpha"));
}

#[test]
fn preamble_returns_everything_when_there_are_no_sections() {
    assert_eq!(preamble("# T\n\nintro\n"), "# T\n\nintro\n");
}

#[test]
fn paragraph_containing_returns_only_the_matching_paragraph() {
    let text = "alpha one\n\nbeta two\nbeta three\n\ngamma four\n";
    assert_eq!(paragraph_containing(text, "alpha"), "alpha one");
    assert_eq!(paragraph_containing(text, "beta"), "beta two\nbeta three");
    assert_eq!(paragraph_containing(text, "gamma"), "gamma four\n");
}

#[test]
#[should_panic(expected = "no occurrence of `missing`")]
fn paragraph_containing_panics_when_the_term_is_absent() {
    paragraph_containing("alpha\n", "missing");
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
fn fenced_block_returns_only_the_requested_language() {
    let md = "intro\n\n```text\nascii\n```\n\n```mermaid\nflowchart TD\n  A --> B\n```\n";
    assert_eq!(fenced_block(md, "text"), Some("ascii"));
    assert_eq!(fenced_block(md, "mermaid"), Some("flowchart TD\n  A --> B"));
}

#[test]
fn fenced_block_is_none_for_a_missing_or_unterminated_block() {
    assert_eq!(fenced_block("no blocks here", "mermaid"), None);
    assert_eq!(fenced_block("```mermaid\nflowchart TD\n", "mermaid"), None);
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

#[test]
fn ordered_list_after_returns_items_and_their_continuations() {
    let text = "lead in:\n\n1. one\n   wrapped\n2. two\n\nafter the list\n";
    let list = ordered_list_after(text, "lead in:");
    assert!(list.contains("one") && list.contains("wrapped") && list.contains("two"));
    assert!(!list.contains("after the list"));
}

#[test]
#[should_panic(expected = "no ordered list follows `lead in:`")]
fn ordered_list_after_panics_when_no_list_follows() {
    ordered_list_after("lead in:\n\nplain prose\n", "lead in:");
}

#[test]
fn time_figures_finds_units_and_ignores_bare_numbers() {
    assert_eq!(
        time_figures("≈0.7–1s/creature on GRQ against ≈11s full"),
        vec!["0.7–1s".to_string(), "11s".to_string()]
    );
    assert_eq!(time_figures("452 ms"), vec!["452 ms".to_string()]);
    assert!(time_figures("`--sample-rate 0.05`, scaled by k^-0.5 so a merge").is_empty());
    assert!(time_figures("5e-2 against 1e-8, issue #111 should land").is_empty());
    assert!(time_figures("").is_empty());
}

#[test]
fn trailing_seconds_reads_ranges_and_units() {
    assert_eq!(trailing_seconds("≈0.7–1s"), Some(vec![0.7, 1.0]));
    assert_eq!(trailing_seconds("0.45 s"), Some(vec![0.45]));
    assert_eq!(trailing_seconds("452 ms"), Some(vec![0.452]));
    assert_eq!(trailing_seconds("0.05"), None);
    assert_eq!(trailing_seconds("87 MB"), None);
    assert_eq!(trailing_seconds(""), None);
}

#[test]
fn per_creature_cost_claims_reads_both_phrasings_and_skips_other_creatures() {
    let claims = per_creature_cost_claims(
        "≈0.7–1s/creature on GRQ, 0.45 s per creature after that, \
         ≈9.9 s before it scores its first creature",
    );
    let seconds: Vec<Vec<f64>> = claims.into_iter().map(|(_, s)| s).collect();
    assert_eq!(seconds, vec![vec![0.7, 1.0], vec![0.45]]);
}

#[test]
fn per_creature_cost_claims_is_empty_without_a_figure() {
    assert!(per_creature_cost_claims("one score per creature").is_empty());
    assert!(per_creature_cost_claims("").is_empty());
}

#[test]
fn measured_marginal_seconds_per_creature_reads_the_result_table() {
    let measured = measured_marginal_seconds_per_creature();
    let screen = measured
        .iter()
        .find(|(phase, _)| phase == "screen")
        .expect("the Result table has a screen row");
    let promote = measured
        .iter()
        .find(|(phase, _)| phase == "promote")
        .expect("the Result table has a promote row");
    // Sanity bounds, not the numbers themselves: the doc stays the source of
    // truth, but a parse that silently returned kilo- or milli-seconds would
    // make the README check vacuous.
    assert!((0.05..5.0).contains(&screen.1), "screen slope {screen:?}");
    assert!(
        promote.1 > screen.1,
        "promote {promote:?} vs screen {screen:?}"
    );
}
