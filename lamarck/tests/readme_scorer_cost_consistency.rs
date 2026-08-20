//! README Phase 5 ↔ `docs/scorer-call-cost.md` cost contract (Issue #172).
//!
//! Phase 5 used to quote per-creature scoring costs written before the
//! fixed/marginal decomposition of issue #112 existed ("≈0.7–1s/creature on GRQ
//! against ≈11s full"), roughly double the measured 452 ms screen and 5 490 ms
//! promote marginal costs the README itself cites as authoritative two sections
//! later. These tests re-derive the measured figures from the document's own
//! Result table and fail on any Phase 5 timing that contradicts them, whichever
//! side of the contradiction drifts next.

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

/// One row of the `## Result` table: a phase and its least-squares fit.
#[derive(Debug, PartialEq)]
struct MeasuredPhase {
    phase: String,
    fixed_ms: f64,
    marginal_ms: f64,
}

/// Body of the section introduced by `heading`, up to the next heading of any
/// level. `heading` carries its own leading newline and `#` marks.
fn section<'a>(text: &'a str, heading: &str) -> &'a str {
    let start = text
        .find(heading)
        .unwrap_or_else(|| panic!("no `{}` section", heading.trim()))
        + heading.len();
    let rest = &text[start..];
    match rest.find("\n#") {
        Some(end) => &rest[..end],
        None => rest,
    }
}

/// Parse a `9 898 ms`-style cell — spaces group thousands, `**` bolds it.
fn cell_ms(cell: &str) -> Option<f64> {
    let digits: String = cell
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    if digits.is_empty() || !cell.contains("ms") {
        return None;
    }
    digits.parse().ok()
}

/// Every phase row of `docs/scorer-call-cost.md`'s `## Result` table.
fn measured_costs(doc: &str) -> Vec<MeasuredPhase> {
    section(doc, "\n## Result")
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('|'))
        .filter_map(|line| {
            let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
            if cells.len() < 5 {
                return None;
            }
            Some(MeasuredPhase {
                phase: cells[0].to_string(),
                fixed_ms: cell_ms(cells[3])?,
                marginal_ms: cell_ms(cells[4])?,
            })
        })
        .collect()
}

/// A numeric token in `text`, with the byte range it occupies.
struct Number {
    start: usize,
    end: usize,
    value: f64,
}

fn numbers(text: &str) -> Vec<Number> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut found = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].1.is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = chars[i].0;
        let mut end = i;
        while end < chars.len() && (chars[end].1.is_ascii_digit() || chars[end].1 == '.') {
            end += 1;
        }
        let raw = &text[start..chars.get(end).map_or(text.len(), |c| c.0)];
        if let Ok(value) = raw.trim_end_matches('.').parse::<f64>() {
            found.push(Number {
                start,
                end: start + raw.trim_end_matches('.').len(),
                value,
            });
        }
        i = end;
    }
    found
}

/// Milliseconds per unit of a `ms`/`s` suffix at the start of `rest`, allowing
/// one separating space. `None` when no duration unit follows.
fn unit_ms(rest: &str) -> Option<f64> {
    let rest = rest.strip_prefix(' ').unwrap_or(rest);
    let boundary = |tail: &str| tail.chars().next().is_none_or(|c| !c.is_alphanumeric());
    if let Some(tail) = rest.strip_prefix("ms") {
        return boundary(tail).then_some(1.0);
    }
    if let Some(tail) = rest.strip_prefix('s') {
        return boundary(tail).then_some(1000.0);
    }
    None
}

/// Every duration `text` states, as `(literal, milliseconds)`.
///
/// The left end of a range inherits the unit written once at the right end, so
/// `0.7–1s` reads as two durations rather than one.
fn durations(text: &str) -> Vec<(String, f64)> {
    let found = numbers(text);
    let mut out = Vec::new();
    for (i, number) in found.iter().enumerate() {
        let mut unit = unit_ms(&text[number.end..]);
        if unit.is_none() {
            let next = found.get(i + 1);
            let gap = next.map(|n| text[number.end..n.start].trim());
            if matches!(gap, Some("–" | "-" | "—" | "to")) {
                unit = next.and_then(|n| unit_ms(&text[n.end..]));
            }
        }
        if let Some(per_unit) = unit {
            out.push((
                text[number.start..number.end].to_string(),
                number.value * per_unit,
            ));
        }
    }
    out
}

fn readme_phase_five() -> String {
    section(&read("README.md"), "\n### Phase 5").to_string()
}

/// The measurement the README defers to has to stay machine-readable, or the
/// consistency check below would pass by finding nothing to compare against.
#[test]
fn the_measured_result_table_still_parses_into_per_phase_fits() {
    let measured = measured_costs(&read("docs/scorer-call-cost.md"));
    let phases: Vec<&str> = measured.iter().map(|m| m.phase.as_str()).collect();
    for phase in ["screen", "promote"] {
        assert!(
            phases.contains(&phase),
            "docs/scorer-call-cost.md `## Result` no longer fits the {phase} phase: {phases:?}"
        );
    }
    for row in &measured {
        assert!(
            row.fixed_ms > 0.0 && row.marginal_ms > 0.0,
            "the {} row reports a non-positive fit: {row:?}",
            row.phase
        );
    }
}

/// Every timing Phase 5 states must be one `docs/scorer-call-cost.md` measured.
#[test]
fn phase_five_states_no_timing_the_measurement_contradicts() {
    let measured = measured_costs(&read("docs/scorer-call-cost.md"));
    assert!(!measured.is_empty(), "no measured costs to check against");
    let supported: Vec<f64> = measured
        .iter()
        .flat_map(|m| [m.fixed_ms, m.marginal_ms])
        .collect();

    let contradicted: Vec<String> = durations(&readme_phase_five())
        .into_iter()
        .filter(|(_, ms)| {
            !supported
                .iter()
                .any(|measured| (measured - ms).abs() <= 0.05 * measured)
        })
        .map(|(literal, ms)| format!("`{literal}` (= {ms} ms)"))
        .collect();

    assert!(
        contradicted.is_empty(),
        "README Phase 5 states scorer timings docs/scorer-call-cost.md does not measure: \
         {contradicted:?} — measured ms: {supported:?}. Cite the document rather than \
         restating its numbers."
    );
}

/// Phase 5 is where a reader decides screen-versus-promote economics, so it has
/// to point at the measurement instead of leaving them to guess.
#[test]
fn phase_five_cites_the_measured_scorer_call_cost_document() {
    assert!(
        readme_phase_five().contains("docs/scorer-call-cost.md"),
        "README Phase 5 does not link docs/scorer-call-cost.md as the source of scoring costs"
    );
}

#[test]
fn section_returns_only_the_requested_heading() {
    let md = "# T\n\n## A\n\nalpha\n\n## B\n\nbeta\n";
    assert!(section(md, "\n## A").contains("alpha"));
    assert!(!section(md, "\n## A").contains("beta"));
}

#[test]
fn section_stops_at_a_deeper_heading_too() {
    let md = "# T\n\n## A\n\nalpha\n\n### A1\n\nnested\n";
    let a = section(md, "\n## A");
    assert!(a.contains("alpha"));
    assert!(!a.contains("nested"));
}

#[test]
#[should_panic(expected = "no `## Missing` section")]
fn section_panics_on_a_missing_heading() {
    section("\n## A\n\nalpha\n", "\n## Missing");
}

#[test]
fn measured_costs_reads_bolded_space_grouped_cells() {
    let doc = "\n## Result\n\n\
        | Phase | Sample rate | Calls | `fixedMs` | `marginalMsPerCreature` | `rSquared` |\n\
        |---|---|---|---|---|---|\n\
        | screen | `0.05` | 9 | **9 898 ms** | 452 ms | 0.957 |\n\
        | promote | full corpus | 6 | **1 977 ms** | 5 490 ms | 0.973 |\n\n\
        ## Next\n";
    assert_eq!(
        measured_costs(doc),
        vec![
            MeasuredPhase {
                phase: "screen".into(),
                fixed_ms: 9898.0,
                marginal_ms: 452.0,
            },
            MeasuredPhase {
                phase: "promote".into(),
                fixed_ms: 1977.0,
                marginal_ms: 5490.0,
            },
        ]
    );
}

#[test]
fn measured_costs_is_empty_when_the_table_carries_no_millisecond_cells() {
    let doc = "\n## Result\n\nProse only, no table.\n";
    assert!(measured_costs(doc).is_empty());
}

#[test]
fn durations_reads_the_pre_fix_phase_five_wording() {
    let old = "`rust_scorer --sample-rate 0.05 …` (≈0.7–1s/creature on GRQ against ≈11s full).";
    let ms: Vec<f64> = durations(old).into_iter().map(|(_, ms)| ms).collect();
    assert_eq!(ms, vec![700.0, 1000.0, 11000.0]);
}

#[test]
fn durations_reads_both_units_and_ignores_bare_numbers() {
    let ms: Vec<f64> = durations("452 ms, 5.49 s, 30% of 87 MB at rate 0.05")
        .into_iter()
        .map(|(_, ms)| ms)
        .collect();
    assert_eq!(ms, vec![452.0, 5490.0]);
}

#[test]
fn durations_ignores_words_that_merely_start_with_a_unit_letter() {
    assert!(durations("0.05 sample, 244 promotions, 1e-6 threshold").is_empty());
    assert!(durations("").is_empty());
}
