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
