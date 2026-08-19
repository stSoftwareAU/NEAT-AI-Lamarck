//! CONTRIBUTING ↔ README contract (Issue #136).
//!
//! The README owns the build/gate instructions. `CONTRIBUTING.md` keeps only
//! the contributor-specific habits and links into the README for the
//! mechanics, so the two cannot drift apart. A copy pasted back into
//! `CONTRIBUTING.md` fails here rather than quietly disagreeing with the
//! README months later.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Consecutive-word run treated as a restatement rather than a coincidence.
const SHINGLE_WORDS: usize = 12;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read_doc(name: &str) -> String {
    let path = repo_root().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Lowercased word stream with markdown punctuation stripped, so a phrase that
/// is re-wrapped or re-linked still compares equal to its original.
fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// Every run of `SHINGLE_WORDS` consecutive words in `text`.
fn shingles(text: &str) -> HashSet<String> {
    let words = words(text);
    words
        .windows(SHINGLE_WORDS)
        .map(|window| window.join(" "))
        .collect()
}

/// Bodies of all ```` ``` ```` fenced blocks, whitespace-normalised.
fn fenced_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current: Option<Vec<&str>> = None;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            match current.take() {
                Some(body) => blocks.push(body.join("\n")),
                None => current = Some(Vec::new()),
            }
            continue;
        }
        if let Some(body) = current.as_mut() {
            body.push(line.trim_end());
        }
    }
    blocks
        .into_iter()
        .map(|block| block.trim().to_string())
        .filter(|block| !block.is_empty())
        .collect()
}

/// GitHub's heading slug: lowercase, punctuation dropped, spaces to hyphens.
fn slug(heading: &str) -> String {
    heading
        .trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '-')
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
}

fn heading_slugs(markdown: &str) -> HashSet<String> {
    markdown
        .lines()
        .filter_map(|line| line.strip_prefix('#'))
        .map(|rest| slug(rest.trim_start_matches('#')))
        .collect()
}

/// Every `README.md#anchor` target linked from `text`.
fn readme_anchor_links(text: &str) -> Vec<String> {
    let mut anchors = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find("README.md#") {
        let tail = &rest[at + "README.md#".len()..];
        let end = tail
            .find(|c: char| !(c.is_alphanumeric() || c == '-'))
            .unwrap_or(tail.len());
        anchors.push(tail[..end].to_string());
        rest = &tail[end..];
    }
    anchors
}

/// CONTRIBUTING points readers at the README's build/gate section instead of
/// carrying its own copy.
#[test]
fn contributing_links_to_the_readme_build_and_quality_gate_section() {
    let contributing = read_doc("CONTRIBUTING.md");
    let anchors = readme_anchor_links(&contributing);
    assert!(
        anchors.iter().any(|a| a == "build-and-quality-gate"),
        "CONTRIBUTING.md does not link to README.md#build-and-quality-gate; \
         it linked {anchors:?}"
    );
}

/// Every README anchor CONTRIBUTING links to must exist, or the de-duplication
/// has simply traded a stale copy for a dead link.
#[test]
fn contributing_readme_anchors_resolve_to_readme_headings() {
    let contributing = read_doc("CONTRIBUTING.md");
    let slugs = heading_slugs(&read_doc("README.md"));
    let dangling: Vec<String> = readme_anchor_links(&contributing)
        .into_iter()
        .filter(|anchor| !slugs.contains(anchor))
        .collect();
    assert!(
        dangling.is_empty(),
        "CONTRIBUTING.md links to README anchors that no heading defines: {dangling:?}"
    );
}

/// No prose is maintained in both documents.
#[test]
fn contributing_does_not_restate_readme_prose() {
    let contributing = read_doc("CONTRIBUTING.md");
    let readme = shingles(&read_doc("README.md"));
    let mut duplicated: Vec<String> = shingles(&contributing)
        .into_iter()
        .filter(|shingle| readme.contains(shingle))
        .collect();
    duplicated.sort();
    assert!(
        duplicated.is_empty(),
        "CONTRIBUTING.md restates README prose ({SHINGLE_WORDS}+ words in common) \
         — link to the README instead: {duplicated:?}"
    );
}

/// No command or diagram is maintained in both documents either: a fenced
/// block copied from the README is the duplication this contract exists to
/// stop.
#[test]
fn contributing_does_not_restate_readme_code_blocks() {
    let readme_blocks: HashSet<String> =
        fenced_blocks(&read_doc("README.md")).into_iter().collect();
    let duplicated: Vec<String> = fenced_blocks(&read_doc("CONTRIBUTING.md"))
        .into_iter()
        .filter(|block| readme_blocks.contains(block))
        .collect();
    assert!(
        duplicated.is_empty(),
        "CONTRIBUTING.md repeats README fenced blocks verbatim: {duplicated:?}"
    );
}

/// Build/gate facts the README owns. CONTRIBUTING must link to the README
/// rather than name any of them, because a second description of the same
/// fact is a second thing to keep current (Issue #136).
const README_OWNED_FACTS: &[&str] = &[
    // Prerequisites and their install commands.
    "rust-toolchain.toml",
    "shellcheck",
    "cargo-deny",
    "codespell",
    // Cargo profiles.
    ".cargo/config.toml",
    "opt-level",
    "line-tables-only",
    // The local gate and the PR workflows it validates.
    "quality.sh",
    "auto-format.yml",
    "version-increment.yml",
    "cargo fmt",
    // The breaking-bump acknowledgement rule.
    "neat-core.expected-version",
];

/// The build/gate mechanics live in the README only.
#[test]
fn contributing_does_not_restate_readme_owned_build_facts() {
    let contributing = read_doc("CONTRIBUTING.md").to_lowercase();
    let restated: Vec<&&str> = README_OWNED_FACTS
        .iter()
        .filter(|fact| contributing.contains(&fact.to_lowercase()))
        .collect();
    assert!(
        restated.is_empty(),
        "CONTRIBUTING.md restates build/gate facts the README owns — link to \
         README.md#build-and-quality-gate instead of naming: {restated:?}"
    );
}

/// The contributor-specific material CONTRIBUTING owns stays in CONTRIBUTING —
/// de-duplication must not hollow the file out.
#[test]
fn contributing_keeps_the_contributor_specific_habits() {
    let contributing = read_doc("CONTRIBUTING.md").to_lowercase();
    for topic in ["lamarck/cargo.toml", "changelog.md", "unreleased"] {
        assert!(
            contributing.contains(topic),
            "CONTRIBUTING.md no longer mentions {topic:?} — the version-bump and \
             CHANGELOG habits are contributor-specific and belong here"
        );
    }
}
