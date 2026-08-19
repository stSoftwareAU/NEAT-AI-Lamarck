//! `docs/architecture.md` ↔ README contract (Issue #173).
//!
//! The README is this repository's single source of truth for how the
//! optimiser is put together. `docs/architecture.md` held a second, thinner
//! description of the same responsibilities, iteration lifecycle and locked
//! contracts, and nothing linked to it — so a reader who found it by browsing
//! `docs/` had no signal it was the secondary copy. It is now a pointer, and
//! these tests guard the two ways that decays: the pointer growing back into a
//! rival architecture document, and a top-level doc going unlinked again.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .expect("repo root")
}

fn read(relative: &str) -> String {
    let path = repo_root().join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {relative}: {e}"))
}

/// Markdown files under `dir`, skipping build output, `.git` and the PR-summary
/// archive (archived summaries describe a past state and cannot keep a doc
/// alive).
fn markdown_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if matches!(
                name.as_ref(),
                "target" | ".git" | "node_modules" | "archive"
            ) {
                continue;
            }
            markdown_files(&path, found);
        } else if name.ends_with(".md") {
            found.push(path);
        }
    }
}

/// Relative link targets in `markdown`, resolved against `dir`, ignoring
/// fenced code blocks (a path inside a fence is a quoted example, not a link).
fn resolved_link_targets(markdown: &str, dir: &Path) -> Vec<PathBuf> {
    let mut resolved = Vec::new();
    let mut in_fence = false;
    for line in markdown.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        let mut rest = line;
        while let Some(open) = rest.find("](") {
            let after = &rest[open + 2..];
            let Some(close) = after.find(')') else { break };
            let raw = after[..close].trim();
            let target = raw.split_whitespace().next().unwrap_or("");
            let target = target.split('#').next().unwrap_or("");
            let resolvable =
                !target.is_empty() && !target.contains("://") && !target.starts_with("mailto:");
            if resolvable && let Ok(path) = dir.join(target).canonicalize() {
                resolved.push(path);
            }
            rest = &after[close + 1..];
        }
    }
    resolved
}

/// Headings (`#`-prefixed lines) in `markdown`, outside fenced code blocks.
fn headings(markdown: &str) -> Vec<String> {
    let mut in_fence = false;
    let mut found = Vec::new();
    for line in markdown.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if !in_fence && line.starts_with('#') {
            found.push(line.to_string());
        }
    }
    found
}

/// Every top-level `docs/*.md` must be linked from at least one other markdown
/// file outside the PR-summary archive. An orphan is unreachable by a reader
/// following the docs, and drifts from whatever it duplicates.
#[test]
fn no_top_level_doc_is_orphaned() {
    let root = repo_root();
    let mut files = Vec::new();
    markdown_files(&root, &mut files);

    let mut orphans = Vec::new();
    for entry in std::fs::read_dir(root.join("docs")).expect("read docs/") {
        let doc = entry.expect("docs entry").path();
        if doc.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let doc = doc.canonicalize().expect("canonicalize doc");
        let linked = files.iter().filter(|file| **file != doc).any(|file| {
            let markdown = std::fs::read_to_string(file).unwrap_or_default();
            let dir = file.parent().expect("markdown file has a parent");
            resolved_link_targets(&markdown, dir).contains(&doc)
        });
        if !linked {
            orphans.push(
                doc.strip_prefix(&root)
                    .unwrap_or(&doc)
                    .display()
                    .to_string(),
            );
        }
    }
    orphans.sort();

    assert!(
        orphans.is_empty(),
        "top-level docs linked from nowhere — a reader cannot find them and they drift: {orphans:?}"
    );
}

/// The pointer must send the reader to the README rather than answer the
/// architecture question itself.
#[test]
fn the_architecture_doc_points_at_the_readme() {
    let pointer = read("docs/architecture.md");
    let dir = repo_root().join("docs");
    let readme = repo_root()
        .join("README.md")
        .canonicalize()
        .expect("README");
    assert!(
        resolved_link_targets(&pointer, &dir).contains(&readme),
        "docs/architecture.md does not link the README it defers to"
    );
}

/// A pointer with sections is a rival architecture document again. One title
/// and a couple of sentences is the whole file.
#[test]
fn the_architecture_pointer_holds_no_sections_of_its_own() {
    let pointer = read("docs/architecture.md");
    let extra: Vec<String> = headings(&pointer)
        .into_iter()
        .filter(|heading| !heading.starts_with("# "))
        .collect();
    assert!(
        extra.is_empty(),
        "docs/architecture.md has grown sections of its own instead of pointing at the README: {extra:?}"
    );
    let prose = pointer
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    assert!(
        prose <= 8,
        "docs/architecture.md is {prose} non-blank lines — a pointer, not a second architecture doc"
    );
}

/// The two notes that lived only in the architecture doc survive the fold into
/// the README's `## Related repositories`, where the same split is described.
#[test]
fn related_repositories_keeps_the_folded_boundary_notes() {
    let readme = read("README.md");
    let start = readme
        .find("\n## Related repositories")
        .expect("README.md has no `## Related repositories` section");
    let rest = &readme[start + 1..];
    let section = match rest.find("\n## ") {
        Some(end) => &rest[..end],
        None => rest,
    };
    for note in [
        "only after the experiment proves useful and its interfaces stabilise",
        "must not duplicate this authority",
    ] {
        assert!(
            section.contains(note),
            "README `## Related repositories` lost the folded architecture note: {note:?}"
        );
    }
}

/// The README's own `## Repository layout` has to name the pointer, so the
/// reader meets it as a signpost rather than stumbling on it in `docs/`.
#[test]
fn repository_layout_links_the_architecture_pointer() {
    let readme = read("README.md");
    let start = readme
        .find("\n## Repository layout")
        .expect("README.md has no `## Repository layout` section");
    let rest = &readme[start + 1..];
    let section = match rest.find("\n## ") {
        Some(end) => &rest[..end],
        None => rest,
    };
    let root = repo_root();
    let pointer = root
        .join("docs/architecture.md")
        .canonicalize()
        .expect("docs/architecture.md");
    assert!(
        resolved_link_targets(section, &root).contains(&pointer),
        "README `## Repository layout` does not link docs/architecture.md — it stays undiscoverable"
    );
}
