//! In-repo markdown link targets must resolve (Issue #137).
//!
//! A relative link in a markdown file resolves against *that file's own
//! directory*, both on GitHub and in local previews. Paths written for the
//! PR-description context (rendered from the repo root) therefore break the
//! moment the text is archived under `docs/archive/pr-summaries/`, and the
//! reader sees a dead link or a broken image. This walks every tracked
//! markdown file and fails on any relative target that does not exist.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .expect("repo root")
}

/// Every markdown file in the repo, excluding build output and `.git`.
fn markdown_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name == "target" || name == ".git" || name == "node_modules" {
                continue;
            }
            markdown_files(&path, found);
        } else if name.ends_with(".md") {
            found.push(path);
        }
    }
}

/// `markdown` with fenced code blocks removed — a path inside a fence is a
/// quoted example, not a link the renderer resolves.
fn without_fenced_blocks(markdown: &str) -> String {
    let mut out = String::new();
    let mut in_fence = false;
    for line in markdown.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if !in_fence {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// `line` with inline code spans removed — a renderer does not resolve links
/// inside backticks either.
fn without_code_spans(line: &str) -> String {
    let mut out = String::new();
    let mut rest = line;
    while let Some(open) = rest.find('`') {
        out.push_str(&rest[..open]);
        let after = &rest[open..];
        let ticks = after.len() - after.trim_start_matches('`').len();
        let fence = "`".repeat(ticks);
        match after[ticks..].find(&fence) {
            Some(close) => rest = &after[ticks + close + ticks..],
            // Unterminated span: nothing after it is code.
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Link and image targets of every `[...](target)` in `markdown`.
fn link_targets(markdown: &str) -> Vec<String> {
    let text: String = without_fenced_blocks(markdown)
        .lines()
        .map(|line| without_code_spans(line) + "\n")
        .collect();
    let mut targets = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while let Some(offset) = text[i..].find("](") {
        let start = i + offset + 2;
        let Some(end) = text[start..].find(')').map(|e| start + e) else {
            break;
        };
        let raw = text[start..end].trim();
        // Strip an optional link title: `](path "Title")`.
        let target = raw.split_whitespace().next().unwrap_or("");
        targets.push(target.trim_matches(|c| c == '<' || c == '>').to_string());
        i = end + 1;
        if i >= bytes.len() {
            break;
        }
    }
    targets
}

/// True for targets the filesystem cannot answer for — external URLs and
/// same-page anchors.
fn is_external(target: &str) -> bool {
    target.is_empty()
        || target.starts_with('#')
        || target.contains("://")
        || target.starts_with("mailto:")
}

#[test]
fn every_relative_markdown_link_resolves_from_its_own_file() {
    let root = repo_root();
    let mut files = Vec::new();
    markdown_files(&root, &mut files);
    assert!(!files.is_empty(), "no markdown files found under the repo");

    let mut broken = Vec::new();
    for file in &files {
        let markdown = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
        let dir = file.parent().expect("markdown file has a parent");
        for target in link_targets(&markdown) {
            if is_external(&target) {
                continue;
            }
            // Drop a trailing anchor: `docs/foo.md#section`.
            let path = target.split('#').next().unwrap_or("");
            if path.is_empty() {
                continue;
            }
            if !dir.join(path).exists() {
                let shown = file.strip_prefix(&root).unwrap_or(file);
                broken.push(format!("{} -> {target}", shown.display()));
            }
        }
    }

    assert!(
        broken.is_empty(),
        "markdown links that do not resolve relative to their own file:\n  {}",
        broken.join("\n  ")
    );
}

#[test]
fn link_targets_ignores_fenced_examples_but_finds_real_links() {
    let markdown = "\
See [the guide](../guide.md) and ![shot](docs/evidence/a.png \"Title\").

```markdown
![not a link](nowhere/at/all.png)
```

Also [anchored](docs/x.md#part) and [external](https://example.com), while
`[...](target)` in a code span is prose about links, not a link.
";
    let targets = link_targets(markdown);
    assert_eq!(
        targets,
        vec![
            "../guide.md",
            "docs/evidence/a.png",
            "docs/x.md#part",
            "https://example.com",
        ]
    );
    assert!(is_external("https://example.com"));
    assert!(!is_external("../guide.md"));
}
