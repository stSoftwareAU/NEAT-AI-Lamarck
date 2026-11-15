//! Provenance of a benchmark output: the `neat-core` source pinned in the
//! workspace `Cargo.lock`, and the git commit and cleanliness of the checkout
//! that produced it.

use std::path::Path;

/// The git state of a checkout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitState {
    /// Full hash of `HEAD`.
    pub commit: String,
    /// Whether `git status --porcelain` printed anything.
    pub dirty: bool,
}

/// The `source` of the single `neat-core` package in a `Cargo.lock` text.
///
/// Errors name `neat-core` and the cause: no such package, more than one, or
/// no quoted `source` line on it.
pub fn neat_core_source(lock: &str) -> Result<String, String> {
    // Each package starts with a `[[package]]` header; keys at the top level of
    // an entry are unindented, so dependency lists (indented) are not matched.
    let mut sources = Vec::new();
    for entry in lock.split("[[package]]").skip(1) {
        let mut name = None;
        let mut source = None;
        for line in entry.lines() {
            if let Some(value) = line.strip_prefix("name = ") {
                name = quoted(value);
            } else if let Some(value) = line.strip_prefix("source = ") {
                source = quoted(value);
            }
        }
        if name == Some("neat-core") {
            sources.push(source);
        }
    }
    match sources.as_slice() {
        [] => Err("Cargo.lock has no neat-core package".into()),
        [_, _, ..] => Err("Cargo.lock has more than one neat-core package".into()),
        [None] => Err("Cargo.lock's neat-core package has no quoted source line".into()),
        [Some(source)] => Ok((*source).to_string()),
    }
}

/// The text inside a TOML basic string written on one line, if it is one.
fn quoted(value: &str) -> Option<&str> {
    value.strip_prefix('"')?.strip_suffix('"')
}

/// Run `git` in `dir` and return its stdout, or the cause it could not.
fn git(dir: &Path, args: &[&str]) -> Result<String, String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("cannot run git: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed ({}): {}",
            args.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Probe the git checkout containing `dir`. Errors name the cause: git cannot
/// be run, `dir` is not in a repository, or the repository has no commit yet.
pub fn git_state(dir: &Path) -> Result<GitState, String> {
    let commit = git(dir, &["rev-parse", "HEAD"])?.trim().to_string();
    let dirty = !git(dir, &["status", "--porcelain"])?.is_empty();
    Ok(GitState { commit, dirty })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCK: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../Cargo.lock"));

    #[test]
    fn real_lock_pins_neat_core_to_its_git_source() {
        let source = neat_core_source(LOCK).unwrap();
        assert!(
            source.starts_with("git+https://github.com/stSoftwareAU/NEAT-AI-core"),
            "{source}"
        );
    }

    #[test]
    fn lock_without_neat_core_errors_naming_it() {
        let lock = "version = 4\n\n[[package]]\nname = \"serde\"\nversion = \"1.0.0\"\n\
            source = \"registry+https://github.com/rust-lang/crates.io-index\"\n";
        let err = neat_core_source(lock).unwrap_err();
        assert!(err.contains("neat-core"), "{err}");
    }

    #[test]
    fn neat_core_entry_without_a_source_line_errors() {
        let lock = "version = 4\n\n[[package]]\nname = \"neat-core\"\nversion = \"0.23.7\"\n";
        let err = neat_core_source(lock).unwrap_err();
        assert!(err.contains("neat-core"), "{err}");
        assert!(err.contains("source"), "{err}");
    }

    #[test]
    fn a_dependency_mention_is_not_the_package() {
        // `lamarck` depends on neat-core but is not it; its own (path) entry has
        // no source, and must not be mistaken for neat-core's.
        let lock = "version = 4\n\n[[package]]\nname = \"lamarck\"\nversion = \"0.1.0\"\n\
            dependencies = [\n \"neat-core 0.23.7\",\n]\n\n[[package]]\n\
            name = \"neat-core\"\nversion = \"0.23.7\"\n\
            source = \"git+https://example.invalid/neat-core?tag=v0.23.7#abc\"\n";
        assert_eq!(
            neat_core_source(lock).unwrap(),
            "git+https://example.invalid/neat-core?tag=v0.23.7#abc"
        );
    }

    /// Run `git` in `dir` with a fixed identity, so commits work anywhere.
    fn git(dir: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args([
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
            ])
            .args(["-c", "commit.gpgsign=false"])
            .args(args)
            .stdin(std::process::Stdio::null())
            .status()
            .expect("git must be installed to run this test");
        assert!(status.success(), "git {args:?} failed");
    }

    #[test]
    fn git_state_reports_the_commit_and_whether_the_tree_is_dirty() {
        let dir = tempfile::tempdir().unwrap();
        // Not a commit yet: the probe reports the failure, naming the command.
        git(dir.path(), &["init", "-q"]);
        let err = git_state(dir.path()).unwrap_err();
        assert!(err.contains("rev-parse"), "{err}");

        git(dir.path(), &["commit", "-q", "--allow-empty", "-m", "init"]);
        let head = std::process::Command::new("git")
            .arg("-C")
            .arg(dir.path())
            .args(["rev-parse", "HEAD"])
            .output()
            .unwrap();
        let head = String::from_utf8(head.stdout).unwrap().trim().to_string();
        let clean = git_state(dir.path()).unwrap();
        assert_eq!(
            clean,
            GitState {
                commit: head,
                dirty: false
            }
        );

        std::fs::write(dir.path().join("untracked.txt"), "x").unwrap();
        assert!(git_state(dir.path()).unwrap().dirty);
    }
}
