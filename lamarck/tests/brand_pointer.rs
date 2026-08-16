//! `docs/brand/` ↔ hub-repository contract (Issue #149).
//!
//! The NEAT-AI family brand set (social previews, transparent mark) is shared
//! by every sibling repo, so it lives in the hub repository
//! <https://github.com/stSoftwareAU/NEAT-AI> and not here. All this repo keeps
//! is a pointer, and these tests guard the two ways that decays: artwork
//! creeping back in so the two copies drift, and the pointer losing the link
//! that tells a reader where the real assets are.

use std::path::{Path, PathBuf};

/// URL of the canonical brand directory in the hub repository.
const BRAND_HOME_URL: &str = "https://github.com/stSoftwareAU/NEAT-AI/tree/Develop/docs/brand";

/// Raw URL the README banner hot-links (Issue #151). Pinned to a branch path
/// rather than a commit, so regenerated hub artwork flows through untouched.
const BANNER_URL: &str = "https://raw.githubusercontent.com/stSoftwareAU/NEAT-AI/Develop/docs/brand/social-previews/neat-ai-lamarck.png";

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn pointer() -> String {
    let path = repo_path("docs/brand/README.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn readme() -> String {
    let path = repo_path("README.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Alt text and link target of the first markdown image in `markdown`.
fn first_image(markdown: &str) -> Option<(String, String)> {
    let start = markdown.find("![")?;
    let rest = &markdown[start + 2..];
    let alt_end = rest.find("](")?;
    let alt = &rest[..alt_end];
    let target_rest = &rest[alt_end + 2..];
    let target_end = target_rest.find(')')?;
    Some((alt.to_string(), target_rest[..target_end].to_string()))
}

/// Every image file found anywhere under `dir`.
fn image_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(image_files(&path));
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("png" | "jpg" | "jpeg" | "svg" | "webp")
        ) {
            found.push(path);
        }
    }
    found
}

/// The artwork moved to the hub repo, so no copy may live here.
#[test]
fn no_brand_artwork_is_kept_in_this_repo() {
    let brand = repo_path("docs/brand");
    let images = image_files(&brand);
    assert!(
        images.is_empty(),
        "docs/brand/ holds brand artwork that belongs in {BRAND_HOME_URL}: {images:?}"
    );
}

/// A reader who lands on `docs/brand/` has to be sent to the real assets.
#[test]
fn the_pointer_names_the_hub_repository_brand_home() {
    let pointer = pointer();
    assert!(
        pointer.contains(BRAND_HOME_URL),
        "docs/brand/README.md does not link the canonical brand home {BRAND_HOME_URL}"
    );
}

/// The README opens with this repo's own brand banner (Issue #151): the first
/// image in the file sits directly under the top-level heading and carries alt
/// text, so a reader (or a screen reader) meets the project mark first.
#[test]
fn the_readme_banner_follows_the_top_level_heading() {
    let readme = readme();
    let heading = "# NEAT-AI-Lamarck\n";
    let after = readme
        .find(heading)
        .map(|start| &readme[start + heading.len()..])
        .expect("README.md has no `# NEAT-AI-Lamarck` heading");

    let banner = after
        .lines()
        .find(|line| !line.trim().is_empty())
        .expect("README.md ends at the heading — no banner follows it");

    let (alt, target) = first_image(banner).unwrap_or_else(|| {
        panic!("the first element under the README heading is not an image: {banner:?}")
    });
    assert!(
        !alt.trim().is_empty(),
        "the README banner has empty alt text: {banner:?}"
    );
    assert_eq!(
        target, BANNER_URL,
        "the README banner does not point at this repo's hub preview"
    );
}

/// The banner is hot-linked, never a committed copy: a repo-relative target
/// would mean artwork had been vendored back in alongside it.
#[test]
fn every_readme_image_is_hot_linked_from_the_hub() {
    let readme = readme();
    let mut rest = readme.as_str();
    let mut images = 0;
    while let Some((alt, target)) = first_image(rest) {
        assert!(
            target.starts_with("https://raw.githubusercontent.com/stSoftwareAU/NEAT-AI/"),
            "README image {alt:?} is not hot-linked from the hub repository: {target}"
        );
        images += 1;
        let consumed = rest.find("](").expect("image had a link target") + 2;
        rest = &rest[consumed..];
    }
    assert!(images > 0, "README.md embeds no brand banner at all");
}

/// The social preview for this repo is part of the family set, so the pointer
/// must name the file a maintainer uploads under Settings → Social preview.
#[test]
fn the_pointer_names_this_repos_social_preview() {
    let pointer = pointer();
    assert!(
        pointer.contains("neat-ai-lamarck.png"),
        "docs/brand/README.md does not name this repo's social preview image"
    );
}
