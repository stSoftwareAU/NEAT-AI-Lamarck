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
