//! README ↔ literature contract (Issue #202).
//!
//! The project is named after a research programme, not a mood: Lamarckian
//! inheritance in evolutionary computation is forty years old and carries a
//! well-known result — write-back converges faster but sheds diversity. The
//! README has to name that literature and say which side of the trade-off this
//! optimiser sits on, because the diversity half is the failure mode an
//! accept-only, single-incumbent optimiser is most exposed to. These tests
//! guard the ways that decays: the section vanishing, a citation losing the
//! author/year that makes it findable, the trade-off softening into a
//! name-drop, and the grounding drifting off code that exists.

use std::path::{Path, PathBuf};

/// A work the README must cite: surname, year, and a fragment of its title.
struct Citation {
    author: &'static str,
    year: &'static str,
    title: &'static str,
}

/// The founding and decisive works for each strand of the design.
const CITATIONS: &[Citation] = &[
    Citation {
        author: "Hinton",
        year: "1987",
        title: "How Learning Can Guide Evolution",
    },
    Citation {
        author: "Nowlan",
        year: "1987",
        title: "How Learning Can Guide Evolution",
    },
    Citation {
        author: "Whitley",
        year: "1994",
        title: "Lamarckian Evolution, the Baldwin Effect and Function Optimization",
    },
    Citation {
        author: "Moscato",
        year: "1989",
        title: "Towards Memetic Algorithms",
    },
    Citation {
        author: "Ong",
        year: "2004",
        title: "Meta-Lamarckian Learning in Memetic Algorithms",
    },
    Citation {
        author: "Krasnogor",
        year: "2005",
        title: "A Tutorial for Competent Memetic Algorithms",
    },
    Citation {
        author: "Such",
        year: "2017",
        title: "Deep Neuroevolution",
    },
    Citation {
        author: "Salimans",
        year: "2017",
        title: "Evolution Strategies as a Scalable Alternative",
    },
    Citation {
        author: "Larra",
        year: "2002",
        title: "Estimation of Distribution Algorithms",
    },
    Citation {
        author: "Pelikan",
        year: "1999",
        title: "The Bayesian Optimization Algorithm",
    },
    Citation {
        author: "Jin",
        year: "2011",
        title: "Surrogate-assisted evolutionary computation",
    },
];

/// Every strand the section has to place, keyed by its subsection heading.
const STRANDS: &[&str] = &[
    "### Lamarckian vs Baldwinian learning",
    "### Memetic algorithms",
    "### Backpropagation-informed variants",
    "### Statistically informed variants",
    "### Adventurous proposal, sceptical acceptance",
];

const LITERATURE_HEADING: &str = "\n## Where this sits in the literature";

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn readme() -> String {
    let path = repo_path("README.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

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

/// Text of the `### ` subsection introduced by `heading`, up to the next `### `.
fn subsection<'a>(text: &'a str, heading: &str) -> &'a str {
    let start = text
        .find(heading)
        .unwrap_or_else(|| panic!("the literature section has no `{heading}` subsection"))
        + heading.len();
    let rest = &text[start..];
    match rest.find("\n### ") {
        Some(end) => &rest[..end],
        None => rest,
    }
}

/// Every `lamarck/src/*.rs` path mentioned in `text`.
fn source_paths(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find("lamarck/src/") {
        let tail = &rest[at..];
        let end = tail
            .find(".rs")
            .map(|e| e + ".rs".len())
            .unwrap_or(tail.len());
        found.push(tail[..end].to_string());
        rest = &tail[end..];
    }
    found
}

fn literature() -> String {
    section(&readme(), LITERATURE_HEADING).to_string()
}

/// The section exists and carries prose, not just a heading.
#[test]
fn the_readme_carries_a_literature_section() {
    let literature = literature();
    assert!(
        literature.split_whitespace().count() > 200,
        "`Where this sits in the literature` is a stub of {} words — it has to place the design, \
         not just name-drop the field",
        literature.split_whitespace().count()
    );
}

/// Each strand named in the issue gets its own subsection, so a reader can find
/// the one that matches the part of the optimiser they are reading about.
#[test]
fn the_literature_section_places_every_strand() {
    let literature = literature();
    for strand in STRANDS {
        assert!(
            literature.contains(strand),
            "the literature section omits the `{strand}` subsection"
        );
    }
}

/// A citation without its author and year is unfindable; without its title it
/// is unverifiable. All three have to be present, in one section.
#[test]
fn every_cited_work_carries_author_year_and_title() {
    let literature = literature();
    let mut missing = Vec::new();
    for citation in CITATIONS {
        for part in [citation.author, citation.year, citation.title] {
            if !literature.contains(part) {
                missing.push(format!("{} {} — {part:?}", citation.author, citation.year));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "the literature section drops these citation parts: {missing:?}"
    );
}

/// Every strand must actually cite something — a subsection with no year in it
/// is commentary that has drifted away from its references.
#[test]
fn every_strand_cites_at_least_one_work() {
    let literature = literature();
    for strand in STRANDS {
        let body = subsection(&literature, strand);
        let cited = CITATIONS
            .iter()
            .filter(|c| body.contains(c.author) && body.contains(c.year))
            .count();
        assert!(
            cited > 0,
            "`{strand}` cites no work — it names a field without placing us in it"
        );
    }
}

/// The whole point of citing Whitley et al.: write-back trades diversity for
/// convergence speed. A section that lost the trade-off kept the name-drop and
/// dropped the finding.
#[test]
fn the_diversity_tradeoff_is_stated_explicitly() {
    let body = subsection(&literature(), STRANDS[0]).to_lowercase();
    for phrase in ["diversity", "converge", "faster", "baldwin"] {
        assert!(
            body.contains(phrase),
            "the Lamarckian-vs-Baldwinian subsection omits {phrase:?} — the headline finding is \
             that write-back converges faster but loses diversity sooner"
        );
    }
}

/// Stating the finding is half of it; the README also has to say which side
/// Lamarck is on and why the loss half matters to an accept-only optimiser.
#[test]
fn the_readme_states_our_position_on_the_tradeoff() {
    let body = subsection(&literature(), STRANDS[0]).to_lowercase();
    assert!(
        body.contains("lamarckian side") || body.contains("lamarckian, not baldwinian"),
        "the subsection never says which side of the trade-off this optimiser is on"
    );
    for phrase in ["incumbent", "premature convergence"] {
        assert!(
            body.contains(phrase),
            "the subsection omits {phrase:?} — it does not say why the diversity loss is our \
             exposure rather than an academic footnote"
        );
    }
}

/// A citation is only worth carrying if it is anchored to code, so every source
/// file the section points a reader at has to exist.
#[test]
fn the_section_is_grounded_in_source_files_that_exist() {
    let literature = literature();
    let paths = source_paths(&literature);
    assert!(
        paths.len() >= 3,
        "the literature section names {} source files — it is not grounded in this codebase",
        paths.len()
    );
    for path in paths {
        assert!(
            repo_path(&path).exists(),
            "the literature section points at a missing {path}"
        );
    }
}

/// 🦒 The giraffe and the tagline stay: the joke is the citation, so citing the
/// literature must not sober the README up (Issue #202 acceptance).
#[test]
fn the_giraffe_and_the_tagline_stay() {
    let readme = readme().to_lowercase();
    assert!(
        readme.contains("giraffe"),
        "the README banner lost the giraffe"
    );
    assert!(
        readme.contains("lamarck would be proud"),
        "the README lost the `Lamarck would be proud` tagline"
    );
}

#[test]
fn section_returns_only_the_requested_section() {
    let readme = "# T\n\n## A\n\nalpha\n\n## B\n\nbeta\n";
    assert!(section(readme, "\n## A").contains("alpha"));
    assert!(!section(readme, "\n## A").contains("beta"));
}

#[test]
#[should_panic(expected = "no `\n## Missing` section")]
fn section_panics_on_a_missing_heading() {
    section("## A\n\nalpha\n", "\n## Missing");
}

#[test]
fn subsection_returns_only_the_requested_subsection() {
    let text = "\n### A\n\nalpha\n\n### B\n\nbeta\n";
    assert!(subsection(text, "\n### A").contains("alpha"));
    assert!(!subsection(text, "\n### A").contains("beta"));
    assert!(subsection(text, "\n### B").contains("beta"));
}

#[test]
#[should_panic(expected = "no `\n### Missing` subsection")]
fn subsection_panics_on_a_missing_subsection() {
    subsection("\n### A\n\nalpha\n", "\n### Missing");
}

#[test]
fn source_paths_finds_every_named_module_and_nothing_else() {
    let text = "see `lamarck/src/run.rs` and `lamarck/src/promote_gate.rs`, but not docs/x.md";
    assert_eq!(
        source_paths(text),
        vec!["lamarck/src/run.rs", "lamarck/src/promote_gate.rs"]
    );
    assert!(source_paths("no modules here").is_empty());
}
