//! Authoritative NEAT-AI-scorer integration.

use crate::config::DEFAULT_MIN_IMPROVEMENT;
use crate::log;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Parsed fields from a scorer result object.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ScoreResult {
    /// Authoritative fitness score (larger-is-better).
    pub score: f64,
    /// Average error (smaller-is-better) — never used alone for acceptance.
    pub error: f64,
    /// Optional complexity penalty.
    #[serde(default)]
    pub complexity_penalty: f64,
}

/// Optional corpus subsample for a scorer directory call (issue #24 screen phase).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreSample {
    /// Fraction of training rows to keep, in `(0, 1]`. `1.0` = full corpus.
    pub rate: f64,
    /// Stratified sample phase (rotates which stratum is kept).
    pub phase: u64,
}

impl ScoreSample {
    /// Full-corpus scoring (no sample flags passed to the scorer).
    pub const fn full() -> Self {
        Self {
            rate: 1.0,
            phase: 0,
        }
    }

    /// True when this requests a proper subsample.
    pub fn is_subsample(self) -> bool {
        self.rate > 0.0 && self.rate < 1.0
    }
}

/// Errors from invoking or interpreting the scorer.
#[derive(Debug)]
pub enum ScorerError {
    /// Process failed to launch or returned non-zero.
    Process(String),
    /// JSON parse failure.
    Json(String),
    /// Missing baseline or candidate stem.
    Missing(String),
    /// Baseline disagreement / invalid comparison.
    Invalid(String),
}

impl std::fmt::Display for ScorerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Process(e) | Self::Json(e) | Self::Missing(e) | Self::Invalid(e) => {
                write!(f, "{e}")
            }
        }
    }
}

impl std::error::Error for ScorerError {}

/// Trait for scoring a directory of creatures (enables fake scorers in tests).
pub trait DirectoryScorer {
    /// Score every `*.json` in `candidates_dir` against the full training corpus.
    fn score_directory(
        &self,
        candidates_dir: &Path,
        training_data: &Path,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        self.score_directory_sampled(candidates_dir, training_data, ScoreSample::full())
    }

    /// Score a directory, optionally on a stratified subsample of training rows.
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        training_data: &Path,
        sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError>;
}

/// Invoke the real `rust_scorer` binary.
#[derive(Debug, Clone)]
pub struct ExternalScorer {
    /// Path to the scorer binary.
    pub binary: PathBuf,
}

impl DirectoryScorer for ExternalScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        training_data: &Path,
        sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        // Full corpus: locked two-arg form (no --gpu / --cost).
        // Screen subsample: add --sample-rate / --sample-phase only (issue #24).
        let mut cmd = Command::new(&self.binary);
        if sample.is_subsample() {
            cmd.arg("--sample-rate")
                .arg(format!("{}", sample.rate))
                .arg("--sample-phase")
                .arg(sample.phase.to_string());
        }
        let output = cmd
            .arg(candidates_dir)
            .arg(training_data)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .output()
            .map_err(|e| ScorerError::Process(format!("failed to spawn scorer: {e}")))?;
        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(ScorerError::Process(format!(
                "scorer exited {}: stdout={stdout}",
                output.status
            )));
        }
        parse_scorer_stdout(&output.stdout)
    }
}

/// Log a compact summary of one scorer batch (timing + score spread).
pub fn log_scorer_batch_stats(
    scores: &BTreeMap<String, ScoreResult>,
    scorer_ms: u128,
    min_improvement: f64,
) {
    log_scorer_batch_stats_labeled(scores, scorer_ms, min_improvement, "scorer");
}

/// Like [`log_scorer_batch_stats`] with a phase label (`screen` / `promote` / …).
pub fn log_scorer_batch_stats_labeled(
    scores: &BTreeMap<String, ScoreResult>,
    scorer_ms: u128,
    min_improvement: f64,
    label: &str,
) {
    let n = scores.len();
    let per = if n > 0 {
        scorer_ms as f64 / n as f64
    } else {
        0.0
    };
    log::ok(&format!(
        "{label} batch: {n} creatures in {scorer_ms}ms ({per:.0} ms/creature, one directory call)"
    ));

    let Some(baseline) = scores.get("baseline") else {
        log::warn(&format!("{label} batch missing baseline"));
        return;
    };
    log::detail(&format!(
        "baseline: score={:.12}  error={:.12}  complexity={}",
        baseline.score, baseline.error, baseline.complexity_penalty
    ));

    let mut deltas: Vec<(&str, f64, f64)> = scores
        .iter()
        .filter(|(stem, _)| stem.as_str() != "baseline")
        .map(|(stem, r)| (stem.as_str(), r.score, r.score - baseline.score))
        .collect();
    deltas.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    if deltas.is_empty() {
        return;
    }

    let best = deltas[0];
    let worst = *deltas.last().unwrap();
    let above = deltas
        .iter()
        .filter(|(_, _, d)| *d > min_improvement)
        .count();
    let improved = deltas.iter().filter(|(_, _, d)| *d > 0.0).count();
    log::detail(&format!(
        "candidates: best Δ {:+.6e} ({})  worst Δ {:+.6e} ({})  >0: {improved}/{}  >threshold: {above}/{}",
        best.2,
        best.0,
        worst.2,
        worst.0,
        deltas.len(),
        deltas.len()
    ));

    let show = deltas.len().min(5);
    for (stem, score, delta) in deltas.iter().take(show) {
        log::detail(&format!("  {stem}: score={score:.12}  Δ {delta:+.6e}"));
    }
    if deltas.len() > show {
        log::detail(&format!("  … {} more candidates", deltas.len() - show));
    }
}

/// Stems (excluding baseline) whose sample score beats baseline by more than `threshold`.
pub fn screen_promote_stems(
    scores: &BTreeMap<String, ScoreResult>,
    threshold: f64,
) -> Result<Vec<String>, ScorerError> {
    let baseline = scores
        .get("baseline")
        .ok_or_else(|| ScorerError::Missing("baseline missing from scorer results".into()))?;
    let mut stems: Vec<(String, f64)> = scores
        .iter()
        .filter(|(stem, _)| stem.as_str() != "baseline")
        .map(|(stem, r)| (stem.clone(), improvement(r.score, baseline.score)))
        .filter(|(_, delta)| *delta > threshold)
        .collect();
    stems.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(stems.into_iter().map(|(s, _)| s).collect())
}

/// Copy `baseline.json` + promoted candidate JSON into a fresh promote directory.
pub fn write_promote_batch(
    promote_dir: &Path,
    source_batch: &Path,
    promote_stems: &[String],
) -> Result<(), String> {
    if promote_dir.exists() {
        fs::remove_dir_all(promote_dir).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(promote_dir).map_err(|e| e.to_string())?;
    let baseline_src = source_batch.join("baseline.json");
    fs::copy(&baseline_src, promote_dir.join("baseline.json")).map_err(|e| {
        format!(
            "copy baseline from {} failed: {e}",
            baseline_src.display()
        )
    })?;
    for stem in promote_stems {
        let name = format!("{stem}.json");
        let src = source_batch.join(&name);
        fs::copy(&src, promote_dir.join(&name))
            .map_err(|e| format!("copy {stem} from {} failed: {e}", src.display()))?;
    }
    Ok(())
}

/// Parse scorer stdout JSON (stem-keyed map).
pub fn parse_scorer_stdout(stdout: &[u8]) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
    let text = std::str::from_utf8(stdout)
        .map_err(|e| ScorerError::Json(format!("scorer stdout not utf-8: {e}")))?;
    // Scorer may print diagnostics on stderr; stdout should be JSON. Tolerate
    // leading/trailing whitespace.
    let trimmed = text.trim();
    serde_json::from_str(trimmed)
        .map_err(|e| ScorerError::Json(format!("malformed scorer JSON: {e}; body={trimmed}")))
}

/// Decide whether a candidate beats the baseline on **score**.
pub fn improvement(candidate: f64, baseline: f64) -> f64 {
    candidate - baseline
}

/// True when the absolute score improvement exceeds the threshold (strict `>`).
pub fn accepts_improvement(
    candidate_score: f64,
    baseline_score: f64,
    min_improvement: f64,
) -> bool {
    improvement(candidate_score, baseline_score) > min_improvement
}

/// Select the best qualifying candidate from a scored batch.
pub fn select_winner(
    results: &BTreeMap<String, ScoreResult>,
    min_improvement: f64,
) -> Result<Option<(&str, &ScoreResult, f64)>, ScorerError> {
    let baseline = results
        .get("baseline")
        .ok_or_else(|| ScorerError::Missing("baseline missing from scorer results".into()))?;
    let mut best: Option<(&str, &ScoreResult, f64)> = None;
    for (stem, result) in results {
        if stem == "baseline" {
            continue;
        }
        let delta = improvement(result.score, baseline.score);
        if delta > min_improvement {
            match best {
                Some((_, _, best_delta)) if delta <= best_delta => {}
                _ => best = Some((stem.as_str(), result, delta)),
            }
        }
    }
    // Microscopic changes below the default threshold must be rejected.
    let _ = DEFAULT_MIN_IMPROVEMENT;
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::Mutex;

    struct FakeScorer {
        payload: Arc<Mutex<BTreeMap<String, ScoreResult>>>,
    }

    impl DirectoryScorer for FakeScorer {
        fn score_directory_sampled(
            &self,
            _candidates_dir: &Path,
            _training_data: &Path,
            _sample: ScoreSample,
        ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
            Ok(self.payload.lock().unwrap().clone())
        }
    }

    #[test]
    fn accepts_only_score_above_threshold() {
        assert!(!accepts_improvement(0.5 + 1e-7, 0.5, 1e-6));
        assert!(accepts_improvement(0.5 + 2e-6, 0.5, 1e-6));
    }

    #[test]
    fn select_winner_picks_best_qualifying() {
        let mut map = BTreeMap::new();
        map.insert(
            "baseline".into(),
            ScoreResult {
                score: 0.4,
                error: 0.6,
                complexity_penalty: 0.0,
            },
        );
        map.insert(
            "candidate-000".into(),
            ScoreResult {
                score: 0.4 + 5e-7,
                error: 0.6,
                complexity_penalty: 0.0,
            },
        );
        map.insert(
            "candidate-001".into(),
            ScoreResult {
                score: 0.4 + 3e-6,
                error: 0.59,
                complexity_penalty: 0.0,
            },
        );
        map.insert(
            "candidate-002".into(),
            ScoreResult {
                score: 0.4 + 2e-6,
                error: 0.595,
                complexity_penalty: 0.0,
            },
        );
        let winner = select_winner(&map, 1e-6).unwrap().unwrap();
        assert_eq!(winner.0, "candidate-001");
        assert!(winner.2 > 1e-6);
    }

    #[test]
    fn malformed_json_is_error() {
        let err = parse_scorer_stdout(b"not-json").unwrap_err();
        assert!(matches!(err, ScorerError::Json(_)));
    }

    #[test]
    fn fake_scorer_round_trip() {
        let mut map = BTreeMap::new();
        map.insert(
            "baseline".into(),
            ScoreResult {
                score: 0.1,
                error: 0.9,
                complexity_penalty: 0.0,
            },
        );
        let fake = FakeScorer {
            payload: Arc::new(Mutex::new(map)),
        };
        let out = fake
            .score_directory(Path::new("."), Path::new("."))
            .unwrap();
        assert!(out.contains_key("baseline"));
    }

    #[test]
    fn screen_promote_stems_keeps_only_positive_deltas() {
        let mut map = BTreeMap::new();
        map.insert(
            "baseline".into(),
            ScoreResult {
                score: 0.5,
                error: 0.5,
                complexity_penalty: 0.0,
            },
        );
        map.insert(
            "candidate-000".into(),
            ScoreResult {
                score: 0.5 + 1e-4,
                error: 0.4,
                complexity_penalty: 0.0,
            },
        );
        map.insert(
            "candidate-001".into(),
            ScoreResult {
                score: 0.5 - 1e-4,
                error: 0.6,
                complexity_penalty: 0.0,
            },
        );
        map.insert(
            "candidate-002".into(),
            ScoreResult {
                score: 0.5 + 2e-4,
                error: 0.3,
                complexity_penalty: 0.0,
            },
        );
        let stems = screen_promote_stems(&map, 0.0).unwrap();
        assert_eq!(stems, vec!["candidate-002", "candidate-000"]);
        assert!(screen_promote_stems(&map, 1e-3).unwrap().is_empty());
    }

    #[test]
    fn write_promote_batch_copies_baseline_and_stems() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("baseline.json"), b"{}").unwrap();
        fs::write(src.join("candidate-000.json"), b"{\"a\":1}").unwrap();
        fs::write(src.join("candidate-001.json"), b"{\"b\":2}").unwrap();
        let promote = dir.path().join("promote");
        write_promote_batch(
            &promote,
            &src,
            &["candidate-001".into()],
        )
        .unwrap();
        assert!(promote.join("baseline.json").is_file());
        assert!(promote.join("candidate-001.json").is_file());
        assert!(!promote.join("candidate-000.json").exists());
    }
}
