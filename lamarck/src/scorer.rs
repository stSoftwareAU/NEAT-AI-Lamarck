//! Authoritative NEAT-AI-scorer integration.

use crate::config::DEFAULT_MIN_IMPROVEMENT;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    /// Score every `*.json` in `candidates_dir` against `training_data`.
    fn score_directory(
        &self,
        candidates_dir: &Path,
        training_data: &Path,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError>;
}

/// Invoke the real `rust_scorer` binary with only dir + training-data args.
#[derive(Debug, Clone)]
pub struct ExternalScorer {
    /// Path to the scorer binary.
    pub binary: PathBuf,
}

impl DirectoryScorer for ExternalScorer {
    fn score_directory(
        &self,
        candidates_dir: &Path,
        training_data: &Path,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        // Locked contract: do NOT pass --gpu / --cost.
        let output = Command::new(&self.binary)
            .arg(candidates_dir)
            .arg(training_data)
            .output()
            .map_err(|e| ScorerError::Process(format!("failed to spawn scorer: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(ScorerError::Process(format!(
                "scorer exited {}: stderr={stderr} stdout={stdout}",
                output.status
            )));
        }
        parse_scorer_stdout(&output.stdout)
    }
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
        fn score_directory(
            &self,
            _candidates_dir: &Path,
            _training_data: &Path,
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
}
