//! Authoritative NEAT-AI-scorer integration.

use crate::config::DEFAULT_MIN_IMPROVEMENT;
use crate::log;
use crate::promote_gate::PromoteGate;
use crate::scorer_cost::{ScorerCallPhase, ScorerCallRecord};
use serde::Deserialize;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

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

/// A [`DirectoryScorer`] that measures every call it forwards (issue #112).
///
/// The wrapper sits between the run and the real scorer, so **every**
/// invocation is measured wherever it is made — Phase-0, Phase-G graft replay,
/// screen, promote and combo scoring alike — without threading a recorder
/// through each call site. Each call records the phase in force, the creature
/// count of the directory handed over, the sample rate and the wall clock, which
/// is what makes a run's `scorerMs` regressable into a fixed per-call cost and a
/// marginal per-creature cost.
pub struct RecordingScorer<'a, S: DirectoryScorer> {
    inner: &'a S,
    phase: Cell<ScorerCallPhase>,
    calls: RefCell<Vec<ScorerCallRecord>>,
    successes: Cell<u64>,
    failures: Cell<u64>,
}

impl<'a, S: DirectoryScorer> RecordingScorer<'a, S> {
    /// Wrap `inner`, attributing calls to Phase-0 until told otherwise.
    pub fn new(inner: &'a S) -> Self {
        Self {
            inner,
            phase: Cell::new(ScorerCallPhase::Phase0),
            calls: RefCell::new(Vec::new()),
            successes: Cell::new(0),
            failures: Cell::new(0),
        }
    }

    /// Attribute subsequent calls to `phase`.
    pub fn set_phase(&self, phase: ScorerCallPhase) {
        self.phase.set(phase);
    }

    /// Take the calls recorded since the last drain, ready to journal.
    pub fn drain(&self) -> Vec<ScorerCallRecord> {
        std::mem::take(&mut *self.calls.borrow_mut())
    }

    /// Calls recorded but not yet drained.
    pub fn pending(&self) -> usize {
        self.calls.borrow().len()
    }

    /// Scorer calls that succeeded, across every phase.
    pub fn successes(&self) -> u64 {
        self.successes.get()
    }

    /// Scorer calls that failed, across every phase.
    pub fn failures(&self) -> u64 {
        self.failures.get()
    }
}

/// Creature files (`*.json`) a scorer call was handed.
///
/// An unreadable directory is an error rather than a zero: a call silently
/// recorded as scoring nothing would drag the fitted fixed cost towards zero.
fn count_creature_files(dir: &Path) -> Result<u64, ScorerError> {
    let entries = fs::read_dir(dir).map_err(|e| {
        ScorerError::Process(format!(
            "failed to list scorer batch directory {}: {e}",
            dir.display()
        ))
    })?;
    let mut count = 0u64;
    for entry in entries {
        let entry = entry.map_err(|e| {
            ScorerError::Process(format!(
                "failed to read scorer batch directory {}: {e}",
                dir.display()
            ))
        })?;
        if entry.path().extension().is_some_and(|ext| ext == "json") {
            count += 1;
        }
    }
    Ok(count)
}

impl<S: DirectoryScorer> DirectoryScorer for RecordingScorer<'_, S> {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        training_data: &Path,
        sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let creatures = count_creature_files(candidates_dir)?;
        let started = Instant::now();
        let result = self
            .inner
            .score_directory_sampled(candidates_dir, training_data, sample);
        let elapsed_ms = started.elapsed().as_millis();
        let failed = result.is_err();
        if failed {
            self.failures.set(self.failures.get() + 1);
        } else {
            self.successes.set(self.successes.get() + 1);
        }
        self.calls.borrow_mut().push(ScorerCallRecord {
            phase: self.phase.get(),
            creatures,
            sample_rate: sample.is_subsample().then_some(sample.rate),
            elapsed_ms,
            failed,
        });
        result
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
    log_scorer_batch_stats_against(
        scores,
        scores.get("baseline"),
        scorer_ms,
        min_improvement,
        label,
    )
}

/// Like [`log_scorer_batch_stats_labeled`] with the baseline supplied.
///
/// A promote call that reused a remembered baseline (issue #113) has no
/// `baseline` stem of its own, so the score its deltas are measured against is
/// passed in. `None` still logs the missing-baseline warning rather than
/// reporting deltas against nothing.
pub fn log_scorer_batch_stats_against(
    scores: &BTreeMap<String, ScoreResult>,
    baseline: Option<&ScoreResult>,
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

    let Some(baseline) = baseline else {
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

/// What one promote gate did to one screened batch (issue #111).
#[derive(Debug, Clone, PartialEq)]
pub struct PromoteDecision {
    /// Stems admitted to full-corpus scoring, best sample Δ first.
    pub stems: Vec<String>,
    /// Candidates the screen tier scored (baseline excluded).
    pub screened: usize,
    /// Screen Δ a candidate had to clear in this batch.
    pub threshold: f64,
    /// σ̂ the gate estimated for this batch; `None` under the absolute gate or
    /// when the batch was too degenerate to price its own noise.
    pub sigma: Option<f64>,
}

/// Stems (excluding baseline) whose sample score beats baseline by more than `threshold`.
pub fn screen_promote_stems(
    scores: &BTreeMap<String, ScoreResult>,
    threshold: f64,
) -> Result<Vec<String>, ScorerError> {
    Ok(screen_promote_decision(scores, &PromoteGate::absolute(threshold))?.stems)
}

/// Apply a promote gate to a screened batch (issue #111).
///
/// The absolute gate reproduces [`screen_promote_stems`] exactly; the
/// noise-aware gate prices the batch's own spread first and can only ever
/// admit a subset of what the absolute one would.
///
/// The batch must carry its own `baseline` stem; use
/// [`screen_promote_decision_against`] when the baseline was scored elsewhere.
pub fn screen_promote_decision(
    scores: &BTreeMap<String, ScoreResult>,
    gate: &PromoteGate,
) -> Result<PromoteDecision, ScorerError> {
    let baseline = scores
        .get("baseline")
        .ok_or_else(|| ScorerError::Missing("baseline missing from scorer results".into()))?;
    Ok(screen_promote_decision_against(scores, baseline, gate))
}

/// Apply a promote gate against a baseline scored outside this batch (#113).
///
/// A promote call that reused a remembered baseline has no `baseline` stem in
/// its map at all, so the score to compare against is a parameter rather than a
/// lookup. A `baseline` stem, when present, is still never treated as a
/// candidate.
pub fn screen_promote_decision_against(
    scores: &BTreeMap<String, ScoreResult>,
    baseline: &ScoreResult,
    gate: &PromoteGate,
) -> PromoteDecision {
    let mut deltas: Vec<(String, f64)> = scores
        .iter()
        .filter(|(stem, _)| stem.as_str() != "baseline")
        .map(|(stem, r)| (stem.clone(), improvement(r.score, baseline.score)))
        .collect();
    let resolved = gate.threshold_for(&deltas.iter().map(|(_, d)| *d).collect::<Vec<_>>());
    let screened = deltas.len();
    deltas.retain(|(_, delta)| *delta > resolved.threshold);
    deltas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    PromoteDecision {
        stems: deltas.into_iter().map(|(s, _)| s).collect(),
        screened,
        threshold: resolved.threshold,
        sigma: resolved.sigma,
    }
}

/// Present `src` at `dst` without copying its bytes, falling back to a copy
/// (issue #114).
///
/// A promote directory exists only to show the scorer a smaller set of the
/// files the screen directory already holds, so a hard link is enough: same
/// inode, same bytes, no second write. Nothing mutates a batch file in place —
/// they are written once and the directory is deleted whole — so the shared
/// inode is safe.
///
/// Linking fails for reasons that are not the run's fault: a destination that
/// already exists, a `src` and `dst` on different filesystems (`EXDEV`), a
/// filesystem with no hard links at all. None of those is worth aborting an
/// experiment over, so the copy stands in and the caller cannot tell.
fn link_or_copy(src: &Path, dst: &Path) -> Result<(), String> {
    if fs::hard_link(src, dst).is_ok() {
        return Ok(());
    }
    fs::copy(src, dst)
        .map(|_| ())
        .map_err(|e| format!("copy {} to {} failed: {e}", src.display(), dst.display()))
}

/// Link (or copy) `baseline.json` + promoted candidate JSON into a fresh
/// promote directory.
pub fn write_promote_batch(
    promote_dir: &Path,
    source_batch: &Path,
    promote_stems: &[String],
) -> Result<(), String> {
    write_promote_batch_with(promote_dir, source_batch, promote_stems, true)
}

/// Write a promote batch of candidates only, for a run that already knows the
/// incumbent's full-corpus score (issue #113).
///
/// The scorer then spends its call on candidates alone — the baseline is ≈20%
/// of a promote call's creature-scores — and the caller supplies the score to
/// compare against.
pub fn write_promote_batch_without_baseline(
    promote_dir: &Path,
    source_batch: &Path,
    promote_stems: &[String],
) -> Result<(), String> {
    write_promote_batch_with(promote_dir, source_batch, promote_stems, false)
}

fn write_promote_batch_with(
    promote_dir: &Path,
    source_batch: &Path,
    promote_stems: &[String],
    include_baseline: bool,
) -> Result<(), String> {
    if promote_dir.exists() {
        fs::remove_dir_all(promote_dir).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(promote_dir).map_err(|e| e.to_string())?;
    if include_baseline {
        let baseline_src = source_batch.join("baseline.json");
        link_or_copy(&baseline_src, &promote_dir.join("baseline.json"))
            .map_err(|e| format!("baseline from {}: {e}", baseline_src.display()))?;
    }
    for stem in promote_stems {
        let name = format!("{stem}.json");
        let src = source_batch.join(&name);
        link_or_copy(&src, &promote_dir.join(&name))
            .map_err(|e| format!("{stem} from {}: {e}", src.display()))?;
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
///
/// The batch must carry its own `baseline` stem; use [`select_winner_against`]
/// when the baseline was scored elsewhere.
pub fn select_winner(
    results: &BTreeMap<String, ScoreResult>,
    min_improvement: f64,
) -> Result<Option<(&str, &ScoreResult, f64)>, ScorerError> {
    let baseline = results
        .get("baseline")
        .ok_or_else(|| ScorerError::Missing("baseline missing from scorer results".into()))?;
    Ok(select_winner_against(results, baseline, min_improvement))
}

/// Select the best qualifying candidate against a baseline scored elsewhere.
///
/// The baseline is a parameter, so a promote call that omitted it (issue #113)
/// is decided by exactly the same rule as a paired one. A `baseline` stem, when
/// present, is still never a candidate for the win.
pub fn select_winner_against<'a>(
    results: &'a BTreeMap<String, ScoreResult>,
    baseline: &ScoreResult,
    min_improvement: f64,
) -> Option<(&'a str, &'a ScoreResult, f64)> {
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
    best
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
        let (winner_stem, winner_delta) = {
            let winner = select_winner(&map, 1e-6).unwrap().unwrap();
            (winner.0.to_string(), winner.2)
        };
        assert_eq!(winner_stem, "candidate-001");
        assert!(winner_delta > 1e-6);

        // Issue #113: the same batch with the baseline supplied rather than
        // present picks the same winner with the same delta. A map with no
        // `baseline` key is the shape a remembered-baseline promote call
        // returns, and it must not be read as a score of 0 — which would
        // promote everything.
        let baseline = map.remove("baseline").expect("removed for this case");
        let parameterised = select_winner_against(&map, &baseline, 1e-6).expect("still a winner");
        assert_eq!(parameterised.0, winner_stem);
        assert!((parameterised.2 - winner_delta).abs() < 1e-15);
        // The existing loud failure is preserved for callers that still require
        // the batch to carry its own baseline.
        assert!(matches!(
            select_winner(&map, 1e-6).unwrap_err(),
            ScorerError::Missing(_)
        ));
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

        // Issue #113: supplying the baseline instead of carrying it in the map
        // admits exactly the same stems, in the same order, and counts the same
        // screened total.
        let baseline = map.get("baseline").cloned().expect("present above");
        let mut without = map.clone();
        without.remove("baseline");
        let parameterised =
            screen_promote_decision_against(&without, &baseline, &PromoteGate::absolute(0.0));
        assert_eq!(parameterised.stems, stems);
        assert_eq!(parameterised.screened, 3);

        // Issue #111 default-drift guard: the absolute gate promotes exactly
        // the stems the pre-#111 gate promoted, in the same order.
        let decision = screen_promote_decision(&map, &PromoteGate::absolute(0.0)).unwrap();
        assert_eq!(decision.stems, stems);
        assert_eq!(decision.screened, 3);
        assert_eq!(decision.threshold, 0.0);
        assert_eq!(decision.sigma, None);
    }

    /// A batch built from one real improver and a core of sampling wobble: the
    /// noise-aware gate keeps the improver and drops the wobble the absolute
    /// gate would have bought full-corpus scores for.
    #[test]
    fn the_noise_aware_gate_drops_wobble_and_keeps_a_real_improver() {
        let mut map = BTreeMap::new();
        let insert = |map: &mut BTreeMap<String, ScoreResult>, stem: &str, delta: f64| {
            map.insert(
                stem.to_string(),
                ScoreResult {
                    score: 0.5 + delta,
                    error: 0.5,
                    complexity_penalty: 0.0,
                },
            );
        };
        insert(&mut map, "baseline", 0.0);
        for (index, delta) in [
            1.2e-6, -1.1e-6, 1.4e-6, -1.3e-6, 1.05e-6, -9e-7, 1.6e-6, -1.5e-6,
        ]
        .into_iter()
        .enumerate()
        {
            insert(&mut map, &format!("candidate-{index:03}"), delta);
        }
        insert(&mut map, "candidate-100", 4.7e-5);

        let absolute = screen_promote_decision(&map, &PromoteGate::absolute(1e-6)).unwrap();
        assert_eq!(absolute.stems.len(), 5, "the absolute gate buys the wobble");

        let noise_aware =
            screen_promote_decision(&map, &PromoteGate::noise_aware(1e-6, 3.0)).unwrap();
        assert_eq!(noise_aware.stems, vec!["candidate-100"]);
        assert_eq!(noise_aware.screened, 9);
        assert!(noise_aware.sigma.is_some_and(|s| s > 0.0));
        assert!(noise_aware.threshold > absolute.threshold);
    }

    #[test]
    fn a_promote_decision_without_a_baseline_fails_loudly() {
        let map = BTreeMap::from([(
            "candidate-000".to_string(),
            ScoreResult {
                score: 0.5,
                error: 0.5,
                complexity_penalty: 0.0,
            },
        )]);
        let err = screen_promote_decision(&map, &PromoteGate::noise_aware(1e-6, 3.0)).unwrap_err();
        assert!(matches!(err, ScorerError::Missing(_)));
    }

    /// A directory of `n` creatures is recorded as `n` creatures, under the
    /// phase in force, with the sample rate the call used.
    #[test]
    fn the_recording_scorer_records_creature_count_phase_and_sample_rate() {
        let dir = tempfile::tempdir().unwrap();
        let batch = dir.path().join("candidates-exp-1");
        fs::create_dir_all(&batch).unwrap();
        fs::write(batch.join("baseline.json"), b"{}").unwrap();
        for i in 0..4 {
            fs::write(batch.join(format!("candidate-{i:03}.json")), b"{}").unwrap();
        }
        // A non-creature file must not be counted as one.
        fs::write(batch.join("notes.txt"), b"ignored").unwrap();

        let mut payload = BTreeMap::new();
        payload.insert(
            "baseline".to_string(),
            ScoreResult {
                score: 0.5,
                error: 0.5,
                complexity_penalty: 0.0,
            },
        );
        let inner = FakeScorer {
            payload: Arc::new(Mutex::new(payload)),
        };
        let recorder = RecordingScorer::new(&inner);
        recorder.set_phase(ScorerCallPhase::Screen);
        recorder
            .score_directory_sampled(
                &batch,
                Path::new("train"),
                ScoreSample {
                    rate: 0.05,
                    phase: 3,
                },
            )
            .unwrap();
        recorder.set_phase(ScorerCallPhase::Promote);
        recorder
            .score_directory(&batch, Path::new("train"))
            .unwrap();

        let calls = recorder.drain();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].phase, ScorerCallPhase::Screen);
        assert_eq!(calls[0].creatures, 5);
        assert_eq!(calls[0].sample_rate, Some(0.05));
        assert!(!calls[0].failed);
        assert_eq!(calls[1].phase, ScorerCallPhase::Promote);
        assert_eq!(calls[1].creatures, 5);
        assert_eq!(calls[1].sample_rate, None, "a full-corpus call has no rate");
        assert_eq!(recorder.successes(), 2);
        assert_eq!(recorder.failures(), 0);
        // Draining hands the calls over exactly once.
        assert!(recorder.drain().is_empty());
        assert_eq!(recorder.pending(), 0);
    }

    /// A failed call is still recorded — a call that vanished from the journal
    /// would fit the cost model to a subset of the run (issue #112).
    #[test]
    fn the_recording_scorer_records_a_failed_call() {
        struct FailingScorer;
        impl DirectoryScorer for FailingScorer {
            fn score_directory_sampled(
                &self,
                _candidates_dir: &Path,
                _training_data: &Path,
                _sample: ScoreSample,
            ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
                Err(ScorerError::Process("boom".into()))
            }
        }
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("baseline.json"), b"{}").unwrap();
        let recorder = RecordingScorer::new(&FailingScorer);
        recorder.set_phase(ScorerCallPhase::Promote);
        assert!(
            recorder
                .score_directory(dir.path(), Path::new("train"))
                .is_err()
        );
        let calls = recorder.drain();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].failed);
        assert_eq!(calls[0].creatures, 1);
        assert_eq!(recorder.failures(), 1);
        assert_eq!(recorder.successes(), 0);
    }

    /// An unlistable batch directory fails loudly instead of recording a call
    /// that scored zero creatures.
    #[test]
    fn a_missing_batch_directory_is_an_error_not_a_zero_creature_call() {
        let inner = FakeScorer {
            payload: Arc::new(Mutex::new(BTreeMap::new())),
        };
        let recorder = RecordingScorer::new(&inner);
        let err = recorder
            .score_directory(Path::new("/nonexistent-lamarck-batch"), Path::new("train"))
            .unwrap_err();
        assert!(matches!(err, ScorerError::Process(_)));
        assert!(recorder.drain().is_empty());
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
        write_promote_batch(&promote, &src, &["candidate-001".into()]).unwrap();
        assert!(promote.join("baseline.json").is_file());
        assert!(promote.join("candidate-001.json").is_file());
        assert!(!promote.join("candidate-000.json").exists());
    }

    /// Issue #113: the baseline-free batch carries the promoted candidates and
    /// nothing else, so the scorer spends the call on candidates alone.
    #[test]
    fn write_promote_batch_without_baseline_copies_only_the_stems() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("baseline.json"), b"{}").unwrap();
        fs::write(src.join("candidate-000.json"), b"{\"a\":1}").unwrap();
        let promote = dir.path().join("promote");
        write_promote_batch_without_baseline(&promote, &src, &["candidate-000".into()]).unwrap();
        assert!(!promote.join("baseline.json").exists());
        assert!(promote.join("candidate-000.json").is_file());
        assert_eq!(count_creature_files(&promote).unwrap(), 1);
    }

    /// Issue #114: a promote file is the screen file's bytes at a second path,
    /// so it is hard-linked — one inode, no second write.
    #[cfg(unix)]
    #[test]
    fn write_promote_batch_hard_links_rather_than_copying() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("baseline.json"), b"{\"baseline\":true}").unwrap();
        fs::write(src.join("candidate-000.json"), b"{\"a\":1}").unwrap();
        let promote = dir.path().join("promote");
        write_promote_batch(&promote, &src, &["candidate-000".into()]).unwrap();

        for name in ["baseline.json", "candidate-000.json"] {
            let source = fs::metadata(src.join(name)).unwrap();
            let linked = fs::metadata(promote.join(name)).unwrap();
            assert!(
                same_file(&source, &linked),
                "{name} must be hard-linked to the screen batch, not copied"
            );
            assert_eq!(
                fs::read(src.join(name)).unwrap(),
                fs::read(promote.join(name)).unwrap()
            );
        }
    }

    /// …and a link that cannot be made falls back to a copy rather than
    /// aborting the experiment (issue #114).
    #[cfg(unix)]
    #[test]
    fn link_or_copy_falls_back_to_a_copy_when_linking_fails() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("candidate-000.json");
        let dst = dir.path().join("promote-candidate-000.json");
        fs::write(&src, b"{\"a\":1}").unwrap();
        // An existing destination makes `hard_link` fail with AlreadyExists —
        // the same shape as EXDEV or a filesystem with no links.
        fs::write(&dst, b"stale").unwrap();

        link_or_copy(&src, &dst).expect("a failed link must fall back to a copy");

        assert_eq!(fs::read(&dst).unwrap(), b"{\"a\":1}");
        assert!(
            !same_file(&fs::metadata(&src).unwrap(), &fs::metadata(&dst).unwrap()),
            "the fallback is a copy, so the two files are separate inodes"
        );
    }

    /// A missing source still fails loudly — the fallback covers link failures,
    /// never a batch file that was never written.
    #[test]
    fn link_or_copy_fails_loudly_when_the_source_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let err = link_or_copy(
            &dir.path().join("absent.json"),
            &dir.path().join("promote.json"),
        )
        .unwrap_err();
        assert!(err.contains("absent.json"), "unhelpful error: {err}");
    }

    #[cfg(unix)]
    fn same_file(a: &fs::Metadata, b: &fs::Metadata) -> bool {
        use std::os::unix::fs::MetadataExt;
        a.dev() == b.dev() && a.ino() == b.ino()
    }
}
