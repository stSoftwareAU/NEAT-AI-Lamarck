//! Remembered full-corpus baseline score (issue #113).
//!
//! Every promote call used to carry the incumbent so the scorer could re-derive
//! a number the run already knew: the incumbent's full-corpus score is
//! established by the Phase-0 parity gate and by the last accept, and between
//! accepts the incumbent does not change. Re-scoring it is ≈20% of a promote
//! call's creature-scores, and promote is the expensive tier.
//!
//! Remembering it removes a guard as well as work — a paired promote call is
//! self-verifying, because candidate and baseline are scored by the same binary
//! on the same corpus in the same process. This module carries the remembered
//! score together with everything that could invalidate it, so a stale value
//! can never decide an accept:
//!
//! * the **creature** — coarse `incumbentId` *and* a content fingerprint, since
//!   a weight-only accept leaves the shape id untouched, and
//! * the **training data** — path plus the size/mtime of every `*.bin` file, so
//!   a corpus that changed under the run (a documented event in
//!   `docs/baseline-economics.md`) invalidates it.

use crate::parity::PHASE0_ERROR_ABS_TOL;
use crate::scorer::ScoreResult;
use neat_core::find_bin_files;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

/// Floor for auto-tuned baseline drift tolerance.
///
/// Matches [`PHASE0_ERROR_ABS_TOL`]. Callers that pass an explicit
/// `--baseline-drift-epsilon` bypass auto-tuning entirely.
pub const DEFAULT_BASELINE_DRIFT_EPSILON: f64 = PHASE0_ERROR_ABS_TOL;

/// Cap for auto-tuned baseline drift tolerance.
///
/// Well above directory-scorer association noise on production corpora, still
/// small enough that a genuine mid-run corpus/scorer swap trips the canary.
/// Accept safety does **not** rest on this value — accepts off a remembered
/// baseline are always re-decided against a freshly scored pair.
pub const MAX_AUTO_BASELINE_DRIFT_EPSILON: f64 = 1e-3;

/// Headroom on the `f32` association noise model used by
/// [`tuned_baseline_drift_epsilon`].
///
/// Directory-mode scoring re-associates `f32` activations across parallel /
/// SIMD partitions between calls (NEAT-AI-scorer `multi_score.rs`). The raw
/// model is `ε_f32 · error · log₂(N)`; this factor covers host-to-host
/// variation. GRQ-10 (~2.26M records) observed `|Δ| ≈ 2.1e-8` under the old
/// fixed `1e-9` default.
const ASSOCIATION_HEADROOM: f64 = 64.0;

/// Auto-tune the absolute score drift tolerated on a baseline re-verification.
///
/// Models directory-scorer `f32` association noise over a mean of `records`
/// losses at typical magnitude `error_scale`, then clamps into
/// [`DEFAULT_BASELINE_DRIFT_EPSILON`] … [`MAX_AUTO_BASELINE_DRIFT_EPSILON`].
/// Lamarck owns this — hosts (e.g. GRQ) must not ship a competing override.
pub fn tuned_baseline_drift_epsilon(records: u64, error_scale: f64) -> f64 {
    let scale = if error_scale.is_finite() {
        error_scale.abs().clamp(1e-6, 2.0)
    } else {
        1.0
    };
    let n = records.max(2) as f64;
    let assoc = (f32::EPSILON as f64) * scale * n.log2() * ASSOCIATION_HEADROOM;
    assoc.clamp(
        DEFAULT_BASELINE_DRIFT_EPSILON,
        MAX_AUTO_BASELINE_DRIFT_EPSILON,
    )
}

/// Resolve the drift epsilon: an explicit configure wins; otherwise auto-tune.
pub fn resolve_baseline_drift_epsilon(
    configured: Option<f64>,
    records: u64,
    error_scale: f64,
) -> Result<f64, String> {
    match configured {
        Some(epsilon) => {
            if !epsilon.is_finite() || epsilon < 0.0 {
                return Err(format!(
                    "--baseline-drift-epsilon must be finite and at least 0 (got {epsilon})"
                ));
            }
            Ok(epsilon)
        }
        None => Ok(tuned_baseline_drift_epsilon(records, error_scale)),
    }
}

/// Estimate how many `f32` training records live under `training_data`.
pub fn estimate_corpus_records(
    training_data: &Path,
    bytes_per_record: usize,
) -> Result<u64, String> {
    if bytes_per_record == 0 {
        return Err("bytes_per_record must be > 0".to_string());
    }
    let files = find_bin_files(training_data).map_err(|e| e.to_string())?;
    let mut total_bytes = 0u64;
    for path in files {
        total_bytes = total_bytes.saturating_add(
            std::fs::metadata(&path)
                .map_err(|e| format!("failed to stat {}: {e}", path.display()))?
                .len(),
        );
    }
    Ok(total_bytes / bytes_per_record as u64)
}

/// Which baseline decided one promote call (issue #113).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BaselineSource {
    /// The promote call carried the incumbent and scored it from scratch.
    Fresh,
    /// The promote call omitted the incumbent and reused a remembered score.
    Remembered,
    /// The promote call reused a remembered score **and** proposed an accept,
    /// so the winner and the incumbent were re-scored together, full corpus,
    /// before the incumbent was swapped.
    RememberedVerified,
}

impl BaselineSource {
    /// Journal / log label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Fresh => "fresh",
            Self::Remembered => "remembered",
            Self::RememberedVerified => "rememberedVerified",
        }
    }

    /// True when the promote call itself omitted the incumbent.
    pub const fn omitted_baseline(self) -> bool {
        matches!(self, Self::Remembered | Self::RememberedVerified)
    }
}

/// Identity a remembered baseline score is valid for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineKey {
    /// Coarse journal `incumbentId` (shape only — never the whole key).
    pub incumbent_id: String,
    /// Content fingerprint of the incumbent creature.
    pub fingerprint: u64,
    /// Fingerprint of the training corpus the score was measured over.
    pub training: u64,
}

/// Fingerprint of the training corpus behind a full-corpus score.
///
/// Hashes the directory path and every `*.bin` entry's name, length and
/// modification time. A corpus that gained, lost or rewrote a file produces a
/// different value, which invalidates any score measured over the old one.
///
/// An unreadable directory is an error rather than a constant: a key that
/// silently collapsed to the same value for every corpus would hand the run a
/// remembered score for training data it never measured.
pub fn training_data_key(training_data: &Path) -> Result<u64, String> {
    let mut entries: Vec<(String, u64, u64)> = Vec::new();
    let listing = std::fs::read_dir(training_data).map_err(|e| {
        format!(
            "failed to list training data {}: {e}",
            training_data.display()
        )
    })?;
    for entry in listing {
        let entry = entry.map_err(|e| {
            format!(
                "failed to read training data {}: {e}",
                training_data.display()
            )
        })?;
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "bin") {
            continue;
        }
        let meta = entry
            .metadata()
            .map_err(|e| format!("failed to stat {}: {e}", path.display()))?;
        let modified = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos() as u64);
        entries.push((
            entry.file_name().to_string_lossy().to_string(),
            meta.len(),
            modified,
        ));
    }
    // Directory order is not defined, so sort before hashing.
    entries.sort();
    let mut hasher = DefaultHasher::new();
    training_data.to_string_lossy().hash(&mut hasher);
    entries.hash(&mut hasher);
    Ok(hasher.finish())
}

/// The authoritative full-corpus baseline score the run is carrying.
///
/// Created from a score the scorer actually produced — never from a default —
/// and only ever handed back for a creature and corpus that still match the key
/// it was created with.
#[derive(Debug, Clone)]
pub struct RememberedBaseline {
    key: BaselineKey,
    result: ScoreResult,
    calls_since_verify: u64,
}

impl RememberedBaseline {
    /// Remember `result` as the full-corpus score of the creature in `key`.
    pub fn new(key: BaselineKey, result: ScoreResult) -> Self {
        Self {
            key,
            result,
            calls_since_verify: 0,
        }
    }

    /// The remembered score.
    pub fn result(&self) -> &ScoreResult {
        &self.result
    }

    /// Promote calls served from this score since it was last scored fresh.
    pub fn calls_since_verify(&self) -> u64 {
        self.calls_since_verify
    }

    /// True when this score still describes `key`'s creature and corpus.
    pub fn is_valid_for(&self, key: &BaselineKey) -> bool {
        &self.key == key
    }

    /// Count one promote call served from this remembered score.
    pub fn note_reuse(&mut self) {
        self.calls_since_verify = self.calls_since_verify.saturating_add(1);
    }

    /// Absolute distance between the remembered score and a fresh one.
    pub fn drift(&self, fresh: &ScoreResult) -> f64 {
        (fresh.score - self.result.score).abs()
    }

    /// Whether a promote call may omit the baseline under `policy`.
    ///
    /// `false` once the re-verification interval is due, so the next promote
    /// call scores the incumbent again and the drift check has something to
    /// compare against.
    pub fn may_serve(&self, key: &BaselineKey, policy: BaselineReusePolicy) -> bool {
        policy.is_enabled() && self.is_valid_for(key) && self.calls_since_verify < policy.interval
    }
}

/// How aggressively a run reuses its remembered baseline (issue #113).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaselineReusePolicy {
    /// Promote calls served from a remembered score before one must be fresh.
    /// `0` disables reuse — every promote call carries the incumbent.
    pub interval: u64,
    /// Absolute score drift that aborts the run on a re-verification.
    pub drift_epsilon: f64,
}

impl BaselineReusePolicy {
    /// Reuse disabled — the pre-#113 run, where every promote call is paired.
    pub const fn off() -> Self {
        Self {
            interval: 0,
            drift_epsilon: DEFAULT_BASELINE_DRIFT_EPSILON,
        }
    }

    /// True when promote calls may omit the baseline at all.
    pub const fn is_enabled(self) -> bool {
        self.interval > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn score(value: f64) -> ScoreResult {
        ScoreResult {
            score: value,
            error: 1.0 - value,
            complexity_penalty: 0.0,
        }
    }

    fn key(incumbent: &str, fingerprint: u64, training: u64) -> BaselineKey {
        BaselineKey {
            incumbent_id: incumbent.to_string(),
            fingerprint,
            training,
        }
    }

    #[test]
    fn a_remembered_score_serves_only_its_own_creature_and_corpus() {
        let remembered = RememberedBaseline::new(key("in1-out1-n2-s2", 7, 99), score(0.5));
        assert!(remembered.is_valid_for(&key("in1-out1-n2-s2", 7, 99)));
        // A weight-only accept leaves the coarse shape id untouched, so the
        // content fingerprint is what actually guards reuse.
        assert!(!remembered.is_valid_for(&key("in1-out1-n2-s2", 8, 99)));
        assert!(!remembered.is_valid_for(&key("in1-out1-n3-s4", 7, 99)));
        // Training data changed under the run.
        assert!(!remembered.is_valid_for(&key("in1-out1-n2-s2", 7, 100)));
    }

    #[test]
    fn reuse_is_off_by_default_and_bounded_by_the_interval() {
        let scope = key("in1-out1-n2-s2", 7, 99);
        let mut remembered = RememberedBaseline::new(scope.clone(), score(0.5));
        assert!(
            !remembered.may_serve(&scope, BaselineReusePolicy::off()),
            "the pre-#113 run pairs every promote call"
        );

        let policy = BaselineReusePolicy {
            interval: 2,
            drift_epsilon: DEFAULT_BASELINE_DRIFT_EPSILON,
        };
        assert!(remembered.may_serve(&scope, policy));
        remembered.note_reuse();
        assert!(remembered.may_serve(&scope, policy));
        remembered.note_reuse();
        assert_eq!(remembered.calls_since_verify(), 2);
        assert!(
            !remembered.may_serve(&scope, policy),
            "the interval is due — the next promote call must score the baseline"
        );
    }

    #[test]
    fn drift_is_the_absolute_distance_from_a_fresh_score() {
        let remembered = RememberedBaseline::new(key("c", 1, 1), score(0.5));
        assert!(remembered.drift(&score(0.5)) < f64::EPSILON);
        assert!((remembered.drift(&score(0.5 + 3e-6)) - 3e-6).abs() < 1e-15);
        assert!((remembered.drift(&score(0.5 - 3e-6)) - 3e-6).abs() < 1e-15);
    }

    /// The auto floor must stay at the Phase-0 absolute error floor — not a
    /// last-bit constant. GRQ-10 aborted on `|Δ| ≈ 2.1e-8` under `1e-9`.
    #[test]
    fn auto_drift_epsilon_covers_production_association_noise() {
        assert_eq!(DEFAULT_BASELINE_DRIFT_EPSILON, PHASE0_ERROR_ABS_TOL);
        // GRQ-10 shape: ~2.26M records, error ≈ 0.65, observed |Δ| ≈ 2.1e-8.
        let tuned = tuned_baseline_drift_epsilon(2_262_277, 0.65);
        assert!(
            tuned > 2.1e-8 * 10.0,
            "auto epsilon {tuned} must clear observed GRQ-10 noise with margin"
        );
        assert!(tuned <= MAX_AUTO_BASELINE_DRIFT_EPSILON);
        // Tiny corpora still get at least the Phase-0 floor.
        assert_eq!(
            tuned_baseline_drift_epsilon(2, 0.1),
            DEFAULT_BASELINE_DRIFT_EPSILON
        );
        // Explicit configure wins; auto is not mixed in.
        assert_eq!(
            resolve_baseline_drift_epsilon(Some(1e-9), 2_262_277, 0.65).unwrap(),
            1e-9
        );
        assert_eq!(
            resolve_baseline_drift_epsilon(None, 2_262_277, 0.65).unwrap(),
            tuned
        );
    }

    #[test]
    fn the_training_key_changes_when_the_corpus_does() {
        let dir = tempfile::tempdir().unwrap();
        let training = dir.path().join("train");
        fs::create_dir_all(&training).unwrap();
        fs::write(training.join("0.bin"), [1u8, 2, 3, 4]).unwrap();
        let first = training_data_key(&training).unwrap();
        assert_eq!(first, training_data_key(&training).unwrap(), "stable");

        // A file added — the corpus the score was measured over is gone.
        fs::write(training.join("1.bin"), [5u8, 6, 7, 8]).unwrap();
        let grown = training_data_key(&training).unwrap();
        assert_ne!(first, grown);

        // A file deleted — the GRQ `node.sh` event `docs/baseline-economics.md`
        // records — must invalidate too.
        fs::remove_file(training.join("1.bin")).unwrap();
        fs::remove_file(training.join("0.bin")).unwrap();
        assert_ne!(grown, training_data_key(&training).unwrap());
    }

    #[test]
    fn a_missing_training_directory_is_an_error_not_a_constant_key() {
        let err = training_data_key(Path::new("/nonexistent-lamarck-corpus")).unwrap_err();
        assert!(err.contains("training data"), "{err}");
    }

    #[test]
    fn the_source_labels_are_the_journal_ones() {
        assert_eq!(BaselineSource::Fresh.label(), "fresh");
        assert_eq!(BaselineSource::Remembered.label(), "remembered");
        assert_eq!(
            BaselineSource::RememberedVerified.label(),
            "rememberedVerified"
        );
        assert!(!BaselineSource::Fresh.omitted_baseline());
        assert!(BaselineSource::Remembered.omitted_baseline());
        assert!(BaselineSource::RememberedVerified.omitted_baseline());
    }
}
