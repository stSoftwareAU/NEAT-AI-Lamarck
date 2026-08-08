//! Versioned human-readable `observations.statistics` cache.

use neat_core::{TrainingDataConfig, TrainingDataIterator, find_bin_files};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Current on-disk format semver.
pub const FORMAT_VERSION: &str = "1.0.0";

/// Current statistics algorithm semver.
pub const ALGORITHM_VERSION: &str = "1.0.0";

/// Filename written into the training-data directory.
pub const STATISTICS_FILENAME: &str = "observations.statistics";

/// Per-observation (or per-target) streaming statistics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ScalarStats {
    /// Sample count.
    pub count: u64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Population variance.
    pub variance: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Minimum observed value.
    pub min: f64,
    /// Maximum observed value.
    pub max: f64,
    /// Count of exact zeros.
    pub zero_count: u64,
    /// Count of non-zero finite values.
    pub non_zero_count: u64,
    /// Count of NaN / Inf values.
    pub non_finite_count: u64,
    /// Mean absolute value.
    pub mean_abs: f64,
    /// Root-mean-square.
    pub rms: f64,
    /// Approximate quantiles: 1%, 5%, 25%, 50%, 75%, 95%, 99%.
    pub quantiles: [f64; 7],
}

/// Full observations statistics document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ObservationsStatistics {
    /// Format semver.
    pub format_version: String,
    /// Algorithm semver.
    pub algorithm_version: String,
    /// Input observation count.
    pub input_count: usize,
    /// Output count.
    pub output_count: usize,
    /// Record count.
    pub record_count: u64,
    /// Deterministic corpus identity.
    pub corpus_identity: String,
    /// Unix creation timestamp (seconds).
    pub created_at_unix: u64,
    /// Per-input statistics.
    pub inputs: Vec<ScalarStats>,
    /// Per-output (target) statistics.
    pub outputs: Vec<ScalarStats>,
    /// Flattened upper-triangular input/input Pearson correlations (optional).
    pub input_correlations: Vec<f64>,
    /// Input/target Pearson correlations (inputs × outputs).
    pub input_target_correlations: Vec<f64>,
}

/// Errors while loading or generating statistics.
#[derive(Debug)]
pub enum ObservationsError {
    /// I/O failure.
    Io(std::io::Error),
    /// JSON parse/serialize failure.
    Json(serde_json::Error),
    /// Training-data failure.
    Training(String),
    /// Stale or unsupported cache.
    Stale(String),
}

impl std::fmt::Display for ObservationsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "observations I/O error: {e}"),
            Self::Json(e) => write!(f, "observations JSON error: {e}"),
            Self::Training(e) => write!(f, "observations training error: {e}"),
            Self::Stale(e) => write!(f, "observations stale: {e}"),
        }
    }
}

impl std::error::Error for ObservationsError {}

impl From<std::io::Error> for ObservationsError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for ObservationsError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

/// Online mean/variance accumulator (Welford).
#[derive(Debug, Clone)]
struct OnlineMoment {
    count: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    zero_count: u64,
    non_zero_count: u64,
    non_finite_count: u64,
    abs_sum: f64,
    sq_sum: f64,
    samples: Vec<f32>,
}

impl OnlineMoment {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            zero_count: 0,
            non_zero_count: 0,
            non_finite_count: 0,
            abs_sum: 0.0,
            sq_sum: 0.0,
            samples: Vec::new(),
        }
    }

    fn push(&mut self, value: f32) {
        let v = f64::from(value);
        if !v.is_finite() {
            self.non_finite_count += 1;
            return;
        }
        self.count += 1;
        let delta = v - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = v - self.mean;
        self.m2 += delta * delta2;
        self.min = self.min.min(v);
        self.max = self.max.max(v);
        if v == 0.0 {
            self.zero_count += 1;
        } else {
            self.non_zero_count += 1;
        }
        self.abs_sum += v.abs();
        self.sq_sum += v * v;
        // Reservoir-style subsample for quantile approximation (cap memory).
        if self.samples.len() < 10_000 {
            self.samples.push(value);
        } else {
            let idx = (self.count as usize) % self.samples.len();
            self.samples[idx] = value;
        }
    }

    fn finish(&self) -> ScalarStats {
        let variance = if self.count > 0 {
            self.m2 / self.count as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let mean_abs = if self.count > 0 {
            self.abs_sum / self.count as f64
        } else {
            0.0
        };
        let rms = if self.count > 0 {
            (self.sq_sum / self.count as f64).sqrt()
        } else {
            0.0
        };
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q = |p: f64| -> f64 {
            if sorted.is_empty() {
                return 0.0;
            }
            let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
            f64::from(sorted[idx.min(sorted.len() - 1)])
        };
        ScalarStats {
            count: self.count,
            mean: self.mean,
            variance,
            std_dev,
            min: if self.count > 0 { self.min } else { 0.0 },
            max: if self.count > 0 { self.max } else { 0.0 },
            zero_count: self.zero_count,
            non_zero_count: self.non_zero_count,
            non_finite_count: self.non_finite_count,
            mean_abs,
            rms,
            quantiles: [
                q(0.01),
                q(0.05),
                q(0.25),
                q(0.50),
                q(0.75),
                q(0.95),
                q(0.99),
            ],
        }
    }
}

/// Compute a deterministic corpus identity from `.bin` file metadata + content mix.
pub fn corpus_identity(
    dir: &Path,
    config: &TrainingDataConfig,
) -> Result<String, ObservationsError> {
    let files = find_bin_files(dir).map_err(|e| ObservationsError::Training(e.to_string()))?;
    let mut state: u64 = 0xcbf2_9ce4_8422_2325;
    let mix = |state: &mut u64, v: u64| {
        *state ^= v;
        *state = state.wrapping_mul(0x100_0000_01b3);
    };
    mix(&mut state, config.num_inputs as u64);
    mix(&mut state, config.num_outputs as u64);
    for path in &files {
        let meta = fs::metadata(path)?;
        mix(&mut state, meta.len());
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .as_bytes();
        for b in name {
            mix(&mut state, u64::from(*b));
        }
        // Mix a small content sample from head/tail for stronger identity.
        let mut file = fs::File::open(path)?;
        let mut buf = [0u8; 64];
        let n = file.read(&mut buf)?;
        for b in &buf[..n] {
            mix(&mut state, u64::from(*b));
        }
        if meta.len() > 64 {
            use std::io::Seek;
            file.seek(std::io::SeekFrom::End(-64))?;
            let n = file.read(&mut buf)?;
            for b in &buf[..n] {
                mix(&mut state, u64::from(*b));
            }
        }
    }
    Ok(format!("{state:016x}"))
}

/// Path to the statistics file inside a training-data directory.
pub fn statistics_path(training_data: &Path) -> PathBuf {
    training_data.join(STATISTICS_FILENAME)
}

/// Load statistics JSON from disk.
pub fn load_statistics(path: &Path) -> Result<ObservationsStatistics, ObservationsError> {
    let text = fs::read_to_string(path)?;
    let stats: ObservationsStatistics = serde_json::from_str(&text)?;
    Ok(stats)
}

/// Validate that cached statistics match the current corpus and supported versions.
pub fn validate_statistics(
    stats: &ObservationsStatistics,
    training_data: &Path,
    config: &TrainingDataConfig,
) -> Result<(), ObservationsError> {
    if stats.format_version != FORMAT_VERSION {
        return Err(ObservationsError::Stale(format!(
            "unsupported format_version {} (want {FORMAT_VERSION})",
            stats.format_version
        )));
    }
    if stats.algorithm_version != ALGORITHM_VERSION {
        return Err(ObservationsError::Stale(format!(
            "unsupported algorithm_version {} (want {ALGORITHM_VERSION})",
            stats.algorithm_version
        )));
    }
    if stats.input_count != config.num_inputs || stats.output_count != config.num_outputs {
        return Err(ObservationsError::Stale(
            "input/output counts do not match creature/training config".into(),
        ));
    }
    let identity = corpus_identity(training_data, config)?;
    if stats.corpus_identity != identity {
        return Err(ObservationsError::Stale(format!(
            "corpus identity mismatch (cache {}, current {identity})",
            stats.corpus_identity
        )));
    }
    Ok(())
}

fn corpus_byte_total(training_data: &Path) -> Result<(usize, u64), ObservationsError> {
    let files =
        find_bin_files(training_data).map_err(|e| ObservationsError::Training(e.to_string()))?;
    let mut total = 0u64;
    for path in &files {
        total += fs::metadata(path)?.len();
    }
    Ok((files.len(), total))
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "?".into();
    }
    let secs = seconds.round() as u64;
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h{m:02}m{s:02}s")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}

/// Scan the corpus and write a fresh statistics file.
pub fn generate_statistics(
    training_data: &Path,
    config: &TrainingDataConfig,
) -> Result<ObservationsStatistics, ObservationsError> {
    let (file_count, total_bytes) = corpus_byte_total(training_data)?;
    let record_bytes = config.bytes_per_record() as u64;
    let approx_records = total_bytes.checked_div(record_bytes).unwrap_or(0);
    let pair_ops = (config.num_inputs as u64) * (config.num_inputs as u64 + 1) / 2;

    eprintln!("observations.statistics: generating cache");
    eprintln!(
        "  corpus: {} files, {}, ~{} records ({} inputs × {} outputs)",
        file_count,
        format_bytes(total_bytes),
        approx_records,
        config.num_inputs,
        config.num_outputs
    );
    eprintln!(
        "  note: each record updates ~{pair_ops} input×input correlation cells — much heavier than scoring"
    );
    eprintln!("  computing corpus identity...");
    let identity_started = Instant::now();
    let identity = corpus_identity(training_data, config)?;
    eprintln!(
        "  corpus identity {} ({:.1}s)",
        identity,
        identity_started.elapsed().as_secs_f64()
    );

    let mut input_acc: Vec<OnlineMoment> = (0..config.num_inputs)
        .map(|_| OnlineMoment::new())
        .collect();
    let mut output_acc: Vec<OnlineMoment> = (0..config.num_outputs)
        .map(|_| OnlineMoment::new())
        .collect();

    // Pairwise accumulators for Pearson (streaming covariance).
    let n_in = config.num_inputs;
    let n_out = config.num_outputs;
    let mut in_sum = vec![0.0f64; n_in];
    let mut out_sum = vec![0.0f64; n_out];
    let mut in_sq = vec![0.0f64; n_in];
    let mut out_sq = vec![0.0f64; n_out];
    let mut in_cross = vec![0.0f64; n_in * n_in];
    let mut in_out_cross = vec![0.0f64; n_in * n_out];
    let mut record_count = 0u64;
    let mut bytes_seen = 0u64;

    let scan_started = Instant::now();
    let mut last_report = Instant::now();
    let report_every = Duration::from_secs(2);

    eprintln!("  scanning training records...");
    let mut iter = TrainingDataIterator::new(training_data, config.clone())
        .map_err(|e| ObservationsError::Training(e.to_string()))?;
    while let Some(record) = iter
        .next_record()
        .map_err(|e| ObservationsError::Training(e.to_string()))?
    {
        record_count += 1;
        bytes_seen += record_bytes;
        for (i, &v) in record.inputs.iter().enumerate() {
            input_acc[i].push(v);
            let vf = f64::from(v);
            in_sum[i] += vf;
            in_sq[i] += vf * vf;
        }
        for (j, &v) in record.outputs.iter().enumerate() {
            output_acc[j].push(v);
            let vf = f64::from(v);
            out_sum[j] += vf;
            out_sq[j] += vf * vf;
        }
        for i in 0..n_in {
            let vi = f64::from(record.inputs[i]);
            for j in i..n_in {
                in_cross[i * n_in + j] += vi * f64::from(record.inputs[j]);
            }
            for j in 0..n_out {
                in_out_cross[i * n_out + j] += vi * f64::from(record.outputs[j]);
            }
        }

        if last_report.elapsed() >= report_every {
            let elapsed = scan_started.elapsed().as_secs_f64().max(1e-6);
            let rec_per_s = record_count as f64 / elapsed;
            let bytes_per_s = bytes_seen as f64 / elapsed;
            let pct = if total_bytes > 0 {
                (bytes_seen as f64 / total_bytes as f64) * 100.0
            } else {
                0.0
            };
            let eta = if bytes_per_s > 0.0 && total_bytes > bytes_seen {
                (total_bytes - bytes_seen) as f64 / bytes_per_s
            } else {
                0.0
            };
            eprintln!(
                "  progress: {pct:5.1}%  records={record_count} ({rec_per_s:.0}/s)  {}/{}  elapsed={}  eta={}",
                format_bytes(bytes_seen),
                format_bytes(total_bytes),
                format_duration(elapsed),
                format_duration(eta)
            );
            last_report = Instant::now();
        }
    }

    eprintln!(
        "  scan complete: {record_count} records in {} ({:.1} records/s)",
        format_duration(scan_started.elapsed().as_secs_f64()),
        record_count as f64 / scan_started.elapsed().as_secs_f64().max(1e-6)
    );
    eprintln!("  finalising correlations and writing JSON...");

    let n = record_count as f64;
    let pearson = |sum_a: f64, sum_b: f64, sum_ab: f64, sq_a: f64, sq_b: f64| -> f64 {
        if n <= 1.0 {
            return 0.0;
        }
        let cov = sum_ab - (sum_a * sum_b) / n;
        let va = sq_a - (sum_a * sum_a) / n;
        let vb = sq_b - (sum_b * sum_b) / n;
        let denom = (va * vb).sqrt();
        if denom <= f64::EPSILON {
            0.0
        } else {
            (cov / denom).clamp(-1.0, 1.0)
        }
    };

    let mut input_correlations = Vec::new();
    for i in 0..n_in {
        for j in i..n_in {
            input_correlations.push(pearson(
                in_sum[i],
                in_sum[j],
                in_cross[i * n_in + j],
                in_sq[i],
                in_sq[j],
            ));
        }
    }
    let mut input_target_correlations = Vec::with_capacity(n_in * n_out);
    for i in 0..n_in {
        for j in 0..n_out {
            input_target_correlations.push(pearson(
                in_sum[i],
                out_sum[j],
                in_out_cross[i * n_out + j],
                in_sq[i],
                out_sq[j],
            ));
        }
    }

    let created_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let stats = ObservationsStatistics {
        format_version: FORMAT_VERSION.into(),
        algorithm_version: ALGORITHM_VERSION.into(),
        input_count: n_in,
        output_count: n_out,
        record_count,
        corpus_identity: identity,
        created_at_unix,
        inputs: input_acc.iter().map(OnlineMoment::finish).collect(),
        outputs: output_acc.iter().map(OnlineMoment::finish).collect(),
        input_correlations,
        input_target_correlations,
    };

    let path = statistics_path(training_data);
    let json = serde_json::to_string_pretty(&stats)?;
    let mut file = fs::File::create(&path)?;
    file.write_all(json.as_bytes())?;
    eprintln!(
        "  wrote {} ({})",
        path.display(),
        format_bytes(json.len() as u64)
    );
    Ok(stats)
}

/// Load cached statistics or regenerate when missing/stale.
pub fn ensure_statistics(
    training_data: &Path,
    config: &TrainingDataConfig,
) -> Result<ObservationsStatistics, ObservationsError> {
    let path = statistics_path(training_data);
    if path.is_file() {
        eprintln!("observations.statistics: found cache at {}", path.display());
        match load_statistics(&path) {
            Ok(stats) => match validate_statistics(&stats, training_data, config) {
                Ok(()) => {
                    eprintln!(
                        "observations.statistics: reusing cache ({} records, identity {})",
                        stats.record_count, stats.corpus_identity
                    );
                    return Ok(stats);
                }
                Err(ObservationsError::Stale(reason)) => {
                    eprintln!("observations.statistics: cache stale — {reason}; regenerating");
                }
                Err(e) => return Err(e),
            },
            Err(ObservationsError::Json(e)) => {
                eprintln!("observations.statistics: cache corrupt ({e}); regenerating");
            }
            Err(e) => return Err(e),
        }
    } else {
        eprintln!(
            "observations.statistics: no cache at {} — generating (one-time, slow)",
            path.display()
        );
    }
    generate_statistics(training_data, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write_bin(dir: &Path, name: &str, values: &[f32]) {
        let path = dir.join(name);
        let mut f = fs::File::create(path).unwrap();
        for v in values {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn generate_and_reuse_statistics() {
        let dir = tempdir().unwrap();
        // 2 inputs + 1 output per record, 3 records.
        write_bin(
            dir.path(),
            "0.bin",
            &[
                1.0, 2.0, 0.5, //
                3.0, 4.0, 1.0, //
                5.0, 6.0, 1.5,
            ],
        );
        let config = TrainingDataConfig::new(2, 1);
        let stats = ensure_statistics(dir.path(), &config).unwrap();
        assert_eq!(stats.record_count, 3);
        assert_eq!(stats.inputs.len(), 2);
        assert!(nearly_mean(&stats.inputs[0], 3.0));

        let reused = ensure_statistics(dir.path(), &config).unwrap();
        assert_eq!(reused.corpus_identity, stats.corpus_identity);
        assert_eq!(reused.created_at_unix, stats.created_at_unix);
    }

    #[test]
    fn stale_identity_regenerates() {
        let dir = tempdir().unwrap();
        write_bin(dir.path(), "0.bin", &[1.0, 0.0, 0.0]);
        let config = TrainingDataConfig::new(2, 1);
        let first = ensure_statistics(dir.path(), &config).unwrap();
        write_bin(dir.path(), "0.bin", &[9.0, 0.0, 0.0]);
        let second = ensure_statistics(dir.path(), &config).unwrap();
        assert_ne!(first.corpus_identity, second.corpus_identity);
    }

    fn nearly_mean(stats: &ScalarStats, expected: f64) -> bool {
        (stats.mean - expected).abs() < 1e-9
    }
}
