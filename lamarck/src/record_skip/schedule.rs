//! Difficulty tracking and skip policies for the record-skip experiment.

use std::fmt;

use rand::Rng;
use rand::seq::index;

/// Identity of the parent creature and corpus a set of difficulty scores
/// belongs to. Scores are never reused across a different scope.
#[derive(Clone, PartialEq, Debug)]
pub struct DifficultyScope {
    /// Fingerprint of the (immutable) parent creature.
    pub parent_fingerprint: u64,
    /// Identity of the training corpus.
    pub corpus_key: String,
    /// Number of training records scored.
    pub record_count: usize,
}

/// Errors raised by the schedule layer.
#[derive(Clone, PartialEq, Debug)]
pub enum ScheduleError {
    /// A schedule configuration field is out of range.
    InvalidConfig(String),
    /// A skip reduction is outside the open interval (0, 1).
    InvalidReduction(f64),
    /// Difficulty scores were offered a different parent or corpus.
    ScopeMismatch {
        /// Scope the scores were recorded under.
        expected: DifficultyScope,
        /// Scope the caller asked to use them for.
        found: DifficultyScope,
    },
    /// The strata vector does not match the record count.
    StrataLength {
        /// Records in the scope.
        expected: usize,
        /// Strata supplied.
        found: usize,
    },
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(m) => write!(f, "invalid schedule config: {m}"),
            Self::InvalidReduction(r) => {
                write!(
                    f,
                    "invalid reduction {r}: must be in the open interval (0, 1)"
                )
            }
            Self::ScopeMismatch { expected, found } => write!(
                f,
                "difficulty scope mismatch: scores belong to parent {:016x} corpus '{}' ({} records), \
                 not parent {:016x} corpus '{}' ({} records)",
                expected.parent_fingerprint,
                expected.corpus_key,
                expected.record_count,
                found.parent_fingerprint,
                found.corpus_key,
                found.record_count
            ),
            Self::StrataLength { expected, found } => write!(
                f,
                "strata length {found} does not match record_count {expected}"
            ),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Tunables for the difficulty schedule.
#[derive(Clone, PartialEq, Debug)]
pub struct ScheduleConfig {
    /// EWMA smoothing factor, in (0, 1].
    pub ewma_alpha: f64,
    /// Epochs processed in full before any skipping, at least 1.
    pub warmup_epochs: u32,
    /// Share of the non-protected slots spent re-visiting easy records, in [0, 1).
    pub revisit_share: f64,
    /// Strata holding less than this share of records are protected, in [0, 1).
    pub rare_stratum_share: f64,
}

impl ScheduleConfig {
    /// Refuse out-of-range fields, naming the field.
    pub fn validate(&self) -> Result<(), ScheduleError> {
        if !(self.ewma_alpha > 0.0 && self.ewma_alpha <= 1.0) {
            return Err(ScheduleError::InvalidConfig(format!(
                "ewma_alpha {} must be in (0, 1]",
                self.ewma_alpha
            )));
        }
        if self.warmup_epochs < 1 {
            return Err(ScheduleError::InvalidConfig(
                "warmup_epochs must be at least 1".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.revisit_share) {
            return Err(ScheduleError::InvalidConfig(format!(
                "revisit_share {} must be in [0, 1)",
                self.revisit_share
            )));
        }
        if !(0.0..1.0).contains(&self.rare_stratum_share) {
            return Err(ScheduleError::InvalidConfig(format!(
                "rare_stratum_share {} must be in [0, 1)",
                self.rare_stratum_share
            )));
        }
        Ok(())
    }
}

/// How records are chosen for an epoch.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SkipPolicy {
    /// Process every record every epoch.
    Full,
    /// Skip the easiest records, keeping a small random revisit share.
    EasySkip {
        /// Fraction of records skipped after warm-up, in (0, 1).
        reduction: f64,
    },
    /// Skip the same number of records, chosen uniformly at random.
    RandomSkip {
        /// Fraction of records skipped after warm-up, in (0, 1).
        reduction: f64,
    },
}

impl SkipPolicy {
    /// Refuse a reduction outside (0, 1).
    pub fn validate(&self) -> Result<(), ScheduleError> {
        match self {
            Self::Full => Ok(()),
            Self::EasySkip { reduction } | Self::RandomSkip { reduction } => {
                if *reduction > 0.0 && *reduction < 1.0 {
                    Ok(())
                } else {
                    Err(ScheduleError::InvalidReduction(*reduction))
                }
            }
        }
    }
}

/// Equal-width stratum boundaries over a set of values.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct StratumEdges {
    min: f64,
    max: f64,
    bins: usize,
}

impl StratumEdges {
    /// Derive edges from `values` (min and max), with at least one bin.
    pub fn from_values(values: &[f32], bins: usize) -> Self {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &v in values {
            let v = f64::from(v);
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        if !min.is_finite() || !max.is_finite() {
            min = 0.0;
            max = 0.0;
        }
        Self {
            min,
            max,
            bins: bins.max(1),
        }
    }

    /// Bin for `value`; out-of-range values clamp to the end bins.
    pub fn assign(&self, value: f32) -> u16 {
        let v = f64::from(value);
        if self.max <= self.min || v.is_nan() {
            return 0;
        }
        let pos = (v - self.min) / (self.max - self.min) * self.bins as f64;
        if pos <= 0.0 {
            0
        } else {
            (pos.floor() as usize).min(self.bins - 1) as u16
        }
    }
}

/// Application-agnostic strata: the equal-width bin of each record's first
/// target value. All-equal values land in bin 0.
pub fn output_strata(first_outputs: &[f32], bins: usize) -> Vec<u16> {
    let edges = StratumEdges::from_values(first_outputs, bins);
    first_outputs.iter().map(|&v| edges.assign(v)).collect()
}

/// Per-stratum flag (indexed by stratum id) that is true when the stratum's
/// share of all records is below `rare_share`.
pub fn rare_flags(strata: &[u16], rare_share: f64) -> Vec<bool> {
    let n = strata.len();
    let Some(&max) = strata.iter().max() else {
        return Vec::new();
    };
    let mut counts = vec![0usize; usize::from(max) + 1];
    for &s in strata {
        counts[usize::from(s)] += 1;
    }
    counts
        .iter()
        .map(|&c| c > 0 && (c as f64 / n as f64) < rare_share)
        .collect()
}

/// The records chosen for one epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    /// Per-record keep flag.
    pub keep: Vec<bool>,
    /// Records kept.
    pub processed: usize,
    /// Protected (rare-stratum) records in the corpus.
    pub protected: usize,
    /// Kept records drawn from the easy pool as revisits.
    pub revisits: usize,
    /// Scheduling operations (candidate reads plus sort comparisons).
    pub schedule_ops: u64,
}

/// Per-record difficulty state for one parent and corpus.
#[derive(Clone, Debug)]
pub struct DifficultyTracker {
    scope: DifficultyScope,
    config: ScheduleConfig,
    strata: Vec<u16>,
    ewma: Vec<Option<f64>>,
    last_seen: Vec<Option<f64>>,
    protected: Vec<bool>,
}

impl DifficultyTracker {
    /// Build a tracker; `strata.len()` must equal `scope.record_count`.
    pub fn new(
        scope: DifficultyScope,
        strata: Vec<u16>,
        config: ScheduleConfig,
    ) -> Result<Self, ScheduleError> {
        config.validate()?;
        if strata.len() != scope.record_count {
            return Err(ScheduleError::StrataLength {
                expected: scope.record_count,
                found: strata.len(),
            });
        }
        let flags = rare_flags(&strata, config.rare_stratum_share);
        let protected = strata.iter().map(|&s| flags[usize::from(s)]).collect();
        let n = strata.len();
        Ok(Self {
            scope,
            config,
            strata,
            ewma: vec![None; n],
            last_seen: vec![None; n],
            protected,
        })
    }

    /// Refuse reuse of these scores for a different parent or corpus.
    pub fn ensure_scope(&self, scope: &DifficultyScope) -> Result<(), ScheduleError> {
        if &self.scope == scope {
            Ok(())
        } else {
            Err(ScheduleError::ScopeMismatch {
                expected: self.scope.clone(),
                found: scope.clone(),
            })
        }
    }

    /// Record the squared error seen for `record` this epoch.
    pub fn observe(&mut self, record: usize, error: f64) {
        self.ewma[record] = Some(match self.ewma[record] {
            None => error,
            Some(prev) => self.config.ewma_alpha * error + (1.0 - self.config.ewma_alpha) * prev,
        });
        self.last_seen[record] = Some(error);
    }

    /// Smoothed difficulty of record `i`, if observed.
    pub fn ewma(&self, i: usize) -> Option<f64> {
        self.ewma[i]
    }

    /// Last observed squared error of record `i`.
    pub fn last_seen(&self, i: usize) -> Option<f64> {
        self.last_seen[i]
    }

    /// Records in protected (rare) strata.
    pub fn protected_count(&self) -> usize {
        self.protected.iter().filter(|&&p| p).count()
    }

    /// Choose the records to process in `epoch`.
    pub fn select(
        &self,
        epoch: u32,
        policy: &SkipPolicy,
        rng: &mut impl Rng,
    ) -> Result<Selection, ScheduleError> {
        policy.validate()?;
        let n = self.strata.len();
        let protected_total = self.protected_count();
        let keep_all = || Selection {
            keep: vec![true; n],
            processed: n,
            protected: protected_total,
            revisits: 0,
            schedule_ops: 0,
        };
        let reduction = match policy {
            SkipPolicy::Full => return Ok(keep_all()),
            SkipPolicy::EasySkip { reduction } | SkipPolicy::RandomSkip { reduction } => *reduction,
        };
        if epoch < self.config.warmup_epochs || self.ewma.iter().any(Option::is_none) {
            return Ok(keep_all());
        }

        let skipped = ((reduction * n as f64).round() as usize).min(n);
        let target = n - skipped;
        let mut keep: Vec<bool> = self.protected.clone();
        let candidates: Vec<usize> = (0..n).filter(|&i| !self.protected[i]).collect();
        let slots = target.saturating_sub(protected_total);
        let mut revisits = 0usize;
        let mut ops = 0u64;

        if slots > 0 {
            match policy {
                SkipPolicy::EasySkip { .. } => {
                    let revisit =
                        ((self.config.revisit_share * slots as f64).round() as usize).min(slots);
                    let mut order = candidates.clone();
                    ops += order.len() as u64;
                    order.sort_by(|&a, &b| {
                        ops += 1;
                        let ea = self.ewma[a].unwrap_or(0.0);
                        let eb = self.ewma[b].unwrap_or(0.0);
                        eb.total_cmp(&ea).then(a.cmp(&b))
                    });
                    let hard = slots - revisit;
                    for &i in &order[..hard] {
                        keep[i] = true;
                    }
                    let pool = &order[hard..];
                    for k in index::sample(rng, pool.len(), revisit).iter() {
                        keep[pool[k]] = true;
                    }
                    revisits = revisit;
                }
                _ => {
                    for k in index::sample(rng, candidates.len(), slots).iter() {
                        keep[candidates[k]] = true;
                    }
                }
            }
        }

        let processed = keep.iter().filter(|&&k| k).count();
        Ok(Selection {
            keep,
            processed,
            protected: protected_total,
            revisits,
            schedule_ops: ops,
        })
    }
}
