//! The predeclared manifest of the record-skip experiment.

use serde::{Deserialize, Serialize};

/// One calibration grid point.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GridPoint {
    /// EWMA smoothing factor.
    pub ewma_alpha: f64,
    /// Revisit share of the non-protected slots.
    pub revisit_share: f64,
}

/// Everything fixed before the experiment runs.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    /// Manifest format version.
    pub version: u32,
    /// Refinement epochs per run.
    pub epochs: u32,
    /// Synthetic train records.
    pub train_records: usize,
    /// Synthetic validation records.
    pub validation_records: usize,
    /// Seed of the synthetic parent (same parent for every run).
    pub parent_seed: u64,
    /// Seeds used only to choose the schedule (training MSE only).
    pub calibration_seeds: Vec<u64>,
    /// Seeds used only for the final comparison.
    pub validation_seeds: Vec<u64>,
    /// Skip reductions to test.
    pub reductions: Vec<f64>,
    /// Bins for output strata.
    pub strata_bins: usize,
    /// Strata below this share of records are protected.
    pub rare_stratum_share: f64,
    /// Epochs processed in full before skipping.
    pub warmup_epochs: u32,
    /// Calibration grid.
    pub calibration_grid: Vec<GridPoint>,
    /// Fixed learning rate.
    pub learning_rate: f64,
    /// Bootstrap resamples.
    pub bootstrap_resamples: usize,
    /// Bootstrap seed.
    pub bootstrap_seed: u64,
    /// A record is forgotten when its error rises by more than this.
    pub forgetting_tolerance: f64,
}

fn grid() -> Vec<GridPoint> {
    let mut g = Vec::new();
    for ewma_alpha in [0.3, 0.7] {
        for revisit_share in [0.1, 0.25] {
            g.push(GridPoint {
                ewma_alpha,
                revisit_share,
            });
        }
    }
    g
}

impl Manifest {
    /// The predeclared full experiment.
    pub fn predeclared() -> Self {
        Self {
            version: 1,
            epochs: 30,
            train_records: 2000,
            validation_records: 1000,
            parent_seed: 267,
            calibration_seeds: (1..=6).collect(),
            validation_seeds: (1001..=1012).collect(),
            reductions: vec![0.25, 0.5, 0.75],
            strata_bins: 8,
            rare_stratum_share: 0.10,
            warmup_epochs: 1,
            calibration_grid: grid(),
            learning_rate: LEARNING_RATE,
            bootstrap_resamples: 2000,
            bootstrap_seed: 267,
            forgetting_tolerance: 1e-4,
        }
    }

    /// A small manifest for smoke runs and tests.
    pub fn smoke() -> Self {
        let mut g = grid();
        g.truncate(2);
        Self {
            epochs: 4,
            train_records: 200,
            validation_records: 100,
            calibration_seeds: vec![1, 2],
            validation_seeds: vec![1001, 1002],
            calibration_grid: g,
            bootstrap_resamples: 200,
            ..Self::predeclared()
        }
    }

    /// Refuse an inconsistent manifest, naming the field.
    pub fn validate(&self) -> Result<(), String> {
        if self.epochs == 0 {
            return Err("epochs must be at least 1".to_string());
        }
        if self.validation_records == 0 {
            return Err("validationRecords must be at least 1".to_string());
        }
        if self.train_records == 0 {
            return Err("trainRecords must be at least 1".to_string());
        }
        if self.calibration_seeds.is_empty() {
            return Err("calibrationSeeds must not be empty".to_string());
        }
        if self.validation_seeds.is_empty() {
            return Err("validationSeeds must not be empty".to_string());
        }
        if let Some(s) = self
            .calibration_seeds
            .iter()
            .find(|s| self.validation_seeds.contains(s))
        {
            return Err(format!(
                "calibrationSeeds and validationSeeds overlap on seed {s}"
            ));
        }
        if self.calibration_grid.is_empty() {
            return Err("calibrationGrid must not be empty".to_string());
        }
        if self.reductions.is_empty() {
            return Err("reductions must not be empty".to_string());
        }
        if let Some(r) = self.reductions.iter().find(|r| !(**r > 0.0 && **r < 1.0)) {
            return Err(format!("reductions contains {r}: must be in (0, 1)"));
        }
        if self.strata_bins == 0 {
            return Err("strataBins must be at least 1".to_string());
        }
        if self.bootstrap_resamples == 0 {
            return Err("bootstrapResamples must be at least 1".to_string());
        }
        if !(self.learning_rate > 0.0 && self.learning_rate.is_finite()) {
            return Err("learningRate must be positive and finite".to_string());
        }
        Ok(())
    }
}

/// Learning rate fixed before the predeclared run: on the synthetic suite the
/// full arm takes training MSE from 0.382 to 0.068 over 30 epochs (0.05 reaches
/// 0.096, 0.5 degrades to 0.077), so 0.2 is clearly converging without diverging.
pub const LEARNING_RATE: f64 = 0.2;
