//! Creature-scale budgets: size-derived limits and the run's resolved budgets
//! (issue #223).
//!
//! Lamarck optimises a creature that keeps evolving, so a limit calibrated
//! against one historical champion stops being right the moment the creature
//! grows. This module holds the two halves of the answer:
//!
//! * [`CreatureScale`] — the supplied creature's dimensions, read from the
//!   creature itself rather than assumed;
//! * [`ResolvedBudgets`] — the budgets the run actually used, resolved from
//!   those dimensions and the wall-clock budget, and journalled in the run
//!   header so a reader never has to infer them.
//!
//! No dimension of any particular creature is embedded here. The derivations
//! are *shapes* — a floor, a sublinear gain, a ceiling — evaluated against
//! whatever creature the run was handed. Measured dimensions of a real
//! production creature belong in `docs/scale-sensitivity.md` as benchmark
//! evidence, never in this file as behaviour.
//!
//! ## Which budgets derive, and which do not
//!
//! `docs/scale-sensitivity.md` carries the full inventory and classifies every
//! scale-sensitive constant in the crate as *dimensionless*, *measured* or
//! *size-dependent*. Only the size-dependent ones are derived here, and only
//! under [`ScaleBudgetMode::Derived`]; [`ScaleBudgetMode::Fixed`] reproduces
//! the pre-#223 literals exactly so the two are an A/B pair.

use std::time::Duration;

use neat_core::CreatureExport;

use crate::config::{DEFAULT_FOCUS_COUNT, LamarckConfig};
use crate::grafts::default_graft_replay_budget;

/// Head of the target-correlation ranking re-scored by residual correlation
/// under [`ScaleBudgetMode::Fixed`] (the pre-#223 literal).
pub const FIXED_RESIDUAL_SHORTLIST: usize = 48;

/// Extra unused hidden sources always folded into the residual refine under
/// [`ScaleBudgetMode::Fixed`] (the pre-#223 literal).
pub const FIXED_RESIDUAL_HIDDEN_EXTRA: usize = 16;

/// Synthetic probe rows drawn when the corpus cannot supply two records, under
/// [`ScaleBudgetMode::Fixed`] (the pre-#223 literal).
pub const FIXED_SYNTHETIC_PROBE_ROWS: usize = 64;

/// Sublinear gain applied to the ranked-source population for the residual
/// shortlist head.
///
/// The shortlist is a *sample* of the sources that could feed the focus. Its
/// cost is linear in the head length, so coverage grows with the square root of
/// the population: a creature with four times the sources examines twice the
/// head, not four times it.
const RESIDUAL_SHORTLIST_GAIN: f64 = 2.0;

/// Ceiling on the derived residual shortlist head.
///
/// The residual scan re-scores every shortlisted source against every folded
/// record, so an uncapped head would let one analysis eat the run budget.
const RESIDUAL_SHORTLIST_CEILING: usize = 512;

/// Sublinear gain applied to the ranked-source population for the extra unused
/// hidden sources folded in beyond the head.
const RESIDUAL_HIDDEN_EXTRA_GAIN: f64 = 0.5;

/// Ceiling on the derived extra unused hidden sources.
const RESIDUAL_HIDDEN_EXTRA_CEILING: usize = 128;

/// Sublinear gain applied to the input width for synthetic probe rows.
///
/// The probes fit residual correlations in an `input`-dimensional space, so a
/// row count fixed while the input width grows becomes rank-deficient. Rows
/// cost one forward pass each, hence a square-root gain rather than a linear
/// one.
const SYNTHETIC_PROBE_GAIN: f64 = 3.0;

/// Ceiling on derived synthetic probe rows.
const SYNTHETIC_PROBE_CEILING: usize = 512;

/// Sublinear gain applied to the non-input neuron count for the focus count.
///
/// One creature-wide analysis is paid per experiment whatever the focus count,
/// so a wider creature — a costlier analysis over more distinguishable regions
/// — should amortise that analysis over more focuses.
const FOCUS_COUNT_GAIN: f64 = 0.125;

/// Ceiling on the derived focus count before the budget caps below are applied.
const FOCUS_COUNT_CEILING: usize = 16;

/// Candidates a derived focus must still be able to spend.
///
/// The candidate budget is split evenly across focuses, so amortising the
/// analysis over more focuses than this starves each one of a usable batch.
const MIN_CANDIDATES_PER_FOCUS: usize = 8;

/// Wall-clock seconds a derived focus must still be able to spend.
///
/// The wall-clock half of the same cap: a short run cannot afford to fan out,
/// however wide the creature is.
const MIN_SECONDS_PER_FOCUS: u64 = 60;

/// Clamp a sublinear budget into `[floor, ceiling]`.
///
/// Returns `round(gain × √population)`, floored at `floor` and capped at
/// `ceiling`. This is the one derivation shape used across the module: cost
/// grows with the square root of the population it covers, never linearly, and
/// is bounded at both ends so a toy creature keeps the fixed budget and a very
/// large one cannot spend the whole run in one analysis.
///
/// `ceiling` below `floor` is raised to `floor` — a budget can never resolve
/// below its own floor.
pub fn sublinear_budget(floor: usize, population: usize, gain: f64, ceiling: usize) -> usize {
    let ceiling = ceiling.max(floor);
    if !gain.is_finite() || gain <= 0.0 {
        return floor;
    }
    let scaled = (gain * (population as f64).sqrt()).round();
    if !scaled.is_finite() || scaled <= floor as f64 {
        return floor;
    }
    (scaled as usize).min(ceiling)
}

/// Dimensions of the supplied creature.
///
/// Read from the creature the run was handed, never assumed. Every field is a
/// count, so the struct is safe to journal and to compare across runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreatureScale {
    /// Observation width — the authoritative input count.
    pub inputs: usize,
    /// Output width.
    pub outputs: usize,
    /// Non-input neurons (hidden plus output), as the export lists them.
    pub non_input_neurons: usize,
    /// Synapse count.
    pub synapses: usize,
    /// Whether the creature is strictly feed-forward.
    pub forward_only: bool,
}

impl CreatureScale {
    /// Read the dimensions of `creature`.
    pub fn from_creature(creature: &CreatureExport) -> Self {
        Self {
            inputs: creature.input,
            outputs: creature.output,
            non_input_neurons: creature.neurons.len(),
            synapses: creature.synapses.len(),
            forward_only: creature.forward_only,
        }
    }

    /// Total neurons, inputs included.
    pub fn neurons(&self) -> usize {
        self.inputs.saturating_add(self.non_input_neurons)
    }

    /// Sources the structural generator may rank as a new edge into a focus.
    ///
    /// Every input and every non-input neuron is a candidate source, so this is
    /// the population the residual shortlist samples from.
    pub fn ranked_sources(&self) -> usize {
        self.neurons()
    }
}

/// Whether scale-sensitive budgets are fixed literals or derived (issue #223).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleBudgetMode {
    /// The pre-#223 run: every scale-sensitive budget is a fixed literal.
    #[default]
    Fixed,
    /// Budgets derived from the supplied creature's dimensions and the run's
    /// wall-clock budget.
    Derived,
}

impl ScaleBudgetMode {
    /// Parse a CLI spelling (`fixed` / `derived`).
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "fixed" => Some(Self::Fixed),
            "derived" => Some(Self::Derived),
            _ => None,
        }
    }

    /// Stable label for logs, journals and reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Derived => "derived",
        }
    }

    /// True when budgets are derived from the supplied creature.
    pub fn is_derived(self) -> bool {
        matches!(self, Self::Derived)
    }
}

/// Limits the residual source refine runs under.
///
/// Bundled so the scan entry points keep a readable signature: the head, the
/// extra hidden sources and the synthetic-probe fallback are always chosen
/// together, from the same creature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualLimits {
    /// Head of the prior ranking re-scored by residual correlation.
    pub shortlist: usize,
    /// Extra unused non-input sources folded in beyond the head.
    pub hidden_extra: usize,
    /// Synthetic probe rows drawn when the corpus has fewer than two records.
    pub synthetic_probes: usize,
}

impl ResidualLimits {
    /// The pre-#223 fixed literals.
    pub const FIXED: Self = Self {
        shortlist: FIXED_RESIDUAL_SHORTLIST,
        hidden_extra: FIXED_RESIDUAL_HIDDEN_EXTRA,
        synthetic_probes: FIXED_SYNTHETIC_PROBE_ROWS,
    };

    /// Limits derived from `scale`.
    ///
    /// Never below [`Self::FIXED`]: a creature smaller than the one the
    /// literals were chosen for keeps them, and a larger one grows sublinearly.
    pub fn derived(scale: CreatureScale) -> Self {
        Self {
            shortlist: sublinear_budget(
                FIXED_RESIDUAL_SHORTLIST,
                scale.ranked_sources(),
                RESIDUAL_SHORTLIST_GAIN,
                RESIDUAL_SHORTLIST_CEILING,
            ),
            hidden_extra: sublinear_budget(
                FIXED_RESIDUAL_HIDDEN_EXTRA,
                scale.ranked_sources(),
                RESIDUAL_HIDDEN_EXTRA_GAIN,
                RESIDUAL_HIDDEN_EXTRA_CEILING,
            ),
            synthetic_probes: sublinear_budget(
                FIXED_SYNTHETIC_PROBE_ROWS,
                scale.inputs,
                SYNTHETIC_PROBE_GAIN,
                SYNTHETIC_PROBE_CEILING,
            ),
        }
    }

    /// Limits for `mode` against `scale`.
    pub fn resolve(mode: ScaleBudgetMode, scale: CreatureScale) -> Self {
        match mode {
            ScaleBudgetMode::Fixed => Self::FIXED,
            ScaleBudgetMode::Derived => Self::derived(scale),
        }
    }
}

/// Budgets a run resolved before its first experiment (issue #223).
///
/// Journalled in the run header beside [`CreatureScale`], so every run records
/// the dimensions it was handed and the budgets it chose from them — a reader
/// never has to reconstruct either from the flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedBudgets {
    /// Mode the budgets were resolved under.
    pub mode: ScaleBudgetMode,
    /// Focus neurons proposed against per experiment.
    pub focus_count: usize,
    /// Candidates generated per experiment.
    pub candidates: usize,
    /// Residual refine limits.
    pub residual: ResidualLimits,
    /// Graft-replay (Phase-G) budget in milliseconds — the resolved value,
    /// including the default fraction of the wall-clock budget.
    pub graft_replay_ms: u64,
    /// Wall-clock budget in seconds.
    pub timeout_seconds: u64,
}

impl ResolvedBudgets {
    /// Resolve every scale-sensitive budget for `config` against `scale`.
    pub fn resolve(config: &LamarckConfig, scale: CreatureScale) -> Self {
        let mode = config.scale_budgets;
        let timeout_seconds = config.timeout.as_secs();
        let focus_count = match mode {
            ScaleBudgetMode::Fixed => config.focus_count,
            ScaleBudgetMode::Derived => {
                derived_focus_count(scale, config.candidates, timeout_seconds)
            }
        };
        let graft_replay = config
            .graft_replay_budget
            .unwrap_or_else(|| default_graft_replay_budget(config.timeout));
        Self {
            mode,
            focus_count,
            candidates: config.candidates,
            residual: ResidualLimits::resolve(mode, scale),
            graft_replay_ms: u64::try_from(graft_replay.as_millis()).unwrap_or(u64::MAX),
            timeout_seconds,
        }
    }

    /// Graft-replay budget as a [`Duration`].
    pub fn graft_replay_budget(&self) -> Duration {
        Duration::from_millis(self.graft_replay_ms)
    }

    /// True when the resolved focus count overrode a configured one.
    ///
    /// Under [`ScaleBudgetMode::Derived`] the focus count is derived, so an
    /// explicit `--focus-count` is reported as overridden rather than silently
    /// dropped.
    pub fn focus_count_overridden(&self, configured: usize) -> bool {
        self.mode.is_derived()
            && configured != self.focus_count
            && configured != DEFAULT_FOCUS_COUNT
    }
}

/// Focus count derived from creature width, the candidate budget and the clock.
///
/// Sublinear in the non-input neuron count, then capped by what a focus can
/// still be given: at least [`MIN_CANDIDATES_PER_FOCUS`] candidates and
/// [`MIN_SECONDS_PER_FOCUS`] of wall clock each.
pub fn derived_focus_count(scale: CreatureScale, candidates: usize, timeout_seconds: u64) -> usize {
    let width = sublinear_budget(
        1,
        scale.non_input_neurons,
        FOCUS_COUNT_GAIN,
        FOCUS_COUNT_CEILING,
    );
    let candidate_cap = (candidates / MIN_CANDIDATES_PER_FOCUS).max(1);
    let clock_cap = usize::try_from(timeout_seconds / MIN_SECONDS_PER_FOCUS)
        .unwrap_or(usize::MAX)
        .max(1);
    width.min(candidate_cap).min(clock_cap).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LamarckConfig;
    use neat_core::parse_creature_json;

    /// Creature with `inputs` inputs, `hidden` hidden neurons and one output.
    fn creature(inputs: usize, hidden: usize) -> CreatureExport {
        let mut neurons = String::new();
        let mut synapses = String::new();
        for h in 0..hidden {
            neurons.push_str(&format!(
                r#"{{"type":"hidden","uuid":"h{h}","bias":0.01,"squash":"TANH"}},"#
            ));
            synapses.push_str(&format!(
                r#"{{"fromUUID":"input-{}","toUUID":"h{h}","weight":0.1}},"#,
                h % inputs.max(1)
            ));
            synapses.push_str(&format!(
                r#"{{"fromUUID":"h{h}","toUUID":"o1","weight":0.1}},"#
            ));
        }
        neurons.push_str(r#"{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}"#);
        synapses.push_str(r#"{"fromUUID":"input-0","toUUID":"o1","weight":0.1}"#);
        let json = format!(
            r#"{{"input":{inputs},"output":1,"neurons":[{neurons}],"synapses":[{synapses}]}}"#
        );
        parse_creature_json(&json).expect("fixture parses")
    }

    #[test]
    fn scale_reads_dimensions_from_the_creature() {
        let scale = CreatureScale::from_creature(&creature(16, 4));
        assert_eq!(scale.inputs, 16);
        assert_eq!(scale.outputs, 1);
        // Four hiddens plus the output.
        assert_eq!(scale.non_input_neurons, 5);
        assert_eq!(scale.neurons(), 21);
        assert_eq!(scale.ranked_sources(), 21);
        assert_eq!(scale.synapses, 9);
    }

    #[test]
    fn sublinear_budget_floors_ceilings_and_grows_with_the_root() {
        // Below the floor the fixed budget stands.
        assert_eq!(sublinear_budget(48, 100, 2.0, 512), 48);
        // 2 × √4096 = 128.
        assert_eq!(sublinear_budget(48, 4096, 2.0, 512), 128);
        // Capped.
        assert_eq!(sublinear_budget(48, 1_000_000, 2.0, 512), 512);
        // Quadrupling the population doubles the budget.
        let small = sublinear_budget(1, 2_500, 2.0, usize::MAX);
        let large = sublinear_budget(1, 10_000, 2.0, usize::MAX);
        assert_eq!(large, small * 2);
        // A ceiling below the floor cannot pull the budget under its floor.
        assert_eq!(sublinear_budget(48, 4096, 2.0, 8), 48);
        // A non-positive or non-finite gain falls back to the floor.
        assert_eq!(sublinear_budget(48, 4096, 0.0, 512), 48);
        assert_eq!(sublinear_budget(48, 4096, f64::NAN, 512), 48);
    }

    #[test]
    fn fixed_mode_reproduces_the_pre_223_literals_at_every_size() {
        for (inputs, hidden) in [(4usize, 2usize), (2_511, 1_590), (2_511, 4_852)] {
            let scale = CreatureScale::from_creature(&creature(inputs, hidden));
            assert_eq!(
                ResidualLimits::resolve(ScaleBudgetMode::Fixed, scale),
                ResidualLimits::FIXED,
                "fixed mode must not vary with creature size"
            );
        }
    }

    #[test]
    fn derived_limits_never_shrink_below_the_fixed_ones() {
        let tiny = CreatureScale::from_creature(&creature(3, 1));
        let derived = ResidualLimits::derived(tiny);
        assert_eq!(derived, ResidualLimits::FIXED);
    }

    #[test]
    fn derived_limits_grow_with_creature_size() {
        let small = ResidualLimits::derived(CreatureScale::from_creature(&creature(64, 16)));
        let medium = ResidualLimits::derived(CreatureScale::from_creature(&creature(2_511, 1_590)));
        let large = ResidualLimits::derived(CreatureScale::from_creature(&creature(2_511, 4_852)));
        assert!(
            small.shortlist <= medium.shortlist && medium.shortlist < large.shortlist,
            "shortlist must grow with the ranked-source population: {small:?} {medium:?} {large:?}"
        );
        assert!(medium.hidden_extra < large.hidden_extra);
        // The probe count follows the input width, which is equal here.
        assert_eq!(medium.synthetic_probes, large.synthetic_probes);
        assert!(large.shortlist <= RESIDUAL_SHORTLIST_CEILING);
    }

    #[test]
    fn derived_focus_count_amortises_wider_creatures_but_respects_budgets() {
        let tiny = CreatureScale::from_creature(&creature(8, 2));
        let wide = CreatureScale::from_creature(&creature(2_511, 4_852));
        assert_eq!(derived_focus_count(tiny, 100, 2_700), 1);
        assert!(derived_focus_count(wide, 100, 2_700) > 1);
        // The candidate budget caps the fan-out: 16 candidates buys two focuses.
        assert_eq!(derived_focus_count(wide, 16, 2_700), 2);
        // So does the clock: a 90-second run buys one focus.
        assert_eq!(derived_focus_count(wide, 100, 90), 1);
        // A zero budget still resolves to a runnable single focus.
        assert_eq!(derived_focus_count(wide, 0, 0), 1);
    }

    #[test]
    fn resolved_budgets_record_the_graft_replay_default() {
        let config = LamarckConfig {
            timeout: Duration::from_secs(2_700),
            graft_replay_budget: None,
            ..LamarckConfig::default()
        };
        let scale = CreatureScale::from_creature(&creature(2_511, 1_590));
        let budgets = ResolvedBudgets::resolve(&config, scale);
        assert_eq!(
            budgets.graft_replay_budget(),
            default_graft_replay_budget(config.timeout),
            "the header must record the resolved Phase-G budget, not None"
        );
        assert_eq!(budgets.timeout_seconds, 2_700);
        assert_eq!(budgets.mode, ScaleBudgetMode::Fixed);
        assert_eq!(budgets.focus_count, config.focus_count);
    }

    #[test]
    fn derived_mode_resolves_focus_count_from_the_creature() {
        let config = LamarckConfig {
            timeout: Duration::from_secs(2_700),
            candidates: 100,
            scale_budgets: ScaleBudgetMode::Derived,
            ..LamarckConfig::default()
        };
        let scale = CreatureScale::from_creature(&creature(2_511, 4_852));
        let budgets = ResolvedBudgets::resolve(&config, scale);
        assert!(budgets.focus_count > config.focus_count);
        assert!(budgets.focus_count_overridden(3));
        assert!(!budgets.focus_count_overridden(DEFAULT_FOCUS_COUNT));
    }
}
