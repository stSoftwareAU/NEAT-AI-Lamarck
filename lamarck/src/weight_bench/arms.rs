//! The in-process optimiser arms and their seeds.
//!
//! The budget-aware router and its landscape probes were removed after an
//! earlier synthetic run (crate 0.1.46): the router never beat CMA-ES alone and
//! the probes did not separate the activation families, so they did not earn
//! their maintenance cost (issue #266).

use serde::{Deserialize, Serialize};

use super::cmaes::{CMA_SIGMA0, cma_es};
use super::objective::{BenchError, BudgetedObjective};
use super::refine::lamarck_refine;

/// An optimiser arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Arm {
    /// The simplified random-axis control; not the production candidate pipeline.
    LamarckRefine,
    /// Full-covariance CMA-ES.
    CmaEs,
    /// The published AMM optimiser, run offline through `scripts/amm-reference.py`; never run in-process.
    AmmReference,
}

/// Per-arm seed: the problem seed XOR a fixed constant for the arm. The
/// constants are unchanged from the router-era run, so its refine and CMA-ES
/// rows are reproduced; a unit test pins them.
pub fn arm_seed(arm: Arm, problem_seed: u64) -> u64 {
    problem_seed
        ^ match arm {
            Arm::LamarckRefine => 0x5245_4649_4e45,
            Arm::CmaEs => 0x434d_4145_5300,
            Arm::AmmReference => 0x414d_4d52_4546,
        }
}

/// Run an in-process `arm` on the budgeted objective.
pub fn run_arm(
    obj: &mut BudgetedObjective<'_>,
    arm: Arm,
    problem_seed: u64,
) -> Result<(), BenchError> {
    match arm {
        Arm::LamarckRefine => lamarck_refine(obj, arm_seed(arm, problem_seed)),
        Arm::CmaEs => cma_es(obj, arm_seed(arm, problem_seed), CMA_SIGMA0),
        Arm::AmmReference => Err(BenchError::InvalidConfig(
            "the AMM reference arm runs offline via the serve mode (--serve-reference), not in-process"
                .into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{Arm, arm_seed};

    /// Pins the per-arm seed streams so paired rows cannot drift silently.
    #[test]
    fn arm_seeds_are_pinned() {
        assert_eq!(arm_seed(Arm::LamarckRefine, 0), 0x5245_4649_4e45);
        assert_eq!(arm_seed(Arm::CmaEs, 1001), 0x434d_4145_5300 ^ 1001);
        assert_eq!(arm_seed(Arm::AmmReference, 1012), 0x414d_4d52_4546 ^ 1012);
        assert_eq!(arm_seed(Arm::LamarckRefine, 1001), 90_457_485_430_188);
        assert_eq!(arm_seed(Arm::CmaEs, 1001), 73_999_086_604_521);
    }
}
