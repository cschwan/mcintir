//! The core module
pub mod estimators;

use crate::core::estimators::Estimators;
use crate::histograms::*;
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::ops::AddAssign;

/// Integrand trait
pub trait Integrand<T: Copy>: Send + Sync {
    /// Call the integrand with a phase space point.
    fn call(&self, x: &Vec<T>, h: &mut Vec<HistogramAccumulator<T>>) -> T;
    /// The dimension of the integrand.
    fn dim(&self) -> usize;
    /// Definitions of the histograms that should be filled during the integration.
    fn histograms(&self) -> Vec<HistogramSpecification<T>> {
        vec![]
    }
}

/// A checkpoint saves the state of the generator after an iteration.
/// Checkpoints can be used to restart or resume iterations.
#[derive(Debug, Deserialize, Serialize)]
pub struct Checkpoint<T, R, E>
where
    T: Copy,
{
    rng_before: R,
    rng_after: R,
    estimators: E,
    histograms: Vec<HistogramEstimators<T>>,
}

impl<T, R, E> Checkpoint<T, R, E>
where
    T: AddAssign + Float + FromPrimitive,
    E: Estimators<T>,
{
    /// Constructor
    pub(crate) fn new(
        rng_before: R,
        rng_after: R,
        estimators: E,
        histograms: Vec<HistogramEstimators<T>>,
    ) -> Self {
        Self {
            rng_before,
            rng_after,
            estimators,
            histograms,
        }
    }

    /// Returns the random number generator before generation of this checkpoint.
    pub fn rng_before(&self) -> &R {
        &self.rng_before
    }

    /// Returns the random number generator after generation of this checkpoint
    pub fn rng_after(&self) -> &R {
        &self.rng_after
    }

    /// Returns the estimators of this checkpoint.
    pub fn estimators(&self) -> &E {
        &self.estimators
    }

    /// Access the histograms
    pub fn histograms(&self) -> &Vec<HistogramEstimators<T>> {
        &self.histograms
    }

    /// Destructure the checkpoint and return its components.
    pub fn destructure(self) -> (R, R, E, Vec<HistogramEstimators<T>>) {
        (
            self.rng_before,
            self.rng_after,
            self.estimators,
            self.histograms,
        )
    }
}

/// Compute the number of calls on a given core, given the total number of cores
/// `n_cores`, the index `core` (zero-based) of the current thread as well as the
/// total number of calls `total_calls` to perform combined on all cores.
pub(crate) fn compute_calls_for_core(core: usize, n_cores: usize, total_calls: usize) -> usize {
    // make sure passed data is valid
    debug_assert!(core < n_cores);
    // naive estimate of the number of cores
    let calls_per_core = (total_calls as f32 / n_cores as f32).ceil() as usize;

    // if we are on the last core, not all of the `calls_per_core` might be needed to reach
    // `total_calls`
    if n_cores == core + 1 {
        total_calls - core * calls_per_core
    } else {
        calls_per_core
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calls_per_core_simple() {
        let n_cores = 3;
        let total_calls = 17;
        let calls_per_core = (0..n_cores)
            .into_iter()
            .map(|core| compute_calls_for_core(core, n_cores, total_calls))
            .collect::<Vec<_>>();

        assert_eq!(calls_per_core[0], 6);
        assert_eq!(calls_per_core[1], 6);
        assert_eq!(calls_per_core[2], 5);
        assert_eq!(total_calls, calls_per_core.into_iter().sum::<usize>());
    }

    #[test]
    fn test_calls_per_core() {
        let n_cores = 13;
        let total_calls = 16490248407;
        let total_calls_check: usize = (0..n_cores)
            .into_iter()
            .map(|core| compute_calls_for_core(core, n_cores, total_calls))
            .sum();
        assert_eq!(total_calls, total_calls_check);
    }
}
