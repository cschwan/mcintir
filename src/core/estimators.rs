//! This module contains everything related to estimators.
use num_traits::Float;
use serde::{Deserialize, Serialize};

/// Basic estimators, like the mean, variance, and the standard deviation.
pub trait BasicEstimators<T: Float> {
    /// Returns the mean value.
    fn mean(&self) -> T;

    /// Returns the variance, $V$.
    fn var(&self) -> T;

    /// Returns the standard deviation, $\sigma = \sqrt{V}$.
    fn std(&self) -> T {
        self.var().sqrt()
    }
}

pub(crate) trait Updateable<T> {
    /// Update this estimator with `value`.
    fn update(&mut self, value: T);
}

/// More estimators.
pub trait Estimators<T: Float>: BasicEstimators<T> {
    /// Returns the number of times, $N$, the integrand has been called.
    fn calls(&self) -> usize;

    /// Returns the number of times, $N_\mathrm{nf}$, the integrand has been called and its return
    /// value was non-finite.
    fn non_finite_calls(&self) -> usize;

    /// Returns the number of times, $N_\mathrm{nz}$, the integrand has been called and its return
    /// value was non-zero.
    fn non_zero_calls(&self) -> usize;
}

/// A struct implementing the `BasicEstimator<T>` trait.
#[derive(Deserialize, Serialize)]
pub struct MeanVar<T> {
    mean: T,
    var: T,
}

impl<T> MeanVar<T> {
    /// Constructor.
    pub const fn new(mean: T, var: T) -> Self {
        Self { mean, var }
    }
}

impl<T: Float> BasicEstimators<T> for MeanVar<T> {
    fn mean(&self) -> T {
        self.mean
    }

    fn var(&self) -> T {
        self.var
    }
}