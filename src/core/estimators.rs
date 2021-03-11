//! This module contains everything related to estimators.
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign};

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

/// More estimators.
pub trait Estimators<T: Float>: BasicEstimators<T> {
    /// Returns the number of times $N$, the integrand has been called.
    fn calls(&self) -> usize;

    /// Returns the number of times, $N_\mathrm{nf}$, the integrand has been called
    /// and its return value was non-finite.
    fn non_finite_calls(&self) -> usize;

    /// Returns the number of times, $N_\mathrm{nz}$, the integrand has been called
    /// and its return value was non-zero.
    fn non_zero_calls(&self) -> usize;
}

/// A struct implementing the `BasicEstimator<T>` trait.
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct MeanVar<T> {
    mean: T,
    var: T,
}

impl<T: std::ops::Add<Output = T>> Add for MeanVar<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            mean: self.mean + other.mean,
            var: self.var + other.var,
        }
    }
}

impl<T: std::ops::Add<Output = T> + AddAssign> AddAssign for MeanVar<T> {
    fn add_assign(&mut self, other: Self) {
        self.mean += other.mean;
        self.var += other.var;
    }
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
