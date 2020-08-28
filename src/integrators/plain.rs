//! Contains the PLAIN integrator as well as supporting structs and functions.
//! 
//! # Example
//! 
//! ```rust
//! use mcintir::core::*;
//! use mcintir::integrators::plain::*;
//! use rand_pcg::Pcg64;
//! use assert_approx_eq::assert_approx_eq;
//! 
//! struct MyIntegrand;
//! 
//! /// Integrating the function x^2 from x=1 to x=3
//! /// Which gives the result: 26/3
//! impl Integrand<f64> for MyIntegrand {
//!     /// Call the integrand.
//!     fn call(&self, x: Vec<f64>) -> CallResult<f64> {
//!         // Jacobian to map from the hypercube into the range [1,3]
//!         let phase_space_weight = (3.0 - 1.0);
//!         let psp = (3.0 - 1.0) * x[0] + 1.0;
//!         let val = psp.powi(2) * phase_space_weight;
//!         CallResult::new(val, vec![(x[0], val)])
//!     }
//! 
//!     /// This method is called by the integrator to decide how many random numbers to generate.
//!     fn dim(&self) -> usize {
//!         1
//!     }
//! 
//!     /// One histograms with 10 bins spanning the range [1.0, 3.0]
//!     fn histograms_1d(&self) -> Vec<HistogramSpecification<f64>> {
//!         vec![HistogramSpecification::<f64>::new(1.0, 3.0, 10)]
//!     }
//! }
//!     
//! fn main() {
//! 
//!     let callback = SimpleCumulativeCallback {};
//!     // Initialize the random number generator.
//!     let rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
//!     let integrand = MyIntegrand {};
//!
//!     let check_point = integrate(
//!         &integrand,
//!         &rng,
//!         &callback,
//!         // One iteration with 500 calls
//!         &[500],
//!     ).remove(0);
//!
//!     assert_eq!(check_point.histograms().len(), 1);
//!     assert_eq!(check_point.histograms()[0].bins().len(), 10);
//!
//! }
//! ```

use num_traits::{Float, FromPrimitive};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign};

use crate::core::*;

/// Estimators for an integration performed by `plain::iteration`.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PlainEstimators<T> {
    sum: T,
    sumsq: T,
    calls: usize,
    non_finite_calls: usize,
    non_zero_calls: usize,
}

impl<T: Float> Default for PlainEstimators<T> {
    fn default() -> Self {
        Self {
            sum: T::zero(),
            sumsq: T::zero(),
            calls: 0,
            non_finite_calls: 0,
            non_zero_calls: 0,
        }
    }
}

impl<T: Float> Add for PlainEstimators<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            sum: self.sum + other.sum,
            sumsq: self.sumsq + other.sumsq,
            calls: self.calls + other.calls,
            non_finite_calls: self.non_finite_calls + other.non_finite_calls,
            non_zero_calls: self.non_zero_calls + other.non_zero_calls,
        }
    }
}

impl<T> BasicEstimators<T> for PlainEstimators<T>
where
    T: Float + FromPrimitive,
{
    fn mean(&self) -> T {
        // TODO: Get rid of unwrap.
        self.sum / T::from_usize(self.calls).unwrap()
    }

    fn var(&self) -> T {
        // TODO: Get rid of unwrap
        let calls = T::from_usize(self.calls).unwrap();
        (self.sumsq - self.sum * self.sum / calls) / calls / (calls - T::one())
    }
}

impl<T> Estimators<T> for PlainEstimators<T>
where
    T: Float + FromPrimitive,
{
    fn calls(&self) -> usize {
        self.calls
    }

    fn non_finite_calls(&self) -> usize {
        self.non_finite_calls
    }

    fn non_zero_calls(&self) -> usize {
        self.non_zero_calls
    }
}

impl<T> Updateable<T> for PlainEstimators<T>
where
    T: AddAssign + Float,
{
    fn update(&mut self, value: T) {
        self.calls += 1;

        if value != T::zero() {
            if value.is_finite() {
                self.sum += value;
                self.sumsq += value * value;
            } else {
                self.non_finite_calls += 1;
            }

            self.non_zero_calls += 1;
        }
    }
}

/// Check point for plain integrator.
pub type PlainCheckpoint<T, R> = Checkpoint<T, R, PlainEstimators<T>>;

// Convenience type definition.
type PlainAccumulator<T> = Accumulator<T, PlainEstimators<T>>;

/// Resume integration from a set of checkpoints.
/// 
/// Given a checkpoint file storing the information of previously 
/// gathered integration iterations, the integration of the integrand
/// `int` can be resumed by providing the check points and the number
/// of calls to be performed for each additional iteration.
/// 
/// The `callback` function determines how the information of intermediate
/// steps is presented to the user as soon as it becomes available. 
pub fn resume_integration_from_checkpoints<T, R>(
    int: &impl Integrand<T>,
    check_points_provided: Vec<PlainCheckpoint<T, R>>,
    callback: &impl Callback<T, R, PlainEstimators<T>>,
    iterations: &[usize],
) -> Vec<PlainCheckpoint<T, R>>
where
    T: AddAssign + Float + FromPrimitive + Serialize + Send + std::fmt::Debug + Sync,
    R: Clone + Rng + Serialize + Send + Sync + std::fmt::Debug,
    Standard: Distribution<T>,
{
    let mut check_points: Vec<PlainCheckpoint<T, R>> =
        Vec::with_capacity(iterations.len() + check_points_provided.len());
    check_points_provided
        .into_iter()
        .for_each(|cp| check_points.push(cp));

    let random_numbers_per_call = int.dim();

    let rng = check_points.iter().last().unwrap().rng_after().clone();

    // Create the accumulator
    for calls in iterations {
        // Choose the random number generator.
        // Copy the one provided by the user if it's the first iteration otherwise take the
        // RNG of the previous iteration in its final state.
        let rng_before = if check_points.is_empty() {
            rng.clone()
        } else {
            check_points.last().unwrap().rng_after().clone()
        };

        // The estimator for the given iteration
        let accumulator_iteration = (0..(*calls) as u32)
            .into_par_iter()
            .map(|call| {
                let rng_clone = rng_before.clone();
                // Provide the random numbers
                let x = rng_clone
                    .sample_iter(&Standard)
                    .skip(call as usize * random_numbers_per_call)
                    .take(random_numbers_per_call)
                    .collect::<Vec<_>>();

                PlainAccumulator::empty(int.histograms_1d()).add_call_result(int.call(x))
            })
            // Combine the estimators
            .reduce(
                || PlainAccumulator::empty(int.histograms_1d()),
                |a, b| a + b,
            );

        // This is ugly. Will the compiler take care of this?
        let mut rng_after = rng_before.clone();
        for _ in 0..*calls {
            let _ = rng_after.gen();
        }

        let histograms = accumulator_iteration
            .get_histograms_1d()
            .clone()
            .iter()
            .zip(int.histograms_1d())
            .map(|(h, s)| {
                HistogramEstimators::new(
                    accumulator_iteration.get_estimators().calls(),
                    s,
                    h.bins().clone(),
                )
            })
            .collect();

        check_points.push(PlainCheckpoint::new(
            rng_before,
            rng_after,
            accumulator_iteration.get_estimators().clone(),
            histograms,
        ));

        callback.print(&check_points);
    }

    check_points
}

/// Resume integration from a checkpoint
pub fn resume_integration_from_checkpoint<T, R>(
    int: &impl Integrand<T>,
    check_point_provided: PlainCheckpoint<T, R>,
    callback: &impl Callback<T, R, PlainEstimators<T>>,
    iterations: &[usize],
) -> Vec<PlainCheckpoint<T, R>>
where
    T: AddAssign + Float + FromPrimitive + Serialize + Send + std::fmt::Debug + Sync,
    R: Clone + Rng + Serialize + Send + Sync + std::fmt::Debug,
    Standard: Distribution<T>,
{
    resume_integration_from_checkpoints(int, vec![check_point_provided], callback, iterations)
}

/// Perform the integration.
/// 
/// Integrate the integrand `int` using random numbers provided by the generator `rng`.
/// The `callback` function determines how intermediate results are provided to the user 
/// as soon as they become available.
/// 
/// `iterations` is a slice that contains, for each iteration the number of calls to the integrand.
pub fn integrate<T, R>(
    int: &impl Integrand<T>,
    rng: &R,
    callback: &impl Callback<T, R, PlainEstimators<T>>,
    iterations: &[usize],
) -> Vec<PlainCheckpoint<T, R>>
where
    T: AddAssign + Float + FromPrimitive + Serialize + Send + std::fmt::Debug + Sync,
    R: Clone + Rng + Serialize + Send + Sync,
    Standard: Distribution<T>,
{
    let mut check_points: Vec<PlainCheckpoint<T, R>> = Vec::with_capacity(iterations.len());

    let random_numbers_per_call = int.dim();

    // Create the accumulator
    for calls in iterations {
        // Choose the random number generator.
        // Copy the one provided by the user if it's the first iteration otherwise take the
        // RNG of the previous iteration in its final state.
        let rng_before = if check_points.is_empty() {
            rng.clone()
        } else {
            check_points.last().unwrap().rng_after().clone()
        };

        // The estimator for the given iteration
        let accumulator_iteration = (0..(*calls) as u32)
            .into_par_iter()
            .map(|call| {
                let rng_clone = rng_before.clone();
                // Provide the random numbers
                let x = rng_clone
                    .sample_iter(&Standard)
                    .skip(call as usize * random_numbers_per_call)
                    .take(random_numbers_per_call)
                    .collect::<Vec<_>>();

                PlainAccumulator::empty(int.histograms_1d()).add_call_result(int.call(x))
            })
            // Combine the estimators
            .reduce(
                || PlainAccumulator::empty(int.histograms_1d()),
                |a, b| a + b,
            );

        // This is ugly. Will the compiler take care of this?
        let mut rng_after = rng_before.clone();
        for _ in 0..*calls {
            let _ = rng_after.gen();
        }

        let histograms = accumulator_iteration
            .get_histograms_1d()
            .clone()
            .iter()
            .zip(int.histograms_1d())
            .map(|(h, s)| {
                HistogramEstimators::new(
                    accumulator_iteration.get_estimators().calls(),
                    s,
                    h.bins().clone(),
                )
            })
            .collect();

        check_points.push(PlainCheckpoint::new(
            rng_before,
            rng_after,
            accumulator_iteration.get_estimators().clone(),
            histograms,
        ));

        callback.print(&check_points);
    }

    check_points
}
