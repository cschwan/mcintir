//! Contains the PLAIN integrator as well as supporting structs and functions.

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

/// Check point for plain integrator
type PlainCheckpoint<T, R> = Checkpoint<T, R, PlainEstimators<T>>;

/// Perform the integration
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
    let mut accumulator = Accumulator::<_, PlainEstimators<_>>::empty(int.histograms_1d());

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

                // Evaluate the integrand on the point & update the estimators & fill the histogramsI
                accumulator
                    .get_empty_accumulator()
                    .add_call_result(int.call(x))
            })
            // Combine the estimators
            .reduce(|| accumulator.get_empty_accumulator(), |a, b| a + b);

        // This is ugly. Will the compiler take care of this?
        let mut rng_after = rng.clone();
        for _ in 0..(iterations.iter().sum::<usize>()) {
            let _ = rng_after.gen();
        }

        accumulator = accumulator + accumulator_iteration;

        let histograms = match accumulator.get_histograms_1d() {
            None => None,
            Some(ref histos) => Some(
                histos
                    .clone()
                    .iter()
                    // TODO: Can we get rid of this unwrap?
                    .zip(int.histograms_1d().unwrap())
                    .map(|(h, s)| {
                        HistogramEstimators::new(
                            accumulator.get_estimators().calls(),
                            s,
                            h.bins().clone(),
                        )
                    })
                    .collect(),
            ),
        };

        check_points.push(PlainCheckpoint::new(
            rng_before,
            rng_after,
            accumulator.get_estimators().clone(),
            histograms,
        ));

        callback.print(&check_points);
    }

    check_points
}
