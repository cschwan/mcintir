//! Implementation of different callback functions.
use crate::core::estimators::Estimators;
use crate::core::Checkpoint;
use num_traits::{Float, FromPrimitive};
use std::fmt::Display;
use std::ops::AddAssign;

/// Trait for implementing callbacks for iterative MC algorithms
pub trait Callback<T, R, E>
where
    T: Copy,
{
    /// This method is called after each successfully finished iteration and may print information
    /// about it.
    fn print(&self, chkpts: &[Checkpoint<T, R, E>]);
}

/// A callback function that does nothing
pub struct SinkCallback {}

impl<T, R, E> Callback<T, R, E> for SinkCallback
where
    T: Copy,
{
    fn print(&self, _: &[Checkpoint<T, R, E>]) {}
}

/// A callback function that prints the result of each individual iteration
pub struct SimpleCallback {}

impl<T, R, E> Callback<T, R, E> for SimpleCallback
where
    T: AddAssign + Display + Float + FromPrimitive,
    E: Estimators<T>,
{
    fn print(&self, chkpts: &[Checkpoint<T, R, E>]) {
        let iteration = chkpts.len();
        // Make sure that there is at least one checkpoint
        // otherwise do nothing.
        if let Some(chkpt) = chkpts.last() {
            let estimators = chkpt.estimators();
            println!("iteration {} finished.", iteration - 1);
            println!(
                "this iteration: N={} E={} \u{b1} {}",
                estimators.calls(),
                estimators.mean(),
                estimators.std()
            );
        }
    }
}

/// Simple cumulative callback that shows the result of the individual integration
/// together with the cumulative result combining it with the previous iterations.
pub struct SimpleCumulativeCallback {}

impl<T, R, E> Callback<T, R, E> for SimpleCumulativeCallback
where
    T: AddAssign + Display + Float + FromPrimitive,
    E: Clone + Estimators<T> + std::default::Default + std::ops::Add<Output = E>,
    R: Clone,
{
    fn print(&self, chkpts: &[Checkpoint<T, R, E>]) {
        let iteration = chkpts.len();

        if iteration == 0 {
            return;
        }

        let it_calls = chkpts[iteration - 1].estimators().calls();
        let it_mean = chkpts[iteration - 1].estimators().mean();
        let it_std = chkpts[iteration - 1].estimators().std();

        // Compute the cumulative result.
        let cumulative = chkpts
            //.clone()
            .into_iter()
            .map(|c| c.estimators())
            .fold(E::default(), |acc, e| acc + e.clone());

        println!(
            "[iteration {}: N={} E={} \u{b1} {}] [Cumulative: N={}, E={} \u{b1} {}]",
            iteration - 1,
            it_calls,
            it_mean,
            it_std,
            cumulative.calls(),
            cumulative.mean(),
            cumulative.std()
        );
    }
}
