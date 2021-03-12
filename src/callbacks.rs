//! Implementation of different callback functions.
//!
//! Callback functions serve the purpose of communicating
//! the results of the integration to the user. This can
//! be done in several ways, such as printing to the terminal
//! with various levels of detail or storing a the checkpoints
//! to a file.
//!
//! A callback function is called after each successful iteration
//! and it is passed a vector of the checkpoints computed up to this
//! point.
use crate::core::estimators::Estimators;
use crate::core::Checkpoint;
use num_traits::{Float, FromPrimitive};
use std::fmt::Display;
use std::fs::OpenOptions;
use std::io::Write;
use std::ops::AddAssign;
use std::path::Path;

use serde::Serialize;

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

/// A callback function that serializes the checkpoints and
/// stores them in a file.
pub struct FileWriterCallback<P> {
    path: P,
}

impl<P: AsRef<Path>> FileWriterCallback<P> {
    /// Create a new `FileWriterCallback` by specifying the
    /// path to the checkpoint file.
    pub fn new(path: P) -> Self {
        Self { path }
    }
}

impl<T, R, E, P> Callback<T, R, E> for FileWriterCallback<P>
where
    T: AddAssign + Display + Float + FromPrimitive + Serialize,
    E: Clone + Estimators<T> + std::default::Default + std::ops::Add<Output = E> + Serialize,
    R: Clone + Serialize,
    P: AsRef<Path>,
{
    fn print(&self, chkpts: &[Checkpoint<T, R, E>]) {
        let file = OpenOptions::new()
            .write(true)
            .open(&self.path)
            .expect("Unable to open checkpoint file.");

        writeln!(
            &file,
            "{}",
            serde_json::to_string(&chkpts).expect("Unable to write checkpoints to string.")
        )
        .expect("Unable to write serialized checkpoint to file.");
    }
}
