//! Plain integrator
use crate::callbacks::Callback;
use crate::core::estimators::*;
use crate::core::*;
use crate::histograms::*;

use num_traits::{Float, FromPrimitive};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign};

use crossbeam as cb;

#[derive(Debug, Clone, Deserialize, Serialize)]
/// Estimators for the plain integrator.
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

/// Perform part of the integration of a given integration on a specific `core`.
fn perform_iteration_contribution_from_core<T, R, I>(
    integrand: &I,
    mut rng: R,
    calls: usize,
    calls_per_core: usize,
    core: usize,
    n_cores: usize,
) -> (PlainEstimators<T>, Vec<HistogramAccumulator<T>>)
where
    I: Integrand<T>,
    T: Float + AddAssign + FromPrimitive + Send + Sync + std::fmt::Debug,
    R: Clone + Rng + Send + Sync + Serialize,
    Standard: Distribution<T>,
{
    // determine how many calls to the random number generator to skip
    let skip = calls_per_core * core * &integrand.dim();

    // initialize the random number generator on the given core
    for _ in 0..skip {
        let _ = rng.gen::<T>();
    }

    // in the last iteration, not all calls might be needed
    let actual_calls = compute_calls_for_core(core, n_cores, calls);

    // create a buffer for the sampled random variables such that
    // we do not need to allocate vectors in every call
    let mut x = vec![T::zero(); integrand.dim()];
    let mut histograms = integrand
        .histograms()
        .iter()
        .map(|h| h.get_accumulator())
        .collect();
    // let mut args = PlainArguments::new(integrand.dim(), &mut histograms);
    let estimators =
        (0..actual_calls)
            .into_iter()
            .fold(PlainEstimators::<T>::default(), |mut acc, _| {
                // sample a new phase space point
                x.iter_mut().for_each(|v| *v = rng.gen());

                // evaluate the integrand
                let value = integrand.call(&x, &mut histograms);

                acc.calls += 1;

                if value != T::zero() {
                    acc.non_zero_calls += 1;

                    if value.is_finite() {
                        acc.sum += value;
                        acc.sumsq += value * value;
                    } else {
                        acc.non_finite_calls += 1;
                    }
                }

                acc
            });
    (estimators, histograms)
}

/// Perform a single iteration of integrating the `integrand` on `n_cores` cores using `calls` samples.
fn integrate_iteration<T, R, I>(
    integrand: &I,
    rng: &R,
    n_cores: usize,
    calls: usize,
) -> Checkpoint<T, R, PlainEstimators<T>>
where
    I: Integrand<T>,
    T: Float + AddAssign + FromPrimitive + Send + Sync + std::fmt::Debug,
    R: Clone + Rng + Send + Sync + Serialize,
    Standard: Distribution<T>,
{
    let calls_per_core = (calls as f32 / n_cores as f32).ceil() as usize;

    let mut rng_global = rng.clone();

    // distribute the workload evenly across the cores
    let collect_results = cb::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_cores);

        for core in 0..n_cores {
            // Needs to be defined before spawning the thread
            let rng_local = rng_global.clone();

            handles.push(s.spawn(move |_| {
                perform_iteration_contribution_from_core(
                    integrand,
                    rng_local,
                    calls,
                    calls_per_core,
                    core,
                    n_cores,
                )
            }));
        }

        // wait for the threads to finish
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    })
    .unwrap();

    // accumulate the intermediate results
    let mut estimators = Vec::with_capacity(collect_results.len());
    let mut histograms = Vec::with_capacity(collect_results.len());
    for (e, h) in collect_results {
        estimators.push(e);
        histograms.push(h);
    }

    let accumulate = estimators
        .into_iter()
        .fold(PlainEstimators::<T>::default(), |acc, r| acc + r);

    let mut histograms_iterator = histograms.into_iter();

    // TODO: Maybe this can be cleaned up using `fold_first` once it becomes stable.
    let histograms_accumulate = if let Some(mut histogram_first) = histograms_iterator.next() {
        histograms_iterator.for_each(|histogram_set| {
            histogram_set
                .into_iter()
                .enumerate()
                .for_each(|(idx, h)| histogram_first[idx] += h)
        });
        histogram_first
    } else {
        vec![]
    };

    let distributions = histograms_accumulate
        .into_iter()
        .map(|h| h.to_histogram_estimator(accumulate.calls()))
        .collect::<Vec<_>>();

    // return the updated rng
    for _ in 0..calls * integrand.dim() {
        let _ = rng_global.gen::<T>();
    }

    // (PlainEstimators::default(), rng_global)
    Checkpoint::new(rng.clone(), rng_global, accumulate, distributions)
}

/// Integrate the `integrate` using `n_cores` cores.
///
/// The random number generator in its initial state is provided in `rng`
/// together with a `callback` function that prints estimates after each
/// iteration.
/// The number of calls of the integrand per iteration is stored in the slice
/// `iterations`.
pub fn integrate<T, R, I>(
    integrand: &I,
    rng: &R,
    callback: &impl Callback<T, R, PlainEstimators<T>>,
    n_cores: usize,
    iterations: &[usize],
) -> Vec<Checkpoint<T, R, PlainEstimators<T>>>
where
    I: Integrand<T>,
    T: Float + AddAssign + FromPrimitive + Send + Sync + Clone + std::fmt::Debug,
    R: Clone + Rng + Send + Sync + Serialize,
    Standard: Distribution<T>,
{
    // storage for the results of each iteration
    let mut checkpoints = Vec::with_capacity(iterations.len());

    let mut rng_global = rng.clone();

    // Integration iterations are treated sequentially
    for calls in iterations {
        let checkpoint = integrate_iteration(integrand, &rng_global, n_cores, *calls);
        // synchronize the random number generation
        rng_global = checkpoint.rng_after().clone();

        checkpoints.push(checkpoint);
        callback.print(&checkpoints);
    }

    checkpoints
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::callbacks::{FileWriterCallback, SinkCallback};
    use crate::integrators::plain;
    use rand::Rng;
    use rand_pcg::Pcg64;
    use serde::Serialize;
    use std::fs::read_to_string;
    use tempfile::NamedTempFile;

    use assert_approx_eq::assert_approx_eq;

    const TOLERANCE: f64 = 1e-16;

    fn assert_eq_rng<R>(lhs: &R, rhs: &R)
    where
        R: Rng + Serialize,
    {
        assert_eq!(
            serde_json::to_string(lhs).unwrap(),
            serde_json::to_string(rhs).unwrap()
        );
    }

    struct MyIntegrand {}

    impl Integrand<f64> for MyIntegrand {
        fn call(&self, args: &Vec<f64>, h: &mut Vec<HistogramAccumulator<f64>>) -> f64 {
            let y = 2.0 * args[0] - 1.0;
            let val = y.abs();
            h[0].fill(&vec![y], val);
            val
        }

        fn dim(&self) -> usize {
            1
        }

        fn histograms(&self) -> Vec<HistogramSpecification<f64>> {
            vec![HistogramSpecification::new(
                vec![-1.0],
                vec![1.0],
                vec![10],
                vec!["x".to_string()],
                "histogram".to_string(),
            )]
        }
    }

    #[test]
    fn test_plain_iteration() {
        // define a callback function
        let callback = SinkCallback {};

        // define the calls per iteration
        let iterations = vec![1000];

        // create a random number generator
        let rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

        // perform a full integration over the four iterations
        let chkpt = plain::integrate(&MyIntegrand {}, &rng.clone(), &callback, 1, &iterations);

        // we requested 1000 calls
        assert_eq!(chkpt[0].estimators().calls(), 1000);

        // check the mean
        assert_approx_eq!(
            chkpt[0].estimators().mean(),
            4.891001827394124e-1,
            TOLERANCE
        );

        // check the variance
        assert_approx_eq!(chkpt[0].estimators().var(), 8.704232037144878e-5, TOLERANCE);

        // there is one histogram
        assert_eq!(chkpt[0].histograms().len(), 1);

        assert_approx_eq!(
            chkpt[0].histograms()[0].mean(),
            4.8910018273941236e-1,
            TOLERANCE
        );
        assert_approx_eq!(
            chkpt[0].histograms()[0].var(),
            2.9477356489145525e-4,
            TOLERANCE
        );

        let bins = &chkpt[0].histograms()[0].bins();

        assert_eq!(bins.len(), 10);

        assert_approx_eq!(bins[0].mean(), 8.42459132957333e-2, TOLERANCE);
        assert_approx_eq!(bins[1].mean(), 7.104385717803988e-2, TOLERANCE);
        assert_approx_eq!(bins[2].mean(), 3.847711846183505e-2, TOLERANCE);
        assert_approx_eq!(bins[3].mean(), 3.1290654087192664e-2, TOLERANCE);
        assert_approx_eq!(bins[4].mean(), 8.040858172069591e-3, TOLERANCE);
        assert_approx_eq!(bins[5].mean(), 1.1121191486361903e-2, TOLERANCE);
        assert_approx_eq!(bins[6].mean(), 3.3109381404971314e-2, TOLERANCE);
        assert_approx_eq!(bins[7].mean(), 5.266620809464682e-2, TOLERANCE);
        assert_approx_eq!(bins[8].mean(), 6.584518948420331e-2, TOLERANCE);
        assert_approx_eq!(bins[9].mean(), 9.325981107435849e-2, TOLERANCE);
    }

    #[test]
    fn test_resume_from_checkpoint() {
        // define a callback function
        let callback = SinkCallback {};

        // define the calls per iteration
        let iterations = vec![1000, 1000, 1000, 1000];

        // create a random number generator
        let rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

        // perform a full integration over the four iterations
        let result_4_it =
            plain::integrate(&MyIntegrand {}, &rng.clone(), &callback, 1, &iterations);

        // check the result after the first iteration
        assert_eq!(result_4_it[0].estimators().mean(), 4.891001827394124e-1);
        assert_eq!(result_4_it[0].estimators().var(), 8.704232037144878e-5);

        // get the random number generator state after the second iteration
        let rng_resume = result_4_it[1].rng_after();

        // perform two integrations starting from this rng state
        let result_2_it =
            plain::integrate(&MyIntegrand {}, rng_resume, &callback, 1, &iterations[2..4]);

        // compare the rng states of both results ar the end
        assert_eq_rng(result_4_it[3].rng_after(), result_2_it[1].rng_after());

        // compare the results of the last iteration
        assert_eq!(
            result_4_it[3].estimators().mean(),
            result_2_it[1].estimators().mean()
        );

        // compare the variance of the results of the last iteration
        assert_eq!(
            result_4_it[3].estimators().var(),
            result_2_it[1].estimators().var()
        );
    }

    #[test]
    fn test_write_checkpoint_to_file() {
        // create a temporary file to write to
        let file = NamedTempFile::new().unwrap();
        let path = file.path();
        // define the calls per iteration
        let iterations = vec![1000, 100];

        // define a callback function
        let callback = FileWriterCallback::new(&path);

        // create a random number generator
        let rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

        // perform a full integration over the four iterations
        let original = plain::integrate(&MyIntegrand {}, &rng.clone(), &callback, 1, &iterations);

        let chkpt_file = read_to_string(&path).expect("Unable to read checkpoint file");
        let chkpts: Vec<Checkpoint<f64, Pcg64, PlainEstimators<f64>>> =
            serde_json::from_str(&chkpt_file).expect("Unable to deserialize checkpoint from json.");

        // make sure all the checkpoints have been written and read
        assert_eq!(original.len(), chkpts.len());

        // calls
        assert_eq!(
            chkpts[0].estimators().calls(),
            original[0].estimators().calls()
        );
        assert_eq!(
            chkpts[1].estimators().calls(),
            original[1].estimators().calls()
        );

        // check the mean
        assert_eq!(
            chkpts[0].estimators().mean(),
            original[0].estimators().mean()
        );

        // check the mean
        assert_eq!(
            chkpts[1].estimators().mean(),
            original[1].estimators().mean()
        );

        // check the variance
        assert_eq!(chkpts[0].estimators().var(), original[0].estimators().var());

        assert_eq!(chkpts[1].estimators().var(), original[1].estimators().var());

        // there is one histogram
        assert_eq!(chkpts[0].histograms().len(), 1);
        assert_eq!(original[0].histograms().len(), 1);

        // check histogram content
        assert_eq!(
            chkpts[0].histograms()[0].mean(),
            original[0].histograms()[0].mean(),
        );

        assert_eq!(
            chkpts[1].histograms()[0].mean(),
            original[1].histograms()[0].mean(),
        );

        assert_eq!(
            chkpts[0].histograms()[0].var(),
            original[0].histograms()[0].var(),
        );

        assert_eq!(
            chkpts[1].histograms()[0].var(),
            original[1].histograms()[0].var(),
        );

        let bins = &chkpts[0].histograms()[0].bins();

        assert_eq!(bins.len(), 10);

        assert_approx_eq!(bins[0].mean(), 8.42459132957333e-2, TOLERANCE);
        assert_approx_eq!(bins[1].mean(), 7.104385717803988e-2, TOLERANCE);
        assert_approx_eq!(bins[2].mean(), 3.847711846183505e-2, TOLERANCE);
        assert_approx_eq!(bins[3].mean(), 3.1290654087192664e-2, TOLERANCE);
        assert_approx_eq!(bins[4].mean(), 8.040858172069591e-3, TOLERANCE);
        assert_approx_eq!(bins[5].mean(), 1.1121191486361903e-2, TOLERANCE);
        assert_approx_eq!(bins[6].mean(), 3.3109381404971314e-2, TOLERANCE);
        assert_approx_eq!(bins[7].mean(), 5.266620809464682e-2, TOLERANCE);
        assert_approx_eq!(bins[8].mean(), 6.584518948420331e-2, TOLERANCE);
        assert_approx_eq!(bins[9].mean(), 9.325981107435849e-2, TOLERANCE);
    }
}
