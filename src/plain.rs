//! Contains the PLAIN integrator and supporting structs and functions.

use num_traits::{Float, FromPrimitive};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::ops::AddAssign;

use crate::core::*;
use crate::core::estimators::*;
use crate::callbacks::Callback;

struct PlainArguments<T> {
    x: Vec<T>,
    weight: T,
}

impl<T: Float> PlainArguments<T> {
    fn new(dim: usize) -> Self {
        Self {
            x: vec![T::zero(); dim],
            weight: T::one(),
        }
    }
}

impl<T> Arguments<T> for PlainArguments<T>
where
    T: Copy,
    Standard: Distribution<T>,
{
    fn weight(&mut self) -> T {
        self.weight
    }

    fn x(&self) -> &[T] {
        &self.x
    }

    fn histo_filler(&mut self) -> Option<&mut dyn HistogramFiller<T>> {
        None
    }
}

impl<T> MutArguments<T> for PlainArguments<T>
where
    T: Copy,
    Standard: Distribution<T>,
{
    fn x_mut(&mut self) -> &mut [T] {
        &mut self.x
    }

    fn check(&mut self) -> bool {
        true
    }
}

/// Estimators for an integration performed by `plain::iteration`.
#[derive(Deserialize, Serialize)]
pub struct PlainEstimators<T> {
    sum: T,
    sumsq: T,
    calls: usize,
    non_finite_calls: usize,
    non_zero_calls: usize,
}

impl<T: Float> PlainEstimators<T> {
    /// Constructor.
    fn new() -> Self {
        Self {
            sum: T::zero(),
            sumsq: T::zero(),
            calls: 0,
            non_finite_calls: 0,
            non_zero_calls: 0,
        }
    }
}

impl<T> BasicEstimators<T> for PlainEstimators<T>
where
    T: Float + FromPrimitive,
{
    fn mean(&self) -> T {
        self.sum / T::from_usize(self.calls).unwrap()
    }

    fn var(&self) -> T {
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

type PlainCheckpoint<T, R> = Checkpoint<T, R, PlainEstimators<T>>;

/// Performs `iterations.len()` iterations of the PLAIN integrator using `iterations[0]`,
/// `interations[1]`, ..., integrand evaluations. For each iteration this integrates the integrand
/// specified as `int` over the $d$ dimensions given as `dim`, using `calls` integrand evaluations.
/// Random numbers are generated using `rng`. If `histo` is not empty, histograms are filled
/// accordingly.
pub fn integrate<T, R>(
    int: &mut impl Integrand<T>,
    rng: &mut R,
    callback: &impl Callback<T, R, PlainEstimators<T>>,
    iterations: &[usize],
) -> Vec<PlainCheckpoint<T, R>>
where
    T: AddAssign + Float + FromPrimitive + Serialize,
    R: Clone + Rng + Serialize,
    Standard: Distribution<T>,
{
    let mut chkpts: Vec<PlainCheckpoint<T, R>> = Vec::with_capacity(iterations.len());

    for calls in iterations {
        chkpts.push(make_chkpt(
            PlainArguments::new(int.dim()),
            PlainEstimators::new(),
            *calls,
            *calls,
            rng,
            int,
        ));
        callback.print(&chkpts);
    }

    chkpts
}

#[cfg(test)]
mod tests {
    use crate::core::*;
    use crate::core::estimators::*;
    use crate::callbacks::*;
    use crate::plain;
    use rand::Rng;
    use rand_pcg::Pcg64;
    use serde::Serialize;

    fn assert_eq_rng<R>(lhs: &R, rhs: &R)
    where
        R: Rng + Serialize,
    {
        assert_eq!(
            serde_json::to_string(lhs).unwrap(),
            serde_json::to_string(rhs).unwrap()
        );
    }

    struct MyIntegrand {
        y: f64,
    }

    impl Integrand<f64> for MyIntegrand {
        // the integral of the function called `abs`:
        // int_0^1 dx |2*x-1| = int^1_0.5 dx (2x-1) + int^0.5_0 (1-2x)
        //                    = [x^2-x]^1_0.5 + [x-x^2]^0.5_0
        //                    = 0 - (0.25-0.5) + (0.5-0.25) - 0
        //                    = 2*(0.5-0.25)
        //                    = 1-0.5
        //                    = 0.5
        fn call(&mut self, args: &mut impl Arguments<f64>) -> f64 {
            let x = 2.0 * args.x()[0] - 1.0;
            self.y = x.abs();

            if let Some(histo) = args.histo_filler() {
                histo.fill(0, x, self.y);
            }

            self.y
        }

        fn dim(&self) -> usize {
            1
        }

        fn histograms(&self) -> Vec<HistogramSpecification<f64>> {
            vec![HistogramSpecification::new(-1.0, 1.0, 10)]
        }
    }

    #[test]
    fn plain_iteration() {
        let mut rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
        let rng_start = rng.clone();
        let chkpt = plain::integrate(
            &mut MyIntegrand { y: 0.0 },
            &mut rng,
            &SimpleCallback {},
            &[1000],
        )
        .remove(0);

        // compare random number generators before iteration
        assert_eq_rng(chkpt.rng_before(), &rng_start);

        // compare random number generators after iteration
        assert_eq_rng(chkpt.rng_after(), &rng);

        // we requested 1000 calls
        assert_eq!(chkpt.estimators().calls(), 1000);

        // check the mean
        assert_eq!(chkpt.estimators().mean(), 4.891001827394124e-1);

        // check the variance
        assert_eq!(chkpt.estimators().var(), 8.704232037144878e-5);

        // there is one histogram
        assert_eq!(chkpt.histograms().len(), 1);

        assert_eq!(chkpt.histograms()[0].mean(), 4.8910018273941236e-1);
        assert_eq!(chkpt.histograms()[0].var(), 2.9477356489145525e-4);

        let bins = &chkpt.histograms()[0].bins();

        assert_eq!(bins.len(), 10);

        assert_eq!(bins[0].mean(), 8.42459132957333e-2);
        assert_eq!(bins[1].mean(), 7.104385717803988e-2);
        assert_eq!(bins[2].mean(), 3.847711846183505e-2);
        assert_eq!(bins[3].mean(), 3.1290654087192664e-2);
        assert_eq!(bins[4].mean(), 8.040858172069591e-3);
        assert_eq!(bins[5].mean(), 1.1121191486361903e-2);
        assert_eq!(bins[6].mean(), 3.3109381404971314e-2);
        assert_eq!(bins[7].mean(), 5.266620809464682e-2);
        assert_eq!(bins[8].mean(), 6.584518948420331e-2);
        assert_eq!(bins[9].mean(), 9.325981107435849e-2);

        assert_eq!(bins[0].var(), 6.962028200632969e-5);
        assert_eq!(bins[1].var(), 4.4835245076113314e-5);
        assert_eq!(bins[2].var(), 1.7959466185901967e-5);
        assert_eq!(bins[3].var(), 8.751209965559931e-6);
        assert_eq!(bins[4].var(), 8.168485255333883e-7);
        assert_eq!(bins[5].var(), 1.3603316021008941e-6);
        assert_eq!(bins[6].var(), 8.955699997281554e-6);
        assert_eq!(bins[7].var(), 2.4509156464831765e-5);
        assert_eq!(bins[8].var(), 4.2660976883510006e-5);
        assert_eq!(bins[9].var(), 7.530434818429274e-5);
    }
}
