//! Core functionality. You don't need to import this modules since all it's public members are
//! part of the crate namespace.

use num_traits::{Float, FromPrimitive};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::ops::AddAssign;

/// Argument type every integrand must accept. Since this is a trait, the integrand must accept
/// (mutable) references to a trait object.
pub trait Arguments<T> {
    /// This function allows to fill histograms. Since it's possible there are no histograms, the
    /// return value is an option.
    fn histo_filler(&mut self) -> Option<&mut dyn HistogramFiller<T>>;

    /// Returns the weight of for this call.
    fn weight(&mut self) -> T;

    /// Returns the point of the hypercube $[0,1)^d$ the integrand is evaluated at. The value $d$
    /// is given by the length of the returned slice.
    fn x(&self) -> &[T];
}

pub(crate) trait MutArguments<T>: Arguments<T> {
    fn x_mut(&mut self) -> &mut [T];
    fn check(&mut self) -> bool;
}

/// Trait which every integrand must implement.
pub trait Integrand<T> {
    /// Calculates the value of the integrand as a numerical value of the type `T` from `args.x()`,
    /// which has as many random numbers as specified by `dim()`.
    fn call(&mut self, args: &mut impl Arguments<T>) -> T;

    /// Returns how many random numbers are needed by the integrand.
    fn dim(&self) -> usize;

    /// Returns the histograms that should be filled during integration. If the provided method is
    /// used no histograms are created.
    fn histograms(&self) -> Vec<HistogramSpecification<T>> {
        Vec::new()
    }
}

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

/// Estimators for histograms.
#[derive(Deserialize, Serialize)]
pub struct HistogramEstimators<T> {
    limits: HistogramSpecification<T>,
    calls: usize,
    mean_var: Vec<MeanVar<T>>,
}

impl<T: Copy> HistogramEstimators<T> {
    /// Returns the estimators for all bins.
    pub fn bins(&self) -> &Vec<MeanVar<T>> {
        &self.mean_var
    }
}

impl<T> HistogramEstimators<T>
where
    T: AddAssign + Float + FromPrimitive,
{
    fn new(calls: usize, limits: HistogramSpecification<T>, bins: Vec<(T, T)>) -> Self {
        Self {
            calls,
            limits,
            mean_var: bins
                .into_iter()
                .map(|(sum, sumsq)| {
                    let calls = T::from_usize(calls).unwrap();
                    MeanVar::new(
                        sum / calls,
                        (sumsq - sum * sum / calls) / calls / (calls - T::one()),
                    )
                })
                .collect(),
        }
    }
}

impl<T> BasicEstimators<T> for HistogramEstimators<T>
where
    T: Float,
{
    fn mean(&self) -> T {
        self.mean_var
            .iter()
            .fold(T::zero(), |mean, x| mean + x.mean())
    }

    fn var(&self) -> T {
        self.mean_var.iter().fold(T::zero(), |var, x| var + x.var())
    }
}

/// Trait whose implementers lets one fill histograms.
pub trait HistogramFiller<T> {
    /// In the histogram with index `hist` fill the bin that contains `x` with `value`.
    fn fill(&mut self, hist: usize, x: T, value: T);
}

/// Everything Monte Carlo integrators need to know about histograms.
#[derive(Deserialize, Serialize)]
pub struct HistogramSpecification<T> {
    left: T,
    right: T,
    bins: usize,
    name: String,
    x_label: String,
    y_label: String,
}

impl<T: Copy> HistogramSpecification<T> {
    /// Constructs a one-dimensional histogram, in which the range from `left` (inclusive) to
    /// `right` (exclusive) is subdivided into `bins` number of bins.
    pub fn new(left: T, right: T, bins: usize) -> Self {
        Self {
            left,
            right,
            bins,
            name: String::new(),
            x_label: String::new(),
            y_label: String::new(),
        }
    }

    /// Constructs a one-dimensional histogram, in which the range from `left` (inclusive) to
    /// `right` (exclusive) is subdivided into `bins` number of bins. The histogram also has a
    /// `name`, and labels for its two axes: `x_label` and `y_label`.
    pub fn with_labels(
        left: T,
        right: T,
        bins: usize,
        name: &str,
        x_label: &str,
        y_label: &str,
    ) -> Self {
        Self {
            left,
            right,
            bins,
            name: name.to_string(),
            x_label: x_label.to_string(),
            y_label: y_label.to_string(),
        }
    }

    /// Returns the left boundary of the binned range.
    pub fn left(&self) -> T {
        self.left
    }

    /// Returns the right boundary of the binned range.
    pub fn right(&self) -> T {
        self.right
    }

    /// Returns the number of bins this histogram has.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Returns the name of this histogram.
    pub fn name(&self) -> &String {
        &self.name
    }

    /// Returns the name the x-axis.
    pub fn x_label(&self) -> &String {
        &self.x_label
    }

    /// Returns the name the y-axis.
    pub fn y_label(&self) -> &String {
        &self.y_label
    }
}

/// A checkpoint saves the state of a generator after an iteration. Checkpoints can be used to
/// restart or resume iterations.
#[derive(Deserialize, Serialize)]
pub struct Checkpoint<T, R, E> {
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
    /// Constructor.
    pub(crate) fn new(
        rng_before: R,
        rng_after: R,
        estimators: E,
        limits: Vec<HistogramSpecification<T>>,
        histograms: Vec<Vec<(T, T)>>,
    ) -> Self {
        let calls = estimators.calls();

        Self {
            rng_before,
            rng_after,
            estimators,
            histograms: histograms
                .into_iter()
                .zip(limits.into_iter())
                .map(move |(h, l)| HistogramEstimators::new(calls, l, h))
                .collect(),
        }
    }

    /// Returns the random number generator that was to generate this checkpoint.
    pub fn rng_before(&self) -> &R {
        &self.rng_before
    }

    /// Returns the state of the random number generator after generating this checkpoint.
    pub fn rng_after(&self) -> &R {
        &self.rng_after
    }

    /// Returns the estimators for this checkpoint.
    pub fn estimators(&self) -> &E {
        &self.estimators
    }

    /// Returns the histograms generated during this iteration.
    pub fn histograms(&self) -> &Vec<HistogramEstimators<T>> {
        &self.histograms
    }
}

struct Accumulator<T, A, E> {
    args: A,
    estimators: E,
    limits: Vec<HistogramSpecification<T>>,
    histograms: Vec<Vec<(T, T)>>,
}

impl<T, A, E> Arguments<T> for Accumulator<T, A, E>
where
    T: AddAssign + Float + FromPrimitive,
    A: Arguments<T>,
{
    fn weight(&mut self) -> T {
        self.args.weight()
    }

    fn x(&self) -> &[T] {
        self.args.x()
    }

    fn histo_filler(&mut self) -> Option<&mut dyn HistogramFiller<T>> {
        if self.histograms.is_empty() {
            None
        } else {
            Some(self)
        }
    }
}

impl<T, A, E> HistogramFiller<T> for Accumulator<T, A, E>
where
    T: AddAssign + Float + FromPrimitive,
    A: Arguments<T>,
{
    fn fill(&mut self, hist: usize, x: T, value: T) {
        if !value.is_finite() || value == T::zero() {
            return;
        }

        let left = self.limits[hist].left();
        let right = self.limits[hist].right();

        if x < left || x >= right {
            return;
        }

        let bins = T::from_usize(self.limits[hist].bins()).unwrap();
        let index = ((x - left) / (right - left) * bins).to_usize().unwrap();
        let value = value * self.args.weight();

        self.histograms[hist][index].0 += value;
        self.histograms[hist][index].1 += value * value;
    }
}

impl<T, A, E> Accumulator<T, A, E>
where
    T: AddAssign + Float + FromPrimitive,
    A: MutArguments<T>,
    E: Estimators<T> + Updateable<T>,
{
    fn new(args: A, estimators: E, limits: Vec<HistogramSpecification<T>>) -> Self {
        Self {
            args,
            estimators,
            histograms: limits
                .iter()
                .map(|l| vec![(T::zero(), T::zero()); l.bins()])
                .collect(),
            limits,
        }
    }

    fn perform_calls<R>(
        mut self,
        calls: usize,
        _total_calls: usize,
        rng: &mut R,
        int: &mut impl Integrand<T>,
    ) -> Checkpoint<T, R, E>
    where
        R: Clone + Rng,
        Standard: Distribution<T>,
    {
        let rng_before = rng.clone();

        for _ in 0..calls {
            self.args.x_mut().iter_mut().for_each(|x| *x = rng.gen());

            if self.args.check() {
                let value = int.call(&mut self);
                self.estimators.update(value);
            }
        }

        Checkpoint::new(
            rng_before,
            rng.clone(),
            self.estimators,
            self.limits,
            self.histograms,
        )
    }
}

pub(crate) fn make_chkpt<T, R, E>(
    args: impl MutArguments<T>,
    estimators: E,
    calls: usize,
    total_calls: usize,
    rng: &mut R,
    int: &mut impl Integrand<T>,
) -> Checkpoint<T, R, E>
where
    T: AddAssign + Float + FromPrimitive,
    E: Estimators<T> + Updateable<T>,
    R: Clone + Rng,
    Standard: Distribution<T>,
{
    Accumulator::new(args, estimators, int.histograms()).perform_calls(calls, total_calls, rng, int)
}
