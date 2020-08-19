//! Core functionality. You don't need to import this modules since all it's public members are
//! part of the crate namespace.

use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::ops::{Add, AddAssign};

/// The result of a call to an integrand.
/// 
/// It contains the weight of the integrand for the given phase space point 
/// and for each one-dimensional histogram (if requested, else `None`) contains 
/// both the value of the observable (used to determine the bin in the histogram)
/// and the weight to be filled into the bin. 
/// 
/// The weight to fill into the bin is not necessarily the weight of the integrand 
/// in order to allow simple counting (by simply filling a 1). 
#[derive(Debug)]
pub struct CallResult<T> {
    /// Contains the value of the integrand evaluated at a phase space point.
    pub val: T,
    /// For each histogram (if present) store both the value of the observable and the weight to be filled in the corresponding bin.
    pub observables_1d: Option<Vec<(T, T)>>,
}

impl<T> CallResult<T> {
    /// Create a new call result.
    pub const fn new(val: T, observables_1d: Option<Vec<(T, T)>>) -> Self {
        Self {
            val,
            observables_1d,
        }
    }
}

/// Trait which every integrand must implement.
pub trait Integrand<T>: Send + Sync {
    /// Calculates the value of the integrand from a point `x` on the
    /// hypercube which has as many random numbers as specified by `dim()`.
    fn call(&self, x: Vec<T>) -> CallResult<T>;

    /// Returns how many random numbers are needed by the integrand.
    fn dim(&self) -> usize;

    /// Defines the one-dimensional histograms to be created
    /// 
    /// If histograms are requested, their corresponding observables and bin contents 
    /// have to be computed during while calling `call` and returned as part of the result.
    fn histograms_1d(&self) -> Option<Vec<HistogramSpecification<T>>> {
        None
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

/// Everything that needs to be updated.
pub trait Updateable<T> {
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

/// HistogramEstimators that are temporary
#[derive(Debug, std::cmp::PartialEq, Clone)]
pub struct HistogramEstimatorsAccumulator<T> {
    bin_contents: Vec<(T, T)>,
    limits: HistogramSpecification<T>,
}

impl<T: Float> HistogramEstimatorsAccumulator<T> {
    /// With empty bins
    pub fn with_empty_bins(bins: usize, limits: HistogramSpecification<T>) -> Self {
        Self {
            bin_contents: vec![(T::zero(), T::zero()); bins],
            limits,
        }
    }

    /// Fill specific weights
    pub fn new(bin_contents: Vec<(T, T)>, limits: HistogramSpecification<T>) -> Self {
        Self {
            bin_contents,
            limits,
        }
    }

    /// Get the bin contents
    pub fn bins(&self) -> &Vec<(T, T)> {
        &self.bin_contents
    }
}

impl<T: Float> Add for HistogramEstimatorsAccumulator<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            bin_contents: self
                .bin_contents
                .into_iter()
                .zip(other.bin_contents)
                .map(|(t1, t2)| (t1.0 + t2.0, t1.1 + t2.1))
                .collect(),
            limits: self.limits,
        }
    }
}

impl<T> HistogramFiller<T> for HistogramEstimatorsAccumulator<T>
where
    T: AddAssign + Float + FromPrimitive,
{
    fn fill(&mut self, x: T, value: T) {
        if !value.is_finite() || value == T::zero() {
            return;
        }

        let left = self.limits.left();
        let right = self.limits.right();

        if x < left || x >= right {
            return;
        }

        let bins = T::from_usize(self.limits.bins()).unwrap();
        let index = ((x - left) / (right - left) * bins).to_usize().unwrap();

        self.bin_contents[index].0 += value;
        self.bin_contents[index].1 += value * value;
    }
}

/// Estimators for histograms.
#[derive(Deserialize, Serialize, Debug)]
pub struct HistogramEstimators<T> {
    limits: HistogramSpecification<T>,
    calls: usize,
    mean_var: Vec<MeanVar<T>>,
}

impl<T: std::ops::Add<Output = T>> Add for HistogramEstimators<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            limits: self.limits,
            calls: self.calls + other.calls,
            mean_var: self
                .mean_var
                .into_iter()
                .zip(other.mean_var)
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>(),
        }
    }
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
    /// New histogram estimator
    pub fn new(calls: usize, limits: HistogramSpecification<T>, bins: Vec<(T, T)>) -> Self {
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
    /// Get mean
    fn mean(&self) -> T {
        self.mean_var
            .iter()
            .fold(T::zero(), |mean, x| mean + x.mean())
    }
    /// Get variance
    fn var(&self) -> T {
        self.mean_var.iter().fold(T::zero(), |var, x| var + x.var())
    }
}

/// Trait whose implementers lets one fill histograms.
pub trait HistogramFiller<T> {
    /// In the histogram with index `hist` fill the bin that contains `x` with `value`.
    fn fill(&mut self, x: T, value: T);
}

/// Everything Monte Carlo integrators need to know about histograms.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct HistogramSpecification<T> {
    left: T,
    right: T,
    bins: usize,
    name: String,
    x_label: String,
    y_label: String,
}

impl<T: Copy + Float + AddAssign + FromPrimitive> HistogramSpecification<T> {
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

    /// From the specification, construct and empty histogram estimator
    fn get_empty_accumulator(&self) -> HistogramEstimatorsAccumulator<T>
    where
        T: Float,
    {
        HistogramEstimatorsAccumulator::with_empty_bins(self.bins(), self.clone())
    }
}

/// A checkpoint saves the state of a generator after an iteration.
/// Checkpoints can be used to restart or resume iterations.
#[derive(Deserialize, Serialize, Debug)]
pub struct Checkpoint<T, R, E> {
    rng_before: R,
    rng_after: R,
    estimators: E,
    histograms: Option<Vec<HistogramEstimators<T>>>, //Vec<HistogramEstimators<T>>
}

impl<T, R, E> Checkpoint<T, R, E>
where
    T: AddAssign + Float + FromPrimitive,
    E: Estimators<T>,
{
    /// Create a new checkpoint
    pub(crate) fn new(
        rng_before: R,
        rng_after: R,
        estimators: E,
        // limits: Option(Vec<HistogramSpecification<T>>),
        histograms: Option<Vec<HistogramEstimators<T>>>, // Vec<Vec<(T, T)>>
    ) -> Self {
        let _calls = estimators.calls();
        Self {
            rng_before,
            rng_after,
            estimators,
            // limits,
            histograms,
        }
    }

    /// Returns the random number generator that was used to generate this checkpoint.
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
    pub fn histograms(&self) -> &Option<Vec<HistogramEstimators<T>>> {
        &self.histograms
    }
}

/// Accumulate results from different threads.
#[derive(Debug, Clone)]
pub struct Accumulator<T, E> {
    estimators: E,
    limits_1d: Option<Vec<HistogramSpecification<T>>>,
    histograms_1d: Option<Vec<HistogramEstimatorsAccumulator<T>>>,
}

impl<T, E> Add for Accumulator<T, E>
where
    T: Float,
    E: std::ops::Add<Output = E>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let histo_sum = match self.histograms_1d {
            None => {
                assert!(other.histograms_1d.is_none());
                None
            }
            Some(histos) => {
                assert!(other.histograms_1d.is_some());
                Some(
                    histos
                        .into_iter()
                        .zip(other.histograms_1d.unwrap())
                        .map(|(a, b)| a + b)
                        .collect(),
                )
            }
        };

        Self {
            estimators: self.estimators + other.estimators,
            histograms_1d: histo_sum,
            limits_1d: self.limits_1d,
        }
    }
}

impl<T, E> Accumulator<T, E>
where
    T: AddAssign + Float + FromPrimitive + std::fmt::Debug,
    // A: MutArguments<T>,
    E: Estimators<T> + Default + Updateable<T> + std::fmt::Debug,
{
    /// Create new accumulator
    fn new(
        estimators: E,
        histograms_1d: Option<Vec<HistogramEstimatorsAccumulator<T>>>,
        limits_1d: Option<Vec<HistogramSpecification<T>>>,
    ) -> Self {
        Self {
            estimators,
            histograms_1d,
            limits_1d,
        }
    }

    /// Create empty accumulator
    pub fn empty(limits_1d: Option<Vec<HistogramSpecification<T>>>) -> Self {
        Self {
            estimators: E::default(),
            histograms_1d: match &limits_1d {
                &None => None,
                Some(limits) => Some(limits.iter().map(|l| l.get_empty_accumulator()).collect()),
            },
            limits_1d,
        }
    }

    /// Get estimators stored in the accumulator.
    pub fn get_estimators(&self) -> &E {
        &self.estimators
    }

    /// Get histograms stored in the accumulator.
    pub fn get_histograms_1d(&self) -> &Option<Vec<HistogramEstimatorsAccumulator<T>>> {
        &self.histograms_1d
    }

    /// Get empty copy of the accumulator.
    pub fn get_empty_accumulator(&self) -> Self {
        Self::empty(self.limits_1d.clone())
    }

    /// Add call result to the accumulator.
    pub fn add_call_result(&mut self, call_result: CallResult<T>) -> Self {
        let CallResult {
            val,
            observables_1d,
        } = call_result;

        let mut estimators = E::default();
        estimators.update(val);
        Self::new(
            estimators,
            match &self.limits_1d {
                None => None,
                Some(limits) => Some(
                    limits
                        .iter()
                        .zip(observables_1d.unwrap())
                        .map(|(h, (o, weight))| {
                            let mut hist = h.get_empty_accumulator();
                            hist.fill(o, weight);
                            hist
                        })
                        .collect(),
                ),
            },
            self.limits_1d.clone(),
        )
    }
}

/// Trait for implementing callbacks for iterative MC algorithms.
pub trait Callback<T, R, E> {
    /// This method is called after each successfully finished iteration and may print information
    /// about it.
    fn print(&self, chkpts: &[Checkpoint<T, R, E>]);
}

/// Implements `Callback` and simply prints the results of each iteration.
pub struct SimpleCallback {}

impl<T, R, E> Callback<T, R, E> for SimpleCallback
where
    T: AddAssign + Display + Float + FromPrimitive,
    E: Estimators<T>,
{
    fn print(&self, chkpts: &[Checkpoint<T, R, E>]) {
        let iteration = chkpts.len() - 1;
        let chkpt = chkpts.last().unwrap();
        let estimators = chkpt.estimators();

        println!("iteration {} finished.", iteration);
        println!(
            "this iteration N={} E={} \u{b1} {}",
            estimators.calls(),
            estimators.mean(),
            estimators.std()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_var_add() {
        let mv_1 = MeanVar::<f64>::new(1.1, 0.5);
        let mv_2 = MeanVar::<f64>::new(5.3, 1.2);
        let sum = mv_1 + mv_2;

        assert_eq!(sum.clone().mean(), 6.4);
        assert_eq!(sum.clone().var(), 1.7);
        assert_eq!(sum.std(), 1.7.sqrt());
    }

    #[test]
    fn test_mean_var_add_assign() {
        let mut mv_1 = MeanVar::<f64>::new(1.1, 0.5);
        mv_1 += MeanVar::<f64>::new(5.3, 1.2);

        assert_eq!(mv_1.clone().mean(), 6.4);
        assert_eq!(mv_1.clone().var(), 1.7);
        assert_eq!(mv_1.std(), 1.7.sqrt());
    }

    #[test]
    fn add_histogram_estimators() {
        let hs = HistogramSpecification::with_labels(0.0, 10.0, 5, "dummy_histogram", "x", "y");

        let mut h1 = hs.get_empty_accumulator();
        let mut h2 = hs.get_empty_accumulator();
        h1.fill(1.1, 2.0);
        h2.fill(3.2, 4.0);
        let sum = h1 + h2;
        let bin_contents = sum.bins();
        assert_eq!(bin_contents[0].0, 2.0);
        assert_eq!(bin_contents[0].1, 4.0);
        assert_eq!(bin_contents[1].0, 4.0);
        assert_eq!(bin_contents[1].1, 16.0);
    }
}
