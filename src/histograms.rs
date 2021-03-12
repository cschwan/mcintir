//! This module contains everything related to histograms.
use crate::core::estimators::BasicEstimators;
use crate::core::estimators::MeanVar;
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign};

/// Define a histogram to be filled by the Monte Carlo integrators.
///
/// Multi-dimensional histograms of `d` dimensions are supported.
/// They are defined by specifying the number of bins in each direction,
/// the `left` and `right` values of the ranges considered in each dimension,
/// as well as the labels associated to each dimension. Furthermode, the name
/// of the distribution is provided.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct HistogramSpecification<T>
where
    T: Copy,
{
    /// Store the lowest value of the range considered for each histogram dimension.
    left: Vec<T>,
    /// Store the highest value of the range considered for each histogram dimension.
    right: Vec<T>,
    /// Store the number of bins to be considered for each dimension in the histogram.
    bins: Vec<usize>,
    /// For each dimension, store the associated label.
    labels: Vec<String>,
    /// Store the name of the histogram.
    name: String,
}

impl<T> HistogramSpecification<T>
where
    T: Copy + Float + FromPrimitive + std::fmt::Debug,
{
    /// Construct a `d`-dimensional histogram, in which the ranges on the
    /// different axes from the `left` (inclusive) to right (exclusive) are
    /// subdivided into `bins` bins. The histogram has the name `name` and
    /// the `d` axes have labels as provided in `labels`. The dimensionality
    /// `d` of the histogram is implicitly given by the length of the vectors.
    pub fn new(
        left: Vec<T>,
        right: Vec<T>,
        bins: Vec<usize>,
        labels: Vec<String>,
        name: String,
    ) -> Self {
        // Consistency check of the input.
        debug_assert_eq!(left.len(), right.len());
        debug_assert_eq!(left.len(), bins.len());
        debug_assert_eq!(left.len(), labels.len());
        Self {
            left,
            right,
            bins,
            labels,
            name,
        }
    }

    /// Returns the left boundaries of the binned ranges.
    pub fn left(&self) -> &Vec<T> {
        &self.left
    }

    /// Returns the right boundaries of the binned ranges.
    pub fn right(&self) -> &Vec<T> {
        &self.right
    }

    /// Returns the number of bins in a given dimension.
    pub fn bins(&self) -> &Vec<usize> {
        &self.bins
    }

    /// Returns the labels associated to each dimension represented in the histogram.
    pub fn labels(&self) -> &Vec<String> {
        &self.labels
    }

    /// Returns the name of the histogram.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get an accumulator for the histogram.
    /// This is supposed to be sent to a computational unit.
    /// After all units have finished computing, the different
    /// accumulators will be combined.
    pub(crate) fn get_accumulator(&self) -> HistogramAccumulator<T> {
        HistogramAccumulator {
            sums: vec![(T::zero(), T::zero()); self.bins.iter().product()],
            specification: self.clone(),
            bin_cumulative: [1]
                .iter()
                .chain(self.bins()[..self.bins().len() - 1].iter())
                .scan(1, |acc, b| {
                    *acc *= b;
                    Some(*acc)
                })
                .collect(),
        }
    }

    /// Compute the index of the bin along the axis `dim` into which
    /// `x` belongs.
    fn compute_bin_in_1d(&self, dim: usize, x: T) -> Option<usize> {
        let left = self.left[dim];
        let right = self.right[dim];

        if x < left || x >= right {
            return None;
        }

        let bins = T::from_usize(self.bins[dim]).unwrap();
        let index = ((x - left) / (right - left) * bins).to_usize().unwrap();

        Some(index)
    }

    /// For each of the `observables` compute the index of the associated
    /// bin on the corresponding axis.
    fn compute_bins(&self, observables: &Vec<T>) -> Option<Vec<usize>> {
        let mut bin_indices = Vec::with_capacity(self.bins.len());
        for (dim, obs) in observables.iter().enumerate() {
            if let Some(bin) = self.compute_bin_in_1d(dim, *obs) {
                bin_indices.push(bin)
            } else {
                return None;
            }
        }

        // if we arrive here, the observables can all be represented in the histogram
        return Some(bin_indices);
    }
}

/// Intermediate representation of a histogram.
///
/// For each histogram, the sum and the sum of the squares of the values
/// filled into each bin are stored for each dimension.
///
/// This can be used to fill the histogram on different cores. Afer the
/// computations on each core finish, the different `HistogramAccumulator`s
/// can be combined using the `Add` trait before converting them to a HistogramEstimator,
/// that stores for each bin in each dimension the mean and the variance in terms of a
/// `MeanVar`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HistogramAccumulator<T>
where
    T: Copy,
{
    /// For each dimension, store the sum of all entries per bin.
    /// The multi-dimensional bin contents are stored in flattened form.
    sums: Vec<(T, T)>,

    /// Store the information about the accumulator
    specification: HistogramSpecification<T>,

    /// A technical component that we store in order to avoid recomputing it on each fill.
    bin_cumulative: Vec<usize>,
}

impl<T> HistogramAccumulator<T>
where
    T: Copy + Float + FromPrimitive + AddAssign + std::fmt::Debug,
{
    /// Add the `value` to the bin in the histogram corresponding to
    /// the provided `observables`.
    pub fn fill(&mut self, observables: &Vec<T>, value: T) {
        debug_assert!(observables.len() == self.specification.bins().len());
        if !value.is_finite() || value == T::zero() {
            return;
        }

        if let Some(bin_indices) = self.specification.compute_bins(&observables) {
            let bin: usize = bin_indices
                .into_iter()
                .zip(self.bin_cumulative.iter())
                .map(|(a, b)| a * b)
                .sum();
            self.sums[bin].0 += value;
            self.sums[bin].1 += value * value;
        } else {
            return;
        }
    }

    /// Convert an accumulator to a `HistogramEstimator`.
    pub fn to_histogram_estimator(self, calls: usize) -> HistogramEstimators<T> {
        HistogramEstimators::new(calls, self)
    }
}

impl<T> Add for HistogramAccumulator<T>
where
    T: Copy + PartialEq + AddAssign,
{
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        debug_assert!(self.specification == other.specification);

        for bin in 0..self.sums.len() {
            self.sums[bin].0 += other.sums[bin].0;
            self.sums[bin].1 += other.sums[bin].1;
        }

        self
    }
}

impl<T> AddAssign for HistogramAccumulator<T>
where
    T: Copy + AddAssign + PartialEq,
{
    fn add_assign(&mut self, other: Self) {
        debug_assert!(self.specification == other.specification);

        for bin in 0..self.sums.len() {
            self.sums[bin].0 += other.sums[bin].0;
            self.sums[bin].1 += other.sums[bin].1;
        }
    }
}

/// Estimators for histograms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistogramEstimators<T: Copy> {
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
    fn new(calls: usize, accumulator: HistogramAccumulator<T>) -> Self {
        Self {
            calls,
            limits: accumulator.specification,
            mean_var: accumulator
                .sums
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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_histogram_accumulator() {
        let specification = HistogramSpecification::<f64>::new(
            vec![0.0, 0.0, 0.0],
            vec![3.0, 2.0, 4.0],
            vec![3, 2, 4],
            vec![
                "dim_1".to_string(),
                "dim_2".to_string(),
                "dim_3".to_string(),
            ],
            "example histogram".to_string(),
        );

        assert_eq!(specification.name(), "example histogram");
        assert_eq!(
            specification.labels(),
            &vec![
                "dim_1".to_string(),
                "dim_2".to_string(),
                "dim_3".to_string()
            ]
        );
        assert_eq!(specification.left(), &vec![0.0, 0.0, 0.0]);
        assert_eq!(specification.right(), &vec![3.0, 2.0, 4.0]);

        let mut accumulator_1 = specification.get_accumulator();
        assert_eq!(&accumulator_1.bin_cumulative, &[1, 3, 6]);

        assert_eq!(&accumulator_1.sums, &vec![(0.0, 0.0); 24]);

        accumulator_1.fill(&vec![2.5, 1.5, 3.5], 1.0);
        assert_eq!(
            accumulator_1.sums[..accumulator_1.sums.len() - 1],
            vec![(0.0, 0.0); 23]
        );
        assert_eq!(accumulator_1.sums[23], (1.0, 1.0));

        let mut accumulator_2 = specification.get_accumulator();
        accumulator_2.fill(&vec![1.5, 1.5, 3.5], 2.0);
        assert_eq!(accumulator_2.sums[22], (2.0, 4.0));

        let accumulator_sum = accumulator_1 + accumulator_2;
        assert_eq!(accumulator_sum.sums[22], (2.0, 4.0));
        assert_eq!(accumulator_sum.sums[23], (1.0, 1.0));
    }
}
