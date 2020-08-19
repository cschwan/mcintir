#![warn(clippy::all, clippy::cargo, clippy::nursery, clippy::pedantic)]
#![warn(missing_docs)]

//! The crate `mcintir` provides [Monte Carlo integration] routines, which allow to efficiently
//! approximate definite multi-dimensional [integrals]. This crate is a redesign and successor of
//! the library [`hep-mc`]. The pronunciation of `mcintir` is the same as of the name `MacIntyre` in
//! English.
//!
//! # Features
//!
//! This library was designed with the following features as essential in mind:
//!
//! - **Generic numeric type**. The numeric type used in this library is not fixed, but instead a
//! generic parameter, so that the integration routines can be used with either `f32`, `f64`, or a
//! custom numeric type that implements the `Float` trait from the `num-traits` crate.
//! - **Generic random number generator**. Every random number generator that implements the `Rng`
//! trait from the `rand` crate can be used with every integrator in this crate.
//! - **Reproducibility**. As far as the numeric type allows this, all results produced with
//! `mcintir` are completely reproducible, in the sense that the results only depend on the used
//! random number generator and the chosen seed. In particular, the results do not depend on the
//! number of cores the integrator was started with or how they are distributed on different cores.
//! - **Non-finite number filtering**. All integrators filter out non-finite numbers such as `inf`
//! or `nan`, which integrands sometimes produce in extreme regions of their integration domain due
//! to finite numerical precision. When this happens the result of the corresponding call is set to
//! zero to not destroy the integration and a counter is increased that keeps track of how often
//! this happened.
//! - **Zero tracking**. If your integrand returns zero, another counter will be increased to keep
//! track of the efficiency of the integration.
//! - **Checkpoints**. An issue in long-running integrations (days, weeks) is that the longer a run
//! is, the more likely it is for something to go wrong or not as planned. For example, a single
//! computer in a cluster crashes, one needs a higher sample size than was anticipated before the
//! run, ones needs to divide up the run time into several chucks so that run-time limitations can
//! be fulfilled, something went wrong in a particular part of the integration domain, etc. All of
//! these problem are solved, or mostly eleviated, when using checkpoints. They allow to save the
//! state of the integration at some point and to resume it or to replay the integration from
//! there, without a difference in the final results.
//! - **Histograms**. Often one is not only interested in the integral itself, but also in
//! integrals over a smaller integration (sub-)domains: histograms! They can be estimated along with
//! the full integral itself, without any additional integrand evaluations.
//!
//! # How do I get started?
//!
//! # What is ...?
//!
//! This section is a dictionary of terms that are used in this documentation. Given
//!
//! $$ I = \prod_{i=1}^d \int_0^1 \mathrm{d} x_i f(x_1, x_2, \ldots, x_d) $$
//!
//! we approximate $I$ using PLAIN Monte Carlo integration with
//!
//! $$ I \approx \frac{1}{N} \sum_{j=1}^N f \left( x_1^{(j)}, x_2^{(j)}, \ldots, x_d^{(j)} \right)
//! $$
//!
//! where for each $j$ the values of the arguments are uniformly distributed in $[0,1)$. We use the
//! following terms:
//!
//! - the number of *calls* or the *sample size* is $N$, which is the number of times the integrand
//! is evaluated. We assume that this is the expensive operation;
//! - the *integrand* is the function, $f(x_1, x_2, \ldots, x_d)$, that is being integrated,
//! - the number of *dimensions*, $d$, is number of dimensions of the integration domain,
//! - the *integral* is the (approximated) numeric value of the integral $I$,
//! - *efficiency* is the percentage of times the integrand evaluated to a value different from
//! zero. If your integrand returns zero very often, for example in 99% of the time, than the
//! efficiency is only 1%. This number should not be too small, otherwise it is possible than the
//! statistical uncertainties are underestimated.
//!
//! [Monte Carlo integration]: https://en.wikipedia.org/wiki/Monte_Carlo_integration
//! [integrals]: https://en.wikipedia.org/wiki/Integral
//! [`hep-mc`]: https://github.com/cschwan/hep-mc

pub mod core;
pub mod integrators;

pub use crate::core::*;
