use mcintir::estimators::BasicEstimators;
use mcintir::integrators::plain::*;
use mcintir::{callbacks::SimpleCumulativeCallback, core::*, histograms::HistogramAccumulator};

use rand_pcg::Pcg64;

struct MyIntegrand;

/// A phase space point together with its weight.
struct PhaseSpacePoint {
    x: Vec<f64>,
    weight: f64,
}

/// The integrand is the function x^2 (c.f. call method)
impl MyIntegrand {
    /// Generate a phase space point from the random numbers, the integrator provides.
    /// In this case, we integrate from  x=1 to x=3.
    fn _generate_psp(&self, random_numbers: &[f64]) -> PhaseSpacePoint {
        assert!(random_numbers.len() == 1);
        PhaseSpacePoint {
            x: random_numbers
                .into_iter()
                .map(|r| 2. * r + 1.)
                .collect::<Vec<_>>(),
            weight: 2.,
        }
    }
}

/// Integrating the function x^2
/// from x=1 to x=3
/// Which gives the result: 26/3
impl Integrand<f64> for MyIntegrand {
    /// Call the integrand with a set of random numbers `x` provided by the integrator.
    ///
    /// The first step is to construct a phase space point from the random numbers using the
    /// phase space generator implemented for the integrand. After that, the integrand is
    /// evaluated on the generated phase space point.
    fn call(&self, x: &[f64], _: &mut Vec<HistogramAccumulator<f64>>) -> f64 {
        let t = x;
        let PhaseSpacePoint { x, weight } = self._generate_psp(t);
        let val = x[0].powi(2) * weight;
        val
    }

    /// The dimension of the integrand.
    ///
    /// This method is called by the integrator to decide how many random numbers to generate.
    fn dim(&self) -> usize {
        1
    }
}

fn main() {
    // Initialize the random number generator.
    let rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let integrand = MyIntegrand {};

    // define a callback function that does nothing
    let callback = SimpleCumulativeCallback {};

    let results_per_iteration = integrate(
        &integrand,
        &rng,
        &callback,
        4,
        &[100_000, 100_000, 100_000, 100_000],
    );

    // combine
    let final_result = results_per_iteration
        .into_iter()
        .map(|cp| cp.estimators().clone())
        .fold(PlainEstimators::default(), |acc, r| acc + r);

    println!("\n{:?} +- {:?}", final_result.mean(), final_result.std());
}
