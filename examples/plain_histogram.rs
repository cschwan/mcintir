use mcintir::estimators::BasicEstimators;
use mcintir::integrators::plain::*;
use mcintir::{
    callbacks::SimpleCumulativeCallback,
    core::*,
    histograms::{HistogramAccumulator, HistogramSpecification},
};

use rand_pcg::Pcg64;

struct MyIntegrand;

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

fn main() {
    // Initialize the random number generator.
    let rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let integrand = MyIntegrand {};

    // define a callback function that does nothing
    let callback = SimpleCumulativeCallback {};

    let results = integrate(
        &integrand,
        &rng,
        &callback,
        1,
        &[100_000, 100_000, 100_000, 100_000],
    );

    let mut cumulative_iter = results
        .into_iter()
        .map(|c| (c.estimators().clone(), c.histograms()[0].clone().bins().clone()));

    let (first_est, first_hist) = cumulative_iter.next().unwrap();

    let cumulative = cumulative_iter.fold((first_est, first_hist), |(acc_e, acc_h), (e, h)| {
        (
            acc_e + e,
            acc_h
                .into_iter()
                .zip(h.into_iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>(),
        )
    });

    println!("\n--------------------------------------");
    println!("Final result: {: >0.8} \u{b1} {: >0.8}", cumulative.0.mean(), cumulative.0.std());
    println!("\nHistogram content:\n");
    for (bin, content) in cumulative.1.into_iter().enumerate() {
        println!("[Bin {}]: {: >0.8} \u{b1} {: >0.8}", bin, content.mean(), content.std());
    }
}
