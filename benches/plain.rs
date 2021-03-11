use criterion::{criterion_group, criterion_main, Criterion};

use mcintir::callbacks::SinkCallback;
use mcintir::core::*;
use mcintir::histograms::HistogramAccumulator;
use mcintir::integrators::plain::*;

use rand_pcg::Pcg64;

struct MyIntegrand;

impl Integrand<f64> for MyIntegrand {
    fn call(&self, args: &Vec<f64>, _: &mut Vec<HistogramAccumulator<f64>>) -> f64 {
        let x = 2.0 * args[0] - 1.0;
        x.abs()
    }

    fn dim(&self) -> usize {
        1
    }
}

fn benchmark_plain() {
    let callback = SinkCallback {};

    // initialize the random number generator
    let rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

    let integrand = MyIntegrand {};

    let _ = integrate(&integrand, &rng, &callback, 1, &[1_000_000, 1_000_000]);
}

fn criterion_plain_benchmark(c: &mut Criterion) {
    c.bench_function("plain univariate", |b| b.iter(|| benchmark_plain()));
}

criterion_group!(benches, criterion_plain_benchmark);
criterion_main!(benches);
