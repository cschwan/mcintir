use mcintir::core::*;
use mcintir::integrators::plain;

use rand_pcg::Pcg64;
use assert_approx_eq::assert_approx_eq;
use rand::Rng;
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

struct MyIntegrand {}

impl Integrand<f64> for MyIntegrand {
    // the integral of the function called `abs`:
    // int_0^1 dx |2*x-1| = int^1_0.5 dx (2x-1) + int^0.5_0 (1-2x)
    //                    = [x^2-x]^1_0.5 + [x-x^2]^0.5_0
    //                    = 0 - (0.25-0.5) + (0.5-0.25) - 0
    //                    = 2*(0.5-0.25)
    //                    = 1-0.5
    //                    = 0.5
    fn call(&self, args: Vec<f64>) -> CallResult<f64> {
        let x = 2.0 * args[0] - 1.0;
        let y = x.abs();

        CallResult::new(y, vec![(x, y)])
    }

    fn dim(&self) -> usize {
        1
    }

    fn histograms_1d(&self) -> Vec<HistogramSpecification<f64>> {
        vec![HistogramSpecification::new(-1.0, 1.0, 10)]
    }
}

#[test]
fn plain_iteration() {

    // TOLERANCE to use in floating point comparisons.
    const TOLERANCE: f64 = 1e-16;

    // The number of calls in the iteration
    const CALLS: usize = 1_000;

    let mut rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let rng_start = rng.clone();
    let chkpt = plain::integrate(&MyIntegrand {}, &rng, &SimpleCallback {}, &[CALLS]).remove(0);

    
    // compare random number generators before iteration
    assert_eq_rng(chkpt.rng_before(), &rng_start);

    // compare random number generators after iteration
    for _ in 0..CALLS {
        let _: f64 = rng.gen();
    }
    assert_eq_rng(chkpt.rng_after(), &rng);

    // we requested 1000 calls
    assert_eq!(chkpt.estimators().calls(), 1000);

    // check the mean
    assert_approx_eq!(chkpt.estimators().mean(), 4.891001827394124e-1, TOLERANCE);

    // check the variance
    assert_approx_eq!(chkpt.estimators().var(), 8.704232037144878e-5, TOLERANCE);

    // there is one histogram
    assert_eq!(chkpt.histograms().len(), 1);

    assert_approx_eq!(
        chkpt.histograms()[0].mean(),
        4.8910018273941236e-1,
        TOLERANCE
    );
    assert_approx_eq!(
        chkpt.histograms()[0].var(),
        2.9477356489145525e-4,
        TOLERANCE
    );

    let bins = &chkpt.histograms()[0].bins();

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

    assert_approx_eq!(bins[0].var(), 6.962028200632969e-5, TOLERANCE);
    assert_approx_eq!(bins[1].var(), 4.4835245076113314e-5, TOLERANCE);
    assert_approx_eq!(bins[2].var(), 1.7959466185901967e-5, TOLERANCE);
    assert_approx_eq!(bins[3].var(), 8.751209965559931e-6, TOLERANCE);
    assert_approx_eq!(bins[4].var(), 8.168485255333883e-7, TOLERANCE);
    assert_approx_eq!(bins[5].var(), 1.3603316021008941e-6, TOLERANCE);
    assert_approx_eq!(bins[6].var(), 8.955699997281554e-6, TOLERANCE);
    assert_approx_eq!(bins[7].var(), 2.4509156464831765e-5, TOLERANCE);
    assert_approx_eq!(bins[8].var(), 4.2660976883510006e-5, TOLERANCE);
    assert_approx_eq!(bins[9].var(), 7.530434818429274e-5, TOLERANCE);
}
