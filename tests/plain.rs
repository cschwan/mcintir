use mcintir::core::*;
use mcintir::integrators::plain;

use assert_approx_eq::assert_approx_eq;
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

fn compare_checkpoints(
    chkpt: &plain::PlainCheckpoint<f64, rand_pcg::Lcg128Xsl64>,
    target: &plain::PlainCheckpoint<f64, rand_pcg::Lcg128Xsl64>,
) {
    // TOLERANCE to use in floating point comparisons.
    const TOLERANCE: f64 = 1e-15;

    // we requested 1000 calls
    assert_eq!(chkpt.estimators().calls(), target.estimators().calls());

    // check the mean
    assert_approx_eq!(
        chkpt.estimators().mean(),
        target.estimators().mean(),
        TOLERANCE
    );

    // check the variance
    assert_approx_eq!(
        chkpt.estimators().var(),
        target.estimators().var(),
        TOLERANCE
    );

    // there is one histogram
    assert_eq!(chkpt.histograms().len(), 1);

    assert_approx_eq!(
        chkpt.histograms()[0].mean(),
        target.histograms()[0].mean(),
        TOLERANCE
    );
    assert_approx_eq!(
        chkpt.histograms()[0].var(),
        target.histograms()[0].var(),
        TOLERANCE
    );

    let bins = &chkpt.histograms()[0].bins();
    let bins_target = &target.histograms()[0].bins();

    assert_eq!(bins.len(), 10);

    assert_approx_eq!(bins[0].mean(), bins_target[0].mean(), TOLERANCE);
    assert_approx_eq!(bins[1].mean(), bins_target[1].mean(), TOLERANCE);
    assert_approx_eq!(bins[2].mean(), bins_target[2].mean(), TOLERANCE);
    assert_approx_eq!(bins[3].mean(), bins_target[3].mean(), TOLERANCE);
    assert_approx_eq!(bins[4].mean(), bins_target[4].mean(), TOLERANCE);
    assert_approx_eq!(bins[5].mean(), bins_target[5].mean(), TOLERANCE);
    assert_approx_eq!(bins[6].mean(), bins_target[6].mean(), TOLERANCE);
    assert_approx_eq!(bins[7].mean(), bins_target[7].mean(), TOLERANCE);
    assert_approx_eq!(bins[8].mean(), bins_target[8].mean(), TOLERANCE);
    assert_approx_eq!(bins[9].mean(), bins_target[9].mean(), TOLERANCE);

    assert_approx_eq!(bins[0].var(), bins_target[0].var(), TOLERANCE);
    assert_approx_eq!(bins[1].var(), bins_target[1].var(), TOLERANCE);
    assert_approx_eq!(bins[2].var(), bins_target[2].var(), TOLERANCE);
    assert_approx_eq!(bins[3].var(), bins_target[3].var(), TOLERANCE);
    assert_approx_eq!(bins[4].var(), bins_target[4].var(), TOLERANCE);
    assert_approx_eq!(bins[5].var(), bins_target[5].var(), TOLERANCE);
    assert_approx_eq!(bins[6].var(), bins_target[6].var(), TOLERANCE);
    assert_approx_eq!(bins[7].var(), bins_target[7].var(), TOLERANCE);
    assert_approx_eq!(bins[8].var(), bins_target[8].var(), TOLERANCE);
    assert_approx_eq!(bins[9].var(), bins_target[9].var(), TOLERANCE);
}

#[test]
fn plain_iteration() {
    // TOLERANCE to use in floating point comparisons.
    const TOLERANCE: f64 = 1e-15;
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

#[test]
fn test_plain_serialization() {
    // The number of calls in the iteration
    const CALLS: usize = 1_000;
    let iterations = &[CALLS, CALLS, CALLS, CALLS, CALLS];

    let rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

    // Perform the integration over all the iterations and store the checkpoints
    let check_points = plain::integrate(
        &MyIntegrand {},
        &rng.clone(),
        &SimpleCallback {},
        iterations,
    );

    // Consistency check
    assert_eq!(check_points.len(), iterations.len());

    // Clone the final result and store it as a target
    let final_target = check_points.iter().last().unwrap().clone();

    // Restart the integration from each checkpoint and make sure the final result agrees with the
    // one computed above.
    check_points
        .into_iter()
        .enumerate()
        .for_each(|(index, cp)| {
            // Resume the iteration from the given checkpoint
            let resumed = plain::resume_integration_from_checkpoints(
                &MyIntegrand {},
                vec![cp],
                &SimpleCallback {},
                &vec![CALLS; iterations.len() - index - 1],
            )
            .into_iter()
            .last()
            .unwrap();
            compare_checkpoints(&resumed, &final_target);
        });
}
