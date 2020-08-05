use rand::distributions::Standard;
use rand::Rng;
use rand_pcg::Pcg64;
use rayon::prelude::*;

fn main() {
    // The number of iterations in the integration
    const ITERATIONS: usize = 4;
    // The number of calls per iteration
    const CALLS: usize = 100;

    // Setup the random number generator
    let rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

    // Collect all the random numbers sequentially at once
    // We clone to maintain a copy of the initial random number generator.
    let v: Vec<f64> = rng.clone().sample_iter(&Standard).take(ITERATIONS*CALLS).collect();

    // Now collect the random numbers iteration by iteration,
    // where the calls per iteration are performed in parallel using rayon.

    // We store the random numbers per iteration here (only for comparison with the sequential approach from above)
    let mut iteration_results = vec![];

    // Loop sequentially over iterations
    for iteration in 0..ITERATIONS {
        let iteration_result = (0..CALLS)
            .into_par_iter()
            .map(|call| {
                rng.clone()
                    .sample_iter(&Standard)
                    .skip(iteration * CALLS + call)
                    .next()
                    .unwrap()
            })
            .collect::<Vec<f64>>();
        // Store the random numbers per iteration (later do more here, resize grids in VEGAS etc)
        iteration_results.push(iteration_result);
    }

    // Flatten the results per iteration and compare to the sequential result.
    let v_clone = iteration_results
        .into_iter()
        .flatten()
        .collect::<Vec<f64>>();

    for (a, b) in v.iter().zip(v_clone.iter()) {
        assert_eq!(a, b);
        // println!("{} --- {}", a, b);
    }
}
