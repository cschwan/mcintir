use core::panic;

use rand::distributions::Standard;
use rand::Rng;
use rand_pcg::Pcg64;
use rayon::prelude::*;

fn square(x: Vec<f64>) -> f64 {
    x[0] * x[0]
}

/// Usual sequential Monte Carlo integration.
fn integrate_sequentially(samples: u32, start: f64, end: f64) {
    // Random number generator
    let mut rnd = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let accumulate = (0..samples).into_iter().fold((0., 0.), |acc, _| {
        let x = vec![(end - start) * rnd.gen::<f64>() + start];
        let f = (end - start) * square(x);
        (acc.0 + f, acc.1 + f * f)
    });
    let mean = accumulate.0 / samples as f64;
    let std = (accumulate.1 / samples as f64 - mean * mean) / (samples as f64 - 1.0);

    println!("Sequential integration: {} +- {}", mean, std);
}

/// Integrate in parallel, but before doing so, collect all the inputs in order
/// to avoid having to clone the rng for each iteration.
/// This is potentially very bad on memory usage.
fn integrate_in_parallel_prepare_inputs(samples: u32, start: f64, end: f64) {
    // Random number generator
    let mut rnd = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

    let x: Vec<Vec<f64>> = (0..samples)
        .into_iter()
        .map(|_| vec![(end - start) * rnd.gen::<f64>() + start])
        .collect();
    let accumulate = x
        .into_par_iter()
        .map(|x| {
            let f = square(x);
            (f, f * f)
        })
        .reduce(
            || (0.0, 0.0),
            |(old_1, old_2), (new_1, new_2)| (old_1 + new_1, old_2 + new_2),
        );
    let mean = accumulate.0 / samples as f64;
    let std = (accumulate.1 / samples as f64 - mean * mean) / (samples as f64 - 1.0);

    println!(
        "Parallel integration with prepared inputs: {} +- {}",
        mean, std
    );
}

/// Integrate in parallel. Since generating a random number modifies the generator,
/// the generator has to be cloned on every iteration. Furthermore, it as to modified
/// for reproducibility with the sequential version. This constitutes a lot of modifications
/// per iteration.
fn integrate_in_parallel_clone_rng(samples: u32, start: f64, end: f64) {
    // Random number generator
    let rnd = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let accumulate = (0..samples)
        .into_par_iter()
        .enumerate()
        .map(|(i, _)| {
            // clone the random number generator and advance it.
            let x = rnd
                .clone()
                .sample_iter(&Standard)
                .skip(i * 1)
                .take(1)
                .map(|x: f64| (end - start) * x + start)
                .collect::<Vec<_>>();

            let f = (end - start) * square(x);
            (f, f * f)
        })
        .reduce(|| (0.0, 0.0), |(x1, x2), (y1, y2)| (x1 + y1, x2 + y2));

    let mean = accumulate.0 / samples as f64;
    let std = (accumulate.1 / samples as f64 - mean * mean) / (samples as f64 - 1.0);

    println!("Parallel integration with cloned rngs: {} +- {}", mean, std);
}

fn integrate_parallel_threads(samples: u32, start: f64, end: f64) {
    let cores = num_cpus::get();
    // Spread the work equally across the cores
    // In production code we would allow to specify the number of cores to use.
    let handles = (0..cores)
        .into_iter()
        .map(|i| {
            let mut range = (samples as f32 / cores as f32).ceil() as u32;
            let skip = range * i as u32 * 1; // 1 for variables per sample
            let computed_end = range * i as u32 + range;
            // On the last core it might be that not all samples are needed
            if computed_end > samples {
                range = computed_end - samples;
            }
            // println!("Computed samples in this thread: {} Start: {}, End: {}", range, skip, std::cmp::min(computed_end, samples)-1);
            // Each thread has its own random number generator.
            let mut rnd = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
            // Sync the rng between the threads
            for _ in 0..skip {
                let _ = rnd.gen::<f64>();
            }
            std::thread::spawn(move || {
                (0..range).into_iter().fold((0., 0.), |acc, _| {
                    let x = vec![(end - start) * rnd.gen::<f64>() + start];
                    let f = (end - start) * square(x);
                    (acc.0 + f, acc.1 + f * f)
                })
            })
        })
        .collect::<Vec<_>>();

    let accumulate = handles.into_iter().fold((0.0, 0.0), |acc, h| {
        let (m, s) = h.join().unwrap();
        (acc.0 + m, acc.1 + s)
    });

    let mean = accumulate.0 / samples as f64;
    let std = (accumulate.1 / samples as f64 - mean * mean) / (samples as f64 - 1.0);
    println!("Parallel integration with threads: {} +- {}", mean, std);
}

fn main() {
    let n = 1000_000;
    let flag = std::env::args()
        .skip(1)
        .next()
        .expect("Need 1 cmd-line arg.")
        .parse::<u32>()
        .unwrap();
    match flag {
        1 => integrate_sequentially(n, 0.0, 1.0),
        2 => integrate_in_parallel_prepare_inputs(n, 0.0, 1.0),
        3 => integrate_in_parallel_clone_rng(n, 0.0, 1.0),
        4 => integrate_parallel_threads(n, 0.0, 1.0),
        _ => panic!("Invalid flag"),
    }
}
