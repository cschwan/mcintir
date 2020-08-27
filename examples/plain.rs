use mcintir::core::*;
use mcintir::integrators::plain::*;

use rand_pcg::Pcg64;
use rand::Rng;

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
    fn _generate_psp(&self, random_numbers: Vec<f64>) -> PhaseSpacePoint {
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
    fn call(&self, x: Vec<f64>) -> CallResult<f64> {
        let t = x;
        let PhaseSpacePoint { x, weight } = self._generate_psp(t);
        let val = x[0].powi(2) * weight;
        CallResult::new(val, vec![(x[0], val)])
    }

    /// The dimension of the integrand.
    ///
    /// This method is called by the integrator to decide how many random numbers to generate.
    fn dim(&self) -> usize {
        1
    }

    fn histograms_1d(&self) -> Vec<HistogramSpecification<f64>> {
        vec![HistogramSpecification::<f64>::new(1.0, 3.0, 10)]
    }
}

fn main() {
    // rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
    let callback = SimpleCumulativeCallback {};
    // Initialize the random number generator.
    let rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let integrand = MyIntegrand {};

    let check_points = integrate(
        &integrand,
        &rng,
        &callback,
        &[100_000, 100_000, 100_000, 100_000],
    );

    println!("\n---------------------------------");
    let exact = 26. / 3.;
    println!("Exact         : {:.4}", exact);
    let approx = check_points.last().unwrap().estimators().mean();
    let std = check_points.last().unwrap().estimators().std();
    let std_devs = (exact - approx).abs() / std;
    println!("Approx        : {:.4}", approx);
    let relative_error = (exact - approx).abs() / exact * 100.0;
    println!("Relative error: {:.4} %", relative_error);
    println!("Standard devs : {:.4}", std_devs);

    println!("\nHistogram:\n");

    let histo = &check_points.last().unwrap().histograms()[0];
    println!("\nHistogram mean: {:?}", histo.mean());
    println!("______________________");
    println!("\n| bin |  content     |");
    println!("______________________");
    let bw = 0.2;
    for (i, mean) in histo.bins().iter().enumerate() {
        println!("| {:2}  |  {:.8}  |", i + 1, mean.mean() / bw);
    }
    println!("______________________");

    let mut rng_original = check_points.last().unwrap().rng_after().clone();

    // Store the checkpoints to a file
    let serialized = serde_json::to_string(&check_points).unwrap();

    // Convert the JSON string back to a Point.
    let deserialized: Vec<Checkpoint<f64, rand_pcg::Lcg128Xsl64, PlainEstimators<f64>>> = serde_json::from_str(&serialized).unwrap();
    
    let mut rng_deserial = deserialized.last().unwrap().rng_after().clone();

    assert_eq!(rng_original.gen::<f64>(), rng_deserial.gen::<f64>());

    println!("{:#?}", deserialized);
}
