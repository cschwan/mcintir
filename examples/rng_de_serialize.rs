use rand_pcg::Pcg64;
use rand::Rng;

fn main() {

    let mut rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

    // Generate some random number
    let first_random_number: f64 = rng.gen();
    println!("x from first rng       : {}", first_random_number);

    // Serialize it
    let serialized = serde_json::to_string(&rng).unwrap();
    println!("Serialized RNG: {}", serialized);

    // Deserialize it
    let mut deserialized: rand_pcg::Lcg128Xsl64 = serde_json::from_str(&serialized).unwrap();

    // Check whether serialized and deserialized give the same number.
    let x_original: f64 = rng.gen();
    let x_deserial: f64 = deserialized.gen();

    println!("x from original rng    : {}", x_original);
    println!("x from deserialized rng: {}", x_deserial);
}