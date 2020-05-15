use mcintir::core::*;
use mcintir::plain;
use rand_pcg::Pcg64;

struct MyIntegrand {}

impl Integrand<f64> for MyIntegrand {
    fn call(&mut self, args: &mut impl Arguments<f64>) -> f64 {
        args.x()[0]
    }

    fn dim(&self) -> usize {
        1
    }
}

fn main() {
    let _chkpt = plain::integrate(
        &mut MyIntegrand {},
        &mut Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96),
        &SimpleCallback {},
        &[1000],
    );
}
