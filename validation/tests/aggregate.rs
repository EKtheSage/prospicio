//! Aggregate parity: compound distributions against brute-force
//! convolution (numpy), which is independent of Panjer's recursion and FFT.

use act_aggregate::{fft, panjer};
use act_prob::{Counting, Grid, NegativeBinomial, Poisson};
use act_validation::{check, reference};

fn check_method(method: &str) {
    let cases = reference("compound_convolution.csv");
    check(&cases, |c| {
        let n: Box<dyn Counting> = match c.get("frequency") {
            "poisson" => Box::new(Poisson::new(c.param("params", "lambda")).ok()?),
            "negative_binomial" => Box::new(
                NegativeBinomial::new(c.param("params", "r"), c.param("params", "beta")).ok()?,
            ),
            _ => return None,
        };
        let severity: Vec<f64> = c
            .get("severity")
            .split(';')
            .map(|p| p.parse().unwrap())
            .collect();
        let sev = Grid::new(1.0, severity).ok()?;
        // Compare the interior points; the last point carries the lumped tail.
        let points = c.number("points")? as usize;
        let index = c.number("index")? as usize;
        let (agg, _) = match method {
            "panjer" => panjer(n.as_ref(), &sev, points + 1),
            _ => fft(n.as_ref(), &sev, points + 1),
        }
        .ok()?;
        agg.probs().get(index).copied()
    });
}

#[test]
fn panjer_matches_convolution() {
    check_method("panjer");
}

#[test]
fn fft_matches_convolution() {
    check_method("fft");
}
