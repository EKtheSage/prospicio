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

/// The fields of `params`, such as `model=ppp;FQ=2;t=1000|2000;...`.
fn fields(c: &act_validation::Case) -> std::collections::HashMap<&str, &str> {
    c.get("params")
        .split(';')
        .filter_map(|kv| kv.split_once('='))
        .collect()
}

#[test]
fn collective_model_matches_r() {
    use act_aggregate::CollectiveModel;
    use act_prob::{PanjerClass, PiecewisePareto, Severity, evt::Gpd};
    let cases = reference("collective_r.csv");
    check(&cases, |c| {
        let f = fields(c);
        let num = |k: &str| -> Option<f64> { f.get(k)?.parse().ok() };
        let list = |k: &str| -> Option<Vec<f64>> {
            f.get(k)?.split('|').map(|v| v.parse().ok()).collect()
        };
        let freq = PanjerClass::from_mean_dispersion(num("FQ")?, num("dispersion")?).ok()?;
        let severity: Box<dyn Severity> = match *f.get("model")? {
            "ppp" => Box::new(PiecewisePareto::new(list("t")?, list("alpha")?).ok()?),
            "pgp" => Box::new(Gpd::riegel(num("t")?, num("alpha_ini")?, num("alpha_tail")?).ok()?),
            _ => return None,
        };
        let model = CollectiveModel::new(freq, severity);
        let arg = |k: &str| -> Option<f64> {
            match c.get(k) {
                "inf" => Some(f64::INFINITY),
                s => s.parse().ok(),
            }
        };
        match c.get("quantity") {
            "excess_frequency" => Some(model.excess_frequency(arg("arg")?)),
            "layer_mean" => Some(model.layer_mean(arg("arg")?, arg("arg2")?)),
            "layer_variance" => Some(model.layer_variance(arg("arg")?, arg("arg2")?)),
            _ => None,
        }
    });
}
