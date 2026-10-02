//! Distribution parity against SciPy, and severity parity against
//! high-precision integration (mpmath).

use act_prob::{Distribution, Lognormal, Severity};
use act_validation::{check, reference};

#[test]
fn lognormal_matches_scipy() {
    let cases: Vec<_> = reference("distributions_scipy.csv")
        .into_iter()
        .filter(|c| c.get("distribution") == "lognormal")
        .collect();
    check(&cases, |c| {
        let d = Lognormal::new(c.param("params", "meanlog"), c.param("params", "sdlog")).ok()?;
        let arg = c.number("arg");
        match c.get("quantity") {
            "mean" => Some(d.mean()),
            "variance" => Some(d.variance()),
            "cdf" => Some(d.cdf(arg?)),
            "quantile" => d.quantile(arg?).ok(),
            _ => None,
        }
    });
}

#[test]
fn lognormal_severity_matches_integration() {
    let cases = reference("severity_mpmath.csv");
    check(&cases, |c| {
        let d = Lognormal::new(c.param("params", "meanlog"), c.param("params", "sdlog")).ok()?;
        let arg = c.number("arg")?;
        match c.get("quantity") {
            "lev" => Some(d.lev(arg)),
            "stop_loss" => Some(d.stop_loss(arg)),
            "layer" => Some(d.layer(arg, c.number("arg2")?)),
            _ => None,
        }
    });
}
