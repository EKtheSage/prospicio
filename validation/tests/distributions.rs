//! Distribution parity against SciPy.

use act_prob::{Distribution, Lognormal};
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
