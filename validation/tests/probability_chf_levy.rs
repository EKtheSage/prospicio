//! Laws of Lévy processes from their characteristic function against
//! `aggregate` 2.1.0's `AggregateCHF` (`validation/scripts/aggregate_chf_levy.py`,
//! #780): the normal inverse Gaussian, variance gamma and CGMY tempered
//! stable laws on the grid rule A chooses.

use prospicio_prob::Distribution;
use prospicio_prob::chf::{ChfDistribution, ChfLaw, ChfOptions, invert};
use prospicio_prob::levy::{NormalInverseGaussian, TemperedStable, VarianceGamma};
use prospicio_validation::{check, reference};
use std::collections::HashMap;

#[test]
fn chf_levy_laws_match_aggregate() {
    let cases = reference("chf_levy_aggregate.csv");
    let mut done: HashMap<String, ChfDistribution> = HashMap::new();
    check(&cases, |c| {
        let v = |name: &str| c.param("params", name);
        let key = format!("{}|{}", c.get("distribution"), c.get("params"));
        let d = done.entry(key).or_insert_with(|| {
            let law = match c.get("distribution") {
                "normal_inverse_gaussian" => ChfLaw::NormalInverseGaussian(
                    NormalInverseGaussian::new(v("alpha"), v("beta"), v("delta"), v("mu")).unwrap(),
                ),
                "variance_gamma" => ChfLaw::VarianceGamma(
                    VarianceGamma::new(v("sigma"), v("nu"), v("theta"), v("mu")).unwrap(),
                ),
                "tempered_stable" => ChfLaw::TemperedStable(
                    TemperedStable::new(v("C"), v("G"), v("M"), v("Y"), v("mu")).unwrap(),
                ),
                other => unreachable!("{other}"),
            };
            invert(&law, &ChfOptions::with_log2(v("log2") as u32)).unwrap()
        });
        let r = d.report();
        match c.get("quantity") {
            "step" => Some(d.step()),
            "x_min" => Some(d.x_min()),
            "truncation_error" => Some(r.truncation_error),
            "outside_below" => Some(r.outside_below),
            "outside_above" => Some(r.outside_above),
            "sum" => Some(r.total_mass),
            "mean" => Some(d.mean()),
            "sd" => Some(d.std_dev()),
            "quantile" => d.quantile(c.number("arg")?).ok(),
            "mass" => Some(d.masses()[c.number("arg")? as usize]),
            _ => None,
        }
    });
}
