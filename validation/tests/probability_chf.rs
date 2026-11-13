//! Distributions from their characteristic function against `aggregate`
//! 2.1.0's `AggregateCHF` (#779).

use prospicio_prob::Distribution;
use prospicio_validation::{check, reference};

/// Distributions by inverting their characteristic function against
/// `aggregate` 2.1.0's `AggregateCHF` (`validation/scripts/aggregate_chf.py`):
/// the grid rule A chooses (or rule C's given bucket), the truncation and
/// Chernoff outside masses, the masses' sum, mean, sd and quantiles, and the
/// masses at 257 indices, untilted (#779).
#[test]
fn chf_inversion_matches_aggregate() {
    use prospicio_prob::chf::{ChfDistribution, ChfLaw, ChfOptions, invert};
    use prospicio_prob::{Gamma, InverseGaussian, Normal, Tweedie};
    use std::collections::HashMap;

    fn law(kind: &str, v: &dyn Fn(&str) -> f64) -> ChfLaw {
        match kind {
            "gamma" => ChfLaw::Gamma(Gamma::new(v("shape"), v("scale")).unwrap()),
            "normal" => ChfLaw::Normal(Normal::new(v("mu"), v("sigma")).unwrap()),
            "inverse_gaussian" => {
                ChfLaw::InverseGaussian(InverseGaussian::new(v("mean"), v("shape")).unwrap())
            }
            "tweedie" => {
                ChfLaw::Tweedie(Tweedie::new(v("mean"), v("dispersion"), v("power")).unwrap())
            }
            _ => unreachable!("{kind}"),
        }
    }

    let cases = reference("chf_aggregate.csv");
    let mut done: HashMap<String, ChfDistribution> = HashMap::new();
    check(&cases, |c| {
        let v = |name: &str| c.param("params", name);
        let key = format!("{}|{}", c.get("distribution"), c.get("params"));
        let d = done.entry(key).or_insert_with(|| {
            let l = match c.get("distribution") {
                base @ ("affine_gamma" | "affine_tweedie") => {
                    let b = |name: &str| c.param("params", &format!("base_{name}"));
                    ChfLaw::affine(law(&base[7..], &b), v("scale"), v("shift")).unwrap()
                }
                kind => law(kind, &v),
            };
            let bs = v("bs");
            let options = ChfOptions {
                log2: v("log2") as u32,
                step: (bs > 0.0).then_some(bs),
                ..ChfOptions::default()
            };
            invert(&l, &options).unwrap()
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
