//! The tail value at risk of a law inverted from its characteristic
//! function against `aggregate` 2.1.0's `AggregateCHF.tvar`
//! (`validation/scripts/aggregate_chf_tvar.py`, #779, #780): light laws,
//! Tweedies, the Lévy laws and stable laws tilted and split, on the grid
//! rule A chooses.

use prospicio_prob::chf::{ChfDistribution, ChfLaw, ChfOptions, Tilt, invert};
use prospicio_prob::levy::{NormalInverseGaussian, TemperedStable, TweediePower, VarianceGamma};
use prospicio_prob::{Distribution, Gamma, InverseGaussian, Normal, Stable, StableParameterization};
use prospicio_validation::{check, reference};
use std::collections::HashMap;

#[test]
fn chf_tvar_matches_aggregate() {
    let cases = reference("chf_tvar_aggregate.csv");
    let mut done: HashMap<String, ChfDistribution> = HashMap::new();
    check(&cases, |c| {
        let v = |name: &str| c.param("params", name);
        let key = format!("{}|{}", c.get("distribution"), c.get("params"));
        let d = done.entry(key).or_insert_with(|| {
            let law = match c.get("distribution") {
                "gamma" => ChfLaw::Gamma(Gamma::new(v("shape"), v("scale")).unwrap()),
                "normal" => ChfLaw::Normal(Normal::new(v("mu"), v("sigma")).unwrap()),
                "inverse_gaussian" => {
                    ChfLaw::InverseGaussian(InverseGaussian::new(v("mean"), v("shape")).unwrap())
                }
                "tweedie_power" => ChfLaw::TweediePower(
                    TweediePower::new(v("power"), v("mean"), v("dispersion")).unwrap(),
                ),
                "normal_inverse_gaussian" => ChfLaw::NormalInverseGaussian(
                    NormalInverseGaussian::new(v("alpha"), v("beta"), v("delta"), v("mu")).unwrap(),
                ),
                "variance_gamma" => ChfLaw::VarianceGamma(
                    VarianceGamma::new(v("sigma"), v("nu"), v("theta"), v("mu")).unwrap(),
                ),
                "tempered_stable" => ChfLaw::TemperedStable(
                    TemperedStable::new(v("C"), v("G"), v("M"), v("Y"), v("mu")).unwrap(),
                ),
                "stable" => {
                    let p = if v("parameterization") == 1.0 {
                        StableParameterization::S0
                    } else {
                        StableParameterization::S1
                    };
                    ChfLaw::Stable(
                        Stable::new(v("alpha"), v("beta"), v("scale"), v("loc"), p).unwrap(),
                    )
                }
                other => unreachable!("{other}"),
            };
            let options = ChfOptions {
                log2: v("log2") as u32,
                tilt: Tilt::Auto,
                ..ChfOptions::default()
            };
            invert(&law, &options).unwrap()
        });
        match c.get("quantity") {
            "tvar" => d.tvar(c.number("arg")?).ok(),
            _ => None,
        }
    });
}

/// The TVaR is the mean of the quantiles above `p`, at least the VaR, and
/// rises with `p` to the largest point of positive mass; it refuses a
/// probability outside `[0, 1]`.
#[test]
fn chf_tvar_is_the_tail_mean_of_the_quantiles() {
    let law = ChfLaw::Gamma(Gamma::new(4.0, 25.0).unwrap());
    let d = invert(&law, &ChfOptions::with_log2(12)).unwrap();
    // Midpoint rule on (0.9, 1): the quantile is a step function, so the
    // integral converges like the bucket count over the steps.
    let n = 200_000;
    let integral: f64 = (0..n)
        .map(|i| d.quantile(0.9 + 0.1 * (i as f64 + 0.5) / n as f64).unwrap())
        .sum::<f64>()
        / n as f64;
    let tvar = d.tvar(0.9).unwrap();
    assert!((tvar - integral).abs() < 1e-3 * tvar, "{tvar} against {integral}");
    let mut last = f64::NEG_INFINITY;
    for p in [0.0, 0.5, 0.9, 0.99, 0.999, 0.999_999] {
        let t = d.tvar(p).unwrap();
        assert!(t >= d.quantile(p).unwrap() && t >= last, "p {p}");
        last = t;
    }
    let top = (0..d.masses().len())
        .rev()
        .find(|&k| d.masses()[k] > 0.0)
        .map(|k| d.x(k))
        .unwrap();
    assert_eq!(d.tvar(1.0).unwrap(), top);
    assert!(d.tvar(1.5).is_err() && d.tvar(-0.1).is_err());
    // The gamma's TVaR at 99% in closed form, mean (1 - F_{k+1}(q)) / 0.01
    // at its exact 99% quantile, to within a bucket of the grid's.
    let q = Gamma::new(4.0, 25.0).unwrap().quantile(0.99).unwrap();
    let above = 1.0 - Gamma::new(5.0, 25.0).unwrap().cdf(q);
    let exact = 100.0 * above / 0.01;
    assert!((d.tvar(0.99).unwrap() - exact).abs() < d.step(), "{exact}");
}
