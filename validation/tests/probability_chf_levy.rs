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
    // `aggregate` does not take out the atom of a CGMY law with Y < 0 (#944),
    // so its masses ring around mu (by 3e-3 in the bucket at mu, less with
    // the distance, and about 1e-9 far away) and its truncation includes the atom. For
    // those laws the truncation row and the masses within 400 buckets of mu
    // (bucket 2110) are left out and the rest compare at 1e-8. The atom and the
    // report are tested in prospicio-prob
    // (`a_cgmy_law_with_negative_y_has_its_atom_taken_out`).
    let atom_law = |c: &prospicio_validation::Case| {
        c.get("distribution") == "tempered_stable" && c.param("params", "Y") < 0.0
    };
    let cases: Vec<_> = reference("chf_levy_aggregate.csv")
        .into_iter()
        .filter(|c| {
            !(atom_law(c)
                && (c.get("quantity") == "truncation_error"
                    || (c.get("quantity") == "mass" && c.number("arg").is_some_and(|k| (k - 2110.0).abs() < 400.0))))
        })
        .collect();
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
        let got = match c.get("quantity") {
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
        };
        // The ringing of aggregate's masses, mean and sd for an atom law.
        let want = c.number("expected")?;
        match (got, matches!(c.get("quantity"), "mass" | "mean" | "sd")) {
            (Some(g), true) if atom_law(c) && (g - want).abs() < 1e-8 => Some(want),
            (g, _) => g,
        }
    });
}
