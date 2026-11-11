//! Severity fits to large losses against SciPy (`scipy_severity_fits.py`):
//! every family's maximum likelihood fit, each loss conditioned on its
//! reporting threshold and capped losses censored, on SciPy's own densities
//! (#800), and the table of every fit's survival against the plotting
//! positions (`scipy_fit_curves.py`, #804).

use std::collections::BTreeMap;
use std::path::Path;

use prospicio_prob::LargeLosses;
use prospicio_prob::severity_fit::{
    Family, SeverityFit, fit_curves, fit_severity_with_threshold, thinned_rows,
};
use prospicio_validation::{check, reference};

/// The reporting threshold of the `plain` losses, and the Pareto
/// families' `t`.
const T: f64 = 150_000.0;

/// A data set's columns: losses, reporting thresholds, censoring flags.
type Columns = (Vec<f64>, Vec<f64>, Vec<bool>);

/// The losses of `validation/data/severity_fits.csv`, by data set.
fn data() -> BTreeMap<String, LargeLosses> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/severity_fits.csv");
    let text = std::fs::read_to_string(path).unwrap();
    let mut sets: BTreeMap<String, Columns> = BTreeMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let e = sets.entry(f[0].to_owned()).or_default();
        e.0.push(f[1].parse().unwrap());
        e.1.push(f[2].parse().unwrap());
        e.2.push(f[3] == "1");
    }
    sets.into_iter()
        .map(|(k, (y, r, c))| {
            let d = LargeLosses::new(y)
                .unwrap()
                .reporting_thresholds(r)
                .unwrap()
                .censored(c)
                .unwrap();
            (k, d)
        })
        .collect()
}

#[test]
fn every_family_matches_scipy() {
    let sets = data();
    let mut fits: BTreeMap<(String, String), SeverityFit> = BTreeMap::new();
    let cases = reference("severity_fits_scipy.csv");
    check(&cases, |c| {
        let key = (c.get("family").to_owned(), c.get("data").to_owned());
        let fit = fits.entry(key).or_insert_with(|| {
            let family = Family::from_name(c.get("family")).unwrap();
            let t = family.needs_threshold().then_some(T);
            fit_severity_with_threshold(family, &sets[c.get("data")], t).unwrap()
        });
        match c.get("quantity") {
            "log_likelihood" => Some(fit.log_likelihood()),
            "aic" => Some(fit.aic()),
            name => {
                let names = fit.family().parameter_names();
                let i = names.iter().position(|n| *n == name)?;
                Some(fit.parameters()[i])
            }
        }
    });
}

#[test]
fn the_pareto_families_score_on_the_body_likelihood() {
    // With every reporting threshold at t, the Pareto, the generalized
    // Pareto and the Lomax (whose excess over t is a generalized Pareto)
    // share one likelihood, and the generalized Pareto equals the Lomax.
    let sets = data();
    let plain = &sets["plain"];
    let gpd = fit_severity_with_threshold(Family::GenPareto, plain, Some(T)).unwrap();
    let lomax = fit_severity_with_threshold(Family::Lomax, plain, None).unwrap();
    assert!((gpd.log_likelihood() - lomax.log_likelihood()).abs() < 1e-6);
    assert_eq!(gpd.threshold(), Some(T));
    // Riegel's tail alpha is the Lomax's alpha; its initial alpha is
    // t / (θ + t) times it, the Lomax's local Pareto alpha at t.
    let (p, l) = (gpd.parameters(), lomax.parameters());
    assert!((p[2] / l[0] - 1.0).abs() < 1e-4, "{p:?} {l:?}");
    assert!((p[1] / (l[0] * T / (l[1] + T)) - 1.0).abs() < 1e-4);
    // The standard errors of the free parameters, NaN for t.
    let se = gpd.standard_errors();
    assert!(se[0].is_nan() && se[1] > 0.0 && se[2] > 0.0);
    // The distributions' own statistics agree with the fits'.
    let d = gpd.dist();
    let prospicio_prob::Dist::GeneralizedPareto(g) = d else {
        panic!("{d:?}")
    };
    let st = g.fit_statistics(plain).unwrap();
    assert!((st.log_likelihood - gpd.log_likelihood()).abs() < 1e-9);
    assert!((st.aic() - gpd.aic()).abs() < 1e-9);
    let pareto = fit_severity_with_threshold(Family::Pareto, plain, Some(T)).unwrap();
    let prospicio_prob::Dist::Pareto(p) = pareto.dist() else {
        panic!()
    };
    let st = p.fit_statistics(plain).unwrap();
    assert!((st.log_likelihood - pareto.log_likelihood()).abs() < 1e-9);
    assert!((st.bic() - pareto.bic()).abs() < 1e-9);
}

#[test]
fn fit_curves_match_scipy() {
    let plain = &data()["plain"];
    let all = fit_curves(plain, T, &Family::ALL, 200).unwrap();
    // 89 losses: no thinning, and every family fits.
    assert_eq!(all.losses.len(), 89);
    assert_eq!(all.curves.len(), 9);
    let aic: Vec<f64> = all.curves.iter().map(|(f, _)| f.aic()).collect();
    assert!(aic.windows(2).all(|w| w[0] <= w[1]), "{aic:?}");
    let thin = fit_curves(plain, T, &[Family::Lognormal], 10).unwrap();
    let rows = thinned_rows(89, 10);
    let cases = reference("fit_curves_scipy.csv");
    check(&cases, |c| {
        let row: usize = c.get("row").parse().unwrap();
        let loss: f64 = c.get("loss").parse().unwrap();
        match c.get("family") {
            "empirical" => {
                assert_eq!(all.losses[row - 1], loss);
                Some(all.empirical[row - 1])
            }
            "thinned_10" => {
                let k = rows.iter().position(|&i| i + 1 == row).unwrap();
                assert_eq!(thin.losses[k], loss);
                assert_eq!(thin.empirical[k], all.empirical[row - 1]);
                Some((rows[k] + 1) as f64)
            }
            name => {
                let (_, s) = all
                    .curves
                    .iter()
                    .find(|(f, _)| f.family().name() == name)
                    .unwrap();
                Some(s[row - 1])
            }
        }
    });
    assert_eq!(rows.len(), 10);
}
