//! The severity functions over every family, reached through the
//! catalogue as Excel reaches them:
//!
//! * `PRO.LEV` and `PRO.LAYER` (a stop loss as an unlimited layer) on the
//!   gamma, log-logistic, lognormal and Weibull against mpmath's
//!   quadrature (`validation/reference/severity_mpmath.csv`), at its
//!   tolerances;
//! * `PRO.SEV.CDF`, `PRO.SEV.SURVIVAL` and `PRO.SEV.QUANTILE` inverse to
//!   each other for every family, and `PRO.SEV.STATS` and `PRO.LER` on
//!   closed forms.

mod common;

use common::{n, num, reference, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn params(spec: &str) -> Vec<f64> {
    spec.split(';')
        .map(|kv| kv.split_once('=').unwrap().1.parse().unwrap())
        .collect()
}

#[test]
fn lev_layer_and_stop_loss_match_mpmath() {
    let mut checked = 0;
    for row in reference("severity_mpmath.csv") {
        let (family, quantity) = (row[0].as_str(), row[2].as_str());
        if !["gamma", "loglogistic", "lognormal", "weibull"].contains(&family) {
            continue;
        }
        let p = params(&row[1]);
        let sev = |first: Vec<Grid>| {
            let mut args = first;
            args.extend([t(family), n(p[0]), n(p[1])]);
            args
        };
        let arg: f64 = row[3].parse().unwrap();
        let out = match quantity {
            "lev" => call("pro_lev", sev(vec![n(arg)])),
            "layer" => call("pro_layer", sev(vec![n(arg), n(row[4].parse().unwrap())])),
            "stop_loss" => call("pro_layer", sev(vec![t("unlimited"), n(arg)])),
            _ => continue,
        };
        let got = num(out.as_scalar().unwrap());
        let expected: f64 = row[5].parse().unwrap();
        let (abs_tol, rel_tol): (f64, f64) = (row[6].parse().unwrap(), row[7].parse().unwrap());
        assert!(
            (got - expected).abs() <= abs_tol.max(rel_tol * expected.abs()),
            "{family} {} {quantity} {arg}: {got} vs {expected}",
            row[1]
        );
        checked += 1;
    }
    assert!(checked > 100, "checked {checked}");
}

/// Every family with parameters, as `family, p1, p2, [p3]`.
fn families() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("pareto", vec![500.0, 2.0]),
        ("pareto", vec![500.0, 1.5, 1e6]),
        ("lognormal", vec![7.0, 1.2]),
        ("lognormal_cv", vec![50_000.0, 3.0]),
        ("gamma", vec![2.5, 400.0]),
        ("gamma_cv", vec![1_000.0, 0.5]),
        ("weibull", vec![0.7, 1_000.0]),
        ("loglogistic", vec![2.5, 1_000.0]),
        ("inverse_gamma", vec![3.0, 2_000.0]),
        ("inverse_gaussian", vec![1_000.0, 500.0]),
        ("burr", vec![3.0, 1.5, 2_000.0]),
        ("gpd", vec![0.3, 1_000.0]),
    ]
}

fn with(family: &str, p: &[f64], first: Grid) -> Vec<Grid> {
    let mut args = vec![first, t(family)];
    args.extend(p.iter().map(|&x| n(x)));
    args
}

#[test]
fn quantile_and_cdf_are_inverse_for_every_family() {
    let ps = Grid::from_rows(vec![
        [0.01, 0.25, 0.5, 0.9, 0.995].map(Value::Number).to_vec(),
    ]);
    for (family, p) in families() {
        let q = call("pro_sev_quantile", with(family, &p, ps.clone()));
        assert_eq!((q.rows, q.columns), (1, 5), "{family}: {:?}", q.as_scalar());
        let back = call("pro_sev_cdf", with(family, &p, q.clone()));
        let surv = call("pro_sev_survival", with(family, &p, q));
        for k in 0..5 {
            let want = num(&ps.cells[k]);
            assert!(
                (num(&back.cells[k]) - want).abs() < 1e-9,
                "{family} cdf {k}"
            );
            assert!(
                (num(&surv.cells[k]) - (1.0 - want)).abs() < 1e-9,
                "{family} survival {k}"
            );
        }
        let stats = call("pro_sev_stats", {
            let mut a = vec![t(family)];
            a.extend(p.iter().map(|&x| n(x)));
            a
        });
        assert_eq!((stats.rows, stats.columns), (4, 2), "{family}");
        // The unlimited LEV is the mean.
        let lev = call("pro_lev", with(family, &p, t("unlimited")));
        let mean = num(stats.get(1, 1));
        assert!(
            (num(lev.as_scalar().unwrap()) - mean).abs() <= 1e-9 * mean,
            "{family}"
        );
    }
}

#[test]
fn closed_forms() {
    // Burr with gamma 1 is the Lomax: mean theta / (alpha - 1), S(theta) = 2^-alpha.
    let stats = call("pro_sev_stats", vec![t("burr"), n(3.0), n(1.0), n(2_000.0)]);
    assert!((num(stats.get(1, 1)) - 1_000.0).abs() < 1e-9);
    let s = call(
        "pro_sev_survival",
        vec![n(2_000.0), t("burr"), n(3.0), n(1.0), n(2_000.0)],
    );
    assert!((num(s.as_scalar().unwrap()) - 0.125).abs() < 1e-15);
    // GPD mean beta / (1 - xi).
    let g = call("pro_sev_stats", vec![t("gpd"), n(0.5), n(2.0)]);
    assert!((num(g.get(1, 1)) - 4.0).abs() < 1e-12);
    // Log-logistic shape 1: F(x) = x / (x + theta), E[min(X, u)] = theta ln(1 + u / theta).
    let f = call(
        "pro_sev_cdf",
        vec![n(3.0), t("loglogistic"), n(1.0), n(2.0)],
    );
    assert!((num(f.as_scalar().unwrap()) - 0.6).abs() < 1e-15);
    let l = call("pro_lev", vec![n(3.0), t("loglogistic"), n(1.0), n(2.0)]);
    assert!((num(l.as_scalar().unwrap()) - 2.0 * 2.5f64.ln()).abs() < 1e-13);
    assert_eq!(g.get(0, 1), &Value::from("generalized_pareto"));
    // An infinite moment is blank: log-logistic shape 1 has no mean.
    let inf = call("pro_sev_stats", vec![t("loglogistic"), n(1.0), n(2.0)]);
    assert_eq!(inf.get(1, 1), &Value::Empty);

    // Loss elimination ratio: exponential (gamma shape 1) mean 1000, d 500.
    let ler = call("pro_ler", vec![n(500.0), t("gamma"), n(1.0), n(1_000.0)]);
    assert!((num(ler.as_scalar().unwrap()) - (1.0 - (-0.5f64).exp())).abs() < 1e-12);
}

#[test]
fn parameter_errors() {
    assert!(
        text(&call(
            "pro_sev_cdf",
            vec![n(1.0), t("burr"), n(3.0), n(1.0)]
        ))
        .contains("p3")
    );
    assert!(
        text(&call(
            "pro_sev_cdf",
            vec![n(1.0), t("gamma"), n(1.0), n(1.0), n(5.0)]
        ))
        .contains("p3 is taken only")
    );
    assert!(
        text(&call(
            "pro_sev_quantile",
            vec![n(1.5), t("gamma"), n(1.0), n(1.0)]
        ))
        .starts_with("#PROSPICIO!")
    );
    assert!(
        text(&call("pro_sev_stats", vec![t("cauchy"), n(1.0), n(1.0)]))
            .contains("family must be one of")
    );
}
