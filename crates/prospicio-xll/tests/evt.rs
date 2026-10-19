//! The EVT functions, reached through the catalogue as Excel reaches them:
//!
//! * `PRO.EVT.THRESHOLDS` against R `ismev` 1.43's `mrl.plot` and
//!   `gpd.fitrange` (`validation/reference/evt_thresholds_r.csv`), at its
//!   tolerances;
//! * `PRO.EVT.FIT`'s shape and scale equal to the diagnostics' at the same
//!   threshold, and its VaR and TVaR on their closed forms;
//! * `PRO.EVT.HILL` on exact Pareto quantiles.

mod common;

use common::{n, num, reference, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

/// The data sets of `validation/scripts/r_evt_thresholds.R`: exact
/// quantiles of a Burr or a Weibull.
fn data(params: &str) -> Vec<f64> {
    let f: std::collections::HashMap<&str, f64> = params
        .split(';')
        .filter_map(|kv| kv.split_once('='))
        .filter_map(|(k, v)| v.parse().ok().map(|v| (k, v)))
        .collect();
    let m = f["n"] as usize;
    let p = (1..=m).map(|i| (i as f64 - 0.5) / m as f64);
    if params.contains("data=burr") {
        let (a, c) = (f["a"], f["c"]);
        p.map(|p| ((1.0 - p).powf(-1.0 / a) - 1.0).powf(1.0 / c))
            .collect()
    } else {
        let k = f["k"];
        p.map(|p| (-(-p).ln_1p()).powf(1.0 / k)).collect()
    }
}

#[test]
fn thresholds_match_ismev() {
    let columns = [
        "n_above",
        "mean_excess",
        "mean_excess_se",
        "xi",
        "xi_se",
        "beta",
        "modified_scale",
        "modified_scale_se",
    ];
    let mut checked = 0;
    for row in reference("evt_thresholds_r.csv") {
        if row[0] != "evt_thresholds" {
            continue;
        }
        let quantity = row[2].strip_prefix("ismev_").unwrap_or(&row[2]);
        let u: f64 = row[3].parse().unwrap();
        let out = call(
            "pro_evt_thresholds",
            vec![Grid::column(&data(&row[1])), n(u)],
        );
        assert_eq!(num(out.get(1, 0)), u);
        let c = 1 + columns.iter().position(|&q| q == quantity).unwrap();
        let got = num(out.get(1, c));
        let (want, abs, rel): (f64, f64, f64) = (
            row[4].parse().unwrap(),
            row[5].parse().unwrap(),
            row[6].parse().unwrap(),
        );
        assert!(
            (got - want).abs() <= abs.max(rel * want.abs()),
            "{} {quantity} at {u}: {got} against ismev's {want}",
            row[1]
        );
        checked += 1;
    }
    assert!(checked >= 100, "{checked}");
}

fn burr() -> Vec<f64> {
    data("data=burr;n=1000;a=2;c=1.5")
}

#[test]
fn default_thresholds_are_the_data_quantiles() {
    let x = burr();
    let out = call("pro_evt_thresholds", vec![Grid::column(&x)]);
    assert_eq!((out.rows, out.columns), (11, 9));
    // The 50% quantile of 1,000 sorted values, R's type 7.
    let median = 0.5 * (x[499] + x[500]);
    assert!((num(out.get(1, 0)) - median).abs() < 1e-12);
    assert_eq!(num(out.get(1, 1)), 500.0);
}

#[test]
fn fit_is_the_diagnostics_gpd_with_its_tail() {
    let x = burr();
    let u = 1.0;
    let diag = call("pro_evt_thresholds", vec![Grid::column(&x), n(u)]);
    let fit = call("pro_evt_fit", vec![Grid::column(&x), n(u)]);
    let value = |label: &str| {
        let r = (0..fit.rows)
            .find(|&r| fit.get(r, 0) == &Value::from(label))
            .unwrap_or_else(|| panic!("no {label}"));
        num(fit.get(r, 1))
    };
    let (xi, beta) = (value("xi"), value("beta"));
    assert_eq!(xi, num(diag.get(1, 4)));
    assert_eq!(num(fit.get(4, 2)), num(diag.get(1, 5)));
    assert_eq!(beta, num(diag.get(1, 6)));
    let above = num(diag.get(1, 1));
    assert_eq!(value("Exceedances"), above);
    let pu = above / 1000.0;
    assert_eq!(value("P(exceed)"), pu);
    for p in [0.99, 0.995, 0.999] {
        let var = u + beta * ((pu / (1.0 - p)).powf(xi) - 1.0) / xi;
        assert!((value(&format!("VaR {p}")) / var - 1.0).abs() < 1e-12);
        let tvar = var + (beta + xi * (var - u)) / (1.0 - xi);
        assert!((value(&format!("TVaR {p}")) / tvar - 1.0).abs() < 1e-12);
    }
    // A probability below the tail says so in its cell.
    let low = call("pro_evt_fit", vec![Grid::column(&x), n(u), n(0.5)]);
    assert!(matches!(low.get(6, 1), Value::Text(s) if s.contains("below the tail")));
    // The shape and scale are PRO.SEV's "gpd" for the exceedances.
    let s = call(
        "pro_sev_survival",
        vec![n(1.0), Grid::scalar(Value::from("gpd")), n(xi), n(beta)],
    );
    assert!((num(&s.cells[0]) - (1.0 + xi / beta).powf(-1.0 / xi)).abs() < 1e-14);
}

#[test]
fn hill_recovers_a_pareto_alpha() {
    // Exact Pareto quantiles with alpha 2 from 1.
    let m = 10_000;
    let x: Vec<f64> = (1..=m)
        .map(|i| (1.0 - (i as f64 - 0.5) / m as f64).powf(-0.5))
        .collect();
    let out = call("pro_evt_hill", vec![Grid::column(&x), n(1000.0)]);
    assert_eq!((out.rows, out.columns), (2, 4));
    assert!((num(out.get(1, 3)) - 2.0).abs() < 0.04);
    let mut sorted = x.clone();
    sorted.sort_by(f64::total_cmp);
    assert_eq!(num(out.get(1, 1)), sorted[m - 1001]);
    let all = call("pro_evt_hill", vec![Grid::column(&x)]);
    assert_eq!(all.rows, 11);
    assert_eq!(num(all.get(1, 0)), 500.0);
    assert_eq!(num(all.get(10, 0)), 5000.0);
}

#[test]
fn bad_inputs_are_named() {
    let x = burr();
    let e = call("pro_evt_fit", vec![Grid::column(&x), n(1e9)]);
    assert!(
        text(&e).contains("leaves 0 values above it"),
        "{}",
        text(&e)
    );
    let e = call("pro_evt_hill", vec![Grid::column(&x), n(1000.0)]);
    assert!(
        text(&e).contains("k must be whole numbers from 1 to 999"),
        "{}",
        text(&e)
    );
    let e = call("pro_evt_thresholds", vec![Grid::column(&[1.0, 2.0])]);
    assert!(text(&e).contains("at least 3 values"), "{}", text(&e));
}
