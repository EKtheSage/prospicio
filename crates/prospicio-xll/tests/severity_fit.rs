//! `PRO.SEV.FIT`, reached through the catalogue as Excel reaches it:
//!
//! * the Pareto families against the R package Pareto 2.4.5
//!   (`validation/reference/pareto_fit_r.csv`: reporting thresholds,
//!   censoring, weights and truncation), at its tolerances;
//! * the body families on their closed forms untruncated, and equal to
//!   `fit_severity` with thresholds and censoring;
//! * a fit's parameters fed to `PRO.SEV.CDF`, the goodness-of-fit table
//!   and the comparison of the body families.

mod common;

use common::{n, num, reference, t, text};
use prospicio_prob::severity_fit::{Family, fit_severity};
use prospicio_prob::{Distribution, LargeLosses};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

/// The losses of `validation/scripts/r_pareto_fit.R`.
const LOSSES: [f64; 14] = [
    1100.0, 1300.0, 1750.0, 2000.0, 2600.0, 3500.0, 4100.0, 5200.0, 7000.0, 9000.0, 12000.0,
    18000.0, 25000.0, 40000.0,
];

fn column(x: &[f64]) -> Grid {
    Grid::column(x)
}

fn missing() -> Grid {
    Grid::missing()
}

/// The estimate of the parameter in row `row` of a `"params"` table.
fn estimate(out: &Grid, label: &str) -> f64 {
    let r = (0..out.rows)
        .find(|&r| out.get(r, 0) == &Value::from(label))
        .unwrap_or_else(|| panic!("no row {label} in {out:?}"));
    num(out.get(r, 1))
}

#[test]
fn pareto_fits_match_r() {
    let mut checked = 0;
    for row in reference("pareto_fit_r.csv") {
        let (model, spec) = (row[0].as_str(), row[1].as_str());
        if model == "model" {
            continue;
        }
        let fields: std::collections::HashMap<&str, &str> = spec
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .collect();
        let list = |k: &str| -> Vec<f64> {
            match fields.get(k).copied().unwrap_or("") {
                "" => vec![],
                v => v.split('|').map(|x| x.parse().unwrap()).collect(),
            }
        };
        let or_missing = |v: Vec<f64>| if v.is_empty() { missing() } else { column(&v) };
        let (r, c, w, thresholds) = (list("r"), list("c"), list("w"), list("t"));
        let censored = if c.is_empty() {
            missing()
        } else {
            Grid::from_rows(c.iter().map(|&x| vec![Value::Bool(x == 1.0)]).collect())
        };
        let truncation = match (fields.get("truncation"), fields.get("type")) {
            (None, _) => missing(),
            (Some(v), None) => n(v.parse().unwrap()),
            (Some(v), Some(kind)) => Grid::from_rows(vec![vec![
                Value::Number(v.parse().unwrap()),
                Value::from(*kind),
            ]]),
        };
        let t_arg = if thresholds.len() == 1 {
            n(thresholds[0])
        } else {
            Grid::row(&thresholds)
        };
        let out = call(
            "pro_sev_fit",
            vec![
                column(&LOSSES),
                t(model),
                t("params"),
                t_arg,
                truncation,
                or_missing(r),
                censored,
                or_missing(w),
            ],
        );
        let index: usize = row[2].parse().unwrap();
        let got = match model {
            "pareto" => estimate(&out, "alpha"),
            "gen_pareto" => estimate(&out, ["alpha_ini", "alpha_tail"][index]),
            _ => estimate(&out, &format!("alpha above {}", thresholds[index])),
        };
        let (want, abs, rel): (f64, f64, f64) = (
            row[3].parse().unwrap(),
            row[4].parse().unwrap(),
            row[5].parse().unwrap(),
        );
        assert!(
            (got - want).abs() <= abs.max(rel * want.abs()),
            "{model} {spec} [{index}]: {got} against R's {want}"
        );
        checked += 1;
    }
    assert!(checked >= 30, "{checked}");
}

#[test]
fn the_lognormal_is_the_mean_and_sd_of_the_logs() {
    let out = call("pro_sev_fit", vec![column(&LOSSES), t("lognormal")]);
    let logs: Vec<f64> = LOSSES.iter().map(|x| x.ln()).collect();
    let k = logs.len() as f64;
    let mu = logs.iter().sum::<f64>() / k;
    let sd = (logs.iter().map(|l| (l - mu).powi(2)).sum::<f64>() / k).sqrt();
    assert!((estimate(&out, "meanlog") - mu).abs() < 1e-7);
    assert!((estimate(&out, "sdlog") / sd - 1.0).abs() < 1e-7);
    // The standard errors sd / sqrt(n) and sd / sqrt(2n), in the third column.
    assert!((num(out.get(1, 2)) / (sd / k.sqrt()) - 1.0).abs() < 1e-4);
    assert!((num(out.get(2, 2)) / (sd / (2.0 * k).sqrt()) - 1.0).abs() < 1e-4);
    assert_eq!(estimate(&out, "Losses"), 14.0);
    assert!((estimate(&out, "AIC") - (4.0 - 2.0 * estimate(&out, "Log-likelihood"))).abs() < 1e-9);
    assert_eq!(out.get(out.rows - 2, 1), &Value::Bool(true));
    assert_eq!(out.get(out.rows - 1, 1), &Value::from("none"));
}

#[test]
fn body_fits_with_thresholds_and_censoring_are_fit_severity() {
    let r: Vec<f64> = LOSSES
        .iter()
        .enumerate()
        .map(|(i, _)| if i % 3 == 0 { 1000.0 } else { 0.0 })
        .collect();
    let c: Vec<bool> = LOSSES.iter().map(|&x| x >= 18_000.0).collect();
    let w: Vec<f64> = (0..14).map(|i| 1.0 + (i % 2) as f64).collect();
    let data = LargeLosses::new(LOSSES.to_vec())
        .unwrap()
        .reporting_thresholds(r.clone())
        .unwrap()
        .censored(c.clone())
        .unwrap()
        .weights(w.clone())
        .unwrap();
    for family in [
        Family::Lognormal,
        Family::Gamma,
        Family::Weibull,
        Family::Burr,
    ] {
        let want = fit_severity(family, &data).unwrap();
        let out = call(
            "pro_sev_fit",
            vec![
                column(&LOSSES),
                t(family.name()),
                missing(),
                missing(),
                missing(),
                column(&r),
                Grid::from_rows(c.iter().map(|&b| vec![Value::Bool(b)]).collect()),
                column(&w),
            ],
        );
        for (i, name) in family.parameter_names().iter().enumerate() {
            assert_eq!(
                estimate(&out, name),
                want.parameters()[i],
                "{family:?} {name}"
            );
            assert_eq!(out.get(1 + i, 2), &Value::number(want.standard_errors()[i]));
        }
        assert_eq!(estimate(&out, "Log-likelihood"), want.log_likelihood());
        assert_eq!(estimate(&out, "BIC"), want.bic());
    }
}

#[test]
fn a_fit_feeds_the_severity_functions() {
    let fit = call("pro_sev_fit", vec![column(&LOSSES), t("burr")]);
    let p: Vec<Grid> = (1..4)
        .map(|r| Grid::scalar(fit.get(r, 1).clone()))
        .collect();
    let data = LargeLosses::new(LOSSES.to_vec()).unwrap();
    let want = fit_severity(Family::Burr, &data).unwrap();
    for x in [1500.0, 10_000.0] {
        let cdf = call(
            "pro_sev_cdf",
            vec![n(x), t("burr"), p[0].clone(), p[1].clone(), p[2].clone()],
        );
        assert!((num(&cdf.cells[0]) - want.dist().cdf(x)).abs() < 1e-14);
    }
    // A Pareto's t and alpha are p1 and p2.
    let fit = call(
        "pro_sev_fit",
        vec![column(&LOSSES), t("pareto"), missing(), n(1000.0)],
    );
    let cdf = call(
        "pro_sev_cdf",
        vec![
            n(5000.0),
            t("pareto"),
            Grid::scalar(fit.get(1, 1).clone()),
            Grid::scalar(fit.get(2, 1).clone()),
        ],
    );
    let alpha = estimate(&fit, "alpha");
    assert!((num(&cdf.cells[0]) - (1.0 - 0.2f64.powf(alpha))).abs() < 1e-14);
}

#[test]
fn gof_compares_with_the_empirical_survival() {
    let out = call(
        "pro_sev_fit",
        vec![column(&LOSSES), t("lognormal"), t("gof")],
    );
    assert_eq!((out.rows, out.columns), (15, 6));
    assert_eq!(out.get(0, 3), &Value::from("Empirical survival"));
    // Uncensored and untruncated: the empirical survival steps by 1/14.
    for (i, &loss) in LOSSES.iter().enumerate() {
        assert_eq!(num(out.get(i + 1, 0)), loss);
        assert!((num(out.get(i + 1, 3)) - (13 - i) as f64 / 14.0).abs() < 1e-12);
    }
    let params = call("pro_sev_fit", vec![column(&LOSSES), t("lognormal")]);
    let ks = (1..15)
        .map(|r| {
            let m = num(out.get(r, 4));
            let before = if r == 1 { 1.0 } else { num(out.get(r - 1, 3)) };
            (num(out.get(r, 3)) - m).abs().max((before - m).abs())
        })
        .fold(0.0, f64::max);
    assert!((estimate(&params, "KS distance") - ks).abs() < 1e-12);
}

#[test]
fn compare_fits_every_body_family() {
    let out = call(
        "pro_sev_fit",
        vec![column(&LOSSES), missing(), t("compare")],
    );
    assert_eq!((out.rows, out.columns), (6, 10));
    let names: Vec<&Value> = (1..6).map(|r| out.get(r, 0)).collect();
    assert_eq!(
        names,
        ["lognormal", "gamma", "weibull", "burr", "lomax"]
            .map(Value::from)
            .iter()
            .collect::<Vec<_>>()
    );
    let lognormal = call("pro_sev_fit", vec![column(&LOSSES), t("lognormal")]);
    assert_eq!(out.get(1, 1), lognormal.get(1, 1));
    assert_eq!(out.get(1, 3), &Value::Empty);
    assert_eq!(out.get(1, 5), &Value::Number(estimate(&lognormal, "AIC")));
    // On these Pareto-like losses the Burr runs to its Pareto limit, with
    // the highest likelihood: the note says so.
    assert!(
        matches!(out.get(4, 9), Value::Text(s) if s.contains("Pareto limit")),
        "{:?}",
        out.get(4, 9)
    );
    assert!(num(out.get(4, 4)) > num(out.get(1, 4)));
}

#[test]
fn blank_cells_are_skipped_and_bad_inputs_named() {
    let mut cells: Vec<f64> = LOSSES.to_vec();
    cells.push(f64::NAN);
    let with_blank = Grid::column(&cells); // NaN becomes a blank cell.
    let a = call("pro_sev_fit", vec![with_blank, t("gamma")]);
    let b = call("pro_sev_fit", vec![column(&LOSSES), t("gamma")]);
    assert_eq!(a, b);
    let e = call(
        "pro_sev_fit",
        vec![column(&LOSSES), t("gamma"), missing(), n(1000.0)],
    );
    assert!(
        text(&e).contains("t is taken only by the Pareto families"),
        "{}",
        text(&e)
    );
    let e = call(
        "pro_sev_fit",
        vec![
            column(&LOSSES),
            t("lognormal"),
            missing(),
            missing(),
            missing(),
            column(&[0.0, 0.0]),
        ],
    );
    assert!(text(&e).contains("thresholds (2 x 1)"), "{}", text(&e));
    let e = call(
        "pro_sev_fit",
        vec![column(&LOSSES), t("pareto"), missing(), n(2000.0)],
    );
    assert!(text(&e).starts_with("#PROSPICIO!"), "{}", text(&e));
    let e = call("pro_sev_fit", vec![column(&LOSSES), t("normal")]);
    assert!(text(&e).contains("family must be"), "{}", text(&e));
}
