//! The piecewise Pareto and Riegel's generalized Pareto as severity
//! families, and `PRO.LAYER.VAR`, reached through the catalogue as Excel
//! reaches them, against the R package Pareto 2.4.5
//! (`validation/reference/piecewise_pareto_r.csv` and
//! `gen_pareto_r.csv`): every cdf, quantile, layer mean, second moment and
//! variance, truncated cases included, at the files' tolerances.

mod common;

use std::collections::HashMap;

use common::{n, num, reference, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

/// `family, p1, p2, p3` for a row's `params`.
fn parameters(family: &str, spec: &str) -> Vec<Grid> {
    let fields: HashMap<&str, &str> = spec
        .split(';')
        .map(|kv| kv.split_once('=').unwrap())
        .collect();
    let row = |k: &str| {
        Grid::from_rows(vec![
            fields[k]
                .split('|')
                .map(|v| Value::Number(v.parse().unwrap()))
                .collect(),
        ])
    };
    match family {
        "piecewise_pareto" => {
            // Thresholds as a row, alphas as a column: any shape reads in order.
            let alphas = row("alpha");
            let alphas = Grid::from_rows(alphas.cells.iter().map(|v| vec![v.clone()]).collect());
            let p3 = match fields.get("truncation") {
                None => Grid::missing(),
                Some(tr) => Grid::from_rows(vec![vec![
                    Value::Number(tr.parse().unwrap()),
                    Value::from(fields["type"]),
                ]]),
            };
            vec![t("piecewise_pareto"), row("t"), alphas, p3]
        }
        _ => vec![
            t("gen_pareto"),
            n(fields["t"].parse().unwrap()),
            n(fields["alpha_ini"].parse().unwrap()),
            n(fields["alpha_tail"].parse().unwrap()),
        ],
    }
}

fn check(file: &str) -> usize {
    let mut checked = 0;
    for row in reference(file).into_iter().skip(1) {
        let (family, quantity) = (row[0].as_str(), row[2].as_str());
        let sev = parameters(family, &row[1]);
        let first = |mut a: Vec<Grid>| {
            a.extend(sev.iter().cloned());
            a
        };
        let arg = || n(row[3].parse().unwrap());
        let layer = || {
            let limit = if row[3] == "inf" {
                t("unlimited")
            } else {
                arg()
            };
            vec![limit, n(row[4].parse().unwrap())]
        };
        let got = match quantity {
            "cdf" => num(call("pro_sev_cdf", first(vec![arg()])).as_scalar().unwrap()),
            "quantile" => num(call("pro_sev_quantile", first(vec![arg()]))
                .as_scalar()
                .unwrap()),
            "layer" => num(call("pro_layer", first(layer())).as_scalar().unwrap()),
            "layer_variance" => num(call("pro_layer_var", first(layer())).as_scalar().unwrap()),
            "layer_second_moment" => {
                let m = num(call("pro_layer", first(layer())).as_scalar().unwrap());
                let v = num(call("pro_layer_var", first(layer())).as_scalar().unwrap());
                v + m * m
            }
            other => panic!("unknown quantity {other}"),
        };
        let expected: f64 = row[5].parse().unwrap();
        let (abs_tol, rel_tol): (f64, f64) = (row[6].parse().unwrap(), row[7].parse().unwrap());
        assert!(
            (got - expected).abs() <= abs_tol.max(rel_tol * expected.abs()),
            "{family} {} {quantity} {} {}: {got} vs {expected}",
            row[1],
            row[3],
            row[4]
        );
        checked += 1;
    }
    checked
}

#[test]
fn piecewise_pareto_matches_r() {
    assert_eq!(check("piecewise_pareto_r.csv"), 182);
}

#[test]
fn gen_pareto_matches_r() {
    assert!(check("gen_pareto_r.csv") > 50);
}

#[test]
fn layer_variance_closed_form_and_errors() {
    // Single-piece Pareto(1000, 3), layer 1000 xs 1000: E[L] and E[L^2]
    // from the survival (1000/x)^3 on [1000, 2000].
    let s = |x: f64| (1000.0 / x).powi(3);
    let (mut m1, mut m2, k) = (0.0, 0.0, 200_000);
    for i in 0..k {
        let x = 1000.0 + (i as f64 + 0.5) * 1000.0 / k as f64;
        let h = 1000.0 / k as f64;
        m1 += s(x) * h;
        m2 += 2.0 * (x - 1000.0) * s(x) * h;
    }
    let v = call(
        "pro_layer_var",
        vec![n(1000.0), n(1000.0), t("pareto"), n(1000.0), n(3.0)],
    );
    assert!((num(v.as_scalar().unwrap()) - (m2 - m1 * m1)).abs() < 1e-3);
    // A piecewise Pareto of one piece is the Pareto.
    let pp = call(
        "pro_layer_var",
        vec![
            n(1000.0),
            n(1000.0),
            t("piecewise_pareto"),
            n(1000.0),
            n(3.0),
        ],
    );
    assert!((num(pp.as_scalar().unwrap()) - num(v.as_scalar().unwrap())).abs() < 1e-9);
    // Alpha 1.5 has no second moment: the unlimited variance is blank.
    let inf = call(
        "pro_layer_var",
        vec![t("unlimited"), n(0.0), t("pareto"), n(1000.0), n(1.5)],
    );
    assert_eq!(inf.as_scalar(), Some(&Value::Empty));

    let pp = |p1: Grid, p2: Grid, p3: Grid| {
        text(&call(
            "pro_sev_cdf",
            vec![n(1.0), t("piecewise_pareto"), p1, p2, p3],
        ))
        .to_owned()
    };
    let ts = Grid::from_rows(vec![vec![Value::Number(1000.0), Value::Number(2000.0)]]);
    let alphas = Grid::from_rows(vec![vec![Value::Number(1.0), Value::Number(2.0)]]);
    assert!(pp(ts.clone(), n(1.0), Grid::missing()).contains("alpha"));
    assert!(
        pp(
            ts.clone(),
            alphas.clone(),
            Grid::from_rows(vec![vec![Value::Number(5000.0), Value::from("xx")]])
        )
        .contains("\"lp\" or \"wd\"")
    );
    assert!(pp(ts, alphas, n(1500.0)).contains("truncation"));
    assert!(
        text(&call(
            "pro_sev_cdf",
            vec![n(1.0), t("gen_pareto"), n(1000.0), n(2.0)]
        ))
        .contains("alpha_tail")
    );
}
