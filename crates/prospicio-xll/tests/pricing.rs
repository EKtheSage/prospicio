//! The layer pricing functions, reached through the catalogue as Excel
//! reaches them: the same numbers as `prospicio_pricing` and
//! `prospicio_prob`, the R package Pareto's vignette, and mpmath's
//! lognormal targets in `validation/reference/severity_mpmath.csv`.

use prospicio_pricing::layer::{ilf, layer_cv};
use prospicio_prob::{Lognormal, Pareto, Severity};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn n(x: f64) -> Grid {
    Grid::scalar(Value::Number(x))
}

fn t(s: &str) -> Grid {
    Grid::scalar(Value::from(s))
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Number(x) => *x,
        other => panic!("expected a number, got {other:?}"),
    }
}

fn one(g: &Grid) -> f64 {
    num(g.as_scalar().expect("one value"))
}

fn text(g: &Grid) -> &str {
    match g.as_scalar() {
        Some(Value::Text(s)) => s,
        other => panic!("expected text, got {other:?}"),
    }
}

fn pareto(limit: Grid, attachment: Grid) -> Vec<Grid> {
    vec![
        limit,
        attachment,
        t("pareto"),
        n(500.0),
        n(2.0),
        Grid::missing(),
    ]
}

#[test]
fn pareto_layer_matches_the_r_package() {
    // R package Pareto, vignette: Pareto(t = 500, alpha = 2), 4000 xs 1000.
    let out = call("pro_layer", pareto(n(4000.0), n(1000.0)));
    assert!((one(&out) - 200.0).abs() < 1e-12);
    // Unlimited: t^a d^(1-a) / (a - 1) = 500^2 / 1000.
    let out = call("pro_layer", pareto(t("unlimited"), n(1000.0)));
    assert!((one(&out) - 250.0).abs() < 1e-12);
}

#[test]
fn lognormal_matches_mpmath() {
    let args = |f: Grid, a: Option<Grid>| {
        let mut v = vec![f];
        v.extend(a);
        v.extend([t("lognormal"), n(0.0), n(1.0), Grid::missing()]);
        v
    };
    let lev = call("pro_lev", args(n(1.0), None));
    assert!((one(&lev) - 0.761_578_291_865_123_4).abs() < 1e-12);
    let layer = call("pro_layer", args(n(1.0), Some(n(1.0))));
    assert!((one(&layer) - 0.352_291_857_299_111_94).abs() < 1e-11);
}

#[test]
fn a_column_of_attachments_against_a_row_of_limits_is_a_table() {
    let attachments = Grid::from_rows(
        [1000.0, 2000.0, 5000.0]
            .map(|a| vec![Value::Number(a)])
            .to_vec(),
    );
    let limits = Grid::from_rows(vec![vec![
        Value::Number(1000.0),
        Value::Number(5000.0),
        Value::from("unlimited"),
    ]]);
    let p = Pareto::new(500.0, 2.0).unwrap();
    let out = call("pro_layer", pareto(limits.clone(), attachments.clone()));
    assert_eq!((out.rows, out.columns), (3, 3));
    for (r, a) in [1000.0, 2000.0, 5000.0].into_iter().enumerate() {
        for (c, l) in [1000.0, 5000.0, f64::INFINITY].into_iter().enumerate() {
            assert_eq!(num(out.get(r, c)), p.layer(l, a));
        }
    }
    let cv = call("pro_layer_cv", pareto(limits, attachments));
    assert_eq!(num(cv.get(2, 1)), layer_cv(&p, 5000.0, 5000.0).unwrap());
    assert_eq!(
        cv.get(0, 2),
        &Value::Empty,
        "alpha 2: no finite unlimited variance"
    );
}

#[test]
fn ilf_and_lev_match_the_crates() {
    let sev = Lognormal::from_mean_cv(50_000.0, 3.0).unwrap();
    let limits = Grid::from_rows(
        [100_000.0, 250_000.0, 1e6]
            .map(|l| vec![Value::Number(l)])
            .into_iter()
            .chain([vec![Value::Empty]])
            .collect(),
    );
    let args = |first: Grid| {
        vec![
            first,
            n(100_000.0),
            t("lognormal_cv"),
            n(50_000.0),
            n(3.0),
            Grid::missing(),
        ]
    };
    let out = call("pro_ilf", args(limits.clone()));
    assert_eq!((out.rows, out.columns), (4, 1));
    assert_eq!(num(&out.cells[0]), 1.0);
    assert_eq!(num(&out.cells[2]), ilf(&sev, 1e6, 1e5).unwrap());
    assert_eq!(out.cells[3], Value::Empty, "a blank limit gives a blank");

    let lev = call(
        "pro_lev",
        vec![
            limits,
            t("Lognormal CV"),
            n(50_000.0),
            n(3.0),
            Grid::missing(),
        ],
    );
    assert_eq!(num(&lev.cells[1]), sev.lev(250_000.0));
}

#[test]
fn a_truncated_pareto() {
    let p = Pareto::new(500.0, 1.5).unwrap().truncated(1e6).unwrap();
    let out = call(
        "pro_layer",
        vec![
            t("unlimited"),
            n(1000.0),
            t("pareto"),
            n(500.0),
            n(1.5),
            n(1e6),
        ],
    );
    assert_eq!(one(&out), p.layer(f64::INFINITY, 1000.0));
}

#[test]
fn errors_come_back_as_text() {
    let bad = |args| text(&call("pro_layer", args)).to_owned();
    assert!(
        bad(vec![
            n(1.0),
            n(1.0),
            t("gamma"),
            n(1.0),
            n(1.0),
            Grid::missing()
        ])
        .contains("family must be")
    );
    assert!(
        bad(vec![
            n(1.0),
            n(1.0),
            t("lognormal"),
            n(0.0),
            n(1.0),
            n(10.0)
        ])
        .contains("truncation is taken only")
    );
    assert!(bad(pareto(n(1.0), n(-1.0))).contains("non-negative attachment"));
    assert!(bad(pareto(t("a lot"), n(1.0))).contains("must be a number or \"unlimited\""));
    assert!(
        bad(vec![
            n(1.0),
            n(1.0),
            t("pareto"),
            n(-5.0),
            n(2.0),
            Grid::missing()
        ])
        .starts_with("#PROSPICIO! ")
    );
    let two = Grid::from_rows(vec![vec![Value::Number(1.0), Value::Number(2.0)]]);
    let three = Grid::from_rows(vec![vec![Value::Number(1.0); 3]]);
    assert!(bad(pareto(two, three)).contains("do not combine"));
    let ilf = call(
        "pro_ilf",
        vec![
            n(1e6),
            t("unlimited"),
            t("pareto"),
            n(500.0),
            n(2.0),
            Grid::missing(),
        ],
    );
    assert!(text(&ilf).contains("basic_limit"));
}
