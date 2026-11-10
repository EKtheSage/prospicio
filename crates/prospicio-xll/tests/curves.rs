//! Development curves over every age, reached through the catalogue as
//! Excel reaches them: R's `lm` and `nls` fits of the six shapes on RAA
//! and GenIns (`validation/reference/reserving_curves_r.csv`), and
//! chainladder-python's `TailCurve` attached at 60 months
//! (`reserving_tails_python.csv`, `tail_curve_attach`) by passing
//! `PRO.RES.CURVE`'s selected factors and tail to `PRO.RES.CL`, as a
//! sheet chains them.

use crate::common;

use std::collections::BTreeMap;

use common::{genins, n, num, raa, reference, t};
use prospicio_xll::call as run;
use prospicio_xll::sheet::{Grid, Value};

fn call(symbol: &str, args: Vec<Grid>) -> Grid {
    let g = run(symbol, args);
    if let Some(Value::Text(s)) = g.as_scalar() {
        assert!(!s.starts_with("#PROSPICIO!"), "{symbol}: {s}");
    }
    g
}

fn error(symbol: &str, args: Vec<Grid>) -> String {
    match run(symbol, args).as_scalar() {
        Some(Value::Text(s)) if s.starts_with("#PROSPICIO!") => s.clone(),
        other => panic!("{symbol}: expected an error, got {other:?}"),
    }
}

fn dataset(name: &str) -> Grid {
    match name {
        "raa" => raa(),
        "genins" => genins().0,
        other => panic!("no dataset {other}"),
    }
}

fn blank() -> Grid {
    Grid::scalar(Value::Empty)
}

fn column(table: &Grid, name: &str) -> usize {
    (0..table.columns)
        .find(|&c| *table.get(0, c) == Value::from(name))
        .unwrap_or_else(|| panic!("no column {name}"))
}

#[test]
fn matches_r_lm_and_nls() {
    let mut fits: BTreeMap<(String, String), (Grid, Grid)> = BTreeMap::new();
    let mut checked = 0;
    for row in reference("reserving_curves_r.csv") {
        let (ds, method, quantity, arg) = (&row[0], &row[1], row[2].as_str(), &row[3]);
        let (curve, stats) = fits.entry((ds.clone(), method.clone())).or_insert_with(|| {
            (
                call("pro_res_curve", vec![dataset(ds), t(method)]),
                call("pro_res_curve_stats", vec![dataset(ds), t(method)]),
            )
        });
        let got = match quantity {
            "param_a" | "param_omega" => num(stats.get(1, 1)),
            "param_b" | "param_theta" => num(stats.get(1, 2)),
            "param_c" => num(stats.get(1, 3)),
            "fitted" => num(curve.get(arg.parse::<usize>().unwrap() + 1, 2)),
            "sum_of_squares" => num(stats.get(1, 6)),
            "r_squared" => num(stats.get(1, 7)),
            "tail" => num(stats.get(1, 9)),
            other => panic!("unknown quantity {other}"),
        };
        let (want, abs, rel): (f64, f64, f64) = (
            row[4].parse().unwrap(),
            row[5].parse().unwrap(),
            row[6].parse().unwrap(),
        );
        assert!(
            (got - want).abs() <= abs.max(rel * want.abs()),
            "{ds} {method} {quantity} {arg}: {got} vs {want}"
        );
        checked += 1;
    }
    assert!(checked > 100, "{checked}");
    // The curve's tail in PRO.RES.CURVE is the comparison's.
    let (curve, stats) = &fits[&("raa".to_owned(), "sherman".to_owned())];
    assert_eq!(curve.get(curve.rows - 1, 2), stats.get(1, 9));
    assert_eq!(*curve.get(curve.rows - 1, 0), Value::from("120-Ult"));
}

/// `PRO.RES.CL` on the curve's selected factors and tail.
fn chain_ladder_on_curve(ds: &str) -> Grid {
    let curve = call(
        "pro_res_curve",
        vec![dataset(ds), blank(), blank(), blank(), n(60.0)],
    );
    let sel = column(&curve, "Selected");
    let links: Vec<f64> = (1..curve.rows - 1)
        .map(|r| num(curve.get(r, sel)))
        .collect();
    let tail = num(curve.get(curve.rows - 1, sel));
    let labels: Vec<f64> = match ds {
        "raa" => (1981..=1990).map(f64::from).collect(),
        _ => (2001..=2010).map(f64::from).collect(),
    };
    call(
        "pro_res_cl",
        vec![
            dataset(ds),
            blank(),
            n(tail),
            Grid::column(&labels),
            Grid::from_rows(vec![links.into_iter().map(Value::Number).collect()]),
        ],
    )
}

#[test]
fn chained_into_the_chain_ladder_matches_chainladder_python_tail_curve() {
    let mut fits: BTreeMap<String, Grid> = BTreeMap::new();
    let mut checked = 0;
    for row in reference("reserving_tails_python.csv") {
        if row[1] != "tail_curve_attach" || !["raa", "genins"].contains(&row[0].as_str()) {
            continue;
        }
        let cl = fits
            .entry(row[0].clone())
            .or_insert_with(|| chain_ladder_on_curve(&row[0]));
        let total = cl.rows - 1;
        let origin = || {
            let year: f64 = row[3].parse().unwrap();
            (1..cl.rows)
                .find(|&r| *cl.get(r, 0) == Value::Number(year))
                .unwrap()
        };
        let got = match row[2].as_str() {
            "ultimate" => num(cl.get(origin(), column(cl, "Ultimate"))),
            "reserve" => num(cl.get(origin(), column(cl, "Reserve"))),
            "total_ultimate" => num(cl.get(total, column(cl, "Ultimate"))),
            "total_reserve" => num(cl.get(total, column(cl, "Reserve"))),
            _ => continue,
        };
        let want: f64 = row[4].parse().unwrap();
        assert!(
            (got - want).abs() <= 1e-6 * want.abs().max(1.0),
            "{} {}: {got} vs {want}",
            row[0],
            row[2]
        );
        checked += 1;
    }
    assert_eq!(checked, 2 * (2 * 10 + 2));
}

#[test]
fn stats_compares_every_shape_and_names_a_failure() {
    let s = call("pro_res_curve_stats", vec![raa()]);
    assert_eq!(s.rows, 7);
    let names: Vec<&Value> = (1..7).map(|r| s.get(r, 0)).collect();
    assert_eq!(names[0], &Value::from("exponential"));
    assert_eq!(names[5], &Value::from("weibull"));
    // Hoerl and Sherman have a third parameter, the others none.
    assert!(matches!(s.get(3, 3), Value::Number(_)));
    assert_eq!(*s.get(1, 3), Value::Empty);
    // Fitted to two ages only, Sherman (four points needed) shows why.
    let s = call(
        "pro_res_curve_stats",
        vec![
            raa(),
            blank(),
            blank(),
            Grid::from_rows(vec![vec![Value::Number(12.0), Value::Number(48.0)]]),
        ],
    );
    let sherman = (1..s.rows)
        .find(|&r| *s.get(r, 0) == Value::from("sherman"))
        .unwrap();
    assert!(
        matches!(s.get(sherman, 1), Value::Text(_)),
        "{:?}",
        s.get(sherman, 1)
    );
    assert!(matches!(s.get(1, 1), Value::Number(_)));
}

#[test]
fn keeps_the_observed_factors_before_the_age_given() {
    let c = call(
        "pro_res_curve",
        vec![raa(), blank(), blank(), blank(), n(48.0), n(0.0)],
    );
    let (obs, fit, sel) = (
        column(&c, "Observed"),
        column(&c, "Fitted"),
        column(&c, "Selected"),
    );
    for r in 1..c.rows - 1 {
        let want = if r < 4 { obs } else { fit };
        assert_eq!(c.get(r, sel), c.get(r, want), "row {r}");
    }
    // tail_periods 0: no tail.
    assert_eq!(num(c.get(c.rows - 1, sel)), 1.0);
}

#[test]
fn errors_name_the_argument() {
    assert!(error("pro_res_curve", vec![raa(), t("gompertz")]).contains("shape must be"));
    assert!(
        error(
            "pro_res_curve",
            vec![raa(), blank(), blank(), blank(), n(500.0)]
        )
        .contains("keep_observed_until")
    );
    assert!(error("pro_res_curve", vec![raa(), blank(), blank(), t("x")]).contains("fit_period"));
    assert!(
        error(
            "pro_res_curve_stats",
            vec![raa(), blank(), blank(), blank(), n(-1.0)]
        )
        .contains("tail_periods")
    );
}
