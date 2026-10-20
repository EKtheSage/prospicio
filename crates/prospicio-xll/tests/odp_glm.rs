//! The ODP GLM, reached through the catalogue as Excel reaches it: the
//! coefficients, standard errors, dispersion and reserves of R's
//! `glm(inc ~ factor(origin) + factor(dev), family = quasipoisson)` on
//! GenIns and ABC (`validation/reference/reserving_glm_r.csv`) at its
//! tolerances, and Taylor and McGuire's printed prediction errors (CAS
//! Monograph 3, Table 5-4) and residual diagnostics (Chapter 6).

mod common;

use std::collections::BTreeMap;

use common::{genins, num, raa, reference, t, text, validation};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

/// A triangle of `validation/data/<name>.csv` (origin, development,
/// cumulative value) as a sheet range, and its origin years as labels.
fn sheet_triangle(name: &str) -> (Grid, Grid) {
    if name == "genins" {
        let labels: Vec<f64> = (2001..=2010).map(f64::from).collect();
        return (genins().0, Grid::column(&labels));
    }
    let mut rows: BTreeMap<i32, Vec<f64>> = BTreeMap::new();
    for line in validation(&format!("data/{name}.csv"))
        .lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
    {
        let f: Vec<&str> = line.split(',').collect();
        rows.entry(f[0].parse().unwrap())
            .or_default()
            .push(f[2].parse().unwrap());
    }
    let ages = rows.values().map(Vec::len).max().unwrap();
    let labels: Vec<f64> = rows.keys().map(|&y| f64::from(y)).collect();
    let grid = Grid::from_rows(
        rows.values()
            .map(|r| {
                (0..ages)
                    .map(|c| r.get(c).map_or(Value::Empty, |&x| Value::Number(x)))
                    .collect()
            })
            .collect(),
    );
    (grid, Grid::column(&labels))
}

fn close(got: f64, row: &[String]) -> bool {
    let (want, abs, rel): (f64, f64, f64) = (
        row[4].parse().unwrap(),
        row[5].parse().unwrap(),
        row[6].parse().unwrap(),
    );
    (got - want).abs() <= abs.max(rel * want.abs())
}

/// The row of `PRO.ODP.GLM.PARAMS` for R's coefficient name.
fn coefficient_row(params: &Grid, name: &str) -> usize {
    let (term, level) = if name == "(Intercept)" {
        ("Intercept", Value::Empty)
    } else if let Some(year) = name.strip_prefix("origin[") {
        let year: f64 = year.trim_end_matches(']').parse().unwrap();
        ("Origin", Value::Number(year))
    } else {
        let age: f64 = name
            .strip_prefix("development[")
            .unwrap()
            .trim_end_matches(']')
            .parse()
            .unwrap();
        ("Development", Value::Number(age / 12.0))
    };
    (1..params.rows)
        .find(|&r| *params.get(r, 0) == Value::from(term) && *params.get(r, 1) == level)
        .unwrap_or_else(|| panic!("no row for {name}"))
}

#[test]
fn matches_r_glm() {
    let mut checked = 0;
    for row in reference("reserving_glm_r.csv") {
        // `PRO.ODP.GLM` fits the ODP; the gamma and Tweedie rows are the
        // crate's (`validation/tests/reserving.rs`).
        if row[1] != "odp_glm" {
            continue;
        }
        let (data, quantity, arg) = (row[0].as_str(), row[2].as_str(), row[3].as_str());
        let (tri, labels) = sheet_triangle(data);
        let first_year = if data == "genins" { 2001 } else { 1977 };
        let got = match quantity {
            "coefficient" | "std_error" => {
                let p = call("pro_odp_glm_params", vec![tri, labels]);
                let r = coefficient_row(&p, arg);
                num(p.get(r, if quantity == "coefficient" { 2 } else { 3 }))
            }
            "dispersion" => {
                let p = call("pro_odp_glm_params", vec![tri, labels]);
                assert_eq!(*p.get(p.rows - 1, 0), Value::from("Dispersion"));
                num(p.get(p.rows - 1, 2))
            }
            "reserve" => {
                let s = call("pro_odp_glm", vec![tri, labels]);
                let year: i32 = arg.parse().unwrap();
                let r = (year - first_year) as usize + 1;
                assert_eq!(num(s.get(r, 0)), f64::from(year));
                num(s.get(r, 3))
            }
            "total_reserve" => {
                let s = call("pro_odp_glm", vec![tri, labels]);
                assert_eq!(*s.get(s.rows - 1, 0), Value::from("Total"));
                num(s.get(s.rows - 1, 3))
            }
            other => panic!("{other}"),
        };
        assert!(
            close(got, &row),
            "{data} {quantity} {arg}: {got} against {}",
            row[4]
        );
        checked += 1;
    }
    assert_eq!(checked, 105);
}

/// Taylor and McGuire's Table 1-1, incremental paid, accident years 1988
/// to 1997, cumulated as a sheet range.
fn taylor_mcguire() -> Grid {
    const INCREMENTS: [&[f64]; 10] = [
        &[
            41821., 34729., 20147., 15965., 11285., 5924., 4775., 3742., 3435., 2958.,
        ],
        &[
            48167., 39495., 24444., 18178., 10840., 7379., 5683., 4758., 3959.,
        ],
        &[52058., 47459., 27359., 17916., 11448., 8846., 5869., 5391.],
        &[57251., 49510., 27036., 20871., 14304., 10552., 7742.],
        &[59213., 54129., 29566., 22484., 14114., 10000.],
        &[59475., 52076., 26836., 22332., 14756.],
        &[65607., 44648., 27062., 22655.],
        &[56748., 39315., 26748.],
        &[52212., 40030.],
        &[43962.],
    ];
    Grid::from_rows(
        INCREMENTS
            .iter()
            .map(|row| {
                let mut c = 0.0;
                (0..10)
                    .map(|j| {
                        row.get(j).map_or(Value::Empty, |x| {
                            c += x;
                            Value::Number(c)
                        })
                    })
                    .collect()
            })
            .collect(),
    )
}

#[test]
fn prediction_error_matches_taylor_mcguire_table_5_4() {
    let s = call("pro_odp_glm", vec![taylor_mcguire()]);
    assert_eq!((s.rows, s.columns), (12, 8));
    assert_eq!(*s.get(0, 6), Value::from("RMSEP"));
    // Forecast and RMSEP by accident year 1989 to 1997, then the total.
    let table = [
        (3_398.0, 924.0),
        (8_155.0, 1_363.0),
        (14_579.0, 1_775.0),
        (22_645.0, 2_169.0),
        (31_865.0, 2_523.0),
        (45_753.0, 3_036.0),
        (60_093.0, 3_577.0),
        (80_983.0, 4_538.0),
        (105_874.0, 6_786.0),
        (373_346.0, 14_076.0),
    ];
    for (k, (forecast, rmsep)) in table.into_iter().enumerate() {
        let r = k + 2;
        assert!((num(s.get(r, 3)) - forecast).abs() <= 0.5, "row {r}");
        assert!((num(s.get(r, 6)) - rmsep).abs() <= 0.5, "row {r}");
        let cv = num(s.get(r, 7));
        assert!((cv - num(s.get(r, 6)) / num(s.get(r, 3))).abs() < 1e-15);
    }
    // The fully developed 1988: no reserve, no error, no CV.
    assert_eq!((num(s.get(1, 3)), num(s.get(1, 6))), (0.0, 0.0));
    assert_eq!(*s.get(1, 7), Value::Empty);
    // The total's split.
    assert!((num(s.get(11, 4)) - 6_539.0).abs() <= 0.5);
    assert!((num(s.get(11, 5)) - 12_465.0).abs() <= 0.5);
    // Ultimate is latest plus reserve.
    for r in 1..12 {
        let (l, u, res) = (num(s.get(r, 1)), num(s.get(r, 2)), num(s.get(r, 3)));
        assert!((u - l - res).abs() < 1e-6 * u);
    }
}

#[test]
fn residuals_match_taylor_mcguire_chapter_6() {
    let cells = call("pro_odp_glm_residuals", vec![taylor_mcguire()]);
    assert_eq!((cells.rows, cells.columns), (56, 11));
    // 1988's last year and 1997's first have leverage 1: no standardized
    // residual.
    let blank: Vec<usize> = (1..56)
        .filter(|&r| *cells.get(r, 9) == Value::Empty)
        .collect();
    assert_eq!(blank, [10, 55]);
    assert_eq!(*cells.get(55, 10), Value::Empty);
    // Origin, development and calendar of 1989's third year.
    let r = 10 + 3;
    assert_eq!(
        (
            num(cells.get(r, 0)),
            num(cells.get(r, 1)),
            num(cells.get(r, 2))
        ),
        (2.0, 3.0, 4.0)
    );
    let leverage: f64 = (1..56).map(|r| num(cells.get(r, 8))).sum();
    assert!((leverage - 19.0).abs() < 1e-8);
    // The actual-to-fitted heat map, Figure 6-22, in percent, row 1988.
    let first: Vec<i64> = (1..=10)
        .map(|r| (100.0 * num(cells.get(r, 5))).round() as i64)
        .collect();
    assert_eq!(first, [98, 100, 100, 104, 113, 87, 96, 92, 100, 100]);

    // The spread by development year 1 to 8 (divisor n in the guide) and
    // the mean by calendar year, to two decimals.
    let by_age = call(
        "pro_odp_glm_residuals",
        vec![taylor_mcguire(), t("development")],
    );
    assert_eq!((by_age.rows, by_age.columns), (11, 4));
    let spread = [1.34, 1.59, 0.88, 0.62, 0.83, 0.77, 0.65, 0.40];
    for (k, want) in spread.into_iter().enumerate() {
        let r = k + 1;
        assert_eq!(num(by_age.get(r, 0)), r as f64);
        let n = num(by_age.get(r, 1));
        let population = num(by_age.get(r, 3)) * ((n - 1.0) / n).sqrt();
        assert!((population - want).abs() <= 0.005, "age {r}: {population}");
    }
    // Age 10's only cell is fitted exactly: no mean, no SD.
    assert_eq!(num(by_age.get(10, 1)), 0.0);
    assert_eq!(*by_age.get(10, 2), Value::Empty);
    let by_calendar = call(
        "pro_odp_glm_residuals",
        vec![taylor_mcguire(), t("calendar")],
    );
    let means = [
        -0.38, -0.13, -0.40, 0.56, 0.42, -0.43, 0.23, -0.27, -0.26, 0.32,
    ];
    for (k, want) in means.into_iter().enumerate() {
        assert_eq!(num(by_calendar.get(k + 1, 0)), k as f64 + 1.0);
        assert!((num(by_calendar.get(k + 1, 2)) - want).abs() <= 0.005);
    }
    let labels = Grid::column(&(1988..=1997).map(f64::from).collect::<Vec<_>>());
    let by_origin = call(
        "pro_odp_glm_residuals",
        vec![taylor_mcguire(), t("origin"), labels],
    );
    assert_eq!(*by_origin.get(0, 0), Value::from("Origin"));
    assert_eq!(num(by_origin.get(1, 0)), 1988.0);
    let n: f64 = (1..11).map(|r| num(by_origin.get(r, 1))).sum();
    assert_eq!(n, 53.0);
}

/// The reserves are the chain ladder's, a negative increment and all.
#[test]
fn reserves_are_the_chain_ladders() {
    let s = call("pro_odp_glm", vec![raa()]);
    let cl = call("pro_cl_reserve", vec![raa()]);
    for i in 0..10 {
        assert!((num(s.get(i + 1, 3)) - num(&cl.cells[i])).abs() < 1e-6);
    }
    let p = call("pro_odp_glm_params", vec![raa()]);
    assert_eq!((p.rows, p.columns), (21, 4));
    assert_eq!(*p.get(1, 0), Value::from("Intercept"));
    assert_eq!(
        (p.get(2, 0).clone(), num(p.get(2, 1))),
        (Value::from("Origin"), 2.0)
    );
    assert_eq!(
        (p.get(11, 0).clone(), num(p.get(11, 1))),
        (Value::from("Development"), 2.0)
    );
    assert_eq!(num(p.get(19, 1)), 10.0);
}

#[test]
fn bad_inputs_are_named() {
    let e = call("pro_odp_glm_residuals", vec![raa(), t("diagonal")]);
    assert!(text(&e).contains("by must be"), "{}", text(&e));
    let e = call("pro_odp_glm", vec![raa(), Grid::column(&[1.0, 2.0])]);
    assert!(text(&e).contains("origins must be"), "{}", text(&e));
    // An origin whose only increment is negative has no positive mean.
    let tri = Grid::from_rows(vec![
        vec![Value::Number(100.0), Value::Number(150.0)],
        vec![Value::Number(-10.0), Value::Empty],
    ]);
    let e = call("pro_odp_glm", vec![tri]);
    assert!(text(&e).starts_with("#PROSPICIO! "), "{e:?}");
}
