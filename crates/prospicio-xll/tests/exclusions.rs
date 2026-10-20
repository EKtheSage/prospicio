//! Link-ratio exclusions in the `average` argument (`docs/design/xll.md`,
//! decision 34), reached through the catalogue as Excel reaches them:
//! averages over the latest calendar periods with the highest and lowest
//! link ratios left out, against chainladder-python's `Development(n_periods,
//! drop_high, drop_low)` on RAA and GenIns
//! (`validation/reference/reserving_averages_python.csv`) and R
//! ChainLadder's `MackChainLadder(weights = W)`
//! (`validation/reference/reserving_averages_r.csv`), at their tolerances;
//! the ODP bootstrap's exclusions against `prospicio_reserving`'s own.

mod common;

use common::{genins, n, num, raa, reference, t, text};
use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, OdpBootstrap, Triangle};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

/// A reference method such as `average=volume;n_periods=3;drop_high=1` as
/// the `average` row `kind, n_periods, drop_high, drop_low`.
fn average_row(method: &str) -> Grid {
    let get = |name: &str| {
        method
            .split(';')
            .find_map(|p| p.strip_prefix(name)?.strip_prefix('='))
    };
    let number = |name: &str| get(name).map_or(Value::Empty, |v| Value::Number(v.parse().unwrap()));
    Grid::from_rows(vec![vec![
        Value::from(get("average").unwrap_or("volume")),
        number("n_periods"),
        number("drop_high"),
        number("drop_low"),
    ]])
}

fn dataset(name: &str) -> Grid {
    match name {
        "raa" => raa(),
        "genins" => genins().0,
        other => panic!("{other}"),
    }
}

fn close(got: f64, row: &[String]) -> bool {
    let (want, abs, rel): (f64, f64, f64) = (
        row[4].parse().unwrap(),
        row[5].parse().unwrap(),
        row[6].parse().unwrap(),
    );
    (got - want).abs() <= abs.max(rel * want.abs())
}

#[test]
fn factors_and_ultimates_match_chainladder_python() {
    let first_year = |d: &str| if d == "raa" { 1981 } else { 2001 };
    let mut checked = 0;
    for row in reference("reserving_averages_python.csv") {
        let (data, method, quantity) = (row[0].as_str(), row[1].as_str(), row[2].as_str());
        if data == "clrd_wkcomp" || !matches!(quantity, "ldf" | "ultimate") {
            continue;
        }
        let tri = dataset(data);
        let arg: usize = row[3].parse().unwrap();
        let got = if quantity == "ldf" {
            let ldf = call("pro_cl_ldf", vec![tri, average_row(method)]);
            num(&ldf.cells[arg / 12 - 1])
        } else {
            let ult = call("pro_cl_ultimate", vec![tri, average_row(method)]);
            num(&ult.cells[arg - first_year(data)])
        };
        assert!(
            close(got, &row),
            "{data} {method} {quantity} {arg}: {got} against {}",
            row[4]
        );
        checked += 1;
    }
    assert_eq!(checked, 570);
}

#[test]
fn mack_matches_r_with_weights() {
    let mut checked = 0;
    for row in reference("reserving_averages_r.csv") {
        let (data, method, quantity) = (row[0].as_str(), row[1].as_str(), row[2].as_str());
        let mack = || call("pro_mack", vec![dataset(data), average_row(method)]);
        let first_year = if data == "raa" { 1981 } else { 2001 };
        let got = match quantity {
            "se" => mack()
                .get(row[3].parse::<usize>().unwrap() - first_year + 1, 6)
                .clone(),
            "total_standard_error" => {
                let m = mack();
                m.get(m.rows - 1, 6).clone()
            }
            _ => continue,
        };
        match got {
            // Excel holds no infinity: an infinite standard error is blank.
            Value::Empty => assert_eq!(row[4], "Inf", "{data} {method} {quantity} {}", row[3]),
            v => assert!(
                close(num(&v), &row),
                "{data} {method} {quantity} {}: {} against {}",
                row[3],
                num(&v),
                row[4]
            ),
        }
        checked += 1;
    }
    assert!(checked >= 100, "{checked}");
}

/// RAA as the crate's triangle.
fn raa_triangle() -> Triangle {
    let (mut origin, mut age, mut value) = (vec![], vec![], vec![]);
    for (i, row) in common::RAA.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            origin.push(Month::january(1981 + i as i32));
            age.push(12 * (j as u32 + 1));
            value.push(x);
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&age),
        values: &[("paid", &value)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap()
}

/// The ODP bootstrap keeps the link ratios the `average` row keeps: its
/// chain-ladder reserves and simulated mean are the crate's under the same
/// exclusions, and the draws differ from those on every link ratio.
#[test]
fn the_odp_bootstrap_takes_the_exclusions() {
    let row = Grid::from_rows(vec![vec![
        Value::from("volume"),
        Value::Number(5.0),
        Value::Number(1.0),
        Value::Number(1.0),
    ]]);
    let none = Grid::missing;
    let sheet = call(
        "pro_odp",
        vec![
            raa(),
            n(2000.0),
            n(3.0),
            none(),
            none(),
            none(),
            none(),
            row.clone(),
        ],
    );
    let fit = OdpBootstrap {
        n_sims: 2000,
        seed: 3,
        n_periods: Some(5),
        drop_high: 1,
        drop_low: 1,
        ..OdpBootstrap::default()
    }
    .fit(&raa_triangle(), "paid")
    .unwrap();
    let total = sheet.rows - 1;
    assert!((num(sheet.get(total, 1)) - fit.chain_ladder.total_reserve()).abs() < 1e-6);
    let mean = prospicio_prob::Distribution::mean(fit.reserves.total());
    assert!((num(sheet.get(total, 2)) - mean).abs() < 1e-6 * mean);
    let plain = call("pro_odp", vec![raa(), n(2000.0), n(3.0)]);
    assert_ne!(plain.get(total, 2), sheet.get(total, 2));

    // The one-year view and the draws take the same row.
    let cdr = call(
        "pro_odp_cdr",
        vec![
            raa(),
            n(500.0),
            n(3.0),
            none(),
            none(),
            none(),
            none(),
            row.clone(),
        ],
    );
    assert!(matches!(cdr.get(1, 1), Value::Number(_)), "{cdr:?}");
    let sims = call(
        "pro_odp_sims",
        vec![raa(), n(10.0), n(3.0), none(), none(), none(), row],
    );
    assert_eq!(sims.rows, 11);
}

/// The plain chain ladder is the `average` row with its exclusions blank;
/// the geometric average reaches the chain ladder but not Mack.
#[test]
fn kinds_and_errors() {
    let plain = call("pro_cl_ldf", vec![raa()]);
    let row = Grid::from_rows(vec![vec![Value::from("volume"), Value::Empty]]);
    assert_eq!(call("pro_cl_ldf", vec![raa(), row]), plain);

    let geo = call("pro_cl_ldf", vec![raa(), t("geometric")]);
    let first: Vec<f64> = common::RAA
        .iter()
        .filter(|r| r.len() > 1)
        .map(|r| (r[1] / r[0]).ln())
        .collect();
    let want = (first.iter().sum::<f64>() / first.len() as f64).exp();
    assert!((num(&geo.cells[0]) - want).abs() < 1e-12 * want);
    let e = call("pro_mack", vec![raa(), t("geometric")]);
    assert!(
        text(&e).to_lowercase().contains("geometric"),
        "{}",
        text(&e)
    );

    let e = call("pro_cl_ldf", vec![raa(), t("medial")]);
    assert!(text(&e).contains("average must be"), "{}", text(&e));
    let e = call("pro_cl_ldf", vec![raa(), Grid::row(&[5.0, 1.0])]);
    assert!(text(&e).contains("first cell is its kind"), "{}", text(&e));
    let zero = Grid::from_rows(vec![vec![Value::from("volume"), Value::Number(0.0)]]);
    let e = call("pro_cl_ldf", vec![raa(), zero]);
    assert!(text(&e).contains("n_periods"), "{}", text(&e));
    let e = call(
        "pro_odp",
        vec![
            raa(),
            n(100.0),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            t("simple"),
        ],
    );
    assert!(text(&e).contains("volume-weighted"), "{}", text(&e));
}
