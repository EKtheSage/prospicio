//! The one-year view on RAA, reached through the catalogue as Excel
//! reaches it:
//!
//! * `PRO.MACK.CDR` and `PRO.MACK.CDR.YEARS` against R ChainLadder's
//!   `CDR(MackChainLadder(RAA), dev = "all")`
//!   (`validation/reference/reserving_cdr_r.csv`): every origin's reserve,
//!   `CDR(1)S.E.` and Mack S.E., every origin's and the total's `CDR(k)S.E.`;
//! * `PRO.ODP.CDR` and `PRO.MACK.BOOT.CDR`: the same simulated claims
//!   development results as `OdpBootstrap::one_year` and
//!   `MackBootstrap::one_year` with the chain ladder.

mod common;

use std::path::PathBuf;

use common::{n, num, raa, t, text};
use prospicio_prob::Distribution;
use prospicio_reserving::{ChainLadder, MackBootstrap, OdpBootstrap, OneYearMethod};
use prospicio_xll::call;
use prospicio_xll::reserving::triangle;
use prospicio_xll::sheet::{Grid, Value};

/// R's CDR targets on RAA, as (quantity, arg, value).
fn r_cdr() -> Vec<(String, String, f64)> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../validation/reference/reserving_cdr_r.csv");
    std::fs::read_to_string(&p)
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("raa,cdr,"))
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (f[2].to_owned(), f[3].to_owned(), f[4].parse().unwrap())
        })
        .collect()
}

fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-9 * want.abs().max(1.0),
        "{what}: {got} vs {want}"
    );
}

#[test]
fn merz_wuthrich_matches_r() {
    let out = call("pro_mack_cdr", vec![raa()]);
    assert_eq!((out.rows, out.columns), (12, 4));
    assert_eq!(out.get(0, 2), &Value::from("CDR(1) SE"));
    let mut checked = 0;
    for (quantity, arg, want) in r_cdr() {
        let origin = arg.parse::<usize>().ok().and_then(|y| y.checked_sub(1980));
        let (row, col) = match (quantity.as_str(), origin) {
            ("reserve", Some(o)) => (o, 1),
            ("one_year_se", Some(o)) => (o, 2),
            ("mack_se", Some(o)) => (o, 3),
            ("total_reserve", _) => (11, 1),
            ("total_one_year_se", _) => (11, 2),
            ("total_mack_se", _) => (11, 3),
            _ => continue,
        };
        close(num(out.get(row, col)), want, &format!("{quantity} {arg}"));
        checked += 1;
    }
    assert_eq!(checked, 33);
}

#[test]
fn cdr_by_calendar_year_matches_r() {
    let out = call("pro_mack_cdr_years", vec![raa()]);
    assert_eq!((out.rows, out.columns), (12, 10), "nine future years");
    let mut checked = 0;
    for (quantity, arg, want) in r_cdr() {
        let (row, k) = match quantity.as_str() {
            "cdr_se" => {
                let (origin, k) = arg.split_once(':').unwrap();
                (
                    origin.parse::<usize>().unwrap() - 1980,
                    k.parse::<usize>().unwrap(),
                )
            }
            "total_cdr_se" => (11, arg.parse().unwrap()),
            _ => continue,
        };
        // R's last year is past the run-off and zero; the table stops at nine.
        if k == 10 {
            assert_eq!(want, 0.0);
            continue;
        }
        close(num(out.get(row, k)), want, &format!("{quantity} {arg}"));
        checked += 1;
    }
    assert_eq!(checked, 99);
}

fn column_means(cdr: &prospicio_prob::PredictiveDistribution) -> Vec<f64> {
    let k = cdr.n_components();
    let rows = cdr.n_sims() as f64;
    (0..k)
        .map(|j| cdr.draw_matrix().iter().skip(j).step_by(k).sum::<f64>() / rows)
        .collect()
}

#[test]
fn simulated_cdr_is_the_crates() {
    let tri = triangle(&raa()).unwrap();
    let cl = OneYearMethod::ChainLadder(ChainLadder::default());
    let odp = OdpBootstrap {
        n_sims: 2_000,
        seed: 3,
        ..OdpBootstrap::default()
    }
    .one_year(&tri, "values", &cl)
    .unwrap();
    let out = call("pro_odp_cdr", vec![raa(), n(2_000.0), n(3.0)]);
    assert_eq!((out.rows, out.columns), (12, 8));
    assert_eq!(out.get(0, 4), &Value::from("P0.5"));
    for (j, m) in column_means(&odp.cdr).into_iter().enumerate() {
        close(num(out.get(j + 1, 2)), m, &format!("ODP mean CDR {j}"));
        close(
            num(out.get(j + 1, 1)),
            odp.opening_reserve[j],
            "opening reserve",
        );
    }
    close(
        num(out.get(11, 3)),
        odp.cdr.total().std_dev(),
        "ODP total SD",
    );
    // The oldest origin is closed: no development in the coming year.
    assert_eq!(num(out.get(1, 3)), 0.0);

    let mack = MackBootstrap {
        n_sims: 2_000,
        seed: 3,
        ..MackBootstrap::default()
    }
    .one_year(&tri, "values", &cl)
    .unwrap();
    let out = call("pro_mack_boot_cdr", vec![raa(), n(2_000.0), n(3.0)]);
    for (j, m) in column_means(&mack.cdr).into_iter().enumerate() {
        close(num(out.get(j + 1, 2)), m, &format!("Mack mean CDR {j}"));
    }
    close(
        num(out.get(11, 3)),
        mack.cdr.total().std_dev(),
        "Mack total SD",
    );
}

#[test]
fn options_and_errors() {
    let tail = call(
        "pro_odp_cdr",
        vec![raa(), n(500.0), n(0.0), Grid::missing(), n(1.05)],
    );
    assert_eq!(tail.rows, 12, "{:?}", tail.as_scalar());
    let sigma = call(
        "pro_mack_cdr",
        vec![
            raa(),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            Grid::missing(),
            t("mack"),
        ],
    );
    assert!(
        num(sigma.get(11, 3)) > 26_900.0,
        "Mack's rule raises the SE"
    );
    assert!(text(&call("pro_mack_boot_cdr", vec![raa(), n(0.0)])).contains("sims"));
}
