//! The Pareto helpers (`PRO.PARETO.EXTRAPOLATE`, `PRO.PARETO.ALPHA`,
//! `PRO.PARETO.ALPHA.FREQLAYER`, `PRO.PARETO.ALPHA.FREQS`), reached through
//! the catalogue as Excel reaches them, against every case of the R package
//! Pareto 2.4.5 in `validation/reference/pareto_rating_r.csv`, at its
//! tolerances, truncated cases included.

mod common;

use std::collections::BTreeMap;

use common::{n, num, reference, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::Grid;

#[test]
fn matches_the_r_package_pareto() {
    let mut checked = 0;
    for row in reference("pareto_rating_r.csv") {
        if row[0] == "quantity" {
            continue;
        }
        let p: BTreeMap<&str, &str> = row[1]
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .collect();
        // A value as Excel would hold it: "Inf" is "unlimited".
        let v = |k: &str| -> Grid {
            match p.get(k).copied() {
                Some("Inf") => t("unlimited"),
                Some(x) => n(x.parse().unwrap()),
                None => Grid::missing(),
            }
        };
        let out = match row[0].as_str() {
            "extrapolation" => call(
                "pro_pareto_extrapolate",
                vec![
                    v("c1"),
                    v("a1"),
                    v("c2"),
                    v("a2"),
                    v("alpha"),
                    v("truncation"),
                ],
            ),
            "alpha_between_layers" => call(
                "pro_pareto_alpha",
                vec![
                    v("c1"),
                    v("a1"),
                    v("e1"),
                    v("c2"),
                    v("a2"),
                    v("e2"),
                    v("truncation"),
                ],
            ),
            "alpha_between_frequency_and_layer" => call(
                "pro_pareto_alpha_freqlayer",
                vec![
                    v("threshold"),
                    v("frequency"),
                    v("c"),
                    v("a"),
                    v("e"),
                    v("truncation"),
                ],
            ),
            "alpha_between_frequencies" => call(
                "pro_pareto_alpha_freqs",
                vec![v("t1"), v("f1"), v("t2"), v("f2"), v("truncation")],
            ),
            other => panic!("unknown quantity {other}"),
        };
        let got = num(out.as_scalar().unwrap_or_else(|| panic!("{row:?}")));
        let expected: f64 = row[2].parse().unwrap();
        let (abs_tol, rel_tol): (f64, f64) = (row[3].parse().unwrap(), row[4].parse().unwrap());
        assert!(
            (got - expected).abs() <= abs_tol.max(rel_tol * expected.abs()),
            "{} {}: {got} vs {expected}",
            row[0],
            row[1]
        );
        checked += 1;
    }
    assert_eq!(checked, 58);
}

#[test]
fn extrapolation_broadcasts_and_errors_are_text() {
    // At alpha 2, 2m xs 2m expects half of 1m xs 1m; 1m xs 2m a third.
    let limits = Grid::from_rows(vec![vec![
        prospicio_xll::sheet::Value::Number(2e6),
        prospicio_xll::sheet::Value::Number(1e6),
    ]]);
    let out = call(
        "pro_pareto_extrapolate",
        vec![n(1e6), n(1e6), limits, n(2e6), n(2.0)],
    );
    assert_eq!((out.rows, out.columns), (1, 2));
    assert!((num(&out.cells[0]) - 0.5).abs() < 1e-14);
    assert!((num(&out.cells[1]) - 1.0 / 3.0).abs() < 1e-14);
    assert!(
        text(&call(
            "pro_pareto_alpha_freqs",
            vec![n(1e6), n(1.0), n(2e6), n(4.0)]
        ))
        .starts_with("#PROSPICIO!")
    );
}
