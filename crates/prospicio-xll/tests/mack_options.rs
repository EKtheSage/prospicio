//! The chain ladder's and Mack's options, reached through the catalogue
//! as Excel reaches them, on RAA:
//!
//! * `sigma = "mack"` against R ChainLadder's `MackChainLadder(RAA,
//!   est.sigma = "Mack")` (`validation/reference/reserving_chainladder_r.csv`):
//!   every link's sigma and standard error, and the total Mack SE;
//! * `selected` factors against chainladder-python 0.10.1's
//!   `DevelopmentConstant` (`validation/reference/reserving_patterns_python.csv`):
//!   age-to-age factors, factors to ultimate, and a 0.6 blend with the
//!   estimates.

mod common;

use std::path::PathBuf;

use common::{num, raa, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn reference(name: &str) -> Vec<Vec<String>> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../validation/reference")
        .join(name);
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split(',').map(str::to_owned).collect())
        .collect()
}

/// `quantity` for `dataset` and `method`, by its `arg`.
fn targets(file: &str, dataset: &str, method: &str, quantity: &str) -> Vec<(String, f64)> {
    reference(file)
        .into_iter()
        .filter(|r| r[0] == dataset && r[1] == method && r[2] == quantity)
        .map(|r| (r[3].clone(), r[4].parse().unwrap()))
        .collect()
}

fn none() -> Grid {
    Grid::missing()
}

#[test]
fn mack_sigma_rule_matches_r() {
    let file = "reserving_chainladder_r.csv";
    let factors = call(
        "pro_cl_factors",
        vec![raa(), none(), none(), none(), none(), none(), t("mack")],
    );
    for (quantity, col) in [("sigma", 3), ("f_se", 4)] {
        let expected = targets(file, "raa", "mack_sigma_mack", quantity);
        assert_eq!(expected.len(), 9, "{quantity}");
        for (k, x) in expected {
            let k: usize = k.parse().unwrap();
            let got = num(factors.get(k + 1, col));
            assert!(
                (got - x).abs() <= 1e-9 * x.abs(),
                "{quantity} {k}: {got} vs {x}"
            );
        }
    }
    let total = targets(file, "raa", "mack_sigma_mack", "total_standard_error")[0].1;
    let mack = call(
        "pro_mack",
        vec![
            raa(),
            none(),
            none(),
            none(),
            none(),
            none(),
            none(),
            t("mack"),
        ],
    );
    assert!((num(mack.get(11, 6)) - total).abs() <= 1e-9 * total);
    // The default rule is log-linear, R's default.
    let default = call("pro_mack", vec![raa()]);
    let log_linear = targets(file, "raa", "mack", "total_standard_error")[0].1;
    assert!((num(default.get(11, 6)) - log_linear).abs() <= 1e-9 * log_linear);
    assert!(
        text(&call(
            "pro_mack",
            vec![
                raa(),
                none(),
                none(),
                none(),
                none(),
                none(),
                none(),
                t("mean")
            ]
        ))
        .contains("sigma must be")
    );
}

/// The selected values of a chainladder-python case, as a row under the
/// triangle's columns (the age-12 value in the first).
fn selected_row(method: &str) -> Grid {
    let mut cells = vec![Value::Empty; 10];
    for (age, x) in targets("reserving_patterns_python.csv", "raa", method, "pattern") {
        let age: usize = age.parse().unwrap();
        cells[age / 12 - 1] = Value::Number(x);
    }
    Grid::from_rows(vec![cells])
}

fn total_ultimate(method: &str) -> f64 {
    targets(
        "reserving_patterns_python.csv",
        "raa",
        method,
        "total_ultimate",
    )[0]
    .1
}

fn cl(selected: Grid, kind: &str, weight: Grid) -> Grid {
    call(
        "pro_cl",
        vec![raa(), none(), none(), none(), selected, t(kind), weight],
    )
}

#[test]
fn selected_factors_match_chainladder_python() {
    for (method, kind) in [
        ("chain_ladder;style=ldf", "ldf"),
        ("chain_ladder;style=cdf", "cdf"),
    ] {
        let out = cl(selected_row(method), kind, none());
        let expected = total_ultimate(method);
        let got = num(out.get(11, 3));
        assert!(
            (got - expected).abs() <= 1e-9 * expected,
            "{method}: {got} vs {expected}"
        );
    }
    // chainladder-python has no blend: the script blends 0.6 s + 0.4 e itself.
    let blend = cl(
        selected_row("chain_ladder;style=ldf"),
        "ldf",
        Grid::scalar(Value::Number(0.6)),
    );
    let expected = total_ultimate("chain_ladder;style=ldf;weight=0.6");
    assert!((num(blend.get(11, 3)) - expected).abs() <= 1e-9 * expected);

    // The selection reaches every chain-ladder function and Mack.
    let row = selected_row("chain_ladder;style=ldf");
    let ldf = call("pro_cl_ldf", vec![raa(), none(), none(), row.clone()]);
    assert_eq!(ldf.cells[0], row.cells[0]);
    let mack = call("pro_mack", vec![raa(), none(), none(), none(), row.clone()]);
    assert!((num(mack.get(11, 2)) - total_ultimate("chain_ladder;style=ldf")).abs() < 1e-6);
}

#[test]
fn a_partial_selection_keeps_the_other_estimates() {
    let estimated = call("pro_cl_ldf", vec![raa()]);
    let mut cells = vec![Value::Empty; 10];
    cells[0] = Value::Number(3.5);
    let partial = call(
        "pro_cl_ldf",
        vec![raa(), none(), none(), Grid::from_rows(vec![cells])],
    );
    assert_eq!(num(&partial.cells[0]), 3.5);
    assert_eq!(partial.cells[1..], estimated.cells[1..]);
}

#[test]
fn selection_errors() {
    let bad = Grid::from_rows(vec![vec![Value::from("x"); 10]]);
    assert!(text(&cl(bad, "ldf", none())).contains("must be a number"));
    assert!(
        text(&cl(
            selected_row("chain_ladder;style=ldf"),
            "fraction",
            none()
        ))
        .contains("kind must be")
    );
    assert!(text(&cl(none(), "cdf", none())).contains("need selected factors"));
}
