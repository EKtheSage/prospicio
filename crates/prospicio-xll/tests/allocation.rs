//! Capital allocation on a sheet (`PRO.ALLOCATE`, `PRO.ALLOCATE.TOTAL`,
//! `docs/design/xll.md` decision 45), reached through the catalogue as
//! Excel reaches it:
//!
//! * every case of `validation/reference/allocation_numpy.csv` (four
//!   units over 400 simulations with many tied totals; TVaR at 90% and
//!   75%, proportional hazard and dual power; the portfolio's and each
//!   unit's stand-alone measure, Euler, covariance, proportional, marginal
//!   and Shapley, written from definitions in numpy);
//! * *Pricing Insurance Risk*, Exercises 205 and 206: a premium of 80
//!   allocated by stand-alone TVaR, as a haircut at 75% and at the equal
//!   risk level;
//! * the header row, the totals and the errors.

mod common;

use common::{n, num, t, text, validation};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn none() -> Grid {
    Grid::missing()
}

/// The reference's four units over 400 simulations
/// (`validation/scripts/numpy_allocation.py`'s `draws`).
fn reference_draws() -> Grid {
    Grid::from_rows(
        (0..400usize)
            .map(|i| {
                let a = (i * 37) % 101;
                [
                    a,
                    (i * 53) % 97 + a / 2,
                    (i * i) % 89,
                    3 * ((i * 29) % 41) + (a / 10).pow(2),
                ]
                .map(|x| Value::Number(x as f64))
                .to_vec()
            })
            .collect(),
    )
}

#[test]
fn allocations_match_numpy() {
    let x = reference_draws();
    let file = validation("reference/allocation_numpy.csv");
    let mut checked = 0;
    for line in file.lines().skip(1) {
        let f: Vec<&str> = line.splitn(8, ',').collect();
        let family = match f[0] {
            "tvar" => "tvar",
            "proportional_hazard" => "ph",
            "dual_power" => "dual",
            other => panic!("unknown distortion {other}"),
        };
        let arg: f64 = f[1].parse().unwrap();
        let (expected, abs_tol, rel_tol): (f64, f64, f64) = (
            f[4].parse().unwrap(),
            f[5].parse().unwrap(),
            f[6].parse().unwrap(),
        );
        let method = match f[2] {
            "total" | "standalone" => "euler",
            m => m,
        };
        let out = call(
            "pro_allocate",
            vec![x.clone(), t(family), n(arg), t(method)],
        );
        assert_eq!(out.rows, 7, "{out:?}");
        let got = match f[2] {
            "total" => num(out.get(6, 1)),
            "standalone" => num(out.get(f[3].parse::<usize>().unwrap() + 1, 1)),
            _ => num(out.get(f[3].parse::<usize>().unwrap() + 1, 2)),
        };
        assert!(
            (got - expected).abs() <= abs_tol + rel_tol * expected.abs(),
            "{line}: got {got}"
        );
        checked += 1;
    }
    assert_eq!(checked, 100);
}

/// *Pricing Insurance Risk*'s Simple Discrete Example: two units over 16
/// equally likely simulations, with a header row.
fn simple_discrete() -> Grid {
    let mut rows = vec![vec![Value::from("X1"), Value::from("X2")]];
    for (a, k) in [(0.0, 2), (8.0, 1), (10.0, 1)] {
        for (b, j) in [(0.0, 2), (1.0, 1), (90.0, 1)] {
            for _ in 0..k * j {
                rows.push(vec![Value::Number(a), Value::Number(b)]);
            }
        }
    }
    Grid::from_rows(rows)
}

#[test]
fn haircut_and_equal_risk_match_pricing_insurance_risk() {
    let x = simple_discrete();
    let h = call(
        "pro_allocate_total",
        vec![x.clone(), n(80.0), t("haircut"), t("tvar"), n(0.75)],
    );
    assert_eq!(h.get(1, 0), &Value::from("X1"));
    assert_eq!((num(h.get(1, 1)), num(h.get(2, 1))), (10.0, 90.0));
    assert!((num(h.get(1, 2)) - 8.0).abs() < 1e-12);
    assert!((num(h.get(2, 2)) - 72.0).abs() < 1e-12);
    assert!((num(h.get(3, 2)) - 80.0).abs() < 1e-12);
    assert_eq!(num(h.get(4, 1)), 0.75);
    let e = call(
        "pro_allocate_total",
        vec![x.clone(), n(80.0), t("equal_risk")],
    );
    assert!((num(e.get(4, 1)) - 0.6796).abs() < 5e-5, "{e:?}");
    assert!((num(e.get(1, 2)) - 9.56).abs() < 5e-3);
    assert!((num(e.get(2, 2)) - 70.44).abs() < 5e-3);
    assert!((num(e.get(1, 3)) + num(e.get(2, 3)) - 1.0).abs() < 1e-12);
    // The haircut's level defaults to 0.99.
    let d = call("pro_allocate_total", vec![x, n(80.0)]);
    assert_eq!(num(d.get(4, 1)), 0.99);
}

#[test]
fn totals_and_names() {
    // Two lines over four simulations, the totals 3, 5, 7, 9: the crate's
    // example, TVaR 50% of 8 shared 2.5 and 5.5.
    let x = Grid::from_rows(
        [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]]
            .iter()
            .map(|r| r.iter().map(|&v| Value::Number(v)).collect())
            .collect(),
    );
    let out = call("pro_allocate", vec![x.clone(), t("tvar"), n(0.5)]);
    assert_eq!(out.get(1, 0), &Value::from("Unit 1"));
    assert_eq!((num(out.get(1, 1)), num(out.get(2, 1))), (3.5, 5.5));
    assert_eq!((num(out.get(1, 2)), num(out.get(2, 2))), (2.5, 5.5));
    assert_eq!(num(out.get(3, 2)), 8.0);
    assert_eq!(num(out.get(4, 2)), 8.0);
    assert_eq!(num(out.get(4, 4)), 1.0);
    assert_eq!(num(out.get(1, 4)), 1.0);
    // The marginal leaves part of the portfolio's measure unallocated.
    let m = call(
        "pro_allocate",
        vec![x.clone(), t("tvar"), n(0.5), t("marginal")],
    );
    assert!(num(m.get(3, 2)) <= num(m.get(4, 2)));

    let error = |args: Vec<Grid>, symbol: &str, needle: &str| {
        let out = call(symbol, args);
        let msg = text(&out);
        assert!(msg.contains(needle), "{msg:?} lacks {needle:?}");
    };
    error(
        vec![x.clone(), t("tvar"), n(0.5), t("average")],
        "pro_allocate",
        "method must be",
    );
    error(vec![x.clone(), t("nope"), n(0.5)], "pro_allocate", "nope");
    error(
        vec![
            Grid::from_rows(vec![
                vec![Value::Number(1.0), Value::Number(2.0)],
                vec![Value::Number(1.0), Value::from("x")],
            ]),
            t("tvar"),
            n(0.5),
        ],
        "pro_allocate",
        "must be a number",
    );
    error(
        vec![x.clone(), n(80.0), t("haircut"), t("epd")],
        "pro_allocate_total",
        "needs a level",
    );
    error(
        vec![x.clone(), n(10.0), t("equal_risk"), none(), n(0.5)],
        "pro_allocate_total",
        "leave level out",
    );
    error(
        vec![x.clone(), n(1000.0), t("equal_risk")],
        "pro_allocate_total",
        "outside the range",
    );
    error(
        vec![x, n(10.0), t("haircut"), t("cte")],
        "pro_allocate_total",
        "measure must be",
    );
}
