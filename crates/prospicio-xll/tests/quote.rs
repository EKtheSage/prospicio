//! `PRO.QUOTE`, reached through the catalogue as Excel reaches it: the
//! technical premium of `docs/guides/reinsurance-price.md`'s middle layer
//! (risk premium 1.24 on an expected loss of 0.586 and a limit of 5, 3%
//! expenses and 10% brokerage) at the guide's printed ratios, the build-up
//! adding up, and the cover at quoted premiums.

mod common;

use common::{n, num, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn get(out: &Grid, label: &str, column: usize) -> Value {
    let r = (0..out.rows)
        .find(|&r| out.get(r, 0) == &Value::from(label))
        .unwrap_or_else(|| panic!("no {label}"));
    out.get(r, column).clone()
}

fn at(out: &Grid, label: &str, column: usize) -> f64 {
    num(&get(out, label, column))
}

fn guide(quoted: Grid) -> Grid {
    call(
        "pro_quote",
        vec![
            n(0.586),
            n(1.24),
            n(7.778),
            n(0.03),
            n(0.10),
            Grid::missing(),
            n(5.0),
            quoted,
        ],
    )
}

#[test]
fn the_guide_s_technical_premium() {
    let out = guide(Grid::missing());
    assert_eq!((out.rows, out.columns), (14, 2));
    let p = at(&out, "Premium", 1);
    assert!((p - 1.24 / 0.87).abs() < 1e-12);
    // The guide prints 1.425, 28.5% (payback 3.5), 2.43 and 41.1%.
    assert_eq!(format!("{p:.3}"), "1.425");
    assert_eq!(format!("{:.3}", at(&out, "Rate on line", 1)), "0.285");
    assert_eq!(format!("{:.1}", at(&out, "Payback", 1)), "3.5");
    assert_eq!(format!("{:.2}", at(&out, "Multiple", 1)), "2.43");
    assert_eq!(format!("{:.3}", at(&out, "Loss ratio", 1)), "0.411");
    // The build-up adds up: premium = risk premium + expenses + brokerage.
    let parts = at(&out, "Risk premium", 1) + at(&out, "Expenses", 1) + at(&out, "Brokerage", 1);
    assert!((parts - p).abs() < 1e-12);
    assert!((at(&out, "Net of brokerage", 1) - 0.9 * p).abs() < 1e-12);
    assert!((at(&out, "Margin", 1) - (1.24 - 0.586)).abs() < 1e-12);
    assert!((at(&out, "Capital", 1) - (7.778 - 1.24)).abs() < 1e-12);
    let roc = (1.24 - 0.586) / (7.778 - 1.24);
    assert!((at(&out, "Return on capital", 1) - roc).abs() < 1e-12);
}

#[test]
fn quoted_premiums_give_their_returns() {
    let tech = 1.24 / 0.87;
    let out = guide(Grid::row(&[tech, 1.2 * tech, 1.0]));
    assert_eq!(out.columns, 5);
    assert_eq!(out.get(0, 2), &Value::from("Quoted"));
    // At the technical premium the quote is the technical one.
    for label in ["Premium", "Risk premium", "Return on capital"] {
        assert!(
            (at(&out, label, 2) - at(&out, label, 1)).abs() < 1e-12,
            "{label}"
        );
    }
    assert!(at(&out, "Return on capital", 3) > at(&out, "Return on capital", 1));
    // A quote of 1 leaves 0.87 to the reinsurer, a margin of 0.284.
    assert!((at(&out, "Risk premium", 4) - 0.87).abs() < 1e-12);
    assert!((at(&out, "Margin", 4) - (0.87 - 0.586)).abs() < 1e-12);
    assert_eq!(at(&out, "Expected loss", 4), 0.586);
}

#[test]
fn a_fixed_expense_and_no_assets_or_limit() {
    let out = call(
        "pro_quote",
        vec![
            n(100.0),
            n(120.0),
            Grid::missing(),
            n(0.05),
            n(0.15),
            n(4.0),
        ],
    );
    assert!((at(&out, "Premium", 1) - 124.0 / 0.8).abs() < 1e-12);
    assert!((at(&out, "Expenses", 1) - (0.05 * 155.0 + 4.0)).abs() < 1e-12);
    for label in ["Capital", "Return on capital", "Rate on line", "Payback"] {
        assert_eq!(get(&out, label, 1), Value::Empty, "{label}");
    }
}

#[test]
fn bad_inputs_are_named() {
    let e = call("pro_quote", vec![n(1.0), n(0.5)]);
    assert!(
        text(&e).contains("at least the expected loss"),
        "{}",
        text(&e)
    );
    let e = call("pro_quote", vec![n(1.0), n(2.0), n(1.5)]);
    assert!(text(&e).contains("assets must exceed"), "{}", text(&e));
    let e = call(
        "pro_quote",
        vec![n(1.0), n(2.0), Grid::missing(), n(0.5), n(0.5)],
    );
    assert!(text(&e).contains("sum below 1"), "{}", text(&e));
    let out = guide(n(20.0));
    assert!(text(&out).contains("no capital"), "{}", text(&out));
}
