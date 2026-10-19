//! The time-value functions, reached through the catalogue as Excel
//! reaches them: `PRO.PAYOUT.PATTERN` and `PRO.PAYOUT.STATS` on closed
//! forms (an annuity, an even run-off) and on RAA's chain ladder through
//! `PRO.CL.CDF`; `PRO.PAYOUT.PRICE` reducing to the one-period price and
//! solving its premium equation term by term.

mod common;

use common::{n, num, raa, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn row(x: &[f64]) -> Grid {
    Grid::row(x)
}

fn stat(out: &Grid, label: &str) -> f64 {
    let r = (0..out.rows)
        .find(|&r| out.get(r, 0) == &Value::from(label))
        .unwrap_or_else(|| panic!("no {label} in {out:?}"));
    num(out.get(r, 1))
}

#[test]
fn an_even_annuity_has_its_closed_forms() {
    let (times, shares) = (row(&[1.0, 2.0, 3.0, 4.0]), row(&[0.25; 4]));
    let out = call(
        "pro_payout_stats",
        vec![times.clone(), shares.clone(), n(0.05)],
    );
    let v: f64 = 1.0 / 1.05;
    let annuity = (1.0 - v.powi(4)) / 0.05;
    assert!((stat(&out, "Discount factor") - annuity / 4.0).abs() < 1e-15);
    assert_eq!(stat(&out, "Duration"), 2.5);
    let years = 1.0 + 0.75 * v + 0.5 * v * v + 0.25 * v.powi(3);
    assert!((stat(&out, "Capital years") - years).abs() < 1e-15);
    // The same pattern as the share paid to date.
    let paid = call(
        "pro_payout_stats",
        vec![
            times.clone(),
            row(&[0.25, 0.5, 0.75, 1.0]),
            n(0.05),
            t("paid"),
        ],
    );
    assert!((stat(&paid, "Discount factor") - annuity / 4.0).abs() < 1e-15);
    let table = call(
        "pro_payout_pattern",
        vec![times, shares, Grid::missing(), n(0.5), n(0.05)],
    );
    assert_eq!((table.rows, table.columns), (5, 5));
    assert_eq!(num(table.get(1, 0)), 1.5, "shifted half a year");
    assert_eq!(num(table.get(2, 2)), 0.5);
    assert_eq!(num(table.get(2, 3)), 0.5);
    assert!((num(table.get(4, 4)) - 1.05f64.powf(-4.5)).abs() < 1e-15);
}

#[test]
fn a_chain_ladder_cdf_is_a_pattern() {
    let cdf = call("pro_cl_cdf", vec![raa()]);
    assert_eq!(cdf.columns, 10);
    let ages: Vec<f64> = (1..=10).map(f64::from).collect();
    let table = call(
        "pro_payout_pattern",
        vec![row(&ages), cdf.clone(), t("factors")],
    );
    // Paid to date at each age is one over its factor to ultimate, so an
    // origin at that age has paid its latest over its ultimate.
    let cl = call("pro_cl", vec![raa()]);
    for origin in 0..10 {
        let age = 10 - origin;
        let paid = num(table.get(age, 2));
        let (latest, ultimate) = (num(cl.get(origin + 1, 1)), num(cl.get(origin + 1, 3)));
        assert!((paid - latest / ultimate).abs() < 1e-12, "origin {origin}");
    }
    assert_eq!(num(table.get(10, 3)), 0.0);
}

#[test]
fn the_payout_price_solves_its_equation() {
    let (e, a, coc) = (10.0, 30.0, 0.1);
    let one_year = (e + coc * a) / (1.0 + coc);
    // Paid within the year, undiscounted: the one-period price.
    let now = call(
        "pro_payout_price",
        vec![n(e), n(a), n(coc), n(0.5), n(1.0), n(0.0)],
    );
    assert!((stat(&now, "Premium") - one_year).abs() < 1e-12);
    assert_eq!(stat(&now, "One-year premium"), one_year);
    // Over three years at 4%: P = E d + coc (a - P) A, term by term.
    let (times, shares) = (row(&[1.0, 2.0, 3.0]), row(&[0.2, 0.3, 0.5]));
    let out = call(
        "pro_payout_price",
        vec![n(e), n(a), n(coc), times.clone(), shares.clone(), n(0.04)],
    );
    let s = call("pro_payout_stats", vec![times, shares, n(0.04)]);
    let (d, big_a) = (stat(&s, "Discount factor"), stat(&s, "Capital years"));
    assert_eq!(stat(&out, "Discount factor"), d);
    assert_eq!(stat(&out, "Capital years"), big_a);
    let p = stat(&out, "Premium");
    assert!((p - (e * d + coc * (a - p) * big_a)).abs() < 1e-12);
    assert!((stat(&out, "Margin") - (p - e * d)).abs() < 1e-12);
    assert_eq!(stat(&out, "Capital"), a - p);
    assert!(p > one_year * d, "capital held for longer costs more");
}

#[test]
fn bad_patterns_are_named() {
    let e = call(
        "pro_payout_stats",
        vec![row(&[1.0, 2.0]), row(&[0.5, 0.4]), n(0.05)],
    );
    assert!(text(&e).contains("must sum to 1"), "{}", text(&e));
    let e = call(
        "pro_payout_stats",
        vec![row(&[1.0, 2.0]), row(&[2.0, 1.1]), n(0.05), t("factors")],
    );
    assert!(
        text(&e).contains("the last factor must be 1"),
        "{}",
        text(&e)
    );
    let e = call(
        "pro_payout_stats",
        vec![row(&[1.0]), row(&[1.0, 0.0]), n(0.05)],
    );
    assert!(text(&e).contains("1 times and 2 values"), "{}", text(&e));
    let e = call("pro_payout_stats", vec![row(&[1.0]), row(&[1.0]), n(-2.0)]);
    assert!(text(&e).contains("rate must be"), "{}", text(&e));
    let e = call(
        "pro_payout_stats",
        vec![row(&[1.0]), row(&[1.0]), n(0.0), t("ratios")],
    );
    assert!(text(&e).contains("kind must be"), "{}", text(&e));
}
