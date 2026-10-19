//! The distortion functions, reached through the catalogue as Excel
//! reaches them, against `aggregate` 1.0.1
//! (`validation/reference/distortion_aggregate.csv`, written by
//! `validation/scripts/aggregate_distortions.py`) at its tolerances:
//! `g(s)` for every one-parameter family and the four capped ones, and
//! the price and calibration of Mildenhall and Major's InsCo totals.

mod common;

use common::{n, num, reference, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

/// Monograph 15's InsCo totals and their probabilities.
const X: [f64; 7] = [22.0, 28.0, 36.0, 40.0, 55.0, 65.0, 100.0];
const P: [f64; 7] = [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1];

fn close(got: f64, row: &[String]) -> bool {
    let (want, abs, rel): (f64, f64, f64) = (
        row[5].parse().unwrap(),
        row[6].parse().unwrap(),
        row[7].parse().unwrap(),
    );
    (got - want).abs() <= abs.max(rel * want.abs())
}

#[test]
fn g_and_insco_match_aggregate() {
    let mut checked = 0;
    for row in reference("distortion_aggregate.csv") {
        let (kind, params, family) = (row[0].as_str(), row[1].as_str(), row[2].as_str());
        let value = |k: &str| -> f64 {
            params
                .split(';')
                .find_map(|kv| kv.strip_prefix(&format!("{k}=")))
                .unwrap()
                .parse()
                .unwrap()
        };
        let arg: f64 = match row[3].parse() {
            Ok(a) => a,
            Err(_) => continue,
        };
        let got = match (kind, family) {
            ("g", "bitvar" | "beta" | "wtdtvar") => continue,
            ("g", _) => {
                let (parameter, r0) = match family {
                    "tvar" => (value("p"), None),
                    "wang" => (value("lambda"), None),
                    "ph" => (value("rho"), None),
                    "dual" => (value("beta"), None),
                    "ccoc" => (value("r"), None),
                    "clin" => (value("slope"), Some(value("r0"))),
                    "cll" => (value("b"), Some(value("r0"))),
                    "lep" | "ly" => (value("r"), Some(value("r0"))),
                    _ => continue,
                };
                let r0 = r0.map_or(Grid::missing(), n);
                num(&call("pro_distortion", vec![n(arg), t(family), n(parameter), r0]).cells[0])
            }
            ("insco_price", _) => {
                let out = call(
                    "pro_distortion_price",
                    vec![Grid::column(&X), Grid::column(&P), t(family), n(arg)],
                );
                num(out.get(1, 1))
            }
            ("insco_calibrate", _) => {
                let out = call(
                    "pro_distortion_calibrate",
                    vec![Grid::column(&X), Grid::column(&P), n(arg), t(family)],
                );
                assert_eq!(out.get(1, 0), &Value::from(family));
                num(out.get(1, 1))
            }
            _ => continue,
        };
        assert!(
            close(got, &row),
            "{kind} {params} {family} at {arg}: {got} against {}",
            row[5]
        );
        checked += 1;
    }
    assert!(checked >= 110, "{checked}");
}

#[test]
fn calibration_defaults_to_the_standard_five_and_prices_back() {
    let premium = (46.6 + 0.15 * 100.0) / 1.15;
    let out = call(
        "pro_distortion_calibrate",
        vec![Grid::column(&X), Grid::column(&P), n(premium)],
    );
    assert_eq!((out.rows, out.columns), (6, 3));
    let names: Vec<&Value> = (1..6).map(|r| out.get(r, 0)).collect();
    assert_eq!(
        names,
        ["ccoc", "ph", "wang", "dual", "tvar"]
            .map(Value::from)
            .iter()
            .collect::<Vec<_>>()
    );
    assert!((num(out.get(1, 1)) - 0.15).abs() < 1e-12);
    for r in 1..6 {
        assert!((num(out.get(r, 2)) - premium).abs() < 1e-8, "row {r}");
        // Each calibrated parameter prices the outcomes at the premium.
        let p = call(
            "pro_distortion_price",
            vec![
                Grid::column(&X),
                Grid::column(&P),
                Grid::scalar(out.get(r, 0).clone()),
                Grid::scalar(out.get(r, 1).clone()),
            ],
        );
        assert!((num(p.get(1, 1)) - premium).abs() < 1e-8);
    }
}

#[test]
fn draws_are_equally_likely_and_any_order() {
    // TVaR at 50% of four draws is the mean of the top two.
    let out = call(
        "pro_distortion_price",
        vec![
            Grid::column(&[6.0, 0.0, 2.0, 0.0]),
            Grid::missing(),
            t("tvar"),
            n(0.5),
        ],
    );
    assert_eq!(num(out.get(0, 1)), 2.0);
    assert_eq!(num(out.get(1, 1)), 4.0);
    assert_eq!(num(out.get(2, 1)), 2.0);
    assert_eq!(num(out.get(3, 1)), 0.5);
}

#[test]
fn bad_inputs_are_named() {
    let e = call("pro_distortion", vec![n(1.5), t("ph"), n(0.5)]);
    assert!(text(&e).contains("s must be a probability"), "{}", text(&e));
    let e = call("pro_distortion", vec![n(0.5), t("ph"), n(0.5), n(0.1)]);
    assert!(text(&e).contains("r0 is taken only by"), "{}", text(&e));
    let e = call("pro_distortion", vec![n(0.5), t("clin"), n(0.5)]);
    assert!(text(&e).contains("r0"), "{}", text(&e));
    let e = call("pro_distortion", vec![n(0.5), t("normal"), n(0.5)]);
    assert!(text(&e).contains("family must be"), "{}", text(&e));
    let e = call(
        "pro_distortion_price",
        vec![Grid::column(&X), Grid::column(&[0.5, 0.5]), t("ph"), n(0.5)],
    );
    assert!(text(&e).contains("7 values and 2 probs"), "{}", text(&e));
    // A premium above the largest outcome is out of reach for every family.
    let out = call(
        "pro_distortion_calibrate",
        vec![Grid::column(&X), Grid::column(&P), n(120.0)],
    );
    assert!(matches!(out.get(2, 1), Value::Text(s) if s.contains("strictly between")));
}
