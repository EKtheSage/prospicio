//! The distortion functions, reached through the catalogue as Excel
//! reaches them, against `aggregate` 1.0.1
//! (`validation/reference/distortion_aggregate.csv`, written by
//! `validation/scripts/aggregate_distortions.py`) at its tolerances:
//! `g(s)` for every one-parameter family, the four capped ones and the
//! bi-TVaR, beta and weighted TVaR, and
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
    let (mut checked, mut several) = (0, 0);
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
            ("g", "bitvar" | "beta" | "wtdtvar") => {
                let list = |k: &str| -> Vec<f64> {
                    params
                        .split(';')
                        .find_map(|kv| kv.strip_prefix(&format!("{k}=")))
                        .unwrap()
                        .split('|')
                        .map(|x| x.parse().unwrap())
                        .collect()
                };
                let parameter = match family {
                    "bitvar" => Grid::row(&[value("p0"), value("p1"), value("w")]),
                    "beta" => Grid::column(&[value("a"), value("b")]),
                    _ => {
                        let (ps, wts) = (list("ps"), list("wts"));
                        Grid::from_rows(
                            ps.iter()
                                .zip(&wts)
                                .map(|(p, w)| vec![Value::Number(*p), Value::Number(*w)])
                                .collect(),
                        )
                    }
                };
                several += 1;
                num(&call("pro_distortion", vec![n(arg), t(family), parameter]).cells[0])
            }
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
    assert!(checked >= 158, "{checked}");
    assert_eq!(several, 48);
}

/// The capped families take `r0` beside the parameter or with it in one
/// range; a weighted TVaR's pairs may run along two rows; CCoC is the
/// bitvar `(0, 1, r / (1 + r))`, on a sheet and in a tower's price.
#[test]
fn several_parameters_in_one_range() {
    let side = call("pro_distortion", vec![n(0.3), t("lep"), n(0.2), n(0.05)]);
    let range = call(
        "pro_distortion",
        vec![n(0.3), t("lep"), Grid::row(&[0.05, 0.2])],
    );
    assert_eq!(side, range);
    let column = Grid::from_rows(vec![
        vec![Value::Number(0.1), Value::Number(0.2)],
        vec![Value::Number(0.5), Value::Number(0.5)],
        vec![Value::Number(0.9), Value::Number(0.3)],
    ]);
    let rows = Grid::from_rows(vec![
        vec![Value::Number(0.1), Value::Number(0.5), Value::Number(0.9)],
        vec![Value::Number(0.2), Value::Number(0.5), Value::Number(0.3)],
    ]);
    let a = call(
        "pro_distortion_price",
        vec![Grid::column(&X), Grid::column(&P), t("wtdtvar"), column],
    );
    let b = call(
        "pro_distortion_price",
        vec![Grid::column(&X), Grid::column(&P), t("wtdtvar"), rows],
    );
    assert_eq!(a, b);
    let ccoc = call(
        "pro_distortion_price",
        vec![Grid::column(&X), Grid::column(&P), t("ccoc"), n(0.15)],
    );
    let bitvar = call(
        "pro_distortion_price",
        vec![
            Grid::column(&X),
            Grid::column(&P),
            t("bitvar"),
            Grid::row(&[0.0, 1.0, 0.15 / 1.15]),
        ],
    );
    assert!((num(ccoc.get(1, 1)) - num(bitvar.get(1, 1))).abs() < 1e-12);
}

#[test]
fn several_parameters_are_checked() {
    let e = call(
        "pro_distortion",
        vec![n(0.5), t("bitvar"), Grid::row(&[0.2, 0.9])],
    );
    assert!(
        text(&e).contains("bitvar takes 3 parameters"),
        "{}",
        text(&e)
    );
    let e = call(
        "pro_distortion",
        vec![n(0.5), t("bitvar"), Grid::row(&[0.9, 0.2, 0.3])],
    );
    assert!(text(&e).contains("p0"), "{}", text(&e));
    let e = call(
        "pro_distortion",
        vec![n(0.5), t("beta"), Grid::row(&[0.6, 1.5]), n(0.1)],
    );
    assert!(text(&e).contains("r0 is taken only by"), "{}", text(&e));
    let e = call(
        "pro_distortion",
        vec![n(0.5), t("wtdtvar"), Grid::row(&[0.1, 0.5, 0.9])],
    );
    assert!(text(&e).contains("two columns"), "{}", text(&e));
    let out = call(
        "pro_distortion_calibrate",
        vec![Grid::column(&X), Grid::column(&P), n(50.0), t("beta")],
    );
    assert!(text(&out).contains("several parameters"), "{}", text(&out));
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
