//! The contract terms, reached through the catalogue as Excel reaches
//! them, against `aggregate` 1.0.1 at its tolerances:
//!
//! * `PRO.SWING`, `PRO.SLIDING.SCALE` and `PRO.PROFIT.COMMISSION` at
//!   losses through every kink (`contract_terms_aggregate.csv`, written by
//!   `validation/scripts/aggregate_contract_terms.py`);
//! * `PRO.LAYER.TERMS`'s expected reinstatement premium on books RI2 and
//!   RI10 (`reinsurance_aggregate.csv`), and its other rows on their
//!   definitions.

mod common;

use std::collections::HashMap;

use common::{n, num, reference, t, text};
use prospicio_xll::call;
use prospicio_xll::sheet::Grid;

fn params(spec: &str) -> HashMap<&str, f64> {
    spec.split(';')
        .filter_map(|kv| kv.split_once('='))
        .filter_map(|(k, v)| v.parse().ok().map(|v| (k, v)))
        .collect()
}

fn opt(p: &HashMap<&str, f64>, k: &str) -> Grid {
    p.get(k).map_or(Grid::missing(), |&v| n(v))
}

#[test]
fn terms_at_a_loss_match_aggregate() {
    let mut checked = 0;
    for row in reference("contract_terms_aggregate.csv") {
        let Ok(x) = row[3].parse::<f64>() else {
            continue;
        };
        let p = params(&row[1]);
        let out = match row[2].as_str() {
            "swing_premium" | "retro_premium" => call(
                "pro_swing",
                vec![
                    n(x),
                    n(p["basic"]),
                    n(p["lcm"]),
                    opt(&p, "minimum"),
                    opt(&p, "maximum"),
                    opt(&p, "share"),
                ],
            ),
            "sliding_commission" => {
                let k = p["n"] as usize;
                let c: Vec<f64> = (0..k).map(|i| p[format!("c{i}").as_str()]).collect();
                let lr: Vec<f64> = (0..k).map(|i| p[format!("lr{i}").as_str()]).collect();
                call(
                    "pro_sliding_scale",
                    vec![n(x), n(p["premium"]), Grid::row(&c), Grid::row(&lr)],
                )
            }
            "profit_commission" => call(
                "pro_profit_commission",
                vec![n(x), n(p["premium"]), n(p["share"]), n(p["allowance"])],
            ),
            _ => continue,
        };
        let got = num(&out.cells[0]);
        let (want, abs, rel): (f64, f64, f64) = (
            row[4].parse().unwrap(),
            row[5].parse().unwrap(),
            row[6].parse().unwrap(),
        );
        assert!(
            (got - want).abs() <= abs.max(rel * want.abs()),
            "{} {} at {x}: {got} against {want}",
            row[0],
            row[2]
        );
        checked += 1;
    }
    // 49 rows for each sliding scale and profit commission, 41 for each
    // retro plan and 33 for each swing.
    assert_eq!(checked, 475);
}

/// `PRO.LAYER.TERMS` on a book: Poisson `lambda`, a lognormal of mean 100
/// and CV 2 capped at 1,000, the layer 500 xs 100 at a 20% rate on line,
/// on a step-1 grid of 2^14 points.
fn book(lambda: f64, rest: Vec<Grid>) -> Grid {
    let mut args = vec![
        n(lambda),
        Grid::missing(),
        t("lognormal_cv"),
        n(100.0),
        n(2.0),
        Grid::missing(),
        n(500.0),
        n(100.0),
        n(0.2 * 500.0),
    ];
    args.extend(rest);
    args.resize(15, Grid::missing());
    args.extend([n(1000.0), n(1.0), n(16384.0)]);
    call("pro_layer_terms", args)
}

#[test]
fn reinstatement_premiums_match_aggregate() {
    let mut checked = 0;
    for row in reference("reinsurance_aggregate.csv") {
        if row[2] != "reinstatement_premium" {
            continue;
        }
        let p = params(&row[1]);
        let rates: Vec<f64> = row[1]
            .split(';')
            .find_map(|kv| kv.strip_prefix("rates="))
            .unwrap()
            .split('+')
            .map(|r| r.parse().unwrap())
            .collect();
        let out = book(p["lambda"], vec![Grid::row(&rates)]);
        let got = num(out.get(2, 1));
        let want: f64 = row[4].parse().unwrap();
        assert!(
            (got / want - 1.0).abs() < 1e-10,
            "{}: {got} against {want}",
            row[0]
        );
        checked += 1;
    }
    assert_eq!(checked, 4);
}

#[test]
fn the_rows_follow_their_definitions() {
    // A flat 25% commission and a profit commission of 20% after 30%.
    let out = book(
        2.0,
        vec![Grid::missing(), n(0.25), Grid::missing(), n(0.2), n(0.3)],
    );
    let v = |r: usize| num(out.get(r, 1));
    let (loss, premium, rip, cc, pc) = (v(0), v(1), v(2), v(3), v(4));
    assert!((premium - 100.0).abs() < 1e-9);
    assert_eq!(rip, 0.0);
    assert!((cc - 25.0).abs() < 1e-12);
    // The profit commission is at least its value at the expected loss
    // (convex in the loss) and at most its value at no loss.
    assert!(pc >= 0.2 * (100.0 * 0.7 - loss).max(0.0) - 1e-9 && pc <= 0.2 * 70.0);
    assert!((v(5) - (premium - cc - pc)).abs() < 1e-12);
    assert!((v(6) - (v(5) - loss)).abs() < 1e-12);
    assert!((v(7) - loss / premium).abs() < 1e-12);
}

#[test]
fn bad_terms_are_named() {
    let e = call(
        "pro_sliding_scale",
        vec![n(50.0), n(100.0), Grid::row(&[0.3, 0.2]), n(0.6)],
    );
    assert!(
        text(&e).contains("2 commissions and 1 loss_ratios"),
        "{}",
        text(&e)
    );
    let e = book(2.0, vec![Grid::missing(), Grid::missing(), n(0.6)]);
    assert!(
        text(&e).contains("loss_ratios go with commissions"),
        "{}",
        text(&e)
    );
    let e = book(
        2.0,
        vec![
            Grid::row(&[1.0]),
            Grid::missing(),
            Grid::missing(),
            n(0.2),
            n(0.3),
        ],
    );
    assert!(text(&e).contains("cannot be combined"), "{}", text(&e));
    let e = book(
        2.0,
        vec![Grid::missing(), Grid::missing(), Grid::missing(), n(0.2)],
    );
    assert!(text(&e).contains("together"), "{}", text(&e));
}
