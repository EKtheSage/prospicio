//! Multi-year contract terms, reached through the catalogue as Excel
//! reaches them: the hand-worked deficits and credits of
//! `prospicio_aggregate`'s own tests (`aggregate.md`, "Contract terms over
//! several years"), and simulated terms against `Tower::apply_aggregate_term`.
//! `aggregate` has no multi-year term, so there is no outside reference.

mod common;

use common::{n, num, t, text};
use prospicio_aggregate::{Layer, Tower};
use prospicio_prob::{Empirical, KeyValue, PredictiveDistribution, Provenance};
use prospicio_xll::call;
use prospicio_xll::sheet::{Grid, Value};

fn rows(r: &[&[f64]]) -> Grid {
    Grid::from_rows(
        r.iter()
            .map(|row| row.iter().map(|&x| Value::Number(x)).collect())
            .collect(),
    )
}

fn row(g: &Grid, i: usize) -> Vec<f64> {
    (0..g.columns).map(|j| num(g.get(i, j))).collect()
}

fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-12, "{a:?} vs {b:?}");
    }
}

#[test]
fn profit_commission_deficits_by_hand() {
    // Premium 100, allowance 20%: the threshold is 80; share 50%. Year
    // results -40, +10, +50, -5, +30.
    let ceded = rows(&[&[120.0, 70.0, 30.0, 85.0, 50.0]]);
    let pc = |years: Grid| {
        call(
            "pro_profit_commission_term",
            vec![ceded.clone(), n(100.0), n(0.5), n(0.2), years],
        )
    };
    // To extinction: 30 of year 1's 40 is left after year 2; year 3's 50
    // covers it and pays 50% of 20.
    close(&row(&pc(Grid::missing()), 0), &[0.0, 0.0, 10.0, 0.0, 12.5]);
    // For one year: year 1's deficit reaches year 2 only.
    close(&row(&pc(n(1.0)), 0), &[0.0, 0.0, 25.0, 0.0, 12.5]);
}

#[test]
fn sliding_scale_deficits_and_credits_by_hand() {
    // 40% at a 50% loss ratio to 20% at 70%, premium 100.
    let ceded = rows(&[&[90.0, 40.0, 30.0, 60.0]]);
    let slide = |carry: &str, years: Grid| {
        call(
            "pro_sliding_scale_term",
            vec![
                ceded.clone(),
                n(100.0),
                Grid::row(&[0.4, 0.2]),
                Grid::row(&[0.5, 0.7]),
                t(carry),
                years,
            ],
        )
    };
    close(
        &row(&slide("deficit", Grid::missing()), 0),
        &[20.0, 30.0, 40.0, 30.0],
    );
    close(
        &row(&slide("credit", Grid::missing()), 0),
        &[20.0, 40.0, 40.0, 40.0],
    );
    close(&row(&slide("both", n(1.0)), 0), &[20.0, 30.0, 40.0, 40.0]);
    // Both is the default.
    let both = call(
        "pro_sliding_scale_term",
        vec![
            ceded.clone(),
            n(100.0),
            Grid::row(&[0.4, 0.2]),
            Grid::row(&[0.5, 0.7]),
        ],
    );
    assert_eq!(both, slide("both", Grid::missing()));
}

/// Each row is a path of its own; on 400 simulated three-year terms the
/// sheet's commissions are `Tower::apply_aggregate_term`'s year by year.
#[test]
fn simulated_terms_match_the_tower() {
    let (n_paths, n_years) = (400, 3);
    // A deterministic spread of ceded losses around the premium of 100.
    let loss = |j: usize, y: usize| 20.0 + ((j * 37 + y * 101) % 160) as f64;
    let losses = Grid::from_rows(
        (0..n_paths)
            .map(|j| (0..n_years).map(|y| Value::Number(loss(j, y))).collect())
            .collect(),
    );
    let years: Vec<PredictiveDistribution> = (0..n_years)
        .map(|y| {
            PredictiveDistribution::from_draws(
                vec!["lob".into()],
                vec![vec![KeyValue::Int(0)]],
                (0..n_paths).map(|j| loss(j, y)).collect(),
                Provenance::new("test"),
            )
            .unwrap()
        })
        .collect();
    let base = || {
        Layer::quota_share("QS", 1.0)
            .unwrap()
            .deposit_premium(100.0)
            .unwrap()
    };
    let check = |layer: Layer, kind: &str, sheet: Grid| {
        let term = Tower::new(vec![layer])
            .unwrap()
            .apply_aggregate_term(&years)
            .unwrap();
        for y in 0..n_years {
            let key = vec![
                KeyValue::from(kind),
                KeyValue::from("QS"),
                KeyValue::Int(y as i64 + 1),
            ];
            let want = Empirical::draws(&term.marginal(&key).unwrap()).to_vec();
            let got: Vec<f64> = (0..n_paths).map(|j| num(sheet.get(j, y))).collect();
            close(&got, &want);
        }
    };
    let pc = base()
        .profit_commission(0.3, 0.15)
        .unwrap()
        .profit_commission_carry_forward(Some(2))
        .unwrap();
    let sheet = call(
        "pro_profit_commission_term",
        vec![losses.clone(), n(100.0), n(0.3), n(0.15), n(2.0)],
    );
    assert_eq!((sheet.rows, sheet.columns), (n_paths, n_years));
    check(pc, "profit_commission", sheet);

    let slide = base()
        .sliding_scale(vec![(0.35, 0.55), (0.25, 0.65), (0.2, 0.8)])
        .unwrap()
        .sliding_scale_carry_forward(true, true, None)
        .unwrap();
    let sheet = call(
        "pro_sliding_scale_term",
        vec![
            losses,
            n(100.0),
            Grid::column(&[0.35, 0.25, 0.2]),
            Grid::column(&[0.55, 0.65, 0.8]),
        ],
    );
    check(slide, "ceding_commission", sheet);
}

#[test]
fn bad_inputs_are_named() {
    let blank = Grid::from_rows(vec![vec![Value::Number(50.0), Value::Empty]]);
    let e = call(
        "pro_profit_commission_term",
        vec![blank, n(100.0), n(0.5), n(0.2)],
    );
    assert!(text(&e).contains("row 1, column 2"), "{}", text(&e));
    let e = call(
        "pro_profit_commission_term",
        vec![rows(&[&[50.0]]), n(100.0), n(0.5), n(0.2), n(0.0)],
    );
    assert!(text(&e).contains("carry_years"), "{}", text(&e));
    let e = call(
        "pro_sliding_scale_term",
        vec![
            rows(&[&[50.0, 60.0]]),
            n(100.0),
            Grid::row(&[0.4, 0.2]),
            Grid::row(&[0.5, 0.7]),
            t("profit"),
        ],
    );
    assert!(text(&e).contains("carry must be"), "{}", text(&e));
}
