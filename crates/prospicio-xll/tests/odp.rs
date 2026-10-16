//! `PRO.ODP` on RAA, reached through the catalogue as Excel reaches it:
//! the same draws as `prospicio_reserving::OdpBootstrap`, and the total's
//! mean, standard deviation and percentiles within R ChainLadder's
//! `BootChainLadder` targets (`validation/reference/reserving_bootstrap_r.csv`,
//! 100,000 R simulations; tolerances of four standard errors for 20,000).

use prospicio_prob::Distribution;
use prospicio_reserving::OdpBootstrap;
use prospicio_xll::call;
use prospicio_xll::reserving::triangle;
use prospicio_xll::sheet::{Grid, Value};

const RAA: [&[f64]; 10] = [
    &[
        5012.0, 8269.0, 10907.0, 11805.0, 13539.0, 16181.0, 18009.0, 18608.0, 18662.0, 18834.0,
    ],
    &[
        106.0, 4285.0, 5396.0, 10666.0, 13782.0, 15599.0, 15496.0, 16169.0, 16704.0,
    ],
    &[
        3410.0, 8992.0, 13873.0, 16141.0, 18735.0, 22214.0, 22863.0, 23466.0,
    ],
    &[5655.0, 11555.0, 15766.0, 21266.0, 23425.0, 26083.0, 27067.0],
    &[1092.0, 9565.0, 15836.0, 22169.0, 25955.0, 26180.0],
    &[1513.0, 6445.0, 11702.0, 12935.0, 15852.0],
    &[557.0, 4020.0, 10946.0, 12314.0],
    &[1351.0, 6947.0, 13112.0],
    &[3133.0, 5395.0],
    &[2063.0],
];

fn raa() -> Grid {
    Grid::from_rows(
        RAA.iter()
            .map(|row| {
                (0..10)
                    .map(|c| row.get(c).map_or(Value::Empty, |&x| Value::Number(x)))
                    .collect()
            })
            .collect(),
    )
}

fn n(x: f64) -> Grid {
    Grid::scalar(Value::Number(x))
}

fn t(s: &str) -> Grid {
    Grid::scalar(Value::from(s))
}

fn odp(sims: f64, seed: f64, percentiles: Grid, process: Grid) -> Grid {
    call(
        "pro_odp",
        vec![
            raa(),
            n(sims),
            n(seed),
            percentiles,
            Grid::missing(),
            process,
            Grid::missing(),
        ],
    )
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Number(x) => *x,
        other => panic!("expected a number, got {other:?}"),
    }
}

fn text(g: &Grid) -> &str {
    match g.as_scalar() {
        Some(Value::Text(s)) => s,
        other => panic!("expected text, got {other:?}"),
    }
}

/// R targets for RAA's total: mean, sd, then the 75%, 95% and 99.5%
/// percentiles, each with its tolerance.
const GAMMA: [(f64, f64); 5] = [
    (53_842.271_840_772_2, 589.413),
    (18_965.346_841_816_2, 483.93),
    (65_058.795_840_656_7, 851.069),
    (87_856.178_069_026_2, 1854.44),
    (115_879.018_890_02, 5430.59),
];
const PARAM: [(f64, f64); 5] = [
    (53_928.250_949_839_6, 506.674),
    (17_453.911_470_332_1, 513.428),
    (64_201.286_185_936_6, 862.028),
    (85_236.880_985_955_2, 1701.59),
    (111_056.205_713_69, 5252.09),
];

fn check_total(process: &str, targets: &[(f64, f64); 5]) {
    let percentiles = Grid::from_rows(vec![vec![
        Value::Number(0.75),
        Value::Number(0.95),
        Value::Number(0.995),
    ]]);
    let out = odp(20_000.0, 1.0, percentiles, t(process));
    assert_eq!((out.rows, out.columns), (12, 8));
    assert_eq!(out.get(0, 7), &Value::from("P99.5"));
    let total = 11;
    assert_eq!(out.get(total, 0), &Value::from("Total"));
    for (k, col) in [2, 3, 5, 6, 7].into_iter().enumerate() {
        let (expected, tol) = targets[k];
        let got = num(out.get(total, col));
        assert!(
            (got - expected).abs() < tol,
            "{process} column {col}: {got} vs R {expected} +- {tol}"
        );
    }
}

#[test]
fn total_matches_r_bootchainladder_with_gamma_process() {
    check_total("gamma", &GAMMA);
}

#[test]
fn total_matches_r_bootchainladder_with_parameter_error_only() {
    check_total("none", &PARAM);
}

#[test]
fn origins_are_the_crates_draws() {
    let fit = OdpBootstrap {
        n_sims: 2_000,
        seed: 7,
        ..OdpBootstrap::default()
    }
    .fit(&triangle(&raa()).unwrap(), "values")
    .unwrap();
    let out = odp(2_000.0, 7.0, Grid::missing(), Grid::missing());
    let n = fit.reserves.n_components();
    for j in 0..n {
        let col: Vec<f64> = fit
            .reserves
            .draw_matrix()
            .iter()
            .skip(j)
            .step_by(n)
            .copied()
            .collect();
        let mean = col.iter().sum::<f64>() / col.len() as f64;
        assert!((num(out.get(j + 1, 2)) - mean).abs() <= 1e-9 * mean.abs().max(1.0));
        assert_eq!(num(out.get(j + 1, 1)), fit.chain_ladder.reserves()[j]);
    }
    assert_eq!(num(out.get(11, 3)), fit.reserves.total().std_dev());
    // The default percentiles head the last four columns.
    let heads: Vec<_> = (5..9).map(|c| out.get(0, c).clone()).collect();
    assert_eq!(heads, ["P75", "P90", "P95", "P99.5"].map(Value::from));
}

#[test]
fn the_same_seed_gives_the_same_table() {
    let a = odp(1_000.0, 3.0, Grid::missing(), Grid::missing());
    assert_eq!(a, odp(1_000.0, 3.0, Grid::missing(), Grid::missing()));
    assert_ne!(a, odp(1_000.0, 4.0, Grid::missing(), Grid::missing()));
}

#[test]
fn errors_come_back_as_text() {
    assert!(
        text(&odp(0.0, 0.0, Grid::missing(), Grid::missing()))
            .contains("sims must be a whole number")
    );
    assert!(text(&odp(2e6, 0.0, Grid::missing(), Grid::missing())).contains("1000000"));
    assert!(text(&odp(100.0, -1.0, Grid::missing(), Grid::missing())).contains("seed"));
    assert!(
        text(&odp(100.0, 0.0, n(99.5), Grid::missing())).contains("probabilities between 0 and 1")
    );
    assert!(text(&odp(100.0, 0.0, Grid::missing(), t("normal"))).contains("process must be"));
}
