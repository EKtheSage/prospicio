//! Selected patterns on the expected-loss functions, a per-origin a priori
//! loss ratio and pattern ages on the triangle functions (#635 and #680,
//! `docs/design/xll.md` decision 40), reached through the catalogue as
//! Excel reaches them:
//!
//! * `PRO.EL`, `PRO.BF`, `PRO.BENKTANDER` and `PRO.CAPECOD` with
//!   `selected`, `kind` and `weight` against every chainladder-python 0.10.1
//!   `DevelopmentConstant` case on the CAS workers' compensation triangle
//!   (`validation/reference/reserving_patterns_python.csv`), each origin's
//!   ultimate and the total, and the percents developed of the same
//!   patterns;
//! * `pattern_ages` on all thirteen functions that take `selected`: 12, 24,
//!   ... given is the default to the bit, a monthly cumulative pattern
//!   gives its annual values' fit, and a pattern off the triangle's ages is
//!   the same fit made directly with `prospicio_reserving`, to the bit;
//! * `apriori` as a column on a triangle and on a latest diagonal (whose
//!   origins may come in any order), against the crate and the BF formula;
//! * the errors.

mod common;

use std::collections::BTreeMap;

use common::{clrd_wkcomp_triangle, genins, n, num, raa, reference, t, text};
use prospicio_reserving::{
    Apriori, BornhuetterFerguson, ChainLadder, Development, Lag, Mack, PatternKind, SelectedPattern,
};
use prospicio_xll::call;
use prospicio_xll::reserving::{COLUMN, EXPOSURE, exposed_triangle, triangle};
use prospicio_xll::sheet::{Grid, Value};

fn none() -> Grid {
    Grid::missing()
}

fn row(values: &[f64]) -> Grid {
    Grid::from_rows(vec![values.iter().map(|&x| Value::Number(x)).collect()])
}

fn col(values: &[f64]) -> Grid {
    Grid::from_rows(values.iter().map(|&x| vec![Value::Number(x)]).collect())
}

/// The numbers in column `c` of an exhibit's origin rows (all but the
/// header and the total).
fn column(out: &Grid, c: usize) -> Vec<f64> {
    (1..out.rows - 1).map(|r| num(out.get(r, c))).collect()
}

/// One chainladder-python case: the selected pattern by age, each origin's
/// ultimate, the total and the relative tolerance.
#[derive(Default)]
struct Case {
    pattern: Vec<(Lag, f64)>,
    ultimate: Vec<f64>,
    total: f64,
    rel: f64,
}

fn cases() -> BTreeMap<String, Case> {
    let mut cases: BTreeMap<String, Case> = BTreeMap::new();
    for f in reference("reserving_patterns_python.csv") {
        if f[0] != "clrd_wkcomp" {
            continue;
        }
        let c = cases.entry(f[1].clone()).or_default();
        let x: f64 = f[4].parse().unwrap();
        match f[2].as_str() {
            "pattern" => c.pattern.push((f[3].parse().unwrap(), x)),
            "ultimate" => c.ultimate.push(x),
            "total_ultimate" => {
                c.total = x;
                c.rel = f[6].parse::<f64>().unwrap_or(1e-9).max(1e-9);
            }
            _ => {}
        }
    }
    cases
}

/// A method spec's numeric setting: `apriori` in `bornhuetter_ferguson;apriori=0.75;...`.
fn param(method: &str, name: &str) -> Grid {
    let x = method
        .split(';')
        .find_map(|p| p.strip_prefix(name)?.strip_prefix('='))
        .unwrap_or_else(|| panic!("{method}: no {name}"));
    n(x.parse().unwrap())
}

/// The selected values as a row under the triangle's ten columns.
fn selected_row(pattern: &[(Lag, f64)], percent: bool) -> Grid {
    let mut cells = vec![Value::Empty; 10];
    for &(age, x) in pattern {
        cells[age as usize / 12 - 1] = Value::Number(if percent { 1.0 / x } else { x });
    }
    Grid::from_rows(vec![cells])
}

/// An expected-loss function on the workers' compensation triangle with a
/// chainladder-python case's settings.
fn run_case(method: &str, selected: Grid, kind: &str) -> Grid {
    let (tri, premium) = clrd_wkcomp_triangle();
    let has = |p: &str| method.split(';').any(|x| x == p);
    let tail = if has("tail=1.05") { n(1.05) } else { none() };
    let weight = if has("weight=0.6") { n(0.6) } else { none() };
    let pattern = [selected, t(kind), weight];
    let (symbol, mut args) = match method.split(';').next().unwrap() {
        "expected_loss" => ("pro_el", vec![tri, premium, param(method, "apriori")]),
        "bornhuetter_ferguson" => ("pro_bf", vec![tri, premium, param(method, "apriori")]),
        "benktander" => (
            "pro_benktander",
            vec![
                tri,
                premium,
                param(method, "apriori"),
                param(method, "n_iters"),
            ],
        ),
        "cape_cod" => (
            "pro_capecod",
            vec![tri, premium, param(method, "trend"), param(method, "decay")],
        ),
        other => panic!("unknown method {other}"),
    };
    args.extend([none(), tail, none()]);
    args.extend(pattern);
    call(symbol, args)
}

#[test]
fn expected_loss_functions_match_chainladder_python_with_a_selected_pattern() {
    let mut checked = 0;
    for (method, case) in cases() {
        let ldf = method.split(';').any(|p| p == "style=ldf");
        let mut kinds = vec![(if ldf { "ldf" } else { "cdf" }, false)];
        if !ldf {
            kinds.push(("percent", true));
        }
        for (kind, percent) in kinds {
            let out = run_case(&method, selected_row(&case.pattern, percent), kind);
            assert_eq!((out.rows, out.columns), (12, 7), "{method} {kind}: {out:?}");
            let close = |got: f64, expected: f64, what: &str| {
                assert!(
                    (got - expected).abs() <= case.rel * expected.abs(),
                    "{method} {kind} {what}: {got} vs {expected}"
                );
            };
            for (i, (&got, &expected)) in column(&out, 5).iter().zip(&case.ultimate).enumerate() {
                close(got, expected, &format!("ultimate {}", i + 1));
            }
            close(num(out.get(11, 5)), case.total, "total");
            checked += 1;
        }
    }
    // Five cases for each of four methods, three of them on factors to
    // ultimate and so also run as percents developed.
    assert_eq!(checked, 32);
}

/// The thirteen functions that take `selected`, each with `selected` as
/// given, `kind` and `pattern_ages` (`weight` omitted), on RAA or, for the
/// expected-loss methods, GenIns with its premium.
fn with_pattern(symbol: &str, selected: Grid, kind: &str, ages: Grid) -> Grid {
    let tail = [selected, t(kind), none()];
    let args = match symbol {
        "pro_cl" => [
            vec![raa(), none(), none(), none()],
            tail.to_vec(),
            vec![ages],
        ]
        .concat(),
        "pro_cl_ldf" | "pro_cl_cdf" | "pro_cl_ultimate" | "pro_cl_reserve" => {
            [vec![raa(), none(), none()], tail.to_vec(), vec![ages]].concat()
        }
        "pro_cl_factors" => [
            vec![raa(), none(), none()],
            tail.to_vec(),
            vec![none(), ages],
        ]
        .concat(),
        "pro_mack" | "pro_mack_cdr" | "pro_mack_cdr_years" => [
            vec![raa(), none(), none(), none()],
            tail.to_vec(),
            vec![none(), ages],
        ]
        .concat(),
        "pro_el" | "pro_bf" => {
            let (tri, premium) = genins();
            [
                vec![tri, premium, n(0.6), none(), none(), none()],
                tail.to_vec(),
                vec![ages],
            ]
            .concat()
        }
        "pro_benktander" | "pro_capecod" => {
            let (tri, premium) = genins();
            let third = if symbol == "pro_benktander" {
                n(0.6)
            } else {
                none()
            };
            [
                vec![tri, premium, third, none(), none(), none(), none()],
                tail.to_vec(),
                vec![ages],
            ]
            .concat()
        }
        other => panic!("{other}"),
    };
    call(symbol, args)
}

const WITH_SELECTED: [&str; 13] = [
    "pro_cl",
    "pro_cl_ldf",
    "pro_cl_cdf",
    "pro_cl_ultimate",
    "pro_cl_reserve",
    "pro_cl_factors",
    "pro_mack",
    "pro_mack_cdr",
    "pro_mack_cdr_years",
    "pro_el",
    "pro_bf",
    "pro_benktander",
    "pro_capecod",
];

#[test]
fn pattern_ages_at_twelve_month_steps_are_the_default() {
    let selected = row(&[2.5, 1.6, 1.2, 1.1, 1.05]);
    let ages = row(&[12.0, 24.0, 36.0, 48.0, 60.0]);
    for symbol in WITH_SELECTED {
        let default = with_pattern(symbol, selected.clone(), "ldf", none());
        assert!(default.cells.len() > 1, "{symbol}: {default:?}");
        let aged = with_pattern(symbol, selected.clone(), "ldf", ages.clone());
        assert_eq!(aged, default, "{symbol}");
        // And the ages change the fit when they are not those.
        let shifted = with_pattern(
            symbol,
            selected.clone(),
            "ldf",
            row(&[12.0, 18.0, 36.0, 48.0, 60.0]),
        );
        assert_ne!(shifted, default, "{symbol}");
    }
}

/// RAA's chain-ladder factors to ultimate at 12, 24, ..., 120, and a
/// monthly pattern through them: `ln CDF` linear between them, and from 1
/// month up to 12 at the slope of the first year.
fn monthly_raa() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let annual = call("pro_cl_cdf", vec![raa()]);
    let annual: Vec<f64> = annual.cells.iter().map(num).collect();
    assert_eq!(annual.len(), 10);
    let ln = |k: usize| annual[k].ln();
    let mut ages = Vec::new();
    let mut values = Vec::new();
    for m in 1..=120usize {
        let x = if m % 12 == 0 {
            annual[m / 12 - 1]
        } else if m < 12 {
            (ln(0) + (ln(0) - ln(1)) * (12 - m) as f64 / 12.0).exp()
        } else {
            let k = m / 12 - 1;
            let s = (m % 12) as f64 / 12.0;
            (ln(k) * (1.0 - s) + ln(k + 1) * s).exp()
        };
        ages.push(m as f64);
        values.push(x);
    }
    (ages, values, annual)
}

#[test]
fn a_monthly_cumulative_pattern_gives_its_annual_values_fit() {
    let (ages, monthly, annual) = monthly_raa();
    for symbol in WITH_SELECTED {
        let by_year = with_pattern(symbol, row(&annual), "cdf", none());
        let by_month = with_pattern(symbol, row(&monthly), "cdf", row(&ages));
        assert_eq!(
            (by_month.rows, by_month.columns),
            (by_year.rows, by_year.columns),
            "{symbol}: {by_month:?}"
        );
        for (a, b) in by_month.cells.iter().zip(&by_year.cells) {
            match (a, b) {
                (Value::Number(x), Value::Number(y)) => assert!(
                    (x - y).abs() <= 1e-12 * y.abs().max(1.0),
                    "{symbol}: {x} vs {y}"
                ),
                _ => assert_eq!(a, b, "{symbol}"),
            }
        }
    }
}

/// A pattern at 6, 18, ..., 114 months, between the triangle's ages: read
/// on its own ages (`reserving-v02.md` decision 18 item 4).
fn off_grid() -> (Vec<Lag>, Vec<f64>) {
    let ages: Vec<Lag> = (0..10).map(|k| 6 + 12 * k).collect();
    let values = vec![4.2, 1.95, 1.42, 1.21, 1.11, 1.06, 1.035, 1.02, 1.01, 1.004];
    (ages, values)
}

#[test]
fn a_pattern_off_the_triangles_ages_is_the_crates_fit() {
    let (ages, values) = off_grid();
    let ages_row = row(&ages.iter().map(|&a| f64::from(a)).collect::<Vec<_>>());
    let pattern = SelectedPattern::new(
        PatternKind::AgeToUltimate,
        ages.iter().copied().zip(values.iter().copied()).collect(),
    );
    let development = Development {
        pattern: Some(pattern),
        ..Default::default()
    };

    // The chain ladder and Mack on RAA.
    let tri = triangle(&raa()).unwrap();
    let cl = ChainLadder {
        development: development.clone(),
        ..Default::default()
    }
    .fit(&tri, COLUMN)
    .unwrap();
    let out = with_pattern("pro_cl_ultimate", row(&values), "cdf", ages_row.clone());
    assert_eq!(out.cells.iter().map(num).collect::<Vec<_>>(), cl.ultimate);
    let mack = Mack {
        development: development.clone(),
        ..Mack::default()
    }
    .fit(&tri, COLUMN)
    .unwrap();
    let out = with_pattern("pro_mack", row(&values), "cdf", ages_row.clone());
    assert_eq!(column(&out, 6), mack.standard_error);
    assert_eq!(num(out.get(11, 6)), mack.total_standard_error);

    // Bornhuetter-Ferguson on GenIns.
    let (g, premium) = genins();
    let fit = BornhuetterFerguson {
        apriori: 0.6.into(),
        chain_ladder: ChainLadder {
            development,
            ..Default::default()
        },
    }
    .fit(&exposed_triangle(&g, &premium).unwrap(), COLUMN, EXPOSURE)
    .unwrap();
    let out = with_pattern("pro_bf", row(&values), "cdf", ages_row);
    assert_eq!(column(&out, 5), fit.ultimate);
    let cl = &fit.chain_ladder;
    let cdf: Vec<f64> = cl.latest_position.iter().map(|&d| cl.cdf[d]).collect();
    assert_eq!(column(&out, 3), cdf);
}

#[test]
fn apriori_per_origin_on_a_triangle() {
    let (tri, premium) = genins();
    let ratios: Vec<f64> = (0..10).map(|i| 0.5 + 0.03 * f64::from(i)).collect();
    for (symbol, iterations) in [
        ("pro_el", None),
        ("pro_bf", None),
        ("pro_benktander", Some(2.0)),
    ] {
        let args = |apriori: Grid| {
            let mut a = vec![tri.clone(), premium.clone(), apriori];
            if let Some(k) = iterations {
                a.push(n(k));
            }
            call(symbol, a)
        };
        let out = args(col(&ratios));
        assert_eq!(column(&out, 4), ratios, "{symbol}: {out:?}");
        // A row reads the same as a column, and one ratio repeated is the
        // scalar.
        assert_eq!(args(row(&ratios)), out, "{symbol}");
        assert_eq!(args(col(&[0.6; 10])), args(n(0.6)), "{symbol}");
    }

    // Bornhuetter-Ferguson: the latest plus each origin's own a priori
    // ultimate's unreported share, and the crate's fit to the bit.
    let out = call("pro_bf", vec![tri.clone(), premium.clone(), col(&ratios)]);
    let (latest, cdf, exposure) = (column(&out, 2), column(&out, 3), column(&out, 1));
    for (i, u) in column(&out, 5).into_iter().enumerate() {
        let by_hand = latest[i] + ratios[i] * exposure[i] * (1.0 - 1.0 / cdf[i]);
        assert!((u - by_hand).abs() <= 1e-9 * by_hand, "origin {}", i + 1);
    }
    let fit = BornhuetterFerguson {
        apriori: Apriori::InOrder(ratios.clone()),
        chain_ladder: ChainLadder::default(),
    }
    .fit(&exposed_triangle(&tri, &premium).unwrap(), COLUMN, EXPOSURE)
    .unwrap();
    assert_eq!(column(&out, 5), fit.ultimate);
}

#[test]
fn apriori_per_origin_on_a_latest_diagonal_follows_the_origins_order() {
    let origins = [2020.0, 2021.0, 2022.0, 2023.0];
    let ages = [48.0, 36.0, 24.0, 12.0];
    let latest = [900.0, 700.0, 450.0, 200.0];
    let exposure = [1000.0, 1100.0, 1200.0, 1300.0];
    let ratios = [0.8, 0.7, 0.75, 0.65];
    let pattern = [2.0, 1.3, 1.1, 1.0];
    let bf = |order: &[usize]| {
        let pick = |v: &[f64]| order.iter().map(|&i| v[i]).collect::<Vec<_>>();
        call(
            "pro_diag_bf",
            vec![
                col(&pick(&origins)),
                col(&pick(&ages)),
                col(&pick(&latest)),
                col(&pick(&exposure)),
                col(&pick(&ratios)),
                row(&pattern),
            ],
        )
    };
    let sorted = bf(&[0, 1, 2, 3]);
    assert_eq!(column(&sorted, 5), ratios, "{sorted:?}");
    // Given newest first, each ratio stays with its own origin; the exhibit
    // is oldest first either way.
    assert_eq!(bf(&[3, 2, 1, 0]), sorted);
    assert_eq!(bf(&[2, 0, 3, 1]), sorted);
}

#[test]
fn errors() {
    let (tri, premium) = genins();
    let short = col(&[0.6, 0.7, 0.8]);
    assert!(
        text(&call("pro_bf", vec![tri.clone(), premium.clone(), short]))
            .contains("a row or column of 10, one per origin")
    );
    let mut cells = vec![Value::Number(0.6); 10];
    cells[4] = Value::from("x");
    let bad = Grid::from_rows(cells.into_iter().map(|v| vec![v]).collect());
    assert!(
        text(&call("pro_bf", vec![tri.clone(), premium.clone(), bad]))
            .contains("apriori 5 must be a number")
    );
    let mut zero = vec![0.6; 10];
    zero[2] = 0.0;
    assert!(
        text(&call(
            "pro_el",
            vec![tri.clone(), premium.clone(), col(&zero)]
        ))
        .contains("positive")
    );
    let selected = row(&[2.5, 1.6, 1.2]);
    assert!(
        text(&with_pattern("pro_cl", none(), "", row(&[6.0, 18.0])))
            .contains("need selected factors")
    );
    assert!(
        text(&with_pattern(
            "pro_bf",
            selected.clone(),
            "ldf",
            row(&[6.0, 18.0])
        ))
        .contains("pattern_ages must have 3 cells, one per selected value, got 2")
    );
    assert!(
        text(&with_pattern(
            "pro_mack",
            selected.clone(),
            "ldf",
            row(&[6.0, 0.0, 30.0])
        ))
        .contains("pattern_ages 2 must be a whole number of months from 1")
    );
    assert!(
        text(&with_pattern(
            "pro_capecod",
            selected.clone(),
            "ldf",
            Grid::from_rows(vec![vec![Value::Number(6.0); 3]; 2])
        ))
        .contains("pattern_ages must be a row or a column")
    );
    // A blank age with a value is an error; a blank value skips its age.
    let mut ages = vec![Value::Number(6.0), Value::Empty, Value::Number(30.0)];
    assert!(
        text(&with_pattern(
            "pro_cl",
            selected.clone(),
            "ldf",
            Grid::from_rows(vec![ages.clone()])
        ))
        .contains("pattern_ages 2 must be a whole number")
    );
    ages[1] = Value::Number(18.0);
    let gap = Grid::from_rows(vec![vec![
        Value::Number(2.5),
        Value::Empty,
        Value::Number(1.2),
    ]]);
    let skipped = with_pattern("pro_cl", gap, "ldf", Grid::from_rows(vec![ages]));
    let direct = with_pattern("pro_cl", row(&[2.5, 1.2]), "ldf", row(&[6.0, 30.0]));
    assert_eq!(skipped, direct);
}
