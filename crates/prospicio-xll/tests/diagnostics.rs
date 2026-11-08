//! Mack's assumption tests and the diagonal backtest as Excel reaches them
//! (`docs/design/xll.md` decision 57): every value of
//! `validation/reference/reserving_assumptions_r.csv` (R ChainLadder
//! 0.2.21's `cyEffTest` and `dfCorTest`) and `_python.csv`
//! (chainladder-python) on RAA, GenIns and ABC, read off
//! `PRO.RES.CY.TEST` and `PRO.RES.DEVCOR.TEST`; the backtest equal to the
//! library's `diagonal_backtest` on the same models and seed.

mod common;

use common::{data_triangle, fails, n, ok, t, validation};
use prospicio_glm::Glm;
use prospicio_models::Terms;
use prospicio_reserving::{GlmCandidate, METRICS, TriangleFrame, TriangleModel, diagonal_backtest};
use prospicio_xll::reserving::{COLUMN, triangle};
use prospicio_xll::sheet::{Grid, Value};

fn none() -> Grid {
    Grid::missing()
}

fn number(v: &Value) -> f64 {
    match v {
        Value::Number(x) => *x,
        Value::Bool(b) => f64::from(u8::from(*b)),
        other => panic!("expected a number, got {other:?}"),
    }
}

fn labelled(out: &Grid, label: &str) -> f64 {
    let r = (0..out.rows)
        .find(|&r| *out.get(r, 0) == Value::from(label))
        .unwrap_or_else(|| panic!("no row {label}"));
    number(out.get(r, 1))
}

/// The reference's rows: dataset, test, quantity, arg, expected, abs, rel.
fn rows(name: &str) -> Vec<Vec<String>> {
    validation(&format!("reference/{name}"))
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("dataset,"))
        .map(|l| l.split(',').map(str::to_owned).collect())
        .collect()
}

fn check(name: &str) -> usize {
    let mut checked = 0;
    for row in rows(name) {
        let (dataset, test, quantity, arg) = (&row[0], &row[1], &row[2], &row[3]);
        let want: f64 = row[4].parse().unwrap();
        let (abs, rel): (f64, f64) = (row[5].parse().unwrap(), row[6].parse().unwrap());
        let (tri, origins) = data_triangle(dataset, "value");
        let calendar = test.starts_with("calendar_year_effect");
        let ci = test
            .split_once("ci=")
            .map_or(none(), |(_, c)| n(c.parse().unwrap()));
        let what = format!("{dataset} {test} {quantity} {arg}");
        let got = if calendar {
            let summary = ok(
                "pro_res_cy_test",
                vec![tri.clone(), ci.clone(), none(), none()],
            );
            let diagonals = ok(
                "pro_res_cy_test",
                vec![tri, ci, t("diagonals"), origins.clone()],
            );
            match quantity.as_str() {
                "total_z" => labelled(&summary, "Z"),
                "total_expected" => labelled(&summary, "E[Z]"),
                "total_variance" => labelled(&summary, "Var[Z]"),
                "lower" => labelled(&summary, "Lower"),
                "upper" => labelled(&summary, "Upper"),
                "passes" => 1.0 - labelled(&summary, "Calendar-year effect"),
                "n_diagonals" => (diagonals.rows - 1) as f64,
                q => {
                    let year: f64 = arg.parse().unwrap();
                    let r = (1..diagonals.rows)
                        .find(|&r| number(diagonals.get(r, 0)) == year)
                        .unwrap_or_else(|| panic!("{what}: no diagonal"));
                    let c = ["", "small", "large", "z", "n", "m", "expected", "variance"]
                        .iter()
                        .position(|h| *h == q)
                        .unwrap_or_else(|| panic!("{what}: unknown"));
                    number(diagonals.get(r, c))
                }
            }
        } else {
            let summary = ok("pro_res_devcor_test", vec![tri.clone(), ci.clone(), none()]);
            let pairs = ok("pro_res_devcor_test", vec![tri, ci, t("pairs")]);
            match quantity.as_str() {
                "t" => labelled(&summary, "T"),
                "variance" => labelled(&summary, "Var[T]"),
                "lower" => labelled(&summary, "Lower"),
                "upper" => labelled(&summary, "Upper"),
                "passes" => 1.0 - labelled(&summary, "Correlated"),
                "n_pairs" => (pairs.rows - 1) as f64,
                q => {
                    let k: usize = arg.parse().unwrap();
                    number(pairs.get(k + 1, if q == "t_k" { 2 } else { 3 }))
                }
            }
        };
        assert!(
            (got - want).abs() <= abs + rel * want.abs(),
            "{what}: {got} against {want}"
        );
        checked += 1;
    }
    checked
}

#[test]
fn mack_tests_match_r_chainladder() {
    assert!(check("reserving_assumptions_r.csv") > 500);
}

#[test]
fn mack_tests_match_chainladder_python() {
    assert!(check("reserving_assumptions_python.csv") > 100);
}

#[test]
fn the_backtest_is_the_librarys() {
    let (tri, origins) = data_triangle("raa", "value");
    let models = Grid::from_rows(vec![vec![Value::from("odp"), Value::from("gamma")]]);
    let args = |output: &str| {
        vec![
            tri.clone(),
            models.clone(),
            n(4.0),
            n(500.0),
            n(11.0),
            none(),
            t(output),
            origins.clone(),
        ]
    };
    let summary = ok("pro_res_backtest", args("summary"));
    let splits = ok("pro_res_backtest", args("diagonals"));
    let library = {
        let cells = TriangleFrame::new(&triangle(&tri).unwrap(), COLUMN, None).unwrap();
        let candidate = |name: &str, glm: Glm| GlmCandidate {
            name: name.into(),
            terms: Terms::new()
                .intercept()
                .factor("origin")
                .factor("development"),
            glm,
        };
        let odp = candidate("odp", Glm::over_dispersed_poisson());
        let gamma = candidate(
            "gamma",
            Glm::new(prospicio_models::Family::Gamma, prospicio_models::Link::Log),
        );
        let refs: [&dyn TriangleModel; 2] = [&odp, &gamma];
        diagonal_backtest(&cells, &refs, 4, 500, 11, 0.9).unwrap()
    };
    assert_eq!(summary.rows, 3);
    assert_eq!(splits.rows, 9);
    for m in 0..2 {
        for (k, _) in METRICS.iter().enumerate() {
            let want = library.mean(m, k);
            let got = summary.get(m + 1, k + 1);
            if want.is_finite() {
                assert_eq!(number(got), want);
            } else {
                assert!(got.is_blank());
            }
            for s in 0..4 {
                let want = library.split_scores(m, k)[s];
                if want.is_finite() {
                    assert_eq!(number(splits.get(1 + 4 * m + s, k + 2)), want);
                }
            }
        }
    }
    // The diagonals are labelled by calendar year: RAA's latest is 1990.
    assert_eq!(number(splits.get(4, 1)), 1990.0);
    assert_eq!(number(splits.get(1, 1)), 1987.0);
    let mut bad = args("summary");
    bad[1] = t("tweedie");
    assert!(fails("pro_res_backtest", bad).contains("models must be"));
}

#[test]
fn errors() {
    let (tri, _) = data_triangle("raa", "value");
    assert!(fails("pro_res_cy_test", vec![tri.clone(), n(1.5), none(), none()]).contains("ci"));
    assert!(fails("pro_res_devcor_test", vec![tri, none(), t("diagonals")]).contains("output"));
}
