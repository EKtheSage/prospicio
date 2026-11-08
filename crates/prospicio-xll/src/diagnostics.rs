//! Reserving diagnostics on a sheet (`docs/design/xll.md`, decision 57):
//! Mack's tests of the chain ladder's assumptions and the backtest on held
//! out calendar diagonals (`prospicio_reserving::assumptions`,
//! `prospicio_reserving::backtest`).

use prospicio_core::Month;
use prospicio_glm::Glm;
use prospicio_models::{Family, Link, Terms};
use prospicio_reserving::{
    CALENDAR_YEAR_CI, DEVELOPMENT_CORRELATION_CI, GlmCandidate, METRICS, TriangleFrame,
    TriangleModel, calendar_year_effect, development_correlation, diagonal_backtest,
};

use crate::reserving::{COLUMN, count, number_or, seed, text_option, triangle};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

fn err(e: impl ToString) -> String {
    e.to_string()
}

/// The label of calendar period `k` (0 the first origin's first diagonal):
/// the first origin's label plus `k` when `origins` starts with a number,
/// else `k + 1`.
fn calendar_label(origins: &Grid, k: i64) -> Value {
    match origins.cells.first() {
        Some(Value::Number(first)) => Value::Number(first + k as f64),
        _ => Value::Number((k + 1) as f64),
    }
}

/// A confidence level argument, `default` when blank.
fn ci(grid: &Grid, default: f64) -> Result<f64> {
    let ci = number_or(grid, "ci", default)?;
    if !(ci > 0.0 && ci < 1.0) {
        return Err(format!("ci must be between 0 and 1, got {ci}"));
    }
    Ok(ci)
}

/// A summary table: label and value rows.
fn summary(rows: Vec<(&str, Value)>) -> Grid {
    Grid::from_rows(
        rows.into_iter()
            .map(|(l, v)| vec![Value::from(l), v])
            .collect(),
    )
}

/// `PRO.RES.CY.TEST(triangle, [ci], [output], [origins])`: Mack's test for
/// calendar-year effects (R ChainLadder's `cyEffTest`), at `ci` (0.95 if
/// omitted). `output` `"summary"` (default) gives Z, its mean and variance,
/// the interval and whether Z lies in it; `"diagonals"` a row per calendar
/// diagonal with its small and large link ratios, Z, n, m, E[Z] and
/// Var[Z].
pub fn calendar_year(args: &[Grid]) -> Result<Grid> {
    let tri = triangle(&args[0])?;
    let test = calendar_year_effect(&tri, COLUMN, ci(&args[1], CALENDAR_YEAR_CI)?).map_err(err)?;
    match text_option(&args[2], "output")?.as_deref() {
        None | Some("summary") => Ok(summary(vec![
            ("Z", Value::number(test.z)),
            ("E[Z]", Value::number(test.expected)),
            ("Var[Z]", Value::number(test.variance)),
            ("Lower", Value::number(test.lower)),
            ("Upper", Value::number(test.upper)),
            ("Confidence", Value::number(test.ci)),
            ("Calendar-year effect", Value::Bool(!test.passes())),
        ])),
        Some("diagonals") => {
            let start = Month::january(2001);
            let mut rows = vec![
                [
                    "Diagonal", "Small", "Large", "Z", "n", "m", "E[Z]", "Var[Z]",
                ]
                .map(Value::from)
                .to_vec(),
            ];
            for d in &test.diagonals {
                let k = i64::from(d.valuation.year() - start.year());
                rows.push(vec![
                    calendar_label(&args[3], k),
                    Value::number(d.small as f64),
                    Value::number(d.large as f64),
                    Value::number(d.z as f64),
                    Value::number(d.n as f64),
                    Value::number(d.m as f64),
                    Value::number(d.expected),
                    Value::number(d.variance),
                ]);
            }
            Ok(Grid::from_rows(rows))
        }
        Some(other) => Err(format!(
            "output must be \"summary\" or \"diagonals\", got {other:?}"
        )),
    }
}

/// `PRO.RES.DEVCOR.TEST(triangle, [ci], [output])`: Mack's test for
/// correlation between adjacent development factors (R ChainLadder's
/// `dfCorTest`), at `ci` (0.5 if omitted, Mack's). `output` `"summary"`
/// (default) gives T, its variance, the interval and whether T lies
/// outside it; `"pairs"` a row per pair of adjacent link columns with its
/// first link, the origins in common, Spearman's T_k and its weight.
pub fn development_factors(args: &[Grid]) -> Result<Grid> {
    let tri = triangle(&args[0])?;
    let test = development_correlation(&tri, COLUMN, ci(&args[1], DEVELOPMENT_CORRELATION_CI)?)
        .map_err(err)?;
    match text_option(&args[2], "output")?.as_deref() {
        None | Some("summary") => Ok(summary(vec![
            ("T", Value::number(test.t)),
            ("Var[T]", Value::number(test.variance)),
            ("Lower", Value::number(test.lower)),
            ("Upper", Value::number(test.upper)),
            ("Confidence", Value::number(test.ci)),
            ("Correlated", Value::Bool(!test.passes())),
        ])),
        Some("pairs") => {
            let mut rows = vec![
                ["Link", "Origins", "T_k", "Weight"]
                    .map(Value::from)
                    .to_vec(),
            ];
            for p in &test.pairs {
                rows.push(vec![
                    Value::Text(format!("{}-{}", p.age, p.age + 12)),
                    Value::number(p.n as f64),
                    if p.t.is_nan() {
                        Value::Empty
                    } else {
                        Value::number(p.t)
                    },
                    Value::number(p.weight),
                ]);
            }
            Ok(Grid::from_rows(rows))
        }
        Some(other) => Err(format!(
            "output must be \"summary\" or \"pairs\", got {other:?}"
        )),
    }
}

/// A backtest model by name: a GLM on the incremental cells with origin
/// and development factors.
fn model(name: &str) -> Result<GlmCandidate> {
    let glm = match name {
        "odp" => Glm::over_dispersed_poisson(),
        "poisson" => Glm::new(Family::Poisson, Link::Log),
        "gamma" => Glm::new(Family::Gamma, Link::Log),
        other => {
            return Err(format!(
                "models must be \"odp\", \"poisson\" or \"gamma\", got {other:?}"
            ));
        }
    };
    Ok(GlmCandidate {
        name: name.into(),
        terms: Terms::new()
            .intercept()
            .factor("origin")
            .factor("development"),
        glm,
    })
}

/// `PRO.RES.BACKTEST(triangle, [models], [diagonals], [sims], [seed],
/// [interval], [output], [origins])`: each model fitted on the earlier
/// diagonals forecasts each of the latest `diagonals` (3 if omitted), with
/// `sims` joint draws (1000) from `seed`. `output` `"summary"` (default)
/// gives a row per model of each metric's mean over the diagonals;
/// `"diagonals"` a row per model and diagonal, with the cells left out of
/// scoring.
pub fn backtest(args: &[Grid]) -> Result<Grid> {
    let tri = triangle(&args[0])?;
    let names: Vec<String> = if args[1].is_blank() {
        vec!["odp".into()]
    } else {
        args[1]
            .cells
            .iter()
            .filter(|v| !v.is_blank())
            .map(|v| match v {
                Value::Text(s) => Ok(s.trim().to_lowercase()),
                other => Err(format!(
                    "models must be names such as \"odp\", got {other:?}"
                )),
            })
            .collect::<Result<_>>()?
    };
    let models = names.iter().map(|n| model(n)).collect::<Result<Vec<_>>>()?;
    let refs: Vec<&dyn TriangleModel> = models.iter().map(|m| m as &dyn TriangleModel).collect();
    let n_diagonals = count(&args[2], "diagonals", 3, 1000)?;
    let n_sims = count(&args[3], "sims", 1000, 1_000_000)?;
    let seed = seed(&args[4])?;
    let interval = number_or(&args[5], "interval", 0.9)?;
    let cells = TriangleFrame::new(&tri, COLUMN, None).map_err(err)?;
    let bt = diagonal_backtest(&cells, &refs, n_diagonals, n_sims, seed, interval).map_err(err)?;
    let header = |first: &[&str]| -> Vec<Value> {
        first
            .iter()
            .chain(METRICS.iter())
            .map(|&s| Value::from(s))
            .collect()
    };
    match text_option(&args[6], "output")?.as_deref() {
        None | Some("summary") => {
            let mut rows = vec![header(&["Model"])];
            for (i, name) in bt.models().iter().enumerate() {
                let mut row = vec![Value::from(name.as_str())];
                row.extend((0..METRICS.len()).map(|m| Value::number(bt.mean(i, m))));
                rows.push(row);
            }
            Ok(Grid::from_rows(rows))
        }
        Some("diagonals") => {
            let mut h = header(&["Model", "Diagonal"]);
            h.push(Value::from("Excluded"));
            let mut rows = vec![h];
            for (i, name) in bt.models().iter().enumerate() {
                for s in 0..bt.n_splits() {
                    let mut row = vec![
                        Value::from(name.as_str()),
                        calendar_label(&args[7], bt.diagonals()[s]),
                    ];
                    row.extend((0..METRICS.len()).map(|m| Value::number(bt.split_scores(i, m)[s])));
                    row.push(Value::number(bt.excluded()[s] as f64));
                    rows.push(row);
                }
            }
            Ok(Grid::from_rows(rows))
        }
        Some(other) => Err(format!(
            "output must be \"summary\" or \"diagonals\", got {other:?}"
        )),
    }
}
