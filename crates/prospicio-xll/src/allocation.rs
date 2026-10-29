//! Capital allocation on a sheet (`docs/design/xll.md`, decision 45): a
//! portfolio's risk measure shared among its units, from columns of
//! simulated draws, over `prospicio_prob::capital`.
//!
//! `draws` is a block with a column per unit and a row per simulation,
//! equally likely, with an optional header row of the units' names; the
//! units must add up to the portfolio (lines or segments, not gross and
//! ceded side by side). `PRO.ALLOCATE` allocates a distortion risk
//! measure of the total (TVaR, proportional hazard, Wang and the others
//! `PRO.DISTORTION` names) by Euler (the natural allocation), covariance,
//! proportional, marginal or Shapley; `PRO.ALLOCATE.TOTAL` allocates an
//! amount given from outside, a premium or a capital, by the haircut or
//! the equal risk rule on a stand-alone VaR, TVaR or EPD.

use prospicio_prob::capital::{AllocationMethod, LevelMeasure};
use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};

use crate::distortions;
use crate::reserving::{number, text_option};
use crate::sheet::{Grid, Value, position};

type Result<T> = std::result::Result<T, String>;

fn err(e: prospicio_core::Error) -> String {
    e.to_string()
}

/// The most cells `draws` may have.
const MAX_CELLS: usize = 1 << 24;

/// The units' names and their joint draws. A first row with text in it
/// is the header; rows blank in every cell are skipped.
fn draws(g: &Grid) -> Result<(Vec<Value>, PredictiveDistribution)> {
    if g.is_blank() {
        return Err("draws is required".into());
    }
    if g.cells.len() > MAX_CELLS {
        return Err(format!("draws has more than {MAX_CELLS} cells"));
    }
    let m = g.columns;
    let header = (0..m).any(|c| matches!(g.get(0, c), Value::Text(s) if !s.trim().is_empty()));
    let names: Vec<Value> = (0..m)
        .map(|c| match g.get(0, c) {
            Value::Text(s) if header && !s.trim().is_empty() => Value::from(s.trim()),
            _ => Value::Text(format!("Unit {}", c + 1)),
        })
        .collect();
    let mut x = Vec::with_capacity(g.cells.len());
    for r in usize::from(header)..g.rows {
        if (0..m).all(|c| g.get(r, c).is_blank()) {
            continue;
        }
        for c in 0..m {
            match g.get(r, c) {
                Value::Number(v) => x.push(*v),
                other => {
                    return Err(format!(
                        "draws {} must be a number, got {other:?}",
                        position(r, c)
                    ));
                }
            }
        }
    }
    if x.is_empty() {
        return Err("draws has no simulations".into());
    }
    let pd = PredictiveDistribution::from_draws(
        vec!["unit".into()],
        (0..m).map(|j| vec![KeyValue::from(j as i64)]).collect(),
        x,
        Provenance::new("worksheet"),
    )
    .map_err(err)?;
    Ok((names, pd))
}

fn method(g: &Grid) -> Result<AllocationMethod> {
    Ok(match text_option(g, "method")?.as_deref() {
        None | Some("euler") | Some("natural") => AllocationMethod::Euler,
        Some("covariance") => AllocationMethod::Covariance,
        Some("covariance_margin") => AllocationMethod::CovarianceMargin,
        Some("proportional") => AllocationMethod::Proportional,
        Some("marginal") => AllocationMethod::Marginal,
        Some("shapley") => AllocationMethod::Shapley,
        Some(other) => {
            return Err(format!(
                "method must be \"euler\" (or \"natural\"), \"covariance\", \"covariance_margin\", \
                 \"proportional\", \"marginal\" or \"shapley\", got {other:?}"
            ));
        }
    })
}

fn header(columns: &[&str]) -> Vec<Value> {
    columns.iter().map(|&c| Value::from(c)).collect()
}

/// A distortion risk measure of the portfolio shared among its units:
/// `PRO.ALLOCATE(draws, family, parameter, [method], [r0])`, `family`,
/// `parameter` and `r0` as `PRO.DISTORTION` takes them. A row per unit:
/// its name, its stand-alone measure `ρ(X_j)`, its allocation, its share
/// of the portfolio's measure and its diversification (stand-alone less
/// allocated); then the units' totals, and the portfolio's measure `ρ(S)`
/// with the diversification benefit `Σ ρ(X_j) − ρ(S)`. Every method but
/// the marginal allocates `ρ(S)` in full; the marginal's shortfall is the
/// capital no single unit is responsible for.
pub fn allocate(args: &[Grid]) -> Result<Grid> {
    let (names, pd) = draws(&args[0])?;
    let d = distortions::parse(&args[1], &args[2], &args[4])?;
    let a = pd.capital(&d, method(&args[3])?).map_err(err)?;
    let mut rows = vec![header(&[
        "Unit",
        "Stand-alone",
        "Allocated",
        "Share",
        "Diversification",
    ])];
    let div = a.diversification();
    for (j, name) in names.into_iter().enumerate() {
        rows.push(vec![
            name,
            Value::number(a.standalone[j]),
            Value::number(a.allocated[j]),
            Value::number(a.allocated[j] / a.total),
            Value::number(div[j]),
        ]);
    }
    let allocated: f64 = a.allocated.iter().sum();
    rows.push(vec![
        Value::from("Total"),
        Value::number(a.standalone.iter().sum()),
        Value::number(allocated),
        Value::number(allocated / a.total),
        Value::number(div.iter().sum()),
    ]);
    rows.push(vec![
        Value::from("Portfolio"),
        Value::number(a.total),
        Value::number(a.total),
        Value::number(1.0),
        Value::number(a.diversification_benefit()),
    ]);
    Ok(Grid::from_rows(rows))
}

/// An amount given from outside shared among the units:
/// `PRO.ALLOCATE.TOTAL(draws, total, [method], [measure], [level])`.
/// `method` is `"haircut"` (the default: each unit's stand-alone measure
/// at `level` scaled to `total`) or `"equal_risk"` (the one level at which
/// the stand-alone measures add up to `total`, which it reports);
/// `measure` is `"tvar"` (the default), `"var"` or `"epd"` (the assets at
/// which the expected policyholder deficit is `level` times the mean).
/// The haircut's `level` defaults to 0.99 for VaR and TVaR and is
/// required for EPD; the equal risk rule takes none. A row per unit: its
/// name, its stand-alone measure at the level, its allocation and share;
/// then the totals and the level.
pub fn allocate_total(args: &[Grid]) -> Result<Grid> {
    let (names, pd) = draws(&args[0])?;
    let total = number(&args[1], "total")?;
    let measure = match text_option(&args[3], "measure")?.as_deref() {
        None | Some("tvar") => LevelMeasure::Tvar,
        Some("var") => LevelMeasure::Var,
        Some("epd") => LevelMeasure::Epd,
        Some(other) => {
            return Err(format!(
                "measure must be \"tvar\", \"var\" or \"epd\", got {other:?}"
            ));
        }
    };
    let a = match text_option(&args[2], "method")?.as_deref() {
        None | Some("haircut") => {
            let level = match (args[4].is_blank(), measure) {
                (true, LevelMeasure::Epd) => {
                    return Err(
                        "the haircut on EPD needs a level, an EPD ratio such as 0.01".into(),
                    );
                }
                (true, _) => 0.99,
                (false, _) => number(&args[4], "level")?,
            };
            pd.haircut(total, measure, level).map_err(err)?
        }
        Some("equal_risk") => {
            if !args[4].is_blank() {
                return Err("the equal risk rule finds the level: leave level out".into());
            }
            pd.equal_risk(total, measure).map_err(err)?
        }
        Some(other) => {
            return Err(format!(
                "method must be \"haircut\" or \"equal_risk\", got {other:?}"
            ));
        }
    };
    let mut rows = vec![header(&["Unit", "Stand-alone", "Allocated", "Share"])];
    for (j, name) in names.into_iter().enumerate() {
        rows.push(vec![
            name,
            Value::number(a.standalone[j]),
            Value::number(a.allocated[j]),
            Value::number(a.allocated[j] / total),
        ]);
    }
    rows.push(vec![
        Value::from("Total"),
        Value::number(a.standalone.iter().sum()),
        Value::number(a.allocated.iter().sum()),
        Value::number(1.0),
    ]);
    rows.push(vec![
        Value::from("Level"),
        Value::number(a.level),
        Value::Empty,
        Value::Empty,
    ]);
    Ok(Grid::from_rows(rows))
}
