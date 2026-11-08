//! Friedland-style exhibits and the weighted selection of ultimates
//! (`docs/design/xll.md`, decision 51): `prospicio_reserving`'s
//! `DevelopmentExhibit` (Friedland's chapter 7 Exhibit I, parts 3 and 4)
//! and `UltimateExhibit::select` (chapter 15 Exhibit I, and
//! chainladder-python's `VotingChainladder`).

use prospicio_core::{Lag, Period};
use prospicio_reserving::{
    ChainLadder, DevelopmentExhibit, MethodWeights, UltimateExhibit, UltimateSelection,
    friedland_averages,
};

use crate::diagonal::months;
use crate::reserving::{COLUMN, development_with, link_labels, origin_labels, tail, triangle};
use crate::sheet::{Grid, Value, position};

type Result<T> = std::result::Result<T, String>;

/// Friedland's names of the six averages, as a sheet labels them.
const AVERAGE_LABELS: [&str; 6] = [
    "Simple latest 5",
    "Simple latest 3",
    "Medial latest 5x1",
    "Volume latest 5",
    "Volume latest 3",
    "Geometric latest 4",
];

/// `PRO.RES.DEV.EXHIBIT(triangle, [average], [tail], [selected], [kind],
/// [weight], [pattern_ages])`: Friedland's averages of the link ratios
/// (simple latest 5 and 3, medial latest 5x1, volume-weighted latest 5
/// and 3, geometric latest 4), then the selected factors, the factor to
/// ultimate and the percent developed, a column per link and the tail's
/// last; the selection is the chain ladder's under the other arguments.
pub fn development(args: &[Grid]) -> Result<Grid> {
    let dev = development_with(
        &args[1],
        [&args[3], &args[4], &args[5], &args[6]],
        &Grid::missing(),
    )?;
    let selection = ChainLadder {
        development: dev,
        tail: tail(&args[2])?,
    };
    let ex = DevelopmentExhibit::with_averages(
        &triangle(&args[0])?,
        COLUMN,
        &selection,
        &friedland_averages(),
    )
    .map_err(|e| e.to_string())?;
    let n = ex.ages.len();
    let mut header = vec![Value::Empty];
    header.extend(link_labels(n));
    let mut rows = vec![header];
    let finite = |x: f64| {
        if x.is_finite() {
            Value::number(x)
        } else {
            Value::Empty
        }
    };
    for (label, (_, factors)) in AVERAGE_LABELS.iter().zip(&ex.averages) {
        let mut row = vec![Value::from(*label)];
        row.extend(factors.iter().take(n - 1).map(|&f| finite(f)));
        row.push(Value::Empty);
        rows.push(row);
    }
    let mut selected = vec![Value::from("Selected")];
    selected.extend(ex.selected.iter().take(n - 1).map(|&f| finite(f)));
    selected.push(Value::number(ex.tail));
    rows.push(selected);
    let mut cdf = vec![Value::from("CDF")];
    cdf.extend(ex.cdf.iter().map(|&f| Value::number(f)));
    rows.push(cdf);
    let mut percent = vec![Value::from("Percent developed")];
    percent.extend(ex.percent_developed.iter().map(|&f| Value::number(f)));
    rows.push(percent);
    Ok(Grid::from_rows(rows))
}

/// A column of `n` numbers, one per origin; `None` when blank.
fn per_origin(grid: &Grid, name: &str, n: usize) -> Result<Option<Vec<f64>>> {
    if grid.is_blank() {
        return Ok(None);
    }
    if grid.cells.len() != n || (grid.rows != 1 && grid.columns != 1) {
        return Err(format!(
            "{name} must be a row or column of {n} values, one per origin"
        ));
    }
    grid.cells
        .iter()
        .enumerate()
        .map(|(i, v)| match v {
            Value::Number(x) => Ok(*x),
            other => Err(format!("{name} {} must be a number, got {other:?}", i + 1)),
        })
        .collect::<Result<_>>()
        .map(Some)
}

/// A weight cell: a blank is 0.
fn weight(v: &Value, r: usize, c: usize) -> Result<f64> {
    match v {
        Value::Number(x) => Ok(*x),
        v if v.is_blank() => Ok(0.0),
        other => Err(format!(
            "weights {} must be a number, got {other:?}",
            position(r, c)
        )),
    }
}

/// `PRO.RES.SELECT(ultimates, weights, [ages], [reported], [paid],
/// [origins], [methods])`: the selected ultimate per origin as a weighting
/// of the methods' ultimates (a column per method, blank where one does
/// not project). `weights` is a row with a weight per method for every
/// origin, a table of the same shape as `ultimates` (a weight per origin
/// and method), or, with `ages` (each origin's age in months), a table
/// whose first column is the age from which its row holds. Each origin's
/// weights must sum to 1. Gives the ultimate, with IBNR (less `reported`)
/// and unpaid (less `paid`) when given, and each method's weight, with a
/// total.
pub fn select(args: &[Grid]) -> Result<Grid> {
    let ultimates = &args[0];
    let (n, m) = (ultimates.rows, ultimates.columns);
    if ultimates.is_blank() || n == 0 || m == 0 {
        return Err("ultimates is required: a column per method, a row per origin".into());
    }
    let labels = origin_labels(&args[5], n)?;
    let names: Vec<String> = if args[6].is_blank() {
        (1..=m).map(|k| format!("Method {k}")).collect()
    } else {
        if args[6].cells.len() != m {
            return Err(format!(
                "methods must have {m} names, one per column of ultimates"
            ));
        }
        args[6]
            .cells
            .iter()
            .map(|v| match v {
                Value::Text(s) => Ok(s.clone()),
                Value::Number(x) => Ok(x.to_string()),
                other => Err(format!("methods must be names, got {other:?}")),
            })
            .collect::<Result<_>>()?
    };
    let ages: Option<Vec<Lag>> = if args[2].is_blank() {
        None
    } else {
        if args[2].cells.len() != n || (args[2].rows != 1 && args[2].columns != 1) {
            return Err(format!(
                "ages must be a row or column of {n} ages in months, one per origin"
            ));
        }
        Some(
            args[2]
                .cells
                .iter()
                .map(|v| {
                    months(v).ok_or_else(|| {
                        format!("ages must be whole numbers of months from 1, got {v:?}")
                    })
                })
                .collect::<Result<_>>()?,
        )
    };
    // An origin's period: its label's year when the label is one, so
    // that messages name it; else 2001, 2002, ...
    let origins: Vec<Period> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| match l {
            Value::Number(y) if y.fract() == 0.0 && (1000.0..=9999.0).contains(y) => {
                Period::year(*y as i32)
            }
            _ => Period::year(2001 + i as i32),
        })
        .collect();
    let origins = if (1..n).any(|i| origins[..i].contains(&origins[i])) {
        (0..n).map(|i| Period::year(2001 + i as i32)).collect()
    } else {
        origins
    };
    let exhibit_ages = ages
        .clone()
        .unwrap_or_else(|| (0..n).map(|i| 12 * (n - i) as Lag).collect());
    let mut ex = UltimateExhibit::new(
        origins,
        exhibit_ages,
        per_origin(&args[3], "reported", n)?,
        per_origin(&args[4], "paid", n)?,
    )
    .map_err(|e| e.to_string())?;
    for (k, name) in names.iter().enumerate() {
        let column = (0..n)
            .map(|r| match ultimates.get(r, k) {
                Value::Number(x) => Ok(*x),
                v if v.is_blank() => Ok(f64::NAN),
                other => Err(format!(
                    "ultimates {} must be a number, got {other:?}",
                    position(r, k)
                )),
            })
            .collect::<Result<Vec<f64>>>()?;
        ex = ex
            .with_method(name.clone(), column)
            .map_err(|e| e.to_string())?;
    }
    let w = &args[1];
    let weights = if ages.is_some() {
        if w.columns != m + 1 {
            return Err(format!(
                "with ages, weights is a table of {} columns: the age from which a row holds, then a weight per method",
                m + 1
            ));
        }
        let rows = (0..w.rows)
            .map(|r| {
                let age = months(w.get(r, 0)).ok_or_else(|| {
                    format!(
                        "weights {} must be an age in months, got {:?}",
                        position(r, 0),
                        w.get(r, 0)
                    )
                })?;
                let row = (0..m)
                    .map(|k| Ok((names[k].clone(), weight(w.get(r, k + 1), r, k + 1)?)))
                    .collect::<Result<_>>()?;
                Ok((age, row))
            })
            .collect::<Result<_>>()?;
        MethodWeights::ByAge(rows)
    } else if w.rows == 1 && w.columns == m && n != 1 {
        MethodWeights::ByOrigin(
            (0..m)
                .map(|k| Ok((names[k].clone(), vec![weight(w.get(0, k), 0, k)?; n])))
                .collect::<Result<_>>()?,
        )
    } else if w.rows == n && w.columns == m {
        MethodWeights::ByOrigin(
            (0..m)
                .map(|k| {
                    Ok((
                        names[k].clone(),
                        (0..n)
                            .map(|r| weight(w.get(r, k), r, k))
                            .collect::<Result<_>>()?,
                    ))
                })
                .collect::<Result<_>>()?,
        )
    } else {
        return Err(format!(
            "weights must be a row of {m} (one per method), a table of {n} x {m} like ultimates, or with ages a table by age"
        ));
    };
    let sel = ex
        .select(&UltimateSelection::new(weights))
        .map_err(|e| e.to_string())?;
    let (ibnr, unpaid) = (sel.ibnr(), sel.unpaid());
    let mut header = vec![Value::from("Origin")];
    let mut columns: Vec<Vec<f64>> = Vec::new();
    for (name, v) in [
        ("Reported", sel.reported.clone()),
        ("Paid", sel.paid.clone()),
        ("Ultimate", Some(sel.ultimate.clone())),
        ("IBNR", ibnr),
        ("Unpaid", unpaid),
    ] {
        if let Some(v) = v {
            header.push(Value::from(name));
            columns.push(v);
        }
    }
    let totals = columns.len();
    for (name, v) in &sel.weights {
        header.push(Value::Text(format!("Weight {name}")));
        columns.push(v.clone());
    }
    let mut rows = vec![header];
    for (o, label) in labels.into_iter().enumerate() {
        let mut row = vec![label];
        row.extend(columns.iter().map(|c| Value::number(c[o])));
        rows.push(row);
    }
    let mut total = vec![Value::from("Total")];
    total.extend(
        columns[..totals]
            .iter()
            .map(|c| Value::number(c.iter().sum())),
    );
    total.extend(std::iter::repeat_n(Value::Empty, columns.len() - totals));
    rows.push(total);
    Ok(Grid::from_rows(rows))
}
