//! The P&L ledger on a sheet (`docs/design/xll.md`, decision 43): a
//! book's premium, annual loss and expenses, and the annual covers it
//! buys, as signed legs over a distribution of annual gross losses, laid
//! out as `aggregate`'s `pnl` or `xpnl` (`prospicio_aggregate::pnl`, the
//! Aggregate lane's `pnl.md`).
//!
//! The gross loss is atoms in cells, `values` with `probs` (equally
//! likely when omitted, as simulated years are), so the ledger reads a
//! `PRO.AGG` grid, a year loss table or a model's simulations alike.
//! Expenses are rows of `label | premium | loss | fixed`; covers a table
//! with a header row (`name`, `limit`, `attachment`, `share`, `premium`,
//! `commission`, `profit_share`, `profit_allowance`), each an annual stop
//! loss on the gross.

use prospicio_aggregate::pnl::{Book, Expense, ExpenseBasis, Face, LADDER, Ledger, Source};
use prospicio_aggregate::{Layer, Tower};

use crate::pricing::numbers;
use crate::reserving::{number, percentiles, text_option};
use crate::sheet::{Grid, Value, position};

type Result<T> = std::result::Result<T, String>;

fn err(e: prospicio_core::Error) -> String {
    e.to_string()
}

/// The most atoms a source may have.
const MAX_ATOMS: usize = 1 << 21;

/// The gross annual losses and their probabilities.
fn source(values: &Grid, probs: &Grid) -> Result<Source> {
    let x = numbers(values, "values")?;
    if x.len() > MAX_ATOMS {
        return Err(format!("values has more than {MAX_ATOMS} atoms"));
    }
    let p = if probs.is_blank() {
        vec![1.0 / x.len() as f64; x.len()]
    } else {
        let p = numbers(probs, "probs")?;
        if p.len() != x.len() {
            return Err(format!(
                "{} probs for {} values: give one probability per value",
                p.len(),
                x.len()
            ));
        }
        p
    };
    Source::new(x, p).map_err(err)
}

fn cell_number(v: &Value, what: &str, r: usize, c: usize) -> Result<Option<f64>> {
    match v {
        Value::Number(x) => Ok(Some(*x)),
        v if v.is_blank() => Ok(None),
        _ => Err(format!("{what} {} must be a number", position(r, c))),
    }
}

/// Expense rows: `label | premium | loss | fixed`, the rate on the gross
/// premium, the rate on the gross loss and an amount, blanks left out; a
/// header row of text is skipped. A blank label takes the crate's name
/// (`premium expense`, `loss expense`, `fixed expense`, or `expense` for
/// several terms).
fn expenses(grid: &Grid) -> Result<Vec<Expense>> {
    if grid.is_blank() {
        return Ok(Vec::new());
    }
    if grid.columns > 4 {
        return Err(format!(
            "expenses has {} columns: give label, premium rate, loss rate and fixed amount",
            grid.columns
        ));
    }
    let header = (1..grid.columns).any(|c| matches!(grid.get(0, c), Value::Text(_)));
    let mut out = Vec::new();
    for r in usize::from(header)..grid.rows {
        let label = match grid.get(r, 0) {
            Value::Text(s) if !s.trim().is_empty() => Some(s.trim().to_owned()),
            v if v.is_blank() => None,
            _ => {
                return Err(format!(
                    "expenses {} must be a label (text) or blank",
                    position(r, 0)
                ));
            }
        };
        let mut terms = Vec::new();
        for (c, basis) in [
            (1, ExpenseBasis::Premium),
            (2, ExpenseBasis::Loss),
            (3, ExpenseBasis::Fixed),
        ] {
            if c < grid.columns
                && let Some(x) = cell_number(grid.get(r, c), "expenses", r, c)?
            {
                terms.push((basis, x));
            }
        }
        if terms.is_empty() {
            if label.is_some() {
                return Err(format!(
                    "expenses row {} has a label and no premium rate, loss rate or amount",
                    r + 1
                ));
            }
            continue;
        }
        out.push(Expense { label, terms });
    }
    Ok(out)
}

/// The columns a covers table may have.
const COVER_COLUMNS: &str =
    "name, limit, attachment, share, premium, commission, profit_share, profit_allowance";

/// The covers table: a header row naming its columns, then one annual
/// stop loss a row, lowest first.
fn covers(grid: &Grid) -> Result<Option<Tower>> {
    if grid.is_blank() {
        return Ok(None);
    }
    if grid.rows < 2 {
        return Err(format!(
            "covers needs a header row ({COVER_COLUMNS}) and a row per cover"
        ));
    }
    let names = (0..grid.columns)
        .map(|c| match grid.get(0, c) {
            Value::Text(s) => {
                let k = s.trim().to_ascii_lowercase().replace([' ', '-'], "_");
                if COVER_COLUMNS.split(", ").any(|n| n == k) {
                    Ok(k)
                } else {
                    Err(format!(
                        "covers {}: unknown column {s:?}; the columns are {COVER_COLUMNS}",
                        position(0, c)
                    ))
                }
            }
            _ => Err(format!(
                "covers {} must be a column name ({COVER_COLUMNS})",
                position(0, c)
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    if let Some(i) = (1..names.len()).find(|&i| names[..i].contains(&names[i])) {
        return Err(format!("covers has two {} columns", names[i]));
    }
    for required in ["limit", "attachment"] {
        if !names.iter().any(|n| n == required) {
            return Err(format!("covers needs a {required} column"));
        }
    }
    let column = |name: &str| names.iter().position(|n| n == name);
    let mut layers = Vec::new();
    for r in 1..grid.rows {
        if (0..grid.columns).all(|c| grid.get(r, c).is_blank()) {
            continue;
        }
        let get = |name: &str| -> Result<Option<f64>> {
            column(name).map_or(Ok(None), |c| cell_number(grid.get(r, c), "covers", r, c))
        };
        let name = match column("name").map(|c| grid.get(r, c)) {
            Some(Value::Text(s)) if !s.trim().is_empty() => s.trim().to_owned(),
            Some(Value::Number(x)) => crate::glm::number_text(*x),
            _ => {
                let (l, a) = (
                    get("limit")?.unwrap_or(0.0),
                    get("attachment")?.unwrap_or(0.0),
                );
                format!(
                    "agg {} xs {}",
                    crate::glm::number_text(l),
                    crate::glm::number_text(a)
                )
            }
        };
        let need = |name: &str| -> Result<f64> {
            get(name)?.ok_or_else(|| format!("covers row {} needs a {name}", r + 1))
        };
        let mut layer = Layer::stop_loss(name, need("limit")?, need("attachment")?).map_err(err)?;
        if let Some(s) = get("share")? {
            layer = layer.share(s).map_err(err)?;
        }
        if let Some(p) = get("premium")? {
            layer = layer.deposit_premium(p).map_err(err)?;
        }
        if let Some(c) = get("commission")? {
            layer = layer.ceding_commission(c).map_err(err)?;
        }
        match (get("profit_share")?, get("profit_allowance")?) {
            (None, None) => {}
            (Some(s), a) => layer = layer.profit_commission(s, a.unwrap_or(0.0)).map_err(err)?,
            (None, Some(_)) => {
                return Err(format!(
                    "covers row {} has a profit_allowance and no profit_share",
                    r + 1
                ));
            }
        }
        layers.push(layer);
    }
    if layers.is_empty() {
        return Err("covers has a header and no covers".into());
    }
    Tower::new(layers).map(Some).map_err(err)
}

fn face(grid: &Grid) -> Result<Face> {
    let f = text_option(grid, "face")?.map(|s| s.replace([' ', '_'], "-"));
    Ok(match f.as_deref() {
        None | Some("walk") => Face::Walk,
        Some("consolidated") => Face::Consolidated,
        Some("peel-top-down" | "top-down") => Face::PeelTopDown,
        Some("peel-bottom-up" | "bottom-up") => Face::PeelBottomUp,
        Some(other) => {
            return Err(format!(
                "face must be \"walk\" (default), \"consolidated\", \"top-down\" or \
                 \"bottom-up\", got {other:?}"
            ));
        }
    })
}

fn opt(x: Option<f64>) -> Value {
    x.map_or(Value::Empty, Value::number)
}

fn level(p: f64) -> String {
    format!("{}%", crate::glm::number_text((p * 1e10).round() / 1e8))
}

/// The sheet: every row with its mean, SD, CV and skewness, then each
/// scenario column, the rows' means given the result at its lower
/// `q`-quantile, which foot down the sheet.
fn sheet(ledger: &Ledger, levels: &[f64]) -> Result<Grid> {
    let mut header: Vec<Value> = ["Step", "Side", "Line", "EX", "SD", "CV", "Skew"]
        .iter()
        .map(|h| Value::from(*h))
        .collect();
    header.extend(
        levels
            .iter()
            .map(|p| Value::from(format!("Scenario {}", level(*p)).as_str())),
    );
    let scenarios = levels
        .iter()
        .map(|q| ledger.scenarios(*q))
        .collect::<prospicio_core::Result<Vec<_>>>()
        .map_err(err)?;
    let mut rows = vec![header];
    for (i, r) in ledger.rows().iter().enumerate() {
        let mut row = vec![
            Value::from(r.step.as_str()),
            Value::from(r.side.name()),
            Value::from(r.display.as_str()),
            Value::number(r.mean()),
            Value::number(r.sd()),
            opt(r.cv()),
            opt(r.skew()),
        ];
        row.extend(scenarios.iter().map(|s| Value::number(s[i])));
        rows.push(row);
    }
    Ok(Grid::from_rows(rows))
}

fn card(ledger: &Ledger) -> Result<Grid> {
    let mut rows = vec![
        [
            "Step", "Side", "EX", "SD", "CV", "Skew", "P01", "Median", "P99",
        ]
        .iter()
        .map(|h| Value::from(*h))
        .collect::<Vec<_>>(),
    ];
    for r in ledger.card().map_err(err)? {
        rows.push(vec![
            Value::from(r.step.as_str()),
            Value::from(r.side.as_str()),
            Value::number(r.ex),
            Value::number(r.sd),
            opt(r.cv),
            opt(r.skew),
            Value::number(r.p01),
            Value::number(r.median),
            Value::number(r.p99),
        ]);
    }
    Ok(Grid::from_rows(rows))
}

fn ratios(ledger: &Ledger) -> Grid {
    let mut rows = vec![
        [
            "Step",
            "Premium",
            "Loss",
            "Expense",
            "Result",
            "SD",
            "LR",
            "ER",
            "CR",
            "E[LR]",
            "E[ER]",
            "E[CR]",
            "Premium share",
            "Result share",
        ]
        .iter()
        .map(|h| Value::from(*h))
        .collect::<Vec<_>>(),
    ];
    for r in ledger.ratios() {
        rows.push(vec![
            Value::from(r.step.as_str()),
            Value::number(r.p),
            Value::number(r.l),
            Value::number(r.e),
            Value::number(r.m),
            opt(r.sd),
            opt(r.lr),
            opt(r.er),
            opt(r.cr),
            opt(r.e_lr),
            opt(r.e_er),
            opt(r.e_cr),
            opt(r.p_share),
            opt(r.m_share),
        ]);
    }
    Grid::from_rows(rows)
}

/// `PRO.PNL(values, [probs], premium, [expenses], [covers], [face],
/// [output], [levels], [expense_at_mean])`: the book's ledger over the
/// gross annual losses. `output` is `"sheet"` (the default: every row
/// with its statistics and the scenario columns at `levels`, 1%, 5%,
/// 10%, 25%, 50%, 75%, 90%, 95% and 99% if omitted), `"card"` (each
/// block's consideration, obligation and margin with marginal
/// quantiles) or `"ratios"` (each block's premium, loss, expense, result
/// and their ratios).
pub fn pnl(args: &[Grid]) -> Result<Grid> {
    let gross = source(&args[0], &args[1])?;
    let mut book = Book::new("Book", number(&args[2], "premium")?);
    for e in expenses(&args[3])? {
        book = book.expense(e);
    }
    let tower = covers(&args[4])?;
    let face = face(&args[5])?;
    if !args[8].is_blank() {
        let at_mean = match args[8].as_scalar() {
            Some(Value::Bool(b)) => *b,
            _ => return Err("expense_at_mean must be TRUE or FALSE".into()),
        };
        if at_mean {
            book = book.expense_at_expected_loss();
        }
    }
    let ledger = book.ledger(&gross, tower.as_ref(), face).map_err(err)?;
    match text_option(&args[6], "output")?.as_deref() {
        None | Some("sheet") => {
            let levels = if args[7].is_blank() {
                LADDER.to_vec()
            } else {
                percentiles(&args[7])?
            };
            sheet(&ledger, &levels)
        }
        Some("card") => card(&ledger),
        Some("ratios") => Ok(ratios(&ledger)),
        Some(other) => Err(format!(
            "output must be \"sheet\" (default), \"card\" or \"ratios\", got {other:?}"
        )),
    }
}
