//! A cover's quote (`docs/design/xll.md`, decision 28): from a risk-loaded
//! price to the premium the cedant pays, grossed up for expenses and
//! brokerage, with the ratios the market quotes, and what a quoted premium
//! leaves the reinsurer (`prospicio_pricing::quote`).

use prospicio_pricing::quote::{Loads, Quote};
use prospicio_pricing::risk_load::Price;

use crate::pricing::{numbers, scalar};
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

/// A non-negative number, `default` if omitted.
fn optional(grid: &Grid, name: &str, default: f64) -> Result<f64> {
    if grid.is_blank() {
        Ok(default)
    } else {
        scalar(grid, name)
    }
}

const ROWS: [&str; 13] = [
    "Premium",
    "Expected loss",
    "Risk premium",
    "Expenses",
    "Brokerage",
    "Net of brokerage",
    "Margin",
    "Capital",
    "Return on capital",
    "Loss ratio",
    "Multiple",
    "Rate on line",
    "Payback",
];

/// One column of the exhibit; capital and its return blank without
/// assets, the rate on line and payback without a limit.
fn column(q: &Quote, with_assets: bool) -> [f64; 13] {
    let nan = f64::NAN;
    [
        q.premium,
        q.expected_loss,
        q.risk_premium,
        q.expenses(),
        q.brokerage(),
        q.net_premium(),
        q.margin(),
        if with_assets { q.capital() } else { nan },
        if with_assets {
            q.return_on_capital()
        } else {
            nan
        },
        q.loss_ratio(),
        q.multiple(),
        q.rate_on_line().unwrap_or(nan),
        q.payback().unwrap_or(nan),
    ]
}

/// `PRO.QUOTE(expected_loss, risk_premium, [assets], [expenses],
/// [brokerage], [fixed], [limit], [quoted])`: the technical premium
/// `(R + f) / (1 - e - b)` of a risk premium `R` (the premium of
/// `PRO.TOWER.PRICE` or another risk-loaded price), its build-up and
/// ratios, and the same cover at each quoted premium: what the quote
/// leaves after the loads is the reinsurer's risk premium, and the margin
/// and return on capital follow.
pub fn quote(args: &[Grid]) -> Result<Grid> {
    let expected = scalar(&args[0], "expected_loss")?;
    let risk = scalar(&args[1], "risk_premium")?;
    let with_assets = !args[2].is_blank();
    let assets = if with_assets {
        scalar(&args[2], "assets")?
    } else {
        f64::INFINITY
    };
    if !(expected.is_finite() && expected >= 0.0 && risk.is_finite() && risk >= expected) {
        return Err(
            "expected_loss must be non-negative and risk_premium at least the expected loss".into(),
        );
    }
    if risk >= assets {
        return Err("assets must exceed the risk premium, or no capital is held".into());
    }
    let e = |e: prospicio_core::Error| e.to_string();
    let loads = Loads::new(
        optional(&args[3], "expenses", 0.0)?,
        optional(&args[4], "brokerage", 0.0)?,
        optional(&args[5], "fixed", 0.0)?,
    )
    .map_err(e)?;
    let limit = if args[6].is_blank() {
        None
    } else {
        Some(scalar(&args[6], "limit")?)
    };
    let price = Price {
        expected_loss: expected,
        premium: risk,
        assets,
    };
    let technical = Quote::technical(&price, loads, limit).map_err(e)?;
    let quoted = if args[7].is_blank() {
        vec![]
    } else {
        numbers(&args[7], "quoted")?
    };
    let mut columns = vec![column(&technical, with_assets)];
    for &p in &quoted {
        columns.push(column(&technical.at(p).map_err(e)?, with_assets));
    }
    let mut header = vec![Value::from("Quantity"), Value::from("Technical")];
    header.extend(quoted.iter().map(|_| Value::from("Quoted")));
    let mut rows = vec![header];
    for (i, label) in ROWS.iter().enumerate() {
        let mut row = vec![Value::from(*label)];
        row.extend(columns.iter().map(|c| Value::number(c[i])));
        rows.push(row);
    }
    Ok(Grid::from_rows(rows))
}
