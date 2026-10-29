//! Cash flows and returns on a sheet (`docs/design/xll.md`, decision 42):
//! the net present value, internal rate of return and multiple on
//! invested capital of flows in cells (`prospicio_pricing::returns`), and
//! a collateralised deal's monthly cash flows over simulated loss ratios
//! (`prospicio_pricing::cashflow`, the Aggregate lane's `time-value.md`,
//! decisions 13 to 20).
//!
//! A deal's terms are a block of rows, a label in the first cell and its
//! values in the cells after it, the labels the Python keyword names
//! (`premium`, `commutation_month`, `collected_rate`, ...). A requirement
//! is a block of `base | load` rows, with `floor | x` and `limit | trigger
//! | cap` rows beside them.

use prospicio_pricing::cashflow::{
    Adjustment, AdjustmentBase, Base, CashflowTerms, Dates, FeeBasis, Funding, FundingRule,
    FundingSource, Interest, InterestBasis, MonthlyPatterns, ObligationReference, PerformanceFee,
    Requirement, Term, Timing, Trace,
};
use prospicio_pricing::returns::{self, Irr, IrrOptions, TieredFee, TimeUnit};

use crate::pricing::numbers;
use crate::reserving::{number_or, text_option};
use crate::sheet::{Grid, Value, position};

type Result<T> = std::result::Result<T, String>;

fn err(e: prospicio_core::Error) -> String {
    e.to_string()
}

/// The most simulations `PRO.CASHFLOW` runs.
const MAX_SIMS: usize = 1_000_000;

/// A label or an option's text: lower case, with `_`, `-` and spaces one
/// separator, so `"Commutation month"`, `"commutation-month"` and
/// `"commutation_month"` are the same.
fn key(s: &str, separator: char) -> String {
    s.trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| {
            if matches!(c, '_' | '-' | ' ') {
                separator
            } else {
                c
            }
        })
        .collect()
}

/// The `unit` argument: what the times are.
enum Unit {
    Offsets(TimeUnit),
    /// Excel dates, read from the first as days over 365 (`XNPV`, `XIRR`).
    Dates,
}

fn unit(grid: &Grid) -> Result<Unit> {
    Ok(match text_option(grid, "unit")?.as_deref() {
        None | Some("years") => Unit::Offsets(TimeUnit::Years),
        Some("months") => Unit::Offsets(TimeUnit::Months),
        Some("days") => Unit::Offsets(TimeUnit::Days),
        Some("dates") => Unit::Dates,
        Some(other) => {
            return Err(format!(
                "unit must be \"years\", \"months\", \"days\" or \"dates\", got {other:?}"
            ));
        }
    })
}

/// The `flows`, `times` and `unit` arguments: the flows as rows of one
/// simulation each, and the times in years. A row or column of flows is
/// one simulation; a block of several rows and columns is one simulation
/// a row. Without `times` the flows are at 0, 1, 2, ... in `unit`.
fn flows_and_times(
    flows: &Grid,
    times: &Grid,
    unit_arg: &Grid,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if flows.is_blank() {
        return Err("flows is required".into());
    }
    let unit = unit(unit_arg)?;
    let cell = |r: usize, c: usize| match flows.get(r, c) {
        Value::Number(x) => Ok(*x),
        v if v.is_blank() => Ok(0.0),
        other => Err(format!(
            "flows {} must be a number, got {other:?}",
            position(r, c)
        )),
    };
    let rows: Vec<Vec<f64>> = if flows.rows == 1 || flows.columns == 1 {
        vec![
            (0..flows.rows)
                .flat_map(|r| (0..flows.columns).map(move |c| (r, c)))
                .map(|(r, c)| cell(r, c))
                .collect::<Result<_>>()?,
        ]
    } else {
        (0..flows.rows)
            .map(|r| (0..flows.columns).map(|c| cell(r, c)).collect())
            .collect::<Result<_>>()?
    };
    let n = rows[0].len();
    let offsets = if times.is_blank() {
        if matches!(unit, Unit::Dates) {
            return Err("unit \"dates\" needs times, the flows' dates".into());
        }
        (0..n).map(|t| t as f64).collect()
    } else {
        let t = numbers(times, "times")?;
        if t.len() != n {
            return Err(format!(
                "{} times for {n} flows a simulation: give one time per flow",
                t.len()
            ));
        }
        t
    };
    let years = match unit {
        Unit::Offsets(u) => u.years(&offsets),
        Unit::Dates => offsets.iter().map(|d| (d - offsets[0]) / 365.0).collect(),
    };
    Ok((rows, years))
}

/// One value, or a column of one per simulation.
fn column_or_scalar(values: Vec<Value>) -> Grid {
    if values.len() == 1 {
        Grid::scalar(values.into_iter().next().expect("one value"))
    } else {
        Grid::from_rows(values.into_iter().map(|v| vec![v]).collect())
    }
}

/// `PRO.NPV(rate, flows, [times], [unit])`: `Σ x_k (1 + rate)^-t_k`, one
/// value, or one per row of a block of simulations.
pub fn npv(args: &[Grid]) -> Result<Grid> {
    let rate = crate::reserving::number(&args[0], "rate")?;
    let (rows, years) = flows_and_times(&args[1], &args[2], &args[3])?;
    let values = rows
        .iter()
        .map(|row| returns::npv(row, &years, rate).map(Value::number))
        .collect::<prospicio_core::Result<Vec<_>>>()
        .map_err(err)?;
    Ok(column_or_scalar(values))
}

/// The IRR search's options: `lower` and `upper` (−0.95 and 1 a year if
/// omitted) and `extend` (search past them, `FALSE` if omitted).
fn irr_options(lower: &Grid, upper: &Grid, extend: &Grid) -> Result<IrrOptions> {
    let d = IrrOptions::default();
    Ok(IrrOptions {
        lower: number_or(lower, "lower", d.lower)?,
        upper: number_or(upper, "upper", d.upper)?,
        extend: flag(extend, "extend", false)?,
        ..d
    })
}

/// An optional `TRUE`/`FALSE` argument (1 and 0 too).
fn flag(grid: &Grid, name: &str, default: bool) -> Result<bool> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(default),
        Some(v) => boolean(v).ok_or_else(|| format!("{name} must be TRUE or FALSE")),
        None => Err(format!("{name} must be TRUE or FALSE")),
    }
}

fn boolean(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::Number(x) if *x == 0.0 || *x == 1.0 => Some(*x == 1.0),
        Value::Text(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" => Some(true),
            "false" | "no" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Why no rate was found, for one simulation's error text.
fn no_root(r: &Irr, o: &IrrOptions) -> String {
    let range = format!("from {}% to {}% a year", o.lower * 100.0, o.upper * 100.0);
    if r.root_above {
        format!("no rate of return {range}; one lies above it (set extend TRUE, or a higher upper)")
    } else if r.root_below {
        format!("no rate of return {range}; one lies below it (set extend TRUE, or a lower lower)")
    } else {
        "no rate of return: the flows' value never crosses 0 (flows of one sign?)".into()
    }
}

const IRR_HEADER: [&str; 6] = ["Rate", "Found", "NPV", "Roots", "Root above", "Root below"];

fn irr_row(r: &Irr) -> Vec<Value> {
    vec![
        Value::number(r.rate),
        Value::Bool(r.found),
        Value::number(r.npv),
        Value::Number(r.sign_changes as f64),
        Value::Bool(r.root_above),
        Value::Bool(r.root_below),
    ]
}

/// `PRO.IRR(flows, [times], [unit], [lower], [upper], [extend],
/// [detail])`: the annual rate at which the flows' value is 0, by the
/// crate's grid search. One series without a root is an error saying
/// why; in a block of simulations such a row is blank. With `detail`
/// TRUE, a table: the rate, whether it was found, the value there, the
/// roots seen and whether one lies above or below the range.
pub fn irr(args: &[Grid]) -> Result<Grid> {
    let (rows, years) = flows_and_times(&args[0], &args[1], &args[2])?;
    let options = irr_options(&args[3], &args[4], &args[5])?;
    let detail = flag(&args[6], "detail", false)?;
    let found = rows
        .iter()
        .map(|row| returns::irr(row, &years, &options))
        .collect::<prospicio_core::Result<Vec<_>>>()
        .map_err(err)?;
    if detail {
        let mut out = vec![IRR_HEADER.iter().map(|h| Value::from(*h)).collect()];
        out.extend(found.iter().map(irr_row));
        return Ok(Grid::from_rows(out));
    }
    if let [r] = found.as_slice()
        && !r.found
    {
        return Err(no_root(r, &options));
    }
    Ok(column_or_scalar(
        found.iter().map(|r| Value::number(r.rate)).collect(),
    ))
}

/// `PRO.MOIC(released, funded, [opening])`: `1 + released / (funded +
/// opening)`, each argument one value or a column (or row) of them, a
/// single value applying to every row.
pub fn moic(args: &[Grid]) -> Result<Grid> {
    let released = numbers(&args[0], "released")?;
    let funded = numbers(&args[1], "funded")?;
    let opening = if args[2].is_blank() {
        vec![0.0]
    } else {
        numbers(&args[2], "opening")?
    };
    let n = released.len().max(funded.len()).max(opening.len());
    for (name, v) in [
        ("released", &released),
        ("funded", &funded),
        ("opening", &opening),
    ] {
        if v.len() != 1 && v.len() != n {
            return Err(format!(
                "{name} has {} values and another argument {n}: give one value or {n}",
                v.len()
            ));
        }
    }
    let at = |v: &[f64], i: usize| if v.len() == 1 { v[0] } else { v[i] };
    let values = (0..n)
        .map(|i| {
            returns::moic(at(&released, i), at(&funded, i), at(&opening, i)).map(Value::number)
        })
        .collect::<prospicio_core::Result<Vec<_>>>()
        .map_err(err)?;
    Ok(column_or_scalar(values))
}

/// A block of label rows: each row's label (normalized by [`key`]) and the
/// values after it, blank cells dropped. Blank rows are skipped.
fn label_rows(grid: &Grid, what: &str) -> Result<Vec<(String, usize, Vec<Value>)>> {
    let mut out = Vec::new();
    for r in 0..grid.rows {
        let values: Vec<Value> = (1..grid.columns)
            .map(|c| grid.get(r, c).clone())
            .filter(|v| !v.is_blank())
            .collect();
        match grid.get(r, 0) {
            Value::Text(s) if !s.trim().is_empty() => out.push((key(s, '_'), r, values)),
            v if v.is_blank() && values.is_empty() => {}
            _ => {
                return Err(format!(
                    "{what} {} must be a label (text), with its values after it",
                    position(r, 0)
                ));
            }
        }
    }
    Ok(out)
}

/// One row of a label block, read by type.
struct Row<'a> {
    what: &'a str,
    label: &'a str,
    r: usize,
    values: &'a [Value],
}

impl Row<'_> {
    fn place(&self) -> String {
        format!("{} {} ({})", self.what, self.label, position(self.r, 0))
    }

    fn one(&self) -> Result<&Value> {
        match self.values {
            [v] => Ok(v),
            [] => Err(format!("{} has no value", self.place())),
            _ => Err(format!(
                "{} takes one value, got {}",
                self.place(),
                self.values.len()
            )),
        }
    }

    fn number(&self) -> Result<f64> {
        match self.one()? {
            Value::Number(x) => Ok(*x),
            _ => Err(format!("{} must be a number", self.place())),
        }
    }

    fn numbers(&self) -> Result<Vec<f64>> {
        if self.values.is_empty() {
            return Err(format!("{} has no values", self.place()));
        }
        self.values
            .iter()
            .map(|v| match v {
                Value::Number(x) => Ok(*x),
                _ => Err(format!("{}'s values must be numbers", self.place())),
            })
            .collect()
    }

    fn whole(&self) -> Result<usize> {
        let x = self.number()?;
        if x.fract() == 0.0 && (0.0..1e9).contains(&x) {
            Ok(x as usize)
        } else {
            Err(format!("{} must be a whole number from 0", self.place()))
        }
    }

    fn flag(&self) -> Result<bool> {
        boolean(self.one()?).ok_or_else(|| format!("{} must be TRUE or FALSE", self.place()))
    }

    fn text(&self) -> Result<String> {
        match self.one()? {
            Value::Text(s) => Ok(key(s, '-')),
            _ => Err(format!("{} must be text", self.place())),
        }
    }

    /// A set of months: `monthly`, `quarterly`, `n` (every `n` months from
    /// month 0), `n | first`, or `months | m1 | m2 | ...`; with
    /// `at_commutation` when the row is a release timing.
    fn timing(&self, release: bool) -> Result<Timing> {
        let whole = |v: &Value| match v {
            Value::Number(x) if x.fract() == 0.0 && (0.0..1e9).contains(x) => Some(*x as usize),
            _ => None,
        };
        let bad = || {
            format!(
                "{} must be \"monthly\", \"quarterly\",{} an interval n, n and the first month, \
                 or \"months\" and the months",
                self.place(),
                if release { " \"at-commutation\"," } else { "" }
            )
        };
        let dates = match self.values {
            [Value::Text(s)] => match key(s, '-').as_str() {
                "monthly" => return Ok(Timing::Monthly),
                "quarterly" => Dates::quarterly(),
                "at-commutation" if release => return Ok(Timing::AtCommutation),
                _ => return Err(bad()),
            },
            [Value::Text(s), months @ ..] if key(s, '-') == "months" && !months.is_empty() => {
                Dates::Months(
                    months
                        .iter()
                        .map(whole)
                        .collect::<Option<_>>()
                        .ok_or_else(bad)?,
                )
            }
            [n] => Dates::Every {
                months: whole(n).ok_or_else(bad)?,
                first: 0,
            },
            [n, first] => Dates::Every {
                months: whole(n).ok_or_else(bad)?,
                first: whole(first).ok_or_else(bad)?,
            },
            _ => return Err(bad()),
        };
        Ok(if release || dates != Dates::monthly() {
            Timing::At(dates)
        } else {
            Timing::Monthly
        })
    }
}

fn base(name: &str, place: &str) -> Result<Base> {
    Ok(match key(name, '-').as_str() {
        "unearned-premium" => Base::UnearnedPremium,
        "reserve" => Base::Reserve,
        "outstanding" => Base::Outstanding,
        "written" => Base::Written,
        "earned" => Base::Earned,
        "fixed" => Base::Fixed,
        other => {
            return Err(format!(
                "{place}: unknown base {other:?}; use unearned-premium, reserve, outstanding, \
                 written, earned or fixed, or a floor or limit row"
            ));
        }
    })
}

/// A requirement block: `base | load` rows, a `floor | x` row and a
/// `limit | trigger | cap` row.
fn requirement(grid: &Grid, what: &str) -> Result<Requirement> {
    let rows = label_rows(grid, what)?;
    if rows.is_empty() {
        return Err(format!("{what} is required: rows of a base and its load"));
    }
    let mut terms = Vec::new();
    let mut floor = 0.0;
    let mut limit = None;
    for (label, r, values) in &rows {
        let row = Row {
            what,
            label,
            r: *r,
            values,
        };
        match label.as_str() {
            "floor" => floor = row.number()?,
            "limit" => match row.numbers()?.as_slice() {
                [trigger, cap] => limit = Some((*trigger, *cap)),
                _ => {
                    return Err(format!(
                        "{} takes two values, a trigger and a cap",
                        row.place()
                    ));
                }
            },
            name => terms.push(Term::new(base(name, &row.place())?, row.number()?)),
        }
    }
    if terms.is_empty() {
        return Err(format!("{what} has no base and load rows"));
    }
    let mut out = Requirement::new(terms).with_floor(floor);
    if let Some((trigger, cap)) = limit {
        out = out.with_limit(trigger, cap);
    }
    Ok(out)
}

/// The patterns argument: three columns, the cumulative shares written,
/// earned and paid at the end of each month from 0, with an optional
/// header row of text.
fn patterns(grid: &Grid) -> Result<MonthlyPatterns> {
    if grid.columns != 3 {
        return Err(format!(
            "patterns must be three columns, written, earned and paid, got {}",
            grid.columns
        ));
    }
    let first = usize::from(matches!(grid.get(0, 0), Value::Text(_)));
    let last = (first..grid.rows)
        .rev()
        .find(|&r| (0..3).any(|c| !grid.get(r, c).is_blank()))
        .ok_or("patterns has no rows of values")?;
    let mut columns = vec![Vec::new(), Vec::new(), Vec::new()];
    for r in first..=last {
        for (c, column) in columns.iter_mut().enumerate() {
            match grid.get(r, c) {
                Value::Number(x) => column.push(*x),
                _ => {
                    return Err(format!(
                        "patterns {} must be a number (cumulative share at month {})",
                        position(r, c),
                        r - first
                    ));
                }
            }
        }
    }
    let [written, earned, paid]: [Vec<f64>; 3] = columns.try_into().expect("three columns");
    MonthlyPatterns::new(written, earned, paid).map_err(err)
}

/// The labels a terms block takes.
const LABELS: &str = "premium, commutation_month, collected_rate, deduct_commission, lag, \
adjustment, funding_dates, funding_rule, premium_offset, reference, funding_source, \
collateral_release, annual_yield, interest_basis, opening_fixed, opening_share, expense_rate, \
buffer_months, buffer_paid, buffer_expenses, equity_release, fee_breakpoints, fee_slopes, \
fee_basis, irr_lower, irr_upper, irr_extend";

/// The deal: the terms block, the patterns and the requirements; and the
/// IRR search's options, from the block's `irr_*` rows.
fn deal(
    terms_arg: &Grid,
    patterns_arg: &Grid,
    requirement_arg: &Grid,
    commutation_arg: &Grid,
) -> Result<(CashflowTerms, IrrOptions)> {
    let what = "terms";
    let rows = label_rows(terms_arg, what)?;
    let row = |label: &str| -> Result<Option<Row<'_>>> {
        let mut found = rows.iter().filter(|(l, ..)| l == label);
        let first = found.next();
        if found.next().is_some() {
            return Err(format!("{what} has two {label} rows"));
        }
        Ok(first.map(|(label, r, values)| Row {
            what,
            label,
            r: *r,
            values,
        }))
    };
    if let Some((label, r, _)) = rows
        .iter()
        .find(|(l, ..)| !LABELS.split(", ").any(|k| k == l))
    {
        return Err(format!(
            "{what} {}: unknown label {label:?}; the labels are {LABELS}",
            position(*r, 0)
        ));
    }
    let premium = row("premium")?
        .ok_or("terms needs a premium row")?
        .number()?;
    let commutation_month = row("commutation_month")?
        .ok_or("terms needs a commutation_month row: the month count T")?
        .whole()?;
    let req = requirement(requirement_arg, "requirement")?;
    let mut t = CashflowTerms::new(premium, commutation_month, patterns(patterns_arg)?, req)
        .map_err(err)?;
    if !commutation_arg.is_blank() {
        t.commutation_requirement = requirement(commutation_arg, "commutation_requirement")?;
    }

    let number = |label: &str, default: f64| -> Result<f64> {
        row(label)?.map_or(Ok(default), |r| r.number())
    };
    let whole = |label: &str| -> Result<usize> { row(label)?.map_or(Ok(0), |r| r.whole()) };
    let flag = |label: &str| -> Result<bool> { row(label)?.map_or(Ok(false), |r| r.flag()) };
    let text = |label: &str, default: &str| -> Result<String> {
        row(label)?.map_or(Ok(default.into()), |r| r.text())
    };
    let choice = |label: &str, value: String, allowed: &str| -> String {
        format!("terms {label}: unknown value {value:?}; use {allowed}")
    };

    t.premium_flow.collected_rate = number("collected_rate", 1.0)?;
    t.premium_flow.deduct_commission = flag("deduct_commission")?;
    t.premium_flow.lag = whole("lag")?;
    for (label, r, values) in rows.iter().filter(|(l, ..)| l == "adjustment") {
        let row = Row {
            what,
            label,
            r: *r,
            values,
        };
        let bad = || {
            format!(
                "{} takes a month, a rate on written premium, and optionally TRUE to take the \
                 commission off and \"ultimate\" or \"to-date\"",
                row.place()
            )
        };
        let (month, written_rate) = match values.get(..2) {
            Some([Value::Number(m), Value::Number(w)]) if m.fract() == 0.0 && *m >= 0.0 => {
                (*m as usize, *w)
            }
            _ => return Err(bad()),
        };
        let less_commission = match values.get(2) {
            None => false,
            Some(v) => boolean(v).ok_or_else(bad)?,
        };
        let base = match values.get(3) {
            None => AdjustmentBase::Ultimate,
            Some(Value::Text(s)) => match key(s, '-').as_str() {
                "ultimate" => AdjustmentBase::Ultimate,
                "to-date" => AdjustmentBase::ToDate,
                _ => return Err(bad()),
            },
            Some(_) => return Err(bad()),
        };
        if values.len() > 4 {
            return Err(bad());
        }
        t.premium_flow.adjustments.push(Adjustment {
            month,
            written_rate,
            less_commission,
            base,
        });
    }

    let rule = match text("funding_rule", "shortfall")?.as_str() {
        "shortfall" => FundingRule::Shortfall,
        "obligation-increase" => FundingRule::ObligationIncrease {
            premium_offset: number("premium_offset", 1.0)?,
            reference: match text("reference", "previous-date")?.as_str() {
                "previous-date" => ObligationReference::PreviousDate,
                "running-max" => ObligationReference::RunningMax,
                other => {
                    return Err(choice(
                        "reference",
                        other.into(),
                        "previous-date or running-max",
                    ));
                }
            },
        },
        other => {
            return Err(choice(
                "funding_rule",
                other.into(),
                "shortfall or obligation-increase",
            ));
        }
    };
    let dates = match row("funding_dates")? {
        None => Dates::monthly(),
        Some(r) => match r.timing(false)? {
            Timing::At(d) => d,
            _ => Dates::monthly(),
        },
    };
    t.funding = Funding {
        dates,
        rule,
        source: match text("funding_source", "investor")?.as_str() {
            "investor" => FundingSource::Investor,
            "operating-first" => FundingSource::OperatingFirst,
            other => {
                return Err(choice(
                    "funding_source",
                    other.into(),
                    "investor or operating-first",
                ));
            }
        },
    };
    if let Some(r) = row("collateral_release")? {
        t.collateral_release = r.timing(true)?;
    }
    t.interest = Interest {
        annual_yield: number("annual_yield", 0.0)?,
        basis: match text("interest_basis", "average")?.as_str() {
            "average" => InterestBasis::Average,
            "opening" => InterestBasis::Opening,
            "closing" => InterestBasis::Closing,
            other => {
                return Err(choice(
                    "interest_basis",
                    other.into(),
                    "average, opening or closing",
                ));
            }
        },
    };
    t.operating.opening_fixed = number("opening_fixed", 0.0)?;
    t.operating.opening_share = number("opening_share", 0.0)?;
    t.operating.expense_rate = number("expense_rate", 0.0)?;
    t.operating.buffer.months = whole("buffer_months")?;
    t.operating.buffer.paid = flag("buffer_paid")?;
    t.operating.buffer.expenses = flag("buffer_expenses")?;
    if let Some(r) = row("equity_release")? {
        t.operating.release = r.timing(true)?;
    }
    t.operating.fee = match (row("fee_breakpoints")?, row("fee_slopes")?) {
        (None, None) => {
            if let Some(r) = row("fee_basis")? {
                return Err(format!(
                    "{} needs fee_breakpoints and fee_slopes rows",
                    r.place()
                ));
            }
            None
        }
        (Some(b), Some(s)) => Some(PerformanceFee {
            tiers: TieredFee::new(b.numbers()?, s.numbers()?).map_err(err)?,
            basis: match text("fee_basis", "profit")?.as_str() {
                "profit" => FeeBasis::Profit,
                "profit-over-premium" => FeeBasis::ProfitOverPremium,
                "profit-over-capital" => FeeBasis::ProfitOverCapital,
                other => {
                    return Err(choice(
                        "fee_basis",
                        other.into(),
                        "profit, profit-over-premium or profit-over-capital",
                    ));
                }
            },
        }),
        _ => return Err("terms needs both fee_breakpoints and fee_slopes rows, or neither".into()),
    };
    let d = IrrOptions::default();
    let options = IrrOptions {
        lower: number("irr_lower", d.lower)?,
        upper: number("irr_upper", d.upper)?,
        extend: flag("irr_extend")?,
        ..d
    };
    // A bad rule is reported now, as the Python constructor does.
    t.trace(0.0, 0.0).map_err(err)?;
    Ok((t, options))
}

/// Each simulation's commission rate: omitted (0), one rate for all, or
/// one per simulation.
fn commissions(grid: &Grid, n: usize) -> Result<Vec<f64>> {
    if grid.is_blank() {
        return Ok(vec![0.0; n]);
    }
    let c = numbers(grid, "commission")?;
    match c.len() {
        1 => Ok(vec![c[0]; n]),
        m if m == n => Ok(c),
        m => Err(format!(
            "commission has {m} rates for {n} loss ratios: give one rate or one per loss ratio"
        )),
    }
}

const SIMULATION_HEADER: [&str; 10] = [
    "Loss ratio",
    "IRR",
    "MOIC",
    "Funded",
    "Opening",
    "Released",
    "Distributed",
    "Fee",
    "Commutation",
    "Paid first",
];

const TRACE_HEADER: [&str; 15] = [
    "Month",
    "Requirement",
    "Premium",
    "Adjustment",
    "Paid",
    "Funding (investor)",
    "Funding (operating)",
    "Interest",
    "Collateral release",
    "Collateral balance",
    "Expenses",
    "Equity release",
    "Operating balance",
    "Investor",
    "Funding reference",
];

fn trace_table(t: &Trace) -> Grid {
    let columns = [
        &t.requirement,
        &t.premium,
        &t.adjustment,
        &t.paid,
        &t.funding_investor,
        &t.funding_operating,
        &t.interest,
        &t.collateral_release,
        &t.collateral_balance,
        &t.expenses,
        &t.equity_release,
        &t.operating_balance,
        &t.investor,
        &t.funding_reference,
    ];
    let mut rows = vec![
        TRACE_HEADER
            .iter()
            .map(|h| Value::from(*h))
            .collect::<Vec<_>>(),
    ];
    for m in 0..t.investor.len() {
        let mut row = vec![Value::Number(m as f64)];
        row.extend(columns.iter().map(|c| Value::number(c[m])));
        rows.push(row);
    }
    let width = TRACE_HEADER.len();
    let label = |name: &str, x: f64| {
        let mut row = vec![Value::from(name), Value::number(x)];
        row.resize(width, Value::Empty);
        row
    };
    rows.push(vec![Value::Empty; width]);
    rows.push(label("Commutation paid", t.commutation_paid));
    rows.push(label("Fee", t.fee));
    rows.push(label("Conservation error", t.conservation_error()));
    Grid::from_rows(rows)
}

/// `PRO.CASHFLOW(u, terms, patterns, requirement, [commission], [output],
/// [rate], [commutation_requirement])`: the deal run on each simulated
/// ultimate loss ratio in `u`. `output` is `"simulations"` (the default:
/// one row a simulation with its IRR, MOIC and the totals, and its NPV
/// at `rate` when given), `"flows"` (the investor's net flow in each
/// month, one row a simulation) or `"trace"` (every account line of the
/// first simulation, month by month).
pub fn cashflow(args: &[Grid]) -> Result<Grid> {
    let u = numbers(&args[0], "u")?;
    if u.len() > MAX_SIMS {
        return Err(format!("u has more than {MAX_SIMS} loss ratios"));
    }
    let (terms, options) = deal(&args[1], &args[2], &args[3], &args[7])?;
    let commission = commissions(&args[4], u.len())?;
    let output = text_option(&args[5], "output")?;
    let rate = if args[6].is_blank() {
        None
    } else {
        Some(crate::reserving::number(&args[6], "rate")?)
    };
    match output.as_deref() {
        Some("trace") => {
            return Ok(trace_table(&terms.trace(u[0], commission[0]).map_err(err)?));
        }
        None | Some("simulations") | Some("flows") => {}
        Some(other) => {
            return Err(format!(
                "output must be \"simulations\", \"flows\" or \"trace\", got {other:?}"
            ));
        }
    }
    let out = terms.run(&u, &commission).map_err(err)?;
    let months = out.months();
    if output.as_deref() == Some("flows") {
        let mut rows = vec![
            (0..months)
                .map(|m| Value::from(format!("Month {m}").as_str()))
                .collect::<Vec<_>>(),
        ];
        rows.extend(
            out.flows()
                .chunks(months)
                .map(|row| row.iter().map(|x| Value::number(*x)).collect()),
        );
        return Ok(Grid::from_rows(rows));
    }
    let irr = out.irr(&options).map_err(err)?;
    let npv = rate.map(|r| out.npv(r)).transpose().map_err(err)?;
    let paid_first = out.paid_first();
    let mut header: Vec<Value> = SIMULATION_HEADER.iter().map(|h| Value::from(*h)).collect();
    if let Some(r) = rate {
        header.insert(2, Value::from(format!("NPV at {}%", r * 100.0).as_str()));
    }
    let mut rows = vec![header];
    for (j, flows) in out.flows().chunks(months).enumerate() {
        let moic = returns::moic(flows.iter().sum(), out.funded()[j], out.opening()[j])
            .map_or(Value::Empty, Value::number);
        let mut row = vec![Value::Number(u[j]), Value::number(irr[j].rate), moic];
        if let Some(v) = &npv {
            row.insert(2, Value::number(v[j]));
        }
        row.extend(
            [
                out.funded()[j],
                out.opening()[j],
                out.released()[j],
                out.distributed()[j],
                out.fee()[j],
                out.commutation()[j],
            ]
            .map(Value::number),
        );
        row.push(Value::Bool(paid_first[j]));
        rows.push(row);
    }
    Ok(Grid::from_rows(rows))
}
