//! The time value of money (`docs/design/xll.md`, decision 27; the
//! Aggregate lane's `time-value.md`): a payment pattern from shares, the
//! share paid to date or factors to ultimate (a paid chain ladder's
//! `PRO.CL.CDF`), its discount factor, duration and capital years, and a
//! cost-of-capital price restated over the payout.

use prospicio_pricing::payout::{PaymentPattern, price_over_payout};
use prospicio_pricing::risk_load::{PremiumRule, Price};

use crate::pricing::{numbers, scalar};
use crate::reserving::text_option;
use crate::sheet::{Grid, Value};

type Result<T> = std::result::Result<T, String>;

/// The pattern from `times` and `values` read as `kind` (`"shares"`, the
/// default; `"paid"`, the share paid to date; `"factors"`, factors to
/// ultimate), `shift` years later (0 if omitted).
fn pattern(times: &Grid, values: &Grid, kind: &Grid, shift: &Grid) -> Result<PaymentPattern> {
    let t = numbers(times, "times")?;
    let v = numbers(values, "values")?;
    if t.len() != v.len() {
        return Err(format!(
            "{} times and {} values: give one value per time",
            t.len(),
            v.len()
        ));
    }
    let e = |e: prospicio_core::Error| e.to_string();
    let p = match text_option(kind, "kind")?.as_deref() {
        None | Some("shares") => PaymentPattern::new(t, v).map_err(e)?,
        Some("factors") => PaymentPattern::from_development(&t, &v).map_err(e)?,
        Some("paid") => {
            if v.iter().any(|&x| !(x > 0.0 && x <= 1.0)) {
                return Err("paid shares must be in (0, 1], the last 1".into());
            }
            let factors: Vec<f64> = v.iter().map(|x| 1.0 / x).collect();
            PaymentPattern::from_development(&t, &factors).map_err(e)?
        }
        Some(other) => {
            return Err(format!(
                "kind must be \"shares\", \"paid\" or \"factors\", got {other:?}"
            ));
        }
    };
    if shift.is_blank() {
        Ok(p)
    } else {
        p.shifted(scalar(shift, "shift")?).map_err(e)
    }
}

/// An annual effective rate above -1; 0 if omitted when `optional`.
fn rate(grid: &Grid, optional: bool) -> Result<f64> {
    if optional && grid.is_blank() {
        return Ok(0.0);
    }
    let r = scalar(grid, "rate")?;
    if r.is_finite() && r > -1.0 {
        Ok(r)
    } else {
        Err("rate must be an annual effective rate above -1, such as 0.04".into())
    }
}

/// `PRO.PAYOUT.PATTERN(times, values, [kind], [shift], [rate])`: per
/// payment time the share paid then, the share paid to date, the share
/// outstanding and the discount `(1 + rate)^-time` (rate 0 if omitted).
pub fn pattern_table(args: &[Grid]) -> Result<Grid> {
    let p = pattern(&args[0], &args[1], &args[2], &args[3])?;
    let r = rate(&args[4], true)?;
    let mut rows = vec![
        ["Time", "Share", "Paid to date", "Outstanding", "Discount"]
            .map(Value::from)
            .to_vec(),
    ];
    let mut paid = 0.0;
    for (&t, &s) in p.times().iter().zip(p.shares()) {
        paid += s;
        rows.push(
            [t, s, paid.min(1.0), p.outstanding(t), (1.0 + r).powf(-t)]
                .map(Value::number)
                .to_vec(),
        );
    }
    Ok(Grid::from_rows(rows))
}

/// `PRO.PAYOUT.STATS(times, values, rate, [kind], [shift])`: the present
/// value of a unit of ultimate loss, the mean time to payment and the
/// discounted capital years. A distortion price over the payout is its
/// one-period premium times the discount factor.
pub fn stats(args: &[Grid]) -> Result<Grid> {
    let p = pattern(&args[0], &args[1], &args[3], &args[4])?;
    let r = rate(&args[2], false)?;
    let row = |label: &str, x: f64| vec![Value::from(label), Value::number(x)];
    Ok(Grid::from_rows(vec![
        row("Discount factor", p.discount_factor(r)),
        row("Duration", p.duration()),
        row("Capital years", p.capital_years(r)),
    ]))
}

/// `PRO.PAYOUT.PRICE(expected_loss, assets, coc, times, values, rate,
/// [kind], [shift])`: the cost-of-capital price restated over the payout.
/// The premium solves `P = E[X] d + coc (assets - P) A`, with `d` the
/// discount factor and `A` the discounted capital years, so capital is
/// held while losses are outstanding and its cost is a spread over the
/// rate.
pub fn price(args: &[Grid]) -> Result<Grid> {
    let expected = scalar(&args[0], "expected_loss")?;
    let assets = scalar(&args[1], "assets")?;
    let coc = scalar(&args[2], "coc")?;
    let p = pattern(&args[3], &args[4], &args[6], &args[7])?;
    let r = rate(&args[5], false)?;
    let e = |e: prospicio_core::Error| e.to_string();
    let rule = PremiumRule::cost_of_capital(coc).map_err(e)?;
    if !(expected.is_finite() && expected >= 0.0 && assets.is_finite() && assets >= expected) {
        return Err(
            "expected_loss must be non-negative and assets at least the expected loss".into(),
        );
    }
    let one_year = Price {
        expected_loss: expected,
        premium: (expected + coc * assets) / (1.0 + coc),
        assets,
    };
    let q = price_over_payout(&one_year, &rule, &p, r).map_err(e)?;
    let row = |label: &str, x: f64| vec![Value::from(label), Value::number(x)];
    Ok(Grid::from_rows(vec![
        row("Expected loss", q.expected_loss),
        row("Discounted loss", q.discounted_loss),
        row("Premium", q.premium),
        row("Margin", q.margin()),
        row("Capital", q.capital()),
        row("Assets", q.assets),
        row("Discount factor", q.discount_factor),
        row("Capital years", q.capital_years),
        row("One-year premium", one_year.premium),
    ]))
}
