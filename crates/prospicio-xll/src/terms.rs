//! Contract terms at a given loss (`docs/design/xll.md`, decision 30):
//! a swing-rated or retrospectively rated premium, a sliding-scale
//! ceding commission and a profit commission, each evaluated at a range
//! of losses as `aggregate`'s `contract_terms` does, over
//! `prospicio_aggregate::Layer`. Their expected values over a layer's
//! annual loss are `PRO.LAYER.TERMS`'s. Over a term of several years,
//! with deficits and credits carried from year to year, they are
//! `PRO.PROFIT.COMMISSION.TERM`'s and `PRO.SLIDING.SCALE.TERM`'s (decision
//! 33).

use prospicio_aggregate::{Layer, LossSensitivePremium};

use crate::pricing::{map1, numbers, scalar};
use crate::reserving::text_option;
use crate::sheet::{Grid, Value, position};

type Result<T> = std::result::Result<T, String>;

fn e(e: prospicio_core::Error) -> String {
    e.to_string()
}

fn optional(grid: &Grid, name: &str) -> Result<Option<f64>> {
    if grid.is_blank() {
        Ok(None)
    } else {
        scalar(grid, name).map(Some)
    }
}

/// A sliding scale's `(commission, loss_ratio)` anchors from two ranges
/// of equal length, in increasing loss ratio.
pub(crate) fn anchors(commissions: &Grid, loss_ratios: &Grid) -> Result<Vec<(f64, f64)>> {
    let c = numbers(commissions, "commissions")?;
    let lr = numbers(loss_ratios, "loss_ratios")?;
    if c.len() != lr.len() {
        return Err(format!(
            "{} commissions and {} loss_ratios: give one loss ratio per commission",
            c.len(),
            lr.len()
        ));
    }
    Ok(c.into_iter().zip(lr).collect())
}

/// `PRO.SWING(loss, basic, lcm, [minimum], [maximum], [share])`: the
/// loss-sensitive premium `clip(basic + lcm × loss, minimum, maximum)` at
/// each loss, a swing-rated layer's (the loss its ceded loss at the placed
/// `share`, the amounts at 100%) or a retro plan's. The minimum defaults to
/// `basic`, the maximum to no cap.
pub fn swing(args: &[Grid]) -> Result<Grid> {
    let terms = LossSensitivePremium::new(
        scalar(&args[1], "basic")?,
        scalar(&args[2], "lcm")?,
        optional(&args[3], "minimum")?,
        optional(&args[4], "maximum")?,
    )
    .map_err(e)?;
    let share = optional(&args[5], "share")?.unwrap_or(1.0);
    let layer = Layer::xol("swing", f64::INFINITY, 0.0)
        .and_then(|l| l.share(share))
        .and_then(|l| l.swing_rated(terms))
        .map_err(e)?;
    map1(&args[0], "loss", false, |x| Ok(layer.premium_for(x)))
}

/// A layer with `premium` (at 100%), for the commissions.
fn priced(premium: &Grid) -> Result<Layer> {
    let p = scalar(premium, "premium")?;
    if !(p > 0.0 && p.is_finite()) {
        return Err(format!("premium must be positive, got {p}"));
    }
    Layer::quota_share("terms", 1.0)
        .and_then(|l| l.deposit_premium(p))
        .map_err(e)
}

/// `PRO.SLIDING.SCALE(loss, premium, commissions, loss_ratios)`: the
/// sliding-scale ceding commission, an amount, at each ceded loss: the
/// commission rate at the loss ratio `loss / premium`, linear between the
/// anchors and flat beyond the first and last, times the premium.
pub fn sliding_scale(args: &[Grid]) -> Result<Grid> {
    let layer = priced(&args[1])?
        .sliding_scale(anchors(&args[2], &args[3])?)
        .map_err(e)?;
    map1(&args[0], "loss", false, |x| {
        Ok(layer.ceding_commission_for(x))
    })
}

/// `PRO.PROFIT.COMMISSION(loss, premium, share, allowance)`: the profit
/// commission at each ceded loss, `share × max(premium × (1 − allowance)
/// − loss, 0)`.
pub fn profit_commission(args: &[Grid]) -> Result<Grid> {
    let layer = priced(&args[1])?
        .profit_commission(scalar(&args[2], "share")?, scalar(&args[3], "allowance")?)
        .map_err(e)?;
    map1(&args[0], "loss", false, |x| {
        Ok(layer.profit_commission_for(x))
    })
}

/// The `carry_years` argument: a whole number of later years a carried
/// amount may be used in, or blank to carry it until it is used up.
fn carry_years(grid: &Grid) -> Result<Option<u32>> {
    match grid.as_scalar() {
        Some(v) if v.is_blank() => Ok(None),
        Some(Value::Number(x)) if x.fract() == 0.0 && *x >= 1.0 && *x <= 1000.0 => {
            Ok(Some(*x as u32))
        }
        _ => {
            Err("carry_years must be a whole number from 1, or blank to carry until used up".into())
        }
    }
}

/// Each row of `losses` a term, its columns the years in order, mapped by
/// `over` to one value per year.
fn by_term(losses: &Grid, over: impl Fn(&[f64]) -> Vec<f64>) -> Result<Grid> {
    let mut rows = Vec::with_capacity(losses.rows);
    for i in 0..losses.rows {
        let ceded = (0..losses.columns)
            .map(|j| match losses.get(i, j) {
                Value::Number(x) if x.is_finite() && *x >= 0.0 => Ok(*x),
                _ => Err(format!(
                    "losses must be non-negative numbers, one row a term and one column a year; {} is not",
                    position(i, j)
                )),
            })
            .collect::<Result<Vec<f64>>>()?;
        rows.push(over(&ceded).into_iter().map(Value::number).collect());
    }
    Ok(Grid::from_rows(rows))
}

/// `PRO.PROFIT.COMMISSION.TERM(losses, premium, share, allowance,
/// [carry_years])`: the profit commission in each year of a term, deficits
/// carried forward: with `M = premium × (1 − allowance)` and `D` carried
/// in, a year pays `share × max(M − loss − D, 0)` and carries
/// `max(loss + D − M, 0)`, used oldest first and dropped after
/// `carry_years` later years.
pub fn profit_commission_term(args: &[Grid]) -> Result<Grid> {
    let years = carry_years(&args[4])?;
    let layer = priced(&args[1])?
        .profit_commission(scalar(&args[2], "share")?, scalar(&args[3], "allowance")?)
        .and_then(|l| l.profit_commission_carry_forward(years))
        .map_err(e)?;
    by_term(&args[0], |ceded| layer.profit_commissions_over(ceded))
}

/// `PRO.SLIDING.SCALE.TERM(losses, premium, commissions, loss_ratios,
/// [carry], [carry_years])`: the sliding-scale ceding commission in each
/// year of a term, a year reading the scale at `(loss + carried) /
/// premium`. A deficit is the loss above the last anchor's loss ratio, a
/// credit the loss below the first's; `carry` is "deficit", "credit" or
/// "both" (the default).
pub fn sliding_scale_term(args: &[Grid]) -> Result<Grid> {
    let (deficit, credit) = match text_option(&args[4], "carry")?.as_deref() {
        None | Some("both") => (true, true),
        Some("deficit" | "deficits") => (true, false),
        Some("credit" | "credits") => (false, true),
        Some(other) => {
            return Err(format!(
                "carry must be \"deficit\", \"credit\" or \"both\", got {other:?}"
            ));
        }
    };
    let layer = priced(&args[1])?
        .sliding_scale(anchors(&args[2], &args[3])?)
        .map_err(e)?;
    let layer = layer
        .sliding_scale_carry_forward(deficit, credit, carry_years(&args[5])?)
        .map_err(e)?;
    by_term(&args[0], |ceded| layer.ceding_commissions_over(ceded))
}
