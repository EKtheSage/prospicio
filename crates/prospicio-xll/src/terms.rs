//! Contract terms at a given loss (`docs/design/xll.md`, decision 30):
//! a swing-rated or retrospectively rated premium, a sliding-scale
//! ceding commission and a profit commission, each evaluated at a range
//! of losses as `aggregate`'s `contract_terms` does, over
//! `prospicio_aggregate::Layer`. Their expected values over a layer's
//! annual loss are `PRO.LAYER.TERMS`'s.

use prospicio_aggregate::{Layer, LossSensitivePremium};

use crate::pricing::{map1, numbers, scalar};
use crate::sheet::Grid;

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
