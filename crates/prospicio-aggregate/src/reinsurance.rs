//! Reinsurance: excess-of-loss layers, quota shares, stop-losses and
//! surplus treaties, and towers of them applied to simulated events,
//! giving gross, ceded and net distributions.

use prospicio_core::{Error, Result};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};

use crate::deal::{DealScale, Deduction};
use crate::monte_carlo::EventSet;
use crate::timeline::CoverBasis;

/// One year's losses for a layer: each finite and non-negative. A negative
/// loss is usually one stored with the sign of money paid out.
fn check_losses(losses: &[f64]) -> Result<()> {
    if let Some((k, &x)) = losses
        .iter()
        .enumerate()
        .find(|(_, x)| !(x.is_finite() && **x >= 0.0))
    {
        return Err(Error::Data(format!(
            "loss {} is {x}: losses must be finite and non-negative (a file that stores \
             losses as negative numbers is read with its sign flipped)",
            k + 1
        )));
    }
    Ok(())
}

mod layer;
mod terms;
mod tower;

pub use layer::*;
pub use terms::*;
pub use tower::*;

/// Event `e`'s limit factor: 1 without interlocking.
fn factor(lf: Option<&[f64]>, e: usize) -> f64 {
    lf.map_or(1.0, |f| f[e])
}

/// Year `t`'s amount on a scale over `premium` read at the loss `key`,
/// carrying deficits and credits in `carry` as `cf` says (a flat rate or
/// no `cf` reads the year alone).
pub(crate) fn scale_in(
    scale: &Commission,
    premium: f64,
    key: f64,
    t: usize,
    carry: &mut Carried,
    cf: Option<&CarryForward>,
) -> f64 {
    let (Commission::SlidingScale(anchors), Some(cf)) = (scale, cf) else {
        return scale.rate(key / premium) * premium;
    };
    let (lr_min, lr_max) = (anchors[0].1, anchors[anchors.len() - 1].1);
    let cf = *cf;
    let total = carry.step(key, t, cf.years, |x| {
        let mut out = 0.0;
        if cf.deficit {
            out += (x - lr_max * premium).max(0.0);
        }
        if cf.credit {
            out += (x - lr_min * premium).min(0.0);
        }
        out
    });
    scale.rate(total / premium) * premium
}

fn carry_text(cf: &CarryForward) -> String {
    match cf.years {
        Some(n) => format!("up to {n} years"),
        None => "to extinction".into(),
    }
}

/// Which layers get a component for each term that varies with the year's
/// ceded loss, in the order a year's components are written.
pub(crate) struct Columns {
    paid: Vec<bool>,
    swing: Vec<usize>,
    commission: Vec<usize>,
    profit: Vec<usize>,
    bonus: Vec<usize>,
    brokerage: Vec<usize>,
    deductions: Vec<usize>,
    keyed: Vec<usize>,
    /// Each deal's sections, by index.
    deals: Vec<Vec<usize>>,
}

/// Whether a deal reads a key other than its sections' ceded loss: some
/// section's scales read the loss before annual terms.
fn deal_keyed(tower: &Tower, sections: &[usize]) -> bool {
    sections
        .iter()
        .any(|&i| tower.layers[i].scales_before_annual_terms)
}

impl Columns {
    pub(crate) fn new(tower: &Tower) -> Self {
        Self {
            paid: tower
                .layers
                .iter()
                .map(|l| !l.reinstatement_rates.is_empty())
                .collect(),
            swing: tower.indices(|l| l.swing.is_some()),
            commission: tower.indices(|l| l.commission.is_some()),
            profit: tower.indices(|l| l.profit_commission.is_some()),
            bonus: tower.indices(|l| l.no_claims_bonus.is_some()),
            brokerage: tower.indices(|l| l.brokerage.is_some()),
            deductions: tower.indices(|l| !l.deductions.is_empty()),
            keyed: tower.indices(|l| l.scales_before_annual_terms),
            deals: tower.deal_sections(),
        }
    }

    /// Components a year has.
    pub(crate) fn len(&self, tower: &Tower) -> usize {
        tower.layers.len()
            + 2
            + self.paid.iter().filter(|&&p| p).count()
            + self.swing.len()
            + self.commission.len()
            + self.profit.len()
            + self.bonus.len()
            + self.brokerage.len()
            + self.keyed.len()
            + self
                .deals
                .iter()
                .zip(&tower.deals)
                .map(|(sections, d)| {
                    usize::from(d.commission.is_some())
                        + usize::from(d.brokerage.is_some())
                        + usize::from(deal_keyed(tower, sections))
                })
                .sum::<usize>()
            + self
                .deductions
                .iter()
                .map(|&i| tower.layers[i].deductions.len())
                .sum::<usize>()
    }

    /// The `(kind, layer)` key of each of a year's components.
    pub(crate) fn keys(&self, tower: &Tower) -> Vec<ComponentKey> {
        let s = tower;
        let key = |kind: &str, layer: &str| -> ComponentKey {
            vec![KeyValue::from(kind), KeyValue::from(layer)]
        };
        let mut components = vec![key("gross", "ground_up")];
        components.extend(s.layers.iter().map(|l| key("ceded", &l.name)));
        components.push(key("net", "retained"));
        components.extend(
            s.layers
                .iter()
                .filter(|l| !l.reinstatement_rates.is_empty())
                .map(|l| key("reinstatement_premium", &l.name)),
        );
        for (kind, layers) in [
            ("swing_premium", &self.swing),
            ("ceding_commission", &self.commission),
            ("profit_commission", &self.profit),
            ("no_claims_bonus", &self.bonus),
            ("brokerage", &self.brokerage),
        ] {
            components.extend(layers.iter().map(|&i| key(kind, &s.layers[i].name)));
        }
        for &i in &self.deductions {
            let l = &s.layers[i];
            components.extend(l.deductions.iter().map(|d| key(&d.label, &l.name)));
        }
        components.extend(
            self.keyed
                .iter()
                .map(|&i| key("loss_before_terms", &s.layers[i].name)),
        );
        for (d, sections) in s.deals.iter().zip(&self.deals) {
            if d.commission.is_some() {
                components.push(key("ceding_commission", &d.name));
            }
            if d.brokerage.is_some() {
                components.push(key("brokerage", &d.name));
            }
            if deal_keyed(s, sections) {
                components.push(key("loss_before_terms", &d.name));
            }
        }
        components
    }
}

/// `(rate, loss_ratio)` anchors checked and sorted by loss ratio: rates in
/// `[0, 1]`, loss ratios finite, non-negative and distinct, and with
/// `falling` rates that do not rise with the loss ratio.
pub(crate) fn checked_anchors(
    mut anchors: Vec<(f64, f64)>,
    falling: bool,
) -> Result<Vec<(f64, f64)>> {
    if anchors.is_empty() {
        return Err(invalid("anchors", 0.0, "must not be empty"));
    }
    for &(c, lr) in &anchors {
        if !(c.is_finite() && (0.0..=1.0).contains(&c)) {
            return Err(invalid("commission", c, "must be in [0, 1]"));
        }
        if !(lr.is_finite() && lr >= 0.0) {
            return Err(invalid("loss_ratio", lr, "must be finite and non-negative"));
        }
    }
    anchors.sort_by(|a, b| a.1.total_cmp(&b.1));
    for w in anchors.windows(2) {
        if w[1].1 == w[0].1 {
            return Err(invalid("loss_ratio", w[1].1, "must not repeat"));
        }
        if falling && w[1].0 > w[0].0 {
            return Err(invalid(
                "commission",
                w[1].0,
                "must not rise with the loss ratio",
            ));
        }
    }
    Ok(anchors)
}

fn check_carry_years(years: Option<u32>) -> Result<()> {
    if years == Some(0) {
        return Err(invalid(
            "years",
            0.0,
            "must be at least 1 (or None, without limit)",
        ));
    }
    Ok(())
}

/// What a profit commission or sliding scale has carried so far in one
/// simulated term: amounts by the year they arose in, oldest first, all of
/// one sign.
#[derive(Debug, Clone, Default)]
pub(crate) struct Carried {
    amounts: Vec<(usize, f64)>,
}

impl Carried {
    /// Records year `t`'s `value` and returns the sum over the last `years`
    /// years recorded, `t` included (a no-claims bonus's window, #592).
    pub(crate) fn window(&mut self, value: f64, t: usize, years: u32) -> f64 {
        self.amounts.retain(|&(o, _)| t - o < years as usize);
        self.amounts.push((t, value));
        self.amounts.iter().map(|&(_, a)| a).sum()
    }

    /// Year `t`: drops amounts older than `years`, adds what is left to the
    /// year's own `own` amount, and keeps `beyond(total)`, the part still
    /// outside the band, attributed to the newest contributions of its
    /// sign (older ones are used up first). Returns the total the year's
    /// commission reads.
    pub(crate) fn step(
        &mut self,
        own: f64,
        t: usize,
        years: Option<u32>,
        beyond: impl Fn(f64) -> f64,
    ) -> f64 {
        if let Some(n) = years {
            self.amounts.retain(|&(o, _)| t - o <= n as usize);
        }
        let total = own + self.amounts.iter().map(|&(_, a)| a).sum::<f64>();
        let mut rest = beyond(total);
        let mut items = std::mem::take(&mut self.amounts);
        items.push((t, own));
        let mut kept = Vec::new();
        for &(o, a) in items.iter().rev() {
            if rest == 0.0 {
                break;
            }
            if a * rest > 0.0 {
                let take = a.signum() * a.abs().min(rest.abs());
                kept.push((o, take));
                rest -= take;
            }
        }
        // Round-off: anything not matched stays with this year.
        if rest != 0.0 {
            kept.push((t, rest));
        }
        kept.reverse();
        self.amounts = kept;
        total
    }
}

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

#[cfg(test)]
mod tests;
