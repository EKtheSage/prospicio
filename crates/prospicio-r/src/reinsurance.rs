//! Reinsurance: wrappers over `prospicio_aggregate::reinsurance` for the R
//! `reinsurance.R` API (layers and towers).

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_aggregate::pnl::ExpenseBasis;
use prospicio_aggregate::{
    Commission, DealScale, Deduction, Layer, LimitAllocation, LossSensitivePremium, Tower,
};

use crate::aggregate::{AnyCount, EventSet, compound_list};
use crate::distributions::{Grid, PredictiveDistribution};
use crate::{to_r, whole};

/// A per-occurrence excess-of-loss layer.
#[extendr]
pub(crate) struct XolLayer {
    pub(crate) inner: Layer,
}

#[extendr]
impl XolLayer {
    /// `aggregate_limit` may be `Inf`; `reinstatements` is a number or
    /// negative for none; `paid` says whether `reinstatement_rates` were
    /// given. At most one of a finite `aggregate_limit`, `reinstatements`
    /// and paid reinstatements; `pro_rata_time` needs paid ones.
    /// `allocation` is `"event_order"` or `"pro_rata"`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        limit: f64,
        attachment: f64,
        share: f64,
        aggregate_deductible: f64,
        aggregate_limit: f64,
        reinstatements: f64,
        premium: f64,
        reinstatement_rates: &[f64],
        paid: bool,
        pro_rata_time: bool,
        allocation: &str,
    ) -> Result<Self> {
        let layer = Layer::xol(name, limit, attachment)
            .and_then(|l| l.share(share))
            .and_then(|l| l.aggregate_deductible(aggregate_deductible))
            .map_err(to_r)?;
        let layer = match (aggregate_limit.is_finite(), reinstatements >= 0.0, paid) {
            (true, false, false) => layer.aggregate_limit(aggregate_limit).map_err(to_r)?,
            (false, true, false) => {
                let n = whole(reinstatements, "reinstatements")?;
                let n = u32::try_from(n)
                    .map_err(|_| Error::Other("reinstatements is too large".into()))?;
                layer.reinstatements(n).map_err(to_r)?
            }
            (false, false, true) => layer
                .paid_reinstatements(premium, reinstatement_rates.to_vec())
                .map_err(to_r)?,
            (false, false, false) => layer,
            _ => {
                return Err(Error::Other(
                    "give at most one of aggregate_limit, reinstatements and reinstatement_rates"
                        .into(),
                ));
            }
        };
        // Without paid reinstatements, which set it themselves.
        let mut layer = layer;
        if !paid {
            if !(premium.is_finite() && premium >= 0.0) {
                return Err(Error::Other(
                    "premium must be finite and non-negative".into(),
                ));
            }
            layer.premium = premium;
        }
        let layer = if pro_rata_time {
            layer.pro_rata_as_to_time().map_err(to_r)?
        } else {
            layer
        };
        let layer = match allocation {
            "event_order" => layer,
            "pro_rata" => layer.allocation(LimitAllocation::ProRata).map_err(to_r)?,
            other => {
                return Err(Error::Other(format!(
                    "allocation must be \"event_order\" or \"pro_rata\", not \"{other}\""
                )));
            }
        };
        Ok(Self { inner: layer })
    }

    fn quota_share(name: &str, cession: f64) -> Result<Self> {
        let inner = Layer::quota_share(name, cession).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn stop_loss(name: &str, limit: f64, retention: f64) -> Result<Self> {
        let inner = Layer::stop_loss(name, limit, retention).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn surplus(name: &str, retention: f64, lines: f64) -> Result<Self> {
        let inner = Layer::surplus(name, retention, lines).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn ceded_with_sums_insured(&self, losses: &[f64], sums_insured: &[f64]) -> Result<f64> {
        if losses.len() != sums_insured.len() {
            return Err(Error::Other("give one sum insured per loss".into()));
        }
        Ok(self.inner.ceded_with_sums_insured(losses, sums_insured))
    }

    fn needs_sums_insured(&self) -> bool {
        self.inner.needs_sums_insured()
    }

    fn sees_aggregate_part(&self) -> bool {
        self.inner.sees_aggregate_part()
    }

    fn name(&self) -> String {
        self.inner.name.clone()
    }

    fn limit(&self) -> f64 {
        self.inner.limit
    }

    fn attachment(&self) -> f64 {
        self.inner.attachment
    }

    fn share(&self) -> f64 {
        self.inner.share
    }

    fn aggregate_deductible(&self) -> f64 {
        self.inner.aggregate_deductible
    }

    fn aggregate_limit(&self) -> f64 {
        self.inner.aggregate_limit
    }

    fn premium(&self) -> f64 {
        self.inner.premium
    }

    fn reinstatement_rates(&self) -> Vec<f64> {
        self.inner.reinstatement_rates.clone()
    }

    fn ceded(&self, losses: &[f64]) -> Result<f64> {
        self.inner.try_ceded(losses).map_err(to_r)
    }

    fn ceded_by_event(&self, losses: &[f64]) -> Result<Vec<f64>> {
        self.inner.try_ceded_by_event(losses).map_err(to_r)
    }

    /// Attachment and exhaustion over an event set's years (#581).
    fn trial_statistics(&self, events: Robj) -> Result<List> {
        let events = <&EventSet>::try_from(&events)
            .map_err(|_| Error::Other("events must be an event_set".into()))?;
        let s = self.inner.trial_statistics(&events.inner).map_err(to_r)?;
        Ok(list!(
            n_years = s.n_years as f64,
            p_attach = s.p_attach,
            p_attach_se = s.p_attach_se,
            p_exhaust_occurrence = s.p_exhaust_occurrence,
            p_exhaust_occurrence_se = s.p_exhaust_occurrence_se,
            p_exhaust_annual = s.p_exhaust_annual,
            p_exhaust_annual_se = s.p_exhaust_annual_se,
            occurrences_mean = s.occurrences_mean,
            occurrences_se = s.occurrences_se,
            ceded_mean = s.ceded_mean,
            ceded_given_attach = s.ceded_given_attach
        ))
    }

    fn pro_rata_time(&self) -> bool {
        self.inner.pro_rata_time
    }

    fn allocation(&self) -> &'static str {
        match self.inner.allocation {
            LimitAllocation::EventOrder => "event_order",
            LimitAllocation::ProRata => "pro_rata",
        }
    }

    fn with_loss_corridor(&self, lower: f64, upper: f64, retained: f64) -> Result<Self> {
        let inner = self
            .inner
            .clone()
            .loss_corridor(lower, upper, retained)
            .map_err(to_r)?;
        Ok(Self { inner })
    }

    /// `c(lower, upper, retained)`, empty when there is none.
    fn loss_corridor(&self) -> Vec<f64> {
        self.inner
            .corridor
            .map(|c| vec![c.lower, c.upper, c.retained])
            .unwrap_or_default()
    }

    fn with_deposit_premium(&self, amount: f64) -> Result<Self> {
        self.map(|l| l.deposit_premium(amount))
    }

    fn with_rate_on_line(&self, rol: f64) -> Result<Self> {
        self.map(|l| l.rate_on_line(rol))
    }

    fn with_premium_rate(&self, rate: f64, subject_premium: f64) -> Result<Self> {
        self.map(|l| l.premium_rate(rate, subject_premium))
    }

    fn with_ceding_commission(&self, rate: f64) -> Result<Self> {
        self.map(|l| l.ceding_commission(rate))
    }

    fn with_sliding_scale(&self, commission: &[f64], loss_ratio: &[f64]) -> Result<Self> {
        if commission.len() != loss_ratio.len() {
            return Err(Error::Other("give one loss ratio per commission".into()));
        }
        let anchors = commission
            .iter()
            .copied()
            .zip(loss_ratio.iter().copied())
            .collect();
        self.map(|l| l.sliding_scale(anchors))
    }

    fn with_profit_commission(&self, share: f64, allowance: f64) -> Result<Self> {
        self.map(|l| l.profit_commission(share, allowance))
    }

    fn with_no_claims_bonus(&self, share: f64, years: f64, threshold: f64) -> Result<Self> {
        let years = whole(years, "years")? as u32;
        self.map(|l| l.no_claims_bonus(share, years, threshold))
    }

    /// `c(share, years, threshold)`, empty when there is none.
    fn no_claims_bonus(&self) -> Vec<f64> {
        self.inner
            .no_claims_bonus
            .map(|b| vec![b.share, f64::from(b.years), b.threshold])
            .unwrap_or_default()
    }

    fn no_claims_bonus_for(&self, ceded: &[f64]) -> Vec<f64> {
        ceded
            .iter()
            .map(|&c| self.inner.no_claims_bonus_for(c))
            .collect()
    }

    /// `years` NaN without limit.
    fn with_profit_commission_carry_forward(&self, years: f64) -> Result<Self> {
        let years = carry_years(years)?;
        self.map(|l| l.profit_commission_carry_forward(years))
    }

    /// `years` NaN without limit.
    fn with_sliding_scale_carry_forward(
        &self,
        deficit: bool,
        credit: bool,
        years: f64,
    ) -> Result<Self> {
        let years = carry_years(years)?;
        self.map(|l| l.sliding_scale_carry_forward(deficit, credit, years))
    }

    /// `c(years)` (NaN without limit), empty when deficits are not carried.
    fn profit_commission_carry(&self) -> Vec<f64> {
        self.inner
            .profit_commission_carry
            .map(|cf| vec![cf.years.map_or(f64::NAN, f64::from)])
            .unwrap_or_default()
    }

    /// `c(deficit, credit, years)` (1 or 0, and NaN without limit), empty
    /// when nothing is carried.
    fn sliding_scale_carry(&self) -> Vec<f64> {
        self.inner
            .sliding_scale_carry
            .map(|cf| {
                vec![
                    f64::from(u8::from(cf.deficit)),
                    f64::from(u8::from(cf.credit)),
                    cf.years.map_or(f64::NAN, f64::from),
                ]
            })
            .unwrap_or_default()
    }

    fn profit_commissions_over(&self, ceded: &[f64]) -> Vec<f64> {
        self.inner.profit_commissions_over(ceded)
    }

    fn ceding_commissions_over(&self, ceded: &[f64]) -> Vec<f64> {
        self.inner.ceding_commissions_over(ceded)
    }

    /// `minimum` and `maximum` NaN for the defaults.
    fn with_swing_rating(&self, basic: f64, lcm: f64, minimum: f64, maximum: f64) -> Result<Self> {
        let terms = loss_sensitive(basic, lcm, minimum, maximum)?;
        self.map(|l| l.swing_rated(terms))
    }

    /// The flat rate, empty when there is none (or the commission slides).
    fn ceding_commission(&self) -> Vec<f64> {
        match self.inner.commission {
            Some(Commission::Flat(c)) => vec![c],
            _ => Vec::new(),
        }
    }

    /// `c(commission..., loss_ratio...)` in increasing loss ratio, empty
    /// when there is none.
    fn sliding_scale(&self) -> Vec<f64> {
        match &self.inner.commission {
            Some(Commission::SlidingScale(a)) => {
                a.iter().map(|p| p.0).chain(a.iter().map(|p| p.1)).collect()
            }
            _ => Vec::new(),
        }
    }

    /// `c(share, allowance)`, empty when there is none.
    fn profit_commission(&self) -> Vec<f64> {
        self.inner
            .profit_commission
            .map(|pc| vec![pc.share, pc.allowance])
            .unwrap_or_default()
    }

    /// `c(basic, lcm, minimum, maximum)` at 100%, empty when there is none.
    fn swing_rating(&self) -> Vec<f64> {
        self.inner
            .swing
            .map(|s| vec![s.basic, s.lcm, s.minimum, s.maximum])
            .unwrap_or_default()
    }

    fn premium_for(&self, ceded: &[f64]) -> Vec<f64> {
        ceded.iter().map(|&c| self.inner.premium_for(c)).collect()
    }

    fn scope(&self) -> Robj {
        self.inner.scope.as_ref().map_or_else(
            || ().into(),
            |s| list!(label = s.label.clone(), values = s.values.clone()).into(),
        )
    }

    fn with_scope(&self, label: &str, values: Vec<String>) -> Result<Self> {
        let values: Vec<&str> = values.iter().map(String::as_str).collect();
        self.map(|l| l.scope(label, &values))
    }

    fn with_franchise(&self) -> Result<Self> {
        self.map(|l| l.franchise())
    }

    fn with_nth_event(&self, n: f64, threshold: f64) -> Result<Self> {
        let n = u32::try_from(whole(n, "n")?).map_err(|_| Error::Other("n is too large".into()))?;
        self.map(|l| l.nth_event(n, threshold))
    }

    fn with_top_and_drop(&self, underlying: &str, attachment: f64) -> Result<Self> {
        self.map(|l| l.top_and_drop(underlying, attachment))
    }

    fn franchise(&self) -> bool {
        self.inner.franchise
    }

    /// `c(n, threshold)`, empty when there is none.
    fn nth_event(&self) -> Vec<f64> {
        self.inner
            .nth_event
            .map(|e| vec![f64::from(e.n), e.threshold])
            .unwrap_or_default()
    }

    /// `list(underlying, attachment)`, or NULL.
    fn drop_down(&self) -> Robj {
        self.inner.drop_down.as_ref().map_or_else(
            || ().into(),
            |d| list!(underlying = d.underlying.clone(), attachment = d.attachment).into(),
        )
    }

    fn ceding_commission_for(&self, ceded: &[f64]) -> Vec<f64> {
        ceded
            .iter()
            .map(|&c| self.inner.ceding_commission_for(c))
            .collect()
    }

    fn with_brokerage(&self, rate: f64) -> Result<Self> {
        self.map(|l| l.brokerage(rate))
    }

    fn with_brokerage_scale(&self, rate: &[f64], loss_ratio: &[f64]) -> Result<Self> {
        if rate.len() != loss_ratio.len() {
            return Err(Error::Other("give one loss ratio per rate".into()));
        }
        let anchors = rate
            .iter()
            .copied()
            .zip(loss_ratio.iter().copied())
            .collect();
        self.map(|l| l.brokerage_scale(anchors))
    }

    fn with_deduction(
        &self,
        label: &str,
        value: f64,
        basis: &str,
        inside_nmbf: bool,
    ) -> Result<Self> {
        let basis = match basis {
            "premium" => ExpenseBasis::Premium,
            "loss" => ExpenseBasis::Loss,
            "fixed" => ExpenseBasis::Fixed,
            other => {
                return Err(Error::Other(format!(
                    "a deduction's basis is premium, loss or fixed, not {other:?}"
                )));
            }
        };
        let mut d = Deduction::new(label, basis, value).map_err(to_r)?;
        if !inside_nmbf {
            d = d.outside_nmbf();
        }
        self.map(|l| l.deduction(d))
    }

    fn with_scales_before_annual_terms(&self) -> Self {
        Self {
            inner: self.inner.clone().scales_before_annual_terms(),
        }
    }

    fn with_brokerage_within_commission_range(&self) -> Self {
        Self {
            inner: self.inner.clone().brokerage_within_commission_range(),
        }
    }

    /// The flat rate, empty when there is none (or it is on a scale).
    fn brokerage(&self) -> Vec<f64> {
        match self.inner.brokerage {
            Some(Commission::Flat(b)) => vec![b],
            _ => Vec::new(),
        }
    }

    /// `c(rate..., loss_ratio...)` in increasing loss ratio, empty when
    /// there is none.
    fn brokerage_scale(&self) -> Vec<f64> {
        match &self.inner.brokerage {
            Some(Commission::SlidingScale(a)) => {
                a.iter().map(|p| p.0).chain(a.iter().map(|p| p.1)).collect()
            }
            _ => Vec::new(),
        }
    }

    /// `list(label, basis, value, inside_nmbf)`, one entry per deduction.
    fn deductions(&self) -> List {
        let d = &self.inner.deductions;
        list!(
            label = d.iter().map(|d| d.label.clone()).collect::<Vec<_>>(),
            basis = d.iter().map(|d| d.basis.name()).collect::<Vec<_>>(),
            value = d.iter().map(|d| d.value).collect::<Vec<_>>(),
            inside_nmbf = d.iter().map(|d| d.inside_nmbf).collect::<Vec<_>>()
        )
    }

    fn scales_before_annual_terms(&self) -> bool {
        self.inner.scales_before_annual_terms
    }

    fn brokerage_within_commission_range(&self) -> bool {
        self.inner.brokerage_within_commission_range
    }

    fn brokerage_for(&self, ceded: &[f64]) -> Vec<f64> {
        ceded.iter().map(|&c| self.inner.brokerage_for(c)).collect()
    }

    fn deduction_for(&self, label: &str, ceded: &[f64]) -> Result<Vec<f64>> {
        let d = self
            .inner
            .deductions
            .iter()
            .find(|d| d.label == label)
            .ok_or_else(|| {
                Error::Other(format!(
                    "layer {:?} has no deduction labelled {label:?}",
                    self.inner.name
                ))
            })?;
        Ok(ceded
            .iter()
            .map(|&c| self.inner.deduction_for(d, c))
            .collect())
    }

    fn profit_commission_for(&self, ceded: &[f64]) -> Vec<f64> {
        ceded
            .iter()
            .map(|&c| self.inner.profit_commission_for(c))
            .collect()
    }

    /// `times` empty when not given.
    fn reinstatement_premium(&self, losses: &[f64], times: &[f64]) -> Result<f64> {
        if times.is_empty() {
            return Ok(self.inner.reinstatement_premium(losses));
        }
        if times.len() != losses.len() {
            return Err(Error::Other("give one time per loss".into()));
        }
        Ok(self.inner.reinstatement_premium_dated(losses, times))
    }
}

/// Layers in inuring stages.
#[extendr]
pub(crate) struct ReinsuranceTower {
    pub(crate) inner: Tower,
}

#[extendr]
impl ReinsuranceTower {
    fn new(layers: List) -> Result<Self> {
        let inner = Tower::new(layer_list(layers)?).map_err(to_r)?;
        Ok(Self { inner })
    }

    /// `stages` is a list of lists of layers.
    fn inuring(stages: List) -> Result<Self> {
        let stages = stages
            .values()
            .map(|stage| {
                List::try_from(&stage)
                    .map_err(|_| Error::Other("stages must be a list of lists of layers".into()))
                    .and_then(layer_list)
            })
            .collect::<Result<_>>()?;
        let inner = Tower::inuring(stages).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn from_json(text: &str) -> Result<Self> {
        let inner = Tower::from_json(text).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    fn layer_names(&self) -> Vec<String> {
        self.inner.layers.iter().map(|l| l.name.clone()).collect()
    }

    /// 1-based stage of each layer.
    fn stages(&self) -> Vec<f64> {
        self.inner.stages.iter().map(|&s| s as f64 + 1.0).collect()
    }

    fn deal_names(&self) -> Vec<String> {
        self.inner.deals.iter().map(|d| d.name.clone()).collect()
    }

    /// A deal's scales on several layers together. Empty vectors leave a
    /// term out; `brokerage` and `carry_years` have at most one value.
    #[allow(clippy::too_many_arguments)]
    fn with_deal_scale(
        &self,
        name: &str,
        sections: Vec<String>,
        commission: &[f64],
        loss_ratio: &[f64],
        brokerage: &[f64],
        brokerage_rate: &[f64],
        brokerage_loss_ratio: &[f64],
        carry_deficit: bool,
        carry_credit: bool,
        carry_years: &[f64],
        brokerage_within_commission_range: bool,
    ) -> Result<Self> {
        let pairs = |a: &[f64], b: &[f64], what: &str| -> Result<Vec<(f64, f64)>> {
            if a.len() != b.len() {
                return Err(Error::Other(format!("give one loss ratio per {what}")));
            }
            Ok(a.iter().copied().zip(b.iter().copied()).collect())
        };
        if brokerage.len() > 1 || carry_years.len() > 1 {
            return Err(Error::Other(
                "brokerage and carry_years take one value".into(),
            ));
        }
        if !brokerage.is_empty() && !brokerage_rate.is_empty() {
            return Err(Error::Other(
                "give brokerage or a brokerage scale, not both".into(),
            ));
        }
        let mut deal = DealScale::new(name, sections).map_err(to_r)?;
        if !commission.is_empty() {
            deal = deal
                .sliding_scale(pairs(commission, loss_ratio, "commission")?)
                .map_err(to_r)?;
        }
        let years = carry_years
            .first()
            .map(|&y| {
                if y.is_finite() && y >= 1.0 && y.fract() == 0.0 {
                    Ok(y as u32)
                } else {
                    Err(Error::Other(
                        "carry_years must be a whole number from 1".into(),
                    ))
                }
            })
            .transpose()?;
        if carry_deficit || carry_credit || years.is_some() {
            deal = deal
                .carry_forward(carry_deficit, carry_credit, years)
                .map_err(to_r)?;
        }
        if let Some(&rate) = brokerage.first() {
            deal = deal.brokerage(rate).map_err(to_r)?;
        }
        if !brokerage_rate.is_empty() {
            deal = deal
                .brokerage_scale(pairs(
                    brokerage_rate,
                    brokerage_loss_ratio,
                    "brokerage rate",
                )?)
                .map_err(to_r)?;
        }
        if brokerage_within_commission_range {
            deal = deal.brokerage_within_commission_range();
        }
        let inner = self.inner.clone().deal_scale(deal).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn ceded(&self, losses: &[f64]) -> Result<Vec<f64>> {
        self.inner.try_ceded(losses).map_err(to_r)
    }

    /// Gross, ceded and net annual distributions on the grid, by FFT:
    /// `list(gross, ceded, net, expected_reinstatement_premium,
    /// expected_premium, expected_ceding_commission,
    /// expected_profit_commission, expected_no_claims_bonus,
    /// expected_deal_ceding_commission,
    /// expected_deal_brokerage, deal_names, on_points)`, with `stage_ceded` one
    /// grid (or NULL) per stage,
    /// with `net` and `total_ceded` NULL when they are not one compound total.
    fn on_grid(&self, frequency: Robj, severity: Robj, points: f64) -> Result<List> {
        let n = AnyCount::from_robj(&frequency)?;
        let sev = <&Grid>::try_from(&severity)
            .map_err(|_| Error::Other("severity must be a grid_distribution".into()))?;
        let points = whole(points, "points")? as usize;
        let r = self
            .inner
            .on_grid(n.as_counting(), &sev.inner, points)
            .map_err(to_r)?;
        let ceded: Vec<Robj> = r
            .ceded
            .into_iter()
            .zip(&r.ceded_reports)
            .map(|(g, report)| Grid::with_report(g, compound_list(report)).into())
            .collect();
        let net: Robj = match r.net {
            Some(g) => Grid::wrap(g).into(),
            None => ().into(),
        };
        let total_ceded: Robj = match r.total_ceded {
            Some(g) => Grid::wrap(g).into(),
            None => ().into(),
        };
        let mut stage_sizes: Vec<f64> = Vec::new();
        for (i, &s) in self.inner.stages.iter().enumerate() {
            if i == 0 || s != self.inner.stages[i - 1] {
                stage_sizes.push(0.0);
            }
            *stage_sizes.last_mut().expect("pushed above") += 1.0;
        }
        let stage_ceded = List::from_values(r.stage_ceded.into_iter().map(|g| -> Robj {
            match g {
                Some(g) => Grid::wrap(g).into(),
                None => ().into(),
            }
        }));
        Ok(list!(
            gross = Grid::with_report(r.gross, compound_list(&r.gross_report)),
            ceded = List::from_values(ceded),
            stage_ceded = stage_ceded,
            stage_sizes = stage_sizes,
            net = net,
            total_ceded = total_ceded,
            expected_reinstatement_premium = r.expected_reinstatement_premium,
            expected_premium = r.expected_premium,
            expected_ceding_commission = r.expected_ceding_commission,
            expected_profit_commission = r.expected_profit_commission,
            expected_no_claims_bonus = r.expected_no_claims_bonus,
            expected_deal_ceding_commission = r.expected_deal_ceding_commission,
            expected_deal_brokerage = r.expected_deal_brokerage,
            deal_names = self
                .inner
                .deals
                .iter()
                .map(|d| d.name.clone())
                .collect::<Vec<_>>(),
            on_points = r.on_points
        ))
    }

    /// The tower on an annual loss grid, each year's total as one loss:
    /// `list(gross, ceded, net, expected_reinstatement_premium,
    /// expected_premium, expected_ceding_commission,
    /// expected_profit_commission, expected_no_claims_bonus,
    /// expected_deal_ceding_commission,
    /// expected_deal_brokerage, deal_names, on_points)`.
    fn on_annual_grid(&self, annual: Robj) -> Result<List> {
        let annual = <&Grid>::try_from(&annual)
            .map_err(|_| Error::Other("annual must be a grid_distribution".into()))?;
        let r = self.inner.on_annual_grid(&annual.inner).map_err(to_r)?;
        let ceded: Vec<Robj> = r.ceded.into_iter().map(|g| Grid::wrap(g).into()).collect();
        Ok(list!(
            gross = Grid::wrap(r.gross),
            ceded = List::from_values(ceded),
            net = Grid::wrap(r.net),
            expected_reinstatement_premium = r.expected_reinstatement_premium,
            expected_premium = r.expected_premium,
            expected_ceding_commission = r.expected_ceding_commission,
            expected_profit_commission = r.expected_profit_commission,
            expected_no_claims_bonus = r.expected_no_claims_bonus,
            expected_deal_ceding_commission = r.expected_deal_ceding_commission,
            expected_deal_brokerage = r.expected_deal_brokerage,
            deal_names = self
                .inner
                .deals
                .iter()
                .map(|d| d.name.clone())
                .collect::<Vec<_>>(),
            on_points = r.on_points
        ))
    }

    /// Each simulation's total of a predictive distribution as one
    /// aggregate loss.
    fn apply_aggregate(&self, losses: Robj) -> Result<PredictiveDistribution> {
        let pd = <&PredictiveDistribution>::try_from(&losses)
            .map_err(|_| Error::Other("losses must be a predictive_distribution".into()))?;
        let inner = self.inner.apply_aggregate(&pd.inner).map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }

    /// The deal ratios of a layer or a deal in a result of this tower.
    fn deal_ratios(&self, result: Robj, layer: &str) -> Result<PredictiveDistribution> {
        let pd = <&PredictiveDistribution>::try_from(&result)
            .map_err(|_| Error::Other("result must be a predictive_distribution".into()))?;
        let inner = self.inner.deal_ratios(&pd.inner, layer).map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }

    /// Plain annual losses, one per simulation, each as one aggregate loss.
    fn apply_annual(&self, losses: &[f64]) -> Result<PredictiveDistribution> {
        let inner = self.inner.apply_annual(losses).map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }

    fn apply(&self, events: Robj) -> Result<PredictiveDistribution> {
        let events = <&EventSet>::try_from(&events)
            .map_err(|_| Error::Other("events must be an event_set".into()))?;
        let inner = self.inner.apply(&events.inner).map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }

    /// Each event's ceded loss to the named layers, as an event set.
    fn ceded_events(&self, events: Robj, layers: Vec<String>) -> Result<EventSet> {
        let events = <&EventSet>::try_from(&events)
            .map_err(|_| Error::Other("events must be an event_set".into()))?;
        let names: Vec<&str> = layers.iter().map(String::as_str).collect();
        let inner = self
            .inner
            .ceded_events(&events.inner, &names)
            .map_err(to_r)?;
        Ok(EventSet { inner })
    }

    /// The tower with its cover basis.
    fn with_cover_basis(&self, basis: &str) -> Result<Self> {
        let basis = crate::aggregate::cover_basis(basis)?;
        Ok(Self {
            inner: self.inner.clone().cover_basis(basis),
        })
    }

    /// The tower with its contracts interlocking or not.
    fn with_interlocking(&self, on: bool) -> Self {
        Self {
            inner: self.inner.clone().interlocking(on),
        }
    }

    /// Whether the tower's contracts interlock.
    fn interlocking(&self) -> bool {
        self.inner.interlocking
    }

    /// The cover basis's name, or `NULL`.
    fn cover_basis(&self) -> Robj {
        use prospicio_aggregate::timeline::CoverBasis;
        match self.inner.cover_basis {
            None => ().into(),
            Some(CoverBasis::LossesOccurring) => "losses_occurring".into(),
            Some(CoverBasis::RisksAttaching) => "risks_attaching".into(),
        }
    }

    /// The term on `timeline`, with the exposure's shares row-major over
    /// `n_years` rows.
    fn apply_timeline(
        &self,
        timeline: Robj,
        shares: &[f64],
        n_years: f64,
    ) -> Result<PredictiveDistribution> {
        let timeline = <&crate::aggregate::Timeline>::try_from(&timeline)
            .map_err(|_| Error::Other("timeline must be a timeline".into()))?;
        let exposure = crate::aggregate::exposure(shares, n_years)?;
        let inner = self
            .inner
            .apply_timeline(&timeline.inner, &exposure)
            .map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }

    /// `years` a list of event sets, one per year of the term.
    fn apply_term(&self, years: List) -> Result<PredictiveDistribution> {
        let years = years
            .values()
            .map(|y| {
                <&EventSet>::try_from(&y)
                    .map(|e| e.inner.clone())
                    .map_err(|_| Error::Other("each year must be an event_set".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let inner = self.inner.apply_term(&years).map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }

    /// `years` a list of predictive distributions, one per year.
    fn apply_aggregate_term(&self, years: List) -> Result<PredictiveDistribution> {
        let years = years
            .values()
            .map(|y| {
                <&PredictiveDistribution>::try_from(&y)
                    .map(|p| p.inner.clone())
                    .map_err(|_| Error::Other("each year must be a predictive_distribution".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let inner = self.inner.apply_aggregate_term(&years).map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

/// A carry's `years` from R: NaN without limit, else a positive whole
/// number.
fn carry_years(years: f64) -> Result<Option<u32>> {
    if years.is_nan() {
        return Ok(None);
    }
    if !(years.fract() == 0.0 && years >= 1.0 && years <= f64::from(u32::MAX)) {
        return Err(Error::Other(
            "years must be a positive whole number or NULL".into(),
        ));
    }
    Ok(Some(years as u32))
}

impl XolLayer {
    /// A copy of the layer with one builder applied.
    fn map(&self, f: impl FnOnce(Layer) -> prospicio_core::Result<Layer>) -> Result<Self> {
        let inner = f(self.inner.clone()).map_err(to_r)?;
        Ok(Self { inner })
    }
}

/// Loss-sensitive premium terms, NaN `minimum` or `maximum` for the
/// defaults.
fn loss_sensitive(
    basic: f64,
    lcm: f64,
    minimum: f64,
    maximum: f64,
) -> Result<LossSensitivePremium> {
    let given = |x: f64| (!x.is_nan()).then_some(x);
    LossSensitivePremium::new(basic, lcm, given(minimum), given(maximum)).map_err(to_r)
}

/// Retrospectively rated premium for each loss: `clip(basic + lcm * x,
/// minimum, maximum)`.
#[extendr]
fn retro_premium_rust(
    losses: &[f64],
    basic: f64,
    lcm: f64,
    minimum: f64,
    maximum: f64,
) -> Result<Vec<f64>> {
    let terms = loss_sensitive(basic, lcm, minimum, maximum)?;
    Ok(losses.iter().map(|&x| terms.premium(x)).collect())
}

fn layer_list(layers: List) -> Result<Vec<Layer>> {
    layers
        .values()
        .map(|l| {
            <&XolLayer>::try_from(&l)
                .map(|l| l.inner.clone())
                .map_err(|_| Error::Other("layers must all be xol_layer objects".into()))
        })
        .collect()
}

fn moments_vec(m: &prospicio_aggregate::CompoundMoments) -> Vec<f64> {
    vec![m.mean, m.variance, m.third_cumulant, m.cv(), m.skew()]
}

/// `list(pr_attach, pr_detach, pr_loss, loss_on_line, count, severity,
/// aggregate)`, the moments each `c(mean, variance, third_cumulant, cv,
/// skew)`.
#[extendr]
fn occurrence_layer_statistics_rust(
    frequency: Robj,
    severity: Robj,
    share: f64,
    limit: f64,
    attachment: f64,
) -> Result<List> {
    let n = AnyCount::from_robj(&frequency)?;
    let sev = <&Grid>::try_from(&severity)
        .map_err(|_| Error::Other("severity must be a grid_distribution".into()))?;
    let s = prospicio_aggregate::occurrence_layer_statistics(
        n.as_counting(),
        &sev.inner,
        share,
        limit,
        attachment,
    )
    .map_err(to_r)?;
    Ok(list!(
        pr_attach = s.pr_attach,
        pr_detach = s.pr_detach,
        pr_loss = s.pr_loss,
        loss_on_line = s.loss_on_line,
        count = moments_vec(&s.count),
        severity = moments_vec(&s.severity),
        aggregate = moments_vec(&s.aggregate)
    ))
}

/// `list(pr_attach, pr_loss, loss_on_line, severity, aggregate)` of the
/// layers' total cession.
#[extendr]
fn occurrence_cession_statistics_rust(
    frequency: Robj,
    severity: Robj,
    shares: &[f64],
    limits: &[f64],
    attachments: &[f64],
) -> Result<List> {
    let n = AnyCount::from_robj(&frequency)?;
    let sev = <&Grid>::try_from(&severity)
        .map_err(|_| Error::Other("severity must be a grid_distribution".into()))?;
    if shares.len() != limits.len() || limits.len() != attachments.len() {
        return Err(Error::Other(
            "give one share, limit and attachment per layer".into(),
        ));
    }
    let layers: Vec<(f64, f64, f64)> = (0..limits.len())
        .map(|i| (shares[i], limits[i], attachments[i]))
        .collect();
    let s =
        prospicio_aggregate::occurrence_cession_statistics(n.as_counting(), &sev.inner, &layers)
            .map_err(to_r)?;
    Ok(list!(
        pr_attach = s.pr_attach,
        pr_loss = s.pr_loss,
        loss_on_line = s.loss_on_line,
        severity = moments_vec(&s.severity),
        aggregate = moments_vec(&s.aggregate)
    ))
}

/// `list(pr_attach, pr_detach, loss_on_line, aggregate)` of an aggregate
/// cover on the annual grid.
#[extendr]
fn aggregate_layer_statistics_rust(
    annual: Robj,
    share: f64,
    limit: f64,
    attachment: f64,
) -> Result<List> {
    let annual = <&Grid>::try_from(&annual)
        .map_err(|_| Error::Other("annual must be a grid_distribution".into()))?;
    let s =
        prospicio_aggregate::aggregate_layer_statistics(&annual.inner, share, limit, attachment)
            .map_err(to_r)?;
    Ok(list!(
        pr_attach = s.pr_attach,
        pr_detach = s.pr_detach,
        loss_on_line = s.loss_on_line,
        aggregate = moments_vec(&s.aggregate)
    ))
}

/// A list of numeric vectors as vectors.
fn numeric_rows(l: &List, what: &str) -> Result<Vec<Vec<f64>>> {
    l.values()
        .map(|v| {
            v.as_real_vector()
                .or_else(|| {
                    v.as_integer_vector()
                        .map(|x| x.into_iter().map(f64::from).collect())
                })
                .or_else(|| v.is_null().then(Vec::new))
                .ok_or_else(|| Error::Other(format!("{what} must hold numeric vectors")))
        })
        .collect()
}

/// A per-risk layer with an event limit inuring to a cat cover:
/// `list(ceded, ceded_by_event, net)`. `years` is a list (years) of lists
/// (events) of numeric vectors (risk losses); `event_limit` NaN for none;
/// `times` NULL or a list of numeric vectors, one time an event.
#[extendr]
fn per_risk_inuring_rust(
    layer: Robj,
    years: List,
    event_limit: f64,
    times: Robj,
    seed: f64,
) -> Result<List> {
    let layer = <&XolLayer>::try_from(&layer)
        .map_err(|_| Error::Other("layer must be an xol_layer".into()))?;
    let ys: Vec<Vec<Vec<f64>>> = years
        .values()
        .map(|y| {
            let events = List::try_from(&y)
                .map_err(|_| Error::Other("each year must be a list of events".into()))?;
            numeric_rows(&events, "each event")
        })
        .collect::<Result<_>>()?;
    let t = if times.is_null() {
        None
    } else {
        let l = List::try_from(&times)
            .map_err(|_| Error::Other("times must be a list of numeric vectors".into()))?;
        Some(numeric_rows(&l, "times")?)
    };
    let limit = (!event_limit.is_nan()).then_some(event_limit);
    let r = prospicio_aggregate::per_risk_inuring(
        &layer.inner,
        limit,
        &ys,
        t.as_deref(),
        whole(seed, "seed")?,
    )
    .map_err(to_r)?;
    Ok(list!(
        ceded = r.ceded,
        ceded_by_event = List::from_values(r.ceded_by_event),
        net = EventSet { inner: r.net }
    ))
}

extendr_module! {
    mod reinsurance;
    fn per_risk_inuring_rust;
    fn occurrence_cession_statistics_rust;
    fn aggregate_layer_statistics_rust;
    fn retro_premium_rust;
    fn occurrence_layer_statistics_rust;
    impl XolLayer;
    impl ReinsuranceTower;
}
