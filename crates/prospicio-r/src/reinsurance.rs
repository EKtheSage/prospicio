//! Reinsurance: wrappers over `prospicio_aggregate::reinsurance` for the R
//! `reinsurance.R` API (layers and towers).

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_aggregate::{Commission, Layer, LimitAllocation, LossSensitivePremium, Tower};

use crate::aggregate::{AnyCount, EventSet, compound_list};
use crate::distributions::{Grid, PredictiveDistribution};
use crate::{to_r, whole};

/// A per-occurrence excess-of-loss layer.
#[extendr]
pub(crate) struct XolLayer {
    inner: Layer,
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

    fn ceded(&self, losses: &[f64]) -> f64 {
        self.inner.ceded(losses)
    }

    fn ceded_by_event(&self, losses: &[f64]) -> Vec<f64> {
        self.inner.ceded_by_event(losses)
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

    fn ceding_commission_for(&self, ceded: &[f64]) -> Vec<f64> {
        ceded
            .iter()
            .map(|&c| self.inner.ceding_commission_for(c))
            .collect()
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

    fn ceded(&self, losses: &[f64]) -> Vec<f64> {
        self.inner.ceded(losses)
    }

    /// Gross, ceded and net annual distributions on the grid, by FFT:
    /// `list(gross, ceded, net, expected_reinstatement_premium,
    /// expected_premium, expected_ceding_commission,
    /// expected_profit_commission, on_points)`, with `stage_ceded` one
    /// grid (or NULL) per stage,
    /// with `net` NULL when it is not one compound total.
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
            expected_reinstatement_premium = r.expected_reinstatement_premium,
            expected_premium = r.expected_premium,
            expected_ceding_commission = r.expected_ceding_commission,
            expected_profit_commission = r.expected_profit_commission,
            on_points = r.on_points
        ))
    }

    /// The tower on an annual loss grid, each year's total as one loss:
    /// `list(gross, ceded, net, expected_reinstatement_premium,
    /// expected_premium, expected_ceding_commission,
    /// expected_profit_commission, on_points)`.
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

extendr_module! {
    mod reinsurance;
    fn occurrence_cession_statistics_rust;
    fn aggregate_layer_statistics_rust;
    fn retro_premium_rust;
    fn occurrence_layer_statistics_rust;
    impl XolLayer;
    impl ReinsuranceTower;
}
