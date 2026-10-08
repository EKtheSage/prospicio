//! Aggregate lane: wrappers over the collective model (`prospicio_aggregate`)
//! and `prospicio_pricing` for the R `pricing.R` API (`docs/design/pareto.md`).
//! A truncation of `Inf` means none; `NA` frequencies are derived.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_aggregate::CollectiveModel as CollectiveInner;
use prospicio_pricing::exposure::{
    ExposureCurve, Mbbefd as MbbefdInner, SeverityCurve, TabulatedCurve,
};
use prospicio_pricing::layer::XsLayer;
use prospicio_pricing::natural::{
    Allocation, Pentagon, Portfolio as NaturalInner, Quantity, Target,
};
use prospicio_pricing::risk_load::{self, PremiumRule, Price};
use prospicio_pricing::tower::{Reference, SelectionRule, TowerModel as TowerInner};

use crate::aggregate::{AnyCount, EventSet};
use crate::distributions::{Grid, PredictiveDistribution, Sampled, severity_from_robj};
use crate::pareto::PiecewisePareto;
use crate::risk::RiskDistortion;
use crate::{to_r, whole};

fn truncation(t: f64) -> Option<f64> {
    t.is_finite().then_some(t)
}

fn xs(limit: f64, attachment: f64) -> Result<XsLayer> {
    XsLayer::new(limit, attachment).map_err(to_r)
}

fn rule(name: &str) -> Result<SelectionRule> {
    match name {
        "minimize" => Ok(SelectionRule::MinimizeAlphaRatio),
        "midpoint" => Ok(SelectionRule::Midpoint),
        _ => Err(Error::Other(
            "rule must be \"minimize\" or \"midpoint\"".into(),
        )),
    }
}

/// The collective risk model: a claim count and a severity.
#[extendr]
pub(crate) struct CollectiveModel {
    inner: CollectiveInner<AnyCount, prospicio_prob::SeverityDist>,
}

#[extendr]
impl CollectiveModel {
    fn new(frequency: Robj, severity: Robj) -> Result<Self> {
        let inner = CollectiveInner::new(
            AnyCount::from_robj(&frequency)?,
            severity_from_robj(&severity)?,
        );
        Ok(Self { inner })
    }

    fn excess_frequency(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.inner.excess_frequency(x)).collect()
    }

    fn layer_mean(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer_mean(limit, attachment)
    }

    fn layer_variance(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer_variance(limit, attachment)
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn simulate(&self, n_sims: f64, seed: f64) -> Result<EventSet> {
        let inner = self
            .inner
            .simulate(whole(n_sims, "n_sims")? as usize, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(EventSet { inner })
    }
}

/// A frequency and a piecewise Pareto severity fitted to tower or
/// reference information.
#[extendr]
pub(crate) struct TowerModel {
    inner: TowerInner,
}

#[extendr]
impl TowerModel {
    fn frequency(&self) -> f64 {
        self.inner.frequency
    }

    fn severity(&self) -> PiecewisePareto {
        PiecewisePareto {
            inner: self.inner.severity.clone(),
        }
    }

    fn excess_frequency(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.inner.excess_frequency(x)).collect()
    }

    fn layer_loss(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer_loss(limit, attachment)
    }

    fn match_tower(
        attachments: &[f64],
        layer_losses: &[f64],
        frequencies: &[f64],
        rule_name: &str,
    ) -> Result<Self> {
        let freq: Vec<Option<f64>> = frequencies
            .iter()
            .map(|&f| (!f.is_nan()).then_some(f))
            .collect();
        let inner = prospicio_pricing::tower::match_tower(
            attachments,
            layer_losses,
            &freq,
            rule(rule_name)?,
        )
        .map_err(to_r)?;
        Ok(Self { inner })
    }

    fn fit_pml_curve(
        return_periods: &[f64],
        amounts: &[f64],
        tail_alpha: f64,
        truncation_at: f64,
    ) -> Result<Self> {
        let inner = prospicio_pricing::tower::fit_pml_curve(
            return_periods,
            amounts,
            tail_alpha,
            truncation(truncation_at),
        )
        .map_err(to_r)?;
        Ok(Self { inner })
    }

    /// Layers as parallel vectors `limits`, `attachments`, `losses`;
    /// frequencies as `thresholds`, `frequencies`.
    fn fit_references(
        limits: &[f64],
        attachments: &[f64],
        losses: &[f64],
        thresholds: &[f64],
        frequencies: &[f64],
        default_alpha: f64,
        rule_name: &str,
    ) -> Result<Self> {
        if limits.len() != attachments.len() || limits.len() != losses.len() {
            return Err(Error::Other(
                "layer limits, attachments and losses must have the same length".into(),
            ));
        }
        if thresholds.len() != frequencies.len() {
            return Err(Error::Other(
                "thresholds and frequencies must have the same length".into(),
            ));
        }
        let mut refs = Vec::new();
        for i in 0..limits.len() {
            refs.push(Reference::Layer {
                layer: xs(limits[i], attachments[i])?,
                expected_loss: losses[i],
            });
        }
        for (&threshold, &frequency) in thresholds.iter().zip(frequencies) {
            refs.push(Reference::Frequency {
                threshold,
                frequency,
            });
        }
        let inner =
            prospicio_pricing::tower::fit_references(&refs, default_alpha, rule(rule_name)?)
                .map_err(to_r)?;
        Ok(Self { inner })
    }
}

/// Increased limit factors `LEV(limit) / LEV(basic_limit)`.
#[extendr]
fn pricing_ilf(severity: Robj, limit: &[f64], basic_limit: f64) -> Result<Vec<f64>> {
    let sev = severity_from_robj(&severity)?;
    limit
        .iter()
        .map(|&l| prospicio_pricing::layer::ilf(&sev, l, basic_limit).map_err(to_r))
        .collect()
}

/// Loss elimination ratios `LEV(d) / E[X]`.
#[extendr]
fn pricing_loss_elimination_ratio(severity: Robj, deductible: &[f64]) -> Result<Vec<f64>> {
    let sev = severity_from_robj(&severity)?;
    deductible
        .iter()
        .map(|&d| prospicio_pricing::layer::loss_elimination_ratio(&sev, d).map_err(to_r))
        .collect()
}

#[extendr]
fn pricing_pareto_extrapolation(
    from: &[f64],
    to: &[f64],
    alpha: f64,
    truncation_at: f64,
) -> Result<f64> {
    if from.len() != 2 || to.len() != 2 {
        return Err(Error::Other("layers are c(limit, attachment)".into()));
    }
    prospicio_pricing::layer::pareto_extrapolation(
        xs(from[0], from[1])?,
        xs(to[0], to[1])?,
        alpha,
        truncation(truncation_at),
    )
    .map_err(to_r)
}

#[extendr]
fn pricing_alpha_between_layers(a: &[f64], b: &[f64], truncation_at: f64) -> Result<f64> {
    if a.len() != 3 || b.len() != 3 {
        return Err(Error::Other(
            "layers are c(limit, attachment, expected_loss)".into(),
        ));
    }
    prospicio_pricing::layer::alpha_between_layers(
        (xs(a[0], a[1])?, a[2]),
        (xs(b[0], b[1])?, b[2]),
        truncation(truncation_at),
    )
    .map_err(to_r)
}

#[extendr]
fn pricing_alpha_between_frequency_and_layer(
    threshold: f64,
    frequency: f64,
    limit: f64,
    attachment: f64,
    expected_loss: f64,
    truncation_at: f64,
) -> Result<f64> {
    prospicio_pricing::layer::alpha_between_frequency_and_layer(
        threshold,
        frequency,
        xs(limit, attachment)?,
        expected_loss,
        truncation(truncation_at),
    )
    .map_err(to_r)
}

#[extendr]
fn pricing_alpha_between_frequencies(
    threshold_1: f64,
    frequency_1: f64,
    threshold_2: f64,
    frequency_2: f64,
    truncation_at: f64,
) -> Result<f64> {
    prospicio_pricing::layer::alpha_between_frequencies(
        threshold_1,
        frequency_1,
        threshold_2,
        frequency_2,
        truncation(truncation_at),
    )
    .map_err(to_r)
}

/// The MBBEFD exposure curve (Bernegger, 1997).
#[extendr]
pub(crate) struct Mbbefd {
    pub(crate) inner: MbbefdInner,
}

#[extendr]
impl Mbbefd {
    fn new(b: f64, g: f64) -> Result<Self> {
        Ok(Self {
            inner: MbbefdInner::new(b, g).map_err(to_r)?,
        })
    }

    fn swiss_re(c: f64) -> Result<Self> {
        Ok(Self {
            inner: MbbefdInner::swiss_re(c).map_err(to_r)?,
        })
    }

    fn b(&self) -> f64 {
        self.inner.b()
    }

    fn g(&self) -> f64 {
        self.inner.g_parameter()
    }

    fn curve(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.g(v)).collect()
    }

    fn cdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.cdf(v)).collect()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn total_loss_probability(&self) -> f64 {
        self.inner.total_loss_probability()
    }

    fn layer_share(&self, limit: f64, attachment: f64, mpl: f64) -> Result<f64> {
        self.inner.layer_share(limit, attachment, mpl).map_err(to_r)
    }

    fn rate_quantile(&self, u: &[f64]) -> Vec<f64> {
        u.iter().map(|&v| self.inner.rate_quantile(v)).collect()
    }
}

/// A tabulated exposure curve, interpolated linearly.
#[extendr]
pub(crate) struct Tabulated {
    pub(crate) inner: TabulatedCurve,
}

#[extendr]
impl Tabulated {
    fn new(x: &[f64], g: &[f64]) -> Result<Self> {
        Ok(Self {
            inner: TabulatedCurve::new(x, g).map_err(to_r)?,
        })
    }

    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    fn g(&self) -> Vec<f64> {
        self.inner.g_values().to_vec()
    }

    fn curve(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.g(v)).collect()
    }

    fn mean(&self) -> f64 {
        self.inner.mean_rate()
    }

    fn layer_share(&self, limit: f64, attachment: f64, mpl: f64) -> Result<f64> {
        self.inner.layer_share(limit, attachment, mpl).map_err(to_r)
    }

    fn rate_quantile(&self, u: &[f64]) -> Vec<f64> {
        u.iter().map(|&v| self.inner.rate_quantile(v)).collect()
    }
}

fn band_curve(obj: &Robj) -> Result<prospicio_pricing::profile::BandCurve> {
    if let Ok(c) = <&Mbbefd>::try_from(obj) {
        return Ok(c.inner.into());
    }
    if let Ok(c) = <&Tabulated>::try_from(obj) {
        return Ok(c.inner.clone().into());
    }
    Err(Error::Other(
        "each band's curve must be an mbbefd or a tabulated_curve".into(),
    ))
}

/// A risk profile; `expected_losses` or `premiums` (with `loss_ratio`, one
/// value or one per band) is empty when not given. `lower` and `upper` are
/// empty, or one per band with NA for a band without bounds; `tilted`
/// spreads a band with bounds about its given sum insured.
#[extendr]
pub(crate) struct RiskProfile {
    inner: prospicio_pricing::profile::RiskProfile,
}

#[extendr]
impl RiskProfile {
    #[allow(clippy::too_many_arguments)]
    fn new(
        sums_insured: &[f64],
        risks: &[f64],
        curves: List,
        expected_losses: &[f64],
        premiums: &[f64],
        loss_ratio: &[f64],
        lower: &[f64],
        upper: &[f64],
        tilted: bool,
    ) -> Result<Self> {
        use prospicio_pricing::profile::{Band, RiskProfile as Inner};
        let n = sums_insured.len();
        if risks.len() != n || curves.len() != n {
            return Err(Error::Other(
                "give one number of risks and one curve per band".into(),
            ));
        }
        let curves = curves
            .values()
            .map(|c| band_curve(&c))
            .collect::<Result<Vec<_>>>()?;
        let bands = if !expected_losses.is_empty() {
            if expected_losses.len() != n {
                return Err(Error::Other("give one expected loss per band".into()));
            }
            (0..n)
                .map(|i| {
                    Band::from_expected_loss(
                        sums_insured[i],
                        risks[i],
                        expected_losses[i],
                        curves[i].clone(),
                    )
                })
                .collect::<prospicio_core::Result<Vec<_>>>()
        } else {
            let lr = |i: usize| loss_ratio[if loss_ratio.len() == 1 { 0 } else { i }];
            if premiums.len() != n || !(loss_ratio.len() == 1 || loss_ratio.len() == n) {
                return Err(Error::Other(
                    "give expected_loss, or one premium per band with a loss_ratio".into(),
                ));
            }
            (0..n)
                .map(|i| {
                    Band::from_premium(
                        sums_insured[i],
                        risks[i],
                        premiums[i],
                        lr(i),
                        curves[i].clone(),
                    )
                })
                .collect::<prospicio_core::Result<Vec<_>>>()
        }
        .map_err(to_r)?;
        let bands = if lower.is_empty() && upper.is_empty() {
            bands
        } else {
            if lower.len() != n || upper.len() != n {
                return Err(Error::Other("give one lower and one upper per band".into()));
            }
            bands
                .into_iter()
                .enumerate()
                .map(|(i, b)| match (lower[i].is_nan(), upper[i].is_nan()) {
                    (true, true) => Ok(b),
                    (false, false) if tilted => {
                        b.with_tilted_bounds(lower[i], upper[i]).map_err(to_r)
                    }
                    (false, false) => b.with_bounds(lower[i], upper[i]).map_err(to_r),
                    _ => Err(Error::Other("a band has both bounds or neither".into())),
                })
                .collect::<Result<_>>()?
        };
        Ok(Self {
            inner: Inner::new(bands).map_err(to_r)?,
        })
    }

    fn expected_loss(&self) -> f64 {
        self.inner.expected_loss()
    }

    fn expected_claims(&self) -> Vec<f64> {
        self.inner
            .bands()
            .iter()
            .map(|b| b.expected_claims())
            .collect()
    }

    /// `surplus_retention` NaN for no surplus.
    fn expected_layer_loss(
        &self,
        limit: f64,
        attachment: f64,
        surplus_retention: f64,
        surplus_lines: f64,
    ) -> Result<f64> {
        let surplus = (!surplus_retention.is_nan()).then_some((surplus_retention, surplus_lines));
        self.inner
            .expected_layer_loss(limit, attachment, surplus)
            .map_err(to_r)
    }

    fn expected_surplus_loss(&self, retention: f64, lines: f64) -> f64 {
        self.inner.expected_surplus_loss(retention, lines)
    }

    fn simulate(&self, n_sims: f64, seed: f64) -> Result<crate::aggregate::EventSet> {
        let inner = self
            .inner
            .simulate(whole(n_sims, "n_sims")? as usize, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(crate::aggregate::EventSet { inner })
    }
}

/// The exposure curve of a severity capped at `mpl`, at each `x`.
#[extendr]
fn pricing_severity_exposure_curve(severity: Robj, mpl: f64, x: &[f64]) -> Result<Vec<f64>> {
    let sev = severity_from_robj(&severity)?;
    let curve = SeverityCurve::new(&sev, mpl).map_err(to_r)?;
    Ok(x.iter().map(|&v| curve.g(v)).collect())
}

fn distortion_arg(d: &Robj, name: &str) -> Result<prospicio_prob::Distortion> {
    <&RiskDistortion>::try_from(d)
        .map(|d| d.inner.clone())
        .map_err(|_| Error::Other(format!("{name} must be a distortion")))
}

/// The premium rule: a cost of capital `rate`, or (`rate` NA) the
/// pricing distortion `pricing`.
fn premium_rule(rate: Option<f64>, pricing: &Robj) -> Result<PremiumRule> {
    match (rate, pricing.is_null()) {
        (Some(r), true) => PremiumRule::cost_of_capital(r).map_err(to_r),
        (None, false) => Ok(PremiumRule::Distortion(distortion_arg(
            pricing,
            "distortion",
        )?)),
        _ => Err(Error::Other(
            "give exactly one of cost_of_capital and distortion".into(),
        )),
    }
}

fn price_list(p: &Price) -> List {
    list!(
        expected_loss = p.expected_loss,
        premium = p.premium,
        assets = p.assets,
        margin = p.margin(),
        capital = p.capital(),
        loss_ratio = p.loss_ratio(),
        return_on_capital = p.return_on_capital()
    )
}

/// Risk-loaded price of a sampled or a predictive distribution's total.
#[extendr]
fn pricing_price(losses: Robj, assets: Robj, rate: Option<f64>, pricing: Robj) -> Result<List> {
    let rule = premium_rule(rate, &pricing)?;
    let assets = distortion_arg(&assets, "assets")?;
    let p = if let Ok(s) = <&Sampled>::try_from(&losses) {
        risk_load::price(&s.inner, &rule, &assets)
    } else if let Ok(pd) = <&PredictiveDistribution>::try_from(&losses) {
        risk_load::price(pd.inner.total(), &rule, &assets)
    } else {
        return Err(Error::Other(
            "losses must be a sampled or a predictive_distribution".into(),
        ));
    }
    .map_err(to_r)?;
    Ok(price_list(&p))
}

/// Portfolio price: `list(total, allocated, standalone)`, the last two
/// as lists of columns, one value per component.
#[extendr]
fn pricing_price_portfolio(
    pd: Robj,
    assets: Robj,
    rate: Option<f64>,
    pricing: Robj,
) -> Result<List> {
    let rule = premium_rule(rate, &pricing)?;
    let assets = distortion_arg(&assets, "assets")?;
    let pd = <&PredictiveDistribution>::try_from(&pd)
        .map_err(|_| Error::Other("expected a predictive_distribution".into()))?;
    let p = risk_load::price_portfolio(&pd.inner, &rule, &assets).map_err(to_r)?;
    let columns = |v: &[Price]| {
        list!(
            expected_loss = v.iter().map(|p| p.expected_loss).collect::<Vec<_>>(),
            premium = v.iter().map(|p| p.premium).collect::<Vec<_>>(),
            assets = v.iter().map(|p| p.assets).collect::<Vec<_>>()
        )
    };
    Ok(list!(
        total = price_list(&p.total),
        allocated = columns(&p.allocated),
        standalone = columns(&p.standalone)
    ))
}

/// A portfolio as its total's distribution and each unit's conditional
/// expectation given the total (`prospicio_pricing::natural`).
#[extendr]
pub(crate) struct NaturalPortfolio {
    inner: NaturalInner,
}

fn natural_allocation(name: &str) -> Result<Allocation> {
    match name {
        "linear" => Ok(Allocation::Linear),
        "lifted" => Ok(Allocation::Lifted),
        other => Err(Error::Other(format!(
            "allocation must be linear or lifted, not {other:?}"
        ))),
    }
}

fn pentagon_list(rows: &[&Pentagon], units: Vec<String>) -> List {
    let col = |f: &dyn Fn(&Pentagon) -> f64| rows.iter().map(|p| f(p)).collect::<Vec<f64>>();
    list!(
        unit = units,
        loss = col(&|p| p.loss),
        margin = col(&|p| p.margin),
        premium = col(&|p| p.premium),
        capital = col(&|p| p.capital),
        assets = col(&|p| p.assets),
        loss_ratio = col(&Pentagon::loss_ratio),
        premium_to_capital = col(&Pentagon::premium_to_capital),
        return_on_capital = col(&Pentagon::return_on_capital)
    )
}

#[extendr]
impl NaturalPortfolio {
    /// From scenarios: `x` holds unit losses scenario-major (`n_rows` rows
    /// of `units.len()`), with `probs` (empty for equal).
    fn from_rows(units: Vec<String>, x: &[f64], probs: &[f64]) -> Result<Self> {
        let m = units.len();
        if m == 0 || x.len() % m != 0 {
            return Err(Error::Other(
                "the loss matrix must have one column per unit".into(),
            ));
        }
        let rows: Vec<Vec<f64>> = x.chunks_exact(m).map(<[f64]>::to_vec).collect();
        let probs = if probs.is_empty() { None } else { Some(probs) };
        Ok(Self {
            inner: NaturalInner::from_rows(units, &rows, probs).map_err(to_r)?,
        })
    }

    fn from_predictive(pd: Robj) -> Result<Self> {
        let pd = <&PredictiveDistribution>::try_from(&pd)
            .map_err(|_| Error::Other("expected a predictive_distribution".into()))?;
        Ok(Self {
            inner: NaturalInner::from_predictive(&pd.inner).map_err(to_r)?,
        })
    }

    /// From independent units, a list of grid distributions with one step.
    fn from_independent(units: Vec<String>, grids: List) -> Result<Self> {
        let grids = grids
            .values()
            .map(|g| {
                <&Grid>::try_from(&g)
                    .map(|g| g.inner.clone())
                    .map_err(|_| Error::Other("expected a list of grid_distribution".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inner: NaturalInner::from_independent(units, &grids).map_err(to_r)?,
        })
    }

    fn units(&self) -> Vec<String> {
        self.inner.units().to_vec()
    }

    fn totals(&self) -> Vec<f64> {
        self.inner.totals().to_vec()
    }

    fn probs(&self) -> Vec<f64> {
        self.inner.probs().to_vec()
    }

    /// Unit `i` (1-based) conditional expectations.
    fn kappa(&self, i: f64) -> Result<Vec<f64>> {
        let i = whole(i, "unit")? as usize;
        if i == 0 || i > self.inner.units().len() {
            return Err(Error::Other(format!("no unit {i}")));
        }
        Ok(self.inner.kappa(i - 1))
    }

    fn expected(&self) -> Vec<f64> {
        self.inner.expected()
    }

    fn assets(&self, p: f64) -> Result<f64> {
        self.inner.assets(p).map_err(to_r)
    }

    fn max(&self) -> f64 {
        self.inner.max()
    }

    fn price(&self, distortion: Robj, assets: f64, allocation: &str) -> Result<List> {
        let g = distortion_arg(&distortion, "distortion")?;
        let p = self
            .inner
            .price(&g, assets, natural_allocation(allocation)?)
            .map_err(to_r)?;
        let rows: Vec<&Pentagon> = p.allocated.iter().chain([&p.total]).collect();
        let mut units = p.units.clone();
        units.push("total".into());
        Ok(pentagon_list(&rows, units))
    }

    /// `target` is "premium", "return_on_capital" or "loss_ratio".
    fn calibrate(
        &self,
        family: &str,
        assets: f64,
        target: &str,
        value: f64,
        r0: f64,
    ) -> Result<RiskDistortion> {
        let target = match target {
            "premium" => Target::Premium(value),
            "return_on_capital" => Target::ReturnOnCapital(value),
            "loss_ratio" => Target::LossRatio(value),
            other => return Err(Error::Other(format!("unknown target {other:?}"))),
        };
        let family = crate::risk::family_from(family, r0)?;
        Ok(RiskDistortion {
            inner: self.inner.calibrate(family, assets, target).map_err(to_r)?,
        })
    }

    fn bodoff(&self, assets: f64) -> Vec<f64> {
        self.inner.bodoff(assets)
    }

    fn epd(&self, assets: f64) -> List {
        let (total, units) = self.inner.epd(assets);
        list!(total = total, units = units)
    }

    fn assets_for_epd(&self, epd: f64) -> Result<f64> {
        self.inner.assets_for_epd(epd).map_err(to_r)
    }
}

/// The pentagon from three named quantities (`names`, `values`).
#[extendr]
fn pricing_pentagon(names: Vec<String>, values: &[f64]) -> Result<List> {
    if names.len() != 3 || values.len() != 3 {
        return Err(Error::Other("give exactly three quantities".into()));
    }
    let q = |n: &str| -> Result<Quantity> {
        Ok(match n {
            "loss" => Quantity::Loss,
            "margin" => Quantity::Margin,
            "premium" => Quantity::Premium,
            "capital" => Quantity::Capital,
            "assets" => Quantity::Assets,
            "loss_ratio" => Quantity::LossRatio,
            "premium_to_capital" => Quantity::PremiumToCapital,
            "return_on_capital" => Quantity::ReturnOnCapital,
            other => return Err(Error::Other(format!("unknown quantity {other:?}"))),
        })
    };
    let known = [
        (q(&names[0])?, values[0]),
        (q(&names[1])?, values[1]),
        (q(&names[2])?, values[2]),
    ];
    let p = Pentagon::solve(known).map_err(to_r)?;
    Ok(pentagon_list(&[&p], vec!["total".into()]))
}

extendr_module! {
    mod pricing;
    impl NaturalPortfolio;
    fn pricing_pentagon;
    impl CollectiveModel;
    impl TowerModel;
    impl Mbbefd;
    impl Tabulated;
    impl RiskProfile;
    fn pricing_severity_exposure_curve;
    fn pricing_price;
    fn pricing_price_portfolio;
    fn pricing_ilf;
    fn pricing_loss_elimination_ratio;
    fn pricing_pareto_extrapolation;
    fn pricing_alpha_between_layers;
    fn pricing_alpha_between_frequency_and_layer;
    fn pricing_alpha_between_frequencies;
}
