//! Aggregate lane: wrappers over the collective model (`act_aggregate`)
//! and `act_pricing` for the R `pricing.R` API (`docs/design/pareto.md`).
//! A truncation of `Inf` means none; `NA` frequencies are derived.

use act_aggregate::CollectiveModel as CollectiveInner;
use act_pricing::exposure::{ExposureCurve, Mbbefd as MbbefdInner, SeverityCurve};
use act_pricing::layer::XsLayer;
use act_pricing::risk_load::{self, PremiumRule, Price};
use act_pricing::tower::{Reference, SelectionRule, TowerModel as TowerInner};
use extendr_api::prelude::*;
use extendr_api::{Error, Result};

use crate::aggregate::{AnyCount, EventSet};
use crate::distributions::{PredictiveDistribution, Sampled, severity_from_robj};
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
    inner: CollectiveInner<AnyCount, act_prob::SeverityDist>,
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
        let inner =
            act_pricing::tower::match_tower(attachments, layer_losses, &freq, rule(rule_name)?)
                .map_err(to_r)?;
        Ok(Self { inner })
    }

    fn fit_pml_curve(
        return_periods: &[f64],
        amounts: &[f64],
        tail_alpha: f64,
        truncation_at: f64,
    ) -> Result<Self> {
        let inner = act_pricing::tower::fit_pml_curve(
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
        let inner = act_pricing::tower::fit_references(&refs, default_alpha, rule(rule_name)?)
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
        .map(|&l| act_pricing::layer::ilf(&sev, l, basic_limit).map_err(to_r))
        .collect()
}

/// Loss elimination ratios `LEV(d) / E[X]`.
#[extendr]
fn pricing_loss_elimination_ratio(severity: Robj, deductible: &[f64]) -> Result<Vec<f64>> {
    let sev = severity_from_robj(&severity)?;
    deductible
        .iter()
        .map(|&d| act_pricing::layer::loss_elimination_ratio(&sev, d).map_err(to_r))
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
    act_pricing::layer::pareto_extrapolation(
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
    act_pricing::layer::alpha_between_layers(
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
    act_pricing::layer::alpha_between_frequency_and_layer(
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
    act_pricing::layer::alpha_between_frequencies(
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
    inner: MbbefdInner,
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
}

/// The exposure curve of a severity capped at `mpl`, at each `x`.
#[extendr]
fn pricing_severity_exposure_curve(severity: Robj, mpl: f64, x: &[f64]) -> Result<Vec<f64>> {
    let sev = severity_from_robj(&severity)?;
    let curve = SeverityCurve::new(&sev, mpl).map_err(to_r)?;
    Ok(x.iter().map(|&v| curve.g(v)).collect())
}

fn distortion_arg(d: &Robj, name: &str) -> Result<act_prob::Distortion> {
    <&RiskDistortion>::try_from(d)
        .map(|d| d.inner)
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

extendr_module! {
    mod pricing;
    impl CollectiveModel;
    impl TowerModel;
    impl Mbbefd;
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
