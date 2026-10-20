//! Aggregate lane: wrappers over the collective model (`prospicio_aggregate`)
//! and `prospicio_pricing` for the R `pricing.R` API (`docs/design/pareto.md`).
//! A truncation of `Inf` means none; `NA` frequencies are derived.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_aggregate::CollectiveModel as CollectiveInner;
use prospicio_pricing::classical::{self, Kind, Principle};
use prospicio_pricing::experience::{
    ClaimValuation, Development, DevelopmentPattern, Experience, Fx, IndexClause, IndexKind,
    LargeLoss, Prepared, Trend, YearBasis,
};
use prospicio_pricing::exposure::{
    ExposureCurve, Interpolation, Mbbefd as MbbefdInner, SeverityCurve, TabulatedCurve,
};
use prospicio_pricing::layer::XsLayer;
use prospicio_pricing::natural::{
    Allocation, AtomAtAssets, BottomLayer, Pentagon, Portfolio as NaturalInner, PriceSettings,
    Quantity, Target,
};
use prospicio_pricing::payout::{self, PaymentPattern};
use prospicio_pricing::quote::{Loads, Quote};
use prospicio_pricing::risk_load::{self, PremiumRule, Price};
use prospicio_pricing::tower::{Reference, SelectionRule, TowerModel as TowerInner};

use crate::aggregate::{AnyCount, EventSet};
use crate::distributions::{Grid, PredictiveDistribution, Sampled, severity_from_robj};
use crate::pareto::PiecewisePareto;
use crate::reinsurance::ReinsuranceTower;
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

    /// `list(min_alpha, max_alpha, max_ratio, infinite_mean, warnings)`.
    fn diagnostics(&self, min_alpha: f64, max_ratio: f64) -> List {
        let d = self.inner.diagnostics(min_alpha, max_ratio);
        list!(
            min_alpha = d.min_alpha,
            max_alpha = d.max_alpha,
            max_ratio = d.max_ratio,
            infinite_mean = d.infinite_mean,
            warnings = d.warnings
        )
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
    /// frequencies as `thresholds`, `frequencies`; `alpha_bounds` empty or
    /// `c(lo, hi)`.
    #[allow(clippy::too_many_arguments)]
    fn fit_references(
        limits: &[f64],
        attachments: &[f64],
        losses: &[f64],
        thresholds: &[f64],
        frequencies: &[f64],
        default_alpha: f64,
        rule_name: &str,
        alpha_bounds: &[f64],
    ) -> Result<Self> {
        let bounds = match *alpha_bounds {
            [] => None,
            [lo, hi] => Some((lo, hi)),
            _ => {
                return Err(Error::Other("alpha_bounds is NULL or c(lo, hi)".into()));
            }
        };
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
        let inner = prospicio_pricing::tower::fit_references_bounded(
            &refs,
            default_alpha,
            rule(rule_name)?,
            bounds,
        )
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

/// The CV of the loss to `limit` xs `attachment` per loss that reaches it.
#[extendr]
fn pricing_layer_cv(severity: Robj, limit: f64, attachment: f64) -> Result<f64> {
    let sev = severity_from_robj(&severity)?;
    prospicio_pricing::layer::layer_cv(&sev, limit, attachment).map_err(to_r)
}

/// The Swiss Re `c` whose curve has mean destruction rate `rate`.
#[extendr]
fn pricing_swiss_re_c_for_mean_rate(rate: f64) -> Result<f64> {
    Ok(MbbefdInner::swiss_re_for_mean_rate(rate).map_err(to_r)?.1)
}

/// The Swiss Re `c` that gives `limit` xs `attachment` the share `share`
/// of a risk's loss with maximum possible loss `mpl`.
#[extendr]
fn pricing_swiss_re_c_for_layer_share(
    limit: f64,
    attachment: f64,
    mpl: f64,
    share: f64,
) -> Result<f64> {
    Ok(
        MbbefdInner::swiss_re_for_layer_share(limit, attachment, mpl, share)
            .map_err(to_r)?
            .1,
    )
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

/// A tabulated exposure curve, interpolated linearly or log-log.
#[extendr]
pub(crate) struct Tabulated {
    pub(crate) inner: TabulatedCurve,
}

#[extendr]
impl Tabulated {
    fn new(x: &[f64], g: &[f64], interpolation: &str) -> Result<Self> {
        let how = match interpolation {
            "linear" => Interpolation::Linear,
            "log-log" => Interpolation::LogLog,
            other => {
                return Err(Error::Other(format!(
                    "interpolation must be \"linear\" or \"log-log\", not {other:?}"
                )));
            }
        };
        Ok(Self {
            inner: TabulatedCurve::with_interpolation(x, g, how).map_err(to_r)?,
        })
    }

    fn interpolation(&self) -> &'static str {
        match self.inner.interpolation() {
            Interpolation::Linear => "linear",
            Interpolation::LogLog => "log-log",
        }
    }

    fn from_severity(severity: Robj, mpl: f64, points: f64) -> Result<Self> {
        let sev = severity_from_robj(&severity)?;
        if !(points.is_finite() && points >= 0.0 && points.fract() == 0.0) {
            return Err(Error::Other("points must be a whole number".into()));
        }
        Ok(Self {
            inner: TabulatedCurve::from_severity(&sev, mpl, points as usize).map_err(to_r)?,
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
/// empty, or one per band with NA for a band without bounds; `spread` is
/// "uniform", "tilted" (about the band's given sum insured) or "auto"
/// (tilted when the sum insured lies strictly between the bounds).
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
        spread: &str,
    ) -> Result<Self> {
        use prospicio_pricing::profile::{Band, RiskProfile as Inner};
        if !matches!(spread, "uniform" | "tilted" | "auto") {
            return Err(Error::Other(format!(
                "spread must be \"uniform\", \"tilted\" or \"auto\", not {spread:?}"
            )));
        }
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
                    (false, false) => match spread {
                        "tilted" => b.with_tilted_bounds(lower[i], upper[i]),
                        "auto" => b.with_spread_bounds(lower[i], upper[i]),
                        _ => b.with_bounds(lower[i], upper[i]),
                    }
                    .map_err(to_r),
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

    fn sums_insured(&self) -> Vec<f64> {
        self.inner.bands().iter().map(|b| b.sum_insured).collect()
    }

    fn mean_sums_insured(&self) -> Vec<f64> {
        self.inner
            .bands()
            .iter()
            .map(|b| b.mean_sum_insured())
            .collect()
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

/// Risk-loaded price of a sampled, a predictive distribution's total or a
/// grid distribution.
#[extendr]
fn pricing_price(losses: Robj, assets: Robj, rate: Option<f64>, pricing: Robj) -> Result<List> {
    let rule = premium_rule(rate, &pricing)?;
    let assets = distortion_arg(&assets, "assets")?;
    let p = if let Ok(s) = <&Sampled>::try_from(&losses) {
        risk_load::price(&s.inner, &rule, &assets)
    } else if let Ok(pd) = <&PredictiveDistribution>::try_from(&losses) {
        risk_load::price(pd.inner.total(), &rule, &assets)
    } else if let Ok(g) = <&Grid>::try_from(&losses) {
        risk_load::price_grid(&g.inner, &rule, &assets)
    } else {
        return Err(Error::Other(
            "losses must be a sampled, a predictive_distribution or a grid_distribution".into(),
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

    /// `bottom_layer` is "limit" or "none", `atom` "separate" or "lumped".
    fn price(
        &self,
        distortion: Robj,
        assets: f64,
        allocation: &str,
        bottom_layer: &str,
        atom: &str,
    ) -> Result<List> {
        let g = distortion_arg(&distortion, "distortion")?;
        let settings = PriceSettings {
            allocation: natural_allocation(allocation)?,
            bottom_layer: match bottom_layer {
                "limit" => BottomLayer::Limit,
                "none" => BottomLayer::Zero,
                other => {
                    return Err(Error::Other(format!(
                        "bottom_layer must be limit or none, not {other:?}"
                    )));
                }
            },
            atom: match atom {
                "separate" => AtomAtAssets::Separate,
                "lumped" => AtomAtAssets::Lumped,
                other => {
                    return Err(Error::Other(format!(
                        "atom must be separate or lumped, not {other:?}"
                    )));
                }
            },
        };
        let p = self.inner.price_with(&g, assets, settings).map_err(to_r)?;
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

    /// Each unit alone at its own VaR, the book together, and their sum:
    /// a pentagon table with rows for the units, "total" and "sum".
    fn price_stand_alone(&self, distortion: Robj, p: f64) -> Result<List> {
        let g = distortion_arg(&distortion, "distortion")?;
        let s = self.inner.price_stand_alone(&g, p).map_err(to_r)?;
        let rows: Vec<&Pentagon> = s.stand_alone.iter().chain([&s.total, &s.sum]).collect();
        let mut units = s.units.clone();
        units.extend(["total".into(), "sum".into()]);
        Ok(pentagon_list(&rows, units))
    }

    /// Each unit's comonotonic allocation at each total, unit by unit.
    fn comonotonic_kappa(&self) -> List {
        List::from_values(self.inner.comonotonic_kappa())
    }

    fn epd(&self, assets: f64) -> List {
        let (total, units) = self.inner.epd(assets);
        list!(total = total, units = units)
    }

    fn assets_for_epd(&self, epd: f64) -> Result<f64> {
        self.inner.assets_for_epd(epd).map_err(to_r)
    }

    /// `list(mean, senior, equal, junior, total)`: each unit's recovery
    /// under each priority rule, and `E[X ∧ a]`.
    fn priority(&self, assets: f64) -> Result<List> {
        let l = self.inner.priority(assets).map_err(to_r)?;
        Ok(list!(
            mean = l.mean,
            senior = l.senior,
            equal = l.equal,
            junior = l.junior,
            total = l.total
        ))
    }

    /// `list(junior, senior)` on the totals, for unit `i` (1-based).
    fn priority_kappa(&self, i: f64, assets: f64) -> Result<List> {
        let i = whole(i, "unit")? as usize;
        if i == 0 {
            return Err(Error::Other("no unit 0".into()));
        }
        let (junior, senior) = self.inner.priority_kappa(i - 1, assets).map_err(to_r)?;
        Ok(list!(junior = junior, senior = senior))
    }

    /// `list(lower, upper, lower_distortion, upper_distortion)`, one entry
    /// per unit in each.
    fn premium_bounds(&self, premium: f64, assets: f64) -> Result<List> {
        let b = self.inner.premium_bounds(premium, assets).map_err(to_r)?;
        let wrap = |d: &prospicio_prob::Distortion| Robj::from(RiskDistortion { inner: d.clone() });
        Ok(list!(
            lower = b.iter().map(|u| u.lower).collect::<Vec<f64>>(),
            upper = b.iter().map(|u| u.upper).collect::<Vec<f64>>(),
            lower_distortion = List::from_values(b.iter().map(|u| wrap(&u.lower_distortion))),
            upper_distortion = List::from_values(b.iter().map(|u| wrap(&u.upper_distortion)))
        ))
    }
}

fn classical_kind(name: &str, q: f64) -> Result<Kind> {
    Ok(match name {
        "expected_value" => Kind::ExpectedValue,
        "variance" => Kind::Variance,
        "standard_deviation" => Kind::StandardDeviation,
        "semi_variance" => Kind::SemiVariance,
        "exponential" => Kind::Exponential,
        "esscher" => Kind::Esscher,
        "dutch" => Kind::Dutch,
        "fischer" => Kind::Fischer { q },
        "var" => Kind::Var,
        other => return Err(Error::Other(format!("unknown premium principle {other:?}"))),
    })
}

/// The premium of `x` under a classical principle with this loading.
#[extendr]
fn pricing_classical_premium(x: Robj, principle: &str, loading: f64, q: f64) -> Result<f64> {
    let (v, p) = crate::risk::discrete_of(&x)?;
    let pr = match classical_kind(principle, q)? {
        Kind::ExpectedValue => Principle::ExpectedValue(loading),
        Kind::Variance => Principle::Variance(loading),
        Kind::StandardDeviation => Principle::StandardDeviation(loading),
        Kind::SemiVariance => Principle::SemiVariance(loading),
        Kind::Exponential => Principle::Exponential(loading),
        Kind::Esscher => Principle::Esscher(loading),
        Kind::Dutch => Principle::Dutch(loading),
        Kind::Fischer { q } => Principle::Fischer { theta: loading, q },
        Kind::Var => Principle::Var(loading),
    };
    pr.premium(&v, &p).map_err(to_r)
}

/// The loading of a classical principle that prices `x` at `premium`.
#[extendr]
fn pricing_calibrate_classical(x: Robj, principle: &str, premium: f64, q: f64) -> Result<f64> {
    let (v, p) = crate::risk::discrete_of(&x)?;
    Ok(
        match classical::calibrate(classical_kind(principle, q)?, &v, &p, premium).map_err(to_r)? {
            Principle::ExpectedValue(t)
            | Principle::Variance(t)
            | Principle::StandardDeviation(t)
            | Principle::SemiVariance(t)
            | Principle::Exponential(t)
            | Principle::Esscher(t)
            | Principle::Dutch(t)
            | Principle::Var(t) => t,
            Principle::Fischer { theta, .. } => theta,
        },
    )
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

/// Pricing bounds: `list(lower, upper, lower_distortion, upper_distortion)`,
/// one entry per risk in each.
#[extendr]
fn pricing_bounds_rust(x: Robj, risks: List, premium: f64, assets: f64) -> Result<List> {
    let (xv, xp) = crate::risk::discrete_of(&x)?;
    let ys = risks
        .iter()
        .map(|(name, y)| {
            let (yv, yp) = crate::risk::discrete_of(&y)?;
            Ok((name.to_string(), yv, yp))
        })
        .collect::<Result<Vec<_>>>()?;
    let pb =
        prospicio_pricing::bounds::PricingBounds::new((&xv, &xp), &ys, assets).map_err(to_r)?;
    let b = pb.bounds(premium).map_err(to_r)?;
    let wrap = |d: &prospicio_prob::Distortion| Robj::from(RiskDistortion { inner: d.clone() });
    Ok(list!(
        lower = b.iter().map(|u| u.lower).collect::<Vec<f64>>(),
        upper = b.iter().map(|u| u.upper).collect::<Vec<f64>>(),
        lower_distortion = List::from_values(b.iter().map(|u| wrap(&u.lower_distortion))),
        upper_distortion = List::from_values(b.iter().map(|u| wrap(&u.upper_distortion)))
    ))
}

/// The distortion envelope: `list(lower, upper)` at each `s`.
#[extendr]
fn distortion_envelope_rust(x: Robj, premium: f64, s: &[f64], assets: f64) -> Result<List> {
    let (xv, xp) = crate::risk::discrete_of(&x)?;
    let env = prospicio_pricing::bounds::distortion_envelope((&xv, &xp), premium, assets, s)
        .map_err(to_r)?;
    let (lower, upper): (Vec<f64>, Vec<f64>) = env.into_iter().unzip();
    Ok(list!(lower = lower, upper = upper))
}

/// A quote: the technical premium of a risk-loaded price (`premium` NaN),
/// or the same cover at a quoted `premium`. `limit` NaN for none.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn pricing_quote(
    expected_loss: f64,
    risk_premium: f64,
    assets: f64,
    expenses: f64,
    brokerage: f64,
    fixed: f64,
    limit: f64,
    premium: f64,
) -> Result<List> {
    let price = Price {
        expected_loss,
        premium: risk_premium,
        assets,
    };
    let loads = Loads::new(expenses, brokerage, fixed).map_err(to_r)?;
    let limit = (!limit.is_nan()).then_some(limit);
    let mut q = Quote::technical(&price, loads, limit).map_err(to_r)?;
    if !premium.is_nan() {
        q = q.at(premium).map_err(to_r)?;
    }
    let na = |x: Option<f64>| x.unwrap_or(f64::NAN);
    Ok(list!(
        expected_loss = q.expected_loss,
        margin = q.margin(),
        risk_premium = q.risk_premium,
        expenses = q.expenses(),
        brokerage = q.brokerage(),
        premium = q.premium,
        net_premium = q.net_premium(),
        assets = q.assets,
        capital = q.capital(),
        return_on_capital = q.return_on_capital(),
        loss_ratio = q.loss_ratio(),
        multiple = q.multiple(),
        rate_on_line = na(q.rate_on_line()),
        payback = na(q.payback())
    ))
}

/// Large losses developed and trended to the prospective period: by `rate`
/// to `to`, or, with `rate` NaN, by `factors`; developed by the pattern
/// `(dev_ages, dev_factors)` at each loss's age at `valuation` (NaN for
/// none), or by `dev_per_loss`, or not at all when both are empty;
/// converted by `fx` first (empty for none, one stated rate, or one per
/// loss).
#[allow(clippy::too_many_arguments)]
fn prepare_large_losses(
    ids: Vec<String>,
    accident: &[f64],
    amount: &[f64],
    threshold: f64,
    rate: f64,
    to: f64,
    factors: &[f64],
    valuation: f64,
    dev_ages: &[f64],
    dev_factors: &[f64],
    dev_per_loss: &[f64],
    fx: &[f64],
) -> Result<Prepared> {
    if ids.len() != accident.len() || ids.len() != amount.len() {
        return Err(Error::Other(format!(
            "{} ids, {} accident dates and {} amounts",
            ids.len(),
            accident.len(),
            amount.len()
        )));
    }
    let losses = ids
        .into_iter()
        .zip(accident)
        .zip(amount)
        .map(|((id, &accident), &amount)| LargeLoss {
            id,
            accident,
            amount,
        })
        .collect();
    let trend = if rate.is_nan() {
        Trend::Factors(factors.to_vec())
    } else {
        Trend::Rate { rate, to }
    };
    let mut e = Experience::new(losses)
        .and_then(|e| e.with_threshold(threshold))
        .map_err(to_r)?;
    if !valuation.is_nan() {
        e = e.with_valuation(valuation).map_err(to_r)?;
    }
    match fx {
        [] => {}
        [r] => e = e.with_fx(Fx::Rate(*r)).map_err(to_r)?,
        r => e = e.with_fx(Fx::PerLoss(r.to_vec())).map_err(to_r)?,
    }
    let development = if !dev_ages.is_empty() {
        Some(Development::Pattern(
            DevelopmentPattern::new(dev_ages.to_vec(), dev_factors.to_vec()).map_err(to_r)?,
        ))
    } else if !dev_per_loss.is_empty() {
        Some(Development::Factors(dev_per_loss.to_vec()))
    } else {
        None
    };
    match development {
        Some(d) => e.prepare_developed(&trend, &d),
        None => e.prepare(&trend),
    }
    .map_err(to_r)
}

fn tower_list(towers: List) -> Result<Vec<prospicio_aggregate::Tower>> {
    towers
        .values()
        .map(|t| {
            <&ReinsuranceTower>::try_from(&t)
                .map(|t| t.inner.clone())
                .map_err(|_| Error::Other("towers must be reinsurance_tower objects".into()))
        })
        .collect()
}

/// Each year's losses under its own program beside the proposed tower:
/// `list(year, reported_losses, trended_losses, historical_layers,
/// historical, proposed_layers, proposed)`, with `historical_layers` and
/// `historical` a list per year and `proposed` year-by-layer, by column.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn pricing_as_if(
    ids: Vec<String>,
    accident: &[f64],
    amount: &[f64],
    threshold: f64,
    rate: f64,
    to: f64,
    factors: &[f64],
    valuation: f64,
    dev_ages: &[f64],
    dev_factors: &[f64],
    dev_per_loss: &[f64],
    fx: &[f64],
    historical: List,
    years: &[i32],
    proposed: Robj,
) -> Result<List> {
    let p = prepare_large_losses(
        ids,
        accident,
        amount,
        threshold,
        rate,
        to,
        factors,
        valuation,
        dev_ages,
        dev_factors,
        dev_per_loss,
        fx,
    )?;
    let historical = tower_list(historical)?;
    let proposed = <&ReinsuranceTower>::try_from(&proposed)
        .map_err(|_| Error::Other("proposed must be a reinsurance_tower".into()))?;
    let a = p.as_if(&historical, years, &proposed.inner).map_err(to_r)?;
    let layers = a.proposed_layers.len();
    let proposed: Vec<f64> = (0..layers)
        .flat_map(|i| a.years.iter().map(move |r| r.proposed[i]))
        .collect();
    Ok(list!(
        year = a.years.iter().map(|r| r.year).collect::<Vec<_>>(),
        reported_losses = a
            .years
            .iter()
            .map(|r| r.reported_losses as i32)
            .collect::<Vec<_>>(),
        trended_losses = a
            .years
            .iter()
            .map(|r| r.trended_losses as i32)
            .collect::<Vec<_>>(),
        historical_layers = List::from_values(a.years.iter().map(|r| r.historical_layers.clone())),
        historical = List::from_values(a.years.iter().map(|r| r.historical.clone())),
        proposed_layers = a.proposed_layers.clone(),
        proposed = proposed
    ))
}

/// A payment pattern, checked: `list(times, shares, duration)`; from
/// `(times, shares)`, or, with `from_development`, from ages and factors to
/// ultimate.
#[extendr]
fn pricing_payment_pattern(times: &[f64], values: &[f64], from_development: bool) -> Result<List> {
    let p = if from_development {
        PaymentPattern::from_development(times, values)
    } else {
        PaymentPattern::new(times.to_vec(), values.to_vec())
    }
    .map_err(to_r)?;
    Ok(list!(
        times = p.times().to_vec(),
        shares = p.shares().to_vec(),
        duration = p.duration()
    ))
}

/// A pattern's discount factor and capital years at `rate`, and the share
/// outstanding at each of `t`.
#[extendr]
fn pricing_pattern_values(times: &[f64], shares: &[f64], rate: f64, t: &[f64]) -> Result<List> {
    let p = PaymentPattern::new(times.to_vec(), shares.to_vec()).map_err(to_r)?;
    Ok(list!(
        discount_factor = p.discount_factor(rate),
        capital_years = p.capital_years(rate),
        outstanding = t.iter().map(|&x| p.outstanding(x)).collect::<Vec<_>>()
    ))
}

/// A one-period price restated over a payout: with a cost of capital
/// `coc`, or, with `coc` NaN, a distortion price discounted with the losses.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn pricing_price_over_payout(
    expected_loss: f64,
    premium: f64,
    assets: f64,
    times: &[f64],
    shares: &[f64],
    rate: f64,
    coc: f64,
) -> Result<List> {
    let p = PaymentPattern::new(times.to_vec(), shares.to_vec()).map_err(to_r)?;
    let price = Price {
        expected_loss,
        premium,
        assets,
    };
    let rule = if coc.is_nan() {
        // Any distortion: only the rule's kind matters here.
        PremiumRule::Distortion(prospicio_prob::Distortion::tvar(0.5).map_err(to_r)?)
    } else {
        PremiumRule::cost_of_capital(coc).map_err(to_r)?
    };
    let q = payout::price_over_payout(&price, &rule, &p, rate).map_err(to_r)?;
    Ok(list!(
        expected_loss = q.expected_loss,
        discounted_loss = q.discounted_loss,
        premium = q.premium,
        assets = q.assets,
        capital = q.capital(),
        margin = q.margin(),
        discount_factor = q.discount_factor,
        capital_years = q.capital_years
    ))
}

/// A development pattern estimated from claim valuations:
/// `list(ages, factors)`.
#[extendr]
fn pricing_development_pattern(
    ids: Vec<String>,
    age: &[f64],
    amount: &[f64],
    ages: &[f64],
) -> Result<List> {
    if ids.len() != age.len() || ids.len() != amount.len() {
        return Err(Error::Other(format!(
            "{} ids, {} ages and {} amounts",
            ids.len(),
            age.len(),
            amount.len()
        )));
    }
    let valuations: Vec<ClaimValuation> = ids
        .into_iter()
        .zip(age)
        .zip(amount)
        .map(|((id, &age), &amount)| ClaimValuation { id, age, amount })
        .collect();
    let p = DevelopmentPattern::from_history(&valuations, ages).map_err(to_r)?;
    Ok(list!(
        ages = p.ages().to_vec(),
        factors = p.factors().to_vec()
    ))
}

/// A pattern's factors to ultimate at each of `x`.
#[extendr]
fn pricing_pattern_at(ages: &[f64], factors: &[f64], x: &[f64]) -> Result<Vec<f64>> {
    let p = DevelopmentPattern::new(ages.to_vec(), factors.to_vec()).map_err(to_r)?;
    Ok(x.iter().map(|&a| p.at(a)).collect())
}

/// Each year's count IBNR factor: the count pattern at the year's average
/// age at `valuation`.
#[extendr]
fn pricing_count_development(
    ages: &[f64],
    factors: &[f64],
    years: &[i32],
    valuation: f64,
) -> Result<Vec<f64>> {
    let p = DevelopmentPattern::new(ages.to_vec(), factors.to_vec()).map_err(to_r)?;
    years
        .iter()
        .map(|&y| {
            YearBasis::from_count_pattern(y, 1.0, &p, valuation)
                .map(|b| b.development)
                .map_err(to_r)
        })
        .collect()
}

/// The trended losses as columns, and `complete_above`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn pricing_trend_large_losses(
    ids: Vec<String>,
    accident: &[f64],
    amount: &[f64],
    threshold: f64,
    rate: f64,
    to: f64,
    factors: &[f64],
    valuation: f64,
    dev_ages: &[f64],
    dev_factors: &[f64],
    dev_per_loss: &[f64],
    fx: &[f64],
) -> Result<List> {
    let p = prepare_large_losses(
        ids,
        accident,
        amount,
        threshold,
        rate,
        to,
        factors,
        valuation,
        dev_ages,
        dev_factors,
        dev_per_loss,
        fx,
    )?;
    let l = p.losses();
    Ok(list!(
        id = l.iter().map(|x| x.id.clone()).collect::<Vec<_>>(),
        year = l.iter().map(|x| x.year).collect::<Vec<_>>(),
        accident = l.iter().map(|x| x.accident).collect::<Vec<_>>(),
        amount = l.iter().map(|x| x.amount).collect::<Vec<_>>(),
        fx = l.iter().map(|x| x.fx).collect::<Vec<_>>(),
        development = l.iter().map(|x| x.development).collect::<Vec<_>>(),
        ultimate = l.iter().map(|x| x.ultimate).collect::<Vec<_>>(),
        factor = l.iter().map(|x| x.factor).collect::<Vec<_>>(),
        trended = l.iter().map(|x| x.trended).collect::<Vec<_>>(),
        complete_above = p.complete_above()
    ))
}

/// The burning cost of trended losses: by year (`layer_loss`, `developed`
/// and `loss_cost` as year-by-layer matrices, filled by column) and overall.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn pricing_burning_cost(
    ids: Vec<String>,
    accident: &[f64],
    amount: &[f64],
    threshold: f64,
    rate: f64,
    to: f64,
    factors: &[f64],
    valuation: f64,
    dev_ages: &[f64],
    dev_factors: &[f64],
    dev_per_loss: &[f64],
    fx: &[f64],
    towers: List,
    years: &[i32],
    premium: &[f64],
    development: &[f64],
    index_franchise: f64,
    index_kind: &str,
    ratios: &[f64],
) -> Result<List> {
    let p = prepare_large_losses(
        ids,
        accident,
        amount,
        threshold,
        rate,
        to,
        factors,
        valuation,
        dev_ages,
        dev_factors,
        dev_per_loss,
        fx,
    )?;
    if premium.len() != years.len() || development.len() != years.len() {
        return Err(Error::Other(format!(
            "{} years, {} premiums and {} development factors",
            years.len(),
            premium.len(),
            development.len()
        )));
    }
    let towers = tower_list(towers)?;
    let basis: Vec<YearBasis> = years
        .iter()
        .zip(premium)
        .zip(development)
        .map(|((&year, &premium), &development)| YearBasis {
            year,
            premium,
            development,
        })
        .collect();
    let b = if index_franchise.is_nan() {
        p.burning_cost(&towers, &basis)
    } else {
        let kind = match index_kind {
            "full" => IndexKind::Full,
            "severe" => IndexKind::Severe,
            other => {
                return Err(Error::Other(format!(
                    "kind must be \"full\" or \"severe\", not {other:?}"
                )));
            }
        };
        let clause = IndexClause::new(index_franchise, kind).map_err(to_r)?;
        p.burning_cost_indexed(&towers, &basis, &clause, ratios)
    }
    .map_err(to_r)?;
    let by_layer = |f: &dyn Fn(&prospicio_pricing::experience::YearCost) -> &Vec<f64>| {
        let layers = b.loss_cost.len();
        (0..layers)
            .flat_map(|i| b.years.iter().map(move |r| f(r)[i]))
            .collect::<Vec<f64>>()
    };
    Ok(list!(
        year = b.years.iter().map(|r| r.year).collect::<Vec<_>>(),
        premium = b.years.iter().map(|r| r.premium).collect::<Vec<_>>(),
        losses = b.years.iter().map(|r| r.losses as i32).collect::<Vec<_>>(),
        development = b.years.iter().map(|r| r.development).collect::<Vec<_>>(),
        layer_loss = by_layer(&|r| &r.layer_loss),
        developed = by_layer(&|r| &r.developed),
        year_loss_cost = by_layer(&|r| &r.loss_cost),
        loss_cost = b.loss_cost.clone(),
        mean_loss_cost = b.mean_loss_cost.clone()
    ))
}

extendr_module! {
    mod pricing;
    fn pricing_trend_large_losses;
    fn pricing_burning_cost;
    fn pricing_development_pattern;
    fn pricing_as_if;
    fn pricing_payment_pattern;
    fn pricing_pattern_values;
    fn pricing_price_over_payout;
    fn pricing_pattern_at;
    fn pricing_count_development;
    fn pricing_quote;
    fn pricing_bounds_rust;
    fn distortion_envelope_rust;
    impl NaturalPortfolio;
    fn pricing_pentagon;
    fn pricing_classical_premium;
    fn pricing_calibrate_classical;
    impl CollectiveModel;
    impl TowerModel;
    impl Mbbefd;
    impl Tabulated;
    impl RiskProfile;
    fn pricing_severity_exposure_curve;
    fn pricing_price;
    fn pricing_price_portfolio;
    fn pricing_ilf;
    fn pricing_layer_cv;
    fn pricing_swiss_re_c_for_mean_rate;
    fn pricing_swiss_re_c_for_layer_share;
    fn pricing_loss_elimination_ratio;
    fn pricing_pareto_extrapolation;
    fn pricing_alpha_between_layers;
    fn pricing_alpha_between_frequency_and_layer;
    fn pricing_alpha_between_frequencies;
}
