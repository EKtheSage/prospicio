//! Distortion risk measures, allocation, copulas and Iman-Conover for R.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_core::StreamRng;
use prospicio_prob::capital::{AllocationMethod, LevelAllocation, LevelMeasure};
use prospicio_prob::copula::{self, Copula};
use prospicio_prob::evt::{Gpd, PotTail};
use prospicio_prob::statistics::{QuantileMethod, StatisticsGrid, TvarMethod};
use prospicio_prob::{
    Archimedean, ArchimedeanCopula, Distortion, Empirical, Family, GaussianCopula, Provenance,
    StudentTCopula,
};

use crate::distributions::{
    Grid, PredictiveDistribution, Sampled, components_from_keys, key_from_list, severity_from_robj,
};
use crate::{to_r, whole};

/// A distortion risk measure.
#[extendr]
pub(crate) struct RiskDistortion {
    pub(crate) inner: Distortion,
}

#[extendr]
impl RiskDistortion {
    /// `kind` is "tvar", "wang", "proportional_hazard", "dual_power",
    /// "exponential", "ccoc", "bitvar" (`params` p0, p1, w),
    /// "weighted_tvar" (`params` the levels, `weights` their weights),
    /// "capped_linear", "capped_log_linear", "lep", "linear_yield" (with
    /// `r0`) or "beta" (`params` a, b).
    fn new(kind: &str, params: &[f64], weights: &[f64], r0: f64) -> Result<Self> {
        let one = || -> Result<f64> {
            match params {
                [x] => Ok(*x),
                _ => Err(Error::Other(format!(
                    "a {kind} distortion takes one parameter"
                ))),
            }
        };
        let inner = match kind {
            "tvar" => Distortion::tvar(one()?),
            "wang" => Distortion::wang(one()?),
            "proportional_hazard" => Distortion::proportional_hazard(one()?),
            "dual_power" => Distortion::dual_power(one()?),
            "exponential" => Distortion::exponential(one()?),
            "ccoc" => Distortion::ccoc(one()?),
            "capped_linear" => Distortion::capped_linear(r0, one()?),
            "capped_log_linear" => Distortion::capped_log_linear(r0, one()?),
            "lep" => Distortion::lep(r0, one()?),
            "linear_yield" => Distortion::linear_yield(r0, one()?),
            "bitvar" => match params {
                [p0, p1, w] => Distortion::bitvar(*p0, *p1, *w),
                _ => {
                    return Err(Error::Other(
                        "a bitvar distortion takes c(p0, p1, w)".into(),
                    ));
                }
            },
            "beta" => match params {
                [a, b] => Distortion::beta(*a, *b),
                _ => return Err(Error::Other("a beta distortion takes c(a, b)".into())),
            },
            "weighted_tvar" => Distortion::weighted_tvar(params.to_vec(), weights.to_vec()),
            other => return Err(Error::Other(format!("unknown distortion {other:?}"))),
        }
        .map_err(to_r)?;
        Ok(Self { inner })
    }

    /// A weighted average of distortions (a list of `RiskDistortion`).
    fn mixture(parts: List, weights: &[f64]) -> Result<Self> {
        let ds = distortions_of(&parts)?;
        if ds.len() != weights.len() {
            return Err(Error::Other("give one weight per distortion".into()));
        }
        let inner = Distortion::mixture(weights.iter().copied().zip(ds).collect()).map_err(to_r)?;
        Ok(Self { inner })
    }

    /// The pointwise minimum of distortions.
    fn minimum(parts: List) -> Result<Self> {
        let inner = Distortion::minimum(distortions_of(&parts)?).map_err(to_r)?;
        Ok(Self { inner })
    }

    /// The smallest concave distortion above the points `(s, g)`.
    fn convex(s: &[f64], g: &[f64]) -> Result<Self> {
        if s.len() != g.len() {
            return Err(Error::Other("s and g must have the same length".into()));
        }
        let pts: Vec<(f64, f64)> = s.iter().copied().zip(g.iter().copied()).collect();
        Ok(Self {
            inner: Distortion::convex(&pts).map_err(to_r)?,
        })
    }

    /// The member of `family` whose price of `x`, capped at `assets` when
    /// it is finite, is `premium`.
    fn calibrate(family: &str, x: Robj, premium: f64, assets: f64, r0: f64) -> Result<Self> {
        let family = family_from(family, r0)?;
        let (mut v, p) = discrete_of(&x)?;
        if assets.is_finite() {
            v.iter_mut().for_each(|x| *x = x.min(assets));
        }
        let inner = prospicio_prob::distortion::calibrate(family, &v, &p, premium).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn kind(&self) -> &'static str {
        self.inner.family()
    }

    /// The parameters: one for the one-parameter kinds (the slope, `b` or
    /// `r` of the kinds with an `r0`), `c(p0, p1, w)` for bitvar, the
    /// levels for weighted_tvar, `c(a, b)` for beta, the weights of a
    /// mixture, and the knots' `s` for convex.
    fn param(&self) -> Vec<f64> {
        match &self.inner {
            Distortion::Tvar(a)
            | Distortion::Wang(a)
            | Distortion::ProportionalHazard(a)
            | Distortion::DualPower(a)
            | Distortion::Exponential(a)
            | Distortion::Ccoc(a) => vec![*a],
            Distortion::BiTvar { p0, p1, w } => vec![*p0, *p1, *w],
            Distortion::WeightedTvar { ps, .. } => ps.clone(),
            Distortion::CappedLinear { slope, .. } => vec![*slope],
            Distortion::CappedLogLinear { b, .. } => vec![*b],
            Distortion::Lep { r, .. } | Distortion::LinearYield { r, .. } => vec![*r],
            Distortion::Beta { a, b } => vec![*a, *b],
            Distortion::Mixture(parts) => parts.iter().map(|(w, _)| *w).collect(),
            Distortion::Minimum(_) => vec![],
            Distortion::Convex(knots) => knots.iter().map(|k| k.0).collect(),
        }
    }

    /// The scalar parameters by name, as the constructor takes them.
    fn parameters(&self) -> List {
        let (names, values): (Vec<&str>, Vec<f64>) = self.inner.parameters().into_iter().unzip();
        List::from_names_and_values(names, values.into_iter().map(Robj::from)).unwrap_or_default()
    }

    /// The probability mass on the largest outcome, `g(0+)`.
    fn mass(&self) -> f64 {
        self.inner.mass()
    }

    fn g_inv(&self, y: &[f64]) -> Vec<f64> {
        y.iter().map(|&y| self.inner.g_inv(y)).collect()
    }

    fn g_dual(&self, s: &[f64]) -> Vec<f64> {
        s.iter().map(|&s| self.inner.g_dual(s)).collect()
    }

    fn g(&self, s: &[f64]) -> Vec<f64> {
        s.iter().map(|&s| self.inner.g(s)).collect()
    }

    fn weights(&self, n: f64) -> Result<Vec<f64>> {
        Ok(self.inner.weights(whole(n, "n")? as usize))
    }

    /// Risk measure of a sampled, grid or predictive distribution (its total).
    fn measure(&self, x: Robj) -> Result<f64> {
        if let Ok(s) = <&Sampled>::try_from(&x) {
            return Ok(s.inner.distortion(&self.inner));
        }
        if let Ok(g) = <&Grid>::try_from(&x) {
            return Ok(g.inner.distortion(&self.inner));
        }
        if let Ok(p) = <&PredictiveDistribution>::try_from(&x) {
            return Ok(p.inner.distortion(&self.inner));
        }
        Err(Error::Other(
            "expected a sampled, grid_distribution or predictive_distribution".into(),
        ))
    }

    /// Allocation by `method` ("euler", "covariance", "covariance_margin", "proportional",
    /// "marginal" or "shapley"): `list(total, standalone, allocated)`.
    fn capital(&self, pd: Robj, method: &str) -> Result<List> {
        let pd = <&PredictiveDistribution>::try_from(&pd)
            .map_err(|_| Error::Other("expected a predictive_distribution".into()))?;
        let method = match method {
            "euler" => AllocationMethod::Euler,
            "covariance" => AllocationMethod::Covariance,
            "covariance_margin" => AllocationMethod::CovarianceMargin,
            "proportional" => AllocationMethod::Proportional,
            "marginal" => AllocationMethod::Marginal,
            "shapley" => AllocationMethod::Shapley,
            other => {
                return Err(Error::Other(format!(
                    "method must be euler, covariance, covariance_margin, proportional, marginal or \
                     shapley, got {other}"
                )));
            }
        };
        let a = pd.inner.capital(&self.inner, method).map_err(to_r)?;
        Ok(list!(
            total = a.total,
            standalone = a.standalone,
            allocated = a.allocated
        ))
    }

    /// Co-measure allocation, one contribution per component.
    fn allocate(&self, pd: Robj) -> Result<Vec<f64>> {
        let pd = <&PredictiveDistribution>::try_from(&pd)
            .map_err(|_| Error::Other("expected a predictive_distribution".into()))?;
        Ok(pd.inner.allocate(&self.inner))
    }
}

/// A distortion family by name, with the fixed `r0` of those that have one.
pub(crate) fn family_from(name: &str, r0: f64) -> Result<Family> {
    Ok(match name {
        "ccoc" => Family::Ccoc,
        "proportional_hazard" | "ph" => Family::ProportionalHazard,
        "wang" => Family::Wang,
        "dual_power" | "dual" => Family::DualPower,
        "tvar" => Family::Tvar,
        "exponential" | "exp" => Family::Exponential,
        "capped_linear" | "clin" => Family::CappedLinear { r0 },
        "capped_log_linear" | "cll" => Family::CappedLogLinear { r0 },
        "lep" => Family::Lep { r0 },
        "linear_yield" | "ly" => Family::LinearYield { r0 },
        other => return Err(Error::Other(format!("unknown distortion family {other:?}"))),
    })
}

fn distortions_of(parts: &List) -> Result<Vec<Distortion>> {
    parts
        .values()
        .map(|d| {
            <&RiskDistortion>::try_from(&d)
                .map(|d| d.inner.clone())
                .map_err(|_| Error::Other("expected a list of distortions".into()))
        })
        .collect()
}

/// The values, ascending, and their probabilities, of a sampled, grid or
/// predictive distribution (its total).
pub(crate) fn discrete_of(x: &Robj) -> Result<(Vec<f64>, Vec<f64>)> {
    let equal = |v: &[f64]| {
        let n = v.len() as f64;
        (v.to_vec(), vec![1.0 / n; v.len()])
    };
    if let Ok(s) = <&Sampled>::try_from(x) {
        return Ok(equal(s.inner.sorted()));
    }
    if let Ok(g) = <&Grid>::try_from(x) {
        let v = (0..g.inner.len()).map(|j| g.inner.x(j)).collect();
        return Ok((v, g.inner.probs().to_vec()));
    }
    if let Ok(p) = <&PredictiveDistribution>::try_from(x) {
        return Ok(equal(p.inner.total().sorted()));
    }
    Err(Error::Other(
        "expected a sampled, grid_distribution or predictive_distribution".into(),
    ))
}

fn as_predictive(pd: &Robj) -> Result<&PredictiveDistribution> {
    <&PredictiveDistribution>::try_from(pd)
        .map_err(|_| Error::Other("expected a predictive_distribution".into()))
}

/// A quantile method by name: "inverted_cdf" or "linear".
pub(crate) fn quantile_method(name: &str) -> Result<QuantileMethod> {
    match name {
        "inverted_cdf" => Ok(QuantileMethod::InvertedCdf),
        "linear" => Ok(QuantileMethod::Linear),
        _ => Err(Error::Other(format!(
            "quantile method must be \"inverted_cdf\" or \"linear\", not {name:?}"
        ))),
    }
}

/// A TVaR method by name: "exact" or "linear".
pub(crate) fn tvar_method(name: &str) -> Result<TvarMethod> {
    match name {
        "exact" => Ok(TvarMethod::Exact),
        "linear" => Ok(TvarMethod::Linear),
        _ => Err(Error::Other(format!(
            "TVaR method must be \"exact\" or \"linear\", not {name:?}"
        ))),
    }
}

/// A statistics table of draws: `names` and `values`, levels in percent.
#[extendr]
fn statistics_rust(
    draws: &[f64],
    percentiles: &[f64],
    tvar_levels: &[f64],
    sd_multiples: &[f64],
    quantile: &str,
    tvar: &str,
) -> Result<List> {
    let grid = StatisticsGrid::new(percentiles.to_vec())
        .and_then(|g| g.tvar_levels(tvar_levels.to_vec()))
        .and_then(|g| g.sd_multiples(sd_multiples.to_vec()))
        .map_err(to_r)?
        .quantile_method(quantile_method(quantile)?)
        .tvar_method(tvar_method(tvar)?);
    let sampled = prospicio_prob::Sampled::new(draws.to_vec()).map_err(to_r)?;
    let st = grid.of(&sampled);
    Ok(list!(names = st.names, values = st.values))
}

/// Bootstrap standard errors of the statistics of `k` columns of `n`
/// simulations (`columns` column-major), one draw per replicate shared by
/// every column; `fixed` NULL, or the fixed part of each column in the same
/// layout. `names`, and `values` a list with each column's errors.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn bootstrap_se_rust(
    columns: &[f64],
    k: f64,
    fixed: Nullable<Vec<f64>>,
    percentiles: &[f64],
    tvar_levels: &[f64],
    sd_multiples: &[f64],
    quantile: &str,
    tvar: &str,
    replicates: f64,
    seed: f64,
) -> Result<List> {
    let grid = StatisticsGrid::new(percentiles.to_vec())
        .and_then(|g| g.tvar_levels(tvar_levels.to_vec()))
        .and_then(|g| g.sd_multiples(sd_multiples.to_vec()))
        .map_err(to_r)?
        .quantile_method(quantile_method(quantile)?)
        .tvar_method(tvar_method(tvar)?);
    let k = k as usize;
    if k == 0 || columns.len() % k != 0 {
        return Err(Error::Other(
            "the columns must all have the same length".into(),
        ));
    }
    let n = columns.len() / k;
    let refs: Vec<&[f64]> = columns.chunks(n.max(1)).collect();
    if !replicates.is_finite() || replicates < 0.0 || !seed.is_finite() || seed < 0.0 {
        return Err(Error::Other(
            "replicates and seed must be non-negative".into(),
        ));
    }
    let st = match fixed {
        Nullable::Null => grid.bootstrap_se(&refs, replicates as usize, seed as u64),
        Nullable::NotNull(f) => {
            if f.len() != columns.len() {
                return Err(Error::Other(format!(
                    "hold_fixed has {} values for {} in the columns",
                    f.len(),
                    columns.len()
                )));
            }
            let held: Vec<&[f64]> = f.chunks(n.max(1)).collect();
            grid.bootstrap_se_holding(&refs, &held, replicates as usize, seed as u64)
        }
    }
    .map_err(to_r)?;
    let names = grid.names();
    let values: Vec<Vec<f64>> = st.into_iter().map(|s| s.values).collect();
    Ok(list!(names = names, values = List::from_values(values)))
}

/// Items (an `n × k` matrix, column-major) read at the driver's mean,
/// percentiles and TVaRs: `names` and `values`, item by item.
#[extendr]
fn co_statistics_rust(
    driver: &[f64],
    items: &[f64],
    percentiles: &[f64],
    tvar_levels: &[f64],
    rule: &str,
) -> Result<List> {
    use prospicio_prob::statistics::IndexRule;
    let rule = match rule {
        "round_half_even" => IndexRule::RoundHalfEven,
        "floor" => IndexRule::Floor,
        "ceil" => IndexRule::Ceil,
        _ => {
            return Err(Error::Other(format!(
                "rule must be \"round_half_even\", \"floor\" or \"ceil\", not {rule:?}"
            )));
        }
    };
    let n = driver.len();
    if n == 0 || items.len() % n != 0 {
        return Err(Error::Other(
            "items must have one row per simulation of the driver".into(),
        ));
    }
    let grid = StatisticsGrid::new(percentiles.to_vec())
        .and_then(|g| g.tvar_levels(tvar_levels.to_vec()))
        .map_err(to_r)?;
    let columns: Vec<&[f64]> = items.chunks(n).collect();
    let stats = grid.co_statistics(driver, &columns, rule).map_err(to_r)?;
    let values: Vec<f64> = stats.iter().flat_map(|st| st.values.clone()).collect();
    Ok(list!(names = grid.co_names(), values = values))
}

/// Risk ratios of draws: `names` and `values`.
#[extendr]
fn risk_ratios_rust(
    draws: &[f64],
    threshold: f64,
    sortino: &str,
    omega: &str,
    tvar_levels: &[f64],
    tvar: &str,
) -> Result<List> {
    use prospicio_prob::statistics::{Omega, RiskRatios, Sortino};
    let sortino = match sortino {
        "downside_deviation" => Sortino::DownsideDeviation,
        "below_threshold_sd" => Sortino::BelowThresholdSd,
        _ => {
            return Err(Error::Other(format!(
                "sortino must be \"downside_deviation\" or \"below_threshold_sd\", not {sortino:?}"
            )));
        }
    };
    let omega = match omega {
        "partial_moments" => Omega::PartialMoments,
        "count" => Omega::Count,
        _ => {
            return Err(Error::Other(format!(
                "omega must be \"partial_moments\" or \"count\", not {omega:?}"
            )));
        }
    };
    let ratios = RiskRatios::new()
        .threshold(threshold)
        .and_then(|r| r.tvar_levels(tvar_levels.to_vec()))
        .map_err(to_r)?
        .sortino(sortino)
        .omega(omega)
        .tvar_method(tvar_method(tvar)?);
    let sampled = prospicio_prob::Sampled::new(draws.to_vec()).map_err(to_r)?;
    let st = ratios.of(&sampled);
    Ok(list!(names = st.names, values = st.values))
}

/// Entropic risk measure of draws at risk aversion `theta`.
#[extendr]
fn entropic_rust(draws: &[f64], theta: f64) -> Result<f64> {
    prospicio_prob::risk::entropic(draws, theta).map_err(to_r)
}

/// Esscher premium of draws at `h`.
#[extendr]
fn esscher_rust(draws: &[f64], h: f64) -> Result<f64> {
    prospicio_prob::risk::esscher(draws, h).map_err(to_r)
}

/// Marginal expected shortfall of each component at level `p`.
#[extendr]
fn mes_rust(pd: Robj, p: f64) -> Result<Vec<f64>> {
    as_predictive(&pd)?
        .inner
        .marginal_expected_shortfall(p)
        .map_err(to_r)
}

/// CoVaR of the component `key`.
#[extendr]
fn covar_rust(pd: Robj, key: List, p: f64, q: f64) -> Result<f64> {
    let key = key_from_list(key)?;
    as_predictive(&pd)?.inner.covar(&key, p, q).map_err(to_r)
}

/// Esscher allocation of the total at `h`, one value per component.
#[extendr]
fn esscher_allocation_rust(pd: Robj, h: f64) -> Result<Vec<f64>> {
    as_predictive(&pd)?
        .inner
        .esscher_allocation(h)
        .map_err(to_r)
}

/// Shapley value of the game whose cost is `cost` (an R function of the
/// sub-portfolio's draws): `list(total, standalone, allocated)`.
#[extendr]
fn shapley_rust(pd: Robj, cost: Function) -> Result<List> {
    // An R error inside `cost` becomes NaN for the Rust side, which
    // rejects it; the original error is returned instead.
    let mut failure = None;
    let result = as_predictive(&pd)?.inner.shapley_with(|x| {
        if failure.is_some() {
            return f64::NAN;
        }
        let value = cost.call(pairlist!(x.to_vec())).and_then(|v| {
            v.as_real()
                .ok_or(Error::Other("cost must return a number".into()))
        });
        match value {
            Ok(v) => v,
            Err(e) => {
                failure.get_or_insert(e);
                f64::NAN
            }
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }
    let a = result.map_err(to_r)?;
    Ok(list!(
        total = a.total,
        standalone = a.standalone,
        allocated = a.allocated
    ))
}

fn level_measure(name: &str) -> Result<LevelMeasure> {
    Ok(match name {
        "var" => LevelMeasure::Var,
        "tvar" => LevelMeasure::Tvar,
        "epd" => LevelMeasure::Epd,
        other => {
            return Err(Error::Other(format!(
                "measure must be var, tvar or epd, got {other}"
            )));
        }
    })
}

fn level_list(a: LevelAllocation) -> List {
    list!(
        level = a.level,
        total = a.total,
        standalone = a.standalone,
        allocated = a.allocated
    )
}

/// Haircut allocation of `total` by stand-alone `measure` at `level`.
#[extendr]
fn haircut_rust(pd: Robj, total: f64, measure: &str, level: f64) -> Result<List> {
    let a = as_predictive(&pd)?
        .inner
        .haircut(total, level_measure(measure)?, level)
        .map_err(to_r)?;
    Ok(level_list(a))
}

/// Equal risk allocation of `total` by stand-alone `measure`.
#[extendr]
fn equal_risk_rust(pd: Robj, total: f64, measure: &str) -> Result<List> {
    let a = as_predictive(&pd)?
        .inner
        .equal_risk(total, level_measure(measure)?)
        .map_err(to_r)?;
    Ok(level_list(a))
}

/// The Rearrangement Algorithm on the columns of a column-major matrix:
/// `list(value, columns, iterations)`; `rotate` and `seed` NaN for none.
#[extendr]
fn rearrange_rust(x: &[f64], ncol: f64, rotate: f64, seed: f64, tolerance: f64) -> Result<List> {
    use prospicio_prob::rearrangement::{Start, rearrange};
    let m = ncol as usize;
    if m == 0 || x.len() % m != 0 {
        return Err(Error::Other("x must be a matrix with ncol columns".into()));
    }
    let n = x.len() / m;
    let columns: Vec<Vec<f64>> = x.chunks(n).map(<[f64]>::to_vec).collect();
    let start = match (rotate.is_nan(), seed.is_nan()) {
        (false, false) => return Err(Error::Other("give rotate or seed, not both".into())),
        (false, true) => Start::Rotated(rotate as usize),
        (true, false) => Start::Shuffled(seed as u64),
        (true, true) => Start::AsGiven,
    };
    let r = rearrange(&columns, start, tolerance).map_err(to_r)?;
    Ok(list!(
        value = r.value,
        columns = r.columns.concat(),
        iterations = r.iterations as f64
    ))
}

pub(crate) enum AnyCopula {
    Gaussian(GaussianCopula),
    StudentT(StudentTCopula),
    Archimedean(ArchimedeanCopula),
    Nested(prospicio_prob::NestedArchimedeanCopula),
    Vine(prospicio_prob::vine::VineCopula),
}

impl AnyCopula {
    pub(crate) fn as_copula(&self) -> &dyn Copula {
        match self {
            Self::Gaussian(c) => c,
            Self::StudentT(c) => c,
            Self::Archimedean(c) => c,
            Self::Nested(c) => c,
            Self::Vine(c) => c,
        }
    }
}

/// A pair copula from `list(family, parameter, rotation)`.
fn pair_copula(p: &Robj) -> Result<prospicio_prob::vine::PairCopula> {
    use prospicio_prob::vine::{PairCopula, PairFamily};
    let bad = || Error::Other("a pair copula is list(family, parameter, rotation)".into());
    let l = List::try_from(p).map_err(|_| bad())?;
    let parts: Vec<Robj> = l.values().collect();
    if parts.len() != 3 {
        return Err(bad());
    }
    let family = match parts[0].as_str().ok_or_else(bad)? {
        "independence" => PairFamily::Independence,
        "gaussian" => PairFamily::Gaussian,
        "clayton" => PairFamily::Clayton,
        "gumbel" => PairFamily::Gumbel,
        "frank" => PairFamily::Frank,
        "joe" => PairFamily::Joe,
        other => return Err(Error::Other(format!("unknown pair family {other:?}"))),
    };
    let parameter = parts[1].as_real().ok_or_else(bad)?;
    let rotation = parts[2].as_real().ok_or_else(bad)?;
    if !matches!(rotation, 0.0 | 90.0 | 180.0 | 270.0) {
        return Err(Error::Other(format!(
            "rotation {rotation} is not 0, 90, 180 or 270"
        )));
    }
    PairCopula::new(family, parameter, rotation as u16).map_err(to_r)
}

/// A node `list(theta, leaves, children)` with 1-based leaves.
fn nested_node(node: &Robj) -> Result<prospicio_prob::NestedNode> {
    let bad = || Error::Other("a node is list(theta, leaves, children)".into());
    let l = List::try_from(node).map_err(|_| bad())?;
    if l.len() != 3 {
        return Err(bad());
    }
    let parts: Vec<Robj> = l.values().collect();
    let theta = parts[0]
        .as_real()
        .ok_or_else(|| Error::Other("theta must be a number".into()))?;
    let leaves = parts[1]
        .as_real_vector()
        .unwrap_or_default()
        .into_iter()
        .map(|j| {
            if j >= 1.0 && j.fract() == 0.0 {
                Ok(j as usize - 1)
            } else {
                Err(Error::Other(format!(
                    "leaf {j} is not a dimension (1, 2, ...)"
                )))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let children = List::try_from(&parts[2])
        .map_err(|_| Error::Other("children must be a list of nodes".into()))?
        .values()
        .map(|c| nested_node(&c))
        .collect::<Result<Vec<_>>>()?;
    Ok(prospicio_prob::NestedNode::new(theta, leaves, children))
}

/// A copula.
#[extendr]
pub(crate) struct RiskCopula {
    pub(crate) inner: AnyCopula,
}

#[extendr]
impl RiskCopula {
    /// `correlation` is a `dim × dim` matrix (symmetric, so R's
    /// column-major order is row-major too).
    fn gaussian(correlation: &[f64], dim: f64) -> Result<Self> {
        let dim = whole(dim, "dim")? as usize;
        let c = GaussianCopula::new(correlation, dim).map_err(to_r)?;
        Ok(Self {
            inner: AnyCopula::Gaussian(c),
        })
    }

    fn student_t(correlation: &[f64], dim: f64, nu: f64) -> Result<Self> {
        let dim = whole(dim, "dim")? as usize;
        let c = StudentTCopula::new(correlation, dim, nu).map_err(to_r)?;
        Ok(Self {
            inner: AnyCopula::StudentT(c),
        })
    }

    fn archimedean(family: &str, theta: f64, dim: f64) -> Result<Self> {
        let family = match family {
            "clayton" => Archimedean::Clayton,
            "gumbel" => Archimedean::Gumbel,
            "frank" => Archimedean::Frank,
            "joe" => Archimedean::Joe,
            other => return Err(Error::Other(format!("unknown family {other:?}"))),
        };
        let dim = whole(dim, "dim")? as usize;
        let c = ArchimedeanCopula::new(family, theta, dim).map_err(to_r)?;
        Ok(Self {
            inner: AnyCopula::Archimedean(c),
        })
    }

    /// A nested Archimedean copula from its root node.
    fn nested(family: &str, root: Robj) -> Result<Self> {
        let family = match family {
            "clayton" => Archimedean::Clayton,
            "gumbel" => Archimedean::Gumbel,
            other => {
                return Err(Error::Other(format!(
                    "a nested copula is \"clayton\" or \"gumbel\", not {other:?}"
                )));
            }
        };
        let c = prospicio_prob::NestedArchimedeanCopula::new(family, nested_node(&root)?)
            .map_err(to_r)?;
        Ok(Self {
            inner: AnyCopula::Nested(c),
        })
    }

    /// A C- or D-vine: `order` 1-based, `pairs` a list of trees, each a
    /// list of `list(family, parameter, rotation)`.
    fn vine(kind: &str, order: &[f64], pairs: List) -> Result<Self> {
        use prospicio_prob::vine::{VineCopula, VineKind};
        let kind = match kind {
            "c" => VineKind::Canonical,
            "d" => VineKind::Drawable,
            other => {
                return Err(Error::Other(format!(
                    "kind must be \"c\" or \"d\", not {other:?}"
                )));
            }
        };
        let order = order
            .iter()
            .map(|&j| {
                if j >= 1.0 && j.fract() == 0.0 {
                    Ok(j as usize - 1)
                } else {
                    Err(Error::Other(format!("{j} is not a dimension (1, 2, ...)")))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let pairs = pairs
            .values()
            .map(|tree| {
                List::try_from(&tree)
                    .map_err(|_| Error::Other("each tree is a list of pair copulas".into()))?
                    .values()
                    .map(|p| pair_copula(&p))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let c = VineCopula::new(kind, order, pairs).map_err(to_r)?;
        Ok(Self {
            inner: AnyCopula::Vine(c),
        })
    }

    /// The vine's log-density at each row of `u` (`n × dim`, column-major).
    fn log_density(&self, u: &[f64], n: f64) -> Result<Vec<f64>> {
        let AnyCopula::Vine(v) = &self.inner else {
            return Err(Error::Other(
                "only a vine copula has a log-density here".into(),
            ));
        };
        let n = whole(n, "n")? as usize;
        let d = v.dim();
        if u.len() != n * d {
            return Err(Error::Other(format!(
                "{} values for {n} points in {d} dimensions",
                u.len()
            )));
        }
        (0..n)
            .map(|i| {
                let row: Vec<f64> = (0..d).map(|j| u[j * n + i]).collect();
                v.log_density(&row).map_err(to_r)
            })
            .collect()
    }

    fn dim(&self) -> f64 {
        self.inner.as_copula().dim() as f64
    }

    /// The cdf at each row of `u`, an `n × dim` matrix in R's column-major
    /// order.
    fn cdf(&self, u: &[f64], n: f64) -> Result<Vec<f64>> {
        let n = whole(n, "n")? as usize;
        let c = self.inner.as_copula();
        let d = c.dim();
        if u.len() != n * d {
            return Err(Error::Other(format!(
                "{} values for {n} points in {d} dimensions",
                u.len()
            )));
        }
        (0..n)
            .map(|i| {
                let point: Vec<f64> = (0..d).map(|j| u[i + j * n]).collect();
                c.cdf(&point).map_err(to_r)
            })
            .collect()
    }

    fn description(&self) -> String {
        match &self.inner {
            AnyCopula::Gaussian(_) => "Gaussian".into(),
            AnyCopula::StudentT(c) => format!("Student t, nu = {}", c.nu()),
            AnyCopula::Archimedean(c) => format!("{:?}, theta = {}", c.family(), c.theta()),
            AnyCopula::Nested(c) => format!("nested {:?}, theta = {}", c.family(), c.root().theta),
            AnyCopula::Vine(c) => format!(
                "{}-vine on {} trees",
                match c.kind() {
                    prospicio_prob::vine::VineKind::Canonical => "C",
                    prospicio_prob::vine::VineKind::Drawable => "D",
                },
                c.pairs().len()
            ),
        }
    }

    /// `n × dim` uniforms in R's column-major order; row `i` uses stream
    /// `i - 1` of `seed`.
    fn sample(&self, n: f64, seed: f64) -> Result<Vec<f64>> {
        let n = whole(n, "n")? as usize;
        let seed = whole(seed, "seed")?;
        let c = self.inner.as_copula();
        let d = c.dim();
        let mut out = vec![0.0; n * d];
        let mut u = vec![0.0; d];
        for i in 0..n {
            c.sample(&mut StreamRng::new(seed, i as u64), &mut u);
            for (j, x) in u.iter().enumerate() {
                out[i + j * n] = *x;
            }
        }
        Ok(out)
    }

    fn simulate(
        &self,
        marginals: List,
        n_sims: f64,
        seed: f64,
        dims: Vec<String>,
        keys: List,
    ) -> Result<PredictiveDistribution> {
        let marginals: Vec<prospicio_prob::SeverityDist> = marginals
            .values()
            .map(|m| severity_from_robj(&m))
            .collect::<Result<_>>()?;
        let refs: Vec<&(dyn prospicio_prob::Distribution + Sync)> = marginals
            .iter()
            .map(|m| m as &(dyn prospicio_prob::Distribution + Sync))
            .collect();
        let components = components_from_keys(&dims, keys, marginals.len())?;
        let inner = copula::simulate(
            self.inner.as_copula(),
            &refs,
            dims,
            components,
            whole(n_sims, "n_sims")? as usize,
            whole(seed, "seed")?,
            Provenance::new("copula"),
        )
        .map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

/// Iman-Conover reordering of a predictive distribution's components:
/// `dof` NaN and `scores` empty for normal scores.
#[extendr]
fn iman_conover_reorder(
    pd: Robj,
    correlation: &[f64],
    seed: f64,
    dof: f64,
    scores: &[f64],
    shuffle_rows: bool,
) -> Result<PredictiveDistribution> {
    let pd = <&PredictiveDistribution>::try_from(&pd)
        .map_err(|_| Error::Other("expected a predictive_distribution".into()))?;
    let scores = match (dof.is_nan(), scores.is_empty()) {
        (true, true) => copula::Scores::Normal,
        (false, true) => copula::Scores::StudentT(dof),
        (true, false) => copula::Scores::Given(scores.to_vec()),
        (false, false) => return Err(Error::Other("give dof or scores, not both".into())),
    };
    let options = copula::ImanConover {
        scores,
        shuffle_rows,
    };
    let inner = copula::iman_conover_with(&pd.inner, correlation, whole(seed, "seed")?, &options)
        .map_err(to_r)?;
    Ok(PredictiveDistribution { inner })
}

/// A peaks-over-threshold tail.
#[extendr]
pub(crate) struct EvtTail {
    inner: PotTail,
}

#[extendr]
impl EvtTail {
    fn fit(draws: Robj, level: f64) -> Result<Self> {
        let s = <&Sampled>::try_from(&draws)
            .map_err(|_| Error::Other("draws must be a sampled object".into()))?;
        let inner = PotTail::fit(&s.inner, level).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn threshold(&self) -> f64 {
        self.inner.threshold()
    }

    fn p_exceed(&self) -> f64 {
        self.inner.p_exceed()
    }

    fn xi(&self) -> f64 {
        self.inner.gpd().xi()
    }

    fn beta(&self) -> f64 {
        self.inner.gpd().beta()
    }

    fn var(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter().map(|&p| self.inner.var(p).map_err(to_r)).collect()
    }

    fn tvar(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.tvar(p).map_err(to_r))
            .collect()
    }
}

/// Maximum likelihood GPD fit: `c(xi, beta)`.
#[extendr]
fn gpd_mle(exceedances: &[f64]) -> Result<Vec<f64>> {
    let g = Gpd::fit(exceedances).map_err(to_r)?;
    Ok(vec![g.xi(), g.beta()])
}

/// Mean excess at each threshold: `list(threshold, mean_excess, n_above)`.
#[extendr]
fn mean_excess_rust(draws: &[f64], thresholds: &[f64]) -> List {
    let rows = prospicio_prob::evt::mean_excess(draws, thresholds);
    list!(
        threshold = rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        mean_excess = rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        n_above = rows.iter().map(|r| r.2 as f64).collect::<Vec<_>>()
    )
}

/// Threshold diagnostics at each threshold: `list(threshold, n_above,
/// mean_excess, mean_excess_se, xi, xi_se, beta, modified_scale,
/// modified_scale_se)`.
#[extendr]
fn threshold_diagnostics_rust(draws: &[f64], thresholds: &[f64]) -> List {
    let rows = prospicio_prob::evt::threshold_diagnostics(draws, thresholds);
    let col = |f: fn(&prospicio_prob::evt::ThresholdDiagnostic) -> f64| {
        rows.iter().map(f).collect::<Vec<_>>()
    };
    list!(
        threshold = col(|r| r.threshold),
        n_above = col(|r| r.n_above as f64),
        mean_excess = col(|r| r.mean_excess),
        mean_excess_se = col(|r| r.mean_excess_se),
        xi = col(|r| r.xi),
        xi_se = col(|r| r.xi_se),
        beta = col(|r| r.beta),
        modified_scale = col(|r| r.modified_scale),
        modified_scale_se = col(|r| r.modified_scale_se)
    )
}

/// Hill estimates of the tail index for each `k`.
#[extendr]
fn hill_rust(draws: &[f64], ks: &[f64]) -> Result<Vec<f64>> {
    let ks = ks
        .iter()
        .map(|&k| whole(k, "k").map(|k| k as usize))
        .collect::<Result<Vec<_>>>()?;
    prospicio_prob::evt::hill(draws, &ks).map_err(to_r)
}

extendr_module! {
    mod risk;
    fn statistics_rust;
    fn bootstrap_se_rust;
    fn co_statistics_rust;
    fn risk_ratios_rust;
    fn entropic_rust;
    fn esscher_rust;
    fn mes_rust;
    fn covar_rust;
    fn esscher_allocation_rust;
    fn shapley_rust;
    fn haircut_rust;
    fn equal_risk_rust;
    fn rearrange_rust;
    fn iman_conover_reorder;
    fn gpd_mle;
    fn mean_excess_rust;
    fn hill_rust;
    fn threshold_diagnostics_rust;
    impl EvtTail;
    impl RiskDistortion;
    impl RiskCopula;
}
