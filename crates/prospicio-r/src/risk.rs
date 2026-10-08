//! Distortion risk measures, allocation, copulas and Iman-Conover for R.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_core::StreamRng;
use prospicio_prob::capital::AllocationMethod;
use prospicio_prob::copula::{self, Copula};
use prospicio_prob::evt::{Gpd, PotTail};
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
        match &self.inner {
            Distortion::Tvar(_) => "tvar",
            Distortion::Wang(_) => "wang",
            Distortion::ProportionalHazard(_) => "proportional_hazard",
            Distortion::DualPower(_) => "dual_power",
            Distortion::Exponential(_) => "exponential",
            Distortion::Ccoc(_) => "ccoc",
            Distortion::BiTvar { .. } => "bitvar",
            Distortion::WeightedTvar { .. } => "weighted_tvar",
            Distortion::CappedLinear { .. } => "capped_linear",
            Distortion::CappedLogLinear { .. } => "capped_log_linear",
            Distortion::Lep { .. } => "lep",
            Distortion::LinearYield { .. } => "linear_yield",
            Distortion::Beta { .. } => "beta",
            Distortion::Mixture(_) => "mixture",
            Distortion::Minimum(_) => "minimum",
            Distortion::Convex(_) => "convex",
        }
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

    /// Allocation by `method` ("euler", "covariance", "proportional",
    /// "marginal" or "shapley"): `list(total, standalone, allocated)`.
    fn capital(&self, pd: Robj, method: &str) -> Result<List> {
        let pd = <&PredictiveDistribution>::try_from(&pd)
            .map_err(|_| Error::Other("expected a predictive_distribution".into()))?;
        let method = match method {
            "euler" => AllocationMethod::Euler,
            "covariance" => AllocationMethod::Covariance,
            "proportional" => AllocationMethod::Proportional,
            "marginal" => AllocationMethod::Marginal,
            "shapley" => AllocationMethod::Shapley,
            other => {
                return Err(Error::Other(format!(
                    "method must be euler, covariance, proportional, marginal or shapley, got {other}"
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
fn discrete_of(x: &Robj) -> Result<(Vec<f64>, Vec<f64>)> {
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

enum AnyCopula {
    Gaussian(GaussianCopula),
    StudentT(StudentTCopula),
    Archimedean(ArchimedeanCopula),
}

impl AnyCopula {
    fn as_copula(&self) -> &dyn Copula {
        match self {
            Self::Gaussian(c) => c,
            Self::StudentT(c) => c,
            Self::Archimedean(c) => c,
        }
    }
}

/// A copula.
#[extendr]
pub(crate) struct RiskCopula {
    inner: AnyCopula,
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

    fn dim(&self) -> f64 {
        self.inner.as_copula().dim() as f64
    }

    fn description(&self) -> String {
        match &self.inner {
            AnyCopula::Gaussian(_) => "Gaussian".into(),
            AnyCopula::StudentT(c) => format!("Student t, nu = {}", c.nu()),
            AnyCopula::Archimedean(c) => format!("{:?}, theta = {}", c.family(), c.theta()),
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

/// Iman-Conover reordering of a predictive distribution's components.
#[extendr]
fn iman_conover_reorder(
    pd: Robj,
    correlation: &[f64],
    seed: f64,
) -> Result<PredictiveDistribution> {
    let pd = <&PredictiveDistribution>::try_from(&pd)
        .map_err(|_| Error::Other("expected a predictive_distribution".into()))?;
    let inner = copula::iman_conover(&pd.inner, correlation, whole(seed, "seed")?).map_err(to_r)?;
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
    fn entropic_rust;
    fn esscher_rust;
    fn mes_rust;
    fn covar_rust;
    fn esscher_allocation_rust;
    fn iman_conover_reorder;
    fn gpd_mle;
    fn mean_excess_rust;
    fn hill_rust;
    impl EvtTail;
    impl RiskDistortion;
    impl RiskCopula;
}
