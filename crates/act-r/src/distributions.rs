//! Probability lane: wrappers over `act_prob` for the R `distributions.R`
//! API. Numeric arguments are vectors, as R users expect.

use act_core::StreamRng;
use act_prob::{
    ComponentKey, Counting, DiscretizationReport, Distribution, Empirical, Grid as GridInner,
    KeyValue, PredictiveDistribution as PdInner, Provenance, Sampled as SampledInner, Severity,
};
use extendr_api::prelude::*;
use extendr_api::{Error, Result};

use crate::{to_r, whole};

/// Lognormal distribution: `ln X ~ Normal(meanlog, sdlog^2)`.
#[extendr]
pub(crate) struct Lognormal {
    inner: act_prob::Lognormal,
}

#[extendr]
impl Lognormal {
    fn new(meanlog: f64, sdlog: f64) -> Result<Self> {
        let inner = act_prob::Lognormal::new(meanlog, sdlog).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn from_mean_cv(mean: f64, cv: f64) -> Result<Self> {
        let inner = act_prob::Lognormal::from_mean_cv(mean, cv).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn meanlog(&self) -> f64 {
        self.inner.meanlog()
    }

    fn sdlog(&self) -> f64 {
        self.inner.sdlog()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn cdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.inner.cdf(x)).collect()
    }

    fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.quantile(p).map_err(to_r))
            .collect()
    }

    fn lev(&self, limit: &[f64]) -> Vec<f64> {
        limit.iter().map(|&l| self.inner.lev(l)).collect()
    }

    fn stop_loss(&self, retention: &[f64]) -> Vec<f64> {
        retention.iter().map(|&d| self.inner.stop_loss(d)).collect()
    }

    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer(limit, attachment)
    }

    /// `n` draws from stream `stream` of the generator keyed by `seed`. R has
    /// no 64-bit integers, so ids arrive as doubles holding whole numbers.
    fn sample(&self, n: f64, seed: f64, stream: f64) -> Result<Vec<f64>> {
        let n = whole(n, "n")?;
        let mut rng = StreamRng::new(whole(seed, "seed")?, whole(stream, "stream")?);
        Ok(self.inner.sample(&mut rng, n as usize))
    }
}

/// A severity accepted wherever a parametric or discretized loss
/// distribution can be used: a `Lognormal`, a `Grid` or a Pareto-family
/// pointer.
pub(crate) enum AnySeverity {
    Lognormal(act_prob::Lognormal),
    Grid(GridInner),
    Pareto(act_prob::Pareto),
    PiecewisePareto(act_prob::PiecewisePareto),
    LogAffinePareto(act_prob::LogAffinePareto),
    GeneralizedPareto(act_prob::evt::Gpd),
    Gamma(act_prob::Gamma),
    Tweedie(act_prob::Tweedie),
    Weibull(act_prob::Weibull),
    Loglogistic(act_prob::Loglogistic),
    Mixture(std::sync::Arc<act_prob::Mixture>),
}

/// Calls `$call` on the inner severity, whichever it is.
macro_rules! each {
    ($self:ident, $d:ident => $call:expr) => {
        match $self {
            Self::Lognormal($d) => $call,
            Self::Grid($d) => $call,
            Self::Pareto($d) => $call,
            Self::PiecewisePareto($d) => $call,
            Self::LogAffinePareto($d) => $call,
            Self::GeneralizedPareto($d) => $call,
            Self::Gamma($d) => $call,
            Self::Tweedie($d) => $call,
            Self::Weibull($d) => $call,
            Self::Loglogistic($d) => $call,
            Self::Mixture($d) => $call,
        }
    };
}

impl AnySeverity {
    pub(crate) fn from_robj(obj: &Robj) -> Result<Self> {
        use crate::pareto::{
            GammaDist, GeneralizedPareto, LogAffinePareto, LoglogisticDist, MixtureDist, Pareto,
            PiecewisePareto, TweedieDist, WeibullDist,
        };
        if let Ok(d) = <&Lognormal>::try_from(obj) {
            return Ok(Self::Lognormal(d.inner));
        }
        if let Ok(g) = <&Grid>::try_from(obj) {
            return Ok(Self::Grid(g.inner.clone()));
        }
        if let Ok(d) = <&Pareto>::try_from(obj) {
            return Ok(Self::Pareto(d.inner));
        }
        if let Ok(d) = <&PiecewisePareto>::try_from(obj) {
            return Ok(Self::PiecewisePareto(d.inner.clone()));
        }
        if let Ok(d) = <&LogAffinePareto>::try_from(obj) {
            return Ok(Self::LogAffinePareto(d.inner));
        }
        if let Ok(d) = <&GeneralizedPareto>::try_from(obj) {
            return Ok(Self::GeneralizedPareto(d.inner));
        }
        if let Ok(d) = <&GammaDist>::try_from(obj) {
            return Ok(Self::Gamma(d.inner));
        }
        if let Ok(d) = <&TweedieDist>::try_from(obj) {
            return Ok(Self::Tweedie(d.inner));
        }
        if let Ok(d) = <&WeibullDist>::try_from(obj) {
            return Ok(Self::Weibull(d.inner));
        }
        if let Ok(d) = <&LoglogisticDist>::try_from(obj) {
            return Ok(Self::Loglogistic(d.inner));
        }
        if let Ok(d) = <&MixtureDist>::try_from(obj) {
            return Ok(Self::Mixture(d.inner.clone()));
        }
        Err(Error::Other(
            "expected a severity: lognormal, gamma, tweedie, weibull, loglogistic, mixture, grid, or a \
             Pareto-family distribution"
                .into(),
        ))
    }
}

impl Distribution for AnySeverity {
    fn mean(&self) -> f64 {
        each!(self, d => d.mean())
    }
    fn variance(&self) -> f64 {
        each!(self, d => d.variance())
    }
    fn cdf(&self, x: f64) -> f64 {
        each!(self, d => d.cdf(x))
    }
    fn survival(&self, x: f64) -> f64 {
        each!(self, d => d.survival(x))
    }
    fn quantile(&self, p: f64) -> act_core::Result<f64> {
        each!(self, d => d.quantile(p))
    }
}

impl Severity for AnySeverity {
    fn lev(&self, limit: f64) -> f64 {
        each!(self, d => d.lev(limit))
    }
    fn stop_loss(&self, retention: f64) -> f64 {
        each!(self, d => d.stop_loss(retention))
    }
    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        each!(self, d => d.layer(limit, attachment))
    }
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        each!(self, d => d.layer_second_moment(limit, attachment))
    }
}

/// Claim counts arrive from R as doubles holding whole numbers.
fn counts(k: &[f64]) -> Result<Vec<u64>> {
    k.iter().map(|&k| whole(k, "k")).collect()
}

/// Poisson claim counts.
#[extendr]
pub(crate) struct Poisson {
    pub(crate) inner: act_prob::Poisson,
}

#[extendr]
impl Poisson {
    fn new(lambda: f64) -> Result<Self> {
        let inner = act_prob::Poisson::new(lambda).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn lambda(&self) -> f64 {
        self.inner.lambda()
    }

    fn pmf(&self, k: &[f64]) -> Result<Vec<f64>> {
        Ok(counts(k)?.into_iter().map(|k| self.inner.pmf(k)).collect())
    }

    fn cdf(&self, k: &[f64]) -> Result<Vec<f64>> {
        Ok(counts(k)?.into_iter().map(|k| self.inner.cdf(k)).collect())
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.quantile(p).map(|k| k as f64).map_err(to_r))
            .collect()
    }

    fn sample(&self, n: f64, seed: f64, stream: f64) -> Result<Vec<f64>> {
        let n = whole(n, "n")?;
        let mut rng = StreamRng::new(whole(seed, "seed")?, whole(stream, "stream")?);
        Ok(self
            .inner
            .sample(&mut rng, n as usize)
            .into_iter()
            .map(|k| k as f64)
            .collect())
    }
}

/// Negative binomial claim counts: mean `r beta`, variance
/// `r beta (1 + beta)`.
#[extendr]
pub(crate) struct NegativeBinomial {
    pub(crate) inner: act_prob::NegativeBinomial,
}

#[extendr]
impl NegativeBinomial {
    fn new(r: f64, beta: f64) -> Result<Self> {
        let inner = act_prob::NegativeBinomial::new(r, beta).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn from_mean_variance(mean: f64, variance: f64) -> Result<Self> {
        let inner = act_prob::NegativeBinomial::from_mean_variance(mean, variance).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn r(&self) -> f64 {
        self.inner.r()
    }

    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    fn pmf(&self, k: &[f64]) -> Result<Vec<f64>> {
        Ok(counts(k)?.into_iter().map(|k| self.inner.pmf(k)).collect())
    }

    fn cdf(&self, k: &[f64]) -> Result<Vec<f64>> {
        Ok(counts(k)?.into_iter().map(|k| self.inner.cdf(k)).collect())
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.quantile(p).map(|k| k as f64).map_err(to_r))
            .collect()
    }

    fn sample(&self, n: f64, seed: f64, stream: f64) -> Result<Vec<f64>> {
        let n = whole(n, "n")?;
        let mut rng = StreamRng::new(whole(seed, "seed")?, whole(stream, "stream")?);
        Ok(self
            .inner
            .sample(&mut rng, n as usize)
            .into_iter()
            .map(|k| k as f64)
            .collect())
    }
}

/// A distribution on `0, step, 2 step, ...`, with the report of how it was
/// discretized when it came from `discretize()`.
#[extendr]
pub(crate) struct Grid {
    pub(crate) inner: GridInner,
    /// How the grid was made (a discretization or compound report), as the
    /// named list R sees, or `None` for a grid built from probabilities.
    report: Option<List>,
}

impl Grid {
    pub(crate) fn wrap(inner: GridInner) -> Self {
        Self {
            inner,
            report: None,
        }
    }

    pub(crate) fn with_report(inner: GridInner, report: List) -> Self {
        Self {
            inner,
            report: Some(report),
        }
    }
}

fn discretization_list(r: &DiscretizationReport) -> List {
    let method = match r.method {
        act_prob::Discretization::LocalMoment => "local_moment",
        act_prob::Discretization::Rounding => "rounding",
        act_prob::Discretization::Lower => "lower",
    };
    list!(
        method = method,
        step = r.step,
        points = r.points as f64,
        tail_mass = r.tail_mass,
        source_mean = r.source_mean,
        grid_mean = r.grid_mean,
        mean_error = r.mean_error()
    )
}

#[extendr]
impl Grid {
    fn new(step: f64, probs: &[f64]) -> Result<Self> {
        let inner = GridInner::new(step, probs.to_vec()).map_err(to_r)?;
        Ok(Self::wrap(inner))
    }

    /// `method` is "local_moment", "rounding" or "lower".
    fn discretize(severity: Robj, step: f64, points: f64, method: &str) -> Result<Self> {
        let sev = AnySeverity::from_robj(&severity)?;
        let points = whole(points, "points")? as usize;
        let (inner, report) = match method {
            "local_moment" => GridInner::local_moment(&sev, step, points),
            "rounding" => GridInner::rounding(&sev, step, points),
            "lower" => GridInner::lower(&sev, step, points),
            other => {
                return Err(Error::Other(format!(
                    "method must be local_moment, rounding or lower, got {other}"
                )));
            }
        }
        .map_err(to_r)?;
        Ok(Self::with_report(inner, discretization_list(&report)))
    }

    fn step(&self) -> f64 {
        self.inner.step()
    }

    fn probs(&self) -> Vec<f64> {
        self.inner.probs().to_vec()
    }

    /// The distribution of `f(X)` on the same step: `list(grid, on_points)`.
    fn map(&self, f: Function) -> Result<List> {
        // An R error inside `f` becomes NaN for the Rust side, which
        // rejects it; the original error is returned instead.
        let mut failure = None;
        let result = self.inner.map(|x| {
            let value = f.call(pairlist!(x)).and_then(|v| {
                v.as_real()
                    .ok_or(Error::Other("f must return a number".into()))
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
        let (inner, on_points) = result.map_err(to_r)?;
        Ok(list!(grid = Self::wrap(inner), on_points = on_points))
    }

    /// The report of how the grid was made, as a named list, or NULL.
    fn report(&self) -> Robj {
        match &self.report {
            None => ().into(),
            Some(list) => list.clone().into(),
        }
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn cdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.inner.cdf(x)).collect()
    }

    fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.quantile(p).map_err(to_r))
            .collect()
    }

    fn lev(&self, limit: &[f64]) -> Vec<f64> {
        limit.iter().map(|&l| self.inner.lev(l)).collect()
    }

    fn stop_loss(&self, retention: &[f64]) -> Vec<f64> {
        retention.iter().map(|&d| self.inner.stop_loss(d)).collect()
    }

    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer(limit, attachment)
    }
}

/// A distribution known only through equally weighted draws.
#[extendr]
pub(crate) struct Sampled {
    pub(crate) inner: SampledInner,
}

#[extendr]
impl Sampled {
    fn new(draws: &[f64]) -> Result<Self> {
        let inner = SampledInner::new(draws.to_vec()).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn draws(&self) -> Vec<f64> {
        self.inner.draws().to_vec()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn cdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.inner.cdf(x)).collect()
    }

    fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.quantile(p).map_err(to_r))
            .collect()
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

/// Component keys from R key columns (a list with one column per dimension
/// and one entry per component).
pub(crate) fn components_from_keys(
    dims: &[String],
    keys: List,
    n_components: usize,
) -> Result<Vec<ComponentKey>> {
    let columns: Vec<Vec<KeyValue>> = keys
        .values()
        .map(|c| key_column(&c))
        .collect::<Result<_>>()?;
    if columns.len() != dims.len() {
        return Err(Error::Other(format!(
            "{} key columns for {} dimensions",
            columns.len(),
            dims.len()
        )));
    }
    if columns.iter().any(|c| c.len() != n_components) {
        return Err(Error::Other(format!(
            "every key column needs one entry per component ({n_components})"
        )));
    }
    Ok((0..n_components)
        .map(|j| columns.iter().map(|c| c[j].clone()).collect())
        .collect())
}

/// Converts one R key column (character, integer or double) to key values.
/// A component key from an R list or vector, one entry per dimension.
pub(crate) fn key_from_list(key: List) -> Result<ComponentKey> {
    key.values()
        .map(|v| key_column(&v).map(|mut c| c.remove(0)))
        .collect()
}

pub(crate) fn key_column(col: &Robj) -> Result<Vec<KeyValue>> {
    if let Some(s) = col.as_str_vector() {
        return Ok(s.into_iter().map(KeyValue::from).collect());
    }
    if let Some(i) = col.as_integer_slice() {
        return Ok(i.iter().map(|&v| KeyValue::Int(i64::from(v))).collect());
    }
    if let Some(d) = col.as_real_slice() {
        return d
            .iter()
            .map(|&v| {
                if v.fract() == 0.0 && v.abs() < 9.0e15 {
                    Ok(KeyValue::Int(v as i64))
                } else {
                    Err(Error::Other(format!("key value {v} is not a whole number")))
                }
            })
            .collect();
    }
    Err(Error::Other(
        "key columns must be character or whole numbers".into(),
    ))
}

/// The joint result every model returns.
#[extendr]
pub(crate) struct PredictiveDistribution {
    pub(crate) inner: PdInner,
}

#[extendr]
impl PredictiveDistribution {
    /// `keys` is a list of columns, one per dimension, each with one entry per
    /// component; `draws` is an `n_sims × n_components` matrix in R's
    /// column-major order.
    fn new(dims: Vec<String>, keys: List, draws: &[f64], n_sims: f64) -> Result<Self> {
        let n_sims = whole(n_sims, "n_sims")? as usize;
        let n_components = draws.len().checked_div(n_sims).unwrap_or(0);
        if n_sims == 0 || draws.len() != n_sims * n_components {
            return Err(Error::Other(
                "draws must be an n_sims x n_components matrix".into(),
            ));
        }
        let components = components_from_keys(&dims, keys, n_components)?;
        // Column-major (R) to simulation-major.
        let mut rows = Vec::with_capacity(draws.len());
        for i in 0..n_sims {
            for j in 0..n_components {
                rows.push(draws[i + j * n_sims]);
            }
        }
        let inner =
            PdInner::from_draws(dims, components, rows, Provenance::new("r")).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn dims(&self) -> Vec<String> {
        self.inner.dims().to_vec()
    }

    fn n_sims(&self) -> f64 {
        self.inner.n_sims() as f64
    }

    fn n_components(&self) -> f64 {
        self.inner.n_components() as f64
    }

    /// Key columns as a named list: integers as doubles, text and periods as
    /// character.
    fn keys(&self) -> Robj {
        let dims = self.inner.dims();
        let comps = self.inner.components();
        let columns: Vec<Robj> = (0..dims.len())
            .map(|d| {
                let all_int = comps.iter().all(|k| matches!(k[d], KeyValue::Int(_)));
                if all_int {
                    comps
                        .iter()
                        .map(|k| match k[d] {
                            KeyValue::Int(i) => i as f64,
                            _ => unreachable!("checked above"),
                        })
                        .collect::<Vec<f64>>()
                        .into()
                } else {
                    comps
                        .iter()
                        .map(|k| k[d].to_string())
                        .collect::<Vec<String>>()
                        .into()
                }
            })
            .collect();
        List::from_names_and_values(dims, columns)
            .map(Robj::from)
            .unwrap_or_else(|_| ().into())
    }

    /// Draws in R's column-major order (`n_sims × n_components`).
    fn draw_matrix(&self) -> Vec<f64> {
        let (n, m) = (self.inner.n_sims(), self.inner.n_components());
        let d = self.inner.draw_matrix();
        let mut out = Vec::with_capacity(d.len());
        for j in 0..m {
            for i in 0..n {
                out.push(d[i * m + j]);
            }
        }
        out
    }

    /// The component with this key (a list or vector, one entry per
    /// dimension), or NULL. An origin period is named by its label
    /// (`"2021"`, `2021`, `"2021Q3"`).
    fn marginal(&self, key: List) -> Result<Robj> {
        let key = key_from_list(key)?;
        Ok(match self.inner.marginal(&key) {
            Some(inner) => Sampled { inner }.into(),
            None => ().into(),
        })
    }

    fn aggregate(&self, keep: Vec<String>) -> Result<Self> {
        let keep: Vec<&str> = keep.iter().map(String::as_str).collect();
        let inner = self.inner.aggregate(&keep).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn total(&self) -> Sampled {
        Sampled {
            inner: self.inner.total().clone(),
        }
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.quantile(p).map_err(to_r))
            .collect()
    }

    fn var(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter().map(|&p| self.inner.var(p).map_err(to_r)).collect()
    }

    fn tvar(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.tvar(p).map_err(to_r))
            .collect()
    }

    /// Provenance as a named list.
    fn provenance(&self) -> Robj {
        let p = self.inner.provenance();
        let pairs = |v: &[(String, String)]| -> Robj {
            List::from_names_and_values(
                v.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
                v.iter().map(|(_, x)| x.clone()).collect::<Vec<_>>(),
            )
            .map(Robj::from)
            .unwrap_or_else(|_| ().into())
        };
        list!(
            model = p.model.clone(),
            parameters = pairs(&p.parameters),
            seed = p.seed.map(|s| s as f64),
            stream_scheme = p.stream_scheme.clone(),
            versions = pairs(&p.versions),
            input_hash = p.input_hash.clone()
        )
        .into()
    }
}

/// Blends predictive distributions with `weights`; see
/// `act_prob::PredictiveDistribution::blend`.
#[extendr]
fn blend_rust(models: List, weights: &[f64], seed: f64) -> Result<PredictiveDistribution> {
    let refs: Vec<&PredictiveDistribution> = models
        .values()
        .map(|m| {
            <&PredictiveDistribution>::try_from(&m)
                .map_err(|_| Error::Other("every model must be a predictive_distribution".into()))
        })
        .collect::<Result<_>>()?;
    let inner: Vec<&PdInner> = refs.iter().map(|p| &p.inner).collect();
    let pd = PdInner::blend(&inner, weights, whole(seed, "seed")?).map_err(to_r)?;
    Ok(PredictiveDistribution { inner: pd })
}

/// Blends with one weight vector per component: `weights` is
/// `components × models` column-major.
#[extendr]
fn blend_by_component_rust(
    models: List,
    weights: &[f64],
    seed: f64,
) -> Result<PredictiveDistribution> {
    let refs: Vec<&PredictiveDistribution> = models
        .values()
        .map(|m| {
            <&PredictiveDistribution>::try_from(&m)
                .map_err(|_| Error::Other("every model must be a predictive_distribution".into()))
        })
        .collect::<Result<_>>()?;
    let k = refs.len();
    if k == 0 || weights.len() % k != 0 {
        return Err(Error::Other("weights need one column per model".into()));
    }
    let c = weights.len() / k;
    let rows: Vec<Vec<f64>> = (0..c)
        .map(|j| (0..k).map(|m| weights[m * c + j]).collect())
        .collect();
    let inner: Vec<&PdInner> = refs.iter().map(|p| &p.inner).collect();
    let pd = PdInner::blend_by_component(&inner, &rows, whole(seed, "seed")?).map_err(to_r)?;
    Ok(PredictiveDistribution { inner: pd })
}

/// Joins predictive distributions: `labels` name the `parts`.
#[extendr]
fn join_rust(
    parts: List,
    labels: Vec<String>,
    dim: &str,
    same_simulations: bool,
) -> Result<PredictiveDistribution> {
    use act_prob::portfolio::Pairing;
    let refs: Vec<&PredictiveDistribution> = parts
        .values()
        .map(|m| {
            <&PredictiveDistribution>::try_from(&m)
                .map_err(|_| Error::Other("every part must be a predictive_distribution".into()))
        })
        .collect::<Result<_>>()?;
    if refs.len() != labels.len() {
        return Err(Error::Other("one label per part".into()));
    }
    let pairs: Vec<(&str, &PdInner)> = labels
        .iter()
        .map(String::as_str)
        .zip(refs.iter().map(|p| &p.inner))
        .collect();
    let pairing = if same_simulations {
        Pairing::SameSimulations
    } else {
        Pairing::Independent
    };
    let inner = PdInner::join(&pairs, dim, pairing).map_err(to_r)?;
    Ok(PredictiveDistribution { inner })
}

/// Iman-Conover on the totals of the groups of `dim`, moving whole rows.
#[extendr]
fn reorder_groups_rust(
    pd: Robj,
    dim: &str,
    correlation: &[f64],
    seed: f64,
) -> Result<PredictiveDistribution> {
    let pd = <&PredictiveDistribution>::try_from(&pd)
        .map_err(|_| Error::Other("expected a predictive_distribution".into()))?;
    let inner = pd
        .inner
        .reorder_groups(dim, correlation, whole(seed, "seed")?)
        .map_err(to_r)?;
    Ok(PredictiveDistribution { inner })
}

extendr_module! {
    mod distributions;
    fn join_rust;
    fn reorder_groups_rust;
    fn blend_rust;
    fn blend_by_component_rust;
    impl Lognormal;
    impl Poisson;
    impl NegativeBinomial;
    impl Grid;
    impl Sampled;
    impl PredictiveDistribution;
}
