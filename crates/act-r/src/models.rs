//! Models lane: GLMs, elastic nets, GAMs, metrics, resampling and MCMC diagnostics for
//! the R `models.R` API (`docs/design/models.md`). R builds the design
//! matrix with `model.matrix`; it arrives here column-major with its
//! column names.

use act_glm::gam::{Gam, GamFit, PSpline, Smoothing};
use act_glm::net::{ElasticNet, ElasticNetFit};
use act_glm::{Dispersion, Glm, GlmFit, Robust};
use act_models::resample;
use act_models::{Design, Family, Fitted, Link, Model, metrics};
use extendr_api::prelude::*;
use extendr_api::{Error, Result};

use crate::distributions::PredictiveDistribution;
use crate::{to_r, whole};

/// A family from its name; `theta` and `power` are NA when not used.
fn family(name: &str, theta: f64, power: f64) -> Result<Family> {
    let f = match name {
        "gaussian" => Family::Gaussian,
        "poisson" => Family::Poisson,
        "gamma" => Family::Gamma,
        "inverse_gaussian" => Family::InverseGaussian,
        "binomial" => Family::Binomial,
        "negative_binomial" => Family::NegativeBinomial { theta },
        "tweedie" => Family::Tweedie { power },
        other => {
            return Err(Error::Other(format!(
                "family must be gaussian, poisson, gamma, inverse_gaussian, binomial, \
                 negative_binomial or tweedie, got {other}"
            )));
        }
    };
    f.validate().map_err(to_r)?;
    Ok(f)
}

/// A link from its name; "" for the family's canonical link.
fn link(name: &str, family: Family, link_power: f64) -> Result<Link> {
    Ok(match name {
        "" => family.canonical_link(),
        "identity" => Link::Identity,
        "log" => Link::Log,
        "logit" => Link::Logit,
        "probit" => Link::Probit,
        "cloglog" => Link::Cloglog,
        "inverse" => Link::Inverse,
        "inverse_squared" => Link::InverseSquared,
        "power" => Link::Power(link_power),
        other => {
            return Err(Error::Other(format!(
                "link must be identity, log, logit, probit, cloglog, inverse, \
                 inverse_squared or power, got {other}"
            )));
        }
    })
}

/// "" for the default, "pearson", "deviance", or "fixed" with `value`.
fn dispersion(kind: &str, value: f64, default: Dispersion) -> Result<Dispersion> {
    match kind {
        "" => Ok(default),
        "pearson" => Ok(Dispersion::Pearson),
        "deviance" => Ok(Dispersion::Deviance),
        "fixed" => Ok(Dispersion::Fixed(value)),
        other => Err(Error::Other(format!(
            "dispersion must be pearson, deviance or a number, got {other}"
        ))),
    }
}

/// A design from a column-major matrix `x` with `names`, offset and
/// weights (empty for none).
fn design(x: &[f64], names: Vec<String>, offset: &[f64], weights: &[f64]) -> Result<Design> {
    let p = names.len();
    if p == 0 || x.len() % p != 0 {
        return Err(Error::Other(
            "the design matrix and its names do not match".into(),
        ));
    }
    let n = x.len() / p;
    let columns = (0..p).map(|j| x[j * n..(j + 1) * n].to_vec()).collect();
    let mut d = Design::new(names, columns).map_err(to_r)?;
    if !offset.is_empty() {
        d = d.with_offset(offset.to_vec()).map_err(to_r)?;
    }
    if !weights.is_empty() {
        d = d.with_weights(weights.to_vec()).map_err(to_r)?;
    }
    Ok(d)
}

/// A fitted GLM.
#[extendr]
pub(crate) struct GlmModel {
    inner: GlmFit,
    /// The training data, for the sandwich covariance; not kept by a model
    /// loaded from an artifact.
    training: Option<(Design, Vec<f64>)>,
}

/// Fits a GLM to the design `x` (column-major, columns `names`).
#[extendr]
#[allow(clippy::too_many_arguments)]
fn glm_fit_design(
    x: &[f64],
    names: Vec<String>,
    y: &[f64],
    offset: &[f64],
    weights: &[f64],
    family_name: &str,
    link_name: &str,
    dispersion_kind: &str,
    dispersion_value: f64,
    theta: f64,
    power: f64,
    link_power: f64,
) -> Result<GlmModel> {
    let f = family(family_name, theta, power)?;
    let l = link(link_name, f, link_power)?;
    let base = Glm::new(f, l);
    let glm = base.dispersion(dispersion(
        dispersion_kind,
        dispersion_value,
        base.dispersion,
    )?);
    let d = design(x, names, offset, weights)?;
    let inner = glm.fit(&d, y).map_err(to_r)?;
    Ok(GlmModel {
        inner,
        training: Some((d, y.to_vec())),
    })
}

#[extendr]
impl GlmModel {
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    fn coefficients(&self) -> Vec<f64> {
        self.inner.coefficients().to_vec()
    }

    fn std_errors(&self) -> Vec<f64> {
        self.inner.std_errors()
    }

    fn p_values(&self) -> Vec<f64> {
        self.inner.p_values()
    }

    /// Row-major; symmetric, so also column-major.
    fn covariance(&self) -> Vec<f64> {
        self.inner.covariance()
    }

    /// Sandwich covariance: `kind` "HC0" or "HC1", or "cluster" with one
    /// positive integer label per training row in `groups`.
    fn robust_covariance(&self, kind: &str, groups: &[i32]) -> Result<Vec<f64>> {
        let labels: Vec<usize> = groups.iter().map(|&g| g.max(0) as usize).collect();
        let kind = match kind {
            "HC0" => Robust::Hc0,
            "HC1" => Robust::Hc1,
            "cluster" => Robust::Cluster(&labels),
            other => return Err(Error::Other(format!("unknown kind {other:?}"))),
        };
        let (design, y) = self.training.as_ref().ok_or_else(|| {
            Error::Other("a loaded model has no training data for a sandwich covariance".into())
        })?;
        self.inner.robust_covariance(design, y, kind).map_err(to_r)
    }

    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    fn from_json(text: &str) -> Result<Self> {
        Ok(Self {
            inner: GlmFit::from_json(text).map_err(to_r)?,
            training: None,
        })
    }

    fn input_hash(&self) -> String {
        self.inner.input_hash().to_string()
    }

    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    fn deviance(&self) -> f64 {
        self.inner.deviance()
    }

    fn null_deviance(&self) -> f64 {
        self.inner.null_deviance()
    }

    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood()
    }

    fn aic(&self) -> f64 {
        self.inner.aic()
    }

    fn df_resid(&self) -> f64 {
        self.inner.df_resid()
    }

    fn iterations(&self) -> f64 {
        self.inner.iterations() as f64
    }

    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted().to_vec()
    }

    fn family(&self) -> String {
        self.inner.spec().family.name().into()
    }

    fn predict(&self, x: &[f64], names: Vec<String>, offset: &[f64]) -> Result<Vec<f64>> {
        let d = design(x, names, offset, &[])?;
        self.inner.predict(&d).map_err(to_r)
    }

    fn predict_distribution(
        &self,
        x: &[f64],
        names: Vec<String>,
        offset: &[f64],
        weights: &[f64],
        n_sims: f64,
        seed: f64,
    ) -> Result<PredictiveDistribution> {
        let d = design(x, names, offset, weights)?;
        let inner = self
            .inner
            .predict_distribution(&d, whole(n_sims, "n_sims")? as usize, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

/// An elastic-net regularization path: one fit per penalty strength.
#[extendr]
pub(crate) struct ElasticNetPath {
    fits: Vec<ElasticNetFit>,
}

/// Fits an elastic net at each of `lambdas`, or, when it is empty, along
/// `nlambda` values log-spaced from lambda_max down to `min_ratio` times it.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn elastic_net_fit_design(
    x: &[f64],
    names: Vec<String>,
    y: &[f64],
    offset: &[f64],
    weights: &[f64],
    family_name: &str,
    link_name: &str,
    alpha: f64,
    lambdas: &[f64],
    nlambda: f64,
    min_ratio: f64,
    standardize: bool,
    penalty_factor: &[f64],
    theta: f64,
    power: f64,
    link_power: f64,
) -> Result<ElasticNetPath> {
    let f = family(family_name, theta, power)?;
    let l = link(link_name, f, link_power)?;
    let mut net = ElasticNet::new(f, l, alpha, 0.0).standardize(standardize);
    if !penalty_factor.is_empty() {
        net.penalty_factor = Some(penalty_factor.to_vec());
    }
    let d = design(x, names, offset, weights)?;
    let lambdas = if lambdas.is_empty() {
        let n = whole(nlambda, "nlambda")? as usize;
        net.lambda_path(&d, y, n, min_ratio).map_err(to_r)?
    } else {
        lambdas.to_vec()
    };
    let fits = net.path(&d, y, &lambdas).map_err(to_r)?;
    Ok(ElasticNetPath { fits })
}

/// Cross-validates an elastic net on folds given as one fold id per row
/// (1 to K): the path's mean deviance and standard error per lambda.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn elastic_net_cv_design(
    x: &[f64],
    names: Vec<String>,
    y: &[f64],
    offset: &[f64],
    weights: &[f64],
    family_name: &str,
    link_name: &str,
    alpha: f64,
    lambdas: &[f64],
    nlambda: f64,
    min_ratio: f64,
    standardize: bool,
    penalty_factor: &[f64],
    theta: f64,
    power: f64,
    link_power: f64,
    foldid: &[f64],
) -> Result<List> {
    let f = family(family_name, theta, power)?;
    let l = link(link_name, f, link_power)?;
    let mut net = ElasticNet::new(f, l, alpha, 0.0).standardize(standardize);
    if !penalty_factor.is_empty() {
        net.penalty_factor = Some(penalty_factor.to_vec());
    }
    let d = design(x, names, offset, weights)?;
    let lambdas = if lambdas.is_empty() {
        let n = whole(nlambda, "nlambda")? as usize;
        net.lambda_path(&d, y, n, min_ratio).map_err(to_r)?
    } else {
        lambdas.to_vec()
    };
    if foldid.len() != y.len() {
        return Err(Error::Other(format!(
            "foldid has {} entries for {} rows",
            foldid.len(),
            y.len()
        )));
    }
    let ids: Vec<u64> = foldid
        .iter()
        .map(|&v| whole(v, "foldid"))
        .collect::<Result<_>>()?;
    let mut distinct = ids.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let splits: Vec<resample::Split> = distinct
        .iter()
        .map(|&k| resample::Split {
            train: (0..ids.len()).filter(|&i| ids[i] != k).collect(),
            test: (0..ids.len()).filter(|&i| ids[i] == k).collect(),
        })
        .collect();
    let cv = net.cross_validate(&d, y, &lambdas, &splits).map_err(to_r)?;
    Ok(list!(
        lambda = cv.lambdas.clone(),
        mean = cv.mean.clone(),
        se = cv.se.clone(),
        lambda_min = cv.lambda_min(),
        lambda_1se = cv.lambda_1se()
    ))
}

impl ElasticNetPath {
    fn at(&self, index: f64) -> Result<&ElasticNetFit> {
        let k = whole(index, "index")? as usize;
        self.fits
            .get(k.wrapping_sub(1))
            .ok_or_else(|| Error::Other(format!("index must be 1 to {}, got {k}", self.fits.len())))
    }
}

#[extendr]
impl ElasticNetPath {
    fn names(&self) -> Vec<String> {
        self.fits[0].names().to_vec()
    }

    fn lambda(&self) -> Vec<f64> {
        self.fits.iter().map(ElasticNetFit::lambda).collect()
    }

    /// Column-major `p × k`, one column per lambda.
    fn coefficients(&self) -> Vec<f64> {
        self.fits
            .iter()
            .flat_map(|f| f.coefficients().to_vec())
            .collect()
    }

    fn deviance(&self) -> Vec<f64> {
        self.fits.iter().map(ElasticNetFit::deviance).collect()
    }

    fn deviance_ratio(&self) -> Vec<f64> {
        self.fits
            .iter()
            .map(ElasticNetFit::deviance_ratio)
            .collect()
    }

    fn df(&self) -> Vec<f64> {
        self.fits.iter().map(|f| f.df() as f64).collect()
    }

    fn null_deviance(&self) -> f64 {
        self.fits[0].null_deviance()
    }

    fn alpha(&self) -> f64 {
        self.fits[0].spec().alpha
    }

    fn family(&self) -> String {
        self.fits[0].spec().family.name().to_string()
    }

    fn predict(
        &self,
        index: f64,
        x: &[f64],
        names: Vec<String>,
        offset: &[f64],
    ) -> Result<Vec<f64>> {
        let d = design(x, names, offset, &[])?;
        self.at(index)?.predict(&d).map_err(to_r)
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_distribution(
        &self,
        index: f64,
        x: &[f64],
        names: Vec<String>,
        offset: &[f64],
        weights: &[f64],
        n_sims: f64,
        seed: f64,
    ) -> Result<PredictiveDistribution> {
        let d = design(x, names, offset, weights)?;
        let inner = self
            .at(index)?
            .predict_distribution(&d, whole(n_sims, "n_sims")? as usize, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

/// A fitted GAM.
#[extendr]
pub(crate) struct GamModel {
    inner: GamFit,
}

/// Fits a GAM: the GLM spec plus P-spline smooths of the named design
/// columns with `n_basis` functions each. `smoothing` is "auto", "gcv",
/// "ubre" or "fixed" with `lambdas`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn gam_fit_design(
    x: &[f64],
    names: Vec<String>,
    y: &[f64],
    offset: &[f64],
    weights: &[f64],
    family_name: &str,
    link_name: &str,
    dispersion_kind: &str,
    dispersion_value: f64,
    theta: f64,
    power: f64,
    smooths: Vec<String>,
    n_basis: &[f64],
    smoothing: &str,
    lambdas: &[f64],
) -> Result<GamModel> {
    let f = family(family_name, theta, power)?;
    let l = link(link_name, f, f64::NAN)?;
    let base = Glm::new(f, l);
    let glm = base.dispersion(dispersion(
        dispersion_kind,
        dispersion_value,
        base.dispersion,
    )?);
    if n_basis.len() != smooths.len() {
        return Err(Error::Other("give one n_basis per smooth".into()));
    }
    let splines = smooths
        .iter()
        .zip(n_basis)
        .map(|(s, &k)| Ok(PSpline::new(s).n_basis(whole(k, "n_basis")? as usize)))
        .collect::<Result<Vec<_>>>()?;
    let smoothing = match smoothing {
        "auto" => Smoothing::Auto,
        "gcv" => Smoothing::Gcv,
        "ubre" => Smoothing::Ubre,
        "fixed" => Smoothing::Fixed(lambdas.to_vec()),
        other => {
            return Err(Error::Other(format!(
                "smoothing must be auto, gcv, ubre or numeric, got {other}"
            )));
        }
    };
    let d = design(x, names, offset, weights)?;
    let inner = Gam::new(glm, splines)
        .smoothing(smoothing)
        .fit(&d, y)
        .map_err(to_r)?;
    Ok(GamModel { inner })
}

#[extendr]
impl GamModel {
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    fn coefficients(&self) -> Vec<f64> {
        self.inner.coefficients().to_vec()
    }

    fn lambdas(&self) -> Vec<f64> {
        self.inner.lambdas().to_vec()
    }

    fn edf(&self) -> f64 {
        self.inner.edf()
    }

    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    fn deviance(&self) -> f64 {
        self.inner.deviance()
    }

    fn score(&self) -> f64 {
        self.inner.score()
    }

    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted().to_vec()
    }

    fn predict(&self, x: &[f64], names: Vec<String>, offset: &[f64]) -> Result<Vec<f64>> {
        let d = design(x, names, offset, &[])?;
        self.inner.predict(&d).map_err(to_r)
    }

    fn predict_distribution(
        &self,
        x: &[f64],
        names: Vec<String>,
        offset: &[f64],
        weights: &[f64],
        n_sims: f64,
        seed: f64,
    ) -> Result<PredictiveDistribution> {
        let d = design(x, names, offset, weights)?;
        let inner = self
            .inner
            .predict_distribution(&d, whole(n_sims, "n_sims")? as usize, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

/// `Σ w d(y, μ)`; empty weights for none.
#[extendr]
fn family_deviance_rust(
    family_name: &str,
    theta: f64,
    power: f64,
    y: &[f64],
    mu: &[f64],
    weights: &[f64],
) -> Result<f64> {
    let f = family(family_name, theta, power)?;
    let w = (!weights.is_empty()).then_some(weights);
    metrics::deviance(f, y, mu, w).map_err(to_r)
}

/// Gini of the ordered Lorenz curve; empty exposure for none.
#[extendr]
fn gini_rust(y: &[f64], pred: &[f64], exposure: &[f64]) -> Result<f64> {
    let e = (!exposure.is_empty()).then_some(exposure);
    metrics::gini(y, pred, e).map_err(to_r)
}

/// Lift bands as `list(exposure, expected, actual)`.
#[extendr]
fn lift_rust(y: &[f64], pred: &[f64], exposure: &[f64], bands: f64) -> Result<List> {
    let e = (!exposure.is_empty()).then_some(exposure);
    let t = metrics::lift(y, pred, e, whole(bands, "bands")? as usize).map_err(to_r)?;
    Ok(list!(
        exposure = t.iter().map(|b| b.exposure).collect::<Vec<_>>(),
        expected = t.iter().map(|b| b.expected).collect::<Vec<_>>(),
        actual = t.iter().map(|b| b.actual).collect::<Vec<_>>()
    ))
}

#[extendr]
fn crps_rust(draws: &[f64], y: f64) -> Result<f64> {
    metrics::crps(draws, y).map_err(to_r)
}

#[extendr]
#[allow(clippy::too_many_arguments)]
fn log_score_rust(
    family_name: &str,
    theta: f64,
    power: f64,
    y: &[f64],
    mu: &[f64],
    dispersion: f64,
    weights: &[f64],
) -> Result<f64> {
    let f = family(family_name, theta, power)?;
    let w = (!weights.is_empty()).then_some(weights);
    metrics::log_score(f, y, mu, dispersion, w).map_err(to_r)
}

#[extendr]
#[allow(clippy::too_many_arguments)]
fn pit_rust(
    family_name: &str,
    theta: f64,
    power: f64,
    y: &[f64],
    mu: &[f64],
    dispersion: f64,
    weights: &[f64],
    seed: f64,
) -> Result<Vec<f64>> {
    let f = family(family_name, theta, power)?;
    let w = (!weights.is_empty()).then_some(weights);
    metrics::pit(f, y, mu, dispersion, w, whole(seed, "seed")?).map_err(to_r)
}

#[extendr]
fn ks_uniform_rust(values: &[f64]) -> Result<f64> {
    metrics::ks_uniform(values).map_err(to_r)
}

/// Splits as a list of `list(train, test)` with 1-based row numbers.
fn splits_r(splits: Vec<resample::Split>) -> List {
    let one = |v: Vec<usize>| v.into_iter().map(|i| i as f64 + 1.0).collect::<Vec<f64>>();
    List::from_values(
        splits
            .into_iter()
            .map(|s| list!(train = one(s.train), test = one(s.test))),
    )
}

#[extendr]
fn k_fold_rust(n: f64, k: f64, seed: f64) -> Result<List> {
    resample::k_fold(
        whole(n, "n")? as usize,
        whole(k, "k")? as usize,
        whole(seed, "seed")?,
    )
    .map(splits_r)
    .map_err(to_r)
}

#[extendr]
fn group_k_fold_rust(groups: Vec<String>, k: f64, seed: f64) -> Result<List> {
    resample::group_k_fold(&groups, whole(k, "k")? as usize, whole(seed, "seed")?)
        .map(splits_r)
        .map_err(to_r)
}

#[extendr]
fn time_ordered_rust(periods: &[f64], n_test: f64) -> Result<List> {
    let p: Vec<i64> = periods.iter().map(|&v| v as i64).collect();
    resample::time_ordered(&p, whole(n_test, "n_test")? as usize)
        .map(splits_r)
        .map_err(to_r)
}

/// MCMC diagnostics of `draws` (column-major, `n_chains` columns).
#[extendr]
fn mcmc_diagnostics_rust(draws: &[f64], n_chains: f64) -> Result<List> {
    let m = whole(n_chains, "n_chains")? as usize;
    if m == 0 || draws.len() % m != 0 {
        return Err(Error::Other(
            "draws must be a matrix with one column per chain".into(),
        ));
    }
    let n = draws.len() / m;
    let chains: Vec<&[f64]> = (0..m).map(|j| &draws[j * n..(j + 1) * n]).collect();
    Ok(list!(
        rhat = act_bayes::rhat(&chains).map_err(to_r)?,
        ess_bulk = act_bayes::ess_bulk(&chains).map_err(to_r)?,
        ess_tail = act_bayes::ess_tail(&chains).map_err(to_r)?,
        ess_mean = act_bayes::ess_mean(&chains).map_err(to_r)?,
        mcse_mean = act_bayes::mcse_mean(&chains).map_err(to_r)?
    ))
}

/// An ELPD estimate as an R list.
fn elpd_list(e: act_bayes::elpd::Elpd) -> List {
    list!(
        estimates = vec![e.elpd, e.se, e.p, e.ic],
        pointwise = e.pointwise
    )
}

#[extendr]
fn elpd_loo_rust(log_lik: &[f64], n: f64, r_eff: &[f64]) -> Result<List> {
    let n = whole(n, "n")? as usize;
    let r = (!r_eff.is_empty()).then_some(r_eff);
    let l = act_bayes::elpd::loo(log_lik, n, r).map_err(to_r)?;
    Ok(list!(
        estimates = vec![l.estimate.elpd, l.estimate.se, l.estimate.p, l.estimate.ic],
        pointwise = l.estimate.pointwise,
        pareto_k = l.pareto_k,
        k_threshold = l.k_threshold
    ))
}

#[extendr]
fn elpd_waic_rust(log_lik: &[f64], n: f64) -> Result<List> {
    let n = whole(n, "n")? as usize;
    Ok(elpd_list(act_bayes::elpd::waic(log_lik, n).map_err(to_r)?))
}

extendr_module! {
    mod models;
    fn glm_fit_design;
    fn elastic_net_fit_design;
    fn elastic_net_cv_design;
    fn gam_fit_design;
    impl GlmModel;
    impl ElasticNetPath;
    impl GamModel;
    fn family_deviance_rust;
    fn gini_rust;
    fn lift_rust;
    fn crps_rust;
    fn log_score_rust;
    fn pit_rust;
    fn ks_uniform_rust;
    fn k_fold_rust;
    fn group_k_fold_rust;
    fn time_ordered_rust;
    fn mcmc_diagnostics_rust;
    fn elpd_loo_rust;
    fn elpd_waic_rust;
}
