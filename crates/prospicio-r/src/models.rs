//! Models lane: GLMs, elastic nets, GAMs, metrics, resampling and MCMC diagnostics for
//! the R `models.R` API (`docs/design/models.md`). R builds the design
//! matrix with `model.matrix`; it arrives here column-major with its
//! column names.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_bayes::ppc::{self, Statistic, StatisticCheck};
use prospicio_glm::gam::{Gam, GamFit, PSpline, Smoothing};
use prospicio_glm::glmm::{Glmm, GlmmFit, Grouping};
use prospicio_glm::net::{ElasticNet, ElasticNetFit};
use prospicio_glm::{Dispersion, Glm, GlmFit, Robust};
use prospicio_models::resample;
use prospicio_models::{Design, Family, Fitted, Link, Model, metrics};

use crate::distributions::PredictiveDistribution;
use crate::{to_r, whole};

/// A family from its name; `theta` and `power` are NA when not used.
pub(crate) fn family(name: &str, theta: f64, power: f64) -> Result<Family> {
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
pub(crate) fn link(name: &str, family: Family, link_power: f64) -> Result<Link> {
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
pub(crate) fn design(
    x: &[f64],
    names: Vec<String>,
    offset: &[f64],
    weights: &[f64],
) -> Result<Design> {
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

/// A Tweedie GLM with its power estimated: `list(model, power, phi,
/// log_likelihood, lower, upper, lower_open, upper_open, at_boundary)`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn tweedie_glm_fit_design(
    x: &[f64],
    names: Vec<String>,
    y: &[f64],
    offset: &[f64],
    weights: &[f64],
    link_name: &str,
    min_power: f64,
    max_power: f64,
) -> Result<List> {
    let l = link(link_name, Family::Poisson, f64::NAN)?;
    let mut spec = prospicio_glm::tweedie::TweedieGlm::new(l);
    spec.min_power = min_power;
    spec.max_power = max_power;
    let d = design(x, names, offset, weights)?;
    let f = spec.fit(&d, y).map_err(to_r)?;
    let (lower, upper) = f.interval();
    let (lower_open, upper_open) = f.interval_open();
    Ok(list!(
        model = GlmModel {
            inner: f.glm().clone(),
            training: Some((d, y.to_vec())),
        },
        power = f.power(),
        phi = f.dispersion(),
        log_likelihood = f.log_likelihood(),
        lower = lower,
        upper = upper,
        lower_open = lower_open,
        upper_open = upper_open,
        at_boundary = f.at_boundary()
    ))
}

/// The Tweedie power's profile: `list(powers, log_likelihood, dispersion,
/// power, phi, max_log_likelihood, model)`.
#[extendr]
fn tweedie_profile_design(
    x: &[f64],
    names: Vec<String>,
    y: &[f64],
    offset: &[f64],
    weights: &[f64],
    link_name: &str,
    powers: &[f64],
) -> Result<List> {
    let l = link(link_name, Family::Poisson, f64::NAN)?;
    let d = design(x, names, offset, weights)?;
    let p = prospicio_glm::tweedie::tweedie_profile(l, &d, y, powers).map_err(to_r)?;
    Ok(list!(
        powers = p.powers,
        log_likelihood = p.log_likelihood,
        dispersion = p.dispersion,
        power = p.power,
        phi = p.phi,
        max_log_likelihood = p.max_log_likelihood,
        model = GlmModel {
            inner: p.fit,
            training: Some((d, y.to_vec())),
        }
    ))
}

/// A negative binomial GLM with `theta` estimated: `list(model, theta,
/// se_theta, log_likelihood, aic)`.
#[extendr]
fn glm_nb_fit_design(
    x: &[f64],
    names: Vec<String>,
    y: &[f64],
    offset: &[f64],
    weights: &[f64],
    link_name: &str,
) -> Result<List> {
    let l = link(link_name, Family::Poisson, f64::NAN)?;
    let d = design(x, names, offset, weights)?;
    let f = prospicio_glm::negbin::NegativeBinomialGlm::new(l)
        .fit(&d, y)
        .map_err(to_r)?;
    Ok(list!(
        model = GlmModel {
            inner: f.glm().clone(),
            training: Some((d, y.to_vec())),
        },
        theta = f.theta(),
        se_theta = f.se_theta(),
        log_likelihood = f.log_likelihood(),
        aic = f.aic()
    ))
}

/// Grouping factors from their names and every row's labels, the factors
/// one after another (`k` blocks of `n` labels).
fn groupings(names: Vec<String>, labels: Vec<String>) -> Result<Vec<Grouping>> {
    let k = names.len();
    if k == 0 {
        return Ok(Vec::new());
    }
    if labels.len() % k != 0 {
        return Err(Error::Other(
            "the grouping labels do not match the factors".into(),
        ));
    }
    let n = labels.len() / k;
    Ok(names
        .iter()
        .enumerate()
        .map(|(j, name)| Grouping::new(name, labels[j * n..(j + 1) * n].to_vec()))
        .collect())
}

/// A GLM with random intercepts (`docs/design/models.md`, "Mixed
/// models"); `nodes` 0 for the Laplace approximation.
#[allow(clippy::too_many_arguments)]
#[extendr]
fn glmm_fit_design(
    x: &[f64],
    names: Vec<String>,
    y: &[f64],
    offset: &[f64],
    weights: &[f64],
    family_name: &str,
    power: f64,
    group_names: Vec<String>,
    group_labels: Vec<String>,
    nodes: f64,
) -> Result<GlmmModel> {
    let f = match family_name {
        "poisson" => Family::Poisson,
        "gamma" => Family::Gamma,
        "tweedie" => Family::Tweedie { power },
        other => {
            return Err(Error::Other(format!(
                "a GLMM's family must be poisson, gamma or tweedie, got {other}"
            )));
        }
    };
    let d = design(x, names, offset, weights)?;
    let groups = groupings(group_names, group_labels)?;
    let mut spec = Glmm::new(f);
    let nodes = whole(nodes, "nodes")?;
    if nodes > 0 {
        spec = spec.quadrature(nodes as usize);
    }
    let inner = spec.fit(&d, &groups, y).map_err(to_r)?;
    Ok(GlmmModel { inner })
}

/// A fitted GLM with random intercepts.
#[extendr]
pub(crate) struct GlmmModel {
    inner: GlmmFit,
}

#[extendr]
impl GlmmModel {
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    fn coefficients(&self) -> Vec<f64> {
        self.inner.coefficients().to_vec()
    }

    fn std_errors(&self) -> Vec<f64> {
        self.inner.standard_errors()
    }

    fn covariance(&self) -> Vec<f64> {
        self.inner.covariance().to_vec()
    }

    fn group_names(&self) -> Vec<String> {
        self.inner.group_names().to_vec()
    }

    fn sd(&self) -> Vec<f64> {
        self.inner.sd().to_vec()
    }

    fn sd_std_errors(&self) -> Vec<f64> {
        self.inner.sd_standard_errors().to_vec()
    }

    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    fn dispersion_std_error(&self) -> f64 {
        self.inner.dispersion_standard_error()
    }

    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood()
    }

    fn aic(&self) -> f64 {
        self.inner.aic()
    }

    fn bic(&self) -> f64 {
        self.inner.bic()
    }

    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted().to_vec()
    }

    fn iterations(&self) -> f64 {
        self.inner.iterations() as f64
    }

    fn converged(&self) -> bool {
        self.inner.converged()
    }

    /// Each factor's levels, modes and variances, as a list by factor.
    fn random_effects(&self) -> List {
        let items: Vec<Robj> = self
            .inner
            .random_effects()
            .iter()
            .map(|r| {
                list!(
                    level = r.levels.clone(),
                    value = r.values.clone(),
                    conditional_variance = r.conditional_variance.clone(),
                    prediction_variance = r.prediction_variance.clone()
                )
                .into()
            })
            .collect();
        let mut out = List::from_values(items);
        let names: Vec<&str> = self
            .inner
            .group_names()
            .iter()
            .map(String::as_str)
            .collect();
        let _ = out.set_names(names);
        out
    }

    fn predict(
        &self,
        x: &[f64],
        names: Vec<String>,
        offset: &[f64],
        group_names: Vec<String>,
        group_labels: Vec<String>,
    ) -> Result<Vec<f64>> {
        let d = design(x, names, offset, &[])?;
        let groups = groupings(group_names, group_labels)?;
        self.inner.predict(&d, &groups).map_err(to_r)
    }
}

impl GlmModel {
    fn training_data(&self) -> Result<(&Design, &[f64])> {
        self.training
            .as_ref()
            .map(|(d, y)| (d, y.as_slice()))
            .ok_or_else(|| Error::Other("a loaded model has no training data".into()))
    }
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

    fn bic(&self) -> f64 {
        self.inner.bic()
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

    /// Residuals, leverages and Cook's distances on the training data.
    fn diagnostics(&self) -> Result<List> {
        let (d, y) = self.training_data()?;
        let g = self.inner.diagnostics(d, y).map_err(to_r)?;
        Ok(list!(
            fitted = g.fitted,
            response = g.response,
            pearson = g.pearson,
            deviance = g.deviance,
            working = g.working,
            working_weights = g.working_weights,
            leverage = g.leverage,
            standardized_pearson = g.standardized_pearson,
            standardized_deviance = g.standardized_deviance,
            cooks_distance = g.cooks_distance
        ))
    }

    /// Standard error of each new row's predicted mean.
    fn prediction_se(&self, x: &[f64], names: Vec<String>, offset: &[f64]) -> Result<Vec<f64>> {
        let d = design(x, names, offset, &[])?;
        self.inner.prediction_se(&d).map_err(to_r)
    }

    /// Prediction error of the sum of the new rows' means.
    fn sum_prediction_error(
        &self,
        x: &[f64],
        names: Vec<String>,
        offset: &[f64],
        weights: &[f64],
    ) -> Result<List> {
        let d = design(x, names, offset, weights)?;
        let e = self.inner.sum_prediction_error(&d).map_err(to_r)?;
        Ok(list!(
            total = e.total,
            parameter_variance = e.parameter_variance,
            process_variance = e.process_variance,
            rmsep = e.rmsep()
        ))
    }

    /// GCV on the training data.
    fn gcv(&self) -> Result<f64> {
        let (d, y) = self.training_data()?;
        self.inner.gcv(d, y).map_err(to_r)
    }

    /// Log-likelihood of the training data at a given dispersion.
    fn log_likelihood_at(&self, dispersion: f64) -> Result<f64> {
        let (d, y) = self.training_data()?;
        self.inner.log_likelihood_at(d, y, dispersion).map_err(to_r)
    }

    /// `parameters` is "normal", "mean_preserving" or "fixed".
    #[allow(clippy::too_many_arguments)]
    fn predict_distribution(
        &self,
        x: &[f64],
        names: Vec<String>,
        offset: &[f64],
        weights: &[f64],
        n_sims: f64,
        seed: f64,
        parameters: &str,
    ) -> Result<PredictiveDistribution> {
        let d = design(x, names, offset, weights)?;
        let parameters = match parameters {
            "normal" => prospicio_glm::ParameterDraws::Normal,
            "mean_preserving" => prospicio_glm::ParameterDraws::MeanPreserving,
            "fixed" => prospicio_glm::ParameterDraws::Fixed,
            other => {
                return Err(Error::Other(format!(
                    "parameters must be normal, mean_preserving or fixed, got {other}"
                )));
            }
        };
        let inner = self
            .inner
            .predict_distribution_with(
                &d,
                whole(n_sims, "n_sims")? as usize,
                whole(seed, "seed")?,
                parameters,
            )
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

    /// One JSON artifact per lambda.
    fn to_json(&self) -> Vec<String> {
        self.fits.iter().map(ElasticNetFit::to_json).collect()
    }

    fn from_json(texts: Vec<String>) -> Result<Self> {
        if texts.is_empty() {
            return Err(Error::Other(
                "an elastic-net path needs at least one fit".into(),
            ));
        }
        let fits = texts
            .iter()
            .map(|t| ElasticNetFit::from_json(t).map_err(to_r))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { fits })
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
/// "ubre", "reml" or "fixed" with `lambdas`.
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
    bs: Vec<String>,
    periods: &[f64],
    smoothing: &str,
    lambdas: &[f64],
    tensors: List,
    tensor_basis: List,
) -> Result<GamModel> {
    let f = family(family_name, theta, power)?;
    let l = link(link_name, f, f64::NAN)?;
    let base = Glm::new(f, l);
    let glm = base.dispersion(dispersion(
        dispersion_kind,
        dispersion_value,
        base.dispersion,
    )?);
    if n_basis.len() != smooths.len()
        || bs.len() != smooths.len()
        || periods.len() != 2 * smooths.len()
    {
        return Err(Error::Other(
            "give one n_basis, bs and period per smooth".into(),
        ));
    }
    let splines = smooths
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let (lo, hi) = (periods[2 * i], periods[2 * i + 1]);
            let spline = match (bs[i].as_str(), lo.is_nan()) {
                ("ps", true) => PSpline::new(s),
                ("cp", true) => PSpline::cyclic(s),
                ("cp", false) => PSpline::cyclic(s).period(lo, hi),
                ("mpi", true) => PSpline::increasing(s),
                ("mpd", true) => PSpline::decreasing(s),
                ("ps" | "mpi" | "mpd", false) => {
                    return Err(Error::Other(format!(
                        "smooth {s}: a period needs bs = \"cp\""
                    )));
                }
                (other, _) => {
                    return Err(Error::Other(format!(
                        "bs must be \"ps\", \"cp\", \"mpi\" or \"mpd\", not {other:?}"
                    )));
                }
            };
            Ok(spline.n_basis(whole(n_basis[i], "n_basis")? as usize))
        })
        .collect::<Result<Vec<_>>>()?;
    let smoothing = match smoothing {
        "auto" => Smoothing::Auto,
        "gcv" => Smoothing::Gcv,
        "ubre" => Smoothing::Ubre,
        "reml" => Smoothing::Reml,
        "fixed" => Smoothing::Fixed(lambdas.to_vec()),
        other => {
            return Err(Error::Other(format!(
                "smoothing must be auto, gcv, ubre, reml or numeric, got {other}"
            )));
        }
    };
    if tensors.len() != tensor_basis.len() {
        return Err(Error::Other("give one tensor_basis per tensor".into()));
    }
    let mut gam = Gam::new(glm, splines);
    for (cols, k) in tensors.values().zip(tensor_basis.values()) {
        let cols: Vec<String> = cols
            .as_string_vector()
            .ok_or_else(|| Error::Other("each tensor is a character vector of columns".into()))?;
        let refs: Vec<&str> = cols.iter().map(String::as_str).collect();
        let k = k
            .as_real_vector()
            .ok_or_else(|| Error::Other("tensor_basis must be numeric".into()))?
            .into_iter()
            .map(|v| whole(v, "tensor_basis").map(|v| v as usize))
            .collect::<Result<Vec<_>>>()?;
        gam = gam.tensor(prospicio_glm::gam::TensorSpline::new(&refs).n_basis(k));
    }
    let d = design(x, names, offset, weights)?;
    let inner = gam.smoothing(smoothing).fit(&d, y).map_err(to_r)?;
    Ok(GamModel { inner })
}

#[extendr]
impl GamModel {
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    fn from_json(text: &str) -> Result<Self> {
        Ok(Self {
            inner: GamFit::from_json(text).map_err(to_r)?,
        })
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

    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood()
    }

    fn aic(&self) -> f64 {
        self.inner.aic()
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

    /// The REML dispersion, `NA` for GCV and UBRE.
    fn reml_scale(&self) -> f64 {
        self.inner.reml_scale().unwrap_or(f64::NAN)
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

/// The F test (`"f"`) or likelihood ratio test (`"lr"`) of `small` within
/// `big`: `list(statistic, df, df_resid, deviance_drop, p_value)`.
#[extendr]
fn glm_nested_test_rust(small: Robj, big: Robj, kind: &str) -> Result<List> {
    let as_glm = |x: &Robj| {
        <&GlmModel>::try_from(x).map_err(|_| Error::Other("expected a glm_model".into()))
    };
    let (small, big) = (as_glm(&small)?, as_glm(&big)?);
    let t = match kind {
        "f" => prospicio_glm::nested::f_test(&small.inner, &big.inner),
        _ => prospicio_glm::nested::likelihood_ratio_test(&small.inner, &big.inner),
    }
    .map_err(to_r)?;
    Ok(list!(
        statistic = t.statistic,
        df = t.df,
        df_resid = t.df_resid,
        deviance_drop = t.deviance_drop,
        p_value = t.p_value
    ))
}

/// Variance inflation factors of a column-major design.
#[extendr]
fn vif_rust(x: &[f64], names: Vec<String>) -> Result<Vec<f64>> {
    let d = design(x, names, &[], &[])?;
    metrics::vif(&d).map_err(to_r)
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

/// Normalized Gini; empty exposure for none.
#[extendr]
fn normalized_gini_rust(y: &[f64], pred: &[f64], exposure: &[f64]) -> Result<f64> {
    let e = (!exposure.is_empty()).then_some(exposure);
    metrics::normalized_gini(y, pred, e).map_err(to_r)
}

/// Double lift bands as a list of columns.
#[extendr]
fn double_lift_rust(
    y: &[f64],
    pred_a: &[f64],
    pred_b: &[f64],
    exposure: &[f64],
    bands: f64,
) -> Result<List> {
    let e = (!exposure.is_empty()).then_some(exposure);
    let t = metrics::double_lift(y, pred_a, pred_b, e, whole(bands, "bands")? as usize)
        .map_err(to_r)?;
    let col = |f: fn(&metrics::DoubleLiftBand) -> f64| t.iter().map(f).collect::<Vec<_>>();
    Ok(list!(
        exposure = col(|b| b.exposure),
        actual = col(|b| b.actual),
        model_a = col(|b| b.model_a),
        model_b = col(|b| b.model_b),
        actual_relativity = col(|b| b.actual_relativity),
        a_relativity = col(|b| b.a_relativity),
        b_relativity = col(|b| b.b_relativity)
    ))
}

/// Loss ratio bands as a list of columns.
#[extendr]
fn loss_ratio_chart_rust(
    y: &[f64],
    pred: &[f64],
    premium: &[f64],
    exposure: &[f64],
    bands: f64,
) -> Result<List> {
    let e = (!exposure.is_empty()).then_some(exposure);
    let t = metrics::loss_ratio_chart(y, pred, premium, e, whole(bands, "bands")? as usize)
        .map_err(to_r)?;
    let col = |f: fn(&metrics::LossRatioBand) -> f64| t.iter().map(f).collect::<Vec<_>>();
    Ok(list!(
        exposure = col(|b| b.exposure),
        premium = col(|b| b.premium),
        expected = col(|b| b.expected),
        actual = col(|b| b.actual),
        loss_ratio = col(metrics::LossRatioBand::loss_ratio),
        expected_loss_ratio = col(metrics::LossRatioBand::expected_loss_ratio)
    ))
}

/// ROC curve: `list(thresholds, false_positive_rate, true_positive_rate, auc)`.
#[extendr]
fn roc_rust(y: &[f64], score: &[f64], weights: &[f64]) -> Result<List> {
    let w = (!weights.is_empty()).then_some(weights);
    let r = metrics::roc(y, score, w).map_err(to_r)?;
    Ok(list!(
        thresholds = r.thresholds,
        false_positive_rate = r.false_positive_rate,
        true_positive_rate = r.true_positive_rate,
        auc = r.auc
    ))
}

/// Confusion matrix at `threshold` as a list of weighted counts.
#[extendr]
fn confusion_matrix_rust(
    y: &[f64],
    score: &[f64],
    threshold: f64,
    weights: &[f64],
) -> Result<List> {
    let w = (!weights.is_empty()).then_some(weights);
    let c = metrics::confusion_matrix(y, score, threshold, w).map_err(to_r)?;
    Ok(list!(
        true_positive = c.true_positive,
        false_positive = c.false_positive,
        true_negative = c.true_negative,
        false_negative = c.false_negative,
        sensitivity = c.sensitivity(),
        specificity = c.specificity()
    ))
}

#[extendr]
fn crps_rust(draws: &[f64], y: f64) -> Result<f64> {
    metrics::crps(draws, y).map_err(to_r)
}

/// Empty `weights` for none.
#[extendr]
fn pinball_rust(y: &[f64], pred: &[f64], alpha: f64, weights: &[f64]) -> Result<f64> {
    let w = (!weights.is_empty()).then_some(weights);
    metrics::pinball(y, pred, alpha, w).map_err(to_r)
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

/// The PIT of each of `y` under the empirical distribution of `draws`.
#[extendr]
fn pit_from_draws_rust(draws: &[f64], y: &[f64], u: f64) -> Result<Vec<f64>> {
    y.iter()
        .map(|&v| metrics::pit_from_draws(draws, v, u).map_err(to_r))
        .collect()
}

#[extendr]
fn pit_histogram_rust(pit: &[f64], bins: f64) -> Result<Vec<f64>> {
    let counts = metrics::pit_histogram(pit, whole(bins, "bins")? as usize).map_err(to_r)?;
    Ok(counts.into_iter().map(|c| c as f64).collect())
}

/// Joint draws from fitted means: `means` is an `n_rows` by `k` matrix,
/// column-major, one column per mean vector (bootstrap refit);
/// `dispersion` one value or one per row; `weights` empty for all ones.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn simulate_from_means_rust(
    family_name: &str,
    theta: f64,
    power: f64,
    means: &[f64],
    n_rows: f64,
    dispersion: &[f64],
    weights: &[f64],
    n_sims: f64,
    seed: f64,
) -> Result<PredictiveDistribution> {
    let f = family(family_name, theta, power)?;
    let n = whole(n_rows, "n_rows")? as usize;
    if n == 0 || means.len() % n != 0 {
        return Err(Error::Other(format!(
            "means has {} values, not a whole number of columns of {n} rows",
            means.len()
        )));
    }
    let means: Vec<Vec<f64>> = means.chunks(n).map(<[f64]>::to_vec).collect();
    let provenance = prospicio_prob::Provenance::new("simulate_from_means")
        .version("prospicio", env!("CARGO_PKG_VERSION"));
    let inner = prospicio_models::simulate::from_means(
        f,
        &means,
        dispersion,
        weights,
        whole(n_sims, "n_sims")? as usize,
        whole(seed, "seed")?,
        provenance,
    )
    .map_err(to_r)?;
    Ok(PredictiveDistribution { inner })
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

/// Out-of-fold target encoding: `list(rows, levels, ratios)`; `trains` and
/// `tests` are the splits' 1-based row numbers.
#[extendr]
fn target_encode_rust(
    values: Vec<String>,
    y: &[f64],
    expected: &[f64],
    exposure: &[f64],
    trains: List,
    tests: List,
    k: f64,
) -> Result<List> {
    let rows = |x: Robj| -> Result<Vec<usize>> {
        x.as_real_vector()
            .ok_or_else(|| Error::Other("split rows must be numeric".into()))
            .map(|v| v.iter().map(|&i| i as usize - 1).collect())
    };
    let splits = trains
        .values()
        .zip(tests.values())
        .map(|(train, test)| {
            Ok(resample::Split {
                train: rows(train)?,
                test: rows(test)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let e = (!exposure.is_empty()).then_some(exposure);
    let (rows, enc) = prospicio_models::encode::target_encode(&values, y, expected, e, &splits, k)
        .map_err(to_r)?;
    Ok(list!(rows = rows, levels = enc.levels, ratios = enc.ratios))
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
        rhat = prospicio_bayes::rhat(&chains).map_err(to_r)?,
        ess_bulk = prospicio_bayes::ess_bulk(&chains).map_err(to_r)?,
        ess_tail = prospicio_bayes::ess_tail(&chains).map_err(to_r)?,
        ess_mean = prospicio_bayes::ess_mean(&chains).map_err(to_r)?,
        mcse_mean = prospicio_bayes::mcse_mean(&chains).map_err(to_r)?
    ))
}

/// An ELPD estimate as an R list.
fn elpd_list(e: prospicio_bayes::elpd::Elpd) -> List {
    list!(
        estimates = vec![e.elpd, e.se, e.p, e.ic],
        pointwise = e.pointwise
    )
}

#[extendr]
fn elpd_loo_rust(log_lik: &[f64], n: f64, r_eff: &[f64]) -> Result<List> {
    let n = whole(n, "n")? as usize;
    let r = (!r_eff.is_empty()).then_some(r_eff);
    let l = prospicio_bayes::elpd::loo(log_lik, n, r).map_err(to_r)?;
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
    Ok(elpd_list(
        prospicio_bayes::elpd::waic(log_lik, n).map_err(to_r)?,
    ))
}

/// Columns of an `n × k` column-major matrix.
fn columns(x: &[f64], k: usize) -> Result<Vec<Vec<f64>>> {
    if k == 0 || x.len() % k != 0 {
        return Err(Error::Other(
            "lpd must be a matrix with one column per model".into(),
        ));
    }
    Ok(x.chunks(x.len() / k).map(<[f64]>::to_vec).collect())
}

/// Stacking weights from an `n × k` column-major matrix of pointwise
/// log predictive densities.
#[extendr]
fn stacking_weights_rust(lpd: &[f64], k: f64) -> Result<Vec<f64>> {
    let cols = columns(lpd, whole(k, "k")? as usize)?;
    prospicio_models::stack::stacking_weights(&cols).map_err(to_r)
}

/// Pseudo-BMA(+) weights; `n_draws = 0` for no bootstrap.
#[extendr]
fn pseudo_bma_weights_rust(lpd: &[f64], k: f64, n_draws: f64, seed: f64) -> Result<Vec<f64>> {
    let cols = columns(lpd, whole(k, "k")? as usize)?;
    let n_draws = whole(n_draws, "n_draws")? as usize;
    let bb = (n_draws > 0).then_some((n_draws, whole(seed, "seed")?));
    prospicio_models::stack::pseudo_bma_weights(&cols, bb).map_err(to_r)
}

/// Actual against expected by period: `list(periods, n, weight, actual,
/// expected, std_dev, total, trend, trend_std_error)`; `periods` are the
/// sorted distinct labels' positions in `labels`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn actual_vs_expected_rust(
    periods: Robj,
    y: &[f64],
    mu: &[f64],
    weights: &[f64],
    family_name: &str,
    theta: f64,
    power: f64,
    dispersion: f64,
) -> Result<List> {
    let keys = crate::distributions::key_column(&periods)?;
    let f = family(family_name, theta, power)?;
    let w = (!weights.is_empty()).then_some(weights);
    let m = prospicio_models::monitor::actual_vs_expected(&keys, y, mu, w, f, dispersion)
        .map_err(to_r)?;
    let first: Vec<f64> = m
        .periods
        .iter()
        .map(|r| {
            let k = r.period.as_ref().expect("a period row has a period");
            (keys.iter().position(|x| x == k).expect("a key") + 1) as f64
        })
        .collect();
    let col =
        |g: fn(&prospicio_models::monitor::PeriodSummary<prospicio_prob::KeyValue>) -> f64| {
            m.periods.iter().map(g).collect::<Vec<f64>>()
        };
    let t = &m.total;
    Ok(list!(
        first_row = first,
        n = col(|r| r.n as f64),
        weight = col(|r| r.weight),
        actual = col(|r| r.actual),
        expected = col(|r| r.expected),
        std_dev = col(|r| r.std_dev),
        total = vec![t.n as f64, t.weight, t.actual, t.expected, t.std_dev],
        trend = m.trend,
        trend_std_error = m.trend_std_error
    ))
}

/// A sampled Bayesian GLM, with its training data for LOO.
#[extendr]
pub(crate) struct BayesGlmModel {
    inner: prospicio_bayes::glm::BayesGlmFit,
    training: (Design, Vec<f64>),
}

/// Samples a Bayesian GLM on the design `x` (column-major, columns
/// `names`). `dispersion` NaN means the family default (sampled for the
/// Gaussian, gamma and inverse Gaussian).
#[extendr]
#[allow(clippy::too_many_arguments)]
fn bayes_glm_fit_design(
    x: &[f64],
    names: Vec<String>,
    y: &[f64],
    offset: &[f64],
    weights: &[f64],
    family_name: &str,
    link_name: &str,
    theta: f64,
    power: f64,
    link_power: f64,
    prior_sd: f64,
    intercept_sd: f64,
    dispersion: f64,
    dispersion_scale: f64,
    sampler: &[f64],
) -> Result<BayesGlmModel> {
    use prospicio_bayes::glm::{BayesGlm, DispersionPrior, Sampler};
    let f = family(family_name, theta, power)?;
    let l = link(link_name, f, link_power)?;
    let [chains, tune, draws, seed, target_accept, max_depth] = sampler else {
        return Err(Error::Other("sampler needs six settings".into()));
    };
    let mut spec = BayesGlm::new(f, l).sampler(Sampler {
        chains: whole(*chains, "chains")? as usize,
        tune: whole(*tune, "tune")? as usize,
        draws: whole(*draws, "draws")? as usize,
        seed: whole(*seed, "seed")?,
        target_accept: *target_accept,
        max_depth: whole(*max_depth, "max_depth")?,
    });
    spec.prior_sd = prior_sd;
    spec.intercept_sd = intercept_sd;
    spec.dispersion = match spec.dispersion {
        _ if !dispersion.is_nan() => DispersionPrior::Fixed(dispersion),
        DispersionPrior::HalfNormal(_) => DispersionPrior::HalfNormal(dispersion_scale),
        fixed => fixed,
    };
    let d = design(x, names, offset, weights)?;
    let inner = spec.fit(&d, y).map_err(to_r)?;
    Ok(BayesGlmModel {
        inner,
        training: (d, y.to_vec()),
    })
}

#[extendr]
impl BayesGlmModel {
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    fn posterior_mean(&self) -> Vec<f64> {
        self.inner.posterior_mean()
    }

    /// Row-major `n_draws × p`.
    fn coefficient_draws(&self) -> Vec<f64> {
        self.inner.coefficient_draws().to_vec()
    }

    fn dispersion_draws(&self) -> Vec<f64> {
        self.inner.dispersion_draws().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.chains() as f64
    }

    fn divergences(&self) -> f64 {
        self.inner.divergences() as f64
    }

    fn summary(&self) -> Result<List> {
        let s = self.inner.summary().map_err(to_r)?;
        let col = |g: fn(&prospicio_bayes::glm::ParamSummary) -> f64| {
            s.iter().map(g).collect::<Vec<f64>>()
        };
        Ok(list!(
            parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            mean = col(|p| p.mean),
            sd = col(|p| p.sd),
            q05 = col(|p| p.q05),
            q50 = col(|p| p.q50),
            q95 = col(|p| p.q95),
            rhat = col(|p| p.rhat),
            ess_bulk = col(|p| p.ess_bulk),
            ess_tail = col(|p| p.ess_tail)
        ))
    }

    /// PSIS-LOO on the training data: `list(estimates, pointwise, pareto_k,
    /// k_threshold)`.
    fn loo(&self) -> Result<List> {
        let (d, y) = &self.training;
        let l = self.inner.loo(d, y).map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
    }

    /// Training log-likelihood, row-major `n_draws × n`.
    fn log_likelihood(&self) -> Result<Vec<f64>> {
        let (d, y) = &self.training;
        self.inner.log_likelihood(d, y).map_err(to_r)
    }

    /// Replicated training datasets, row-major `n_draws × n`.
    fn replicate(&self, seed: f64) -> Result<Vec<f64>> {
        let (d, _) = &self.training;
        self.inner.replicate(d, whole(seed, "seed")?).map_err(to_r)
    }

    /// Posterior predictive checks on the training data.
    fn ppc(&self, statistics: Vec<String>, seed: f64) -> Result<List> {
        let (d, y) = &self.training;
        let stats = ppc_statistics(&statistics)?;
        let checks = self
            .inner
            .posterior_predictive_check(d, y, &stats, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(checks_list(checks))
    }

    /// The leave-one-out PIT of the training data.
    fn loo_pit(&self, seed: f64) -> Result<List> {
        let (d, y) = &self.training;
        let p = self
            .inner
            .loo_pit(d, y, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(list!(pit = p.pit, pareto_k = p.pareto_k, ks = p.ks))
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

/// Posterior stacking weights.
#[extendr]
pub(crate) struct StackingModel {
    inner: prospicio_bayes::stacking::StackingFit,
    k: usize,
}

fn stacking_sampler(sampler: &[f64]) -> Result<prospicio_bayes::glm::Sampler> {
    let [chains, tune, draws, seed] = sampler else {
        return Err(Error::Other("sampler needs four settings".into()));
    };
    Ok(prospicio_bayes::glm::Sampler {
        chains: whole(*chains, "chains")? as usize,
        tune: whole(*tune, "tune")? as usize,
        draws: whole(*draws, "draws")? as usize,
        seed: whole(*seed, "seed")?,
        ..prospicio_bayes::glm::Sampler::default()
    })
}

/// Bayesian stacking of an `n × k` column-major matrix of log densities;
/// `concentration` empty for a uniform Dirichlet.
#[extendr]
fn bayes_stacking_rust(
    lpd: &[f64],
    k: f64,
    concentration: &[f64],
    sampler: &[f64],
) -> Result<StackingModel> {
    let k = whole(k, "k")? as usize;
    let cols = columns(lpd, k)?;
    let spec = prospicio_bayes::stacking::BayesStacking {
        concentration: (!concentration.is_empty()).then(|| concentration.to_vec()),
        sampler: stacking_sampler(sampler)?,
    };
    let inner = spec.fit(&cols).map_err(to_r)?;
    Ok(StackingModel { inner, k })
}

/// Hierarchical stacking; `x` is `n × p` column-major covariates, `priors`
/// is `c(alpha_loc, alpha_scale, beta_loc, beta_scale)`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn hierarchical_stacking_rust(
    lpd: &[f64],
    k: f64,
    x: &[f64],
    p: f64,
    priors: &[f64],
    pooling: &[f64],
    adaptive: Option<f64>,
    discrete: f64,
    sampler: &[f64],
) -> Result<StackingModel> {
    let k = whole(k, "k")? as usize;
    let cols = columns(lpd, k)?;
    let xs = columns(x, whole(p, "p")? as usize)?;
    let [alpha_loc, alpha_scale, beta_loc, beta_scale] = priors else {
        return Err(Error::Other("priors needs four values".into()));
    };
    let spec = prospicio_bayes::stacking::HierarchicalStacking {
        alpha_loc: *alpha_loc,
        alpha_scale: *alpha_scale,
        beta_loc: *beta_loc,
        beta_scale: *beta_scale,
        pooling: match pooling {
            [] => None,
            [g, md, mc, sd, sc] => Some(prospicio_bayes::stacking::Pooling {
                tau_mu_global: *g,
                tau_mu_discrete: *md,
                tau_mu_continuous: *mc,
                tau_sigma_discrete: *sd,
                tau_sigma_continuous: *sc,
            }),
            _ => return Err(Error::Other("pooling needs five scales".into())),
        },
        adaptive,
        discrete: whole(discrete, "discrete")? as usize,
        sampler: stacking_sampler(sampler)?,
    };
    let inner = spec.fit(&cols, &xs).map_err(to_r)?;
    Ok(StackingModel { inner, k })
}

#[extendr]
impl StackingModel {
    /// Posterior mean weights, `n × k` column-major, for `n × p`
    /// column-major covariates (`p = 0`: one row).
    fn weights(&self, x: &[f64], p: f64) -> Result<Vec<f64>> {
        let p = whole(p, "p")? as usize;
        let xs = if p == 0 { Vec::new() } else { columns(x, p)? };
        let w = self.inner.weights(&xs).map_err(to_r)?;
        let n = w.len() / self.k;
        Ok((0..self.k)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .map(|(i, j)| w[i * self.k + j])
            .collect())
    }

    fn alpha_draws(&self) -> Vec<f64> {
        self.inner.alpha_draws()
    }

    fn beta_draws(&self) -> Vec<f64> {
        self.inner.beta_draws()
    }

    fn divergences(&self) -> f64 {
        self.inner.divergences() as f64
    }

    /// `list(rhat, ess_bulk)` per sampled parameter.
    fn rhat_ess(&self) -> Result<List> {
        let r = self.inner.rhat_ess().map_err(to_r)?;
        Ok(list!(
            rhat = r.iter().map(|v| v.0).collect::<Vec<_>>(),
            ess_bulk = r.iter().map(|v| v.1).collect::<Vec<_>>()
        ))
    }
}

/// Test statistics from R: names, or numbers (or `"q<p>"`) for quantiles.
fn ppc_statistics(names: &[String]) -> Result<Vec<Statistic>> {
    names
        .iter()
        .map(|name| match name.as_str() {
            "mean" => Ok(Statistic::Mean),
            "sd" => Ok(Statistic::Sd),
            "min" => Ok(Statistic::Min),
            "max" => Ok(Statistic::Max),
            "proportion_zero" => Ok(Statistic::ProportionZero),
            other => other
                .strip_prefix('q')
                .unwrap_or(other)
                .parse::<f64>()
                .map(Statistic::Quantile)
                .map_err(|_| {
                    Error::Other(format!(
                        "a statistic is \"mean\", \"sd\", \"min\", \"max\", \"proportion_zero\" or a quantile's probability, got {other}"
                    ))
                }),
        })
        .collect()
}

/// `list(name, observed, p_value, replicated)`, `replicated` row-major
/// draws × statistics.
fn checks_list(checks: Vec<StatisticCheck>) -> List {
    let s = checks.first().map_or(0, |c| c.replicated.len());
    let mut replicated = Vec::with_capacity(s * checks.len());
    for d in 0..s {
        replicated.extend(checks.iter().map(|c| c.replicated[d]));
    }
    list!(
        name = checks.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        observed = checks.iter().map(|c| c.observed).collect::<Vec<_>>(),
        p_value = checks.iter().map(|c| c.p_value).collect::<Vec<_>>(),
        replicated = replicated
    )
}

/// Posterior predictive checks of any model's replicates (row-major
/// `S × n`); `groups` empty for none, else from 1.
#[extendr]
fn ppc_check_rust(
    observed: &[f64],
    replicated: &[f64],
    statistics: Vec<String>,
    groups: &[f64],
) -> Result<List> {
    let stats = ppc_statistics(&statistics)?;
    let checks = if groups.is_empty() {
        ppc::check(observed, replicated, &stats)
    } else {
        let g = groups
            .iter()
            .map(|&v| {
                let k = whole(v, "groups")?;
                if k == 0 {
                    return Err(Error::Other("groups run from 1".into()));
                }
                Ok(k as usize - 1)
            })
            .collect::<Result<Vec<_>>>()?;
        ppc::check_grouped(observed, replicated, &g, &stats)
    }
    .map_err(to_r)?;
    Ok(checks_list(checks))
}

/// The leave-one-out PIT of any model (`replicated` and `log_lik` row-major
/// `S × n`; `r_eff` empty for 1).
#[extendr]
fn loo_pit_rust(
    observed: &[f64],
    replicated: &[f64],
    log_lik: &[f64],
    r_eff: &[f64],
    seed: f64,
) -> Result<List> {
    let r = (!r_eff.is_empty()).then_some(r_eff);
    let p = ppc::loo_pit(observed, replicated, log_lik, r, whole(seed, "seed")?).map_err(to_r)?;
    Ok(list!(pit = p.pit, pareto_k = p.pareto_k, ks = p.ks))
}

extendr_module! {
    mod models;
    fn bayes_stacking_rust;
    fn hierarchical_stacking_rust;
    impl StackingModel;
    fn bayes_glm_fit_design;
    impl BayesGlmModel;
    fn ppc_check_rust;
    fn loo_pit_rust;
    fn actual_vs_expected_rust;
    fn stacking_weights_rust;
    fn pseudo_bma_weights_rust;
    fn glm_fit_design;
    fn tweedie_glm_fit_design;
    fn tweedie_profile_design;
    fn glm_nb_fit_design;
    fn glm_nested_test_rust;
    fn vif_rust;
    fn elastic_net_fit_design;
    fn elastic_net_cv_design;
    fn gam_fit_design;
    impl GlmModel;
    impl ElasticNetPath;
    impl GamModel;
    fn family_deviance_rust;
    fn gini_rust;
    fn lift_rust;
    fn normalized_gini_rust;
    fn double_lift_rust;
    fn loss_ratio_chart_rust;
    fn roc_rust;
    fn confusion_matrix_rust;
    fn crps_rust;
    fn pinball_rust;
    fn log_score_rust;
    fn pit_rust;
    fn ks_uniform_rust;
    fn pit_from_draws_rust;
    fn pit_histogram_rust;
    fn simulate_from_means_rust;
    fn k_fold_rust;
    fn target_encode_rust;
    fn group_k_fold_rust;
    fn time_ordered_rust;
    fn mcmc_diagnostics_rust;
    fn elpd_loo_rust;
    fn elpd_waic_rust;
    fn glmm_fit_design;
    impl GlmmModel;
}
