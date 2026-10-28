//! Aggregate lane: classical credibility and Hachemeister's regression
//! model, and Models lane: Bayesian
//! hierarchical credibility, for the R `credibility.R` API
//! (`docs/design/credibility.md`).

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_bayes::credibility::{
    CredibilityPriors, HierarchicalCredibility, HierarchicalCredibilityFit, Sampler, ScalePrior,
};
use prospicio_pricing::credibility::{
    BuhlmannStraub, Hachemeister, LimitedFluctuation, StructureEstimator,
};

use crate::distributions::PredictiveDistribution;
use crate::{to_r, whole};

/// Full limited-fluctuation standards for `p`, `k` and severity `cv`:
/// `c(frequency, severity, pure_premium)`.
#[extendr]
fn credibility_standards(p: f64, k: f64, cv: f64) -> Result<Vec<f64>> {
    let lf = LimitedFluctuation::new(p, k).map_err(to_r)?;
    Ok(vec![
        lf.frequency_standard(),
        lf.severity_standard(cv),
        lf.pure_premium_standard(cv),
    ])
}

/// Rows of a column-major `nrow` x `ncol` matrix.
fn rows(values: &[f64], nrow: usize, ncol: usize) -> Result<Vec<Vec<f64>>> {
    if values.len() != nrow * ncol {
        return Err(Error::Other(
            "the matrix's length is not nrow * ncol".into(),
        ));
    }
    Ok((0..nrow)
        .map(|i| (0..ncol).map(|j| values[i + j * nrow]).collect())
        .collect())
}

/// The Bühlmann–Straub model on a groups x periods matrix (column-major
/// `ratios`, `weights` empty for equal weights).
#[extendr]
fn buhlmann_straub_rust(
    ratios: &[f64],
    weights: &[f64],
    nrow: f64,
    ncol: f64,
    estimator: &str,
    tol: f64,
    max_iter: f64,
) -> Result<List> {
    let (nrow, ncol) = (whole(nrow, "nrow")? as usize, whole(ncol, "ncol")? as usize);
    let estimator = structure_estimator(estimator, tol, max_iter)?;
    let r = rows(ratios, nrow, ncol)?;
    let w = if weights.is_empty() {
        None
    } else {
        Some(rows(weights, nrow, ncol)?)
    };
    let fit = BuhlmannStraub::new(estimator)
        .fit(&r, w.as_deref())
        .map_err(to_r)?;
    Ok(list!(
        collective_mean = fit.collective_mean,
        between_variance = fit.between_variance,
        within_variance = fit.within_variance,
        k = fit.k(),
        group_means = fit.group_means,
        group_weights = fit.group_weights,
        credibility = fit.credibility,
        premiums = fit.premiums,
        iterations = fit.iterations as f64
    ))
}

fn structure_estimator(estimator: &str, tol: f64, max_iter: f64) -> Result<StructureEstimator> {
    match estimator {
        "unbiased" => Ok(StructureEstimator::Unbiased),
        "iterative" => Ok(StructureEstimator::Iterative {
            tol,
            max_iter: whole(max_iter, "max_iter")? as usize,
        }),
        other => Err(Error::Other(format!(
            "estimator must be \"unbiased\" or \"iterative\", got {other}"
        ))),
    }
}

/// Hachemeister's regression model on a risks x periods matrix and a
/// periods x p design (both column-major, `weights` empty for equal
/// weights). Matrices come back column-major; `credibility` holds each
/// risk's `p x p` matrix in turn, as R's `array(, c(p, p, risks))`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn hachemeister_rust(
    ratios: &[f64],
    weights: &[f64],
    nrow: f64,
    ncol: f64,
    design: &[f64],
    p: f64,
    estimator: &str,
    adjust_intercept: bool,
    tol: f64,
    max_iter: f64,
) -> Result<List> {
    let (nrow, ncol) = (whole(nrow, "nrow")? as usize, whole(ncol, "ncol")? as usize);
    let p = whole(p, "p")? as usize;
    let r = rows(ratios, nrow, ncol)?;
    let w = if weights.is_empty() {
        None
    } else {
        Some(rows(weights, nrow, ncol)?)
    };
    let y = rows(design, ncol, p)?;
    let fit = Hachemeister::new(
        structure_estimator(estimator, tol, max_iter)?,
        adjust_intercept,
    )
    .fit(&r, w.as_deref(), &y)
    .map_err(to_r)?;
    // Column-major from row-major nested rows.
    let cm = |m: &[Vec<f64>]| -> Vec<f64> {
        let cols = m.first().map_or(0, Vec::len);
        (0..cols)
            .flat_map(|j| m.iter().map(move |row| row[j]))
            .collect()
    };
    Ok(list!(
        collective = fit.collective.clone(),
        coefficients = cm(&fit.coefficients),
        individual = cm(&fit.individual),
        credibility = fit
            .credibility
            .iter()
            .flat_map(|z| cm(z))
            .collect::<Vec<f64>>(),
        between = cm(&fit.between),
        within_variance = fit.within_variance,
        risk_weights = fit.risk_weights.clone(),
        transition = fit.transition.as_deref().map(cm).unwrap_or_default(),
        iterations = fit.iterations as f64
    ))
}

/// The hierarchy from R: a list of integer vectors, one per level
/// (outermost first), each group's node from 1.
fn levels_from(levels: List, groups: usize) -> Result<Vec<Vec<usize>>> {
    levels
        .values()
        .map(|v| {
            let v = v
                .as_real_vector()
                .ok_or_else(|| Error::Other("each level must be a numeric vector".into()))?;
            if v.len() != groups {
                return Err(Error::Other(format!(
                    "a level names {} nodes for {groups} groups",
                    v.len()
                )));
            }
            v.iter()
                .map(|&x| {
                    let k = whole(x, "levels")?;
                    if k == 0 {
                        return Err(Error::Other("levels' nodes run from 1".into()));
                    }
                    Ok(k as usize - 1)
                })
                .collect()
        })
        .collect()
}

/// A scale prior from `kind` ("half_normal" or "half_cauchy") and `scale`.
fn scale_prior(kind: &str, scale: f64) -> Result<ScalePrior> {
    match kind {
        "half_normal" => Ok(ScalePrior::HalfNormal(scale)),
        "half_cauchy" => Ok(ScalePrior::HalfCauchy(scale)),
        other => Err(Error::Other(format!(
            "a scale prior is \"half_normal\" or \"half_cauchy\", got {other}"
        ))),
    }
}

/// Bayesian hierarchical credibility on a groups x periods matrix
/// (column-major). `collective_mean` empty for a flat prior or
/// `c(mean, sd)`; `within_kind` empty for the default; `between_kinds`
/// and `between_scales` empty for the defaults or one per level.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn hierarchical_credibility_rust(
    ratios: &[f64],
    weights: &[f64],
    nrow: f64,
    ncol: f64,
    levels: List,
    collective_mean: &[f64],
    within_kind: &str,
    within_scale: f64,
    between_kinds: Vec<String>,
    between_scales: &[f64],
    chains: f64,
    tune: f64,
    draws: f64,
    seed: f64,
) -> Result<HierarchicalCredibilityR> {
    let (nrow, ncol) = (whole(nrow, "nrow")? as usize, whole(ncol, "ncol")? as usize);
    let r = rows(ratios, nrow, ncol)?;
    let w = rows(weights, nrow, ncol)?;
    let levels = levels_from(levels, nrow)?;
    let collective_mean = match collective_mean {
        [] => None,
        [m, s] => Some((*m, *s)),
        _ => {
            return Err(Error::Other(
                "collective_mean is c(mean, sd) or NULL".into(),
            ));
        }
    };
    let within = if within_kind.is_empty() {
        None
    } else {
        Some(scale_prior(within_kind, within_scale)?)
    };
    if between_kinds.len() != between_scales.len() {
        return Err(Error::Other("one scale per between prior".into()));
    }
    let between = between_kinds
        .iter()
        .zip(between_scales)
        .map(|(k, &s)| scale_prior(k, s))
        .collect::<Result<Vec<_>>>()?;
    let spec = HierarchicalCredibility::default()
        .priors(CredibilityPriors {
            collective_mean,
            within,
            between,
        })
        .sampler(Sampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: whole(seed, "seed")?,
            ..Sampler::default()
        });
    let inner = spec.fit(&r, &w, &levels).map_err(to_r)?;
    Ok(HierarchicalCredibilityR { inner })
}

/// Credibility premiums at given structure parameters.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn credibility_premiums_rust(
    ratios: &[f64],
    weights: &[f64],
    nrow: f64,
    ncol: f64,
    levels: List,
    within_variance: f64,
    between_variances: &[f64],
    collective_mean: &[f64],
) -> Result<List> {
    let (nrow, ncol) = (whole(nrow, "nrow")? as usize, whole(ncol, "ncol")? as usize);
    let r = rows(ratios, nrow, ncol)?;
    let w = rows(weights, nrow, ncol)?;
    let levels = levels_from(levels, nrow)?;
    let collective_mean = match collective_mean {
        [] => None,
        [m, s] => Some((*m, *s)),
        _ => {
            return Err(Error::Other(
                "collective_mean is c(mean, sd) or NULL".into(),
            ));
        }
    };
    let p = prospicio_bayes::credibility::credibility_premiums(
        &r,
        &w,
        &levels,
        within_variance,
        between_variances,
        collective_mean,
    )
    .map_err(to_r)?;
    Ok(list!(
        premiums = p.premiums,
        credibility = p.credibility,
        level_premiums = List::from_values(p.level_premiums),
        level_credibility = List::from_values(p.level_credibility),
        collective_mean = p.collective_mean
    ))
}

/// A fitted Bayesian hierarchical credibility model.
#[extendr]
pub(crate) struct HierarchicalCredibilityR {
    inner: HierarchicalCredibilityFit,
}

#[extendr]
impl HierarchicalCredibilityR {
    fn values(&self) -> List {
        let f = &self.inner;
        list!(
            premiums = f.premiums.clone(),
            credibility = f.credibility.clone(),
            level_premiums = List::from_values(f.level_premiums.clone()),
            collective_mean = f.collective_mean,
            group_means = f.group_means.clone(),
            group_weights = f.group_weights.clone(),
            divergences = f.divergences() as f64
        )
    }

    /// `name, mean, sd, q05, q50, q95, rhat, ess_bulk, ess_tail` columns.
    fn summary(&self) -> Result<List> {
        let s = self.inner.summary().map_err(to_r)?;
        let col = |f: fn(&prospicio_bayes::credibility::ParamSummary) -> f64| -> Vec<f64> {
            s.iter().map(f).collect()
        };
        Ok(list!(
            name = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
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

    /// `weights` empty for the risk premiums.
    fn predictive(
        &self,
        weights: &[f64],
        n_sims: f64,
        seed: f64,
    ) -> Result<PredictiveDistribution> {
        let w = (!weights.is_empty()).then_some(weights);
        let inner = self
            .inner
            .predictive(w, whole(n_sims, "n_sims")? as usize, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

extendr_module! {
    mod credibility;
    fn credibility_standards;
    fn buhlmann_straub_rust;
    fn hachemeister_rust;
    fn hierarchical_credibility_rust;
    fn credibility_premiums_rust;
    impl HierarchicalCredibilityR;
}
