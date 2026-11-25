//! Models lane: the arithmetic R's boosters share with Python's
//! (`prospicio_models::boosting`, `metrics::mean_deviance`; #977), for
//! `boosting.R` and `deviance_score()`.

use extendr_api::Result;
use extendr_api::prelude::*;
use prospicio_models::{Family, boosting, metrics};

use crate::models::{family, link};
use crate::to_r;

/// `Σ w d(y, μ) / Σ w`; empty weights for none.
#[extendr]
fn family_mean_deviance_rust(
    family_name: &str,
    theta: f64,
    power: f64,
    y: &[f64],
    mu: &[f64],
    weights: &[f64],
) -> Result<f64> {
    let f = family(family_name, theta, power)?;
    let w = (!weights.is_empty()).then_some(weights);
    metrics::mean_deviance(f, y, mu, w).map_err(to_r)
}

/// The smallest `y` whose cumulative weight reaches `alpha` of the total.
#[extendr]
fn weighted_quantile_rust(y: &[f64], w: &[f64], alpha: f64) -> Result<f64> {
    boosting::weighted_quantile(y, w, alpha).map_err(to_r)
}

/// A booster's starting score for a `"log"` or `"identity"` link.
#[extendr]
fn booster_base_score_rust(link_name: &str, y: &[f64], w: &[f64], offset: &[f64]) -> Result<f64> {
    let l = link(link_name, Family::Gaussian, f64::NAN)?;
    boosting::base_score(l, y, w, offset).map_err(to_r)
}

/// Each row's Pearson term `w (y - μ)² / V(μ)`.
#[extendr]
fn pearson_terms_rust(
    family_name: &str,
    theta: f64,
    power: f64,
    y: &[f64],
    mu: &[f64],
    w: &[f64],
) -> Result<Vec<f64>> {
    let f = family(family_name, theta, power)?;
    boosting::pearson_terms(f, y, mu, w).map_err(to_r)
}

/// The Pearson dispersion `Σ w (y - μ)² / V(μ) / n`.
#[extendr]
fn pearson_dispersion_rust(
    family_name: &str,
    theta: f64,
    power: f64,
    y: &[f64],
    mu: &[f64],
    w: &[f64],
) -> Result<f64> {
    let f = family(family_name, theta, power)?;
    boosting::pearson_dispersion(f, y, mu, w).map_err(to_r)
}

/// The 1-based positions of `means` whose step from the one before goes
/// against `direction`.
#[extendr]
fn monotone_violations_rust(means: &[f64], direction: i32, tol: f64) -> Result<Vec<i32>> {
    let bad = boosting::monotone_violations(means, direction, tol).map_err(to_r)?;
    Ok(bad.into_iter().map(|i| i as i32 + 1).collect())
}

extendr_module! {
    mod boosting;
    fn family_mean_deviance_rust;
    fn weighted_quantile_rust;
    fn booster_base_score_rust;
    fn pearson_terms_rust;
    fn pearson_dispersion_rust;
    fn monotone_violations_rust;
}
