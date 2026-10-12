//! Aggregate lane: classical credibility for the R `credibility.R` API
//! (`docs/design/credibility.md`).

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_pricing::credibility::{BuhlmannStraub, LimitedFluctuation, StructureEstimator};

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
    let estimator = match estimator {
        "unbiased" => StructureEstimator::Unbiased,
        "iterative" => StructureEstimator::Iterative {
            tol,
            max_iter: whole(max_iter, "max_iter")? as usize,
        },
        other => {
            return Err(Error::Other(format!(
                "estimator must be \"unbiased\" or \"iterative\", got {other}"
            )));
        }
    };
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

extendr_module! {
    mod credibility;
    fn credibility_standards;
    fn buhlmann_straub_rust;
}
