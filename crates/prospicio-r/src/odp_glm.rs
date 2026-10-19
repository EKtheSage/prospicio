//! `odp_glm()` (issue #333): the ODP model as a quasi-Poisson GLM on a
//! triangle, over `prospicio_reserving::OdpGlm`, with the reserves'
//! analytic prediction error.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_glm::ParameterDraws;
use prospicio_reserving::{OdpGlm, OdpGlmFit as OdpGlmFitInner, ResidualGroup, SumPredictionError};

use crate::distributions::PredictiveDistribution;
use crate::reserving::Triangle;
use crate::whole;

fn to_r(e: prospicio_reserving::Error) -> Error {
    Error::Other(e.to_string())
}

/// A fitted ODP GLM.
#[extendr]
pub(crate) struct OdpGlmFit {
    inner: OdpGlmFitInner,
}

#[extendr]
impl OdpGlmFit {
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    fn reserves(&self) -> Vec<f64> {
        self.inner.reserves.clone()
    }

    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    fn dispersion(&self) -> f64 {
        self.inner.glm.dispersion()
    }

    fn coefficient_names(&self) -> Vec<String> {
        self.inner.coding.names()
    }

    fn coefficients(&self) -> Vec<f64> {
        self.inner.glm.coefficients().to_vec()
    }

    fn std_errors(&self) -> Vec<f64> {
        self.inner.glm.std_errors()
    }

    fn covariance(&self) -> Vec<f64> {
        self.inner.glm.covariance()
    }

    fn future(&self) -> List {
        let origin: Vec<String> = self
            .inner
            .future
            .iter()
            .map(|(o, _)| o.to_string())
            .collect();
        let development: Vec<f64> = self
            .inner
            .future
            .iter()
            .map(|(_, a)| f64::from(*a))
            .collect();
        list!(
            origin = origin,
            development = development,
            mean = self.inner.future_means.clone(),
            predictive_mean = self.inner.predictive_means()
        )
    }

    fn prediction_error(&self) -> Result<List> {
        let error = self.inner.prediction_error().map_err(to_r)?;
        let rows: Vec<&SumPredictionError> = error.by_origin.iter().chain([&error.total]).collect();
        let col = |f: fn(&SumPredictionError) -> f64| rows.iter().map(|e| f(e)).collect::<Vec<_>>();
        let mut origin = self.origins();
        origin.push("Total".into());
        Ok(list!(
            origin = origin,
            reserve = col(|e| e.total),
            process_se = col(|e| e.process_variance.sqrt()),
            parameter_se = col(|e| e.parameter_variance.sqrt()),
            rmsep = col(SumPredictionError::rmsep),
            cv = col(|e| if e.total == 0.0 {
                f64::NAN
            } else {
                e.rmsep() / e.total
            })
        ))
    }

    fn residuals(&self) -> Result<List> {
        let r = self.inner.residuals().map_err(to_r)?;
        let origin: Vec<String> = r.origin.iter().map(ToString::to_string).collect();
        let development: Vec<f64> = r.development.iter().map(|&a| f64::from(a)).collect();
        let calendar: Vec<String> = r.calendar.iter().map(|c| c.1.clone()).collect();
        let calendar_index: Vec<f64> = r.calendar.iter().map(|c| c.0 as f64).collect();
        Ok(list!(
            origin = origin,
            development = development,
            calendar = calendar,
            calendar_index = calendar_index,
            actual = r.actual,
            fitted = r.fitted,
            actual_to_fitted = r.actual_to_fitted,
            pearson = r.pearson,
            deviance = r.deviance,
            leverage = r.leverage,
            standardized_deviance = r.standardized_deviance,
            standardized_pearson = r.standardized_pearson
        ))
    }

    fn residual_summary(&self, by: &str) -> Result<List> {
        let group = match by {
            "development" => ResidualGroup::Development,
            "origin" => ResidualGroup::Origin,
            "calendar" => ResidualGroup::Calendar,
            _ => {
                return Err(Error::Other(
                    "by must be \"development\", \"origin\" or \"calendar\"".into(),
                ));
            }
        };
        let rows = self.inner.residuals().map_err(to_r)?.summary(group);
        let label: Vec<String> = rows.iter().map(|g| g.label.clone()).collect();
        let n: Vec<f64> = rows.iter().map(|g| g.n as f64).collect();
        let mean: Vec<f64> = rows.iter().map(|g| g.mean).collect();
        let sd: Vec<f64> = rows.iter().map(|g| g.sd).collect();
        Ok(list!(label = label, n = n, mean = mean, sd = sd))
    }

    fn predict_distribution(
        &self,
        n_sims: f64,
        seed: f64,
        parameters: &str,
    ) -> Result<PredictiveDistribution> {
        let parameters = match parameters {
            "mean_preserving" => ParameterDraws::MeanPreserving,
            "normal" => ParameterDraws::Normal,
            "fixed" => ParameterDraws::Fixed,
            _ => {
                return Err(Error::Other(
                    "parameters must be \"mean_preserving\", \"normal\" or \"fixed\"".into(),
                ));
            }
        };
        let inner = self
            .inner
            .predict_distribution_with(
                whole(n_sims, "n_sims")? as usize,
                whole(seed, "seed")?,
                parameters,
            )
            .map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

/// Fits the ODP GLM of `odp_glm()` to `column` of a single-segment triangle.
#[extendr]
fn odp_glm_fit_rust(
    triangle: Robj,
    column: &str,
    tolerance: f64,
    max_iterations: f64,
) -> Result<OdpGlmFit> {
    let triangle = <&Triangle>::try_from(&triangle)
        .map_err(|_| Error::Other("triangle must be a triangle".into()))?;
    let model = OdpGlm {
        tolerance,
        max_iterations: whole(max_iterations, "max_iterations")? as usize,
    };
    let inner = model.fit(&triangle.inner, column).map_err(to_r)?;
    Ok(OdpGlmFit { inner })
}

extendr_module! {
    mod odp_glm;
    impl OdpGlmFit;
    fn odp_glm_fit_rust;
}
