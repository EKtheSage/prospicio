//! `odp_glm()` (issue #333): the ODP model as a quasi-Poisson GLM on a
//! triangle, over `prospicio_reserving::OdpGlm`, with the reserves'
//! analytic prediction error.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_glm::ParameterDraws;
use prospicio_reserving::{OdpGlm, OdpGlmFit as OdpGlmFitInner, SumPredictionError};

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
