//! Mack's distribution-free chain ladder: the chain-ladder projection plus
//! the mean squared error of each origin's and the total reserve.

use crate::chain_ladder::{ChainLadder, ChainLadderFit};
use crate::development::Development;
use crate::error::{Error, Result};
use crate::triangle::Triangle;

/// Mack's chain ladder (Mack 1993, 1999), with the process and parameter
/// risk recursions of R ChainLadder's `MackChainLadder` and no tail.
///
/// The factors use the development estimator's `alpha`: the conditional
/// variance of `C[k+1]` given `C[k]` is `sigma_k^2 C[k]^(2 - alpha)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mack {
    /// How factors are estimated and unestimable sigmas filled in.
    pub development: Development,
}

/// A fitted Mack model. Risks are standard errors (square roots of the
/// variance components) of each origin's ultimate, which equal those of its
/// reserve since the latest value is known.
#[derive(Debug, Clone, PartialEq)]
pub struct MackFit {
    /// The underlying chain-ladder projection.
    pub chain_ladder: ChainLadderFit,
    /// Process standard error per origin.
    pub process_risk: Vec<f64>,
    /// Parameter (estimation) standard error per origin.
    pub parameter_risk: Vec<f64>,
    /// Mack standard error per origin: `sqrt(process^2 + parameter^2)`.
    pub standard_error: Vec<f64>,
    /// Process standard error of the total reserve.
    pub total_process_risk: f64,
    /// Parameter standard error of the total reserve, including the
    /// correlation between origins that share estimated factors.
    pub total_parameter_risk: f64,
    /// Mack standard error of the total reserve.
    pub total_standard_error: f64,
}

impl Mack {
    /// Fits `column` of a single-segment triangle. Needs at least three
    /// development ages, and every sigma estimable (two or more link ratios
    /// at that age) or fillable: log-linear needs two estimated positive
    /// sigmas, Mack's rule the two before the gap. A square triangle
    /// therefore needs at least four ages.
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<MackFit> {
        let n_dev = triangle.shape()[3];
        if n_dev < 3 {
            return Err(Error::TooFewAges {
                needed: 3,
                found: n_dev,
            });
        }
        let chain_ladder = ChainLadder {
            development: self.development,
            tail: 1.0,
        }
        .fit(triangle, column)?;

        let dev = &chain_ladder.development;
        let (ldf, sigma, std_err, alpha) = (&dev.ldf, &dev.sigma, &dev.std_err, dev.alpha);
        if let Some(age) = sigma.iter().position(|s| s.is_nan()) {
            return Err(Error::Factor {
                age,
                reason: "sigma can neither be estimated nor interpolated",
            });
        }
        let n_links = ldf.len();
        let n_origins = chain_ladder.origins.len();

        // Per-origin recursions over the projected ages, and the total
        // parameter variance, which chains the sum of the projected values of
        // every origin still developing at each age.
        let mut process_var = vec![0.0; n_origins];
        let mut parameter_var = vec![0.0; n_origins];
        let mut total_parameter_var = 0.0;
        let projections: Vec<Vec<f64>> =
            (0..n_origins).map(|o| chain_ladder.projection(o)).collect();
        for k in 0..n_links {
            let f2 = ldf[k] * ldf[k];
            let mut developing = 0.0;
            for o in 0..n_origins {
                let start = chain_ladder.latest_position[o];
                if k < start {
                    continue;
                }
                let c = projections[o][k - start];
                process_var[o] = process_var[o] * f2 + sigma[k].powi(2) * c.powf(2.0 - alpha);
                parameter_var[o] = parameter_var[o] * f2 + (c * std_err[k]).powi(2);
                developing += c;
            }
            total_parameter_var = total_parameter_var * f2 + (developing * std_err[k]).powi(2);
        }

        let total_process_var: f64 = process_var.iter().sum();
        Ok(MackFit {
            process_risk: process_var.iter().map(|v| v.sqrt()).collect(),
            parameter_risk: parameter_var.iter().map(|v| v.sqrt()).collect(),
            standard_error: process_var
                .iter()
                .zip(&parameter_var)
                .map(|(p, q)| (p + q).sqrt())
                .collect(),
            total_process_risk: total_process_var.sqrt(),
            total_parameter_risk: total_parameter_var.sqrt(),
            total_standard_error: (total_process_var + total_parameter_var).sqrt(),
            chain_ladder,
        })
    }
}

impl MackFit {
    /// Coefficient of variation of the total reserve.
    pub fn total_cv(&self) -> f64 {
        self.total_standard_error / self.chain_ladder.total_reserve()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain_ladder::ChainLadder;
    use crate::development::SigmaInterpolation;
    use crate::triangle::tests::{annual, raa};

    fn close(got: f64, want: f64, tol: f64) {
        assert!((got - want).abs() < tol, "got {got}, want {want}");
    }

    #[test]
    fn raa_standard_error_mack_sigma() {
        // R ChainLadder: MackChainLadder(RAA, est.sigma = "Mack").
        let mack = Mack {
            development: Development {
                sigma_interpolation: SigmaInterpolation::Mack,
                ..Default::default()
            },
        }
        .fit(&raa(), "values")
        .unwrap();
        close(mack.total_standard_error, 26_909.01, 0.01);
        close(mack.total_cv(), 0.5161, 1e-4);
        assert_eq!(mack.standard_error[0], 0.0);
    }

    #[test]
    fn raa_standard_error_log_linear() {
        // chainladder-python: MackChainladder().fit(raa).total_mack_std_err_.
        let mack = Mack::default().fit(&raa(), "values").unwrap();
        close(mack.total_standard_error, 26_880.740_33, 1e-4);
    }

    #[test]
    fn needs_three_ages() {
        let t = annual(2020, &[&[1.0, 2.0], &[1.0, 3.0], &[1.0]]);
        assert_eq!(
            Mack::default().fit(&t, "values"),
            Err(Error::TooFewAges {
                needed: 3,
                found: 2
            })
        );
    }

    #[test]
    fn rejects_sigma_that_cannot_be_filled() {
        // Three ages but only one estimable sigma: log-linear needs two.
        let t = annual(2020, &[&[1.0, 2.0, 3.0], &[1.0, 3.0], &[1.0]]);
        assert!(ChainLadder::default().fit(&t, "values").is_ok());
        assert!(matches!(
            Mack::default().fit(&t, "values"),
            Err(Error::Factor { age: 1, .. })
        ));
    }
}
