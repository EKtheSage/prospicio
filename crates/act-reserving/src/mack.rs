//! Mack's distribution-free chain ladder: the chain-ladder projection plus
//! the mean squared error of each origin's and the total reserve.

use crate::chain_ladder::{ChainLadder, ChainLadderFit};
use crate::development::Development;
use crate::error::{Error, Result};
use crate::segments::{SegmentFits, fit_each};
use crate::tail::Tail;
use crate::triangle::{Segment, Triangle};
use act_core::Lag;

/// Mack's chain ladder (Mack 1993, 1999), with the process and parameter
/// risk recursions of R ChainLadder's `MackChainLadder`.
///
/// The factors use the development estimator's `alpha`: the conditional
/// variance of `C[k+1]` given `C[k]` is `sigma_k^2 C[k]^(2 - alpha)`.
///
/// A tail other than 1 is one more development step, from the oldest age
/// to ultimate, with its own sigma and standard error, as R's
/// `MackChainLadder(tail = ...)` and chainladder-python's `MackChainladder`
/// on a tailed pattern do. Unless given, both are extrapolated log-linearly
/// (see [`TailFit`](crate::TailFit)). Every origin, the oldest included,
/// carries the tail's risk. A tail below 1 follows chainladder-python: it
/// scales the ultimates and carries the risk read where a tail of 1.001
/// would be. R's `MackChainLadder` ignores a tail below 1 altogether.
///
/// ```
/// use act_reserving::{Mack, Tail};
/// # use act_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// # let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
/// # let tri = Triangle::from_long(&Long {
/// #     keys: &[],
/// #     origin: &origin,
/// #     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
/// #     values: &[("paid", &[100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0])],
/// #     origin_grain: Grain::Year,
/// #     development_grain: Grain::Year,
/// #     cumulative: true,
/// # })
/// # .unwrap();
/// let mack = Mack { tail: Tail::LogLinear, ..Default::default() }.fit(&tri, "paid").unwrap();
/// assert!(mack.chain_ladder.tail.factor > 1.0);
/// assert!(mack.standard_error[0] > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Mack {
    /// How factors are estimated and unestimable sigmas filled in.
    pub development: Development,
    /// Development past the oldest age; the default is no tail.
    pub tail: Tail,
    /// The tail's sigma, R's `tail.sigma`; `None` extrapolates it. Unused
    /// when the tail factor is 1.
    pub tail_sigma: Option<f64>,
    /// The tail factor's standard error, R's `tail.se`; `None` extrapolates
    /// it. Unused when the tail factor is 1.
    pub tail_std_err: Option<f64>,
}

/// A fitted Mack model. Risks are standard errors (square roots of the
/// variance components) of each origin's ultimate, which equal those of its
/// reserve since the latest value is known. For the one-year view of the
/// same risk, see [`MackFit::claims_development_result`].
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
        let segment = triangle.segment(column)?;
        self.fit_segment(&segment, &segment.ages)
    }

    /// Fits `column` in every segment of `triangle`, each on its own; see
    /// [`SegmentFits`] for the long tables. A failure names its segment.
    pub fn fit_segments(&self, triangle: &Triangle, column: &str) -> Result<SegmentFits<MackFit>> {
        fit_each(triangle, column, |s| self.fit_segment(s, &s.ages))
    }

    pub(crate) fn fit_segment(&self, segment: &Segment, ages: &[Lag]) -> Result<MackFit> {
        if segment.n_dev < 3 {
            return Err(Error::TooFewAges {
                needed: 3,
                found: segment.n_dev,
            });
        }
        let mut chain_ladder = ChainLadder {
            development: self.development,
            tail: self.tail,
        }
        .fit_segment(segment, ages)?;

        let dev = &chain_ladder.development;
        if let Some(age) = dev.sigma.iter().position(|s| s.is_nan()) {
            return Err(Error::Factor {
                age,
                reason: "sigma can neither be estimated nor interpolated",
            });
        }
        let tail = &mut chain_ladder.tail;
        if tail.factor != 1.0 {
            for (given, fitted) in [
                (self.tail_sigma, &mut tail.sigma),
                (self.tail_std_err, &mut tail.std_err),
            ] {
                if let Some(v) = given {
                    if !v.is_finite() || v < 0.0 {
                        return Err(Error::Tail(
                            "a given tail sigma or standard error must be finite and non-negative",
                        ));
                    }
                    *fitted = v;
                }
            }
            if !tail.sigma.is_finite() || !tail.std_err.is_finite() {
                return Err(Error::Tail(
                    "the tail's sigma or standard error cannot be extrapolated; give them",
                ));
            }
        }
        let tail = &chain_ladder.tail;
        let dev = &chain_ladder.development;
        let alpha = dev.alpha;
        // The selected factors, then the tail as one more step to ultimate.
        let ldf: Vec<f64> = chain_ladder
            .ldf()
            .iter()
            .copied()
            .chain([tail.factor])
            .collect();
        let sigma: Vec<f64> = dev.sigma.iter().copied().chain([tail.sigma]).collect();
        let std_err: Vec<f64> = dev.std_err.iter().copied().chain([tail.std_err]).collect();
        let n_origins = chain_ladder.origins.len();

        // Per-origin recursions over the projected ages, and the total
        // parameter variance, which chains the sum of the projected values of
        // every origin still developing at each age. With no tail (factor 1,
        // sigma and standard error 0) the last step changes nothing.
        let mut process_var = vec![0.0; n_origins];
        let mut parameter_var = vec![0.0; n_origins];
        let mut total_parameter_var = 0.0;
        let projections: Vec<Vec<f64>> =
            (0..n_origins).map(|o| chain_ladder.projection(o)).collect();
        for k in 0..ldf.len() {
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
            ..Default::default()
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
    fn raa_with_r_log_linear_tail() {
        // R ChainLadder: MackChainLadder(RAA, tail = TRUE): Total.Mack.S.E
        // and the oldest origin's Mack.S.E (validation/reference/
        // reserving_tails_r.csv).
        let mack = Mack {
            tail: Tail::LogLinear,
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        close(mack.standard_error[0], 149.205_919_099_168, 1e-6);
        close(mack.chain_ladder.ultimate[0], 19_011.712_945_280_9, 1e-6);
    }

    #[test]
    fn given_tail_sigma_and_std_err() {
        // R ChainLadder: MackChainLadder(RAA, tail = 1.05, tail.se = ...,
        // tail.sigma = ...) reports the given values back.
        let mack = Mack {
            tail: 1.05.into(),
            tail_sigma: Some(1.5),
            tail_std_err: Some(0.003),
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        assert_eq!(
            (mack.chain_ladder.tail.sigma, mack.chain_ladder.tail.std_err),
            (1.5, 0.003)
        );
        // The oldest origin's process variance is the tail's alone.
        let latest = mack.chain_ladder.latest[0];
        close(mack.process_risk[0], 1.5 * latest.sqrt(), 1e-9);
        close(mack.parameter_risk[0], 0.003 * latest, 1e-9);
        let bad = Mack {
            tail: 1.05.into(),
            tail_sigma: Some(f64::NAN),
            ..Default::default()
        };
        assert!(matches!(bad.fit(&raa(), "values"), Err(Error::Tail(_))));
    }

    #[test]
    fn tail_below_one_follows_chainladder_python() {
        // chainladder-python 0.10.1: MackChainladder on TailConstant(0.98)
        // of RAA gives a total ultimate of 208859.783696 and a total
        // standard error of 26343.488605. R's MackChainLadder(RAA,
        // tail = 0.98) ignores the tail (213122.2 and 26880.74).
        let mack = Mack {
            tail: 0.98.into(),
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        close(mack.chain_ladder.total_ultimate(), 208_859.783_696, 1e-5);
        close(mack.total_standard_error, 26_343.488_605, 1e-5);
        assert!(mack.standard_error[0] > 0.0);
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
