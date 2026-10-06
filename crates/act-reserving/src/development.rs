//! Development factors: age-to-age factors, their variance parameters and
//! standard errors, estimated as the weighted regressions of Mack (1993,
//! 1999) used by R ChainLadder and chainladder-python.

use crate::error::{Error, Result};
use crate::triangle::{Segment, Triangle};
use act_core::Lag;

/// How individual link ratios are averaged into one factor per age.
///
/// Each is the weighted regression `C[k+1] = f C[k]` with weights
/// `C[k]^(alpha - 2)`, so `f = sum(C[k]^(alpha-1) C[k+1]) / sum(C[k]^alpha)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Average {
    /// `alpha = 1`: `sum(C[k+1]) / sum(C[k])`, the chain-ladder factor.
    #[default]
    Volume,
    /// `alpha = 0`: the mean of the individual link ratios.
    Simple,
    /// `alpha = 2`: ordinary least squares through the origin.
    Regression,
}

impl Average {
    /// Mack's `alpha`.
    pub const fn alpha(self) -> f64 {
        match self {
            Self::Volume => 1.0,
            Self::Simple => 0.0,
            Self::Regression => 2.0,
        }
    }
}

/// How a variance parameter `sigma_k` that cannot be estimated (an age with
/// a single link ratio) is filled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SigmaInterpolation {
    /// Regress `ln(sigma_k)` on `k` over the estimated, positive sigmas and
    /// extrapolate. The default in R ChainLadder and chainladder-python.
    #[default]
    LogLinear,
    /// Mack (1993): `sigma_k^2 = min(sigma_{k-1}^4 / sigma_{k-2}^2,
    /// sigma_{k-2}^2, sigma_{k-1}^2)`.
    Mack,
}

/// Development-factor estimator.
///
/// ```
/// use act_reserving::Development;
/// # use act_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// # let origin = [2020, 2020, 2021].map(Month::january);
/// # let tri = Triangle::from_long(&Long {
/// #     keys: &[],
/// #     origin: &origin,
/// #     development: DevelopmentColumn::Age(&[12, 24, 12]),
/// #     values: &[("paid", &[100.0, 150.0, 110.0])],
/// #     origin_grain: Grain::Year,
/// #     development_grain: Grain::Year,
/// #     cumulative: true,
/// # })
/// # .unwrap();
/// let fit = Development::default().fit(&tri, "paid").unwrap();
/// assert_eq!(fit.ldf, vec![1.5]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Development {
    pub average: Average,
    pub sigma_interpolation: SigmaInterpolation,
}

/// Estimated development pattern of one triangle segment.
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopmentFit {
    /// Age of each development position; factor `k` links age `k` to `k + 1`.
    pub development: Vec<Lag>,
    /// Age-to-age factors `f_k`, one fewer than the ages.
    pub ldf: Vec<f64>,
    /// Variance parameters `sigma_k`, with unestimable ones interpolated.
    pub sigma: Vec<f64>,
    /// Standard error of each factor, `sigma_k / sqrt(volume_k)`.
    pub std_err: Vec<f64>,
    /// Total regression weight behind each factor, `sum(C[k]^alpha)` over
    /// the origins observed at both ages: with volume weighting, the sum of
    /// the earlier values.
    pub volume: Vec<f64>,
    /// Mack's `alpha` the factors were estimated with.
    pub alpha: f64,
}

impl Development {
    /// Estimates the development pattern of `column` in a single-segment
    /// triangle, from every origin observed at both ends of each link.
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<DevelopmentFit> {
        let segment = triangle.segment(column)?;
        let mut fit = self.fit_segment(&segment)?;
        fit.development = segment.ages.clone();
        Ok(fit)
    }

    pub(crate) fn fit_segment(&self, segment: &Segment) -> Result<DevelopmentFit> {
        let alpha = self.average.alpha();
        let n_links = segment.n_dev.saturating_sub(1);
        let mut ldf = Vec::with_capacity(n_links);
        let mut sigma = Vec::with_capacity(n_links);
        let mut volume = Vec::with_capacity(n_links);
        for age in 0..n_links {
            let pairs = segment.link_pairs(age);
            if pairs.is_empty() {
                return Err(Error::Factor {
                    age,
                    reason: "no origin is observed at both ages",
                });
            }
            let weight = |x: f64, power: f64| {
                let w = x.powf(power);
                if w.is_finite() {
                    Ok(w)
                } else {
                    Err(Error::Factor {
                        age,
                        reason: "a zero value gets an infinite weight",
                    })
                }
            };
            let mut num = 0.0;
            let mut den = 0.0;
            for &(x, y) in &pairs {
                num += weight(x, alpha - 1.0)? * y;
                den += weight(x, alpha)?;
            }
            if den == 0.0 {
                return Err(Error::Factor {
                    age,
                    reason: "earlier values sum to zero",
                });
            }
            let f = num / den;
            // An origin at zero says nothing about the variance given a
            // positive value, and its weight C^(alpha - 2) is infinite, so it
            // does not enter sigma.
            let informative: Vec<(f64, f64)> =
                pairs.iter().copied().filter(|&(x, _)| x != 0.0).collect();
            sigma.push(if informative.len() > 1 {
                let ss: f64 = informative
                    .iter()
                    .map(|&(x, y)| x.powf(alpha - 2.0) * (y - f * x).powi(2))
                    .sum();
                Some((ss / (informative.len() - 1) as f64).sqrt())
            } else {
                None
            });
            ldf.push(f);
            volume.push(den);
        }
        let sigma = interpolate_sigma(&sigma, self.sigma_interpolation);
        let std_err = sigma
            .iter()
            .zip(&volume)
            .map(|(s, v)| s / v.sqrt())
            .collect();
        Ok(DevelopmentFit {
            development: Vec::new(),
            ldf,
            sigma,
            std_err,
            volume,
            alpha,
        })
    }
}

/// Fills the sigmas that could not be estimated. Those that cannot be
/// filled either (too few estimated sigmas to extrapolate from) are NaN;
/// the chain ladder does not need them and Mack rejects them.
fn interpolate_sigma(sigma: &[Option<f64>], method: SigmaInterpolation) -> Vec<f64> {
    match method {
        SigmaInterpolation::LogLinear => {
            let points: Vec<(f64, f64)> = sigma
                .iter()
                .enumerate()
                .filter_map(|(k, s)| s.filter(|&s| s > 0.0).map(|s| (k as f64, s.ln())))
                .collect();
            let (intercept, slope) = if points.len() < 2 {
                (f64::NAN, f64::NAN)
            } else {
                let n = points.len() as f64;
                let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
                let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
                let sxy: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
                let sxx: f64 = points.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
                let slope = sxy / sxx;
                (mean_y - slope * mean_x, slope)
            };
            sigma
                .iter()
                .enumerate()
                .map(|(k, s)| s.unwrap_or_else(|| (intercept + slope * k as f64).exp()))
                .collect()
        }
        SigmaInterpolation::Mack => {
            let mut out: Vec<f64> = Vec::with_capacity(sigma.len());
            for &s in sigma {
                let filled = match (s, &out[..]) {
                    (Some(s), _) => s,
                    (None, [.., a, b]) => {
                        let (a2, b2) = (a * a, b * b);
                        let floor = a2.min(b2);
                        let s2 = if a2 == 0.0 {
                            floor
                        } else {
                            (b2 * b2 / a2).min(floor)
                        };
                        s2.sqrt()
                    }
                    (None, _) => f64::NAN,
                };
                out.push(filled);
            }
            out
        }
    }
}

/// Age-to-ultimate factors: element `k` is `tail` times the product of
/// `ldf[k..]`, so the last element (the oldest age) is `tail` itself.
pub fn cumulative_factors(ldf: &[f64], tail: f64) -> Vec<f64> {
    let mut out = vec![tail; ldf.len() + 1];
    for k in (0..ldf.len()).rev() {
        out[k] = out[k + 1] * ldf[k];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::{annual, raa};

    fn close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < tol, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn raa_volume_factors() {
        // R ChainLadder: ata(RAA) vwtd.
        let fit = Development::default().fit(&raa(), "values").unwrap();
        close(
            &fit.ldf,
            &[
                2.999359, 1.623523, 1.270888, 1.171675, 1.113385, 1.041935, 1.033264, 1.016936,
                1.009217,
            ],
            1e-6,
        );
        assert_eq!(fit.development[0], 12);
    }

    #[test]
    fn raa_simple_factors() {
        // R ChainLadder: ata(RAA) smpl.
        let dev = Development {
            average: Average::Simple,
            ..Default::default()
        };
        close(
            &dev.fit(&raa(), "values").unwrap().ldf,
            &[
                8.206099, 1.695894, 1.314510, 1.182926, 1.126962, 1.043328, 1.034355, 1.017995,
                1.009217,
            ],
            1e-6,
        );
    }

    #[test]
    fn raa_sigma_both_interpolations() {
        // R ChainLadder: MackChainLadder(RAA, est.sigma = ...)$sigma.
        let head = [
            166.9834704,
            33.2945384,
            26.2952997,
            7.8249598,
            10.9288176,
            6.3890424,
            1.1590623,
            2.8077043,
        ];
        let log_linear = Development::default().fit(&raa(), "values").unwrap();
        close(&log_linear.sigma[..8], &head, 1e-6);
        close(&log_linear.sigma[8..], &[0.8033494], 1e-6);
        let mack = Development {
            sigma_interpolation: SigmaInterpolation::Mack,
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        close(&mack.sigma[8..], &[1.159062], 1e-6);
    }

    #[test]
    fn mack_rule_cases() {
        let fill = |s: &[Option<f64>]| interpolate_sigma(s, SigmaInterpolation::Mack);
        close(
            &fill(&[Some(2.0), Some(1.0), None]),
            &[2.0, 1.0, 0.5],
            1e-12,
        );
        close(
            &fill(&[Some(1.0), Some(2.0), None]),
            &[1.0, 2.0, 1.0],
            1e-12,
        );
        close(
            &fill(&[Some(0.0), Some(2.0), None]),
            &[0.0, 2.0, 0.0],
            1e-12,
        );
        assert!(interpolate_sigma(&[Some(1.0), None], SigmaInterpolation::Mack)[1].is_nan());
        assert!(interpolate_sigma(&[Some(1.0), None], SigmaInterpolation::LogLinear)[1].is_nan());
    }

    #[test]
    fn simple_average_rejects_zero_origin_value() {
        let t = annual(2020, &[&[0.0, 5.0], &[4.0]]);
        let dev = Development {
            average: Average::Simple,
            ..Default::default()
        };
        assert!(matches!(
            dev.fit(&t, "values"),
            Err(Error::Factor { age: 0, .. })
        ));
        // Volume weighting keeps the zero origin's later value in the factor
        // (the regression formula) but leaves it out of sigma. R ChainLadder
        // fails on this input, so there is no reference value.
        let t = annual(2020, &[&[0.0, 5.0], &[4.0, 6.0], &[2.0, 4.0], &[1.0]]);
        let f = Development::default().fit(&t, "values").unwrap();
        close(&f.ldf, &[15.0 / 6.0], 1e-12);
        assert!(f.sigma[0].is_finite());
    }

    #[test]
    fn cumulative_factors_chain_to_tail() {
        close(
            &cumulative_factors(&[2.0, 1.5], 1.1),
            &[3.3, 1.65, 1.1],
            1e-12,
        );
        assert_eq!(cumulative_factors(&[], 1.05), vec![1.05]);
    }
}
