//! A Bayesian Bornhuetter–Ferguson prior for an excess layer, from the
//! layer share and its uncertainty (`docs/design/reserving-excess.md`,
//! section 3, PR 4).
//!
//! A thin adapter over [`BayesBfNormalised`]: the layer's a-priori ultimate
//! for origin `i` is `E_i * ELR * s_i`, the ground-up exposure times the
//! ground-up expected ultimate loss ratio times the layer share. The share
//! `s` and the loss ratio carry uncertainty of their own, so the prior of
//! the normalised BF on the layer's expected ultimate is centred on that
//! product with the coefficient of variation of a product of two independent
//! variables,
//!
//! ```text
//! cv^2 = (1 + cv_elr^2) (1 + cv_share^2) - 1,
//! ```
//!
//! whose first two moments are exact (for independent `X` and `Y`,
//! `Var(XY) = E(X)^2 E(Y)^2 ((1 + c_x^2)(1 + c_y^2) - 1)`).
//!
//! One share error is common to every origin: if the share is wrong, it is
//! wrong for all of them ([`ShareDependence::Common`], the default,
//! `reserving-excess.md` decision 23(a)). The model then has one more
//! parameter, a multiplier `s` with mean 1 and cv `cv_share` (a gamma),
//! sampled with the others, and each origin's prior ultimate is a gamma with
//! mean `m_i s` and cv `cv_elr`: every origin keeps its marginal mean and the
//! cv above, the origins' priors are correlated through `s`, and the data
//! update the share for every origin at once. With
//! [`ShareDependence::PerOrigin`] the share error is an independent gamma
//! with the cv above at each origin, which understates the total's interval
//! (the PR 4 behaviour, kept for comparison). The model fits the layer's
//! triangle, so the layer pattern is estimated from the layer's own data
//! under the normalised BF's vague prior on the pattern; the derived excess
//! pattern does not enter (open decision 22).

use prospicio_prob::Severity;

use super::{BayesBfNormalised, BayesBfNormalisedFit, BayesOneYear, BayesOneYearFit};
use crate::error::{Error, Result};
use crate::excess::{Layer, layer_exposure, layer_share};
use crate::triangle::Triangle;

/// The name of the layer exposure column [`LayerSharePrior::triangle`] adds.
pub const LAYER_EXPOSURE_COLUMN: &str = "layer_exposure";

/// How the share's error enters the origins' priors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShareDependence {
    /// One multiplier common to all origins (the default): a wrong share is
    /// wrong for every origin, the origins' priors are positively
    /// correlated, and the total's prior is wider than under
    /// [`PerOrigin`](Self::PerOrigin). Needs a positive, finite
    /// `apriori_cv` when `share_cv` is positive and finite; a flat prior on
    /// either has nothing to share and is the per-origin model.
    #[default]
    Common,
    /// An independent share error at each origin, folded into that origin's
    /// gamma prior cv (the first version, PR 4).
    PerOrigin,
}

/// The prior of a layer's expected ultimate from the ground-up expected
/// loss ratio and the layer share, each with a coefficient of variation.
///
/// ```
/// use prospicio_reserving::bayes::LayerSharePrior;
///
/// // Loss ratio 0.65 (cv 0.1) and a share of 0.2 (cv 0.3): the layer's
/// // loss ratio is 0.13, and the cv is that of a product of independents.
/// let prior = LayerSharePrior::new(0.65, 0.1, &[0.2], 0.3)?;
/// assert!((prior.layer_apriori(0) - 0.13).abs() < 1e-12);
/// assert!((prior.cv().powi(2) - (1.01 * 1.09 - 1.0)).abs() < 1e-12);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSharePrior {
    /// The ground-up expected ultimate loss ratio.
    pub apriori: f64,
    /// Its coefficient of variation (0 for a loss ratio held as known).
    pub apriori_cv: f64,
    /// The layer share of each origin, or one share for all of them.
    pub shares: Vec<f64>,
    /// The coefficient of variation of the share, the same at every origin
    /// (0 for a share held as known).
    pub share_cv: f64,
    /// Whether the share error is common to all origins or independent at
    /// each ([`ShareDependence::Common`] unless set otherwise).
    pub dependence: ShareDependence,
}

impl LayerSharePrior {
    /// A prior from the ground-up loss ratio and its cv, the layer share
    /// (one, or one per origin, each in `(0, 1]`) and its cv. Both cvs 0
    /// give a prior held as known, which the normalised BF cannot fit (its
    /// `cv` must be positive): use a small cv for a tight prior.
    pub fn new(apriori: f64, apriori_cv: f64, shares: &[f64], share_cv: f64) -> Result<Self> {
        if !(apriori.is_finite() && apriori > 0.0) {
            return Err(Error::InvalidSetting {
                name: "apriori",
                value: apriori,
                expected: "a finite, positive loss ratio",
            });
        }
        for (name, cv) in [("apriori_cv", apriori_cv), ("share_cv", share_cv)] {
            if cv.is_nan() || cv < 0.0 {
                return Err(Error::InvalidSetting {
                    name,
                    value: cv,
                    expected: "a non-negative coefficient of variation (infinity for flat)",
                });
            }
        }
        if shares.is_empty() {
            return Err(Error::LengthMismatch {
                column: "shares".into(),
                expected: 1,
                found: 0,
            });
        }
        if let Some(&s) = shares
            .iter()
            .find(|s| !(s.is_finite() && **s > 0.0 && **s <= 1.0))
        {
            return Err(Error::InvalidSetting {
                name: "share",
                value: s,
                expected: "a share in (0, 1]",
            });
        }
        Ok(Self {
            apriori,
            apriori_cv,
            shares: shares.to_vec(),
            share_cv,
            dependence: ShareDependence::default(),
        })
    }

    /// This prior with the share error `dependence`.
    pub fn with_dependence(mut self, dependence: ShareDependence) -> Self {
        self.dependence = dependence;
        self
    }

    /// A prior with one share given by its standard error: the share's cv
    /// is `share_se / share`.
    pub fn from_share_se(apriori: f64, apriori_cv: f64, share: f64, share_se: f64) -> Result<Self> {
        if !(share_se.is_finite() && share_se >= 0.0) {
            return Err(Error::InvalidSetting {
                name: "share_se",
                value: share_se,
                expected: "a finite, non-negative standard error",
            });
        }
        Self::new(apriori, apriori_cv, &[share], share_se / share)
    }

    /// A prior with the share of a `severity` in `layer` (its
    /// [`layer_share`], ultimate basis) and its cv.
    pub fn from_severity(
        apriori: f64,
        apriori_cv: f64,
        severity: &dyn Severity,
        layer: Layer,
        share_cv: f64,
    ) -> Result<Self> {
        Self::new(
            apriori,
            apriori_cv,
            &[layer_share(severity, layer)?],
            share_cv,
        )
    }

    /// The marginal coefficient of variation of the prior on each origin's
    /// expected ultimate: that of the product of the loss ratio and the
    /// share, `sqrt((1 + c_elr^2)(1 + c_share^2) - 1)`; infinite if either
    /// is. It is the gamma's cv under
    /// [`PerOrigin`](ShareDependence::PerOrigin), and the marginal cv under
    /// [`Common`](ShareDependence::Common), where the model's own cv is
    /// `apriori_cv`.
    pub fn cv(&self) -> f64 {
        if self.apriori_cv.is_infinite() || self.share_cv.is_infinite() {
            return f64::INFINITY;
        }
        // Expanded so that small cvs keep their precision.
        let (a, b) = (self.apriori_cv.powi(2), self.share_cv.powi(2));
        (a + b + a * b).sqrt()
    }

    /// The layer's a-priori loss ratio at origin `i`: the ground-up loss
    /// ratio times that origin's share.
    pub fn layer_apriori(&self, i: usize) -> f64 {
        let s = if self.shares.len() == 1 {
            self.shares[0]
        } else {
            self.shares[i]
        };
        self.apriori * s
    }

    /// Whether the model carries a common multiplier: the share error is
    /// [`Common`](ShareDependence::Common) and both it and the loss ratio's
    /// are positive and finite. (A known share, `share_cv` 0, has no error
    /// to share, and a flat prior on either has nothing to share it with:
    /// each is the per-origin model.)
    pub fn has_common_factor(&self) -> bool {
        self.dependence == ShareDependence::Common
            && self.share_cv > 0.0
            && self.share_cv.is_finite()
            && self.apriori_cv.is_finite()
    }

    /// `base` with this prior's ground-up `apriori` and its cv: use it with
    /// the triangle of [`triangle`](Self::triangle), whose layer exposure is
    /// the ground-up exposure times the share. Under a common share error
    /// the model's `cv` is `apriori_cv` and its `common_cv` the share's;
    /// otherwise its `cv` is [`cv`](Self::cv) and it has no common
    /// multiplier (`base`'s `common_cv` is replaced either way).
    ///
    /// # Errors
    ///
    /// A common share error needs the loss ratio's cv positive: with
    /// `apriori_cv` 0 every origin's ultimate would be the same multiple
    /// of its prior, which the model cannot represent. Use a small cv, or
    /// [`ShareDependence::PerOrigin`].
    pub fn model(&self, base: &BayesBfNormalised) -> Result<BayesBfNormalised> {
        if self.has_common_factor() {
            if self.apriori_cv <= 0.0 {
                return Err(Error::InvalidSetting {
                    name: "apriori_cv",
                    value: self.apriori_cv,
                    expected: "positive when the share error is common to all origins \
                               (use a small cv, or ShareDependence::PerOrigin)",
                });
            }
            return Ok(BayesBfNormalised {
                apriori: self.apriori,
                cv: self.apriori_cv,
                common_cv: Some(self.share_cv),
                ..base.clone()
            });
        }
        Ok(BayesBfNormalised {
            apriori: self.apriori,
            cv: self.cv(),
            common_cv: None,
            ..base.clone()
        })
    }

    /// The mean and standard deviation of the prior of the layer's total
    /// expected ultimate over origins with ground-up `exposure`: with
    /// `m_i = apriori * exposure_i * share_i`, the mean is `sum m_i` and the
    /// variance, in closed form,
    ///
    /// ```text
    /// Common:    c_e^2 (1 + c_s^2) sum m_i^2 + c_s^2 (sum m_i)^2
    /// PerOrigin: ((1 + c_e^2)(1 + c_s^2) - 1) sum m_i^2
    /// ```
    ///
    /// (`c_e` the loss ratio's cv, `c_s` the share's). The two have the
    /// same variance at each origin; the common share error adds the
    /// covariance `c_s^2 m_i m_j` between origins. A flat prior has an
    /// infinite standard deviation.
    pub fn prior_total(&self, exposure: &[f64]) -> Result<(f64, f64)> {
        if !(self.shares.len() == 1 || self.shares.len() == exposure.len()) {
            return Err(Error::LengthMismatch {
                column: "exposure".into(),
                expected: self.shares.len(),
                found: exposure.len(),
            });
        }
        let m: Vec<f64> = exposure
            .iter()
            .enumerate()
            .map(|(i, e)| e * self.layer_apriori(i))
            .collect();
        let (sum, sum_sq) = (m.iter().sum::<f64>(), m.iter().map(|x| x * x).sum::<f64>());
        if self.apriori_cv.is_infinite() || self.share_cv.is_infinite() {
            return Ok((sum, f64::INFINITY));
        }
        let (e2, s2) = (self.apriori_cv.powi(2), self.share_cv.powi(2));
        let var = match self.dependence {
            ShareDependence::Common => e2 * (1.0 + s2) * sum_sq + s2 * sum * sum,
            ShareDependence::PerOrigin => (e2 + s2 + e2 * s2) * sum_sq,
        };
        Ok((sum, var.sqrt()))
    }

    /// `triangle` with the layer exposure column [`LAYER_EXPOSURE_COLUMN`]:
    /// the ground-up `exposure` column times the share ([`layer_exposure`]).
    pub fn triangle(&self, triangle: &Triangle, exposure: &str) -> Result<Triangle> {
        layer_exposure(triangle, exposure, &self.shares, LAYER_EXPOSURE_COLUMN)
    }

    /// Fits the layer losses `column` of `triangle`, which holds the
    /// ground-up `exposure` column too, with this prior.
    pub fn fit(
        &self,
        base: &BayesBfNormalised,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
    ) -> Result<BayesBfNormalisedFit> {
        self.model(base)?.fit(
            &self.triangle(triangle, exposure)?,
            column,
            LAYER_EXPOSURE_COLUMN,
        )
    }

    /// The one-year view of the same fit
    /// ([`BayesBfNormalised::one_year`]).
    pub fn one_year(
        &self,
        base: &BayesBfNormalised,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
        settings: &BayesOneYear,
    ) -> Result<BayesOneYearFit<BayesBfNormalisedFit>> {
        self.model(base)?.one_year(
            &self.triangle(triangle, exposure)?,
            column,
            LAYER_EXPOSURE_COLUMN,
            settings,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cv_is_that_of_a_product_of_independents() {
        // Exact moments of X Y, X and Y independent, from raw moments:
        // E(X^2) = m^2 (1 + c^2).
        let (mx, cx, my, cy) = (0.65_f64, 0.1_f64, 0.2_f64, 0.3_f64);
        let ex2 = mx * mx * (1.0 + cx * cx);
        let ey2 = my * my * (1.0 + cy * cy);
        let var = ex2 * ey2 - (mx * my).powi(2);
        let want = var.sqrt() / (mx * my);
        let prior = LayerSharePrior::new(mx, cx, &[my], cy).unwrap();
        assert!((prior.cv() - want).abs() < 1e-14, "{} {want}", prior.cv());
        // One factor known: the other's cv alone; flat is flat.
        let known = LayerSharePrior::new(0.5, 0.0, &[0.2], 0.3).unwrap();
        assert!((known.cv() - 0.3).abs() < 1e-15);
        let known = LayerSharePrior::new(0.5, 0.2, &[0.2], 0.0).unwrap();
        assert!((known.cv() - 0.2).abs() < 1e-15);
        let flat = LayerSharePrior::new(0.5, f64::INFINITY, &[0.2], 0.3).unwrap();
        assert!(flat.cv().is_infinite());
    }

    #[test]
    fn the_standard_error_and_the_severity_routes_agree() {
        let a = LayerSharePrior::from_share_se(0.6, 0.1, 0.25, 0.05).unwrap();
        let b = LayerSharePrior::new(0.6, 0.1, &[0.25], 0.2).unwrap();
        assert!((a.cv() - b.cv()).abs() < 1e-15);
        let sev = prospicio_prob::Pareto::new(1.0, 2.0).unwrap();
        let c = LayerSharePrior::from_severity(0.6, 0.1, &sev, Layer::unlimited(4.0).unwrap(), 0.2)
            .unwrap();
        assert!((c.shares[0] - 0.125).abs() < 1e-12);
        assert!((c.layer_apriori(7) - 0.075).abs() < 1e-12);
    }

    /// The generative law: `s ~ Gamma(mean 1, cv c_s)`, then `U_i | s ~
    /// Gamma(mean m_i s, cv c_e)` (common), or `U_i ~ Gamma(mean m_i, cv
    /// the product's)` independently. The total's closed-form variance
    /// against a large simulation, and the covariance between origins.
    #[test]
    fn the_total_prior_moments_are_exact() {
        use crate::bootstrap::{ProcessDistribution, process_draw};
        use prospicio_core::StreamRng;
        let exposure = [100.0, 150.0, 220.0, 90.0];
        let (elr, ce, shares, cs) = (0.6, 0.25, [0.2, 0.25, 0.3, 0.2], 0.7);
        let prior = LayerSharePrior::new(elr, ce, &shares, cs).unwrap();
        let m: Vec<f64> = (0..4)
            .map(|i| exposure[i] * prior.layer_apriori(i))
            .collect();
        // Closed forms from first principles, not through prior_total.
        let (sum, sum_sq) = (m.iter().sum::<f64>(), m.iter().map(|x| x * x).sum::<f64>());
        let common_var = ce * ce * (1.0 + cs * cs) * sum_sq + cs * cs * sum * sum;
        let per_var = ((1.0 + ce * ce) * (1.0 + cs * cs) - 1.0) * sum_sq;
        let (mean, sd) = prior.prior_total(&exposure).unwrap();
        assert!((mean - sum).abs() < 1e-12 && (sd - common_var.sqrt()).abs() < 1e-9);
        let per = prior.clone().with_dependence(ShareDependence::PerOrigin);
        assert!((per.prior_total(&exposure).unwrap().1 - per_var.sqrt()).abs() < 1e-9);
        assert!(sd > per.prior_total(&exposure).unwrap().1);
        // Simulation of the generative law (gamma with mean mu and cv c has
        // shape 1 / c^2 and scale mu c^2).
        let gamma = |mean: f64, cv: f64, rng: &mut StreamRng| {
            process_draw(ProcessDistribution::Gamma, mean * cv * cv, mean, rng)
        };
        let n = 400_000;
        let mut rng = StreamRng::new(11, 0);
        let (mut a, mut a2, mut u0, mut u1, mut u01) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for _ in 0..n {
            let s = gamma(1.0, cs, &mut rng);
            let u: Vec<f64> = m.iter().map(|mi| gamma(mi * s, ce, &mut rng)).collect();
            let total: f64 = u.iter().sum();
            a += total;
            a2 += total * total;
            u0 += u[0];
            u1 += u[1];
            u01 += u[0] * u[1];
        }
        let nf = n as f64;
        let (mean_hat, var_hat) = (a / nf, a2 / nf - (a / nf).powi(2));
        assert!((mean_hat / sum - 1.0).abs() < 0.005, "{mean_hat} vs {sum}");
        assert!(
            (var_hat / common_var - 1.0).abs() < 0.03,
            "{var_hat} vs {common_var}"
        );
        // Cov(U_0, U_1) = cs^2 m_0 m_1 under a common share.
        let cov_hat = u01 / nf - (u0 / nf) * (u1 / nf);
        let cov = cs * cs * m[0] * m[1];
        assert!((cov_hat / cov - 1.0).abs() < 0.05, "{cov_hat} vs {cov}");
    }

    #[test]
    fn the_model_follows_the_dependence() {
        let base = BayesBfNormalised::default();
        let prior = LayerSharePrior::new(0.5, 0.2, &[0.25], 0.3).unwrap();
        let common = prior.model(&base).unwrap();
        assert_eq!((common.cv, common.common_cv), (0.2, Some(0.3)));
        let per = prior
            .clone()
            .with_dependence(ShareDependence::PerOrigin)
            .model(&base)
            .unwrap();
        assert_eq!(per.common_cv, None);
        assert!((per.cv - prior.cv()).abs() < 1e-15);
        // A known, flat or absent share error has nothing to share.
        for (ce, cs) in [(0.2, 0.0), (0.2, f64::INFINITY), (f64::INFINITY, 0.3)] {
            let p = LayerSharePrior::new(0.5, ce, &[0.25], cs).unwrap();
            let m = p.model(&base).unwrap();
            assert_eq!(m.common_cv, None);
            assert!(!p.has_common_factor());
        }
        // A known loss ratio cannot be common-shared.
        let p = LayerSharePrior::new(0.5, 0.0, &[0.25], 0.3).unwrap();
        assert!(p.model(&base).is_err());
        assert!(
            p.with_dependence(ShareDependence::PerOrigin)
                .model(&base)
                .is_ok()
        );
    }

    #[test]
    fn bad_inputs_are_errors() {
        assert!(LayerSharePrior::new(0.0, 0.1, &[0.2], 0.1).is_err());
        assert!(LayerSharePrior::new(0.5, -0.1, &[0.2], 0.1).is_err());
        assert!(LayerSharePrior::new(0.5, 0.1, &[1.2], 0.1).is_err());
        assert!(LayerSharePrior::new(0.5, 0.1, &[], 0.1).is_err());
        assert!(LayerSharePrior::from_share_se(0.5, 0.1, 0.2, f64::NAN).is_err());
    }
}
