//! A Bayesian Bornhuetter–Ferguson prior for an excess layer, from the
//! layer share and its uncertainty (`docs/design/reserving-excess.md`,
//! section 3, PR 4).
//!
//! A thin adapter over [`BayesBfNormalised`]: the layer's a-priori ultimate
//! for origin `i` is `E_i * ELR * s_i`, the ground-up exposure times the
//! ground-up expected ultimate loss ratio times the layer share. The share
//! `s` and the loss ratio carry uncertainty of their own, so the gamma
//! prior of the normalised BF, on the layer's expected ultimate, is centred
//! on that product with the coefficient of variation of a product of two
//! independent variables,
//!
//! ```text
//! cv^2 = (1 + cv_elr^2) (1 + cv_share^2) - 1,
//! ```
//!
//! whose first two moments are exact (for independent `X` and `Y`,
//! `Var(XY) = E(X)^2 E(Y)^2 ((1 + c_x^2)(1 + c_y^2) - 1)`). The gamma prior
//! is the moment match of that product, not its exact law. The model fits
//! the layer's triangle, so the layer pattern is estimated from the layer's
//! own data under the normalised BF's vague prior on the pattern; the
//! derived excess pattern does not enter (open decision 22). The prior is
//! independent across origins, though one share error is common to all of
//! them (open decision 23).

use prospicio_prob::Severity;

use super::{BayesBfNormalised, BayesBfNormalisedFit, BayesOneYear, BayesOneYearFit};
use crate::error::{Error, Result};
use crate::excess::{Layer, layer_exposure, layer_share};
use crate::triangle::Triangle;

/// The name of the layer exposure column [`LayerSharePrior::triangle`] adds.
pub const LAYER_EXPOSURE_COLUMN: &str = "layer_exposure";

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
        })
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

    /// The coefficient of variation of the gamma prior on the layer's
    /// expected ultimate: that of the product of the loss ratio and the
    /// share, `sqrt((1 + c_elr^2)(1 + c_share^2) - 1)`; infinite if either
    /// is.
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

    /// `base` with this prior's ground-up `apriori` and its [`cv`](Self::cv):
    /// use it with the triangle of [`triangle`](Self::triangle), whose layer
    /// exposure is the ground-up exposure times the share.
    pub fn model(&self, base: &BayesBfNormalised) -> BayesBfNormalised {
        BayesBfNormalised {
            apriori: self.apriori,
            cv: self.cv(),
            ..base.clone()
        }
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
        self.model(base).fit(
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
        self.model(base).one_year(
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

    #[test]
    fn bad_inputs_are_errors() {
        assert!(LayerSharePrior::new(0.0, 0.1, &[0.2], 0.1).is_err());
        assert!(LayerSharePrior::new(0.5, -0.1, &[0.2], 0.1).is_err());
        assert!(LayerSharePrior::new(0.5, 0.1, &[1.2], 0.1).is_err());
        assert!(LayerSharePrior::new(0.5, 0.1, &[], 0.1).is_err());
        assert!(LayerSharePrior::from_share_se(0.5, 0.1, 0.2, f64::NAN).is_err());
    }
}
