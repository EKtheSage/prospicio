//! A negative binomial GLM with its `θ` estimated with the coefficients, by
//! maximum likelihood, as R's `MASS::glm.nb`.
//!
//! The GLM's variance is `μ + μ²/θ`. For a fixed `θ` IRLS gives the
//! coefficients; for fixed means the likelihood's score in `θ` is solved by
//! Newton's method (`MASS::theta.ml`). The two alternate, each GLM fit
//! warm-started from the last, until `θ` and the log-likelihood settle: the
//! joint maximum likelihood estimate. The standard error of `θ` is
//! `MASS`'s, from the observed information at the final means, which treats
//! the coefficients as known.

use prospicio_core::{Error, Result};
use prospicio_math::special::{digamma, trigamma};
use prospicio_models::{Design, Family, Fitted, Link, Model};
use prospicio_prob::PredictiveDistribution;

use crate::{Glm, GlmFit};

/// A negative binomial GLM whose `θ` is estimated.
///
/// ```
/// use prospicio_glm::negbin::NegativeBinomialGlm;
/// use prospicio_models::{Design, Fitted, Link, Model};
///
/// let x: Vec<f64> = (0..60).map(|i| f64::from(i % 3)).collect();
/// let y: Vec<f64> = (0..60).map(|i| f64::from((i * 7) % 11)).collect();
/// let d = Design::new(vec!["(Intercept)".into(), "x".into()], vec![vec![1.0; 60], x]).unwrap();
/// let fit = NegativeBinomialGlm::new(Link::Log).fit(&d, &y).unwrap();
/// assert!(fit.theta() > 0.0 && fit.se_theta() > 0.0);
/// assert_eq!(fit.predict(&d).unwrap().len(), 60);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomialGlm {
    pub link: Link,
    /// Relative change in `θ` (and absolute change in the log-likelihood)
    /// below which the alternation stops.
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl NegativeBinomialGlm {
    /// Tolerance `1e-10`, at most 50 alternations.
    pub fn new(link: Link) -> Self {
        Self {
            link,
            tolerance: 1e-10,
            max_iterations: 50,
        }
    }
}

impl Model for NegativeBinomialGlm {
    type Fitted = NegativeBinomialFit;

    /// Starts from the Poisson GLM's means and `θ` by the method of
    /// moments, then alternates as in the [module](self). Fails if the
    /// response is not a count or `θ` does not settle (an overdispersion
    /// too small to estimate drives it to infinity: use the Poisson GLM).
    fn fit(&self, design: &Design, y: &[f64]) -> Result<NegativeBinomialFit> {
        let w = design.weights();
        let mut fit = Glm::new(Family::Poisson, self.link).fit(design, y)?;
        let mut theta = theta_ml(y, fit.fitted(), w)?;
        let mut ll = f64::NEG_INFINITY;
        for iteration in 1..=self.max_iterations {
            fit = Glm::new(Family::NegativeBinomial { theta }, self.link).fit_from(
                design,
                y,
                Some(fit.fitted()),
            )?;
            let next = theta_ml(y, fit.fitted(), w)?;
            let next_ll = fit.log_likelihood();
            let settled = ((next - theta) / theta).abs() < self.tolerance
                && (next_ll - ll).abs() < self.tolerance * next_ll.abs().max(1.0);
            theta = next;
            ll = next_ll;
            if settled {
                let fit = Glm::new(Family::NegativeBinomial { theta }, self.link).fit_from(
                    design,
                    y,
                    Some(fit.fitted()),
                )?;
                let se_theta = 1.0 / information(y, fit.fitted(), w, theta).sqrt();
                return Ok(NegativeBinomialFit {
                    theta,
                    se_theta,
                    log_likelihood: fit.log_likelihood(),
                    iterations: iteration,
                    glm: fit,
                });
            }
        }
        Err(Error::Data(format!(
            "theta did not settle in {} alternations (last {theta}); the counts may not be \
             overdispersed",
            self.max_iterations
        )))
    }
}

/// `MASS::theta.ml`: Newton's method on the score of `θ` at fixed means,
/// from the method-of-moments start `n / Σ w (y/μ − 1)²`.
fn theta_ml(y: &[f64], mu: &[f64], w: &[f64]) -> Result<f64> {
    if let Some(&bad) = y.iter().find(|&&v| !(v >= 0.0 && v.fract() == 0.0)) {
        return Err(Error::InvalidParameter {
            name: "y",
            value: bad,
            reason: "the negative binomial needs whole, non-negative counts",
        });
    }
    let n: f64 = w.iter().sum();
    let moments: f64 = (0..y.len())
        .map(|i| w[i] * (y[i] / mu[i] - 1.0).powi(2))
        .sum();
    let mut theta = n / moments;
    for _ in 0..100 {
        let score: f64 = (0..y.len())
            .map(|i| {
                let (yi, m) = (y[i], mu[i]);
                w[i] * (digamma(theta + yi) - digamma(theta) + theta.ln() + 1.0
                    - (theta + m).ln()
                    - (yi + theta) / (m + theta))
            })
            .sum();
        let info = information(y, mu, w, theta);
        let step = score / info;
        let mut next = theta + step;
        // Halve a step that would leave θ > 0.
        let mut halvings = 0;
        while (next <= 0.0 || next.is_nan()) && halvings < 60 {
            next = theta + step / 2f64.powi(halvings + 1);
            halvings += 1;
        }
        if !next.is_finite() || next > 1e12 {
            return Err(Error::Data(
                "theta grows without bound: the counts are not overdispersed".into(),
            ));
        }
        let done = ((next - theta) / theta).abs() < 1e-13;
        theta = next;
        if done {
            break;
        }
    }
    Ok(theta)
}

/// The observed information for `θ` at fixed means (`MASS::theta.ml`).
fn information(y: &[f64], mu: &[f64], w: &[f64], theta: f64) -> f64 {
    (0..y.len())
        .map(|i| {
            let (yi, m) = (y[i], mu[i]);
            w[i] * (-trigamma(theta + yi) + trigamma(theta) - 1.0 / theta + 2.0 / (m + theta)
                - (yi + theta) / (m + theta).powi(2))
        })
        .sum()
}

/// A negative binomial GLM with `θ` estimated, from
/// [`NegativeBinomialGlm`].
#[derive(Debug, Clone, PartialEq)]
pub struct NegativeBinomialFit {
    theta: f64,
    se_theta: f64,
    log_likelihood: f64,
    iterations: usize,
    glm: GlmFit,
}

impl NegativeBinomialFit {
    /// The maximum-likelihood `θ`.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// `MASS`'s standard error of `θ`, `1 / sqrt(observed information)` at
    /// the fitted means.
    pub fn se_theta(&self) -> f64 {
        self.se_theta
    }

    /// The maximized log-likelihood.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// `-2 ℓ + 2 (p + 1)`: the coefficients and `θ`, as R's `AIC` on a
    /// `glm.nb` fit.
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * (self.glm.coefficients().len() + 1) as f64
    }

    /// Alternations of GLM fit and `θ` used.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The GLM at the estimated `θ`: coefficients, standard errors
    /// (conditional on `θ`) and the rest.
    pub fn glm(&self) -> &GlmFit {
        &self.glm
    }
}

impl Fitted for NegativeBinomialFit {
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        self.glm.predict(design)
    }

    /// The GLM's joint draws at the estimated `θ`: the coefficients'
    /// uncertainty is included, `θ`'s is not.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        self.glm.predict_distribution(design, n_sims, seed)
    }
}
