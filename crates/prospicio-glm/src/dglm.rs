//! Double generalized linear models: a GLM for the mean and a gamma GLM
//! with the log link for the dispersion, each with its own design (Smyth
//! 1989; Smyth and Verbyla 1999; R's `dglm`).
//!
//! Row `i` has mean `g(μᵢ) = xᵢᵀ β + oᵢ`, variance `φᵢ V(μᵢ) / wᵢ` and
//! `log φᵢ = zᵢᵀ γ + qᵢ`, so the dispersion varies with its own rating
//! variables (a severity whose spread differs by territory, a pure premium
//! whose Tweedie scale differs by class). The unit deviances
//! `dᵢ = wᵢ d(yᵢ, μᵢ)` are, by the saddlepoint approximation, gamma
//! responses with mean `φᵢ` and shape ½, and for the gamma family they
//! follow the digamma model of the exact gamma likelihood instead.
//! [`DoubleGlm::fit`] alternates a scoring step for `γ` at fixed means and
//! one for `β` at fixed dispersions until both settle, as `dglm`.
//!
//! The dispersion is estimated by restricted maximum likelihood by
//! default ([`DispersionMethod::Reml`]: each deviance's mean is corrected
//! by its leverage in the mean model, Smyth and Verbyla's adjusted
//! profile likelihood), or by maximum likelihood (`dglm`'s default).
//!
//! The mean step uses the scoring weights `w μ'(η)² / (φ V(μ))` with
//! `μ'` at the full linear predictor. `dglm` 1.8.6 uses `w V(μ) / φ` and
//! leaves out the offset, which is right only for a canonical link without
//! an offset, and its dispersion step leaves out the dispersion's offset
//! the same way; this fit agrees with it where it is right and with
//! `glm()` at a constant dispersion everywhere
//! (`knowledge/findings/dglm-mean-step-weights.md`).
//!
//! ```
//! use prospicio_glm::dglm::DoubleGlm;
//! use prospicio_models::{Design, Family, Link};
//!
//! // Severities in two territories, the second more spread out.
//! let urban = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
//! let y = [9.0, 11.0, 10.0, 10.5, 4.0, 18.0, 6.0, 15.0];
//! let d = Design::new(
//!     vec!["(Intercept)".into(), "urban".into()],
//!     vec![vec![1.0; 8], urban],
//! )
//! .unwrap();
//! let fit = DoubleGlm::new(Family::Gamma, Link::Log).fit(&d, &d, &y).unwrap();
//! assert!(fit.converged());
//! // The urban territory's dispersion is the larger.
//! assert!(fit.dispersion_coefficients()[1] > 0.0);
//! let phi = fit.dispersion_fitted();
//! assert!(phi[4] > phi[0]);
//! ```

use prospicio_core::{Error, Result};
use prospicio_math::linalg::{cholesky, cholesky_inverse, cholesky_solve};
use prospicio_math::special::{digamma, ln_gamma, trigamma};
use prospicio_models::{Design, Family, Link, Model};

use crate::Glm;

/// How the dispersion's coefficients are estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispersionMethod {
    /// Maximum likelihood (`dglm(method = "ml")`).
    Ml,
    /// Restricted maximum likelihood (`dglm(method = "reml")`).
    Reml,
}

/// A double GLM specification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleGlm {
    /// The Gaussian, Poisson, gamma, or Tweedie with `1 < p < 2`.
    pub family: Family,
    pub link: Link,
    pub method: DispersionMethod,
    /// The alternation stops when no coefficient moves by more than this,
    /// relative to its size (at least 1).
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl DoubleGlm {
    /// REML (`dglm`'s default is ML: [`ml`](Self::ml)), tolerance `1e-11`, at
    /// most 1000 alternations.
    pub fn new(family: Family, link: Link) -> Self {
        Self {
            family,
            link,
            method: DispersionMethod::Reml,
            tolerance: 1e-11,
            max_iterations: 1000,
        }
    }

    /// The same model by maximum likelihood.
    pub fn ml(self) -> Self {
        Self {
            method: DispersionMethod::Ml,
            ..self
        }
    }

    /// Fits the mean on `mean` (its offset and prior weights apply) and
    /// the log dispersion on `dispersion` (its offset applies; its weights
    /// must be 1). Starts from the GLM at a constant dispersion. Fails on
    /// another family, a response the family does not allow, designs of
    /// different lengths, a singular design, or no convergence.
    pub fn fit(&self, mean: &Design, dispersion: &Design, y: &[f64]) -> Result<DoubleGlmFit> {
        let family = self.family;
        match family {
            Family::Gaussian | Family::Poisson | Family::Gamma => {}
            Family::Tweedie { power } if power > 1.0 && power < 2.0 => {}
            _ => {
                return Err(Error::Data(format!(
                    "a double GLM takes the Gaussian, Poisson, gamma or Tweedie with 1 < p < 2, \
                     not the {}",
                    family.name()
                )));
            }
        }
        let n = y.len();
        if mean.n_rows() != n || dispersion.n_rows() != n {
            return Err(Error::Data(format!(
                "{n} responses, {} rows in the mean design and {} in the dispersion design",
                mean.n_rows(),
                dispersion.n_rows()
            )));
        }
        if let Some(&bad) = y.iter().find(|&&v| !family.valid_y(v)) {
            return Err(Error::InvalidParameter {
                name: "y",
                value: bad,
                reason: "outside the family's support",
            });
        }
        if dispersion.weights().iter().any(|&w| w != 1.0) {
            return Err(Error::Data(
                "the dispersion design takes no weights: the prior weights are the mean \
                 design's"
                    .into(),
            ));
        }
        let state = State {
            family,
            link: self.link,
            reml: self.method == DispersionMethod::Reml,
            x: mean,
            z: dispersion,
            y,
            w: mean.weights(),
        };

        // The start: the GLM's means, and the log deviances, by least
        // squares, plus 1.27036 (the mean of the log of a chi-square on
        // one degree of freedom, negated), as dglm.
        let start = Glm::new(family, self.link).fit(mean, y)?;
        let off = mean.offset();
        let mut eta: Vec<f64> = (0..n)
            .map(|i| state.link.link(start.fitted()[i]) - off[i])
            .collect();
        let mut mu = state.means(&eta)?;
        let mut d = state.deviances(&mu);
        let qoff = dispersion.offset();
        let logd: Vec<f64> = (0..n)
            .map(|i| (d[i] + if d[i] == 0.0 { 1.0 / 6.0 } else { 0.0 }).ln() - qoff[i])
            .collect();
        let gamma0 = wls(dispersion, &vec![1.0; n], &logd)?.coefficients;
        let mut deta: Vec<f64> = (0..n)
            .map(|i| row(dispersion, i, &gamma0) + 1.27036 + qoff[i])
            .collect();
        let mut phi: Vec<f64> = deta.iter().map(|e| e.exp()).collect();

        let mut mfit = state.mean_step(&eta, &mu, &phi)?;
        eta = mfit.fitted_without_offset(mean);
        mu = state.means(&eta)?;
        d = state.deviances(&mu);
        let mut theta: Vec<f64> = mfit.coefficients.clone();
        theta.extend(std::iter::repeat_n(f64::INFINITY, dispersion.n_cols()));
        for iteration in 1..=self.max_iterations {
            let (dfit, _) = state.dispersion_step(&phi, &deta, &mfit, &d)?;
            deta = (0..n)
                .map(|i| row(dispersion, i, &dfit.coefficients) + qoff[i])
                .collect();
            phi = deta.iter().map(|e| e.exp()).collect();
            mfit = state.mean_step(&eta, &mu, &phi)?;
            eta = mfit.fitted_without_offset(mean);
            mu = state.means(&eta)?;
            d = state.deviances(&mu);
            let now: Vec<f64> = mfit
                .coefficients
                .iter()
                .chain(&dfit.coefficients)
                .copied()
                .collect();
            let moved = now
                .iter()
                .zip(&theta)
                .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
                .fold(0.0, f64::max);
            theta = now;
            if moved < self.tolerance {
                return state.finish(eta, mu, phi, deta, iteration);
            }
        }
        Err(Error::Data(format!(
            "the double GLM did not converge in {} alternations",
            self.max_iterations
        )))
    }
}

/// A fitted double GLM.
#[derive(Debug, Clone)]
pub struct DoubleGlmFit {
    family: Family,
    link: Link,
    method: DispersionMethod,
    names: Vec<String>,
    coefficients: Vec<f64>,
    covariance: Vec<f64>,
    dispersion_names: Vec<String>,
    dispersion_coefficients: Vec<f64>,
    dispersion_covariance: Vec<f64>,
    fitted: Vec<f64>,
    dispersion_fitted: Vec<f64>,
    minus_two_log_likelihood: f64,
    reml_criterion: f64,
    iterations: usize,
}

impl DoubleGlmFit {
    pub fn family(&self) -> Family {
        self.family
    }

    pub fn link(&self) -> Link {
        self.link
    }

    pub fn method(&self) -> DispersionMethod {
        self.method
    }

    /// The mean design's column names.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// `β`.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// `β`'s covariance `(Xᵀ W X)⁻¹`, row-major, with the fitted
    /// dispersions in `W` (`summary(fit, dispersion = 1)` in `dglm`).
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    pub fn standard_errors(&self) -> Vec<f64> {
        diagonal_sqrt(&self.covariance, self.coefficients.len())
    }

    /// The dispersion design's column names.
    pub fn dispersion_names(&self) -> &[String] {
        &self.dispersion_names
    }

    /// `γ`, on the log scale.
    pub fn dispersion_coefficients(&self) -> &[f64] {
        &self.dispersion_coefficients
    }

    /// `γ`'s covariance `2 (Zᵀ W_d Z)⁻¹`, row-major
    /// (`summary(fit$dispersion.fit, dispersion = 2)`).
    pub fn dispersion_covariance(&self) -> &[f64] {
        &self.dispersion_covariance
    }

    pub fn dispersion_standard_errors(&self) -> Vec<f64> {
        diagonal_sqrt(
            &self.dispersion_covariance,
            self.dispersion_coefficients.len(),
        )
    }

    /// `μ` per row.
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// `φ` per row.
    pub fn dispersion_fitted(&self) -> &[f64] {
        &self.dispersion_fitted
    }

    /// `dglm`'s `m2loglik`: minus twice the log-likelihood, exact for the
    /// Gaussian and gamma and the saddlepoint approximation otherwise, with
    /// `dglm`'s constant `Σ log(2π V(y))` (`V(0)` taken as `V(1/6)`).
    pub fn minus_two_log_likelihood(&self) -> f64 {
        self.minus_two_log_likelihood
    }

    /// The criterion REML minimizes: [`minus_two_log_likelihood`](Self::minus_two_log_likelihood)
    /// plus `log det(Xᵀ W X)`. (`dglm` 1.8.6 means to report it under REML
    /// but adds 0 for the determinant.)
    pub fn reml_criterion(&self) -> f64 {
        self.reml_criterion
    }

    /// `m2loglik + 2 (p + q)` for `p` mean and `q` dispersion
    /// coefficients, as `AIC(dglm)`.
    pub fn aic(&self) -> f64 {
        self.minus_two_log_likelihood
            + 2.0 * (self.coefficients.len() + self.dispersion_coefficients.len()) as f64
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Always true: a fit that does not converge is an error.
    pub fn converged(&self) -> bool {
        true
    }

    /// `μ` on a new mean design, with its offset.
    pub fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        check_names(design, &self.names)?;
        Ok(design
            .linear_predictor(&self.coefficients)
            .into_iter()
            .map(|e| self.link.inverse(e))
            .collect())
    }

    /// `φ` on a new dispersion design, with its offset.
    pub fn predict_dispersion(&self, design: &Design) -> Result<Vec<f64>> {
        check_names(design, &self.dispersion_names)?;
        Ok(design
            .linear_predictor(&self.dispersion_coefficients)
            .into_iter()
            .map(f64::exp)
            .collect())
    }
}

fn check_names(design: &Design, names: &[String]) -> Result<()> {
    if design.names() != names {
        return Err(Error::Data(format!(
            "the design's columns {:?} are not the fit's {names:?}",
            design.names()
        )));
    }
    Ok(())
}

fn diagonal_sqrt(cov: &[f64], p: usize) -> Vec<f64> {
    (0..p).map(|j| cov[j * p + j].sqrt()).collect()
}

/// `zᵢᵀ c`, without the offset.
fn row(design: &Design, i: usize, c: &[f64]) -> f64 {
    c.iter()
        .enumerate()
        .map(|(j, b)| b * design.column(j)[i])
        .sum()
}

/// A weighted least-squares fit.
struct Wls {
    coefficients: Vec<f64>,
    /// The Cholesky factor of `Xᵀ W X`.
    factor: Vec<f64>,
    weights: Vec<f64>,
}

impl Wls {
    /// `X β`, without the offset.
    fn fitted_without_offset(&self, design: &Design) -> Vec<f64> {
        (0..design.n_rows())
            .map(|i| row(design, i, &self.coefficients))
            .collect()
    }

    /// `log det(Xᵀ W X)`.
    fn log_det(&self) -> f64 {
        let p = self.coefficients.len();
        2.0 * (0..p).map(|j| self.factor[j * p + j].ln()).sum::<f64>()
    }

    /// The hat values `wᵢ xᵢᵀ (Xᵀ W X)⁻¹ xᵢ`.
    fn hat(&self, design: &Design) -> Vec<f64> {
        let wt = &self.weights;
        let p = self.coefficients.len();
        let inv = cholesky_inverse(&self.factor, p);
        (0..design.n_rows())
            .map(|i| {
                let x: Vec<f64> = (0..p).map(|j| design.column(j)[i]).collect();
                let mut q = 0.0;
                for a in 0..p {
                    for b in 0..p {
                        q += x[a] * inv[a * p + b] * x[b];
                    }
                }
                wt[i] * q
            })
            .collect()
    }
}

/// Least squares of `z` on the design's columns with weights `wt`.
fn wls(design: &Design, wt: &[f64], z: &[f64]) -> Result<Wls> {
    let (n, p) = (design.n_rows(), design.n_cols());
    let mut a = vec![0.0; p * p];
    let mut b = vec![0.0; p];
    for j in 0..p {
        let xj = design.column(j);
        b[j] = (0..n).map(|i| wt[i] * xj[i] * z[i]).sum();
        for k in 0..=j {
            let xk = design.column(k);
            let v: f64 = (0..n).map(|i| wt[i] * xj[i] * xk[i]).sum();
            a[j * p + k] = v;
            a[k * p + j] = v;
        }
    }
    let factor = cholesky(&a, p).ok_or_else(|| {
        Error::Data(format!(
            "the design {:?} is singular at the working weights",
            design.names()
        ))
    })?;
    let coefficients = cholesky_solve(&factor, &b);
    Ok(Wls {
        coefficients,
        factor,
        weights: wt.to_vec(),
    })
}

struct State<'a> {
    family: Family,
    link: Link,
    reml: bool,
    x: &'a Design,
    z: &'a Design,
    y: &'a [f64],
    w: &'a [f64],
}

impl State<'_> {
    fn digamma(&self) -> bool {
        matches!(self.family, Family::Gamma)
    }

    fn means(&self, eta: &[f64]) -> Result<Vec<f64>> {
        let off = self.x.offset();
        eta.iter()
            .zip(off)
            .map(|(e, o)| {
                let full = e + o;
                let mu = self.link.inverse(full);
                if self.link.valid_eta(full) && self.family.valid_mu(mu) {
                    Ok(mu)
                } else {
                    Err(Error::Data(format!(
                        "the mean left the {}'s range at linear predictor {full}",
                        self.family.name()
                    )))
                }
            })
            .collect()
    }

    fn deviances(&self, mu: &[f64]) -> Vec<f64> {
        (0..mu.len())
            .map(|i| self.w[i] * self.family.unit_deviance(self.y[i], mu[i]))
            .collect()
    }

    /// The mean's scoring weights `w μ'(η)² / (V(μ) φ)`.
    fn mean_weights(&self, eta: &[f64], mu: &[f64], phi: &[f64]) -> Vec<f64> {
        let off = self.x.offset();
        (0..mu.len())
            .map(|i| {
                let me = self.link.mu_eta(eta[i] + off[i]);
                self.w[i] * me * me / self.family.variance(mu[i]) / phi[i]
            })
            .collect()
    }

    /// One scoring step for `β` at fixed dispersions.
    fn mean_step(&self, eta: &[f64], mu: &[f64], phi: &[f64]) -> Result<Wls> {
        let off = self.x.offset();
        let z: Vec<f64> = (0..mu.len())
            .map(|i| eta[i] + (self.y[i] - mu[i]) / self.link.mu_eta(eta[i] + off[i]))
            .collect();
        wls(self.x, &self.mean_weights(eta, mu, phi), &z)
    }

    /// One scoring step for `γ` at fixed means, with its weights.
    fn dispersion_step(
        &self,
        phi: &[f64],
        deta: &[f64],
        mfit: &Wls,
        d: &[f64],
    ) -> Result<(Wls, Vec<f64>)> {
        let n = phi.len();
        let qoff = self.z.offset();
        // The hat values of the mean's last step, as dglm.
        let h = if self.reml {
            Some(mfit.hat(self.x))
        } else {
            None
        };
        let mut zd = vec![0.0; n];
        let mut wd = vec![0.0; n];
        for i in 0..n {
            // d log φ / dφ's reciprocal for the log link: φ.
            let hdot = 1.0 / deta[i].exp();
            let (mut delta, u, fdot) = if self.digamma() {
                let s = self.w[i] / phi[i];
                let u = 2.0 * self.w[i] * self.w[i] * (trigamma(s) - 1.0 / s);
                (
                    2.0 * self.w[i] * (s.ln() - digamma(s)),
                    u,
                    phi[i] * phi[i] / u * hdot,
                )
            } else {
                (phi[i], phi[i] * phi[i], hdot)
            };
            let mut weight = 1.0 / (fdot * fdot * u);
            if let Some(h) = &h {
                delta -= phi[i] * h[i];
                weight += h[i] * h[i] - 2.0 * h[i] / (hdot * hdot * phi[i] * phi[i]);
            }
            if weight.is_nan() || weight <= 0.0 {
                return Err(Error::Data(format!(
                    "the dispersion's working weight is {weight} at row {i}: a point of high \
                     leverage in the mean model"
                )));
            }
            wd[i] = weight;
            zd[i] = deta[i] - qoff[i] + (d[i] - delta) * fdot;
        }
        Ok((wls(self.z, &wd, &zd)?, wd))
    }

    /// `dglm`'s `m2loglik` (see [`DoubleGlmFit::minus_two_log_likelihood`]).
    fn m2ll(&self, mu: &[f64], phi: &[f64]) -> f64 {
        let n = mu.len();
        let constant: f64 = if self.digamma() {
            2.0 * self.y.iter().map(|y| y.ln()).sum::<f64>()
        } else {
            self.y
                .iter()
                .map(|&y| {
                    let v = self.family.variance(y);
                    let v = if v == 0.0 {
                        self.family.variance(y + 1.0 / 6.0)
                    } else {
                        v
                    };
                    (2.0 * std::f64::consts::PI * v).ln()
                })
                .sum()
        };
        let d = self.deviances(mu);
        constant
            + (0..n)
                .map(|i| {
                    let (w, p) = (self.w[i], phi[i]);
                    let h = if self.digamma() {
                        2.0 * (ln_gamma(w / p) + (1.0 + (p / w).ln()) * w / p)
                    } else {
                        (p / w).ln()
                    };
                    h + d[i] / p
                })
                .sum::<f64>()
    }

    /// One more pass of each step at the converged values, so that the
    /// coefficients and weights are those of the solution.
    fn finish(
        &self,
        eta: Vec<f64>,
        mu: Vec<f64>,
        phi: Vec<f64>,
        deta: Vec<f64>,
        iterations: usize,
    ) -> Result<DoubleGlmFit> {
        let mfit = self.mean_step(&eta, &mu, &phi)?;
        let d = self.deviances(&mu);
        let (dfit, _) = self.dispersion_step(&phi, &deta, &mfit, &d)?;
        let (p, q) = (self.x.n_cols(), self.z.n_cols());
        let covariance = cholesky_inverse(&mfit.factor, p);
        let dispersion_covariance: Vec<f64> = cholesky_inverse(&dfit.factor, q)
            .into_iter()
            .map(|v| 2.0 * v)
            .collect();
        let m2 = self.m2ll(&mu, &phi);
        Ok(DoubleGlmFit {
            family: self.family,
            link: self.link,
            method: if self.reml {
                DispersionMethod::Reml
            } else {
                DispersionMethod::Ml
            },
            names: self.x.names().to_vec(),
            dispersion_names: self.z.names().to_vec(),
            fitted: self
                .x
                .linear_predictor(&mfit.coefficients)
                .into_iter()
                .map(|e| self.link.inverse(e))
                .collect(),
            dispersion_fitted: self
                .z
                .linear_predictor(&dfit.coefficients)
                .into_iter()
                .map(f64::exp)
                .collect(),
            coefficients: mfit.coefficients.clone(),
            covariance,
            dispersion_coefficients: dfit.coefficients,
            dispersion_covariance,
            minus_two_log_likelihood: m2,
            reml_criterion: m2 + mfit.log_det(),
            iterations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Design, Vec<f64>) {
        let x: Vec<f64> = (0..40u32).map(|i| f64::from(i % 5) / 4.0).collect();
        let y: Vec<f64> = (0..40u32)
            .map(|i| 100.0 * (1.0 + x[i as usize]) * (0.5 + f64::from((i * 7) % 13) / 12.0))
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; 40], x],
        )
        .unwrap();
        (d, y)
    }

    #[test]
    fn a_constant_dispersion_gives_the_glm() {
        let (d, y) = data();
        let one = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; 40]]).unwrap();
        let glm = Glm::new(Family::Gamma, Link::Log).fit(&d, &y).unwrap();
        for spec in [
            DoubleGlm::new(Family::Gamma, Link::Log),
            DoubleGlm::new(Family::Gamma, Link::Log).ml(),
        ] {
            let fit = spec.fit(&d, &one, &y).unwrap();
            for (a, b) in fit.coefficients().iter().zip(glm.coefficients()) {
                assert!((a - b).abs() < 1e-9, "{a} against {b}");
            }
            assert_eq!(fit.predict(&d).unwrap().len(), 40);
            let phi = fit.predict_dispersion(&one).unwrap();
            assert!((phi[0] - fit.dispersion_fitted()[0]).abs() < 1e-12);
        }
        // REML's dispersion is the larger: it allows for the estimated mean.
        let reml = DoubleGlm::new(Family::Gamma, Link::Log)
            .fit(&d, &one, &y)
            .unwrap();
        let ml = DoubleGlm::new(Family::Gamma, Link::Log)
            .ml()
            .fit(&d, &one, &y)
            .unwrap();
        assert!(reml.dispersion_fitted()[0] > ml.dispersion_fitted()[0]);
    }

    #[test]
    fn it_rejects_what_it_does_not_fit() {
        let (d, y) = data();
        assert!(
            DoubleGlm::new(Family::Binomial, Link::Logit)
                .fit(&d, &d, &y)
                .is_err()
        );
        assert!(
            DoubleGlm::new(Family::Tweedie { power: 2.5 }, Link::Log)
                .fit(&d, &d, &y)
                .is_err()
        );
        let short = d.select(&[0, 1, 2]);
        assert!(
            DoubleGlm::new(Family::Gamma, Link::Log)
                .fit(&d, &short, &y)
                .is_err()
        );
        let weighted = d.clone().with_weights(vec![2.0; 40]).unwrap();
        assert!(
            DoubleGlm::new(Family::Gamma, Link::Log)
                .fit(&d, &weighted, &y)
                .is_err()
        );
    }
}
