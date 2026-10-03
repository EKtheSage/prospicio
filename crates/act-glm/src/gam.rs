//! Generalized additive models: a GLM with penalized B-spline smooths
//! (P-splines), fitted by penalized IRLS with smoothing parameters chosen
//! by GCV or UBRE.
//!
//! Each smooth replaces a numeric column of the design with a cubic
//! B-spline basis on equally spaced knots, with a second-order difference
//! penalty: mgcv's `s(x, bs = "ps")`. The basis is reparameterized so the
//! smooth sums to zero over the data (mgcv's centering constraint), which
//! keeps it identifiable next to the intercept.

use act_core::{Error, Result};
use act_math::linalg::{cholesky, cholesky_inverse};
use act_math::optimize::minimize;
use act_math::spline::{bspline_basis, difference_penalty, pspline_knots};
use act_models::{Design, Fitted, Model};
use act_prob::{PredictiveDistribution, Provenance};

use crate::{Dispersion, Glm, Irls, dev, irls, simulate_responses};

/// A P-spline smooth of one numeric design column.
#[derive(Debug, Clone, PartialEq)]
pub struct PSpline {
    /// The design column it replaces.
    pub column: String,
    /// Number of B-spline basis functions before the centering constraint
    /// (mgcv's `k`); the smooth has `n_basis - 1` coefficients.
    pub n_basis: usize,
    /// B-spline degree (3: cubic).
    pub degree: usize,
    /// Order of the difference penalty (2 penalizes curvature, leaving a
    /// straight line unpenalized).
    pub order: usize,
}

impl PSpline {
    /// A cubic P-spline with 10 basis functions and a second-order
    /// penalty, mgcv's `s(column, bs = "ps")` default.
    pub fn new(column: &str) -> Self {
        Self {
            column: column.into(),
            n_basis: 10,
            degree: 3,
            order: 2,
        }
    }

    /// The same smooth with `k` basis functions.
    pub fn n_basis(mut self, k: usize) -> Self {
        self.n_basis = k;
        self
    }
}

/// How smoothing parameters are chosen.
#[derive(Debug, Clone, PartialEq)]
pub enum Smoothing {
    /// UBRE (Mallows' Cp) when the dispersion is fixed, GCV when it is
    /// estimated: mgcv's `method = "GCV.Cp"`.
    Auto,
    /// Generalized cross-validation, `n D / (n - edf)²`.
    Gcv,
    /// Un-biased risk estimator, `D / n + 2 φ edf / n - φ`.
    Ubre,
    /// Fixed smoothing parameters, one per smooth.
    Fixed(Vec<f64>),
}

/// A GAM specification: a [`Glm`] (family, link, dispersion) plus smooths.
///
/// ```
/// use act_glm::{Glm, gam::{Gam, PSpline}};
/// use act_models::{Design, Family, Fitted, Link, Model};
///
/// // A sine curve plus noise, smoothed.
/// let x: Vec<f64> = (0..200).map(|i| f64::from(i) / 199.0).collect();
/// let y: Vec<f64> = x
///     .iter()
///     .enumerate()
///     .map(|(i, v)| (6.0 * v).sin() + 0.1 * ((i * 7919 % 13) as f64 / 6.0 - 1.0))
///     .collect();
/// let design = Design::new(
///     vec!["(Intercept)".into(), "x".into()],
///     vec![vec![1.0; 200], x.clone()],
/// )
/// .unwrap();
/// let gam = Gam::new(Glm::new(Family::Gaussian, Link::Identity), vec![PSpline::new("x")]);
/// let fit = gam.fit(&design, &y).unwrap();
/// let pred = fit.predict(&design).unwrap();
/// assert!((pred[100] - (6.0 * x[100]).sin()).abs() < 0.05);
/// assert!(fit.edf() > 4.0 && fit.edf() < 10.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Gam {
    pub glm: Glm,
    pub smooths: Vec<PSpline>,
    pub smoothing: Smoothing,
}

impl Gam {
    /// A GAM with smoothing parameters chosen by [`Smoothing::Auto`].
    pub fn new(glm: Glm, smooths: Vec<PSpline>) -> Self {
        Self {
            glm,
            smooths,
            smoothing: Smoothing::Auto,
        }
    }

    /// The same GAM with smoothing parameters chosen by `smoothing`.
    pub fn smoothing(mut self, smoothing: Smoothing) -> Self {
        self.smoothing = smoothing;
        self
    }
}

/// A smooth's basis as fitted: knots from the training data and the
/// constraint's null space `Z` (`k × (k - 1)`, row-major).
#[derive(Debug, Clone, PartialEq)]
struct Basis {
    spline: PSpline,
    knots: Vec<f64>,
    z: Vec<f64>,
}

impl Basis {
    /// Learns knots from `x` and the centering constraint from its basis.
    fn fit(spline: &PSpline, x: &[f64]) -> Result<Self> {
        let k = spline.n_basis;
        if k < spline.degree + 2 || spline.order >= k {
            return Err(Error::InvalidParameter {
                name: "n_basis",
                value: k as f64,
                reason: "is too small for the degree and penalty order",
            });
        }
        let (lo, hi) = x
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        if hi <= lo || hi.is_nan() || lo.is_nan() {
            return Err(Error::Data(format!(
                "smooth column {:?} has no spread",
                spline.column
            )));
        }
        let knots = pspline_knots(lo, hi, k, spline.degree);
        // c = column sums of the basis; Z spans c⊥ (Householder).
        let mut c = vec![0.0; k];
        for &v in x {
            for (cj, b) in c.iter_mut().zip(bspline_basis(v, &knots, spline.degree)) {
                *cj += b;
            }
        }
        let norm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut v = c;
        v[0] += norm.copysign(v[0]);
        let vv: f64 = v.iter().map(|a| a * a).sum();
        let mut z = vec![0.0; k * (k - 1)];
        for i in 0..k {
            for j in 1..k {
                let h = f64::from(u8::from(i == j)) - 2.0 * v[i] * v[j] / vv;
                z[i * (k - 1) + j - 1] = h;
            }
        }
        Ok(Self {
            spline: spline.clone(),
            knots,
            z,
        })
    }

    fn width(&self) -> usize {
        self.spline.n_basis - 1
    }

    /// The constrained basis columns `B Z` at `x`.
    fn columns(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let (k, m) = (self.spline.n_basis, self.width());
        let mut cols = vec![vec![0.0; x.len()]; m];
        for (r, &v) in x.iter().enumerate() {
            let b = bspline_basis(v, &self.knots, self.spline.degree);
            for (j, col) in cols.iter_mut().enumerate() {
                col[r] = (0..k).map(|i| b[i] * self.z[i * m + j]).sum();
            }
        }
        cols
    }

    /// `Zᵀ P Z`, `(k - 1) × (k - 1)`.
    fn penalty(&self) -> Vec<f64> {
        let (k, m) = (self.spline.n_basis, self.width());
        let p = difference_penalty(k, self.spline.order);
        let mut out = vec![0.0; m * m];
        for a in 0..m {
            for b in 0..m {
                let mut s = 0.0;
                for i in 0..k {
                    for j in 0..k {
                        s += self.z[i * m + a] * p[i * k + j] * self.z[j * m + b];
                    }
                }
                out[a * m + b] = s;
            }
        }
        out
    }
}

/// Replaces each smooth's raw column of `design` by its basis columns,
/// appended after the parametric columns.
fn expand(design: &Design, bases: &[Basis]) -> Result<Design> {
    let raw: Vec<&str> = bases.iter().map(|b| b.spline.column.as_str()).collect();
    let mut names = Vec::new();
    let mut cols = Vec::new();
    for (j, name) in design.names().iter().enumerate() {
        if !raw.contains(&name.as_str()) {
            names.push(name.clone());
            cols.push(design.column(j).to_vec());
        }
    }
    for b in bases {
        let j = design
            .names()
            .iter()
            .position(|n| *n == b.spline.column)
            .ok_or_else(|| Error::Data(format!("no design column {:?}", b.spline.column)))?;
        for (i, col) in b.columns(design.column(j)).into_iter().enumerate() {
            names.push(format!("s({}).{}", b.spline.column, i + 1));
            cols.push(col);
        }
    }
    Design::new(names, cols)?
        .with_offset(design.offset().to_vec())?
        .with_weights(design.weights().to_vec())
}

/// A fitted GAM.
#[derive(Debug, Clone, PartialEq)]
pub struct GamFit {
    spec: Gam,
    bases: Vec<Basis>,
    names: Vec<String>,
    coefficients: Vec<f64>,
    lambdas: Vec<f64>,
    edf: f64,
    covariance: Vec<f64>,
    dispersion: f64,
    deviance: f64,
    score: f64,
    fitted: Vec<f64>,
}

/// One penalized fit at fixed smoothing parameters.
struct Trial {
    irls: Irls,
    penalty: Vec<f64>,
    edf: f64,
    deviance: f64,
}

impl Model for Gam {
    type Fitted = GamFit;

    fn fit(&self, design: &Design, y: &[f64]) -> Result<GamFit> {
        let glm = self.glm;
        glm.family.validate()?;
        if self.smooths.is_empty() {
            return Err(Error::InvalidParameter {
                name: "smooths",
                value: 0.0,
                reason: "must not be empty (fit a Glm instead)",
            });
        }
        let bases = self
            .smooths
            .iter()
            .map(|s| {
                let j = design
                    .names()
                    .iter()
                    .position(|n| *n == s.column)
                    .ok_or_else(|| Error::Data(format!("no design column {:?}", s.column)))?;
                Basis::fit(s, design.column(j))
            })
            .collect::<Result<Vec<_>>>()?;
        let x = expand(design, &bases)?;
        let (n, p) = (x.n_rows(), x.n_cols());
        if y.len() != n {
            return Err(Error::Data(format!(
                "{} responses for {n} design rows",
                y.len()
            )));
        }
        if let Some(bad) = y.iter().find(|&&v| !glm.family.valid_y(v)) {
            return Err(Error::InvalidParameter {
                name: "y",
                value: *bad,
                reason: "is outside the family's range",
            });
        }
        // Each smooth's penalty block sits at the end, in order.
        let blocks: Vec<(usize, Vec<f64>)> = {
            let mut start = p - bases.iter().map(Basis::width).sum::<usize>();
            bases
                .iter()
                .map(|b| {
                    let at = start;
                    start += b.width();
                    (at, b.penalty())
                })
                .collect()
        };
        let w = x.weights();
        let y_mean = y.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / w.iter().sum::<f64>();
        let trial = |lambdas: &[f64]| -> Result<Trial> {
            let mut s = vec![0.0; p * p];
            for ((at, block), (lambda, basis)) in blocks.iter().zip(lambdas.iter().zip(&bases)) {
                let m = basis.width();
                for a in 0..m {
                    for b in 0..m {
                        s[(at + a) * p + at + b] += lambda * block[a * m + b];
                    }
                }
            }
            let fit = irls(
                &glm,
                &x,
                y,
                |i| glm.family.initial_mu(y[i], w[i], y_mean),
                Some(&s),
            )?;
            // edf = tr((XᵀWX + S)⁻¹ XᵀWX) = p - tr((XᵀWX + S)⁻¹ S).
            let mut a = fit.information.clone();
            for (v, si) in a.iter_mut().zip(&s) {
                *v += si;
            }
            let l = cholesky(&a, p)
                .ok_or_else(|| Error::Data("the penalized system is singular".into()))?;
            let inv = cholesky_inverse(&l, p);
            let trace: f64 = (0..p)
                .map(|i| (0..p).map(|k| inv[i * p + k] * s[k * p + i]).sum::<f64>())
                .sum();
            let deviance = dev(glm.family, y, &fit.mu, w);
            Ok(Trial {
                irls: fit,
                penalty: s,
                edf: p as f64 - trace,
                deviance,
            })
        };
        let fixed = match glm.dispersion {
            Dispersion::Fixed(phi) => Some(phi),
            _ => None,
        };
        let use_ubre = match self.smoothing {
            Smoothing::Ubre => true,
            Smoothing::Gcv => false,
            _ => fixed.is_some(),
        };
        let nf = n as f64;
        let score = |t: &Trial| -> f64 {
            if use_ubre {
                let phi = fixed.unwrap_or(1.0);
                t.deviance / nf + 2.0 * phi * t.edf / nf - phi
            } else {
                nf * t.deviance / (nf - t.edf).powi(2)
            }
        };
        let lambdas = match &self.smoothing {
            Smoothing::Fixed(l) => {
                if l.len() != bases.len() || l.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
                    return Err(Error::InvalidParameter {
                        name: "smoothing",
                        value: l.len() as f64,
                        reason: "needs one finite, non-negative value per smooth",
                    });
                }
                l.clone()
            }
            _ => {
                // Coordinate search on ln λ, smooth by smooth, three passes.
                let mut log_l = vec![0.0; bases.len()];
                let passes = if bases.len() == 1 { 1 } else { 3 };
                for _ in 0..passes {
                    for j in 0..bases.len() {
                        let objective = |v: f64| {
                            let mut l: Vec<f64> = log_l.iter().map(|a: &f64| a.exp()).collect();
                            l[j] = v.exp();
                            trial(&l).map_or(f64::INFINITY, |t| score(&t))
                        };
                        log_l[j] = minimize(-15.0, 20.0, objective);
                    }
                }
                log_l.iter().map(|v| v.exp()).collect()
            }
        };
        let best = trial(&lambdas)?;
        let df_resid = nf - best.edf;
        let dispersion = match glm.dispersion {
            Dispersion::Fixed(phi) => phi,
            Dispersion::Deviance => best.deviance / df_resid,
            Dispersion::Pearson => {
                (0..n)
                    .map(|i| {
                        let m = best.irls.mu[i];
                        w[i] * (y[i] - m).powi(2) / glm.family.variance(m)
                    })
                    .sum::<f64>()
                    / df_resid
            }
        };
        // Bayesian posterior covariance φ (XᵀWX + S)⁻¹ (mgcv's Vp).
        let mut a = best.irls.information.clone();
        for (v, si) in a.iter_mut().zip(&best.penalty) {
            *v += si;
        }
        let l = cholesky(&a, p)
            .ok_or_else(|| Error::Data("the penalized system is singular".into()))?;
        let covariance = cholesky_inverse(&l, p)
            .into_iter()
            .map(|v| v * dispersion)
            .collect();
        Ok(GamFit {
            spec: self.clone(),
            score: score(&best),
            bases,
            names: x.names().to_vec(),
            coefficients: best.irls.beta,
            lambdas,
            edf: best.edf,
            covariance,
            dispersion,
            deviance: best.deviance,
            fitted: best.irls.mu,
        })
    }
}

impl GamFit {
    /// Coefficient names: the parametric columns, then `s(x).1`, … for
    /// each smooth.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Coefficients.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Smoothing parameter of each smooth.
    pub fn lambdas(&self) -> &[f64] {
        &self.lambdas
    }

    /// Effective degrees of freedom of the whole model,
    /// `tr((XᵀWX + S)⁻¹ XᵀWX)`.
    pub fn edf(&self) -> f64 {
        self.edf
    }

    /// Bayesian posterior covariance of the coefficients,
    /// `φ (XᵀWX + S)⁻¹`, row-major.
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Dispersion `φ` (Pearson's, on `n - edf` degrees of freedom, when
    /// estimated).
    pub fn dispersion(&self) -> f64 {
        self.dispersion
    }

    /// Residual deviance.
    pub fn deviance(&self) -> f64 {
        self.deviance
    }

    /// The minimized GCV or UBRE score (or the score at fixed smoothing
    /// parameters).
    pub fn score(&self) -> f64 {
        self.score
    }

    /// Fitted means on the training data.
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }
}

impl Fitted for GamFit {
    /// Predictions for a design with the same columns as the training
    /// design (raw smooth columns included).
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        let x = expand(design, &self.bases)?;
        if x.names() != self.names.as_slice() {
            return Err(Error::Data(
                "design columns do not match the fitted GAM".into(),
            ));
        }
        Ok(x.linear_predictor(&self.coefficients)
            .into_iter()
            .map(|e| self.spec.glm.link.inverse(e))
            .collect())
    }

    /// Joint draws as for a GLM, with `β` drawn from its posterior
    /// `N(β̂, φ (XᵀWX + S)⁻¹)`.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        let x = expand(design, &self.bases)?;
        let provenance = Provenance::new("gam")
            .version("act-glm", env!("CARGO_PKG_VERSION"))
            .param("family", self.spec.glm.family.name())
            .param("link", format!("{:?}", self.spec.glm.link))
            .param("lambdas", format!("{:?}", self.lambdas));
        simulate_responses(
            &self.spec.glm,
            self.dispersion,
            &self.coefficients,
            &self.covariance,
            &x,
            n_sims,
            seed,
            provenance,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use act_models::{Family, Link};

    #[test]
    fn constrained_basis_sums_to_zero_over_the_data() {
        let x: Vec<f64> = (0..50).map(|i| f64::from(i).sqrt()).collect();
        let b = Basis::fit(&PSpline::new("x"), &x).unwrap();
        for col in b.columns(&x) {
            assert!(col.iter().sum::<f64>().abs() < 1e-10);
        }
    }

    #[test]
    fn huge_smoothing_gives_a_straight_line() {
        // λ → ∞ leaves only the unpenalized linear part: the GLM on x.
        let x: Vec<f64> = (0..60).map(|i| f64::from(i) / 10.0).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|v| 1.0 + 0.5 * v + (3.0 * v).sin() * 0.3)
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; 60], x.clone()],
        )
        .unwrap();
        let glm = Glm::new(Family::Gaussian, Link::Identity);
        let gam = Gam::new(glm, vec![PSpline::new("x")]).smoothing(Smoothing::Fixed(vec![1e8]));
        let fit = gam.fit(&d, &y).unwrap();
        let line = glm.fit(&d, &y).unwrap();
        for (a, b) in fit.fitted().iter().zip(line.fitted()) {
            assert!((a - b).abs() < 1e-6, "{a} {b}");
        }
        assert!((fit.edf() - 2.0).abs() < 1e-6);
    }
}
