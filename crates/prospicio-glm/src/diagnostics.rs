//! Residuals, leverages and prediction errors of a fitted [`GlmFit`], as
//! R's `residuals.glm`, `statmod::qresiduals`, `hatvalues`, `rstandard`,
//! `cooks.distance` and `predict(..., se.fit = TRUE)`.
//!
//! The fit keeps its coefficients and covariance but not its training
//! data, so each method takes the design and, where it needs one, the
//! response.

use prospicio_core::{Error, Result};
use prospicio_models::{Design, Fitted, metrics};

use crate::GlmFit;

/// Per-row diagnostics of a GLM on its data, from [`GlmFit::diagnostics`].
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    /// The fitted means `μ`.
    pub fitted: Vec<f64>,
    /// `y − μ`.
    pub response: Vec<f64>,
    /// `(y − μ) √w / √V(μ)`.
    pub pearson: Vec<f64>,
    /// `sign(y − μ) √(w d(y, μ))`, whose squares add up to the deviance.
    pub deviance: Vec<f64>,
    /// `(y − μ) / (dμ/dη)`, IRLS's working residual.
    pub working: Vec<f64>,
    /// `w (dμ/dη)² / V(μ)`, IRLS's working weight.
    pub working_weights: Vec<f64>,
    /// The hat values `h`, the diagonal of `W^½ X (XᵀWX)⁻¹ Xᵀ W^½`.
    pub leverage: Vec<f64>,
    /// `pearson / √(φ (1 − h))`.
    pub standardized_pearson: Vec<f64>,
    /// `deviance / √(φ (1 − h))`, R's `rstandard`.
    pub standardized_deviance: Vec<f64>,
    /// Cook's distance, `(pearson / (1 − h))² h / (φ p)`.
    pub cooks_distance: Vec<f64>,
}

/// The prediction error of a sum of predicted means, from
/// [`GlmFit::sum_prediction_error`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SumPredictionError {
    /// `Σ μ`, the predicted total.
    pub total: f64,
    /// `g' Σ g` with `g = Σ (dμ/dη) x` over the rows: the parameter error's
    /// variance by the delta method.
    pub parameter_variance: f64,
    /// `Σ φ V(μ) / w`: the rows' own (process) variance.
    pub process_variance: f64,
}

impl SumPredictionError {
    /// `√(parameter + process variance)`, the root mean square error of
    /// prediction.
    pub fn rmsep(&self) -> f64 {
        (self.parameter_variance + self.process_variance).sqrt()
    }
}

impl GlmFit {
    /// Residuals, leverages and Cook's distances of the fit on `design`
    /// and `y` (normally its training data).
    ///
    /// Fails if `design`'s columns are not the fit's or `y` has the wrong
    /// length.
    pub fn diagnostics(&self, design: &Design, y: &[f64]) -> Result<Diagnostics> {
        self.check_rows(design, y.len())?;
        let (family, link) = (self.spec.family, self.spec.link);
        let eta = design.linear_predictor(&self.coefficients);
        let p = self.coefficients.len();
        let n = y.len();
        let phi = self.dispersion;
        let mut d = Diagnostics {
            fitted: Vec::with_capacity(n),
            response: Vec::with_capacity(n),
            pearson: Vec::with_capacity(n),
            deviance: Vec::with_capacity(n),
            working: Vec::with_capacity(n),
            working_weights: Vec::with_capacity(n),
            leverage: Vec::with_capacity(n),
            standardized_pearson: Vec::with_capacity(n),
            standardized_deviance: Vec::with_capacity(n),
            cooks_distance: Vec::with_capacity(n),
        };
        let mut x = vec![0.0; p];
        for i in 0..n {
            let mu = link.inverse(eta[i]);
            let w = design.weights()[i];
            let dmu = link.mu_eta(eta[i]);
            let v = family.variance(mu);
            let r = y[i] - mu;
            let pearson = r * (w / v).sqrt();
            let deviance = r.signum() * (w * family.unit_deviance(y[i], mu)).sqrt();
            let ww = w * dmu * dmu / v;
            for (j, xj) in x.iter_mut().enumerate() {
                *xj = design.column(j)[i];
            }
            let h = ww * self.quadratic_form(&x);
            let s = (phi * (1.0 - h)).sqrt();
            d.fitted.push(mu);
            d.response.push(r);
            d.pearson.push(pearson);
            d.deviance.push(deviance);
            d.working.push(r / dmu);
            d.working_weights.push(ww);
            d.leverage.push(h);
            d.standardized_pearson.push(pearson / s);
            d.standardized_deviance.push(deviance / s);
            d.cooks_distance
                .push((pearson / (1.0 - h)).powi(2) * h / (phi * p as f64));
        }
        Ok(d)
    }

    /// Randomized quantile residuals of the fit on `design` and `y` at its
    /// dispersion, standard normal when the model is right (Dunn and Smyth,
    /// 1996; [`metrics::quantile_residuals`]): exact for a continuous family,
    /// drawn from `seed` where the response has an atom (the counts, the
    /// Tweedie at zero). R's `statmod::qresiduals(m, dispersion =
    /// summary(m)$dispersion)`; statmod's own default dispersion differs
    /// for the inverse Gaussian.
    ///
    /// Fails as [`diagnostics`](Self::diagnostics).
    pub fn quantile_residuals(&self, design: &Design, y: &[f64], seed: u64) -> Result<Vec<f64>> {
        self.check_rows(design, y.len())?;
        let mu = self.predict(design)?;
        metrics::quantile_residuals(
            self.spec.family,
            y,
            &mu,
            self.dispersion,
            Some(design.weights()),
            seed,
        )
    }

    /// The standard error of each row's predicted mean, `|dμ/dη| √(x'Σx)`
    /// with `Σ` the coefficients' covariance (parameter error only), as R's
    /// `predict(type = "response", se.fit = TRUE)`.
    ///
    /// Fails if `design`'s columns are not the fit's.
    pub fn prediction_se(&self, design: &Design) -> Result<Vec<f64>> {
        self.check_rows(design, design.n_rows())?;
        let eta = design.linear_predictor(&self.coefficients);
        let p = self.coefficients.len();
        let mut x = vec![0.0; p];
        Ok((0..design.n_rows())
            .map(|i| {
                for (j, xj) in x.iter_mut().enumerate() {
                    *xj = design.column(j)[i];
                }
                self.spec.link.mu_eta(eta[i]).abs()
                    * (self.dispersion * self.quadratic_form(&x)).sqrt()
            })
            .collect())
    }

    /// The prediction error of the sum of the predicted means of
    /// `design`'s rows (a reserve: the future cells of an origin), by the
    /// delta method for the parameter error and the family's variance for
    /// the process error (Taylor and McGuire, CAS Monograph 3, (5-11) to
    /// (5-13)).
    ///
    /// Fails if `design`'s columns are not the fit's.
    pub fn sum_prediction_error(&self, design: &Design) -> Result<SumPredictionError> {
        self.check_rows(design, design.n_rows())?;
        let eta = design.linear_predictor(&self.coefficients);
        let p = self.coefficients.len();
        let mut g = vec![0.0; p];
        let (mut total, mut process) = (0.0, 0.0);
        for (i, &e) in eta.iter().enumerate() {
            let mu = self.spec.link.inverse(e);
            let dmu = self.spec.link.mu_eta(e);
            for (j, gj) in g.iter_mut().enumerate() {
                *gj += dmu * design.column(j)[i];
            }
            total += mu;
            process += self.dispersion * self.spec.family.variance(mu) / design.weights()[i];
        }
        Ok(SumPredictionError {
            total,
            parameter_variance: self.dispersion * self.quadratic_form(&g),
            process_variance: process,
        })
    }

    /// Generalized cross-validation on `design` and `y`,
    /// `n Σ (y − μ)² / (n − p)²` (Taylor and McGuire, section 4.4).
    ///
    /// Fails as [`diagnostics`](Self::diagnostics), or with no more rows
    /// than coefficients.
    pub fn gcv(&self, design: &Design, y: &[f64]) -> Result<f64> {
        self.check_rows(design, y.len())?;
        let mu = self.predict(design)?;
        let n = y.len() as f64;
        let df = n - self.coefficients.len() as f64;
        if df <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "y",
                value: n,
                reason: "needs more rows than coefficients",
            });
        }
        let rss: f64 = y.iter().zip(&mu).map(|(y, m)| (y - m) * (y - m)).sum();
        Ok(n * rss / (df * df))
    }

    /// The log-likelihood of `y` at the fit's means on `design` and a
    /// given `dispersion`, to compare models at one held dispersion (the
    /// family's full log-likelihood, constants included).
    ///
    /// Fails as [`diagnostics`](Self::diagnostics).
    pub fn log_likelihood_at(&self, design: &Design, y: &[f64], dispersion: f64) -> Result<f64> {
        self.check_rows(design, y.len())?;
        let mu = self.predict(design)?;
        Ok(y.iter()
            .zip(&mu)
            .zip(design.weights())
            .map(|((&y, &m), &w)| self.spec.family.log_likelihood(y, m, w, dispersion))
            .sum())
    }

    /// `x' (XᵀWX)⁻¹ x`.
    fn quadratic_form(&self, x: &[f64]) -> f64 {
        let p = x.len();
        let c = &self.unscaled_covariance;
        (0..p)
            .map(|a| x[a] * (0..p).map(|b| c[a * p + b] * x[b]).sum::<f64>())
            .sum()
    }

    fn check_rows(&self, design: &Design, n: usize) -> Result<()> {
        self.check_design(design)?;
        if design.n_rows() != n {
            return Err(Error::Data(format!(
                "y has {n} rows and the design {}",
                design.n_rows()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::Glm;
    use prospicio_models::{Design, Family, Link, Model};

    fn fit() -> (crate::GlmFit, Design, Vec<f64>) {
        let n = 30;
        let x: Vec<f64> = (0..n).map(|i| (i % 6) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i * 7) % 5 + i % 6) as f64).collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; n], x],
        )
        .unwrap();
        let glm = Glm::new(Family::Poisson, Link::Log).dispersion(crate::Dispersion::Pearson);
        (glm.fit(&d, &y).unwrap(), d, y)
    }

    #[test]
    fn one_row_sum_is_its_prediction_error() {
        let (fit, d, _) = fit();
        let one = d.select(&[4]);
        let se = fit.prediction_se(&one).unwrap()[0];
        let e = fit.sum_prediction_error(&one).unwrap();
        assert!((e.parameter_variance - se * se).abs() < 1e-12 * se * se);
        // ODP: the process variance is φ μ.
        assert!((e.process_variance - fit.dispersion() * e.total).abs() < 1e-12 * e.total);
        assert!(e.rmsep() > se);
    }

    #[test]
    fn leverages_add_up_to_the_coefficients_and_gcv() {
        let (fit, d, y) = fit();
        let diag = fit.diagnostics(&d, &y).unwrap();
        let trace: f64 = diag.leverage.iter().sum();
        assert!((trace - 2.0).abs() < 1e-9, "{trace}");
        let rss: f64 = diag.response.iter().map(|r| r * r).sum();
        let gcv = fit.gcv(&d, &y).unwrap();
        assert!((gcv - 30.0 * rss / (28.0 * 28.0)).abs() < 1e-9 * gcv);
        let at_one = fit.log_likelihood_at(&d, &y, 1.0).unwrap();
        let poisson = Glm::new(Family::Poisson, Link::Log).fit(&d, &y).unwrap();
        assert!((at_one - poisson.log_likelihood()).abs() < 1e-8);
        assert!(fit.diagnostics(&d, &y[1..]).is_err());
    }
}
