//! Elastic-net GLMs: the lasso, ridge and everything between, by
//! coordinate descent inside IRLS (Friedman, Hastie and Tibshirani,
//! *Regularization Paths for Generalized Linear Models via Coordinate
//! Descent*, 2010).
//!
//! [`ElasticNet`] minimizes, over the coefficients `β` and an unpenalized
//! intercept,
//!
//! ```text
//! Σ wᵢ dᵢ / (2 Σ w)  +  λ Σⱼ pfⱼ ((1 - α)/2 bⱼ² + α |bⱼ|)
//! ```
//!
//! where `dᵢ` is the family's unit deviance and `bⱼ = sⱼ βⱼ` is the
//! coefficient of column `j` standardized to unit weighted standard
//! deviation (`sⱼ = 1` without standardization). This is glmnet's
//! objective, so `λ` and `α` mean what they mean there; the parity suite
//! checks against glmnet. `α = 1` is the lasso, `α = 0` ridge.

use prospicio_core::{Error, Result};
use prospicio_models::resample::{Split, map_splits};
use prospicio_models::{Design, Family, Fitted, Link, Model};
use prospicio_prob::{PredictiveDistribution, Provenance};

use crate::{Dispersion, Glm, dev, simulate_responses};

/// An elastic-net GLM specification.
///
/// The design's first all-ones column, if any, is the unpenalized
/// intercept; every other column is penalized, scaled by its penalty
/// factor (all 1 by default; 0 leaves a column unpenalized). Columns are
/// standardized for the penalty by default, and coefficients are always
/// reported on the design's own scale.
///
/// ```
/// use prospicio_glm::net::ElasticNet;
/// use prospicio_models::{Design, Family, Link, Model};
///
/// let x1 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
/// let x2 = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
/// let design = Design::new(
///     vec!["(Intercept)".into(), "x1".into(), "x2".into()],
///     vec![vec![1.0; 6], x1, x2],
/// )
/// .unwrap();
/// let y = [1.0, 3.1, 4.9, 7.2, 9.0, 10.8];
/// let lasso = ElasticNet::new(Family::Gaussian, Link::Identity, 1.0, 0.0);
/// // Above λ_max every penalized coefficient is zero.
/// let lmax = lasso.lambda_max(&design, &y).unwrap();
/// let fit = lasso.with_lambda(lmax * 1.01).fit(&design, &y).unwrap();
/// assert_eq!(&fit.coefficients()[1..], &[0.0, 0.0]);
/// // Below it, x1 enters first; x2 is pure noise and stays out a while.
/// let fit = lasso.with_lambda(lmax * 0.5).fit(&design, &y).unwrap();
/// assert!(fit.coefficients()[1] > 0.0 && fit.coefficients()[2] == 0.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticNet {
    pub family: Family,
    pub link: Link,
    /// Mixing: 1 is the lasso, 0 ridge.
    pub alpha: f64,
    /// Penalty strength `λ >= 0`.
    pub lambda: f64,
    /// Penalize the standardized coefficients (glmnet's default).
    pub standardize: bool,
    /// One factor per design column (the intercept's is ignored), rescaled
    /// to sum to the number of penalized columns as glmnet does.
    pub penalty_factor: Option<Vec<f64>>,
    /// Convergence: relative change in the objective, and coefficient
    /// changes below `1e-3 √tolerance` relative, as [`Glm`].
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl ElasticNet {
    /// An elastic net with mixing `alpha` and strength `lambda`,
    /// standardized, tolerance `1e-12`, at most 100 IRLS iterations.
    pub fn new(family: Family, link: Link, alpha: f64, lambda: f64) -> Self {
        Self {
            family,
            link,
            alpha,
            lambda,
            standardize: true,
            penalty_factor: None,
            tolerance: 1e-12,
            max_iterations: 100,
        }
    }

    /// The same spec at another `λ`.
    pub fn with_lambda(&self, lambda: f64) -> Self {
        Self {
            lambda,
            ..self.clone()
        }
    }

    /// The same spec with or without standardization.
    pub fn standardize(mut self, on: bool) -> Self {
        self.standardize = on;
        self
    }

    /// The same spec with per-column penalty factors.
    pub fn penalty_factor(mut self, factors: Vec<f64>) -> Self {
        self.penalty_factor = Some(factors);
        self
    }

    /// The smallest `λ` at which every penalized coefficient is zero:
    /// `max |∂/∂bⱼ| / (α pfⱼ)` at the model with only the intercept and
    /// unpenalized columns. For `α < 0.001` it uses `α = 0.001`, as glmnet
    /// does, since ridge never zeroes a coefficient.
    pub fn lambda_max(&self, design: &Design, y: &[f64]) -> Result<f64> {
        let prep = self.prepare(design, y)?;
        let null = self.solve(&prep, design, y, f64::INFINITY, None)?;
        let grad = prep.gradient(self, design, y, &null.eta);
        let alpha = self.alpha.max(1e-3);
        Ok(prep
            .penalized()
            .map(|j| grad[j].abs() / (alpha * prep.pf[j]))
            .fold(0.0, f64::max))
    }

    /// `n` values of `λ`, log-spaced from [`lambda_max`](Self::lambda_max)
    /// down to `min_ratio` times it (glmnet uses 100 and `1e-4`, or `0.01`
    /// when there are fewer rows than columns).
    pub fn lambda_path(
        &self,
        design: &Design,
        y: &[f64],
        n: usize,
        min_ratio: f64,
    ) -> Result<Vec<f64>> {
        if n < 1 || !(min_ratio > 0.0 && min_ratio < 1.0) {
            return Err(Error::Data(
                "a lambda path needs at least one value and 0 < min_ratio < 1".into(),
            ));
        }
        let top = self.lambda_max(design, y)?;
        if n == 1 {
            return Ok(vec![top]);
        }
        let step = min_ratio.ln() / (n - 1) as f64;
        Ok((0..n).map(|k| top * (step * k as f64).exp()).collect())
    }

    /// Fits at each `λ` in `lambdas` (any order), each starting from the
    /// previous solution, which is much faster than cold starts down a
    /// decreasing path.
    pub fn path(&self, design: &Design, y: &[f64], lambdas: &[f64]) -> Result<Vec<ElasticNetFit>> {
        let prep = self.prepare(design, y)?;
        let null_deviance = self.null_deviance(&prep, design, y)?;
        let mut start: Option<Solution> = None;
        let mut out = Vec::with_capacity(lambdas.len());
        for &lambda in lambdas {
            check_lambda(lambda)?;
            let sol = self.solve(&prep, design, y, lambda, start.as_ref())?;
            out.push(
                self.with_lambda(lambda)
                    .finish(&prep, design, y, &sol, null_deviance),
            );
            start = Some(sol);
        }
        Ok(out)
    }

    fn null_deviance(&self, prep: &Prepared, design: &Design, y: &[f64]) -> Result<f64> {
        let null = self.solve(prep, design, y, f64::INFINITY, None)?;
        Ok(dev(self.family, y, &null.mu, design.weights()))
    }

    fn check(&self) -> Result<()> {
        self.family.validate()?;
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(Error::InvalidParameter {
                name: "alpha",
                value: self.alpha,
                reason: "must be between 0 and 1",
            });
        }
        if self.tolerance.is_nan() || self.tolerance <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "tolerance",
                value: self.tolerance,
                reason: "must be positive",
            });
        }
        Ok(())
    }

    /// Validates the data and standardizes the columns.
    fn prepare(&self, design: &Design, y: &[f64]) -> Result<Prepared> {
        self.check()?;
        let n = design.n_rows();
        let p = design.n_cols();
        if y.len() != n {
            return Err(Error::Data(format!(
                "{} responses for {n} design rows",
                y.len()
            )));
        }
        if let Some(bad) = y.iter().find(|&&v| !self.family.valid_y(v)) {
            return Err(Error::InvalidParameter {
                name: "y",
                value: *bad,
                reason: "is outside the family's range",
            });
        }
        let total: f64 = design.weights().iter().sum();
        let wn: Vec<f64> = design.weights().iter().map(|w| w / total).collect();
        let intercept = (0..p).find(|&j| design.column(j).iter().all(|&v| v == 1.0));
        let mut means = vec![0.0; p];
        let mut scales = vec![1.0; p];
        let mut columns = vec![Vec::new(); p];
        let mut constant = vec![false; p];
        for j in (0..p).filter(|&j| Some(j) != intercept) {
            let x = design.column(j);
            let m = if intercept.is_some() {
                x.iter().zip(&wn).map(|(a, w)| a * w).sum()
            } else {
                0.0
            };
            let spread = x
                .iter()
                .zip(&wn)
                .map(|(a, w)| w * (a - m).powi(2))
                .sum::<f64>()
                .sqrt();
            constant[j] = spread.is_nan() || spread <= 1e-12 * (m.abs() + 1.0);
            means[j] = m;
            if self.standardize && !constant[j] {
                scales[j] = spread;
            }
            columns[j] = x.iter().map(|a| (a - m) / scales[j]).collect();
        }
        let mut pf = match &self.penalty_factor {
            None => vec![1.0; p],
            Some(f) if f.len() == p => f.clone(),
            Some(f) => {
                return Err(Error::Data(format!(
                    "{} penalty factors for {p} design columns",
                    f.len()
                )));
            }
        };
        if let Some(bad) = pf.iter().find(|v| !(v.is_finite() && **v >= 0.0)) {
            return Err(Error::InvalidParameter {
                name: "penalty_factor",
                value: *bad,
                reason: "must be finite and non-negative",
            });
        }
        if let Some(i) = intercept {
            pf[i] = 0.0;
        }
        let free: Vec<usize> = (0..p)
            .filter(|&j| Some(j) != intercept && !constant[j])
            .collect();
        let sum: f64 = free.iter().map(|&j| pf[j]).sum();
        if sum > 0.0 {
            let scale = free.len() as f64 / sum;
            for &j in &free {
                pf[j] *= scale;
            }
        }
        Ok(Prepared {
            wn,
            intercept,
            means,
            scales,
            columns,
            constant,
            pf,
        })
    }

    /// The penalized objective at `(eta, mu, b)`.
    fn objective(&self, prep: &Prepared, y: &[f64], mu: &[f64], b: &[f64], lambda: f64) -> f64 {
        let d: f64 = (0..y.len())
            .map(|i| prep.wn[i] * self.family.unit_deviance(y[i], mu[i]))
            .sum::<f64>()
            / 2.0;
        // At λ = ∞ the penalized coefficients are held at zero.
        if lambda == 0.0 || lambda == f64::INFINITY {
            return d;
        }
        let pen: f64 = prep
            .penalized()
            .map(|j| {
                prep.pf[j] * ((1.0 - self.alpha) / 2.0 * b[j] * b[j] + self.alpha * b[j].abs())
            })
            .sum();
        d + lambda * pen
    }

    /// IRLS with a coordinate-descent inner loop, at one `λ`; `λ = ∞`
    /// fits only the intercept and unpenalized columns.
    fn solve(
        &self,
        prep: &Prepared,
        design: &Design,
        y: &[f64],
        lambda: f64,
        start: Option<&Solution>,
    ) -> Result<Solution> {
        let (family, link) = (self.family, self.link);
        let n = design.n_rows();
        let p = design.n_cols();
        let offset = design.offset();
        let (mut b0, mut b, mut eta) = match start {
            Some(s) => (s.b0, s.b.clone(), s.eta.clone()),
            None => {
                // Start at the family's initial means, as the intercept only.
                let total: f64 = (0..n).map(|i| prep.wn[i] * y[i]).sum();
                let mu0: Vec<f64> = (0..n)
                    .map(|i| family.initial_mu(y[i], design.weights()[i], total))
                    .collect();
                let eta: Vec<f64> = mu0.iter().map(|&m| link.link(m)).collect();
                let b0 = if prep.intercept.is_some() {
                    (0..n).map(|i| prep.wn[i] * (eta[i] - offset[i])).sum()
                } else {
                    0.0
                };
                let eta = (0..n).map(|i| offset[i] + b0).collect();
                (b0, vec![0.0; p], eta)
            }
        };
        let mut mu: Vec<f64> = eta.iter().map(|&e| link.inverse(e)).collect();
        if !mu.iter().all(|&m| family.valid_mu(m)) {
            return Err(Error::Data(
                "the starting means are outside the family's range".into(),
            ));
        }
        let mut current = self.objective(prep, y, &mu, &b, lambda);
        for iteration in 1..=self.max_iterations {
            // Working weights (normalized) and response.
            let mut v = vec![0.0; n];
            let mut z = vec![0.0; n];
            for i in 0..n {
                let d = link.mu_eta(eta[i]);
                v[i] = prep.wn[i] * d * d / family.variance(mu[i]);
                z[i] = eta[i] - offset[i] + (y[i] - mu[i]) / d;
                if !(v[i].is_finite() && z[i].is_finite()) {
                    return Err(Error::Data(format!(
                        "IRLS weights are not finite at row {i} (mean {})",
                        mu[i]
                    )));
                }
            }
            let (t0, t) = coordinate_descent(prep, &v, &z, b0, &b, lambda, self.alpha);
            // Step, halved while the objective rises or the mean is invalid.
            let mut step = 1.0;
            let (prev0, prev) = (b0, b.clone());
            loop {
                let c0 = prev0 + step * (t0 - prev0);
                let c: Vec<f64> = prev
                    .iter()
                    .zip(&t)
                    .map(|(p, q)| p + step * (q - p))
                    .collect();
                let trial_eta = prep.eta(offset, c0, &c);
                let trial_mu: Vec<f64> = trial_eta.iter().map(|&e| link.inverse(e)).collect();
                let ok = trial_eta.iter().all(|&e| link.valid_eta(e))
                    && trial_mu.iter().all(|&m| family.valid_mu(m));
                let obj = if ok {
                    self.objective(prep, y, &trial_mu, &c, lambda)
                } else {
                    f64::INFINITY
                };
                if obj.is_finite() && obj <= current + 1e-13 * current.abs() {
                    b0 = c0;
                    b = c;
                    eta = trial_eta;
                    mu = trial_mu;
                    break;
                }
                step *= 0.5;
                if step < 1e-10 {
                    // No descent left: at the minimum to rounding.
                    return Ok(Solution {
                        b0,
                        b,
                        eta,
                        mu,
                        iterations: iteration,
                    });
                }
            }
            let new = self.objective(prep, y, &mu, &b, lambda);
            let settled = (current - new).abs() / (new.abs() + 1e-10) < self.tolerance;
            let tol = self.tolerance.sqrt() * 1e-3;
            let still = (b0 - prev0).abs() <= tol * (b0.abs() + 1e-3)
                && b.iter()
                    .zip(&prev)
                    .all(|(x, y)| (x - y).abs() <= tol * (x.abs() + 1e-3));
            current = new;
            if settled && still {
                return Ok(Solution {
                    b0,
                    b,
                    eta,
                    mu,
                    iterations: iteration,
                });
            }
        }
        Err(Error::Data(format!(
            "the elastic net did not converge in {} iterations",
            self.max_iterations
        )))
    }

    /// The fit on the design's scale, from a solution on the standardized one.
    fn finish(
        &self,
        prep: &Prepared,
        design: &Design,
        y: &[f64],
        sol: &Solution,
        null_deviance: f64,
    ) -> ElasticNetFit {
        let p = design.n_cols();
        let n = design.n_rows();
        let mut coefficients = vec![0.0; p];
        let mut intercept = sol.b0;
        for j in prep.penalized_or_free() {
            let beta = sol.b[j] / prep.scales[j];
            coefficients[j] = beta;
            intercept -= beta * prep.means[j];
        }
        if let Some(i) = prep.intercept {
            coefficients[i] = intercept;
        }
        let deviance = dev(self.family, y, &sol.mu, design.weights());
        let df = prep
            .penalized_or_free()
            .filter(|&j| sol.b[j] != 0.0)
            .count();
        let used = df + usize::from(prep.intercept.is_some());
        let w = design.weights();
        let dispersion = if self.family.unit_dispersion() {
            1.0
        } else {
            let resid = n.saturating_sub(used).max(1) as f64;
            (0..n)
                .map(|i| w[i] * (y[i] - sol.mu[i]).powi(2) / self.family.variance(sol.mu[i]))
                .sum::<f64>()
                / resid
        };
        ElasticNetFit {
            spec: self.clone(),
            names: design.names().to_vec(),
            coefficients,
            deviance,
            null_deviance,
            df,
            dispersion,
            iterations: sol.iterations,
            fitted: sol.mu.clone(),
            input_hash: crate::input_hash(design, y),
        }
    }
}

impl ElasticNet {
    /// Cross-validates the path: on each split, fits `lambdas` (warm
    /// starts, as [`path`](Self::path)) to the training rows and scores
    /// each fit by the family's mean deviance on the test rows,
    /// `Σ w d / Σ w`. As glmnet's `cv.glmnet`, the score per `λ` is the
    /// mean over folds weighted by each test fold's total weight, with a
    /// standard error from the folds' weighted spread. Pass the `λ` values
    /// from [`lambda_path`](Self::lambda_path) on the full data, largest
    /// first. Folds run in parallel.
    ///
    /// ```
    /// use prospicio_glm::net::ElasticNet;
    /// use prospicio_models::resample::k_fold;
    /// use prospicio_models::{Design, Family, Link};
    ///
    /// let n = 60;
    /// let x: Vec<f64> = (0..n).map(|i| f64::from(i) / 10.0).collect();
    /// let z: Vec<f64> = (0..n).map(|i| (f64::from(i) * 1.7).sin()).collect();
    /// let y: Vec<f64> = (0..n).map(|i| 1.0 + x[i as usize] + (f64::from(i) * 2.3).cos()).collect();
    /// let d = Design::new(
    ///     vec!["(Intercept)".into(), "x".into(), "z".into()],
    ///     vec![vec![1.0; n as usize], x, z],
    /// )
    /// .unwrap();
    /// let net = ElasticNet::new(Family::Gaussian, Link::Identity, 1.0, 0.0);
    /// let lambdas = net.lambda_path(&d, &y, 30, 1e-3).unwrap();
    /// let cv = net.cross_validate(&d, &y, &lambdas, &k_fold(60, 5, 1).unwrap()).unwrap();
    /// // The one-standard-error choice is never smaller than the best.
    /// assert!(cv.lambda_1se() >= cv.lambda_min());
    /// ```
    pub fn cross_validate(
        &self,
        design: &Design,
        y: &[f64],
        lambdas: &[f64],
        splits: &[Split],
    ) -> Result<CvPath> {
        if lambdas.is_empty() || splits.len() < 2 {
            return Err(Error::Data(
                "cross-validation needs at least one lambda and two splits".into(),
            ));
        }
        if y.len() != design.n_rows() {
            return Err(Error::Data(format!(
                "{} responses for {} design rows",
                y.len(),
                design.n_rows()
            )));
        }
        let folds = map_splits(splits, |s| {
            let train = design.select(&s.train);
            let y_train: Vec<f64> = s.train.iter().map(|&i| y[i]).collect();
            let test = design.select(&s.test);
            let y_test: Vec<f64> = s.test.iter().map(|&i| y[i]).collect();
            let total: f64 = test.weights().iter().sum();
            let fits = self.path(&train, &y_train, lambdas)?;
            let scores = fits
                .iter()
                .map(|f| {
                    let mu = f.predict(&test)?;
                    Ok(dev(self.family, &y_test, &mu, test.weights()) / total)
                })
                .collect::<Result<Vec<f64>>>()?;
            Ok((scores, total))
        })?;
        let (fold_scores, fold_weights): (Vec<Vec<f64>>, Vec<f64>) = folds.into_iter().unzip();
        Ok(CvPath::new(lambdas.to_vec(), fold_scores, &fold_weights))
    }
}

/// Cross-validated scores along an elastic-net path, from
/// [`ElasticNet::cross_validate`].
#[derive(Debug, Clone, PartialEq)]
pub struct CvPath {
    /// The `λ` values, in the order given.
    pub lambdas: Vec<f64>,
    /// Weighted mean of the folds' mean deviance, per `λ`.
    pub mean: Vec<f64>,
    /// Its standard error, per `λ`.
    pub se: Vec<f64>,
    /// Each fold's mean deviance per `λ`: `fold_scores[fold][λ]`.
    pub fold_scores: Vec<Vec<f64>>,
    /// Index of the lowest mean.
    pub best: usize,
    /// Index of the largest `λ` whose mean is within one standard error of
    /// the lowest: a sparser model that is not detectably worse.
    pub one_se: usize,
}

impl CvPath {
    fn new(lambdas: Vec<f64>, fold_scores: Vec<Vec<f64>>, weights: &[f64]) -> Self {
        let k = fold_scores.len() as f64;
        let total: f64 = weights.iter().sum();
        let (mut mean, mut se) = (Vec::new(), Vec::new());
        for j in 0..lambdas.len() {
            let m = fold_scores
                .iter()
                .zip(weights)
                .map(|(f, w)| w * f[j])
                .sum::<f64>()
                / total;
            let v = fold_scores
                .iter()
                .zip(weights)
                .map(|(f, w)| w * (f[j] - m).powi(2))
                .sum::<f64>()
                / total;
            mean.push(m);
            se.push((v / (k - 1.0)).sqrt());
        }
        let best = (0..mean.len())
            .min_by(|&a, &b| mean[a].total_cmp(&mean[b]))
            .expect("at least one lambda");
        let bar = mean[best] + se[best];
        let one_se = (0..mean.len())
            .filter(|&j| mean[j] <= bar)
            .max_by(|&a, &b| lambdas[a].total_cmp(&lambdas[b]))
            .unwrap_or(best);
        Self {
            lambdas,
            mean,
            se,
            fold_scores,
            best,
            one_se,
        }
    }

    /// `λ` with the lowest cross-validated deviance.
    pub fn lambda_min(&self) -> f64 {
        self.lambdas[self.best]
    }

    /// The largest `λ` within one standard error of the lowest.
    pub fn lambda_1se(&self) -> f64 {
        self.lambdas[self.one_se]
    }
}

impl Model for ElasticNet {
    type Fitted = ElasticNetFit;

    /// Fits at `self.lambda` from the intercept-only start.
    fn fit(&self, design: &Design, y: &[f64]) -> Result<ElasticNetFit> {
        check_lambda(self.lambda)?;
        let prep = self.prepare(design, y)?;
        let null_deviance = self.null_deviance(&prep, design, y)?;
        let sol = self.solve(&prep, design, y, self.lambda, None)?;
        Ok(self.finish(&prep, design, y, &sol, null_deviance))
    }
}

fn check_lambda(lambda: f64) -> Result<()> {
    if !(lambda.is_finite() && lambda >= 0.0) {
        return Err(Error::InvalidParameter {
            name: "lambda",
            value: lambda,
            reason: "must be finite and non-negative",
        });
    }
    Ok(())
}

/// Weighted least squares with the elastic-net penalty, by cyclic
/// coordinate descent from `(b0, b)`: minimizes
/// `½ Σ vᵢ (zᵢ - b0 - Σ x̃ᵢⱼ bⱼ)² + λ Σ pfⱼ ((1-α)/2 bⱼ² + α |bⱼ|)`.
fn coordinate_descent(
    prep: &Prepared,
    v: &[f64],
    z: &[f64],
    b0: f64,
    b: &[f64],
    lambda: f64,
    alpha: f64,
) -> (f64, Vec<f64>) {
    let n = v.len();
    let mut b0 = b0;
    let mut b = b.to_vec();
    let fixed: Vec<usize> = prep.penalized_or_free().collect();
    let mut r: Vec<f64> = (0..n)
        .map(|i| {
            z[i] - b0
                - fixed
                    .iter()
                    .map(|&j| prep.columns[j][i] * b[j])
                    .sum::<f64>()
        })
        .collect();
    let curvature: Vec<f64> = (0..b.len())
        .map(|j| {
            if prep.columns[j].is_empty() {
                0.0
            } else {
                (0..n).map(|i| v[i] * prep.columns[j][i].powi(2)).sum()
            }
        })
        .collect();
    let v_sum: f64 = v.iter().sum();
    let scale: f64 = (0..n).map(|i| v[i] * z[i] * z[i]).sum::<f64>().max(1e-300);
    for _ in 0..100_000 {
        let mut biggest: f64 = 0.0;
        if prep.intercept.is_some() {
            let delta = (0..n).map(|i| v[i] * r[i]).sum::<f64>() / v_sum;
            if delta != 0.0 {
                b0 += delta;
                r.iter_mut().for_each(|ri| *ri -= delta);
                biggest = biggest.max(v_sum * delta * delta);
            }
        }
        for &j in &fixed {
            let x = &prep.columns[j];
            let u = (0..n).map(|i| v[i] * x[i] * r[i]).sum::<f64>() + curvature[j] * b[j];
            let new = if lambda == f64::INFINITY {
                if prep.pf[j] > 0.0 {
                    0.0
                } else {
                    u / curvature[j]
                }
            } else {
                let pen = lambda * prep.pf[j];
                soft(u, pen * alpha) / (curvature[j] + pen * (1.0 - alpha))
            };
            let delta = new - b[j];
            if delta != 0.0 {
                for i in 0..n {
                    r[i] -= delta * x[i];
                }
                b[j] = new;
                biggest = biggest.max(curvature[j] * delta * delta);
            }
        }
        if biggest <= 1e-30 * scale {
            break;
        }
    }
    (b0, b)
}

fn soft(u: f64, t: f64) -> f64 {
    if u > t {
        u - t
    } else if u < -t {
        u + t
    } else {
        0.0
    }
}

/// Standardized columns and penalty factors.
struct Prepared {
    /// Weights normalized to sum to 1.
    wn: Vec<f64>,
    intercept: Option<usize>,
    means: Vec<f64>,
    scales: Vec<f64>,
    /// `(x - mean) / scale`; empty for the intercept.
    columns: Vec<Vec<f64>>,
    /// Columns with no spread: their coefficient stays 0.
    constant: Vec<bool>,
    pf: Vec<f64>,
}

impl Prepared {
    /// Non-intercept, non-constant columns.
    fn penalized_or_free(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.columns.len()).filter(|&j| Some(j) != self.intercept && !self.constant[j])
    }

    /// Columns with a positive penalty factor.
    fn penalized(&self) -> impl Iterator<Item = usize> + '_ {
        self.penalized_or_free().filter(|&j| self.pf[j] > 0.0)
    }

    fn eta(&self, offset: &[f64], b0: f64, b: &[f64]) -> Vec<f64> {
        let mut eta: Vec<f64> = offset.iter().map(|o| o + b0).collect();
        for j in self.penalized_or_free() {
            if b[j] != 0.0 {
                for (e, x) in eta.iter_mut().zip(&self.columns[j]) {
                    *e += b[j] * x;
                }
            }
        }
        eta
    }

    /// `∂/∂bⱼ` of `Σ wᵢ dᵢ / (2 Σ w)` at `η`, by column (0 where unused).
    fn gradient(&self, spec: &ElasticNet, design: &Design, y: &[f64], eta: &[f64]) -> Vec<f64> {
        let (family, link) = (spec.family, spec.link);
        let score: Vec<f64> = (0..y.len())
            .map(|i| {
                let mu = link.inverse(eta[i]);
                self.wn[i] * (y[i] - mu) * link.mu_eta(eta[i]) / family.variance(mu)
            })
            .collect();
        (0..design.n_cols())
            .map(|j| {
                if self.columns[j].is_empty() || self.constant[j] {
                    0.0
                } else {
                    -self.columns[j]
                        .iter()
                        .zip(&score)
                        .map(|(x, s)| x * s)
                        .sum::<f64>()
                }
            })
            .collect()
    }
}

/// A solution on the standardized scale.
struct Solution {
    b0: f64,
    b: Vec<f64>,
    eta: Vec<f64>,
    mu: Vec<f64>,
    iterations: usize,
}

/// A fitted elastic net at one `λ`.
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticNetFit {
    pub(crate) spec: ElasticNet,
    pub(crate) names: Vec<String>,
    pub(crate) coefficients: Vec<f64>,
    pub(crate) deviance: f64,
    pub(crate) null_deviance: f64,
    pub(crate) df: usize,
    pub(crate) dispersion: f64,
    pub(crate) iterations: usize,
    pub(crate) fitted: Vec<f64>,
    pub(crate) input_hash: String,
}

impl ElasticNetFit {
    /// Hash of the training data, as [`GlmFit::input_hash`](crate::GlmFit::input_hash).
    pub fn input_hash(&self) -> &str {
        &self.input_hash
    }

    /// The specification that was fitted, including its `λ`.
    pub fn spec(&self) -> &ElasticNet {
        &self.spec
    }

    /// `λ`.
    pub fn lambda(&self) -> f64 {
        self.spec.lambda
    }

    /// Column names, in coefficient order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Coefficients on the design's scale; exact zeros for columns the
    /// penalty dropped.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// `Σ w d(y, μ)`.
    pub fn deviance(&self) -> f64 {
        self.deviance
    }

    /// Deviance with only the intercept and unpenalized columns.
    pub fn null_deviance(&self) -> f64 {
        self.null_deviance
    }

    /// Share of the null deviance explained (glmnet's `dev.ratio`).
    pub fn deviance_ratio(&self) -> f64 {
        1.0 - self.deviance / self.null_deviance
    }

    /// Number of non-zero coefficients, intercept excluded.
    pub fn df(&self) -> usize {
        self.df
    }

    /// 1 for the Poisson, binomial and negative binomial; otherwise
    /// Pearson's statistic over `n -` (non-zero coefficients + intercept).
    pub fn dispersion(&self) -> f64 {
        self.dispersion
    }

    /// IRLS iterations used.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Fitted means on the training rows.
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    fn check_design(&self, design: &Design) -> Result<()> {
        if design.names() != self.names.as_slice() {
            return Err(Error::Data(format!(
                "design columns {:?} do not match the fitted {:?}",
                design.names(),
                self.names
            )));
        }
        Ok(())
    }
}

impl Fitted for ElasticNetFit {
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        self.check_design(design)?;
        Ok(design
            .linear_predictor(&self.coefficients)
            .into_iter()
            .map(|e| self.spec.link.inverse(e))
            .collect())
    }

    /// Joint draws across the rows of `design`, components keyed
    /// `row = 0, 1, …`: process uncertainty only. Penalized estimates have
    /// no standard errors, so the coefficients are held at their values;
    /// for parameter uncertainty, bootstrap the fit.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        self.check_design(design)?;
        let provenance = Provenance::new("elastic_net")
            .version("prospicio-glm", env!("CARGO_PKG_VERSION"))
            .param("family", self.spec.family.name())
            .param("link", format!("{:?}", self.spec.link))
            .param("alpha", self.spec.alpha)
            .param("lambda", self.spec.lambda)
            .param("dispersion", self.dispersion);
        let glm = Glm::new(self.spec.family, self.spec.link)
            .dispersion(Dispersion::Fixed(self.dispersion));
        simulate_responses(
            &glm,
            self.dispersion,
            &self.coefficients,
            None,
            None,
            design,
            n_sims,
            seed,
            provenance,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design() -> (Design, Vec<f64>) {
        let n = 40u32;
        let x1: Vec<f64> = (0..n).map(|i| f64::from(i) / 4.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| (f64::from(i) * 7.3).sin()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let k = i as usize;
                2.0 + 0.8 * x1[k] + 0.3 * x2[k] + (f64::from(i) * 3.1).cos()
            })
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x1".into(), "x2".into()],
            vec![vec![1.0; n as usize], x1, x2],
        )
        .unwrap();
        (d, y)
    }

    /// At `λ = 0` the elastic net is the unpenalized GLM.
    #[test]
    fn lambda_zero_is_the_glm() {
        let (d, y) = design();
        let net = ElasticNet::new(Family::Gaussian, Link::Identity, 0.5, 0.0)
            .fit(&d, &y)
            .unwrap();
        let glm = Glm::new(Family::Gaussian, Link::Identity)
            .fit(&d, &y)
            .unwrap();
        for (a, b) in net.coefficients().iter().zip(glm.coefficients()) {
            assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }

    /// The KKT conditions hold for the lasso: zero coefficients have
    /// gradients within `λ`, non-zero ones exactly `-λ sign(b)`.
    #[test]
    fn lasso_kkt() {
        let (d, y) = design();
        let spec = ElasticNet::new(Family::Gaussian, Link::Identity, 1.0, 0.0);
        let lmax = spec.lambda_max(&d, &y).unwrap();
        for ratio in [0.9, 0.3, 0.05] {
            let lambda = lmax * ratio;
            let fit = spec.with_lambda(lambda).fit(&d, &y).unwrap();
            let prep = spec.prepare(&d, &y).unwrap();
            let eta = d.linear_predictor(fit.coefficients());
            let grad = prep.gradient(&spec, &d, &y, &eta);
            for (j, (&b, g)) in fit.coefficients().iter().zip(&grad).enumerate().skip(1) {
                if b == 0.0 {
                    assert!(g.abs() <= lambda * (1.0 + 1e-9));
                } else {
                    assert!((g + lambda * b.signum()).abs() < 1e-8 * lambda, "{j}");
                }
            }
        }
    }

    /// Warm-started paths give the cold-start fits.
    #[test]
    fn path_matches_single_fits() {
        let (d, y) = design();
        let spec = ElasticNet::new(Family::Gaussian, Link::Identity, 0.5, 0.0);
        let lambdas = spec.lambda_path(&d, &y, 8, 1e-3).unwrap();
        let path = spec.path(&d, &y, &lambdas).unwrap();
        for fit in &path {
            let cold = spec.with_lambda(fit.lambda()).fit(&d, &y).unwrap();
            for (a, b) in fit.coefficients().iter().zip(cold.coefficients()) {
                assert!((a - b).abs() < 1e-8 * (1.0 + b.abs()));
            }
        }
        // At λ_max itself the penalized coefficients are zero to rounding.
        assert!(path[0].coefficients()[1..].iter().all(|&b| b.abs() < 1e-10));
    }

    #[test]
    fn rejects_bad_parameters() {
        let (d, y) = design();
        assert!(
            ElasticNet::new(Family::Gaussian, Link::Identity, 1.5, 1.0)
                .fit(&d, &y)
                .is_err()
        );
        assert!(
            ElasticNet::new(Family::Gaussian, Link::Identity, 1.0, -1.0)
                .fit(&d, &y)
                .is_err()
        );
        let bad =
            ElasticNet::new(Family::Gaussian, Link::Identity, 1.0, 1.0).penalty_factor(vec![1.0]);
        assert!(bad.fit(&d, &y).is_err());
    }
}
