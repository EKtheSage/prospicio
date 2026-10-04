//! Generalized linear models by iteratively reweighted least squares, on
//! the families and links of `act-models` (`docs/design/models.md`).
//!
//! [`Glm`] is a spec (family, link, how to estimate the dispersion);
//! [`Glm::fit`](act_models::Model::fit) returns a [`GlmFit`] with
//! coefficients, standard errors, deviance, log-likelihood and AIC, and
//! implements [`Fitted`]: predictions and joint predictive distributions
//! with parameter and process uncertainty, and sandwich covariances
//! ([`GlmFit::robust_covariance`]). [`gam::Gam`] adds penalized B-spline
//! smooths, with smoothing chosen by GCV or UBRE.

pub mod gam;
pub mod net;
mod robust;
pub mod tweedie;

pub use robust::Robust;

use act_core::{Error, Result};
use act_math::linalg::{cholesky, cholesky_inverse, cholesky_solve, lower_mul};
use act_math::special::{norm_quantile, student_t_cdf};
use act_models::{Design, Family, Fitted, Link, Model};
use act_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};

/// How a GLM's dispersion `φ` is set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dispersion {
    /// A fixed value, 1 for the Poisson, binomial and negative binomial.
    Fixed(f64),
    /// Pearson's estimate `Σ w (y - μ)² / V(μ) / (n - p)`: the default for
    /// families with a free dispersion, and the over-dispersed Poisson.
    Pearson,
    /// `deviance / (n - p)`.
    Deviance,
}

/// A GLM specification.
///
/// ```
/// use act_glm::Glm;
/// use act_models::{Design, Family, Link, Model};
///
/// // A Poisson frequency model with log exposure as offset.
/// let x = vec![0.0, 0.0, 1.0, 1.0];
/// let design = Design::new(
///     vec!["(Intercept)".into(), "young".into()],
///     vec![vec![1.0; 4], x],
/// )
/// .unwrap()
/// .with_offset(vec![0.0, 1f64.ln(), 2f64.ln(), 0.5f64.ln()])
/// .unwrap();
/// let fit = Glm::new(Family::Poisson, Link::Log)
///     .fit(&design, &[1.0, 2.0, 6.0, 1.0])
///     .unwrap();
/// // Old drivers: 3 claims over 2 years; young: 7 over 2.5.
/// assert!((fit.coefficients()[0] - 1.5f64.ln()).abs() < 1e-10);
/// assert!((fit.coefficients()[1] - (2.8f64 / 1.5).ln()).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glm {
    pub family: Family,
    pub link: Link,
    pub dispersion: Dispersion,
    /// IRLS stops when the relative change in deviance is below this and
    /// each coefficient has moved by less than `1e-3 √tolerance` relative.
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Glm {
    /// A GLM with the given family and link: dispersion fixed at 1 for
    /// the Poisson, binomial and negative binomial and Pearson's estimate
    /// otherwise; tolerance `1e-10`; at most 100 iterations.
    pub fn new(family: Family, link: Link) -> Self {
        Self {
            family,
            link,
            dispersion: if family.unit_dispersion() {
                Dispersion::Fixed(1.0)
            } else {
                Dispersion::Pearson
            },
            tolerance: 1e-10,
            max_iterations: 100,
        }
    }

    /// The same GLM with dispersion set by `dispersion`.
    pub fn dispersion(mut self, dispersion: Dispersion) -> Self {
        self.dispersion = dispersion;
        self
    }

    /// The over-dispersed Poisson (quasi-Poisson) with log link: Poisson
    /// estimates and Pearson's dispersion. Its fitted values on a triangle
    /// with origin and development factors are the Chain Ladder's.
    pub fn over_dispersed_poisson() -> Self {
        Self::new(Family::Poisson, Link::Log).dispersion(Dispersion::Pearson)
    }
}

/// A fitted GLM.
#[derive(Debug, Clone, PartialEq)]
pub struct GlmFit {
    spec: Glm,
    names: Vec<String>,
    coefficients: Vec<f64>,
    /// `(Xᵀ W X)⁻¹`, row-major, before scaling by the dispersion.
    unscaled_covariance: Vec<f64>,
    dispersion: f64,
    deviance: f64,
    null_deviance: f64,
    log_likelihood: f64,
    n_obs: usize,
    iterations: usize,
    fitted: Vec<f64>,
}

impl Model for Glm {
    type Fitted = GlmFit;

    /// Fits by IRLS. Starts from the family's initial means, solves the
    /// weighted least-squares system by Cholesky each step, halves a step
    /// that raises the deviance or leaves the family's range, and stops
    /// when the relative change in deviance is below the tolerance (R's
    /// rule). Fails if the design is collinear (not positive definite), a
    /// response is outside the family's range, or IRLS does not converge.
    fn fit(&self, design: &Design, y: &[f64]) -> Result<GlmFit> {
        self.fit_from(design, y, None)
    }
}

impl Glm {
    /// [`fit`](Model::fit), with IRLS starting from the means `start` (one
    /// per row) when given: a warm start from a nearby model's fit.
    pub(crate) fn fit_from(
        &self,
        design: &Design,
        y: &[f64],
        start: Option<&[f64]>,
    ) -> Result<GlmFit> {
        self.family.validate()?;
        let n = design.n_rows();
        let p = design.n_cols();
        if y.len() != n {
            return Err(Error::Data(format!(
                "{} responses for {n} design rows",
                y.len()
            )));
        }
        if n <= p {
            return Err(Error::Data(format!(
                "{n} observations for {p} coefficients"
            )));
        }
        if let Some(bad) = y.iter().find(|&&v| !self.family.valid_y(v)) {
            return Err(invalid("y", *bad, "is outside the family's range"));
        }
        let w = design.weights();
        let offset = design.offset();
        let y_mean = y.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / w.iter().sum::<f64>();

        let Irls {
            beta,
            information,
            iterations,
            mu,
        } = irls(
            self,
            design,
            y,
            |i| match start {
                Some(mu) if self.family.valid_mu(mu[i]) => mu[i],
                _ => self.family.initial_mu(y[i], w[i], y_mean),
            },
            None,
        )?;
        let l = cholesky(&information, p).ok_or_else(|| {
            Error::Data("the design matrix is collinear: drop or merge columns".into())
        })?;
        let deviance = dev(self.family, y, &mu, w);

        // Null model: an intercept (if the design has one) and the offset.
        let has_intercept = (0..p).any(|j| design.column(j).iter().all(|&v| v == 1.0));
        let null_deviance = if has_intercept {
            let ones = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; n]])?
                .with_offset(offset.to_vec())?
                .with_weights(w.to_vec())?;
            let null = irls(
                self,
                &ones,
                y,
                |i| self.family.initial_mu(y[i], w[i], y_mean),
                None,
            )?;
            dev(self.family, y, &null.mu, w)
        } else {
            let mu0: Vec<f64> = offset.iter().map(|&o| self.link.inverse(o)).collect();
            dev(self.family, y, &mu0, w)
        };

        let df_resid = (n - p) as f64;
        let dispersion = match self.dispersion {
            Dispersion::Fixed(phi) => phi,
            Dispersion::Deviance => deviance / df_resid,
            Dispersion::Pearson => {
                (0..n)
                    .map(|i| w[i] * (y[i] - mu[i]).powi(2) / self.family.variance(mu[i]))
                    .sum::<f64>()
                    / df_resid
            }
        };
        // The Gaussian's log-likelihood uses the maximum-likelihood variance
        // `deviance / n`, as statsmodels and R report it.
        let ll_dispersion = match self.family {
            Family::Gaussian => deviance / n as f64,
            _ => dispersion,
        };
        let log_likelihood = (0..n)
            .map(|i| self.family.log_likelihood(y[i], mu[i], w[i], ll_dispersion))
            .sum();
        Ok(GlmFit {
            spec: *self,
            names: design.names().to_vec(),
            coefficients: beta,
            unscaled_covariance: cholesky_inverse(&l, p),
            dispersion,
            deviance,
            null_deviance,
            log_likelihood,
            n_obs: n,
            iterations,
            fitted: mu,
        })
    }
}

/// What IRLS returns: coefficients, `Xᵀ W X` (unpenalized) at the
/// solution, iterations used and fitted means.
pub(crate) struct Irls {
    pub(crate) beta: Vec<f64>,
    pub(crate) information: Vec<f64>,
    pub(crate) iterations: usize,
    pub(crate) mu: Vec<f64>,
}

/// `Xᵀ W X` and `Xᵀ W z` for the working weights and response at `η`.
fn working_system(
    spec: &Glm,
    design: &Design,
    y: &[f64],
    eta: &[f64],
    mu: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let (family, link) = (spec.family, spec.link);
    let p = design.n_cols();
    let w = design.weights();
    let offset = design.offset();
    let mut xtwx = vec![0.0; p * p];
    let mut xtwz = vec![0.0; p];
    for i in 0..design.n_rows() {
        let d = link.mu_eta(eta[i]);
        let z = eta[i] - offset[i] + (y[i] - mu[i]) / d;
        let wi = w[i] * d * d / family.variance(mu[i]);
        if !(wi.is_finite() && z.is_finite()) {
            return Err(Error::Data(format!(
                "IRLS weights are not finite at row {i} (mean {})",
                mu[i]
            )));
        }
        for a in 0..p {
            let xa = design.column(a)[i] * wi;
            xtwz[a] += xa * z;
            for b in 0..=a {
                xtwx[a * p + b] += xa * design.column(b)[i];
            }
        }
    }
    for a in 0..p {
        for b in a + 1..p {
            xtwx[a * p + b] = xtwx[b * p + a];
        }
    }
    Ok((xtwx, xtwz))
}

/// `βᵀ S β` for a row-major `S`.
fn quadratic(s: &[f64], beta: &[f64]) -> f64 {
    let p = beta.len();
    (0..p)
        .map(|a| beta[a] * (0..p).map(|b| s[a * p + b] * beta[b]).sum::<f64>())
        .sum()
}

/// (Penalized) IRLS from starting means `mu0(i)`: minimizes the deviance
/// plus `βᵀ S β` when a `penalty` `S` (row-major `p × p`) is given.
pub(crate) fn irls(
    spec: &Glm,
    design: &Design,
    y: &[f64],
    mu0: impl Fn(usize) -> f64,
    penalty: Option<&[f64]>,
) -> Result<Irls> {
    let (family, link) = (spec.family, spec.link);
    let n = design.n_rows();
    let p = design.n_cols();
    let w = design.weights();
    let objective = |mu: &[f64], beta: &[f64]| {
        dev(family, y, mu, w) + penalty.map_or(0.0, |s| quadratic(s, beta))
    };
    let mut eta: Vec<f64> = (0..n).map(|i| link.link(mu0(i))).collect();
    let mut mu: Vec<f64> = eta.iter().map(|&e| link.inverse(e)).collect();
    let mut current = f64::INFINITY;
    let mut beta = vec![0.0; p];
    for iteration in 1..=spec.max_iterations {
        let (mut xtwx, xtwz) = working_system(spec, design, y, &eta, &mu)?;
        if let Some(s) = penalty {
            for (a, b) in xtwx.iter_mut().zip(s) {
                *a += b;
            }
        }
        let l = cholesky(&xtwx, p).ok_or_else(|| {
            Error::Data("the design matrix is collinear: drop or merge columns".into())
        })?;
        let target = cholesky_solve(&l, &xtwz);
        // Step, halved while the objective rises or the mean is invalid.
        let mut step = 1.0;
        let previous = beta.clone();
        loop {
            let trial: Vec<f64> = previous
                .iter()
                .zip(&target)
                .map(|(b, t)| b + step * (t - b))
                .collect();
            let trial_eta = design.linear_predictor(&trial);
            let trial_mu: Vec<f64> = trial_eta.iter().map(|&e| link.inverse(e)).collect();
            let ok = trial_eta.iter().all(|&e| link.valid_eta(e))
                && trial_mu.iter().all(|&m| family.valid_mu(m));
            let trial_obj = if ok {
                objective(&trial_mu, &trial)
            } else {
                f64::INFINITY
            };
            // The first iteration has nothing to compare with.
            if trial_obj.is_finite() && (trial_obj <= current * (1.0 + 1e-12) || iteration == 1) {
                beta = trial;
                eta = trial_eta;
                mu = trial_mu;
                break;
            }
            step *= 0.5;
            if step < 1e-10 {
                return Err(Error::Data(
                    "IRLS could not reduce the deviance: check the link and the data".into(),
                ));
            }
        }
        let new_obj = objective(&mu, &beta);
        // Converged when the objective has settled and so have the
        // coefficients: the deviance alone pins β only to about the square
        // root of its tolerance.
        let settled = (current - new_obj).abs() / (new_obj.abs() + 0.1) < spec.tolerance;
        let still = beta
            .iter()
            .zip(&previous)
            .all(|(b, p)| (b - p).abs() <= spec.tolerance.sqrt() * 1e-3 * (b.abs() + 1e-3));
        current = new_obj;
        if settled && still {
            // Information matrix at the solution, for standard errors.
            let (information, _) = working_system(spec, design, y, &eta, &mu)?;
            return Ok(Irls {
                beta,
                information,
                iterations: iteration,
                mu,
            });
        }
    }
    Err(Error::Data(format!(
        "IRLS did not converge in {} iterations",
        spec.max_iterations
    )))
}

pub(crate) fn dev(family: Family, y: &[f64], mu: &[f64], w: &[f64]) -> f64 {
    (0..y.len())
        .map(|i| w[i] * family.unit_deviance(y[i], mu[i]))
        .sum()
}

impl GlmFit {
    /// The specification that was fitted.
    pub fn spec(&self) -> &Glm {
        &self.spec
    }

    /// Coefficient names, as the design's columns.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Estimated coefficients `β`.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Covariance of `β`: `φ (Xᵀ W X)⁻¹`, row-major.
    pub fn covariance(&self) -> Vec<f64> {
        self.unscaled_covariance
            .iter()
            .map(|c| c * self.dispersion)
            .collect()
    }

    /// Standard errors of `β`.
    pub fn std_errors(&self) -> Vec<f64> {
        let p = self.coefficients.len();
        (0..p)
            .map(|j| (self.dispersion * self.unscaled_covariance[j * p + j]).sqrt())
            .collect()
    }

    /// Two-sided p-values of `β = 0`: from the normal when the dispersion
    /// is fixed, from Student's t with `n - p` degrees of freedom when it
    /// is estimated (as R reports).
    pub fn p_values(&self) -> Vec<f64> {
        let df = self.df_resid();
        let fixed = matches!(self.spec.dispersion, Dispersion::Fixed(_));
        self.coefficients
            .iter()
            .zip(self.std_errors())
            .map(|(b, se)| {
                let t = (b / se).abs();
                if fixed {
                    2.0 * act_math::special::norm_cdf(-t)
                } else {
                    2.0 * student_t_cdf(-t, df)
                }
            })
            .collect()
    }

    /// Dispersion `φ`.
    pub fn dispersion(&self) -> f64 {
        self.dispersion
    }

    /// Residual deviance.
    pub fn deviance(&self) -> f64 {
        self.deviance
    }

    /// Deviance of the null model: the intercept (if the design has one)
    /// and the offset.
    pub fn null_deviance(&self) -> f64 {
        self.null_deviance
    }

    /// Log-likelihood at the estimates and dispersion (for the Gaussian,
    /// at the maximum-likelihood variance `deviance / n`).
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// `-2 ℓ + 2 p`, with `p` the number of coefficients (statsmodels'
    /// convention, which does not count an estimated dispersion).
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * self.coefficients.len() as f64
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Residual degrees of freedom, `n - p`.
    pub fn df_resid(&self) -> f64 {
        (self.n_obs - self.coefficients.len()) as f64
    }

    /// IRLS iterations used.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Fitted means on the training data.
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

impl Fitted for GlmFit {
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        self.check_design(design)?;
        Ok(design
            .linear_predictor(&self.coefficients)
            .into_iter()
            .map(|e| self.spec.link.inverse(e))
            .collect())
    }

    /// Joint draws across the rows of `design`, components keyed
    /// `row = 0, 1, …`. Simulation `i` uses stream `i` of `seed`: it draws
    /// `β` from its normal approximation `N(β̂, φ (XᵀWX)⁻¹)` (parameter
    /// uncertainty, shared by every row, which makes the rows dependent),
    /// then each row's response from the family with that row's mean,
    /// the dispersion and the row's weight (process uncertainty), in row
    /// order.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        self.check_design(design)?;
        let provenance = Provenance::new("glm")
            .version("act-glm", env!("CARGO_PKG_VERSION"))
            .param("family", self.spec.family.name())
            .param("link", format!("{:?}", self.spec.link))
            .param("dispersion", self.dispersion);
        simulate_responses(
            &self.spec,
            self.dispersion,
            &self.coefficients,
            Some(&self.covariance()),
            design,
            n_sims,
            seed,
            provenance,
        )
    }
}

/// Joint draws of the responses for the rows of `design`: simulation `i`
/// (stream `i` of `seed`) draws `β ~ N(coefficients, covariance)`, shared
/// by every row (or keeps `β` fixed without a covariance), then each row's
/// response from the family, in row order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_responses(
    spec: &Glm,
    dispersion: f64,
    coefficients: &[f64],
    covariance: Option<&[f64]>,
    design: &Design,
    n_sims: usize,
    seed: u64,
    provenance: Provenance,
) -> Result<PredictiveDistribution> {
    let p = coefficients.len();
    let factor = covariance
        .map(|c| {
            cholesky(c, p)
                .ok_or_else(|| Error::Data("the coefficient covariance is singular".into()))
        })
        .transpose()?;
    let n = design.n_rows();
    let components: Vec<ComponentKey> = (0..n).map(|i| vec![KeyValue::from(i as i64)]).collect();
    let (family, link, phi) = (spec.family, spec.link, dispersion);
    let weights = design.weights();
    PredictiveDistribution::simulate(
        vec!["row".into()],
        components,
        n_sims,
        seed,
        provenance,
        |rng, row| {
            let mut shift = vec![0.0; p];
            if let Some(factor) = &factor {
                let z: Vec<f64> = (0..p).map(|_| norm_quantile(rng.next_open01())).collect();
                lower_mul(factor, &z, &mut shift);
            }
            let beta: Vec<f64> = coefficients
                .iter()
                .zip(&shift)
                .map(|(b, s)| b + s)
                .collect();
            let eta = design.linear_predictor(&beta);
            for (i, out) in row.iter_mut().enumerate() {
                let mu = link.inverse(eta[i]);
                *out = if family.valid_mu(mu) {
                    family
                        .draw(mu, phi, weights[i], rng.next_open01())
                        .unwrap_or(f64::NAN)
                } else {
                    f64::NAN
                };
            }
        },
    )
}

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use act_prob::Distribution;

    fn design(x: &[f64]) -> Design {
        Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; x.len()], x.to_vec()],
        )
        .unwrap()
    }

    #[test]
    fn gaussian_identity_is_least_squares() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [1.1, 2.9, 5.2, 7.1, 8.8];
        let fit = Glm::new(Family::Gaussian, Link::Identity)
            .fit(&design(&x), &y)
            .unwrap();
        // Closed-form slope and intercept.
        let (mx, my) = (2.0, y.iter().sum::<f64>() / 5.0);
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let slope = sxy / 10.0;
        assert!((fit.coefficients()[1] - slope).abs() < 1e-12);
        assert!((fit.coefficients()[0] - (my - slope * mx)).abs() < 1e-12);
        // σ² = RSS / (n - p) and SE(slope) = σ / sqrt(Sxx).
        assert!((fit.std_errors()[1] - (fit.dispersion() / 10.0).sqrt()).abs() < 1e-12);
        assert!((fit.deviance() - fit.dispersion() * 3.0).abs() < 1e-12);
    }

    #[test]
    fn over_dispersed_poisson_reproduces_the_chain_ladder() {
        // Incremental triangle, 4 origins × 4 developments.
        let tri = [
            [100.0, 60.0, 20.0, 5.0],
            [110.0, 70.0, 25.0, f64::NAN],
            [120.0, 65.0, f64::NAN, f64::NAN],
            [130.0, f64::NAN, f64::NAN, f64::NAN],
        ];
        let cells: Vec<(usize, usize)> = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .filter(|&(i, j)| i + j < 4)
            .collect();
        let column = |f: &dyn Fn(usize, usize) -> f64, cells: &[(usize, usize)]| {
            cells.iter().map(|&(i, j)| f(i, j)).collect::<Vec<f64>>()
        };
        let build = |cells: &[(usize, usize)]| {
            let mut names = vec!["(Intercept)".to_string()];
            let mut cols = vec![vec![1.0; cells.len()]];
            for k in 1..4 {
                names.push(format!("origin[{k}]"));
                cols.push(column(&|i, _| f64::from(u8::from(i == k)), cells));
            }
            for k in 1..4 {
                names.push(format!("dev[{k}]"));
                cols.push(column(&|_, j| f64::from(u8::from(j == k)), cells));
            }
            Design::new(names, cols).unwrap()
        };
        let y: Vec<f64> = cells.iter().map(|&(i, j)| tri[i][j]).collect();
        let fit = Glm::over_dispersed_poisson()
            .fit(&build(&cells), &y)
            .unwrap();
        // Chain Ladder: volume-weighted development factors on cumulatives.
        let mut cum = [[f64::NAN; 4]; 4];
        for i in 0..4 {
            let mut c = 0.0;
            for j in 0..4 - i {
                c += tri[i][j];
                cum[i][j] = c;
            }
        }
        let f: Vec<f64> = (0..3)
            .map(|j| {
                let rows = 0..3 - j;
                rows.clone().map(|i| cum[i][j + 1]).sum::<f64>()
                    / rows.map(|i| cum[i][j]).sum::<f64>()
            })
            .collect();
        let future: Vec<(usize, usize)> = (1..4)
            .flat_map(|i| (4 - i..4).map(move |j| (i, j)))
            .collect();
        let pred = fit.predict(&build(&future)).unwrap();
        for (&(i, j), p) in future.iter().zip(&pred) {
            // Chain Ladder incremental: latest × (Π f up to j) - (Π f up to j - 1).
            let latest = cum[i][3 - i];
            let up_to = |k: usize| (3 - i..k).map(|m| f[m]).product::<f64>();
            let want = latest * (up_to(j) - up_to(j - 1));
            assert!((p - want).abs() < 1e-8 * want, "({i},{j}): {p} vs {want}");
        }
    }

    #[test]
    fn collinear_designs_and_bad_responses_fail() {
        let d = Design::new(
            vec!["a".into(), "b".into()],
            vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]],
        )
        .unwrap();
        assert!(
            Glm::new(Family::Poisson, Link::Log)
                .fit(&d, &[1.0, 2.0, 3.0])
                .is_err()
        );
        assert!(
            Glm::new(Family::Gamma, Link::Log)
                .fit(&design(&[0.0, 1.0, 2.0]), &[1.0, 0.0, 2.0])
                .is_err()
        );
    }

    #[test]
    fn predictive_distribution_has_the_fitted_mean() {
        let x: Vec<f64> = (0..40).map(|i| f64::from(i % 4)).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| (2.0 + v + (i % 3) as f64) * 10.0)
            .collect();
        let d = design(&x);
        let fit = Glm::new(Family::Gamma, Link::Log).fit(&d, &y).unwrap();
        let new = design(&[0.0, 3.0]);
        let mean = fit.predict(&new).unwrap();
        let pd = fit.predict_distribution(&new, 20_000, 3).unwrap();
        for (j, m) in mean.iter().enumerate() {
            let marg = pd.marginal(&vec![KeyValue::from(j as i64)]).unwrap();
            // Parameter draws add a little upward bias under the log link
            // (E[e^Z] > 1): a few percent here.
            assert!(
                (marg.mean() / m - 1.0).abs() < 0.03,
                "{j}: {} vs {m}",
                marg.mean()
            );
        }
        // Same seed, same draws.
        let again = fit.predict_distribution(&new, 20_000, 3).unwrap();
        assert_eq!(pd.draw_matrix(), again.draw_matrix());
    }
}
