//! Generalized linear mixed models with random intercepts: the Poisson,
//! gamma and Tweedie with the log link, one random intercept per level of
//! each grouping factor, the factors one, nested or crossed
//! (`docs/design/models.md`, "Mixed models").
//!
//! The model for row `i` is `log μᵢ = xᵢᵀ β + oᵢ + Σₖ u_{k, gₖ(i)}` with
//! `u_{k, l} ~ N(0, σₖ²)` independent, and `yᵢ` from the family with mean
//! `μᵢ`, variance `φ V(μᵢ) / wᵢ` and `V(μ) = μᵖ` (`p = 1` for the Poisson,
//! whose `φ` is 1, `p = 2` for the gamma). [`Glmm::fit`] maximizes the
//! marginal likelihood of `β`, the `σₖ` and `φ`, with the random effects
//! integrated out:
//!
//! - by the Laplace approximation (the default, as R's `glmer` and
//!   glmmTMB): at the mode `û` of the joint log density
//!   `g(u) = Σᵢ log f(yᵢ | u) − Σ u² / 2σ²`,
//!   `log L ≈ g(û) − Σ log σ − ½ log det H`, where
//!   `H = Zᵀ W Z + diag(1/σ²)` is minus the Hessian of `g` in `u`, with the
//!   observed weights `W` (glmmTMB's choice; for the Poisson they are
//!   the expected ones, so `glmer` agrees);
//! - or, with a single grouping factor, by adaptive Gauss–Hermite
//!   quadrature on each level's intercept, centred at `û` and scaled by
//!   `H^(-1/2)` (`glmer(nAGQ = n)`); one node is the Laplace
//!   approximation.
//!
//! The inner mode is found by Newton's method on `u`, to machine
//! precision, so that the outer objective is smooth; the outer
//! maximization over `(β, log σ, log φ)` is quasi-Newton (BFGS) on
//! central-difference gradients, polished by Newton steps on a
//! finite-difference Hessian, whose inverse is the covariance of the
//! estimates. `H` is dense in the random effects (crossed factors couple
//! them), so a fit with several factors costs `O(q³)` per evaluation in
//! the `q` levels; one factor keeps `H` diagonal.
//!
//! The fit gives each level's predicted intercept `û` with its
//! conditional variance `(H⁻¹)ⱼⱼ` at the estimates (`lme4`'s
//! `condVar`), and the prediction variance that also carries the
//! estimates' uncertainty, `(H⁻¹)ⱼⱼ + Jⱼ Σ Jⱼᵀ` with `J = ∂û/∂(β, θ)`
//! (Kass and Steffey's correction, glmmTMB's `condVar`). A level the fit
//! has not seen has intercept 0 in [`GlmmFit::predict`].
//!
//! ```
//! use prospicio_glm::glmm::{Glmm, Grouping};
//! use prospicio_models::{Design, Family};
//!
//! // Claim counts for six policies in three territories.
//! let territory = Grouping::new(
//!     "territory",
//!     ["a", "a", "b", "b", "c", "c"].map(String::from).to_vec(),
//! );
//! let design = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; 6]]).unwrap();
//! let y = [1.0, 2.0, 4.0, 6.0, 0.0, 1.0];
//! let fit = Glmm::new(Family::Poisson).fit(&design, &[territory], &y).unwrap();
//! let re = &fit.random_effects()[0];
//! assert_eq!(re.levels, ["a", "b", "c"]);
//! // Territory b has the most claims, so the largest intercept.
//! assert!(re.values[1] > re.values[0] && re.values[0] > re.values[2]);
//! // A territory the fit has not seen gets the population mean.
//! let new = Grouping::new("territory", vec!["d".into()]);
//! let one = Design::new(vec!["(Intercept)".into()], vec![vec![1.0]]).unwrap();
//! let mu = fit.predict(&one, &[new]).unwrap();
//! assert!((mu[0] - fit.coefficients()[0].exp()).abs() < 1e-12);
//! ```

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};

use prospicio_core::{Error, Result};
use prospicio_math::linalg::{cholesky, cholesky_inverse, cholesky_solve};
use prospicio_models::{Design, Family, Link, Model};

use crate::{Dispersion, Glm};

/// One grouping factor: a name and each row's level.
#[derive(Debug, Clone, PartialEq)]
pub struct Grouping {
    pub name: String,
    pub labels: Vec<String>,
}

impl Grouping {
    /// A grouping factor `name` with `labels[i]` the level of row `i`.
    pub fn new(name: &str, labels: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            labels,
        }
    }

    /// The factor `inner` nested in `outer`: named `outer:inner`, with
    /// levels `a:b`, so that the same inner label under two outer levels is
    /// two levels (lme4's `(1 | outer/inner)` is `outer` and this).
    pub fn nested(outer: &Grouping, inner: &Grouping) -> Result<Self> {
        if outer.labels.len() != inner.labels.len() {
            return Err(Error::Data(format!(
                "{} has {} rows and {} has {}",
                outer.name,
                outer.labels.len(),
                inner.name,
                inner.labels.len()
            )));
        }
        Ok(Self {
            name: format!("{}:{}", outer.name, inner.name),
            labels: outer
                .labels
                .iter()
                .zip(&inner.labels)
                .map(|(a, b)| format!("{a}:{b}"))
                .collect(),
        })
    }
}

/// How the random effects are integrated out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Integration {
    /// The Laplace approximation (`glmer`'s `nAGQ = 1`, glmmTMB).
    Laplace,
    /// Adaptive Gauss–Hermite quadrature with `nodes` nodes on each
    /// level's intercept (`glmer(nAGQ = nodes)`); a single grouping
    /// factor only.
    Quadrature { nodes: usize },
}

/// A GLMM specification: the family (log link) and how to integrate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glmm {
    /// Poisson, gamma, or Tweedie with power in `(1, 2)`.
    pub family: Family,
    pub integration: Integration,
    /// The outer maximization stops when every gradient component is below
    /// `tolerance` times `max(1, |log L|)`.
    pub tolerance: f64,
    /// Quasi-Newton iterations at most.
    pub max_iterations: usize,
}

impl Glmm {
    /// A GLMM with the given family, the Laplace approximation, tolerance
    /// `1e-9` and at most 500 iterations.
    pub fn new(family: Family) -> Self {
        Self {
            family,
            integration: Integration::Laplace,
            tolerance: 1e-9,
            max_iterations: 500,
        }
    }

    /// The same GLMM integrated by adaptive Gauss–Hermite quadrature with
    /// `nodes` nodes.
    pub fn quadrature(mut self, nodes: usize) -> Self {
        self.integration = Integration::Quadrature { nodes };
        self
    }

    /// The variance power `p` and whether `φ` is estimated.
    fn power(&self) -> Result<(f64, bool)> {
        match self.family {
            Family::Poisson => Ok((1.0, false)),
            Family::Gamma => Ok((2.0, true)),
            Family::Tweedie { power } if power > 1.0 && power < 2.0 => Ok((power, true)),
            Family::Tweedie { power } => Err(Error::InvalidParameter {
                name: "power",
                value: power,
                reason: "must be in (1, 2) for a Tweedie GLMM",
            }),
            f => Err(Error::Data(format!(
                "a GLMM takes the Poisson, gamma or Tweedie, not the {}",
                f.name()
            ))),
        }
    }

    /// Fits the GLMM of `y` on the fixed effects `design` (its offset and
    /// prior weights included) with a random intercept for each factor in
    /// `groups`.
    ///
    /// Starts from the GLM without random effects and `σ = 0.5`. Fails if
    /// there is no grouping factor, a factor's labels do not match the
    /// rows, a response is outside the family's range, the design is
    /// collinear, quadrature is asked for with more than one factor or no
    /// nodes, or the optimization cannot leave its start; a fit that stops
    /// at the iteration limit is returned with
    /// [`converged`](GlmmFit::converged) false.
    pub fn fit(&self, design: &Design, groups: &[Grouping], y: &[f64]) -> Result<GlmmFit> {
        let (power, free_dispersion) = self.power()?;
        let n = design.n_rows();
        if y.len() != n {
            return Err(Error::Data(format!(
                "{} responses for {n} design rows",
                y.len()
            )));
        }
        if groups.is_empty() {
            return Err(Error::Data(
                "a GLMM needs at least one grouping factor; fit a GLM instead".into(),
            ));
        }
        if let Integration::Quadrature { nodes } = self.integration {
            if groups.len() != 1 {
                return Err(Error::Data(format!(
                    "quadrature takes a single grouping factor, not {}; use the Laplace approximation",
                    groups.len()
                )));
            }
            if nodes == 0 {
                return Err(Error::Data("quadrature needs at least one node".into()));
            }
        }
        if let Some(bad) = y.iter().find(|&&v| !self.family.valid_y(v)) {
            return Err(Error::Data(format!(
                "response {bad} is outside the {} family's range",
                self.family.name()
            )));
        }
        let mut levels = Vec::with_capacity(groups.len());
        let mut index = vec![Vec::with_capacity(groups.len()); n];
        let mut start = 0;
        for g in groups {
            if g.labels.len() != n {
                return Err(Error::Data(format!(
                    "grouping factor {} has {} labels for {n} rows",
                    g.name,
                    g.labels.len()
                )));
            }
            let set: BTreeSet<&String> = g.labels.iter().collect();
            let names: Vec<String> = set.into_iter().cloned().collect();
            let lookup: HashMap<&str, usize> = names
                .iter()
                .enumerate()
                .map(|(j, l)| (l.as_str(), j))
                .collect();
            for (i, l) in g.labels.iter().enumerate() {
                index[i].push(start + lookup[l.as_str()]);
            }
            start += names.len();
            levels.push(names);
        }
        let factor_of: Vec<usize> = levels
            .iter()
            .enumerate()
            .flat_map(|(k, l)| std::iter::repeat_n(k, l.len()))
            .collect();

        // The GLM without random effects starts β and φ.
        let glm = Glm::new(self.family, Link::Log).dispersion(if free_dispersion {
            Dispersion::Pearson
        } else {
            Dispersion::Fixed(1.0)
        });
        let start_fit = glm.fit(design, y)?;
        let p = design.n_cols();
        let k = groups.len();
        let mut x0 = start_fit.coefficients().to_vec();
        x0.extend(std::iter::repeat_n(0.5f64.ln(), k));
        if free_dispersion {
            x0.push(start_fit.dispersion().ln());
        }

        let problem = Problem {
            design,
            y,
            power,
            free_dispersion,
            index: &index,
            factor_of: &factor_of,
            n_factors: k,
            integration: self.integration,
            nodes: match self.integration {
                Integration::Quadrature { nodes } => gauss_hermite(nodes),
                Integration::Laplace => (Vec::new(), Vec::new()),
            },
        };
        let q = factor_of.len();
        // Each evaluation's inner iteration starts at the last one's mode.
        let start_u = match problem.marginal(&x0, &vec![0.0; q], false) {
            Ok(m) if m.log_likelihood.is_finite() => m.u,
            _ => {
                return Err(Error::Data(
                    "the marginal likelihood is not finite at the GLM's estimates".into(),
                ));
            }
        };
        let warm = RefCell::new(start_u);
        let mut objective = |x: &[f64]| -> f64 {
            let m = problem.marginal(x, &warm.borrow(), false);
            match m {
                Ok(m) if m.log_likelihood.is_finite() => {
                    *warm.borrow_mut() = m.u;
                    -m.log_likelihood
                }
                _ => f64::INFINITY,
            }
        };
        let (x, iterations, converged) =
            minimize(&mut objective, x0, self.tolerance, self.max_iterations);

        let u = warm.borrow().clone();
        let at = problem.marginal(&x, &u, true)?;
        let hessian = fd_hessian(&mut objective, &x, at.log_likelihood.abs());
        let d = x.len();
        // A σ at the boundary (below 1e-4) leaves the likelihood flat in
        // log σ: the other parameters' covariance is then taken with it
        // held at 0, and its own standard error is NaN.
        let boundary: Vec<bool> = (0..d)
            .map(|a| a >= p && a < p + k && x[a] < 1e-4f64.ln())
            .collect();
        let free: Vec<usize> = (0..d).filter(|&a| !boundary[a]).collect();
        let f = free.len();
        let sub: Vec<f64> = free
            .iter()
            .flat_map(|&a| free.iter().map(move |&b| (a, b)))
            .map(|(a, b)| hessian[a * d + b])
            .collect();
        let mut covariance = vec![f64::NAN; d * d];
        if let Some(l) = cholesky(&sub, f) {
            let inv = cholesky_inverse(&l, f);
            for (i, &a) in free.iter().enumerate() {
                for (j, &b) in free.iter().enumerate() {
                    covariance[a * d + b] = inv[i * f + j];
                }
            }
        }

        // ∂û/∂(β, θ) by central differences, for the prediction variance.
        let mut jac = vec![vec![0.0; d]; q];
        for a in 0..d {
            let h = 1e-5 * x[a].abs().max(1.0);
            let mut xp = x.clone();
            xp[a] += h;
            let up = problem.marginal(&xp, &at.u, false)?.u;
            xp[a] = x[a] - h;
            let um = problem.marginal(&xp, &at.u, false)?.u;
            for j in 0..q {
                jac[j][a] = (up[j] - um[j]) / (2.0 * h);
            }
        }
        let mut random_effects = Vec::with_capacity(k);
        let mut offset = 0;
        for (g, names) in groups.iter().zip(&levels) {
            let m = names.len();
            let values = at.u[offset..offset + m].to_vec();
            let conditional_variance = at.conditional_variance[offset..offset + m].to_vec();
            let prediction_variance = (0..m)
                .map(|j| {
                    let row = &jac[offset + j];
                    let extra: f64 = free
                        .iter()
                        .map(|&a| {
                            free.iter()
                                .map(|&b| row[a] * covariance[a * d + b] * row[b])
                                .sum::<f64>()
                        })
                        .sum();
                    conditional_variance[j] + extra
                })
                .collect();
            random_effects.push(RandomEffects {
                name: g.name.clone(),
                levels: names.clone(),
                values,
                conditional_variance,
                prediction_variance,
            });
            offset += m;
        }

        let sd: Vec<f64> = x[p..p + k].iter().map(|v| v.exp()).collect();
        let sd_standard_errors = (0..k)
            .map(|j| sd[j] * covariance[(p + j) * d + p + j].sqrt())
            .collect();
        let (dispersion, dispersion_standard_error) = if free_dispersion {
            let phi = x[p + k].exp();
            (phi, phi * covariance[(p + k) * d + p + k].sqrt())
        } else {
            (1.0, 0.0)
        };
        let beta_covariance = (0..p)
            .flat_map(|a| (0..p).map(move |b| (a, b)))
            .map(|(a, b)| covariance[a * d + b])
            .collect();
        Ok(GlmmFit {
            spec: *self,
            names: design.names().to_vec(),
            coefficients: x[..p].to_vec(),
            covariance: beta_covariance,
            group_names: groups.iter().map(|g| g.name.clone()).collect(),
            sd,
            sd_standard_errors,
            dispersion,
            dispersion_standard_error,
            log_likelihood: at.log_likelihood,
            n_parameters: d,
            n_obs: n,
            fitted: at.mu,
            random_effects,
            iterations,
            converged,
        })
    }
}

/// The predicted intercepts of one grouping factor.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomEffects {
    /// The grouping factor's name.
    pub name: String,
    /// Its levels, sorted.
    pub levels: Vec<String>,
    /// Each level's predicted intercept `û`, the conditional mode.
    pub values: Vec<f64>,
    /// `(H⁻¹)ⱼⱼ`: the variance of `u` given the data at the estimates
    /// (lme4's `condVar`).
    pub conditional_variance: Vec<f64>,
    /// The conditional variance plus the part from the estimates'
    /// uncertainty, `Jⱼ Σ Jⱼᵀ` (glmmTMB's `condVar`).
    pub prediction_variance: Vec<f64>,
}

/// A fitted GLMM.
#[derive(Debug, Clone, PartialEq)]
pub struct GlmmFit {
    spec: Glmm,
    names: Vec<String>,
    coefficients: Vec<f64>,
    covariance: Vec<f64>,
    group_names: Vec<String>,
    sd: Vec<f64>,
    sd_standard_errors: Vec<f64>,
    dispersion: f64,
    dispersion_standard_error: f64,
    log_likelihood: f64,
    n_parameters: usize,
    n_obs: usize,
    fitted: Vec<f64>,
    random_effects: Vec<RandomEffects>,
    iterations: usize,
    converged: bool,
}

impl GlmmFit {
    /// The specification fitted.
    pub fn spec(&self) -> &Glmm {
        &self.spec
    }

    /// The fixed effects' names.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The fixed effects `β`.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// The covariance of `β`, row-major: its block of the inverse Hessian
    /// of `−log L` in `(β, log σ, log φ)`.
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// The standard errors of `β`.
    pub fn standard_errors(&self) -> Vec<f64> {
        let p = self.coefficients.len();
        (0..p).map(|j| self.covariance[j * p + j].sqrt()).collect()
    }

    /// The grouping factors' names, in the order fitted.
    pub fn group_names(&self) -> &[String] {
        &self.group_names
    }

    /// Each grouping factor's standard deviation `σₖ`.
    pub fn sd(&self) -> &[f64] {
        &self.sd
    }

    /// The standard errors of the `σₖ`, by the delta method from `log σₖ`.
    pub fn sd_standard_errors(&self) -> &[f64] {
        &self.sd_standard_errors
    }

    /// The dispersion `φ` (the response's variance is `φ μᵖ / w`): 1 for
    /// the Poisson; for the gamma glmmTMB's `sigma()` is `√φ`, for the
    /// Tweedie `φ`.
    pub fn dispersion(&self) -> f64 {
        self.dispersion
    }

    /// The standard error of `φ` (0 for the Poisson).
    pub fn dispersion_standard_error(&self) -> f64 {
        self.dispersion_standard_error
    }

    /// The marginal log-likelihood at the estimates, by the approximation
    /// fitted.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// `−2 log L + 2 k` with `k` the fixed effects, the `σₖ` and `φ` if
    /// estimated.
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * self.n_parameters as f64
    }

    /// `−2 log L + k log n`.
    pub fn bic(&self) -> f64 {
        -2.0 * self.log_likelihood + self.n_parameters as f64 * (self.n_obs as f64).ln()
    }

    /// The conditional means `μᵢ` of the training rows, at `û`.
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// The predicted intercepts, one entry per grouping factor.
    pub fn random_effects(&self) -> &[RandomEffects] {
        &self.random_effects
    }

    /// Quasi-Newton iterations used.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Whether the outer maximization met its tolerance.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Means for new rows: `exp(xᵀ β + offset + Σ û)`, each factor's
    /// intercept looked up by `groups` (matched by name; a level the fit
    /// has not seen, or a factor not given, contributes 0, the population
    /// mean). An empty `groups` predicts the population means.
    pub fn predict(&self, design: &Design, groups: &[Grouping]) -> Result<Vec<f64>> {
        if design.names() != self.names.as_slice() {
            return Err(Error::Data(
                "the design's columns differ from the fit's".into(),
            ));
        }
        let mut eta = design.linear_predictor(&self.coefficients);
        for g in groups {
            if g.labels.len() != eta.len() {
                return Err(Error::Data(format!(
                    "grouping factor {} has {} labels for {} rows",
                    g.name,
                    g.labels.len(),
                    eta.len()
                )));
            }
            let re = self
                .random_effects
                .iter()
                .find(|r| r.name == g.name)
                .ok_or_else(|| Error::Data(format!("the fit has no grouping factor {}", g.name)))?;
            let lookup: HashMap<&str, f64> = re
                .levels
                .iter()
                .map(String::as_str)
                .zip(re.values.iter().copied())
                .collect();
            for (e, l) in eta.iter_mut().zip(&g.labels) {
                *e += lookup.get(l.as_str()).copied().unwrap_or(0.0);
            }
        }
        Ok(eta.into_iter().map(f64::exp).collect())
    }
}

/// The data and settings shared by every evaluation.
struct Problem<'a> {
    design: &'a Design,
    y: &'a [f64],
    power: f64,
    free_dispersion: bool,
    /// Each row's random effects, as indices into `u`.
    index: &'a [Vec<usize>],
    /// Each random effect's factor.
    factor_of: &'a [usize],
    n_factors: usize,
    integration: Integration,
    /// Gauss–Hermite nodes and log weights (`log w + x²`).
    nodes: (Vec<f64>, Vec<f64>),
}

/// The joint log density `g(u)` of the data and the random effects, its
/// gradient and minus its Hessian in `u` (diagonal with one factor), and
/// the conditional means.
struct Joint {
    g: f64,
    grad: Vec<f64>,
    hess: Vec<f64>,
    mu: Vec<f64>,
}

/// One evaluation of the marginal likelihood.
struct Marginal {
    log_likelihood: f64,
    u: Vec<f64>,
    conditional_variance: Vec<f64>,
    mu: Vec<f64>,
}

impl Problem<'_> {
    /// Log density of row `i` and its first two derivatives in `η`
    /// (the second negated, the observed weight).
    fn row(&self, i: usize, eta: f64, phi: f64) -> (f64, f64, f64) {
        let (y, w, p) = (self.y[i], self.design.weights()[i], self.power);
        let mu = eta.exp();
        let family = if p == 1.0 {
            Family::Poisson
        } else if p == 2.0 {
            Family::Gamma
        } else {
            Family::Tweedie { power: p }
        };
        let ll = family.log_likelihood(y, mu, w, phi);
        let m1p = mu.powf(1.0 - p);
        let s = w / phi * (y - mu) * m1p;
        let h = w / phi * m1p * (mu - (1.0 - p) * (y - mu));
        (ll, s, h)
    }

    /// The marginal log-likelihood at `x = (β, log σ, log φ)`, with the
    /// inner Newton iteration started at `u0`; the conditional variances
    /// only `with_variance` (empty otherwise, for several factors).
    fn marginal(&self, x: &[f64], u0: &[f64], with_variance: bool) -> Result<Marginal> {
        let p = self.design.n_cols();
        let k = self.n_factors;
        let beta = &x[..p];
        let sigma2: Vec<f64> = x[p..p + k].iter().map(|v| (2.0 * v).exp()).collect();
        let phi = if self.free_dispersion {
            x[p + k].exp()
        } else {
            1.0
        };
        if !(phi.is_finite() && phi > 0.0) || sigma2.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
            return Err(Error::Data("parameters out of range".into()));
        }
        let eta0 = self.design.linear_predictor(beta);
        let n = eta0.len();
        let q = self.factor_of.len();
        let diagonal = k == 1;
        let mut u = u0.to_vec();

        // g(u), its gradient and minus its Hessian at u.
        let evaluate = |u: &[f64]| -> Option<Joint> {
            let mut g = 0.0;
            let mut grad = vec![0.0; q];
            let mut hess = vec![0.0; if diagonal { q } else { q * q }];
            let mut mu = Vec::with_capacity(n);
            for (i, (idx, e0)) in self.index.iter().zip(&eta0).enumerate() {
                let eta = e0 + idx.iter().map(|&j| u[j]).sum::<f64>();
                let (ll, s, h) = self.row(i, eta, phi);
                if !ll.is_finite() {
                    return None;
                }
                g += ll;
                mu.push(eta.exp());
                for &a in idx {
                    grad[a] += s;
                    if diagonal {
                        hess[a] += h;
                    } else {
                        for &b in idx {
                            hess[a * q + b] += h;
                        }
                    }
                }
            }
            for j in 0..q {
                let s2 = sigma2[self.factor_of[j]];
                g -= u[j] * u[j] / (2.0 * s2);
                grad[j] -= u[j] / s2;
                if diagonal {
                    hess[j] += 1.0 / s2;
                } else {
                    hess[j * q + j] += 1.0 / s2;
                }
            }
            Some(Joint { g, grad, hess, mu })
        };
        let solve = |hess: &[f64], grad: &[f64]| -> Option<(Vec<f64>, Vec<f64>)> {
            if diagonal {
                Some((
                    grad.iter().zip(hess).map(|(g, h)| g / h).collect(),
                    hess.to_vec(),
                ))
            } else {
                cholesky(hess, q).map(|l| (cholesky_solve(&l, grad), l))
            }
        };

        let mut state = match evaluate(&u) {
            Some(s) => s,
            None => {
                u = vec![0.0; q];
                evaluate(&u).ok_or_else(|| Error::Data("the joint density is not finite".into()))?
            }
        };
        for _ in 0..200 {
            let (step, _) = solve(&state.hess, &state.grad)
                .ok_or_else(|| Error::Data("the random effects' Hessian is singular".into()))?;
            let size = step.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let mut t = 1.0;
            let mut moved = false;
            for _ in 0..50 {
                let trial: Vec<f64> = u.iter().zip(&step).map(|(a, b)| a + t * b).collect();
                if let Some(next) = evaluate(&trial) {
                    if next.g >= state.g - 1e-12 * state.g.abs().max(1.0) {
                        u = trial;
                        state = next;
                        moved = true;
                        break;
                    }
                }
                t *= 0.5;
            }
            // Newton converges quadratically: after a full step below
            // 1e-10 the error left is of order its square, below round-off
            // (a looser stop leaves enough error in the log-determinant to
            // blur the outer finite differences).
            if !moved || (t == 1.0 && size < 1e-10) || size * t < 1e-14 {
                break;
            }
        }
        let Joint { g, hess, mu, .. } = state;
        let (_, factor) = solve(&hess, &vec![0.0; q])
            .ok_or_else(|| Error::Data("the random effects' Hessian is singular".into()))?;
        let (log_det, conditional_variance) = if diagonal {
            (
                factor.iter().map(|h| h.ln()).sum::<f64>(),
                factor.iter().map(|h| 1.0 / h).collect::<Vec<_>>(),
            )
        } else {
            let log_det = 2.0 * (0..q).map(|j| factor[j * q + j].ln()).sum::<f64>();
            // The inverse costs as much as the rest of an evaluation:
            // only the final one needs it.
            let variance = if with_variance {
                let inv = cholesky_inverse(&factor, q);
                (0..q).map(|j| inv[j * q + j]).collect()
            } else {
                Vec::new()
            };
            (log_det, variance)
        };
        let log_sigma: f64 = self.factor_of.iter().map(|&f| 0.5 * sigma2[f].ln()).sum();
        let log_likelihood = match self.integration {
            Integration::Laplace => g - log_sigma - 0.5 * log_det,
            Integration::Quadrature { .. } => self.quadrature(&eta0, &u, &hess, &sigma2, phi)?,
        };
        Ok(Marginal {
            log_likelihood,
            u,
            conditional_variance,
            mu,
        })
    }

    /// Adaptive Gauss–Hermite quadrature on each level of the single
    /// factor, centred at the mode `u` and scaled by `hess^(-1/2)`.
    fn quadrature(
        &self,
        eta0: &[f64],
        u: &[f64],
        hess: &[f64],
        sigma2: &[f64],
        phi: f64,
    ) -> Result<f64> {
        let q = u.len();
        let mut rows = vec![Vec::new(); q];
        for (i, idx) in self.index.iter().enumerate() {
            rows[idx[0]].push(i);
        }
        let s2 = sigma2[0];
        let (x, lw) = &self.nodes;
        let mut total = 0.0;
        for j in 0..q {
            let tau = (2.0 / hess[j]).sqrt();
            let terms: Vec<f64> = x
                .iter()
                .zip(lw)
                .map(|(xk, lwk)| {
                    let v = u[j] + tau * xk;
                    let ll: f64 = rows[j]
                        .iter()
                        .map(|&i| self.row(i, eta0[i] + v, phi).0)
                        .sum();
                    lwk + ll - v * v / (2.0 * s2)
                })
                .collect();
            let top = terms.iter().fold(f64::NEG_INFINITY, |m, &t| m.max(t));
            if !top.is_finite() {
                return Err(Error::Data(
                    "a quadrature node's density is not finite".into(),
                ));
            }
            let sum: f64 = terms.iter().map(|t| (t - top).exp()).sum();
            total += top + sum.ln() + tau.ln() - 0.5 * (2.0 * std::f64::consts::PI * s2).ln();
        }
        Ok(total)
    }
}

/// Gauss–Hermite nodes `x` and `log w + x²` for `∫ e^(-x²) f(x) dx`, by
/// Newton's method on the orthonormal Hermite recurrence (Press et al.,
/// *Numerical Recipes*, 4.6).
fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let pim4 = std::f64::consts::PI.powf(-0.25);
    let nf = n as f64;
    let mut z = 0.0f64;
    for i in 0..n.div_ceil(2) {
        z = match i {
            0 => (2.0 * nf + 1.0).sqrt() - 1.85575 * (2.0 * nf + 1.0).powf(-1.0 / 6.0),
            1 => z - 1.14 * nf.powf(0.426) / z,
            2 => 1.86 * z - 0.86 * x[0],
            3 => 1.91 * z - 0.91 * x[1],
            _ => 2.0 * z - x[i - 2],
        };
        let mut pp = 0.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (pim4, 0.0);
            for j in 1..=n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = z * (2.0 / jf).sqrt() * p2 - ((jf - 1.0) / jf).sqrt() * p3;
            }
            pp = (2.0 * nf).sqrt() * p2;
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() <= 1e-15 * z.abs().max(1.0) {
                break;
            }
        }
        x[i] = z;
        x[n - 1 - i] = -z;
        w[i] = 2.0 / (pp * pp);
        w[n - 1 - i] = w[i];
    }
    let lw = x.iter().zip(&w).map(|(xi, wi)| wi.ln() + xi * xi).collect();
    (x, lw)
}

/// Central-difference gradient of `f` at `x`.
fn gradient(f: &mut impl FnMut(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let mut xp = x.to_vec();
    (0..x.len())
        .map(|a| {
            let h = 1e-5 * x[a].abs().max(1.0);
            xp[a] = x[a] + h;
            let up = f(&xp);
            xp[a] = x[a] - h;
            let down = f(&xp);
            xp[a] = x[a];
            (up - down) / (2.0 * h)
        })
        .collect()
}

/// Finite-difference Hessian of `f` at `x`, symmetric, row-major; the step
/// grows with `scale` (`|f|`) so that round-off stays below the curvature.
fn fd_hessian(f: &mut impl FnMut(&[f64]) -> f64, x: &[f64], scale: f64) -> Vec<f64> {
    let d = x.len();
    let rel = 1e-4 * scale.max(1.0).powf(0.25).min(10.0);
    let h: Vec<f64> = x.iter().map(|v| rel * v.abs().max(1.0)).collect();
    let f0 = f(x);
    let mut xp = x.to_vec();
    let mut at = |xp: &mut Vec<f64>, a: usize, sa: f64, b: usize, sb: f64| {
        xp[a] += sa * h[a];
        xp[b] += sb * h[b];
        let v = f(xp);
        xp[a] = x[a];
        xp[b] = x[b];
        v
    };
    let mut hess = vec![0.0; d * d];
    for a in 0..d {
        let up = at(&mut xp, a, 1.0, a, 0.0);
        let down = at(&mut xp, a, -1.0, a, 0.0);
        hess[a * d + a] = (up - 2.0 * f0 + down) / (h[a] * h[a]);
        for b in 0..a {
            let v = (at(&mut xp, a, 1.0, b, 1.0)
                - at(&mut xp, a, 1.0, b, -1.0)
                - at(&mut xp, a, -1.0, b, 1.0)
                + at(&mut xp, a, -1.0, b, -1.0))
                / (4.0 * h[a] * h[b]);
            hess[a * d + b] = v;
            hess[b * d + a] = v;
        }
    }
    hess
}

/// Minimizes `f` from `x0` by BFGS on central-difference gradients with a
/// backtracking (Armijo) line search, then polishes with up to three
/// Newton steps on a finite-difference Hessian. Returns the minimizer, the
/// iterations and whether the gradient met the tolerance.
fn minimize(
    f: &mut impl FnMut(&[f64]) -> f64,
    x0: Vec<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, usize, bool) {
    let d = x0.len();
    let identity = |s: f64| {
        let mut m = vec![0.0; d * d];
        for a in 0..d {
            m[a * d + a] = s;
        }
        m
    };
    let mut x = x0;
    let mut fx = f(&x);
    let mut g = gradient(f, &x);
    let mut inv = identity(1.0);
    let mut fresh = true;
    let small = |g: &[f64], fx: f64| {
        g.iter().fold(0.0f64, |m, v| m.max(v.abs())) <= tolerance * fx.abs().max(1.0)
    };
    // A variance running to 0 drives its log scale off to -inf with a
    // gradient that never quite vanishes: three steps in a row that lower
    // `f` by no more than round-off end the search at that boundary.
    let mut flat = 0;
    let mut iterations = 0;
    while iterations < max_iterations && !small(&g, fx) && flat < 3 {
        iterations += 1;
        let mut dir: Vec<f64> = (0..d)
            .map(|a| -(0..d).map(|b| inv[a * d + b] * g[b]).sum::<f64>())
            .collect();
        let mut slope: f64 = dir.iter().zip(&g).map(|(a, b)| a * b).sum();
        if slope >= 0.0 {
            inv = identity(1.0);
            fresh = true;
            dir = g.iter().map(|v| -v).collect();
            slope = -g.iter().map(|v| v * v).sum::<f64>();
        }
        // Steps of more than 5 in any parameter (log scales) are cut back.
        let longest = dir.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut t = if longest > 5.0 { 5.0 / longest } else { 1.0 };
        let mut next = None;
        for _ in 0..60 {
            let trial: Vec<f64> = x.iter().zip(&dir).map(|(a, b)| a + t * b).collect();
            let ft = f(&trial);
            if ft.is_finite() && ft <= fx + 1e-4 * t * slope {
                next = Some((trial, ft));
                break;
            }
            t *= 0.5;
        }
        let Some((xn, fnew)) = next else {
            if fresh {
                break;
            }
            inv = identity(1.0);
            fresh = true;
            continue;
        };
        let gn = gradient(f, &xn);
        let s: Vec<f64> = xn.iter().zip(&x).map(|(a, b)| a - b).collect();
        let yv: Vec<f64> = gn.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(&yv).map(|(a, b)| a * b).sum();
        let yy: f64 = yv.iter().map(|v| v * v).sum();
        let ss: f64 = s.iter().map(|v| v * v).sum();
        if sy > 1e-12 * (ss * yy).sqrt() {
            if fresh {
                inv = identity(sy / yy);
                fresh = false;
            }
            // H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ.
            let rho = 1.0 / sy;
            let hy: Vec<f64> = (0..d)
                .map(|a| (0..d).map(|b| inv[a * d + b] * yv[b]).sum())
                .collect();
            let yhy: f64 = yv.iter().zip(&hy).map(|(a, b)| a * b).sum();
            for a in 0..d {
                for b in 0..d {
                    inv[a * d + b] += -rho * (hy[a] * s[b] + s[a] * hy[b])
                        + (rho * rho * yhy + rho) * s[a] * s[b];
                }
            }
        }
        flat = if fx - fnew <= 1e-13 * fx.abs().max(1.0) {
            flat + 1
        } else {
            0
        };
        x = xn;
        fx = fnew;
        g = gn;
    }
    // Newton polish: BFGS's inverse is only an estimate near the optimum;
    // one finite-difference Hessian serves every step.
    let hess = fd_hessian(f, &x, fx.abs());
    let factor = cholesky(&hess, d);
    for _ in 0..3 {
        let Some(l) = &factor else { break };
        let step = cholesky_solve(l, &g);
        let trial: Vec<f64> = x.iter().zip(&step).map(|(a, b)| a - b).collect();
        let ft = f(&trial);
        if !(ft.is_finite() && ft <= fx + 1e-12 * fx.abs().max(1.0)) {
            break;
        }
        x = trial;
        fx = ft;
        g = gradient(f, &x);
    }
    let converged = small(&g, fx) || flat >= 3;
    (x, iterations, converged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_hermite_integrates_polynomials() {
        let (x, lw) = gauss_hermite(7);
        let w: Vec<f64> = x
            .iter()
            .zip(&lw)
            .map(|(xi, l)| (l - xi * xi).exp())
            .collect();
        let pi = std::f64::consts::PI;
        let m = |k: i32| {
            x.iter()
                .zip(&w)
                .map(|(xi, wi)| wi * xi.powi(k))
                .sum::<f64>()
        };
        assert!((m(0) - pi.sqrt()).abs() < 1e-13);
        assert!((m(2) - pi.sqrt() / 2.0).abs() < 1e-13);
        assert!((m(4) - 3.0 * pi.sqrt() / 4.0).abs() < 1e-12);
        assert!((m(12) - 10395.0 * pi.sqrt() / 64.0).abs() < 1e-9);
    }

    /// Forty rows in eight groups with a group effect and one covariate.
    fn book() -> (Design, Vec<String>, Vec<f64>) {
        let n = 40;
        let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 / 10.0 - 0.5).collect();
        let labels: Vec<String> = (0..n).map(|i| format!("g{}", i % 8)).collect();
        let effect = [0.3, -0.2, 0.5, -0.4, 0.1, 0.0, -0.1, 0.2];
        let y: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * (1.0 + 0.4 * x[i] + effect[i % 8]).exp()
                    * (1.0 + 0.3 * ((i * 13) % 5) as f64 - 0.6))
                    .round()
                    .max(0.0)
            })
            .collect();
        let design = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; n], x],
        )
        .unwrap();
        (design, labels, y)
    }

    #[test]
    fn one_node_is_the_laplace_fit() {
        let (design, labels, y) = book();
        let g = [Grouping::new("g", labels)];
        let laplace = Glmm::new(Family::Poisson).fit(&design, &g, &y).unwrap();
        let one = Glmm::new(Family::Poisson)
            .quadrature(1)
            .fit(&design, &g, &y)
            .unwrap();
        assert!(laplace.converged() && one.converged());
        assert!((laplace.log_likelihood() - one.log_likelihood()).abs() < 1e-9);
        for (a, b) in laplace.coefficients().iter().zip(one.coefficients()) {
            assert!((a - b).abs() < 1e-7, "{a} {b}");
        }
        // More nodes move the fit only a little.
        let nine = Glmm::new(Family::Poisson)
            .quadrature(9)
            .fit(&design, &g, &y)
            .unwrap();
        assert!((nine.log_likelihood() - laplace.log_likelihood()).abs() < 0.1);
        assert!((nine.sd()[0] - laplace.sd()[0]).abs() < 0.05 * laplace.sd()[0]);
    }

    #[test]
    fn the_fit_is_a_maximum_and_predicts_by_label() {
        let (design, labels, y) = book();
        let g = [Grouping::new("g", labels.clone())];
        let fit = Glmm::new(Family::Poisson).fit(&design, &g, &y).unwrap();
        assert_eq!(fit.random_effects()[0].levels.len(), 8);
        // The training rows' means are the predictions at their labels.
        let mu = fit.predict(&design, &g).unwrap();
        for (a, b) in mu.iter().zip(fit.fitted()) {
            assert!((a - b).abs() < 1e-9 * b);
        }
        // Without the factor, or at an unseen level, the population mean.
        let population = fit.predict(&design, &[]).unwrap();
        let unseen = fit
            .predict(&design, &[Grouping::new("g", vec!["new".into(); 40])])
            .unwrap();
        assert_eq!(population, unseen);
        let re = &fit.random_effects()[0];
        for j in 0..8 {
            assert!(re.conditional_variance[j] > 0.0);
            assert!(re.prediction_variance[j] > re.conditional_variance[j]);
        }
        assert!(fit.standard_errors().iter().all(|s| *s > 0.0));
        assert!(fit.aic() > -2.0 * fit.log_likelihood());
    }

    #[test]
    fn groups_without_an_effect_run_to_a_zero_sd() {
        // Every group has the same responses, so the maximum is at σ = 0.
        let n = 40;
        let y: Vec<f64> = (0..n).map(|i| [1.0, 3.0, 2.0, 5.0, 0.0][i / 8]).collect();
        let labels: Vec<String> = (0..n).map(|i| format!("g{}", i % 8)).collect();
        let design = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; n]]).unwrap();
        let fit = Glmm::new(Family::Poisson)
            .fit(&design, &[Grouping::new("g", labels)], &y)
            .unwrap();
        // The search stops at the boundary instead of at the iteration
        // limit; β and its SE are the GLM's: log 2.2 and 1 / √(40 · 2.2).
        assert!(fit.converged() && fit.iterations() < 100);
        assert!(fit.sd()[0] < 1e-4);
        assert!(fit.sd_standard_errors()[0].is_nan());
        assert!((fit.coefficients()[0] - 2.2f64.ln()).abs() < 1e-6);
        assert!((fit.standard_errors()[0] - 1.0 / 88f64.sqrt()).abs() < 1e-5);
        let effects = &fit.random_effects()[0];
        assert!(effects.values.iter().all(|v| v.abs() < 1e-4));
        assert!(effects.prediction_variance.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn nested_labels_join_the_levels() {
        let a = Grouping::new("a", ["x", "x", "y"].map(String::from).to_vec());
        let b = Grouping::new("b", ["1", "2", "1"].map(String::from).to_vec());
        let ab = Grouping::nested(&a, &b).unwrap();
        assert_eq!(ab.name, "a:b");
        assert_eq!(ab.labels, ["x:1", "x:2", "y:1"]);
        let short = Grouping::new("b", vec!["1".into()]);
        assert!(Grouping::nested(&a, &short).is_err());
    }

    #[test]
    fn bad_inputs_are_errors() {
        let (design, labels, y) = book();
        let g = [Grouping::new("g", labels.clone())];
        let msg = |r: Result<GlmmFit>| r.unwrap_err().to_string();
        assert!(
            msg(Glmm::new(Family::Gaussian).fit(&design, &g, &y))
                .contains("Poisson, gamma or Tweedie")
        );
        assert!(
            Glmm::new(Family::Tweedie { power: 2.5 })
                .fit(&design, &g, &y)
                .is_err()
        );
        assert!(
            msg(Glmm::new(Family::Poisson).fit(&design, &[], &y)).contains("at least one grouping")
        );
        let two = [g[0].clone(), Grouping::new("h", labels.clone())];
        assert!(
            msg(Glmm::new(Family::Poisson)
                .quadrature(5)
                .fit(&design, &two, &y))
            .contains("single grouping")
        );
        assert!(
            Glmm::new(Family::Poisson)
                .quadrature(0)
                .fit(&design, &g, &y)
                .is_err()
        );
        let short = [Grouping::new("g", vec!["a".into()])];
        assert!(
            msg(Glmm::new(Family::Poisson).fit(&design, &short, &y))
                .contains("1 labels for 40 rows")
        );
        // The gamma needs every response positive.
        let mut zero = y.clone();
        zero[0] = 0.0;
        assert!(
            msg(Glmm::new(Family::Gamma).fit(&design, &g, &zero)).contains("outside the gamma")
        );
    }
}
