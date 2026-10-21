//! Generalized linear mixed models: the Poisson, gamma and Tweedie with
//! the log link, with random intercepts and slopes for each level of each
//! grouping factor, the factors one, nested or crossed
//! (`docs/design/models.md`, "Mixed models").
//!
//! The model for row `i` is `log μᵢ = xᵢᵀ β + oᵢ + Σₖ zₖᵢᵀ b_{k, gₖ(i)}`,
//! where `zₖᵢ` holds factor `k`'s terms at row `i` (1 for its intercept,
//! the covariate for a slope) and each level's effects
//! `b_{k, l} ~ N(0, Σₖ)` are independent, with `Σₖ` unstructured: a
//! standard deviation for each term and their correlations. `yᵢ` is from
//! the family with mean `μᵢ`, variance `φ V(μᵢ) / wᵢ` and `V(μ) = μᵖ`
//! (`p = 1` for the Poisson, whose `φ` is 1, `p = 2` for the gamma).
//! [`Glmm::fit`] maximizes the marginal likelihood of `β`, the `Σₖ` and
//! `φ`, with the random effects integrated out.
//!
//! The effects are written `b_{k, l} = Λₖ v_{k, l}` with `v ~ N(0, I)`
//! and `Λₖ` the Cholesky factor of `Σₖ` (lme4's parameterization):
//!
//! - by the Laplace approximation (the default, as R's `glmer` and
//!   glmmTMB): at the mode `v̂` of the joint log density
//!   `g(v) = Σᵢ log f(yᵢ | v) − vᵀv / 2`,
//!   `log L ≈ g(v̂) − ½ log det H`, where `H = ΛᵀZᵀ W Z Λ + I` is minus the
//!   Hessian of `g` in `v`, with the observed weights `W` (glmmTMB's
//!   choice; for the Poisson they are the expected ones, so `glmer`
//!   agrees);
//! - or, with a single grouping factor of a single term, by adaptive
//!   Gauss–Hermite quadrature on each level's effect, centred at `v̂` and
//!   scaled by `H^(-1/2)` (`glmer(nAGQ = n)`); one node is the Laplace
//!   approximation.
//!
//! `Σₖ = Λₖ Λₖᵀ` is parameterized as glmmTMB's `us` structure: the log
//! standard deviations, then the strictly lower entries, row by row, of a
//! unit lower triangle `L` whose rows normalized give the Cholesky factor
//! of the correlation matrix (with two terms, `ρ = θ / √(1 + θ²)`).
//!
//! The inner mode is found by Newton's method on `v`, to machine
//! precision, so that the outer objective is smooth; the outer
//! maximization over `(β, θ, log φ)` is quasi-Newton (BFGS) on
//! central-difference gradients, polished by Newton steps on a
//! finite-difference Hessian, whose inverse is the covariance of the
//! estimates. `H` is dense in the random effects (crossed factors and
//! slopes couple them), so a fit costs `O(q³)` per evaluation in the `q`
//! effects; one factor with one term keeps `H` diagonal.
//!
//! The fit gives each level's predicted effects `b̂ = Λ v̂` with their
//! conditional variances, the diagonal of `Λ H⁻¹ Λᵀ`, at the estimates
//! (lme4's `condVar`), and the prediction variances that also carry the
//! estimates' uncertainty, adding `Jⱼ C Jⱼᵀ` with `J = ∂b̂/∂(β, θ, log φ)`
//! and `C` their covariance (Kass and Steffey's correction, glmmTMB's
//! `condVar`). A level the fit has not seen has effects 0 in
//! [`GlmmFit::predict`].
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
//! assert_eq!(re.terms, ["(Intercept)"]);
//! // Territory b has the most claims, so the largest intercept.
//! let u = &re.values[0];
//! assert!(u[1] > u[0] && u[0] > u[2]);
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

/// The name of a factor's random intercept among its terms.
pub const INTERCEPT: &str = "(Intercept)";

/// One grouping factor: a name, each row's level and the terms that vary
/// by level, a random intercept and any random slopes, all correlated
/// (lme4's `(1 + x | g)`).
///
/// Terms that are independent (lme4's `(1 + x || g)`) are separate
/// factors on the same labels, one for each term.
#[derive(Debug, Clone, PartialEq)]
pub struct Grouping {
    pub name: String,
    pub labels: Vec<String>,
    /// Whether the factor has a random intercept.
    pub intercept: bool,
    /// The random slopes: each covariate's name and its value on every
    /// row.
    pub slopes: Vec<(String, Vec<f64>)>,
}

impl Grouping {
    /// A grouping factor `name` with `labels[i]` the level of row `i` and
    /// a random intercept.
    pub fn new(name: &str, labels: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            labels,
            intercept: true,
            slopes: Vec::new(),
        }
    }

    /// The same factor with a random slope on the covariate `name`, whose
    /// value on row `i` is `values[i]`, correlated with the factor's other
    /// terms.
    pub fn slope(mut self, name: &str, values: Vec<f64>) -> Self {
        self.slopes.push((name.to_string(), values));
        self
    }

    /// The same factor without its random intercept (lme4's
    /// `(0 + x | g)`).
    pub fn without_intercept(mut self) -> Self {
        self.intercept = false;
        self
    }

    /// The factor `inner` nested in `outer`: named `outer:inner`, with
    /// levels `a:b`, so that the same inner label under two outer levels is
    /// two levels (lme4's `(1 | outer/inner)` is `outer` and this), and
    /// `inner`'s terms.
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
            intercept: inner.intercept,
            slopes: inner.slopes.clone(),
        })
    }

    /// The terms' names: the intercept first, then the slopes.
    pub fn terms(&self) -> Vec<String> {
        let mut terms = Vec::with_capacity(self.slopes.len() + 1);
        if self.intercept {
            terms.push(INTERCEPT.to_string());
        }
        terms.extend(self.slopes.iter().map(|(n, _)| n.clone()));
        terms
    }

    /// The factor's layout on `n` rows: its sorted levels, each row's
    /// level and each term's value on every row. For the sampler in
    /// `prospicio-bayes`, which needs the fit's own layout.
    #[doc(hidden)]
    pub fn layout(&self, n: usize) -> Result<FactorLayout> {
        let columns = self.columns(n)?;
        let set: BTreeSet<&String> = self.labels.iter().collect();
        let levels: Vec<String> = set.into_iter().cloned().collect();
        let lookup: HashMap<&str, usize> = levels
            .iter()
            .enumerate()
            .map(|(j, l)| (l.as_str(), j))
            .collect();
        Ok(FactorLayout {
            terms: self.terms(),
            level: self.labels.iter().map(|l| lookup[l.as_str()]).collect(),
            levels,
            columns,
        })
    }

    /// Each term's value on every row, checked against `n` rows.
    fn columns(&self, n: usize) -> Result<Vec<Vec<f64>>> {
        if self.labels.len() != n {
            return Err(Error::Data(format!(
                "grouping factor {} has {} labels for {n} rows",
                self.name,
                self.labels.len()
            )));
        }
        let mut columns = Vec::with_capacity(self.slopes.len() + 1);
        if self.intercept {
            columns.push(vec![1.0; n]);
        }
        for (name, values) in &self.slopes {
            if name == INTERCEPT {
                return Err(Error::Data(format!(
                    "grouping factor {} has a slope named {INTERCEPT}",
                    self.name
                )));
            }
            if values.len() != n {
                return Err(Error::Data(format!(
                    "slope {name} of grouping factor {} has {} values for {n} rows",
                    self.name,
                    values.len()
                )));
            }
            if let Some(bad) = values.iter().find(|v| !v.is_finite()) {
                return Err(Error::Data(format!(
                    "slope {name} of grouping factor {} has the value {bad}",
                    self.name
                )));
            }
            columns.push(values.clone());
        }
        Ok(columns)
    }
}

/// A grouping factor's layout ([`Grouping::layout`]).
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct FactorLayout {
    pub terms: Vec<String>,
    /// The levels, sorted.
    pub levels: Vec<String>,
    /// Each row's level, an index into `levels`.
    pub level: Vec<usize>,
    /// Each term's value on every row.
    pub columns: Vec<Vec<f64>>,
}

/// How the random effects are integrated out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Integration {
    /// The Laplace approximation (`glmer`'s `nAGQ = 1`, glmmTMB).
    Laplace,
    /// Adaptive Gauss–Hermite quadrature with `nodes` nodes on each
    /// level's effect (`glmer(nAGQ = nodes)`); a single grouping factor
    /// with a single term only.
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
    #[doc(hidden)]
    pub fn power(&self) -> Result<(f64, bool)> {
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
    /// prior weights included) with the random terms of each factor in
    /// `groups`.
    ///
    /// Starts from the GLM without random effects, each standard
    /// deviation 0.5 and each correlation 0. Fails if there is no grouping
    /// factor, a factor has no term, a factor's labels or slopes do not
    /// match the rows, a response is outside the family's range, the
    /// design is collinear, quadrature is asked for with more than one
    /// factor or term or with no nodes, or the optimization cannot leave
    /// its start; a fit that stops at the iteration limit is returned with
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
        if let Some(g) = groups.iter().find(|g| !g.intercept && g.slopes.is_empty()) {
            return Err(Error::Data(format!(
                "grouping factor {} has neither an intercept nor a slope",
                g.name
            )));
        }
        if let Integration::Quadrature { nodes } = self.integration {
            let terms: usize = groups.iter().map(|g| g.terms().len()).sum();
            if groups.len() != 1 || terms != 1 {
                return Err(Error::Data(format!(
                    "quadrature takes a single grouping factor with a single term, not {} factors with {terms} terms; use the Laplace approximation",
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
        let p = design.n_cols();
        let mut factors = Vec::with_capacity(groups.len());
        let mut levels = Vec::with_capacity(groups.len());
        let (mut start, mut theta) = (0, p);
        for g in groups {
            let layout = g.layout(n)?;
            let d = layout.columns.len();
            let m = layout.levels.len();
            factors.push(Factor {
                d,
                level: layout.level,
                n_levels: m,
                columns: layout.columns,
                start,
                theta,
            });
            start += m * d;
            theta += d * (d + 1) / 2;
            levels.push(layout.levels);
        }
        let q = start;
        let n_theta = theta - p;

        // The GLM without random effects starts β and φ.
        let glm = Glm::new(self.family, Link::Log).dispersion(if free_dispersion {
            Dispersion::Pearson
        } else {
            Dispersion::Fixed(1.0)
        });
        let start_fit = glm.fit(design, y)?;
        let mut x0 = start_fit.coefficients().to_vec();
        for f in &factors {
            x0.extend(std::iter::repeat_n(0.5f64.ln(), f.d));
            x0.extend(std::iter::repeat_n(0.0, f.d * (f.d - 1) / 2));
        }
        if free_dispersion {
            x0.push(start_fit.dispersion().ln());
        }

        let problem = Problem {
            design,
            y,
            power,
            free_dispersion,
            factors: &factors,
            n_theta,
            q,
            integration: self.integration,
            nodes: match self.integration {
                Integration::Quadrature { nodes } => gauss_hermite(nodes),
                Integration::Laplace => (Vec::new(), Vec::new()),
            },
        };
        // Each evaluation's inner iteration starts at the last one's mode.
        let start_v = match problem.marginal(&x0, &vec![0.0; q], false) {
            Ok(m) if m.log_likelihood.is_finite() => m.v,
            _ => {
                return Err(Error::Data(
                    "the marginal likelihood is not finite at the GLM's estimates".into(),
                ));
            }
        };
        let warm = RefCell::new(start_v);
        let mut objective = |x: &[f64]| -> f64 {
            let m = problem.marginal(x, &warm.borrow(), false);
            match m {
                Ok(m) if m.log_likelihood.is_finite() => {
                    *warm.borrow_mut() = m.v;
                    -m.log_likelihood
                }
                _ => f64::INFINITY,
            }
        };
        let (x, iterations, converged) =
            minimize(&mut objective, x0, self.tolerance, self.max_iterations);

        let v = warm.borrow().clone();
        let at = problem.marginal(&x, &v, true)?;
        let hessian = fd_hessian(&mut objective, &x, at.log_likelihood.abs());
        let d = x.len();
        // A standard deviation at the boundary (below 1e-4) leaves the
        // likelihood flat in its log, and in the correlations of its term:
        // the other parameters' covariance is then taken with them held,
        // and their own standard errors are NaN.
        let mut boundary = vec![false; d];
        for f in &factors {
            let at_zero: Vec<bool> = (0..f.d).map(|c| x[f.theta + c] < 1e-4f64.ln()).collect();
            for c in 0..f.d {
                boundary[f.theta + c] = at_zero[c];
            }
            let mut a = f.theta + f.d;
            for r in 1..f.d {
                for c in 0..r {
                    boundary[a] = at_zero[r] || at_zero[c];
                    a += 1;
                }
            }
        }
        let free: Vec<usize> = (0..d).filter(|&a| !boundary[a]).collect();
        let nf = free.len();
        let sub: Vec<f64> = free
            .iter()
            .flat_map(|&a| free.iter().map(move |&b| (a, b)))
            .map(|(a, b)| hessian[a * d + b])
            .collect();
        let mut covariance = vec![f64::NAN; d * d];
        if let Some(l) = cholesky(&sub, nf) {
            let inv = cholesky_inverse(&l, nf);
            for (i, &a) in free.iter().enumerate() {
                for (j, &b) in free.iter().enumerate() {
                    covariance[a * d + b] = inv[i * nf + j];
                }
            }
        }

        // ∂b̂/∂(β, θ, log φ) by central differences, for the prediction
        // variance.
        let mut jac = vec![vec![0.0; d]; q];
        for a in 0..d {
            let h = 1e-5 * x[a].abs().max(1.0);
            let mut xp = x.clone();
            xp[a] += h;
            let up = problem.marginal(&xp, &at.v, false)?.b;
            xp[a] = x[a] - h;
            let um = problem.marginal(&xp, &at.v, false)?.b;
            for j in 0..q {
                jac[j][a] = (up[j] - um[j]) / (2.0 * h);
            }
        }
        let mut random_effects = Vec::with_capacity(factors.len());
        let (mut group_names, mut term_names) = (Vec::new(), Vec::new());
        let (mut sd, mut sd_standard_errors) = (Vec::new(), Vec::new());
        let mut correlations = Vec::with_capacity(factors.len());
        for ((g, names), f) in groups.iter().zip(&levels).zip(&factors) {
            let m = names.len();
            let terms = g.terms();
            let mut values = vec![vec![0.0; m]; f.d];
            let mut conditional_variance = vec![vec![0.0; m]; f.d];
            let mut prediction_variance = vec![vec![0.0; m]; f.d];
            for l in 0..m {
                for c in 0..f.d {
                    let j = f.start + l * f.d + c;
                    let row = &jac[j];
                    let extra: f64 = free
                        .iter()
                        .map(|&a| {
                            free.iter()
                                .map(|&b| row[a] * covariance[a * d + b] * row[b])
                                .sum::<f64>()
                        })
                        .sum();
                    values[c][l] = at.b[j];
                    conditional_variance[c][l] = at.conditional_variance[j];
                    prediction_variance[c][l] = at.conditional_variance[j] + extra;
                }
            }
            for (c, term) in terms.iter().enumerate() {
                let s = x[f.theta + c].exp();
                group_names.push(g.name.clone());
                term_names.push(term.clone());
                sd.push(s);
                sd_standard_errors.push(s * covariance[(f.theta + c) * d + f.theta + c].sqrt());
            }
            correlations.push(correlation(
                &x[f.theta + f.d..f.theta + f.d * (f.d + 1) / 2],
                f.d,
            ));
            random_effects.push(RandomEffects {
                name: g.name.clone(),
                terms,
                levels: names.clone(),
                values,
                conditional_variance,
                prediction_variance,
            });
        }

        let (dispersion, dispersion_standard_error) = if free_dispersion {
            let phi = x[p + n_theta].exp();
            (
                phi,
                phi * covariance[(p + n_theta) * d + p + n_theta].sqrt(),
            )
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
            group_names,
            term_names,
            sd,
            sd_standard_errors,
            correlations,
            parameters: x.clone(),
            parameter_covariance: covariance,
            dispersion,
            dispersion_standard_error,
            log_likelihood: at.log_likelihood,
            n_obs: n,
            fitted: at.mu,
            random_effects,
            iterations,
            converged,
        })
    }
}

/// The unit lower triangle `L` with the strictly lower entries `theta`,
/// row by row, and its rows normalized: the Cholesky factor of the
/// correlation matrix, row-major.
#[doc(hidden)]
pub fn correlation_factor(theta: &[f64], d: usize) -> Vec<f64> {
    let mut l = vec![0.0; d * d];
    let mut a = 0;
    for r in 0..d {
        l[r * d + r] = 1.0;
        for c in 0..r {
            l[r * d + c] = theta[a];
            a += 1;
        }
        let norm = l[r * d..r * d + r + 1]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        for c in 0..=r {
            l[r * d + c] /= norm;
        }
    }
    l
}

/// The correlation matrix of glmmTMB's `us` parameters `theta`,
/// row-major.
fn correlation(theta: &[f64], d: usize) -> Vec<f64> {
    let l = correlation_factor(theta, d);
    (0..d * d)
        .map(|ab| {
            let (a, b) = (ab / d, ab % d);
            (0..d).map(|c| l[a * d + c] * l[b * d + c]).sum()
        })
        .collect()
}

/// The predicted random effects of one grouping factor.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomEffects {
    /// The grouping factor's name.
    pub name: String,
    /// Its terms: [`INTERCEPT`] first if it has one, then its slopes.
    pub terms: Vec<String>,
    /// Its levels, sorted.
    pub levels: Vec<String>,
    /// `values[c][l]`: term `c`'s predicted effect `b̂` at level `l`, the
    /// conditional mode.
    pub values: Vec<Vec<f64>>,
    /// The variance of each effect given the data at the estimates, the
    /// diagonal of `Λ H⁻¹ Λᵀ` (lme4's `condVar`), indexed as `values`.
    pub conditional_variance: Vec<Vec<f64>>,
    /// The conditional variance plus the part from the estimates'
    /// uncertainty, `Jⱼ C Jⱼᵀ` (glmmTMB's `condVar`).
    pub prediction_variance: Vec<Vec<f64>>,
}

/// A fitted GLMM.
#[derive(Debug, Clone, PartialEq)]
pub struct GlmmFit {
    spec: Glmm,
    names: Vec<String>,
    coefficients: Vec<f64>,
    covariance: Vec<f64>,
    group_names: Vec<String>,
    term_names: Vec<String>,
    sd: Vec<f64>,
    sd_standard_errors: Vec<f64>,
    correlations: Vec<Vec<f64>>,
    parameters: Vec<f64>,
    parameter_covariance: Vec<f64>,
    dispersion: f64,
    dispersion_standard_error: f64,
    log_likelihood: f64,
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
    /// of `−log L` in `(β, θ, log φ)`.
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// The standard errors of `β`.
    pub fn standard_errors(&self) -> Vec<f64> {
        let p = self.coefficients.len();
        (0..p).map(|j| self.covariance[j * p + j].sqrt()).collect()
    }

    /// Each random term's grouping factor, in the order fitted, a factor
    /// repeated for each of its terms (parallel to [`sd`](Self::sd)).
    pub fn group_names(&self) -> &[String] {
        &self.group_names
    }

    /// Each random term's name ([`INTERCEPT`] or a slope's covariate),
    /// parallel to [`group_names`](Self::group_names).
    pub fn term_names(&self) -> &[String] {
        &self.term_names
    }

    /// Each random term's standard deviation.
    pub fn sd(&self) -> &[f64] {
        &self.sd
    }

    /// The standard errors of the standard deviations, by the delta method
    /// from their logs.
    pub fn sd_standard_errors(&self) -> &[f64] {
        &self.sd_standard_errors
    }

    /// Each grouping factor's correlation matrix of its terms, row-major
    /// (`[1.0]` for a factor with one term).
    pub fn correlations(&self) -> &[Vec<f64>] {
        &self.correlations
    }

    /// Every parameter on the scale maximized: `β`, each factor's log
    /// standard deviations and correlation parameters (glmmTMB's `theta`),
    /// and `log φ` if estimated.
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// The covariance of [`parameters`](Self::parameters), row-major (NaN
    /// in the rows of parameters at the boundary).
    pub fn parameter_covariance(&self) -> &[f64] {
        &self.parameter_covariance
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

    /// `−2 log L + 2 k` with `k` the parameters: the fixed effects, each
    /// factor's standard deviations and correlations, and `φ` if
    /// estimated.
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * self.parameters.len() as f64
    }

    /// `−2 log L + k log n`.
    pub fn bic(&self) -> f64 {
        -2.0 * self.log_likelihood + self.parameters.len() as f64 * (self.n_obs as f64).ln()
    }

    /// The conditional means `μᵢ` of the training rows, at `b̂`.
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// The predicted random effects, one entry per grouping factor.
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

    /// Means for new rows: `exp(xᵀ β + offset + Σ zᵀ b̂)`, each factor's
    /// effects looked up by `groups` (matched by name; a level the fit has
    /// not seen, or a factor not given, contributes 0, the population
    /// mean). The factors given under a name must carry the slopes of the
    /// fitted factors of that name, by covariate. An empty `groups`
    /// predicts the population means.
    pub fn predict(&self, design: &Design, groups: &[Grouping]) -> Result<Vec<f64>> {
        if design.names() != self.names.as_slice() {
            return Err(Error::Data(
                "the design's columns differ from the fit's".into(),
            ));
        }
        let mut eta = design.linear_predictor(&self.coefficients);
        let names: Vec<&str> = self
            .random_effects
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        check_groups(groups, &names, eta.len())?;
        for re in &self.random_effects {
            let Some(given) = given_effects(&re.name, &re.terms, &re.levels, groups, eta.len())?
            else {
                continue;
            };
            for (i, e) in eta.iter_mut().enumerate() {
                if let Some(j) = given.level[i] {
                    for (c, column) in given.columns.iter().enumerate() {
                        *e += re.values[c][j] * column[i];
                    }
                }
            }
        }
        Ok(eta.into_iter().map(f64::exp).collect())
    }
}

/// Checks the factors given for a prediction on `n` rows: each has `n`
/// labels and a name among the fitted factors' `names`.
#[doc(hidden)]
pub fn check_groups(groups: &[Grouping], names: &[&str], n: usize) -> Result<()> {
    for g in groups {
        if g.labels.len() != n {
            return Err(Error::Data(format!(
                "grouping factor {} has {} labels for {n} rows",
                g.name,
                g.labels.len()
            )));
        }
        if !names.contains(&g.name.as_str()) {
            return Err(Error::Data(format!(
                "the fit has no grouping factor {}",
                g.name
            )));
        }
    }
    Ok(())
}

/// A fitted factor's effects looked up for new rows
/// ([`given_effects`]).
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct GivenEffects {
    /// Each row's level among the fitted ones, `None` for a label the fit
    /// has not seen.
    pub level: Vec<Option<usize>>,
    /// Each term's value on every row.
    pub columns: Vec<Vec<f64>>,
}

/// The rows' levels and term values of the fitted factor `name` (with
/// `terms` and sorted `levels`) from the factors given under that name:
/// the first one's labels, a slope's values from whichever carries them.
/// `None` when no factor of that name is given.
#[doc(hidden)]
pub fn given_effects(
    name: &str,
    terms: &[String],
    levels: &[String],
    groups: &[Grouping],
    n: usize,
) -> Result<Option<GivenEffects>> {
    let given: Vec<&Grouping> = groups.iter().filter(|g| g.name == name).collect();
    let Some(first) = given.first() else {
        return Ok(None);
    };
    let lookup: HashMap<&str, usize> = levels
        .iter()
        .enumerate()
        .map(|(j, l)| (l.as_str(), j))
        .collect();
    let mut columns = Vec::with_capacity(terms.len());
    for term in terms {
        if term == INTERCEPT {
            columns.push(vec![1.0; n]);
            continue;
        }
        let values = given
            .iter()
            .flat_map(|g| &g.slopes)
            .find(|(s, _)| s == term)
            .map(|(_, v)| v)
            .ok_or_else(|| {
                Error::Data(format!(
                    "grouping factor {name} has no values for its slope {term}"
                ))
            })?;
        if values.len() != n {
            return Err(Error::Data(format!(
                "slope {term} of grouping factor {name} has {} values for {n} rows",
                values.len()
            )));
        }
        columns.push(values.clone());
    }
    Ok(Some(GivenEffects {
        level: first
            .labels
            .iter()
            .map(|l| lookup.get(l.as_str()).copied())
            .collect(),
        columns,
    }))
}

/// One grouping factor's layout.
struct Factor {
    /// Its number of terms.
    d: usize,
    /// Each row's level.
    level: Vec<usize>,
    n_levels: usize,
    /// Each term's value on every row.
    columns: Vec<Vec<f64>>,
    /// Where its effects start in `v`: level `l`'s term `c` is
    /// `start + l d + c`.
    start: usize,
    /// Where its parameters start in `x`: `d` log standard deviations,
    /// then `d (d − 1) / 2` correlation parameters.
    theta: usize,
}

impl Factor {
    /// `Λ`, row-major lower triangular: the correlation factor's rows
    /// scaled by the standard deviations.
    fn lambda(&self, x: &[f64]) -> Option<Vec<f64>> {
        let d = self.d;
        let mut l = correlation_factor(&x[self.theta + d..self.theta + d * (d + 1) / 2], d);
        for r in 0..d {
            let s = x[self.theta + r].exp();
            if !(s.is_finite() && s > 0.0) {
                return None;
            }
            for c in 0..=r {
                l[r * d + c] *= s;
            }
        }
        l.iter().all(|v| v.is_finite()).then_some(l)
    }
}

/// The data and settings shared by every evaluation.
struct Problem<'a> {
    design: &'a Design,
    y: &'a [f64],
    power: f64,
    free_dispersion: bool,
    factors: &'a [Factor],
    /// The factors' parameters in all.
    n_theta: usize,
    /// The random effects in all.
    q: usize,
    integration: Integration,
    /// Gauss–Hermite nodes and log weights (`log w + x²`).
    nodes: (Vec<f64>, Vec<f64>),
}

/// The joint log density `g(v)` of the data and the random effects, its
/// gradient and minus its Hessian in `v` (diagonal with one factor of one
/// term), and the conditional means.
struct Joint {
    g: f64,
    grad: Vec<f64>,
    hess: Vec<f64>,
    mu: Vec<f64>,
}

/// One evaluation of the marginal likelihood.
struct Marginal {
    log_likelihood: f64,
    /// The mode `v̂`.
    v: Vec<f64>,
    /// The effects `b̂ = Λ v̂`.
    b: Vec<f64>,
    /// The diagonal of `Λ H⁻¹ Λᵀ` (empty unless asked for).
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

    /// The marginal log-likelihood at `x = (β, θ, log φ)`, with the inner
    /// Newton iteration started at `v0`; the conditional variances only
    /// `with_variance`.
    fn marginal(&self, x: &[f64], v0: &[f64], with_variance: bool) -> Result<Marginal> {
        let p = self.design.n_cols();
        let beta = &x[..p];
        let phi = if self.free_dispersion {
            x[p + self.n_theta].exp()
        } else {
            1.0
        };
        let out_of_range = || Error::Data("parameters out of range".into());
        if !(phi.is_finite() && phi > 0.0) {
            return Err(out_of_range());
        }
        let lambdas: Vec<Vec<f64>> = self
            .factors
            .iter()
            .map(|f| f.lambda(x))
            .collect::<Option<_>>()
            .ok_or_else(out_of_range)?;
        let eta0 = self.design.linear_predictor(beta);
        let n = eta0.len();
        let q = self.q;
        // Each row's effects and their coefficients `(Λᵀ z)_c` in `η`.
        let rows: Vec<Vec<(usize, f64)>> = (0..n)
            .map(|i| {
                let mut row = Vec::new();
                for (f, lam) in self.factors.iter().zip(&lambdas) {
                    let base = f.start + f.level[i] * f.d;
                    for c in 0..f.d {
                        let a: f64 = (c..f.d).map(|r| f.columns[r][i] * lam[r * f.d + c]).sum();
                        row.push((base + c, a));
                    }
                }
                row
            })
            .collect();
        let diagonal = self.factors.len() == 1 && self.factors[0].d == 1;
        let mut v = v0.to_vec();

        // g(v), its gradient and minus its Hessian at v.
        let evaluate = |v: &[f64]| -> Option<Joint> {
            let mut g = 0.0;
            let mut grad = vec![0.0; q];
            let mut hess = vec![0.0; if diagonal { q } else { q * q }];
            let mut mu = Vec::with_capacity(n);
            for (i, (row, e0)) in rows.iter().zip(&eta0).enumerate() {
                let eta = e0 + row.iter().map(|&(j, a)| a * v[j]).sum::<f64>();
                let (ll, s, h) = self.row(i, eta, phi);
                if !ll.is_finite() {
                    return None;
                }
                g += ll;
                mu.push(eta.exp());
                for &(j, a) in row {
                    grad[j] += s * a;
                    if diagonal {
                        hess[j] += h * a * a;
                    } else {
                        for &(k, b) in row {
                            hess[j * q + k] += h * a * b;
                        }
                    }
                }
            }
            for j in 0..q {
                g -= v[j] * v[j] / 2.0;
                grad[j] -= v[j];
                if diagonal {
                    hess[j] += 1.0;
                } else {
                    hess[j * q + j] += 1.0;
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

        let mut state = match evaluate(&v) {
            Some(s) => s,
            None => {
                v = vec![0.0; q];
                evaluate(&v).ok_or_else(|| Error::Data("the joint density is not finite".into()))?
            }
        };
        for _ in 0..200 {
            let (step, _) = solve(&state.hess, &state.grad)
                .ok_or_else(|| Error::Data("the random effects' Hessian is singular".into()))?;
            let size = step.iter().fold(0.0f64, |m, s| m.max(s.abs()));
            let mut t = 1.0;
            let mut moved = false;
            for _ in 0..50 {
                let trial: Vec<f64> = v.iter().zip(&step).map(|(a, b)| a + t * b).collect();
                if let Some(next) = evaluate(&trial) {
                    if next.g >= state.g - 1e-12 * state.g.abs().max(1.0) {
                        v = trial;
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
        let log_det = if diagonal {
            factor.iter().map(|h| h.ln()).sum::<f64>()
        } else {
            2.0 * (0..q).map(|j| factor[j * q + j].ln()).sum::<f64>()
        };
        // b = Λ v, level by level.
        let mut b = vec![0.0; q];
        for (f, lam) in self.factors.iter().zip(&lambdas) {
            for l in 0..f.n_levels {
                let base = f.start + l * f.d;
                for r in 0..f.d {
                    b[base + r] = (0..=r).map(|c| lam[r * f.d + c] * v[base + c]).sum();
                }
            }
        }
        // The inverse costs as much as the rest of an evaluation: only the
        // final one needs it.
        let conditional_variance = if !with_variance {
            Vec::new()
        } else if diagonal {
            let lam = lambdas[0][0];
            factor.iter().map(|h| lam * lam / h).collect()
        } else {
            let inv = cholesky_inverse(&factor, q);
            let mut out = vec![0.0; q];
            for (f, lam) in self.factors.iter().zip(&lambdas) {
                for l in 0..f.n_levels {
                    let base = f.start + l * f.d;
                    for r in 0..f.d {
                        let mut s = 0.0;
                        for a in 0..=r {
                            for c in 0..=r {
                                s += lam[r * f.d + a]
                                    * inv[(base + a) * q + base + c]
                                    * lam[r * f.d + c];
                            }
                        }
                        out[base + r] = s;
                    }
                }
            }
            out
        };
        let log_likelihood = match self.integration {
            Integration::Laplace => g - 0.5 * log_det,
            Integration::Quadrature { .. } => self.quadrature(&eta0, &rows, &v, &hess, phi)?,
        };
        Ok(Marginal {
            log_likelihood,
            v,
            b,
            conditional_variance,
            mu,
        })
    }

    /// Adaptive Gauss–Hermite quadrature on each level of the single
    /// factor's single term, centred at the mode `v` and scaled by
    /// `hess^(-1/2)`.
    fn quadrature(
        &self,
        eta0: &[f64],
        rows: &[Vec<(usize, f64)>],
        v: &[f64],
        hess: &[f64],
        phi: f64,
    ) -> Result<f64> {
        let q = v.len();
        let mut members = vec![Vec::new(); q];
        for (i, row) in rows.iter().enumerate() {
            members[row[0].0].push((i, row[0].1));
        }
        let (x, lw) = &self.nodes;
        let mut total = 0.0;
        for j in 0..q {
            let tau = (2.0 / hess[j]).sqrt();
            let terms: Vec<f64> = x
                .iter()
                .zip(lw)
                .map(|(xk, lwk)| {
                    let t = v[j] + tau * xk;
                    let ll: f64 = members[j]
                        .iter()
                        .map(|&(i, a)| self.row(i, eta0[i] + a * t, phi).0)
                        .sum();
                    lwk + ll - t * t / 2.0
                })
                .collect();
            let top = terms.iter().fold(f64::NEG_INFINITY, |m, &t| m.max(t));
            if !top.is_finite() {
                return Err(Error::Data(
                    "a quadrature node's density is not finite".into(),
                ));
            }
            let sum: f64 = terms.iter().map(|t| (t - top).exp()).sum();
            total += top + sum.ln() + tau.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
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
            assert!(re.conditional_variance[0][j] > 0.0);
            assert!(re.prediction_variance[0][j] > re.conditional_variance[0][j]);
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
        assert!(effects.values[0].iter().all(|v| v.abs() < 1e-4));
        assert!(effects.prediction_variance[0].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn slopes_predict_by_label_and_covariate() {
        let (design, labels, y) = book();
        let x = design.column(1).to_vec();
        let g = [Grouping::new("g", labels.clone()).slope("x", x.clone())];
        let fit = Glmm::new(Family::Poisson).fit(&design, &g, &y).unwrap();
        assert!(fit.converged());
        assert_eq!(fit.group_names(), ["g", "g"]);
        assert_eq!(fit.term_names(), [INTERCEPT, "x"]);
        assert_eq!(fit.parameters().len(), 2 + 3);
        let re = &fit.random_effects()[0];
        assert_eq!(re.terms, [INTERCEPT, "x"]);
        assert_eq!((re.values.len(), re.values[1].len()), (2, 8));
        let r = &fit.correlations()[0];
        assert!(r[0] == 1.0 && r[3] == 1.0 && (r[1] - r[2]).abs() < 1e-15 && r[1].abs() < 1.0);
        // The training rows' means are the predictions at their labels and
        // covariates.
        let mu = fit.predict(&design, &g).unwrap();
        for (a, b) in mu.iter().zip(fit.fitted()) {
            assert!((a - b).abs() < 1e-9 * b);
        }
        // A factor given without the slope's values is an error.
        let bare = [Grouping::new("g", labels.clone())];
        assert!(
            fit.predict(&design, &bare)
                .unwrap_err()
                .to_string()
                .contains("slope x")
        );
        // The slope adds a parameter for its sd and one for the
        // correlation, and cannot lower the likelihood.
        let intercepts = Glmm::new(Family::Poisson)
            .fit(&design, &[Grouping::new("g", labels.clone())], &y)
            .unwrap();
        assert!(fit.log_likelihood() >= intercepts.log_likelihood() - 1e-6);
        // Independent terms: a factor for each, on the same labels.
        let split = [
            Grouping::new("g", labels.clone()),
            Grouping::new("g", labels).without_intercept().slope("x", x),
        ];
        let diag = Glmm::new(Family::Poisson).fit(&design, &split, &y).unwrap();
        assert_eq!(diag.parameters().len(), 2 + 2);
        assert!(diag.log_likelihood() <= fit.log_likelihood() + 1e-6);
        let mu = diag.predict(&design, &split).unwrap();
        for (a, b) in mu.iter().zip(diag.fitted()) {
            assert!((a - b).abs() < 1e-9 * b);
        }
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
            .contains("single grouping factor with a single term")
        );
        assert!(
            Glmm::new(Family::Poisson)
                .quadrature(0)
                .fit(&design, &g, &y)
                .is_err()
        );
        let slope = [g[0].clone().slope("x", vec![0.5; 40])];
        assert!(
            msg(Glmm::new(Family::Poisson)
                .quadrature(5)
                .fit(&design, &slope, &y))
            .contains("2 terms")
        );
        let none = [g[0].clone().without_intercept()];
        assert!(
            msg(Glmm::new(Family::Poisson).fit(&design, &none, &y))
                .contains("neither an intercept nor a slope")
        );
        let ragged = [g[0].clone().slope("x", vec![0.5; 3])];
        assert!(
            msg(Glmm::new(Family::Poisson).fit(&design, &ragged, &y))
                .contains("3 values for 40 rows")
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
