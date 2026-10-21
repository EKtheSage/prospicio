//! Bayesian GLMMs sampled with NUTS (`docs/design/models.md`, "Mixed
//! models").
//!
//! [`BayesGlmm`] is the model of [`prospicio_glm::glmm`] (the Poisson,
//! gamma or Tweedie with the log link, random intercepts and slopes by
//! level of each grouping factor) with priors on its parameters, sampled
//! jointly with the random effects instead of integrating them out:
//!
//! - each fixed coefficient `N(0, σ²)`, `σ = intercept_sd` for an
//!   all-ones column and `prior_sd` for the rest, as [`crate::glm`];
//! - each random term's standard deviation half-normal with scale
//!   `sd_scale`;
//! - each factor's correlation matrix LKJ(`lkj`) (uniform at 1);
//! - for the gamma and Tweedie, the dispersion `φ` half-normal with scale
//!   `dispersion_scale`.
//!
//! The effects are sampled non-centred, `b_{k, l} = Λₖ v_{k, l}` with
//! `v ~ N(0, I)`, which keeps NUTS out of the funnel between a standard
//! deviation and its effects. The parameters are those of the Laplace fit
//! ([`prospicio_glm::glmm::Glmm`]): `β`, each factor's log standard
//! deviations and glmmTMB's `us` correlation parameters `θ`, `log φ` when
//! it is free, then `v`, factor by factor, level by level, term by term.
//! The chains start at the Laplace fit, its effects mapped to `v`.
//!
//! In `θ` the LKJ density is `−(d + 2η − 1) Σ_{r ≥ 1} log nᵣ`, with
//! `nᵣ = √(1 + |θᵣ|²)` for row `r`'s parameters `θᵣ`: row `r` of the
//! correlation's Cholesky factor `L` is `(θᵣ, 1) / nᵣ`, Stan's
//! `lkj_corr_cholesky` density on the strictly lower entries of `L` is
//! `Σ_{r ≥ 1} (d − r + 2η − 3) log L_rr` (rows from 0), `L_rr = 1 / nᵣ`,
//! and the map `θᵣ ↦ θᵣ / nᵣ` has Jacobian determinant `nᵣ^−(r + 2)`.
//! Each correlation then has the LKJ marginal, a beta on `(−1, 1)` with
//! variance `1 / (2η + d − 1)` (Lewandowski, Kurowicka and Joe, 2009),
//! which the tests check.
//!
//! The gradient is analytic but for the Tweedie's in `log φ`, a central
//! difference of its series density.
//!
//! ```
//! use prospicio_bayes::glmm::{BayesGlmm, Sampler};
//! use prospicio_glm::glmm::Grouping;
//! use prospicio_models::{Design, Family};
//!
//! // Claim counts for 40 policies in eight territories.
//! let labels: Vec<String> = (0..40).map(|i| format!("t{}", i % 8)).collect();
//! let y: Vec<f64> = (0..40).map(|i| [2.0, 3.0, 5.0, 1.0, 4.0, 2.0, 6.0, 3.0][i % 8]).collect();
//! let design = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; 40]]).unwrap();
//! let spec = BayesGlmm::new(Family::Poisson).sampler(Sampler {
//!     chains: 2,
//!     tune: 500,
//!     draws: 500,
//!     ..Sampler::default()
//! });
//! let fit = spec.fit(&design, &[Grouping::new("territory", labels)], &y).unwrap();
//! let summary = fit.summary().unwrap();
//! assert_eq!(summary[1].name, "sd[territory|(Intercept)]");
//! assert!(summary.iter().all(|s| s.rhat < 1.05));
//! // The posterior mean of the intercept is near the Laplace fit's.
//! assert!((summary[0].mean - fit.laplace().coefficients()[0]).abs() < 0.15);
//! ```

use prospicio_core::{Error, Result};
use prospicio_glm::glmm::{
    FactorLayout, Glmm, GlmmFit, Grouping, check_groups, correlation_factor, given_effects,
};
use prospicio_models::{Design, Family};

use crate::glm::dll_dphi;
use crate::nuts::{LogDensity, sample};
pub use crate::nuts::{ParamSummary, PosteriorDraws, Sampler};

/// A Bayesian GLMM: the family (log link), the priors and the sampler.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesGlmm {
    /// Poisson, gamma, or Tweedie with power in `(1, 2)`.
    pub family: Family,
    /// The prior standard deviation of each fixed coefficient but the
    /// intercept, on the log scale.
    pub prior_sd: f64,
    /// The prior standard deviation of the intercept (an all-ones column).
    pub intercept_sd: f64,
    /// The scale of the half-normal prior on each random term's standard
    /// deviation, on the log scale.
    pub sd_scale: f64,
    /// The shape `η` of the LKJ prior on each factor's correlations.
    pub lkj: f64,
    /// The scale of the half-normal prior on `φ` (gamma and Tweedie).
    pub dispersion_scale: f64,
    pub sampler: Sampler,
}

impl BayesGlmm {
    /// Priors `N(0, 2.5²)` on the coefficients and `N(0, 10²)` on the
    /// intercept, half-normal(1) on each standard deviation, LKJ(1) on the
    /// correlations, half-normal(10) on `φ`, and the default sampler.
    pub fn new(family: Family) -> Self {
        Self {
            family,
            prior_sd: 2.5,
            intercept_sd: 10.0,
            sd_scale: 1.0,
            lkj: 1.0,
            dispersion_scale: 10.0,
            sampler: Sampler::default(),
        }
    }

    /// The same model with other sampler settings.
    pub fn sampler(mut self, sampler: Sampler) -> Self {
        self.sampler = sampler;
        self
    }

    fn check(&self) -> Result<()> {
        for (name, v) in [
            ("prior_sd", self.prior_sd),
            ("intercept_sd", self.intercept_sd),
            ("sd_scale", self.sd_scale),
            ("lkj", self.lkj),
            ("dispersion_scale", self.dispersion_scale),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::InvalidParameter {
                    name,
                    value: v,
                    reason: "must be finite and positive",
                });
            }
        }
        self.sampler.validate()
    }

    /// Samples the posterior of the GLMM of `y` on `design` (its offset
    /// and prior weights included) with the random terms of each factor in
    /// `groups`.
    ///
    /// Fits the Laplace approximation first (with its checks of the
    /// inputs) and starts every chain there, a standard deviation below
    /// 0.05 raised to 0.05 and a correlation parameter kept within ±5, so
    /// that a fit at the boundary does not start the chains in the neck of
    /// the funnel.
    pub fn fit(&self, design: &Design, groups: &[Grouping], y: &[f64]) -> Result<BayesGlmmFit> {
        self.check()?;
        let laplace = Glmm::new(self.family).fit(design, groups, y)?;
        let (power, free_dispersion) = Glmm::new(self.family).power()?;
        let n = design.n_rows();
        let p = design.n_cols();
        let layouts: Vec<FactorLayout> =
            groups.iter().map(|g| g.layout(n)).collect::<Result<_>>()?;
        let mut factors = Vec::with_capacity(layouts.len());
        let (mut start, mut theta) = (0, p);
        for layout in layouts {
            let d = layout.columns.len();
            let m = layout.levels.len();
            factors.push(Factor {
                d,
                layout,
                start,
                theta,
            });
            start += m * d;
            theta += d * (d + 1) / 2;
        }
        let q = start;
        let n_theta = theta - p;
        let v_at = p + n_theta + usize::from(free_dispersion);
        let prior_sd: Vec<f64> = (0..p)
            .map(|j| {
                if design.column(j).iter().all(|&v| v == 1.0) {
                    self.intercept_sd
                } else {
                    self.prior_sd
                }
            })
            .collect();

        // The start: the Laplace fit, its effects as v = Λ⁻¹ b̂.
        let mut x0 = laplace.parameters().to_vec();
        for f in &factors {
            for c in 0..f.d {
                x0[f.theta + c] = x0[f.theta + c].max(0.05f64.ln());
            }
            for t in &mut x0[f.theta + f.d..f.theta + f.d * (f.d + 1) / 2] {
                *t = t.clamp(-5.0, 5.0);
            }
        }
        x0.resize(v_at + q, 0.0);
        for (f, re) in factors.iter().zip(laplace.random_effects()) {
            let lambda = f.lambda(&x0).ok_or_else(|| {
                Error::Data("the Laplace fit's covariance is out of range".into())
            })?;
            for l in 0..f.layout.levels.len() {
                // Forward substitution in the lower triangle Λ.
                for r in 0..f.d {
                    let done: f64 = (0..r)
                        .map(|c| lambda[r * f.d + c] * x0[v_at + f.start + l * f.d + c])
                        .sum();
                    x0[v_at + f.start + l * f.d + r] =
                        (re.values[r][l] - done) / lambda[r * f.d + r];
                }
            }
        }

        let density = Posterior {
            design,
            y,
            family: self.family,
            power,
            free_dispersion,
            factors: &factors,
            n_theta,
            q,
            prior_sd: &prior_sd,
            sd_scale: self.sd_scale,
            lkj: self.lkj,
            dispersion_scale: self.dispersion_scale,
        };
        let draws = sample(&density, &x0, self.sampler)?;

        // The parameters on their natural scale, and the effects b = Λ v.
        let mut names = design.names().to_vec();
        let mut effect_names = Vec::with_capacity(q);
        let mut layout = Vec::with_capacity(factors.len());
        for (g, f) in groups.iter().zip(&factors) {
            for t in &f.layout.terms {
                names.push(format!("sd[{}|{t}]", g.name));
            }
            for r in 1..f.d {
                for c in 0..r {
                    names.push(format!(
                        "cor[{}|{},{}]",
                        g.name, f.layout.terms[r], f.layout.terms[c]
                    ));
                }
            }
            for level in &f.layout.levels {
                for t in &f.layout.terms {
                    effect_names.push(format!("b[{}|{level}|{t}]", g.name));
                }
            }
            layout.push(EffectsLayout {
                name: g.name.clone(),
                terms: f.layout.terms.clone(),
                levels: f.layout.levels.clone(),
                start: f.start,
                d: f.d,
            });
        }
        if free_dispersion {
            names.push("dispersion".into());
        }
        let parameters = draws.transform(names, |x| {
            let mut out = x[..p].to_vec();
            for f in &factors {
                out.extend(x[f.theta..f.theta + f.d].iter().map(|v| v.exp()));
                let l = correlation_factor(&x[f.theta + f.d..f.theta + f.d * (f.d + 1) / 2], f.d);
                for r in 1..f.d {
                    for c in 0..r {
                        out.push((0..f.d).map(|k| l[r * f.d + k] * l[c * f.d + k]).sum());
                    }
                }
            }
            if free_dispersion {
                out.push(x[p + n_theta].exp());
            }
            out
        })?;
        let effects = draws.transform(effect_names, |x| {
            let mut out = vec![0.0; q];
            for f in &factors {
                // Inside the support every Λ is finite.
                let lambda = f.lambda(x).unwrap_or_else(|| vec![f64::NAN; f.d * f.d]);
                for l in 0..f.layout.levels.len() {
                    let v = &x[v_at + f.start + l * f.d..][..f.d];
                    for r in 0..f.d {
                        out[f.start + l * f.d + r] =
                            (0..=r).map(|c| lambda[r * f.d + c] * v[c]).sum();
                    }
                }
            }
            out
        })?;
        Ok(BayesGlmmFit {
            spec: self.clone(),
            names: design.names().to_vec(),
            parameters,
            effects,
            layout,
            laplace,
        })
    }
}

/// One grouping factor's layout in the parameter vector.
struct Factor {
    d: usize,
    layout: FactorLayout,
    /// Where its effects start in `v`: level `l`'s term `c` is
    /// `start + l d + c`.
    start: usize,
    /// Where its `d` log standard deviations, then its correlation
    /// parameters, start in the parameters.
    theta: usize,
}

impl Factor {
    /// `Λ`, row-major lower triangular, at the parameters `x`.
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

/// The log LKJ(`eta`) density of a `d × d` correlation matrix in glmmTMB's
/// parameters `theta` (the module's note), adding its gradient to `grad`.
fn lkj_log_density(theta: &[f64], d: usize, eta: f64, grad: &mut [f64]) -> f64 {
    let k = d as f64 + 2.0 * eta - 1.0;
    let mut lp = 0.0;
    let mut a = 0;
    for r in 1..d {
        let row = &theta[a..a + r];
        let n2 = 1.0 + row.iter().map(|t| t * t).sum::<f64>();
        lp -= 0.5 * k * n2.ln();
        for (g, t) in grad[a..a + r].iter_mut().zip(row) {
            *g -= k * t / n2;
        }
        a += r;
    }
    lp
}

/// The log posterior density on the unconstrained scale.
struct Posterior<'a> {
    design: &'a Design,
    y: &'a [f64],
    family: Family,
    power: f64,
    free_dispersion: bool,
    factors: &'a [Factor],
    n_theta: usize,
    q: usize,
    prior_sd: &'a [f64],
    sd_scale: f64,
    lkj: f64,
    dispersion_scale: f64,
}

impl Posterior<'_> {
    /// `∂ log f(y; μ, φ / w) / ∂ log φ`.
    fn dll_dlog_phi(&self, y: f64, mu: f64, w: f64, phi: f64) -> f64 {
        match self.family {
            Family::Gamma => phi * dll_dphi(Family::Gamma, y, mu, w, phi),
            family => {
                let h: f64 = 1e-5;
                (family.log_likelihood(y, mu, w, phi * h.exp())
                    - family.log_likelihood(y, mu, w, phi * (-h).exp()))
                    / (2.0 * h)
            }
        }
    }
}

impl LogDensity for Posterior<'_> {
    fn dim(&self) -> usize {
        self.design.n_cols() + self.n_theta + usize::from(self.free_dispersion) + self.q
    }

    fn log_density(&self, x: &[f64], grad: &mut [f64]) -> Option<f64> {
        let p = self.design.n_cols();
        let v_at = p + self.n_theta + usize::from(self.free_dispersion);
        let phi = if self.free_dispersion {
            x[p + self.n_theta].exp()
        } else {
            1.0
        };
        if !(phi.is_finite() && phi > 0.0) {
            return None;
        }
        grad.iter_mut().for_each(|g| *g = 0.0);
        // Each factor's standard deviations, unit rows `(θᵣ, 1)` and their
        // norms, and Λ.
        struct Pieces {
            s: Vec<f64>,
            u: Vec<f64>,
            norm: Vec<f64>,
            lambda: Vec<f64>,
        }
        let mut pieces = Vec::with_capacity(self.factors.len());
        for f in self.factors {
            let d = f.d;
            let lambda = f.lambda(x)?;
            let mut u = vec![0.0; d * d];
            let mut norm = vec![1.0; d];
            let mut a = f.theta + d;
            for r in 0..d {
                u[r * d + r] = 1.0;
                for c in 0..r {
                    u[r * d + c] = x[a];
                    a += 1;
                }
                norm[r] = u[r * d..r * d + r + 1]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt();
            }
            let s = (0..d).map(|r| x[f.theta + r].exp()).collect();
            pieces.push(Pieces { s, u, norm, lambda });
        }
        let beta = &x[..p];
        let v = &x[v_at..];
        let eta0 = self.design.linear_predictor(beta);
        let w = self.design.weights();
        let mut score = vec![0.0; eta0.len()];
        let mut logp = 0.0;
        let mut dlog_phi = 0.0;
        let mut b = Vec::new();
        for (i, &e0) in eta0.iter().enumerate() {
            // Row i's effects b = Λ v_l of each factor, and η.
            b.clear();
            let mut eta = e0;
            for (f, pc) in self.factors.iter().zip(&pieces) {
                let vl = &v[f.start + f.layout.level[i] * f.d..][..f.d];
                for r in 0..f.d {
                    let br: f64 = (0..=r).map(|c| pc.lambda[r * f.d + c] * vl[c]).sum();
                    eta += f.layout.columns[r][i] * br;
                    b.push(br);
                }
            }
            let mu = eta.exp();
            let ll = self.family.log_likelihood(self.y[i], mu, w[i], phi);
            if !(mu.is_finite() && ll.is_finite()) {
                return None;
            }
            logp += ll;
            let s = w[i] / phi * (self.y[i] - mu) * mu.powf(1.0 - self.power);
            score[i] = s;
            if self.free_dispersion {
                dlog_phi += self.dll_dlog_phi(self.y[i], mu, w[i], phi);
            }
            let mut k = 0;
            for (f, pc) in self.factors.iter().zip(&pieces) {
                let d = f.d;
                let at = v_at + f.start + f.layout.level[i] * d;
                let vl = &v[at - v_at..][..d];
                for r in 0..d {
                    let z = f.layout.columns[r][i];
                    let br = b[k + r];
                    // ∂η/∂v_c = Σ_{r ≥ c} z_r Λ[r][c]: row r's share.
                    for c in 0..=r {
                        grad[at + c] += s * z * pc.lambda[r * d + c];
                    }
                    // ∂b_r/∂ log s_r = b_r.
                    grad[f.theta + r] += s * z * br;
                    if r > 0 {
                        // b_r = s_r (u_r · v) / n_r in θ_{r, c}.
                        let (n, sr) = (pc.norm[r], pc.s[r]);
                        let dot: f64 = (0..=r).map(|c| pc.u[r * d + c] * vl[c]).sum();
                        let base = f.theta + d + r * (r - 1) / 2;
                        for c in 0..r {
                            grad[base + c] +=
                                s * z * sr * (vl[c] / n - pc.u[r * d + c] * dot / (n * n * n));
                        }
                    }
                }
                k += d;
            }
        }
        for j in 0..p {
            let col = self.design.column(j);
            grad[j] += score.iter().zip(col).map(|(s, x)| s * x).sum::<f64>();
            let sd = self.prior_sd[j];
            logp -= 0.5 * (beta[j] / sd).powi(2);
            grad[j] -= beta[j] / (sd * sd);
        }
        for f in self.factors {
            // Half-normal on each sd, with the Jacobian of s = exp(x).
            for r in 0..f.d {
                let a = f.theta + r;
                let s = x[a].exp();
                logp += -0.5 * (s / self.sd_scale).powi(2) + x[a];
                grad[a] += -(s / self.sd_scale).powi(2) + 1.0;
            }
            let at = f.theta + f.d;
            let len = f.d * (f.d - 1) / 2;
            logp += lkj_log_density(&x[at..at + len], f.d, self.lkj, &mut grad[at..at + len]);
        }
        if self.free_dispersion {
            let a = p + self.n_theta;
            logp += -0.5 * (phi / self.dispersion_scale).powi(2) + x[a];
            grad[a] = dlog_phi - (phi / self.dispersion_scale).powi(2) + 1.0;
        }
        for (j, vj) in v.iter().enumerate() {
            logp -= 0.5 * vj * vj;
            grad[v_at + j] -= vj;
        }
        logp.is_finite().then_some(logp)
    }
}

/// One fitted factor's effects in the draws.
#[derive(Debug, Clone, PartialEq)]
struct EffectsLayout {
    name: String,
    terms: Vec<String>,
    levels: Vec<String>,
    start: usize,
    d: usize,
}

/// The posterior of one grouping factor's random effects.
#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorEffects {
    /// The grouping factor's name.
    pub name: String,
    /// Its terms: the intercept first if it has one, then its slopes.
    pub terms: Vec<String>,
    /// Its levels, sorted.
    pub levels: Vec<String>,
    /// `mean[c][l]`: the posterior mean of term `c`'s effect at level `l`.
    pub mean: Vec<Vec<f64>>,
    /// The posterior standard deviations, indexed as `mean`.
    pub sd: Vec<Vec<f64>>,
}

/// A sampled Bayesian GLMM.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesGlmmFit {
    spec: BayesGlmm,
    names: Vec<String>,
    parameters: PosteriorDraws,
    effects: PosteriorDraws,
    layout: Vec<EffectsLayout>,
    laplace: GlmmFit,
}

impl BayesGlmmFit {
    /// The specification that was sampled.
    pub fn spec(&self) -> &BayesGlmm {
        &self.spec
    }

    /// The fixed coefficients' names, as the design's columns.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The Laplace fit the chains started from.
    pub fn laplace(&self) -> &GlmmFit {
        &self.laplace
    }

    /// Draws of the parameters on their natural scale: the coefficients,
    /// then each factor's standard deviations `sd[g|term]` and
    /// correlations `cor[g|a,b]` (the strictly lower triangle, row by
    /// row), then the `dispersion` when it is free.
    pub fn parameters(&self) -> &PosteriorDraws {
        &self.parameters
    }

    /// Draws of the random effects `b[g|level|term]`, factor by factor,
    /// level by level, term by term.
    pub fn effects(&self) -> &PosteriorDraws {
        &self.effects
    }

    /// Divergent transitions among the kept draws, all chains: any at all
    /// means the posterior geometry may be biasing the draws.
    pub fn divergences(&self) -> usize {
        self.parameters.divergences().iter().sum()
    }

    /// Mean, sd, 5/50/95% quantiles, R̂ and bulk and tail ESS of each
    /// parameter, in the order of [`parameters`](Self::parameters).
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        self.parameters.summary()
    }

    /// The posterior mean and standard deviation of every random effect.
    pub fn random_effects(&self) -> Vec<PosteriorEffects> {
        let mean = self.effects.mean();
        let s = self.effects.n_draws() as f64;
        let q = self.effects.dim();
        let mut var = vec![0.0; q];
        for d in 0..self.effects.n_draws() {
            for (j, (vj, x)) in var.iter_mut().zip(self.effects.draw(d)).enumerate() {
                *vj += (x - mean[j]).powi(2) / (s - 1.0);
            }
        }
        self.layout
            .iter()
            .map(|f| {
                let m = f.levels.len();
                let at = |c: usize, l: usize| f.start + l * f.d + c;
                PosteriorEffects {
                    name: f.name.clone(),
                    terms: f.terms.clone(),
                    levels: f.levels.clone(),
                    mean: (0..f.d)
                        .map(|c| (0..m).map(|l| mean[at(c, l)]).collect())
                        .collect(),
                    sd: (0..f.d)
                        .map(|c| (0..m).map(|l| var[at(c, l)].sqrt()).collect())
                        .collect(),
                }
            })
            .collect()
    }

    /// Posterior means of `μ` for new rows: the average over the draws of
    /// `exp(xᵀ β + offset + Σ zᵀ b)`, each factor's effects looked up by
    /// `groups` as [`GlmmFit::predict`] does. A level the fit has not
    /// seen, or a factor not given, has effects 0 in every draw, so its
    /// rows get the mean at the population level.
    pub fn predict(&self, design: &Design, groups: &[Grouping]) -> Result<Vec<f64>> {
        if design.names() != self.names.as_slice() {
            return Err(Error::Data(
                "the design's columns differ from the fit's".into(),
            ));
        }
        let n = design.n_rows();
        let names: Vec<&str> = self.layout.iter().map(|f| f.name.as_str()).collect();
        check_groups(groups, &names, n)?;
        let given: Vec<Option<_>> = self
            .layout
            .iter()
            .map(|f| given_effects(&f.name, &f.terms, &f.levels, groups, n))
            .collect::<Result<_>>()?;
        let p = self.names.len();
        let s = self.parameters.n_draws();
        let mut mu = vec![0.0; n];
        for d in 0..s {
            let mut eta = design.linear_predictor(&self.parameters.draw(d)[..p]);
            let b = self.effects.draw(d);
            for (f, g) in self.layout.iter().zip(&given) {
                let Some(g) = g else { continue };
                for (i, e) in eta.iter_mut().enumerate() {
                    if let Some(l) = g.level[i] {
                        for (c, column) in g.columns.iter().enumerate() {
                            *e += b[f.start + l * f.d + c] * column[i];
                        }
                    }
                }
            }
            for (m, e) in mu.iter_mut().zip(eta) {
                *m += e.exp() / s as f64;
            }
        }
        Ok(mu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small book: `n` rows over `m` levels of `g`, a covariate `x` and
    /// a second covariate `w` that vary by row.
    fn book(n: usize, m: usize) -> (Vec<String>, Vec<f64>, Vec<f64>) {
        let labels = (0..n).map(|i| format!("g{:02}", i % m)).collect();
        let x = (0..n).map(|i| ((i * 7) % 11) as f64 / 5.0 - 1.0).collect();
        let w = (0..n).map(|i| ((i * 5) % 13) as f64 / 6.0 - 1.0).collect();
        (labels, x, w)
    }

    fn central_difference(density: &Posterior<'_>, x: &[f64]) -> Vec<f64> {
        let mut scratch = vec![0.0; x.len()];
        (0..x.len())
            .map(|a| {
                let h = 1e-6;
                let mut xp = x.to_vec();
                xp[a] += h;
                let up = density.log_density(&xp, &mut scratch).unwrap();
                xp[a] = x[a] - h;
                let down = density.log_density(&xp, &mut scratch).unwrap();
                (up - down) / (2.0 * h)
            })
            .collect()
    }

    #[test]
    fn the_gradient_is_the_density_s() {
        let (n, m) = (60, 6);
        let (labels, x, w) = book(n, m);
        let design = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; n], x.clone()],
        )
        .unwrap();
        let three = Grouping::new("g", labels.clone())
            .slope("x", x.clone())
            .slope("w", w.clone());
        let second = Grouping::new("h", (0..n).map(|i| format!("h{}", i % 4)).collect());
        let y: Vec<f64> = (0..n).map(|i| 0.5 + ((i * 3) % 7) as f64).collect();
        for family in [
            Family::Gamma,
            Family::Tweedie { power: 1.5 },
            Family::Poisson,
        ] {
            let (power, free) = Glmm::new(family).power().unwrap();
            let mut factors = Vec::new();
            let (mut start, mut theta) = (0, 2);
            for g in [&three, &second] {
                let layout = g.layout(n).unwrap();
                let d = layout.columns.len();
                let levels = layout.levels.len();
                factors.push(Factor {
                    d,
                    layout,
                    start,
                    theta,
                });
                start += levels * d;
                theta += d * (d + 1) / 2;
            }
            let density = Posterior {
                design: &design,
                y: &y,
                family,
                power,
                free_dispersion: free,
                factors: &factors,
                n_theta: theta - 2,
                q: start,
                prior_sd: &[10.0, 2.5],
                sd_scale: 1.0,
                lkj: 2.0,
                dispersion_scale: 10.0,
            };
            let dim = density.dim();
            let position: Vec<f64> = (0..dim)
                .map(|a| 0.3 * ((a * 37 % 17) as f64 / 8.0 - 1.0))
                .collect();
            let mut grad = vec![0.0; dim];
            density.log_density(&position, &mut grad).unwrap();
            let fd = central_difference(&density, &position);
            for (a, (g, f)) in grad.iter().zip(&fd).enumerate() {
                assert!(
                    (g - f).abs() < 1e-5 * f.abs().max(1.0),
                    "{} parameter {a}: {g} against {f}",
                    family.name()
                );
            }
        }
    }

    /// The LKJ density alone, in glmmTMB's parameters.
    struct Lkj {
        d: usize,
        eta: f64,
    }

    impl LogDensity for Lkj {
        fn dim(&self) -> usize {
            self.d * (self.d - 1) / 2
        }
        fn log_density(&self, x: &[f64], grad: &mut [f64]) -> Option<f64> {
            grad.iter_mut().for_each(|g| *g = 0.0);
            Some(lkj_log_density(x, self.d, self.eta, grad))
        }
    }

    #[test]
    fn the_correlations_have_the_lkj_marginals() {
        // Each correlation of an LKJ(η) matrix of dimension d is a beta on
        // (−1, 1) with mean 0 and variance 1 / (2η + d − 1).
        for (d, eta) in [(2, 1.0), (3, 1.0), (3, 2.0), (4, 0.5)] {
            let density = Lkj { d, eta };
            let s = Sampler {
                chains: 4,
                tune: 1000,
                draws: 4000,
                seed: 3,
                ..Sampler::default()
            };
            let draws = sample(&density, &vec![0.0; density.dim()], s).unwrap();
            let target = 1.0 / (2.0 * eta + d as f64 - 1.0);
            for r in 1..d {
                for c in 0..r {
                    let rho: Vec<f64> = (0..draws.n_draws())
                        .map(|k| {
                            let l = correlation_factor(draws.draw(k), d);
                            (0..d).map(|j| l[r * d + j] * l[c * d + j]).sum()
                        })
                        .collect();
                    let len = rho.len() as f64;
                    let mean = rho.iter().sum::<f64>() / len;
                    let var = rho.iter().map(|v| v * v).sum::<f64>() / len - mean * mean;
                    assert!(mean.abs() < 0.03, "d {d} eta {eta}: mean {mean}");
                    assert!(
                        (var - target).abs() < 0.04 * target.max(0.25),
                        "d {d} eta {eta} ({r}, {c}): variance {var} against {target}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_posterior_is_the_exact_one_on_a_small_book() {
        // One factor with a random intercept, eight levels, a Poisson
        // count: the posterior of (β, s) by integrating each level's
        // effect on a fine grid, against the draws.
        let m = 8;
        let rates: [f64; 8] = [3.0, 5.0, 4.0, 9.0, 2.0, 6.0, 4.0, 7.0];
        let counts: Vec<[f64; 4]> = (0..m)
            .map(|l| {
                let r = rates[l];
                [r, r + 1.0, (r - 1.0).max(0.0), r + 2.0]
            })
            .collect();
        let y: Vec<f64> = counts.iter().flatten().copied().collect();
        let n = y.len();
        let labels: Vec<String> = (0..n).map(|i| format!("l{}", i / 4)).collect();
        let design = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; n]]).unwrap();
        let spec = BayesGlmm::new(Family::Poisson).sampler(Sampler {
            chains: 4,
            tune: 1000,
            draws: 2000,
            seed: 11,
            target_accept: 0.9,
            ..Sampler::default()
        });
        let fit = spec
            .fit(&design, &[Grouping::new("g", labels)], &y)
            .unwrap();
        assert_eq!(fit.divergences(), 0);

        // The exact posterior on a grid of (β, log s).
        let log_factorial: Vec<f64> = y
            .iter()
            .map(|&k| (1..=k as u64).map(|j| (j as f64).ln()).sum())
            .collect();
        let level_log_lik = |l: usize, beta: f64, s: f64| -> f64 {
            // ∫ Π_i Pois(y_i | e^{β + s v}) φ(v) dv on v in [−8, 8].
            let k = 1601;
            let h = 16.0 / (k - 1) as f64;
            let terms: Vec<f64> = (0..k)
                .map(|j| {
                    let v = -8.0 + j as f64 * h;
                    let eta = beta + s * v;
                    let ll: f64 = (0..4)
                        .map(|t| {
                            let i = 4 * l + t;
                            y[i] * eta - eta.exp() - log_factorial[i]
                        })
                        .sum();
                    ll - 0.5 * v * v
                })
                .collect();
            let top = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            top + (terms.iter().map(|t| (t - top).exp()).sum::<f64>() * h).ln()
                - 0.5 * (2.0 * std::f64::consts::PI).ln()
        };
        let (nb, ns) = (101, 101);
        let mut grid = Vec::with_capacity(nb * ns);
        for a in 0..nb {
            let beta = 0.4 + 2.6 * a as f64 / (nb - 1) as f64;
            for c in 0..ns {
                let ls = -6.0 + 7.0 * c as f64 / (ns - 1) as f64;
                let s = ls.exp();
                let lp = (0..m).map(|l| level_log_lik(l, beta, s)).sum::<f64>()
                    - 0.5 * (beta / 10.0).powi(2)
                    - 0.5 * s * s
                    + ls;
                grid.push((beta, s, lp));
            }
        }
        let top = grid.iter().map(|g| g.2).fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = grid.iter().map(|g| (g.2 - top).exp()).sum();
        let moment = |f: &dyn Fn(f64, f64) -> f64| -> f64 {
            grid.iter()
                .map(|g| f(g.0, g.1) * (g.2 - top).exp())
                .sum::<f64>()
                / total
        };
        let beta_mean = moment(&|b, _| b);
        let beta_sd = (moment(&|b, _| b * b) - beta_mean * beta_mean).sqrt();
        let s_mean = moment(&|_, s| s);
        let s_sd = (moment(&|_, s| s * s) - s_mean * s_mean).sqrt();

        let summary = fit.summary().unwrap();
        assert!(summary.iter().all(|s| s.rhat < 1.02));
        let (b, s) = (&summary[0], &summary[1]);
        assert!(
            (b.mean - beta_mean).abs() < 0.06 * beta_sd,
            "beta {} against {beta_mean}",
            b.mean
        );
        assert!(
            (b.sd / beta_sd - 1.0).abs() < 0.06,
            "beta sd {} against {beta_sd}",
            b.sd
        );
        assert!(
            (s.mean - s_mean).abs() < 0.06 * s_sd,
            "sd {} against {s_mean}",
            s.mean
        );
        assert!(
            (s.sd / s_sd - 1.0).abs() < 0.06,
            "sd's sd {} against {s_sd}",
            s.sd
        );
    }

    #[test]
    fn slopes_predict_by_label_and_covariate() {
        let (n, m) = (120, 8);
        let (labels, x, _) = book(n, m);
        let design = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; n], x.clone()],
        )
        .unwrap();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                ((1.0 + 0.3 * x[i] + 0.2 * (i % m) as f64 * x[i]).exp() + (i % 3) as f64).round()
            })
            .collect();
        let g = Grouping::new("g", labels.clone()).slope("x", x.clone());
        let spec = BayesGlmm::new(Family::Poisson).sampler(Sampler {
            chains: 2,
            tune: 500,
            draws: 500,
            seed: 5,
            ..Sampler::default()
        });
        let fit = spec.fit(&design, std::slice::from_ref(&g), &y).unwrap();
        let names = fit.parameters().names();
        assert_eq!(
            names,
            [
                "(Intercept)",
                "x",
                "sd[g|(Intercept)]",
                "sd[g|x]",
                "cor[g|x,(Intercept)]"
            ]
        );
        let re = &fit.random_effects()[0];
        assert_eq!(re.terms, ["(Intercept)", "x"]);
        assert_eq!(re.levels.len(), m);
        // Predicting the training rows is the posterior mean of μ; a level
        // not seen gets the population mean.
        let mu = fit.predict(&design, &[g]).unwrap();
        assert!(mu.iter().all(|v| v.is_finite() && *v > 0.0));
        let one = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0], vec![0.5]],
        )
        .unwrap();
        let new = Grouping::new("g", vec!["unseen".into()]).slope("x", vec![0.5]);
        let pop = fit.predict(&one, &[new]).unwrap();
        let none = fit.predict(&one, &[]).unwrap();
        assert_eq!(pop, none);
        let missing = Grouping::new("g", vec!["g01".into()]);
        let err = fit.predict(&one, &[missing]).unwrap_err().to_string();
        assert!(err.contains("no values for its slope x"), "{err}");
    }

    #[test]
    fn bad_priors_are_errors() {
        let design = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; 4]]).unwrap();
        let g = Grouping::new("g", ["a", "a", "b", "b"].map(String::from).to_vec());
        let mut spec = BayesGlmm::new(Family::Poisson);
        spec.lkj = 0.0;
        let err = spec.fit(&design, &[g], &[1.0, 2.0, 3.0, 4.0]).unwrap_err();
        assert!(err.to_string().contains("lkj"), "{err}");
    }
}
