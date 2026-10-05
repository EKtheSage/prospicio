//! Bayesian stacking: model weights with a posterior, and weights that vary
//! with covariates (Yao, Pirš, Vehtari and Gelman, 2022, "Bayesian
//! hierarchical stacking: some models are (somewhere) useful").
//!
//! Both take the pointwise held-out log predictive densities `lpd[k][i]`
//! of `K` models (PSIS-LOO pointwise values, or cross-validated log
//! densities of any model) and treat the log score of the mixture,
//! `Σᵢ log Σₖ wᵢₖ exp(lpdᵢₖ)`, as a log-likelihood, sampled with NUTS
//! (`nuts-rs`, as [`crate::glm`]).
//!
//! - [`BayesStacking`]: one weight vector with a Dirichlet prior, the
//!   Bayesian version of `act_models::stack::stacking_weights`.
//! - [`HierarchicalStacking`]: weights `wᵢ = softmax(α + Bᵀ xᵢ)` against
//!   the last model as reference, so a model can be trusted in one part of
//!   the portfolio and not another. The priors are those of BayesBlend's
//!   `HierarchicalBayesStacking` (MIT, Ledger Investing), non-centred:
//!   - no pooling: `αₘ ~ N(alpha_loc, (alpha_scale δ)²)`,
//!     `βₘⱼ ~ N(beta_loc, (beta_scale δ)²)`;
//!   - partial pooling ([`Pooling`]): each model's slopes on the discrete
//!     covariates, and separately on the continuous ones, are drawn around
//!     a model-level mean, `βₘⱼ ~ N(μₘ, (σₘ δ)²)`, with
//!     `μₘ ~ N(μ, (tau_mu δ)²)` around a global mean
//!     `μ ~ N(0, (tau_mu_global δ)²)` and `σₘ ~ N⁺(0, tau_sigma²)`;
//!   - adaptive priors: `δ = N^λ` with `λ ~ Exponential(rate)`, which
//!     widens the priors as the data grow (otherwise `δ = 1`).
//!
//! Covariates enter as given; BayesBlend divides continuous ones by twice
//! their standard deviation (Gelman, 2008) and dummy-codes discrete ones,
//! which `act_models::Terms` does.

use std::collections::HashMap;
use std::fmt;

use act_core::{Error, Result};
use nuts_rs::{CpuLogpFunc, CpuMathError, HasDims, LogpError};
use rayon::prelude::*;

use crate::nuts::{Sampler, run_chain};

/// Bayesian stacking: a Dirichlet(`concentration`) prior on one weight
/// vector, and the log score of the mixture as the likelihood.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BayesStacking {
    /// Dirichlet concentration, one per model (1 is uniform on the simplex).
    pub concentration: Option<Vec<f64>>,
    pub sampler: Sampler,
}

/// Hierarchical (covariate-dependent) stacking.
///
/// The first [`discrete`](Self::discrete) covariates are dummy codes of
/// discrete covariates and the rest continuous; the split matters only
/// with [`pooling`](Self::pooling), which pools each group separately.
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalStacking {
    pub alpha_loc: f64,
    pub alpha_scale: f64,
    /// Prior of the slopes without pooling.
    pub beta_loc: f64,
    pub beta_scale: f64,
    /// Partial pooling of the slopes; `None` for none.
    pub pooling: Option<Pooling>,
    /// Rate of the exponential prior on `λ` in `δ = N^λ`; `None` keeps
    /// `δ = 1`. BayesBlend's default rate is 4.
    pub adaptive: Option<f64>,
    /// Number of leading covariates that are dummy codes.
    pub discrete: usize,
    pub sampler: Sampler,
}

impl Default for HierarchicalStacking {
    /// BayesBlend's defaults: `N(0, 1)` priors on every intercept and slope,
    /// no pooling, no adaptation, every covariate continuous.
    fn default() -> Self {
        Self {
            alpha_loc: 0.0,
            alpha_scale: 1.0,
            beta_loc: 0.0,
            beta_scale: 1.0,
            pooling: None,
            adaptive: None,
            discrete: 0,
            sampler: Sampler::default(),
        }
    }
}

/// Partial pooling of the slopes, as BayesBlend's pooling model. A scale
/// of 0 removes that level: `tau_mu_global = 0` fixes the global mean at
/// 0, `tau_mu_* = 0` makes every model share it (complete pooling), and
/// `tau_sigma_* = 0` sets every slope in the group to its model's mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pooling {
    pub tau_mu_global: f64,
    pub tau_mu_discrete: f64,
    pub tau_mu_continuous: f64,
    pub tau_sigma_discrete: f64,
    pub tau_sigma_continuous: f64,
}

impl Default for Pooling {
    /// BayesBlend's defaults: every scale 1.
    fn default() -> Self {
        Self {
            tau_mu_global: 1.0,
            tau_mu_discrete: 1.0,
            tau_mu_continuous: 1.0,
            tau_sigma_discrete: 1.0,
            tau_sigma_continuous: 1.0,
        }
    }
}

/// Posterior draws of stacking weights.
#[derive(Debug, Clone, PartialEq)]
pub struct StackingFit {
    k: usize,
    /// Chain-major draws of the unconstrained parameters.
    draws: Vec<f64>,
    dim: usize,
    /// Covariates per model-weight equation (0 for [`BayesStacking`]).
    p: usize,
    /// The hierarchical parameterization; `None` for [`BayesStacking`],
    /// whose parameters are the logits themselves.
    hier: Option<Hier>,
    chains: usize,
    divergences: usize,
}

/// `lpd` as `n × K` row-major, each row shifted by its maximum (the
/// weights do not change).
fn scaled_densities(lpd: &[Vec<f64>]) -> Result<(Vec<f64>, usize, usize)> {
    let k = lpd.len();
    if k < 2 {
        return Err(Error::Data("stacking needs at least two models".into()));
    }
    let n = lpd[0].len();
    if n == 0 || lpd.iter().any(|l| l.len() != n) {
        return Err(Error::Data(
            "every model needs the same, non-zero number of pointwise log densities".into(),
        ));
    }
    if lpd.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Data("pointwise log densities must be finite".into()));
    }
    let mut e = vec![0.0; n * k];
    for i in 0..n {
        let m = lpd.iter().map(|l| l[i]).fold(f64::NEG_INFINITY, f64::max);
        for j in 0..k {
            e[i * k + j] = (lpd[j][i] - m).exp();
        }
    }
    Ok((e, n, k))
}

fn softmax_ref(logits: &[f64], out: &mut [f64]) {
    // The last weight is the reference, logit 0.
    let k = out.len();
    let m = logits.iter().copied().fold(0.0f64, f64::max);
    let mut total = (-m).exp();
    out[k - 1] = total;
    for (o, l) in out.iter_mut().zip(logits) {
        *o = (l - m).exp();
        total += *o;
    }
    out.iter_mut().for_each(|o| *o /= total);
}

fn check_sampler(s: &Sampler) -> Result<()> {
    if s.chains == 0 || s.draws < 2 {
        return Err(Error::Data(
            "the sampler needs at least one chain and two draws".into(),
        ));
    }
    Ok(())
}

impl BayesStacking {
    /// Samples the posterior of the weights.
    pub fn fit(&self, lpd: &[Vec<f64>]) -> Result<StackingFit> {
        check_sampler(&self.sampler)?;
        let (e, n, k) = scaled_densities(lpd)?;
        let a = match &self.concentration {
            Some(a) if a.len() == k && a.iter().all(|v| v.is_finite() && *v > 0.0) => a.clone(),
            Some(_) => {
                return Err(Error::Data(format!(
                    "concentration needs {k} finite positive values"
                )));
            }
            None => vec![1.0; k],
        };
        let density = |_| Density {
            e: &e,
            n,
            k,
            kind: Kind::Dirichlet(&a),
        };
        sample(density, vec![0.0; k - 1], self.sampler, k, 0, None)
    }
}

impl HierarchicalStacking {
    /// Samples the posterior of the intercepts and slopes; `x[j][i]` is
    /// covariate `j` of observation `i` (no intercept column).
    pub fn fit(&self, lpd: &[Vec<f64>], x: &[Vec<f64>]) -> Result<StackingFit> {
        check_sampler(&self.sampler)?;
        let (e, n, k) = scaled_densities(lpd)?;
        let p = x.len();
        if x.iter()
            .any(|c| c.len() != n || c.iter().any(|v| !v.is_finite()))
        {
            return Err(Error::Data(format!(
                "every covariate needs {n} finite values"
            )));
        }
        for (name, v) in [
            ("alpha_scale", self.alpha_scale),
            ("beta_scale", self.beta_scale),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::InvalidParameter {
                    name,
                    value: v,
                    reason: "must be finite and positive",
                });
            }
        }
        if let Some(pool) = &self.pooling {
            for (name, v) in [
                ("tau_mu_global", pool.tau_mu_global),
                ("tau_mu_discrete", pool.tau_mu_discrete),
                ("tau_mu_continuous", pool.tau_mu_continuous),
                ("tau_sigma_discrete", pool.tau_sigma_discrete),
                ("tau_sigma_continuous", pool.tau_sigma_continuous),
            ] {
                if !(v.is_finite() && v >= 0.0) {
                    return Err(Error::InvalidParameter {
                        name,
                        value: v,
                        reason: "must be finite and non-negative",
                    });
                }
            }
        }
        if let Some(rate) = self.adaptive
            && !(rate.is_finite() && rate > 0.0)
        {
            return Err(Error::InvalidParameter {
                name: "adaptive",
                value: rate,
                reason: "must be finite and positive",
            });
        }
        if self.discrete > p {
            return Err(Error::Data(format!(
                "{} discrete covariates of {p}",
                self.discrete
            )));
        }
        // Row-major n × p for the density.
        let xr: Vec<f64> = (0..n).flat_map(|i| x.iter().map(move |c| c[i])).collect();
        let hier = Hier {
            m: k - 1,
            p,
            discrete: self.discrete,
            alpha_loc: self.alpha_loc,
            alpha_scale: self.alpha_scale,
            beta_loc: self.beta_loc,
            beta_scale: self.beta_scale,
            pooling: self.pooling,
            adaptive: self.adaptive,
            log_n: (n as f64).ln(),
        };
        let density = |_| Density {
            e: &e,
            n,
            k,
            kind: Kind::Hierarchical { x: &xr, hier },
        };
        sample(
            density,
            vec![0.0; hier.dim()],
            self.sampler,
            k,
            p,
            Some(hier),
        )
    }
}

fn sample<'a, D>(
    density: D,
    start: Vec<f64>,
    s: Sampler,
    k: usize,
    p: usize,
    hier: Option<Hier>,
) -> Result<StackingFit>
where
    D: Fn(usize) -> Density<'a> + Sync,
{
    let results: Vec<Result<(Vec<f64>, usize)>> = (0..s.chains)
        .into_par_iter()
        .map(|c| run_chain(density(c), &start, s, c))
        .collect();
    let mut draws = Vec::new();
    let mut divergences = 0;
    for r in results {
        let (d, div) = r?;
        draws.extend(d);
        divergences += div;
    }
    Ok(StackingFit {
        k,
        draws,
        dim: start.len(),
        p,
        hier,
        chains: s.chains,
        divergences,
    })
}

impl StackingFit {
    /// Total posterior draws.
    pub fn n_draws(&self) -> usize {
        self.draws.len() / self.dim.max(1)
    }

    /// Number of chains.
    pub fn chains(&self) -> usize {
        self.chains
    }

    /// Divergent transitions among the kept draws.
    pub fn divergences(&self) -> usize {
        self.divergences
    }

    /// Unconstrained draws, row-major `n_draws × dim`.
    fn draw(&self, d: usize) -> &[f64] {
        &self.draws[d * self.dim..(d + 1) * self.dim]
    }

    /// Intercepts (`K - 1`) and slopes (`(K - 1) × p`, row-major) in draw
    /// `d`, on the natural scale (the prior's non-centring undone).
    fn coefficients(&self, d: usize) -> (Vec<f64>, Vec<f64>) {
        let raw = self.draw(d);
        match &self.hier {
            None => (raw.to_vec(), Vec::new()),
            Some(h) => h.natural(raw),
        }
    }

    /// Posterior draws of the intercepts `α` (`n_draws × (K - 1)`, row-major;
    /// the logits for [`BayesStacking`]).
    pub fn alpha_draws(&self) -> Vec<f64> {
        (0..self.n_draws())
            .flat_map(|d| self.coefficients(d).0)
            .collect()
    }

    /// Posterior draws of the slopes `β`, row-major
    /// `n_draws × (K - 1) × p`; empty for [`BayesStacking`].
    pub fn beta_draws(&self) -> Vec<f64> {
        (0..self.n_draws())
            .flat_map(|d| self.coefficients(d).1)
            .collect()
    }

    /// Posterior draws of `δ = N^λ`, the factor on the prior scales; all 1
    /// unless the priors are adaptive.
    pub fn delta_draws(&self) -> Vec<f64> {
        (0..self.n_draws())
            .map(|d| self.hier.as_ref().map_or(1.0, |h| h.delta(self.draw(d)).0))
            .collect()
    }

    /// Posterior mean weights of observations with covariates `x[j][i]`
    /// (`x` empty for [`BayesStacking`], giving one row), averaged over
    /// draws as BayesBlend does: `n × K` row-major.
    pub fn weights(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        if x.len() != self.p {
            return Err(Error::Data(format!(
                "{} covariates for a fit with {}",
                x.len(),
                self.p
            )));
        }
        let n = if self.p == 0 { 1 } else { x[0].len() };
        if x.iter().any(|c| c.len() != n) {
            return Err(Error::Data("covariates differ in length".into()));
        }
        let k = self.k;
        let s = self.n_draws() as f64;
        let mut out = vec![0.0; n * k];
        let mut logits = vec![0.0; k - 1];
        let mut w = vec![0.0; k];
        let p = self.p;
        for d in 0..self.n_draws() {
            let (alpha, beta) = self.coefficients(d);
            for i in 0..n {
                for (m, a) in alpha.iter().enumerate() {
                    let b = &beta[m * p..(m + 1) * p];
                    logits[m] = a + b.iter().zip(x).map(|(bj, c)| bj * c[i]).sum::<f64>();
                }
                softmax_ref(&logits, &mut w);
                for (o, wk) in out[i * k..(i + 1) * k].iter_mut().zip(&w) {
                    *o += wk / s;
                }
            }
        }
        Ok(out)
    }

    /// Posterior summary of the unconstrained-scale coefficients: R̂ and
    /// bulk ESS of every intercept and slope, for convergence checks.
    pub fn rhat_ess(&self) -> Result<Vec<(f64, f64)>> {
        let per_chain = self.n_draws() / self.chains;
        (0..self.dim)
            .map(|j| {
                let chains: Vec<Vec<f64>> = (0..self.chains)
                    .map(|c| {
                        (0..per_chain)
                            .map(|d| self.draw(c * per_chain + d)[j])
                            .collect()
                    })
                    .collect();
                let refs: Vec<&[f64]> = chains.iter().map(Vec::as_slice).collect();
                Ok((crate::rhat(&refs)?, crate::ess_bulk(&refs)?))
            })
            .collect()
    }
}

enum Kind<'a> {
    Dirichlet(&'a [f64]),
    Hierarchical { x: &'a [f64], hier: Hier },
}

/// The hierarchical parameterization. The unconstrained vector holds the
/// standard-normal `α′` (`m = K - 1`), then `β′` (`m × p`), then with
/// pooling the global `μ′` and, per covariate group present (discrete,
/// then continuous), `μₘ′` (`m`) and `log σₘ′` (`m`), and last `log λ`
/// when adaptive.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Hier {
    m: usize,
    p: usize,
    discrete: usize,
    alpha_loc: f64,
    alpha_scale: f64,
    beta_loc: f64,
    beta_scale: f64,
    pooling: Option<Pooling>,
    adaptive: Option<f64>,
    log_n: f64,
}

/// A covariate group under pooling: its columns, `tau_mu`, `tau_sigma`
/// and the offset of its `μₘ′` in the parameter vector (`log σₘ′`
/// follows).
struct Group {
    cols: std::ops::Range<usize>,
    tau_mu: f64,
    tau_sigma: f64,
    offset: usize,
}

impl Hier {
    fn groups(&self) -> Vec<Group> {
        let Some(pool) = self.pooling else {
            return Vec::new();
        };
        let mut offset = self.m + self.m * self.p + 1;
        let mut out = Vec::new();
        for (cols, tau_mu, tau_sigma) in [
            (
                0..self.discrete,
                pool.tau_mu_discrete,
                pool.tau_sigma_discrete,
            ),
            (
                self.discrete..self.p,
                pool.tau_mu_continuous,
                pool.tau_sigma_continuous,
            ),
        ] {
            if !cols.is_empty() {
                out.push(Group {
                    cols,
                    tau_mu,
                    tau_sigma,
                    offset,
                });
                offset += 2 * self.m;
            }
        }
        out
    }

    fn dim(&self) -> usize {
        let pooled = if self.pooling.is_some() {
            1 + 2 * self.m * self.groups().len()
        } else {
            0
        };
        self.m + self.m * self.p + pooled + usize::from(self.adaptive.is_some())
    }

    /// `(δ, λ)`: `(N^λ, λ)` when adaptive, else `(1, 0)`.
    fn delta(&self, raw: &[f64]) -> (f64, f64) {
        match self.adaptive {
            Some(_) => {
                let lambda = raw[raw.len() - 1].exp();
                ((lambda * self.log_n).exp(), lambda)
            }
            None => (1.0, 0.0),
        }
    }

    /// Intercepts and row-major slopes on the natural scale.
    fn natural(&self, raw: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (m, p) = (self.m, self.p);
        let (delta, _) = self.delta(raw);
        let alpha = (0..m)
            .map(|j| self.alpha_loc + self.alpha_scale * delta * raw[j])
            .collect();
        let b = &raw[m..m + m * p];
        let beta = match self.pooling {
            None => b
                .iter()
                .map(|v| self.beta_loc + self.beta_scale * delta * v)
                .collect(),
            Some(pool) => {
                let mu_global = pool.tau_mu_global * delta * raw[m + m * p];
                let mut beta = vec![0.0; m * p];
                for g in self.groups() {
                    for j in 0..m {
                        let mu = mu_global + g.tau_mu * delta * raw[g.offset + j];
                        let sigma = g.tau_sigma * raw[g.offset + m + j].exp();
                        for c in g.cols.clone() {
                            beta[j * p + c] = mu + sigma * delta * b[j * p + c];
                        }
                    }
                }
                beta
            }
        };
        (alpha, beta)
    }

    /// Adds to `grad` the gradient through the parameterization of the
    /// likelihood's gradient in the natural intercepts (`ga`) and slopes
    /// (`gb`), plus the log prior and its gradient; returns the log prior.
    fn backward(&self, raw: &[f64], ga: &[f64], gb: &[f64], grad: &mut [f64]) -> f64 {
        let (m, p) = (self.m, self.p);
        let (delta, lambda) = self.delta(raw);
        // Gradient in δ, collected from every term that scales with it.
        let mut gd = 0.0;
        for j in 0..m {
            grad[j] += self.alpha_scale * delta * ga[j];
            gd += self.alpha_scale * raw[j] * ga[j];
        }
        let mut log_scales = Vec::new();
        match self.pooling {
            None => {
                for i in 0..m * p {
                    grad[m + i] += self.beta_scale * delta * gb[i];
                    gd += self.beta_scale * raw[m + i] * gb[i];
                }
            }
            Some(pool) => {
                let g_off = m + m * p;
                let mut g_mu_global = 0.0;
                for g in self.groups() {
                    for j in 0..m {
                        let sigma = g.tau_sigma * raw[g.offset + m + j].exp();
                        let (mut g_mu, mut g_sigma) = (0.0, 0.0);
                        for c in g.cols.clone() {
                            let i = j * p + c;
                            let b = raw[m + i];
                            grad[m + i] += sigma * delta * gb[i];
                            gd += sigma * b * gb[i];
                            g_mu += gb[i];
                            g_sigma += delta * b * gb[i];
                        }
                        grad[g.offset + j] += g.tau_mu * delta * g_mu;
                        gd += g.tau_mu * raw[g.offset + j] * g_mu;
                        g_mu_global += g_mu;
                        // σ = tau_sigma e^u, so ∂σ/∂u = σ.
                        grad[g.offset + m + j] += g_sigma * sigma;
                        log_scales.push(g.offset + m + j);
                    }
                }
                grad[g_off] += pool.tau_mu_global * delta * g_mu_global;
                gd += pool.tau_mu_global * raw[g_off] * g_mu_global;
            }
        }
        let mut logp = 0.0;
        let last = raw.len() - 1;
        if let Some(rate) = self.adaptive {
            // δ = exp(λ log N), λ = e^v.
            grad[last] += gd * delta * self.log_n * lambda;
            // Exponential(rate) on λ, plus the Jacobian v.
            logp += -rate * lambda + raw[last];
            grad[last] += -rate * lambda + 1.0;
        }
        for (i, v) in raw.iter().enumerate() {
            if self.adaptive.is_some() && i == last {
                continue;
            }
            if log_scales.contains(&i) {
                // Half-normal on σ′ = e^u, plus the Jacobian u.
                let s = v.exp();
                logp += -0.5 * s * s + v;
                grad[i] += -s * s + 1.0;
            } else {
                logp -= 0.5 * v * v;
                grad[i] -= v;
            }
        }
        logp
    }
}

/// The posterior density of the stacking parameters.
struct Density<'a> {
    e: &'a [f64],
    n: usize,
    k: usize,
    kind: Kind<'a>,
}

impl fmt::Debug for Density<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Density")
            .field("n", &self.n)
            .field("k", &self.k)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct Never;

impl fmt::Display for Never {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("stacking density error")
    }
}

impl std::error::Error for Never {}

impl LogpError for Never {
    fn is_recoverable(&self) -> bool {
        true
    }
}

impl HasDims for Density<'_> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        [("parameter".to_string(), self.dim() as u64)]
            .into_iter()
            .collect()
    }
}

impl CpuLogpFunc for Density<'_> {
    type LogpError = Never;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        match &self.kind {
            Kind::Dirichlet(_) => self.k - 1,
            Kind::Hierarchical { hier, .. } => hier.dim(),
        }
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> std::result::Result<f64, Never> {
        let (n, k) = (self.n, self.k);
        grad.iter_mut().for_each(|g| *g = 0.0);
        let mut logits = vec![0.0; k - 1];
        let mut w = vec![0.0; k];
        let mut logp = 0.0;
        match self.kind {
            Kind::Dirichlet(a) => {
                softmax_ref(position, &mut w);
                for row in self.e.chunks_exact(k).take(n) {
                    let mix: f64 = row.iter().zip(&w).map(|(e, wk)| e * wk).sum();
                    logp += mix.ln();
                    for m in 0..k - 1 {
                        // ∂/∂zₘ log Σ wₖ eₖ = wₘ eₘ / mix - wₘ.
                        grad[m] += w[m] * row[m] / mix - w[m];
                    }
                }
                // Dirichlet prior plus the Jacobian Π wₖ of the
                // additive-logistic map: Σ aₖ log wₖ.
                let total: f64 = a.iter().sum();
                for (j, (aj, wj)) in a.iter().zip(&w).enumerate() {
                    logp += aj * wj.ln();
                    if j < k - 1 {
                        grad[j] += aj - total * wj;
                    }
                }
            }
            Kind::Hierarchical { x, hier } => {
                let p = hier.p;
                let (alpha, beta) = hier.natural(position);
                let mut ga = vec![0.0; k - 1];
                let mut gb = vec![0.0; (k - 1) * p];
                for (i, row) in self.e.chunks_exact(k).take(n).enumerate() {
                    let xi = &x[i * p..(i + 1) * p];
                    for m in 0..k - 1 {
                        let b = &beta[m * p..(m + 1) * p];
                        logits[m] =
                            alpha[m] + b.iter().zip(xi).map(|(bj, xj)| bj * xj).sum::<f64>();
                    }
                    softmax_ref(&logits, &mut w);
                    let mix: f64 = row.iter().zip(&w).map(|(e, wk)| e * wk).sum();
                    logp += mix.ln();
                    for m in 0..k - 1 {
                        let ds = w[m] * row[m] / mix - w[m];
                        ga[m] += ds;
                        for (j, xj) in xi.iter().enumerate() {
                            gb[m * p + j] += ds * xj;
                        }
                    }
                }
                logp += hier.backward(position, &ga, &gb, grad);
            }
        }
        if logp.is_finite() {
            Ok(logp)
        } else {
            Err(Never)
        }
    }

    fn expand_vector<R: nuts_rs::rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        position: &[f64],
    ) -> std::result::Result<Vec<f64>, CpuMathError> {
        Ok(position.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Sampler {
        Sampler {
            chains: 2,
            tune: 400,
            draws: 400,
            seed: 9,
            ..Sampler::default()
        }
    }

    #[test]
    fn gradients_match_finite_differences() {
        let lpd = vec![
            vec![-1.0, -2.0, -0.5, -1.5, -3.0],
            vec![-1.5, -1.0, -0.7, -2.5, -1.0],
            vec![-1.2, -1.8, -0.4, -1.0, -2.0],
        ];
        let (e, n, k) = scaled_densities(&lpd).unwrap();
        let a = [1.0, 2.0, 0.5];
        let x = [0.3, -1.0, 0.5, 2.0, 1.2, -0.4, 0.0, 0.9, -1.5, 0.1];
        let base = Hier {
            m: 2,
            p: 2,
            discrete: 1,
            alpha_loc: 0.1,
            alpha_scale: 1.3,
            beta_loc: -0.2,
            beta_scale: 0.7,
            pooling: None,
            adaptive: None,
            log_n: (n as f64).ln(),
        };
        let pooled = Hier {
            pooling: Some(Pooling {
                tau_mu_global: 0.8,
                tau_mu_discrete: 1.1,
                tau_mu_continuous: 0.6,
                tau_sigma_discrete: 0.9,
                tau_sigma_continuous: 1.4,
            }),
            ..base
        };
        let mut kinds = vec![(Kind::Dirichlet(&a), vec![0.2, -0.4])];
        for hier in [
            base,
            Hier {
                adaptive: Some(4.0),
                ..base
            },
            pooled,
            Hier {
                adaptive: Some(2.0),
                ..pooled
            },
            Hier {
                discrete: 0,
                adaptive: Some(2.0),
                ..pooled
            },
        ] {
            let at: Vec<f64> = (0..hier.dim())
                .map(|i| 0.3 * ((i * 7 % 11) as f64 / 5.0 - 1.0))
                .collect();
            kinds.push((Kind::Hierarchical { x: &x, hier }, at));
        }
        for (kind, at) in kinds {
            let mut d = Density { e: &e, n, k, kind };
            let mut g = vec![0.0; at.len()];
            d.logp(&at, &mut g).unwrap();
            for j in 0..at.len() {
                let h = 1e-6;
                let (mut up, mut dn) = (at.clone(), at.clone());
                up[j] += h;
                dn[j] -= h;
                let mut s = vec![0.0; at.len()];
                let num = (d.logp(&up, &mut s).unwrap() - d.logp(&dn, &mut s).unwrap()) / (2.0 * h);
                assert!(
                    (g[j] - num).abs() < 1e-6 * num.abs().max(1.0),
                    "{j}: {} vs {num}",
                    g[j]
                );
            }
        }
    }

    #[test]
    fn a_covariate_shifts_the_weights_where_each_model_is_better() {
        // Model 0 is better for x < 0, model 1 for x > 0.
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 / (n - 1) as f64) - 0.5).collect();
        let a: Vec<f64> = x
            .iter()
            .map(|v| if *v < 0.0 { -0.5 } else { -2.0 })
            .collect();
        let b: Vec<f64> = x
            .iter()
            .map(|v| if *v < 0.0 { -2.0 } else { -0.5 })
            .collect();
        let spec = HierarchicalStacking {
            sampler: small(),
            ..HierarchicalStacking::default()
        };
        let fit = spec.fit(&[a.clone(), b.clone()], &[x]).unwrap();
        assert!(fit.rhat_ess().unwrap().iter().all(|(r, _)| *r < 1.05));
        let w = fit.weights(&[vec![-0.4, 0.4]]).unwrap();
        assert!(w[0] > 0.7 && w[2] < 0.3, "{w:?}");
        assert_eq!(
            fit,
            spec.fit(
                &[a.clone(), b.clone()],
                &[(0..n).map(|i| (i as f64 / (n - 1) as f64) - 0.5).collect()]
            )
            .unwrap()
        );

        let flat = BayesStacking {
            sampler: small(),
            ..BayesStacking::default()
        }
        .fit(&[a, b])
        .unwrap();
        let w = flat.weights(&[]).unwrap();
        assert!((w[0] - 0.5).abs() < 0.1 && (w[0] + w[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pooling_scales_of_zero_fix_the_slopes() {
        let hier = Hier {
            m: 2,
            p: 3,
            discrete: 2,
            alpha_loc: 0.0,
            alpha_scale: 1.0,
            beta_loc: 0.0,
            beta_scale: 1.0,
            pooling: Some(Pooling {
                tau_mu_global: 0.0,
                tau_mu_discrete: 1.0,
                tau_mu_continuous: 1.0,
                tau_sigma_discrete: 0.0,
                tau_sigma_continuous: 0.0,
            }),
            adaptive: None,
            log_n: 0.0,
        };
        // 2 α′, 6 β′, the global μ′, and μ′ and log σ′ for two groups.
        assert_eq!(hier.dim(), 2 + 6 + 1 + 2 * 2 * 2);
        let raw: Vec<f64> = (0..hier.dim()).map(|i| i as f64 / 10.0).collect();
        let (_, beta) = hier.natural(&raw);
        // σ = 0: each slope is its model's group mean μₘ (the global mean
        // is 0); model 0's discrete mean is μ′ at offset 9, continuous at 13.
        assert_eq!(&beta[..3], [0.9, 0.9, 1.3]);
        assert_eq!(&beta[3..], [1.0, 1.0, 1.4]);
    }

    #[test]
    fn pooled_and_adaptive_fits_find_the_better_model() {
        // Four regions (three dummies) and a continuous covariate. Model 0
        // is better in regions 0 and 1, model 1 in regions 2 and 3.
        let n = 120;
        let region: Vec<usize> = (0..n).map(|i| i % 4).collect();
        let dummies: Vec<Vec<f64>> = (1..4)
            .map(|r| {
                region
                    .iter()
                    .map(|&g| f64::from(u8::from(g == r)))
                    .collect()
            })
            .collect();
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 37) % 101) as f64 / 100.0 - 0.5)
            .collect();
        let good = |g: usize, model: usize| (g < 2) == (model == 0);
        let lpd = |model: usize| -> Vec<f64> {
            region
                .iter()
                .map(|&g| if good(g, model) { -0.5 } else { -2.0 })
                .collect()
        };
        let mut covariates = dummies;
        covariates.push(x);
        let at = |g: usize| -> Vec<f64> { (1..4).map(|r| f64::from(u8::from(g == r))).collect() };
        let probe: Vec<Vec<f64>> = (0..4)
            .map(|c| {
                if c < 3 {
                    vec![at(0)[c], at(3)[c]]
                } else {
                    vec![0.0, 0.0]
                }
            })
            .collect();
        for adaptive in [None, Some(4.0)] {
            let spec = HierarchicalStacking {
                pooling: Some(Pooling::default()),
                adaptive,
                discrete: 3,
                sampler: Sampler {
                    target_accept: 0.95,
                    tune: 300,
                    draws: 300,
                    ..small()
                },
                ..HierarchicalStacking::default()
            };
            let fit = spec.fit(&[lpd(0), lpd(1)], &covariates).unwrap();
            assert!(
                fit.rhat_ess().unwrap().iter().all(|(r, _)| *r < 1.05),
                "{adaptive:?}"
            );
            // Region 0 favours model 0, region 3 model 1.
            let w = fit.weights(&probe).unwrap();
            assert!(w[0] > 0.7 && w[2] < 0.3, "{w:?}");
            assert_eq!(fit.beta_draws().len(), fit.n_draws() * 4);
            assert!(fit.delta_draws().iter().all(|v| *v >= 1.0));
        }
    }

    #[test]
    fn rejects_bad_input() {
        let one = vec![vec![-1.0, -2.0]];
        assert!(BayesStacking::default().fit(&one).is_err());
        let two = vec![vec![-1.0, -2.0], vec![-1.5, -1.0]];
        let bad = BayesStacking {
            concentration: Some(vec![1.0]),
            ..BayesStacking::default()
        };
        assert!(bad.fit(&two).is_err());
        assert!(
            HierarchicalStacking::default()
                .fit(&two, &[vec![0.0]])
                .is_err()
        );
        let x = [vec![0.0, 1.0]];
        for spec in [
            HierarchicalStacking {
                discrete: 2,
                ..HierarchicalStacking::default()
            },
            HierarchicalStacking {
                adaptive: Some(0.0),
                ..HierarchicalStacking::default()
            },
            HierarchicalStacking {
                pooling: Some(Pooling {
                    tau_mu_global: -1.0,
                    ..Pooling::default()
                }),
                ..HierarchicalStacking::default()
            },
        ] {
            assert!(spec.fit(&two, &x).is_err());
        }
    }
}
