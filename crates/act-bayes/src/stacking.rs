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
//!   the portfolio and not another. Normal priors, non-centred, as
//!   BayesBlend's `HierarchicalBayesStacking` without partial pooling
//!   (MIT, Ledger Investing): `αₘ ~ N(alpha_loc, alpha_scale²)`,
//!   `βₘⱼ ~ N(beta_loc, beta_scale²)`.
//!
//! Covariates enter as given; BayesBlend divides continuous ones by twice
//! their standard deviation (Gelman, 2008) and dummy-codes discrete ones,
//! which `act_models::Terms` does.

use std::collections::HashMap;
use std::fmt;

use act_core::{Error, Result};
use nuts_rs::{CpuLogpFunc, CpuMathError, HasDims, LogpError};
use rayon::prelude::*;

use crate::glm::{Sampler, run_chain};

/// Bayesian stacking: a Dirichlet(`concentration`) prior on one weight
/// vector, and the log score of the mixture as the likelihood.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BayesStacking {
    /// Dirichlet concentration, one per model (1 is uniform on the simplex).
    pub concentration: Option<Vec<f64>>,
    pub sampler: Sampler,
}

/// Hierarchical (covariate-dependent) stacking.
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalStacking {
    pub alpha_loc: f64,
    pub alpha_scale: f64,
    pub beta_loc: f64,
    pub beta_scale: f64,
    pub sampler: Sampler,
}

impl Default for HierarchicalStacking {
    /// BayesBlend's defaults: `N(0, 1)` priors on every intercept and slope.
    fn default() -> Self {
        Self {
            alpha_loc: 0.0,
            alpha_scale: 1.0,
            beta_loc: 0.0,
            beta_scale: 1.0,
            sampler: Sampler::default(),
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
    /// Prior location and scale of the intercepts and slopes; `None` for
    /// [`BayesStacking`], whose parameters are the logits themselves.
    hier: Option<[f64; 4]>,
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
        // Row-major n × p for the density.
        let xr: Vec<f64> = (0..n).flat_map(|i| x.iter().map(move |c| c[i])).collect();
        let hier = [
            self.alpha_loc,
            self.alpha_scale,
            self.beta_loc,
            self.beta_scale,
        ];
        let density = |_| Density {
            e: &e,
            n,
            k,
            kind: Kind::Hierarchical { x: &xr, p, hier },
        };
        sample(
            density,
            vec![0.0; (k - 1) * (1 + p)],
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
    hier: Option<[f64; 4]>,
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

    /// Intercept and slopes of model `m < K - 1` in draw `d`, on the
    /// natural scale (the prior's non-centring undone).
    fn coefficients(&self, d: usize, m: usize) -> (f64, Vec<f64>) {
        let raw = self.draw(d);
        let p = self.p;
        match self.hier {
            None => (raw[m], Vec::new()),
            Some([al, asc, bl, bsc]) => {
                let alpha = al + asc * raw[m];
                let off = (self.k - 1) + m * p;
                let beta = raw[off..off + p].iter().map(|b| bl + bsc * b).collect();
                (alpha, beta)
            }
        }
    }

    /// Posterior draws of the intercepts `α` (`n_draws × (K - 1)`, row-major;
    /// the logits for [`BayesStacking`]).
    pub fn alpha_draws(&self) -> Vec<f64> {
        (0..self.n_draws())
            .flat_map(|d| (0..self.k - 1).map(move |m| (d, m)))
            .map(|(d, m)| self.coefficients(d, m).0)
            .collect()
    }

    /// Posterior draws of the slopes `β`, row-major
    /// `n_draws × (K - 1) × p`; empty for [`BayesStacking`].
    pub fn beta_draws(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for d in 0..self.n_draws() {
            for m in 0..self.k - 1 {
                out.extend(self.coefficients(d, m).1);
            }
        }
        out
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
        for d in 0..self.n_draws() {
            let coef: Vec<(f64, Vec<f64>)> = (0..k - 1).map(|m| self.coefficients(d, m)).collect();
            for i in 0..n {
                for (m, (a, b)) in coef.iter().enumerate() {
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
    Hierarchical {
        x: &'a [f64],
        p: usize,
        hier: [f64; 4],
    },
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
        match self.kind {
            Kind::Dirichlet(_) => self.k - 1,
            Kind::Hierarchical { p, .. } => (self.k - 1) * (1 + p),
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
            Kind::Hierarchical {
                x,
                p,
                hier: [al, asc, bl, bsc],
            } => {
                let alpha: Vec<f64> = (0..k - 1).map(|m| al + asc * position[m]).collect();
                let beta = &position[k - 1..];
                for (i, row) in self.e.chunks_exact(k).take(n).enumerate() {
                    let xi = &x[i * p..(i + 1) * p];
                    for m in 0..k - 1 {
                        let b = &beta[m * p..(m + 1) * p];
                        logits[m] = alpha[m]
                            + b.iter()
                                .zip(xi)
                                .map(|(bj, xj)| (bl + bsc * bj) * xj)
                                .sum::<f64>();
                    }
                    softmax_ref(&logits, &mut w);
                    let mix: f64 = row.iter().zip(&w).map(|(e, wk)| e * wk).sum();
                    logp += mix.ln();
                    for m in 0..k - 1 {
                        let ds = w[m] * row[m] / mix - w[m];
                        grad[m] += asc * ds;
                        for (j, xj) in xi.iter().enumerate() {
                            grad[k - 1 + m * p + j] += bsc * ds * xj;
                        }
                    }
                }
                // Standard normal priors on the non-centred parameters.
                for (g, v) in grad.iter_mut().zip(position) {
                    logp -= 0.5 * v * v;
                    *g -= v;
                }
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
        let kinds = [
            (Kind::Dirichlet(&a), vec![0.2, -0.4]),
            (
                Kind::Hierarchical {
                    x: &x,
                    p: 2,
                    hier: [0.1, 1.3, -0.2, 0.7],
                },
                vec![0.2, -0.4, 0.5, -0.3, 0.1, 0.8],
            ),
        ];
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
    }
}
