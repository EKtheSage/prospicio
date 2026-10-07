//! Bayesian GLMs sampled with NUTS, by `nuts-rs`, the Rust core of nutpie
//! (`docs/design/models.md`: "Sampling: nutpie").
//!
//! [`BayesGlm`] is the GLM of `prospicio-glm` with normal priors on the
//! coefficients and, for the Gaussian, gamma and inverse Gaussian, a
//! half-normal prior on the dispersion, which is then sampled too (on the
//! log scale). Chains start near the maximum-likelihood fit, run in
//! parallel, and replay exactly from a seed: chain `c` takes its random
//! numbers from stream `c` of the seed.
//!
//! The fit gives posterior draws and their diagnostics (R̂, bulk and tail
//! ESS), the pointwise log-likelihood for PSIS-LOO and WAIC
//! ([`crate::elpd`]), and implements [`Fitted`]: posterior-mean
//! predictions and posterior predictive draws.

use std::collections::HashMap;
use std::fmt;

use nuts_rs::{CpuLogpFunc, CpuMathError, HasDims, LogpError};
use prospicio_core::{Error, Result};
use prospicio_math::special::digamma;
use prospicio_models::{Design, Family, Fitted, Link, Model};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};
use rayon::prelude::*;

pub use crate::nuts::{ParamSummary, Sampler};
use crate::nuts::{run_chain, summarize};

/// How the dispersion `φ` is set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DispersionPrior {
    /// Fixed (1 for the Poisson, binomial and negative binomial).
    Fixed(f64),
    /// Sampled, with a half-normal prior of this scale on `φ`. For the
    /// Gaussian, gamma and inverse Gaussian.
    HalfNormal(f64),
}

/// A Bayesian GLM: family, link, priors and sampler.
///
/// Every coefficient has a normal prior with mean 0: standard deviation
/// `intercept_sd` for an all-ones column, `prior_sd` for the rest
/// (on the link scale, so standardize covariates for the default to be
/// weakly informative).
///
/// ```
/// use prospicio_bayes::glm::{BayesGlm, Sampler};
/// use prospicio_models::{Design, Family, Link, Model};
///
/// let x: Vec<f64> = (0..40).map(|i| (i % 4) as f64 - 1.5).collect();
/// let y: Vec<f64> = (0..40).map(|i| [1.0, 2.0, 3.0, 5.0][i % 4]).collect();
/// let d = Design::new(vec!["(Intercept)".into(), "x".into()], vec![vec![1.0; 40], x]).unwrap();
/// let spec = BayesGlm::new(Family::Poisson, Link::Log).sampler(Sampler {
///     chains: 2,
///     tune: 300,
///     draws: 300,
///     ..Sampler::default()
/// });
/// let fit = spec.fit(&d, &y).unwrap();
/// let summary = fit.summary().unwrap();
/// assert!(summary.iter().all(|s| s.rhat < 1.05));
/// assert!((summary[1].mean - 0.5).abs() < 0.15);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BayesGlm {
    pub family: Family,
    pub link: Link,
    pub prior_sd: f64,
    pub intercept_sd: f64,
    pub dispersion: DispersionPrior,
    pub sampler: Sampler,
}

impl BayesGlm {
    /// Priors `N(0, 2.5²)` on slopes and `N(0, 10²)` on the intercept; the
    /// dispersion fixed at 1 for the Poisson, binomial and negative
    /// binomial, sampled with a half-normal(10) prior for the Gaussian,
    /// gamma and inverse Gaussian; a Tweedie needs a fixed dispersion.
    pub fn new(family: Family, link: Link) -> Self {
        let dispersion = match family {
            Family::Gaussian | Family::Gamma | Family::InverseGaussian => {
                DispersionPrior::HalfNormal(10.0)
            }
            _ => DispersionPrior::Fixed(1.0),
        };
        Self {
            family,
            link,
            prior_sd: 2.5,
            intercept_sd: 10.0,
            dispersion,
            sampler: Sampler::default(),
        }
    }

    /// The same model with other sampler settings.
    pub fn sampler(mut self, sampler: Sampler) -> Self {
        self.sampler = sampler;
        self
    }

    /// The same model with another dispersion prior.
    pub fn dispersion(mut self, dispersion: DispersionPrior) -> Self {
        self.dispersion = dispersion;
        self
    }

    fn check(&self) -> Result<()> {
        self.family.validate()?;
        let positive = |name, v: f64| {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(Error::InvalidParameter {
                    name,
                    value: v,
                    reason: "must be finite and positive",
                })
            }
        };
        positive("prior_sd", self.prior_sd)?;
        positive("intercept_sd", self.intercept_sd)?;
        match self.dispersion {
            DispersionPrior::Fixed(v) => positive("dispersion", v)?,
            DispersionPrior::HalfNormal(s) => {
                positive("dispersion scale", s)?;
                if !matches!(
                    self.family,
                    Family::Gaussian | Family::Gamma | Family::InverseGaussian
                ) {
                    return Err(Error::Data(format!(
                        "the dispersion of a {} GLM cannot be sampled; fix it",
                        self.family.name()
                    )));
                }
            }
        }
        self.sampler.validate()
    }
}

/// A sampled Bayesian GLM.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesGlmFit {
    spec: BayesGlm,
    names: Vec<String>,
    /// Chain-major, then draw, then coefficient.
    coefficients: Vec<f64>,
    /// One per draw (chain-major); constant when the dispersion is fixed.
    dispersion: Vec<f64>,
    divergences: usize,
}

impl Model for BayesGlm {
    type Fitted = BayesGlmFit;

    /// Samples the posterior: `chains` chains in parallel, each started
    /// at the maximum-likelihood estimates (or 0 if that fit fails), with
    /// a small jitter, then `tune` warm-up and `draws` kept draws.
    fn fit(&self, design: &Design, y: &[f64]) -> Result<BayesGlmFit> {
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
            return Err(Error::Data(format!(
                "response {bad} is outside the family's range"
            )));
        }
        let sampled = matches!(self.dispersion, DispersionPrior::HalfNormal(_));
        let prior_sd: Vec<f64> = (0..p)
            .map(|j| {
                if design.column(j).iter().all(|&v| v == 1.0) {
                    self.intercept_sd
                } else {
                    self.prior_sd
                }
            })
            .collect();
        // Start near the maximum-likelihood fit.
        let mle = prospicio_glm::Glm::new(self.family, self.link)
            .fit(design, y)
            .ok();
        let mut start: Vec<f64> = match &mle {
            Some(f) => f.coefficients().to_vec(),
            None => vec![0.0; p],
        };
        if sampled {
            let phi = mle.as_ref().map_or(1.0, |f| f.dispersion()).max(1e-8);
            start.push(phi.ln());
        }
        let s = self.sampler;
        let results: Vec<Result<(Vec<f64>, usize)>> = (0..s.chains)
            .into_par_iter()
            .map(|c| {
                let density = Posterior {
                    family: self.family,
                    link: self.link,
                    design,
                    y,
                    prior_sd: &prior_sd,
                    dispersion: self.dispersion,
                };
                run_chain(density, &start, s, c)
            })
            .collect();
        let mut coefficients = Vec::with_capacity(s.chains * s.draws * p);
        let mut dispersion = Vec::with_capacity(s.chains * s.draws);
        let mut divergences = 0;
        for r in results {
            let (draws, div) = r?;
            divergences += div;
            let width = p + usize::from(sampled);
            for d in draws.chunks_exact(width) {
                coefficients.extend_from_slice(&d[..p]);
                dispersion.push(match self.dispersion {
                    DispersionPrior::Fixed(v) => v,
                    DispersionPrior::HalfNormal(_) => d[p].exp(),
                });
            }
        }
        Ok(BayesGlmFit {
            spec: self.clone(),
            names: design.names().to_vec(),
            coefficients,
            dispersion,
            divergences,
        })
    }
}

/// The log posterior density on the unconstrained scale: `β`, then
/// `log φ` when the dispersion is sampled.
struct Posterior<'a> {
    family: Family,
    link: Link,
    design: &'a Design,
    y: &'a [f64],
    prior_sd: &'a [f64],
    dispersion: DispersionPrior,
}

impl fmt::Debug for Posterior<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Posterior")
            .field("family", &self.family)
            .field("link", &self.link)
            .finish_non_exhaustive()
    }
}

/// A position where the density is not defined (a mean outside the
/// family's range): NUTS treats it as a divergence.
#[derive(Debug)]
struct OutOfRange;

impl fmt::Display for OutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the mean left the family's range")
    }
}

impl std::error::Error for OutOfRange {}

impl LogpError for OutOfRange {
    fn is_recoverable(&self) -> bool {
        true
    }
}

impl HasDims for Posterior<'_> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        [("parameter".to_string(), self.dim() as u64)]
            .into_iter()
            .collect()
    }
}

impl CpuLogpFunc for Posterior<'_> {
    type LogpError = OutOfRange;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.design.n_cols()
            + usize::from(matches!(self.dispersion, DispersionPrior::HalfNormal(_)))
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> std::result::Result<f64, OutOfRange> {
        let p = self.design.n_cols();
        let beta = &position[..p];
        let phi = match self.dispersion {
            DispersionPrior::Fixed(v) => v,
            DispersionPrior::HalfNormal(_) => position[p].exp(),
        };
        if !(phi.is_finite() && phi > 0.0) {
            return Err(OutOfRange);
        }
        grad.iter_mut().for_each(|g| *g = 0.0);
        let eta = self.design.linear_predictor(beta);
        let w = self.design.weights();
        let mut logp = 0.0;
        let mut dphi = 0.0;
        for i in 0..self.y.len() {
            let mu = self.link.inverse(eta[i]);
            if !(self.link.valid_eta(eta[i]) && self.family.valid_mu(mu)) {
                return Err(OutOfRange);
            }
            let ll = self.family.log_likelihood(self.y[i], mu, w[i], phi);
            if !ll.is_finite() {
                return Err(OutOfRange);
            }
            logp += ll;
            // Score in η: w (y - μ) μ'(η) / (φ V(μ)).
            let s = w[i] * (self.y[i] - mu) * self.link.mu_eta(eta[i])
                / (phi * self.family.variance(mu));
            for (j, g) in grad[..p].iter_mut().enumerate() {
                *g += s * self.design.column(j)[i];
            }
            if matches!(self.dispersion, DispersionPrior::HalfNormal(_)) {
                dphi += dll_dphi(self.family, self.y[i], mu, w[i], phi);
            }
        }
        for j in 0..p {
            let sd = self.prior_sd[j];
            logp -= 0.5 * (beta[j] / sd).powi(2);
            grad[j] -= beta[j] / (sd * sd);
        }
        if let DispersionPrior::HalfNormal(scale) = self.dispersion {
            // Half-normal on φ, plus the Jacobian log φ of φ = exp(position).
            logp += -0.5 * (phi / scale).powi(2) + phi.ln();
            grad[p] = phi * dphi - (phi / scale).powi(2) + 1.0;
        }
        Ok(logp)
    }

    fn expand_vector<R: nuts_rs::rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        position: &[f64],
    ) -> std::result::Result<Vec<f64>, CpuMathError> {
        Ok(position.to_vec())
    }
}

/// `∂ log f(y; μ, φ/w) / ∂φ` for the families whose dispersion is sampled.
fn dll_dphi(family: Family, y: f64, mu: f64, w: f64, phi: f64) -> f64 {
    match family {
        Family::Gaussian => -0.5 / phi + 0.5 * w * (y - mu).powi(2) / (phi * phi),
        Family::Gamma => {
            // Shape s = w / φ: ∂ℓ/∂s = ln(s y / μ) + 1 - y / μ - ψ(s).
            let s = w / phi;
            let r = y / mu;
            ((s * r).ln() + 1.0 - r - digamma(s)) * (-s / phi)
        }
        Family::InverseGaussian => {
            -0.5 / phi + 0.5 * w * (y - mu).powi(2) / (phi * phi * y * mu * mu)
        }
        _ => 0.0,
    }
}

impl BayesGlmFit {
    /// The specification that was sampled.
    pub fn spec(&self) -> &BayesGlm {
        &self.spec
    }

    /// Coefficient names, as the design's columns.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Number of chains.
    pub fn chains(&self) -> usize {
        self.spec.sampler.chains
    }

    /// Kept draws per chain.
    pub fn draws_per_chain(&self) -> usize {
        self.spec.sampler.draws
    }

    /// Total kept draws, `chains × draws`.
    pub fn n_draws(&self) -> usize {
        self.dispersion.len()
    }

    /// Divergent transitions among the kept draws: any at all means the
    /// posterior geometry may be biasing the draws.
    pub fn divergences(&self) -> usize {
        self.divergences
    }

    /// Coefficient draws, row-major `n_draws × p`, chain by chain.
    pub fn coefficient_draws(&self) -> &[f64] {
        &self.coefficients
    }

    /// Dispersion draws, one per draw (constant when fixed).
    pub fn dispersion_draws(&self) -> &[f64] {
        &self.dispersion
    }

    /// Posterior means of the coefficients.
    pub fn posterior_mean(&self) -> Vec<f64> {
        let p = self.names.len();
        let s = self.n_draws() as f64;
        let mut m = vec![0.0; p];
        for d in self.coefficients.chunks_exact(p) {
            for (mj, v) in m.iter_mut().zip(d) {
                *mj += v / s;
            }
        }
        m
    }

    /// One parameter's draws split by chain.
    fn chains_of(&self, values: impl Fn(usize) -> f64) -> Vec<Vec<f64>> {
        let k = self.draws_per_chain();
        (0..self.chains())
            .map(|c| (0..k).map(|d| values(c * k + d)).collect())
            .collect()
    }

    /// Mean, sd, 5/50/95% quantiles, R̂ and bulk and tail ESS of each
    /// coefficient, then of the dispersion when it was sampled.
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        let p = self.names.len();
        let mut out = Vec::with_capacity(p + 1);
        let mut one = |name: String, chains: Vec<Vec<f64>>| -> Result<()> {
            out.push(summarize(name, &chains)?);
            Ok(())
        };
        for j in 0..p {
            one(
                self.names[j].clone(),
                self.chains_of(|d| self.coefficients[d * p + j]),
            )?;
        }
        if matches!(self.spec.dispersion, DispersionPrior::HalfNormal(_)) {
            one("dispersion".into(), self.chains_of(|d| self.dispersion[d]))?;
        }
        Ok(out)
    }

    /// Pointwise log-likelihood, row-major `n_draws × n` (the layout of
    /// [`crate::elpd`]), of `y` given `design`.
    pub fn log_likelihood(&self, design: &Design, y: &[f64]) -> Result<Vec<f64>> {
        self.check_design(design)?;
        if y.len() != design.n_rows() {
            return Err(Error::Data(format!(
                "{} responses for {} design rows",
                y.len(),
                design.n_rows()
            )));
        }
        let p = self.names.len();
        let w = design.weights();
        let (family, link) = (self.spec.family, self.spec.link);
        let rows: Vec<Vec<f64>> = self
            .coefficients
            .par_chunks_exact(p)
            .zip(self.dispersion.par_iter())
            .map(|(beta, &phi)| {
                design
                    .linear_predictor(beta)
                    .iter()
                    .enumerate()
                    .map(|(i, &e)| family.log_likelihood(y[i], link.inverse(e), w[i], phi))
                    .collect()
            })
            .collect();
        Ok(rows.concat())
    }

    /// PSIS-LOO of `y` given `design` (normally the training data), with
    /// each observation's relative efficiency `r_eff` estimated from the
    /// chains, as `loo::relative_eff` does.
    pub fn loo(&self, design: &Design, y: &[f64]) -> Result<crate::elpd::Loo> {
        let ll = self.log_likelihood(design, y)?;
        let n = y.len();
        let s = self.n_draws() as f64;
        let r_eff: Vec<f64> = (0..n)
            .map(|i| {
                let chains = self.chains_of(|d| ll[d * n + i].exp());
                let refs: Vec<&[f64]> = chains.iter().map(Vec::as_slice).collect();
                let ess = crate::ess_mean(&refs).unwrap_or(s);
                // Constant likelihood: as independent draws.
                if ess.is_finite() && ess > 0.0 {
                    ess / s
                } else {
                    1.0
                }
            })
            .collect();
        crate::elpd::loo(&ll, n, Some(&r_eff))
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

impl Fitted for BayesGlmFit {
    /// Posterior mean of each row's mean `μ`.
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        self.check_design(design)?;
        let p = self.names.len();
        let s = self.n_draws() as f64;
        let mut out = vec![0.0; design.n_rows()];
        for beta in self.coefficients.chunks_exact(p) {
            for (o, e) in out.iter_mut().zip(design.linear_predictor(beta)) {
                *o += self.spec.link.inverse(e) / s;
            }
        }
        Ok(out)
    }

    /// Posterior predictive draws, components keyed `row = 0, 1, …`:
    /// simulation `i` takes a posterior draw (coefficients and dispersion)
    /// chosen by stream `i` of `seed`, shared by every row, then each
    /// row's response from the family.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        self.check_design(design)?;
        let p = self.names.len();
        let n = design.n_rows();
        let components: Vec<ComponentKey> =
            (0..n).map(|i| vec![KeyValue::from(i as i64)]).collect();
        let (family, link) = (self.spec.family, self.spec.link);
        let w = design.weights();
        let total = self.n_draws();
        let provenance = Provenance::new("bayes_glm")
            .version("prospicio-bayes", env!("CARGO_PKG_VERSION"))
            .param("family", family.name())
            .param("link", format!("{link:?}"))
            .param("posterior_draws", total);
        PredictiveDistribution::simulate(
            vec!["row".into()],
            components,
            n_sims,
            seed,
            provenance,
            |rng, row| {
                let d = ((rng.next_open01() * total as f64) as usize).min(total - 1);
                let beta = &self.coefficients[d * p..(d + 1) * p];
                let phi = self.dispersion[d];
                for (i, (out, e)) in row
                    .iter_mut()
                    .zip(design.linear_predictor(beta))
                    .enumerate()
                {
                    let mu = link.inverse(e);
                    *out = if family.valid_mu(mu) {
                        family
                            .draw(mu, phi, w[i], rng.next_open01())
                            .unwrap_or(f64::NAN)
                    } else {
                        f64::NAN
                    };
                }
            },
        )
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
            seed: 3,
            ..Sampler::default()
        }
    }

    #[test]
    fn the_gradient_matches_a_finite_difference() {
        let x: Vec<f64> = (0..30).map(|i| (i % 5) as f64 / 2.0 - 1.0).collect();
        let w: Vec<f64> = (0..30).map(|i| 0.5 + (i % 3) as f64).collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; 30], x],
        )
        .unwrap()
        .with_weights(w)
        .unwrap();
        let y: Vec<f64> = (0..30).map(|i| 0.5 + ((i * 7) % 5) as f64).collect();
        let prior = [10.0, 2.5];
        let cases = [
            (
                Family::Gaussian,
                Link::Identity,
                DispersionPrior::HalfNormal(3.0),
            ),
            (Family::Gamma, Link::Log, DispersionPrior::HalfNormal(3.0)),
            (
                Family::InverseGaussian,
                Link::Log,
                DispersionPrior::HalfNormal(3.0),
            ),
            (Family::Poisson, Link::Log, DispersionPrior::Fixed(1.0)),
            (
                Family::NegativeBinomial { theta: 2.0 },
                Link::Log,
                DispersionPrior::Fixed(1.0),
            ),
            (
                Family::Tweedie { power: 1.5 },
                Link::Log,
                DispersionPrior::Fixed(0.7),
            ),
        ];
        for (family, link, dispersion) in cases {
            let mut post = Posterior {
                family,
                link,
                design: &d,
                y: &y,
                prior_sd: &prior,
                dispersion,
            };
            let at: Vec<f64> = [0.9, 0.2, -0.4][..post.dim()].to_vec();
            let mut g = vec![0.0; at.len()];
            post.logp(&at, &mut g).unwrap();
            for k in 0..at.len() {
                let h = 1e-6;
                let (mut up, mut down) = (at.clone(), at.clone());
                up[k] += h;
                down[k] -= h;
                let mut scratch = vec![0.0; at.len()];
                let numeric = (post.logp(&up, &mut scratch).unwrap()
                    - post.logp(&down, &mut scratch).unwrap())
                    / (2.0 * h);
                assert!(
                    (g[k] - numeric).abs() < 1e-5 * numeric.abs().max(1.0),
                    "{family:?} {k}: {} vs {numeric}",
                    g[k]
                );
            }
        }
    }

    #[test]
    fn sampling_replays_from_the_seed_and_predicts() {
        let x: Vec<f64> = (0..60).map(|i| (i % 6) as f64 / 3.0 - 0.8).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|v| 2.0 + 1.5 * v + [0.3, -0.2, 0.1][(v * 10.0) as usize % 3])
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; 60], x],
        )
        .unwrap();
        let spec = BayesGlm::new(Family::Gaussian, Link::Identity).sampler(small());
        let a = spec.fit(&d, &y).unwrap();
        let b = spec.fit(&d, &y).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.n_draws(), 800);
        let summary = a.summary().unwrap();
        assert_eq!(summary.len(), 3);
        assert!(summary.iter().all(|s| s.rhat < 1.05), "{summary:?}");
        let mle = prospicio_glm::Glm::new(Family::Gaussian, Link::Identity)
            .fit(&d, &y)
            .unwrap();
        for (j, s) in summary.iter().take(2).enumerate() {
            let se = mle.std_errors()[j];
            assert!((s.mean - mle.coefficients()[j]).abs() < 0.3 * se);
        }
        let pred = a.predict(&d).unwrap();
        assert!((pred[0] - mle.fitted()[0]).abs() < 0.05);
        let pd = a.predict_distribution(&d, 500, 1).unwrap();
        assert_eq!(pd.n_sims(), 500);
        let loo = a.loo(&d, &y).unwrap();
        assert!(loo.estimate.elpd.is_finite());
        assert_eq!(a.log_likelihood(&d, &y).unwrap().len(), 800 * 60);
    }

    #[test]
    fn rejects_bad_specs() {
        let d = Design::new(vec!["x".into()], vec![vec![1.0; 4]]).unwrap();
        let y = [1.0, 2.0, 0.0, 1.0];
        let poisson = BayesGlm::new(Family::Poisson, Link::Log);
        assert!(
            poisson
                .clone()
                .dispersion(DispersionPrior::HalfNormal(1.0))
                .fit(&d, &y)
                .is_err()
        );
        let mut no_draws = poisson.clone();
        no_draws.sampler.draws = 1;
        assert!(no_draws.fit(&d, &y).is_err());
        assert!(poisson.fit(&d, &[1.0, -1.0, 0.0, 1.0]).is_err());
    }
}
