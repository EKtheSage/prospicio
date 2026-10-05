//! The No-U-Turn sampler (NUTS) for any differentiable log density, by
//! `nuts-rs`, the Rust core of nutpie (`docs/design/models.md`:
//! "Sampling: nutpie").
//!
//! A model implements [`LogDensity`] on an unconstrained parameter
//! vector (log scales, logits), and [`sample`] runs the chains in
//! parallel. Chain `c` takes its random numbers from stream `c` of the
//! seed, so a run replays exactly. The draws ([`PosteriorDraws`]) carry
//! their diagnostics (R̂, bulk and tail ESS, divergences), map to the
//! natural scale with [`PosteriorDraws::transform`], and become a
//! posterior predictive [`PredictiveDistribution`] with
//! [`PosteriorDraws::predictive`], which then feeds blending, capital
//! allocation, reinsurance and pricing like any other simulation.
//!
//! [`crate::glm::BayesGlm`] and [`crate::stacking`] sample through the
//! same driver.

use std::collections::HashMap;
use std::fmt;

use act_core::{Error, Result, StreamRng};
use act_prob::{ComponentKey, PredictiveDistribution, Provenance};
use nuts_rs::rand::SeedableRng;
use nuts_rs::rand::rngs::ChaCha20Rng;
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
};
use rayon::prelude::*;

/// NUTS settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampler {
    pub chains: usize,
    /// Warm-up draws per chain (step size and mass matrix adaptation),
    /// discarded.
    pub tune: usize,
    /// Kept draws per chain.
    pub draws: usize,
    pub seed: u64,
    /// Target mean acceptance rate for step-size adaptation.
    pub target_accept: f64,
    /// Largest tree depth (at most `2^max_depth` leapfrog steps a draw).
    pub max_depth: u64,
}

impl Default for Sampler {
    /// Four chains of 1000 warm-up and 1000 kept draws, target acceptance
    /// 0.8, tree depth at most 10, seed 0.
    fn default() -> Self {
        Self {
            chains: 4,
            tune: 1000,
            draws: 1000,
            seed: 0,
            target_accept: 0.8,
            max_depth: 10,
        }
    }
}

impl Sampler {
    /// Checks the settings: at least one chain and two kept draws, and a
    /// target acceptance in `(0, 1)`.
    pub fn validate(&self) -> Result<()> {
        if self.chains == 0 || self.draws < 2 {
            return Err(Error::Data(
                "the sampler needs at least one chain and two draws".into(),
            ));
        }
        if !(self.target_accept > 0.0 && self.target_accept < 1.0) {
            return Err(Error::InvalidParameter {
                name: "target_accept",
                value: self.target_accept,
                reason: "must be in (0, 1)",
            });
        }
        Ok(())
    }
}

/// Posterior summary of one parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamSummary {
    pub name: String,
    pub mean: f64,
    pub sd: f64,
    /// 5%, 50% and 95% quantiles.
    pub q05: f64,
    pub q50: f64,
    pub q95: f64,
    pub rhat: f64,
    pub ess_bulk: f64,
    pub ess_tail: f64,
}

/// Mean, sd (divisor `n - 1`), 5/50/95% quantiles, R̂ and bulk and tail
/// ESS of one parameter's draws, split by chain.
pub(crate) fn summarize(name: String, chains: &[Vec<f64>]) -> Result<ParamSummary> {
    let refs: Vec<&[f64]> = chains.iter().map(Vec::as_slice).collect();
    let mut all: Vec<f64> = chains.concat();
    let n = all.len() as f64;
    let mean = all.iter().sum::<f64>() / n;
    let sd = (all.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    all.sort_by(f64::total_cmp);
    let q = |prob: f64| quantile_sorted(&all, prob);
    Ok(ParamSummary {
        name,
        mean,
        sd,
        q05: q(0.05),
        q50: q(0.5),
        q95: q(0.95),
        rhat: crate::rhat(&refs)?,
        ess_bulk: crate::ess_bulk(&refs)?,
        ess_tail: crate::ess_tail(&refs)?,
    })
}

/// A log density, up to a constant, on an unconstrained parameter
/// vector, with its gradient.
///
/// Put constrained parameters on an unconstrained scale (a log for a
/// scale, a logit for a probability) and add the log Jacobian of the map
/// to the density, so the posterior on the natural scale is the one
/// intended.
pub trait LogDensity: Sync {
    /// Number of parameters.
    fn dim(&self) -> usize;

    /// The log density at `position`, writing its gradient into
    /// `gradient` (length [`dim`](Self::dim)). `None`, or a value that is
    /// not finite, marks a position outside the support: the sampler
    /// treats it as a divergence and moves on.
    fn log_density(&self, position: &[f64], gradient: &mut [f64]) -> Option<f64>;
}

/// Runs NUTS on `density` from `start`, one chain per
/// [`Sampler::chains`], in parallel.
///
/// Each chain starts at `start` jittered by up to `±0.01` per
/// coordinate; a start near the mode (a maximum-likelihood fit) shortens
/// the warm-up.
///
/// ```
/// use act_bayes::nuts::{LogDensity, Sampler, sample};
///
/// /// A standard normal in two dimensions.
/// struct Normal2;
///
/// impl LogDensity for Normal2 {
///     fn dim(&self) -> usize {
///         2
///     }
///     fn log_density(&self, x: &[f64], grad: &mut [f64]) -> Option<f64> {
///         grad[0] = -x[0];
///         grad[1] = -x[1];
///         Some(-0.5 * (x[0] * x[0] + x[1] * x[1]))
///     }
/// }
///
/// let s = Sampler { chains: 2, tune: 300, draws: 500, seed: 1, ..Sampler::default() };
/// let post = sample(&Normal2, &[0.0, 0.0], s).unwrap();
/// assert_eq!(post.n_draws(), 1000);
/// let summary = post.summary().unwrap();
/// assert!(summary[0].mean.abs() < 0.2 && summary[0].rhat < 1.05);
/// ```
pub fn sample<D: LogDensity + ?Sized>(
    density: &D,
    start: &[f64],
    sampler: Sampler,
) -> Result<PosteriorDraws> {
    sampler.validate()?;
    let dim = density.dim();
    if dim == 0 || start.len() != dim {
        return Err(Error::Data(format!(
            "start has {} values for a density of dimension {dim}",
            start.len()
        )));
    }
    if let Some(&bad) = start.iter().find(|v| !v.is_finite()) {
        return Err(Error::InvalidParameter {
            name: "start",
            value: bad,
            reason: "must be finite",
        });
    }
    let mut grad = vec![0.0; dim];
    if !density
        .log_density(start, &mut grad)
        .is_some_and(f64::is_finite)
    {
        return Err(Error::Data(
            "the log density is not finite at the start".into(),
        ));
    }
    let results: Vec<Result<(Vec<f64>, usize)>> = (0..sampler.chains)
        .into_par_iter()
        .map(|c| run_chain(Adapter(density), start, sampler, c))
        .collect();
    let mut values = Vec::with_capacity(sampler.chains * sampler.draws * dim);
    let mut divergences = Vec::with_capacity(sampler.chains);
    for r in results {
        let (draws, div) = r?;
        values.extend(draws);
        divergences.push(div);
    }
    Ok(PosteriorDraws {
        names: (0..dim).map(|j| format!("theta[{j}]")).collect(),
        dim,
        chains: sampler.chains,
        values,
        divergences,
        seed: sampler.seed,
    })
}

/// Posterior draws from [`sample`]: chains of equal length, each draw a
/// parameter vector.
#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorDraws {
    names: Vec<String>,
    dim: usize,
    chains: usize,
    /// Chain-major, then draw, then parameter.
    values: Vec<f64>,
    divergences: Vec<usize>,
    seed: u64,
}

impl PosteriorDraws {
    /// Parameter names; `theta[0]`, `theta[1]`, … unless set with
    /// [`with_names`](Self::with_names) or [`transform`](Self::transform).
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The same draws with parameter names. Fails unless there is one
    /// name per parameter.
    pub fn with_names(mut self, names: Vec<String>) -> Result<Self> {
        if names.len() != self.dim {
            return Err(Error::Data(format!(
                "{} names for {} parameters",
                names.len(),
                self.dim
            )));
        }
        self.names = names;
        Ok(self)
    }

    /// Number of parameters.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of chains.
    pub fn chains(&self) -> usize {
        self.chains
    }

    /// Kept draws per chain.
    pub fn draws_per_chain(&self) -> usize {
        self.n_draws() / self.chains
    }

    /// Total draws over all chains.
    pub fn n_draws(&self) -> usize {
        self.values.len() / self.dim
    }

    /// Draw `d` (chain-major: chain `d / draws_per_chain`).
    pub fn draw(&self, d: usize) -> &[f64] {
        &self.values[d * self.dim..(d + 1) * self.dim]
    }

    /// Every draw, row-major (`n_draws × dim`), chain-major.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Divergent transitions among the kept draws, per chain.
    pub fn divergences(&self) -> &[usize] {
        &self.divergences
    }

    /// Parameter `j`'s draws, one vector per chain (the input of
    /// [`crate::rhat`] and the ESS estimators).
    pub fn parameter(&self, j: usize) -> Vec<Vec<f64>> {
        let k = self.draws_per_chain();
        (0..self.chains)
            .map(|c| {
                (0..k)
                    .map(|d| self.values[(c * k + d) * self.dim + j])
                    .collect()
            })
            .collect()
    }

    /// Posterior mean of each parameter.
    pub fn mean(&self) -> Vec<f64> {
        let n = self.n_draws() as f64;
        let mut m = vec![0.0; self.dim];
        for d in self.values.chunks_exact(self.dim) {
            for (mj, v) in m.iter_mut().zip(d) {
                *mj += v / n;
            }
        }
        m
    }

    /// [`ParamSummary`] of each parameter.
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        (0..self.dim)
            .map(|j| summarize(self.names[j].clone(), &self.parameter(j)))
            .collect()
    }

    /// Maps every draw through `f`, for example from the sampling scale
    /// to the natural one (`exp` of a log scale), keeping the chains, so
    /// the summary and diagnostics are on the new scale. `f` must return
    /// one value per name.
    ///
    /// ```
    /// # use act_bayes::nuts::{LogDensity, Sampler, sample};
    /// # struct Normal1;
    /// # impl LogDensity for Normal1 {
    /// #     fn dim(&self) -> usize { 1 }
    /// #     fn log_density(&self, x: &[f64], g: &mut [f64]) -> Option<f64> {
    /// #         g[0] = -x[0];
    /// #         Some(-0.5 * x[0] * x[0])
    /// #     }
    /// # }
    /// let s = Sampler { chains: 2, tune: 200, draws: 200, ..Sampler::default() };
    /// let log_sigma = sample(&Normal1, &[0.0], s).unwrap();
    /// let sigma = log_sigma.transform(vec!["sigma".into()], |x| vec![x[0].exp()]).unwrap();
    /// assert!(sigma.values().iter().all(|&v| v > 0.0));
    /// ```
    pub fn transform(&self, names: Vec<String>, f: impl Fn(&[f64]) -> Vec<f64>) -> Result<Self> {
        let dim = names.len();
        if dim == 0 {
            return Err(Error::Data("transform needs at least one name".into()));
        }
        let mut values = Vec::with_capacity(self.n_draws() * dim);
        for d in self.values.chunks_exact(self.dim) {
            let y = f(d);
            if y.len() != dim {
                return Err(Error::Data(format!(
                    "transform returned {} values for {dim} names",
                    y.len()
                )));
            }
            values.extend(y);
        }
        Ok(Self {
            names,
            dim,
            chains: self.chains,
            values,
            divergences: self.divergences.clone(),
            seed: self.seed,
        })
    }

    /// Posterior predictive draws: simulation `i` picks a posterior draw
    /// uniformly with stream `i` of `seed`, then calls
    /// `simulate(theta, rng, row)` with the same stream to draw the
    /// outcome given `theta`, one value per component. Parameter and
    /// process uncertainty are both in the result.
    ///
    /// ```
    /// # use act_bayes::nuts::{LogDensity, Sampler, sample};
    /// # use act_math::special::norm_quantile;
    /// # use act_prob::{Distribution, KeyValue, Provenance};
    /// # struct Normal1;
    /// # impl LogDensity for Normal1 {
    /// #     fn dim(&self) -> usize { 1 }
    /// #     fn log_density(&self, x: &[f64], g: &mut [f64]) -> Option<f64> {
    /// #         g[0] = -x[0];
    /// #         Some(-0.5 * x[0] * x[0])
    /// #     }
    /// # }
    /// let s = Sampler { chains: 2, tune: 300, draws: 1000, ..Sampler::default() };
    /// let mu = sample(&Normal1, &[0.0], s).unwrap();
    /// // Next year's loss is N(mu, 1); predictive variance 1 + Var(mu) = 2.
    /// let pd = mu
    ///     .predictive(
    ///         vec!["year".into()],
    ///         vec![vec![KeyValue::from(1)]],
    ///         20_000,
    ///         7,
    ///         Provenance::new("example"),
    ///         |theta, rng, row| row[0] = theta[0] + norm_quantile(rng.next_open01()),
    ///     )
    ///     .unwrap();
    /// assert!((pd.total().variance() - 2.0).abs() < 0.2);
    /// ```
    pub fn predictive<F>(
        &self,
        dims: Vec<String>,
        components: Vec<ComponentKey>,
        n_sims: usize,
        seed: u64,
        provenance: Provenance,
        simulate: F,
    ) -> Result<PredictiveDistribution>
    where
        F: Fn(&[f64], &mut StreamRng, &mut [f64]) + Sync,
    {
        let total = self.n_draws();
        let provenance = provenance
            .param("posterior_draws", total)
            .param("posterior_seed", self.seed);
        PredictiveDistribution::simulate(dims, components, n_sims, seed, provenance, |rng, row| {
            let d = ((rng.next_open01() * total as f64) as usize).min(total - 1);
            simulate(self.draw(d), rng, row);
        })
    }
}

/// A [`LogDensity`] as `nuts-rs` wants it.
struct Adapter<'a, D: ?Sized>(&'a D);

impl<D: ?Sized> fmt::Debug for Adapter<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("LogDensity")
    }
}

/// A position outside the support.
#[derive(Debug)]
struct OutsideSupport;

impl fmt::Display for OutsideSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the log density is not finite here")
    }
}

impl std::error::Error for OutsideSupport {}

impl LogpError for OutsideSupport {
    fn is_recoverable(&self) -> bool {
        true
    }
}

impl<D: LogDensity + ?Sized> HasDims for Adapter<'_, D> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        [("parameter".to_string(), self.0.dim() as u64)]
            .into_iter()
            .collect()
    }
}

impl<D: LogDensity + ?Sized> CpuLogpFunc for Adapter<'_, D> {
    type LogpError = OutsideSupport;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn logp(
        &mut self,
        position: &[f64],
        grad: &mut [f64],
    ) -> std::result::Result<f64, OutsideSupport> {
        match self.0.log_density(position, grad) {
            Some(v) if v.is_finite() && grad.iter().all(|g| g.is_finite()) => Ok(v),
            _ => Err(OutsideSupport),
        }
    }

    fn expand_vector<R: nuts_rs::rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> std::result::Result<Vec<f64>, CpuMathError> {
        Ok(array.to_vec())
    }
}

/// Runs one chain; returns its kept draws (row-major) and divergences.
pub(crate) fn run_chain<F>(
    density: F,
    start: &[f64],
    s: Sampler,
    chain: usize,
) -> Result<(Vec<f64>, usize)>
where
    F: CpuLogpFunc<FlowParameters = (), ExpandedVector = Vec<f64>>,
{
    let dim = start.len();
    // The chain's random numbers: ChaCha20 keyed from stream `chain` of
    // the seed, so chains are independent and replay exactly.
    let mut key_stream = StreamRng::new(s.seed, chain as u64);
    let mut key = [0u8; 32];
    for chunk in key.chunks_exact_mut(8) {
        chunk.copy_from_slice(&key_stream.next_u64().to_le_bytes());
    }
    let mut rng = ChaCha20Rng::from_seed(key);
    let mut settings = DiagNutsSettings {
        num_tune: s.tune as u64,
        num_draws: s.draws as u64,
        maxdepth: s.max_depth,
        ..DiagNutsSettings::default()
    };
    settings.adapt_options.step_size_settings.target_accept = s.target_accept;
    let mut sampler = settings
        .new_chain(chain as u64, CpuMath::new(density), &mut rng)
        .map_err(|e| Error::Data(format!("NUTS setup failed: {e}")))?;
    let init: Vec<f64> = start
        .iter()
        .map(|v| v + 0.02 * (key_stream.next_open01() - 0.5))
        .collect();
    sampler
        .set_position(&init)
        .map_err(|e| Error::Data(format!("NUTS could not start: {e}")))?;
    let mut draws = Vec::with_capacity(s.draws * dim);
    let mut divergences = 0;
    for i in 0..s.tune + s.draws {
        let (draw, progress) = sampler
            .draw()
            .map_err(|e| Error::Data(format!("NUTS failed: {e}")))?;
        if i >= s.tune {
            draws.extend_from_slice(&draw);
            divergences += usize::from(progress.diverging);
        }
    }
    Ok((draws, divergences))
}

fn quantile_sorted(sorted: &[f64], prob: f64) -> f64 {
    // Linear interpolation (R's type 7).
    let h = (sorted.len() - 1) as f64 * prob;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

#[cfg(test)]
mod tests {
    use super::*;
    use act_prob::{Distribution, KeyValue};

    /// A correlated bivariate normal: means (1, -2), sds (1, 3), corr 0.8.
    struct Correlated;

    impl LogDensity for Correlated {
        fn dim(&self) -> usize {
            2
        }
        fn log_density(&self, x: &[f64], g: &mut [f64]) -> Option<f64> {
            let (s1, s2, r) = (1.0, 3.0, 0.8);
            let z1 = (x[0] - 1.0) / s1;
            let z2 = (x[1] + 2.0) / s2;
            let k = 1.0 / (1.0 - r * r);
            g[0] = -k * (z1 - r * z2) / s1;
            g[1] = -k * (z2 - r * z1) / s2;
            Some(-0.5 * k * (z1 * z1 - 2.0 * r * z1 * z2 + z2 * z2))
        }
    }

    /// A half-line: the density is undefined below 0.
    struct Exponential;

    impl LogDensity for Exponential {
        fn dim(&self) -> usize {
            1
        }
        fn log_density(&self, x: &[f64], g: &mut [f64]) -> Option<f64> {
            g[0] = -1.0;
            (x[0] >= 0.0).then_some(-x[0])
        }
    }

    fn sampler(seed: u64) -> Sampler {
        Sampler {
            chains: 4,
            tune: 500,
            draws: 1000,
            seed,
            ..Sampler::default()
        }
    }

    #[test]
    fn recovers_a_correlated_normal() {
        let post = sample(&Correlated, &[0.0, 0.0], sampler(3)).unwrap();
        let s = post.summary().unwrap();
        for (p, (mean, sd)) in s.iter().zip([(1.0, 1.0), (-2.0, 3.0)]) {
            // Within four Monte Carlo standard errors.
            assert!(
                (p.mean - mean).abs() < 4.0 * sd / p.ess_bulk.sqrt(),
                "{p:?}"
            );
            assert!((p.sd / sd - 1.0).abs() < 0.1, "{p:?}");
            assert!(p.rhat < 1.01 && p.ess_bulk > 400.0, "{p:?}");
        }
        assert_eq!(post.divergences().iter().sum::<usize>(), 0);
        assert_eq!(s[0].name, "theta[0]");
    }

    #[test]
    fn replays_from_the_seed() {
        let a = sample(&Correlated, &[0.0, 0.0], sampler(5)).unwrap();
        let b = sample(&Correlated, &[0.0, 0.0], sampler(5)).unwrap();
        let c = sample(&Correlated, &[0.0, 0.0], sampler(6)).unwrap();
        assert_eq!(a, b);
        assert_ne!(a.values(), c.values());
        assert_eq!(a.parameter(1)[2][7], a.draw(2 * 1000 + 7)[1]);
    }

    #[test]
    fn treats_positions_outside_the_support_as_divergent() {
        let post = sample(&Exponential, &[1.0], sampler(1)).unwrap();
        assert!(post.values().iter().all(|&v| v >= 0.0));
        let s = post.summary().unwrap();
        assert!((s[0].mean - 1.0).abs() < 0.15, "{:?}", s[0]);
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(sample(&Correlated, &[0.0], sampler(1)).is_err());
        assert!(sample(&Exponential, &[-1.0], sampler(1)).is_err());
        assert!(sample(&Exponential, &[f64::NAN], sampler(1)).is_err());
        let bad = Sampler {
            chains: 0,
            ..sampler(1)
        };
        assert!(sample(&Exponential, &[1.0], bad).is_err());
    }

    #[test]
    fn transform_and_predictive() {
        let post = sample(&Correlated, &[0.0, 0.0], sampler(2)).unwrap();
        let named = post
            .clone()
            .with_names(vec!["a".into(), "b".into()])
            .unwrap();
        assert_eq!(named.summary().unwrap()[1].name, "b");
        assert!(post.clone().with_names(vec!["a".into()]).is_err());
        let sum = post
            .transform(vec!["sum".into()], |x| vec![x[0] + x[1]])
            .unwrap();
        assert_eq!(sum.dim(), 1);
        assert_eq!(sum.chains(), 4);
        assert!((sum.mean()[0] - (post.mean()[0] + post.mean()[1])).abs() < 1e-12);
        assert!(post.transform(vec!["x".into()], |x| x.to_vec()).is_err());

        // Two components that share the posterior draw.
        let pd = post
            .predictive(
                vec!["k".into()],
                vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
                5_000,
                9,
                Provenance::new("test"),
                |theta, _, row| row.copy_from_slice(theta),
            )
            .unwrap();
        let a = pd.marginal(&vec![KeyValue::from("a")]).unwrap();
        assert!((a.mean() - 1.0).abs() < 0.15);
        assert_eq!(pd.provenance().seed, Some(9));
    }
}
