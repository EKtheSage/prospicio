//! The Bayesian over-dispersed Poisson chain ladder
//! (`docs/design/reserving-bayes.md`, decisions 3 to 6).

use prospicio_bayes::elpd::Loo;
use prospicio_bayes::nuts::{ParamSummary, PosteriorDraws, Sampler, sample};
use prospicio_core::StreamRng;
use prospicio_models::Family;
use prospicio_prob::{InputHasher, PredictiveDistribution, Provenance};
use rayon::prelude::*;

use super::laplace::{Posterior, Whitened, mode};
use super::{Diagnostics, Stream, derive_seed};
use crate::bootstrap::{
    ProcessDistribution, component_sums, hash_segment, prepare, process_draw, push_moments,
    segment_sums,
};
use crate::chain_ladder::ChainLadderFit;
use crate::dependence::SegmentDependence;
use crate::error::{Error, Result};
use crate::segments::{FitTable, ReserveFit, SegmentFits};
use crate::tail::Tail;
use crate::triangle::{Label, Segment, Triangle};

/// The Bayesian chain ladder of England and Verrall (2002, 2006): the
/// over-dispersed Poisson model of the incremental values `X_wd`,
///
/// ```text
/// eta_wd = c + a_w + b_d,   a_1 = b_1 = 0
/// log p(X | theta) = sum over observed cells of (X_wd eta_wd - exp(eta_wd)) / phi
/// c ~ N(c0, intercept_sd^2),  a_w ~ N(0, origin_sd^2),  b_d ~ N(0, development_sd^2)
/// ```
///
/// with `phi` the chain ladder's Pearson scale (the
/// [`OdpBootstrapFit::scale`](crate::OdpBootstrapFit::scale)), plugged in,
/// and `c0` the log of the mean observed increment, which is the log of
/// the chain ladder's mean fitted increment. The log posterior is strictly
/// concave, so it has one mode; NUTS samples it on a scale whitened by the
/// negative Hessian at the mode, and the draws are mapped back to `c`,
/// `a[2]..a[n]` and `b[2]..b[m]`. The reserves add each future cell's
/// process error (`process`) to its mean `exp(eta_wd)` given a posterior
/// draw, up to the oldest observed age (no tail).
///
/// Every origin's and every development column's observed increments must
/// sum to more than zero; otherwise the log-linear mean does not exist
/// ([`Error::Bayes`]). Negative cells are allowed.
///
/// ```
/// use prospicio_reserving::bayes::BayesOdp;
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// use prospicio_bayes::nuts::Sampler;
/// use prospicio_prob::Distribution;
///
/// let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
///     values: &[(
///         "paid",
///         &[100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
///     )],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let model = BayesOdp {
///     sampler: Sampler { tune: 300, draws: 300, ..Sampler::default() },
///     n_sims: 2_000,
///     seed: 1,
///     ..BayesOdp::default()
/// };
/// let fit = model.fit(&tri, "paid")?;
/// assert_eq!(fit.reserves.n_components(), 4);
/// assert!(fit.reserves.mean() > 0.0);
/// assert_eq!(fit.posterior.names()[0], "c");
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BayesOdp {
    /// Standard deviation of the normal prior on the intercept `c`,
    /// centred on `c0`.
    pub intercept_sd: f64,
    /// Standard deviation of the normal prior on each origin effect
    /// `a_w`, `w >= 2`, centred on 0.
    pub origin_sd: f64,
    /// Standard deviation of the normal prior on each development effect
    /// `b_d`, `d >= 2`, centred on 0.
    pub development_sd: f64,
    /// Process error on each simulated future increment, as in the ODP
    /// bootstrap: Gamma with mean `mu` and variance `phi mu`, or none
    /// (parameter error only).
    pub process: ProcessDistribution,
    /// NUTS settings. Its `seed` is not used: the chains take a seed
    /// derived from [`seed`](Self::seed) (module docs, "Seeds").
    pub sampler: Sampler,
    /// Number of predictive simulations, independent of the number of
    /// posterior draws.
    pub n_sims: usize,
    /// Seed of the chains and of the simulations, each through its own
    /// derived seed (module docs, "Seeds").
    pub seed: u64,
    /// How segments depend on each other in a fit of several segments
    /// (decision 11, to come); [`fit`](Self::fit) has one segment and does
    /// not use it.
    pub dependence: SegmentDependence,
}

impl Default for BayesOdp {
    /// Prior standard deviations 10 on the log scale, Gamma process error,
    /// four chains of 1,000 warm-up and 1,000 kept draws with target
    /// acceptance 0.95 (above the sampler's 0.8: at 0.8 a triangle whose
    /// last development column has one small cell, such as RAA, usually
    /// gives a few divergent transitions), 10,000 simulations, seed 0.
    fn default() -> Self {
        Self {
            intercept_sd: 10.0,
            origin_sd: 10.0,
            development_sd: 10.0,
            process: ProcessDistribution::Gamma,
            sampler: Sampler {
                target_accept: 0.95,
                ..Sampler::default()
            },
            n_sims: 10_000,
            seed: 0,
            dependence: SegmentDependence::Independent,
        }
    }
}

/// A fitted [`BayesOdp`].
#[derive(Debug, Clone)]
pub struct BayesOdpFit {
    /// The volume-weighted chain ladder of the triangle: latest values,
    /// factors and the plug-in `phi` come from it, and the long tables'
    /// `ultimate` and `reserve` are its.
    pub chain_ladder: ChainLadderFit,
    /// The plugged-in scale `phi`, the chain ladder's Pearson scale
    /// (degrees of freedom `N - p`), the same as the ODP bootstrap's.
    pub scale: f64,
    /// Posterior draws of `c`, `a[2]..a[n]` and `b[2]..b[m]`, by chain.
    pub posterior: PosteriorDraws,
    /// R-hat, ESS and divergences, with warnings for failed thresholds.
    pub diagnostics: Diagnostics,
    /// Joint posterior predictive distribution of the reserve (sum of
    /// future incremental values) by origin: dimension `origin`, one
    /// component per origin, as [`OdpBootstrapFit::reserves`](crate::OdpBootstrapFit::reserves).
    pub reserves: PredictiveDistribution,
    /// The model, for the pointwise log likelihood of [`loo`](Self::loo).
    model: Model,
}

impl BayesOdp {
    /// Fits `column` of a single-segment cumulative triangle. Every origin
    /// must be observed from the first age to its latest, there must be
    /// more observed cells than parameters, and every origin's and every
    /// development column's increments must sum to more than zero.
    /// Failed diagnostics do not fail the fit: they are in
    /// [`BayesOdpFit::diagnostics`].
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<BayesOdpFit> {
        self.validate()?;
        let segment = triangle.segment(column)?;
        let Prepared {
            chain_ladder,
            model,
            start,
        } = self.prepare(&segment, true)?;
        let p = model.dim();
        let mode = mode(&model, start)?;
        let whitened = Whitened::new(&model, &mode)?;
        let draws = sample(
            &whitened,
            &vec![0.0; p],
            Sampler {
                seed: derive_seed(self.seed, Stream::Chains),
                ..self.sampler
            },
        )?;
        let posterior = draws.transform(model.names(), |z| whitened.theta(z))?;
        let summary = posterior.summary()?;
        let diagnostics = Diagnostics::new(
            &summary,
            posterior.chains(),
            posterior.divergences().iter().sum(),
        );

        let mut hasher = InputHasher::new();
        hasher.str(column);
        hash_segment(&mut hasher, &segment, &chain_ladder, &segment.ages);
        let latest = &chain_ladder.latest_position;
        let nd = segment.n_dev;
        let (process, phi) = (self.process, model.phi);
        let reserves = posterior.predictive(
            vec!["origin".into()],
            segment.origins.iter().map(|&o| vec![o.into()]).collect(),
            self.n_sims,
            derive_seed(self.seed, Stream::Predictive),
            self.provenance(column, &model, hasher),
            |theta, rng: &mut StreamRng, row| {
                for (w, reserve) in row.iter_mut().enumerate() {
                    *reserve = (latest[w] + 1..nd)
                        .map(|d| process_draw(process, phi, model.eta(theta, w, d).exp(), rng))
                        .sum();
                }
            },
        )?;
        Ok(BayesOdpFit {
            scale: model.phi,
            chain_ladder,
            posterior,
            diagnostics,
            reserves,
            model,
        })
    }

    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("intercept_sd", self.intercept_sd),
            ("origin_sd", self.origin_sd),
            ("development_sd", self.development_sd),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::InvalidSetting {
                    name,
                    value,
                    expected: "a finite, positive prior standard deviation",
                });
            }
        }
        if self.n_sims == 0 {
            return Err(Error::InvalidSetting {
                name: "n_sims",
                value: 0.0,
                expected: "a positive number of simulations",
            });
        }
        Ok(())
    }

    /// The chain ladder, the model and the Newton start of one segment;
    /// without `priors`, the model is the likelihood alone (for the
    /// sanity tests).
    pub(crate) fn prepare(&self, segment: &Segment, priors: bool) -> Result<Prepared> {
        let (fit, _) = prepare(segment, &segment.ages, Tail::default())?;
        let (no, nd) = (segment.n_origins, segment.n_dev);
        let latest = &fit.chain_ladder.latest_position;
        let mut cells = Vec::new();
        for (w, &last) in latest.iter().enumerate() {
            let mut previous = 0.0;
            for d in 0..=last {
                let cum = segment.get(w, d).expect("checked when prepared");
                cells.push((w, d, cum - previous));
                previous = cum;
            }
        }
        for w in 0..no {
            let sum: f64 = cells.iter().filter(|c| c.0 == w).map(|c| c.2).sum();
            if sum.is_nan() || sum <= 0.0 {
                return Err(Error::Bayes(format!(
                    "origin {}: the observed increments sum to {sum}, so no log-linear mean \
                     exists",
                    segment.origins[w]
                )));
            }
        }
        for d in 0..nd {
            let sum: f64 = cells.iter().filter(|c| c.1 == d).map(|c| c.2).sum();
            if sum.is_nan() || sum <= 0.0 {
                return Err(Error::Bayes(format!(
                    "development age {} months: the observed increments sum to {sum}, so no \
                     log-linear mean exists",
                    segment.ages[d]
                )));
            }
        }
        let total: f64 = cells.iter().map(|c| c.2).sum();
        let c0 = (total / cells.len() as f64).ln();
        let p = no + nd - 1;
        let mut prior_mean = vec![0.0; p];
        prior_mean[0] = c0;
        let mut prior_precision = vec![0.0; p];
        if priors {
            prior_precision[0] = self.intercept_sd.powi(-2);
            prior_precision[1..no].fill(self.origin_sd.powi(-2));
            prior_precision[no..].fill(self.development_sd.powi(-2));
        }
        // The chain ladder's fitted increments are U_w g_d, the ODP GLM's
        // MLE: their logs give the log-linear parameters when all are
        // positive. Otherwise start from the prior's centre.
        let fitted = &fit.fitted;
        let start = if cells.iter().all(|&(w, d, _)| fitted[w * nd + d] > 0.0) {
            let ln = |w: usize, d: usize| fitted[w * nd + d].ln();
            let mut start = vec![ln(0, 0); p];
            for (w, s) in start.iter_mut().enumerate().take(no).skip(1) {
                *s = ln(w, 0) - ln(0, 0);
            }
            for d in 1..nd {
                let w = (0..no).find(|&w| latest[w] >= d).expect("checked above");
                start[no - 1 + d] = ln(w, d) - ln(w, 0);
            }
            start
        } else {
            prior_mean.clone()
        };
        Ok(Prepared {
            chain_ladder: fit.chain_ladder,
            model: Model {
                n_origins: no,
                n_dev: nd,
                cells,
                phi: fit.scale,
                c0,
                prior_mean,
                prior_precision,
            },
            start,
        })
    }

    fn provenance(&self, column: &str, model: &Model, hasher: InputHasher) -> Provenance {
        let s = &self.sampler;
        Provenance::new("bayes_odp")
            .param("intercept_sd", self.intercept_sd)
            .param("origin_sd", self.origin_sd)
            .param("development_sd", self.development_sd)
            .param("c0", model.c0)
            .param("scale", model.phi)
            .param("process", format!("{:?}", self.process))
            .param("chains", s.chains)
            .param("tune", s.tune)
            .param("draws", s.draws)
            .param("target_accept", s.target_accept)
            .param("max_depth", s.max_depth)
            .param("n_sims", self.n_sims)
            .param("seed", self.seed)
            .param("column", column)
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(hasher.finish())
    }
}

/// What [`BayesOdp::prepare`] gives.
pub(crate) struct Prepared {
    pub(crate) chain_ladder: ChainLadderFit,
    pub(crate) model: Model,
    pub(crate) start: Vec<f64>,
}

impl BayesOdpFit {
    /// Posterior summary of each parameter: mean, sd, 5/50/95%
    /// quantiles, R-hat, bulk and tail ESS.
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        Ok(self.posterior.summary()?)
    }

    /// PSIS-LOO of the observed increments, with each cell's relative
    /// efficiency from the chains. The pointwise term is the normalised
    /// density of the over-dispersed Poisson
    /// ([`Family::log_density`] at `mu_wd` and `phi`), not the quasi
    /// likelihood the model is fitted with, so the result is a log
    /// predictive density; compare it only with models of the same
    /// incremental response. A triangle with a negative increment has no
    /// such density and is an error.
    pub fn loo(&self) -> Result<Loo> {
        let model = &self.model;
        if let Some(&(w, d, x)) = model.cells.iter().find(|c| c.2 < 0.0) {
            return Err(Error::Bayes(format!(
                "loo needs non-negative increments, and origin {} at development age {} \
                 months is {x}",
                self.chain_ladder.origins[w], self.chain_ladder.development.development[d]
            )));
        }
        let n = model.cells.len();
        let s = self.posterior.n_draws();
        // Each term integrates the continuous Poisson's normalising
        // constant, so the draws run in parallel; the result does not
        // depend on the threads.
        let rows: Vec<Vec<f64>> = self
            .posterior
            .values()
            .par_chunks_exact(self.posterior.dim())
            .map(|theta| {
                model
                    .cells
                    .iter()
                    .map(|&(w, d, x)| {
                        let mu = model.eta(theta, w, d).exp();
                        Family::Poisson.log_density(x, mu, model.phi, 1.0)
                    })
                    .collect::<prospicio_core::Result<Vec<f64>>>()
            })
            .collect::<prospicio_core::Result<_>>()?;
        let ll = rows.concat();
        let chains = self.posterior.chains();
        let per_chain = s / chains;
        let r_eff: Vec<f64> = (0..n)
            .map(|i| {
                let by_chain: Vec<Vec<f64>> = (0..chains)
                    .map(|c| {
                        (0..per_chain)
                            .map(|k| ll[(c * per_chain + k) * n + i].exp())
                            .collect()
                    })
                    .collect();
                let refs: Vec<&[f64]> = by_chain.iter().map(Vec::as_slice).collect();
                let ess = prospicio_bayes::ess_mean(&refs).unwrap_or(s as f64);
                // A constant likelihood: as independent draws.
                if ess.is_finite() && ess > 0.0 {
                    ess / s as f64
                } else {
                    1.0
                }
            })
            .collect();
        Ok(prospicio_bayes::elpd::loo(&ll, n, Some(&r_eff))?)
    }

    /// One row per origin: the chain ladder's `latest`, `ultimate` and
    /// `reserve`, and the `mean` and `std_dev` of the posterior predictive
    /// reserve.
    pub fn to_long(&self) -> FitTable {
        let mut table = self.plug_in().to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        table
    }

    /// One row: the chain ladder's totals, the `scale`, and the `mean` and
    /// `std_dev` of the posterior predictive total reserve.
    pub fn totals(&self) -> FitTable {
        let fits = self.plug_in();
        let mut table = fits.totals();
        push_moments(&mut table, "", &segment_sums(&fits, &self.reserves));
        table
    }

    /// The chain ladder's development factors, one row per age.
    pub fn development_table(&self) -> FitTable {
        self.plug_in().development_table()
    }

    /// The fit as the single segment of the long tables.
    fn plug_in(&self) -> SegmentFits<PlugIn> {
        SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![PlugIn {
                chain_ladder: self.chain_ladder.clone(),
                scale: self.scale,
            }],
        }
    }
}

impl ReserveFit for BayesOdpFit {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("scale", self.scale)]
    }
}

/// The chain ladder and scale of a fit, for its long tables.
#[derive(Debug, Clone)]
struct PlugIn {
    chain_ladder: ChainLadderFit,
    scale: f64,
}

impl ReserveFit for PlugIn {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("scale", self.scale)]
    }
}

/// The log posterior of decision 3 on `theta = (c, a_2..a_n, b_2..b_m)`.
#[derive(Debug, Clone)]
pub(crate) struct Model {
    n_origins: usize,
    n_dev: usize,
    /// Observed increments: (origin, development position, value).
    cells: Vec<(usize, usize, f64)>,
    phi: f64,
    /// Log of the mean observed increment, the prior centre of `c`.
    c0: f64,
    prior_mean: Vec<f64>,
    /// `1 / sd^2` per parameter; zero without priors.
    prior_precision: Vec<f64>,
}

impl Model {
    fn names(&self) -> Vec<String> {
        let mut names = vec!["c".to_string()];
        names.extend((2..=self.n_origins).map(|w| format!("a[{w}]")));
        names.extend((2..=self.n_dev).map(|d| format!("b[{d}]")));
        names
    }

    /// The positions in `theta` of the parameters of cell `(w, d)`.
    fn params(&self, w: usize, d: usize) -> [Option<usize>; 3] {
        [
            Some(0),
            (w > 0).then_some(w),
            (d > 0).then(|| self.n_origins - 1 + d),
        ]
    }

    pub(crate) fn eta(&self, theta: &[f64], w: usize, d: usize) -> f64 {
        self.params(w, d).iter().flatten().map(|&j| theta[j]).sum()
    }
}

/// The log posterior of decision 3. It is strictly concave, so [`mode`]
/// finds its one mode with the analytic negative Hessian and never needs a
/// ridge.
impl Posterior for Model {
    fn dim(&self) -> usize {
        self.n_origins + self.n_dev - 1
    }

    fn log_posterior(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        let mut f = 0.0;
        for j in 0..theta.len() {
            let r = theta[j] - self.prior_mean[j];
            f -= 0.5 * self.prior_precision[j] * r * r;
            grad[j] = -self.prior_precision[j] * r;
        }
        for &(w, d, x) in &self.cells {
            let eta = self.eta(theta, w, d);
            let mu = eta.exp();
            f += (x * eta - mu) / self.phi;
            let g = (x - mu) / self.phi;
            for j in self.params(w, d).into_iter().flatten() {
                grad[j] += g;
            }
        }
        f
    }

    /// `X' diag(mu / phi) X + diag(1 / sd^2)`, dense and row-major.
    fn negative_hessian(&self, theta: &[f64]) -> Vec<f64> {
        let p = self.dim();
        let mut h = vec![0.0; p * p];
        for (j, &prec) in self.prior_precision.iter().enumerate() {
            h[j * p + j] = prec;
        }
        for &(w, d, _) in &self.cells {
            let weight = self.eta(theta, w, d).exp() / self.phi;
            let params = self.params(w, d);
            for i in params.iter().flatten() {
                for j in params.iter().flatten() {
                    h[i * p + j] += weight;
                }
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bayes::laplace::tests::assert_gradient;
    use crate::triangle::tests::{GENINS, annual, raa};
    use crate::{OdpBootstrap, triangle::Long};
    use prospicio_bayes::nuts::LogDensity;
    use prospicio_core::Period;
    use prospicio_prob::Distribution;

    fn genins() -> Triangle {
        annual(2001, &GENINS)
    }

    fn quick(seed: u64) -> BayesOdp {
        BayesOdp {
            sampler: Sampler {
                tune: 300,
                draws: 300,
                ..Sampler::default()
            },
            n_sims: 2_000,
            seed,
            ..BayesOdp::default()
        }
    }

    fn prepared(triangle: &Triangle, model: &BayesOdp, priors: bool) -> Prepared {
        model
            .prepare(&triangle.segment("values").unwrap(), priors)
            .unwrap()
    }

    #[test]
    fn the_gradient_matches_a_finite_difference() {
        let Prepared { model, start, .. } = prepared(&raa(), &BayesOdp::default(), true);
        // Off the mode, so the gradient is not zero.
        let x: Vec<f64> = start
            .iter()
            .enumerate()
            .map(|(j, v)| v + 0.01 * (j as f64 - 5.0))
            .collect();
        assert_gradient(|t, g| model.log_posterior(t, g), &x);
        let mode = mode(&model, start).unwrap();
        let whitened = Whitened::new(&model, &mode).unwrap();
        let z: Vec<f64> = (0..model.dim())
            .map(|j| 0.3 * ((j % 3) as f64 - 1.0))
            .collect();
        assert_gradient(|z, g| whitened.log_density(z, g).unwrap(), &z);
    }

    #[test]
    fn without_priors_the_mode_is_the_chain_ladder() {
        for triangle in [raa(), genins()] {
            let segment = triangle.segment("values").unwrap();
            let Prepared { model, start, .. } =
                BayesOdp::default().prepare(&segment, false).unwrap();
            // From the prior's centre, not the chain ladder's start.
            let mut from = vec![0.0; model.dim()];
            from[0] = model.c0;
            let mode = mode(&model, from).unwrap();
            let (fit, _) = prepare(&segment, &segment.ages, Tail::default()).unwrap();
            for &(w, d, _) in &model.cells {
                let mu = model.eta(&mode, w, d).exp();
                let cl = fit.fitted[w * segment.n_dev + d];
                assert!((mu / cl - 1.0).abs() < 1e-8, "({w}, {d}): {mu} vs {cl}");
            }
            for (a, b) in mode.iter().zip(&start) {
                assert!((a - b).abs() < 1e-8, "{a} vs the chain ladder's {b}");
            }
        }
    }

    #[test]
    fn the_priors_shift_the_mode_by_one_over_sd_squared() {
        let triangle = genins();
        let shift = |sd: f64| {
            let model = BayesOdp {
                intercept_sd: sd,
                origin_sd: sd,
                development_sd: sd,
                ..BayesOdp::default()
            };
            let none = prepared(&triangle, &model, false);
            let with = prepared(&triangle, &model, true);
            let a = mode(&none.model, none.start).unwrap();
            let b = mode(&with.model, with.start).unwrap();
            a.iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max)
        };
        let (s100, s1000) = (shift(100.0), shift(1000.0));
        assert!(s1000 > 0.0);
        let ratio = s100 / s1000;
        assert!((ratio / 100.0 - 1.0).abs() < 0.01, "ratio {ratio}");
    }

    #[test]
    fn a_seed_replays_whatever_the_threads() {
        let a = quick(3).fit(&raa(), "values").unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let b = pool.install(|| quick(3).fit(&raa(), "values").unwrap());
        assert_eq!(a.posterior, b.posterior);
        assert_eq!(a.reserves.draw_matrix(), b.reserves.draw_matrix());
        let c = quick(4).fit(&raa(), "values").unwrap();
        assert_ne!(a.posterior.values(), c.posterior.values());
        assert_ne!(a.reserves.draw_matrix(), c.reserves.draw_matrix());
        // The sampler's own seed is not used.
        let mut d = quick(3);
        d.sampler.seed = 99;
        assert_eq!(d.fit(&raa(), "values").unwrap().posterior, a.posterior);
    }

    #[test]
    fn a_short_run_returns_the_fit_with_warnings() {
        let model = BayesOdp {
            sampler: Sampler {
                tune: 50,
                draws: 20,
                ..Sampler::default()
            },
            n_sims: 100,
            ..BayesOdp::default()
        };
        let fit = model.fit(&raa(), "values").unwrap();
        // 4 x 20 draws cannot reach a bulk ESS of 400.
        assert!(fit.diagnostics.min_ess_bulk < 400.0);
        assert!(
            fit.diagnostics
                .warnings
                .iter()
                .any(|w| w.contains("bulk ESS")),
            "{:?}",
            fit.diagnostics.warnings
        );
    }

    #[test]
    fn errors() {
        let tri = raa();
        let bad_sd = BayesOdp {
            origin_sd: 0.0,
            ..BayesOdp::default()
        };
        assert!(matches!(
            bad_sd.fit(&tri, "values"),
            Err(Error::InvalidSetting {
                name: "origin_sd",
                ..
            })
        ));
        let nan_sd = BayesOdp {
            intercept_sd: f64::NAN,
            ..BayesOdp::default()
        };
        assert!(matches!(
            nan_sd.fit(&tri, "values"),
            Err(Error::InvalidSetting {
                name: "intercept_sd",
                ..
            })
        ));
        let no_sims = BayesOdp {
            n_sims: 0,
            ..BayesOdp::default()
        };
        assert!(matches!(
            no_sims.fit(&tri, "values"),
            Err(Error::InvalidSetting { name: "n_sims", .. })
        ));
        assert!(matches!(
            BayesOdp::default().fit(&tri, "paid"),
            Err(Error::UnknownColumn(_))
        ));

        // The last development column falls: its only increment is -5.
        let falling = annual(
            2020,
            &[
                &[100.0, 150.0, 160.0, 155.0],
                &[110.0, 170.0, 180.0],
                &[120.0, 175.0],
                &[130.0],
            ],
        );
        let e = quick(0).fit(&falling, "values").unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("development age 48 months")),
            "{e}"
        );
        // The newest origin's only cell is negative.
        let negative = annual(
            2020,
            &[
                &[100.0, 150.0, 160.0, 165.0],
                &[110.0, 170.0, 180.0],
                &[120.0, 175.0],
                &[-130.0],
            ],
        );
        let e = quick(0).fit(&negative, "values").unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("origin 2023")),
            "{e}"
        );
        // Too few cells, from the bootstrap's preparation.
        let tiny = annual(2020, &[&[1.0, 2.0], &[1.0]]);
        assert!(matches!(
            quick(0).fit(&tiny, "values"),
            Err(Error::Bootstrap(_))
        ));

        let origin = [2020, 2020, 2021].map(crate::Month::january);
        let two = Triangle::from_long(&Long {
            keys: &[("lob", &["a", "a", "a", "b", "b", "b"])],
            origin: &[origin, origin].concat(),
            development: crate::DevelopmentColumn::Age(&[12, 24, 12, 12, 24, 12]),
            values: &[("paid", &[1.0, 2.0, 1.0, 1.0, 2.0, 1.0])],
            origin_grain: crate::Grain::Year,
            development_grain: crate::Grain::Year,
            cumulative: true,
        })
        .unwrap();
        assert!(matches!(
            quick(0).fit(&two, "paid"),
            Err(Error::MultipleSegments(2))
        ));
    }

    #[test]
    fn reserves_by_origin() {
        let fit = quick(5).fit(&raa(), "values").unwrap();
        assert_eq!(fit.reserves.dims(), ["origin"]);
        assert_eq!(fit.reserves.n_components(), 10);
        assert_eq!(fit.reserves.n_sims(), 2_000);
        let oldest = fit
            .reserves
            .marginal(&vec![Period::year(1981).into()])
            .unwrap();
        assert_eq!(oldest.mean(), 0.0);
        assert_eq!(oldest.std_dev(), 0.0);
        assert!(fit.reserves.total().mean() > 0.0);
        assert_eq!(fit.posterior.names().len(), 19);
        assert_eq!(fit.posterior.names()[9], "a[10]");
        assert_eq!(fit.posterior.names()[10], "b[2]");
        let long = fit.to_long();
        assert_eq!(long.n_rows(), 10);
        assert!(long.column("mean").is_some() && long.column("std_dev").is_some());
        let totals = fit.totals();
        assert_eq!(totals.column("scale").unwrap(), [fit.scale]);
        assert!((totals.column("mean").unwrap()[0] - fit.reserves.total().mean()).abs() < 1e-6);
        // Nine age-to-age factors and the tail row.
        assert_eq!(fit.development_table().n_rows(), 10);
        assert_eq!(fit.summary().unwrap().len(), 19);
    }

    #[test]
    fn provenance_records_the_settings() {
        let fit = quick(5).fit(&raa(), "values").unwrap();
        let provenance = fit.reserves.provenance();
        assert_eq!(provenance.model, "bayes_odp");
        let keys: Vec<&str> = provenance
            .parameters
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        for key in [
            "intercept_sd",
            "origin_sd",
            "development_sd",
            "c0",
            "scale",
            "process",
            "chains",
            "tune",
            "draws",
            "target_accept",
            "max_depth",
            "n_sims",
            "seed",
            "column",
            "posterior_draws",
            "posterior_seed",
        ] {
            assert!(keys.contains(&key), "{key} missing from {keys:?}");
        }
    }

    #[test]
    fn process_error_widens_the_distribution() {
        let full = quick(6).fit(&raa(), "values").unwrap();
        let param = BayesOdp {
            process: ProcessDistribution::None,
            ..quick(6)
        }
        .fit(&raa(), "values")
        .unwrap();
        assert!(param.reserves.total().std_dev() < full.reserves.total().std_dev());
    }

    #[test]
    fn the_scale_is_the_bootstraps() {
        let fit = quick(7).fit(&raa(), "values").unwrap();
        let boot = OdpBootstrap {
            n_sims: 10,
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        assert_eq!(fit.scale, boot.scale);
    }

    #[test]
    fn raa_fits_but_has_no_loo() {
        // RAA has a negative increment (1982 at 84 months).
        let fit = quick(8).fit(&raa(), "values").unwrap();
        let e = fit.loo().unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.contains("non-negative")
                && m.contains("origin 1982 at development age 84 months is")),
            "{e}"
        );
        // A short run keeps the quadrature cheap; validation/tests/reserving_bayes.rs
        // checks the defaults.
        let short = BayesOdp {
            sampler: Sampler {
                tune: 100,
                draws: 25,
                ..Sampler::default()
            },
            n_sims: 100,
            ..BayesOdp::default()
        };
        let loo = short.fit(&genins(), "values").unwrap().loo().unwrap();
        assert_eq!(loo.pareto_k.len(), 55);
        assert!(loo.estimate.elpd.is_finite());
    }
}
