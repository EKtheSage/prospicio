//! The Bayesian Bornhuetter–Ferguson with a normalised development pattern:
//! the over-dispersed Poisson model whose means are an origin's ultimate
//! times the share of it paid at each age, with the informative gamma prior
//! on the ultimate itself (`docs/design/reserving-bayes.md`, decision 23).

use prospicio_bayes::elpd::Loo;
use prospicio_bayes::nuts::{ParamSummary, PosteriorDraws, Sampler, sample};
use prospicio_core::StreamRng;
use prospicio_models::Family;
use prospicio_prob::{InputHasher, PredictiveDistribution, Provenance};
use rayon::prelude::*;

use super::bf::{PlugIn, input_hasher};
use super::laplace::{Posterior, Whitened, mode};
use super::one_year::{self, BayesOneYear, BayesOneYearFit, OdpCells, OdpYear};
use super::{Diagnostics, Stream, derive_seed, one_year_seed};
use crate::bootstrap::{
    ProcessDistribution, component_sums, prepare, process_draw, push_moments, segment_sums,
};
use crate::chain_ladder::ChainLadderFit;
use crate::error::{Error, Result};
use crate::expected_loss::origin_exposure;
use crate::segments::{FitTable, ReserveFit, SegmentFits};
use crate::tail::Tail;
use crate::triangle::{Label, Segment, Triangle};

/// The Bayesian Bornhuetter–Ferguson whose prior sits on the ultimate: the
/// over-dispersed Poisson model of the incremental values `X_wd` with means
/// `U_w p_d`, the origin's expected ultimate `U_w` times the share `p_d`
/// of it that falls at age `d`, the shares summing to 1,
///
/// ```text
/// log p(X | theta) = sum over observed cells of (X_wd eta_wd - exp(eta_wd)) / phi,
/// eta_wd = alpha_w + beta_d - log(sum_k exp(beta_k)),   beta_1 = 0
/// U_w = exp(alpha_w) ~ Gamma(shape 1 / cv^2, mean m_w),   m_w = apriori * exposure_w
/// beta_d ~ N(0, development_sd^2)  for d = 2, ..., m
/// ```
///
/// with `phi` the chain ladder's Pearson scale, plugged in as in
/// [`BayesOdp`](super::BayesOdp). The gamma prior is on the origin's
/// expected ultimate, so `apriori` is the expected ultimate loss ratio and
/// `cv` how firmly it is held, as in the classical
/// [`BornhuetterFerguson`](crate::BornhuetterFerguson); the shares
/// `p_d = exp(beta_d) / sum_k exp(beta_k)` have a vague normal prior on
/// their log ratios to the first age's (`beta_1 = 0`, the corner of
/// [`BayesOdp`](super::BayesOdp)). [`BayesBf`](super::BayesBf), England,
/// Verrall and Wüthrich's (2012) Model 3.1, instead leaves the pattern
/// unnormalised and puts the prior on `mu_w`, which is not the ultimate:
/// there the level of `apriori` is cancelled by the pattern and barely
/// moves the reserve (`knowledge/findings/bayes-bf-apriori-scale.md`). This
/// model is Verrall's (2004) Bayesian Bornhuetter–Ferguson in the sense
/// that the ultimates and the pattern are separate parameters, the prior on
/// the former and the pattern a proportion; it estimates the pattern jointly
/// with the ultimates, not from the chain ladder first.
///
/// The prior's strength moves the reserve between two limits:
///
/// * `cv = f64::INFINITY` (a flat prior on `log U_w`): the posterior is the
///   Bayesian chain ladder's, and the reserves agree with
///   [`BayesOdp`](super::BayesOdp)'s within Monte Carlo error (the two
///   vague priors differ slightly);
/// * `cv -> 0`: `U_w` is fixed at `m_w`, and the shares maximise the
///   likelihood given the ultimates: `p_d = S_d / (M_d + lambda)` with
///   `S_d` the observed increments of column `d`, `M_d` the prior ultimates
///   of the origins observed there and `lambda` such that the shares sum
///   to 1. When the prior ultimates are the chain ladder's, `lambda` is 0,
///   `p_d` is the chain ladder's pattern and the reserve is the classical
///   Bornhuetter–Ferguson reserve `(1 - 1 / cdf_w) m_w` exactly; otherwise
///   the shares move with the ultimates (the classical method takes the
///   chain ladder's pattern whatever the prior), by an amount that the
///   design note's decision 23 measures.
///
/// Every development column's observed increments must sum to more than
/// zero, and every origin's to more than `-phi / cv^2`, the bound below
/// which its posterior is improper ([`Error::Bayes`]). Negative cells are
/// allowed. The log posterior is not concave in general (the softmax), but
/// the mode finder's ridge handles that; NUTS samples it on a scale
/// whitened at the mode, as for `BayesOdp`.
///
/// ```
/// use prospicio_reserving::bayes::BayesBfNormalised;
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// use prospicio_bayes::nuts::Sampler;
/// use prospicio_prob::Distribution;
///
/// let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
/// let premium = [200.0, 200.0, 200.0, 200.0, 220.0, 220.0, 220.0, 240.0, 240.0, 260.0];
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
///     values: &[
///         ("paid", &[100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]),
///         ("premium", &premium),
///     ],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let model = BayesBfNormalised {
///     apriori: 0.85,
///     cv: 0.1,
///     sampler: Sampler { tune: 300, draws: 300, ..Sampler::default() },
///     n_sims: 2_000,
///     seed: 1,
///     ..BayesBfNormalised::default()
/// };
/// let fit = model.fit(&tri, "paid", "premium")?;
/// assert_eq!(fit.reserves.n_components(), 4);
/// assert!(fit.reserves.mean() > 0.0);
/// assert_eq!(fit.prior_ultimate[3], 0.85 * 260.0);
/// assert_eq!(fit.posterior.names()[0], "alpha[1]");
/// assert!((fit.pattern().iter().sum::<f64>() - 1.0).abs() < 1e-9);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BayesBfNormalised {
    /// Prior mean of each origin's expected ultimate per unit of exposure:
    /// the expected ultimate loss ratio.
    pub apriori: f64,
    /// Coefficient of variation of the gamma prior on each origin's
    /// expected ultimate, so its shape is `1 / cv^2`. `f64::INFINITY` is a
    /// flat prior on `log U_w`, the Bayesian chain ladder.
    pub cv: f64,
    /// Standard deviation of the normal prior on each `beta_d` (the log
    /// ratio of the share at age `d` to the first age's), centred on 0.
    pub development_sd: f64,
    /// Process error on each simulated future increment, as in the ODP
    /// bootstrap and [`BayesOdp`](super::BayesOdp).
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
}

impl Default for BayesBfNormalised {
    /// `apriori` 1 (as [`BornhuetterFerguson`](crate::BornhuetterFerguson)),
    /// `cv` 0.1 (EVW's example, shape 100), `development_sd` 10, Gamma
    /// process error, four chains of 1,000 warm-up and 1,000 kept draws
    /// with target acceptance 0.95 (as `BayesOdp`), 10,000 simulations,
    /// seed 0.
    fn default() -> Self {
        Self {
            apriori: 1.0,
            cv: 0.1,
            development_sd: 10.0,
            process: ProcessDistribution::Gamma,
            sampler: Sampler {
                target_accept: 0.95,
                ..Sampler::default()
            },
            n_sims: 10_000,
            seed: 0,
        }
    }
}

/// A fitted [`BayesBfNormalised`].
#[derive(Debug, Clone)]
pub struct BayesBfNormalisedFit {
    /// The volume-weighted chain ladder of the triangle: latest values and
    /// the plug-in `phi` come from it, and the long tables' `ultimate` and
    /// `reserve` are its.
    pub chain_ladder: ChainLadderFit,
    /// The plugged-in scale `phi`, the chain ladder's Pearson scale, as
    /// [`BayesOdpFit::scale`](super::BayesOdpFit::scale).
    pub scale: f64,
    /// Exposure per origin: the exposure column's latest observed value.
    pub exposure: Vec<f64>,
    /// The prior mean `m_w` of each origin's expected ultimate,
    /// `apriori * exposure`.
    pub prior_ultimate: Vec<f64>,
    /// Posterior draws of `alpha[1]..alpha[n]` (the log expected
    /// ultimates) and `beta[2]..beta[m]` (the log ratios of the shares to
    /// the first age's), by chain.
    pub posterior: PosteriorDraws,
    /// R-hat, ESS and divergences, with warnings for failed thresholds.
    pub diagnostics: Diagnostics,
    /// Joint posterior predictive distribution of the reserve (sum of
    /// future incremental values) by origin: dimension `origin`, one
    /// component per origin, as [`BayesOdpFit::reserves`](super::BayesOdpFit::reserves).
    pub reserves: PredictiveDistribution,
    /// The model, for the pointwise log likelihood of [`loo`](Self::loo).
    model: Model,
}

impl BayesBfNormalised {
    /// Fits loss `column` of a single-segment cumulative triangle with
    /// exposure from its `exposure` column. Every origin must be observed
    /// from the first age to its latest and there must be more observed
    /// cells than chain-ladder parameters, as for `BayesOdp`. Failed
    /// diagnostics do not fail the fit: they are in
    /// [`BayesBfNormalisedFit::diagnostics`].
    pub fn fit(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
    ) -> Result<BayesBfNormalisedFit> {
        self.validate()?;
        let segment = triangle.segment(column)?;
        let exposures = origin_exposure(&segment, &triangle.segment(exposure)?, exposure)?;
        self.fit_segment(&segment, column, exposure, exposures)
    }

    /// The one-year view of loss `column` of a single-segment cumulative
    /// triangle with exposure from its `exposure` column (decisions 16, 17
    /// and 23): the claims development result over the twelve months after
    /// the valuation, by reweighting [`fit`](Self::fit)'s posterior, draw
    /// for draw, with Pareto smoothed importance sampling, as
    /// [`BayesBf::one_year`](super::BayesBf::one_year) does with this
    /// model's means `exp(alpha_w + beta_d - log sum exp(beta))`. A
    /// simulation whose Pareto `k̂` is above [`BayesOneYear::k_threshold`]
    /// is handled as [`BayesOneYear::refit`] says: by default
    /// moment-matched (decision 18) and, only if that cannot bring its `k̂`
    /// to the threshold, refitted with NUTS on the triangle extended by its
    /// year. There is no view over segments: the model fits one segment.
    ///
    /// ```
    /// use prospicio_bayes::nuts::Sampler;
    /// use prospicio_reserving::bayes::{BayesBfNormalised, BayesOneYear};
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
    /// use prospicio_prob::Distribution;
    ///
    /// let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
    /// let premium = [200.0, 200.0, 200.0, 200.0, 220.0, 220.0, 220.0, 240.0, 240.0, 260.0];
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
    ///     values: &[
    ///         ("paid", &[100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]),
    ///         ("premium", &premium),
    ///     ],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })?;
    /// let model = BayesBfNormalised {
    ///     apriori: 0.85,
    ///     cv: 0.1,
    ///     sampler: Sampler { tune: 300, draws: 300, ..Sampler::default() },
    ///     n_sims: 500,
    ///     ..BayesBfNormalised::default()
    /// };
    /// let view = model.one_year(&tri, "paid", "premium", &BayesOneYear::default())?;
    /// assert_eq!(view.cdr.n_components(), 4);
    /// // 2020 is fully developed: no cell in the year, no change.
    /// assert!(view.cdr.draw_matrix().chunks(4).all(|row| row[0] == 0.0));
    /// assert!(view.cdr.std_dev() < view.fit.reserves.std_dev());
    /// # Ok::<(), prospicio_reserving::Error>(())
    /// ```
    pub fn one_year(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
        settings: &BayesOneYear,
    ) -> Result<BayesOneYearFit<BayesBfNormalisedFit>> {
        self.validate()?;
        settings.validate()?;
        let segment = triangle.segment(column)?;
        let exposures = origin_exposure(&segment, &triangle.segment(exposure)?, exposure)?;
        let fit = self.fit_segment(&segment, column, exposure, exposures)?;
        let cells = NormalisedRefit::new(self, &fit);
        let year = OdpYear::new(
            &cells,
            self.process,
            triangle,
            &segment,
            &fit.chain_ladder.latest_position,
        );
        let draws = one_year::simulate(
            &year,
            &fit.posterior,
            settings,
            settings.sims(self.n_sims),
            one_year_seed(derive_seed(self.seed, Stream::Predictive)),
        )?;
        let hasher = input_hasher(column, exposure, &segment, &fit.chain_ladder, &fit.exposure);
        let mut provenance = self.provenance(column, exposure, &fit.model, hasher);
        provenance.model = "bayes_bf_normalised_one_year".into();
        let provenance = one_year::provenance(provenance, settings, settings.sims(self.n_sims));
        one_year::assemble(fit, draws, &segment, settings, provenance)
    }

    /// The fit of one segment, with exposure `exposures` from the column
    /// named `exposure`.
    fn fit_segment(
        &self,
        segment: &Segment,
        column: &str,
        exposure: &str,
        exposures: Vec<f64>,
    ) -> Result<BayesBfNormalisedFit> {
        let (chain_ladder, model, start) = self.prepare(segment, &exposures)?;
        let (posterior, diagnostics) =
            self.sample(&model, start, derive_seed(self.seed, Stream::Chains))?;

        let hasher = input_hasher(column, exposure, segment, &chain_ladder, &exposures);
        let latest = &chain_ladder.latest_position;
        let (no, nd) = (segment.n_origins, segment.n_dev);
        let (process, phi) = (self.process, model.phi);
        let reserves = posterior.predictive(
            vec!["origin".into()],
            segment.origins.iter().map(|&o| vec![o.into()]).collect(),
            self.n_sims,
            derive_seed(self.seed, Stream::Predictive),
            self.provenance(column, exposure, &model, hasher),
            |theta, rng: &mut StreamRng, row| {
                let log_share = log_pattern(no, nd, theta);
                for (w, reserve) in row.iter_mut().enumerate() {
                    *reserve = (latest[w] + 1..nd)
                        .map(|d| process_draw(process, phi, (theta[w] + log_share[d]).exp(), rng))
                        .sum();
                }
            },
        )?;
        Ok(BayesBfNormalisedFit {
            scale: model.phi,
            prior_ultimate: model.prior_ultimate.clone(),
            exposure: exposures,
            chain_ladder,
            posterior,
            diagnostics,
            reserves,
            model,
        })
    }

    /// NUTS on `model` whitened at its mode (found from `start`, on the
    /// sampling scale `(u, beta)`), with the chains' `seed`; the draws on
    /// the natural scale `(alpha, beta)` and their diagnostics.
    fn sample(
        &self,
        model: &Model,
        start: Vec<f64>,
        seed: u64,
    ) -> Result<(PosteriorDraws, Diagnostics)> {
        let p = model.dim();
        let mode = mode(model, start)?;
        let whitened = Whitened::new(model, &mode)?;
        let draws = sample(
            &whitened,
            &vec![0.0; p],
            Sampler {
                seed,
                ..self.sampler
            },
        )?;
        let posterior = draws.transform(model.names(), |z| model.natural(&whitened.theta(z)))?;
        let summary = posterior.summary()?;
        let diagnostics = Diagnostics::new(
            &summary,
            posterior.chains(),
            posterior.divergences().iter().sum(),
        );
        Ok((posterior, diagnostics))
    }

    fn validate(&self) -> Result<()> {
        if !(self.apriori.is_finite() && self.apriori > 0.0) {
            return Err(Error::InvalidSetting {
                name: "apriori",
                value: self.apriori,
                expected: "a finite, positive loss ratio",
            });
        }
        // Infinity is the flat prior; NaN, zero and negatives are not.
        if self.cv.is_nan() || self.cv <= 0.0 {
            return Err(Error::InvalidSetting {
                name: "cv",
                value: self.cv,
                expected: "a positive coefficient of variation (infinity for a flat prior)",
            });
        }
        if !(self.development_sd.is_finite() && self.development_sd > 0.0) {
            return Err(Error::InvalidSetting {
                name: "development_sd",
                value: self.development_sd,
                expected: "a finite, positive prior standard deviation",
            });
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

    /// The chain ladder, the model and the Newton start of one segment.
    fn prepare(
        &self,
        segment: &Segment,
        exposure: &[f64],
    ) -> Result<(ChainLadderFit, Model, Vec<f64>)> {
        let (fit, _) = prepare(segment, &segment.ages, Tail::default())?;
        let (no, nd) = (segment.n_origins, segment.n_dev);
        let mut cells = Vec::new();
        for (w, &last) in fit.chain_ladder.latest_position.iter().enumerate() {
            let mut previous = 0.0;
            for d in 0..=last {
                let cum = segment.get(w, d).expect("checked when prepared");
                cells.push((w, d, cum - previous));
                previous = cum;
            }
        }
        let phi = fit.scale;
        let shape = self.cv.powi(-2);
        let prior_ultimate: Vec<f64> = exposure.iter().map(|e| self.apriori * e).collect();
        // The posterior of U_w given the pattern is a gamma of shape
        // shape + S_w / phi: proper only when that is positive.
        for w in 0..no {
            let sum: f64 = cells.iter().filter(|c| c.0 == w).map(|c| c.2).sum();
            if sum.is_nan() || shape + sum / phi <= 0.0 {
                let bound = if shape == 0.0 {
                    "so no log-linear mean exists".to_string()
                } else {
                    format!(
                        "at most -phi / cv^2 = {}, so the posterior of its ultimate is improper",
                        -shape * phi
                    )
                };
                return Err(Error::Bayes(format!(
                    "origin {}: the observed increments sum to {sum}, {bound}",
                    segment.origins[w]
                )));
            }
        }
        let mut column_sum = vec![0.0; nd];
        let mut column_prior = vec![0.0; nd];
        for &(_, d, x) in &cells {
            column_sum[d] += x;
        }
        for (w, &last) in fit.chain_ladder.latest_position.iter().enumerate() {
            for prior in &mut column_prior[..=last] {
                *prior += prior_ultimate[w];
            }
        }
        for (d, &sum) in column_sum.iter().enumerate() {
            if sum.is_nan() || sum <= 0.0 {
                return Err(Error::Bayes(format!(
                    "development age {} months: the observed increments sum to {sum}, so no \
                     log-linear mean exists",
                    segment.ages[d]
                )));
            }
        }
        // Start at the strong-prior mode: the ultimates at their prior
        // (u = 0) and the shares that maximise the likelihood given them,
        // S_d / (M_d + lambda). That is exactly the mode when the prior
        // ultimates are the chain ladder's (lambda = 0), and a concave
        // start under a tight prior otherwise: from the raw S_d / M_d the
        // negative Hessian is indefinite when the prior ultimates are far
        // above the data's, and the ridge, scaled by the prior's stiff
        // curvature, then crawls.
        let share = strong_prior_shares(&column_sum, &column_prior);
        let mut start = vec![0.0; no];
        start.extend((1..nd).map(|d| (share[d] / share[0]).ln()));
        let model = Model {
            n_origins: no,
            n_dev: nd,
            cells,
            phi,
            shape,
            log_prior: prior_ultimate.iter().map(|m| m.ln()).collect(),
            prior_ultimate,
            development_precision: self.development_sd.powi(-2),
        };
        Ok((fit.chain_ladder, model, start))
    }

    fn provenance(
        &self,
        column: &str,
        exposure: &str,
        model: &Model,
        hasher: InputHasher,
    ) -> Provenance {
        let s = &self.sampler;
        Provenance::new("bayes_bf_normalised")
            .param("apriori", self.apriori)
            .param("cv", self.cv)
            .param("development_sd", self.development_sd)
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
            .param("exposure", exposure)
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(hasher.finish())
    }
}

impl BayesBfNormalisedFit {
    /// Posterior summary of each parameter: mean, sd, 5/50/95%
    /// quantiles, R-hat, bulk and tail ESS.
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        Ok(self.posterior.summary()?)
    }

    /// The posterior mean of each age's share `p_d` of the ultimate, in
    /// development order; the shares sum to 1.
    pub fn pattern(&self) -> Vec<f64> {
        let (no, nd) = (self.model.n_origins, self.model.n_dev);
        let dim = self.posterior.dim();
        let n = self.posterior.n_draws() as f64;
        let mut share = vec![0.0; nd];
        for theta in self.posterior.values().chunks_exact(dim) {
            for (s, l) in share.iter_mut().zip(log_pattern(no, nd, theta)) {
                *s += l.exp() / n;
            }
        }
        share
    }

    /// The posterior mean of each origin's expected ultimate `U_w`, the
    /// quantity that `apriori` and `cv` put the prior on (the reserve's
    /// process error and the cells already paid are not in it).
    pub fn expected_ultimate(&self) -> Vec<f64> {
        let no = self.model.n_origins;
        let dim = self.posterior.dim();
        let n = self.posterior.n_draws() as f64;
        let mut ultimate = vec![0.0; no];
        for theta in self.posterior.values().chunks_exact(dim) {
            for (u, a) in ultimate.iter_mut().zip(theta) {
                *u += a.exp() / n;
            }
        }
        ultimate
    }

    /// PSIS-LOO of the observed increments, as
    /// [`BayesOdpFit::loo`](super::BayesOdpFit::loo): the pointwise term
    /// is the normalised density of the over-dispersed Poisson at
    /// `U_w p_d` and `phi`, so the result compares with `BayesOdp`'s on the
    /// same triangle. A triangle with a negative increment has no such
    /// density and is an error.
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
        let rows: Vec<Vec<f64>> = self
            .posterior
            .values()
            .par_chunks_exact(self.posterior.dim())
            .map(|theta| {
                let log_share = log_pattern(model.n_origins, model.n_dev, theta);
                model
                    .cells
                    .iter()
                    .map(|&(w, d, x)| {
                        let mu = (theta[w] + log_share[d]).exp();
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
    /// `reserve`, the `exposure` and `prior_ultimate`, and the `mean` and
    /// `std_dev` of the posterior predictive reserve.
    pub fn to_long(&self) -> FitTable {
        let mut table = self.plug_in().to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        table
    }

    /// One row: the chain ladder's totals, the total `exposure` and
    /// `prior_ultimate`, the `scale`, and the `mean` and `std_dev` of the
    /// posterior predictive total reserve.
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
                exposure: self.exposure.clone(),
                prior_ultimate: self.prior_ultimate.clone(),
            }],
        }
    }
}

impl ReserveFit for BayesBfNormalisedFit {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        super::bf::origin_columns(&self.exposure, &self.prior_ultimate)
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        super::bf::total_columns(self.scale, &self.exposure, &self.prior_ultimate)
    }
}

/// The log posterior on `theta = (u_1..u_n, beta_2..beta_m)`, sampled with
/// `u_w = alpha_w - log m_w`; `beta_1 = 0` is not a parameter.
#[derive(Debug, Clone)]
struct Model {
    n_origins: usize,
    n_dev: usize,
    /// Observed increments: (origin, development position, value).
    cells: Vec<(usize, usize, f64)>,
    phi: f64,
    /// Shape `1 / cv^2` of the gamma prior on each `U_w`; 0 is flat on
    /// `alpha_w`.
    shape: f64,
    /// Prior mean `m_w` of each `U_w`.
    prior_ultimate: Vec<f64>,
    /// `log m_w`.
    log_prior: Vec<f64>,
    /// `1 / development_sd^2`.
    development_precision: f64,
}

impl Model {
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = (1..=self.n_origins)
            .map(|w| format!("alpha[{w}]"))
            .collect();
        names.extend((2..=self.n_dev).map(|d| format!("beta[{d}]")));
        names
    }

    /// `log sum_k exp(beta_k)` with `beta_1 = 0`, on either scale (the
    /// development block is the same).
    fn log_sum(&self, theta: &[f64]) -> f64 {
        log_sum(self.n_origins, theta)
    }

    /// `eta_wd` on the sampling scale `theta = (u, beta)`: centred on the
    /// prior, `u` keeps full precision under a tight prior, where `alpha_w`
    /// near `log m_w` (about 15) would round its steps to 2e-15.
    fn eta(&self, theta: &[f64], log_sum: f64, w: usize, d: usize) -> f64 {
        let beta = if d == 0 {
            0.0
        } else {
            theta[self.n_origins + d - 1]
        };
        theta[w] + self.log_prior[w] + beta - log_sum
    }

    /// `(alpha, beta)` from the sampling scale's `(u, beta)`.
    fn natural(&self, theta: &[f64]) -> Vec<f64> {
        let mut natural = theta.to_vec();
        for (a, l) in natural.iter_mut().zip(&self.log_prior) {
            *a += l;
        }
        natural
    }

    /// `(u, beta)` from the natural scale's `(alpha, beta)`.
    fn sampling(&self, natural: &[f64]) -> Vec<f64> {
        let mut theta = natural.to_vec();
        for (u, l) in theta.iter_mut().zip(&self.log_prior) {
            *u -= l;
        }
        theta
    }
}

/// The shares `S_d / (M_d + lambda)` that maximise the likelihood given
/// the ultimates fixed at their prior, `S_d` the observed increments of
/// column `d` and `M_d` the prior ultimates of the origins observed there,
/// `lambda` the root of `sum_d S_d / (M_d + lambda) = 1` above `-min M_d`
/// (by bisection: the sum falls from infinity to 0 there).
fn strong_prior_shares(column_sum: &[f64], column_prior: &[f64]) -> Vec<f64> {
    let excess = |lambda: f64| {
        column_sum
            .iter()
            .zip(column_prior)
            .map(|(s, m)| s / (m + lambda))
            .sum::<f64>()
            - 1.0
    };
    let floor = column_prior.iter().cloned().fold(f64::INFINITY, f64::min);
    let (mut lo, mut hi) = (-floor * (1.0 - 1e-12), floor);
    while excess(hi) > 0.0 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if excess(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let lambda = 0.5 * (lo + hi);
    column_sum
        .iter()
        .zip(column_prior)
        .map(|(s, m)| s / (m + lambda))
        .collect()
}

/// `log sum_k exp(beta_k)` of the development block of `theta` (after
/// `n_origins` entries), with `beta_1 = 0`.
fn log_sum(n_origins: usize, theta: &[f64]) -> f64 {
    let rest = &theta[n_origins..];
    let top = rest.iter().fold(0.0_f64, |m, &b| m.max(b));
    top + ((-top).exp() + rest.iter().map(|b| (b - top).exp()).sum::<f64>()).ln()
}

/// `log p_d` of every age from a draw of `(alpha, beta)`.
fn log_pattern(n_origins: usize, n_dev: usize, theta: &[f64]) -> Vec<f64> {
    let l = log_sum(n_origins, theta);
    (0..n_dev)
        .map(|d| {
            if d == 0 {
                -l
            } else {
                theta[n_origins + d - 1] - l
            }
        })
        .collect()
}

/// `eta_wd` from a draw of `(alpha, beta)` with `n_origins` origins.
fn natural_eta(n_origins: usize, theta: &[f64], w: usize, d: usize) -> f64 {
    let beta = if d == 0 {
        0.0
    } else {
        theta[n_origins + d - 1]
    };
    theta[w] + beta - log_sum(n_origins, theta)
}

/// A fitted [`BayesBfNormalised`] as the one-year view sees it
/// ([`OdpCells`], decision 17): the draws are stored as `(alpha, beta)`, the
/// model samples `(u, beta)` with `u_w = alpha_w - log m_w`, a shift, so the
/// log posterior moves to `u` and needs no Jacobian, and a refit starts
/// Newton at the opening posterior mean moved to `u`. Every origin and age
/// has an observed cell, so the draws carry every effect a future cell
/// needs; the prior ultimates and `phi` stay at their opening values.
struct NormalisedRefit<'a> {
    bf: &'a BayesBfNormalised,
    model: &'a Model,
    /// The opening posterior mean on the sampling scale, the refits'
    /// Newton start.
    start: Vec<f64>,
}

impl<'a> NormalisedRefit<'a> {
    fn new(bf: &'a BayesBfNormalised, fit: &'a BayesBfNormalisedFit) -> Self {
        let model = &fit.model;
        Self {
            bf,
            model,
            start: model.sampling(&fit.posterior.mean()),
        }
    }

    fn extended(&self, extra: &[(usize, usize, f64)]) -> Model {
        let mut extended = self.model.clone();
        extended.cells.extend_from_slice(extra);
        extended
    }
}

impl OdpCells for NormalisedRefit<'_> {
    fn eta(&self, theta: &[f64], w: usize, d: usize) -> f64 {
        natural_eta(self.model.n_origins, theta, w, d)
    }

    fn phi(&self) -> f64 {
        self.model.phi
    }

    fn log_posterior(&self, theta: &[f64], extra: &[(usize, usize, f64)]) -> f64 {
        let mut grad = vec![0.0; theta.len()];
        if extra.is_empty() {
            // Moment matching's many calls on the observed triangle.
            return self
                .model
                .log_posterior(&self.model.sampling(theta), &mut grad);
        }
        self.extended(extra)
            .log_posterior(&self.model.sampling(theta), &mut grad)
    }

    /// Sampled as [`BayesBfNormalised::fit`] samples.
    fn refit(&self, extra: &[(usize, usize, f64)], seed: u64) -> Result<(PosteriorDraws, bool)> {
        let (posterior, diagnostics) =
            self.bf
                .sample(&self.extended(extra), self.start.clone(), seed)?;
        Ok((posterior, !diagnostics.warnings.is_empty()))
    }
}

/// The log posterior of [`BayesBfNormalised`], on `(u, beta)`. The
/// likelihood is concave in `(alpha', beta')` of the unnormalised
/// parameterisation, but through the softmax it is not in `(u, beta)`; the
/// posterior is unimodal in practice and [`mode`]'s ridge covers a
/// non-concave start.
impl Posterior for Model {
    fn dim(&self) -> usize {
        self.n_origins + self.n_dev - 1
    }

    fn log_posterior(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        let (no, nd) = (self.n_origins, self.n_dev);
        let l = self.log_sum(theta);
        let mut f = 0.0;
        // Gamma(shape, rate shape / m) on U = exp(alpha), with the
        // Jacobian: shape alpha - shape exp(alpha) / m, here less its value
        // at alpha = log m, in u = alpha - log m (as BayesBf).
        for w in 0..no {
            let u = theta[w];
            let em1 = u.exp_m1();
            f += self.shape * (u - em1);
            grad[w] = -self.shape * em1;
        }
        for j in no..self.dim() {
            f -= 0.5 * self.development_precision * theta[j] * theta[j];
            grad[j] = -self.development_precision * theta[j];
        }
        let mut total = 0.0;
        for &(w, d, x) in &self.cells {
            let eta = self.eta(theta, l, w, d);
            let mu = eta.exp();
            f += (x * eta - mu) / self.phi;
            let g = (x - mu) / self.phi;
            grad[w] += g;
            if d > 0 {
                grad[no + d - 1] += g;
            }
            total += g;
        }
        // d eta / d beta_k = [d = k] - p_k, through the normalisation.
        for k in 1..nd {
            let share = (theta[no + k - 1] - l).exp();
            grad[no + k - 1] -= share * total;
        }
        f
    }

    fn negative_hessian(&self, theta: &[f64]) -> Vec<f64> {
        let (no, nd, p) = (self.n_origins, self.n_dev, self.dim());
        let l = self.log_sum(theta);
        let share: Vec<f64> = (1..nd).map(|k| (theta[no + k - 1] - l).exp()).collect();
        let mut h = vec![0.0; p * p];
        for w in 0..no {
            h[w * p + w] = self.shape * theta[w].exp();
        }
        for j in no..p {
            h[j * p + j] = self.development_precision;
        }
        let mut total = 0.0;
        let mut v = vec![0.0; p];
        for &(w, d, x) in &self.cells {
            let eta = self.eta(theta, l, w, d);
            let mu = eta.exp();
            let weight = mu / self.phi;
            total += (x - mu) / self.phi;
            // The gradient of eta_wd: e_w, plus e_d, less the shares.
            v.fill(0.0);
            v[w] = 1.0;
            for (k, s) in share.iter().enumerate() {
                v[no + k] = -s;
            }
            if d > 0 {
                v[no + d - 1] += 1.0;
            }
            for i in 0..p {
                if v[i] != 0.0 {
                    for j in 0..p {
                        h[i * p + j] += weight * v[i] * v[j];
                    }
                }
            }
        }
        // The curvature of the normalisation, weighted by the cells'
        // summed score: total * (diag(p) - p p').
        for a in 0..nd - 1 {
            for b in 0..nd - 1 {
                let c = if a == b { share[a] } else { 0.0 } - share[a] * share[b];
                h[(no + a) * p + no + b] += total * c;
            }
        }
        h
    }
}

/// The log posterior written once over [`Scalar`](super::scalar::Scalar),
/// the oracle of the compartmental note's decision 4: test-only, beside
/// the hand-written [`Posterior::log_posterior`], which stays the fast
/// path.
#[cfg(test)]
impl super::autodiff::GenericDensity for Model {
    fn n_params(&self) -> usize {
        self.dim()
    }

    fn log_density<S: super::scalar::Scalar>(&self, theta: &[S]) -> S {
        let no = self.n_origins;
        let mut f = S::from_f64(0.0);
        for &u in &theta[..no] {
            f += (u - u.exp_m1()) * self.shape;
        }
        let mut sum = S::from_f64(1.0);
        for &b in &theta[no..] {
            f -= b * b * (0.5 * self.development_precision);
            sum += b.exp();
        }
        let l = sum.ln();
        for &(w, d, x) in &self.cells {
            let beta = if d == 0 {
                S::from_f64(0.0)
            } else {
                theta[no + d - 1]
            };
            let eta = theta[w] + self.log_prior[w] + beta - l;
            f += (eta * x - eta.exp()) / self.phi;
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bayes::laplace::tests::assert_gradient;
    use crate::bayes::{BayesBf, BayesOdp};
    use crate::triangle::tests::{GENINS, annual};
    use crate::{BornhuetterFerguson, ChainLadder, Lag};
    use crate::{DevelopmentColumn, Grain, Long};
    use prospicio_bayes::nuts::LogDensity;
    use prospicio_core::{Month, Period};
    use prospicio_prob::Distribution;

    /// A cumulative annual triangle with `values` and a `premium` column,
    /// one premium per origin repeated on each age.
    fn with_premium(first_year: i32, rows: &[&[f64]], premium: &[f64]) -> Triangle {
        let (mut origin, mut ages, mut values, mut prem) = (vec![], vec![], vec![], vec![]);
        for (k, row) in rows.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                origin.push(Month::january(first_year + k as i32));
                ages.push(12 * (d as Lag + 1));
                values.push(v);
                prem.push(premium[k]);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("values", &values), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    /// GenIns with R ChainLadder's `ClarkCapeCod` premium,
    /// `10,000,000 + 400,000 k`.
    fn genins() -> Triangle {
        let premium: Vec<f64> = (0..10).map(|k| 1e7 + 4e5 * k as f64).collect();
        with_premium(2001, &GENINS, &premium)
    }

    fn quick(seed: u64) -> BayesBfNormalised {
        BayesBfNormalised {
            apriori: 0.5,
            cv: 0.2,
            sampler: Sampler {
                tune: 300,
                draws: 300,
                target_accept: 0.95,
                ..Sampler::default()
            },
            n_sims: 2_000,
            seed,
            ..BayesBfNormalised::default()
        }
    }

    fn sampler(draws: usize) -> Sampler {
        Sampler {
            tune: 300,
            draws,
            target_accept: 0.95,
            ..Sampler::default()
        }
    }

    fn prepared(tri: &Triangle, model: &BayesBfNormalised) -> (ChainLadderFit, Model, Vec<f64>) {
        let segment = tri.segment("values").unwrap();
        let exposure =
            origin_exposure(&segment, &tri.segment("premium").unwrap(), "premium").unwrap();
        model.prepare(&segment, &exposure).unwrap()
    }

    /// The mean reserve by origin at `theta` on the sampling scale: the
    /// future cells' means.
    fn mean_reserves(m: &Model, latest: &[usize], theta: &[f64]) -> Vec<f64> {
        let l = m.log_sum(theta);
        (0..m.n_origins)
            .map(|w| {
                (latest[w] + 1..m.n_dev)
                    .map(|d| m.eta(theta, l, w, d).exp())
                    .sum()
            })
            .collect()
    }

    /// The share at every age at `theta` on the sampling scale.
    fn shares(m: &Model, theta: &[f64]) -> Vec<f64> {
        let l = m.log_sum(theta);
        (0..m.n_dev)
            .map(|d| {
                let beta = if d == 0 {
                    0.0
                } else {
                    theta[m.n_origins + d - 1]
                };
                (beta - l).exp()
            })
            .collect()
    }

    /// The strong-prior limit's shares: `U_w = m_w` and `p_d = S_d / (M_d
    /// + lambda)` with `lambda` such that they sum to 1.
    fn closed_shares(m: &Model) -> Vec<f64> {
        let nd = m.n_dev;
        let (mut s, mut big_m) = (vec![0.0; nd], vec![0.0; nd]);
        for &(w, d, x) in &m.cells {
            s[d] += x;
            big_m[d] += m.prior_ultimate[w];
        }
        strong_prior_shares(&s, &big_m)
    }

    #[test]
    fn the_gradient_and_hessian_match_finite_differences() {
        for cv in [0.2, f64::INFINITY] {
            let model = BayesBfNormalised { cv, ..quick(0) };
            let (_, model, start) = prepared(&genins(), &model);
            let x: Vec<f64> = start
                .iter()
                .enumerate()
                .map(|(j, v)| v + 0.01 * (j as f64 - 5.0))
                .collect();
            assert_gradient(|t, g| model.log_posterior(t, g), &x);
            // The analytic negative Hessian against the trait's default
            // central differences of the gradient.
            struct Numeric<'a>(&'a Model);
            impl Posterior for Numeric<'_> {
                fn dim(&self) -> usize {
                    self.0.dim()
                }
                fn log_posterior(&self, t: &[f64], g: &mut [f64]) -> f64 {
                    self.0.log_posterior(t, g)
                }
            }
            let (exact, numeric) = (
                model.negative_hessian(&x),
                Numeric(&model).negative_hessian(&x),
            );
            for (a, b) in exact.iter().zip(&numeric) {
                assert!((a - b).abs() <= 1e-5 * (1.0 + a.abs()), "{a} vs {b}");
            }
            let mode = mode(&model, start).unwrap();
            let whitened = Whitened::new(&model, &mode).unwrap();
            let z: Vec<f64> = (0..model.dim())
                .map(|j| 0.3 * ((j % 3) as f64 - 1.0))
                .collect();
            assert_gradient(|z, g| whitened.log_density(z, g).unwrap(), &z);
        }
    }

    #[test]
    fn the_generic_density_matches_the_hand_written_one() {
        use crate::bayes::autodiff::{assert_oracle, posterior_points};
        for (label, cv) in [
            ("BayesBfNormalised cv 0.2", 0.2),
            ("BayesBfNormalised flat", f64::INFINITY),
        ] {
            let model = BayesBfNormalised { cv, ..quick(0) };
            let (_, model, start) = prepared(&genins(), &model);
            let mode = mode(&model, start).unwrap();
            let whitened = Whitened::new(&model, &mode).unwrap();
            let points = posterior_points(&whitened, model.dim(), 20);
            assert_oracle(label, &model, &points);
        }
    }

    /// With a flat prior on `alpha` the mode's fitted means are the chain
    /// ladder's, up to the vague prior on `beta`, which at sd 1,000 moves
    /// them by less than 1e-4.
    #[test]
    fn a_flat_prior_has_the_chain_ladder_at_its_mode() {
        let model = BayesBfNormalised {
            cv: f64::INFINITY,
            development_sd: 1e3,
            ..quick(0)
        };
        let tri = genins();
        let (_, m, start) = prepared(&tri, &model);
        let theta = mode(&m, start).unwrap();
        let l = m.log_sum(&theta);
        let segment = tri.segment("values").unwrap();
        let (fit, _) = prepare(&segment, &segment.ages, Tail::default()).unwrap();
        for &(w, d, _) in &m.cells {
            let mu = m.eta(&theta, l, w, d).exp();
            let expected = fit.fitted[w * segment.n_dev + d];
            assert!(
                (mu / expected - 1.0).abs() < 1e-4,
                "({w}, {d}): {mu} vs {expected}"
            );
        }
    }

    /// When the prior ultimates are the chain ladder's, the prior's point is
    /// the posterior's mode whatever the `cv`: the ultimates stay put, the
    /// shares are the chain ladder's pattern, and the reserve is the
    /// classical Bornhuetter–Ferguson reserve, which is the chain ladder's.
    #[test]
    fn the_prior_at_the_chain_ladder_ultimates_gives_the_classical_reserve() {
        let tri = genins();
        let cl = ChainLadder::default().fit(&tri, "values").unwrap();
        let rows: Vec<&[f64]> = GENINS.to_vec();
        let at_cl = with_premium(2001, &rows, &cl.ultimate);
        let classical = BornhuetterFerguson {
            apriori: 1.0,
            ..Default::default()
        }
        .fit(&at_cl, "values", "premium")
        .unwrap();
        assert!((classical.total_reserve() / cl.total_reserve() - 1.0).abs() < 1e-9);
        for cv in [1e-3, 0.2, 1.0] {
            let model = BayesBfNormalised {
                apriori: 1.0,
                cv,
                development_sd: 100.0,
                ..quick(0)
            };
            let (fit, m, start) = prepared(&at_cl, &model);
            let theta = mode(&m, start).unwrap();
            for u in &theta[..m.n_origins] {
                assert!(u.abs() < 1e-4, "cv {cv}: u {u}");
            }
            let total: f64 = mean_reserves(&m, &fit.latest_position, &theta).iter().sum();
            assert!(
                (total / classical.total_reserve() - 1.0).abs() < 1e-5,
                "cv {cv}: {total} vs {}",
                classical.total_reserve()
            );
        }
    }

    /// A very tight prior at other ultimates: `U_w` stays at `m_w` and the
    /// shares are the closed form's `S_d / (M_d + lambda)`.
    #[test]
    fn a_strong_prior_fixes_the_ultimates_and_the_shares_follow() {
        for (apriori, cv) in [
            (0.4, 1e-4),
            (0.5, 1e-4),
            (0.6, 1e-4),
            (0.6, 1e-6),
            (0.3, 1e-6),
        ] {
            let model = BayesBfNormalised {
                apriori,
                cv,
                development_sd: 100.0,
                ..quick(0)
            };
            let (_, m, start) = prepared(&genins(), &model);
            let theta = mode(&m, start).unwrap();
            for (w, u) in theta.iter().take(m.n_origins).enumerate() {
                assert!(u.abs() < 1e-5, "{w}: u {u}");
            }
            let closed = closed_shares(&m);
            assert!((closed.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            for (d, (got, want)) in shares(&m, &theta).iter().zip(&closed).enumerate() {
                assert!(
                    (got / want - 1.0).abs() < 1e-3,
                    "apriori {apriori}, age {d}: {got} vs {want}"
                );
            }
        }
    }

    /// The flat prior gives `BayesOdp`'s reserves within Monte Carlo
    /// error: the means within `0.1 s` and the sds within 10% by origin
    /// and in total, `s` the sd of `BayesOdp`'s reserve.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_flat_prior_reproduces_bayes_odp() {
        let tri = genins();
        let bf = BayesBfNormalised {
            cv: f64::INFINITY,
            sampler: sampler(1000),
            n_sims: 10_000,
            ..quick(1)
        }
        .fit(&tri, "values", "premium")
        .unwrap();
        let odp = BayesOdp {
            sampler: sampler(1000),
            n_sims: 10_000,
            seed: 2,
            ..BayesOdp::default()
        }
        .fit(&tri, "values")
        .unwrap();
        let mut keys: Vec<Option<Period>> = (2002..=2010).map(|y| Some(Period::year(y))).collect();
        keys.push(None);
        for key in keys {
            let pick = |r: &PredictiveDistribution| match key {
                Some(o) => r.marginal(&vec![o.into()]).unwrap(),
                None => r.total().clone(),
            };
            let (a, b) = (pick(&bf.reserves), pick(&odp.reserves));
            let s = b.std_dev();
            assert!(
                (a.mean() - b.mean()).abs() < 0.1 * s,
                "{key:?}: mean {} vs {} (sd {s})",
                a.mean(),
                b.mean()
            );
            assert!(
                (a.std_dev() / s - 1.0).abs() < 0.10,
                "{key:?}: sd {} vs {s}",
                a.std_dev()
            );
        }
    }

    /// A very tight prior at the chain ladder's ultimates gives the
    /// classical Bornhuetter–Ferguson reserve in the fit, with no process
    /// error, within a hundredth of the sd of the reserve given the
    /// parameters; at other ultimates its mean is the closed form's shares
    /// applied to the future cells, within 1% (the posterior mean of the
    /// shares against their mode).
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_tight_prior_reproduces_the_bornhuetter_ferguson_limit() {
        let tri = genins();
        let tight = BayesBfNormalised {
            cv: 1e-3,
            development_sd: 100.0,
            process: ProcessDistribution::None,
            sampler: sampler(1000),
            ..quick(3)
        };
        let cl = ChainLadder::default().fit(&tri, "values").unwrap();
        let rows: Vec<&[f64]> = GENINS.to_vec();
        let at_cl = with_premium(2001, &rows, &cl.ultimate);
        let classical = BornhuetterFerguson {
            apriori: 1.0,
            ..Default::default()
        }
        .fit(&at_cl, "values", "premium")
        .unwrap()
        .total_reserve();
        let fit = BayesBfNormalised {
            apriori: 1.0,
            ..tight.clone()
        }
        .fit(&at_cl, "values", "premium")
        .unwrap();
        let total = fit.reserves.total();
        assert!(
            (total.mean() / classical - 1.0).abs() < 5e-3,
            "{} vs {classical}",
            total.mean()
        );

        let fit = tight.fit(&tri, "values", "premium").unwrap();
        let (_, m, _) = prepared(&tri, &tight);
        let closed = closed_shares(&m);
        let latest = &fit.chain_ladder.latest_position;
        let expected: f64 = (0..m.n_origins)
            .map(|w| {
                (latest[w] + 1..m.n_dev)
                    .map(|d| m.prior_ultimate[w] * closed[d])
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (fit.reserves.total().mean() / expected - 1.0).abs() < 0.01,
            "{} vs {expected}",
            fit.reserves.total().mean()
        );
    }

    /// The a-priori loss ratio is a real prior on the ultimate: a 20%
    /// change in it moves the immature origins' reserves, where
    /// `BayesBf`'s, whose pattern cancels the level, hardly move.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_apriori_moves_the_reserve() {
        let tri = genins();
        let reserves = |apriori: f64| {
            let fit = BayesBfNormalised {
                apriori,
                cv: 0.1,
                process: ProcessDistribution::None,
                ..quick(8)
            }
            .fit(&tri, "values", "premium")
            .unwrap();
            let bf = BayesBf {
                apriori,
                cv: 0.1,
                process: ProcessDistribution::None,
                ..BayesBf::default()
            }
            .fit(&tri, "values", "premium")
            .unwrap();
            // The newest three origins and the total.
            let young = |r: &PredictiveDistribution| -> f64 {
                (2008..=2010)
                    .map(|y| r.marginal(&vec![Period::year(y).into()]).unwrap().mean())
                    .sum()
            };
            (young(&fit.reserves), young(&bf.reserves))
        };
        let (low, high) = (reserves(0.4), reserves(0.6));
        // A 50% rise in the prior mean: the new model's young reserves rise
        // by well over 20%, BayesBf's by under 10%.
        let new = high.0 / low.0 - 1.0;
        let old = high.1 / low.1 - 1.0;
        eprintln!("apriori 0.4 to 0.6: BayesBfNormalised {new:+.3}, BayesBf {old:+.3}");
        assert!(new > 0.2, "BayesBfNormalised moved {new}");
        assert!(old.abs() < 0.1, "BayesBf moved {old}");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_seed_replays_whatever_the_threads() {
        let tri = genins();
        let a = quick(3).fit(&tri, "values", "premium").unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let b = pool.install(|| quick(3).fit(&tri, "values", "premium").unwrap());
        assert_eq!(a.posterior, b.posterior);
        assert_eq!(a.reserves.draw_matrix(), b.reserves.draw_matrix());
        let c = quick(4).fit(&tri, "values", "premium").unwrap();
        assert_ne!(a.posterior.values(), c.posterior.values());
        let mut d = quick(3);
        d.sampler.seed = 99;
        assert_eq!(
            d.fit(&tri, "values", "premium").unwrap().posterior,
            a.posterior
        );
    }

    #[test]
    fn errors() {
        let tri = genins();
        let fit = |m: BayesBfNormalised| m.fit(&tri, "values", "premium");
        for (model, name) in [
            (
                BayesBfNormalised {
                    apriori: 0.0,
                    ..quick(0)
                },
                "apriori",
            ),
            (
                BayesBfNormalised {
                    cv: 0.0,
                    ..quick(0)
                },
                "cv",
            ),
            (
                BayesBfNormalised {
                    cv: f64::NAN,
                    ..quick(0)
                },
                "cv",
            ),
            (
                BayesBfNormalised {
                    development_sd: f64::INFINITY,
                    ..quick(0)
                },
                "development_sd",
            ),
            (
                BayesBfNormalised {
                    n_sims: 0,
                    ..quick(0)
                },
                "n_sims",
            ),
        ] {
            assert!(
                matches!(fit(model), Err(Error::InvalidSetting { name: n, .. }) if n == name),
                "{name}"
            );
        }
        assert!(matches!(
            quick(0).fit(&tri, "values", "earned"),
            Err(Error::UnknownColumn(_))
        ));
        let zero = with_premium(
            2020,
            &[&[1.0, 2.0, 3.0], &[1.0, 2.0], &[1.0]],
            &[1.0, 0.0, 1.0],
        );
        assert!(matches!(
            quick(0).fit(&zero, "values", "premium"),
            Err(Error::InvalidExposure { .. })
        ));

        // The newest origin's only cell is negative: improper under a flat
        // prior, proper under an informative one.
        let rows: [&[f64]; 4] = [
            &[100.0, 150.0, 160.0, 165.0],
            &[110.0, 170.0, 180.0],
            &[120.0, 175.0],
            &[-1.0],
        ];
        let negative = with_premium(2020, &rows, &[200.0; 4]);
        let flat = BayesBfNormalised {
            cv: f64::INFINITY,
            ..quick(0)
        };
        let e = flat.fit(&negative, "values", "premium").unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("origin 2023")
                && m.contains("no log-linear mean")),
            "{e}"
        );
        let weak = BayesBfNormalised {
            cv: 1e3,
            ..quick(0)
        };
        let e = weak.fit(&negative, "values", "premium").unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("origin 2023") && m.contains("improper")),
            "{e}"
        );
        // The last development column falls.
        let falling = with_premium(
            2020,
            &[
                &[100.0, 150.0, 160.0, 155.0],
                &[110.0, 170.0, 180.0],
                &[120.0, 175.0],
                &[130.0],
            ],
            &[200.0; 4],
        );
        let e = quick(0).fit(&falling, "values", "premium").unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("development age 48 months")),
            "{e}"
        );
        assert!(matches!(
            quick(0).fit(&annual(2020, &[&[1.0, 2.0], &[1.0]]), "values", "values"),
            Err(Error::Bootstrap(_))
        ));
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_fit_and_its_tables() {
        let fit = quick(5).fit(&genins(), "values", "premium").unwrap();
        assert_eq!(fit.reserves.dims(), ["origin"]);
        assert_eq!(fit.reserves.n_components(), 10);
        let oldest = fit
            .reserves
            .marginal(&vec![Period::year(2001).into()])
            .unwrap();
        assert_eq!((oldest.mean(), oldest.std_dev()), (0.0, 0.0));
        // Ten alphas and nine betas: the first age's share is the corner.
        assert_eq!(fit.posterior.names().len(), 19);
        assert_eq!(fit.posterior.names()[9], "alpha[10]");
        assert_eq!(fit.posterior.names()[10], "beta[2]");
        assert_eq!(fit.prior_ultimate[9], 0.5 * 13.6e6);
        assert_eq!(fit.exposure[0], 1e7);
        let pattern = fit.pattern();
        assert_eq!(pattern.len(), 10);
        assert!((pattern.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(pattern.iter().all(|p| *p > 0.0));
        let ultimate = fit.expected_ultimate();
        assert_eq!(ultimate.len(), 10);
        // The prior mean is 0.5 premium; the posterior ultimates sit near
        // the chain ladder's.
        for (u, cl) in ultimate.iter().zip(&fit.chain_ladder.ultimate) {
            assert!((u / cl - 1.0).abs() < 0.5, "{u} vs {cl}");
        }
        let long = fit.to_long();
        assert_eq!(long.n_rows(), 10);
        assert_eq!(long.column("prior_ultimate").unwrap(), fit.prior_ultimate);
        assert!(long.column("mean").is_some() && long.column("std_dev").is_some());
        let totals = fit.totals();
        assert_eq!(totals.column("scale").unwrap(), [fit.scale]);
        assert_eq!(totals.column("exposure").unwrap(), [118e6]);
        assert_eq!(fit.development_table().n_rows(), 10);
        assert_eq!(fit.summary().unwrap().len(), 19);
        let provenance = fit.reserves.provenance();
        assert_eq!(provenance.model, "bayes_bf_normalised");
        for key in [
            "apriori",
            "cv",
            "development_sd",
            "scale",
            "exposure",
            "seed",
        ] {
            assert!(
                provenance.parameters.iter().any(|(k, _)| k == key),
                "{key} missing"
            );
        }
        // The scale is the ODP bootstrap's (and BayesOdp's), the chain
        // ladder's Pearson scale.
        let boot = crate::OdpBootstrap {
            n_sims: 10,
            ..Default::default()
        }
        .fit(&genins(), "values")
        .unwrap();
        assert_eq!(fit.scale, boot.scale);
        let short = BayesBfNormalised {
            sampler: Sampler {
                tune: 100,
                draws: 25,
                ..Sampler::default()
            },
            n_sims: 100,
            ..quick(5)
        };
        let loo = short
            .fit(&genins(), "values", "premium")
            .unwrap()
            .loo()
            .unwrap();
        assert_eq!(loo.pareto_k.len(), 55);
        assert!(loo.estimate.elpd.is_finite());
    }

    /// The one-year conformance suite (decision 17) on GenIns with an
    /// informative and a flat prior, with a fully developed square as the
    /// triangle without a year.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_one_year_view_conforms() {
        let square = with_premium(
            2001,
            &[
                &[357.0, 1124.0, 1735.0, 2218.0],
                &[352.0, 1236.0, 2170.0, 3353.0],
                &[290.0, 1292.0, 2219.0, 3235.0],
                &[310.0, 1418.0, 2195.0, 3757.0],
            ],
            &[5000.0, 6000.0, 6000.0, 7000.0],
        );
        for (name, cv) in [
            ("BayesBfNormalised cv 0.2", 0.2),
            ("BayesBfNormalised flat", f64::INFINITY),
        ] {
            let model = BayesBfNormalised { cv, ..quick(6) };
            let fitted: Vec<(Triangle, Segment, BayesBfNormalisedFit)> = [genins(), square.clone()]
                .into_iter()
                .map(|tri| {
                    let segment = tri.segment("values").unwrap();
                    let exposure =
                        origin_exposure(&segment, &tri.segment("premium").unwrap(), "premium")
                            .unwrap();
                    let fit = model
                        .fit_segment(&segment, "values", "premium", exposure)
                        .unwrap();
                    (tri, segment, fit)
                })
                .collect();
            let cells: Vec<NormalisedRefit> = fitted
                .iter()
                .map(|(_, _, fit)| NormalisedRefit::new(&model, fit))
                .collect();
            let years: Vec<_> = fitted
                .iter()
                .zip(&cells)
                .map(|((tri, segment, fit), cells)| {
                    OdpYear::new(
                        cells,
                        model.process,
                        tri,
                        segment,
                        &fit.chain_ladder.latest_position,
                    )
                })
                .collect();
            crate::bayes::one_year::conformance::check(
                name,
                &years[0],
                &fitted[0].2.posterior,
                &years[1],
                &fitted[1].2.posterior,
            );
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_one_year_view_reweights_the_fits_posterior() {
        let model = BayesBfNormalised {
            n_sims: 1_000,
            ..quick(7)
        };
        let settings = BayesOneYear {
            refit: Some(crate::bayes::OneYearRefit::ReweightOnly),
            ..BayesOneYear::default()
        };
        let view = model
            .one_year(&genins(), "values", "premium", &settings)
            .unwrap();
        // The posterior and lifetime reserves are fit()'s, draw for draw.
        let fit = model.fit(&genins(), "values", "premium").unwrap();
        assert_eq!(view.fit.posterior, fit.posterior);
        assert_eq!(view.fit.reserves.draw_matrix(), fit.reserves.draw_matrix());
        assert_eq!(view.cdr.dims(), ["origin"]);
        assert_eq!(view.pareto_k.len(), 1_000);
        assert!(view.refitted.is_empty());
        // 2001 is at the last age: no cell, no change.
        assert!(view.cdr.draw_matrix().chunks(10).all(|r| r[0] == 0.0));
        assert_eq!(view.opening_reserve[0], 0.0);
        // The tables carry the prior's columns, as the lifetime fit's do.
        let long = view.to_long();
        assert_eq!(long.n_rows(), 10);
        assert_eq!(long.column("prior_ultimate").unwrap(), fit.prior_ultimate);
        assert!(long.column("opening_reserve").is_some());
        let totals = view.totals();
        assert_eq!(totals.column("exposure").unwrap(), [118e6]);
        assert_eq!(
            totals.column("flagged_share").unwrap(),
            [view.flagged_share()]
        );
        let provenance = view.cdr.provenance();
        assert_eq!(provenance.model, "bayes_bf_normalised_one_year");
        for key in ["apriori", "cv", "k_threshold", "refit", "exposure"] {
            assert!(
                provenance.parameters.iter().any(|(k, _)| k == key),
                "{key} missing"
            );
        }
        // The one-year view is narrower than the lifetime view.
        assert!(view.cdr.total().std_dev() < fit.reserves.total().std_dev());
        let e = model
            .one_year(
                &genins(),
                "values",
                "premium",
                &BayesOneYear {
                    k_threshold: f64::NAN,
                    ..settings
                },
            )
            .unwrap_err();
        assert!(
            matches!(
                e,
                Error::InvalidSetting {
                    name: "k_threshold",
                    ..
                }
            ),
            "{e}"
        );
    }
}
