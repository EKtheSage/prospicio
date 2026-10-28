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

use super::bf::{PlugIn, input_hasher, push_posterior_ultimate};
use super::laplace::{Posterior, Whitened, mode};
use super::one_year::{self, BayesOneYear, BayesOneYearFit, OdpCells, OdpYear};
use super::{Diagnostics, Stream, derive_seed, one_year_seed};
use crate::bootstrap::{
    ProcessDistribution, component_sums, prepare, process_draw, push_moments, segment_sums,
};
use crate::chain_ladder::ChainLadderFit;
use crate::error::{Error, Result};
use crate::expected_loss::origin_exposure;
use crate::pattern_draws::{DevelopmentPatterns, predictive_with_patterns};
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
/// which its posterior is improper ([`Error::Bayes`], naming the age or
/// origin, its number of observed cells, their sum and the remedies).
/// Negative cells are allowed. The column rule is not about propriety: the
/// means `U_w p_d` are the positive products `exp(a_w + b_d)` with free
/// `b_d` (the normalisation only reparametrises them), and the likelihood's
/// score in `b_d` is `(S_d - sum_w exp(a_w + b_d)) / phi`, negative
/// everywhere when the column's sum `S_d` is not positive, so the
/// likelihood has no finite maximum: it rises as the share falls to 0, the
/// share that maximises it given the ultimates (`S_d / (M_d + lambda)`
/// above) is negative and the strong-prior start would take its log. The
/// vague normal prior on `beta_d` would still give a proper posterior, with
/// `beta_d` near `development_sd^2 S_d / phi` and so a share of about 0,
/// set by the prior alone: the model cannot say that an age's
/// expected movement is negative, which is what such a column says. The
/// chain ladder accepts the same triangle (a factor below 1 there);
/// [`BayesOdp`](super::BayesOdp) and [`BayesBf`](super::BayesBf) reject it
/// as this model does. A sparse last age (one cell below the previous one,
/// say) is best dropped before the fit: keep the ages before it with
/// [`Triangle::max_age`] (Python `Triangle.max_age`, R `max_age()`), or
/// fold the age into the previous one, and carry the movement left out as
/// a tail outside the model. A falling first age is dropped by rebuilding
/// the triangle from [`Triangle::to_long`] without its rows and passing
/// them to [`Triangle::from_long`] (Python `to_frame` and `from_frame`, R
/// `as.data.frame()` and `triangle()`); a falling middle age has no remedy
/// in this model.
/// Supporting such a column would need a term at the late ages whose sign
/// is free (an additive or tail factor with a normal prior, outside the
/// log-linear ODP likelihood, whose variance `phi * mean` needs a positive
/// mean); the design note's decision 26 has the details.
///
/// The log posterior is not concave in general (the softmax), but
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
    /// The coefficient of variation of one multiplier `s` common to all
    /// origins (`None`, the default, has no such multiplier and the model is
    /// unchanged). With `Some(c)` every origin's prior is conditional on
    /// `s ~ Gamma(shape 1 / c^2, mean 1)`: `U_w | s ~ Gamma(shape 1 / cv^2,
    /// mean s m_w)`, and `s` is sampled with the other parameters as the
    /// last one, `log_share` (`t = log s`), so the data update it for every
    /// origin at once. Each `U_w` keeps the mean `m_w`, and its marginal cv
    /// is `sqrt((1 + cv^2)(1 + c^2) - 1)`; the origins' priors are no longer
    /// independent (design note `reserving-bayes.md`, decision 25). It
    /// needs a finite `cv` and a finite, positive `c`.
    pub common_cv: Option<f64>,
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
            common_cv: None,
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
    /// The pattern `p_d` of the posterior draw each simulation took, as
    /// cumulative shares, row `i` simulation `i` of `reserves`
    /// (`reserving-v02.md` decision 28); [`pattern`](Self::pattern) is
    /// their posterior mean.
    pub patterns: DevelopmentPatterns,
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
        let (reserves, patterns) = predictive_with_patterns(
            &posterior,
            segment,
            self.n_sims,
            derive_seed(self.seed, Stream::Predictive),
            self.provenance(column, exposure, &model, hasher),
            |theta, rng: &mut StreamRng, row, increments| {
                let log_share = log_pattern(no, nd, theta);
                for (w, reserve) in row.iter_mut().enumerate() {
                    *reserve = (latest[w] + 1..nd)
                        .map(|d| process_draw(process, phi, (theta[w] + log_share[d]).exp(), rng))
                        .sum();
                }
                for (x, s) in increments.iter_mut().zip(&log_share) {
                    *x = s.exp();
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
            patterns,
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
        if let Some(c) = self.common_cv {
            if !(c.is_finite() && c > 0.0) {
                return Err(Error::InvalidSetting {
                    name: "common_cv",
                    value: c,
                    expected: "a finite, positive coefficient of variation (None for no \
                               common multiplier)",
                });
            }
            if !self.cv.is_finite() {
                return Err(Error::InvalidSetting {
                    name: "cv",
                    value: self.cv,
                    expected: "a finite coefficient of variation when common_cv is set \
                               (the common multiplier is not identified under a flat prior)",
                });
            }
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
            let row = cells.iter().filter(|c| c.0 == w);
            let (n, sum) = row.fold((0, 0.0), |(n, s), c| (n + 1, s + c.2));
            if sum.is_nan() || shape + sum / phi <= 0.0 {
                return Err(origin_error(&segment.origins[w], n, sum, shape, phi));
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
                let n = cells.iter().filter(|c| c.1 == d).count();
                let factor = d
                    .checked_sub(1)
                    .and_then(|k| fit.chain_ladder.development.ldf.get(k).copied());
                let last_here: Vec<String> = fit
                    .chain_ladder
                    .latest_position
                    .iter()
                    .zip(&segment.origins)
                    .filter(|&(&last, _)| last == d)
                    .map(|(_, origin)| origin.to_string())
                    .collect();
                return Err(column_error(
                    &segment.ages,
                    d,
                    n,
                    sum,
                    factor,
                    &last_here,
                    segment.alone,
                ));
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
        // The common multiplier starts at its prior mean, t = 0.
        let common_shape = self.common_cv.map(|c| c.powi(-2));
        if common_shape.is_some() {
            start.push(0.0);
        }
        let model = Model {
            n_origins: no,
            n_dev: nd,
            cells,
            phi,
            shape,
            common_shape,
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
        let provenance = Provenance::new("bayes_bf_normalised")
            .param("apriori", self.apriori)
            .param("cv", self.cv);
        // Only a model with the multiplier names it, so the others'
        // provenance is unchanged.
        let provenance = match self.common_cv {
            Some(c) => provenance.param("common_cv", c),
            None => provenance,
        };
        provenance
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
    /// `reserve`, the `exposure`, `prior_ultimate` and `ulr` (the chain
    /// ladder's ultimate over the exposure), the `mean` and `std_dev` of
    /// the posterior predictive reserve, and `ultimate_mean` (`latest` plus
    /// `mean`, the posterior mean ultimate) and `ulr_mean` (`ultimate_mean`
    /// over the exposure).
    pub fn to_long(&self) -> FitTable {
        let mut table = self.plug_in().to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        push_posterior_ultimate(&mut table);
        table
    }

    /// One row: the chain ladder's totals, the total `exposure` and
    /// `prior_ultimate`, the `scale`, the total `ulr`, the `mean` and
    /// `std_dev` of the posterior predictive total reserve, and the total
    /// `ultimate_mean` and `ulr_mean` (over the total exposure).
    pub fn totals(&self) -> FitTable {
        let fits = self.plug_in();
        let mut table = fits.totals();
        push_moments(&mut table, "", &segment_sums(&fits, &self.reserves));
        push_posterior_ultimate(&mut table);
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

    fn exposure(&self) -> Option<&[f64]> {
        Some(&self.exposure)
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
    /// Shape `1 / common_cv^2` of the gamma prior (mean 1) on the common
    /// multiplier `s = exp(t)`, whose `t` is the last parameter; `None`
    /// has no such parameter. With it `U_w` is a gamma with mean `s m_w`.
    common_shape: Option<f64>,
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
        if self.common_shape.is_some() {
            names.push("log_share".into());
        }
        names
    }

    /// The index of `t = log s` (the last parameter), if there is a common
    /// multiplier.
    fn t_index(&self) -> Option<usize> {
        self.common_shape.map(|_| self.n_origins + self.n_dev - 1)
    }

    /// `t = log s` of `theta`, 0 without a common multiplier (so that
    /// `u - t` is `u` bit for bit).
    fn t(&self, theta: &[f64]) -> f64 {
        self.t_index().map_or(0.0, |j| theta[j])
    }

    /// `log sum_k exp(beta_k)` with `beta_1 = 0`, on either scale (the
    /// development block is the same).
    fn log_sum(&self, theta: &[f64]) -> f64 {
        log_sum(self.n_origins, self.n_dev, theta)
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

/// `1 cell` or `n cells`.
fn n_cells(n: usize) -> String {
    if n == 1 {
        "1 cell".to_string()
    } else {
        format!("{n} cells")
    }
}

/// The error for an origin whose `n` observed increments sum to `sum`, at
/// or below `-phi / cv^2` (`shape = 1 / cv^2`): under a flat prior no
/// log-linear mean exists, under an informative one the posterior of the
/// origin's ultimate is a gamma of shape `shape + sum / phi <= 0`, improper.
/// A finite `cv` below `sqrt(-phi / sum)` makes it proper.
fn origin_error(origin: &crate::Period, n: usize, sum: f64, shape: f64, phi: f64) -> Error {
    let cells = n_cells(n);
    let cause = if shape == 0.0 {
        "so no log-linear mean exists under a flat prior (an infinite cv): the model's mean \
         for an origin is its ultimate times positive shares, which cannot fit a row that \
         sums to zero or less"
            .to_string()
    } else {
        format!(
            "at most -phi / cv^2 = {}, so the posterior of its ultimate is improper",
            -shape * phi
        )
    };
    let cv = if sum < 0.0 {
        format!(
            "a finite cv below sqrt(-phi / sum) = {}",
            (-phi / sum).sqrt()
        )
    } else {
        "a finite cv".to_string()
    };
    Error::Bayes(format!(
        "origin {origin}: the observed increments ({cells}) sum to {sum}, {cause}. The chain \
         ladder accepts this triangle. Give the prior {cv}, or rebuild the triangle from its \
         long table without this origin's rows"
    ))
}

/// The error for the development column at position `d` of `ages` whose `n`
/// observed increments sum to `sum`, with the chain ladder's factor into
/// that age (`None` at the first age) and the origins last observed there:
/// the mean `U_w p_d` is positive, so the likelihood increases as the share
/// falls to 0 and has no finite maximum (design note `reserving-bayes.md`,
/// decision 26). The remedy depends on where the age is: dropping the last
/// age with `Triangle::max_age` leaves its movement out of the model (a
/// tail), dropping the first removes the origins observed only there, and
/// a middle age cannot be dropped, since `max_age` keeps every age up to
/// one, `Triangle::from_long` fills every age from the first to the last
/// and the chain ladder then refuses the empty column.
fn column_error(
    ages: &[crate::Lag],
    d: usize,
    n: usize,
    sum: f64,
    factor: Option<f64>,
    last_here: &[String],
    alone: bool,
) -> Error {
    let age = ages[d];
    let cause = format!(
        "development age {age} months: the observed increments ({}) sum to {sum}, so no \
         log-linear mean exists: the model's mean at an age is an ultimate times a positive \
         share, which cannot fit a column that sums to zero or less",
        n_cells(n)
    );
    let origins = match last_here {
        [] => String::new(),
        [one] => format!("origin {one}"),
        many => format!("origins {}", many.join(", ")),
    };
    let accepts = |previous: crate::Lag| {
        let factor = factor
            .filter(|f| f.is_finite())
            .map(|f| format!(" (its factor from {previous} to {age} months is {f:.4})"))
            .unwrap_or_default();
        format!("The chain ladder accepts this triangle{factor}")
    };
    let remedy = match (d.checked_sub(1).map(|k| ages[k]), ages.get(d + 1)) {
        (Some(previous), None) => format!(
            "{}. Keep the ages up to {previous} months with the triangle's max_age({previous}), \
             which leaves out the rows at {age} months{}, or fold that age into {previous} months (book each origin's value at {age} months \
             at {previous} months), and carry the movement left out as a tail outside this model",
            accepts(previous),
            super::odp::every_segment(previous, alone)
        ),
        (Some(previous), Some(_)) => format!(
            "{}, and so does Mack's model. This model cannot take it: no change to the triangle \
             removes an age between the first and the last, since the development ages run \
             without a gap from the first to the last, so without the rows at {age} months the \
             column comes back with no cells and the chain ladder refuses it too, and the \
             triangle's max_age keeps every age up to one, so it cannot leave out a middle age \
             alone (design note reserving-bayes.md, decisions 18 and 26)",
            accepts(previous)
        ),
        (None, Some(next)) => {
            let removed = if last_here.is_empty() {
                String::new()
            } else {
                let its = if last_here.len() == 1 { "its" } else { "their" };
                format!(
                    ". This also removes {origins}, observed only at {age} months: estimate \
                     {its} reserve outside this model, for example with the chain ladder"
                )
            };
            format!(
                "The chain ladder accepts this triangle. The first age has no earlier one to fold \
                 into: rebuild the triangle from its long table without the rows at {age} months \
                 (in an incremental triangle, first add each origin's value there to its value at \
                 {next} months), so that development starts at {next} months{removed}"
            )
        }
        (None, None) => "The chain ladder accepts this triangle".to_string(),
    };
    Error::Bayes(format!("{cause}. {remedy}"))
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
fn log_sum(n_origins: usize, n_dev: usize, theta: &[f64]) -> f64 {
    let rest = &theta[n_origins..n_origins + n_dev - 1];
    let top = rest.iter().fold(0.0_f64, |m, &b| m.max(b));
    top + ((-top).exp() + rest.iter().map(|b| (b - top).exp()).sum::<f64>()).ln()
}

/// `log p_d` of every age from a draw of `(alpha, beta)`.
fn log_pattern(n_origins: usize, n_dev: usize, theta: &[f64]) -> Vec<f64> {
    let l = log_sum(n_origins, n_dev, theta);
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
fn natural_eta(n_origins: usize, n_dev: usize, theta: &[f64], w: usize, d: usize) -> f64 {
    let beta = if d == 0 {
        0.0
    } else {
        theta[n_origins + d - 1]
    };
    theta[w] + beta - log_sum(n_origins, n_dev, theta)
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
        natural_eta(self.model.n_origins, self.model.n_dev, theta, w, d)
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
        self.n_origins + self.n_dev - 1 + usize::from(self.common_shape.is_some())
    }

    fn log_posterior(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        let (no, nd) = (self.n_origins, self.n_dev);
        let l = self.log_sum(theta);
        let mut f = 0.0;
        // Gamma(shape, rate shape / m) on U = exp(alpha), with the
        // Jacobian: shape alpha - shape exp(alpha) / m, here less its value
        // at alpha = log m, in u = alpha - log m (as BayesBf). With a
        // common multiplier s = exp(t) the mean is m s, so the term is in
        // v = u - t (t = 0 without one: v = u exactly), and t has the
        // Gamma(common shape, mean 1) prior in log s, k (t - expm1 t).
        let t = self.t(theta);
        let mut t_grad = 0.0;
        for w in 0..no {
            let v = theta[w] - t;
            let em1 = v.exp_m1();
            f += self.shape * (v - em1);
            grad[w] = -self.shape * em1;
            t_grad += self.shape * em1;
        }
        for j in no..no + nd - 1 {
            f -= 0.5 * self.development_precision * theta[j] * theta[j];
            grad[j] = -self.development_precision * theta[j];
        }
        if let (Some(k), Some(j)) = (self.common_shape, self.t_index()) {
            f += k * (t - t.exp_m1());
            grad[j] = t_grad - k * t.exp_m1();
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
        let t = self.t(theta);
        let mut t_curvature = 0.0;
        for w in 0..no {
            let curvature = self.shape * (theta[w] - t).exp();
            h[w * p + w] = curvature;
            t_curvature += curvature;
        }
        for j in no..no + nd - 1 {
            h[j * p + j] = self.development_precision;
        }
        if let (Some(k), Some(j)) = (self.common_shape, self.t_index()) {
            // v = u - t: d2/du2 = c, d2/du dt = -c, d2/dt2 = sum c.
            h[j * p + j] = t_curvature + k * t.exp();
            for w in 0..no {
                let cross = -self.shape * (theta[w] - t).exp();
                h[w * p + j] = cross;
                h[j * p + w] = cross;
            }
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
        let t = match self.t_index() {
            Some(j) => theta[j],
            None => S::from_f64(0.0),
        };
        for &u in &theta[..no] {
            let v = u - t;
            f += (v - v.exp_m1()) * self.shape;
        }
        if let Some(k) = self.common_shape {
            f += (t - t.exp_m1()) * k;
        }
        let mut sum = S::from_f64(1.0);
        for &b in &theta[no..no + self.n_dev - 1] {
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
    use crate::{BornhuetterFerguson, ChainLadder, Lag, Mack};
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
        for (cv, common_cv) in [(0.2, None), (f64::INFINITY, None), (0.2, Some(0.5))] {
            let model = BayesBfNormalised {
                cv,
                common_cv,
                ..quick(0)
            };
            let (_, model, start) = prepared(&genins(), &model);
            assert_eq!(model.dim(), 10 + 9 + usize::from(common_cv.is_some()));
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
        for (label, cv, common_cv) in [
            ("BayesBfNormalised cv 0.2", 0.2, None),
            ("BayesBfNormalised flat", f64::INFINITY, None),
            ("BayesBfNormalised common 0.5", 0.2, Some(0.5)),
        ] {
            let model = BayesBfNormalised {
                cv,
                common_cv,
                ..quick(0)
            };
            let (_, model, start) = prepared(&genins(), &model);
            let mode = mode(&model, start).unwrap();
            let whitened = Whitened::new(&model, &mode).unwrap();
            let points = posterior_points(&whitened, model.dim(), 20);
            assert_oracle(label, &model, &points);
        }
    }

    /// The prior part of the log posterior with a common multiplier is the
    /// hierarchical gamma law: `s ~ Gamma(k, mean 1)` and `U_w | s ~
    /// Gamma(a, mean m_w s)`, each on the log scale of its variable. Two
    /// points' difference removes the constants.
    #[test]
    fn the_common_prior_is_the_hierarchical_gamma() {
        use prospicio_math::special::ln_gamma;
        let (cv, c) = (0.3_f64, 0.6_f64);
        let model = BayesBfNormalised {
            cv,
            common_cv: Some(c),
            ..quick(0)
        };
        let (_, mut model, _) = prepared(&genins(), &model);
        model.cells.clear();
        let (a, k) = (cv.powi(-2), c.powi(-2));
        // log density of (alpha, t): the Gamma(k, k) density of s on log s,
        // and the Gamma(a, a / (m s)) density of U on log U.
        let reference = |alpha: &[f64], t: f64| {
            let mut f = k * t - k * t.exp() + k * k.ln() - ln_gamma(k);
            for (w, al) in alpha.iter().enumerate() {
                let v = al - model.log_prior[w] - t;
                f += a * v - a * v.exp() + a * a.ln() - ln_gamma(a);
            }
            f
        };
        let no = model.n_origins;
        let beta = vec![0.0; model.n_dev - 1];
        let point = |u: &[f64], t: f64| {
            let mut theta = u.to_vec();
            theta.extend_from_slice(&beta);
            theta.push(t);
            theta
        };
        let mut grad = vec![0.0; model.dim()];
        let (u0, t0) = (vec![0.0; no], 0.0);
        let u1: Vec<f64> = (0..no).map(|w| 0.3 - 0.07 * w as f64).collect();
        let t1 = 0.4;
        let alpha = |u: &[f64]| -> Vec<f64> {
            u.iter().zip(&model.log_prior).map(|(u, l)| u + l).collect()
        };
        let model_diff = model.log_posterior(&point(&u1, t1), &mut grad)
            - model.log_posterior(&point(&u0, t0), &mut grad);
        let ref_diff = reference(&alpha(&u1), t1) - reference(&alpha(&u0), t0);
        assert!(
            (model_diff - ref_diff).abs() < 1e-9 * (1.0 + ref_diff.abs()),
            "{model_diff} vs {ref_diff}"
        );
        // Without the multiplier it is the independent gamma of PR 4.
        let plain = BayesBfNormalised { cv, ..quick(0) };
        let (_, mut plain, _) = prepared(&genins(), &plain);
        plain.cells.clear();
        let mut theta = u1.clone();
        theta.extend_from_slice(&beta);
        let mut g = vec![0.0; plain.dim()];
        let mut theta0 = u0.clone();
        theta0.extend_from_slice(&beta);
        let plain_diff = plain.log_posterior(&theta, &mut g) - plain.log_posterior(&theta0, &mut g);
        let want: f64 = u1
            .iter()
            .zip(&u0)
            .map(|(a1, a0)| {
                let f = |v: f64| a * v - a * v.exp();
                f(*a1) - f(*a0)
            })
            .sum();
        assert!((plain_diff - want).abs() < 1e-9 * (1.0 + want.abs()));
    }

    /// An invalid combination is an error, not a silent fallback.
    #[test]
    fn a_common_multiplier_needs_a_proper_prior() {
        for (cv, common_cv) in [
            (f64::INFINITY, Some(0.3)),
            (0.2, Some(0.0)),
            (0.2, Some(-0.1)),
            (0.2, Some(f64::INFINITY)),
            (0.2, Some(f64::NAN)),
        ] {
            let model = BayesBfNormalised {
                cv,
                common_cv,
                ..quick(0)
            };
            let e = model.fit(&genins(), "values", "premium").unwrap_err();
            assert!(matches!(e, Error::InvalidSetting { .. }), "{e}");
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
            apriori: 1.0.into(),
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
            apriori: 1.0.into(),
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
            matches!(&e, Error::Bayes(m) if m.starts_with("origin 2023") && m.contains("improper")
                && m.contains("(1 cell) sum to -1")
                && m.contains("without this origin's rows")),
            "{e}"
        );
        // The remedy it names: a cv below the bound it prints.
        let Error::Bayes(m) = &e else { unreachable!() };
        let bound: f64 = m
            .split("sqrt(-phi / sum) = ")
            .nth(1)
            .and_then(|rest| rest.split(',').next())
            .and_then(|b| b.parse().ok())
            .unwrap_or_else(|| panic!("no cv bound in {m}"));
        let proper = BayesBfNormalised {
            cv: 0.9 * bound,
            ..quick(0)
        };
        prepared(&negative, &proper);
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

    /// A triangle rebuilt from its long table without the rows at `age`.
    fn without_age(tri: &Triangle, age: Lag) -> Triangle {
        let long = tri.to_long();
        let keep: Vec<usize> = (0..long.development.len())
            .filter(|&i| long.development[i] != age)
            .collect();
        let origin: Vec<Month> = keep.iter().map(|&i| long.origin[i]).collect();
        let development: Vec<Lag> = keep.iter().map(|&i| long.development[i]).collect();
        let values: Vec<(&str, Vec<f64>)> = long
            .values
            .iter()
            .map(|(name, v)| (name.as_str(), keep.iter().map(|&i| v[i]).collect()))
            .collect();
        let values: Vec<(&str, &[f64])> = values.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&development),
            values: &values,
            origin_grain: tri.origin_grain(),
            development_grain: tri.development_grain(),
            cumulative: tri.is_cumulative(),
        })
        .unwrap()
    }

    /// Four origins whose oldest age, 48 months, has one cell 5 below the
    /// one at 36 months (issue #610).
    fn falling() -> Triangle {
        with_premium(
            2020,
            &[
                &[100.0, 150.0, 160.0, 155.0],
                &[110.0, 170.0, 180.0],
                &[120.0, 175.0],
                &[130.0],
            ],
            &[200.0; 4],
        )
    }

    /// The oldest age's one cell falls: the chain ladder takes the
    /// triangle, the model refuses it and says which age and what to do,
    /// and the triangle without that age prepares.
    #[test]
    fn a_falling_late_column_names_its_age_and_the_remedy() {
        let falling = falling();
        let chain_ladder = ChainLadder::default().fit(&falling, "values").unwrap();
        let factor = chain_ladder.development.ldf[2];
        assert!((factor - 155.0 / 160.0).abs() < 1e-12);
        let e = quick(0).fit(&falling, "values", "premium").unwrap_err();
        let Error::Bayes(m) = &e else { panic!("{e}") };
        for part in [
            "development age 48 months: the observed increments (1 cell) sum to -5,",
            "no log-linear mean exists",
            "The chain ladder accepts this triangle (its factor from 36 to 48 months is 0.9688)",
            "Keep the ages up to 36 months with the triangle's max_age(36), which leaves out the \
             rows at 48 months",
            "fold that age into 36 months",
            "as a tail outside this model",
        ] {
            assert!(m.contains(part), "{part:?} not in {m}");
        }
        // BayesOdp and BayesBf refuse the same column and name the same
        // remedy, which they accept.
        for e in [
            BayesOdp::default().fit(&falling, "values").unwrap_err(),
            BayesBf::default()
                .fit(&falling, "values", "premium")
                .unwrap_err(),
        ] {
            let Error::Bayes(m) = &e else { panic!("{e}") };
            for part in [
                "development age 48 months: the observed increments sum to -5,",
                "Keep the ages up to 36 months with the triangle's max_age(36)",
                "as a tail outside this model",
            ] {
                assert!(m.contains(part), "{part:?} not in {m}");
            }
            // On a triangle of one segment there are no others to warn of.
            assert!(!m.contains("cuts every segment"), "{m}");
        }
        let segment = |t: &Triangle| t.segment("values").unwrap();
        assert!(
            BayesOdp::default()
                .prepare(&segment(&falling), true)
                .is_err()
        );
        assert!(
            BayesOdp::default()
                .prepare(&segment(&falling.max_age(36).unwrap()), true)
                .is_ok()
        );
        // The first age has nothing to fold into.
        let first = with_premium(
            2020,
            &[
                &[-10.0, 50.0, 60.0, 70.0],
                &[-20.0, 40.0, 55.0],
                &[5.0, 30.0],
                &[8.0],
            ],
            &[200.0; 4],
        );
        let e = quick(0).fit(&first, "values", "premium").unwrap_err();
        let Error::Bayes(m) = &e else { panic!("{e}") };
        for part in [
            "development age 12 months: the observed increments (4 cells) sum to -17,",
            "development starts at 24 months. This also removes origin 2023, observed only at \
             12 months: estimate its reserve outside this model",
        ] {
            assert!(m.contains(part), "{part:?} not in {m}");
        }
        assert!(!m.contains("tail"), "{m}");
        // Its remedy keeps the origins with a second age and drops 2023.
        let started = without_age(&first, 12);
        assert_eq!(started.development(), &[24, 36, 48]);
        let (fit, model, start) = prepared(&started, &quick(0));
        assert_eq!(fit.latest, vec![70.0, 55.0, 30.0]);
        assert_eq!(model.n_origins, 3);
        assert_eq!(model.n_dev, 3);
        assert!(start.iter().all(|t| t.is_finite()));

        // The remedy: the triangle without the rows at 48 months, kept by
        // max_age as the rebuild from the long table keeps it.
        let dropped = falling.max_age(36).unwrap();
        assert_eq!(dropped, without_age(&falling, 48));
        assert_eq!(dropped.development(), &[12, 24, 36]);
        let (fit, model, start) = prepared(&dropped, &quick(0));
        assert_eq!(fit.latest, vec![160.0, 180.0, 175.0, 130.0]);
        assert_eq!(model.n_dev, 3);
        assert!(start.iter().all(|t| t.is_finite()));
    }

    /// On a triangle of several segments the `max_age` remedy says that it
    /// cuts every segment, since a segment trimmed to its own ages can end
    /// before another one that develops further.
    #[test]
    fn the_last_age_remedy_warns_that_max_age_cuts_every_segment() {
        let rows: [&[f64]; 4] = [
            &[100.0, 150.0, 160.0, 155.0],
            &[110.0, 170.0, 180.0],
            &[120.0, 175.0],
            &[130.0],
        ];
        let (mut keys, mut origin, mut ages, mut values) = (vec![], vec![], vec![], vec![]);
        for (lob, extra) in [("Auto", 0.0), ("Home", 20.0)] {
            for (k, row) in rows.iter().enumerate() {
                for (d, &v) in row.iter().enumerate() {
                    keys.push(lob);
                    origin.push(Month::january(2020 + k as i32));
                    ages.push(12 * (d as Lag + 1));
                    values.push(v + extra * d as f64);
                }
            }
        }
        let tri = Triangle::from_long(&Long {
            keys: &[("lob", &keys)],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("values", &values)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let e = BayesOdp::default()
            .fit_segments(&tri, "values")
            .unwrap_err();
        let m = e.to_string();
        for part in [
            "development age 48 months: the observed increments sum to -5,",
            "Keep the ages up to 36 months with the triangle's max_age(36), which leaves out the \
             rows at 48 months (max_age cuts every segment of the triangle at 36 months, so \
             select this segment first if another one develops further)",
        ] {
            assert!(m.contains(part), "{part:?} not in {m}");
        }
        // The normalised BF's message has the same clause.
        let m = column_error(&[12, 24, 36, 48], 3, 1, -5.0, None, &[], false).to_string();
        assert!(
            m.contains("rows at 48 months (max_age cuts every segment of the triangle at 36"),
            "{m}"
        );
    }

    /// A middle age whose column falls: the message names no tail and no
    /// rebuild, since the triangle without that age's rows keeps the age
    /// with no cells and the chain ladder refuses it, and `max_age` keeps
    /// every age up to one (decision 18).
    #[test]
    fn a_falling_middle_column_has_no_rebuild_remedy() {
        let falling = with_premium(
            2020,
            &[
                &[100.0, 150.0, 140.0, 170.0],
                &[110.0, 170.0, 165.0],
                &[120.0, 175.0],
                &[130.0],
            ],
            &[200.0; 4],
        );
        assert!(ChainLadder::default().fit(&falling, "values").is_ok());
        assert!(Mack::default().fit(&falling, "values").is_ok());
        let e = quick(0).fit(&falling, "values", "premium").unwrap_err();
        let Error::Bayes(m) = &e else { panic!("{e}") };
        for part in [
            "development age 36 months: the observed increments (2 cells) sum to -15,",
            "The chain ladder accepts this triangle (its factor from 24 to 36 months is \
             0.9531), and so does Mack's model",
            "no change to the triangle removes an age between the first and the last",
            "max_age keeps every age up to one, so it cannot leave out a middle age alone",
        ] {
            assert!(m.contains(part), "{part:?} not in {m}");
        }
        for wrong in ["tail", "fold"] {
            assert!(!m.contains(wrong), "{wrong:?} in {m}");
        }
        // Why: without the rows at 36 months the age comes back empty and
        // the chain ladder has no factor from 24 to 36 months.
        let dropped = without_age(&falling, 36);
        assert_eq!(dropped.development(), &[12, 24, 36, 48]);
        assert!(matches!(
            ChainLadder::default().fit(&dropped, "values"),
            Err(Error::Factor { age: 1, .. })
        ));
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_triangle_without_its_falling_age_fits() {
        let fit = quick(0)
            .fit(&falling().max_age(36).unwrap(), "values", "premium")
            .unwrap();
        assert_eq!(fit.chain_ladder.latest, vec![160.0, 180.0, 175.0, 130.0]);
        let total = fit.reserves.total().mean();
        assert!(total.is_finite() && total > 0.0, "{total}");
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
        // The posterior mean ultimate, latest plus the mean reserve, and it
        // over the exposure, per origin and in total.
        let latest = long.column("latest").unwrap();
        let mean = long.column("mean").unwrap();
        let ultimate_mean: Vec<f64> = latest.iter().zip(mean).map(|(l, m)| l + m).collect();
        assert_eq!(long.column("ultimate_mean").unwrap(), ultimate_mean);
        let ulr_mean: Vec<f64> = (ultimate_mean.iter())
            .zip(&fit.exposure)
            .map(|(u, e)| u / e)
            .collect();
        assert_eq!(long.column("ulr_mean").unwrap(), ulr_mean);
        // Independently of the table: the triangle's latest diagonal plus
        // each origin's posterior mean reserve, and in total the sum of the
        // latest values plus the mean of the total reserve.
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
        let diagonal: Vec<f64> = GENINS.iter().map(|row| *row.last().unwrap()).collect();
        for (k, (d, u)) in diagonal
            .iter()
            .zip(long.column("ultimate_mean").unwrap())
            .enumerate()
        {
            let origin = vec![Period::year(2001 + k as i32).into()];
            let reserve = fit.reserves.marginal(&origin).unwrap().mean();
            assert!(close(*u, d + reserve), "origin {k}: {u} vs {}", d + reserve);
        }
        assert!(long.column("ultimate_std_dev").is_none());
        assert_eq!(long.values.last().unwrap().0, "ulr_mean");
        let totals = fit.totals();
        assert_eq!(totals.column("scale").unwrap(), [fit.scale]);
        assert_eq!(totals.column("exposure").unwrap(), [118e6]);
        let total_mean = totals.column("latest").unwrap()[0] + totals.column("mean").unwrap()[0];
        assert_eq!(totals.column("ultimate_mean").unwrap(), [total_mean]);
        assert_eq!(totals.column("ulr_mean").unwrap(), [total_mean / 118e6]);
        let independent = diagonal.iter().sum::<f64>() + fit.reserves.total().mean();
        assert!(
            close(total_mean, independent),
            "{total_mean} vs {independent}"
        );
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
        for (name, cv, common_cv) in [
            ("BayesBfNormalised cv 0.2", 0.2, None),
            ("BayesBfNormalised flat", f64::INFINITY, None),
            ("BayesBfNormalised common 0.5", 0.2, Some(0.5)),
        ] {
            let model = BayesBfNormalised {
                cv,
                common_cv,
                ..quick(6)
            };
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

    /// The multiplier is a parameter of the fit, named `log_share` and
    /// last, and a vanishing cv for it is the plain model: the reserves
    /// agree within Monte Carlo error and `s` stays at 1.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_vanishing_common_multiplier_is_the_plain_model() {
        let plain = quick(8).fit(&genins(), "values", "premium").unwrap();
        let common = BayesBfNormalised {
            common_cv: Some(1e-4),
            ..quick(8)
        }
        .fit(&genins(), "values", "premium")
        .unwrap();
        assert_eq!(common.posterior.dim(), plain.posterior.dim() + 1);
        assert_eq!(common.posterior.names().last().unwrap(), "log_share");
        let t = common.posterior.mean()[common.posterior.dim() - 1];
        assert!(t.abs() < 1e-3, "{t}");
        let (a, b) = (plain.reserves.total(), common.reserves.total());
        assert!(
            (a.mean() / b.mean() - 1.0).abs() < 0.02,
            "{} {}",
            a.mean(),
            b.mean()
        );
        assert!((a.std_dev() / b.std_dev() - 1.0).abs() < 0.08);
        assert!(common.diagnostics.warnings.is_empty());
        // Provenance names the multiplier only when there is one.
        let has = |fit: &BayesBfNormalisedFit| {
            fit.reserves
                .provenance()
                .parameters
                .iter()
                .any(|(k, _)| k == "common_cv")
        };
        assert!(has(&common) && !has(&plain));
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
