//! The Bayesian over-dispersed Poisson chain ladder
//! (`docs/design/reserving-bayes.md`, decisions 3 to 6), its segments
//! (decision 11) and its place in the diagonal backtest (decision 8).

use prospicio_bayes::elpd::Loo;
use prospicio_bayes::nuts::{ParamSummary, PosteriorDraws, Sampler, sample};
use prospicio_core::StreamRng;
use prospicio_models::Family;
use prospicio_prob::{InputHasher, PredictiveDistribution, Provenance};
use rayon::prelude::*;

use super::laplace::{Posterior, Whitened, mode};
use super::one_year::{
    self, BayesOneYear, BayesOneYearFit, BayesOneYearFits, BayesOneYearSegment, OdpCells, OdpYear,
    YearDraws,
};
use super::{Diagnostics, Stream, derive_seed, one_year_seed};
use crate::backtest::{CellForecast, TriangleModel};
use crate::bootstrap::{
    ProcessDistribution, component_sums, hash_segment, origin_keys, pick_segment, prepare,
    process_draw, push_moments, segment_sums,
};
use crate::chain_ladder::ChainLadderFit;
use crate::dependence::SegmentDependence;
use crate::error::{Error, Result};
use crate::frame::TriangleFrame;
use crate::segments::{FitTable, ReserveFit, SegmentFits, fit_each};
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
    /// ([`fit_segments`](Self::fit_segments), decision 11; `Synchronized` is
    /// an error there); [`fit`](Self::fit) has one segment and does
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

/// One segment of a [`BayesOdpFits`]: the posterior of the segment's own
/// [`BayesOdp`] model. Its reserves are in the joint
/// [`BayesOdpFits::reserves`].
#[derive(Debug, Clone)]
pub struct BayesOdpSegment {
    /// The segment's volume-weighted chain ladder, as in
    /// [`BayesOdpFit::chain_ladder`].
    pub chain_ladder: ChainLadderFit,
    /// The segment's plugged-in scale `phi`.
    pub scale: f64,
    /// Posterior draws of the segment's `c`, `a[2]..a[n]` and `b[2]..b[m]`.
    pub posterior: PosteriorDraws,
    /// R-hat, ESS and divergences of the segment's chains.
    pub diagnostics: Diagnostics,
    /// The model, for [`loo`](Self::loo).
    model: Model,
}

impl BayesOdpSegment {
    /// Posterior summary of each parameter, as [`BayesOdpFit::summary`].
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        Ok(self.posterior.summary()?)
    }

    /// PSIS-LOO of the segment's observed increments, as
    /// [`BayesOdpFit::loo`].
    pub fn loo(&self) -> Result<Loo> {
        loo(&self.model, &self.posterior, &self.chain_ladder)
    }
}

impl ReserveFit for BayesOdpSegment {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("scale", self.scale)]
    }
}

/// A [`BayesOdp`] fit of every segment of a triangle column
/// ([`BayesOdp::fit_segments`]): each segment's own posterior, and the
/// joint distribution of the reserves, with the layout of
/// [`OdpBootstrapFits`](crate::OdpBootstrapFits).
///
/// ```
/// use prospicio_bayes::nuts::Sampler;
/// use prospicio_reserving::bayes::BayesOdp;
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
///
/// let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
/// let ages = [12, 24, 36, 48, 12, 24, 36, 12, 24, 12];
/// let paid = [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0];
/// let tri = Triangle::from_long(&Long {
///     keys: &[("lob", &[["Auto"; 10], ["Home"; 10]].concat())],
///     origin: &[origin, origin].concat(),
///     development: DevelopmentColumn::Age(&[ages, ages].concat()),
///     values: &[("paid", &[paid, paid.map(|v| v * 2.0)].concat())],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let model = BayesOdp {
///     sampler: Sampler { tune: 300, draws: 300, ..Sampler::default() },
///     n_sims: 1_000,
///     ..BayesOdp::default()
/// };
/// let fits = model.fit_segments(&tri, "paid")?;
/// assert_eq!(fits.reserves.dims(), ["lob", "origin"]);
/// assert_eq!(fits.reserves.n_components(), 8);
/// assert_eq!(fits.segments.len(), 2);
/// assert_eq!(fits.totals().n_rows(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct BayesOdpFits {
    /// Each segment's chain ladder, scale, posterior and diagnostics.
    pub segments: SegmentFits<BayesOdpSegment>,
    /// Joint posterior predictive distribution of the reserve by segment
    /// and origin: the key names and `origin` are its dimensions, and its
    /// components run over the origins of each segment in turn, like the
    /// rows of [`to_long`](Self::to_long).
    pub reserves: PredictiveDistribution,
}

impl BayesOdpFits {
    /// One row per segment × origin: the chain ladder's `latest`,
    /// `ultimate` and `reserve`, and the `mean` and `std_dev` of the
    /// posterior predictive reserve.
    pub fn to_long(&self) -> FitTable {
        let mut table = self.segments.to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        table
    }

    /// One row per segment: the chain ladder's totals, the `scale`, and the
    /// `mean` and `std_dev` of the segment's posterior predictive total
    /// reserve.
    pub fn totals(&self) -> FitTable {
        let mut table = self.segments.totals();
        push_moments(
            &mut table,
            "",
            &segment_sums(&self.segments, &self.reserves),
        );
        table
    }

    /// The chain ladders' development factors, one row per segment × age.
    pub fn development_table(&self) -> FitTable {
        self.segments.development_table()
    }

    /// The one segment chosen as in [`SegmentFits::position`], with its
    /// part of the joint reserves (same dimensions).
    pub fn segment(&self, keys: &[(&str, &str)]) -> Result<Self> {
        let (segments, reserves) = pick_segment(&self.segments, &self.reserves, keys)?;
        Ok(Self { segments, reserves })
    }
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
        let (fit, reserves) =
            self.fit_segment(&segment, column, Stream::Chains, Stream::Predictive)?;
        let BayesOdpSegment {
            chain_ladder,
            scale,
            posterior,
            diagnostics,
            model,
        } = fit;
        Ok(BayesOdpFit {
            chain_ladder,
            scale,
            posterior,
            diagnostics,
            reserves,
            model,
        })
    }

    /// Fits `column` in every segment of a cumulative triangle, each with
    /// its own posterior (the same model and settings, sampled
    /// independently), into one joint distribution of the reserves, the
    /// segments depending on each other as
    /// [`dependence`](Self::dependence) says (decision 11):
    ///
    /// * [`Independent`](SegmentDependence::Independent): each segment's
    ///   simulations as drawn. Segment `k` (in index order) takes its own
    ///   derived seeds for its chains and its simulations (module docs,
    ///   "Seeds"), so the segments share no random numbers; a segment's
    ///   draws therefore differ from fitting it alone.
    /// * [`RankCorrelation`](SegmentDependence::RankCorrelation): the
    ///   segments' simulations reordered as whole rows by Iman–Conover on
    ///   the segment totals, exactly as in the bootstraps, so row `i` is no
    ///   longer simulation `i`.
    /// * [`Synchronized`](SegmentDependence::Synchronized) is an
    ///   [`Error::Bayes`]: it draws the residuals of the same positions in
    ///   every segment, and a posterior has no residuals to share.
    ///
    /// A failure names its segment.
    pub fn fit_segments(&self, triangle: &Triangle, column: &str) -> Result<BayesOdpFits> {
        self.validate()?;
        // Synchronized and a bad rank-correlation matrix fail here, before
        // any sampling.
        self.check_dependence(triangle)?;
        let mut k = 0;
        let fitted = fit_each(triangle, column, |segment| {
            let fit = self.fit_segment(
                segment,
                column,
                Stream::SegmentChains(k),
                Stream::SegmentPredictive(k),
            );
            k += 1;
            fit
        })?;

        let mut hasher = InputHasher::new();
        hasher.str(column);
        let mut dims = fitted.key_names.clone();
        dims.push("origin".into());
        let mut components = Vec::new();
        for ((label, (fit, _)), segment) in fitted.iter().zip(triangle.segments(column)?) {
            hasher.str(&label.to_string());
            hash_segment(&mut hasher, &segment, &fit.chain_ladder, &segment.ages);
            components.extend(origin_keys(label, &segment.origins));
        }
        let n = components.len();
        let mut draws = Vec::with_capacity(self.n_sims * n);
        for i in 0..self.n_sims {
            for (_, reserves) in &fitted.fits {
                let m = reserves.n_components();
                draws.extend_from_slice(&reserves.draw_matrix()[i * m..(i + 1) * m]);
            }
        }
        let list = |f: fn(&BayesOdpSegment) -> f64| {
            format!(
                "{:?}",
                fitted.fits.iter().map(|(s, _)| f(s)).collect::<Vec<_>>()
            )
        };
        let provenance = self
            .base_provenance(column)
            .param("c0", list(|s| s.model.c0))
            .param("scale", list(|s| s.scale))
            .param("segments", fitted.len())
            .param("dependence", format!("{:?}", self.dependence))
            .param(
                "posterior_draws",
                fitted
                    .fits
                    .first()
                    .map_or(0, |(s, _)| s.posterior.n_draws()),
            )
            .input_hash(hasher.finish());
        let reserves = PredictiveDistribution::from_draws(dims, components, draws, provenance)?;
        let segments = fitted.map(|(fit, _)| fit.clone());
        let reserves = self.dependence.reorder(reserves, &segments, self.seed)?;
        Ok(BayesOdpFits { segments, reserves })
    }

    /// The one-year view of `column` of a single-segment cumulative
    /// triangle (decision 16): the claims development result over the
    /// twelve months after the valuation, by reweighting the posterior with
    /// Pareto smoothed importance sampling.
    ///
    /// The posterior is [`fit`](Self::fit)'s, draw for draw. Each of
    /// [`n_sims`](Self::n_sims) simulations picks a posterior draw `θ`,
    /// draws the increments `X` of the year's cells (the cells valued in the
    /// twelve months, as in [`OdpBootstrap::one_year`](crate::OdpBootstrap::one_year);
    /// one per open origin with an annual grain) with
    /// [`process`](Self::process) error, weights every posterior draw `θ_s`
    /// by `exp(Σ (X η_s - exp η_s) / φ)`, the quasi likelihood of the year
    /// with `φ` and the priors unchanged, and smooths the weights (Pareto
    /// `k̂` per simulation). Origin `w`'s claims development result is
    ///
    /// ```text
    /// CDR_w = E[R_w | D0] - (X_w + E[R'_w | D0, X])
    /// ```
    ///
    /// with `R_w` the sum of its future increments' means `exp(η_wd)` at
    /// the opening, `X_w` its increments in the year and `R'_w` the means of
    /// the increments after the year, each expectation the (weighted) mean
    /// over the posterior draws. Its mean is zero up to Monte Carlo and
    /// importance sampling error. A simulation whose `k̂` is above
    /// [`BayesOneYear::k_threshold`] is handled as [`BayesOneYear::refit`]
    /// says: by default moment-matched (decision 18) and, only if that
    /// cannot bring its `k̂` to the threshold, refitted with NUTS on the
    /// triangle extended by its year, with this model's sampler and a seed
    /// drawn from the simulation's own stream. An origin with every remaining cell in the year has its whole
    /// run-off as its one-year view; one at the last age has none. A
    /// triangle with no cells in the year (every origin at the last age)
    /// leaves the posterior unchanged: `k̂` is 0 and nothing is refitted.
    ///
    /// Unlike the ODP bootstrap, which leaves the cells of an origin that
    /// lags the valuation unobserved before the year, every cell up to the
    /// end of the year is observed and conditioned on here.
    ///
    /// ```
    /// use prospicio_bayes::nuts::Sampler;
    /// use prospicio_reserving::bayes::{BayesOdp, BayesOneYear};
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
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
    ///     n_sims: 500,
    ///     ..BayesOdp::default()
    /// };
    /// let view = model.one_year(&tri, "paid", &BayesOneYear::default())?;
    /// assert_eq!(view.cdr.n_components(), 4);
    /// // 2020 is fully developed: no cell in the year, no change.
    /// assert!(view.cdr.draw_matrix().chunks(4).all(|row| row[0] == 0.0));
    /// assert_eq!(view.pareto_k.len(), 500);
    /// // The one-year view is narrower than the lifetime view.
    /// assert!(view.cdr.std_dev() < view.fit.reserves.std_dev());
    /// # Ok::<(), prospicio_reserving::Error>(())
    /// ```
    pub fn one_year(
        &self,
        triangle: &Triangle,
        column: &str,
        settings: &BayesOneYear,
    ) -> Result<BayesOneYearFit> {
        self.validate()?;
        settings.validate()?;
        let segment = triangle.segment(column)?;
        let (fit, reserves) =
            self.fit_segment(&segment, column, Stream::Chains, Stream::Predictive)?;
        let year = self.year_draws(triangle, &segment, &fit, Stream::Predictive, settings)?;
        let mut hasher = InputHasher::new();
        hasher.str(column);
        hash_segment(&mut hasher, &segment, &fit.chain_ladder, &segment.ages);
        let provenance = self
            .one_year_provenance(column, &fit.model, settings)
            .input_hash(hasher.finish());
        let BayesOdpSegment {
            chain_ladder,
            scale,
            posterior,
            diagnostics,
            model,
        } = fit;
        let fit = BayesOdpFit {
            chain_ladder,
            scale,
            posterior,
            diagnostics,
            reserves,
            model,
        };
        one_year::assemble(fit, year, &segment, settings, provenance)
    }

    /// The one-year view of `column` in every segment of a cumulative
    /// triangle, each segment's posterior and simulations as in
    /// [`fit_segments`](Self::fit_segments) (segment `k`'s own seeds), into
    /// one joint claims development result, the segments depending on each
    /// other as [`dependence`](Self::dependence) says: `Independent` as
    /// drawn, `RankCorrelation` by Iman–Conover on the segments' total
    /// claims development results, as the bootstraps' one-year view does;
    /// `Synchronized` is an [`Error::Bayes`]. A failure names its segment.
    pub fn one_year_segments(
        &self,
        triangle: &Triangle,
        column: &str,
        settings: &BayesOneYear,
    ) -> Result<BayesOneYearFits> {
        self.validate()?;
        settings.validate()?;
        self.check_dependence(triangle)?;
        let mut k = 0;
        let fitted = fit_each(triangle, column, |segment| {
            let predictive = Stream::SegmentPredictive(k);
            let fit = self
                .fit_segment(segment, column, Stream::SegmentChains(k), predictive)
                .and_then(|(fit, _)| {
                    let year = self.year_draws(triangle, segment, &fit, predictive, settings)?;
                    Ok((fit, year))
                });
            k += 1;
            fit
        })?;

        let mut hasher = InputHasher::new();
        hasher.str(column);
        let mut dims = fitted.key_names.clone();
        dims.push("origin".into());
        let mut components = Vec::new();
        for ((label, (fit, _)), segment) in fitted.iter().zip(triangle.segments(column)?) {
            hasher.str(&label.to_string());
            hash_segment(&mut hasher, &segment, &fit.chain_ladder, &segment.ages);
            components.extend(origin_keys(label, &segment.origins));
        }
        let n_sims = settings.sims(self.n_sims);
        let mut draws = Vec::with_capacity(n_sims * components.len());
        for i in 0..n_sims {
            for (_, year) in &fitted.fits {
                let m = year.opening.len();
                draws.extend_from_slice(&year.cdr[i * m..(i + 1) * m]);
            }
        }
        let first = &fitted.fits[0].0.model;
        let provenance = self
            .one_year_provenance(column, first, settings)
            .param("segments", fitted.len())
            .param("dependence", format!("{:?}", self.dependence))
            .input_hash(hasher.finish());
        let cdr = PredictiveDistribution::from_draws(dims, components, draws, provenance)?;
        let segments = fitted.map(|(fit, year)| BayesOneYearSegment {
            fit: fit.clone(),
            opening_ultimate: one_year::opening_ultimate(&fit.chain_ladder, &year.opening),
            opening_reserve: year.opening.clone(),
            pareto_k: year.pareto_k.clone(),
            refitted: year.refitted.clone(),
            moment_matched: year.moment_matched.clone(),
            unrefitted: year.unrefitted.clone(),
            refit_warnings: year.refit_warnings,
            k_threshold: settings.k_threshold,
        });
        let cdr = self.dependence.reorder(cdr, &segments, self.seed)?;
        Ok(BayesOneYearFits { segments, cdr })
    }

    /// `Synchronized` and a bad rank-correlation matrix fail before any
    /// sampling.
    fn check_dependence(&self, triangle: &Triangle) -> Result<()> {
        if self.dependence == SegmentDependence::Synchronized {
            return Err(super::synchronized_error());
        }
        self.dependence.validate(triangle.index().len())
    }

    /// The one-year simulations of one fitted segment, from the seed the
    /// segment's `predictive` stream derives (module docs, "Seeds").
    fn year_draws(
        &self,
        triangle: &Triangle,
        segment: &Segment,
        fit: &BayesOdpSegment,
        predictive: Stream,
        settings: &BayesOneYear,
    ) -> Result<YearDraws> {
        let cells = OdpRefit::new(self, fit);
        let year = OdpYear::new(
            &cells,
            self.process,
            triangle,
            segment,
            &fit.chain_ladder.latest_position,
        );
        one_year::simulate(
            &year,
            &fit.posterior,
            settings,
            settings.sims(self.n_sims),
            one_year_seed(derive_seed(self.seed, predictive)),
        )
    }

    fn one_year_provenance(
        &self,
        column: &str,
        model: &Model,
        settings: &BayesOneYear,
    ) -> Provenance {
        let mut provenance = self.base_provenance(column);
        provenance.model = "bayes_odp_one_year".into();
        one_year::provenance(provenance, settings, settings.sims(self.n_sims))
            .param("c0", model.c0)
            .param("scale", model.phi)
    }

    /// The posterior of one segment and its reserves by origin, with the
    /// seeds of `chains` and `predictive`.
    fn fit_segment(
        &self,
        segment: &Segment,
        column: &str,
        chains: Stream,
        predictive: Stream,
    ) -> Result<(BayesOdpSegment, PredictiveDistribution)> {
        let Prepared {
            chain_ladder,
            model,
            start,
        } = self.prepare(segment, true)?;
        let (posterior, diagnostics) =
            self.sample(&model, start, derive_seed(self.seed, chains))?;

        let mut hasher = InputHasher::new();
        hasher.str(column);
        hash_segment(&mut hasher, segment, &chain_ladder, &segment.ages);
        let latest = &chain_ladder.latest_position;
        let nd = segment.n_dev;
        let (process, phi) = (self.process, model.phi);
        let reserves = posterior.predictive(
            vec!["origin".into()],
            segment.origins.iter().map(|&o| vec![o.into()]).collect(),
            self.n_sims,
            derive_seed(self.seed, predictive),
            self.provenance(column, &model, hasher),
            |theta, rng: &mut StreamRng, row| {
                for (w, reserve) in row.iter_mut().enumerate() {
                    *reserve = (latest[w] + 1..nd)
                        .map(|d| process_draw(process, phi, model.eta(theta, w, d).exp(), rng))
                        .sum();
                }
            },
        )?;
        Ok((
            BayesOdpSegment {
                chain_ladder,
                scale: model.phi,
                posterior,
                diagnostics,
                model,
            },
            reserves,
        ))
    }

    /// NUTS on `model` whitened at its mode (found from `start`), with
    /// the chains' `seed`; the draws on the natural scale and their
    /// diagnostics.
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
        let posterior = draws.transform(model.names(), |z| whitened.theta(z))?;
        let summary = posterior.summary()?;
        let diagnostics = Diagnostics::new(
            &summary,
            posterior.chains(),
            posterior.divergences().iter().sum(),
        );
        Ok((posterior, diagnostics))
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
        let model = self
            .model(
                no,
                nd,
                cells,
                fit.scale,
                priors,
                |w| format!("origin {}", segment.origins[w]),
                |d| format!("development age {} months", segment.ages[d]),
            )
            .map_err(|e| last_age_remedy(e, &segment.ages, segment.alone))?;
        // The chain ladder's fitted increments are U_w g_d, the ODP GLM's
        // MLE: their logs give the log-linear parameters when all are
        // positive. Otherwise start from the prior's centre.
        let fitted = &fit.fitted;
        let start = if model
            .cells
            .iter()
            .all(|&(w, d, _)| fitted[w * nd + d] > 0.0)
        {
            let p = model.dim();
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
            model.prior_mean.clone()
        };
        Ok(Prepared {
            chain_ladder: fit.chain_ladder,
            model,
            start,
        })
    }

    /// The model of decision 3 on `cells` (origin, development, increment)
    /// with `n_origins` origins and `n_dev` ages, each observed at least
    /// once, and the scale `phi`. Every origin's and age's increments must
    /// sum to more than zero ([`Error::Bayes`], naming it by `origin` or
    /// `age`). Without `priors`, the likelihood alone.
    #[allow(clippy::too_many_arguments)]
    fn model(
        &self,
        n_origins: usize,
        n_dev: usize,
        cells: Vec<(usize, usize, f64)>,
        phi: f64,
        priors: bool,
        origin: impl Fn(usize) -> String,
        age: impl Fn(usize) -> String,
    ) -> Result<Model> {
        let (no, nd) = (n_origins, n_dev);
        let (mut by_origin, mut by_age) = (vec![0.0; no], vec![0.0; nd]);
        for &(w, d, x) in &cells {
            by_origin[w] += x;
            by_age[d] += x;
        }
        let bad = |sum: &f64| sum.is_nan() || *sum <= 0.0;
        for (name, sums) in [
            (&origin as &dyn Fn(usize) -> String, &by_origin),
            (&age, &by_age),
        ] {
            if let Some((i, sum)) = sums.iter().enumerate().find(|(_, s)| bad(s)) {
                return Err(Error::Bayes(format!(
                    "{}: the observed increments sum to {sum}, so no log-linear mean exists",
                    name(i)
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
        Ok(Model {
            n_origins: no,
            n_dev: nd,
            cells,
            phi,
            c0,
            prior_mean,
            prior_precision,
        })
    }

    /// The provenance of every fit: the method and the settings.
    fn base_provenance(&self, column: &str) -> Provenance {
        let s = &self.sampler;
        Provenance::new("bayes_odp")
            .param("intercept_sd", self.intercept_sd)
            .param("origin_sd", self.origin_sd)
            .param("development_sd", self.development_sd)
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
    }

    fn provenance(&self, column: &str, model: &Model, hasher: InputHasher) -> Provenance {
        self.base_provenance(column)
            .param("c0", model.c0)
            .param("scale", model.phi)
            .input_hash(hasher.finish())
    }
}

/// What [`BayesOdp::prepare`] gives.
pub(crate) struct Prepared {
    pub(crate) chain_ladder: ChainLadderFit,
    pub(crate) model: Model,
    pub(crate) start: Vec<f64>,
}

/// The diagonal backtest's view of a [`BayesOdp`] (decision 8): the model
/// fitted on the train rows' increments, with origin and development
/// effects for the levels the train rows have, forecasting the test rows.
///
/// * The scale `phi` is the Pearson scale of the train rows at the
///   likelihood's mode (the ODP GLM's MLE, which is the chain ladder's
///   fit when the train rows form a triangle), on `N - p` degrees of
///   freedom, as [`BayesOdp::fit`] and the quasi-Poisson GLM have it.
/// * Every train row is fitted, negative increments included
///   ([`fit_rows`](TriangleModel::fit_rows) is the default), but each
///   origin's and age's train increments must sum to more than zero
///   ([`Error::Bayes`]).
/// * The chains and the simulations take seeds derived from the
///   backtest's `seed` as [`BayesOdp::fit`] does from [`BayesOdp::seed`];
///   the backtest's `n_sims` replaces [`BayesOdp::n_sims`]. The fit's
///   diagnostics come back in [`CellForecast::diagnostics`] and
///   [`crate::Backtest::diagnostics`]; a fit with warnings is still scored.
/// * `mean` is each test cell's posterior mean `E[exp(eta)]`, the
///   distribution adds [`BayesOdp::process`] to a posterior draw, and the
///   log density is the log of the mean, over every posterior draw, of
///   the over-dispersed Poisson's normalised density
///   ([`Family::log_density`] at `mu` and `phi`, decision 5): the log
///   predictive density, as [`GlmCandidate`](crate::GlmCandidate) gives
///   it. A negative test increment has density zero there (log density
///   minus infinity), for the GLM too.
///
/// Its name in the backtest is `"bayes_odp"`.
impl TriangleModel for BayesOdp {
    fn name(&self) -> &str {
        "bayes_odp"
    }

    fn forecast(
        &self,
        cells: &TriangleFrame,
        train: &[usize],
        test: &[usize],
        n_sims: usize,
        seed: u64,
    ) -> Result<CellForecast> {
        self.validate()?;
        // The levels the train rows have, renumbered from 0.
        let levels = |of: &dyn Fn(usize) -> usize| {
            let mut levels: Vec<usize> = train.iter().map(|&r| of(r)).collect();
            levels.sort_unstable();
            levels.dedup();
            levels
        };
        let origins = levels(&|r| cells.origin_of(r));
        let ages = levels(&|r| cells.development_of(r));
        let position = |levels: &[usize], level: usize| levels.binary_search(&level).ok();
        let response = cells.response();
        let train_cells: Vec<(usize, usize, f64)> = train
            .iter()
            .map(|&r| {
                (
                    position(&origins, cells.origin_of(r)).expect("a train level"),
                    position(&ages, cells.development_of(r)).expect("a train level"),
                    response[r],
                )
            })
            .collect();
        let test_cells = test
            .iter()
            .map(|&r| {
                match (
                    position(&origins, cells.origin_of(r)),
                    position(&ages, cells.development_of(r)),
                ) {
                    (Some(w), Some(d)) => Ok((w, d)),
                    _ => Err(Error::Bayes(format!(
                        "test row {r} has an origin or age without a train row"
                    ))),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let (no, nd) = (origins.len(), ages.len());
        let p = no + nd - 1;
        if train_cells.len() <= p {
            return Err(Error::Bayes(format!(
                "{} train rows for {p} parameters: the scale needs more rows than parameters",
                train_cells.len()
            )));
        }
        let origin = |w: usize| format!("origin {}", cells.origins()[origins[w]]);
        let age = |d: usize| format!("development age {} months", cells.ages()[ages[d]]);

        // The scale: Pearson's at the likelihood's mode, which does not
        // depend on phi.
        let likelihood = self.model(no, nd, train_cells.clone(), 1.0, false, origin, age)?;
        let mle = mode(&likelihood, likelihood.prior_mean.clone())?;
        let pearson: f64 = train_cells
            .iter()
            .map(|&(w, d, x)| {
                let mu = likelihood.eta(&mle, w, d).exp();
                (x - mu).powi(2) / mu
            })
            .sum();
        let phi = pearson / (train_cells.len() - p) as f64;
        if !(phi.is_finite() && phi > 0.0) {
            return Err(Error::Bayes(format!(
                "the train rows' Pearson scale is {phi}, not a positive number"
            )));
        }
        let model = self.model(no, nd, train_cells, phi, true, origin, age)?;
        let (posterior, diagnostics) =
            self.sample(&model, mle, derive_seed(seed, Stream::Chains))?;

        let n_draws = posterior.n_draws();
        let mean: Vec<f64> = test_cells
            .iter()
            .map(|&(w, d)| {
                (0..n_draws)
                    .map(|s| model.eta(posterior.draw(s), w, d).exp())
                    .sum::<f64>()
                    / n_draws as f64
            })
            .collect();
        let process = self.process;
        let distribution = posterior.predictive(
            vec!["row".into()],
            test.iter().map(|&r| vec![(r as i64).into()]).collect(),
            n_sims,
            derive_seed(seed, Stream::Predictive),
            self.base_provenance("backtest")
                .param("scale", phi)
                .param("c0", model.c0),
            |theta, rng: &mut StreamRng, row| {
                for (x, &(w, d)) in row.iter_mut().zip(&test_cells) {
                    *x = process_draw(process, phi, model.eta(theta, w, d).exp(), rng);
                }
            },
        )?;
        // Each density integrates the continuous Poisson's normalising
        // constant, so the test cells run in parallel.
        let log_density = test_cells
            .par_iter()
            .zip(test.par_iter())
            .map(|(&(w, d), &r)| {
                let ln_f = (0..n_draws)
                    .map(|s| {
                        let mu = model.eta(posterior.draw(s), w, d).exp();
                        Family::Poisson.log_density(response[r], mu, phi, 1.0)
                    })
                    .collect::<prospicio_core::Result<Vec<f64>>>()?;
                Ok(log_mean_exp(&ln_f))
            })
            .collect::<Result<Vec<f64>>>()?;
        Ok(CellForecast {
            mean,
            distribution,
            log_density: Some(log_density),
            diagnostics: Some(diagnostics),
        })
    }
}

/// `ln(mean(exp(v)))` without overflow; minus infinity when every value
/// is.
pub(super) fn log_mean_exp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + (v.iter().map(|x| (x - m).exp()).sum::<f64>() / v.len() as f64).ln()
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
        loo(&self.model, &self.posterior, &self.chain_ladder)
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

/// `e` with the remedy for a falling last age appended when it names the
/// last of `ages` (design note `reserving-bayes.md`, decisions 18 and 26):
/// keep the earlier ages with `Triangle::max_age`. Unless the segment is
/// `alone` in its triangle, the remedy adds that `max_age` cuts every
/// segment.
pub(super) fn last_age_remedy(e: Error, ages: &[crate::Lag], alone: bool) -> Error {
    match (e, ages) {
        (Error::Bayes(m), [.., previous, last])
            if m.starts_with(&format!("development age {last} months:")) =>
        {
            Error::Bayes(format!(
                "{m}. The chain ladder accepts this triangle. Keep the ages up to {previous} \
                 months with the triangle's max_age({previous}), which leaves out the rows at \
                 {last} months{}, and carry the movement left out as a tail outside this model",
                every_segment(*previous, alone)
            ))
        }
        (e, _) => e,
    }
}

/// The clause a `max_age(previous)` remedy adds on a triangle of several
/// segments: the cut reaches every segment, so a segment that develops
/// further loses its later ages unless this one is selected first.
pub(super) fn every_segment(previous: crate::Lag, alone: bool) -> String {
    if alone {
        String::new()
    } else {
        format!(
            " (max_age cuts every segment of the triangle at {previous} months, so select \
             this segment first if another one develops further)"
        )
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
        let mut f = S::from_f64(0.0);
        for (j, &t) in theta.iter().enumerate() {
            let r = t - self.prior_mean[j];
            f -= r * r * (0.5 * self.prior_precision[j]);
        }
        for &(w, d, x) in &self.cells {
            let eta = self
                .params(w, d)
                .into_iter()
                .flatten()
                .fold(S::from_f64(0.0), |acc, j| acc + theta[j]);
            f += (eta * x - eta.exp()) / self.phi;
        }
        f
    }
}

/// A fitted [`BayesOdp`] segment as the one-year view sees it
/// ([`OdpCells`], decision 17): `eta` is [`Model::eta`] on the stored
/// draws `(c, a, b)`, and a refit or the year's log posterior adds the
/// year's increments as observed cells to the same model (priors, `c0` and
/// `phi` unchanged). Every origin and age has an observed cell, so the
/// draws carry every effect a future cell needs.
struct OdpRefit<'a> {
    odp: &'a BayesOdp,
    model: &'a Model,
    /// The opening posterior mean, the refits' Newton start.
    start: Vec<f64>,
}

impl<'a> OdpRefit<'a> {
    fn new(odp: &'a BayesOdp, fit: &'a BayesOdpSegment) -> Self {
        Self {
            odp,
            model: &fit.model,
            start: fit.posterior.mean(),
        }
    }

    fn extended(&self, extra: &[(usize, usize, f64)]) -> Model {
        let mut extended = self.model.clone();
        extended.cells.extend_from_slice(extra);
        extended
    }
}

impl OdpCells for OdpRefit<'_> {
    fn eta(&self, theta: &[f64], w: usize, d: usize) -> f64 {
        self.model.eta(theta, w, d)
    }

    fn phi(&self) -> f64 {
        self.model.phi
    }

    /// The draws are stored on the model's own scale, an affine map of
    /// the whitened one NUTS samples, so no Jacobian enters.
    fn log_posterior(&self, theta: &[f64], extra: &[(usize, usize, f64)]) -> f64 {
        let mut grad = vec![0.0; theta.len()];
        if extra.is_empty() {
            // Moment matching's many calls on the observed triangle.
            return self.model.log_posterior(theta, &mut grad);
        }
        self.extended(extra).log_posterior(theta, &mut grad)
    }

    /// Sampled as [`BayesOdp::fit`] samples, from the opening posterior
    /// mean.
    fn refit(&self, extra: &[(usize, usize, f64)], seed: u64) -> Result<(PosteriorDraws, bool)> {
        let (posterior, diagnostics) =
            self.odp
                .sample(&self.extended(extra), self.start.clone(), seed)?;
        Ok((posterior, !diagnostics.warnings.is_empty()))
    }
}

/// PSIS-LOO of `model`'s observed increments under `posterior` ([`BayesOdpFit::loo`]).
fn loo(model: &Model, posterior: &PosteriorDraws, chain_ladder: &ChainLadderFit) -> Result<Loo> {
    if let Some(&(w, d, x)) = model.cells.iter().find(|c| c.2 < 0.0) {
        return Err(Error::Bayes(format!(
            "loo needs non-negative increments, and origin {} at development age {} \
             months is {x}",
            chain_ladder.origins[w], chain_ladder.development.development[d]
        )));
    }
    let n = model.cells.len();
    let s = posterior.n_draws();
    // Each term integrates the continuous Poisson's normalising
    // constant, so the draws run in parallel; the result does not
    // depend on the threads.
    let rows: Vec<Vec<f64>> = posterior
        .values()
        .par_chunks_exact(posterior.dim())
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
    let chains = posterior.chains();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bayes::OneYearRefit;
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
    fn the_generic_density_matches_the_hand_written_one() {
        use crate::bayes::autodiff::{assert_oracle, posterior_points};
        for (label, triangle) in [("BayesOdp GenIns", genins()), ("BayesOdp RAA", raa())] {
            let Prepared { model, start, .. } = prepared(&triangle, &BayesOdp::default(), true);
            let mode = mode(&model, start).unwrap();
            let whitened = Whitened::new(&model, &mode).unwrap();
            let points = posterior_points(&whitened, model.dim(), 20);
            assert_oracle(label, &model, &points);
        }
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
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
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
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
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

    /// GenIns twice, as lines `a` and `b` of a `lob` key.
    fn twins() -> Triangle {
        let (mut keys, mut origin, mut ages, mut values) = (vec![], vec![], vec![], vec![]);
        for line in ["a", "b"] {
            for (k, row) in GENINS.iter().enumerate() {
                for (d, &v) in row.iter().enumerate() {
                    keys.push(line);
                    origin.push(crate::Month::january(2001 + k as i32));
                    ages.push(12 * (d as u32 + 1));
                    values.push(v);
                }
            }
        }
        Triangle::from_long(&Long {
            keys: &[("lob", &keys)],
            origin: &origin,
            development: crate::DevelopmentColumn::Age(&ages),
            values: &[("values", &values)],
            origin_grain: crate::Grain::Year,
            development_grain: crate::Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    /// Each line's total reserve, simulation by simulation.
    fn line_totals(reserves: &PredictiveDistribution) -> Vec<Vec<f64>> {
        let by = reserves.aggregate(&["lob"]).unwrap();
        let n = by.n_components();
        (0..n)
            .map(|j| by.draw_matrix().chunks(n).map(|r| r[j]).collect())
            .collect()
    }

    fn spearman(x: &[f64], y: &[f64]) -> f64 {
        let ranks = |v: &[f64]| {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
            let mut r = vec![0.0; v.len()];
            for (k, &i) in idx.iter().enumerate() {
                r[i] = k as f64;
            }
            r
        };
        let (rx, ry) = (ranks(x), ranks(y));
        let m = (x.len() as f64 - 1.0) / 2.0;
        let sxy: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - m) * (b - m)).sum();
        let sxx: f64 = rx.iter().map(|a| (a - m).powi(2)).sum();
        sxy / sxx
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn each_segment_has_its_own_seeds() {
        let tri = twins();
        let fits = quick(9).fit_segments(&tri, "values").unwrap();
        assert_eq!(fits.reserves.dims(), ["lob", "origin"]);
        assert_eq!(fits.reserves.n_components(), 20);
        assert_eq!(fits.segments.len(), 2);
        let (a, b) = (&fits.segments.fits[0], &fits.segments.fits[1]);
        // The same data, so the same scale, but their own chains.
        assert_eq!(a.scale, b.scale);
        assert_ne!(a.posterior.values(), b.posterior.values());
        // Not the single-segment fit's seeds either.
        let alone = quick(9).fit(&genins(), "values").unwrap();
        assert_ne!(a.posterior.values(), alone.posterior.values());
        // Shared streams would make the twins' totals almost perfectly
        // rank correlated; their own make them independent.
        let totals = line_totals(&fits.reserves);
        let r = spearman(&totals[0], &totals[1]);
        assert!(r.abs() < 3.0 / 2_000f64.sqrt(), "rank correlation {r}");

        assert_eq!(fits.to_long().n_rows(), 20);
        let t = fits.totals();
        assert_eq!(t.n_rows(), 2);
        assert_eq!(t.column("scale").unwrap(), [a.scale, b.scale]);
        let mean_b = totals[1].iter().sum::<f64>() / 2e3;
        assert!((t.column("mean").unwrap()[1] / mean_b - 1.0).abs() < 1e-9);
        assert_eq!(fits.development_table().n_rows(), 20);
        let one = fits.segment(&[("lob", "b")]).unwrap();
        assert_eq!(one.reserves.n_components(), 10);
        assert_eq!(one.segments.fits[0].posterior, b.posterior);
        assert_eq!(b.summary().unwrap().len(), 19);

        let provenance = fits.reserves.provenance();
        assert_eq!(provenance.model, "bayes_odp");
        assert!(
            provenance
                .parameters
                .contains(&("segments".into(), "2".into()))
        );

        let again = quick(9).fit_segments(&tri, "values").unwrap();
        assert_eq!(again.reserves.draw_matrix(), fits.reserves.draw_matrix());
    }

    #[test]
    fn synchronized_segments_are_an_error() {
        let model = BayesOdp {
            dependence: SegmentDependence::Synchronized,
            ..quick(0)
        };
        let e = model.fit_segments(&twins(), "values").unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("Synchronized dependence")),
            "{e}"
        );
    }

    #[test]
    fn a_bad_rank_correlation_fails_before_sampling() {
        // Each bad matrix fails at once: with a sampler this long, any
        // sampling before the check would not finish.
        let slow = BayesOdp {
            sampler: Sampler {
                tune: 1_000_000,
                draws: 1_000_000,
                ..Sampler::default()
            },
            ..BayesOdp::default()
        };
        let bad = [
            // Three segments' matrix on two segments.
            vec![1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0],
            // Not a unit diagonal.
            vec![0.9, 0.5, 0.5, 1.0],
            // Not symmetric.
            vec![1.0, 0.5, 0.4, 1.0],
            // An entry out of [-1, 1]. (On two segments any valid matrix
            // is positive definite; `dependence.rs` tests that check.)
            vec![1.0, 1.5, 1.5, 1.0],
        ];
        for spearman in bad {
            let model = BayesOdp {
                dependence: SegmentDependence::RankCorrelation { spearman },
                ..slow.clone()
            };
            let e = model.fit_segments(&twins(), "values").unwrap_err();
            assert!(
                matches!(
                    &e,
                    Error::InvalidSetting {
                        name: "spearman",
                        ..
                    }
                ),
                "{e}"
            );
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn rank_correlation_reorders_whole_simulations() {
        let tri = twins();
        let independent = quick(10).fit_segments(&tri, "values").unwrap();
        let ranked = BayesOdp {
            dependence: SegmentDependence::RankCorrelation {
                spearman: vec![1.0, 0.8, 0.8, 1.0],
            },
            ..quick(10)
        }
        .fit_segments(&tri, "values")
        .unwrap();
        // Each line keeps its own simulations, as whole rows.
        let rows = |pd: &PredictiveDistribution, s: usize| {
            let mut rows: Vec<Vec<f64>> = pd
                .draw_matrix()
                .chunks(20)
                .map(|r| r[s * 10..(s + 1) * 10].to_vec())
                .collect();
            rows.sort_by(|x, y| {
                x.iter()
                    .zip(y)
                    .map(|(a, b)| a.total_cmp(b))
                    .find(|o| o.is_ne())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            rows
        };
        for s in 0..2 {
            assert_eq!(rows(&independent.reserves, s), rows(&ranked.reserves, s));
        }
        let totals = line_totals(&ranked.reserves);
        let r = spearman(&totals[0], &totals[1]);
        assert!((r - 0.8).abs() < 0.05, "rank correlation {r}");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_backtest_model_is_the_fit_on_the_train_rows() {
        let tri = genins();
        let cells = TriangleFrame::new(&tri, "values", None).unwrap();
        let train = cells.observed().to_vec();
        // In-sample cells of the oldest origin, at 12 and 24 months.
        let test = [0, 1];
        let model = quick(11);
        assert_eq!(TriangleModel::name(&model), "bayes_odp");
        let f = model.forecast(&cells, &train, &test, 500, 3).unwrap();
        let fit = model.fit(&tri, "values").unwrap();
        // The scale is the chain ladder's: the Pearson scale at the MLE.
        let scale: f64 = f
            .distribution
            .provenance()
            .parameters
            .iter()
            .find(|(k, _)| k == "scale")
            .unwrap()
            .1
            .parse()
            .unwrap();
        assert!(
            (scale / fit.scale - 1.0).abs() < 1e-8,
            "{scale} vs {}",
            fit.scale
        );
        // The posterior mean of a well-determined cell is near the chain
        // ladder's fitted increment.
        let segment = tri.segment("values").unwrap();
        let (cl, _) = prepare(&segment, &segment.ages, Tail::default()).unwrap();
        for (j, &r) in test.iter().enumerate() {
            let fitted = cl.fitted[r];
            assert!(
                (f.mean[j] / fitted - 1.0).abs() < 0.05,
                "{} vs {fitted}",
                f.mean[j]
            );
        }
        assert_eq!(f.distribution.n_components(), 2);
        assert_eq!(f.distribution.n_sims(), 500);
        let lpd = f.log_density.unwrap();
        assert!(
            lpd.len() == 2 && lpd.iter().all(|l| l.is_finite()),
            "{lpd:?}"
        );
        // Deterministic for the backtest's seed.
        let again = model.forecast(&cells, &train, &test, 500, 3).unwrap();
        assert_eq!(
            again.distribution.draw_matrix(),
            f.distribution.draw_matrix()
        );
        assert_eq!(again.log_density.unwrap(), lpd);
        // The diagnostics of the sampled posterior come with the forecast,
        // as deterministic as the draws.
        let d = f
            .diagnostics
            .as_ref()
            .expect("a Bayesian model reports diagnostics");
        assert!(d.max_rhat.is_finite() && d.max_rhat < 1.1, "{d:?}");
        assert_eq!(again.diagnostics.as_ref(), Some(d));

        // Without its only cell, the newest origin (row 90) has no train
        // row.
        let without: Vec<usize> = train.iter().copied().filter(|&r| r != 90).collect();
        let e = model.forecast(&cells, &without, &[90], 100, 3).unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.contains("without a train row")),
            "{e}"
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_backtest_reports_diagnostics_and_still_scores_a_short_run() {
        let cells = TriangleFrame::new(&raa(), "values", None).unwrap();
        // One chain of 50 kept draws cannot reach 100 effective draws per
        // chain, so every split warns.
        let short = BayesOdp {
            sampler: Sampler {
                chains: 1,
                tune: 50,
                draws: 50,
                ..Sampler::default()
            },
            ..BayesOdp::default()
        };
        let bt = crate::diagonal_backtest(&cells, &[&short], 2, 200, 5, 0.9).unwrap();
        let diagnostics = bt.diagnostics(0);
        assert_eq!(diagnostics.len(), 2);
        for d in diagnostics {
            let d = d.as_ref().expect("a Bayesian model reports diagnostics");
            assert!(!d.warnings.is_empty(), "{d:?}");
        }
        let warnings = bt.warnings();
        assert!(warnings.iter().any(|(m, s, _)| (*m, *s) == (0, 0)));
        assert!(warnings.iter().any(|(m, s, _)| (*m, *s) == (0, 1)));
        // Scored all the same.
        for metric in 0..crate::METRICS.len() {
            assert!(bt.split_scores(0, metric).iter().all(|v| v.is_finite()));
        }
    }

    fn columns(draws: &PredictiveDistribution) -> Vec<Vec<f64>> {
        let n = draws.n_components();
        (0..n)
            .map(|j| draws.draw_matrix().chunks(n).map(|r| r[j]).collect())
            .collect()
    }

    fn mean_sd(x: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        let m = x.iter().sum::<f64>() / n;
        (
            m,
            (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n).sqrt(),
        )
    }

    fn total(d: &PredictiveDistribution) -> Vec<f64> {
        let n = d.n_components();
        d.draw_matrix().chunks(n).map(|r| r.iter().sum()).collect()
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_one_year_view_reweights_the_fits_posterior() {
        let model = quick(4);
        let settings = BayesOneYear {
            refit: Some(OneYearRefit::ReweightOnly),
            ..BayesOneYear::default()
        };
        let view = model.one_year(&genins(), "values", &settings).unwrap();
        // The posterior and lifetime reserves are fit()'s, draw for draw.
        let fit = model.fit(&genins(), "values").unwrap();
        assert_eq!(view.fit.posterior, fit.posterior);
        assert_eq!(view.fit.reserves.draw_matrix(), fit.reserves.draw_matrix());
        assert_eq!(view.cdr.dims(), ["origin"]);
        assert_eq!(view.pareto_k.len(), 2_000);
        assert!(view.refitted.is_empty());

        let cdr = columns(&view.cdr);
        let life = columns(&fit.reserves);
        // 2001 is at the last age: no cell, no change.
        assert!(cdr[0].iter().all(|&x| x == 0.0));
        assert_eq!(view.opening_reserve[0], 0.0);
        // The opening reserve is the posterior mean of the expected run-off.
        let (life_mean, life_sd) = mean_sd(&total(&fit.reserves));
        let opening: f64 = view.opening_reserve.iter().sum();
        assert!(
            (opening - life_mean).abs() < 4.0 * life_sd / 2_000f64.sqrt(),
            "{opening} {life_mean}"
        );
        for (o, ((u, l), r)) in view
            .opening_ultimate
            .iter()
            .zip(&fit.chain_ladder.latest)
            .zip(&view.opening_reserve)
            .enumerate()
        {
            assert!((u - l - r).abs() < 1e-6 * u, "origin {o}");
        }
        // 2002 has one cell left, in the year: its one-year view is its
        // whole run-off, the opening reserve less the lifetime reserve.
        let (_, s) = mean_sd(&cdr[1]);
        let (_, s_life) = mean_sd(&life[1]);
        assert!((s / s_life - 1.0).abs() < 0.1, "{s} {s_life}");
        // Every origin's mean is zero within Monte Carlo error, and its
        // spread at most its lifetime spread.
        for (o, (c, l)) in cdr.iter().zip(&life).enumerate().skip(1) {
            let (m, s) = mean_sd(c);
            let (_, s_life) = mean_sd(l);
            assert!(m.abs() < 4.0 * s / 2_000f64.sqrt(), "origin {o}: {m} {s}");
            assert!(s <= 1.1 * s_life, "origin {o}: {s} {s_life}");
        }
        // The total is narrower than the lifetime total.
        assert!(mean_sd(&total(&view.cdr)).1 < 0.9 * life_sd);

        let long = view.to_long();
        assert_eq!(long.n_rows(), 10);
        for (a, b) in long
            .column("opening_reserve")
            .unwrap()
            .iter()
            .zip(&view.opening_reserve)
        {
            assert!((a - b).abs() < 1e-6 * (1.0 + b));
        }
        assert!(long.column("cdr_std_dev").is_some());
        let totals = view.totals();
        assert_eq!(
            totals.column("flagged_share").unwrap(),
            [view.flagged_share()]
        );
        assert_eq!(totals.column("scale").unwrap(), [view.fit.scale]);
        let provenance = view.cdr.provenance();
        assert_eq!(provenance.model, "bayes_odp_one_year");
        assert!(
            provenance
                .parameters
                .contains(&("refit".into(), "ReweightOnly".into()))
        );

        // A seed replays the view, whatever the threads.
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| model.one_year(&genins(), "values", &settings).unwrap());
        assert_eq!(one.cdr.draw_matrix(), view.cdr.draw_matrix());
        assert_eq!(one.pareto_k, view.pareto_k);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn refits_see_the_same_years_and_agree_with_the_reweighting() {
        let model = BayesOdp {
            n_sims: 40,
            ..quick(5)
        };
        let run = |refit, k_threshold| {
            model
                .one_year(
                    &genins(),
                    "values",
                    &BayesOneYear {
                        k_threshold,
                        refit: Some(refit),
                        ..BayesOneYear::default()
                    },
                )
                .unwrap()
        };
        let only = run(OneYearRefit::ReweightOnly, 0.7);
        let all = run(OneYearRefit::RefitAll, 0.7);
        assert_eq!(all.refitted, (0..40).collect::<Vec<_>>());
        assert_eq!(all.pareto_k, only.pareto_k);
        // Refitting the flagged simulations gives the refitted rows for
        // them and the reweighted rows for the others: the same years, the
        // same refit seeds.
        let flagged = run(OneYearRefit::RefitFlagged, 0.7);
        let is_flagged: Vec<bool> = only
            .pareto_k
            .iter()
            .map(|&k| k.is_nan() || k > 0.7)
            .collect();
        assert_eq!(
            flagged.refitted,
            (0..40).filter(|&i| is_flagged[i]).collect::<Vec<_>>()
        );
        assert!(!flagged.refitted.is_empty() && flagged.refitted.len() < 40);
        let rows =
            |v: &BayesOneYearFit, i: usize| v.cdr.draw_matrix()[i * 10..(i + 1) * 10].to_vec();
        for (i, &f) in is_flagged.iter().enumerate() {
            let expected = if f { rows(&all, i) } else { rows(&only, i) };
            assert_eq!(rows(&flagged, i), expected, "simulation {i}");
        }
        // 2002's CDR is its whole run-off, with no closing reserve to
        // estimate, so it is the same either way.
        let (a, b) = (columns(&only.cdr), columns(&all.cdr));
        assert_eq!(a[1], b[1]);
        // Where k is below the threshold, a reweighted and a refitted
        // simulation differ by the two estimators' Monte Carlo errors,
        // small against the total's spread (about 0.06 sd at these 1,200
        // draws).
        let (ta, tb) = (total(&only.cdr), total(&all.cdr));
        let (_, sd) = mean_sd(&ta);
        let gaps: Vec<f64> = (0..40)
            .filter(|&i| !is_flagged[i])
            .map(|i| (ta[i] - tb[i]) / sd)
            .collect();
        let rms = (gaps.iter().map(|g| g * g).sum::<f64>() / gaps.len() as f64).sqrt();
        assert!(
            rms < 0.2,
            "rms gap {rms} sd over {} simulations",
            gaps.len()
        );
    }

    #[test]
    fn one_year_errors() {
        let e = quick(0)
            .one_year(
                &genins(),
                "values",
                &BayesOneYear {
                    k_threshold: f64::NAN,
                    ..BayesOneYear::default()
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
        let model = BayesOdp {
            dependence: SegmentDependence::Synchronized,
            ..quick(0)
        };
        let e = model
            .one_year_segments(&twins(), "values", &BayesOneYear::default())
            .unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("Synchronized dependence")),
            "{e}"
        );
    }

    #[test]
    fn a_fully_developed_triangle_has_no_flags_and_no_refits() {
        // Every origin at the last age: the coming year has no cells, so
        // the reweighting has nothing to do and the default refits nothing.
        let square = annual(
            2001,
            &[
                &[357.0, 1124.0, 1735.0, 2218.0],
                &[352.0, 1236.0, 2170.0, 3353.0],
                &[290.0, 1292.0, 2219.0, 3235.0],
                &[310.0, 1418.0, 2195.0, 3757.0],
            ],
        );
        let model = BayesOdp {
            n_sims: 200,
            ..quick(4)
        };
        let view = model
            .one_year(&square, "values", &BayesOneYear::default())
            .unwrap();
        assert_eq!(view.flagged_share(), 0.0);
        assert!(view.refitted.is_empty());
        assert!(view.pareto_k.iter().all(|&k| k == 0.0));
        assert!(view.cdr.draw_matrix().iter().all(|v| v.abs() < 1e-9));
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_one_year_view_over_segments() {
        let settings = BayesOneYear {
            refit: Some(OneYearRefit::ReweightOnly),
            ..BayesOneYear::default()
        };
        let tri = twins();
        let fits = quick(9)
            .one_year_segments(&tri, "values", &settings)
            .unwrap();
        assert_eq!(fits.cdr.dims(), ["lob", "origin"]);
        assert_eq!(fits.cdr.n_components(), 20);
        // Each segment's posterior is fit_segments'.
        let lifetime = quick(9).fit_segments(&tri, "values").unwrap();
        for (a, b) in fits.segments.fits.iter().zip(&lifetime.segments.fits) {
            assert_eq!(a.fit.posterior, b.posterior);
            assert_eq!(a.pareto_k.len(), 2_000);
        }
        let totals = line_totals(&fits.cdr);
        let r = spearman(&totals[0], &totals[1]);
        assert!(r.abs() < 4.5 / 2_000f64.sqrt(), "rank correlation {r}");
        assert_eq!(fits.to_long().n_rows(), 20);
        let t = fits.totals();
        assert_eq!(t.n_rows(), 2);
        assert!(t.column("flagged_share").is_some());
        assert!(t.column("cdr_std_dev").is_some());
        let b = fits.segment(&[("lob", "b")]).unwrap();
        assert_eq!(b.cdr.n_components(), 10);

        let correlated = BayesOdp {
            dependence: SegmentDependence::RankCorrelation {
                spearman: vec![1.0, 0.7, 0.7, 1.0],
            },
            ..quick(9)
        };
        let fits = correlated
            .one_year_segments(&tri, "values", &settings)
            .unwrap();
        let totals = line_totals(&fits.cdr);
        let r = spearman(&totals[0], &totals[1]);
        assert!((r - 0.7).abs() < 0.05, "rank correlation {r}");
    }

    /// `BayesOneYear::max_refits` applies to each segment: each refits the
    /// first simulations due for a refit within its own budget.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_refit_budget_applies_to_each_segment() {
        let model = BayesOdp {
            n_sims: 200,
            ..quick(9)
        };
        let settings = |max_refits| BayesOneYear {
            refit: Some(OneYearRefit::RefitFlagged),
            max_refits,
            ..BayesOneYear::default()
        };
        let all = model
            .one_year_segments(&twins(), "values", &settings(None))
            .unwrap();
        let one = model
            .one_year_segments(&twins(), "values", &settings(Some(1)))
            .unwrap();
        assert_eq!(one.segments.fits.len(), 2);
        for (a, b) in all.segments.fits.iter().zip(&one.segments.fits) {
            assert!(a.refitted.len() >= 2, "{:?}", a.refitted);
            assert_eq!(b.refitted, a.refitted[..1]);
            assert_eq!(b.unrefitted, a.refitted[1..]);
        }
    }

    /// An FNV-1a hash of the bits of `values`, to pin a run exactly.
    fn bits_hash(values: &[f64]) -> u64 {
        values.iter().fold(0xcbf2_9ce4_8422_2325, |h, v| {
            (h ^ v.to_bits()).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Seed-pinned one-year views, bit for bit: a change to the engine
    /// (decision 17) must leave them unchanged. The hashes were captured on
    /// Windows x86_64 (MSVC); another platform's `exp` and `ln` round
    /// differently in the last bit, NUTS trajectories then diverge, and the
    /// draws differ (Ubuntu CI fails it). So it guards a refactor on the
    /// machine that captured it and is not run by CI: run it with
    /// `--ignored` there, before and after the change.
    #[test]
    #[ignore = "platform-specific pins (Windows x86_64); a local refactor guard, run with --ignored"]
    fn the_one_year_view_is_pinned_by_seed() {
        let only = quick(4)
            .one_year(
                &genins(),
                "values",
                &BayesOneYear {
                    refit: Some(OneYearRefit::ReweightOnly),
                    ..BayesOneYear::default()
                },
            )
            .unwrap();
        let all = BayesOdp {
            n_sims: 40,
            ..quick(5)
        }
        .one_year(
            &genins(),
            "values",
            &BayesOneYear {
                refit: Some(OneYearRefit::RefitAll),
                ..BayesOneYear::default()
            },
        )
        .unwrap();
        // Captured from the engine before decision 17 (commit aa6cbe6).
        let pins = [
            (
                &only,
                [
                    0x4cc3_6ede_8d24_e4c1,
                    0x60f8_b2e8_2cfa_9edf,
                    0xc865_9e9b_1ef3_bb95,
                ],
                [0.0, 93132.50520453123, 100440.25771184723],
                [0.18344198587327784, 0.5487737326308497, 1.1487178720844788],
            ),
            (
                &all,
                [
                    0xbcd8_cd4c_efd7_772d,
                    0x0803_8c32_e92d_9604,
                    0xf02b_6074_1996_c9cf,
                ],
                [0.0, 41944.76182826993, 51877.40122388203],
                [0.152359561840937, 0.39860461448610535, 0.2447379364603276],
            ),
        ];
        for (view, hashes, first, k) in pins {
            let got = [
                bits_hash(view.cdr.draw_matrix()),
                bits_hash(&view.pareto_k),
                bits_hash(&view.opening_reserve),
            ];
            assert_eq!(got, hashes, "{got:#x?}");
            assert_eq!(view.cdr.draw_matrix()[10..13], first);
            assert_eq!(view.pareto_k[..3], k);
        }
        let segments = quick(9)
            .one_year_segments(
                &twins(),
                "values",
                &BayesOneYear {
                    refit: Some(OneYearRefit::ReweightOnly),
                    ..BayesOneYear::default()
                },
            )
            .unwrap();
        let k: Vec<f64> = segments
            .segments
            .fits
            .iter()
            .flat_map(|s| s.pareto_k.clone())
            .collect();
        assert_eq!(
            [bits_hash(segments.cdr.draw_matrix()), bits_hash(&k)],
            [0x1b86_9a1b_102e_2965, 0xa555_8ff3_9fe3_7baf]
        );
    }

    /// The one-year conformance suite (decision 17) on GenIns, with a
    /// fully developed square as the triangle without a year.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_one_year_view_conforms() {
        let model = quick(4);
        let square = annual(
            2001,
            &[
                &[357.0, 1124.0, 1735.0, 2218.0],
                &[352.0, 1236.0, 2170.0, 3353.0],
                &[290.0, 1292.0, 2219.0, 3235.0],
                &[310.0, 1418.0, 2195.0, 3757.0],
            ],
        );
        let fitted: Vec<(Triangle, Segment, BayesOdpSegment)> = [genins(), square]
            .into_iter()
            .map(|tri| {
                let segment = tri.segment("values").unwrap();
                let (fit, _) = model
                    .fit_segment(&segment, "values", Stream::Chains, Stream::Predictive)
                    .unwrap();
                (tri, segment, fit)
            })
            .collect();
        let cells: Vec<OdpRefit> = fitted
            .iter()
            .map(|(_, _, fit)| OdpRefit::new(&model, fit))
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
            "BayesOdp",
            &years[0],
            &fitted[0].2.posterior,
            &years[1],
            &fitted[1].2.posterior,
        );
    }

    /// A year whose log likelihood is not a number in some posterior draw
    /// fails its simulation, reported as a failed refit is, where PSIS
    /// cannot smooth the ratios (decision 16, "PSIS").
    #[test]
    fn a_log_likelihood_that_is_not_a_number_is_an_error() {
        use crate::bayes::one_year::YearModel;

        /// `OdpYear` with the year's log likelihood NaN at one draw.
        struct NanAt<'a, M>(&'a M, Vec<f64>);
        impl<M: YearModel> YearModel for NanAt<'_, M> {
            fn n_origins(&self) -> usize {
                self.0.n_origins()
            }
            fn n_year(&self) -> usize {
                self.0.n_year()
            }
            fn draw_year(&self, theta: &[f64], rng: &mut StreamRng, year: &mut [f64]) {
                self.0.draw_year(theta, rng, year);
            }
            fn year_log_likelihood(&self, theta: &[f64], year: &[f64]) -> f64 {
                if theta == self.1.as_slice() {
                    f64::NAN
                } else {
                    self.0.year_log_likelihood(theta, year)
                }
            }
            fn paid(&self, year: &[f64], paid: &mut [f64]) {
                self.0.paid(year, paid);
            }
            fn opening(&self, theta: &[f64], reserve: &mut [f64]) {
                self.0.opening(theta, reserve);
            }
            fn closing(&self, theta: &[f64], year: &[f64], reserve: &mut [f64]) {
                self.0.closing(theta, year, reserve);
            }
            fn refit(&self, year: &[f64], seed: u64) -> Result<(PosteriorDraws, bool)> {
                self.0.refit(year, seed)
            }
            fn log_posterior(&self, theta: &[f64], year: &[f64]) -> f64 {
                self.0.log_posterior(theta, year)
            }
        }

        let model = quick(4);
        let tri = genins();
        let segment = tri.segment("values").unwrap();
        let (fit, _) = model
            .fit_segment(&segment, "values", Stream::Chains, Stream::Predictive)
            .unwrap();
        let cells = OdpRefit::new(&model, &fit);
        let year = OdpYear::new(
            &cells,
            model.process,
            &tri,
            &segment,
            &fit.chain_ladder.latest_position,
        );
        let poisoned = NanAt(&year, fit.posterior.draw(7).to_vec());
        let e = one_year::simulate(&poisoned, &fit.posterior, &BayesOneYear::default(), 20, 1)
            .unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.contains("in 20 of 20 simulations")
                && m.contains("not a number")),
            "{e}"
        );
    }
}
