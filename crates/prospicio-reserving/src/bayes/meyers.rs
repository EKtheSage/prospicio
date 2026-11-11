//! Meyers' correlated chain ladder (CRC), changing settlement rate (CSR) and
//! correlated accident year (CAY) models on cumulative values with premium
//! (`docs/design/reserving-bayes.md`, decision 7), over segments (decision
//! 11), in the diagonal backtest (decision 8) and in the one-year view
//! (decision 19, [`year`]).

mod backtest;
mod model;
mod results;
#[cfg(test)]
pub(crate) mod tests;
mod year;

use std::collections::BTreeMap;

use prospicio_bayes::elpd::Loo;
use prospicio_bayes::nuts::{ParamSummary, PosteriorDraws, Sampler, sample};
use prospicio_core::{Lag, Period, StreamRng};
use prospicio_math::linalg::{cholesky, cholesky_solve};
use prospicio_math::special::norm_quantile;
use prospicio_models::Column;
use prospicio_prob::{InputHasher, KeyValue, PredictiveDistribution, Provenance};

use super::laplace::{Posterior, Whitened};
use super::new_year::{self, NewYear};
use super::odp::log_mean_exp;
use super::one_year::{
    self, BayesOneYear, BayesOneYearFit, BayesOneYearFits, BayesOneYearSegment, YearDraws,
};
use super::{Diagnostics, Stream, derive_seed, one_year_seed, pilot_seed, psis_loo};
use crate::backtest::{CellForecast, TriangleModel};
use crate::bootstrap::{
    component_sums, hash_segment, origin_keys, pick_segment, push_combined, push_moments,
    segment_sums,
};
use crate::chain_ladder::{ChainLadder, ChainLadderFit};
use crate::dependence::SegmentDependence;
use crate::error::{Error, Result};
use crate::expected_loss::origin_exposure;
use crate::frame::TriangleFrame;
use crate::pattern_draws::{DevelopmentPatterns, predictive_with_shares};
use crate::segments::{FitTable, ReserveFit, SegmentFits, fit_each_with_exposure};
use crate::triangle::{Label, Segment, Triangle};

pub use results::{MeyersFit, MeyersFits, MeyersSegment};

use backtest::Rebuilt;
use model::{Cell, Model, S_FLOOR, logistic};

/// Which of Meyers' models [`Meyers`] fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeyersModel {
    /// The correlated chain ladder on cumulative values (Meyers 2019,
    /// sec. 5), without its correlation between origins: CRC.
    #[default]
    Crc,
    /// The changing settlement rate model (Meyers 2019, sec. 7): CRC with
    /// each development effect scaled by `(1 - gamma)^(w - 1)`.
    Csr,
    /// The correlated accident year model (Meyers 2019, sec. 8): CRC with
    /// `rho` times the previous origin's residual at the same age added to
    /// each mean, which makes neighbouring origins correlated. Meyers fits
    /// it to incurred data.
    Cay,
}

impl MeyersModel {
    /// `crc`, `csr` or `cay`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Crc => "crc",
            Self::Csr => "csr",
            Self::Cay => "cay",
        }
    }
}

/// The level of a new origin in [`MeyersFit::new_year`]: how its
/// `logelr + alpha` is drawn from a posterior draw (`reserving-bayes.md`
/// decision 28). Meyers' models give each origin its own effect `alpha_w`
/// with a vague prior and no hyperparameter, so the posterior alone says
/// nothing about the effect of an origin with no data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NewOriginLevel {
    /// The new origin is exchangeable with the triangle's: its level is a
    /// draw from the posterior predictive of a normal fitted to the draw's
    /// `n` levels `l_w = logelr + alpha_w` (`alpha_1 = 0`) under the
    /// reference prior on its mean and variance, a Student t with `n - 1`
    /// degrees of freedom, location their mean and scale their sd times
    /// `sqrt(1 + 1 / n)`. The level's spread is the triangle's
    /// origin-to-origin spread, with its estimation error. Needs at least
    /// three origins.
    #[default]
    Exchangeable,
    /// The model's own prior: `logelr + alpha`, `alpha ~ N(0, origin_sd^2)`,
    /// the literal posterior predictive of Meyers' model. Under Meyers'
    /// `origin_sd` of 3.162 this is too wide to price on (a log sd of
    /// 3.16); it is useful with an `origin_sd` set to the believed
    /// origin-to-origin spread.
    Prior,
}

impl NewOriginLevel {
    /// `exchangeable` or `prior`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Exchangeable => "exchangeable",
            Self::Prior => "prior",
        }
    }
}

/// Meyers' (2019, CAS Monograph 8) log-normal models of cumulative values
/// `C_wd` with premium `P_w`, for origins `w = 1..n` and ages `d = 1..m`:
///
/// ```text
/// mu_wd = log P_w + logelr + alpha_w + beta_d g_w + rho r_(w-1)d,   alpha_1 = 0,  beta_m = 0
/// g_w   = 1 (CRC, CAY)  or  (1 - gamma)^(w - 1) (CSR)
/// r_wd  = log C_wd - mu_wd,   rho = 0 (CRC, CSR),   r_0d = 0
/// sigma_d^2 = s_d + s_(d+1) + ... + s_m,   s_i ~ U(1 - exp(-1e-5), 1)
/// log C_wd ~ N(mu_wd, sigma_d^2)
/// logelr ~ N(-0.4, 3.162^2),  alpha_w, beta_d ~ N(0, 3.162^2),  gamma ~ N(0, 0.05^2)
/// (1 - rho) / 2 ~ Beta(2, 2)
/// ```
///
/// with the priors of Meyers' Stan code (`CRC.R`, `CSR.R` and `CAY.R` of
/// the CAS appendix; 3.162 is his rounding of `sqrt(10)`). In CAY each
/// mean takes `rho` times the previous origin's residual at the same age,
/// itself a residual from its own (recursive) mean, so `log C_wd` is a
/// moving average of order 1 down each age's column: the correlation of
/// neighbouring origins is `rho / (1 + rho^2)`, of origins further apart 0
/// (Meyers 2019, eq. 8.3). He draws
/// `a_i ~ InvGamma(1, 1)` bounded by `1e5` and sets
/// `s_i = 1 - exp(-1 / a_i)`, which is uniform on `(1 - exp(-1e-5), 1)`:
/// the floor of about `1e-5` keeps each `sigma_d` at least
/// `sqrt((m - d + 1) 1e-5)` and matters, since without it the late
/// `sigma_d` fall about 15% lower on incurred data. The ultimate is the
/// value at the oldest age `m` (`beta_m = 0`; no tail): given a posterior
/// draw, each origin's ultimate is a log-normal draw with mean
/// `log P_w + logelr + alpha_w` (CAY: plus `rho` times the previous
/// origin's residual at age `m`, observed or drawn, origin by origin) and
/// sd `sigma_m`, and an origin already observed at age `m` keeps its
/// observed value, as Meyers computes it. Ages are independent given the
/// parameters (in CRC and CSR every cell is), so the ultimate is drawn
/// from age `m` alone, and the reserve (ultimate less the latest value of
/// the fitted column: an outstanding reserve on paid data, IBNR on
/// incurred data) can be negative.
///
/// The model samples `s_i` through a logit,
/// `s_i = s0 + (1 - s0) / (1 + exp(-u_i))` with `s0 = 1 - exp(-1e-5)`, and
/// the log Jacobian so that the prior on `s_i` is exactly Meyers'; the
/// posterior is the same as his. CAY samples `rho = tanh(v)`, with the log
/// Jacobian, so that `(1 - rho) / 2` has Meyers' `Beta(2, 2)` prior. NUTS
/// samples a dense whitened scale: the
/// joint mode lies deep in the funnel of the late `sigma_d`, far from the
/// posterior's bulk, so the scale is not a Laplace approximation (as for
/// [`BayesOdp`](super::BayesOdp)) but the mean and covariance of a pilot
/// run of `tune` warm-up and `max(tune, 100)` kept draws per chain, itself
/// on a scale whitened at a least-squares start (as Stan's dense metric
/// adapts). The pilot's draws are discarded.
///
/// Every observed cell must be positive (Meyers' code silently raises
/// values below 1 to 1; here a non-positive cell is an error). CAY also
/// needs, for each observed cell after the first origin, the previous
/// origin's cell at the same age.
///
/// [`fit_segments`](Self::fit_segments) fits every segment of a triangle
/// into one joint distribution, and `Meyers` takes part in the diagonal
/// backtest through [`TriangleModel`] (see there for how it reads the
/// incremental cells).
///
/// ```
/// use prospicio_reserving::bayes::{Meyers, MeyersModel};
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// use prospicio_bayes::nuts::Sampler;
/// use prospicio_prob::Distribution;
///
/// // Six origins of cumulative paid, each with its premium.
/// let paid: [&[f64]; 6] = [
///     &[952.0, 1529.0, 2813.0, 3647.0, 3724.0, 3832.0],
///     &[849.0, 1564.0, 2202.0, 2432.0, 2468.0],
///     &[983.0, 2211.0, 2830.0, 3832.0],
///     &[1657.0, 2685.0, 3169.0],
///     &[932.0, 1940.0],
///     &[1162.0],
/// ];
/// let premium = [5812.0, 4908.0, 5454.0, 5165.0, 5214.0, 5230.0];
/// let (mut origin, mut age, mut values, mut prem) = (vec![], vec![], vec![], vec![]);
/// for (w, row) in paid.iter().enumerate() {
///     for (d, &v) in row.iter().enumerate() {
///         origin.push(Month::january(1988 + w as i32));
///         age.push(12 * (d as u32 + 1));
///         values.push(v);
///         prem.push(premium[w]);
///     }
/// }
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&age),
///     values: &[("paid", &values), ("premium", &prem)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let model = Meyers {
///     model: MeyersModel::Csr,
///     sampler: Sampler { tune: 300, draws: 300, ..Meyers::default().sampler },
///     n_sims: 2_000,
///     seed: 1,
///     ..Meyers::default()
/// };
/// let fit = model.fit(&tri, "paid", "premium")?;
/// assert_eq!(fit.reserves.n_components(), 6);
/// assert!(fit.ultimates.total().mean() > fit.chain_ladder.latest.iter().sum::<f64>());
/// assert_eq!(fit.posterior.names()[0], "logelr");
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Meyers {
    /// CRC, CSR or CAY.
    pub model: MeyersModel,
    /// Mean of the normal prior on `logelr`, the log expected loss ratio.
    pub logelr_mean: f64,
    /// Standard deviation of the normal prior on `logelr`.
    pub logelr_sd: f64,
    /// Standard deviation of the normal prior on each origin effect
    /// `alpha_w`, `w >= 2`, centred on 0.
    pub origin_sd: f64,
    /// Standard deviation of the normal prior on each development effect
    /// `beta_d`, `d < m`, centred on 0.
    pub development_sd: f64,
    /// Standard deviation of the normal prior on `gamma`, centred on 0
    /// (CSR only).
    pub gamma_sd: f64,
    /// Both shape parameters of the symmetric beta prior on
    /// `(1 - rho) / 2` (CAY only; Meyers' 2 makes the prior density of
    /// `rho` proportional to `1 - rho^2`).
    pub rho_shape: f64,
    /// NUTS settings. Its `seed` is not used: the chains take a seed
    /// derived from [`seed`](Self::seed) (module docs, "Seeds").
    pub sampler: Sampler,
    /// Number of predictive simulations, independent of the number of
    /// posterior draws. The one-year view runs at most
    /// [`ONE_YEAR_SIMS`](Self::ONE_YEAR_SIMS) of them unless its settings'
    /// [`n_sims`](BayesOneYear::n_sims) says otherwise.
    pub n_sims: usize,
    /// Seed of the chains and of the simulations, each through its own
    /// derived seed (module docs, "Seeds").
    pub seed: u64,
    /// How segments depend on each other in a fit of several segments
    /// ([`fit_segments`](Self::fit_segments), decision 11; `Synchronized` is
    /// an error there); [`fit`](Self::fit) has one segment and does not
    /// use it.
    pub dependence: SegmentDependence,
}

impl Default for Meyers {
    /// CRC with Meyers' priors (`logelr ~ N(-0.4, 3.162^2)`, origin and
    /// development effects `N(0, 3.162^2)`, `gamma ~ N(0, 0.05^2)`,
    /// `(1 - rho) / 2 ~ Beta(2, 2)`), four
    /// chains of 1,000 warm-up and 2,000 kept draws with target acceptance
    /// 0.95, 10,000 simulations, seed 0, independent segments. On Meyers'
    /// group 353 data, at
    /// target acceptance 0.8 the paid fits gave 1 to 56 divergent
    /// transitions at nearly every one of 30 seeds (the late `sigma_d`,
    /// near 0.02, make a strongly curved posterior), and with 1,000 kept
    /// draws the late `sigma_d` had an R-hat above 1.01 (at most 1.016) in
    /// 5 of 120 fits; with these defaults all 120 pass every threshold.
    fn default() -> Self {
        Self {
            model: MeyersModel::Crc,
            logelr_mean: -0.4,
            logelr_sd: 3.162,
            origin_sd: 3.162,
            development_sd: 3.162,
            gamma_sd: 0.05,
            rho_shape: 2.0,
            sampler: Sampler {
                target_accept: 0.95,
                draws: 2000,
                ..Sampler::default()
            },
            n_sims: 10_000,
            seed: 0,
            dependence: SegmentDependence::Independent,
        }
    }
}

/// Fewest kept draws per chain of the pilot run.
const MIN_PILOT_DRAWS: usize = 100;

impl Meyers {
    /// The most one-year simulations [`one_year`](Self::one_year) and
    /// [`one_year_segments`](Self::one_year_segments) run by default
    /// (`BayesOneYear::n_sims` `None`): 2,000, against the 10,000 of the
    /// lifetime view, since nearly every simulation of these models is
    /// flagged and costs a moment match or a NUTS refit (about 249 s at
    /// 10,000 for CRC on group 353 in release on 16 threads; the user's
    /// decision of 2026-10-16, `docs/design/reserving-bayes.md` decision
    /// 19). A model whose [`n_sims`](Self::n_sims) is smaller runs that many.
    pub const ONE_YEAR_SIMS: usize = 2_000;

    /// Fits `column` of a single-segment cumulative triangle with premium
    /// from its `premium` column (the latest value per origin, as
    /// [`CapeCod`](crate::CapeCod) reads exposure). Every origin must be
    /// observed from the first age to its latest, every observed cell must
    /// be positive, and there must be more observed cells than parameters.
    /// Failed diagnostics do not fail the fit: they are in
    /// [`MeyersFit::diagnostics`].
    pub fn fit(&self, triangle: &Triangle, column: &str, premium: &str) -> Result<MeyersFit> {
        self.validate()?;
        let segment = triangle.segment(column)?;
        let premium_segment = triangle.segment(premium)?;
        let (fit, ultimates, reserves, patterns) = self.fit_segment(
            &segment,
            &premium_segment,
            column,
            premium,
            Stream::Chains,
            Stream::Predictive,
        )?;
        Ok(fit.into_fit(ultimates, reserves, patterns))
    }

    /// The one-year view of `column` of a single-segment cumulative
    /// triangle with premium from its `premium` column: the claims
    /// development result over the twelve months after the valuation, by
    /// reweighting the posterior with Pareto smoothed importance sampling
    /// (decisions 16 to 19 of `docs/design/reserving-bayes.md`).
    ///
    /// The posterior and the lifetime ultimates and reserves are
    /// [`fit`](Self::fit)'s, draw for draw. Each of the one-year
    /// simulations (the settings' [`n_sims`](BayesOneYear::n_sims), by
    /// default [`n_sims`](Self::n_sims) capped at
    /// [`ONE_YEAR_SIMS`](Self::ONE_YEAR_SIMS), 2,000) picks a posterior draw
    /// `θ`, draws the cumulative values `C` of the year's cells (the cells valued in the twelve months, as
    /// in [`OdpBootstrap::one_year`](crate::OdpBootstrap::one_year); one per
    /// open origin with an annual grain) from the model given `θ`, weights
    /// every posterior draw by the likelihood of those values given the
    /// observed ones and smooths the weights (Pareto `k̂` per simulation).
    /// Origin `w`'s claims development result is
    ///
    /// ```text
    /// CDR_w = E[U_w | D0] - E[U_w | D0, C]
    /// ```
    ///
    /// with `U_w` its ultimate (the value at the oldest age) and each
    /// expectation the (weighted) posterior mean of `E[U_w | θ]`, which is
    /// `exp(mu_wm + sigma_m^2 / 2)`; CAY's mean adds `rho` times the
    /// previous origin's residual at the oldest age where that cell is
    /// observed by the end of the year, and its variance is
    /// `(1 + rho^2) sigma_m^2` where it is not. An origin the year takes to
    /// the oldest age has `E[U_w | D0] - C_wm` as its claims development
    /// result, one already there has none. The opening reserve,
    /// `E[U_w | D0]` less the latest value, is the posterior mean of the
    /// lifetime reserve. A flagged simulation (`k̂` above the threshold)
    /// is moment-matched and, if that cannot rescue it, refitted with NUTS
    /// on the triangle extended by its year, as [`BayesOneYear::refit`]
    /// says.
    ///
    /// The draws are reweighted on the unconstrained scale NUTS samples
    /// (`u_i` for the steps of `sigma_d^2`, `atanh(rho)` for CAY), which
    /// the fit keeps beside its natural-scale
    /// [`posterior`](MeyersFit::posterior). In CAY each year cell's
    /// previous origin is observed or in the year at the same age (it is
    /// valued earlier), so the year's residuals follow from the draw.
    ///
    /// ```
    /// use prospicio_bayes::nuts::Sampler;
    /// use prospicio_reserving::bayes::{BayesOneYear, Meyers, OneYearRefit};
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
    /// use prospicio_prob::Distribution;
    ///
    /// let paid: [&[f64]; 6] = [
    ///     &[952.0, 1529.0, 2813.0, 3647.0, 3724.0, 3832.0],
    ///     &[849.0, 1564.0, 2202.0, 2432.0, 2468.0],
    ///     &[983.0, 2211.0, 2830.0, 3832.0],
    ///     &[1657.0, 2685.0, 3169.0],
    ///     &[932.0, 1940.0],
    ///     &[1162.0],
    /// ];
    /// let premium = [5812.0, 4908.0, 5454.0, 5165.0, 5214.0, 5230.0];
    /// let (mut origin, mut age, mut values, mut prem) = (vec![], vec![], vec![], vec![]);
    /// for (w, row) in paid.iter().enumerate() {
    ///     for (d, &v) in row.iter().enumerate() {
    ///         origin.push(Month::january(1988 + w as i32));
    ///         age.push(12 * (d as u32 + 1));
    ///         values.push(v);
    ///         prem.push(premium[w]);
    ///     }
    /// }
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&age),
    ///     values: &[("paid", &values), ("premium", &prem)],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })?;
    /// let model = Meyers {
    ///     sampler: Sampler { tune: 300, draws: 300, ..Meyers::default().sampler },
    ///     n_sims: 500,
    ///     ..Meyers::default()
    /// };
    /// // Reweighting only, for speed: the default moment-matches and refits
    /// // the flagged simulations, nearly all of them for this model.
    /// let only = BayesOneYear { refit: Some(OneYearRefit::ReweightOnly), ..BayesOneYear::default() };
    /// let view = model.one_year(&tri, "paid", "premium", &only)?;
    /// assert_eq!(view.cdr.n_components(), 6);
    /// // 1988 is fully developed: no cell in the year, no change.
    /// assert!(view.cdr.draw_matrix().chunks(6).all(|row| row[0] == 0.0));
    /// // The one-year view is narrower than the lifetime view.
    /// assert!(view.cdr.std_dev() < view.fit.reserves.std_dev());
    /// # Ok::<(), prospicio_reserving::Error>(())
    /// ```
    pub fn one_year(
        &self,
        triangle: &Triangle,
        column: &str,
        premium: &str,
        settings: &BayesOneYear,
    ) -> Result<BayesOneYearFit<MeyersFit>> {
        self.validate()?;
        settings.validate()?;
        let segment = triangle.segment(column)?;
        let premium_segment = triangle.segment(premium)?;
        let (fit, ultimates, reserves, patterns) = self.fit_segment(
            &segment,
            &premium_segment,
            column,
            premium,
            Stream::Chains,
            Stream::Predictive,
        )?;
        let year = self.year_draws(triangle, &segment, &fit, Stream::Predictive, settings)?;
        let mut hasher = InputHasher::new();
        hasher.str(column);
        hasher.str(premium);
        hash_segment(&mut hasher, &segment, &fit.chain_ladder, &segment.ages);
        hasher.f64s(&fit.premium);
        let provenance = self.one_year_provenance(column, premium, hasher, settings);
        one_year::assemble(
            fit.into_fit(ultimates, reserves, patterns),
            year,
            &segment,
            settings,
            provenance,
        )
    }

    /// The one-year view of `column` in every segment of a cumulative
    /// triangle, with premium from the same segment's `premium` column,
    /// each segment's posterior and simulations as in
    /// [`fit_segments`](Self::fit_segments) (segment `k`'s own seeds) and
    /// its view as in [`one_year`](Self::one_year), into one joint claims
    /// development result, the segments depending on each other as
    /// [`dependence`](Self::dependence) says: `Independent` as drawn,
    /// `RankCorrelation` by Iman–Conover on the segments' total claims
    /// development results, as the bootstraps' one-year view does;
    /// `Synchronized` is an [`Error::NotSupported`]. A failure names its segment.
    pub fn one_year_segments(
        &self,
        triangle: &Triangle,
        column: &str,
        premium: &str,
        settings: &BayesOneYear,
    ) -> Result<BayesOneYearFits<MeyersSegment>> {
        self.validate()?;
        settings.validate()?;
        if self.dependence == SegmentDependence::Synchronized {
            return Err(super::synchronized_error());
        }
        self.dependence.validate(triangle.index().len())?;
        let mut k = 0;
        let fitted = fit_each_with_exposure(triangle, column, premium, |segment, prem| {
            let predictive = Stream::SegmentPredictive(k);
            let fit = self
                .fit_segment(
                    segment,
                    prem,
                    column,
                    premium,
                    Stream::SegmentChains(k),
                    predictive,
                )
                .and_then(|(fit, _, _, _)| {
                    let year = self.year_draws(triangle, segment, &fit, predictive, settings)?;
                    Ok((fit, year))
                });
            k += 1;
            fit
        })?;

        let mut hasher = InputHasher::new();
        hasher.str(column);
        hasher.str(premium);
        let mut dims = fitted.key_names.clone();
        dims.push("origin".into());
        let mut components = Vec::new();
        for ((label, (fit, _)), segment) in fitted.iter().zip(triangle.segments(column)?) {
            hasher.str(&label.to_string());
            hash_segment(&mut hasher, &segment, &fit.chain_ladder, &segment.ages);
            hasher.f64s(&fit.premium);
            components.extend(origin_keys(label, &segment.origins));
        }
        let n_sims = self.one_year_sims(settings);
        let mut draws = Vec::with_capacity(n_sims * components.len());
        for i in 0..n_sims {
            for (_, year) in &fitted.fits {
                let m = year.opening.len();
                draws.extend_from_slice(&year.cdr[i * m..(i + 1) * m]);
            }
        }
        let provenance = self
            .one_year_provenance(column, premium, hasher, settings)
            .param("segments", fitted.len())
            .param("dependence", format!("{:?}", self.dependence));
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

    /// The one-year simulations of one fitted segment, from the seed the
    /// segment's `predictive` stream derives (module docs, "Seeds").
    fn year_draws(
        &self,
        triangle: &Triangle,
        segment: &Segment,
        fit: &MeyersSegment,
        predictive: Stream,
        settings: &BayesOneYear,
    ) -> Result<YearDraws> {
        let year = year::MeyersYear::new(
            self,
            &fit.model,
            triangle,
            segment,
            &fit.chain_ladder,
            &fit.premium,
        )?;
        one_year::simulate(
            &year,
            &fit.sampled,
            settings,
            self.one_year_sims(settings),
            one_year_seed(derive_seed(self.seed, predictive)),
        )
    }

    /// The number of one-year simulations under `settings`: their
    /// [`n_sims`](BayesOneYear::n_sims), or this model's
    /// [`n_sims`](Self::n_sims) capped at [`ONE_YEAR_SIMS`](Self::ONE_YEAR_SIMS).
    fn one_year_sims(&self, settings: &BayesOneYear) -> usize {
        settings.sims(self.n_sims.min(Self::ONE_YEAR_SIMS))
    }

    fn one_year_provenance(
        &self,
        column: &str,
        premium: &str,
        hasher: InputHasher,
        settings: &BayesOneYear,
    ) -> Provenance {
        let mut provenance = self.provenance(column, premium, hasher);
        provenance.model = format!("meyers_{}_one_year", self.model.name());
        one_year::provenance(provenance, settings, self.one_year_sims(settings))
    }

    /// Fits `column` in every segment of a cumulative triangle, with
    /// premium from the same segment's `premium` column, each with its own
    /// posterior (the same model and settings, sampled independently), into
    /// joint distributions of the ultimates and the reserves, the segments
    /// depending on each other as [`dependence`](Self::dependence) says
    /// (decision 11), as in
    /// [`BayesOdp::fit_segments`](super::BayesOdp::fit_segments):
    ///
    /// * [`Independent`](SegmentDependence::Independent): each segment's
    ///   simulations as drawn. Segment `k` (in index order) takes its own
    ///   derived seeds for its chains (and its pilot run) and its
    ///   simulations (module docs, "Seeds"), so the segments share no
    ///   random numbers; a segment's draws therefore differ from fitting it
    ///   alone.
    /// * [`RankCorrelation`](SegmentDependence::RankCorrelation): the
    ///   segments' simulations reordered as whole rows by Iman–Conover on
    ///   the segment totals of the reserves, exactly as in the bootstraps;
    ///   each segment's ultimates move with its reserves, row by row.
    /// * [`Synchronized`](SegmentDependence::Synchronized) is an
    ///   [`Error::NotSupported`]: it draws the residuals of the same positions in
    ///   every segment, and a posterior has no residuals to share.
    ///
    /// A bad rank-correlation matrix fails before any sampling, and a
    /// failure names its segment.
    pub fn fit_segments(
        &self,
        triangle: &Triangle,
        column: &str,
        premium: &str,
    ) -> Result<MeyersFits> {
        self.validate()?;
        if self.dependence == SegmentDependence::Synchronized {
            return Err(super::synchronized_error());
        }
        self.dependence.validate(triangle.index().len())?;
        let mut k = 0;
        let fitted = fit_each_with_exposure(triangle, column, premium, |segment, prem| {
            let fit = self.fit_segment(
                segment,
                prem,
                column,
                premium,
                Stream::SegmentChains(k),
                Stream::SegmentPredictive(k),
            );
            k += 1;
            fit
        })?;

        let mut hasher = InputHasher::new();
        hasher.str(column);
        hasher.str(premium);
        let mut dims = fitted.key_names.clone();
        dims.push("origin".into());
        let mut components = Vec::new();
        for ((label, (fit, _, _, _)), segment) in fitted.iter().zip(triangle.segments(column)?) {
            hasher.str(&label.to_string());
            hash_segment(&mut hasher, &segment, &fit.chain_ladder, &segment.ages);
            hasher.f64s(&fit.premium);
            components.extend(origin_keys(label, &segment.origins));
        }
        let n = components.len();
        // Each simulation's row holds every segment's reserves and then its
        // ultimates, so the rank correlation moves both together: a
        // segment's total over its range is twice its total reserve plus
        // its total latest value, which ranks the simulations as the
        // reserves do.
        let mut ranges = Vec::with_capacity(fitted.len());
        let mut start = 0;
        for (fit, _, _, _) in &fitted.fits {
            let m = fit.chain_ladder.origins.len();
            ranges.push(start..start + 2 * m);
            start += 2 * m;
        }
        let mut both = Vec::with_capacity(self.n_sims * 2 * n);
        for i in 0..self.n_sims {
            for (_, ultimates, reserves, _) in &fitted.fits {
                let m = reserves.n_components();
                both.extend_from_slice(&reserves.draw_matrix()[i * m..(i + 1) * m]);
                both.extend_from_slice(&ultimates.draw_matrix()[i * m..(i + 1) * m]);
            }
        }
        let list = |f: fn(&MeyersSegment) -> f64| {
            format!(
                "{:?}",
                fitted
                    .fits
                    .iter()
                    .map(|(s, _, _, _)| f(s))
                    .collect::<Vec<_>>()
            )
        };
        let provenance = self
            .provenance(column, premium, hasher)
            .param("premium_total", list(|s| s.premium.iter().sum()))
            .param("segments", fitted.len())
            .param("dependence", format!("{:?}", self.dependence))
            .param(
                "posterior_draws",
                fitted
                    .fits
                    .first()
                    .map_or(0, |(s, _, _, _)| s.posterior.n_draws()),
            );
        let both = PredictiveDistribution::from_draws(
            vec!["column".into()],
            (0..2 * n).map(|j| vec![KeyValue::Int(j as i64)]).collect(),
            both,
            provenance,
        )?;
        let mut patterns = DevelopmentPatterns::joined(
            &fitted.labels,
            fitted.fits.iter().map(|(_, _, _, p)| p.clone()).collect(),
        );
        let both = self
            .dependence
            .reorder_ranges_with(both, &ranges, self.seed, &mut patterns)?;
        let (mut reserve_draws, mut ultimate_draws) = (
            Vec::with_capacity(self.n_sims * n),
            Vec::with_capacity(self.n_sims * n),
        );
        for row in both.draw_matrix().chunks_exact(2 * n) {
            for range in &ranges {
                let half = range.start + range.len() / 2;
                reserve_draws.extend_from_slice(&row[range.start..half]);
                ultimate_draws.extend_from_slice(&row[half..range.end]);
            }
        }
        let provenance = both.provenance().clone();
        let reserves = PredictiveDistribution::from_draws(
            dims.clone(),
            components.clone(),
            reserve_draws,
            provenance.clone().param("output", "reserves"),
        )?;
        let ultimates = PredictiveDistribution::from_draws(
            dims,
            components,
            ultimate_draws,
            provenance.param("output", "ultimates"),
        )?;
        let segments = fitted.map(|(fit, _, _, _)| fit.clone());
        Ok(MeyersFits {
            segments,
            ultimates,
            reserves,
            patterns,
        })
    }

    /// The posterior of one segment, its ultimates and its reserves by
    /// origin and its development pattern in each simulation, with the
    /// seeds of `chains` (and its pilot run) and `predictive`.
    fn fit_segment(
        &self,
        segment: &Segment,
        premium_segment: &Segment,
        column: &str,
        premium: &str,
        chains: Stream,
        predictive: Stream,
    ) -> Result<(
        MeyersSegment,
        PredictiveDistribution,
        PredictiveDistribution,
        DevelopmentPatterns,
    )> {
        let chain_ladder = ChainLadder::default().fit_segment(segment, &segment.ages)?;
        let premiums = origin_exposure(segment, premium_segment, premium)?;
        let model = self.model_of(segment, &chain_ladder, &premiums)?;
        let (posterior, sampled, diagnostics) =
            self.sample(&model, derive_seed(self.seed, chains))?;

        let mut hasher = InputHasher::new();
        hasher.str(column);
        hasher.str(premium);
        hash_segment(&mut hasher, segment, &chain_ladder, &segment.ages);
        hasher.f64s(&premiums);
        let provenance = self.provenance(column, premium, hasher);
        let nd = segment.n_dev;
        let closed: Vec<bool> = chain_ladder
            .latest_position
            .iter()
            .map(|&d| d + 1 == nd)
            .collect();
        let latest = &chain_ladder.latest;
        let no = segment.n_origins;
        let (ultimates, patterns) = predictive_with_shares(
            &posterior,
            segment,
            self.n_sims,
            derive_seed(self.seed, predictive),
            provenance.param("output", "ultimates"),
            |nat, rng: &mut StreamRng, row, shares| {
                model.draw_ultimates(nat, &premiums, &closed, latest, rng, row);
                // No random number: the ultimates are what they were.
                model.pattern(nat, shares);
            },
        )?;
        let reserve_draws: Vec<f64> = ultimates
            .draw_matrix()
            .chunks_exact(no)
            .flat_map(|row| row.iter().zip(latest).map(|(u, l)| u - l))
            .collect();
        // The ultimates' provenance (seed, posterior draws) with the output
        // renamed.
        let mut provenance = ultimates.provenance().clone();
        for (key, value) in &mut provenance.parameters {
            if key == "output" {
                *value = "reserves".into();
            }
        }
        let reserves = PredictiveDistribution::from_draws(
            ultimates.dims().to_vec(),
            ultimates.components().to_vec(),
            reserve_draws,
            provenance,
        )?;
        Ok((
            MeyersSegment {
                chain_ladder,
                premium: premiums,
                posterior,
                diagnostics,
                model,
                sampled,
            },
            ultimates,
            reserves,
            patterns,
        ))
    }

    /// NUTS on `model` with the chains' seed `chains`: the scale comes from
    /// a pilot run (decision 7, "Sampling scale"), whitened roughly at the
    /// start point, then by the pilot draws' mean and covariance, since the
    /// joint mode lies deep in the funnel of the late `sigma_d`, far from
    /// the posterior's bulk. The draws on the natural scale, the same draws
    /// on the unconstrained scale `Model::log_posterior` takes (an affine
    /// map of the whitened one NUTS samples, so its density needs no
    /// Jacobian), and their diagnostics.
    fn sample(
        &self,
        model: &Model,
        chains: u64,
    ) -> Result<(PosteriorDraws, PosteriorDraws, Diagnostics)> {
        let p = model.dim();
        let start = model.start();
        let rough = Whitened::regularised(model, &start)?;
        let pilot = sample(
            &rough,
            &vec![0.0; p],
            Sampler {
                seed: pilot_seed(chains),
                draws: self.sampler.tune.max(MIN_PILOT_DRAWS),
                ..self.sampler
            },
        )?;
        let pilot = pilot.transform(model.names(), |z| rough.theta(z))?;
        let whitened = Whitened::from_draws(model, &pilot)?;
        let draws = sample(
            &whitened,
            &vec![0.0; p],
            Sampler {
                seed: chains,
                ..self.sampler
            },
        )?;
        let sampled = draws.transform(model.sampled_names(), |z| whitened.theta(z))?;
        let posterior = sampled.transform(model.names(), |theta| model.natural(theta))?;
        let summary = posterior.summary()?;
        let diagnostics = Diagnostics::new(
            &summary,
            posterior.chains(),
            posterior.divergences().iter().sum(),
        );
        Ok((posterior, sampled, diagnostics))
    }

    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("logelr_sd", self.logelr_sd),
            ("origin_sd", self.origin_sd),
            ("development_sd", self.development_sd),
            ("gamma_sd", self.gamma_sd),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::InvalidSetting {
                    name,
                    value,
                    expected: "a finite, positive prior standard deviation",
                });
            }
        }
        if !(self.rho_shape.is_finite() && self.rho_shape > 0.0) {
            return Err(Error::InvalidSetting {
                name: "rho_shape",
                value: self.rho_shape,
                expected: "a finite, positive beta shape",
            });
        }
        if !self.logelr_mean.is_finite() {
            return Err(Error::InvalidSetting {
                name: "logelr_mean",
                value: self.logelr_mean,
                expected: "a finite prior mean",
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

    /// The log posterior of the observed cells of `segment`.
    fn model_of(
        &self,
        segment: &Segment,
        chain_ladder: &ChainLadderFit,
        premiums: &[f64],
    ) -> Result<Model> {
        let mut values = Vec::new();
        for (w, &last) in chain_ladder.latest_position.iter().enumerate() {
            for d in 0..=last {
                let value = segment
                    .get(w, d)
                    .expect("the chain ladder checked the gaps");
                values.push((w, d, value));
            }
        }
        self.model_from(
            segment.n_origins,
            segment.n_dev,
            &values,
            premiums,
            &segment.origins,
            &segment.ages,
        )
    }

    /// The log posterior of the cumulative `values` `(w, d, C_wd)`, origin
    /// by origin and age by age within an origin, with origins `origins`,
    /// ages `ages` and each origin's premium.
    fn model_from(
        &self,
        n_origins: usize,
        n_dev: usize,
        values: &[(usize, usize, f64)],
        premiums: &[f64],
        origins: &[Period],
        ages: &[Lag],
    ) -> Result<Model> {
        let mut cells = Vec::with_capacity(values.len());
        for &(w, d, value) in values {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::Bayes(format!(
                    "origin {} at development age {} months is {value}; the log-normal \
                     model needs positive cumulative values",
                    origins[w], ages[d]
                )));
            }
            cells.push(Cell {
                w,
                d,
                y: value.ln(),
                offset: premiums[w].ln(),
                prev: None,
            });
        }
        if self.model == MeyersModel::Cay {
            // Each cell's mean takes the previous origin's residual at the
            // same age; cells run origin by origin, so it comes first.
            let at = cell_index(&cells, n_origins, n_dev);
            for c in &mut cells {
                if c.w == 0 {
                    continue;
                }
                c.prev = at[c.w - 1][c.d];
                if c.prev.is_none() {
                    return Err(Error::Bayes(format!(
                        "origin {} is observed at development age {} months but origin {} is \
                         not; CAY needs the previous origin's cell at each age",
                        origins[c.w],
                        ages[c.d],
                        origins[c.w - 1]
                    )));
                }
            }
        }
        let model = Model {
            n_origins,
            n_dev,
            kind: self.model,
            cells,
            logelr_mean: self.logelr_mean,
            logelr_sd: self.logelr_sd,
            origin_sd: self.origin_sd,
            development_sd: self.development_sd,
            gamma_sd: self.gamma_sd,
            rho_shape: self.rho_shape,
            priors: true,
        };
        if model.cells.len() <= model.dim() {
            return Err(Error::Bayes(format!(
                "{} observed cells for {} parameters; the model needs more cells than \
                 parameters",
                model.cells.len(),
                model.dim()
            )));
        }
        Ok(model)
    }

    fn provenance(&self, column: &str, premium: &str, hasher: InputHasher) -> Provenance {
        let s = &self.sampler;
        let mut provenance = Provenance::new(format!("meyers_{}", self.model.name()))
            .param("logelr_mean", self.logelr_mean)
            .param("logelr_sd", self.logelr_sd)
            .param("origin_sd", self.origin_sd)
            .param("development_sd", self.development_sd);
        match self.model {
            MeyersModel::Crc => {}
            MeyersModel::Csr => provenance = provenance.param("gamma_sd", self.gamma_sd),
            MeyersModel::Cay => provenance = provenance.param("rho_shape", self.rho_shape),
        }
        provenance
            .param("chains", s.chains)
            .param("tune", s.tune)
            .param("draws", s.draws)
            .param("target_accept", s.target_accept)
            .param("max_depth", s.max_depth)
            .param("n_sims", self.n_sims)
            .param("seed", self.seed)
            .param("column", column)
            .param("premium", premium)
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(hasher.finish())
    }
}

/// The position in `cells` of each origin's cell at each age, if any.
fn cell_index(cells: &[Cell], n_origins: usize, n_dev: usize) -> Vec<Vec<Option<usize>>> {
    let mut at = vec![vec![None; n_dev]; n_origins];
    for (i, c) in cells.iter().enumerate() {
        at[c.w][c.d] = Some(i);
    }
    at
}

/// PSIS-LOO of `model`'s observed cells from `posterior` (see
/// [`MeyersFit::loo`]).
fn loo(model: &Model, posterior: &PosteriorDraws) -> Result<Loo> {
    let n = model.cells.len();
    let s = posterior.n_draws();
    let mut ll = Vec::with_capacity(s * n);
    let mut residuals = Vec::with_capacity(n);
    for nat in posterior.values().chunks_exact(posterior.dim()) {
        model.log_densities(nat, &mut residuals, &mut ll);
    }
    psis_loo(&ll, n, posterior)
}
