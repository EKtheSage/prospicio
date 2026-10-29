//! Meyers' correlated chain ladder (CRC), changing settlement rate (CSR) and
//! correlated accident year (CAY) models on cumulative values with premium
//! (`docs/design/reserving-bayes.md`, decision 7), over segments (decision
//! 11), in the diagonal backtest (decision 8) and in the one-year view
//! (decision 19, [`year`]).

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
use super::odp::log_mean_exp;
use super::one_year::{
    self, BayesOneYear, BayesOneYearFit, BayesOneYearFits, BayesOneYearSegment, YearDraws,
};
use super::{Diagnostics, Stream, derive_seed, one_year_seed, pilot_seed};
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
use crate::pattern_draws::{DevelopmentPatterns, predictive_with_cumulative_patterns};
use crate::segments::{FitTable, ReserveFit, SegmentFits, fit_each_with_exposure};
use crate::triangle::{Label, Segment, Triangle};

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

/// A fitted [`Meyers`].
#[derive(Debug, Clone)]
pub struct MeyersFit {
    /// The volume-weighted chain ladder of the fitted column: latest values
    /// and the long tables' `ultimate` and `reserve` come from it.
    pub chain_ladder: ChainLadderFit,
    /// The premium of each origin, the latest value of the premium column.
    pub premium: Vec<f64>,
    /// Posterior draws of `logelr`, `alpha[2]..alpha[n]`,
    /// `beta[1]..beta[m-1]`, `gamma` (CSR only) or `rho` (CAY only) and
    /// `sigma[1]..sigma[m]`,
    /// by chain.
    pub posterior: PosteriorDraws,
    /// R-hat, ESS and divergences, with warnings for failed thresholds.
    pub diagnostics: Diagnostics,
    /// Joint posterior predictive distribution of the ultimate (the value
    /// at the oldest age) by origin: dimension `origin`, one component per
    /// origin; an origin observed at the oldest age keeps its value.
    pub ultimates: PredictiveDistribution,
    /// The ultimates less the latest values, simulation by simulation:
    /// the same layout as [`OdpBootstrapFit::reserves`](crate::OdpBootstrapFit::reserves).
    pub reserves: PredictiveDistribution,
    /// The development pattern of the posterior draw each simulation took:
    /// the ratio of the expected cumulative values at each age to the one
    /// at the oldest age, `exp(beta_d g + sigma_d^2 / 2)` over
    /// `exp(sigma_m^2 / 2)`, for an origin with nothing observed, row `i`
    /// simulation `i` of `ultimates` and `reserves` (`reserving-bayes.md`
    /// decision 27). It ends at 1 at the oldest age (no tail), and can fall
    /// where `sigma_d` differs by age. In CSR, `g` is the speedup of the
    /// origin after the last. CAY's `rho` does not enter: the previous
    /// origin's residual is process, not development.
    pub patterns: DevelopmentPatterns,
    /// The model, for the pointwise log likelihood of [`loo`](Self::loo).
    model: Model,
}

/// One segment of a [`MeyersFits`]: its chain ladder, premium, posterior
/// and diagnostics, as in [`MeyersFit`]; its ultimates and reserves are
/// its part of the joint distributions of [`MeyersFits`].
#[derive(Debug, Clone)]
pub struct MeyersSegment {
    /// The volume-weighted chain ladder of the segment's fitted column.
    pub chain_ladder: ChainLadderFit,
    /// The premium of each origin, the latest value of the premium column.
    pub premium: Vec<f64>,
    /// Posterior draws, named as in [`MeyersFit::posterior`].
    pub posterior: PosteriorDraws,
    /// R-hat, ESS and divergences, with warnings for failed thresholds.
    pub diagnostics: Diagnostics,
    model: Model,
    /// The posterior draws on the unconstrained scale NUTS samples (an
    /// affine map of it), for the one-year view (decision 19); a
    /// [`MeyersFit`] does not keep them.
    sampled: PosteriorDraws,
}

impl MeyersSegment {
    /// The model that produced this fit, CRC, CSR or CAY.
    pub fn model(&self) -> MeyersModel {
        self.model.kind
    }

    /// Posterior summary of each parameter, as [`MeyersFit::summary`].
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        Ok(self.posterior.summary()?)
    }

    /// PSIS-LOO of the segment's observed cells, as [`MeyersFit::loo`].
    pub fn loo(&self) -> Result<Loo> {
        loo(&self.model, &self.posterior)
    }
}

impl ReserveFit for MeyersSegment {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn exposure(&self) -> Option<&[f64]> {
        Some(&self.premium)
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        vec![("premium", self.premium.clone())]
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("premium", self.premium.iter().sum())]
    }
}

/// A [`Meyers`] fit of every segment of a triangle column
/// ([`Meyers::fit_segments`]): each segment's own posterior, and the joint
/// distributions of the ultimates and of the reserves, with the layout of
/// [`OdpBootstrapFits`](crate::OdpBootstrapFits).
///
/// ```
/// use prospicio_bayes::nuts::Sampler;
/// use prospicio_reserving::bayes::Meyers;
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
///
/// // Five origins of cumulative paid with premium, and the same doubled.
/// let paid: [&[f64]; 5] = [
///     &[952.0, 1529.0, 2813.0, 3647.0, 3724.0],
///     &[849.0, 1564.0, 2202.0, 2432.0],
///     &[983.0, 2211.0, 2830.0],
///     &[1657.0, 2685.0],
///     &[932.0],
/// ];
/// let premium = [5812.0, 4908.0, 5454.0, 5165.0, 5214.0];
/// let (mut lob, mut origin, mut age, mut values, mut prem) = (vec![], vec![], vec![], vec![], vec![]);
/// for (line, scale) in [("Auto", 1.0), ("Home", 2.0)] {
///     for (w, row) in paid.iter().enumerate() {
///         for (d, &v) in row.iter().enumerate() {
///             lob.push(line);
///             origin.push(Month::january(1988 + w as i32));
///             age.push(12 * (d as u32 + 1));
///             values.push(v * scale);
///             prem.push(premium[w] * scale);
///         }
///     }
/// }
/// let tri = Triangle::from_long(&Long {
///     keys: &[("lob", &lob)],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&age),
///     values: &[("paid", &values), ("premium", &prem)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let model = Meyers {
///     sampler: Sampler { tune: 200, draws: 200, ..Meyers::default().sampler },
///     n_sims: 1_000,
///     ..Meyers::default()
/// };
/// let fits = model.fit_segments(&tri, "paid", "premium")?;
/// assert_eq!(fits.reserves.dims(), ["lob", "origin"]);
/// assert_eq!(fits.reserves.n_components(), 10);
/// assert_eq!(fits.ultimates.n_components(), 10);
/// assert_eq!(fits.segments.len(), 2);
/// assert_eq!(fits.totals().n_rows(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct MeyersFits {
    /// Each segment's chain ladder, premium, posterior and diagnostics.
    pub segments: SegmentFits<MeyersSegment>,
    /// Joint posterior predictive distribution of the ultimate by segment
    /// and origin: the key names and `origin` are its dimensions, and its
    /// components run over the origins of each segment in turn, like the
    /// rows of [`to_long`](Self::to_long).
    pub ultimates: PredictiveDistribution,
    /// The ultimates less the latest values, simulation by simulation,
    /// with the same layout.
    pub reserves: PredictiveDistribution,
    /// Each segment's development pattern in every simulation, as
    /// [`MeyersFit::patterns`], moving with the segment's reserves and
    /// ultimates through a rank correlation.
    pub patterns: DevelopmentPatterns,
}

/// Appends `ulr_mean`, the row's posterior mean ultimate `ultimate_mean`
/// over its `premium`, to a Meyers table.
fn push_ulr_mean(table: &mut FitTable) {
    push_combined(table, "ulr_mean", "ultimate_mean", "premium", |u, p| u / p);
}

impl MeyersFits {
    /// One row per segment × origin: the chain ladder's `latest`,
    /// `ultimate` and `reserve`, the `premium`, `ulr` (the chain ladder's
    /// ultimate over the premium), the `mean` and `std_dev` of the
    /// posterior predictive reserve, the `ultimate_mean` and
    /// `ultimate_std_dev` of the predictive ultimate, and `ulr_mean`, the
    /// `ultimate_mean` over the premium.
    pub fn to_long(&self) -> FitTable {
        let mut table = self.segments.to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        push_moments(&mut table, "ultimate_", &component_sums(&self.ultimates));
        push_ulr_mean(&mut table);
        table
    }

    /// One row per segment: the chain ladder's totals, the total
    /// `premium` and `ulr`, the `mean` and `std_dev` of the segment's total
    /// reserve, the `ultimate_mean` and `ultimate_std_dev` of its total
    /// ultimate, and `ulr_mean`, the total `ultimate_mean` over the total
    /// premium.
    pub fn totals(&self) -> FitTable {
        let mut table = self.segments.totals();
        push_moments(
            &mut table,
            "",
            &segment_sums(&self.segments, &self.reserves),
        );
        push_moments(
            &mut table,
            "ultimate_",
            &segment_sums(&self.segments, &self.ultimates),
        );
        push_ulr_mean(&mut table);
        table
    }

    /// The chain ladders' development factors, one row per segment × age.
    pub fn development_table(&self) -> FitTable {
        self.segments.development_table()
    }

    /// The one segment chosen as in [`SegmentFits::position`], with its
    /// part of the joint ultimates and reserves (same dimensions).
    pub fn segment(&self, keys: &[(&str, &str)]) -> Result<Self> {
        let (segments, reserves) = pick_segment(&self.segments, &self.reserves, keys)?;
        let (_, ultimates) = pick_segment(&self.segments, &self.ultimates, keys)?;
        let patterns = self.patterns.only(&segments.labels[0]);
        Ok(Self {
            segments,
            ultimates,
            reserves,
            patterns,
        })
    }
}

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
    /// `Synchronized` is an [`Error::Bayes`]. A failure names its segment.
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
    ///   [`Error::Bayes`]: it draws the residuals of the same positions in
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
    /// origin, with the seeds of `chains` (and its pilot run) and
    /// `predictive`.
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
        let (ultimates, patterns) = predictive_with_cumulative_patterns(
            &posterior,
            segment,
            self.n_sims,
            derive_seed(self.seed, predictive),
            provenance.param("output", "ultimates"),
            |nat, rng: &mut StreamRng, row, shares| {
                model.draw_ultimates(nat, &premiums, &closed, latest, rng, row);
                model.pattern_shares(nat, shares);
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

/// The train rows of a [`TriangleFrame`] rebuilt as Meyers' cumulative
/// cells, and the test rows as increments of each origin's path.
#[derive(Debug, Clone)]
struct Rebuilt {
    /// The frame's origin position of each model origin `w`.
    origins: Vec<usize>,
    /// Number of ages, the oldest train age plus one.
    n_dev: usize,
    /// Each origin's latest train age.
    latest: Vec<usize>,
    /// Each origin's cumulative value at its latest train age.
    last_value: Vec<f64>,
    /// Each origin's premium (the frame's `exposure`).
    premiums: Vec<f64>,
    /// The cumulative cells `(w, d, C_wd)`, origin by origin.
    values: Vec<(usize, usize, f64)>,
    /// `(w, d)` of each test row, in order.
    test: Vec<(usize, usize)>,
    /// Each origin's oldest test age, if it has a test row: its path runs
    /// from its latest train age to there.
    path_end: Vec<Option<usize>>,
}

impl Rebuilt {
    fn new(cells: &TriangleFrame, train: &[usize], test: &[usize]) -> Result<Self> {
        let response = cells.response();
        let exposure = match cells.frame().column("exposure") {
            Some(Column::Numeric(v)) => v,
            _ => unreachable!("TriangleFrame::new builds a numeric exposure column"),
        };
        let mut by_origin: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for &r in train {
            by_origin
                .entry(cells.origin_of(r))
                .or_default()
                .push((cells.development_of(r), r));
        }
        let origins: Vec<usize> = by_origin.keys().copied().collect();
        let label = |o: usize| cells.origins()[o];
        if let Some(gap) = origins.windows(2).find(|p| p[1] != p[0] + 1) {
            return Err(Error::Bayes(format!(
                "origin {} has no train row but origins {} and {} have; Meyers' models need \
                 the train origins without a gap",
                label(gap[0] + 1),
                label(gap[0]),
                label(gap[1])
            )));
        }
        let (mut latest, mut last_value, mut premiums, mut values) =
            (vec![], vec![], vec![], vec![]);
        for (w, (&o, rows)) in by_origin.iter_mut().enumerate() {
            rows.sort_unstable();
            let mut cumulative = 0.0;
            for (k, &(d, r)) in rows.iter().enumerate() {
                if d != k {
                    return Err(Error::Bayes(format!(
                        "origin {} has no train row at development age {} months but has one \
                         later; Meyers' models rebuild each origin's cumulative values from \
                         its first age",
                        label(o),
                        cells.ages()[k]
                    )));
                }
                cumulative += response[r];
                values.push((w, d, cumulative));
            }
            latest.push(rows.len() - 1);
            last_value.push(cumulative);
            premiums.push(exposure[rows[0].1]);
        }
        let n_dev = latest.iter().max().map_or(0, |&l| l + 1);
        let mut path_end = vec![None; origins.len()];
        let test = test
            .iter()
            .map(|&r| {
                let (o, d) = (cells.origin_of(r), cells.development_of(r));
                let w = origins.binary_search(&o).map_err(|_| {
                    Error::Bayes(format!("test row {r} has an origin without a train row"))
                })?;
                if d >= n_dev {
                    return Err(Error::Bayes(format!(
                        "test row {r} has an age without a train row"
                    )));
                }
                if d <= latest[w] {
                    return Err(Error::Bayes(format!(
                        "test row {r} is not after its origin's latest train row; Meyers' \
                         models forecast each origin's cumulative path past it"
                    )));
                }
                let end: &mut Option<usize> = &mut path_end[w];
                *end = Some(end.map_or(d, |e| e.max(d)));
                Ok((w, d))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            origins,
            n_dev,
            latest,
            last_value,
            premiums,
            values,
            test,
            path_end,
        })
    }

    /// Whether origin `w`'s path holds age `d`.
    fn in_path(&self, w: usize, d: usize) -> bool {
        d > self.latest[w] && self.path_end[w].is_some_and(|e| d <= e)
    }

    /// The ages some path holds, in order.
    fn path_ages(&self) -> Vec<usize> {
        (0..self.n_dev)
            .filter(|&d| (0..self.origins.len()).any(|w| self.in_path(w, d)))
            .collect()
    }
}

/// The diagonal backtest's view of a [`Meyers`] (decision 8). The frame's
/// rows are increments, and Meyers models cumulative values with premium,
/// so the model is fitted to the train rows rebuilt as cumulative cells:
///
/// * **Cumulative values.** Each origin's train rows must run from the
///   first age without a gap (an [`Error::Bayes`] otherwise), and their
///   running sums are its cumulative values, which must be positive. The
///   train origins must have no gap between them either (CAY's previous
///   origin is the one before). The model's ages are the first age to the
///   oldest train age, which takes `beta_m = 0`.
/// * **Premium** is the frame's `exposure` column (the origin's exposure,
///   given to [`TriangleFrame::new`]). A frame built without exposure has
///   1 there, which the model fits but which puts its `logelr` far from
///   Meyers' prior on a loss ratio; give the premium.
/// * **Test rows** must be after their origin's latest train row, at an
///   age with train rows. Each origin's cumulative value is simulated
///   along one path, from the age after its latest train row to its oldest
///   test row, with one posterior draw per simulation; a test row's
///   increment is the difference of consecutive values on that path (the
///   first from the latest train value), so an origin's increments sum to
///   its last simulated value less its latest train value. Given the
///   parameters the values are independent log-normals (in CAY, a moving
///   average down each age, the previous origin's residual observed for a
///   train cell and drawn otherwise), so a simulated increment can be
///   negative, as Meyers' reserves can.
/// * **Mean** is each test row's posterior mean increment: the mean of
///   the log-normal `E[C_wd]` less the latest train value (or less
///   `E[C_w,d-1]` past the first age of the path), over every posterior
///   draw.
/// * **Log density.** When every test row follows its origin's latest
///   train row (always so in [`diagonal_backtest`](crate::diagonal_backtest)),
///   a test increment `y` is `C_wd` less a known value, so its density is
///   the log-normal density of `C_wd` at `y` plus the latest train value,
///   per unit of the response, and the log density is the log of its mean
///   over every posterior draw (the log predictive density, as
///   [`BayesOdp`](super::BayesOdp) gives it); minus infinity when that sum
///   is not positive. In CAY a test row whose previous origin has no train
///   cell at its age takes that origin's residual integrated out (variance
///   `(1 + rho^2) sigma_d^2`). When some test row is further out, its
///   increment is a difference of two random log-normals, with no closed
///   density, and the log density is `None`.
/// * **Seeds.** The chains (and the pilot run) and the simulations take
///   seeds derived from the backtest's `seed` as [`Meyers::fit`] does from
///   [`Meyers::seed`]; the backtest's `n_sims` replaces [`Meyers::n_sims`].
///   The fit's diagnostics come back in [`CellForecast::diagnostics`] and
///   [`crate::Backtest::diagnostics`], and a fit with warnings is still scored.
///   A held-out triangle is smaller than the whole: on Meyers'
///   group 353 the default target acceptance 0.95 failed (R-hat above
///   1.05) in 21 of 3,960 backtest fits on its 28- to 45-cell training
///   triangles, and 0.99 in 4 of 5,400; raise [`Meyers::sampler`]'s
///   `target_accept` for small triangles.
///
/// Its name in the backtest is `"meyers_crc"`, `"meyers_csr"` or
/// `"meyers_cay"`.
impl TriangleModel for Meyers {
    fn name(&self) -> &str {
        match self.model {
            MeyersModel::Crc => "meyers_crc",
            MeyersModel::Csr => "meyers_csr",
            MeyersModel::Cay => "meyers_cay",
        }
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
        let rebuilt = Rebuilt::new(cells, train, test)?;
        let no = rebuilt.origins.len();
        let origins: Vec<Period> = rebuilt
            .origins
            .iter()
            .map(|&o| cells.origins()[o])
            .collect();
        let model = self.model_from(
            no,
            rebuilt.n_dev,
            &rebuilt.values,
            &rebuilt.premiums,
            &origins,
            cells.ages(),
        )?;
        let (posterior, _, diagnostics) = self.sample(&model, derive_seed(seed, Stream::Chains))?;
        let at = cell_index(&model.cells, no, rebuilt.n_dev);
        let ages = rebuilt.path_ages();

        let n_draws = posterior.n_draws();
        let response = cells.response();
        let closed = rebuilt
            .test
            .iter()
            .all(|&(w, d)| d == rebuilt.latest[w] + 1);
        let mut mean = vec![0.0; test.len()];
        let mut ln_f = vec![Vec::with_capacity(n_draws); test.len()];
        let mut residuals = Vec::new();
        for s in 0..n_draws {
            let nat = posterior.draw(s);
            model.cay_residuals(nat, &mut residuals);
            let moments = |w: usize, d: usize| {
                model.log_moments(nat, &residuals, &at, &rebuilt.premiums, w, d)
            };
            for (j, (&r, &(w, d))) in test.iter().zip(&rebuilt.test).enumerate() {
                let (m, v) = moments(w, d);
                let before = if d == rebuilt.latest[w] + 1 {
                    rebuilt.last_value[w]
                } else {
                    let (m0, v0) = moments(w, d - 1);
                    (m0 + 0.5 * v0).exp()
                };
                mean[j] += ((m + 0.5 * v).exp() - before) / n_draws as f64;
                if closed {
                    let c = response[r] + rebuilt.last_value[w];
                    ln_f[j].push(if c > 0.0 && c.is_finite() {
                        let z = c.ln() - m;
                        -c.ln() - 0.5 * (std::f64::consts::TAU * v).ln() - 0.5 * z * z / v
                    } else {
                        f64::NEG_INFINITY
                    });
                }
            }
        }
        let log_density = closed.then(|| ln_f.iter().map(|f| log_mean_exp(f)).collect());

        let distribution = posterior.predictive(
            vec!["row".into()],
            test.iter().map(|&r| vec![(r as i64).into()]).collect(),
            n_sims,
            derive_seed(seed, Stream::Predictive),
            self.provenance("backtest", "exposure", InputHasher::new())
                .param("output", "test increments"),
            |nat, rng: &mut StreamRng, row| {
                model.draw_paths(nat, &rebuilt, &at, &ages, rng, row);
            },
        )?;
        Ok(CellForecast {
            mean,
            distribution,
            log_density,
            diagnostics: Some(diagnostics),
        })
    }
}

impl MeyersSegment {
    /// The fit of a single segment, with its lifetime predictive.
    fn into_fit(
        self,
        ultimates: PredictiveDistribution,
        reserves: PredictiveDistribution,
        patterns: DevelopmentPatterns,
    ) -> MeyersFit {
        let MeyersSegment {
            chain_ladder,
            premium,
            posterior,
            diagnostics,
            model,
            sampled: _,
        } = self;
        MeyersFit {
            chain_ladder,
            premium,
            posterior,
            diagnostics,
            ultimates,
            reserves,
            patterns,
            model,
        }
    }
}

impl MeyersFit {
    /// The model that produced this fit, CRC, CSR or CAY.
    pub fn model(&self) -> MeyersModel {
        self.model.kind
    }

    /// Posterior summary of each parameter: mean, sd, 5/50/95%
    /// quantiles, R-hat, bulk and tail ESS.
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        Ok(self.posterior.summary()?)
    }

    /// PSIS-LOO of the observed cells, with each cell's relative
    /// efficiency from the chains. The pointwise term is Meyers', the
    /// normal density of `log C_wd` (`normal_lpdf` in his Stan code), so
    /// the result is on the log scale and comparable only between CRC, CSR
    /// and CAY fits of the same column (Meyers 2019, Tables 6.4, 7.3 and
    /// 8.3). In CAY a cell's mean holds the previous origin's observed
    /// residual, so its term is conditional on that cell, as in Meyers'
    /// code.
    pub fn loo(&self) -> Result<Loo> {
        loo(&self.model, &self.posterior)
    }

    /// One row per origin: the chain ladder's `latest`, `ultimate` and
    /// `reserve`, the `premium`, `ulr` (the chain ladder's ultimate over
    /// the premium), the `mean` and `std_dev` of the posterior predictive
    /// reserve, the `ultimate_mean` and `ultimate_std_dev` of the
    /// predictive ultimate, and `ulr_mean`, the `ultimate_mean` over the
    /// premium.
    pub fn to_long(&self) -> FitTable {
        let mut table = self.plug_in().to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        push_moments(&mut table, "ultimate_", &component_sums(&self.ultimates));
        push_ulr_mean(&mut table);
        table
    }

    /// One row: the chain ladder's totals, the total `premium` and `ulr`,
    /// the `mean` and `std_dev` of the posterior predictive total reserve,
    /// the `ultimate_mean` and `ultimate_std_dev` of the total ultimate,
    /// and `ulr_mean`, the total `ultimate_mean` over the total premium.
    pub fn totals(&self) -> FitTable {
        let fits = self.plug_in();
        let mut table = fits.totals();
        push_moments(&mut table, "", &segment_sums(&fits, &self.reserves));
        push_moments(
            &mut table,
            "ultimate_",
            &segment_sums(&fits, &self.ultimates),
        );
        push_ulr_mean(&mut table);
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
                premium: self.premium.clone(),
            }],
        }
    }
}

impl ReserveFit for MeyersFit {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn exposure(&self) -> Option<&[f64]> {
        Some(&self.premium)
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        vec![("premium", self.premium.clone())]
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("premium", self.premium.iter().sum())]
    }
}

/// The chain ladder and premium of a fit, for its long tables.
#[derive(Debug, Clone)]
struct PlugIn {
    chain_ladder: ChainLadderFit,
    premium: Vec<f64>,
}

impl ReserveFit for PlugIn {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn exposure(&self) -> Option<&[f64]> {
        Some(&self.premium)
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        vec![("premium", self.premium.clone())]
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("premium", self.premium.iter().sum())]
    }
}

/// An observed cell: origin and development positions, `log C_wd` and
/// `log P_w`, and (CAY only, after the first origin) the index of the
/// previous origin's cell at the same age, which comes earlier.
#[derive(Debug, Clone, Copy)]
struct Cell {
    w: usize,
    d: usize,
    y: f64,
    offset: f64,
    prev: Option<usize>,
}

/// Smallest step `s_i` of `sigma_d^2`, `1 - exp(-1e-5)`: Meyers bounds
/// `a_ig` above by `1e5` in his Stan code, and `s_i = 1 - exp(-1 / a_ig)`.
pub(crate) const S_FLOOR: f64 = 9.999_950_000_166_667e-6;

/// The log posterior on the unconstrained
/// `theta = (logelr, alpha_2..alpha_n, beta_1..beta_(m-1), [gamma | v], u_1..u_m)`
/// with `s_i = S_FLOOR + (1 - S_FLOOR) logistic(u_i)` and (CAY)
/// `rho = tanh(v)`.
#[derive(Debug, Clone)]
struct Model {
    n_origins: usize,
    n_dev: usize,
    kind: MeyersModel,
    cells: Vec<Cell>,
    logelr_mean: f64,
    logelr_sd: f64,
    origin_sd: f64,
    development_sd: f64,
    gamma_sd: f64,
    rho_shape: f64,
    /// Without priors, the likelihood alone (for the tests).
    priors: bool,
}

/// `log(1 + exp(x))` without overflow.
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// `1 / (1 + exp(-u))` without overflow.
fn logistic(u: f64) -> f64 {
    if u >= 0.0 {
        1.0 / (1.0 + (-u).exp())
    } else {
        let e = u.exp();
        e / (1.0 + e)
    }
}

impl Model {
    fn csr(&self) -> bool {
        self.kind == MeyersModel::Csr
    }

    fn cay(&self) -> bool {
        self.kind == MeyersModel::Cay
    }

    /// Position of `beta_1` in `theta`.
    fn beta0(&self) -> usize {
        self.n_origins
    }

    /// Position of `gamma` (CSR), of `v` or `rho` (CAY) or of `u_1` (CRC).
    fn after_beta(&self) -> usize {
        self.n_origins + self.n_dev - 1
    }

    /// Position of `u_1`.
    fn u0(&self) -> usize {
        self.after_beta() + usize::from(self.kind != MeyersModel::Crc)
    }

    /// The names of the unconstrained parameters: `v` is `atanh(rho)` and
    /// `u[d]` the logit of `sigma[d]^2`'s step.
    fn sampled_names(&self) -> Vec<String> {
        let mut names = self.names();
        if self.cay() {
            names[self.after_beta()] = "v".into();
        }
        for (d, name) in names[self.u0()..].iter_mut().enumerate() {
            *name = format!("u[{}]", d + 1);
        }
        names
    }

    fn names(&self) -> Vec<String> {
        let mut names = vec!["logelr".to_string()];
        names.extend((2..=self.n_origins).map(|w| format!("alpha[{w}]")));
        names.extend((1..self.n_dev).map(|d| format!("beta[{d}]")));
        match self.kind {
            MeyersModel::Crc => {}
            MeyersModel::Csr => names.push("gamma".into()),
            MeyersModel::Cay => names.push("rho".into()),
        }
        names.extend((1..=self.n_dev).map(|d| format!("sigma[{d}]")));
        names
    }

    /// `sigma_d^2` for each age from `theta`'s `u`, and each
    /// `l_i = logistic(u_i)` and `1 - l_i`, with
    /// `s_i = S_FLOOR + (1 - S_FLOOR) l_i`.
    fn variances(&self, theta: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let m = self.n_dev;
        let u = &theta[self.u0()..self.u0() + m];
        let l: Vec<f64> = u.iter().map(|&u| logistic(u)).collect();
        let lc: Vec<f64> = u.iter().map(|&u| logistic(-u)).collect();
        let mut var = vec![0.0; m];
        let mut acc = 0.0;
        for d in (0..m).rev() {
            acc += S_FLOOR + (1.0 - S_FLOOR) * l[d];
            var[d] = acc;
        }
        (var, l, lc)
    }

    /// `theta` on the natural scale: `logelr`, `alpha`, `beta`, `gamma`
    /// or `rho = tanh(v)`, and `sigma_d = sqrt(sigma_d^2)`.
    fn natural(&self, theta: &[f64]) -> Vec<f64> {
        let (var, _, _) = self.variances(theta);
        let mut out = theta[..self.u0()].to_vec();
        if self.cay() {
            let k = self.after_beta();
            out[k] = out[k].tanh();
        }
        out.extend(var.iter().map(|v| v.sqrt()));
        out
    }

    /// `g_w` and its derivative in `gamma`, for 0-based origin `w`.
    fn speedup(&self, gamma: f64, w: usize) -> (f64, f64) {
        if !self.csr() || w == 0 {
            return (1.0, 0.0);
        }
        let k = w as i32;
        let base = 1.0 - gamma;
        (base.powi(k), -(w as f64) * base.powi(k - 1))
    }

    /// The development pattern of a natural-scale draw (`reserving-bayes.md`
    /// decision 27), into `shares` (one per age): the expected cumulative
    /// value of an origin with nothing observed at age `d` over the one at
    /// the oldest age, `exp(beta_d g + sigma_d^2 / 2) / exp(sigma_m^2 / 2)`,
    /// where `logelr`, `alpha` and the premium cancel. `g` is the speedup of
    /// the origin after the last (CSR; 1 otherwise), `beta_m = 0` and the
    /// last share is exactly 1. No random number is drawn.
    fn pattern_shares(&self, nat: &[f64], shares: &mut [f64]) {
        let m = self.n_dev;
        let gamma = if self.csr() {
            nat[self.after_beta()]
        } else {
            0.0
        };
        let g = self.speedup(gamma, self.n_origins).0;
        let log_mean = |d: usize| {
            let beta = if d + 1 < m {
                nat[self.beta0() + d]
            } else {
                0.0
            };
            let sigma = nat[self.u0() + d];
            beta * g + 0.5 * sigma * sigma
        };
        let last = log_mean(m - 1);
        for (d, share) in shares.iter_mut().enumerate() {
            *share = (log_mean(d) - last).exp();
        }
        shares[m - 1] = 1.0;
    }

    /// `mu_wd` without CAY's `rho` term, from natural-scale or
    /// unconstrained `theta` (both hold `logelr`, `alpha`, `beta` and
    /// `gamma` at the same positions).
    fn mu(&self, theta: &[f64], cell: &Cell) -> f64 {
        let alpha = if cell.w > 0 { theta[cell.w] } else { 0.0 };
        let beta = if cell.d + 1 < self.n_dev {
            theta[self.beta0() + cell.d]
        } else {
            0.0
        };
        let gamma = if self.csr() {
            theta[self.after_beta()]
        } else {
            0.0
        };
        cell.offset + theta[0] + alpha + beta * self.speedup(gamma, cell.w).0
    }

    /// The residual `log C_wd - mu_wd` of each cell, in `residuals`, with
    /// CAY's `rho` (0 otherwise): each cell's mean takes `rho` times the
    /// previous origin's residual, which comes earlier.
    fn residuals(&self, theta: &[f64], rho: f64, residuals: &mut Vec<f64>) {
        residuals.clear();
        for c in &self.cells {
            let mut r = c.y - self.mu(theta, c);
            if let Some(p) = c.prev {
                r -= rho * residuals[p];
            }
            residuals.push(r);
        }
    }

    /// Meyers' pointwise log likelihood, `normal_lpdf(log C | mu, sigma_d)`,
    /// of every cell from a natural-scale draw, appended to `out`
    /// (`residuals` is scratch).
    fn log_densities(&self, nat: &[f64], residuals: &mut Vec<f64>, out: &mut Vec<f64>) {
        let rho = if self.cay() {
            nat[self.after_beta()]
        } else {
            0.0
        };
        self.residuals(nat, rho, residuals);
        for (c, r) in self.cells.iter().zip(residuals.iter()) {
            let sigma = nat[self.u0() + c.d];
            let z = r / sigma;
            out.push(-0.5 * (std::f64::consts::TAU.ln() + z * z) - sigma.ln());
        }
    }

    /// One simulation of the ultimate (the value at age `m`) of each origin
    /// into `row`, given a natural-scale draw: a closed origin keeps its
    /// latest value, any other is log-normal with mean
    /// `log P_w + logelr + alpha_w` (`beta_m = 0`, so neither beta nor
    /// gamma enters) and sd `sigma_m`. In CAY each mean also takes `rho`
    /// times the previous origin's residual at age `m`, observed for a
    /// closed origin and drawn otherwise, origin by origin (Meyers 2019,
    /// sec. 8).
    fn draw_ultimates(
        &self,
        nat: &[f64],
        premiums: &[f64],
        closed: &[bool],
        latest: &[f64],
        rng: &mut StreamRng,
        row: &mut [f64],
    ) {
        // The natural-scale draw ends with sigma[m].
        let sigma_m = nat[self.dim() - 1];
        if !self.cay() {
            for (w, ultimate) in row.iter_mut().enumerate() {
                *ultimate = if closed[w] {
                    latest[w]
                } else {
                    let alpha = if w > 0 { nat[w] } else { 0.0 };
                    let z = norm_quantile(rng.next_open01());
                    (premiums[w].ln() + nat[0] + alpha + sigma_m * z).exp()
                };
            }
            return;
        }
        let rho = nat[self.after_beta()];
        let mut residual = 0.0;
        for (w, ultimate) in row.iter_mut().enumerate() {
            let alpha = if w > 0 { nat[w] } else { 0.0 };
            let mean = premiums[w].ln() + nat[0] + alpha + rho * residual;
            if closed[w] {
                *ultimate = latest[w];
                residual = latest[w].ln() - mean;
            } else {
                residual = sigma_m * norm_quantile(rng.next_open01());
                *ultimate = (mean + residual).exp();
            }
        }
    }

    /// `mu_wd` without CAY's `rho` term at any origin and age, from a
    /// natural-scale draw and each origin's premium.
    fn mu_at(&self, nat: &[f64], premiums: &[f64], w: usize, d: usize) -> f64 {
        let cell = Cell {
            w,
            d,
            y: 0.0,
            offset: premiums[w].ln(),
            prev: None,
        };
        self.mu(nat, &cell)
    }

    /// CAY's residuals of the observed cells from a natural-scale draw, in
    /// `residuals`; empty for CRC and CSR, which do not use them.
    fn cay_residuals(&self, nat: &[f64], residuals: &mut Vec<f64>) {
        if self.cay() {
            self.residuals(nat, nat[self.after_beta()], residuals);
        } else {
            residuals.clear();
        }
    }

    /// The mean and variance of `log C_wd` for a cell that is not observed,
    /// given a natural-scale draw: `mu_wd` and `sigma_d^2`; in CAY, plus
    /// `rho` times the previous origin's residual at age `d` when that cell
    /// is observed (`at`, with `residuals` from
    /// [`cay_residuals`](Self::cay_residuals)), and with that residual,
    /// `N(0, sigma_d^2)`, integrated out when it is not, which makes the
    /// variance `(1 + rho^2) sigma_d^2`.
    fn log_moments(
        &self,
        nat: &[f64],
        residuals: &[f64],
        at: &[Vec<Option<usize>>],
        premiums: &[f64],
        w: usize,
        d: usize,
    ) -> (f64, f64) {
        let sigma = nat[self.u0() + d];
        let mut m = self.mu_at(nat, premiums, w, d);
        let mut v = sigma * sigma;
        if self.cay() && w > 0 {
            let rho = nat[self.after_beta()];
            match at[w - 1][d] {
                Some(i) => m += rho * residuals[i],
                None => v *= 1.0 + rho * rho,
            }
        }
        (m, v)
    }

    /// One simulation of the test increments of `rebuilt` into `row`, given
    /// a natural-scale draw: each origin's cumulative values along its path
    /// (log-normal, independent given the draw in CRC and CSR), and each
    /// test row's increment the difference of consecutive values on it, the
    /// first from the latest train value. In CAY the ages of the paths run
    /// in order and, within an age, the origins: each mean takes `rho`
    /// times the previous origin's residual at that age, observed for a
    /// train cell and drawn otherwise (also for a cell on no path, which
    /// integrates it out). `at` indexes the model's cells and `ages` are
    /// [`Rebuilt::path_ages`].
    fn draw_paths(
        &self,
        nat: &[f64],
        rebuilt: &Rebuilt,
        at: &[Vec<Option<usize>>],
        ages: &[usize],
        rng: &mut StreamRng,
        row: &mut [f64],
    ) {
        let cay = self.cay();
        let rho = if cay { nat[self.after_beta()] } else { 0.0 };
        let mut residuals = Vec::new();
        self.cay_residuals(nat, &mut residuals);
        let no = rebuilt.origins.len();
        // Origin w's value at age d is path[w][d - latest[w] - 1].
        let mut path: Vec<Vec<f64>> = (0..no)
            .map(|w| vec![0.0; rebuilt.path_end[w].map_or(0, |e| e - rebuilt.latest[w])])
            .collect();
        for &d in ages {
            let sigma = nat[self.u0() + d];
            // The previous origin's residual at age d (0 before the first).
            let mut previous = 0.0;
            for (w, values) in path.iter_mut().enumerate() {
                let on_path = rebuilt.in_path(w, d);
                previous = match at[w][d] {
                    Some(i) if cay => residuals[i],
                    Some(_) => 0.0,
                    None if on_path || cay => {
                        let e = sigma * norm_quantile(rng.next_open01());
                        if on_path {
                            let mu = self.mu_at(nat, &rebuilt.premiums, w, d);
                            values[d - rebuilt.latest[w] - 1] = (mu + rho * previous + e).exp();
                        }
                        e
                    }
                    None => 0.0,
                };
            }
        }
        for (x, &(w, d)) in row.iter_mut().zip(&rebuilt.test) {
            let k = d - rebuilt.latest[w] - 1;
            let before = if k == 0 {
                rebuilt.last_value[w]
            } else {
                path[w][k - 1]
            };
            *x = path[w][k] - before;
        }
    }

    /// The Newton start: least squares of `log C - log P` on `logelr`,
    /// `alpha` and `beta` (CRC, `gamma = 0`), then each age's mean squared
    /// residual made non-increasing from the oldest age, its steps as
    /// `s_i` within `[0.001, 0.5]`.
    fn start(&self) -> Vec<f64> {
        let k = self.after_beta();
        let (n, m) = (self.n_origins, self.n_dev);
        let columns = |c: &Cell| {
            let mut x = vec![0];
            if c.w > 0 {
                x.push(c.w);
            }
            if c.d + 1 < m {
                x.push(n + c.d);
            }
            x
        };
        let mut xtx = vec![0.0; k * k];
        let mut xty = vec![0.0; k];
        for c in &self.cells {
            let x = columns(c);
            for &i in &x {
                xty[i] += c.y - c.offset;
                for &j in &x {
                    xtx[i * k + j] += 1.0;
                }
            }
        }
        for j in 0..k {
            xtx[j * k + j] += 1e-8;
        }
        let mut theta = vec![0.0; self.dim()];
        if let Some(l) = cholesky(&xtx, k) {
            theta[..k].copy_from_slice(&cholesky_solve(&l, &xty));
        } else {
            theta[0] = self.logelr_mean;
        }
        let mut sum = vec![0.0; m];
        let mut count = vec![0.0; m];
        for c in &self.cells {
            let r = c.y - self.mu(&theta, c);
            sum[c.d] += r * r;
            count[c.d] += 1.0;
        }
        let mut var: Vec<f64> = sum
            .iter()
            .zip(&count)
            .map(|(s, n)| if *n > 0.0 { s / n } else { 0.0 })
            .collect();
        for d in (0..m.saturating_sub(1)).rev() {
            var[d] = var[d].max(var[d + 1]);
        }
        let u0 = self.u0();
        for d in 0..m {
            let next = if d + 1 < m { var[d + 1] } else { 0.0 };
            let s = (var[d] - next).clamp(1e-3, 0.5);
            let l = (s - S_FLOOR) / (1.0 - S_FLOOR);
            theta[u0 + d] = (l / (1.0 - l)).ln();
        }
        theta
    }
}

impl Posterior for Model {
    fn dim(&self) -> usize {
        self.u0() + self.n_dev
    }

    fn log_posterior(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        grad.fill(0.0);
        let (n, m) = (self.n_origins, self.n_dev);
        let mut f = 0.0;
        if self.priors {
            let mut prior = |j: usize, mean: f64, sd: f64| {
                let r = theta[j] - mean;
                f -= 0.5 * r * r / (sd * sd);
                grad[j] -= r / (sd * sd);
            };
            prior(0, self.logelr_mean, self.logelr_sd);
            for j in 1..n {
                prior(j, 0.0, self.origin_sd);
            }
            for j in n..n + m - 1 {
                prior(j, 0.0, self.development_sd);
            }
            if self.csr() {
                prior(self.after_beta(), 0.0, self.gamma_sd);
            }
            if self.cay() {
                // (1 - rho) / 2 = logistic(-2 v) ~ Beta(a, a), with its
                // Jacobian: a (log logistic(2 v) + log logistic(-2 v)).
                let (k, a) = (self.after_beta(), self.rho_shape);
                let v = theta[k];
                f -= a * (softplus(2.0 * v) + softplus(-2.0 * v));
                grad[k] -= 2.0 * a * v.tanh();
            }
        }
        let (var, s, sc) = self.variances(theta);
        let gamma = if self.csr() {
            theta[self.after_beta()]
        } else {
            0.0
        };
        let rho = if self.cay() {
            theta[self.after_beta()].tanh()
        } else {
            0.0
        };
        let mut r = Vec::with_capacity(self.cells.len());
        self.residuals(theta, rho, &mut r);
        // CAY: h_i = d f / d mu_i at fixed rho, through every later
        // residual of its age's column: h_i = r_i / v_i - rho h_(next i),
        // collected backwards.
        let mut h = Vec::new();
        if self.cay() {
            h = vec![0.0; self.cells.len()];
            for (i, c) in self.cells.iter().enumerate().rev() {
                h[i] += r[i] / var[c.d];
                if let Some(p) = c.prev {
                    h[p] -= rho * h[i];
                }
            }
        }
        let mut drho = 0.0;
        // d f / d sigma_d^2.
        let mut dvar = vec![0.0; m];
        for (i, c) in self.cells.iter().enumerate() {
            let v = var[c.d];
            let r_i = r[i];
            f -= 0.5 * (v.ln() + r_i * r_i / v);
            dvar[c.d] += 0.5 * (r_i * r_i / v - 1.0) / v;
            let e = if self.cay() { h[i] } else { r_i / v };
            if let Some(p) = c.prev {
                drho += e * r[p];
            }
            grad[0] += e;
            if c.w > 0 {
                grad[c.w] += e;
            }
            if c.d + 1 < m {
                let (g, dg) = self.speedup(gamma, c.w);
                let beta = theta[self.beta0() + c.d];
                grad[self.beta0() + c.d] += e * g;
                if self.csr() {
                    grad[self.after_beta()] += e * beta * dg;
                }
            }
        }
        if self.cay() {
            grad[self.after_beta()] += drho * (1.0 - rho * rho);
        }
        // sigma_d^2 = sum over i >= d of s_i, so d f / d s_i sums dvar
        // over d <= i, and ds_i / du_i = (1 - S_FLOOR) l_i (1 - l_i); the
        // U(S_FLOOR, 1) prior on s_i is l_i (1 - l_i) in u_i.
        let u0 = self.u0();
        let mut acc = 0.0;
        for i in 0..m {
            acc += dvar[i];
            let u = theta[u0 + i];
            f -= softplus(-u) + softplus(u);
            grad[u0 + i] = (1.0 - S_FLOOR) * s[i] * sc[i] * acc + (sc[i] - s[i]);
        }
        if f.is_finite() { f } else { f64::NAN }
    }
}

/// The log posterior written once over [`Scalar`](super::scalar::Scalar),
/// the oracle of the compartmental note's decision 4: test-only, beside
/// the hand-written [`Posterior::log_posterior`], which stays the fast
/// path. It computes each cell's residual forward, as the value does, and
/// lets the dual carry CAY's chain of residuals.
#[cfg(test)]
impl super::autodiff::GenericDensity for Model {
    fn n_params(&self) -> usize {
        self.dim()
    }

    fn log_density<S: super::scalar::Scalar>(&self, theta: &[S]) -> S {
        use super::scalar::{logistic, softplus};
        let (n, m) = (self.n_origins, self.n_dev);
        let zero = S::from_f64(0.0);
        let mut f = zero;
        if self.priors {
            let prior = |x: S, mean: f64, sd: f64| {
                let r = x - mean;
                r * r * (0.5 / (sd * sd))
            };
            f -= prior(theta[0], self.logelr_mean, self.logelr_sd);
            for &a in &theta[1..n] {
                f -= prior(a, 0.0, self.origin_sd);
            }
            for &b in &theta[n..n + m - 1] {
                f -= prior(b, 0.0, self.development_sd);
            }
            if self.csr() {
                f -= prior(theta[self.after_beta()], 0.0, self.gamma_sd);
            }
            if self.cay() {
                let v = theta[self.after_beta()] * 2.0;
                f -= (softplus(v) + softplus(-v)) * self.rho_shape;
            }
        }
        let u0 = self.u0();
        let mut var = vec![zero; m];
        let mut acc = zero;
        for d in (0..m).rev() {
            acc += logistic(theta[u0 + d]) * (1.0 - S_FLOOR) + S_FLOOR;
            var[d] = acc;
        }
        for &u in &theta[u0..u0 + m] {
            f -= softplus(-u) + softplus(u);
        }
        let k = self.after_beta();
        let rho = if self.cay() { theta[k].tanh() } else { zero };
        let mut r: Vec<S> = Vec::with_capacity(self.cells.len());
        for c in &self.cells {
            let mut mu = theta[0] + c.offset;
            if c.w > 0 {
                mu += theta[c.w];
            }
            if c.d + 1 < m {
                let beta = theta[self.beta0() + c.d];
                mu += if self.csr() {
                    beta * (-theta[k] + 1.0).powi(c.w as i32)
                } else {
                    beta
                };
            }
            let mut res = -mu + c.y;
            if let Some(p) = c.prev {
                res -= rho * r[p];
            }
            r.push(res);
            let v = var[c.d];
            f -= (v.ln() + res * res / v) * 0.5;
        }
        f
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::bayes::laplace::tests::assert_gradient;
    use crate::triangle::Long;
    use crate::{DevelopmentColumn, Grain, Month};
    use prospicio_bayes::nuts::LogDensity;
    use prospicio_core::Period;
    use prospicio_prob::Distribution;

    /// Meyers' illustrative insurer, group 353 Commercial Auto paid
    /// (Meyers 2019, Table 2.2; validation/data/clrd_comauto_353.csv), the
    /// cells at calendar years up to 1997.
    const PAID: [[f64; 10]; 10] = [
        [
            952., 1529., 2813., 3647., 3724., 3832., 3899., 3907., 3911., 3912.,
        ],
        [
            849., 1564., 2202., 2432., 2468., 2487., 2513., 2526., 2531., 2527.,
        ],
        [
            983., 2211., 2830., 3832., 4039., 4065., 4102., 4155., 4268., 4274.,
        ],
        [
            1657., 2685., 3169., 3600., 3900., 4320., 4332., 4338., 4341., 4341.,
        ],
        [
            932., 1940., 2626., 3332., 3368., 3491., 3531., 3540., 3540., 3583.,
        ],
        [
            1162., 2402., 2799., 2996., 3034., 3042., 3230., 3238., 3241., 3268.,
        ],
        [
            1478., 2980., 3945., 4714., 5462., 5680., 5682., 5683., 5684., 5684.,
        ],
        [
            1240., 2080., 2607., 3080., 3678., 4116., 4117., 4125., 4128., 4128.,
        ],
        [
            1326., 2412., 3367., 3843., 3965., 4127., 4133., 4141., 4142., 4144.,
        ],
        [
            1413., 2683., 3173., 3674., 3805., 4005., 4020., 4095., 4132., 4139.,
        ],
    ];
    const PREMIUM: [f64; 10] = [
        5812., 4908., 5454., 5165., 5214., 5230., 4992., 5466., 5226., 4962.,
    ];

    /// The training triangle of `rows` (the first `10 - w` cells of row
    /// `w`), columns `paid` and `premium`, origins from 1988.
    fn triangle(rows: &[[f64; 10]], premium: &[f64]) -> Triangle {
        let n = rows.len();
        let (mut origin, mut ages, mut paid, mut prem) = (vec![], vec![], vec![], vec![]);
        for (w, row) in rows.iter().enumerate() {
            for (d, &v) in row.iter().enumerate().take(n - w) {
                origin.push(Month::january(1988 + w as i32));
                ages.push(12 * (d as crate::Lag + 1));
                paid.push(v);
                prem.push(premium[w]);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn comauto() -> Triangle {
        triangle(&PAID, &PREMIUM)
    }

    fn quick(model: MeyersModel, seed: u64) -> Meyers {
        Meyers {
            model,
            sampler: Sampler {
                tune: 300,
                draws: 300,
                ..Meyers::default().sampler
            },
            n_sims: 2_000,
            seed,
            ..Meyers::default()
        }
    }

    fn prepared(model: MeyersModel) -> Model {
        let tri = comauto();
        let segment = tri.segment("paid").unwrap();
        let cl = ChainLadder::default()
            .fit_segment(&segment, &segment.ages)
            .unwrap();
        let premium =
            origin_exposure(&segment, &tri.segment("premium").unwrap(), "premium").unwrap();
        Meyers {
            model,
            ..Meyers::default()
        }
        .model_of(&segment, &cl, &premium)
        .unwrap()
    }

    /// A point off the mode with `gamma` or `rho` away from 0.
    fn off_mode(model: &Model) -> Vec<f64> {
        let mut x = model.start();
        for (j, v) in x.iter_mut().enumerate() {
            *v += 0.01 * (j as f64 % 7.0 - 3.0);
        }
        match model.kind {
            MeyersModel::Crc => {}
            MeyersModel::Csr => x[model.after_beta()] = 0.04,
            MeyersModel::Cay => x[model.after_beta()] = 0.6,
        }
        x
    }

    #[test]
    fn the_gradient_matches_a_finite_difference() {
        for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
            let model = prepared(kind);
            assert_eq!(model.dim(), if kind == MeyersModel::Crc { 29 } else { 30 });
            assert_eq!(model.cells.len(), 55);
            let x = off_mode(&model);
            assert_gradient(|t, g| model.log_posterior(t, g), &x);
            let whitened = Whitened::regularised(&model, &model.start()).unwrap();
            let z: Vec<f64> = (0..model.dim())
                .map(|j| 0.3 * ((j % 3) as f64 - 1.0))
                .collect();
            assert_gradient(|z, g| whitened.log_density(z, g).unwrap(), &z);
        }
    }

    #[test]
    fn the_generic_density_matches_the_hand_written_one() {
        use crate::bayes::autodiff::{assert_oracle, posterior_points};
        for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
            let model = prepared(kind);
            // The pilot run's scale at the start: the posterior's bulk,
            // not the joint mode deep in the funnel.
            let whitened = Whitened::regularised(&model, &model.start()).unwrap();
            let points = posterior_points(&whitened, model.dim(), 20);
            assert_oracle(&format!("Meyers {}", kind.name()), &model, &points);
        }
    }

    /// The benchmark's row for Meyers' CRC on group 353's paid column
    /// (`validation/data/clrd_comauto_353.csv`, the training cells in
    /// [`PAID`]): 29 parameters, 55 cells, at the start point.
    pub(crate) fn bench_crc_353() {
        let model = prepared(MeyersModel::Crc);
        let theta = model.start();
        crate::bayes::autodiff::bench_row::<_, 29>(
            "Meyers CRC, group 353 paid",
            &model,
            &theta,
            Some(&|t: &[f64], g: &mut [f64]| model.log_posterior(t, g)),
        );
    }

    #[test]
    fn csr_at_gamma_zero_is_crc_plus_the_gamma_prior() {
        let (crc, csr) = (prepared(MeyersModel::Crc), prepared(MeyersModel::Csr));
        let theta = off_mode(&crc);
        // The same point with gamma = 0 inserted before u.
        let mut with_gamma = theta[..crc.after_beta()].to_vec();
        with_gamma.push(0.0);
        with_gamma.extend(&theta[crc.after_beta()..]);
        let (mut g1, mut g2) = (vec![0.0; crc.dim()], vec![0.0; csr.dim()]);
        let a = crc.log_posterior(&theta, &mut g1);
        let b = csr.log_posterior(&with_gamma, &mut g2);
        // The gamma prior at 0 is 0 (up to its constant, left out).
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        let mut g2_without = g2.clone();
        g2_without.remove(csr.after_beta());
        for (x, y) in g1.iter().zip(&g2_without) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn cay_at_rho_zero_is_crc_plus_the_rho_prior() {
        let (crc, cay) = (prepared(MeyersModel::Crc), prepared(MeyersModel::Cay));
        let theta = off_mode(&crc);
        let k = crc.after_beta();
        let mut with_v = theta[..k].to_vec();
        with_v.push(0.0);
        with_v.extend(&theta[k..]);
        let (mut g1, mut g2) = (vec![0.0; crc.dim()], vec![0.0; cay.dim()]);
        let a = crc.log_posterior(&theta, &mut g1);
        let b = cay.log_posterior(&with_v, &mut g2);
        // The Beta(2, 2) prior with its Jacobian at v = 0: 2 log(1/4).
        assert!((b - a - 2.0 * 0.25_f64.ln()).abs() < 1e-10, "{a} vs {b}");
        let mut g2_without = g2.clone();
        g2_without.remove(k);
        for (x, y) in g1.iter().zip(&g2_without) {
            assert!((x - y).abs() < 1e-10, "{x} vs {y}");
        }
        // At rho = 0 the rho gradient is the sum over cells after the first
        // origin of (r_i / v_i) r_(w-1)d: not zero on these data.
        assert!(g2[k].abs() > 1e-3, "{}", g2[k]);
    }

    #[test]
    fn cay_follows_meyers_recursion() {
        // Meyers' CAY.R: cells sorted by (age, origin), and
        // mu[i] = base[i] + rho (y[i - 1] - mu[i - 1]) for origins after
        // the first; here recomputed that way from a natural-scale point.
        let model = prepared(MeyersModel::Cay);
        let nat = model.natural(&off_mode(&model));
        let rho = nat[model.after_beta()];
        assert!((rho - 0.6_f64.tanh()).abs() < 1e-15);
        let mut order: Vec<usize> = (0..model.cells.len()).collect();
        order.sort_by_key(|&i| (model.cells[i].d, model.cells[i].w));
        let mut expected = vec![0.0; model.cells.len()];
        let (mut last_y, mut last_mu) = (0.0, 0.0);
        for &i in &order {
            let c = &model.cells[i];
            let mut mu = model.mu(&nat, c);
            if c.w > 0 {
                mu += rho * (last_y - last_mu);
            }
            let sigma = nat[model.u0() + c.d];
            expected[i] =
                -0.5 * ((c.y - mu) / sigma).powi(2) - sigma.ln() - 0.5 * std::f64::consts::TAU.ln();
            (last_y, last_mu) = (c.y, mu);
        }
        let mut got = Vec::new();
        model.log_densities(&nat, &mut Vec::new(), &mut got);
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-12, "{g} vs {e}");
        }
        // The log posterior without priors is their sum less the
        // constant, plus the log Jacobian of each u_i.
        let unpriored = Model {
            priors: false,
            ..model.clone()
        };
        let theta = off_mode(&model);
        let mut grad = vec![0.0; model.dim()];
        let f = unpriored.log_posterior(&theta, &mut grad);
        let jacobian: f64 = theta[model.u0()..]
            .iter()
            .map(|&u| -(softplus(-u) + softplus(u)))
            .sum();
        let n = model.cells.len() as f64;
        let sum: f64 = expected.iter().sum();
        assert!(
            (f - (sum + 0.5 * n * std::f64::consts::TAU.ln() + jacobian)).abs() < 1e-9,
            "{f} vs {sum}"
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn cay_draws_the_oldest_age_as_a_moving_average() {
        // Given one natural-scale draw with a large sigma_m and rho, the
        // log ultimates down age m follow Meyers' recursion: origin 2's
        // mean takes rho times origin 1's observed residual, and the open
        // origins after it are a moving average of order 1, with lag-one
        // correlation rho / (1 + rho^2) and none further (Meyers 2019,
        // eq. 8.3). On group 353 sigma_m is near 0.013, so this barely
        // moves the predictive there; this test is what checks it.
        let model = prepared(MeyersModel::Cay);
        let (n, k) = (model.n_origins, model.after_beta());
        let (rho, sigma) = (0.8, 0.5);
        let mut nat = vec![0.0; model.dim()];
        nat[0] = -0.4;
        for (w, alpha) in nat.iter_mut().enumerate().take(n).skip(1) {
            *alpha = 0.05 * w as f64;
        }
        nat[k] = rho;
        nat[model.dim() - 1] = sigma;
        let premiums = PREMIUM;
        let mut closed = vec![false; n];
        closed[0] = true;
        let mut latest = vec![0.0_f64; n];
        latest[0] = 3912.0;
        let base = |w: usize| premiums[w].ln() + nat[0] + if w > 0 { nat[w] } else { 0.0 };
        let r1 = latest[0].ln() - base(0);
        let sims = 200_000;
        let mut rng = StreamRng::new(7, 0);
        let mut row = vec![0.0; n];
        let mut logs = vec![Vec::with_capacity(sims); n];
        for _ in 0..sims {
            model.draw_ultimates(&nat, &premiums, &closed, &latest, &mut rng, &mut row);
            assert_eq!(row[0], latest[0]);
            for (w, u) in row.iter().enumerate() {
                logs[w].push(u.ln());
            }
        }
        let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
        let cov = |x: &[f64], y: &[f64]| {
            let (mx, my) = (mean(x), mean(y));
            x.iter()
                .zip(y)
                .map(|(a, b)| (a - mx) * (b - my))
                .sum::<f64>()
                / (x.len() as f64 - 1.0)
        };
        let se = sigma / (sims as f64).sqrt();
        assert!((mean(&logs[1]) - (base(1) + rho * r1)).abs() < 5.0 * se);
        assert!((cov(&logs[1], &logs[1]).sqrt() / sigma - 1.0).abs() < 0.01);
        for (w, log) in logs.iter().enumerate().skip(2) {
            assert!((mean(log) - base(w)).abs() < 5.0 * se * 1.3);
            let sd = cov(log, log).sqrt();
            assert!((sd / (sigma * (1.0 + rho * rho).sqrt()) - 1.0).abs() < 0.01);
        }
        let corr = |a: usize, b: usize| {
            cov(&logs[a], &logs[b]) / (cov(&logs[a], &logs[a]) * cov(&logs[b], &logs[b])).sqrt()
        };
        let ma1 = rho / (1.0 + rho * rho);
        for w in 2..n - 1 {
            assert!((corr(w, w + 1) - ma1).abs() < 0.01, "{}", corr(w, w + 1));
        }
        assert!(corr(3, 5).abs() < 0.01);
        // Origin 2 has no earlier random term: its correlation with origin
        // 3 is rho / sqrt(1 + rho^2).
        assert!((corr(1, 2) - rho / (1.0 + rho * rho).sqrt()).abs() < 0.01);
        // CRC draws the same origins independently.
        let crc = prepared(MeyersModel::Crc);
        let mut crc_nat = nat.clone();
        crc_nat.remove(k);
        let mut a = Vec::with_capacity(sims);
        let mut b = Vec::with_capacity(sims);
        for _ in 0..sims {
            crc.draw_ultimates(&crc_nat, &premiums, &closed, &latest, &mut rng, &mut row);
            a.push(row[3].ln());
            b.push(row[4].ln());
        }
        assert!((cov(&a, &b) / (cov(&a, &a) * cov(&b, &b)).sqrt()).abs() < 0.01);
    }

    #[test]
    fn the_rho_prior_is_meyers_beta() {
        // With no cells and only the rho prior, the density of v is
        // Beta(a, a) of r = (1 - tanh v) / 2 times |dr/dv| = 2 r (1 - r),
        // up to its constant, so differences from v = 0 match exactly.
        for a in [2.0, 0.7, 5.0] {
            let model = Model {
                cells: Vec::new(),
                rho_shape: a,
                ..prepared(MeyersModel::Cay)
            };
            let k = model.after_beta();
            let mut theta = vec![0.0; model.dim()];
            let mut grad = vec![0.0; model.dim()];
            let base = model.log_posterior(&theta, &mut grad);
            for v in [-3.0, -0.4, 0.9, 2.5_f64] {
                theta[k] = v;
                let f = model.log_posterior(&theta, &mut grad) - base;
                let r = (1.0 - v.tanh()) / 2.0;
                let density =
                    |r: f64| (a - 1.0) * (r * (1.0 - r)).ln() + (2.0 * r * (1.0 - r)).ln();
                let expected = density(r) - density(0.5);
                assert!(
                    (f - expected).abs() < 1e-10,
                    "a {a} v {v}: {f} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn the_logit_gives_meyers_uniform_prior_on_s() {
        // Meyers' floor: a_ig < 1e5, s = 1 - exp(-1 / a_ig).
        assert!((S_FLOOR + (-1e-5_f64).exp_m1()).abs() < 1e-20);
        // With no cells and no priors, the density in u is
        // log l + log(1 - l) for l = logistic(u): the U(S_FLOOR, 1) density
        // of s = S_FLOOR + (1 - S_FLOOR) l times |ds/du|.
        let model = Model {
            cells: Vec::new(),
            priors: false,
            ..prepared(MeyersModel::Crc)
        };
        let mut theta = vec![0.0; model.dim()];
        let mut grad = vec![0.0; model.dim()];
        for u in [-6.0, -1.0, 0.0, 0.5, 3.0] {
            theta[model.u0()..].fill(u);
            let f = model.log_posterior(&theta, &mut grad);
            let s = logistic(u);
            let expected = model.n_dev as f64 * (s * (1.0 - s)).ln();
            assert!((f - expected).abs() < 1e-12, "u {u}: {f} vs {expected}");
        }
        // Far down, every step is the floor.
        theta[model.u0()..].fill(-60.0);
        let (var, _, _) = model.variances(&theta);
        assert!((var[model.n_dev - 1] / S_FLOOR - 1.0).abs() < 1e-12);
        assert!((var[0] / (10.0 * S_FLOOR) - 1.0).abs() < 1e-12);
        // Its integral over u is 1 (trapezoid on a wide grid).
        let h = 1e-3;
        let total: f64 = (-40_000..=40_000)
            .map(|k| {
                let s = logistic(k as f64 * h);
                s * (1.0 - s) * h
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-9, "{total}");
    }

    #[test]
    fn the_start_is_least_squares_with_falling_sigmas() {
        for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
            let model = prepared(kind);
            let start = model.start();
            // The location part is least squares: its gradient without
            // priors, at equal sigmas, is zero.
            let ols = Model {
                priors: false,
                ..model.clone()
            };
            // All sigma_d^2 equal (to 0.5, up to 1e-22): only s_m is not
            // negligible.
            let mut theta = start.clone();
            theta[model.u0()..].fill(-50.0);
            theta[model.dim() - 1] = 0.0;
            let mut grad = vec![0.0; model.dim()];
            ols.log_posterior(&theta, &mut grad);
            let worst = grad[..model.after_beta()]
                .iter()
                .fold(0.0_f64, |m, g| m.max(g.abs()));
            // Up to the floor's 1e-4 relative difference between the sigmas.
            assert!(worst < 1e-3, "{kind:?}: gradient {worst}");
            if kind != MeyersModel::Crc {
                assert_eq!(start[model.after_beta()], 0.0);
            }
            let sigma = &model.natural(&start)[model.u0()..];
            assert!(sigma.windows(2).all(|p| p[0] > p[1]), "{sigma:?}");
            assert!(sigma.iter().all(|&s| s > 0.0 && s < 1.0), "{sigma:?}");
        }
    }

    #[test]
    fn the_joint_mode_is_deep_in_the_funnel() {
        // Why Meyers samples on a pilot scale and not at the mode as
        // BayesOdp does: the joint mode puts sigma[10] far below its
        // posterior mean (0.020 in Stan's reference), where a Laplace scale
        // says little about the bulk.
        let model = prepared(MeyersModel::Crc);
        let theta = match crate::bayes::laplace::mode(&model, model.start()) {
            Ok(theta) => theta,
            // Newton creeps down the funnel and may not converge.
            Err(Error::Bayes(_)) => return,
            Err(e) => panic!("{e}"),
        };
        let sigma = model.natural(&theta)[model.dim() - 1];
        assert!(sigma < 0.005, "sigma[10] at the mode {sigma}");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_seed_replays_whatever_the_threads() {
        let tri = comauto();
        let a = quick(MeyersModel::Csr, 3)
            .fit(&tri, "paid", "premium")
            .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let b = pool.install(|| {
            quick(MeyersModel::Csr, 3)
                .fit(&tri, "paid", "premium")
                .unwrap()
        });
        assert_eq!(a.posterior, b.posterior);
        assert_eq!(a.reserves.draw_matrix(), b.reserves.draw_matrix());
        let c = quick(MeyersModel::Csr, 4)
            .fit(&tri, "paid", "premium")
            .unwrap();
        assert_ne!(a.posterior.values(), c.posterior.values());
        assert_ne!(a.reserves.draw_matrix(), c.reserves.draw_matrix());
        // The sampler's own seed is not used.
        let mut d = quick(MeyersModel::Csr, 3);
        d.sampler.seed = 99;
        assert_eq!(
            d.fit(&tri, "paid", "premium").unwrap().posterior,
            a.posterior
        );
        // CAY's sequential prediction replays too.
        let e = quick(MeyersModel::Cay, 3)
            .fit(&tri, "paid", "premium")
            .unwrap();
        let f = pool.install(|| {
            quick(MeyersModel::Cay, 3)
                .fit(&tri, "paid", "premium")
                .unwrap()
        });
        assert_eq!(e.posterior, f.posterior);
        assert_eq!(e.reserves.draw_matrix(), f.reserves.draw_matrix());
    }

    #[test]
    fn a_short_run_returns_the_fit_with_warnings() {
        let model = Meyers {
            sampler: Sampler {
                tune: 50,
                draws: 20,
                ..Meyers::default().sampler
            },
            n_sims: 100,
            ..Meyers::default()
        };
        let fit = model.fit(&comauto(), "paid", "premium").unwrap();
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
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn ultimates_and_reserves_by_origin() {
        let fit = quick(MeyersModel::Crc, 5)
            .fit(&comauto(), "paid", "premium")
            .unwrap();
        for pd in [&fit.ultimates, &fit.reserves] {
            assert_eq!(pd.dims(), ["origin"]);
            assert_eq!(pd.n_components(), 10);
            assert_eq!(pd.n_sims(), 2_000);
        }
        let oldest = vec![Period::year(1988).into()];
        let reserve = fit.reserves.marginal(&oldest).unwrap();
        assert_eq!((reserve.mean(), reserve.std_dev()), (0.0, 0.0));
        assert_eq!(fit.ultimates.marginal(&oldest).unwrap().mean(), 3912.0);
        // Reserves are the ultimates less the latest values, draw by draw.
        for (row_u, row_r) in fit
            .ultimates
            .draw_matrix()
            .chunks_exact(10)
            .zip(fit.reserves.draw_matrix().chunks_exact(10))
        {
            for ((u, r), l) in row_u.iter().zip(row_r).zip(&fit.chain_ladder.latest) {
                assert_eq!(*r, u - l);
            }
        }
        let names = fit.posterior.names();
        assert_eq!(names.len(), 29);
        assert_eq!(names[0], "logelr");
        assert_eq!(names[9], "alpha[10]");
        assert_eq!(names[10], "beta[1]");
        assert_eq!(names[19], "sigma[1]");
        assert_eq!(names[28], "sigma[10]");
        assert_eq!(fit.premium, PREMIUM);
        let long = fit.to_long();
        assert_eq!(long.n_rows(), 10);
        let ulr: Vec<f64> = (long.column("ultimate").unwrap().iter())
            .zip(&PREMIUM)
            .map(|(u, p)| u / p)
            .collect();
        assert_eq!(long.column("ulr").unwrap(), ulr);
        for column in [
            "premium",
            "mean",
            "std_dev",
            "ultimate_mean",
            "ultimate_std_dev",
        ] {
            assert!(long.column(column).is_some(), "{column}");
        }
        // The posterior mean ultimate over the premium, per origin and in
        // total.
        let ulr_mean: Vec<f64> = (long.column("ultimate_mean").unwrap().iter())
            .zip(&PREMIUM)
            .map(|(u, p)| u / p)
            .collect();
        assert_eq!(long.column("ulr_mean").unwrap(), ulr_mean);
        assert_eq!(long.values.last().unwrap().0, "ulr_mean");
        let totals = fit.totals();
        assert_eq!(
            totals.column("ulr_mean").unwrap(),
            [totals.column("ultimate_mean").unwrap()[0] / PREMIUM.iter().sum::<f64>()]
        );
        assert_eq!(
            totals.column("premium").unwrap(),
            [PREMIUM.iter().sum::<f64>()]
        );
        assert!((totals.column("mean").unwrap()[0] - fit.reserves.total().mean()).abs() < 1e-6);
        assert!(
            (totals.column("ultimate_mean").unwrap()[0] - fit.ultimates.total().mean()).abs()
                < 1e-6
        );
        assert_eq!(fit.development_table().n_rows(), 10);
        assert_eq!(fit.summary().unwrap().len(), 29);
        let loo = fit.loo().unwrap();
        assert_eq!(loo.pareto_k.len(), 55);
        assert!(loo.estimate.elpd.is_finite());
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn provenance_records_the_settings() {
        let fit = quick(MeyersModel::Csr, 5)
            .fit(&comauto(), "paid", "premium")
            .unwrap();
        assert_eq!(fit.posterior.names()[19], "gamma");
        assert_eq!(fit.model(), MeyersModel::Csr);
        for (pd, output) in [(&fit.ultimates, "ultimates"), (&fit.reserves, "reserves")] {
            let provenance = pd.provenance();
            assert_eq!(provenance.model, "meyers_csr");
            let keys: Vec<&str> = provenance
                .parameters
                .iter()
                .map(|(k, _)| k.as_str())
                .collect();
            for key in [
                "logelr_mean",
                "logelr_sd",
                "origin_sd",
                "development_sd",
                "gamma_sd",
                "chains",
                "tune",
                "draws",
                "target_accept",
                "max_depth",
                "n_sims",
                "seed",
                "column",
                "premium",
                "output",
                "posterior_draws",
            ] {
                assert!(keys.contains(&key), "{key} missing from {keys:?}");
            }
            assert!(
                provenance
                    .parameters
                    .iter()
                    .any(|(k, v)| k == "output" && v == output)
            );
        }
        let crc = quick(MeyersModel::Crc, 5)
            .fit(&comauto(), "paid", "premium")
            .unwrap();
        assert_eq!(crc.reserves.provenance().model, "meyers_crc");
        let cay = quick(MeyersModel::Cay, 5)
            .fit(&comauto(), "paid", "premium")
            .unwrap();
        assert_eq!(cay.model(), MeyersModel::Cay);
        assert_eq!(cay.posterior.names()[19], "rho");
        let rho = cay.posterior.parameter(19).concat();
        assert!(rho.iter().all(|r| r.abs() < 1.0));
        let provenance = cay.reserves.provenance();
        assert_eq!(provenance.model, "meyers_cay");
        assert!(
            provenance
                .parameters
                .iter()
                .any(|(k, v)| k == "rho_shape" && v == "2")
        );
        assert!(
            !provenance.parameters.iter().any(|(k, _)| k == "gamma_sd"),
            "{:?}",
            provenance.parameters
        );
        // The oldest origin keeps its value; the CAY chain starts from its
        // observed residual.
        let oldest = vec![Period::year(1988).into()];
        assert_eq!(cay.ultimates.marginal(&oldest).unwrap().mean(), 3912.0);
        assert_eq!(cay.loo().unwrap().pareto_k.len(), 55);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn errors() {
        let tri = comauto();
        for (name, model) in [
            (
                "logelr_sd",
                Meyers {
                    logelr_sd: 0.0,
                    ..Meyers::default()
                },
            ),
            (
                "origin_sd",
                Meyers {
                    origin_sd: f64::NAN,
                    ..Meyers::default()
                },
            ),
            (
                "development_sd",
                Meyers {
                    development_sd: -1.0,
                    ..Meyers::default()
                },
            ),
            (
                "gamma_sd",
                Meyers {
                    gamma_sd: f64::INFINITY,
                    ..Meyers::default()
                },
            ),
            (
                "logelr_mean",
                Meyers {
                    logelr_mean: f64::NAN,
                    ..Meyers::default()
                },
            ),
            (
                "n_sims",
                Meyers {
                    n_sims: 0,
                    ..Meyers::default()
                },
            ),
            (
                "rho_shape",
                Meyers {
                    model: MeyersModel::Cay,
                    rho_shape: 0.0,
                    ..Meyers::default()
                },
            ),
        ] {
            let e = model.fit(&tri, "paid", "premium").unwrap_err();
            assert!(
                matches!(e, Error::InvalidSetting { name: n, .. } if n == name),
                "{name}: {e}"
            );
        }
        assert!(matches!(
            Meyers::default().fit(&tri, "incurred", "premium"),
            Err(Error::UnknownColumn(_))
        ));

        // A non-positive cell: 1990 at 24 months.
        let mut rows = PAID;
        rows[2][1] = 0.0;
        let e = quick(MeyersModel::Crc, 0)
            .fit(&triangle(&rows, &PREMIUM), "paid", "premium")
            .unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("origin 1990 at development age 24 months is 0")),
            "{e}"
        );
        // A missing or non-positive premium.
        let mut premium = PREMIUM;
        premium[4] = 0.0;
        let e = quick(MeyersModel::Crc, 0)
            .fit(&triangle(&PAID, &premium), "paid", "premium")
            .unwrap_err();
        assert!(
            matches!(&e, Error::InvalidExposure { origin, .. } if origin == "1992"),
            "{e}"
        );
        // Too few cells: 3 origins, 6 cells, 3 + 2 + 3 = 8 parameters.
        let small = triangle(&PAID[..3], &PREMIUM[..3]);
        let e = quick(MeyersModel::Crc, 0)
            .fit(&small, "paid", "premium")
            .unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("6 observed cells for 8 parameters")),
            "{e}"
        );
        // CAY without the previous origin's cell: 1992 stops at 36 months,
        // 1993 runs to 60.
        let (mut origin, mut ages, mut paid, mut prem) = (vec![], vec![], vec![], vec![]);
        for (w, row) in PAID.iter().enumerate() {
            let cells = if w == 4 { 3 } else { 10 - w };
            for (d, &v) in row.iter().enumerate().take(cells) {
                origin.push(Month::january(1988 + w as i32));
                ages.push(12 * (d as crate::Lag + 1));
                paid.push(v);
                prem.push(PREMIUM[w]);
            }
        }
        let ragged = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let e = quick(MeyersModel::Cay, 0)
            .fit(&ragged, "paid", "premium")
            .unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with(
                "origin 1993 is observed at development age 48 months but origin 1992 is not"
            )),
            "{e}"
        );
        // CRC takes the same triangle.
        quick(MeyersModel::Crc, 0)
            .fit(&ragged, "paid", "premium")
            .unwrap();

        let origin = [2020, 2020, 2021].map(Month::january);
        let two = Triangle::from_long(&Long {
            keys: &[("lob", &["a", "a", "a", "b", "b", "b"])],
            origin: &[origin, origin].concat(),
            development: DevelopmentColumn::Age(&[12, 24, 12, 12, 24, 12]),
            values: &[
                ("paid", &[1.0, 2.0, 1.0, 1.0, 2.0, 1.0]),
                ("premium", &[3.0; 6]),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        assert!(matches!(
            quick(MeyersModel::Crc, 0).fit(&two, "paid", "premium"),
            Err(Error::MultipleSegments(2))
        ));
    }

    /// Group 353's paid triangle twice, as lines `a` and `b` of a `lob`
    /// key, with its premium.
    fn twins() -> Triangle {
        let (mut keys, mut origin, mut ages, mut paid, mut prem) =
            (vec![], vec![], vec![], vec![], vec![]);
        for line in ["a", "b"] {
            for (w, row) in PAID.iter().enumerate() {
                for (d, &v) in row.iter().enumerate().take(10 - w) {
                    keys.push(line);
                    origin.push(Month::january(1988 + w as i32));
                    ages.push(12 * (d as crate::Lag + 1));
                    paid.push(v);
                    prem.push(PREMIUM[w]);
                }
            }
        }
        Triangle::from_long(&Long {
            keys: &[("lob", &keys)],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    /// Each line's total of `pd`, simulation by simulation.
    fn line_totals(pd: &PredictiveDistribution) -> Vec<Vec<f64>> {
        let by = pd.aggregate(&["lob"]).unwrap();
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

    /// Every row's ultimates less its reserves are the latest values.
    fn assert_ultimates_follow_reserves(fits: &MeyersFits) {
        let latest: Vec<f64> = fits
            .segments
            .fits
            .iter()
            .flat_map(|s| s.chain_ladder.latest.clone())
            .collect();
        let n = latest.len();
        for (u, r) in fits
            .ultimates
            .draw_matrix()
            .chunks_exact(n)
            .zip(fits.reserves.draw_matrix().chunks_exact(n))
        {
            for ((u, r), l) in u.iter().zip(r).zip(&latest) {
                assert_eq!(*r, u - l);
            }
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn each_segment_has_its_own_seeds() {
        let tri = twins();
        let fits = quick(MeyersModel::Crc, 9)
            .fit_segments(&tri, "paid", "premium")
            .unwrap();
        for pd in [&fits.reserves, &fits.ultimates] {
            assert_eq!(pd.dims(), ["lob", "origin"]);
            assert_eq!(pd.n_components(), 20);
            assert_eq!(pd.n_sims(), 2_000);
        }
        assert_eq!(fits.segments.len(), 2);
        let (a, b) = (&fits.segments.fits[0], &fits.segments.fits[1]);
        assert_eq!(a.premium, PREMIUM);
        assert_eq!(a.model(), MeyersModel::Crc);
        assert_ne!(a.posterior.values(), b.posterior.values());
        // Not the single-segment fit's seeds either.
        let alone = quick(MeyersModel::Crc, 9)
            .fit(&comauto(), "paid", "premium")
            .unwrap();
        assert_ne!(a.posterior.values(), alone.posterior.values());
        assert_ultimates_follow_reserves(&fits);
        // Shared streams would make the twins' totals almost perfectly
        // rank correlated; their own make them independent.
        let totals = line_totals(&fits.reserves);
        let r = spearman(&totals[0], &totals[1]);
        assert!(r.abs() < 4.5 / 2_000f64.sqrt(), "rank correlation {r}");

        let long = fits.to_long();
        assert_eq!(long.n_rows(), 20);
        for column in ["premium", "mean", "ultimate_mean", "ultimate_std_dev"] {
            assert!(long.column(column).is_some(), "{column}");
        }
        let ulr_mean: Vec<f64> = (long.column("ultimate_mean").unwrap().iter())
            .zip(PREMIUM.iter().chain(&PREMIUM))
            .map(|(u, p)| u / p)
            .collect();
        assert_eq!(long.column("ulr_mean").unwrap(), ulr_mean);
        let t = fits.totals();
        assert_eq!(t.n_rows(), 2);
        let premium: f64 = PREMIUM.iter().sum();
        assert_eq!(t.column("premium").unwrap(), [premium, premium]);
        let ultimate_mean = t.column("ultimate_mean").unwrap();
        assert_eq!(
            t.column("ulr_mean").unwrap(),
            [ultimate_mean[0] / premium, ultimate_mean[1] / premium]
        );
        let mean_b = totals[1].iter().sum::<f64>() / 2e3;
        assert!((t.column("mean").unwrap()[1] / mean_b - 1.0).abs() < 1e-9);
        assert_eq!(fits.development_table().n_rows(), 20);
        let one = fits.segment(&[("lob", "b")]).unwrap();
        assert_eq!(one.reserves.n_components(), 10);
        assert_eq!(one.ultimates.n_components(), 10);
        assert_eq!(one.segments.fits[0].posterior, b.posterior);
        assert_eq!(b.summary().unwrap().len(), 29);
        assert_eq!(b.loo().unwrap().pareto_k.len(), 55);

        for (pd, output) in [(&fits.reserves, "reserves"), (&fits.ultimates, "ultimates")] {
            let provenance = pd.provenance();
            assert_eq!(provenance.model, "meyers_crc");
            for (key, value) in [
                ("segments", "2"),
                ("output", output),
                ("dependence", "Independent"),
            ] {
                assert!(
                    provenance.parameters.contains(&(key.into(), value.into())),
                    "{key} {value}: {:?}",
                    provenance.parameters
                );
            }
        }

        let again = quick(MeyersModel::Crc, 9)
            .fit_segments(&tri, "paid", "premium")
            .unwrap();
        assert_eq!(again.reserves.draw_matrix(), fits.reserves.draw_matrix());
        assert_eq!(again.ultimates.draw_matrix(), fits.ultimates.draw_matrix());
    }

    #[test]
    fn synchronized_segments_are_an_error() {
        let model = Meyers {
            dependence: SegmentDependence::Synchronized,
            ..quick(MeyersModel::Csr, 0)
        };
        let e = model.fit_segments(&twins(), "paid", "premium").unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("Synchronized dependence")),
            "{e}"
        );
    }

    #[test]
    fn a_bad_rank_correlation_fails_before_sampling() {
        // With a sampler this long, any sampling before the check would
        // not finish.
        let slow = Meyers {
            sampler: Sampler {
                tune: 1_000_000,
                draws: 1_000_000,
                ..Sampler::default()
            },
            ..Meyers::default()
        };
        for spearman in [
            vec![1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0],
            vec![0.9, 0.5, 0.5, 1.0],
            vec![1.0, 0.5, 0.4, 1.0],
            vec![1.0, 1.5, 1.5, 1.0],
        ] {
            let model = Meyers {
                dependence: SegmentDependence::RankCorrelation { spearman },
                ..slow.clone()
            };
            let e = model.fit_segments(&twins(), "paid", "premium").unwrap_err();
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
        // A bad setting fails first too.
        let e = Meyers {
            n_sims: 0,
            ..slow.clone()
        }
        .fit_segments(&twins(), "paid", "premium")
        .unwrap_err();
        assert!(
            matches!(e, Error::InvalidSetting { name: "n_sims", .. }),
            "{e}"
        );
        assert!(matches!(
            slow.fit_segments(&twins(), "paid", "exposure"),
            Err(Error::UnknownColumn(_))
        ));
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn rank_correlation_moves_reserves_and_ultimates_together() {
        let tri = twins();
        let independent = quick(MeyersModel::Cay, 10)
            .fit_segments(&tri, "paid", "premium")
            .unwrap();
        let ranked = Meyers {
            dependence: SegmentDependence::RankCorrelation {
                spearman: vec![1.0, 0.8, 0.8, 1.0],
            },
            ..quick(MeyersModel::Cay, 10)
        }
        .fit_segments(&tri, "paid", "premium")
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
            assert_eq!(rows(&independent.ultimates, s), rows(&ranked.ultimates, s));
        }
        assert_ultimates_follow_reserves(&ranked);
        let totals = line_totals(&ranked.reserves);
        let r = spearman(&totals[0], &totals[1]);
        assert!((r - 0.8).abs() < 0.05, "rank correlation {r}");
        assert!(
            ranked
                .reserves
                .provenance()
                .parameters
                .iter()
                .any(|(k, _)| k == "rank_correlation")
        );
    }

    /// Group 353's paid cells as a frame, with its premium as exposure.
    fn frame() -> TriangleFrame {
        TriangleFrame::new(&comauto(), "paid", Some(&PREMIUM)).unwrap()
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_backtest_reports_diagnostics_and_still_scores_a_short_run() {
        // One chain of 50 kept draws cannot reach 100 effective draws per
        // chain, so every split warns.
        let short = Meyers {
            sampler: Sampler {
                chains: 1,
                tune: 50,
                draws: 50,
                ..Meyers::default().sampler
            },
            ..Meyers::default()
        };
        let bt = crate::diagonal_backtest(&frame(), &[&short], 2, 200, 5, 0.9).unwrap();
        assert_eq!(bt.models(), ["meyers_crc"]);
        assert_eq!(bt.diagnostics(0).len(), 2);
        for d in bt.diagnostics(0) {
            let d = d.as_ref().expect("a Bayesian model reports diagnostics");
            assert!(!d.warnings.is_empty(), "{d:?}");
        }
        let warnings = bt.warnings();
        assert!(warnings.iter().any(|(m, s, _)| (*m, *s) == (0, 0)));
        assert!(warnings.iter().any(|(m, s, _)| (*m, *s) == (0, 1)));
        // Scored all the same.
        assert!(bt.split_scores(0, 0).iter().all(|v| v.is_finite()));
        assert!(bt.split_scores(0, 3).iter().all(|v| v.is_finite()));
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_backtest_model_is_the_fit_on_the_train_rows() {
        // The latest diagonal held out: the train rows are the triangle of
        // the first nine origins without it (row 10 w + d is origin w at
        // age d).
        let cells = frame();
        let held: Vec<usize> = (1..9).map(|w| w * 10 + 9 - w).collect();
        let train: Vec<usize> = cells
            .observed()
            .iter()
            .copied()
            .filter(|&r| r / 10 + r % 10 < 9)
            .collect();
        let short = triangle(&PAID[..9], &PREMIUM[..9]);
        for kind in [MeyersModel::Crc, MeyersModel::Csr] {
            let model = quick(kind, 12);
            assert_eq!(
                TriangleModel::name(&model),
                format!("meyers_{}", kind.name())
            );
            let f = model.forecast(&cells, &train, &held, 2_000, 12).unwrap();
            let fit = model.fit(&short, "paid", "premium").unwrap();
            // The same posterior: each test row's mean and log density from
            // the fit's draws, written out here.
            let p = fit.posterior.dim();
            let lpd = f.log_density.clone().unwrap();
            for (j, w) in (1..9).enumerate() {
                let d = 9 - w;
                let (before, after) = (PAID[w][d - 1], PAID[w][d]);
                let (mut mean, mut dens) = (0.0, 0.0);
                let n = fit.posterior.n_draws() as f64;
                for nat in fit.posterior.values().chunks_exact(p) {
                    let beta = if d < 8 { nat[9 + d] } else { 0.0 };
                    let g = if kind == MeyersModel::Csr {
                        (1.0 - nat[17]).powi(w as i32)
                    } else {
                        1.0
                    };
                    let mu = PREMIUM[w].ln() + nat[0] + nat[w] + beta * g;
                    let sigma = nat[p - 9 + d];
                    mean += ((mu + 0.5 * sigma * sigma).exp() - before) / n;
                    let z = (after.ln() - mu) / sigma;
                    dens +=
                        (-0.5 * z * z).exp() / (after * sigma * std::f64::consts::TAU.sqrt()) / n;
                }
                assert!(
                    (f.mean[j] / mean - 1.0).abs() < 1e-9,
                    "{kind:?} {w}: {} vs {mean}",
                    f.mean[j]
                );
                assert!(
                    (lpd[j] - dens.ln()).abs() < 1e-9,
                    "{kind:?} {w}: {} vs {}",
                    lpd[j],
                    dens.ln()
                );
                // The draws' mean is the mean, up to Monte Carlo error.
                let x: Vec<f64> = f
                    .distribution
                    .draw_matrix()
                    .chunks_exact(8)
                    .map(|r| r[j])
                    .collect();
                let m = x.iter().sum::<f64>() / x.len() as f64;
                let sd = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
                assert!(
                    (m - mean).abs() < 5.0 * sd / (x.len() as f64).sqrt(),
                    "{m} vs {mean}"
                );
            }
            assert_eq!(f.distribution.n_components(), 8);
            let d = f
                .diagnostics
                .as_ref()
                .expect("a Bayesian model reports diagnostics");
            assert!(d.max_rhat.is_finite() && d.max_rhat < 1.1, "{d:?}");
            assert_eq!(f.distribution.n_sims(), 2_000);
            let again = model.forecast(&cells, &train, &held, 2_000, 12).unwrap();
            assert_eq!(again.diagnostics.as_ref(), Some(d));
            assert_eq!(
                again.distribution.draw_matrix(),
                f.distribution.draw_matrix()
            );
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_next_cell_simulates_as_the_fit() {
        // On the whole triangle, origin 1989's cell at 120 months is its
        // ultimate: the backtest's draws of its increment are the fit's
        // reserve draws, simulation by simulation (integer cells, so the
        // rebuilt cumulative values are exact).
        let cells = frame();
        let train = cells.observed().to_vec();
        for kind in [MeyersModel::Crc, MeyersModel::Cay] {
            let model = quick(kind, 13);
            let f = model.forecast(&cells, &train, &[19], 1_000, 13).unwrap();
            let fit = Meyers {
                n_sims: 1_000,
                ..model.clone()
            }
            .fit(&comauto(), "paid", "premium")
            .unwrap();
            let reserve: Vec<f64> = fit
                .reserves
                .draw_matrix()
                .chunks_exact(10)
                .map(|r| r[1])
                .collect();
            for (a, b) in f.distribution.draw_matrix().iter().zip(&reserve) {
                assert!(
                    (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                    "{kind:?}: {a} vs {b}"
                );
            }
            // A future cell has no response, so no density there.
            assert_eq!(f.log_density.unwrap(), [f64::NEG_INFINITY]);
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn an_origin_path_telescopes() {
        // Origin 1997's nine future cells, one path per simulation: the
        // increments sum to its ultimate less its latest value, with the
        // fit's reserve's mean and spread (independent cells would add up
        // nine variances).
        let cells = frame();
        let train = cells.observed().to_vec();
        let test: Vec<usize> = (91..100).collect();
        let model = quick(MeyersModel::Crc, 14);
        let f = model.forecast(&cells, &train, &test, 4_000, 14).unwrap();
        assert!(f.log_density.is_none());
        let sums: Vec<f64> = f
            .distribution
            .draw_matrix()
            .chunks_exact(9)
            .map(|r| r.iter().sum())
            .collect();
        let fit = Meyers {
            n_sims: 4_000,
            ..model.clone()
        }
        .fit(&comauto(), "paid", "premium")
        .unwrap();
        let reserve = fit
            .reserves
            .marginal(&vec![Period::year(1997).into()])
            .unwrap();
        let m = sums.iter().sum::<f64>() / sums.len() as f64;
        let sd = (sums.iter().map(|v| (v - m).powi(2)).sum::<f64>() / sums.len() as f64).sqrt();
        let se = reserve.std_dev() / (sums.len() as f64).sqrt();
        assert!(
            (m - reserve.mean()).abs() < 6.0 * se,
            "{m} vs {}",
            reserve.mean()
        );
        assert!(
            (sd / reserve.std_dev() - 1.0).abs() < 0.1,
            "{sd} vs {}",
            reserve.std_dev()
        );
        // The analytic means telescope too.
        let total: f64 = f.mean.iter().sum();
        assert!((total / m - 1.0).abs() < 0.05, "{total} vs {m}");
    }

    #[test]
    fn cay_paths_take_the_previous_origins_residual() {
        // A fixed draw with a large sigma and rho; origins 1996 and 1997
        // both on paths at 36 months, so 1997's mean takes 1996's drawn
        // residual, while 1996's takes 1995's observed one. 1997's path
        // runs from 24 months.
        let cells = frame();
        let train = cells.observed().to_vec();
        let test = [82, 91, 92];
        let rebuilt = Rebuilt::new(&cells, &train, &test).unwrap();
        let model = prepared(MeyersModel::Cay);
        let (k, rho, sigma) = (model.after_beta(), 0.7, 0.4);
        let mut nat = model.natural(&model.start());
        nat[k] = rho;
        for s in &mut nat[model.u0()..] {
            *s = sigma;
        }
        let at = cell_index(&model.cells, 10, 10);
        let mut residuals = Vec::new();
        model.cay_residuals(&nat, &mut residuals);
        let (m8, v8) = model.log_moments(&nat, &residuals, &at, &PREMIUM, 8, 2);
        let (m9, v9) = model.log_moments(&nat, &residuals, &at, &PREMIUM, 9, 2);
        assert!((v8 - sigma * sigma).abs() < 1e-15);
        assert!((v9 - (1.0 + rho * rho) * sigma * sigma).abs() < 1e-15);
        let ages = rebuilt.path_ages();
        assert_eq!(ages, [1, 2]);
        let mut rng = StreamRng::new(3, 0);
        let mut row = [0.0; 3];
        let n = 100_000;
        let (mut logs8, mut logs9) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            model.draw_paths(&nat, &rebuilt, &at, &ages, &mut rng, &mut row);
            logs8.push((row[0] + rebuilt.last_value[8]).ln());
            logs9.push((row[1] + row[2] + rebuilt.last_value[9]).ln());
        }
        let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
        let var = |x: &[f64]| {
            let m = mean(x);
            x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64
        };
        for (logs, m, v) in [(&logs8, m8, v8), (&logs9, m9, v9)] {
            assert!((mean(logs) - m).abs() < 5.0 * (v / n as f64).sqrt());
            assert!((var(logs) / v - 1.0).abs() < 0.02, "{} vs {v}", var(logs));
        }
        // 1997's log value moves with 1996's residual: covariance
        // rho sigma^2.
        let (a, b) = (mean(&logs8), mean(&logs9));
        let cov = logs8
            .iter()
            .zip(&logs9)
            .map(|(x, y)| (x - a) * (y - b))
            .sum::<f64>()
            / (n - 1) as f64;
        assert!((cov / (rho * sigma * sigma) - 1.0).abs() < 0.03, "{cov}");
    }

    #[test]
    fn backtest_errors() {
        let cells = frame();
        let train = cells.observed().to_vec();
        let model = quick(MeyersModel::Crc, 0);
        // A test row at or before its origin's latest train row.
        let e = model.forecast(&cells, &train, &[5], 100, 0).unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.contains("not after")),
            "{e}"
        );
        // Origin 1988 without its 12-month row.
        let gap: Vec<usize> = train.iter().copied().filter(|&r| r != 0).collect();
        let e = model.forecast(&cells, &gap, &[19], 100, 0).unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m)
                if m.starts_with("origin 1988 has no train row at development age 12")),
            "{e}"
        );
        // Origin 1990 without train rows, between 1989 and 1991.
        let gap: Vec<usize> = train.iter().copied().filter(|&r| r / 10 != 2).collect();
        let e = model.forecast(&cells, &gap, &[19], 100, 0).unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("origin 1990 has no train row")),
            "{e}"
        );
        // The newest origin without its only cell.
        let without: Vec<usize> = train.iter().copied().filter(|&r| r != 90).collect();
        let e = model.forecast(&cells, &without, &[91], 100, 0).unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.contains("origin without a train row")),
            "{e}"
        );
        // An age past the oldest train age.
        let young: Vec<usize> = train.iter().copied().filter(|&r| r % 10 < 8).collect();
        let e = model.forecast(&cells, &young, &[18], 100, 0).unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.contains("an age without a train row")),
            "{e}"
        );
        // Settings are checked first.
        let e = Meyers {
            gamma_sd: 0.0,
            ..model
        }
        .forecast(&cells, &train, &[19], 100, 0)
        .unwrap_err();
        assert!(
            matches!(
                e,
                Error::InvalidSetting {
                    name: "gamma_sd",
                    ..
                }
            ),
            "{e}"
        );
    }

    // The one-year view (decision 19).

    /// A fully developed 4 x 4 square of paid with premium.
    fn square() -> Triangle {
        let rows: [[f64; 4]; 4] = [
            [357.0, 1124.0, 1735.0, 2218.0],
            [352.0, 1236.0, 2170.0, 3353.0],
            [290.0, 1292.0, 2219.0, 3235.0],
            [310.0, 1418.0, 2195.0, 3757.0],
        ];
        let (mut origin, mut ages, mut paid, mut prem) = (vec![], vec![], vec![], vec![]);
        for (w, row) in rows.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                origin.push(Month::january(2001 + w as i32));
                ages.push(12 * (d as crate::Lag + 1));
                paid.push(v);
                prem.push(5000.0 + 500.0 * w as f64);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    /// The pieces a [`year::MeyersYear`] borrows: the segment, its chain
    /// ladder and premiums, and the model.
    fn year_parts(meyers: &Meyers, tri: &Triangle) -> (Segment, ChainLadderFit, Vec<f64>, Model) {
        let segment = tri.segment("paid").unwrap();
        let cl = ChainLadder::default()
            .fit_segment(&segment, &segment.ages)
            .unwrap();
        let premium =
            origin_exposure(&segment, &tri.segment("premium").unwrap(), "premium").unwrap();
        let model = meyers.model_of(&segment, &cl, &premium).unwrap();
        (segment, cl, premium, model)
    }

    /// The year's pieces without sampling: the importance ratio is the
    /// change in the log posterior, the expected ultimate of CRC and CSR
    /// does not depend on the year, and the opening and closing reserves
    /// are the means of the lifetime simulation's reserve given a draw
    /// (with CAY's integrated residual), the closing one on the triangle
    /// extended by the year.
    #[test]
    fn the_year_follows_the_model() {
        use crate::bayes::one_year::YearModel;
        let tri = comauto();
        for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
            let meyers = Meyers {
                model: kind,
                ..Meyers::default()
            };
            let (segment, cl, premium, model) = year_parts(&meyers, &tri);
            let year =
                year::MeyersYear::new(&meyers, &model, &tri, &segment, &cl, &premium).unwrap();
            assert_eq!(year.n_year(), 9, "{kind:?}");
            assert_eq!(year.n_origins(), 10);
            let theta = off_mode(&model);
            let mut other = theta.clone();
            for (j, v) in other.iter_mut().enumerate() {
                *v -= 0.02 * (j as f64 % 5.0 - 2.0);
            }
            let mut rng = StreamRng::new(3, 0);
            let mut values = vec![0.0; 9];
            year.draw_year(&theta, &mut rng, &mut values);
            assert!(values.iter().all(|v| v.is_finite() && *v > 0.0));
            // 1. The ratio is the change in the log posterior.
            let change = |t: &[f64]| {
                year.log_posterior(t, &values)
                    - year.log_posterior(t, &[])
                    - year.year_log_likelihood(t, &values)
            };
            let (a, b) = (change(&theta), change(&other));
            assert!((a - b).abs() < 1e-9 * (1.0 + a.abs()), "{kind:?}: {a} {b}");
            // 2. The year's payment and closing reserve.
            let mut opening = vec![0.0; 10];
            let mut paid = vec![0.0; 10];
            let mut closing = vec![0.0; 10];
            year.opening(&theta, &mut opening);
            year.paid(&values, &mut paid);
            year.closing(&theta, &values, &mut closing);
            assert_eq!((opening[0], paid[0], closing[0]), (0.0, 0.0, 0.0));
            // 1989 reaches the oldest age in the year.
            assert_eq!(closing[1], 0.0);
            assert_eq!(paid[1], values[0] - cl.latest[1]);
            if kind != MeyersModel::Cay {
                for w in 2..10 {
                    let moved = closing[w] + paid[w];
                    assert!(
                        (moved - opening[w]).abs() < 1e-9 * opening[w].abs().max(1.0),
                        "{kind:?} origin {w}: {moved} {}",
                        opening[w]
                    );
                }
            }
            // 3. The opening reserve is the lifetime reserve's mean given
            // the draw.
            let nat = model.natural(&theta);
            let closed: Vec<bool> = cl.latest_position.iter().map(|&d| d == 9).collect();
            let n = 50_000;
            let mut sum = [0.0; 10];
            let mut sq = [0.0; 10];
            let mut row = vec![0.0; 10];
            let mut rng = StreamRng::new(5, 0);
            for _ in 0..n {
                model.draw_ultimates(&nat, &premium, &closed, &cl.latest, &mut rng, &mut row);
                for w in 0..10 {
                    let r = row[w] - cl.latest[w];
                    sum[w] += r;
                    sq[w] += r * r;
                }
            }
            for w in 1..10 {
                let mean = sum[w] / n as f64;
                let se = ((sq[w] / n as f64 - mean * mean) / n as f64).sqrt();
                assert!(
                    (mean - opening[w]).abs() < 5.0 * se,
                    "{kind:?} origin {w}: simulated {mean}, opening {} (se {se})",
                    opening[w]
                );
            }
            // 4. The closing reserve is the lifetime reserve's mean given
            // the draw on the triangle extended by the year: each origin's
            // latest value is its year value, and 1989, which the year
            // takes to the oldest age, is closed. In CAY this checks the
            // conditioning on the year independently of `closing`: `rho`
            // times 1989's residual at the oldest age for 1990, and the
            // integrated residual of the previous origin for the others.
            // At a draw with `rho` 0.6 (CAY) and `sigma_10^2` about 0.5,
            // where both terms move the closing reserve by many standard
            // errors (near the mode they are within the noise).
            let mut wide = theta.clone();
            if kind == MeyersModel::Cay {
                wide[model.after_beta()] = 0.6f64.atanh();
            }
            wide[model.u0() + 9] = 0.0;
            year.closing(&wide, &values, &mut closing);
            let nat = model.natural(&wide);
            let mut latest = cl.latest.clone();
            for (j, &v) in values.iter().enumerate() {
                latest[j + 1] = v;
            }
            let closed: Vec<bool> = cl.latest_position.iter().map(|&d| d + 1 >= 9).collect();
            let (mut sum, mut sq) = ([0.0; 10], [0.0; 10]);
            let mut rng = StreamRng::new(7, 0);
            for _ in 0..n {
                model.draw_ultimates(&nat, &premium, &closed, &latest, &mut rng, &mut row);
                for w in 0..10 {
                    let r = row[w] - latest[w];
                    sum[w] += r;
                    sq[w] += r * r;
                }
            }
            for w in 2..10 {
                let mean = sum[w] / n as f64;
                let se = ((sq[w] / n as f64 - mean * mean) / n as f64).sqrt();
                assert!(
                    (mean - closing[w]).abs() < 5.0 * se,
                    "{kind:?} origin {w}: simulated {mean}, closing {} (se {se})",
                    closing[w]
                );
            }
        }
        // A square has no year.
        let closed = square();
        let meyers = Meyers::default();
        let (segment, cl, premium, model) = year_parts(&meyers, &closed);
        let year =
            year::MeyersYear::new(&meyers, &model, &closed, &segment, &cl, &premium).unwrap();
        assert_eq!(year.n_year(), 0);
    }

    /// The one-year conformance suite (decision 17) on CRC, CSR and CAY,
    /// with a fully developed square as the triangle without a year.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_one_year_view_conforms() {
        for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
            let meyers = quick(kind, 4);
            let fitted: Vec<(Triangle, Segment, MeyersSegment)> = [comauto(), square()]
                .into_iter()
                .map(|tri| {
                    let segment = tri.segment("paid").unwrap();
                    let premium = tri.segment("premium").unwrap();
                    let (fit, _, _, _) = meyers
                        .fit_segment(
                            &segment,
                            &premium,
                            "paid",
                            "premium",
                            Stream::Chains,
                            Stream::Predictive,
                        )
                        .unwrap();
                    (tri, segment, fit)
                })
                .collect();
            let years: Vec<year::MeyersYear> = fitted
                .iter()
                .map(|(tri, segment, fit)| {
                    year::MeyersYear::new(
                        &meyers,
                        &fit.model,
                        tri,
                        segment,
                        &fit.chain_ladder,
                        &fit.premium,
                    )
                    .unwrap()
                })
                .collect();
            crate::bayes::one_year::conformance::check(
                &format!("Meyers {kind:?}"),
                &years[0],
                &fitted[0].2.sampled,
                &years[1],
                &fitted[1].2.sampled,
            );
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_one_year_view_reweights_the_fits_posterior() {
        let model = quick(MeyersModel::Csr, 6);
        let only = BayesOneYear {
            refit: Some(crate::bayes::OneYearRefit::ReweightOnly),
            ..BayesOneYear::default()
        };
        let view = model
            .one_year(&comauto(), "paid", "premium", &only)
            .unwrap();
        // The posterior and lifetime predictive are fit()'s, draw for draw.
        let fit = model.fit(&comauto(), "paid", "premium").unwrap();
        assert_eq!(view.fit.posterior, fit.posterior);
        assert_eq!(view.fit.reserves.draw_matrix(), fit.reserves.draw_matrix());
        assert_eq!(
            view.fit.ultimates.draw_matrix(),
            fit.ultimates.draw_matrix()
        );
        assert_eq!(view.cdr.dims(), ["origin"]);
        assert_eq!(view.cdr.n_components(), 10);
        assert_eq!(view.pareto_k.len(), 2_000);
        assert!(view.refitted.is_empty() && view.moment_matched.is_empty());
        // The settings' n_sims overrides the model's for the year only, and
        // gives the first simulations of the larger count.
        let fewer = model
            .one_year(
                &comauto(),
                "paid",
                "premium",
                &BayesOneYear {
                    n_sims: Some(500),
                    ..only
                },
            )
            .unwrap();
        assert_eq!(fewer.cdr.n_sims(), 500);
        assert_eq!(fewer.fit.reserves.n_sims(), 2_000);
        assert_eq!(fewer.pareto_k, view.pareto_k[..500]);
        assert_eq!(fewer.cdr.draw_matrix(), &view.cdr.draw_matrix()[..500 * 10]);
        // 1988 is fully developed.
        assert!(view.cdr.draw_matrix().chunks(10).all(|r| r[0] == 0.0));
        // The opening reserve is the lifetime reserve's posterior mean.
        let lifetime = fit.reserves.total();
        let opening: f64 = view.opening_reserve.iter().sum();
        let se = lifetime.std_dev() / 2_000f64.sqrt();
        assert!(
            (opening - lifetime.mean()).abs() < 5.0 * se,
            "opening {opening}, lifetime {} (se {se})",
            lifetime.mean()
        );
        assert!(view.cdr.std_dev() < fit.reserves.std_dev());
        assert_eq!(view.cdr.provenance().model, "meyers_csr_one_year");
        let long = view.to_long();
        assert_eq!(long.n_rows(), 10);
        assert!(long.column("premium").is_some());
        assert!(view.totals().column("flagged_share").is_some());
        // The default moment-matches the flagged simulations, refitting
        // only those matching cannot rescue.
        let matched = Meyers {
            n_sims: 300,
            ..model.clone()
        }
        .one_year(&comauto(), "paid", "premium", &BayesOneYear::default())
        .unwrap();
        let flagged: Vec<usize> = (0..300).filter(|&i| view.pareto_k[i] > 0.7).collect();
        assert_eq!(matched.pareto_k, view.pareto_k[..300]);
        let mut either = matched.moment_matched.clone();
        either.extend(&matched.refitted);
        either.sort_unstable();
        assert_eq!(either, flagged);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_one_year_view_over_segments() {
        let only = BayesOneYear {
            refit: Some(crate::bayes::OneYearRefit::ReweightOnly),
            ..BayesOneYear::default()
        };
        let tri = twins();
        let model = quick(MeyersModel::Crc, 9);
        let views = model
            .one_year_segments(&tri, "paid", "premium", &only)
            .unwrap();
        assert_eq!(views.cdr.dims(), ["lob", "origin"]);
        assert_eq!(views.cdr.n_components(), 20);
        let lifetime = model.fit_segments(&tri, "paid", "premium").unwrap();
        for (a, b) in views.segments.fits.iter().zip(&lifetime.segments.fits) {
            assert_eq!(a.fit.posterior, b.posterior);
            assert_eq!(a.pareto_k.len(), 2_000);
        }
        let totals = line_totals(&views.cdr);
        let r = spearman(&totals[0], &totals[1]);
        assert!(r.abs() < 4.5 / 2_000f64.sqrt(), "rank correlation {r}");
        assert_eq!(views.to_long().n_rows(), 20);
        assert_eq!(views.totals().n_rows(), 2);
        assert_eq!(
            views.segment(&[("lob", "b")]).unwrap().cdr.n_components(),
            10
        );
        let correlated = Meyers {
            dependence: SegmentDependence::RankCorrelation {
                spearman: vec![1.0, 0.7, 0.7, 1.0],
            },
            ..model
        };
        let views = correlated
            .one_year_segments(&tri, "paid", "premium", &only)
            .unwrap();
        let totals = line_totals(&views.cdr);
        let r = spearman(&totals[0], &totals[1]);
        assert!((r - 0.7).abs() < 0.05, "rank correlation {r}");
    }

    #[test]
    fn one_year_errors() {
        let e = Meyers::default()
            .one_year(
                &comauto(),
                "paid",
                "premium",
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
        let e = Meyers {
            dependence: SegmentDependence::Synchronized,
            ..Meyers::default()
        }
        .one_year_segments(&twins(), "paid", "premium", &BayesOneYear::default())
        .unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("Synchronized dependence")),
            "{e}"
        );
        assert!(matches!(
            Meyers::default().one_year(&twins(), "paid", "premium", &BayesOneYear::default()),
            Err(Error::MultipleSegments(2))
        ));
        let e = Meyers::default()
            .one_year(
                &comauto(),
                "paid",
                "premium",
                &BayesOneYear {
                    n_sims: Some(0),
                    ..BayesOneYear::default()
                },
            )
            .unwrap_err();
        assert!(
            matches!(e, Error::InvalidSetting { name: "n_sims", .. }),
            "{e}"
        );
    }

    #[test]
    fn the_one_year_view_runs_at_most_2000_simulations_by_default() {
        let s = BayesOneYear::default();
        assert_eq!(Meyers::default().n_sims, 10_000);
        assert_eq!(Meyers::default().one_year_sims(&s), 2_000);
        let small = Meyers {
            n_sims: 500,
            ..Meyers::default()
        };
        assert_eq!(small.one_year_sims(&s), 500);
        let more = BayesOneYear {
            n_sims: Some(10_000),
            ..s
        };
        assert_eq!(Meyers::default().one_year_sims(&more), 10_000);
        assert_eq!(small.one_year_sims(&more), 10_000);
    }

    /// The position of the parameter `name` in a fit's posterior draws.
    fn position(fit: &MeyersFit, name: &str) -> usize {
        position_in(&fit.posterior, name)
    }

    fn position_in(posterior: &PosteriorDraws, name: &str) -> usize {
        posterior
            .names()
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no parameter {name}"))
    }

    /// Simulation `i`'s posterior draw, found as the fit found it: the
    /// first uniform of the simulation's stream picks it, and the rest of
    /// the stream is what the ultimates drew from.
    fn replay(posterior: &PosteriorDraws, seed: u64, i: usize) -> (Vec<f64>, StreamRng) {
        let mut rng = StreamRng::new(seed, i as u64);
        let total = posterior.n_draws();
        let d = ((rng.next_open01() * total as f64) as usize).min(total - 1);
        (posterior.draw(d).to_vec(), rng)
    }

    /// The pattern of one posterior draw, written out from the model's
    /// definition (`reserving-bayes.md` decision 27) and the draw's named
    /// parameters, not from `Model::pattern_shares`: the ratio of the
    /// expected cumulative value at each age to the one at the oldest age.
    fn expected_pattern(
        posterior: &PosteriorDraws,
        draw: &[f64],
        kind: MeyersModel,
        n_origins: usize,
        n_dev: usize,
    ) -> Vec<f64> {
        let get = |name: &str| draw[position_in(posterior, name)];
        // CSR: origin w (0-based) develops with beta_d (1 - gamma)^w, and the
        // next origin after the last is w = n_origins.
        let speed = match kind {
            MeyersModel::Csr => (1.0 - get("gamma")).powi(n_origins as i32),
            _ => 1.0,
        };
        let log_mean = |d: usize| {
            let beta = if d + 1 < n_dev {
                get(&format!("beta[{}]", d + 1))
            } else {
                0.0
            };
            let sigma = get(&format!("sigma[{}]", d + 1));
            beta * speed + sigma * sigma / 2.0
        };
        (0..n_dev)
            .map(|d| (log_mean(d) - log_mean(n_dev - 1)).exp())
            .collect()
    }

    /// Every origin's ultimate by the model's definition from the draw and
    /// the rest of its stream (as `draw_ultimates` reads it), CAY's
    /// recursion included. Group 353's first origin is observed at the
    /// oldest age.
    fn expected_ultimates(
        fit: &MeyersFit,
        kind: MeyersModel,
        draw: &[f64],
        rng: &mut StreamRng,
    ) -> Vec<f64> {
        let get = |name: &str| draw[position(fit, name)];
        let sigma_m = get("sigma[10]");
        let rho = if kind == MeyersModel::Cay {
            get("rho")
        } else {
            0.0
        };
        let mut residual = 0.0;
        (0..10)
            .map(|w| {
                let alpha = if w > 0 {
                    get(&format!("alpha[{}]", w + 1))
                } else {
                    0.0
                };
                let mean = PREMIUM[w].ln() + get("logelr") + alpha + rho * residual;
                if w == 0 {
                    residual = fit.chain_ladder.latest[0].ln() - mean;
                    fit.chain_ladder.latest[0]
                } else {
                    residual = sigma_m * norm_quantile(rng.next_open01());
                    (mean + residual).exp()
                }
            })
            .collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn each_simulations_pattern_is_its_posterior_draws() {
        for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
            let model = quick(kind, 21);
            let fit = model.fit(&comauto(), "paid", "premium").unwrap();
            let seed = derive_seed(21, Stream::Predictive);
            let segment = fit.patterns.segment(&[]).unwrap();
            assert_eq!(fit.patterns.n_sims(), 2_000);
            assert_eq!(segment.ages(), (1..=10).map(|d| 12 * d).collect::<Vec<_>>());
            let ultimates = fit.ultimates.draw_matrix();
            let mut falling = 0;
            let (mut at, mut worst) = ([0usize; 9], 0.0f64);
            for i in 0..fit.patterns.n_sims() {
                let (draw, mut rng) = replay(&fit.posterior, seed, i);
                // The pattern is the draw's, recomputed from its parameters.
                let expected = expected_pattern(&fit.posterior, &draw, kind, 10, 10);
                let row = segment.draw(i);
                assert_eq!(row[9], 1.0);
                for (d, (p, e)) in row.iter().zip(&expected).enumerate() {
                    assert!(close(*p, *e), "{kind:?} sim {i} age {d}: {p} against {e}");
                }
                falling += usize::from(row.windows(2).any(|w| w[1] < w[0]));
                for (d, w) in row.windows(2).enumerate() {
                    if w[1] < w[0] {
                        at[d] += 1;
                        worst = worst.max(1.0 - w[1] / w[0]);
                    }
                }
                // And the ultimates are what the draw and the stream give
                // without the pattern: no random number went to it.
                let ult = expected_ultimates(&fit, kind, &draw, &mut rng);
                for (w, (u, e)) in ultimates[i * 10..(i + 1) * 10].iter().zip(&ult).enumerate() {
                    if kind == MeyersModel::Cay {
                        assert!(
                            close(*u, *e),
                            "{kind:?} sim {i} origin {w}: {u} against {e}"
                        );
                    } else {
                        assert_eq!(u.to_bits(), e.to_bits(), "{kind:?} sim {i} origin {w}");
                    }
                }
            }
            eprintln!(
                "{kind:?}: {falling} of 2000 patterns fall; by step {at:?}, worst relative fall {worst:.5}, mean pattern {:.4?}",
                segment.mean()
            );
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn csr_develops_faster_for_the_origin_after_the_last() {
        let fit = quick(MeyersModel::Csr, 22)
            .fit(&comauto(), "paid", "premium")
            .unwrap();
        let segment = fit.patterns.segment(&[]).unwrap();
        let seed = derive_seed(22, Stream::Predictive);
        let (draw, _) = replay(&fit.posterior, seed, 0);
        let gamma = draw[position(&fit, "gamma")];
        // The same draw with no speedup is the CRC pattern; the speedup moves
        // the early shares towards 1 where beta is negative.
        let crc = expected_pattern(&fit.posterior, &draw, MeyersModel::Crc, 10, 10);
        let csr = segment.draw(0);
        if gamma > 0.0 {
            assert!(csr[0] > crc[0], "{} against {}", csr[0], crc[0]);
        } else if gamma < 0.0 {
            assert!(csr[0] < crc[0]);
        }
    }

    #[test]
    fn the_pattern_of_a_hand_set_draw() {
        // Two origins and three ages: beta = (-1, -0.5, 0) and sigma =
        // (0.6, 0.4, 0.2), so the expected cumulative values are
        // exp(beta + sigma^2 / 2).
        let model = Model {
            n_origins: 2,
            n_dev: 3,
            kind: MeyersModel::Crc,
            cells: Vec::new(),
            logelr_mean: 0.0,
            logelr_sd: 1.0,
            origin_sd: 1.0,
            development_sd: 1.0,
            gamma_sd: 0.05,
            rho_shape: 2.0,
            priors: true,
        };
        // logelr, alpha[2], beta[1], beta[2], sigma[1..3]
        let nat = [-0.4, 0.1, -1.0, -0.5, 0.6, 0.4, 0.2];
        let mut shares = [0.0; 3];
        model.pattern_shares(&nat, &mut shares);
        let top = (0.2f64 * 0.2 / 2.0).exp();
        assert_eq!(shares[2], 1.0);
        assert!(close(shares[0], (-1.0 + 0.36 / 2.0f64).exp() / top));
        assert!(close(shares[1], (-0.5 + 0.16 / 2.0f64).exp() / top));
        // CSR at gamma = 0.5 scales beta by (1 - 0.5)^2 for the origin after
        // the second.
        let csr = Model {
            kind: MeyersModel::Csr,
            ..model
        };
        let nat = [-0.4, 0.1, -1.0, -0.5, 0.5, 0.6, 0.4, 0.2];
        csr.pattern_shares(&nat, &mut shares);
        assert!(close(shares[0], (-0.25 + 0.36 / 2.0f64).exp() / top));
        assert!(close(shares[1], (-0.125 + 0.16 / 2.0f64).exp() / top));
        assert_eq!(shares[2], 1.0);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn patterns_follow_the_reserves_through_a_rank_correlation() {
        let tri = twins();
        let independent = quick(MeyersModel::Csr, 23)
            .fit_segments(&tri, "paid", "premium")
            .unwrap();
        let ranked = Meyers {
            dependence: SegmentDependence::RankCorrelation {
                spearman: vec![1.0, 0.8, 0.8, 1.0],
            },
            ..quick(MeyersModel::Csr, 23)
        }
        .fit_segments(&tri, "paid", "premium")
        .unwrap();
        let rows = |pd: &PredictiveDistribution, s: usize| -> Vec<Vec<u64>> {
            pd.draw_matrix()
                .chunks(20)
                .map(|r| {
                    r[s * 10..(s + 1) * 10]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect()
                })
                .collect()
        };
        let mut moved = 0;
        for (s, label) in ["a", "b"].iter().enumerate() {
            let from = rows(&independent.reserves, s);
            let to = rows(&ranked.reserves, s);
            let p_from = independent.patterns.segment(&[label]).unwrap();
            let p_to = ranked.patterns.segment(&[label]).unwrap();
            for (i, row) in to.iter().enumerate() {
                let j = from.iter().position(|r| r == row).expect("a source row");
                assert_eq!(p_to.draw(i), p_from.draw(j), "segment {label}, row {i}");
                moved += usize::from(i != j);
            }
            // And the unreordered rows are the segment's own posterior draws.
            let seed = derive_seed(23, Stream::SegmentPredictive(s));
            let posterior = &independent.segments.fits[s].posterior;
            for i in (0..2_000).step_by(97) {
                let (draw, _) = replay(posterior, seed, i);
                let e = expected_pattern(posterior, &draw, MeyersModel::Csr, 10, 10);
                assert!(p_from.draw(i).iter().zip(&e).all(|(a, b)| close(*a, *b)));
            }
        }
        assert!(moved > 100, "the reorder moved only {moved} rows");
        // A segment picked from the fits keeps its patterns.
        let b = ranked.segment(&[("lob", "b")]).unwrap();
        assert_eq!(b.patterns.segments().len(), 1);
        assert_eq!(
            b.patterns.segment(&[]).unwrap(),
            ranked.patterns.segment(&["b"]).unwrap()
        );
        for s in ranked.patterns.segments() {
            assert!((0..2_000).all(|i| s.draw(i)[9] == 1.0));
        }
    }
}
