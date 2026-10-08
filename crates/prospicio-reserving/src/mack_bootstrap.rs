//! Mack's model bootstrapped, after England, Verrall and Wüthrich (2019),
//! Appendix 1, for the lifetime view ([`MackBootstrap::fit`]) and the
//! simulated one-year view ([`MackBootstrap::one_year`];
//! `docs/design/reserving-v02.md`, decision 8).
//!
//! Mack's model has `E[C_k+1 | C_k] = f_k C_k` and
//! `Var[C_k+1 | C_k] = sigma_k^2 C_k^(2 - alpha)`, `alpha` the development
//! estimator's (1 for volume weighting). Each simulation, on its own random
//! stream:
//!
//! 1. resamples the scaled bias-adjusted residuals of the link ratios
//!    `F = C_k+1 / C_k`,
//!    `r = sqrt(n_k / (n_k - 1)) C_k^(alpha / 2) (F - f_k) / sigma_k` with
//!    `n_k` the link ratios behind `f_k`, pooled over every factor with two
//!    or more, into a pseudo link ratio
//!    `F* = f_k + r* sigma_k / C_k^(alpha / 2)` for every observed link,
//!    and re-estimates each factor as their weighted average
//!    `f*_k = sum(C_k^alpha F*) / sum(C_k^alpha)` (parameter error);
//! 2. draws every cumulative value of the coming year in development order
//!    (the cells of [`crate::one_year_bootstrap`]), each from the one
//!    before it `C`, the observed latest value for the first, with mean
//!    `f*_k C` and variance `sigma_k^2 |C|^(2 - alpha)` ([`MackProcess`]),
//!    and the same pseudo factors all year. Mack's model is conditional on
//!    the latest diagonal, so unlike the ODP it projects from the observed
//!    value, not a pseudo one, and is Markov: a later cell of the year
//!    develops from the drawn value of the cell before.
//!
//! The re-reserving that follows is the ODP's
//! ([`crate::one_year_bootstrap`]). The lifetime view draws instead every
//! future cell to the last age the same way, each from the one before with
//! one set of pseudo factors (EVW's steps 7(d) and (e)), and records each
//! origin's last value less its latest; an origin at the last age has no
//! reserve. Its standard deviation is Mack's analytic standard error to
//! first order, with the parameter error scaled by the variance of the
//! resampled residuals (`validation/tests/reserving_mack_bootstrap.rs`).
//! The sigmas are the observed triangle's,
//! those behind a single link ratio interpolated as [`Mack`] does. A link
//! from a zero value has no variance in Mack's model: it keeps its observed
//! later value and gives no residual. Nor does a factor whose sigma is zero
//! (its link ratios all equal): its residuals would be `0 / 0`, and zeros
//! in their place would shrink the pool's variance and every factor's
//! parameter error, so it gives none and its pseudo factor is its factor.
//!
//! The residuals of each factor have a zero `C_k^(alpha / 2)`-weighted
//! sum, not a zero mean, so the pool's mean `m` is not zero (RAA 0.14,
//! GenIns 0.01, ABC -0.06) and `E[f*_k] = f_k + m sigma_k
//! sum(C_k^(alpha / 2)) / sum(C_k^alpha)`: the pseudo factors are biased,
//! and so are the CDR, whose expectation under Mack's model is zero, and
//! the lifetime view's mean reserve, compounded over each origin's
//! remaining factors (in total about +17% of the chain ladder's on RAA,
//! +0.7% on GenIns, -0.8% on ABC). EVW's Appendix 1 resamples the residuals
//! as they are, which is the default;
//! [`MackBootstrap::centre_residuals`] subtracts `m` from the pool first,
//! and then the mean reserve is the chain ladder's. EVW's Table 4 expected
//! reserves on Taylor–Ashe agree with the centred bootstrap, not the
//! uncentred one.

use prospicio_core::StreamRng;
use prospicio_math::special::norm_quantile;
use prospicio_prob::{
    Distribution, Gamma, InputHasher, Lognormal, PredictiveDistribution, Provenance,
};

use crate::bootstrap::{
    component_sums, hash_segment, origin_keys, pick_segment, push_moments, segment_sums,
};
use crate::chain_ladder::ChainLadderFit;
use crate::development::{Development, DevelopmentFit};
use crate::error::{Error, Result};
use crate::mack::{Mack, MackFit};
use crate::one_year_bootstrap::{
    NextYear, OneYearFit, OneYearFits, OneYearMethod, Sims, YearCells,
};
use crate::segments::{FitTable, ReserveFit, SegmentFits, fit_each};
use crate::triangle::{Segment, Triangle};

/// Process error on the next cumulative value of Mack's bootstrap, with
/// mean `f*_k C` and variance `sigma_k^2 |C|^(2 - alpha)`.
///
/// England, Verrall and Wüthrich (2019), Appendix 1, step 7(d), draw it
/// either from a parametric distribution, Gamma or lognormal so that the
/// cumulative value stays positive, or by resampling the residuals again.
/// Mack's model itself is distribution-free, so the normal is offered as
/// well. `Gamma`, `Lognormal` and `Normal` have exactly that mean and
/// variance and differ only in shape. `Residuals` has the resampled pool's
/// moments instead: mean `f*_k C + m sd` and variance `(1 - m^2) sd^2`,
/// `sd` the standard deviation above and `m` the pool's mean (its mean
/// square is 1), so with uncentred residuals it adds a bias of its own;
/// with [`MackBootstrap::centre_residuals`] its mean is `f*_k C`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MackProcess {
    /// Gamma with that mean and variance (EVW's parametric example). A
    /// negative mean (a pseudo factor below zero) gets the Gamma of its
    /// absolute value, negated, as the ODP's process does.
    #[default]
    Gamma,
    /// Lognormal with that mean and variance, negated for a negative mean
    /// as `Gamma` is.
    Lognormal,
    /// The mean plus a resampled residual times the standard deviation,
    /// EVW's non-parametric choice; it carries the pool's mean and variance
    /// (see above).
    Residuals,
    /// Normal with that mean and variance; it can go below zero.
    Normal,
    /// No process error: the mean, i.e. parameter error only.
    None,
}

impl MackProcess {
    /// A draw with the given mean and variance; `pool` holds the residuals
    /// to resample.
    fn draw(self, mean: f64, variance: f64, pool: &[f64], rng: &mut StreamRng) -> f64 {
        if variance == 0.0 {
            return mean;
        }
        match self {
            Self::None => mean,
            Self::Normal => mean + variance.sqrt() * norm_quantile(rng.next_open01()),
            Self::Residuals => mean + variance.sqrt() * resample(pool, rng),
            Self::Gamma | Self::Lognormal if mean == 0.0 => mean,
            Self::Gamma => {
                let m = mean.abs();
                // `m * m` underflows for a value near zero, which a later
                // cell of the year can be drawn from: see `vanishing`.
                match Gamma::new(m * m / variance, variance / m) {
                    Ok(gamma) => mean.signum() * gamma.sample(rng, 1)[0],
                    Err(_) => vanishing(mean, variance),
                }
            }
            Self::Lognormal => {
                let m = mean.abs();
                match Lognormal::from_mean_cv(m, variance.sqrt() / m) {
                    Ok(lognormal) => mean.signum() * lognormal.sample(rng, 1)[0],
                    Err(_) => vanishing(mean, variance),
                }
            }
        }
    }
}

/// The draw of a Gamma or lognormal whose parameters are out of floating
/// point range. With a finite mean and variance that happens only when the
/// mean is negligible next to the standard deviation (a cumulative value
/// drawn near zero, from which the next is drawn): the shape goes to zero,
/// or the coefficient of variation to infinity, and the distribution's
/// mass to zero, which is the draw. Anything else is NaN, so that the
/// simulation fails and is counted.
fn vanishing(mean: f64, variance: f64) -> f64 {
    if mean.is_finite() && variance.is_finite() {
        0.0
    } else {
        f64::NAN
    }
}

/// Mack's model bootstrapped, for the lifetime view ([`fit`](Self::fit),
/// [`fit_segments`](Self::fit_segments)) and the one-year view
/// ([`one_year`](Self::one_year), [`one_year_segments`](Self::one_year_segments));
/// see the [module documentation](crate::mack_bootstrap).
///
/// Next to [`OdpBootstrap`](crate::OdpBootstrap), whose process is the
/// over-dispersed Poisson's (variance `phi` times the mean increment), this
/// is Mack's (variance `sigma_k^2` times the cumulative value). With the
/// volume-weighted chain ladder and no tail, the standard deviation of its
/// one-year view reproduces Merz and Wüthrich's
/// ([`MackFit::claims_development_result`](crate::MackFit::claims_development_result))
/// within Monte Carlo error, and any other method, weighting or tail is
/// re-reserved as the ODP's is.
///
/// Its mean is not Merz and Wüthrich's zero unless
/// [`centre_residuals`](Self::centre_residuals) is set: with EVW's
/// uncentred residuals the mean CDR is about -0.21 (RAA), -0.04 (GenIns)
/// and +0.18 (ABC) times its standard deviation, which shifts every
/// quantile; centred, it is within Monte Carlo error of zero and the
/// standard deviation still reconciles
/// (`knowledge/findings/one-year-bootstrap-vs-merz-wuthrich.md`).
///
/// ```
/// use prospicio_reserving::{
///     ChainLadder, DevelopmentColumn, Grain, Long, MackBootstrap, Month, OneYearMethod, Triangle,
/// };
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
/// let boot = MackBootstrap { n_sims: 2_000, seed: 42, ..Default::default() };
/// let fit = boot.one_year(&tri, "paid", &OneYearMethod::ChainLadder(ChainLadder::default()))?;
/// assert_eq!(fit.cdr.dims(), ["origin"]);
/// // 2020 is fully developed: no new cell, no change.
/// assert!(fit.cdr.draw_matrix().chunks(4).all(|row| row[0] == 0.0));
/// // The one-year view is narrower than Mack's lifetime view.
/// assert!(fit.cdr.std_dev() < fit.bootstrap.mack.total_standard_error);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MackBootstrap {
    /// Number of simulations.
    pub n_sims: usize,
    /// Seed of the simulation streams; simulation `i` uses stream `i`.
    pub seed: u64,
    /// Process error on the next cumulative values.
    pub process: MackProcess,
    /// Mack's factors (their `alpha`) and how unestimable sigmas are filled
    /// in. Mack's tail is a step from the oldest age to ultimate that no
    /// coming year holds, so the model has none: in the one-year view
    /// development past the oldest age moves only through the refitted
    /// method's tail, as with the ODP. Nor does the lifetime view simulate
    /// one: a tail factor is not an average of link ratios, so no residual
    /// gives its parameter error, and its reserves run to the oldest age.
    pub development: Development,
    /// Subtract the pool's mean from the residuals before resampling them,
    /// for the pseudo factors and the `Residuals` process, so that the
    /// pseudo factors are unbiased and the CDR's mean is about zero. Off by
    /// default, as EVW's Appendix 1; see the [module
    /// documentation](crate::mack_bootstrap).
    pub centre_residuals: bool,
}

impl Default for MackBootstrap {
    fn default() -> Self {
        Self {
            n_sims: 10_000,
            seed: 0,
            process: MackProcess::Gamma,
            development: Development::default(),
            centre_residuals: false,
        }
    }
}

/// What Mack's bootstrap estimates in one segment before simulating.
#[derive(Debug, Clone, PartialEq)]
pub struct MackBootstrapSegment {
    /// Mack's model on the observed triangle, without a tail: the factors
    /// and sigmas the simulation uses, and its lifetime standard errors.
    pub mack: MackFit,
    /// The scaled bias-adjusted residuals of the link ratios, row-major
    /// over origin × development: element `(o, k)` is the link from age `k`
    /// to `k + 1`. NaN where there is no link, its earlier value is zero,
    /// or its factor rests on a single link ratio or has a zero sigma.
    /// Never centred, whatever
    /// [`MackBootstrap::centre_residuals`] says.
    pub residuals: Vec<f64>,
}

impl ReserveFit for MackBootstrapSegment {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.mack.chain_ladder
    }
}

/// A fitted bootstrap of Mack's model, the lifetime view
/// ([`MackBootstrap::fit`]): the model on the observed triangle and the
/// joint distribution of the simulated reserves.
#[derive(Debug, Clone)]
pub struct MackBootstrapFit {
    /// Mack's model on the observed triangle, without a tail: the factors
    /// and sigmas the simulation uses, and the analytic standard errors
    /// the simulated reserves' standard deviations approximate.
    pub mack: MackFit,
    /// The scaled bias-adjusted residuals of the link ratios, as
    /// [`MackBootstrapSegment::residuals`].
    pub residuals: Vec<f64>,
    /// Joint distribution of the reserve (each origin's last simulated
    /// cumulative value less its latest observed value) by origin:
    /// dimension `origin`, one component per origin period.
    pub reserves: PredictiveDistribution,
}

/// A bootstrap of Mack's model, the lifetime view, in every segment of a
/// triangle column ([`MackBootstrap::fit_segments`]).
///
/// Segments are bootstrapped independently, each with its own Mack model
/// and residuals. Simulation `i` uses stream `i` for every segment, in
/// index order, as in [`OdpBootstrapFits`](crate::OdpBootstrapFits).
///
/// ```
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, MackBootstrap, Month, Triangle};
/// use prospicio_prob::Distribution;
///
/// let origin = [2019, 2019, 2019, 2019, 2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
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
/// let boot = MackBootstrap { n_sims: 1_000, seed: 1, ..Default::default() }
///     .fit_segments(&tri, "paid")?;
/// assert_eq!(boot.reserves.dims(), ["lob", "origin"]);
/// assert_eq!(boot.reserves.n_components(), 8);
/// let by_lob = boot.reserves.aggregate(&["lob"])?;
/// assert_eq!(by_lob.n_components(), 2);
/// assert_eq!(boot.totals().column("std_dev").unwrap().len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct MackBootstrapFits {
    /// Each segment's Mack model and residuals.
    pub segments: SegmentFits<MackBootstrapSegment>,
    /// Joint distribution of the reserve by segment and origin: the key
    /// names and `origin` are its dimensions, and its components run over
    /// the origins of each segment in turn, like the rows of
    /// [`to_long`](Self::to_long).
    pub reserves: PredictiveDistribution,
}

impl MackBootstrapFits {
    /// One row per segment × origin: the chain ladder's `latest`,
    /// `ultimate` and `reserve` (Mack's averaging), and the `mean` and
    /// `std_dev` of the bootstrapped reserve.
    pub fn to_long(&self) -> FitTable {
        let mut table = self.segments.to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        table
    }

    /// One row per segment: the chain ladder's totals and the `mean` and
    /// `std_dev` of the segment's bootstrapped total reserve.
    pub fn totals(&self) -> FitTable {
        let mut table = self.segments.totals();
        let totals = segment_sums(&self.segments, &self.reserves);
        push_moments(&mut table, "", &totals);
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

/// What Mack's bootstrap resamples in each simulation.
pub(crate) struct MackDraw {
    /// Per factor, the `(C_k, C_k+1)` of every origin observed at both ages.
    links: Vec<Vec<(f64, f64)>>,
    pool: Vec<f64>,
    process: MackProcess,
}

/// A residual drawn from `pool` with replacement.
fn resample(pool: &[f64], rng: &mut StreamRng) -> f64 {
    pool[((rng.next_open01() * pool.len() as f64) as usize).min(pool.len() - 1)]
}

/// `x^e`, exactly for the exponents the three averages give.
fn power(x: f64, e: f64) -> f64 {
    if e == 0.0 {
        1.0
    } else if e == 0.5 {
        x.sqrt()
    } else if e == 1.0 {
        x
    } else {
        x.powf(e)
    }
}

impl MackDraw {
    /// The pseudo factors of one simulation (its parameter error): a
    /// resampled residual for every observed link, turned into a pseudo
    /// link ratio, and their weighted averages with the observed weights.
    fn factors(&self, dev: &DevelopmentFit, rng: &mut StreamRng) -> Vec<f64> {
        let (f, sigma, alpha) = (&dev.ldf, &dev.sigma, dev.alpha);
        let mut factors = Vec::with_capacity(self.links.len());
        for (k, pairs) in self.links.iter().enumerate() {
            let (mut num, mut den) = (0.0, 0.0);
            for &(c, c1) in pairs {
                if c == 0.0 {
                    num += power(c, alpha - 1.0) * c1;
                    continue;
                }
                let r = if sigma[k] == 0.0 {
                    0.0
                } else {
                    resample(&self.pool, rng)
                };
                num += power(c, alpha - 1.0) * (c * (f[k] + r * sigma[k] / power(c, alpha / 2.0)));
                den += power(c, alpha);
            }
            factors.push(if den == 0.0 { f[k] } else { num / den });
        }
        factors
    }

    /// The next cumulative value from `c` across factor `d`, with mean
    /// `f*_d c` and variance `sigma_d^2 |c|^(2 - alpha)`.
    fn next(
        &self,
        dev: &DevelopmentFit,
        factors: &[f64],
        d: usize,
        c: f64,
        rng: &mut StreamRng,
    ) -> f64 {
        let variance = dev.sigma[d].powi(2) * power(c.abs(), 2.0 - dev.alpha);
        self.process.draw(factors[d] * c, variance, &self.pool, rng)
    }
}

impl NextYear for MackBootstrapSegment {
    type Draw = MackDraw;

    fn year_cells(
        &self,
        draw: &MackDraw,
        _segment: &Segment,
        year: &[YearCells],
        rng: &mut StreamRng,
    ) -> Vec<(usize, usize, f64)> {
        let cl = &self.mack.chain_ladder;
        let dev = &cl.development;
        let factors = draw.factors(dev, rng);

        // Each cell from the one before, drawn or observed, with the same
        // pseudo factors all year.
        let mut cells = Vec::new();
        for y in year {
            let mut c = cl.latest[y.origin];
            for d in y.latest..y.last {
                c = draw.next(dev, &factors, d, c, rng);
                if d + 1 >= y.first {
                    cells.push((y.origin, d + 1, c));
                }
            }
        }
        cells
    }
}

impl MackBootstrapSegment {
    /// One simulation of the run-off: fills `reserves` with each origin's
    /// last cumulative value, drawn cell by cell from its observed latest
    /// value to the last age with one set of pseudo factors, less that
    /// latest value (EVW's Appendix 1, steps 7(a) to (g)).
    fn run_off(&self, draw: &MackDraw, rng: &mut StreamRng, reserves: &mut [f64]) {
        let cl = &self.mack.chain_ladder;
        let dev = &cl.development;
        let factors = draw.factors(dev, rng);
        for (o, reserve) in reserves.iter_mut().enumerate() {
            let latest = cl.latest[o];
            let mut c = latest;
            for d in cl.latest_position[o]..factors.len() {
                c = draw.next(dev, &factors, d, c, rng);
            }
            *reserve = c - latest;
        }
    }
}

impl MackBootstrap {
    /// The lifetime view of `column` of a single-segment cumulative
    /// triangle: each origin's reserve, simulated to the last age with
    /// Mack's parameter and process error (EVW's Appendix 1, steps 7(a) to
    /// (g)); see the [module documentation](crate::mack_bootstrap). Every
    /// origin must be observed from the first age to its latest, with no
    /// negative value, and Mack's model must fit ([`Mack::fit`]).
    ///
    /// ```
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, MackBootstrap, Month, Triangle};
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
    /// let boot = MackBootstrap { n_sims: 2_000, seed: 42, centre_residuals: true, ..Default::default() };
    /// let fit = boot.fit(&tri, "paid")?;
    /// assert_eq!(fit.reserves.dims(), ["origin"]);
    /// // 2020 is fully developed: no reserve.
    /// assert!(fit.reserves.draw_matrix().chunks(4).all(|row| row[0] == 0.0));
    /// // Centred, the mean is the chain ladder's reserve within Monte
    /// // Carlo error, and the standard deviation approximates Mack's.
    /// let cl = fit.mack.chain_ladder.total_reserve();
    /// assert!((fit.reserves.mean() / cl - 1.0).abs() < 0.05);
    /// assert!((fit.reserves.std_dev() / fit.mack.total_standard_error - 1.0).abs() < 0.2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<MackBootstrapFit> {
        self.check()?;
        let segment = triangle.segment(column)?;
        let (fit, draw) = self.model(&segment)?;
        let mut hasher = InputHasher::new();
        hasher.str(column);
        hash_segment(&mut hasher, &segment, &fit.mack.chain_ladder, &segment.ages);
        let reserves = PredictiveDistribution::simulate(
            vec!["origin".into()],
            segment.origins.iter().map(|&p| vec![p.into()]).collect(),
            self.n_sims,
            self.seed,
            self.lifetime_provenance(column, hasher),
            |rng, row| fit.run_off(&draw, rng, row),
        )?;
        let MackBootstrapSegment { mack, residuals } = fit;
        Ok(MackBootstrapFit {
            mack,
            residuals,
            reserves,
        })
    }

    /// The lifetime view of `column` in every segment of a cumulative
    /// triangle, each with its own Mack model and residuals, into one joint
    /// distribution of the reserves; see [`MackBootstrapFits`]. A failure
    /// names its segment.
    pub fn fit_segments(&self, triangle: &Triangle, column: &str) -> Result<MackBootstrapFits> {
        self.check()?;
        let prepared = fit_each(triangle, column, |s| {
            let (fit, draw) = self.model(s)?;
            Ok((fit, draw, s.clone()))
        })?;

        let mut hasher = InputHasher::new();
        hasher.str(column);
        let mut dims = prepared.key_names.clone();
        dims.push("origin".into());
        let mut components = Vec::new();
        for (label, (fit, _, segment)) in prepared.iter() {
            hasher.str(&label.to_string());
            hash_segment(&mut hasher, segment, &fit.mack.chain_ladder, &segment.ages);
            components.extend(origin_keys(label, &segment.origins));
        }
        let runs: Vec<_> = prepared.fits.iter().collect();
        let reserves = PredictiveDistribution::simulate(
            dims,
            components,
            self.n_sims,
            self.seed,
            self.lifetime_provenance(column, hasher)
                .param("segments", prepared.len()),
            |rng, row| {
                let mut start = 0;
                for (fit, draw, segment) in &runs {
                    let end = start + segment.n_origins;
                    fit.run_off(draw, rng, &mut row[start..end]);
                    start = end;
                }
            },
        )?;
        Ok(MackBootstrapFits {
            segments: prepared.map(|(fit, _, _)| fit.clone()),
            reserves,
        })
    }

    fn check(&self) -> Result<()> {
        if self.n_sims == 0 {
            return Err(Error::Bootstrap("n_sims must be positive"));
        }
        Ok(())
    }

    fn lifetime_provenance(&self, column: &str, hasher: InputHasher) -> Provenance {
        Provenance::new("mack_bootstrap")
            .param("n_sims", self.n_sims)
            .param("process", format!("{:?}", self.process))
            .param("development", format!("{:?}", self.development))
            .param("centre_residuals", self.centre_residuals)
            .param("column", column)
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(hasher.finish())
    }

    /// The one-year view of `column` of a single-segment cumulative
    /// triangle: the claims development result of `method` over the twelve
    /// months after the valuation, at any development grain, by
    /// re-reserving on Mack's bootstrap; see the [module
    /// documentation](crate::mack_bootstrap). Every origin must be observed
    /// from the first age to its latest (it may stop short of the latest
    /// diagonal), with no negative value, and Mack's model must fit
    /// ([`Mack::fit`]).
    pub fn one_year(
        &self,
        triangle: &Triangle,
        column: &str,
        method: &OneYearMethod,
    ) -> Result<OneYearFit<MackBootstrapSegment>> {
        self.sims().one_year(
            triangle,
            column,
            method,
            |s| self.model(s),
            |hasher| self.provenance(column, method, hasher),
        )
    }

    /// The one-year view of `column` in every segment of a cumulative
    /// triangle, each with its own Mack model and residuals, into one joint
    /// distribution of the claims development result, as
    /// [`OdpBootstrap::one_year_segments`](crate::OdpBootstrap::one_year_segments).
    pub fn one_year_segments(
        &self,
        triangle: &Triangle,
        column: &str,
        method: &OneYearMethod,
    ) -> Result<OneYearFits<MackBootstrapSegment>> {
        self.sims().one_year_segments(
            triangle,
            column,
            method,
            |s| self.model(s),
            |hasher| self.provenance(column, method, hasher),
        )
    }

    fn sims(&self) -> Sims {
        Sims {
            n_sims: self.n_sims,
            seed: self.seed,
        }
    }

    /// Mack's model of one segment, its residuals and links.
    pub(crate) fn model(&self, segment: &Segment) -> Result<(MackBootstrapSegment, MackDraw)> {
        for o in 0..segment.n_origins {
            let (last, _) = segment.latest(o)?;
            for d in 0..=last {
                match segment.get(o, d) {
                    None => {
                        return Err(Error::Bootstrap(
                            "every origin must be observed from the first age to its latest",
                        ));
                    }
                    Some(v) if v < 0.0 => {
                        return Err(Error::Bootstrap(
                            "Mack's bootstrap needs non-negative cumulative values",
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
        let mack = Mack {
            development: self.development,
            ..Default::default()
        }
        .fit_segment(segment, &segment.ages)?;
        let cl = &mack.chain_ladder;

        let (no, nd) = (segment.n_origins, segment.n_dev);
        let dev = &cl.development;
        let (f, sigma, alpha) = (&dev.ldf, &dev.sigma, dev.alpha);
        let mut residuals = vec![f64::NAN; no * nd];
        let mut pool = Vec::new();
        let mut links = Vec::with_capacity(nd - 1);
        for k in 0..nd - 1 {
            let pairs: Vec<(usize, f64, f64)> = (0..no)
                .filter_map(|o| Some((o, segment.get(o, k)?, segment.get(o, k + 1)?)))
                .collect();
            let informative = pairs.iter().filter(|p| p.1 != 0.0).count();
            if informative > 1 && sigma[k] != 0.0 {
                let n = informative as f64;
                for &(o, c, c1) in pairs.iter().filter(|p| p.1 != 0.0) {
                    let r =
                        (n / (n - 1.0)).sqrt() * power(c, alpha / 2.0) * (c1 / c - f[k]) / sigma[k];
                    residuals[o * nd + k] = r;
                    pool.push(r);
                }
            }
            links.push(pairs.iter().map(|&(_, c, c1)| (c, c1)).collect());
        }
        // With every sigma zero nothing is resampled: no parameter or
        // process error.
        if pool.is_empty() && sigma.iter().any(|&s| s != 0.0) {
            return Err(Error::Bootstrap("no residuals to resample"));
        }
        if self.centre_residuals {
            let m = pool.iter().sum::<f64>() / pool.len() as f64;
            pool.iter_mut().for_each(|r| *r -= m);
        }
        Ok((
            MackBootstrapSegment { mack, residuals },
            MackDraw {
                links,
                pool,
                process: self.process,
            },
        ))
    }

    fn provenance(&self, column: &str, method: &OneYearMethod, hasher: InputHasher) -> Provenance {
        Provenance::new("mack_bootstrap_one_year")
            .param("n_sims", self.n_sims)
            .param("process", format!("{:?}", self.process))
            .param("development", format!("{:?}", self.development))
            .param("centre_residuals", self.centre_residuals)
            .param("column", column)
            .param("method", format!("{method:?}"))
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(hasher.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain_ladder::ChainLadder;
    use crate::development::Average;
    use crate::expected_loss::BornhuetterFerguson;
    use crate::tail::Tail;
    use crate::triangle::tests::{RAA, annual, genins, raa};
    use crate::triangle::{DevelopmentColumn, Long};
    use crate::{Grain, Month, OdpBootstrap};
    use prospicio_prob::PredictiveDistribution;

    fn boot(n_sims: usize, seed: u64, process: MackProcess) -> MackBootstrap {
        MackBootstrap {
            n_sims,
            seed,
            process,
            development: Development::default(),
            centre_residuals: false,
        }
    }

    fn chain_ladder() -> OneYearMethod {
        OneYearMethod::ChainLadder(ChainLadder::default())
    }

    /// Column `j` of a distribution's draws.
    fn column(cdr: &PredictiveDistribution, j: usize) -> Vec<f64> {
        let n = cdr.n_components();
        cdr.draw_matrix().chunks(n).map(|row| row[j]).collect()
    }

    /// Standard deviation of `x` and its Monte Carlo standard error,
    /// `sd * sqrt((kurtosis - 1) / (4 n))`.
    fn sd_and_error(x: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        let m = x.iter().sum::<f64>() / n;
        let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
        let m4 = x.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n;
        (
            m2.sqrt(),
            m2.sqrt() * ((m4 / (m2 * m2) - 1.0) / (4.0 * n)).sqrt(),
        )
    }

    fn average(average: Average) -> Development {
        Development {
            average,
            ..Default::default()
        }
    }

    /// RAA as `paid`, with a `premium` of 20,000 per origin.
    fn raa_premium() -> Triangle {
        let (mut origin, mut ages, mut paid) = (vec![], vec![], vec![]);
        for (k, row) in RAA.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                origin.push(Month::january(1981 + k as i32));
                ages.push(12 * (d as u32 + 1));
                paid.push(v);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("premium", &[20_000.0; 55])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn residuals_are_standardized_per_factor() {
        // sigma_k^2 = sum(C^alpha (F - f)^2) / (n_k - 1), so the
        // n_k / (n_k - 1) adjustment makes each factor's squared residuals
        // sum to n_k, for every weighting. The last factor rests on one link
        // ratio: none.
        for avg in [Average::Volume, Average::Simple, Average::Regression] {
            let b = MackBootstrap {
                development: average(avg),
                ..boot(10, 0, MackProcess::Gamma)
            };
            let fit = b.one_year(&raa(), "values", &chain_ladder()).unwrap();
            let r = &fit.bootstrap.residuals;
            assert_eq!(
                fit.bootstrap.mack.chain_ladder.development.alpha,
                avg.alpha()
            );
            for k in 0..9 {
                let links: Vec<f64> = (0..9 - k).map(|o| r[o * 10 + k]).collect();
                if k == 8 {
                    assert!(links.iter().all(|x| x.is_nan()), "{avg:?}");
                    continue;
                }
                let n = links.len() as f64;
                let ss: f64 = links.iter().map(|x| x * x).sum();
                assert!((ss - n).abs() < 1e-9, "{avg:?} factor {k}: {ss} vs {n}");
            }
            // No link from an origin's latest value.
            assert!((0..10).all(|o| r[o * 10 + 9 - o].is_nan()));
        }
    }

    #[test]
    fn one_cell_left_follows_mack_for_every_weighting() {
        // RAA 1982 has one cell left, from the last factor, which rests on
        // 1981's link alone: f* = f + r sigma / C81^(alpha / 2). So its CDR,
        // U0 - C*, has variance sigma^2 (C82^2 v / C81^alpha + C82^(2 - alpha))
        // with v the variance of the resampled residuals: Mack's own
        // parameter and process variance for 1982 when v = 1.
        let n_sims = 5_000;
        for avg in [Average::Volume, Average::Simple, Average::Regression] {
            let alpha = avg.alpha();
            let fit = MackBootstrap {
                development: average(avg),
                ..boot(n_sims, 5, MackProcess::Gamma)
            }
            .one_year(&raa(), "values", &chain_ladder())
            .unwrap();
            let pool: Vec<f64> = fit
                .bootstrap
                .residuals
                .iter()
                .copied()
                .filter(|r| !r.is_nan())
                .collect();
            let m = pool.iter().sum::<f64>() / pool.len() as f64;
            let v = pool.iter().map(|r| (r - m).powi(2)).sum::<f64>() / pool.len() as f64;
            let sigma = fit.bootstrap.mack.chain_ladder.development.sigma[8];
            let (c81, c82) = (RAA[0][8], RAA[1][8]);
            let want =
                (sigma.powi(2) * (c82 * c82 * v / c81.powf(alpha) + c82.powf(2.0 - alpha))).sqrt();
            let (sd, error) = sd_and_error(&column(&fit.cdr, 1));
            assert!(
                (sd - want).abs() < 5.0 * error,
                "{avg:?}: {sd} vs {want} (5 SE {})",
                5.0 * error
            );
            // And so Mack's analytic standard error, up to v.
            let mack = Mack {
                development: average(avg),
                ..Default::default()
            }
            .fit(&raa(), "values")
            .unwrap();
            assert!((v - 1.0).abs() < 0.05, "{avg:?}: v = {v}");
            assert!(
                (want / mack.standard_error[1] - 1.0).abs() < 0.05,
                "{avg:?}: {want} vs Mack {}",
                mack.standard_error[1]
            );
        }
    }

    #[test]
    fn centred_residuals_remove_the_mean_bias() {
        // RAA's pool of residuals has mean m = 0.14 (its mean square is 1),
        // so EVW's uncentred resampling biases every pseudo factor upwards
        // and the total CDR's mean is about -0.2 of its standard deviation,
        // far beyond the mean's Monte Carlo standard error sd / sqrt(n).
        // Centred, the Gamma's and the residuals process's means are both
        // within that error of Merz and Wuthrich's zero.
        let n_sims = 4_000;
        let mean_and_error = |centre_residuals, process| {
            let fit = MackBootstrap {
                centre_residuals,
                ..boot(n_sims, 11, process)
            }
            .one_year(&raa(), "values", &chain_ladder())
            .unwrap();
            let pool: Vec<f64> = fit
                .bootstrap
                .residuals
                .iter()
                .copied()
                .filter(|r| !r.is_nan())
                .collect();
            let n = pool.len() as f64;
            let m = pool.iter().sum::<f64>() / n;
            assert!((m - 0.14).abs() < 0.005, "pool mean {m}");
            assert!((pool.iter().map(|r| r * r).sum::<f64>() / n - 1.0).abs() < 1e-9);
            let totals: Vec<f64> = fit
                .cdr
                .draw_matrix()
                .chunks(10)
                .map(|r| r.iter().sum())
                .collect();
            let (sd, _) = sd_and_error(&totals);
            let mean = totals.iter().sum::<f64>() / n_sims as f64;
            (mean, sd / (n_sims as f64).sqrt())
        };
        let (mean, error) = mean_and_error(false, MackProcess::Gamma);
        assert!(mean < -8.0 * error, "uncentred: {mean} ({error})");
        for process in [MackProcess::Gamma, MackProcess::Residuals] {
            let (mean, error) = mean_and_error(true, process);
            assert!(
                mean.abs() < 4.0 * error,
                "{process:?} centred: {mean} ({error})"
            );
        }
    }

    #[test]
    fn every_process_has_mack_variance() {
        // The process shapes differ, not their variance: on GenIns every
        // one gives Merz-Wuthrich's total within Monte Carlo error, and
        // parameter error alone is narrower.
        let mw = Mack::default()
            .fit(&genins(), "values")
            .unwrap()
            .claims_development_result()
            .unwrap()
            .total_one_year_standard_error;
        let total = |process| {
            let fit = boot(5_000, 9, process)
                .one_year(&genins(), "values", &chain_ladder())
                .unwrap();
            let totals: Vec<f64> = fit
                .cdr
                .draw_matrix()
                .chunks(10)
                .map(|r| r.iter().sum())
                .collect();
            sd_and_error(&totals)
        };
        for process in [
            MackProcess::Gamma,
            MackProcess::Lognormal,
            MackProcess::Residuals,
            MackProcess::Normal,
        ] {
            let (sd, error) = total(process);
            assert!((sd - mw).abs() < 5.0 * error, "{process:?}: {sd} vs {mw}");
        }
        assert!(total(MackProcess::None).0 < 0.8 * mw);
    }

    #[test]
    fn positive_processes_stay_positive() {
        // RAA's 1990 starts at 2,063 with sigma_0 = 167: its next value is
        // often negative under the normal. Under the Gamma or lognormal it
        // is negative only when its mean is, a pseudo first factor below
        // zero (its standard deviation is a third of the factor). Both draw
        // one uniform per value, so they share the pseudo factors and go
        // negative in the same simulations. The closing ultimate is that
        // value times the refitted factors to ultimate, all positive. (The
        // Gamma's shape is below 1 here, so a draw can be exactly zero.)
        let negative = |process| {
            let fit = boot(2_000, 3, process)
                .one_year(&raa(), "values", &chain_ladder())
                .unwrap();
            let u0 = fit.opening_ultimate[9];
            column(&fit.cdr, 9)
                .iter()
                .filter(|&&x| u0 - x < 0.0)
                .count()
        };
        let gamma = negative(MackProcess::Gamma);
        assert_eq!(gamma, negative(MackProcess::Lognormal));
        assert!(negative(MackProcess::Normal) > 10 * gamma.max(1));
    }

    #[test]
    fn reproducible_by_seed_and_thread_count() {
        let run = |seed| {
            boot(300, seed, MackProcess::Gamma)
                .one_year(&raa(), "values", &chain_ladder())
                .unwrap()
        };
        let a = run(7);
        assert_eq!(a.cdr.draw_matrix(), run(7).cdr.draw_matrix());
        assert_ne!(a.cdr.draw_matrix(), run(8).cdr.draw_matrix());
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| run(7));
        assert_eq!(a.cdr.draw_matrix(), one_thread.cdr.draw_matrix());
        assert_eq!(a.cdr.provenance().model, "mack_bootstrap_one_year");
        // The ODP on the same seed opens on the same ultimate but draws
        // differently.
        let odp = OdpBootstrap {
            n_sims: 300,
            seed: 7,
            ..Default::default()
        }
        .one_year(&raa(), "values", &chain_ladder())
        .unwrap();
        assert_ne!(a.cdr.draw_matrix(), odp.cdr.draw_matrix());
        assert_eq!(a.opening_ultimate, odp.opening_ultimate);
    }

    #[test]
    fn every_method_rereserves() {
        // The oldest origin gets no new cell: with no tail it does not
        // move; a log-linear tail refitted on the new factors moves it.
        let tri = raa_premium();
        let b = boot(300, 2, MackProcess::Gamma);
        let plain = b.one_year(&tri, "paid", &chain_ladder()).unwrap();
        assert!(column(&plain.cdr, 0).iter().all(|&x| x == 0.0));
        let tailed = OneYearMethod::ChainLadder(ChainLadder {
            tail: Tail::LogLinear,
            ..Default::default()
        });
        let fit = b.one_year(&tri, "paid", &tailed).unwrap();
        assert!(column(&fit.cdr, 0).iter().any(|&x| x != 0.0));
        let bf = BornhuetterFerguson {
            apriori: 0.8,
            ..Default::default()
        };
        let method = OneYearMethod::BornhuetterFerguson(bf, "premium".into());
        let fit = b.one_year(&tri, "paid", &method).unwrap();
        assert_eq!(
            fit.opening_ultimate,
            bf.fit(&tri, "paid", "premium").unwrap().ultimate
        );
        assert!(column(&fit.cdr, 9).iter().any(|&x| x != 0.0));
    }

    #[test]
    fn segments_share_one_joint_distribution() {
        let origin =
            [2019, 2019, 2019, 2019, 2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
        let ages = [12, 24, 36, 48, 12, 24, 36, 12, 24, 12];
        let paid = [
            100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0,
        ];
        let tri = Triangle::from_long(&Long {
            keys: &[("lob", &[["Auto"; 10], ["Home"; 10]].concat())],
            origin: &[origin, origin].concat(),
            development: DevelopmentColumn::Age(&[ages, ages].concat()),
            values: &[("paid", &[paid, paid.map(|v| v * 3.0)].concat())],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let fits = boot(400, 5, MackProcess::Gamma)
            .one_year_segments(&tri, "paid", &chain_ladder())
            .unwrap();
        assert_eq!(fits.cdr.dims(), ["lob", "origin"]);
        assert_eq!(fits.cdr.n_components(), 8);
        // Mack's bootstrap has no scale column.
        let totals = fits.totals();
        let names: Vec<&str> = totals.values.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "latest",
                "opening_ultimate",
                "opening_reserve",
                "cdr_mean",
                "cdr_std_dev"
            ]
        );
        let home = fits.segment(&[("lob", "Home")]).unwrap();
        assert_eq!(home.cdr.n_components(), 4);
        assert_eq!(home.cdr.dims(), ["lob", "origin"]);
    }

    #[test]
    fn exact_pattern_runs_off_without_variance() {
        // On an exact pattern every sigma is zero: no parameter or process
        // error, so every simulation's reserve is the chain ladder's. (A
        // fifth origin gives the last factor two link ratios: a lone one's
        // sigma cannot be interpolated from sigmas that are all zero.)
        let tri = annual(
            2019,
            &[
                &[4.0, 8.0, 12.0, 15.0],
                &[8.0, 16.0, 24.0, 30.0],
                &[12.0, 24.0, 36.0],
                &[16.0, 32.0],
                &[20.0],
            ],
        );
        for process in [MackProcess::Gamma, MackProcess::Residuals] {
            let fit = boot(50, 3, process).fit(&tri, "values").unwrap();
            let cl = &fit.mack.chain_ladder;
            assert!(cl.development.sigma.iter().all(|&s| s == 0.0));
            let want: Vec<f64> = cl
                .ultimate
                .iter()
                .zip(&cl.latest)
                .map(|(u, l)| u - l)
                .collect();
            assert_eq!(want, [0.0, 0.0, 9.0, 28.0, 55.0]);
            assert!(fit.reserves.draw_matrix().chunks(5).all(|row| row == want));
        }
    }

    #[test]
    fn one_cell_left_draws_as_the_one_year_view() {
        // RAA 1982 has one cell left, and the oldest origin none, so 1982's
        // cell is the first process draw after the pseudo factors in both
        // views, from the same stream: its lifetime reserve plus its
        // one-year CDR is its opening reserve in every simulation.
        let b = boot(500, 13, MackProcess::Gamma);
        let lifetime = b.fit(&raa(), "values").unwrap();
        let one_year = b.one_year(&raa(), "values", &chain_ladder()).unwrap();
        let opening = one_year.opening_reserve[1];
        let reserve = column(&lifetime.reserves, 1);
        let cdr = column(&one_year.cdr, 1);
        assert!(reserve.iter().any(|&r| r != reserve[0]));
        for (r, c) in reserve.iter().zip(&cdr) {
            assert!(
                (r + c - opening).abs() < 1e-9 * opening,
                "{r} + {c} vs {opening}"
            );
        }
        // The fully developed origin has no reserve.
        assert!(column(&lifetime.reserves, 0).iter().all(|&r| r == 0.0));
    }

    #[test]
    fn lifetime_mean_is_the_chain_ladder_only_when_centred() {
        // RAA's pool has mean 0.14, which biases every pseudo factor
        // upwards, and compounded over an origin's remaining factors the
        // mean total reserve far above the chain ladder's (about 17%: 1.154
        // to 1.180 times it over seeds 1 to 20 at 4,000 simulations, or 17.8
        // to 20.3 Monte Carlo standard errors, so the bound of 12 does not
        // hang on the seed). Centred, the pseudo factors are unbiased and independent
        // of each other and of the cell they multiply, so the mean reserve
        // is the chain ladder's: within four Monte Carlo standard errors.
        let n_sims = 4_000;
        let z = |centre_residuals| {
            let fit = MackBootstrap {
                centre_residuals,
                ..boot(n_sims, 17, MackProcess::Gamma)
            }
            .fit(&raa(), "values")
            .unwrap();
            let cl = fit.mack.chain_ladder.total_reserve();
            let totals: Vec<f64> = fit
                .reserves
                .draw_matrix()
                .chunks(10)
                .map(|r| r.iter().sum())
                .collect();
            let (sd, _) = sd_and_error(&totals);
            let mean = totals.iter().sum::<f64>() / n_sims as f64;
            (mean - cl) / (sd / (n_sims as f64).sqrt())
        };
        let uncentred = z(false);
        assert!(uncentred > 12.0, "uncentred: {uncentred}");
        let centred = z(true);
        assert!(centred.abs() < 4.0, "centred: {centred}");
    }

    #[test]
    fn lifetime_process_error_widens_and_is_wider_than_one_year() {
        // On GenIns Mack's parameter error is 0.64 of the total standard
        // error and Merz and Wuthrich's one-year view 0.73.
        let tri = genins();
        let total_sd = |process| {
            let fit = boot(2_000, 4, process).fit(&tri, "values").unwrap();
            fit.reserves.std_dev()
        };
        let full = total_sd(MackProcess::Gamma);
        assert!(total_sd(MackProcess::None) < 0.8 * full);
        let one_year = boot(2_000, 4, MackProcess::Gamma)
            .one_year(&tri, "values", &chain_ladder())
            .unwrap();
        assert!(one_year.cdr.std_dev() < 0.85 * full);
    }

    #[test]
    fn lifetime_reproducible_by_seed_and_thread_count() {
        let run = |seed| {
            boot(300, seed, MackProcess::Gamma)
                .fit(&raa(), "values")
                .unwrap()
        };
        let a = run(7);
        assert_eq!(a.reserves.draw_matrix(), run(7).reserves.draw_matrix());
        assert_ne!(a.reserves.draw_matrix(), run(8).reserves.draw_matrix());
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| run(7));
        assert_eq!(a.reserves.draw_matrix(), one_thread.reserves.draw_matrix());
        assert_eq!(a.reserves.provenance().model, "mack_bootstrap");
        assert_eq!(a.reserves.dims(), ["origin"]);
        // A single segment through `fit_segments` draws the same values.
        let fits = boot(300, 7, MackProcess::Gamma)
            .fit_segments(&raa(), "values")
            .unwrap();
        assert_eq!(a.reserves.draw_matrix(), fits.reserves.draw_matrix());
        let bits = |r: &[f64]| r.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.residuals), bits(&fits.segments.fits[0].residuals));
    }

    #[test]
    fn lifetime_segments_share_one_joint_distribution() {
        let origin =
            [2019, 2019, 2019, 2019, 2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
        let ages = [12, 24, 36, 48, 12, 24, 36, 12, 24, 12];
        let paid = [
            100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0,
        ];
        let tri = Triangle::from_long(&Long {
            keys: &[("lob", &[["Auto"; 10], ["Home"; 10]].concat())],
            origin: &[origin, origin].concat(),
            development: DevelopmentColumn::Age(&[ages, ages].concat()),
            values: &[("paid", &[paid, paid.map(|v| v * 3.0)].concat())],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let fits = boot(400, 5, MackProcess::Gamma)
            .fit_segments(&tri, "paid")
            .unwrap();
        assert_eq!(fits.reserves.dims(), ["lob", "origin"]);
        assert_eq!(fits.reserves.n_components(), 8);
        let long = fits.to_long();
        assert_eq!(long.n_rows(), 8);
        let names: Vec<&str> = long.values.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["latest", "ultimate", "reserve", "mean", "std_dev"]);
        let totals = fits.totals();
        assert_eq!(totals.column("std_dev").unwrap().len(), 2);
        let home = fits.segment(&[("lob", "Home")]).unwrap();
        assert_eq!(home.reserves.n_components(), 4);
        assert_eq!(home.reserves.dims(), ["lob", "origin"]);
        let home_draws = column(&fits.reserves, 7);
        assert_eq!(column(&home.reserves, 3), home_draws);
        assert!(fits.segment(&[("lob", "Boat")]).is_err());
    }

    #[test]
    fn lifetime_errors() {
        let negative = annual(
            2020,
            &[
                &[4.0, 8.0, 12.0, 15.0],
                &[-8.0, 16.0, 24.0],
                &[12.0, 25.0],
                &[16.0],
            ],
        );
        let b = boot(10, 0, MackProcess::Gamma);
        assert_eq!(
            b.fit(&negative, "values").unwrap_err(),
            Error::Bootstrap("Mack's bootstrap needs non-negative cumulative values")
        );
        assert_eq!(
            boot(0, 0, MackProcess::Gamma)
                .fit_segments(&raa(), "values")
                .unwrap_err(),
            Error::Bootstrap("n_sims must be positive")
        );
    }

    #[test]
    fn errors() {
        let b = boot(10, 0, MackProcess::Gamma);
        let negative = annual(
            2020,
            &[
                &[4.0, 8.0, 12.0, 15.0],
                &[-8.0, 16.0, 24.0],
                &[12.0, 25.0],
                &[16.0],
            ],
        );
        assert_eq!(
            b.one_year(&negative, "values", &chain_ladder())
                .unwrap_err(),
            Error::Bootstrap("Mack's bootstrap needs non-negative cumulative values")
        );
        // 2021 has no value at 24 months.
        let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2022, 2022, 2023].map(Month::january);
        let holes = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 36, 12, 24, 12]),
            values: &[(
                "values",
                &[4.0, 8.0, 12.0, 15.0, 7.0, 25.0, 12.0, 25.0, 16.0],
            )],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        assert_eq!(
            b.one_year(&holes, "values", &chain_ladder()).unwrap_err(),
            Error::Bootstrap("every origin must be observed from the first age to its latest")
        );
        let tiny = annual(2020, &[&[1.0, 2.0], &[1.0]]);
        assert!(matches!(
            b.one_year(&tiny, "values", &chain_ladder()),
            Err(Error::TooFewAges { .. })
        ));
        assert_eq!(
            boot(0, 0, MackProcess::Gamma)
                .one_year(&raa(), "values", &chain_ladder())
                .unwrap_err(),
            Error::Bootstrap("n_sims must be positive")
        );
    }
}
