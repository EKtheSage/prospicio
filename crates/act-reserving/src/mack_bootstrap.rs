//! Mack's model bootstrapped, after England, Verrall and Wüthrich (2019),
//! Appendix 1, for the simulated one-year view
//! ([`MackBootstrap::one_year`]; `docs/design/reserving-v02.md`,
//! decision 8).
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
//! 2. draws the next cumulative value of every origin short of the last age
//!    from its observed latest value `C`, with mean `f*_k C` and variance
//!    `sigma_k^2 C^(2 - alpha)` ([`MackProcess`]). Mack's model is
//!    conditional on the latest diagonal, so unlike the ODP it projects from
//!    the observed value, not a pseudo one.
//!
//! The re-reserving that follows is the ODP's
//! ([`crate::one_year_bootstrap`]). The sigmas are the observed triangle's,
//! those behind a single link ratio interpolated as [`Mack`] does. A link
//! from a zero value has no variance in Mack's model: it keeps its observed
//! later value and gives no residual.

use act_core::StreamRng;
use act_math::special::norm_quantile;
use act_prob::{Distribution, Gamma, InputHasher, Lognormal, Provenance};

use crate::chain_ladder::ChainLadderFit;
use crate::development::Development;
use crate::error::{Error, Result};
use crate::mack::{Mack, MackFit};
use crate::one_year_bootstrap::{NextDiagonal, OneYearFit, OneYearFits, OneYearMethod, Sims};
use crate::segments::ReserveFit;
use crate::triangle::{Segment, Triangle};

/// Process error on the next cumulative value of Mack's bootstrap, with
/// mean `f*_k C` and variance `sigma_k^2 C^(2 - alpha)`.
///
/// England, Verrall and Wüthrich (2019), Appendix 1, step 7(d), draw it
/// either from a parametric distribution, Gamma or lognormal so that the
/// cumulative value stays positive, or by resampling the residuals again.
/// Mack's model itself is distribution-free, so the normal is offered as
/// well. Only the shape differs: every choice but `None` has the same mean
/// and variance.
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
    /// EVW's non-parametric choice.
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
                let gamma = Gamma::new(m * m / variance, variance / m)
                    .expect("shape and scale are finite and positive");
                mean.signum() * gamma.sample(rng, 1)[0]
            }
            Self::Lognormal => {
                let m = mean.abs();
                let lognormal = Lognormal::from_mean_cv(m, variance.sqrt() / m)
                    .expect("mean and coefficient of variation are finite and positive");
                mean.signum() * lognormal.sample(rng, 1)[0]
            }
        }
    }
}

/// Mack's model bootstrapped for the one-year view; see the
/// [module documentation](crate::mack_bootstrap).
///
/// Next to [`OdpBootstrap`](crate::OdpBootstrap), whose process is the
/// over-dispersed Poisson's (variance `phi` times the mean increment), this
/// is Mack's (variance `sigma_k^2` times the cumulative value). With the
/// volume-weighted chain ladder and no tail, its one-year view reproduces
/// Merz and Wüthrich's
/// ([`MackFit::claims_development_result`](crate::MackFit::claims_development_result))
/// within Monte Carlo error, and any other method, weighting or tail is
/// re-reserved as the ODP's is.
///
/// ```
/// use act_reserving::{
///     ChainLadder, DevelopmentColumn, Grain, Long, MackBootstrap, Month, OneYearMethod, Triangle,
/// };
/// use act_prob::Distribution;
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
    /// coming year holds, so the model has none: development past the
    /// oldest age moves only through the refitted method's tail, as with
    /// the ODP.
    pub development: Development,
}

impl Default for MackBootstrap {
    fn default() -> Self {
        Self {
            n_sims: 10_000,
            seed: 0,
            process: MackProcess::Gamma,
            development: Development::default(),
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
    /// or its factor rests on a single link ratio.
    pub residuals: Vec<f64>,
}

impl ReserveFit for MackBootstrapSegment {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.mack.chain_ladder
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

impl NextDiagonal for MackBootstrapSegment {
    type Draw = MackDraw;

    fn next_cells(
        &self,
        draw: &MackDraw,
        segment: &Segment,
        rng: &mut StreamRng,
    ) -> Vec<(usize, usize, f64)> {
        let cl = &self.mack.chain_ladder;
        let dev = &cl.development;
        let (f, sigma, alpha) = (&dev.ldf, &dev.sigma, dev.alpha);

        // Pseudo link ratios and their weighted averages.
        let mut factors = Vec::with_capacity(draw.links.len());
        for (k, pairs) in draw.links.iter().enumerate() {
            let (mut num, mut den) = (0.0, 0.0);
            for &(c, c1) in pairs {
                if c == 0.0 {
                    num += power(c, alpha - 1.0) * c1;
                    continue;
                }
                let r = resample(&draw.pool, rng);
                num += power(c, alpha - 1.0) * (c * (f[k] + r * sigma[k] / power(c, alpha / 2.0)));
                den += power(c, alpha);
            }
            factors.push(if den == 0.0 { f[k] } else { num / den });
        }

        let nd = segment.n_dev;
        cl.latest_position
            .iter()
            .zip(&cl.latest)
            .enumerate()
            .filter(|&(_, (&d, _))| d + 1 < nd)
            .map(|(o, (&d, &c))| {
                let (mean, variance) = (factors[d] * c, sigma[d].powi(2) * power(c, 2.0 - alpha));
                (o, d + 1, draw.process.draw(mean, variance, &draw.pool, rng))
            })
            .collect()
    }
}

impl MackBootstrap {
    /// The one-year view of `column` of a single-segment cumulative
    /// triangle: the claims development result of `method` over the next
    /// development period, by re-reserving on Mack's bootstrap; see the
    /// [module documentation](crate::mack_bootstrap). Every origin must be
    /// observed from the first age to its latest, with no negative value,
    /// and Mack's model must fit ([`Mack::fit`]).
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
    fn model(&self, segment: &Segment) -> Result<(MackBootstrapSegment, MackDraw)> {
        let mack = Mack {
            development: self.development,
            ..Default::default()
        }
        .fit_segment(segment, &segment.ages)?;
        let cl = &mack.chain_ladder;
        for (o, &last) in cl.latest_position.iter().enumerate() {
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
            if informative > 1 {
                let n = informative as f64;
                for &(o, c, c1) in pairs.iter().filter(|p| p.1 != 0.0) {
                    let r = if sigma[k] == 0.0 {
                        0.0
                    } else {
                        (n / (n - 1.0)).sqrt() * power(c, alpha / 2.0) * (c1 / c - f[k]) / sigma[k]
                    };
                    residuals[o * nd + k] = r;
                    pool.push(r);
                }
            }
            links.push(pairs.iter().map(|&(_, c, c1)| (c, c1)).collect());
        }
        if pool.is_empty() {
            return Err(Error::Bootstrap("no residuals to resample"));
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
            .param("column", column)
            .param("method", format!("{method:?}"))
            .version("act-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(hasher.finish())
    }
}
