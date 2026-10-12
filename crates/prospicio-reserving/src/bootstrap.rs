//! Over-dispersed Poisson (ODP) bootstrap of the chain ladder, after England
//! and Verrall (2002), as implemented by R ChainLadder's `BootChainLadder`.
//!
//! 1. Fit the volume-weighted chain ladder and back out fitted incremental
//!    values from each origin's latest diagonal.
//! 2. Pearson residuals `(X - m) / sqrt(|m|)` on every observed cell, the
//!    scale `phi = sum(r^2) / (n - p)` and residuals adjusted by
//!    `sqrt(n / (n - p))`, with `n` observed cells and `p = origins + ages - 1`
//!    parameters.
//! 3. Per simulation: resample adjusted residuals with replacement into a
//!    pseudo triangle `m + r* sqrt(|m|)`, re-estimate the factors, project
//!    the expected future incrementals and add process error to each.
//!
//! Every simulation draws from its own [`StreamRng`] stream
//! (`PredictiveDistribution::simulate`), so results do not depend on the
//! number of threads.
//!
//! With a [`Tail`] ([`OdpBootstrap::tail`]) each simulation also develops
//! past the oldest age (`docs/design/reserving-v02.md`, decision 9): an
//! estimated tail is refitted on the pseudo factors, a constant one is
//! fixed or, with a standard error, drawn from a lognormal, and the step
//! to ultimate is one more future increment with the ODP's process error.
//! A constant tail attached before the oldest age moves each factor it
//! replaces by the pseudo factor's deviation from the estimate, so those
//! ages keep their parameter error.

use std::sync::Mutex;

use prospicio_core::{Lag, Period, StreamRng};
use prospicio_prob::{
    ComponentKey, Distribution, Gamma, InputHasher, KeyValue, Lognormal, PredictiveDistribution,
    Provenance, Sampled,
};

use crate::chain_ladder::{ChainLadder, ChainLadderFit};
use crate::dependence::{Resample, SegmentDependence, synchronize};
use crate::error::{Error, Result};
use crate::one_year_bootstrap::Failures;
use crate::segments::{FitTable, ReserveFit, SegmentFits, fit_each};
use crate::tail::Tail;
use crate::triangle::{Label, Segment, Triangle};

/// Process error added to each simulated future incremental value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProcessDistribution {
    /// Gamma with the expected value as mean and variance `phi |m|`, signed
    /// like the mean (R's `process.distr = "gamma"`).
    #[default]
    Gamma,
    /// No process error: the expected future values of each resampled
    /// triangle, i.e. parameter error only (R's `ParamDist`).
    None,
}

/// ODP bootstrap of the chain ladder.
///
/// ```
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, OdpBootstrap, Triangle};
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
/// let boot = OdpBootstrap { n_sims: 2_000, seed: 42, ..Default::default() }.fit(&tri, "paid")?;
/// assert_eq!(boot.reserves.n_components(), 4);
/// assert!(boot.reserves.mean() > 0.0);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OdpBootstrap {
    /// Number of simulations.
    pub n_sims: usize,
    /// Seed of the simulation streams; simulation `i` uses stream `i`.
    pub seed: u64,
    /// Process error on the simulated future values.
    pub process: ProcessDistribution,
    /// Development past the oldest age in the lifetime view
    /// ([`fit`](Self::fit), [`fit_segments`](Self::fit_segments)); the
    /// default is none, the reserves running to the oldest age as R's
    /// `BootChainLadder`'s do. An estimated tail ([`Tail::Curve`],
    /// [`Tail::Bondy`], [`Tail::LogLinear`]) is refitted on each
    /// simulation's pseudo factors, which gives its parameter error; a
    /// constant one is fixed unless [`tail_std_err`](Self::tail_std_err)
    /// is given. Either way the step from the oldest age to ultimate is one
    /// more future increment, with mean the pseudo value at the oldest age
    /// times the tail factor less 1 and the process error of every other
    /// increment. The one-year view takes its tail from the refitted method
    /// instead, and a tail here is an error there.
    ///
    /// [`Tail::LogLinear`] refits R's rule, whose guards were written for
    /// one estimate: it is exactly 1 when the product of the third- and
    /// second-last factors is at most 1.0001, and its line leaves out every
    /// factor at or below 1. The ODP's late pseudo factors cross both
    /// often: on RAA (20,000 simulations) 5.7% have no tail at all, half
    /// the others leave a factor out and have a higher tail, and the oldest
    /// origin's mean reserve is 78% above the plug-in (decision 9). A
    /// constant tail, with a `tail_std_err` for its parameter error, has
    /// neither.
    pub tail: Tail,
    /// Standard error of a constant tail factor: each simulation draws the
    /// factor from the lognormal with the factor as mean and this standard
    /// deviation. `None` (or zero) keeps the factor fixed; the ODP has no
    /// estimate of its own, unlike Mack's extrapolated `tail.se`. Unused
    /// when the factor is 1, and an error with an estimated tail, whose
    /// refit gives its parameter error.
    pub tail_std_err: Option<f64>,
    /// How the segments of [`fit_segments`](Self::fit_segments) and
    /// [`one_year_segments`](Self::one_year_segments) depend on each other;
    /// independent by default.
    pub dependence: SegmentDependence,
}

impl Default for OdpBootstrap {
    fn default() -> Self {
        Self {
            n_sims: 10_000,
            seed: 0,
            process: ProcessDistribution::Gamma,
            tail: Tail::default(),
            tail_std_err: None,
            dependence: SegmentDependence::Independent,
        }
    }
}

/// A fitted ODP bootstrap.
#[derive(Debug, Clone)]
pub struct OdpBootstrapFit {
    /// The deterministic volume-weighted chain ladder the bootstrap is
    /// centred on, with the bootstrap's tail.
    pub chain_ladder: ChainLadderFit,
    /// Fitted incremental values, row-major over origin × development; NaN
    /// where the triangle is not observed.
    pub fitted: Vec<f64>,
    /// Adjusted Pearson residuals, laid out like `fitted`; NaN where not
    /// observed or where the fitted value is zero.
    pub residuals: Vec<f64>,
    /// The scale parameter `phi`.
    pub scale: f64,
    /// Joint distribution of the reserve (sum of future incremental values)
    /// by origin: dimension `origin`, one component per origin period.
    pub reserves: PredictiveDistribution,
}

/// What the bootstrap estimates in one segment before simulating: the
/// fields of [`OdpBootstrapFit`] but the reserves.
#[derive(Debug, Clone, PartialEq)]
pub struct OdpBootstrapSegment {
    /// The volume-weighted chain ladder of the segment, with the
    /// bootstrap's tail in the lifetime view (none in the one-year view).
    pub chain_ladder: ChainLadderFit,
    /// Fitted incremental values, row-major over origin × development.
    pub fitted: Vec<f64>,
    /// Adjusted Pearson residuals, laid out like `fitted`.
    pub residuals: Vec<f64>,
    /// The segment's scale parameter `phi`.
    pub scale: f64,
}

impl ReserveFit for OdpBootstrapSegment {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("scale", self.scale)]
    }
}

impl Resample for OdpBootstrapSegment {
    fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// Every observed cell, row-major: each draws a residual.
    fn draw_positions(&self) -> Vec<usize> {
        let nd = self.residuals.len() / self.chain_ladder.latest_position.len();
        self.chain_ladder
            .latest_position
            .iter()
            .enumerate()
            .flat_map(|(o, &last)| (0..=last).map(move |d| o * nd + d))
            .collect()
    }
}

/// `residuals` at `positions`, in that order: a synchronized pool.
pub(crate) fn pool_at(residuals: &[f64], positions: &[usize]) -> Vec<f64> {
    positions.iter().map(|&p| residuals[p]).collect()
}

/// An ODP bootstrap of every segment of a triangle column.
///
/// Each segment is bootstrapped with its own residuals and scale,
/// independently of the others unless [`OdpBootstrap::dependence`] says
/// otherwise. Simulation `i` uses stream `i` for every segment, in index
/// order, so the result is reproducible and does not depend on the number
/// of threads; a segment's draws differ from bootstrapping it alone. With
/// [`RankCorrelation`](crate::SegmentDependence::RankCorrelation) the
/// segments' simulations are then paired anew and put in a random order,
/// so row `i` is no longer simulation `i`.
///
/// ```
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, OdpBootstrap, Triangle};
/// use prospicio_prob::Distribution;
///
/// let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
/// let ages = [12, 24, 36, 12, 24, 12];
/// let paid = [100.0, 150.0, 165.0, 110.0, 170.0, 120.0];
/// let tri = Triangle::from_long(&Long {
///     keys: &[("lob", &[["Auto"; 6], ["Home"; 6]].concat())],
///     origin: &[origin, origin].concat(),
///     development: DevelopmentColumn::Age(&[ages, ages].concat()),
///     values: &[("paid", &[paid, paid.map(|v| v * 2.0)].concat())],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let boot = OdpBootstrap { n_sims: 1_000, seed: 1, ..Default::default() }
///     .fit_segments(&tri, "paid")?;
/// assert_eq!(boot.reserves.dims(), ["lob", "origin"]);
/// assert_eq!(boot.reserves.n_components(), 6);
/// let by_lob = boot.reserves.aggregate(&["lob"])?;
/// assert_eq!(by_lob.n_components(), 2);
/// assert_eq!(boot.totals().column("scale").unwrap().len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct OdpBootstrapFits {
    /// Each segment's chain ladder, fitted values, residuals and scale.
    pub segments: SegmentFits<OdpBootstrapSegment>,
    /// Joint distribution of the reserve by segment and origin: the key
    /// names and `origin` are its dimensions, and its components run over
    /// the origins of each segment in turn, like the rows of
    /// [`to_long`](Self::to_long).
    pub reserves: PredictiveDistribution,
}

impl OdpBootstrapFits {
    /// One row per segment × origin: the chain ladder's `latest`,
    /// `ultimate` and `reserve`, and the `mean` and `std_dev` of the
    /// bootstrapped reserve.
    pub fn to_long(&self) -> FitTable {
        let mut table = self.segments.to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        table
    }

    /// One row per segment: the chain ladder's totals, the `scale`, and the
    /// `mean` and `std_dev` of the segment's bootstrapped total reserve.
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

/// Positions of segment `s`'s components in a joint distribution whose
/// components run over the origins of each segment of `fits` in turn.
pub(crate) fn components_of<T: ReserveFit>(
    fits: &SegmentFits<T>,
    s: usize,
) -> std::ops::Range<usize> {
    let n_origins = |f: &T| f.chain_ladder().origins.len();
    let start: usize = fits.fits[..s].iter().map(n_origins).sum();
    start..start + n_origins(&fits.fits[s])
}

/// Per simulation, the sum of the components of `draws` in `range`.
fn sums(draws: &PredictiveDistribution, range: std::ops::Range<usize>) -> Sampled {
    let n = draws.n_components();
    let sums = draws
        .draw_matrix()
        .chunks_exact(n)
        .map(|row| row[range.clone()].iter().sum())
        .collect();
    Sampled::new(sums).expect("draws are finite and non-empty")
}

/// Each component of `draws` on its own, in component order.
pub(crate) fn component_sums(draws: &PredictiveDistribution) -> Vec<Sampled> {
    (0..draws.n_components())
        .map(|j| sums(draws, j..j + 1))
        .collect()
}

/// Each segment's total of `draws`, whose components run over the origins
/// of each segment of `fits` in turn.
pub(crate) fn segment_sums<T: ReserveFit>(
    fits: &SegmentFits<T>,
    draws: &PredictiveDistribution,
) -> Vec<Sampled> {
    (0..fits.len())
        .map(|s| sums(draws, components_of(fits, s)))
        .collect()
}

/// The one segment of `fits` chosen as in [`SegmentFits::position`], with
/// its part of the joint `draws` (same dimensions).
pub(crate) fn pick_segment<T: ReserveFit + Clone>(
    fits: &SegmentFits<T>,
    draws: &PredictiveDistribution,
    keys: &[(&str, &str)],
) -> Result<(SegmentFits<T>, PredictiveDistribution)> {
    let s = fits.position(keys)?;
    let range = components_of(fits, s);
    let n = draws.n_components();
    let part = draws
        .draw_matrix()
        .chunks_exact(n)
        .flat_map(|row| row[range.clone()].iter().copied())
        .collect();
    let part = PredictiveDistribution::from_draws(
        draws.dims().to_vec(),
        draws.components()[range].to_vec(),
        part,
        draws.provenance().clone(),
    )?;
    let one = SegmentFits {
        key_names: fits.key_names.clone(),
        labels: vec![fits.labels[s].clone()],
        fits: vec![fits.fits[s].clone()],
    };
    Ok((one, part))
}

/// Appends the `mean` and `std_dev` of each row's draws to `table`, their
/// names led by `prefix`.
pub(crate) fn push_moments(table: &mut FitTable, prefix: &str, draws: &[Sampled]) {
    table.values.push((
        format!("{prefix}mean"),
        draws.iter().map(|d| d.mean()).collect(),
    ));
    table.values.push((
        format!("{prefix}std_dev"),
        draws.iter().map(|d| d.std_dev()).collect(),
    ));
}

/// The key of each origin of a segment labelled `label`: the label's parts,
/// then the origin, as the components of a joint distribution over segments.
pub(crate) fn origin_keys(label: &Label, origins: &[Period]) -> Vec<Vec<KeyValue>> {
    origins
        .iter()
        .map(|&origin| {
            let mut key: Vec<KeyValue> = label.parts().iter().map(|p| p.as_str().into()).collect();
            key.push(origin.into());
            key
        })
        .collect()
}

impl OdpBootstrap {
    /// Bootstraps `column` of a single-segment cumulative triangle. Every
    /// origin must be observed at every age from the first up to its latest.
    /// With a [`tail`](Self::tail), a simulation whose pseudo factors the
    /// tail cannot be refitted on fails, and the call returns
    /// [`Error::TailRefit`] with the number that failed.
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<OdpBootstrapFit> {
        if self.n_sims == 0 {
            return Err(Error::Bootstrap("n_sims must be positive"));
        }
        let segment = triangle.segment(column)?;
        let ages = &segment.ages;
        let (fit, pool) = prepare(&segment, ages, self.tail)?;
        let tail = self.tail_draw(&fit.chain_ladder)?;

        let mut hasher = InputHasher::new();
        hasher.str(column);
        hash_segment(&mut hasher, &segment, &fit.chain_ladder, ages);
        let sim = Simulation {
            segment: &segment,
            latest: &fit.chain_ladder.latest_position,
            fitted: &fit.fitted,
            pool: &pool,
            scale: fit.scale,
            process: self.process,
            tail: tail.as_ref(),
        };
        let reserves = simulate_lifetime(
            vec!["origin".into()],
            segment.origins.iter().map(|&p| vec![p.into()]).collect(),
            self.n_sims,
            self.seed,
            self.provenance(column, hasher),
            |rng, row| sim.run(rng, row, None),
        )?;
        let OdpBootstrapSegment {
            chain_ladder,
            fitted,
            residuals,
            scale,
        } = fit;
        Ok(OdpBootstrapFit {
            chain_ladder,
            fitted,
            residuals,
            scale,
            reserves,
        })
    }

    /// Bootstraps `column` in every segment of a cumulative triangle, each
    /// with its own residuals and scale, into one joint distribution of the
    /// reserves, the segments depending on each other as
    /// [`dependence`](Self::dependence) says; see [`OdpBootstrapFits`]. A
    /// failure names its segment.
    pub fn fit_segments(&self, triangle: &Triangle, column: &str) -> Result<OdpBootstrapFits> {
        if self.n_sims == 0 {
            return Err(Error::Bootstrap("n_sims must be positive"));
        }
        let mut prepared = fit_each(triangle, column, |s| {
            let (fit, pool) = prepare(s, &s.ages, self.tail)?;
            let tail = self.tail_draw(&fit.chain_ladder)?;
            Ok((fit, pool, tail, s.clone()))
        })?;
        let shared = match self.dependence {
            SegmentDependence::Synchronized => {
                let (shared, positions) =
                    synchronize(prepared.fits.iter().map(|(fit, _, _, s)| (s, fit)))?;
                for (fit, pool, _, _) in &mut prepared.fits {
                    *pool = pool_at(&fit.residuals, &positions);
                }
                Some(shared)
            }
            _ => None,
        };

        let keyed = !prepared.key_names.is_empty();
        let mut hasher = InputHasher::new();
        hasher.str(column);
        let mut dims = prepared.key_names.clone();
        dims.push("origin".into());
        let mut components = Vec::new();
        let mut sims = Vec::with_capacity(prepared.len());
        for (label, (fit, pool, tail, segment)) in prepared.iter() {
            hasher.str(&label.to_string());
            hash_segment(&mut hasher, segment, &fit.chain_ladder, &segment.ages);
            components.extend(origin_keys(label, &segment.origins));
            let sim = Simulation {
                segment,
                latest: &fit.chain_ladder.latest_position,
                fitted: &fit.fitted,
                pool,
                scale: fit.scale,
                process: self.process,
                tail: tail.as_ref(),
            };
            sims.push((sim, keyed.then(|| label.to_string())));
        }
        let reserves = simulate_lifetime(
            dims,
            components,
            self.n_sims,
            self.seed,
            self.provenance(column, hasher)
                .param("segments", prepared.len())
                .param("dependence", format!("{:?}", self.dependence)),
            |rng, row| {
                let picks = shared.map(|s| s.picks(rng));
                let mut start = 0;
                for (sim, label) in &sims {
                    let end = start + sim.segment.n_origins;
                    sim.run(rng, &mut row[start..end], picks.as_deref())
                        .map_err(|e| in_segment(label, e))?;
                    start = end;
                }
                Ok(())
            },
        )?;
        let segments = prepared.map(|(fit, _, _, _)| fit.clone());
        let reserves = self.dependence.reorder(reserves, &segments, self.seed)?;
        Ok(OdpBootstrapFits { segments, reserves })
    }

    /// The tail each simulation draws, `None` without one; see
    /// [`tail`](Self::tail).
    fn tail_draw(&self, cl: &ChainLadderFit) -> Result<Option<TailDraw>> {
        TailDraw::new(self.tail, cl, self.tail_std_err)
    }

    fn provenance(&self, column: &str, hasher: InputHasher) -> Provenance {
        let provenance = Provenance::new("odp_bootstrap")
            .param("n_sims", self.n_sims)
            .param("process", format!("{:?}", self.process));
        // Without a tail, the provenance is as it was before tails.
        let provenance = if self.tail.is_none() {
            provenance
        } else {
            provenance
                .param("tail", format!("{:?}", self.tail))
                .param("tail_std_err", format!("{:?}", self.tail_std_err))
        };
        provenance
            .param("column", column)
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(hasher.finish())
    }
}

/// The tail of a bootstrap's lifetime view, drawn in each simulation from
/// its pseudo factors (`docs/design/reserving-v02.md`, decision 9).
#[derive(Debug, Clone)]
pub(crate) struct TailDraw {
    tail: Tail,
    /// The ages the tail is fitted between.
    ages: Vec<Lag>,
    /// The factors estimated on the observed triangle, against which a
    /// constant tail attached before the oldest age measures the pseudo
    /// factors' deviations.
    estimated: Vec<f64>,
    /// For a constant tail with a standard error, the lognormal its factor
    /// to ultimate is drawn from; `None` keeps the fitted factor: a
    /// constant one fixed, an estimated one refitted.
    constant: Option<Lognormal>,
}

impl TailDraw {
    /// The tail of `cl` (fitted with `tail`), `None` when there is none.
    /// `std_err` is the standard error of a constant tail factor (`None`
    /// or zero keeps it fixed), and must be `None` for an estimated tail,
    /// whose parameter error comes from refitting it.
    pub(crate) fn new(
        tail: Tail,
        cl: &ChainLadderFit,
        std_err: Option<f64>,
    ) -> Result<Option<Self>> {
        if tail.is_none() {
            return Ok(None);
        }
        let factor = cl.tail.factor;
        let constant = match (tail, std_err) {
            (_, Some(se)) if !se.is_finite() || se < 0.0 => {
                return Err(Error::Tail(
                    "a tail standard error must be finite and non-negative",
                ));
            }
            (Tail::Constant(_), Some(se)) if se > 0.0 && factor != 1.0 => {
                Some(Lognormal::from_mean_cv(factor, se / factor)?)
            }
            (Tail::Constant(_), _) | (_, None) => None,
            (_, Some(_)) => {
                return Err(Error::Tail(
                    "an estimated tail's parameter error comes from refitting it on each \
                     simulation: a standard error is for a constant tail",
                ));
            }
        };
        Ok(Some(Self {
            tail,
            ages: cl.development.development.clone(),
            estimated: cl.development.ldf.clone(),
            constant,
        }))
    }

    /// One simulation's selected factors within the triangle (its pseudo
    /// factors `pseudo`, with the tail's from its attachment on) and its
    /// factor from the oldest age to ultimate.
    ///
    /// A refitted tail's factors from its attachment on move with the
    /// pseudo factors; a constant tail's do not. So a constant attached
    /// before the oldest age moves each of its factors within the triangle
    /// by the pseudo factor's deviation from its estimate, `f*_k - f_k`:
    /// those ages keep their estimates' parameter error, which Mack's
    /// analytic standard error charges on the selected factors too.
    pub(crate) fn draw(&self, pseudo: &[f64], rng: &mut StreamRng) -> Result<(Vec<f64>, f64)> {
        let (attachment, mut ldf, factor) = self.tail.select(pseudo, &self.ages)?;
        ldf.truncate(pseudo.len());
        if let Tail::Constant(_) = self.tail {
            for ((f, p), e) in ldf
                .iter_mut()
                .zip(pseudo)
                .zip(&self.estimated)
                .skip(attachment)
            {
                *f += p - e;
            }
        }
        let factor = match &self.constant {
            Some(lognormal) => lognormal.sample(rng, 1)[0],
            None => factor,
        };
        Ok((ldf, factor))
    }
}

/// `error`, inside the segment `label` when there are several.
pub(crate) fn in_segment(label: &Option<String>, error: Error) -> Error {
    match label {
        Some(label) => Error::InSegment {
            label: label.clone(),
            source: Box::new(error),
        },
        None => error,
    }
}

/// [`PredictiveDistribution::simulate`] for a lifetime view whose
/// simulations can fail (a tail that cannot be refitted on a simulation's
/// pseudo factors). A failed simulation's row is zeroed, and once all have
/// run the call returns [`Error::TailRefit`] with the number that failed
/// and the failure whose message sorts first, so the error does not depend
/// on the threads. Without failures the draws are `simulate`'s.
pub(crate) fn simulate_lifetime(
    dims: Vec<String>,
    components: Vec<ComponentKey>,
    n_sims: usize,
    seed: u64,
    provenance: Provenance,
    run: impl Fn(&mut StreamRng, &mut [f64]) -> Result<()> + Sync,
) -> Result<PredictiveDistribution> {
    let failures = Mutex::new(Failures::default());
    let draws = PredictiveDistribution::simulate(
        dims,
        components,
        n_sims,
        seed,
        provenance,
        |rng, row| {
            if let Err(e) = run(rng, row) {
                // Keep the draws finite; the failure is reported.
                row.fill(0.0);
                failures
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .record(e);
            }
        },
    )?;
    let failures = failures
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match failures.first {
        Some((_, source)) => Err(Error::TailRefit {
            failed: failures.count,
            n_sims,
            source: Box::new(source),
        }),
        None => Ok(draws),
    }
}

/// Hashes the observed cells of `segment` that the bootstrap uses.
pub(crate) fn hash_segment(
    hasher: &mut InputHasher,
    segment: &Segment,
    cl: &ChainLadderFit,
    ages: &[Lag],
) {
    for (o, &last) in cl.latest_position.iter().enumerate() {
        hasher.str(&segment.origins[o].to_string());
        for (d, &age) in ages[..=last].iter().enumerate() {
            hasher.i64(age as i64);
            hasher.f64s(&[segment.get(o, d).expect("checked when prepared")]);
        }
    }
}

/// The chain ladder (with `tail`, which leaves the fitted values and
/// residuals as they are), fitted values, residuals and scale of one
/// segment, and the pool of residuals to resample.
pub(crate) fn prepare(
    segment: &Segment,
    ages: &[Lag],
    tail: Tail,
) -> Result<(OdpBootstrapSegment, Vec<f64>)> {
    let (no, nd) = (segment.n_origins, segment.n_dev);
    let chain_ladder = ChainLadder {
        tail,
        ..Default::default()
    }
    .fit_segment(segment, ages)?;
    let ldf = &chain_ladder.development.ldf;
    let latest = &chain_ladder.latest_position;
    for (o, &last) in latest.iter().enumerate() {
        if (0..=last).any(|d| segment.get(o, d).is_none()) {
            return Err(Error::Bootstrap(
                "every origin must be observed from the first age to its latest",
            ));
        }
    }

    // Fitted cumulative values back from the latest diagonal, then
    // incrementals.
    let mut fitted = vec![f64::NAN; no * nd];
    let mut observed = vec![f64::NAN; no * nd];
    for o in 0..no {
        let mut cum = vec![0.0; latest[o] + 1];
        cum[latest[o]] = chain_ladder.latest[o];
        for d in (0..latest[o]).rev() {
            cum[d] = cum[d + 1] / ldf[d];
        }
        let mut previous_fit = 0.0;
        let mut previous_obs = 0.0;
        for d in 0..=latest[o] {
            let obs = segment.get(o, d).expect("checked above");
            fitted[o * nd + d] = cum[d] - previous_fit;
            observed[o * nd + d] = obs - previous_obs;
            previous_fit = cum[d];
            previous_obs = obs;
        }
    }

    let n_obs = fitted.iter().filter(|m| !m.is_nan()).count();
    let n_params = no + nd - 1;
    if n_obs <= n_params {
        return Err(Error::Bootstrap(
            "too few observed cells: degrees of freedom must be positive",
        ));
    }
    let unscaled: Vec<f64> = fitted
        .iter()
        .zip(&observed)
        .map(|(&m, &x)| {
            if m.is_nan() || m == 0.0 {
                f64::NAN
            } else {
                (x - m) / m.abs().sqrt()
            }
        })
        .collect();
    let dof = (n_obs - n_params) as f64;
    let scale = unscaled
        .iter()
        .filter(|r| !r.is_nan())
        .map(|r| r * r)
        .sum::<f64>()
        / dof;
    let adjust = (n_obs as f64 / dof).sqrt();
    let residuals: Vec<f64> = unscaled.iter().map(|r| r * adjust).collect();
    let pool: Vec<f64> = residuals.iter().copied().filter(|r| !r.is_nan()).collect();
    if pool.is_empty() {
        return Err(Error::Bootstrap("no residuals to resample"));
    }
    Ok((
        OdpBootstrapSegment {
            chain_ladder,
            fitted,
            residuals,
            scale,
        },
        pool,
    ))
}

/// Inputs shared by every simulation.
pub(crate) struct Simulation<'a> {
    pub(crate) segment: &'a Segment,
    pub(crate) latest: &'a [usize],
    pub(crate) fitted: &'a [f64],
    pub(crate) pool: &'a [f64],
    pub(crate) scale: f64,
    pub(crate) process: ProcessDistribution,
    /// The lifetime view's tail; `None` in the one-year view.
    pub(crate) tail: Option<&'a TailDraw>,
}

impl Simulation<'_> {
    /// One bootstrap replicate: fills `reserves` with each origin's
    /// simulated reserve; `picks` as in [`resample`](Self::resample). With
    /// a tail, the selected pseudo factors and the tail factor are drawn
    /// after the pseudo triangle, and the step to ultimate is each origin's
    /// last increment; it fails when the tail cannot be refitted on the
    /// pseudo factors. A synchronized bootstrap shares only the residuals'
    /// positions: the tail is refitted on this segment's pseudo factors
    /// (so it follows the shared residuals), and a constant tail's
    /// lognormal draw is this segment's own.
    fn run(
        &self,
        rng: &mut StreamRng,
        reserves: &mut [f64],
        picks: Option<&[usize]>,
    ) -> Result<()> {
        let nd = self.segment.n_dev;
        let (pseudo, factors) = self.resample(rng, picks);
        let (factors, tail) = match self.tail {
            None => (factors, None),
            Some(tail) => {
                let (selected, factor) = tail.draw(&factors, rng)?;
                (selected, Some(factor))
            }
        };
        for (o, reserve) in reserves.iter_mut().enumerate() {
            let mut cum = pseudo[o * nd + self.latest[o]];
            let mut total = 0.0;
            for f in factors[self.latest[o]..].iter().chain(&tail) {
                let next = cum * f;
                total += self.with_process(next - cum, rng);
                cum = next;
            }
            *reserve = total;
        }
        Ok(())
    }

    /// A pseudo cumulative triangle from resampled residuals, row-major
    /// over origin × development (zero where not observed), and its
    /// volume-weighted factors: the parameter error of one replicate. The
    /// residual of the `c`-th observed cell (row-major) is drawn from `rng`,
    /// or, synchronized, is `pool[picks[c]]`.
    pub(crate) fn resample(
        &self,
        rng: &mut StreamRng,
        picks: Option<&[usize]>,
    ) -> (Vec<f64>, Vec<f64>) {
        let (no, nd) = (self.segment.n_origins, self.segment.n_dev);

        // Pseudo cumulative triangle from resampled residuals.
        let mut pseudo = vec![0.0; no * nd];
        let mut cell = 0;
        for o in 0..no {
            let mut cum = 0.0;
            for d in 0..=self.latest[o] {
                let m = self.fitted[o * nd + d];
                let k = match picks {
                    Some(picks) => picks[cell],
                    None => ((rng.next_open01() * self.pool.len() as f64) as usize)
                        .min(self.pool.len() - 1),
                };
                cell += 1;
                cum += m + self.pool[k] * m.abs().sqrt();
                pseudo[o * nd + d] = cum;
            }
        }

        // Volume-weighted factors of the pseudo triangle; 1 where the
        // earlier values sum to zero, as R does.
        let factors: Vec<f64> = (0..nd.saturating_sub(1))
            .map(|k| {
                let (mut num, mut den) = (0.0, 0.0);
                for o in (0..no).filter(|&o| self.latest[o] > k) {
                    num += pseudo[o * nd + k + 1];
                    den += pseudo[o * nd + k];
                }
                if den == 0.0 { 1.0 } else { num / den }
            })
            .collect();
        (pseudo, factors)
    }

    /// An incremental value with expected value `mean` and the process
    /// error of [`ProcessDistribution`].
    pub(crate) fn with_process(&self, mean: f64, rng: &mut StreamRng) -> f64 {
        match self.process {
            ProcessDistribution::None => mean,
            ProcessDistribution::Gamma => {
                if mean == 0.0 || self.scale == 0.0 {
                    return mean;
                }
                let gamma = Gamma::new(mean.abs() / self.scale, self.scale)
                    .expect("shape and scale are finite and positive");
                mean.signum() * gamma.sample(rng, 1)[0]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::{annual, raa};
    use crate::{DevelopmentFit, TailConstant};
    use prospicio_core::Period;

    fn boot(process: ProcessDistribution, n_sims: usize) -> OdpBootstrapFit {
        OdpBootstrap {
            n_sims,
            seed: 7,
            process,
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap()
    }

    #[test]
    fn raa_residuals_and_scale() {
        // R ChainLadder BootChainLadder(RAA): sum of squared unscaled
        // residuals / 36 degrees of freedom.
        let fit = boot(ProcessDistribution::None, 10);
        assert_eq!(fit.residuals.iter().filter(|r| !r.is_nan()).count(), 55);
        // The fitted values reproduce each origin's latest cumulative.
        for o in 0..10 {
            let sum: f64 = fit.fitted[o * 10..o * 10 + 10 - o].iter().sum();
            assert!((sum - fit.chain_ladder.latest[o]).abs() < 1e-6);
        }
        // Corners fit exactly: the oldest origin's last cell, the newest's
        // first.
        assert!(fit.residuals[9].abs() < 1e-9);
        assert!(fit.residuals[90].abs() < 1e-9);
        assert!(fit.scale > 0.0);
    }

    #[test]
    fn parameter_only_mean_is_near_chain_ladder() {
        let fit = boot(ProcessDistribution::None, 4_000);
        let cl = fit.chain_ladder.total_reserve();
        let mean = fit.reserves.total().mean();
        assert!((mean / cl - 1.0).abs() < 0.05, "{mean} vs {cl}");
        assert_eq!(
            fit.reserves
                .marginal(&vec![Period::year(1981).into()])
                .unwrap()
                .mean(),
            0.0
        );
    }

    #[test]
    fn reproducible_by_seed() {
        let a = boot(ProcessDistribution::Gamma, 300);
        let b = boot(ProcessDistribution::Gamma, 300);
        assert_eq!(a.reserves.draw_matrix(), b.reserves.draw_matrix());
        let c = OdpBootstrap {
            n_sims: 300,
            seed: 8,
            process: ProcessDistribution::Gamma,
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        assert_ne!(a.reserves.draw_matrix(), c.reserves.draw_matrix());
        assert_eq!(a.reserves.provenance().model, "odp_bootstrap");
    }

    #[test]
    fn process_error_widens_the_distribution() {
        let param = boot(ProcessDistribution::None, 2_000);
        let full = boot(ProcessDistribution::Gamma, 2_000);
        assert!(full.reserves.total().std_dev() > param.reserves.total().std_dev());
    }

    #[test]
    fn rejects_holes_and_tiny_triangles() {
        use crate::{DevelopmentColumn, Grain, Long, Month};
        let origin = [2019, 2019, 2019, 2020, 2020, 2021].map(Month::january);
        let holes = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 36, 12, 36, 12]),
            values: &[("paid", &[1.0, 2.0, 3.0, 1.0, 3.0, 1.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let b = OdpBootstrap::default();
        assert!(matches!(b.fit(&holes, "paid"), Err(Error::Bootstrap(_))));
        let tiny = annual(2020, &[&[1.0, 2.0], &[1.0]]);
        assert!(matches!(b.fit(&tiny, "values"), Err(Error::Bootstrap(_))));
        let none = OdpBootstrap {
            n_sims: 0,
            ..Default::default()
        };
        assert!(matches!(
            none.fit(&raa(), "values"),
            Err(Error::Bootstrap(_))
        ));
    }

    /// RAA's bootstrap with this tail and process, and simulation `i`'s
    /// pseudo triangle and factors, drawn as the bootstrap draws them.
    fn tailed(tail: Tail, tail_std_err: Option<f64>, process: ProcessDistribution) -> OdpBootstrap {
        OdpBootstrap {
            n_sims: 2_000,
            seed: 11,
            process,
            tail,
            tail_std_err,
            dependence: SegmentDependence::Independent,
        }
    }

    /// The pseudo triangle and factors of simulation `i` of RAA, and the
    /// stream positioned after them.
    fn pseudo(seed: u64, i: u64) -> (Vec<f64>, Vec<f64>, StreamRng) {
        let segment = raa().segment("values").unwrap();
        let (fit, pool) = prepare(&segment, &segment.ages, Tail::default()).unwrap();
        let sim = Simulation {
            segment: &segment,
            latest: &fit.chain_ladder.latest_position,
            fitted: &fit.fitted,
            pool: &pool,
            scale: fit.scale,
            process: ProcessDistribution::None,
            tail: None,
        };
        let mut rng = StreamRng::new(seed, i);
        let (pseudo, factors) = sim.resample(&mut rng, None);
        (pseudo, factors, rng)
    }

    #[test]
    fn no_tail_draws_as_before() {
        // A constant 1 at the oldest age is no tail whatever its decay:
        // the same draws and provenance as the default.
        let plain = boot(ProcessDistribution::Gamma, 300);
        let one = OdpBootstrap {
            tail: Tail::Constant(TailConstant {
                factor: 1.0,
                decay: 0.75,
                attachment_age: None,
            }),
            tail_std_err: Some(0.01),
            ..tailed(Tail::default(), None, ProcessDistribution::Gamma)
        };
        let one = OdpBootstrap {
            n_sims: 300,
            seed: 7,
            ..one
        }
        .fit(&raa(), "values")
        .unwrap();
        assert_eq!(plain.reserves.draw_matrix(), one.reserves.draw_matrix());
        assert_eq!(plain.reserves.provenance(), one.reserves.provenance());
    }

    #[test]
    fn constant_tail_is_one_more_increment() {
        // Parameter error alone: every simulation's reserve of the oldest
        // origin is its pseudo latest value times the tail less 1, and with
        // a standard error the factor is the lognormal's draw after the
        // pseudo triangle, on the same stream.
        let nd = 10;
        for std_err in [None, Some(0.02)] {
            let fit = tailed(1.05.into(), std_err, ProcessDistribution::None)
                .fit(&raa(), "values")
                .unwrap();
            assert!((fit.chain_ladder.tail.factor - 1.05).abs() < 1e-12);
            let lognormal = Lognormal::from_mean_cv(
                fit.chain_ladder.tail.factor,
                0.02 / fit.chain_ladder.tail.factor,
            )
            .unwrap();
            for i in [0, 1, 777] {
                let (pseudo, factors, mut rng) = pseudo(11, i);
                let factor = match std_err {
                    None => fit.chain_ladder.tail.factor,
                    Some(_) => lognormal.sample(&mut rng, 1)[0],
                };
                let cum = pseudo[9];
                let want = 0.0 + (cum * factor - cum);
                assert_eq!(
                    fit.reserves.row(i as usize).unwrap()[0],
                    want,
                    "{std_err:?} {i}"
                );
                // The next origin develops on its pseudo factor, then the tail.
                let (c1, c2) = (pseudo[nd + 8], pseudo[nd + 8] * factors[8]);
                let want = 0.0 + (c2 - c1) + (c2 * factor - c2);
                assert_eq!(fit.reserves.row(i as usize).unwrap()[1], want);
            }
        }
    }

    #[test]
    fn constant_tail_attached_early_moves_with_the_pseudo_factors() {
        // RAA with a constant 1.05 attached at 84 months: the pseudo factors
        // before the attachment are kept, and each of the tail's factors
        // within the triangle moves by its pseudo factor's deviation from
        // the estimate, so those ages keep their parameter error, which
        // Mack charges on them (decision 9). The factor to ultimate is the
        // tail's.
        let tail = Tail::Constant(TailConstant {
            factor: 1.05,
            attachment_age: Some(84),
            ..Default::default()
        });
        let cl = crate::ChainLadder {
            tail,
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        assert_eq!(cl.tail.attachment, 6);
        let draw = TailDraw::new(tail, &cl, None).unwrap().unwrap();
        let estimated = &cl.development.ldf;
        let mut rng = StreamRng::new(1, 0);
        // At the estimates the selected factors are the chain ladder's.
        let (ldf, factor) = draw.draw(estimated, &mut rng).unwrap();
        assert_eq!(ldf, cl.ldf());
        assert_eq!(factor, cl.tail.factor);
        let pseudo: Vec<f64> = estimated
            .iter()
            .enumerate()
            .map(|(k, f)| f + 0.001 * (k as f64 - 4.0))
            .collect();
        let (ldf, factor) = draw.draw(&pseudo, &mut rng).unwrap();
        assert_eq!(ldf[..6], pseudo[..6]);
        for k in 6..9 {
            assert_eq!(ldf[k], cl.ldf()[k] + (pseudo[k] - estimated[k]), "{k}");
        }
        assert_eq!(factor, cl.tail.factor);
    }

    #[test]
    fn estimated_tail_is_refitted_on_the_pseudo_factors() {
        let fit = tailed(Tail::LogLinear, None, ProcessDistribution::None)
            .fit(&raa(), "values")
            .unwrap();
        let dev = &fit.chain_ladder.development;
        let mut factors_seen = Vec::new();
        for i in [0, 5, 1999] {
            let (pseudo, factors, _) = pseudo(11, i);
            let refit = Tail::LogLinear
                .fit(&DevelopmentFit {
                    ldf: factors.clone(),
                    ..dev.clone()
                })
                .unwrap()
                .factor;
            let cum = pseudo[9];
            assert_eq!(
                fit.reserves.row(i as usize).unwrap()[0],
                0.0 + (cum * refit - cum)
            );
            factors_seen.push(refit);
        }
        // The tail moves with the pseudo factors: its parameter error.
        assert!(factors_seen.windows(2).all(|w| w[0] != w[1]));
    }

    #[test]
    fn tail_process_error_has_the_odp_variance() {
        // The oldest origin's reserve is the Gamma of mean P (T - 1) and
        // variance phi |mean|, P its pseudo latest value: the sum of its
        // fitted increments m plus a resampled residual times sqrt(|m|).
        // So its variance is (T - 1)^2 v sum|m| + phi (T - 1) E[P], with v
        // the pool's variance and E[P] = latest + mean(pool) sum(sqrt|m|).
        let n_sims = 20_000;
        let t = 1.05;
        let fit = OdpBootstrap {
            n_sims,
            ..tailed(t.into(), None, ProcessDistribution::Gamma)
        }
        .fit(&raa(), "values")
        .unwrap();
        let pool: Vec<f64> = fit
            .residuals
            .iter()
            .copied()
            .filter(|r| !r.is_nan())
            .collect();
        let n = pool.len() as f64;
        let mean = pool.iter().sum::<f64>() / n;
        let v = pool.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let m = &fit.fitted[..10];
        let (abs, root): (f64, f64) = m
            .iter()
            .fold((0.0, 0.0), |(a, r), x| (a + x.abs(), r + x.abs().sqrt()));
        let p = fit.chain_ladder.latest[0] + mean * root;
        let want = ((t - 1.0).powi(2) * v * abs + fit.scale * (t - 1.0) * p).sqrt();
        let x: Vec<f64> = fit
            .reserves
            .draw_matrix()
            .chunks(10)
            .map(|r| r[0])
            .collect();
        let mu = x.iter().sum::<f64>() / n_sims as f64;
        let m2 = x.iter().map(|y| (y - mu).powi(2)).sum::<f64>() / n_sims as f64;
        let m4 = x.iter().map(|y| (y - mu).powi(4)).sum::<f64>() / n_sims as f64;
        let sd = m2.sqrt();
        let error = sd * ((m4 / (m2 * m2) - 1.0) / (4.0 * n_sims as f64)).sqrt();
        assert!(
            (sd - want).abs() < 5.0 * error,
            "{sd} vs {want} (SE {error})"
        );
        assert!((mu - p * (t - 1.0)).abs() < 5.0 * sd / (n_sims as f64).sqrt());
    }

    #[test]
    fn tail_settings_are_checked() {
        let raa = raa();
        let bad =
            |tail, std_err| tailed(tail, std_err, ProcessDistribution::Gamma).fit(&raa, "values");
        assert!(matches!(
            bad(Tail::LogLinear, Some(0.01)),
            Err(Error::Tail(_))
        ));
        assert!(matches!(bad(1.05.into(), Some(-0.01)), Err(Error::Tail(_))));
        assert!(matches!(
            bad(1.05.into(), Some(f64::NAN)),
            Err(Error::Tail(_))
        ));
        // The one-year view takes its tail from the method.
        let one_year = tailed(1.05.into(), None, ProcessDistribution::Gamma).one_year(
            &raa,
            "values",
            &crate::OneYearMethod::ChainLadder(ChainLadder::default()),
        );
        assert!(matches!(one_year, Err(Error::Bootstrap(_))));
    }
}
