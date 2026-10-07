//! The one-year view by re-reserving on the ODP bootstrap ("actuary in the
//! box": Ohlsson and Lauzeningks 2009; England, Verrall and Wüthrich 2019),
//! for any method, weighting or tail (`docs/design/reserving-v02.md`,
//! decision 8).
//!
//! Merz and Wüthrich's claims development result
//! ([`MackFit::claims_development_result`](crate::MackFit::claims_development_result))
//! is exact only for volume-weighted factors without a tail. Here each
//! simulation, on its own random stream:
//!
//! 1. resamples the adjusted residuals into a pseudo triangle and
//!    re-estimates the volume-weighted factors `f*`, as the
//!    [`OdpBootstrap`] does (parameter error), once for the whole year;
//! 2. simulates every cell of every origin in the coming year (below), in
//!    development order from the origin's latest cell, as the
//!    [`OdpBootstrap`] projects (England 2002): the increment into age
//!    `j + 1` has mean `C*_j (f*_j - 1)`, with `C*_j` the origin's pseudo
//!    latest value carried forward on the pseudo factors
//!    (`C*_j+1 = C*_j f*_j`), and its own process error with the
//!    bootstrap's scale. The pseudo latest value carries the estimation
//!    error of the origin's level. An origin already at the last age gets
//!    no new cell;
//! 3. adds the increments to the origin's observed latest value, appends
//!    the cells of the coming year to the observed triangle and refits the
//!    method on it (an exposure column keeps each origin's latest value,
//!    and Cape Cod trends to the valuation twelve months later);
//! 4. records the claims development result `CDR = U0 - U1`, the opening
//!    ultimate less the re-estimated one. It equals the opening reserve less
//!    the year's simulated payment and the closing reserve, so a negative
//!    CDR is an adverse development.
//!
//! The coming year is the twelve months after the segment's valuation `V`,
//! the valuation month of its latest cell (the triangle's valuation unless
//! the segment stops earlier): a cell is in it when its valuation month `v`
//! has `V < v <= V + 12 months`. With an annual development grain that is
//! one cell per origin, the next diagonal; with a quarterly grain four,
//! fewer for an origin that reaches the last age within the year. An
//! origin whose latest cell lags `V` develops from that cell: the cells
//! between it and the coming year are drawn as steps on the way but stay
//! unobserved, and only the year's cells are appended.
//!
//! An origin whose remaining cells all fall in the coming year therefore
//! has a one-year view that is its whole run-off, distributed as its
//! lifetime [`OdpBootstrap`] reserve.
//!
//! [`MackBootstrap::one_year`](crate::MackBootstrap::one_year) replaces
//! steps 1 and 2 with Mack's model (England, Verrall and Wüthrich 2019,
//! Appendix 1; see [`crate::mack_bootstrap`]) and keeps steps 3 and 4, so
//! the two process models can be compared on the same re-reserving: with
//! the volume-weighted chain ladder and no tail, Mack's reproduces Merz and
//! Wüthrich's standard error.
//!
//! A new origin period written in the coming year is not simulated, as in
//! Merz–Wüthrich, and the development beyond the last age moves only
//! through the refitted tail.

use std::sync::Mutex;

use prospicio_core::{Month, StreamRng};
use prospicio_prob::{InputHasher, KeyValue, PredictiveDistribution, Provenance};

use crate::bootstrap::{
    OdpBootstrap, OdpBootstrapSegment, ProcessDistribution, Simulation, component_sums,
    hash_segment, origin_keys, pick_segment, prepare, push_moments, segment_sums,
};
use crate::chain_ladder::{ChainLadder, ChainLadderFit};
use crate::error::{Error, Result};
use crate::expected_loss::{Benktander, BornhuetterFerguson, CapeCod, ExpectedLoss};
use crate::segments::{FitTable, ReserveFit, SegmentFits, fit_each, fit_each_with_exposure};
use crate::triangle::{Segment, Triangle};

/// The reserving method the one-year bootstrap refits at the end of the
/// year. The expected-loss methods name their exposure column.
///
/// ```
/// use prospicio_reserving::{BornhuetterFerguson, ChainLadder, OneYearMethod};
///
/// let cl = OneYearMethod::ChainLadder(ChainLadder::default());
/// assert_eq!(cl.exposure(), None);
/// let bf = OneYearMethod::BornhuetterFerguson(
///     BornhuetterFerguson { apriori: 0.7, ..Default::default() },
///     "premium".into(),
/// );
/// assert_eq!(bf.exposure(), Some("premium"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum OneYearMethod {
    /// The chain ladder, with any development estimator and tail.
    ChainLadder(ChainLadder),
    /// The expected loss ratio method and its exposure column.
    ExpectedLoss(ExpectedLoss, String),
    /// Bornhuetter–Ferguson and its exposure column.
    BornhuetterFerguson(BornhuetterFerguson, String),
    /// Benktander and its exposure column.
    Benktander(Benktander, String),
    /// Cape Cod and its exposure column.
    CapeCod(CapeCod, String),
}

impl OneYearMethod {
    /// The exposure column, for the expected-loss methods.
    pub fn exposure(&self) -> Option<&str> {
        match self {
            Self::ChainLadder(_) => None,
            Self::ExpectedLoss(_, e)
            | Self::BornhuetterFerguson(_, e)
            | Self::Benktander(_, e)
            | Self::CapeCod(_, e) => Some(e),
        }
    }

    /// The method's ultimate per origin of `segment`, with `exposure` the
    /// exposure column's segment at the same index position (required by
    /// the expected-loss methods) and Cape Cod's trend to `valuation`.
    fn ultimate(
        &self,
        segment: &Segment,
        exposure: Option<&Segment>,
        valuation: Month,
    ) -> Result<Vec<f64>> {
        let exposure = || exposure.expect("an exposure method gets its exposure segment");
        Ok(match self {
            Self::ChainLadder(cl) => cl.fit_segment(segment, &segment.ages)?.ultimate,
            Self::ExpectedLoss(m, c) => {
                m.as_benktander()
                    .fit_segment(segment, exposure(), c)?
                    .ultimate
            }
            Self::BornhuetterFerguson(m, c) => {
                m.as_benktander()
                    .fit_segment(segment, exposure(), c)?
                    .ultimate
            }
            Self::Benktander(m, c) => m.fit_segment(segment, exposure(), c)?.ultimate,
            Self::CapeCod(m, c) => {
                m.fit_segment(segment, exposure(), c, valuation)?
                    .expected_loss
                    .ultimate
            }
        })
    }
}

/// The one-year bootstrap of a single-segment triangle.
///
/// ```
/// use prospicio_reserving::{
///     ChainLadder, DevelopmentColumn, Grain, Long, Month, OdpBootstrap, OneYearMethod, Triangle,
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
/// let boot = OdpBootstrap { n_sims: 2_000, seed: 42, ..Default::default() };
/// let fit = boot.one_year(&tri, "paid", &OneYearMethod::ChainLadder(ChainLadder::default()))?;
/// assert_eq!(fit.cdr.dims(), ["origin"]);
/// // 2020 is fully developed: no new cell, no change.
/// assert_eq!(fit.opening_reserve[0], 0.0);
/// assert!(fit.cdr.draw_matrix().chunks(4).all(|row| row[0] == 0.0));
/// // The one-year view is narrower than the lifetime view.
/// let lifetime = boot.fit(&tri, "paid")?.reserves.std_dev();
/// assert!(fit.cdr.std_dev() < lifetime);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// `B` is what the bootstrap estimated before simulating: the ODP's
/// [`OdpBootstrapSegment`] by default, Mack's
/// [`MackBootstrapSegment`](crate::MackBootstrapSegment) from
/// [`MackBootstrap::one_year`](crate::MackBootstrap::one_year).
#[derive(Debug, Clone)]
pub struct OneYearFit<B = OdpBootstrapSegment> {
    /// The bootstrap model fitted to the observed triangle: for the ODP its
    /// volume-weighted chain ladder, fitted values, residuals and scale.
    pub bootstrap: B,
    /// The method's ultimate per origin on the observed triangle.
    pub opening_ultimate: Vec<f64>,
    /// The opening ultimate less the latest value, per origin.
    pub opening_reserve: Vec<f64>,
    /// Joint distribution of the claims development result by origin:
    /// dimension `origin`, one component per origin period.
    pub cdr: PredictiveDistribution,
}

/// What the one-year bootstrap estimates in one segment before simulating:
/// the fields of [`OneYearFit`] but the claims development result.
///
/// ```
/// use prospicio_reserving::{ChainLadder, OdpBootstrap, OneYearMethod};
/// # use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// # let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
/// # let tri = Triangle::from_long(&Long {
/// #     keys: &[("lob", &["Auto"; 6])],
/// #     origin: &origin,
/// #     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
/// #     values: &[("paid", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0])],
/// #     origin_grain: Grain::Year,
/// #     development_grain: Grain::Year,
/// #     cumulative: true,
/// # })?;
/// let method = OneYearMethod::ChainLadder(ChainLadder::default());
/// let fits = OdpBootstrap { n_sims: 100, ..Default::default() }
///     .one_year_segments(&tri, "paid", &method)?;
/// let auto = &fits.segments.fits[0];
/// assert_eq!(auto.opening_ultimate, auto.bootstrap.chain_ladder.ultimate);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OneYearSegment<B = OdpBootstrapSegment> {
    /// The segment's bootstrap model: for the ODP its volume-weighted chain
    /// ladder, fitted values, residuals and scale.
    pub bootstrap: B,
    /// The method's ultimate per origin on the observed triangle.
    pub opening_ultimate: Vec<f64>,
    /// The opening ultimate less the latest value, per origin.
    pub opening_reserve: Vec<f64>,
}

/// The long tables of the bootstrap model's chain ladder, with the
/// method's opening ultimate and the model's own segment totals (the
/// ODP's `scale`).
impl<B: ReserveFit> ReserveFit for OneYearSegment<B> {
    fn chain_ladder(&self) -> &ChainLadderFit {
        self.bootstrap.chain_ladder()
    }

    fn ultimate(&self) -> &[f64] {
        &self.opening_ultimate
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        self.bootstrap.total_columns()
    }
}

/// The one-year bootstrap of every segment of a triangle column.
///
/// Segments are bootstrapped independently, each with its own residuals
/// and scale; simulation `i` uses stream `i` for every segment in turn, as
/// in [`OdpBootstrapFits`](crate::OdpBootstrapFits).
///
/// ```
/// use prospicio_reserving::{
///     BornhuetterFerguson, DevelopmentColumn, Grain, Long, Month, OdpBootstrap, OneYearMethod,
///     Triangle,
/// };
///
/// let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
/// let ages = [12, 24, 36, 12, 24, 12];
/// let paid = [100.0, 150.0, 165.0, 110.0, 170.0, 120.0];
/// let premium = [200.0, 200.0, 200.0, 220.0, 220.0, 240.0];
/// let tri = Triangle::from_long(&Long {
///     keys: &[("lob", &[["Auto"; 6], ["Home"; 6]].concat())],
///     origin: &[origin, origin].concat(),
///     development: DevelopmentColumn::Age(&[ages, ages].concat()),
///     values: &[
///         ("paid", &[paid, paid.map(|v| v * 2.0)].concat()),
///         ("premium", &[premium, premium.map(|v| v * 2.0)].concat()),
///     ],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let bf = BornhuetterFerguson { apriori: 0.8, ..Default::default() };
/// let method = OneYearMethod::BornhuetterFerguson(bf, "premium".into());
/// let fits = OdpBootstrap { n_sims: 500, seed: 3, ..Default::default() }
///     .one_year_segments(&tri, "paid", &method)?;
/// assert_eq!(fits.cdr.dims(), ["lob", "origin"]);
/// assert_eq!(fits.cdr.n_components(), 6);
/// let totals = fits.totals();
/// assert_eq!(totals.column("cdr_std_dev").unwrap().len(), 2);
/// let home = fits.segment(&[("lob", "Home")])?;
/// assert_eq!(home.cdr.n_components(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct OneYearFits<B = OdpBootstrapSegment> {
    /// Each segment's bootstrap and opening ultimate.
    pub segments: SegmentFits<OneYearSegment<B>>,
    /// Joint distribution of the claims development result by segment and
    /// origin: the key names and `origin` are its dimensions, and its
    /// components run over the origins of each segment in turn, like the
    /// rows of [`to_long`](Self::to_long).
    pub cdr: PredictiveDistribution,
}

impl<B: ReserveFit + Clone> OneYearFits<B> {
    /// One row per segment × origin: the `latest` value, the method's
    /// `opening_ultimate` and `opening_reserve`, and the `cdr_mean` and
    /// `cdr_std_dev` of the simulated claims development result.
    pub fn to_long(&self) -> FitTable {
        let mut table = opening_names(self.segments.to_long());
        push_moments(&mut table, "cdr_", &component_sums(&self.cdr));
        table
    }

    /// One row per segment: the totals of `to_long`'s columns, the ODP
    /// bootstrap's `scale` (none for Mack's), and the `cdr_mean` and
    /// `cdr_std_dev` of the segment's total claims development result.
    pub fn totals(&self) -> FitTable {
        let mut table = opening_names(self.segments.totals());
        let totals = segment_sums(&self.segments, &self.cdr);
        push_moments(&mut table, "cdr_", &totals);
        table
    }

    /// The one segment chosen as in [`SegmentFits::position`], with its
    /// part of the joint claims development result (same dimensions).
    pub fn segment(&self, keys: &[(&str, &str)]) -> Result<Self> {
        let (segments, cdr) = pick_segment(&self.segments, &self.cdr, keys)?;
        Ok(Self { segments, cdr })
    }
}

/// Renames a table's `ultimate` and `reserve` to `opening_ultimate` and
/// `opening_reserve`.
fn opening_names(mut table: FitTable) -> FitTable {
    for (name, _) in &mut table.values {
        if name == "ultimate" || name == "reserve" {
            *name = format!("opening_{name}");
        }
    }
    table
}

/// The cells one origin develops through in the coming year: from its
/// latest observed position `latest`, positions `latest + 1` to `last` are
/// drawn in turn, and those from `first` on, whose valuation falls in the
/// year, are appended to the triangle (`latest < first <= last`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct YearCells {
    pub(crate) origin: usize,
    pub(crate) latest: usize,
    pub(crate) first: usize,
    pub(crate) last: usize,
}

impl YearCells {
    /// The cells of every origin of `segment`, whose latest positions are
    /// `latest`, in the twelve months after the segment's valuation;
    /// `valuation` gives a cell's valuation month. An origin with no cell in
    /// the year (at the last age) has none.
    fn of(
        segment: &Segment,
        latest: &[usize],
        valuation: impl Fn(usize, usize) -> Month,
    ) -> Vec<Self> {
        let opening = latest
            .iter()
            .enumerate()
            .map(|(o, &d)| valuation(o, d))
            .max()
            .expect("a segment has an origin");
        let closing = opening.add_months(12);
        latest
            .iter()
            .enumerate()
            .filter_map(|(origin, &d)| {
                let last = (d + 1..segment.n_dev)
                    .take_while(|&e| valuation(origin, e) <= closing)
                    .last()?;
                let first = (d + 1..=last).find(|&e| valuation(origin, e) > opening)?;
                Some(Self {
                    origin,
                    latest: d,
                    first,
                    last,
                })
            })
            .collect()
    }
}

/// A bootstrap model that simulates a segment's coming year: what it
/// estimated on the observed triangle (`self`) and what each simulation
/// resamples ([`Draw`](Self::Draw)).
pub(crate) trait NextYear: ReserveFit + Clone + Send + Sync {
    /// What each simulation draws from beyond the fit: the residuals to
    /// resample and the process error.
    type Draw: Send + Sync;

    /// The cells of `year` to append to `segment`, as `(origin, position,
    /// cumulative value)`, each origin's in development order.
    fn year_cells(
        &self,
        draw: &Self::Draw,
        segment: &Segment,
        year: &[YearCells],
        rng: &mut StreamRng,
    ) -> Vec<(usize, usize, f64)>;
}

/// What the ODP bootstrap resamples in each simulation.
pub(crate) struct OdpDraw {
    pool: Vec<f64>,
    process: ProcessDistribution,
}

impl NextYear for OdpBootstrapSegment {
    type Draw = OdpDraw;

    /// The observed latest value plus the increments projected from the
    /// pseudo latest value with the resampled factors, each with its own
    /// process error.
    fn year_cells(
        &self,
        draw: &OdpDraw,
        segment: &Segment,
        year: &[YearCells],
        rng: &mut StreamRng,
    ) -> Vec<(usize, usize, f64)> {
        let cl = &self.chain_ladder;
        let nd = segment.n_dev;
        let sim = Simulation {
            segment,
            latest: &cl.latest_position,
            fitted: &self.fitted,
            pool: &draw.pool,
            scale: self.scale,
            process: draw.process,
        };
        let (pseudo, factors) = sim.resample(rng);
        let mut cells = Vec::new();
        for y in year {
            let o = y.origin;
            let (mut expected, mut value) = (pseudo[o * nd + y.latest], cl.latest[o]);
            for (d, &f) in factors.iter().enumerate().take(y.last).skip(y.latest) {
                value += sim.with_process(expected * (f - 1.0), rng);
                expected *= f;
                if d + 1 >= y.first {
                    cells.push((o, d + 1, value));
                }
            }
        }
        cells
    }
}

/// One segment, prepared: its bootstrap and opening ultimate, what each
/// simulation draws from, its cells and exposure, and the cells of its
/// coming year.
struct Prepared<B: NextYear> {
    fit: OneYearSegment<B>,
    draw: B::Draw,
    segment: Segment,
    exposure: Option<Segment>,
    year: Vec<YearCells>,
}

impl<B: NextYear> Prepared<B> {
    /// `model` fits the bootstrap model to the segment.
    fn new(
        triangle: &Triangle,
        segment: &Segment,
        exposure: Option<&Segment>,
        method: &OneYearMethod,
        opening: Month,
        model: &impl Fn(&Segment) -> Result<(B, B::Draw)>,
    ) -> Result<Self> {
        let (bootstrap, draw) = model(segment)?;
        let year = YearCells::of(
            segment,
            &bootstrap.chain_ladder().latest_position,
            |o, d| triangle.valuation_of(o + segment.origin_offset, d + segment.dev_offset),
        );
        let opening_ultimate = method.ultimate(segment, exposure, opening)?;
        let opening_reserve = opening_ultimate
            .iter()
            .zip(&bootstrap.chain_ladder().latest)
            .map(|(u, l)| u - l)
            .collect();
        Ok(Self {
            fit: OneYearSegment {
                bootstrap,
                opening_ultimate,
                opening_reserve,
            },
            draw,
            segment: segment.clone(),
            exposure: exposure.cloned(),
            year,
        })
    }

    fn hash(&self, hasher: &mut InputHasher) {
        let cl = self.fit.bootstrap.chain_ladder();
        hash_segment(hasher, &self.segment, cl, &self.segment.ages);
        if let Some(exposure) = &self.exposure {
            for o in 0..exposure.n_origins {
                if let Ok((d, v)) = exposure.latest(o) {
                    hasher.i64(d as i64);
                    hasher.f64s(&[v]);
                }
            }
        }
    }
}

/// Re-reserving one segment, once per simulation.
struct Run<'a, B: NextYear> {
    prepared: &'a Prepared<B>,
    method: &'a OneYearMethod,
    /// The valuation a year after the triangle's.
    closing: Month,
    /// The segment's label, to name it in an error; `None` without keys.
    label: Option<String>,
}

impl<B: NextYear> Run<'_, B> {
    fn n_origins(&self) -> usize {
        self.prepared.segment.n_origins
    }

    /// One simulation: fills `cdr` with each origin's claims development
    /// result.
    fn run(&self, rng: &mut StreamRng, cdr: &mut [f64]) -> Result<()> {
        let p = self.prepared;
        let next = p
            .fit
            .bootstrap
            .year_cells(&p.draw, &p.segment, &p.year, rng);
        self.rereserve(next, cdr)
    }

    /// Appends `next` to the observed triangle, refits the method on it and
    /// fills `cdr` with each origin's opening less closing ultimate.
    fn rereserve(&self, next: Vec<(usize, usize, f64)>, cdr: &mut [f64]) -> Result<()> {
        let p = self.prepared;
        let closing = p.segment.with_values(next);
        let ultimate = self
            .method
            .ultimate(&closing, p.exposure.as_ref(), self.closing)?;
        for ((x, u0), u1) in cdr.iter_mut().zip(&p.fit.opening_ultimate).zip(&ultimate) {
            *x = u0 - u1;
            if !x.is_finite() {
                return Err(Error::Bootstrap("re-reserving gave a non-finite ultimate"));
            }
        }
        Ok(())
    }
}

/// The failures of re-reserving: how many, and the one whose message sorts
/// first, so the error reported does not depend on the threads.
#[derive(Default)]
struct Failures {
    count: usize,
    first: Option<(String, Error)>,
}

impl Failures {
    fn record(&mut self, error: Error) {
        self.count += 1;
        let message = error.to_string();
        if self.first.as_ref().is_none_or(|(m, _)| message < *m) {
            self.first = Some((message, error));
        }
    }
}

/// The simulations of a one-year bootstrap: how many, from which seed.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sims {
    pub(crate) n_sims: usize,
    pub(crate) seed: u64,
}

impl Sims {
    fn check(self) -> Result<()> {
        if self.n_sims == 0 {
            return Err(Error::Bootstrap("n_sims must be positive"));
        }
        Ok(())
    }

    /// The one-year view of `column` of a single-segment triangle, the
    /// bootstrap model fitted by `model`; `provenance` names it, given the
    /// hash of the inputs.
    pub(crate) fn one_year<B: NextYear>(
        self,
        triangle: &Triangle,
        column: &str,
        method: &OneYearMethod,
        model: impl Fn(&Segment) -> Result<(B, B::Draw)>,
        provenance: impl FnOnce(InputHasher) -> Provenance,
    ) -> Result<OneYearFit<B>> {
        self.check()?;
        let (opening, closing) = valuations(triangle)?;
        let segment = triangle.segment(column)?;
        let exposure = method.exposure().map(|e| triangle.segment(e)).transpose()?;
        let prepared = Prepared::new(
            triangle,
            &segment,
            exposure.as_ref(),
            method,
            opening,
            &model,
        )?;

        let mut hasher = InputHasher::new();
        hasher.str(column);
        prepared.hash(&mut hasher);
        let run = Run {
            prepared: &prepared,
            method,
            closing,
            label: None,
        };
        let cdr = self.simulate_cdr(
            vec!["origin".into()],
            segment.origins.iter().map(|&p| vec![p.into()]).collect(),
            &[run],
            provenance(hasher),
        )?;
        let OneYearSegment {
            bootstrap,
            opening_ultimate,
            opening_reserve,
        } = prepared.fit;
        Ok(OneYearFit {
            bootstrap,
            opening_ultimate,
            opening_reserve,
            cdr,
        })
    }

    /// The one-year view of `column` in every segment of a triangle, each
    /// with its own bootstrap model fitted by `model`, into one joint
    /// distribution; `provenance` names it, given the hash of the inputs.
    pub(crate) fn one_year_segments<B: NextYear>(
        self,
        triangle: &Triangle,
        column: &str,
        method: &OneYearMethod,
        model: impl Fn(&Segment) -> Result<(B, B::Draw)>,
        provenance: impl FnOnce(InputHasher) -> Provenance,
    ) -> Result<OneYearFits<B>> {
        self.check()?;
        let (opening, closing) = valuations(triangle)?;
        let prepared = match method.exposure() {
            None => fit_each(triangle, column, |s| {
                Prepared::new(triangle, s, None, method, opening, &model)
            })?,
            Some(exposure) => fit_each_with_exposure(triangle, column, exposure, |s, e| {
                Prepared::new(triangle, s, Some(e), method, opening, &model)
            })?,
        };

        let keyed = !prepared.key_names.is_empty();
        let mut hasher = InputHasher::new();
        hasher.str(column);
        let mut dims = prepared.key_names.clone();
        dims.push("origin".into());
        let mut components: Vec<Vec<KeyValue>> = Vec::new();
        let mut runs = Vec::with_capacity(prepared.len());
        for (label, p) in prepared.iter() {
            hasher.str(&label.to_string());
            p.hash(&mut hasher);
            components.extend(origin_keys(label, &p.segment.origins));
            runs.push(Run {
                prepared: p,
                method,
                closing,
                label: keyed.then(|| label.to_string()),
            });
        }
        let cdr = self.simulate_cdr(
            dims,
            components,
            &runs,
            provenance(hasher).param("segments", prepared.len()),
        )?;
        Ok(OneYearFits {
            segments: prepared.map(|p| p.fit.clone()),
            cdr,
        })
    }

    /// Simulates the claims development result of every segment in `runs`,
    /// in turn, into one joint distribution. A failed simulation is
    /// reported after all have run, with the number that failed.
    fn simulate_cdr<B: NextYear>(
        self,
        dims: Vec<String>,
        components: Vec<Vec<KeyValue>>,
        runs: &[Run<'_, B>],
        provenance: Provenance,
    ) -> Result<PredictiveDistribution> {
        let failures = Mutex::new(Failures::default());
        let cdr = PredictiveDistribution::simulate(
            dims,
            components,
            self.n_sims,
            self.seed,
            provenance,
            |rng, row| {
                let mut start = 0;
                for run in runs {
                    let end = start + run.n_origins();
                    if let Err(e) = run.run(rng, &mut row[start..end]) {
                        // Keep the draws finite; the failure is reported.
                        row.fill(0.0);
                        let e = match &run.label {
                            Some(label) => Error::InSegment {
                                label: label.clone(),
                                source: Box::new(e),
                            },
                            None => e,
                        };
                        failures
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .record(e);
                        return;
                    }
                    start = end;
                }
            },
        )?;
        let failures = failures
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match failures.first {
            Some((_, source)) => Err(Error::OneYear {
                failed: failures.count,
                n_sims: self.n_sims,
                source: Box::new(source),
            }),
            None => Ok(cdr),
        }
    }
}

impl OdpBootstrap {
    /// The one-year view of `column` of a single-segment cumulative
    /// triangle: the claims development result of `method` over the twelve
    /// months after the valuation, at any development grain, by
    /// re-reserving on the ODP bootstrap; see the [module
    /// documentation](crate::one_year_bootstrap). Every origin must be
    /// observed from the first age to its latest (it may stop short of the
    /// latest diagonal), and an expected-loss method's exposure column must
    /// have a positive value for every origin.
    pub fn one_year(
        &self,
        triangle: &Triangle,
        column: &str,
        method: &OneYearMethod,
    ) -> Result<OneYearFit> {
        self.sims().one_year(
            triangle,
            column,
            method,
            |s| self.model(s),
            |hasher| self.one_year_provenance(column, method, hasher),
        )
    }

    /// The one-year view of `column` in every segment of a cumulative
    /// triangle, each bootstrapped with its own residuals and scale and
    /// refitted with its own exposure, into one joint distribution of the
    /// claims development result; see [`OneYearFits`]. A failure names its
    /// segment.
    pub fn one_year_segments(
        &self,
        triangle: &Triangle,
        column: &str,
        method: &OneYearMethod,
    ) -> Result<OneYearFits> {
        self.sims().one_year_segments(
            triangle,
            column,
            method,
            |s| self.model(s),
            |hasher| self.one_year_provenance(column, method, hasher),
        )
    }

    fn sims(&self) -> Sims {
        Sims {
            n_sims: self.n_sims,
            seed: self.seed,
        }
    }

    /// The bootstrap of one segment and the residuals it resamples.
    fn model(&self, segment: &Segment) -> Result<(OdpBootstrapSegment, OdpDraw)> {
        let (fit, pool) = prepare(segment, &segment.ages)?;
        Ok((
            fit,
            OdpDraw {
                pool,
                process: self.process,
            },
        ))
    }

    fn one_year_provenance(
        &self,
        column: &str,
        method: &OneYearMethod,
        hasher: InputHasher,
    ) -> Provenance {
        Provenance::new("odp_bootstrap_one_year")
            .param("n_sims", self.n_sims)
            .param("process", format!("{:?}", self.process))
            .param("column", column)
            .param("method", format!("{method:?}"))
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(hasher.finish())
    }
}

/// The triangle's valuation and the one twelve months later, which the
/// refit at the end of the year trends to.
fn valuations(triangle: &Triangle) -> Result<(Month, Month)> {
    let opening = triangle.valuation();
    Ok((opening, opening.add_months(12)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::development::{Average, Development};
    use crate::triangle::tests::{RAA, genins, raa};
    use crate::triangle::{DevelopmentColumn, Long};
    use crate::{Grain, ProcessDistribution};
    use prospicio_prob::Distribution;

    /// A cumulative annual triangle from rows of `paid` values and one
    /// `premium` per origin, repeated at every observed age. Origins start in
    /// `first_year`.
    fn with_premium(first_year: i32, rows: &[&[f64]], premium: &[f64]) -> Triangle {
        let (mut origin, mut ages, mut paid, mut prem) = (vec![], vec![], vec![], vec![]);
        for (k, row) in rows.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                origin.push(Month::january(first_year + k as i32));
                ages.push(12 * (d as u32 + 1));
                paid.push(v);
                prem.push(premium[k]);
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

    /// Four origins exactly on the pattern 1 : 2 : 3 : 3.75 (factors 2, 1.5
    /// and 1.25), every value exact in binary.
    fn exact() -> Triangle {
        with_premium(
            2020,
            &[
                &[4.0, 8.0, 12.0, 15.0],
                &[8.0, 16.0, 24.0],
                &[12.0, 24.0],
                &[16.0],
            ],
            &[20.0, 40.0, 50.0, 80.0],
        )
    }

    /// RAA with a premium of 20,000 per origin.
    fn raa_premium() -> Triangle {
        with_premium(1981, &RAA, &[20_000.0; 10])
    }

    fn boot(n_sims: usize, seed: u64) -> OdpBootstrap {
        OdpBootstrap {
            n_sims,
            seed,
            process: ProcessDistribution::Gamma,
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

    #[test]
    fn exact_pattern_has_zero_scale_and_cdr() {
        let tailed = OneYearMethod::ChainLadder(ChainLadder {
            tail: 1.1.into(),
            ..Default::default()
        });
        for method in [chain_ladder(), tailed] {
            let fit = boot(200, 1).one_year(&exact(), "paid", &method).unwrap();
            assert_eq!(fit.bootstrap.scale, 0.0);
            assert!(
                fit.cdr.draw_matrix().iter().all(|&x| x == 0.0),
                "{method:?}"
            );
        }
    }

    #[test]
    fn exact_pattern_bornhuetter_ferguson_moves_by_hand() {
        // With no noise the next diagonal is `latest * f`, so the closing
        // BF ultimate is `latest * f + (1 - f / cdf) * a * E`, and
        // CDR = (f - 1) * (a * E / cdf - latest), the same in every
        // simulation. 2021 is at 36 months: f = 1.25, cdf = 1.25.
        let bf = BornhuetterFerguson {
            apriori: 0.5,
            ..Default::default()
        };
        let method = OneYearMethod::BornhuetterFerguson(bf, "premium".into());
        let fit = boot(50, 2).one_year(&exact(), "paid", &method).unwrap();
        let (latest, cdf, f) = (
            [15.0, 24.0, 24.0, 16.0],
            [1.0, 1.25, 1.875, 3.75],
            [1.0, 1.25, 1.5, 2.0],
        );
        let exposure = [20.0, 40.0, 50.0, 80.0];
        for o in 0..4 {
            let want = (f[o] - 1.0) * (0.5 * exposure[o] / cdf[o] - latest[o]);
            for x in column(&fit.cdr, o) {
                assert!((x - want).abs() < 1e-9, "origin {o}: {x} vs {want}");
            }
        }
        let opening = bf.fit(&exact(), "paid", "premium").unwrap();
        assert_eq!(fit.opening_ultimate, opening.ultimate);
        assert_eq!(fit.opening_reserve, opening.reserves());
    }

    #[test]
    fn reproducible_by_seed_and_thread_count() {
        let tri = raa();
        let a = boot(300, 7)
            .one_year(&tri, "values", &chain_ladder())
            .unwrap();
        let b = boot(300, 7)
            .one_year(&tri, "values", &chain_ladder())
            .unwrap();
        assert_eq!(a.cdr.draw_matrix(), b.cdr.draw_matrix());
        let c = boot(300, 8)
            .one_year(&tri, "values", &chain_ladder())
            .unwrap();
        assert_ne!(a.cdr.draw_matrix(), c.cdr.draw_matrix());
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| boot(300, 7).one_year(&tri, "values", &chain_ladder()))
            .unwrap();
        assert_eq!(a.cdr.draw_matrix(), one_thread.cdr.draw_matrix());
        assert_eq!(a.cdr.provenance().model, "odp_bootstrap_one_year");
    }

    #[test]
    fn chain_ladder_cdr_mean_is_near_zero() {
        // The CDR is centred on zero up to the bootstrap's factor bias (on
        // RAA about -8% of a standard deviation with 20,000 simulations).
        let fit = boot(4_000, 11)
            .one_year(&raa(), "values", &chain_ladder())
            .unwrap();
        let (mean, sd) = (fit.cdr.mean(), fit.cdr.std_dev());
        assert!(mean.abs() < 0.15 * sd, "mean {mean}, sd {sd}");
        let total_reserve: f64 = fit.opening_reserve.iter().sum();
        assert!((total_reserve - 52_135.228).abs() < 0.01);
    }

    #[test]
    fn an_origin_at_the_last_age_gets_no_new_cell() {
        // 1981 is at the last age: it gets no new cell, so with no tail or
        // a constant one its ultimate, and the CDR, do not move. 1982 gets
        // its last cell and does move.
        let constant_tail = OneYearMethod::ChainLadder(ChainLadder {
            tail: 1.05.into(),
            ..Default::default()
        });
        let bf = OneYearMethod::BornhuetterFerguson(
            BornhuetterFerguson {
                apriori: 0.8,
                ..Default::default()
            },
            "premium".into(),
        );
        for method in [chain_ladder(), constant_tail, bf] {
            let fit = boot(200, 3)
                .one_year(&raa_premium(), "paid", &method)
                .unwrap();
            assert!(column(&fit.cdr, 0).iter().all(|&x| x == 0.0), "{method:?}");
            assert!(column(&fit.cdr, 1).iter().any(|&x| x != 0.0), "{method:?}");
        }
    }

    #[test]
    fn expected_loss_methods_open_on_their_own_ultimate() {
        let tri = raa_premium();
        let el = ExpectedLoss {
            apriori: 0.8,
            ..Default::default()
        };
        let fit = boot(100, 4)
            .one_year(
                &tri,
                "paid",
                &OneYearMethod::ExpectedLoss(el, "premium".into()),
            )
            .unwrap();
        // The expected loss ratio ultimate ignores the losses: no CDR.
        assert_eq!(fit.opening_ultimate, vec![16_000.0; 10]);
        assert!(fit.cdr.draw_matrix().iter().all(|&x| x == 0.0));

        let bk = Benktander {
            apriori: 0.8,
            n_iters: 2,
            ..Default::default()
        };
        let fit = boot(100, 4)
            .one_year(
                &tri,
                "paid",
                &OneYearMethod::Benktander(bk, "premium".into()),
            )
            .unwrap();
        assert_eq!(
            fit.opening_ultimate,
            bk.fit(&tri, "paid", "premium").unwrap().ultimate
        );
        assert!(fit.cdr.std_dev() > 0.0);

        let cc = CapeCod {
            trend: 0.03,
            decay: 0.8,
            ..Default::default()
        };
        let fit = boot(100, 4)
            .one_year(&tri, "paid", &OneYearMethod::CapeCod(cc, "premium".into()))
            .unwrap();
        assert_eq!(
            fit.opening_ultimate,
            cc.fit(&tri, "paid", "premium")
                .unwrap()
                .expected_loss
                .ultimate
        );
        assert!(fit.cdr.std_dev() > 0.0);
    }

    #[test]
    fn errors() {
        let bf = |column: &str| {
            OneYearMethod::BornhuetterFerguson(BornhuetterFerguson::default(), column.into())
        };
        let b = boot(10, 0);
        assert_eq!(
            b.one_year(&raa(), "values", &bf("premium")).unwrap_err(),
            Error::UnknownColumn("premium".into())
        );
        // 1990 has no premium.
        let mut premium = [20_000.0; 10];
        premium[9] = f64::NAN;
        let tri = with_premium(1981, &RAA, &premium);
        assert_eq!(
            b.one_year(&tri, "paid", &bf("premium")).unwrap_err(),
            Error::InvalidExposure {
                column: "premium".into(),
                origin: "1990".into()
            }
        );
        let none = boot(0, 0);
        assert_eq!(
            none.one_year(&raa(), "values", &chain_ladder())
                .unwrap_err(),
            Error::Bootstrap("n_sims must be positive")
        );
        assert!(matches!(
            none.one_year_segments(&raa(), "values", &chain_ladder()),
            Err(Error::Bootstrap(_))
        ));
    }

    #[test]
    fn a_failed_refit_is_counted_and_reported() {
        // The newest origin is at zero, so its next cell is zero too, and a
        // simple average gives that link an infinite weight in every
        // simulation's refit (but not in the opening fit, where the origin
        // has no link).
        let tri = with_premium(2020, &[&[4.0, 8.0, 12.0], &[8.0, 15.0], &[0.0]], &[1.0; 3]);
        let simple = OneYearMethod::ChainLadder(ChainLadder {
            development: Development {
                average: Average::Simple,
                ..Default::default()
            },
            ..Default::default()
        });
        let err = boot(20, 0).one_year(&tri, "paid", &simple).unwrap_err();
        assert_eq!(
            err,
            Error::OneYear {
                failed: 20,
                n_sims: 20,
                source: Box::new(Error::Factor {
                    age: 0,
                    reason: "a zero value gets an infinite weight"
                }),
            }
        );
        assert_eq!(
            err.to_string(),
            "one-year bootstrap: re-reserving failed in 20 of 20 simulations, for example: \
             factor from development index 0: a zero value gets an infinite weight"
        );
    }

    /// The cells of the coming year of `column`'s only segment, as
    /// `(origin, latest, first, last)` positions.
    fn year_of(tri: &Triangle, column: &str) -> Vec<(usize, usize, usize, usize)> {
        let segment = tri.segment(column).unwrap();
        let latest: Vec<usize> = (0..segment.n_origins)
            .map(|o| segment.latest(o).unwrap().0)
            .collect();
        YearCells::of(&segment, &latest, |o, d| {
            tri.valuation_of(o + segment.origin_offset, d + segment.dev_offset)
        })
        .iter()
        .map(|y| (y.origin, y.latest, y.first, y.last))
        .collect()
    }

    /// Annual origins from `first_year` with cumulative `rows` at annual
    /// ages, split into a quarterly development grain: each year's
    /// increment in four equal quarters, so the value at 12 k months is
    /// unchanged. `premium` per origin as in `with_premium`.
    fn quarterly(first_year: i32, rows: &[&[f64]], premium: &[f64]) -> Triangle {
        let (mut origin, mut ages, mut paid, mut prem) = (vec![], vec![], vec![], vec![]);
        for (k, row) in rows.iter().enumerate() {
            let mut previous = 0.0;
            for (d, &v) in row.iter().enumerate() {
                for q in 1..=4 {
                    origin.push(Month::january(first_year + k as i32));
                    ages.push(12 * d as u32 + 3 * q);
                    paid.push(previous + (v - previous) * f64::from(q) / 4.0);
                    prem.push(premium[k]);
                }
                previous = v;
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Quarter,
            cumulative: true,
        })
        .unwrap()
    }

    /// `exact()` with 2021 stopping at 24 months, a year short of the
    /// latest diagonal (December 2023).
    fn lagging() -> Triangle {
        with_premium(
            2020,
            &[
                &[4.0, 8.0, 12.0, 15.0],
                &[8.0, 16.0],
                &[12.0, 24.0],
                &[16.0],
            ],
            &[20.0, 40.0, 50.0, 80.0],
        )
    }

    #[test]
    fn the_coming_year_is_the_cells_valued_in_the_next_twelve_months() {
        // Annual: the next diagonal, one cell per origin short of the last
        // age.
        let annual: Vec<_> = (1..10).map(|o| (o, 9 - o, 10 - o, 10 - o)).collect();
        assert_eq!(year_of(&raa(), "values"), annual);

        // Quarterly development of annual origins: four cells, from the
        // quarter after the latest to the one twelve months on (ages are
        // positions times 3 months, plus 3).
        let split = quarterly(
            2020,
            &[
                &[4.0, 8.0, 12.0, 15.0],
                &[8.0, 16.0, 24.0],
                &[12.0, 24.0],
                &[16.0],
            ],
            &[1.0; 4],
        );
        assert_eq!(
            year_of(&split, "paid"),
            [(1, 11, 12, 15), (2, 7, 8, 11), (3, 3, 4, 7)]
        );

        // Quarterly origins: fewer cells for those that reach the last age
        // (12 months) within the year. Valuation December 2022.
        let origin: Vec<Month> = [1, 1, 1, 1, 4, 4, 4, 7, 7, 10]
            .iter()
            .map(|&m| Month::new(2022, m).unwrap())
            .collect();
        let by_quarter = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[3, 6, 9, 12, 3, 6, 9, 3, 6, 3]),
            values: &[("paid", &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0])],
            origin_grain: Grain::Quarter,
            development_grain: Grain::Quarter,
            cumulative: true,
        })
        .unwrap();
        assert_eq!(
            year_of(&by_quarter, "paid"),
            [(1, 2, 3, 3), (2, 1, 2, 3), (3, 0, 1, 3)]
        );

        // A lagging origin develops from its own latest cell: 2021, at 24
        // months (December 2022), steps through 36 (December 2023, the
        // valuation) to 48, the only cell in the coming year.
        assert_eq!(
            year_of(&lagging(), "paid"),
            [(1, 1, 3, 3), (2, 1, 2, 2), (3, 0, 1, 1)]
        );
        // Quarterly, 2021 stopping a quarter short (33 months, September
        // 2023): 36 is a step, 39 to 48 are the year's four cells.
        let short = quarterly(
            2020,
            &[
                &[4.0, 8.0, 12.0, 15.0],
                &[8.0, 16.0, 24.0],
                &[12.0, 24.0],
                &[16.0],
            ],
            &[1.0; 4],
        );
        let segment = short.segment("paid").unwrap();
        let cells: Vec<(usize, usize, f64)> = (0..segment.n_origins)
            .flat_map(|o| (0..segment.n_dev).map(move |d| (o, d)))
            .filter(|&(o, d)| o != 1 || d < 11)
            .filter_map(|(o, d)| Some((o, d, segment.get(o, d)?)))
            .collect();
        let origin: Vec<Month> = cells
            .iter()
            .map(|&(o, _, _)| Month::january(2020 + o as i32))
            .collect();
        let ages: Vec<u32> = cells.iter().map(|&(_, d, _)| 3 * d as u32 + 3).collect();
        let values: Vec<f64> = cells.iter().map(|c| c.2).collect();
        let short = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &values)],
            origin_grain: Grain::Year,
            development_grain: Grain::Quarter,
            cumulative: true,
        })
        .unwrap();
        assert_eq!(
            year_of(&short, "paid"),
            [(1, 10, 12, 15), (2, 7, 8, 11), (3, 3, 4, 7)]
        );
    }

    #[test]
    fn a_lagging_origin_moves_by_hand() {
        // On the exact pattern the year's cells are `latest` times the
        // factors, so as in `exact_pattern_bornhuetter_ferguson_moves_by_hand`
        // CDR = (F - 1) * (a * E / cdf - latest), with F now the growth over
        // the year: for 2021, lagging at 24 months, its two factors to 48
        // (F = cdf = 1.875), not the one to 36 alone.
        let bf = BornhuetterFerguson {
            apriori: 0.5,
            ..Default::default()
        };
        let method = OneYearMethod::BornhuetterFerguson(bf, "premium".into());
        let fit = boot(50, 2).one_year(&lagging(), "paid", &method).unwrap();
        assert_eq!(fit.bootstrap.scale, 0.0);
        let (latest, cdf, growth) = (
            [15.0, 16.0, 24.0, 16.0],
            [1.0, 1.875, 1.875, 3.75],
            [1.0, 1.875, 1.5, 2.0],
        );
        let exposure = [20.0, 40.0, 50.0, 80.0];
        for o in 0..4 {
            let want = (growth[o] - 1.0) * (0.5 * exposure[o] / cdf[o] - latest[o]);
            for x in column(&fit.cdr, o) {
                assert!((x - want).abs() < 1e-9, "origin {o}: {x} vs {want}");
            }
        }
        // 2021's CDR is -14 + 20 * (1 - 1 / 1.875).
        assert!((column(&fit.cdr, 1)[0] + 14.0 - 20.0 * (1.0 - 1.0 / 1.875)).abs() < 1e-9);

        // On RAA, 1985 cut back a year to 60 months reveals two years of
        // development in the coming one: a wider CDR than on the full
        // triangle, for both process models (measured 1.98 times for the
        // ODP, 1.38 for Mack's, at these seeds).
        let cut: Vec<&[f64]> = RAA
            .iter()
            .enumerate()
            .map(|(k, row)| if k == 4 { &row[..5] } else { *row })
            .collect();
        let cut = with_premium(1981, &cut, &[1.0; 10]);
        let full = raa_premium();
        let sd = |tri: &Triangle, mack: bool| {
            let cdr = if mack {
                crate::MackBootstrap {
                    n_sims: 2_000,
                    seed: 4,
                    ..Default::default()
                }
                .one_year(tri, "paid", &chain_ladder())
                .unwrap()
                .cdr
            } else {
                boot(2_000, 4)
                    .one_year(tri, "paid", &chain_ladder())
                    .unwrap()
                    .cdr
            };
            let x = column(&cdr, 4);
            let m = x.iter().sum::<f64>() / x.len() as f64;
            (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
        };
        for mack in [false, true] {
            let (lag, on) = (sd(&cut, mack), sd(&full, mack));
            assert!(
                lag > 1.2 * on,
                "mack {mack}: lagging {lag}, on the diagonal {on}"
            );
        }
    }

    #[test]
    fn a_lagging_origin_appends_only_the_year() {
        // RAA with 1985 cut back to 60 months (December 1989, a year short
        // of the valuation): its step to 72 months, valued December 1990,
        // is drawn but not appended, so its only cell is 84 months
        // (position 6). Every other origin appends its next diagonal cell.
        let cut: Vec<&[f64]> = RAA
            .iter()
            .enumerate()
            .map(|(k, row)| if k == 4 { &row[..5] } else { *row })
            .collect();
        let tri = with_premium(1981, &cut, &[1.0; 10]);
        let segment = tri.segment("paid").unwrap();
        let year: Vec<YearCells> = year_of(&tri, "paid")
            .into_iter()
            .map(|(origin, latest, first, last)| YearCells {
                origin,
                latest,
                first,
                last,
            })
            .collect();
        let want: Vec<(usize, usize)> = (1..10)
            .map(|o| (o, if o == 4 { 6 } else { 10 - o }))
            .collect();
        let positions = |cells: Vec<(usize, usize, f64)>| -> Vec<(usize, usize)> {
            cells.iter().map(|&(o, d, _)| (o, d)).collect()
        };
        let (odp, draw) = boot(1, 0).model(&segment).unwrap();
        let mut rng = StreamRng::new(0, 0);
        let cells = odp.year_cells(&draw, &segment, &year, &mut rng);
        assert_eq!(positions(cells), want, "ODP");
        let (mack, draw) = crate::MackBootstrap::default().model(&segment).unwrap();
        let cells = mack.year_cells(&draw, &segment, &year, &mut rng);
        assert_eq!(positions(cells), want, "Mack");
    }

    #[test]
    fn quarterly_exact_pattern_has_zero_cdr() {
        // `exact()` split into quarters is still exactly on a pattern (every
        // origin a multiple of the first), so four quarterly cells a year
        // move nothing.
        let split = quarterly(
            2020,
            &[
                &[4.0, 8.0, 12.0, 15.0],
                &[8.0, 16.0, 24.0],
                &[12.0, 24.0],
                &[16.0],
            ],
            &[20.0, 40.0, 50.0, 80.0],
        );
        let tailed = OneYearMethod::ChainLadder(ChainLadder {
            tail: 1.1.into(),
            ..Default::default()
        });
        for method in [chain_ladder(), tailed] {
            let fit = boot(200, 1).one_year(&split, "paid", &method).unwrap();
            assert_eq!(fit.bootstrap.scale, 0.0);
            assert!(
                fit.cdr.draw_matrix().iter().all(|&x| x == 0.0),
                "{method:?}"
            );
            // The opening reserve is the annual triangle's: the quarterly
            // volume-weighted factors of a year telescope to the annual one.
            let annual = boot(1, 1).one_year(&exact(), "paid", &method).unwrap();
            for (q, a) in fit.opening_reserve.iter().zip(&annual.opening_reserve) {
                assert!((q - a).abs() < 1e-9, "{q} vs {a}");
            }
        }
    }

    #[test]
    fn quarterly_mack_gamma_survives_a_draw_near_zero() {
        // RAA split into quarters chains four Gamma draws a year, each from
        // the one before. With seed 3 one of them comes out near 1e-200, and
        // the next draw's shape `m^2 / variance` underflowed to zero and
        // panicked (3 of seeds 0-9 did, the annual RAA none).
        let split = quarterly(1981, &RAA, &[1.0; 10]);
        for process in [crate::MackProcess::Gamma, crate::MackProcess::Lognormal] {
            let fit = crate::MackBootstrap {
                n_sims: 2_000,
                seed: 3,
                process,
                ..Default::default()
            }
            .one_year(&split, "paid", &chain_ladder())
            .unwrap();
            assert!(fit.cdr.draw_matrix().iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn zero_sigma_links_give_mack_no_residuals() {
        // A split's first-year quarterly link ratios are 2, 3/2 and 4/3 for
        // every origin, so those three sigmas are zero and their residuals
        // 0 / 0. Left out, the pool keeps the annual one's mean square of 1
        // (each volume-weighted factor's squared residuals sum to its
        // number of link ratios); as zeros it was 0.854 (176 of 206).
        let split = quarterly(1981, &RAA, &[1.0; 10]);
        let fit = crate::MackBootstrap {
            n_sims: 1,
            ..Default::default()
        }
        .one_year(&split, "paid", &chain_ladder())
        .unwrap();
        let sigma = &fit.bootstrap.mack.chain_ladder.development.sigma;
        assert!(sigma[..3].iter().all(|&s| s == 0.0), "{sigma:?}");
        assert!(sigma[3..].iter().all(|&s| s > 0.0), "{sigma:?}");
        let nd = 40;
        let r = &fit.bootstrap.residuals;
        assert!((0..10).all(|o| (0..3).all(|k| r[o * nd + k].is_nan())));
        let pool: Vec<f64> = r.iter().copied().filter(|x| !x.is_nan()).collect();
        assert_eq!(pool.len(), 176);
        let mean_square = pool.iter().map(|x| x * x).sum::<f64>() / pool.len() as f64;
        assert!((mean_square - 1.0).abs() < 1e-12, "{mean_square}");
    }

    #[test]
    fn quarterly_exact_pattern_has_zero_mack_cdr() {
        // Mack's model on an exact pattern has every sigma zero, so no
        // parameter or process error, at either grain. (A fifth origin
        // gives the last factor two link ratios: a lone one's sigma cannot
        // be interpolated from sigmas that are all zero.)
        let rows: [&[f64]; 5] = [
            &[4.0, 8.0, 12.0, 15.0],
            &[8.0, 16.0, 24.0, 30.0],
            &[12.0, 24.0, 36.0],
            &[16.0, 32.0],
            &[20.0],
        ];
        let split = quarterly(2019, &rows, &[1.0; 5]);
        let annual = with_premium(2019, &rows, &[1.0; 5]);
        for tri in [annual, split] {
            let fit = crate::MackBootstrap {
                n_sims: 50,
                ..Default::default()
            }
            .one_year(&tri, "paid", &chain_ladder())
            .unwrap();
            let sigma = &fit.bootstrap.mack.chain_ladder.development.sigma;
            assert!(sigma.iter().all(|&s| s == 0.0), "{sigma:?}");
            assert!(fit.cdr.draw_matrix().iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn segments_may_end_on_different_diagonals() {
        // Home's latest diagonal is a year before Auto's, and each
        // simulates the twelve months after its own.
        let origin = [
            2020, 2020, 2020, 2021, 2021, 2022, 2019, 2019, 2019, 2020, 2020, 2021,
        ]
        .map(Month::january);
        let ages = [12, 24, 36, 12, 24, 12];
        let paid = [100.0, 150.0, 165.0, 110.0, 170.0, 120.0];
        let staggered = Triangle::from_long(&Long {
            keys: &[("lob", &[["Auto"; 6], ["Home"; 6]].concat())],
            origin: &origin,
            development: DevelopmentColumn::Age(&[ages, ages].concat()),
            values: &[("paid", &[paid, paid].concat())],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let fits = boot(10, 0)
            .one_year_segments(&staggered, "paid", &chain_ladder())
            .unwrap();
        assert_eq!(fits.cdr.n_components(), 6);
    }

    #[test]
    fn one_cell_left_is_the_lifetime_run_off() {
        // 1982 has one cell left: its one-year CDR is the opening reserve
        // less that cell's increment, which is drawn as the lifetime
        // bootstrap draws 1982's reserve. The standard deviations agree
        // within Monte Carlo error (sd * sqrt((kurtosis - 1) / (4 n))).
        let n = 20_000;
        let one = boot(n, 21)
            .one_year(&raa(), "values", &chain_ladder())
            .unwrap();
        let lifetime = boot(n, 22).fit(&raa(), "values").unwrap();
        let sd = |x: &[f64]| {
            let m = x.iter().sum::<f64>() / x.len() as f64;
            (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
        };
        let (a, b) = (sd(&column(&one.cdr, 1)), sd(&column(&lifetime.reserves, 1)));
        // Five standard errors of each, with a kurtosis of about 4.
        let tol = 5.0 * (a + b) * (3.0 / (4.0 * n as f64)).sqrt();
        assert!((a - b).abs() < tol, "one-year {a}, lifetime {b}");
    }

    /// England, Verrall and Wüthrich (2019), Appendix 1, steps 6 and 7(a) to
    /// (d): Mack's model bootstrapped from the scaled bias-adjusted residuals
    /// of the link ratios, with a Gamma next cumulative value of mean
    /// `f* C` and variance `sigma^2 C` from the observed latest `C`, fed
    /// through the same re-reserving as the ODP bootstrap. Their Section 4
    /// shows that this reproduces Merz–Wüthrich's one-year standard error
    /// on Taylor–Ashe (GenIns), which `MackFit::claims_development_result`
    /// equals R ChainLadder's `CDR(1)S.E.` for
    /// (`validation/tests/reserving_cdr.rs`). So this checks the
    /// re-reserving, independently of the ODP's process model, within five
    /// Monte Carlo standard errors. (On RAA the first pseudo factor's
    /// standard deviation is a third of the factor, so a Gamma mean can be
    /// negative.)
    #[test]
    fn mack_bootstrap_rereserving_reproduces_merz_wuthrich() {
        let tri = genins();
        let mack = crate::Mack::default().fit(&tri, "values").unwrap();
        let mw = mack.claims_development_result().unwrap();
        let (f, sigma) = (
            &mack.chain_ladder.development.ldf,
            &mack.chain_ladder.development.sigma,
        );

        let segment = tri.segment("values").unwrap();
        let (no, nd) = (segment.n_origins, segment.n_dev);
        let method = chain_ladder();
        let (opening, closing) = valuations(&tri).unwrap();
        let odp = boot(1, 0);
        let prepared =
            Prepared::new(&tri, &segment, None, &method, opening, &|s| odp.model(s)).unwrap();
        let run = Run {
            prepared: &prepared,
            method: &method,
            closing,
            label: None,
        };

        // The link pairs (C_k, C_k+1) behind each factor, and the scaled
        // bias-adjusted residuals sqrt(n / (n - 1)) sqrt(C) (F - f) / sigma
        // of the factors estimated from two or more.
        let pairs: Vec<Vec<f64>> = (0..nd - 1)
            .map(|k| {
                (0..no)
                    .filter_map(|o| segment.get(o, k + 1).and(segment.get(o, k)))
                    .collect()
            })
            .collect();
        let mut pool = Vec::new();
        for k in 0..nd - 1 {
            let n = pairs[k].len() as f64;
            if n < 2.0 {
                continue;
            }
            for o in (0..no).filter(|&o| segment.get(o, k + 1).is_some()) {
                let (c, c1) = (segment.get(o, k).unwrap(), segment.get(o, k + 1).unwrap());
                pool.push((n / (n - 1.0)).sqrt() * c.sqrt() * (c1 / c - f[k]) / sigma[k]);
            }
        }
        let cl = &prepared.fit.bootstrap.chain_ladder;

        let n_sims = 20_000;
        let draws = PredictiveDistribution::simulate(
            vec!["origin".into()],
            segment.origins.iter().map(|&p| vec![p.into()]).collect(),
            n_sims,
            31,
            Provenance::new("mack_bootstrap_one_year"),
            |rng, row| {
                // Pseudo-ratios f + r sigma / sqrt(C) and their volume-weighted
                // averages.
                let factors: Vec<f64> = (0..nd - 1)
                    .map(|k| {
                        let (num, den) = pairs[k].iter().fold((0.0, 0.0), |(num, den), &c| {
                            let r = pool[((rng.next_open01() * pool.len() as f64) as usize)
                                .min(pool.len() - 1)];
                            (num + c * (f[k] + r * sigma[k] / c.sqrt()), den + c)
                        });
                        num / den
                    })
                    .collect();
                let next = cl
                    .latest_position
                    .iter()
                    .zip(&cl.latest)
                    .enumerate()
                    .filter(|&(_, (&d, _))| d + 1 < nd)
                    .map(|(o, (&d, &c))| {
                        let (mean, variance) = (factors[d] * c, sigma[d].powi(2) * c);
                        let gamma =
                            prospicio_prob::Gamma::new(mean * mean / variance, variance / mean)
                                .unwrap();
                        (o, d + 1, gamma.sample(rng, 1)[0])
                    })
                    .collect();
                run.rereserve(next, row).unwrap();
            },
        )
        .unwrap();

        let sd_and_error = |x: &[f64]| {
            let n = x.len() as f64;
            let m = x.iter().sum::<f64>() / n;
            let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
            let m4 = x.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n;
            let sd = m2.sqrt();
            (sd, sd * ((m4 / (m2 * m2) - 1.0) / (4.0 * n)).sqrt())
        };
        let mut checks: Vec<(String, Vec<f64>, f64)> = (1..no)
            .map(|o| {
                (
                    segment.origins[o].to_string(),
                    column(&draws, o),
                    mw.one_year_standard_error[o],
                )
            })
            .collect();
        let total = draws
            .draw_matrix()
            .chunks(no)
            .map(|r| r.iter().sum())
            .collect();
        checks.push(("total".into(), total, mw.total_one_year_standard_error));
        for (origin, x, want) in checks {
            let (sd, error) = sd_and_error(&x);
            assert!(
                (sd - want).abs() < 5.0 * error,
                "{origin}: {sd} against Merz-Wuthrich {want} (5 SE {})",
                5.0 * error
            );
        }

        // `MackBootstrap` is this harness: the same draws, bit for bit.
        // Simulation `i` uses stream `i`, so its first few hundred are the
        // harness's first few hundred.
        let few = 300;
        let built = crate::MackBootstrap {
            n_sims: few,
            seed: 31,
            ..Default::default()
        }
        .one_year(&tri, "values", &method)
        .unwrap();
        assert_eq!(built.cdr.draw_matrix(), &draws.draw_matrix()[..few * no]);
        assert_eq!(built.cdr.provenance().model, "mack_bootstrap_one_year");
    }

    #[test]
    fn segments_share_one_joint_distribution() {
        let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
        let ages = [12, 24, 36, 12, 24, 12];
        let paid = [100.0, 150.0, 165.0, 110.0, 170.0, 120.0];
        let tri = Triangle::from_long(&Long {
            keys: &[("lob", &[["Auto"; 6], ["Home"; 6]].concat())],
            origin: &[origin, origin].concat(),
            development: DevelopmentColumn::Age(&[ages, ages].concat()),
            values: &[("paid", &[paid, paid.map(|v| v * 3.0)].concat())],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let fits = boot(400, 5)
            .one_year_segments(&tri, "paid", &chain_ladder())
            .unwrap();
        assert_eq!(fits.cdr.dims(), ["lob", "origin"]);
        assert_eq!(fits.cdr.n_components(), 6);
        let long = fits.to_long();
        let names: Vec<&str> = long.values.iter().map(|(n, _)| n.as_str()).collect();
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
        assert_eq!(long.column("opening_reserve").unwrap()[0], 0.0);
        assert_eq!(long.column("cdr_std_dev").unwrap()[0], 0.0);
        let totals = fits.totals();
        assert_eq!(totals.column("scale").unwrap().len(), 2);
        // Home is Auto scaled by 3: its scale is three times Auto's.
        let scale = totals.column("scale").unwrap();
        assert!((scale[1] / scale[0] - 3.0).abs() < 1e-9);
        let home = fits.segment(&[("lob", "Home")]).unwrap();
        assert_eq!(home.cdr.n_components(), 3);
        assert_eq!(home.cdr.dims(), ["lob", "origin"]);
        assert_eq!(
            home.totals().column("cdr_std_dev").unwrap()[0],
            totals.column("cdr_std_dev").unwrap()[1]
        );
    }
}
