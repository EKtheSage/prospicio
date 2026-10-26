//! A latest diagonal on its own: one value per origin at its own age, as a
//! bordereau holds it, projected with a selected pattern from elsewhere
//! (`reserving-v02.md`, decisions 15, 16 and 18).
//!
//! [`LatestDiagonal`] places one observed cell per origin on a grid and
//! runs the triangle methods on it through the fully selected path of
//! decision 15, so the chain ladder and the expected-loss methods give the
//! same fits as on any triangle. What it adds is the grid: its
//! ages are the diagonal's and the pattern's together, at the coarsest
//! development grain they share, so an age off the pattern's grid (18
//! months against an annual pattern) is read by the pattern's log-linear
//! rule between its ages. An origin older than the pattern's end is an
//! error unless [`PastPatternEnd`] says how to develop it (decision 18).

use std::collections::BTreeMap;

use prospicio_core::{Grain, Lag, Month, Period};

use crate::expected_loss::origin_exposure;
use crate::pattern::{check_factors, grid, on_ages, own_cdf, span_end, tail_on_own_ages};
use crate::segments::{collect_labelled, find_segment, labelled};
use crate::triangle::Segment;
use crate::{
    Benktander, BornhuetterFerguson, CapeCod, CapeCodFit, ChainLadder, ChainLadderFit, Development,
    Error, ExpectedLoss, ExpectedLossFit, Label, PatternKind, PatternWeight, Result, SegmentFits,
    SelectedPattern, Tail, TailConstant, Triangle,
};

/// The exposure column named in errors.
const EXPOSURE: &str = "exposure";

/// One latest value per origin at its own age, with an optional exposure
/// (premium) per origin, as a cedant's bordereau holds it. The chain ladder and the
/// expected-loss methods project it with a fully selected pattern through
/// their `fit_diagonal`; the methods that need link ratios (Mack, the
/// bootstraps) have none to work with.
///
/// ```
/// use prospicio_reserving::{
///     BornhuetterFerguson, ChainLadder, Development, Grain, LatestDiagonal, Month, PatternKind,
///     SelectedPattern,
/// };
///
/// // Three accident years at 31 December 2024, and a pattern from a prior
/// // study as factors to ultimate.
/// let origins = [2022, 2023, 2024].map(Month::january);
/// let diagonal = LatestDiagonal::at_valuation(
///     Grain::Year,
///     &origins,
///     Month::new(2024, 12).unwrap(),
///     &[900.0, 600.0, 150.0],
/// )
/// .unwrap()
/// .with_exposure(&[1000.0, 1000.0, 1000.0])
/// .unwrap();
/// assert_eq!(diagonal.ages(), [36, 24, 12]);
/// let pattern = SelectedPattern::new(
///     PatternKind::AgeToUltimate,
///     vec![(12, 2.0), (24, 1.25), (36, 1.0)],
/// );
/// let chain_ladder = ChainLadder {
///     development: Development { pattern: Some(pattern), ..Default::default() },
///     ..Default::default()
/// };
/// let cl = chain_ladder.fit_diagonal(&diagonal).unwrap();
/// assert_eq!(cl.ultimate, vec![900.0, 750.0, 300.0]);
/// let bf = BornhuetterFerguson { apriori: 0.7, chain_ladder }
///     .fit_diagonal(&diagonal)
///     .unwrap();
/// // 150 + 0.7 × 1000 × (1 - 1/2)
/// assert!((bf.ultimate[2] - 500.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LatestDiagonal {
    origin_grain: Grain,
    origins: Vec<Month>,
    ages: Vec<Lag>,
    latest: Vec<f64>,
    exposure: Option<Vec<f64>>,
    past_pattern_end: PastPatternEnd,
}

/// How `fit_diagonal` develops an origin older than the pattern's end (its
/// last age; for age-to-age factors, the end of their span),
/// `reserving-v02.md` decision 18. Set it with
/// [`LatestDiagonal::with_past_pattern_end`].
///
/// ```
/// use prospicio_reserving::{
///     ChainLadder, Development, Grain, LatestDiagonal, Month, PastPatternEnd, PatternKind,
///     SelectedPattern, Tail,
/// };
///
/// // Age-to-age factors from 12 to 36 months, and an origin at 48 months.
/// let pattern = SelectedPattern::new(PatternKind::AgeToAge, vec![(12, 1.5), (24, 1.2)]);
/// let diagonal = LatestDiagonal::new(
///     Grain::Year,
///     &[2021, 2022, 2023, 2024].map(Month::january),
///     &[48, 36, 24, 12],
///     &[100.0, 100.0, 100.0, 100.0],
/// )
/// .unwrap();
/// let chain_ladder = |tail: Tail| ChainLadder {
///     development: Development { pattern: Some(pattern.clone()), ..Default::default() },
///     tail,
/// };
/// // By default the origin at 48 months is an error.
/// assert!(chain_ladder(Tail::default()).fit_diagonal(&diagonal).is_err());
/// // A factor of 1 past the pattern's end.
/// let none = diagonal.clone().with_past_pattern_end(PastPatternEnd::NoDevelopment);
/// let cl = chain_ladder(Tail::default()).fit_diagonal(&none).unwrap();
/// assert!((cl.ultimate[0] - 100.0).abs() < 1e-12);
/// // A tail of 1.05 from 36 months, spread at one step per origin year:
/// // the origin at 48 months takes what remains of it after a year.
/// let tailed = diagonal.with_past_pattern_end(PastPatternEnd::Tail);
/// let cl = chain_ladder(Tail::from(1.05)).fit_diagonal(&tailed).unwrap();
/// assert!((cl.ultimate[0] - 102.53798721500054).abs() < 1e-9);
/// assert!((cl.ultimate[1] - 105.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PastPatternEnd {
    /// An error that names the origin, its age, the pattern's ages and the
    /// two other settings. The default.
    #[default]
    Reject,
    /// No development past the pattern's end: a factor of 1 there. The
    /// method's tail must be none. A cumulative pattern's level at its end
    /// is rescaled to 100% for every origin, as a given tail rescales it
    /// (decision 12 item 5), and reported as
    /// [`pattern_rescale`](crate::ChainLadderFit::pattern_rescale). So with
    /// a level other than 100% every origin's ultimate moves, the young
    /// ones too and also when no origin is past the end: a level of 95%
    /// cuts each factor to ultimate by 5%, where
    /// [`Reject`](Self::Reject) keeps the level as the implied tail.
    NoDevelopment,
    /// The method's tail continues the pattern from its end: an a-priori
    /// [`TailConstant`] (factor and decay, `attachment_age` unset) or a
    /// `TailCurve`, `TailBondy` or `Tail::LogLinear` fitted on the
    /// pattern's own ages (decision 16 item 3) and carried as a constant at
    /// decay 0.5. The factor is spread from the end at one step per origin
    /// period, so an origin one origin period past the end takes the tail's
    /// factor less its first period's development, whatever the pattern's
    /// grain; an origin past the end takes the remaining development from
    /// its own age. The decay is per origin period, so the same tail runs
    /// off over fewer months for accident quarters than for accident years
    /// (a quarter one quarter past the end and a year one year past it
    /// both take 1.02538 of a tail of 1.05), as decided (decision 18 item
    /// 11). An error with no tail: a factor of 1 is
    /// [`NoDevelopment`](Self::NoDevelopment).
    Tail,
}

fn invalid(why: impl Into<String>) -> Error {
    Error::Diagonal(why.into())
}

impl LatestDiagonal {
    /// A diagonal from each origin's start month (floored to
    /// `origin_grain`), its age in months and its latest cumulative value.
    /// Every origin is given once, at a positive age with a finite value;
    /// an origin with nothing reported yet is given as 0. The origins need
    /// not be in order, but they span every period from the earliest to
    /// the latest: one that is skipped has no latest value, an error when
    /// fitting.
    pub fn new(
        origin_grain: Grain,
        origins: &[Month],
        ages: &[Lag],
        latest: &[f64],
    ) -> Result<Self> {
        if origins.is_empty() {
            return Err(Error::Empty);
        }
        for (column, found) in [("ages", ages.len()), ("latest", latest.len())] {
            if found != origins.len() {
                return Err(Error::LengthMismatch {
                    column: column.into(),
                    expected: origins.len(),
                    found,
                });
            }
        }
        if let Some(row) = ages.iter().position(|&a| a == 0) {
            return Err(Error::NonPositiveAge { row });
        }
        if let Some(row) = latest.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFinite {
                column: "latest".into(),
                row,
            });
        }
        let origins: Vec<Month> = origins.iter().map(|m| m.floor(origin_grain)).collect();
        let mut sorted = origins.clone();
        sorted.sort();
        if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
            return Err(invalid(format!(
                "origin {} is given twice",
                Period::containing(w[0], origin_grain)
            )));
        }
        Ok(Self {
            origin_grain,
            origins,
            ages: ages.to_vec(),
            latest: latest.to_vec(),
            exposure: None,
            past_pattern_end: PastPatternEnd::Reject,
        })
    }

    /// The diagonal with `rule` for origins older than the pattern's end
    /// (decision 18). [`new`](Self::new), [`at_valuation`](Self::at_valuation)
    /// and [`from_triangle`](Self::from_triangle) start at
    /// [`PastPatternEnd::Reject`].
    pub fn with_past_pattern_end(mut self, rule: PastPatternEnd) -> Self {
        self.past_pattern_end = rule;
        self
    }

    /// The rule for origins older than the pattern's end.
    pub fn past_pattern_end(&self) -> PastPatternEnd {
        self.past_pattern_end
    }

    /// A diagonal valued at `valuation`: each origin's age is the months
    /// from its start to the valuation month inclusive, as a triangle's
    /// valuation dates give them (an accident year 2024 at December 2024
    /// is 12 months, at June 2024 six). The valuation must not be before
    /// any origin starts.
    pub fn at_valuation(
        origin_grain: Grain,
        origins: &[Month],
        valuation: Month,
        latest: &[f64],
    ) -> Result<Self> {
        let ages = origins
            .iter()
            .enumerate()
            .map(|(row, &m)| age_at(origin_grain, m, valuation, row))
            .collect::<Result<Vec<_>>>()?;
        Self::new(origin_grain, origins, &ages, latest)
    }

    /// The latest diagonal of `column` in a single-segment `triangle`
    /// (`reserving-v02.md`, decision 17): each origin at the age of its last
    /// observed cell, with its cumulative value there, at the triangle's
    /// origin grain. With `exposure`, each origin also takes the last
    /// observed value of that column, as the expected-loss methods read the
    /// exposure on a triangle. The origins run over the ones the column
    /// observes, oldest first.
    ///
    /// This is how a development fitted on one book is applied to another:
    /// `source.to_chain_ladder().fit_diagonal(&LatestDiagonal::from_triangle(&book, "paid", None)?)`,
    /// chainladder-python's `fit(source).predict(book)`
    /// ([`ChainLadderFit::to_chain_ladder`]). An origin whose cell on the
    /// latest diagonal is missing is projected from its last observed cell.
    ///
    /// The diagonal is valued at `column`'s own latest month, so Cape Cod's
    /// `fit_diagonal` trends to it, where [`CapeCod::fit`] on the triangle
    /// trends to the triangle's valuation, the latest month of any column.
    /// When another column (a premium) runs later than `column`, the trended
    /// aprioris differ by the trend over the gap; the detrended aprioris and
    /// the ultimates agree as long as every origin ends by `column`'s
    /// valuation.
    ///
    /// Errors: [`Error::MultipleSegments`] for a triangle with several index
    /// positions, [`Error::UnknownColumn`] for a column it does not have,
    /// [`Error::EmptyOrigin`] for an origin with no observed cell and
    /// [`Error::InvalidExposure`] for an exposure that is not finite and
    /// positive.
    ///
    /// ```
    /// use prospicio_reserving::{
    ///     BornhuetterFerguson, ChainLadder, DevelopmentColumn, Grain, LatestDiagonal, Long, Month,
    ///     Triangle,
    /// };
    ///
    /// let origin = [2020, 2020, 2021].map(Month::january);
    /// let book = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&[12, 24, 12]),
    ///     values: &[
    ///         ("paid", &[100.0, 150.0, 200.0]),
    ///         ("premium", &[400.0, 450.0, 500.0]),
    ///     ],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })
    /// .unwrap();
    /// let diagonal = LatestDiagonal::from_triangle(&book, "paid", Some("premium")).unwrap();
    /// assert_eq!(diagonal.ages(), [24, 12]);
    /// assert_eq!(diagonal.latest(), [150.0, 200.0]);
    /// // Each origin's last observed premium: 450 at 24 months for 2020.
    /// assert_eq!(diagonal.exposure(), Some([450.0, 500.0].as_slice()));
    ///
    /// // The book's own development, applied to its diagonal.
    /// let own = ChainLadder::default().fit(&book, "paid").unwrap();
    /// let cl = own.to_chain_ladder().fit_diagonal(&diagonal).unwrap();
    /// assert_eq!(cl.ultimate, own.ultimate);
    /// assert_eq!(cl.ultimate, [150.0, 300.0]);
    /// let bf = BornhuetterFerguson { apriori: 0.5, chain_ladder: own.to_chain_ladder() }
    ///     .fit_diagonal(&diagonal)
    ///     .unwrap();
    /// assert_eq!(bf.exposure, [450.0, 500.0]);
    /// // 150 + 0.5 × 450 × (1 - 1) and 200 + 0.5 × 500 × (1 - 1/1.5).
    /// assert_eq!(bf.ultimate[0], 150.0);
    /// assert!((bf.ultimate[1] - (200.0 + 250.0 / 3.0)).abs() < 1e-9);
    /// ```
    pub fn from_triangle(
        triangle: &Triangle,
        column: &str,
        exposure: Option<&str>,
    ) -> Result<Self> {
        let segment = triangle.segment(column)?;
        let exposure = exposure
            .map(|e| Ok::<_, Error>((triangle.segment(e)?, e)))
            .transpose()?;
        Self::from_segment(
            triangle.origin_grain(),
            &segment,
            exposure.as_ref().map(|(s, e)| (s, *e)),
        )
    }

    /// The latest diagonal of one segment of a triangle, with the exposure
    /// of the same index position's `(segment, column)` when given.
    fn from_segment(
        origin_grain: Grain,
        segment: &Segment,
        exposure: Option<(&Segment, &str)>,
    ) -> Result<Self> {
        let mut origins = Vec::with_capacity(segment.n_origins);
        let mut ages = Vec::with_capacity(segment.n_origins);
        let mut latest = Vec::with_capacity(segment.n_origins);
        for o in 0..segment.n_origins {
            let (d, v) = segment.latest(o)?;
            origins.push(segment.origins[o].start());
            ages.push(segment.ages[d]);
            latest.push(v);
        }
        let diagonal = Self::new(origin_grain, &origins, &ages, &latest)?;
        match exposure {
            Some((e, column)) => diagonal.with_exposure(&origin_exposure(segment, e, column)?),
            None => Ok(diagonal),
        }
    }

    /// The diagonal with an exposure (premium) per origin, in the order of
    /// the origins given, for the expected-loss methods; each finite and
    /// positive.
    pub fn with_exposure(mut self, exposure: &[f64]) -> Result<Self> {
        if exposure.len() != self.origins.len() {
            return Err(Error::LengthMismatch {
                column: EXPOSURE.into(),
                expected: self.origins.len(),
                found: exposure.len(),
            });
        }
        if let Some(o) = exposure.iter().position(|e| !(e.is_finite() && *e > 0.0)) {
            return Err(Error::InvalidExposure {
                column: EXPOSURE.into(),
                origin: Period::containing(self.origins[o], self.origin_grain).to_string(),
            });
        }
        self.exposure = Some(exposure.to_vec());
        Ok(self)
    }

    /// The origins' grain.
    pub fn origin_grain(&self) -> Grain {
        self.origin_grain
    }

    /// Each origin's start month, in the order given. The fits list the
    /// origins oldest first instead, every period from the earliest to the
    /// latest.
    pub fn origins(&self) -> &[Month] {
        &self.origins
    }

    /// Each origin's age in months, in the order given.
    pub fn ages(&self) -> &[Lag] {
        &self.ages
    }

    /// Each origin's latest value, in the order given.
    pub fn latest(&self) -> &[f64] {
        &self.latest
    }

    /// Each origin's exposure, if given, in the order given.
    pub fn exposure(&self) -> Option<&[f64]> {
        self.exposure.as_deref()
    }

    /// The diagonal's valuation: the latest month any origin is valued at,
    /// to which Cape Cod's `fit_diagonal` trends.
    pub fn valuation(&self) -> Month {
        self.origins
            .iter()
            .zip(&self.ages)
            .map(|(m, &a)| m.add_months(i64::from(a) - 1))
            .max()
            .expect("a diagonal has an origin")
    }

    /// What the methods fit: the diagonal on its grid, the exposure there
    /// when `exposure` asks for it (it must then be given), `development`
    /// with the pattern as it resolves on the grid, and the tail to use
    /// there.
    fn prepare(&self, development: &Development, tail: Tail, exposure: bool) -> Result<Prepared> {
        if development.n_periods.is_some() || development.drop_high > 0 || development.drop_low > 0
        {
            return Err(invalid(
                "n_periods and the drops choose link ratios, and there are none",
            ));
        }
        let Some(pattern) = &development.pattern else {
            return Err(invalid(
                "a selected pattern is needed: there are no link ratios to estimate one",
            ));
        };
        let fully = match &pattern.weight {
            PatternWeight::All(w) => *w == 1.0,
            PatternWeight::ByAge(by_age) => by_age.iter().all(|&(_, w)| w == 1.0),
        };
        if !fully {
            return Err(invalid(
                "the pattern must be fully selected (weight 1): there is no estimate to blend with",
            ));
        }
        let exposure = match (exposure, &self.exposure) {
            (false, _) => None,
            (true, Some(e)) => Some(e.as_slice()),
            (true, None) => {
                return Err(invalid(
                    "the expected-loss methods need an exposure per origin: give it with with_exposure",
                ));
            }
        };
        let gridded = self.on_grid(development, pattern, tail)?;
        let segment = self.segment(&gridded, &self.latest);
        let exposure = exposure.map(|e| self.segment(&gridded, e));
        Ok(Prepared {
            segment,
            exposure,
            development: Development {
                pattern: Some(gridded.pattern),
                ..development.clone()
            },
            tail: gridded.tail,
            converted: gridded.converted,
            rescale: gridded.rescale,
        })
    }

    /// The pattern's span checked against the diagonal's ages, the grid
    /// ages, the pattern to resolve on them and the tail (decisions 16 and
    /// 18).
    fn on_grid(
        &self,
        development: &Development,
        pattern: &SelectedPattern,
        tail: Tail,
    ) -> Result<Gridded> {
        let mut values = pattern.values.clone();
        values.sort_by_key(|&(age, _)| age);
        if values.is_empty() {
            return Err(Error::Pattern {
                age: None,
                reason: "the pattern has no values",
            });
        }
        let pattern_ages: Vec<Lag> = values.iter().map(|&(a, _)| a).collect();
        // An age-to-age factor runs from its age to the next one given, and
        // the last one by the spacing before it (or the origin grain, for a
        // single factor), so a pattern of factors spans one age more than
        // it gives.
        if pattern.kind == PatternKind::AgeToAge {
            check_factors(&values, &pattern.weight)?;
        }
        let single = self.origin_grain.months();
        let end = span_end(pattern.kind, &values, single)?;
        let first = pattern_ages[0];
        let last = end.unwrap_or(pattern_ages[pattern_ages.len() - 1]);
        match (self.past_pattern_end, tail) {
            (PastPatternEnd::NoDevelopment, t) if !t.is_none() => {
                return Err(invalid(
                    "PastPatternEnd::NoDevelopment is a factor of 1 past the pattern's end: \
                     give no tail, or use PastPatternEnd::Tail",
                ));
            }
            (PastPatternEnd::Tail, t) if t.is_none() => {
                return Err(invalid(
                    "PastPatternEnd::Tail needs a tail: for a factor of 1 use \
                     PastPatternEnd::NoDevelopment",
                ));
            }
            (PastPatternEnd::Tail, Tail::Constant(t)) if t.attachment_age.is_some() => {
                return Err(invalid(
                    "a tail past the pattern's end attaches at the pattern's end: leave \
                     attachment_age unset",
                ));
            }
            _ => {}
        }
        let mut oldest = None;
        for (o, &age) in self.ages.iter().enumerate() {
            let outside = || {
                format!(
                    "origin {} is at {age} months, outside the pattern's ages from {first} to {last} months",
                    Period::containing(self.origins[o], self.origin_grain)
                )
            };
            if age < first {
                return Err(invalid(outside()));
            }
            if age > last {
                if self.past_pattern_end == PastPatternEnd::Reject {
                    return Err(invalid(format!(
                        "{}: an origin older than the pattern's end needs \
                         LatestDiagonal::with_past_pattern_end with PastPatternEnd::NoDevelopment \
                         (a factor of 1), or PastPatternEnd::Tail with an a-priori tail (a \
                         TailConstant) or a tail estimated on the pattern's own factors \
                         (TailCurve, TailBondy, Tail::LogLinear)",
                        outside()
                    )));
                }
                oldest = oldest.max(Some(age));
            }
        }
        if let Some(oldest) = oldest {
            return self.past_end(development, pattern, &values, end, tail, oldest);
        }
        // No origin is past the end: decision 16's grid. With no
        // development past the end, the tail is a factor of 1 that attaches
        // there, so a cumulative pattern's level is rescaled to 100%.
        let tail = match self.past_pattern_end {
            PastPatternEnd::NoDevelopment => Tail::Constant(TailConstant {
                factor: 1.0,
                attachment_age: Some(last),
                ..Default::default()
            }),
            _ => tail,
        };
        let own: Vec<Lag> = pattern_ages.iter().chain(&end).copied().collect();
        let all: Vec<Lag> = own.iter().chain(&self.ages).copied().collect();
        let ages = grid(&all, self.origin_grain)?;
        let (pattern_here, converted) = on_ages(pattern, &values, end, &ages);
        // An estimated tail is fitted to the factors in use, and on a grid
        // finer than the pattern's own those are the interpolated factors
        // of the shorter periods. So it is fitted on the pattern's own
        // grid, as a triangle at the pattern's ages would fit it, and
        // carried over as a constant.
        // The pattern's own grid follows its ages alone, not the origins.
        let own_ages = grid(&own, Grain::Year)?;
        let tail = if !matches!(tail, Tail::Constant(_)) && own_ages != ages {
            Tail::from(tail_on_own_ages(pattern, development, tail, single)?)
        } else {
            tail
        };
        Ok(Gridded {
            pattern: pattern_here,
            ages,
            tail,
            converted,
            rescale: None,
        })
    }

    /// Decision 18: the pattern with its tail spread past its end `E` to
    /// the `oldest` origin's age, at one step per origin period. The tail's
    /// factor `T` is spread from `E` as a constant tail attached there, the
    /// factor to ultimate at `E + k g` being what is left of it after `k`
    /// steps, and the pattern's own factors to ultimate are rescaled to `T`
    /// at `E`; together they are one pattern of factors to ultimate.
    fn past_end(
        &self,
        development: &Development,
        pattern: &SelectedPattern,
        values: &[(Lag, f64)],
        end: Option<Lag>,
        tail: Tail,
        oldest: Lag,
    ) -> Result<Gridded> {
        let own = own_cdf(pattern.kind, values, end);
        let (e, level) = own[own.len() - 1];
        let (factor, decay) = match (self.past_pattern_end, tail) {
            (PastPatternEnd::NoDevelopment, _) => (1.0, 0.5),
            (_, Tail::Constant(t)) => (t.factor, t.decay),
            (_, estimated) => (
                tail_on_own_ages(pattern, development, estimated, self.origin_grain.months())?,
                0.5,
            ),
        };
        let g = self.origin_grain.months();
        let steps = (oldest - e).div_ceil(g) as usize;
        let ext: Vec<Lag> = (0..=steps).map(|k| e + k as Lag * g).collect();
        let (_, spread, _) = Tail::Constant(TailConstant {
            factor,
            decay,
            attachment_age: Some(e),
        })
        .select(&vec![1.0; steps], &ext)?;
        // What is left of the tail after k steps; all of it at the end.
        let left = |k: usize| -> f64 {
            if k == 0 {
                factor
            } else {
                spread[k..].iter().product()
            }
        };
        let mut cdf: Vec<(Lag, f64)> = own.iter().map(|&(a, c)| (a, c / level * factor)).collect();
        cdf.extend((1..=steps).map(|k| (ext[k], left(k))));
        let all: Vec<Lag> = cdf
            .iter()
            .map(|&(a, _)| a)
            .chain(self.ages.iter().copied())
            .collect();
        let ages = grid(&all, self.origin_grain)?;
        let rescale = match pattern.kind {
            PatternKind::AgeToAge => None,
            _ => Some(level / factor),
        };
        Ok(Gridded {
            pattern: SelectedPattern {
                kind: PatternKind::AgeToUltimate,
                values: cdf,
                weight: PatternWeight::All(1.0),
            },
            ages,
            tail: Tail::Constant(TailConstant {
                factor: left(steps),
                decay,
                attachment_age: None,
            }),
            converted: false,
            rescale: Some(rescale),
        })
    }

    /// The diagonal's `values` (the latest values or the exposure) as a
    /// segment on the grid: one observed cell per origin, at its age, and
    /// the grid's other ages, which span the pattern's, unobserved. The
    /// origins run from the earliest to the latest at the origin grain.
    fn segment(&self, gridded: &Gridded, values: &[f64]) -> Segment {
        let first = *self.origins.iter().min().expect("a diagonal has an origin");
        let last = *self.origins.iter().max().expect("a diagonal has an origin");
        let step = i64::from(self.origin_grain.months());
        let n_origins = (last.months_since(first) / step + 1) as usize;
        let origins: Vec<Period> = (0..n_origins)
            .map(|k| Period::containing(first.add_months(k as i64 * step), self.origin_grain))
            .collect();
        let n_dev = gridded.ages.len();
        let mut cells = vec![None; n_origins * n_dev];
        for ((m, age), &v) in self.origins.iter().zip(&self.ages).zip(values) {
            let o = (m.months_since(first) / step) as usize;
            let d = gridded
                .ages
                .iter()
                .position(|a| a == age)
                .expect("the grid holds every age of the diagonal");
            cells[o * n_dev + d] = Some(v);
        }
        Segment::from_cells(origins, gridded.ages.clone(), cells)
    }
}

/// A pattern placed on a diagonal's grid.
struct Gridded {
    pattern: SelectedPattern,
    /// The tail to use on the grid.
    tail: Tail,
    /// Whether an age-to-age pattern was turned into factors to ultimate.
    converted: bool,
    /// For a pattern with its tail spread past its end (decision 18), the
    /// `pattern_rescale` to report: the given level at the end over the
    /// tail, `None` for age-to-age factors. The fit reports no
    /// `pattern_tail` then.
    rescale: Option<Option<f64>>,
    /// The grid's ages, from the earliest age of the diagonal and the
    /// pattern to the latest, at the coarsest grain that holds them all.
    ages: Vec<Lag>,
}

/// A diagonal ready to fit.
struct Prepared {
    segment: Segment,
    exposure: Option<Segment>,
    development: Development,
    tail: Tail,
    converted: bool,
    rescale: Option<Option<f64>>,
}

/// What a fit reports of the pattern as given: an age-to-age pattern
/// `converted` into factors to ultimate carries no tail of its own, and one
/// with its tail spread past its end reports the `rescale` of its own
/// level and no pattern tail.
fn report(converted: bool, rescale: Option<Option<f64>>, fit: &mut ChainLadderFit) {
    if converted {
        unconverted(fit);
    }
    if let Some(rescale) = rescale {
        fit.pattern_tail = None;
        fit.pattern_rescale = rescale;
    }
}

impl ChainLadder {
    /// Projects a latest diagonal with the fully selected pattern of
    /// `development` (decision 16): each origin's latest value times the
    /// pattern's factor to ultimate at its age, and the tail. The fit's
    /// development axis is the grid the diagonal is placed on, which spans
    /// the pattern's ages as well as the diagonal's, and its origins run
    /// oldest first over every period from the earliest to the latest. The
    /// average and sigma settings have no effect, since there are no link
    /// ratios; an estimated tail is fitted to the pattern on its own ages.
    /// An origin older than the pattern's end is developed as the diagonal's
    /// [`PastPatternEnd`] says (decision 18): an error by default, a factor
    /// of 1, or the tail spread from the pattern's end at one step per
    /// origin period.
    pub fn fit_diagonal(&self, diagonal: &LatestDiagonal) -> Result<ChainLadderFit> {
        let p = diagonal.prepare(&self.development, self.tail, false)?;
        let mut fit = ChainLadder {
            development: p.development,
            tail: p.tail,
        }
        .fit_segment(&p.segment, &p.segment.ages)?;
        report(p.converted, p.rescale, &mut fit);
        Ok(fit)
    }
}

/// The pattern as given was a pattern of age-to-age factors, which carries
/// no tail of its own: the factors to ultimate it was turned into end at 1,
/// so nothing is rescaled.
fn unconverted(fit: &mut ChainLadderFit) {
    fit.pattern_tail = None;
    fit.pattern_rescale = None;
}

impl ExpectedLoss {
    /// The expected loss method on a latest diagonal with its exposure; see
    /// [`ChainLadder::fit_diagonal`].
    pub fn fit_diagonal(&self, diagonal: &LatestDiagonal) -> Result<ExpectedLossFit> {
        self.as_benktander().fit_diagonal(diagonal)
    }
}

impl BornhuetterFerguson {
    /// Bornhuetter-Ferguson on a latest diagonal with its exposure; see
    /// [`ChainLadder::fit_diagonal`].
    pub fn fit_diagonal(&self, diagonal: &LatestDiagonal) -> Result<ExpectedLossFit> {
        self.as_benktander().fit_diagonal(diagonal)
    }
}

impl Benktander {
    /// Benktander on a latest diagonal with its exposure; see
    /// [`ChainLadder::fit_diagonal`].
    pub fn fit_diagonal(&self, diagonal: &LatestDiagonal) -> Result<ExpectedLossFit> {
        let p = diagonal.prepare(&self.chain_ladder.development, self.chain_ladder.tail, true)?;
        let mut fit = Benktander {
            chain_ladder: ChainLadder {
                development: p.development,
                tail: p.tail,
            },
            ..self.clone()
        }
        .fit_segment(&p.segment, &p.exposure.expect("asked for"), EXPOSURE)?;
        report(p.converted, p.rescale, &mut fit.chain_ladder);
        Ok(fit)
    }
}

impl CapeCod {
    /// Cape Cod on a latest diagonal with its exposure, trended to the
    /// diagonal's valuation (the latest month any origin is valued at); see
    /// [`ChainLadder::fit_diagonal`].
    pub fn fit_diagonal(&self, diagonal: &LatestDiagonal) -> Result<CapeCodFit> {
        self.fit_diagonal_at(diagonal, diagonal.valuation())
    }

    /// Cape Cod on a latest diagonal, trended to `valuation`.
    fn fit_diagonal_at(&self, diagonal: &LatestDiagonal, valuation: Month) -> Result<CapeCodFit> {
        let p = diagonal.prepare(&self.chain_ladder.development, self.chain_ladder.tail, true)?;
        let mut fit = CapeCod {
            chain_ladder: ChainLadder {
                development: p.development,
                tail: p.tail,
            },
            ..self.clone()
        }
        .fit_segment(
            &p.segment,
            &p.exposure.expect("asked for"),
            EXPOSURE,
            valuation,
        )?;
        report(p.converted, p.rescale, &mut fit.expected_loss.chain_ladder);
        Ok(fit)
    }
}

/// Latest diagonals of several segments (lines of business, states, ...),
/// each a [`LatestDiagonal`], labelled by key values as a triangle's
/// segments are (`reserving-v02.md`, decision 19). The methods'
/// `fit_diagonal_segments` project each segment on its own and return a
/// [`SegmentFits`], as `fit_segments` does on a triangle; with
/// `fit_diagonal_segments_from` each segment takes its development from a
/// segment of a source fit, chosen by [`developments`](Self::developments).
///
/// The segments are kept sorted by label, as a triangle sorts its index. A
/// diagonal without keys has one segment, labelled `"Total"`.
///
/// ```
/// use prospicio_reserving::{
///     ChainLadder, DiagonalAge, Grain, Label, LatestDiagonals, LongDiagonal, Month,
/// };
///
/// // Two lines at December 2024, given line by line in any order.
/// let origin = [2024, 2023, 2023, 2024].map(Month::january);
/// let diagonals = LatestDiagonals::from_long(&LongDiagonal {
///     keys: &[("lob", &["b", "b", "a", "a"])],
///     origin: &origin,
///     age: DiagonalAge::Valuation(Month::new(2024, 12).unwrap()),
///     latest: &[50.0, 80.0, 400.0, 150.0],
///     exposure: None,
///     origin_grain: Grain::Year,
/// })
/// .unwrap();
/// assert_eq!(diagonals.labels(), [Label::from("a"), Label::from("b")]);
/// // Line b keeps its origins in the order given.
/// assert_eq!(diagonals.diagonals()[1].ages(), [12, 24]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LatestDiagonals {
    key_names: Vec<String>,
    labels: Vec<Label>,
    diagonals: Vec<LatestDiagonal>,
}

/// How a [`LongDiagonal`] gives each row's age.
#[derive(Debug, Clone, Copy)]
pub enum DiagonalAge<'a> {
    /// Each row's age in months.
    Ages(&'a [Lag]),
    /// One valuation month for every row, ages as
    /// [`LatestDiagonal::at_valuation`] gives them.
    Valuation(Month),
    /// Each row's own valuation month.
    Valuations(&'a [Month]),
}

/// Latest diagonals as a long table: one row per segment and origin, with the
/// key values, the origin's start month, its age (or valuation) and its
/// latest value, and an optional exposure.
#[derive(Debug, Clone, Copy)]
pub struct LongDiagonal<'a> {
    /// Key columns as `(name, values)`, one value per row; empty for a
    /// single segment.
    pub keys: &'a [(&'a str, &'a [&'a str])],
    /// Each row's origin start month (floored to `origin_grain`).
    pub origin: &'a [Month],
    /// Each row's age, or the valuation it is read from.
    pub age: DiagonalAge<'a>,
    /// Each row's latest cumulative value.
    pub latest: &'a [f64],
    /// Each row's exposure (premium), for the expected-loss methods.
    pub exposure: Option<&'a [f64]>,
    /// The origins' grain.
    pub origin_grain: Grain,
}

/// Which segment of a source fit develops each segment of the diagonals in
/// [`LatestDiagonals::developments`].
#[derive(Debug, Clone, Default, PartialEq)]
pub enum SegmentMap {
    /// The source segment of the same label, every source segment for a
    /// source with one segment. The default.
    #[default]
    ByLabel,
    /// `(target, source)` pairs: each segment of the diagonals listed once,
    /// with the source segment that develops it.
    Pairs(Vec<(Label, Label)>),
}

/// The age at `valuation` of an origin starting in `origin`, as
/// [`LatestDiagonal::at_valuation`] reads it; `row` names the row in the
/// error.
fn age_at(origin_grain: Grain, origin: Month, valuation: Month, row: usize) -> Result<Lag> {
    let age = valuation.months_since(origin.floor(origin_grain)) + 1;
    if age <= 0 {
        return Err(Error::NonPositiveAge { row });
    }
    Ok(age as Lag)
}

fn check_length(column: &str, expected: usize, found: usize) -> Result<()> {
    if found == expected {
        Ok(())
    } else {
        Err(Error::LengthMismatch {
            column: column.into(),
            expected,
            found,
        })
    }
}

impl From<LatestDiagonal> for LatestDiagonals {
    /// One segment without keys, labelled `"Total"`.
    fn from(diagonal: LatestDiagonal) -> Self {
        Self {
            key_names: Vec::new(),
            labels: vec![Label::default()],
            diagonals: vec![diagonal],
        }
    }
}

impl LatestDiagonals {
    /// Segments from their labels and diagonals, under `key_names`. Each
    /// label has one value per key (none without keys, for one segment);
    /// the labels differ and every diagonal has the same origin grain. The
    /// segments are sorted by label.
    pub fn new(key_names: Vec<String>, segments: Vec<(Label, LatestDiagonal)>) -> Result<Self> {
        if segments.is_empty() {
            return Err(Error::Empty);
        }
        for (k, name) in key_names.iter().enumerate() {
            if key_names[..k].contains(name) {
                return Err(Error::DuplicateKey(name.clone()));
            }
        }
        let grain = segments[0].1.origin_grain;
        let mut segments = segments;
        segments.sort_by(|a, b| a.0.cmp(&b.0));
        for (k, (label, diagonal)) in segments.iter().enumerate() {
            if label.parts().len() != key_names.len() {
                return Err(invalid(format!(
                    "segment {label} has {} key values for {} keys",
                    label.parts().len(),
                    key_names.len()
                )));
            }
            if k > 0 && segments[k - 1].0 == *label {
                return Err(invalid(format!("segment {label} is given twice")));
            }
            if diagonal.origin_grain != grain {
                return Err(invalid(format!(
                    "segment {label} has origins of another grain: every segment's origins \
                     are on one grain"
                )));
            }
        }
        let (labels, diagonals) = segments.into_iter().unzip();
        Ok(Self {
            key_names,
            labels,
            diagonals,
        })
    }

    /// The diagonals of a long table, one row per segment and origin. The
    /// ages, latest values and lengths are checked over the whole table, so
    /// [`Error::NonPositiveAge`] and [`Error::NonFinite`] name the table's
    /// row; then each segment is built as [`LatestDiagonal::new`] builds
    /// it, its origins in the order of its rows, and an error there (an
    /// origin given twice, an exposure that is not finite and positive) is
    /// reported with the segment's label when there are keys.
    pub fn from_long(long: &LongDiagonal<'_>) -> Result<Self> {
        let n = long.origin.len();
        if n == 0 {
            return Err(Error::Empty);
        }
        let mut key_names: Vec<String> = Vec::with_capacity(long.keys.len());
        for (name, values) in long.keys {
            if key_names.iter().any(|k| k == name) {
                return Err(Error::DuplicateKey(name.to_string()));
            }
            check_length(name, n, values.len())?;
            key_names.push(name.to_string());
        }
        let grain = long.origin_grain;
        let ages: Vec<Lag> = match long.age {
            DiagonalAge::Ages(ages) => {
                check_length("ages", n, ages.len())?;
                if let Some(row) = ages.iter().position(|&a| a == 0) {
                    return Err(Error::NonPositiveAge { row });
                }
                ages.to_vec()
            }
            DiagonalAge::Valuation(v) => (0..n)
                .map(|row| age_at(grain, long.origin[row], v, row))
                .collect::<Result<_>>()?,
            DiagonalAge::Valuations(v) => {
                check_length("valuation", n, v.len())?;
                (0..n)
                    .map(|row| age_at(grain, long.origin[row], v[row], row))
                    .collect::<Result<_>>()?
            }
        };
        check_length("latest", n, long.latest.len())?;
        if let Some(e) = long.exposure {
            check_length(EXPOSURE, n, e.len())?;
        }
        if let Some(row) = long.latest.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFinite {
                column: "latest".into(),
                row,
            });
        }
        let mut rows: BTreeMap<Label, Vec<usize>> = BTreeMap::new();
        for row in 0..n {
            let label = Label::new(long.keys.iter().map(|(_, values)| values[row]));
            rows.entry(label).or_default().push(row);
        }
        let keyed = !key_names.is_empty();
        let mut labels = Vec::with_capacity(rows.len());
        let mut diagonals = Vec::with_capacity(rows.len());
        for (label, rows) in rows {
            let origins: Vec<Month> = rows.iter().map(|&r| long.origin[r]).collect();
            let segment_ages: Vec<Lag> = rows.iter().map(|&r| ages[r]).collect();
            let latest: Vec<f64> = rows.iter().map(|&r| long.latest[r]).collect();
            let diagonal = LatestDiagonal::new(grain, &origins, &segment_ages, &latest)
                .and_then(|d| match long.exposure {
                    Some(e) => d.with_exposure(&rows.iter().map(|&r| e[r]).collect::<Vec<_>>()),
                    None => Ok(d),
                })
                .map_err(|e| labelled(keyed, &label, e))?;
            labels.push(label);
            diagonals.push(diagonal);
        }
        Ok(Self {
            key_names,
            labels,
            diagonals,
        })
    }

    /// The latest diagonal of `column` in every segment of `triangle`, in
    /// its index order, each taken as [`LatestDiagonal::from_triangle`]
    /// takes a single segment's (on the segment's own trimmed range), with
    /// the exposure from the same segment's `exposure` column. An error is
    /// reported with the segment's label when the triangle has keys.
    pub fn from_triangle(
        triangle: &Triangle,
        column: &str,
        exposure: Option<&str>,
    ) -> Result<Self> {
        let segments = triangle.segments(column)?;
        let exposures = exposure.map(|e| triangle.segments(e)).transpose()?;
        let keyed = !triangle.key_names().is_empty();
        let diagonals = segments
            .iter()
            .enumerate()
            .zip(triangle.index())
            .map(|((i, segment), label)| {
                let exposure = exposures
                    .as_ref()
                    .zip(exposure)
                    .map(|(segments, name)| (&segments[i], name));
                LatestDiagonal::from_segment(triangle.origin_grain(), segment, exposure)
                    .map_err(|e| labelled(keyed, label, e))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            key_names: triangle.key_names().to_vec(),
            labels: triangle.index().to_vec(),
            diagonals,
        })
    }

    /// The diagonals with an exposure per row, in the order of the rows:
    /// the segments in their (label) order and, within one, its origins in
    /// the order given. Each finite and positive.
    pub fn with_exposure(mut self, exposure: &[f64]) -> Result<Self> {
        let n: usize = self.diagonals.iter().map(|d| d.origins.len()).sum();
        check_length(EXPOSURE, n, exposure.len())?;
        let keyed = !self.key_names.is_empty();
        let mut start = 0;
        let mut diagonals = Vec::with_capacity(self.diagonals.len());
        for (diagonal, label) in self.diagonals.into_iter().zip(&self.labels) {
            let end = start + diagonal.origins.len();
            diagonals.push(
                diagonal
                    .with_exposure(&exposure[start..end])
                    .map_err(|e| labelled(keyed, label, e))?,
            );
            start = end;
        }
        self.diagonals = diagonals;
        Ok(self)
    }

    /// The diagonals with `rule` for origins older than the pattern's end,
    /// in every segment (decision 18).
    pub fn with_past_pattern_end(mut self, rule: PastPatternEnd) -> Self {
        for d in &mut self.diagonals {
            d.past_pattern_end = rule;
        }
        self
    }

    /// Names of the key columns; empty without keys.
    pub fn key_names(&self) -> &[String] {
        &self.key_names
    }

    /// Label of each segment, in order.
    pub fn labels(&self) -> &[Label] {
        &self.labels
    }

    /// Each segment's diagonal, following `labels`.
    pub fn diagonals(&self) -> &[LatestDiagonal] {
        &self.diagonals
    }

    /// Number of segments.
    pub fn len(&self) -> usize {
        self.diagonals.len()
    }

    /// Whether there are no segments (never, once built).
    pub fn is_empty(&self) -> bool {
        self.diagonals.is_empty()
    }

    /// The diagonal of the segment labelled `label`.
    pub fn get(&self, label: &Label) -> Option<&LatestDiagonal> {
        let i = self.labels.iter().position(|l| l == label)?;
        Some(&self.diagonals[i])
    }

    /// Each segment's label and diagonal, in order.
    pub fn iter(&self) -> impl Iterator<Item = (&Label, &LatestDiagonal)> {
        self.labels.iter().zip(&self.diagonals)
    }

    /// Position of the one segment whose keys have the given values, as
    /// [`SegmentFits::position`] chooses one.
    pub fn position(&self, keys: &[(&str, &str)]) -> Result<usize> {
        find_segment(&self.key_names, &self.labels, keys)
    }

    /// The one segment chosen as in [`position`](Self::position), with the
    /// same key names.
    pub fn segment(&self, keys: &[(&str, &str)]) -> Result<Self> {
        let i = self.position(keys)?;
        Ok(Self {
            key_names: self.key_names.clone(),
            labels: vec![self.labels[i].clone()],
            diagonals: vec![self.diagonals[i].clone()],
        })
    }

    /// The origins' grain, the same in every segment.
    pub fn origin_grain(&self) -> Grain {
        self.diagonals[0].origin_grain
    }

    /// The latest month any origin of any segment is valued at, to which
    /// Cape Cod's `fit_diagonal_segments` trends every segment.
    pub fn valuation(&self) -> Month {
        self.diagonals
            .iter()
            .map(LatestDiagonal::valuation)
            .max()
            .expect("diagonals have a segment")
    }

    /// The development of each segment, from a segment of `source` frozen
    /// by [`ChainLadderFit::to_chain_ladder`], keyed by the diagonals'
    /// labels, for a method's `fit_diagonal_segments_from` (decision 19).
    /// [`SegmentMap::ByLabel`] takes the source segment of the same label,
    /// or the only one of a single-segment source for every segment; the
    /// source's other segments are not used. [`SegmentMap::Pairs`] names
    /// the source segment of each target. Each error names the label.
    pub fn developments(
        &self,
        source: &SegmentFits<ChainLadderFit>,
        map: &SegmentMap,
    ) -> Result<SegmentFits<ChainLadder>> {
        let find = |labels: &[Label], label: &Label| labels.iter().position(|l| l == label);
        let positions: Vec<usize> = match map {
            SegmentMap::ByLabel if source.len() == 1 => vec![0; self.len()],
            SegmentMap::ByLabel => {
                if source.key_names != self.key_names {
                    return Err(invalid(format!(
                        "the source has keys [{}] and the diagonals [{}]: map the segments \
                         with SegmentMap::Pairs",
                        source.key_names.join(", "),
                        self.key_names.join(", ")
                    )));
                }
                self.labels
                    .iter()
                    .map(|l| {
                        find(&source.labels, l).ok_or_else(|| {
                            invalid(format!(
                                "segment {l} of the diagonals has no segment of the same \
                                 label in the source"
                            ))
                        })
                    })
                    .collect::<Result<_>>()?
            }
            SegmentMap::Pairs(pairs) => {
                let mut chosen: Vec<Option<usize>> = vec![None; self.len()];
                for (target, from) in pairs {
                    let t = find(&self.labels, target).ok_or_else(|| {
                        invalid(format!(
                            "the map's target {target} is not a segment of the diagonals"
                        ))
                    })?;
                    if chosen[t].is_some() {
                        return Err(invalid(format!("the map lists the target {target} twice")));
                    }
                    let s = find(&source.labels, from).ok_or_else(|| {
                        invalid(format!(
                            "the map's source {from} is not a segment of the source"
                        ))
                    })?;
                    chosen[t] = Some(s);
                }
                chosen
                    .iter()
                    .zip(&self.labels)
                    .map(|(c, l)| {
                        c.ok_or_else(|| {
                            invalid(format!(
                                "the map gives segment {l} of the diagonals no source"
                            ))
                        })
                    })
                    .collect::<Result<_>>()?
            }
        };
        Ok(SegmentFits {
            key_names: self.key_names.clone(),
            labels: self.labels.clone(),
            fits: positions
                .into_iter()
                .map(|i| source.fits[i].to_chain_ladder())
                .collect(),
        })
    }

    /// `fit` of each segment, labelled as `fit_segments` labels a failure.
    fn fit_each<T>(
        &self,
        mut fit: impl FnMut(&LatestDiagonal) -> Result<T>,
    ) -> Result<SegmentFits<T>> {
        collect_labelled(
            &self.key_names,
            &self.labels,
            self.diagonals.iter().map(&mut fit),
        )
    }

    /// `fit` of each segment with its chain ladder from `development`, when
    /// the method's own `chain_ladder` sets no development of its own.
    fn fit_each_from<T>(
        &self,
        chain_ladder: &ChainLadder,
        development: &SegmentFits<ChainLadder>,
        mut fit: impl FnMut(&ChainLadder, &LatestDiagonal) -> Result<T>,
    ) -> Result<SegmentFits<T>> {
        let d = &chain_ladder.development;
        if d.pattern.is_some()
            || !chain_ladder.tail.is_none()
            || d.n_periods.is_some()
            || d.drop_high > 0
            || d.drop_low > 0
        {
            return Err(invalid(
                "the development comes from the source fit: give the method no pattern, no \
                 tail and no link ratio exclusions",
            ));
        }
        if development.key_names != self.key_names || development.labels != self.labels {
            return Err(invalid("the developments' segments are not the diagonals'"));
        }
        collect_labelled(
            &self.key_names,
            &self.labels,
            self.diagonals
                .iter()
                .zip(&development.fits)
                .map(|(diagonal, cl)| fit(cl, diagonal)),
        )
    }
}

impl ChainLadder {
    /// [`fit_diagonal`](Self::fit_diagonal) on every segment (decision 19).
    pub fn fit_diagonal_segments(
        &self,
        diagonals: &LatestDiagonals,
    ) -> Result<SegmentFits<ChainLadderFit>> {
        diagonals.fit_each(|d| self.fit_diagonal(d))
    }

    /// [`fit_diagonal`](Self::fit_diagonal) on every segment, each with its
    /// development from `development` (from
    /// [`LatestDiagonals::developments`]), whose segments must be the
    /// diagonals'. The method itself must set no pattern, tail or link
    /// ratio exclusion: the development is the source's.
    pub fn fit_diagonal_segments_from(
        &self,
        diagonals: &LatestDiagonals,
        development: &SegmentFits<ChainLadder>,
    ) -> Result<SegmentFits<ChainLadderFit>> {
        diagonals.fit_each_from(self, development, |cl, d| cl.fit_diagonal(d))
    }
}

impl ExpectedLoss {
    /// [`fit_diagonal`](Self::fit_diagonal) on every segment; see
    /// [`ChainLadder::fit_diagonal_segments`].
    pub fn fit_diagonal_segments(
        &self,
        diagonals: &LatestDiagonals,
    ) -> Result<SegmentFits<ExpectedLossFit>> {
        diagonals.fit_each(|d| self.fit_diagonal(d))
    }

    /// The expected loss method on every segment with each segment's
    /// development from `development`, and this method's apriori; see
    /// [`ChainLadder::fit_diagonal_segments_from`].
    pub fn fit_diagonal_segments_from(
        &self,
        diagonals: &LatestDiagonals,
        development: &SegmentFits<ChainLadder>,
    ) -> Result<SegmentFits<ExpectedLossFit>> {
        diagonals.fit_each_from(&self.chain_ladder, development, |cl, d| {
            Self {
                chain_ladder: cl.clone(),
                ..self.clone()
            }
            .fit_diagonal(d)
        })
    }
}

impl BornhuetterFerguson {
    /// [`fit_diagonal`](Self::fit_diagonal) on every segment; see
    /// [`ChainLadder::fit_diagonal_segments`].
    pub fn fit_diagonal_segments(
        &self,
        diagonals: &LatestDiagonals,
    ) -> Result<SegmentFits<ExpectedLossFit>> {
        diagonals.fit_each(|d| self.fit_diagonal(d))
    }

    /// Bornhuetter-Ferguson on every segment with each segment's
    /// development from `development`, and this method's apriori; see
    /// [`ChainLadder::fit_diagonal_segments_from`].
    pub fn fit_diagonal_segments_from(
        &self,
        diagonals: &LatestDiagonals,
        development: &SegmentFits<ChainLadder>,
    ) -> Result<SegmentFits<ExpectedLossFit>> {
        diagonals.fit_each_from(&self.chain_ladder, development, |cl, d| {
            Self {
                chain_ladder: cl.clone(),
                ..self.clone()
            }
            .fit_diagonal(d)
        })
    }
}

impl Benktander {
    /// [`fit_diagonal`](Self::fit_diagonal) on every segment; see
    /// [`ChainLadder::fit_diagonal_segments`].
    pub fn fit_diagonal_segments(
        &self,
        diagonals: &LatestDiagonals,
    ) -> Result<SegmentFits<ExpectedLossFit>> {
        diagonals.fit_each(|d| self.fit_diagonal(d))
    }

    /// Benktander on every segment with each segment's development from
    /// `development`, and this method's apriori and `n_iters`; see
    /// [`ChainLadder::fit_diagonal_segments_from`].
    pub fn fit_diagonal_segments_from(
        &self,
        diagonals: &LatestDiagonals,
        development: &SegmentFits<ChainLadder>,
    ) -> Result<SegmentFits<ExpectedLossFit>> {
        diagonals.fit_each_from(&self.chain_ladder, development, |cl, d| {
            Self {
                chain_ladder: cl.clone(),
                ..self.clone()
            }
            .fit_diagonal(d)
        })
    }
}

impl CapeCod {
    /// Cape Cod on every segment, each trended to one valuation, the latest
    /// month any origin of any segment is valued at
    /// ([`LatestDiagonals::valuation`]), as [`CapeCod::fit_segments`]
    /// trends every segment to the triangle's valuation (decision 19). With
    /// one segment this is [`fit_diagonal`](Self::fit_diagonal).
    pub fn fit_diagonal_segments(
        &self,
        diagonals: &LatestDiagonals,
    ) -> Result<SegmentFits<CapeCodFit>> {
        let valuation = diagonals.valuation();
        diagonals.fit_each(|d| self.fit_diagonal_at(d, valuation))
    }

    /// Cape Cod on every segment with each segment's development from
    /// `development`, and this method's trend and decay, trended as
    /// [`fit_diagonal_segments`](Self::fit_diagonal_segments) trends; see
    /// [`ChainLadder::fit_diagonal_segments_from`].
    pub fn fit_diagonal_segments_from(
        &self,
        diagonals: &LatestDiagonals,
        development: &SegmentFits<ChainLadder>,
    ) -> Result<SegmentFits<CapeCodFit>> {
        let valuation = diagonals.valuation();
        diagonals.fit_each_from(&self.chain_ladder, development, |cl, d| {
            Self {
                chain_ladder: cl.clone(),
                ..self.clone()
            }
            .fit_diagonal_at(d, valuation)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagonal(grain: Grain, latest: &[f64]) -> LatestDiagonal {
        let origins: Vec<Month> = (0..latest.len())
            .map(|k| Month::january(2024).add_months(-(k as i64) * i64::from(grain.months())))
            .collect();
        let ages: Vec<Lag> = (0..latest.len())
            .map(|k| (k as Lag + 1) * grain.months())
            .collect();
        LatestDiagonal::new(grain, &origins, &ages, latest).unwrap()
    }

    #[test]
    fn new_sorts_the_segments_and_checks_their_labels() {
        let d = |v: f64| diagonal(Grain::Year, &[v, v]);
        let diagonals = LatestDiagonals::new(
            vec!["lob".into()],
            vec![(Label::from("b"), d(2.0)), (Label::from("a"), d(1.0))],
        )
        .unwrap();
        assert_eq!(diagonals.labels(), [Label::from("a"), Label::from("b")]);
        assert_eq!(diagonals.diagonals()[0].latest(), [1.0, 1.0]);
        assert_eq!(diagonals.position(&[("lob", "b")]), Ok(1));
        let twice = LatestDiagonals::new(
            vec!["lob".into()],
            vec![(Label::from("a"), d(2.0)), (Label::from("a"), d(1.0))],
        );
        assert!(matches!(twice, Err(Error::Diagonal(m)) if m.contains("given twice")));
        let parts = LatestDiagonals::new(vec!["lob".into()], vec![(Label::default(), d(1.0))]);
        assert!(matches!(parts, Err(Error::Diagonal(m)) if m.contains("0 key values for 1 keys")));
        let grains = LatestDiagonals::new(
            vec!["lob".into()],
            vec![
                (Label::from("a"), d(1.0)),
                (Label::from("b"), diagonal(Grain::Quarter, &[1.0])),
            ],
        );
        assert!(matches!(grains, Err(Error::Diagonal(m)) if m.contains("another grain")));
        assert_eq!(
            LatestDiagonals::new(Vec::new(), Vec::new()),
            Err(Error::Empty)
        );
    }

    #[test]
    fn exposure_and_the_rule_reach_every_segment() {
        let one = LatestDiagonals::from(diagonal(Grain::Year, &[1.0, 2.0]));
        assert_eq!(one.labels(), [Label::default()]);
        assert!(one.key_names().is_empty());
        let two = LatestDiagonals::new(
            vec!["lob".into()],
            vec![
                (Label::from("a"), diagonal(Grain::Year, &[1.0, 2.0])),
                (Label::from("b"), diagonal(Grain::Year, &[3.0])),
            ],
        )
        .unwrap()
        .with_past_pattern_end(PastPatternEnd::Tail);
        assert!(
            two.diagonals()
                .iter()
                .all(|d| d.past_pattern_end() == PastPatternEnd::Tail)
        );
        let set = two.clone().with_exposure(&[10.0, 20.0, 30.0]).unwrap();
        assert_eq!(set.diagonals()[1].exposure(), Some([30.0].as_slice()));
        assert!(matches!(
            two.clone().with_exposure(&[10.0, 20.0]),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            two.with_exposure(&[10.0, 20.0, -1.0]),
            Err(Error::InSegment { label, .. }) if label == "b"
        ));
    }
}
