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

use prospicio_core::{Grain, Lag, Month, Period};

use crate::expected_loss::origin_exposure;
use crate::pattern::{check_factors, grid, on_ages, own_cdf, span_end, tail_on_own_ages};
use crate::triangle::Segment;
use crate::{
    Benktander, BornhuetterFerguson, CapeCod, CapeCodFit, ChainLadder, ChainLadderFit, Development,
    Error, ExpectedLoss, ExpectedLossFit, PatternKind, PatternWeight, Result, SelectedPattern,
    Tail, TailConstant, Triangle,
};

/// The exposure column named in errors.
const EXPOSURE: &str = "exposure";

/// One latest value per origin at its own age, with an optional exposure
/// (premium) per origin: a bordereau's diagonal. The chain ladder and the
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
        let mut ages = Vec::with_capacity(origins.len());
        for (row, m) in origins.iter().enumerate() {
            let age = valuation.months_since(m.floor(origin_grain)) + 1;
            if age <= 0 {
                return Err(Error::NonPositiveAge { row });
            }
            ages.push(age as Lag);
        }
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
        let mut origins = Vec::with_capacity(segment.n_origins);
        let mut ages = Vec::with_capacity(segment.n_origins);
        let mut latest = Vec::with_capacity(segment.n_origins);
        for o in 0..segment.n_origins {
            let (d, v) = segment.latest(o)?;
            origins.push(segment.origins[o].start());
            ages.push(segment.ages[d]);
            latest.push(v);
        }
        let diagonal = Self::new(triangle.origin_grain(), &origins, &ages, &latest)?;
        match exposure {
            Some(e) => {
                diagonal.with_exposure(&origin_exposure(&segment, &triangle.segment(e)?, e)?)
            }
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

    /// What the methods fit: the diagonal on its grid, the exposure there
    /// when `exposure` asks for it (it must then be given), the valuation,
    /// `development` with the pattern as it resolves on the grid, and the
    /// tail to use there.
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
        let valuation = self
            .origins
            .iter()
            .zip(&self.ages)
            .map(|(m, &a)| m.add_months(i64::from(a) - 1))
            .max()
            .expect("a diagonal has an origin");
        Ok(Prepared {
            segment,
            exposure,
            valuation,
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
    valuation: Month,
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
            p.valuation,
        )?;
        report(p.converted, p.rescale, &mut fit.expected_loss.chain_ladder);
        Ok(fit)
    }
}
