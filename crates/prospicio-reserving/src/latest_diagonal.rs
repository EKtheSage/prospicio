//! A latest diagonal on its own: one value per origin at its own age, as a
//! bordereau holds it, projected with a selected pattern from elsewhere
//! (`reserving-v02.md`, decisions 15 and 16).
//!
//! [`LatestDiagonal`] places one observed cell per origin on a grid and
//! runs the triangle methods on it through the fully selected path of
//! decision 15, so the chain ladder and the expected-loss methods give the
//! same fits as on any triangle. What it adds is the grid: its
//! ages are the diagonal's and the pattern's together, at the coarsest
//! development grain they share, so an age off the pattern's grid (18
//! months against an annual pattern) is read by the pattern's log-linear
//! rule between its ages.

use prospicio_core::{Grain, Lag, Month, Period};

use crate::expected_loss::origin_exposure;
use crate::triangle::Segment;
use crate::{
    Benktander, BornhuetterFerguson, CapeCod, CapeCodFit, ChainLadder, ChainLadderFit, Development,
    Error, ExpectedLoss, ExpectedLossFit, PatternKind, PatternWeight, Result, SelectedPattern,
    Tail, Triangle,
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
        })
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
    ///         ("premium", &[400.0, 400.0, 500.0]),
    ///     ],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })
    /// .unwrap();
    /// let diagonal = LatestDiagonal::from_triangle(&book, "paid", Some("premium")).unwrap();
    /// assert_eq!(diagonal.ages(), [24, 12]);
    /// assert_eq!(diagonal.latest(), [150.0, 200.0]);
    /// assert_eq!(diagonal.exposure(), Some([400.0, 500.0].as_slice()));
    ///
    /// // The book's own development, applied to its diagonal.
    /// let own = ChainLadder::default().fit(&book, "paid").unwrap();
    /// let cl = own.to_chain_ladder().fit_diagonal(&diagonal).unwrap();
    /// assert_eq!(cl.ultimate, own.ultimate);
    /// assert_eq!(cl.ultimate, [150.0, 300.0]);
    /// let bf = BornhuetterFerguson { apriori: 0.6, chain_ladder: own.to_chain_ladder() }
    ///     .fit_diagonal(&diagonal)
    ///     .unwrap();
    /// // 200 + 0.6 × 500 × (1 - 1/1.5)
    /// assert!((bf.ultimate[1] - 300.0).abs() < 1e-9);
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
        })
    }

    /// The pattern's span checked against the diagonal's ages, the grid
    /// ages, the pattern to resolve on them and the tail (decision 16).
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
        let end = match pattern.kind {
            PatternKind::AgeToAge => {
                check_factors(&values, &pattern.weight)?;
                let n = pattern_ages.len();
                let step = if n > 1 {
                    pattern_ages[n - 1] - pattern_ages[n - 2]
                } else {
                    self.origin_grain.months()
                };
                Some(
                    pattern_ages[n - 1]
                        .checked_add(step)
                        .ok_or(Error::Pattern {
                            age: Some(pattern_ages[n - 1]),
                            reason: "the age is too large",
                        })?,
                )
            }
            _ => None,
        };
        let first = pattern_ages[0];
        let last = end.unwrap_or(pattern_ages[pattern_ages.len() - 1]);
        for (o, &age) in self.ages.iter().enumerate() {
            if age < first || age > last {
                return Err(invalid(format!(
                    "origin {} is at {age} months, outside the pattern's ages from {first} to {last} months",
                    Period::containing(self.origins[o], self.origin_grain)
                )));
            }
        }
        let own: Vec<Lag> = pattern_ages.iter().chain(&end).copied().collect();
        let all: Vec<Lag> = own.iter().chain(&self.ages).copied().collect();
        let ages = grid(&all, self.origin_grain)?;
        let on = |grid: &[Lag]| on_ages(pattern, &values, end, grid);
        let (pattern_here, converted) = on(&ages);
        // An estimated tail is fitted to the factors in use, and on a grid
        // finer than the pattern's own those are the interpolated factors
        // of the shorter periods. So it is fitted on the pattern's own
        // grid, as a triangle at the pattern's ages would fit it, and
        // carried over as a constant.
        // The pattern's own grid follows its ages alone, not the origins.
        let own_ages = grid(&own, Grain::Year)?;
        let tail = if !matches!(tail, Tail::Constant(_)) && own_ages != ages {
            let (own_pattern, _) = on(&own_ages);
            let cells = (0..own_ages.len())
                .map(|d| (d == 0).then_some(1.0))
                .collect();
            let one = Segment::from_cells(
                vec![Period::containing(self.origins[0], self.origin_grain)],
                own_ages.clone(),
                cells,
            );
            let fit = ChainLadder {
                development: Development {
                    pattern: Some(own_pattern),
                    ..development.clone()
                },
                tail,
            }
            .fit_segment(&one, &own_ages)?;
            Tail::from(fit.tail.factor)
        } else {
            tail
        };
        Ok(Gridded {
            pattern: pattern_here,
            ages,
            tail,
            converted,
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
}

/// The grid that holds `ages`: from the earliest to the latest, at the
/// coarsest grain that divides `origin_grain` and every gap.
fn grid(ages: &[Lag], origin_grain: Grain) -> Result<Vec<Lag>> {
    let mut sorted = ages.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let step = grain_of(&sorted, origin_grain).months();
    let (first, last) = (sorted[0], sorted[sorted.len() - 1]);
    if (last - first) / step >= MAX_GRID {
        return Err(invalid(format!(
            "the ages from {first} to {last} months need more than {MAX_GRID} development periods of {step} months",
        )));
    }
    Ok((first..=last).step_by(step as usize).collect())
}

/// The most development periods a diagonal's grid may have: 200 years by
/// month.
const MAX_GRID: Lag = 2400;

/// An age-to-age pattern's factors and weights checked as the pattern's
/// resolution checks them, before they are turned into factors to
/// ultimate, so that an error names the factor that was given.
fn check_factors(values: &[(Lag, f64)], weight: &PatternWeight) -> Result<()> {
    if let Some(w) = values.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(Error::Pattern {
            age: Some(w[0].0),
            reason: "the age is given twice",
        });
    }
    if let Some(&(age, _)) = values.iter().find(|&&(_, f)| !(f.is_finite() && f > 0.0)) {
        return Err(Error::Pattern {
            age: Some(age),
            reason: "an age-to-age factor must be finite and positive",
        });
    }
    if let PatternWeight::ByAge(by_age) = weight {
        for (k, &(age, _)) in by_age.iter().enumerate() {
            if by_age[..k].iter().any(|&(a, _)| a == age) {
                return Err(Error::Pattern {
                    age: Some(age),
                    reason: "the weight's age is given twice",
                });
            }
            if !values.iter().any(|&(a, _)| a == age) {
                return Err(Error::Pattern {
                    age: Some(age),
                    reason: "a weight is given at an age with no selected factor",
                });
            }
        }
    }
    Ok(())
}

/// `pattern` (its `values` sorted) to resolve on the `grid` ages, and
/// whether it was turned into factors to ultimate: on a grid finer than
/// the factors' spacing an age-to-age factor would run to the next grid
/// age, so the factors are given as factors to ultimate (their products,
/// 1 at `end`), which the log-linear rule spreads over the ages between.
fn on_ages(
    pattern: &SelectedPattern,
    values: &[(Lag, f64)],
    end: Option<Lag>,
    grid: &[Lag],
) -> (SelectedPattern, bool) {
    match end {
        Some(end) if values.len() + 1 < grid.len() => {
            let mut cdf = vec![(end, 1.0)];
            for &(age, f) in values.iter().rev() {
                let next = cdf[cdf.len() - 1].1;
                cdf.push((age, next * f));
            }
            cdf.reverse();
            let pattern = SelectedPattern {
                kind: PatternKind::AgeToUltimate,
                values: cdf,
                weight: PatternWeight::All(1.0),
            };
            (pattern, true)
        }
        _ => (pattern.clone(), false),
    }
}

/// The coarsest development grain that divides `origin_grain` and every
/// gap between the sorted `ages`.
fn grain_of(ages: &[Lag], origin_grain: Grain) -> Grain {
    [Grain::Year, Grain::Semester, Grain::Quarter, Grain::Month]
        .into_iter()
        .find(|g| {
            origin_grain.is_multiple_of(*g)
                && ages.windows(2).all(|w| (w[1] - w[0]) % g.months() == 0)
        })
        .unwrap_or(Grain::Month)
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
    pub fn fit_diagonal(&self, diagonal: &LatestDiagonal) -> Result<ChainLadderFit> {
        let p = diagonal.prepare(&self.development, self.tail, false)?;
        let mut fit = ChainLadder {
            development: p.development,
            tail: p.tail,
        }
        .fit_segment(&p.segment, &p.segment.ages)?;
        if p.converted {
            unconverted(&mut fit);
        }
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
        if p.converted {
            unconverted(&mut fit.chain_ladder);
        }
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
        if p.converted {
            unconverted(&mut fit.expected_loss.chain_ladder);
        }
        Ok(fit)
    }
}
