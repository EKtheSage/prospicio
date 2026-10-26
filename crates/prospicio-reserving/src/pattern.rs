//! Selected development patterns: benchmark factors that replace, or blend
//! with, the factors estimated from the triangle
//! (`docs/design/reserving-v02.md`, decisions 12 and 18).
//!
//! A [`SelectedPattern`] is given as age-to-age factors, factors to
//! ultimate or percents developed, by age. It is converted to age-to-age
//! factors on the triangle's development ages; each selected factor is then
//! blended with the triangle's estimate as `w * selected + (1 - w) *
//! estimated`, `w` its [`PatternWeight`]. Factors it does not select stay
//! estimated. A pattern at ages the triangle does not have (a monthly
//! pattern on an annual triangle, ages between the triangle's, ages past
//! its oldest) is read on its own ages, and what it gives past the oldest
//! age folds into a constant tail there (decision 18).
//! [`SelectedPattern::averaged_over_origins`] turns a pattern measured on
//! one origin basis (accident months) into the pattern of a coarser one
//! (accident years or quarters).
//!
//! A selected factor carries no parameter error: Mack's standard errors and
//! the bootstraps charge a blended factor `(1 - w)` times the estimate's
//! error, and a fully selected one none. A supplied pattern therefore lowers
//! the reported uncertainty, and a wrong benchmark is not reflected in it.
//! The ODP bootstrap keeps one source of parameter error even with every
//! factor selected: it projects each origin from its pseudo latest value,
//! so the estimation error of the origins' levels remains (on RAA about
//! 73% of the unblended standard deviation, with no process error). Mack's
//! bootstrap projects from the observed latest value and keeps none.

use prospicio_core::{Grain, Lag, Month, Period};

use crate::error::{Error, Result};
use crate::triangle::Segment;
use crate::{ChainLadder, Development, Tail};

/// How the values of a [`SelectedPattern`] are expressed. The user names
/// it: age-to-age factors and factors to ultimate are both at least 1 in
/// the usual case and cannot be told apart, so no check on the values
/// could tell them apart either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternKind {
    /// `f(a)`: the factor from age `a` to the next development age of the
    /// triangle; for a pattern with an age that is not one, to the next age
    /// the pattern gives, the last by the spacing before it (decision 18).
    /// Any value that is finite and positive; below 1 is negative
    /// development.
    AgeToAge,
    /// `CDF(a)`: the factor from age `a` to ultimate. Any value that is
    /// finite and positive, in any order: below 1 is negative development
    /// to ultimate, as in a reported pattern that ends under 1.
    AgeToUltimate,
    /// `p(a) = 1 / CDF(a)`: the share of the ultimate developed at age `a`.
    /// Any value that is finite and positive, in any order: above 1 is
    /// negative development to ultimate.
    PercentDeveloped,
}

/// The weight `w` on a selected factor in `w * selected + (1 - w) *
/// estimated`, from 0 (the estimate) to 1 (the selection, the default).
#[derive(Debug, Clone, PartialEq)]
pub enum PatternWeight {
    /// The same weight on every selected factor.
    All(f64),
    /// A weight per age, keyed by the age that starts the factor's link; a
    /// selected factor whose age is not listed gets 1.
    ByAge(Vec<(Lag, f64)>),
}

/// Which factors a bootstrap without a [`SelectedPattern`] of its own
/// simulates in the one-year view
/// ([`OdpBootstrap::one_year`](crate::OdpBootstrap::one_year),
/// [`MackBootstrap::one_year`](crate::MackBootstrap::one_year)), when the
/// re-reserving method has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OneYearPattern {
    /// The method's pattern: each simulation's pseudo factors are blended
    /// with it as the method's factors are, so the coming year develops as
    /// the method projects it. The default, as the Python and R bindings'
    /// `bootstrap_ldf = True`. A bootstrap with no tail of its own reads
    /// the pattern as the method does: a cumulative pattern to the oldest
    /// age implies its tail when the method has no tail either, and none
    /// when the method has a tail of its own (to which the method rescales
    /// the pattern's level), so the method's tail never makes the default
    /// fail.
    #[default]
    Method,
    /// The triangle's own estimated factors, unblended: the method's
    /// pattern enters only when it re-reserves (the bindings'
    /// `bootstrap_ldf = False`). An error for a bootstrap with a pattern of
    /// its own.
    Estimated,
}

impl OneYearPattern {
    /// The pattern a bootstrap with `own` simulates the one-year view of a
    /// method with `method`'s pattern with.
    pub(crate) fn resolve(
        self,
        own: Option<&SelectedPattern>,
        method: Option<&SelectedPattern>,
    ) -> Result<Option<SelectedPattern>> {
        match (own, self) {
            (Some(_), Self::Estimated) => Err(Error::Bootstrap(
                "a bootstrap with a pattern of its own cannot simulate from the estimated \
                 factors: give it no pattern, or OneYearPattern::Method",
            )),
            (Some(p), Self::Method) => Ok(Some(p.clone())),
            (None, Self::Method) => Ok(method.cloned()),
            (None, Self::Estimated) => Ok(None),
        }
    }
}

impl Default for PatternWeight {
    fn default() -> Self {
        Self::All(1.0)
    }
}

/// A development pattern chosen by the user, which replaces or blends with
/// the factors estimated from the triangle at the ages it gives.
///
/// * An age-to-age pattern selects the factor of each age it gives, from
///   that age to the next; any ages but the oldest (development past it is
///   the tail's).
/// * A cumulative pattern ([`PatternKind::AgeToUltimate`] or
///   [`PatternKind::PercentDeveloped`]) selects the factors between its
///   first and last ages, `CDF(a_k) / CDF(a_k+1)`. A development age it
///   skips is interpolated: between two given ages `b < c`, `ln CDF` is
///   linear in the age, so the link from `a_k` to `a_k+1` is
///   `(CDF(b) / CDF(c))^((a_k+1 - a_k) / (c - b))` (percents developed
///   are converted to CDFs first). Ages before the first and after the
///   last given age stay estimated.
/// * A cumulative pattern's level at its last age selects no factor. With
///   no [`Tail`] (a constant 1 at the oldest age) and a last
///   age that is the oldest, the level is the tail: the tail factor is the
///   pattern's CDF there (1 over its percent developed), a constant with no
///   parameter error, as chainladder-python's `DevelopmentConstant` keeps it
///   ([`ChainLadderFit::pattern_tail`](crate::ChainLadderFit::pattern_tail)).
///   Otherwise what develops after the last age comes from the estimated
///   factors and the tail, and the chain ladder reports how much the given
///   level is rescaled to match them
///   ([`ChainLadderFit::pattern_rescale`](crate::ChainLadderFit::pattern_rescale)):
///   with a tail, percent developed at the oldest age times the tail is
///   100%.
/// * A cumulative pattern with one value is the factor to ultimate from
///   its age, the tail from there (`reserving-v02.md`, decision 20). It
///   selects no link, so it is given at the oldest age, fully selected and
///   with no [`Tail`], and is the tail as above; on a triangle observed at
///   one age (one origin, or every origin at one age) it is the whole
///   projection. At another age, with a tail or with a weight below 1 it
///   would select nothing, and it is an error.
///
/// Factors the pattern does not select are estimated, as without one, and
/// the triangle must still estimate every factor (the estimate is reported
/// next to the selection and gives Mack's sigma). A selected factor at or
/// after the age where the [`Tail`] takes over is an error.
///
/// **Ages off the triangle's** (`reserving-v02.md`, decision 18). When every
/// age the pattern gives is a development age of the triangle, the rules
/// above apply as they are. Otherwise the pattern is read on its own ages,
/// as [`LatestDiagonal`](crate::LatestDiagonal) reads it: a cumulative
/// pattern ends at its last age, an age-to-age factor runs to the next age
/// given (the last by the spacing before it, and 1 at the end of that
/// span), and `ln CDF` is linear in the age between the pattern's own ages.
/// A link of the triangle within the pattern's span is selected: the
/// product of the age-to-age factors given over it, or the ratio of the
/// factors to ultimate at its ends. So a monthly pattern selects the annual
/// factors of an annual triangle, and a pattern at 6, 18 and 30 months is
/// read between its ages. Links outside the span stay estimated, and a
/// cumulative pattern's level is the factor to ultimate at the triangle's
/// last age within the span. A single age-to-age factor off the triangle's
/// ages is still an error. Adding one off-grid age to a sparse age-to-age
/// pattern changes how its other factors are read, from each running to the
/// triangle's next age to each running to the pattern's next, with no
/// warning: `{12: 1.5, 36: 1.2}` on an annual triangle selects 1.5 from 12
/// to 24 months, and `{12: 1.5, 18: 1.1, 36: 1.2}` selects 1.5484 there,
/// since 1.5 then runs from 12 to 18 months.
///
/// **Past the oldest age** (decision 18). What such a pattern gives from
/// the triangle's oldest age to its end folds into a constant tail at the
/// oldest age ([`ChainLadderFit::pattern_tail`](crate::ChainLadderFit::pattern_tail)):
/// with no tail, the pattern's factor to ultimate there, its level
/// included; with a [`TailConstant`](crate::TailConstant) (its attachment
/// age unset), the pattern's development to its end times the constant,
/// spread at the constant's decay; with an estimated tail, times the tail
/// fitted on the pattern's own ages. The fold takes the pattern at full
/// weight, so such a pattern must be fully selected: any weight below 1 is
/// an error (cut the pattern at the oldest age to blend it).
///
/// ```
/// use prospicio_reserving::{ChainLadder, Development, PatternKind, SelectedPattern};
/// # use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// # let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
/// # let tri = Triangle::from_long(&Long {
/// #     keys: &[],
/// #     origin: &origin,
/// #     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
/// #     values: &[("paid", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0])],
/// #     origin_grain: Grain::Year,
/// #     development_grain: Grain::Year,
/// #     cumulative: true,
/// # })
/// # .unwrap();
/// // 50%, 80% and 100% developed at 12, 24 and 36 months: factors 1.6 and 1.25.
/// let pattern = SelectedPattern::new(
///     PatternKind::PercentDeveloped,
///     vec![(12, 0.5), (24, 0.8), (36, 1.0)],
/// );
/// let cl = ChainLadder {
///     development: Development { pattern: Some(pattern), ..Default::default() },
///     ..Default::default()
/// }
/// .fit(&tri, "paid")
/// .unwrap();
/// assert!((cl.ldf()[0] - 1.6).abs() < 1e-12 && (cl.ldf()[1] - 1.25).abs() < 1e-12);
/// assert!((cl.ultimate[2] - 120.0 * 2.0).abs() < 1e-9);
/// // The triangle's own estimate is kept next to the selection.
/// assert!((cl.development.ldf[1] - 165.0 / 150.0).abs() < 1e-12);
/// assert_eq!(cl.pattern_rescale, Some(1.0));
///
/// // 40% at 12 and 95% at the oldest age, 36 months, with no tail: 24
/// // months is interpolated, and the 5% left is the tail.
/// let pattern = SelectedPattern::new(PatternKind::PercentDeveloped, vec![(12, 0.4), (36, 0.95)]);
/// let cl = ChainLadder {
///     development: Development { pattern: Some(pattern), ..Default::default() },
///     ..Default::default()
/// }
/// .fit(&tri, "paid")
/// .unwrap();
/// assert!((cl.ldf()[0] - (0.95f64 / 0.4).sqrt()).abs() < 1e-12);
/// assert!((cl.pattern_tail.unwrap() - 1.0 / 0.95).abs() < 1e-12);
/// assert!((cl.ultimate[2] - 120.0 / 0.4).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedPattern {
    /// How `values` are expressed.
    pub kind: PatternKind,
    /// `(age in months, value)`, in any order.
    pub values: Vec<(Lag, f64)>,
    /// The weight on the selected factors; the default selects them fully.
    pub weight: PatternWeight,
}

impl SelectedPattern {
    /// A pattern of `kind` with `values` by age, fully selected.
    pub fn new(kind: PatternKind, values: Vec<(Lag, f64)>) -> Self {
        Self {
            kind,
            values,
            weight: PatternWeight::default(),
        }
    }

    /// The pattern with `weight` on its selected factors.
    pub fn with_weight(mut self, weight: PatternWeight) -> Self {
        self.weight = weight;
        self
    }

    /// The pattern on the development `ages` of a triangle: per link, the
    /// selected age-to-age factor and its weight, and for a cumulative
    /// pattern its last age's position and factor to ultimate there.
    pub(crate) fn resolve(&self, ages: &[Lag]) -> Result<Resolved> {
        let n_links = ages.len().saturating_sub(1);
        let err = |age: Option<Lag>, reason: &'static str| Err(Error::Pattern { age, reason });
        if self.values.is_empty() {
            return err(None, "the pattern has no values");
        }
        let mut values = self.values.clone();
        values.sort_by_key(|&(age, _)| age);
        // A pattern with an age off the triangle's is read on its own ages
        // (decision 18); one on them by decision 12, as it always was.
        if values.iter().any(|(age, _)| !ages.contains(age)) {
            return self.resolve_off_grid(&values, ages);
        }
        let mut positions = Vec::with_capacity(values.len());
        for (i, &(age, _)) in values.iter().enumerate() {
            if i > 0 && values[i - 1].0 == age {
                return err(Some(age), "the age is given twice");
            }
            match ages.iter().position(|&a| a == age) {
                Some(p) => positions.push(p),
                None => return err(Some(age), "not a development age of the triangle"),
            }
        }

        let mut selected = vec![None; n_links];
        let mut level = None;
        match self.kind {
            PatternKind::AgeToAge => {
                for (&(age, f), &p) in values.iter().zip(&positions) {
                    if !f.is_finite() || f <= 0.0 {
                        return err(
                            Some(age),
                            "an age-to-age factor must be finite and positive",
                        );
                    }
                    if p == n_links {
                        return err(
                            Some(age),
                            "an age-to-age factor at the oldest age is development past it: use a tail",
                        );
                    }
                    selected[p] = Some(f);
                }
            }
            PatternKind::AgeToUltimate | PatternKind::PercentDeveloped => {
                let percent = self.kind == PatternKind::PercentDeveloped;
                let mut cdf = Vec::with_capacity(values.len());
                for &(age, v) in &values {
                    if !(v.is_finite() && v > 0.0) {
                        return err(
                            Some(age),
                            if percent {
                                "a percent developed must be finite and positive"
                            } else {
                                "a factor to ultimate must be finite and positive"
                            },
                        );
                    }
                    cdf.push(if percent { 1.0 / v } else { v });
                }
                // One value is the factor to ultimate from its age, the
                // tail from there (decision 20): it fixes no link, so only
                // at the oldest age is there a tail for it to be.
                if values.len() == 1 && positions[0] != n_links {
                    return err(
                        Some(values[0].0),
                        "a cumulative pattern with one value is the tail from its age: give it \
                         at the oldest age, or give two or more ages",
                    );
                }
                // Between two given ages `b < c`, ln CDF is linear in the
                // age: the link from `a_k` to `a_k+1` gets the given ratio
                // raised to its share of the span, `(CDF(b) / CDF(c))^((a_k+1
                // - a_k) / (c - b))`.
                for (i, w) in positions.windows(2).enumerate() {
                    let ratio = cdf[i] / cdf[i + 1];
                    if w[1] == w[0] + 1 {
                        selected[w[0]] = Some(ratio);
                        continue;
                    }
                    let span = (ages[w[1]] - ages[w[0]]) as f64;
                    for k in w[0]..w[1] {
                        let share = (ages[k + 1] - ages[k]) as f64 / span;
                        selected[k] = Some((ratio.ln() * share).exp());
                    }
                }
                let last = positions.len() - 1;
                level = Some((positions[last], cdf[last]));
            }
        }
        let weight = self.weights(&selected, ages)?;
        Ok(Resolved {
            selected,
            weight,
            level,
            beyond: None,
        })
    }

    /// Decision 18: a pattern with an age that is not a development age of
    /// the triangle, read on its own ages as decision 16 reads it. `values`
    /// are sorted by age.
    fn resolve_off_grid(&self, values: &[(Lag, f64)], ages: &[Lag]) -> Result<Resolved> {
        check_unique(values)?;
        if self.kind == PatternKind::AgeToAge && values.len() == 1 {
            return Err(Error::Pattern {
                age: Some(values[0].0),
                reason: "not a development age of the triangle",
            });
        }
        check_values(values, self.kind)?;
        let end = span_end(self.kind, values, 0)?;
        let own = own_cdf(self.kind, values, end);
        let (first, end) = (own[0].0, own[own.len() - 1].0);
        let at = |age: Lag| log_linear_at(&own, age);
        let is_own = |age: Lag| own.binary_search_by_key(&age, |&(a, _)| a).is_ok();
        let n_links = ages.len().saturating_sub(1);
        let mut selected = vec![None; n_links];
        for (k, s) in selected.iter_mut().enumerate() {
            let (a, b) = (ages[k], ages[k + 1]);
            if a < first || b > end {
                continue;
            }
            // Factors given from a to b are the link's to the bit; any
            // other link is the ratio of the factors to ultimate at its
            // ends, given or read between the pattern's ages.
            *s = Some(
                if self.kind == PatternKind::AgeToAge && is_own(a) && is_own(b) {
                    values
                        .iter()
                        .filter(|&&(v, _)| a <= v && v < b)
                        .map(|&(_, f)| f)
                        .product()
                } else {
                    at(a) / at(b)
                },
            );
        }
        let oldest = ages.last().copied().unwrap_or(0);
        let level = match self.kind {
            PatternKind::AgeToAge => None,
            _ => ages
                .iter()
                .rposition(|&a| first <= a && a <= end.min(oldest))
                .map(|j| (j, at(ages[j]))),
        };
        let beyond = (end > oldest && first <= oldest).then(|| Beyond {
            oldest: at(oldest),
            level: at(end),
        });
        if beyond.is_none() && selected.iter().all(Option::is_none) {
            return Err(Error::Pattern {
                age: Some(first),
                reason: "the pattern selects no factor of the triangle",
            });
        }
        let weight = self.weights(&selected, ages)?;
        Ok(Resolved {
            selected,
            weight,
            level,
            beyond,
        })
    }

    /// The weight on each `selected` link on `ages`, 0 where none is.
    fn weights(&self, selected: &[Option<f64>], ages: &[Lag]) -> Result<Vec<f64>> {
        let n_links = selected.len();
        let err = |age: Option<Lag>, reason: &'static str| Err(Error::Pattern { age, reason });
        let check_weight = |age: Option<Lag>, w: f64| {
            if w.is_finite() && (0.0..=1.0).contains(&w) {
                Ok(w)
            } else {
                Err(Error::Pattern {
                    age,
                    reason: "a weight must be between 0 and 1",
                })
            }
        };
        let mut weight = vec![0.0; n_links];
        match &self.weight {
            PatternWeight::All(w) => {
                let w = check_weight(None, *w)?;
                for (k, s) in selected.iter().enumerate() {
                    if s.is_some() {
                        weight[k] = w;
                    }
                }
            }
            PatternWeight::ByAge(by_age) => {
                for (k, s) in selected.iter().enumerate() {
                    if s.is_some() {
                        weight[k] = 1.0;
                    }
                }
                let mut seen: Vec<Lag> = Vec::with_capacity(by_age.len());
                for &(age, w) in by_age {
                    if seen.contains(&age) {
                        return err(Some(age), "the weight's age is given twice");
                    }
                    seen.push(age);
                    let w = check_weight(Some(age), w)?;
                    match ages.iter().position(|&a| a == age) {
                        Some(k) if k < n_links && selected[k].is_some() => weight[k] = w,
                        _ => {
                            return err(
                                Some(age),
                                "a weight is given at an age with no selected factor",
                            );
                        }
                    }
                }
            }
        }
        Ok(weight)
    }

    /// This pattern, measured on origin periods of `pattern_origin` (an
    /// accident-month pattern is `Grain::Month`), as the pattern of origin
    /// periods of `origin`, a whole multiple of it, with the exposure even
    /// through the period (`reserving-v02.md`, decision 18). An origin of
    /// `n = origin / pattern_origin` sub-periods of `s` months is at age
    /// `a` when its `q`-th sub-period from the start (`q = 0` the earliest)
    /// is at age `a - q s`, so
    ///
    /// ```text
    /// P(a) = (1/n) * sum_{q=0}^{n-1} p(a - q s)
    /// ```
    ///
    /// with `p` this pattern's percent developed read at any age:
    ///
    /// * 0 at an age of 0 or less (a sub-period that has not begun);
    /// * linear in `p` from 0 at age 0 to `p(first)` between 0 and the
    ///   first age;
    /// * decision 12's rule between the given ages (`ln CDF` linear in the
    ///   age; an age-to-age pattern is first turned into factors to
    ///   ultimate, 1 at the end of its span, decision 16 item 3);
    /// * held at its value at the end, past the end.
    ///
    /// The result is a [`PatternKind::PercentDeveloped`] pattern at every
    /// month from this pattern's first age to its end plus `(n - 1) s`
    /// months, where every sub-period is past the end and `P` is this
    /// pattern's level (1 for age-to-age factors). Development past the
    /// result's end is the tail's, for the whole period. An origin still in
    /// progress (an accident year at June, 6 months) projects to the whole
    /// period's ultimate. For origins up to this pattern's end the
    /// ultimates are exact when the exposure is even; between its end and
    /// the result's, `P` holds this pattern's level for the sub-periods past
    /// the end, so any development a tail gives there is deferred to the
    /// period's last age, which slightly overstates the ultimate.
    ///
    /// The default origin basis, read at the origin's age, is the pattern
    /// as given, with no call. An estimated tail on the result is fitted on
    /// its monthly factors; a constant tail taken from this pattern's own
    /// tail is the safer choice.
    ///
    /// A weight `All(w)` carries over. With `n = 1` the pattern comes back
    /// unchanged. Errors ([`Error::Pattern`]): `origin` not a whole multiple
    /// of `pattern_origin`; a `ByAge` weight (weight the result with
    /// [`with_weight`](Self::with_weight)); a single age-to-age factor,
    /// which has no span; a result longer than 2,400 months; and any error
    /// the pattern has on its own ages (no values, an age given twice, a
    /// value that is not finite and positive). A single cumulative value is
    /// held past its age, as any pattern's end is (decision 20).
    ///
    /// ```
    /// use prospicio_reserving::{Grain, PatternKind, SelectedPattern};
    ///
    /// // An accident month reports t/12 of its ultimate at t months, fully
    /// // at 12.
    /// let month = SelectedPattern::new(
    ///     PatternKind::PercentDeveloped,
    ///     (1..=12).map(|t| (t, f64::from(t) / 12.0)).collect(),
    /// );
    /// let year = month.averaged_over_origins(Grain::Month, Grain::Year).unwrap();
    /// assert_eq!(year.kind, PatternKind::PercentDeveloped);
    /// let at = |age| year.values.iter().find(|v| v.0 == age).unwrap().1;
    /// // An accident year at December: (12 + 11 + ... + 1) / 12 / 12.
    /// assert!((at(12) - 78.0 / 144.0).abs() < 1e-15);
    /// // At June, its first six months: (6 + ... + 1) / 12 / 12.
    /// assert!((at(6) - 21.0 / 144.0).abs() < 1e-15);
    /// // Every month is fully developed 11 months after the pattern's end.
    /// assert_eq!(year.values.last(), Some(&(23, 1.0)));
    /// ```
    pub fn averaged_over_origins(
        &self,
        pattern_origin: Grain,
        origin: Grain,
    ) -> Result<SelectedPattern> {
        let err = |age: Option<Lag>, reason: &'static str| Err(Error::Pattern { age, reason });
        if !origin.is_multiple_of(pattern_origin) {
            return err(
                None,
                "the origin grain must be a whole multiple of the pattern's origin grain",
            );
        }
        let n = origin.months() / pattern_origin.months();
        if n == 1 {
            return Ok(self.clone());
        }
        let s = pattern_origin.months();
        if let PatternWeight::ByAge(_) = self.weight {
            return err(
                None,
                "weights by age do not carry over to another origin basis: give one weight \
                 for every factor",
            );
        }
        if self.values.is_empty() {
            return err(None, "the pattern has no values");
        }
        let mut values = self.values.clone();
        values.sort_by_key(|&(age, _)| age);
        check_unique(&values)?;
        if self.kind == PatternKind::AgeToAge && values.len() == 1 {
            return err(
                Some(values[0].0),
                "a single age-to-age factor has no span to average: give two or more, or \
                 factors to ultimate",
            );
        }
        check_values(&values, self.kind)?;
        // Percents developed on the pattern's own ages, the given value
        // itself where one is given.
        let points: Vec<(Lag, f64)> = match self.kind {
            PatternKind::PercentDeveloped => values,
            PatternKind::AgeToUltimate => values.iter().map(|&(a, c)| (a, 1.0 / c)).collect(),
            PatternKind::AgeToAge => {
                let end = span_end(self.kind, &values, 0)?;
                own_cdf(self.kind, &values, end)
                    .into_iter()
                    .map(|(a, c)| (a, 1.0 / c))
                    .collect()
            }
        };
        let (first, end) = (points[0], points[points.len() - 1]);
        let last = (n - 1)
            .checked_mul(s)
            .and_then(|extra| end.0.checked_add(extra))
            .ok_or(Error::Pattern {
                age: Some(end.0),
                reason: "the age is too large",
            })?;
        if last - first.0 >= MAX_GRID {
            return err(
                None,
                "the averaged pattern needs more than 2,400 monthly ages",
            );
        }
        let p = |age: i64| -> f64 {
            if age <= 0 {
                0.0
            } else if age < i64::from(first.0) {
                first.1 * age as f64 / f64::from(first.0)
            } else if age > i64::from(end.0) {
                end.1
            } else {
                log_linear_at(&points, age as Lag)
            }
        };
        let averaged = (first.0..=last)
            .map(|a| {
                let sum: f64 = (0..n).map(|q| p(i64::from(a) - i64::from(q * s))).sum();
                (a, sum / f64::from(n))
            })
            .collect();
        Ok(SelectedPattern {
            kind: PatternKind::PercentDeveloped,
            values: averaged,
            weight: self.weight.clone(),
        })
    }
}

/// Each age of the sorted `values` given once.
fn check_unique(values: &[(Lag, f64)]) -> Result<()> {
    match values.windows(2).find(|w| w[0].0 == w[1].0) {
        Some(w) => Err(Error::Pattern {
            age: Some(w[0].0),
            reason: "the age is given twice",
        }),
        None => Ok(()),
    }
}

/// Each value of a pattern of `kind` finite and positive, with decision
/// 12's texts. A cumulative pattern may have one value (decision 20).
fn check_values(values: &[(Lag, f64)], kind: PatternKind) -> Result<()> {
    let reason = match kind {
        PatternKind::AgeToAge => "an age-to-age factor must be finite and positive",
        PatternKind::AgeToUltimate => "a factor to ultimate must be finite and positive",
        PatternKind::PercentDeveloped => "a percent developed must be finite and positive",
    };
    match values.iter().find(|&&(_, v)| !(v.is_finite() && v > 0.0)) {
        Some(&(age, _)) => Err(Error::Pattern {
            age: Some(age),
            reason,
        }),
        None => Ok(()),
    }
}

/// The end of an age-to-age pattern's span (decision 16 item 3): each
/// factor runs to the next age given, and the last one by the spacing
/// before it, or by `single` for a single factor. `None` for a cumulative
/// pattern, which ends at its last age. `values` are sorted.
pub(crate) fn span_end(
    kind: PatternKind,
    values: &[(Lag, f64)],
    single: Lag,
) -> Result<Option<Lag>> {
    if kind != PatternKind::AgeToAge {
        return Ok(None);
    }
    let n = values.len();
    let step = if n > 1 {
        values[n - 1].0 - values[n - 2].0
    } else {
        single
    };
    values[n - 1]
        .0
        .checked_add(step)
        .map(Some)
        .ok_or(Error::Pattern {
            age: Some(values[n - 1].0),
            reason: "the age is too large",
        })
}

/// The factors to ultimate of a pattern of `kind` on its own ages, sorted:
/// a cumulative pattern's values (1 over a percent developed), or an
/// age-to-age pattern's products from each age to `end`, the end of its
/// span, with 1 there.
pub(crate) fn own_cdf(
    kind: PatternKind,
    values: &[(Lag, f64)],
    end: Option<Lag>,
) -> Vec<(Lag, f64)> {
    match (kind, end) {
        (PatternKind::AgeToAge, Some(end)) => {
            let mut cdf = vec![(end, 1.0)];
            for &(age, f) in values.iter().rev() {
                let next = cdf[cdf.len() - 1].1;
                cdf.push((age, next * f));
            }
            cdf.reverse();
            cdf
        }
        (PatternKind::PercentDeveloped, _) => values.iter().map(|&(a, p)| (a, 1.0 / p)).collect(),
        _ => values.to_vec(),
    }
}

/// The value at `age` of a curve given at the sorted `points`, `ln` of it
/// linear in the age between two of them: the given value itself at one of
/// them. `age` lies within the points' span.
fn log_linear_at(points: &[(Lag, f64)], age: Lag) -> f64 {
    match points.binary_search_by_key(&age, |&(a, _)| a) {
        Ok(i) => points[i].1,
        Err(i) => {
            let ((b, vb), (c, vc)) = (points[i - 1], points[i]);
            let share = f64::from(age - b) / f64::from(c - b);
            (vb.ln() + share * (vc.ln() - vb.ln())).exp()
        }
    }
}

/// The grid that holds `ages`: from the earliest to the latest, at the
/// coarsest grain that divides `origin_grain` and every gap.
pub(crate) fn grid(ages: &[Lag], origin_grain: Grain) -> Result<Vec<Lag>> {
    let mut sorted = ages.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let step = grain_of(&sorted, origin_grain).months();
    let (first, last) = (sorted[0], sorted[sorted.len() - 1]);
    if (last - first) / step >= MAX_GRID {
        return Err(Error::Diagonal(format!(
            "the ages from {first} to {last} months need more than {MAX_GRID} development periods of {step} months",
        )));
    }
    Ok((first..=last).step_by(step as usize).collect())
}

/// The most development periods a diagonal's grid may have: 200 years by
/// month.
pub(crate) const MAX_GRID: Lag = 2400;

/// An age-to-age pattern's factors and weights checked as the pattern's
/// resolution checks them, before they are turned into factors to
/// ultimate, so that an error names the factor that was given.
pub(crate) fn check_factors(values: &[(Lag, f64)], weight: &PatternWeight) -> Result<()> {
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
pub(crate) fn on_ages(
    pattern: &SelectedPattern,
    values: &[(Lag, f64)],
    end: Option<Lag>,
    grid: &[Lag],
) -> (SelectedPattern, bool) {
    match end {
        Some(end) if values.len() + 1 < grid.len() => {
            let pattern = SelectedPattern {
                kind: PatternKind::AgeToUltimate,
                values: own_cdf(PatternKind::AgeToAge, values, Some(end)),
                weight: PatternWeight::All(1.0),
            };
            (pattern, true)
        }
        _ => (pattern.clone(), false),
    }
}

/// The coarsest development grain that divides `origin_grain` and every
/// gap between the sorted `ages`.
pub(crate) fn grain_of(ages: &[Lag], origin_grain: Grain) -> Grain {
    [Grain::Year, Grain::Semester, Grain::Quarter, Grain::Month]
        .into_iter()
        .find(|g| {
            origin_grain.is_multiple_of(*g)
                && ages.windows(2).all(|w| (w[1] - w[0]) % g.months() == 0)
        })
        .unwrap_or(Grain::Month)
}

/// The factor an estimated `tail` gives from the end of `pattern` on, fitted
/// on the pattern's own ages (decision 16 item 3): a one-origin fit on the
/// grid of the pattern's ages (and the end of an age-to-age pattern's span,
/// by `single` months for a single factor), with the pattern fully selected
/// there, as a triangle at the pattern's ages would fit the tail.
/// `development`'s average and sigma settings carry over; its `n_periods`
/// and drops, which choose link ratios, do not.
pub(crate) fn tail_on_own_ages(
    pattern: &SelectedPattern,
    development: &Development,
    tail: Tail,
    single: Lag,
) -> Result<f64> {
    let mut values = pattern.values.clone();
    values.sort_by_key(|&(age, _)| age);
    if values.is_empty() {
        return Err(Error::Pattern {
            age: None,
            reason: "the pattern has no values",
        });
    }
    if pattern.kind == PatternKind::AgeToAge {
        check_factors(&values, &PatternWeight::All(1.0))?;
    }
    let end = span_end(pattern.kind, &values, single)?;
    let own: Vec<Lag> = values.iter().map(|&(a, _)| a).chain(end).collect();
    let own_ages = grid(&own, Grain::Year)?;
    let (own_pattern, _) = on_ages(pattern, &values, end, &own_ages);
    let cells = (0..own_ages.len())
        .map(|d| (d == 0).then_some(1.0))
        .collect();
    let one = Segment::from_cells(
        vec![Period::containing(Month::january(2000), Grain::Year)],
        own_ages.clone(),
        cells,
    );
    let fit = ChainLadder {
        development: Development {
            pattern: Some(own_pattern.with_weight(PatternWeight::All(1.0))),
            n_periods: None,
            drop_high: 0,
            drop_low: 0,
            ..development.clone()
        },
        tail,
    }
    .fit_segment(&one, &own_ages)?;
    Ok(fit.tail.factor)
}

impl SelectedPattern {
    /// Whether `self` and `other` select the same factors with the same
    /// weights on the development `ages`: the comparison of their resolved
    /// forms, so the order of `values`, the kind they are given in and the
    /// form of the weight do not matter (`All(1.0)` is `ByAge` with every
    /// weight 1). The factors agree to a relative `1e-12`, since a pattern
    /// given as percents developed and as factors to ultimate differ by
    /// rounding. A cumulative pattern's level at its last age, which only
    /// reports `pattern_rescale`, is not compared. An error if either
    /// pattern does not resolve on `ages`.
    pub(crate) fn same_selection(&self, other: &Self, ages: &[Lag]) -> Result<bool> {
        let (a, b) = (self.resolve(ages)?, other.resolve(ages)?);
        let factors = a
            .selected
            .iter()
            .zip(&b.selected)
            .all(|(x, y)| match (x, y) {
                (Some(x), Some(y)) => (x - y).abs() <= 1e-12 * x.abs().max(y.abs()),
                (None, None) => true,
                _ => false,
            });
        Ok(factors && a.weight == b.weight)
    }
}

/// A [`SelectedPattern`] on a triangle's development ages.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Resolved {
    /// Per link, the selected age-to-age factor.
    pub(crate) selected: Vec<Option<f64>>,
    /// Per link, the weight on it; 0 where none is selected.
    pub(crate) weight: Vec<f64>,
    /// For a cumulative pattern, the position of its last age and its
    /// factor to ultimate there; for one read off the triangle's ages
    /// (decision 18), of the last development age within its span.
    pub(crate) level: Option<(usize, f64)>,
    /// For a pattern read off the triangle's ages that runs past the
    /// oldest age (decision 18), its factors to ultimate there and at its
    /// end, which fold into a constant tail at the oldest age.
    pub(crate) beyond: Option<Beyond>,
}

/// A pattern past the triangle's oldest age `a_n`, to its end `E`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Beyond {
    /// `C(a_n)`, the pattern's factor to ultimate at the oldest age.
    pub(crate) oldest: f64,
    /// `C(E)`, its level at its end (1 for age-to-age factors).
    pub(crate) level: f64,
}

impl Beyond {
    /// `C(a_n) / C(E)`: the pattern's development from the oldest age to
    /// its end.
    pub(crate) fn ratio(&self) -> f64 {
        self.oldest / self.level
    }
}

impl Resolved {
    /// The tail the pattern implies on `ages` when no tail is given: a
    /// cumulative pattern's factor to ultimate at the oldest age; `None`
    /// for one that stops before the oldest age, or an age-to-age pattern.
    pub(crate) fn implied_tail(&self, ages: &[Lag]) -> Option<f64> {
        self.level
            .filter(|&(last, _)| last + 1 == ages.len())
            .map(|(_, cdf)| cdf)
    }

    /// Whether the pattern is only a level, the tail: a cumulative pattern
    /// with one value at the oldest age (decision 20). Any other pattern
    /// that resolves selects a link or runs past the oldest age.
    pub(crate) fn is_tail_only(&self) -> bool {
        self.level.is_some() && self.beyond.is_none() && self.selected.iter().all(Option::is_none)
    }
}

impl SelectedPattern {
    /// The age-to-age pattern on `ages` that selects the same factors with
    /// the same weights and has no level, so it implies no tail: how a
    /// bootstrap reads the pattern it takes from a re-reserving method that
    /// has a tail of its own, as the method reads it (the method rescales
    /// the level to its tail).
    ///
    /// A pattern that runs past the oldest age is an error: what it gives
    /// there folds into the method's tail (decision 18), which a bootstrap
    /// with no tail of its own does not have.
    pub(crate) fn without_level(&self, ages: &[Lag]) -> Result<Self> {
        let resolved = self.resolve(ages)?;
        if resolved.beyond.is_some() {
            return Err(Error::Bootstrap(
                "a pattern that runs past the oldest age carries the re-reserving method's tail \
                 into it: give the bootstrap the method's tail",
            ));
        }
        if resolved.is_tail_only() {
            return Err(Error::Bootstrap(
                "a cumulative pattern with one value is the re-reserving method's tail: give \
                 the bootstrap the method's tail",
            ));
        }
        let links: Vec<(Lag, f64, f64)> = resolved
            .selected
            .iter()
            .zip(&resolved.weight)
            .enumerate()
            .filter_map(|(k, (s, &w))| s.map(|f| (ages[k], f, w)))
            .collect();
        Ok(Self {
            kind: PatternKind::AgeToAge,
            values: links.iter().map(|&(a, f, _)| (a, f)).collect(),
            weight: PatternWeight::ByAge(links.iter().map(|&(a, _, w)| (a, w)).collect()),
        })
    }
}

/// `w * selected + (1 - w) * estimated` per link: the selected factors
/// blended with `estimated`, which are the estimate itself where none is
/// selected.
pub(crate) fn blend(selected: &[Option<f64>], weight: &[f64], estimated: &[f64]) -> Vec<f64> {
    estimated
        .iter()
        .zip(selected.iter().zip(weight))
        .map(|(&e, (s, &w))| match s {
            Some(s) if w == 1.0 => *s,
            Some(s) => w * s + (1.0 - w) * e,
            None => e,
        })
        .collect()
}

/// What a bootstrap blends each simulation's pseudo factors with: the
/// selected factors and their weights. `None` from [`Blend::of`] without a
/// pattern, so that the draws are as they were before patterns.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Blend {
    selected: Vec<Option<f64>>,
    weight: Vec<f64>,
}

impl Blend {
    /// The blend of a development fit, `None` when nothing is selected.
    pub(crate) fn of(development: &crate::DevelopmentFit) -> Option<Self> {
        development
            .selected
            .iter()
            .any(Option::is_some)
            .then(|| Self {
                selected: development.selected.clone(),
                weight: development.weight.clone(),
            })
    }

    /// `pseudo` blended: `w * selected + (1 - w) * pseudo` per link.
    pub(crate) fn apply(&self, pseudo: &[f64]) -> Vec<f64> {
        blend(&self.selected, &self.weight, pseudo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AGES: [Lag; 4] = [12, 24, 36, 48];

    fn close(got: &[Option<f64>], want: &[Option<f64>]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            match (g, w) {
                (Some(g), Some(w)) => assert!((g - w).abs() < 1e-12, "{got:?} vs {want:?}"),
                (None, None) => {}
                _ => panic!("{got:?} vs {want:?}"),
            }
        }
    }

    fn pattern(kind: PatternKind, values: &[(Lag, f64)]) -> SelectedPattern {
        SelectedPattern::new(kind, values.to_vec())
    }

    fn reason(p: &SelectedPattern) -> (Option<Lag>, &'static str) {
        match p.resolve(&AGES) {
            Err(Error::Pattern { age, reason }) => (age, reason),
            other => panic!("expected a pattern error, got {other:?}"),
        }
    }

    #[test]
    fn each_kind_converts_to_the_same_factors() {
        // p = (0.5, 0.8, 0.95) gives links 1.6 and 1.1875.
        let p = pattern(
            PatternKind::PercentDeveloped,
            &[(12, 0.5), (24, 0.8), (36, 0.95)],
        )
        .resolve(&AGES)
        .unwrap();
        let cdf = pattern(
            PatternKind::AgeToUltimate,
            &[(36, 1.0 / 0.95), (12, 2.0), (24, 1.25)],
        )
        .resolve(&AGES)
        .unwrap();
        let ata = pattern(PatternKind::AgeToAge, &[(12, 1.6), (24, 1.1875)])
            .resolve(&AGES)
            .unwrap();
        let want = [Some(1.6), Some(1.1875), None];
        close(&p.selected, &want);
        close(&cdf.selected, &want);
        close(&ata.selected, &want);
        assert_eq!(p.weight, [1.0, 1.0, 0.0]);
        assert_eq!(p.level.map(|l| l.0), Some(2));
        assert!((p.level.unwrap().1 - 1.0 / 0.95).abs() < 1e-15);
        assert_eq!(cdf.level.map(|l| l.0), Some(2));
        assert_eq!(ata.level, None);
    }

    #[test]
    fn round_trips_between_kinds() {
        // Age-to-age factors to CDFs and percents developed and back.
        let f = [1.7, 1.2, 1.05];
        let mut cdf = [1.0; 4];
        for k in (0..3).rev() {
            cdf[k] = cdf[k + 1] * f[k];
        }
        let as_cdf: Vec<(Lag, f64)> = AGES.iter().copied().zip(cdf.iter().copied()).collect();
        let as_pct: Vec<(Lag, f64)> = AGES
            .iter()
            .copied()
            .zip(cdf.iter().map(|c| 1.0 / c))
            .collect();
        let want: Vec<Option<f64>> = f.iter().map(|&x| Some(x)).collect();
        close(
            &pattern(PatternKind::AgeToUltimate, &as_cdf)
                .resolve(&AGES)
                .unwrap()
                .selected,
            &want,
        );
        close(
            &pattern(PatternKind::PercentDeveloped, &as_pct)
                .resolve(&AGES)
                .unwrap()
                .selected,
            &want,
        );
    }

    #[test]
    fn partial_patterns_select_only_their_links() {
        // Young end: CDFs at 12 and 24 select the 12-24 link only.
        let young = pattern(PatternKind::AgeToUltimate, &[(12, 3.0), (24, 1.5)])
            .resolve(&AGES)
            .unwrap();
        close(&young.selected, &[Some(2.0), None, None]);
        assert_eq!(young.level, Some((1, 1.5)));
        // Old end: percents at 24, 36 and 48 select 24-36 and 36-48.
        let old = pattern(
            PatternKind::PercentDeveloped,
            &[(24, 0.8), (36, 0.9), (48, 1.0)],
        )
        .resolve(&AGES)
        .unwrap();
        close(&old.selected, &[None, Some(1.125), Some(1.0 / 0.9)]);
        assert_eq!(old.level, Some((3, 1.0)));
        // Age-to-age: any ages but the oldest, below 1 allowed.
        let ata = pattern(PatternKind::AgeToAge, &[(36, 0.98)])
            .resolve(&AGES)
            .unwrap();
        close(&ata.selected, &[None, None, Some(0.98)]);
    }

    #[test]
    fn missing_ages_are_interpolated_log_linearly_in_age() {
        // CDF 4 at 12 and 1 at 48: ln CDF falls by ln 4 over 36 months, a
        // third per 12-month link.
        let p = pattern(PatternKind::AgeToUltimate, &[(12, 4.0), (48, 1.0)])
            .resolve(&AGES)
            .unwrap();
        let third = 4f64.powf(1.0 / 3.0);
        close(&p.selected, &[Some(third), Some(third), Some(third)]);
        assert_eq!(p.level, Some((3, 1.0)));
        // Percents developed are converted to CDFs first: 25% at 12, 50% at
        // 36 and 100% at 48 give sqrt(2), sqrt(2) and 2.
        let p = pattern(
            PatternKind::PercentDeveloped,
            &[(12, 0.25), (36, 0.5), (48, 1.0)],
        )
        .resolve(&AGES)
        .unwrap();
        let root = 2f64.sqrt();
        close(&p.selected, &[Some(root), Some(root), Some(2.0)]);
        // Uneven ages: each link gets its share of the span in months.
        let ages = [12, 24, 48, 60];
        let p = pattern(PatternKind::AgeToUltimate, &[(12, 3.0), (60, 1.0)])
            .resolve(&ages)
            .unwrap();
        close(
            &p.selected,
            &[
                Some(3f64.powf(0.25)),
                Some(3f64.powf(0.5)),
                Some(3f64.powf(0.25)),
            ],
        );
        // Ages before the first and after the last given age stay
        // estimated, and an interpolated age takes a weight.
        let p = pattern(PatternKind::AgeToUltimate, &[(12, 2.0), (36, 1.21)])
            .with_weight(PatternWeight::ByAge(vec![(24, 0.5)]))
            .resolve(&AGES)
            .unwrap();
        let half = 2f64.sqrt() / 1.1;
        close(&p.selected, &[Some(half), Some(half), None]);
        assert_eq!(p.weight, [1.0, 0.5, 0.0]);
        // The interpolated pattern selects what the same factors given in
        // full do.
        let full = pattern(
            PatternKind::AgeToUltimate,
            &[(12, 4.0), (24, 4.0 / third), (36, third), (48, 1.0)],
        );
        let gapped = pattern(PatternKind::AgeToUltimate, &[(12, 4.0), (48, 1.0)]);
        assert!(gapped.same_selection(&full, &AGES).unwrap());
    }

    #[test]
    fn weights_all_and_by_age() {
        let p = pattern(PatternKind::AgeToAge, &[(12, 2.0), (24, 1.5)]);
        let all = p
            .clone()
            .with_weight(PatternWeight::All(0.25))
            .resolve(&AGES)
            .unwrap();
        assert_eq!(all.weight, [0.25, 0.25, 0.0]);
        let by_age = p
            .with_weight(PatternWeight::ByAge(vec![(24, 0.4)]))
            .resolve(&AGES)
            .unwrap();
        assert_eq!(by_age.weight, [1.0, 0.4, 0.0]);
        let blended = blend(&by_age.selected, &by_age.weight, &[3.0, 1.0, 1.1]);
        assert_eq!(blended[0], 2.0);
        assert!((blended[1] - (0.4 * 1.5 + 0.6 * 1.0)).abs() < 1e-15);
        assert_eq!(blended[2], 1.1);
    }

    #[test]
    fn blend_limits() {
        // w = 0 is the estimate, w = 1 the selection.
        let estimated = [2.5, 1.3, 1.1];
        for (w, want) in [(0.0, estimated), (1.0, [2.0, 1.5, 1.1])] {
            let r = pattern(PatternKind::AgeToAge, &[(12, 2.0), (24, 1.5)])
                .with_weight(PatternWeight::All(w))
                .resolve(&AGES)
                .unwrap();
            assert_eq!(blend(&r.selected, &r.weight, &estimated), want);
        }
    }

    #[test]
    fn every_error_names_its_age() {
        use PatternKind::*;
        assert_eq!(reason(&pattern(AgeToAge, &[])).0, None);
        assert_eq!(
            reason(&pattern(AgeToAge, &[(12, 2.0), (12, 2.0)])).0,
            Some(12)
        );
        assert_eq!(
            reason(&pattern(AgeToAge, &[(18, 2.0)])),
            (Some(18), "not a development age of the triangle")
        );
        assert_eq!(reason(&pattern(AgeToAge, &[(12, 0.0)])).0, Some(12));
        assert_eq!(reason(&pattern(AgeToAge, &[(12, f64::NAN)])).0, Some(12));
        assert_eq!(reason(&pattern(AgeToAge, &[(48, 1.01)])).0, Some(48));
        assert_eq!(
            reason(&pattern(AgeToUltimate, &[(12, 0.0), (24, 0.9)])).0,
            Some(12)
        );
        assert_eq!(
            reason(&pattern(AgeToUltimate, &[(12, 1.5), (24, f64::NAN)])).0,
            Some(24)
        );
        assert_eq!(
            reason(&pattern(PercentDeveloped, &[(12, 0.0), (24, 0.5)])).0,
            Some(12)
        );
        assert_eq!(
            reason(&pattern(PercentDeveloped, &[(12, 0.5), (24, -1.2)])).0,
            Some(24)
        );
        assert_eq!(
            reason(&pattern(
                PercentDeveloped,
                &[(12, 0.6), (24, f64::INFINITY)]
            ))
            .0,
            Some(24)
        );
        assert_eq!(reason(&pattern(PercentDeveloped, &[(24, 0.5)])).0, Some(24));
        let ata = pattern(AgeToAge, &[(12, 2.0)]);
        for w in [-0.1, 1.5, f64::NAN] {
            assert_eq!(
                reason(&ata.clone().with_weight(PatternWeight::All(w))).0,
                None
            );
        }
        assert_eq!(
            reason(
                &ata.clone()
                    .with_weight(PatternWeight::ByAge(vec![(12, 2.0)]))
            )
            .0,
            Some(12)
        );
        assert_eq!(
            reason(
                &ata.clone()
                    .with_weight(PatternWeight::ByAge(vec![(24, 0.5)]))
            )
            .0,
            Some(24)
        );
        assert_eq!(
            reason(&ata.with_weight(PatternWeight::ByAge(vec![(12, 0.5), (12, 0.5)]))).0,
            Some(12)
        );
        // A cumulative pattern's last age selects no factor, so has no weight.
        let cdf = pattern(AgeToUltimate, &[(12, 2.0), (24, 1.2)])
            .with_weight(PatternWeight::ByAge(vec![(24, 0.5)]));
        assert_eq!(reason(&cdf).0, Some(24));
    }

    #[test]
    fn a_pattern_off_the_triangles_ages_is_read_on_its_own() {
        use PatternKind::*;
        // Factors to ultimate at 6, 18 and 30 months: 12 to 24 is read
        // between them, 24 to 36 ends past the pattern and is estimated.
        let r = pattern(AgeToUltimate, &[(6, 4.0), (18, 2.0), (30, 1.25)])
            .resolve(&AGES)
            .unwrap();
        close(&r.selected, &[Some(3.2f64.sqrt()), None, None]);
        assert_eq!(r.weight, [1.0, 0.0, 0.0]);
        assert_eq!(r.level.map(|l| l.0), Some(1));
        assert!((r.level.unwrap().1 - 2.5f64.sqrt()).abs() < 1e-15);
        assert_eq!(r.beyond, None);
        // Monthly factors from 1 to 59 months end at 60, past the oldest
        // age: each annual link is twelve of them, and the rest folds.
        let monthly: Vec<(Lag, f64)> = (1..=59).map(|t| (t, 1.01)).collect();
        let r = pattern(AgeToAge, &monthly).resolve(&AGES).unwrap();
        let year = 1.01f64.powi(12);
        close(&r.selected, &[Some(year), Some(year), Some(year)]);
        assert_eq!(r.level, None);
        let beyond = r.beyond.unwrap();
        assert_eq!(beyond.level, 1.0);
        assert!((beyond.ratio() - year).abs() < 1e-14);
        // A cumulative pattern past the oldest age keeps its level there.
        let r = pattern(PercentDeveloped, &[(12, 0.5), (36, 0.8), (60, 0.9)])
            .resolve(&AGES)
            .unwrap();
        assert_eq!(r.level.map(|l| l.0), Some(3));
        let beyond = r.beyond.unwrap();
        assert!((beyond.level - 1.0 / 0.9).abs() < 1e-15);
        assert!((r.level.unwrap().1 - beyond.oldest).abs() < 1e-15);
        // The errors of the off-grid reading.
        assert_eq!(
            reason(&pattern(AgeToUltimate, &[(13, 1.5), (23, 1.2)])),
            (Some(13), "the pattern selects no factor of the triangle")
        );
        assert_eq!(
            reason(&pattern(AgeToAge, &[(18, 2.0), (18, 2.0), (30, 1.0)])),
            (Some(18), "the age is given twice")
        );
        assert_eq!(
            reason(&pattern(AgeToAge, &[(18, 2.0), (30, f64::NAN)])),
            (Some(30), "an age-to-age factor must be finite and positive")
        );
    }

    #[test]
    fn averaging_over_origins_spreads_each_period_over_its_months() {
        // A pattern by accident quarter averaged to accident half years:
        // P(a) = (p(a) + p(a - 3)) / 2, with p linear from 0 below its first
        // age and held at 1 past its end.
        let quarter = pattern(
            PatternKind::PercentDeveloped,
            &[(3, 0.3), (6, 0.6), (9, 1.0)],
        );
        let half = quarter
            .averaged_over_origins(Grain::Quarter, Grain::Semester)
            .unwrap();
        assert_eq!(half.values.first().unwrap().0, 3);
        assert_eq!(half.values.last(), Some(&(12, 1.0)));
        let at = |age: Lag| half.values.iter().find(|v| v.0 == age).unwrap().1;
        assert!((at(3) - 0.15).abs() < 1e-15);
        assert!((at(6) - 0.45).abs() < 1e-15);
        assert!((at(9) - 0.8).abs() < 1e-15);
        // Between 6 and 9 months p is ln-linear: p(7) = 0.6^(2/3).
        let p7 = (0.6f64.ln() * 2.0 / 3.0).exp();
        assert!((at(10) - (1.0 + p7) / 2.0).abs() < 1e-15);
        assert_eq!(half.weight, PatternWeight::All(1.0));
    }
}

/// The pattern through the methods: the chain ladder and its tail, the
/// expected-loss family, Mack and Merz and Wüthrich, the bootstraps and
/// segments.
#[cfg(test)]
mod fits {
    use super::*;
    use crate::triangle::tests::{annual, genins, raa};
    use crate::triangle::{DevelopmentColumn, Long};
    use crate::{
        BornhuetterFerguson, ChainLadder, Development, Grain, Mack, MackBootstrap, MackProcess,
        OdpBootstrap, OneYearMethod, OneYearPattern, ProcessDistribution, Tail, TailConstant,
        Triangle,
    };
    use prospicio_core::Month;
    use prospicio_prob::Distribution;

    fn close(got: f64, want: f64, rel: f64) {
        assert!(
            (got - want).abs() <= rel * want.abs().max(1e-300),
            "got {got}, want {want}"
        );
    }

    fn with(pattern: SelectedPattern) -> Development {
        Development {
            pattern: Some(pattern),
            ..Default::default()
        }
    }

    fn chain_ladder(pattern: SelectedPattern, tail: Tail) -> ChainLadder {
        ChainLadder {
            development: with(pattern),
            tail,
        }
    }

    /// RAA's own estimated factors, as an age-to-age pattern with `weight`.
    fn own(weight: PatternWeight) -> SelectedPattern {
        let dev = Development::default().fit(&raa(), "values").unwrap();
        SelectedPattern::new(
            PatternKind::AgeToAge,
            dev.development.iter().copied().zip(dev.ldf).collect(),
        )
        .with_weight(weight)
    }

    /// Three ages: 100 : 150 : 165 for the oldest origin.
    fn small() -> Triangle {
        annual(2020, &[&[100.0, 150.0, 165.0], &[110.0, 170.0], &[120.0]])
    }

    #[test]
    fn percent_developed_with_a_tail_is_rescaled_to_100_percent() {
        // p = (0.5, 0.8, 0.95), tail 1.02: links 1.6 and 1.1875, and
        // pattern_rescale = 1 / (0.95 * 1.02).
        let p = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            vec![(12, 0.5), (24, 0.8), (36, 0.95)],
        );
        let cl = chain_ladder(p, 1.02.into())
            .fit(&small(), "values")
            .unwrap();
        close(cl.ldf()[0], 1.6, 1e-15);
        close(cl.ldf()[1], 1.1875, 1e-15);
        let rescale = cl.pattern_rescale.unwrap();
        close(rescale, 1.0 / (0.95 * 1.02), 1e-15);
        close(0.95 * rescale * cl.tail.factor, 1.0, 1e-15);
        close(cl.ultimate[2], 120.0 * 1.6 * 1.1875 * 1.02, 1e-15);
        // The estimate is kept, and the youngest origin's ultimate is its
        // latest value over the rescaled percent developed.
        close(cl.development.ldf[1], 165.0 / 150.0, 1e-15);
        close(cl.ultimate[2], 120.0 / (0.5 * rescale), 1e-14);
        // CDFs rescale the same way: divided by the factor.
        let cdf = SelectedPattern::new(
            PatternKind::AgeToUltimate,
            vec![(12, 2.0), (24, 1.25), (36, 1.0 / 0.95)],
        );
        let by_cdf = chain_ladder(cdf, 1.02.into())
            .fit(&small(), "values")
            .unwrap();
        close(by_cdf.pattern_rescale.unwrap(), rescale, 1e-14);
        close(by_cdf.ultimate[2], cl.ultimate[2], 1e-14);
    }

    #[test]
    fn age_to_age_pattern_stops_and_the_tail_multiplies() {
        let p = SelectedPattern::new(PatternKind::AgeToAge, vec![(12, 1.6), (24, 1.1)]);
        let cl = chain_ladder(p, 1.05.into())
            .fit(&small(), "values")
            .unwrap();
        assert_eq!(cl.pattern_rescale, None);
        close(cl.ultimate[2], 120.0 * 1.6 * 1.1 * 1.05, 1e-15);
        close(cl.cdf[2], 1.05, 1e-15);
    }

    #[test]
    fn partial_patterns_keep_the_estimate_elsewhere() {
        let t = raa();
        let base = ChainLadder::default().fit(&t, "values").unwrap();
        // Young end: CDFs at 12, 24 and 36 select 12-24 and 24-36; the
        // rest is estimated, and the level at 36 is rescaled to the
        // estimated factors from 36 on.
        let young = SelectedPattern::new(
            PatternKind::AgeToUltimate,
            vec![(12, 9.0), (24, 3.0), (36, 1.8)],
        );
        let cl = chain_ladder(young, Tail::default())
            .fit(&t, "values")
            .unwrap();
        close(cl.ldf()[0], 3.0, 1e-15);
        close(cl.ldf()[1], 3.0 / 1.8, 1e-15);
        assert_eq!(&cl.ldf()[2..], &base.ldf()[2..]);
        close(cl.pattern_rescale.unwrap(), 1.8 / base.cdf[2], 1e-14);
        assert_eq!(cl.ultimate[..8], base.ultimate[..8]);
        // Old end: an age-to-age factor at 96 only.
        let old = SelectedPattern::new(PatternKind::AgeToAge, vec![(96, 1.0)]);
        let cl = chain_ladder(old, Tail::default())
            .fit(&t, "values")
            .unwrap();
        assert_eq!(&cl.ldf()[..7], &base.ldf()[..7]);
        assert_eq!(cl.ldf()[7], 1.0);
        assert_eq!(cl.ldf()[8], base.ldf()[8]);
    }

    #[test]
    fn blend_limits_through_the_chain_ladder() {
        let t = raa();
        let selected = SelectedPattern::new(PatternKind::AgeToAge, vec![(12, 2.5), (24, 1.5)]);
        let at = |w: f64| {
            chain_ladder(
                selected.clone().with_weight(PatternWeight::All(w)),
                Tail::default(),
            )
            .fit(&t, "values")
            .unwrap()
        };
        let base = ChainLadder::default().fit(&t, "values").unwrap();
        assert_eq!(at(0.0).ultimate, base.ultimate);
        assert_eq!(&at(1.0).ldf()[..2], &[2.5, 1.5]);
        let half = at(0.5);
        close(half.ldf()[0], 0.5 * 2.5 + 0.5 * base.ldf()[0], 1e-15);
        assert_eq!(half.development.ldf, base.development.ldf);
        assert_eq!(half.development.blended_ldf(), half.ldf());
    }

    #[test]
    fn pattern_and_tail_cannot_overlap() {
        let t = raa();
        let tail = Tail::Constant(TailConstant {
            factor: 1.05,
            attachment_age: Some(72),
            ..Default::default()
        });
        // A selected factor at the attachment age is rejected.
        let at = SelectedPattern::new(PatternKind::AgeToAge, vec![(72, 1.02)]);
        assert_eq!(
            chain_ladder(at, tail).fit(&t, "values").unwrap_err(),
            Error::Pattern {
                age: Some(72),
                reason: "the factor is selected and replaced by the tail: \
                         select factors before the tail's attachment age only",
            }
        );
        // A cumulative pattern may end there: its last age selects nothing,
        // and its level is rescaled to the tail's factor.
        let to = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            vec![(48, 0.85), (60, 0.9), (72, 0.95)],
        );
        let cl = chain_ladder(to, tail).fit(&t, "values").unwrap();
        close(cl.pattern_rescale.unwrap(), 1.0 / (0.95 * cl.cdf[5]), 1e-14);
        close(cl.cdf[5], 1.05, 1e-12);
    }

    #[test]
    fn an_estimated_tail_extrapolates_the_blended_factors() {
        let t = raa();
        let p = SelectedPattern::new(PatternKind::AgeToAge, vec![(84, 1.05), (96, 1.04)]);
        let cl = chain_ladder(p, Tail::LogLinear).fit(&t, "values").unwrap();
        let mut blended = cl.development.ldf.clone();
        blended[6] = 1.05;
        blended[7] = 1.04;
        let refit = Tail::LogLinear
            .select(&blended, &cl.development.development)
            .unwrap();
        close(cl.tail.factor, refit.2, 1e-15);
        // Its sigma and standard error are the estimate's lines, read at
        // the fitted factor.
        let plain = Tail::LogLinear
            .fit(&Development::default().fit(&t, "values").unwrap())
            .unwrap();
        assert!(cl.tail.factor > plain.factor);
        assert!(cl.tail.std_err.is_finite() && cl.tail.sigma.is_finite());
    }

    #[test]
    fn expected_loss_methods_take_the_pattern() {
        let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
        let ages = [12, 24, 36, 12, 24, 12];
        let t = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[
                ("paid", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0]),
                ("premium", &[200.0, 200.0, 200.0, 220.0, 220.0, 240.0]),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let p = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            vec![(12, 0.4), (24, 0.8), (36, 1.0)],
        );
        let bf = BornhuetterFerguson {
            apriori: 0.75,
            chain_ladder: chain_ladder(p, Tail::default()),
        }
        .fit(&t, "paid", "premium")
        .unwrap();
        // 2022 is 40% developed: 120 + 0.6 * 0.75 * 240.
        close(bf.ultimate[2], 120.0 + 0.6 * 0.75 * 240.0, 1e-14);
        close(bf.ultimate[1], 170.0 + 0.2 * 0.75 * 220.0, 1e-14);
    }

    #[test]
    fn mack_with_a_fully_selected_pattern_has_process_risk_only() {
        let t = raa();
        let base = Mack::default().fit(&t, "values").unwrap();
        let full = Mack {
            development: with(own(PatternWeight::All(1.0))),
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap();
        assert!(full.parameter_risk.iter().all(|&p| p == 0.0));
        assert_eq!(full.total_parameter_risk, 0.0);
        for (a, b) in full.process_risk.iter().zip(&base.process_risk) {
            close(*a, *b, 1e-12);
        }
        close(full.total_process_risk, base.total_process_risk, 1e-12);
        close(full.total_standard_error, base.total_process_risk, 1e-12);
        // A uniform weight scales the parameter risk by 1 - w.
        let blended = Mack {
            development: with(own(PatternWeight::All(0.4))),
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap();
        for (a, b) in blended.parameter_risk.iter().zip(&base.parameter_risk) {
            close(*a, 0.6 * b, 1e-12);
        }
        close(
            blended.total_parameter_risk,
            0.6 * base.total_parameter_risk,
            1e-12,
        );
    }

    #[test]
    fn mack_matches_the_closed_form_for_a_partial_pattern() {
        // mse(R_i) = U_i^2 sum_k (sigma_k^2 / f_k^2) (1 / C_ik + g_k^2 / S_k)
        // with the used factors, computed apart from the recursion; the
        // total adds 2 U_i U_j sum_k g_k^2 sigma_k^2 / (f_k^2 S_k) over the
        // factors both origins still develop through.
        let t = genins();
        let p = SelectedPattern::new(
            PatternKind::AgeToAge,
            vec![(12, 3.3), (36, 1.4), (60, 1.12)],
        )
        .with_weight(PatternWeight::ByAge(vec![(12, 0.3), (60, 0.8)]));
        let mack = Mack {
            development: with(p),
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap();
        let cl = &mack.chain_ladder;
        let dev = &cl.development;
        let f = cl.ldf();
        let g: Vec<f64> = dev.weight.iter().map(|w| 1.0 - w).collect();
        close(g[0], 0.7, 1e-15);
        assert_eq!((g[1], g[2], g[3]), (1.0, 0.0, 1.0));
        close(g[4], 0.2, 1e-14);
        let n = cl.origins.len();
        let mut shared = vec![0.0; f.len()];
        let mut mse = vec![0.0; n];
        for k in 0..f.len() {
            shared[k] = (dev.sigma[k] / f[k]).powi(2) * g[k] * g[k] / dev.volume[k];
        }
        for (o, m) in mse.iter_mut().enumerate() {
            let mut c = cl.latest[o];
            for k in cl.latest_position[o]..f.len() {
                let s2 = (dev.sigma[k] / f[k]).powi(2);
                *m += s2 * (1.0 / c + g[k] * g[k] / dev.volume[k]);
                c *= f[k];
            }
            *m *= cl.ultimate[o].powi(2);
            close(mack.standard_error[o], m.sqrt(), 1e-10);
        }
        let mut total: f64 = mse.iter().sum();
        for a in 0..n {
            for b in a + 1..n {
                let start = cl.latest_position[a].max(cl.latest_position[b]);
                let common: f64 = shared[start..].iter().sum();
                total += 2.0 * cl.ultimate[a] * cl.ultimate[b] * common;
            }
        }
        close(mack.total_standard_error, total.sqrt(), 1e-10);
    }

    #[test]
    fn merz_wuthrich_at_the_limits() {
        let t = raa();
        let base = Mack::default().fit(&t, "values").unwrap();
        let cdr = base.claims_development_result().unwrap();
        // w = 0 is Merz and Wüthrich.
        let zero = Mack {
            development: with(own(PatternWeight::All(0.0))),
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap()
        .claims_development_result()
        .unwrap();
        for (a, b) in zero
            .by_calendar_year
            .iter()
            .flatten()
            .zip(cdr.by_calendar_year.iter().flatten())
        {
            close(*a, *b, 1e-12);
        }
        // w = 1: process error only, no covariance between origins, and the
        // run-off adds up to Mack's process risk.
        let full = Mack {
            development: with(own(PatternWeight::All(1.0))),
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap();
        let one = full.claims_development_result().unwrap();
        for (year, total) in one.by_calendar_year.iter().zip(&one.total_by_calendar_year) {
            close(total.powi(2), year.iter().map(|s| s * s).sum(), 1e-12);
        }
        for (a, b) in one.run_off_standard_error().iter().zip(&full.process_risk) {
            close(*a, *b, 1e-10);
        }
        close(
            one.total_run_off_standard_error(),
            full.total_process_risk,
            1e-10,
        );
        // Between, every origin's one-year risk lies between the two.
        let half = Mack {
            development: with(own(PatternWeight::All(0.5))),
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap()
        .claims_development_result()
        .unwrap();
        for o in 1..10 {
            let h = half.one_year_standard_error[o];
            assert!(one.one_year_standard_error[o] < h && h < cdr.one_year_standard_error[o]);
        }
    }

    /// Four origins exactly on the pattern 1 : 2 : 3 : 3.75.
    fn exact() -> Triangle {
        annual(
            2020,
            &[
                &[4.0, 8.0, 12.0, 15.0],
                &[8.0, 16.0, 24.0],
                &[12.0, 24.0],
                &[16.0],
            ],
        )
    }

    #[test]
    fn odp_bootstrap_blends_every_simulation() {
        // Zero residuals: every pseudo triangle is the fitted one, so every
        // simulation's reserve is the deterministic chain ladder's with the
        // pattern blended in.
        let p = SelectedPattern::new(PatternKind::AgeToAge, vec![(12, 2.5), (24, 1.2)])
            .with_weight(PatternWeight::ByAge(vec![(24, 0.5)]));
        let boot = OdpBootstrap {
            n_sims: 50,
            seed: 1,
            pattern: Some(p.clone()),
            ..Default::default()
        };
        let fit = boot.fit(&exact(), "values").unwrap();
        let reserves = fit.chain_ladder.reserves();
        close(fit.chain_ladder.ldf()[1], 0.5 * 1.2 + 0.5 * 1.5, 1e-15);
        for row in fit.reserves.draw_matrix().chunks(4) {
            for (got, want) in row.iter().zip(&reserves) {
                assert!((got - want).abs() < 1e-12, "{row:?} vs {reserves:?}");
            }
        }
        // The one-year view re-reserves with the same pattern.
        let method = OneYearMethod::ChainLadder(chain_ladder(p, Tail::default()));
        let one = boot.one_year(&exact(), "values", &method).unwrap();
        assert!(one.bootstrap.chain_ladder.development.has_pattern());
        assert_eq!(one.opening_ultimate, fit.chain_ladder.ultimate);
    }

    #[test]
    fn odp_bootstrap_keeps_the_levels_error_only() {
        // With every factor selected (at the estimate) and no process error,
        // what remains is the pseudo latest values' error: narrower than
        // without the pattern (on RAA 73% of its standard deviation, seed
        // 5), centred on the chain ladder.
        let t = raa();
        let boot = |pattern| OdpBootstrap {
            n_sims: 2_000,
            seed: 5,
            process: ProcessDistribution::None,
            pattern,
            ..Default::default()
        };
        let free = boot(None).fit(&t, "values").unwrap();
        let fixed = boot(Some(own(PatternWeight::All(1.0))))
            .fit(&t, "values")
            .unwrap();
        let (sd_fixed, sd_free) = (
            fixed.reserves.total().std_dev(),
            free.reserves.total().std_dev(),
        );
        assert!(sd_fixed < 0.8 * sd_free, "{sd_fixed} vs {sd_free}");
        let cl = fixed.chain_ladder.total_reserve();
        close(fixed.reserves.total().mean(), cl, 0.03);
    }

    #[test]
    fn mack_bootstrap_with_a_fully_selected_pattern() {
        let t = raa();
        let full = own(PatternWeight::All(1.0));
        let boot = MackBootstrap {
            n_sims: 200,
            seed: 3,
            process: MackProcess::None,
            development: with(full.clone()),
            ..Default::default()
        };
        // No process error and no parameter error: the run-off is the
        // deterministic reserve in every simulation.
        let fit = boot.fit(&t, "values").unwrap();
        let reserves = fit.mack.chain_ladder.reserves();
        for row in fit.reserves.draw_matrix().chunks(10) {
            for (got, want) in row.iter().zip(&reserves) {
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                    "{got} vs {want}"
                );
            }
        }
        // The one-year view, re-reserved with the same pattern: the year
        // develops as the pattern says, so every CDR is 0.
        let method = OneYearMethod::ChainLadder(chain_ladder(full.clone(), Tail::default()));
        let one = boot.one_year(&t, "values", &method).unwrap();
        assert!(one.cdr.draw_matrix().iter().all(|x| x.abs() < 1e-8));
        // With Mack's process its standard deviation is Mack's process risk
        // (exactly so in expectation with volume weighting).
        let n_sims = 10_000;
        let fit = MackBootstrap {
            n_sims,
            seed: 3,
            process: MackProcess::Normal,
            development: with(full),
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap();
        let totals: Vec<f64> = fit
            .reserves
            .draw_matrix()
            .chunks(10)
            .map(|r| r.iter().sum())
            .collect();
        let n = n_sims as f64;
        let mean = totals.iter().sum::<f64>() / n;
        let m2 = totals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let m4 = totals.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
        let sd = m2.sqrt();
        let error = sd * ((m4 / (m2 * m2) - 1.0) / (4.0 * n)).sqrt();
        let want = fit.mack.total_process_risk;
        assert!(
            (sd - want).abs() < 5.0 * error,
            "sd {sd} vs Mack's process risk {want} (Monte Carlo se {error})"
        );
    }

    #[test]
    fn a_uniform_blend_scales_the_parameter_spread() {
        // Selecting the estimate itself with weight w leaves the factors as
        // they are and scales each pseudo factor's deviation by 1 - w.
        let t = raa();
        let sd = |w: f64| {
            MackBootstrap {
                n_sims: 3_000,
                seed: 9,
                process: MackProcess::None,
                development: with(own(PatternWeight::All(w))),
                ..Default::default()
            }
            .fit(&t, "values")
            .unwrap()
            .reserves
            .total()
            .std_dev()
        };
        let ratio = sd(0.6) / sd(0.0);
        assert!((ratio - 0.4).abs() < 0.02, "{ratio}");
    }

    #[test]
    fn one_year_bootstrap_pattern_must_be_the_methods() {
        let t = raa();
        let p = own(PatternWeight::All(1.0));
        let plain = OneYearMethod::ChainLadder(ChainLadder::default());
        let patterned = OneYearMethod::ChainLadder(chain_ladder(p.clone(), Tail::default()));
        let want =
            Error::Bootstrap("the bootstrap's pattern must be the re-reserving method's pattern");
        let odp = OdpBootstrap {
            n_sims: 10,
            pattern: Some(p.clone()),
            ..Default::default()
        };
        assert_eq!(odp.one_year(&t, "values", &plain).unwrap_err(), want);
        let mack = MackBootstrap {
            n_sims: 10,
            development: with(p),
            ..Default::default()
        };
        assert_eq!(mack.one_year(&t, "values", &plain).unwrap_err(), want);
        // A method with a pattern and a bootstrap without one is allowed: the
        // bootstrap takes the method's (OneYearPattern::Method).
        let free = OdpBootstrap {
            n_sims: 10,
            ..Default::default()
        };
        assert!(free.one_year(&t, "values", &patterned).is_ok());
    }

    #[test]
    fn one_year_bootstrap_pattern_is_compared_by_what_it_selects() {
        // The same selection given in another order, as factors to ultimate
        // or percents developed, or with the weight in another form, is the
        // method's pattern; another factor or weight is not.
        let t = raa();
        let base = own(PatternWeight::All(1.0));
        let mut cdf: Vec<(Lag, f64)> = Vec::new();
        let mut c = 1.0;
        for &(age, f) in base.values.iter().rev() {
            c *= f;
            cdf.push((age, c));
        }
        // The level at 120 months: 1, so the pattern selects every factor.
        cdf.push((120, 1.0));
        let percent: Vec<(Lag, f64)> = cdf.iter().map(|&(a, v)| (a, 1.0 / v)).collect();
        let method = OneYearMethod::ChainLadder(chain_ladder(base.clone(), Tail::default()));
        let run = |p: SelectedPattern| {
            OdpBootstrap {
                n_sims: 10,
                pattern: Some(p),
                ..Default::default()
            }
            .one_year(&t, "values", &method)
        };
        let mut reversed = base.clone();
        reversed.values.reverse();
        let by_age = base
            .clone()
            .with_weight(PatternWeight::ByAge(vec![(36, 1.0), (12, 1.0)]));
        let same = [
            reversed,
            by_age,
            SelectedPattern::new(PatternKind::AgeToUltimate, cdf),
            SelectedPattern::new(PatternKind::PercentDeveloped, percent),
        ];
        for p in same {
            assert!(run(p.clone()).is_ok(), "{p:?}");
        }
        let want =
            Error::Bootstrap("the bootstrap's pattern must be the re-reserving method's pattern");
        let mut other = base.clone();
        other.values[0].1 *= 1.01;
        assert_eq!(run(other).unwrap_err(), want);
        let blended = base.clone().with_weight(PatternWeight::All(0.5));
        assert_eq!(run(blended).unwrap_err(), want);
        let partial = SelectedPattern::new(PatternKind::AgeToAge, base.values[..3].to_vec());
        assert_eq!(run(partial).unwrap_err(), want);
        // Mack's bootstrap compares the same way.
        let mut reversed = base;
        reversed.values.reverse();
        let mack = MackBootstrap {
            n_sims: 10,
            development: with(reversed),
            ..Default::default()
        };
        assert!(mack.one_year(&t, "values", &method).is_ok());
    }

    #[test]
    fn segments_check_the_pattern_against_each_and_report_it() {
        let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
        let ages = [12, 24, 36, 12, 24, 12];
        let paid = [100.0, 150.0, 165.0, 110.0, 170.0, 120.0];
        let t = Triangle::from_long(&Long {
            keys: &[("lob", &[["Auto"; 6], ["Home"; 6]].concat())],
            origin: &[origin, origin].concat(),
            development: DevelopmentColumn::Age(&[ages, ages].concat()),
            values: &[("paid", &[paid, paid.map(|v| v * 2.0)].concat())],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let p = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            vec![(12, 0.5), (24, 0.8), (36, 0.95)],
        );
        let fits = chain_ladder(p, Tail::default())
            .fit_segments(&t, "paid")
            .unwrap();
        let table = fits.development_table();
        assert_eq!(
            table
                .values
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            [
                "ldf",
                "cdf",
                "sigma",
                "std_err",
                "estimated_ldf",
                "selected_ldf",
                "ldf_weight",
                "blended_ldf",
                "estimated_std_err"
            ]
        );
        assert_eq!(table.column("ldf_weight").unwrap()[..2], [1.0, 1.0]);
        assert_eq!(table.column("std_err").unwrap()[..2], [0.0, 0.0]);
        let totals = fits.totals();
        // 95% at the oldest age with no tail: the level is the tail.
        let rescale = totals.column("pattern_rescale").unwrap();
        close(rescale[0], 1.0, 1e-15);
        close(rescale[1], 1.0, 1e-15);
        let implied = totals.column("pattern_tail").unwrap();
        close(implied[0], 1.0 / 0.95, 1e-15);
        close(implied[1], 1.0 / 0.95, 1e-15);
        close(totals.column("tail").unwrap()[0], 1.0 / 0.95, 1e-14);
        // Without a pattern the tables are as before.
        let plain = ChainLadder::default().fit_segments(&t, "paid").unwrap();
        assert_eq!(plain.development_table().values.len(), 4);
        assert!(plain.totals().column("pattern_rescale").is_none());
        assert!(plain.totals().column("pattern_tail").is_none());
        // An age the segment does not have names the segment.
        let bad = SelectedPattern::new(PatternKind::AgeToAge, vec![(48, 1.1)]);
        assert_eq!(
            chain_ladder(bad, Tail::default())
                .fit_segments(&t, "paid")
                .unwrap_err(),
            Error::InSegment {
                label: "Auto".into(),
                source: Box::new(Error::Pattern {
                    age: Some(48),
                    reason: "not a development age of the triangle"
                })
            }
        );
    }

    /// RAA's own factors as factors to ultimate, with `level` at the
    /// oldest age.
    fn own_cdf(level: f64) -> SelectedPattern {
        let base = own(PatternWeight::All(1.0));
        let mut values = vec![(120, level)];
        let mut c = level;
        for &(age, f) in base.values.iter().rev() {
            c *= f;
            values.push((age, c));
        }
        SelectedPattern::new(PatternKind::AgeToUltimate, values)
    }

    #[test]
    fn a_cumulative_pattern_without_a_tail_keeps_its_level_as_the_tail() {
        // 50%, 80% and 95% at 12, 24 and 36 months, the oldest age: the 5%
        // left is the tail, 1 / 0.95.
        let p = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            vec![(12, 0.5), (24, 0.8), (36, 0.95)],
        );
        let cl = chain_ladder(p.clone(), Tail::default())
            .fit(&small(), "values")
            .unwrap();
        close(cl.pattern_tail.unwrap(), 1.0 / 0.95, 1e-15);
        close(cl.tail.factor, 1.0 / 0.95, 1e-14);
        close(cl.pattern_rescale.unwrap(), 1.0, 1e-14);
        assert_eq!(cl.tail.std_err, 0.0);
        close(cl.ultimate[2], 120.0 / 0.5, 1e-14);
        close(cl.ultimate[0], 165.0 / 0.95, 1e-14);
        // It is the constant tail of that factor, but with no parameter
        // error.
        let constant = chain_ladder(p.clone(), (1.0 / 0.95).into())
            .fit(&small(), "values")
            .unwrap();
        assert_eq!(constant.pattern_tail, None);
        close(constant.pattern_rescale.unwrap(), 1.0, 1e-14);
        assert_eq!(constant.ultimate, cl.ultimate);
        assert_eq!(constant.tail.sigma.to_bits(), cl.tail.sigma.to_bits());
        // With a tail the level is rescaled to it, as before.
        let tailed = chain_ladder(p, 1.02.into())
            .fit(&small(), "values")
            .unwrap();
        assert_eq!(tailed.pattern_tail, None);
        close(tailed.tail.factor, 1.02, 1e-14);
        // A pattern that stops before the oldest age implies no tail: its
        // level is rescaled to the estimated factors after it.
        let young = SelectedPattern::new(PatternKind::AgeToUltimate, vec![(12, 9.0), (24, 3.0)]);
        let cl = chain_ladder(young, Tail::default())
            .fit(&raa(), "values")
            .unwrap();
        assert_eq!(cl.pattern_tail, None);
        assert_eq!(cl.tail.factor, 1.0);
        // An age-to-age pattern implies none either.
        let ata = chain_ladder(own(PatternWeight::All(1.0)), Tail::default())
            .fit(&raa(), "values")
            .unwrap();
        assert_eq!((ata.pattern_tail, ata.tail.factor), (None, 1.0));
        // 100% at the oldest age implies a tail of 1: no tail.
        let full = chain_ladder(own_cdf(1.0), Tail::default())
            .fit(&raa(), "values")
            .unwrap();
        assert_eq!(full.pattern_tail, Some(1.0));
        assert_eq!(full.ultimate, ata.ultimate);
    }

    #[test]
    fn mack_takes_the_implied_tail_without_parameter_error() {
        let t = raa();
        let implied = Mack {
            development: with(own_cdf(1.05)),
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap();
        // The same as the constant tail with a standard error of 0: sigma
        // extrapolated, no parameter error anywhere.
        let constant = Mack {
            development: with(own(PatternWeight::All(1.0))),
            tail: 1.05.into(),
            tail_std_err: Some(0.0),
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap();
        assert_eq!(implied.chain_ladder.pattern_tail, Some(1.05));
        assert_eq!(implied.chain_ladder.tail.std_err, 0.0);
        close(
            implied.chain_ladder.tail.sigma,
            constant.chain_ladder.tail.sigma,
            1e-12,
        );
        assert!(implied.chain_ladder.tail.sigma > 0.0);
        assert_eq!(implied.total_parameter_risk, 0.0);
        for (a, b) in implied.process_risk.iter().zip(&constant.process_risk) {
            close(*a, *b, 1e-10);
        }
        close(
            implied.total_standard_error,
            constant.total_standard_error,
            1e-10,
        );
        // A tail standard error from the user applies.
        let given = Mack {
            development: with(own_cdf(1.05)),
            tail_std_err: Some(0.01),
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap();
        assert_eq!(given.chain_ladder.tail.std_err, 0.01);
        assert!(given.total_parameter_risk > 0.0);
        // Merz and Wüthrich's closed form has no tail.
        assert_eq!(
            implied.claims_development_result().unwrap_err(),
            Error::ClaimsDevelopment("needs no tail factor")
        );
    }

    #[test]
    fn bootstraps_keep_the_implied_tail_fixed() {
        // Exact factors 2, 1.5 and 1.25, 90% developed at the oldest age:
        // no residuals, so every ODP simulation's reserve is the chain
        // ladder's with the tail 1 / 0.9.
        let p = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            vec![(12, 0.24), (24, 0.48), (36, 0.72), (48, 0.9)],
        );
        let fit = OdpBootstrap {
            n_sims: 50,
            seed: 1,
            pattern: Some(p.clone()),
            ..Default::default()
        }
        .fit(&exact(), "values")
        .unwrap();
        close(fit.chain_ladder.tail.factor, 1.0 / 0.9, 1e-14);
        let reserves = fit.chain_ladder.reserves();
        close(reserves[0], 15.0 / 0.9 - 15.0, 1e-12);
        for row in fit.reserves.draw_matrix().chunks(4) {
            for (got, want) in row.iter().zip(&reserves) {
                assert!((got - want).abs() < 1e-9, "{row:?} vs {reserves:?}");
            }
        }
        // Mack's bootstrap with every factor selected and no process error:
        // the deterministic reserve, tail included, in every simulation.
        let t = raa();
        let fit = MackBootstrap {
            n_sims: 100,
            seed: 3,
            process: MackProcess::None,
            development: with(own_cdf(1.05)),
            ..Default::default()
        }
        .fit(&t, "values")
        .unwrap();
        let reserves = fit.mack.chain_ladder.reserves();
        let free = ChainLadder::default().fit(&t, "values").unwrap();
        close(reserves[0], free.latest[0] * 0.05, 1e-12);
        for row in fit.reserves.draw_matrix().chunks(10) {
            for (got, want) in row.iter().zip(&reserves) {
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                    "{got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn one_year_views_share_the_implied_tail() {
        let p = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            vec![(12, 0.24), (24, 0.48), (36, 0.72), (48, 0.9)],
        );
        let method = OneYearMethod::ChainLadder(chain_ladder(p.clone(), Tail::default()));
        // The bootstrap takes the method's pattern and so its tail; on the
        // exact triangle the year develops as projected, the oldest origin
        // takes its tail step and closes, and every CDR is 0.
        let boot = OdpBootstrap {
            n_sims: 20,
            seed: 2,
            ..Default::default()
        };
        let one = boot.one_year(&exact(), "values", &method).unwrap();
        assert_eq!(one.bootstrap.chain_ladder.pattern_tail, Some(1.0 / 0.9));
        assert!(one.cdr.draw_matrix().iter().all(|x| x.abs() < 1e-9));
        // A method with a tail of its own rescales the pattern's level to
        // it, and the bootstrap that takes the pattern reads it the same
        // way: no implied tail. A bootstrap's own pattern implies its tail,
        // which is not the method's.
        let other = OneYearMethod::ChainLadder(chain_ladder(p.clone(), 1.02.into()));
        let taken = boot.one_year(&exact(), "values", &other).unwrap();
        assert_eq!(taken.bootstrap.chain_ladder.pattern_tail, None);
        let own = OdpBootstrap {
            pattern: Some(p.clone()),
            ..boot.clone()
        };
        assert_eq!(
            own.one_year(&exact(), "values", &other).unwrap_err(),
            Error::Bootstrap(
                "the tail the bootstrap's pattern implies must be the re-reserving method's tail"
            )
        );
        // The same factor as a constant tail is the same tail.
        let constant = OneYearMethod::ChainLadder(chain_ladder(p, (1.0 / 0.9).into()));
        let same = boot.one_year(&exact(), "values", &constant).unwrap();
        assert_eq!(same.cdr.draw_matrix(), one.cdr.draw_matrix());
        // Simulating from the estimated factors, the bootstrap has no tail
        // and the method's is allowed.
        let estimated = OdpBootstrap {
            one_year_pattern: OneYearPattern::Estimated,
            ..boot
        };
        assert!(estimated.one_year(&exact(), "values", &other).is_ok());
    }

    #[test]
    fn a_taken_pattern_is_read_as_the_method_with_a_tail_reads_it() {
        // 90% developed at the oldest age and a method tail of 1.05: the
        // method rescales the level to its tail, so the pattern it lends a
        // bootstrap without a tail implies none, in both bootstraps, and
        // the default one-year view runs.
        let p = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            vec![(12, 0.24), (24, 0.48), (36, 0.72), (48, 0.9)],
        );
        let method = OneYearMethod::ChainLadder(chain_ladder(p, 1.05.into()));
        let boot = OdpBootstrap {
            n_sims: 20,
            seed: 2,
            ..Default::default()
        };
        let taken = boot.one_year(&exact(), "values", &method).unwrap();
        assert_eq!(taken.bootstrap.chain_ladder.pattern_tail, None);
        assert_eq!(taken.bootstrap.chain_ladder.tail.factor, 1.0);
        // The same as a bootstrap given the selected factors age to age.
        let factors = SelectedPattern::new(
            PatternKind::AgeToAge,
            vec![(12, 2.0), (24, 1.5), (36, 1.25)],
        );
        let given = OdpBootstrap {
            pattern: Some(factors),
            ..boot.clone()
        }
        .one_year(&exact(), "values", &method)
        .unwrap();
        for (a, b) in taken.cdr.draw_matrix().iter().zip(given.cdr.draw_matrix()) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        assert!(boot.one_year_segments(&exact(), "values", &method).is_ok());
        // Mack's bootstrap on RAA, its own factors to ultimate with 95% at
        // the oldest age.
        let method = OneYearMethod::ChainLadder(chain_ladder(own_cdf(1.0 / 0.95), 1.05.into()));
        let mack = MackBootstrap {
            n_sims: 20,
            seed: 2,
            ..Default::default()
        };
        let taken = mack.one_year(&raa(), "values", &method).unwrap();
        assert_eq!(taken.bootstrap.mack.chain_ladder.pattern_tail, None);
        assert!(mack.one_year_segments(&raa(), "values", &method).is_ok());
        let own = MackBootstrap {
            development: with(own_cdf(1.0 / 0.95)),
            ..mack
        };
        assert_eq!(
            own.one_year(&raa(), "values", &method).unwrap_err(),
            Error::Bootstrap(
                "the tail the bootstrap's pattern implies must be the re-reserving method's tail"
            )
        );
    }

    #[test]
    fn a_bootstraps_tail_may_be_the_one_the_methods_pattern_implies() {
        // A method with no tail whose pattern implies 1 / 0.9, and a
        // bootstrap with that factor as its own constant tail: the same
        // tail, compared as each uses it.
        let p = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            vec![(12, 0.24), (24, 0.48), (36, 0.72), (48, 0.9)],
        );
        let method = OneYearMethod::ChainLadder(chain_ladder(p, Tail::default()));
        let tailed = OdpBootstrap {
            n_sims: 20,
            seed: 2,
            tail: (1.0 / 0.9).into(),
            ..Default::default()
        };
        let one = tailed.one_year(&exact(), "values", &method).unwrap();
        assert!(one.cdr.draw_matrix().iter().all(|x| x.abs() < 1e-9));
        assert!(
            tailed
                .one_year_segments(&exact(), "values", &method)
                .is_ok()
        );
        let mack = MackBootstrap {
            n_sims: 20,
            seed: 2,
            tail: (1.0 / 0.95).into(),
            ..Default::default()
        };
        let implied =
            OneYearMethod::ChainLadder(chain_ladder(own_cdf(1.0 / 0.95), Tail::default()));
        assert!(mack.one_year(&raa(), "values", &implied).is_ok());
        // Another factor is another tail.
        let other = OdpBootstrap {
            tail: 1.02.into(),
            ..tailed
        };
        assert_eq!(
            other.one_year(&exact(), "values", &method).unwrap_err(),
            Error::Bootstrap("the bootstrap's tail must be the re-reserving method's tail")
        );
        // A method with neither a tail nor a pattern has no tail.
        let plain = OneYearMethod::ChainLadder(ChainLadder::default());
        assert_eq!(
            other.one_year(&exact(), "values", &plain).unwrap_err(),
            Error::Bootstrap("the bootstrap's tail must be the re-reserving method's tail")
        );
    }

    #[test]
    fn the_provenance_names_the_settings_of_an_implied_tail() {
        // The implied tail's standard error (and Mack's sigma) change the
        // draws, so they change the provenance too.
        let p = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            vec![(12, 0.24), (24, 0.48), (36, 0.72), (48, 0.9)],
        );
        let odp = |tail_std_err| OdpBootstrap {
            n_sims: 20,
            seed: 2,
            pattern: Some(p.clone()),
            tail_std_err,
            ..Default::default()
        };
        let (a, b) = (odp(None), odp(Some(0.01)));
        let fa = a.fit(&exact(), "values").unwrap();
        let fb = b.fit(&exact(), "values").unwrap();
        assert_ne!(fa.reserves.draw_matrix(), fb.reserves.draw_matrix());
        assert_ne!(fa.reserves.provenance(), fb.reserves.provenance());
        let method = OneYearMethod::ChainLadder(chain_ladder(p.clone(), Tail::default()));
        let (oa, ob) = (
            a.one_year(&exact(), "values", &method).unwrap(),
            b.one_year(&exact(), "values", &method).unwrap(),
        );
        assert_ne!(oa.cdr.provenance(), ob.cdr.provenance());
        let mack = |tail_sigma, tail_std_err| MackBootstrap {
            n_sims: 20,
            seed: 2,
            development: with(own_cdf(1.05)),
            tail_sigma,
            tail_std_err,
            ..Default::default()
        };
        let t = raa();
        let base = mack(None, None).fit(&t, "values").unwrap();
        let sigma = mack(Some(0.1), None).fit(&t, "values").unwrap();
        let std_err = mack(None, Some(0.01)).fit(&t, "values").unwrap();
        assert_ne!(base.reserves.provenance(), sigma.reserves.provenance());
        assert_ne!(base.reserves.provenance(), std_err.reserves.provenance());
    }

    #[test]
    fn a_bootstrap_without_a_pattern_takes_the_methods_in_the_one_year_view() {
        let t = raa();
        let p = own(PatternWeight::All(0.5));
        let method = OneYearMethod::ChainLadder(chain_ladder(p.clone(), Tail::default()));
        let odp = |pattern: Option<SelectedPattern>, choice: OneYearPattern| {
            OdpBootstrap {
                n_sims: 200,
                seed: 4,
                pattern,
                one_year_pattern: choice,
                ..Default::default()
            }
            .one_year(&t, "values", &method)
        };
        let by_default = odp(None, OneYearPattern::Method).unwrap();
        let given = odp(Some(p.clone()), OneYearPattern::Method).unwrap();
        let estimated = odp(None, OneYearPattern::Estimated).unwrap();
        assert_eq!(by_default.cdr.draw_matrix(), given.cdr.draw_matrix());
        assert_ne!(by_default.cdr.draw_matrix(), estimated.cdr.draw_matrix());
        assert!(by_default.bootstrap.chain_ladder.development.has_pattern());
        let want = Error::Bootstrap(
            "a bootstrap with a pattern of its own cannot simulate from the estimated \
             factors: give it no pattern, or OneYearPattern::Method",
        );
        assert_eq!(
            odp(Some(p.clone()), OneYearPattern::Estimated).unwrap_err(),
            want
        );
        // Mack's bootstrap the same way.
        let mack = |development: Development, choice: OneYearPattern| {
            MackBootstrap {
                n_sims: 200,
                seed: 4,
                development,
                one_year_pattern: choice,
                ..Default::default()
            }
            .one_year(&t, "values", &method)
        };
        let by_default = mack(Development::default(), OneYearPattern::Method).unwrap();
        let given = mack(with(p.clone()), OneYearPattern::Method).unwrap();
        let estimated = mack(Development::default(), OneYearPattern::Estimated).unwrap();
        assert_eq!(by_default.cdr.draw_matrix(), given.cdr.draw_matrix());
        assert_ne!(by_default.cdr.draw_matrix(), estimated.cdr.draw_matrix());
        assert_eq!(mack(with(p), OneYearPattern::Estimated).unwrap_err(), want);
    }

    #[test]
    fn provenance_without_a_pattern_is_unchanged() {
        assert_eq!(
            format!("{:?}", Development::default()),
            "Development { average: Volume, sigma_interpolation: LogLinear }"
        );
        let patterned = format!("{:?}", with(own(PatternWeight::All(1.0))));
        assert!(patterned.contains("pattern: SelectedPattern"));
    }
}
