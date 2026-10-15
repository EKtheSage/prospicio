//! Selected development patterns: benchmark factors that replace, or blend
//! with, the factors estimated from the triangle
//! (`docs/design/reserving-v02.md`, decision 12).
//!
//! A [`SelectedPattern`] is given as age-to-age factors, factors to
//! ultimate or percents developed, by age. It is converted to age-to-age
//! factors on the triangle's development ages; each selected factor is then
//! blended with the triangle's estimate as `w * selected + (1 - w) *
//! estimated`, `w` its [`PatternWeight`]. Factors it does not select stay
//! estimated.
//!
//! A selected factor carries no parameter error: Mack's standard errors and
//! the bootstraps charge a blended factor `(1 - w)` times the estimate's
//! error, and a fully selected one none. A supplied pattern therefore lowers
//! the reported uncertainty, and a wrong benchmark is not reflected in it.

use prospicio_core::Lag;

use crate::error::{Error, Result};

/// How the values of a [`SelectedPattern`] are expressed. The user names
/// it: age-to-age factors and factors to ultimate are both at least 1 in
/// the usual case and cannot be told apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternKind {
    /// `f(a)`: the factor from age `a` to the next development age of the
    /// triangle. Any value that is finite and positive; below 1 is negative
    /// development.
    AgeToAge,
    /// `CDF(a)`: the factor from age `a` to ultimate; at least 1 and not
    /// increasing with age.
    AgeToUltimate,
    /// `p(a) = 1 / CDF(a)`: the share of the ultimate developed at age `a`;
    /// in `(0, 1]` and not decreasing with age.
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
///   [`PatternKind::PercentDeveloped`]) gives consecutive ages and selects
///   the factors between them, `CDF(a_k) / CDF(a_k+1)`. Its level at its
///   last age selects no factor: what develops after that age comes from
///   the estimated factors and the tail, and the chain ladder reports how
///   much the given level is rescaled to match them
///   ([`ChainLadderFit::pattern_rescale`](crate::ChainLadderFit::pattern_rescale)).
///
/// Factors the pattern does not select are estimated, as without one, and
/// the triangle must still estimate every factor (the estimate is reported
/// next to the selection and gives Mack's sigma). A selected factor at or
/// after the age where the [`Tail`](crate::Tail) takes over is an error.
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
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedPattern {
    /// How `values` are expressed.
    pub kind: PatternKind,
    /// `(age in months, value)`, in any order; each age a development age
    /// of the triangle.
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
                for (i, &(age, v)) in values.iter().enumerate() {
                    if percent {
                        if !(v.is_finite() && v > 0.0 && v <= 1.0) {
                            return err(Some(age), "a percent developed must be in (0, 1]");
                        }
                        if i > 0 && v < values[i - 1].1 {
                            return err(Some(age), "percents developed must not decrease with age");
                        }
                        cdf.push(1.0 / v);
                    } else {
                        if !(v.is_finite() && v >= 1.0) {
                            return err(
                                Some(age),
                                "a factor to ultimate must be finite and at least 1",
                            );
                        }
                        if i > 0 && v > values[i - 1].1 {
                            return err(
                                Some(age),
                                "factors to ultimate must not increase with age",
                            );
                        }
                        cdf.push(v);
                    }
                }
                if values.len() < 2 {
                    return err(
                        Some(values[0].0),
                        "a cumulative pattern needs two or more ages: one value selects no factor",
                    );
                }
                for (i, w) in positions.windows(2).enumerate() {
                    if w[1] != w[0] + 1 {
                        return err(
                            Some(ages[w[0] + 1]),
                            "a cumulative pattern must give every development age between its first and last",
                        );
                    }
                    selected[w[0]] = Some(cdf[i] / cdf[i + 1]);
                }
                let last = positions.len() - 1;
                level = Some((positions[last], cdf[last]));
            }
        }

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
        Ok(Resolved {
            selected,
            weight,
            level,
        })
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
    /// factor to ultimate there.
    pub(crate) level: Option<(usize, f64)>,
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
            reason(&pattern(AgeToUltimate, &[(12, 0.99), (24, 0.9)])).0,
            Some(12)
        );
        assert_eq!(
            reason(&pattern(AgeToUltimate, &[(12, 1.5), (24, 1.6)])).0,
            Some(24)
        );
        assert_eq!(
            reason(&pattern(PercentDeveloped, &[(12, 0.0), (24, 0.5)])).0,
            Some(12)
        );
        assert_eq!(
            reason(&pattern(PercentDeveloped, &[(12, 0.5), (24, 1.2)])).0,
            Some(24)
        );
        assert_eq!(
            reason(&pattern(PercentDeveloped, &[(12, 0.6), (24, 0.5)])).0,
            Some(24)
        );
        assert_eq!(reason(&pattern(PercentDeveloped, &[(24, 0.5)])).0, Some(24));
        // A gap names the missing age.
        assert_eq!(
            reason(&pattern(AgeToUltimate, &[(12, 2.0), (36, 1.2)])).0,
            Some(24)
        );
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
        OdpBootstrap, OneYearMethod, ProcessDistribution, Tail, TailConstant, Triangle,
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
        // A method with a pattern and a bootstrap without one is allowed.
        let free = OdpBootstrap {
            n_sims: 10,
            ..Default::default()
        };
        assert!(free.one_year(&t, "values", &patterned).is_ok());
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
        let rescale = totals.column("pattern_rescale").unwrap();
        close(rescale[0], 1.0 / 0.95, 1e-15);
        close(rescale[1], 1.0 / 0.95, 1e-15);
        // Without a pattern the tables are as before.
        let plain = ChainLadder::default().fit_segments(&t, "paid").unwrap();
        assert_eq!(plain.development_table().values.len(), 4);
        assert!(plain.totals().column("pattern_rescale").is_none());
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
