//! Tests of the chain ladder's assumptions: Mack's calendar-year effect
//! test and his test for correlation between adjacent development factors
//! (Mack 1994, "Measuring the variability of chain ladder reserve
//! estimates", appendices H and G), as R ChainLadder's `cyEffTest` and
//! `dfCorTest` compute them.
//!
//! Both run on a triangle's own link ratios ([`calendar_year_effect`],
//! [`development_correlation`]) or on the link ratios a Mack fit averaged
//! ([`MackFit::calendar_year_effect`], [`MackFit::development_correlation`]),
//! which leaves out those `n_periods`, `drop_high` and `drop_low` exclude
//! (`docs/design/reserving-v02.md`, decision 26).
//!
//! ```
//! use prospicio_reserving::{Triangle, calendar_year_effect, development_correlation};
//! # use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month};
//! # let rows: [&[f64]; 5] = [
//! #     &[100.0, 160.0, 180.0, 186.0, 188.0],
//! #     &[110.0, 170.0, 195.0, 200.0],
//! #     &[120.0, 200.0, 220.0],
//! #     &[130.0, 205.0],
//! #     &[140.0],
//! # ];
//! # let (mut origin, mut age, mut value) = (vec![], vec![], vec![]);
//! # for (i, row) in rows.iter().enumerate() {
//! #     for (d, v) in row.iter().enumerate() {
//! #         origin.push(Month::january(2020 + i as i32));
//! #         age.push(12 * (d as u32 + 1));
//! #         value.push(*v);
//! #     }
//! # }
//! # let tri = Triangle::from_long(&Long {
//! #     keys: &[],
//! #     origin: &origin,
//! #     development: DevelopmentColumn::Age(&age),
//! #     values: &[("paid", &value)],
//! #     origin_grain: Grain::Year,
//! #     development_grain: Grain::Year,
//! #     cumulative: true,
//! # })?;
//! let cy = calendar_year_effect(&tri, "paid", 0.95)?;
//! assert!(cy.lower <= cy.upper);
//! let dc = development_correlation(&tri, "paid", 0.5)?;
//! assert!(dc.variance > 0.0);
//! # Ok::<(), prospicio_reserving::Error>(())
//! ```

use std::collections::BTreeMap;

use prospicio_core::{Lag, Month};
use prospicio_math::special::{ln_gamma, norm_quantile};

use crate::development::Development;
use crate::error::{Error, Result};
use crate::mack::MackFit;
use crate::triangle::{Segment, Triangle};

/// The calendar-year test's default confidence level, Mack's and R's
/// `cyEffTest`'s 95%.
pub const CALENDAR_YEAR_CI: f64 = 0.95;

/// The correlation test's default confidence level, Mack's and R's
/// `dfCorTest`'s 50%: the test is approximate and meant to catch
/// correlation in a substantial part of the triangle.
pub const DEVELOPMENT_CORRELATION_CI: f64 = 0.5;

/// The link ratios a test reads: each origin's ratio from age `d` to age
/// `d + 1`, observed where both cells are observed, the earlier one is not
/// zero and (for a fit) the link ratio was kept.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LinkRatios {
    n_origins: usize,
    n_links: usize,
    /// Row-major over origin × link.
    ratios: Vec<Option<f64>>,
    /// The valuation month of each link's later cell.
    valuations: Vec<Month>,
    /// The age at the start of each link.
    ages: Vec<Lag>,
}

impl LinkRatios {
    /// Every observed link ratio of `segment`, or only those `development`
    /// keeps when given.
    pub(crate) fn of(segment: &Segment, development: Option<&Development>) -> Result<Self> {
        let n_links = segment.n_dev.saturating_sub(1);
        let kept = match development {
            Some(d) if d.has_exclusions() => Some(d.kept_mask(segment)?),
            _ => None,
        };
        let mut ratios = vec![None; segment.n_origins * n_links];
        let mut valuations = Vec::with_capacity(ratios.len());
        for o in 0..segment.n_origins {
            for d in 0..n_links {
                valuations.push(segment.valuation_of(o, d + 1));
                if kept.as_ref().is_some_and(|k| !k[o * n_links + d]) {
                    continue;
                }
                if let (Some(from), Some(to)) = (segment.get(o, d), segment.get(o, d + 1)) {
                    let r = to / from;
                    if from != 0.0 && r.is_finite() {
                        ratios[o * n_links + d] = Some(r);
                    }
                }
            }
        }
        Ok(Self {
            n_origins: segment.n_origins,
            n_links,
            ratios,
            valuations,
            ages: segment.ages[..n_links].to_vec(),
        })
    }

    fn get(&self, origin: usize, link: usize) -> Option<f64> {
        self.ratios[origin * self.n_links + link]
    }
}

/// One calendar diagonal of the calendar-year test: Mack's `A_j`, the link
/// ratios whose later cell lies on it.
#[derive(Debug, Clone, PartialEq)]
pub struct CalendarDiagonal {
    /// The valuation month of the diagonal's later cells.
    pub valuation: Month,
    /// `S_j`: link ratios below their age's median.
    pub small: usize,
    /// `L_j`: link ratios above their age's median.
    pub large: usize,
    /// `Z_j = min(S_j, L_j)`.
    pub z: usize,
    /// `n = S_j + L_j`; a link ratio equal to its age's median counts in
    /// neither.
    pub n: usize,
    /// `m = floor((n - 1) / 2)`.
    pub m: usize,
    /// `E[Z_j]` under no calendar-year effect, Mack's (H1).
    pub expected: f64,
    /// `Var[Z_j]` under no calendar-year effect, Mack's (H2).
    pub variance: f64,
}

/// Mack's test for calendar-year effects (Mack 1994, appendix H; R
/// ChainLadder's `cyEffTest`).
///
/// At each age the link ratios are split at their median into small and
/// large ones (one equal to the median is in neither); each calendar
/// diagonal counts its small ones `S_j` and large ones `L_j`, and
/// `Z = sum_j min(S_j, L_j)` is compared with the normal interval
/// `E[Z] ± z sqrt(Var[Z])`, `z` the standard normal's `(1 + ci) / 2`
/// quantile. A diagonal where small (or large) factors prevail lowers `Z`.
#[derive(Debug, Clone, PartialEq)]
pub struct CalendarYearTest {
    /// Every calendar diagonal with at least two link ratios, oldest first
    /// (on a square triangle R's rows `j = 2, ..., I - 1`). A diagonal with
    /// `n <= 1` has `Z_j = E[Z_j] = Var[Z_j] = 0`, which Mack leaves out;
    /// it changes no total.
    pub diagonals: Vec<CalendarDiagonal>,
    /// `Z = sum_j Z_j`.
    pub z: f64,
    /// `E[Z] = sum_j E[Z_j]`.
    pub expected: f64,
    /// `Var[Z] = sum_j Var[Z_j]`.
    pub variance: f64,
    /// The confidence level of the interval.
    pub ci: f64,
    /// The interval's lower end.
    pub lower: f64,
    /// The interval's upper end.
    pub upper: f64,
}

impl CalendarYearTest {
    /// Whether `Z` lies within the interval (ends included): no
    /// calendar-year effect is detected. R prints the opposite as
    /// "Calendar Year Effect: TRUE".
    pub fn passes(&self) -> bool {
        self.lower <= self.z && self.z <= self.upper
    }
}

/// One pair of adjacent link-ratio columns in the correlation test.
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopmentPair {
    /// The age at the start of the first link: the pair is the link ratios
    /// from `age` and from the next age (Mack's `T_k` with `k` the second
    /// link's position counted from 1, R's `i` the first's).
    pub age: Lag,
    /// The number of origins with both link ratios.
    pub n: usize,
    /// `T_k`: Spearman's rank correlation of the two columns over those
    /// origins, with average ranks for ties. NaN where a column's ranks
    /// are all tied.
    pub t: f64,
    /// The weight of `T_k` in `T`, `n - 1` (Mack's `I - k - 1`), 1 over
    /// its variance; 0 where `T_k` is NaN.
    pub weight: f64,
}

/// Mack's test for correlation between adjacent development factors
/// (Mack 1994, appendix G; R ChainLadder's `dfCorTest`).
///
/// `T` is the weighted mean of the pairs' Spearman correlations, with
/// variance `1 / sum(weights)`, which is Mack's `1 / ((I - 2)(I - 3) / 2)`
/// on a full triangle; it is compared with `± z sqrt(Var[T])`, `z` the
/// standard normal's `(1 + ci) / 2` quantile.
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopmentCorrelationTest {
    /// Every pair of adjacent link columns with at least two origins in
    /// common, youngest first.
    pub pairs: Vec<DevelopmentPair>,
    /// `T`, the weighted mean of the pairs' `T_k`.
    pub t: f64,
    /// `Var[T] = 1 / sum(weights)`; `E[T] = 0`.
    pub variance: f64,
    /// The confidence level of the interval.
    pub ci: f64,
    /// The interval's lower end, `-upper`.
    pub lower: f64,
    /// The interval's upper end.
    pub upper: f64,
}

impl DevelopmentCorrelationTest {
    /// Whether `T` lies within the interval (ends included): no correlation
    /// between adjacent development factors is detected.
    pub fn passes(&self) -> bool {
        self.lower <= self.t && self.t <= self.upper
    }
}

/// Mack's calendar-year effect test on every observed link ratio of
/// `column` in a single-segment triangle, at confidence level `ci`
/// ([`CALENDAR_YEAR_CI`] is Mack's and R's). See [`CalendarYearTest`].
///
/// Errors: `ci` outside `(0, 1)`, a triangle with several segments or
/// without the column, and one with no calendar diagonal of two link
/// ratios (fewer than three ages).
pub fn calendar_year_effect(
    triangle: &Triangle,
    column: &str,
    ci: f64,
) -> Result<CalendarYearTest> {
    let segment = triangle.segment(column)?;
    LinkRatios::of(&segment, None)?.calendar_year_effect(ci)
}

/// Mack's test for correlation between adjacent development factors on
/// every observed link ratio of `column` in a single-segment triangle, at
/// confidence level `ci` ([`DEVELOPMENT_CORRELATION_CI`] is Mack's and
/// R's). See [`DevelopmentCorrelationTest`].
///
/// Errors: `ci` outside `(0, 1)`, a triangle with several segments or
/// without the column, and one with no pair of adjacent link columns with
/// two origins in common and untied ranks (fewer than four ages on a
/// square triangle).
pub fn development_correlation(
    triangle: &Triangle,
    column: &str,
    ci: f64,
) -> Result<DevelopmentCorrelationTest> {
    let segment = triangle.segment(column)?;
    LinkRatios::of(&segment, None)?.development_correlation(ci)
}

impl MackFit {
    /// Mack's calendar-year effect test on the link ratios this fit
    /// averaged: those of its column, without the ones `n_periods`,
    /// `drop_high` and `drop_low` exclude (decision 26). A selected pattern
    /// changes no link ratio and so not the test. See
    /// [`calendar_year_effect`].
    pub fn calendar_year_effect(&self, ci: f64) -> Result<CalendarYearTest> {
        self.links.calendar_year_effect(ci)
    }

    /// Mack's test for correlation between adjacent development factors on
    /// the link ratios this fit averaged, as
    /// [`calendar_year_effect`](Self::calendar_year_effect) reads them. See
    /// [`development_correlation`].
    pub fn development_correlation(&self, ci: f64) -> Result<DevelopmentCorrelationTest> {
        self.links.development_correlation(ci)
    }
}

/// The standard normal quantile of `(1 + ci) / 2`, for `ci` in `(0, 1)`.
fn interval_z(ci: f64) -> Result<f64> {
    if !(ci > 0.0 && ci < 1.0) {
        return Err(Error::InvalidSetting {
            name: "ci",
            value: ci,
            expected: "a confidence level strictly between 0 and 1",
        });
    }
    Ok(norm_quantile(0.5 + ci / 2.0))
}

/// The median of `values` (sorted in place): the middle value, or the mean
/// of the two middle ones, as R's `median`.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// `choose(n - 1, m) / 2^n`, in Mack's (H1) and (H2).
fn choose_over_power(n: usize, m: usize) -> f64 {
    if n <= 1000 {
        // Exact while the binomial coefficient stays below 2^53 (n up to
        // about 55), and within a few ulps beyond.
        let k = m.min(n - 1 - m);
        let mut c = 1.0_f64;
        for i in 1..=k {
            c = c * (n - 1 - k + i) as f64 / i as f64;
        }
        c / 2.0_f64.powi(n as i32)
    } else {
        let (n1, m) = ((n - 1) as f64, m as f64);
        (ln_gamma(n1 + 1.0)
            - ln_gamma(m + 1.0)
            - ln_gamma(n1 - m + 1.0)
            - n as f64 * std::f64::consts::LN_2)
            .exp()
    }
}

/// Mack's (H1) and (H2): the mean and variance of `Z_j` given `n`.
fn z_moments(n: usize) -> (usize, f64, f64) {
    if n == 0 {
        return (0, 0.0, 0.0);
    }
    let m = (n - 1) / 2;
    let c = choose_over_power(n, m);
    let nf = n as f64;
    let e = nf / 2.0 - c * nf;
    let var = nf * (nf - 1.0) / 4.0 - c * nf * (nf - 1.0) + e - e * e;
    (m, e, var)
}

/// Average ranks (1-based) of `values`, ties sharing their mean rank.
fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut out = vec![0.0; values.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && values[order[j + 1]] == values[order[i]] {
            j += 1;
        }
        let rank = (i + j) as f64 / 2.0 + 1.0;
        for &o in &order[i..=j] {
            out[o] = rank;
        }
        i = j + 1;
    }
    out
}

/// Spearman's rank correlation: Pearson's on average ranks, as R's
/// `cor(method = "spearman")`. NaN if either column's ranks are all tied.
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let (rx, ry) = (ranks(x), ranks(y));
    let n = rx.len() as f64;
    let (mx, my) = (rx.iter().sum::<f64>() / n, ry.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in rx.iter().zip(&ry) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    if sxx == 0.0 || syy == 0.0 {
        f64::NAN
    } else {
        sxy / (sxx * syy).sqrt()
    }
}

impl LinkRatios {
    pub(crate) fn calendar_year_effect(&self, ci: f64) -> Result<CalendarYearTest> {
        let zq = interval_z(ci)?;
        // (link ratios, small, large) of each calendar diagonal.
        let mut diagonals: BTreeMap<Month, (usize, usize, usize)> = BTreeMap::new();
        for d in 0..self.n_links {
            let observed: Vec<(usize, f64)> = (0..self.n_origins)
                .filter_map(|o| Some((o, self.get(o, d)?)))
                .collect();
            if observed.is_empty() {
                continue;
            }
            let mut values: Vec<f64> = observed.iter().map(|&(_, r)| r).collect();
            let med = median(&mut values);
            for (o, r) in observed {
                let entry = diagonals
                    .entry(self.valuations[o * self.n_links + d])
                    .or_default();
                entry.0 += 1;
                if r < med {
                    entry.1 += 1;
                } else if r > med {
                    entry.2 += 1;
                }
            }
        }
        let diagonals: Vec<CalendarDiagonal> = diagonals
            .into_iter()
            .filter(|(_, (count, _, _))| *count >= 2)
            .map(|(valuation, (_, small, large))| {
                let n = small + large;
                let (m, expected, variance) = z_moments(n);
                CalendarDiagonal {
                    valuation,
                    small,
                    large,
                    z: small.min(large),
                    n,
                    m,
                    expected,
                    variance,
                }
            })
            .collect();
        if diagonals.is_empty() {
            return Err(Error::TooFewAges {
                needed: 3,
                found: self.n_links + 1,
            });
        }
        let z = diagonals.iter().map(|d| d.z as f64).sum();
        let expected: f64 = diagonals.iter().map(|d| d.expected).sum();
        let variance: f64 = diagonals.iter().map(|d| d.variance).sum();
        let half = zq * variance.sqrt();
        Ok(CalendarYearTest {
            diagonals,
            z,
            expected,
            variance,
            ci,
            lower: expected - half,
            upper: expected + half,
        })
    }

    pub(crate) fn development_correlation(&self, ci: f64) -> Result<DevelopmentCorrelationTest> {
        let zq = interval_z(ci)?;
        let mut pairs = Vec::new();
        for d in 0..self.n_links.saturating_sub(1) {
            let (x, y): (Vec<f64>, Vec<f64>) = (0..self.n_origins)
                .filter_map(|o| Some((self.get(o, d)?, self.get(o, d + 1)?)))
                .unzip();
            if x.len() < 2 {
                continue;
            }
            let t = spearman(&x, &y);
            pairs.push(DevelopmentPair {
                age: self.ages[d],
                n: x.len(),
                t,
                weight: if t.is_nan() {
                    0.0
                } else {
                    (x.len() - 1) as f64
                },
            });
        }
        let total: f64 = pairs.iter().map(|p| p.weight).sum();
        if total == 0.0 {
            return Err(Error::TooFewAges {
                needed: 4,
                found: self.n_links + 1,
            });
        }
        let t = pairs
            .iter()
            .filter(|p| p.weight > 0.0)
            .map(|p| p.weight * p.t)
            .sum::<f64>()
            / total;
        let variance = 1.0 / total;
        let half = zq * variance.sqrt();
        Ok(DevelopmentCorrelationTest {
            pairs,
            t,
            variance,
            ci,
            lower: -half,
            upper: half,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mack::Mack;
    use crate::triangle::tests::{annual, raa};

    fn close(got: f64, want: f64, tol: f64) {
        assert!((got - want).abs() <= tol, "got {got}, want {want}");
    }

    #[test]
    fn reproduces_mack_appendix_g_on_raa() {
        // Mack (1994), appendix G, p. 161: T_k = 4/21, -9/28, 3/7, -1/5,
        // 2/5, -1/2, 1 with weights 7 to 1, T = .070, Var(T) = 1/28.
        let dc = development_correlation(&raa(), "values", 0.5).unwrap();
        let printed = [4.0 / 21.0, -9.0 / 28.0, 3.0 / 7.0, -0.2, 0.4, -0.5, 1.0];
        assert_eq!(dc.pairs.len(), printed.len());
        for (k, (p, want)) in dc.pairs.iter().zip(printed).enumerate() {
            close(p.t, want, 1e-14);
            assert_eq!(p.weight, (7 - k) as f64);
        }
        close(dc.t, 0.070, 5e-4);
        close(dc.variance, 1.0 / 28.0, 1e-15);
        // His 50% limits, +-.67 / sqrt(28) = +-.127.
        close(dc.upper, 0.127, 5e-4);
        assert!(dc.passes());
    }

    #[test]
    fn reproduces_mack_appendix_h_on_raa() {
        // Mack (1994), appendix H, p. 167: S_j, L_j for j = 2..9, Z = 14,
        // E(Z) = 12.875, Var(Z) = 3.9785, interval 12.875 +- 2 sd =
        // (8.886, 16.864).
        let s = [1, 3, 3, 1, 1, 2, 4, 4];
        let l = [1, 0, 1, 3, 3, 4, 4, 4];
        let ci = 2.0 * prospicio_math::special::norm_cdf(2.0) - 1.0;
        let cy = calendar_year_effect(&raa(), "values", ci).unwrap();
        let got: Vec<(usize, usize)> = cy.diagonals.iter().map(|d| (d.small, d.large)).collect();
        assert_eq!(got, s.into_iter().zip(l).collect::<Vec<_>>());
        assert_eq!(cy.diagonals[0].valuation, Month::new(1983, 12).unwrap());
        assert_eq!(cy.z, 14.0);
        close(cy.expected, 12.875, 1e-12);
        close(cy.variance, 3.9785, 5e-5);
        close(cy.lower, 8.886, 5e-4);
        close(cy.upper, 16.864, 5e-4);
        assert!(cy.passes());
    }

    #[test]
    fn moments_match_mack_n_equal_five() {
        // Mack's worked example: n = 5 gives E = 50/32, Var = 95/256.
        let (m, e, v) = z_moments(5);
        assert_eq!(m, 2);
        close(e, 50.0 / 32.0, 1e-15);
        close(v, 95.0 / 256.0, 1e-15);
        assert_eq!(z_moments(1), (0, 0.0, 0.0));
        // The log-gamma branch agrees with the product where both apply.
        let n = 1001;
        let direct = {
            let mut c = 1.0_f64;
            let m = (n - 1) / 2;
            for i in 1..=m {
                c *= (n - 1 - m + i) as f64 / i as f64 / 4.0;
            }
            c / 2.0_f64.powi((n - 2 * m) as i32)
        };
        close(choose_over_power(n, (n - 1) / 2) / direct, 1.0, 1e-9);
    }

    #[test]
    fn ranks_average_ties_as_r() {
        assert_eq!(ranks(&[3.0, 1.0, 3.0, 2.0]), vec![3.5, 1.0, 3.5, 2.0]);
        assert!(spearman(&[1.0, 1.0], &[1.0, 2.0]).is_nan());
        close(spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), -1.0, 1e-15);
    }

    #[test]
    fn rejects_bad_levels_and_small_triangles() {
        for ci in [0.0, 1.0, f64::NAN, -0.5] {
            assert!(matches!(
                calendar_year_effect(&raa(), "values", ci),
                Err(Error::InvalidSetting { name: "ci", .. })
            ));
        }
        let two = annual(2020, &[&[1.0, 2.0], &[1.0]]);
        assert!(matches!(
            calendar_year_effect(&two, "values", 0.95),
            Err(Error::TooFewAges { .. })
        ));
        let three = annual(2020, &[&[1.0, 2.0, 3.0], &[1.0, 3.0], &[1.0]]);
        assert!(matches!(
            development_correlation(&three, "values", 0.5),
            Err(Error::TooFewAges { .. })
        ));
    }

    #[test]
    fn a_fit_without_exclusions_tests_the_whole_triangle() {
        let fit = Mack::default().fit(&raa(), "values").unwrap();
        assert_eq!(
            fit.calendar_year_effect(0.95).unwrap(),
            calendar_year_effect(&raa(), "values", 0.95).unwrap()
        );
        assert_eq!(
            fit.development_correlation(0.5).unwrap(),
            development_correlation(&raa(), "values", 0.5).unwrap()
        );
    }

    #[test]
    fn a_fit_with_exclusions_tests_only_its_kept_link_ratios() {
        // n_periods = 4 keeps the link ratios ending on the last four
        // diagonals; the test sees those diagonals only.
        let mack = Mack {
            development: Development {
                n_periods: Some(4),
                ..Default::default()
            },
            ..Default::default()
        };
        let fit = mack.fit(&raa(), "values").unwrap();
        let cy = fit.calendar_year_effect(0.95).unwrap();
        let years: Vec<i32> = cy.diagonals.iter().map(|d| d.valuation.year()).collect();
        assert_eq!(years, vec![1987, 1988, 1989, 1990]);
        // drop_high and drop_low remove one large and one small factor at
        // each age with enough link ratios.
        let dropped = Mack {
            development: Development {
                drop_high: 1,
                drop_low: 1,
                ..Default::default()
            },
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        let all = calendar_year_effect(&raa(), "values", 0.95).unwrap();
        let kept = dropped.calendar_year_effect(0.95).unwrap();
        let count = |t: &CalendarYearTest| t.diagonals.iter().map(|d| d.n).sum::<usize>();
        assert!(count(&kept) < count(&all));
        let dc = dropped.development_correlation(0.5).unwrap();
        assert!(dc.pairs.iter().all(|p| p.n >= 2));
    }
}
