//! Development factors: age-to-age factors, their variance parameters and
//! standard errors, estimated as the weighted regressions of Mack (1993,
//! 1999) used by R ChainLadder and chainladder-python.

use std::fmt;

use crate::error::{Error, Result};
use crate::pattern::{SelectedPattern, blend};
use crate::triangle::{Segment, Triangle};
use prospicio_core::{Lag, Month};

/// How individual link ratios are averaged into one factor per age.
///
/// Each is the weighted regression `C[k+1] = f C[k]` with weights
/// `C[k]^(alpha - 2)`, so `f = sum(C[k]^(alpha-1) C[k+1]) / sum(C[k]^alpha)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Average {
    /// `alpha = 1`: `sum(C[k+1]) / sum(C[k])`, the chain-ladder factor.
    #[default]
    Volume,
    /// `alpha = 0`: the mean of the individual link ratios.
    Simple,
    /// `alpha = 2`: ordinary least squares through the origin.
    Regression,
    /// The geometric mean of the individual link ratios, `exp(mean(ln r))`.
    /// It is not a weighted regression and Mack's model has no variance for
    /// it, so its sigma and standard errors are NaN and the methods built on
    /// Mack's model reject it.
    Geometric,
}

impl Average {
    /// Mack's `alpha`, or `None` for the geometric average, which is not a
    /// weighted regression.
    pub const fn alpha(self) -> Option<f64> {
        match self {
            Self::Volume => Some(1.0),
            Self::Simple => Some(0.0),
            Self::Regression => Some(2.0),
            Self::Geometric => None,
        }
    }
}

/// How a variance parameter `sigma_k` that cannot be estimated (an age with
/// a single link ratio) is filled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SigmaInterpolation {
    /// Regress `ln(sigma_k)` on `k` over the estimated, positive sigmas and
    /// extrapolate. The default in R ChainLadder and chainladder-python.
    #[default]
    LogLinear,
    /// Mack (1993): `sigma_k^2 = min(sigma_{k-1}^4 / sigma_{k-2}^2,
    /// sigma_{k-2}^2, sigma_{k-1}^2)`.
    Mack,
}

/// Development-factor estimator.
///
/// ```
/// use prospicio_reserving::Development;
/// # use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// # let origin = [2020, 2020, 2021].map(Month::january);
/// # let tri = Triangle::from_long(&Long {
/// #     keys: &[],
/// #     origin: &origin,
/// #     development: DevelopmentColumn::Age(&[12, 24, 12]),
/// #     values: &[("paid", &[100.0, 150.0, 110.0])],
/// #     origin_grain: Grain::Year,
/// #     development_grain: Grain::Year,
/// #     cumulative: true,
/// # })
/// # .unwrap();
/// let fit = Development::default().fit(&tri, "paid").unwrap();
/// assert_eq!(fit.ldf, vec![1.5]);
/// ```
///
/// With a [`pattern`](Self::pattern) the fit also carries the selected
/// factors and their weights ([`DevelopmentFit::selected`],
/// [`DevelopmentFit::weight`]); its estimated fields stay the triangle's.
#[derive(Clone, PartialEq, Default)]
pub struct Development {
    pub average: Average,
    pub sigma_interpolation: SigmaInterpolation,
    /// Factors chosen by the user, which replace or blend with the
    /// estimated ones at the ages they give; `None` estimates every factor.
    /// See [`SelectedPattern`]: a selected factor carries no parameter
    /// error, so a pattern lowers the reported uncertainty.
    pub pattern: Option<SelectedPattern>,
    /// Average only the link ratios that start in the latest `n` origin
    /// periods of calendar time, that is, whose starting cell is valued no
    /// more than `n` origin periods before the segment's latest valuation;
    /// `None` uses every one. An age with fewer link ratios uses all of
    /// them. `Some(0)` is an error.
    pub n_periods: Option<usize>,
    /// Leave out the `drop_high` highest of the link ratios kept by
    /// `n_periods`, at each age. Link ratios are ranked by value, ties by
    /// the starting cell's value. At an age where the drops would leave no
    /// link ratio, none is dropped.
    pub drop_high: usize,
    /// Leave out the `drop_low` lowest of the link ratios kept by
    /// `n_periods`, at each age; see [`drop_high`](Self::drop_high).
    pub drop_low: usize,
}

/// As derived, but without `pattern` when there is none and without the
/// link exclusions at their defaults, so that the settings recorded in a
/// bootstrap's provenance are as they were before them.
impl fmt::Debug for Development {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("Development");
        d.field("average", &self.average)
            .field("sigma_interpolation", &self.sigma_interpolation);
        if let Some(pattern) = &self.pattern {
            d.field("pattern", pattern);
        }
        if let Some(n) = &self.n_periods {
            d.field("n_periods", n);
        }
        if self.drop_high != 0 {
            d.field("drop_high", &self.drop_high);
        }
        if self.drop_low != 0 {
            d.field("drop_low", &self.drop_low);
        }
        d.finish()
    }
}

/// Estimated development pattern of one triangle segment.
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopmentFit {
    /// Age of each development position; factor `k` links age `k` to `k + 1`.
    pub development: Vec<Lag>,
    /// Age-to-age factors `f_k`, one fewer than the ages.
    pub ldf: Vec<f64>,
    /// Variance parameters `sigma_k`, with unestimable ones interpolated.
    pub sigma: Vec<f64>,
    /// Standard error of each factor, `sigma_k / sqrt(volume_k)`.
    pub std_err: Vec<f64>,
    /// Total regression weight behind each factor, `sum(C[k]^alpha)` over
    /// the origins observed at both ages: with volume weighting, the sum of
    /// the earlier values.
    pub volume: Vec<f64>,
    /// Mack's `alpha` the factors were estimated with; NaN for the
    /// geometric average.
    pub alpha: f64,
    /// The number of link ratios behind each factor, after `n_periods` and
    /// the drops.
    pub n_links: Vec<usize>,
    /// Whether `n_periods` or the drops left out a link ratio at some age.
    pub excluded: bool,
    /// The selected pattern's age-to-age factor of each link, after
    /// conversion; `None` where the pattern selects none (or there is no
    /// pattern).
    pub selected: Vec<Option<f64>>,
    /// The weight `w` on each selected factor; 0 where none is selected.
    pub weight: Vec<f64>,
}

impl DevelopmentFit {
    /// The factors before the tail: `w * selected + (1 - w) * ldf` where a
    /// factor is selected, the estimate `ldf` elsewhere.
    pub fn blended_ldf(&self) -> Vec<f64> {
        blend(&self.selected, &self.weight, &self.ldf)
    }

    /// Standard error of each blended factor, `(1 - w) * std_err`: a
    /// selected factor carries no estimation error (0, even where the
    /// estimate's is NaN), a blended one the estimate's share.
    pub fn blended_std_err(&self) -> Vec<f64> {
        self.std_err
            .iter()
            .zip(&self.weight)
            .map(|(s, w)| if *w == 1.0 { 0.0 } else { (1.0 - w) * s })
            .collect()
    }

    /// Whether a pattern selects any factor.
    pub fn has_pattern(&self) -> bool {
        self.selected.iter().any(Option::is_some)
    }
}

/// An origin's index and the `(from, to)` values of one of its link ratios.
type KeptLink = (usize, f64, f64);

impl Development {
    /// Estimates the development pattern of `column` in a single-segment
    /// triangle, from every origin observed at both ends of each link.
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<DevelopmentFit> {
        let segment = triangle.segment(column)?;
        let mut fit = self.fit_segment(&segment)?;
        fit.development = segment.ages.clone();
        Ok(fit)
    }

    /// Whether any link ratio exclusion (`n_periods`, `drop_high`,
    /// `drop_low`) is set.
    pub(crate) fn has_exclusions(&self) -> bool {
        self.n_periods.is_some() || self.drop_high != 0 || self.drop_low != 0
    }

    /// Errors if the geometric average is set, for a method built on
    /// Mack's model, which has no variance for it.
    pub(crate) fn reject_geometric(&self, method: &'static str) -> Result<()> {
        if self.average == Average::Geometric {
            return Err(Error::NotSupported {
                setting: "the geometric average",
                method,
                reason: "it is not a regression, so it has no variance",
            });
        }
        Ok(())
    }

    /// Errors if a link ratio exclusion is set, for a method that assumes
    /// every link ratio.
    pub(crate) fn reject_exclusions(&self, method: &'static str) -> Result<()> {
        if self.has_exclusions() {
            return Err(Error::NotSupported {
                setting: "n_periods, drop_high and drop_low",
                method,
                reason: "not yet supported, the method assumes every link ratio",
            });
        }
        Ok(())
    }

    /// The oldest starting valuation `n_periods` keeps, `None` without it.
    /// An error if `n_periods` is zero.
    fn cutoff(&self, segment: &Segment) -> Result<Option<Month>> {
        if self.n_periods == Some(0) {
            return Err(Error::InvalidSetting {
                name: "n_periods",
                value: 0.0,
                expected: "at least 1, or none for every origin",
            });
        }
        Ok(
            match (
                self.n_periods,
                segment.latest_valuation(),
                segment.origins.first(),
            ) {
                (Some(n), Some(latest), Some(origin)) => {
                    // A million periods reach before any triangle's first
                    // cell, and keep the product far from overflow.
                    let n = n.min(1_000_000) as i64;
                    Some(latest.add_months(-(n * i64::from(origin.grain().months()))))
                }
                _ => None,
            },
        )
    }

    /// The link ratios the fit averages, indexed by `origin * (n_dev - 1) +
    /// link`: true where the link ratio is observed and kept by
    /// `n_periods` and the drops, the same ones [`fit_segment`] uses.
    pub(crate) fn kept_mask(&self, segment: &Segment) -> Result<Vec<bool>> {
        let cutoff = self.cutoff(segment)?;
        let n_links = segment.n_dev.saturating_sub(1);
        let mut mask = vec![false; segment.n_origins * n_links];
        for age in 0..n_links {
            for (o, _, _) in self.kept_links(segment, age, cutoff)?.0 {
                mask[o * n_links + age] = true;
            }
        }
        Ok(mask)
    }

    pub(crate) fn fit_segment(&self, segment: &Segment) -> Result<DevelopmentFit> {
        let cutoff = self.cutoff(segment)?;
        let n_links = segment.n_dev.saturating_sub(1);
        // The pattern is checked before the estimate, so that a bad pattern
        // is reported as such.
        let (selected, weight) = match &self.pattern {
            Some(p) => {
                let r = p.resolve(&segment.ages)?;
                (r.selected, r.weight)
            }
            None => (vec![None; n_links], vec![0.0; n_links]),
        };
        let mut ldf = Vec::with_capacity(n_links);
        let mut sigma = Vec::with_capacity(n_links);
        let mut volume = Vec::with_capacity(n_links);
        let mut counts = Vec::with_capacity(n_links);
        let mut excluded = false;
        for age in 0..n_links {
            let (pairs, observed) = self.kept_links(segment, age, cutoff)?;
            excluded |= pairs.len() < observed;
            counts.push(pairs.len());
            let Some(alpha) = self.average.alpha() else {
                let pairs: Vec<(f64, f64)> = pairs.iter().map(|&(_, x, y)| (x, y)).collect();
                ldf.push(geometric_mean(&pairs, age)?);
                sigma.push(None);
                volume.push(f64::NAN);
                continue;
            };
            let weight = |x: f64, power: f64| {
                let w = x.powf(power);
                if w.is_finite() {
                    Ok(w)
                } else {
                    Err(Error::Factor {
                        age,
                        reason: "a zero value gets an infinite weight",
                    })
                }
            };
            let mut num = 0.0;
            let mut den = 0.0;
            for &(_, x, y) in &pairs {
                num += weight(x, alpha - 1.0)? * y;
                den += weight(x, alpha)?;
            }
            if den == 0.0 {
                return Err(Error::Factor {
                    age,
                    reason: "earlier values sum to zero",
                });
            }
            let f = num / den;
            // An origin at zero says nothing about the variance given a
            // positive value, and its weight C^(alpha - 2) is infinite, so it
            // does not enter sigma.
            let informative: Vec<(f64, f64)> = pairs
                .iter()
                .filter(|&&(_, x, _)| x != 0.0)
                .map(|&(_, x, y)| (x, y))
                .collect();
            sigma.push(if informative.len() > 1 {
                let ss: f64 = informative
                    .iter()
                    .map(|&(x, y)| x.powf(alpha - 2.0) * (y - f * x).powi(2))
                    .sum();
                Some((ss / (informative.len() - 1) as f64).sqrt())
            } else {
                None
            });
            ldf.push(f);
            volume.push(den);
        }
        // The geometric average has no variance: every sigma is NaN, and
        // there is nothing to interpolate.
        let sigma = if self.average.alpha().is_some() {
            interpolate_sigma(&sigma, self.sigma_interpolation)
        } else {
            vec![f64::NAN; n_links]
        };
        let std_err = sigma
            .iter()
            .zip(&volume)
            .map(|(s, v)| s / v.sqrt())
            .collect();
        Ok(DevelopmentFit {
            development: segment.ages.clone(),
            ldf,
            sigma,
            std_err,
            volume,
            alpha: self.average.alpha().unwrap_or(f64::NAN),
            n_links: counts,
            excluded,
            selected,
            weight,
        })
    }

    /// The `(origin, from, to)` of the link ratios of one age that the
    /// averages use: those starting at or after `cutoff`, less the
    /// `drop_high` highest and `drop_low` lowest by value (ties by the
    /// starting value) unless that would leave none. In origin order.
    fn kept_links(
        &self,
        segment: &Segment,
        age: usize,
        cutoff: Option<Month>,
    ) -> Result<(Vec<KeptLink>, usize)> {
        let mut links = segment.link_pairs_indexed(age);
        if links.is_empty() {
            return Err(Error::Factor {
                age,
                reason: "no origin is observed at both ages",
            });
        }
        let observed = links.len();
        if let Some(cutoff) = cutoff {
            links.retain(|&(o, _, _)| segment.valuation_of(o, age) >= cutoff);
            if links.is_empty() {
                return Err(Error::Factor {
                    age,
                    reason: "no link ratio is left after n_periods",
                });
            }
        }
        let drops = self.drop_high.saturating_add(self.drop_low);
        if drops > 0 {
            // Only link ratios with a finite value take part in the ranking
            // (a 0/0 or x/0 has no rank): they are never dropped and do not
            // count toward the number of links the drops must leave.
            let mut order: Vec<usize> = (0..links.len())
                .filter(|&i| (links[i].2 / links[i].1).is_finite())
                .collect();
            if order.len() > drops {
                order.sort_by(|&a, &b| {
                    let (_, xa, ya) = links[a];
                    let (_, xb, yb) = links[b];
                    (ya / xa).total_cmp(&(yb / xb)).then(xa.total_cmp(&xb))
                });
                let mut keep = vec![true; links.len()];
                for &i in &order[..self.drop_low] {
                    keep[i] = false;
                }
                for &i in &order[order.len() - self.drop_high..] {
                    keep[i] = false;
                }
                let mut flags = keep.into_iter();
                links.retain(|_| flags.next().unwrap_or(false));
            }
        }
        Ok((links, observed))
    }
}

/// `exp(mean(ln(y / x)))` over the link ratios of one age.
fn geometric_mean(pairs: &[(f64, f64)], age: usize) -> Result<f64> {
    let mut sum = 0.0;
    for &(x, y) in pairs {
        let ratio = y / x;
        if !(ratio > 0.0 && ratio.is_finite()) {
            return Err(Error::Factor {
                age,
                reason: "the geometric average needs positive, finite link ratios",
            });
        }
        sum += ratio.ln();
    }
    Ok((sum / pairs.len() as f64).exp())
}

/// Fills the sigmas that could not be estimated. Those that cannot be
/// filled either (too few estimated sigmas to extrapolate from) are NaN;
/// the chain ladder does not need them and Mack rejects them.
fn interpolate_sigma(sigma: &[Option<f64>], method: SigmaInterpolation) -> Vec<f64> {
    match method {
        SigmaInterpolation::LogLinear => {
            let points: Vec<(f64, f64)> = sigma
                .iter()
                .enumerate()
                .filter_map(|(k, s)| s.filter(|&s| s > 0.0).map(|s| (k as f64, s.ln())))
                .collect();
            let (intercept, slope) = if points.len() < 2 {
                (f64::NAN, f64::NAN)
            } else {
                let n = points.len() as f64;
                let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
                let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
                let sxy: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
                let sxx: f64 = points.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
                let slope = sxy / sxx;
                (mean_y - slope * mean_x, slope)
            };
            sigma
                .iter()
                .enumerate()
                .map(|(k, s)| s.unwrap_or_else(|| (intercept + slope * k as f64).exp()))
                .collect()
        }
        SigmaInterpolation::Mack => {
            let mut out: Vec<f64> = Vec::with_capacity(sigma.len());
            for &s in sigma {
                let filled = match (s, &out[..]) {
                    (Some(s), _) => s,
                    (None, [.., a, b]) => {
                        let (a2, b2) = (a * a, b * b);
                        let floor = a2.min(b2);
                        let s2 = if a2 == 0.0 {
                            floor
                        } else {
                            (b2 * b2 / a2).min(floor)
                        };
                        s2.sqrt()
                    }
                    (None, _) => f64::NAN,
                };
                out.push(filled);
            }
            out
        }
    }
}

/// Age-to-ultimate factors: element `k` is `tail` times the product of
/// `ldf[k..]`, so the last element (the oldest age) is `tail` itself.
pub fn cumulative_factors(ldf: &[f64], tail: f64) -> Vec<f64> {
    let mut out = vec![tail; ldf.len() + 1];
    for k in (0..ldf.len()).rev() {
        out[k] = out[k + 1] * ldf[k];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::{annual, raa};

    fn close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < tol, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn raa_volume_factors() {
        // R ChainLadder: ata(RAA) vwtd.
        let fit = Development::default().fit(&raa(), "values").unwrap();
        close(
            &fit.ldf,
            &[
                2.999359, 1.623523, 1.270888, 1.171675, 1.113385, 1.041935, 1.033264, 1.016936,
                1.009217,
            ],
            1e-6,
        );
        assert_eq!(fit.development[0], 12);
    }

    #[test]
    fn raa_simple_factors() {
        // R ChainLadder: ata(RAA) smpl.
        let dev = Development {
            average: Average::Simple,
            ..Default::default()
        };
        close(
            &dev.fit(&raa(), "values").unwrap().ldf,
            &[
                8.206099, 1.695894, 1.314510, 1.182926, 1.126962, 1.043328, 1.034355, 1.017995,
                1.009217,
            ],
            1e-6,
        );
    }

    #[test]
    fn raa_sigma_both_interpolations() {
        // R ChainLadder: MackChainLadder(RAA, est.sigma = ...)$sigma.
        let head = [
            166.9834704,
            33.2945384,
            26.2952997,
            7.8249598,
            10.9288176,
            6.3890424,
            1.1590623,
            2.8077043,
        ];
        let log_linear = Development::default().fit(&raa(), "values").unwrap();
        close(&log_linear.sigma[..8], &head, 1e-6);
        close(&log_linear.sigma[8..], &[0.8033494], 1e-6);
        let mack = Development {
            sigma_interpolation: SigmaInterpolation::Mack,
            ..Default::default()
        }
        .fit(&raa(), "values")
        .unwrap();
        close(&mack.sigma[8..], &[1.159062], 1e-6);
    }

    #[test]
    fn mack_rule_cases() {
        let fill = |s: &[Option<f64>]| interpolate_sigma(s, SigmaInterpolation::Mack);
        close(
            &fill(&[Some(2.0), Some(1.0), None]),
            &[2.0, 1.0, 0.5],
            1e-12,
        );
        close(
            &fill(&[Some(1.0), Some(2.0), None]),
            &[1.0, 2.0, 1.0],
            1e-12,
        );
        close(
            &fill(&[Some(0.0), Some(2.0), None]),
            &[0.0, 2.0, 0.0],
            1e-12,
        );
        assert!(interpolate_sigma(&[Some(1.0), None], SigmaInterpolation::Mack)[1].is_nan());
        assert!(interpolate_sigma(&[Some(1.0), None], SigmaInterpolation::LogLinear)[1].is_nan());
    }

    #[test]
    fn simple_average_rejects_zero_origin_value() {
        let t = annual(2020, &[&[0.0, 5.0], &[4.0]]);
        let dev = Development {
            average: Average::Simple,
            ..Default::default()
        };
        assert!(matches!(
            dev.fit(&t, "values"),
            Err(Error::Factor { age: 0, .. })
        ));
        // Volume weighting keeps the zero origin's later value in the factor
        // (the regression formula) but leaves it out of sigma. R ChainLadder
        // fails on this input, so there is no reference value.
        let t = annual(2020, &[&[0.0, 5.0], &[4.0, 6.0], &[2.0, 4.0], &[1.0]]);
        let f = Development::default().fit(&t, "values").unwrap();
        close(&f.ldf, &[15.0 / 6.0], 1e-12);
        assert!(f.sigma[0].is_finite());
    }

    #[test]
    fn cumulative_factors_chain_to_tail() {
        close(
            &cumulative_factors(&[2.0, 1.5], 1.1),
            &[3.3, 1.65, 1.1],
            1e-12,
        );
        assert_eq!(cumulative_factors(&[], 1.05), vec![1.05]);
    }

    fn with_exclusions(n_periods: Option<usize>, drop_high: usize, drop_low: usize) -> Development {
        Development {
            n_periods,
            drop_high,
            drop_low,
            ..Default::default()
        }
    }

    #[test]
    fn default_settings_keep_the_debug_text_and_the_fit() {
        assert_eq!(
            format!("{:?}", Development::default()),
            "Development { average: Volume, sigma_interpolation: LogLinear }"
        );
        assert_eq!(
            format!("{:?}", with_exclusions(Some(3), 1, 0)),
            "Development { average: Volume, sigma_interpolation: LogLinear, n_periods: 3, drop_high: 1 }"
        );
        // A window past the triangle, and no drops, change nothing.
        let base = Development::default().fit(&raa(), "values").unwrap();
        let wide = with_exclusions(Some(50), 0, 0)
            .fit(&raa(), "values")
            .unwrap();
        assert_eq!(base, wide);
        assert_eq!(base.n_links, vec![9, 8, 7, 6, 5, 4, 3, 2, 1]);
        assert!(!base.excluded);
    }

    #[test]
    fn n_periods_keeps_the_link_ratios_of_the_latest_diagonals() {
        // Link 1: (10, 20), (20, 50), (30, 90) start on diagonals 2020,
        // 2021 and 2022; link 2: (20, 30), (50, 70); link 3: (30, 36).
        let t = annual(
            2020,
            &[
                &[10.0, 20.0, 30.0, 36.0],
                &[20.0, 50.0, 70.0],
                &[30.0, 90.0],
                &[40.0],
            ],
        );
        let fit = |n| with_exclusions(Some(n), 0, 0).fit(&t, "values").unwrap();
        let all = Development::default().fit(&t, "values").unwrap();
        assert_eq!(all.n_links, vec![3, 2, 1]);
        let two = fit(2);
        assert_eq!(two.n_links, vec![2, 2, 1]);
        close(&two.ldf, &[140.0 / 50.0, 100.0 / 70.0, 36.0 / 30.0], 1e-12);
        assert!(two.excluded);
        let one = fit(1);
        assert_eq!(one.n_links, vec![1, 1, 1]);
        close(&one.ldf, &[3.0, 1.4, 1.2], 1e-12);
        // One link ratio leaves no sigma to estimate.
        assert!(one.sigma.iter().all(|s| s.is_nan()));
        // A window past the triangle is every link ratio.
        assert_eq!(fit(3), all);
        assert!(!fit(3).excluded);
        assert_eq!(
            with_exclusions(Some(0), 0, 0).fit(&t, "values"),
            Err(Error::InvalidSetting {
                name: "n_periods",
                value: 0.0,
                expected: "at least 1, or none for every origin",
            })
        );
    }

    /// Annual origins from `first_year`, one row of cumulative values per
    /// origin as `(first age in months, values)` with ages every
    /// `step` months; `None` rows leave an origin out.
    fn long_triangle(
        first_year: i32,
        step: Lag,
        rows: &[Option<&[f64]>],
        quarterly: bool,
    ) -> Triangle {
        use crate::triangle::{DevelopmentColumn, Long};
        use prospicio_core::Grain;
        let (mut origin, mut ages, mut values) = (Vec::new(), Vec::new(), Vec::new());
        for (k, row) in rows.iter().enumerate() {
            for (d, &v) in row.iter().flat_map(|r| r.iter()).enumerate() {
                origin.push(Month::january(first_year + k as i32));
                ages.push(step * (d as Lag + 1));
                values.push(v);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("values", &values)],
            origin_grain: Grain::Year,
            development_grain: if quarterly {
                Grain::Quarter
            } else {
                Grain::Year
            },
            cumulative: true,
        })
        .unwrap()
    }

    /// Cumulative values of an origin, `len` cells.
    fn growing(origin: usize, len: usize) -> Vec<f64> {
        (0..len)
            .map(|d| (10.0 + 3.0 * origin as f64) * ((d + 1) as f64).powf(1.3))
            .collect()
    }

    #[test]
    fn n_periods_is_calendar_time_with_quarterly_development_in_annual_origins() {
        // Origins 2020 to 2024 with ages every quarter, all valued at
        // December 2024: 20, 16, 12, 8 and 4 cells, so 19 age-to-age
        // factors. Months from January 2020: a
        // link of origin o at age index k starts at 12 o + 3 (k + 1) - 1,
        // and the latest valuation is 59.
        let rows: Vec<Vec<f64>> = (0..5).map(|o| growing(o, 4 * (5 - o))).collect();
        let refs: Vec<Option<&[f64]>> = rows.iter().map(|r| Some(r.as_slice())).collect();
        let t = long_triangle(2020, 3, &refs, true);
        let fit = |n| with_exclusions(Some(n), 0, 0).fit(&t, "values").unwrap();
        // The latest year holds one link ratio per age: fewer link ratios
        // than origins at every age, where counting origins would keep
        // up to n. chainladder-python 0.10.1 gives the same counts (the
        // link ratios with weight w_ above 0), as long as n is below its
        // number of origins less one; at or past that it keeps every
        // link, where the calendar rule would still cut the earliest
        // quarters.
        assert_eq!(fit(1).n_links, vec![1; 19]);
        let two = [vec![2; 15], vec![1; 4]].concat();
        assert_eq!(fit(2).n_links, two);
        let three = [vec![3; 11], vec![2; 4], vec![1; 4]].concat();
        assert_eq!(fit(3).n_links, three);
        // chainladder-python's first factors with n_periods = 1.
        close(
            &fit(1).ldf[..4],
            &[
                2.4622888266898326,
                1.6940204031852832,
                1.4535178100924682,
                1.336543249988985,
            ],
            1e-9,
        );
        // Five years reach back to the first quarter: every link ratio.
        assert_eq!(fit(5), Development::default().fit(&t, "values").unwrap());
    }

    #[test]
    fn n_periods_follows_diagonals_on_a_sparse_triangle() {
        // 2021 has no rows and 2020 stops at December 2022, while the
        // latest valuation, December 2024, is held by 2022, 2023 and 2024.
        // Months from January 2020: the links start at 11 (2020, age
        // index 0), 23 (2020, 1), 35 (2022, 0), 47 (2022, 1) and 47 (2023,
        // 0); the latest valuation is 59. chainladder-python 0.10.1
        // cannot fit a triangle that is not a staircase, so these counts
        // are derived by hand from the rule.
        let (a, c, d, e) = (growing(0, 3), growing(2, 3), growing(3, 2), growing(4, 1));
        let t = long_triangle(
            2020,
            12,
            &[Some(&a), None, Some(&c), Some(&d), Some(&e)],
            false,
        );
        let fit = |n| with_exclusions(Some(n), 0, 0).fit(&t, "values").unwrap();
        // Counting origins instead would keep three link ratios at the
        // first age with three periods.
        assert_eq!(fit(1).n_links, vec![1, 1]);
        close(&fit(1).ldf, &[d[1] / d[0], c[2] / c[1]], 1e-12);
        assert_eq!(fit(2).n_links, vec![2, 1]);
        assert_eq!(fit(3).n_links, vec![2, 2]);
        assert_eq!(fit(4).n_links, vec![3, 2]);

        // With 2022 and 2023 one cell shorter, the latest valuation is 47
        // and one period reaches back to a start at 35: nothing starts
        // there at age index 1, so that age has no link ratio left.
        let (c, d) = (growing(2, 2), growing(3, 1));
        let t = long_triangle(2020, 12, &[Some(&a), None, Some(&c), Some(&d)], false);
        let one = with_exclusions(Some(1), 0, 0).fit(&t, "values");
        assert!(matches!(one, Err(Error::Factor { age: 1, .. })));
    }

    #[test]
    fn huge_settings_behave_as_more_than_there_are() {
        let base = Development::default().fit(&raa(), "values").unwrap();
        let fit = |n, high, low| with_exclusions(n, high, low).fit(&raa(), "values").unwrap();
        assert_eq!(fit(Some(usize::MAX), 0, 0), base);
        // The sum of the drops would overflow; it is more than any age has,
        // so every link ratio is kept.
        assert_eq!(fit(None, usize::MAX, 1), base);
        assert_eq!(fit(None, usize::MAX, usize::MAX), base);
        assert_eq!(fit(Some(usize::MAX), usize::MAX, 1), base);
    }

    #[test]
    fn link_ratios_without_a_finite_value_are_never_dropped_or_ranked() {
        // At the first age the link ratios are 2 (from 5), 3 (from 4),
        // 0/0 (from 0) and 1.5 (from 2).
        let t = annual(
            2020,
            &[
                &[5.0, 10.0, 12.0],
                &[4.0, 12.0],
                &[0.0, 0.0],
                &[2.0, 3.0],
                &[7.0],
            ],
        );
        let fit = |high, low| with_exclusions(None, high, low).fit(&t, "values").unwrap();
        let all = fit(0, 0);
        assert_eq!(all.n_links[0], 4);
        close(&all.ldf[..1], &[25.0 / 11.0], 1e-12);
        // The highest finite ratio, 3 (from 4), goes, and the 0/0 stays.
        let high = fit(1, 0);
        assert_eq!(high.n_links[0], 3);
        close(&high.ldf[..1], &[13.0 / 7.0], 1e-12);
        // The lowest finite ratio, 1.5 (from 2), goes, and the 0/0 stays.
        let low = fit(0, 1);
        assert_eq!(low.n_links[0], 3);
        close(&low.ldf[..1], &[22.0 / 9.0], 1e-12);
        // Three finite ratios: dropping three leaves none, so none goes
        // (the 0/0 does not count toward the three).
        assert_eq!(fit(2, 1).n_links[0], 4);
        assert_eq!(fit(1, 1).n_links[0], 2);
    }

    #[test]
    fn drops_rank_by_value_with_ties_by_starting_value_and_keep_one() {
        // Link ratios at the first age: 2 (from 5), 2 (from 3), 4 (from 1)
        // and 1 (from 2).
        let t = annual(
            2020,
            &[
                &[5.0, 10.0, 12.0, 14.0, 16.0],
                &[3.0, 6.0, 7.0, 8.0],
                &[1.0, 4.0, 5.0],
                &[2.0, 2.0],
                &[9.0],
            ],
        );
        let simple = |high, low| {
            Development {
                average: Average::Simple,
                drop_high: high,
                drop_low: low,
                ..Default::default()
            }
            .fit(&t, "values")
            .unwrap()
        };
        close(&simple(0, 0).ldf[..1], &[2.25], 1e-12);
        close(&simple(1, 0).ldf[..1], &[5.0 / 3.0], 1e-12);
        close(&simple(0, 1).ldf[..1], &[8.0 / 3.0], 1e-12);
        close(&simple(1, 1).ldf[..1], &[2.0], 1e-12);
        // Links with three, two and one ratios: the drops leave one, then
        // would leave none (all kept), then the single one stays.
        assert_eq!(simple(1, 1).n_links, vec![2, 1, 2, 1]);
        let all = simple(2, 2);
        assert_eq!(all.n_links, vec![4, 3, 2, 1]);
        assert!(!all.excluded);
        close(&all.ldf[..1], &[2.25], 1e-12);

        // Ratios 2 (from 5), 2 (from 3), 1 (from 4): the tie at the top is
        // broken by the starting value, so the 2 from 5 is the highest.
        let tied = annual(
            2020,
            &[
                &[5.0, 10.0, 12.0, 13.0],
                &[3.0, 6.0, 7.0],
                &[4.0, 4.0],
                &[9.0],
            ],
        );
        let volume = |high, low| {
            Development {
                drop_high: high,
                drop_low: low,
                ..Default::default()
            }
            .fit(&tied, "values")
            .unwrap()
            .ldf[0]
        };
        close(&[volume(1, 0)], &[10.0 / 7.0], 1e-12);
        close(&[volume(0, 1)], &[16.0 / 8.0], 1e-12);
    }

    #[test]
    fn an_excluded_zero_starting_value_no_longer_stops_the_simple_average() {
        let t = annual(2020, &[&[0.0, 5.0], &[4.0, 6.0], &[2.0, 4.0], &[1.0]]);
        let simple = |n_periods| Development {
            average: Average::Simple,
            n_periods,
            ..Default::default()
        };
        assert!(simple(None).fit(&t, "values").is_err());
        let f = simple(Some(2)).fit(&t, "values").unwrap();
        close(&f.ldf, &[(1.5 + 2.0) / 2.0], 1e-12);
        assert_eq!(f.n_links, vec![2]);
    }

    #[test]
    fn geometric_average_is_the_geometric_mean_without_a_variance() {
        let t = annual(2020, &[&[1.0, 2.0, 3.0], &[2.0, 8.0], &[5.0]]);
        let dev = Development {
            average: Average::Geometric,
            ..Default::default()
        };
        let f = dev.fit(&t, "values").unwrap();
        close(&f.ldf, &[8.0_f64.sqrt(), 1.5], 1e-12);
        assert!(f.sigma.iter().all(|s| s.is_nan()));
        assert!(f.std_err.iter().all(|s| s.is_nan()));
        assert!(f.alpha.is_nan());
        assert_eq!(f.n_links, vec![2, 1]);
        assert_eq!(Average::Geometric.alpha(), None);
        assert_eq!(Average::Volume.alpha(), Some(1.0));
        // The latest 1 is the latest link ratio, of the two.
        let latest = Development {
            n_periods: Some(1),
            ..dev.clone()
        };
        close(&latest.fit(&t, "values").unwrap().ldf[..1], &[4.0], 1e-12);
        // A link ratio of zero has no logarithm.
        let zero = annual(2020, &[&[1.0, 0.0, 3.0], &[2.0, 8.0], &[5.0]]);
        assert!(matches!(
            dev.fit(&zero, "values"),
            Err(Error::Factor { age: 0, .. })
        ));
        let from_zero = annual(2020, &[&[0.0, 2.0, 3.0], &[2.0, 8.0], &[5.0]]);
        assert!(dev.fit(&from_zero, "values").is_err());
    }

    #[test]
    fn methods_that_assume_every_link_or_a_variance_reject_the_new_settings() {
        use crate::{ChainLadder, Mack, MackBootstrap, OdpBootstrap, OneYearMethod};
        let geometric = Development {
            average: Average::Geometric,
            ..Default::default()
        };
        let trimmed = with_exclusions(Some(3), 0, 0);

        // Mack's lifetime view takes the exclusions, not the geometric
        // average; the chain ladder takes both.
        let mack = |development: Development| Mack {
            development,
            ..Default::default()
        };
        fn not_supported<T>(r: Result<T>) -> bool {
            matches!(r, Err(Error::NotSupported { .. }))
        }
        assert!(not_supported(mack(geometric.clone()).fit(&raa(), "values")));
        assert!(mack(trimmed.clone()).fit(&raa(), "values").is_ok());
        let cl = |development: Development| ChainLadder {
            development,
            ..Default::default()
        };
        assert!(cl(geometric.clone()).fit(&raa(), "values").is_ok());
        assert!(cl(trimmed.clone()).fit(&raa(), "values").is_ok());
        let message = mack(geometric.clone())
            .fit(&raa(), "values")
            .unwrap_err()
            .to_string();
        assert!(message.contains("geometric average"), "{message}");

        // Mack's bootstrap takes the exclusions in its lifetime view.
        let boot = |development: Development| MackBootstrap {
            n_sims: 10,
            development,
            ..Default::default()
        };
        assert!(not_supported(boot(geometric.clone()).fit(&raa(), "values")));
        assert!(boot(trimmed.clone()).fit(&raa(), "values").is_ok());
        assert!(
            boot(with_exclusions(None, 1, 0))
                .fit(&raa(), "values")
                .is_ok()
        );
        assert!(boot(Development::default()).fit(&raa(), "values").is_ok());
        // Its one-year view still rejects them.
        let cl_method = OneYearMethod::ChainLadder(cl(Development::default()));
        assert!(not_supported(boot(trimmed.clone()).one_year(
            &raa(),
            "values",
            &cl_method
        )));

        // The one-year views, through the re-reserving method.
        let odp = OdpBootstrap {
            n_sims: 10,
            ..Default::default()
        };
        for development in [geometric.clone(), trimmed.clone()] {
            let method = OneYearMethod::ChainLadder(cl(development));
            assert!(not_supported(odp.one_year(&raa(), "values", &method)));
            assert!(not_supported(boot(Development::default()).one_year(
                &raa(),
                "values",
                &method
            )));
        }

        // Merz and Wüthrich's closed form.
        let fit = mack(trimmed).fit(&raa(), "values").unwrap();
        assert!(matches!(
            fit.claims_development_result(),
            Err(Error::ClaimsDevelopment(_))
        ));
        // A window past the triangle removes nothing today but slides with
        // next year's diagonal, so any setting is rejected.
        let wide = mack(with_exclusions(Some(50), 0, 0))
            .fit(&raa(), "values")
            .unwrap();
        assert!(matches!(
            wide.claims_development_result(),
            Err(Error::ClaimsDevelopment(_))
        ));
    }
}
