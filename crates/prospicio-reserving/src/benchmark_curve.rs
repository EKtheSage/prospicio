//! A growth curve fitted to a benchmark pattern (`docs/design/reserving-v02.md`,
//! decision 20; issue #637).
//!
//! A benchmark pattern comes at a few ages (12, 24, ..., 120 months, say),
//! as percents developed or factors to ultimate. [`BenchmarkCurve`] fits
//! one of Clark's growth curves ([`GrowthCurve`]) to it by weighted least
//! squares on the percents developed, which gives the pattern at any age, a
//! smooth pattern between the benchmark's ages, and a tail past its last.
//!
//! **Ages.** Benchmark ages are development ages in months from the start
//! of the origin period, as a triangle's. The curve is evaluated at those
//! ages, or, with `origin_width` set, at Clark's ages from the average date
//! of loss, the middle of an origin period of that many months
//! ([`ClarkFit::curve_age`](crate::ClarkFit::curve_age)): an age `a` at or
//! past the width `w` becomes `a - w / 2`, an earlier one `a / 2`.
//!
//! **The fitted pattern.** The percent developed at age `a` is `p(a) =
//! G(c(a)) / G(c(m))`, `c` the curve age and `m` the `max_age` at which
//! development stops (`G(c(m)) = 1` without one), as Clark's LDF method
//! develops a latest value, `latest * G(c(m)) / G(c(a))`. The objective is
//! `sum w_i (p_i - p(a_i))^2`, minimized with [`nelder_mead`] on `(ln omega,
//! ln theta)` from the straight line through the linearized benchmark
//! (`logit p` or `ln(-ln(1 - p))` on `ln c(a)`).
//!
//! **The tail.** [`BenchmarkCurveFit::pattern`] gives the fitted percents
//! developed at the ages it is asked for as a fully selected
//! [`SelectedPattern`]. Passed to the chain ladder or an expected-loss
//! method with no [`Tail`](crate::Tail), on ages that end at the triangle's
//! oldest, its level there is the tail (decision 12):
//! `1 / p(oldest)`, the curve's development past the oldest age. Ages past
//! the oldest fold into the same tail (decision 18).

use prospicio_core::Lag;
use prospicio_math::optimize::{NelderMead, nelder_mead};

use crate::clark::{GrowthCurve, shift};
use crate::error::{Error, Result};
use crate::pattern::{PatternKind, SelectedPattern};

/// The settings of a growth curve fitted to a benchmark pattern: the curve,
/// where development stops and the age convention (see the
/// [module](self) documentation).
///
/// ```
/// use prospicio_reserving::{BenchmarkCurve, ChainLadder, Development, GrowthCurve, PatternKind};
/// # use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
///
/// // A benchmark at 12 to 60 months, as percents developed.
/// let bench = [(12.0, 0.30), (24.0, 0.62), (36.0, 0.80), (48.0, 0.89), (60.0, 0.94)];
/// let fit = BenchmarkCurve::default().fit(PatternKind::PercentDeveloped, &bench, None)?;
/// assert!(fit.omega > 0.0 && fit.theta > 0.0);
/// assert!(fit.residuals.iter().all(|r| r.abs() < 0.02));
/// // Between and past the benchmark's ages.
/// assert!(fit.percent_developed(30.0) > fit.percent_developed(24.0));
/// assert!(fit.cdf(120.0) > 1.0);
///
/// // The fitted pattern at a triangle's ages, with its tail at the oldest.
/// # let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
/// # let tri = Triangle::from_long(&Long {
/// #     keys: &[],
/// #     origin: &origin,
/// #     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
/// #     values: &[("paid", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0])],
/// #     origin_grain: Grain::Year,
/// #     development_grain: Grain::Year,
/// #     cumulative: true,
/// # })?;
/// let cl = ChainLadder {
///     development: Development { pattern: Some(fit.pattern(&[12, 24, 36])?), ..Default::default() },
///     ..Default::default()
/// }
/// .fit(&tri, "paid")?;
/// assert!((cl.pattern_tail.unwrap() - fit.cdf(36.0)).abs() < 1e-12);
/// assert!((cl.ultimate[2] - 120.0 * fit.cdf(12.0)).abs() < 1e-9);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BenchmarkCurve {
    /// The growth curve.
    pub curve: GrowthCurve,
    /// Development age in months at which development stops, so the
    /// pattern is 100% there; `None` develops to infinity. At least the
    /// benchmark's last age. A benchmark that reaches 100% at a finite age
    /// needs it: an untruncated curve is below 1 at every finite age.
    pub max_age: Option<f64>,
    /// Measure the curve's ages from the average date of loss, the middle
    /// of an origin period of this many months, as Clark's methods do (12
    /// for accident years); `None` measures them from the origin period's
    /// start.
    pub origin_width: Option<f64>,
}

/// A growth curve fitted to a benchmark pattern by [`BenchmarkCurve::fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkCurveFit {
    /// The growth curve.
    pub curve: GrowthCurve,
    /// Fitted shape of the curve.
    pub omega: f64,
    /// Fitted scale of the curve, in months of curve age.
    pub theta: f64,
    /// Age in months at which development stops; `None` for infinity.
    pub max_age: Option<f64>,
    /// The origin width of Clark's average-date shift; `None` for none.
    pub origin_width: Option<f64>,
    /// The benchmark's development ages in months, in increasing order.
    pub ages: Vec<f64>,
    /// The benchmark's percents developed at `ages` (1 over its factors to
    /// ultimate when given as those).
    pub observed: Vec<f64>,
    /// The weight of each age in the least squares.
    pub weights: Vec<f64>,
    /// The fitted percents developed at `ages`.
    pub fitted: Vec<f64>,
    /// `observed - fitted` at `ages`, unweighted.
    pub residuals: Vec<f64>,
    /// The weighted sum of squared residuals, the minimized objective.
    pub sum_of_squares: f64,
}

/// The linearized percent developed: `logit p` for the log-logistic,
/// `ln(-ln(1 - p))` for the Weibull, each `omega ln x - omega ln theta`.
fn linearized(curve: GrowthCurve, p: f64) -> f64 {
    match curve {
        GrowthCurve::LogLogistic => (p / (1.0 - p)).ln(),
        GrowthCurve::Weibull => (-(-p).ln_1p()).ln(),
    }
}

impl BenchmarkCurve {
    /// Fits the curve to a benchmark: `values` are `(age in months,
    /// value)` in any order, percents developed or factors to ultimate as
    /// `kind` says, each finite and positive; age-to-age factors are an
    /// error, since they give no level to fit. `weights`, one per value in
    /// the same order, finite and not negative, weight the squared
    /// residuals; `None` weights every age 1. At least two distinct ages
    /// with positive weight are needed.
    pub fn fit(
        &self,
        kind: PatternKind,
        values: &[(f64, f64)],
        weights: Option<&[f64]>,
    ) -> Result<BenchmarkCurveFit> {
        let bad = |why: String| Err(Error::BenchmarkCurve(why));
        if kind == PatternKind::AgeToAge {
            return bad(
                "age-to-age factors give no level to fit: give percents developed or \
                 factors to ultimate"
                    .into(),
            );
        }
        if let Some(w) = weights
            && w.len() != values.len()
        {
            return bad(format!(
                "{} weights for {} values: give one weight per value",
                w.len(),
                values.len()
            ));
        }
        let mut points = Vec::with_capacity(values.len());
        for (i, &(age, value)) in values.iter().enumerate() {
            if !(age.is_finite() && age > 0.0) {
                return bad(format!("age {age} is not finite and positive"));
            }
            if !(value.is_finite() && value > 0.0) {
                return bad(format!("the value at age {age} is not finite and positive"));
            }
            let w = weights.map_or(1.0, |w| w[i]);
            if !(w.is_finite() && w >= 0.0) {
                return bad(format!(
                    "the weight at age {age} is not finite and not negative"
                ));
            }
            let p = match kind {
                PatternKind::AgeToUltimate => 1.0 / value,
                _ => value,
            };
            points.push((age, p, w));
        }
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        if let Some(pair) = points.windows(2).find(|p| p[0].0 == p[1].0) {
            return bad(format!("age {} is given twice", pair[0].0));
        }
        if points.iter().filter(|p| p.2 > 0.0).count() < 2 {
            return bad(
                "the curve's two parameters need at least two ages with positive weight".into(),
            );
        }
        let last = points[points.len() - 1].0;
        if let Some(m) = self.max_age
            && (m.is_nan() || m < last)
        {
            return Err(Error::InvalidSetting {
                name: "max_age",
                value: m,
                expected: "at least the benchmark's last age",
            });
        }
        if let Some(w) = self.origin_width
            && !(w.is_finite() && w > 0.0)
        {
            return Err(Error::InvalidSetting {
                name: "origin_width",
                value: w,
                expected: "finite and positive",
            });
        }

        let ages: Vec<f64> = points.iter().map(|p| p.0).collect();
        let observed: Vec<f64> = points.iter().map(|p| p.1).collect();
        let weights: Vec<f64> = points.iter().map(|p| p.2).collect();
        let shape = |omega: f64, theta: f64| BenchmarkCurveFit {
            curve: self.curve,
            omega,
            theta,
            max_age: self.max_age,
            origin_width: self.origin_width,
            ages: Vec::new(),
            observed: Vec::new(),
            weights: Vec::new(),
            fitted: Vec::new(),
            residuals: Vec::new(),
            sum_of_squares: f64::NAN,
        };
        let objective = |x: &[f64]| {
            let s = shape(x[0].exp(), x[1].exp());
            ages.iter()
                .zip(&observed)
                .zip(&weights)
                .map(|((&a, &p), &w)| w * (p - s.percent_developed(a)).powi(2))
                .sum::<f64>()
        };
        let x0 = self.start(&ages, &observed, &weights);
        let options = NelderMead {
            tolerance: 1e-12,
            ..NelderMead::default()
        };
        let min = nelder_mead(objective, &x0, &options);
        if !min.converged || !min.value.is_finite() {
            return bad("the least-squares search did not converge".into());
        }
        let mut fit = shape(min.x[0].exp(), min.x[1].exp());
        fit.fitted = ages.iter().map(|&a| fit.percent_developed(a)).collect();
        fit.residuals = observed
            .iter()
            .zip(&fit.fitted)
            .map(|(o, f)| o - f)
            .collect();
        fit.sum_of_squares = min.value;
        fit.ages = ages;
        fit.observed = observed;
        fit.weights = weights;
        Ok(fit)
    }

    /// The start `(ln omega, ln theta)`: the weighted straight line of the
    /// linearized percents developed below 1 on the log curve ages, or
    /// Clark's start when that line is not increasing or has too few points.
    fn start(&self, ages: &[f64], observed: &[f64], weights: &[f64]) -> [f64; 2] {
        let curve_ages: Vec<f64> = ages.iter().map(|&a| self.curve_age(a)).collect();
        let pts: Vec<(f64, f64, f64)> = curve_ages
            .iter()
            .zip(observed)
            .zip(weights)
            .filter(|&((_, &p), &w)| p > 0.0 && p < 1.0 && w > 0.0)
            .map(|((&c, &p), &w)| (c.ln(), linearized(self.curve, p), w))
            .collect();
        let sw: f64 = pts.iter().map(|p| p.2).sum();
        if pts.len() >= 2 {
            let mx = pts.iter().map(|p| p.2 * p.0).sum::<f64>() / sw;
            let my = pts.iter().map(|p| p.2 * p.1).sum::<f64>() / sw;
            let sxx: f64 = pts.iter().map(|p| p.2 * (p.0 - mx).powi(2)).sum();
            let sxy: f64 = pts.iter().map(|p| p.2 * (p.0 - mx) * (p.1 - my)).sum();
            let omega = sxy / sxx;
            if omega.is_finite() && omega > 0.0 {
                let ln_theta = mx - my / omega;
                if ln_theta.is_finite() {
                    return [omega.ln(), ln_theta];
                }
            }
        }
        self.curve.start(&curve_ages).map(f64::ln)
    }

    /// The curve age of a development age (both in months).
    fn curve_age(&self, age: f64) -> f64 {
        self.origin_width.map_or(age, |w| shift(age, w))
    }
}

impl BenchmarkCurveFit {
    /// The curve age of a development age (both in months): the age itself,
    /// or with `origin_width` set the age from the average date of loss.
    pub fn curve_age(&self, age: f64) -> f64 {
        self.origin_width.map_or(age, |w| shift(age, w))
    }

    /// The fitted percent developed at development age `age` (months),
    /// `G(c(age)) / G(c(max_age))`: 0 at age 0, 1 at `max_age` and after
    /// (at infinity without one).
    pub fn percent_developed(&self, age: f64) -> f64 {
        if let Some(m) = self.max_age
            && age >= m
        {
            return 1.0;
        }
        let g = |a: f64| self.curve.value(self.curve_age(a), self.omega, self.theta);
        let level = self.max_age.map_or(1.0, g);
        g(age) / level
    }

    /// The fitted factor from development age `age` (months) to ultimate,
    /// `1 / percent_developed(age)`: the curve's tail past `age`. Infinite
    /// at age 0.
    pub fn cdf(&self, age: f64) -> f64 {
        1.0 / self.percent_developed(age)
    }

    /// The fitted percents developed at the development `ages` (months), as
    /// a fully selected [`SelectedPattern`] of
    /// [`PatternKind::PercentDeveloped`] for the chain ladder, Mack and the
    /// expected-loss methods. With no tail on the method and ages ending at
    /// the triangle's oldest, the pattern's level there is the tail, the
    /// curve's `cdf(oldest)` (see the [module](self) documentation). An
    /// error for no ages, an age given twice, or an age where the fitted
    /// percent is not positive (age 0).
    pub fn pattern(&self, ages: &[Lag]) -> Result<SelectedPattern> {
        if ages.is_empty() {
            return Err(Error::BenchmarkCurve(
                "the pattern needs at least one age".into(),
            ));
        }
        let mut sorted = ages.to_vec();
        sorted.sort_unstable();
        if let Some(pair) = sorted.windows(2).find(|p| p[0] == p[1]) {
            return Err(Error::BenchmarkCurve(format!(
                "age {} is given twice",
                pair[0]
            )));
        }
        let values = ages
            .iter()
            .map(|&a| {
                let p = self.percent_developed(f64::from(a));
                if p > 0.0 && p.is_finite() {
                    Ok((a, p))
                } else {
                    Err(Error::BenchmarkCurve(format!(
                        "the fitted percent developed at age {a} is not positive"
                    )))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SelectedPattern::new(PatternKind::PercentDeveloped, values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ChainLadder, ClarkLdf, Development, DevelopmentColumn, Grain, Long, Month, Triangle,
    };

    /// The curve's percents developed at 12 to 120 months.
    fn exact(
        curve: GrowthCurve,
        omega: f64,
        theta: f64,
        settings: BenchmarkCurve,
    ) -> Vec<(f64, f64)> {
        let truth = BenchmarkCurveFit {
            curve,
            omega,
            theta,
            max_age: settings.max_age,
            origin_width: settings.origin_width,
            ages: vec![],
            observed: vec![],
            weights: vec![],
            fitted: vec![],
            residuals: vec![],
            sum_of_squares: 0.0,
        };
        (1..=10)
            .map(|k| {
                let a = 12.0 * f64::from(k);
                (a, truth.percent_developed(a))
            })
            .collect()
    }

    #[test]
    fn an_exact_curve_is_recovered() {
        for curve in [GrowthCurve::LogLogistic, GrowthCurve::Weibull] {
            for (max_age, origin_width) in [
                (None, None),
                (Some(180.0), None),
                (None, Some(12.0)),
                (Some(240.0), Some(12.0)),
            ] {
                let settings = BenchmarkCurve {
                    curve,
                    max_age,
                    origin_width,
                };
                let bench = exact(curve, 1.4, 30.0, settings);
                let fit = settings
                    .fit(PatternKind::PercentDeveloped, &bench, None)
                    .unwrap();
                assert!(
                    (fit.omega - 1.4).abs() < 1e-7,
                    "{curve:?} {max_age:?} {origin_width:?}: {}",
                    fit.omega
                );
                assert!(
                    (fit.theta / 30.0 - 1.0).abs() < 1e-7,
                    "{curve:?}: {}",
                    fit.theta
                );
                assert!(fit.sum_of_squares < 1e-20);
                assert!(fit.residuals.iter().all(|r| r.abs() < 1e-9));
                // The same benchmark as factors to ultimate.
                let cdfs: Vec<(f64, f64)> = bench.iter().map(|&(a, p)| (a, 1.0 / p)).collect();
                let by_cdf = settings
                    .fit(PatternKind::AgeToUltimate, &cdfs, None)
                    .unwrap();
                assert!((by_cdf.omega - fit.omega).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn a_noisy_benchmark_is_fitted_near_the_truth() {
        let settings = BenchmarkCurve::default();
        let bench = exact(GrowthCurve::LogLogistic, 1.6, 26.0, settings);
        // A fixed alternating perturbation of about 1% of the percents.
        let noisy: Vec<(f64, f64)> = bench
            .iter()
            .enumerate()
            .map(|(i, &(a, p))| (a, p * if i % 2 == 0 { 1.01 } else { 0.99 }))
            .collect();
        let fit = settings
            .fit(PatternKind::PercentDeveloped, &noisy, None)
            .unwrap();
        assert!((fit.omega / 1.6 - 1.0).abs() < 0.05, "{}", fit.omega);
        assert!((fit.theta / 26.0 - 1.0).abs() < 0.05, "{}", fit.theta);
        assert!(fit.residuals.iter().all(|r| r.abs() < 0.015));
        let ss: f64 = fit.residuals.iter().map(|r| r * r).sum();
        assert!((ss - fit.sum_of_squares).abs() < 1e-15);
        // A zero weight drops an age: the fit equals the fit without it.
        let mut w = vec![1.0; noisy.len()];
        w[3] = 0.0;
        let weighted = settings
            .fit(PatternKind::PercentDeveloped, &noisy, Some(&w))
            .unwrap();
        let mut dropped = noisy.clone();
        dropped.remove(3);
        let without = settings
            .fit(PatternKind::PercentDeveloped, &dropped, None)
            .unwrap();
        assert!((weighted.omega - without.omega).abs() < 1e-7);
        assert!((weighted.theta - without.theta).abs() < 1e-6);
    }

    #[test]
    fn bad_input_is_an_error() {
        let s = BenchmarkCurve::default();
        let ok = [(12.0, 0.4), (24.0, 0.7)];
        assert!(s.fit(PatternKind::AgeToAge, &ok, None).is_err());
        assert!(
            s.fit(PatternKind::PercentDeveloped, &ok[..1], None)
                .is_err()
        );
        assert!(
            s.fit(
                PatternKind::PercentDeveloped,
                &[(12.0, 0.4), (12.0, 0.7)],
                None
            )
            .is_err()
        );
        assert!(
            s.fit(
                PatternKind::PercentDeveloped,
                &[(0.0, 0.4), (12.0, 0.7)],
                None
            )
            .is_err()
        );
        assert!(
            s.fit(
                PatternKind::PercentDeveloped,
                &[(12.0, -0.4), (24.0, 0.7)],
                None
            )
            .is_err()
        );
        assert!(
            s.fit(PatternKind::PercentDeveloped, &ok, Some(&[1.0]))
                .is_err()
        );
        assert!(
            s.fit(PatternKind::PercentDeveloped, &ok, Some(&[1.0, 0.0]))
                .is_err()
        );
        let short = BenchmarkCurve {
            max_age: Some(18.0),
            ..s
        };
        assert!(short.fit(PatternKind::PercentDeveloped, &ok, None).is_err());
        let fit = s.fit(PatternKind::PercentDeveloped, &ok, None).unwrap();
        assert!(fit.pattern(&[]).is_err());
        assert!(fit.pattern(&[12, 12]).is_err());
        assert!(fit.pattern(&[0, 12]).is_err());
    }

    #[test]
    fn max_age_ends_development() {
        let s = BenchmarkCurve {
            max_age: Some(120.0),
            ..Default::default()
        };
        let fit = s
            .fit(
                PatternKind::PercentDeveloped,
                &[(12.0, 0.3), (24.0, 0.6), (60.0, 0.9), (120.0, 1.0)],
                None,
            )
            .unwrap();
        assert_eq!(fit.percent_developed(120.0), 1.0);
        assert_eq!(fit.percent_developed(240.0), 1.0);
        assert!(fit.residuals[3].abs() < 1e-15);
    }

    /// Clark's LDF method's ultimates are the chain ladder's under the
    /// pattern of the same curve fitted to Clark's own percents developed.
    #[test]
    fn the_pattern_reproduces_clark_in_the_chain_ladder() {
        let rows: [&[f64]; 6] = [
            &[110.0, 290.0, 370.0, 420.0, 440.0, 452.0],
            &[95.0, 300.0, 390.0, 425.0, 447.0],
            &[130.0, 320.0, 410.0, 470.0],
            &[105.0, 305.0, 395.0],
            &[120.0, 310.0],
            &[125.0],
        ];
        let (mut origin, mut age, mut paid) = (vec![], vec![], vec![]);
        for (i, row) in rows.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                origin.push(Month::january(2020 + i as i32));
                age.push(12 * (d as u32 + 1));
                paid.push(v);
            }
        }
        let tri = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&age),
            values: &[("paid", &paid)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        for curve in [GrowthCurve::LogLogistic, GrowthCurve::Weibull] {
            for max_age in [None, Some(144.0)] {
                let clark = ClarkLdf { curve, max_age }.fit(&tri, "paid").unwrap();
                let level = max_age.map_or(1.0, |m| clark.growth(m));
                let bench: Vec<(f64, f64)> = (1..=8)
                    .map(|k| {
                        let a = 12.0 * f64::from(k);
                        (a, clark.growth(a) / level)
                    })
                    .collect();
                let fit = BenchmarkCurve {
                    curve,
                    max_age,
                    origin_width: Some(12.0),
                }
                .fit(PatternKind::PercentDeveloped, &bench, None)
                .unwrap();
                assert!((fit.omega / clark.omega - 1.0).abs() < 1e-7);
                assert!((fit.theta / clark.theta - 1.0).abs() < 1e-7);

                let ages = [12, 24, 36, 48, 60, 72];
                let cl = ChainLadder {
                    development: Development {
                        pattern: Some(fit.pattern(&ages).unwrap()),
                        ..Default::default()
                    },
                    ..Default::default()
                }
                .fit(&tri, "paid")
                .unwrap();
                for k in 0..5 {
                    let a = f64::from(ages[k]);
                    let b = f64::from(ages[k + 1]);
                    let link = fit.percent_developed(b) / fit.percent_developed(a);
                    assert!((cl.ldf()[k] - link).abs() < 1e-12);
                }
                assert!((cl.pattern_tail.unwrap() - fit.cdf(72.0)).abs() < 1e-12);
                for (u, c) in cl.ultimate.iter().zip(&clark.ultimate) {
                    assert!(
                        (u / c - 1.0).abs() < 1e-7,
                        "{curve:?} {max_age:?}: {u} vs {c}"
                    );
                }
            }
        }
    }
}
