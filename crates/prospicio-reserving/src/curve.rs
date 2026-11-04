//! Development curves fitted over every age (`docs/design/reserving-v02.md`,
//! decision 31; issue #556).
//!
//! A [`DevelopmentCurve`] fits a [`CurveShape`] to a triangle's estimated
//! age-to-age factors and gives a [`DevelopmentCurveFit`]: the fitted factor
//! at every age, the residuals, the sum of squares and R-squared, and the
//! fitted factors as a fully selected [`SelectedPattern`] for
//! [`Development::pattern`](crate::Development::pattern). The observed
//! factors are kept up to `keep_observed_until` and the curve is used from
//! there on; [`DevelopmentCurve::compare`] fits several shapes to the same
//! factors for a table of fit statistics.
//!
//! **The shapes.** `k` is the 1-based development index (1 for the factor
//! from the youngest age to the next), as chainladder-python's `TailCurve`
//! reads it, for every shape but the growth curves:
//!
//! * [`CurveShape::Exponential`]: `ln(f_k - 1) = a + b k`;
//! * [`CurveShape::InversePower`]: `ln(f_k - 1) = a + b ln k`;
//! * [`CurveShape::Hoerl`]: `ln(f_k - 1) = a + b k + c ln k`;
//! * [`CurveShape::Sherman`]: `f_k - 1 = a (k + c)^(-b)` (Sherman 1984);
//! * [`CurveShape::Growth`]: Clark's growth curve `G` as percents developed
//!   at the development ages `t` in months, `f_k = G(t_(k+1)) / G(t_k)`.
//!
//! The first three are least-squares lines on `ln(f - 1)`, the last two
//! nonlinear least squares on the factors themselves. The points are the
//! same for every shape: the estimated factors in `fit_period` above
//! 1.00001 (chainladder-python's `reg_threshold`).
//!
//! **Fit statistics** are on the factor scale for every shape, so that
//! [`DevelopmentCurve::compare`] gives one table: the residual is `f_k -
//! fitted_k`, the sum of squares and R-squared are over the points of the
//! fit. For the three log-linear shapes this is not the quantity the fit
//! minimizes.
//!
//! **A selected pattern carries no parameter error** (decision 12): the
//! fitted factors are selected at full weight, so Mack's standard errors
//! and the bootstraps charge them none, and the curve's own estimation error
//! is not reflected in the reported uncertainty.

use prospicio_core::Lag;
use prospicio_math::linalg::solve;
use prospicio_math::optimize::{NelderMead, nelder_mead};

use crate::clark::GrowthCurve;
use crate::development::DevelopmentFit;
use crate::error::{Error, Result};
use crate::pattern::{PatternKind, SelectedPattern};
use crate::tail::CurveShape;

/// Lower bound on the factors a curve is fitted to, chainladder-python's
/// `reg_threshold`.
pub(crate) const CURVE_THRESHOLD: f64 = 1.00001;

/// A curve fitted over every development age (see the [module](self)
/// documentation).
///
/// ```
/// use prospicio_reserving::{
///     ChainLadder, CurveShape, Development, DevelopmentCurve, Tail,
/// };
/// # use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// # let origin = [2020, 2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021, 2022, 2022, 2022, 2023, 2023, 2024]
/// #     .map(Month::january);
/// # let tri = Triangle::from_long(&Long {
/// #     keys: &[],
/// #     origin: &origin,
/// #     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 60, 12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
/// #     values: &[("paid", &[
/// #         100.0, 180.0, 215.0, 228.0, 233.0, 110.0, 196.0, 236.0, 249.0, 120.0, 218.0, 259.0,
/// #         130.0, 232.0, 140.0,
/// #     ])],
/// #     origin_grain: Grain::Year,
/// #     development_grain: Grain::Year,
/// #     cumulative: true,
/// # })?;
/// let dev = Development::default().fit(&tri, "paid")?;
/// let curve = DevelopmentCurve { shape: CurveShape::InversePower, ..Default::default() };
/// let fit = curve.fit(&dev)?;
/// assert_eq!(fit.fitted.len(), 4);
/// assert!(fit.r_squared > 0.9);
///
/// // The fitted factors as the selected pattern, and the curve's tail.
/// let cl = ChainLadder {
///     development: Development { pattern: Some(fit.pattern()), ..Default::default() },
///     tail: Tail::from(fit.tail(100)),
/// }
/// .fit(&tri, "paid")?;
/// assert_eq!(&cl.ldf()[..4], &fit.fitted[..]);
///
/// // Several shapes on the same factors.
/// let table = curve.compare(&dev, &[CurveShape::Exponential, CurveShape::InversePower])?;
/// assert_eq!(table.len(), 2);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DevelopmentCurve {
    /// The curve fitted to the factors.
    pub shape: CurveShape,
    /// Ages whose factors enter the fit, as [`TailCurve`](crate::TailCurve)'s
    /// `fit_period`: from the last age at or before the first (inclusive)
    /// to the last age at or before the second (exclusive). `None` is
    /// open-ended, and a start before the youngest age starts there.
    pub fit_period: (Option<Lag>, Option<Lag>),
    /// The observed factors are kept before this age and the curve is used
    /// from the first age at or after it on, as a
    /// [`TailCurve`](crate::TailCurve) attaches; `None` uses the curve at
    /// every age. An age past the last factor's is an error: it would leave
    /// the curve nothing to replace.
    pub keep_observed_until: Option<Lag>,
}

/// A curve fitted by [`DevelopmentCurve::fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopmentCurveFit {
    /// The settings it was fitted with.
    pub curve: DevelopmentCurve,
    /// Age of each development position; factor `k` links age `k` to
    /// `k + 1`.
    pub development: Vec<Lag>,
    /// The curve's parameters: `[a, b]` for the exponential and inverse
    /// power, `[a, b, c]` for Hoerl and Sherman, `[omega, theta]` for a
    /// growth curve.
    pub parameters: Vec<f64>,
    /// The triangle's estimated factors, NaN where one could not be
    /// estimated.
    pub observed: Vec<f64>,
    /// The curve's factor at every development position.
    pub fitted: Vec<f64>,
    /// `observed - fitted` at every development position, NaN where the
    /// observed factor is missing.
    pub residuals: Vec<f64>,
    /// Whether each position's factor entered the fit: in `fit_period`,
    /// finite and above 1.00001.
    pub in_fit: Vec<bool>,
    /// The first position the curve replaces: the observed factors are
    /// kept before it.
    pub attachment: usize,
    /// The selected factors: `observed` before `attachment`, `fitted` from
    /// it on.
    pub ldf: Vec<f64>,
    /// The number of points in the fit.
    pub n_points: usize,
    /// The number of parameters of the curve.
    pub n_parameters: usize,
    /// The sum of squared residuals over the points of the fit, on the
    /// factor scale.
    pub sum_of_squares: f64,
    /// `1 - SSE / SST` over the points of the fit, `SST` the sum of squares
    /// of their factors about their mean; NaN when the factors are all
    /// equal.
    pub r_squared: f64,
    /// `1 - (1 - R^2) (n - 1) / (n - p)`, with `n` points and `p`
    /// parameters; NaN when `n <= p`.
    pub adjusted_r_squared: f64,
    shape_fit: ShapeFit,
}

impl DevelopmentCurve {
    /// Fits the curve to the estimated factors of `development`
    /// ([`DevelopmentFit::ldf`], the triangle's own, not a selected
    /// pattern's). An error when the fit period holds too few factors above
    /// 1.00001 for the shape (two for the exponential and inverse power,
    /// three for Hoerl and a growth curve, four for Sherman), when a
    /// nonlinear fit does not converge, or when `keep_observed_until` leaves
    /// nothing to replace.
    pub fn fit(&self, development: &DevelopmentFit) -> Result<DevelopmentCurveFit> {
        let ages = &development.development;
        let observed = &development.ldf;
        let n_links = observed.len();
        if n_links == 0 || ages.len() != n_links + 1 {
            return Err(Error::DevelopmentCurve(
                "the development fit has no factors".into(),
            ));
        }
        let points = curve_points(observed, ages, self.fit_period)?;
        let shape_fit = ShapeFit::fit(self.shape, &points, ages)
            .map_err(|why| Error::DevelopmentCurve(why.into()))?;
        let attachment = match self.keep_observed_until {
            None => 0,
            Some(age) => match ages[..n_links].iter().position(|&a| a >= age) {
                Some(p) => p,
                None => {
                    return Err(Error::DevelopmentCurve(format!(
                        "keep_observed_until = {age} is past the last factor's age, {}: the \
                         curve would replace no factor",
                        ages[n_links - 1]
                    )));
                }
            },
        };
        let fitted: Vec<f64> = (0..n_links).map(|p| shape_fit.factor(p, ages)).collect();
        if fitted.iter().any(|f| !(f.is_finite() && *f > 0.0)) {
            return Err(Error::DevelopmentCurve(
                "the fitted curve gives a factor that is not finite and positive".into(),
            ));
        }
        let residuals: Vec<f64> = observed.iter().zip(&fitted).map(|(o, f)| o - f).collect();
        let mut in_fit = vec![false; n_links];
        for &(p, _) in &points {
            in_fit[p] = true;
        }
        let ldf = (0..n_links)
            .map(|p| {
                if p < attachment {
                    observed[p]
                } else {
                    fitted[p]
                }
            })
            .collect();
        let n = points.len();
        let sse: f64 = points.iter().map(|&(p, _)| residuals[p].powi(2)).sum();
        let mean = points.iter().map(|p| p.1).sum::<f64>() / n as f64;
        let sst: f64 = points.iter().map(|p| (p.1 - mean).powi(2)).sum();
        let r_squared = if sst > 0.0 { 1.0 - sse / sst } else { f64::NAN };
        let n_parameters = shape_fit.params.len();
        let adjusted_r_squared = if n > n_parameters {
            1.0 - (1.0 - r_squared) * (n - 1) as f64 / (n - n_parameters) as f64
        } else {
            f64::NAN
        };
        Ok(DevelopmentCurveFit {
            curve: *self,
            development: ages.clone(),
            parameters: shape_fit.params.clone(),
            observed: observed.clone(),
            fitted,
            residuals,
            in_fit,
            attachment,
            ldf,
            n_points: n,
            n_parameters,
            sum_of_squares: sse,
            r_squared,
            adjusted_r_squared,
            shape_fit,
        })
    }

    /// Fits each of `shapes`, with this curve's `fit_period` and
    /// `keep_observed_until`, to the same factors of `development`: a row
    /// per shape of a table of fit statistics (`n_points`, `n_parameters`,
    /// `sum_of_squares`, `r_squared`, `adjusted_r_squared`). An error, the
    /// first shape's, when a shape cannot be fitted.
    pub fn compare(
        &self,
        development: &DevelopmentFit,
        shapes: &[CurveShape],
    ) -> Result<Vec<DevelopmentCurveFit>> {
        shapes
            .iter()
            .map(|&shape| DevelopmentCurve { shape, ..*self }.fit(development))
            .collect()
    }
}

impl DevelopmentCurveFit {
    /// The curve's factor at a development position (0-based, factor `p`
    /// links age `p` to `p + 1`), past the oldest age too. Past the oldest
    /// age a growth curve reads its ages on at the last spacing.
    pub fn factor(&self, position: usize) -> f64 {
        self.shape_fit.factor(position, &self.development)
    }

    /// The fitted factors from `attachment` on as a fully selected
    /// age-to-age [`SelectedPattern`] for
    /// [`Development::pattern`](crate::Development::pattern); the factors
    /// before it stay estimated.
    pub fn pattern(&self) -> SelectedPattern {
        SelectedPattern::new(
            PatternKind::AgeToAge,
            (self.attachment..self.fitted.len())
                .map(|p| (self.development[p], self.fitted[p]))
                .collect(),
        )
    }

    /// The curve's development past the oldest age, as a
    /// [`TailCurve`](crate::TailCurve) extrapolates it: the product of the
    /// curve's factors over the `extrap_periods` periods after the oldest
    /// age (1 for none).
    pub fn tail(&self, extrap_periods: usize) -> f64 {
        let n = self.fitted.len();
        (n..n + extrap_periods).map(|p| self.factor(p)).product()
    }
}

/// The `(position, factor)` points a curve is fitted to: the factors in
/// `fit_period` (positions as chainladder-python reads them, the last age
/// at or before each end) above 1.00001.
pub(crate) fn curve_points(
    estimated: &[f64],
    ages: &[Lag],
    fit_period: (Option<Lag>, Option<Lag>),
) -> Result<Vec<(usize, f64)>> {
    let n_links = estimated.len();
    if ages.len() != n_links + 1 {
        return Err(Error::Tail("the development fit has no ages"));
    }
    let position = |age: Option<Lag>, open: usize| match age {
        Some(age) => ages[..=n_links]
            .iter()
            .rposition(|&a| a <= age)
            .unwrap_or(0),
        None => open,
    };
    let range = position(fit_period.0, 0)..position(fit_period.1, n_links);
    Ok(estimated
        .iter()
        .enumerate()
        .filter(|&(k, &f)| f > CURVE_THRESHOLD && range.contains(&k))
        .map(|(k, &f)| (k, f))
        .collect())
}

/// A shape fitted to `(position, factor)` points: what [`TailCurve`]
/// (crate::TailCurve) and [`DevelopmentCurve`] share.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ShapeFit {
    shape: CurveShape,
    params: Vec<f64>,
}

impl ShapeFit {
    /// Fits `shape` to `points`; `ages` are the development ages, read by a
    /// growth curve.
    pub(crate) fn fit(
        shape: CurveShape,
        points: &[(usize, f64)],
        ages: &[Lag],
    ) -> std::result::Result<Self, &'static str> {
        let k = |p: usize| (p + 1) as f64;
        let log_points = |x_of: fn(f64) -> f64| -> Vec<(f64, f64)> {
            points
                .iter()
                .map(|&(p, f)| (x_of(k(p)), (f - 1.0).ln()))
                .collect()
        };
        let params = match shape {
            CurveShape::Exponential | CurveShape::InversePower => {
                let x_of = if shape == CurveShape::Exponential {
                    (|k| k) as fn(f64) -> f64
                } else {
                    f64::ln
                };
                let (a, b) = least_squares_line(&log_points(x_of))
                    .ok_or("a curve needs two factors above 1 in the fit period")?;
                vec![a, b]
            }
            CurveShape::Hoerl => fit_hoerl(points)?,
            CurveShape::Sherman => fit_sherman(points)?,
            CurveShape::Growth(curve) => fit_growth(curve, points, ages)?,
        };
        Ok(Self { shape, params })
    }

    /// The fitted factor at 0-based position `p`, past the oldest age too.
    pub(crate) fn factor(&self, p: usize, ages: &[Lag]) -> f64 {
        let k = (p + 1) as f64;
        let q = &self.params;
        match self.shape {
            CurveShape::Exponential => 1.0 + (q[0] + q[1] * k).exp(),
            CurveShape::InversePower => 1.0 + (q[0] + q[1] * k.ln()).exp(),
            CurveShape::Hoerl => 1.0 + (q[0] + q[1] * k + q[2] * k.ln()).exp(),
            CurveShape::Sherman => 1.0 + sherman(q[0], q[1], q[2], k),
            CurveShape::Growth(curve) => {
                curve.value(age_at(ages, p + 1), q[0], q[1])
                    / curve.value(age_at(ages, p), q[0], q[1])
            }
        }
    }
}

/// Intercept and slope of the least-squares line through `points`, or
/// `None` with fewer than two distinct `x`.
pub(crate) fn least_squares_line(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
    if sxx == 0.0 {
        return None;
    }
    let sxy: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
    let slope = sxy / sxx;
    Some((mean_y - slope * mean_x, slope))
}

/// The development age in months at position `p`, read on at the last
/// spacing past the oldest age.
fn age_at(ages: &[Lag], p: usize) -> f64 {
    let n = ages.len();
    if p < n {
        return ages[p] as f64;
    }
    let step = if n >= 2 {
        (ages[n - 1] - ages[n - 2]) as f64
    } else {
        12.0
    };
    ages[n - 1] as f64 + step * (p + 1 - n) as f64
}

/// Hoerl's curve, `ln(f - 1) = a + b k + c ln k`, by least squares on the
/// normal equations, centred for conditioning.
fn fit_hoerl(points: &[(usize, f64)]) -> std::result::Result<Vec<f64>, &'static str> {
    const NEED: &str = "the Hoerl curve needs three factors above 1 in the fit period";
    if points.len() < 3 {
        return Err(NEED);
    }
    let rows: Vec<[f64; 3]> = points
        .iter()
        .map(|&(p, f)| {
            let k = (p + 1) as f64;
            [k, k.ln(), (f - 1.0).ln()]
        })
        .collect();
    let n = rows.len() as f64;
    let mean = |j: usize| rows.iter().map(|r| r[j]).sum::<f64>() / n;
    let (m1, m2, my) = (mean(0), mean(1), mean(2));
    let (mut s11, mut s12, mut s22, mut s1y, mut s2y) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for r in &rows {
        let (x1, x2, y) = (r[0] - m1, r[1] - m2, r[2] - my);
        s11 += x1 * x1;
        s12 += x1 * x2;
        s22 += x2 * x2;
        s1y += x1 * y;
        s2y += x2 * y;
    }
    let bc = solve(vec![s11, s12, s12, s22], vec![s1y, s2y], 2)
        .ok_or("the Hoerl curve needs three distinct development indices")?;
    let (b, c) = (bc[0], bc[1]);
    let a = my - b * m1 - c * m2;
    if !(a.is_finite() && b.is_finite() && c.is_finite()) {
        return Err(NEED);
    }
    Ok(vec![a, b, c])
}

/// `a (k + c)^(-b)`.
fn sherman(a: f64, b: f64, c: f64, k: f64) -> f64 {
    a * (k + c).powf(-b)
}

/// Sherman's inverse power curve, `f - 1 = a (k + c)^(-b)`, by nonlinear
/// least squares on `f - 1`: for given `(b, c)` the best `a` is `sum y x /
/// sum x^2` with `x = (k + c)^(-b)`, so the search runs over `(b, u)` with
/// `c = exp(u) - 1 > -1` (every `k + c` positive, `k >= 1`), from the
/// inverse power's line at `c = 0`, and a Gauss-Newton step polishes the
/// three parameters.
fn fit_sherman(points: &[(usize, f64)]) -> std::result::Result<Vec<f64>, &'static str> {
    if points.len() < 4 {
        return Err("the Sherman curve needs four factors above 1 in the fit period");
    }
    let data: Vec<(f64, f64)> = points
        .iter()
        .map(|&(p, f)| ((p + 1) as f64, f - 1.0))
        .collect();
    let best_a = |b: f64, c: f64| {
        let (mut sxy, mut sxx) = (0.0, 0.0);
        for &(k, y) in &data {
            let x = (k + c).powf(-b);
            sxy += x * y;
            sxx += x * x;
        }
        sxy / sxx
    };
    let sse = |a: f64, b: f64, c: f64| {
        data.iter()
            .map(|&(k, y)| (y - sherman(a, b, c, k)).powi(2))
            .sum::<f64>()
    };
    let profile = |v: &[f64]| {
        let (b, c) = (v[0], v[1].exp() - 1.0);
        sse(best_a(b, c), b, c)
    };
    let logs: Vec<(f64, f64)> = data.iter().map(|&(k, y)| (k.ln(), y.ln())).collect();
    let b0 = least_squares_line(&logs).map_or(1.0, |(_, slope)| -slope);
    let b0 = if b0.is_finite() { b0 } else { 1.0 };
    let options = NelderMead {
        tolerance: 1e-14,
        max_iterations: 20_000,
        ..NelderMead::default()
    };
    // From the inverse power (c = 0), and from a few shifts, keeping the
    // best: the profile can have a shallow second valley.
    let mut best: Option<(f64, Vec<f64>)> = None;
    for u0 in [0.0, 1.0, 2.0] {
        let m = nelder_mead(profile, &[b0, u0], &options);
        if m.value.is_finite() && best.as_ref().is_none_or(|(v, _)| m.value < *v) {
            best = Some((m.value, m.x));
        }
    }
    let (_, v) = best.ok_or("the Sherman curve could not be fitted")?;
    let (b, c) = (v[0], v[1].exp() - 1.0);
    let start = vec![best_a(b, c), b, c];
    let polished = gauss_newton(&start, |q| {
        if q[2].is_nan() || q[2] <= -1.0 {
            return None;
        }
        let mut r = Vec::with_capacity(data.len());
        let mut j = Vec::with_capacity(data.len());
        for &(k, y) in &data {
            let x = (k + q[2]).powf(-q[1]);
            let m = q[0] * x;
            r.push(y - m);
            j.push(vec![x, -m * (k + q[2]).ln(), -q[1] * m / (k + q[2])]);
        }
        Some((r, j))
    });
    let q = polished.unwrap_or(start);
    if q.iter().all(|v| v.is_finite()) && q[2] > -1.0 {
        Ok(q)
    } else {
        Err("the Sherman curve could not be fitted")
    }
}

/// A growth curve as percents developed, `f_k = G(t_(k+1)) / G(t_k)` at
/// the development ages `t` in months, by nonlinear least squares on the
/// factors: a simplex search on `(ln omega, ln theta)` from the line through
/// the linearized percents developed the factors imply (with no development
/// past the oldest age), then a Gauss-Newton polish.
fn fit_growth(
    curve: GrowthCurve,
    points: &[(usize, f64)],
    ages: &[Lag],
) -> std::result::Result<Vec<f64>, &'static str> {
    if points.len() < 3 {
        return Err("a growth curve needs three factors above 1 in the fit period");
    }
    let model = |omega: f64, theta: f64, p: usize| {
        curve.value(age_at(ages, p + 1), omega, theta) / curve.value(age_at(ages, p), omega, theta)
    };
    let objective = |x: &[f64]| {
        let (omega, theta) = (x[0].exp(), x[1].exp());
        points
            .iter()
            .map(|&(p, f)| (f - model(omega, theta, p)).powi(2))
            .sum::<f64>()
    };
    // Percents developed implied by the factors in the fit, to the last of
    // them: p(t_j) = 1 / prod_(i >= j) f_i.
    let mut implied = Vec::with_capacity(points.len());
    let mut cdf = 1.0;
    for &(p, f) in points.iter().rev() {
        cdf *= f;
        implied.push((age_at(ages, p).ln(), 1.0 / cdf));
    }
    let line: Vec<(f64, f64)> = implied
        .iter()
        .filter(|(_, pd)| *pd > 0.0 && *pd < 1.0)
        .map(|&(x, pd)| (x, crate::benchmark_curve::linearized(curve, pd)))
        .collect();
    let mut starts = Vec::new();
    if let Some((intercept, omega)) = least_squares_line(&line)
        && omega.is_finite()
        && omega > 0.0
    {
        starts.push([omega.ln(), -intercept / omega]);
    }
    let fit_ages: Vec<f64> = points.iter().map(|&(p, _)| age_at(ages, p)).collect();
    starts.push(curve.start(&fit_ages).map(f64::ln));
    let options = NelderMead {
        tolerance: 1e-14,
        max_iterations: 20_000,
        ..NelderMead::default()
    };
    let mut best: Option<(f64, Vec<f64>)> = None;
    for x0 in starts {
        let m = nelder_mead(objective, &x0, &options);
        if m.value.is_finite() && best.as_ref().is_none_or(|(v, _)| m.value < *v) {
            best = Some((m.value, m.x));
        }
    }
    let (_, x) = best.ok_or("the growth curve could not be fitted")?;
    let start = vec![x[0].exp(), x[1].exp()];
    let polished = gauss_newton(&start, |q| {
        let (omega, theta) = (q[0], q[1]);
        if !(omega > 0.0 && theta > 0.0) {
            return None;
        }
        let mut r = Vec::with_capacity(points.len());
        let mut j = Vec::with_capacity(points.len());
        for &(p, f) in points {
            let (t0, t1) = (age_at(ages, p), age_at(ages, p + 1));
            let (g0, g1) = (curve.value(t0, omega, theta), curve.value(t1, omega, theta));
            let (d0, d1) = (
                curve.gradient(t0, omega, theta),
                curve.gradient(t1, omega, theta),
            );
            r.push(f - g1 / g0);
            j.push(
                (0..2)
                    .map(|i| (d1[i] * g0 - g1 * d0[i]) / (g0 * g0))
                    .collect(),
            );
        }
        Some((r, j))
    });
    let q = polished.unwrap_or(start);
    let (omega, theta) = (q[0], q[1]);
    if !(omega.is_finite() && theta.is_finite() && omega > 1e-6 && omega < 1e6 && theta > 0.0) {
        return Err(
            "the growth curve ran off to the edge of its parameters: the factors do not \
             follow a growth curve",
        );
    }
    Ok(q)
}

/// Polishes a least-squares fit of `y = m(q)` by Levenberg-Marquardt steps
/// on the normal equations, from `start`. `residuals` gives `y - m(q)` and
/// the Jacobian of `m` by row, or `None` outside the parameters' range.
/// `None` when the start itself is outside the range.
fn gauss_newton(
    start: &[f64],
    residuals: impl Fn(&[f64]) -> Option<(Vec<f64>, Vec<Vec<f64>>)>,
) -> Option<Vec<f64>> {
    let n = start.len();
    let sse = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>();
    let mut q = start.to_vec();
    let (mut r, mut j) = residuals(&q)?;
    let mut value = sse(&r);
    let mut lambda = 1e-6;
    for _ in 0..200 {
        let mut a = vec![0.0; n * n];
        let mut g = vec![0.0; n];
        for (row, ri) in j.iter().zip(&r) {
            for u in 0..n {
                g[u] += row[u] * ri;
                for v in 0..n {
                    a[u * n + v] += row[u] * row[v];
                }
            }
        }
        let mut improved = false;
        while lambda < 1e12 {
            let mut damped = a.clone();
            for u in 0..n {
                damped[u * n + u] *= 1.0 + lambda;
            }
            let Some(step) = solve(damped, g.clone(), n) else {
                lambda *= 10.0;
                continue;
            };
            let trial: Vec<f64> = q.iter().zip(&step).map(|(x, s)| x + s).collect();
            if let Some((tr, tj)) = residuals(&trial) {
                let tv = sse(&tr);
                if tv.is_finite() && tv <= value {
                    let small = step
                        .iter()
                        .zip(&q)
                        .all(|(s, x)| s.abs() <= 1e-14 * x.abs().max(1e-8));
                    let done = small || value - tv <= 1e-16 * value;
                    q = trial;
                    r = tr;
                    j = tj;
                    value = tv;
                    lambda = (lambda / 10.0).max(1e-12);
                    improved = true;
                    if done {
                        return Some(q);
                    }
                    break;
                }
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    Some(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::development::Development;
    use crate::tail::{Tail, TailCurve};
    use crate::triangle::tests::raa;

    const AGES: [Lag; 11] = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132];

    /// A development fit with the given factors at annual ages.
    fn fit_of(ldf: &[f64]) -> DevelopmentFit {
        let n = ldf.len();
        DevelopmentFit {
            development: AGES[..=n].to_vec(),
            ldf: ldf.to_vec(),
            sigma: vec![f64::NAN; n],
            std_err: vec![f64::NAN; n],
            volume: vec![1.0; n],
            alpha: 1.0,
            n_links: vec![1; n],
            excluded: false,
            selected: vec![None; n],
            weight: vec![0.0; n],
        }
    }

    /// Each shape recovers the parameters of factors generated from it.
    #[test]
    fn round_trip_recovers_each_shape() {
        let cases: [(CurveShape, Vec<f64>); 6] = [
            (CurveShape::Exponential, vec![0.8, -0.6]),
            (CurveShape::InversePower, vec![0.5, -2.2]),
            (CurveShape::Hoerl, vec![0.9, -0.3, -1.1]),
            (CurveShape::Sherman, vec![3.0, 2.4, 0.6]),
            (
                CurveShape::Growth(GrowthCurve::LogLogistic),
                vec![1.6, 22.0],
            ),
            (CurveShape::Growth(GrowthCurve::Weibull), vec![1.1, 30.0]),
        ];
        for (shape, truth) in cases {
            let generator = ShapeFit {
                shape,
                params: truth.clone(),
            };
            let ldf: Vec<f64> = (0..9).map(|p| generator.factor(p, &AGES)).collect();
            let fit = DevelopmentCurve {
                shape,
                ..Default::default()
            }
            .fit(&fit_of(&ldf))
            .unwrap_or_else(|e| panic!("{shape:?}: {e}"));
            for (got, want) in fit.parameters.iter().zip(&truth) {
                assert!(
                    (got - want).abs() <= 1e-7 * want.abs().max(1.0),
                    "{shape:?}: got {:?}, want {truth:?}",
                    fit.parameters
                );
            }
            assert!(fit.sum_of_squares < 1e-20, "{shape:?}");
            assert!((fit.r_squared - 1.0).abs() < 1e-12, "{shape:?}");
        }
    }

    /// The exponential and inverse power over every age are the
    /// `TailCurve` attached at the youngest age, factor for factor.
    #[test]
    fn matches_tail_curve_attached_at_the_youngest_age() {
        let dev = Development::default().fit(&raa(), "values").unwrap();
        for shape in [
            CurveShape::Exponential,
            CurveShape::InversePower,
            CurveShape::Hoerl,
            CurveShape::Sherman,
            CurveShape::Growth(GrowthCurve::Weibull),
        ] {
            let curve = DevelopmentCurve {
                shape,
                ..Default::default()
            }
            .fit(&dev)
            .unwrap();
            let tail = Tail::Curve(TailCurve {
                curve: shape,
                attachment_age: Some(12),
                ..Default::default()
            })
            .fit(&dev)
            .unwrap();
            assert_eq!(tail.attachment, 0);
            assert_eq!(&tail.ldf[..9], &curve.fitted[..], "{shape:?}");
            let rel = (tail.factor / curve.tail(100) - 1.0).abs();
            assert!(
                rel < 1e-12,
                "{shape:?}: {} against {}",
                tail.factor,
                curve.tail(100)
            );
        }
    }

    #[test]
    fn keeps_the_observed_factors_before_the_attachment() {
        let dev = Development::default().fit(&raa(), "values").unwrap();
        let fit = DevelopmentCurve {
            shape: CurveShape::Sherman,
            keep_observed_until: Some(40),
            ..Default::default()
        }
        .fit(&dev)
        .unwrap();
        assert_eq!(fit.attachment, 3);
        assert_eq!(&fit.ldf[..3], &dev.ldf[..3]);
        assert_eq!(&fit.ldf[3..], &fit.fitted[3..]);
        let p = fit.pattern();
        assert_eq!(p.kind, PatternKind::AgeToAge);
        assert_eq!(p.values.first(), Some(&(48, fit.fitted[3])));
        assert_eq!(p.values.len(), 6);
        // Past the last factor's age nothing is left to replace.
        let err = DevelopmentCurve {
            keep_observed_until: Some(120),
            ..Default::default()
        }
        .fit(&dev)
        .unwrap_err();
        assert!(matches!(err, Error::DevelopmentCurve(_)), "{err}");
    }

    #[test]
    fn statistics_are_on_the_factor_scale_over_the_points_of_the_fit() {
        let dev = Development::default().fit(&raa(), "values").unwrap();
        let fit = DevelopmentCurve {
            shape: CurveShape::Hoerl,
            fit_period: (Some(24), None),
            ..Default::default()
        }
        .fit(&dev)
        .unwrap();
        assert!(!fit.in_fit[0] && fit.in_fit[1..].iter().all(|&b| b));
        assert_eq!(fit.n_points, 8);
        let sse: f64 = (1..9).map(|p| (dev.ldf[p] - fit.fitted[p]).powi(2)).sum();
        assert!((fit.sum_of_squares - sse).abs() < 1e-15);
        let mean = dev.ldf[1..].iter().sum::<f64>() / 8.0;
        let sst: f64 = dev.ldf[1..].iter().map(|f| (f - mean).powi(2)).sum();
        assert!((fit.r_squared - (1.0 - sse / sst)).abs() < 1e-14);
        let adj = 1.0 - (1.0 - fit.r_squared) * 7.0 / 5.0;
        assert!((fit.adjusted_r_squared - adj).abs() < 1e-14);
        assert!((fit.residuals[0] - (dev.ldf[0] - fit.fitted[0])).abs() < 1e-15);
    }

    #[test]
    fn too_few_points_are_an_error() {
        let dev = fit_of(&[1.5, 1.2, 1.1]);
        let err = DevelopmentCurve {
            shape: CurveShape::Sherman,
            ..Default::default()
        }
        .fit(&dev)
        .unwrap_err();
        assert!(err.to_string().contains("four factors"), "{err}");
        assert!(
            DevelopmentCurve {
                shape: CurveShape::Hoerl,
                ..Default::default()
            }
            .fit(&fit_of(&[1.5, 1.2]))
            .is_err()
        );
    }
}
