//! Clark's growth-curve methods (`docs/design/reserving-v02.md`, decision
//! 5): [`ClarkLdf`] and [`ClarkCapeCod`], with the definitions of R
//! ChainLadder's `ClarkLDF` and `ClarkCapeCod`.
//!
//! Clark (2003, *LDF Curve-Fitting and Stochastic Reserving: A Maximum
//! Likelihood Approach*) models the expected incremental loss of origin `i`
//! between ages `x` and `y` as `U_i * (G(y) - G(x))`: an expected ultimate
//! times the share of it a growth curve `G` develops in the interval. The
//! LDF method estimates each `U_i`; the Cape Cod method sets
//! `U_i = ELR * exposure_i` with one expected loss ratio. Incremental losses
//! are over-dispersed Poisson: their variance is `scale` times their mean.
//!
//! Ages are measured from the average date of loss, which R's
//! `adol = TRUE` puts at the middle of the origin period: an age `a` at or
//! past the origin period's length `w` becomes `a - w / 2`, and an earlier
//! age becomes `a * (1 - 1 / 2) = a / 2`, so the curve starts at 0 when the
//! origin period starts. A `max_age` truncates the curve: development stops
//! at `max_age` (shifted the same way) instead of at infinity.
//!
//! Given `(omega, theta)` the maximum-likelihood `U_i` (or `ELR`) has a
//! closed form, the sum of the origin's (all) incremental losses over the
//! sum of their curve shares, so the likelihood is profiled onto the two
//! curve parameters and minimized with [`nelder_mead`] on
//! `(ln omega, ln theta)`. The parameter covariance is the inverse of the
//! observed Fisher information (the negative Hessian of the log-likelihood
//! in all parameters, from analytic derivatives of the curve) times the
//! scale, and a reserve's parameter risk is the delta method on it.

use act_math::linalg::solve;
use act_math::optimize::{NelderMead, nelder_mead};

use crate::chain_ladder::{ChainLadder, ChainLadderFit};
use crate::error::{Error, Result};
use crate::expected_loss::origin_exposure;
use crate::segments::{ReserveFit, SegmentFits, fit_each, fit_each_with_exposure};
use crate::triangle::{Segment, Triangle};

/// The growth curve `G` of Clark's methods: the share of the ultimate
/// developed by age `x`, with shape `omega` and scale `theta` (in the
/// triangle's age unit, months).
///
/// ```
/// use act_reserving::GrowthCurve;
///
/// // At age theta the log-logistic has developed half the ultimate.
/// assert_eq!(GrowthCurve::LogLogistic.value(48.0, 1.5, 48.0), 0.5);
/// let w = GrowthCurve::Weibull.value(48.0, 1.5, 48.0);
/// assert!((w - (1.0 - (-1.0f64).exp())).abs() < 1e-15);
/// assert_eq!(GrowthCurve::Weibull.value(f64::INFINITY, 1.5, 48.0), 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrowthCurve {
    /// `G(x) = x^omega / (x^omega + theta^omega)`, R's `"loglogistic"`.
    #[default]
    LogLogistic,
    /// `G(x) = 1 - exp(-(x / theta)^omega)`, R's `"weibull"`.
    Weibull,
}

impl GrowthCurve {
    /// `G(age)`: 0 at age 0 or below, 1 at infinity.
    pub fn value(self, age: f64, omega: f64, theta: f64) -> f64 {
        if age <= 0.0 {
            return 0.0;
        }
        if age == f64::INFINITY {
            return 1.0;
        }
        match self {
            Self::LogLogistic => 1.0 / (1.0 + (theta / age).powf(omega)),
            Self::Weibull => -(-(age / theta).powf(omega)).exp_m1(),
        }
    }

    /// `(dG/domega, dG/dtheta)` at `age`; 0 at age 0 and at infinity.
    fn gradient(self, age: f64, omega: f64, theta: f64) -> [f64; 2] {
        if !(age > 0.0 && age.is_finite()) {
            return [0.0; 2];
        }
        let l = (age / theta).ln();
        let g = match self {
            Self::LogLogistic => {
                let y = self.value(age, omega, theta);
                let yy = y * (1.0 - y);
                [yy * l, -yy * omega / theta]
            }
            Self::Weibull => {
                let u = (age / theta).powf(omega);
                let v = (-u).exp() * u;
                [v * l, -v * omega / theta]
            }
        };
        g.map(finite_or_zero)
    }

    /// `(d2G/domega2, d2G/domega dtheta, d2G/dtheta2)` at `age`; 0 at age 0
    /// and at infinity. R's Weibull `d2G/domega2` is `2 v ln(x/theta) (1 -
    /// u)`; the derivative of its own `dG/domega = v ln(x/theta)` is
    /// `v ln(x/theta)^2 (1 - u)`, used here (`u = (x/theta)^omega`,
    /// `v = u exp(-u)`).
    fn hessian(self, age: f64, omega: f64, theta: f64) -> [f64; 3] {
        if !(age > 0.0 && age.is_finite()) {
            return [0.0; 3];
        }
        let l = (age / theta).ln();
        let h = match self {
            Self::LogLogistic => {
                let y = self.value(age, omega, theta);
                let yy = y * (1.0 - y);
                let m = 1.0 - 2.0 * y;
                [
                    yy * l * l * m,
                    -yy / theta * (1.0 + omega * l * m),
                    yy * omega / (theta * theta) * (1.0 + omega * m),
                ]
            }
            Self::Weibull => {
                let u = (age / theta).powf(omega);
                let v = (-u).exp() * u;
                let w = 1.0 - u;
                [
                    v * l * l * w,
                    -v * (1.0 + omega * l * w) / theta,
                    v * omega / (theta * theta) * (1.0 + omega * w),
                ]
            }
        };
        h.map(finite_or_zero)
    }

    /// R's starting point from the curve ages of the observed cells: the
    /// log-logistic starts at `omega = 2`, `theta` their median; the Weibull
    /// at `omega = 1.5` and the `theta` that develops 95% by the oldest.
    fn start(self, ages: &[f64]) -> [f64; 2] {
        let mut sorted = ages.to_vec();
        sorted.sort_by(f64::total_cmp);
        match self {
            Self::LogLogistic => {
                let n = sorted.len();
                let median = if n % 2 == 1 {
                    sorted[n / 2]
                } else {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                };
                [2.0, median]
            }
            Self::Weibull => {
                let omega = 1.5;
                [
                    omega,
                    sorted[sorted.len() - 1] * 20f64.ln().powf(-1.0 / omega),
                ]
            }
        }
    }
}

fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

/// Clark's LDF method: each origin's expected ultimate is a parameter,
/// fitted with the growth curve to the incremental losses by maximum
/// likelihood, as R ChainLadder's `ClarkLDF(adol = TRUE)`.
///
/// The ultimate reported is R's: the latest value developed by the fitted
/// curve, `latest * G(max_age) / G(age)`, at the shifted ages (see the
/// [module](self) documentation). The process variance is `scale` times
/// the fitted reserve `U * (G(max_age) - G(age))`, where R takes `max_age`
/// unshifted (a different age from the reserve's when `max_age` is given;
/// this is kept for parity). The parameter variance is the delta method on
/// the fitted reserve with the shifted `max_age`.
///
/// ```
/// use act_reserving::{ClarkLdf, DevelopmentColumn, Grain, GrowthCurve, Long, Month, Triangle};
///
/// let rows: [&[f64]; 5] = [
///     &[110.0, 290.0, 370.0, 420.0, 440.0],
///     &[95.0, 300.0, 390.0, 425.0],
///     &[130.0, 320.0, 410.0],
///     &[105.0, 305.0],
///     &[120.0],
/// ];
/// let (mut origin, mut age, mut paid) = (vec![], vec![], vec![]);
/// for (i, row) in rows.iter().enumerate() {
///     for (d, &v) in row.iter().enumerate() {
///         origin.push(Month::january(2020 + i as i32));
///         age.push(12 * (d as u32 + 1));
///         paid.push(v);
///     }
/// }
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&age),
///     values: &[("paid", &paid)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let fit = ClarkLdf { curve: GrowthCurve::Weibull, max_age: Some(120.0) }.fit(&tri, "paid")?;
/// assert!(fit.omega > 0.0 && fit.theta > 0.0 && fit.scale > 0.0);
/// // Each ultimate is the latest value divided by the share developed.
/// let share = fit.growth(36.0) / fit.growth(120.0);
/// assert!((fit.ultimate[2] - 410.0 / share).abs() < 1e-9);
/// assert!(fit.total_standard_error > fit.total_process_risk);
/// # Ok::<(), act_reserving::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ClarkLdf {
    /// The growth curve.
    pub curve: GrowthCurve,
    /// Age in months at which development stops; `None` develops to
    /// infinity. At least the triangle's last age.
    pub max_age: Option<f64>,
}

/// Clark's Cape Cod method: each origin's expected ultimate is one expected
/// loss ratio times its exposure, fitted with the growth curve to the
/// incremental losses by maximum likelihood, as R ChainLadder's
/// `ClarkCapeCod(adol = TRUE)`.
///
/// The reserve is the fitted `ELR * exposure * (G(max_age) - G(age))`, its
/// process variance `scale` times the reserve, and its parameter variance
/// the delta method on it.
///
/// ```
/// use act_reserving::{ClarkCapeCod, DevelopmentColumn, Grain, Long, Month, Triangle};
///
/// let rows: [&[f64]; 5] = [
///     &[110.0, 290.0, 370.0, 420.0, 440.0],
///     &[95.0, 300.0, 390.0, 425.0],
///     &[130.0, 320.0, 410.0],
///     &[105.0, 305.0],
///     &[120.0],
/// ];
/// let (mut origin, mut age, mut paid, mut premium) = (vec![], vec![], vec![], vec![]);
/// for (i, row) in rows.iter().enumerate() {
///     for (d, &v) in row.iter().enumerate() {
///         origin.push(Month::january(2020 + i as i32));
///         age.push(12 * (d as u32 + 1));
///         paid.push(v);
///         premium.push(800.0);
///     }
/// }
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&age),
///     values: &[("paid", &paid), ("premium", &premium)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let fit = ClarkCapeCod::default().fit(&tri, "paid", "premium")?;
/// let elr = fit.elr.unwrap();
/// assert!(elr > 0.0 && elr < 1.0);
/// assert_eq!(fit.expected_ultimate, vec![elr * 800.0; 5]);
/// # Ok::<(), act_reserving::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ClarkCapeCod {
    /// The growth curve.
    pub curve: GrowthCurve,
    /// Age in months at which development stops; `None` develops to
    /// infinity. At least the triangle's last age.
    pub max_age: Option<f64>,
}

/// A fitted Clark LDF or Cape Cod model, per origin.
///
/// Standard errors follow R: `process_risk` is the square root of `scale`
/// times the fitted reserve, `parameter_risk` the delta-method standard
/// error from the parameter covariance (a negative variance is set to 0),
/// and `standard_error` the root of their squares' sum; totals add the
/// variances, with the parameter covariance across origins. If the Fisher
/// information is numerically singular (reciprocal condition number below
/// machine epsilon, as R tests) the covariance and the parameter and total
/// standard errors are NaN.
///
/// ```
/// use act_reserving::{ClarkLdf, DevelopmentColumn, Grain, Long, Month, Triangle};
///
/// let rows: [&[f64]; 5] = [
///     &[110.0, 290.0, 370.0, 420.0, 440.0],
///     &[95.0, 300.0, 390.0, 425.0],
///     &[130.0, 320.0, 410.0],
///     &[105.0, 305.0],
///     &[120.0],
/// ];
/// let (mut origin, mut age, mut paid) = (vec![], vec![], vec![]);
/// for (i, row) in rows.iter().enumerate() {
///     for (d, &v) in row.iter().enumerate() {
///         origin.push(Month::january(2020 + i as i32));
///         age.push(12 * (d as u32 + 1));
///         paid.push(v);
///     }
/// }
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&age),
///     values: &[("paid", &paid)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let fit = ClarkLdf::default().fit(&tri, "paid")?;
/// // Five expected ultimates, omega and theta.
/// assert_eq!(fit.covariance.len(), 7);
/// assert_eq!(fit.curve_age(12.0), 6.0);
/// let total: f64 = fit.reserves().iter().sum();
/// assert!((fit.total_reserve() - total).abs() < 1e-9);
/// # Ok::<(), act_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ClarkFit {
    /// The volume-weighted chain ladder: origins, latest values and the
    /// development pattern. Its `ultimate` is the chain ladder's.
    pub chain_ladder: ChainLadderFit,
    /// The growth curve.
    pub curve: GrowthCurve,
    /// Fitted shape of the curve.
    pub omega: f64,
    /// Fitted scale of the curve, in months.
    pub theta: f64,
    /// Expected loss ratio (Cape Cod only).
    pub elr: Option<f64>,
    /// Exposure per origin (Cape Cod only): the exposure column's latest
    /// observed value.
    pub exposure: Option<Vec<f64>>,
    /// Over-dispersion `sigma^2`: the sum of squared Pearson residuals over
    /// the observed incremental values less the number of parameters.
    pub scale: f64,
    /// Age in months at which development stops; `None` for infinity.
    pub max_age: Option<f64>,
    /// Length of the origin period in months.
    pub origin_width: f64,
    /// Expected ultimate per origin, `U_i`: fitted (LDF) or `ELR *
    /// exposure` (Cape Cod), developed to infinity.
    pub expected_ultimate: Vec<f64>,
    /// Ultimate per origin: the latest value plus the reserve.
    pub ultimate: Vec<f64>,
    /// Process standard error per origin.
    pub process_risk: Vec<f64>,
    /// Parameter standard error per origin.
    pub parameter_risk: Vec<f64>,
    /// Total standard error per origin.
    pub standard_error: Vec<f64>,
    /// Process standard error of the total reserve.
    pub total_process_risk: f64,
    /// Parameter standard error of the total reserve.
    pub total_parameter_risk: f64,
    /// Standard error of the total reserve.
    pub total_standard_error: f64,
    /// Covariance of the parameters: the expected ultimates (LDF) or the
    /// expected loss ratio (Cape Cod), then `omega` and `theta`.
    pub covariance: Vec<Vec<f64>>,
    /// Number of observed incremental values fitted.
    pub n_observations: usize,
}

impl ClarkFit {
    /// Reserve (ultimate minus latest) per origin.
    pub fn reserves(&self) -> Vec<f64> {
        self.ultimate
            .iter()
            .zip(&self.chain_ladder.latest)
            .map(|(u, l)| u - l)
            .collect()
    }

    /// Total ultimate across origins.
    pub fn total_ultimate(&self) -> f64 {
        self.ultimate.iter().sum()
    }

    /// Total reserve across origins.
    pub fn total_reserve(&self) -> f64 {
        self.total_ultimate() - self.chain_ladder.latest.iter().sum::<f64>()
    }

    /// The curve age of a development age (both in months): measured from
    /// the average date of loss, the middle of the origin period.
    pub fn curve_age(&self, age: f64) -> f64 {
        shift(age, self.origin_width)
    }

    /// Share of the expected ultimate developed by development age `age`
    /// (months), `G(curve_age(age))`; 1 at infinity.
    pub fn growth(&self, age: f64) -> f64 {
        self.curve
            .value(self.curve_age(age), self.omega, self.theta)
    }
}

/// R's `adol = TRUE` age: measured from the middle of an origin period of
/// `width` months.
fn shift(age: f64, width: f64) -> f64 {
    let offset = width / 2.0;
    if age < width {
        age * (1.0 - offset / width)
    } else {
        age - offset
    }
}

impl ReserveFit for ClarkFit {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn ultimate(&self) -> &[f64] {
        &self.ultimate
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        let mut columns = Vec::new();
        if let Some(exposure) = &self.exposure {
            columns.push(("exposure", exposure.clone()));
        }
        columns.extend([
            ("expected_ultimate", self.expected_ultimate.clone()),
            ("process_risk", self.process_risk.clone()),
            ("parameter_risk", self.parameter_risk.clone()),
            ("standard_error", self.standard_error.clone()),
        ]);
        columns
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        let mut columns = vec![
            ("process_risk", self.total_process_risk),
            ("parameter_risk", self.total_parameter_risk),
            ("standard_error", self.total_standard_error),
            ("omega", self.omega),
            ("theta", self.theta),
            ("scale", self.scale),
        ];
        if let Some(elr) = self.elr {
            columns.push(("elr", elr));
        }
        columns
    }
}

impl ClarkLdf {
    /// Fits loss `column` of a single-segment triangle.
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<ClarkFit> {
        fit_segment(&triangle.segment(column)?, None, self.curve, self.max_age)
    }

    /// Fits `column` in every segment of `triangle`, each on its own; see
    /// [`SegmentFits`] for the long tables. A failure names its segment.
    pub fn fit_segments(&self, triangle: &Triangle, column: &str) -> Result<SegmentFits<ClarkFit>> {
        fit_each(triangle, column, |s| {
            fit_segment(s, None, self.curve, self.max_age)
        })
    }
}

impl ClarkCapeCod {
    /// Fits loss `column` of a single-segment triangle with exposure from
    /// its `exposure` column.
    pub fn fit(&self, triangle: &Triangle, column: &str, exposure: &str) -> Result<ClarkFit> {
        let segment = triangle.segment(column)?;
        let premium = origin_exposure(&segment, &triangle.segment(exposure)?, exposure)?;
        fit_segment(&segment, Some(premium), self.curve, self.max_age)
    }

    /// Fits `column` in every segment of `triangle`, each with its own
    /// exposure; see [`SegmentFits`] for the long tables. A failure names
    /// its segment.
    pub fn fit_segments(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
    ) -> Result<SegmentFits<ClarkFit>> {
        fit_each_with_exposure(triangle, column, exposure, |s, e| {
            let premium = origin_exposure(s, e, exposure)?;
            fit_segment(s, Some(premium), self.curve, self.max_age)
        })
    }
}

/// One observed incremental value: `value` between curve ages `from` and
/// `to`, with mean `a[group] * weight * (G(to) - G(from))`.
struct Cell {
    group: usize,
    weight: f64,
    from: f64,
    to: f64,
    value: f64,
}

/// The over-dispersed Poisson likelihood of the cells, with one parameter
/// `a` per group (an origin's expected ultimate, or the one expected loss
/// ratio) and the curve's `(omega, theta)`.
struct Model {
    curve: GrowthCurve,
    cells: Vec<Cell>,
    /// Sum of the values of each group.
    group_total: Vec<f64>,
}

impl Model {
    fn delta(&self, cell: &Cell, omega: f64, theta: f64) -> f64 {
        self.curve.value(cell.to, omega, theta) - self.curve.value(cell.from, omega, theta)
    }

    /// The maximum-likelihood `a` of each group given the curve: the
    /// group's total over the sum of its weighted curve shares. `None` if
    /// any is not finite and positive.
    fn profile(&self, omega: f64, theta: f64) -> Option<Vec<f64>> {
        let mut share = vec![0.0; self.group_total.len()];
        for c in &self.cells {
            share[c.group] += c.weight * self.delta(c, omega, theta);
        }
        let a: Vec<f64> = self
            .group_total
            .iter()
            .zip(&share)
            .map(|(t, s)| t / s)
            .collect();
        a.iter().all(|a| a.is_finite() && *a > 0.0).then_some(a)
    }

    /// Mean of each cell, floored at machine epsilon as R floors it.
    fn means(&self, a: &[f64], omega: f64, theta: f64) -> Vec<f64> {
        self.cells
            .iter()
            .map(|c| (a[c.group] * c.weight * self.delta(c, omega, theta)).max(f64::EPSILON))
            .collect()
    }

    /// Negative log-likelihood (up to a constant) at `x = (ln omega, ln
    /// theta)` with `a` profiled out; NaN where `a` does not exist.
    fn profiled_loss(&self, x: &[f64]) -> f64 {
        let (omega, theta) = (x[0].exp(), x[1].exp());
        let Some(a) = self.profile(omega, theta) else {
            return f64::NAN;
        };
        let mu = self.means(&a, omega, theta);
        -self
            .cells
            .iter()
            .zip(&mu)
            .map(|(c, m)| c.value * m.ln() - m)
            .sum::<f64>()
    }

    /// Hessian of the log-likelihood in `(a..., omega, theta)`, row-major:
    /// `sum (c / mu - 1) d2mu - (c / mu^2) dmu dmu'` over the cells, as R's
    /// `d2LL.ODPdt2`.
    fn hessian(&self, a: &[f64], omega: f64, theta: f64, mu: &[f64]) -> Vec<f64> {
        let k = a.len();
        let p = k + 2;
        let mut h = vec![0.0; p * p];
        for (c, &m) in self.cells.iter().zip(mu) {
            let delta = self.delta(c, omega, theta);
            let [go_to, gt_to] = self.curve.gradient(c.to, omega, theta);
            let [go_from, gt_from] = self.curve.gradient(c.from, omega, theta);
            let dg = [go_to - go_from, gt_to - gt_from];
            let h_to = self.curve.hessian(c.to, omega, theta);
            let h_from = self.curve.hessian(c.from, omega, theta);
            let d2g = [
                h_to[0] - h_from[0],
                h_to[1] - h_from[1],
                h_to[2] - h_from[2],
            ];
            let ag = a[c.group] * c.weight;
            // First derivatives of mu at indices (group, k, k + 1).
            let idx = [c.group, k, k + 1];
            let d = [c.weight * delta, ag * dg[0], ag * dg[1]];
            // Second derivatives: d2mu/da2 = 0, d2mu/da dg = w dG,
            // d2mu/dg dg' = a w d2G.
            let second = |i: usize, j: usize| -> f64 {
                match (i, j) {
                    (0, 0) => 0.0,
                    (0, j) | (j, 0) => c.weight * dg[j - 1],
                    (1, 1) => ag * d2g[0],
                    (2, 2) => ag * d2g[2],
                    _ => ag * d2g[1],
                }
            };
            let r = c.value / m - 1.0;
            let q = c.value / (m * m);
            for i in 0..3 {
                for j in 0..3 {
                    h[idx[i] * p + idx[j]] += r * second(i, j) - q * d[i] * d[j];
                }
            }
        }
        h
    }
}

/// The inverse of the `n x n` matrix `a`, or `None` if it is singular or
/// its reciprocal 1-norm condition number is below machine epsilon (R's
/// `rcond(FI) < .Machine$double.eps`).
fn inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut inv = vec![0.0; n * n];
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let x = solve(a.to_vec(), e, n)?;
        for i in 0..n {
            inv[i * n + j] = x[i];
        }
    }
    let norm1 = |m: &[f64]| {
        (0..n)
            .map(|j| (0..n).map(|i| m[i * n + j].abs()).sum::<f64>())
            .fold(0.0, f64::max)
    };
    let rcond = 1.0 / (norm1(a) * norm1(&inv));
    (rcond >= f64::EPSILON).then_some(inv)
}

/// Fits Clark's LDF method (`exposure` is `None`) or Cape Cod (`exposure`
/// per origin) to one segment.
fn fit_segment(
    segment: &Segment,
    exposure: Option<Vec<f64>>,
    curve: GrowthCurve,
    max_age: Option<f64>,
) -> Result<ClarkFit> {
    if segment.n_dev < 4 {
        return Err(Error::TooFewAges {
            needed: 4,
            found: segment.n_dev,
        });
    }
    let last_age = f64::from(*segment.ages.last().expect("a segment has ages"));
    if let Some(m) = max_age
        && (m.is_nan() || m < last_age)
    {
        return Err(Error::InvalidSetting {
            name: "max_age",
            value: m,
            expected: "an age in months at or past the triangle's last age",
        });
    }
    let chain_ladder = ChainLadder::default().fit_segment(segment, &segment.ages)?;
    let n = segment.n_origins;
    let width = f64::from(segment.origins[0].grain().months());
    let cape_cod = exposure.is_some();

    // Work in units of the largest chain-ladder ultimate, as R's ClarkLDF
    // does, so the likelihood is of order 1; the Cape Cod exposure is
    // scaled with the losses, leaving the loss ratio as it is.
    let unit = chain_ladder
        .ultimate
        .iter()
        .copied()
        .filter(|u| u.is_finite())
        .fold(0.0, f64::max);
    let unit = if unit > 0.0 { unit } else { 1.0 };
    let weight: Vec<f64> = match &exposure {
        Some(e) => e.iter().map(|e| e / unit).collect(),
        None => vec![1.0; n],
    };

    let mut cells = Vec::new();
    for (o, &w) in weight.iter().enumerate() {
        let (mut prev_age, mut prev) = (0.0, 0.0);
        for d in 0..segment.n_dev {
            if let Some(v) = segment.get(o, d) {
                let age = f64::from(segment.ages[d]);
                cells.push(Cell {
                    group: if cape_cod { 0 } else { o },
                    weight: w,
                    from: shift(prev_age, width),
                    to: shift(age, width),
                    value: (v - prev) / unit,
                });
                (prev_age, prev) = (age, v);
            }
        }
    }
    let n_groups = if cape_cod { 1 } else { n };
    let mut group_total = vec![0.0; n_groups];
    for c in &cells {
        group_total[c.group] += c.value;
    }
    if let Some(g) = group_total.iter().position(|t| t.is_nan() || *t <= 0.0) {
        return Err(Error::Clark(if cape_cod {
            "the latest values must sum to a positive amount".into()
        } else {
            format!("origin {} has no positive latest value", segment.origins[g])
        }));
    }
    let n_par = n_groups + 2;
    let n_obs = cells.len();
    if n_obs <= n_par {
        return Err(Error::Clark(format!(
            "{n_obs} observed values cannot fit {n_par} parameters"
        )));
    }
    let model = Model {
        curve,
        cells,
        group_total,
    };

    let ages: Vec<f64> = model.cells.iter().map(|c| c.to).collect();
    let x0 = curve.start(&ages).map(f64::ln);
    let options = NelderMead {
        tolerance: 1e-10,
        ..NelderMead::default()
    };
    let min = nelder_mead(|x| model.profiled_loss(x), &x0, &options);
    if !min.converged || !min.value.is_finite() {
        return Err(Error::Clark(
            "the maximum-likelihood search did not converge".into(),
        ));
    }
    let (omega, theta) = (min.x[0].exp(), min.x[1].exp());
    let a = model
        .profile(omega, theta)
        .expect("the minimum is a feasible point");
    let mu = model.means(&a, omega, theta);
    let pearson: f64 = model
        .cells
        .iter()
        .zip(&mu)
        .map(|(c, m)| (c.value - m).powi(2) / m)
        .sum();
    let scale = pearson / (n_obs - n_par) as f64;

    // Parameter covariance: scale times the inverse observed information.
    let information: Vec<f64> = model
        .hessian(&a, omega, theta, &mu)
        .iter()
        .map(|h| -h)
        .collect();
    let covariance =
        inverse(&information, n_par).map(|inv| inv.iter().map(|v| v * scale).collect::<Vec<f64>>());

    // Reserves at each origin's latest curve age.
    let g = |x: f64| curve.value(x, omega, theta);
    let dg = |x: f64| curve.gradient(x, omega, theta);
    let max_used = max_age.map_or(f64::INFINITY, |m| shift(m, width));
    let max_raw = max_age.unwrap_or(f64::INFINITY);
    let (g_max, dg_max) = (g(max_used), dg(max_used));
    let mut expected = vec![0.0; n];
    let mut reserve = vec![0.0; n];
    let mut process_var = vec![0.0; n];
    let mut gradient = vec![vec![0.0; n_par]; n];
    for i in 0..n {
        let age = shift(
            f64::from(segment.ages[chain_ladder.latest_position[i]]),
            width,
        );
        let (g_age, dg_age) = (g(age), dg(age));
        let (u, a_index) = if cape_cod {
            (a[0] * weight[i], 0)
        } else {
            (a[i], i)
        };
        expected[i] = u;
        let fitted = u * (g_max - g_age);
        if cape_cod {
            reserve[i] = fitted * unit;
            process_var[i] = scale * fitted;
            gradient[i][a_index] = weight[i] * (g_max - g_age);
        } else {
            let latest = chain_ladder.latest[i];
            reserve[i] = latest * (g_max / g_age - 1.0);
            process_var[i] = scale * u * (g(max_raw) - g_age);
            gradient[i][a_index] = g_max - g_age;
        }
        gradient[i][n_groups] = u * (dg_max[0] - dg_age[0]);
        gradient[i][n_groups + 1] = u * (dg_max[1] - dg_age[1]);
    }

    // Delta method: var(R_i, R_j) = grad_i' cov grad_j.
    let (parameter_var, total_parameter_var) = match &covariance {
        Some(cov) => {
            let cov_grad: Vec<Vec<f64>> = gradient
                .iter()
                .map(|gj| {
                    (0..n_par)
                        .map(|r| (0..n_par).map(|s| cov[r * n_par + s] * gj[s]).sum())
                        .collect()
                })
                .collect();
            let cross = |i: usize, j: usize| -> f64 {
                gradient[i]
                    .iter()
                    .zip(&cov_grad[j])
                    .map(|(x, y)| x * y)
                    .sum()
            };
            let per_origin: Vec<f64> = (0..n).map(|i| cross(i, i).max(0.0)).collect();
            let total: f64 = (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| cross(i, j))
                .sum();
            (per_origin, total.max(0.0))
        }
        None => (vec![f64::NAN; n], f64::NAN),
    };

    let process_risk: Vec<f64> = process_var.iter().map(|v| v.sqrt() * unit).collect();
    let parameter_risk: Vec<f64> = parameter_var.iter().map(|v| v.sqrt() * unit).collect();
    let standard_error = process_var
        .iter()
        .zip(&parameter_var)
        .map(|(p, q)| (p + q).sqrt() * unit)
        .collect();
    let total_process_var: f64 = process_var.iter().sum();
    let ultimate = chain_ladder
        .latest
        .iter()
        .zip(&reserve)
        .map(|(l, r)| l + r)
        .collect();
    // Back to the losses' unit: expected ultimates scale with it, the loss
    // ratio and the curve do not.
    let factor = |r: usize| if r < n_groups && !cape_cod { unit } else { 1.0 };
    let covariance = (0..n_par)
        .map(|r| {
            (0..n_par)
                .map(|s| match &covariance {
                    Some(cov) => cov[r * n_par + s] * factor(r) * factor(s),
                    None => f64::NAN,
                })
                .collect()
        })
        .collect();
    let expected_ultimate = match &exposure {
        Some(e) => e.iter().map(|e| a[0] * e).collect(),
        None => expected.iter().map(|u| u * unit).collect(),
    };
    Ok(ClarkFit {
        chain_ladder,
        curve,
        omega,
        theta,
        elr: cape_cod.then(|| a[0]),
        exposure,
        scale: scale * unit,
        max_age,
        origin_width: width,
        expected_ultimate,
        ultimate,
        process_risk,
        parameter_risk,
        standard_error,
        total_process_risk: total_process_var.sqrt() * unit,
        total_parameter_risk: total_parameter_var.sqrt() * unit,
        total_standard_error: (total_process_var + total_parameter_var).sqrt() * unit,
        covariance,
        n_observations: n_obs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::{GENINS, RAA, annual};
    use crate::triangle::{DevelopmentColumn, Long};
    use crate::{Grain, Label, Month};

    /// GenIns with R's `?ClarkCapeCod` premium, `10e6 + 0.4e6 * (0:9)`, as
    /// a `premium` column next to `paid`.
    fn genins_premium() -> Triangle {
        with_premium(2001, &GENINS, |k| 10_000_000.0 + 400_000.0 * k as f64)
    }

    /// Annual cumulative `rows` from origin `start` as `paid`, with
    /// `premium(k)` for origin `k` as a `premium` column.
    fn with_premium(start: i32, rows: &[&[f64]], premium_of: impl Fn(usize) -> f64) -> Triangle {
        let (mut origin, mut ages, mut paid, mut premium) = (vec![], vec![], vec![], vec![]);
        for (k, row) in rows.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                origin.push(Month::january(start + k as i32));
                ages.push(12 * (d as u32 + 1));
                paid.push(v);
                premium.push(premium_of(k));
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("premium", &premium)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn close(got: f64, want: f64, rel: f64) {
        assert!(
            (got - want).abs() <= rel * want.abs(),
            "got {got}, want {want}"
        );
    }

    #[test]
    fn raa_loglogistic_matches_r() {
        // R ChainLadder 0.2.21: ClarkLDF(RAA, G = "loglogistic") with
        // L-BFGS-B run to convergence (factr = 1), ages in months; see
        // validation/reference/reserving_clark_r.csv.
        let fit = ClarkLdf::default()
            .fit(&annual(1981, &RAA), "values")
            .unwrap();
        close(fit.omega, 1.34649060338587, 1e-6);
        close(fit.theta, 36.5506893209726, 1e-6);
        close(fit.scale, 934.249299218311, 1e-6);
        close(fit.covariance[10][10].sqrt(), 0.158396400604264, 1e-5);
        close(fit.reserves()[9], 23504.4494416442, 1e-6);
        close(fit.process_risk[9], 4686.04582746626, 1e-6);
        close(fit.parameter_risk[9], 16545.031945866, 1e-5);
        close(fit.total_ultimate(), 271993.625765179, 1e-7);
        close(fit.total_process_risk, 10183.7063372018, 1e-6);
        close(fit.total_parameter_risk, 34608.0070082816, 1e-5);
        close(fit.total_standard_error, 36075.2272875394, 1e-5);
        assert_eq!(fit.n_observations, 55);
        assert_eq!(fit.elr, None);
    }

    #[test]
    fn raa_weibull_truncated_matches_r() {
        // R ChainLadder 0.2.21: ClarkLDF(RAA, G = "weibull", maxage = 240)
        // with factr = 1 and the corrected Weibull d2G/domega2.
        let fit = ClarkLdf {
            curve: GrowthCurve::Weibull,
            max_age: Some(240.0),
        }
        .fit(&annual(1981, &RAA), "values")
        .unwrap();
        close(fit.omega, 1.21301210273774, 1e-6);
        close(fit.theta, 37.7390195700206, 1e-6);
        close(fit.scale, 869.954522987584, 1e-6);
        close(fit.covariance[10][10].sqrt(), 0.105200944254447, 1e-5);
        close(fit.reserves()[9], 18182.9190042388, 1e-6);
        close(fit.process_risk[9], 3977.28308829896, 1e-6);
        close(fit.parameter_risk[9], 12324.5072857254, 1e-5);
        close(fit.total_reserve(), 59929.8463698341, 1e-6);
        close(fit.total_standard_error, 19147.1022958312, 1e-5);
    }

    #[test]
    fn genins_cape_cod_matches_r() {
        // R ChainLadder 0.2.21: ClarkCapeCod(GenIns, Premium = 10e6 +
        // 0.4e6 * (0:9), maxage = 240) with factr = 1.
        let fit = ClarkCapeCod {
            curve: GrowthCurve::LogLogistic,
            max_age: Some(240.0),
        }
        .fit(&genins_premium(), "paid", "premium")
        .unwrap();
        close(fit.elr.unwrap(), 0.597010699352741, 1e-6);
        close(fit.omega, 1.44877619016508, 1e-6);
        close(fit.theta, 47.9174093982723, 1e-6);
        close(fit.scale, 61146.508087408, 1e-6);
        close(fit.covariance[1][1].sqrt(), 0.0883435633424129, 1e-5);
        close(fit.total_reserve(), 29655386.4392939, 1e-6);
        close(fit.total_process_risk, 1346596.94294377, 1e-6);
        close(fit.total_parameter_risk, 3124937.13579887, 1e-5);
        close(fit.total_standard_error, 3402727.64549859, 1e-5);
        assert_eq!(fit.covariance.len(), 3);
        assert_eq!(fit.exposure.as_ref().unwrap()[9], 13_600_000.0);
    }

    #[test]
    fn cape_cod_elr_is_unbounded_unlike_r() {
        // R ChainLadder 0.2.21's ClarkCapeCod bounds the ELR at 10 without
        // a warning: on RAA with Premium = 1000 it stops at ELR = 10, omega
        // = 1.76167, theta = 22.510, reserve 27,418.99 (factr = 1). The
        // unbounded maximum is R's own fit with Premium = 4000 (ELR
        // 6.90230116779241, below the cap; factr = 1) scaled by 4: the
        // same curve and reserve. See
        // knowledge/references/r-chainladder-clark.md.
        let fit = ClarkCapeCod::default()
            .fit(&with_premium(1981, &RAA, |_| 1000.0), "paid", "premium")
            .unwrap();
        close(fit.elr.unwrap(), 4.0 * 6.90230116779241, 1e-6);
        close(fit.omega, 1.37550879649416, 1e-6);
        close(fit.theta, 36.12485655106858, 1e-6);
        close(fit.total_reserve(), 115105.046815558, 1e-6);
    }

    #[test]
    fn ldf_expected_ultimate_is_latest_over_growth() {
        // On a full triangle each origin's incremental values start at age
        // 0, so the profiled U_i is latest / G(latest age); and without a
        // max_age the reserve is the fitted one, U_i (1 - G(age)).
        let fit = ClarkLdf::default()
            .fit(&annual(1981, &RAA), "values")
            .unwrap();
        for i in 0..10 {
            let age = f64::from(12 * (10 - i as u32));
            let g = fit.growth(age);
            close(
                fit.expected_ultimate[i],
                fit.chain_ladder.latest[i] / g,
                1e-12,
            );
            close(
                fit.reserves()[i],
                fit.expected_ultimate[i] * (1.0 - g),
                1e-9,
            );
        }
        assert_eq!(fit.growth(f64::INFINITY), 1.0);
    }

    #[test]
    fn curve_derivatives_match_differences() {
        for curve in [GrowthCurve::LogLogistic, GrowthCurve::Weibull] {
            let (omega, theta) = (1.3, 48.0);
            for x in [6.0, 18.0, 54.0, 114.0] {
                let h = 1e-6;
                let g = curve.gradient(x, omega, theta);
                let num = [
                    (curve.value(x, omega + h, theta) - curve.value(x, omega - h, theta))
                        / (2.0 * h),
                    (curve.value(x, omega, theta + h) - curve.value(x, omega, theta - h))
                        / (2.0 * h),
                ];
                for (a, b) in g.iter().zip(num) {
                    assert!(
                        (a - b).abs() < 1e-8,
                        "{curve:?} gradient at {x}: {a} vs {b}"
                    );
                }
                let d2 = curve.hessian(x, omega, theta);
                let dw = |o: f64, t: f64| curve.gradient(x, o, t);
                let num = [
                    (dw(omega + h, theta)[0] - dw(omega - h, theta)[0]) / (2.0 * h),
                    (dw(omega, theta + h)[0] - dw(omega, theta - h)[0]) / (2.0 * h),
                    (dw(omega, theta + h)[1] - dw(omega, theta - h)[1]) / (2.0 * h),
                ];
                for (a, b) in d2.iter().zip(num) {
                    assert!((a - b).abs() < 1e-8, "{curve:?} hessian at {x}: {a} vs {b}");
                }
            }
            assert_eq!(curve.gradient(0.0, 1.3, 48.0), [0.0; 2]);
            assert_eq!(curve.hessian(f64::INFINITY, 1.3, 48.0), [0.0; 3]);
        }
    }

    #[test]
    fn rejects_bad_settings_and_data() {
        let raa = annual(1981, &RAA);
        let short = ClarkLdf {
            max_age: Some(119.0),
            ..Default::default()
        };
        assert!(matches!(
            short.fit(&raa, "values"),
            Err(Error::InvalidSetting {
                name: "max_age",
                ..
            })
        ));
        let three = annual(2020, &[&[1.0, 2.0, 3.0], &[1.0, 2.0], &[1.0]]);
        assert_eq!(
            ClarkLdf::default().fit(&three, "values"),
            Err(Error::TooFewAges {
                needed: 4,
                found: 3
            })
        );
        let mut rows: Vec<&[f64]> = RAA.to_vec();
        rows[9] = &[0.0];
        assert_eq!(
            ClarkLdf::default().fit(&annual(1981, &rows), "values"),
            Err(Error::Clark(
                "origin 1990 has no positive latest value".into()
            ))
        );
    }

    #[test]
    fn segments_fit_on_their_own() {
        // Two copies of RAA as segments, one scaled by 10: the curve is the
        // same and the amounts scale.
        let (mut keys, mut origin, mut ages, mut values) = (vec![], vec![], vec![], vec![]);
        for (lob, factor) in [("A", 1.0), ("B", 10.0)] {
            for (k, row) in RAA.iter().enumerate() {
                for (d, &v) in row.iter().enumerate() {
                    keys.push(lob);
                    origin.push(Month::january(1981 + k as i32));
                    ages.push(12 * (d as u32 + 1));
                    values.push(v * factor);
                }
            }
        }
        let tri = Triangle::from_long(&Long {
            keys: &[("lob", &keys)],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &values)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let fits = ClarkLdf::default().fit_segments(&tri, "paid").unwrap();
        let a = fits.get(&Label::new(["A"])).unwrap();
        let b = fits.get(&Label::new(["B"])).unwrap();
        close(b.omega, a.omega, 1e-7);
        close(b.theta, a.theta, 1e-7);
        close(b.total_reserve(), 10.0 * a.total_reserve(), 1e-7);
        close(b.total_standard_error, 10.0 * a.total_standard_error, 1e-6);
        let totals = fits.totals();
        assert_eq!(totals.column("omega").unwrap().len(), 2);
        let long = fits.to_long();
        assert_eq!(long.column("standard_error").unwrap().len(), 20);
    }
}
