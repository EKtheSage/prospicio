//! Tail factors: development beyond the oldest age of the triangle.
//!
//! A [`Tail`] is fitted against an estimated development pattern and gives a
//! [`TailFit`]: the selected age-to-age factors (the estimated ones, with the
//! tail's own from its attachment age on), the factor from the oldest age to
//! ultimate, and the tail's variance parameter and standard error for Mack.
//! The estimators follow chainladder-python (`TailConstant`, `TailCurve`,
//! `TailBondy`) and R ChainLadder (`tail = TRUE` in `MackChainLadder`).

use crate::development::DevelopmentFit;
use crate::error::{Error, Result};
use prospicio_core::Lag;

/// How development beyond the oldest age is estimated.
///
/// The default is a constant factor of 1 (no tail), and a number converts
/// to a constant tail:
///
/// ```
/// use prospicio_reserving::{ChainLadder, Tail, TailConstant, TailCurve};
///
/// let none = ChainLadder::default();
/// assert_eq!(none.tail, Tail::Constant(TailConstant::default()));
/// let five_percent = ChainLadder { tail: 1.05.into(), ..Default::default() };
/// let curve = ChainLadder { tail: TailCurve::default().into(), ..Default::default() };
/// # let _ = (five_percent, curve);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tail {
    /// A given factor, spread over the periods after its attachment age with
    /// chainladder-python's decay.
    Constant(TailConstant),
    /// An exponential or inverse-power curve fitted to `f - 1` and
    /// extrapolated.
    Curve(TailCurve),
    /// The (generalized) Bondy method.
    Bondy(TailBondy),
    /// R ChainLadder's `tail = TRUE` rule (its `tailfactor` function): when
    /// the factors before the last are still above 1, regress `ln(f - 1)` on
    /// the development index over the factors above 1 and multiply the next
    /// 100 extrapolated factors. A tail above 2 is reset to 1, as R does.
    LogLinear,
}

impl Default for Tail {
    fn default() -> Self {
        Self::Constant(TailConstant::default())
    }
}

impl From<f64> for Tail {
    fn from(factor: f64) -> Self {
        Self::Constant(TailConstant {
            factor,
            ..Default::default()
        })
    }
}

impl From<TailConstant> for Tail {
    fn from(t: TailConstant) -> Self {
        Self::Constant(t)
    }
}

impl From<TailCurve> for Tail {
    fn from(t: TailCurve) -> Self {
        Self::Curve(t)
    }
}

impl From<TailBondy> for Tail {
    fn from(t: TailBondy) -> Self {
        Self::Bondy(t)
    }
}

/// A given tail factor, as chainladder-python's `TailConstant`.
///
/// The factor applies from `attachment_age` (the oldest age when `None`) to
/// ultimate. It is spread over the following periods as
/// `1 + x * decay^k`, with `x` chosen so that the factors multiply to about
/// the given one and the last factor making up the difference; this only
/// shapes the factors past the attachment, not the factor to ultimate. An
/// earlier attachment replaces the estimated factors from that age on.
///
/// ```
/// use prospicio_reserving::{ChainLadder, Tail, TailConstant};
/// # use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// # let origin = [2020, 2020, 2021].map(Month::january);
/// # let tri = Triangle::from_long(&Long {
/// #     keys: &[],
/// #     origin: &origin,
/// #     development: DevelopmentColumn::Age(&[12, 24, 12]),
/// #     values: &[("paid", &[100.0, 150.0, 200.0])],
/// #     origin_grain: Grain::Year,
/// #     development_grain: Grain::Year,
/// #     cumulative: true,
/// # })
/// # .unwrap();
/// let tail = Tail::Constant(TailConstant { factor: 1.05, ..Default::default() });
/// let cl = ChainLadder { tail, ..Default::default() }.fit(&tri, "paid").unwrap();
/// assert_eq!(cl.tail.factor, 1.05);
/// assert!((cl.ultimate[1] - 200.0 * 1.5 * 1.05).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailConstant {
    /// Factor from the attachment age to ultimate; finite and positive.
    pub factor: f64,
    /// Share of each period's development kept in the next, from 0 to 1.
    pub decay: f64,
    /// Age the factor attaches at: the first age at or after it. `None` is
    /// the oldest age. An age at or before the youngest replaces every
    /// estimated factor (chainladder-python ignores such an attachment and
    /// attaches at the oldest age).
    pub attachment_age: Option<Lag>,
}

impl Default for TailConstant {
    fn default() -> Self {
        Self {
            factor: 1.0,
            decay: 0.5,
            attachment_age: None,
        }
    }
}

/// Form of the curve a [`TailCurve`] fits to `f - 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CurveShape {
    /// `ln(f_k - 1) = a + b k`: development decays geometrically.
    #[default]
    Exponential,
    /// `ln(f_k - 1) = a + b ln(k)`: development decays as a power of the
    /// period, a heavier tail.
    InversePower,
}

/// A curve fitted to the estimated factors and extrapolated, as
/// chainladder-python's `TailCurve`.
///
/// Factors above 1.00001 in the fit period are regressed by least squares
/// (`ln(f - 1)` on the 1-based development index `k`, or on `ln(k)`); the
/// fitted curve replaces the factors from the attachment age on and runs
/// `extrap_periods` periods past the oldest age.
///
/// ```
/// use prospicio_reserving::{ChainLadder, CurveShape, Tail, TailCurve};
/// # use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// # let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
/// # let tri = Triangle::from_long(&Long {
/// #     keys: &[],
/// #     origin: &origin,
/// #     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
/// #     values: &[("paid", &[100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0])],
/// #     origin_grain: Grain::Year,
/// #     development_grain: Grain::Year,
/// #     cumulative: true,
/// # })
/// # .unwrap();
/// let tail = Tail::Curve(TailCurve { curve: CurveShape::Exponential, ..Default::default() });
/// let cl = ChainLadder { tail, ..Default::default() }.fit(&tri, "paid").unwrap();
/// assert!(cl.tail.factor > 1.0 && cl.tail.factor < 1.05);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailCurve {
    /// Curve fitted to `f - 1`.
    pub curve: CurveShape,
    /// Ages whose factors enter the fit, as chainladder-python's
    /// `fit_period`: from the last age at or before the first (inclusive)
    /// to the last age at or before the second (exclusive). `None` is
    /// open-ended, and a start before the youngest age starts there.
    pub fit_period: (Option<Lag>, Option<Lag>),
    /// Number of periods past the oldest age the curve is extrapolated.
    pub extrap_periods: usize,
    /// Age the curve attaches at: the first age at or after it. `None` is
    /// the oldest age.
    pub attachment_age: Option<Lag>,
}

impl Default for TailCurve {
    fn default() -> Self {
        Self {
            curve: CurveShape::Exponential,
            fit_period: (None, None),
            extrap_periods: 100,
            attachment_age: None,
        }
    }
}

/// The Bondy tail, as chainladder-python's `TailBondy`.
///
/// Each log factor from `earliest_age` on is taken as `b` times the one
/// before it; `b` and the first log factor are fitted by least squares. The
/// fitted factors are `f0^(b^j)`, from the observed factor `f0` at
/// `earliest_age`, and the factors past the next one multiply to the last
/// fitted factor raised to `b / (1 - b)`. With the default `earliest_age`
/// (the age of the last factor) `b` stays at 1/2 and the tail repeats the
/// last factor: the classic Bondy method.
///
/// ```
/// use prospicio_reserving::{ChainLadder, Tail, TailBondy};
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
/// let tail = Tail::Bondy(TailBondy::default());
/// let cl = ChainLadder { tail, ..Default::default() }.fit(&tri, "paid").unwrap();
/// assert!((cl.tail.factor - 1.1).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TailBondy {
    /// First age whose factor enters the fit: the last age at or before
    /// it, as chainladder-python reads it. `None` is the age of the last
    /// factor.
    pub earliest_age: Option<Lag>,
    /// The factor from this age (the last age at or before it) to the next
    /// is kept; the fitted factors replace those after it. `None` is the age
    /// of the last factor. Not before `earliest_age`.
    pub attachment_age: Option<Lag>,
}

/// A tail fitted to a development pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct TailFit {
    /// Development position of the first factor the tail replaced; the
    /// number of estimated factors when it attaches at the oldest age.
    pub attachment: usize,
    /// Selected age-to-age factors: the estimated ones before
    /// `attachment`, the tail's from there to the oldest age, then the
    /// factors past the oldest age (as chainladder-python's `ldf_`: one per
    /// development period of the following year and one to ultimate; R's
    /// rule gives a single factor to ultimate).
    pub ldf: Vec<f64>,
    /// Factor from the oldest age to ultimate: the product of the factors
    /// past the oldest age.
    pub factor: f64,
    /// Variance parameter of the tail factor for Mack, extrapolated
    /// log-linearly as R's `tail_SE` and chainladder-python do; 0 for a
    /// factor of exactly 1 (no tail), NaN when it cannot be extrapolated. A
    /// factor below 1 is read where a factor of 1.001 would be, as
    /// chainladder-python does (R ignores a tail below 1).
    pub sigma: f64,
    /// Standard error of the tail factor, extrapolated the same way.
    pub std_err: f64,
}

/// Lower bound on the factors a curve is fitted to, chainladder-python's
/// `reg_threshold`.
const CURVE_THRESHOLD: f64 = 1.00001;

impl Tail {
    /// Fits the tail to an estimated development pattern, whose ages
    /// (`development`) locate the attachment and fit ages.
    pub fn fit(&self, development: &DevelopmentFit) -> Result<TailFit> {
        let (attachment, ldf, factor) = self.select(&development.ldf, &development.development)?;
        // No tail carries no risk. A tail below 1 has no position on the
        // line through ln(f - 1), so chainladder-python's
        // `_get_tail_weighted_time_period` reads it where 1.001 would be.
        let (sigma, std_err) = if factor == 1.0 {
            (0.0, 0.0)
        } else {
            tail_statistics(development, if factor > 1.0 { factor } else { 1.001 })
        };
        Ok(TailFit {
            attachment,
            ldf,
            factor,
            sigma,
            std_err,
        })
    }

    /// The selected factors of [`TailFit`] (`attachment`, `ldf` and
    /// `factor`) for the `estimated` factors between `ages`, without the
    /// tail's sigma and standard error: what a bootstrap refits on each
    /// simulation's pseudo factors.
    pub(crate) fn select(&self, estimated: &[f64], ages: &[Lag]) -> Result<(usize, Vec<f64>, f64)> {
        let n_links = estimated.len();
        let periods = periods_per_year(ages);
        let (attachment, ldf) = match self {
            Self::Constant(t) => t.select(estimated, ages, periods)?,
            Self::Curve(t) => t.select(estimated, ages, periods)?,
            Self::Bondy(t) => t.select(estimated, ages)?,
            Self::LogLinear => {
                let mut ldf = estimated.to_vec();
                ldf.push(log_linear_factor(estimated)?);
                (n_links, ldf)
            }
        };
        let factor: f64 = ldf[n_links..].iter().product();
        if !factor.is_finite() || factor <= 0.0 {
            return Err(Error::Tail(
                "the fitted tail factor is not finite and positive",
            ));
        }
        Ok((attachment, ldf, factor))
    }

    /// Whether this is no tail: a constant factor of 1 at the oldest age,
    /// which leaves the estimated factors as they are.
    pub(crate) fn is_none(&self) -> bool {
        matches!(self, Self::Constant(t) if t.factor == 1.0 && t.attachment_age.is_none())
    }
}

impl TailConstant {
    fn select(&self, estimated: &[f64], ages: &[Lag], periods: usize) -> Result<(usize, Vec<f64>)> {
        let (factor, decay) = (self.factor, self.decay);
        if !factor.is_finite() || factor <= 0.0 {
            return Err(Error::InvalidTail(factor));
        }
        if !(0.0..=1.0).contains(&decay) {
            return Err(Error::Tail("decay must be between 0 and 1"));
        }
        let n_links = estimated.len();
        let attach = match self.attachment_age {
            Some(age) => first_age_at_or_after(ages, n_links, age)?,
            None => n_links,
        };
        let n_tail = n_links + periods + 1 - attach;
        let mut series = vec![1.0; n_tail];
        if factor != 1.0 {
            let x = initial_development(decay, factor)?;
            for (k, f) in series[..n_tail - 1].iter_mut().enumerate() {
                *f = 1.0 + x * decay.powi(k as i32);
            }
            let spread: f64 = series[..n_tail - 1].iter().product();
            series[n_tail - 1] = factor / spread;
        }
        let mut ldf = estimated[..attach].to_vec();
        ldf.extend(series);
        Ok((attach, ldf))
    }
}

/// chainladder-python's seed for a decayed tail: the positive root of
/// `a x^2 + b x - ln(factor) = 0`, with `a` and `b` the sums of
/// `decay^(2k)` and `decay^k` over 1000 periods.
fn initial_development(decay: f64, factor: f64) -> Result<f64> {
    let (mut a, mut b) = (0.0, 0.0);
    let mut d = 1.0;
    for _ in 0..1000 {
        a += d * d;
        b += d;
        d *= decay;
    }
    let c = -factor.ln();
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return Err(Error::Tail(
            "the factor is too small to spread with this decay",
        ));
    }
    Ok((-b + disc.sqrt()) / (2.0 * a))
}

impl TailCurve {
    fn select(&self, estimated: &[f64], ages: &[Lag], periods: usize) -> Result<(usize, Vec<f64>)> {
        let n_links = estimated.len();
        if self.extrap_periods == 0 {
            return Err(Error::Tail("extrap_periods must be at least 1"));
        }
        // Positions as chainladder-python's `int(age / grain - 1)`: the last
        // age at or before the given one.
        let position = |age: Option<Lag>, open: usize| match age {
            Some(age) => Ok(last_age_at_or_before(ages, n_links, age)?.unwrap_or(0)),
            None => Ok::<_, Error>(open),
        };
        let fitted_range = position(self.fit_period.0, 0)?..position(self.fit_period.1, n_links)?;
        let x_of = |k: f64| match self.curve {
            CurveShape::Exponential => k,
            CurveShape::InversePower => k.ln(),
        };
        let points: Vec<(f64, f64)> = estimated
            .iter()
            .enumerate()
            .filter(|&(k, &f)| f > CURVE_THRESHOLD && fitted_range.contains(&k))
            .map(|(k, &f)| (x_of((k + 1) as f64), (f - 1.0).ln()))
            .collect();
        let (a, b) = least_squares_line(&points).ok_or(Error::Tail(
            "a curve needs two factors above 1 in the fit period",
        ))?;
        // Fitted factor at 0-based position p (1-based index p + 1), for the
        // positions up to `extrap_periods` past the oldest age.
        let last = n_links + self.extrap_periods;
        let fitted = |p: usize| {
            if p < last {
                1.0 + (a + b * x_of((p + 1) as f64)).exp()
            } else {
                1.0
            }
        };
        let attach = match self.attachment_age {
            Some(age) => first_age_at_or_after(ages, n_links, age)?,
            None => n_links,
        };
        let individual = n_links + periods;
        let mut ldf = estimated[..attach].to_vec();
        ldf.extend((attach..individual).map(fitted));
        ldf.push((individual..last).map(fitted).product());
        Ok((attach, ldf))
    }
}

impl TailBondy {
    fn select(&self, estimated: &[f64], ages: &[Lag]) -> Result<(usize, Vec<f64>)> {
        let n_links = estimated.len();
        if n_links == 0 {
            return Err(Error::Tail("the Bondy tail needs a development factor"));
        }
        let initial = match self.earliest_age {
            Some(age) => last_age_at_or_before(ages, n_links, age)?
                .ok_or(Error::Tail("earliest_age is before the youngest age"))?,
            None => n_links - 1,
        };
        if initial >= n_links {
            return Err(Error::Tail("earliest_age must be before the oldest age"));
        }
        let kept = match self.attachment_age {
            Some(age) => last_age_at_or_before(ages, n_links, age)?
                .ok_or(Error::Tail("attachment_age is before the youngest age"))?,
            None => n_links - 1,
        };
        if kept < initial {
            return Err(Error::Tail(
                "attachment_age must not be before earliest_age",
            ));
        }
        if kept >= n_links {
            return Err(Error::Tail("attachment_age must be before the oldest age"));
        }
        let logs: Vec<f64> = estimated[initial..].iter().map(|f| f.ln()).collect();
        if logs.iter().any(|v| !v.is_finite()) {
            return Err(Error::Tail("the Bondy tail needs positive factors"));
        }
        let b = if logs.len() == 1 {
            0.5
        } else {
            bondy_exponent(&logs)?
        };
        let f0 = estimated[initial];
        let fitted = |p: usize| f0.powf(b.powi((p - initial) as i32));
        let mut ldf = estimated[..=kept].to_vec();
        ldf.extend((kept + 1..=n_links).map(fitted));
        ldf.push(fitted(n_links).powf(b / (1.0 - b)));
        Ok((kept + 1, ldf))
    }
}

/// The Bondy exponent `b` in (0, 1) minimizing `sum (d_j - c b^j)^2` over
/// `b` and `c`. For a given `b` the best `c` is `A / B`, with
/// `A = sum d_j b^j` and `B = sum b^(2j)`, so `b` maximizes `A^2 / B`: the
/// best point of a grid is refined by bisection on the sign of the
/// derivative, `A (2 A' B - A B')`.
fn bondy_exponent(logs: &[f64]) -> Result<f64> {
    let sums = |b: f64| {
        let (mut a, mut da, mut s, mut ds) = (0.0, 0.0, 0.0, 0.0);
        for (j, d) in logs.iter().enumerate() {
            let j = j as f64;
            let p = b.powf(j);
            a += d * p;
            s += p * p;
            if j > 0.0 {
                da += j * d * b.powf(j - 1.0);
                ds += 2.0 * j * b.powf(2.0 * j - 1.0);
            }
        }
        (a, da, s, ds)
    };
    let gain = |b: f64| {
        let (a, _, s, _) = sums(b);
        a * a / s
    };
    let slope = |b: f64| {
        let (a, da, s, ds) = sums(b);
        a * (2.0 * da * s - a * ds)
    };
    const GRID: usize = 1000;
    let best = (1..GRID)
        .max_by(|&i, &j| {
            let (gi, gj) = (gain(i as f64 / GRID as f64), gain(j as f64 / GRID as f64));
            gi.total_cmp(&gj)
        })
        .expect("grid is not empty");
    if best == 1 || best == GRID - 1 {
        return Err(Error::Tail("the Bondy exponent is not between 0 and 1"));
    }
    let (mut lo, mut hi) = (
        (best - 1) as f64 / GRID as f64,
        (best + 1) as f64 / GRID as f64,
    );
    if slope(lo) <= 0.0 || slope(hi) >= 0.0 {
        return Err(Error::Tail("the Bondy exponent could not be fitted"));
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if slope(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// R ChainLadder's `tailfactor`: when the third- and second-last factors
/// multiply to more than 1.0001, the product of the 100 factors after the
/// last one above 1 on the line fitted to `ln(f - 1)` over the factors above
/// 1 (1-based index); otherwise 1. A product above 2 is reset to 1.
fn log_linear_factor(estimated: &[f64]) -> Result<f64> {
    let n = estimated.len();
    if n < 3 {
        return Err(Error::Tail(
            "R's log-linear tail needs three development factors",
        ));
    }
    if estimated[n - 3] * estimated[n - 2] <= 1.0001 {
        return Ok(1.0);
    }
    let points: Vec<(f64, f64)> = estimated
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f > 1.0)
        .map(|(k, &f)| ((k + 1) as f64, (f - 1.0).ln()))
        .collect();
    let (a, b) = least_squares_line(&points)
        .ok_or(Error::Tail("R's log-linear tail needs two factors above 1"))?;
    let last = points.last().map_or(0.0, |p| p.0);
    let tail: f64 = (1..=100)
        .map(|j| (a + b * (last + j as f64)).exp() + 1.0)
        .product();
    Ok(if tail > 2.0 { 1.0 } else { tail })
}

/// Sigma and standard error of the tail factor, as R's `tail_SE` and
/// chainladder-python's `_get_tail_stats`: the tail's position on the line
/// fitted to `ln(f - 1)` (over the estimated factors above 1) is where it
/// reaches `ln(factor - 1)`, and lines fitted to `ln(sigma)` and
/// `ln(std_err)` (over the positive ones) are read there. NaN when a line
/// cannot be fitted.
fn tail_statistics(development: &DevelopmentFit, factor: f64) -> (f64, f64) {
    let line = |values: &[f64], above: f64, shift: f64| {
        let points: Vec<(f64, f64)> = values
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v.is_finite() && v > above)
            .map(|(k, &v)| ((k + 1) as f64, (v - shift).ln()))
            .collect();
        least_squares_line(&points)
    };
    let Some((a, b)) = line(&development.ldf, 1.0, 1.0) else {
        return (f64::NAN, f64::NAN);
    };
    let position = ((factor - 1.0).ln() - a) / b;
    let at =
        |values: &[f64]| line(values, 0.0, 0.0).map_or(f64::NAN, |(a, b)| (a + b * position).exp());
    (at(&development.sigma), at(&development.std_err))
}

/// Intercept and slope of the least-squares line through `points`, or
/// `None` with fewer than two distinct `x`.
fn least_squares_line(points: &[(f64, f64)]) -> Option<(f64, f64)> {
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

/// Index of the first age at or after `age`, at most the oldest
/// (`n_links`).
fn first_age_at_or_after(ages: &[Lag], n_links: usize, age: Lag) -> Result<usize> {
    if ages.len() != n_links + 1 {
        return Err(Error::Tail("the development fit has no ages"));
    }
    ages.iter()
        .position(|&a| a >= age)
        .ok_or(Error::Tail("the age is past the oldest age"))
}

/// Index of the last age at or before `age`, `None` when `age` is before
/// the youngest. On ages that are multiples of the grain from one grain
/// on, this is chainladder-python's position `int(age / grain) - 1`.
fn last_age_at_or_before(ages: &[Lag], n_links: usize, age: Lag) -> Result<Option<usize>> {
    if ages.len() != n_links + 1 {
        return Err(Error::Tail("the development fit has no ages"));
    }
    Ok(ages.iter().rposition(|&a| a <= age))
}

/// Development periods in a year, the number of factors past the oldest age
/// chainladder-python keeps apart before the one to ultimate (its
/// `projection_period` of 12 months).
fn periods_per_year(ages: &[Lag]) -> usize {
    match ages {
        [a, b, ..] if b > a => (12 / (b - a) as usize).max(1),
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::development::Development;
    use crate::triangle::tests::raa;

    fn close(got: f64, want: f64, tol: f64) {
        assert!((got - want).abs() <= tol, "got {got}, want {want}");
    }

    fn raa_development() -> DevelopmentFit {
        Development::default().fit(&raa(), "values").unwrap()
    }

    #[test]
    fn constant_spreads_with_decay() {
        // chainladder-python TailConstant docstring, RAA: tail = 1.05 gives
        // a 132-Ult cdf of 1.02538; tail = 1.10, decay = 0.75 gives factors
        // 1.023512 and 1.074731 past the oldest age.
        let dev = raa_development();
        let fit = Tail::from(1.05).fit(&dev).unwrap();
        assert_eq!(fit.ldf.len(), 11);
        assert_eq!(fit.attachment, 9);
        close(fit.factor, 1.05, 1e-15);
        close(fit.ldf[10], 1.02538, 5e-6);
        let fit = Tail::Constant(TailConstant {
            factor: 1.1,
            decay: 0.75,
            attachment_age: None,
        })
        .fit(&dev)
        .unwrap();
        close(fit.ldf[9], 1.023512, 5e-7);
        close(fit.ldf[10], 1.074731, 5e-7);
    }

    #[test]
    fn constant_one_is_no_tail() {
        let dev = raa_development();
        let fit = Tail::default().fit(&dev).unwrap();
        assert_eq!(&fit.ldf[..9], &dev.ldf[..]);
        assert_eq!(&fit.ldf[9..], &[1.0, 1.0]);
        assert_eq!((fit.factor, fit.sigma, fit.std_err), (1.0, 0.0, 0.0));
    }

    #[test]
    fn constant_attached_early() {
        // chainladder-python TailConstant docstring, RAA, attachment_age=72:
        // the 72-Ult cdf is the given 1.05, 120-Ult is 1.004156.
        let fit = Tail::Constant(TailConstant {
            factor: 1.05,
            attachment_age: Some(72),
            ..Default::default()
        })
        .fit(&raa_development())
        .unwrap();
        assert_eq!(fit.attachment, 5);
        close(fit.ldf[5..].iter().product(), 1.05, 1e-12);
        close(fit.factor, 1.004156, 5e-7);
    }

    #[test]
    fn constant_attached_at_the_youngest_age_replaces_every_factor() {
        // A deliberate departure: chainladder-python 0.10.1 tests
        // `if attach_idx:` in `_apply_decay`, so attachment_age=12 on RAA
        // (index 0) attaches at the oldest age instead. Here every estimated
        // factor is replaced by the spread of 1.05, whose first factor is
        // chainladder-python's first factor past the oldest age for
        // TailConstant(1.05), 1.0240107.
        let fit = Tail::Constant(TailConstant {
            factor: 1.05,
            attachment_age: Some(12),
            ..Default::default()
        })
        .fit(&raa_development())
        .unwrap();
        assert_eq!(fit.attachment, 0);
        close(fit.ldf[0], 1.024_010_7, 5e-8);
        close(fit.ldf.iter().product(), 1.05, 1e-12);
    }

    #[test]
    fn ages_off_the_grid_take_the_age_at_or_before() {
        // chainladder-python 0.10.1 indexes by int(age / grain - 1), so on
        // RAA TailCurve(fit_period=(30, 102)) is (24, 96): a tail of
        // 1.011280345; TailBondy(earliest_age=30) fits from age 24.
        let dev = raa_development();
        let curve = |fit_period| {
            Tail::Curve(TailCurve {
                fit_period,
                ..Default::default()
            })
            .fit(&dev)
            .unwrap()
        };
        let off = curve((Some(30), Some(102)));
        close(off.factor, 1.011_280_345, 5e-10);
        assert_eq!(off, curve((Some(24), Some(96))));
        let bondy = |earliest_age| {
            Tail::Bondy(TailBondy {
                earliest_age: Some(earliest_age),
                attachment_age: None,
            })
            .fit(&dev)
            .unwrap()
        };
        assert_eq!(bondy(30), bondy(24));
        let before = Tail::Bondy(TailBondy {
            earliest_age: Some(6),
            attachment_age: None,
        });
        assert!(matches!(before.fit(&dev), Err(Error::Tail(_))));
    }

    #[test]
    fn below_one_reads_risk_at_1_001() {
        // chainladder-python 0.10.1 TailConstant(0.98) on RAA: tail sigma
        // 0.112752185 and standard error 0.000601256 (validation/reference/
        // reserving_tails_python.csv).
        let fit = Tail::from(0.98).fit(&raa_development()).unwrap();
        close(fit.sigma, 0.112_752_185_376_787, 1e-9);
        close(fit.std_err, 0.000_601_256_276_110, 1e-12);
    }

    #[test]
    fn exponential_curve_matches_r_rule() {
        // R ChainLadder MackChainLadder(RAA, tail = TRUE)$f[n] = 1.00943575;
        // chainladder-python TailCurve() gives the same on RAA.
        let dev = raa_development();
        let r = Tail::LogLinear.fit(&dev).unwrap();
        let curve = Tail::Curve(TailCurve::default()).fit(&dev).unwrap();
        close(r.factor, 1.009_435_751_581_23, 1e-12);
        close(curve.factor, 1.009_435_751_581_23, 1e-12);
        // R tail_SE: MackChainLadder(RAA, tail = TRUE)$sigma[n], $f.se[n].
        close(r.sigma, 0.905_382_897_553_742, 1e-10);
        close(r.std_err, 0.004_386_030_595_599_17, 1e-12);
    }

    #[test]
    fn bondy_classic_repeats_the_last_factor() {
        // chainladder-python TailBondy docstring, RAA: 1.0046 twice.
        let dev = raa_development();
        let fit = Tail::Bondy(TailBondy::default()).fit(&dev).unwrap();
        close(fit.factor, dev.ldf[8], 1e-12);
        close(fit.ldf[9], dev.ldf[8].sqrt(), 1e-12);
        close(fit.ldf[10], dev.ldf[8].sqrt(), 1e-12);
    }

    #[test]
    fn bondy_generalized() {
        // chainladder-python TailBondy docstring, RAA, earliest_age=12:
        // b = 0.4845 and a tail of 1.0031.
        let fit = Tail::Bondy(TailBondy {
            earliest_age: Some(12),
            attachment_age: None,
        })
        .fit(&raa_development())
        .unwrap();
        close(fit.factor, 1.0031, 5e-5);
    }

    #[test]
    fn bondy_exponent_is_the_least_squares_optimum() {
        // RAA log factors from age 36: scipy least_squares with xtol, ftol
        // and gtol of 1e-15 gives b = 0.61474312367; minimize_scalar on the
        // profiled cost gives 0.61474312423. chainladder-python's default
        // tolerances stop at 0.61474308024.
        let dev = raa_development();
        let logs: Vec<f64> = dev.ldf[2..].iter().map(|f| f.ln()).collect();
        close(bondy_exponent(&logs).unwrap(), 0.614_743_124, 1e-9);
    }

    #[test]
    fn rejects_bad_settings() {
        let dev = raa_development();
        assert_eq!(Tail::from(-1.0).fit(&dev), Err(Error::InvalidTail(-1.0)));
        let bondy = TailBondy {
            earliest_age: Some(60),
            attachment_age: Some(36),
        };
        assert!(matches!(Tail::Bondy(bondy).fit(&dev), Err(Error::Tail(_))));
        let curve = TailCurve {
            fit_period: (Some(108), None),
            ..Default::default()
        };
        assert!(matches!(Tail::Curve(curve).fit(&dev), Err(Error::Tail(_))));
    }
}
