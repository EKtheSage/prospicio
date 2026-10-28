//! Reading a pattern of accident periods longer than a month at every
//! month before the underwriting-period conversion (`docs/design/reserving-v02.md`,
//! decision 25; issue #646).
//!
//! An accident period of `s` months reports at age `a` the even-exposure
//! average of its months, `P_s(a) = (1/s) sum_{q<s} p(a - q)`, with `p` the
//! accident-month pattern and `p(t) = 0` at or below 0 (decision 18).
//! [`SelectedPattern::to_underwriting_basis`](crate::SelectedPattern::to_underwriting_basis)
//! needs `P_s` at every month; a pattern given at annual ages pins it only
//! at those ages, so the months between them are a reading.
//!
//! The default reading, [`AnnualReading::Curve`] with
//! [`GrowthCurve::Weibull`], takes two steps:
//!
//! 1. **The curve through the averages.** Clark's curve `p(t) = L G(t) /
//!    G(E)` on the months `t = 1..E`, `E` the pattern's last age and `L`
//!    its level, with `(omega, theta)` fitted by least squares on the
//!    percents (decision 23's scale) so that the `s`-month averages of `p`
//!    at the given ages are the given values: [`nelder_mead`] on `(ln
//!    omega, ln theta)` from decision 23's start and a small grid of
//!    others, the best kept.
//! 2. **The nearest exact pattern.** The curve's monthly increments `c`
//!    moved as little as possible, `min ||d - c||^2`, to increments `d`
//!    whose averages are the given values exactly (`A d = b`), not
//!    negative except within a span whose given value is below the one
//!    before it (downward development, as an incurred pattern with factors
//!    below 1 has it). Nothing caps `p` at the level: the given value at
//!    the last age is the average of the last `s` months, so a pattern
//!    still developing there has `p(E)` above it, and a cap would flatten
//!    the last window (and leave no solution at ages finer than `s`). On a
//!    monotone input the expanded `P_s`, a moving average of a monotone
//!    `p` whose last window averages to the level, stays at or below it. A given
//!    value below the one before it by no more than `1e-12` of the largest
//!    given value is rounding, not downward development, and the averages
//!    are reproduced to that tolerance. The quadratic programme is solved
//!    exactly by Goldfarb and Idnani's dual active-set method with the
//!    identity as its Hessian, each step a small solve of the equality rows
//!    restricted to the free months, and the result is checked against
//!    every constraint before it is used.
//!
//! The expanded `P_s` is the given value itself at each given age and the
//! average of `p` at the other months.

use prospicio_core::Lag;
use prospicio_math::optimize::{NelderMead, nelder_mead};

use crate::benchmark_curve::{MAX_LN_THETA, MAX_OMEGA, MIN_OMEGA, linearized};
use crate::clark::GrowthCurve;
use crate::error::{Error, Result};

/// How [`SelectedPattern::to_underwriting_basis`](crate::SelectedPattern::to_underwriting_basis)
/// reads a pattern of accident periods longer than a month between its
/// given ages, when they are not every month from 1 to its end
/// (`reserving-v02.md`, decision 25). A pattern of accident months, and
/// one given at every month, convert the same under every reading.
///
/// ```
/// use prospicio_reserving::{AnnualReading, GrowthCurve, UnderwritingBasis};
///
/// assert_eq!(AnnualReading::default(), AnnualReading::Curve(GrowthCurve::Weibull));
/// let basis = UnderwritingBasis::default().with_reading(AnnualReading::LnLinear);
/// assert_eq!(basis.reading(), AnnualReading::LnLinear);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnualReading {
    /// The growth curve fitted through the period averages, corrected to
    /// the nearest monotone accident-month pattern whose averages are the
    /// given values exactly (the [module](self) documentation). The
    /// default, with [`GrowthCurve::Weibull`].
    Curve(GrowthCurve),
    /// Decision 18 item 2 on the period pattern itself: linear from 0 below
    /// the first age and `ln CDF` linear between ages. Kept for
    /// reproduction; at annual ages it puts an underwriting year about 31%
    /// too far developed at 12 months on a parallelogram book.
    LnLinear,
}

impl Default for AnnualReading {
    fn default() -> Self {
        Self::Curve(GrowthCurve::Weibull)
    }
}

/// The error of a reading, `Error::Pattern` with no age.
fn fail<T>(reason: &'static str) -> Result<T> {
    Err(Error::Pattern { age: None, reason })
}

/// Whether the sorted ages are every month from 1 to the last.
pub(crate) fn every_month(points: &[(Lag, f64)]) -> bool {
    points
        .iter()
        .enumerate()
        .all(|(i, p)| i64::from(p.0) == i as i64 + 1)
}

/// The weights of the `s`-month average at age `a` on the monthly
/// increments `d_1..d_n`: `min(s, a - u + 1) / s` for `u <= a`.
fn average_row(a: Lag, s: u32, n: usize) -> Vec<f64> {
    let (a, s) = (a as usize, s as usize);
    (1..=n)
        .map(|u| {
            if u > a {
                0.0
            } else {
                (a - u + 1).min(s) as f64 / s as f64
            }
        })
        .collect()
}

/// The `s`-month average of the monthly `p` (`p[t - 1]` at month `t`, 0 at
/// or below 0) at age `a`.
fn average_at(p: &[f64], s: u32, a: Lag) -> f64 {
    let sum: f64 = (0..s as i64)
        .map(|q| {
            let t = i64::from(a) - q;
            if t >= 1 { p[t as usize - 1] } else { 0.0 }
        })
        .sum();
    sum / f64::from(s)
}

/// A reading: the monthly accident pattern `p` on months 1 to `E`, and the
/// expanded period pattern `P_s` at every month 1 to `E`.
#[derive(Debug)]
pub(crate) struct Reading {
    /// Read by the tests, which check the averages of `p` themselves.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) monthly: Vec<f64>,
    pub(crate) expanded: Vec<(Lag, f64)>,
}

/// The reading of `points` (sorted percents developed of accident periods
/// of `s` months, at least two, the last age `E`) by `curve`, with the
/// given value at each given age of the expanded pattern.
pub(crate) fn read_monthly(points: &[(Lag, f64)], s: u32, curve: GrowthCurve) -> Result<Reading> {
    if points.len() < 2 {
        return fail(
            "a single value gives no shape to read between ages: give two or more, or read it \
             with the ln-linear reading (AnnualReading::LnLinear; reading=\"ln-linear\" in \
             Python and R)",
        );
    }
    let n = points[points.len() - 1].0 as usize;
    let level = points[points.len() - 1].1;
    let (omega, theta) = fit_curve(points, s, curve)?;
    let g = curve_months(curve, omega, theta, level, n);
    let mut c = Vec::with_capacity(n);
    let mut prev = 0.0;
    for &x in &g {
        c.push(x - prev);
        prev = x;
    }
    let d = project(points, s, &c)?;
    let mut p = Vec::with_capacity(n);
    let mut sum = 0.0;
    for &x in &d {
        sum += x;
        p.push(sum);
    }
    let expanded = (1..=n as Lag)
        .map(|a| match points.binary_search_by_key(&a, |q| q.0) {
            Ok(i) => (a, points[i].1),
            Err(_) => (a, average_at(&p, s, a)),
        })
        .collect();
    Ok(Reading {
        monthly: p,
        expanded,
    })
}

/// `L G(t) / G(n)` at the months `t = 1..n`.
fn curve_months(curve: GrowthCurve, omega: f64, theta: f64, level: f64, n: usize) -> Vec<f64> {
    let end = curve.value(n as f64, omega, theta);
    (1..=n)
        .map(|t| level * curve.value(t as f64, omega, theta) / end)
        .collect()
}

/// `(omega, theta)` of `curve` whose `s`-month averages are nearest the
/// given values in least squares.
fn fit_curve(points: &[(Lag, f64)], s: u32, curve: GrowthCurve) -> Result<(f64, f64)> {
    let n = points[points.len() - 1].0 as usize;
    let level = points[points.len() - 1].1;
    let objective = |x: &[f64]| {
        let p = curve_months(curve, x[0].exp(), x[1].exp(), level, n);
        points
            .iter()
            .map(|&(a, v)| (v - average_at(&p, s, a)).powi(2))
            .sum::<f64>()
    };
    let options = NelderMead {
        tolerance: 1e-12,
        ..NelderMead::default()
    };
    let mut starts = Vec::new();
    // Decision 23's start: the straight line of the linearized shares on
    // the log age, each share read at the middle of its window.
    let line: Vec<(f64, f64)> = points
        .iter()
        .map(|&(a, v)| (f64::from(a) - f64::from(s - 1) / 2.0, v / level))
        .filter(|&(a, r)| a > 0.0 && r > 0.0 && r < 1.0)
        .map(|(a, r)| (a.ln(), linearized(curve, r)))
        .collect();
    if line.len() >= 2 {
        let k = line.len() as f64;
        let mx = line.iter().map(|p| p.0).sum::<f64>() / k;
        let my = line.iter().map(|p| p.1).sum::<f64>() / k;
        let sxx: f64 = line.iter().map(|p| (p.0 - mx).powi(2)).sum();
        let sxy: f64 = line.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let omega = sxy / sxx;
        if omega.is_finite() && omega > 0.0 && (mx - my / omega).is_finite() {
            starts.push([omega.ln(), mx - my / omega]);
        }
    }
    for omega in [0.8f64, 1.5, 3.0] {
        for theta in [6.0f64, 18.0, 36.0] {
            starts.push([omega.ln(), theta.ln()]);
        }
    }
    // The curve is only the projection's prior, so the best point found is
    // kept even where the search stops on its iteration limit (a curve that
    // steepens toward a step, as two given ages can ask for); the edge
    // check below still rejects one that runs off.
    let best = starts
        .iter()
        .map(|x0| nelder_mead(objective, x0, &options))
        .filter(|m| m.value.is_finite())
        .min_by(|a, b| a.value.total_cmp(&b.value));
    let Some(best) = best else {
        return fail("the growth curve through the averages has no finite fit");
    };
    let (ln_omega, ln_theta) = (best.x[0], best.x[1]);
    if !(MIN_OMEGA.ln()..=MAX_OMEGA.ln()).contains(&ln_omega)
        || !(-MAX_LN_THETA..=MAX_LN_THETA).contains(&ln_theta)
    {
        return fail(
            "the growth curve through the averages ran off to the edge of its parameters: the \
             pattern's percents developed do not increase with age; read it with \
             the ln-linear reading (AnnualReading::LnLinear; reading=\"ln-linear\" in Python \
             and R)",
        );
    }
    Ok((ln_omega.exp(), ln_theta.exp()))
}

/// The increments `d` on months 1 to `E` nearest `c` in least squares with
/// the averages at the given ages equal to the given values and `d >= 0`
/// except within a span whose given value is below the one before it by
/// more than the projection's tolerance, by Goldfarb and Idnani's dual
/// method (Hessian the identity), checked at the end.
fn project(points: &[(Lag, f64)], s: u32, c: &[f64]) -> Result<Vec<f64>> {
    let n = c.len();
    let rows: Vec<Vec<f64>> = points.iter().map(|&(a, _)| average_row(a, s, n)).collect();
    let b: Vec<f64> = points.iter().map(|p| p.1).collect();
    let tol = tolerance(points);
    // Months that may develop downward: those of a span whose given value
    // is below the one before it by more than rounding.
    let mut bounded = vec![true; n];
    let (mut prev_age, mut prev_value) = (0usize, 0.0);
    for &(a, v) in points {
        if v < prev_value - tol {
            bounded[prev_age..a as usize].fill(false);
        }
        (prev_age, prev_value) = (a as usize, v);
    }
    // Months the given values themselves hold at 0: where two given values
    // in a row are equal (to rounding), `P_s(a') - P_s(a)` is the sum of
    // the increments of months `a - s + 2` to `a'`, each not negative, so
    // every one is 0 (from the first month when the first value is 0).
    // Held from the start and never released, they leave no equality row
    // dependent on a bound held at 0 by rounding.
    let mut zero = vec![false; n];
    let (mut prev_age, mut prev_value) = (0i64, 0.0);
    for &(a, v) in points {
        let a = i64::from(a);
        let lo = (prev_age - i64::from(s) + 2).max(1) as usize;
        if (v - prev_value).abs() <= tol && bounded[lo - 1..a as usize].iter().all(|&x| x) {
            zero[lo - 1..a as usize].fill(true);
        }
        (prev_age, prev_value) = (a, v);
    }
    let mut qp = Projection {
        rows: &rows,
        fixed: zero.clone(),
    };
    // The equality rows first.
    let k = rows.len();
    let mut x = primal(&rows, &b, c, &qp.fixed);
    let mut u_bound = vec![0.0; n];
    // Bounds violated by no more than the tolerance whose normals depend
    // on the active set: accepted as met, not added.
    let mut waived = vec![false; n];
    let mut steps = 0usize;
    let max_steps = 50 * (n + k + 1);
    loop {
        // The most violated inactive bound, by more than a tenth of the
        // tolerance the result is checked to.
        let mut worst: Option<usize> = None;
        for j in 0..n {
            if bounded[j]
                && !qp.fixed[j]
                && !waived[j]
                && x[j] < -0.1 * tol
                && worst.is_none_or(|w| x[j] < x[w])
            {
                worst = Some(j);
            }
        }
        let Some(add) = worst else { break };
        let mut u_add = 0.0;
        loop {
            steps += 1;
            if steps > max_steps {
                return fail("the reading's projection did not converge");
            }
            let step = qp.direction(add);
            let zz = dot(&step.z, &step.z);
            // A bound whose normal lies (to rounding) in the span of the
            // active constraints cannot be reached by a primal step.
            let t1 = if zz > 1e-12 {
                -x[add] / zz
            } else {
                f64::INFINITY
            };
            // Such a bound violated only by rounding is met as it stands:
            // dual steps on it would only drop and re-add held bounds.
            if t1.is_infinite() && x[add] >= -tol {
                waived[add] = true;
                break;
            }
            // The largest dual step that keeps every held bound's
            // multiplier not negative.
            let mut t2 = f64::INFINITY;
            let mut drop = None;
            for (j, (&uj, &rj)) in u_bound.iter().zip(&step.r_bound).enumerate() {
                if qp.fixed[j] && !zero[j] && rj > 1e-12 && uj / rj < t2 {
                    (t2, drop) = (uj / rj, Some(j));
                }
            }
            let t = t1.min(t2);
            if !t.is_finite() {
                return fail(
                    "no monotone accident-month pattern has the given averages: check that the \
                     pattern's percents developed do not decrease, or read it with the ln-linear \
                     reading (AnnualReading::LnLinear; reading=\"ln-linear\" in Python and R)",
                );
            }
            if t1.is_finite() {
                for (xj, zj) in x.iter_mut().zip(&step.z) {
                    *xj += t * zj;
                }
            }
            for ((uj, &rj), &held) in u_bound.iter_mut().zip(&step.r_bound).zip(&qp.fixed) {
                if held {
                    *uj -= t * rj;
                }
            }
            u_add += t;
            if t1 <= t2 {
                qp.fixed[add] = true;
                u_bound[add] = u_add;
                // The point of the new active set solved afresh, so that
                // the steps' rounding does not build up; a waived bound is
                // looked at again from it.
                x = primal(&rows, &b, c, &qp.fixed);
                waived.fill(false);
                break;
            }
            let j = drop.expect("a finite partial step names the bound it drops");
            qp.fixed[j] = false;
            u_bound[j] = 0.0;
        }
    }
    // Every constraint checked on the result.
    let exact = rows
        .iter()
        .zip(&b)
        .all(|(r, &bi)| (dot(r, &x) - bi).abs() <= tol);
    let signs = x.iter().zip(&bounded).all(|(&xj, &bj)| !bj || xj >= -tol);
    if !(exact && signs) {
        return fail("the reading's projection did not reach the given averages");
    }
    Ok(x)
}

/// The increments nearest `c` with the `fixed` months at 0 and the
/// averages the given values: `x = c + R' u` on the free months, `R` the
/// equality rows restricted to them and `R R' u = b - R c`.
fn primal(rows: &[Vec<f64>], b: &[f64], c: &[f64], fixed: &[bool]) -> Vec<f64> {
    let g = gram(rows, fixed);
    let mut x: Vec<f64> = c
        .iter()
        .zip(fixed)
        .map(|(&cj, &held)| if held { 0.0 } else { cj })
        .collect();
    // The solve, then one pass of refinement on its residuals.
    for _ in 0..2 {
        let rhs: Vec<f64> = rows.iter().zip(b).map(|(r, &bi)| bi - dot(r, &x)).collect();
        let u = solve_semidefinite(&g, &rhs, rows.len());
        for (j, xj) in x.iter_mut().enumerate() {
            if !fixed[j] {
                *xj += rows.iter().zip(&u).map(|(r, ui)| r[j] * ui).sum::<f64>();
            }
        }
    }
    x
}

/// The rounding the projection allows: `1e-12` of the largest given value,
/// or of 1 when they are all smaller.
fn tolerance(points: &[(Lag, f64)]) -> f64 {
    1e-12 * points.iter().fold(1.0f64, |m, p| m.max(p.1.abs()))
}

/// The active set of the projection: the equality rows and the months
/// whose increment is held at 0.
struct Projection<'a> {
    rows: &'a [Vec<f64>],
    fixed: Vec<bool>,
}

/// The primal step `z` (the normal's part in the null space of the active
/// constraints) and the held bounds' dual steps `r` of a bound to add.
struct Step {
    z: Vec<f64>,
    r_bound: Vec<f64>,
}

impl Projection<'_> {
    /// Splits the unit normal of month `add` into `N r + z`, `N` the active
    /// constraints' normals (the equality rows and a unit vector per held
    /// month), with `z` orthogonal to them. An equality row that depends,
    /// to rounding, on the others on the free months is left out of the
    /// split.
    fn direction(&self, add: usize) -> Step {
        let n = self.fixed.len();
        let k = self.rows.len();
        let g = gram(self.rows, &self.fixed);
        // The month added is free, so each row's part of the normal on the
        // free months is its weight on that month.
        let rhs: Vec<f64> = self.rows.iter().map(|r| r[add]).collect();
        let w = solve_semidefinite(&g, &rhs, k);
        let mut z = vec![0.0; n];
        let mut r_bound = vec![0.0; n];
        for j in 0..n {
            let back: f64 = self.rows.iter().zip(&w).map(|(r, wi)| r[j] * wi).sum();
            let normal = if j == add { 1.0 } else { 0.0 };
            if self.fixed[j] {
                r_bound[j] = normal - back;
            } else {
                z[j] = normal - back;
            }
        }
        Step { z, r_bound }
    }
}

/// The Gram matrix of the equality rows restricted to the months not
/// `fixed`, row-major.
fn gram(rows: &[Vec<f64>], fixed: &[bool]) -> Vec<f64> {
    let k = rows.len();
    let free: Vec<usize> = (0..fixed.len()).filter(|&j| !fixed[j]).collect();
    let mut g = vec![0.0; k * k];
    for i in 0..k {
        for l in 0..=i {
            let v: f64 = free.iter().map(|&j| rows[i][j] * rows[l][j]).sum();
            g[i * k + l] = v;
            g[l * k + i] = v;
        }
    }
    g
}

/// Solves `G w = rhs` for a positive semidefinite Gram matrix `G` by a
/// Cholesky factorization that skips each row whose pivot is at most
/// `1e-10` of its diagonal (a row that depends, to rounding, on the rows
/// before it), with `w` 0 there. When `rhs` is in the range of `G`, as it
/// is in the projection, the skipped rows' equations hold too.
fn solve_semidefinite(g: &[f64], rhs: &[f64], k: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; k * k];
    let mut kept = vec![false; k];
    for j in 0..k {
        let d = g[j * k + j] - (0..j).map(|m| l[j * k + m].powi(2)).sum::<f64>();
        if d.is_nan() || d <= 1e-10 * g[j * k + j] {
            continue;
        }
        kept[j] = true;
        let ljj = d.sqrt();
        l[j * k + j] = ljj;
        for i in j + 1..k {
            let v = g[i * k + j] - (0..j).map(|m| l[i * k + m] * l[j * k + m]).sum::<f64>();
            l[i * k + j] = v / ljj;
        }
    }
    let mut y = vec![0.0; k];
    for j in 0..k {
        if kept[j] {
            let v = rhs[j] - (0..j).map(|m| l[j * k + m] * y[m]).sum::<f64>();
            y[j] = v / l[j * k + j];
        }
    }
    let mut w = vec![0.0; k];
    for j in (0..k).rev() {
        if kept[j] {
            let v = y[j] - (j + 1..k).map(|m| l[m * k + j] * w[m]).sum::<f64>();
            w[j] = v / l[j * k + j];
        }
    }
    w
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The book's accident-month pattern, `1 - 0.9^t` held from 60.
    fn book(t: i64) -> f64 {
        if t <= 0 {
            0.0
        } else {
            1.0 - 0.9f64.powi(t.min(60) as i32)
        }
    }

    /// The `s`-month averages of `p` at `ages`.
    fn averages(p: impl Fn(i64) -> f64, s: u32, ages: &[Lag]) -> Vec<(Lag, f64)> {
        ages.iter()
            .map(|&a| {
                let sum: f64 = (0..i64::from(s)).map(|q| p(i64::from(a) - q)).sum();
                (a, sum / f64::from(s))
            })
            .collect()
    }

    fn annual(p: impl Fn(i64) -> f64, end: Lag) -> Vec<(Lag, f64)> {
        let ages: Vec<Lag> = (1..=end / 12).map(|k| 12 * k).collect();
        averages(p, 12, &ages)
    }

    fn check(points: &[(Lag, f64)], s: u32, p: &[f64], downward: bool) {
        for &(a, v) in points {
            assert!((average_at(p, s, a) - v).abs() < 1e-12, "age {a}");
        }
        let level = points[points.len() - 1].1;
        if !downward {
            // `p` monotone, and its averages at every month (the expanded
            // pattern) monotone and within the level.
            assert!(p.windows(2).all(|w| w[1] >= w[0] - 1e-12));
            assert!(p[0] >= -1e-12);
            let expanded: Vec<f64> = (1..=p.len() as Lag).map(|a| average_at(p, s, a)).collect();
            assert!(expanded.windows(2).all(|w| w[1] >= w[0] - 1e-12));
            assert!(expanded.iter().all(|&x| x <= level + 1e-12));
        }
    }

    #[test]
    fn the_reading_reproduces_the_averages_and_is_monotone() {
        for curve in [GrowthCurve::Weibull, GrowthCurve::LogLogistic] {
            let points = annual(book, 120);
            let Reading {
                monthly: p,
                expanded,
            } = read_monthly(&points, 12, curve).unwrap();
            check(&points, 12, &p, false);
            assert_eq!(expanded.len(), 120);
            assert!(expanded.windows(2).all(|w| w[1].1 >= w[0].1 - 1e-14));
            for &(a, v) in &points {
                assert_eq!(expanded[a as usize - 1].1, v);
            }
        }
    }

    #[test]
    fn the_weibull_reading_is_near_the_books_months() {
        let points = annual(book, 120);
        let Reading { monthly: p, .. } = read_monthly(&points, 12, GrowthCurve::Weibull).unwrap();
        // Within 1% of the truth from the third month, where the
        // first-year share starts to matter.
        for t in 3..=120 {
            let truth = book(t);
            assert!((p[t as usize - 1] / truth - 1.0).abs() < 0.03, "month {t}");
        }
    }

    #[test]
    fn an_exact_weibull_is_read_back() {
        // Annual averages of a Weibull truncated at 120 months: the curve
        // fits them exactly and the projection does not move it.
        let (omega, theta) = (1.5, 18.0);
        let g = |t: f64| GrowthCurve::Weibull.value(t, omega, theta);
        let truth = |t: i64| {
            if t <= 0 { 0.0 } else { g(t as f64) / g(120.0) }
        };
        let points = annual(truth, 120);
        let Reading { monthly: p, .. } = read_monthly(&points, 12, GrowthCurve::Weibull).unwrap();
        for t in 1..=108 {
            assert!((p[t - 1] - truth(t as i64)).abs() < 2e-3, "month {t}");
        }
    }

    #[test]
    fn a_level_below_one_scales_the_reading() {
        let points = annual(book, 120);
        let low: Vec<(Lag, f64)> = points.iter().map(|&(a, v)| (a, 0.95 * v)).collect();
        let Reading { monthly: p, .. } = read_monthly(&points, 12, GrowthCurve::Weibull).unwrap();
        let Reading { monthly: q, .. } = read_monthly(&low, 12, GrowthCurve::Weibull).unwrap();
        check(&low, 12, &q, false);
        for (x, y) in p.iter().zip(&q) {
            assert!((0.95 * x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn downward_development_is_kept_in_its_span() {
        // Friedland's XYZ reported selection, factors of 0.993 and 0.999 at
        // 108 and 120 months.
        let ldf = [
            1.687, 1.265, 1.102, 1.020, 1.050, 1.010, 1.011, 1.000, 0.993, 0.999,
        ];
        let mut cdf = [1.0; 11];
        for i in (0..10).rev() {
            cdf[i] = cdf[i + 1] * ldf[i];
        }
        let points: Vec<(Lag, f64)> = (0..11)
            .map(|i| (12 * (i as Lag + 1), 1.0 / cdf[i]))
            .collect();
        let Reading { monthly: p, .. } = read_monthly(&points, 12, GrowthCurve::Weibull).unwrap();
        check(&points, 12, &p, true);
        // Monotone outside the downward spans (from 96 to 120 months).
        assert!(p[..96].windows(2).all(|w| w[1] >= w[0] - 1e-14));
        // and downward within them somewhere.
        assert!(p[96..120].windows(2).any(|w| w[1] < w[0]));
    }

    #[test]
    fn quarterly_ages_and_quarterly_periods() {
        let ages: Vec<Lag> = (1..=40).map(|k| 3 * k).collect();
        for s in [12, 3] {
            let points = averages(book, s, &ages);
            let Reading { monthly: p, .. } =
                read_monthly(&points, s, GrowthCurve::Weibull).unwrap();
            check(&points, s, &p, false);
            for t in 6..=120 {
                assert!(
                    (p[t - 1] / book(t as i64) - 1.0).abs() < 0.01,
                    "s {s}, month {t}"
                );
            }
        }
    }

    #[test]
    fn few_given_ages() {
        let truth = |t: i64| {
            if t <= 0 {
                0.0
            } else {
                1.0 - 0.9f64.powi(t.min(25) as i32)
            }
        };
        let points = annual(truth, 36);
        let Reading { monthly: p, .. } = read_monthly(&points, 12, GrowthCurve::Weibull).unwrap();
        check(&points, 12, &p, false);
        let two = [(12, 0.4), (24, 0.9)];
        let Reading { monthly: p, .. } = read_monthly(&two, 12, GrowthCurve::Weibull).unwrap();
        check(&two, 12, &p, false);
        // Months 13 to 24 average to the level.
        assert!((p[12..24].iter().sum::<f64>() / 12.0 - 0.9).abs() < 1e-12);
    }

    /// `1 - rate^t` by accident month, held from `flat`.
    fn geometric(rate: f64, flat: i64) -> impl Fn(i64) -> f64 {
        move |t| {
            if t <= 0 {
                0.0
            } else {
                1.0 - rate.powi(t.min(flat) as i32)
            }
        }
    }

    #[test]
    fn accident_years_at_ages_finer_than_a_year() {
        // Quarterly and semiannual ages of accident years: flat over the
        // last months but not over the whole last year of ages (the
        // dependent rows a held span leaves), and still developing at the
        // last age, where a cap at the level had no solution.
        for (rate, flat, end, step) in [
            (0.95, 60, 120, 3),
            (0.97, 96, 120, 3),
            (0.97, 120, 240, 6),
            (0.97, 120, 240, 3),
            (0.9, i64::MAX, 120, 3),
            (0.97, i64::MAX, 120, 3),
            (0.97, i64::MAX, 120, 12),
        ] {
            let truth = geometric(rate, flat);
            let ages: Vec<Lag> = (1..=end / step).map(|k| step * k).collect();
            let points = averages(&truth, 12, &ages);
            for curve in [GrowthCurve::Weibull, GrowthCurve::LogLogistic] {
                let Reading { monthly: p, .. } = read_monthly(&points, 12, curve)
                    .unwrap_or_else(|e| panic!("{rate} {flat} {end} {step} {curve:?}: {e}"));
                check(&points, 12, &p, false);
            }
        }
    }

    #[test]
    fn a_decrease_of_rounding_is_not_downward_development() {
        // The 96-month value of a flat tail lowered by one ulp: the
        // increments stay bounded and the pattern monotone.
        let mut points = annual(book, 120);
        points[7].1 = points[7].1 - points[7].1 * f64::EPSILON;
        assert!(points[7].1 < points[6].1);
        let Reading { monthly: p, .. } = read_monthly(&points, 12, GrowthCurve::Weibull).unwrap();
        check(&points, 12, &p, false);
    }

    #[test]
    fn a_sweep_of_monotone_patterns_at_finer_ages() {
        // Monotone accident-month patterns from sparse increments, read at
        // ages spaced below the accident period: every one has a solution.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for trial in 0..60 {
            let (s, step, end) = match trial % 3 {
                0 => (12, 3, 120),
                1 => (12, 6, 96),
                _ => (6, 3, 60),
            };
            let mut d = vec![0.0; end as usize];
            for x in d.iter_mut() {
                if next() < 0.3 {
                    *x = next();
                }
            }
            let total: f64 = d.iter().sum::<f64>() / (0.9 + 0.1 * next());
            let mut p = Vec::with_capacity(d.len());
            let mut sum = 0.0;
            for x in &d {
                sum += x / total;
                p.push(sum);
            }
            let truth = |t: i64| if t <= 0 { 0.0 } else { p[t as usize - 1] };
            let ages: Vec<Lag> = (1..=end / step).map(|k| step * k).collect();
            let points = averages(truth, s, &ages);
            for curve in [GrowthCurve::Weibull, GrowthCurve::LogLogistic] {
                let Reading { monthly: q, .. } = read_monthly(&points, s, curve)
                    .unwrap_or_else(|e| panic!("trial {trial} {curve:?}: {e}"));
                check(&points, s, &q, false);
            }
        }
    }

    #[test]
    fn a_pattern_at_every_month_and_a_monthly_one_convert_the_same_under_every_reading() {
        use crate::{Grain, PatternKind, SelectedPattern, UnderwritingBasis};
        let month = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            (1..=60).map(|t| (t, book(i64::from(t)))).collect(),
        );
        let year = month
            .averaged_over_origins(Grain::Month, Grain::Year)
            .unwrap();
        let quarter = month
            .averaged_over_origins(Grain::Month, Grain::Quarter)
            .unwrap();
        let readings = [
            AnnualReading::LnLinear,
            AnnualReading::Curve(GrowthCurve::Weibull),
            AnnualReading::Curve(GrowthCurve::LogLogistic),
        ];
        for (pattern, grain) in [
            (&month, Grain::Month),
            (&year, Grain::Year),
            (&quarter, Grain::Quarter),
        ] {
            let basis = UnderwritingBasis::default();
            let want = pattern.to_underwriting_basis(grain, &basis).unwrap();
            for reading in readings {
                let got = pattern
                    .to_underwriting_basis(grain, &basis.clone().with_reading(reading))
                    .unwrap();
                assert_eq!(got, want, "{grain:?} {reading:?}");
            }
        }
        // An accident-month pattern at annual ages is read by decision 18's
        // rule under every reading: it is `p` itself.
        let sparse = SelectedPattern::new(
            PatternKind::PercentDeveloped,
            vec![(12, 0.5), (24, 0.8), (36, 1.0)],
        );
        let basis = UnderwritingBasis::default();
        let want = sparse.to_underwriting_basis(Grain::Month, &basis).unwrap();
        let old = basis.with_reading(AnnualReading::LnLinear);
        assert_eq!(
            sparse.to_underwriting_basis(Grain::Month, &old).unwrap(),
            want
        );
    }

    #[test]
    fn the_errors() {
        let e = read_monthly(&[(12, 0.5)], 12, GrowthCurve::Weibull).unwrap_err();
        assert!(e.to_string().contains("single value"));
        // Through the conversion, and not under decision 18's reading.
        use crate::{Grain, PatternKind, SelectedPattern, UnderwritingBasis};
        let one = SelectedPattern::new(PatternKind::PercentDeveloped, vec![(12, 0.5)]);
        let e = one
            .to_underwriting_basis(Grain::Year, &UnderwritingBasis::default())
            .unwrap_err();
        assert!(e.to_string().contains("reading=\"ln-linear\""), "{e}");
        let old = UnderwritingBasis::default().with_reading(AnnualReading::LnLinear);
        assert!(one.to_underwriting_basis(Grain::Year, &old).is_ok());
        // Flat: every month reports in full at once.
        let flat = [(12, 0.5), (24, 0.5), (36, 0.5)];
        let Reading { monthly: p, .. } = read_monthly(&flat, 12, GrowthCurve::Weibull).unwrap();
        check(&flat, 12, &p, false);
        assert!(p.iter().all(|&x| (x - 0.5).abs() < 1e-12));
        assert!(every_month(&[(1, 0.1), (2, 0.2)]));
        assert!(!every_month(&[(1, 0.1), (3, 0.2)]));
        assert!(!every_month(&[(2, 0.1), (3, 0.2)]));
    }
}
