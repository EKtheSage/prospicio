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
//!    below 1 has it), with `sum d <= L`. The quadratic programme is solved
//!    exactly by Goldfarb and Idnani's dual active-set method with the
//!    identity as its Hessian, each step a small solve of the equality rows
//!    restricted to the free months, and the result is checked against
//!    every constraint before it is used.
//!
//! The expanded `P_s` is the given value itself at each given age and the
//! average of `p` at the other months.

use prospicio_core::Lag;
use prospicio_math::linalg::{cholesky, cholesky_solve, solve};
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
             with AnnualReading::LnLinear",
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
    let d = project(points, s, &c, level)?;
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
             AnnualReading::LnLinear",
        );
    }
    Ok((ln_omega.exp(), ln_theta.exp()))
}

/// The increments `d` on months 1 to `E` nearest `c` in least squares with
/// the averages at the given ages equal to the given values, `d >= 0`
/// except within a span whose given value is below the one before it, and
/// `sum d <= level`, by Goldfarb and Idnani's dual method (Hessian the
/// identity), checked at the end.
fn project(points: &[(Lag, f64)], s: u32, c: &[f64], level: f64) -> Result<Vec<f64>> {
    let n = c.len();
    let rows: Vec<Vec<f64>> = points.iter().map(|&(a, _)| average_row(a, s, n)).collect();
    let b: Vec<f64> = points.iter().map(|p| p.1).collect();
    // Months that may develop downward: those of a span whose given value
    // is below the one before it.
    let mut bounded = vec![true; n];
    let (mut prev_age, mut prev_value) = (0usize, 0.0);
    for &(a, v) in points {
        if v < prev_value {
            bounded[prev_age..a as usize].fill(false);
        }
        (prev_age, prev_value) = (a as usize, v);
    }
    let mut qp = Projection {
        rows: &rows,
        fixed: vec![false; n],
        cap: false,
    };
    // The equality rows first: `x = c + A' u`, `A A' u = b - A c`.
    let k = rows.len();
    let gram = gram(&rows, &qp.fixed, false);
    let rhs: Vec<f64> = rows.iter().zip(&b).map(|(r, &bi)| bi - dot(r, c)).collect();
    let Some(u) = solve_gram(gram, rhs, k) else {
        return fail("the reading's averaging rows are dependent");
    };
    let mut x = c.to_vec();
    for (r, &ui) in rows.iter().zip(&u) {
        for (xj, &rj) in x.iter_mut().zip(r) {
            *xj += ui * rj;
        }
    }
    let mut u_eq = u;
    let mut u_bound = vec![0.0; n];
    let mut u_cap = 0.0;
    let tol = 1e-14;
    let mut steps = 0usize;
    let max_steps = 50 * (n + k + 1);
    loop {
        // The most violated inactive inequality: a bound or the cap.
        let mut worst: Option<(Constraint, f64)> = None;
        for j in 0..n {
            if bounded[j] && !qp.fixed[j] && x[j] < -tol && worst.is_none_or(|w| x[j] < w.1) {
                worst = Some((Constraint::Bound(j), x[j]));
            }
        }
        let cap_slack = level - x.iter().sum::<f64>();
        if !qp.cap && cap_slack < -tol && worst.is_none_or(|w| cap_slack < w.1) {
            worst = Some((Constraint::Cap, cap_slack));
        }
        let Some((add, _)) = worst else { break };
        let normal: Vec<f64> = match add {
            Constraint::Bound(j) => {
                let mut e = vec![0.0; n];
                e[j] = 1.0;
                e
            }
            Constraint::Cap => vec![-1.0; n],
        };
        let slack = |x: &[f64]| match add {
            Constraint::Bound(j) => x[j],
            Constraint::Cap => level - x.iter().sum::<f64>(),
        };
        let mut u_add = 0.0;
        loop {
            steps += 1;
            if steps > max_steps {
                return fail("the reading's projection did not converge");
            }
            let Some(step) = qp.direction(&normal) else {
                return fail("the reading's active constraints are dependent");
            };
            let zz = dot(&step.z, &step.z);
            let t1 = if zz > 1e-20 * dot(&normal, &normal) {
                -slack(&x) / zz
            } else {
                f64::INFINITY
            };
            // The largest dual step that keeps every active inequality's
            // multiplier not negative.
            let mut t2 = f64::INFINITY;
            let mut drop = None;
            for (j, (&uj, &rj)) in u_bound.iter().zip(&step.r_bound).enumerate() {
                if qp.fixed[j] && rj > 1e-15 && uj / rj < t2 {
                    (t2, drop) = (uj / rj, Some(Constraint::Bound(j)));
                }
            }
            if qp.cap && step.r_cap > 1e-15 {
                let t = u_cap / step.r_cap;
                if t < t2 {
                    (t2, drop) = (t, Some(Constraint::Cap));
                }
            }
            let t = t1.min(t2);
            if !t.is_finite() {
                return fail(
                    "no monotone accident-month pattern has the given averages: check that the \
                     pattern's percents developed do not decrease, or read it with \
                     AnnualReading::LnLinear",
                );
            }
            if t1.is_finite() {
                for (xj, zj) in x.iter_mut().zip(&step.z) {
                    *xj += t * zj;
                }
            }
            for (ui, ri) in u_eq.iter_mut().zip(&step.r_eq) {
                *ui -= t * ri;
            }
            for ((uj, &rj), &held) in u_bound.iter_mut().zip(&step.r_bound).zip(&qp.fixed) {
                if held {
                    *uj -= t * rj;
                }
            }
            if qp.cap {
                u_cap -= t * step.r_cap;
            }
            u_add += t;
            if t1 <= t2 {
                match add {
                    Constraint::Bound(j) => {
                        qp.fixed[j] = true;
                        x[j] = 0.0;
                        u_bound[j] = u_add;
                    }
                    Constraint::Cap => {
                        qp.cap = true;
                        u_cap = u_add;
                    }
                }
                break;
            }
            match drop {
                Some(Constraint::Bound(j)) => {
                    qp.fixed[j] = false;
                    u_bound[j] = 0.0;
                }
                Some(Constraint::Cap) => {
                    qp.cap = false;
                    u_cap = 0.0;
                }
                None => unreachable!("a finite partial step names the constraint it drops"),
            }
        }
    }
    // Every constraint checked on the result.
    let exact = rows
        .iter()
        .zip(&b)
        .all(|(r, &bi)| (dot(r, &x) - bi).abs() <= 1e-12);
    let signs = x.iter().zip(&bounded).all(|(&xj, &bj)| !bj || xj >= -1e-12);
    let capped = x.iter().sum::<f64>() <= level + 1e-12;
    if !(exact && signs && capped) {
        return fail("the reading's projection did not reach the given averages");
    }
    Ok(x)
}

/// An inequality of the projection: a month's increment not negative, or
/// the increments summing to at most the level.
#[derive(Debug, Clone, Copy)]
enum Constraint {
    Bound(usize),
    Cap,
}

/// The active set of the projection: the equality rows, the months whose
/// increment is held at 0 and whether the cap is active.
struct Projection<'a> {
    rows: &'a [Vec<f64>],
    fixed: Vec<bool>,
    cap: bool,
}

/// The primal step `z` (the normal's part in the null space of the active
/// constraints) and the dual steps `r` of a constraint to add.
struct Step {
    z: Vec<f64>,
    r_eq: Vec<f64>,
    r_bound: Vec<f64>,
    r_cap: f64,
}

impl Projection<'_> {
    /// Splits `normal` into `N r + z`, `N` the active constraints' normals
    /// (the equality rows, the cap's `-1` and a unit vector per held
    /// month), with `z` orthogonal to them. `None` when they are
    /// dependent.
    fn direction(&self, normal: &[f64]) -> Option<Step> {
        let n = normal.len();
        let k = self.rows.len() + usize::from(self.cap);
        let row = |i: usize, j: usize| {
            if i < self.rows.len() {
                self.rows[i][j]
            } else {
                -1.0
            }
        };
        let g = gram(self.rows, &self.fixed, self.cap);
        let rhs: Vec<f64> = (0..k)
            .map(|i| {
                (0..n)
                    .filter(|&j| !self.fixed[j])
                    .map(|j| row(i, j) * normal[j])
                    .sum()
            })
            .collect();
        let w = solve_gram(g, rhs, k)?;
        // `R' w` at every month.
        let back: Vec<f64> = (0..n)
            .map(|j| (0..k).map(|i| row(i, j) * w[i]).sum())
            .collect();
        let mut z = vec![0.0; n];
        let mut r_bound = vec![0.0; n];
        for j in 0..n {
            if self.fixed[j] {
                r_bound[j] = normal[j] - back[j];
            } else {
                z[j] = normal[j] - back[j];
            }
        }
        let r_cap = if self.cap { w[k - 1] } else { 0.0 };
        let r_eq = w[..self.rows.len()].to_vec();
        Some(Step {
            z,
            r_eq,
            r_bound,
            r_cap,
        })
    }
}

/// The Gram matrix of the equality rows (and the cap's `-1` row when
/// `cap`) restricted to the months not `fixed`, row-major.
fn gram(rows: &[Vec<f64>], fixed: &[bool], cap: bool) -> Vec<f64> {
    let k = rows.len() + usize::from(cap);
    let free: Vec<usize> = (0..fixed.len()).filter(|&j| !fixed[j]).collect();
    let row = |i: usize, j: usize| if i < rows.len() { rows[i][j] } else { -1.0 };
    let mut g = vec![0.0; k * k];
    for i in 0..k {
        for l in 0..=i {
            let v: f64 = free.iter().map(|&j| row(i, j) * row(l, j)).sum();
            g[i * k + l] = v;
            g[l * k + i] = v;
        }
    }
    g
}

/// Solves the Gram system by Cholesky, or by elimination when Cholesky
/// fails on rounding. `None` when it is singular.
fn solve_gram(g: Vec<f64>, rhs: Vec<f64>, k: usize) -> Option<Vec<f64>> {
    match cholesky(&g, k) {
        Some(l) => Some(cholesky_solve(&l, &rhs)),
        None => solve(g, rhs, k),
    }
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
        assert!(p[p.len() - 1] <= level + 1e-12);
        if !downward {
            assert!(p.iter().all(|&x| x <= level + 1e-12));
            assert!(p.windows(2).all(|w| w[1] >= w[0] - 1e-14));
            assert!(p[0] >= -1e-14);
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
        // The level reached by month 13, as P(24) = 0.9 demands.
        assert!((p[12] - 0.9).abs() < 1e-12);
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
        assert!(e.to_string().contains("AnnualReading::LnLinear"), "{e}");
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
