//! Root finding in one variable.
//!
//! Every solver here takes a bracket and returns a point inside it; none
//! extrapolates. Callers own the bracket, because only they know the
//! domain (an alpha in `[1e-6, 1e6]`, a probability in `(0, 1)`).

use std::convert::Infallible;

/// The boundary in `(lo, hi)` of a predicate that holds below a point and
/// fails above it, by bisection to full precision: the loop stops when
/// the midpoint no longer lies strictly between the ends.
///
/// For `f(x) = target` with `f` increasing, pass `|x| f(x) < target`.
/// If `below` is not monotone the result is some point where it changes.
///
/// ```
/// use prospicio_math::roots::bisect;
///
/// let r = bisect(0.0, 2.0, |x| x * x < 2.0);
/// assert!((r - 2f64.sqrt()).abs() < 1e-15);
/// ```
pub fn bisect(lo: f64, hi: f64, mut below: impl FnMut(f64) -> bool) -> f64 {
    let Ok((lo, hi)) = try_bisect(lo, hi, Stop::FULL, |x| Ok::<_, Infallible>(below(x)));
    0.5 * (lo + hi)
}

/// When [`try_bisect`] stops: after `max_steps` halvings, or as soon as
/// the bracket is no wider than `absolute`, or than `relative` times the
/// larger of its ends in size. Bisection always stops once the midpoint
/// no longer lies strictly between the ends, whatever the rule.
///
/// Each constructor keeps one rule and sets the others so they never
/// apply; chain [`with_steps`](Self::with_steps) to cap the halvings too.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stop {
    /// The most halvings.
    pub max_steps: usize,
    /// Stop when `hi - lo <= absolute`.
    pub absolute: f64,
    /// Stop when `hi - lo <= relative * max(|lo|, |hi|)`.
    pub relative: f64,
}

impl Stop {
    /// To full precision: until the midpoint is one of the ends (2100
    /// halvings span the whole `f64` range, so the cap never binds).
    pub const FULL: Self = Self {
        max_steps: 2100,
        absolute: 0.0,
        relative: 0.0,
    };

    /// Exactly `n` halvings, unless full precision comes first.
    pub const fn steps(n: usize) -> Self {
        Self {
            max_steps: n,
            ..Self::FULL
        }
    }

    /// Until the bracket is no wider than `tol`.
    pub const fn absolute(tol: f64) -> Self {
        Self {
            absolute: tol,
            ..Self::FULL
        }
    }

    /// Until the bracket is no wider than `tol` times its larger end in
    /// size.
    pub const fn relative(tol: f64) -> Self {
        Self {
            relative: tol,
            ..Self::FULL
        }
    }

    /// This rule with at most `n` halvings.
    pub const fn with_steps(self, n: usize) -> Self {
        Self {
            max_steps: n,
            ..self
        }
    }

    fn done(&self, lo: f64, hi: f64) -> bool {
        let width = hi - lo;
        width <= self.absolute || width <= self.relative * lo.abs().max(hi.abs())
    }
}

/// [`bisect`] with a stop rule and a test that can fail: the final bracket
/// `(lo, hi)` of the boundary in `(lo, hi)` of `below`, which should hold
/// below a point and fail above it. The first error `below` returns stops
/// the search and is returned.
///
/// The bracket, not one point, comes back, so a caller picks the end it
/// needs: the lowest value known to fail is `hi`, the highest known to hold
/// is `lo`. The ends are not checked: `below(lo)` and `below(hi)` are
/// never evaluated.
///
/// ```
/// use prospicio_math::roots::{Stop, try_bisect};
///
/// let (lo, hi) = try_bisect(0.0, 2.0, Stop::absolute(1e-9), |x| {
///     if x.is_finite() { Ok(x * x < 2.0) } else { Err("not finite") }
/// })
/// .unwrap();
/// assert!(hi - lo <= 1e-9 && lo <= 2f64.sqrt() && 2f64.sqrt() <= hi);
/// ```
pub fn try_bisect<E>(
    lo: f64,
    hi: f64,
    stop: Stop,
    mut below: impl FnMut(f64) -> Result<bool, E>,
) -> Result<(f64, f64), E> {
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..stop.max_steps {
        if stop.done(lo, hi) {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if below(mid)? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((lo, hi))
}

/// [`bisect`] on a log scale, for a bracket `0 < lo < hi` that spans orders
/// of magnitude: each step splits at the geometric mean, so relative
/// precision improves at the same rate everywhere in the bracket.
///
/// ```
/// use prospicio_math::roots::bisect_log;
///
/// let r = bisect_log(1e-6, 1e6, |x| x.ln() < 3.0);
/// assert!((r / 3f64.exp() - 1.0).abs() < 1e-14);
/// ```
pub fn bisect_log(lo: f64, hi: f64, mut below: impl FnMut(f64) -> bool) -> f64 {
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..2100 {
        let mid = lo.sqrt() * hi.sqrt();
        if mid <= lo || mid >= hi {
            break;
        }
        if below(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo.sqrt() * hi.sqrt()
}

/// A root of `f` between `x0` and `x1`, where `f0 = f(x0)` and `f1 = f(x1)`
/// have opposite signs, by the Illinois variant of regula falsi.
///
/// Regula falsi keeps a bracket like bisection but steps to where the
/// chord crosses zero; Illinois halves the stale end's value whenever the
/// same end is kept twice, which restores superlinear convergence. On a
/// smooth score it needs a handful of evaluations where bisection needs
/// fifty. Returns an exact zero as soon as one is evaluated.
///
/// ```
/// use prospicio_math::roots::illinois;
///
/// let f = |x: f64| x.exp() - 2.0;
/// let r = illinois(0.0, 1.0, f(0.0), f(1.0), f);
/// assert!((r - 2f64.ln()).abs() < 1e-15);
/// ```
pub fn illinois(x0: f64, x1: f64, f0: f64, f1: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
    let (mut x0, mut x1, mut f0, mut f1) = (x0, x1, f0, f1);
    let positive_left = f0 > 0.0;
    let mut side = 0;
    for _ in 0..200 {
        let x = (x0 * f1 - x1 * f0) / (f1 - f0);
        if !(x > x0.min(x1) && x < x0.max(x1)) || (x1 - x0).abs() <= 1e-15 * x1.abs().max(1.0) {
            break;
        }
        let fx = f(x);
        if fx == 0.0 {
            return x;
        }
        if (fx > 0.0) == positive_left {
            x0 = x;
            f0 = fx;
            if side == -1 {
                f1 *= 0.5;
            }
            side = -1;
        } else {
            x1 = x;
            f1 = fx;
            if side == 1 {
                f0 *= 0.5;
            }
            side = 1;
        }
    }
    0.5 * (x0 + x1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_bisect_stops_by_its_rule() {
        let sqrt2 = 2f64.sqrt();
        let ok = |x: f64| Ok::<_, ()>(x * x < 2.0);
        // Full precision: the same bracket bisect ends on.
        let (lo, hi) = try_bisect(0.0, 2.0, Stop::FULL, ok).unwrap();
        assert_eq!(0.5 * (lo + hi), bisect(0.0, 2.0, |x| x * x < 2.0));
        // A fixed number of steps halves the width exactly that often.
        let (lo, hi) = try_bisect(0.0, 2.0, Stop::steps(10), ok).unwrap();
        assert_eq!(hi - lo, 2.0 / 1024.0);
        assert!(lo <= sqrt2 && sqrt2 <= hi);
        // An absolute and a relative width.
        let (lo, hi) = try_bisect(0.0, 2.0, Stop::absolute(1e-6), ok).unwrap();
        assert!(hi - lo <= 1e-6 && hi - lo > 0.5e-6);
        let (lo, hi) =
            try_bisect(1e3, 1e4, Stop::relative(1e-12), |x| Ok::<_, ()>(x < 5e3)).unwrap();
        assert!(hi - lo <= 1e-12 * hi && hi - lo > 0.5e-12 * hi);
        // The cap binds before the width.
        let (lo, hi) = try_bisect(0.0, 1.0, Stop::absolute(1e-12).with_steps(3), ok).unwrap();
        assert_eq!(hi - lo, 0.125);
        // An error from the test stops the search.
        let mut calls = 0;
        let r = try_bisect(0.0, 2.0, Stop::FULL, |x| {
            calls += 1;
            if calls == 3 { Err(x) } else { Ok(x * x < 2.0) }
        });
        assert_eq!(r, Err(1.25));
        assert_eq!(calls, 3);
    }

    #[test]
    fn bisect_reaches_full_precision() {
        let r = bisect(1.0, 2.0, |x| x * x * x < 3.0);
        let next = f64::from_bits(r.to_bits() + 1);
        let prev = f64::from_bits(r.to_bits() - 1);
        assert!(prev * prev * prev < 3.0 || next * next * next >= 3.0);
        assert!((r - 3f64.cbrt()).abs() <= 2.0 * f64::EPSILON);
    }

    #[test]
    fn bisect_handles_tiny_roots_in_wide_brackets() {
        let r = bisect(1e-12, 1e3, |x| x < 3e-10);
        assert!((r / 3e-10 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn bisect_log_handles_huge_brackets() {
        let r = bisect_log(1e-300, 1e300, |x| x < 7e200);
        assert!((r / 7e200 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn illinois_accepts_either_sign_order() {
        let f = |x: f64| 2.0 - x * x;
        let r = illinois(0.0, 3.0, f(0.0), f(3.0), f);
        assert!((r - 2f64.sqrt()).abs() < 1e-14);
        let g = |x: f64| x * x - 2.0;
        let r = illinois(0.0, 3.0, g(0.0), g(3.0), g);
        assert!((r - 2f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn illinois_is_fast_on_smooth_functions() {
        let mut calls = 0;
        let f = |x: f64| x.powi(5) - 7.0;
        let r = illinois(0.0, 3.0, f(0.0), f(3.0), |x| {
            calls += 1;
            f(x)
        });
        assert!((r - 7f64.powf(0.2)).abs() < 1e-14);
        assert!(calls < 30, "{calls} calls");
    }
}
