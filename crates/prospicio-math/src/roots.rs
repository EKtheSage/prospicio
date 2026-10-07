//! Root finding in one variable.
//!
//! Every solver here takes a bracket and returns a point inside it; none
//! extrapolates. Callers own the bracket, because only they know the
//! domain (an alpha in `[1e-6, 1e6]`, a probability in `(0, 1)`).

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
    let (mut lo, mut hi) = (lo, hi);
    // 2100 halvings span the whole f64 range; the exact stop comes first.
    for _ in 0..2100 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if below(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
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
