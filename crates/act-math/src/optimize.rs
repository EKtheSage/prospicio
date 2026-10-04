//! Optimization in one variable.

/// Minimizes `f` on the open interval `(lo, hi)`: a scan of 64 equal cells
/// picks the best interior grid point, then golden-section search runs
/// between its neighbours to full precision.
///
/// The scan guards against local minima a pure golden-section search
/// would settle in; a minimum narrower than a cell can still be missed.
/// The result never is an end point.
///
/// ```
/// use act_math::optimize::minimize;
///
/// let x = minimize(0.0, 4.0, |x| (x - 1.3).powi(2) + 0.1 * (8.0 * x).cos());
/// let f = |x: f64| (x - 1.3).powi(2) + 0.1 * (8.0 * x).cos();
/// assert!(f(x) <= f(x + 1e-6) && f(x) <= f(x - 1e-6));
/// ```
pub fn minimize(lo: f64, hi: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
    const CELLS: usize = 64;
    let h = (hi - lo) / CELLS as f64;
    let x = |j: usize| lo + h * j as f64;
    let mut best = 1;
    let mut f_best = f(x(1));
    for j in 2..CELLS {
        let v = f(x(j));
        if v.total_cmp(&f_best).is_lt() {
            (best, f_best) = (j, v);
        }
    }
    golden_section(x(best - 1), x(best + 1), f)
}

/// Minimizes a unimodal `f` on `[a, b]` by golden-section search, to the
/// width of `1e-15` relative to `b`.
///
/// ```
/// use act_math::optimize::golden_section;
///
/// let x = golden_section(0.0, 3.0, |x| (x - 2.0).powi(2));
/// assert!((x - 2.0).abs() < 1e-7);
/// ```
pub fn golden_section(a: f64, b: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
    let (mut a, mut b) = (a, b);
    let g = 0.5 * (5f64.sqrt() - 1.0);
    let (mut c, mut d) = (b - g * (b - a), a + g * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..200 {
        if b - a <= 1e-15 * b.abs() {
            break;
        }
        if fc < fd {
            (b, d, fd) = (d, c, fc);
            c = b - g * (b - a);
            fc = f(c);
        } else {
            (a, c, fc) = (c, d, fd);
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

/// Minimizes `f` on `[a, b]` by Brent's method: parabolic interpolation
/// through the three best points when it lands inside the bracket and
/// shrinks it, golden-section steps otherwise (Brent, *Algorithms for
/// Minimization without Derivatives*, 1973, ch. 5). Stops when the bracket
/// is within `tol` (relative, plus a tiny absolute floor) of the best
/// point, and returns `(x, f(x), evaluations)`. Near a minimum `f` is flat
/// to second order, so `x` is only determined to about `√ε` relative, and
/// `tol` below that is raised to it.
///
/// On a smooth unimodal `f` it needs far fewer evaluations than
/// [`golden_section`], which matters when each one is expensive (a model
/// refit).
///
/// ```
/// use act_math::optimize::brent;
///
/// let (x, fx, n) = brent(0.0, 3.0, 1e-10, |x| (x - 2.0).powi(2) + 1.0);
/// assert!((x - 2.0).abs() < 1e-7 && (fx - 1.0).abs() < 1e-14 && n < 20);
/// ```
pub fn brent(a: f64, b: f64, tol: f64, mut f: impl FnMut(f64) -> f64) -> (f64, f64, usize) {
    const GOLD: f64 = 0.381_966_011_250_105_1; // (3 - √5) / 2
    // Below √ε the steps only chase rounding.
    let tol = tol.max(f64::EPSILON.sqrt());
    let (mut a, mut b) = (a.min(b), a.max(b));
    let mut x = a + GOLD * (b - a);
    let (mut w, mut v) = (x, x);
    let mut fx = f(x);
    let (mut fw, mut fv) = (fx, fx);
    let mut evaluations = 1;
    let (mut d, mut e) = (0.0f64, 0.0f64);
    for _ in 0..500 {
        let m = 0.5 * (a + b);
        let tol1 = tol * x.abs() + 1e-12 * tol;
        let tol2 = 2.0 * tol1;
        if (x - m).abs() <= tol2 - 0.5 * (b - a) {
            break;
        }
        let mut golden = true;
        if e.abs() > tol1 {
            // Parabola through (v, fv), (w, fw), (x, fx).
            let r = (x - w) * (fx - fv);
            let q0 = (x - v) * (fx - fw);
            let mut p = (x - v) * q0 - (x - w) * r;
            let mut q = 2.0 * (q0 - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            if p.abs() < (0.5 * q * e).abs() && p > q * (a - x) && p < q * (b - x) {
                e = d;
                d = p / q;
                let u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = if x < m { tol1 } else { -tol1 };
                }
                golden = false;
            }
        }
        if golden {
            e = if x < m { b - x } else { a - x };
            d = GOLD * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = f(u);
        evaluations += 1;
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            (v, fv, w, fw, x, fx) = (w, fw, x, fx, u, fu);
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                (v, fv, w, fw) = (w, fw, u, fu);
            } else if fu <= fv || v == x || v == w {
                (v, fv) = (u, fu);
            }
        }
    }
    (x, fx, evaluations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brent_beats_golden_section_on_smooth_functions() {
        let f = |x: f64| x.exp() - 3.0 * x;
        let (x, _, n) = brent(0.0, 3.0, 1e-12, f);
        assert!((x - 3f64.ln()).abs() < 1e-7, "{x}");
        assert!(n < 20, "{n}");
        // A minimum at the end of the bracket.
        let (x, _, _) = brent(1.0, 2.0, 1e-12, |x| x * x);
        assert!((x - 1.0).abs() < 1e-7);
    }

    #[test]
    fn minimize_escapes_a_local_minimum() {
        // A shallow local minimum near 0.5, the global one near 3.
        let f = |x: f64| (x - 3.0).powi(2) * (x - 0.5).powi(2) - 0.5 * x;
        let x = minimize(0.0, 4.0, f);
        assert!(x > 2.5, "{x}");
    }
}
