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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimize_escapes_a_local_minimum() {
        // A shallow local minimum near 0.5, the global one near 3.
        let f = |x: f64| (x - 3.0).powi(2) * (x - 0.5).powi(2) - 0.5 * x;
        let x = minimize(0.0, 4.0, f);
        assert!(x > 2.5, "{x}");
    }
}
