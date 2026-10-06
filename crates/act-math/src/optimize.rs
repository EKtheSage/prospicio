//! Optimization: one-variable searches and the Nelder–Mead simplex.

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

/// Options for [`nelder_mead`].
///
/// `initial_step` sizes the starting simplex: each coordinate of the
/// start point is moved by `initial_step` times its absolute value (a
/// relative step), or by `initial_step` itself when the coordinate is zero
/// (an absolute step). The search has converged when the spread of the
/// simplex's values is within `tolerance` times `max(1, |best value|)` and
/// every vertex is within `tolerance` times `max(1, |best x|)` of the best
/// one in each coordinate. `max_iterations` counts simplex moves over both
/// passes (see [`nelder_mead`]).
///
/// ```
/// use act_math::optimize::NelderMead;
///
/// let options = NelderMead { tolerance: 1e-12, ..NelderMead::default() };
/// assert_eq!(options.initial_step, 0.1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NelderMead {
    /// Size of the starting simplex, relative to each nonzero coordinate.
    pub initial_step: f64,
    /// Convergence tolerance on both the values and the simplex diameter.
    pub tolerance: f64,
    /// Most simplex moves allowed in total.
    pub max_iterations: usize,
}

impl Default for NelderMead {
    /// A 10% starting step, tolerance `1e-10` and 10 000 iterations.
    fn default() -> Self {
        NelderMead {
            initial_step: 0.1,
            tolerance: 1e-10,
            max_iterations: 10_000,
        }
    }
}

/// The result of [`nelder_mead`]: the best point found, its value, the
/// number of simplex moves made and whether the tolerance was met.
///
/// ```
/// use act_math::optimize::{nelder_mead, NelderMead};
///
/// let m = nelder_mead(|x| (x[0] - 3.0).powi(2), &[0.0], &NelderMead::default());
/// assert!(m.converged && (m.x[0] - 3.0).abs() < 1e-8 && m.value < 1e-16);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Minimum {
    /// The best point found.
    pub x: Vec<f64>,
    /// `f(x)`, or `+inf` if no feasible point was found.
    pub value: f64,
    /// Simplex moves made.
    pub iterations: usize,
    /// Whether the last pass met the tolerance within `max_iterations`.
    pub converged: bool,
}

/// Minimizes `f` from the start point `x0` by the Nelder–Mead simplex
/// method, with the standard moves: reflection 1, expansion 2, contraction
/// 1/2 (outside or inside) and shrink 1/2 toward the best vertex (Nelder
/// and Mead, 1965; Lagarias et al., 1998).
///
/// A NaN or infinite value of `f` counts as `+inf`, so `f` can reject an
/// infeasible point by returning NaN or infinity and the simplex moves
/// away from it. Once the tolerance is met the search restarts once from
/// the best point with a fresh simplex, which recovers from a simplex that
/// collapsed onto a subspace; the result reports that second pass. No
/// derivatives are used, so `f` need not be smooth, but the method only
/// finds a local minimum.
///
/// ```
/// use act_math::optimize::{nelder_mead, NelderMead};
///
/// // Rosenbrock's banana-shaped valley, minimum 0 at (1, 1).
/// let rosenbrock = |x: &[f64]| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2);
/// let m = nelder_mead(rosenbrock, &[-1.2, 1.0], &NelderMead::default());
/// assert!(m.converged);
/// assert!((m.x[0] - 1.0).abs() < 1e-6 && (m.x[1] - 1.0).abs() < 1e-6);
/// ```
pub fn nelder_mead(mut f: impl FnMut(&[f64]) -> f64, x0: &[f64], options: &NelderMead) -> Minimum {
    let mut eval = |x: &[f64]| {
        let v = f(x);
        if v.is_finite() { v } else { f64::INFINITY }
    };
    let mut x = x0.to_vec();
    let mut value = eval(&x);
    let mut iterations = 0;
    let mut converged = x0.is_empty();
    if !converged {
        for _pass in 0..2 {
            (x, value, converged) = simplex_pass(&mut eval, x, value, options, &mut iterations);
            if !converged {
                break;
            }
        }
    }
    Minimum {
        x,
        value,
        iterations,
        converged,
    }
}

/// One Nelder–Mead run from a fresh simplex around `start`. Returns the
/// best vertex, its value and whether the tolerance was met.
fn simplex_pass(
    f: &mut impl FnMut(&[f64]) -> f64,
    start: Vec<f64>,
    f_start: f64,
    options: &NelderMead,
    iterations: &mut usize,
) -> (Vec<f64>, f64, bool) {
    let n = start.len();
    let tol = options.tolerance;
    let mut simplex = vec![(start.clone(), f_start)];
    for i in 0..n {
        let mut v = start.clone();
        v[i] += if v[i] == 0.0 {
            options.initial_step
        } else {
            options.initial_step * v[i].abs()
        };
        let fv = f(&v);
        simplex.push((v, fv));
    }
    // The point `c + t (p - c)` on the line through `c` and `p`.
    let along = |c: &[f64], p: &[f64], t: f64| -> Vec<f64> {
        c.iter().zip(p).map(|(c, p)| c + t * (p - c)).collect()
    };
    loop {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (x_best, f_best) = (&simplex[0].0, simplex[0].1);
        let spread = simplex[n].1 - f_best;
        let scale = x_best.iter().fold(1.0f64, |m, x| m.max(x.abs()));
        let diameter = simplex[1..]
            .iter()
            .flat_map(|(v, _)| v.iter().zip(x_best).map(|(a, b)| (a - b).abs()))
            .fold(0.0f64, f64::max);
        // `spread` is NaN, so never small, while every value is infinite.
        if spread <= tol * f_best.abs().max(1.0) && diameter <= tol * scale {
            return (x_best.clone(), f_best, true);
        }
        if *iterations >= options.max_iterations {
            return (x_best.clone(), f_best, false);
        }
        *iterations += 1;

        let mut centroid = vec![0.0; n];
        for (v, _) in &simplex[..n] {
            for (c, x) in centroid.iter_mut().zip(v) {
                *c += x / n as f64;
            }
        }
        let (f_second, f_worst) = (simplex[n - 1].1, simplex[n].1);
        let xr = along(&centroid, &simplex[n].0, -1.0);
        let fr = f(&xr);
        if fr < f_best {
            let xe = along(&centroid, &simplex[n].0, -2.0);
            let fe = f(&xe);
            simplex[n] = if fe < fr { (xe, fe) } else { (xr, fr) };
            continue;
        }
        if fr < f_second {
            simplex[n] = (xr, fr);
            continue;
        }
        // Contract outside, toward the reflected point, or inside.
        let (xc, accept) = if fr < f_worst {
            let xc = along(&centroid, &xr, 0.5);
            let fc = f(&xc);
            ((xc, fc), fc <= fr)
        } else {
            let xc = along(&centroid, &simplex[n].0, 0.5);
            let fc = f(&xc);
            ((xc, fc), fc < f_worst)
        };
        if accept {
            simplex[n] = xc;
            continue;
        }
        let x_best = simplex[0].0.clone();
        for (v, fv) in &mut simplex[1..] {
            *v = along(&x_best, v, 0.5);
            *fv = f(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(x: &[f64]) -> f64 {
        100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
    }

    #[test]
    fn nelder_mead_finds_the_bottom_of_a_quadratic_bowl() {
        // A positive definite quadratic with minimum 2 at (1, -2, 0.5),
        // by construction.
        let f = |x: &[f64]| {
            let (a, b, c) = (x[0] - 1.0, x[1] + 2.0, x[2] - 0.5);
            a * a + 4.0 * b * b + 0.5 * c * c + 0.3 * a * b + 2.0
        };
        let m = nelder_mead(f, &[0.0, 0.0, 0.0], &NelderMead::default());
        assert!(m.converged, "{m:?}");
        for (x, want) in m.x.iter().zip([1.0, -2.0, 0.5]) {
            assert!((x - want).abs() < 1e-7, "{m:?}");
        }
        assert!((m.value - 2.0).abs() < 1e-12, "{m:?}");
    }

    #[test]
    fn nelder_mead_solves_rosenbrock() {
        // Rosenbrock (1960) from the standard start (-1.2, 1); minimum 0 at (1, 1).
        let m = nelder_mead(rosenbrock, &[-1.2, 1.0], &NelderMead::default());
        assert!(m.converged, "{m:?}");
        assert!(
            (m.x[0] - 1.0).abs() < 1e-6 && (m.x[1] - 1.0).abs() < 1e-6,
            "{m:?}"
        );
        assert!(m.value < 1e-12, "{m:?}");
    }

    #[test]
    fn nelder_mead_works_in_one_dimension() {
        // exp(x) - 3x has its minimum at ln 3.
        let m = nelder_mead(|x| x[0].exp() - 3.0 * x[0], &[5.0], &NelderMead::default());
        assert!(m.converged, "{m:?}");
        assert!((m.x[0] - 3f64.ln()).abs() < 1e-7, "{m:?}");
    }

    #[test]
    fn nelder_mead_steps_around_a_nan_region() {
        // NaN for x <= 0; the minimum at (0.05, 1) sits next to that
        // region, and a step of the starting simplex lands in it.
        let mut infeasible_calls = 0;
        let f = |x: &[f64]| {
            if x[0] <= 0.0 {
                infeasible_calls += 1;
                return f64::NAN;
            }
            (x[0] - 0.05).powi(2) + (x[1] - 1.0).powi(2)
        };
        let m = nelder_mead(f, &[0.5, 0.0], &NelderMead::default());
        assert!(m.converged, "{m:?}");
        assert!(
            (m.x[0] - 0.05).abs() < 1e-7 && (m.x[1] - 1.0).abs() < 1e-7,
            "{m:?}"
        );
        assert!(infeasible_calls > 0, "the test never probed the NaN region");
    }

    #[test]
    fn nelder_mead_reports_an_infeasible_start() {
        let m = nelder_mead(|_| f64::NAN, &[1.0, 1.0], &NelderMead::default());
        assert!(!m.converged && m.value == f64::INFINITY, "{m:?}");
    }

    #[test]
    fn nelder_mead_stops_at_max_iterations() {
        let options = NelderMead {
            max_iterations: 10,
            ..NelderMead::default()
        };
        let m = nelder_mead(rosenbrock, &[-1.2, 1.0], &options);
        assert!(!m.converged && m.iterations == 10, "{m:?}");
        assert!(m.value <= rosenbrock(&[-1.2, 1.0]), "{m:?}");
    }

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
