//! B-spline bases and difference penalties, for P-splines.

/// Knots for a B-spline basis of `n_basis` functions of degree `degree`
/// over `[lo, hi]`: equally spaced, extended `degree` intervals beyond each
/// end, `n_basis + degree + 1` in all.
///
/// The range is widened by 0.1% on each side first, as mgcv's `"ps"` basis
/// does, so data at the ends sit strictly inside it.
pub fn pspline_knots(lo: f64, hi: f64, n_basis: usize, degree: usize) -> Vec<f64> {
    let pad = (hi - lo) * 0.001;
    let (lo, hi) = (lo - pad, hi + pad);
    let intervals = n_basis - degree;
    let dx = (hi - lo) / intervals as f64;
    (0..n_basis + degree + 1)
        .map(|i| lo + dx * (i as f64 - degree as f64))
        .collect()
}

/// The values at `x` of the `knots.len() - degree - 1` B-splines of degree
/// `degree` on `knots`, by the Cox–de Boor recursion. Zero outside the
/// knots' span; inside the inner span the values sum to 1.
///
/// ```
/// use prospicio_math::spline::{bspline_basis, pspline_knots};
///
/// let knots = pspline_knots(0.0, 1.0, 8, 3);
/// let b = bspline_basis(0.37, &knots, 3);
/// assert_eq!(b.len(), 8);
/// assert!((b.iter().sum::<f64>() - 1.0).abs() < 1e-14);
/// ```
pub fn bspline_basis(x: f64, knots: &[f64], degree: usize) -> Vec<f64> {
    let m = knots.len();
    // Degree 0: the indicator of each knot interval.
    let mut b: Vec<f64> = (0..m - 1)
        .map(|i| f64::from(u8::from(knots[i] <= x && x < knots[i + 1])))
        .collect();
    for d in 1..=degree {
        let next: Vec<f64> = (0..m - 1 - d)
            .map(|i| {
                let left = {
                    let w = knots[i + d] - knots[i];
                    if w > 0.0 {
                        (x - knots[i]) / w * b[i]
                    } else {
                        0.0
                    }
                };
                let right = {
                    let w = knots[i + d + 1] - knots[i + 1];
                    if w > 0.0 {
                        (knots[i + d + 1] - x) / w * b[i + 1]
                    } else {
                        0.0
                    }
                };
                left + right
            })
            .collect();
        b = next;
    }
    b
}

/// `Dᵀ D` for the `order`-th difference matrix `D` on `k` coefficients,
/// row-major `k × k`: the P-spline penalty, which leaves polynomials of
/// degree below `order` in the coefficients unpenalized.
///
/// ```
/// use prospicio_math::spline::difference_penalty;
///
/// // First differences on 3 coefficients: D = [[-1, 1, 0], [0, -1, 1]].
/// assert_eq!(
///     difference_penalty(3, 1),
///     [1.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 1.0]
/// );
/// ```
pub fn difference_penalty(k: usize, order: usize) -> Vec<f64> {
    // Rows of D: binomial coefficients with alternating signs.
    let mut row = vec![1.0];
    for _ in 0..order {
        let mut next = vec![0.0; row.len() + 1];
        for (i, v) in row.iter().enumerate() {
            next[i] -= v;
            next[i + 1] += v;
        }
        row = next;
    }
    let mut p = vec![0.0; k * k];
    for start in 0..k.saturating_sub(order) {
        for (a, va) in row.iter().enumerate() {
            for (b, vb) in row.iter().enumerate() {
                p[(start + a) * k + start + b] += va * vb;
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_is_a_partition_of_unity_with_local_support() {
        let knots = pspline_knots(-2.0, 5.0, 10, 3);
        for &x in &[-2.0, -1.3, 0.0, 2.2, 4.999, 5.0] {
            let b = bspline_basis(x, &knots, 3);
            assert!((b.iter().sum::<f64>() - 1.0).abs() < 1e-13, "{x}");
            assert!(b.iter().all(|&v| v >= 0.0));
            assert!(b.iter().filter(|&&v| v > 0.0).count() <= 4);
        }
    }

    #[test]
    fn penalty_ignores_low_order_polynomials() {
        let p = difference_penalty(6, 2);
        let linear: Vec<f64> = (0..6).map(|i| 2.0 + 0.5 * f64::from(i)).collect();
        for r in 0..6 {
            let v: f64 = (0..6).map(|c| p[r * 6 + c] * linear[c]).sum();
            assert!(v.abs() < 1e-12);
        }
    }
}
