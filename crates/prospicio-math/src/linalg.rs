//! Dense linear algebra on small row-major matrices.
//!
//! Enough for correlation matrices in dependence models and the small
//! Newton systems of fits; not a general linear algebra library.

/// Cholesky factor of a symmetric positive definite `n × n` matrix `a`
/// (row-major): the lower-triangular `l` with `l lᵀ = a`, row-major with
/// zeros above the diagonal. `None` if `a` is not positive definite (or
/// has the wrong length). Only the lower triangle of `a` is read.
///
/// ```
/// use prospicio_math::linalg::cholesky;
///
/// let l = cholesky(&[4.0, 2.0, 2.0, 5.0], 2).unwrap();
/// assert_eq!(l, [2.0, 0.0, 1.0, 2.0]);
/// assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
/// ```
pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    if a.len() != n * n {
        return None;
    }
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            if i == j {
                let d = a[i * n + i] - dot;
                if d.is_nan() || d <= 0.0 || d.is_infinite() {
                    return None;
                }
                l[i * n + i] = d.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - dot) / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// `y = l x` for a lower-triangular row-major `l` (`n × n`), into `y`.
pub fn lower_mul(l: &[f64], x: &[f64], y: &mut [f64]) {
    let n = x.len();
    for i in 0..n {
        y[i] = (0..=i).map(|k| l[i * n + k] * x[k]).sum();
    }
}

/// Solves `l x = b` for a lower-triangular row-major `l` (`n × n`) with a
/// non-zero diagonal, into `x` (forward substitution).
pub fn lower_solve(l: &[f64], b: &[f64], x: &mut [f64]) {
    let n = b.len();
    for i in 0..n {
        let dot: f64 = (0..i).map(|k| l[i * n + k] * x[k]).sum();
        x[i] = (b[i] - dot) / l[i * n + i];
    }
}

/// Solves `l lᵀ x = b` for the Cholesky factor `l` (`n × n`, row-major)
/// of a symmetric positive definite matrix: forward then back
/// substitution.
///
/// ```
/// use prospicio_math::linalg::{cholesky, cholesky_solve};
///
/// let a = [4.0, 2.0, 2.0, 5.0];
/// let l = cholesky(&a, 2).unwrap();
/// let x = cholesky_solve(&l, &[6.0, 7.0]);
/// assert!((x[0] - 1.0).abs() < 1e-15 && (x[1] - 1.0).abs() < 1e-15);
/// ```
pub fn cholesky_solve(l: &[f64], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = vec![0.0; n];
    lower_solve(l, b, &mut y);
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let dot: f64 = (i + 1..n).map(|k| l[k * n + i] * x[k]).sum();
        x[i] = (y[i] - dot) / l[i * n + i];
    }
    x
}

/// The inverse of `l lᵀ` from its Cholesky factor `l` (`n × n`,
/// row-major), row-major: the covariance of coefficients from the factor
/// of their information matrix.
pub fn cholesky_inverse(l: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0; n * n];
    let mut e = vec![0.0; n];
    for j in 0..n {
        e.fill(0.0);
        e[j] = 1.0;
        let col = cholesky_solve(l, &e);
        for i in 0..n {
            inv[i * n + j] = col[i];
        }
    }
    inv
}

/// Solves the `n × n` system `a x = b` (`a` row-major) by Gaussian
/// elimination with partial pivoting. `None` if `a` is singular, holds a
/// non-finite pivot, or the lengths do not match `n`.
///
/// ```
/// use prospicio_math::linalg::solve;
///
/// let x = solve(vec![2.0, 1.0, 1.0, 3.0], vec![3.0, 5.0], 2).unwrap();
/// assert!((x[0] - 0.8).abs() < 1e-15 && (x[1] - 1.4).abs() < 1e-15);
/// assert!(solve(vec![1.0, 2.0, 2.0, 4.0], vec![1.0, 1.0], 2).is_none());
/// ```
pub fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    if a.len() != n * n || b.len() != n {
        return None;
    }
    for col in 0..n {
        let p = (col..n).max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))?;
        if a[p * n + col] == 0.0 || !a[p * n + col].is_finite() {
            return None;
        }
        for j in 0..n {
            a.swap(col * n + j, p * n + j);
        }
        b.swap(col, p);
        for i in col + 1..n {
            let f = a[i * n + col] / a[col * n + col];
            for j in col..n {
                a[i * n + j] -= f * a[col * n + j];
            }
            b[i] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| a[i * n + j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i * n + i];
    }
    Some(x)
}

/// The product `a b` of two row-major `n × n` matrices.
///
/// Each row of the result is built as a sum over `k` of `a[i, k]` times
/// row `k` of `b`, in increasing `k`, and a zero `a[i, k]` is skipped.
/// For finite `b` the skip changes nothing, because adding `0 × b` to a
/// partial sum that is never `-0` leaves it unchanged.
///
/// ```
/// use prospicio_math::linalg::mat_mul;
///
/// let c = mat_mul(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2);
/// assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
/// ```
pub fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik != 0.0 {
                for j in 0..n {
                    c[i * n + j] += aik * b[k * n + j];
                }
            }
        }
    }
    c
}

/// The inverse of the row-major `n × n` matrix `a`, column by column from
/// [`solve`] on each unit vector. `None` if a solve fails (`a` singular,
/// a non-finite pivot or a wrong length).
///
/// A column can still hold non-finite values or come from an
/// ill-conditioned `a`: the caller decides what to do with those, for
/// example by testing a condition number.
///
/// ```
/// use prospicio_math::linalg::inverse;
///
/// let inv = inverse(&[4.0, 7.0, 2.0, 6.0], 2).unwrap();
/// let want = [0.6, -0.7, -0.2, 0.4];
/// assert!(inv.iter().zip(want).all(|(x, y)| (x - y).abs() < 1e-15));
/// assert!(inverse(&[1.0, 2.0, 2.0, 4.0], 2).is_none());
/// ```
pub fn inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut inv = vec![0.0; n * n];
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let x = solve(a.to_vec(), e, n)?;
        for i in 0..n {
            inv[i * n + j] = x[i];
        }
    }
    Some(inv)
}

/// The log determinant of `l lᵀ` from its Cholesky factor `l` (`n × n`,
/// row-major): `2 Σ ln l[i, i]`.
///
/// ```
/// use prospicio_math::linalg::{cholesky, cholesky_log_det};
///
/// let l = cholesky(&[4.0, 2.0, 2.0, 5.0], 2).unwrap();
/// assert!((cholesky_log_det(&l, 2) - 16f64.ln()).abs() < 1e-15);
/// ```
pub fn cholesky_log_det(l: &[f64], n: usize) -> f64 {
    2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>()
}

/// The least-squares line through some points: [`line_fit`]'s result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineFit {
    /// The fitted value at `x = 0`.
    pub intercept: f64,
    /// The change in `y` per unit of `x`.
    pub slope: f64,
    /// The coefficient of determination, `sxy² / (sxx syy)`; 1 when every
    /// `y` is equal (or with two points).
    pub r_squared: f64,
    /// The number of points fitted.
    pub n: usize,
}

/// The ordinary least-squares line `y = intercept + slope x` through
/// `points` (`(x, y)` pairs), from the centred sums `sxx`, `sxy` and `syy`
/// about the means. `None` with fewer than two points or when every `x`
/// is equal. Non-finite values are not filtered: the caller decides which
/// points to fit.
///
/// ```
/// use prospicio_math::linalg::line_fit;
///
/// let fit = line_fit(&[(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)]).unwrap();
/// assert_eq!((fit.intercept, fit.slope, fit.r_squared, fit.n), (1.0, 2.0, 1.0, 3));
/// assert!(line_fit(&[(1.0, 1.0), (1.0, 2.0)]).is_none());
/// ```
pub fn line_fit(points: &[(f64, f64)]) -> Option<LineFit> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.0 - mean_x)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let sxy: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
    let syy: f64 = points.iter().map(|p| (p.1 - mean_y) * (p.1 - mean_y)).sum();
    let slope = sxy / sxx;
    let r_squared = if syy > 0.0 {
        sxy * sxy / (sxx * syy)
    } else {
        1.0
    };
    Some(LineFit {
        intercept: mean_y - slope * mean_x,
        slope,
        r_squared,
        n: points.len(),
    })
}

/// The eigenvalues of the symmetric `n × n` matrix `a` (row-major; only
/// its lower triangle is read), in increasing order, by cyclic Jacobi
/// rotations. Accurate to a few ulps of the largest eigenvalue, which
/// suits the small penalty matrices of smoothing splines (`n` up to a few
/// hundred).
///
/// ```
/// use prospicio_math::linalg::symmetric_eigenvalues;
///
/// let e = symmetric_eigenvalues(&[2.0, 1.0, 1.0, 2.0], 2);
/// assert!((e[0] - 1.0).abs() < 1e-15 && (e[1] - 3.0).abs() < 1e-15);
/// ```
pub fn symmetric_eigenvalues(a: &[f64], n: usize) -> Vec<f64> {
    let (m, _) = jacobi(a, n, false);
    let mut e: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    e.sort_by(f64::total_cmp);
    e
}

/// The eigenvalues of the symmetric `n × n` matrix `a` (row-major; only
/// its lower triangle is read) in increasing order, and their unit
/// eigenvectors as the columns of a row-major `n × n` matrix, column `j`
/// for value `j`, by the cyclic Jacobi rotations of
/// [`symmetric_eigenvalues`] (the same values). Each vector's sign is set
/// so its largest entry in size is positive.
///
/// ```
/// use prospicio_math::linalg::symmetric_eigen;
///
/// let (e, v) = symmetric_eigen(&[2.0, 1.0, 1.0, 2.0], 2);
/// assert!((e[1] - 3.0).abs() < 1e-15);
/// let h = 0.5f64.sqrt();
/// assert!((v[1] - h).abs() < 1e-15 && (v[3] - h).abs() < 1e-15);
/// ```
pub fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let (m, v) = jacobi(a, n, true);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| m[i * n + i].total_cmp(&m[j * n + j]));
    let values = order.iter().map(|&i| m[i * n + i]).collect();
    let mut vectors = vec![0.0; n * n];
    for (col, &i) in order.iter().enumerate() {
        let big = (0..n)
            .map(|k| v[k * n + i])
            .fold(0.0_f64, |b, x| if x.abs() > b.abs() { x } else { b });
        let sign = if big < 0.0 { -1.0 } else { 1.0 };
        for k in 0..n {
            vectors[k * n + col] = sign * v[k * n + i];
        }
    }
    (values, vectors)
}

/// Cyclic Jacobi rotations of the symmetric matrix `a` to diagonal form,
/// and, with `vectors`, the product of the rotations (the eigenvectors as
/// columns; the identity otherwise, untouched).
fn jacobi(a: &[f64], n: usize, vectors: bool) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), n * n, "a must be n × n");
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            m[i * n + j] = a[i * n + j];
            m[j * n + i] = a[i * n + j];
        }
    }
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let scale: f64 = m.iter().map(|v| v * v).sum::<f64>().sqrt();
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..i).map(move |j| (i, j)))
            .map(|(i, j)| m[i * n + j] * m[i * n + j])
            .sum::<f64>()
            .sqrt();
        if off <= 1e-15 * scale || off == 0.0 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let (app, aqq) = (m[p * n + p], m[q * n + q]);
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                if vectors {
                    for k in 0..n {
                        let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                        v[k * n + p] = c * vkp - s * vkq;
                        v[k * n + q] = s * vkp + c * vkq;
                    }
                }
            }
        }
    }
    (m, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_eigen_reconstructs_the_matrix() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, -0.2, 0.5, -0.2, 1.0];
        let (e, v) = symmetric_eigen(&a, 3);
        assert_eq!(e, symmetric_eigenvalues(&a, 3));
        for i in 0..3 {
            for j in 0..3 {
                let back: f64 = (0..3).map(|k| v[i * 3 + k] * e[k] * v[j * 3 + k]).sum();
                assert!((back - a[i * 3 + j]).abs() < 1e-13, "{i} {j}");
                let dot: f64 = (0..3).map(|k| v[k * 3 + i] * v[k * 3 + j]).sum();
                assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn line_fit_on_noisy_points() {
        let pts = [(1.0, 2.1), (2.0, 3.9), (3.0, 6.2), (4.0, 7.8)];
        let f = line_fit(&pts).unwrap();
        // numpy.polyfit(x, y, 1) and scipy.stats.linregress
        assert!((f.slope - 1.94).abs() < 1e-12);
        assert!((f.intercept - 0.15).abs() < 1e-12);
        assert!((f.r_squared - 0.995_661_375_661_375_5).abs() < 1e-12);
        assert!(line_fit(&pts[..1]).is_none());
        let flat = line_fit(&[(0.0, 5.0), (1.0, 5.0)]).unwrap();
        assert_eq!((flat.slope, flat.r_squared), (0.0, 1.0));
    }

    #[test]
    fn mat_mul_matches_the_plain_triple_loop() {
        let a = [1.5, 0.0, -2.0, 0.25, 3.0, 0.0, 0.0, -1.0, 4.0];
        let b = [2.0, -0.5, 1.0, 0.0, 7.0, 3.0, -1.25, 2.0, 0.5];
        let mut want = vec![0.0; 9];
        for i in 0..3 {
            for k in 0..3 {
                for j in 0..3 {
                    want[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
                }
            }
        }
        assert_eq!(mat_mul(&a, &b, 3), want);
    }

    #[test]
    fn inverse_inverts_and_refuses_singular() {
        let a = [2.0, 1.0, 0.5, 1.0, 3.0, -1.0, 0.0, 2.0, 4.0];
        let inv = inverse(&a, 3).unwrap();
        let id = mat_mul(&a, &inv, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((id[i * 3 + j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-14);
            }
        }
        assert!(inverse(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 1.0, 1.0], 3).is_none());
        assert!(inverse(&[1.0, 2.0], 2).is_none());
    }

    #[test]
    fn cholesky_log_det_is_twice_the_half() {
        let a = [4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];
        let l = cholesky(&a, 3).unwrap();
        let half: f64 = (0..3).map(|i| l[i * 3 + i].ln()).sum();
        assert_eq!(cholesky_log_det(&l, 3), 2.0 * half);
        let det = 4.0 * (5.0 * 3.0 - 1.0) - 2.0 * (2.0 * 3.0 - 0.6) + 0.6 * (2.0 - 5.0 * 0.6);
        assert!((cholesky_log_det(&l, 3) - f64::ln(det)).abs() < 1e-14);
    }

    #[test]
    fn cholesky_inverse_inverts() {
        let a = [4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];
        let l = cholesky(&a, 3).unwrap();
        let inv = cholesky_inverse(&l, 3);
        for i in 0..3 {
            for j in 0..3 {
                let v: f64 = (0..3).map(|k| a[i * 3 + k] * inv[k * 3 + j]).sum();
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn solve_pivots_past_a_zero_diagonal() {
        let x = solve(vec![0.0, 1.0, 1.0, 0.0], vec![2.0, 3.0], 2).unwrap();
        assert_eq!(x, [3.0, 2.0]);
    }

    #[test]
    fn factor_reproduces_the_matrix() {
        let a = [
            1.0, 0.5, 0.2, 0.1, //
            0.5, 1.0, 0.3, 0.0, //
            0.2, 0.3, 1.0, -0.4, //
            0.1, 0.0, -0.4, 1.0,
        ];
        let n = 4;
        let l = cholesky(&a, n).unwrap();
        for i in 0..n {
            for j in 0..n {
                let v: f64 = (0..n).map(|k| l[i * n + k] * l[j * n + k]).sum();
                assert!((v - a[i * n + j]).abs() < 1e-15, "({i}, {j})");
                if j > i {
                    assert_eq!(l[i * n + j], 0.0);
                }
            }
        }
        let mut y = [0.0; 4];
        lower_mul(&l, &[1.0, 0.0, 0.0, 0.0], &mut y);
        assert_eq!(y, [l[0], l[4], l[8], l[12]]);
        let x = [0.3, -1.0, 2.0, 0.5];
        lower_mul(&l, &x, &mut y);
        let mut back = [0.0; 4];
        lower_solve(&l, &y, &mut back);
        for (a, b) in back.iter().zip(&x) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn symmetric_eigenvalues_of_a_difference_penalty() {
        // The second-difference penalty DᵀD on 6 points: rank 4, trace 20,
        // eigenvalues of a known matrix match its trace and determinant
        // structure; check against the characteristic quantities.
        let n = 6;
        let mut d = vec![0.0; (n - 2) * n];
        for r in 0..n - 2 {
            d[r * n + r] = 1.0;
            d[r * n + r + 1] = -2.0;
            d[r * n + r + 2] = 1.0;
        }
        let mut p = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                p[i * n + j] = (0..n - 2).map(|r| d[r * n + i] * d[r * n + j]).sum();
            }
        }
        let e = symmetric_eigenvalues(&p, n);
        assert!(e[0].abs() < 1e-13 && e[1].abs() < 1e-13, "{e:?}");
        let trace: f64 = (0..n).map(|i| p[i * n + i]).sum();
        assert!((e.iter().sum::<f64>() - trace).abs() < 1e-12);
        // Each eigenvalue λ has a vector: det(P − λI) = 0, checked by the
        // Frobenius norm identity Σλ² = ‖P‖².
        let fro: f64 = p.iter().map(|v| v * v).sum();
        assert!((e.iter().map(|v| v * v).sum::<f64>() - fro).abs() < 1e-11);
    }

    #[test]
    fn rejects_non_positive_definite() {
        assert!(cholesky(&[1.0, 0.0, 0.0, 0.0], 2).is_none());
        assert!(cholesky(&[1.0, f64::NAN, f64::NAN, 1.0], 2).is_none());
        assert!(cholesky(&[1.0, 0.0, 0.0], 2).is_none());
    }
}
