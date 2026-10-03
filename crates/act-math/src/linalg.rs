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
/// use act_math::linalg::cholesky;
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
/// use act_math::linalg::{cholesky, cholesky_solve};
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
/// use act_math::linalg::solve;
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn rejects_non_positive_definite() {
        assert!(cholesky(&[1.0, 0.0, 0.0, 0.0], 2).is_none());
        assert!(cholesky(&[1.0, f64::NAN, f64::NAN, 1.0], 2).is_none());
        assert!(cholesky(&[1.0, 0.0, 0.0], 2).is_none());
    }
}
