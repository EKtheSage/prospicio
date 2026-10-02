//! Dense linear algebra on small row-major matrices.
//!
//! Enough for correlation matrices in dependence models; not a general
//! linear algebra library.

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

#[cfg(test)]
mod tests {
    use super::*;

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
