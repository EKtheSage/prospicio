//! The multivariate normal distribution function by Genz's method.
//!
//! [`mvn_probability`] gives `P(a < X < b)` for `X ~ N(0, R)`, `R` a
//! correlation (or covariance) matrix, and [`mvn_cdf`] its case
//! `a = −∞`. One and two dimensions are exact (`Φ`, and Genz's BVND to about
//! `1e-15`). In more, the probability is Genz's (1992) integral over the
//! unit cube in `d − 1` dimensions after the variables are separated by a
//! Cholesky factor, with the variables reordered as Genz and Bretz (2009,
//! section 4.1.3) do so that the most constrained come first. The integral
//! is estimated by randomized quasi-Monte Carlo: `scrambles` independently
//! scrambled Sobol sequences ([`crate::qmc`]), each over the same doubling
//! number of points, their mean the estimate and three standard errors of it
//! the error, which is doubled until it is at most `abs_tol` or the points
//! reach `max_points`. A seed fixes the scrambles, so a call is reproducible.
//! This is the method of R's `mvtnorm::pmvnorm` (`GenzBretz`) and SciPy's
//! `multivariate_normal.cdf`, with Sobol points in place of their lattice
//! rules, so the results agree within the errors and not digit for digit.

use crate::linalg::cholesky;
use crate::qmc::{Scramble, Sobol, SplitMix};
use crate::special::{bivariate_norm_cdf, norm_cdf, norm_pdf, norm_quantile};

/// Settings of [`mvn_probability`] and [`mvn_cdf`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MvnSettings {
    /// Absolute error to reach, three standard errors of the estimate.
    pub abs_tol: f64,
    /// Most integrand evaluations over all scrambles.
    pub max_points: usize,
    /// Independent scrambles (at least 2), for the error estimate.
    pub scrambles: usize,
    /// Seed of the scrambles.
    pub seed: u64,
}

impl Default for MvnSettings {
    /// `abs_tol` `1e-5`, `max_points` `2^22`, ten scrambles, seed 0.
    fn default() -> Self {
        MvnSettings {
            abs_tol: 1e-5,
            max_points: 1 << 22,
            scrambles: 10,
            seed: 0,
        }
    }
}

/// An estimate of a multivariate normal probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MvnEstimate {
    /// The probability, in `[0, 1]`.
    pub value: f64,
    /// Three standard errors of `value` over the scrambles; 0 when exact.
    /// With the default ten scrambles the standard error is itself an
    /// estimate on 9 degrees of freedom, so the true error exceeds this in
    /// about 1.5% of calls.
    pub error: f64,
    /// Integrand evaluations used.
    pub points: usize,
    /// Whether `error` reached `abs_tol` within `max_points`.
    pub converged: bool,
}

impl MvnEstimate {
    fn exact(value: f64) -> Self {
        MvnEstimate {
            value: value.clamp(0.0, 1.0),
            error: 0.0,
            points: 0,
            converged: true,
        }
    }
}

/// `P(X < b)` for `X ~ N(0, R)`; see [`mvn_probability`].
///
/// ```
/// use prospicio_math::mvn::{mvn_cdf, MvnSettings};
///
/// // Three equicorrelated normals at ρ = ½: P(all < 0) = 1/4.
/// let r = [1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0];
/// let p = mvn_cdf(&[0.0; 3], &r, MvnSettings::default()).unwrap();
/// assert!((p.value - 0.25).abs() <= p.error.max(1e-12));
/// assert!(p.error < 1e-5);
/// ```
pub fn mvn_cdf(upper: &[f64], r: &[f64], settings: MvnSettings) -> Result<MvnEstimate, String> {
    let lower = vec![f64::NEG_INFINITY; upper.len()];
    mvn_probability(&lower, upper, r, settings)
}

/// `P(a < X < b)` for `X ~ N(0, R)`, `R` a positive definite `d × d`
/// matrix, row-major. Limits may be infinite; a coordinate with
/// `a_i = −∞` and `b_i = ∞` drops out. Exact in one and two dimensions.
///
/// A matrix that is not square, symmetric and positive definite, limits of
/// the wrong length, a NaN, `a_i > b_i` or bad settings are errors.
pub fn mvn_probability(
    lower: &[f64],
    upper: &[f64],
    r: &[f64],
    settings: MvnSettings,
) -> Result<MvnEstimate, String> {
    let n = upper.len();
    if lower.len() != n || r.len() != n * n {
        return Err(format!(
            "{} lower and {} upper limits need a {}-entry matrix, not {}",
            lower.len(),
            n,
            n * n,
            r.len()
        ));
    }
    if settings.abs_tol.is_nan()
        || settings.abs_tol <= 0.0
        || settings.scrambles < 2
        || settings.max_points == 0
    {
        return Err(
            "abs_tol must be positive, scrambles at least 2 and max_points positive".into(),
        );
    }
    for i in 0..n {
        if lower[i].is_nan() || upper[i].is_nan() || lower[i] > upper[i] {
            return Err(format!(
                "limits {} and {} of coordinate {i}",
                lower[i], upper[i]
            ));
        }
        for j in 0..n {
            let (x, y) = (r[i * n + j], r[j * n + i]);
            if !x.is_finite() || (x - y).abs() > 1e-12 * (1.0 + x.abs()) {
                return Err("the matrix must be finite and symmetric".into());
            }
        }
    }
    if cholesky(r, n).is_none() {
        return Err("the matrix is not positive definite".into());
    }
    // Standardize, and drop the coordinates that are not constrained.
    let keep: Vec<usize> = (0..n)
        .filter(|&i| lower[i] > f64::NEG_INFINITY || upper[i] < f64::INFINITY)
        .collect();
    let d = keep.len();
    let sd: Vec<f64> = keep.iter().map(|&i| r[i * n + i].sqrt()).collect();
    let a: Vec<f64> = keep.iter().zip(&sd).map(|(&i, s)| lower[i] / s).collect();
    let b: Vec<f64> = keep.iter().zip(&sd).map(|(&i, s)| upper[i] / s).collect();
    let mut c = vec![0.0; d * d];
    for (p, &i) in keep.iter().enumerate() {
        for (q, &j) in keep.iter().enumerate() {
            c[p * d + q] = r[i * n + j] / (sd[p] * sd[q]);
        }
    }
    match d {
        0 => return Ok(MvnEstimate::exact(1.0)),
        1 => return Ok(MvnEstimate::exact(norm_cdf(b[0]) - norm_cdf(a[0]))),
        2 => {
            let rho = c[1];
            let f = |x: f64, y: f64| -> f64 {
                if x == f64::NEG_INFINITY || y == f64::NEG_INFINITY {
                    0.0
                } else if x == f64::INFINITY {
                    norm_cdf(y)
                } else if y == f64::INFINITY {
                    norm_cdf(x)
                } else {
                    bivariate_norm_cdf(x, y, rho)
                }
            };
            let p = f(b[0], b[1]) - f(a[0], b[1]) - f(b[0], a[1]) + f(a[0], a[1]);
            return Ok(MvnEstimate::exact(p));
        }
        _ => {}
    }
    let (l, a, b) = reorder(&c, a, b, d)?;
    if let Some(p) = degenerate(&l, &a, &b, d) {
        return Ok(MvnEstimate::exact(p));
    }
    Ok(integrate(&l, &a, &b, d, settings))
}

/// The Cholesky factor of the reordered matrix and the reordered limits.
#[allow(clippy::type_complexity)]
fn reorder(
    c: &[f64],
    mut a: Vec<f64>,
    mut b: Vec<f64>,
    d: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let mut s = c.to_vec();
    let mut l = vec![0.0; d * d];
    let mut y = vec![0.0; d];
    for i in 0..d {
        // The remaining variable with the smallest conditional probability.
        let mut best = (i, f64::INFINITY);
        for j in i..d {
            let shift: f64 = (0..i).map(|k| l[j * d + k] * y[k]).sum();
            let var = s[j * d + j] - (0..i).map(|k| l[j * d + k].powi(2)).sum::<f64>();
            let den = var.max(0.0).sqrt();
            let prob = if den > 0.0 {
                norm_cdf((b[j] - shift) / den) - norm_cdf((a[j] - shift) / den)
            } else {
                0.0
            };
            if prob < best.1 {
                best = (j, prob);
            }
        }
        let j = best.0;
        if j != i {
            for k in 0..d {
                s.swap(i * d + k, j * d + k);
            }
            for k in 0..d {
                s.swap(k * d + i, k * d + j);
            }
            for k in 0..i {
                l.swap(i * d + k, j * d + k);
            }
            a.swap(i, j);
            b.swap(i, j);
        }
        let var = s[i * d + i] - (0..i).map(|k| l[i * d + k].powi(2)).sum::<f64>();
        if var <= 1e-14 {
            return Err("the matrix is not positive definite".into());
        }
        let lii = var.sqrt();
        l[i * d + i] = lii;
        for row in i + 1..d {
            let dot: f64 = (0..i).map(|k| l[row * d + k] * l[i * d + k]).sum();
            l[row * d + i] = (s[row * d + i] - dot) / lii;
        }
        let shift: f64 = (0..i).map(|k| l[i * d + k] * y[k]).sum();
        let (lo, hi) = ((a[i] - shift) / lii, (b[i] - shift) / lii);
        let mass = norm_cdf(hi) - norm_cdf(lo);
        y[i] = if mass > 1e-300 {
            (pdf(lo) - pdf(hi)) / mass
        } else if hi < 0.0 {
            hi
        } else {
            lo
        };
    }
    Ok((l, a, b))
}

/// `φ` with `φ(±∞) = 0`.
fn pdf(x: f64) -> f64 {
    if x.is_finite() { norm_pdf(x) } else { 0.0 }
}

/// 0 when the first variable's interval has no mass, which the integrand
/// would give at every point.
fn degenerate(l: &[f64], a: &[f64], b: &[f64], _d: usize) -> Option<f64> {
    let p = norm_cdf(b[0] / l[0]) - norm_cdf(a[0] / l[0]);
    (p <= 0.0).then_some(0.0)
}

/// Genz's integrand at `w ∈ [0, 1)^(d − 1)`.
fn integrand(l: &[f64], a: &[f64], b: &[f64], d: usize, w: &[f64], y: &mut [f64]) -> f64 {
    let (mut lo, mut hi) = (norm_cdf(a[0] / l[0]), norm_cdf(b[0] / l[0]));
    let mut p = hi - lo;
    for i in 1..d {
        if p <= 0.0 {
            return 0.0;
        }
        let u = (lo + w[i - 1] * (hi - lo)).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON / 2.0);
        y[i - 1] = norm_quantile(u);
        let shift: f64 = (0..i).map(|k| l[i * d + k] * y[k]).sum();
        let lii = l[i * d + i];
        lo = norm_cdf((a[i] - shift) / lii);
        hi = norm_cdf((b[i] - shift) / lii);
        p *= hi - lo;
    }
    p.max(0.0)
}

fn integrate(l: &[f64], a: &[f64], b: &[f64], d: usize, settings: MvnSettings) -> MvnEstimate {
    let k = settings.scrambles;
    let mut seeds = SplitMix(settings.seed);
    let mut seqs: Vec<Sobol> = (0..k)
        .map(|_| {
            Sobol::new(d - 1, Scramble::Lms { seed: seeds.next() })
                .expect("dimensions below the table's")
        })
        .collect();
    let mut sums = vec![0.0; k];
    let mut w = vec![0.0; d - 1];
    let mut y = vec![0.0; d];
    let mut n = 0usize;
    let mut batch = 1usize << 8;
    loop {
        for (seq, sum) in seqs.iter_mut().zip(&mut sums) {
            for _ in 0..batch {
                seq.next_point(&mut w);
                *sum += integrand(l, a, b, d, &w, &mut y);
            }
        }
        n += batch;
        let means: Vec<f64> = sums.iter().map(|s| s / n as f64).collect();
        let mean = means.iter().sum::<f64>() / k as f64;
        let var = means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (k * (k - 1)) as f64;
        let error = 3.0 * var.sqrt();
        let converged = error <= settings.abs_tol;
        // The next round doubles the points, so stop before it would pass
        // the budget.
        if converged || 2 * n * k > settings.max_points {
            return MvnEstimate {
                value: mean.clamp(0.0, 1.0),
                error,
                points: n * k,
                converged,
            };
        }
        batch = n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn equi(d: usize, rho: f64) -> Vec<f64> {
        let mut r = vec![rho; d * d];
        for i in 0..d {
            r[i * d + i] = 1.0;
        }
        r
    }

    #[test]
    fn one_and_two_dimensions_are_exact() {
        let s = MvnSettings::default();
        let p = mvn_cdf(&[0.3], &[1.0], s).unwrap();
        assert_eq!((p.value, p.error), (norm_cdf(0.3), 0.0));
        let p = mvn_cdf(&[0.3, -0.2], &[1.0, 0.4, 0.4, 1.0], s).unwrap();
        assert_eq!(p.value, bivariate_norm_cdf(0.3, -0.2, 0.4));
        // A variance of 4 scales the limit.
        let p = mvn_cdf(&[1.0], &[4.0], s).unwrap();
        assert!((p.value - norm_cdf(0.5)).abs() < 1e-15);
        // An unconstrained coordinate drops out.
        let p = mvn_cdf(&[0.3, f64::INFINITY, -0.2], &equi(3, 0.4), s).unwrap();
        assert_eq!(p.value, bivariate_norm_cdf(0.3, -0.2, 0.4));
    }

    #[test]
    fn orthant_probabilities_have_closed_forms() {
        // Equicorrelated at ½: P(all < 0) = 1 / (d + 1).
        for d in 3..=8 {
            let p = mvn_cdf(&vec![0.0; d], &equi(d, 0.5), MvnSettings::default()).unwrap();
            assert!(p.converged);
            assert!((p.value - 1.0 / (d as f64 + 1.0)).abs() < 2e-5, "{d} {p:?}");
        }
        // Independence: the product.
        let b = [0.1, -0.5, 1.2, 0.7];
        let p = mvn_cdf(&b, &equi(4, 0.0), MvnSettings::default()).unwrap();
        let want: f64 = b.iter().map(|&x| norm_cdf(x)).product();
        assert!((p.value - want).abs() < 1e-5);
    }

    #[test]
    fn limits_and_matrices_are_checked() {
        let s = MvnSettings::default();
        assert!(mvn_cdf(&[0.0; 3], &equi(3, 1.0), s).is_err());
        assert!(mvn_cdf(&[0.0; 3], &equi(2, 0.1), s).is_err());
        assert!(mvn_probability(&[1.0; 3], &[0.0; 3], &equi(3, 0.1), s).is_err());
        let mut r = equi(3, 0.1);
        r[1] = 0.2;
        assert!(mvn_cdf(&[0.0; 3], &r, s).is_err());
        let bad = MvnSettings { scrambles: 1, ..s };
        assert!(mvn_cdf(&[0.0; 3], &equi(3, 0.1), bad).is_err());
        // An empty box has probability 0, and the whole space 1.
        let p = mvn_probability(&[0.0; 3], &[0.0; 3], &equi(3, 0.1), s).unwrap();
        assert_eq!(p.value, 0.0);
        let p = mvn_cdf(&[f64::INFINITY; 3], &equi(3, 0.1), s).unwrap();
        assert_eq!(p.value, 1.0);
    }

    #[test]
    fn a_call_is_reproducible_by_seed() {
        let r = equi(5, 0.3);
        let b = [0.2, 0.5, -0.1, 1.0, 0.3];
        let s = MvnSettings::default();
        assert_eq!(mvn_cdf(&b, &r, s).unwrap(), mvn_cdf(&b, &r, s).unwrap());
        let t = MvnSettings { seed: 9, ..s };
        let (p, q) = (mvn_cdf(&b, &r, s).unwrap(), mvn_cdf(&b, &r, t).unwrap());
        assert!((p.value - q.value).abs() < p.error + q.error);
    }
}
