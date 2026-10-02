//! Copulas: dependence between marginals, separate from the marginals.
//!
//! A copula draws one vector of uniforms per simulation; marginals are
//! applied by inverse transform ([`simulate`]). Simulation `i` uses only
//! `StreamRng::new(seed, i)`, so results do not depend on thread count and
//! any simulation replays alone (see `docs/design/risk.md`).

use act_core::{Error, Result, StreamRng};
use act_math::linalg::{cholesky, lower_mul};
use act_math::special::{norm_cdf, norm_quantile, student_t_cdf};

use crate::distribution::Distribution;
use crate::predictive::{ComponentKey, PredictiveDistribution};
use crate::provenance::Provenance;

/// A `d`-dimensional copula.
pub trait Copula: Sync {
    /// Number of dimensions.
    fn dim(&self) -> usize;

    /// Fills `u` (length [`dim`](Self::dim)) with one draw of uniforms in
    /// `(0, 1)`.
    fn sample(&self, rng: &mut StreamRng, u: &mut [f64]);
}

/// The Gaussian copula with correlation matrix `R`.
///
/// One draw takes `d` standard normals `z` by inverse transform, in order,
/// sets `y = L z` with `L Lᵀ = R`, and returns `u_j = Φ(y_j)`. Kendall's
/// tau between dimensions `i` and `j` is `(2 / π) asin(R_ij)`; there is no
/// tail dependence.
///
/// ```
/// use act_core::StreamRng;
/// use act_prob::copula::{Copula, GaussianCopula};
///
/// let c = GaussianCopula::new(&[1.0, 0.5, 0.5, 1.0], 2).unwrap();
/// let mut u = [0.0; 2];
/// c.sample(&mut StreamRng::new(1, 0), &mut u);
/// assert!(u.iter().all(|&x| x > 0.0 && x < 1.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianCopula {
    dim: usize,
    chol: Vec<f64>,
}

impl GaussianCopula {
    /// A Gaussian copula from a `d × d` correlation matrix, row-major. It
    /// must be symmetric with a unit diagonal and positive definite.
    pub fn new(correlation: &[f64], dim: usize) -> Result<Self> {
        Ok(Self {
            dim,
            chol: correlation_factor(correlation, dim)?,
        })
    }

    /// The Cholesky factor `L` of the correlation matrix, row-major.
    pub fn factor(&self) -> &[f64] {
        &self.chol
    }
}

impl Copula for GaussianCopula {
    fn dim(&self) -> usize {
        self.dim
    }

    fn sample(&self, rng: &mut StreamRng, u: &mut [f64]) {
        correlated_normals(&self.chol, rng, u);
        for x in u.iter_mut() {
            *x = norm_cdf(*x);
        }
    }
}

/// The Student t copula with correlation matrix `R` and `nu` degrees of
/// freedom.
///
/// One draw takes `y = L z` as the Gaussian copula does, then a chi-square
/// `w` with `nu` degrees of freedom (Marsaglia–Tsang gamma from the same
/// stream), and returns `u_j = T_nu(y_j / sqrt(w / nu))`. Kendall's tau is
/// `(2 / π) asin(R_ij)`, as for the Gaussian, but extremes occur together:
/// the tail dependence is `2 T_{nu+1}(-sqrt((nu + 1)(1 - R_ij) / (1 + R_ij)))`.
#[derive(Debug, Clone, PartialEq)]
pub struct StudentTCopula {
    dim: usize,
    chol: Vec<f64>,
    nu: f64,
}

impl StudentTCopula {
    /// A t copula from a `d × d` correlation matrix (row-major, as for
    /// [`GaussianCopula::new`]) and `nu > 0` degrees of freedom.
    pub fn new(correlation: &[f64], dim: usize, nu: f64) -> Result<Self> {
        if !nu.is_finite() || nu <= 0.0 {
            return Err(invalid("nu", nu, "must be finite and positive"));
        }
        Ok(Self {
            dim,
            chol: correlation_factor(correlation, dim)?,
            nu,
        })
    }

    /// Degrees of freedom.
    pub fn nu(&self) -> f64 {
        self.nu
    }
}

impl Copula for StudentTCopula {
    fn dim(&self) -> usize {
        self.dim
    }

    fn sample(&self, rng: &mut StreamRng, u: &mut [f64]) {
        correlated_normals(&self.chol, rng, u);
        let w = 2.0 * gamma(rng, 0.5 * self.nu);
        let scale = (w / self.nu).sqrt();
        for x in u.iter_mut() {
            *x = student_t_cdf(*x / scale, self.nu);
        }
    }
}

/// Simulates marginals joined by a copula: in simulation `i`, draws `u`
/// from `copula` with `StreamRng::new(seed, i)` and sets component `j` to
/// `marginals[j].quantile(u_j)`.
///
/// The result has one component per marginal, keyed by `components`
/// under `dims`, and records the seed in its provenance.
///
/// ```
/// use act_prob::copula::{GaussianCopula, simulate};
/// use act_prob::{Distribution, KeyValue, Lognormal, Provenance};
///
/// let motor = Lognormal::from_mean_cv(100.0, 0.2).unwrap();
/// let property = Lognormal::from_mean_cv(50.0, 1.0).unwrap();
/// let copula = GaussianCopula::new(&[1.0, 0.4, 0.4, 1.0], 2).unwrap();
/// let pd = simulate(
///     &copula,
///     &[&motor, &property],
///     vec!["lob".into()],
///     vec![vec![KeyValue::from("motor")], vec![KeyValue::from("property")]],
///     10_000,
///     42,
///     Provenance::new("portfolio"),
/// )
/// .unwrap();
/// assert!((pd.mean() - 150.0).abs() < 3.0);
/// ```
pub fn simulate(
    copula: &dyn Copula,
    marginals: &[&(dyn Distribution + Sync)],
    dims: Vec<String>,
    components: Vec<ComponentKey>,
    n_sims: usize,
    seed: u64,
    provenance: Provenance,
) -> Result<PredictiveDistribution> {
    if marginals.len() != copula.dim() {
        return Err(invalid(
            "marginals",
            marginals.len() as f64,
            "must have one marginal per copula dimension",
        ));
    }
    PredictiveDistribution::simulate(dims, components, n_sims, seed, provenance, |rng, row| {
        copula.sample(rng, row);
        for (x, m) in row.iter_mut().zip(marginals) {
            *x = m.quantile(*x).expect("copula uniforms lie in (0, 1)");
        }
    })
}

/// Checks a correlation matrix and returns its Cholesky factor.
fn correlation_factor(r: &[f64], dim: usize) -> Result<Vec<f64>> {
    if dim == 0 || r.len() != dim * dim {
        return Err(invalid(
            "correlation",
            r.len() as f64,
            "must be a non-empty dim × dim matrix",
        ));
    }
    for i in 0..dim {
        if r[i * dim + i] != 1.0 {
            return Err(invalid("correlation", r[i * dim + i], "diagonal must be 1"));
        }
        for j in 0..i {
            let (a, b) = (r[i * dim + j], r[j * dim + i]);
            if a != b {
                return Err(invalid("correlation", a, "must be symmetric"));
            }
            if !(-1.0..=1.0).contains(&a) {
                return Err(invalid("correlation", a, "entries must be in [-1, 1]"));
            }
        }
    }
    cholesky(r, dim).ok_or_else(|| invalid("correlation", 0.0, "must be positive definite"))
}

/// Fills `out` with `L z` for standard normals `z` drawn in order by
/// inverse transform.
fn correlated_normals(chol: &[f64], rng: &mut StreamRng, out: &mut [f64]) {
    let z: Vec<f64> = (0..out.len())
        .map(|_| norm_quantile(rng.next_open01()))
        .collect();
    lower_mul(chol, &z, out);
}

/// A Gamma(`shape`, 1) draw by Marsaglia and Tsang (2000), with the
/// `U^(1/shape)` boost for `shape < 1`. Normals are by inverse transform.
fn gamma(rng: &mut StreamRng, shape: f64) -> f64 {
    if shape < 1.0 {
        let g = gamma(rng, shape + 1.0);
        return g * rng.next_open01().powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = norm_quantile(rng.next_open01());
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.next_open01();
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn draws(c: &dyn Copula, n: usize, seed: u64) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let mut u = vec![0.0; c.dim()];
                c.sample(&mut StreamRng::new(seed, i as u64), &mut u);
                u
            })
            .collect()
    }

    fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        let mut s = 0.0;
        for i in 0..n {
            for j in 0..i {
                s += ((x[i] - x[j]) * (y[i] - y[j])).signum();
            }
        }
        s / (n * (n - 1) / 2) as f64
    }

    /// Kolmogorov–Smirnov distance from the uniform distribution.
    fn ks_uniform(mut u: Vec<f64>) -> f64 {
        u.sort_by(f64::total_cmp);
        let n = u.len() as f64;
        u.iter()
            .enumerate()
            .map(|(i, &x)| (x - i as f64 / n).max((i + 1) as f64 / n - x))
            .fold(0.0, f64::max)
    }

    const R: [f64; 9] = [1.0, 0.6, -0.3, 0.6, 1.0, 0.1, -0.3, 0.1, 1.0];

    #[test]
    fn kendall_tau_matches_the_arcsine_law() {
        let gauss = GaussianCopula::new(&R, 3).unwrap();
        let t = StudentTCopula::new(&R, 3, 4.0).unwrap();
        let t_half = StudentTCopula::new(&R, 3, 0.7).unwrap();
        let n = 4_000;
        for c in [&gauss as &dyn Copula, &t, &t_half] {
            let u = draws(c, n, 9);
            for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                let x: Vec<f64> = u.iter().map(|r| r[i]).collect();
                let y: Vec<f64> = u.iter().map(|r| r[j]).collect();
                let want = 2.0 / PI * R[i * 3 + j].asin();
                // Standard error of tau is below 0.012 at n = 4,000.
                let got = kendall_tau(&x, &y);
                assert!((got - want).abs() < 0.04, "({i}, {j}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn margins_are_uniform() {
        let n = 50_000;
        for c in [
            &GaussianCopula::new(&R, 3).unwrap() as &dyn Copula,
            &StudentTCopula::new(&R, 3, 3.0).unwrap(),
            &StudentTCopula::new(&R, 3, 0.7).unwrap(),
        ] {
            let u = draws(c, n, 2);
            for j in 0..3 {
                let d = ks_uniform(u.iter().map(|r| r[j]).collect());
                // 0.1% critical value: 1.95 / sqrt(n).
                assert!(d < 1.95 / (n as f64).sqrt(), "dimension {j}: {d}");
                assert!(u.iter().all(|r| r[j] > 0.0 && r[j] < 1.0));
            }
        }
    }

    #[test]
    fn t_copula_has_joint_extremes() {
        let r = [1.0, 0.5, 0.5, 1.0];
        let n = 200_000;
        let q = 0.995;
        let joint = |c: &dyn Copula| {
            draws(c, n, 4)
                .iter()
                .filter(|u| u[0] > q && u[1] > q)
                .count() as f64
                / (n as f64 * (1.0 - q))
        };
        let gauss = joint(&GaussianCopula::new(&r, 2).unwrap());
        let t = joint(&StudentTCopula::new(&r, 2, 3.0).unwrap());
        // Limits as q -> 1: 0 for the Gaussian, 0.3125 for t(3) at 0.5.
        let lambda = 2.0 * student_t_cdf(-(4.0f64 * 0.5 / 1.5).sqrt(), 4.0);
        assert!((lambda - 0.3125).abs() < 1e-3, "{lambda}");
        assert!(t > 1.5 * gauss, "t {t} vs Gaussian {gauss}");
        assert!((t - lambda).abs() < 0.1, "{t} vs {lambda}");
    }

    #[test]
    fn gamma_moments() {
        for shape in [0.35, 1.0, 2.5, 40.0] {
            let n = 100_000;
            let mut rng = StreamRng::new(3, 0);
            let x: Vec<f64> = (0..n).map(|_| gamma(&mut rng, shape)).collect();
            let mean = x.iter().sum::<f64>() / n as f64;
            let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            let se = (shape / n as f64).sqrt();
            assert!(
                (mean - shape).abs() < 4.0 * se,
                "shape {shape}: mean {mean}"
            );
            assert!((var / shape - 1.0).abs() < 0.05, "shape {shape}: var {var}");
        }
    }

    #[test]
    fn rejects_bad_correlations() {
        assert!(GaussianCopula::new(&[1.0, 0.5, 0.4, 1.0], 2).is_err());
        assert!(GaussianCopula::new(&[1.0, 1.5, 1.5, 1.0], 2).is_err());
        assert!(GaussianCopula::new(&[2.0, 0.0, 0.0, 1.0], 2).is_err());
        assert!(GaussianCopula::new(&[1.0, 0.0, 0.0], 2).is_err());
        // Each pair is valid; the matrix is not positive definite.
        let r = [1.0, 0.9, -0.9, 0.9, 1.0, 0.9, -0.9, 0.9, 1.0];
        assert!(GaussianCopula::new(&r, 3).is_err());
        assert!(StudentTCopula::new(&[1.0], 1, 0.0).is_err());
    }

    #[test]
    fn simulate_is_reproducible_and_checks_dimensions() {
        use crate::{KeyValue, Lognormal};
        let a = Lognormal::from_mean_cv(10.0, 0.5).unwrap();
        let b = Lognormal::from_mean_cv(20.0, 0.5).unwrap();
        let c = StudentTCopula::new(&[1.0, 0.3, 0.3, 1.0], 2, 5.0).unwrap();
        let keys = || vec![vec![KeyValue::Int(0)], vec![KeyValue::Int(1)]];
        let run = || {
            simulate(
                &c,
                &[&a, &b],
                vec!["lob".into()],
                keys(),
                500,
                8,
                Provenance::new("t"),
            )
            .unwrap()
        };
        let (x, y) = (run(), run());
        assert_eq!(x.draw_matrix(), y.draw_matrix());
        // Row 17 replays alone.
        let mut u = [0.0; 2];
        c.sample(&mut StreamRng::new(8, 17), &mut u);
        assert_eq!(
            x.row(17).unwrap(),
            [a.quantile(u[0]).unwrap(), b.quantile(u[1]).unwrap()]
        );
        assert!(
            simulate(
                &c,
                &[&a],
                vec!["lob".into()],
                keys(),
                10,
                1,
                Provenance::new("t")
            )
            .is_err()
        );
    }
}
