//! Copulas: dependence between marginals, separate from the marginals.
//!
//! A copula draws one vector of uniforms per simulation; marginals are
//! applied by inverse transform ([`simulate`]). Simulation `i` uses only
//! `StreamRng::new(seed, i)`, so results do not depend on thread count and
//! any simulation replays alone (see `docs/design/risk.md`).

use prospicio_core::{Error, Result, StreamRng};
use prospicio_math::linalg::{cholesky, lower_mul, lower_solve};
use prospicio_math::special::{ln_gamma, norm_cdf, norm_quantile, student_t_cdf};

use crate::distribution::Distribution;
use crate::gamma::standard_gamma;
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
/// use prospicio_core::StreamRng;
/// use prospicio_prob::copula::{Copula, GaussianCopula};
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
            *x = open01(norm_cdf(*x));
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
        let w = 2.0 * standard_gamma(rng, 0.5 * self.nu);
        let scale = (w / self.nu).sqrt();
        for x in u.iter_mut() {
            *x = open01(student_t_cdf(*x / scale, self.nu));
        }
    }
}

/// An Archimedean copula family; see [`ArchimedeanCopula`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Archimedean {
    Clayton,
    Gumbel,
    Frank,
    Joe,
}

/// An exchangeable `d`-dimensional Archimedean copula with generator
/// `ψ_θ`, sampled by Marshall and Olkin's frailty method: draw a frailty
/// `V` (whose Laplace transform is `ψ`), then `d` unit exponentials `E_j`,
/// and return `u_j = ψ(E_j / V)`.
///
/// | Family | `ψ(t)` | Frailty `V` | `θ` | Kendall's tau |
/// |---|---|---|---|---|
/// | Clayton | `(1 + t)^(-1/θ)` | Gamma(`1/θ`) | `> 0` | `θ / (θ + 2)` |
/// | Gumbel | `exp(-t^(1/θ))` | positive stable(`1/θ`) | `>= 1` | `1 - 1/θ` |
/// | Frank | `-ln(1 - (1 - e^-θ) e^-t) / θ` | logarithmic(`1 - e^-θ`) | `> 0` | `1 + 4 (D_1(θ) - 1) / θ` |
/// | Joe | `1 - (1 - e^-t)^(1/θ)` | Sibuya(`1/θ`) | `>= 1` | `1 - 4 Σ_k 1 / (k (θk + 2)(θ(k - 1) + 2))` |
///
/// Clayton has lower-tail dependence `2^(-1/θ)`; Gumbel and Joe have
/// upper-tail dependence `2 - 2^(1/θ)`; Frank has none. Draws use the
/// frailty first, then the exponentials, all from the simulation's stream.
///
/// ```
/// use prospicio_core::StreamRng;
/// use prospicio_prob::copula::{Archimedean, ArchimedeanCopula, Copula};
///
/// // Clayton with tau = 0.5.
/// let c = ArchimedeanCopula::new(Archimedean::Clayton, 2.0, 3).unwrap();
/// let mut u = [0.0; 3];
/// c.sample(&mut StreamRng::new(1, 0), &mut u);
/// assert!(u.iter().all(|&x| x > 0.0 && x < 1.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ArchimedeanCopula {
    family: Archimedean,
    theta: f64,
    dim: usize,
}

impl ArchimedeanCopula {
    /// A `dim`-dimensional copula of `family` with parameter `theta`.
    pub fn new(family: Archimedean, theta: f64, dim: usize) -> Result<Self> {
        let ok = theta.is_finite()
            && match family {
                Archimedean::Clayton | Archimedean::Frank => theta > 0.0,
                Archimedean::Gumbel | Archimedean::Joe => theta >= 1.0,
            };
        if !ok {
            let reason = match family {
                Archimedean::Clayton | Archimedean::Frank => "must be finite and positive",
                Archimedean::Gumbel | Archimedean::Joe => "must be finite and at least 1",
            };
            return Err(invalid("theta", theta, reason));
        }
        if dim == 0 {
            return Err(invalid("dim", 0.0, "must be positive"));
        }
        Ok(Self { family, theta, dim })
    }

    pub fn family(&self) -> Archimedean {
        self.family
    }

    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// The generator `ψ(t)` for `t >= 0`.
    pub fn generator(&self, t: f64) -> f64 {
        let th = self.theta;
        match self.family {
            Archimedean::Clayton => (-(t.ln_1p()) / th).exp(),
            Archimedean::Gumbel => (-t.powf(1.0 / th)).exp(),
            // -ln(1 + e^-t (e^-θ - 1)) / θ.
            Archimedean::Frank => -((-t).exp() * (-th).exp_m1()).ln_1p() / th,
            // 1 - (1 - e^-t)^(1/θ).
            Archimedean::Joe => -((-(-t).exp_m1()).ln() / th).exp_m1(),
        }
    }

    fn frailty(&self, rng: &mut StreamRng) -> f64 {
        let th = self.theta;
        match self.family {
            Archimedean::Clayton => standard_gamma(rng, 1.0 / th),
            Archimedean::Gumbel => positive_stable(rng, 1.0 / th),
            Archimedean::Frank => logarithmic(rng, th),
            Archimedean::Joe => sibuya(rng, 1.0 / th),
        }
    }
}

impl Copula for ArchimedeanCopula {
    fn dim(&self) -> usize {
        self.dim
    }

    fn sample(&self, rng: &mut StreamRng, u: &mut [f64]) {
        let v = self.frailty(rng);
        for x in u.iter_mut() {
            let e = -rng.next_open01().ln();
            *x = open01(self.generator(e / v));
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
/// use prospicio_prob::copula::{GaussianCopula, simulate};
/// use prospicio_prob::{Distribution, KeyValue, Lognormal, Provenance};
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
    let parallel = marginals.iter().all(|m| m.is_parallel_safe());
    PredictiveDistribution::simulate_with(
        parallel,
        dims,
        components,
        n_sims,
        seed,
        provenance,
        |rng, row| {
            copula.sample(rng, row);
            for (x, m) in row.iter_mut().zip(marginals) {
                *x = m.quantile(*x).expect("copula uniforms lie in (0, 1)");
            }
        },
    )
}

/// Iman–Conover target ranks: for each of `m` columns, the rank (0-based,
/// ascending) that each of `n` rows should take so that the columns' rank
/// correlation is close to `correlation`. Column `j > 0` shuffles its
/// normal scores with stream `j` of `seed`.
pub(crate) fn target_ranks(
    n: usize,
    m: usize,
    correlation: &[f64],
    seed: u64,
) -> Result<Vec<Vec<usize>>> {
    let target = correlation_factor(correlation, m)?;
    if n < m + 1 {
        return Err(invalid(
            "n_sims",
            n as f64,
            "must exceed the number of components",
        ));
    }

    // Shuffled normal scores, column-major.
    let scores: Vec<f64> = (1..=n)
        .map(|i| norm_quantile(i as f64 / (n + 1) as f64))
        .collect();
    let mut cols: Vec<Vec<f64>> = (0..m)
        .map(|j| {
            let mut c = scores.clone();
            if j > 0 {
                shuffle(&mut c, &mut StreamRng::new(seed, j as u64));
            }
            c
        })
        .collect();

    // Rotate to exactly the target correlation: T = M F^-T P^T, row by row.
    let actual = correlation_factor(&sample_correlation(&cols), m)
        .map_err(|_| invalid("n_sims", n as f64, "too few to decorrelate the scores"))?;
    let (mut row, mut y, mut t) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
    for i in 0..n {
        for (r, col) in row.iter_mut().zip(&cols) {
            *r = col[i];
        }
        lower_solve(&actual, &row, &mut y);
        lower_mul(&target, &y, &mut t);
        for (col, v) in cols.iter_mut().zip(&t) {
            col[i] = *v;
        }
    }

    Ok(cols
        .iter()
        .map(|col| {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| col[a].total_cmp(&col[b]));
            let mut rank = vec![0; n];
            for (r, &i) in order.iter().enumerate() {
                rank[i] = r;
            }
            rank
        })
        .collect())
}

/// Reorders each component's draws so the components have (close to) the
/// target correlation of normal scores, by Iman and Conover (1982). Every component
/// keeps exactly its own draws; only their pairing across simulations
/// changes.
///
/// Builds an `n × m` matrix whose columns are the normal scores
/// `Φ⁻¹(i / (n + 1))`, shuffled independently (column `j` by
/// `StreamRng::new(seed, j)`; column 0 is not shuffled), transforms it to
/// have exactly the correlation `correlation`, and gives each component's
/// draws the ranks of the matching column. The correlation of the
/// result's normal scores (van der Waerden) is then close to
/// `correlation`, not exact, and Spearman's rho is close to
/// `(6 / π) asin(correlation / 2)`, as for a Gaussian copula.
///
/// Use it to join results simulated separately, for example a reserve and
/// a premium-risk distribution, without resimulating either.
///
/// ```
/// use prospicio_prob::copula::iman_conover;
/// use prospicio_prob::{Empirical, KeyValue, PredictiveDistribution, Provenance};
///
/// // Two components, both 1..=1000, simulated independently.
/// let n = 1000;
/// let draws: Vec<f64> = (0..n).flat_map(|i| [i as f64, ((i * 7919) % n) as f64]).collect();
/// let pd = PredictiveDistribution::from_draws(
///     vec!["lob".into()],
///     vec![vec![KeyValue::Int(0)], vec![KeyValue::Int(1)]],
///     draws,
///     Provenance::new("example"),
/// )
/// .unwrap();
/// let joined = iman_conover(&pd, &[1.0, 0.7, 0.7, 1.0], 3).unwrap();
/// assert_eq!(joined.marginal(&vec![KeyValue::Int(1)]).unwrap().sorted(),
///            pd.marginal(&vec![KeyValue::Int(1)]).unwrap().sorted());
/// ```
pub fn iman_conover(
    pd: &PredictiveDistribution,
    correlation: &[f64],
    seed: u64,
) -> Result<PredictiveDistribution> {
    let m = pd.n_components();
    let n = pd.n_sims();
    let ranks = target_ranks(n, m, correlation, seed)?;

    // Each component takes its sorted draws in the target ranks.
    let mut draws = vec![0.0; n * m];
    for (j, rank) in ranks.iter().enumerate() {
        let mut sorted: Vec<f64> = (0..n).map(|i| pd.row(i).expect("in range")[j]).collect();
        sorted.sort_by(f64::total_cmp);
        for (i, &r) in rank.iter().enumerate() {
            draws[i * m + j] = sorted[r];
        }
    }
    let provenance = pd
        .provenance()
        .clone()
        .param("iman_conover_correlation", format!("{correlation:?}"))
        .param("iman_conover_seed", seed);
    PredictiveDistribution::from_draws(
        pd.dims().to_vec(),
        pd.components().to_vec(),
        draws,
        provenance,
    )
}

/// Pearson correlation matrix of columns, row-major.
fn sample_correlation(cols: &[Vec<f64>]) -> Vec<f64> {
    let m = cols.len();
    let n = cols[0].len() as f64;
    let centred: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| {
            let mean = c.iter().sum::<f64>() / n;
            c.iter().map(|x| x - mean).collect()
        })
        .collect();
    let norms: Vec<f64> = centred
        .iter()
        .map(|c| c.iter().map(|x| x * x).sum::<f64>().sqrt())
        .collect();
    let mut r = vec![0.0; m * m];
    for i in 0..m {
        r[i * m + i] = 1.0;
        for j in 0..i {
            let dot: f64 = centred[i].iter().zip(&centred[j]).map(|(a, b)| a * b).sum();
            let v = dot / (norms[i] * norms[j]);
            r[i * m + j] = v;
            r[j * m + i] = v;
        }
    }
    r
}

/// Keeps a uniform inside `(0, 1)`, where `norm_cdf` or a generator can
/// round to 0 or 1 far in a tail; marginal quantiles are infinite there.
fn open01(u: f64) -> f64 {
    u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON / 2.0)
}

/// A positive stable draw with Laplace transform `exp(-t^alpha)`,
/// `0 < alpha <= 1`, by Kanter's representation (Chambers, Mallows and
/// Stuck): one uniform angle, then one unit exponential.
fn positive_stable(rng: &mut StreamRng, alpha: f64) -> f64 {
    if alpha == 1.0 {
        return 1.0;
    }
    let theta = std::f64::consts::PI * rng.next_open01();
    let w = -rng.next_open01().ln();
    let a = (alpha * theta).sin() / theta.sin().powf(1.0 / alpha);
    let b = (((1.0 - alpha) * theta).sin() / w).powf((1.0 - alpha) / alpha);
    a * b
}

/// A logarithmic-series draw, `P(V = k) = p^k / (-k ln(1 - p))` with
/// `p = 1 - e^-theta`, by Kemp's (1981) LK algorithm: two uniforms.
fn logarithmic(rng: &mut StreamRng, theta: f64) -> f64 {
    let p = -(-theta).exp_m1();
    let v = rng.next_open01();
    let u = rng.next_open01();
    if v > p {
        return 1.0;
    }
    // q = 1 - (1 - p)^u = 1 - e^(-theta u).
    let q = -(-theta * u).exp_m1();
    if v < q * q {
        (1.0 + v.ln() / q.ln()).floor()
    } else if v > q {
        1.0
    } else {
        2.0
    }
}

/// A Sibuya draw, `P(V > k) = Γ(k + 1 - alpha) / (Γ(k + 1) Γ(1 - alpha))`
/// for `0 < alpha <= 1`, by inverting the distribution function with one
/// uniform. The tail is heavy (no mean), so the search runs in `f64`.
fn sibuya(rng: &mut StreamRng, alpha: f64) -> f64 {
    let u = rng.next_open01();
    if alpha == 1.0 || u <= alpha {
        return 1.0;
    }
    let target = (1.0 - u).ln();
    let ln_survival =
        |k: f64| ln_gamma(k + 1.0 - alpha) - ln_gamma(k + 1.0) - ln_gamma(1.0 - alpha);
    // Smallest k with P(V > k) <= 1 - u: double, then bisect.
    let (mut lo, mut hi) = (1.0, 2.0);
    while ln_survival(hi) > target {
        lo = hi;
        hi *= 2.0;
        if hi > 1e300 {
            return hi;
        }
    }
    while hi - lo > 1.0 {
        let mid = (lo + (hi - lo) / 2.0).floor();
        if ln_survival(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// Fisher–Yates shuffle with uniform indices from `rng`.
fn shuffle(x: &mut [f64], rng: &mut StreamRng) {
    for i in (1..x.len()).rev() {
        let j = ((rng.next_open01() * (i + 1) as f64) as usize).min(i);
        x.swap(i, j);
    }
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
            let x: Vec<f64> = (0..n).map(|_| standard_gamma(&mut rng, shape)).collect();
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

    /// Pearson correlation of `f(rank / (n + 1))` for 1-based ranks.
    fn rank_correlation(x: &[f64], y: &[f64], f: fn(f64) -> f64) -> f64 {
        let n = x.len();
        let scores = |v: &[f64]| {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
            let mut r = vec![0.0; n];
            for (k, &i) in order.iter().enumerate() {
                r[i] = f((k + 1) as f64 / (n + 1) as f64);
            }
            r
        };
        sample_correlation(&[scores(x), scores(y)])[1]
    }

    #[test]
    fn iman_conover_keeps_marginals_and_reaches_the_target() {
        use crate::{Empirical, KeyValue, Lognormal};
        let a = Lognormal::from_mean_cv(100.0, 0.3).unwrap();
        let b = Lognormal::from_mean_cv(50.0, 1.5).unwrap();
        let c = Lognormal::from_mean_cv(10.0, 0.8).unwrap();
        let independent =
            GaussianCopula::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3).unwrap();
        let keys: Vec<ComponentKey> = (0..3).map(|j| vec![KeyValue::Int(j)]).collect();
        let n = 10_000;
        let pd = simulate(
            &independent,
            &[&a, &b, &c],
            vec!["lob".into()],
            keys.clone(),
            n,
            1,
            Provenance::new("t"),
        )
        .unwrap();
        let joined = iman_conover(&pd, &R, 7).unwrap();
        for key in &keys {
            assert_eq!(
                joined.marginal(key).unwrap().sorted(),
                pd.marginal(key).unwrap().sorted()
            );
        }
        let col = |j: usize| -> Vec<f64> { (0..n).map(|i| joined.row(i).unwrap()[j]).collect() };
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let r = R[i * 3 + j];
            let normal_scores = rank_correlation(&col(i), &col(j), norm_quantile);
            assert!(
                (normal_scores - r).abs() < 0.01,
                "({i}, {j}): {normal_scores} vs {r}"
            );
            let spearman = rank_correlation(&col(i), &col(j), |u| u);
            let want = 6.0 / PI * (r / 2.0).asin();
            assert!(
                (spearman - want).abs() < 0.01,
                "({i}, {j}): {spearman} vs {want}"
            );
        }
        // Reproducible, and recorded.
        assert_eq!(
            iman_conover(&pd, &R, 7).unwrap().draw_matrix(),
            joined.draw_matrix()
        );
        assert!(
            joined
                .provenance()
                .parameters
                .iter()
                .any(|(k, _)| k == "iman_conover_seed")
        );
        // Bad inputs.
        assert!(iman_conover(&pd, &[1.0, 0.5, 0.5, 1.0], 7).is_err());
    }

    fn frank_tau(theta: f64) -> f64 {
        // Debye D_1(θ) = (1/θ) ∫_0^θ t / (e^t - 1) dt, by Simpson's rule.
        let n = 20_000;
        let h = theta / n as f64;
        let f = |t: f64| if t == 0.0 { 1.0 } else { t / t.exp_m1() };
        let mut sum = f(0.0) + f(theta);
        for i in 1..n {
            sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        let d1 = sum * h / 3.0 / theta;
        1.0 + 4.0 * (d1 - 1.0) / theta
    }

    fn joe_tau(theta: f64) -> f64 {
        let s: f64 = (1..2_000_000)
            .map(|k| {
                let k = k as f64;
                1.0 / (k * (theta * k + 2.0) * (theta * (k - 1.0) + 2.0))
            })
            .sum();
        1.0 - 4.0 * s
    }

    #[test]
    fn archimedean_kendall_tau() {
        use Archimedean::*;
        let n = 3_000;
        for (family, theta, want) in [
            (Clayton, 2.0, 0.5),
            (Clayton, 0.3, 0.3 / 2.3),
            (Gumbel, 1.0, 0.0),
            (Gumbel, 2.5, 0.6),
            (Frank, 5.0, frank_tau(5.0)),
            (Frank, 0.5, frank_tau(0.5)),
            (Joe, 2.0, joe_tau(2.0)),
            (Joe, 6.0, joe_tau(6.0)),
        ] {
            let c = ArchimedeanCopula::new(family, theta, 3).unwrap();
            let u = draws(&c, n, 21);
            for (i, j) in [(0, 1), (1, 2)] {
                let x: Vec<f64> = u.iter().map(|r| r[i]).collect();
                let y: Vec<f64> = u.iter().map(|r| r[j]).collect();
                let got = kendall_tau(&x, &y);
                assert!(
                    (got - want).abs() < 0.04,
                    "{family:?}({theta}) ({i}, {j}): {got} vs {want}"
                );
            }
        }
        // Frank at θ = 5 has tau 0.4567 (Nelsen, Table 5.1 rounding).
        assert!((frank_tau(5.0) - 0.4567).abs() < 1e-3);
    }

    #[test]
    fn archimedean_margins_are_uniform() {
        use Archimedean::*;
        let n = 30_000;
        for (family, theta) in [(Clayton, 1.5), (Gumbel, 3.0), (Frank, 8.0), (Joe, 4.0)] {
            let c = ArchimedeanCopula::new(family, theta, 2).unwrap();
            let u = draws(&c, n, 6);
            for j in 0..2 {
                let d = ks_uniform(u.iter().map(|r| r[j]).collect());
                assert!(
                    d < 1.95 / (n as f64).sqrt(),
                    "{family:?} dimension {j}: {d}"
                );
            }
        }
    }

    #[test]
    fn archimedean_tails() {
        use Archimedean::*;
        let n = 200_000;
        let q = 0.995;
        let upper = |c: &dyn Copula| {
            draws(c, n, 13)
                .iter()
                .filter(|u| u[0] > q && u[1] > q)
                .count() as f64
                / (n as f64 * (1.0 - q))
        };
        let lower = |c: &dyn Copula| {
            draws(c, n, 13)
                .iter()
                .filter(|u| u[0] < 1.0 - q && u[1] < 1.0 - q)
                .count() as f64
                / (n as f64 * (1.0 - q))
        };
        let clayton = ArchimedeanCopula::new(Clayton, 2.0, 2).unwrap();
        let gumbel = ArchimedeanCopula::new(Gumbel, 2.0, 2).unwrap();
        // Clayton: lower tail 2^(-1/2) = 0.707, little in the upper tail.
        assert!((lower(&clayton) - 0.5f64.sqrt()).abs() < 0.08);
        assert!(upper(&clayton) < 0.2);
        // Gumbel: upper tail 2 - 2^(1/2) = 0.586, little in the lower tail.
        assert!((upper(&gumbel) - (2.0 - 2f64.sqrt())).abs() < 0.08);
        assert!(lower(&gumbel) < 0.2);
    }

    #[test]
    fn frailty_samplers() {
        let mut rng = StreamRng::new(17, 0);
        let n = 200_000;
        // Logarithmic: mean p / (-(1 - p) ln(1 - p)) with p = 1 - e^-θ.
        let theta: f64 = 3.0;
        let p = 1.0 - (-theta).exp();
        let mean = (0..n).map(|_| logarithmic(&mut rng, theta)).sum::<f64>() / n as f64;
        let want = p / ((1.0 - p) * theta);
        assert!((mean / want - 1.0).abs() < 0.02, "{mean} vs {want}");
        // Sibuya: P(V = 1) = alpha, P(V = 2) = alpha (1 - alpha) / 2.
        let alpha = 0.4;
        let v: Vec<f64> = (0..n).map(|_| sibuya(&mut rng, alpha)).collect();
        let share = |k: f64| v.iter().filter(|&&x| x == k).count() as f64 / n as f64;
        assert!((share(1.0) - alpha).abs() < 0.005);
        assert!((share(2.0) - alpha * (1.0 - alpha) / 2.0).abs() < 0.005);
        assert!(v.iter().all(|&x| x >= 1.0 && x.fract() == 0.0));
        // Positive stable: E[exp(-V)] = exp(-1) for any alpha.
        for alpha in [0.3, 0.7] {
            let m = (0..n)
                .map(|_| (-positive_stable(&mut rng, alpha)).exp())
                .sum::<f64>()
                / n as f64;
            assert!((m - (-1f64).exp()).abs() < 0.005, "alpha {alpha}: {m}");
        }
    }

    #[test]
    fn archimedean_rejects_bad_parameters() {
        use Archimedean::*;
        assert!(ArchimedeanCopula::new(Clayton, 0.0, 2).is_err());
        assert!(ArchimedeanCopula::new(Gumbel, 0.9, 2).is_err());
        assert!(ArchimedeanCopula::new(Frank, -1.0, 2).is_err());
        assert!(ArchimedeanCopula::new(Joe, f64::INFINITY, 2).is_err());
        assert!(ArchimedeanCopula::new(Clayton, 1.0, 0).is_err());
        let g = ArchimedeanCopula::new(Gumbel, 2.0, 2).unwrap();
        assert_eq!(g.generator(0.0), 1.0);
        assert!(g.generator(1e6) < 1e-300);
    }
}
