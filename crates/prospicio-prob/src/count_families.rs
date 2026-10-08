//! Claim counts beyond the `(a, b, 0)` class: zero-modified and
//! zero-truncated counts, the logarithmic, mixed Poisson counts, compound
//! (Poisson-stopped) counts and empirical counts. These are the families of
//! Mildenhall's `aggregate` (`zm`, `zt`, `logarithmic`, `mixed gamma`,
//! `delaporte`, `ig`, `sig`, `neymana`, `pascal`, `dfreq`) and of Klugman,
//! Panjer and Willmot, *Loss Models*, chapters 6 and 7.
//!
//! Counts outside the `(a, b, 1)` class keep their probabilities in a table
//! computed on construction out to where the remaining mass is below
//! `1e-16`, so `pmf` is a lookup.

use prospicio_core::{Error, Result};

use crate::counting::{Counting, NegativeBinomial, Poisson};

/// Remaining mass at which a probability table stops.
const TAIL: f64 = 1e-16;
/// The longest table a count may need.
const MAX_TABLE: usize = 50_000_000;

// Complex arithmetic on (re, im) pairs, for the pgfs.
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn cexp(z: (f64, f64)) -> (f64, f64) {
    let m = z.0.exp();
    (m * z.1.cos(), m * z.1.sin())
}
fn cln(z: (f64, f64)) -> (f64, f64) {
    (z.0.hypot(z.1).ln(), z.1.atan2(z.0))
}
fn csqrt(z: (f64, f64)) -> (f64, f64) {
    let r = z.0.hypot(z.1).sqrt();
    let t = 0.5 * z.1.atan2(z.0);
    (r * t.cos(), r * t.sin())
}
fn cscale(z: (f64, f64), k: f64) -> (f64, f64) {
    (z.0 * k, z.1 * k)
}

/// The mean and variance of a probability table.
fn table_moments(p: &[f64]) -> (f64, f64) {
    let mean: f64 = p.iter().enumerate().map(|(k, q)| k as f64 * q).sum();
    let second: f64 = p
        .iter()
        .enumerate()
        .map(|(k, q)| (k as f64).powi(2) * q)
        .sum();
    (mean, second - mean * mean)
}

/// Extends a table by a recursion `next(k, table)` past `at_least`
/// entries, until the mass left is below [`TAIL`] or the terms fall below
/// `1e-20`.
fn extend_table(
    mut p: Vec<f64>,
    at_least: usize,
    mut next: impl FnMut(usize, &[f64]) -> f64,
) -> Result<Vec<f64>> {
    let mut total: f64 = p.iter().sum();
    while p.len() < at_least || 1.0 - total > TAIL {
        if p.len() >= MAX_TABLE {
            return Err(Error::Data(format!(
                "this count needs more than {MAX_TABLE} probabilities; its mean is too large"
            )));
        }
        let k = p.len();
        let q = next(k, &p).max(0.0);
        // Rounding in the running total can leave it a few ulps short of
        // 1 for good; past the bulk, stop once the terms are negligible.
        if k > at_least && q < 1e-20 {
            break;
        }
        total += q;
        p.push(q);
    }
    Ok(p)
}

/// A count `N` with its probability at zero replaced: `P(N = 0) = p0` and
/// `P(N = k) = (1 - p0) / (1 - q0) q_k` for `k ≥ 1`, where `q` is the base
/// count. `p0 = 0` is the zero-truncated count. A zero-modified count of
/// the `(a, b, 0)` class is in the `(a, b, 1)` class, so Panjer's
/// recursion applies.
///
/// ```
/// use prospicio_prob::{Counting, Poisson};
/// use prospicio_prob::count_families::ZeroModified;
///
/// let zt = ZeroModified::truncated(Poisson::new(2.0).unwrap()).unwrap();
/// assert_eq!(zt.pmf(0), 0.0);
/// assert!((zt.mean() - 2.0 / (1.0 - (-2f64).exp())).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ZeroModified<C> {
    base: C,
    p0: f64,
}

impl<C: Counting> ZeroModified<C> {
    /// The base count with `P(N = 0) = p0`, for `p0` in `[0, 1)`.
    pub fn new(base: C, p0: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&p0) {
            return Err(Error::InvalidParameter {
                name: "p0",
                value: p0,
                reason: "must be in [0, 1)",
            });
        }
        if base.pmf(0) >= 1.0 {
            return Err(Error::Data("the base count is always zero".into()));
        }
        Ok(Self { base, p0 })
    }

    /// The zero-truncated base count, `P(N = 0) = 0`.
    pub fn truncated(base: C) -> Result<Self> {
        Self::new(base, 0.0)
    }

    /// The base count.
    pub fn base(&self) -> &C {
        &self.base
    }

    /// `P(N = 0)`.
    pub fn p0(&self) -> f64 {
        self.p0
    }

    /// `(1 - p0) / (1 - q0)`, the scale on the base probabilities above 0.
    fn scale(&self) -> f64 {
        (1.0 - self.p0) / (1.0 - self.base.pmf(0))
    }
}

impl<C: Counting> Counting for ZeroModified<C> {
    fn pmf(&self, k: u64) -> f64 {
        if k == 0 {
            self.p0
        } else {
            self.scale() * self.base.pmf(k)
        }
    }

    fn mean(&self) -> f64 {
        self.scale() * self.base.mean()
    }

    fn variance(&self) -> f64 {
        let second = self.scale() * (self.base.variance() + self.base.mean().powi(2));
        second - self.mean().powi(2)
    }

    fn panjer_ab(&self) -> Option<(f64, f64)> {
        self.base.panjer_ab()
    }

    /// `p0 + (1 - p0) (P(z) - q0) / (1 - q0)`.
    fn pgf(&self, z: f64) -> f64 {
        let q0 = self.base.pmf(0);
        self.p0 + (1.0 - self.p0) * (self.base.pgf(z) - q0) / (1.0 - q0)
    }

    fn pgf_complex(&self, z: (f64, f64)) -> (f64, f64) {
        let q0 = self.base.pmf(0);
        let s = self.scale();
        let (re, im) = self.base.pgf_complex(z);
        (self.p0 + s * (re - q0), s * im)
    }
}

/// The logarithmic count on `1, 2, …`: `P(N = k) = -p^k / (k ln(1 - p))`,
/// for `p` in `(0, 1)`. In the `(a, b, 1)` class with `a = p`, `b = -p`.
///
/// ```
/// use prospicio_prob::Counting;
/// use prospicio_prob::count_families::Logarithmic;
///
/// let n = Logarithmic::new(0.5).unwrap();
/// assert!((n.pmf(1) - 0.5 / 2f64.ln()).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Logarithmic {
    p: f64,
}

impl Logarithmic {
    /// The logarithmic count with `p` in `(0, 1)`.
    pub fn new(p: f64) -> Result<Self> {
        if !(p > 0.0 && p < 1.0) {
            return Err(Error::InvalidParameter {
                name: "p",
                value: p,
                reason: "must be in (0, 1)",
            });
        }
        Ok(Self { p })
    }

    /// The parameter `p`.
    pub fn p(&self) -> f64 {
        self.p
    }

    fn norm(&self) -> f64 {
        -1.0 / (-self.p).ln_1p()
    }
}

impl Counting for Logarithmic {
    fn pmf(&self, k: u64) -> f64 {
        if k == 0 {
            return 0.0;
        }
        let kf = k as f64;
        self.norm() * (kf * self.p.ln() - kf.ln()).exp()
    }

    fn mean(&self) -> f64 {
        self.norm() * self.p / (1.0 - self.p)
    }

    fn variance(&self) -> f64 {
        let second = self.norm() * self.p / (1.0 - self.p).powi(2);
        second - self.mean().powi(2)
    }

    fn panjer_ab(&self) -> Option<(f64, f64)> {
        Some((self.p, -self.p))
    }

    /// `ln(1 - p z) / ln(1 - p)`.
    fn pgf(&self, z: f64) -> f64 {
        (-self.p * z).ln_1p() / (-self.p).ln_1p()
    }

    fn pgf_complex(&self, z: (f64, f64)) -> (f64, f64) {
        let l = cln((1.0 - self.p * z.0, -self.p * z.1));
        cscale(l, 1.0 / (-self.p).ln_1p())
    }
}

/// The mixing distribution of a [`MixedPoisson`] count. `cv` is the
/// coefficient of variation of the whole mixing variable `Θ` (mean 1),
/// fixed part included, as in `aggregate`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mixing {
    /// Gamma: a negative binomial, or a Delaporte with a fixed part.
    Gamma { cv: f64 },
    /// Inverse Gaussian: the Poisson-inverse Gaussian, or its shifted
    /// version with a fixed part.
    InverseGaussian { cv: f64 },
}

/// A mixed Poisson count: Poisson with mean `λ Θ`, where the mixing
/// variable `Θ = c + (1 - c) G` has mean 1, a fixed part `c` in `[0, 1)`,
/// and `G` gamma (`c = 0`: negative binomial; `c > 0`: Delaporte) or
/// inverse Gaussian (`c = 0`: Poisson-inverse Gaussian; `c > 0`: the
/// shifted version, aggregate's `sig`). Mean `λ`, variance `λ + λ² cv²`
/// with `cv` the coefficient of variation of `Θ`, so `G`'s is
/// `cv / (1 - c)`. Mixing one `Θ` across several lines makes their counts
/// dependent, which is how `aggregate` correlates lines.
///
/// The probabilities come from the Poisson-inverse Gaussian recursion of
/// Willmot (1987) or the negative binomial's, convolved with
/// `Poisson(c λ)` when `c > 0`.
///
/// ```
/// use prospicio_prob::Counting;
/// use prospicio_prob::count_families::{MixedPoisson, Mixing};
///
/// let n = MixedPoisson::new(10.0, Mixing::InverseGaussian { cv: 0.5 }, 0.0).unwrap();
/// assert!((n.mean() - 10.0).abs() < 1e-10);
/// assert!((n.variance() - (10.0 + 100.0 * 0.25)).abs() < 1e-8);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MixedPoisson {
    lambda: f64,
    mixing: Mixing,
    shift: f64,
    table: Vec<f64>,
}

impl MixedPoisson {
    /// Mean `lambda`, the mixing of `G` and the fixed part `shift` of `Θ`.
    pub fn new(lambda: f64, mixing: Mixing, shift: f64) -> Result<Self> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(Error::InvalidParameter {
                name: "lambda",
                value: lambda,
                reason: "must be finite and positive",
            });
        }
        if !(0.0..1.0).contains(&shift) {
            return Err(Error::InvalidParameter {
                name: "shift",
                value: shift,
                reason: "must be in [0, 1)",
            });
        }
        let cv = match mixing {
            Mixing::Gamma { cv } | Mixing::InverseGaussian { cv } => cv,
        };
        if !(cv.is_finite() && cv > 0.0) {
            return Err(Error::InvalidParameter {
                name: "cv",
                value: cv,
                reason: "must be finite and positive",
            });
        }
        let l = (1.0 - shift) * lambda;
        // The coefficient of variation of G.
        let g_cv = cv / (1.0 - shift);
        let v2 = g_cv * g_cv;
        let sd = (lambda + lambda * lambda * cv * cv).sqrt();
        let at_least = (lambda + 10.0 * sd).ceil() as usize + 2;
        let mixed = match mixing {
            Mixing::Gamma { .. } => {
                // Negative binomial with r = 1 / cv² and mean l.
                let nb = NegativeBinomial::new(1.0 / v2, l * v2)?;
                extend_table(Vec::new(), at_least, |k, _| nb.pmf(k as u64))?
            }
            Mixing::InverseGaussian { .. } => {
                // Willmot (1987): with mixing variance β = cv² and
                // τ = 1 + 2 β l, p_0 = exp((1 - √τ) / β), p_1 = l p_0 / √τ,
                // p_k = (2 β l / τ)(1 - 3 / (2k)) p_{k-1} + l² / (τ k (k - 1)) p_{k-2}.
                let tau = 1.0 + 2.0 * v2 * l;
                let p0 = ((1.0 - tau.sqrt()) / v2).exp();
                let p1 = l * p0 / tau.sqrt();
                extend_table(vec![p0, p1], at_least, |k, p| {
                    let kf = k as f64;
                    2.0 * v2 * l / tau * (1.0 - 1.5 / kf) * p[k - 1]
                        + l * l / (tau * kf * (kf - 1.0)) * p[k - 2]
                })?
            }
        };
        let table = if shift > 0.0 {
            let pois = Poisson::new(shift * lambda)?;
            let pk: Vec<f64> = (0..mixed.len()).map(|k| pois.pmf(k as u64)).collect();
            let mut out = vec![0.0; mixed.len()];
            for (k, o) in out.iter_mut().enumerate() {
                *o = (0..=k).map(|j| pk[j] * mixed[k - j]).sum();
            }
            out
        } else {
            mixed
        };
        Ok(Self {
            lambda,
            mixing,
            shift,
            table,
        })
    }

    /// The mean `λ`.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The mixing of `G`.
    pub fn mixing(&self) -> Mixing {
        self.mixing
    }

    /// The fixed part `c` of `Θ`.
    pub fn shift(&self) -> f64 {
        self.shift
    }

    /// The moment generating function of `G` at complex `t`.
    fn mgf_g(&self, t: (f64, f64)) -> (f64, f64) {
        let scale = 1.0 / (1.0 - self.shift);
        match self.mixing {
            Mixing::Gamma { cv } => {
                // (1 - v² t)^(-1 / v²), v the cv of G
                let v2 = (cv * scale).powi(2);
                cexp(cscale(cln((1.0 - v2 * t.0, -v2 * t.1)), -1.0 / v2))
            }
            Mixing::InverseGaussian { cv } => {
                // exp((1 - √(1 - 2 v² t)) / v²)
                let v2 = (cv * scale).powi(2);
                let root = csqrt((1.0 - 2.0 * v2 * t.0, -2.0 * v2 * t.1));
                cexp(cscale((1.0 - root.0, -root.1), 1.0 / v2))
            }
        }
    }
}

impl Counting for MixedPoisson {
    fn pmf(&self, k: u64) -> f64 {
        self.table.get(k as usize).copied().unwrap_or(0.0)
    }

    fn mean(&self) -> f64 {
        self.lambda
    }

    fn variance(&self) -> f64 {
        let cv = match self.mixing {
            Mixing::Gamma { cv } | Mixing::InverseGaussian { cv } => cv,
        };
        self.lambda + (self.lambda * cv).powi(2)
    }

    fn panjer_ab(&self) -> Option<(f64, f64)> {
        None
    }

    /// `E[exp(λ Θ (z - 1))]`.
    fn pgf(&self, z: f64) -> f64 {
        self.pgf_complex((z, 0.0)).0
    }

    fn pgf_complex(&self, z: (f64, f64)) -> (f64, f64) {
        let t = cscale((z.0 - 1.0, z.1), self.lambda);
        let fixed = cexp(cscale(t, self.shift));
        cmul(fixed, self.mgf_g(cscale(t, 1.0 - self.shift)))
    }
}

/// A Poisson-stopped sum: `N = M_1 + … + M_K` with `K ~ Poisson(λ)` and
/// the `M_i` independent copies of a secondary count. Poisson secondary
/// counts give the Neyman type A, negative binomial ones aggregate's
/// `pascal`, zero-truncated geometric ones the Pólya-Aeppli. Probabilities
/// by the compound Poisson recursion
/// `p_k = (λ / k) Σ_{j=1..k} j q_j p_{k-j}`.
///
/// ```
/// use prospicio_prob::{Counting, Poisson};
/// use prospicio_prob::count_families::CompoundPoisson;
///
/// // Neyman type A: Poisson(2) clusters of Poisson(3) claims.
/// let n = CompoundPoisson::new(2.0, Poisson::new(3.0).unwrap()).unwrap();
/// assert!((n.mean() - 6.0).abs() < 1e-12);
/// assert!((n.variance() - 2.0 * (3.0 + 9.0)).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundPoisson<C> {
    lambda: f64,
    secondary: C,
    table: Vec<f64>,
}

impl<C: Counting> CompoundPoisson<C> {
    /// `lambda` Poisson-distributed clusters of `secondary` counts.
    pub fn new(lambda: f64, secondary: C) -> Result<Self> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(Error::InvalidParameter {
                name: "lambda",
                value: lambda,
                reason: "must be finite and positive",
            });
        }
        let mean = lambda * secondary.mean();
        let var = lambda * (secondary.variance() + secondary.mean().powi(2));
        let at_least = (mean + 10.0 * var.sqrt()).ceil() as usize + 2;
        let q0 = secondary.pmf(0);
        let mut q: Vec<f64> = vec![q0];
        let table = extend_table(vec![(-lambda * (1.0 - q0)).exp()], at_least, |k, p| {
            while q.len() <= k {
                q.push(secondary.pmf(q.len() as u64));
            }
            let s: f64 = (1..=k).map(|j| j as f64 * q[j] * p[k - j]).sum();
            lambda / k as f64 * s
        })?;
        Ok(Self {
            lambda,
            secondary,
            table,
        })
    }

    /// The Poisson rate of clusters.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The secondary count.
    pub fn secondary(&self) -> &C {
        &self.secondary
    }
}

impl<C: Counting> Counting for CompoundPoisson<C> {
    fn pmf(&self, k: u64) -> f64 {
        self.table.get(k as usize).copied().unwrap_or(0.0)
    }

    fn mean(&self) -> f64 {
        self.lambda * self.secondary.mean()
    }

    fn variance(&self) -> f64 {
        self.lambda * (self.secondary.variance() + self.secondary.mean().powi(2))
    }

    fn panjer_ab(&self) -> Option<(f64, f64)> {
        None
    }

    /// `exp(λ (Q(z) - 1))`.
    fn pgf(&self, z: f64) -> f64 {
        (self.lambda * (self.secondary.pgf(z) - 1.0)).exp()
    }

    fn pgf_complex(&self, z: (f64, f64)) -> (f64, f64) {
        let q = self.secondary.pgf_complex(z);
        cexp(cscale((q.0 - 1.0, q.1), self.lambda))
    }
}

/// An empirical count: `P(N = k) = probs[k]`.
///
/// ```
/// use prospicio_prob::Counting;
/// use prospicio_prob::count_families::EmpiricalCount;
///
/// let n = EmpiricalCount::new(vec![0.5, 0.25, 0.25]).unwrap();
/// assert_eq!(n.mean(), 0.75);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalCount {
    probs: Vec<f64>,
}

impl EmpiricalCount {
    /// Probabilities of `0, 1, …`: non-negative, summing to 1 within
    /// `1e-9` (then rescaled).
    pub fn new(probs: Vec<f64>) -> Result<Self> {
        if probs.is_empty() || probs.iter().any(|p| !(p.is_finite() && *p >= 0.0)) {
            return Err(Error::Data(
                "count probabilities must be non-negative and not empty".into(),
            ));
        }
        let sum: f64 = probs.iter().sum();
        if (sum - 1.0).abs() > 1e-9 {
            return Err(Error::Data(format!(
                "count probabilities sum to {sum}, not 1"
            )));
        }
        Ok(Self {
            probs: probs.into_iter().map(|p| p / sum).collect(),
        })
    }

    /// The probabilities.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }
}

impl Counting for EmpiricalCount {
    fn pmf(&self, k: u64) -> f64 {
        self.probs.get(k as usize).copied().unwrap_or(0.0)
    }

    fn mean(&self) -> f64 {
        table_moments(&self.probs).0
    }

    fn variance(&self) -> f64 {
        table_moments(&self.probs).1
    }

    fn panjer_ab(&self) -> Option<(f64, f64)> {
        None
    }

    fn pgf(&self, z: f64) -> f64 {
        self.probs.iter().rev().fold(0.0, |acc, p| acc * z + p)
    }

    fn pgf_complex(&self, z: (f64, f64)) -> (f64, f64) {
        self.probs.iter().rev().fold((0.0, 0.0), |acc, p| {
            let m = cmul(acc, z);
            (m.0 + p, m.1)
        })
    }
}

/// Any claim count, as one type: what the bindings and the compound
/// functions take when the count is chosen at run time.
#[derive(Debug, Clone, PartialEq)]
pub enum CountDist {
    Poisson(Poisson),
    NegativeBinomial(NegativeBinomial),
    Binomial(crate::counting::Binomial),
    Logarithmic(Logarithmic),
    ZeroModified(Box<ZeroModified<CountDist>>),
    MixedPoisson(MixedPoisson),
    CompoundPoisson(Box<CompoundPoisson<CountDist>>),
    Empirical(EmpiricalCount),
}

impl CountDist {
    fn inner(&self) -> &dyn Counting {
        match self {
            Self::Poisson(n) => n,
            Self::NegativeBinomial(n) => n,
            Self::Binomial(n) => n,
            Self::Logarithmic(n) => n,
            Self::ZeroModified(n) => n.as_ref(),
            Self::MixedPoisson(n) => n,
            Self::CompoundPoisson(n) => n.as_ref(),
            Self::Empirical(n) => n,
        }
    }
}

impl Counting for CountDist {
    fn pmf(&self, k: u64) -> f64 {
        self.inner().pmf(k)
    }
    fn mean(&self) -> f64 {
        self.inner().mean()
    }
    fn variance(&self) -> f64 {
        self.inner().variance()
    }
    fn panjer_ab(&self) -> Option<(f64, f64)> {
        self.inner().panjer_ab()
    }
    fn pgf(&self, z: f64) -> f64 {
        self.inner().pgf(z)
    }
    fn pgf_complex(&self, z: (f64, f64)) -> (f64, f64) {
        self.inner().pgf_complex(z)
    }
    fn cdf(&self, k: u64) -> f64 {
        self.inner().cdf(k)
    }
    fn quantile(&self, p: f64) -> Result<u64> {
        self.inner().quantile(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counting::Binomial;
    use prospicio_math::special::ln_gamma;

    fn ln_fact(k: u64) -> f64 {
        ln_gamma(k as f64 + 1.0)
    }

    fn check_table<C: Counting>(n: &C, name: &str) {
        let k_max = (n.mean() + 40.0 * n.variance().sqrt()) as u64 + 10;
        let p: Vec<f64> = (0..=k_max).map(|k| n.pmf(k)).collect();
        let (mean, var) = table_moments(&p);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{name}: sum");
        assert!(
            (mean - n.mean()).abs() < 1e-9 * n.mean().max(1.0),
            "{name}: mean {mean} vs {}",
            n.mean()
        );
        assert!(
            (var - n.variance()).abs() < 1e-8 * n.variance().max(1.0),
            "{name}: variance {var} vs {}",
            n.variance()
        );
        // The pgf agrees with the table, at a real and a complex point.
        for z in [0.3f64, 0.9] {
            let direct: f64 = p
                .iter()
                .enumerate()
                .map(|(k, q)| q * z.powi(k as i32))
                .sum();
            assert!((n.pgf(z) - direct).abs() < 1e-12, "{name}: pgf({z})");
        }
        let z = (0.6, 0.5);
        let mut zk = (1.0, 0.0);
        let mut direct = (0.0, 0.0);
        for q in &p {
            direct = (direct.0 + q * zk.0, direct.1 + q * zk.1);
            zk = cmul(zk, z);
        }
        let got = n.pgf_complex(z);
        assert!(
            (got.0 - direct.0).abs() < 1e-12 && (got.1 - direct.1).abs() < 1e-12,
            "{name}: complex pgf"
        );
    }

    #[test]
    fn every_family_is_a_consistent_distribution() {
        check_table(
            &ZeroModified::new(Poisson::new(2.5).unwrap(), 0.4).unwrap(),
            "zm poisson",
        );
        check_table(
            &ZeroModified::truncated(NegativeBinomial::new(2.0, 1.5).unwrap()).unwrap(),
            "zt nb",
        );
        check_table(
            &ZeroModified::new(Binomial::new(8, 0.3).unwrap(), 0.1).unwrap(),
            "zm binomial",
        );
        check_table(&Logarithmic::new(0.7).unwrap(), "logarithmic");
        check_table(
            &ZeroModified::new(Logarithmic::new(0.4).unwrap(), 0.3).unwrap(),
            "zm logarithmic",
        );
        for (mixing, shift) in [
            (Mixing::Gamma { cv: 0.4 }, 0.0),
            (Mixing::Gamma { cv: 0.4 }, 0.3),
            (Mixing::InverseGaussian { cv: 0.6 }, 0.0),
            (Mixing::InverseGaussian { cv: 0.6 }, 0.25),
        ] {
            check_table(
                &MixedPoisson::new(12.0, mixing, shift).unwrap(),
                &format!("{mixing:?} {shift}"),
            );
        }
        check_table(
            &CompoundPoisson::new(2.0, Poisson::new(3.0).unwrap()).unwrap(),
            "neyman a",
        );
        check_table(
            &CompoundPoisson::new(1.5, NegativeBinomial::new(2.0, 0.8).unwrap()).unwrap(),
            "pascal",
        );
        check_table(
            &EmpiricalCount::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
            "empirical",
        );
    }

    #[test]
    fn closed_forms() {
        // Gamma mixing with no shift is the negative binomial.
        let nb = NegativeBinomial::from_mean_variance(10.0, 10.0 + 100.0 * 0.16).unwrap();
        let mixed = MixedPoisson::new(10.0, Mixing::Gamma { cv: 0.4 }, 0.0).unwrap();
        for k in [0, 3, 10, 30] {
            assert!((mixed.pmf(k) - nb.pmf(k)).abs() < 1e-14);
        }
        // Neyman A at 0: exp(-λ (1 - e^-θ)).
        let ney = CompoundPoisson::new(2.0, Poisson::new(3.0).unwrap()).unwrap();
        assert!((ney.pmf(0) - (-2.0 * (1.0 - (-3f64).exp())).exp()).abs() < 1e-15);
        // Zero-truncated Poisson: e^-λ λ^k / (k! (1 - e^-λ)).
        let zt = ZeroModified::truncated(Poisson::new(2.0).unwrap()).unwrap();
        let want = (-2.0 + 3.0 * 2f64.ln() - ln_fact(3)).exp() / (1.0 - (-2f64).exp());
        assert!((zt.pmf(3) - want).abs() < 1e-15);
        assert_eq!(zt.panjer_ab(), Some((0.0, 2.0)));
        assert!(MixedPoisson::new(1.0, Mixing::Gamma { cv: 0.0 }, 0.0).is_err());
        assert!(ZeroModified::new(Poisson::new(1.0).unwrap(), 1.0).is_err());
        assert!(EmpiricalCount::new(vec![0.5, 0.4]).is_err());
    }
}
