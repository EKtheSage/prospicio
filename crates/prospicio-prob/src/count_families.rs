//! Claim counts beyond the `(a, b, 0)` class: zero-modified and
//! zero-truncated counts, the logarithmic, mixed Poisson counts, compound
//! (Poisson-stopped) counts and empirical counts. These are the families of
//! Mildenhall's `aggregate` (`zm`, `zt`, `logarithmic`, `mixed gamma`,
//! `delaporte`, `ig`, `sig`, `beta`, `sichel`, `sichel.gamma`, `sichel.ig`, `neymana`, `pascal`, `dfreq`) and of Klugman,
//! Panjer and Willmot, *Loss Models*, chapters 6 and 7.
//!
//! Counts outside the `(a, b, 1)` class keep their probabilities in a table
//! computed on construction out to where the remaining mass is below
//! `1e-16`, so `pmf` is a lookup.

use prospicio_core::{Error, Result};

use prospicio_math::roots::{bisect, bisect_log};
use prospicio_math::special::{ln_bessel_k, ln_gamma};

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

    /// The base's moment scaled like its probabilities above 0.
    fn third_moment(&self) -> f64 {
        self.scale() * self.base.third_moment()
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
    /// Beta: the Poisson mean `λ Θ` is `max` times a beta variable, so it
    /// never exceeds `max`, which must exceed `λ (1 + cv²)`. `aggregate`'s
    /// `mixed beta cv max`. With no fixed part.
    Beta { cv: f64, max: f64 },
    /// Generalized inverse Gaussian with the given shape (the Sichel
    /// count), its other two parameters set by the mean 1 and `cv`; shape
    /// `-1/2` is the inverse Gaussian, `1/2` the reciprocal inverse
    /// Gaussian; a positive shape needs `cv² < 1 / shape` and one below
    /// `-2` needs `cv² < 1 / (-2 - shape)`.
    /// `aggregate`'s `mixed sichel cv shape`. With no fixed part.
    Sichel { cv: f64, shape: f64 },
    /// Generalized inverse Gaussian with the mean 1, `cv` and the given
    /// skewness, its shape solved: [`Mixing::sichel_gamma`] and
    /// [`Mixing::sichel_ig`] give `aggregate`'s `sichel.gamma` and
    /// `sichel.ig`. The skewness lies between the gamma's `2 cv` and, for
    /// `cv < 1`, the inverse gamma's `4 cv / (1 - cv²)`. With no fixed part.
    SichelSkewness { cv: f64, skewness: f64 },
}

impl Mixing {
    /// The coefficient of variation of `Θ`.
    pub fn cv(&self) -> f64 {
        match *self {
            Mixing::Gamma { cv }
            | Mixing::InverseGaussian { cv }
            | Mixing::Beta { cv, .. }
            | Mixing::Sichel { cv, .. }
            | Mixing::SichelSkewness { cv, .. } => cv,
        }
    }

    /// `aggregate`'s `sichel.gamma cv f`: the generalized inverse Gaussian
    /// with the mean, `cv` and skewness of the Delaporte's mixing
    /// variable with fixed part `f`, `2 cv / (1 - f)`.
    pub fn sichel_gamma(cv: f64, f: f64) -> Self {
        Mixing::SichelSkewness {
            cv,
            skewness: 2.0 * cv / (1.0 - f),
        }
    }

    /// `aggregate`'s `sichel.ig cv f`: the generalized inverse Gaussian
    /// with the mean, `cv` and skewness of the shifted inverse Gaussian's
    /// mixing variable with fixed part `f`, `3 cv / (1 - f)`.
    pub fn sichel_ig(cv: f64, f: f64) -> Self {
        Mixing::SichelSkewness {
            cv,
            skewness: 3.0 * cv / (1.0 - f),
        }
    }
}

/// The parameters of a generalized inverse Gaussian mixing variable with
/// mean 1: density proportional to `x^(shape - 1) exp(-ω (x / η + η / x) / 2)`,
/// so `E[Θ^r] = η^r K_{shape + r}(ω) / K_shape(ω)` with `K` the modified
/// Bessel function of the second kind. `aggregate` writes `μ = η` and
/// `β = η / ω`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gig {
    /// The shape `λ`.
    pub shape: f64,
    /// The concentration `ω`.
    pub omega: f64,
    /// The scale `η`.
    pub eta: f64,
}

impl Gig {
    /// The scale `η = K_λ(ω) / K_{λ+1}(ω)` that makes the mean 1, and
    /// `E[Θ²]` and `E[Θ³]` there, each ratio of Bessel functions taken
    /// from their logarithms: the recurrence `K_{ν+1} = K_{ν-1} + (2ν / ω) K_ν`
    /// cancels at small `ω`.
    fn moments(shape: f64, omega: f64) -> (f64, f64, f64) {
        let k0 = ln_bessel_k(shape, omega);
        let ln_eta = k0 - ln_bessel_k(shape + 1.0, omega);
        let m = |r: f64| (r * ln_eta + ln_bessel_k(shape + r, omega) - k0).exp();
        (ln_eta.exp(), m(2.0), m(3.0))
    }

    /// `ω` for a shape and coefficient of variation, by bisection on a log
    /// scale (the variance falls as `ω` grows).
    fn solve_omega(shape: f64, cv: f64) -> Result<f64> {
        let target = 1.0 + cv * cv;
        let (lo, hi) = (1e-10, 1e10);
        let at_lo = Self::moments(shape, lo).1;
        if at_lo.is_nan() || at_lo <= target {
            return Err(Error::InvalidParameter {
                name: "cv",
                value: cv,
                reason: "is too large for the Sichel shape: cv² must be below 1 / shape for a \
                         positive shape and below 1 / (-2 - shape) below -2",
            });
        }
        Ok(bisect_log(lo, hi, |w| Self::moments(shape, w).1 > target))
    }

    /// The mixing with mean 1, `cv` and shape.
    fn with_shape(cv: f64, shape: f64) -> Result<Self> {
        if !shape.is_finite() {
            return Err(Error::InvalidParameter {
                name: "shape",
                value: shape,
                reason: "must be finite",
            });
        }
        let omega = Self::solve_omega(shape, cv)?;
        let eta = Self::moments(shape, omega).0;
        Ok(Self { shape, omega, eta })
    }

    /// The skewness at mean 1, `cv` and shape.
    fn skewness(shape: f64, cv: f64) -> Result<f64> {
        let omega = Self::solve_omega(shape, cv)?;
        let (_, m2, m3) = Self::moments(shape, omega);
        Ok((m3 - 3.0 * m2 + 2.0) / (m2 - 1.0).powf(1.5))
    }

    /// The mixing with mean 1, `cv` and skewness: the shape lies in
    /// `(-2 - 1 / cv², 1 / cv²)`, where the variance is finite, and the
    /// skewness falls as it grows, from the inverse gamma's at the bottom
    /// to the gamma's at the top.
    fn with_skewness(cv: f64, skewness: f64) -> Result<Self> {
        let bad = |reason| Error::InvalidParameter {
            name: "skewness",
            value: skewness,
            reason,
        };
        if !(skewness.is_finite() && skewness > 2.0 * cv) {
            return Err(bad("must exceed the gamma's 2 cv"));
        }
        let hi = (1.0 - 1e-6) / (cv * cv);
        if Self::skewness(hi, cv)? >= skewness {
            return Err(bad("is too close to the gamma's 2 cv to solve"));
        }
        // Below shape -2 the variance is finite only for cv² < 1 / (-2 - shape).
        let lo = -2.0 - (1.0 - 1e-6) / (cv * cv);
        if Self::skewness(lo, cv)? <= skewness {
            return Err(bad("is beyond the inverse gamma's 4 cv / (1 - cv²)"));
        }
        let mut failed = None;
        let shape = bisect(lo, hi, |l| match Self::skewness(l, cv) {
            Ok(s) => s > skewness,
            Err(e) => {
                failed = Some(e);
                false
            }
        });
        if let Some(e) = failed {
            return Err(e);
        }
        Self::with_shape(cv, shape)
    }
}

/// `P(N = k)` for a Poisson whose mean is `max` times a `Beta(a, b)`
/// variable: `max^k / k! · B(a + k, b) / B(a, b) · 1F1(a + k; a + b + k; -max)`,
/// with Kummer's transformation `1F1(α; γ; -x) = e^(-x) 1F1(γ - α; γ; x)`,
/// whose series has positive terms, summed on a log scale.
fn beta_poisson_pmf(a: f64, b: f64, max: f64, k: usize) -> f64 {
    let kf = k as f64;
    let ln_front =
        kf * max.ln() - ln_gamma(kf + 1.0) + ln_gamma(a + kf) - ln_gamma(a + b + kf) - ln_gamma(a)
            + ln_gamma(a + b)
            - max;
    // Σ_j (b)_j / (a + b + k)_j max^j / j!, scaled by its largest term.
    let mut ln_term = 0.0f64;
    let mut ln_top = 0.0f64;
    let mut sum = 1.0f64;
    let mut j = 0.0f64;
    loop {
        ln_term += ((b + j) * max / ((a + b + kf + j) * (j + 1.0))).ln();
        j += 1.0;
        if ln_term > ln_top {
            sum = sum * (ln_top - ln_term).exp() + 1.0;
            ln_top = ln_term;
        } else {
            let r = (ln_term - ln_top).exp();
            sum += r;
            if r < 1e-18 * sum {
                break;
            }
        }
    }
    (ln_front + ln_top + sum.ln()).exp()
}

/// A mixed Poisson count: Poisson with mean `λ Θ`, where the mixing
/// variable `Θ = c + (1 - c) G` has mean 1, a fixed part `c` in `[0, 1)`,
/// and `G` gamma (`c = 0`: negative binomial; `c > 0`: Delaporte) or
/// inverse Gaussian (`c = 0`: Poisson-inverse Gaussian; `c > 0`: the
/// shifted version, aggregate's `sig`), or, with no fixed part, beta or
/// generalized inverse Gaussian (the Sichel count). Mean `λ`, variance
/// `λ + λ² cv²` with `cv` the coefficient of variation of `Θ`, so `G`'s
/// is `cv / (1 - c)`. Mixing one `Θ` across several lines makes their
/// counts dependent, which is how `aggregate` correlates lines.
///
/// The probabilities come from the Poisson-inverse Gaussian recursion of
/// Willmot (1987) or the negative binomial's, convolved with
/// `Poisson(c λ)` when `c > 0`; for the beta from Kummer's series of the
/// confluent hypergeometric function; for the Sichel count from the
/// Bessel functions' recurrence in the order.
///
/// ```
/// use prospicio_prob::Counting;
/// use prospicio_prob::count_families::{MixedPoisson, Mixing};
///
/// let n = MixedPoisson::new(10.0, Mixing::InverseGaussian { cv: 0.5 }, 0.0).unwrap();
/// assert!((n.mean() - 10.0).abs() < 1e-10);
/// assert!((n.variance() - (10.0 + 100.0 * 0.25)).abs() < 1e-8);
///
/// // The Sichel count with shape -1/2 is the Poisson-inverse Gaussian.
/// let s = MixedPoisson::new(10.0, Mixing::Sichel { cv: 0.5, shape: -0.5 }, 0.0).unwrap();
/// assert!((s.pmf(7) - n.pmf(7)).abs() < 1e-14);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MixedPoisson {
    lambda: f64,
    mixing: Mixing,
    shift: f64,
    gig: Option<Gig>,
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
        let cv = mixing.cv();
        if !(cv.is_finite() && cv > 0.0) {
            return Err(Error::InvalidParameter {
                name: "cv",
                value: cv,
                reason: "must be finite and positive",
            });
        }
        let unshifted = matches!(
            mixing,
            Mixing::Beta { .. } | Mixing::Sichel { .. } | Mixing::SichelSkewness { .. }
        );
        if unshifted && shift != 0.0 {
            return Err(Error::InvalidParameter {
                name: "shift",
                value: shift,
                reason: "must be 0 for the beta and Sichel mixings",
            });
        }
        let l = (1.0 - shift) * lambda;
        // The coefficient of variation of G.
        let g_cv = cv / (1.0 - shift);
        let v2 = g_cv * g_cv;
        let sd = (lambda + lambda * lambda * cv * cv).sqrt();
        let at_least = (lambda + 10.0 * sd).ceil() as usize + 2;
        let mut gig = None;
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
            Mixing::Beta { max, .. } => {
                // λ Θ = max B with B ~ Beta(a, b), E[B] = λ / max and
                // cv[B] = cv, as aggregate solves them.
                let c = v2;
                if !(max.is_finite() && max > lambda * (1.0 + c)) {
                    return Err(Error::InvalidParameter {
                        name: "max",
                        value: max,
                        reason: "must exceed lambda (1 + cv²)",
                    });
                }
                let a = (max - lambda * (1.0 + c)) / (c * max);
                let b = a * (max - lambda) / lambda;
                extend_table(Vec::new(), at_least, |k, _| beta_poisson_pmf(a, b, max, k))?
            }
            Mixing::Sichel { .. } | Mixing::SichelSkewness { .. } => {
                let g = match mixing {
                    Mixing::Sichel { shape, .. } => Gig::with_shape(cv, shape)?,
                    Mixing::SichelSkewness { skewness, .. } => Gig::with_skewness(cv, skewness)?,
                    _ => unreachable!(),
                };
                gig = Some(g);
                // P(N = k) = (λ η)^k / k! (1 + 2 λ η / ω)^(-(k + shape) / 2)
                //            K_{k + shape}(ω s) / K_shape(ω), s = √(1 + 2 λ η / ω),
                // the Bessel ratios R_k = K_{shape+k+1} / K_{shape+k} at ω s by
                // R_k = 1 / R_{k-1} + 2 (shape + k) / (ω s).
                let s = (1.0 + 2.0 * lambda * g.eta / g.omega).sqrt();
                let x = g.omega * s;
                let ln_k0 = ln_bessel_k(g.shape, x);
                let p0 = (-g.shape * s.ln() + ln_k0 - ln_bessel_k(g.shape, g.omega)).exp();
                let mut ratio = (ln_bessel_k(g.shape + 1.0, x) - ln_k0).exp();
                let step = lambda * g.eta / s;
                extend_table(vec![p0], at_least, |k, p| {
                    let kf = k as f64;
                    if k > 1 {
                        ratio = 1.0 / ratio + 2.0 * (g.shape + kf - 1.0) / x;
                    }
                    p[k - 1] * step / kf * ratio
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
            gig,
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

    /// The solved generalized inverse Gaussian of a Sichel mixing.
    pub fn gig(&self) -> Option<Gig> {
        self.gig
    }

    /// The moment generating function of `G` at complex `t`, for the
    /// mixings with a closed form.
    fn mgf_g(&self, t: (f64, f64)) -> Option<(f64, f64)> {
        let scale = 1.0 / (1.0 - self.shift);
        match self.mixing {
            Mixing::Gamma { cv } => {
                // (1 - v² t)^(-1 / v²), v the cv of G
                let v2 = (cv * scale).powi(2);
                Some(cexp(cscale(cln((1.0 - v2 * t.0, -v2 * t.1)), -1.0 / v2)))
            }
            Mixing::InverseGaussian { cv } => {
                // exp((1 - √(1 - 2 v² t)) / v²)
                let v2 = (cv * scale).powi(2);
                let root = csqrt((1.0 - 2.0 * v2 * t.0, -2.0 * v2 * t.1));
                Some(cexp(cscale((1.0 - root.0, -root.1), 1.0 / v2)))
            }
            _ => None,
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
        self.lambda + (self.lambda * self.mixing.cv()).powi(2)
    }

    fn panjer_ab(&self) -> Option<(f64, f64)> {
        None
    }

    /// `E[exp(λ Θ (z - 1))]`.
    fn pgf(&self, z: f64) -> f64 {
        self.pgf_complex((z, 0.0)).0
    }

    /// In closed form for the gamma and inverse Gaussian; for the beta and
    /// Sichel mixings, the probability table summed by Horner's rule,
    /// exact to its `1e-16` of missing mass for `|z| ≤ 1`.
    fn pgf_complex(&self, z: (f64, f64)) -> (f64, f64) {
        let t = cscale((z.0 - 1.0, z.1), self.lambda);
        match self.mgf_g(cscale(t, 1.0 - self.shift)) {
            Some(g) => cmul(cexp(cscale(t, self.shift)), g),
            None => self.table.iter().rev().fold((0.0, 0.0), |acc, &p| {
                let m = cmul(acc, z);
                (m.0 + p, m.1)
            }),
        }
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

    /// A fixed count, `N = n` with probability 1, `aggregate`'s `fixed`.
    ///
    /// ```
    /// use prospicio_prob::Counting;
    /// use prospicio_prob::count_families::EmpiricalCount;
    ///
    /// let n = EmpiricalCount::fixed(3);
    /// assert_eq!((n.mean(), n.variance()), (3.0, 0.0));
    /// ```
    pub fn fixed(n: u64) -> Self {
        let mut probs = vec![0.0; n as usize + 1];
        probs[n as usize] = 1.0;
        Self { probs }
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
    fn third_moment(&self) -> f64 {
        self.inner().third_moment()
    }
    fn third_cumulant(&self) -> f64 {
        self.inner().third_cumulant()
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
        // Out past 40 standard deviations and then the long tails (a Sichel
        // count with a negative shape), until the probabilities vanish.
        let mut k_max = (n.mean() + 40.0 * n.variance().sqrt()) as u64 + 10;
        while n.pmf(k_max) > 1e-20 {
            k_max += 100;
        }
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
            (Mixing::Beta { cv: 0.3, max: 30.0 }, 0.0),
            (Mixing::Beta { cv: 0.5, max: 20.0 }, 0.0),
            (
                Mixing::Sichel {
                    cv: 0.4,
                    shape: 1.5,
                },
                0.0,
            ),
            (
                Mixing::Sichel {
                    cv: 0.6,
                    shape: -2.0,
                },
                0.0,
            ),
            (Mixing::sichel_gamma(0.4, 0.3), 0.0),
            (Mixing::sichel_ig(0.5, 0.3), 0.0),
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
        // The Sichel count with shape -1/2 is the Poisson-inverse Gaussian.
        let ig = MixedPoisson::new(10.0, Mixing::InverseGaussian { cv: 0.5 }, 0.0).unwrap();
        let sichel = MixedPoisson::new(
            10.0,
            Mixing::Sichel {
                cv: 0.5,
                shape: -0.5,
            },
            0.0,
        )
        .unwrap();
        for k in [0, 5, 10, 40] {
            assert!((sichel.pmf(k) - ig.pmf(k)).abs() < 1e-14);
        }
        let g = sichel.gig().unwrap();
        assert!((g.omega - 4.0).abs() < 1e-12 && (g.eta - 1.0).abs() < 1e-12);
        // The matched Sichel has the Delaporte mixing's skewness: the count's
        // third cumulant is λ + 3 λ² cv² + λ³ cv³ skew.
        let (lambda, cv, f) = (10.0, 0.4, 0.3);
        let m = MixedPoisson::new(lambda, Mixing::sichel_gamma(cv, f), 0.0).unwrap();
        let skew = 2.0 * cv / (1.0 - f);
        let want = lambda + 3.0 * lambda * lambda * cv * cv + lambda.powi(3) * cv.powi(3) * skew;
        assert!((m.third_cumulant() / want - 1.0).abs() < 1e-10);
        // What the beta and Sichel mixings reject.
        let beta = Mixing::Beta { cv: 0.3, max: 30.0 };
        assert!(MixedPoisson::new(10.0, beta, 0.2).is_err());
        assert!(MixedPoisson::new(10.0, Mixing::Beta { cv: 0.3, max: 10.5 }, 0.0).is_err());
        assert!(
            MixedPoisson::new(
                10.0,
                Mixing::Sichel {
                    cv: 0.9,
                    shape: 2.0
                },
                0.0
            )
            .is_err()
        );
        assert!(
            MixedPoisson::new(
                10.0,
                Mixing::Sichel {
                    cv: 0.9,
                    shape: -4.0
                },
                0.0
            )
            .is_err()
        );
        let skew = |s| Mixing::SichelSkewness {
            cv: 0.4,
            skewness: s,
        };
        assert!(MixedPoisson::new(10.0, skew(0.7), 0.0).is_err());
        assert!(MixedPoisson::new(10.0, skew(2.0), 0.0).is_err());
        assert!(ZeroModified::new(Poisson::new(1.0).unwrap(), 1.0).is_err());
        assert!(EmpiricalCount::new(vec![0.5, 0.4]).is_err());
    }
}
