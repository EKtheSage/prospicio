//! Laws of Lévy processes at time one with closed-form characteristic
//! functions, whose distributions the characteristic-function engine
//! computes on a grid ([`crate::chf::invert`], `aggregate`'s
//! `norminvgauss`, `variancegamma` and `temperedstable`, #780).
//!
//! Each law gives its characteristic function on the closed upper half
//! plane, its cumulant generating function `K` and the open interval where
//! `K` is finite, and its first three moments. All three have light tails
//! on both sides.

use prospicio_core::{Error, Result};
use prospicio_math::special::ln_gamma;
use rustfft::num_complex::Complex64;

use std::f64::consts::PI;

fn positive(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name,
            value,
            reason: "must be finite and positive",
        })
    }
}

fn finite(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name,
            value,
            reason: "must be finite",
        })
    }
}

/// `Γ(x)` for any `x` that is not 0 or a negative integer, by reflection
/// below 1/2.
fn gamma_fn(x: f64) -> f64 {
    if x >= 0.5 {
        ln_gamma(x).exp()
    } else {
        PI / ((PI * x).sin() * ln_gamma(1.0 - x).exp())
    }
}

/// The normal inverse Gaussian law `NIG(α, β, δ, μ)`, Barndorff-Nielsen
/// (1997): with `|β| < α`, `δ > 0` and `γ = √(α² − β²)`,
/// `K(s) = μs + δ(γ − √(α² − (β + s)²))` for `−α − β < s < α − β`.
///
/// SciPy's `norminvgauss(a, b, loc, scale)` is this law with `α = a /
/// scale`, `β = b / scale`, `δ = scale`, `μ = loc`.
///
/// ```
/// use prospicio_prob::levy::NormalInverseGaussian;
///
/// let d = NormalInverseGaussian::new(2.0, 0.5, 1.0, 0.0).unwrap();
/// let g = (4.0f64 - 0.25).sqrt();
/// assert!((d.mean() - 0.5 / g).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalInverseGaussian {
    alpha: f64,
    beta: f64,
    delta: f64,
    mu: f64,
}

impl NormalInverseGaussian {
    /// The law with tail heaviness `alpha`, skewness `beta` (`|beta| <
    /// alpha`), scale `delta` and location `mu`.
    pub fn new(alpha: f64, beta: f64, delta: f64, mu: f64) -> Result<Self> {
        positive("alpha", alpha)?;
        finite("beta", beta)?;
        positive("delta", delta)?;
        finite("mu", mu)?;
        if beta.abs() >= alpha {
            return Err(Error::InvalidParameter {
                name: "beta",
                value: beta,
                reason: "must be below alpha in absolute value",
            });
        }
        Ok(Self {
            alpha,
            beta,
            delta,
            mu,
        })
    }

    /// `α`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// `β`.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// `δ`.
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// `μ`.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    fn gamma(&self) -> f64 {
        (self.alpha * self.alpha - self.beta * self.beta).sqrt()
    }

    /// `φ(s) = exp K(is)` on the principal square root.
    pub fn chf(&self, s: Complex64) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        let (a, b, d, m) = (self.alpha, self.beta, self.delta, self.mu);
        let inner = (a * a - (b + i * s) * (b + i * s)).sqrt();
        (i * m * s + d * (self.gamma() - inner)).exp()
    }

    /// `K(s)`, NaN outside [`Self::cgf_domain`].
    pub fn cgf(&self, s: f64) -> f64 {
        let (lo, hi) = self.cgf_domain();
        if !(s > lo && s < hi) {
            return f64::NAN;
        }
        let (a, b) = (self.alpha, self.beta);
        self.mu * s + self.delta * (self.gamma() - (a * a - (b + s) * (b + s)).sqrt())
    }

    /// `(−α − β, α − β)`.
    pub fn cgf_domain(&self) -> (f64, f64) {
        (-self.alpha - self.beta, self.alpha - self.beta)
    }

    /// `μ + δβ/γ`.
    pub fn mean(&self) -> f64 {
        self.mu + self.delta * self.beta / self.gamma()
    }

    /// `δα²/γ³`.
    pub fn variance(&self) -> f64 {
        self.delta * self.alpha * self.alpha / self.gamma().powi(3)
    }

    /// `3β / (α √(δγ))`.
    pub fn skewness(&self) -> f64 {
        3.0 * self.beta / (self.alpha * (self.delta * self.gamma()).sqrt())
    }
}

/// The variance gamma law of Madan, Carr and Chang (1998), `VG(σ, ν, θ)`
/// with a location `μ`: `φ(s) = e^{iμs} (1 − iθνs + σ²νs²/2)^(−1/ν)`,
/// `K(s) = μs − ln(1 − θνs − σ²νs²/2) / ν` between the roots of the
/// quadratic.
///
/// At `θ = 0`, `ν = 1` it is the Laplace law with scale `σ/√2`.
/// `|φ|` decays only as `t^(−2/ν)`, so a large `ν` needs a small step.
///
/// ```
/// use prospicio_prob::levy::VarianceGamma;
///
/// let d = VarianceGamma::new(0.2, 0.5, -0.1, 0.0).unwrap();
/// assert!((d.variance() - (0.04 + 0.5 * 0.01)).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarianceGamma {
    sigma: f64,
    nu: f64,
    theta: f64,
    mu: f64,
}

impl VarianceGamma {
    /// The law with volatility `sigma`, variance rate `nu` (both
    /// positive), drift `theta` and location `mu`.
    pub fn new(sigma: f64, nu: f64, theta: f64, mu: f64) -> Result<Self> {
        positive("sigma", sigma)?;
        positive("nu", nu)?;
        finite("theta", theta)?;
        finite("mu", mu)?;
        Ok(Self {
            sigma,
            nu,
            theta,
            mu,
        })
    }

    /// `σ`.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// `ν`.
    pub fn nu(&self) -> f64 {
        self.nu
    }

    /// `θ`.
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// `μ`.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// `e^{iμs} (1 − iθνs + σ²νs²/2)^(−1/ν)` on the principal power.
    pub fn chf(&self, s: Complex64) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        let base = 1.0 - i * self.theta * self.nu * s + 0.5 * self.sigma.powi(2) * self.nu * s * s;
        (i * self.mu * s).exp() * base.powf(-1.0 / self.nu)
    }

    /// `K(s)`, NaN outside [`Self::cgf_domain`].
    pub fn cgf(&self, s: f64) -> f64 {
        let (lo, hi) = self.cgf_domain();
        if !(s > lo && s < hi) {
            return f64::NAN;
        }
        let q = 1.0 - self.theta * self.nu * s - 0.5 * self.sigma.powi(2) * self.nu * s * s;
        self.mu * s - q.ln() / self.nu
    }

    /// The roots of `1 − θνs − σ²νs²/2`.
    pub fn cgf_domain(&self) -> (f64, f64) {
        let disc =
            (self.theta.powi(2) * self.nu.powi(2) + 2.0 * self.sigma.powi(2) * self.nu).sqrt();
        let denom = self.sigma.powi(2) * self.nu;
        (
            (-self.theta * self.nu - disc) / denom,
            (-self.theta * self.nu + disc) / denom,
        )
    }

    /// `μ + θ`.
    pub fn mean(&self) -> f64 {
        self.mu + self.theta
    }

    /// `σ² + νθ²`.
    pub fn variance(&self) -> f64 {
        self.sigma.powi(2) + self.nu * self.theta.powi(2)
    }

    /// `(2θ³ν² + 3σ²θν) / variance^(3/2)`.
    pub fn skewness(&self) -> f64 {
        let k3 = 2.0 * self.theta.powi(3) * self.nu.powi(2)
            + 3.0 * self.sigma.powi(2) * self.theta * self.nu;
        k3 / self.variance().powf(1.5)
    }
}

/// The CGMY tempered stable law of Carr, Geman, Madan and Yor (2002) with a
/// location `μ`: with `C, G, M > 0` and `Y < 2`, `Y ∉ {0, 1}`,
/// `K(s) = μs + CΓ(−Y)((M − s)^Y − M^Y + (G + s)^Y − G^Y)` for
/// `−G < s < M`, and cumulants `κ_n = CΓ(n − Y)(M^(Y−n) + (−1)^n G^(Y−n))`.
///
/// ```
/// use prospicio_prob::levy::TemperedStable;
///
/// let d = TemperedStable::new(1.0, 5.0, 10.0, 0.5, 0.0).unwrap();
/// assert!(d.variance() > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperedStable {
    c: f64,
    g: f64,
    m: f64,
    y: f64,
    mu: f64,
}

impl TemperedStable {
    /// The law with activity `c`, left and right tempering `g` and `m`
    /// (all positive), index `y` below 2 and not 0 or 1, and location `mu`.
    pub fn new(c: f64, g: f64, m: f64, y: f64, mu: f64) -> Result<Self> {
        positive("C", c)?;
        positive("G", g)?;
        positive("M", m)?;
        finite("mu", mu)?;
        if !y.is_finite() || y >= 2.0 || y == 0.0 || y == 1.0 {
            return Err(Error::InvalidParameter {
                name: "Y",
                value: y,
                reason: "must be below 2 and not 0 or 1",
            });
        }
        Ok(Self { c, g, m, y, mu })
    }

    /// `C`.
    pub fn c(&self) -> f64 {
        self.c
    }

    /// `G`.
    pub fn g(&self) -> f64 {
        self.g
    }

    /// `M`.
    pub fn m(&self) -> f64 {
        self.m
    }

    /// `Y`.
    pub fn y(&self) -> f64 {
        self.y
    }

    /// `μ`.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    fn k(&self, s: Complex64) -> Complex64 {
        let (c, g, m, y) = (self.c, self.g, self.m, self.y);
        let m_c = Complex64::new(m, 0.0);
        let g_c = Complex64::new(g, 0.0);
        self.mu * s
            + c * gamma_fn(-y) * ((m_c - s).powf(y) - m.powf(y) + (g_c + s).powf(y) - g.powf(y))
    }

    /// `φ(s) = exp K(is)` on the principal powers.
    pub fn chf(&self, s: Complex64) -> Complex64 {
        self.k(Complex64::new(0.0, 1.0) * s).exp()
    }

    /// `K(s)`, NaN outside `(−G, M)`.
    pub fn cgf(&self, s: f64) -> f64 {
        if !(s > -self.g && s < self.m) {
            return f64::NAN;
        }
        self.k(Complex64::new(s, 0.0)).re
    }

    /// `(−G, M)`.
    pub fn cgf_domain(&self) -> (f64, f64) {
        (-self.g, self.m)
    }

    /// The `n`-th cumulant, `CΓ(n − Y)(M^(Y−n) + (−1)^n G^(Y−n))`.
    pub fn cumulant(&self, n: i32) -> f64 {
        let (c, g, m, y) = (self.c, self.g, self.m, self.y);
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        c * gamma_fn(n as f64 - y) * (m.powf(y - n as f64) + sign * g.powf(y - n as f64))
    }

    /// `μ + κ₁`.
    pub fn mean(&self) -> f64 {
        self.mu + self.cumulant(1)
    }

    /// `κ₂`.
    pub fn variance(&self) -> f64 {
        self.cumulant(2)
    }

    /// `κ₃ / κ₂^(3/2)`.
    pub fn skewness(&self) -> f64 {
        self.cumulant(3) / self.cumulant(2).powf(1.5)
    }
}

/// The Tweedie law at any power `p` outside `(0, 1]`, in its
/// reproductive form (mean `μ`, dispersion `φ`, variance `φ μ^p`):
/// Jørgensen's exponential dispersion model with `λ = 1/φ`, canonical
/// `θ = (α − 1) μ^(1−p)`, `α = (p − 2)/(p − 1)` and
/// `K(s) = λ(κ(θ + s/λ) − κ(θ))`, `κ(θ) = ((α − 1)/α) (θ/(α − 1))^α`
/// (`−ln(−θ)` at `p = 2`) (Jørgensen 1997, eq. 4.13).
///
/// `p = 0` is the normal, `p < 0` the extreme stable laws on the whole
/// line, `1 < p < 2` the compound Poisson-gamma with its atom `e^(−λ_N)`
/// at 0 (as [`crate::Tweedie`]), `p = 2` the gamma, `p = 3` the inverse
/// Gaussian and `p > 2` the positive extreme stable laws, each tilted by
/// its own `θ` so that both tails are light. `p = 1` is a lattice law and
/// `p = ∞` has a power tail; both are refused.
///
/// ```
/// use prospicio_prob::levy::TweediePower;
///
/// let d = TweediePower::new(3.0, 10.0, 0.5).unwrap();
/// assert!((d.variance() - 0.5 * 1000.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweediePower {
    power: f64,
    mean: f64,
    dispersion: f64,
}

impl TweediePower {
    /// The law with power `power` (at most 0, or above 1), mean `mean`
    /// (positive) and dispersion `dispersion` (positive).
    pub fn new(power: f64, mean: f64, dispersion: f64) -> Result<Self> {
        if !power.is_finite() || (power > 0.0 && power <= 1.0) {
            return Err(Error::InvalidParameter {
                name: "power",
                value: power,
                reason: "must be finite and at most 0 or above 1: p = 1 is a lattice law \
                         and no Tweedie law has 0 < p < 1",
            });
        }
        positive("mean", mean)?;
        positive("dispersion", dispersion)?;
        Ok(Self {
            power,
            mean,
            dispersion,
        })
    }

    /// The law with power `power`, mean `mean` and coefficient of
    /// variation `cv`: dispersion `cv² μ^(2−p)`.
    pub fn from_mean_cv(power: f64, mean: f64, cv: f64) -> Result<Self> {
        positive("cv", cv)?;
        Self::new(power, mean, cv * cv * mean.powf(2.0 - power))
    }

    /// `p`.
    pub fn power(&self) -> f64 {
        self.power
    }

    /// `μ`.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// `φ`.
    pub fn dispersion(&self) -> f64 {
        self.dispersion
    }

    fn alpha(&self) -> f64 {
        (self.power - 2.0) / (self.power - 1.0)
    }

    fn theta(&self) -> f64 {
        (self.alpha() - 1.0) * self.mean.powf(1.0 - self.power)
    }

    fn kappa(&self, theta: Complex64) -> Complex64 {
        if self.power == 2.0 {
            return -(-theta).ln();
        }
        let a = self.alpha();
        (a - 1.0) / a * (theta / (a - 1.0)).powf(a)
    }

    /// `φ(s) = exp K(is)`, on the closed upper half plane.
    pub fn chf(&self, s: Complex64) -> Complex64 {
        let lam = 1.0 / self.dispersion;
        let theta = Complex64::new(self.theta(), 0.0);
        let i = Complex64::new(0.0, 1.0);
        (lam * (self.kappa(theta + i * s / lam) - self.kappa(theta))).exp()
    }

    /// `K(s)`, NaN outside [`Self::cgf_domain`].
    pub fn cgf(&self, s: f64) -> f64 {
        let (lo, hi) = self.cgf_domain();
        if !(s > lo && s < hi) && s != 0.0 {
            return f64::NAN;
        }
        let lam = 1.0 / self.dispersion;
        let theta = Complex64::new(self.theta(), 0.0);
        (lam * (self.kappa(theta + s / lam) - self.kappa(theta))).re
    }

    /// The whole line at `p = 0`, `(−λθ, ∞)` below it, `(−∞, −λθ)` above.
    pub fn cgf_domain(&self) -> (f64, f64) {
        let edge = -self.theta() / self.dispersion;
        if self.power == 0.0 {
            (f64::NEG_INFINITY, f64::INFINITY)
        } else if self.power < 0.0 {
            (edge, f64::INFINITY)
        } else {
            (f64::NEG_INFINITY, edge)
        }
    }

    /// The whole line for `p ≤ 0`, `[0, ∞)` above 1.
    pub fn support(&self) -> (f64, f64) {
        if self.power <= 0.0 {
            (f64::NEG_INFINITY, f64::INFINITY)
        } else {
            (0.0, f64::INFINITY)
        }
    }

    /// `(0, e^(−λ_N))` for `1 < p < 2`, the compound Poisson's mass at 0,
    /// `λ_N = μ^(2−p) / (φ (2 − p))`.
    pub fn atom(&self) -> Option<(f64, f64)> {
        let p = self.power;
        if p > 1.0 && p < 2.0 {
            let frequency = self.mean.powf(2.0 - p) / (self.dispersion * (2.0 - p));
            Some((0.0, (-frequency).exp()))
        } else {
            None
        }
    }

    /// `φ μ^p`.
    pub fn variance(&self) -> f64 {
        self.dispersion * self.mean.powf(self.power)
    }

    /// `p` times the coefficient of variation.
    pub fn skewness(&self) -> f64 {
        self.power * self.variance().sqrt() / self.mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derivatives of `K` at 0, by central differences, against the
    /// stated mean and variance.
    fn check_moments(cgf: impl Fn(f64) -> f64, mean: f64, variance: f64) {
        let h = 1e-4;
        let k1 = (cgf(h) - cgf(-h)) / (2.0 * h);
        let k2 = (cgf(h) - 2.0 * cgf(0.0) + cgf(-h)) / (h * h);
        assert!(
            (k1 - mean).abs() < 1e-6 * mean.abs().max(1.0),
            "{k1} {mean}"
        );
        assert!((k2 / variance - 1.0).abs() < 1e-4, "{k2} {variance}");
    }

    #[test]
    fn moments_are_the_cgf_derivatives() {
        let nig = NormalInverseGaussian::new(2.0, 0.5, 1.5, 0.3).unwrap();
        check_moments(|s| nig.cgf(s), nig.mean(), nig.variance());
        let vg = VarianceGamma::new(0.2, 0.5, -0.1, 0.05).unwrap();
        check_moments(|s| vg.cgf(s), vg.mean(), vg.variance());
        for y in [0.5, 1.5, -0.5] {
            let ts = TemperedStable::new(1.0, 5.0, 10.0, y, 0.2).unwrap();
            check_moments(|s| ts.cgf(s), ts.mean(), ts.variance());
        }
    }

    #[test]
    fn the_chf_is_the_cgf_on_the_imaginary_axis() {
        let i = Complex64::new(0.0, 1.0);
        let u = 0.3;
        let nig = NormalInverseGaussian::new(2.0, 0.5, 1.5, 0.3).unwrap();
        assert!((nig.chf(-i * u).re - nig.cgf(u).exp()).abs() < 1e-14);
        let vg = VarianceGamma::new(0.2, 0.5, -0.1, 0.05).unwrap();
        assert!((vg.chf(-i * u).re - vg.cgf(u).exp()).abs() < 1e-14);
        let ts = TemperedStable::new(1.0, 5.0, 10.0, 0.5, 0.2).unwrap();
        assert!((ts.chf(-i * u).re - ts.cgf(u).exp()).abs() < 1e-14);
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(NormalInverseGaussian::new(1.0, 1.0, 1.0, 0.0).is_err());
        assert!(NormalInverseGaussian::new(1.0, 0.5, 0.0, 0.0).is_err());
        assert!(VarianceGamma::new(0.0, 1.0, 0.0, 0.0).is_err());
        assert!(TemperedStable::new(1.0, 1.0, 1.0, 1.0, 0.0).is_err());
        assert!(TemperedStable::new(1.0, 1.0, 1.0, 2.0, 0.0).is_err());
        assert!((gamma_fn(-0.5) + 2.0 * PI.sqrt()).abs() < 1e-13);
    }

    #[test]
    fn tweedie_power_meets_the_named_laws() {
        // p = 0 is the normal with variance phi; p = 2 the gamma with shape
        // 1/phi; p = 3 the inverse Gaussian with shape 1/phi.
        let n = TweediePower::new(0.0, 5.0, 4.0).unwrap();
        assert!((n.cgf(0.3) - (5.0 * 0.3 + 0.5 * 4.0 * 0.09)).abs() < 1e-12);
        let g = TweediePower::new(2.0, 10.0, 0.25).unwrap();
        // gamma shape 4, scale 2.5: K(s) = -4 ln(1 - 2.5 s)
        assert!((g.cgf(0.1) + 4.0 * (1.0f64 - 0.25).ln()).abs() < 1e-12);
        let ig = TweediePower::new(3.0, 10.0, 0.5).unwrap();
        // inverse Gaussian mean 10, shape 2
        let s = 0.005;
        let k = 2.0 / 10.0 * (1.0 - (1.0 - 2.0 * 100.0 * s / 2.0f64).sqrt());
        assert!((ig.cgf(s) - k).abs() < 1e-12);
        for t in [
            TweediePower::new(-1.0, 3.0, 2.0).unwrap(),
            TweediePower::new(2.5, 3.0, 0.4).unwrap(),
            TweediePower::new(1.5, 10.0, 2.0).unwrap(),
            TweediePower::new(5.0, 2.0, 0.1).unwrap(),
        ] {
            check_moments(|s| t.cgf(s), t.mean(), t.variance());
        }
        assert!(TweediePower::new(0.5, 1.0, 1.0).is_err());
        assert!(TweediePower::new(1.0, 1.0, 1.0).is_err());
        assert!(TweediePower::new(f64::INFINITY, 1.0, 1.0).is_err());
        let cv = TweediePower::from_mean_cv(2.5, 10.0, 0.3).unwrap();
        assert!((cv.variance().sqrt() / 10.0 - 0.3).abs() < 1e-12);
    }
}
