//! The alpha-stable law's parameters and closed forms, as the
//! characteristic-function engine needs them (`aggregate`'s `Stable`,
//! #779). Its distribution is computed on a grid by [`crate::chf::invert`].

use prospicio_core::{Error, Result};
use prospicio_math::fft::Complex64;
use prospicio_math::integrate::adaptive_gauss_legendre;
use prospicio_math::roots::{Stop, illinois, try_bisect};
use prospicio_math::special::{ln_gamma, norm_cdf};

use crate::distribution::{Distribution, check_probability};

use std::f64::consts::PI;

/// Above `α = 1` the tail series is asymptotic: its error is about its
/// smallest term, which must be at most this fraction of the sum.
const TAIL_SERIES_TOL: f64 = 1e-3;

/// How a stable law's location is read (Nolan 2020, Def. 1.3 to 1.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StableParameterization {
    /// `ln E e^{itX} = −γ^α |t|^α (1 − iβ sign(t) tan(πα/2)) + iδ₁t` for
    /// `α ≠ 1`, and `−γ|t|(1 + iβ (2/π) sign(t) ln|t|) + iδ₁t` at `α = 1`:
    /// SciPy's default, the form in which sums and scaling are clean.
    #[default]
    S1,
    /// `δ₀ = δ₁ + βγ tan(πα/2)`, or `δ₁ + (2/π)βγ ln γ` at `α = 1`:
    /// continuous in `α`.
    S0,
}

/// An alpha-stable law `S(α, β, γ, δ)`, held in the S1 form.
///
/// `0 < α ≤ 2`, `−1 ≤ β ≤ 1`, `γ > 0`. At `α = 2` the law is normal with
/// variance `2γ²` and `β` is set to 0. For `α < 2` the tails are
/// `P(X > x) ~ A x^(−α)` and `P(X < −x) ~ B x^(−α)` with
/// `A = (1 + β) C_α γ^α`, `B = (1 − β) C_α γ^α`,
/// `C_α = Γ(α) sin(πα/2) / π` (Nolan 2020, Thm. 1.2).
///
/// ```
/// use prospicio_prob::{Stable, StableParameterization};
///
/// let s = Stable::new(1.5, 1.0, 2.0, 0.0, StableParameterization::S1).unwrap();
/// assert!((s.tail_constant() - 1.128379).abs() < 1e-6);
/// assert!((s.loc0() - (-2.0)).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stable {
    alpha: f64,
    beta: f64,
    scale: f64,
    loc: f64,
}

impl Stable {
    /// The law with location `loc` read in `parameterization`.
    pub fn new(
        alpha: f64,
        beta: f64,
        scale: f64,
        loc: f64,
        parameterization: StableParameterization,
    ) -> Result<Self> {
        if !(alpha > 0.0 && alpha <= 2.0) {
            return Err(Error::InvalidParameter {
                name: "alpha",
                value: alpha,
                reason: "must be in (0, 2]",
            });
        }
        if !(-1.0..=1.0).contains(&beta) {
            return Err(Error::InvalidParameter {
                name: "beta",
                value: beta,
                reason: "must be in [-1, 1]",
            });
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err(Error::InvalidParameter {
                name: "scale",
                value: scale,
                reason: "must be finite and positive",
            });
        }
        if !loc.is_finite() {
            return Err(Error::InvalidParameter {
                name: "loc",
                value: loc,
                reason: "must be finite",
            });
        }
        let beta = if alpha == 2.0 { 0.0 } else { beta };
        let loc = match parameterization {
            StableParameterization::S1 => loc,
            StableParameterization::S0 => loc - location_shift(alpha, beta, scale),
        };
        Ok(Self {
            alpha,
            beta,
            scale,
            loc,
        })
    }

    /// The index of stability `α`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The skewness `β`.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// The scale `γ`.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The S1 location `δ₁`.
    pub fn loc(&self) -> f64 {
        self.loc
    }

    /// The S0 location `δ₀`.
    pub fn loc0(&self) -> f64 {
        self.loc + location_shift(self.alpha, self.beta, self.scale)
    }

    /// The mean: `δ₁` for `α > 1`; `∞` for `β = 1` and `−∞` for `β = −1`
    /// below it, NaN otherwise.
    pub fn mean(&self) -> f64 {
        if self.alpha > 1.0 {
            self.loc
        } else if self.beta == 1.0 {
            f64::INFINITY
        } else if self.beta == -1.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        }
    }

    /// The variance: `2γ²` at `α = 2`, infinite below.
    pub fn variance(&self) -> f64 {
        if self.alpha == 2.0 {
            2.0 * self.scale * self.scale
        } else {
            f64::INFINITY
        }
    }

    fn c_alpha_gamma(&self) -> f64 {
        let a = self.alpha;
        if a == 2.0 {
            return 0.0;
        }
        ln_gamma(a).exp() * (PI * a / 2.0).sin() / PI * self.scale.powf(a)
    }

    /// `A` in `P(X > x) ~ A x^(−α)`; 0 for the normal.
    pub fn tail_constant(&self) -> f64 {
        (1.0 + self.beta) * self.c_alpha_gamma()
    }

    /// `B` in `P(X < −x) ~ B x^(−α)`; 0 for the normal.
    pub fn left_tail_constant(&self) -> f64 {
        (1.0 - self.beta) * self.c_alpha_gamma()
    }

    /// The characteristic function at a real `t`, the S1 form.
    pub fn chf_real(&self, t: f64) -> Complex64 {
        let (a, b, g, d) = (self.alpha, self.beta, self.scale, self.loc);
        let i = Complex64::new(0.0, 1.0);
        let abs_t = t.abs();
        let sign = if t > 0.0 {
            1.0
        } else if t < 0.0 {
            -1.0
        } else {
            0.0
        };
        let psi = if a == 1.0 {
            let log_t = if abs_t > 0.0 { abs_t.ln() } else { 0.0 };
            -g * abs_t * (1.0 + i * b * (2.0 / PI) * sign * log_t)
        } else {
            -(g * abs_t).powf(a) * (1.0 - i * b * sign * (PI * a / 2.0).tan())
        };
        (psi + i * d * t).exp()
    }

    /// The characteristic function at `s`: the S1 form on the real line
    /// and, for `β = 1` below `α = 2`, its continuation onto the upper half
    /// plane, `ln φ(s) = −γ^α sec(πα/2) (−is)^α + iδ₁s` (`(2γ/π)(−is)
    /// ln(−is) + iδ₁s` at `α = 1`), principal branches. Fails off the real
    /// line otherwise: `β = −1` is inverted by reflection.
    pub fn chf(&self, s: Complex64) -> Result<Complex64> {
        if s.im == 0.0 {
            return Ok(self.chf_real(s.re));
        }
        if self.beta != 1.0 || self.alpha == 2.0 {
            return Err(Error::Data(
                "the stable characteristic function is continued off the real line only for \
                 beta = 1; beta = -1 is inverted by reflection"
                    .into(),
            ));
        }
        let (a, g, d) = (self.alpha, self.scale, self.loc);
        let i = Complex64::new(0.0, 1.0);
        let w = -i * s;
        let body = if a == 1.0 {
            if w == Complex64::new(0.0, 0.0) {
                Complex64::new(0.0, 0.0)
            } else {
                (2.0 * g / PI) * w * w.ln()
            }
        } else {
            -g.powf(a) / (PI * a / 2.0).cos() * w.powf(a)
        };
        Ok((body + i * d * s).exp())
    }

    /// `ln E e^{−uX}` of the light side's body for `β = 1`, `u ≥ 0`, before
    /// the location.
    fn laplace(&self, u: f64) -> f64 {
        let (a, g) = (self.alpha, self.scale);
        if a == 1.0 {
            if u > 0.0 {
                2.0 * g / PI * u * u.ln()
            } else {
                0.0
            }
        } else {
            -g.powf(a) / (PI * a / 2.0).cos() * u.powf(a)
        }
    }

    /// The cumulant generating function at a real `s`, NaN outside its
    /// domain ([`Stable::cgf_domain`]).
    pub fn cgf(&self, s: f64) -> f64 {
        if self.alpha == 2.0 {
            return self.loc * s + self.scale * self.scale * s * s;
        }
        if self.beta == 1.0 && s <= 0.0 {
            self.laplace(-s) + self.loc * s
        } else if self.beta == -1.0 && s >= 0.0 {
            self.laplace(s) + self.loc * s
        } else if s == 0.0 {
            0.0
        } else {
            f64::NAN
        }
    }

    /// `(s₋, s₊)`: the whole line at `α = 2`, `(−∞, 0)` for `β = 1`,
    /// `(0, ∞)` for `β = −1`, `(0, 0)` with two heavy tails.
    pub fn cgf_domain(&self) -> (f64, f64) {
        if self.alpha == 2.0 {
            (f64::NEG_INFINITY, f64::INFINITY)
        } else if self.beta == 1.0 {
            (f64::NEG_INFINITY, 0.0)
        } else if self.beta == -1.0 {
            (0.0, f64::INFINITY)
        } else {
            (0.0, 0.0)
        }
    }

    /// The support: `[δ₁, ∞)` for `α < 1` and `β = 1`, `(−∞, δ₁]` for
    /// `β = −1`, the whole line otherwise.
    pub fn support(&self) -> (f64, f64) {
        if self.alpha < 1.0 && self.beta == 1.0 {
            (self.loc, f64::INFINITY)
        } else if self.alpha < 1.0 && self.beta == -1.0 {
            (f64::NEG_INFINITY, self.loc)
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        }
    }

    /// The body scale: `√2 γ` at `α = 2`, else `γ`.
    pub fn body_scale(&self) -> f64 {
        if self.alpha == 2.0 {
            2f64.sqrt() * self.scale
        } else {
            self.scale
        }
    }

    /// `sup f = Γ(1 + 1/α) / (πγ)`, from `(1/2π) ∫ |φ|`.
    pub fn density_bound(&self) -> f64 {
        ln_gamma(1.0 + 1.0 / self.alpha).exp() / (PI * self.scale)
    }

    /// The two totally skewed laws whose sum is this one, for `|β| < 1`
    /// below `α = 2`: `X₊ ~ S(α, 1, γ₊, δ₁)` and `X₋ ~ S(α, −1, γ₋, 0)` with
    /// `γ± = γ ((1 ± β)/2)^(1/α)`, independent (`ln φ` is linear in `β`
    /// with a coefficient proportional to `γ^α`, and S1 at `α = 1` has no
    /// `γ ln γ` term). `None` for a law already one-sided or normal.
    pub fn halves(&self) -> Option<(Self, Self)> {
        if self.alpha == 2.0 || self.beta.abs() == 1.0 {
            return None;
        }
        let (a, g) = (self.alpha, self.scale);
        let plus = Self {
            alpha: a,
            beta: 1.0,
            scale: g * ((1.0 + self.beta) / 2.0).powf(1.0 / a),
            loc: self.loc,
        };
        let minus = Self {
            alpha: a,
            beta: -1.0,
            scale: g * ((1.0 - self.beta) / 2.0).powf(1.0 / a),
            loc: 0.0,
        };
        Some((plus, minus))
    }

    /// The law of the sum of independent `self` and `other`, which share
    /// `α` (Nolan 2020, Prop. 1.3): `γ^α` adds, `β` is the `γ^α`-weighted
    /// average and the locations add.
    pub fn add(&self, other: &Self) -> Result<Self> {
        if other.alpha != self.alpha {
            return Err(Error::Data(format!(
                "only stable laws with a common alpha add to a stable law, got {} and {}",
                self.alpha, other.alpha
            )));
        }
        let a = self.alpha;
        let (w1, w2) = (self.scale.powf(a), other.scale.powf(a));
        let beta = ((self.beta * w1 + other.beta * w2) / (w1 + w2)).clamp(-1.0, 1.0);
        Ok(Self {
            alpha: a,
            beta: if a == 2.0 { 0.0 } else { beta },
            scale: (w1 + w2).powf(1.0 / a),
            loc: self.loc + other.loc,
        })
    }

    /// `X + b`: the location shifted.
    pub fn shift(&self, b: f64) -> Self {
        Self {
            loc: self.loc + b,
            ..*self
        }
    }

    /// `c X` for a real `c ≠ 0` (Nolan 2020, Prop. 1.4): skewness
    /// `sign(c) β`, scale `|c| γ`, location `c δ₁`, less
    /// `(2/π) β γ c ln|c|` at `α = 1`.
    pub fn scale_by(&self, c: f64) -> Result<Self> {
        if c == 0.0 || !c.is_finite() {
            return Err(Error::InvalidParameter {
                name: "c",
                value: c,
                reason: "must be finite and not 0: scaling by 0 gives a point mass",
            });
        }
        let mut loc = c * self.loc;
        if self.alpha == 1.0 {
            loc -= 2.0 / PI * self.beta * self.scale * c * c.abs().ln();
        }
        Ok(Self {
            alpha: self.alpha,
            beta: c.signum() * self.beta,
            scale: c.abs() * self.scale,
            loc,
        })
    }

    /// `t @ X`, the sum of `t > 0` independent copies (`t` need not be a
    /// whole number): the characteristic function `φ^t`, so scale
    /// `t^(1/α) γ`, location `t δ₁` and the same `β`, at every `α`.
    pub fn copies(&self, t: f64) -> Result<Self> {
        if !(t.is_finite() && t > 0.0) {
            return Err(Error::InvalidParameter {
                name: "t",
                value: t,
                reason: "must be finite and positive",
            });
        }
        Ok(Self {
            scale: t.powf(1.0 / self.alpha) * self.scale,
            loc: t * self.loc,
            ..*self
        })
    }

    /// `P(X > δ₁) = 1/2 + arctan(β tan(πα/2)) / (πα)` for `α ≠ 1`; 1/2 at
    /// `α = 2` and at `α = 1`, `β = 0`; NaN at `α = 1` otherwise, where
    /// `X − δ₁` is not strictly stable.
    pub fn prob_above_loc(&self) -> f64 {
        let (a, b) = (self.alpha, self.beta);
        if a == 2.0 {
            0.5
        } else if a == 1.0 {
            if b == 0.0 { 0.5 } else { f64::NAN }
        } else {
            0.5 + (b * (PI * a / 2.0).tan()).atan() / (PI * a)
        }
    }

    /// The stable limit of a Poisson compound with a Pareto tail (the
    /// generalized central limit theorem, Feller 1971, ch. XVII): with
    /// expected count `λ`, severity tail `P(X > x) = θ^α x^(−α)(1 + o(1))`,
    /// `1 < α < 2`, and severity mean `m`, `S(α, 1, γ, λ m; 1)` with
    /// `γ^α = λ θ^α Γ(1 − α) cos(πα/2)`: the compound's mean and right
    /// tail exactly.
    pub fn from_compound_pareto(
        frequency: f64,
        alpha: f64,
        theta: f64,
        sev_mean: f64,
    ) -> Result<Self> {
        if !(alpha > 1.0 && alpha < 2.0) {
            return Err(Error::InvalidParameter {
                name: "alpha",
                value: alpha,
                reason: "the limit theorem form needs a tail index strictly between 1 and 2",
            });
        }
        for (name, x) in [("frequency", frequency), ("theta", theta)] {
            if !(x.is_finite() && x > 0.0) {
                return Err(Error::InvalidParameter {
                    name,
                    value: x,
                    reason: "must be finite and positive",
                });
            }
        }
        // Γ(1 − α) < 0 and cos(πα/2) < 0 on (1, 2): Γ by reflection.
        let gamma_1ma = PI / ((PI * (1.0 - alpha)).sin() * ln_gamma(alpha).exp());
        let g = (frequency * theta.powf(alpha) * gamma_1ma * (PI * alpha / 2.0).cos())
            .powf(1.0 / alpha);
        Self::new(
            alpha,
            1.0,
            g,
            frequency * sev_mean,
            StableParameterization::S1,
        )
    }

    /// The law of `−X`: `β` and the location negated.
    pub fn reflected(&self) -> Self {
        Self {
            beta: -self.beta,
            loc: -self.loc,
            ..*self
        }
    }

    /// `P(X > x)` for `β = 1`, far in the tail: for `α ≠ 1` the series
    /// `−(1/π) Σ_k (−1)^k Γ(kα) sin(kπα) s^k u^(−kα) / k!`,
    /// `s = γ^α sec(πα/2)`, `u = x − δ₁` (Nolan 2020, Thm. 3.10), summed to
    /// round-off below `α = 1` and to its smallest term above, where it is
    /// asymptotic; at `α = 1` its first two terms in `v = (u − (2γ/π) ln γ)
    /// / γ`, `a/v + a²(ln v + γ_E − 1)/v²`, `a = 2/π` (`aggregate`'s
    /// `tail_mass_above`).
    ///
    /// The series is a tail expansion, not the whole function. Where it is
    /// not valid the call returns an error and never a number that is not a
    /// probability: at or below the location (`u ≤ 0`), where `α > 1` and
    /// the smallest term of the asymptotic series is above `10⁻³` of the
    /// sum, where `α < 1` and the 400 terms have not converged or cancel
    /// more than `10⁷`-fold, where `α = 1` and `v < 20`, and where the sum is
    /// outside `[0, 1]`. `α = 2` is the normal law, exact everywhere.
    pub fn tail_mass_above(&self, x: f64) -> Result<f64> {
        let (a, g) = (self.alpha, self.scale);
        let u = x - self.loc;
        if a == 2.0 {
            return Ok(prospicio_math::special::norm_cdf(-u / (2f64.sqrt() * g)));
        }
        let outside = |why: &str| {
            Error::Data(format!(
                "the stable tail series is not valid at {x} (location {}): {why}; \
                 it holds far above the location",
                self.loc
            ))
        };
        if u <= 0.0 {
            return Err(outside("at or below the location"));
        }
        if a == 1.0 {
            let v = (u - 2.0 * g / PI * g.ln()) / g;
            if v < 20.0 {
                return Err(outside("too close for the two-term expansion at alpha 1"));
            }
            const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
            let c = 2.0 / PI;
            return Ok(c / v + c * c * (v.ln() + EULER_GAMMA - 1.0) / (v * v));
        }
        let s_signed = g.powf(a) / (PI * a / 2.0).cos();
        let log_s = s_signed.abs().ln();
        let sign_s: f64 = if s_signed > 0.0 { 1.0 } else { -1.0 };
        let mut total: f64 = 0.0;
        let mut last = f64::INFINITY;
        let mut largest: f64 = 0.0;
        let mut settled = false;
        for k in 1..400 {
            let kf = k as f64;
            let log_mag = ln_gamma(kf * a) - ln_gamma(kf + 1.0) + kf * (log_s - a * u.ln());
            if log_mag > 700.0 {
                return Err(Error::Data(format!(
                    "the stable tail series does not converge at {x}, too close to the location {}",
                    self.loc
                )));
            }
            let mag = log_mag.exp();
            if a > 1.0 && mag > last {
                settled = last <= TAIL_SERIES_TOL * total.abs();
                break;
            }
            last = mag;
            largest = largest.max(mag);
            let parity = if k % 2 == 0 { 1.0 } else { -1.0 };
            let term = -parity * (kf * PI * a).sin() * mag * sign_s.powi(k) / PI;
            total += term;
            if mag < 1e-18 * total.abs().max(1e-300) {
                settled = true;
                break;
            }
        }
        if !settled {
            return Err(outside("the series has not reached its accuracy"));
        }
        if a < 1.0 && largest > 1e7 * total.abs() {
            return Err(outside("the terms cancel too much"));
        }
        if !(0.0..=1.0).contains(&total) {
            return Err(outside("the sum is not a probability"));
        }
        Ok(total)
    }
}

/// The integrals of the stable law's cumulative distribution function are
/// taken to this absolute accuracy.
const CDF_TOL: f64 = 1e-14;

/// `(P(Z ≤ z), P(Z > z))` for the standard S0 law `S(α, β, 1, 0; 0)`, each
/// computed directly, so that neither tail loses precision to `1 − p`:
/// Nolan's (1997) integral representation, `Z ≤ ζ` by reflection,
/// `F(z; α, β) = 1 − F(−z; α, −β)`.
fn standard_tails(alpha: f64, beta: f64, z: f64) -> (f64, f64) {
    if alpha == 2.0 {
        return (norm_cdf(z / 2f64.sqrt()), norm_cdf(-z / 2f64.sqrt()));
    }
    let swap = |(c, s): (f64, f64)| (s, c);
    if alpha == 1.0 {
        if beta == 0.0 {
            // Cauchy, each tail as an arctangent of `1/|z|`.
            return if z > 0.0 {
                let s = (1.0 / z).atan() / PI;
                (1.0 - s, s)
            } else if z < 0.0 {
                let c = (-1.0 / z).atan() / PI;
                (c, 1.0 - c)
            } else {
                (0.5, 0.5)
            };
        }
        if beta < 0.0 {
            return swap(standard_tails(alpha, -beta, -z));
        }
        // β > 0: F(z) = (1/π) ∫_{−π/2}^{π/2} exp(−e^{−πz/(2β)} V(θ)) dθ,
        // integrated in the distance `d` from the nearer end, so that
        // `cos θ = sin d` keeps its precision there.
        // In logarithms, so that `e^{−πz/(2β)}` and the exponential in `V`
        // do not underflow and overflow separately.
        let ln_c = -PI * z / (2.0 * beta);
        // `h = π/2 + βθ` and `tan θ` from the distance `d` to either end.
        let g = |h: f64, d: f64, tan: f64| {
            (ln_c + (2.0 / PI * h / d.sin()).ln() + h * tan / beta).exp()
        };
        let left = |d: f64| g((1.0 - beta) * PI / 2.0 + beta * d, d, -1.0 / d.tan());
        let right = |d: f64| g((1.0 + beta) * PI / 2.0 - beta * d, d, 1.0 / d.tan());
        let (i1, j1) = integrals(left, 0.0, PI / 2.0);
        let (i2, j2) = integrals(right, 0.0, PI / 2.0);
        let (i, j) = (i1 + i2, j1 + j2);
        return (i / PI, j / PI);
    }
    let tan = (PI * alpha / 2.0).tan();
    let zeta = -beta * tan;
    // θ₀ = arctan(β tan(πα/2)) / α, exact at |β| = 1, where an end of the
    // range is a zero of the integrand's factors.
    let theta0 = if beta.abs() == 1.0 {
        beta * if alpha < 1.0 {
            PI / 2.0
        } else {
            PI / 2.0 - PI / alpha
        }
    } else {
        (beta * tan).atan() / alpha
    };
    let at_zeta = (PI / 2.0 - theta0) / PI;
    if z < zeta {
        return swap(standard_tails(alpha, -beta, -z));
    }
    if z == zeta {
        return (at_zeta, 1.0 - at_zeta);
    }
    let e = alpha / (alpha - 1.0);
    let ln_scale = e * (z - zeta).ln() + (alpha * theta0).cos().ln() / (alpha - 1.0);
    // In the distances `a = θ + θ₀` and `b = π/2 − θ` from the two ends,
    // `V = (sin b / sin αa)^(α/(α−1)) sin(αa + b) / sin b`, each factor
    // accurate near its end; each half of the range is integrated in the
    // distance from its own end, and `g` is formed in logarithms.
    let width = PI / 2.0 + theta0;
    let g = |a: f64, b: f64| {
        (ln_scale + e * (b.sin() / (alpha * a).sin()).ln() + ((alpha * a + b).sin() / b.sin()).ln())
            .exp()
    };
    let (i1, j1) = integrals(|a| g(a, width - a), 0.0, width / 2.0);
    let (i2, j2) = integrals(|b| g(width - b, b), 0.0, width / 2.0);
    // I = ∫ exp(−g), J = ∫ (1 − exp(−g)) over [−θ₀, π/2], I + J = π/2 + θ₀.
    let (i, j) = (i1 + i2, j1 + j2);
    if alpha < 1.0 {
        (at_zeta + i / PI, j / PI)
    } else {
        (at_zeta + j / PI, i / PI)
    }
}

/// The midpoint of [`try_bisect`]'s final bracket.
fn halve(lo: f64, hi: f64, stop: Stop, mut below: impl FnMut(f64) -> bool) -> f64 {
    let Ok((a, b)) = try_bisect(lo, hi, stop, |x| {
        Ok::<_, std::convert::Infallible>(below(x))
    });
    0.5 * (a + b)
}

/// `(∫ exp(−g), ∫ (1 − exp(−g)))` over `[lo, hi]` for a monotone exponent
/// `g ≥ 0`, infinite or undefined at an end read as its limit there.
///
/// Each integrand changes from near 0 to near 1 in a band that can be
/// narrow, and far in a tail one of them is tiny everywhere and lives near
/// the end where `g` is least. So the range is split where `g` crosses
/// fixed levels and levels above its least value, and each piece is
/// integrated adaptively, again to a tolerance relative to its first
/// estimate when that is small.
fn integrals(g: impl Fn(f64) -> f64, lo: f64, hi: f64) -> (f64, f64) {
    let g = |t: f64| {
        let v = g(t);
        if v.is_nan() {
            f64::INFINITY
        } else {
            v.max(0.0)
        }
    };
    if hi <= lo {
        return (0.0, 0.0);
    }
    let nudge = 1e-12 * (hi - lo);
    let (g_lo, g_hi) = (g(lo + nudge), g(hi - nudge));
    let rising = g_hi >= g_lo;
    let least = g_lo.min(g_hi);
    let mut cuts = vec![lo, hi];
    let levels = [0.01, 0.1, 1.0, 10.0]
        .into_iter()
        .chain([0.1, 1.0, 4.0, 10.0, 30.0, 100.0].map(|c| least + c));
    for level in levels {
        if level > g_lo.min(g_hi) && level < g_lo.max(g_hi) {
            // The point where g crosses `level`: below it on the low side.
            // A cut need not be exact, only near the crossing.
            cuts.push(halve(lo + nudge, hi - nudge, Stop::absolute(1e-9), |t| {
                (g(t) < level) == rising
            }));
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    // A first pass to `CDF_TOL`; a second, only for a total below 0.1, to
    // 1e-12 of that total, so that a far tail keeps its relative accuracy.
    let total = |f: &dyn Fn(f64) -> f64| {
        let rough: f64 = cuts
            .windows(2)
            .map(|w| adaptive_gauss_legendre(f, w[0], w[1], CDF_TOL * (w[1] - w[0]), 40))
            .sum();
        if rough > 1e-250 && rough < 0.1 {
            let width = hi - lo;
            cuts.windows(2)
                .map(|w| {
                    let tol = 1e-12 * rough * (w[1] - w[0]) / width;
                    adaptive_gauss_legendre(f, w[0], w[1], tol, 30)
                })
                .sum()
        } else {
            rough
        }
    };
    let i = total(&|t| (-g(t)).exp());
    let j = total(&|t| -(-g(t)).exp_m1());
    (i, j)
}

impl Stable {
    /// `(P(X ≤ x), P(X > x))`.
    fn tails(&self, x: f64) -> (f64, f64) {
        if x.is_nan() {
            return (f64::NAN, f64::NAN);
        }
        let (lo, hi) = self.support();
        if x < lo {
            return (0.0, 1.0);
        }
        if x >= hi {
            return (1.0, 0.0);
        }
        let z = (x - self.loc0()) / self.scale;
        let (c, s) = standard_tails(self.alpha, self.beta, z);
        (c.clamp(0.0, 1.0), s.clamp(0.0, 1.0))
    }
}

/// The stable law's distribution function by Nolan's (1997) integral
/// representation, each tail computed on its own to about `1e-14`
/// absolute, and its quantile by bisection on it. The mean and variance
/// are infinite or undefined below `α = 2` and `α = 1` (see
/// [`Stable::mean`]), so the summary statistics that need them do too.
impl Distribution for Stable {
    fn mean(&self) -> f64 {
        Stable::mean(self)
    }

    fn variance(&self) -> f64 {
        Stable::variance(self)
    }

    fn cdf(&self, x: f64) -> f64 {
        self.tails(x).0
    }

    fn survival(&self, x: f64) -> f64 {
        self.tails(x).1
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        let (lo, hi) = self.support();
        if p == 0.0 {
            return Ok(lo);
        }
        if p == 1.0 {
            return Ok(hi);
        }
        // A score increasing in `x` and zero at the quantile: the log of the
        // distribution function below the median and of the survival
        // function above it, so that a far tail keeps its precision and its
        // power law is near linear for the root finder.
        let score = |x: f64| {
            let (c, s) = self.tails(x);
            if p <= 0.5 {
                c.max(1e-300).ln() - p.ln()
            } else {
                (1.0 - p).ln() - s.max(1e-300).ln()
            }
        };
        // A bracket from the centre outwards, the step growing fourfold.
        let centre = self.loc0().clamp(lo, hi);
        let f_centre = score(centre);
        if f_centre == 0.0 {
            return Ok(centre);
        }
        let up = f_centre < 0.0;
        let (mut near, mut f_near) = (centre, f_centre);
        let mut step = self.body_scale();
        loop {
            let far = if up {
                (centre + step).min(hi)
            } else {
                (centre - step).max(lo)
            };
            if !far.is_finite() {
                // Past the largest float: the quantile is beyond f64's range.
                return Ok(far);
            }
            let f_far = score(far);
            if (f_far >= 0.0) == up || far == lo || far == hi {
                let (a, b, fa, fb) = if up {
                    (near, far, f_near, f_far)
                } else {
                    (far, near, f_far, f_near)
                };
                let x = illinois(a, b, fa, fb, score);
                if score(x).abs() <= 1e-13 {
                    return Ok(x);
                }
                // Regula falsi can stall where the score bends hard (the
                // light end of a one-sided law): finish by bisection.
                let stop = Stop {
                    max_steps: 400,
                    absolute: 1e-15 * self.body_scale(),
                    relative: 4e-16,
                };
                return Ok(halve(a, b, stop, |x| score(x) < 0.0));
            }
            (near, f_near) = (far, f_far);
            step *= 4.0;
        }
    }
}

/// `δ₀ − δ₁`.
fn location_shift(alpha: f64, beta: f64, scale: f64) -> f64 {
    if alpha == 1.0 {
        2.0 / PI * beta * scale * scale.ln()
    } else {
        beta * scale * (PI * alpha / 2.0).tan()
    }
}

/// The Hurwitz zeta function `ζ(s, q) = Σ_{k≥0} (q + k)^(−s)` for `s > 1`,
/// `q > 0`, by Euler–Maclaurin summation after ten terms.
pub(crate) fn hurwitz_zeta(s: f64, q: f64) -> f64 {
    // B_{2j} / (2j)!
    const B: [f64; 7] = [
        1.0 / 12.0,
        -1.0 / 720.0,
        1.0 / 30240.0,
        -1.0 / 1_209_600.0,
        1.0 / 47_900_160.0,
        -691.0 / 1_307_674_368_000.0,
        1.0 / 74_724_249_600.0,
    ];
    let n = 10;
    let mut sum: f64 = (0..n).map(|k| (q + k as f64).powf(-s)).sum();
    let a = q + n as f64;
    sum += a.powf(1.0 - s) / (s - 1.0) + 0.5 * a.powf(-s);
    // s (s + 1) ... (s + 2j - 2) a^(-s - 2j + 1)
    let mut rising = s;
    let mut power = a.powf(-s - 1.0);
    for (j, b) in B.iter().enumerate() {
        sum += b * rising * power;
        let m = 2.0 * j as f64;
        rising *= (s + m + 1.0) * (s + m + 2.0);
        power /= a * a;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hurwitz_zeta_matches_riemann_and_a_sum() {
        // ζ(2, 1) = π²/6, ζ(3, 1) = 1.2020569031595942.
        assert!((hurwitz_zeta(2.0, 1.0) - PI * PI / 6.0).abs() < 1e-14);
        assert!((hurwitz_zeta(3.0, 1.0) - 1.202_056_903_159_594_2).abs() < 1e-14);
        // ζ(2.5, 2.3) by a long sum and its tail integral.
        let direct: f64 = (0..200_000)
            .map(|k| (2.3 + k as f64).powf(-2.5))
            .sum::<f64>()
            + (2.3 + 200_000.0f64).powf(-1.5) / 1.5;
        assert!((hurwitz_zeta(2.5, 2.3) / direct - 1.0).abs() < 1e-10);
    }

    #[test]
    fn s0_and_s1_locations_and_tails() {
        let s1 = Stable::new(1.5, 1.0, 2.0, 0.0, StableParameterization::S1).unwrap();
        let s0 = Stable::new(1.5, 1.0, 2.0, s1.loc0(), StableParameterization::S0).unwrap();
        assert!((s0.loc() - s1.loc()).abs() < 1e-12);
        // aggregate's print: Pr(X > x) ~ 1.12838 x^-1.5, which is
        // 2 C_1.5 2^1.5 = 2 / sqrt(pi) exactly.
        assert!((s1.tail_constant() - std::f64::consts::FRAC_2_SQRT_PI).abs() < 1e-14);
        assert_eq!(s1.left_tail_constant(), 0.0);
        let r = s1.reflected();
        assert_eq!((r.beta(), r.loc()), (-1.0, -0.0));
        assert!(Stable::new(2.5, 0.0, 1.0, 0.0, StableParameterization::S1).is_err());
        assert!(Stable::new(1.5, 1.5, 1.0, 0.0, StableParameterization::S1).is_err());
    }

    #[test]
    fn the_continuation_meets_the_real_line() {
        // Just above the real line the continued form equals the S1 form.
        for a in [0.7, 1.0, 1.5] {
            let st = Stable::new(a, 1.0, 1.3, 0.4, StableParameterization::S1).unwrap();
            for t in [-3.0, -0.5, 0.2, 2.0] {
                let near = st.chf(Complex64::new(t, 1e-12)).unwrap();
                let on = st.chf_real(t);
                assert!((near - on).norm() < 1e-9, "{a} {t}");
            }
            // At s = i u it is the Laplace transform E e^{-uX}.
            let u = 0.8;
            let at = st.chf(Complex64::new(0.0, u)).unwrap();
            assert!((at.re - st.cgf(-u).exp()).abs() < 1e-12 && at.im.abs() < 1e-12);
        }
    }

    #[test]
    fn the_tail_series_starts_at_the_leading_term() {
        let st = Stable::new(0.7, 1.0, 1.0, 0.0, StableParameterization::S1).unwrap();
        let x: f64 = 1e6;
        let lead = st.tail_constant() * x.powf(-0.7);
        assert!((st.tail_mass_above(x).unwrap() / lead - 1.0).abs() < 1e-3);
    }

    #[test]
    fn the_algebra_matches_the_characteristic_function() {
        let x = Stable::new(1.5, 0.4, 2.0, 1.0, StableParameterization::S1).unwrap();
        let y = Stable::new(1.5, -0.8, 1.0, -3.0, StableParameterization::S1).unwrap();
        let z = x.add(&y).unwrap();
        for t in [-2.0, -0.3, 0.7, 1.9] {
            let lhs = z.chf_real(t);
            let rhs = x.chf_real(t) * y.chf_real(t);
            assert!((lhs - rhs).norm() < 1e-14, "sum {t}");
            // c X: phi_X(c t), for both signs and at alpha = 1.
            for st in [
                x,
                Stable::new(1.0, 0.6, 1.7, 0.5, StableParameterization::S1).unwrap(),
            ] {
                for c in [2.5, -0.4] {
                    let lhs = st.scale_by(c).unwrap().chf_real(t);
                    assert!((lhs - st.chf_real(c * t)).norm() < 1e-13, "scale {c} {t}");
                }
                let lhs = st.copies(2.7).unwrap().chf_real(t);
                assert!(
                    (lhs - st.chf_real(t).powf(2.7)).norm() < 1e-13,
                    "copies {t}"
                );
            }
        }
        assert!(
            x.add(&Stable::new(1.2, 0.0, 1.0, 0.0, StableParameterization::S1).unwrap())
                .is_err()
        );
        assert!(x.scale_by(0.0).is_err() && x.copies(-1.0).is_err());
        // Pr(X > delta_1) = 1 - 1/alpha for beta = 1, 1 < alpha < 2.
        let one = Stable::new(1.5, 1.0, 1.0, 0.0, StableParameterization::S1).unwrap();
        assert!((one.prob_above_loc() - (1.0 - 1.0 / 1.5)).abs() < 1e-15);
    }

    #[test]
    fn the_compound_pareto_limit_has_the_compound_tail() {
        // The right tail constant 2 C_alpha gamma^alpha equals lambda theta^alpha.
        let (lam, a, th) = (50.0, 1.6, 3.0);
        let st = Stable::from_compound_pareto(lam, a, th, 12.0).unwrap();
        assert!((st.tail_constant() / (lam * th.powf(a)) - 1.0).abs() < 1e-12);
        assert_eq!(st.loc(), 600.0);
        assert!(Stable::from_compound_pareto(lam, 0.9, th, 1.0).is_err());
    }

    #[test]
    fn the_distribution_function_matches_scipy() {
        // scipy.stats.levy_stable.cdf (S1, its default), scipy 1.16.
        let cases: [(f64, f64, f64, f64, f64, f64); 36] = [
            (1.5, 0.5, 2.0, 1.0, -5.0, 0.03920759052742784),
            (1.5, 0.5, 2.0, 1.0, -1.0, 0.3219871538583492),
            (1.5, 0.5, 2.0, 1.0, 0.0, 0.46218656010166814),
            (1.5, 0.5, 2.0, 1.0, 0.5, 0.5322046078223568),
            (1.5, 0.5, 2.0, 1.0, 3.0, 0.7967806891350713),
            (1.5, 0.5, 2.0, 1.0, 20.0, 0.9895258435629473),
            (0.7, -0.3, 1.0, 0.0, -5.0, 0.15460731267444516),
            (0.7, -0.3, 1.0, 0.0, -1.0, 0.4238623893483695),
            (0.7, -0.3, 1.0, 0.0, 0.0, 0.741975078342593),
            (0.7, -0.3, 1.0, 0.0, 0.5, 0.8173177988840885),
            (0.7, -0.3, 1.0, 0.0, 3.0, 0.9103212782894183),
            (0.7, -0.3, 1.0, 0.0, 20.0, 0.9707959198151488),
            (1.0, 0.8, 1.5, -1.0, -5.0, 0.019941231896847524),
            (1.0, 0.8, 1.5, -1.0, -1.0, 0.3367906527936897),
            (1.0, 0.8, 1.5, -1.0, 0.0, 0.5091069682588063),
            (1.0, 0.8, 1.5, -1.0, 0.5, 0.5764179460436523),
            (1.0, 0.8, 1.5, -1.0, 3.0, 0.7708707262327552),
            (1.0, 0.8, 1.5, -1.0, 20.0, 0.9551625746219026),
            (0.5, 1.0, 1.0, 0.0, -5.0, 0.0),
            (0.5, 1.0, 1.0, 0.0, -1.0, 0.0),
            (0.5, 1.0, 1.0, 0.0, 0.0, 0.0),
            (0.5, 1.0, 1.0, 0.0, 0.5, 0.15729920705028522),
            (0.5, 1.0, 1.0, 0.0, 3.0, 0.5637028616507731),
            (0.5, 1.0, 1.0, 0.0, 20.0, 0.8230632737581215),
            (1.9, -1.0, 0.5, 0.0, -5.0, 0.001284492862568598),
            (1.9, -1.0, 0.5, 0.0, -1.0, 0.08507177997488236),
            (1.9, -1.0, 0.5, 0.0, 0.0, 0.47368421052631576),
            (1.9, -1.0, 0.5, 0.0, 0.5, 0.7425401384601096),
            (1.9, -1.0, 0.5, 0.0, 3.0, 0.9999962624439694),
            (1.9, -1.0, 0.5, 0.0, 20.0, 1.0),
            (1.2, 0.0, 1.0, 0.0, -5.0, 0.042285439635577404),
            (1.2, 0.0, 1.0, 0.0, -1.0, 0.24663218873659032),
            (1.2, 0.0, 1.0, 0.0, 0.0, 0.5),
            (1.2, 0.0, 1.0, 0.0, 0.5, 0.6428420576949292),
            (1.2, 0.0, 1.0, 0.0, 3.0, 0.9205024558200386),
            (1.2, 0.0, 1.0, 0.0, 20.0, 0.9922810413566971),
        ];
        for (a, b, g, d, x, want) in cases {
            let s = Stable::new(a, b, g, d, StableParameterization::S1).unwrap();
            let (c, sf) = (Distribution::cdf(&s, x), Distribution::survival(&s, x));
            assert!(
                (c - want).abs() < 2e-8,
                "{a} {b} {g} {d} at {x}: {c} against {want}"
            );
            assert!((c + sf - 1.0).abs() < 1e-12, "{c} + {sf}");
        }
    }

    #[test]
    fn closed_forms_and_far_tails() {
        use prospicio_math::special::norm_cdf;
        // Lévy: S(1/2, 1, c, μ; 1) has P(X ≤ x) = erfc(√(c / (2(x − μ)))).
        let levy = Stable::new(0.5, 1.0, 2.0, 1.0, StableParameterization::S1).unwrap();
        for x in [1.5, 3.0, 10.0, 1e3] {
            let want = 2.0 * norm_cdf(-(2.0f64 / (x - 1.0)).sqrt());
            assert!((Distribution::cdf(&levy, x) - want).abs() < 1e-12, "{x}");
        }
        assert_eq!(Distribution::cdf(&levy, 0.5), 0.0);
        // Cauchy and normal.
        let cauchy = Stable::new(1.0, 0.0, 2.0, 1.0, StableParameterization::S1).unwrap();
        assert!(
            (Distribution::survival(&cauchy, 1e9) - (2.0f64 / (1e9 - 1.0)).atan() / PI).abs()
                < 1e-24
        );
        let normal = Stable::new(2.0, 0.0, 1.5, 0.0, StableParameterization::S1).unwrap();
        assert!(
            (Distribution::cdf(&normal, 1.0) - norm_cdf(1.0 / (1.5 * 2f64.sqrt()))).abs() < 1e-15
        );
        // The far right tail follows A x^(−α) without cancelling.
        let s = Stable::new(1.5, 0.5, 1.0, 0.0, StableParameterization::S1).unwrap();
        let x = 1e6;
        let ratio = Distribution::survival(&s, x) / (s.tail_constant() * x.powf(-1.5));
        assert!((ratio - 1.0).abs() < 1e-3, "{ratio}");
    }

    #[test]
    fn the_quantile_inverts_the_distribution_function() {
        for (a, b) in [
            (1.5, 0.5),
            (0.7, -0.3),
            (1.0, 0.8),
            (0.5, 1.0),
            (1.9, -1.0),
            (1.0, 0.0),
        ] {
            let s = Stable::new(a, b, 2.0, 1.0, StableParameterization::S1).unwrap();
            for p in [1e-4, 0.01, 0.3, 0.5, 0.9, 0.999] {
                let q = s.quantile(p).unwrap();
                // Each tail against its own probability, relatively.
                let (back, want) = if p <= 0.5 {
                    (Distribution::cdf(&s, q), p)
                } else {
                    (Distribution::survival(&s, q), 1.0 - p)
                };
                assert!(
                    (back / want - 1.0).abs() < 1e-11,
                    "{a} {b} at {p}: {q} gives {back}"
                );
            }
        }
        let levy = Stable::new(0.5, 1.0, 1.0, 0.0, StableParameterization::S1).unwrap();
        assert_eq!(levy.quantile(0.0).unwrap(), 0.0);
        assert_eq!(levy.quantile(1.0).unwrap(), f64::INFINITY);
        assert!(levy.quantile(1.5).is_err());
    }
}
