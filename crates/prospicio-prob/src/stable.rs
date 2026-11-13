//! The alpha-stable law's parameters and closed forms, as the
//! characteristic-function engine needs them (`aggregate`'s `Stable`,
//! #779). Its distribution is computed on a grid by [`crate::chf::invert`].

use prospicio_core::{Error, Result};
use prospicio_math::special::ln_gamma;
use rustfft::num_complex::Complex64;

use std::f64::consts::PI;

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
    pub fn tail_mass_above(&self, x: f64) -> Result<f64> {
        let (a, g) = (self.alpha, self.scale);
        let u = x - self.loc;
        if u <= 0.0 {
            return Ok(1.0);
        }
        if a == 2.0 {
            return Ok(prospicio_math::special::norm_cdf(-u / (2f64.sqrt() * g)));
        }
        if a == 1.0 {
            let v = (u - 2.0 * g / PI * g.ln()) / g;
            if v <= 0.0 {
                return Ok(1.0);
            }
            const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
            let c = 2.0 / PI;
            return Ok(c / v + c * c * (v.ln() + EULER_GAMMA - 1.0) / (v * v));
        }
        let s_signed = g.powf(a) / (PI * a / 2.0).cos();
        let log_s = s_signed.abs().ln();
        let sign_s: f64 = if s_signed > 0.0 { 1.0 } else { -1.0 };
        let mut total = 0.0;
        let mut last = f64::INFINITY;
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
                break;
            }
            last = mag;
            let parity = if k % 2 == 0 { 1.0 } else { -1.0 };
            let term = -parity * (kf * PI * a).sin() * mag * sign_s.powi(k) / PI;
            total += term;
            if mag < 1e-18 * total.abs().max(1e-300) {
                break;
            }
        }
        Ok(total)
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
        // aggregate's print: Pr(X > x) ~ 1.12838 x^-1.5.
        assert!((s1.tail_constant() - 1.128_379).abs() < 1e-6);
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
}
