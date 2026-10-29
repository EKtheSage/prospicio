//! Vine copulas: a `d`-dimensional copula built from `d(d − 1)/2`
//! bivariate pair copulas, each joining two variables given others, on the
//! trees of a canonical (C-) or drawable (D-) vine (Aas, Czado, Frigessi
//! and Bakken, 2009). See `docs/design/risk.md`.
//!
//! A pair copula `C(u, v)` has two conditional distributions:
//! `h₂(u | v) = ∂C/∂v`, the first argument given the second, and
//! `h₁(v | u) = ∂C/∂u`. Draws invert them; the density multiplies the pair
//! densities along the trees. Every edge's first argument is its earlier
//! variable in the vine's order.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_math::special::{
    bivariate_norm_cdf, ln_gamma, norm_cdf, norm_quantile, student_t_cdf, student_t_quantile,
};

use crate::copula::Copula;

/// A bivariate family for [`PairCopula`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairFamily {
    /// `C(u, v) = uv`; no parameter.
    Independence,
    /// The Gaussian copula with correlation `ρ` in `(−1, 1)`.
    Gaussian,
    /// Clayton, `θ > 0`: lower-tail dependence.
    Clayton,
    /// Gumbel, `θ ≥ 1`: upper-tail dependence.
    Gumbel,
    /// Frank, `θ ≠ 0` (negative for negative dependence): no tail
    /// dependence.
    Frank,
    /// Joe, `θ ≥ 1`: upper-tail dependence.
    Joe,
    /// Student's t with correlation `ρ` in `(−1, 1)` and `ν > 0` degrees of
    /// freedom ([`PairCopula::student`]): symmetric tail dependence, the
    /// family most often chosen for insurance losses.
    Student,
}

/// A bivariate copula for a vine: a family, its parameter and a rotation
/// by 0, 90, 180 or 270 degrees (for Clayton, Gumbel and Joe; 90 and 270
/// give negative dependence, 180 moves the tail dependence to the other
/// corner), with the conventions of `VineCopula` and pyvinecopulib:
///
/// | Rotation | `C_r(u, v)` | density |
/// |---|---|---|
/// | 90 | `v − C(1 − u, v)` | `c(1 − u, v)` |
/// | 180 | `u + v − 1 + C(1 − u, 1 − v)` | `c(1 − u, 1 − v)` |
/// | 270 | `u − C(u, 1 − v)` | `c(u, 1 − v)` |
///
/// ```
/// use prospicio_prob::vine::{PairCopula, PairFamily};
///
/// // Clayton: h₂(u | v) = v^(−θ−1) (u^−θ + v^−θ − 1)^(−1 − 1/θ).
/// let c = PairCopula::new(PairFamily::Clayton, 2.0, 0).unwrap();
/// let h = c.h2(0.3, 0.6);
/// let want = 0.6f64.powf(-3.0) * (0.3f64.powi(-2) + 0.6f64.powi(-2) - 1.0).powf(-1.5);
/// assert!((h - want).abs() < 1e-14);
/// assert!((c.h2_inverse(h, 0.6) - 0.3).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairCopula {
    family: PairFamily,
    parameter: f64,
    rotation: u16,
    /// Degrees of freedom of the Student t; 0 for the other families.
    nu: f64,
}

impl PairCopula {
    /// A pair copula of `family` with `parameter` (ignored for
    /// independence) and `rotation` in degrees.
    pub fn new(family: PairFamily, parameter: f64, rotation: u16) -> Result<Self> {
        let p = parameter;
        if family == PairFamily::Student {
            return Err(Error::Data(
                "the Student t pair has two parameters; use PairCopula::student(rho, nu)".into(),
            ));
        }
        let ok = match family {
            PairFamily::Independence => true,
            PairFamily::Gaussian => p > -1.0 && p < 1.0,
            PairFamily::Clayton => p.is_finite() && p > 0.0,
            PairFamily::Gumbel | PairFamily::Joe => p.is_finite() && p >= 1.0,
            PairFamily::Frank => p.is_finite() && p != 0.0,
            PairFamily::Student => unreachable!("handled above"),
        };
        if !ok {
            return Err(Error::InvalidParameter {
                name: "parameter",
                value: p,
                reason: "out of range for the family (Gaussian in (−1, 1), Clayton > 0, \
                         Gumbel and Joe ≥ 1, Frank ≠ 0)",
            });
        }
        let rotates = matches!(
            family,
            PairFamily::Clayton | PairFamily::Gumbel | PairFamily::Joe
        );
        if !matches!(rotation, 0 | 90 | 180 | 270) || (rotation != 0 && !rotates) {
            return Err(Error::InvalidParameter {
                name: "rotation",
                value: f64::from(rotation),
                reason: "must be 0, 90, 180 or 270, and 0 for independence, Gaussian and Frank",
            });
        }
        let parameter = if family == PairFamily::Independence {
            0.0
        } else {
            p
        };
        Ok(Self {
            family,
            parameter,
            rotation,
            nu: 0.0,
        })
    }

    /// The Student t pair with correlation `rho` in `(−1, 1)` and `nu`
    /// degrees of freedom, positive (no rotation).
    ///
    /// ```
    /// use prospicio_prob::vine::PairCopula;
    ///
    /// let t = PairCopula::student(0.5, 4.0).unwrap();
    /// assert_eq!((t.parameter(), t.nu()), (0.5, Some(4.0)));
    /// // Symmetric: C(u, v) = C(v, u), and h₂ inverts.
    /// assert!((t.cdf(0.3, 0.7) - t.cdf(0.7, 0.3)).abs() < 1e-12);
    /// assert!((t.h2_inverse(t.h2(0.3, 0.6), 0.6) - 0.3).abs() < 1e-12);
    /// ```
    pub fn student(rho: f64, nu: f64) -> Result<Self> {
        if !(rho > -1.0 && rho < 1.0) {
            return Err(Error::InvalidParameter {
                name: "rho",
                value: rho,
                reason: "must be in (−1, 1)",
            });
        }
        if !(nu.is_finite() && nu > 0.0) {
            return Err(Error::InvalidParameter {
                name: "nu",
                value: nu,
                reason: "must be finite and positive",
            });
        }
        Ok(Self {
            family: PairFamily::Student,
            parameter: rho,
            rotation: 0,
            nu,
        })
    }

    /// The independence copula.
    pub fn independence() -> Self {
        Self {
            family: PairFamily::Independence,
            parameter: 0.0,
            rotation: 0,
            nu: 0.0,
        }
    }

    /// The Student t's degrees of freedom; `None` for the other families.
    pub fn nu(&self) -> Option<f64> {
        (self.family == PairFamily::Student).then_some(self.nu)
    }

    pub fn family(&self) -> PairFamily {
        self.family
    }

    pub fn parameter(&self) -> f64 {
        self.parameter
    }

    pub fn rotation(&self) -> u16 {
        self.rotation
    }

    /// The copula with its arguments swapped, `C(v, u)`: the same for the
    /// unrotated and 180-degree families, 90 and 270 exchanged.
    pub fn transposed(&self) -> Self {
        let rotation = match self.rotation {
            90 => 270,
            270 => 90,
            r => r,
        };
        Self { rotation, ..*self }
    }

    /// The distribution function `C(u, v)`.
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        let (u, v) = (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
        let c = match self.rotation {
            90 => v - self.base_cdf(1.0 - u, v),
            180 => u + v - 1.0 + self.base_cdf(1.0 - u, 1.0 - v),
            270 => u - self.base_cdf(u, 1.0 - v),
            _ => self.base_cdf(u, v),
        };
        c.clamp(0.0, 1.0)
    }

    /// The density `c(u, v)`.
    pub fn density(&self, u: f64, v: f64) -> f64 {
        self.log_density(u, v).exp()
    }

    /// `ln c(u, v)`.
    pub fn log_density(&self, u: f64, v: f64) -> f64 {
        let (u, v) = (clamp01(u), clamp01(v));
        match self.rotation {
            90 => self.base_log_density(1.0 - u, v),
            180 => self.base_log_density(1.0 - u, 1.0 - v),
            270 => self.base_log_density(u, 1.0 - v),
            _ => self.base_log_density(u, v),
        }
    }

    /// `h₂(u | v) = ∂C(u, v)/∂v`, the distribution of the first argument
    /// given the second.
    pub fn h2(&self, u: f64, v: f64) -> f64 {
        let (u, v) = (clamp01(u), clamp01(v));
        let h = match self.rotation {
            90 => 1.0 - self.base_h(1.0 - u, v),
            180 => 1.0 - self.base_h(1.0 - u, 1.0 - v),
            270 => self.base_h(u, 1.0 - v),
            _ => self.base_h(u, v),
        };
        clamp01(h)
    }

    /// `h₁(v | u) = ∂C(u, v)/∂u`, the distribution of the second argument
    /// given the first.
    pub fn h1(&self, v: f64, u: f64) -> f64 {
        self.transposed().h2(v, u)
    }

    /// The `u` with `h₂(u | v) = p`.
    pub fn h2_inverse(&self, p: f64, v: f64) -> f64 {
        let (p, v) = (clamp01(p), clamp01(v));
        let u = match self.rotation {
            90 => 1.0 - self.base_h_inverse(1.0 - p, v),
            180 => 1.0 - self.base_h_inverse(1.0 - p, 1.0 - v),
            270 => self.base_h_inverse(p, 1.0 - v),
            _ => self.base_h_inverse(p, v),
        };
        clamp01(u)
    }

    /// The `v` with `h₁(v | u) = p`.
    pub fn h1_inverse(&self, p: f64, u: f64) -> f64 {
        self.transposed().h2_inverse(p, u)
    }

    fn base_cdf(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || v <= 0.0 {
            return 0.0;
        }
        let th = self.parameter;
        match self.family {
            PairFamily::Independence => u * v,
            PairFamily::Gaussian => {
                if u >= 1.0 || v >= 1.0 {
                    return u.min(v);
                }
                bivariate_norm_cdf(norm_quantile(u), norm_quantile(v), th)
            }
            PairFamily::Clayton => (u.powf(-th) + v.powf(-th) - 1.0).powf(-1.0 / th),
            PairFamily::Gumbel => {
                let a = (-u.ln()).powf(th) + (-v.ln()).powf(th);
                (-a.powf(1.0 / th)).exp()
            }
            PairFamily::Frank => {
                let a = (-th).exp_m1();
                -((-th * u).exp_m1() * (-th * v).exp_m1() / a).ln_1p() / th
            }
            PairFamily::Joe => {
                let (a, b) = ((1.0 - u).powf(th), (1.0 - v).powf(th));
                1.0 - (a + b - a * b).powf(1.0 / th)
            }
            PairFamily::Student => {
                if u >= 1.0 || v >= 1.0 {
                    return u.min(v);
                }
                self.student_cdf(u, v)
            }
        }
    }

    /// The Student t pair's `C(u, v) = ∫₀ᵛ h₂(u | s) ds`, the integrand in
    /// `s = v w²`, which softens its `s^(2/ν)` approach as `s → 0`, by
    /// adaptive 16-point Gauss–Legendre on `w` in `(0, 1)`: each interval is
    /// halved until its two halves agree with it to `1e-15` (absolute), at
    /// most 40 levels deep.
    fn student_cdf(&self, u: f64, v: f64) -> f64 {
        let f = |w: f64| -> std::result::Result<f64, ()> {
            let s = v * w * w;
            Ok(self.base_h(u, s.max(f64::MIN_POSITIVE)) * 2.0 * v * w)
        };
        let gl = |a: f64, b: f64| -> f64 {
            prospicio_math::integrate::gauss_legendre(f, a, b).expect("the integrand does not fail")
        };
        fn adapt(gl: &dyn Fn(f64, f64) -> f64, a: f64, b: f64, whole: f64, depth: u32) -> f64 {
            let m = 0.5 * (a + b);
            let (left, right) = (gl(a, m), gl(m, b));
            if depth == 0 || (left + right - whole).abs() <= 1e-15 {
                return left + right;
            }
            adapt(gl, a, m, left, depth - 1) + adapt(gl, m, b, right, depth - 1)
        }
        let mut sum = 0.0;
        // Eight starting panels, so a narrow feature is not missed.
        for k in 0..8 {
            let (a, b) = (k as f64 / 8.0, (k + 1) as f64 / 8.0);
            sum += adapt(&gl, a, b, gl(a, b), 40);
        }
        sum.clamp((u + v - 1.0).max(0.0), u.min(v))
    }

    fn base_log_density(&self, u: f64, v: f64) -> f64 {
        let th = self.parameter;
        match self.family {
            PairFamily::Independence => 0.0,
            PairFamily::Gaussian => {
                let (x, y) = (norm_quantile(u), norm_quantile(v));
                let s = 1.0 - th * th;
                -0.5 * s.ln() - (th * th * (x * x + y * y) - 2.0 * th * x * y) / (2.0 * s)
            }
            PairFamily::Clayton => {
                let t = u.powf(-th) + v.powf(-th) - 1.0;
                (1.0 + th).ln() - (1.0 + th) * (u * v).ln() - (2.0 + 1.0 / th) * t.ln()
            }
            PairFamily::Gumbel => {
                let (x, y) = (-u.ln(), -v.ln());
                let a = x.powf(th) + y.powf(th);
                let ln_c = -a.powf(1.0 / th);
                ln_c - (u * v).ln()
                    + (th - 1.0) * (x * y).ln()
                    + (2.0 / th - 2.0) * a.ln()
                    + (1.0 + (th - 1.0) * a.powf(-1.0 / th)).ln()
            }
            PairFamily::Frank => {
                let a = (-th).exp_m1();
                let d = a + (-th * u).exp_m1() * (-th * v).exp_m1();
                (-th * a).ln() - th * (u + v) - 2.0 * d.abs().ln()
            }
            PairFamily::Joe => {
                let (ub, vb) = (1.0 - u, 1.0 - v);
                let (a, b) = (ub.powf(th), vb.powf(th));
                let s = a + b - a * b;
                (1.0 / th - 2.0) * s.ln() + (th - 1.0) * (ub * vb).ln() + (th - 1.0 + s).ln()
            }
            PairFamily::Student => {
                let nu = self.nu;
                let (x, y) = (student_t_quantile(u, nu), student_t_quantile(v, nu));
                let s = 1.0 - th * th;
                let q = (x * x - 2.0 * th * x * y + y * y) / (nu * s);
                // ln t₂(x, y) − ln t₁(x) − ln t₁(y); the π and ν terms cancel
                // to these.
                ln_gamma(0.5 * (nu + 2.0)) + ln_gamma(0.5 * nu)
                    - 2.0 * ln_gamma(0.5 * (nu + 1.0))
                    - 0.5 * s.ln()
                    - 0.5 * (nu + 2.0) * q.ln_1p()
                    + 0.5 * (nu + 1.0) * ((x * x / nu).ln_1p() + (y * y / nu).ln_1p())
            }
        }
    }

    fn base_h(&self, u: f64, v: f64) -> f64 {
        let th = self.parameter;
        match self.family {
            PairFamily::Independence => u,
            PairFamily::Gaussian => {
                let (x, y) = (norm_quantile(u), norm_quantile(v));
                norm_cdf((x - th * y) / (1.0 - th * th).sqrt())
            }
            PairFamily::Clayton => {
                let t = u.powf(-th) + v.powf(-th) - 1.0;
                (-(th + 1.0) * v.ln() - (1.0 + 1.0 / th) * t.ln()).exp()
            }
            PairFamily::Gumbel => {
                let (x, y) = (-u.ln(), -v.ln());
                let a = x.powf(th) + y.powf(th);
                (-a.powf(1.0 / th) + (1.0 / th - 1.0) * a.ln() + (th - 1.0) * y.ln() - v.ln()).exp()
            }
            PairFamily::Frank => {
                let a = (-th).exp_m1();
                let (bu, bv) = ((-th * u).exp_m1(), (-th * v).exp_m1());
                (-th * v).exp() * bu / (a + bu * bv)
            }
            PairFamily::Joe => {
                let (ub, vb) = (1.0 - u, 1.0 - v);
                let (a, b) = (ub.powf(th), vb.powf(th));
                let s = a + b - a * b;
                s.powf(1.0 / th - 1.0) * vb.powf(th - 1.0) * (1.0 - a)
            }
            PairFamily::Student => {
                let nu = self.nu;
                let (x, y) = (student_t_quantile(u, nu), student_t_quantile(v, nu));
                let scale = ((nu + y * y) * (1.0 - th * th) / (nu + 1.0)).sqrt();
                student_t_cdf((x - th * y) / scale, nu + 1.0)
            }
        }
    }

    fn base_h_inverse(&self, p: f64, v: f64) -> f64 {
        let th = self.parameter;
        match self.family {
            PairFamily::Independence => p,
            PairFamily::Gaussian => {
                norm_cdf(norm_quantile(p) * (1.0 - th * th).sqrt() + th * norm_quantile(v))
            }
            PairFamily::Clayton => {
                let t = (p.ln() + (th + 1.0) * v.ln()) * (-th / (1.0 + th));
                (t.exp() + 1.0 - v.powf(-th)).powf(-1.0 / th)
            }
            PairFamily::Frank => {
                let a = (-th).exp_m1();
                let bv = (-th * v).exp_m1();
                let bu = p * a / ((-th * v).exp() - p * bv);
                -bu.ln_1p() / th
            }
            PairFamily::Student => {
                let nu = self.nu;
                let y = student_t_quantile(v, nu);
                let scale = ((nu + y * y) * (1.0 - th * th) / (nu + 1.0)).sqrt();
                student_t_cdf(student_t_quantile(p, nu + 1.0) * scale + th * y, nu)
            }
            PairFamily::Gumbel | PairFamily::Joe => {
                // h is increasing in u; bisect on (0, 1) to a relative 1e-15.
                let (mut lo, mut hi) = (0.0f64, 1.0f64);
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if self.base_h(mid, v) < p {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                    if hi - lo <= 1e-15 * hi.max(1e-300) {
                        break;
                    }
                }
                0.5 * (lo + hi)
            }
        }
    }
}

/// Keeps `u` strictly inside `(0, 1)`, where every density is finite.
pub(crate) fn clamp01(u: f64) -> f64 {
    u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON / 2.0)
}

/// The vine's shape: canonical (each tree a star) or drawable (each tree a
/// path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VineKind {
    /// C-vine: tree `t` joins the `t`-th variable in the order to every
    /// later one, given the first `t`.
    Canonical,
    /// D-vine: tree `t` joins variables `t + 1` places apart in the order,
    /// given those between them.
    Drawable,
}

/// A C- or D-vine copula in `d` dimensions.
///
/// `order` lists the dimensions (0-based) in vine order. `pairs[t]` holds
/// tree `t`'s `d − 1 − t` pair copulas: for a C-vine, edge `e` joins the
/// `t`-th variable in the order (the first argument) to the `(t + 1 + e)`-th
/// given the first `t`; for a D-vine, edge `e` joins the `e`-th (first
/// argument) and the `(e + t + 1)`-th given those between. Fewer than
/// `d − 1` trees truncate the vine: the missing trees are independence.
///
/// Draws invert the Rosenblatt transform one variable at a time, from `d`
/// uniforms taken in order from the simulation's stream; the density
/// multiplies the pair densities along the trees.
///
/// ```
/// use prospicio_prob::copula::Copula;
/// use prospicio_prob::vine::{PairCopula, PairFamily, VineCopula, VineKind};
///
/// let gumbel = PairCopula::new(PairFamily::Gumbel, 2.0, 0).unwrap();
/// let clayton = PairCopula::new(PairFamily::Clayton, 1.0, 90).unwrap();
/// let frank = PairCopula::new(PairFamily::Frank, 3.0, 0).unwrap();
/// let v = VineCopula::new(
///     VineKind::Drawable,
///     vec![0, 1, 2],
///     vec![vec![gumbel, clayton], vec![frank]],
/// )
/// .unwrap();
/// assert_eq!(v.dim(), 3);
/// // ln c = ln c₀₁(u₀, u₁) + ln c₁₂(u₁, u₂) + ln c₀₂|₁(h₂(u₀|u₁), h₁(u₂|u₁)).
/// let u = [0.2, 0.7, 0.4];
/// let want = gumbel.log_density(0.2, 0.7)
///     + clayton.log_density(0.7, 0.4)
///     + frank.log_density(gumbel.h2(0.2, 0.7), clayton.h1(0.4, 0.7));
/// assert!((v.log_density(&u).unwrap() - want).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct VineCopula {
    kind: VineKind,
    order: Vec<usize>,
    pairs: Vec<Vec<PairCopula>>,
}

impl VineCopula {
    /// A vine of `kind` over `order` (a permutation of `0..d`, `d ≥ 2`)
    /// with `pairs[t]` for each tree `t` kept, `d − 1 − t` pair copulas
    /// each.
    pub fn new(kind: VineKind, order: Vec<usize>, pairs: Vec<Vec<PairCopula>>) -> Result<Self> {
        let d = order.len();
        let mut seen = order.clone();
        seen.sort_unstable();
        if d < 2 || seen.iter().enumerate().any(|(i, &j)| i != j) {
            return Err(Error::Data(format!(
                "the order must be 0..d once each, with d at least 2; got {order:?}"
            )));
        }
        if pairs.len() > d - 1 {
            return Err(Error::Data(format!(
                "{} trees for a {d}-dimensional vine, which has {}",
                pairs.len(),
                d - 1
            )));
        }
        for (t, tree) in pairs.iter().enumerate() {
            if tree.len() != d - 1 - t {
                return Err(Error::Data(format!(
                    "tree {t} of a {d}-dimensional vine needs {} pair copulas, got {}",
                    d - 1 - t,
                    tree.len()
                )));
            }
        }
        Ok(Self { kind, order, pairs })
    }

    pub fn kind(&self) -> VineKind {
        self.kind
    }

    pub fn order(&self) -> &[usize] {
        &self.order
    }

    pub fn pairs(&self) -> &[Vec<PairCopula>] {
        &self.pairs
    }

    /// The pair copula of tree `t`, edge `e`; independence past a
    /// truncation.
    fn pair(&self, t: usize, e: usize) -> PairCopula {
        self.pairs
            .get(t)
            .map_or_else(PairCopula::independence, |tree| tree[e])
    }

    /// The vine's log-density at `u` (one value per dimension, in `(0, 1)`).
    pub fn log_density(&self, u: &[f64]) -> Result<f64> {
        let d = self.order.len();
        if u.len() != d {
            return Err(Error::Data(format!(
                "{} arguments for a {d}-dimensional copula",
                u.len()
            )));
        }
        if let Some(&bad) = u.iter().find(|x| !(0.0..=1.0).contains(*x)) {
            return Err(Error::InvalidParameter {
                name: "u",
                value: bad,
                reason: "must be in [0, 1]",
            });
        }
        let x: Vec<f64> = self.order.iter().map(|&j| u[j]).collect();
        let mut ll = 0.0;
        match self.kind {
            VineKind::Canonical => {
                // z[m] = u_m given the first t variables.
                let mut z = x;
                for t in 0..d - 1 {
                    let root = z[t];
                    for (e, zm) in z[t + 1..].iter_mut().enumerate() {
                        let c = self.pair(t, e);
                        ll += c.log_density(root, *zm);
                        *zm = c.h1(*zm, root);
                    }
                }
            }
            VineKind::Drawable => {
                // left[i] = x_i given i+1..i+t, right[i] = x_{i+t} given i..i+t−1
                // at the start of tree t (both u at tree 0).
                let (mut left, mut right) = (x.clone(), x);
                for t in 0..d - 1 {
                    let n = d - 1 - t;
                    let (mut next_left, mut next_right) = (vec![0.0; n], vec![0.0; n]);
                    for e in 0..n {
                        let (a, b) = (left[e], right[e + 1]);
                        let c = self.pair(t, e);
                        ll += c.log_density(a, b);
                        next_left[e] = c.h2(a, b);
                        next_right[e] = c.h1(b, a);
                    }
                    left = next_left;
                    right = next_right;
                }
            }
        }
        Ok(ll)
    }

    /// The inverse Rosenblatt transform: the point whose conditional
    /// distributions, in vine order, are the uniforms `w` (also in vine
    /// order). Uniform `w` give a draw from the vine.
    pub fn inverse_rosenblatt(&self, w: &[f64]) -> Vec<f64> {
        let d = self.order.len();
        let mut x = vec![0.0; d];
        match self.kind {
            VineKind::Canonical => {
                // In a C-vine, x_t given the first t variables is w_t.
                for k in 0..d {
                    let mut val = w[k];
                    for t in (0..k).rev() {
                        val = self.pair(t, k - t - 1).h1_inverse(val, w[t]);
                    }
                    x[k] = val;
                }
            }
            VineKind::Drawable => {
                // lc[t][i] = x_i given i+1..i+t, filled as variables arrive.
                let mut lc: Vec<Vec<f64>> = vec![vec![0.0; d]; d];
                for k in 0..d {
                    // r[t] = x_k given k−t..k−1, from r[k] = w_k downwards.
                    let mut r = vec![0.0; k + 1];
                    r[k] = w[k];
                    for t in (1..=k).rev() {
                        let e = k - t;
                        r[t - 1] = self.pair(t - 1, e).h1_inverse(r[t], lc[t - 1][e]);
                    }
                    x[k] = r[0];
                    lc[0][k] = r[0];
                    for t in 1..=k {
                        let e = k - t;
                        lc[t][e] = self.pair(t - 1, e).h2(lc[t - 1][e], r[t - 1]);
                    }
                }
            }
        }
        let mut u = vec![0.0; d];
        for (k, &j) in self.order.iter().enumerate() {
            u[j] = x[k];
        }
        u
    }
}

impl Copula for VineCopula {
    fn dim(&self) -> usize {
        self.order.len()
    }

    /// `d` uniforms from the stream in vine order, then the inverse
    /// Rosenblatt transform.
    fn sample(&self, rng: &mut StreamRng, u: &mut [f64]) {
        let w: Vec<f64> = (0..self.order.len()).map(|_| rng.next_open01()).collect();
        u.copy_from_slice(&self.inverse_rosenblatt(&w));
        for x in u.iter_mut() {
            *x = clamp01(*x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copula::GaussianCopula;
    use prospicio_math::linalg::{cholesky, cholesky_inverse};

    fn all_pairs() -> Vec<PairCopula> {
        let mut v = vec![
            PairCopula::independence(),
            PairCopula::new(PairFamily::Gaussian, 0.6, 0).unwrap(),
            PairCopula::new(PairFamily::Gaussian, -0.4, 0).unwrap(),
            PairCopula::new(PairFamily::Frank, 4.0, 0).unwrap(),
            PairCopula::new(PairFamily::Frank, -3.0, 0).unwrap(),
        ];
        for f in [PairFamily::Clayton, PairFamily::Gumbel, PairFamily::Joe] {
            for r in [0, 90, 180, 270] {
                v.push(PairCopula::new(f, 2.5, r).unwrap());
            }
        }
        v.push(PairCopula::student(0.6, 4.0).unwrap());
        v.push(PairCopula::student(-0.3, 2.5).unwrap());
        v
    }

    #[test]
    fn the_student_cdf_matches_quadrature_at_any_degrees_of_freedom() {
        // SciPy 1.14.1 quad of h₂ over s, to 1e-13 (pyvinecopulib's t cdf
        // rounds the degrees of freedom).
        let t = PairCopula::student(0.5, 2.5).unwrap();
        assert!((t.cdf(0.3, 0.6) - 0.240_598_834_693_245_2).abs() < 1e-12);
        assert!((t.cdf(0.999, 0.998) - 0.997_469_702_656_018_3).abs() < 1e-12);
        let t = PairCopula::student(0.9, 30.0).unwrap();
        assert!((t.cdf(0.999, 0.998) - 0.997_654_677_808_199_6).abs() < 1e-12);
        assert!(PairCopula::new(PairFamily::Student, 0.5, 0).is_err());
        assert!(PairCopula::student(1.0, 4.0).is_err());
        assert!(PairCopula::student(0.5, 0.0).is_err());
    }

    #[test]
    fn pair_functions_are_the_cdf_derivatives() {
        let e = 1e-5;
        for c in all_pairs() {
            for (u, v) in [(0.3, 0.6), (0.8, 0.2), (0.55, 0.45), (0.1, 0.9)] {
                let d2 = (c.cdf(u, v + e) - c.cdf(u, v - e)) / (2.0 * e);
                assert!((c.h2(u, v) - d2).abs() < 1e-6, "{c:?} h2 {u} {v}");
                let d1 = (c.cdf(u + e, v) - c.cdf(u - e, v)) / (2.0 * e);
                assert!((c.h1(v, u) - d1).abs() < 1e-6, "{c:?} h1 {u} {v}");
                let e2 = 1e-4;
                let mixed = (c.cdf(u + e2, v + e2) - c.cdf(u + e2, v - e2) - c.cdf(u - e2, v + e2)
                    + c.cdf(u - e2, v - e2))
                    / (4.0 * e2 * e2);
                assert!(
                    (c.density(u, v) - mixed).abs() < 1e-4 * mixed.max(1.0),
                    "{c:?} density {u} {v}: {} {mixed}",
                    c.density(u, v)
                );
                assert!((c.h2_inverse(c.h2(u, v), v) - u).abs() < 1e-10, "{c:?}");
                assert!((c.h1_inverse(c.h1(v, u), u) - v).abs() < 1e-10, "{c:?}");
            }
        }
    }

    /// Partial correlations of a correlation matrix for a D-vine on 0..d
    /// (tree t, edge e: ρ(e, e+t+1 | between)).
    fn partial(r: &[f64], d: usize, i: usize, j: usize, given: &[usize]) -> f64 {
        // From the inverse of the submatrix on {i, j} ∪ given.
        let idx: Vec<usize> = [i, j].iter().chain(given).copied().collect();
        let n = idx.len();
        let sub: Vec<f64> = idx
            .iter()
            .flat_map(|&a| idx.iter().map(move |&b| r[a * d + b]))
            .collect();
        let p = cholesky_inverse(&cholesky(&sub, n).unwrap(), n);
        -p[1] / (p[0] * p[n + 1]).sqrt()
    }

    fn gaussian_log_density(r: &[f64], d: usize, u: &[f64]) -> f64 {
        let z: Vec<f64> = u.iter().map(|&x| norm_quantile(x)).collect();
        let l = cholesky(r, d).unwrap();
        let p = cholesky_inverse(&l, d);
        let ln_det: f64 = (0..d).map(|i| 2.0 * l[i * d + i].ln()).sum();
        let mut q = 0.0;
        for i in 0..d {
            for j in 0..d {
                q += z[i] * (p[i * d + j] - if i == j { 1.0 } else { 0.0 }) * z[j];
            }
        }
        -0.5 * q - 0.5 * ln_det
    }

    #[test]
    fn gaussian_vines_are_the_gaussian_copula() {
        let d = 4;
        let r = [
            1.0, 0.5, 0.3, 0.2, 0.5, 1.0, 0.4, 0.1, 0.3, 0.4, 1.0, 0.6, 0.2, 0.1, 0.6, 1.0,
        ];
        let order = vec![2, 0, 3, 1];
        let gauss = |rho: f64| PairCopula::new(PairFamily::Gaussian, rho, 0).unwrap();
        // D-vine on the order: tree t, edge e joins order[e], order[e+t+1].
        let dpairs: Vec<Vec<PairCopula>> = (0..d - 1)
            .map(|t| {
                (0..d - 1 - t)
                    .map(|e| {
                        let between: Vec<usize> = (e + 1..e + t + 1).map(|k| order[k]).collect();
                        gauss(partial(&r, d, order[e], order[e + t + 1], &between))
                    })
                    .collect()
            })
            .collect();
        let cpairs: Vec<Vec<PairCopula>> = (0..d - 1)
            .map(|t| {
                (0..d - 1 - t)
                    .map(|e| {
                        let given: Vec<usize> = (0..t).map(|k| order[k]).collect();
                        gauss(partial(&r, d, order[t], order[t + 1 + e], &given))
                    })
                    .collect()
            })
            .collect();
        let dv = VineCopula::new(VineKind::Drawable, order.clone(), dpairs).unwrap();
        let cv = VineCopula::new(VineKind::Canonical, order, cpairs).unwrap();
        for u in [
            [0.2, 0.7, 0.4, 0.9],
            [0.5, 0.5, 0.5, 0.5],
            [0.05, 0.3, 0.8, 0.6],
        ] {
            let want = gaussian_log_density(&r, d, &u);
            assert!(
                (dv.log_density(&u).unwrap() - want).abs() < 1e-10,
                "D {u:?}"
            );
            assert!(
                (cv.log_density(&u).unwrap() - want).abs() < 1e-10,
                "C {u:?}"
            );
        }
        // Draws have the Gaussian copula's pairwise normal-score correlation.
        let g = GaussianCopula::new(&r, d).unwrap();
        assert_eq!(g.dim(), cv.dim());
        let n = 20_000;
        for v in [&dv, &cv] {
            let z: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let mut u = vec![0.0; d];
                    v.sample(&mut StreamRng::new(3, i), &mut u);
                    u.iter().map(|&x| norm_quantile(x)).collect()
                })
                .collect();
            for (a, b) in [(0, 1), (2, 3), (0, 3)] {
                let s: f64 = z.iter().map(|x| x[a] * x[b]).sum::<f64>() / n as f64;
                assert!(
                    (s - r[a * d + b]).abs() < 0.03,
                    "{:?} {a}{b}: {s}",
                    v.kind()
                );
            }
        }
    }

    #[test]
    fn sampling_agrees_with_the_density() {
        // P(U ≤ p) from draws against ∫ c over [0, p] by a midpoint grid,
        // with asymmetric (rotated) pairs in both shapes.
        let gumbel = PairCopula::new(PairFamily::Gumbel, 2.0, 0).unwrap();
        let clayton = PairCopula::new(PairFamily::Clayton, 1.5, 90).unwrap();
        let joe = PairCopula::new(PairFamily::Joe, 1.8, 270).unwrap();
        for kind in [VineKind::Canonical, VineKind::Drawable] {
            let v = VineCopula::new(kind, vec![1, 2, 0], vec![vec![gumbel, clayton], vec![joe]])
                .unwrap();
            let p = [0.6, 0.5, 0.7];
            let m = 60;
            let mut integral = 0.0;
            for i in 0..m {
                for j in 0..m {
                    for k in 0..m {
                        let u = [
                            p[0] * (i as f64 + 0.5) / m as f64,
                            p[1] * (j as f64 + 0.5) / m as f64,
                            p[2] * (k as f64 + 0.5) / m as f64,
                        ];
                        integral += v.log_density(&u).unwrap().exp();
                    }
                }
            }
            integral *= p[0] * p[1] * p[2] / (m * m * m) as f64;
            let n = 40_000;
            let hit = (0..n)
                .filter(|&i| {
                    let mut u = [0.0; 3];
                    v.sample(&mut StreamRng::new(8, i), &mut u);
                    u.iter().zip(&p).all(|(x, q)| x <= q)
                })
                .count() as f64
                / n as f64;
            let se = (integral * (1.0 - integral) / n as f64).sqrt();
            assert!(
                (hit - integral).abs() < 4.0 * se + 2e-3,
                "{kind:?}: {hit} vs {integral}"
            );
        }
    }

    #[test]
    fn refuses_bad_vines() {
        let c = PairCopula::new(PairFamily::Clayton, 1.0, 0).unwrap();
        assert!(VineCopula::new(VineKind::Drawable, vec![0, 2], vec![vec![c]]).is_err());
        assert!(VineCopula::new(VineKind::Drawable, vec![0, 1, 2], vec![vec![c]]).is_err());
        assert!(PairCopula::new(PairFamily::Gaussian, 0.5, 90).is_err());
        assert!(PairCopula::new(PairFamily::Gumbel, 0.5, 0).is_err());
        // A truncated vine: one tree kept.
        let v = VineCopula::new(VineKind::Drawable, vec![0, 1, 2], vec![vec![c, c]]).unwrap();
        assert!(v.log_density(&[0.3, 0.4, 0.5]).unwrap().is_finite());
    }
}
