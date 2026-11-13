//! Distributions computed on a grid by inverting their characteristic
//! function: `aggregate`'s `AggregateCHF` and its `chfagg` statement
//! (#779).
//!
//! The engine samples a law's characteristic function `φ` on the FFT's
//! frequency grid and inverts it with one inverse transform. It needs no
//! discretized severity and no claim count: a law with a closed-form `φ`
//! (an infinitely divisible law, the time-one law of a Lévy process, so a
//! limit of compound Poisson aggregates) is computed directly.
//!
//! **Bucket masses, not density samples.** On the grid `x_k = x_0 + k h`,
//! with `U` uniform on `[-h/2, h/2]` and independent of `X`,
//! `P(X ∈ [x_k − h/2, x_k + h/2)) = h f_{X+U}(x_k)` and
//! `φ_{X+U}(t) = φ_X(t) sinc(t h / 2)`. Inverting `φ_X · sinc` gives the
//! bucket probabilities of the rounding discretization, the one
//! [`crate::Grid::rounding`] and the compound engines use, up to frequency
//! truncation (frequencies past `π / h` are dropped) and aliasing (mass
//! outside the window wraps onto it).
//!
//! **The window from the tails.** Every law here has a finite cumulant
//! generating function `K` near 0 on each side that is not bounded, so the
//! Chernoff bound `P(X > x) ≤ exp(K(s) − s x)` gives the right edge
//! `x_R = inf_s (K(s) − ln ε) / s`, and the left edge likewise for `−X`.
//! Rule A sizes the window from these edges with mass `1 − p` beyond it,
//! half on each side, and rounds the bucket up to a "nice" value
//! (`aggregate`'s `round_bucket`); rule C takes a given bucket. When the frequency
//! truncation `|φ(π / h)|` is above `1e-6`, the bucket walks down the
//! ladder while that lowers the larger of the truncation and the mass the
//! shortened window leaves outside.
//!
//! **An atom** (the Tweedie's point mass at 0) is taken out of `φ` before
//! inversion and added back to its bucket, so frequency truncation does
//! not ring around it.
//!
//! **Tilting.** The kernel inverts along the line `Im s = θ`, where the
//! integrand decays like the damped law `e^{−θx} f(x)`: the discretized
//! integral wraps the damped sequence, and `e^{θ x_k}` recovers the masses.
//! The laws of this module have light tails, which barely wrap, so they are
//! inverted on the real line; the tilt is what a law with a power tail
//! needs.
//!
//! The masses are never renormalized; they can carry round-off of either
//! sign far from the body, at the level of [`ChfReport::truncation_error`].
//!
//! # Example
//!
//! ```
//! use prospicio_prob::chf::{ChfLaw, ChfOptions, invert};
//! use prospicio_prob::{Distribution, Gamma};
//!
//! let law = ChfLaw::Gamma(Gamma::new(4.0, 25.0).unwrap());
//! let d = invert(&law, &ChfOptions::with_log2(12)).unwrap();
//! assert_eq!(d.step(), 0.25);
//! assert!((d.mean() - 100.0).abs() < 1e-6);
//! assert!(d.report().truncation_error < 1e-9);
//! ```

use std::fmt;
use std::sync::Arc;

use prospicio_core::{Error, Result};
use prospicio_math::optimize::brent;
use rustfft::FftPlanner;
/// The complex type of [`ChfLaw::chf`] and a user's [`ChfFn`].
pub use rustfft::num_complex::Complex64;

use crate::dist::Dist;
use crate::distribution::{Distribution, check_probability};
use crate::gamma::Gamma;
use crate::grid::Grid;
use crate::inverse_gaussian::InverseGaussian;
use crate::levy::{NormalInverseGaussian, TemperedStable, VarianceGamma};
use crate::normal::Normal;
use crate::stable::{Stable, hurwitz_zeta};
use crate::tweedie::Tweedie;

/// The mass rule A leaves beyond the window, `1 − p`: `aggregate`'s
/// `bucket_sizing_p`, 0.99999.
pub const BUCKET_SIZING_P: f64 = 0.99999;

/// Frequency truncation `|φ(π / h)|` above which the bucket walks down the
/// ladder, and [`ChfReport::is_truncated`] is true.
pub const TRUNCATION_FLAG: f64 = 1e-6;

/// A finite support bound replaces a Chernoff edge that lies within this
/// share of the window of it.
const SUPPORT_EDGE_SHARE: f64 = 0.05;

/// The mantissas of [`round_bucket`]'s ladder for buckets of 1 and more.
const BUCKET_LADDER: [f64; 6] = [1.0, 2.0, 4.0, 5.0, 8.0, 10.0];

/// A characteristic function supplied by the user, evaluated at real
/// frequencies: `φ(t) = E e^{itX}` for each `t`.
pub type ChfFn = dyn Fn(&[f64]) -> Result<Vec<Complex64>> + Send + Sync;

/// A user's characteristic function, with no closed forms: the window
/// cannot be sized from the law, so [`ChfOptions`] must give the bucket
/// and the left edge (`aggregate`'s `AggregateCHF.from_chf`).
#[derive(Clone)]
pub struct UserChf {
    f: Arc<ChfFn>,
    label: String,
    parallel_safe: bool,
}

impl UserChf {
    /// Wraps `f`, labelled `label` in messages. `parallel_safe` is false
    /// for a callback that must stay on its thread (a Python or R
    /// function).
    pub fn new(f: Arc<ChfFn>, label: impl Into<String>, parallel_safe: bool) -> Self {
        Self {
            f,
            label: label.into(),
            parallel_safe,
        }
    }

    /// The label.
    pub fn label(&self) -> &str {
        &self.label
    }
}

impl fmt::Debug for UserChf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UserChf({:?})", self.label)
    }
}

/// A law with a closed-form characteristic function, as the engine reads
/// it: `φ`, the cumulant generating function `K` and its domain, the
/// support, an atom and the moments.
#[derive(Debug, Clone)]
pub enum ChfLaw {
    /// `φ(s) = exp(iμs − σ²s²/2)`, entire.
    Normal(Normal),
    /// `φ(s) = (1 − iθs)^(−k)` on the principal branch.
    Gamma(Gamma),
    /// `K(s) = (λ/μ)(1 − √(1 − 2μ²s/λ))`, `φ(s) = exp K(is)`.
    InverseGaussian(InverseGaussian),
    /// The Tweedie with `1 < p < 2`, compound Poisson with gamma
    /// severities: `K(s) = λ((1 − θs)^(−α) − 1)`, an atom `e^(−λ)` at 0.
    Tweedie(Tweedie),
    /// An alpha-stable law. With `β = ±1` its power tail is inverted along
    /// a tilted line ([`Tilt`]); with `|β| < 1` it is inverted untilted
    /// only.
    Stable(Stable),
    /// The normal inverse Gaussian law, light on both sides.
    NormalInverseGaussian(NormalInverseGaussian),
    /// The variance gamma law, light on both sides.
    VarianceGamma(VarianceGamma),
    /// The CGMY tempered stable law, light on both sides.
    TemperedStable(TemperedStable),
    /// `Y = scale · X + shift` with `scale > 0`:
    /// `φ_Y(s) = e^{i shift s} φ_X(scale s)`.
    Affine {
        base: Box<ChfLaw>,
        scale: f64,
        shift: f64,
    },
    /// A user's characteristic function on a given grid.
    User(UserChf),
}

impl ChfLaw {
    /// `scale · X + shift`, `scale` finite and positive, `shift` finite.
    pub fn affine(base: ChfLaw, scale: f64, shift: f64) -> Result<Self> {
        if matches!(base, Self::Stable(_)) {
            return Err(Error::Data(
                "a stable law takes its own scale and location: build it with them".into(),
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "scale",
                value: scale,
                reason: "must be finite and positive",
            });
        }
        if !shift.is_finite() {
            return Err(Error::InvalidParameter {
                name: "shift",
                value: shift,
                reason: "must be finite",
            });
        }
        Ok(Self::Affine {
            base: Box::new(base),
            scale,
            shift,
        })
    }

    /// The law of a native distribution with a closed-form characteristic
    /// function: a gamma, an exponential (a gamma of shape 1), an inverse
    /// Gaussian, a Tweedie, or an [`crate::Affine`] of one of them.
    pub fn from_dist(d: &Dist) -> Result<Self> {
        match d {
            Dist::Gamma(g) => Ok(Self::Gamma(*g)),
            Dist::Exponential(e) => Ok(Self::Gamma(Gamma::new(1.0, e.mean())?)),
            Dist::InverseGaussian(g) => Ok(Self::InverseGaussian(*g)),
            Dist::Tweedie(t) => Ok(Self::Tweedie(*t)),
            Dist::Affine(a) => {
                Self::affine(Self::from_dist(a.inner().dist())?, a.scale(), a.shift())
            }
            _ => Err(Error::Data(
                "the characteristic-function engine takes a normal, gamma, exponential, \
                 inverse Gaussian or Tweedie law, a scaled and shifted one, or a user's \
                 characteristic function"
                    .into(),
            )),
        }
    }

    /// The characteristic function at each `s`, on the closed upper half
    /// plane for the closed forms; a user's function only on the real line.
    pub fn chf(&self, s: &[Complex64]) -> Result<Vec<Complex64>> {
        let i = Complex64::new(0.0, 1.0);
        let one = Complex64::new(1.0, 0.0);
        Ok(match self {
            Self::Normal(d) => {
                let (mu, sd) = (d.mu(), d.sigma());
                s.iter()
                    .map(|&s| (i * mu * s - 0.5 * sd * sd * s * s).exp())
                    .collect()
            }
            Self::Gamma(d) => {
                let (k, th) = (d.shape(), d.scale());
                s.iter()
                    .map(|&s| (-k * (one - i * th * s).ln()).exp())
                    .collect()
            }
            Self::InverseGaussian(d) => {
                let (mu, lam) = (d.mean_param(), d.shape());
                s.iter()
                    .map(|&s| (lam / mu * (one - (one - 2.0 * i * mu * mu * s / lam).sqrt())).exp())
                    .collect()
            }
            Self::Tweedie(t) => {
                let g = t.severity();
                let (lam, a, th) = (t.lambda(), g.shape(), g.scale());
                s.iter()
                    .map(|&s| (lam * ((-a * (one - i * th * s).ln()).exp() - one)).exp())
                    .collect()
            }
            Self::Stable(st) => s.iter().map(|&s| st.chf(s)).collect::<Result<Vec<_>>>()?,
            Self::NormalInverseGaussian(d) => s.iter().map(|&s| d.chf(s)).collect(),
            Self::VarianceGamma(d) => s.iter().map(|&s| d.chf(s)).collect(),
            Self::TemperedStable(d) => s.iter().map(|&s| d.chf(s)).collect(),
            Self::Affine { base, scale, shift } => {
                let inner: Vec<Complex64> = s.iter().map(|&s| s * *scale).collect();
                base.chf(&inner)?
                    .into_iter()
                    .zip(s)
                    .map(|(c, &s)| (i * *shift * s).exp() * c)
                    .collect()
            }
            Self::User(u) => {
                if s.iter().any(|z| z.im != 0.0) {
                    return Err(Error::Data(format!(
                        "{}: a user's characteristic function is evaluated on the real line only",
                        u.label
                    )));
                }
                let t: Vec<f64> = s.iter().map(|z| z.re).collect();
                let out = (u.f)(&t)?;
                if out.len() != t.len() {
                    return Err(Error::Data(format!(
                        "{}: the characteristic function returned {} values for {} frequencies",
                        u.label,
                        out.len(),
                        t.len()
                    )));
                }
                if let Some(z) = out.iter().find(|z| !z.re.is_finite() || !z.im.is_finite()) {
                    return Err(Error::Data(format!(
                        "{}: the characteristic function returned {z}, which is not finite",
                        u.label
                    )));
                }
                out
            }
        })
    }

    /// `K(s) = ln E e^{sX}` at a real `s`, NaN outside its domain.
    pub fn cgf(&self, s: f64) -> f64 {
        let (lo, hi) = self.cgf_domain();
        if !(s > lo && s < hi) && s != 0.0 {
            return f64::NAN;
        }
        match self {
            Self::Normal(d) => d.mu() * s + 0.5 * d.sigma().powi(2) * s * s,
            Self::Gamma(d) => -d.shape() * (-d.scale() * s).ln_1p(),
            Self::InverseGaussian(d) => {
                let (mu, lam) = (d.mean_param(), d.shape());
                lam / mu * (1.0 - (1.0 - 2.0 * mu * mu * s / lam).sqrt())
            }
            Self::Tweedie(t) => {
                let g = t.severity();
                t.lambda() * ((1.0 - g.scale() * s).powf(-g.shape()) - 1.0)
            }
            Self::Stable(st) => st.cgf(s),
            Self::NormalInverseGaussian(d) => d.cgf(s),
            Self::VarianceGamma(d) => d.cgf(s),
            Self::TemperedStable(d) => d.cgf(s),
            Self::Affine { base, scale, shift } => base.cgf(scale * s) + shift * s,
            Self::User(_) => 0.0,
        }
    }

    /// `(s₋, s₊)` with `K` finite strictly inside; `(0, 0)` for a user's
    /// function, which has no `K`.
    pub fn cgf_domain(&self) -> (f64, f64) {
        match self {
            Self::Normal(_) => (f64::NEG_INFINITY, f64::INFINITY),
            Self::Gamma(d) => (f64::NEG_INFINITY, 1.0 / d.scale()),
            Self::InverseGaussian(d) => (
                f64::NEG_INFINITY,
                d.shape() / (2.0 * d.mean_param().powi(2)),
            ),
            Self::Tweedie(t) => (f64::NEG_INFINITY, 1.0 / t.severity().scale()),
            Self::Stable(st) => st.cgf_domain(),
            Self::NormalInverseGaussian(d) => d.cgf_domain(),
            Self::VarianceGamma(d) => d.cgf_domain(),
            Self::TemperedStable(d) => d.cgf_domain(),
            Self::Affine { base, scale, .. } => {
                let (lo, hi) = base.cgf_domain();
                (lo / scale, hi / scale)
            }
            Self::User(_) => (0.0, 0.0),
        }
    }

    /// `(lower, upper)` of the law.
    pub fn support(&self) -> (f64, f64) {
        match self {
            Self::Normal(_)
            | Self::User(_)
            | Self::NormalInverseGaussian(_)
            | Self::VarianceGamma(_)
            | Self::TemperedStable(_) => (f64::NEG_INFINITY, f64::INFINITY),
            Self::Gamma(_) | Self::InverseGaussian(_) | Self::Tweedie(_) => (0.0, f64::INFINITY),
            Self::Stable(st) => st.support(),
            Self::Affine { base, scale, shift } => {
                let (lo, hi) = base.support();
                (scale * lo + shift, scale * hi + shift)
            }
        }
    }

    /// `(x, w)` for a point mass `w` at `x`.
    pub fn atom(&self) -> Option<(f64, f64)> {
        match self {
            Self::Tweedie(t) => Some((0.0, (-t.lambda()).exp())),
            Self::Affine { base, scale, shift } => base.atom().map(|(x, w)| (scale * x + shift, w)),
            _ => None,
        }
    }

    /// The mean, NaN for a user's function.
    pub fn mean(&self) -> f64 {
        match self {
            Self::Normal(d) => d.mu(),
            Self::Gamma(d) => d.mean(),
            Self::InverseGaussian(d) => d.mean(),
            Self::Tweedie(t) => t.mean(),
            Self::Stable(st) => st.mean(),
            Self::NormalInverseGaussian(d) => d.mean(),
            Self::VarianceGamma(d) => d.mean(),
            Self::TemperedStable(d) => d.mean(),
            Self::Affine { base, scale, shift } => scale * base.mean() + shift,
            Self::User(_) => f64::NAN,
        }
    }

    /// The variance, NaN for a user's function.
    pub fn variance(&self) -> f64 {
        match self {
            Self::Normal(d) => d.sigma().powi(2),
            Self::Gamma(d) => d.variance(),
            Self::InverseGaussian(d) => d.variance(),
            Self::Tweedie(t) => t.variance(),
            Self::Stable(st) => st.variance(),
            Self::NormalInverseGaussian(d) => d.variance(),
            Self::VarianceGamma(d) => d.variance(),
            Self::TemperedStable(d) => d.variance(),
            Self::Affine { base, scale, .. } => scale * scale * base.variance(),
            Self::User(_) => f64::NAN,
        }
    }

    /// The body scale: the standard deviation, NaN when it is not finite
    /// and positive.
    pub fn scale(&self) -> f64 {
        if let Self::Stable(st) = self {
            return st.body_scale();
        }
        let v = self.variance();
        if v.is_finite() && v > 0.0 {
            v.sqrt()
        } else {
            f64::NAN
        }
    }

    /// The frequency truncation of a bucket `h`: `|φ(π/h)|` of the part the
    /// inversion transforms, the atom taken out (`aggregate`'s
    /// `truncation_at`).
    pub fn truncation_at(&self, h: f64) -> Result<f64> {
        let s = Complex64::new(std::f64::consts::PI / h, 0.0);
        let mut c = self.chf(&[s])?[0];
        if let Some((x, w)) = self.atom() {
            c -= w * Complex64::new(0.0, std::f64::consts::PI * x / h).exp();
        }
        Ok(c.norm())
    }

    /// `(A, B, α)` of a power tail `P(X > x) ~ A x^(−α)`,
    /// `P(X < −x) ~ B x^(−α)`: a stable law below `α = 2`.
    pub fn power_tails(&self) -> Option<(f64, f64, f64)> {
        match self {
            Self::Stable(st) if st.alpha() < 2.0 => {
                Some((st.tail_constant(), st.left_tail_constant(), st.alpha()))
            }
            _ => None,
        }
    }

    /// The side of a power tail, `None` without one.
    pub fn tilt_side(&self) -> Option<TiltSide> {
        match self {
            Self::Stable(st) if st.alpha() < 2.0 => Some(if st.beta() == 1.0 {
                TiltSide::Right
            } else if st.beta() == -1.0 {
                TiltSide::Left
            } else {
                TiltSide::Both
            }),
            _ => None,
        }
    }

    /// The law that tilting inverts: this one for a right power tail, its
    /// reflection for a left one, `None` otherwise.
    fn tilt_family(&self) -> Option<Stable> {
        match (self, self.tilt_side()) {
            (Self::Stable(st), Some(TiltSide::Right)) => Some(*st),
            (Self::Stable(st), Some(TiltSide::Left)) => Some(st.reflected()),
            _ => None,
        }
    }

    fn is_parallel_safe(&self) -> bool {
        match self {
            Self::User(u) => u.parallel_safe,
            Self::Affine { base, .. } => base.is_parallel_safe(),
            _ => true,
        }
    }
}

/// Which side of a law has a power tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiltSide {
    /// `β = 1`: the right tail.
    Right,
    /// `β = −1`: the left tail, inverted by reflection.
    Left,
    /// `|β| < 1`: both.
    Both,
}

/// Whether to invert along the line `Im s = θ`, with `τ = θ W` e-folds of
/// damping across the window of width `W` (`aggregate`'s `tilt`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Tilt {
    /// A law with one power tail (a stable law with `β = ±1`) at the `τ`
    /// that balances the damped wrap `ρ e^(−τ)` against the amplified
    /// round-off `ε κ e^τ`, its bucket capped; every other law untilted.
    #[default]
    Auto,
    /// Never tilt and never cap: a heavy tail wraps onto the window.
    Off,
    /// Exactly `τ ≥ 0` e-folds, the cap applied; only for a law with one
    /// power tail.
    Fixed(f64),
}

/// How the grid is chosen.
#[derive(Debug, Clone, PartialEq)]
pub struct ChfOptions {
    /// The grid has `2^log2` buckets, `log2` in 1 to 28. Default 16.
    pub log2: u32,
    /// The bucket. `None` sizes it by rule A from the tails.
    pub step: Option<f64>,
    /// The grid's left edge, with `step` given. `None` places it by the
    /// rule. Required, with `step`, for a user's function.
    pub x_min: Option<f64>,
    /// The mass rule A's window covers, in `(0, 1)`.
    pub bucket_sizing_p: f64,
    /// The tilt.
    pub tilt: Tilt,
}

impl Default for ChfOptions {
    fn default() -> Self {
        Self {
            log2: 16,
            step: None,
            x_min: None,
            bucket_sizing_p: BUCKET_SIZING_P,
            tilt: Tilt::Auto,
        }
    }
}

impl ChfOptions {
    /// Rule A at `2^log2` buckets.
    pub fn with_log2(log2: u32) -> Self {
        Self {
            log2,
            ..Self::default()
        }
    }
}

/// What the inversion did and the errors it introduced.
#[derive(Debug, Clone, PartialEq)]
pub struct ChfReport {
    /// The grid has `2^log2` buckets.
    pub log2: u32,
    /// The bucket `h`.
    pub step: f64,
    /// The first grid point.
    pub x_min: f64,
    /// Whether the bucket was given (rule C) rather than sized (rule A).
    pub step_given: bool,
    /// `|φ(π/h)|` of the continuous part: the size of the frequencies the
    /// inversion drops. Above [`TRUNCATION_FLAG`] the masses ring.
    pub truncation_error: f64,
    /// The law's mass below the window: a Chernoff bound, which wraps onto
    /// the window's top end, or under a tilt the tail mass beyond a left
    /// power tail's window, which is added to the first bucket; 0 on a
    /// grid whose left edge was given.
    pub outside_below: f64,
    /// The law's mass above the window: a Chernoff bound, which wraps onto
    /// the window's bottom end, or the tail mass beyond a right power
    /// tail's window, which a tilted grid leaves out.
    pub outside_above: f64,
    /// `(left, right)`: the e-folds of damping across the window on each
    /// side, `(0, τ)` for a right power tail tilted, `(τ, 0)` for a left
    /// one, `(0, 0)` untilted.
    pub tilt: (f64, f64),
    /// The modelled worst relative error of a bucket at the heavy end of a
    /// law with one power tail, `ρ e^(−τ) + ε κ e^τ`; NaN for any other
    /// law.
    pub edge_error: f64,
    /// The law's mean (NaN for a user's function).
    pub law_mean: f64,
    /// The law's variance (NaN for a user's function).
    pub law_variance: f64,
    /// The sum of the masses.
    pub total_mass: f64,
}

impl ChfReport {
    /// Whether the frequency truncation is above [`TRUNCATION_FLAG`].
    pub fn is_truncated(&self) -> bool {
        self.truncation_error > TRUNCATION_FLAG
    }

    /// `outside_below + outside_above`.
    pub fn outside_mass(&self) -> f64 {
        self.outside_below + self.outside_above
    }
}

/// The bucket masses of a law on `x_min + k · step`, `k < 2^log2`, from its
/// characteristic function: a [`Distribution`] whose statistics are those
/// of the masses on the window.
#[derive(Debug, Clone)]
pub struct ChfDistribution {
    x_min: f64,
    step: f64,
    masses: Vec<f64>,
    cumulative: Vec<f64>,
    report: ChfReport,
    parallel_safe: bool,
}

impl ChfDistribution {
    /// The first grid point.
    pub fn x_min(&self) -> f64 {
        self.x_min
    }

    /// The bucket.
    pub fn step(&self) -> f64 {
        self.step
    }

    /// The grid point `x_min + k · step`.
    pub fn x(&self, k: usize) -> f64 {
        self.x_min + self.step * k as f64
    }

    /// The grid points.
    pub fn xs(&self) -> Vec<f64> {
        (0..self.masses.len()).map(|k| self.x(k)).collect()
    }

    /// The bucket masses, as inverted: not renormalized, round-off of
    /// either sign kept.
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    /// The grid, the errors and the law's moments.
    pub fn report(&self) -> &ChfReport {
        &self.report
    }

    /// The masses as a [`Grid`] on `0, step, …`, with negative round-off
    /// set to 0 and the sum rescaled to 1, for the compound engines and the
    /// risk measures on a grid. Fails when the window starts below 0 or
    /// its left edge is not a whole number of buckets.
    pub fn to_grid(&self) -> Result<Grid> {
        let offset = self.x_min / self.step;
        let k0 = offset.round();
        if self.x_min < 0.0 || (offset - k0).abs() > 1e-9 * offset.abs().max(1.0) {
            return Err(Error::InvalidParameter {
                name: "x_min",
                value: self.x_min,
                reason: "a Grid starts at 0: the window must start at 0 or above, on a whole number of buckets",
            });
        }
        let mut probs = vec![0.0; k0 as usize];
        probs.extend(self.masses.iter().map(|p| p.max(0.0)));
        Grid::normalized(self.step, probs)
    }

    fn index_at_or_below(&self, x: f64) -> Option<usize> {
        let k = ((x - self.x_min) / self.step + 1e-9).floor();
        if k < 0.0 {
            None
        } else {
            Some((k as usize).min(self.masses.len() - 1))
        }
    }

    fn window_moment(&self, centre: f64, power: i32) -> f64 {
        let total = self.report.total_mass;
        self.masses
            .iter()
            .enumerate()
            .map(|(k, p)| (self.x(k) - centre).powi(power) * p)
            .sum::<f64>()
            / total
    }
}

impl Distribution for ChfDistribution {
    /// The mean of the masses on the window, over their sum.
    fn mean(&self) -> f64 {
        self.window_moment(0.0, 1)
    }

    /// The variance of the masses on the window.
    fn variance(&self) -> f64 {
        self.window_moment(self.mean(), 2)
    }

    /// The sum of the masses at grid points at or below `x`: the law's
    /// `F(x_k + h/2)` at a grid point `x_k`.
    fn cdf(&self, x: f64) -> f64 {
        match self.index_at_or_below(x) {
            None => 0.0,
            Some(k) => self.cumulative[k],
        }
    }

    /// The first grid point whose cumulative mass reaches `p`, the last
    /// one when none does.
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        let k = self.cumulative.partition_point(|&c| c < p);
        Ok(self.x(k.min(self.masses.len() - 1)))
    }

    fn is_parallel_safe(&self) -> bool {
        self.parallel_safe
    }
}

/// The law's distribution on the grid `options` chooses, by inverting its
/// characteristic function.
///
/// Fails when the options are out of range, when a user's function has no
/// `step` and `x_min`, when `x_min` is given without `step`, or when the
/// window cannot be sized from the law.
pub fn invert(law: &ChfLaw, options: &ChfOptions) -> Result<ChfDistribution> {
    if !(1..=28).contains(&options.log2) {
        return Err(Error::InvalidParameter {
            name: "log2",
            value: options.log2 as f64,
            reason: "must be 1 to 28",
        });
    }
    let p = options.bucket_sizing_p;
    if !(p > 0.0 && p < 1.0) {
        return Err(Error::InvalidParameter {
            name: "bucket_sizing_p",
            value: p,
            reason: "must be in (0, 1)",
        });
    }
    if let Some(h) = options.step
        && (!h.is_finite() || h <= 0.0)
    {
        return Err(Error::InvalidParameter {
            name: "step",
            value: h,
            reason: "must be finite and positive",
        });
    }
    if let Some(x) = options.x_min {
        if !x.is_finite() {
            return Err(Error::InvalidParameter {
                name: "x_min",
                value: x,
                reason: "must be finite",
            });
        }
        if options.step.is_none() {
            return Err(Error::Data(
                "x_min places a grid whose bucket is given: give step too".into(),
            ));
        }
    }
    if let ChfLaw::User(u) = law
        && (options.step.is_none() || options.x_min.is_none())
    {
        return Err(Error::Data(format!(
            "{}: a user's characteristic function says nothing of the law's tails, \
             so the grid must be given: step and x_min",
            u.label
        )));
    }
    let n = 1usize << options.log2;
    let fixed_grid = options.x_min.is_some();
    if let Tilt::Fixed(tau) = options.tilt {
        if !(tau.is_finite() && tau >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "tilt",
                value: tau,
                reason: "must be finite and at least 0",
            });
        }
        if law.power_tails().is_none() {
            return Err(Error::Data(
                "a tilt damps a power tail, which this law does not have: a light tail \
                 barely wraps, and a tilt only amplifies the truncation ringing; use the \
                 automatic tilt or none"
                    .into(),
            ));
        }
    }
    let tilting = options.tilt != Tilt::Off && !fixed_grid;
    if tilting && law.tilt_side() == Some(TiltSide::Both) {
        return Err(Error::Data(
            "a stable law with |beta| < 1 has two power tails, which no tilted line serves: \
             it is split into its two totally skewed halves, which is not built yet (#780); \
             invert it untilted"
                .into(),
        ));
    }
    let tilt_family = if tilting { law.tilt_family() } else { None };
    let (h, x0, tilt, edge_error, mut masses, below, above);
    match tilt_family {
        None => {
            (h, x0) = match (options.step, options.x_min) {
                (Some(h), Some(x)) => (h, x),
                _ => plain_window(law, n, p, options.step)?,
            };
            masses = invert_on(law, x0, h, n, 0.0, law.atom())?;
            (below, above) = outside_at(law, h, x0, n, fixed_grid)?;
            edge_error = match law.tilt_family() {
                Some(fam) if law.tilt_side() != Some(TiltSide::Both) => {
                    let y0 = if law.tilt_side() == Some(TiltSide::Right) {
                        x0
                    } else {
                        -(x0 + (n - 1) as f64 * h)
                    };
                    let (rho, eps, kappa) = edge_terms(&fam, y0, h, n);
                    edge_error_at(0.0, rho, eps, kappa)
                }
                _ => f64::NAN,
            };
            tilt = (0.0, 0.0);
        }
        Some(fam) => {
            let (hh, y0, tau) = one_sided_window(&fam, n, p, options.step, options.tilt);
            h = hh;
            let left = law.tilt_side() == Some(TiltSide::Left);
            x0 = if left { -(y0 + (n - 1) as f64 * h) } else { y0 };
            let w = n as f64 * h;
            masses = invert_on(&ChfLaw::Stable(fam), y0, h, n, tau / w, None)?;
            let (rho, eps, kappa) = edge_terms(&fam, y0, h, n);
            edge_error = edge_error_at(tau, rho, eps, kappa);
            if left {
                masses.reverse();
            }
            (below, above) = outside_at(law, h, x0, n, fixed_grid)?;
            masses[0] += below;
            tilt = if left { (tau, 0.0) } else { (0.0, tau) };
        }
    }
    let total_mass: f64 = masses.iter().sum();
    let mut cumulative = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &m in &masses {
        acc += m;
        cumulative.push(acc);
    }
    let report = ChfReport {
        log2: options.log2,
        step: h,
        x_min: x0,
        step_given: options.step.is_some(),
        truncation_error: law.truncation_at(h)?,
        outside_below: below,
        outside_above: above,
        tilt,
        edge_error,
        law_mean: law.mean(),
        law_variance: law.variance(),
        total_mass,
    };
    Ok(ChfDistribution {
        x_min: x0,
        step: h,
        masses,
        cumulative,
        report,
        parallel_safe: law.is_parallel_safe(),
    })
}

/// `sin z / z` at a complex `z`, 1 at 0.
fn csinc(z: Complex64) -> Complex64 {
    if z == Complex64::new(0.0, 0.0) {
        Complex64::new(1.0, 0.0)
    } else {
        z.sin() / z
    }
}

/// The bucket masses on `x0 + h · k`, `k < n`, inverted along `Im s = θ`
/// (`aggregate`'s `invert_characteristic_function`).
///
/// With `θ ≥ 0` the coefficient at frequency `t_j = 2πj/(nh)` is
/// `φ(−t_j + iθ) sinc((−t_j + iθ)h/2) e^{i t_j x0}`; the inverse transform
/// gives the damped masses `e^{−θ x_k} p_k`, and `e^{θ x_k}` the masses.
/// An atom `(x, w)` is taken out of `φ` and added back to its bucket, which
/// must be a grid point.
pub(crate) fn invert_on(
    law: &ChfLaw,
    x0: f64,
    h: f64,
    n: usize,
    theta: f64,
    atom: Option<(f64, f64)>,
) -> Result<Vec<f64>> {
    let i = Complex64::new(0.0, 1.0);
    let half = n / 2;
    let t: Vec<f64> = (0..=half)
        .map(|j| 2.0 * std::f64::consts::PI * j as f64 / (n as f64 * h))
        .collect();
    let s: Vec<Complex64> = t.iter().map(|&t| Complex64::new(-t, theta)).collect();
    let mut c = law.chf(&s)?;
    for ((cj, &sj), &tj) in c.iter_mut().zip(&s).zip(&t) {
        if let Some((xa, wa)) = atom {
            *cj -= wa * (i * sj * xa).exp();
        }
        *cj *= csinc(sj * h / 2.0) * (i * tj * x0).exp();
    }
    // numpy's irfft: the spectrum filled by Hermitian symmetry, the
    // imaginary parts at 0 and n/2 dropped.
    let mut buffer = vec![Complex64::new(0.0, 0.0); n];
    for j in 0..=half {
        let mut cj = c[j];
        if j == 0 || j == half {
            cj.im = 0.0;
        }
        buffer[j % n] = cj;
        if j > 0 && j < half {
            buffer[n - j] = cj.conj();
        }
    }
    let mut planner = FftPlanner::<f64>::new();
    planner.plan_fft_inverse(n).process(&mut buffer);
    let mut p: Vec<f64> = buffer
        .iter()
        .enumerate()
        .map(|(k, z)| {
            let v = z.re / n as f64;
            if theta != 0.0 {
                v * (theta * (x0 + h * k as f64)).exp()
            } else {
                v
            }
        })
        .collect();
    if let Some((xa, wa)) = atom {
        let kf = ((xa - x0) / h).round();
        let k = kf as isize;
        if k < 0 || k as usize >= n || (x0 + kf * h - xa).abs() > 1e-9 * h.max(1.0) {
            return Err(Error::Data(format!(
                "the atom at {xa} is not a grid point of x_min = {x0}, step = {h}, {n} buckets"
            )));
        }
        p[k as usize] += wa;
    }
    Ok(p)
}

/// The smallest "nice" bucket at or above `bs`: `{1, 2, 4, 5, 8} · 10^k`
/// from 1 up, a power of two below 1 (`aggregate`'s `round_bucket`, as
/// `prospicio_aggregate::round_bucket`, which this crate cannot depend on).
fn round_bucket(bs: f64) -> f64 {
    if bs >= 1.0 {
        let exp = bs.log10().floor() as i32;
        let base = 10f64.powi(exp);
        let m = bs / base;
        for rung in BUCKET_LADDER {
            if m <= rung * (1.0 + 1e-9) {
                return rung * base;
            }
        }
        return 10.0 * base;
    }
    // the reciprocal of the largest power of two at or below floor(1 / bs)
    let n = (1.0 / bs).floor();
    1.0 / 2f64.powi(n.log2().floor() as i32)
}

/// The largest rung of [`round_bucket`]'s ladder at or below `bs`.
fn round_bucket_down(bs: f64) -> f64 {
    if bs >= 1.0 {
        let exp = bs.log10().floor() as i32;
        let base = 10f64.powi(exp);
        let m = bs / base;
        let mut best = 1.0;
        for rung in &BUCKET_LADDER[..5] {
            if *rung <= m * (1.0 + 1e-9) {
                best = *rung;
            }
        }
        return best * base;
    }
    2f64.powi(bs.log2().floor() as i32)
}

/// The scan of `s` on a log ladder of 217 points over `1e-9` to `1e9` body
/// scales below `s_max`, the best point refined by Brent between its
/// neighbours: `min_s g(s)`.
fn scan_min(s_max: f64, scale: f64, g: impl Fn(f64) -> f64) -> f64 {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    let s: Vec<f64> = (0..217)
        .map(|k| 10f64.powf(-9.0 + 18.0 * k as f64 / 216.0) / scale)
        .filter(|&s| s < s_max)
        .collect();
    if s.is_empty() {
        return f64::INFINITY;
    }
    let vals: Vec<f64> = s
        .iter()
        .map(|&u| {
            let v = g(u);
            if v.is_finite() { v } else { f64::INFINITY }
        })
        .collect();
    let mut i = 0;
    for (k, &v) in vals.iter().enumerate() {
        if v < vals[i] {
            i = k;
        }
    }
    let best = vals[i];
    if !best.is_finite() {
        return f64::INFINITY;
    }
    let lo = s[i.saturating_sub(1)];
    let hi = s[(i + 1).min(s.len() - 1)];
    if hi > lo {
        let (_, fx, _) = brent(lo, hi, 1e-10, |u| {
            let v = g(u);
            if v.is_finite() { v } else { f64::INFINITY }
        });
        if fx.is_finite() && fx < best {
            return fx;
        }
    }
    best
}

/// The Chernoff edge `inf_{0<s<s_max} (K(s) − ln ε) / s`: a point with
/// mass at most `ε` above it.
fn chernoff_edge(cgf: impl Fn(f64) -> f64, s_max: f64, log_eps: f64, scale: f64) -> f64 {
    scan_min(s_max, scale, |s| (cgf(s) - log_eps) / s)
}

/// The Chernoff bound `exp inf_{0<s<s_max} (K(s) − s x)` on `P(X > x)`, at
/// most 1.
fn chernoff_tail(cgf: impl Fn(f64) -> f64, s_max: f64, x: f64, scale: f64) -> f64 {
    let best = scan_min(s_max, scale, |s| cgf(s) - s * x);
    if best.is_finite() {
        best.min(0.0).exp().min(1.0)
    } else {
        1.0
    }
}

/// The window edges with mass `eps_side` beyond each.
fn window_edges(law: &ChfLaw, eps_side: f64) -> (f64, f64) {
    let (lo, hi) = law.support();
    let log_eps = eps_side.ln();
    let scale = law.scale();
    let (s_minus, s_plus) = law.cgf_domain();
    let (a, b, alpha) = law.power_tails().unwrap_or((0.0, 0.0, 1.0));
    let location = match law {
        ChfLaw::Stable(st) => st.loc(),
        _ => 0.0,
    };
    let mut x_r = if a > 0.0 {
        location + (a / eps_side).powf(1.0 / alpha)
    } else if s_plus > 0.0 {
        chernoff_edge(|s| law.cgf(s), s_plus, log_eps, scale)
    } else {
        f64::INFINITY
    };
    let mut x_l = if b > 0.0 {
        location - (b / eps_side).powf(1.0 / alpha)
    } else if s_minus < 0.0 {
        -chernoff_edge(|u| law.cgf(-u), -s_minus, log_eps, scale)
    } else {
        f64::NEG_INFINITY
    };
    x_r = x_r.min(hi);
    x_l = x_l.max(lo);
    if lo.is_finite() && x_l > lo && x_l - lo < SUPPORT_EDGE_SHARE * (x_r - lo) {
        x_l = lo;
    }
    if hi.is_finite() && x_r < hi && hi - x_r < SUPPORT_EDGE_SHARE * (hi - x_l) {
        x_r = hi;
    }
    (x_l, x_r)
}

/// The Chernoff bounds on the mass below and above `[x0, x0 + n h)`.
fn light_outside(law: &ChfLaw, h: f64, x0: f64, n: usize) -> (f64, f64) {
    let scale = law.scale();
    if !scale.is_finite() && law.cgf_domain() == (0.0, 0.0) {
        return (0.0, 0.0);
    }
    let (lo, hi) = law.support();
    let (s_minus, s_plus) = law.cgf_domain();
    let x_top = x0 + n as f64 * h;
    let above = if x_top >= hi {
        0.0
    } else if s_plus > 0.0 {
        chernoff_tail(|s| law.cgf(s), s_plus, x_top, scale)
    } else {
        1.0
    };
    let below = if x0 <= lo {
        0.0
    } else if s_minus < 0.0 {
        chernoff_tail(|u| law.cgf(-u), -s_minus, -x0, scale)
    } else {
        1.0
    };
    (below, above)
}

/// Rule A (or rule C with `step` given) with no tilt: `(h, x0)`.
///
/// The window is sized from the edges at half the budget each and the
/// slack of the rounding split between the sides, cut at a support bound;
/// 0, or the atom, stays a grid point. A bucket whose truncation is above
/// [`TRUNCATION_FLAG`] walks down the ladder (at most twenty rungs) while
/// that lowers the larger of the truncation and the mass outside.
fn plain_window(law: &ChfLaw, n: usize, p: f64, step: Option<f64>) -> Result<(f64, f64)> {
    let eps = 1.0 - p;
    let (x_l, x_r) = window_edges(law, eps / 2.0);
    if !(x_l.is_finite() && x_r.is_finite()) || x_r <= x_l {
        return Err(Error::Data(
            "cannot size a window for this law from its tails: give step and x_min".into(),
        ));
    }
    let (lo, _) = law.support();
    let anchor = law.atom().map_or(0.0, |(x, _)| x);
    let place = |h: f64| {
        let width = n as f64 * h;
        let x_left = if width >= x_r - x_l {
            let slack = width - (x_r - x_l);
            lo.max(x_l - slack / 2.0)
        } else {
            let mean = law.mean();
            let centre = if mean.is_finite() {
                mean
            } else {
                0.5 * (x_l + x_r)
            };
            lo.max((centre - width / 2.0).max(lo).min(x_r - width))
        };
        anchor + h * ((x_left - anchor) / h + 1e-12).floor()
    };
    if let Some(h) = step {
        return Ok((h, place(h)));
    }
    let mut h = round_bucket((x_r - x_l) / n as f64);
    let trunc = law.truncation_at(h)?;
    if trunc <= TRUNCATION_FLAG {
        return Ok((h, place(h)));
    }
    let (mut best_h, mut best_x0) = (h, place(h));
    let outside = |h: f64, x0: f64| {
        let (b, a) = light_outside(law, h, x0, n);
        b + a
    };
    let mut best = trunc.max(outside(h, best_x0));
    for _ in 0..20 {
        h = round_bucket_down(h * (1.0 - 1e-9));
        let x0 = place(h);
        let trunc = law.truncation_at(h)?;
        let score = trunc.max(outside(h, x0));
        if score < best {
            (best, best_h, best_x0) = (score, h, x0);
        } else if trunc < score {
            break;
        }
    }
    Ok((best_h, best_x0))
}

/// The mass allowed below a tilted window: the tilt amplifies what wraps
/// from below by `e^τ`, and the light side's super-exponential tail makes
/// this edge cost a few scales.
const TILT_LEFT_EPS: f64 = 1e-40;

/// Buckets per scale a tilted grid keeps in the body.
const TILT_BODY_BUCKETS: f64 = 10.0;

/// The largest bucket a tilted grid may use, a rung of the ladder:
/// `min(γ/10, πγ/(52 ln 2)^(1/α))`, so the truncation `|φ(π/h)|`, which the
/// tilt amplifies by `e^τ`, sits at round-off.
fn bucket_cap(fam: &Stable) -> f64 {
    let gamma = fam.body_scale();
    let h_trunc = std::f64::consts::PI * gamma / (52.0 * 2f64.ln()).powf(1.0 / fam.alpha());
    round_bucket_down((gamma / TILT_BODY_BUCKETS).min(h_trunc))
}

/// `(ρ, ε, κ)` of the one-sided tilt model on `[x0, x0 + n h]`: the wrap
/// the tilt damps, `ρ e^(−τ)`, with `ρ = q^(α+1) ζ(α+1, 1+q)`,
/// `q = (x_R − δ₁)/W`; the amplified error `ε κ e^τ`, `ε = 2^(−52) +
/// |φ(π/h)|`, `κ = sup f (x_R − δ₁)^(α+1) / (A α)`, both relative to the
/// bucket mass at the right edge.
fn edge_terms(fam: &Stable, x0: f64, h: f64, n: usize) -> (f64, f64, f64) {
    let (a, alpha) = (fam.tail_constant(), fam.alpha());
    let w = n as f64 * h;
    let x_r = x0 + w;
    let q = ((x_r - fam.loc()) / w).max(1e-12);
    let rho = q.powf(alpha + 1.0) * hurwitz_zeta(alpha + 1.0, 1.0 + q);
    let eps = 2f64.powi(-52) + fam.chf_real(std::f64::consts::PI / h).norm();
    let kappa = fam.density_bound() * (x_r - fam.loc()).powf(alpha + 1.0) / (a * alpha);
    (rho, eps, kappa)
}

/// `τ* = max(0, ½ ln(ρ / (ε κ)))`.
fn tau_star(rho: f64, eps: f64, kappa: f64) -> f64 {
    (0.5 * (rho / (eps * kappa)).ln()).max(0.0)
}

/// `ρ e^(−τ) + ε κ e^τ`.
fn edge_error_at(tau: f64, rho: f64, eps: f64, kappa: f64) -> f64 {
    rho * (-tau).exp() + eps * kappa * tau.exp()
}

/// Rule A (or C) for a law with a right power tail, to be tilted:
/// `(h, x0, τ)`. The left edge is the support bound or the `1e-40`
/// Chernoff edge, the right edge `δ₁ + (A/ε)^(1/α)`, the bucket rounded up
/// and capped.
fn one_sided_window(
    fam: &Stable,
    n: usize,
    p: f64,
    step: Option<f64>,
    tilt: Tilt,
) -> (f64, f64, f64) {
    let eps = 1.0 - p;
    let (lo, _) = fam.support();
    let x_l = if lo.is_finite() {
        lo
    } else {
        let (s_minus, _) = fam.cgf_domain();
        -chernoff_edge(
            |u| fam.cgf(-u),
            -s_minus,
            TILT_LEFT_EPS.ln(),
            fam.body_scale(),
        )
    };
    let x_r = fam.loc() + (fam.tail_constant() / eps).powf(1.0 / fam.alpha());
    let h = match step {
        Some(h) => h,
        None => round_bucket((x_r - x_l) / n as f64).min(bucket_cap(fam)),
    };
    let x0 = h * (x_l / h + 1e-12).floor();
    let tau = match tilt {
        Tilt::Fixed(tau) => tau,
        _ => {
            let (rho, e, kappa) = edge_terms(fam, x0, h, n);
            tau_star(rho, e, kappa)
        }
    };
    (h, x0, tau)
}

/// `(below, above)`: the law's mass outside `[x0, x0 + n h)`. A law with
/// one power tail gives the tail mass beyond its window (on the heavy
/// side); two power tails an estimate from the leading terms on a window
/// placed to balance them; any other law the Chernoff bounds, 0 on a grid
/// whose left edge was given.
fn outside_at(law: &ChfLaw, h: f64, x0: f64, n: usize, fixed_grid: bool) -> Result<(f64, f64)> {
    match law.tilt_side() {
        Some(TiltSide::Both) => {
            let (a, b, alpha) = law
                .power_tails()
                .expect("a stable law with two power tails");
            let w = n as f64 * h;
            let d_plus = w / (1.0 + (b / a).powf(1.0 / alpha));
            let d_minus = w - d_plus;
            Ok((b * d_minus.powf(-alpha), a * d_plus.powf(-alpha)))
        }
        Some(TiltSide::Right) => {
            let fam = law.tilt_family().expect("a right power tail");
            Ok((0.0, fam.tail_mass_above(x0 + (n - 1) as f64 * h + h / 2.0)?))
        }
        Some(TiltSide::Left) => {
            let fam = law.tilt_family().expect("a left power tail");
            Ok((fam.tail_mass_above(-(x0 - h / 2.0))?, 0.0))
        }
        None if fixed_grid => Ok((0.0, 0.0)),
        None => Ok(light_outside(law, h, x0, n)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Severity;

    fn gamma() -> ChfLaw {
        ChfLaw::Gamma(Gamma::new(4.0, 25.0).unwrap())
    }

    #[test]
    fn round_bucket_matches_the_ladder() {
        for (bs, up) in [
            (1.0, 1.0),
            (1.1, 2.0),
            (2.5, 4.0),
            (5.5, 8.0),
            (9.9, 10.0),
            (13.0, 20.0),
            (457.0, 500.0),
            (0.3, 0.5),
            (0.26, 0.5),
            (0.25, 0.25),
            (1.0 / 3.0, 0.5),
            (0.0011, 1.0 / 512.0),
        ] {
            assert_eq!(round_bucket(bs), up, "{bs}");
        }
        for (bs, down) in [
            (1.0, 1.0),
            (3.9, 2.0),
            (7.0, 5.0),
            (99.0, 80.0),
            (0.3, 0.25),
        ] {
            assert_eq!(round_bucket_down(bs), down, "{bs}");
        }
    }

    #[test]
    fn a_light_law_inverts_alike_on_a_tilted_line() {
        // The contour moves within the strip where phi is analytic, so the
        // masses are the same but for the ringing of the dropped
        // frequencies (|phi(pi/h)| = 1e-10 here), which the tilt amplifies
        // by e^(theta x), e^2 at the top of the window.
        let law = gamma();
        let theta = 2.0 / 1024.0;
        let plain = invert_on(&law, 0.0, 0.25, 4096, 0.0, None).unwrap();
        let tilted = invert_on(&law, 0.0, 0.25, 4096, theta, None).unwrap();
        let mut compared = 0;
        for (k, (a, b)) in plain.iter().zip(&tilted).enumerate() {
            if *a > 1e-8 {
                assert!((a / b - 1.0).abs() < 1e-8, "{k}: {a} {b}");
                compared += 1;
            }
        }
        assert!(compared > 1000);
    }

    #[test]
    fn masses_are_the_rounding_discretization() {
        // Bucket k holds P(x_k - h/2 < X <= x_k + h/2), the law's own
        // probability, not a density sample.
        let law = gamma();
        let d = invert(&law, &ChfOptions::with_log2(12)).unwrap();
        let g = Gamma::new(4.0, 25.0).unwrap();
        for k in [10, 200, 400, 800, 1600] {
            let x = d.x(k);
            let exact = g.cdf(x + 0.125) - g.cdf((x - 0.125).max(0.0));
            assert!((d.masses()[k] - exact).abs() < 1e-12, "{k}");
        }
        assert!((d.cdf(100.0) - g.cdf(100.125)).abs() < 1e-11);
        assert!((d.mean() - g.mean()).abs() < 1e-6);
    }

    #[test]
    fn the_atom_is_added_back_to_its_bucket() {
        let t = Tweedie::new(100.0, 2.0, 1.5).unwrap();
        let law = ChfLaw::Tweedie(t);
        let d = invert(&law, &ChfOptions::with_log2(12)).unwrap();
        assert_eq!(d.x_min(), 0.0);
        let p0 = (-t.lambda()).exp();
        // The bucket at 0 holds the atom and the continuous mass below h/2.
        let cont = t.cdf(d.step() / 2.0) - p0;
        assert!((d.masses()[0] - p0 - cont).abs() < 1e-7);
        assert!((d.mean() - 100.0).abs() < 1e-3);
    }

    #[test]
    fn affine_moves_the_window_and_the_atom() {
        let t = ChfLaw::Tweedie(Tweedie::new(100.0, 2.0, 1.5).unwrap());
        let law = ChfLaw::affine(t, 0.5, 10.0).unwrap();
        assert_eq!(law.atom().unwrap().0, 10.0);
        let d = invert(&law, &ChfOptions::with_log2(12)).unwrap();
        assert!(d.x_min() <= 10.0);
        assert!((d.mean() - 60.0).abs() < 1e-3);
        assert!(ChfLaw::affine(gamma(), 0.0, 1.0).is_err());
    }

    #[test]
    fn a_normal_reaches_below_zero() {
        let law = ChfLaw::Normal(Normal::new(0.0, 1.0).unwrap());
        let d = invert(&law, &ChfOptions::with_log2(12)).unwrap();
        assert!(d.x_min() < -7.0);
        assert!((d.quantile(0.5).unwrap()).abs() < 1e-12);
        assert!(d.to_grid().is_err());
    }

    #[test]
    fn a_user_function_needs_its_grid() {
        let f: Arc<ChfFn> = Arc::new(|t: &[f64]| {
            Ok(t.iter()
                .map(|&t| (Complex64::new(1.0, -25.0 * t)).powf(-4.0))
                .collect())
        });
        let law = ChfLaw::User(UserChf::new(f, "gamma", true));
        assert!(invert(&law, &ChfOptions::with_log2(12)).is_err());
        let opts = ChfOptions {
            log2: 12,
            step: Some(0.25),
            x_min: Some(0.0),
            ..ChfOptions::default()
        };
        let d = invert(&law, &opts).unwrap();
        let r = invert(&gamma(), &ChfOptions::with_log2(12)).unwrap();
        for (a, b) in d.masses().iter().zip(r.masses()) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!(d.report().law_mean.is_nan());
        assert_eq!(d.report().outside_mass(), 0.0);
    }

    #[test]
    fn to_grid_pads_from_zero() {
        let law = ChfLaw::affine(gamma(), 1.0, 1000.0).unwrap();
        let d = invert(&law, &ChfOptions::with_log2(12)).unwrap();
        let g = d.to_grid().unwrap();
        assert!((g.mean() - d.mean()).abs() < 1e-6);
        assert!((g.mean() - 1100.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_bad_options() {
        let law = gamma();
        for o in [
            ChfOptions::with_log2(0),
            ChfOptions::with_log2(29),
            ChfOptions {
                bucket_sizing_p: 1.0,
                ..ChfOptions::default()
            },
            ChfOptions {
                step: Some(-1.0),
                ..ChfOptions::default()
            },
            ChfOptions {
                x_min: Some(0.0),
                ..ChfOptions::default()
            },
        ] {
            assert!(invert(&law, &o).is_err(), "{o:?}");
        }
        assert!(ChfLaw::from_dist(&Dist::from(crate::Lognormal::new(0.0, 1.0).unwrap())).is_err());
    }

    #[test]
    fn lev_of_the_window_is_the_laws() {
        let law = gamma();
        let d = invert(&law, &ChfOptions::with_log2(14)).unwrap();
        let g = d.to_grid().unwrap();
        let exact = Gamma::new(4.0, 25.0).unwrap();
        assert!((g.lev(150.0) / exact.lev(150.0) - 1.0).abs() < 1e-4);
    }

    fn stable(alpha: f64, beta: f64, scale: f64, loc: f64) -> ChfLaw {
        ChfLaw::Stable(
            Stable::new(alpha, beta, scale, loc, crate::StableParameterization::S1).unwrap(),
        )
    }

    #[test]
    fn a_left_tail_is_the_reflection_of_a_right_one() {
        let right = invert(&stable(1.5, 1.0, 2.0, 3.0), &ChfOptions::with_log2(12)).unwrap();
        let left = invert(&stable(1.5, -1.0, 2.0, -3.0), &ChfOptions::with_log2(12)).unwrap();
        let (r, l) = (right.report(), left.report());
        assert_eq!((r.tilt.0, l.tilt.1), (0.0, 0.0));
        assert!(r.tilt.1 > 0.0 && (r.tilt.1 - l.tilt.0).abs() < 1e-12);
        assert!((left.x_min() + right.x(right.masses().len() - 1)).abs() < 1e-9);
        // The left grid's first bucket also holds the mass below the window.
        let n = right.masses().len();
        assert!((left.masses()[0] - right.masses()[n - 1] - l.outside_below).abs() < 1e-12);
        for k in 1..n {
            assert!((left.masses()[k] - right.masses()[n - 1 - k]).abs() < 1e-12);
        }
    }

    #[test]
    fn the_automatic_tilt_shrinks_the_edge_error() {
        let law = stable(1.5, 1.0, 1.0, 0.0);
        let auto = invert(&law, &ChfOptions::with_log2(12)).unwrap();
        let off = invert(
            &law,
            &ChfOptions {
                log2: 12,
                tilt: Tilt::Off,
                ..ChfOptions::default()
            },
        )
        .unwrap();
        assert!(auto.report().edge_error < 1e-4 && off.report().edge_error > 1.0);
        // Under a tilt the mass above the window is left out and reported.
        let r = auto.report();
        assert!((r.total_mass + r.outside_above - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tilts_are_refused_where_they_do_not_apply() {
        let fixed = ChfOptions {
            tilt: Tilt::Fixed(4.0),
            ..ChfOptions::with_log2(10)
        };
        assert!(invert(&gamma(), &fixed).is_err());
        assert!(invert(&gamma(), &ChfOptions::with_log2(10)).is_ok());
        let two_sided = stable(1.5, 0.5, 1.0, 0.0);
        assert!(invert(&two_sided, &ChfOptions::with_log2(10)).is_err());
        let off = ChfOptions {
            tilt: Tilt::Off,
            ..ChfOptions::with_log2(10)
        };
        assert!(invert(&two_sided, &off).is_ok());
        assert!(ChfLaw::affine(two_sided, 2.0, 1.0).is_err());
        let bad = ChfOptions {
            tilt: Tilt::Fixed(-1.0),
            ..ChfOptions::with_log2(10)
        };
        assert!(invert(&stable(1.5, 1.0, 1.0, 0.0), &bad).is_err());
    }
}
