//! Exposure curves for property per-risk rating: the share of a risk's
//! expected loss below a fraction of its maximum possible loss (MPL), and
//! from it the share in a layer.
//!
//! An exposure curve is `G(x) = E[min(X, x)] / E[X]` for the destruction
//! rate `X = loss / MPL` in `[0, 1]`: concave, `G(0) = 0`, `G(1) = 1`. A
//! layer `l xs a` on a risk with MPL `M` takes the share
//! `G(min((a + l)/M, 1)) − G(min(a/M, 1))` of the risk's expected loss.
//!
//! - [`Mbbefd`]: Bernegger's (1997) MBBEFD class, two parameters `b` and
//!   `g` (`1/g` is the probability of a total loss), and the Swiss Re
//!   curves `c = 1.5, 2, 3, 4` (and Lloyd's `c = 5`) as
//!   [`Mbbefd::swiss_re`].
//! - [`SeverityCurve`]: the exposure curve of any [`Severity`] capped at
//!   an MPL, `LEV(x M) / LEV(M)` (a Pareto gives Riebesell's scale).
//! - [`TabulatedCurve`]: a published table of `(x, G(x))` points (Salzmann,
//!   Ludwig, ISO PSOLD, a reinsurer's curves), interpolated linearly or
//!   log-log ([`Interpolation`]).
//!
//! Every curve is also a destruction-rate distribution: the survival is
//! `G'(x) / G'(0)`, so [`ExposureCurve::rate_quantile`] draws the loss as a
//! share of the MPL and [`ExposureCurve::mean_rate`] is `1 / G'(0)`.
//!
//! Bernegger, S. (1997). The Swiss Re exposure curves and the MBBEFD
//! distribution class. ASTIN Bulletin 27(1), 99–111.

use prospicio_core::{Error, Result};
use prospicio_prob::Severity;

/// An exposure curve: `G(x)` on `[0, 1]`, the expected loss below a
/// fraction `x` of the MPL over the expected loss.
pub trait ExposureCurve {
    /// `G(x)`; `x` is clamped to `[0, 1]`.
    fn g(&self, x: f64) -> f64;

    /// The destruction rate (loss over MPL, in `[0, 1]`) at probability
    /// `u` in `(0, 1)`: the quantile of the distribution whose survival is
    /// `G'(x) / G'(0)`. Simulating `mpl × rate_quantile(u)` gives losses
    /// whose expected layer shares are this curve's.
    fn rate_quantile(&self, u: f64) -> f64;

    /// Mean destruction rate, `1 / G'(0)`: a risk's expected loss is its
    /// MPL times this.
    fn mean_rate(&self) -> f64;

    /// Share of a risk's expected loss in the layer `limit` xs
    /// `attachment`, for a risk with maximum possible loss `mpl`.
    ///
    /// ```
    /// use prospicio_pricing::exposure::{ExposureCurve, Mbbefd};
    ///
    /// let c3 = Mbbefd::swiss_re(3.0).unwrap();
    /// // A risk of MPL 10m: the layer 5m xs 5m takes the top half.
    /// let top = c3.layer_share(5e6, 5e6, 10e6).unwrap();
    /// let bottom = c3.layer_share(5e6, 0.0, 10e6).unwrap();
    /// assert!((top + bottom - 1.0).abs() < 1e-12);
    /// assert!(top < bottom); // concave: most of the loss is low
    /// ```
    fn layer_share(&self, limit: f64, attachment: f64, mpl: f64) -> Result<f64> {
        if !(mpl.is_finite() && mpl > 0.0) {
            return Err(invalid("mpl", mpl, "must be positive and finite"));
        }
        if limit.is_nan() || limit <= 0.0 {
            return Err(invalid("limit", limit, "must be positive"));
        }
        if !(attachment.is_finite() && attachment >= 0.0) {
            return Err(invalid(
                "attachment",
                attachment,
                "must be non-negative and finite",
            ));
        }
        Ok(self.g((attachment + limit) / mpl) - self.g(attachment / mpl))
    }
}

/// The MBBEFD exposure curve and destruction-rate distribution of
/// Bernegger (1997), with `b ≥ 0` and `g ≥ 1`.
///
/// | Case | `G(x)` | `F(x)`, `x < 1` |
/// |---|---|---|
/// | `g = 1` or `b = 0` | `x` | `0` (a total loss) |
/// | `b = 1` | `ln(1 + (g − 1)x) / ln g` | `1 − 1/(1 + (g − 1)x)` |
/// | `bg = 1` | `(1 − bˣ)/(1 − b)` | `1 − bˣ` |
/// | otherwise | `ln(((g − 1)b + (1 − gb)bˣ)/(1 − b)) / ln(gb)` | `1 − (1 − b)/((g − 1)b¹⁻ˣ + 1 − gb)` |
///
/// `F(1) = 1`, so a total loss has probability `1/g`.
///
/// ```
/// use prospicio_pricing::exposure::{ExposureCurve, Mbbefd};
///
/// let m = Mbbefd::new(0.5, 4.0).unwrap();
/// assert_eq!(m.g(0.0), 0.0);
/// assert!((m.g(1.0) - 1.0).abs() < 1e-15);
/// assert_eq!(m.total_loss_probability(), 0.25);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mbbefd {
    b: f64,
    g: f64,
}

/// Distance from the special cases `b = 1` and `bg = 1` within which their
/// closed forms are used (the general one cancels there).
const SPECIAL: f64 = 1e-10;

impl Mbbefd {
    /// The curve with parameters `b ≥ 0` and `g ≥ 1`, both finite.
    pub fn new(b: f64, g: f64) -> Result<Self> {
        if !(b.is_finite() && b >= 0.0) {
            return Err(invalid("b", b, "must be finite and non-negative"));
        }
        if !(g.is_finite() && g >= 1.0) {
            return Err(invalid("g", g, "must be finite and at least 1"));
        }
        Ok(Self { b, g })
    }

    /// Bernegger's one-parameter family through the Swiss Re curves:
    /// `b = exp(3.1 − 0.15(1 + c)c)`, `g = exp((0.78 + 0.12c)c)`, so
    /// `c = 1.5, 2, 3, 4` are Swiss Re's Y1–Y4 and `c = 5` is the Lloyd's
    /// curve; `c = 0` is the straight line (always a total loss).
    pub fn swiss_re(c: f64) -> Result<Self> {
        if !(c.is_finite() && c >= 0.0) {
            return Err(invalid("c", c, "must be finite and non-negative"));
        }
        Self::new(
            (3.1 - 0.15 * (1.0 + c) * c).exp(),
            ((0.78 + 0.12 * c) * c).exp(),
        )
    }

    /// The Swiss Re curve ([`Mbbefd::swiss_re`]) whose mean destruction
    /// rate (expected loss over MPL) is `rate`, in `(0, 1]`: the `c` that
    /// matches a cedant's loss cost per unit of sum insured, by bisection
    /// on `c` in `[0, 10]`, where the mean rate falls from 1. Returns the
    /// curve and its `c`.
    ///
    /// ```
    /// use prospicio_pricing::exposure::{ExposureCurve, Mbbefd};
    ///
    /// let target = Mbbefd::swiss_re(3.0).unwrap().mean_rate();
    /// let (curve, c) = Mbbefd::swiss_re_for_mean_rate(target).unwrap();
    /// assert!((c - 3.0).abs() < 1e-9);
    /// assert!((curve.mean_rate() / target - 1.0).abs() < 1e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// A rate outside what `c` in `[0, 10]` reaches.
    pub fn swiss_re_for_mean_rate(rate: f64) -> Result<(Self, f64)> {
        Self::fit_swiss_re("mean rate", rate, |m| m.mean_rate())
    }

    /// The Swiss Re curve whose share of a risk's expected loss in `limit`
    /// xs `attachment`, for a risk of MPL `mpl`, is `share`: the `c` that
    /// matches a cedant's experience in one layer, by bisection on `c` in
    /// `[0, 10]` (a share above the layer's straight-line share needs a
    /// layer at the bottom, one below it a layer higher up). Returns the
    /// curve and its `c`.
    ///
    /// ```
    /// use prospicio_pricing::exposure::{ExposureCurve, Mbbefd};
    ///
    /// let share = Mbbefd::swiss_re(4.0).unwrap().layer_share(5e6, 5e6, 10e6).unwrap();
    /// let (_, c) = Mbbefd::swiss_re_for_layer_share(5e6, 5e6, 10e6, share).unwrap();
    /// assert!((c - 4.0).abs() < 1e-8);
    /// ```
    ///
    /// # Errors
    ///
    /// An invalid layer, or a share outside what `c` in `[0, 10]` reaches.
    pub fn swiss_re_for_layer_share(
        limit: f64,
        attachment: f64,
        mpl: f64,
        share: f64,
    ) -> Result<(Self, f64)> {
        Self::swiss_re(1.0)?.layer_share(limit, attachment, mpl)?;
        Self::fit_swiss_re("layer share", share, |m| {
            m.layer_share(limit, attachment, mpl).unwrap_or(f64::NAN)
        })
    }

    /// Bisection on `c` in `[0, 10]` for `f(swiss_re(c)) = target`, with
    /// `f` monotone in `c`.
    fn fit_swiss_re(what: &str, target: f64, f: impl Fn(&Self) -> f64) -> Result<(Self, f64)> {
        let at = |c: f64| -> f64 { f(&Self::swiss_re(c).expect("c in [0, 10]")) - target };
        let (mut lo, mut hi) = (0.0, 10.0);
        let (flo, fhi) = (at(lo), at(hi));
        if !(target.is_finite() && flo.is_finite() && fhi.is_finite()) || flo * fhi > 0.0 {
            return Err(Error::Data(format!(
                "a {what} of {target} is outside what the Swiss Re curves reach (c from 0 to \
                 10 gives {} to {})",
                flo + target,
                fhi + target
            )));
        }
        if flo == 0.0 {
            return Ok((Self::swiss_re(0.0)?, 0.0));
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if at(mid) * flo > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < 1e-13 {
                break;
            }
        }
        let c = 0.5 * (lo + hi);
        Ok((Self::swiss_re(c)?, c))
    }

    pub fn b(&self) -> f64 {
        self.b
    }

    pub fn g_parameter(&self) -> f64 {
        self.g
    }

    /// Probability of a total loss, `1/g`.
    pub fn total_loss_probability(&self) -> f64 {
        1.0 / self.g
    }

    fn case(&self) -> Case {
        let (b, g) = (self.b, self.g);
        if g == 1.0 || b == 0.0 {
            Case::Linear
        } else if (b - 1.0).abs() < SPECIAL {
            Case::BOne
        } else if (b * g - 1.0).abs() < SPECIAL {
            Case::BgOne
        } else {
            Case::General
        }
    }

    /// The distribution function of the destruction rate: `F(x)` for
    /// `x < 1` from the table above, 1 from `x = 1`, 0 below 0.
    pub fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        let (b, g) = (self.b, self.g);
        match self.case() {
            Case::Linear => 0.0,
            Case::BOne => 1.0 - 1.0 / (1.0 + (g - 1.0) * x),
            Case::BgOne => 1.0 - b.powf(x),
            Case::General => 1.0 - (1.0 - b) / ((g - 1.0) * b.powf(1.0 - x) + 1.0 - g * b),
        }
    }

    /// Mean destruction rate `E[X] = 1/G'(0)`.
    pub fn mean(&self) -> f64 {
        let (b, g) = (self.b, self.g);
        let slope = match self.case() {
            Case::Linear => 1.0,
            Case::BOne => (g - 1.0) / g.ln(),
            Case::BgOne => -b.ln() / (1.0 - b),
            // G'(0) = (1 − gb) ln b / (((g − 1)b + 1 − gb) ln(gb)).
            Case::General => (1.0 - g * b) * b.ln() / ((1.0 - b) * (g * b).ln()),
        };
        1.0 / slope
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Case {
    Linear,
    BOne,
    BgOne,
    General,
}

impl ExposureCurve for Mbbefd {
    fn rate_quantile(&self, u: f64) -> f64 {
        let (b, g) = (self.b, self.g);
        if u >= 1.0 - 1.0 / g {
            return 1.0;
        }
        let x = match self.case() {
            Case::Linear => 1.0,
            Case::BOne => (1.0 / (1.0 - u) - 1.0) / (g - 1.0),
            Case::BgOne => (1.0 - u).ln() / b.ln(),
            Case::General => {
                1.0 - (((1.0 - b) / (1.0 - u) - 1.0 + g * b) / (g - 1.0)).ln() / b.ln()
            }
        };
        x.clamp(0.0, 1.0)
    }

    fn mean_rate(&self) -> f64 {
        self.mean()
    }

    fn g(&self, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        let (b, g) = (self.b, self.g);
        match self.case() {
            Case::Linear => x,
            Case::BOne => (1.0 + (g - 1.0) * x).ln() / g.ln(),
            Case::BgOne => (1.0 - b.powf(x)) / (1.0 - b),
            Case::General => {
                (((g - 1.0) * b + (1.0 - g * b) * b.powf(x)) / (1.0 - b)).ln() / (g * b).ln()
            }
        }
    }
}

/// The exposure curve of a severity capped at the maximum possible loss
/// `mpl`: `G(x) = LEV(x · mpl) / LEV(mpl)`.
///
/// ```
/// use prospicio_prob::Lognormal;
/// use prospicio_pricing::exposure::{ExposureCurve, SeverityCurve};
///
/// let sev = Lognormal::from_mean_cv(2e5, 2.0).unwrap();
/// let curve = SeverityCurve::new(&sev, 1e6).unwrap();
/// assert!((curve.g(1.0) - 1.0).abs() < 1e-15);
/// assert!(curve.g(0.5) > 0.5);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SeverityCurve<'a, S: Severity + ?Sized> {
    severity: &'a S,
    mpl: f64,
    lev_mpl: f64,
}

impl<'a, S: Severity + ?Sized> SeverityCurve<'a, S> {
    /// The curve of `severity` capped at `mpl > 0`.
    pub fn new(severity: &'a S, mpl: f64) -> Result<Self> {
        if !(mpl.is_finite() && mpl > 0.0) {
            return Err(invalid("mpl", mpl, "must be positive and finite"));
        }
        let lev_mpl = severity.lev(mpl);
        if lev_mpl.is_nan() || lev_mpl <= 0.0 {
            return Err(invalid("mpl", mpl, "gives no expected loss below it"));
        }
        Ok(Self {
            severity,
            mpl,
            lev_mpl,
        })
    }
}

impl<S: Severity + ?Sized> ExposureCurve for SeverityCurve<'_, S> {
    fn g(&self, x: f64) -> f64 {
        self.severity.lev(x.clamp(0.0, 1.0) * self.mpl) / self.lev_mpl
    }

    /// `min(q(u), mpl) / mpl` for the severity's quantile `q`.
    fn rate_quantile(&self, u: f64) -> f64 {
        let q = self.severity.quantile(u).unwrap_or(f64::NAN);
        (q / self.mpl).min(1.0)
    }

    fn mean_rate(&self) -> f64 {
        self.lev_mpl / self.mpl
    }
}

/// How a [`TabulatedCurve`] runs between its points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Straight lines between the points (the default).
    #[default]
    Linear,
    /// Straight lines on log-log axes: `G(x) = Gₖ (x / xₖ)^pₖ` between
    /// `xₖ` and `xₖ₊₁`, with `pₖ = ln(Gₖ₊₁ / Gₖ) / ln(xₖ₊₁ / xₖ)`, the
    /// curve's elasticity on the piece. A power law follows a curve's bend
    /// between sparse points better than a chord, the way published
    /// scales are read off log-scaled charts. The first piece, from
    /// `(0, 0)`, stays straight: `ln 0` is undefined, and a power below 1
    /// there would make `G'(0)` infinite and the mean rate 0.
    LogLog,
}

/// A tabulated exposure curve: points `(x, G(x))` from `(0, 0)` to
/// `(1, 1)`, interpolated linearly (or log-log, [`Interpolation`]), as
/// published curves are given (Salzmann's homeowners scale, Ludwig's, ISO
/// PSOLD tables, the reinsurers' own).
///
/// The table must be concave (slopes that never increase), as every
/// exposure curve is. Linear interpolation keeps it so, and makes the
/// destruction rate discrete: it takes the value `xₖ` with probability
/// `(sₖ − sₖ₊₁)/s₁` for the slopes `s`, and a total loss with probability
/// `sₙ/s₁`. Its mean rate is `x₁ / G(x₁)`, the first chord's: a curve is
/// steepest near 0, so a table needs fine first points for the expected
/// loss of a risk (MPL × mean rate) to be right.
///
/// Interpolated log-log ([`with_interpolation`](Self::with_interpolation)),
/// the curve is concave when the elasticities `pₖ` never increase and stay
/// at most 1 after the straight first piece, which the constructor checks
/// (a table concave in straight lines can fail it). Its destruction rate
/// is then continuous within each piece past the first: `F(x) = 1 −
/// G'(x)/G'(0)` rises as `x^(pₖ − 1)` there, with an atom `(pₖ₋₁ − pₖ)
/// Gₖ / (xₖ G'(0))` at each interior point `xₖ` and a total loss with the
/// last piece's elasticity over `G'(0)` as its probability. The mean rate
/// is the first chord's, as for linear.
///
/// ```
/// use prospicio_pricing::exposure::{ExposureCurve, TabulatedCurve};
///
/// let t = TabulatedCurve::new(&[0.0, 0.1, 0.5, 1.0], &[0.0, 0.4, 0.8, 1.0]).unwrap();
/// assert!((t.g(0.3) - 0.6).abs() < 1e-15);
/// // Slopes 4, 1, 0.4: a total loss with probability 0.4 / 4.
/// assert_eq!(t.rate_quantile(0.95), 1.0);
/// assert!((t.mean_rate() - 0.25).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TabulatedCurve {
    x: Vec<f64>,
    g: Vec<f64>,
    /// `F(xₖ)` at each point after the first: `1 − s_{k+1}/s₁` at interior
    /// points, 1 at `x = 1` (linear interpolation).
    cdf: Vec<f64>,
    interpolation: Interpolation,
    /// The elasticity `pₖ` of each piece, from point `k` to `k + 1`: 1 for
    /// the first, which is straight (log-log interpolation; empty for
    /// linear).
    powers: Vec<f64>,
}

impl TabulatedCurve {
    /// The curve through the points, interpolated linearly; `x` increases
    /// from 0 to 1 and `g` from 0 to 1 (both ends within 1e-9), concave.
    pub fn new(x: &[f64], g: &[f64]) -> Result<Self> {
        Self::with_interpolation(x, g, Interpolation::Linear)
    }

    /// The curve through the points, interpolated as `interpolation` says.
    /// Log-log also needs the elasticities `pₖ` (see [`Interpolation`]) to
    /// be at most 1 and never to increase, which makes the interpolated
    /// curve concave.
    ///
    /// ```
    /// use prospicio_pricing::exposure::{ExposureCurve, Interpolation, TabulatedCurve};
    ///
    /// let (x, g) = ([0.0, 0.1, 0.5, 1.0], [0.0, 0.4, 0.8, 1.0]);
    /// let t = TabulatedCurve::with_interpolation(&x, &g, Interpolation::LogLog).unwrap();
    /// // Between 0.1 and 0.5: 0.4 (x / 0.1)^p with p = ln 2 / ln 5.
    /// let p = 2f64.ln() / 5f64.ln();
    /// assert!((t.g(0.3) - 0.4 * 3f64.powf(p)).abs() < 1e-15);
    /// assert!(t.g(0.3) > 0.6); // above the chord: the power law bends
    /// assert!((t.mean_rate() - 0.25).abs() < 1e-15); // the first chord's
    /// ```
    pub fn with_interpolation(x: &[f64], g: &[f64], interpolation: Interpolation) -> Result<Self> {
        if x.len() != g.len() || x.len() < 2 {
            return Err(Error::Data(
                "a tabulated curve needs at least two points, as many x as g".into(),
            ));
        }
        let n = x.len();
        if x[0].abs() > 1e-9 || (x[n - 1] - 1.0).abs() > 1e-9 {
            return Err(invalid("x", x[0], "must run from 0 to 1"));
        }
        if g[0].abs() > 1e-9 || (g[n - 1] - 1.0).abs() > 1e-9 {
            return Err(invalid("g", g[0], "must run from G(0) = 0 to G(1) = 1"));
        }
        let mut slopes = Vec::with_capacity(n - 1);
        for k in 1..n {
            let dx = x[k] - x[k - 1];
            if dx.is_nan() || dx <= 0.0 {
                return Err(invalid("x", x[k], "must increase strictly"));
            }
            let s = (g[k] - g[k - 1]) / dx;
            if s.is_nan() || s < 0.0 {
                return Err(invalid("g", g[k], "must not decrease"));
            }
            if let Some(&prev) = slopes.last()
                && s > prev * (1.0 + 1e-9) + 1e-12
            {
                return Err(invalid(
                    "g",
                    g[k],
                    "must be concave: its slopes may not increase",
                ));
            }
            slopes.push(s);
        }
        let s1 = slopes[0];
        let cdf = slopes[1..]
            .iter()
            .map(|s| 1.0 - s / s1)
            .chain(std::iter::once(1.0))
            .collect();
        let mut x = x.to_vec();
        let mut g = g.to_vec();
        x[0] = 0.0;
        g[0] = 0.0;
        x[n - 1] = 1.0;
        g[n - 1] = 1.0;
        let mut powers = Vec::new();
        if interpolation == Interpolation::LogLog {
            powers.push(1.0);
            for k in 2..n {
                let p = (g[k] / g[k - 1]).ln() / (x[k] / x[k - 1]).ln();
                let prev = powers[k - 2];
                if p.is_nan() || p > prev * (1.0 + 1e-9) + 1e-12 {
                    // Name the point where the elasticity rises: past it the
                    // power law bends convex.
                    return Err(invalid(
                        "x",
                        x[k - 1],
                        "is where the table's elasticity ln(G_k+1 / G_k) / ln(x_k+1 / x_k) \
                         rises (or exceeds 1 after the first piece), so log-log interpolation \
                         is not concave there; interpolate linearly",
                    ));
                }
                powers.push(p.min(prev));
            }
        }
        Ok(Self {
            x,
            g,
            cdf,
            interpolation,
            powers,
        })
    }

    /// The curve of `severity` capped at `mpl` ([`SeverityCurve`]),
    /// tabulated at `points` points chosen so linear interpolation stays
    /// close to it: half evenly spaced in `G` (the curve's own scale) and
    /// half geometric in `x` from `1e-6`, where the curve is steepest and
    /// its first chord sets the mean rate. A coarse hand-made grid can
    /// misprice layers; 200 points prices layers to about `1e-4`.
    ///
    /// ```
    /// use prospicio_pricing::exposure::{ExposureCurve, SeverityCurve, TabulatedCurve};
    /// use prospicio_prob::Lognormal;
    ///
    /// let sev = Lognormal::from_mean_cv(2e5, 2.0).unwrap();
    /// let exact = SeverityCurve::new(&sev, 1e6).unwrap();
    /// let table = TabulatedCurve::from_severity(&sev, 1e6, 200).unwrap();
    /// let (e, t) = (exact.layer_share(5e5, 5e5, 1e6).unwrap(), table.layer_share(5e5, 5e5, 1e6).unwrap());
    /// assert!((t / e - 1.0).abs() < 1e-3);
    /// ```
    ///
    /// # Errors
    ///
    /// Fewer than 4 points, or what [`SeverityCurve::new`] refuses.
    pub fn from_severity<S: Severity + ?Sized>(
        severity: &S,
        mpl: f64,
        points: usize,
    ) -> Result<Self> {
        if points < 4 {
            return Err(Error::Data(format!(
                "a tabulated curve from a severity needs at least 4 points, not {points}"
            )));
        }
        let curve = SeverityCurve::new(severity, mpl)?;
        let half = points / 2;
        let mut xs: Vec<f64> = (0..half)
            .map(|j| 10f64.powf(-6.0 + 6.0 * j as f64 / half as f64))
            .collect();
        // x where G(x) = k / n, by bisection: G rises from 0 to 1.
        let n = points - half;
        for k in 1..n {
            let target = k as f64 / n as f64;
            let (mut lo, mut hi) = (0.0, 1.0);
            for _ in 0..100 {
                let mid = 0.5 * (lo + hi);
                if curve.g(mid) < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            xs.push(0.5 * (lo + hi));
        }
        xs.push(0.0);
        xs.push(1.0);
        xs.sort_by(f64::total_cmp);
        xs.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);
        let gs: Vec<f64> = xs.iter().map(|&x| curve.g(x)).collect();
        Self::new(&xs, &gs)
    }

    /// The points' `x`.
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// The points' `G(x)`.
    pub fn g_values(&self) -> &[f64] {
        &self.g
    }

    /// How the curve runs between its points.
    pub fn interpolation(&self) -> Interpolation {
        self.interpolation
    }

    /// `G'(x)` on a log-log piece past the first, for tests.
    #[cfg(test)]
    fn g_prime_for_test(&self, x: f64) -> f64 {
        let k = self
            .x
            .partition_point(|&p| p < x)
            .clamp(2, self.x.len() - 1);
        let p = self.powers[k - 1];
        p * self.g(x) / x
    }

    /// The log-log destruction rate at `u`: the smallest `x` with
    /// `F(x) = 1 − G'(x)/s₁ ≥ u`.
    fn log_log_rate_quantile(&self, u: f64) -> f64 {
        let (x, g, p) = (&self.x, &self.g, &self.powers);
        let n = x.len();
        let s1 = g[1] / x[1];
        // F just after point k (1 ≤ k < n − 1): 1 − p_k G_k / (x_k s₁),
        // with p_k the power of the piece from k to k + 1.
        let after = |k: usize| 1.0 - p[k] * g[k] / (x[k] * s1);
        for k in 1..n - 1 {
            if u <= after(k) {
                return x[k];
            }
            // On the piece from k to k + 1, F rises to its value just
            // before point k + 1.
            let before_next = 1.0 - p[k] * g[k + 1] / (x[k + 1] * s1);
            if u <= before_next && p[k] < 1.0 {
                // p_k G_k (x/x_k)^(p_k − 1) / x_k = (1 − u) s₁.
                let ratio = (1.0 - u) * s1 * x[k] / (p[k] * g[k]);
                return (x[k] * ratio.powf(1.0 / (p[k] - 1.0))).clamp(x[k], x[k + 1]);
            }
        }
        1.0
    }
}

impl ExposureCurve for TabulatedCurve {
    fn g(&self, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        let k = self
            .x
            .partition_point(|&p| p < x)
            .clamp(1, self.x.len() - 1);
        let (x0, x1, g0, g1) = (self.x[k - 1], self.x[k], self.g[k - 1], self.g[k]);
        if self.interpolation == Interpolation::LogLog && k > 1 {
            return (g0 * (x / x0).powf(self.powers[k - 1])).min(g1);
        }
        g0 + (g1 - g0) * (x - x0) / (x1 - x0)
    }

    /// Linear: the smallest point `xₖ` (k ≥ 1) with `F(xₖ) ≥ u`. Log-log:
    /// the smallest `x` with `F(x) ≥ u`, a point or within a piece.
    fn rate_quantile(&self, u: f64) -> f64 {
        if self.interpolation == Interpolation::LogLog {
            return self.log_log_rate_quantile(u);
        }
        let k = self.cdf.partition_point(|&f| f < u);
        self.x[(k + 1).min(self.x.len() - 1)]
    }

    fn mean_rate(&self) -> f64 {
        (self.x[1] - self.x[0]) / (self.g[1] - self.g[0])
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

    /// `∫₀ˣ (1 − F)` by composite Simpson on 20,000 panels.
    fn integral_of_survival(m: &Mbbefd, x: f64) -> f64 {
        let n = 20_000;
        let h = x / n as f64;
        let s = |t: f64| 1.0 - m.cdf(t);
        let mut total = s(0.0) + s(x - 1e-15);
        for i in 1..n {
            total += if i % 2 == 1 { 4.0 } else { 2.0 } * s(i as f64 * h);
        }
        total * h / 3.0
    }

    #[test]
    fn curve_is_the_normalized_integral_of_the_survival() {
        // One of each case, and the Swiss Re curves.
        let mut curves = vec![
            Mbbefd::new(0.5, 4.0).unwrap(),
            Mbbefd::new(3.0, 0.9f64.recip()).unwrap(),
            Mbbefd::new(1.0, 7.0).unwrap(),
            Mbbefd::new(0.25, 4.0).unwrap(),
        ];
        curves.extend([1.5, 2.0, 3.0, 4.0, 5.0].map(|c| Mbbefd::swiss_re(c).unwrap()));
        for m in curves {
            let mean = integral_of_survival(&m, 1.0);
            assert!(
                (m.mean() / mean - 1.0).abs() < 1e-8,
                "{m:?}: {} vs {mean}",
                m.mean()
            );
            for x in [0.05, 0.2, 0.5, 0.9] {
                let want = integral_of_survival(&m, x) / mean;
                assert!(
                    (m.g(x) - want).abs() < 1e-8,
                    "{m:?} at {x}: {} vs {want}",
                    m.g(x)
                );
            }
            assert!((m.g(1.0) - 1.0).abs() < 1e-12);
            assert!((1.0 - m.cdf(1.0 - 1e-12) - m.total_loss_probability()).abs() < 1e-9);
        }
    }

    #[test]
    fn special_cases_meet_the_general_formula() {
        // Approaching b = 1 and bg = 1 from the general case.
        for (b, g) in [(1.0 + 1e-6, 5.0), (0.25 * (1.0 + 1e-6), 4.0)] {
            let near = Mbbefd::new(b, g).unwrap();
            let exact = if (b - 1.0).abs() < 1e-3 {
                Mbbefd::new(1.0, g).unwrap()
            } else {
                Mbbefd::new(0.25, 4.0).unwrap()
            };
            for x in [0.1, 0.5, 0.9] {
                assert!((near.g(x) - exact.g(x)).abs() < 1e-5, "{b} {g} {x}");
            }
        }
        let line = Mbbefd::swiss_re(0.0).unwrap();
        assert_eq!(line.g(0.3), 0.3);
        assert_eq!(line.total_loss_probability(), 1.0);
    }

    /// Draws' mean of `min(D, x)` over the curve's `mean_rate` must be
    /// `G(x)`: the sampled destruction rate has this exposure curve.
    fn check_sampling(c: &dyn ExposureCurve) {
        let n = 200_000;
        let draws: Vec<f64> = (0..n)
            .map(|i| c.rate_quantile((i as f64 + 0.5) / n as f64))
            .collect();
        let mean: f64 = draws.iter().sum::<f64>() / n as f64;
        assert!(
            (mean / c.mean_rate() - 1.0).abs() < 1e-3,
            "{mean} vs {}",
            c.mean_rate()
        );
        for x in [0.05, 0.3, 0.7] {
            let lev = draws.iter().map(|d| d.min(x)).sum::<f64>() / n as f64;
            assert!((lev / c.mean_rate() - c.g(x)).abs() < 2e-3, "at {x}");
        }
    }

    #[test]
    fn every_curve_is_a_destruction_rate_distribution() {
        for c in [0.0, 1.5, 3.0, 5.0] {
            check_sampling(&Mbbefd::swiss_re(c).unwrap());
        }
        check_sampling(&Mbbefd::new(1.0, 7.0).unwrap());
        check_sampling(&Mbbefd::new(0.25, 4.0).unwrap());
        let sev = prospicio_prob::Lognormal::from_mean_cv(2e5, 2.0).unwrap();
        check_sampling(&SeverityCurve::new(&sev, 1e6).unwrap());
        let t = TabulatedCurve::new(&[0.0, 0.1, 0.5, 1.0], &[0.0, 0.4, 0.8, 1.0]).unwrap();
        check_sampling(&t);
        // P(D = 0.1) = (4 − 1)/4, P(D = 0.5) = (1 − 0.4)/4, P(D = 1) = 0.4/4.
        assert_eq!(t.rate_quantile(0.75), 0.1);
        assert_eq!(t.rate_quantile(0.76), 0.5);
        assert_eq!(t.rate_quantile(0.9), 0.5);
        assert_eq!(t.rate_quantile(0.91), 1.0);
    }

    #[test]
    fn log_log_tables_are_destruction_rate_distributions() {
        let (x, g) = ([0.0, 0.1, 0.5, 1.0], [0.0, 0.4, 0.8, 1.0]);
        let t = TabulatedCurve::with_interpolation(&x, &g, Interpolation::LogLog).unwrap();
        check_sampling(&t);
        // Elasticities 1, ln 2 / ln 5, ln 1.25 / ln 2; s₁ = 4.
        let (p1, p2) = (2f64.ln() / 5f64.ln(), 1.25f64.ln() / 2f64.ln());
        // F jumps at 0.1 from 0 to 1 − p₁ 0.4 / (0.1 · 4).
        let f1 = 1.0 - p1;
        assert_eq!(t.rate_quantile(f1 * 0.999), 0.1);
        let mid = t.rate_quantile(f1 + 0.01);
        assert!(mid > 0.1 && mid < 0.5);
        assert!((1.0 - t.g_prime_for_test(mid) / 4.0 - (f1 + 0.01)).abs() < 1e-12);
        // A total loss with probability p₂ / 4.
        assert_eq!(t.rate_quantile(1.0 - p2 / 4.0 * 0.999), 1.0);
        assert!(t.rate_quantile(1.0 - p2 / 4.0 * 1.001) < 1.0);
        for (k, (&xk, &gk)) in x.iter().zip(&g).enumerate() {
            assert!((t.g(xk) - gk).abs() < 1e-15, "point {k}");
        }
        // A piece boundary: u exactly at F just before 0.5 is 0.5.
        let before = 1.0 - p1 * 0.8 / (0.5 * 4.0);
        assert!((t.rate_quantile(before) - 0.5).abs() < 1e-12);
        assert_eq!(t.rate_quantile(before + 1e-9), 0.5);
        // A flat top: elasticity 0, no mass above 0.5.
        let (x, g) = ([0.0, 0.2, 0.5, 1.0], [0.0, 0.6, 1.0, 1.0]);
        let t = TabulatedCurve::with_interpolation(&x, &g, Interpolation::LogLog).unwrap();
        check_sampling(&t);
        assert_eq!(t.rate_quantile(0.999), 0.5);
    }

    #[test]
    fn log_log_reads_a_sparse_table_closer_to_the_curve() {
        // Ten points of a curve whose elasticity falls throughout: log-log
        // interpolation follows it closer than straight lines.
        let x: Vec<f64> = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0].to_vec();
        let ln = prospicio_prob::Lognormal::from_mean_cv(2e5, 2.0).unwrap();
        let pa = prospicio_prob::Pareto::new(1e4, 1.5).unwrap();
        let curves: Vec<Box<dyn ExposureCurve>> = vec![
            Box::new(Mbbefd::swiss_re(4.0).unwrap()),
            Box::new(Mbbefd::swiss_re(5.0).unwrap()),
            Box::new(SeverityCurve::new(&ln, 1e6).unwrap()),
            Box::new(SeverityCurve::new(&pa, 1e6).unwrap()),
        ];
        for (i, m) in curves.iter().enumerate() {
            let g: Vec<f64> = x.iter().map(|&v| m.g(v)).collect();
            let lin = TabulatedCurve::new(&x, &g).unwrap();
            let log = TabulatedCurve::with_interpolation(&x, &g, Interpolation::LogLog).unwrap();
            let (mut e_lin, mut e_log) = (0.0f64, 0.0f64);
            // From the first point on: the first piece is straight in both.
            for v in (10..1000).map(|i| i as f64 / 1000.0) {
                e_lin = e_lin.max((lin.g(v) - m.g(v)).abs());
                e_log = e_log.max((log.g(v) - m.g(v)).abs());
            }
            // Measured: 0.36 to 0.67 of the linear error.
            assert!(
                e_log < 0.75 * e_lin,
                "curve {i}: log-log {e_log}, linear {e_lin}"
            );
        }
    }

    #[test]
    fn log_log_refuses_a_table_whose_elasticity_rises() {
        // Swiss Re c = 3: the total-loss atom makes the elasticity rise over
        // the top half, so no power law through the points is concave.
        let x = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0];
        let m = Mbbefd::swiss_re(3.0).unwrap();
        let g: Vec<f64> = x.iter().map(|&v| m.g(v)).collect();
        assert!(TabulatedCurve::new(&x, &g).is_ok());
        let err = TabulatedCurve::with_interpolation(&x, &g, Interpolation::LogLog).unwrap_err();
        assert!(matches!(err, Error::InvalidParameter { name: "x", value, .. } if value == 0.7));
    }

    #[test]
    fn tabulated_curves_are_checked() {
        let t = TabulatedCurve::new(&[0.0, 0.5, 1.0], &[0.0, 0.7, 1.0]).unwrap();
        assert!((t.g(0.25) - 0.35).abs() < 1e-15);
        assert_eq!(t.g(2.0), 1.0);
        assert!(TabulatedCurve::new(&[0.0, 0.5, 1.0], &[0.0, 0.3, 1.0]).is_err()); // convex
        assert!(TabulatedCurve::new(&[0.0, 0.5, 1.0], &[0.0, 0.7, 0.9]).is_err()); // G(1) ≠ 1
        assert!(TabulatedCurve::new(&[0.1, 0.5, 1.0], &[0.0, 0.7, 1.0]).is_err());
        assert!(TabulatedCurve::new(&[0.0, 0.5, 0.5, 1.0], &[0.0, 0.7, 0.7, 1.0]).is_err());
        assert!(TabulatedCurve::new(&[0.0, 1.0], &[0.0]).is_err());
        // A Swiss Re curve tabulated finely matches the curve.
        let m = Mbbefd::swiss_re(3.0).unwrap();
        let xs: Vec<f64> = (0..=1000).map(|i| i as f64 / 1000.0).collect();
        let gs: Vec<f64> = xs.iter().map(|&x| m.g(x)).collect();
        let t = TabulatedCurve::new(&xs, &gs).unwrap();
        assert!((t.g(0.3333) - m.g(0.3333)).abs() < 1e-6);
        // The table's mean rate is its first chord's, 0.001 / G(0.001):
        // the curve is steep near 0, so a table needs fine first points.
        assert!((t.mean_rate() - 0.001 / m.g(0.001)).abs() < 1e-15);
        assert!(t.mean_rate() > m.mean_rate());
    }

    #[test]
    fn severity_curves_and_layer_shares() {
        let sev = prospicio_prob::Pareto::new(1e5, 1.5).unwrap();
        let curve = SeverityCurve::new(&sev, 1e7).unwrap();
        let shares: f64 = [(1e6, 0.0), (4e6, 1e6), (5e6, 5e6)]
            .iter()
            .map(|&(l, a)| curve.layer_share(l, a, 1e7).unwrap())
            .sum();
        assert!((shares - 1.0).abs() < 1e-12);
        // A layer above the MPL takes nothing.
        let m = Mbbefd::swiss_re(2.0).unwrap();
        assert_eq!(m.layer_share(1e6, 2e7, 1e7).unwrap(), 0.0);
        assert!(m.layer_share(0.0, 0.0, 1e7).is_err());
        assert!(m.layer_share(1.0, 0.0, 0.0).is_err());
        assert!(Mbbefd::new(-1.0, 2.0).is_err());
        assert!(Mbbefd::new(0.5, 0.5).is_err());
        assert!(Mbbefd::swiss_re(-1.0).is_err());
    }
}
