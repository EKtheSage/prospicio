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
//!   an MPL, `LEV(x M) / LEV(M)`.
//!
//! Bernegger, S. (1997). The Swiss Re exposure curves and the MBBEFD
//! distribution class. ASTIN Bulletin 27(1), 99–111.

use act_core::{Error, Result};
use act_prob::Severity;

/// An exposure curve: `G(x)` on `[0, 1]`, the expected loss below a
/// fraction `x` of the MPL over the expected loss.
pub trait ExposureCurve {
    /// `G(x)`; `x` is clamped to `[0, 1]`.
    fn g(&self, x: f64) -> f64;

    /// Share of a risk's expected loss in the layer `limit` xs
    /// `attachment`, for a risk with maximum possible loss `mpl`.
    ///
    /// ```
    /// use act_pricing::exposure::{ExposureCurve, Mbbefd};
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
/// use act_pricing::exposure::{ExposureCurve, Mbbefd};
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
/// use act_prob::Lognormal;
/// use act_pricing::exposure::{ExposureCurve, SeverityCurve};
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

    #[test]
    fn severity_curves_and_layer_shares() {
        let sev = act_prob::Pareto::new(1e5, 1.5).unwrap();
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
