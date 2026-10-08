//! Distortion risk measures: `ρ(X) = ∫ g(S(x)) dx` for a concave
//! distortion `g` of the survival function, with `g(0) = 0`, `g(1) = 1`.
//!
//! On a discrete distribution the integral is a weighted sum of its
//! values: the value `x_k` gets weight `g(P(X >= x_k)) - g(P(X > x_k))`.
//! The weights depend only on ranks, which is what lets capital
//! allocation reuse them on a joint distribution (see
//! `docs/design/risk.md`).

use prospicio_core::{Error, Result};
use prospicio_math::roots::{bisect, bisect_log};
use prospicio_math::special::{beta_inc, norm_cdf, norm_quantile};

use crate::distribution::check_probability;

/// A distortion of the survival function. Every variant is concave, so
/// every measure here is coherent.
///
/// | Variant | `g(s)` for `0 < s < 1` | Parameters |
/// |---|---|---|
/// | `Tvar(p)` | `min(s / (1 - p), 1)` | `p` in `[0, 1]` |
/// | `Wang(λ)` | `Φ(Φ⁻¹(s) + λ)` | `λ >= 0` |
/// | `ProportionalHazard(ρ)` | `s^ρ` | `ρ` in `(0, 1]` |
/// | `DualPower(β)` | `1 - (1 - s)^β` | `β >= 1` |
/// | `Exponential(k)` | `(1 - e^(-k s)) / (1 - e^(-k))` | `k > 0` |
/// | `Ccoc(r)` | `min(1, δ + ν s)`, `δ = r / (1 + r)`, `ν = 1 - δ` | `r >= 0` |
/// | `BiTvar { p0, p1, w }` | `(1 - w) TVaR_p0 + w TVaR_p1` | `0 <= p0 <= p1 <= 1`, `w` in `[0, 1]` |
/// | `WeightedTvar { ps, wts }` | `Σ wᵢ min(s / (1 - pᵢ), 1)` | `ps` ascending in `[0, 1]`, `wts >= 0` summing to 1 |
/// | `CappedLinear { r0, slope }` | `min(1, r0 + slope s)` | `r0` in `[0, 1)`, `slope >= 1 - r0` |
/// | `CappedLogLinear { r0, b }` | `min(1, e^r0 s^b)` | `r0 >= 0`, `b` in `(0, 1]` |
/// | `Lep { r0, r }` | `min(1, d + (1 - d) s + (δ - d) √(s (1 - s)))`, `d = r0 / (1 + r0)`, `δ = r / (1 + r)` | `0 <= r0 <= r` |
/// | `LinearYield { r0, r }` | `(r0 + (1 + r) s) / (1 + r0 + r s)` | `r0, r >= 0` |
/// | `Beta { a, b }` | the Beta(a, b) distribution function | `0 < a <= 1 <= b` |
/// | `Mixture(parts)` | `Σ wᵢ gᵢ(s)` | weights `>= 0` summing to 1 |
/// | `Minimum(parts)` | `min gᵢ(s)` | at least one part |
/// | `Convex(knots)` | piecewise linear through the knots | see [`convex`](Self::convex) |
///
/// Every `g` has `g(0) = 0` and `g(1) = 1`. `Ccoc`, `CappedLinear`, `Lep`
/// and `LinearYield` with `r0 > 0` jump at 0: they put a probability
/// [`mass`](Self::mass) on the largest outcome, which is how the constant
/// cost of capital prices the assets. The names and parameters follow
/// Mildenhall and Major (*Pricing Insurance Risk*, 2022) and the Python
/// `aggregate` package (`ccoc`, `bitvar`, `wtdtvar`, `clin`, `cll`, `lep`,
/// `ly`, `beta`, `mixture`, `minimum`); `ProportionalHazard` is `ph` and
/// `DualPower` is `dual`.
///
/// Each one-parameter family has a value that gives the mean (`Tvar(0)`,
/// `Wang(0)`, `ProportionalHazard(1)`, `DualPower(1)`, `Ccoc(0)`, and
/// `Exponential(k)` as `k → 0`); [`calibrate`] solves for the parameter
/// that gives a target price.
///
/// A concave distortion is a spectral risk measure,
/// `ρ = ∫₀¹ φ(u) VaR_u du` with the non-decreasing risk-aversion spectrum
/// `φ(u) = g'(1 - u)`. `Exponential(k)` is the spectral measure with
/// exponential risk aversion, `φ(u) = k e^(-k(1-u)) / (1 - e^(-k))`
/// (Acerbi, 2002; Dowd, Cotter and Sorwar, 2008); `Tvar(p)` is the one
/// whose spectrum is flat above `p`.
///
/// ```
/// use prospicio_prob::Distortion;
///
/// let x = [1.0, 2.0, 3.0, 4.0];
/// // TVaR at 50%: the mean of the top half.
/// assert_eq!(Distortion::tvar(0.5).unwrap().apply_sorted(&x), 3.5);
/// // Wang with λ = 0 is the mean.
/// assert!((Distortion::wang(0.0).unwrap().apply_sorted(&x) - 2.5).abs() < 1e-15);
/// // A constant 25% cost of capital: (mean + 0.25 max) / 1.25.
/// let ccoc = Distortion::ccoc(0.25).unwrap().apply_sorted(&x);
/// assert!((ccoc - (2.5 + 0.25 * 4.0) / 1.25).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Distortion {
    Tvar(f64),
    Wang(f64),
    ProportionalHazard(f64),
    DualPower(f64),
    Exponential(f64),
    Ccoc(f64),
    BiTvar { p0: f64, p1: f64, w: f64 },
    WeightedTvar { ps: Vec<f64>, wts: Vec<f64> },
    CappedLinear { r0: f64, slope: f64 },
    CappedLogLinear { r0: f64, b: f64 },
    Lep { r0: f64, r: f64 },
    LinearYield { r0: f64, r: f64 },
    Beta { a: f64, b: f64 },
    Mixture(Vec<(f64, Distortion)>),
    Minimum(Vec<Distortion>),
    Convex(Vec<(f64, f64)>),
}

impl Distortion {
    /// Tail value at risk at level `p`.
    pub fn tvar(p: f64) -> Result<Self> {
        check_probability(p)?;
        Ok(Self::Tvar(p))
    }

    /// Wang transform with market price of risk `lambda >= 0`.
    pub fn wang(lambda: f64) -> Result<Self> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(invalid("lambda", lambda, "must be finite and non-negative"));
        }
        Ok(Self::Wang(lambda))
    }

    /// Proportional hazard transform with `rho` in `(0, 1]`.
    pub fn proportional_hazard(rho: f64) -> Result<Self> {
        if !(rho > 0.0 && rho <= 1.0) {
            return Err(invalid("rho", rho, "must be in (0, 1]"));
        }
        Ok(Self::ProportionalHazard(rho))
    }

    /// Dual power transform with `beta >= 1`.
    pub fn dual_power(beta: f64) -> Result<Self> {
        if !beta.is_finite() || beta < 1.0 {
            return Err(invalid("beta", beta, "must be finite and at least 1"));
        }
        Ok(Self::DualPower(beta))
    }

    /// The spectral measure with exponential risk aversion `k > 0`: the
    /// larger `k`, the more weight on the worst outcomes.
    pub fn exponential(k: f64) -> Result<Self> {
        if !(k.is_finite() && k > 0.0) {
            return Err(invalid("k", k, "must be finite and positive"));
        }
        Ok(Self::Exponential(k))
    }

    /// A constant cost of capital `r >= 0`: the price of a loss `X` backed
    /// by assets `a = max X` is `P = ν E[X] + δ a`, so the margin
    /// `P - E[X]` is `r` times the capital `a - P`.
    pub fn ccoc(r: f64) -> Result<Self> {
        if !(r.is_finite() && r >= 0.0) {
            return Err(invalid("r", r, "must be finite and non-negative"));
        }
        Ok(Self::Ccoc(r))
    }

    /// A mix of two TVaRs: weight `1 - w` on `TVaR_p0` and `w` on
    /// `TVaR_p1`, with `p0 <= p1`. These are the extreme distortions
    /// among all those that give a price (Mildenhall and Major, 2022,
    /// chapter 11); `Ccoc(r)` is `BiTvar { p0: 0, p1: 1, w: r / (1 + r) }`.
    pub fn bitvar(p0: f64, p1: f64, w: f64) -> Result<Self> {
        check_probability(p0)?;
        check_probability(p1)?;
        check_probability(w)?;
        if p0 > p1 {
            return Err(invalid("p0", p0, "must not exceed p1"));
        }
        Ok(Self::BiTvar { p0, p1, w })
    }

    /// A weighted average of TVaRs at the ascending levels `ps`. The
    /// weights must be non-negative and sum to 1 (to within `1e-9`; they
    /// are then rescaled to sum exactly). Any concave distortion is a
    /// limit of these (Kusuoka's representation).
    pub fn weighted_tvar(ps: Vec<f64>, wts: Vec<f64>) -> Result<Self> {
        if ps.is_empty() || ps.len() != wts.len() {
            return Err(Error::Data(
                "weighted TVaR needs as many weights as levels, and at least one".into(),
            ));
        }
        for &p in &ps {
            check_probability(p)?;
        }
        if !ps.windows(2).all(|w| w[0] < w[1]) {
            return Err(Error::Data(
                "weighted TVaR levels must be strictly ascending".into(),
            ));
        }
        let wts = normalized(wts)?;
        Ok(Self::WeightedTvar { ps, wts })
    }

    /// Capped linear, `min(1, r0 + slope s)` for `s > 0`: a mass `r0` on the
    /// largest outcome and a constant loading below the cap.
    pub fn capped_linear(r0: f64, slope: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&r0) {
            return Err(invalid("r0", r0, "must be in [0, 1)"));
        }
        if !(slope.is_finite() && r0 + slope >= 1.0) {
            return Err(invalid(
                "slope",
                slope,
                "must be finite with r0 + slope >= 1",
            ));
        }
        Ok(Self::CappedLinear { r0, slope })
    }

    /// Capped log-linear, `min(1, e^r0 s^b)`: a proportional hazard
    /// `s^b` scaled up by `e^r0`.
    pub fn capped_log_linear(r0: f64, b: f64) -> Result<Self> {
        if !(r0.is_finite() && r0 >= 0.0) {
            return Err(invalid("r0", r0, "must be finite and non-negative"));
        }
        if !(b > 0.0 && b <= 1.0) {
            return Err(invalid("b", b, "must be in (0, 1]"));
        }
        Ok(Self::CappedLogLinear { r0, b })
    }

    /// Leverage-equivalent pricing with a minimum rate on line `r0` and a
    /// target return `r >= r0`.
    pub fn lep(r0: f64, r: f64) -> Result<Self> {
        if !(r0.is_finite() && r0 >= 0.0) {
            return Err(invalid("r0", r0, "must be finite and non-negative"));
        }
        if !(r.is_finite() && r >= r0) {
            return Err(invalid("r", r, "must be finite and at least r0"));
        }
        Ok(Self::Lep { r0, r })
    }

    /// Linear yield, `(r0 + (1 + r) s) / (1 + r0 + r s)`: the price of a
    /// layer with exceedance probability `s` when the yield is linear in
    /// `s`, with a mass `r0 / (1 + r0)` at 0.
    pub fn linear_yield(r0: f64, r: f64) -> Result<Self> {
        if !(r0.is_finite() && r0 >= 0.0) {
            return Err(invalid("r0", r0, "must be finite and non-negative"));
        }
        if !(r.is_finite() && r >= 0.0) {
            return Err(invalid("r", r, "must be finite and non-negative"));
        }
        Ok(Self::LinearYield { r0, r })
    }

    /// The Beta(a, b) distribution function; concave, and so a coherent
    /// distortion, when `a <= 1 <= b`.
    pub fn beta(a: f64, b: f64) -> Result<Self> {
        if !(a > 0.0 && a <= 1.0) {
            return Err(invalid("a", a, "must be in (0, 1]"));
        }
        if !(b.is_finite() && b >= 1.0) {
            return Err(invalid("b", b, "must be finite and at least 1"));
        }
        Ok(Self::Beta { a, b })
    }

    /// A weighted average of distortions. The weights must be
    /// non-negative and sum to 1 (to within `1e-9`).
    pub fn mixture(parts: Vec<(f64, Distortion)>) -> Result<Self> {
        if parts.is_empty() {
            return Err(Error::Data(
                "a mixture needs at least one distortion".into(),
            ));
        }
        let (wts, ds): (Vec<f64>, Vec<Distortion>) = parts.into_iter().unzip();
        let wts = normalized(wts)?;
        Ok(Self::Mixture(wts.into_iter().zip(ds).collect()))
    }

    /// The pointwise minimum of distortions: the least conservative of
    /// them at every probability level.
    pub fn minimum(parts: Vec<Distortion>) -> Result<Self> {
        if parts.is_empty() {
            return Err(Error::Data(
                "a minimum needs at least one distortion".into(),
            ));
        }
        Ok(Self::Minimum(parts))
    }

    /// The smallest concave distortion above the points `(s, g)`: the
    /// upper convex hull of the points with `(0, 0)` and `(1, 1)`. Each
    /// point is a layer's exceedance probability and its price per unit of
    /// limit, for example a cat bond's expected loss and its spread. The
    /// hull can jump at `s = 0` (a point `(0, g)` with `g > 0` is a mass).
    ///
    /// ```
    /// use prospicio_prob::Distortion;
    ///
    /// // (0.5, 0.55) lies below the chord from (0.1, 0.3) to (1, 1): dropped.
    /// let d = Distortion::convex(&[(0.1, 0.3), (0.5, 0.55)]).unwrap();
    /// assert_eq!(d, Distortion::Convex(vec![(0.0, 0.0), (0.1, 0.3), (1.0, 1.0)]));
    /// assert!((d.g(0.05) - 0.15).abs() < 1e-15);
    /// ```
    pub fn convex(points: &[(f64, f64)]) -> Result<Self> {
        let mut pts: Vec<(f64, f64)> = Vec::with_capacity(points.len() + 2);
        pts.push((0.0, 0.0));
        for &(s, g) in points {
            if !((0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&g)) {
                return Err(Error::Data(format!(
                    "convex distortion points must lie in the unit square, not ({s}, {g})"
                )));
            }
            if g < s {
                return Err(Error::Data(format!(
                    "a distortion point must have g >= s, not ({s}, {g})"
                )));
            }
            pts.push((s, g));
        }
        pts.push((1.0, 1.0));
        // At each s > 0 keep the largest g. Points at s = 0 stay: the hull
        // keeps the origin and the highest of them, a jump at 0.
        pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        pts.dedup_by(|b, a| {
            a.0 == b.0 && a.0 > 0.0 && {
                a.1 = a.1.max(b.1);
                true
            }
        });
        // Upper hull (monotone chain), left to right.
        let mut hull: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
        for p in pts {
            while hull.len() >= 2 {
                let (o, a) = (hull[hull.len() - 2], hull[hull.len() - 1]);
                let cross = (a.0 - o.0) * (p.1 - o.1) - (a.1 - o.1) * (p.0 - o.0);
                if cross >= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(p);
        }
        Ok(Self::Convex(hull))
    }

    /// The distortion `g(s)` of a survival probability `s` in `[0, 1]`.
    pub fn g(&self, s: f64) -> f64 {
        if s <= 0.0 {
            return 0.0;
        }
        if s >= 1.0 {
            return 1.0;
        }
        match self {
            Self::Tvar(p) => tvar_g(*p, s),
            Self::Wang(lambda) => norm_cdf(norm_quantile(s) + lambda),
            Self::ProportionalHazard(rho) => s.powf(*rho),
            Self::DualPower(beta) => -((-s).ln_1p() * beta).exp_m1(),
            Self::Exponential(k) => (-k * s).exp_m1() / (-k).exp_m1(),
            Self::Ccoc(r) => ((r + s) / (1.0 + r)).min(1.0),
            Self::BiTvar { p0, p1, w } => (1.0 - w) * tvar_g(*p0, s) + w * tvar_g(*p1, s),
            Self::WeightedTvar { ps, wts } => ps
                .iter()
                .zip(wts)
                .map(|(&p, w)| w * tvar_g(p, s))
                .sum::<f64>()
                .min(1.0),
            Self::CappedLinear { r0, slope } => (r0 + slope * s).min(1.0),
            Self::CappedLogLinear { r0, b } => (r0.exp() * s.powf(*b)).min(1.0),
            Self::Lep { r0, r } => {
                let d = r0 / (1.0 + r0);
                let delta = r / (1.0 + r);
                (d + (1.0 - d) * s + (delta - d) * (s * (1.0 - s)).sqrt()).min(1.0)
            }
            Self::LinearYield { r0, r } => (r0 + (1.0 + r) * s) / (1.0 + r0 + r * s),
            Self::Beta { a, b } => beta_inc(*a, *b, s),
            Self::Mixture(parts) => parts.iter().map(|(w, d)| w * d.g(s)).sum::<f64>().min(1.0),
            Self::Minimum(parts) => parts.iter().map(|d| d.g(s)).fold(1.0, f64::min),
            Self::Convex(knots) => {
                // The first knot past s; s is in (0, 1), so one exists.
                let j = knots.partition_point(|k| k.0 < s);
                let (lo, hi) = (knots[j - 1], knots[j]);
                if hi.0 == s {
                    hi.1
                } else {
                    lo.1 + (hi.1 - lo.1) * (s - lo.0) / (hi.0 - lo.0)
                }
            }
        }
    }

    /// The probability mass the distortion puts on the largest outcome:
    /// `g(0+)`, the limit of `g(s)` as `s` falls to 0. Zero except for
    /// `Ccoc`, and for `CappedLinear`, `Lep` and `LinearYield` with
    /// `r0 > 0`, and their mixtures.
    pub fn mass(&self) -> f64 {
        match self {
            Self::Ccoc(r) => r / (1.0 + r),
            Self::CappedLinear { r0, .. } => *r0,
            Self::Lep { r0, .. } | Self::LinearYield { r0, .. } => r0 / (1.0 + r0),
            Self::BiTvar { p1, w, .. } if *p1 >= 1.0 => *w,
            Self::Tvar(p) if *p >= 1.0 => 1.0,
            Self::WeightedTvar { ps, wts } => ps
                .iter()
                .zip(wts)
                .filter(|(p, _)| **p >= 1.0)
                .map(|(_, w)| w)
                .sum(),
            Self::Mixture(parts) => parts.iter().map(|(w, d)| w * d.mass()).sum(),
            Self::Minimum(parts) => parts.iter().map(Self::mass).fold(1.0, f64::min),
            Self::Convex(knots) => knots.get(1).filter(|k| k.0 == 0.0).map_or(0.0, |k| k.1),
            _ => 0.0,
        }
    }

    /// The generalized inverse, the infimum of the `s` with `g(s) >= y`,
    /// for `y` in `[0, 1]`: 0 when `y` is at most the [`mass`](Self::mass),
    /// otherwise found by bisection to full precision.
    pub fn g_inv(&self, y: f64) -> f64 {
        if y <= self.mass() {
            return 0.0;
        }
        if y >= 1.0 {
            // The smallest s at which g reaches 1, which is below 1 for a
            // capped distortion.
            return bisect(0.0, 1.0, |s| self.g(s) < 1.0);
        }
        bisect(0.0, 1.0, |s| self.g(s) < y)
    }

    /// The dual distortion `1 - g(1 - s)`: it prices the loss from the
    /// other side, giving the bid where `g` gives the ask. It is convex,
    /// and its measure is at most the mean.
    pub fn g_dual(&self, s: f64) -> f64 {
        1.0 - self.g(1.0 - s)
    }

    /// The risk measure under the dual distortion: the bid price of a
    /// discrete distribution, `ρ*(X) = -ρ(-X)`. See [`apply_discrete`](Self::apply_discrete).
    pub fn apply_discrete_dual(&self, values: &[f64], probs: &[f64]) -> f64 {
        let neg: Vec<f64> = values.iter().rev().map(|x| -x).collect();
        let p: Vec<f64> = probs.iter().rev().copied().collect();
        -self.apply_discrete(&neg, &p)
    }

    /// Weights for `n` equally likely values sorted ascending: value `i`
    /// (0-based) gets `g((n - i) / n) - g((n - i - 1) / n)`. They are
    /// non-negative and sum to 1.
    ///
    /// ```
    /// use prospicio_prob::Distortion;
    ///
    /// // TVaR at 50% on four values: the top two, equally.
    /// assert_eq!(Distortion::tvar(0.5).unwrap().weights(4), [0.0, 0.0, 0.5, 0.5]);
    /// ```
    pub fn weights(&self, n: usize) -> Vec<f64> {
        let nf = n as f64;
        (0..n)
            .map(|i| self.g((n - i) as f64 / nf) - self.g((n - i - 1) as f64 / nf))
            .collect()
    }

    /// The risk measure of equally likely draws sorted ascending.
    ///
    /// For `Tvar(p)` this agrees with [`tvar_sorted`](crate::risk::tvar_sorted)
    /// up to rounding, including the fractional weight at the VaR. `sorted`
    /// must be non-empty and sorted ascending; this is checked only in
    /// debug builds.
    pub fn apply_sorted(&self, sorted: &[f64]) -> f64 {
        debug_assert!(!sorted.is_empty(), "draws must not be empty");
        debug_assert!(
            sorted.windows(2).all(|w| w[0] <= w[1]),
            "draws must be sorted ascending"
        );
        self.weights(sorted.len())
            .iter()
            .zip(sorted)
            .map(|(w, x)| w * x)
            .sum()
    }

    /// The risk measure of a discrete distribution with ascending `values`
    /// and their `probs` (which should sum to 1).
    ///
    /// Survival probabilities are summed from the top, so small tail
    /// probabilities keep their precision.
    ///
    /// ```
    /// use prospicio_prob::Distortion;
    ///
    /// let ph = Distortion::proportional_hazard(0.5).unwrap();
    /// // P(X = 0) = 0.75, P(X = 1) = 0.25: ρ = g(0.25) = 0.5.
    /// assert_eq!(ph.apply_discrete(&[0.0, 1.0], &[0.75, 0.25]), 0.5);
    /// ```
    pub fn apply_discrete(&self, values: &[f64], probs: &[f64]) -> f64 {
        debug_assert_eq!(values.len(), probs.len());
        debug_assert!(
            values.windows(2).all(|w| w[0] <= w[1]),
            "values must be sorted ascending"
        );
        let mut above = 0.0; // P(X > x_k)
        let mut g_above = 0.0;
        let mut total = 0.0;
        for (x, p) in values.iter().zip(probs).rev() {
            let at_or_above = above + p;
            let g_at_or_above = self.g(at_or_above);
            total += (g_at_or_above - g_above) * x;
            above = at_or_above;
            g_above = g_at_or_above;
        }
        total
    }
}

/// A one-parameter family of distortions, for [`calibrate`]. The standard
/// set of Mildenhall and Major is [`Family::STANDARD`]: CCoC, PH, Wang,
/// dual and TVaR, from the most tail-heavy to the most body-heavy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Family {
    /// [`Distortion::Ccoc`]; the parameter is the return `r`.
    Ccoc,
    /// [`Distortion::ProportionalHazard`]; the parameter is `ρ`.
    ProportionalHazard,
    /// [`Distortion::Wang`]; the parameter is `λ`.
    Wang,
    /// [`Distortion::DualPower`]; the parameter is `β`.
    DualPower,
    /// [`Distortion::Tvar`]; the parameter is `p`.
    Tvar,
    /// [`Distortion::Exponential`]; the parameter is `k`.
    Exponential,
    /// [`Distortion::CappedLinear`] with this `r0`; the parameter is the slope.
    CappedLinear { r0: f64 },
    /// [`Distortion::CappedLogLinear`] with this `r0`; the parameter is `b`.
    CappedLogLinear { r0: f64 },
    /// [`Distortion::Lep`] with this `r0`; the parameter is `r`.
    Lep { r0: f64 },
    /// [`Distortion::LinearYield`] with this `r0`; the parameter is `r`.
    LinearYield { r0: f64 },
}

impl Family {
    /// CCoC, PH, Wang, dual and TVaR.
    pub const STANDARD: [Family; 5] = [
        Family::Ccoc,
        Family::ProportionalHazard,
        Family::Wang,
        Family::DualPower,
        Family::Tvar,
    ];

    /// The family's member with this parameter.
    pub fn with(self, param: f64) -> Result<Distortion> {
        match self {
            Self::Ccoc => Distortion::ccoc(param),
            Self::ProportionalHazard => Distortion::proportional_hazard(param),
            Self::Wang => Distortion::wang(param),
            Self::DualPower => Distortion::dual_power(param),
            Self::Tvar => Distortion::tvar(param),
            Self::Exponential => Distortion::exponential(param),
            Self::CappedLinear { r0 } => Distortion::capped_linear(r0, param),
            Self::CappedLogLinear { r0 } => Distortion::capped_log_linear(r0, param),
            Self::Lep { r0 } => Distortion::lep(r0, param),
            Self::LinearYield { r0 } => Distortion::linear_yield(r0, param),
        }
    }

    /// The short name used by Mildenhall and Major and the Python
    /// `aggregate` package: `ccoc`, `ph`, `wang`, `dual`, `tvar`, `exp`,
    /// `clin`, `cll`, `lep`, `ly`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Ccoc => "ccoc",
            Self::ProportionalHazard => "ph",
            Self::Wang => "wang",
            Self::DualPower => "dual",
            Self::Tvar => "tvar",
            Self::Exponential => "exp",
            Self::CappedLinear { .. } => "clin",
            Self::CappedLogLinear { .. } => "cll",
            Self::Lep { .. } => "lep",
            Self::LinearYield { .. } => "ly",
        }
    }

    /// The parameter range, from the end that gives the smaller price to
    /// the end that gives the larger; an infinite end is searched by
    /// doubling.
    fn range(self) -> (f64, f64) {
        match self {
            Self::Ccoc | Self::Wang | Self::Exponential => (0.0, f64::INFINITY),
            Self::ProportionalHazard => (1.0, 0.0),
            Self::DualPower => (1.0, f64::INFINITY),
            Self::Tvar => (0.0, 1.0),
            Self::CappedLinear { r0 } => (1.0 - r0, f64::INFINITY),
            Self::CappedLogLinear { .. } => (1.0, 0.0),
            Self::Lep { r0 } => (r0, f64::INFINITY),
            Self::LinearYield { .. } => (0.0, f64::INFINITY),
        }
    }
}

/// The member of a family whose risk measure of the discrete distribution
/// (`values` ascending, `probs` summing to 1) equals `premium`.
///
/// To price a loss backed by assets `a`, pass the capped values
/// `min(x, a)`: the premium then lies between `E[X ∧ a]` and `a`. Every
/// family's price rises with its parameter (falls, for PH and capped
/// log-linear), so the solution is unique and found by bisection to full
/// precision. `Ccoc` is solved in closed form, `r = (P - E[X]) / (max X - P)`.
/// Fails unless the premium lies strictly between the family's smallest
/// and largest prices (the mean, and the maximum or the family's limit).
///
/// ```
/// use prospicio_prob::distortion::{calibrate, Family};
///
/// // Mildenhall and Major's InsCo: ten equally likely totals, with
/// // assets of 100 (the largest total), priced at a 15% cost of capital.
/// let x = [22.0, 28.0, 36.0, 40.0, 55.0, 65.0, 100.0];
/// let p = [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1];
/// let premium = (46.6 + 0.15 * 100.0) / 1.15;
/// let ccoc = calibrate(Family::Ccoc, &x, &p, premium).unwrap();
/// assert!(matches!(ccoc, prospicio_prob::Distortion::Ccoc(r) if (r - 0.15).abs() < 1e-12));
/// // The proportional hazard with the same price.
/// let ph = calibrate(Family::ProportionalHazard, &x, &p, premium).unwrap();
/// assert!((ph.apply_discrete(&x, &p) - premium).abs() < 1e-9);
/// ```
pub fn calibrate(
    family: Family,
    values: &[f64],
    probs: &[f64],
    premium: f64,
) -> Result<Distortion> {
    if values.is_empty() || values.len() != probs.len() {
        return Err(Error::Data(
            "calibration needs one probability per value".into(),
        ));
    }
    if !values.windows(2).all(|w| w[0] <= w[1]) {
        return Err(Error::Data("calibration values must be ascending".into()));
    }
    let mean: f64 = values.iter().zip(probs).map(|(x, p)| x * p).sum();
    let top = values
        .iter()
        .zip(probs)
        .rev()
        .find(|(_, p)| **p > 0.0)
        .map(|(x, _)| *x)
        .unwrap_or(values[values.len() - 1]);
    if !(premium > mean && premium < top) {
        return Err(Error::Data(format!(
            "a calibrated premium must lie strictly between the mean {mean} and the maximum {top}, \
             not {premium}"
        )));
    }
    if family == Family::Ccoc {
        return Distortion::ccoc((premium - mean) / (top - premium));
    }
    let price = |x: f64| family.with(x).map(|d| d.apply_discrete(values, probs));
    let (lo, hi) = family.range();
    let below = |x: f64| price(x).map_or(true, |v| v < premium);
    let x = if hi.is_infinite() {
        // Double past the target, then bisect on a log scale above a
        // positive lower end.
        let mut b = if lo > 0.0 { 2.0 * lo } else { 1.0 };
        while below(b) {
            b *= 2.0;
            if b > 1e300 {
                return Err(Error::Data(format!(
                    "no {} distortion reaches the premium {premium}",
                    family.name()
                )));
            }
        }
        let a = if lo > 0.0 {
            lo
        } else {
            (b * 1e-300).max(f64::MIN_POSITIVE)
        };
        if lo == 0.0 && !below(a) {
            0.0
        } else {
            bisect_log(a, b, below)
        }
    } else if lo < hi {
        bisect(lo, hi, below)
    } else {
        // Price falls as the parameter rises: search the reflected bracket.
        bisect(hi, lo, |x| !below(x))
    };
    let d = family.with(x)?;
    let got = d.apply_discrete(values, probs);
    if (got - premium).abs() > 1e-9 * premium.abs().max(1.0) {
        return Err(Error::Data(format!(
            "no {} distortion gives the premium {premium} (closest {got})",
            family.name()
        )));
    }
    Ok(d)
}

fn tvar_g(p: f64, s: f64) -> f64 {
    if p >= 1.0 {
        1.0
    } else {
        (s / (1.0 - p)).min(1.0)
    }
}

/// Weights rescaled to sum to 1, after checking they are non-negative and
/// already sum to 1 within `1e-9`.
fn normalized(wts: Vec<f64>) -> Result<Vec<f64>> {
    if wts.iter().any(|w| !(w.is_finite() && *w >= 0.0)) {
        return Err(Error::Data(
            "weights must be finite and non-negative".into(),
        ));
    }
    let sum: f64 = wts.iter().sum();
    if (sum - 1.0).abs() > 1e-9 {
        return Err(Error::Data(format!("weights must sum to 1, not {sum}")));
    }
    Ok(wts.into_iter().map(|w| w / sum).collect())
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

    /// On a uniform, `∫₀¹ φ(u) u du = (e^k (k - 1) + 1) / (k (e^k - 1))`.
    #[test]
    fn exponential_spectral_on_a_uniform() {
        let n = 100_000;
        let u: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
        for k in [0.5f64, 3.0, 20.0] {
            let want = (k.exp() * (k - 1.0) + 1.0) / (k * k.exp_m1());
            let got = Distortion::exponential(k).unwrap().apply_sorted(&u);
            assert!((got - want).abs() < 1e-6, "{k}: {got} vs {want}");
        }
        assert!(Distortion::exponential(0.0).is_err());
    }
    use crate::risk::tvar_sorted;

    const X: [f64; 5] = [10.0, 20.0, 30.0, 40.0, 50.0];

    fn all() -> Vec<Distortion> {
        vec![
            Distortion::tvar(0.7).unwrap(),
            Distortion::wang(0.5).unwrap(),
            Distortion::proportional_hazard(0.6).unwrap(),
            Distortion::dual_power(2.5).unwrap(),
            Distortion::ccoc(0.15).unwrap(),
            Distortion::bitvar(0.2, 0.9, 0.3).unwrap(),
            Distortion::weighted_tvar(vec![0.1, 0.5, 1.0], vec![0.2, 0.5, 0.3]).unwrap(),
            Distortion::capped_linear(0.05, 1.4).unwrap(),
            Distortion::capped_log_linear(0.1, 0.8).unwrap(),
            Distortion::lep(0.02, 0.2).unwrap(),
            Distortion::linear_yield(0.03, 0.5).unwrap(),
            Distortion::beta(0.6, 1.5).unwrap(),
            Distortion::mixture(vec![
                (0.4, Distortion::wang(0.3).unwrap()),
                (0.6, Distortion::ccoc(0.1).unwrap()),
            ])
            .unwrap(),
            Distortion::minimum(vec![
                Distortion::proportional_hazard(0.5).unwrap(),
                Distortion::tvar(0.6).unwrap(),
            ])
            .unwrap(),
            Distortion::convex(&[(0.0, 0.02), (0.01, 0.05), (0.1, 0.25)]).unwrap(),
        ]
    }

    /// A discrete distribution for calibration: P(X = 0) is large, the tail
    /// is long.
    const VALUES: [f64; 6] = [0.0, 1.0, 2.0, 5.0, 10.0, 100.0];
    const PROBS: [f64; 6] = [0.5, 0.2, 0.15, 0.1, 0.04, 0.01];

    #[test]
    fn concave_with_the_right_ends_and_mass() {
        let s: Vec<f64> = (0..=1000).map(|i| i as f64 / 1000.0).collect();
        for d in all() {
            assert_eq!(d.g(0.0), 0.0, "{d:?}");
            assert!((d.g(1.0) - 1.0).abs() < 1e-15, "{d:?}");
            let g: Vec<f64> = s.iter().map(|&s| d.g(s)).collect();
            // Non-decreasing, above the diagonal, and concave on (0, 1].
            for i in 1..g.len() {
                assert!(g[i] >= g[i - 1] - 1e-15, "{d:?} at {}", s[i]);
                assert!(g[i] >= s[i] - 1e-15, "{d:?} at {}", s[i]);
            }
            for i in 2..g.len() {
                assert!(
                    g[i] - 2.0 * g[i - 1] + g[i - 2] <= 1e-12,
                    "{d:?} at {}",
                    s[i]
                );
            }
            // The mass is the limit at 0.
            assert!(
                (d.g(1e-13) - d.mass()).abs() < 1e-6,
                "{d:?}: {} vs {}",
                d.g(1e-13),
                d.mass()
            );
        }
        assert!((Distortion::ccoc(0.25).unwrap().mass() - 0.2).abs() < 1e-15);
        assert_eq!(Distortion::proportional_hazard(0.5).unwrap().mass(), 0.0);
    }

    #[test]
    fn ccoc_is_a_bitvar_and_prices_mean_plus_capital_return() {
        let ccoc = Distortion::ccoc(0.15).unwrap();
        let bt = Distortion::bitvar(0.0, 1.0, 0.15 / 1.15).unwrap();
        for s in [0.001, 0.2, 0.7, 0.99] {
            assert!((ccoc.g(s) - bt.g(s)).abs() < 1e-15);
        }
        let p = ccoc.apply_discrete(&VALUES, &PROBS);
        let mean: f64 = VALUES.iter().zip(&PROBS).map(|(x, p)| x * p).sum();
        assert!((p - (mean + 0.15 * 100.0) / 1.15).abs() < 1e-12);
        // The margin is 15% of the capital 100 - P.
        assert!(((p - mean) / (100.0 - p) - 0.15).abs() < 1e-12);
    }

    #[test]
    fn inverse_and_dual() {
        for d in all() {
            for y in [0.05, 0.3, 0.8] {
                let s = d.g_inv(y);
                // At s = 0 the jump g(0+) = mass already reaches y.
                let reached = if s == 0.0 { d.mass() } else { d.g(s) };
                assert!(reached >= y - 1e-12, "{d:?}");
                // Nothing smaller reaches y.
                assert!(s == 0.0 || d.g(s * (1.0 - 1e-9)) < y + 1e-9, "{d:?}");
            }
            // The bid is at most the mean, the ask at least.
            let mean: f64 = VALUES.iter().zip(&PROBS).map(|(x, p)| x * p).sum();
            let bid = d.apply_discrete_dual(&VALUES, &PROBS);
            let ask = d.apply_discrete(&VALUES, &PROBS);
            assert!(
                bid <= mean + 1e-12 && mean <= ask + 1e-12,
                "{d:?}: {bid} {mean} {ask}"
            );
        }
        let ph = Distortion::proportional_hazard(0.5).unwrap();
        assert!((ph.g_inv(0.5) - 0.25).abs() < 1e-15);
        assert!((ph.g_dual(0.75) - 0.5).abs() < 1e-15);
        // A capped distortion reaches 1 before s = 1.
        assert!((Distortion::tvar(0.6).unwrap().g_inv(1.0) - 0.4).abs() < 1e-15);
    }

    #[test]
    fn convex_hull_drops_interior_points_and_keeps_a_jump() {
        let d = Distortion::convex(&[(0.0, 0.02), (0.01, 0.05), (0.1, 0.25), (0.05, 0.1)]).unwrap();
        // (0.05, 0.1) lies below the chord from (0.01, 0.05) to (0.1, 0.25).
        assert_eq!(
            d,
            Distortion::Convex(vec![
                (0.0, 0.0),
                (0.0, 0.02),
                (0.01, 0.05),
                (0.1, 0.25),
                (1.0, 1.0)
            ])
        );
        assert_eq!(d.mass(), 0.02);
        assert!((d.g(0.005) - 0.035).abs() < 1e-15);
        assert!(Distortion::convex(&[(0.5, 0.4)]).is_err()); // below the diagonal
        assert!(Distortion::convex(&[(1.5, 0.4)]).is_err());
    }

    #[test]
    fn calibration_hits_the_premium_in_every_family() {
        let mean: f64 = VALUES.iter().zip(&PROBS).map(|(x, p)| x * p).sum();
        for premium in [mean * 1.05, mean * 1.6, 20.0] {
            for family in [
                Family::Ccoc,
                Family::ProportionalHazard,
                Family::Wang,
                Family::DualPower,
                Family::Tvar,
                Family::Exponential,
                Family::CappedLinear { r0: 0.0 },
                Family::CappedLogLinear { r0: 0.0 },
                Family::Lep { r0: 0.0 },
                Family::LinearYield { r0: 0.0 },
            ] {
                if premium == 20.0 && family == (Family::Lep { r0: 0.0 }) {
                    // LEP's price is bounded below the maximum: its g
                    // tends to min(1, s + √(s (1 - s))) as r grows.
                    assert!(calibrate(family, &VALUES, &PROBS, premium).is_err());
                    continue;
                }
                let d = calibrate(family, &VALUES, &PROBS, premium)
                    .unwrap_or_else(|e| panic!("{family:?} at {premium}: {e}"));
                let got = d.apply_discrete(&VALUES, &PROBS);
                assert!(
                    (got - premium).abs() < 1e-9,
                    "{family:?}: {got} vs {premium}"
                );
            }
        }
        // Outside (mean, max) there is nothing to find.
        assert!(calibrate(Family::Wang, &VALUES, &PROBS, mean).is_err());
        assert!(calibrate(Family::Wang, &VALUES, &PROBS, 100.0).is_err());
        // A floor that already overprices fails cleanly.
        assert!(
            calibrate(
                Family::CappedLinear { r0: 0.5 },
                &VALUES,
                &PROBS,
                mean * 1.05
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_bad_new_parameters() {
        assert!(Distortion::ccoc(-0.1).is_err());
        assert!(Distortion::bitvar(0.9, 0.2, 0.5).is_err());
        assert!(Distortion::weighted_tvar(vec![0.5, 0.1], vec![0.5, 0.5]).is_err());
        assert!(Distortion::weighted_tvar(vec![0.1, 0.5], vec![0.5, 0.4]).is_err());
        assert!(Distortion::capped_linear(0.2, 0.5).is_err());
        assert!(Distortion::capped_log_linear(0.0, 1.5).is_err());
        assert!(Distortion::lep(0.3, 0.2).is_err());
        assert!(Distortion::linear_yield(0.0, -0.5).is_err());
        assert!(Distortion::beta(1.5, 2.0).is_err());
        assert!(Distortion::mixture(vec![]).is_err());
        assert!(Distortion::minimum(vec![]).is_err());
    }

    #[test]
    fn tvar_matches_the_risk_module() {
        for i in 0..=100 {
            let p = i as f64 / 100.0;
            let d = Distortion::tvar(p).unwrap().apply_sorted(&X);
            let t = tvar_sorted(&X, p).unwrap();
            assert!((d - t).abs() <= 1e-12 * t, "p = {p}: {d} vs {t}");
        }
    }

    #[test]
    fn identity_parameters_give_the_mean() {
        for d in [
            Distortion::tvar(0.0).unwrap(),
            Distortion::wang(0.0).unwrap(),
            Distortion::proportional_hazard(1.0).unwrap(),
            Distortion::dual_power(1.0).unwrap(),
        ] {
            assert!((d.apply_sorted(&X) - 30.0).abs() < 1e-12, "{d:?}");
        }
    }

    #[test]
    fn weights_are_a_probability_vector_rising_to_the_tail() {
        for d in all() {
            let w = d.weights(1000);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{d:?}");
            assert!(w.iter().all(|&w| w >= 0.0));
            // Concave g: weights never fall towards the larger values.
            assert!(w.windows(2).all(|p| p[1] >= p[0] - 1e-15), "{d:?}");
        }
    }

    #[test]
    fn coherence_properties() {
        let mean = 30.0;
        for d in all() {
            let r = d.apply_sorted(&X);
            assert!(r >= mean && r <= 50.0, "{d:?}: {r}");
            // Translation and positive scaling.
            let shifted: Vec<f64> = X.iter().map(|x| 2.0 * x + 7.0).collect();
            assert!((d.apply_sorted(&shifted) - (2.0 * r + 7.0)).abs() < 1e-12);
        }
        // A larger parameter is more conservative.
        let w = |l| Distortion::wang(l).unwrap().apply_sorted(&X);
        assert!(w(0.2) < w(0.5) && w(0.5) < w(1.0));
        let ph = |r| Distortion::proportional_hazard(r).unwrap().apply_sorted(&X);
        assert!(ph(0.9) < ph(0.5));
        let dp = |b| Distortion::dual_power(b).unwrap().apply_sorted(&X);
        assert!(dp(1.5) < dp(3.0));
    }

    #[test]
    fn discrete_agrees_with_equal_weights_and_merges_ties() {
        let p = [0.2; 5];
        for d in all() {
            let a = d.apply_discrete(&X, &p);
            let b = d.apply_sorted(&X);
            assert!((a - b).abs() < 1e-12, "{d:?}");
        }
        // Tied draws and one atom with their total mass are the same.
        let d = Distortion::dual_power(2.0).unwrap();
        let ties = d.apply_sorted(&[1.0, 5.0, 5.0, 5.0]);
        let atom = d.apply_discrete(&[1.0, 5.0], &[0.25, 0.75]);
        assert!((ties - atom).abs() < 1e-15);
        // Closed form: g(0.75) × 5 + (1 - g(0.75)) × 1.
        assert!((atom - (0.9375 * 5.0 + 0.0625)).abs() < 1e-15);
    }

    #[test]
    fn g_closed_forms() {
        let s = 0.3;
        assert_eq!(Distortion::tvar(0.8).unwrap().g(s), 1.0);
        assert!((Distortion::tvar(0.4).unwrap().g(s) - 0.5).abs() < 1e-15);
        assert!((Distortion::proportional_hazard(0.5).unwrap().g(s) - s.sqrt()).abs() < 1e-15);
        assert!((Distortion::dual_power(2.0).unwrap().g(s) - 0.51).abs() < 1e-15);
        // Φ(Φ⁻¹(0.5) + 1) = Φ(1).
        assert!((Distortion::wang(1.0).unwrap().g(0.5) - norm_cdf(1.0)).abs() < 1e-15);
        assert_eq!(Distortion::tvar(1.0).unwrap().g(1e-300), 1.0);
        for d in all() {
            assert_eq!(d.g(0.0), 0.0);
            assert_eq!(d.g(1.0), 1.0);
        }
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(Distortion::tvar(1.5).is_err());
        assert!(Distortion::wang(-0.1).is_err());
        assert!(Distortion::wang(f64::INFINITY).is_err());
        assert!(Distortion::proportional_hazard(0.0).is_err());
        assert!(Distortion::proportional_hazard(1.5).is_err());
        assert!(Distortion::dual_power(0.5).is_err());
    }
}
