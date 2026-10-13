//! Pricing bounds: the range of prices that the distortions pricing a
//! reference risk `X` at `P` give another risk, and the envelope of those
//! distortions. Mildenhall, *Pricing bounds* (IME 2022), and `aggregate`'s
//! `PricingBounds` and `Bounds`.
//!
//! A coherent distortion has a Kusuoka measure `μ` on `[0, 1]` and prices
//! any risk `Z` at `ρ(Z) = ∫ TVaR_p(Z) μ(dp)`. The distortions with
//! `ρ(X) = P` form a convex set whose extreme points are BiTVaRs,
//! `(1 - w) TVaR_p0 + w TVaR_p1`. So the pairs `(ρ(X), ρ(Y))` they give
//! are the convex hull of the curve `p ↦ (TVaR_p(X), TVaR_p(Y))`, and the
//! range of `ρ(Y)` is the hull's vertical slice at `P`.
//!
//! For discrete `X` and `Y`, with `u = 1 / (1 - p)`, `TVaR_p` is affine in
//! `u` within an atom of the quantile function, so between consecutive
//! points of the union of the two distributions' breakpoints (the
//! cumulative probabilities) both coordinates are affine in `u`, and the
//! curve is a polyline: the hull is exact. As `p → 1` the curve ends at
//! `(max X, max Y)`, the pricing measure `TVaR_1` (`aggregate` stops at the
//! last breakpoint below 1, which misses the upper bound when `Y`'s top atom
//! is smaller than `X`'s).
//!
//! [`distortion_envelope`]: a BiTVaR's `g(s)` is the price of a Bernoulli
//! risk with `P(1) = s`, since `TVaR_p` of that risk is `min(1, s / (1 - p))`.
//! So the smallest and largest `g(s)` over the distortions pricing `X` at
//! `P` are the bounds for that Bernoulli: exact, at any `s`. `aggregate`'s
//! `Bounds` searches a grid of 256 levels.
//!
//! The uniform reference (the Gini lens, `TVaR_p(U) = (1 + p) / 2`) is not
//! provided: it is affine in `p`, not `u`, so the hull at the breakpoints
//! is not exact there.

use prospicio_core::{Error, Result};
use prospicio_prob::Distortion;

/// A discrete distribution capped at the assets, ascending and merged, with
/// its tail masses and tail means summed from the top.
#[derive(Debug, Clone, PartialEq)]
struct Discrete {
    x: Vec<f64>,
    /// `P(X > x_k)`, exactly 0 at the top.
    tail: Vec<f64>,
    /// `Σ_{j > k} x_j p_j`.
    above: Vec<f64>,
    p: Vec<f64>,
}

impl Discrete {
    fn new(values: &[f64], probs: &[f64], cap: f64) -> Result<Self> {
        if values.is_empty() || values.len() != probs.len() {
            return Err(Error::Data(
                "give one probability per value, at least one".into(),
            ));
        }
        if values.iter().any(|v| !v.is_finite()) || probs.iter().any(|p| p.is_nan() || *p < 0.0) {
            return Err(Error::Data(
                "values must be finite and probabilities non-negative".into(),
            ));
        }
        let sum: f64 = probs.iter().sum();
        if (sum - 1.0).abs() > 1e-9 {
            return Err(Error::Data(format!("probabilities sum to {sum}, not 1")));
        }
        let mut pairs: Vec<(f64, f64)> = values
            .iter()
            .zip(probs)
            .filter(|(_, p)| **p > 0.0)
            .map(|(&v, &p)| (v.min(cap), p / sum))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut x, mut p): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        for (v, q) in pairs {
            if x.last() == Some(&v) {
                *p.last_mut().expect("pushed with x") += q;
            } else {
                x.push(v);
                p.push(q);
            }
        }
        let n = x.len();
        let (mut tail, mut above) = (vec![0.0; n], vec![0.0; n]);
        for k in (0..n - 1).rev() {
            tail[k] = tail[k + 1] + p[k + 1];
            above[k] = above[k + 1] + x[k + 1] * p[k + 1];
        }
        Ok(Self { x, tail, above, p })
    }

    /// The tail masses `1 - p` where the quantile function steps.
    fn breakpoints(&self) -> &[f64] {
        &self.tail[..self.tail.len() - 1]
    }

    /// `TVaR` at tail mass `s = 1 - p`: the mean of the top `s` of the
    /// distribution, `max X` at `s = 0`.
    fn tvar(&self, s: f64) -> f64 {
        let n = self.x.len();
        if s <= 0.0 {
            return self.x[n - 1];
        }
        // The atom k holding the level: tail[k] < s <= tail[k] + p[k].
        let k = self.tail.partition_point(|&t| t >= s).min(n - 1);
        (self.above[k] + self.x[k] * (s - self.tail[k])) / s
    }

    fn mean(&self) -> f64 {
        self.tvar(1.0)
    }

    fn max(&self) -> f64 {
        self.x[self.x.len() - 1]
    }
}

/// One risk's price range, from [`PricingBounds::bounds`].
#[derive(Debug, Clone, PartialEq)]
pub struct PriceRange {
    /// The risk's name.
    pub risk: String,
    /// The lowest price.
    pub lower: f64,
    /// The highest price.
    pub upper: f64,
    /// A distortion pricing `X` at `P` that gives the lowest.
    pub lower_distortion: Distortion,
    /// One that gives the highest.
    pub upper_distortion: Distortion,
}

/// The convex hull of the curve `(TVaR_p(X), TVaR_p(Y))`: see the
/// [module](self).
#[derive(Debug, Clone, PartialEq)]
pub struct PricingBounds {
    x: Discrete,
    names: Vec<String>,
    /// Vertices with strictly increasing `TVaR_p(X)`.
    t: Vec<f64>,
    /// Each vertex's tail mass `1 - p`, at the lowest and highest `Y`.
    s_lo: Vec<Vec<f64>>,
    s_hi: Vec<Vec<f64>>,
    /// Each risk's lowest and highest `TVaR_p(Y)` at each vertex (they
    /// differ only where `TVaR_p(X)` saturates at `max X`).
    y_lo: Vec<Vec<f64>>,
    y_hi: Vec<Vec<f64>>,
    lower_hull: Vec<Vec<usize>>,
    upper_hull: Vec<Vec<usize>>,
}

impl PricingBounds {
    /// The bounds for the risks `ys` (name, values, probabilities) given
    /// the reference `X` (values, probabilities), each capped at `assets`
    /// (`f64::INFINITY` for none). Values need not be sorted or distinct.
    ///
    /// ```
    /// use prospicio_pricing::bounds::PricingBounds;
    ///
    /// // X is 0 or 10; Y is 0, 4 or 20. Price X at 6 (its mean is 5).
    /// let pb = PricingBounds::new(
    ///     (&[0.0, 10.0], &[0.5, 0.5]),
    ///     &[("Y".into(), vec![0.0, 4.0, 20.0], vec![0.5, 0.3, 0.2])],
    ///     f64::INFINITY,
    /// )
    /// .unwrap();
    /// let r = &pb.bounds(6.0).unwrap()[0];
    /// // Y's mean is 5.2; every distortion pricing X at 6 prices Y here.
    /// assert!(r.lower > 5.2 && r.upper < 20.0 && r.lower < r.upper);
    /// // Each bound is attained by a distortion that prices X at 6.
    /// assert!((r.lower_distortion.apply_discrete(&[0.0, 10.0], &[0.5, 0.5]) - 6.0).abs() < 1e-12);
    /// ```
    pub fn new(
        x: (&[f64], &[f64]),
        ys: &[(String, Vec<f64>, Vec<f64>)],
        assets: f64,
    ) -> Result<Self> {
        if assets.is_nan() || assets <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "assets",
                value: assets,
                reason: "must be positive (infinite for no cap)",
            });
        }
        if ys.is_empty() {
            return Err(Error::Data("give at least one risk to bound".into()));
        }
        let x = Discrete::new(x.0, x.1, assets)?;
        let yd = ys
            .iter()
            .map(|(_, v, p)| Discrete::new(v, p, assets))
            .collect::<Result<Vec<_>>>()?;
        // Vertices: p = 0, every breakpoint, and p → 1, as tail masses
        // descending from 1 to 0.
        let mut s: Vec<f64> = [1.0, 0.0]
            .into_iter()
            .chain(x.breakpoints().iter().copied())
            .chain(yd.iter().flat_map(|y| y.breakpoints().iter().copied()))
            .collect();
        s.sort_by(|a, b| b.total_cmp(a));
        s.dedup();
        let m = ys.len();
        let (mut t, mut s_lo, mut s_hi) = (Vec::new(), vec![Vec::new(); m], vec![Vec::new(); m]);
        let (mut y_lo, mut y_hi) = (vec![Vec::new(); m], vec![Vec::new(); m]);
        for &sv in &s {
            let tv = x.tvar(sv);
            let tied = t.last().is_some_and(|&last: &f64| tv <= last);
            for j in 0..m {
                let yv = yd[j].tvar(sv);
                if tied {
                    let k = t.len() - 1;
                    if yv < y_lo[j][k] {
                        y_lo[j][k] = yv;
                        s_lo[j][k] = sv;
                    }
                    if yv > y_hi[j][k] {
                        y_hi[j][k] = yv;
                        s_hi[j][k] = sv;
                    }
                } else {
                    y_lo[j].push(yv);
                    y_hi[j].push(yv);
                    s_lo[j].push(sv);
                    s_hi[j].push(sv);
                }
            }
            if !tied {
                t.push(tv);
            }
        }
        let lower_hull = (0..m).map(|j| hull(&t, &y_lo[j], 1.0)).collect();
        let upper_hull = (0..m).map(|j| hull(&t, &y_hi[j], -1.0)).collect();
        Ok(Self {
            x,
            names: ys.iter().map(|(n, _, _)| n.clone()).collect(),
            t,
            s_lo,
            s_hi,
            y_lo,
            y_hi,
            lower_hull,
            upper_hull,
        })
    }

    /// The premiums `X` can be priced at: from its mean to its maximum.
    pub fn premium_range(&self) -> (f64, f64) {
        (self.x.mean(), self.x.max())
    }

    /// The vertices of the curve: `TVaR_p(X)` and, for each risk, its
    /// lowest and highest `TVaR_p(Y)` there.
    pub fn curve(&self) -> (&[f64], &[Vec<f64>], &[Vec<f64>]) {
        (&self.t, &self.y_lo, &self.y_hi)
    }

    /// Each risk's price range over the distortions that price `X` at
    /// `premium`, and distortions attaining each end.
    pub fn bounds(&self, premium: f64) -> Result<Vec<PriceRange>> {
        let (lo, hi) = self.premium_range();
        let tol = 1e-12 * hi.abs().max(1.0);
        if !(premium >= lo - tol && premium <= hi + tol) {
            return Err(Error::Data(format!(
                "the premium must lie between E[X] = {lo} and max X = {hi}, not {premium}"
            )));
        }
        let premium = premium.clamp(lo, hi);
        Ok((0..self.names.len())
            .map(|j| {
                let (lower, lower_distortion) =
                    self.slice(&self.lower_hull[j], &self.y_lo[j], &self.s_lo[j], premium);
                let (upper, upper_distortion) =
                    self.slice(&self.upper_hull[j], &self.y_hi[j], &self.s_hi[j], premium);
                PriceRange {
                    risk: self.names[j].clone(),
                    lower,
                    upper,
                    lower_distortion,
                    upper_distortion,
                }
            })
            .collect())
    }

    /// The hull edge over `premium`, its height there, and the BiTVaR on
    /// its two ends.
    fn slice(&self, h: &[usize], y: &[f64], s: &[f64], premium: f64) -> (f64, Distortion) {
        if h.len() == 1 {
            return (y[h[0]], Distortion::Tvar(1.0 - s[h[0]]));
        }
        let e = h
            .windows(2)
            .position(|w| premium <= self.t[w[1]])
            .unwrap_or(h.len() - 2);
        let (i0, i1) = (h[e], h[e + 1]);
        let w = ((premium - self.t[i0]) / (self.t[i1] - self.t[i0])).clamp(0.0, 1.0);
        let value = (1.0 - w) * y[i0] + w * y[i1];
        let (p0, p1) = (1.0 - s[i0], 1.0 - s[i1]);
        let g = if w == 0.0 {
            Distortion::Tvar(p0)
        } else if w == 1.0 {
            Distortion::Tvar(p1)
        } else {
            Distortion::BiTvar { p0, p1, w }
        };
        (value, g)
    }
}

/// Andrew's monotone chain over points with strictly increasing `t`: the
/// lower envelope for `sign = 1`, the upper for `-1`.
fn hull(t: &[f64], y: &[f64], sign: f64) -> Vec<usize> {
    let mut idx: Vec<usize> = Vec::with_capacity(t.len());
    for i in 0..t.len() {
        while idx.len() >= 2 {
            let (o, a) = (idx[idx.len() - 2], idx[idx.len() - 1]);
            let cross = (t[a] - t[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (t[i] - t[o]);
            if sign * cross <= 0.0 {
                idx.pop();
            } else {
                break;
            }
        }
        idx.push(i);
    }
    idx
}

/// The smallest and largest `g(s)` at each `s` over the distortions that
/// price `X` (values, probabilities, capped at `assets`) at `premium`:
/// `aggregate`'s `Bounds` envelopes, exact. Each is the price range of a
/// Bernoulli risk with `P(1) = s`; see the [module](self). The lower
/// envelope is a concave distortion; the upper, in general, is not.
///
/// ```
/// use prospicio_pricing::bounds::distortion_envelope;
///
/// // X is 0 or 10, priced at 6: g(1/2) = 0.6 for every such distortion
/// // (it is the price of X / 10), and g ranges more elsewhere.
/// let env = distortion_envelope((&[0.0, 10.0], &[0.5, 0.5]), 6.0, f64::INFINITY, &[0.25, 0.5]).unwrap();
/// assert!((env[1].0 - 0.6).abs() < 1e-12 && (env[1].1 - 0.6).abs() < 1e-12);
/// assert!(env[0].0 < env[0].1);
/// ```
pub fn distortion_envelope(
    x: (&[f64], &[f64]),
    premium: f64,
    assets: f64,
    s: &[f64],
) -> Result<Vec<(f64, f64)>> {
    if assets.is_nan() || assets <= 0.0 {
        return Err(Error::InvalidParameter {
            name: "assets",
            value: assets,
            reason: "must be positive (infinite for no cap)",
        });
    }
    s.iter()
        .map(|&s| {
            if !(0.0..=1.0).contains(&s) {
                return Err(Error::InvalidParameter {
                    name: "s",
                    value: s,
                    reason: "must be in [0, 1]",
                });
            }
            if s == 0.0 || s == 1.0 {
                return Ok((s, s));
            }
            let bernoulli = ("s".to_string(), vec![0.0, 1.0], vec![1.0 - s, s]);
            // Cap X only: the Bernoulli's values are not losses.
            let capped: Vec<f64> = x.0.iter().map(|v| v.min(assets)).collect();
            let pb = PricingBounds::new(
                (&capped, x.1),
                std::slice::from_ref(&bernoulli),
                f64::INFINITY,
            )?;
            let r = &pb.bounds(premium)?[0];
            Ok((r.lower, r.upper))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reprice(g: &Distortion, x: &[f64], p: &[f64]) -> f64 {
        g.apply_discrete(x, p)
    }

    #[test]
    fn bounds_are_attained_and_price_x_at_p() {
        let (xv, xp) = (
            vec![0.0, 1.0, 3.0, 8.0, 20.0],
            vec![0.3, 0.3, 0.2, 0.15, 0.05],
        );
        let (yv, yp) = (vec![0.0, 2.0, 5.0, 40.0], vec![0.5, 0.25, 0.2, 0.05]);
        let pb = PricingBounds::new(
            (&xv, &xp),
            &[
                ("Y".into(), yv.clone(), yp.clone()),
                ("X".into(), xv.clone(), xp.clone()),
            ],
            f64::INFINITY,
        )
        .unwrap();
        for premium in [3.2, 4.0, 6.5, 12.0, 19.0] {
            let b = pb.bounds(premium).unwrap();
            for (r, (v, p)) in b.iter().zip([(&yv, &yp), (&xv, &xp)]) {
                for (value, g) in [
                    (r.lower, &r.lower_distortion),
                    (r.upper, &r.upper_distortion),
                ] {
                    assert!((reprice(g, &xv, &xp) - premium).abs() < 1e-10, "{g:?}");
                    assert!((reprice(g, v, p) - value).abs() < 1e-10, "{g:?}");
                }
                assert!(r.lower <= r.upper + 1e-12);
            }
            // Y = X is priced at P by every member.
            assert!((b[1].lower - premium).abs() < 1e-10 && (b[1].upper - premium).abs() < 1e-10);
            // Spot check against TVaR and CCoC members of the set.
            let p_star = prospicio_math::roots::bisect(0.0, 1.0, |p| {
                reprice(&Distortion::Tvar(p), &xv, &xp) < premium
            });
            let t = reprice(&Distortion::Tvar(p_star), &yv, &yp);
            assert!(t >= b[0].lower - 1e-8 && t <= b[0].upper + 1e-8);
        }
        assert!(pb.bounds(2.0).is_err());
        // Y's top (40) is above X's: the upper bound near max X uses TVaR_1.
        let top = pb.bounds(19.9).unwrap();
        assert!(top[0].upper > 35.0);
    }

    #[test]
    fn envelope_contains_the_members() {
        let (xv, xp) = (vec![0.0, 1.0, 3.0, 8.0], vec![0.4, 0.3, 0.2, 0.1]);
        let mean: f64 = xv.iter().zip(&xp).map(|(a, b)| a * b).sum();
        let premium = mean * 1.4;
        let s: Vec<f64> = (0..=40).map(|k| k as f64 / 40.0).collect();
        let env = distortion_envelope((&xv, &xp), premium, f64::INFINITY, &s).unwrap();
        // The CCoC member pricing X at P.
        let r = prospicio_math::roots::bisect(0.0, 100.0, |r| {
            Distortion::ccoc(r).unwrap().apply_discrete(&xv, &xp) < premium
        });
        let g = Distortion::ccoc(r).unwrap();
        for (&sv, &(lo, hi)) in s.iter().zip(&env) {
            let v = g.g(sv);
            assert!(
                v >= lo - 1e-9 && v <= hi + 1e-9,
                "s = {sv}: {lo} <= {v} <= {hi}"
            );
        }
        // Capping at the assets.
        let capped = distortion_envelope((&xv, &xp), 2.0, 3.0, &[0.2]).unwrap();
        assert!(capped[0].0 <= capped[0].1);
    }
}
