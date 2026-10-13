//! Layer loss picks: reshaping a severity so that its expected loss in each
//! layer of a tower hits a target, `aggregate`'s `Aggregate.picks`.
//!
//! An actuary often has a pick for the loss to each layer (from experience
//! or exposure rating) and a severity curve that does not quite reproduce
//! them. The attachments `0 = a₀ < a₁ < … < a_k` cut the severity into
//! layers `(a_{i-1}, a_i]`; the tail above `a_k` is kept. In each layer the
//! survival function is rescaled about its value at the top,
//!
//! `S*(x) = ωᵢ + wᵢ (S(x) - S(aᵢ))`,
//!
//! where `ωᵢ = S*(aᵢ)` is the adjusted chance of a loss above the layer and
//! the weight `wᵢ` makes `∫ S*` over the layer equal its target `tᵢ`:
//! `wᵢ = (tᵢ - ωᵢ yᵢ) / mᵢ`, with `yᵢ` the layer's width and
//! `mᵢ = ∫ (S - S(aᵢ))` over it. Working from the top, `ω_k = S(a_k)` and
//! `ω_{i-1} = ωᵢ + wᵢ (S(a_{i-1}) - S(aᵢ))`, so `S*` is continuous. The
//! shape within each layer is the original severity's.
//!
//! A pick below `ωᵢ yᵢ`, what the layers above already force through it,
//! needs a negative weight: the picks are infeasible and an error says
//! which layer. A bottom-layer pick too large for the rescaled curve to stay
//! at or below 1 takes `aggregate`'s shape instead: `S* = 1` up to a point,
//! then a straight line down to `ω₁` at `a₁`, whose length is solved so
//! that the layer's loss is the pick (`aggregate` rounds a continuous
//! length to the grid and misses by up to about half a step).

use prospicio_core::{Error, Result};
use prospicio_prob::Grid;

/// One layer of [`layer_loss_picks`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PickedLayer {
    /// The layer's bottom, `a_{i-1}`.
    pub bottom: f64,
    /// Its top, `aᵢ`.
    pub top: f64,
    /// The expected loss to the layer per ground-up claim before,
    /// `E[(X - a_{i-1})⁺ ∧ yᵢ]`.
    pub before: f64,
    /// The target per claim, the pick over the expected claim count.
    pub target: f64,
    /// The weight `wᵢ` on the survival function in the layer.
    pub weight: f64,
    /// `ωᵢ`, the adjusted chance of a loss above the layer.
    pub above: f64,
}

/// The severity reshaped to the picks, from [`layer_loss_picks`].
#[derive(Debug, Clone, PartialEq)]
pub struct Picks {
    /// The adjusted severity, on the same grid.
    pub severity: Grid,
    /// Each layer, bottom up.
    pub layers: Vec<PickedLayer>,
    /// Whether the bottom layer took the straight-line shape (its pick was
    /// too large for a rescaled survival function).
    pub straight_bottom: bool,
}

/// Reshapes `severity` so that the expected loss to each layer ending at
/// `attachments[i]` is `picks[i] / claims`: `picks` are the layer loss
/// picks for a year with `claims` expected ground-up claims (give
/// `claims = 1` for picks per claim). The attachments must be increasing,
/// positive grid points; the bottom layer starts at 0. See the
/// [module](self).
///
/// ```
/// use prospicio_aggregate::picks::layer_loss_picks;
/// use prospicio_prob::{Distribution, Grid, Lognormal};
///
/// let sev = Lognormal::from_mean_cv(100.0, 2.0).unwrap();
/// let (grid, _) = Grid::rounding(&sev, 1.0, 1 << 14).unwrap();
/// // Picks for 10 claims: 600 to the first 100, 250 to 400 xs 100, 50 to
/// // 500 xs 500; above 1,000 the severity is kept.
/// let picked = layer_loss_picks(&grid, &[100.0, 500.0, 1000.0], &[600.0, 250.0, 50.0], 10.0).unwrap();
/// let lev = |g: &Grid, a: f64| (0..g.len()).map(|j| g.x(j).min(a) * g.probs()[j]).sum::<f64>();
/// let layer = |a: f64, b: f64| lev(&picked.severity, b) - lev(&picked.severity, a);
/// assert!((layer(0.0, 100.0) - 60.0).abs() < 1e-9);
/// assert!((layer(100.0, 500.0) - 25.0).abs() < 1e-9);
/// assert!((layer(500.0, 1000.0) - 5.0).abs() < 1e-9);
/// ```
pub fn layer_loss_picks(
    severity: &Grid,
    attachments: &[f64],
    picks: &[f64],
    claims: f64,
) -> Result<Picks> {
    if attachments.is_empty() || attachments.len() != picks.len() {
        return Err(Error::Data(
            "give one layer loss pick per attachment".into(),
        ));
    }
    if !(claims.is_finite() && claims > 0.0) {
        return Err(Error::InvalidParameter {
            name: "claims",
            value: claims,
            reason: "must be finite and positive",
        });
    }
    let h = severity.step();
    let n = severity.len();
    // Each attachment's grid index; it must be a grid point inside the grid.
    let mut index = Vec::with_capacity(attachments.len());
    for &a in attachments {
        let j = (a / h).round();
        if !(a > 0.0 && (a / h - j).abs() <= 1e-9 * j.max(1.0) && (j as usize) < n) {
            return Err(Error::Data(format!(
                "attachment {a} is not a positive point of the grid (step {h}, {n} points): a \
                 pick constrains the loss between two boundaries, which must be grid points"
            )));
        }
        if index.last().is_some_and(|&last| j as usize <= last) {
            return Err(Error::Data("the attachments must increase".into()));
        }
        index.push(j as usize);
    }
    // S_j = P(X > jh); E[X ∧ jh] = h Σ_{i<j} S_i.
    let p = severity.probs();
    let mut s = vec![0.0; n];
    let mut above = 0.0;
    for j in (0..n).rev() {
        s[j] = above;
        above += p[j];
    }
    let integral = |from: usize, to: usize| h * s[from..to].iter().sum::<f64>();
    let k = index.len();
    let mut layers = Vec::with_capacity(k);
    let mut omega = s[index[k - 1]];
    let mut weights = vec![0.0; k];
    let mut omegas = vec![0.0; k];
    for i in (0..k).rev() {
        let lo = if i == 0 { 0 } else { index[i - 1] };
        let hi = index[i];
        let y = (hi - lo) as f64 * h;
        let before = integral(lo, hi);
        let m = before - s[hi] * y;
        let target = picks[i] / claims;
        if !(target.is_finite() && target >= 0.0) {
            return Err(Error::Data(format!(
                "layer loss picks must be finite and non-negative, not {}",
                picks[i]
            )));
        }
        if m <= 0.0 {
            return Err(Error::Data(format!(
                "the severity has no loss within the layer to {}, so it cannot be reshaped",
                attachments[i]
            )));
        }
        let w = (target - omega * y) / m;
        if w < 0.0 {
            return Err(Error::Data(format!(
                "the pick for the layer to {} is below the {} the layers above force through \
                 it: the picks are infeasible",
                attachments[i],
                omega * y * claims
            )));
        }
        weights[i] = w;
        omegas[i] = omega;
        layers.push(PickedLayer {
            bottom: lo as f64 * h,
            top: hi as f64 * h,
            before,
            target,
            weight: w,
            above: omega,
        });
        omega += w * (s[lo] - s[hi]);
    }
    layers.reverse();
    // The adjusted survival; layer i holds the points in (a_{i-1}, a_i],
    // and 0 is in the bottom one. It is continuous at the attachments.
    let mut adjusted = s.clone();
    let mut capped = false;
    for i in 0..k {
        let lo = if i == 0 { 0 } else { index[i - 1] + 1 };
        for j in lo..=index[i] {
            let v = omegas[i] + (s[j] - s[index[i]]) * weights[i];
            if v > 1.0 + 1e-12 {
                if i > 0 || omegas[0] > 1.0 {
                    return Err(Error::Data(format!(
                        "the picks are infeasible: the layers from {} up need more than every                          claim to reach them",
                        layers[i.max(1) - 1].top
                    )));
                }
                capped = true;
            }
            adjusted[j] = v.min(1.0);
        }
    }
    if capped {
        let (a1, t1, w1) = (layers[0].top, layers[0].target, omegas[0]);
        if t1 > a1 {
            return Err(Error::Data(format!(
                "the pick for the layer to {a1} is more than its width per claim: the picks                  are infeasible"
            )));
        }
        // S* = 1 on [0, a₁ - s₀], then a line down to ω₁ at a₁. In the
        // continuous model s₀ = 2 (a₁ - t₁) / (1 - ω₁); `aggregate` snaps that
        // to the grid and misses the pick by up to about h / 2. Here s₀ is
        // solved on the grid: the layer's loss falls as s₀ grows, from a₁
        // towards ω₁ a₁ <= t₁.
        let ramp = |s0: f64, out: &mut [f64]| {
            let start = a1 - s0;
            for (j, v) in out.iter_mut().enumerate().take(index[0] + 1) {
                let t = ((j as f64 * h - start) / s0).max(0.0);
                *v = 1.0 - t * (1.0 - w1);
            }
        };
        let mut work = adjusted.clone();
        let mut loss = |s0: f64| {
            ramp(s0, &mut work);
            h * work[..index[0]].iter().sum::<f64>()
        };
        let s0 = prospicio_math::roots::bisect_log(1e-9 * h, 1e9 * a1, |s0| loss(s0) > t1);
        ramp(s0, &mut adjusted);
    }
    let mut probs = Vec::with_capacity(n);
    let mut previous = 1.0;
    for (j, &v) in adjusted.iter().enumerate() {
        let q = previous - v;
        if q < -1e-12 {
            return Err(Error::Data(format!(
                "the picks are infeasible: the adjusted severity has probability {q} at {}",
                j as f64 * h
            )));
        }
        // Above the top attachment the severity is kept as it was.
        probs.push(if j > index[k - 1] { p[j] } else { q.max(0.0) });
        previous = v;
    }
    Ok(Picks {
        severity: Grid::new(h, probs)?,
        layers,
        straight_bottom: capped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(g: &Grid, a: f64, b: f64) -> f64 {
        (0..g.len())
            .map(|j| (g.x(j).min(b) - g.x(j).min(a)) * g.probs()[j])
            .sum()
    }

    #[test]
    fn hits_the_picks_and_keeps_the_tail() {
        let mut p = [0.0; 21];
        for (j, v) in p.iter_mut().enumerate().skip(1) {
            *v = 0.9f64.powi(j as i32);
        }
        let sum: f64 = p.iter().sum();
        let p: Vec<f64> = p.iter().map(|v| v / sum).collect();
        let g = Grid::new(1.0, p.clone()).unwrap();
        let picked = layer_loss_picks(&g, &[4.0, 10.0], &[3.0, 2.0], 1.0).unwrap();
        assert!(!picked.straight_bottom);
        assert!((layer(&picked.severity, 0.0, 4.0) - 3.0).abs() < 1e-12);
        assert!((layer(&picked.severity, 4.0, 10.0) - 2.0).abs() < 1e-12);
        // Above the top attachment the severity is unchanged.
        assert_eq!(&picked.severity.probs()[11..], &p[11..]);
        // The picks as they are reproduce the severity.
        let same = layer_loss_picks(
            &g,
            &[4.0, 10.0],
            &[layer(&g, 0.0, 4.0), layer(&g, 4.0, 10.0)],
            1.0,
        )
        .unwrap();
        for (a, b) in same.severity.probs().iter().zip(&p) {
            assert!((a - b).abs() < 1e-14);
        }
        // Infeasible: almost nothing through a layer the one above crosses.
        assert!(layer_loss_picks(&g, &[4.0, 10.0], &[0.01, 2.0], 1.0).is_err());
        assert!(layer_loss_picks(&g, &[4.5], &[1.0], 1.0).is_err());
        assert!(layer_loss_picks(&g, &[10.0, 4.0], &[1.0, 1.0], 1.0).is_err());
    }

    #[test]
    fn a_large_bottom_pick_takes_the_straight_line() {
        let mut p = [0.0; 21];
        for (j, v) in p.iter_mut().enumerate().skip(1) {
            *v = 0.7f64.powi(j as i32);
        }
        let sum: f64 = p.iter().sum();
        let g = Grid::new(1.0, p.iter().map(|v| v / sum).collect()).unwrap();
        let picked = layer_loss_picks(&g, &[10.0], &[7.0], 1.0).unwrap();
        assert!(picked.straight_bottom);
        let got = layer(&picked.severity, 0.0, 10.0);
        assert!((got - 7.0).abs() < 1e-9, "{got}");
        assert!(picked.severity.probs()[0] == 0.0);
    }
}
