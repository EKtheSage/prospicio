//! Tower matching: one collective model (a claim count above the lowest
//! attachment point and a piecewise Pareto severity) that reproduces the
//! expected loss of every layer in a reinsurance tower.
//!
//! This is Riegel's Matching Algorithm 2 (Riegel 2018, "Matching tower
//! information with piecewise Pareto", European Actuarial Journal 8),
//! implemented from the paper as restated in `docs/design/pareto.md`.

use act_core::{Error, Result};
use act_prob::{PiecewisePareto, Severity};

use crate::layer::{XsLayer, alpha_between_layers};

/// How the free threshold inside each limited layer is chosen. Every
/// choice in the feasible range reproduces the tower exactly; the rules
/// differ in the shape between the attachment points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionRule {
    /// Make the two alphas of each layer as close as possible: minimize
    /// `max(α_lower / α_upper, α_upper / α_lower)`. The paper's example
    /// and the R package Pareto use this rule.
    #[default]
    MinimizeAlphaRatio,
    /// The midpoint of the feasible range: cheaper, and deterministic
    /// without a search.
    Midpoint,
}

/// The matched model: `frequency` losses a year above the lowest
/// attachment point, each with the piecewise Pareto `severity`.
#[derive(Debug, Clone, PartialEq)]
pub struct TowerModel {
    pub frequency: f64,
    pub severity: PiecewisePareto,
}

impl TowerModel {
    /// Expected losses a year above `x`.
    pub fn excess_frequency(&self, x: f64) -> f64 {
        use act_prob::Distribution;
        self.frequency * self.severity.survival(x)
    }

    /// Expected loss a year to `limit` xs `attachment`.
    pub fn layer_loss(&self, limit: f64, attachment: f64) -> f64 {
        self.frequency * self.severity.layer(limit, attachment)
    }
}

/// Layer losses `e_i = u_i − u_{i+1}` (and `e_k = u_k`) from the expected
/// losses `u_i` of the unlimited layers `∞ xs a_i`, which must decrease
/// strictly.
pub fn layer_losses_from_unlimited(unlimited: &[f64]) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(unlimited.len());
    for (i, &u) in unlimited.iter().enumerate() {
        let next = unlimited.get(i + 1).copied().unwrap_or(0.0);
        if !u.is_finite() || u <= next {
            return Err(invalid(
                "unlimited losses",
                u,
                "must be finite and strictly decreasing to a positive last value",
            ));
        }
        out.push(u - next);
    }
    Ok(out)
}

/// Matches the tower of contiguous layers `a_{i+1} − a_i xs a_i`, the
/// last one unlimited (`∞ xs a_k`), with expected losses `layer_losses`.
///
/// `frequencies` gives the expected number of losses a year above each
/// attachment point: empty to derive them all, or one entry per
/// attachment point with `None` where it is to be derived. A derived
/// frequency comes from the Pareto alpha between the layer and its
/// neighbour (step 1 of the algorithm). With a single (unlimited) layer the
/// frequency must be given.
///
/// The severity has thresholds at every attachment point and one more
/// inside each limited layer, except where a single Pareto piece already
/// matches the layer.
///
/// Fails, before any search, unless the risk rates on line `e_i / c_i` of
/// the limited layers decrease strictly and every frequency lies strictly
/// between the rates on line of the layers around its attachment point.
///
/// ```
/// use act_pricing::tower::{SelectionRule, match_tower};
///
/// let model = match_tower(
///     &[1000.0, 1500.0, 2000.0],
///     &[100.0, 90.0, 120.0],
///     &[Some(0.25), None, None],
///     SelectionRule::MinimizeAlphaRatio,
/// )
/// .unwrap();
/// assert!((model.layer_loss(500.0, 1500.0) - 90.0).abs() < 1e-9);
/// assert!((model.layer_loss(f64::INFINITY, 2000.0) - 120.0).abs() < 1e-9);
/// assert!((model.excess_frequency(1000.0) - 0.25).abs() < 1e-12);
/// ```
pub fn match_tower(
    attachments: &[f64],
    layer_losses: &[f64],
    frequencies: &[Option<f64>],
    rule: SelectionRule,
) -> Result<TowerModel> {
    let k = attachments.len();
    if k == 0 || layer_losses.len() != k {
        return Err(invalid(
            "layer_losses",
            layer_losses.len() as f64,
            "must have one expected loss per attachment point, and at least one",
        ));
    }
    if !frequencies.is_empty() && frequencies.len() != k {
        return Err(invalid(
            "frequencies",
            frequencies.len() as f64,
            "must be empty or have one entry per attachment point",
        ));
    }
    for (i, &a) in attachments.iter().enumerate() {
        if !a.is_finite() || a <= 0.0 || (i > 0 && a <= attachments[i - 1]) {
            return Err(invalid(
                "attachments",
                a,
                "must be finite, positive and strictly increasing",
            ));
        }
    }
    for &e in layer_losses {
        if !e.is_finite() || e <= 0.0 {
            return Err(invalid("layer_losses", e, "must be finite and positive"));
        }
    }
    let width = |i: usize| -> f64 {
        if i + 1 < k {
            attachments[i + 1] - attachments[i]
        } else {
            f64::INFINITY
        }
    };
    // Risk rates on line; 0 for the unlimited top layer.
    let rrol: Vec<f64> = (0..k).map(|i| layer_losses[i] / width(i)).collect();
    for i in 1..k.saturating_sub(1) {
        if rrol[i] >= rrol[i - 1] {
            return Err(invalid(
                "layer_losses",
                layer_losses[i],
                "risk rates on line must decrease strictly up the tower",
            ));
        }
    }

    let freq = frequencies_for(attachments, layer_losses, frequencies, &width)?;
    for i in 0..k {
        let above = if i == 0 { f64::INFINITY } else { rrol[i - 1] };
        if !(freq[i] < above && freq[i] > rrol[i]) {
            return Err(invalid(
                "frequencies",
                freq[i],
                "must lie strictly between the risk rates on line of the layers around the attachment point",
            ));
        }
    }

    let f1 = freq[0];
    let mut thresholds = Vec::with_capacity(2 * k - 1);
    let mut alphas = Vec::with_capacity(2 * k - 1);
    for i in 0..k - 1 {
        let layer = LayerFit {
            a: attachments[i],
            b: attachments[i + 1],
            s_a: freq[i] / f1,
            s_b: freq[i + 1] / f1,
            loss: layer_losses[i] / f1,
        };
        match layer.fit(rule) {
            Fit::Single(alpha) => {
                thresholds.push(layer.a);
                alphas.push(alpha);
            }
            Fit::Two { tau, alpha, sigma } => {
                thresholds.extend([layer.a, tau]);
                alphas.extend([alpha, sigma]);
            }
        }
    }
    // The unlimited top layer: mean excess a_k / (α − 1) per loss above a_k.
    thresholds.push(attachments[k - 1]);
    alphas.push(freq[k - 1] * attachments[k - 1] / layer_losses[k - 1] + 1.0);
    Ok(TowerModel {
        frequency: f1,
        severity: PiecewisePareto::new(thresholds, alphas)?,
    })
}

/// Frequencies above each attachment point: given, or from the alpha
/// between neighbouring layers.
fn frequencies_for(
    attachments: &[f64],
    losses: &[f64],
    given: &[Option<f64>],
    width: &dyn Fn(usize) -> f64,
) -> Result<Vec<f64>> {
    let k = attachments.len();
    let layer = |i: usize| XsLayer::new(width(i), attachments[i]);
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        if let Some(Some(f)) = given.get(i) {
            out.push(*f);
            continue;
        }
        if k == 1 {
            return Err(invalid(
                "frequencies",
                f64::NAN,
                "a tower of one unlimited layer needs its frequency",
            ));
        }
        // Layer i with the alpha towards its neighbour (above for the
        // lowest layer, below otherwise).
        let (lo, hi) = if i == 0 { (0, 1) } else { (i - 1, i) };
        let alpha = alpha_between_layers((layer(lo)?, losses[lo]), (layer(hi)?, losses[hi]), None)?;
        out.push(losses[i] / piece_integral(attachments[i], attachments[i] + width(i), alpha));
    }
    Ok(out)
}

/// The pieces fitted to one limited layer.
enum Fit {
    /// The single Pareto through both end points already matches the
    /// layer's loss (as it does when the frequencies were derived from the
    /// alpha to the next layer): one piece.
    Single(f64),
    /// Pieces `alpha` on `[a, tau)` and `sigma` on `[tau, b)`.
    Two { tau: f64, alpha: f64, sigma: f64 },
}

/// One limited layer `[a, b)` to fit with two Pareto pieces, normalized to
/// one loss above the lowest attachment point: survival `s_a` at `a` and
/// `s_b` at `b`, expected loss `loss` per loss.
struct LayerFit {
    a: f64,
    b: f64,
    s_a: f64,
    s_b: f64,
    loss: f64,
}

impl LayerFit {
    /// `ln(s_b / s_a)`, negative.
    fn log_drop(&self) -> f64 {
        (self.s_b / self.s_a).ln()
    }

    /// The upper alpha `σ(τ, α)` that reaches `s_b` at `b`.
    fn sigma(&self, tau: f64, alpha: f64) -> f64 {
        (self.log_drop() - alpha * (self.a / tau).ln()) / (tau / self.b).ln()
    }

    /// The largest lower alpha at `τ`, where `σ = 0`.
    fn alpha_cap(&self, tau: f64) -> f64 {
        self.log_drop() / (self.a / tau).ln()
    }

    /// `λ(τ, α)`: expected loss per loss of the layer with pieces `α` on
    /// `[a, τ)` and `σ(τ, α)` on `[τ, b)`.
    fn lambda(&self, tau: f64, alpha: f64) -> f64 {
        let sigma = self.sigma(tau, alpha).max(0.0);
        self.s_a
            * (piece_integral(self.a, tau, alpha)
                + (alpha * (self.a / tau).ln()).exp() * piece_integral(tau, self.b, sigma))
    }

    /// The alpha of the single Pareto through both end points.
    fn alpha_single(&self) -> f64 {
        self.log_drop() / (self.a / self.b).ln()
    }

    /// `λ` of the single Pareto through both end points, the common limit
    /// of both bounding curves at the ends of `(a, b)`.
    fn lambda_single(&self) -> f64 {
        self.s_a * piece_integral(self.a, self.b, self.alpha_single())
    }

    /// The feasible range `(τ_l, τ_u)` for the free threshold.
    fn feasible(&self) -> (f64, f64) {
        let single = self.lambda_single();
        // λ(τ, 0) rises from `single` to s_a (b − a); λ(τ, cap) rises from
        // s_b (b − a) to `single`.
        let tau_l = if single >= self.loss {
            self.a
        } else {
            root_increasing(self.a, self.b, self.loss, |tau| self.lambda(tau, 0.0))
        };
        let tau_u = if single <= self.loss {
            self.b
        } else {
            root_increasing(self.a, self.b, self.loss, |tau| {
                self.lambda(tau, self.alpha_cap(tau))
            })
        };
        (tau_l, tau_u)
    }

    /// The lower alpha that matches the layer's loss at `τ`.
    fn alpha_at(&self, tau: f64) -> f64 {
        // λ falls in α on [0, cap]: bisection on its negative.
        let cap = self.alpha_cap(tau);
        root_increasing(0.0, cap, -self.loss, |alpha| -self.lambda(tau, alpha))
    }

    /// The pieces under `rule`.
    fn fit(&self, rule: SelectionRule) -> Fit {
        if (self.lambda_single() / self.loss - 1.0).abs() <= 1e-12 {
            return Fit::Single(self.alpha_single());
        }
        let (lo, hi) = self.feasible();
        let tau = match rule {
            SelectionRule::Midpoint => 0.5 * (lo + hi),
            SelectionRule::MinimizeAlphaRatio => {
                let spread = |tau: f64| {
                    let alpha = self.alpha_at(tau);
                    (alpha.ln() - self.sigma(tau, alpha).ln()).abs()
                };
                minimize(lo, hi, spread)
            }
        };
        let alpha = self.alpha_at(tau);
        Fit::Two {
            tau,
            alpha,
            sigma: self.sigma(tau, alpha).max(0.0),
        }
    }
}

/// `∫_a^b (a/x)^α dx` for `α ≥ 0` and `a < b ≤ ∞` (`b = ∞` needs
/// `α > 1`): `a L exprel((1 − α) L)` with `L = ln(b/a)`.
fn piece_integral(a: f64, b: f64, alpha: f64) -> f64 {
    if b == f64::INFINITY {
        return if alpha > 1.0 {
            a / (alpha - 1.0)
        } else {
            f64::INFINITY
        };
    }
    let l = (b / a).ln();
    let z = (1.0 - alpha) * l;
    let exprel = if z == 0.0 { 1.0 } else { z.exp_m1() / z };
    a * l * exprel
}

/// The `x` in `(lo, hi)` with `f(x) = target`, for `f` increasing, by
/// bisection to full precision. Never returns an end point.
fn root_increasing(lo: f64, hi: f64, target: f64, f: impl Fn(f64) -> f64) -> f64 {
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if f(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Minimizes `f` on the open interval `(lo, hi)`: a scan for the best
/// cell, then golden-section search inside it.
fn minimize(lo: f64, hi: f64, f: impl Fn(f64) -> f64) -> f64 {
    const CELLS: usize = 64;
    let h = (hi - lo) / CELLS as f64;
    let x = |j: usize| lo + h * j as f64;
    let best = (1..CELLS)
        .min_by(|&i, &j| f(x(i)).total_cmp(&f(x(j))))
        .expect("CELLS > 1");
    let (mut a, mut b) = (x(best - 1), x(best + 1));
    let g = 0.5 * (5f64.sqrt() - 1.0);
    let (mut c, mut d) = (b - g * (b - a), a + g * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..200 {
        if b - a <= 1e-15 * b.abs() {
            break;
        }
        if fc < fd {
            (b, d, fd) = (d, c, fc);
            c = b - g * (b - a);
            fc = f(c);
        } else {
            (a, c, fc) = (c, d, fd);
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
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
    use act_prob::Distribution;

    /// Riegel (2018), Table 3.
    const ATTACHMENTS: [f64; 7] = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 5000.0, 10_000.0];
    const LOSSES: [f64; 7] = [100.0, 90.0, 50.0, 40.0, 100.0, 50.0, 50.0];

    fn example_4(rule: SelectionRule) -> TowerModel {
        let mut freq = vec![None; 7];
        freq[0] = Some(0.25);
        match_tower(&ATTACHMENTS, &LOSSES, &freq, rule).unwrap()
    }

    fn reproduces_the_tower(model: &TowerModel, frequencies: &[f64]) {
        for i in 0..7 {
            let limit = if i < 6 {
                ATTACHMENTS[i + 1] - ATTACHMENTS[i]
            } else {
                f64::INFINITY
            };
            let got = model.layer_loss(limit, ATTACHMENTS[i]);
            assert!((got / LOSSES[i] - 1.0).abs() < 1e-10, "layer {i}: {got}");
            if let Some(&f) = frequencies.get(i) {
                let got = model.excess_frequency(ATTACHMENTS[i]);
                assert!((got / f - 1.0).abs() < 1e-12, "frequency {i}: {got}");
            }
        }
    }

    #[test]
    fn example_4_matches_the_paper() {
        let model = example_4(SelectionRule::MinimizeAlphaRatio);
        let t = model.severity.thresholds();
        let alpha = model.severity.alphas();
        assert_eq!(t.len(), 13);
        // Table 4's first six pieces, to the printed digits.
        let want_t = [1000.0, 1097.0, 1500.0, 1932.0, 2000.0, 2148.0];
        let want_alpha = [2.374, 0.199, 0.175, 9.685, 3.539, 0.817];
        let want_g = [0.250, 0.201, 0.189, 0.180, 0.129, 0.100];
        for j in 0..6 {
            assert!((t[j] - want_t[j]).abs() <= 0.5, "t[{j}] = {}", t[j]);
            assert!(
                (alpha[j] - want_alpha[j]).abs() <= 0.0005,
                "alpha[{j}] = {}",
                alpha[j]
            );
            let g = model.excess_frequency(t[j]);
            assert!((g - want_g[j]).abs() <= 0.0005, "g[{j}] = {g}");
        }
        reproduces_the_tower(&model, &[0.25]);
        assert!(model.frequency == 0.25 && model.severity.mean().is_finite());
    }

    #[test]
    fn both_rules_reproduce_given_frequencies() {
        let freq = [0.3, 0.19, 0.15, 0.09, 0.06, 0.02, 0.008];
        let given: Vec<Option<f64>> = freq.iter().map(|&f| Some(f)).collect();
        for rule in [SelectionRule::MinimizeAlphaRatio, SelectionRule::Midpoint] {
            let model = match_tower(&ATTACHMENTS, &LOSSES, &given, rule).unwrap();
            reproduces_the_tower(&model, &freq);
            assert!(model.severity.alphas().iter().all(|&a| a > 0.0));
        }
        // Derived frequencies everywhere.
        let model = match_tower(&ATTACHMENTS, &LOSSES, &[], SelectionRule::Midpoint).unwrap();
        reproduces_the_tower(&model, &[]);
    }

    #[test]
    fn minimize_rule_balances_the_alphas() {
        let min = example_4(SelectionRule::MinimizeAlphaRatio);
        let mid = example_4(SelectionRule::Midpoint);
        let spread = |m: &TowerModel, i: usize| {
            let a = m.severity.alphas();
            (a[2 * i].ln() - a[2 * i + 1].ln()).abs()
        };
        for i in 0..6 {
            assert!(spread(&min, i) <= spread(&mid, i) + 1e-12, "layer {i}");
        }
    }

    #[test]
    fn property_tower_is_scale_invariant_and_minimal() {
        // 5m xs 5m, 15m xs 10m, Inf xs 25m; one loss a year above 5m.
        let a = [5e6, 1e7, 2.5e7];
        let e = [2.4e6, 1.5e6, 1.2e6];
        let freq = [Some(1.0), None, None];
        let rule = SelectionRule::MinimizeAlphaRatio;
        let big = match_tower(&a, &e, &freq, rule).unwrap();
        for i in 0..3 {
            let limit = if i < 2 {
                a[i + 1] - a[i]
            } else {
                f64::INFINITY
            };
            assert!((big.layer_loss(limit, a[i]) / e[i] - 1.0).abs() < 1e-10);
        }
        let small =
            match_tower(&a.map(|x| x / 5000.0), &e.map(|x| x / 5000.0), &freq, rule).unwrap();
        for (x, y) in big.severity.alphas().iter().zip(small.severity.alphas()) {
            assert!((x / y - 1.0).abs() < 1e-8, "{x} {y}");
        }
        // The middle layer's threshold minimizes the alpha spread.
        let layer = LayerFit {
            a: 2000.0,
            b: 5000.0,
            s_a: small.excess_frequency(2000.0),
            s_b: small.excess_frequency(5000.0),
            loss: 300.0,
        };
        let spread = |tau: f64| {
            let alpha = layer.alpha_at(tau);
            (alpha / layer.sigma(tau, alpha)).ln().abs()
        };
        let tau = small.severity.thresholds()[3];
        for step in [1e-3, 1e-2, 0.1] {
            assert!(spread(tau) <= spread(tau * (1.0 + step)));
            assert!(spread(tau) <= spread(tau * (1.0 - step)));
        }
        assert!(spread(tau) < 0.0646, "{}", spread(tau));
    }

    #[test]
    fn single_unlimited_layer() {
        // 0.5 losses above 1000 a year expecting 1000: mean excess 2000,
        // so alpha = 1000 / 2000 + 1.
        let model =
            match_tower(&[1000.0], &[1000.0], &[Some(0.5)], SelectionRule::default()).unwrap();
        assert_eq!(model.severity.alphas(), &[1.5]);
        assert!(match_tower(&[1000.0], &[1000.0], &[], SelectionRule::default()).is_err());
    }

    #[test]
    fn rejects_inconsistent_towers() {
        let rule = SelectionRule::default();
        // Risk rate on line rising from layer 2 to layer 3.
        let bad = [100.0, 40.0, 50.0, 40.0, 100.0, 50.0, 50.0];
        assert!(match_tower(&ATTACHMENTS, &bad, &[], rule).is_err());
        // f_1 below the first layer's rate on line, 100 / 500.
        let mut freq = vec![None; 7];
        freq[0] = Some(0.15);
        assert!(match_tower(&ATTACHMENTS, &LOSSES, &freq, rule).is_err());
        // A frequency above the rate on line of the layer below.
        freq[0] = None;
        freq[1] = Some(0.21);
        assert!(match_tower(&ATTACHMENTS, &LOSSES, &freq, rule).is_err());
        assert!(match_tower(&ATTACHMENTS, &LOSSES[..6], &[], rule).is_err());
        assert!(match_tower(&[2.0, 1.0], &[1.0, 1.0], &[], rule).is_err());
    }

    #[test]
    fn unlimited_losses_convert() {
        let u = [480.0, 380.0, 290.0];
        assert_eq!(
            layer_losses_from_unlimited(&u).unwrap(),
            vec![100.0, 90.0, 290.0]
        );
        assert!(layer_losses_from_unlimited(&[1.0, 2.0]).is_err());
    }
}
