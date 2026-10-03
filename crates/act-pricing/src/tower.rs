//! Tower matching: one collective model (a claim count above the lowest
//! attachment point and a piecewise Pareto severity) that reproduces the
//! expected loss of every layer in a reinsurance tower.
//!
//! This is Riegel's Matching Algorithm 2 (Riegel 2018, "Matching tower
//! information with piecewise Pareto", European Actuarial Journal 8),
//! implemented from the paper as restated in `docs/design/pareto.md`.

use act_core::{Error, Result};
use act_math::linalg::solve;
use act_math::optimize::minimize;
use act_math::roots::bisect;
use act_prob::{PiecewisePareto, Severity, Truncation};
use microlp::{ComparisonOp, LinearExpr, OptimizationDirection, Problem};

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

/// The model through the points of a PML curve: `amounts[j]` is exceeded
/// once in `return_periods[j]` years, so the excess frequency at
/// `amounts[j]` is `1 / return_periods[j]`.
///
/// The severity has a threshold at every amount, the alpha between two
/// consecutive points (`ln(f_j / f_{j+1}) / ln(A_{j+1} / A_j)`), and
/// `tail_alpha` above the largest amount, optionally truncated at
/// `truncation` (the last piece only, which leaves every point of the
/// curve in place). Larger amounts must have longer return periods.
///
/// ```
/// use act_pricing::tower::fit_pml_curve;
///
/// let model = fit_pml_curve(&[10.0, 40.0, 100.0], &[1e6, 2e6, 3e6], 2.0, None).unwrap();
/// assert!((model.excess_frequency(2e6) - 1.0 / 40.0).abs() < 1e-15);
/// assert!((model.severity.alphas()[0] - 2.0).abs() < 1e-14); // 4 = 2^α
/// ```
pub fn fit_pml_curve(
    return_periods: &[f64],
    amounts: &[f64],
    tail_alpha: f64,
    truncation: Option<f64>,
) -> Result<TowerModel> {
    if amounts.is_empty() || return_periods.len() != amounts.len() {
        return Err(invalid(
            "return_periods",
            return_periods.len() as f64,
            "must have one return period per amount, and at least one",
        ));
    }
    if !tail_alpha.is_finite() || tail_alpha <= 0.0 {
        return Err(invalid(
            "tail_alpha",
            tail_alpha,
            "must be finite and positive",
        ));
    }
    let mut points: Vec<(f64, f64)> = amounts
        .iter()
        .copied()
        .zip(return_periods.iter().copied())
        .collect();
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (j, &(amount, period)) in points.iter().enumerate() {
        if !period.is_finite() || period <= 0.0 {
            return Err(invalid(
                "return_periods",
                period,
                "must be finite and positive",
            ));
        }
        if j > 0 && (amount == points[j - 1].0 || period <= points[j - 1].1) {
            return Err(invalid(
                "return_periods",
                period,
                "must increase strictly with the amount",
            ));
        }
    }
    let thresholds: Vec<f64> = points.iter().map(|p| p.0).collect();
    let mut alphas: Vec<f64> = points
        .windows(2)
        .map(|w| (w[1].1 / w[0].1).ln() / (w[1].0 / w[0].0).ln())
        .collect();
    alphas.push(tail_alpha);
    let severity = PiecewisePareto::new(thresholds, alphas)?;
    let severity = match truncation {
        None => severity,
        Some(tr) => severity.truncated(tr, Truncation::LastPiece)?,
    };
    Ok(TowerModel {
        frequency: 1.0 / points[0].1,
        severity,
    })
}

/// One piece of reference information for [`fit_references`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reference {
    /// The expected loss a year to a layer.
    Layer { layer: XsLayer, expected_loss: f64 },
    /// The expected number of losses a year above a threshold.
    Frequency { threshold: f64, frequency: f64 },
}

/// A model that reproduces every reference: expected layer losses (the
/// layers may overlap or leave gaps) and excess frequencies.
///
/// The references pin down the excess-loss function `u(x)` (the expected
/// loss to `∞ xs x`) and its slope `−g(x)` (the excess frequency) only at
/// some points. The fit completes both at every reference point: layer
/// losses are differences of `u`, frequencies are values of `g`, and
/// between consecutive points the risk rate on line must lie strictly
/// between the frequencies at its ends, which is what [`match_tower`]
/// needs. Above the highest point the alpha is `default_alpha`, unless a
/// reference to an unlimited layer determines the tail.
///
/// 1. A linear program (`microlp`) finds a completion with a positive
///    relative margin in every inequality; bisection on the margin finds
///    the largest, and half of it is used.
/// 2. That point is projected exactly onto the reference equalities, then
///    moved to the analytic center of the inequalities (maximizing the sum
///    of their logarithms, by Newton's method), so free frequencies and
///    rates on line sit well inside their ranges rather than at a vertex.
///    A frequency at the lowest point that no reference gives is unbounded
///    above, so it is derived as in step 1 of the algorithm instead.
/// 3. The completed tower is matched with `rule`.
///
/// Fails if the references are inconsistent: no completion has a
/// decreasing, convex excess-loss function.
///
/// ```
/// use act_pricing::layer::XsLayer;
/// use act_pricing::tower::{Reference, SelectionRule, fit_references};
///
/// let refs = [
///     Reference::Layer { layer: XsLayer::new(1000.0, 1000.0).unwrap(), expected_loss: 150.0 },
///     Reference::Layer { layer: XsLayer::new(3000.0, 1500.0).unwrap(), expected_loss: 160.0 },
///     Reference::Frequency { threshold: 1000.0, frequency: 0.3 },
/// ];
/// let model = fit_references(&refs, 2.0, SelectionRule::default()).unwrap();
/// assert!((model.layer_loss(1000.0, 1000.0) / 150.0 - 1.0).abs() < 1e-11);
/// assert!((model.layer_loss(3000.0, 1500.0) / 160.0 - 1.0).abs() < 1e-11);
/// assert!((model.excess_frequency(1000.0) / 0.3 - 1.0).abs() < 1e-11);
/// ```
pub fn fit_references(
    references: &[Reference],
    default_alpha: f64,
    rule: SelectionRule,
) -> Result<TowerModel> {
    if references.is_empty() {
        return Err(invalid("references", 0.0, "need at least one reference"));
    }
    if !default_alpha.is_finite() || default_alpha <= 1.0 {
        return Err(invalid(
            "default_alpha",
            default_alpha,
            "must be finite and above 1",
        ));
    }
    let mut points = Vec::new();
    let mut unlimited = false;
    for r in references {
        match *r {
            Reference::Layer {
                layer,
                expected_loss,
            } => {
                if !expected_loss.is_finite() || expected_loss <= 0.0 {
                    return Err(invalid(
                        "expected_loss",
                        expected_loss,
                        "must be finite and positive",
                    ));
                }
                points.push(layer.attachment());
                if layer.top().is_finite() {
                    points.push(layer.top());
                } else {
                    unlimited = true;
                }
            }
            Reference::Frequency {
                threshold,
                frequency,
            } => {
                if !threshold.is_finite() || threshold <= 0.0 {
                    return Err(invalid(
                        "threshold",
                        threshold,
                        "must be finite and positive",
                    ));
                }
                if !frequency.is_finite() || frequency <= 0.0 {
                    return Err(invalid(
                        "frequency",
                        frequency,
                        "must be finite and positive",
                    ));
                }
                points.push(threshold);
            }
        }
    }
    points.sort_by(f64::total_cmp);
    points.dedup();
    let completion = Completion {
        points: &points,
        references,
        default_alpha,
        unlimited,
    };
    // Bisection on the relative margin: feasibility shrinks as it grows.
    // Below about 1e-6 the margin drowns in the solver's tolerance.
    let tiny = 1e-6;
    let Some(mut best) = completion.solve(tiny) else {
        return Err(invalid(
            "references",
            f64::NAN,
            "are inconsistent: no decreasing, convex excess-loss function meets them",
        ));
    };
    let (mut lo, mut hi) = (tiny, 1.0);
    if completion.solve(hi).is_some() {
        lo = hi;
    } else {
        for _ in 0..40 {
            let mid = (lo * hi).sqrt();
            if completion.solve(mid).is_some() {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi / lo < 1.01 {
                break;
            }
        }
    }
    if let Some(sol) = completion.solve(0.5 * lo) {
        best = sol;
    }
    let x = completion.center(best).ok_or_else(|| {
        invalid(
            "references",
            f64::NAN,
            "could not be completed to a strictly consistent tower",
        )
    })?;
    let m = points.len();
    let scale = points[0];
    let losses: Vec<f64> = (0..m)
        .map(|i| (x[i] - if i + 1 < m { x[i + 1] } else { 0.0 }) * scale)
        .collect();
    let mut freq: Vec<Option<f64>> = x[m..].iter().map(|&g| Some(g)).collect();
    if m > 1 && !completion.lowest_frequency_given() {
        // Unbounded above by the references: derived from the alpha
        // between the two lowest layers, as in step 1 of the algorithm.
        freq[0] = None;
    }
    match_tower(&points, &losses, &freq, rule)
}

/// The linear program behind [`fit_references`]: `u` and `g` at every
/// point, scaled by the lowest point so the variables are of the order of
/// frequencies.
struct Completion<'a> {
    points: &'a [f64],
    references: &'a [Reference],
    default_alpha: f64,
    unlimited: bool,
}

impl Completion<'_> {
    /// `x = (u / scale, g)` at the points with every inequality holding
    /// with relative margin `eps`, or `None` if there is none.
    fn solve(&self, eps: f64) -> Option<Vec<f64>> {
        let scale = self.points[0];
        let p: Vec<f64> = self.points.iter().map(|x| x / scale).collect();
        let m = p.len();
        let mut lp = Problem::new(OptimizationDirection::Minimize);
        let u: Vec<_> = (0..m)
            .map(|_| lp.add_var(0.0, (0.0, f64::INFINITY)))
            .collect();
        let g: Vec<_> = (0..m)
            .map(|_| lp.add_var(0.0, (0.0, f64::INFINITY)))
            .collect();
        let index = |x: f64| {
            self.points
                .iter()
                .position(|&q| q == x)
                .expect("a reference point")
        };
        for r in self.references {
            match *r {
                Reference::Layer {
                    layer,
                    expected_loss,
                } => {
                    let mut e = LinearExpr::empty();
                    e.add(u[index(layer.attachment())], 1.0);
                    if layer.top().is_finite() {
                        e.add(u[index(layer.top())], -1.0);
                    }
                    lp.add_constraint(e, ComparisonOp::Eq, expected_loss / scale);
                }
                Reference::Frequency {
                    threshold,
                    frequency,
                } => {
                    lp.add_constraint([(g[index(threshold)], 1.0)], ComparisonOp::Eq, frequency);
                }
            }
        }
        let k = 1.0 + eps;
        for i in 0..m - 1 {
            // RRoL_i = (u_i − u_{i+1}) / Δ_i: g_i ≥ k RRoL_i ≥ k² g_{i+1}.
            let d = p[i + 1] - p[i];
            lp.add_constraint(
                [(g[i], 1.0), (u[i], -k / d), (u[i + 1], k / d)],
                ComparisonOp::Ge,
                0.0,
            );
            lp.add_constraint(
                [(u[i], 1.0 / d), (u[i + 1], -1.0 / d), (g[i + 1], -k)],
                ComparisonOp::Ge,
                0.0,
            );
        }
        // The unlimited top layer: α_top − 1 = g_m p_m / u_m. Its frequency
        // keeps the same relative margin above 0 as the others keep from
        // each other: g_m ≥ eps RRoL_{m−1}.
        let (gm, um, pm) = (g[m - 1], u[m - 1], p[m - 1]);
        if m > 1 {
            let d = p[m - 1] - p[m - 2];
            lp.add_constraint(
                [(gm, 1.0), (u[m - 2], -eps / d), (um, eps / d)],
                ComparisonOp::Ge,
                0.0,
            );
        }
        if self.unlimited {
            lp.add_constraint([(gm, pm), (um, -eps)], ComparisonOp::Ge, 0.0);
            lp.add_constraint([(um, 1.0), (gm, -eps * pm)], ComparisonOp::Ge, 0.0);
        } else {
            lp.add_constraint(
                [(um, self.default_alpha - 1.0), (gm, -pm)],
                ComparisonOp::Eq,
                0.0,
            );
        }
        let solution = lp.solve().ok()?.into_solution().ok()?;
        let x: Vec<f64> = u.iter().chain(&g).map(|&v| solution.var_value(v)).collect();
        // The solver meets constraints only to its tolerance: check the
        // strict inequalities in floating point.
        self.inequalities(true)
            .iter()
            .all(|a| dot(a, &x) > 0.0)
            .then_some(x)
    }

    /// Whether a reference gives the frequency at the lowest point.
    fn lowest_frequency_given(&self) -> bool {
        self.references.iter().any(
            |r| matches!(*r, Reference::Frequency { threshold, .. } if threshold == self.points[0]),
        )
    }

    /// The equalities `A x = b` on `x = (u / scale, g)`.
    fn equalities(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let m = self.points.len();
        let scale = self.points[0];
        let index = |x: f64| {
            self.points
                .iter()
                .position(|&q| q == x)
                .expect("a reference point")
        };
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for r in self.references {
            let mut row = vec![0.0; 2 * m];
            match *r {
                Reference::Layer {
                    layer,
                    expected_loss,
                } => {
                    row[index(layer.attachment())] = 1.0;
                    if layer.top().is_finite() {
                        row[index(layer.top())] = -1.0;
                    }
                    b.push(expected_loss / scale);
                }
                Reference::Frequency {
                    threshold,
                    frequency,
                } => {
                    row[m + index(threshold)] = 1.0;
                    b.push(frequency);
                }
            }
            a.push(row);
        }
        if !self.unlimited {
            let mut row = vec![0.0; 2 * m];
            row[m - 1] = self.default_alpha - 1.0;
            row[2 * m - 1] = -self.points[m - 1] / scale;
            a.push(row);
            b.push(0.0);
        }
        (a, b)
    }

    /// The strict inequalities `a · x > 0`: frequencies and risk rates on
    /// line interleave, decreasing up the tower, and the top frequency
    /// and unlimited loss are positive. With `with_lowest` false, the
    /// lowest frequency's bound is left out.
    fn inequalities(&self, with_lowest: bool) -> Vec<Vec<f64>> {
        let m = self.points.len();
        let scale = self.points[0];
        let mut out = Vec::new();
        for i in 0..m - 1 {
            let d = (self.points[i + 1] - self.points[i]) / scale;
            // g_i − RRoL_i and RRoL_i − g_{i+1}.
            let mut above = vec![0.0; 2 * m];
            above[m + i] = 1.0;
            above[i] = -1.0 / d;
            above[i + 1] = 1.0 / d;
            if i > 0 || with_lowest {
                out.push(above);
            }
            let mut below = vec![0.0; 2 * m];
            below[i] = 1.0 / d;
            below[i + 1] = -1.0 / d;
            below[m + i + 1] = -1.0;
            out.push(below);
        }
        let mut top = vec![0.0; 2 * m];
        top[2 * m - 1] = 1.0;
        out.push(top);
        let mut unlimited = vec![0.0; 2 * m];
        unlimited[m - 1] = 1.0;
        out.push(unlimited);
        out
    }

    /// The analytic center of the strict inequalities within the
    /// equalities, from a strictly feasible `x0`: the point maximizing
    /// `Σ ln(a_j · x)`, which keeps every free frequency and rate on line
    /// well inside its range instead of at an end. Newton's method on the
    /// null space of the equalities, after projecting `x0` onto them
    /// exactly. With no reference at the lowest frequency it is held
    /// fixed (it is unbounded above, and derived later).
    fn center(&self, x0: Vec<f64>) -> Option<Vec<f64>> {
        let m = self.points.len();
        let (mut a, mut b) = self.equalities();
        let free_lowest = m > 1 && !self.lowest_frequency_given();
        if free_lowest {
            let mut row = vec![0.0; 2 * m];
            row[m] = 1.0;
            a.push(row);
            b.push(x0[m]);
        }
        let (rows, rhs, null) = reduce(&a, &b)?;
        // Project onto A x = b: x0 − Aᵀ (A Aᵀ)⁻¹ (A x0 − b).
        let r = rows.len();
        let residual: Vec<f64> = (0..r).map(|i| dot(&rows[i], &x0) - rhs[i]).collect();
        let gram: Vec<f64> = (0..r * r)
            .map(|k| dot(&rows[k / r], &rows[k % r]))
            .collect();
        let y = solve(gram, residual, r)?;
        let mut x = x0;
        for (i, row) in rows.iter().enumerate() {
            for (xj, aj) in x.iter_mut().zip(row) {
                *xj -= y[i] * aj;
            }
        }
        let ineq = self.inequalities(!free_lowest);
        let phi = |x: &[f64]| -> Option<f64> {
            ineq.iter()
                .map(|a| {
                    let s = dot(a, x);
                    (s > 0.0).then(|| s.ln())
                })
                .sum()
        };
        let mut value = phi(&x)?;
        let k = null.len();
        for _ in 0..200 {
            if k == 0 {
                break;
            }
            // Gradient and (negated) Hessian of φ along the null space.
            let mut grad = vec![0.0; k];
            let mut hess = vec![0.0; k * k];
            for a in &ineq {
                let s = dot(a, &x);
                let an: Vec<f64> = null.iter().map(|n| dot(a, n) / s).collect();
                for i in 0..k {
                    grad[i] += an[i];
                    for j in 0..k {
                        hess[i * k + j] += an[i] * an[j];
                    }
                }
            }
            let step = solve(hess, grad.clone(), k)?;
            let decrement = dot(&grad, &step);
            if decrement < 1e-24 {
                break;
            }
            let mut t = 1.0;
            loop {
                let trial: Vec<f64> = (0..2 * m)
                    .map(|j| x[j] + t * (0..k).map(|i| step[i] * null[i][j]).sum::<f64>())
                    .collect();
                if let Some(v) = phi(&trial)
                    && v >= value + 0.25 * t * decrement
                {
                    x = trial;
                    value = v;
                    break;
                }
                t *= 0.5;
                if t < 1e-12 {
                    return Some(x);
                }
            }
        }
        Some(x)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Row-reduces `A x = b`: independent rows (in reduced form), their
/// right-hand sides, and a basis of the null space of `A`. `None` if the
/// system is inconsistent.
#[allow(clippy::type_complexity)]
fn reduce(a: &[Vec<f64>], b: &[f64]) -> Option<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>)> {
    let n = a.first().map_or(0, Vec::len);
    let mut rows: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(r, &v)| {
            let mut r = r.clone();
            r.push(v);
            r
        })
        .collect();
    let mut pivots = Vec::new();
    let mut rank = 0;
    for col in 0..n {
        let Some(p) =
            (rank..rows.len()).max_by(|&i, &j| rows[i][col].abs().total_cmp(&rows[j][col].abs()))
        else {
            break;
        };
        if rows[p][col].abs() < 1e-12 {
            continue;
        }
        rows.swap(rank, p);
        let lead = rows[rank][col];
        for v in rows[rank].iter_mut() {
            *v /= lead;
        }
        let pivot = rows[rank].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            if i != rank && row[col] != 0.0 {
                let f = row[col];
                for (v, p) in row.iter_mut().zip(&pivot) {
                    *v -= f * p;
                }
            }
        }
        pivots.push(col);
        rank += 1;
    }
    // A zero row with a non-zero right-hand side: inconsistent.
    let scale = b.iter().fold(1.0f64, |s, v| s.max(v.abs()));
    if rows[rank..].iter().any(|r| r[n].abs() > 1e-9 * scale) {
        return None;
    }
    rows.truncate(rank);
    let rhs = rows
        .iter_mut()
        .map(|r| r.pop().expect("augmented"))
        .collect();
    let null = (0..n)
        .filter(|c| !pivots.contains(c))
        .map(|free| {
            let mut v = vec![0.0; n];
            v[free] = 1.0;
            for (k, &p) in pivots.iter().enumerate() {
                v[p] = -rows[k][free];
            }
            v
        })
        .collect();
    Some((rows, rhs, null))
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
            bisect(self.a, self.b, |tau| self.lambda(tau, 0.0) < self.loss)
        };
        let tau_u = if single <= self.loss {
            self.b
        } else {
            bisect(self.a, self.b, |tau| {
                self.lambda(tau, self.alpha_cap(tau)) < self.loss
            })
        };
        (tau_l, tau_u)
    }

    /// The lower alpha that matches the layer's loss at `τ`.
    fn alpha_at(&self, tau: f64) -> f64 {
        // λ falls in α on [0, cap].
        let cap = self.alpha_cap(tau);
        bisect(0.0, cap, |alpha| self.lambda(tau, alpha) > self.loss)
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
    fn pml_curve_points_are_reproduced() {
        let periods = [5.0, 20.0, 50.0, 200.0];
        let amounts = [2e6, 5e6, 9e6, 2e7];
        let model = fit_pml_curve(&periods, &amounts, 1.8, None).unwrap();
        for (rp, x) in periods.iter().zip(amounts) {
            assert!((model.excess_frequency(x) * rp - 1.0).abs() < 1e-13);
        }
        assert_eq!(model.severity.alphas()[3], 1.8);
        // Unsorted input, and a truncated last piece keeps the points.
        let t = fit_pml_curve(
            &[50.0, 5.0, 200.0, 20.0],
            &[9e6, 2e6, 2e7, 5e6],
            1.8,
            Some(1e8),
        )
        .unwrap();
        for (rp, x) in periods.iter().zip(amounts) {
            assert!((t.excess_frequency(x) * rp - 1.0).abs() < 1e-13);
        }
        assert_eq!(t.excess_frequency(1e8), 0.0);
        assert!(fit_pml_curve(&[5.0, 4.0], &[1.0, 2.0], 2.0, None).is_err());
        assert!(fit_pml_curve(&[5.0], &[1.0], 0.0, None).is_err());
    }

    fn layer_ref(limit: f64, attachment: f64, loss: f64) -> Reference {
        Reference::Layer {
            layer: XsLayer::new(limit, attachment).unwrap(),
            expected_loss: loss,
        }
    }

    fn reproduces(model: &TowerModel, refs: &[Reference]) {
        for r in refs {
            let (got, want) = match *r {
                Reference::Layer {
                    layer,
                    expected_loss,
                } => (
                    model.layer_loss(layer.limit(), layer.attachment()),
                    expected_loss,
                ),
                Reference::Frequency {
                    threshold,
                    frequency,
                } => (model.excess_frequency(threshold), frequency),
            };
            assert!((got / want - 1.0).abs() < 1e-11, "{r:?}: {got}");
        }
    }

    #[test]
    fn complete_references_are_the_tower() {
        // Every layer of Table 3 and every frequency of a matched model:
        // nothing is left to choose.
        let base = example_4(SelectionRule::MinimizeAlphaRatio);
        let mut refs: Vec<Reference> = (0..7)
            .map(|i| {
                let limit = if i < 6 {
                    ATTACHMENTS[i + 1] - ATTACHMENTS[i]
                } else {
                    f64::INFINITY
                };
                layer_ref(limit, ATTACHMENTS[i], LOSSES[i])
            })
            .collect();
        refs.extend(ATTACHMENTS.iter().map(|&a| Reference::Frequency {
            threshold: a,
            frequency: base.excess_frequency(a),
        }));
        let fit = fit_references(&refs, 2.0, SelectionRule::MinimizeAlphaRatio).unwrap();
        reproduces(&fit, &refs);
        for (x, y) in fit.severity.alphas().iter().zip(base.severity.alphas()) {
            assert!((x / y - 1.0).abs() < 1e-6, "{x} {y}");
        }
    }

    #[test]
    fn partial_references_are_reproduced() {
        // Overlapping layers, a gap, a frequency, no unlimited layer.
        let refs = [
            layer_ref(1000.0, 1000.0, 120.0),
            layer_ref(1500.0, 1500.0, 110.0),
            layer_ref(5000.0, 5000.0, 60.0),
            Reference::Frequency {
                threshold: 2500.0,
                frequency: 0.05,
            },
        ];
        for rule in [SelectionRule::MinimizeAlphaRatio, SelectionRule::Midpoint] {
            let fit = fit_references(&refs, 2.0, rule).unwrap();
            reproduces(&fit, &refs);
            // The default alpha above the highest point.
            assert!((fit.severity.alphas().last().unwrap() - 2.0).abs() < 1e-9);
        }
        // Centred, the gaps get moderate alphas (a vertex of the linear
        // program gives alphas near 100 here).
        let fit = fit_references(&refs, 2.0, SelectionRule::default()).unwrap();
        assert!(
            fit.severity.alphas().iter().all(|&a| a < 10.0),
            "{:?}",
            fit.severity.alphas()
        );
        // A single unlimited layer with its frequency.
        let refs = [
            layer_ref(f64::INFINITY, 1000.0, 500.0),
            Reference::Frequency {
                threshold: 1000.0,
                frequency: 0.25,
            },
        ];
        let fit = fit_references(&refs, 2.0, SelectionRule::default()).unwrap();
        reproduces(&fit, &refs);
        assert!((fit.severity.alphas()[0] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn inconsistent_references_fail() {
        // The upper layer has the higher rate on line.
        let refs = [
            layer_ref(1000.0, 1000.0, 100.0),
            layer_ref(1000.0, 2000.0, 120.0),
        ];
        assert!(fit_references(&refs, 2.0, SelectionRule::default()).is_err());
        // A frequency below the rate on line of the layer above it.
        let refs = [
            layer_ref(1000.0, 1000.0, 100.0),
            Reference::Frequency {
                threshold: 1000.0,
                frequency: 0.09,
            },
        ];
        assert!(fit_references(&refs, 2.0, SelectionRule::default()).is_err());
        assert!(fit_references(&[], 2.0, SelectionRule::default()).is_err());
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
