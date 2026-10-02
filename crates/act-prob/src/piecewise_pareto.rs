//! Piecewise Pareto: a different Pareto alpha above each threshold, the
//! output of tower matching and the general large-loss model in
//! `docs/design/pareto.md`.

use act_core::Result;

use crate::distribution::{Distribution, check_probability};
use crate::pareto::{invalid, power_integral, raw_integral};
use crate::severity::Severity;

/// How a [`PiecewisePareto`] is truncated at a point `T` above its last
/// threshold. Both match the R package Pareto's `truncation_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Truncation {
    /// The last piece is a truncated Pareto between `t_n` and `T`; the
    /// pieces below are unchanged (`"lp"` in R).
    LastPiece,
    /// The whole distribution is conditioned on `X < T` (`"wd"` in R).
    WholeDistribution,
}

/// Piecewise Pareto distribution with thresholds `t_1 < … < t_n` and
/// alphas `α_1, …, α_n`: `P(X > x) = 1` below `t_1`, and on
/// `[t_k, t_{k+1})`
///
/// ```text
/// P(X > x) = P(X > t_k) (t_k / x)^α_k,
/// ```
///
/// so `α_k` is the local Pareto alpha on the `k`-th piece. Interior
/// alphas may be 0 (no losses end in that piece); the last must be
/// positive. Matches `pPiecewisePareto(x, t, alpha)` in the R package
/// Pareto.
///
/// Survival at the thresholds is kept as a logarithm, so steep pieces far
/// in the tail do not underflow. Every layer moment is a sum of
/// single-piece Pareto integrals, in closed form.
///
/// ```
/// use act_prob::{Distribution, PiecewisePareto, Severity};
///
/// let pp = PiecewisePareto::new(vec![1000.0, 2000.0], vec![1.0, 2.0]).unwrap();
/// // P(X > 2000) = 1/2, then Pareto(2000, 2) above.
/// assert!((pp.survival(4000.0) - 0.125).abs() < 1e-15);
/// // E[X − 2000]+ = P(X > 2000) · 2000 / (2 − 1).
/// assert!((pp.stop_loss(2000.0) - 1000.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PiecewisePareto {
    t: Vec<f64>,
    alpha: Vec<f64>,
    /// `ln P(X > t_k)` for the untruncated distribution.
    log_s: Vec<f64>,
    truncation: Option<(f64, Truncation)>,
}

impl PiecewisePareto {
    /// Piecewise Pareto from strictly increasing positive thresholds `t`
    /// and alphas `alpha` of the same length: interior alphas `≥ 0`, the
    /// last `> 0`.
    pub fn new(t: Vec<f64>, alpha: Vec<f64>) -> Result<Self> {
        if t.is_empty() || t.len() != alpha.len() {
            return Err(invalid(
                "alpha",
                alpha.len() as f64,
                "must be non-empty and have one alpha per threshold",
            ));
        }
        for (i, &x) in t.iter().enumerate() {
            if !x.is_finite() || x <= 0.0 {
                return Err(invalid("t", x, "must be finite and positive"));
            }
            if i > 0 && x <= t[i - 1] {
                return Err(invalid("t", x, "must be strictly increasing"));
            }
        }
        for &a in &alpha {
            if !a.is_finite() || a < 0.0 {
                return Err(invalid("alpha", a, "must be finite and non-negative"));
            }
        }
        let last = alpha[alpha.len() - 1];
        if last <= 0.0 {
            return Err(invalid("alpha", last, "the last alpha must be positive"));
        }
        let mut log_s = Vec::with_capacity(t.len());
        let mut acc = 0.0;
        for k in 0..t.len() {
            if k > 0 {
                acc += alpha[k - 1] * (t[k - 1] / t[k]).ln();
            }
            log_s.push(acc);
        }
        Ok(Self {
            t,
            alpha,
            log_s,
            truncation: None,
        })
    }

    /// The same distribution truncated at `truncation`, which must be
    /// finite and above the last threshold (as in the R package).
    pub fn truncated(self, truncation: f64, kind: Truncation) -> Result<Self> {
        let last = self.t[self.t.len() - 1];
        if !truncation.is_finite() || truncation <= last {
            return Err(invalid(
                "truncation",
                truncation,
                "must be finite and above the last threshold",
            ));
        }
        Ok(Self {
            truncation: Some((truncation, kind)),
            ..self
        })
    }

    /// Thresholds `t_1 < … < t_n`.
    pub fn thresholds(&self) -> &[f64] {
        &self.t
    }

    /// Alphas `α_1, …, α_n`.
    pub fn alphas(&self) -> &[f64] {
        &self.alpha
    }

    /// Truncation point and kind, if any.
    pub fn truncation(&self) -> Option<(f64, Truncation)> {
        self.truncation
    }

    /// `P(X > x)`.
    pub fn survival(&self, x: f64) -> f64 {
        match self.truncation {
            Some((tr, Truncation::WholeDistribution)) => {
                if x >= tr {
                    return 0.0;
                }
                if x < self.t[0] {
                    return 1.0;
                }
                // S(x) − S(T) = S(x) (1 − S(T)/S(x)), without cancellation.
                let (k, k_tr) = (self.piece(x), self.piece(tr));
                let log_ratio = if k == k_tr {
                    self.alpha[k] * (x / tr).ln()
                } else {
                    self.log_survival(tr) - self.log_survival(x)
                };
                let (_, one_minus) = self.whole_mass(tr);
                self.log_survival(x).exp() * -log_ratio.exp_m1() / one_minus
            }
            _ => self.base_survival(x),
        }
    }

    /// Index of the piece containing `x ≥ t_1`.
    fn piece(&self, x: f64) -> usize {
        self.t.partition_point(|&t| t <= x) - 1
    }

    /// `T` if the last piece is truncated at `T`.
    fn last_piece_truncation(&self) -> Option<f64> {
        match self.truncation {
            Some((tr, Truncation::LastPiece)) => Some(tr),
            _ => None,
        }
    }

    /// `(q, 1 − q)` with `q = (t_n / T)^α_n`, the truncated last piece.
    fn last_piece_mass(&self, tr: f64) -> (f64, f64) {
        let n = self.t.len() - 1;
        let log_q = self.alpha[n] * (self.t[n] / tr).ln();
        (log_q.exp(), -log_q.exp_m1())
    }

    /// `ln P(X > x)` for the untruncated distribution and `x ≥ t_1`.
    fn log_survival(&self, x: f64) -> f64 {
        let k = self.piece(x);
        self.log_s[k] + self.alpha[k] * (self.t[k] / x).ln()
    }

    /// `(S(T), 1 − S(T))` for the untruncated distribution.
    fn whole_mass(&self, tr: f64) -> (f64, f64) {
        let log = self.log_survival(tr);
        (log.exp(), -log.exp_m1())
    }

    /// Survival with any last-piece truncation, before whole-distribution
    /// truncation.
    fn base_survival(&self, x: f64) -> f64 {
        if x < self.t[0] {
            return 1.0;
        }
        let k = self.piece(x);
        match self.last_piece_truncation() {
            Some(tr) if k == self.t.len() - 1 => {
                if x >= tr {
                    return 0.0;
                }
                // (t/x)^α − (t/T)^α = (t/x)^α (1 − (x/T)^α).
                let (_, one_minus_q) = self.last_piece_mass(tr);
                let rest = -(self.alpha[k] * (x / tr).ln()).exp_m1();
                self.log_s[k].exp() * (self.t[k] / x).powf(self.alpha[k]) * rest / one_minus_q
            }
            _ => self.log_survival(x).exp(),
        }
    }

    /// `∫_a^b x^k S(x) dx` for `k ∈ {0, 1}` and `0 ≤ a ≤ b ≤ ∞`.
    fn integral(&self, k: i32, a: f64, b: f64) -> f64 {
        match self.truncation {
            Some((tr, Truncation::WholeDistribution)) => {
                let (a, b) = (a.min(tr), b.min(tr));
                if a >= b {
                    return 0.0;
                }
                let (s_tr, one_minus) = self.whole_mass(tr);
                (self.base_integral(k, a, b) - s_tr * power_integral(k, a, b)) / one_minus
            }
            _ => self.base_integral(k, a, b),
        }
    }

    /// `∫_a^b x^k S(x) dx` with `S` the survival before whole-distribution
    /// truncation: one Pareto integral per piece.
    fn base_integral(&self, k: i32, a: f64, b: f64) -> f64 {
        if a >= b {
            return 0.0;
        }
        let n = self.t.len();
        let mut sum = if a < self.t[0] {
            power_integral(k, a, b.min(self.t[0]))
        } else {
            0.0
        };
        for j in 0..n {
            let lo = a.max(self.t[j]);
            let mut hi = if j + 1 < n { b.min(self.t[j + 1]) } else { b };
            let truncated = if j + 1 == n {
                self.last_piece_truncation()
            } else {
                None
            };
            if let Some(tr) = truncated {
                hi = hi.min(tr);
            }
            if lo >= hi {
                continue;
            }
            let s_j = self.log_s[j].exp();
            if s_j == 0.0 {
                break;
            }
            let piece = raw_integral(k, self.t[j], self.alpha[j], lo, hi);
            sum += match truncated {
                Some(tr) => {
                    let (q, one_minus_q) = self.last_piece_mass(tr);
                    s_j * (piece - q * power_integral(k, lo, hi)) / one_minus_q
                }
                None => s_j * piece,
            };
        }
        sum
    }

    /// The `x` with base survival `s`, for `0 < s ≤ 1`.
    fn base_inverse(&self, s: f64) -> f64 {
        let log = s.ln();
        // Last piece whose threshold survival is at least s.
        let k = self.log_s.partition_point(|&l| l >= log) - 1;
        match self.last_piece_truncation() {
            Some(tr) if k == self.t.len() - 1 => {
                let (q, one_minus_q) = self.last_piece_mass(tr);
                let rel = (log - self.log_s[k]).exp();
                (self.t[k] * (q + rel * one_minus_q).powf(-1.0 / self.alpha[k])).min(tr)
            }
            _ => self.t[k] * ((self.log_s[k] - log) / self.alpha[k]).exp(),
        }
    }
}

impl Distribution for PiecewisePareto {
    fn mean(&self) -> f64 {
        self.integral(0, 0.0, f64::INFINITY)
    }

    fn variance(&self) -> f64 {
        let m = self.mean();
        if m == f64::INFINITY {
            return f64::INFINITY;
        }
        2.0 * self.integral(1, 0.0, f64::INFINITY) - m * m
    }

    fn cdf(&self, x: f64) -> f64 {
        1.0 - self.survival(x)
    }

    /// `p = 1` gives `+∞`, or the truncation point.
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        let s = match self.truncation {
            Some((tr, Truncation::WholeDistribution)) => {
                let (s_tr, one_minus) = self.whole_mass(tr);
                return Ok(self.base_inverse(s_tr + (1.0 - p) * one_minus).min(tr));
            }
            Some((tr, Truncation::LastPiece)) if p == 1.0 => return Ok(tr),
            _ => 1.0 - p,
        };
        if s <= 0.0 {
            return Ok(f64::INFINITY);
        }
        Ok(self.base_inverse(s))
    }
}

impl Severity for PiecewisePareto {
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        self.integral(0, 0.0, limit)
    }

    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean() - retention;
        }
        self.integral(0, retention, f64::INFINITY)
    }

    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        self.integral(0, a, a + limit)
    }

    /// `2 ∫_a^b (x − a) S(x) dx` with `b = a + limit`.
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        let b = a + limit;
        2.0 * (self.integral(1, a, b) - a * self.integral(0, a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pareto;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    fn example() -> PiecewisePareto {
        PiecewisePareto::new(vec![1000.0, 2000.0, 3000.0], vec![1.0, 1.5, 2.0]).unwrap()
    }

    #[test]
    fn one_piece_is_a_pareto() {
        let pp = PiecewisePareto::new(vec![500.0], vec![1.7]).unwrap();
        let p = Pareto::new(500.0, 1.7).unwrap();
        let ppt = pp.clone().truncated(9000.0, Truncation::LastPiece).unwrap();
        let ppw = pp
            .clone()
            .truncated(9000.0, Truncation::WholeDistribution)
            .unwrap();
        let pt = p.truncated(9000.0).unwrap();
        for (a, b) in [(1000.0, 0.0), (4000.0, 1000.0), (f64::INFINITY, 2000.0)] {
            assert!(close(pp.layer(a, b), p.layer(a, b), 1e-14));
            assert!(close(
                pp.layer_second_moment(a.min(1e5), b),
                p.layer_second_moment(a.min(1e5), b),
                1e-13
            ));
            // With one piece both truncation kinds are the truncated Pareto.
            for q in [&ppt, &ppw] {
                assert!(close(q.layer(a, b), pt.layer(a, b), 1e-13));
                assert!(close(
                    q.layer_second_moment(a, b),
                    pt.layer_second_moment(a, b),
                    1e-12
                ));
            }
        }
        for x in [100.0, 500.0, 700.0, 8999.0] {
            assert!(close(ppt.survival(x), pt.survival(x), 1e-13));
            assert!(close(ppw.survival(x), pt.survival(x), 1e-13));
        }
    }

    #[test]
    fn survival_is_continuous_with_the_stated_alphas() {
        let pp = example();
        assert_eq!(pp.survival(999.0), 1.0);
        assert!(close(pp.survival(2000.0), 0.5, 1e-15));
        assert!(close(
            pp.survival(3000.0),
            0.5 * (2.0f64 / 3.0).powf(1.5),
            1e-15
        ));
        for &t in pp.thresholds() {
            let below = pp.survival(t * (1.0 - 1e-12));
            assert!(close(below, pp.survival(t), 1e-10), "{t}");
        }
        // Local alpha −x S'(x) / S(x) on each piece.
        for (x, alpha) in [(1500.0, 1.0), (2500.0, 1.5), (9000.0, 2.0)] {
            let h = x * 1e-6;
            let d = (pp.survival(x + h).ln() - pp.survival(x - h).ln()) / (2.0 * h);
            assert!(close(-x * d, alpha, 1e-6));
        }
    }

    #[test]
    fn severity_identities() {
        for pp in [
            example(),
            example().truncated(5000.0, Truncation::LastPiece).unwrap(),
            example()
                .truncated(5000.0, Truncation::WholeDistribution)
                .unwrap(),
            PiecewisePareto::new(vec![100.0, 200.0, 400.0], vec![0.5, 0.0, 3.0]).unwrap(),
        ] {
            for d in [500.0, 1000.0, 2500.0, 4000.0] {
                assert!(
                    close(pp.lev(d) + pp.stop_loss(d), pp.mean(), 1e-12),
                    "{pp:?} at {d}"
                );
                assert!(close(
                    pp.layer(1700.0, d),
                    pp.stop_loss(d) - pp.stop_loss(d + 1700.0),
                    1e-10
                ));
            }
            let m2 = pp.layer_second_moment(f64::INFINITY, 0.0);
            if pp.variance().is_finite() {
                assert!(close(m2 - pp.mean() * pp.mean(), pp.variance(), 1e-10));
            } else {
                assert_eq!(m2, f64::INFINITY);
            }
        }
        let heavy = PiecewisePareto::new(vec![1.0, 2.0], vec![3.0, 0.9]).unwrap();
        assert_eq!(heavy.mean(), f64::INFINITY);
        assert_eq!(heavy.variance(), f64::INFINITY);
    }

    #[test]
    fn cdf_quantile_round_trip() {
        for pp in [
            example(),
            example().truncated(5000.0, Truncation::LastPiece).unwrap(),
            example()
                .truncated(5000.0, Truncation::WholeDistribution)
                .unwrap(),
            PiecewisePareto::new(vec![100.0, 200.0, 400.0], vec![0.5, 0.0, 3.0]).unwrap(),
        ] {
            for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999] {
                let x = pp.quantile(q).unwrap();
                assert!(close(pp.cdf(x), q, 1e-12) || q == 0.0, "{pp:?} {q}");
            }
            assert_eq!(pp.quantile(0.0).unwrap(), pp.thresholds()[0]);
        }
        assert_eq!(example().quantile(1.0).unwrap(), f64::INFINITY);
        let t = example().truncated(5000.0, Truncation::LastPiece).unwrap();
        assert_eq!(t.quantile(1.0).unwrap(), 5000.0);
        assert_eq!(t.survival(5000.0), 0.0);
    }

    #[test]
    fn steep_pieces_do_not_underflow() {
        let pp = PiecewisePareto::new(vec![1.0, 10.0, 100.0], vec![200.0, 150.0, 2.0]).unwrap();
        // P(X > 100) = 10^-350: below f64, but each piece is finite.
        assert_eq!(pp.survival(200.0), 0.0);
        assert!(pp.mean().is_finite());
        let x = pp.quantile(0.5).unwrap();
        assert!(close(pp.cdf(x), 0.5, 1e-12));
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(PiecewisePareto::new(vec![], vec![]).is_err());
        assert!(PiecewisePareto::new(vec![1.0, 2.0], vec![1.0]).is_err());
        assert!(PiecewisePareto::new(vec![2.0, 1.0], vec![1.0, 1.0]).is_err());
        assert!(PiecewisePareto::new(vec![1.0, 2.0], vec![1.0, 0.0]).is_err());
        assert!(PiecewisePareto::new(vec![1.0, 2.0], vec![-1.0, 1.0]).is_err());
        assert!(PiecewisePareto::new(vec![0.0, 2.0], vec![1.0, 1.0]).is_err());
        assert!(example().truncated(2500.0, Truncation::LastPiece).is_err());
    }
}
