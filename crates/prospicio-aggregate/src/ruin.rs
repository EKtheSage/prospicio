//! Eventual ruin of the classical (Cramér–Lundberg) surplus process by the
//! Pollaczeck–Khinchine formula: `aggregate`'s `pollaczeck_khinchine` and
//! `eventual_ruin`.
//!
//! Claims arrive as a Poisson process and the premium comes in continuously
//! at `(1 + ρ)` times the expected loss rate, with `ρ > 0`. The maximal
//! aggregate loss `L` over all time, the most the surplus ever falls below
//! its start, is a compound geometric: a geometric number of ladder heights
//! with `P(N = n) = (1 - q) qⁿ`, `q = 1 / (1 + ρ)`, each drawn from the
//! equilibrium (integrated tail) severity `f_I(x) = S(x) / E[X]`. The
//! chance of eventual ruin from initial surplus `u` is `ψ(u) = P(L > u)`
//! (Embrechts, Klüppelberg and Mikosch, *Modelling Extremal Events*, 1.2,
//! formula 1.11). The Poisson rate cancels: `ψ` depends only on the
//! severity and `ρ`.
//!
//! On a grid with step `h` the equilibrium severity is `h S(jh) / E[X]` at
//! `jh`, with `S(jh) = P(X > jh)` and `E[X]` the grid's mean, so it sums
//! to 1. The geometric is a negative binomial with `r = 1` and `β = 1/ρ`,
//! compounded by [`fft`](crate::fft()) with its aliasing check.

use prospicio_core::{Error, Result};
use prospicio_prob::{Grid, NegativeBinomial};

use crate::compound::CompoundReport;

/// The part of the severity the ruin calculation uses: `aggregate`'s
/// `cap`, `excess` and `stop_loss` arguments.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SeverityPart {
    /// The whole severity.
    #[default]
    All,
    /// `X | X <= c`: the points above `c` dropped and the rest rescaled.
    AtMost(f64),
    /// `X | X > c`: the points at or below `c` dropped and the rest
    /// rescaled (not shifted).
    Above(f64),
    /// `X ∧ s`: the mass above the first point at or above `s` moved onto
    /// it, a stop loss on each claim.
    StopLoss(f64),
}

impl SeverityPart {
    /// The severity probabilities after the change.
    fn apply(self, g: &Grid) -> Result<Vec<f64>> {
        let mut p = g.probs().to_vec();
        let x = |j: usize| g.x(j);
        match self {
            Self::All => {}
            Self::AtMost(c) => {
                p.iter_mut()
                    .enumerate()
                    .filter(|&(j, _)| x(j) > c)
                    .for_each(|(_, v)| *v = 0.0);
                rescale(&mut p)?;
            }
            Self::Above(c) => {
                p.iter_mut()
                    .enumerate()
                    .filter(|&(j, _)| x(j) <= c)
                    .for_each(|(_, v)| *v = 0.0);
                rescale(&mut p)?;
            }
            Self::StopLoss(s) => {
                let at = (0..p.len()).find(|&j| x(j) >= s).unwrap_or(p.len() - 1);
                let above: f64 = p[at + 1..].iter().sum();
                p[at] += above;
                p[at + 1..].iter_mut().for_each(|v| *v = 0.0);
            }
        }
        Ok(p)
    }
}

fn rescale(p: &mut [f64]) -> Result<()> {
    let sum: f64 = p.iter().sum();
    if sum <= 0.0 {
        return Err(Error::Data(
            "no severity mass is left in that part of the severity".into(),
        ));
    }
    p.iter_mut().for_each(|v| *v /= sum);
    Ok(())
}

/// The probability of eventual ruin at each initial surplus on the grid,
/// from [`pollaczeck_khinchine`].
#[derive(Debug, Clone, PartialEq)]
pub struct Ruin {
    rho: f64,
    step: f64,
    /// `ψ(jh)`, non-increasing. `ψ(0)` is below the continuous model's
    /// `1 / (1 + ρ)`: the grid puts the ladder heights below `h` at 0.
    psi: Vec<f64>,
    /// The (possibly conditioned) severity's probabilities.
    severity: Vec<f64>,
    severity_mean: f64,
    equilibrium: Grid,
    report: CompoundReport,
}

/// `ψ(u)`, the probability of eventual ruin from initial surplus `u`, for a
/// Poisson stream of claims with severity `severity` and premium at
/// `(1 + rho)` times the expected loss rate, on `points` points of the
/// severity's step. See the [module](self). `rho` must be positive: at
/// `rho <= 0` ruin is certain.
///
/// The FFT's [`aliasing_error`](CompoundReport::aliasing_error) is in the
/// [`report`](Ruin::report); `ψ` at the last point is the compound's mass
/// beyond the grid, `P(L > (n - 1)h)`.
///
/// ```
/// use prospicio_aggregate::ruin::{SeverityPart, pollaczeck_khinchine};
/// use prospicio_prob::Grid;
///
/// // Every claim is 1: the equilibrium severity is uniform on [0, 1),
/// // which the grid puts at 0. So L = 0 and ruin needs no surplus below
/// // the grid's step: ψ(0) is the chance of a ladder height at all.
/// let sev = Grid::new(1.0, vec![0.0, 1.0]).unwrap();
/// let ruin = pollaczeck_khinchine(&sev, 0.25, 64, SeverityPart::All).unwrap();
/// assert!(ruin.psi_at(0.0) < 1e-12);
/// ```
pub fn pollaczeck_khinchine(
    severity: &Grid,
    rho: f64,
    points: usize,
    part: SeverityPart,
) -> Result<Ruin> {
    if !(rho.is_finite() && rho > 0.0) {
        return Err(Error::InvalidParameter {
            name: "rho",
            value: rho,
            reason: "must be finite and positive: at rho <= 0 ruin is certain",
        });
    }
    let h = severity.step();
    let p = part.apply(severity)?;
    let mean: f64 = p.iter().enumerate().map(|(j, v)| j as f64 * h * v).sum();
    if mean <= 0.0 {
        return Err(Error::Data(
            "the severity has no mass above zero, so ruin is impossible".into(),
        ));
    }
    // h S(jh) / E[X], S(jh) = P(X > jh) summed from the top.
    let mut eq = vec![0.0; p.len()];
    let mut above = 0.0;
    for j in (0..p.len()).rev() {
        eq[j] = h * above / mean;
        above += p[j];
    }
    let equilibrium = Grid::new(h, eq)?;
    let geometric = NegativeBinomial::new(1.0, 1.0 / rho)?;
    let (maximal_loss, report) = crate::fft(&geometric, &equilibrium, points)?;
    let probs = maximal_loss.probs();
    let mut psi = Vec::with_capacity(points);
    let mut cum = 0.0;
    for &g in &probs[..points - 1] {
        cum += g;
        psi.push((1.0 - cum).max(0.0));
    }
    psi.push(report.tail_mass);
    Ok(Ruin {
        rho,
        step: h,
        psi,
        severity: p,
        severity_mean: mean,
        equilibrium,
        report,
    })
}

impl Ruin {
    /// The margin `ρ`; the loss ratio is `1 / (1 + ρ)`.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// `ψ(jh)` at each grid point `jh`.
    pub fn psi(&self) -> &[f64] {
        &self.psi
    }

    /// The grid's step.
    pub fn step(&self) -> f64 {
        self.step
    }

    /// `ψ(u)`: `L` lives on the grid, so `ψ(u) = ψ(jh)` for `jh <= u <
    /// (j + 1)h`. Beyond the grid the last value, an upper bound.
    pub fn psi_at(&self, u: f64) -> f64 {
        if u < 0.0 {
            return 1.0;
        }
        let j = ((u / self.step) * (1.0 + 1e-12)).floor() as usize;
        self.psi[j.min(self.psi.len() - 1)]
    }

    /// The smallest initial surplus on the grid with `ψ(u) < p`, or `None`
    /// when the grid does not reach it: `aggregate`'s `find_u` with
    /// `kind='index'`.
    pub fn capital(&self, p: f64) -> Option<f64> {
        let j = self.psi.partition_point(|&v| v >= p);
        (j < self.psi.len()).then_some(j as f64 * self.step)
    }

    /// The initial surplus with `ψ(u) = p`, linear between the bracketing
    /// grid points: `aggregate`'s `find_u` with `kind='interp'`. `None`
    /// when the grid does not reach it or `p >= ψ(0)`.
    pub fn capital_interpolated(&self, p: f64) -> Option<f64> {
        let j = self.psi.partition_point(|&v| v >= p);
        if j == 0 || j >= self.psi.len() {
            return None;
        }
        let (lo, hi) = (self.psi[j - 1], self.psi[j]);
        let x = |k: usize| k as f64 * self.step;
        Some(x(j) + (p - hi) / (lo - hi) * (x(j - 1) - x(j)))
    }

    /// The severity's mean on the grid, after any [`SeverityPart`].
    pub fn severity_mean(&self) -> f64 {
        self.severity_mean
    }

    /// The equilibrium (integrated tail) severity on the grid.
    pub fn equilibrium(&self) -> &Grid {
        &self.equilibrium
    }

    /// The compound calculation's report.
    pub fn report(&self) -> &CompoundReport {
        &self.report
    }

    /// The Lundberg (adjustment) exponent `R > 0` solving
    /// `M_X(R) = 1 + (1 + ρ) E[X] R` on the grid's severity, which gives
    /// Lundberg's bound `ψ(u) <= e^{-Ru}`; `None` when no root lies below
    /// `700 / max(x)`, where `e^{Rx}` would overflow (a heavy tail on a wide
    /// grid). As `aggregate`'s `_lundberg_exponent`.
    pub fn lundberg_exponent(&self) -> Option<f64> {
        let h = self.step;
        let top = self.severity.iter().rposition(|&v| v > 0.0)? as f64 * h;
        if top <= 0.0 {
            return None;
        }
        let f = |r: f64| {
            self.severity
                .iter()
                .enumerate()
                .map(|(j, v)| v * (r * j as f64 * h).exp())
                .sum::<f64>()
                - 1.0
                - (1.0 + self.rho) * self.severity_mean * r
        };
        let r_max = 700.0 / top;
        if f(r_max) <= 0.0 {
            return None;
        }
        // f is convex, 0 at 0 with slope -ρ E[X] < 0: one root in (0, r_max].
        Some(prospicio_math::roots::bisect(1e-12, r_max, |r| f(r) < 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panjer;
    use prospicio_prob::Gamma;

    #[test]
    fn exponential_claims_match_the_closed_form() {
        // ψ(u) = e^{-ρu / ((1 + ρ)μ)} / (1 + ρ) for exponential claims; the
        // grid's error is O(h) against a mean of 100.
        let (mu, rho) = (100.0, 0.2);
        let (sev, _) =
            Grid::rounding(&Gamma::from_mean_cv(mu, 1.0).unwrap(), 0.25, 1 << 14).unwrap();
        let ruin = pollaczeck_khinchine(&sev, rho, 1 << 15, SeverityPart::All).unwrap();
        assert!(ruin.report().aliasing_error < 1e-12);
        for u in [0.0, 100.0, 500.0, 1000.0, 2000.0] {
            let exact = (-rho * u / ((1.0 + rho) * mu)).exp() / (1.0 + rho);
            assert!(
                (ruin.psi_at(u) / exact - 1.0).abs() < 5e-3,
                "u = {u}: {} vs {exact}",
                ruin.psi_at(u)
            );
        }
        // R = ρ / ((1 + ρ)μ) for exponential claims.
        let r = ruin.lundberg_exponent().unwrap();
        assert!((r / (rho / ((1.0 + rho) * mu)) - 1.0).abs() < 5e-3);
        let u = ruin.capital(0.01).unwrap();
        assert!(ruin.psi_at(u) < 0.01 && ruin.psi_at(u - 0.25) >= 0.01);
        let ui = ruin.capital_interpolated(0.01).unwrap();
        assert!(ui <= u && ui >= u - 0.25);
    }

    #[test]
    fn fft_matches_panjer_and_parts_are_honoured() {
        let sev = Grid::new(1.0, vec![0.0, 0.5, 0.3, 0.2]).unwrap();
        let ruin = pollaczeck_khinchine(&sev, 0.1, 256, SeverityPart::All).unwrap();
        let (exact, _) = panjer(
            &NegativeBinomial::new(1.0, 10.0).unwrap(),
            ruin.equilibrium(),
            256,
        )
        .unwrap();
        let mut cum = 0.0;
        for (k, g) in exact.probs()[..255].iter().enumerate() {
            cum += g;
            assert!((ruin.psi()[k] - (1.0 - cum)).abs() < 1e-12);
        }
        // ψ(0) = P(L > 0): 1 / (1 + ρ) less the ladder heights the grid
        // puts at 0, with f_I(0) = h / E[X].
        let (q, f0) = (1.0 / 1.1, 1.0 / 1.7);
        assert!((ruin.psi()[0] - q * (1.0 - f0) / (1.0 - q * f0)).abs() < 1e-12);
        let capped = SeverityPart::AtMost(2.0).apply(&sev).unwrap();
        assert!((capped[1] - 0.625).abs() < 1e-15 && capped[3] == 0.0);
        let above = SeverityPart::Above(1.0).apply(&sev).unwrap();
        assert!((above[2] - 0.6).abs() < 1e-15 && above[1] == 0.0);
        let stop = SeverityPart::StopLoss(2.0).apply(&sev).unwrap();
        assert!((stop[2] - 0.5).abs() < 1e-15 && stop[3] == 0.0);
        assert!(pollaczeck_khinchine(&sev, 0.0, 16, SeverityPart::All).is_err());
        assert!(pollaczeck_khinchine(&sev, 0.1, 16, SeverityPart::Above(5.0)).is_err());
    }
}
