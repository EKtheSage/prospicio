//! Extreme value tails: a generalized Pareto distribution (GPD) fitted to
//! the exceedances over a threshold, for VaR and TVaR beyond the draws.
//!
//! Peaks over threshold: above a high threshold `u`, `X - u` given
//! `X > u` is approximately GPD (Pickands–Balkema–de Haan). The tail model
//! is `P(X > x) = p_u (1 + ξ (x - u) / β)^(-1/ξ)` for `x >= u`, with `p_u`
//! the share of draws above `u` (see `docs/design/risk.md`).

use act_core::{Error, Result};

use crate::distribution::{Distribution, check_probability};

/// The generalized Pareto distribution with shape `xi` and scale `beta`,
/// on `x >= 0` (and `x <= -beta / xi` when `xi < 0`), as SciPy's
/// `genpareto(c=xi, scale=beta)`.
///
/// `P(X > x) = (1 + xi x / beta)^(-1/xi)`, and `exp(-x / beta)` at
/// `xi = 0`.
///
/// ```
/// use act_prob::{Distribution, evt::Gpd};
///
/// let g = Gpd::new(0.5, 2.0).unwrap();
/// assert!((g.mean() - 4.0).abs() < 1e-12); // beta / (1 - xi)
/// assert!((g.cdf(g.quantile(0.99).unwrap()) - 0.99).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gpd {
    xi: f64,
    beta: f64,
}

impl Gpd {
    pub fn new(xi: f64, beta: f64) -> Result<Self> {
        if !xi.is_finite() {
            return Err(invalid("xi", xi, "must be finite"));
        }
        if !beta.is_finite() || beta <= 0.0 {
            return Err(invalid("beta", beta, "must be finite and positive"));
        }
        Ok(Self { xi, beta })
    }

    /// Shape `ξ`: heavier tails as it grows; moments of order `1/ξ` and
    /// above are infinite.
    pub fn xi(&self) -> f64 {
        self.xi
    }

    /// Scale `β`.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// `P(X > x)`.
    pub fn survival(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        let z = self.xi * x / self.beta;
        if self.xi == 0.0 {
            return (-x / self.beta).exp();
        }
        if z <= -1.0 {
            return 0.0;
        }
        (-z.ln_1p() / self.xi).exp()
    }

    /// Maximum likelihood fit to exceedances (values over a threshold,
    /// minus the threshold), all non-negative.
    ///
    /// Maximizes the profile likelihood in `θ = ξ / β` (Grimshaw 1993):
    /// for fixed `θ` the likelihood is maximized by
    /// `ξ(θ) = mean(ln(1 + θ y))`, leaving a one-dimensional search: a scan
    /// over `θ` finds the maximum, then bisection on the score pins it. `ξ` is restricted to
    /// `ξ > -1`, where the maximum likelihood estimate exists. Needs at
    /// least 3 exceedances, not all equal.
    ///
    /// ```
    /// use act_core::StreamRng;
    /// use act_prob::{Distribution, evt::Gpd};
    ///
    /// let truth = Gpd::new(0.3, 10.0).unwrap();
    /// let y = truth.sample(&mut StreamRng::new(1, 0), 20_000);
    /// let fit = Gpd::fit(&y).unwrap();
    /// assert!((fit.xi() - 0.3).abs() < 0.05);
    /// assert!((fit.beta() / 10.0 - 1.0).abs() < 0.05);
    /// ```
    pub fn fit(exceedances: &[f64]) -> Result<Self> {
        let y = exceedances;
        let n = y.len();
        if n < 3 {
            return Err(invalid("exceedances", n as f64, "needs at least 3 values"));
        }
        if let Some(&bad) = y.iter().find(|v| !v.is_finite() || **v < 0.0) {
            return Err(invalid(
                "exceedances",
                bad,
                "must be finite and non-negative",
            ));
        }
        let max = y.iter().copied().fold(0.0, f64::max);
        let mean = y.iter().sum::<f64>() / n as f64;
        if max == 0.0 || y.iter().all(|&v| v == y[0]) {
            return Err(invalid("exceedances", max, "must not all be equal"));
        }

        // Profile log-likelihood per observation in θ; θ = 0 is the
        // exponential limit.
        let profile = |theta: f64| -> f64 {
            if theta == 0.0 {
                return -mean.ln() - 1.0;
            }
            let xi = y.iter().map(|&v| (theta * v).ln_1p()).sum::<f64>() / n as f64;
            if !xi.is_finite() || xi <= -1.0 || xi == 0.0 {
                return f64::NEG_INFINITY;
            }
            let beta = xi / theta;
            if beta <= 0.0 {
                return f64::NEG_INFINITY;
            }
            -beta.ln() - (1.0 + xi)
        };

        // θ ranges over (-1/max, ∞). Scan it on log scales in units of
        // 1/max: positive θ from 1e-8 to 1e12, negative θ from -1e-8 to
        // within 1e-12 of the -1/max boundary, ascending.
        // Twenty points a decade.
        let decade = |j: i32| 10f64.powf(f64::from(j) / 20.0);
        let mut thetas: Vec<f64> = Vec::with_capacity(800);
        thetas.extend((1..=240).rev().map(|j| -(1.0 - decade(-j)) / max));
        thetas.extend((-160..=-1).rev().map(|j| -decade(j) / max));
        thetas.extend((-160..=240).map(|j| decade(j) / max));
        let (mut best_k, mut best) = (0, f64::NEG_INFINITY);
        for (k, &theta) in thetas.iter().enumerate() {
            let v = profile(theta);
            if v > best {
                best = v;
                best_k = k;
            }
        }
        if best <= profile(0.0) {
            return Self::new(0.0, mean);
        }

        // Refine by bisection on the score (the derivative of the profile
        // likelihood) between the scan's neighbours of the best point. A
        // search on the likelihood value would pin the maximum only to
        // about sqrt(eps); the score pins it to rounding.
        let score = |theta: f64| -> f64 {
            let (mut xi, mut dxi) = (0.0, 0.0);
            for &v in y {
                xi += (theta * v).ln_1p();
                dxi += v / (1.0 + theta * v);
            }
            let (xi, dxi) = (xi / n as f64, dxi / n as f64);
            -dxi / xi + 1.0 / theta - dxi
        };
        let theta_at = |k: usize| thetas[k];
        let mut a = theta_at(best_k.saturating_sub(1));
        let mut b = theta_at((best_k + 1).min(thetas.len() - 1));
        // The scan skips θ = 0, so a bracket never straddles it unless the
        // best point is next to it on both sides.
        if a < 0.0 && b > 0.0 {
            if theta_at(best_k) > 0.0 {
                a = theta_at(best_k) * 1e-3;
            } else {
                b = theta_at(best_k) * 1e-3;
            }
        }
        let theta = if score(a) > 0.0 && score(b) < 0.0 {
            for _ in 0..200 {
                let mid = 0.5 * (a + b);
                if mid == a || mid == b {
                    break;
                }
                if score(mid) > 0.0 {
                    a = mid;
                } else {
                    b = mid;
                }
            }
            0.5 * (a + b)
        } else {
            // The maximum is at the edge of the scan; keep the best point.
            theta_at(best_k)
        };
        let xi = y.iter().map(|&v| (theta * v).ln_1p()).sum::<f64>() / n as f64;
        Self::new(xi, xi / theta)
    }
}

impl Distribution for Gpd {
    /// `beta / (1 - xi)`; infinite for `xi >= 1`.
    fn mean(&self) -> f64 {
        if self.xi >= 1.0 {
            f64::INFINITY
        } else {
            self.beta / (1.0 - self.xi)
        }
    }

    /// `beta^2 / ((1 - xi)^2 (1 - 2 xi))`; infinite for `xi >= 1/2`.
    fn variance(&self) -> f64 {
        if self.xi >= 0.5 {
            f64::INFINITY
        } else {
            self.beta * self.beta / ((1.0 - self.xi).powi(2) * (1.0 - 2.0 * self.xi))
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        1.0 - self.survival(x)
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        // -ln(1 - p), accurate for small p.
        let e = -(-p).ln_1p();
        if self.xi == 0.0 {
            return Ok(self.beta * e);
        }
        Ok(self.beta * (self.xi * e).exp_m1() / self.xi)
    }
}

/// A peaks-over-threshold tail: draws above `threshold` modelled by a GPD.
///
/// `P(X > x) = p_u · S_GPD(x - u)` for `x >= u`, where `p_u` is the share
/// of draws above the threshold. VaR and TVaR at levels `p >= 1 - p_u`
/// come from the GPD, so they extend smoothly past the largest draw.
///
/// ```
/// use act_core::StreamRng;
/// use act_prob::{Distribution, Lognormal, Sampled, evt::PotTail};
///
/// let d = Lognormal::new(0.0, 1.0).unwrap();
/// let s = Sampled::new(d.sample(&mut StreamRng::new(3, 0), 100_000)).unwrap();
/// let tail = PotTail::fit(&s, 0.95).unwrap();
/// // The 99.9% quantile of LN(0, 1) is 21.98.
/// let q = tail.var(0.999).unwrap();
/// assert!((q / 21.98 - 1.0).abs() < 0.05, "{q}");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PotTail {
    threshold: f64,
    p_exceed: f64,
    gpd: Gpd,
}

impl PotTail {
    /// Fits a GPD to the draws above the empirical `level` quantile of
    /// `draws` (for example `0.95` for the top 5%).
    pub fn fit(draws: &impl crate::Empirical, level: f64) -> Result<Self> {
        check_probability(level)?;
        let sorted = draws.sorted();
        let threshold = crate::risk::var_sorted(sorted, level)?;
        let above: Vec<f64> = sorted
            .iter()
            .filter(|&&x| x > threshold)
            .map(|&x| x - threshold)
            .collect();
        let gpd = Gpd::fit(&above)?;
        Ok(Self {
            threshold,
            p_exceed: above.len() as f64 / sorted.len() as f64,
            gpd,
        })
    }

    /// A tail from known parts.
    pub fn new(threshold: f64, p_exceed: f64, gpd: Gpd) -> Result<Self> {
        if !threshold.is_finite() {
            return Err(invalid("threshold", threshold, "must be finite"));
        }
        if !(p_exceed > 0.0 && p_exceed <= 1.0) {
            return Err(invalid("p_exceed", p_exceed, "must be in (0, 1]"));
        }
        Ok(Self {
            threshold,
            p_exceed,
            gpd,
        })
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Share of draws above the threshold, `p_u`.
    pub fn p_exceed(&self) -> f64 {
        self.p_exceed
    }

    pub fn gpd(&self) -> &Gpd {
        &self.gpd
    }

    /// `P(X > x)` for `x >= threshold`.
    pub fn survival(&self, x: f64) -> f64 {
        self.p_exceed * self.gpd.survival(x - self.threshold)
    }

    /// VaR at `p`, for `p >= 1 - p_exceed`:
    /// `u + β ((p_u / (1 - p))^ξ - 1) / ξ`.
    pub fn var(&self, p: f64) -> Result<f64> {
        self.check_level(p)?;
        // P(Y > y) = (1 - p) / p_u for the exceedance Y.
        // Clamped: at p = 1 - p_u rounding can leave q a hair below 0.
        let q = (1.0 - (1.0 - p) / self.p_exceed).max(0.0);
        Ok(self.threshold + self.gpd.quantile(q)?)
    }

    /// TVaR at `p`, for `p >= 1 - p_exceed` and `ξ < 1`:
    /// `(VaR + β - ξ u) / (1 - ξ)`, the GPD's mean excess added to the VaR.
    pub fn tvar(&self, p: f64) -> Result<f64> {
        let var = self.var(p)?;
        let (xi, beta) = (self.gpd.xi, self.gpd.beta);
        if xi >= 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(var + (beta + xi * (var - self.threshold)) / (1.0 - xi))
    }

    fn check_level(&self, p: f64) -> Result<()> {
        check_probability(p)?;
        if p < 1.0 - self.p_exceed {
            return Err(invalid(
                "p",
                p,
                "is below the tail: must be at least 1 - p_exceed",
            ));
        }
        Ok(())
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
    use act_core::StreamRng;

    #[test]
    fn gpd_closed_forms() {
        let g = Gpd::new(0.25, 4.0).unwrap();
        assert!((g.mean() - 4.0 / 0.75).abs() < 1e-12);
        assert!((g.variance() - 16.0 / (0.5625 * 0.5)).abs() < 1e-12);
        // Exponential limit.
        let e = Gpd::new(0.0, 2.0).unwrap();
        assert!((e.cdf(2.0) - (1.0 - (-1f64).exp())).abs() < 1e-15);
        assert!((e.quantile(0.5).unwrap() - 2.0 * 2f64.ln()).abs() < 1e-15);
        // Bounded support for xi < 0: x <= beta / |xi|.
        let b = Gpd::new(-0.5, 1.0).unwrap();
        assert_eq!(b.survival(2.0), 0.0);
        assert!((b.quantile(1.0).unwrap() - 2.0).abs() < 1e-15);
        assert_eq!(Gpd::new(1.5, 1.0).unwrap().mean(), f64::INFINITY);
        for p in [1e-12, 0.1, 0.5, 0.99, 1.0 - 1e-12] {
            let x = g.quantile(p).unwrap();
            assert!((g.cdf(x) - p).abs() < 1e-12 * p.max(1e-3), "p = {p}");
        }
        assert!(Gpd::new(0.1, 0.0).is_err());
    }

    #[test]
    fn fit_recovers_parameters() {
        for (xi, beta) in [(0.5, 1.0), (0.0, 3.0), (-0.3, 2.0), (1.2, 0.5)] {
            let truth = Gpd::new(xi, beta).unwrap();
            let y = truth.sample(&mut StreamRng::new(11, 0), 50_000);
            let fit = Gpd::fit(&y).unwrap();
            assert!((fit.xi() - xi).abs() < 0.03, "xi {xi}: {fit:?}");
            assert!(
                (fit.beta() / beta - 1.0).abs() < 0.04,
                "beta {beta}: {fit:?}"
            );
        }
    }

    #[test]
    fn fit_rejects_degenerate_input() {
        assert!(Gpd::fit(&[1.0, 2.0]).is_err());
        assert!(Gpd::fit(&[1.0, 1.0, 1.0]).is_err());
        assert!(Gpd::fit(&[1.0, -1.0, 2.0]).is_err());
    }

    #[test]
    fn pot_tail_is_exact_for_a_gpd_tail() {
        // A tail from known parts: VaR and TVaR from the closed forms.
        let tail = PotTail::new(10.0, 0.05, Gpd::new(0.25, 2.0).unwrap()).unwrap();
        // At p = 0.95 the VaR is the threshold.
        assert!((tail.var(0.95).unwrap() - 10.0).abs() < 1e-12);
        // TVaR at the threshold is u + mean excess = 10 + 2 / 0.75.
        assert!((tail.tvar(0.95).unwrap() - (10.0 + 2.0 / 0.75)).abs() < 1e-12);
        // P(X > VaR(p)) = 1 - p.
        for p in [0.96, 0.99, 0.9999] {
            let v = tail.var(p).unwrap();
            assert!((tail.survival(v) - (1.0 - p)).abs() < 1e-14);
        }
        assert!(tail.var(0.9).is_err());
    }
}
