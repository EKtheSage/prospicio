//! Extreme value tails: a generalized Pareto distribution (GPD) fitted to
//! the exceedances over a threshold, for VaR and TVaR beyond the draws.
//!
//! Peaks over threshold: above a high threshold `u`, `X - u` given
//! `X > u` is approximately GPD (Pickands–Balkema–de Haan). The tail model
//! is `P(X > x) = p_u (1 + ξ (x - u) / β)^(-1/ξ)` for `x >= u`, with `p_u`
//! the share of draws above `u` (see `docs/design/risk.md`).

use act_core::{Error, Result};

use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// The generalized Pareto distribution with shape `xi`, scale `beta` and
/// location `u` (0 unless set), on `x >= u` (and `x <= u - beta / xi`
/// when `xi < 0`), as SciPy's `genpareto(c=xi, loc=u, scale=beta)`.
///
/// `P(X > x) = (1 + xi (x - u) / beta)^(-1/xi)`, and
/// `exp(-(x - u) / beta)` at `xi = 0`.
///
/// As a [`Severity`] it has closed-form layer means and second moments
/// for every `xi`, so it also serves as Riegel's generalized Pareto for
/// treaty pricing ([`Gpd::riegel`]).
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
    location: f64,
}

impl Gpd {
    pub fn new(xi: f64, beta: f64) -> Result<Self> {
        if !xi.is_finite() {
            return Err(invalid("xi", xi, "must be finite"));
        }
        if !beta.is_finite() || beta <= 0.0 {
            return Err(invalid("beta", beta, "must be finite and positive"));
        }
        Ok(Self {
            xi,
            beta,
            location: 0.0,
        })
    }

    /// The same distribution shifted to start at `location` (finite).
    pub fn shifted(self, location: f64) -> Result<Self> {
        if !location.is_finite() {
            return Err(invalid("location", location, "must be finite"));
        }
        Ok(Self { location, ..self })
    }

    /// Riegel's generalized Pareto with threshold `t`, initial alpha
    /// `alpha_ini` and tail alpha `alpha_tail` (all positive):
    ///
    /// ```text
    /// P(X > x) = (1 + (alpha_ini / alpha_tail) (x / t - 1))^(-alpha_tail),  x >= t,
    /// ```
    ///
    /// whose local Pareto alpha moves from `alpha_ini` at `t` to
    /// `alpha_tail` as `x` grows. It is this GPD with `xi = 1/alpha_tail`,
    /// `beta = t/alpha_ini` and location `t`, and matches
    /// `pGenPareto(x, t, alpha_ini, alpha_tail)` in the R package Pareto.
    ///
    /// ```
    /// use act_prob::{Severity, evt::Gpd};
    ///
    /// let g = Gpd::riegel(1000.0, 2.0, 1.5).unwrap();
    /// // P(X > 2000) = (1 + 2/1.5)^-1.5.
    /// assert!((g.survival(2000.0) - (7.0f64 / 3.0).powf(-1.5)).abs() < 1e-15);
    /// assert!(g.layer(4000.0, 1000.0) > 0.0);
    /// ```
    pub fn riegel(t: f64, alpha_ini: f64, alpha_tail: f64) -> Result<Self> {
        if !t.is_finite() || t <= 0.0 {
            return Err(invalid("t", t, "must be finite and positive"));
        }
        if !alpha_ini.is_finite() || alpha_ini <= 0.0 {
            return Err(invalid(
                "alpha_ini",
                alpha_ini,
                "must be finite and positive",
            ));
        }
        if !alpha_tail.is_finite() || alpha_tail <= 0.0 {
            return Err(invalid(
                "alpha_tail",
                alpha_tail,
                "must be finite and positive",
            ));
        }
        Self::new(1.0 / alpha_tail, t / alpha_ini)?.shifted(t)
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

    /// Location `u`, where the support starts.
    pub fn location(&self) -> f64 {
        self.location
    }

    /// `P(X > x)`.
    pub fn survival(&self, x: f64) -> f64 {
        self.excess_survival(x - self.location)
    }

    /// `P(X − u > z)`.
    fn excess_survival(&self, x: f64) -> f64 {
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
    /// `u + beta / (1 - xi)`; infinite for `xi >= 1`.
    fn mean(&self) -> f64 {
        if self.xi >= 1.0 {
            f64::INFINITY
        } else {
            self.location + self.beta / (1.0 - self.xi)
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
            return Ok(self.location + self.beta * e);
        }
        Ok(self.location + self.beta * (self.xi * e).exp_m1() / self.xi)
    }
}

impl Gpd {
    /// End of the support above the location: `-beta / xi` for `xi < 0`.
    fn excess_end(&self) -> f64 {
        if self.xi < 0.0 {
            -self.beta / self.xi
        } else {
            f64::INFINITY
        }
    }

    /// `(∫_m^b S dx, ∫_m^b (x − m) S dx)` for `u ≤ m ≤ b ≤ ∞`.
    ///
    /// Above `m` the excess is again a GPD, with scale `beta + xi (m − u)`
    /// and weight `S(m)`, so both integrals start at 0 of that GPD.
    fn excess_moments(&self, m: f64, b: f64) -> (f64, f64) {
        let z0 = m - self.location;
        let s0 = self.excess_survival(z0);
        if s0 == 0.0 || m >= b {
            return (0.0, 0.0);
        }
        let beta = self.beta + self.xi * z0;
        let c = (b - m).min(self.excess_end() - z0);
        let (k0, k1) = gpd_partial_moments(self.xi, beta, c);
        (s0 * k0, s0 * k1)
    }
}

impl Severity for Gpd {
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        self.layer(limit, 0.0)
    }

    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean() - retention;
        }
        self.layer(f64::INFINITY, retention)
    }

    /// `∫_a^b S(x) dx`: the part below the location pays in full.
    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        let b = a + limit;
        let below = (b.min(self.location) - a).max(0.0);
        below + self.excess_moments(a.max(self.location), b).0
    }

    /// `2 ∫_a^b (x − a) S(x) dx`.
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        let b = a + limit;
        let below = (b.min(self.location) - a).max(0.0);
        let m = a.max(self.location);
        let (k0, k1) = self.excess_moments(m, b);
        below * below + 2.0 * (k1 + (m - a) * k0)
    }
}

/// `(∫_0^c S(w) dw, ∫_0^c w S(w) dw)` for the GPD with shape `xi`, scale
/// `beta` and location 0, for `0 ≤ c ≤` the end of the support.
///
/// With `y = 1 + xi c / beta` and `L = ln y`, the first is
/// `(beta / xi) L exprel((1 − 1/xi) L)` (no cancellation for any `xi`),
/// and the second is
/// `(beta / xi)^2 L (exprel((2 − 1/xi) L) − exprel((1 − 1/xi) L))`. For
/// `|xi| < 1/4` the second is written without `1/xi` as
/// `beta^2 (1 − A (1 + (1 − xi) c / beta)) / ((1 − xi)(1 − 2 xi))` with
/// `A = S(c) y`, which is also the `xi = 0` (exponential) case. For
/// `c / beta` small both forms cancel, so the second uses its Taylor
/// series there.
fn gpd_partial_moments(xi: f64, beta: f64, c: f64) -> (f64, f64) {
    let full = c == f64::INFINITY || (xi < 0.0 && c >= -beta / xi);
    if full {
        let k0 = if xi < 1.0 {
            beta / (1.0 - xi)
        } else {
            f64::INFINITY
        };
        let k1 = if xi < 0.5 {
            beta * beta / ((1.0 - xi) * (1.0 - 2.0 * xi))
        } else {
            f64::INFINITY
        };
        return (k0, k1);
    }
    if c <= 0.0 {
        return (0.0, 0.0);
    }
    let r = c / beta;
    let l = (xi * r).ln_1p();
    let k0 = if xi == 0.0 {
        beta * -(-r).exp_m1()
    } else {
        beta / xi * l * exprel((1.0 - 1.0 / xi) * l)
    };
    let k1 = if r * xi.abs().max(1.0) < 0.05 {
        // S(w) = Σ_n (−1)^n Π_{j<n} (1 + j xi) (w/beta)^n / n!, integrated
        // against w: terms shrink by at most 0.05 each.
        let mut coef = 1.0;
        let mut sum = 0.0;
        for n in 0..14 {
            sum += coef / (f64::from(n) + 2.0);
            coef *= -(1.0 + f64::from(n) * xi) * r / (f64::from(n) + 1.0);
        }
        c * c * sum
    } else if xi.abs() < 0.25 {
        let a = if xi == 0.0 {
            (-r).exp()
        } else {
            ((1.0 - 1.0 / xi) * l).exp()
        };
        beta * beta * (1.0 - a * (1.0 + (1.0 - xi) * r)) / ((1.0 - xi) * (1.0 - 2.0 * xi))
    } else {
        let q = 1.0 - 1.0 / xi;
        (beta / xi).powi(2) * l * (exprel((q + 1.0) * l) - exprel(q * l))
    };
    (k0, k1)
}

/// `(e^z − 1) / z`, and 1 at `z = 0`.
fn exprel(z: f64) -> f64 {
    if z == 0.0 { 1.0 } else { z.exp_m1() / z }
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

    /// A tail from known parts; `gpd` models the exceedances, so its
    /// location must be 0.
    pub fn new(threshold: f64, p_exceed: f64, gpd: Gpd) -> Result<Self> {
        if gpd.location != 0.0 {
            return Err(invalid(
                "location",
                gpd.location,
                "the exceedance GPD must have location 0",
            ));
        }
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

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    /// `∫_a^b x^k S(x) dx` by composite Gauss–Legendre on a log scale,
    /// split at the kink at the location.
    fn quad(g: &Gpd, k: i32, a: f64, b: f64) -> f64 {
        let u = g.location();
        if a < u && u < b {
            return quad_smooth(g, k, a, u) + quad_smooth(g, k, u, b);
        }
        quad_smooth(g, k, a, b)
    }

    fn quad_smooth(g: &Gpd, k: i32, a: f64, b: f64) -> f64 {
        // 5-point Gauss–Legendre nodes and weights on [-1, 1].
        const X: [f64; 5] = [
            0.0,
            -0.538_469_310_105_683_1,
            0.538_469_310_105_683_1,
            -0.906_179_845_938_664,
            0.906_179_845_938_664,
        ];
        const W: [f64; 5] = [
            0.568_888_888_888_888_9,
            0.478_628_670_499_366_5,
            0.478_628_670_499_366_5,
            0.236_926_885_056_189_1,
            0.236_926_885_056_189_1,
        ];
        let n = 4000;
        let (la, lb) = (a.ln(), b.ln());
        let h = (lb - la) / f64::from(n);
        let mut sum = 0.0;
        for i in 0..n {
            let mid = la + (f64::from(i) + 0.5) * h;
            for (x, w) in X.iter().zip(W) {
                let v = (mid + 0.5 * h * x).exp();
                sum += w * 0.5 * h * v * v.powi(k) * g.survival(v);
            }
        }
        sum
    }

    #[test]
    fn riegel_matches_its_definition() {
        let g = Gpd::riegel(1000.0, 2.0, 1.5).unwrap();
        assert!(close(g.xi(), 1.0 / 1.5, 1e-15));
        assert!(close(g.beta(), 500.0, 1e-15));
        assert_eq!(g.location(), 1000.0);
        for x in [1000.0, 1500.0, 1e4, 1e7] {
            let want = (1.0f64 + (2.0 / 1.5) * (x / 1000.0 - 1.0)).powf(-1.5);
            assert!(close(g.survival(x), want, 1e-14));
        }
        assert_eq!(g.survival(999.0), 1.0);
        assert!(close(g.cdf(g.quantile(0.9).unwrap()), 0.9, 1e-14));
        assert!(close(g.mean(), 1000.0 + 500.0 / (1.0 - 1.0 / 1.5), 1e-15));
        // Equal alphas give the Pareto.
        let p = crate::Pareto::new(1000.0, 2.5).unwrap();
        let r = Gpd::riegel(1000.0, 2.5, 2.5).unwrap();
        for (l, a) in [(4000.0f64, 1000.0), (1e5, 5e4), (f64::INFINITY, 2000.0)] {
            assert!(close(r.layer(l, a), p.layer(l, a), 1e-13));
            assert!(close(
                r.layer_second_moment(l.min(1e9), a),
                p.layer_second_moment(l.min(1e9), a),
                1e-12
            ));
        }
        assert!(Gpd::riegel(0.0, 1.0, 1.0).is_err());
        assert!(Gpd::riegel(1.0, 0.0, 1.0).is_err());
        assert!(Gpd::riegel(1.0, 1.0, -1.0).is_err());
        assert!(PotTail::new(0.0, 0.1, r).is_err());
    }

    #[test]
    fn layer_moments_match_quadrature() {
        for (xi, beta, u) in [
            (0.0, 100.0, 0.0),
            (1e-9, 100.0, 0.0),
            (0.1, 100.0, 50.0),
            (-0.2, 100.0, 0.0),
            (-0.6, 100.0, 10.0),
            (0.25, 100.0, 0.0),
            (0.5, 100.0, 0.0),
            (0.8, 100.0, 0.0),
            (1.0, 100.0, 0.0),
            (1.7, 100.0, 20.0),
        ] {
            let g = Gpd::new(xi, beta).unwrap().shifted(u).unwrap();
            for (l, a) in [
                (1.0f64, 30.0f64),
                (10.0, 30.0),
                (200.0, 30.0),
                (500.0, 100.0),
                (2e4, 1e3),
            ] {
                let b = (a + l).min(u + g.excess_end());
                if a >= b {
                    assert_eq!(g.layer(l, a), 0.0);
                    continue;
                }
                let m1 = quad(&g, 0, a, b);
                let m2 = 2.0 * (quad(&g, 1, a, b) - a * m1);
                assert!(close(g.layer(l, a), m1, 1e-12), "{xi} {l} xs {a}");
                assert!(
                    close(g.layer_second_moment(l, a), m2, 1e-11),
                    "{xi} {l} xs {a}: {} {m2}",
                    g.layer_second_moment(l, a)
                );
            }
        }
    }

    #[test]
    fn partial_moments_are_continuous_across_branches() {
        // Series below r = 0.05, rational form for |xi| < 1/4, exprel form above.
        for xi in [-0.3, -0.25, -0.2, 0.0, 0.2, 0.25, 0.3, 2.0] {
            let below =
                gpd_partial_moments(xi, 1.0, 0.05 / f64::max(1.0, f64::abs(xi)) * (1.0 - 1e-9));
            let above =
                gpd_partial_moments(xi, 1.0, 0.05 / f64::max(1.0, f64::abs(xi)) * (1.0 + 1e-9));
            assert!(close(below.1, above.1, 1e-8), "{xi} {below:?} {above:?}");
        }
        for c in [0.1, 1.0, 3.0] {
            let lo = gpd_partial_moments(0.25 * (1.0 - 1e-12), 1.0, c);
            let hi = gpd_partial_moments(0.25, 1.0, c);
            assert!(close(lo.1, hi.1, 1e-11), "{c}");
        }
        // Exponential: ∫_0^c w e^-w dw = 1 − e^-c (1 + c).
        let (k0, k1) = gpd_partial_moments(0.0, 1.0, 2.0);
        assert!(close(k0, 1.0 - (-2.0f64).exp(), 1e-15));
        assert!(close(k1, 1.0 - 3.0 * (-2.0f64).exp(), 1e-15));
    }

    #[test]
    fn severity_identities() {
        for g in [
            Gpd::riegel(1000.0, 1.2, 2.5).unwrap(),
            Gpd::new(-0.3, 50.0).unwrap(),
            Gpd::new(0.0, 50.0).unwrap().shifted(10.0).unwrap(),
        ] {
            for d in [5.0, 100.0, 1500.0, 1e4] {
                assert!(
                    close(g.lev(d) + g.stop_loss(d), g.mean(), 1e-12),
                    "{g:?} {d}"
                );
            }
            let m2 = g.layer_second_moment(f64::INFINITY, 0.0);
            let loc = g.location();
            let raw2 = g.variance() + g.mean() * g.mean();
            assert!(close(m2, raw2, 1e-12), "{g:?} {m2} {raw2} {loc}");
        }
        let heavy = Gpd::riegel(1.0, 1.0, 0.9).unwrap();
        assert_eq!(heavy.stop_loss(10.0), f64::INFINITY);
        assert!(heavy.layer(10.0, 10.0).is_finite());
    }
}
