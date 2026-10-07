//! Choosing a Tweedie GLM's power by profile likelihood.
//!
//! A Tweedie GLM fixes the power `p` (and its dispersion `φ` only scales the
//! variance), so IRLS gives the same coefficients for any `φ` but different
//! ones for each `p`. The power is chosen by maximizing the profile
//! log-likelihood: for each `p`, fit the GLM, then maximize the exact Tweedie
//! log-likelihood (Dunn and Smyth's series density, from `prospicio-prob`) over
//! `φ`. This is the method of R's `tweedie.profile` (Dunn and Smyth, 2005,
//! 2008).
//!
//! Since the coefficients at a fixed `p` do not depend on `φ`, maximizing
//! the likelihood jointly over coefficients, `p` and `φ` is maximizing this
//! one-dimensional profile. [`TweedieGlm`] does that by Brent's method, with
//! each fit warm-started from the last, and reports a profile-likelihood
//! interval for `p`; [`tweedie_profile`] evaluates the profile on a grid,
//! for plotting.

use prospicio_core::{Error, Result};
use prospicio_math::optimize::brent;
use prospicio_math::roots::illinois;
use prospicio_models::{Design, Family, Fitted, Link, Model};
use prospicio_prob::PredictiveDistribution;

use crate::{Dispersion, Glm, GlmFit};

/// The profile log-likelihood of a Tweedie GLM's power, from
/// [`tweedie_profile`].
#[derive(Debug, Clone)]
pub struct TweedieProfile {
    /// The powers tried, in the order given.
    pub powers: Vec<f64>,
    /// The log-likelihood at each power, maximized over `φ`.
    pub log_likelihood: Vec<f64>,
    /// The maximum-likelihood dispersion at each power.
    pub dispersion: Vec<f64>,
    /// The power maximizing the profile, refined between the grid points
    /// around the best one.
    pub power: f64,
    /// The maximum-likelihood dispersion at `power`.
    pub phi: f64,
    /// The profile log-likelihood at `power`.
    pub max_log_likelihood: f64,
    /// The GLM at `power`, its dispersion fixed at `phi`.
    pub fit: GlmFit,
}

/// Profiles the power of a Tweedie GLM with `link` over `powers` (each in
/// `(1, 2)`, increasing), then refines the best by golden-section search
/// between its neighbours to about `1e-4`.
///
/// ```
/// use prospicio_glm::tweedie::tweedie_profile;
/// use prospicio_models::{Design, Family, Link, Model};
/// use prospicio_prob::{Distribution, Tweedie};
///
/// // Pure premiums from a Tweedie with power 1.5 and a log-linear mean.
/// let n = 400;
/// let x: Vec<f64> = (0..n).map(|i| f64::from(i % 4)).collect();
/// let y: Vec<f64> = (0..n)
///     .map(|i| {
///         let mu = (0.5 + 0.3 * x[i as usize]).exp();
///         let u = (f64::from(i) * 0.618_034).fract() * 0.98 + 0.01;
///         Tweedie::new(mu, 2.0, 1.5).unwrap().quantile(u).unwrap()
///     })
///     .collect();
/// let d = Design::new(vec!["(Intercept)".into(), "x".into()], vec![vec![1.0; n as usize], x])
///     .unwrap();
/// let prof = tweedie_profile(Link::Log, &d, &y, &[1.2, 1.4, 1.6, 1.8]).unwrap();
/// assert!((prof.power - 1.5).abs() < 0.1);
/// ```
pub fn tweedie_profile(
    link: Link,
    design: &Design,
    y: &[f64],
    powers: &[f64],
) -> Result<TweedieProfile> {
    if powers.is_empty() || powers.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::Data(
            "powers must be non-empty and increasing".into(),
        ));
    }
    let point = |p: f64| -> Result<(f64, f64, GlmFit)> {
        let glm = Glm::new(Family::Tweedie { power: p }, link);
        let fit = glm.fit(design, y)?;
        let (phi, ll) = max_over_phi(p, y, fit.fitted(), design.weights(), fit.dispersion())?;
        Ok((ll, phi, fit))
    };
    let mut lls = Vec::with_capacity(powers.len());
    let mut phis = Vec::with_capacity(powers.len());
    for &p in powers {
        let (ll, phi, _) = point(p)?;
        lls.push(ll);
        phis.push(phi);
    }
    let best = (0..lls.len())
        .max_by(|&a, &b| lls[a].total_cmp(&lls[b]))
        .expect("at least one power");
    let lo = powers[best.saturating_sub(1)];
    let hi = powers[(best + 1).min(powers.len() - 1)];
    let power = if lo < hi {
        golden_max(lo, hi, 1e-5, |p| {
            point(p).map_or(f64::NEG_INFINITY, |r| r.0)
        })
    } else {
        powers[best]
    };
    let (max_log_likelihood, phi, _) = point(power)?;
    let fit = Glm::new(Family::Tweedie { power }, link)
        .dispersion(Dispersion::Fixed(phi))
        .fit(design, y)?;
    Ok(TweedieProfile {
        powers: powers.to_vec(),
        log_likelihood: lls,
        dispersion: phis,
        power,
        phi,
        max_log_likelihood,
        fit,
    })
}

/// Half the 95% point of the χ² distribution with one degree of freedom:
/// the drop from the maximum that bounds a 95% profile-likelihood interval.
const HALF_CHI2_95: f64 = 1.920_729_410_347_062;

/// A Tweedie GLM whose power is estimated with its coefficients and
/// dispersion, by maximum likelihood.
///
/// ```
/// use prospicio_glm::tweedie::TweedieGlm;
/// use prospicio_models::{Design, Fitted, Link, Model};
/// use prospicio_prob::{Distribution, Tweedie};
///
/// let n = 400;
/// let x: Vec<f64> = (0..n).map(|i| f64::from(i % 4)).collect();
/// let y: Vec<f64> = (0..n)
///     .map(|i| {
///         let mu = (0.5 + 0.3 * x[i as usize]).exp();
///         let u = (f64::from(i) * 0.618_034).fract() * 0.98 + 0.01;
///         Tweedie::new(mu, 2.0, 1.5).unwrap().quantile(u).unwrap()
///     })
///     .collect();
/// let d = Design::new(vec!["(Intercept)".into(), "x".into()], vec![vec![1.0; n as usize], x])
///     .unwrap();
/// let fit = TweedieGlm::new(Link::Log).fit(&d, &y).unwrap();
/// let (lo, hi) = fit.interval();
/// assert!(lo < fit.power() && fit.power() < hi && !fit.at_boundary());
/// assert!((fit.power() - 1.5).abs() < 0.1);
/// assert_eq!(fit.predict(&d).unwrap().len(), 400);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweedieGlm {
    pub link: Link,
    /// The power is searched in `[min_power, max_power]`, inside `(1, 2)`.
    pub min_power: f64,
    pub max_power: f64,
    /// Brent's relative tolerance on the power.
    pub tolerance: f64,
}

impl TweedieGlm {
    /// Powers searched in `[1.01, 1.99]` to a relative tolerance of `1e-6`.
    pub fn new(link: Link) -> Self {
        Self {
            link,
            min_power: 1.01,
            max_power: 1.99,
            tolerance: 1e-6,
        }
    }
}

impl Model for TweedieGlm {
    type Fitted = TweedieFit;

    /// Maximizes the profile log-likelihood over the power by Brent's
    /// method (each GLM fit warm-started from the previous one's means),
    /// then finds where the profile falls 1.92 below its maximum on each
    /// side by the Illinois method: the 95% interval. Fails if a bound is
    /// outside `(1, 2)` or no power gives a finite likelihood.
    fn fit(&self, design: &Design, y: &[f64]) -> Result<TweedieFit> {
        let (lo, hi) = (self.min_power, self.max_power);
        if !(lo > 1.0 && lo < hi && hi < 2.0) {
            return Err(Error::InvalidParameter {
                name: "min_power",
                value: lo,
                reason: "the power bounds must satisfy 1 < min_power < max_power < 2",
            });
        }
        let last_mu = std::cell::RefCell::new(None::<Vec<f64>>);
        let evaluations = std::cell::Cell::new(0usize);
        let profile = |p: f64| -> Result<(f64, f64, GlmFit)> {
            evaluations.set(evaluations.get() + 1);
            let glm = Glm::new(Family::Tweedie { power: p }, self.link);
            let fit = glm.fit_from(design, y, last_mu.borrow().as_deref())?;
            let (phi, ll) = max_over_phi(p, y, fit.fitted(), design.weights(), fit.dispersion())?;
            *last_mu.borrow_mut() = Some(fit.fitted().to_vec());
            Ok((ll, phi, fit))
        };
        let ll_at = |p: f64| profile(p).map_or(f64::NEG_INFINITY, |r| r.0);
        let (power, neg, _) = brent(lo, hi, self.tolerance, |p| -ll_at(p));
        if !neg.is_finite() {
            return Err(Error::Data(
                "no Tweedie power gives a finite log-likelihood".into(),
            ));
        }
        let (log_likelihood, phi, _) = profile(power)?;
        let target = log_likelihood - HALF_CHI2_95;
        let edge = |end: f64| -> (f64, bool) {
            let g_end = ll_at(end) - target;
            if g_end >= 0.0 {
                (end, true)
            } else {
                let g = |p: f64| ll_at(p) - target;
                (
                    illinois(end, power, g_end, log_likelihood - target, g),
                    false,
                )
            }
        };
        let (lower, lower_open) = edge(lo);
        let (upper, upper_open) = edge(hi);
        let width = hi - lo;
        let at_boundary = power - lo < 1e-3 * width || hi - power < 1e-3 * width;
        let glm = Glm::new(Family::Tweedie { power }, self.link)
            .dispersion(Dispersion::Fixed(phi))
            .fit_from(design, y, last_mu.borrow().as_deref())?;
        Ok(TweedieFit {
            power,
            phi,
            log_likelihood,
            interval: (lower, upper),
            interval_open: (lower_open, upper_open),
            at_boundary,
            evaluations: evaluations.get(),
            glm,
        })
    }
}

/// A Tweedie GLM with its power estimated, from [`TweedieGlm`].
#[derive(Debug, Clone, PartialEq)]
pub struct TweedieFit {
    power: f64,
    phi: f64,
    log_likelihood: f64,
    interval: (f64, f64),
    interval_open: (bool, bool),
    at_boundary: bool,
    evaluations: usize,
    glm: GlmFit,
}

impl TweedieFit {
    /// The maximum-likelihood power.
    pub fn power(&self) -> f64 {
        self.power
    }

    /// The maximum-likelihood dispersion at that power.
    pub fn dispersion(&self) -> f64 {
        self.phi
    }

    /// The maximized log-likelihood.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// The 95% profile-likelihood interval for the power: where the
    /// profile is within 1.92 of its maximum. An end at a search bound
    /// means the profile had not fallen that far there
    /// ([`interval_open`](Self::interval_open)).
    pub fn interval(&self) -> (f64, f64) {
        self.interval
    }

    /// Whether each end of [`interval`](Self::interval) is a search bound
    /// rather than a crossing.
    pub fn interval_open(&self) -> (bool, bool) {
        self.interval_open
    }

    /// Whether the power is at a search bound: the likelihood pushes it
    /// towards 1 (a Poisson-like response) or 2 (gamma-like, typically no
    /// zeros), and the estimate is the bound, not an interior maximum.
    pub fn at_boundary(&self) -> bool {
        self.at_boundary
    }

    /// Profile evaluations (GLM fits) used.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// The GLM at the estimated power, its dispersion fixed at the
    /// estimate: coefficients, standard errors (conditional on the power)
    /// and the rest.
    pub fn glm(&self) -> &GlmFit {
        &self.glm
    }
}

impl Fitted for TweedieFit {
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        self.glm.predict(design)
    }

    /// The GLM's joint draws at the estimated power and dispersion: the
    /// coefficients' uncertainty is included, the power's is not.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        self.glm.predict_distribution(design, n_sims, seed)
    }
}

/// The dispersion maximizing the Tweedie log-likelihood at fixed means, and
/// that maximum, searching `ln φ` within a factor of 100 of `start`.
fn max_over_phi(p: f64, y: &[f64], mu: &[f64], w: &[f64], start: f64) -> Result<(f64, f64)> {
    let family = Family::Tweedie { power: p };
    let ll = |phi: f64| -> f64 {
        (0..y.len())
            .map(|i| family.log_likelihood(y[i], mu[i], w[i], phi))
            .sum()
    };
    let (a, b) = (start.ln() - 100f64.ln(), start.ln() + 100f64.ln());
    let (ln_phi, _, _) = brent(a, b, 1e-10, |t| -ll(t.exp()));
    let phi = ln_phi.exp();
    let value = ll(phi);
    if !value.is_finite() {
        return Err(Error::Data(format!(
            "the Tweedie log-likelihood is not finite at power {p}"
        )));
    }
    Ok((phi, value))
}

/// Golden-section search for the maximum of a unimodal `f` on `[a, b]`.
fn golden_max(mut a: f64, mut b: f64, tol: f64, f: impl Fn(f64) -> f64) -> f64 {
    let r = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = b - r * (b - a);
    let mut d = a + r * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    while (b - a).abs() > tol * (1.0 + c.abs()) {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - r * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + r * (b - a);
            fd = f(d);
        }
    }
    (a + b) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_finds_a_parabola_peak() {
        let x = golden_max(-3.0, 5.0, 1e-12, |x| -(x - 1.25).powi(2));
        assert!((x - 1.25).abs() < 1e-9);
    }

    /// Positive, gamma-like responses push the power to its upper bound.
    #[test]
    fn flags_a_power_at_the_boundary() {
        use prospicio_prob::{Distribution, Gamma};
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| f64::from(i % 2)).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let u = (f64::from(i) * 0.618_034).fract() * 0.98 + 0.01;
                Gamma::new(4.0, 25.0 * (1.0 + x[i as usize]))
                    .unwrap()
                    .quantile(u)
                    .unwrap()
            })
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; n as usize], x],
        )
        .unwrap();
        let fit = TweedieGlm::new(Link::Log).fit(&d, &y).unwrap();
        assert!(fit.at_boundary() && fit.power() > 1.98, "{}", fit.power());
        assert!(fit.interval_open().1);
        let mut bad = TweedieGlm::new(Link::Log);
        bad.max_power = 2.0;
        assert!(bad.fit(&d, &y).is_err());
    }

    #[test]
    fn rejects_unordered_powers() {
        let d = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; 3]]).unwrap();
        assert!(tweedie_profile(Link::Log, &d, &[1.0, 0.0, 2.0], &[1.5, 1.3]).is_err());
    }
}
