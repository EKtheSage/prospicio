//! Choosing a Tweedie GLM's power by profile likelihood.
//!
//! A Tweedie GLM fixes the power `p` (and its dispersion `φ` only scales the
//! variance), so IRLS gives the same coefficients for any `φ` but different
//! ones for each `p`. The power is chosen by maximizing the profile
//! log-likelihood: for each `p`, fit the GLM, then maximize the exact Tweedie
//! log-likelihood (Dunn and Smyth's series density, from `act-prob`) over
//! `φ`. This is the method of R's `tweedie.profile` (Dunn and Smyth, 2005,
//! 2008).

use act_core::{Error, Result};
use act_models::{Design, Family, Link, Model};

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
/// use act_glm::tweedie::tweedie_profile;
/// use act_models::{Design, Family, Link, Model};
/// use act_prob::{Distribution, Tweedie};
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
    let ln_phi = golden_max(a, b, 1e-10, |t| ll(t.exp()));
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

    #[test]
    fn rejects_unordered_powers() {
        let d = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; 3]]).unwrap();
        assert!(tweedie_profile(Link::Log, &d, &[1.0, 0.0, 2.0], &[1.5, 1.3]).is_err());
    }
}
