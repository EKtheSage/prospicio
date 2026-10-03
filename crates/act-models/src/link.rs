//! Link functions: `η = g(μ)` between the mean and the linear predictor.

use act_math::special::{norm_cdf, norm_pdf, norm_quantile};

/// A link function `η = g(μ)`, with its inverse and derivative.
///
/// A closed enum, so a fitted model's link serializes as data.
///
/// ```
/// use act_models::Link;
///
/// let g = Link::Log;
/// assert!((g.inverse(g.link(3.0)) - 3.0).abs() < 1e-15);
/// assert_eq!(g.mu_eta(0.0), 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Link {
    /// `η = μ`.
    Identity,
    /// `η = ln μ`: multiplicative models, the actuarial default.
    Log,
    /// `η = ln(μ / (1 - μ))`.
    Logit,
    /// `η = Φ⁻¹(μ)`.
    Probit,
    /// `η = ln(-ln(1 - μ))`.
    Cloglog,
    /// `η = 1 / μ`, the gamma family's canonical link.
    Inverse,
    /// `η = 1 / μ²`, the inverse Gaussian's canonical link.
    InverseSquared,
    /// `η = μ^λ` for `λ ≠ 0` (`λ = 0` is the log link).
    Power(f64),
}

impl Link {
    /// `η = g(μ)`.
    pub fn link(&self, mu: f64) -> f64 {
        match *self {
            Self::Identity => mu,
            Self::Log => mu.ln(),
            Self::Logit => (mu / (1.0 - mu)).ln(),
            Self::Probit => norm_quantile(mu),
            Self::Cloglog => (-(-mu).ln_1p()).ln(),
            Self::Inverse => 1.0 / mu,
            Self::InverseSquared => 1.0 / (mu * mu),
            Self::Power(0.0) => mu.ln(),
            Self::Power(l) => mu.powf(l),
        }
    }

    /// `μ = g⁻¹(η)`.
    pub fn inverse(&self, eta: f64) -> f64 {
        match *self {
            Self::Identity => eta,
            Self::Log => eta.exp(),
            Self::Logit => 1.0 / (1.0 + (-eta).exp()),
            Self::Probit => norm_cdf(eta),
            Self::Cloglog => -(-eta.exp()).exp_m1(),
            Self::Inverse => 1.0 / eta,
            Self::InverseSquared => 1.0 / eta.sqrt(),
            Self::Power(0.0) => eta.exp(),
            Self::Power(l) => eta.powf(1.0 / l),
        }
    }

    /// `dμ/dη` at `η`.
    pub fn mu_eta(&self, eta: f64) -> f64 {
        match *self {
            Self::Identity => 1.0,
            Self::Log => eta.exp(),
            Self::Logit => {
                let e = (-eta.abs()).exp();
                e / ((1.0 + e) * (1.0 + e))
            }
            Self::Probit => norm_pdf(eta),
            Self::Cloglog => (eta - eta.exp()).exp(),
            Self::Inverse => -1.0 / (eta * eta),
            Self::InverseSquared => -0.5 * eta.powf(-1.5),
            Self::Power(0.0) => eta.exp(),
            Self::Power(l) => eta.powf(1.0 / l - 1.0) / l,
        }
    }

    /// Whether `η` maps to a mean the link can produce (for example, a
    /// positive `η` under the inverse link).
    pub fn valid_eta(&self, eta: f64) -> bool {
        eta.is_finite()
            && match *self {
                Self::Inverse | Self::InverseSquared => eta > 0.0,
                Self::Power(l) if l != 0.0 => eta > 0.0,
                _ => true,
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverses_and_derivatives_agree() {
        let links = [
            (Link::Identity, 2.5),
            (Link::Log, 2.5),
            (Link::Logit, 0.3),
            (Link::Probit, 0.3),
            (Link::Cloglog, 0.3),
            (Link::Inverse, 2.5),
            (Link::InverseSquared, 2.5),
            (Link::Power(0.5), 2.5),
            (Link::Power(0.0), 2.5),
        ];
        for (g, mu) in links {
            let eta = g.link(mu);
            assert!((g.inverse(eta) - mu).abs() < 1e-13, "{g:?}");
            let h = 1e-6;
            let numeric = (g.inverse(eta + h) - g.inverse(eta - h)) / (2.0 * h);
            assert!(
                (g.mu_eta(eta) - numeric).abs() < 1e-7 * numeric.abs().max(1.0),
                "{g:?}"
            );
        }
    }
}
