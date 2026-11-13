//! Link functions: `η = g(μ)` between the mean and the linear predictor.

use prospicio_math::special::{norm_cdf, norm_pdf, norm_quantile};

/// A link function `η = g(μ)`, with its inverse and derivative.
///
/// A closed enum, so a fitted model's link serializes as data.
///
/// ```
/// use prospicio_models::Link;
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

    /// `d²μ/dη²` at `η`, for the observed information of a non-canonical
    /// link.
    pub fn mu_eta2(&self, eta: f64) -> f64 {
        match *self {
            Self::Identity => 0.0,
            Self::Log | Self::Power(0.0) => eta.exp(),
            Self::Logit => {
                let mu = self.inverse(eta);
                self.mu_eta(eta) * (1.0 - 2.0 * mu)
            }
            Self::Probit => -eta * norm_pdf(eta),
            Self::Cloglog => self.mu_eta(eta) * (1.0 - eta.exp()),
            Self::Inverse => 2.0 / (eta * eta * eta),
            Self::InverseSquared => 0.75 * eta.powf(-2.5),
            Self::Power(l) => (1.0 / l - 1.0) / l * eta.powf(1.0 / l - 2.0),
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

/// The kind of a [`Link`], without the power link's exponent: what the
/// Python, R and Excel front ends read from a link's name (#834).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkKind {
    Identity,
    Log,
    Logit,
    Probit,
    Cloglog,
    Inverse,
    InverseSquared,
    Power,
}

impl LinkKind {
    /// Every name the Python, R and Excel front ends accept, with its kind,
    /// the canonical name of each first (#834: one vocabulary, one edit for
    /// a new variant).
    pub const NAMES: &'static [(&'static str, Self)] = &[
        ("identity", Self::Identity),
        ("log", Self::Log),
        ("logit", Self::Logit),
        ("probit", Self::Probit),
        ("cloglog", Self::Cloglog),
        ("inverse", Self::Inverse),
        ("inverse_squared", Self::InverseSquared),
        ("power", Self::Power),
    ];

    /// The kind named `name` ([`NAMES`](Self::NAMES)), or `None`.
    ///
    /// ```
    /// use prospicio_models::link::LinkKind;
    ///
    /// assert_eq!(LinkKind::from_name("cloglog"), Some(LinkKind::Cloglog));
    /// assert_eq!(LinkKind::from_name("sqrt"), None);
    /// ```
    pub fn from_name(name: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|&(_, k)| k)
    }

    /// The canonical name, the first [`NAMES`](Self::NAMES) entry of the
    /// kind.
    pub fn name(self) -> &'static str {
        Self::NAMES
            .iter()
            .find(|&&(_, k)| k == self)
            .map(|&(n, _)| n)
            .expect("every kind is named")
    }

    /// The link of this kind, `power` the exponent of [`Link::Power`] (read
    /// only for that kind).
    pub fn link(self, power: f64) -> Link {
        match self {
            Self::Identity => Link::Identity,
            Self::Log => Link::Log,
            Self::Logit => Link::Logit,
            Self::Probit => Link::Probit,
            Self::Cloglog => Link::Cloglog,
            Self::Inverse => Link::Inverse,
            Self::InverseSquared => Link::InverseSquared,
            Self::Power => Link::Power(power),
        }
    }
}

impl Link {
    /// The link's kind, without the power link's exponent.
    pub fn kind(&self) -> LinkKind {
        match self {
            Self::Identity => LinkKind::Identity,
            Self::Log => LinkKind::Log,
            Self::Logit => LinkKind::Logit,
            Self::Probit => LinkKind::Probit,
            Self::Cloglog => LinkKind::Cloglog,
            Self::Inverse => LinkKind::Inverse,
            Self::InverseSquared => LinkKind::InverseSquared,
            Self::Power(_) => LinkKind::Power,
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
            let numeric2 = (g.mu_eta(eta + h) - g.mu_eta(eta - h)) / (2.0 * h);
            assert!(
                (g.mu_eta2(eta) - numeric2).abs() < 1e-7 * numeric2.abs().max(1.0),
                "{g:?}"
            );
        }
    }
}

#[cfg(test)]
mod names {
    use super::*;

    #[test]
    fn every_name_round_trips() {
        for &(name, kind) in LinkKind::NAMES {
            assert_eq!(LinkKind::from_name(name), Some(kind));
            assert_eq!(LinkKind::from_name(kind.name()), Some(kind), "{name}");
        }
        for &(_, kind) in LinkKind::NAMES {
            assert_eq!(kind.link(0.5).kind(), kind);
        }
    }
}
