//! [`Dist`], the closed enum of every native distribution, for the places
//! where the family is chosen at run time: the Python and R bindings,
//! serialization and model outputs (`docs/design/distributions.md`,
//! "Static vs dynamic dispatch").
//!
//! Hot loops stay generic over `D: Distribution` and monomorphize; `Dist`
//! dispatches once per call with a `match`, not through a vtable.

use std::sync::Arc;

use act_core::{Result, StreamRng};

use crate::distribution::Distribution;
use crate::evt::Gpd;
use crate::severity::Severity;
use crate::{
    Custom, Gamma, Grid, LogAffinePareto, Loglogistic, Lognormal, Mixture, Pareto, PiecewisePareto,
    Sampled, Tweedie, Weibull,
};

/// Any native univariate distribution.
///
/// Every variant is a [`Distribution`]; all but [`Dist::Sampled`] are also
/// a [`Severity`], which [`Dist::as_severity`] exposes. A `Mixture` is held
/// in an [`Arc`] (its components are trait objects), so cloning a `Dist`
/// never copies one.
///
/// ```
/// use act_prob::{Dist, Distribution, Lognormal, Sampled};
///
/// let dists = vec![
///     Dist::from(Lognormal::from_mean_cv(1000.0, 0.5).unwrap()),
///     Dist::from(Sampled::new(vec![800.0, 1000.0, 1200.0]).unwrap()),
/// ];
/// for d in &dists {
///     assert!((d.mean() - 1000.0).abs() < 1e-9);
/// }
/// // Limited expected values exist for the parametric family only.
/// assert!(dists[0].as_severity().is_some());
/// assert!(dists[1].as_severity().is_none());
/// ```
#[derive(Debug, Clone)]
pub enum Dist {
    Lognormal(Lognormal),
    Pareto(Pareto),
    PiecewisePareto(PiecewisePareto),
    LogAffinePareto(LogAffinePareto),
    GeneralizedPareto(Gpd),
    Gamma(Gamma),
    Tweedie(Tweedie),
    Weibull(Weibull),
    Loglogistic(Loglogistic),
    Mixture(Arc<Mixture>),
    Grid(Grid),
    Sampled(Sampled),
    /// A user-defined severity (a Python or R callback): the slow path.
    Custom(Custom),
}

/// Calls `$call` on the inner distribution, whichever it is.
macro_rules! each {
    ($self:ident, $d:ident => $call:expr) => {
        match $self {
            Dist::Lognormal($d) => $call,
            Dist::Pareto($d) => $call,
            Dist::PiecewisePareto($d) => $call,
            Dist::LogAffinePareto($d) => $call,
            Dist::GeneralizedPareto($d) => $call,
            Dist::Gamma($d) => $call,
            Dist::Tweedie($d) => $call,
            Dist::Weibull($d) => $call,
            Dist::Loglogistic($d) => $call,
            Dist::Mixture($d) => $call,
            Dist::Grid($d) => $call,
            Dist::Sampled($d) => $call,
            Dist::Custom($d) => $call,
        }
    };
}

impl Dist {
    /// Short name of the family: `"lognormal"`, `"pareto"`,
    /// `"piecewise_pareto"`, `"log_affine_pareto"`, `"generalized_pareto"`,
    /// `"gamma"`, `"tweedie"`, `"weibull"`, `"loglogistic"`, `"mixture"`,
    /// `"grid"`, `"sampled"` or `"custom"`.
    pub fn family(&self) -> &'static str {
        match self {
            Self::Lognormal(_) => "lognormal",
            Self::Pareto(_) => "pareto",
            Self::PiecewisePareto(_) => "piecewise_pareto",
            Self::LogAffinePareto(_) => "log_affine_pareto",
            Self::GeneralizedPareto(_) => "generalized_pareto",
            Self::Gamma(_) => "gamma",
            Self::Tweedie(_) => "tweedie",
            Self::Weibull(_) => "weibull",
            Self::Loglogistic(_) => "loglogistic",
            Self::Mixture(_) => "mixture",
            Self::Grid(_) => "grid",
            Self::Sampled(_) => "sampled",
            Self::Custom(_) => "custom",
        }
    }

    /// The distribution as a [`Severity`] (limited expected values, layers),
    /// or `None` for [`Dist::Sampled`]: draws have no exact layer moments
    /// (`docs/design/distributions.md`).
    pub fn as_severity(&self) -> Option<&(dyn Severity + Send + Sync)> {
        Some(match self {
            Self::Lognormal(d) => d,
            Self::Pareto(d) => d,
            Self::PiecewisePareto(d) => d,
            Self::LogAffinePareto(d) => d,
            Self::GeneralizedPareto(d) => d,
            Self::Gamma(d) => d,
            Self::Tweedie(d) => d,
            Self::Weibull(d) => d,
            Self::Loglogistic(d) => d,
            Self::Mixture(d) => d.as_ref(),
            Self::Grid(d) => d,
            Self::Custom(d) => d,
            Self::Sampled(_) => return None,
        })
    }
}

impl Distribution for Dist {
    fn mean(&self) -> f64 {
        each!(self, d => d.mean())
    }

    fn variance(&self) -> f64 {
        each!(self, d => d.variance())
    }

    fn cdf(&self, x: f64) -> f64 {
        each!(self, d => d.cdf(x))
    }

    fn survival(&self, x: f64) -> f64 {
        each!(self, d => d.survival(x))
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        each!(self, d => d.quantile(p))
    }

    fn sample(&self, rng: &mut StreamRng, n: usize) -> Vec<f64> {
        each!(self, d => d.sample(rng, n))
    }

    fn is_parallel_safe(&self) -> bool {
        each!(self, d => d.is_parallel_safe())
    }
}

macro_rules! from_family {
    ($($variant:ident($ty:ty)),* $(,)?) => {
        $(
            impl From<$ty> for Dist {
                fn from(d: $ty) -> Self {
                    Self::$variant(d)
                }
            }
        )*
    };
}

from_family!(
    Lognormal(Lognormal),
    Pareto(Pareto),
    PiecewisePareto(PiecewisePareto),
    LogAffinePareto(LogAffinePareto),
    GeneralizedPareto(Gpd),
    Gamma(Gamma),
    Tweedie(Tweedie),
    Weibull(Weibull),
    Loglogistic(Loglogistic),
    Grid(Grid),
    Sampled(Sampled),
    Custom(Custom),
);

/// A [`Dist`] known to be a [`Severity`]: every variant but
/// [`Dist::Sampled`].
///
/// This is what the bindings accept wherever a loss severity is required
/// (collective models, layers, copula marginals, mixture components). It
/// dispatches by `match`, like [`Dist`], with no vtable.
///
/// ```
/// use act_prob::{Dist, Lognormal, Sampled, Severity, SeverityDist};
///
/// let ln = Lognormal::from_mean_cv(1000.0, 0.5).unwrap();
/// let s = SeverityDist::try_from(Dist::from(ln)).unwrap();
/// assert_eq!(s.lev(800.0), ln.lev(800.0));
///
/// let draws = Dist::from(Sampled::new(vec![1.0, 2.0]).unwrap());
/// assert!(SeverityDist::try_from(draws).is_err());
/// ```
#[derive(Debug, Clone)]
pub struct SeverityDist(Dist);

impl SeverityDist {
    /// The distribution.
    pub fn dist(&self) -> &Dist {
        &self.0
    }

    /// The distribution, by value.
    pub fn into_dist(self) -> Dist {
        self.0
    }
}

impl TryFrom<Dist> for SeverityDist {
    /// The distribution back, when it is [`Dist::Sampled`].
    type Error = Dist;

    fn try_from(d: Dist) -> std::result::Result<Self, Dist> {
        match d {
            Dist::Sampled(_) => Err(d),
            d => Ok(Self(d)),
        }
    }
}

impl From<SeverityDist> for Dist {
    fn from(s: SeverityDist) -> Self {
        s.0
    }
}

/// Calls `$call` on the inner severity, whichever it is.
macro_rules! each_severity {
    ($self:ident, $d:ident => $call:expr) => {
        match &$self.0 {
            Dist::Lognormal($d) => $call,
            Dist::Pareto($d) => $call,
            Dist::PiecewisePareto($d) => $call,
            Dist::LogAffinePareto($d) => $call,
            Dist::GeneralizedPareto($d) => $call,
            Dist::Gamma($d) => $call,
            Dist::Tweedie($d) => $call,
            Dist::Weibull($d) => $call,
            Dist::Loglogistic($d) => $call,
            Dist::Mixture($d) => $call,
            Dist::Grid($d) => $call,
            Dist::Custom($d) => $call,
            Dist::Sampled(_) => unreachable!("SeverityDist never holds Sampled"),
        }
    };
}

impl Distribution for SeverityDist {
    fn mean(&self) -> f64 {
        self.0.mean()
    }

    fn variance(&self) -> f64 {
        self.0.variance()
    }

    fn cdf(&self, x: f64) -> f64 {
        self.0.cdf(x)
    }

    fn survival(&self, x: f64) -> f64 {
        self.0.survival(x)
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        self.0.quantile(p)
    }

    fn sample(&self, rng: &mut StreamRng, n: usize) -> Vec<f64> {
        self.0.sample(rng, n)
    }

    fn is_parallel_safe(&self) -> bool {
        self.0.is_parallel_safe()
    }
}

impl Severity for SeverityDist {
    fn lev(&self, limit: f64) -> f64 {
        each_severity!(self, d => d.lev(limit))
    }

    fn stop_loss(&self, retention: f64) -> f64 {
        each_severity!(self, d => d.stop_loss(retention))
    }

    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        each_severity!(self, d => d.layer(limit, attachment))
    }

    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        each_severity!(self, d => d.layer_second_moment(limit, attachment))
    }

    fn layer_variance(&self, limit: f64, attachment: f64) -> f64 {
        each_severity!(self, d => d.layer_variance(limit, attachment))
    }
}

impl From<Mixture> for Dist {
    fn from(d: Mixture) -> Self {
        Self::Mixture(Arc::new(d))
    }
}

impl From<Arc<Mixture>> for Dist {
    fn from(d: Arc<Mixture>) -> Self {
        Self::Mixture(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Vec<Dist> {
        let ln = Lognormal::from_mean_cv(1000.0, 0.8).unwrap();
        vec![
            ln.into(),
            Pareto::new(500.0, 2.5).unwrap().into(),
            Gamma::new(2.0, 500.0).unwrap().into(),
            Weibull::new(1.5, 1000.0).unwrap().into(),
            Loglogistic::new(4.0, 900.0).unwrap().into(),
            Tweedie::new(1000.0, 2.0, 1.5).unwrap().into(),
            Gpd::new(0.2, 300.0).unwrap().into(),
            Mixture::new(vec![
                (0.5, Box::new(ln) as Box<dyn Severity + Send + Sync>),
                (0.5, Box::new(Gamma::new(2.0, 500.0).unwrap())),
            ])
            .unwrap()
            .into(),
            Sampled::new(vec![1.0, 2.0, 3.0, 10.0]).unwrap().into(),
            Custom::new(
                "exponential",
                Arc::new(|x: f64| Ok((1.0 - (-x / 500.0).exp()).max(0.0))),
                None,
                false,
            )
            .unwrap()
            .into(),
        ]
    }

    #[test]
    fn dispatch_matches_the_family() {
        let ln = Lognormal::from_mean_cv(1000.0, 0.8).unwrap();
        let d = Dist::from(ln);
        assert_eq!(d.family(), "lognormal");
        assert_eq!(d.mean(), ln.mean());
        assert_eq!(d.variance(), ln.variance());
        assert_eq!(d.cdf(700.0), ln.cdf(700.0));
        assert_eq!(d.quantile(0.9).unwrap(), ln.quantile(0.9).unwrap());
        let s = d.as_severity().unwrap();
        assert_eq!(s.lev(1500.0), ln.lev(1500.0));
        assert_eq!(s.layer(500.0, 1000.0), ln.layer(500.0, 1000.0));
        let (mut a, mut b) = (StreamRng::new(1, 2), StreamRng::new(1, 2));
        assert_eq!(d.sample(&mut a, 5), ln.sample(&mut b, 5));
    }

    #[test]
    fn every_variant_is_a_distribution_and_severities_have_layers() {
        for d in all() {
            assert!(d.mean().is_finite(), "{}", d.family());
            assert_eq!(d.is_parallel_safe(), d.family() != "custom");
            let q = d.quantile(0.5).unwrap();
            assert!(d.cdf(q) >= 0.5 - 1e-9, "{}", d.family());
            match d.as_severity() {
                Some(s) => {
                    let m = s.lev(f64::INFINITY);
                    assert!((m / d.mean() - 1.0).abs() < 1e-6, "{}", d.family());
                }
                None => assert_eq!(d.family(), "sampled"),
            }
            // Clones share a mixture rather than copying it.
            let c = d.clone();
            assert_eq!(c.mean(), d.mean());
        }
    }

    #[test]
    fn severity_dist_matches_as_severity_and_rejects_sampled() {
        for d in all() {
            let family = d.family();
            match SeverityDist::try_from(d.clone()) {
                Ok(s) => {
                    let r = d.as_severity().unwrap();
                    assert_eq!(s.lev(700.0), r.lev(700.0), "{family}");
                    assert_eq!(s.stop_loss(700.0), r.stop_loss(700.0), "{family}");
                    assert_eq!(s.layer(500.0, 200.0), r.layer(500.0, 200.0), "{family}");
                    assert_eq!(
                        s.layer_second_moment(500.0, 200.0),
                        r.layer_second_moment(500.0, 200.0),
                        "{family}"
                    );
                    assert_eq!(s.mean(), d.mean(), "{family}");
                    assert_eq!(s.dist().family(), family);
                }
                Err(back) => {
                    assert_eq!(family, "sampled");
                    assert_eq!(back.family(), "sampled");
                }
            }
        }
    }
}
