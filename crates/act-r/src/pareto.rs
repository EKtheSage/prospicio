//! Probability lane: wrappers over the `act_prob` Pareto family, the gamma
//! and Tweedie distributions, and claim counts by dispersion, for the R `pareto.R` API
//! (`docs/design/pareto.md`). Numeric arguments are vectors where R users
//! expect them; a truncation of `Inf` means none.

use act_core::StreamRng;
use act_prob::{Counting, Distribution, LargeLosses, Severity, Truncation};
use extendr_api::prelude::*;
use extendr_api::{Error, Result};

use crate::{to_r, whole};

/// The methods every Pareto-family severity shares, generated into the
/// type's single `#[extendr] impl` block after its own items.
macro_rules! severity_class {
    ($ty:ident { $($own:tt)* }) => {
        #[extendr]
        impl $ty {
            $($own)*

            fn mean(&self) -> f64 {
                self.inner.mean()
            }

            fn variance(&self) -> f64 {
                self.inner.variance()
            }

            fn cdf(&self, x: &[f64]) -> Vec<f64> {
                x.iter().map(|&x| self.inner.cdf(x)).collect()
            }

            fn survival(&self, x: &[f64]) -> Vec<f64> {
                x.iter().map(|&x| self.inner.survival(x)).collect()
            }

            fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
                p.iter().map(|&p| self.inner.quantile(p).map_err(to_r)).collect()
            }

            fn lev(&self, limit: &[f64]) -> Vec<f64> {
                limit.iter().map(|&l| self.inner.lev(l)).collect()
            }

            fn stop_loss(&self, retention: &[f64]) -> Vec<f64> {
                retention.iter().map(|&d| self.inner.stop_loss(d)).collect()
            }

            fn layer(&self, limit: f64, attachment: f64) -> f64 {
                self.inner.layer(limit, attachment)
            }

            fn layer_variance(&self, limit: f64, attachment: f64) -> f64 {
                self.inner.layer_variance(limit, attachment)
            }

            fn sample(&self, n: f64, seed: f64, stream: f64) -> Result<Vec<f64>> {
                let n = whole(n, "n")?;
                let mut rng = StreamRng::new(whole(seed, "seed")?, whole(stream, "stream")?);
                Ok(self.inner.sample(&mut rng, n as usize))
            }
        }
    };
}

/// `None` for an infinite truncation.
fn truncation(t: f64) -> Option<f64> {
    t.is_finite().then_some(t)
}

/// Large-loss data; empty vectors mean "not given", censoring arrives as
/// 0/1 doubles.
fn large_losses(
    losses: &[f64],
    reporting: &[f64],
    censored: &[f64],
    weights: &[f64],
) -> Result<LargeLosses> {
    let mut data = LargeLosses::new(losses.to_vec()).map_err(to_r)?;
    if !reporting.is_empty() {
        data = data
            .reporting_thresholds(reporting.to_vec())
            .map_err(to_r)?;
    }
    if !censored.is_empty() {
        data = data
            .censored(censored.iter().map(|&c| c != 0.0).collect())
            .map_err(to_r)?;
    }
    if !weights.is_empty() {
        data = data.weights(weights.to_vec()).map_err(to_r)?;
    }
    Ok(data)
}

fn truncation_kind(kind: &str) -> Result<Truncation> {
    match kind {
        "lp" => Ok(Truncation::LastPiece),
        "wd" => Ok(Truncation::WholeDistribution),
        _ => Err(Error::Other(
            "truncation_type must be \"lp\" (last piece) or \"wd\" (whole distribution)".into(),
        )),
    }
}

/// Single-parameter Pareto, optionally truncated.
#[extendr]
pub(crate) struct Pareto {
    pub(crate) inner: act_prob::Pareto,
}

severity_class!(Pareto {
    fn new(t: f64, alpha: f64, truncation_at: f64) -> Result<Self> {
        let p = act_prob::Pareto::new(t, alpha).map_err(to_r)?;
        let inner = match truncation(truncation_at) {
            None => p,
            Some(tr) => p.truncated(tr).map_err(to_r)?,
        };
        Ok(Self { inner })
    }

    fn fit(
        losses: &[f64],
        t: f64,
        reporting: &[f64],
        censored: &[f64],
        weights: &[f64],
        truncation_at: f64,
    ) -> Result<Self> {
        let data = large_losses(losses, reporting, censored, weights)?;
        let inner = act_prob::Pareto::fit(t, &data, truncation(truncation_at)).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn t(&self) -> f64 {
        self.inner.t()
    }

    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    fn truncation(&self) -> f64 {
        self.inner.truncation().unwrap_or(f64::INFINITY)
    }
});

/// Piecewise Pareto, optionally truncated.
#[extendr]
pub(crate) struct PiecewisePareto {
    pub(crate) inner: act_prob::PiecewisePareto,
}

severity_class!(PiecewisePareto {
    fn new(t: &[f64], alpha: &[f64], truncation_at: f64, truncation_type: &str) -> Result<Self> {
        let kind = truncation_kind(truncation_type)?;
        let pp = act_prob::PiecewisePareto::new(t.to_vec(), alpha.to_vec()).map_err(to_r)?;
        let inner = match truncation(truncation_at) {
            None => pp,
            Some(tr) => pp.truncated(tr, kind).map_err(to_r)?,
        };
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn fit(
        losses: &[f64],
        t: &[f64],
        reporting: &[f64],
        censored: &[f64],
        weights: &[f64],
        truncation_at: f64,
        truncation_type: &str,
    ) -> Result<Self> {
        let data = large_losses(losses, reporting, censored, weights)?;
        let kind = truncation_kind(truncation_type)?;
        let truncation = truncation(truncation_at).map(|tr| (tr, kind));
        let inner = act_prob::PiecewisePareto::fit(t.to_vec(), &data, truncation).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn t(&self) -> Vec<f64> {
        self.inner.thresholds().to_vec()
    }

    fn alpha(&self) -> Vec<f64> {
        self.inner.alphas().to_vec()
    }

    fn truncation(&self) -> f64 {
        self.inner.truncation().map_or(f64::INFINITY, |(tr, _)| tr)
    }

    fn truncation_type(&self) -> &'static str {
        match self.inner.truncation() {
            Some((_, Truncation::WholeDistribution)) => "wd",
            _ => "lp",
        }
    }
});

/// Log-affine local Pareto.
#[extendr]
pub(crate) struct LogAffinePareto {
    pub(crate) inner: act_prob::LogAffinePareto,
}

severity_class!(LogAffinePareto {
    fn new(t: f64, alpha0: f64, gamma: f64) -> Result<Self> {
        let inner = act_prob::LogAffinePareto::new(t, alpha0, gamma).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn from_delta(t: f64, alpha0: f64, delta: f64) -> Result<Self> {
        let inner = act_prob::LogAffinePareto::from_delta(t, alpha0, delta).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn t(&self) -> f64 {
        self.inner.t()
    }

    fn alpha0(&self) -> f64 {
        self.inner.alpha0()
    }

    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    fn delta(&self) -> f64 {
        self.inner.delta()
    }

    fn local_alpha(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.inner.local_alpha(x)).collect()
    }
});

/// Generalized Pareto severity with a location.
#[extendr]
pub(crate) struct GeneralizedPareto {
    pub(crate) inner: act_prob::evt::Gpd,
}

severity_class!(GeneralizedPareto {
    fn new(xi: f64, beta: f64, location: f64) -> Result<Self> {
        let inner = act_prob::evt::Gpd::new(xi, beta)
            .and_then(|g| g.shifted(location))
            .map_err(to_r)?;
        Ok(Self { inner })
    }

    fn riegel(t: f64, alpha_ini: f64, alpha_tail: f64) -> Result<Self> {
        let inner = act_prob::evt::Gpd::riegel(t, alpha_ini, alpha_tail).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn fit_riegel(
        losses: &[f64],
        t: f64,
        reporting: &[f64],
        censored: &[f64],
        weights: &[f64],
    ) -> Result<Self> {
        let data = large_losses(losses, reporting, censored, weights)?;
        let inner = act_prob::evt::Gpd::fit_riegel(t, &data).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn xi(&self) -> f64 {
        self.inner.xi()
    }

    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    fn location(&self) -> f64 {
        self.inner.location()
    }
});

/// Binomial claim counts.
#[extendr]
pub(crate) struct Binomial {
    pub(crate) inner: act_prob::Binomial,
}

#[extendr]
impl Binomial {
    fn new(n: f64, p: f64) -> Result<Self> {
        let inner = act_prob::Binomial::new(whole(n, "n")?, p).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn n(&self) -> f64 {
        self.inner.n() as f64
    }

    fn p(&self) -> f64 {
        self.inner.p()
    }

    fn pmf(&self, k: &[f64]) -> Result<Vec<f64>> {
        k.iter()
            .map(|&k| Ok(self.inner.pmf(whole(k, "k")?)))
            .collect()
    }

    fn cdf(&self, k: &[f64]) -> Result<Vec<f64>> {
        k.iter()
            .map(|&k| Ok(self.inner.cdf(whole(k, "k")?)))
            .collect()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.quantile(p).map(|k| k as f64).map_err(to_r))
            .collect()
    }

    fn sample(&self, n: f64, seed: f64, stream: f64) -> Result<Vec<f64>> {
        let n = whole(n, "n")?;
        let mut rng = StreamRng::new(whole(seed, "seed")?, whole(stream, "stream")?);
        Ok(self
            .inner
            .sample(&mut rng, n as usize)
            .into_iter()
            .map(|k| k as f64)
            .collect())
    }
}

/// The claim count for a mean and dispersion, as
/// `list(kind, parameters...)` for the R layer to construct.
#[extendr]
fn claim_count_parameters(mean: f64, dispersion: f64) -> Result<List> {
    use act_prob::PanjerClass;
    Ok(
        match PanjerClass::from_mean_dispersion(mean, dispersion).map_err(to_r)? {
            PanjerClass::Binomial(b) => list!(kind = "binomial", n = b.n() as f64, p = b.p()),
            PanjerClass::Poisson(p) => list!(kind = "poisson", lambda = p.lambda()),
            PanjerClass::NegativeBinomial(nb) => {
                list!(kind = "negative_binomial", r = nb.r(), beta = nb.beta())
            }
        },
    )
}

/// Converts a local Pareto with local alpha given by the R function
/// `alpha` to a piecewise Pareto: `list(severity, max_relative_error,
/// approximated_to)`.
#[extendr]
fn local_pareto_convert(
    t: f64,
    alpha: Function,
    rel_tolerance: f64,
    stop_survival: f64,
    stop_at: f64,
) -> Result<List> {
    // An R error inside `alpha` becomes NaN for the Rust side, which
    // rejects it; the original error is returned instead.
    let failure = std::cell::RefCell::new(None);
    let call = |x: f64| -> f64 {
        let value = alpha.call(pairlist!(x)).and_then(|v| {
            v.as_real()
                .ok_or(Error::Other("alpha must return a number".into()))
        });
        match value {
            Ok(v) => v,
            Err(e) => {
                failure.borrow_mut().get_or_insert(e);
                f64::NAN
            }
        }
    };
    let options = act_prob::LocalParetoConversion {
        rel_tolerance,
        stop_survival,
        stop_at,
    };
    let result = act_prob::local_pareto_to_piecewise(t, call, options);
    if let Some(e) = failure.into_inner() {
        return Err(e);
    }
    let approx = result.map_err(to_r)?;
    Ok(list!(
        severity = PiecewisePareto {
            inner: approx.severity
        },
        max_relative_error = approx.max_relative_error,
        approximated_to = approx.approximated_to
    ))
}

/// Gamma distribution.
#[extendr]
pub(crate) struct GammaDist {
    pub(crate) inner: act_prob::Gamma,
}

severity_class!(GammaDist {
    fn new(shape: f64, scale: f64) -> Result<Self> {
        let inner = act_prob::Gamma::new(shape, scale).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn from_mean_cv(mean: f64, cv: f64) -> Result<Self> {
        let inner = act_prob::Gamma::from_mean_cv(mean, cv).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn from_mean_dispersion(mean: f64, dispersion: f64) -> Result<Self> {
        let inner = act_prob::Gamma::from_mean_dispersion(mean, dispersion).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn shape(&self) -> f64 {
        self.inner.shape()
    }

    fn scale(&self) -> f64 {
        self.inner.scale()
    }

    fn ln_pdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.inner.ln_pdf(x)).collect()
    }
});

/// Tweedie (compound Poisson-gamma) distribution.
#[extendr]
pub(crate) struct TweedieDist {
    pub(crate) inner: act_prob::Tweedie,
}

severity_class!(TweedieDist {
    fn new(mean: f64, dispersion: f64, power: f64) -> Result<Self> {
        let inner = act_prob::Tweedie::new(mean, dispersion, power).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn from_poisson_gamma(lambda: f64, shape: f64, scale: f64) -> Result<Self> {
        let inner = act_prob::Tweedie::from_poisson_gamma(lambda, shape, scale).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    fn power(&self) -> f64 {
        self.inner.power()
    }

    fn lambda(&self) -> f64 {
        self.inner.lambda()
    }

    fn severity(&self) -> GammaDist {
        GammaDist {
            inner: self.inner.severity(),
        }
    }

    fn ln_pdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.inner.ln_pdf(x)).collect()
    }
});

/// Loglogistic distribution.
#[extendr]
pub(crate) struct LoglogisticDist {
    pub(crate) inner: act_prob::Loglogistic,
}

severity_class!(LoglogisticDist {
    fn new(shape: f64, scale: f64) -> Result<Self> {
        let inner = act_prob::Loglogistic::new(shape, scale).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn shape(&self) -> f64 {
        self.inner.shape()
    }

    fn scale(&self) -> f64 {
        self.inner.scale()
    }
});

/// Weibull distribution.
#[extendr]
pub(crate) struct WeibullDist {
    pub(crate) inner: act_prob::Weibull,
}

severity_class!(WeibullDist {
    fn new(shape: f64, scale: f64) -> Result<Self> {
        let inner = act_prob::Weibull::new(shape, scale).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn shape(&self) -> f64 {
        self.inner.shape()
    }

    fn scale(&self) -> f64 {
        self.inner.scale()
    }
});

/// A finite mixture of severities.
#[extendr]
pub(crate) struct MixtureDist {
    pub(crate) inner: std::sync::Arc<act_prob::Mixture>,
}

severity_class!(MixtureDist {
    /// `components` is a list of severity pointers, `weights` their
    /// probabilities.
    fn new(weights: &[f64], components: List) -> Result<Self> {
        if weights.len() != components.len() {
            return Err(Error::Other("give one weight per component".into()));
        }
        let parts = weights
            .iter()
            .zip(components.values())
            .map(|(&w, c)| {
                let sev = crate::distributions::severity_from_robj(&c)?;
                Ok((w, Box::new(sev) as Box<dyn Severity + Send + Sync>))
            })
            .collect::<Result<Vec<_>>>()?;
        let inner = act_prob::Mixture::new(parts).map_err(to_r)?;
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    fn weights(&self) -> Vec<f64> {
        self.inner.weights().to_vec()
    }
});

extendr_module! {
    mod pareto;
    fn local_pareto_convert;
    impl Pareto;
    impl PiecewisePareto;
    impl LogAffinePareto;
    impl GeneralizedPareto;
    impl GammaDist;
    impl TweedieDist;
    impl WeibullDist;
    impl LoglogisticDist;
    impl MixtureDist;
    impl Binomial;
    fn claim_count_parameters;
}
