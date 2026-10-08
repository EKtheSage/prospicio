//! Probability lane: claim counts beyond the Poisson, negative binomial
//! and binomial (`prospicio_prob::count_families`) for R's
//! `distributions.R`.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_core::StreamRng;
use prospicio_prob::Counting;
use prospicio_prob::count_families::{
    CompoundPoisson, CountDist, EmpiricalCount, Logarithmic, MixedPoisson, Mixing, ZeroModified,
};

use crate::distributions::{NegativeBinomial, Poisson};
use crate::{to_r, whole};

/// Any count R holds, as one CountDist.
pub(crate) fn count_dist(obj: &Robj) -> Result<CountDist> {
    if let Ok(n) = <&Poisson>::try_from(obj) {
        return Ok(CountDist::Poisson(n.inner));
    }
    if let Ok(n) = <&NegativeBinomial>::try_from(obj) {
        return Ok(CountDist::NegativeBinomial(n.inner));
    }
    if let Ok(n) = <&crate::pareto::Binomial>::try_from(obj) {
        return Ok(CountDist::Binomial(n.inner));
    }
    if let Ok(n) = <&ClaimCount>::try_from(obj) {
        return Ok(n.inner.clone());
    }
    Err(Error::Other("expected a claim count".into()))
}

/// A claim count from the families beyond the Poisson, negative binomial
/// and binomial.
#[extendr]
pub(crate) struct ClaimCount {
    pub(crate) inner: CountDist,
}

#[extendr]
impl ClaimCount {
    fn zero_modified(base: Robj, p0: f64) -> Result<Self> {
        let z = ZeroModified::new(count_dist(&base)?, p0).map_err(to_r)?;
        Ok(Self {
            inner: CountDist::ZeroModified(Box::new(z)),
        })
    }

    fn logarithmic(p: f64) -> Result<Self> {
        Ok(Self {
            inner: CountDist::Logarithmic(Logarithmic::new(p).map_err(to_r)?),
        })
    }

    /// `mixing` is "gamma" or "inverse_gaussian".
    fn mixed_poisson(mean: f64, cv: f64, mixing: &str, shift: f64) -> Result<Self> {
        let mixing = match mixing {
            "gamma" => Mixing::Gamma { cv },
            "inverse_gaussian" => Mixing::InverseGaussian { cv },
            other => return Err(Error::Other(format!("unknown mixing {other:?}"))),
        };
        Ok(Self {
            inner: CountDist::MixedPoisson(MixedPoisson::new(mean, mixing, shift).map_err(to_r)?),
        })
    }

    fn compound_poisson(rate: f64, secondary: Robj) -> Result<Self> {
        let c = CompoundPoisson::new(rate, count_dist(&secondary)?).map_err(to_r)?;
        Ok(Self {
            inner: CountDist::CompoundPoisson(Box::new(c)),
        })
    }

    fn empirical(probs: &[f64]) -> Result<Self> {
        Ok(Self {
            inner: CountDist::Empirical(EmpiricalCount::new(probs.to_vec()).map_err(to_r)?),
        })
    }

    fn kind(&self) -> &'static str {
        match &self.inner {
            CountDist::ZeroModified(_) => "zero_modified",
            CountDist::Logarithmic(_) => "logarithmic",
            CountDist::MixedPoisson(_) => "mixed_poisson",
            CountDist::CompoundPoisson(_) => "compound_poisson",
            CountDist::Empirical(_) => "empirical",
            CountDist::Poisson(_) => "poisson",
            CountDist::NegativeBinomial(_) => "negative_binomial",
            CountDist::Binomial(_) => "binomial",
        }
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

extendr_module! {
    mod counts;
    impl ClaimCount;
}
