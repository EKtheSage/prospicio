//! R bindings, built by `R CMD INSTALL R/actuarialrs`.
//!
//! Wrappers convert arguments and map [`act_core::Error`] to R errors. They
//! hold no numerical code; the idiomatic R API (functions and S3 methods)
//! lives in `R/actuarialrs/R`.

use act_core::StreamRng;
use act_prob::Distribution;
use extendr_api::prelude::*;
use extendr_api::{Error, Result};

fn to_r(e: act_core::Error) -> Error {
    Error::Other(e.to_string())
}

/// Lognormal distribution: `ln X ~ Normal(meanlog, sdlog^2)`.
#[extendr]
struct Lognormal {
    inner: act_prob::Lognormal,
}

#[extendr]
impl Lognormal {
    fn new(meanlog: f64, sdlog: f64) -> Result<Self> {
        let inner = act_prob::Lognormal::new(meanlog, sdlog).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn from_mean_cv(mean: f64, cv: f64) -> Result<Self> {
        let inner = act_prob::Lognormal::from_mean_cv(mean, cv).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn meanlog(&self) -> f64 {
        self.inner.meanlog()
    }

    fn sdlog(&self) -> f64 {
        self.inner.sdlog()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn cdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.inner.cdf(x)).collect()
    }

    fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.quantile(p).map_err(to_r))
            .collect()
    }

    /// `n` draws from stream `stream` of the generator keyed by `seed`. R has
    /// no 64-bit integers, so ids arrive as doubles holding whole numbers.
    fn sample(&self, n: f64, seed: f64, stream: f64) -> Result<Vec<f64>> {
        let n = whole(n, "n")?;
        let mut rng = StreamRng::new(whole(seed, "seed")?, whole(stream, "stream")?);
        Ok(self.inner.sample(&mut rng, n as usize))
    }
}

/// Converts an R double holding a non-negative whole number below 2^53.
fn whole(x: f64, name: &str) -> Result<u64> {
    if x.is_finite() && x >= 0.0 && x.fract() == 0.0 && x < 9_007_199_254_740_992.0 {
        Ok(x as u64)
    } else {
        Err(Error::Other(format!(
            "{name} must be a non-negative whole number below 2^53, got {x}"
        )))
    }
}

extendr_module! {
    mod actuarialrs;
    impl Lognormal;
}
