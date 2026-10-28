//! Probability lane: the normal distribution and the ogive
//! (`prospicio_prob::Normal`, `prospicio_prob::Ogive`) for R's `normal.R`.

use extendr_api::Result;
use extendr_api::prelude::*;
use prospicio_core::StreamRng;
use prospicio_prob::{Distribution, Normal, Ogive, Severity};

use crate::{to_r, whole};

/// The normal distribution.
#[extendr]
pub(crate) struct NormalDist {
    pub(crate) inner: Normal,
}

#[extendr]
impl NormalDist {
    fn new(mu: f64, sigma: f64) -> Result<Self> {
        Ok(Self {
            inner: Normal::new(mu, sigma).map_err(to_r)?,
        })
    }

    fn mu(&self) -> f64 {
        self.inner.mu()
    }

    fn sigma(&self) -> f64 {
        self.inner.sigma()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn ln_pdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.pdf(v).ln()).collect()
    }

    fn cdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.cdf(v)).collect()
    }

    fn survival(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.survival(v)).collect()
    }

    fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.quantile(p).map_err(to_r))
            .collect()
    }

    fn tvar(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.tvar(p).map_err(to_r))
            .collect()
    }

    fn sample(&self, n: f64, seed: f64, stream: f64) -> Result<Vec<f64>> {
        let n = whole(n, "n")?;
        let mut rng = StreamRng::new(whole(seed, "seed")?, whole(stream, "stream")?);
        Ok(self.inner.sample(&mut rng, n as usize))
    }
}

/// The piecewise linear distribution function of grouped data.
#[extendr]
pub(crate) struct OgiveDist {
    inner: Ogive,
}

#[extendr]
impl OgiveDist {
    fn new(x: &[f64], cdf: &[f64]) -> Result<Self> {
        Ok(Self {
            inner: Ogive::new(x.to_vec(), cdf.to_vec()).map_err(to_r)?,
        })
    }

    fn from_counts(bounds: &[f64], counts: &[f64]) -> Result<Self> {
        Ok(Self {
            inner: Ogive::from_counts(bounds, counts).map_err(to_r)?,
        })
    }

    fn x(&self) -> Vec<f64> {
        self.inner.x().to_vec()
    }

    fn cdf_values(&self) -> Vec<f64> {
        self.inner.f().to_vec()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn ln_pdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.pdf(v).ln()).collect()
    }

    fn cdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.cdf(v)).collect()
    }

    fn survival(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.survival(v)).collect()
    }

    fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.quantile(p).map_err(to_r))
            .collect()
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

    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        self.inner.layer_second_moment(limit, attachment)
    }

    fn sample(&self, n: f64, seed: f64, stream: f64) -> Result<Vec<f64>> {
        let n = whole(n, "n")?;
        let mut rng = StreamRng::new(whole(seed, "seed")?, whole(stream, "stream")?);
        Ok(self.inner.sample(&mut rng, n as usize))
    }
}

extendr_module! {
    mod normal;
    impl NormalDist;
    impl OgiveDist;
}
