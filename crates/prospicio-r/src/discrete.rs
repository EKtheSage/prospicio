//! Probability lane: a distribution on finitely many values of either sign
//! (`prospicio_prob::Discrete`) for R's `discrete.R`.

use extendr_api::Result;
use extendr_api::prelude::*;
use prospicio_prob::discrete::RETURN_PERIODS;
use prospicio_prob::{Discrete, Distribution};

use crate::to_r;

/// A discrete distribution on any finite set of values.
#[extendr]
pub(crate) struct DiscreteDist {
    pub(crate) inner: Discrete,
}

#[extendr]
impl DiscreteDist {
    fn new(values: &[f64], probs: &[f64]) -> Result<Self> {
        Ok(Self {
            inner: Discrete::new(values.to_vec(), probs.to_vec()).map_err(to_r)?,
        })
    }

    fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    fn probs(&self) -> Vec<f64> {
        self.inner.probs().to_vec()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// NA when the variance is 0.
    fn skewness(&self) -> f64 {
        self.inner.skewness().unwrap_or(f64::NAN)
    }

    fn cdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.cdf(v)).collect()
    }

    fn survival(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.survival(v)).collect()
    }

    fn pmf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.pmf(v)).collect()
    }

    fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.quantile(p).map_err(to_r))
            .collect()
    }

    /// NA where there is none (`p = 1`).
    fn quantile_upper(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| {
                self.inner
                    .quantile_upper(p)
                    .map(|q| q.unwrap_or(f64::NAN))
                    .map_err(to_r)
            })
            .collect()
    }

    fn tvar(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.tvar(p).map_err(to_r))
            .collect()
    }

    /// The return-period table; an empty `periods` for `aggregate`'s.
    fn tail(&self, periods: &[f64]) -> Result<List> {
        let periods = if periods.is_empty() {
            RETURN_PERIODS.to_vec()
        } else {
            periods.to_vec()
        };
        let rows = self.inner.tail(&periods).map_err(to_r)?;
        let col =
            |f: fn(&prospicio_prob::TailRow) -> f64| -> Vec<f64> { rows.iter().map(f).collect() };
        Ok(list!(
            p = col(|r| r.p),
            period = col(|r| r.period),
            var = col(|r| r.var),
            tvar = col(|r| r.tvar),
            excess = col(|r| r.excess),
            var_to_mean = col(|r| r.var_to_mean.unwrap_or(f64::NAN))
        ))
    }
}

extendr_module! {
    mod discrete;
    impl DiscreteDist;
}
