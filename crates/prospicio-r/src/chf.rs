//! Probability lane: a law's distribution on a grid from its characteristic
//! function (`prospicio_prob::chf`, #779) for R's `chf.R`.

use std::sync::Arc;

use extendr_api::Result;
use extendr_api::prelude::*;
use prospicio_prob::Distribution;
use prospicio_prob::chf::{ChfDistribution, ChfFn, ChfLaw, ChfOptions, Complex64, UserChf, invert};

use crate::distributions::{Grid, dist_from_robj};
use crate::normal::NormalDist;
use crate::{to_r, whole};

/// An R function of the frequencies returning `c(Re(phi), Im(phi))`, or
/// the message of an error it caught.
struct RChf(Function);

// SAFETY: R must only be entered from its main thread. The law that holds
// this is built with `parallel_safe = false`, and the inversion calls it on
// the calling thread, which for an R binding is R's main thread.
unsafe impl Send for RChf {}
unsafe impl Sync for RChf {}

impl RChf {
    fn call(&self, t: &[f64]) -> prospicio_core::Result<Robj> {
        self.0
            .call(pairlist!(t.to_vec()))
            .map_err(|e| prospicio_core::Error::Data(e.to_string()))
    }
}

fn r_chf(f: Function) -> Arc<ChfFn> {
    let f = RChf(f);
    Arc::new(move |t: &[f64]| {
        let out = f.call(t)?;
        if let Some(message) = out.as_str() {
            return Err(prospicio_core::Error::Data(message.to_string()));
        }
        let v = out.as_real_slice().ok_or_else(|| {
            prospicio_core::Error::Data("the characteristic function must return numbers".into())
        })?;
        if v.len() != 2 * t.len() {
            return Err(prospicio_core::Error::Data(format!(
                "the characteristic function returned {} values for {} frequencies",
                v.len() / 2,
                t.len()
            )));
        }
        let n = t.len();
        Ok((0..n).map(|k| Complex64::new(v[k], v[n + k])).collect())
    })
}

/// A law's bucket masses from its characteristic function.
#[extendr]
pub(crate) struct ChfDist {
    inner: ChfDistribution,
}

#[extendr]
impl ChfDist {
    /// `law` is a distribution's pointer or an R function returning
    /// `c(Re, Im)`; `step` and `x_min` are `NaN` when not given.
    fn new(law: Robj, log2: f64, step: f64, x_min: f64, bucket_sizing_p: f64) -> Result<Self> {
        let law = if let Ok(n) = <&NormalDist>::try_from(&law) {
            ChfLaw::Normal(n.inner)
        } else if let Ok(f) = Function::try_from(law.clone()) {
            ChfLaw::User(UserChf::new(r_chf(f), "chf", false))
        } else {
            ChfLaw::from_dist(&dist_from_robj(&law)?).map_err(to_r)?
        };
        let options = ChfOptions {
            log2: whole(log2, "log2")? as u32,
            step: (!step.is_nan()).then_some(step),
            x_min: (!x_min.is_nan()).then_some(x_min),
            bucket_sizing_p,
        };
        Ok(Self {
            inner: invert(&law, &options).map_err(to_r)?,
        })
    }

    fn x_min(&self) -> f64 {
        self.inner.x_min()
    }

    fn step(&self) -> f64 {
        self.inner.step()
    }

    fn xs(&self) -> Vec<f64> {
        self.inner.xs()
    }

    fn masses(&self) -> Vec<f64> {
        self.inner.masses().to_vec()
    }

    fn report(&self) -> List {
        let r = self.inner.report();
        list!(
            log2 = r.log2 as f64,
            step = r.step,
            x_min = r.x_min,
            step_given = r.step_given,
            truncation_error = r.truncation_error,
            is_truncated = r.is_truncated(),
            outside_below = r.outside_below,
            outside_above = r.outside_above,
            total_mass = r.total_mass,
            law_mean = r.law_mean,
            law_variance = r.law_variance
        )
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    fn cdf(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.inner.cdf(v)).collect()
    }

    fn quantile(&self, p: &[f64]) -> Result<Vec<f64>> {
        p.iter()
            .map(|&p| self.inner.quantile(p).map_err(to_r))
            .collect()
    }

    fn to_grid(&self) -> Result<Grid> {
        Ok(Grid::wrap(self.inner.to_grid().map_err(to_r)?))
    }
}

extendr_module! {
    mod chf;
    impl ChfDist;
}
