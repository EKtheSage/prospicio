//! Probability lane: a law's distribution on a grid from its characteristic
//! function (`prospicio_prob::chf`, #779) for R's `chf.R`.

use std::sync::Arc;

use extendr_api::Result;
use extendr_api::prelude::*;
use prospicio_prob::Distribution;
use prospicio_prob::chf::{
    ChfDistribution, ChfFn, ChfLaw, ChfOptions, Complex64, Tilt, UserChf, invert,
};
use prospicio_prob::levy::{NormalInverseGaussian, TemperedStable, TweediePower, VarianceGamma};
use prospicio_prob::{Stable, StableParameterization};

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

/// An alpha-stable law's parameters.
#[extendr]
pub(crate) struct StableLaw {
    inner: Stable,
}

#[extendr]
impl StableLaw {
    /// `parameterization` is "S1" or "S0".
    fn new(alpha: f64, beta: f64, scale: f64, loc: f64, parameterization: &str) -> Result<Self> {
        let p = match parameterization.to_ascii_uppercase().as_str() {
            "S1" => StableParameterization::S1,
            "S0" => StableParameterization::S0,
            other => {
                return Err(Error::Other(format!(
                    "parameterization must be \"S1\" or \"S0\", got \"{other}\""
                )));
            }
        };
        Ok(Self {
            inner: Stable::new(alpha, beta, scale, loc, p).map_err(to_r)?,
        })
    }

    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    fn scale(&self) -> f64 {
        self.inner.scale()
    }

    fn loc(&self) -> f64 {
        self.inner.loc()
    }

    fn loc0(&self) -> f64 {
        self.inner.loc0()
    }

    fn tail_constant(&self) -> f64 {
        self.inner.tail_constant()
    }

    fn left_tail_constant(&self) -> f64 {
        self.inner.left_tail_constant()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }
}

/// The normal inverse Gaussian law's parameters.
#[extendr]
pub(crate) struct NigLaw {
    inner: NormalInverseGaussian,
}

#[extendr]
impl NigLaw {
    fn new(alpha: f64, beta: f64, delta: f64, mu: f64) -> Result<Self> {
        Ok(Self {
            inner: NormalInverseGaussian::new(alpha, beta, delta, mu).map_err(to_r)?,
        })
    }

    /// `c(alpha, beta, delta, mu)`.
    fn parameters(&self) -> Vec<f64> {
        let d = self.inner;
        vec![d.alpha(), d.beta(), d.delta(), d.mu()]
    }

    /// `c(mean, variance, skewness)`.
    fn moments(&self) -> Vec<f64> {
        vec![
            self.inner.mean(),
            self.inner.variance(),
            self.inner.skewness(),
        ]
    }
}

/// The variance gamma law's parameters.
#[extendr]
pub(crate) struct VarianceGammaLaw {
    inner: VarianceGamma,
}

#[extendr]
impl VarianceGammaLaw {
    fn new(sigma: f64, nu: f64, theta: f64, mu: f64) -> Result<Self> {
        Ok(Self {
            inner: VarianceGamma::new(sigma, nu, theta, mu).map_err(to_r)?,
        })
    }

    /// `c(sigma, nu, theta, mu)`.
    fn parameters(&self) -> Vec<f64> {
        let d = self.inner;
        vec![d.sigma(), d.nu(), d.theta(), d.mu()]
    }

    /// `c(mean, variance, skewness)`.
    fn moments(&self) -> Vec<f64> {
        vec![
            self.inner.mean(),
            self.inner.variance(),
            self.inner.skewness(),
        ]
    }
}

/// The CGMY tempered stable law's parameters.
#[extendr]
pub(crate) struct TemperedStableLaw {
    inner: TemperedStable,
}

#[extendr]
impl TemperedStableLaw {
    fn new(c: f64, g: f64, m: f64, y: f64, mu: f64) -> Result<Self> {
        Ok(Self {
            inner: TemperedStable::new(c, g, m, y, mu).map_err(to_r)?,
        })
    }

    /// `c(C, G, M, Y, mu)`.
    fn parameters(&self) -> Vec<f64> {
        let d = self.inner;
        vec![d.c(), d.g(), d.m(), d.y(), d.mu()]
    }

    /// `c(mean, variance, skewness)`.
    fn moments(&self) -> Vec<f64> {
        vec![
            self.inner.mean(),
            self.inner.variance(),
            self.inner.skewness(),
        ]
    }
}

/// The Tweedie law at any power.
#[extendr]
pub(crate) struct TweediePowerLaw {
    inner: TweediePower,
}

#[extendr]
impl TweediePowerLaw {
    /// `cv` is `NaN` when the dispersion is given.
    fn new(power: f64, mean: f64, dispersion: f64, cv: f64) -> Result<Self> {
        let inner = if cv.is_nan() {
            TweediePower::new(power, mean, dispersion)
        } else {
            TweediePower::from_mean_cv(power, mean, cv)
        }
        .map_err(to_r)?;
        Ok(Self { inner })
    }

    /// `c(power, mean, dispersion)`.
    fn parameters(&self) -> Vec<f64> {
        let d = self.inner;
        vec![d.power(), d.mean(), d.dispersion()]
    }

    /// `c(mean, variance, skewness)`.
    fn moments(&self) -> Vec<f64> {
        vec![
            self.inner.mean(),
            self.inner.variance(),
            self.inner.skewness(),
        ]
    }
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
    /// `tilt` is "auto", "off" or "fixed" with `tau`.
    fn new(
        law: Robj,
        log2: f64,
        step: f64,
        x_min: f64,
        bucket_sizing_p: f64,
        tilt: &str,
        tau: f64,
    ) -> Result<Self> {
        let tilt = match tilt {
            "auto" => Tilt::Auto,
            "off" => Tilt::Off,
            _ => Tilt::Fixed(tau),
        };
        let law = if let Ok(n) = <&NormalDist>::try_from(&law) {
            ChfLaw::Normal(n.inner)
        } else if let Ok(st) = <&StableLaw>::try_from(&law) {
            ChfLaw::Stable(st.inner)
        } else if let Ok(d) = <&NigLaw>::try_from(&law) {
            ChfLaw::NormalInverseGaussian(d.inner)
        } else if let Ok(d) = <&VarianceGammaLaw>::try_from(&law) {
            ChfLaw::VarianceGamma(d.inner)
        } else if let Ok(d) = <&TemperedStableLaw>::try_from(&law) {
            ChfLaw::TemperedStable(d.inner)
        } else if let Ok(d) = <&TweediePowerLaw>::try_from(&law) {
            ChfLaw::TweediePower(d.inner)
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
            tilt,
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
            tilt_left = r.tilt.0,
            tilt_right = r.tilt.1,
            edge_error = r.edge_error,
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
    impl StableLaw;
    impl NigLaw;
    impl VarianceGammaLaw;
    impl TemperedStableLaw;
    impl TweediePowerLaw;
}
