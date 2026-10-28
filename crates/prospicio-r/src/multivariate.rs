//! Probability lane: multivariate normal and lognormal joints
//! (`prospicio_prob::multivariate`) for R's `multivariate.R`.

use extendr_api::Result;
use extendr_api::prelude::*;
use prospicio_prob::{GaussianJoint, MultivariateLognormal, MultivariateNormal};

use crate::distributions::{Lognormal, PredictiveDistribution, components_from_keys};
use crate::normal::NormalDist;
use crate::{to_r, whole};

enum Joint {
    Normal(MultivariateNormal),
    Lognormal(MultivariateLognormal),
}

/// A multivariate normal or lognormal joint.
#[extendr]
pub(crate) struct GaussianJointR {
    inner: Joint,
}

/// A row-major `d × d` matrix from R's column-major one (they are equal
/// for a symmetric matrix, but a matrix that is not symmetric must be
/// reported as R shows it).
fn row_major(m: &[f64], d: usize) -> Vec<f64> {
    let mut r = vec![0.0; m.len()];
    if m.len() == d * d {
        for i in 0..d {
            for j in 0..d {
                r[i * d + j] = m[j * d + i];
            }
        }
        r
    } else {
        m.to_vec()
    }
}

/// A row-major matrix as R's column-major vector.
fn column_major(m: &[f64], d: usize) -> Vec<f64> {
    row_major(m, d)
}

/// A 1-based component index.
fn index(j: f64, d: usize) -> Result<usize> {
    let j = whole(j, "j")? as usize;
    if j == 0 || j > d {
        return Err(Error::Other(format!(
            "j must be between 1 and {d}, not {j}"
        )));
    }
    Ok(j - 1)
}

#[extendr]
impl GaussianJointR {
    fn normal(mean: &[f64], cov: &[f64]) -> Result<Self> {
        let d = mean.len();
        Ok(Self {
            inner: Joint::Normal(
                MultivariateNormal::new(mean.to_vec(), row_major(cov, d)).map_err(to_r)?,
            ),
        })
    }

    fn lognormal_moments(mean: &[f64], cov: &[f64]) -> Result<Self> {
        let d = mean.len();
        Ok(Self {
            inner: Joint::Lognormal(
                MultivariateLognormal::from_moments(mean.to_vec(), row_major(cov, d))
                    .map_err(to_r)?,
            ),
        })
    }

    fn lognormal_log_scale(mu: &[f64], log_cov: &[f64]) -> Result<Self> {
        let d = mu.len();
        Ok(Self {
            inner: Joint::Lognormal(
                MultivariateLognormal::from_log_scale(mu.to_vec(), row_major(log_cov, d))
                    .map_err(to_r)?,
            ),
        })
    }

    fn kind(&self) -> &'static str {
        match self.inner {
            Joint::Normal(_) => "normal",
            Joint::Lognormal(_) => "lognormal",
        }
    }

    fn dim(&self) -> f64 {
        self.joint().dim() as f64
    }

    fn mean(&self) -> Vec<f64> {
        match &self.inner {
            Joint::Normal(n) => n.mean().to_vec(),
            Joint::Lognormal(l) => l.mean(),
        }
    }

    /// Column-major, for `matrix(..., d)`.
    fn cov(&self) -> Vec<f64> {
        let d = self.joint().dim();
        match &self.inner {
            Joint::Normal(n) => column_major(n.cov(), d),
            Joint::Lognormal(l) => column_major(&l.cov(), d),
        }
    }

    fn correlation(&self) -> Vec<f64> {
        let d = self.joint().dim();
        match &self.inner {
            Joint::Normal(n) => column_major(&n.correlation(), d),
            Joint::Lognormal(l) => column_major(&l.correlation(), d),
        }
    }

    /// The log-scale mean of a lognormal; empty for a normal.
    fn mu(&self) -> Vec<f64> {
        match &self.inner {
            Joint::Normal(_) => Vec::new(),
            Joint::Lognormal(l) => l.log_normal().mean().to_vec(),
        }
    }

    /// The log-scale covariance of a lognormal, column-major; empty for a
    /// normal.
    fn log_cov(&self) -> Vec<f64> {
        match &self.inner {
            Joint::Normal(_) => Vec::new(),
            Joint::Lognormal(l) => column_major(l.log_normal().cov(), l.dim()),
        }
    }

    fn marginal_normal(&self, j: f64) -> Result<NormalDist> {
        match &self.inner {
            Joint::Normal(n) => Ok(NormalDist {
                inner: n.marginal(index(j, n.dim())?).map_err(to_r)?,
            }),
            Joint::Lognormal(_) => Err(Error::Other("not a multivariate normal".into())),
        }
    }

    fn marginal_lognormal(&self, j: f64) -> Result<Lognormal> {
        match &self.inner {
            Joint::Lognormal(l) => Ok(Lognormal {
                inner: l.marginal(index(j, l.dim())?).map_err(to_r)?,
            }),
            Joint::Normal(_) => Err(Error::Other("not a multivariate lognormal".into())),
        }
    }

    fn total_normal(&self) -> Result<NormalDist> {
        match &self.inner {
            Joint::Normal(n) => Ok(NormalDist { inner: n.total() }),
            Joint::Lognormal(_) => Err(Error::Other(
                "a sum of lognormals is not lognormal: use total_mean and total_variance, \
                 or draw the total with predictive_from_gaussian()"
                    .into(),
            )),
        }
    }

    fn total_mean(&self) -> f64 {
        match &self.inner {
            Joint::Normal(n) => n.total().mu(),
            Joint::Lognormal(l) => l.total_mean(),
        }
    }

    fn total_variance(&self) -> f64 {
        match &self.inner {
            Joint::Normal(n) => n.cov().iter().sum(),
            Joint::Lognormal(l) => l.total_variance(),
        }
    }

    fn simulate(
        &self,
        n_sims: f64,
        seed: f64,
        dims: Vec<String>,
        keys: List,
    ) -> Result<PredictiveDistribution> {
        let joint = self.joint();
        let components = components_from_keys(&dims, keys, joint.dim())?;
        let inner = prospicio_prob::PredictiveDistribution::from_gaussian(
            dims,
            components,
            joint,
            whole(n_sims, "n_sims")? as usize,
            whole(seed, "seed")?,
        )
        .map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

impl GaussianJointR {
    fn joint(&self) -> &dyn GaussianJoint {
        match &self.inner {
            Joint::Normal(n) => n,
            Joint::Lognormal(l) => l,
        }
    }
}

extendr_module! {
    mod multivariate;
    impl GaussianJointR;
}
