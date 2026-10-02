//! Distortion risk measures, allocation, copulas and Iman-Conover for R.

use act_core::StreamRng;
use act_prob::copula::{self, Copula};
use act_prob::{
    Archimedean, ArchimedeanCopula, Distortion, Empirical, GaussianCopula, Provenance,
    StudentTCopula,
};
use extendr_api::prelude::*;
use extendr_api::{Error, Result};

use crate::distributions::{
    AnySeverity, Grid, PredictiveDistribution, Sampled, components_from_keys,
};
use crate::{to_r, whole};

/// A distortion risk measure.
#[extendr]
pub(crate) struct RiskDistortion {
    inner: Distortion,
}

#[extendr]
impl RiskDistortion {
    /// `kind` is "tvar", "wang", "proportional_hazard" or "dual_power".
    fn new(kind: &str, param: f64) -> Result<Self> {
        let inner = match kind {
            "tvar" => Distortion::tvar(param),
            "wang" => Distortion::wang(param),
            "proportional_hazard" => Distortion::proportional_hazard(param),
            "dual_power" => Distortion::dual_power(param),
            other => return Err(Error::Other(format!("unknown distortion {other:?}"))),
        }
        .map_err(to_r)?;
        Ok(Self { inner })
    }

    fn kind(&self) -> &'static str {
        match self.inner {
            Distortion::Tvar(_) => "tvar",
            Distortion::Wang(_) => "wang",
            Distortion::ProportionalHazard(_) => "proportional_hazard",
            Distortion::DualPower(_) => "dual_power",
        }
    }

    fn param(&self) -> f64 {
        match self.inner {
            Distortion::Tvar(a)
            | Distortion::Wang(a)
            | Distortion::ProportionalHazard(a)
            | Distortion::DualPower(a) => a,
        }
    }

    fn g(&self, s: &[f64]) -> Vec<f64> {
        s.iter().map(|&s| self.inner.g(s)).collect()
    }

    fn weights(&self, n: f64) -> Result<Vec<f64>> {
        Ok(self.inner.weights(whole(n, "n")? as usize))
    }

    /// Risk measure of a sampled, grid or predictive distribution (its total).
    fn measure(&self, x: Robj) -> Result<f64> {
        if let Ok(s) = <&Sampled>::try_from(&x) {
            return Ok(s.inner.distortion(&self.inner));
        }
        if let Ok(g) = <&Grid>::try_from(&x) {
            return Ok(g.inner.distortion(&self.inner));
        }
        if let Ok(p) = <&PredictiveDistribution>::try_from(&x) {
            return Ok(p.inner.distortion(&self.inner));
        }
        Err(Error::Other(
            "expected a sampled, grid_distribution or predictive_distribution".into(),
        ))
    }

    /// Co-measure allocation, one contribution per component.
    fn allocate(&self, pd: Robj) -> Result<Vec<f64>> {
        let pd = <&PredictiveDistribution>::try_from(&pd)
            .map_err(|_| Error::Other("expected a predictive_distribution".into()))?;
        Ok(pd.inner.allocate(&self.inner))
    }
}

enum AnyCopula {
    Gaussian(GaussianCopula),
    StudentT(StudentTCopula),
    Archimedean(ArchimedeanCopula),
}

impl AnyCopula {
    fn as_copula(&self) -> &dyn Copula {
        match self {
            Self::Gaussian(c) => c,
            Self::StudentT(c) => c,
            Self::Archimedean(c) => c,
        }
    }
}

/// A copula.
#[extendr]
pub(crate) struct RiskCopula {
    inner: AnyCopula,
}

#[extendr]
impl RiskCopula {
    /// `correlation` is a `dim × dim` matrix (symmetric, so R's
    /// column-major order is row-major too).
    fn gaussian(correlation: &[f64], dim: f64) -> Result<Self> {
        let dim = whole(dim, "dim")? as usize;
        let c = GaussianCopula::new(correlation, dim).map_err(to_r)?;
        Ok(Self {
            inner: AnyCopula::Gaussian(c),
        })
    }

    fn student_t(correlation: &[f64], dim: f64, nu: f64) -> Result<Self> {
        let dim = whole(dim, "dim")? as usize;
        let c = StudentTCopula::new(correlation, dim, nu).map_err(to_r)?;
        Ok(Self {
            inner: AnyCopula::StudentT(c),
        })
    }

    fn archimedean(family: &str, theta: f64, dim: f64) -> Result<Self> {
        let family = match family {
            "clayton" => Archimedean::Clayton,
            "gumbel" => Archimedean::Gumbel,
            "frank" => Archimedean::Frank,
            "joe" => Archimedean::Joe,
            other => return Err(Error::Other(format!("unknown family {other:?}"))),
        };
        let dim = whole(dim, "dim")? as usize;
        let c = ArchimedeanCopula::new(family, theta, dim).map_err(to_r)?;
        Ok(Self {
            inner: AnyCopula::Archimedean(c),
        })
    }

    fn dim(&self) -> f64 {
        self.inner.as_copula().dim() as f64
    }

    fn description(&self) -> String {
        match &self.inner {
            AnyCopula::Gaussian(_) => "Gaussian".into(),
            AnyCopula::StudentT(c) => format!("Student t, nu = {}", c.nu()),
            AnyCopula::Archimedean(c) => format!("{:?}, theta = {}", c.family(), c.theta()),
        }
    }

    /// `n × dim` uniforms in R's column-major order; row `i` uses stream
    /// `i - 1` of `seed`.
    fn sample(&self, n: f64, seed: f64) -> Result<Vec<f64>> {
        let n = whole(n, "n")? as usize;
        let seed = whole(seed, "seed")?;
        let c = self.inner.as_copula();
        let d = c.dim();
        let mut out = vec![0.0; n * d];
        let mut u = vec![0.0; d];
        for i in 0..n {
            c.sample(&mut StreamRng::new(seed, i as u64), &mut u);
            for (j, x) in u.iter().enumerate() {
                out[i + j * n] = *x;
            }
        }
        Ok(out)
    }

    fn simulate(
        &self,
        marginals: List,
        n_sims: f64,
        seed: f64,
        dims: Vec<String>,
        keys: List,
    ) -> Result<PredictiveDistribution> {
        let marginals: Vec<AnySeverity> = marginals
            .values()
            .map(|m| AnySeverity::from_robj(&m))
            .collect::<Result<_>>()?;
        let refs: Vec<&(dyn act_prob::Distribution + Sync)> = marginals
            .iter()
            .map(|m| m as &(dyn act_prob::Distribution + Sync))
            .collect();
        let components = components_from_keys(&dims, keys, marginals.len())?;
        let inner = copula::simulate(
            self.inner.as_copula(),
            &refs,
            dims,
            components,
            whole(n_sims, "n_sims")? as usize,
            whole(seed, "seed")?,
            Provenance::new("copula"),
        )
        .map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

/// Iman-Conover reordering of a predictive distribution's components.
#[extendr]
fn iman_conover_reorder(
    pd: Robj,
    correlation: &[f64],
    seed: f64,
) -> Result<PredictiveDistribution> {
    let pd = <&PredictiveDistribution>::try_from(&pd)
        .map_err(|_| Error::Other("expected a predictive_distribution".into()))?;
    let inner = copula::iman_conover(&pd.inner, correlation, whole(seed, "seed")?).map_err(to_r)?;
    Ok(PredictiveDistribution { inner })
}

extendr_module! {
    mod risk;
    fn iman_conover_reorder;
    impl RiskDistortion;
    impl RiskCopula;
}
