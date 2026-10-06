//! Aggregate lane: wrappers over `act_aggregate` for the R `aggregate.R`
//! API (compound distributions and simulated events).

use act_aggregate::{CompoundMethod, CompoundReport, EventSet as EventSetInner};
use act_prob::Counting;
use extendr_api::prelude::*;
use extendr_api::{Error, Result};

use crate::distributions::{AnySeverity, Grid, NegativeBinomial, Poisson, PredictiveDistribution};
use crate::{to_r, whole};

/// A claim count accepted by the aggregation functions.
pub(crate) enum AnyCount {
    Poisson(act_prob::Poisson),
    NegativeBinomial(act_prob::NegativeBinomial),
    Binomial(act_prob::Binomial),
}

impl AnyCount {
    pub(crate) fn from_robj(obj: &Robj) -> Result<Self> {
        if let Ok(n) = <&Poisson>::try_from(obj) {
            return Ok(Self::Poisson(n.inner));
        }
        if let Ok(n) = <&NegativeBinomial>::try_from(obj) {
            return Ok(Self::NegativeBinomial(n.inner));
        }
        if let Ok(n) = <&crate::pareto::Binomial>::try_from(obj) {
            return Ok(Self::Binomial(n.inner));
        }
        Err(Error::Other(
            "frequency must be a poisson_count, negative_binomial_count or binomial_count".into(),
        ))
    }

    pub(crate) fn as_counting(&self) -> &(dyn Counting + Sync) {
        match self {
            Self::Poisson(n) => n,
            Self::NegativeBinomial(n) => n,
            Self::Binomial(n) => n,
        }
    }
}

impl Counting for AnyCount {
    fn pmf(&self, k: u64) -> f64 {
        self.as_counting().pmf(k)
    }
    fn mean(&self) -> f64 {
        self.as_counting().mean()
    }
    fn variance(&self) -> f64 {
        self.as_counting().variance()
    }
    fn panjer_ab(&self) -> (f64, f64) {
        self.as_counting().panjer_ab()
    }
    fn pgf(&self, z: f64) -> f64 {
        self.as_counting().pgf(z)
    }
    fn pgf_complex(&self, z: (f64, f64)) -> (f64, f64) {
        self.as_counting().pgf_complex(z)
    }
}

pub(crate) fn compound_list(r: &CompoundReport) -> List {
    let method = match r.method {
        CompoundMethod::Panjer => "panjer",
        CompoundMethod::Fft => "fft",
    };
    list!(
        method = method,
        points = r.points as f64,
        tail_mass = r.tail_mass,
        aliasing_error = r.aliasing_error,
        expected_mean = r.expected_mean,
        grid_mean = r.grid_mean,
        mean_error = r.mean_error()
    )
}

/// Compound distribution of `S = X_1 + ... + X_N` on the severity grid's
/// step; `method` is "panjer" or "fft".
#[extendr]
fn compound(frequency: Robj, severity: Robj, points: f64, method: &str) -> Result<Grid> {
    let n = AnyCount::from_robj(&frequency)?;
    let sev = <&Grid>::try_from(&severity)
        .map_err(|_| Error::Other("severity must be a grid_distribution".into()))?;
    let points = whole(points, "points")? as usize;
    let (inner, report) = match method {
        "panjer" => act_aggregate::panjer(n.as_counting(), &sev.inner, points),
        "fft" => act_aggregate::fft(n.as_counting(), &sev.inner, points),
        other => {
            return Err(Error::Other(format!(
                "method must be panjer or fft, got {other}"
            )));
        }
    }
    .map_err(to_r)?;
    Ok(Grid::with_report(inner, compound_list(&report)))
}

/// Simulated years of individual losses.
#[extendr]
pub(crate) struct EventSet {
    pub(crate) inner: EventSetInner,
}

#[extendr]
impl EventSet {
    fn simulate(frequency: Robj, severity: Robj, n_sims: f64, seed: f64) -> Result<Self> {
        let n = AnyCount::from_robj(&frequency)?;
        let sev = AnySeverity::from_robj(&severity)?;
        let n_sims = whole(n_sims, "n_sims")? as usize;
        let inner =
            act_aggregate::simulate_events(n.as_counting(), &sev, n_sims, whole(seed, "seed")?)
                .map_err(to_r)?;
        Ok(Self { inner })
    }

    fn n_sims(&self) -> f64 {
        self.inner.n_sims() as f64
    }

    fn seed(&self) -> f64 {
        self.inner.seed() as f64
    }

    /// Year `sim`'s losses; `sim` is 1-based, as in R.
    fn events(&self, sim: f64) -> Result<Vec<f64>> {
        let sim = whole(sim, "sim")? as usize;
        if sim == 0 || sim > self.inner.n_sims() {
            return Err(Error::Other(format!(
                "year {sim} out of range for {} simulated years",
                self.inner.n_sims()
            )));
        }
        Ok(self.inner.events(sim - 1).to_vec())
    }

    fn counts(&self) -> Vec<f64> {
        self.inner.counts().into_iter().map(|c| c as f64).collect()
    }

    fn totals(&self) -> Result<PredictiveDistribution> {
        let inner = self.inner.totals().map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

extendr_module! {
    mod aggregate;
    fn compound;
    impl EventSet;
}
