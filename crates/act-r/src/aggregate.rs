//! Aggregate lane: wrappers over `act_aggregate` for the R `aggregate.R`
//! API (compound distributions, simulated events, reinsurance).

use act_aggregate::{CompoundMethod, CompoundReport, EventSet as EventSetInner, Layer, Tower};
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

    fn as_counting(&self) -> &(dyn Counting + Sync) {
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

fn compound_list(r: &CompoundReport) -> List {
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

/// A per-occurrence excess-of-loss layer.
#[extendr]
pub(crate) struct XolLayer {
    inner: Layer,
}

#[extendr]
impl XolLayer {
    /// `aggregate_limit` may be `Inf`; `reinstatements` is a number or
    /// negative for none; `paid` says whether `reinstatement_rates` were
    /// given. At most one of a finite `aggregate_limit`, `reinstatements`
    /// and paid reinstatements.
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        limit: f64,
        attachment: f64,
        share: f64,
        aggregate_deductible: f64,
        aggregate_limit: f64,
        reinstatements: f64,
        premium: f64,
        reinstatement_rates: &[f64],
        paid: bool,
    ) -> Result<Self> {
        let layer = Layer::xol(name, limit, attachment)
            .and_then(|l| l.share(share))
            .and_then(|l| l.aggregate_deductible(aggregate_deductible))
            .map_err(to_r)?;
        let layer = match (aggregate_limit.is_finite(), reinstatements >= 0.0, paid) {
            (true, false, false) => layer.aggregate_limit(aggregate_limit).map_err(to_r)?,
            (false, true, false) => {
                let n = whole(reinstatements, "reinstatements")?;
                let n = u32::try_from(n)
                    .map_err(|_| Error::Other("reinstatements is too large".into()))?;
                layer.reinstatements(n).map_err(to_r)?
            }
            (false, false, true) => layer
                .paid_reinstatements(premium, reinstatement_rates.to_vec())
                .map_err(to_r)?,
            (false, false, false) => layer,
            _ => {
                return Err(Error::Other(
                    "give at most one of aggregate_limit, reinstatements and reinstatement_rates"
                        .into(),
                ));
            }
        };
        Ok(Self { inner: layer })
    }

    fn quota_share(name: &str, cession: f64) -> Result<Self> {
        let inner = Layer::quota_share(name, cession).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn stop_loss(name: &str, limit: f64, retention: f64) -> Result<Self> {
        let inner = Layer::stop_loss(name, limit, retention).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn name(&self) -> String {
        self.inner.name.clone()
    }

    fn limit(&self) -> f64 {
        self.inner.limit
    }

    fn attachment(&self) -> f64 {
        self.inner.attachment
    }

    fn share(&self) -> f64 {
        self.inner.share
    }

    fn aggregate_deductible(&self) -> f64 {
        self.inner.aggregate_deductible
    }

    fn aggregate_limit(&self) -> f64 {
        self.inner.aggregate_limit
    }

    fn premium(&self) -> f64 {
        self.inner.premium
    }

    fn reinstatement_rates(&self) -> Vec<f64> {
        self.inner.reinstatement_rates.clone()
    }

    fn ceded(&self, losses: &[f64]) -> f64 {
        self.inner.ceded(losses)
    }

    fn ceded_by_event(&self, losses: &[f64]) -> Vec<f64> {
        self.inner.ceded_by_event(losses)
    }

    fn reinstatement_premium(&self, losses: &[f64]) -> f64 {
        self.inner.reinstatement_premium(losses)
    }
}

/// Layers in inuring stages.
#[extendr]
pub(crate) struct ReinsuranceTower {
    inner: Tower,
}

#[extendr]
impl ReinsuranceTower {
    fn new(layers: List) -> Result<Self> {
        let inner = Tower::new(layer_list(layers)?).map_err(to_r)?;
        Ok(Self { inner })
    }

    /// `stages` is a list of lists of layers.
    fn inuring(stages: List) -> Result<Self> {
        let stages = stages
            .values()
            .map(|stage| {
                List::try_from(&stage)
                    .map_err(|_| Error::Other("stages must be a list of lists of layers".into()))
                    .and_then(layer_list)
            })
            .collect::<Result<_>>()?;
        let inner = Tower::inuring(stages).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn layer_names(&self) -> Vec<String> {
        self.inner.layers.iter().map(|l| l.name.clone()).collect()
    }

    /// 1-based stage of each layer.
    fn stages(&self) -> Vec<f64> {
        self.inner.stages.iter().map(|&s| s as f64 + 1.0).collect()
    }

    fn ceded(&self, losses: &[f64]) -> Vec<f64> {
        self.inner.ceded(losses)
    }

    fn apply(&self, events: Robj) -> Result<PredictiveDistribution> {
        let events = <&EventSet>::try_from(&events)
            .map_err(|_| Error::Other("events must be an event_set".into()))?;
        let inner = self.inner.apply(&events.inner).map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

fn layer_list(layers: List) -> Result<Vec<Layer>> {
    layers
        .values()
        .map(|l| {
            <&XolLayer>::try_from(&l)
                .map(|l| l.inner.clone())
                .map_err(|_| Error::Other("layers must all be xol_layer objects".into()))
        })
        .collect()
}

extendr_module! {
    mod aggregate;
    fn compound;
    impl EventSet;
    impl XolLayer;
    impl ReinsuranceTower;
}
