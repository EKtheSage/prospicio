//! Reinsurance: wrappers over `prospicio_aggregate::reinsurance` for the R
//! `reinsurance.R` API (layers and towers).

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_aggregate::{Layer, Tower};

use crate::aggregate::{AnyCount, EventSet, compound_list};
use crate::distributions::{Grid, PredictiveDistribution};
use crate::{to_r, whole};

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
    /// and paid reinstatements; `pro_rata_time` needs paid ones.
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
        pro_rata_time: bool,
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
        let layer = if pro_rata_time {
            layer.pro_rata_as_to_time().map_err(to_r)?
        } else {
            layer
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

    fn surplus(name: &str, retention: f64, lines: f64) -> Result<Self> {
        let inner = Layer::surplus(name, retention, lines).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn ceded_with_sums_insured(&self, losses: &[f64], sums_insured: &[f64]) -> Result<f64> {
        if losses.len() != sums_insured.len() {
            return Err(Error::Other("give one sum insured per loss".into()));
        }
        Ok(self.inner.ceded_with_sums_insured(losses, sums_insured))
    }

    fn needs_sums_insured(&self) -> bool {
        self.inner.needs_sums_insured()
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

    fn pro_rata_time(&self) -> bool {
        self.inner.pro_rata_time
    }

    fn with_loss_corridor(&self, lower: f64, upper: f64, retained: f64) -> Result<Self> {
        let inner = self
            .inner
            .clone()
            .loss_corridor(lower, upper, retained)
            .map_err(to_r)?;
        Ok(Self { inner })
    }

    /// `c(lower, upper, retained)`, empty when there is none.
    fn loss_corridor(&self) -> Vec<f64> {
        self.inner
            .corridor
            .map(|c| vec![c.lower, c.upper, c.retained])
            .unwrap_or_default()
    }

    /// `times` empty when not given.
    fn reinstatement_premium(&self, losses: &[f64], times: &[f64]) -> Result<f64> {
        if times.is_empty() {
            return Ok(self.inner.reinstatement_premium(losses));
        }
        if times.len() != losses.len() {
            return Err(Error::Other("give one time per loss".into()));
        }
        Ok(self.inner.reinstatement_premium_dated(losses, times))
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

    fn from_json(text: &str) -> Result<Self> {
        let inner = Tower::from_json(text).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn to_json(&self) -> String {
        self.inner.to_json()
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

    /// Gross, ceded and net annual distributions on the grid, by FFT:
    /// `list(gross, ceded, net, expected_reinstatement_premium, on_points)`,
    /// with `net` NULL when it is not one compound total.
    fn on_grid(&self, frequency: Robj, severity: Robj, points: f64) -> Result<List> {
        let n = AnyCount::from_robj(&frequency)?;
        let sev = <&Grid>::try_from(&severity)
            .map_err(|_| Error::Other("severity must be a grid_distribution".into()))?;
        let points = whole(points, "points")? as usize;
        let r = self
            .inner
            .on_grid(n.as_counting(), &sev.inner, points)
            .map_err(to_r)?;
        let ceded: Vec<Robj> = r
            .ceded
            .into_iter()
            .zip(&r.ceded_reports)
            .map(|(g, report)| Grid::with_report(g, compound_list(report)).into())
            .collect();
        let net: Robj = match r.net {
            Some(g) => Grid::wrap(g).into(),
            None => ().into(),
        };
        Ok(list!(
            gross = Grid::with_report(r.gross, compound_list(&r.gross_report)),
            ceded = List::from_values(ceded),
            net = net,
            expected_reinstatement_premium = r.expected_reinstatement_premium,
            on_points = r.on_points
        ))
    }

    /// Each simulation's total of a predictive distribution as one
    /// aggregate loss.
    fn apply_aggregate(&self, losses: Robj) -> Result<PredictiveDistribution> {
        let pd = <&PredictiveDistribution>::try_from(&losses)
            .map_err(|_| Error::Other("losses must be a predictive_distribution".into()))?;
        let inner = self.inner.apply_aggregate(&pd.inner).map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
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
    mod reinsurance;
    impl XolLayer;
    impl ReinsuranceTower;
}
