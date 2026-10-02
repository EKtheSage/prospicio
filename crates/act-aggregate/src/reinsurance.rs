//! Reinsurance: excess-of-loss layers and towers applied to simulated
//! events, giving gross, ceded and net distributions.

use act_core::{Error, Result};
use act_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};

use crate::monte_carlo::EventSet;

/// A per-occurrence excess-of-loss layer: `limit` xs `attachment` on each
/// loss, then annual terms.
///
/// For one year with losses `x_1, …, x_n`:
///
/// ```text
/// recovery  = Σ min(max(x_e - attachment, 0), limit)
/// after AAD = max(recovery - aggregate_deductible, 0)
/// ceded     = share × min(after AAD, aggregate_limit)
/// ```
///
/// Terms are plain data, so a tower can be stored and replayed.
///
/// ```
/// use act_aggregate::Layer;
///
/// // 5m xs 5m with one reinstatement: at most 10m a year.
/// let layer = Layer::xol("5x5", 5e6, 5e6).unwrap().reinstatements(1).unwrap();
/// assert_eq!(layer.ceded(&[7e6]), 2e6);
/// assert_eq!(layer.ceded(&[12e6, 20e6, 30e6]), 10e6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub name: String,
    pub attachment: f64,
    /// Per-occurrence limit; may be infinite.
    pub limit: f64,
    /// Share of the layer placed, in `(0, 1]`.
    pub share: f64,
    /// Annual aggregate deductible (AAD), retained before the layer pays.
    pub aggregate_deductible: f64,
    /// Annual aggregate limit (AAL); infinite when unlimited.
    pub aggregate_limit: f64,
}

impl Layer {
    /// `limit` xs `attachment`, fully placed, with no annual terms.
    pub fn xol(name: impl Into<String>, limit: f64, attachment: f64) -> Result<Self> {
        if !attachment.is_finite() || attachment < 0.0 {
            return Err(invalid(
                "attachment",
                attachment,
                "must be finite and non-negative",
            ));
        }
        if limit.is_nan() || limit <= 0.0 {
            return Err(invalid("limit", limit, "must be positive"));
        }
        Ok(Self {
            name: name.into(),
            attachment,
            limit,
            share: 1.0,
            aggregate_deductible: 0.0,
            aggregate_limit: f64::INFINITY,
        })
    }

    /// Places `share` of the layer.
    pub fn share(mut self, share: f64) -> Result<Self> {
        if !(share > 0.0 && share <= 1.0) {
            return Err(invalid("share", share, "must be in (0, 1]"));
        }
        self.share = share;
        Ok(self)
    }

    /// Annual aggregate deductible.
    pub fn aggregate_deductible(mut self, aad: f64) -> Result<Self> {
        if !aad.is_finite() || aad < 0.0 {
            return Err(invalid(
                "aggregate_deductible",
                aad,
                "must be finite and non-negative",
            ));
        }
        self.aggregate_deductible = aad;
        Ok(self)
    }

    /// Annual aggregate limit.
    pub fn aggregate_limit(mut self, aal: f64) -> Result<Self> {
        if aal.is_nan() || aal <= 0.0 {
            return Err(invalid("aggregate_limit", aal, "must be positive"));
        }
        self.aggregate_limit = aal;
        Ok(self)
    }

    /// `n` reinstatements: the annual limit becomes `limit × (n + 1)`.
    /// Reinstatement premiums are not modelled yet.
    pub fn reinstatements(self, n: u32) -> Result<Self> {
        let aal = self.limit * (f64::from(n) + 1.0);
        self.aggregate_limit(aal)
    }

    /// Ceded loss for one year's losses.
    pub fn ceded(&self, losses: &[f64]) -> f64 {
        let recovery: f64 = losses
            .iter()
            .map(|&x| (x - self.attachment).max(0.0).min(self.limit))
            .sum();
        let after_aad = (recovery - self.aggregate_deductible).max(0.0);
        self.share * after_aad.min(self.aggregate_limit)
    }
}

/// Layers applied to the same ground-up losses.
///
/// Each layer sees the gross losses (no inuring order yet), so layers that
/// overlap would both pay.
#[derive(Debug, Clone, PartialEq)]
pub struct Tower {
    pub layers: Vec<Layer>,
}

impl Tower {
    /// A tower of `layers`; fails if there are none or two share a name.
    pub fn new(layers: Vec<Layer>) -> Result<Self> {
        if layers.is_empty() {
            return Err(invalid("layers", 0.0, "must not be empty"));
        }
        for (i, layer) in layers.iter().enumerate() {
            if layers[..i].iter().any(|l| l.name == layer.name) {
                return Err(invalid("layers", i as f64, "repeats an earlier layer name"));
            }
        }
        Ok(Self { layers })
    }

    /// Applies the tower to every simulated year.
    ///
    /// The result has dimensions `["kind", "layer"]` and components
    /// `(gross, ground_up)`, `(ceded, <layer name>)` for each layer, and
    /// `(net, retained)`, kept joint per year. `aggregate(&["kind"])` gives
    /// gross, total ceded and net; `net = gross - Σ ceded` in every year.
    ///
    /// ```
    /// use act_aggregate::{Layer, Tower, simulate_events};
    /// use act_prob::{Distribution, Lognormal, Poisson};
    ///
    /// let events = simulate_events(
    ///     &Poisson::new(2.0).unwrap(),
    ///     &Lognormal::from_mean_cv(3e6, 1.5).unwrap(),
    ///     10_000,
    ///     7,
    /// )
    /// .unwrap();
    /// let tower = Tower::new(vec![
    ///     Layer::xol("5x5", 5e6, 5e6).unwrap(),
    ///     Layer::xol("15x10", 15e6, 10e6).unwrap(),
    /// ])
    /// .unwrap();
    /// let result = tower.apply(&events).unwrap();
    /// let by_kind = result.aggregate(&["kind"]).unwrap();
    /// assert_eq!(by_kind.n_components(), 3); // gross, ceded, net
    /// ```
    pub fn apply(&self, events: &EventSet) -> Result<PredictiveDistribution> {
        let n_layers = self.layers.len();
        let mut draws = Vec::with_capacity(events.n_sims() * (n_layers + 2));
        for sim in 0..events.n_sims() {
            let losses = events.events(sim);
            let gross: f64 = losses.iter().sum();
            draws.push(gross);
            let mut ceded_total = 0.0;
            for layer in &self.layers {
                let ceded = layer.ceded(losses);
                ceded_total += ceded;
                draws.push(ceded);
            }
            draws.push(gross - ceded_total);
        }

        let key = |kind: &str, layer: &str| -> ComponentKey {
            vec![KeyValue::from(kind), KeyValue::from(layer)]
        };
        let mut components = vec![key("gross", "ground_up")];
        components.extend(self.layers.iter().map(|l| key("ceded", &l.name)));
        components.push(key("net", "retained"));

        let mut provenance = Provenance::new("reinsurance_tower")
            .version("act-aggregate", env!("CARGO_PKG_VERSION"))
            .seed(events.seed(), act_prob::provenance::SIM_INDEX_SCHEME);
        for l in &self.layers {
            provenance = provenance.param(
                format!("layer:{}", l.name),
                format!(
                    "{} xs {}, share {}, aad {}, aal {}",
                    l.limit, l.attachment, l.share, l.aggregate_deductible, l.aggregate_limit
                ),
            );
        }
        PredictiveDistribution::from_draws(
            vec!["kind".into(), "layer".into()],
            components,
            draws,
            provenance,
        )
    }
}

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulate_events;
    use act_prob::{Distribution, Lognormal, Poisson, Severity};

    #[test]
    fn the_seven_million_example() {
        // Retain 5m; 5m xs 5m; 15m xs 10m. A 7m loss: 5m kept, 2m to layer 2.
        let l2 = Layer::xol("5x5", 5e6, 5e6).unwrap();
        let l3 = Layer::xol("15x10", 15e6, 10e6).unwrap();
        assert_eq!(l2.ceded(&[7e6]), 2e6);
        assert_eq!(l3.ceded(&[7e6]), 0.0);
        // A 30m loss exhausts both layers and leaves 5m + 5m retained.
        assert_eq!(l2.ceded(&[30e6]), 5e6);
        assert_eq!(l3.ceded(&[30e6]), 15e6);
    }

    #[test]
    fn annual_terms() {
        let base = || Layer::xol("L", 10.0, 5.0).unwrap();
        let losses = [8.0, 20.0, 12.0]; // recoveries 3, 10, 7 = 20
        assert_eq!(base().ceded(&losses), 20.0);
        assert_eq!(
            base().aggregate_deductible(4.0).unwrap().ceded(&losses),
            16.0
        );
        assert_eq!(base().aggregate_limit(15.0).unwrap().ceded(&losses), 15.0);
        assert_eq!(base().reinstatements(0).unwrap().ceded(&losses), 10.0);
        assert_eq!(base().share(0.25).unwrap().ceded(&losses), 5.0);
        let all = base()
            .aggregate_deductible(4.0)
            .unwrap()
            .aggregate_limit(15.0)
            .unwrap()
            .share(0.5)
            .unwrap();
        assert_eq!(all.ceded(&losses), 7.5);
        assert_eq!(base().ceded(&[]), 0.0);
    }

    #[test]
    fn rejects_bad_terms() {
        assert!(Layer::xol("L", 0.0, 1.0).is_err());
        assert!(Layer::xol("L", 1.0, -1.0).is_err());
        assert!(Layer::xol("L", 1.0, 0.0).unwrap().share(1.5).is_err());
        assert!(
            Layer::xol("L", 1.0, 0.0)
                .unwrap()
                .aggregate_limit(0.0)
                .is_err()
        );
        assert!(Tower::new(vec![]).is_err());
        let l = Layer::xol("L", 1.0, 0.0).unwrap();
        assert!(Tower::new(vec![l.clone(), l]).is_err());
    }

    fn events() -> EventSet {
        simulate_events(
            &Poisson::new(2.0).unwrap(),
            &Lognormal::from_mean_cv(3e6, 1.5).unwrap(),
            100_000,
            11,
        )
        .unwrap()
    }

    #[test]
    fn gross_equals_ceded_plus_net_every_year() {
        let tower = Tower::new(vec![
            Layer::xol("5x5", 5e6, 5e6)
                .unwrap()
                .reinstatements(1)
                .unwrap(),
            Layer::xol("15x10", 15e6, 10e6).unwrap().share(0.6).unwrap(),
        ])
        .unwrap();
        let result = tower.apply(&events()).unwrap();
        assert_eq!(result.n_components(), 4);
        for sim in 0..result.n_sims() {
            let row = result.row(sim).unwrap();
            assert!((row[0] - row[1] - row[2] - row[3]).abs() <= 1e-6 * row[0].max(1.0));
        }
        let by_kind = result.aggregate(&["kind"]).unwrap();
        let kinds: Vec<_> = by_kind
            .components()
            .iter()
            .map(|k| k[0].to_string())
            .collect();
        assert_eq!(kinds, ["gross", "ceded", "net"]);
        assert_eq!(result.provenance().seed, Some(11));
    }

    #[test]
    fn mean_ceded_matches_the_exact_layer_value() {
        // With no annual terms, E[ceded] = E[N] × E[min(max(X - a, 0), l)].
        let sev = Lognormal::from_mean_cv(3e6, 1.5).unwrap();
        let layer = Layer::xol("5x5", 5e6, 5e6).unwrap();
        let exact = 2.0 * sev.layer(5e6, 5e6);
        let result = Tower::new(vec![layer]).unwrap().apply(&events()).unwrap();
        let ceded = result
            .marginal(&vec![KeyValue::from("ceded"), KeyValue::from("5x5")])
            .unwrap();
        // Four standard errors.
        let se = ceded.std_dev() / (ceded.len() as f64).sqrt();
        assert!(
            (ceded.mean() - exact).abs() < 4.0 * se,
            "{} vs {exact}",
            ceded.mean()
        );
    }
}
