//! A joint predictive distribution: simulations by component.

use crate::convert::{Obj, err, strings};
use js_sys::{Array, Float64Array};
use prospicio_prob::{Distribution, Empirical, KeyValue, PredictiveDistribution};
use wasm_bindgen::prelude::*;

/// Simulated outcomes of several components (origins, layers, segments)
/// on shared simulations: row `i` of every component is the same
/// simulation, so sums and tail measures keep the dependence between them.
#[wasm_bindgen]
pub struct Predictive {
    pub(crate) inner: PredictiveDistribution,
}

impl From<PredictiveDistribution> for Predictive {
    fn from(inner: PredictiveDistribution) -> Self {
        Self { inner }
    }
}

fn key_text(k: &KeyValue) -> String {
    match k {
        KeyValue::Int(v) => v.to_string(),
        KeyValue::Text(s) => s.clone(),
        KeyValue::Period(p) => p.to_string(),
    }
}

#[wasm_bindgen]
impl Predictive {
    /// The names of the dimensions that label a component, such as
    /// `["origin"]` or `["layer", "kind"]`.
    #[wasm_bindgen(getter)]
    pub fn dims(&self) -> Array {
        strings(self.inner.dims())
    }

    /// Each component's label: one string per dimension.
    #[wasm_bindgen(getter)]
    pub fn components(&self) -> Array {
        self.inner
            .components()
            .iter()
            .map(|c| JsValue::from(strings(c.iter().map(key_text))))
            .collect()
    }

    /// The number of simulations.
    #[wasm_bindgen(getter, js_name = nSims)]
    pub fn n_sims(&self) -> usize {
        self.inner.n_sims()
    }

    /// The number of components.
    #[wasm_bindgen(getter, js_name = nComponents)]
    pub fn n_components(&self) -> usize {
        self.inner.n_components()
    }

    /// Every draw, row by row: simulation `i`, component `j` is at
    /// `i * nComponents + j`.
    pub fn draws(&self) -> Float64Array {
        Float64Array::from(self.inner.draw_matrix())
    }

    /// The total over all components in each simulation.
    pub fn total(&self) -> Float64Array {
        Float64Array::from(self.inner.total().draws())
    }

    /// The mean of the total.
    pub fn mean(&self) -> f64 {
        self.inner.total().mean()
    }

    /// The standard deviation of the total.
    #[wasm_bindgen(js_name = standardDeviation)]
    pub fn standard_deviation(&self) -> f64 {
        self.inner.total().std_dev()
    }

    /// The value at risk of the total at level `p`: its lower quantile.
    pub fn var(&self, p: f64) -> Result<f64, JsError> {
        self.inner.total().var(p).map_err(err)
    }

    /// The tail value at risk of the total at level `p`.
    pub fn tvar(&self, p: f64) -> Result<f64, JsError> {
        self.inner.total().tvar(p).map_err(err)
    }

    /// One row per component: its label under each dimension's name, and
    /// the `mean` and `std_dev` of its draws. An error when a dimension is
    /// itself named `mean` or `std_dev` (a key column of that name), whose
    /// labels the statistics would replace.
    pub fn summary(&self) -> Result<JsValue, JsError> {
        if let Some(name) = self
            .inner
            .dims()
            .iter()
            .find(|d| matches!(d.as_str(), "mean" | "std_dev"))
        {
            return Err(JsError::new(&format!(
                "summary: the dimension {name:?} has the name of a summary column; \
                 rename that key column, or aggregate() it away first"
            )));
        }
        let marginals: Vec<_> = self
            .inner
            .components()
            .iter()
            .filter_map(|c| self.inner.marginal(c))
            .collect();
        let mean: Vec<f64> = marginals.iter().map(Distribution::mean).collect();
        let sd: Vec<f64> = marginals.iter().map(Distribution::std_dev).collect();
        let mut o = Obj::new();
        for (d, name) in self.inner.dims().iter().enumerate() {
            o = o.strings(
                name,
                self.inner.components().iter().map(|c| key_text(&c[d])),
            );
        }
        Ok(o.numbers("mean", &mean).numbers("std_dev", &sd).done())
    }

    /// The distribution summed over every dimension not in `keep`, on the
    /// same simulations.
    pub fn aggregate(&self, keep: Vec<String>) -> Result<Predictive, JsError> {
        let keep: Vec<&str> = keep.iter().map(String::as_str).collect();
        self.inner.aggregate(&keep).map(Into::into).map_err(err)
    }
}
