//! Distributions: a severity from its saved JSON, and its functions.

use crate::convert::err;
use prospicio_prob as core;
use prospicio_prob::Distribution;
use wasm_bindgen::prelude::*;

/// A distribution of prospicio's `Dist` enum: the lognormal, gamma, Pareto
/// family, Weibull, Burr, mixtures, truncations and the rest, built from
/// the versioned JSON document that Rust, Python and R save.
#[wasm_bindgen]
pub struct Dist {
    pub(crate) inner: core::Dist,
}

impl Dist {
    fn severity(&self, what: &str) -> Result<&(dyn core::Severity + Send + Sync), JsError> {
        self.inner.as_severity().ok_or_else(|| {
            JsError::new(&format!(
                "{what} needs a severity; {} is not one",
                self.inner.family()
            ))
        })
    }
}

#[wasm_bindgen]
impl Dist {
    /// A distribution from its JSON document, as text or as an object:
    ///
    /// ```js
    /// const d = Dist.fromJson({ format: "risk_rs.distribution", format_version: 1,
    ///                           family: "lognormal", meanlog: 7, sdlog: 0.5 });
    /// ```
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(document: JsValue) -> Result<Dist, JsError> {
        let text = crate::convert::json_text(&document, "Dist.fromJson")?;
        let inner = core::Dist::from_json(&text).map_err(err)?;
        Ok(Self { inner })
    }

    /// The distribution's JSON document, as text.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsError> {
        self.inner.to_json().map_err(err)
    }

    /// The family's name, such as `"lognormal"`.
    #[wasm_bindgen(getter)]
    pub fn family(&self) -> String {
        self.inner.family().to_owned()
    }

    pub fn mean(&self) -> f64 {
        self.inner.mean()
    }

    pub fn variance(&self) -> f64 {
        self.inner.variance()
    }

    #[wasm_bindgen(js_name = standardDeviation)]
    pub fn standard_deviation(&self) -> f64 {
        self.inner.std_dev()
    }

    /// `P(X <= x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(x)
    }

    /// `P(X > x)`.
    pub fn survival(&self, x: f64) -> f64 {
        self.inner.survival(x)
    }

    /// The lower quantile at level `p`.
    pub fn quantile(&self, p: f64) -> Result<f64, JsError> {
        self.inner.quantile(p).map_err(err)
    }

    /// The limited expected value `E[min(X, limit)]`.
    pub fn lev(&self, limit: f64) -> Result<f64, JsError> {
        Ok(self.severity("lev")?.lev(limit))
    }

    /// The expected loss to the layer `limit` xs `attachment`.
    pub fn layer(&self, limit: f64, attachment: f64) -> Result<f64, JsError> {
        Ok(self.severity("layer")?.layer(limit, attachment))
    }

    /// The tail value at risk at level `p`:
    /// `VaR(p) + E[(X - VaR(p))+] / (1 - p)`.
    pub fn tvar(&self, p: f64) -> Result<f64, JsError> {
        self.severity("tvar")?.tvar(p).map_err(err)
    }
}
