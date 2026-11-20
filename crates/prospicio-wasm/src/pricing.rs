//! Risk-loaded pricing: distortions and the price of a cover.

use crate::convert::{Obj, err};
use prospicio_pricing::risk_load::{self, PremiumRule};
use prospicio_prob as prob;
use prospicio_prob::distortion::Family;
use prospicio_prob::{Empirical, Sampled};
use wasm_bindgen::prelude::*;

/// A distortion `g` of the survival function: a spectral risk measure.
#[wasm_bindgen]
pub struct Distortion {
    inner: prob::Distortion,
}

#[wasm_bindgen]
impl Distortion {
    /// The member of a one-parameter family with this parameter:
    /// `"tvar"` (level p), `"wang"` (lambda), `"ph"` or
    /// `"proportional_hazard"` (rho), `"dual"` (beta), `"ccoc"` (the cost
    /// of capital r), `"tsallis"`, `"exponential"`, and the others of
    /// `DistortionKind::NAMES`. `r0`, the risk-free rate, is read only by
    /// the families that need one.
    #[wasm_bindgen(js_name = fromName)]
    pub fn from_name(name: &str, parameter: f64, r0: Option<f64>) -> Result<Distortion, JsError> {
        let family = Family::from_name(name, r0.unwrap_or(0.0)).map_err(err)?;
        let inner = family.with(parameter).map_err(err)?;
        Ok(Self { inner })
    }

    /// The family's name.
    #[wasm_bindgen(getter)]
    pub fn family(&self) -> String {
        self.inner.family().to_owned()
    }

    /// The parameter of a one-parameter family.
    #[wasm_bindgen(getter)]
    pub fn parameter(&self) -> Option<f64> {
        self.inner.parameter()
    }

    /// `g(s)` for a survival probability `s` in `[0, 1]`.
    pub fn g(&self, s: f64) -> f64 {
        self.inner.g(s)
    }

    /// The distortion risk measure of equally likely `losses`:
    /// the integral of `g(S(x))`.
    pub fn apply(&self, losses: Vec<f64>) -> Result<f64, JsError> {
        let sample = Sampled::new(losses).map_err(err)?;
        Ok(self.inner.apply_sorted(sample.sorted()))
    }
}

fn priced(p: risk_load::Price) -> JsValue {
    Obj::new()
        .set("expectedLoss", p.expected_loss)
        .set("premium", p.premium)
        .set("assets", p.assets)
        .set("margin", p.margin())
        .set("capital", p.capital())
        .set("lossRatio", p.loss_ratio())
        .set("returnOnCapital", p.return_on_capital())
        .done()
}

/// The price of a cover from its equally likely `losses`, the premium set
/// by the distortion `premium` and the assets by the distortion `assets`
/// (a capital standard such as `Distortion.fromName("tvar", 0.99)`).
///
/// Returns `{ expectedLoss, premium, assets, margin, capital, lossRatio,
/// returnOnCapital }`.
#[wasm_bindgen]
pub fn price(
    losses: Vec<f64>,
    premium: &Distortion,
    assets: &Distortion,
) -> Result<JsValue, JsError> {
    let sample = Sampled::new(losses).map_err(err)?;
    let rule = PremiumRule::Distortion(premium.inner.clone());
    risk_load::price(&sample, &rule, &assets.inner)
        .map(priced)
        .map_err(err)
}

/// The price of a cover at a constant cost of capital `rate`: the margin
/// is `rate` times the capital, with the assets set by the distortion
/// `assets`. Returns what `price` returns.
#[wasm_bindgen(js_name = priceAtCostOfCapital)]
pub fn price_at_cost_of_capital(
    losses: Vec<f64>,
    rate: f64,
    assets: &Distortion,
) -> Result<JsValue, JsError> {
    let sample = Sampled::new(losses).map_err(err)?;
    let rule = PremiumRule::cost_of_capital(rate).map_err(err)?;
    risk_load::price(&sample, &rule, &assets.inner)
        .map(priced)
        .map_err(err)
}
