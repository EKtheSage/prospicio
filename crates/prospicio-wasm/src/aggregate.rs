//! Aggregate losses and reinsurance: claim counts, the compound
//! distribution by FFT, simulated events, towers and the occurrence
//! exceedance curve.

use crate::convert::{Fields, err, options as read_options, value_of};
use crate::predictive::Predictive;
use crate::prob::Dist;
use js_sys::Float64Array;
use prospicio_aggregate as core;
use prospicio_aggregate::Sizing;
use prospicio_prob::{Counting, Distribution, SeverityDist};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// A claim count distribution.
#[wasm_bindgen]
pub struct Count {
    inner: Box<dyn Counting + Send + Sync>,
}

#[wasm_bindgen]
impl Count {
    /// Poisson with mean `lambda`.
    pub fn poisson(lambda: f64) -> Result<Count, JsError> {
        let n = prospicio_prob::Poisson::new(lambda).map_err(err)?;
        Ok(Self { inner: Box::new(n) })
    }

    /// Negative binomial with size `r` and scale `beta`: mean `r beta`,
    /// variance `r beta (1 + beta)`.
    #[wasm_bindgen(js_name = negativeBinomial)]
    pub fn negative_binomial(r: f64, beta: f64) -> Result<Count, JsError> {
        let n = prospicio_prob::NegativeBinomial::new(r, beta).map_err(err)?;
        Ok(Self { inner: Box::new(n) })
    }

    /// Binomial with `n` trials of probability `p`.
    pub fn binomial(n: u32, p: f64) -> Result<Count, JsError> {
        let n = prospicio_prob::Binomial::new(u64::from(n), p).map_err(err)?;
        Ok(Self { inner: Box::new(n) })
    }

    pub fn mean(&self) -> f64 {
        self.inner.mean()
    }

    pub fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// `P(N = k)`.
    pub fn pmf(&self, k: u32) -> f64 {
        self.inner.pmf(u64::from(k))
    }
}

/// A distribution on the grid `0, step, 2 step, ...`: a compound
/// distribution from `fft`.
#[wasm_bindgen]
pub struct Grid {
    inner: prospicio_prob::Grid,
    report: Option<core::CompoundReport>,
}

#[wasm_bindgen]
impl Grid {
    /// The grid's bucket.
    #[wasm_bindgen(getter)]
    pub fn step(&self) -> f64 {
        self.inner.step()
    }

    /// The probability at each grid point.
    pub fn probs(&self) -> Float64Array {
        Float64Array::from(self.inner.probs())
    }

    /// The probability mass the grid's extent leaves out, from `fft`.
    #[wasm_bindgen(getter, js_name = tailMass)]
    pub fn tail_mass(&self) -> Option<f64> {
        self.report.as_ref().map(|r| r.tail_mass)
    }

    /// The estimated wrap-around error of the FFT, from `fft`.
    #[wasm_bindgen(getter, js_name = aliasingError)]
    pub fn aliasing_error(&self) -> Option<f64> {
        self.report.as_ref().map(|r| r.aliasing_error)
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

    /// `P(S <= x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(x)
    }

    /// The lower quantile at level `p`.
    pub fn quantile(&self, p: f64) -> Result<f64, JsError> {
        self.inner.quantile(p).map_err(err)
    }

    /// The tail value at risk at level `p`.
    pub fn tvar(&self, p: f64) -> Result<f64, JsError> {
        self.inner.tvar(p).map_err(err)
    }
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FftOptions {
    /// The grid has `2^log2` points when the step is chosen for you.
    #[serde(default)]
    log2: Option<u32>,
    /// The bucket; give it together with `points`.
    #[serde(default)]
    step: Option<f64>,
    #[serde(default)]
    points: Option<usize>,
}

impl Fields for FftOptions {
    const FIELDS: &'static [&'static str] = &["log2", "step", "points"];
}

/// The compound distribution of `count` claims of `severity` by FFT,
/// the severity discretized by rounding.
///
/// `options`: `{ log2 }` to let the grid be sized from the tails (16 by
/// default, 65,536 points), or `{ step, points }` to give the grid. The
/// claim count is independent of the severities and the severities are
/// independent and identically distributed.
#[wasm_bindgen]
pub fn fft(count: &Count, severity: &Dist, options: JsValue) -> Result<Grid, JsError> {
    let o: FftOptions = read_options(options, "fft options")?;
    let sev = SeverityDist::try_from(severity.inner.clone())
        .map_err(|_| JsError::new("fft needs a severity with exact layer moments"))?;
    let (inner, report) = match (o.step, o.points) {
        (Some(step), Some(points)) => {
            let (grid, _) = prospicio_prob::Grid::rounding(&sev, step, points).map_err(err)?;
            core::fft(count.inner.as_ref(), &grid, points).map_err(err)?
        }
        (None, None) => {
            let sizing = Sizing {
                log2: o.log2.unwrap_or(Sizing::default().log2),
                ..Sizing::default()
            };
            let (grid, report, _) =
                core::fft_auto(count.inner.as_ref(), &sev, &sizing).map_err(err)?;
            (grid, report)
        }
        _ => {
            return Err(JsError::new(
                "fft options: give both step and points, or neither",
            ));
        }
    };
    Ok(Grid {
        inner,
        report: Some(report),
    })
}

/// Simulated years of events: each year's losses in the order they occur.
#[wasm_bindgen]
pub struct EventSet {
    inner: core::EventSet,
}

#[wasm_bindgen]
impl EventSet {
    /// An event set from each year's losses: an array of arrays.
    #[wasm_bindgen(js_name = fromYears)]
    pub fn from_years(years: JsValue, seed: Option<f64>) -> Result<EventSet, JsError> {
        let years: Vec<Vec<f64>> = value_of(years, "EventSet.fromYears")?;
        let inner =
            core::EventSet::from_years(years, whole(seed.unwrap_or(0.0), "seed")?).map_err(err)?;
        Ok(Self { inner })
    }

    /// The number of simulated years.
    #[wasm_bindgen(getter, js_name = nSims)]
    pub fn n_sims(&self) -> usize {
        self.inner.n_sims()
    }

    /// The number of events in each year.
    pub fn counts(&self) -> Vec<u32> {
        self.inner.counts().iter().map(|&c| c as u32).collect()
    }

    /// The losses of year `year` (from 0).
    pub fn events(&self, year: usize) -> Result<Float64Array, JsError> {
        if year >= self.inner.n_sims() {
            return Err(JsError::new(&format!(
                "year {year} is past the last, {}",
                self.inner.n_sims() - 1
            )));
        }
        Ok(Float64Array::from(self.inner.events(year)))
    }

    /// Each year's total loss.
    pub fn totals(&self) -> Result<Predictive, JsError> {
        self.inner.totals().map(Into::into).map_err(err)
    }
}

/// A seed or count given as a JavaScript number: a whole number from 0 to
/// 2^53.
fn whole(x: f64, name: &str) -> Result<u64, JsError> {
    if x.is_finite() && x >= 0.0 && x.fract() == 0.0 && x <= 9_007_199_254_740_992.0 {
        Ok(x as u64)
    } else {
        Err(JsError::new(&format!(
            "{name} must be a whole number from 0 to 2^53, got {x}"
        )))
    }
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SimulateOptions {
    #[serde(default)]
    n_sims: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
}

impl Fields for SimulateOptions {
    const FIELDS: &'static [&'static str] = &["nSims", "seed"];
}

/// Simulated years of `count` events of `severity`.
///
/// `options`: `{ nSims, seed }`, by default 10,000 years and seed 0. One
/// random stream per year, so the same seed gives the same years as in
/// Rust, Python and R.
#[wasm_bindgen(js_name = simulateEvents)]
pub fn simulate_events(
    count: &Count,
    severity: &Dist,
    options: JsValue,
) -> Result<EventSet, JsError> {
    let o: SimulateOptions = read_options(options, "simulateEvents options")?;
    let inner = core::simulate_events(
        count.inner.as_ref(),
        &severity.inner,
        o.n_sims.unwrap_or(10_000),
        o.seed.unwrap_or(0),
    )
    .map_err(err)?;
    Ok(EventSet { inner })
}

/// A reinsurance tower: layers in inuring stages, with their terms.
#[wasm_bindgen]
pub struct Tower {
    inner: core::Tower,
}

#[wasm_bindgen]
impl Tower {
    /// A tower from the versioned JSON document that Rust, Python and R
    /// save (`Tower.to_json`), as text or as an object.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(document: JsValue) -> Result<Tower, JsError> {
        let text = crate::convert::json_text(&document, "Tower.fromJson")?;
        let inner = core::Tower::from_json(&text).map_err(err)?;
        Ok(Self { inner })
    }

    /// The tower's JSON document, as text.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// The tower applied to simulated events: the gross, each layer's
    /// ceded loss and the net, by simulation.
    pub fn apply(&self, events: &EventSet) -> Result<Predictive, JsError> {
        self.inner.apply(&events.inner).map(Into::into).map_err(err)
    }

    /// The tower applied to plain annual losses, one per simulation, for
    /// layers with annual terms only (quota shares, stop losses).
    #[wasm_bindgen(js_name = applyAnnual)]
    pub fn apply_annual(&self, losses: Vec<f64>) -> Result<Predictive, JsError> {
        self.inner
            .apply_annual(&losses)
            .map(Into::into)
            .map_err(err)
    }
}

/// The occurrence exceedance (OEP) curve: the probability that the
/// largest event of a year exceeds each of `losses`.
#[wasm_bindgen]
pub fn oep(count: &Count, severity: &Dist, losses: Vec<f64>) -> Float64Array {
    let p: Vec<f64> = losses
        .iter()
        .map(|&x| core::pml::occurrence_exceedance(count.inner.as_ref(), &severity.inner, x))
        .collect();
    Float64Array::from(p.as_slice())
}

/// The occurrence PML: the loss that the largest event of a year exceeds
/// with each of the annual `probabilities` (0.01 is the 100-year loss).
#[wasm_bindgen(js_name = occurrencePml)]
pub fn occurrence_pml(
    count: &Count,
    severity: &Dist,
    probabilities: Vec<f64>,
) -> Result<Float64Array, JsError> {
    let losses: Vec<f64> = probabilities
        .iter()
        .map(|&p| {
            core::pml::occurrence_pml(count.inner.as_ref(), &severity.inner, p).map(|pml| pml.loss)
        })
        .collect::<Result<_, _>>()
        .map_err(err)?;
    Ok(Float64Array::from(losses.as_slice()))
}
