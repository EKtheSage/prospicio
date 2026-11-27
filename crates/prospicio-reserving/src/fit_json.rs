//! Fitted reserving models as JSON documents, for saving a fit exactly as
//! it is and for Python's `pickle` (#978): [`ChainLadderFit`] and the
//! fits of every segment of a triangle ([`SegmentFits`]), [`MackFit`],
//! [`ExpectedLossFit`] (the expected-loss, Bornhuetter-Ferguson and
//! Benktander fits), [`CapeCodFit`], and the ODP and Mack bootstraps'
//! fits ([`OdpBootstrapFit`], [`OdpBootstrapFits`], [`MackBootstrapFit`],
//! [`MackBootstrapFits`]), whose simulated reserves and cells are embedded
//! as [`PredictiveDistribution::to_json`] writes them.
//!
//! Each document is an object with `"format"` and `"format_version"`, as
//! the triangle's (`triangle/json.rs`) is. Numbers round-trip bit for bit;
//! a non-finite number is written as the string `"NaN"`, `"inf"` or
//! `"-inf"`, and an absent optional number as `null`. A period is written
//! as its first month (`"YYYY-MM"`) and its grain (`"M"`, `"Q"`, `"S"` or
//! `"Y"`).
//!
//! ```json
//! {"format": "risk_rs.chain_ladder_fits", "format_version": 1,
//!  "keys": ["lob"], "labels": [["Auto"]],
//!  "fits": [{"origins": [["2020-01", "Y"], ...], "development": {...},
//!            "tail": {...}, "cdf": [...], ...}]}
//! ```

use prospicio_core::Period;
use serde_json::{Value, json};

use prospicio_prob::PredictiveDistribution;

use crate::assumptions::LinkRatios;
use crate::bootstrap::{OdpBootstrapFit, OdpBootstrapFits, OdpBootstrapSegment};
use crate::chain_ladder::ChainLadderFit;
use crate::development::DevelopmentFit;
use crate::error::{Error, Result};
use crate::expected_loss::{CapeCodFit, ExpectedLossFit};
use crate::mack::MackFit;
use crate::mack_bootstrap::{MackBootstrapFit, MackBootstrapFits, MackBootstrapSegment};
use crate::pattern_draws::{DevelopmentPatterns, SegmentPatterns};
use crate::segments::SegmentFits;
use crate::tail::TailFit;
use crate::triangle::Label;
use crate::triangle::json::{
    grain_code, grain_from_json, month_from_json, month_to_json, number_from_json, number_to_json,
};

/// The format version this build writes and the newest it reads.
const FORMAT_VERSION: u64 = 1;

type Read<T> = std::result::Result<T, String>;

/// `to_json` and `from_json` on a type, from its writer (`$value`) and
/// reader (`$from`): a document of format `$format`, named `$what` in
/// errors.
macro_rules! json_document {
    ($(#[$doc:meta])* $ty:ty, $format:expr, $what:expr, $value:path, $from:path) => {
        impl $ty {
            $(#[$doc])*
            pub fn to_json(&self) -> String {
                let mut doc = $value(self);
                let obj = doc.as_object_mut().expect("an object");
                obj.insert("format".into(), json!($format));
                obj.insert("format_version".into(), json!(FORMAT_VERSION));
                serde_json::to_string(&doc).expect("a JSON value serializes")
            }

            /// Reads a document written by [`to_json`](Self::to_json). Fails
            /// on malformed JSON, another format, a newer format version, or
            /// parts that do not fit together.
            pub fn from_json(text: &str) -> Result<Self> {
                document(text, $format)
                    .and_then(|doc| $from(&doc))
                    .map_err(|msg| data_error($what, msg))
            }
        }
    };
}

/// [`json_document!`] on a fit type (format `$one`) and on the fits of
/// every segment (format `$many`, with the key names and labels).
macro_rules! json_fit {
    ($(#[$doc:meta])* $fit:ty, $one:literal, $many:literal, $what:literal, $value:path, $from:path) => {
        json_document!($(#[$doc])* $fit, $one, $what, $value, $from);

        impl SegmentFits<$fit> {
            /// The fits of every segment as one JSON document, with the key
            /// names and labels; [`from_json`](Self::from_json) reads it
            /// back.
            pub fn to_json(&self) -> String {
                segments_to_json(self, $many, $value)
            }

            /// Reads a document written by [`to_json`](Self::to_json).
            pub fn from_json(text: &str) -> Result<Self> {
                document(text, $many)
                    .and_then(|doc| segments_from_value(&doc, $from))
                    .map_err(|msg| data_error(concat!($what, "s"), msg))
            }
        }
    };
}

json_fit!(
    /// This fit as a JSON document; [`from_json`](Self::from_json) reads it
    /// back to the same fit, every number bit for bit (a NaN sigma
    /// included, which `==` would not call equal).
    ///
    /// ```
    /// use prospicio_reserving::{ChainLadder, ChainLadderFit, DevelopmentColumn, Grain, Long, Month, Triangle};
    ///
    /// let origin = [2020, 2020, 2021].map(Month::january);
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&[12, 24, 12]),
    ///     values: &[("paid", &[100.0, 150.0, 110.0])],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })?;
    /// let fit = ChainLadder::default().fit(&tri, "paid")?;
    /// let back = ChainLadderFit::from_json(&fit.to_json())?;
    /// assert_eq!(back.ultimate, fit.ultimate);
    /// assert_eq!(back.to_json(), fit.to_json());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ChainLadderFit,
    "risk_rs.chain_ladder_fit",
    "risk_rs.chain_ladder_fits",
    "chain-ladder fit",
    chain_ladder_value,
    chain_ladder_from_value
);

json_fit!(
    /// This fit as a JSON document, its chain ladder and the link ratios
    /// that Mack's assumption tests read included;
    /// [`from_json`](Self::from_json) reads it back to the same fit, every
    /// number bit for bit.
    MackFit,
    "risk_rs.mack_fit",
    "risk_rs.mack_fits",
    "Mack fit",
    mack_value,
    mack_from_value
);

json_fit!(
    /// This fit (an expected-loss, Bornhuetter-Ferguson or Benktander fit)
    /// as a JSON document, its chain ladder included;
    /// [`from_json`](Self::from_json) reads it back to the same fit, every
    /// number bit for bit.
    ExpectedLossFit,
    "risk_rs.expected_loss_fit",
    "risk_rs.expected_loss_fits",
    "expected-loss fit",
    expected_loss_value,
    expected_loss_from_value
);

json_fit!(
    /// This fit as a JSON document, its expected-loss fit and the trended
    /// apriori included; [`from_json`](Self::from_json) reads it back to the
    /// same fit, every number bit for bit.
    CapeCodFit,
    "risk_rs.cape_cod_fit",
    "risk_rs.cape_cod_fits",
    "Cape Cod fit",
    cape_cod_value,
    cape_cod_from_value
);

json_document!(
    /// This fit as a JSON document: its chain ladder, fitted values,
    /// residuals and scale, the simulated reserves, the patterns each
    /// simulation drew and the cells if the fit kept them;
    /// [`from_json`](Self::from_json) reads it back to the same fit, every
    /// number bit for bit.
    OdpBootstrapFit,
    "risk_rs.odp_bootstrap_fit",
    "ODP bootstrap fit",
    odp_bootstrap_value,
    odp_bootstrap_from_value
);

json_document!(
    /// The fits of every segment as one JSON document: each segment's chain
    /// ladder, fitted values, residuals and scale with the key names and
    /// labels, the joint reserves, the patterns and the cells if kept;
    /// [`from_json`](Self::from_json) reads it back, every number bit for
    /// bit.
    OdpBootstrapFits,
    "risk_rs.odp_bootstrap_fits",
    "ODP bootstrap fits",
    odp_bootstrap_fits_value,
    odp_bootstrap_fits_from_value
);

json_document!(
    /// This fit as a JSON document: its Mack model, residuals, the
    /// simulated reserves, the patterns each simulation drew and the cells
    /// if the fit kept them; [`from_json`](Self::from_json) reads it back to
    /// the same fit, every number bit for bit.
    MackBootstrapFit,
    "risk_rs.mack_bootstrap_fit",
    "Mack bootstrap fit",
    mack_bootstrap_value,
    mack_bootstrap_from_value
);

json_document!(
    /// The fits of every segment as one JSON document: each segment's Mack
    /// model and residuals with the key names and labels, the joint
    /// reserves, the patterns and the cells if kept;
    /// [`from_json`](Self::from_json) reads it back, every number bit for
    /// bit.
    MackBootstrapFits,
    "risk_rs.mack_bootstrap_fits",
    "Mack bootstrap fits",
    mack_bootstrap_fits_value,
    mack_bootstrap_fits_from_value
);

/// The fits of every segment as a document of format `format`, each fit
/// written by `fit`.
pub(crate) fn segments_to_json<T>(
    fits: &SegmentFits<T>,
    format: &str,
    fit: impl Fn(&T) -> Value,
) -> String {
    let mut doc = segments_value(fits, fit);
    let obj = doc.as_object_mut().expect("an object");
    obj.insert("format".into(), json!(format));
    obj.insert("format_version".into(), json!(FORMAT_VERSION));
    serde_json::to_string(&doc).expect("a JSON value serializes")
}

/// The key names, labels and each segment's fit written by `fit`, the
/// fields [`segments_from_value`] reads.
pub(crate) fn segments_value<T>(fits: &SegmentFits<T>, fit: impl Fn(&T) -> Value) -> Value {
    json!({
        "keys": fits.key_names,
        "labels": fits.labels.iter().map(Label::parts).collect::<Vec<_>>(),
        "fits": fits.fits.iter().map(fit).collect::<Vec<_>>(),
    })
}

fn data_error(what: &str, msg: String) -> Error {
    Error::Core(prospicio_core::Error::Data(format!("{what} JSON: {msg}")))
}

/// The parsed document, checked to be `format` at a version this build
/// reads.
pub(crate) fn document(text: &str, format: &str) -> Read<Value> {
    let doc: Value = serde_json::from_str(text).map_err(|e| format!("not a JSON document: {e}"))?;
    if doc.get("format").and_then(Value::as_str) != Some(format) {
        return Err(format!("format is not {format:?}"));
    }
    match doc.get("format_version").and_then(Value::as_u64) {
        Some(v) if v <= FORMAT_VERSION => Ok(doc),
        v => Err(format!("format version {v:?} is not supported")),
    }
}

/// Key names, labels and one fit per label, each read by `fit`.
pub(crate) fn segments_from_value<T>(
    doc: &Value,
    fit: impl Fn(&Value) -> Read<T>,
) -> Read<SegmentFits<T>> {
    let key_names = strings(field(doc, "keys")?, "\"keys\"")?;
    let labels = array(doc, "labels")?
        .iter()
        .map(|l| strings(l, "a label").map(Label::new))
        .collect::<Read<Vec<_>>>()?;
    if labels.iter().any(|l| l.parts().len() != key_names.len()) {
        return Err("a label does not have one part per key".into());
    }
    let fits = array(doc, "fits")?
        .iter()
        .map(fit)
        .collect::<Read<Vec<_>>>()?;
    if fits.len() != labels.len() {
        return Err("\"fits\" does not have one fit per label".into());
    }
    Ok(SegmentFits {
        key_names,
        labels,
        fits,
    })
}

fn chain_ladder_value(fit: &ChainLadderFit) -> Value {
    json!({
        "origins": fit.origins.iter().map(|&p| period_to_json(p)).collect::<Vec<_>>(),
        "development": development_value(&fit.development),
        "tail": tail_value(&fit.tail),
        "cdf": numbers(&fit.cdf),
        "latest_position": fit.latest_position,
        "latest": numbers(&fit.latest),
        "ultimate": numbers(&fit.ultimate),
        "pattern_rescale": fit.pattern_rescale.map(number_to_json),
        "pattern_tail": fit.pattern_tail.map(number_to_json),
    })
}

fn chain_ladder_from_value(v: &Value) -> Read<ChainLadderFit> {
    let origins = array(v, "origins")?
        .iter()
        .map(period_from_json)
        .collect::<Read<Vec<_>>>()?;
    let fit = ChainLadderFit {
        development: development_from_value(field(v, "development")?)?,
        tail: tail_from_value(field(v, "tail")?)?,
        cdf: floats(v, "cdf")?,
        latest_position: counts(v, "latest_position")?,
        latest: floats(v, "latest")?,
        ultimate: floats(v, "ultimate")?,
        pattern_rescale: optional_float(v, "pattern_rescale")?,
        pattern_tail: optional_float(v, "pattern_tail")?,
        origins,
    };
    let n = fit.origins.len();
    if [
        fit.latest_position.len(),
        fit.latest.len(),
        fit.ultimate.len(),
    ] != [n; 3]
    {
        return Err("the per-origin vectors do not have one entry per origin".into());
    }
    Ok(fit)
}

fn mack_value(fit: &MackFit) -> Value {
    json!({
        "chain_ladder": chain_ladder_value(&fit.chain_ladder),
        "process_risk": numbers(&fit.process_risk),
        "parameter_risk": numbers(&fit.parameter_risk),
        "standard_error": numbers(&fit.standard_error),
        "total_process_risk": number_to_json(fit.total_process_risk),
        "total_parameter_risk": number_to_json(fit.total_parameter_risk),
        "total_standard_error": number_to_json(fit.total_standard_error),
        "exclusions_set": fit.exclusions_set,
        "links": links_value(&fit.links),
    })
}

fn mack_from_value(v: &Value) -> Read<MackFit> {
    let fit = MackFit {
        chain_ladder: chain_ladder_from_value(field(v, "chain_ladder")?)?,
        process_risk: floats(v, "process_risk")?,
        parameter_risk: floats(v, "parameter_risk")?,
        standard_error: floats(v, "standard_error")?,
        total_process_risk: number_from_json(field(v, "total_process_risk")?)?,
        total_parameter_risk: number_from_json(field(v, "total_parameter_risk")?)?,
        total_standard_error: number_from_json(field(v, "total_standard_error")?)?,
        exclusions_set: field(v, "exclusions_set")?
            .as_bool()
            .ok_or("\"exclusions_set\" is not a boolean")?,
        links: links_from_value(field(v, "links")?)?,
    };
    let n = fit.chain_ladder.origins.len();
    if [
        fit.process_risk.len(),
        fit.parameter_risk.len(),
        fit.standard_error.len(),
    ] != [n; 3]
    {
        return Err("the per-origin standard errors do not have one entry per origin".into());
    }
    Ok(fit)
}

fn expected_loss_value(fit: &ExpectedLossFit) -> Value {
    json!({
        "chain_ladder": chain_ladder_value(&fit.chain_ladder),
        "exposure": numbers(&fit.exposure),
        "apriori": numbers(&fit.apriori),
        "ultimate": numbers(&fit.ultimate),
    })
}

fn expected_loss_from_value(v: &Value) -> Read<ExpectedLossFit> {
    let fit = ExpectedLossFit {
        chain_ladder: chain_ladder_from_value(field(v, "chain_ladder")?)?,
        exposure: floats(v, "exposure")?,
        apriori: floats(v, "apriori")?,
        ultimate: floats(v, "ultimate")?,
    };
    let n = fit.chain_ladder.origins.len();
    if [fit.exposure.len(), fit.apriori.len(), fit.ultimate.len()] != [n; 3] {
        return Err("the per-origin vectors do not have one entry per origin".into());
    }
    Ok(fit)
}

fn cape_cod_value(fit: &CapeCodFit) -> Value {
    json!({
        "expected_loss": expected_loss_value(&fit.expected_loss),
        "trended_apriori": numbers(&fit.trended_apriori),
    })
}

fn cape_cod_from_value(v: &Value) -> Read<CapeCodFit> {
    let fit = CapeCodFit {
        expected_loss: expected_loss_from_value(field(v, "expected_loss")?)?,
        trended_apriori: floats(v, "trended_apriori")?,
    };
    if fit.trended_apriori.len() != fit.expected_loss.chain_ladder.origins.len() {
        return Err("\"trended_apriori\" does not have one entry per origin".into());
    }
    Ok(fit)
}

fn odp_segment_value(s: &OdpBootstrapSegment) -> Value {
    json!({
        "chain_ladder": chain_ladder_value(&s.chain_ladder),
        "fitted": numbers(&s.fitted),
        "residuals": numbers(&s.residuals),
        "scale": number_to_json(s.scale),
    })
}

fn odp_segment_from_value(v: &Value) -> Read<OdpBootstrapSegment> {
    let s = OdpBootstrapSegment {
        chain_ladder: chain_ladder_from_value(field(v, "chain_ladder")?)?,
        fitted: floats(v, "fitted")?,
        residuals: floats(v, "residuals")?,
        scale: number_from_json(field(v, "scale")?)?,
    };
    if s.fitted.len() != s.residuals.len() {
        return Err("\"fitted\" and \"residuals\" differ in length".into());
    }
    Ok(s)
}

fn odp_bootstrap_value(fit: &OdpBootstrapFit) -> Value {
    json!({
        "chain_ladder": chain_ladder_value(&fit.chain_ladder),
        "fitted": numbers(&fit.fitted),
        "residuals": numbers(&fit.residuals),
        "scale": number_to_json(fit.scale),
        "reserves": distribution_value(&fit.reserves),
        "patterns": patterns_value(&fit.patterns),
        "cells": fit.cells.as_ref().map(distribution_value),
    })
}

fn odp_bootstrap_from_value(v: &Value) -> Read<OdpBootstrapFit> {
    let s = odp_segment_from_value(v)?;
    Ok(OdpBootstrapFit {
        chain_ladder: s.chain_ladder,
        fitted: s.fitted,
        residuals: s.residuals,
        scale: s.scale,
        reserves: distribution_from_value(field(v, "reserves")?)?,
        patterns: patterns_from_value(field(v, "patterns")?)?,
        cells: optional_distribution(v, "cells")?,
    })
}

fn odp_bootstrap_fits_value(fits: &OdpBootstrapFits) -> Value {
    json!({
        "segments": segments_value(&fits.segments, odp_segment_value),
        "reserves": distribution_value(&fits.reserves),
        "patterns": patterns_value(&fits.patterns),
        "cells": fits.cells.as_ref().map(distribution_value),
    })
}

fn odp_bootstrap_fits_from_value(v: &Value) -> Read<OdpBootstrapFits> {
    Ok(OdpBootstrapFits {
        segments: segments_from_value(field(v, "segments")?, odp_segment_from_value)?,
        reserves: distribution_from_value(field(v, "reserves")?)?,
        patterns: patterns_from_value(field(v, "patterns")?)?,
        cells: optional_distribution(v, "cells")?,
    })
}

fn mack_segment_value(s: &MackBootstrapSegment) -> Value {
    json!({
        "mack": mack_value(&s.mack),
        "residuals": numbers(&s.residuals),
    })
}

fn mack_segment_from_value(v: &Value) -> Read<MackBootstrapSegment> {
    Ok(MackBootstrapSegment {
        mack: mack_from_value(field(v, "mack")?)?,
        residuals: floats(v, "residuals")?,
    })
}

fn mack_bootstrap_value(fit: &MackBootstrapFit) -> Value {
    json!({
        "mack": mack_value(&fit.mack),
        "residuals": numbers(&fit.residuals),
        "reserves": distribution_value(&fit.reserves),
        "patterns": patterns_value(&fit.patterns),
        "cells": fit.cells.as_ref().map(distribution_value),
    })
}

fn mack_bootstrap_from_value(v: &Value) -> Read<MackBootstrapFit> {
    let s = mack_segment_from_value(v)?;
    Ok(MackBootstrapFit {
        mack: s.mack,
        residuals: s.residuals,
        reserves: distribution_from_value(field(v, "reserves")?)?,
        patterns: patterns_from_value(field(v, "patterns")?)?,
        cells: optional_distribution(v, "cells")?,
    })
}

fn mack_bootstrap_fits_value(fits: &MackBootstrapFits) -> Value {
    json!({
        "segments": segments_value(&fits.segments, mack_segment_value),
        "reserves": distribution_value(&fits.reserves),
        "patterns": patterns_value(&fits.patterns),
        "cells": fits.cells.as_ref().map(distribution_value),
    })
}

fn mack_bootstrap_fits_from_value(v: &Value) -> Read<MackBootstrapFits> {
    Ok(MackBootstrapFits {
        segments: segments_from_value(field(v, "segments")?, mack_segment_from_value)?,
        reserves: distribution_from_value(field(v, "reserves")?)?,
        patterns: patterns_from_value(field(v, "patterns")?)?,
        cells: optional_distribution(v, "cells")?,
    })
}

/// A distribution as the document [`PredictiveDistribution::to_json`]
/// writes, embedded.
fn distribution_value(d: &PredictiveDistribution) -> Value {
    serde_json::from_str(&d.to_json()).expect("a distribution writes JSON")
}

fn distribution_from_value(v: &Value) -> Read<PredictiveDistribution> {
    PredictiveDistribution::from_json(&v.to_string()).map_err(|e| e.to_string())
}

fn optional_distribution(v: &Value, name: &str) -> Read<Option<PredictiveDistribution>> {
    match field(v, name)? {
        Value::Null => Ok(None),
        d => distribution_from_value(d).map(Some),
    }
}

fn patterns_value(p: &DevelopmentPatterns) -> Value {
    json!({
        "n_sims": p.n_sims,
        "segments": p.segments.iter().map(|s| json!({
            "label": s.label,
            "origin_grain": grain_code(s.origin_grain),
            "ages": s.ages,
            "draws": numbers(&s.draws),
        })).collect::<Vec<_>>(),
    })
}

fn patterns_from_value(v: &Value) -> Read<DevelopmentPatterns> {
    let n_sims = count(field(v, "n_sims")?, "\"n_sims\"")?;
    let segments = array(v, "segments")?
        .iter()
        .map(|s| {
            let segment = SegmentPatterns {
                label: strings(field(s, "label")?, "\"label\"")?,
                origin_grain: grain_from_json(field(s, "origin_grain")?)?,
                ages: ages(s, "ages")?,
                draws: floats(s, "draws")?,
            };
            if segment.draws.len() != n_sims * segment.ages.len() {
                return Err(
                    "a segment's patterns do not have one draw per simulation and age".into(),
                );
            }
            Ok(segment)
        })
        .collect::<Read<Vec<_>>>()?;
    Ok(DevelopmentPatterns { n_sims, segments })
}

fn links_value(l: &LinkRatios) -> Value {
    json!({
        "n_origins": l.n_origins,
        "n_links": l.n_links,
        "ratios": l.ratios.iter().map(|r| r.map(number_to_json)).collect::<Vec<_>>(),
        "valuations": l.valuations.iter().map(|&m| month_to_json(m)).collect::<Vec<_>>(),
        "ages": l.ages,
    })
}

fn links_from_value(v: &Value) -> Read<LinkRatios> {
    let n_origins = count(field(v, "n_origins")?, "\"n_origins\"")?;
    let n_links = count(field(v, "n_links")?, "\"n_links\"")?;
    let ratios = array(v, "ratios")?
        .iter()
        .map(|r| match r {
            Value::Null => Ok(None),
            r => number_from_json(r).map(Some),
        })
        .collect::<Read<Vec<_>>>()?;
    let valuations = array(v, "valuations")?
        .iter()
        .map(month_from_json)
        .collect::<Read<Vec<_>>>()?;
    let ages = ages(v, "ages")?;
    if ratios.len() != n_origins * n_links
        || valuations.len() != ratios.len()
        || ages.len() != n_links
    {
        return Err("the link ratios do not have one entry per origin and link".into());
    }
    Ok(LinkRatios {
        n_origins,
        n_links,
        ratios,
        valuations,
        ages,
    })
}

/// The ages (whole months) in the array `name`.
fn ages(v: &Value, name: &str) -> Read<Vec<u32>> {
    array(v, name)?
        .iter()
        .map(|a| {
            a.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("{name:?} holds a non-age"))
        })
        .collect()
}

pub(crate) fn development_value(d: &DevelopmentFit) -> Value {
    json!({
        "development": d.development,
        "ldf": numbers(&d.ldf),
        "sigma": numbers(&d.sigma),
        "std_err": numbers(&d.std_err),
        "volume": numbers(&d.volume),
        "alpha": number_to_json(d.alpha),
        "n_links": d.n_links,
        "excluded": d.excluded,
        "selected": d.selected.iter().map(|s| s.map(number_to_json)).collect::<Vec<_>>(),
        "weight": numbers(&d.weight),
    })
}

pub(crate) fn development_from_value(v: &Value) -> Read<DevelopmentFit> {
    let development = ages(v, "development")?;
    let selected = array(v, "selected")?
        .iter()
        .map(|s| match s {
            Value::Null => Ok(None),
            s => number_from_json(s).map(Some),
        })
        .collect::<Read<Vec<_>>>()?;
    Ok(DevelopmentFit {
        development,
        ldf: floats(v, "ldf")?,
        sigma: floats(v, "sigma")?,
        std_err: floats(v, "std_err")?,
        volume: floats(v, "volume")?,
        alpha: number_from_json(field(v, "alpha")?)?,
        n_links: counts(v, "n_links")?,
        excluded: field(v, "excluded")?
            .as_bool()
            .ok_or("\"excluded\" is not a boolean")?,
        selected,
        weight: floats(v, "weight")?,
    })
}

pub(crate) fn tail_value(t: &TailFit) -> Value {
    json!({
        "attachment": t.attachment,
        "ldf": numbers(&t.ldf),
        "factor": number_to_json(t.factor),
        "sigma": number_to_json(t.sigma),
        "std_err": number_to_json(t.std_err),
    })
}

pub(crate) fn tail_from_value(v: &Value) -> Read<TailFit> {
    Ok(TailFit {
        attachment: count(field(v, "attachment")?, "\"attachment\"")?,
        ldf: floats(v, "ldf")?,
        factor: number_from_json(field(v, "factor")?)?,
        sigma: number_from_json(field(v, "sigma")?)?,
        std_err: number_from_json(field(v, "std_err")?)?,
    })
}

pub(crate) fn period_to_json(p: Period) -> Value {
    json!([month_to_json(p.start()), grain_code(p.grain())])
}

pub(crate) fn period_from_json(v: &Value) -> Read<Period> {
    match v.as_array().map(Vec::as_slice) {
        Some([start, grain]) => {
            let start = month_from_json(start)?;
            let grain = grain_from_json(grain)?;
            let p = Period::containing(start, grain);
            if p.start() != start {
                return Err(format!("{v} does not start a period of its grain"));
            }
            Ok(p)
        }
        _ => Err(format!("{v} is not a period ([\"YYYY-MM\", grain])")),
    }
}

pub(crate) fn numbers(x: &[f64]) -> Vec<Value> {
    x.iter().map(|&v| number_to_json(v)).collect()
}

pub(crate) fn field<'a>(v: &'a Value, name: &str) -> Read<&'a Value> {
    v.get(name).ok_or_else(|| format!("missing {name:?}"))
}

pub(crate) fn array<'a>(v: &'a Value, name: &str) -> Read<&'a Vec<Value>> {
    field(v, name)?
        .as_array()
        .ok_or_else(|| format!("{name:?} is not an array"))
}

pub(crate) fn floats(v: &Value, name: &str) -> Read<Vec<f64>> {
    array(v, name)?.iter().map(number_from_json).collect()
}

pub(crate) fn optional_float(v: &Value, name: &str) -> Read<Option<f64>> {
    match field(v, name)? {
        Value::Null => Ok(None),
        x => number_from_json(x).map(Some),
    }
}

pub(crate) fn count(v: &Value, what: &str) -> Read<usize> {
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("{what} is not a count"))
}

pub(crate) fn counts(v: &Value, name: &str) -> Read<Vec<usize>> {
    array(v, name)?
        .iter()
        .map(|c| count(c, &format!("an entry of {name:?}")))
        .collect()
}

pub(crate) fn strings(v: &Value, what: &str) -> Read<Vec<String>> {
    v.as_array()
        .ok_or_else(|| format!("{what} is not an array"))?
        .iter()
        .map(|s| {
            s.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{what} holds a non-string"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChainLadder, DevelopmentColumn, Grain, Long, Month, Tail, Triangle};

    fn triangle() -> Triangle {
        let origin = [2020, 2020, 2020, 2021, 2021, 2022, 2020, 2020, 2021].map(Month::january);
        Triangle::from_long(&Long {
            keys: &[("lob", &["A", "A", "A", "A", "A", "A", "B", "B", "B"])],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12, 12, 24, 12]),
            values: &[(
                "paid",
                &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0, 50.0, 80.0, 60.0],
            )],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn one_segment() -> Triangle {
        let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
            values: &[("paid", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn bits(x: &[f64]) -> Vec<u64> {
        x.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn round_trips_every_segment_bit_for_bit() {
        let cl = ChainLadder {
            tail: Tail::from(1.05),
            ..ChainLadder::default()
        };
        let fits = cl.fit_segments(&triangle(), "paid").unwrap();
        let back = SegmentFits::<ChainLadderFit>::from_json(&fits.to_json()).unwrap();
        assert_eq!(back.to_json(), fits.to_json());
        assert_eq!(
            (&back.key_names, &back.labels),
            (&fits.key_names, &fits.labels)
        );
        for (a, b) in back.fits.iter().zip(&fits.fits) {
            assert_eq!(bits(&a.ultimate), bits(&b.ultimate));
            assert_eq!(bits(&a.cdf), bits(&b.cdf));
            assert_eq!(bits(&a.development.sigma), bits(&b.development.sigma));
        }
        let one = &fits.fits[0];
        assert_eq!(
            ChainLadderFit::from_json(&one.to_json()).unwrap().to_json(),
            one.to_json()
        );
    }

    #[test]
    fn non_finite_and_absent_numbers_survive() {
        let mut fit = ChainLadder::default().fit(&one_segment(), "paid").unwrap();
        fit.development.sigma[0] = f64::NAN;
        fit.development.std_err[0] = f64::INFINITY;
        fit.development.selected = vec![None, Some(1.2)];
        fit.development.weight = vec![0.0, 1.0];
        fit.development.sigma[1] = 0.25;
        fit.pattern_rescale = Some(f64::NEG_INFINITY);
        let back = ChainLadderFit::from_json(&fit.to_json()).unwrap();
        assert!(back.development.sigma[0].is_nan());
        assert_eq!(back.development.std_err[0], f64::INFINITY);
        assert_eq!(back.development.selected, vec![None, Some(1.2)]);
        assert_eq!(back.pattern_rescale, Some(f64::NEG_INFINITY));
        assert_eq!(back.pattern_tail, None);
    }

    #[test]
    fn refuses_other_documents() {
        let fit = ChainLadder::default().fit(&one_segment(), "paid").unwrap();
        let text = fit.to_json();
        assert!(ChainLadderFit::from_json("{}").is_err());
        assert!(ChainLadderFit::from_json(&triangle().to_json()).is_err());
        assert!(
            ChainLadderFit::from_json(
                &text.replace("\"format_version\":1", "\"format_version\":2")
            )
            .is_err()
        );
        assert!(
            ChainLadderFit::from_json(&text.replace("[\"2021-01\",\"Y\"]", "[\"2021-02\",\"Y\"]"))
                .is_err()
        );
        let fits = ChainLadder::default()
            .fit_segments(&triangle(), "paid")
            .unwrap();
        assert!(
            SegmentFits::<ChainLadderFit>::from_json(
                &fits.to_json().replace("[\"B\"]", "[\"B\",\"x\"]")
            )
            .is_err()
        );
    }

    /// Four origins, four ages: what Mack needs, in two segments.
    fn square() -> Triangle {
        let years = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023];
        let ages = [12, 24, 36, 48, 12, 24, 36, 12, 24, 12];
        let a = [
            100.0, 150.0, 165.0, 170.0, 110.0, 172.0, 185.0, 120.0, 175.0, 130.0,
        ];
        let origin: Vec<Month> = years
            .iter()
            .chain(&years)
            .map(|&y| Month::january(y))
            .collect();
        let development: Vec<u32> = ages.iter().chain(&ages).copied().collect();
        let paid: Vec<f64> = a.iter().chain(&a.map(|x| x * 0.4 + 3.0)).copied().collect();
        let lob: Vec<&str> = (0..20).map(|i| if i < 10 { "A" } else { "B" }).collect();
        Triangle::from_long(&Long {
            keys: &[("lob", &lob)],
            origin: &origin,
            development: DevelopmentColumn::Age(&development),
            values: &[("paid", &paid)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn a_mack_fit_round_trips_with_its_assumption_tests() {
        let fits = crate::Mack::default()
            .fit_segments(&square(), "paid")
            .unwrap();
        let back = SegmentFits::<MackFit>::from_json(&fits.to_json()).unwrap();
        assert_eq!(back.to_json(), fits.to_json());
        for (a, b) in back.fits.iter().zip(&fits.fits) {
            assert_eq!(bits(&a.standard_error), bits(&b.standard_error));
            assert_eq!(
                a.total_standard_error.to_bits(),
                b.total_standard_error.to_bits()
            );
            assert_eq!(
                format!("{:?}", a.calendar_year_effect(0.5)),
                format!("{:?}", b.calendar_year_effect(0.5))
            );
        }
        let one = &fits.fits[1];
        let read = MackFit::from_json(&one.to_json()).unwrap();
        assert_eq!(read.to_json(), one.to_json());
        assert!(MackFit::from_json(&one.chain_ladder.to_json()).is_err());
        assert!(
            MackFit::from_json(&one.to_json().replace("\"n_links\":3", "\"n_links\":2")).is_err()
        );
    }

    /// `square()` with an exposure column.
    fn with_premium() -> Triangle {
        let years = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023];
        let ages = [12, 24, 36, 48, 12, 24, 36, 12, 24, 12];
        let paid = [
            100.0, 150.0, 165.0, 170.0, 110.0, 172.0, 185.0, 120.0, 175.0, 130.0,
        ];
        let premium = [
            250.0, 250.0, 250.0, 250.0, 260.0, 260.0, 260.0, 280.0, 280.0, 300.0,
        ];
        let origin = years.map(Month::january);
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("premium", &premium)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn expected_loss_and_cape_cod_fits_round_trip() {
        let tri = with_premium();
        let bf = crate::BornhuetterFerguson {
            apriori: crate::Apriori::from(0.7),
            ..Default::default()
        }
        .fit_segments(&tri, "paid", "premium")
        .unwrap();
        let back = SegmentFits::<ExpectedLossFit>::from_json(&bf.to_json()).unwrap();
        assert_eq!(back.to_json(), bf.to_json());
        assert_eq!(bits(&back.fits[0].ultimate), bits(&bf.fits[0].ultimate));
        let one = &bf.fits[0];
        assert_eq!(
            ExpectedLossFit::from_json(&one.to_json())
                .unwrap()
                .to_json(),
            one.to_json()
        );

        let cc = crate::CapeCod::default()
            .fit(&tri, "paid", "premium")
            .unwrap();
        let read = CapeCodFit::from_json(&cc.to_json()).unwrap();
        assert_eq!(read.to_json(), cc.to_json());
        assert_eq!(bits(&read.trended_apriori), bits(&cc.trended_apriori));
        assert!(CapeCodFit::from_json(&one.to_json()).is_err());
        assert!(ExpectedLossFit::from_json(&cc.to_json()).is_err());
    }

    #[test]
    fn bootstrap_fits_round_trip_with_their_simulations() {
        let odp = crate::OdpBootstrap {
            n_sims: 50,
            seed: 3,
            ..Default::default()
        };
        let one = odp.fit_with_cells(&one_segment(), "paid").unwrap();
        let back = OdpBootstrapFit::from_json(&one.to_json()).unwrap();
        assert_eq!(back.to_json(), one.to_json());
        assert_eq!(back.reserves.draw_matrix(), one.reserves.draw_matrix());
        assert_eq!(back.patterns, one.patterns);
        assert!(back.cells.is_some());
        let plain = odp.fit(&one_segment(), "paid").unwrap();
        assert!(
            OdpBootstrapFit::from_json(&plain.to_json())
                .unwrap()
                .cells
                .is_none()
        );
        let all = odp.fit_segments(&square(), "paid").unwrap();
        let read = OdpBootstrapFits::from_json(&all.to_json()).unwrap();
        assert_eq!(read.to_json(), all.to_json());
        assert_eq!(read.segments.labels, all.segments.labels);

        let mack = crate::MackBootstrap {
            n_sims: 50,
            seed: 3,
            ..Default::default()
        };
        let one = mack.fit_with_cells(&square_one(), "paid").unwrap();
        let back = MackBootstrapFit::from_json(&one.to_json()).unwrap();
        assert_eq!(back.to_json(), one.to_json());
        assert_eq!(back.reserves.draw_matrix(), one.reserves.draw_matrix());
        let all = mack.fit_segments(&square(), "paid").unwrap();
        let read = MackBootstrapFits::from_json(&all.to_json()).unwrap();
        assert_eq!(read.to_json(), all.to_json());
        assert!(OdpBootstrapFits::from_json(&all.to_json()).is_err());
        assert!(
            MackBootstrapFits::from_json(&all.to_json().replace("\"n_sims\":50", "\"n_sims\":49"))
                .is_err()
        );
    }

    /// The first segment of `square()`.
    fn square_one() -> Triangle {
        let years = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023];
        let ages = [12, 24, 36, 48, 12, 24, 36, 12, 24, 12];
        let paid = [
            100.0, 150.0, 165.0, 170.0, 110.0, 172.0, 185.0, 120.0, 175.0, 130.0,
        ];
        let origin = years.map(Month::january);
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }
}
