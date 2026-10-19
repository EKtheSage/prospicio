//! The envelope every model artifact shares (`docs/design/models.md`,
//! "Model artifacts"): the format check and version rule, provenance, and
//! the JSON helpers the engines' readers and writers use. Non-finite
//! numbers are written as the strings `"NaN"`, `"inf"` and `"-inf"`; with
//! the workspace's `float_roundtrip` every other number round-trips bit
//! for bit.

use prospicio_core::{Error, Result};
use serde_json::{Map, Number, Value, json};

use crate::{Family, Link};

/// Provenance: the model, the writing crate and its version, and the hash
/// of the training data.
pub fn provenance(model: &str, krate: &str, version: &str, input_hash: &str) -> Value {
    json!({
        "model": model,
        "versions": [[krate, version]],
        "input_hash": input_hash,
    })
}

/// Parses `text` and checks its format and version (at most `newest`).
pub fn parse(text: &str, format: &str, newest: u64) -> Result<Value> {
    let doc: Value = serde_json::from_str(text)
        .map_err(|e| Error::Data(format!("model artifact is not JSON: {e}")))?;
    let top = object(&doc, "artifact")?;
    if top.get("format").and_then(Value::as_str) != Some(format) {
        return Err(Error::Data(format!("not a {format} artifact")));
    }
    let version = uint(top, "format_version")?;
    if version == 0 || version > newest {
        return Err(Error::Data(format!(
            "artifact format version {version}; this build reads 1 to {newest}"
        )));
    }
    Ok(doc)
}

/// The provenance's `input_hash`.
pub fn input_hash(top: &Map<String, Value>) -> Result<String> {
    let provenance = object(field(top, "provenance")?, "provenance")?;
    Ok(field(provenance, "input_hash")?
        .as_str()
        .ok_or_else(|| bad("provenance.input_hash"))?
        .to_string())
}

/// The array of strings at `key`.
pub fn strings(o: &Map<String, Value>, key: &str) -> Result<Vec<String>> {
    array(o, key)?
        .iter()
        .map(|v| v.as_str().map(String::from).ok_or_else(|| bad(key)))
        .collect()
}

/// The array at `key`.
pub fn array<'a>(o: &'a Map<String, Value>, key: &str) -> Result<&'a Vec<Value>> {
    field(o, key)?.as_array().ok_or_else(|| bad(key))
}

/// A family as `{"name", ...parameters}`.
pub fn family_json(f: Family) -> Value {
    match f {
        Family::NegativeBinomial { theta } => json!({"name": f.name(), "theta": num(theta)}),
        Family::Tweedie { power } => json!({"name": f.name(), "power": num(power)}),
        _ => json!({"name": f.name()}),
    }
}

/// The family [`family_json`] wrote.
pub fn family_from(o: &Map<String, Value>) -> Result<Family> {
    Ok(match field(o, "name")?.as_str() {
        Some("gaussian") => Family::Gaussian,
        Some("poisson") => Family::Poisson,
        Some("gamma") => Family::Gamma,
        Some("inverse_gaussian") => Family::InverseGaussian,
        Some("binomial") => Family::Binomial,
        Some("negative_binomial") => Family::NegativeBinomial {
            theta: float(field(o, "theta")?, "family.theta")?,
        },
        Some("tweedie") => Family::Tweedie {
            power: float(field(o, "power")?, "family.power")?,
        },
        _ => return Err(bad("spec.family.name")),
    })
}

/// A link as `{"name", ...parameters}`.
pub fn link_json(l: Link) -> Value {
    let name = match l {
        Link::Identity => "identity",
        Link::Log => "log",
        Link::Logit => "logit",
        Link::Probit => "probit",
        Link::Cloglog => "cloglog",
        Link::Inverse => "inverse",
        Link::InverseSquared => "inverse_squared",
        Link::Power(p) => return json!({"name": "power", "power": num(p)}),
    };
    json!({"name": name})
}

/// The link [`link_json`] wrote.
pub fn link_from(o: &Map<String, Value>) -> Result<Link> {
    Ok(match field(o, "name")?.as_str() {
        Some("identity") => Link::Identity,
        Some("log") => Link::Log,
        Some("logit") => Link::Logit,
        Some("probit") => Link::Probit,
        Some("cloglog") => Link::Cloglog,
        Some("inverse") => Link::Inverse,
        Some("inverse_squared") => Link::InverseSquared,
        Some("power") => Link::Power(float(field(o, "power")?, "link.power")?),
        _ => return Err(bad("spec.link.name")),
    })
}

/// A number, or the string `"NaN"`, `"inf"` or `"-inf"`.
pub fn num(x: f64) -> Value {
    match Number::from_f64(x) {
        Some(n) => Value::Number(n),
        None if x.is_nan() => Value::String("NaN".into()),
        None if x > 0.0 => Value::String("inf".into()),
        None => Value::String("-inf".into()),
    }
}

/// An array of [`num`]s.
pub fn nums(xs: &[f64]) -> Value {
    Value::Array(xs.iter().map(|&x| num(x)).collect())
}

/// The number [`num`] wrote.
pub fn float(v: &Value, what: &str) -> Result<f64> {
    match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| bad(what)),
        Value::String(s) => match s.as_str() {
            "NaN" => Ok(f64::NAN),
            "inf" => Ok(f64::INFINITY),
            "-inf" => Ok(f64::NEG_INFINITY),
            _ => Err(bad(what)),
        },
        _ => Err(bad(what)),
    }
}

/// The array of [`num`]s at `key`.
pub fn floats(o: &Map<String, Value>, key: &str) -> Result<Vec<f64>> {
    field(o, key)?
        .as_array()
        .ok_or_else(|| bad(key))?
        .iter()
        .map(|v| float(v, key))
        .collect()
}

/// The unsigned integer at `key`.
pub fn uint(o: &Map<String, Value>, key: &str) -> Result<u64> {
    field(o, key)?.as_u64().ok_or_else(|| bad(key))
}

/// The value at `key`, or an error naming it.
pub fn field<'a>(o: &'a Map<String, Value>, key: &str) -> Result<&'a Value> {
    o.get(key)
        .ok_or_else(|| Error::Data(format!("model artifact has no {key:?}")))
}

/// `v` as an object.
pub fn object<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    v.as_object().ok_or_else(|| bad(what))
}

/// The error for an invalid field.
pub fn bad(what: &str) -> Error {
    Error::Data(format!("model artifact field {what:?} is invalid"))
}
