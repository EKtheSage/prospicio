//! Values crossing between JavaScript and Rust: options objects read with
//! serde, result objects and tables built by hand so that numeric columns
//! are `Float64Array`s.

use js_sys::{Array, Float64Array, Object, Reflect};
use prospicio_reserving::FitTable;
use serde::de::{DeserializeOwned, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::fmt;
use std::marker::PhantomData;
use wasm_bindgen::prelude::*;

/// A core error as a JavaScript `Error` carrying the core's message.
pub(crate) fn err(e: impl fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// An object argument: its field names as JavaScript spells them, so that
/// a mistyped or unknown one is an error. `serde-wasm-bindgen` reads the
/// fields a struct declares and ignores the rest, so `deny_unknown_fields`
/// alone would let `{ nSim: 500 }` through as the defaults.
pub(crate) trait Fields {
    const FIELDS: &'static [&'static str];
}

fn check_fields<T: Fields>(value: &JsValue, what: &str) -> Result<(), JsError> {
    if !value.is_object() || Array::is_array(value) {
        return Err(JsError::new(&format!("{what}: expected an object")));
    }
    for key in Object::keys(JsCast::unchecked_ref::<Object>(value)).iter() {
        let key = key.as_string().unwrap_or_default();
        if !T::FIELDS.contains(&key.as_str()) {
            return Err(JsError::new(&format!(
                "{what}: unknown field {key:?}; the fields are {}",
                T::FIELDS.join(", ")
            )));
        }
    }
    Ok(())
}

/// An options object read into `T`; `undefined` and `null` are `T`'s
/// defaults. An unknown or mistyped field is an error naming `what`.
pub(crate) fn options<T: DeserializeOwned + Default + Fields>(
    value: JsValue,
    what: &str,
) -> Result<T, JsError> {
    if value.is_undefined() || value.is_null() {
        return Ok(T::default());
    }
    read(value, what)
}

/// A required object argument read into `T`.
pub(crate) fn read<T: DeserializeOwned + Fields>(value: JsValue, what: &str) -> Result<T, JsError> {
    check_fields::<T>(&value, what)?;
    value_of(value, what)
}

/// Any value read into `T`.
pub(crate) fn value_of<T: DeserializeOwned>(value: JsValue, what: &str) -> Result<T, JsError> {
    serde_wasm_bindgen::from_value(value).map_err(|e| JsError::new(&format!("{what}: {e}")))
}

/// A JSON document given as text or as an object, as text.
pub(crate) fn json_text(document: &JsValue, what: &str) -> Result<String, JsError> {
    if let Some(text) = document.as_string() {
        return Ok(text);
    }
    js_sys::JSON::stringify(document)
        .ok()
        .and_then(|s| s.as_string())
        .ok_or_else(|| JsError::new(&format!("{what}: not a JSON document")))
}

/// A JavaScript object's own properties in their order, as pairs: a map
/// type would sort the names or need a hasher.
#[derive(Debug, Clone, Default)]
pub(crate) struct Pairs<T>(pub Vec<(String, T)>);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Pairs<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PairsVisitor<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> Visitor<'de> for PairsVisitor<T> {
            type Value = Pairs<T>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object of named columns")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some(entry) = map.next_entry::<String, T>()? {
                    out.push(entry);
                }
                Ok(Pairs(out))
            }
        }
        deserializer.deserialize_map(PairsVisitor(PhantomData))
    }
}

/// A numeric column going in: a missing value may be `null` or `NaN`.
pub(crate) fn numbers(values: &[Option<f64>]) -> Vec<f64> {
    values.iter().map(|v| v.unwrap_or(f64::NAN)).collect()
}

/// A result object under construction.
pub(crate) struct Obj(Object);

impl Obj {
    pub(crate) fn new() -> Self {
        Self(Object::new())
    }

    pub(crate) fn set(self, key: &str, value: impl Into<JsValue>) -> Self {
        // Setting a property of a plain object does not fail.
        let _ = Reflect::set(&self.0, &JsValue::from_str(key), &value.into());
        self
    }

    /// A numeric column or vector, as a `Float64Array`.
    pub(crate) fn numbers(self, key: &str, values: &[f64]) -> Self {
        self.set(key, Float64Array::from(values))
    }

    /// A column of labels, as an array of strings.
    pub(crate) fn strings<S: AsRef<str>>(
        self,
        key: &str,
        values: impl IntoIterator<Item = S>,
    ) -> Self {
        self.set(key, strings(values))
    }

    pub(crate) fn done(self) -> JsValue {
        self.0.into()
    }
}

/// An array of strings.
pub(crate) fn strings<S: AsRef<str>>(values: impl IntoIterator<Item = S>) -> Array {
    values
        .into_iter()
        .map(|s| JsValue::from_str(s.as_ref()))
        .collect()
}

/// A fit's long table as an object of columns: the key columns and
/// `origin` as strings, `age` in months, and each value column as a
/// `Float64Array`, under the core's column names.
pub(crate) fn table(t: &FitTable) -> JsValue {
    let mut o = Obj::new();
    for (name, values) in &t.keys {
        o = o.strings(name, values);
    }
    if let Some(origin) = &t.origin {
        o = o.strings("origin", origin.iter().map(ToString::to_string));
    }
    if let Some(age) = &t.age {
        let months: Vec<f64> = age.iter().map(|&a| f64::from(a)).collect();
        o = o.numbers("age", &months);
    }
    for (name, values) in &t.values {
        o = o.numbers(name, values);
    }
    o.done()
}
