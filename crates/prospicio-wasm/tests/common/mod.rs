//! Helpers shared by the tests: JavaScript values, and the parity rows of
//! `validation/reference/`, read at compile time since a WASM test has no
//! file system.
#![allow(dead_code)]

use js_sys::{Float64Array, Object, Reflect};
use wasm_bindgen::prelude::*;

/// A plain object from `(key, value)` pairs.
pub fn object(entries: &[(&str, JsValue)]) -> JsValue {
    let o = Object::new();
    for (key, value) in entries {
        Reflect::set(&o, &JsValue::from_str(key), value).unwrap();
    }
    o.into()
}

pub fn get(o: &JsValue, key: &str) -> JsValue {
    Reflect::get(o, &JsValue::from_str(key)).unwrap()
}

/// A `Float64Array` column of a table.
pub fn column(o: &JsValue, key: &str) -> Vec<f64> {
    let value = get(o, key);
    assert!(
        value.is_instance_of::<Float64Array>(),
        "{key} is not a Float64Array"
    );
    Float64Array::from(value).to_vec()
}

/// A column of strings.
pub fn labels(o: &JsValue, key: &str) -> Vec<String> {
    js_sys::Array::from(&get(o, key))
        .iter()
        .map(|v| v.as_string().unwrap())
        .collect()
}

/// The result of a binding, or a panic with the JavaScript error's message.
pub fn ok<T>(result: Result<T, JsError>) -> T {
    match result {
        Ok(value) => value,
        Err(e) => panic!("{}", message(e)),
    }
}

/// The `message` of the JavaScript `Error` a binding threw.
pub fn message(e: JsError) -> String {
    let value = JsValue::from(e);
    assert!(
        value.is_instance_of::<js_sys::Error>(),
        "not a JavaScript Error"
    );
    String::from(js_sys::Error::from(value).message())
}

/// One parity row of a reference file.
pub struct Case {
    fields: Vec<(String, String)>,
}

impl Case {
    pub fn get(&self, name: &str) -> &str {
        self.fields
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("no column {name}"))
    }

    pub fn number(&self, name: &str) -> f64 {
        self.get(name)
            .parse()
            .unwrap_or_else(|_| panic!("{name} is not a number"))
    }

    /// `key=value` inside a `;`-separated column.
    pub fn param(&self, column: &str, key: &str) -> f64 {
        self.get(column)
            .split(';')
            .find_map(|kv| kv.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
            .unwrap_or_else(|| panic!("no {key} in {column}"))
            .parse()
            .unwrap()
    }

    /// Whether `got` is within the row's own tolerance of its `expected`,
    /// by the rule of `prospicio_validation::Case::passes`.
    pub fn check(&self, got: f64) {
        let want = self.number("expected");
        let (abs_tol, rel_tol) = (self.number("abs_tol"), self.number("rel_tol"));
        let tol = abs_tol.max(rel_tol * want.abs());
        assert!(
            (got - want).abs() <= tol,
            "got {got}, want {want} (tolerance {tol}): {:?}",
            self.fields
        );
    }
}

/// The rows of a reference file's text: `#` lines skipped, the last
/// column (`source`) taking the rest of the line.
pub fn cases(text: &str) -> Vec<Case> {
    let mut lines = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    lines
        .map(|line| Case {
            fields: header
                .iter()
                .zip(line.splitn(header.len(), ','))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        })
        .collect()
}
