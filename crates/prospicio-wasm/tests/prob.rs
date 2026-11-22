//! Distributions from their JSON documents, against the mpmath parity rows
//! of the lognormal's limited expected values.

mod common;

use common::{cases, message, object, ok};
use prospicio_prob::{Distribution, Lognormal};
use prospicio_wasm::Dist;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

const SEVERITY: &str = include_str!("../../../validation/reference/severity_mpmath.csv");

fn lognormal(meanlog: f64, sdlog: f64) -> Dist {
    ok(Dist::from_json(object(&[
        ("format", "risk_rs.distribution".into()),
        ("format_version", 1.into()),
        ("family", "lognormal".into()),
        ("meanlog", meanlog.into()),
        ("sdlog", sdlog.into()),
    ])))
}

#[wasm_bindgen_test]
fn a_lognormals_layers_match_mpmath() {
    let mut checked = 0;
    for c in cases(SEVERITY)
        .iter()
        .filter(|c| c.get("distribution") == "lognormal")
    {
        let d = lognormal(c.param("params", "meanlog"), c.param("params", "sdlog"));
        let arg = c.number("arg");
        let got = match c.get("quantity") {
            "lev" => ok(d.lev(arg)),
            "layer" => ok(d.layer(arg, c.number("arg2"))),
            _ => continue,
        };
        c.check(got);
        checked += 1;
    }
    assert_eq!(checked, 48);
}

#[wasm_bindgen_test]
fn a_distribution_round_trips_its_json_and_equals_the_core() {
    let d = lognormal(7.0, 0.5);
    assert_eq!(d.family(), "lognormal");
    let native = Lognormal::new(7.0, 0.5).unwrap();
    assert_eq!(d.mean(), native.mean());
    assert_eq!(d.variance(), native.variance());
    assert_eq!(d.standard_deviation(), native.std_dev());
    assert_eq!(d.cdf(1500.0), native.cdf(1500.0));
    assert_eq!(d.survival(1500.0), native.survival(1500.0));
    assert_eq!(ok(d.quantile(0.99)), native.quantile(0.99).unwrap());
    let (var, tvar) = (ok(d.quantile(0.99)), ok(d.tvar(0.99)));
    assert!(tvar > var);
    // From the saved text too.
    let again = ok(Dist::from_json(JsValue::from_str(&ok(d.to_json()))));
    assert_eq!(again.mean(), d.mean());
}

#[wasm_bindgen_test]
fn a_bad_document_or_level_is_an_error() {
    let unknown = Dist::from_json(object(&[
        ("format", "risk_rs.distribution".into()),
        ("format_version", 1.into()),
        ("family", "lognormle".into()),
    ]));
    assert!(message(unknown.err().unwrap()).contains("lognormle"));
    assert!(Dist::from_json(JsValue::from_f64(3.0)).is_err());
    assert!(lognormal(0.0, 1.0).quantile(1.5).is_err());
}
