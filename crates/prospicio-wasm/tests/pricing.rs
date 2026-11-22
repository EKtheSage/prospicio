//! Distortions and the price of a cover, against the core's documented
//! values.

mod common;

use common::{get, message, ok};
use prospicio_wasm::{Distortion, price, price_at_cost_of_capital, version};
use wasm_bindgen_test::*;

fn number(o: &wasm_bindgen::JsValue, key: &str) -> f64 {
    get(o, key).as_f64().unwrap()
}

#[wasm_bindgen_test]
fn a_cover_is_priced_at_a_cost_of_capital() {
    // The example of `prospicio_pricing::risk_load::price`: four equally
    // likely years, assets at TVaR 50% (4), cost of capital 25%.
    let assets = ok(Distortion::from_name("tvar", 0.5, None));
    let p = ok(price_at_cost_of_capital(
        vec![0.0, 0.0, 2.0, 6.0],
        0.25,
        &assets,
    ));
    assert_eq!(number(&p, "expectedLoss"), 2.0);
    assert_eq!(number(&p, "assets"), 4.0);
    assert_eq!(number(&p, "premium"), (2.0 + 0.25 * 4.0) / 1.25);
    assert!((number(&p, "returnOnCapital") - 0.25).abs() < 1e-15);
    assert!((number(&p, "margin") - 0.4).abs() < 1e-15);
    assert!((number(&p, "capital") - 1.6).abs() < 1e-15);
    assert!((number(&p, "lossRatio") - 2.0 / 2.4).abs() < 1e-15);
    assert!(price_at_cost_of_capital(vec![0.0, 2.0], -0.1, &assets).is_err());
}

#[wasm_bindgen_test]
fn a_distortion_prices_draws_as_the_core_does() {
    let wang = ok(Distortion::from_name("wang", 0.5, None));
    assert_eq!(wang.family(), "wang");
    assert_eq!(wang.parameter(), Some(0.5));
    let native = prospicio_prob::Distortion::wang(0.5).unwrap();
    assert_eq!(wang.g(0.1), native.g(0.1));
    let losses = vec![0.0, 1.0, 3.0, 10.0, 2.0];
    let mut sorted = losses.clone();
    sorted.sort_by(f64::total_cmp);
    assert_eq!(ok(wang.apply(losses.clone())), native.apply_sorted(&sorted));
    // Premium by the Wang transform, assets at the maximum (TVaR 100% is
    // refused, so TVaR 80% of five draws: the largest).
    let assets = ok(Distortion::from_name("tvar", 0.8, None));
    let p = ok(price(losses, &wang, &assets));
    assert_eq!(number(&p, "assets"), 10.0);
    assert_eq!(number(&p, "premium"), native.apply_sorted(&sorted));
    assert!(number(&p, "premium") > number(&p, "expectedLoss"));
    // `aggregate`'s short names are the core's too.
    assert_eq!(
        ok(Distortion::from_name("ph", 0.7, None)).family(),
        "proportional_hazard"
    );
    let unknown = Distortion::from_name("wong", 0.5, None);
    assert!(message(unknown.err().unwrap()).contains("wong"));
}

#[wasm_bindgen_test]
fn the_module_reports_its_version() {
    assert_eq!(version(), env!("CARGO_PKG_VERSION"));
}
