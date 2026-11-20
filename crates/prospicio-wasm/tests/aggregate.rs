//! The compound distribution, the occurrence PML and a tower, against the
//! `aggregate` parity rows and the core called directly.

mod common;

use common::{cases, message, object, ok};
use prospicio_aggregate as core;
use prospicio_prob::{Dist as CoreDist, Lognormal, Poisson};
use prospicio_wasm::{Count, Dist, EventSet, Tower, fft, occurrence_pml, oep, simulate_events};
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

const SIZING: &str = include_str!("../../../validation/reference/sizing_aggregate.csv");
const OEP: &str = include_str!("../../../validation/reference/oep_aggregate.csv");

/// A lognormal with this mean and cv, through its JSON document.
fn lognormal(mean: f64, cv: f64) -> (Dist, CoreDist) {
    let native = CoreDist::from(Lognormal::from_mean_cv(mean, cv).unwrap());
    let d = ok(Dist::from_json(JsValue::from_str(
        &native.to_json().unwrap(),
    )));
    (d, native)
}

#[wasm_bindgen_test]
fn the_fft_aggregate_matches_aggregate_on_the_grid_sized_for_it() {
    let mut checked = 0;
    for c in cases(SIZING) {
        let params = c.get("params");
        if !params.contains("severity=lognormal") || params.contains("mixing_cv") {
            continue;
        }
        let n = ok(Count::poisson(c.param("params", "lambda")));
        let (sev, _) = lognormal(c.param("params", "mean"), c.param("params", "cv"));
        let agg = ok(fft(&n, &sev, JsValue::UNDEFINED));
        let got = match c.get("quantity") {
            "mean" => agg.mean(),
            "quantile" => ok(agg.quantile(c.number("arg"))),
            _ => continue,
        };
        c.check(got);
        checked += 1;
    }
    assert_eq!(checked, 5);
}

#[wasm_bindgen_test]
fn a_given_grid_gives_the_cores_masses() {
    let n = ok(Count::poisson(5.0));
    let (sev, native) = lognormal(100.0, 1.0);
    let agg = ok(fft(
        &n,
        &sev,
        object(&[("step", 0.5.into()), ("points", 8192.into())]),
    ));
    let severity = prospicio_prob::SeverityDist::try_from(native).unwrap();
    let (grid, _) = prospicio_prob::Grid::rounding(&severity, 0.5, 8192).unwrap();
    let (want, report) = core::fft(&Poisson::new(5.0).unwrap(), &grid, 8192).unwrap();
    assert_eq!(agg.probs().to_vec(), want.probs());
    assert_eq!(agg.step(), 0.5);
    assert_eq!(agg.tail_mass(), Some(report.tail_mass));
    assert_eq!(agg.aliasing_error(), Some(report.aliasing_error));
    assert!((agg.mean() / 500.0 - 1.0).abs() < 1e-3);
    assert!(ok(agg.tvar(0.99)) > ok(agg.quantile(0.99)));
    assert!(agg.cdf(500.0) > 0.5 && agg.standard_deviation() > 0.0 && agg.variance() > 0.0);
    let half = fft(&n, &sev, object(&[("step", 0.5.into())]));
    assert!(message(half.err().unwrap()).contains("both step and points"));
}

#[wasm_bindgen_test]
fn the_occurrence_pml_matches_aggregate() {
    let mut checked = 0;
    for c in cases(OEP) {
        let params = c.get("params");
        if !params.contains("severity=lognormal") || c.get("quantity") != "loss" {
            continue;
        }
        let n = ok(Count::poisson(c.param("params", "lambda")));
        let (sev, _) = lognormal(c.param("params", "mean"), c.param("params", "cv"));
        let loss = ok(occurrence_pml(&n, &sev, vec![c.number("arg")])).to_vec()[0];
        c.check(loss);
        // The curve the other way: the exceedance probability at that loss.
        let p = oep(&n, &sev, vec![loss]).to_vec()[0];
        assert!((p - c.number("arg")).abs() < 1e-9, "{p}");
        checked += 1;
    }
    assert!(checked >= 3, "{checked}");
}

#[wasm_bindgen_test]
fn a_tower_from_json_cedes_what_the_core_cedes() {
    let native = core::Tower::new(vec![
        core::Layer::xol("first", 200.0, 100.0).unwrap(),
        core::Layer::xol("second", 700.0, 300.0).unwrap(),
    ])
    .unwrap();
    let tower = ok(Tower::from_json(JsValue::from_str(&native.to_json())));
    assert_eq!(tower.to_json(), native.to_json());
    let n = ok(Count::negative_binomial(4.0, 0.5));
    let (sev, core_sev) = lognormal(100.0, 2.0);
    let events = ok(simulate_events(
        &n,
        &sev,
        object(&[("nSims", 2000.into()), ("seed", 11.into())]),
    ));
    assert_eq!(events.n_sims(), 2000);
    // The same years on the core: one random stream per year.
    let count = prospicio_prob::NegativeBinomial::new(4.0, 0.5).unwrap();
    let want_events = core::simulate_events(&count, &core_sev, 2000, 11).unwrap();
    assert_eq!(events.events(0).unwrap().to_vec(), want_events.events(0));
    assert_eq!(
        events.counts().iter().map(|&c| c as usize).sum::<usize>(),
        want_events.counts().iter().sum::<usize>()
    );
    let ceded = ok(tower.apply(&events));
    let want = native.apply(&want_events).unwrap();
    assert_eq!(ceded.draws().to_vec(), want.draw_matrix());
    assert_eq!(ceded.n_components(), want.n_components());
    // Each year's gross total, and events given as arrays of arrays.
    let totals = ok(events.totals());
    assert_eq!(totals.n_sims(), 2000);
    let years: js_sys::Array = [[5.0, 250.0].as_slice(), &[], &[900.0]]
        .iter()
        .map(|y| {
            JsValue::from(
                y.iter()
                    .map(|&x| JsValue::from_f64(x))
                    .collect::<js_sys::Array>(),
            )
        })
        .collect();
    let given = ok(EventSet::from_years(years.into(), None));
    assert_eq!(given.counts(), vec![2, 0, 1]);
    let small = ok(tower.apply(&given));
    assert_eq!(small.n_sims(), 3);
    assert!(given.events(3).is_err());
    assert!(Tower::from_json(JsValue::from_str("{}")).is_err());
}
