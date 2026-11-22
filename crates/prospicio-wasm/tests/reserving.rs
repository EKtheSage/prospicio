//! The reserving functions as JavaScript calls them, run under Node by
//! `wasm-bindgen-test-runner`, against the R ChainLadder parity rows the
//! Rust core is checked against.

mod common;

use common::{cases, column, get, labels, message, object, ok};
use js_sys::{Array, Float64Array};
use prospicio_reserving as core;
use prospicio_wasm::{Triangle, bornhuetter_ferguson, chain_ladder, mack, odp_bootstrap};
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

/// GenIns (Taylor and Ashe), cumulative, annual, as a long table.
const GENINS: &str = include_str!("../../../validation/data/genins.csv");
const R_CHAIN_LADDER: &str =
    include_str!("../../../validation/reference/reserving_chainladder_r.csv");

fn genins_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut out = (Vec::new(), Vec::new(), Vec::new());
    for line in GENINS.lines().filter(|l| !l.starts_with('#')).skip(1) {
        let f: Vec<f64> = line.split(',').map(|v| v.parse().unwrap()).collect();
        out.0.push(f[0]);
        out.1.push(f[1]);
        out.2.push(f[2]);
    }
    out
}

fn numbers(values: &[f64]) -> JsValue {
    values
        .iter()
        .map(|&v| JsValue::from_f64(v))
        .collect::<Array>()
        .into()
}

/// GenIns from plain arrays, with a premium column of 5,000,000 a year.
fn genins() -> Triangle {
    let (origin, development, value) = genins_columns();
    let premium = vec![5_000_000.0; value.len()];
    let values = object(&[("paid", numbers(&value)), ("premium", numbers(&premium))]);
    ok(Triangle::from_long(object(&[
        ("origin", numbers(&origin)),
        ("development", numbers(&development)),
        ("values", values),
    ])))
}

#[wasm_bindgen_test]
fn a_triangle_reads_plain_and_typed_arrays_and_gives_its_long_table_back() {
    let tri = genins();
    assert_eq!(tri.shape(), vec![1, 2, 10, 10]);
    let names: Vec<String> = tri
        .columns()
        .iter()
        .map(|v| v.as_string().unwrap())
        .collect();
    assert_eq!(names, ["paid", "premium"]);
    assert_eq!(tri.development()[..3], [12, 24, 36]);
    assert_eq!(tri.valuation(), "2010-12");
    let (origin, development, value) = genins_columns();
    // The same table from typed arrays, with one measure under `value`.
    let typed = ok(Triangle::from_long(object(&[
        ("origin", Float64Array::from(origin.as_slice()).into()),
        (
            "development",
            Float64Array::from(development.as_slice()).into(),
        ),
        ("value", Float64Array::from(value.as_slice()).into()),
    ])));
    let long = typed.to_long();
    assert_eq!(column(&long, "value"), value);
    assert_eq!(column(&long, "development"), development);
    assert_eq!(labels(&long, "origin")[0], "2001-01");
    // Incremental and back.
    let back = typed.to_incremental().to_cumulative().to_long();
    for (a, b) in column(&back, "value").iter().zip(&value) {
        assert!((a - b).abs() <= 1e-9 * b.abs());
    }
}

#[wasm_bindgen_test]
fn the_chain_ladder_and_mack_match_r_chainladder_on_genins() {
    let tri = genins();
    let cl = ok(chain_ladder(&tri, "paid", JsValue::UNDEFINED));
    let mk = ok(mack(&tri, "paid", JsValue::UNDEFINED));
    let by_origin = get(&cl, "byOrigin");
    let origins = labels(&by_origin, "origin");
    let at = |arg: &str| origins.iter().position(|o| o == arg).unwrap();
    let mut checked = 0;
    for c in cases(R_CHAIN_LADDER)
        .iter()
        .filter(|c| c.get("dataset") == "genins")
    {
        let got = match (c.get("method"), c.get("quantity")) {
            ("chain_ladder", "reserve") => column(&by_origin, "reserve")[at(c.get("arg"))],
            ("chain_ladder", "ultimate") => column(&by_origin, "ultimate")[at(c.get("arg"))],
            ("chain_ladder", "total_reserve") => column(&get(&cl, "totals"), "reserve")[0],
            ("chain_ladder", "total_ultimate") => column(&get(&cl, "totals"), "ultimate")[0],
            ("mack", "se") => column(&get(&mk, "byOrigin"), "standard_error")[at(c.get("arg"))],
            ("mack", "total_standard_error") => column(&get(&mk, "totals"), "standard_error")[0],
            _ => continue,
        };
        c.check(got);
        checked += 1;
    }
    assert_eq!(checked, 10 + 10 + 2 + 10 + 1);
    // The development table: one row per age, the factors as columns.
    let development = get(&cl, "development");
    assert_eq!(column(&development, "age")[..2], [12.0, 24.0]);
    assert!(column(&development, "ldf")[0] > 3.0);
}

#[wasm_bindgen_test]
fn bornhuetter_ferguson_equals_the_core() {
    let tri = genins();
    let options = object(&[("exposure", "premium".into()), ("apriori", 0.9.into())]);
    let fit = ok(bornhuetter_ferguson(&tri, "paid", options));
    let (origin, development, value) = genins_columns();
    let months: Vec<core::Month> = origin
        .iter()
        .map(|&y| core::Month::january(y as i32))
        .collect();
    let ages: Vec<u32> = development.iter().map(|&a| a as u32).collect();
    let premium = vec![5_000_000.0; value.len()];
    let native = core::Triangle::from_long(&core::Long {
        keys: &[],
        origin: &months,
        development: core::DevelopmentColumn::Age(&ages),
        values: &[("paid", &value), ("premium", &premium)],
        origin_grain: core::Grain::Year,
        development_grain: core::Grain::Year,
        cumulative: true,
    })
    .unwrap();
    let want = core::BornhuetterFerguson {
        apriori: core::Apriori::All(0.9),
        chain_ladder: Default::default(),
    }
    .fit_segments(&native, "paid", "premium")
    .unwrap()
    .to_long();
    let by_origin = get(&fit, "byOrigin");
    for name in ["ultimate", "reserve", "ulr"] {
        assert_eq!(
            column(&by_origin, name),
            want.column(name).unwrap(),
            "{name}"
        );
    }
    // One a-priori ratio per origin, oldest first, as an array.
    let per_origin = object(&[
        ("exposure", "premium".into()),
        ("apriori", numbers(&[0.9; 10])),
    ]);
    let same = ok(bornhuetter_ferguson(&tri, "paid", per_origin));
    assert_eq!(
        column(&get(&same, "byOrigin"), "ultimate"),
        want.column("ultimate").unwrap()
    );
}

#[wasm_bindgen_test]
fn the_odp_bootstrap_draws_what_the_core_draws_for_a_seed() {
    let tri = genins();
    let options = object(&[("nSims", 400.into()), ("seed", 7.into())]);
    let boot = ok(odp_bootstrap(&tri, "paid", options));
    let reserves: prospicio_wasm::Predictive = {
        use wasm_bindgen::convert::TryFromJsValue;
        prospicio_wasm::Predictive::try_from_js_value(get(&boot, "reserves")).unwrap()
    };
    assert_eq!((reserves.n_sims(), reserves.n_components()), (400, 10));
    // The same fit on the core, with no JavaScript in between.
    let (origin, development, value) = genins_columns();
    let months: Vec<core::Month> = origin
        .iter()
        .map(|&y| core::Month::january(y as i32))
        .collect();
    let ages: Vec<u32> = development.iter().map(|&a| a as u32).collect();
    let native = core::Triangle::from_long(&core::Long {
        keys: &[],
        origin: &months,
        development: core::DevelopmentColumn::Age(&ages),
        values: &[("paid", &value)],
        origin_grain: core::Grain::Year,
        development_grain: core::Grain::Year,
        cumulative: true,
    })
    .unwrap();
    let want = core::OdpBootstrap {
        n_sims: 400,
        seed: 7,
        ..Default::default()
    }
    .fit_segments(&native, "paid")
    .unwrap();
    assert_eq!(reserves.draws().to_vec(), want.reserves.draw_matrix());
    // The total's risk measures are the core's, and ordered.
    let (mean, var, tvar) = (
        reserves.mean(),
        ok(reserves.var(0.99)),
        ok(reserves.tvar(0.99)),
    );
    assert!(mean < var && var <= tvar);
    assert!((mean / 18_680_855.61 - 1.0).abs() < 0.05, "{mean}");
    let summary = ok(reserves.summary());
    assert_eq!(labels(&summary, "origin").len(), 10);
    assert_eq!(column(&summary, "mean").len(), 10);
    // Summed over origins there is one component: the total.
    let total = ok(reserves.aggregate(Vec::new()));
    assert_eq!(total.n_components(), 1);
    assert_eq!(total.draws().to_vec(), reserves.total().to_vec());
}

#[wasm_bindgen_test]
fn an_error_is_a_javascript_error_with_the_cores_message() {
    let tri = genins();
    let unknown = chain_ladder(&tri, "incurred", JsValue::UNDEFINED)
        .err()
        .unwrap();
    assert!(message(unknown).contains("incurred"));
    let bad_option = mack(&tri, "paid", object(&[("tial", 1.05.into())]))
        .err()
        .unwrap();
    let text = message(bad_option);
    assert!(
        text.contains("mack options") && text.contains("tial"),
        "{text}"
    );
    let both = Triangle::from_long(object(&[
        ("origin", numbers(&[2020.0])),
        ("development", numbers(&[12.0])),
    ]));
    assert!(message(both.err().unwrap()).contains("value"));
    let bad_origin = Triangle::from_long(object(&[
        (
            "origin",
            [JsValue::from_str("soon")].iter().collect::<Array>().into(),
        ),
        ("development", numbers(&[12.0])),
        ("value", numbers(&[1.0])),
    ]));
    assert!(message(bad_origin.err().unwrap()).contains("soon"));
}

#[wasm_bindgen_test]
fn from_long_says_which_of_value_and_values_is_wrong() {
    let (origin, development, value) = genins_columns();
    let table = |extra: &[(&str, JsValue)]| {
        let mut entries = vec![
            ("origin", numbers(&origin)),
            ("development", numbers(&development)),
        ];
        entries.extend(extra.iter().cloned());
        object(&entries)
    };
    let fails = |t: JsValue| match Triangle::from_long(t) {
        Ok(_) => panic!("expected an error"),
        Err(e) => message(e),
    };
    // Neither: the message asks for one, and does not say "not both".
    let neither = fails(table(&[]));
    assert!(neither.contains("give value"), "{neither}");
    assert!(!neither.contains("not both"), "{neither}");
    let both = fails(table(&[
        ("value", numbers(&value)),
        ("values", object(&[("paid", numbers(&value))])),
    ]));
    assert!(both.contains("not both"), "{both}");
}

#[wasm_bindgen_test]
fn a_summary_refuses_a_dimension_named_as_one_of_its_columns() {
    // A key column called "mean" is a dimension of the simulated reserves:
    // the summary's own "mean" column would replace its labels.
    let (origin, development, value) = genins_columns();
    let lob: Array = value.iter().map(|_| JsValue::from_str("motor")).collect();
    let tri = ok(Triangle::from_long(object(&[
        ("origin", numbers(&origin)),
        ("development", numbers(&development)),
        ("value", numbers(&value)),
        ("keys", object(&[("mean", lob.into())])),
    ])));
    let options = object(&[("nSims", JsValue::from_f64(50.0))]);
    let fit = ok(odp_bootstrap(&tri, "value", options));
    let reserves = {
        use wasm_bindgen::convert::TryFromJsValue;
        prospicio_wasm::Predictive::try_from_js_value(get(&fit, "reserves")).unwrap()
    };
    let e = match reserves.summary() {
        Ok(_) => panic!("expected an error"),
        Err(e) => message(e),
    };
    assert!(e.contains("\"mean\"") && e.contains("aggregate"), "{e}");
    // With the key summed away the summary is by origin, as without keys.
    let by_origin = ok(reserves.aggregate(vec!["origin".to_owned()]));
    assert_eq!(labels(&ok(by_origin.summary()), "origin").len(), 10);
}
