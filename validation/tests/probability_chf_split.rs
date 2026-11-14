//! Stable laws with two power tails, split into their totally skewed halves
//! and tilted, against
//! `aggregate` 2.1.0's `AggregateCHF` (`validation/scripts/aggregate_chf_split.py`,
//! #780): the grid the crop places, the tilt on each side, the modelled edge
//! error, the outside masses, the truncation, the masses' sum and
//! quantiles, and the masses at 257 indices.

use prospicio_prob::Distribution;
use prospicio_prob::chf::{ChfDistribution, ChfLaw, ChfOptions, Tilt, invert};
use prospicio_prob::{Stable, StableParameterization};
use prospicio_validation::{check, reference};
use std::collections::HashMap;

#[test]
fn chf_split_stable_matches_aggregate() {
    let cases = reference("chf_split_aggregate.csv");
    let mut done: HashMap<String, ChfDistribution> = HashMap::new();
    check(&cases, |c| {
        let v = |name: &str| c.param("params", name);
        let d = done.entry(c.get("params").to_string()).or_insert_with(|| {
            let parameterization = if v("parameterization") == 1.0 {
                StableParameterization::S0
            } else {
                StableParameterization::S1
            };
            let st = Stable::new(v("alpha"), v("beta"), v("scale"), v("loc"), parameterization)
                .unwrap();
            // -1 is the automatic tilt, -2 none, anything else the e-folds.
            let code = v("tilt");
            let tilt = if code == -1.0 {
                Tilt::Auto
            } else if code == -2.0 {
                Tilt::Off
            } else {
                Tilt::Fixed(code)
            };
            let bs = v("bs");
            let options = ChfOptions {
                log2: v("log2") as u32,
                step: (bs > 0.0).then_some(bs),
                tilt,
                ..ChfOptions::default()
            };
            invert(&ChfLaw::Stable(st), &options).unwrap()
        });
        let r = d.report();
        match c.get("quantity") {
            "step" => Some(d.step()),
            "x_min" => Some(d.x_min()),
            "tilt_left" => Some(r.tilt.0),
            "tilt_right" => Some(r.tilt.1),
            "edge_error" => Some(r.edge_error),
            "truncation_error" => Some(r.truncation_error),
            "outside_below" => Some(r.outside_below),
            "outside_above" => Some(r.outside_above),
            "sum" => Some(r.total_mass),
            "quantile" => d.quantile(c.number("arg")?).ok(),
            "mass" => Some(d.masses()[c.number("arg")? as usize]),
            _ => None,
        }
    });
}
