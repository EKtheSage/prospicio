//! The Tweedie law at any power from its characteristic function against
//! `aggregate` 2.1.0's `AggregateCHF` (`validation/scripts/aggregate_chf_tweedie.py`,
//! #780): `p` from -3 to 5 on the grid rule A chooses.

use prospicio_prob::Distribution;
use prospicio_prob::chf::{ChfDistribution, ChfLaw, ChfOptions, invert};
use prospicio_prob::levy::TweediePower;
use prospicio_validation::{check, reference};
use std::collections::HashMap;

#[test]
fn chf_tweedie_power_matches_aggregate() {
    let cases = reference("chf_tweedie_aggregate.csv");
    let mut done: HashMap<String, ChfDistribution> = HashMap::new();
    check(&cases, |c| {
        let v = |name: &str| c.param("params", name);
        let d = done.entry(c.get("params").to_string()).or_insert_with(|| {
            let law = ChfLaw::TweediePower(
                TweediePower::new(v("power"), v("mean"), v("dispersion")).unwrap(),
            );
            invert(&law, &ChfOptions::with_log2(v("log2") as u32)).unwrap()
        });
        let r = d.report();
        match c.get("quantity") {
            "step" => Some(d.step()),
            "x_min" => Some(d.x_min()),
            "truncation_error" => Some(r.truncation_error),
            "outside_below" => Some(r.outside_below),
            "outside_above" => Some(r.outside_above),
            "sum" => Some(r.total_mass),
            "mean" => Some(d.mean()),
            "sd" => Some(d.std_dev()),
            "quantile" => d.quantile(c.number("arg")?).ok(),
            "mass" => Some(d.masses()[c.number("arg")? as usize]),
            _ => None,
        }
    });
}
