//! Pricing parity: Pareto layer rating and implied alphas against the R
//! package Pareto (reference values only; see `docs/design/pareto.md`).

use act_pricing::layer::{
    XsLayer, alpha_between_frequencies, alpha_between_frequency_and_layer, alpha_between_layers,
    pareto_extrapolation,
};
use act_validation::{check, reference};

#[test]
fn pareto_rating_matches_r() {
    let cases = reference("pareto_rating_r.csv");
    check(&cases, |c| {
        let fields: std::collections::HashMap<&str, &str> = c
            .get("params")
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .collect();
        let num = |k: &str| -> Option<f64> {
            match *fields.get(k)? {
                "Inf" => Some(f64::INFINITY),
                v => v.parse().ok(),
            }
        };
        let truncation = num("truncation");
        let layer = |c: &str, a: &str| XsLayer::new(num(c)?, num(a)?).ok();
        match c.get("quantity") {
            "extrapolation" => pareto_extrapolation(
                layer("c1", "a1")?,
                layer("c2", "a2")?,
                num("alpha")?,
                truncation,
            )
            .ok(),
            "alpha_between_layers" => alpha_between_layers(
                (layer("c1", "a1")?, num("e1")?),
                (layer("c2", "a2")?, num("e2")?),
                truncation,
            )
            .ok(),
            "alpha_between_frequency_and_layer" => alpha_between_frequency_and_layer(
                num("threshold")?,
                num("frequency")?,
                layer("c", "a")?,
                num("e")?,
                truncation,
            )
            .ok(),
            "alpha_between_frequencies" => alpha_between_frequencies(
                num("t1")?,
                num("f1")?,
                num("t2")?,
                num("f2")?,
                truncation,
            )
            .ok(),
            _ => None,
        }
    });
}
