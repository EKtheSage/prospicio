//! The running premium-weighted loss ratio against the four values the AP
//! pricing demo prints (#806), on its inputs
//! (`validation/data/ap_pricing_demo_inputs.json`, a verbatim copy of the
//! demo's `examples/data/inputs.json`).

use std::path::Path;

use prospicio_pricing::ratemaking::cumulative_loss_ratio;

/// The demo's printed rows (1-based) and values, to the 12 decimals printed.
const PRINTED: [(usize, f64); 4] = [
    (1, 0.610000000000),
    (2, 0.594705882353),
    (3, 0.638120662040),
    (14, 0.637034303676),
];

#[test]
fn ap_demo_cumulative_loss_ratios_match_the_printed_values() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/ap_pricing_demo_inputs.json");
    let inputs: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let column = |name: &str| -> Vec<f64> {
        inputs[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    let lr = cumulative_loss_ratio(&column("loss_ratios"), &column("earned_premium")).unwrap();
    assert_eq!(lr.len(), 14);
    for (row, want) in PRINTED {
        let got = lr[row - 1];
        assert!(
            (got - want).abs() < 5e-13,
            "row {row}: got {got:.15}, the demo prints {want:.12}"
        );
    }
}
