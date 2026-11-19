//! `PRO.PRC.LR.CUMULATIVE` against the four running premium-weighted loss
//! ratios the AP pricing demo prints (#806), on its inputs
//! (`validation/data/ap_pricing_demo_inputs.json`, a verbatim copy of the
//! demo's `examples/data/inputs.json`), as the Rust, Python and R tests
//! check them (#923).

use crate::common;

use common::{num, ok};
use prospicio_xll::sheet::{Grid, Value};

/// The demo's printed rows (1-based) and values, to the 12 decimals printed.
const PRINTED: [(usize, f64); 4] = [
    (1, 0.610000000000),
    (2, 0.594705882353),
    (3, 0.638120662040),
    (14, 0.637034303676),
];

/// One array of the demo's inputs as a sheet column.
fn column(inputs: &serde_json::Value, name: &str) -> Grid {
    Grid::from_rows(
        inputs[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| vec![Value::Number(v.as_f64().unwrap())])
            .collect(),
    )
}

#[test]
fn ap_demo_cumulative_loss_ratios_match_the_printed_values() {
    let inputs: serde_json::Value =
        serde_json::from_str(&common::validation("data/ap_pricing_demo_inputs.json")).unwrap();
    let g = ok(
        "pro_prc_lr_cumulative",
        vec![
            column(&inputs, "loss_ratios"),
            column(&inputs, "earned_premium"),
        ],
    );
    assert_eq!((g.rows, g.columns), (14, 1));
    for (row, want) in PRINTED {
        let got = num(&g.cells[row - 1]);
        assert!(
            (got - want).abs() < 5e-13,
            "row {row}: got {got:.15}, the demo prints {want:.12}"
        );
    }
}
