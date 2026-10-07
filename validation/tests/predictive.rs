//! PredictiveDistribution Arrow IPC files, against a fixture written by
//! pyarrow (`validation/scripts/predictive_ipc.py`) from the documented
//! format rather than by the Rust writer.

use std::fs::File;
use std::path::Path;

use prospicio_core::{Grain, Month, Period};
use prospicio_prob::{KeyValue, PredictiveDistribution};

#[test]
fn reads_pyarrow_fixture() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("reference")
        .join("predictive_distribution_v1.arrow");
    let pd = PredictiveDistribution::read_ipc(File::open(path).unwrap()).unwrap();

    assert_eq!(pd.dims(), ["lob", "origin"]);
    let h2 = Period::containing(Month::new(2020, 7).unwrap(), Grain::Semester);
    assert_eq!(
        pd.components(),
        [
            vec![KeyValue::from("Auto"), Period::year(2019).into()],
            vec!["Auto".into(), h2.into()],
            vec!["Home".into(), 7.into()],
        ]
    );
    // Columns [1.5, -2, 1e300], [0, 4.25, 5], [7, 8, 9], read back row by row.
    assert_eq!(
        pd.draw_matrix(),
        [1.5, 0.0, 7.0, -2.0, 4.25, 8.0, 1e300, 5.0, 9.0]
    );

    let p = pd.provenance();
    assert_eq!(p.model, "fixture");
    assert_eq!(p.parameters, [("n_sims".to_string(), "3".to_string())]);
    assert_eq!(p.seed, Some(u64::MAX));
    assert_eq!(p.stream_scheme.as_deref(), Some("chacha20/sim-index/v1"));
    assert_eq!(
        p.versions,
        // The fixture was written before the crates were renamed from act-*.
        [("act-prob".to_string(), "0.0.1".to_string())]
    );
    assert_eq!(p.input_hash, None);
}
