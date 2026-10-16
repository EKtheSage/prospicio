//! ONNX import against onnxruntime (`validation/scripts/onnx_models.py`):
//! three small float32 networks of the kinds a PyTorch tabular model
//! exports, each row's output within float32 rounding of onnxruntime's.
#![cfg(feature = "onnx")]

use std::path::PathBuf;

use prospicio_models::{Design, Family, Fitted, Link};
use prospicio_nn::onnx::{OnnxModel, OnnxPredictor, OnnxTensor, OutputScale};
use prospicio_prob::Distribution;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../validation")
}

fn model(name: &str) -> OnnxModel {
    OnnxModel::load(root().join("data/onnx").join(name)).unwrap()
}

/// One reference row: `(model, inputs, codes, expected, rel_tol)`.
type Case = (String, Vec<f64>, Vec<f64>, f64, f64);

fn cases() -> Vec<Case> {
    let text = std::fs::read_to_string(root().join("reference/onnx_outputs.csv")).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let nums = |s: &str| -> Vec<f64> {
        s.split('|')
            .filter(|v| !v.is_empty())
            .map(|v| v.parse().unwrap())
            .collect()
    };
    lines
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (
                f[col("model")].to_string(),
                nums(f[col("inputs")]),
                nums(f[col("codes")]),
                f[col("expected")].parse().unwrap(),
                f[col("rel_tol")].parse().unwrap(),
            )
        })
        .collect()
}

#[test]
fn every_model_matches_onnxruntime() {
    let cases = cases();
    assert_eq!(cases.len(), 36);
    for name in ["mlp", "embedding", "gemm"] {
        let m = model(&format!("{name}.onnx"));
        let rows: Vec<_> = cases.iter().filter(|c| c.0 == name).collect();
        let n = rows.len();
        let width = rows[0].1.len();
        let x = OnnxTensor::new(
            vec![n, width],
            rows.iter().flat_map(|r| r.1.clone()).collect(),
        )
        .unwrap();
        let mut inputs = vec![x];
        if name == "embedding" {
            inputs.push(OnnxTensor::new(vec![n], rows.iter().map(|r| r.2[0]).collect()).unwrap());
        }
        let out = m.run(&inputs).unwrap();
        assert_eq!(out[0].values.len(), n, "{name}");
        for (r, got) in rows.iter().zip(&out[0].values) {
            let (want, rel) = (r.3, r.4);
            assert!(
                (got - want).abs() <= rel * want.abs().max(1e-1),
                "{name}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn the_graphs_are_read_as_written() {
    assert_eq!(model("mlp.onnx").ops(), ["Gemm", "Relu", "Gemm", "Exp"]);
    // MatMul then Add fuse to Linear.
    assert_eq!(
        model("embedding.onnx").ops(),
        [
            "Gather", "Concat", "Linear", "Tanh", "Linear", "Softplus", "Squeeze"
        ]
    );
    let e = model("embedding.onnx");
    assert_eq!(e.inputs().len(), 2);
    assert!(!e.inputs()[0].integer && e.inputs()[1].integer);
    assert_eq!(e.inputs()[0].shape, Some(vec![None, Some(3)]));
    // A wrong width is refused.
    let x = OnnxTensor::new(vec![1, 5], vec![0.0; 5]).unwrap();
    assert!(model("mlp.onnx").run(&[x]).is_err());
}

#[test]
fn a_predictor_binds_design_columns_and_draws() {
    let cases = cases();
    let rows: Vec<_> = cases.iter().filter(|c| c.0 == "embedding").collect();
    let n = rows.len();
    let names: Vec<String> = ["a", "b", "c", "region"].map(String::from).to_vec();
    let mut cols: Vec<Vec<f64>> = (0..3)
        .map(|j| rows.iter().map(|r| r.1[j]).collect())
        .collect();
    cols.push(rows.iter().map(|r| r.2[0]).collect());
    let design = Design::new(names.clone(), cols.clone()).unwrap();
    let net = OnnxPredictor::new(model("embedding.onnx"), Family::Gamma)
        .input("x_num", &["a", "b", "c"])
        .input("x_cat", &["region"])
        .dispersion(0.5);
    let mu = net.predict(&design).unwrap();
    for (r, m) in rows.iter().zip(&mu) {
        assert!((m - r.3).abs() <= 1e-5 * r.3.abs());
    }
    let pd = net.predict_distribution(&design, 20_000, 3).unwrap();
    assert_eq!(pd.n_components(), n);
    let total: f64 = mu.iter().sum();
    assert!((pd.total().mean() - total).abs() < 0.02 * total);

    // On the log scale with an offset: mu = exp(offset + output).
    let offset = vec![0.3; n];
    let d = Design::new(names.clone(), cols.clone())
        .unwrap()
        .with_offset(offset)
        .unwrap();
    let log_net = net.clone().scale(OutputScale::Link(Link::Log));
    let lm = log_net.predict(&d).unwrap();
    for (l, m) in lm.iter().zip(&mu) {
        assert!((l - (0.3 + m).exp()).abs() < 1e-12);
    }

    // Unbound inputs, missing columns and non-integer codes are refused.
    let unbound =
        OnnxPredictor::new(model("embedding.onnx"), Family::Gamma).input("x_num", &["a", "b", "c"]);
    assert!(unbound.predict(&design).is_err());
    let missing = net.clone().input("x_cat", &["zone"]);
    assert!(missing.predict(&design).is_err());
    let mut bad = cols;
    bad[3][0] = 0.5;
    assert!(net.predict(&Design::new(names, bad).unwrap()).is_err());
}
