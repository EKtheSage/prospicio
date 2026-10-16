//! Models lane: neural networks (`prospicio-nn`) for the R `nn.R` API: the
//! CANN, the attention CANN and networks imported from ONNX. Designs
//! arrive column-major with their column names, as in `models.rs`.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_models::{Family, Fitted, Link, Model};
use prospicio_nn::onnx::{OnnxModel, OnnxPredictor, OnnxTensor, OutputScale};
use prospicio_nn::{AttentionCann, AttentionCannFit, Cann, CannFit, EarlyStopping};

use crate::distributions::PredictiveDistribution;
use crate::models::{design, family, link};
use crate::{to_r, whole};

/// The network's link: the one named, else log (identity for the
/// Gaussian), the only links the networks support.
fn network_link(name: &str, family: Family) -> Result<Link> {
    match name {
        "" if family == Family::Gaussian => Ok(Link::Identity),
        "" => Ok(Link::Log),
        _ => link(name, family, f64::NAN),
    }
}

/// Early stopping when `validation_share` is not NA.
fn early_stopping(validation_share: f64, patience: f64) -> Result<Option<EarlyStopping>> {
    if validation_share.is_nan() {
        return Ok(None);
    }
    Ok(Some(EarlyStopping {
        validation_share,
        patience: whole(patience, "patience")? as usize,
    }))
}

fn counts(values: &[f64], name: &str) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|&v| whole(v, name).map(|k| k as usize))
        .collect()
}

/// A fitted CANN.
#[extendr]
pub(crate) struct CannModel {
    inner: CannFit,
}

/// Fits a CANN to the design `x` (column-major, columns `names`); the
/// offset carries the GLM's linear predictor.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn cann_fit_design(
    x: &[f64],
    names: Vec<String>,
    y: &[f64],
    offset: &[f64],
    weights: &[f64],
    family_name: &str,
    link_name: &str,
    power: f64,
    hidden: &[f64],
    epochs: f64,
    batch_size: f64,
    learning_rate: f64,
    seed: f64,
    validation_share: f64,
    patience: f64,
) -> Result<CannModel> {
    let f = family(family_name, f64::NAN, power)?;
    let mut spec = Cann::new(f, network_link(link_name, f)?);
    spec.hidden = counts(hidden, "hidden")?;
    spec.epochs = whole(epochs, "epochs")? as usize;
    spec.batch_size = whole(batch_size, "batch_size")? as usize;
    spec.learning_rate = learning_rate;
    spec.seed = whole(seed, "seed")?;
    spec.early_stopping = early_stopping(validation_share, patience)?;
    let d = design(x, names, offset, weights)?;
    let inner = spec.fit(&d, y).map_err(to_r)?;
    Ok(CannModel { inner })
}

#[extendr]
impl CannModel {
    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted().to_vec()
    }

    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    /// NA without early stopping.
    fn best_epoch(&self) -> f64 {
        self.inner.best_epoch().map_or(f64::NAN, |e| e as f64)
    }

    fn validation_history(&self) -> Vec<f64> {
        self.inner.validation_history().to_vec()
    }

    fn predict(&self, x: &[f64], names: Vec<String>, offset: &[f64]) -> Result<Vec<f64>> {
        let d = design(x, names, offset, &[])?;
        self.inner.predict(&d).map_err(to_r)
    }

    fn correction(&self, x: &[f64], names: Vec<String>) -> Result<Vec<f64>> {
        let d = design(x, names, &[], &[])?;
        self.inner.correction(&d).map_err(to_r)
    }

    fn predict_distribution(
        &self,
        x: &[f64],
        names: Vec<String>,
        offset: &[f64],
        weights: &[f64],
        n_sims: f64,
        seed: f64,
    ) -> Result<PredictiveDistribution> {
        let d = design(x, names, offset, weights)?;
        let inner = self
            .inner
            .predict_distribution(&d, whole(n_sims, "n_sims")? as usize, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

/// A fitted attention CANN.
#[extendr]
pub(crate) struct AttentionCannModel {
    inner: AttentionCannFit,
}

/// Fits an attention CANN; a factor's indicator columns are named
/// `factor[level]` so they share one token.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn attention_cann_fit_design(
    x: &[f64],
    names: Vec<String>,
    y: &[f64],
    offset: &[f64],
    weights: &[f64],
    family_name: &str,
    link_name: &str,
    power: f64,
    shape: &[f64],
    epochs: f64,
    batch_size: f64,
    learning_rate: f64,
    seed: f64,
    validation_share: f64,
    patience: f64,
) -> Result<AttentionCannModel> {
    let f = family(family_name, f64::NAN, power)?;
    let [d_model, heads, layers, feed_forward] = shape else {
        return Err(Error::Other(
            "shape is d_model, heads, layers, feed_forward".into(),
        ));
    };
    let mut spec = AttentionCann::new(f, network_link(link_name, f)?);
    spec.d_model = whole(*d_model, "d_model")? as usize;
    spec.heads = whole(*heads, "heads")? as usize;
    spec.layers = whole(*layers, "layers")? as usize;
    spec.feed_forward = whole(*feed_forward, "feed_forward")? as usize;
    spec.epochs = whole(epochs, "epochs")? as usize;
    spec.batch_size = whole(batch_size, "batch_size")? as usize;
    spec.learning_rate = learning_rate;
    spec.seed = whole(seed, "seed")?;
    spec.early_stopping = early_stopping(validation_share, patience)?;
    let d = design(x, names, offset, weights)?;
    let inner = spec.fit(&d, y).map_err(to_r)?;
    Ok(AttentionCannModel { inner })
}

#[extendr]
impl AttentionCannModel {
    fn tokens(&self) -> Vec<String> {
        self.inner.tokens().to_vec()
    }

    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted().to_vec()
    }

    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    fn best_epoch(&self) -> f64 {
        self.inner.best_epoch().map_or(f64::NAN, |e| e as f64)
    }

    fn validation_history(&self) -> Vec<f64> {
        self.inner.validation_history().to_vec()
    }

    fn predict(&self, x: &[f64], names: Vec<String>, offset: &[f64]) -> Result<Vec<f64>> {
        let d = design(x, names, offset, &[])?;
        self.inner.predict(&d).map_err(to_r)
    }

    fn correction(&self, x: &[f64], names: Vec<String>) -> Result<Vec<f64>> {
        let d = design(x, names, &[], &[])?;
        self.inner.correction(&d).map_err(to_r)
    }

    /// `rows`, `heads`, `tokens` and `weights` row-major `[row][head][token]`.
    fn attention(&self, x: &[f64], names: Vec<String>) -> Result<List> {
        let d = design(x, names, &[], &[])?;
        let a = self.inner.attention(&d).map_err(to_r)?;
        Ok(list!(
            rows = a.rows as f64,
            heads = a.heads as f64,
            tokens = a.tokens,
            weights = a.weights
        ))
    }

    fn predict_distribution(
        &self,
        x: &[f64],
        names: Vec<String>,
        offset: &[f64],
        weights: &[f64],
        n_sims: f64,
        seed: f64,
    ) -> Result<PredictiveDistribution> {
        let d = design(x, names, offset, weights)?;
        let inner = self
            .inner
            .predict_distribution(&d, whole(n_sims, "n_sims")? as usize, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

/// A network loaded from an ONNX file.
#[extendr]
pub(crate) struct OnnxNet {
    inner: OnnxModel,
}

#[extendr]
fn onnx_load_rust(path: &str) -> Result<OnnxNet> {
    Ok(OnnxNet {
        inner: OnnxModel::load(path).map_err(to_r)?,
    })
}

#[extendr]
fn onnx_from_bytes_rust(bytes: Raw) -> Result<OnnxNet> {
    Ok(OnnxNet {
        inner: OnnxModel::from_bytes(bytes.as_slice()).map_err(to_r)?,
    })
}

#[extendr]
impl OnnxNet {
    /// `name`, `rank`, `integer`, and `shape` with NA for a free size and
    /// an empty vector when the file gives none.
    fn inputs(&self) -> List {
        let i = self.inner.inputs();
        let shapes: Vec<Robj> = i
            .iter()
            .map(|s| {
                s.shape
                    .as_ref()
                    .map(|v| {
                        v.iter()
                            .map(|d| d.map_or(f64::NAN, |k| k as f64))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
                    .into()
            })
            .collect();
        list!(
            name = i.iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
            rank = i.iter().map(|s| s.rank as f64).collect::<Vec<_>>(),
            integer = i.iter().map(|s| s.integer).collect::<Vec<_>>(),
            shape = List::from_values(shapes)
        )
    }

    fn outputs(&self) -> Vec<String> {
        self.inner.outputs().to_vec()
    }

    fn ops(&self) -> Vec<String> {
        self.inner.ops().to_vec()
    }

    /// `shapes` and `values` are lists, one entry per graph input, values
    /// row-major; returns a list of `list(shape, values)`.
    fn run(&self, shapes: List, values: List) -> Result<List> {
        if shapes.len() != values.len() {
            return Err(Error::Other("one shape per input".into()));
        }
        let mut tensors = Vec::with_capacity(shapes.len());
        for (s, v) in shapes.values().zip(values.values()) {
            let s: Vec<f64> = s
                .as_real_vector()
                .ok_or_else(|| Error::Other("a shape must be numeric".into()))?;
            let v: Vec<f64> = v
                .as_real_vector()
                .ok_or_else(|| Error::Other("values must be numeric".into()))?;
            tensors.push(OnnxTensor::new(counts(&s, "shape")?, v).map_err(to_r)?);
        }
        let out = self.inner.run(&tensors).map_err(to_r)?;
        Ok(List::from_values(out.into_iter().map(|t| {
            list!(
                shape = t.shape.iter().map(|&k| k as f64).collect::<Vec<_>>(),
                values = t.values
            )
        })))
    }
}

/// An `OnnxNet` bound to design columns.
#[extendr]
pub(crate) struct OnnxPredictorModel {
    inner: OnnxPredictor,
}

/// `inputs` names graph inputs and `columns` their design columns, one
/// character vector each; `scale` is "mean" or "link".
#[extendr]
#[allow(clippy::too_many_arguments)]
fn onnx_predictor_rust(
    net: &OnnxNet,
    family_name: &str,
    inputs: Vec<String>,
    columns: List,
    output: f64,
    scale: &str,
    link_name: &str,
    dispersion: f64,
    theta: f64,
    power: f64,
) -> Result<OnnxPredictorModel> {
    let f = family(family_name, theta, power)?;
    let scale = match (scale, link_name) {
        ("mean", "") => OutputScale::Mean,
        ("mean", _) => return Err(Error::Other("link is only for scale = \"link\"".into())),
        ("link", l) => OutputScale::Link(link(l, f, f64::NAN)?),
        (other, _) => {
            return Err(Error::Other(format!(
                "scale must be \"mean\" or \"link\", got {other}"
            )));
        }
    };
    let mut p = OnnxPredictor::new(net.inner.clone(), f)
        .output(whole(output, "output")? as usize)
        .scale(scale)
        .dispersion(dispersion);
    if inputs.len() != columns.len() {
        return Err(Error::Other("one set of columns per graph input".into()));
    }
    for (name, cols) in inputs.iter().zip(columns.values()) {
        let cols: Vec<String> = cols
            .as_string_vector()
            .ok_or_else(|| Error::Other("columns must be character vectors".into()))?;
        let cols: Vec<&str> = cols.iter().map(String::as_str).collect();
        p = p.input(name, &cols);
    }
    Ok(OnnxPredictorModel { inner: p })
}

#[extendr]
impl OnnxPredictorModel {
    fn family(&self) -> String {
        self.inner.family.name().into()
    }

    fn predict(&self, x: &[f64], names: Vec<String>, offset: &[f64]) -> Result<Vec<f64>> {
        let d = design(x, names, offset, &[])?;
        self.inner.predict(&d).map_err(to_r)
    }

    fn predict_distribution(
        &self,
        x: &[f64],
        names: Vec<String>,
        offset: &[f64],
        weights: &[f64],
        n_sims: f64,
        seed: f64,
    ) -> Result<PredictiveDistribution> {
        let d = design(x, names, offset, weights)?;
        let inner = self
            .inner
            .predict_distribution(&d, whole(n_sims, "n_sims")? as usize, whole(seed, "seed")?)
            .map_err(to_r)?;
        Ok(PredictiveDistribution { inner })
    }
}

extendr_module! {
    mod nn;
    fn cann_fit_design;
    impl CannModel;
    fn attention_cann_fit_design;
    impl AttentionCannModel;
    fn onnx_load_rust;
    fn onnx_from_bytes_rust;
    impl OnnxNet;
    fn onnx_predictor_rust;
    impl OnnxPredictorModel;
}
