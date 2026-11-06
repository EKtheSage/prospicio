//! Neural networks (`prospicio-nn`): the CANN, the attention CANN and
//! networks imported from ONNX, behind the same `Design` and predictive
//! distributions as the GLM.

use prospicio_models::{Fitted, Model};
use prospicio_nn::onnx::{OnnxModel, OnnxPredictor, OnnxTensor, OutputScale};
use prospicio_nn::{
    AttentionCann, AttentionCannFit, Cann, CannFit, EarlyStopping, Head, TabularNet, TabularNetFit,
    Target, TokenAttention,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::distributions::PyPredictiveDistribution;
use crate::models::{PyDesign, family, link, link_name};
use crate::to_py;

/// The network's link: the one asked for, else log (identity for the
/// Gaussian), the only links the networks support.
fn network_link(
    name: Option<&str>,
    family: prospicio_models::Family,
) -> PyResult<prospicio_models::Link> {
    match name {
        Some(_) => link(name, family, None),
        None if family == prospicio_models::Family::Gaussian => {
            Ok(prospicio_models::Link::Identity)
        }
        None => Ok(prospicio_models::Link::Log),
    }
}

fn early_stopping(validation_share: Option<f64>, patience: usize) -> Option<EarlyStopping> {
    validation_share.map(|validation_share| EarlyStopping {
        validation_share,
        patience,
    })
}

/// A Combined Actuarial Neural Network (Wüthrich and Merz): the design's
/// offset, a GLM's linear predictor, plus a feed-forward network of the
/// design's columns, ``eta = offset + NN(x)``.
///
/// The output layer starts at zero, so training starts exactly at the GLM
/// and learns only what it misses. Training minimizes the family's
/// weighted deviance by Adam on mini-batches, on the CPU in ``float64``,
/// and is reproducible from ``seed``.
///
/// Parameters
/// ----------
/// family : {"poisson", "gamma", "tweedie", "gaussian"}
/// link : str, optional
///     ``"log"`` for the Poisson, gamma and Tweedie, ``"identity"`` for the
///     Gaussian: the only links supported, and the defaults.
/// hidden : list of int, default [16, 8]
///     Widths of the hidden ReLU layers.
/// epochs : int, default 200
/// batch_size : int, default 64
/// learning_rate : float, default 1e-3
/// seed : int, default 0
///     Seeds the initial weights and the mini-batch order.
/// validation_share : float, optional
///     Turns on early stopping: hold out this share of the rows, score the
///     mean deviance on them after each epoch, and keep the best epoch.
/// patience : int, default 10
///     With early stopping, epochs without improvement before stopping.
/// power : float, optional
///     The Tweedie power.
///
/// Examples
/// --------
/// >>> from prospicio.models import Cann, Design
/// >>> x = [i / 50 - 1 for i in range(100)]
/// >>> d = Design([x], ["x"], offset=[0.0] * 100)
/// >>> fit = Cann("poisson", hidden=[8], epochs=5, seed=1).fit(d, [1.0] * 100)
/// >>> len(fit.predict(d))
/// 100
#[pyclass(name = "Cann", module = "prospicio.models", frozen)]
pub(crate) struct PyCann {
    inner: Cann,
}

#[pymethods]
impl PyCann {
    #[new]
    #[pyo3(signature = (family, link = None, hidden = None, epochs = 200, batch_size = 64, learning_rate = 1e-3, seed = 0, validation_share = None, patience = 10, power = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        family: &str,
        link: Option<&str>,
        hidden: Option<Vec<usize>>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        seed: u64,
        validation_share: Option<f64>,
        patience: usize,
        power: Option<f64>,
    ) -> PyResult<Self> {
        let f = self::family(family, None, power)?;
        let l = network_link(link, f)?;
        let mut inner = Cann::new(f, l);
        if let Some(h) = hidden {
            inner.hidden = h;
        }
        inner.epochs = epochs;
        inner.batch_size = batch_size;
        inner.learning_rate = learning_rate;
        inner.seed = seed;
        inner.early_stopping = early_stopping(validation_share, patience);
        Ok(Self { inner })
    }

    /// Fits the network.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    ///     Columns are standardized internally; the offset carries the
    ///     GLM's linear predictor (zeros for a network alone).
    /// y : list of float
    ///
    /// Returns
    /// -------
    /// CannFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the family or link is not supported, a setting is out of
    ///     range, or a response is out of the family's range.
    fn fit(&self, py: Python<'_>, design: PyRef<'_, PyDesign>, y: Vec<f64>) -> PyResult<PyCannFit> {
        let (spec, d) = (&self.inner, &design.inner);
        let inner = py.detach(|| spec.fit(d, &y)).map_err(to_py)?;
        Ok(PyCannFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "Cann(family={:?}, link={:?}, hidden={:?}, epochs={})",
            self.inner.family.name(),
            link_name(self.inner.link),
            self.inner.hidden,
            self.inner.epochs
        )
    }
}

/// A fitted CANN, from ``Cann.fit``.
#[pyclass(name = "CannFit", module = "prospicio.models", frozen)]
pub(crate) struct PyCannFit {
    inner: CannFit,
}

#[pymethods]
impl PyCannFit {
    /// The fit as a JSON artifact (spec, scaling, every weight by its path,
    /// fitted values, provenance); ``from_json`` reads it back exactly and
    /// pickling uses it too.
    ///
    /// Returns
    /// -------
    /// str
    ///
    /// Examples
    /// --------
    /// >>> import pickle
    /// >>> from prospicio.models import Cann, Design
    /// >>> d = Design([[0.0, 1.0, 2.0, 3.0]], ["x"])
    /// >>> fit = Cann("poisson", hidden=[4], epochs=5).fit(d, [1.0, 2.0, 2.0, 5.0])
    /// >>> pickle.loads(pickle.dumps(fit)).predict(d) == fit.predict(d)
    /// True
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// Reads an artifact written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// CannFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For malformed JSON, another format, a newer format version or
    ///     weights that do not fit the spec.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CannFit::from_json(text).map_err(to_py)?,
        })
    }

    /// Hash of the training data (design, offset, weights, response).
    #[getter]
    fn input_hash(&self) -> &str {
        self.inner.input_hash()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let from_json = slf.get_type().getattr("from_json")?;
        Ok((from_json, (slf.borrow().inner.to_json(),)))
    }

    /// Expected response for each row of ``design`` (same columns as the
    /// training design), ``g^-1(offset + NN(x))``.
    fn predict(&self, py: Python<'_>, design: PyRef<'_, PyDesign>) -> PyResult<Vec<f64>> {
        let (fit, d) = (&self.inner, &design.inner);
        py.detach(|| fit.predict(d)).map_err(to_py)
    }

    /// The network's correction to the linear predictor, ``NN(x)``, per row.
    fn correction(&self, py: Python<'_>, design: PyRef<'_, PyDesign>) -> PyResult<Vec<f64>> {
        let (fit, d) = (&self.inner, &design.inner);
        py.detach(|| fit.correction(d)).map_err(to_py)
    }

    /// Joint predictive distribution across the rows, keyed ``row = 0, 1,
    /// ...``: process draws from the family around the predicted means (a
    /// trained network carries no parameter uncertainty).
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// n_sims : int
    /// seed : int
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    fn predict_distribution(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        n_sims: usize,
        seed: u64,
    ) -> PyResult<PyPredictiveDistribution> {
        let (fit, d) = (&self.inner, &design.inner);
        let inner = py
            .detach(|| fit.predict_distribution(d, n_sims, seed))
            .map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    /// Fitted means on the training rows.
    #[getter]
    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted().to_vec()
    }

    /// With early stopping, the epoch (from 1) whose network was kept.
    #[getter]
    fn best_epoch(&self) -> Option<usize> {
        self.inner.best_epoch()
    }

    /// With early stopping, the validation mean deviance after each epoch.
    #[getter]
    fn validation_history(&self) -> Vec<f64> {
        self.inner.validation_history().to_vec()
    }

    /// Dispersion: 1 for the Poisson, otherwise Pearson's estimate on the
    /// training data.
    #[getter]
    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    fn __repr__(&self) -> String {
        format!(
            "CannFit(n={}, dispersion={})",
            self.inner.fitted().len(),
            self.inner.dispersion()
        )
    }
}

/// A CANN whose correction is a transformer over feature tokens (an
/// FT-Transformer in the CANN setting).
///
/// Each numeric column is a token, and each factor's indicator columns
/// (``region[B]``, ``region[C]``, ...) together form one token. A learned
/// ``[CLS]`` token joins them; pre-norm blocks of multi-head
/// self-attention and a feed-forward layer mix them; the ``[CLS]`` state,
/// through a zero-initialized head, corrects the linear predictor, so
/// training starts exactly at the GLM.
///
/// Parameters
/// ----------
/// family : {"poisson", "gamma", "tweedie", "gaussian"}
/// link : str, optional
///     As for ``Cann``: log, or identity for the Gaussian.
/// d_model : int, default 16
///     Token width; a multiple of ``heads``.
/// heads : int, default 2
/// layers : int, default 2
///     Transformer blocks.
/// feed_forward : int, default 32
///     Width of each block's feed-forward layer.
/// epochs : int, default 100
/// batch_size : int, default 64
/// learning_rate : float, default 1e-3
/// seed : int, default 0
/// validation_share : float, optional
///     Turns on early stopping, as for ``Cann``.
/// patience : int, default 10
/// power : float, optional
///     The Tweedie power.
#[pyclass(name = "AttentionCann", module = "prospicio.models", frozen)]
pub(crate) struct PyAttentionCann {
    inner: AttentionCann,
}

#[pymethods]
impl PyAttentionCann {
    #[new]
    #[pyo3(signature = (family, link = None, d_model = 16, heads = 2, layers = 2, feed_forward = 32, epochs = 100, batch_size = 64, learning_rate = 1e-3, seed = 0, validation_share = None, patience = 10, power = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        family: &str,
        link: Option<&str>,
        d_model: usize,
        heads: usize,
        layers: usize,
        feed_forward: usize,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        seed: u64,
        validation_share: Option<f64>,
        patience: usize,
        power: Option<f64>,
    ) -> PyResult<Self> {
        let f = self::family(family, None, power)?;
        let l = network_link(link, f)?;
        let mut inner = AttentionCann::new(f, l);
        inner.d_model = d_model;
        inner.heads = heads;
        inner.layers = layers;
        inner.feed_forward = feed_forward;
        inner.epochs = epochs;
        inner.batch_size = batch_size;
        inner.learning_rate = learning_rate;
        inner.seed = seed;
        inner.early_stopping = early_stopping(validation_share, patience);
        Ok(Self { inner })
    }

    /// Fits the network.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// y : list of float
    ///
    /// Returns
    /// -------
    /// AttentionCannFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the family or link is not supported, ``d_model`` is not a
    ///     multiple of ``heads``, or a response is out of range.
    fn fit(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
    ) -> PyResult<PyAttentionCannFit> {
        let (spec, d) = (&self.inner, &design.inner);
        let inner = py.detach(|| spec.fit(d, &y)).map_err(to_py)?;
        Ok(PyAttentionCannFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "AttentionCann(family={:?}, d_model={}, heads={}, layers={})",
            self.inner.family.name(),
            self.inner.d_model,
            self.inner.heads,
            self.inner.layers
        )
    }
}

/// A fitted attention CANN, from ``AttentionCann.fit``.
#[pyclass(name = "AttentionCannFit", module = "prospicio.models", frozen)]
pub(crate) struct PyAttentionCannFit {
    inner: AttentionCannFit,
}

#[pymethods]
impl PyAttentionCannFit {
    /// The fit as a JSON artifact (spec, scaling, every weight by its path,
    /// fitted values, provenance); ``from_json`` reads it back exactly and
    /// pickling uses it too.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// Reads an artifact written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// AttentionCannFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For malformed JSON, another format, a newer format version or
    ///     weights that do not fit the spec.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: AttentionCannFit::from_json(text).map_err(to_py)?,
        })
    }

    /// Hash of the training data (design, offset, weights, response).
    #[getter]
    fn input_hash(&self) -> &str {
        self.inner.input_hash()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let from_json = slf.get_type().getattr("from_json")?;
        Ok((from_json, (slf.borrow().inner.to_json(),)))
    }

    /// Feature tokens, in column order.
    #[getter]
    fn tokens(&self) -> Vec<String> {
        self.inner.tokens().to_vec()
    }

    /// Expected response for each row of ``design``.
    fn predict(&self, py: Python<'_>, design: PyRef<'_, PyDesign>) -> PyResult<Vec<f64>> {
        let (fit, d) = (&self.inner, &design.inner);
        py.detach(|| fit.predict(d)).map_err(to_py)
    }

    /// The network's correction to the linear predictor, per row.
    fn correction(&self, py: Python<'_>, design: PyRef<'_, PyDesign>) -> PyResult<Vec<f64>> {
        let (fit, d) = (&self.inner, &design.inner);
        py.detach(|| fit.correction(d)).map_err(to_py)
    }

    /// The last block's attention from ``[CLS]`` to each feature token, per
    /// row of ``design`` and head: where the model looked for each risk. A
    /// diagnostic, not a decomposition of the prediction.
    ///
    /// Returns
    /// -------
    /// TokenAttention
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the model has no blocks (``layers=0``).
    fn attention(&self, py: Python<'_>, design: PyRef<'_, PyDesign>) -> PyResult<PyTokenAttention> {
        let (fit, d) = (&self.inner, &design.inner);
        let inner = py.detach(|| fit.attention(d)).map_err(to_py)?;
        Ok(PyTokenAttention { inner })
    }

    /// Joint predictive distribution across the rows: process draws from
    /// the family around the predicted means.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// n_sims : int
    /// seed : int
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    fn predict_distribution(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        n_sims: usize,
        seed: u64,
    ) -> PyResult<PyPredictiveDistribution> {
        let (fit, d) = (&self.inner, &design.inner);
        let inner = py
            .detach(|| fit.predict_distribution(d, n_sims, seed))
            .map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    /// Fitted means on the training rows.
    #[getter]
    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted().to_vec()
    }

    /// With early stopping, the epoch (from 1) whose network was kept.
    #[getter]
    fn best_epoch(&self) -> Option<usize> {
        self.inner.best_epoch()
    }

    /// With early stopping, the validation mean deviance after each epoch.
    #[getter]
    fn validation_history(&self) -> Vec<f64> {
        self.inner.validation_history().to_vec()
    }

    /// Dispersion: 1 for the Poisson, otherwise Pearson's estimate.
    #[getter]
    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    fn __repr__(&self) -> String {
        format!("AttentionCannFit(tokens={:?})", self.inner.tokens())
    }
}

/// Attention from ``[CLS]`` to each feature token in the last block, per
/// row and head, from ``AttentionCannFit.attention``.
#[pyclass(name = "TokenAttention", module = "prospicio.models", frozen)]
pub(crate) struct PyTokenAttention {
    inner: TokenAttention,
}

#[pymethods]
impl PyTokenAttention {
    /// Feature tokens, in column order.
    #[getter]
    fn tokens(&self) -> Vec<String> {
        self.inner.tokens.clone()
    }

    /// Number of rows.
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    /// Number of heads.
    #[getter]
    fn heads(&self) -> usize {
        self.inner.heads
    }

    /// Weight of token ``token`` (an index into ``tokens``) for ``row``
    /// and ``head``. Per row and head the weights over the feature tokens
    /// and ``[CLS]`` itself sum to 1.
    fn get(&self, row: usize, head: usize, token: usize) -> PyResult<f64> {
        let a = &self.inner;
        if row >= a.rows || head >= a.heads || token >= a.tokens.len() {
            return Err(PyValueError::new_err(format!(
                "row, head or token out of range (rows {}, heads {}, tokens {})",
                a.rows,
                a.heads,
                a.tokens.len()
            )));
        }
        Ok(a.get(row, head, token))
    }

    /// Mean weight per token over the rows and heads, or over ``rows``
    /// only: what the model attends to in a segment.
    ///
    /// Parameters
    /// ----------
    /// rows : list of int, optional
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (rows = None))]
    fn mean(&self, rows: Option<Vec<usize>>) -> PyResult<Vec<f64>> {
        if let Some(r) = &rows
            && let Some(bad) = r.iter().find(|&&i| i >= self.inner.rows)
        {
            return Err(PyValueError::new_err(format!(
                "row {bad} is out of range ({} rows)",
                self.inner.rows
            )));
        }
        Ok(self.inner.mean(rows.as_deref()))
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenAttention(rows={}, heads={}, tokens={:?})",
            self.inner.rows, self.inner.heads, self.inner.tokens
        )
    }
}

/// A network trained elsewhere (PyTorch, Keras), loaded from an ONNX file
/// and run in ``float64``.
///
/// Supports the operators a tabular network exports: ``Gemm``, ``Linear``,
/// ``MatMul``, broadcasting arithmetic, the common activations,
/// ``Softmax``, ``Concat``, ``Gather`` embeddings, reshapes and reductions.
/// Anything else is refused at load, with the operators the file uses.
///
/// Parameters
/// ----------
/// path : str
///
/// Raises
/// ------
/// ValueError
///     If the file cannot be read or parsed, or uses an unsupported
///     operator.
#[pyclass(name = "OnnxModel", module = "prospicio.models", frozen)]
pub(crate) struct PyOnnxModel {
    inner: OnnxModel,
}

#[pymethods]
impl PyOnnxModel {
    #[new]
    fn new(py: Python<'_>, path: String) -> PyResult<Self> {
        let inner = py.detach(|| OnnxModel::load(&path)).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Loads a model from the bytes of an ONNX file.
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///
    /// Returns
    /// -------
    /// OnnxModel
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: Vec<u8>) -> PyResult<Self> {
        let inner = py.detach(|| OnnxModel::from_bytes(&data)).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The graph's inputs as dicts with ``name``, ``rank``, ``shape`` (sizes
    /// the file fixes, ``None`` where it leaves one free) and ``integer``
    /// (category codes for an embedding).
    #[getter]
    fn inputs<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
        self.inner
            .inputs()
            .iter()
            .map(|i| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("name", &i.name)?;
                d.set_item("rank", i.rank)?;
                d.set_item("shape", i.shape.clone())?;
                d.set_item("integer", i.integer)?;
                Ok(d)
            })
            .collect()
    }

    /// The graph's output names.
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.inner.outputs().to_vec()
    }

    /// The operators the graph uses.
    #[getter]
    fn ops(&self) -> Vec<String> {
        self.inner.ops().to_vec()
    }

    /// Runs the graph.
    ///
    /// Parameters
    /// ----------
    /// inputs : list of (list of int, list of float)
    ///     One ``(shape, values)`` per graph input, in input order, values
    ///     row-major.
    ///
    /// Returns
    /// -------
    /// list of (list of int, list of float)
    ///     One ``(shape, values)`` per output.
    fn run(
        &self,
        py: Python<'_>,
        inputs: Vec<(Vec<usize>, Vec<f64>)>,
    ) -> PyResult<Vec<(Vec<usize>, Vec<f64>)>> {
        let tensors = inputs
            .into_iter()
            .map(|(shape, values)| OnnxTensor::new(shape, values))
            .collect::<prospicio_core::Result<Vec<_>>>()
            .map_err(to_py)?;
        let model = &self.inner;
        let out = py.detach(|| model.run(&tensors)).map_err(to_py)?;
        Ok(out.into_iter().map(|t| (t.shape, t.values)).collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "OnnxModel(inputs={:?}, outputs={:?})",
            self.inner
                .inputs()
                .iter()
                .map(|i| i.name.as_str())
                .collect::<Vec<_>>(),
            self.inner.outputs()
        )
    }
}

/// An ``OnnxModel`` behind the model interface: each graph input bound to
/// design columns, one output read as each row's mean, and a family for
/// predictive draws.
///
/// Parameters
/// ----------
/// model : OnnxModel
/// family : str
///     The family of the predictive draws (``poisson``, ``gamma``, ...).
/// inputs : dict of str to list of str
///     Each graph input's design columns: a rank-2 input takes its columns
///     in order (``[rows, columns]``), a rank-1 input one column. An
///     integer input (an embedding's category codes) must hold whole
///     numbers.
/// output : int, default 0
///     Which output is the mean.
/// scale : {"mean", "link"}, default "mean"
///     ``"mean"``: the output is the mean. ``"link"``: the output is on the
///     link's scale and added to the design's offset,
///     ``mu = g^-1(offset + output)`` (a CANN's correction, or a network
///     trained with an exposure offset).
/// link : str, optional
///     The link for ``scale="link"``; the family's canonical link by
///     default.
/// dispersion : float, default 1
///     The family's dispersion for predictive draws.
/// theta, power : float, optional
///     The negative binomial's ``theta`` or the Tweedie power.
#[pyclass(name = "OnnxPredictor", module = "prospicio.models", frozen)]
pub(crate) struct PyOnnxPredictor {
    inner: OnnxPredictor,
}

#[pymethods]
impl PyOnnxPredictor {
    #[new]
    #[pyo3(signature = (model, family, inputs, output = 0, scale = "mean", link = None, dispersion = 1.0, theta = None, power = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: PyRef<'_, PyOnnxModel>,
        family: &str,
        inputs: std::collections::BTreeMap<String, Vec<String>>,
        output: usize,
        scale: &str,
        link: Option<&str>,
        dispersion: f64,
        theta: Option<f64>,
        power: Option<f64>,
    ) -> PyResult<Self> {
        let f = self::family(family, theta, power)?;
        let scale = match scale {
            "mean" if link.is_none() => OutputScale::Mean,
            "mean" => {
                return Err(PyValueError::new_err("link is only for scale=\"link\""));
            }
            "link" => OutputScale::Link(self::link(link, f, None)?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "scale must be \"mean\" or \"link\", got {other:?}"
                )));
            }
        };
        let mut inner = OnnxPredictor::new(model.inner.clone(), f)
            .output(output)
            .scale(scale)
            .dispersion(dispersion);
        for (name, columns) in &inputs {
            let cols: Vec<&str> = columns.iter().map(String::as_str).collect();
            inner = inner.input(name, &cols);
        }
        Ok(Self { inner })
    }

    /// Expected response for each row of ``design``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a graph input is not bound, a bound column is missing, or an
    ///     integer input holds a fraction.
    fn predict(&self, py: Python<'_>, design: PyRef<'_, PyDesign>) -> PyResult<Vec<f64>> {
        let (p, d) = (&self.inner, &design.inner);
        py.detach(|| p.predict(d)).map_err(to_py)
    }

    /// Joint predictive distribution across the rows: process draws from
    /// the family and dispersion around the predicted means.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// n_sims : int
    /// seed : int
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    fn predict_distribution(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        n_sims: usize,
        seed: u64,
    ) -> PyResult<PyPredictiveDistribution> {
        let (p, d) = (&self.inner, &design.inner);
        let inner = py
            .detach(|| p.predict_distribution(d, n_sims, seed))
            .map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "OnnxPredictor(family={:?}, inputs={:?})",
            self.inner.family.name(),
            self.inner.bindings
        )
    }
}

/// One output of a ``TabularNet``: a family, its link and its weight in the
/// training loss.
///
/// Parameters
/// ----------
/// name : str
/// family : {"poisson", "gamma", "tweedie", "gaussian"}
/// link : str, optional
///     Log for the Poisson, gamma and Tweedie, identity for the Gaussian:
///     the only links supported, and the defaults.
/// loss_weight : float, default 1.0
///     The head's weight in the loss, the sum of each head's weighted mean
///     deviance times its loss weight.
/// power : float, optional
///     The Tweedie power.
#[pyclass(name = "Head", module = "prospicio.models", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyHead {
    inner: Head,
}

#[pymethods]
impl PyHead {
    #[new]
    #[pyo3(signature = (name, family, link = None, loss_weight = 1.0, power = None))]
    fn new(
        name: &str,
        family: &str,
        link: Option<&str>,
        loss_weight: f64,
        power: Option<f64>,
    ) -> PyResult<Self> {
        let f = self::family(family, None, power)?;
        let l = network_link(link, f)?;
        Ok(Self {
            inner: Head::new(name, f).link(l).loss_weight(loss_weight),
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn family(&self) -> &'static str {
        self.inner.family.name()
    }

    #[getter]
    fn loss_weight(&self) -> f64 {
        self.inner.loss_weight
    }

    fn __repr__(&self) -> String {
        format!(
            "Head({:?}, {:?}, link={:?}, loss_weight={})",
            self.inner.name,
            self.inner.family.name(),
            link_name(self.inner.link),
            self.inner.loss_weight
        )
    }
}

/// A tabular network: rating factors entered as learned embeddings, a
/// shared body of hidden ReLU layers, and one or more heads, each a linear
/// output that starts at zero on its own offset (a GLM's linear predictor
/// or the log exposure), ``eta_h = offset_h + out_h(body(x, e(f1), ...))``.
///
/// A factor named in ``embeddings`` is read from its indicator columns in
/// the design (``region[B]``, ...; a row with none set is the reference
/// level) and enters as a vector of the given dimensions in place of them.
/// Training minimizes the sum over heads of each head's weighted mean
/// deviance times its loss weight, by Adam on mini-batches, on the CPU in
/// ``float64``, reproducible from ``seed``. One head and no embeddings is
/// ``Cann``, bit for bit.
///
/// Parameters
/// ----------
/// heads : list of Head
/// embeddings : dict of str to int, optional
///     Each embedded factor and its dimensions, in order.
/// references : dict of str to str, optional
///     The reference level's label of an embedded factor, for
///     ``embeddings()``; ``"(reference)"`` otherwise.
/// hidden : list of int, default [16, 8]
/// epochs : int, default 200
/// batch_size : int, default 64
/// learning_rate : float, default 1e-3
/// seed : int, default 0
/// validation_share : float, optional
///     Turns on early stopping on the summed loss, as for ``Cann``.
/// patience : int, default 10
///
/// Examples
/// --------
/// >>> from prospicio.models import Design, Head, TabularNet
/// >>> d = Design([[0.0, 1.0, 2.0, 3.0] * 5, [0.0, 1.0] * 10], ["x", "g[b]"])
/// >>> counts = [0.0, 1.0, 0.0, 2.0] * 5
/// >>> sev = [0.0, 3.0, 0.0, 2.5] * 5
/// >>> spec = TabularNet([Head("freq", "poisson"), Head("sev", "gamma")],
/// ...                   embeddings={"g": 2}, hidden=[4], epochs=3)
/// >>> fit = spec.fit_heads(d, [counts, sev], weights=[None, counts])
/// >>> fit.head_names
/// ['freq', 'sev']
/// >>> len(fit.embeddings("g")["levels"])
/// 2
#[pyclass(name = "TabularNet", module = "prospicio.models", frozen)]
pub(crate) struct PyTabularNet {
    inner: TabularNet,
}

#[pymethods]
impl PyTabularNet {
    #[new]
    #[pyo3(signature = (heads, embeddings = None, references = None, hidden = None, epochs = 200, batch_size = 64, learning_rate = 1e-3, seed = 0, validation_share = None, patience = 10))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        heads: Vec<PyHead>,
        embeddings: Option<&Bound<'_, pyo3::types::PyDict>>,
        references: Option<std::collections::HashMap<String, String>>,
        hidden: Option<Vec<usize>>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        seed: u64,
        validation_share: Option<f64>,
        patience: usize,
    ) -> PyResult<Self> {
        let mut inner = TabularNet::new();
        if let Some(e) = embeddings {
            for (factor, dim) in e.iter() {
                inner = inner.embedding(&factor.extract::<String>()?, dim.extract::<usize>()?);
            }
        }
        let references = references.unwrap_or_default();
        for (factor, level) in &references {
            if !inner.embeddings.iter().any(|e| &e.factor == factor) {
                return Err(PyValueError::new_err(format!(
                    "references names {factor:?}, which is not an embedded factor"
                )));
            }
            inner = inner.reference(factor, level);
        }
        for h in heads {
            inner = inner.head(h.inner);
        }
        if let Some(h) = hidden {
            inner.hidden = h;
        }
        inner.epochs = epochs;
        inner.batch_size = batch_size;
        inner.learning_rate = learning_rate;
        inner.seed = seed;
        inner.early_stopping = early_stopping(validation_share, patience);
        Ok(Self { inner })
    }

    /// Fits a one-head network on ``y`` with the design's offset and
    /// weights.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// y : list of float
    ///
    /// Returns
    /// -------
    /// TabularNetFit
    fn fit(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
    ) -> PyResult<PyTabularNetFit> {
        let (spec, d) = (&self.inner, &design.inner);
        let inner = py.detach(|| spec.fit(d, &y)).map_err(to_py)?;
        Ok(PyTabularNetFit { inner })
    }

    /// Fits every head at once, one response per head in the heads' order.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// ys : list of list of float
    /// weights : list of (list of float or None), optional
    ///     Each head's prior weights (claim counts for a severity head); the
    ///     design's where ``None``. A row with weight 0 does not enter the
    ///     head's loss, and its response is not read.
    /// offsets : list of (list of float or None), optional
    ///     Each head's offset (a GLM's linear predictor, the log exposure);
    ///     the design's where ``None``.
    ///
    /// Returns
    /// -------
    /// TabularNetFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the number of responses is not the number of heads, an
    ///     embedded factor has no indicator columns, a setting is out of
    ///     range, or a response with a positive weight is out of its
    ///     family's range.
    #[pyo3(signature = (design, ys, weights = None, offsets = None))]
    fn fit_heads(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        ys: Vec<Vec<f64>>,
        weights: Option<Vec<Option<Vec<f64>>>>,
        offsets: Option<Vec<Option<Vec<f64>>>>,
    ) -> PyResult<PyTabularNetFit> {
        let k = ys.len();
        let weights = weights.unwrap_or_else(|| vec![None; k]);
        let offsets = offsets.unwrap_or_else(|| vec![None; k]);
        if weights.len() != k || offsets.len() != k {
            return Err(PyValueError::new_err(format!(
                "{k} responses, {} weights and {} offsets: give one of each per head",
                weights.len(),
                offsets.len()
            )));
        }
        let (spec, d) = (&self.inner, &design.inner);
        let inner = py
            .detach(|| {
                let targets: Vec<Target<'_>> = ys
                    .iter()
                    .zip(&weights)
                    .zip(&offsets)
                    .map(|((y, w), o)| Target {
                        y,
                        weights: w.as_deref(),
                        offset: o.as_deref(),
                    })
                    .collect();
                spec.fit_heads(d, &targets)
            })
            .map_err(to_py)?;
        Ok(PyTabularNetFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "TabularNet(heads={:?}, embeddings={:?}, hidden={:?}, epochs={})",
            self.inner
                .heads
                .iter()
                .map(|h| h.name.as_str())
                .collect::<Vec<_>>(),
            self.inner
                .embeddings
                .iter()
                .map(|e| (e.factor.as_str(), e.dim))
                .collect::<Vec<_>>(),
            self.inner.hidden,
            self.inner.epochs
        )
    }
}

/// A fitted tabular network, from ``TabularNet.fit`` or ``fit_heads``.
#[pyclass(name = "TabularNetFit", module = "prospicio.models", frozen)]
pub(crate) struct PyTabularNetFit {
    inner: TabularNetFit,
}

impl PyTabularNetFit {
    fn head(&self, head: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(name) = head.extract::<String>() {
            return self.inner.head_index(&name).map_err(to_py);
        }
        let k: usize = head.extract()?;
        if k >= self.inner.spec().heads.len() {
            return Err(PyValueError::new_err(format!(
                "head {k}: the network has {} heads",
                self.inner.spec().heads.len()
            )));
        }
        Ok(k)
    }
}

#[pymethods]
impl PyTabularNetFit {
    /// The fit as a JSON artifact; ``from_json`` reads it back exactly and
    /// pickling uses it too.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// Reads an artifact written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// TabularNetFit
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: TabularNetFit::from_json(text).map_err(to_py)?,
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let from_json = slf.get_type().getattr("from_json")?;
        Ok((from_json, (slf.borrow().inner.to_json(),)))
    }

    /// Hash of the training data (design, and each head's response,
    /// weights and offset).
    #[getter]
    fn input_hash(&self) -> &str {
        self.inner.input_hash()
    }

    /// The heads' names, in order.
    #[getter]
    fn head_names(&self) -> Vec<String> {
        self.inner
            .head_names()
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// Expected response of one head for each row of ``design``.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    ///     The training design's columns.
    /// head : int or str, default 0
    /// offset : list of float, optional
    ///     The head's offset; the design's if not given.
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (design, head = None, offset = None))]
    fn predict(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        head: Option<&Bound<'_, PyAny>>,
        offset: Option<Vec<f64>>,
    ) -> PyResult<Vec<f64>> {
        let k = match head {
            Some(h) => self.head(h)?,
            None => 0,
        };
        let (fit, d) = (&self.inner, &design.inner);
        py.detach(|| fit.predict_head(k, d, offset.as_deref()))
            .map_err(to_py)
    }

    /// Each head's correction to its linear predictor, per row: a list per
    /// head.
    fn corrections(&self, py: Python<'_>, design: PyRef<'_, PyDesign>) -> PyResult<Vec<Vec<f64>>> {
        let (fit, d) = (&self.inner, &design.inner);
        py.detach(|| fit.corrections(d)).map_err(to_py)
    }

    /// The embedding of ``factor``: a dict with ``levels`` (the reference
    /// first, then the design's order), ``columns`` (``factor:e1``, ...)
    /// and ``values``, a list per level.
    ///
    /// Parameters
    /// ----------
    /// factor : str
    ///
    /// Returns
    /// -------
    /// dict
    fn embeddings<'py>(
        &self,
        py: Python<'py>,
        factor: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let t = self.inner.embeddings(factor).map_err(to_py)?;
        let out = pyo3::types::PyDict::new(py);
        out.set_item("factor", &t.factor)?;
        out.set_item("levels", &t.levels)?;
        out.set_item("columns", t.column_names())?;
        let values: Vec<Vec<f64>> = (0..t.levels.len()).map(|k| t.level(k).to_vec()).collect();
        out.set_item("values", values)?;
        Ok(out)
    }

    /// ``design`` with each embedded factor's indicator columns replaced by
    /// its embedding coordinates (``factor:e1``, ...), for a GLM.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    ///
    /// Returns
    /// -------
    /// Design
    fn embedding_design(&self, design: PyRef<'_, PyDesign>) -> PyResult<PyDesign> {
        Ok(PyDesign {
            inner: self.inner.embedding_design(&design.inner).map_err(to_py)?,
        })
    }

    /// Joint predictive distribution of the first head across the rows
    /// (process draws around its means, as ``CannFit``'s).
    fn predict_distribution(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        n_sims: usize,
        seed: u64,
    ) -> PyResult<PyPredictiveDistribution> {
        let (fit, d) = (&self.inner, &design.inner);
        let inner = py
            .detach(|| fit.predict_distribution(d, n_sims, seed))
            .map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    /// Fitted means of one head on the training rows.
    #[pyo3(signature = (head = None))]
    fn fitted(&self, head: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
        let k = match head {
            Some(h) => self.head(h)?,
            None => 0,
        };
        Ok(self.inner.fitted_head(k).to_vec())
    }

    /// Each head's dispersion: 1 for the Poisson, otherwise Pearson's
    /// estimate on the rows with a positive weight.
    #[getter]
    fn dispersions(&self) -> Vec<f64> {
        self.inner.dispersions().to_vec()
    }

    /// With early stopping, the epoch (from 1) whose network was kept.
    #[getter]
    fn best_epoch(&self) -> Option<usize> {
        self.inner.best_epoch()
    }

    /// With early stopping, the validation loss after each epoch.
    #[getter]
    fn validation_history(&self) -> Vec<f64> {
        self.inner.validation_history().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "TabularNetFit(heads={:?}, n={})",
            self.inner.head_names(),
            self.inner.fitted_head(0).len()
        )
    }
}
