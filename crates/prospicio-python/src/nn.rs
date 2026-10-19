//! Neural networks (`prospicio-nn`): the CANN, the attention CANN and
//! networks imported from ONNX, behind the same `Design` and predictive
//! distributions as the GLM.

use prospicio_models::{Fitted, Model};
use prospicio_nn::onnx::{OnnxModel, OnnxPredictor, OnnxTensor, OutputScale};
use prospicio_nn::{AttentionCann, AttentionCannFit, Cann, CannFit, EarlyStopping, TokenAttention};
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
