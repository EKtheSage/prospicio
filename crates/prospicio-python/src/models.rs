//! `prospicio.models`: model terms and designs, GLMs, elastic nets, GAMs, metrics,
//! resampling and MCMC diagnostics (Models lane; `docs/design/models.md`).

use std::collections::HashMap;

use prospicio_bayes::ppc::{self, LooPit, Statistic, StatisticCheck};
use prospicio_glm::gam::{Gam, GamFit, PSpline, Smoothing, TensorSpline};
use prospicio_glm::net::{CvPath, ElasticNet, ElasticNetFit};
use prospicio_glm::{Dispersion, Glm, GlmFit, Robust};
use prospicio_models::resample::{self, Split};
use prospicio_models::{
    Coding, Column, Design, Family, Fitted, Frame, Link, Model, Terms, metrics,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::distributions::{KeyArg, PyPredictiveDistribution, key_from_py};
use crate::to_py;

/// A family from its name and parameter.
pub(crate) fn family(name: &str, theta: Option<f64>, power: Option<f64>) -> PyResult<Family> {
    let f = match name {
        "gaussian" => Family::Gaussian,
        "poisson" => Family::Poisson,
        "gamma" => Family::Gamma,
        "inverse_gaussian" => Family::InverseGaussian,
        "binomial" => Family::Binomial,
        "negative_binomial" => Family::NegativeBinomial {
            theta: theta.ok_or_else(|| PyValueError::new_err("negative_binomial needs theta"))?,
        },
        "tweedie" => Family::Tweedie {
            power: power.ok_or_else(|| PyValueError::new_err("tweedie needs power"))?,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "family must be gaussian, poisson, gamma, inverse_gaussian, binomial, \
                 negative_binomial or tweedie, got {other:?}"
            )));
        }
    };
    f.validate().map_err(to_py)?;
    Ok(f)
}

/// A link from its name (`None` for the family's canonical link).
pub(crate) fn link(name: Option<&str>, family: Family, power: Option<f64>) -> PyResult<Link> {
    Ok(match name {
        None => family.canonical_link(),
        Some("identity") => Link::Identity,
        Some("log") => Link::Log,
        Some("logit") => Link::Logit,
        Some("probit") => Link::Probit,
        Some("cloglog") => Link::Cloglog,
        Some("inverse") => Link::Inverse,
        Some("inverse_squared") => Link::InverseSquared,
        Some("power") => Link::Power(
            power.ok_or_else(|| PyValueError::new_err("the power link needs link_power"))?,
        ),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "link must be identity, log, logit, probit, cloglog, inverse, \
                 inverse_squared or power, got {other:?}"
            )));
        }
    })
}

pub(crate) fn link_name(link: Link) -> String {
    match link {
        Link::Identity => "identity".into(),
        Link::Log => "log".into(),
        Link::Logit => "logit".into(),
        Link::Probit => "probit".into(),
        Link::Cloglog => "cloglog".into(),
        Link::Inverse => "inverse".into(),
        Link::InverseSquared => "inverse_squared".into(),
        Link::Power(p) => format!("power({p})"),
    }
}

/// A frame from a dict of columns: lists of numbers are numeric, lists of
/// strings are categorical.
fn frame(data: &Bound<'_, PyAny>) -> PyResult<Frame> {
    let map: HashMap<String, Bound<'_, PyAny>> = data
        .extract()
        .map_err(|_| PyTypeError::new_err("data must be a dict of column name to list"))?;
    let mut names: Vec<&String> = map.keys().collect();
    names.sort();
    let mut columns = Vec::with_capacity(names.len());
    for name in names {
        let col = &map[name];
        let column = if let Ok(v) = col.extract::<Vec<f64>>() {
            Column::Numeric(v)
        } else if let Ok(v) = col.extract::<Vec<String>>() {
            Column::Categorical(v)
        } else {
            return Err(PyTypeError::new_err(format!(
                "column {name:?} must be a list of numbers or of strings"
            )));
        };
        columns.push((name.clone(), column));
    }
    Frame::new(columns).map_err(to_py)
}

fn with_offset_weights(
    mut d: Design,
    offset: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
) -> PyResult<Design> {
    if let Some(o) = offset {
        d = d.with_offset(o).map_err(to_py)?;
    }
    if let Some(w) = weights {
        d = d.with_weights(w).map_err(to_py)?;
    }
    Ok(d)
}

/// The terms of a model: an intercept, numeric columns and factors.
///
/// Build them up, then ``fit`` them to training data to learn the factor
/// levels; the result builds the same design matrix on any data.
///
/// Examples
/// --------
/// >>> from prospicio.models import Terms
/// >>> data = {"age": [30.0, 45.0, 60.0], "region": ["N", "S", "W"]}
/// >>> coding = Terms().intercept().numeric("age").factor("region").fit(data)
/// >>> coding.names
/// ['(Intercept)', 'age', 'region[S]', 'region[W]']
#[pyclass(name = "Terms", module = "prospicio.models", frozen)]
pub(crate) struct PyTerms {
    inner: Terms,
}

#[pymethods]
impl PyTerms {
    #[new]
    fn new() -> Self {
        Self {
            inner: Terms::new(),
        }
    }

    /// Adds an intercept.
    ///
    /// Returns
    /// -------
    /// Terms
    fn intercept(&self) -> Self {
        Self {
            inner: self.inner.clone().intercept(),
        }
    }

    /// Adds a numeric column.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///
    /// Returns
    /// -------
    /// Terms
    fn numeric(&self, name: &str) -> Self {
        Self {
            inner: self.inner.clone().numeric(name),
        }
    }

    /// Adds a factor in treatment coding.
    ///
    /// Parameters
    /// ----------
    /// name : str
    /// reference : str, optional
    ///     Reference level; the first in sorted order by default.
    ///
    /// Returns
    /// -------
    /// Terms
    #[pyo3(signature = (name, reference = None))]
    fn factor(&self, name: &str, reference: Option<&str>) -> Self {
        let inner = match reference {
            Some(r) => self.inner.clone().factor_with_reference(name, r),
            None => self.inner.clone().factor(name),
        };
        Self { inner }
    }

    /// Learns factor levels from training data.
    ///
    /// Parameters
    /// ----------
    /// data : dict of str to list
    ///     Numeric columns as lists of numbers, factors as lists of strings.
    ///
    /// Returns
    /// -------
    /// Coding
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a column is missing or has the wrong kind.
    fn fit(&self, data: &Bound<'_, PyAny>) -> PyResult<PyCoding> {
        let inner = self.inner.fit(&frame(data)?).map_err(to_py)?;
        Ok(PyCoding { inner })
    }
}

/// Terms with factor levels learned from training data, from
/// ``Terms.fit``.
#[pyclass(name = "Coding", module = "prospicio.models", frozen)]
pub(crate) struct PyCoding {
    inner: Coding,
}

#[pymethods]
impl PyCoding {
    /// Design matrix column names.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    /// The design matrix for ``data``.
    ///
    /// Parameters
    /// ----------
    /// data : dict of str to list
    /// offset : list of float, optional
    /// weights : list of float, optional
    ///
    /// Returns
    /// -------
    /// Design
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a column is missing or a factor level was not seen in training.
    #[pyo3(signature = (data, offset = None, weights = None))]
    fn design(
        &self,
        data: &Bound<'_, PyAny>,
        offset: Option<Vec<f64>>,
        weights: Option<Vec<f64>>,
    ) -> PyResult<PyDesign> {
        let d = self.inner.design(&frame(data)?).map_err(to_py)?;
        Ok(PyDesign {
            inner: with_offset_weights(d, offset, weights)?,
        })
    }
}

/// A design matrix with an offset and prior weights.
///
/// Parameters
/// ----------
/// columns : list of list of float
///     One list per column.
/// names : list of str
/// offset : list of float, optional
/// weights : list of float, optional
///
/// Examples
/// --------
/// >>> from prospicio.models import Design
/// >>> d = Design([[1.0, 1.0], [0.0, 2.0]], ["(Intercept)", "x"])
/// >>> d.n_rows, d.names
/// (2, ['(Intercept)', 'x'])
#[pyclass(name = "Design", module = "prospicio.models", frozen)]
pub(crate) struct PyDesign {
    pub(crate) inner: Design,
}

#[pymethods]
impl PyDesign {
    #[new]
    #[pyo3(signature = (columns, names, offset = None, weights = None))]
    fn new(
        columns: Vec<Vec<f64>>,
        names: Vec<String>,
        offset: Option<Vec<f64>>,
        weights: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let d = Design::new(names, columns).map_err(to_py)?;
        Ok(Self {
            inner: with_offset_weights(d, offset, weights)?,
        })
    }

    /// Column names.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    /// Number of rows.
    #[getter]
    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    /// Offset per row.
    #[getter]
    fn offset(&self) -> Vec<f64> {
        self.inner.offset().to_vec()
    }

    /// Prior weight per row.
    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.inner.weights().to_vec()
    }

    /// Column ``j``.
    ///
    /// Parameters
    /// ----------
    /// j : int
    ///
    /// Returns
    /// -------
    /// list of float
    fn column(&self, j: usize) -> PyResult<Vec<f64>> {
        if j >= self.inner.n_cols() {
            return Err(PyValueError::new_err("column index out of range"));
        }
        Ok(self.inner.column(j).to_vec())
    }

    /// The rows ``rows``, in that order.
    ///
    /// Parameters
    /// ----------
    /// rows : list of int
    ///
    /// Returns
    /// -------
    /// Design
    fn select(&self, rows: Vec<usize>) -> PyResult<Self> {
        if rows.iter().any(|&i| i >= self.inner.n_rows()) {
            return Err(PyValueError::new_err("row index out of range"));
        }
        Ok(Self {
            inner: self.inner.select(&rows),
        })
    }
}

/// A generalized linear model, fitted by IRLS.
///
/// Parameters
/// ----------
/// family : str
///     ``"gaussian"``, ``"poisson"``, ``"gamma"``, ``"inverse_gaussian"``,
///     ``"binomial"``, ``"negative_binomial"`` (needs ``theta``) or
///     ``"tweedie"`` (needs ``power``).
/// link : str, optional
///     ``"identity"``, ``"log"``, ``"logit"``, ``"probit"``,
///     ``"cloglog"``, ``"inverse"``, ``"inverse_squared"`` or ``"power"``
///     (needs ``link_power``); the family's canonical link by default.
/// dispersion : str or float, optional
///     ``"pearson"``, ``"deviance"`` or a fixed value. By default 1 for the
///     Poisson, binomial and negative binomial and Pearson's estimate
///     otherwise; ``"pearson"`` with the Poisson is the over-dispersed
///     (quasi-) Poisson, which accepts negative responses as long as the
///     fitted means stay positive.
/// theta : float, optional
/// power : float, optional
/// link_power : float, optional
///
/// Examples
/// --------
/// >>> from prospicio.models import Design, Glm
/// >>> d = Design([[1.0] * 4, [0.0, 0.0, 1.0, 1.0]], ["(Intercept)", "young"],
/// ...            offset=[0.0, 0.0, 0.0, 0.0])
/// >>> fit = Glm("poisson", "log").fit(d, [1.0, 3.0, 4.0, 6.0])
/// >>> round(fit.coefficients[1], 10) == round(__import__("math").log(5 / 2), 10)
/// True
#[pyclass(name = "Glm", module = "prospicio.models", frozen)]
pub(crate) struct PyGlm {
    inner: Glm,
}

fn dispersion(arg: Option<&Bound<'_, PyAny>>, default: Dispersion) -> PyResult<Dispersion> {
    let Some(arg) = arg else {
        return Ok(default);
    };
    if let Ok(v) = arg.extract::<f64>() {
        return Ok(Dispersion::Fixed(v));
    }
    match arg.extract::<String>()?.as_str() {
        "pearson" => Ok(Dispersion::Pearson),
        "deviance" => Ok(Dispersion::Deviance),
        other => Err(PyValueError::new_err(format!(
            "dispersion must be \"pearson\", \"deviance\" or a number, got {other:?}"
        ))),
    }
}

#[pymethods]
impl PyGlm {
    #[new]
    #[pyo3(signature = (family, link = None, dispersion = None, theta = None, power = None, link_power = None))]
    fn new(
        family: &str,
        link: Option<&str>,
        dispersion: Option<&Bound<'_, PyAny>>,
        theta: Option<f64>,
        power: Option<f64>,
        link_power: Option<f64>,
    ) -> PyResult<Self> {
        let f = self::family(family, theta, power)?;
        let l = self::link(link, f, link_power)?;
        let base = Glm::new(f, l);
        let inner = base.dispersion(self::dispersion(dispersion, base.dispersion)?);
        Ok(Self { inner })
    }

    /// Fits the model.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// y : list of float
    ///
    /// Returns
    /// -------
    /// GlmFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the design is collinear, a response is out of the family's
    ///     range, or IRLS does not converge.
    fn fit(&self, py: Python<'_>, design: PyRef<'_, PyDesign>, y: Vec<f64>) -> PyResult<PyGlmFit> {
        let (glm, d) = (self.inner, &design.inner);
        let inner = py.detach(|| glm.fit(d, &y)).map_err(to_py)?;
        Ok(PyGlmFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "Glm(family={:?}, link={:?})",
            self.inner.family.name(),
            link_name(self.inner.link)
        )
    }
}

impl PyGlmFit {
    fn robust(
        &self,
        py: Python<'_>,
        design: &Design,
        y: &[f64],
        kind: &str,
        groups: Option<Vec<KeyArg>>,
    ) -> PyResult<Vec<f64>> {
        let labels: Option<Vec<usize>> = groups.map(|g| {
            let mut index = HashMap::new();
            key_from_py(g)
                .into_iter()
                .map(|k| {
                    let next = index.len();
                    *index.entry(k).or_insert(next)
                })
                .collect()
        });
        let kind = match (kind, &labels) {
            ("HC0", None) => Robust::Hc0,
            ("HC1", None) => Robust::Hc1,
            ("cluster", Some(l)) => Robust::Cluster(l),
            ("cluster", None) => {
                return Err(PyValueError::new_err("kind=\"cluster\" needs groups"));
            }
            ("HC0" | "HC1", Some(_)) => {
                return Err(PyValueError::new_err(
                    "groups are only for kind=\"cluster\"",
                ));
            }
            (other, _) => {
                return Err(PyValueError::new_err(format!(
                    "kind must be \"HC0\", \"HC1\" or \"cluster\", got {other:?}"
                )));
            }
        };
        let fit = &self.inner;
        py.detach(|| fit.robust_covariance(design, y, kind))
            .map_err(to_py)
    }
}

/// A fitted GLM, from ``Glm.fit``.
#[pyclass(name = "GlmFit", module = "prospicio.models", frozen)]
pub(crate) struct PyGlmFit {
    inner: GlmFit,
}

#[pymethods]
impl PyGlmFit {
    /// Coefficient names.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    /// Estimated coefficients.
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.inner.coefficients().to_vec()
    }

    /// Standard errors.
    #[getter]
    fn std_errors(&self) -> Vec<f64> {
        self.inner.std_errors()
    }

    /// Two-sided p-values (normal for a fixed dispersion, Student's t when
    /// it is estimated).
    #[getter]
    fn p_values(&self) -> Vec<f64> {
        self.inner.p_values()
    }

    /// Sandwich (heteroskedasticity- or cluster-robust) covariance of the
    /// coefficients, as statsmodels' ``cov_type="HC0"`` and ``"cluster"``.
    ///
    /// It stays valid when the variance function or dispersion is wrong,
    /// as long as the mean is right. The dispersion cancels. For a
    /// non-canonical link it uses the observed information, as
    /// statsmodels does (R's ``sandwich`` uses the expected).
    ///
    /// Parameters
    /// ----------
    /// design : Design
    ///     The design the model was fitted on.
    /// y : list of float
    ///     The response the model was fitted on.
    /// kind : {"HC0", "HC1", "cluster"}, default "HC0"
    ///     ``"HC1"`` scales HC0 by ``n / (n - p)``; ``"cluster"`` sums the
    ///     scores within each cluster and scales by
    ///     ``G / (G - 1) * (n - 1) / (n - p)``.
    /// groups : list of int or str, optional
    ///     One cluster label per row, for ``kind="cluster"``: a policy or
    ///     an event, say.
    ///
    /// Returns
    /// -------
    /// list of list of float
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.models import Design, Glm
    /// >>> d = Design([[1.0] * 6, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], ["(Intercept)", "x"])
    /// >>> y = [1.0, 2.0, 6.0, 1.0, 4.0, 2.0]
    /// >>> fit = Glm("poisson").fit(d, y)
    /// >>> round(fit.robust_covariance(d, y)[0][0] * 81, 10)
    /// 14.0
    /// >>> se = fit.robust_std_errors(d, y, "cluster", groups=[1, 1, 2, 2, 3, 3])
    #[pyo3(signature = (design, y, kind = "HC0", groups = None))]
    fn robust_covariance(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
        kind: &str,
        groups: Option<Vec<KeyArg>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let p = self.inner.coefficients().len();
        let v = self.robust(py, &design.inner, &y, kind, groups)?;
        Ok(v.chunks(p).map(<[f64]>::to_vec).collect())
    }

    /// Square roots of the diagonal of ``robust_covariance``, with the same
    /// arguments.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// y : list of float
    /// kind : {"HC0", "HC1", "cluster"}, default "HC0"
    /// groups : list of int or str, optional
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (design, y, kind = "HC0", groups = None))]
    fn robust_std_errors(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
        kind: &str,
        groups: Option<Vec<KeyArg>>,
    ) -> PyResult<Vec<f64>> {
        let p = self.inner.coefficients().len();
        let v = self.robust(py, &design.inner, &y, kind, groups)?;
        Ok((0..p).map(|j| v[j * p + j].sqrt()).collect())
    }

    /// The fit as a versioned JSON artifact: spec, estimates, covariance,
    /// fit statistics, fitted values and provenance (crate version and a
    /// hash of the training data). ``GlmFit.from_json`` reads it back
    /// exactly; pickling uses it too.
    ///
    /// Returns
    /// -------
    /// str
    ///
    /// Examples
    /// --------
    /// >>> import pickle
    /// >>> from prospicio.models import Design, Glm, GlmFit
    /// >>> d = Design([[1.0] * 4, [0.0, 1.0, 2.0, 3.0]], ["(Intercept)", "x"])
    /// >>> fit = Glm("poisson").fit(d, [1.0, 2.0, 2.0, 5.0])
    /// >>> GlmFit.from_json(fit.to_json()).coefficients == fit.coefficients
    /// True
    /// >>> pickle.loads(pickle.dumps(fit)).input_hash == fit.input_hash
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
    /// GlmFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For malformed JSON, another format, a newer format version or
    ///     inconsistent fields.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: GlmFit::from_json(text).map_err(to_py)?,
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

    /// Covariance of the coefficients, as a list of rows.
    #[getter]
    fn covariance(&self) -> Vec<Vec<f64>> {
        let p = self.inner.coefficients().len();
        self.inner
            .covariance()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Dispersion.
    #[getter]
    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    /// Residual deviance.
    #[getter]
    fn deviance(&self) -> f64 {
        self.inner.deviance()
    }

    /// Deviance of the intercept-and-offset model.
    #[getter]
    fn null_deviance(&self) -> f64 {
        self.inner.null_deviance()
    }

    /// Log-likelihood.
    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood()
    }

    /// AIC, ``-2 loglik + 2 p``.
    #[getter]
    fn aic(&self) -> f64 {
        self.inner.aic()
    }

    /// BIC, ``-2 loglik + p log(n)``, counting the coefficients as ``aic``
    /// does (R's ``BIC`` also counts an estimated dispersion).
    #[getter]
    fn bic(&self) -> f64 {
        self.inner.bic()
    }

    /// Residual degrees of freedom.
    #[getter]
    fn df_resid(&self) -> f64 {
        self.inner.df_resid()
    }

    /// IRLS iterations used.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations()
    }

    /// Fitted means on the training data.
    #[getter]
    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted().to_vec()
    }

    /// Expected response for each row.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    ///     Same columns as the training design.
    ///
    /// Returns
    /// -------
    /// list of float
    fn predict(&self, design: PyRef<'_, PyDesign>) -> PyResult<Vec<f64>> {
        self.inner.predict(&design.inner).map_err(to_py)
    }

    /// Joint predictive distribution across the rows, with parameter and
    /// process uncertainty, keyed ``row = 0, 1, ...``.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// n_sims : int
    /// seed : int
    /// parameters : {"normal", "mean_preserving", "fixed"}, default "normal"
    ///     How the coefficients are drawn: ``beta ~ N(beta_hat, Sigma)``
    ///     (through a log link the draws' mean is
    ///     ``mu_hat * exp(x' Sigma x / 2)``); the same with each row's linear
    ///     predictor shifted so its draws average the fitted mean exactly
    ///     (log or identity link); or fixed at ``beta_hat`` (process
    ///     uncertainty only).
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    #[pyo3(signature = (design, n_sims, seed, parameters = "normal"))]
    fn predict_distribution(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        n_sims: usize,
        seed: u64,
        parameters: &str,
    ) -> PyResult<PyPredictiveDistribution> {
        let parameters = match parameters {
            "normal" => prospicio_glm::ParameterDraws::Normal,
            "mean_preserving" => prospicio_glm::ParameterDraws::MeanPreserving,
            "fixed" => prospicio_glm::ParameterDraws::Fixed,
            _ => {
                return Err(PyValueError::new_err(
                    "parameters must be \"normal\", \"mean_preserving\" or \"fixed\"",
                ));
            }
        };
        let (fit, d) = (&self.inner, &design.inner);
        let inner = py
            .detach(|| fit.predict_distribution_with(d, n_sims, seed, parameters))
            .map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "GlmFit(family={:?}, deviance={}, n_coefficients={})",
            self.inner.spec().family.name(),
            self.inner.deviance(),
            self.inner.coefficients().len()
        )
    }
}

/// An elastic-net GLM: the lasso (``alpha=1``), ridge (``alpha=0``) and
/// everything between, minimizing glmnet's objective
/// ``sum(w * d) / (2 * sum(w)) + lam * sum(pf * ((1 - alpha) / 2 * b**2 + alpha * |b|))``
/// over coefficients ``b`` of standardized columns. The design's first
/// all-ones column is the unpenalized intercept; coefficients are reported
/// on the design's scale.
///
/// Parameters
/// ----------
/// family : str
///     As ``Glm``.
/// link : str, optional
///     As ``Glm``; the canonical link by default.
/// alpha : float, default 1.0
///     Mixing between ridge (0) and the lasso (1).
/// lam : float, default 0.0
///     Penalty strength (``lambda`` in glmnet).
/// standardize : bool, default True
///     Penalize the coefficients of columns scaled to unit standard
///     deviation.
/// penalty_factor : list of float, optional
///     One factor per design column (the intercept's is ignored); 0 leaves
///     a column unpenalized.
/// theta : float, optional
/// power : float, optional
/// link_power : float, optional
///
/// Examples
/// --------
/// >>> from prospicio.models import Design, ElasticNet
/// >>> x = [float(i) for i in range(6)]
/// >>> d = Design([[1.0] * 6, x, [1.0, 0.0] * 3], ["(Intercept)", "x1", "x2"])
/// >>> y = [1.0, 3.1, 4.9, 7.2, 9.0, 10.8]
/// >>> net = ElasticNet("gaussian", alpha=1.0)
/// >>> top = net.lambda_max(d, y)
/// >>> net.with_lam(1.01 * top).fit(d, y).coefficients[1:]
/// [0.0, 0.0]
#[pyclass(name = "ElasticNet", module = "prospicio.models", frozen)]
pub(crate) struct PyElasticNet {
    inner: ElasticNet,
}

#[pymethods]
impl PyElasticNet {
    #[new]
    #[pyo3(signature = (family, link = None, alpha = 1.0, lam = 0.0, standardize = true, penalty_factor = None, theta = None, power = None, link_power = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        family: &str,
        link: Option<&str>,
        alpha: f64,
        lam: f64,
        standardize: bool,
        penalty_factor: Option<Vec<f64>>,
        theta: Option<f64>,
        power: Option<f64>,
        link_power: Option<f64>,
    ) -> PyResult<Self> {
        let f = self::family(family, theta, power)?;
        let l = self::link(link, f, link_power)?;
        let mut inner = ElasticNet::new(f, l, alpha, lam).standardize(standardize);
        inner.penalty_factor = penalty_factor;
        Ok(Self { inner })
    }

    /// Mixing parameter.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    /// Penalty strength.
    #[getter]
    fn lam(&self) -> f64 {
        self.inner.lambda
    }

    /// The same spec at another penalty strength.
    ///
    /// Parameters
    /// ----------
    /// lam : float
    ///
    /// Returns
    /// -------
    /// ElasticNet
    fn with_lam(&self, lam: f64) -> Self {
        Self {
            inner: self.inner.with_lambda(lam),
        }
    }

    /// Fits at ``lam``.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// y : list of float
    ///
    /// Returns
    /// -------
    /// ElasticNetFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a parameter is out of range, a response is outside the
    ///     family's range, or the fit does not converge.
    fn fit(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
    ) -> PyResult<PyElasticNetFit> {
        let (net, d) = (&self.inner, &design.inner);
        let inner = py.detach(|| net.fit(d, &y)).map_err(to_py)?;
        Ok(PyElasticNetFit { inner })
    }

    /// The smallest ``lam`` at which every penalized coefficient is zero.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// y : list of float
    ///
    /// Returns
    /// -------
    /// float
    fn lambda_max(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
    ) -> PyResult<f64> {
        let (net, d) = (&self.inner, &design.inner);
        py.detach(|| net.lambda_max(d, &y)).map_err(to_py)
    }

    /// ``n`` penalty strengths, log-spaced from ``lambda_max`` down to
    /// ``min_ratio`` times it.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// y : list of float
    /// n : int, default 100
    /// min_ratio : float, default 1e-4
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (design, y, n = 100, min_ratio = 1e-4))]
    fn lambda_path(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
        n: usize,
        min_ratio: f64,
    ) -> PyResult<Vec<f64>> {
        let (net, d) = (&self.inner, &design.inner);
        py.detach(|| net.lambda_path(d, &y, n, min_ratio))
            .map_err(to_py)
    }

    /// Fits at each of ``lams`` in turn, each from the previous solution.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// y : list of float
    /// lams : list of float
    ///
    /// Returns
    /// -------
    /// list of ElasticNetFit
    fn path(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
        lams: Vec<f64>,
    ) -> PyResult<Vec<PyElasticNetFit>> {
        let (net, d) = (&self.inner, &design.inner);
        let fits = py.detach(|| net.path(d, &y, &lams)).map_err(to_py)?;
        Ok(fits
            .into_iter()
            .map(|inner| PyElasticNetFit { inner })
            .collect())
    }

    /// Cross-validates the path, like glmnet's ``cv.glmnet``: on each
    /// split, fits ``lams`` (warm starts) to the training rows and scores
    /// the mean deviance on the test rows. Folds run in parallel.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// y : list of float
    /// lams : list of float
    ///     Penalty strengths, largest first (from ``lambda_path``).
    /// splits : list of (list of int, list of int)
    ///     ``(train, test)`` rows, as ``k_fold`` returns.
    ///
    /// Returns
    /// -------
    /// CvPath
    fn cross_validate(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
        lams: Vec<f64>,
        splits: Vec<(Vec<usize>, Vec<usize>)>,
    ) -> PyResult<PyCvPath> {
        let splits: Vec<Split> = splits
            .into_iter()
            .map(|(train, test)| Split { train, test })
            .collect();
        let (net, d) = (&self.inner, &design.inner);
        let inner = py
            .detach(|| net.cross_validate(d, &y, &lams, &splits))
            .map_err(to_py)?;
        Ok(PyCvPath { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ElasticNet(family={:?}, link={:?}, alpha={:?}, lam={:?})",
            self.inner.family.name(),
            link_name(self.inner.link),
            self.inner.alpha,
            self.inner.lambda
        )
    }
}

/// Cross-validated scores along an elastic-net path, from
/// ``ElasticNet.cross_validate``.
#[pyclass(name = "CvPath", module = "prospicio.models", frozen)]
pub(crate) struct PyCvPath {
    inner: CvPath,
}

#[pymethods]
impl PyCvPath {
    /// Penalty strengths, in the order given.
    #[getter]
    fn lams(&self) -> Vec<f64> {
        self.inner.lambdas.clone()
    }

    /// Mean deviance over folds (weighted by fold weight), per ``lam``.
    #[getter]
    fn mean(&self) -> Vec<f64> {
        self.inner.mean.clone()
    }

    /// Standard error of ``mean``, per ``lam``.
    #[getter]
    fn se(&self) -> Vec<f64> {
        self.inner.se.clone()
    }

    /// Each fold's mean deviance per ``lam``.
    #[getter]
    fn fold_scores(&self) -> Vec<Vec<f64>> {
        self.inner.fold_scores.clone()
    }

    /// ``lam`` with the lowest mean deviance.
    #[getter]
    fn lam_min(&self) -> f64 {
        self.inner.lambda_min()
    }

    /// The largest ``lam`` within one standard error of the lowest.
    #[getter]
    fn lam_1se(&self) -> f64 {
        self.inner.lambda_1se()
    }

    fn __repr__(&self) -> String {
        format!(
            "CvPath(n_lams={}, lam_min={}, lam_1se={})",
            self.inner.lambdas.len(),
            self.inner.lambda_min(),
            self.inner.lambda_1se()
        )
    }
}

/// A fitted elastic net, from ``ElasticNet.fit`` or ``ElasticNet.path``.
#[pyclass(name = "ElasticNetFit", module = "prospicio.models", frozen)]
pub(crate) struct PyElasticNetFit {
    inner: ElasticNetFit,
}

#[pymethods]
impl PyElasticNetFit {
    /// The fit as a versioned JSON artifact (spec with its lambda and alpha, coefficients, fit statistics), with provenance (crate
    /// version and a hash of the training data). ``ElasticNetFit.from_json`` reads
    /// it back exactly; pickling uses it too.
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
    /// ElasticNetFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For malformed JSON, another format, a newer format version or
    ///     inconsistent fields.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ElasticNetFit::from_json(text).map_err(to_py)?,
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

    /// Coefficient names.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    /// Coefficients on the design's scale; exact zeros where the penalty
    /// dropped a column.
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.inner.coefficients().to_vec()
    }

    /// Penalty strength.
    #[getter]
    fn lam(&self) -> f64 {
        self.inner.lambda()
    }

    /// Residual deviance.
    #[getter]
    fn deviance(&self) -> f64 {
        self.inner.deviance()
    }

    /// Deviance with only the intercept and unpenalized columns.
    #[getter]
    fn null_deviance(&self) -> f64 {
        self.inner.null_deviance()
    }

    /// Share of the null deviance explained (glmnet's ``dev.ratio``).
    #[getter]
    fn deviance_ratio(&self) -> f64 {
        self.inner.deviance_ratio()
    }

    /// Number of non-zero coefficients, intercept excluded.
    #[getter]
    fn df(&self) -> usize {
        self.inner.df()
    }

    /// Dispersion.
    #[getter]
    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    /// Fitted means on the training data.
    #[getter]
    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted().to_vec()
    }

    /// Expected response for each row.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    ///     Same columns as the training design.
    ///
    /// Returns
    /// -------
    /// list of float
    fn predict(&self, design: PyRef<'_, PyDesign>) -> PyResult<Vec<f64>> {
        self.inner.predict(&design.inner).map_err(to_py)
    }

    /// Joint predictive distribution across the rows, keyed
    /// ``row = 0, 1, ...``: process uncertainty only (penalized
    /// coefficients have no standard errors; bootstrap the fit for
    /// parameter uncertainty).
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

    fn __repr__(&self) -> String {
        format!(
            "ElasticNetFit(family={:?}, lam={}, df={}, deviance={})",
            self.inner.spec().family.name(),
            self.inner.lambda(),
            self.inner.df(),
            self.inner.deviance()
        )
    }
}

/// A smooth from ``{"column": ..., "n_basis": ..., "bs": ..., "period": ...}``.
fn smooth_from_dict(d: &Bound<'_, pyo3::types::PyDict>) -> PyResult<PSpline> {
    for key in d.keys() {
        let key: String = key.extract()?;
        if !matches!(key.as_str(), "column" | "n_basis" | "bs" | "period") {
            return Err(PyValueError::new_err(format!(
                "unknown smooth key {key:?}: use column, n_basis, bs and period"
            )));
        }
    }
    let column: String = d
        .get_item("column")?
        .ok_or_else(|| PyValueError::new_err("a smooth needs a \"column\""))?
        .extract()?;
    let bs: String = match d.get_item("bs")? {
        Some(v) => v.extract()?,
        None => "ps".into(),
    };
    let period: Option<(f64, f64)> = match d.get_item("period")? {
        Some(v) if !v.is_none() => Some(v.extract()?),
        _ => None,
    };
    let mut spline = match bs.as_str() {
        "ps" if period.is_none() => PSpline::new(&column),
        "ps" => {
            return Err(PyValueError::new_err(
                "a period needs a cyclic smooth (bs=\"cp\")",
            ));
        }
        "cp" => match period {
            Some((lo, hi)) => PSpline::cyclic(&column).period(lo, hi),
            None => PSpline::cyclic(&column),
        },
        "mpi" | "mpd" if period.is_some() => {
            return Err(PyValueError::new_err(
                "a period needs a cyclic smooth (bs=\"cp\")",
            ));
        }
        "mpi" => PSpline::increasing(&column),
        "mpd" => PSpline::decreasing(&column),
        other => {
            return Err(PyValueError::new_err(format!(
                "bs must be \"ps\", \"cp\", \"mpi\" or \"mpd\", not {other:?}"
            )));
        }
    };
    if let Some(k) = d.get_item("n_basis")? {
        spline = spline.n_basis(k.extract()?);
    }
    Ok(spline)
}

/// A generalized additive model: a ``Glm`` plus P-spline smooths of
/// numeric design columns, with smoothing chosen by GCV or UBRE.
///
/// Parameters
/// ----------
/// glm : Glm
///     Family, link and dispersion.
/// smooths : list of str, (str, int) or dict
///     The design columns to smooth, optionally with the number of basis
///     functions (10 by default), or a dict ``{"column": ..., "n_basis":
///     ..., "bs": ..., "period": ...}``: ``bs`` is ``"ps"`` (the default)
///     or ``"cp"`` for a cyclic smooth (mgcv's ``bs = "cp"``), whose ends
///     meet over ``period``, a ``(lo, hi)`` pair (the data's range when
///     left out), or ``"mpi"`` / ``"mpd"`` for a monotone increasing or
///     decreasing smooth (scam's), which takes GCV, UBRE or fixed
///     smoothing but not REML.
/// smoothing : str or list of float, default "auto"
///     ``"auto"`` (UBRE for a fixed dispersion, GCV otherwise), ``"gcv"``,
///     ``"ubre"``, ``"reml"`` (mgcv's ``method = "REML"``, estimating an
///     unknown dispersion with the smoothing parameters), or fixed
///     smoothing parameters: one per smooth, then one per tensor margin.
/// tensors : list, optional
///     Tensor product smooths (mgcv's ``te(..., bs = "ps", np = FALSE)``):
///     each a list of two or more column names, or ``(columns, n_basis)``
///     with one basis size per column (5 by default). Each margin has its
///     own smoothing parameter.
///
/// Examples
/// --------
/// >>> import math
/// >>> from prospicio.models import Design, Gam, Glm
/// >>> x = [i / 99 for i in range(100)]
/// >>> y = [math.sin(6 * v) for v in x]
/// >>> d = Design([[1.0] * 100, x], ["(Intercept)", "x"])
/// >>> fit = Gam(Glm("gaussian"), ["x"]).fit(d, y)
/// >>> abs(fit.predict(d)[50] - y[50]) < 0.01
/// True
#[pyclass(name = "Gam", module = "prospicio.models", frozen)]
pub(crate) struct PyGam {
    inner: Gam,
}

#[pymethods]
impl PyGam {
    #[new]
    #[pyo3(signature = (glm, smooths, smoothing = None, tensors = None))]
    fn new(
        glm: PyRef<'_, PyGlm>,
        smooths: Vec<Bound<'_, PyAny>>,
        smoothing: Option<&Bound<'_, PyAny>>,
        tensors: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Self> {
        let tensors = tensors
            .unwrap_or_default()
            .iter()
            .map(|t| {
                if let Ok(cols) = t.extract::<Vec<String>>() {
                    let refs: Vec<&str> = cols.iter().map(String::as_str).collect();
                    Ok(TensorSpline::new(&refs))
                } else if let Ok((cols, k)) = t.extract::<(Vec<String>, Vec<usize>)>() {
                    let refs: Vec<&str> = cols.iter().map(String::as_str).collect();
                    Ok(TensorSpline::new(&refs).n_basis(k))
                } else {
                    Err(PyTypeError::new_err(
                        "each tensor must be a list of column names, or (columns, n_basis per column)",
                    ))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        let splines = smooths
            .iter()
            .map(|s| {
                if let Ok(name) = s.extract::<String>() {
                    Ok(PSpline::new(&name))
                } else if let Ok((name, k)) = s.extract::<(String, usize)>() {
                    Ok(PSpline::new(&name).n_basis(k))
                } else if let Ok(d) = s.cast::<pyo3::types::PyDict>() {
                    smooth_from_dict(d)
                } else {
                    Err(PyTypeError::new_err(
                        "each smooth must be a column name, (column name, n_basis) or a dict \
                         with column, n_basis, bs and period",
                    ))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        let smoothing = match smoothing {
            None => Smoothing::Auto,
            Some(s) => {
                if let Ok(v) = s.extract::<Vec<f64>>() {
                    Smoothing::Fixed(v)
                } else {
                    match s.extract::<String>()?.as_str() {
                        "auto" => Smoothing::Auto,
                        "gcv" => Smoothing::Gcv,
                        "ubre" => Smoothing::Ubre,
                        "reml" => Smoothing::Reml,
                        other => {
                            return Err(PyValueError::new_err(format!(
                                "smoothing must be \"auto\", \"gcv\", \"ubre\", \"reml\" or a \
                                 list of numbers, got {other:?}"
                            )));
                        }
                    }
                }
            }
        };
        let gam = tensors
            .into_iter()
            .fold(Gam::new(glm.inner, splines), Gam::tensor);
        Ok(Self {
            inner: gam.smoothing(smoothing),
        })
    }

    /// Fits the model.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    ///     Includes the raw columns to smooth.
    /// y : list of float
    ///
    /// Returns
    /// -------
    /// GamFit
    fn fit(&self, py: Python<'_>, design: PyRef<'_, PyDesign>, y: Vec<f64>) -> PyResult<PyGamFit> {
        let (gam, d) = (&self.inner, &design.inner);
        let inner = py.detach(|| gam.fit(d, &y)).map_err(to_py)?;
        Ok(PyGamFit { inner })
    }
}

/// A fitted GAM, from ``Gam.fit``.
#[pyclass(name = "GamFit", module = "prospicio.models", frozen)]
pub(crate) struct PyGamFit {
    inner: GamFit,
}

#[pymethods]
impl PyGamFit {
    /// The fit as a versioned JSON artifact (GLM spec, smooths with their knots and constraints, smoothing parameters, estimates), with provenance (crate
    /// version and a hash of the training data). ``GamFit.from_json`` reads
    /// it back exactly; pickling uses it too.
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
    /// GamFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For malformed JSON, another format, a newer format version or
    ///     inconsistent fields.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: GamFit::from_json(text).map_err(to_py)?,
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

    /// Coefficient names: parametric columns, then ``s(x).1``, ...
    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    /// Coefficients.
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.inner.coefficients().to_vec()
    }

    /// Smoothing parameter of each smooth.
    #[getter]
    fn lambdas(&self) -> Vec<f64> {
        self.inner.lambdas().to_vec()
    }

    /// Effective degrees of freedom.
    #[getter]
    fn edf(&self) -> f64 {
        self.inner.edf()
    }

    /// Log-likelihood at the fitted means and the dispersion, as
    /// ``GlmFit.log_likelihood`` (the Gaussian's at the maximum-likelihood
    /// variance). NaN for a fit loaded from an older artifact.
    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood()
    }

    /// ``-2 log_likelihood + 2 edf``, not counting an estimated dispersion,
    /// as ``GlmFit.aic`` (mgcv's ``AIC`` adds 1 when the scale is
    /// estimated).
    #[getter]
    fn aic(&self) -> f64 {
        self.inner.aic()
    }

    /// Dispersion.
    #[getter]
    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    /// Residual deviance.
    #[getter]
    fn deviance(&self) -> f64 {
        self.inner.deviance()
    }

    /// The minimized GCV, UBRE or REML score (mgcv's ``gcv.ubre``).
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score()
    }

    /// The dispersion REML estimates with the smoothing parameters
    /// (mgcv's ``reml.scale``), or ``None`` for GCV and UBRE.
    #[getter]
    fn reml_scale(&self) -> Option<f64> {
        self.inner.reml_scale()
    }

    /// Fitted means on the training data.
    #[getter]
    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted().to_vec()
    }

    /// Expected response for each row.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    ///     Same columns as the training design, raw smooth columns included.
    ///
    /// Returns
    /// -------
    /// list of float
    fn predict(&self, design: PyRef<'_, PyDesign>) -> PyResult<Vec<f64>> {
        self.inner.predict(&design.inner).map_err(to_py)
    }

    /// Joint predictive distribution across the rows, keyed ``row``.
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
}

/// Deviance ``sum w d(y, mu)`` of a family.
///
/// Parameters
/// ----------
/// family : str
/// y : list of float
/// mu : list of float
/// weights : list of float, optional
/// theta : float, optional
/// power : float, optional
///
/// Returns
/// -------
/// float
#[pyfunction]
#[pyo3(signature = (family, y, mu, weights = None, theta = None, power = None))]
pub(crate) fn deviance(
    family: &str,
    y: Vec<f64>,
    mu: Vec<f64>,
    weights: Option<Vec<f64>>,
    theta: Option<f64>,
    power: Option<f64>,
) -> PyResult<f64> {
    let f = self::family(family, theta, power)?;
    metrics::deviance(f, &y, &mu, weights.as_deref()).map_err(to_py)
}

/// The F test of a GLM within a larger one that contains it, as R's
/// ``anova(small, big, test = "F")``: ``F = (D_small - D_big) / (q phi_big)``
/// on ``F(q, n - p_big)`` (CAS Monograph 5, section 6.2.1). For a family
/// whose dispersion is estimated.
///
/// The models are not checked to be nested: ``small``'s columns must lie in
/// the span of ``big``'s, on the same rows, family and weights.
///
/// Parameters
/// ----------
/// small : GlmFit
/// big : GlmFit
///
/// Returns
/// -------
/// dict
///     ``statistic``, ``df`` (``q``), ``df_resid``, ``deviance_drop`` and
///     ``p_value``.
///
/// Raises
/// ------
/// ValueError
///     If the models differ in rows or ``big`` has no more coefficients.
#[pyfunction]
pub(crate) fn f_test<'py>(
    py: Python<'py>,
    small: PyRef<'_, PyGlmFit>,
    big: PyRef<'_, PyGlmFit>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    nested_dict(py, prospicio_glm::nested::f_test(&small.inner, &big.inner))
}

/// The likelihood ratio test of a GLM within a larger one, as R's
/// ``anova(small, big, test = "Chisq")``: the scaled deviance drop
/// ``(D_small - D_big) / phi_big`` on ``chi2(q)``. For a family whose
/// dispersion is known (Poisson, binomial).
///
/// Parameters
/// ----------
/// small : GlmFit
/// big : GlmFit
///
/// Returns
/// -------
/// dict
///     As ``f_test``, with ``df_resid`` infinite.
///
/// Raises
/// ------
/// ValueError
///     As ``f_test``.
#[pyfunction]
pub(crate) fn likelihood_ratio_test<'py>(
    py: Python<'py>,
    small: PyRef<'_, PyGlmFit>,
    big: PyRef<'_, PyGlmFit>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    nested_dict(
        py,
        prospicio_glm::nested::likelihood_ratio_test(&small.inner, &big.inner),
    )
}

fn nested_dict<'py>(
    py: Python<'py>,
    test: prospicio_core::Result<prospicio_glm::nested::NestedTest>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let t = test.map_err(to_py)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("statistic", t.statistic)?;
    out.set_item("df", t.df)?;
    out.set_item("df_resid", t.df_resid)?;
    out.set_item("deviance_drop", t.deviance_drop)?;
    out.set_item("p_value", t.p_value)?;
    Ok(out)
}

/// Variance inflation factors of a design's columns, ``1 / (1 - R2_j)``
/// with ``R2_j`` that of column ``j`` on the others and an intercept (CAS
/// Monograph 5, section 2.9). A constant column (the intercept) gets
/// ``nan``.
///
/// Parameters
/// ----------
/// design : Design
///
/// Returns
/// -------
/// list of float
///     One per column, in the design's order.
///
/// Raises
/// ------
/// ValueError
///     If the non-constant columns are collinear.
///
/// Examples
/// --------
/// >>> from prospicio.models import Design, vif
/// >>> d = Design([[1.0] * 4, [1.0, 2.0, 3.0, 4.0], [1.0, -1.0, -1.0, 1.0]],
/// ...            ["(Intercept)", "x", "z"])
/// >>> [round(v, 12) for v in vif(d)[1:]]
/// [1.0, 1.0]
#[pyfunction]
pub(crate) fn vif(design: PyRef<'_, PyDesign>) -> PyResult<Vec<f64>> {
    metrics::vif(&design.inner).map_err(to_py)
}

/// Gini index of the ordered Lorenz curve.
///
/// Parameters
/// ----------
/// y : list of float
/// pred : list of float
/// exposure : list of float, optional
///
/// Returns
/// -------
/// float
#[pyfunction]
#[pyo3(signature = (y, pred, exposure = None))]
pub(crate) fn gini(y: Vec<f64>, pred: Vec<f64>, exposure: Option<Vec<f64>>) -> PyResult<f64> {
    metrics::gini(&y, &pred, exposure.as_deref()).map_err(to_py)
}

/// Lift table: rows sorted by predicted rate, cut into bands of about
/// equal exposure.
///
/// Parameters
/// ----------
/// y : list of float
/// pred : list of float
/// exposure : list of float, optional
/// bands : int, default 10
///
/// Returns
/// -------
/// list of dict
///     ``exposure``, ``expected`` and ``actual`` per band.
#[pyfunction]
#[pyo3(signature = (y, pred, exposure = None, bands = 10))]
pub(crate) fn lift(
    y: Vec<f64>,
    pred: Vec<f64>,
    exposure: Option<Vec<f64>>,
    bands: usize,
) -> PyResult<Vec<HashMap<String, f64>>> {
    let table = metrics::lift(&y, &pred, exposure.as_deref(), bands).map_err(to_py)?;
    Ok(table
        .into_iter()
        .map(|b| {
            HashMap::from([
                ("exposure".to_string(), b.exposure),
                ("expected".to_string(), b.expected),
                ("actual".to_string(), b.actual),
            ])
        })
        .collect())
}

/// Weighted mean pinball (quantile) loss of predictions of the ``alpha``
/// quantile: ``sum(w * rho(y - q)) / sum(w)`` with
/// ``rho(u) = u * (alpha - (u < 0))``. Lowest in expectation at the true
/// ``alpha`` quantile; lower is better.
///
/// Parameters
/// ----------
/// y : list of float
/// pred : list of float
///     Predicted ``alpha`` quantiles.
/// alpha : float
///     In ``(0, 1)``.
/// weights : list of float, optional
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.models import pinball_loss
/// >>> pinball_loss([1.0, 0.0], [0.0, 1.0], 0.9)
/// 0.5
#[pyfunction]
#[pyo3(signature = (y, pred, alpha, weights = None))]
pub(crate) fn pinball_loss(
    y: Vec<f64>,
    pred: Vec<f64>,
    alpha: f64,
    weights: Option<Vec<f64>>,
) -> PyResult<f64> {
    metrics::pinball(&y, &pred, alpha, weights.as_deref()).map_err(to_py)
}

/// Continuous ranked probability score of equally likely draws for an
/// outcome; lower is better.
///
/// Parameters
/// ----------
/// draws : list of float
/// y : float
///
/// Returns
/// -------
/// float
#[pyfunction]
pub(crate) fn crps(draws: Vec<f64>, y: f64) -> PyResult<f64> {
    metrics::crps(&draws, y).map_err(to_py)
}

/// Mean log score ``-(1/n) sum log f(y_i)`` of the outcomes under the
/// family's predictive distribution; lower is better.
///
/// Parameters
/// ----------
/// family : str
/// y : list of float
/// mu : list of float
/// dispersion : float, default 1.0
/// weights : list of float, optional
/// theta : float, optional
/// power : float, optional
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.models import log_score
/// >>> round(log_score("poisson", [0.0], [1.0]), 12)
/// 1.0
#[pyfunction]
#[pyo3(signature = (family, y, mu, dispersion = 1.0, weights = None, theta = None, power = None))]
pub(crate) fn log_score(
    family: &str,
    y: Vec<f64>,
    mu: Vec<f64>,
    dispersion: f64,
    weights: Option<Vec<f64>>,
    theta: Option<f64>,
    power: Option<f64>,
) -> PyResult<f64> {
    let f = self::family(family, theta, power)?;
    metrics::log_score(f, &y, &mu, dispersion, weights.as_deref()).map_err(to_py)
}

/// Probability integral transform of each outcome under the family's
/// predictive distribution, randomized where it has atoms (counts, a
/// Tweedie's zero); uniform when the model is calibrated.
///
/// Parameters
/// ----------
/// family : str
/// y : list of float
/// mu : list of float
/// dispersion : float, default 1.0
/// weights : list of float, optional
/// seed : int, default 0
///     Seeds the randomization.
/// theta : float, optional
/// power : float, optional
///
/// Returns
/// -------
/// list of float
#[pyfunction]
#[pyo3(signature = (family, y, mu, dispersion = 1.0, weights = None, seed = 0, theta = None, power = None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pit(
    family: &str,
    y: Vec<f64>,
    mu: Vec<f64>,
    dispersion: f64,
    weights: Option<Vec<f64>>,
    seed: u64,
    theta: Option<f64>,
    power: Option<f64>,
) -> PyResult<Vec<f64>> {
    let f = self::family(family, theta, power)?;
    metrics::pit(f, &y, &mu, dispersion, weights.as_deref(), seed).map_err(to_py)
}

/// The PIT of ``y`` under the empirical distribution of ``draws``,
/// randomized over ties by ``u``.
///
/// Parameters
/// ----------
/// draws : list of float
/// y : float
/// u : float, default 0.5
///
/// Returns
/// -------
/// float
#[pyfunction]
#[pyo3(signature = (draws, y, u = 0.5))]
pub(crate) fn pit_from_draws(draws: Vec<f64>, y: f64, u: f64) -> PyResult<f64> {
    metrics::pit_from_draws(&draws, y, u).map_err(to_py)
}

/// Counts of PIT values in ``bins`` equal-width bins of ``[0, 1]``.
///
/// Parameters
/// ----------
/// pit : list of float
/// bins : int, default 10
///
/// Returns
/// -------
/// list of int
#[pyfunction]
#[pyo3(signature = (pit, bins = 10))]
pub(crate) fn pit_histogram(pit: Vec<f64>, bins: usize) -> PyResult<Vec<usize>> {
    metrics::pit_histogram(&pit, bins).map_err(to_py)
}

/// Kolmogorov-Smirnov distance from the uniform on ``[0, 1]``; about
/// ``1.36 / sqrt(n)`` or less 95% of the time under uniformity.
///
/// Parameters
/// ----------
/// values : list of float
///
/// Returns
/// -------
/// float
#[pyfunction]
pub(crate) fn ks_uniform(values: Vec<f64>) -> PyResult<f64> {
    metrics::ks_uniform(&values).map_err(to_py)
}

fn splits(s: Vec<Split>) -> Vec<(Vec<usize>, Vec<usize>)> {
    s.into_iter().map(|s| (s.train, s.test)).collect()
}

/// ``k``-fold splits of ``n`` rows, shuffled with ``seed``.
///
/// Parameters
/// ----------
/// n : int
/// k : int
/// seed : int
///
/// Returns
/// -------
/// list of (list of int, list of int)
///     ``(train, test)`` row indices per fold.
#[pyfunction]
pub(crate) fn k_fold(n: usize, k: usize, seed: u64) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
    resample::k_fold(n, k, seed).map(splits).map_err(to_py)
}

/// Grouped ``k``-fold splits: each group's rows stay in one fold.
///
/// Parameters
/// ----------
/// groups : list of str
/// k : int
/// seed : int
///
/// Returns
/// -------
/// list of (list of int, list of int)
#[pyfunction]
pub(crate) fn group_k_fold(
    groups: Vec<String>,
    k: usize,
    seed: u64,
) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
    resample::group_k_fold(&groups, k, seed)
        .map(splits)
        .map_err(to_py)
}

/// Time-ordered splits: for each of the last ``n_test`` periods, train on
/// earlier periods and test on that one (for a triangle, the calendar
/// diagonal backtest).
///
/// Parameters
/// ----------
/// periods : list of int
/// n_test : int
///
/// Returns
/// -------
/// list of (list of int, list of int)
#[pyfunction]
pub(crate) fn time_ordered(
    periods: Vec<i64>,
    n_test: usize,
) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
    resample::time_ordered(&periods, n_test)
        .map(splits)
        .map_err(to_py)
}

/// An ELPD estimate from ``elpd_loo`` or ``elpd_waic``.
#[pyclass(name = "Elpd", module = "prospicio.models", frozen)]
pub(crate) struct PyElpd {
    inner: prospicio_bayes::elpd::Elpd,
    pareto_k: Option<Vec<f64>>,
    k_threshold: Option<f64>,
}

impl PyElpd {
    /// A PSIS-LOO result, as ``BayesGlmFit.loo`` returns it; also used by
    /// the reserving lane's ``BayesOdpFit.loo``.
    pub(crate) fn from_loo(l: prospicio_bayes::elpd::Loo) -> Self {
        Self {
            inner: l.estimate,
            pareto_k: Some(l.pareto_k),
            k_threshold: Some(l.k_threshold),
        }
    }
}

#[pymethods]
impl PyElpd {
    /// Expected log pointwise predictive density, summed.
    #[getter]
    fn elpd(&self) -> f64 {
        self.inner.elpd
    }

    /// Its standard error, ``sqrt(N var(pointwise))``.
    #[getter]
    fn se(&self) -> f64 {
        self.inner.se
    }

    /// Effective number of parameters, ``lppd - elpd``.
    #[getter]
    fn p(&self) -> f64 {
        self.inner.p
    }

    /// The information criterion, ``-2 elpd`` (LOOIC or WAIC).
    #[getter]
    fn ic(&self) -> f64 {
        self.inner.ic
    }

    /// ELPD per observation.
    #[getter]
    fn pointwise(&self) -> Vec<f64> {
        self.inner.pointwise.clone()
    }

    /// PSIS-LOO only: the fitted Pareto shape per observation.
    #[getter]
    fn pareto_k(&self) -> Option<Vec<f64>> {
        self.pareto_k.clone()
    }

    /// PSIS-LOO only: ``min(1 - 1/log10(S), 0.7)``; observations with a
    /// larger ``pareto_k`` are unreliable.
    #[getter]
    fn k_threshold(&self) -> Option<f64> {
        self.k_threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "Elpd(elpd={:.4}, se={:.4}, p={:.4}, ic={:.4})",
            self.inner.elpd, self.inner.se, self.inner.p, self.inner.ic
        )
    }
}

fn flatten_draws(log_lik: &[Vec<f64>]) -> PyResult<(Vec<f64>, usize)> {
    let n = log_lik.first().map_or(0, Vec::len);
    if log_lik.iter().any(|r| r.len() != n) {
        return Err(PyValueError::new_err(
            "log_lik must be draws × observations with equal-length rows",
        ));
    }
    Ok((log_lik.concat(), n))
}

/// Leave-one-out cross-validation by Pareto-smoothed importance sampling
/// (PSIS-LOO), from one fit's pointwise log-likelihood draws. Matches the
/// R package ``loo``.
///
/// Parameters
/// ----------
/// log_lik : list of list of float
///     One row per posterior draw, one column per observation:
///     ``log p(y_i | theta_s)``.
/// r_eff : list of float, optional
///     Relative efficiency of the draws per observation (1 for
///     independent draws).
///
/// Returns
/// -------
/// Elpd
///     With ``pareto_k`` and ``k_threshold``.
#[pyfunction]
#[pyo3(signature = (log_lik, r_eff = None))]
pub(crate) fn elpd_loo(log_lik: Vec<Vec<f64>>, r_eff: Option<Vec<f64>>) -> PyResult<PyElpd> {
    let (flat, n) = flatten_draws(&log_lik)?;
    let l = prospicio_bayes::elpd::loo(&flat, n, r_eff.as_deref()).map_err(to_py)?;
    Ok(PyElpd {
        inner: l.estimate,
        pareto_k: Some(l.pareto_k),
        k_threshold: Some(l.k_threshold),
    })
}

/// WAIC from pointwise log-likelihood draws: ``lppd`` less the variance of
/// each observation's log-likelihood.
///
/// Parameters
/// ----------
/// log_lik : list of list of float
///     One row per posterior draw, one column per observation.
///
/// Returns
/// -------
/// Elpd
#[pyfunction]
pub(crate) fn elpd_waic(log_lik: Vec<Vec<f64>>) -> PyResult<PyElpd> {
    let (flat, n) = flatten_draws(&log_lik)?;
    let inner = prospicio_bayes::elpd::waic(&flat, n).map_err(to_py)?;
    Ok(PyElpd {
        inner,
        pareto_k: None,
        k_threshold: None,
    })
}

/// In-sample log pointwise predictive density,
/// ``sum_i log(mean_s p(y_i | theta_s))``.
///
/// Parameters
/// ----------
/// log_lik : list of list of float
///     One row per posterior draw, one column per observation.
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> import math
/// >>> from prospicio.models import lppd
/// >>> round(lppd([[math.log(0.5)], [math.log(0.25)]]), 12) == round(math.log(0.375), 12)
/// True
#[pyfunction]
pub(crate) fn lppd(log_lik: Vec<Vec<f64>>) -> PyResult<f64> {
    let (flat, n) = flatten_draws(&log_lik)?;
    prospicio_bayes::elpd::lppd(&flat, n).map_err(to_py)
}

/// MCMC diagnostics of chains of draws (Vehtari et al. 2021, as R's
/// ``posterior``): rank-normalized split R-hat, bulk and tail effective
/// sample sizes, the effective sample size of the mean and its Monte Carlo
/// standard error.
///
/// Parameters
/// ----------
/// chains : list of list of float
///     Equal-length chains, at least 4 draws each.
///
/// Returns
/// -------
/// dict
///     ``rhat``, ``ess_bulk``, ``ess_tail``, ``ess_mean``, ``mcse_mean``.
///
/// Examples
/// --------
/// >>> from prospicio.models import mcmc_diagnostics
/// >>> a = [float((i * 37) % 101) for i in range(400)]
/// >>> b = [float((i * 53 + 7) % 101) for i in range(400)]
/// >>> mcmc_diagnostics([a, b])["rhat"] < 1.01
/// True
#[pyfunction]
pub(crate) fn mcmc_diagnostics(chains: Vec<Vec<f64>>) -> PyResult<HashMap<String, f64>> {
    let refs: Vec<&[f64]> = chains.iter().map(Vec::as_slice).collect();
    Ok(HashMap::from([
        (
            "rhat".to_string(),
            prospicio_bayes::rhat(&refs).map_err(to_py)?,
        ),
        (
            "ess_bulk".to_string(),
            prospicio_bayes::ess_bulk(&refs).map_err(to_py)?,
        ),
        (
            "ess_tail".to_string(),
            prospicio_bayes::ess_tail(&refs).map_err(to_py)?,
        ),
        (
            "ess_mean".to_string(),
            prospicio_bayes::ess_mean(&refs).map_err(to_py)?,
        ),
        (
            "mcse_mean".to_string(),
            prospicio_bayes::mcse_mean(&refs).map_err(to_py)?,
        ),
    ]))
}

/// Stacking weights from pointwise held-out log predictive densities
/// (Yao et al., 2018): the weights on the simplex that maximize the log
/// score of the mixture of the models' predictive distributions.
///
/// Works for any model: pass PSIS-LOO pointwise values (``Elpd.pointwise``)
/// for a Bayesian fit, or cross-validated log densities for any other. A
/// model that adds nothing gets weight exactly 0.
///
/// Parameters
/// ----------
/// lpd : list of list of float
///     One list per model, each with one log density per observation.
///
/// Returns
/// -------
/// list of float
///     One weight per model, summing to 1.
///
/// Examples
/// --------
/// >>> from prospicio.models import stacking_weights
/// >>> w = stacking_weights([[-0.1, -0.1, -3.0, -3.0], [-3.0, -3.0, -0.1, -0.1]])
/// >>> [round(x, 9) for x in w]
/// [0.5, 0.5]
#[pyfunction]
pub(crate) fn stacking_weights(py: Python<'_>, lpd: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    py.detach(|| prospicio_models::stack::stacking_weights(&lpd))
        .map_err(to_py)
}

/// Pseudo-BMA weights, ``w_k`` proportional to ``exp(elpd_k)``; with
/// ``bootstrap=True``, pseudo-BMA+ weights averaged over Bayesian-bootstrap
/// replicates of the observations, which keeps a model that is only
/// slightly better from taking all the weight.
///
/// Parameters
/// ----------
/// lpd : list of list of float
///     One list per model, as for ``stacking_weights``.
/// bootstrap : bool, default True
/// n_draws : int, default 1000
///     Bootstrap replicates.
/// seed : int, default 0
///     Replicate ``b`` uses stream ``b`` of ``seed``.
///
/// Returns
/// -------
/// list of float
#[pyfunction]
#[pyo3(signature = (lpd, bootstrap = true, n_draws = 1000, seed = 0))]
pub(crate) fn pseudo_bma_weights(
    py: Python<'_>,
    lpd: Vec<Vec<f64>>,
    bootstrap: bool,
    n_draws: usize,
    seed: u64,
) -> PyResult<Vec<f64>> {
    let bb = bootstrap.then_some((n_draws, seed));
    py.detach(|| prospicio_models::stack::pseudo_bma_weights(&lpd, bb))
        .map_err(to_py)
}

/// Actual against expected by period for a stored model's predictions on
/// new data, with each period's z-score under the model and a test for
/// drift.
///
/// ``A = sum(w * y)`` and ``E = sum(w * mu)`` per period; the z-score is
/// ``(A - E) / sqrt(dispersion * sum(w * V(mu)))`` with the family's
/// variance function ``V``, about standard normal while the model holds.
/// ``trend`` is the slope of ``A / E - 1`` per period step (periods in
/// sorted order), weighted by each period's precision.
///
/// Parameters
/// ----------
/// periods : list of int or str
///     One period label per row.
/// family : str
/// y : list of float
///     Actuals.
/// mu : list of float
///     The model's predicted means.
/// weights : list of float, optional
///     Prior weights as fitted (exposure for a rate); leave out for counts
///     with exposure in the offset.
/// dispersion : float, default 1.0
/// theta, power : float, optional
///     Negative binomial ``theta``, Tweedie ``power``.
///
/// Returns
/// -------
/// dict
///     ``periods`` (a list of dicts with ``period``, ``n``, ``weight``,
///     ``actual``, ``expected``, ``ratio``, ``std_dev`` and ``z``),
///     ``total`` (the same without ``period``), ``trend``,
///     ``trend_std_error`` and ``trend_z``.
///
/// Examples
/// --------
/// >>> from prospicio.models import actual_vs_expected
/// >>> m = actual_vs_expected([2023, 2023, 2024, 2024], "poisson",
/// ...                        [1.0, 3.0, 2.0, 6.0], [2.0, 2.0, 2.0, 2.0])
/// >>> m["periods"][1]["ratio"], m["periods"][1]["z"]
/// (2.0, 2.0)
#[pyfunction]
#[pyo3(signature = (periods, family, y, mu, weights = None, dispersion = 1.0, theta = None, power = None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn actual_vs_expected<'py>(
    py: Python<'py>,
    periods: Vec<KeyArg>,
    family: &str,
    y: Vec<f64>,
    mu: Vec<f64>,
    weights: Option<Vec<f64>>,
    dispersion: f64,
    theta: Option<f64>,
    power: Option<f64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use pyo3::types::PyDict;
    let f = self::family(family, theta, power)?;
    let keys = key_from_py(periods);
    let m = prospicio_models::monitor::actual_vs_expected(
        &keys,
        &y,
        &mu,
        weights.as_deref(),
        f,
        dispersion,
    )
    .map_err(to_py)?;
    let row = |r: &prospicio_models::monitor::PeriodSummary<prospicio_prob::KeyValue>| -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = PyDict::new(py);
        if let Some(p) = &r.period {
            match p {
                prospicio_prob::KeyValue::Int(i) => d.set_item("period", i)?,
                other => d.set_item("period", other.to_string())?,
            }
        }
        d.set_item("n", r.n)?;
        d.set_item("weight", r.weight)?;
        d.set_item("actual", r.actual)?;
        d.set_item("expected", r.expected)?;
        d.set_item("ratio", r.ratio())?;
        d.set_item("std_dev", r.std_dev)?;
        d.set_item("z", r.z())?;
        Ok(d)
    };
    let out = PyDict::new(py);
    let rows = m.periods.iter().map(&row).collect::<PyResult<Vec<_>>>()?;
    out.set_item("periods", rows)?;
    out.set_item("total", row(&m.total)?)?;
    out.set_item("trend", m.trend)?;
    out.set_item("trend_std_error", m.trend_std_error)?;
    out.set_item("trend_z", m.trend_z())?;
    Ok(out)
}

/// A Bayesian GLM sampled with NUTS (nuts-rs, the Rust core of nutpie).
///
/// Normal priors with mean 0 on the coefficients: standard deviation
/// ``intercept_sd`` for an all-ones column, ``prior_sd`` for the others
/// (on the link scale; standardize covariates). For the Gaussian, gamma
/// and inverse Gaussian the dispersion is sampled too, with a half-normal
/// prior of scale ``dispersion_scale``, unless ``dispersion`` fixes it.
/// Chains run in parallel, start near the maximum-likelihood fit, and
/// replay exactly from ``seed``.
///
/// Parameters
/// ----------
/// family : str
/// link : str, optional
/// prior_sd : float, default 2.5
/// intercept_sd : float, default 10.0
/// dispersion : float, optional
///     A fixed dispersion; 1 by default for the Poisson, binomial and
///     negative binomial. A Tweedie needs one.
/// dispersion_scale : float, default 10.0
/// chains, tune, draws : int, default 4, 1000, 1000
/// seed : int, default 0
/// target_accept : float, default 0.8
/// max_depth : int, default 10
/// theta, power, link_power : float, optional
///
/// Examples
/// --------
/// >>> from prospicio.models import BayesGlm, Design
/// >>> x = [(i % 4) - 1.5 for i in range(40)]
/// >>> y = [[1.0, 2.0, 3.0, 5.0][i % 4] for i in range(40)]
/// >>> d = Design([[1.0] * 40, x], ["(Intercept)", "x"])
/// >>> fit = BayesGlm("poisson", chains=2, tune=300, draws=300).fit(d, y)
/// >>> all(s["rhat"] < 1.05 for s in fit.summary())
/// True
#[pyclass(name = "BayesGlm", module = "prospicio.models", frozen)]
pub(crate) struct PyBayesGlm {
    inner: prospicio_bayes::glm::BayesGlm,
}

#[pymethods]
impl PyBayesGlm {
    #[new]
    #[pyo3(signature = (family, link = None, prior_sd = 2.5, intercept_sd = 10.0, dispersion = None, dispersion_scale = 10.0, chains = 4, tune = 1000, draws = 1000, seed = 0, target_accept = 0.8, max_depth = 10, theta = None, power = None, link_power = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        family: &str,
        link: Option<&str>,
        prior_sd: f64,
        intercept_sd: f64,
        dispersion: Option<f64>,
        dispersion_scale: f64,
        chains: usize,
        tune: usize,
        draws: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        theta: Option<f64>,
        power: Option<f64>,
        link_power: Option<f64>,
    ) -> PyResult<Self> {
        use prospicio_bayes::glm::{BayesGlm, DispersionPrior, Sampler};
        let f = self::family(family, theta, power)?;
        let l = self::link(link, f, link_power)?;
        let mut inner = BayesGlm::new(f, l).sampler(Sampler {
            chains,
            tune,
            draws,
            seed,
            target_accept,
            max_depth,
        });
        inner.prior_sd = prior_sd;
        inner.intercept_sd = intercept_sd;
        inner.dispersion = match (dispersion, inner.dispersion) {
            (Some(v), _) => DispersionPrior::Fixed(v),
            (None, DispersionPrior::HalfNormal(_)) => DispersionPrior::HalfNormal(dispersion_scale),
            (None, fixed) => fixed,
        };
        Ok(Self { inner })
    }

    /// Samples the posterior.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// y : list of float
    ///
    /// Returns
    /// -------
    /// BayesGlmFit
    fn fit(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
    ) -> PyResult<PyBayesGlmFit> {
        let (spec, d) = (&self.inner, &design.inner);
        let inner = py.detach(|| spec.fit(d, &y)).map_err(to_py)?;
        Ok(PyBayesGlmFit { inner })
    }
}

/// A sampled Bayesian GLM, from ``BayesGlm.fit``.
#[pyclass(name = "BayesGlmFit", module = "prospicio.models", frozen)]
pub(crate) struct PyBayesGlmFit {
    inner: prospicio_bayes::glm::BayesGlmFit,
}

#[pymethods]
impl PyBayesGlmFit {
    /// Coefficient names.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    /// Posterior means of the coefficients.
    #[getter]
    fn posterior_mean(&self) -> Vec<f64> {
        self.inner.posterior_mean()
    }

    /// Coefficient draws, one row per draw (chain by chain).
    #[getter]
    fn coefficient_draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.names().len();
        self.inner
            .coefficient_draws()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Dispersion draws, one per draw (constant when fixed).
    #[getter]
    fn dispersion_draws(&self) -> Vec<f64> {
        self.inner.dispersion_draws().to_vec()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.chains()
    }

    /// Divergent transitions among the kept draws.
    #[getter]
    fn divergences(&self) -> usize {
        self.inner.divergences()
    }

    /// Posterior summary: one dict per parameter with ``name``, ``mean``,
    /// ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
    /// ``ess_tail``; the dispersion last when it was sampled.
    ///
    /// Returns
    /// -------
    /// list of dict
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
        self.inner
            .summary()
            .map_err(to_py)?
            .into_iter()
            .map(|s| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("name", s.name)?;
                d.set_item("mean", s.mean)?;
                d.set_item("sd", s.sd)?;
                d.set_item("q05", s.q05)?;
                d.set_item("q50", s.q50)?;
                d.set_item("q95", s.q95)?;
                d.set_item("rhat", s.rhat)?;
                d.set_item("ess_bulk", s.ess_bulk)?;
                d.set_item("ess_tail", s.ess_tail)?;
                Ok(d)
            })
            .collect()
    }

    /// Pointwise log-likelihood of ``y`` given ``design``: one row per draw,
    /// one column per observation, for ``elpd_loo`` or ``elpd_waic``.
    ///
    /// Returns
    /// -------
    /// list of list of float
    fn log_likelihood(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (fit, d) = (&self.inner, &design.inner);
        let n = y.len().max(1);
        let ll = py.detach(|| fit.log_likelihood(d, &y)).map_err(to_py)?;
        Ok(ll.chunks(n).map(<[f64]>::to_vec).collect())
    }

    /// PSIS-LOO of ``y`` given ``design``, with each observation's relative
    /// efficiency estimated from the chains.
    ///
    /// Returns
    /// -------
    /// Elpd
    fn loo(&self, py: Python<'_>, design: PyRef<'_, PyDesign>, y: Vec<f64>) -> PyResult<PyElpd> {
        let (fit, d) = (&self.inner, &design.inner);
        let l = py.detach(|| fit.loo(d, &y)).map_err(to_py)?;
        Ok(PyElpd::from_loo(l))
    }

    /// Posterior mean of each row's mean.
    ///
    /// Returns
    /// -------
    /// list of float
    fn predict(&self, design: PyRef<'_, PyDesign>) -> PyResult<Vec<f64>> {
        self.inner.predict(&design.inner).map_err(to_py)
    }

    /// Posterior predictive draws across the rows, keyed ``row = 0, 1, ...``.
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

    /// One replicated dataset per posterior draw: one row per draw, one
    /// column per row of ``design``, each response drawn from the family at
    /// that draw's coefficients and dispersion.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// seed : int, default 0
    ///
    /// Returns
    /// -------
    /// list of list of float
    #[pyo3(signature = (design, seed = 0))]
    fn replicate(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        seed: u64,
    ) -> PyResult<Vec<Vec<f64>>> {
        let (fit, d) = (&self.inner, &design.inner);
        let n = d.n_rows().max(1);
        let reps = py.detach(|| fit.replicate(d, seed)).map_err(to_py)?;
        Ok(reps.chunks(n).map(<[f64]>::to_vec).collect())
    }

    /// Posterior predictive checks of ``y`` given ``design``: each test
    /// statistic of ``y`` against its distribution over ``replicate``'s
    /// datasets, with the p-value ``P(T(y_rep) >= T(y))`` (ties count half).
    /// A p-value near 0 or 1 says the model cannot reproduce that feature.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// y : list of float
    /// statistics : list of str or float, default ["mean", "sd", "min", "max"]
    ///     ``"mean"``, ``"sd"``, ``"min"``, ``"max"``, ``"proportion_zero"``,
    ///     or a probability for that quantile.
    /// seed : int, default 0
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     ``name``, ``observed``, ``replicated`` and ``p_value`` per statistic.
    #[pyo3(signature = (design, y, statistics = None, seed = 0))]
    fn posterior_predictive_check<'py>(
        &self,
        py: Python<'py>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
        statistics: Option<Vec<Bound<'py, PyAny>>>,
        seed: u64,
    ) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
        let stats = ppc_statistics(statistics)?;
        let (fit, d) = (&self.inner, &design.inner);
        let checks = py
            .detach(|| fit.posterior_predictive_check(d, &y, &stats, seed))
            .map_err(to_py)?;
        checks_to_py(py, checks)
    }

    /// The leave-one-out PIT of ``y`` given ``design``, by Pareto-smoothed
    /// importance weights on ``replicate``'s datasets, randomized over ties;
    /// uniform for a calibrated model.
    ///
    /// Parameters
    /// ----------
    /// design : Design
    /// y : list of float
    /// seed : int, default 0
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``pit`` and ``pareto_k`` per observation, and ``ks``, the
    ///     Kolmogorov–Smirnov distance from the uniform.
    #[pyo3(signature = (design, y, seed = 0))]
    fn loo_pit<'py>(
        &self,
        py: Python<'py>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
        seed: u64,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let (fit, d) = (&self.inner, &design.inner);
        let p = py.detach(|| fit.loo_pit(d, &y, seed)).map_err(to_py)?;
        loo_pit_to_py(py, p)
    }
}

/// Test statistics from Python: names, or probabilities for quantiles.
fn ppc_statistics(statistics: Option<Vec<Bound<'_, PyAny>>>) -> PyResult<Vec<Statistic>> {
    let Some(list) = statistics else {
        return Ok(vec![
            Statistic::Mean,
            Statistic::Sd,
            Statistic::Min,
            Statistic::Max,
        ]);
    };
    list.iter()
        .map(|v| {
            if let Ok(p) = v.extract::<f64>() {
                return Ok(Statistic::Quantile(p));
            }
            let name: String = v.extract()?;
            match name.as_str() {
                "mean" => Ok(Statistic::Mean),
                "sd" => Ok(Statistic::Sd),
                "min" => Ok(Statistic::Min),
                "max" => Ok(Statistic::Max),
                "proportion_zero" => Ok(Statistic::ProportionZero),
                other => Err(PyValueError::new_err(format!(
                    "a statistic is \"mean\", \"sd\", \"min\", \"max\", \"proportion_zero\" or a quantile's probability, got {other:?}"
                ))),
            }
        })
        .collect()
}

fn checks_to_py(
    py: Python<'_>,
    checks: Vec<StatisticCheck>,
) -> PyResult<Vec<Bound<'_, pyo3::types::PyDict>>> {
    checks
        .into_iter()
        .map(|c| {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("name", c.name)?;
            d.set_item("observed", c.observed)?;
            d.set_item("replicated", c.replicated)?;
            d.set_item("p_value", c.p_value)?;
            Ok(d)
        })
        .collect()
}

fn loo_pit_to_py(py: Python<'_>, p: LooPit) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("pit", p.pit)?;
    d.set_item("pareto_k", p.pareto_k)?;
    d.set_item("ks", p.ks)?;
    Ok(d)
}

/// Rows of a list of equal-length lists, flattened row-major.
fn flatten_rows(rows: &[Vec<f64>], n: usize, what: &str) -> PyResult<Vec<f64>> {
    if rows.iter().any(|r| r.len() != n) {
        return Err(PyValueError::new_err(format!(
            "every row of {what} needs {n} values, one per observation"
        )));
    }
    Ok(rows.concat())
}

/// Posterior predictive checks of any model: each test statistic of
/// ``observed`` against its distribution over the replicated datasets,
/// with the p-value ``P(T(y_rep) >= T(y))`` (ties count half). With
/// ``groups``, each statistic is checked within each group.
///
/// Parameters
/// ----------
/// observed : list of float
/// replicated : list of list of float
///     One replicated dataset per posterior draw, each as long as
///     ``observed``.
/// statistics : list of str or float, default ["mean", "sd", "min", "max"]
///     ``"mean"``, ``"sd"``, ``"min"``, ``"max"``, ``"proportion_zero"``, or a
///     probability for that quantile.
/// groups : list of int, optional
///     Each observation's group, from 0.
///
/// Returns
/// -------
/// list of dict
///     ``name``, ``observed``, ``replicated`` and ``p_value``.
///
/// Examples
/// --------
/// >>> from prospicio.models import ppc_check
/// >>> c = ppc_check([1.0, 3.0], [[0.0, 2.0], [2.0, 4.0], [1.0, 3.0]], ["mean"])
/// >>> c[0]["observed"], c[0]["p_value"]
/// (2.0, 0.5)
#[pyfunction]
#[pyo3(signature = (observed, replicated, statistics = None, groups = None))]
pub(crate) fn ppc_check<'py>(
    py: Python<'py>,
    observed: Vec<f64>,
    replicated: Vec<Vec<f64>>,
    statistics: Option<Vec<Bound<'py, PyAny>>>,
    groups: Option<Vec<usize>>,
) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
    let stats = ppc_statistics(statistics)?;
    let reps = flatten_rows(&replicated, observed.len(), "replicated")?;
    let checks = match groups {
        Some(g) => ppc::check_grouped(&observed, &reps, &g, &stats),
        None => ppc::check(&observed, &reps, &stats),
    }
    .map_err(to_py)?;
    checks_to_py(py, checks)
}

/// The leave-one-out PIT of any model's observations: each observation's
/// place among its replicates under the Pareto-smoothed importance weights
/// for leaving it out, randomized over ties; uniform for a calibrated
/// model. Matches R ``loo``'s ``E_loo`` of ``y_rep <= y``.
///
/// Parameters
/// ----------
/// observed : list of float
/// replicated : list of list of float
///     One replicated dataset per posterior draw.
/// log_lik : list of list of float
///     The pointwise log-likelihood, one row per draw (as ``elpd_loo``).
/// r_eff : list of float, optional
///     Each observation's relative efficiency; 1 if omitted.
/// seed : int, default 0
///     Seeds the randomization over ties.
///
/// Returns
/// -------
/// dict
///     ``pit`` and ``pareto_k`` per observation, and ``ks``.
///
/// Examples
/// --------
/// >>> from prospicio.models import loo_pit
/// >>> loo_pit([2.5], [[1.0], [2.0], [3.0], [4.0]], [[-1.0]] * 4)["pit"]
/// [0.5]
#[pyfunction]
#[pyo3(signature = (observed, replicated, log_lik, r_eff = None, seed = 0))]
pub(crate) fn loo_pit<'py>(
    py: Python<'py>,
    observed: Vec<f64>,
    replicated: Vec<Vec<f64>>,
    log_lik: Vec<Vec<f64>>,
    r_eff: Option<Vec<f64>>,
    seed: u64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let n = observed.len();
    let reps = flatten_rows(&replicated, n, "replicated")?;
    let ll = flatten_rows(&log_lik, n, "log_lik")?;
    let p = ppc::loo_pit(&observed, &reps, &ll, r_eff.as_deref(), seed).map_err(to_py)?;
    loo_pit_to_py(py, p)
}

fn stacking_sampler(
    chains: usize,
    tune: usize,
    draws: usize,
    seed: u64,
) -> prospicio_bayes::glm::Sampler {
    prospicio_bayes::glm::Sampler {
        chains,
        tune,
        draws,
        seed,
        ..prospicio_bayes::glm::Sampler::default()
    }
}

/// Bayesian stacking: a posterior for the stacking weights, with a
/// Dirichlet prior, sampled by NUTS from pointwise held-out log densities
/// (Yao et al., 2018). ``stacking_weights`` gives the optimum alone.
///
/// Parameters
/// ----------
/// concentration : list of float, optional
///     Dirichlet concentration, one per model (default 1, uniform).
/// chains, tune, draws : int, default 4, 1000, 1000
/// seed : int, default 0
#[pyclass(name = "BayesStacking", module = "prospicio.models", frozen)]
pub(crate) struct PyBayesStacking {
    inner: prospicio_bayes::stacking::BayesStacking,
}

#[pymethods]
impl PyBayesStacking {
    #[new]
    #[pyo3(signature = (concentration = None, chains = 4, tune = 1000, draws = 1000, seed = 0))]
    fn new(
        concentration: Option<Vec<f64>>,
        chains: usize,
        tune: usize,
        draws: usize,
        seed: u64,
    ) -> Self {
        Self {
            inner: prospicio_bayes::stacking::BayesStacking {
                concentration,
                sampler: stacking_sampler(chains, tune, draws, seed),
            },
        }
    }

    /// Samples the weights.
    ///
    /// Parameters
    /// ----------
    /// lpd : list of list of float
    ///     One list per model, one held-out log density per observation.
    ///
    /// Returns
    /// -------
    /// StackingFit
    fn fit(&self, py: Python<'_>, lpd: Vec<Vec<f64>>) -> PyResult<PyStackingFit> {
        let spec = &self.inner;
        let inner = py.detach(|| spec.fit(&lpd)).map_err(to_py)?;
        Ok(PyStackingFit { inner })
    }
}

/// Hierarchical stacking (Yao, Pirš, Vehtari and Gelman, 2022): model
/// weights that vary with covariates, ``w = softmax(alpha + B x)`` against
/// the last model as reference, so a model can be trusted in one part of
/// the portfolio and not another. The priors are those of BayesBlend's
/// ``HierarchicalBayesStacking``; sampled by NUTS.
///
/// Scale continuous covariates (BayesBlend divides by twice the standard
/// deviation) and dummy-code discrete ones before fitting, with the
/// dummies first.
///
/// With ``partial_pooling``, each model's slopes on the discrete
/// covariates, and separately on the continuous ones, are drawn around a
/// model-level mean, itself drawn around a global mean. A scale of 0
/// removes a level: ``tau_mu_global=0`` fixes the global mean at 0,
/// ``tau_mu_*=0`` pools completely and ``tau_sigma_*=0`` sets every slope
/// to its model's mean. BayesBlend warns that pooling needs at least three
/// covariates. ``adaptive`` multiplies the prior scales by ``N**lambda``
/// with ``lambda ~ Exponential(adaptive)``, weakening them as the data
/// grow.
///
/// Parameters
/// ----------
/// discrete : int, default 0
///     Number of leading covariates that are dummy codes.
/// alpha_loc, alpha_scale : float, default 0.0, 1.0
/// beta_loc, beta_scale : float, default 0.0, 1.0
///     Slope prior without pooling.
/// partial_pooling : bool, default False
/// tau_mu_global, tau_mu_discrete, tau_mu_continuous : float, default 1.0
/// tau_sigma_discrete, tau_sigma_continuous : float, default 1.0
///     Pooling scales, BayesBlend's defaults.
/// adaptive : float, optional
///     Rate of the exponential prior on ``lambda`` (BayesBlend uses 4).
/// chains, tune, draws : int, default 4, 1000, 1000
/// seed : int, default 0
///
/// Examples
/// --------
/// >>> from prospicio.models import HierarchicalStacking
/// >>> x = [i / 99 - 0.5 for i in range(100)]
/// >>> a = [-0.5 if v < 0 else -2.0 for v in x]
/// >>> b = [-2.0 if v < 0 else -0.5 for v in x]
/// >>> fit = HierarchicalStacking(chains=2, tune=300, draws=300).fit([a, b], [x])
/// >>> w = fit.weights([[-0.4, 0.4]])
/// >>> w[0][0] > 0.7 and w[1][0] < 0.3
/// True
#[pyclass(name = "HierarchicalStacking", module = "prospicio.models", frozen)]
pub(crate) struct PyHierarchicalStacking {
    inner: prospicio_bayes::stacking::HierarchicalStacking,
}

#[pymethods]
impl PyHierarchicalStacking {
    #[new]
    #[pyo3(signature = (discrete = 0, alpha_loc = 0.0, alpha_scale = 1.0, beta_loc = 0.0, beta_scale = 1.0, partial_pooling = false, tau_mu_global = 1.0, tau_mu_discrete = 1.0, tau_mu_continuous = 1.0, tau_sigma_discrete = 1.0, tau_sigma_continuous = 1.0, adaptive = None, chains = 4, tune = 1000, draws = 1000, seed = 0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        discrete: usize,
        alpha_loc: f64,
        alpha_scale: f64,
        beta_loc: f64,
        beta_scale: f64,
        partial_pooling: bool,
        tau_mu_global: f64,
        tau_mu_discrete: f64,
        tau_mu_continuous: f64,
        tau_sigma_discrete: f64,
        tau_sigma_continuous: f64,
        adaptive: Option<f64>,
        chains: usize,
        tune: usize,
        draws: usize,
        seed: u64,
    ) -> Self {
        let pooling = partial_pooling.then_some(prospicio_bayes::stacking::Pooling {
            tau_mu_global,
            tau_mu_discrete,
            tau_mu_continuous,
            tau_sigma_discrete,
            tau_sigma_continuous,
        });
        Self {
            inner: prospicio_bayes::stacking::HierarchicalStacking {
                alpha_loc,
                alpha_scale,
                beta_loc,
                beta_scale,
                pooling,
                adaptive,
                discrete,
                sampler: stacking_sampler(chains, tune, draws, seed),
            },
        }
    }

    /// Samples the intercepts and slopes.
    ///
    /// Parameters
    /// ----------
    /// lpd : list of list of float
    ///     One list per model, one held-out log density per observation.
    /// covariates : list of list of float
    ///     One list per covariate, one value per observation.
    ///
    /// Returns
    /// -------
    /// StackingFit
    fn fit(
        &self,
        py: Python<'_>,
        lpd: Vec<Vec<f64>>,
        covariates: Vec<Vec<f64>>,
    ) -> PyResult<PyStackingFit> {
        let spec = &self.inner;
        let inner = py.detach(|| spec.fit(&lpd, &covariates)).map_err(to_py)?;
        Ok(PyStackingFit { inner })
    }
}

/// Posterior stacking weights, from ``BayesStacking.fit`` or
/// ``HierarchicalStacking.fit``.
#[pyclass(name = "StackingFit", module = "prospicio.models", frozen)]
pub(crate) struct PyStackingFit {
    inner: prospicio_bayes::stacking::StackingFit,
}

#[pymethods]
impl PyStackingFit {
    /// Posterior mean weights: one row per observation of ``covariates``
    /// (one list per covariate), one weight per model. For Bayesian
    /// stacking leave ``covariates`` empty: one row.
    ///
    /// Parameters
    /// ----------
    /// covariates : list of list of float, optional
    ///
    /// Returns
    /// -------
    /// list of list of float
    #[pyo3(signature = (covariates = None))]
    fn weights(&self, covariates: Option<Vec<Vec<f64>>>) -> PyResult<Vec<Vec<f64>>> {
        let x = covariates.unwrap_or_default();
        let w = self.inner.weights(&x).map_err(to_py)?;
        let k = w.len() / if x.is_empty() { 1 } else { x[0].len().max(1) };
        Ok(w.chunks(k.max(1)).map(<[f64]>::to_vec).collect())
    }

    /// Intercept draws (the logits for Bayesian stacking), one row per draw,
    /// one per model but the reference (last).
    #[getter]
    fn alpha_draws(&self) -> Vec<f64> {
        self.inner.alpha_draws()
    }

    /// Slope draws, flattened draw by draw, model by model, covariate by
    /// covariate.
    #[getter]
    fn beta_draws(&self) -> Vec<f64> {
        self.inner.beta_draws()
    }

    /// Divergent transitions among the kept draws.
    #[getter]
    fn divergences(&self) -> usize {
        self.inner.divergences()
    }

    /// R-hat and bulk ESS of each sampled parameter.
    ///
    /// Returns
    /// -------
    /// list of (float, float)
    fn rhat_ess(&self) -> PyResult<Vec<(f64, f64)>> {
        self.inner.rhat_ess().map_err(to_py)
    }
}

/// A float, or a list of floats.
#[derive(FromPyObject)]
pub(crate) enum OneOrMany {
    One(f64),
    Many(Vec<f64>),
}

/// Joint predictive draws from fitted means, for engines that give only
/// a mean per row (the boosting adapters): the family adds process noise
/// and several mean vectors (bootstrap refits) add parameter uncertainty.
///
/// Simulation ``i`` uses stream ``i`` of ``seed``: it picks one mean vector
/// uniformly, then draws each row's response from the family with that
/// mean, the row's dispersion and the row's weight. Components are keyed
/// ``row = 0, 1, ...``, as ``GlmFit.predict_distribution`` keys them.
///
/// Parameters
/// ----------
/// family : str
///     As in ``Glm``.
/// means : list of list of float
///     One or more mean vectors, one value per row each.
/// n_sims : int
/// seed : int
/// dispersion : float or list of float, optional
///     One value for every row, or one per row (from a dispersion model);
///     1 by default.
/// weights : list of float, optional
///     Prior weights; 1 by default.
/// theta, power : float, optional
///     Negative binomial ``theta``, Tweedie ``power``.
///
/// Returns
/// -------
/// PredictiveDistribution
///
/// Examples
/// --------
/// >>> from prospicio.models import simulate_from_means
/// >>> pd = simulate_from_means("poisson", [[0.1, 0.4]], 20_000, 7)
/// >>> round(pd.mean(), 1)
/// 0.5
#[pyfunction]
#[pyo3(signature = (family, means, n_sims, seed, dispersion = None, weights = None, theta = None, power = None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_from_means(
    py: Python<'_>,
    family: &str,
    means: Vec<Vec<f64>>,
    n_sims: usize,
    seed: u64,
    dispersion: Option<OneOrMany>,
    weights: Option<Vec<f64>>,
    theta: Option<f64>,
    power: Option<f64>,
) -> PyResult<PyPredictiveDistribution> {
    let f = self::family(family, theta, power)?;
    let w = weights.unwrap_or_default();
    let dispersion = match dispersion {
        None => vec![1.0],
        Some(OneOrMany::One(d)) => vec![d],
        Some(OneOrMany::Many(d)) => d,
    };
    let provenance = prospicio_prob::Provenance::new("simulate_from_means")
        .version("prospicio", env!("CARGO_PKG_VERSION"));
    let inner = py
        .detach(|| {
            prospicio_models::simulate::from_means(
                f,
                &means,
                &dispersion,
                &w,
                n_sims,
                seed,
                provenance,
            )
        })
        .map_err(to_py)?;
    Ok(PyPredictiveDistribution { inner })
}

fn link_or_log(name: Option<&str>) -> PyResult<Link> {
    link(Some(name.unwrap_or("log")), Family::Poisson, None)
}

/// A Tweedie GLM whose power is estimated with its coefficients and
/// dispersion by maximum likelihood: the profile log-likelihood over the
/// power, maximized by Brent's method, with a 95% profile-likelihood
/// interval (R's ``tweedie.profile``'s method).
///
/// Parameters
/// ----------
/// link : str, default "log"
/// min_power, max_power : float, default 1.01 and 1.99
///     The search bounds, inside ``(1, 2)``.
///
/// Examples
/// --------
/// >>> from prospicio.models import Design, TweedieGlm
/// >>> from prospicio.distributions import Tweedie
/// >>> import math
/// >>> x = [float(i % 4) for i in range(400)]
/// >>> y = [Tweedie(math.exp(0.5 + 0.3 * v), 2.0, 1.5).quantile((i * 0.618034) % 1 * 0.98 + 0.01)
/// ...      for i, v in enumerate(x)]
/// >>> fit = TweedieGlm().fit(Design([[1.0] * 400, x], ["(Intercept)", "x"]), y)
/// >>> abs(fit.power - 1.5) < 0.1
/// True
#[pyclass(name = "TweedieGlm", module = "prospicio.models", frozen)]
pub(crate) struct PyTweedieGlm {
    inner: prospicio_glm::tweedie::TweedieGlm,
}

#[pymethods]
impl PyTweedieGlm {
    #[new]
    #[pyo3(signature = (link = None, min_power = 1.01, max_power = 1.99))]
    fn new(link: Option<&str>, min_power: f64, max_power: f64) -> PyResult<Self> {
        let mut inner = prospicio_glm::tweedie::TweedieGlm::new(link_or_log(link)?);
        inner.min_power = min_power;
        inner.max_power = max_power;
        Ok(Self { inner })
    }

    /// Fits the power, the dispersion and the coefficients.
    ///
    /// Returns
    /// -------
    /// TweedieFit
    fn fit(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
    ) -> PyResult<PyTweedieFit> {
        let (spec, d) = (&self.inner, &design.inner);
        let inner = py.detach(|| spec.fit(d, &y)).map_err(to_py)?;
        Ok(PyTweedieFit { inner })
    }
}

/// A Tweedie GLM with its power estimated, from ``TweedieGlm.fit``.
#[pyclass(name = "TweedieFit", module = "prospicio.models", frozen)]
pub(crate) struct PyTweedieFit {
    inner: prospicio_glm::tweedie::TweedieFit,
}

#[pymethods]
impl PyTweedieFit {
    /// The maximum-likelihood power.
    #[getter]
    fn power(&self) -> f64 {
        self.inner.power()
    }

    /// The maximum-likelihood dispersion at that power.
    #[getter]
    fn dispersion(&self) -> f64 {
        self.inner.dispersion()
    }

    /// The maximized log-likelihood.
    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood()
    }

    /// The 95% profile-likelihood interval for the power.
    #[getter]
    fn interval(&self) -> (f64, f64) {
        self.inner.interval()
    }

    /// Whether each end of ``interval`` is a search bound rather than a
    /// crossing.
    #[getter]
    fn interval_open(&self) -> (bool, bool) {
        self.inner.interval_open()
    }

    /// Whether the power is at a search bound.
    #[getter]
    fn at_boundary(&self) -> bool {
        self.inner.at_boundary()
    }

    /// The GLM at the estimated power, its dispersion fixed at the estimate.
    #[getter]
    fn glm(&self) -> PyGlmFit {
        PyGlmFit {
            inner: self.inner.glm().clone(),
        }
    }

    /// Expected response for each row of ``design``.
    fn predict(&self, py: Python<'_>, design: PyRef<'_, PyDesign>) -> PyResult<Vec<f64>> {
        let (fit, d) = (&self.inner, &design.inner);
        py.detach(|| fit.predict(d)).map_err(to_py)
    }
}

/// The profile log-likelihood of a Tweedie GLM's power over ``powers``
/// (increasing, in ``(1, 2)``), the best refined between its neighbours.
///
/// Returns
/// -------
/// dict
///     ``powers``, ``log_likelihood`` and ``dispersion`` (one per power),
///     the refined ``power``, its ``phi`` and ``max_log_likelihood``, and
///     the ``fit`` there (``GlmFit``).
#[pyfunction]
#[pyo3(signature = (design, y, powers, link = None))]
pub(crate) fn tweedie_profile(
    py: Python<'_>,
    design: PyRef<'_, PyDesign>,
    y: Vec<f64>,
    powers: Vec<f64>,
    link: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let l = link_or_log(link)?;
    let d = &design.inner;
    let p = py
        .detach(|| prospicio_glm::tweedie::tweedie_profile(l, d, &y, &powers))
        .map_err(to_py)?;
    let out = pyo3::types::PyDict::new(py);
    out.set_item("powers", p.powers)?;
    out.set_item("log_likelihood", p.log_likelihood)?;
    out.set_item("dispersion", p.dispersion)?;
    out.set_item("power", p.power)?;
    out.set_item("phi", p.phi)?;
    out.set_item("max_log_likelihood", p.max_log_likelihood)?;
    out.set_item("fit", Py::new(py, PyGlmFit { inner: p.fit })?)?;
    Ok(out.into_any().unbind())
}

/// A negative binomial GLM with its ``theta`` estimated with the
/// coefficients by maximum likelihood, as R's ``MASS::glm.nb``: IRLS and
/// Newton's method on ``theta`` alternate until both settle.
///
/// Parameters
/// ----------
/// link : str, default "log"
///
/// Examples
/// --------
/// >>> from prospicio.models import Design, NegativeBinomialGlm
/// >>> x = [float(i % 3) for i in range(60)]
/// >>> y = [float((i * 7) % 11) for i in range(60)]
/// >>> fit = NegativeBinomialGlm().fit(Design([[1.0] * 60, x], ["(Intercept)", "x"]), y)
/// >>> fit.theta > 0 and fit.se_theta > 0
/// True
#[pyclass(name = "NegativeBinomialGlm", module = "prospicio.models", frozen)]
pub(crate) struct PyNegativeBinomialGlm {
    inner: prospicio_glm::negbin::NegativeBinomialGlm,
}

#[pymethods]
impl PyNegativeBinomialGlm {
    #[new]
    #[pyo3(signature = (link = None))]
    fn new(link: Option<&str>) -> PyResult<Self> {
        Ok(Self {
            inner: prospicio_glm::negbin::NegativeBinomialGlm::new(link_or_log(link)?),
        })
    }

    /// Fits ``theta`` and the coefficients.
    ///
    /// Returns
    /// -------
    /// NegativeBinomialFit
    fn fit(
        &self,
        py: Python<'_>,
        design: PyRef<'_, PyDesign>,
        y: Vec<f64>,
    ) -> PyResult<PyNegativeBinomialFit> {
        let (spec, d) = (&self.inner, &design.inner);
        let inner = py.detach(|| spec.fit(d, &y)).map_err(to_py)?;
        Ok(PyNegativeBinomialFit { inner })
    }
}

/// A negative binomial GLM with ``theta`` estimated, from
/// ``NegativeBinomialGlm.fit``.
#[pyclass(name = "NegativeBinomialFit", module = "prospicio.models", frozen)]
pub(crate) struct PyNegativeBinomialFit {
    inner: prospicio_glm::negbin::NegativeBinomialFit,
}

#[pymethods]
impl PyNegativeBinomialFit {
    /// The maximum-likelihood ``theta``.
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta()
    }

    /// ``MASS``'s standard error of ``theta``.
    #[getter]
    fn se_theta(&self) -> f64 {
        self.inner.se_theta()
    }

    /// The maximized log-likelihood.
    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood()
    }

    /// ``-2 log_likelihood + 2 (p + 1)``, counting ``theta``, as R's ``AIC``.
    #[getter]
    fn aic(&self) -> f64 {
        self.inner.aic()
    }

    /// The GLM at the estimated ``theta``.
    #[getter]
    fn glm(&self) -> PyGlmFit {
        PyGlmFit {
            inner: self.inner.glm().clone(),
        }
    }

    /// Expected response for each row of ``design``.
    fn predict(&self, py: Python<'_>, design: PyRef<'_, PyDesign>) -> PyResult<Vec<f64>> {
        let (fit, d) = (&self.inner, &design.inner);
        py.detach(|| fit.predict(d)).map_err(to_py)
    }
}
