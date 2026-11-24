//! The Munich chain ladder, `MunichChainLadder`.

use super::*;

/// The Munich chain ladder (Quarg and Mack 2004): paid and incurred
/// developed together, each factor corrected by how far the origin's
/// paid-to-incurred ratio sits from its average at that age, as R
/// ChainLadder's ``MunichChainLadder`` and chainladder-python's
/// ``MunichAdjustment`` (``docs/design/reserving-v02.md``, decision 33).
///
/// Each column has Mack's volume-weighted model; the slopes ``lambda_paid``
/// and ``lambda_incurred`` of its link-ratio residuals on the ratio
/// residuals set the correction. With both 0 the projections are the two
/// separate chain ladders.
///
/// Parameters
/// ----------
/// sigma_interpolation_paid : str, default "log-linear"
///     How the paid column's unestimable sigma is filled in (R's
///     ``est.sigmaP``): ``"log-linear"`` or ``"mack"``. R's default call
///     falls back to ``"mack"`` with a warning when its log-linear model
///     is poor (as on its ``MCLpaid``); prospicio does not, so give
///     ``"mack"`` to match it there.
/// sigma_interpolation_incurred : str, default "log-linear"
///     The same for incurred (R's ``est.sigmaI``).
/// tail_paid : float or tail estimator, optional
///     The paid tail (R's ``tailP``), as ``Mack``'s ``tail``; none by
///     default.
/// tail_incurred : float or tail estimator, optional
///     The incurred tail (R's ``tailI``).
///
/// Examples
/// --------
/// >>> from prospicio.reserving import MunichChainLadder, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     {
/// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ...         "incurred": [160.0, 175.0, 172.0, 171.0, 150.0, 190.0, 185.0, 200.0, 185.0, 210.0],
/// ...     },
/// ... )
/// >>> fit = MunichChainLadder().fit(tri, "paid", "incurred")
/// >>> fit.ultimate_paid[0]
/// 170.0
#[pyclass(name = "MunichChainLadder", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMunichChainLadder {
    inner: MunichChainLadder,
}

#[pymethods]
impl PyMunichChainLadder {
    #[new]
    #[pyo3(signature = (
        sigma_interpolation_paid = "log-linear",
        sigma_interpolation_incurred = "log-linear",
        tail_paid = None,
        tail_incurred = None,
    ))]
    fn new(
        sigma_interpolation_paid: &str,
        sigma_interpolation_incurred: &str,
        tail_paid: Option<&Bound<'_, PyAny>>,
        tail_incurred: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dev = |name: &str| -> PyResult<Development> {
            Ok(Development {
                sigma_interpolation: sigma_interpolation(name)?,
                ..Default::default()
            })
        };
        Ok(Self {
            inner: MunichChainLadder {
                paid: dev(sigma_interpolation_paid)?,
                incurred: dev(sigma_interpolation_incurred)?,
                tail_paid: tail_arg(tail_paid)?,
                tail_incurred: tail_arg(tail_incurred)?,
            },
        })
    }

    /// How the paid column's unestimable sigma is filled in.
    #[getter]
    fn sigma_interpolation_paid(&self) -> &'static str {
        sigma_interpolation_name(self.inner.paid.sigma_interpolation)
    }

    /// How the incurred column's unestimable sigma is filled in.
    #[getter]
    fn sigma_interpolation_incurred(&self) -> &'static str {
        sigma_interpolation_name(self.inner.incurred.sigma_interpolation)
    }

    /// The paid tail: a constant factor as a number, otherwise its
    /// estimator.
    #[getter]
    fn tail_paid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.tail_paid)
    }

    /// The incurred tail.
    #[getter]
    fn tail_incurred<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.tail_incurred)
    }

    /// Fits the paid and incurred columns of a triangle, in every segment
    /// on its own.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Holds both columns, observed at the same cells.
    /// paid : str
    ///     The paid column.
    /// incurred : str
    ///     The incurred column.
    ///
    /// Returns
    /// -------
    /// MunichFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the two columns are not observed at the same cells (a missing
    ///     incurred cell, say), a cell is not positive, or either column
    ///     fails as in ``Mack.fit``.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        incurred: &str,
    ) -> PyResult<PyMunichFit> {
        let tri = &triangle.inner;
        let inner = py
            .detach(|| self.inner.fit_segments(tri, paid, incurred))
            .map_err(err)?;
        Ok(PyMunichFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "MunichChainLadder(sigma_interpolation_paid={:?}, sigma_interpolation_incurred={:?}, \
             tail_paid={}, tail_incurred={})",
            self.sigma_interpolation_paid(),
            self.sigma_interpolation_incurred(),
            tail_arg_repr(&self.inner.tail_paid),
            tail_arg_repr(&self.inner.tail_incurred),
        )
    }
}

/// A fitted Munich chain ladder of every segment.
///
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()``. Per-age lists and the lambdas need a
/// single-segment fit; for several segments use ``development_frame()``,
/// ``totals_frame()`` or ``segment(...)``.
#[pyclass(name = "MunichFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMunichFit {
    inner: SegmentFits<MunichFit>,
}

impl PyMunichFit {
    fn one(&self, field: &str, instead: &str) -> PyResult<&MunichFit> {
        single(&self.inner, field, instead)
    }

    fn per_origin(&self, f: impl Fn(&MunichFit) -> Vec<f64>) -> Vec<f64> {
        by_origin(&self.inner, f)
    }
}

#[pymethods]
impl PyMunichFit {
    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner)
    }

    /// Origin period of each per-origin value.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .fits
            .iter()
            .flat_map(|f| f.origins.iter().map(ToString::to_string))
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> PyResult<Vec<Lag>> {
        Ok(self
            .one("development", "development_frame()")?
            .development
            .clone())
    }

    /// The Munich paid ultimate per origin.
    #[getter]
    fn ultimate_paid(&self) -> Vec<f64> {
        self.per_origin(|f| f.ultimate_paid.clone())
    }

    /// The Munich incurred ultimate per origin.
    #[getter]
    fn ultimate_incurred(&self) -> Vec<f64> {
        self.per_origin(|f| f.ultimate_incurred.clone())
    }

    /// Latest paid per origin.
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        self.per_origin(|f| f.latest_paid.clone())
    }

    /// Latest incurred per origin.
    #[getter]
    fn latest_incurred(&self) -> Vec<f64> {
        self.per_origin(|f| f.latest_incurred.clone())
    }

    /// Paid ultimate minus latest paid per origin.
    #[getter]
    fn reserve_paid(&self) -> Vec<f64> {
        self.per_origin(MunichFit::reserve_paid)
    }

    /// Incurred ultimate minus latest incurred per origin.
    #[getter]
    fn ibnr_incurred(&self) -> Vec<f64> {
        self.per_origin(MunichFit::ibnr_incurred)
    }

    /// Latest paid over latest incurred per origin.
    #[getter]
    fn latest_ratio(&self) -> Vec<f64> {
        self.per_origin(MunichFit::latest_ratio)
    }

    /// Paid ultimate over incurred ultimate per origin.
    #[getter]
    fn ultimate_ratio(&self) -> Vec<f64> {
        self.per_origin(MunichFit::ultimate_ratio)
    }

    /// The separate chain ladder's paid ultimate per origin.
    #[getter]
    fn chain_ladder_paid(&self) -> Vec<f64> {
        self.per_origin(|f| f.paid.chain_ladder.ultimate.clone())
    }

    /// The separate chain ladder's incurred ultimate per origin.
    #[getter]
    fn chain_ladder_incurred(&self) -> Vec<f64> {
        self.per_origin(|f| f.incurred.chain_ladder.ultimate.clone())
    }

    /// Total Munich paid ultimate over every segment.
    #[getter]
    fn total_ultimate_paid(&self) -> f64 {
        self.inner
            .fits
            .iter()
            .map(MunichFit::total_ultimate_paid)
            .sum()
    }

    /// Total Munich incurred ultimate over every segment.
    #[getter]
    fn total_ultimate_incurred(&self) -> f64 {
        self.inner
            .fits
            .iter()
            .map(MunichFit::total_ultimate_incurred)
            .sum()
    }

    /// The slope of the paid link-ratio residuals on the incurred-to-paid
    /// residuals (R's ``lambdaP``); 0 when the ratios carry no information.
    #[getter]
    fn lambda_paid(&self) -> PyResult<f64> {
        Ok(self.one("lambda_paid", "totals_frame()")?.lambda_paid)
    }

    /// The slope of the incurred link-ratio residuals on the
    /// paid-to-incurred residuals (R's ``lambdaI``).
    #[getter]
    fn lambda_incurred(&self) -> PyResult<f64> {
        Ok(self
            .one("lambda_incurred", "totals_frame()")?
            .lambda_incurred)
    }

    /// The number of residual pairs behind ``lambda_paid``.
    #[getter]
    fn n_residuals_paid(&self) -> PyResult<usize> {
        Ok(self
            .one("n_residuals_paid", "totals_frame()")?
            .n_residuals_paid)
    }

    /// The number of residual pairs behind ``lambda_incurred``: fewer than
    /// paid's where incurred's sigma is 0 at a link (residuals 0/0, left
    /// out of incurred's regression only, as in R).
    #[getter]
    fn n_residuals_incurred(&self) -> PyResult<usize> {
        Ok(self
            .one("n_residuals_incurred", "totals_frame()")?
            .n_residuals_incurred)
    }

    /// The paid-to-incurred ratio at each age, ``sum P / sum I`` (R's
    /// ``q.f``).
    #[getter]
    fn q(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("q", "development_frame()")?.q.clone())
    }

    /// The incurred-to-paid ratio at each age, fitted on its own,
    /// ``sum I / sum P`` (R's ``qinverse.f``).
    #[getter]
    fn q_inverse(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("q_inverse", "development_frame()")?
            .q_inverse
            .clone())
    }

    /// The deviation of ``I/P`` around ``q_inverse`` (R's ``rhoP.sigma``).
    #[getter]
    fn rho_paid(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("rho_paid", "development_frame()")?
            .rho_paid
            .clone())
    }

    /// The deviation of ``P/I`` around ``q`` (R's ``rhoI.sigma``).
    #[getter]
    fn rho_incurred(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("rho_incurred", "development_frame()")?
            .rho_incurred
            .clone())
    }

    /// Mack's model of the paid column alone.
    #[getter]
    fn mack_paid(&self) -> PyMackFit {
        PyMackFit {
            inner: self.inner.map(|f| f.paid.clone()),
        }
    }

    /// Mack's model of the incurred column alone.
    #[getter]
    fn mack_incurred(&self) -> PyMackFit {
        PyMackFit {
            inner: self.inner.map(|f| f.incurred.clone()),
        }
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest_paid``, ``latest_incurred``, ``latest_ratio``,
    /// ``ultimate_paid``, ``ultimate_incurred``, ``ultimate_ratio``,
    /// ``reserve_paid``, ``ibnr_incurred``, ``chain_ladder_paid`` and
    /// ``chain_ladder_incurred``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.munich_long())
    }

    /// One row per segment and age: ``q``, ``q_inverse``, ``rho_paid``,
    /// ``rho_incurred``, and each column's ``ldf_paid``, ``sigma_paid``,
    /// ``ldf_incurred`` and ``sigma_incurred`` (NaN at the oldest age).
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.munich_development_table())
    }

    /// One row per segment: the totals of ``to_frame()``'s columns (the
    /// ratios of the totals), ``lambda_paid``, ``lambda_incurred``,
    /// ``n_residuals_paid`` and ``n_residuals_incurred``.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.munich_totals())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``.
    ///
    /// Returns
    /// -------
    /// MunichFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MunichFit({}origins={}, total_ultimate_paid={:?}, total_ultimate_incurred={:?})",
            segments_prefix(&self.inner),
            self.origins().len(),
            self.total_ultimate_paid(),
            self.total_ultimate_incurred(),
        )
    }
}
