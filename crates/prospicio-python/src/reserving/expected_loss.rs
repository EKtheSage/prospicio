//! The expected-loss methods: `ExpectedLoss`, `BornhuetterFerguson`, `Benktander` and `CapeCod`, and their fits.

use super::*;

/// An ``apriori`` argument: a number for every origin, a sequence of one
/// per origin in origin order (oldest first, over the triangle's or the
/// diagonals' origins), or a dict from origin (a label as fits report
/// origins, ``"2021"`` or ``"2021Q3"``, or an integer year) to its loss
/// ratio. A pandas Series is read by its index, as a dict; one with a
/// default ``RangeIndex`` is an error, since its order cannot be checked.
/// ``None`` is 1.
fn apriori_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Apriori> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(Apriori::default());
    };
    let type_error = || {
        PyTypeError::new_err(format!(
            "apriori must be a number, a sequence of numbers in origin order or a dict from \
             origin to number, got {}",
            py_repr(obj)
        ))
    };
    if obj.is_instance_of::<PyBool>() {
        return Err(type_error());
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        return apriori_by_origin(dict.iter().map(Ok));
    }
    if is_pandas_series(obj)? {
        // A default index, 0 to n - 1 by 1; `range(1988, 1998)` is years.
        let index = obj.getattr("index")?;
        if index.get_type().name()?.extract::<String>()? == "RangeIndex"
            && index.getattr("start")?.extract::<i64>()? == 0
            && index.getattr("step")?.extract::<i64>()? == 1
        {
            return Err(PyValueError::new_err(
                "apriori is a pandas Series with a default RangeIndex, so the origin of each \
                 value cannot be checked: index it by origin label or integer year (as \
                 df.set_index(\"origin\")[\"elr\"]), or pass .tolist() for values in origin \
                 order",
            ));
        }
        // `items()`, not `to_dict()`, so an origin given twice stays an error.
        return apriori_by_origin(
            obj.call_method0("items")?
                .try_iter()?
                .map(|item| item?.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()),
        );
    }
    if let Ok(value) = obj.extract::<f64>() {
        return Ok(Apriori::All(value));
    }
    plain(obj)?
        .extract::<Vec<f64>>()
        .map(Apriori::InOrder)
        .map_err(|_| type_error())
}

/// Whether `obj` is a pandas Series, found without importing pandas: its
/// type comes from a `pandas` module and is named `Series`.
fn is_pandas_series(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let ty = obj.get_type();
    let module = ty
        .getattr("__module__")?
        .extract::<String>()
        .unwrap_or_default();
    Ok((module == "pandas" || module.starts_with("pandas."))
        && ty.name()?.extract::<String>()? == "Series")
}

/// An ``apriori`` keyed by origin from `(key, value)` pairs: a key is an
/// origin label or an integer year.
pub(crate) fn apriori_by_origin<'py>(
    items: impl Iterator<Item = PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>>,
) -> PyResult<Apriori> {
    let pairs = items
        .map(|item| {
            let (k, v) = item?;
            let label = if k.is_instance_of::<PyString>() {
                k.extract::<String>()?
            } else if !k.is_instance_of::<PyBool>()
                && (k.is_instance_of::<PyInt>() || k.hasattr("__index__")?)
            {
                k.extract::<i32>()?.to_string()
            } else {
                return Err(PyTypeError::new_err(format!(
                    "apriori's keys must be origin labels such as \"2021\" or \"2021Q3\", or \
                     integer years, got {}",
                    py_repr(&k)
                )));
            };
            let value = v.extract::<f64>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "apriori's value for {label} must be a number, got {}",
                    py_repr(&v)
                ))
            })?;
            Ok((label, value))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Apriori::from_labels(pairs).map_err(err)
}

/// A method's ``apriori`` as Python sees it: a float, a list in origin
/// order or a dict by origin label.
fn apriori_object<'py>(py: Python<'py>, apriori: &Apriori) -> PyResult<Bound<'py, PyAny>> {
    Ok(match apriori {
        Apriori::All(a) => PyFloat::new(py, *a).into_any(),
        Apriori::InOrder(values) => PyList::new(py, values)?.into_any(),
        Apriori::ByOrigin(pairs) => {
            let d = PyDict::new(py);
            for (origin, v) in pairs {
                d.set_item(origin.to_string(), v)?;
            }
            d.into_any()
        }
    })
}

/// ``apriori`` in a method's repr.
fn apriori_repr(apriori: &Apriori) -> String {
    match apriori {
        Apriori::All(a) => format!("{a:?}"),
        Apriori::InOrder(values) => format!("{values:?}"),
        Apriori::ByOrigin(pairs) => {
            let items: Vec<String> = pairs.iter().map(|(o, v)| format!("'{o}': {v:?}")).collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

/// The expected loss ratio method: each origin's ultimate is ``apriori``
/// times its exposure, whatever has been observed. The chain ladder is
/// still fitted for the development pattern the fit reports.
///
/// The exposure is a measure column of the same triangle (premium, say):
/// each origin's latest observed cumulative value in the segment fitted.
///
/// Parameters
/// ----------
/// apriori : float, sequence or dict, default 1.0
///     Expected loss ratio: the ultimate per unit of exposure, positive:
///     one for every origin, one per origin in origin order (oldest first,
///     over the triangle's origins or, for a latest diagonal, its origins),
///     or a dict from origin (its label as fits report it, ``"2021"`` or
///     ``"2021Q3"``, or an integer year) to its loss ratio. Every origin
///     fitted needs one; a value for an origin the triangle does not have
///     is an error. Over segments each segment takes its own origins'
///     values. A pandas Series is read by its index, as a dict, so
///     ``df.set_index("origin")["elr"]`` works in any row order; one with
///     a default ``RangeIndex`` is an error, since its order cannot be
///     checked. A Polars Series or numpy array is a sequence. On a latest
///     diagonal a sequence needs the rows oldest first, each origin once,
///     and is otherwise an error, since the exposure is read by row.
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
///     How link ratios are averaged, as in ``ChainLadder``.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``: average only the latest diagonals and leave out
///     the highest and lowest link ratios at each age.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``: the
///     development pattern's factors at the ages it gives, so ``cdf`` and
///     the share developed ``1 / cdf`` follow it.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ExpectedLoss, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = ExpectedLoss(apriori=0.5).fit(tri, "paid", "premium")
/// >>> fit.ultimate, fit.reserve
/// ([125.0, 200.0], [-25.0, 0.0])
#[pyclass(name = "ExpectedLoss", module = "prospicio.reserving", frozen)]
pub(crate) struct PyExpectedLoss {
    inner: ExpectedLoss,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyExpectedLoss {
    #[new]
    #[pyo3(signature = (apriori = None, average = "volume", sigma_interpolation = "log-linear", tail = None, ldf = None, ldf_kind = None, ldf_weight = None, n_periods = None, drop_high = 0, drop_low = 0, ldf_origin = None, ldf_underwriting = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: Option<&Bound<'_, PyAny>>,
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        Ok(Self {
            ldf_basis,
            inner: ExpectedLoss {
                apriori: apriori_arg(apriori)?,
                chain_ladder: pattern(
                    average,
                    sigma_interpolation,
                    tail,
                    ldf,
                    ldf_kind,
                    ldf_weight,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
            },
        })
    }

    /// Expected loss ratio: a float, a list in origin order or a dict by origin
    /// label, as given.
    #[getter]
    fn apriori<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        apriori_object(py, &self.inner.apriori)
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.chain_ladder.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.chain_ladder.development.sigma_interpolation)
    }

    /// Number of latest diagonals averaged; ``None`` uses every link ratio.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.chain_ladder.development.n_periods
    }

    /// Highest link ratios left out at each age.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.chain_ladder.development.drop_high
    }

    /// Lowest link ratios left out at each age.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.chain_ladder.development.drop_low
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.chain_ladder.tail)
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure (an incremental triangle's is cumulated).
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, if ``apriori`` is not positive, or if an
    ///     origin has no observed, finite, positive exposure (the message
    ///     names it and, with keys, its segment).
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyExpectedLossFit> {
        let tri = &triangle.inner;
        let m = self.resolved(tri.origin_grain())?;
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    /// ``ExpectedLoss`` on a latest diagonal with its exposure, in every
    /// segment, as ``ChainLadder.fit_diagonal`` projects one.
    ///
    /// Parameters
    /// ----------
    /// diagonal : LatestDiagonal
    ///     With an exposure per origin.
    /// development : ChainLadderFit, optional
    ///     As ``ChainLadder.fit_diagonal``; this method keeps its own
    ///     ``apriori``.
    /// segment_map : dict, optional
    ///     As ``ChainLadder.fit_diagonal``.
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    #[pyo3(signature = (diagonal, development = None, segment_map = None))]
    fn fit_diagonal(
        &self,
        py: Python<'_>,
        diagonal: PyRef<'_, PyLatestDiagonal>,
        development: Option<&Bound<'_, PyAny>>,
        segment_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyExpectedLossFit> {
        let d = &diagonal.inner;
        let m = self.resolved(d.origin_grain())?;
        let source = diagonal_development(d, development, segment_map)?;
        let inner = py
            .detach(|| match &source {
                None => m.fit_diagonal_segments(d),
                Some(s) => m.fit_diagonal_segments_from(d, s),
            })
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl_resolved!(PyExpectedLoss => ExpectedLoss, chain_ladder);

impl PyExpectedLoss {
    fn base_repr(&self) -> String {
        format!(
            "ExpectedLoss(apriori={}, {})",
            apriori_repr(&self.inner.apriori),
            pattern_repr(&self.inner.chain_ladder)
        )
    }
}

/// The Bornhuetter–Ferguson method: each origin's latest value plus the
/// expected loss ``apriori * exposure`` times the share still to develop,
/// ``1 - 1 / cdf``, as chainladder-python's ``BornhuetterFerguson``.
///
/// The exposure is a measure column of the same triangle (premium, say):
/// each origin's latest observed cumulative value in the segment fitted.
///
/// Parameters
/// ----------
/// apriori : float, sequence or dict, default 1.0
///     Expected loss ratio: the expected ultimate per unit of exposure,
///     positive: one for every origin, one per origin in origin order
///     (oldest first, over the triangle's origins or, for a latest
///     diagonal, its origins), or a dict from origin (its label as fits
///     report it, ``"2021"`` or ``"2021Q3"``, or an integer year) to its
///     loss ratio. Every origin fitted needs one; a value for an origin the
///     triangle does not have is an error. Over segments each segment takes
///     its own origins' values. A pandas Series is read by its index, as
///     a dict, so ``df.set_index("origin")["elr"]`` works in any row
///     order; one with a default ``RangeIndex`` is an error, since its
///     order cannot be checked. A Polars Series or numpy array is a
///     sequence. On a latest diagonal a sequence needs the rows oldest
///     first, each origin once, and is otherwise an error, since the
///     exposure is read by row.
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
///     How link ratios are averaged, as in ``ChainLadder``.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``: average only the latest diagonals and leave out
///     the highest and lowest link ratios at each age.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``: the
///     development pattern's factors at the ages it gives, so ``cdf`` and
///     the share developed ``1 / cdf`` follow it.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BornhuetterFerguson, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = BornhuetterFerguson(apriori=0.5).fit(tri, "paid", "premium")
/// >>> [round(u, 2) for u in fit.ultimate]
/// [150.0, 266.67]
#[pyclass(name = "BornhuetterFerguson", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBornhuetterFerguson {
    inner: BornhuetterFerguson,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyBornhuetterFerguson {
    #[new]
    #[pyo3(signature = (apriori = None, average = "volume", sigma_interpolation = "log-linear", tail = None, ldf = None, ldf_kind = None, ldf_weight = None, n_periods = None, drop_high = 0, drop_low = 0, ldf_origin = None, ldf_underwriting = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: Option<&Bound<'_, PyAny>>,
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        Ok(Self {
            ldf_basis,
            inner: BornhuetterFerguson {
                apriori: apriori_arg(apriori)?,
                chain_ladder: pattern(
                    average,
                    sigma_interpolation,
                    tail,
                    ldf,
                    ldf_kind,
                    ldf_weight,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
            },
        })
    }

    /// Expected loss ratio: a float, a list in origin order or a dict by origin
    /// label, as given.
    #[getter]
    fn apriori<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        apriori_object(py, &self.inner.apriori)
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.chain_ladder.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.chain_ladder.development.sigma_interpolation)
    }

    /// Number of latest diagonals averaged; ``None`` uses every link ratio.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.chain_ladder.development.n_periods
    }

    /// Highest link ratios left out at each age.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.chain_ladder.development.drop_high
    }

    /// Lowest link ratios left out at each age.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.chain_ladder.development.drop_low
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.chain_ladder.tail)
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure (an incremental triangle's is cumulated).
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ExpectedLoss.fit``.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyExpectedLossFit> {
        let tri = &triangle.inner;
        let m = self.resolved(tri.origin_grain())?;
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    /// ``BornhuetterFerguson`` on a latest diagonal with its exposure, in every
    /// segment, as ``ChainLadder.fit_diagonal`` projects one.
    ///
    /// Parameters
    /// ----------
    /// diagonal : LatestDiagonal
    ///     With an exposure per origin.
    /// development : ChainLadderFit, optional
    ///     As ``ChainLadder.fit_diagonal``; this method keeps its own
    ///     ``apriori``.
    /// segment_map : dict, optional
    ///     As ``ChainLadder.fit_diagonal``.
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    #[pyo3(signature = (diagonal, development = None, segment_map = None))]
    fn fit_diagonal(
        &self,
        py: Python<'_>,
        diagonal: PyRef<'_, PyLatestDiagonal>,
        development: Option<&Bound<'_, PyAny>>,
        segment_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyExpectedLossFit> {
        let d = &diagonal.inner;
        let m = self.resolved(d.origin_grain())?;
        let source = diagonal_development(d, development, segment_map)?;
        let inner = py
            .detach(|| match &source {
                None => m.fit_diagonal_segments(d),
                Some(s) => m.fit_diagonal_segments_from(d, s),
            })
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl_resolved!(PyBornhuetterFerguson => BornhuetterFerguson, chain_ladder);

impl PyBornhuetterFerguson {
    fn base_repr(&self) -> String {
        format!(
            "BornhuetterFerguson(apriori={}, {})",
            apriori_repr(&self.inner.apriori),
            pattern_repr(&self.inner.chain_ladder)
        )
    }
}

/// The Benktander (iterated Bornhuetter–Ferguson) method: starting from
/// ``U(0) = apriori * exposure``, ``U(k) = latest + (1 - 1 / cdf) * U(k-1)``
/// for ``n_iters`` steps, as chainladder-python's ``Benktander``.
/// ``n_iters=0`` is the expected loss method, 1 is Bornhuetter–Ferguson,
/// and many iterations approach the chain ladder. The steps are summed in
/// closed form, so a large ``n_iters`` is cheap; where an origin's ``cdf``
/// is below 1/2 they diverge instead.
///
/// Parameters
/// ----------
/// apriori : float, sequence or dict, default 1.0
///     Expected loss ratio of the starting ultimate, positive: one for
///     every origin, one per origin in origin order (oldest first, over the
///     triangle's origins or, for a latest diagonal, its origins), or a
///     dict from origin (its label as fits report it, ``"2021"`` or
///     ``"2021Q3"``, or an integer year) to its loss ratio. Every origin
///     fitted needs one; a value for an origin the triangle does not have
///     is an error. Over segments each segment takes its own origins'
///     values. A pandas Series is read by its index, as a dict, so
///     ``df.set_index("origin")["elr"]`` works in any row order; one with
///     a default ``RangeIndex`` is an error, since its order cannot be
///     checked. A Polars Series or numpy array is a sequence. On a latest
///     diagonal a sequence needs the rows oldest first, each origin once,
///     and is otherwise an error, since the exposure is read by row.
/// n_iters : int, default 1
///     Number of Bornhuetter–Ferguson steps.
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
///     How link ratios are averaged, as in ``ChainLadder``.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``: average only the latest diagonals and leave out
///     the highest and lowest link ratios at each age.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``: the
///     development pattern's factors at the ages it gives, so ``cdf`` and
///     the share developed ``1 / cdf`` follow it.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Benktander, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = Benktander(apriori=0.5, n_iters=2).fit(tri, "paid", "premium")
/// >>> [round(u, 2) for u in fit.ultimate]
/// [150.0, 288.89]
#[pyclass(name = "Benktander", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBenktander {
    inner: Benktander,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyBenktander {
    #[new]
    #[pyo3(signature = (apriori = None, n_iters = 1, average = "volume", sigma_interpolation = "log-linear", tail = None, ldf = None, ldf_kind = None, ldf_weight = None, n_periods = None, drop_high = 0, drop_low = 0, ldf_origin = None, ldf_underwriting = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: Option<&Bound<'_, PyAny>>,
        n_iters: usize,
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        Ok(Self {
            ldf_basis,
            inner: Benktander {
                apriori: apriori_arg(apriori)?,
                n_iters,
                chain_ladder: pattern(
                    average,
                    sigma_interpolation,
                    tail,
                    ldf,
                    ldf_kind,
                    ldf_weight,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
            },
        })
    }

    /// Expected loss ratio of the starting ultimate: a float, a list in origin
    /// order or a dict by origin label, as given.
    #[getter]
    fn apriori<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        apriori_object(py, &self.inner.apriori)
    }

    /// Number of Bornhuetter–Ferguson steps.
    #[getter]
    fn n_iters(&self) -> usize {
        self.inner.n_iters
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.chain_ladder.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.chain_ladder.development.sigma_interpolation)
    }

    /// Number of latest diagonals averaged; ``None`` uses every link ratio.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.chain_ladder.development.n_periods
    }

    /// Highest link ratios left out at each age.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.chain_ladder.development.drop_high
    }

    /// Lowest link ratios left out at each age.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.chain_ladder.development.drop_low
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.chain_ladder.tail)
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure (an incremental triangle's is cumulated).
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ExpectedLoss.fit``.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyExpectedLossFit> {
        let tri = &triangle.inner;
        let m = self.resolved(tri.origin_grain())?;
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    /// ``Benktander`` on a latest diagonal with its exposure, in every
    /// segment, as ``ChainLadder.fit_diagonal`` projects one.
    ///
    /// Parameters
    /// ----------
    /// diagonal : LatestDiagonal
    ///     With an exposure per origin.
    /// development : ChainLadderFit, optional
    ///     As ``ChainLadder.fit_diagonal``; this method keeps its own
    ///     ``apriori`` and ``n_iters``.
    /// segment_map : dict, optional
    ///     As ``ChainLadder.fit_diagonal``.
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    #[pyo3(signature = (diagonal, development = None, segment_map = None))]
    fn fit_diagonal(
        &self,
        py: Python<'_>,
        diagonal: PyRef<'_, PyLatestDiagonal>,
        development: Option<&Bound<'_, PyAny>>,
        segment_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyExpectedLossFit> {
        let d = &diagonal.inner;
        let m = self.resolved(d.origin_grain())?;
        let source = diagonal_development(d, development, segment_map)?;
        let inner = py
            .detach(|| match &source {
                None => m.fit_diagonal_segments(d),
                Some(s) => m.fit_diagonal_segments_from(d, s),
            })
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl_resolved!(PyBenktander => Benktander, chain_ladder);

impl PyBenktander {
    fn base_repr(&self) -> String {
        format!(
            "Benktander(apriori={}, n_iters={}, {})",
            apriori_repr(&self.inner.apriori),
            self.inner.n_iters,
            pattern_repr(&self.inner.chain_ladder)
        )
    }
}

/// The Cape Cod (Stanard–Bühlmann) method: Bornhuetter–Ferguson with each
/// origin's apriori estimated from the triangle, as chainladder-python's
/// ``CapeCod``.
///
/// Origin ``j``'s used-up exposure is ``exposure[j] / cdf[j]`` and its
/// latest value is trended to the triangle's valuation by
/// ``(1 + trend) ** (months / 12)``, the months running from the end of
/// the origin period. Origin ``i``'s trended apriori is the sum of the
/// trended latest values weighted by ``decay ** abs(i - j)`` over the same
/// weighted sum of used-up exposures; dividing by its own trend factor
/// gives the apriori of its Bornhuetter–Ferguson ultimate.
///
/// Parameters
/// ----------
/// trend : float, default 0.0
///     Annual trend of the loss ratio; above -1.
/// decay : float, default 1.0
///     Weight of an origin ``n`` periods away, ``decay ** n``; from 0 to 1.
///     With 1 every origin shares one loss ratio.
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
///     How link ratios are averaged, as in ``ChainLadder``.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``: average only the latest diagonals and leave out
///     the highest and lowest link ratios at each age.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``: the
///     development pattern's factors at the ages it gives, so ``cdf`` and
///     the share developed ``1 / cdf`` follow it.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import CapeCod, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = CapeCod().fit(tri, "paid", "premium")
/// >>> [round(a, 4) for a in fit.apriori], [round(u, 2) for u in fit.ultimate]
/// ([0.6774, 0.6774], [150.0, 290.32])
#[pyclass(name = "CapeCod", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCapeCod {
    inner: CapeCod,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyCapeCod {
    #[new]
    #[pyo3(signature = (trend = 0.0, decay = 1.0, average = "volume", sigma_interpolation = "log-linear", tail = None, ldf = None, ldf_kind = None, ldf_weight = None, n_periods = None, drop_high = 0, drop_low = 0, ldf_origin = None, ldf_underwriting = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        trend: f64,
        decay: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        Ok(Self {
            ldf_basis,
            inner: CapeCod {
                trend,
                decay,
                chain_ladder: pattern(
                    average,
                    sigma_interpolation,
                    tail,
                    ldf,
                    ldf_kind,
                    ldf_weight,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
            },
        })
    }

    /// Annual trend of the loss ratio.
    #[getter]
    fn trend(&self) -> f64 {
        self.inner.trend
    }

    /// Weight of an origin one period away.
    #[getter]
    fn decay(&self) -> f64 {
        self.inner.decay
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.chain_ladder.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.chain_ladder.development.sigma_interpolation)
    }

    /// Number of latest diagonals averaged; ``None`` uses every link ratio.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.chain_ladder.development.n_periods
    }

    /// Highest link ratios left out at each age.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.chain_ladder.development.drop_high
    }

    /// Lowest link ratios left out at each age.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.chain_ladder.development.drop_low
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.chain_ladder.tail)
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.chain_ladder.development.pattern.as_ref())
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure and apriori. Trend runs to the triangle's valuation.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure (an incremental triangle's is cumulated).
    ///
    /// Returns
    /// -------
    /// CapeCodFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, if ``trend`` or ``decay`` is out of
    ///     range, or if an origin has no observed, finite, positive exposure.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyCapeCodFit> {
        let tri = &triangle.inner;
        let m = self.resolved(tri.origin_grain())?;
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyCapeCodFit { inner })
    }

    /// ``CapeCod`` on a latest diagonal with its exposure, in every
    /// segment, as ``ChainLadder.fit_diagonal`` projects one: its apriori
    /// is estimated on the diagonal and trended to one valuation, the
    /// latest month any origin of any segment is valued at.
    ///
    /// Parameters
    /// ----------
    /// diagonal : LatestDiagonal
    ///     With an exposure per origin.
    /// development : ChainLadderFit, optional
    ///     As ``ChainLadder.fit_diagonal``; this method keeps its own
    ///     ``trend`` and ``decay``.
    /// segment_map : dict, optional
    ///     As ``ChainLadder.fit_diagonal``.
    ///
    /// Returns
    /// -------
    /// CapeCodFit
    #[pyo3(signature = (diagonal, development = None, segment_map = None))]
    fn fit_diagonal(
        &self,
        py: Python<'_>,
        diagonal: PyRef<'_, PyLatestDiagonal>,
        development: Option<&Bound<'_, PyAny>>,
        segment_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyCapeCodFit> {
        let d = &diagonal.inner;
        let m = self.resolved(d.origin_grain())?;
        let source = diagonal_development(d, development, segment_map)?;
        let inner = py
            .detach(|| match &source {
                None => m.fit_diagonal_segments(d),
                Some(s) => m.fit_diagonal_segments_from(d, s),
            })
            .map_err(err)?;
        Ok(PyCapeCodFit { inner })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl_resolved!(PyCapeCod => CapeCod, chain_ladder);

impl PyCapeCod {
    fn base_repr(&self) -> String {
        format!(
            "CapeCod(trend={:?}, decay={:?}, {})",
            self.inner.trend,
            self.inner.decay,
            pattern_repr(&self.inner.chain_ladder)
        )
    }
}

/// A fitted expected-loss method (``ExpectedLoss``,
/// ``BornhuetterFerguson`` or ``Benktander``) of every segment of a
/// triangle column.
///
/// Per-origin lists (``origins``, ``latest``, ``exposure``, ``apriori``,
/// ``ultimate``, ``reserve``) run over the origins of each segment in turn,
/// like the rows of ``to_frame()``. ``ultimate`` and ``reserve`` are this
/// method's; ``chain_ladder`` holds the chain ladder's. Per-age lists need
/// a single-segment fit; for several segments use ``development_frame()``
/// or ``segment(...)``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BornhuetterFerguson, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021] * 2,
/// ...     [12, 24, 12] * 2,
/// ...     {"paid": [100.0, 150.0, 200.0, 10.0, 20.0, 30.0],
/// ...      "premium": [250.0, 250.0, 400.0, 500.0, 500.0, 800.0]},
/// ...     keys={"lob": ["Auto"] * 3 + ["Home"] * 3},
/// ... )
/// >>> fit = BornhuetterFerguson(apriori=0.5).fit(tri, "paid", "premium")
/// >>> fit.exposure, fit.segment(lob="Home").ultimate
/// ([250.0, 400.0, 500.0, 800.0], [20.0, 230.0])
#[pyclass(name = "ExpectedLossFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyExpectedLossFit {
    inner: SegmentFits<ExpectedLossFit>,
}

#[pymethods]
impl PyExpectedLossFit {
    /// The underlying chain-ladder projection, with the chain ladder's
    /// ultimate.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.map(|f| f.chain_ladder.clone()),
        }
    }

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
        origin_labels(&self.inner)
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.fits[0]
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// Age-to-age factors; factor ``k`` links age ``k`` to ``k + 1``.
    #[getter]
    fn ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "ldf", "development_frame()")?;
        Ok(f.chain_ladder.development.ldf.clone())
    }

    /// Age-to-ultimate factors, one per age, including the tail.
    #[getter]
    fn cdf(&self) -> PyResult<Vec<f64>> {
        Ok(single(&self.inner, "cdf", "development_frame()")?
            .chain_ladder
            .cdf
            .clone())
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.chain_ladder.latest.clone())
    }

    /// Exposure per origin: the exposure column's latest observed cumulative
    /// value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.exposure.clone())
    }

    /// Expected loss ratio applied per origin.
    #[getter]
    fn apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.apriori.clone())
    }

    /// This method's ultimate per origin.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    /// Reserve (ultimate minus latest) per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ExpectedLossFit::reserves)
    }

    /// Total ultimate across segments and origins.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    /// Total reserve across segments and origins.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``ultimate``, ``reserve``, ``exposure``, ``apriori`` and
    /// ``ulr``, the ultimate over the exposure (a loss ratio only when the
    /// exposure is premium). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns and the segment's total
    /// ``latest``, ``ultimate``, ``reserve``, ``exposure`` and ``ulr``, the
    /// total ultimate over the total exposure. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per segment and age, as ``ChainLadderFit.development_frame``.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``.
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ExpectedLossFit({}origins={}, total_ultimate={:?}, total_reserve={:?})",
            segments_prefix(&self.inner),
            self.origins().len(),
            self.inner.total_ultimate(),
            self.inner.total_reserve()
        )
    }
}

/// A fitted Cape Cod of every segment of a triangle column: the fields of
/// ``ExpectedLossFit``, with ``apriori`` the detrended loss ratio applied
/// to each origin (chainladder-python's ``detrended_apriori_``), plus
/// ``trended_apriori`` before detrending (its ``apriori_``).
///
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import CapeCod, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = CapeCod(trend=0.1).fit(tri, "paid", "premium")
/// >>> round(fit.trended_apriori[0] / fit.apriori[0], 10)
/// 1.1
#[pyclass(name = "CapeCodFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCapeCodFit {
    inner: SegmentFits<CapeCodFit>,
}

#[pymethods]
impl PyCapeCodFit {
    /// The expected-loss fit: ultimates, exposures and the detrended
    /// apriori.
    #[getter]
    fn expected_loss(&self) -> PyExpectedLossFit {
        PyExpectedLossFit {
            inner: self.inner.map(|f| f.expected_loss.clone()),
        }
    }

    /// The underlying chain-ladder projection, with the chain ladder's
    /// ultimate.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.map(|f| f.expected_loss.chain_ladder.clone()),
        }
    }

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
        origin_labels(&self.inner)
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.fits[0]
            .expected_loss
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// Age-to-age factors; factor ``k`` links age ``k`` to ``k + 1``.
    #[getter]
    fn ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "ldf", "development_frame()")?;
        Ok(f.expected_loss.chain_ladder.development.ldf.clone())
    }

    /// Age-to-ultimate factors, one per age, including the tail.
    #[getter]
    fn cdf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "cdf", "development_frame()")?;
        Ok(f.expected_loss.chain_ladder.cdf.clone())
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_loss.chain_ladder.latest.clone())
    }

    /// Exposure per origin: the exposure column's latest observed cumulative
    /// value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_loss.exposure.clone())
    }

    /// Detrended expected loss ratio applied per origin.
    #[getter]
    fn apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_loss.apriori.clone())
    }

    /// Expected loss ratio per origin at the valuation date's price level,
    /// before detrending. For a treaty year that incepts later, trend it
    /// one more year (or the years from the valuation date to the treaty
    /// year's average date) beyond this value.
    #[getter]
    fn trended_apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.trended_apriori.clone())
    }

    /// Cape Cod ultimate per origin.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_loss.ultimate.clone())
    }

    /// Reserve (ultimate minus latest) per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, CapeCodFit::reserves)
    }

    /// Total ultimate across segments and origins.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    /// Total reserve across segments and origins.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``ultimate``, ``reserve``, ``exposure``, ``apriori``,
    /// ``trended_apriori`` and ``ulr``, the ultimate over the exposure (a
    /// loss ratio only when the exposure is premium). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns and the segment's total
    /// ``latest``, ``ultimate``, ``reserve``, ``exposure`` and ``ulr``, the
    /// total ultimate over the total exposure. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per segment and age, as ``ChainLadderFit.development_frame``.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``.
    ///
    /// Returns
    /// -------
    /// CapeCodFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CapeCodFit({}origins={}, total_ultimate={:?}, total_reserve={:?})",
            segments_prefix(&self.inner),
            self.origins().len(),
            self.inner.total_ultimate(),
            self.inner.total_reserve()
        )
    }
}
