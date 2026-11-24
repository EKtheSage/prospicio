//! Growth curves: Clark's `ClarkLdf` and `ClarkCapeCod`, `BenchmarkCurve` and `DevelopmentCurve`, and their fits.

use super::*;

/// The growth curve named `name`.
fn growth_curve(name: &str) -> PyResult<GrowthCurve> {
    GrowthCurve::from_name(name)
        .ok_or_else(|| crate::unknown_name("growth curve", name, GrowthCurve::NAMES))
}

fn growth_curve_name(curve: GrowthCurve) -> &'static str {
    curve.name()
}

/// Clark's LDF method (Clark 2003), as R ChainLadder's ``ClarkLDF``: each
/// origin's expected ultimate and a growth curve are fitted to the
/// incremental losses by over-dispersed Poisson maximum likelihood, with
/// ages measured from the average date of loss (the middle of the origin
/// period, R's ``adol = TRUE``).
///
/// The ultimate is the latest value developed by the fitted curve to
/// ``max_age``. Process risk is the scale times the fitted reserve, and
/// parameter risk the delta method on the parameters' covariance, the
/// scale times the inverse Fisher information.
///
/// Parameters
/// ----------
/// curve : {"loglogistic", "weibull"}, default "loglogistic"
///     The growth curve ``G``: ``x**omega / (x**omega + theta**omega)`` or
///     ``1 - exp(-(x / theta)**omega)``.
/// max_age : float, optional
///     Age in months at which development stops; at least the triangle's
///     last age. ``None`` develops to infinity.
///
/// Raises
/// ------
/// ValueError
///     If ``curve`` is unknown.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ClarkLdf, Triangle
/// >>> rows = [[110.0, 290.0, 370.0, 420.0, 440.0], [95.0, 300.0, 390.0, 425.0],
/// ...         [130.0, 320.0, 410.0], [105.0, 305.0], [120.0]]
/// >>> tri = Triangle.from_long(
/// ...     [2020 + i for i, row in enumerate(rows) for _ in row],
/// ...     [12 * (d + 1) for row in rows for d in range(len(row))],
/// ...     [v for row in rows for v in row],
/// ... )
/// >>> fit = ClarkLdf(curve="weibull", max_age=120).fit(tri, "values")
/// >>> fit.omega > 0 and fit.total_standard_error > fit.total_process_risk
/// True
/// >>> round(fit.ultimate[2] * fit.growth(36) / fit.growth(120), 6)
/// 410.0
#[pyclass(name = "ClarkLdf", module = "prospicio.reserving", frozen)]
pub(crate) struct PyClarkLdf {
    inner: ClarkLdf,
}

#[pymethods]
impl PyClarkLdf {
    #[new]
    #[pyo3(signature = (curve = "loglogistic", max_age = None))]
    fn new(curve: &str, max_age: Option<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: ClarkLdf {
                curve: growth_curve(curve)?,
                max_age,
            },
        })
    }

    /// The growth curve.
    #[getter]
    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.curve)
    }

    /// Age in months at which development stops; ``None`` for infinity.
    #[getter]
    fn max_age(&self) -> Option<f64> {
        self.inner.max_age
    }

    /// Fits one loss column in every segment of a triangle, each on its
    /// own.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///
    /// Returns
    /// -------
    /// ClarkFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, if the triangle has fewer than four ages,
    ///     ``max_age`` is before its last age, an origin's latest value is
    ///     not positive, or the likelihood search does not converge.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyClarkFit> {
        let (m, tri) = (self.inner, &triangle.inner);
        let inner = py.detach(|| m.fit_segments(tri, column)).map_err(err)?;
        Ok(PyClarkFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ClarkLdf(curve={:?}, max_age={})",
            self.curve(),
            max_age_repr(self.inner.max_age)
        )
    }
}

fn max_age_repr(max_age: Option<f64>) -> String {
    max_age.map_or_else(|| "None".to_string(), |m| format!("{m:?}"))
}

/// `(age, value)` pairs of a dict from age in months (any positive number)
/// to a number, in the dict's order.
fn real_age_values(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<(f64, f64)>> {
    if obj.is_instance_of::<PyString>() || !obj.hasattr("items")? {
        return Err(PyTypeError::new_err(format!(
            "{what} must be a dict from age in months to a number, got {}",
            py_repr(obj)
        )));
    }
    let number = |x: &Bound<'_, PyAny>| -> Option<f64> {
        if x.is_instance_of::<PyBool>() {
            None
        } else {
            x.extract().ok()
        }
    };
    let mut out = Vec::new();
    for item in obj.call_method0("items")?.try_iter()? {
        let (key, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
        let age = number(&key).ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{what}: an age must be a number of months, got {}",
                py_repr(&key)
            ))
        })?;
        let value = number(&value).ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{what}: the value at age {age} must be a number, got {}",
                py_repr(&value)
            ))
        })?;
        out.push((age, value));
    }
    Ok(out)
}

/// A growth curve fitted to a benchmark pattern (issue #637; design note
/// ``reserving-v02.md``, decision 23): Clark's log-logistic or Weibull
/// curve ``G`` by weighted least squares on the percents developed, which
/// gives the pattern at any age, a smooth pattern between the benchmark's
/// ages and a tail past its last.
///
/// The fitted percent developed at age ``a`` is ``G(c(a)) / G(c(max_age))``
/// (``G(c(max_age)) = 1`` without ``max_age``), as Clark's LDF method
/// develops a latest value; ``c(a)`` is the age itself, or with
/// ``origin_width`` set Clark's age from the average date of loss, ``a -
/// w / 2`` at or past the width ``w`` and ``a / 2`` before it. Benchmark
/// ages are development ages from the start of the origin period, as a
/// triangle's.
///
/// Parameters
/// ----------
/// curve : {"loglogistic", "weibull"}, default "loglogistic"
///     The growth curve ``G``: ``x**omega / (x**omega + theta**omega)`` or
///     ``1 - exp(-(x / theta)**omega)``.
/// max_age : float, optional
///     Age in months at which development stops, so the pattern is 100%
///     there; at least the benchmark's last age. ``None`` develops to
///     infinity. A benchmark that reaches 100% needs it.
/// origin_width : float, optional
///     Measure the curve's ages from the middle of an origin period of
///     this many months (12 for accident years), as ``ClarkLdf`` does.
///     ``None`` measures them from the origin period's start.
///
/// Raises
/// ------
/// ValueError
///     If ``curve`` is unknown.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BenchmarkCurve, ChainLadder, Triangle
/// >>> bench = {12: 0.30, 24: 0.62, 36: 0.80, 48: 0.89, 60: 0.94}
/// >>> fit = BenchmarkCurve().fit(bench)
/// >>> all(abs(r) < 0.02 for r in fit.residuals)
/// True
/// >>> fit.percent_developed(30) > fit.percent_developed(24)
/// True
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2020, 2021, 2021, 2022], [12, 24, 36, 12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 165.0, 110.0, 170.0, 120.0]},
/// ... )
/// >>> cl = ChainLadder(**fit.selected([12, 24, 36])).fit(tri, "paid")
/// >>> abs(cl.pattern_tail - fit.cdf(36)) < 1e-12
/// True
#[pyclass(name = "BenchmarkCurve", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBenchmarkCurve {
    inner: BenchmarkCurve,
}

#[pymethods]
impl PyBenchmarkCurve {
    #[new]
    #[pyo3(signature = (curve = "loglogistic", max_age = None, origin_width = None))]
    fn new(curve: &str, max_age: Option<f64>, origin_width: Option<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: BenchmarkCurve {
                curve: growth_curve(curve)?,
                max_age,
                origin_width,
            },
        })
    }

    /// The growth curve.
    #[getter]
    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.curve)
    }

    /// Age in months at which development stops; ``None`` for infinity.
    #[getter]
    fn max_age(&self) -> Option<f64> {
        self.inner.max_age
    }

    /// The origin width of Clark's average-date shift; ``None`` for none.
    #[getter]
    fn origin_width(&self) -> Option<f64> {
        self.inner.origin_width
    }

    /// Fits the curve to a benchmark pattern.
    ///
    /// Parameters
    /// ----------
    /// pattern : dict
    ///     From development age in months (any positive number) to the
    ///     benchmark's value there, finite and positive; at least two ages.
    /// kind : {"percent_developed", "cdf"}, default "percent_developed"
    ///     Whether the values are percents developed or factors to ultimate.
    ///     Age-to-age factors (``"ldf"``) give no level to fit and are an
    ///     error.
    /// weights : dict, optional
    ///     From age in months to the weight, finite and not negative, of
    ///     that age's squared residual; an age not listed gets 1. ``None``
    ///     weights every age 1.
    ///
    /// Returns
    /// -------
    /// BenchmarkCurveFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a value, age or weight is out of range, an age is given
    ///     twice, a weight names an age the pattern does not give,
    ///     ``max_age`` is before the last age, or the search does not
    ///     converge.
    #[pyo3(signature = (pattern, kind = "percent_developed", weights = None))]
    fn fit(
        &self,
        pattern: &Bound<'_, PyAny>,
        kind: &str,
        weights: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyBenchmarkCurveFit> {
        let kind = match PatternKind::from_name(kind) {
            Some(PatternKind::AgeToAge) => {
                return Err(PyValueError::new_err(
                    "kind=\"ldf\": age-to-age factors give no level to fit; give percents \
                     developed or factors to ultimate",
                ));
            }
            Some(k) => k,
            None => return Err(crate::unknown_name("kind", kind, PatternKind::NAMES)),
        };
        let values = real_age_values(pattern, "pattern")?;
        let weights = match weights.filter(|w| !w.is_none()) {
            None => None,
            Some(w) => {
                let by_age = real_age_values(w, "weights")?;
                if let Some(&(age, _)) = by_age
                    .iter()
                    .find(|(a, _)| !values.iter().any(|(b, _)| b == a))
                {
                    return Err(PyValueError::new_err(format!(
                        "weights: age {age} is not an age of the pattern"
                    )));
                }
                Some(
                    values
                        .iter()
                        .map(|(a, _)| by_age.iter().find(|(b, _)| b == a).map_or(1.0, |w| w.1))
                        .collect::<Vec<f64>>(),
                )
            }
        };
        let inner = self
            .inner
            .fit(kind, &values, weights.as_deref())
            .map_err(err)?;
        Ok(PyBenchmarkCurveFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "BenchmarkCurve(curve={:?}, max_age={}, origin_width={})",
            self.curve(),
            max_age_repr(self.inner.max_age),
            max_age_repr(self.inner.origin_width)
        )
    }
}

/// A growth curve fitted to a benchmark pattern by ``BenchmarkCurve.fit``.
///
/// ``selected(ages)`` gives the fitted pattern at a triangle's ages as the
/// ``ldf``, ``ldf_kind`` and ``ldf_weight`` arguments of ``ChainLadder``,
/// ``Mack`` and the expected-loss methods. With no ``tail`` on the method
/// and ages ending at the triangle's oldest, the pattern's level there is
/// the tail: ``cdf(oldest)``, the curve's development past the oldest age.
#[pyclass(name = "BenchmarkCurveFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBenchmarkCurveFit {
    inner: BenchmarkCurveFit,
}

#[pymethods]
impl PyBenchmarkCurveFit {
    /// The growth curve.
    #[getter]
    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.curve)
    }

    /// Fitted shape of the curve.
    #[getter]
    fn omega(&self) -> f64 {
        self.inner.omega
    }

    /// Fitted scale of the curve, in months of curve age.
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }

    /// Age in months at which development stops; ``None`` for infinity.
    #[getter]
    fn max_age(&self) -> Option<f64> {
        self.inner.max_age
    }

    /// The origin width of Clark's average-date shift; ``None`` for none.
    #[getter]
    fn origin_width(&self) -> Option<f64> {
        self.inner.origin_width
    }

    /// The benchmark's ages in months, in increasing order.
    #[getter]
    fn ages(&self) -> Vec<f64> {
        self.inner.ages.clone()
    }

    /// The benchmark's percents developed at ``ages``.
    #[getter]
    fn observed(&self) -> Vec<f64> {
        self.inner.observed.clone()
    }

    /// The weight of each age in the least squares.
    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.inner.weights.clone()
    }

    /// The fitted percents developed at ``ages``.
    #[getter]
    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted.clone()
    }

    /// ``observed - fitted`` at ``ages``, unweighted.
    #[getter]
    fn residuals(&self) -> Vec<f64> {
        self.inner.residuals.clone()
    }

    /// The weighted sum of squared residuals, the minimized objective.
    #[getter]
    fn sum_of_squares(&self) -> f64 {
        self.inner.sum_of_squares
    }

    /// The curve age of a development age in months.
    fn curve_age(&self, age: f64) -> f64 {
        self.inner.curve_age(age)
    }

    /// The fitted percent developed at a development age in months: 1 at
    /// ``max_age`` and after.
    fn percent_developed(&self, age: f64) -> f64 {
        self.inner.percent_developed(age)
    }

    /// The fitted factor from a development age in months to ultimate,
    /// ``1 / percent_developed(age)``.
    fn cdf(&self, age: f64) -> f64 {
        self.inner.cdf(age)
    }

    /// The fitted pattern at development ``ages`` as the arguments of the
    /// chain ladder, Mack and the expected-loss methods.
    ///
    /// Parameters
    /// ----------
    /// ages : list of int
    ///     Development ages in months, usually the triangle's; each once
    ///     and positive.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``ldf`` (the percents developed by age), ``ldf_kind``
    ///     (``"percent_developed"``) and ``ldf_weight`` (1.0), to be passed
    ///     as keyword arguments. Give the method no ``tail``: the pattern's
    ///     level at the oldest age is the curve's tail.
    fn selected<'py>(&self, py: Python<'py>, ages: Vec<Lag>) -> PyResult<Bound<'py, PyDict>> {
        let pattern = self.inner.pattern(&ages).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("ldf", age_dict(py, &pattern.values)?)?;
        d.set_item("ldf_kind", pattern_kind_name(pattern.kind))?;
        d.set_item("ldf_weight", 1.0)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "BenchmarkCurveFit(curve={:?}, omega={}, theta={}, sum_of_squares={})",
            self.curve(),
            self.inner.omega,
            self.inner.theta,
            self.inner.sum_of_squares
        )
    }
}

/// A curve fitted to a triangle's age-to-age factors over every age (issue
/// #556; design note ``reserving-v02.md``, decision 31), to smooth the
/// whole pattern: the observed factors are kept before
/// ``keep_observed_until`` and the curve's are used from there on.
///
/// ``k`` is the 1-based development index (1 for the factor from the
/// youngest age to the next), as ``TailCurve`` reads it:
///
/// * ``"exponential"``: ``ln(f_k - 1) = a + b k``;
/// * ``"inverse_power"``: ``ln(f_k - 1) = a + b ln(k)``;
/// * ``"hoerl"``: ``ln(f_k - 1) = a + b k + c ln(k)``;
/// * ``"sherman"``: ``f_k - 1 = a (k + c)**(-b)`` (Sherman 1984);
/// * ``"loglogistic"``, ``"weibull"``: Clark's growth curve ``G`` as
///   percents developed at the ages ``t`` in months,
///   ``f_k = G(t_(k+1)) / G(t_k)``.
///
/// The first three are least-squares lines on ``ln(f - 1)``, the others
/// nonlinear least squares on the factors. Every shape is fitted to the
/// estimated factors in ``fit_period`` above 1.00001; the residuals, the sum
/// of squares and R-squared are on the factor scale, so that ``compare``
/// gives one table. The parameters match R's ``lm`` and ``nls``.
///
/// ``fit.selected()`` gives the fitted factors as the ``ldf``, ``ldf_kind``
/// and ``ldf_weight`` arguments of ``ChainLadder``, ``Mack`` and the
/// expected-loss methods, and ``fit.tail()`` the curve's development past
/// the oldest age for their ``tail``. A selected factor carries no
/// parameter error, so the curve lowers Mack's standard errors.
///
/// Parameters
/// ----------
/// shape : {"exponential", "inverse_power", "hoerl", "sherman", "loglogistic", "weibull"}, default "exponential"
/// fit_period : tuple of (int or None, int or None), default (None, None)
///     Ages in months whose factors enter the fit: from the last age at or
///     before the first (inclusive) to the last age at or before the second
///     (exclusive), as ``TailCurve`` reads them; ``None`` is open-ended.
/// keep_observed_until : int, optional
///     The observed factors are kept before this age in months and the
///     curve is used from the first age at or after it on; ``None`` uses
///     the curve at every age.
///
/// Raises
/// ------
/// ValueError
///     If ``shape`` is unknown.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, DevelopmentCurve, Triangle
/// >>> rows = [[100.0, 180.0, 215.0, 228.0, 233.0], [110.0, 196.0, 236.0, 249.0],
/// ...         [120.0, 218.0, 259.0], [130.0, 232.0], [140.0]]
/// >>> tri = Triangle.from_long(
/// ...     [2020 + i for i, row in enumerate(rows) for _ in row],
/// ...     [12 * (d + 1) for row in rows for d in range(len(row))],
/// ...     [v for row in rows for v in row],
/// ... )
/// >>> fit = DevelopmentCurve(shape="inverse_power").fit(tri, "values")
/// >>> fit.r_squared > 0.9
/// True
/// >>> cl = ChainLadder(**fit.selected(), tail=fit.tail()).fit(tri, "values")
/// >>> cl.ldf[:4] == fit.fitted
/// True
/// >>> table = DevelopmentCurve().compare(tri, "values", ["exponential", "inverse_power"])
/// >>> table["shape"]
/// ['exponential', 'inverse_power']
#[pyclass(name = "DevelopmentCurve", module = "prospicio.reserving", frozen)]
pub(crate) struct PyDevelopmentCurve {
    inner: DevelopmentCurve,
}

/// The estimated development of a single-segment triangle's column, for a
/// curve to be fitted to.
fn curve_development(
    triangle: &Triangle,
    column: &str,
    average_: &str,
    n_periods: Option<usize>,
    drop_high: usize,
    drop_low: usize,
) -> PyResult<prospicio_reserving::DevelopmentFit> {
    development(average_, "log-linear", None, n_periods, drop_high, drop_low)?
        .fit(triangle, column)
        .map_err(err)
}

#[pymethods]
impl PyDevelopmentCurve {
    #[new]
    #[pyo3(signature = (shape = "exponential", fit_period = (None, None), keep_observed_until = None))]
    fn new(
        shape: &str,
        fit_period: (Option<Lag>, Option<Lag>),
        keep_observed_until: Option<Lag>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: DevelopmentCurve {
                shape: curve_shape(shape, "shape")?,
                fit_period,
                keep_observed_until,
            },
        })
    }

    /// The curve fitted to the factors.
    #[getter]
    fn shape(&self) -> &'static str {
        curve_name(self.inner.shape)
    }

    /// Ages whose factors enter the fit, from (inclusive) and to
    /// (exclusive).
    #[getter]
    fn fit_period(&self) -> (Option<Lag>, Option<Lag>) {
        self.inner.fit_period
    }

    /// Age in months from which the curve replaces the observed factors;
    /// ``None`` for every age.
    #[getter]
    fn keep_observed_until(&self) -> Option<Lag> {
        self.inner.keep_observed_until
    }

    /// Fits the curve to the estimated factors of a single-segment
    /// triangle's column.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative or incremental, with one segment.
    /// column : str
    /// average : {"volume", "simple", "regression", "geometric"}, default "volume"
    ///     How the link ratios are averaged into the factors the curve is
    ///     fitted to, as ``ChainLadder``'s.
    /// n_periods : int, optional
    /// drop_high : int, default 0
    /// drop_low : int, default 0
    ///     The link ratio exclusions, as ``ChainLadder``'s.
    ///
    /// Returns
    /// -------
    /// DevelopmentCurveFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the column is unknown, a factor cannot be estimated, the fit
    ///     period holds too few factors above 1.00001 for the shape (two for
    ///     the exponential and inverse power, three for Hoerl and the growth
    ///     curves, four for Sherman), a nonlinear fit does not converge or
    ///     ``keep_observed_until`` is past the last factor's age.
    #[pyo3(signature = (triangle, column, average = "volume", n_periods = None, drop_high = 0, drop_low = 0))]
    fn fit(
        &self,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        average: &str,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
    ) -> PyResult<PyDevelopmentCurveFit> {
        let dev = curve_development(
            &triangle.inner,
            column,
            average,
            n_periods,
            drop_high,
            drop_low,
        )?;
        let inner = self.inner.fit(&dev).map_err(err)?;
        Ok(PyDevelopmentCurveFit { inner })
    }

    /// Fits each of ``shapes``, with this curve's ``fit_period`` and
    /// ``keep_observed_until``, to the same factors: a table of fit
    /// statistics, one row per shape.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    /// shapes : list of str, optional
    ///     The shapes to compare; every shape by default.
    /// average, n_periods, drop_high, drop_low
    ///     As in ``fit``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Columns ``shape``, ``n_points``, ``n_parameters``,
    ///     ``sum_of_squares``, ``r_squared`` and ``adjusted_r_squared``, a
    ///     list each (``pandas.DataFrame(table)`` makes a frame of it).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``, for the first shape that cannot be fitted.
    #[pyo3(signature = (triangle, column, shapes = None, average = "volume", n_periods = None, drop_high = 0, drop_low = 0))]
    #[allow(clippy::too_many_arguments)]
    fn compare<'py>(
        &self,
        py: Python<'py>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        shapes: Option<Vec<String>>,
        average: &str,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let shapes = match shapes {
            Some(names) => names
                .iter()
                .map(|s| curve_shape(s, "shapes"))
                .collect::<PyResult<Vec<_>>>()?,
            None => ALL_CURVE_SHAPES.to_vec(),
        };
        let dev = curve_development(
            &triangle.inner,
            column,
            average,
            n_periods,
            drop_high,
            drop_low,
        )?;
        let rows = self.inner.compare(&dev, &shapes).map_err(err)?;
        let d = PyDict::new(py);
        let names: Vec<&str> = rows.iter().map(|r| curve_name(r.curve.shape)).collect();
        d.set_item("shape", names)?;
        let column = |f: fn(&DevelopmentCurveFit) -> f64| rows.iter().map(f).collect::<Vec<_>>();
        let count = |f: fn(&DevelopmentCurveFit) -> usize| rows.iter().map(f).collect::<Vec<_>>();
        d.set_item("n_points", count(|r| r.n_points))?;
        d.set_item("n_parameters", count(|r| r.n_parameters))?;
        d.set_item("sum_of_squares", column(|r| r.sum_of_squares))?;
        d.set_item("r_squared", column(|r| r.r_squared))?;
        d.set_item("adjusted_r_squared", column(|r| r.adjusted_r_squared))?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        let age = |a: Option<Lag>| a.map_or("None".to_string(), |a| a.to_string());
        format!(
            "DevelopmentCurve(shape={:?}, fit_period=({}, {}), keep_observed_until={})",
            self.shape(),
            age(self.inner.fit_period.0),
            age(self.inner.fit_period.1),
            age(self.inner.keep_observed_until)
        )
    }
}

/// A curve fitted by ``DevelopmentCurve.fit``.
///
/// Per development position (factor ``k`` links ``development[k]`` to
/// ``development[k + 1]``): ``observed`` (the triangle's estimated factors,
/// NaN where one is missing), ``fitted`` (the curve's), ``residuals``
/// (``observed - fitted``), ``in_fit`` (whether the factor entered the fit)
/// and ``ldf`` (``observed`` before ``attachment``, ``fitted`` from it on).
#[pyclass(name = "DevelopmentCurveFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyDevelopmentCurveFit {
    inner: DevelopmentCurveFit,
}

#[pymethods]
impl PyDevelopmentCurveFit {
    /// The curve fitted to the factors.
    #[getter]
    fn shape(&self) -> &'static str {
        curve_name(self.inner.curve.shape)
    }

    /// Ages whose factors entered the fit, from (inclusive) and to
    /// (exclusive).
    #[getter]
    fn fit_period(&self) -> (Option<Lag>, Option<Lag>) {
        self.inner.curve.fit_period
    }

    /// Age in months from which the curve replaces the observed factors.
    #[getter]
    fn keep_observed_until(&self) -> Option<Lag> {
        self.inner.curve.keep_observed_until
    }

    /// The curve's parameters: ``[a, b]`` for the exponential and inverse
    /// power, ``[a, b, c]`` for Hoerl and Sherman, ``[omega, theta]`` for a
    /// growth curve.
    #[getter]
    fn parameters(&self) -> Vec<f64> {
        self.inner.parameters.clone()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.development.clone()
    }

    /// The triangle's estimated factors, NaN where one is missing.
    #[getter]
    fn observed(&self) -> Vec<f64> {
        self.inner.observed.clone()
    }

    /// The curve's factor at every development position.
    #[getter]
    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted.clone()
    }

    /// ``observed - fitted`` at every development position.
    #[getter]
    fn residuals(&self) -> Vec<f64> {
        self.inner.residuals.clone()
    }

    /// Whether each position's factor entered the fit.
    #[getter]
    fn in_fit(&self) -> Vec<bool> {
        self.inner.in_fit.clone()
    }

    /// The first position the curve replaces.
    #[getter]
    fn attachment(&self) -> usize {
        self.inner.attachment
    }

    /// The selected factors: ``observed`` before ``attachment``,
    /// ``fitted`` from it on.
    #[getter]
    fn ldf(&self) -> Vec<f64> {
        self.inner.ldf.clone()
    }

    /// The number of points in the fit.
    #[getter]
    fn n_points(&self) -> usize {
        self.inner.n_points
    }

    /// The number of parameters of the curve.
    #[getter]
    fn n_parameters(&self) -> usize {
        self.inner.n_parameters
    }

    /// The sum of squared residuals over the points of the fit, on the
    /// factor scale.
    #[getter]
    fn sum_of_squares(&self) -> f64 {
        self.inner.sum_of_squares
    }

    /// ``1 - SSE / SST`` over the points of the fit.
    #[getter]
    fn r_squared(&self) -> f64 {
        self.inner.r_squared
    }

    /// R-squared adjusted for the number of parameters; NaN with no more
    /// points than parameters.
    #[getter]
    fn adjusted_r_squared(&self) -> f64 {
        self.inner.adjusted_r_squared
    }

    /// The curve's factor at a 0-based development position, past the
    /// oldest age too.
    fn factor(&self, position: usize) -> f64 {
        self.inner.factor(position)
    }

    /// The curve's development past the oldest age: the product of its
    /// factors over the ``extrap_periods`` periods after it, as
    /// ``TailCurve`` extrapolates.
    #[pyo3(signature = (extrap_periods = 100))]
    fn tail(&self, extrap_periods: usize) -> f64 {
        self.inner.tail(extrap_periods)
    }

    /// The fitted factors from ``attachment`` on as the arguments of the
    /// chain ladder, Mack and the expected-loss methods.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``ldf`` (the age-to-age factors by age), ``ldf_kind`` (``"ldf"``)
    ///     and ``ldf_weight`` (1.0), to be passed as keyword arguments, with
    ///     ``tail=fit.tail()`` for the curve's development past the oldest
    ///     age.
    fn selected<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let pattern = self.inner.pattern();
        let d = PyDict::new(py);
        d.set_item("ldf", age_dict(py, &pattern.values)?)?;
        d.set_item("ldf_kind", pattern_kind_name(pattern.kind))?;
        d.set_item("ldf_weight", 1.0)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "DevelopmentCurveFit(shape={:?}, parameters={:?}, r_squared={})",
            self.shape(),
            self.inner.parameters,
            self.inner.r_squared
        )
    }
}

/// Every curve shape, in the order `DevelopmentCurve.compare` lists them.
const ALL_CURVE_SHAPES: [CurveShape; 6] = [
    CurveShape::Exponential,
    CurveShape::InversePower,
    CurveShape::Hoerl,
    CurveShape::Sherman,
    CurveShape::Growth(GrowthCurve::LogLogistic),
    CurveShape::Growth(GrowthCurve::Weibull),
];

/// A curve shape by its name in the bindings.
pub(crate) fn curve_shape(name: &str, what: &str) -> PyResult<CurveShape> {
    CurveShape::from_name(name).ok_or_else(|| crate::unknown_name(what, name, CurveShape::NAMES))
}

/// Clark's Cape Cod method (Clark 2003), as R ChainLadder's
/// ``ClarkCapeCod``: one expected loss ratio times each origin's exposure
/// and a growth curve are fitted to the incremental losses by
/// over-dispersed Poisson maximum likelihood, with ages measured from the
/// average date of loss.
///
/// The reserve is the fitted ``elr * exposure * (G(max_age) - G(age))``;
/// process and parameter risk are as in ``ClarkLdf``.
///
/// Parameters
/// ----------
/// curve : {"loglogistic", "weibull"}, default "loglogistic"
///     The growth curve, as in ``ClarkLdf``.
/// max_age : float, optional
///     Age in months at which development stops; at least the triangle's
///     last age. ``None`` develops to infinity.
///
/// Raises
/// ------
/// ValueError
///     If ``curve`` is unknown.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ClarkCapeCod, Triangle
/// >>> rows = [[110.0, 290.0, 370.0, 420.0, 440.0], [95.0, 300.0, 390.0, 425.0],
/// ...         [130.0, 320.0, 410.0], [105.0, 305.0], [120.0]]
/// >>> tri = Triangle.from_long(
/// ...     [2020 + i for i, row in enumerate(rows) for _ in row],
/// ...     [12 * (d + 1) for row in rows for d in range(len(row))],
/// ...     {"paid": [v for row in rows for v in row],
/// ...      "premium": [800.0 for row in rows for _ in row]},
/// ... )
/// >>> fit = ClarkCapeCod().fit(tri, "paid", "premium")
/// >>> 0 < fit.elr < 1 and fit.expected_ultimate == [fit.elr * 800.0] * 5
/// True
#[pyclass(name = "ClarkCapeCod", module = "prospicio.reserving", frozen)]
pub(crate) struct PyClarkCapeCod {
    inner: ClarkCapeCod,
}

#[pymethods]
impl PyClarkCapeCod {
    #[new]
    #[pyo3(signature = (curve = "loglogistic", max_age = None))]
    fn new(curve: &str, max_age: Option<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: ClarkCapeCod {
                curve: growth_curve(curve)?,
                max_age,
            },
        })
    }

    /// The growth curve.
    #[getter]
    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.curve)
    }

    /// Age in months at which development stops; ``None`` for infinity.
    #[getter]
    fn max_age(&self) -> Option<f64> {
        self.inner.max_age
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
    ///     The exposure column; each origin's latest observed value is its
    ///     exposure.
    ///
    /// Returns
    /// -------
    /// ClarkFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ClarkLdf.fit``, and if an origin has no observed, finite,
    ///     positive exposure.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyClarkFit> {
        let (m, tri) = (self.inner, &triangle.inner);
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyClarkFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ClarkCapeCod(curve={:?}, max_age={})",
            self.curve(),
            max_age_repr(self.inner.max_age)
        )
    }
}

/// A fitted Clark LDF or Cape Cod model of every segment of a triangle
/// column.
///
/// Per-origin lists (``origins``, ``latest``, ``expected_ultimate``,
/// ``ultimate``, ``reserve`` and the standard errors) run over the origins
/// of each segment in turn, like the rows of ``to_frame()``. The fitted
/// parameters and the standard errors of the total need a single-segment
/// fit; for several segments use ``totals_frame()`` or ``segment(...)``.
/// ``total_ultimate`` and ``total_reserve`` sum over every segment.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ClarkLdf, Triangle
/// >>> rows = [[110.0, 290.0, 370.0, 420.0, 440.0], [95.0, 300.0, 390.0, 425.0],
/// ...         [130.0, 320.0, 410.0], [105.0, 305.0], [120.0]]
/// >>> tri = Triangle.from_long(
/// ...     [2020 + i for i, row in enumerate(rows) for _ in row],
/// ...     [12 * (d + 1) for row in rows for d in range(len(row))],
/// ...     [v for row in rows for v in row],
/// ... )
/// >>> fit = ClarkLdf().fit(tri, "values")
/// >>> len(fit.covariance), fit.elr, fit.growth(float("inf"))
/// (7, None, 1.0)
#[pyclass(name = "ClarkFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyClarkFit {
    inner: SegmentFits<ClarkFit>,
}

impl PyClarkFit {
    /// The fit of a single-segment result, for `field`.
    fn one(&self, field: &str) -> PyResult<&ClarkFit> {
        single(&self.inner, field, "totals_frame()")
    }
}

#[pymethods]
impl PyClarkFit {
    /// The volume-weighted chain ladder of the same column, with the chain
    /// ladder's ultimate.
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

    /// The growth curve.
    #[getter]
    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.fits[0].curve)
    }

    /// Age in months at which development stops; ``None`` for infinity.
    #[getter]
    fn max_age(&self) -> Option<f64> {
        self.inner.fits[0].max_age
    }

    /// Fitted shape of the growth curve.
    #[getter]
    fn omega(&self) -> PyResult<f64> {
        Ok(self.one("omega")?.omega)
    }

    /// Fitted scale of the growth curve, in months.
    #[getter]
    fn theta(&self) -> PyResult<f64> {
        Ok(self.one("theta")?.theta)
    }

    /// Expected loss ratio (Cape Cod), or ``None`` (LDF).
    #[getter]
    fn elr(&self) -> PyResult<Option<f64>> {
        if self.inner.fits[0].elr.is_none() {
            return Ok(None);
        }
        Ok(self.one("elr")?.elr)
    }

    /// Length of the origin period in months; ages are shifted by half of
    /// it to the average date of loss.
    #[getter]
    fn origin_width(&self) -> f64 {
        self.inner.fits[0].origin_width
    }

    /// Number of observed incremental values fitted; ``scale`` divides by
    /// this less the number of parameters.
    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        Ok(single(&self.inner, "n_observations", "segment(...)")?.n_observations)
    }

    /// Over-dispersion ``sigma**2``: squared Pearson residuals over the
    /// observed incremental values less the number of parameters.
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        Ok(self.one("scale")?.scale)
    }

    /// Covariance of the parameters: the expected ultimates (LDF) or the
    /// expected loss ratio (Cape Cod), then ``omega`` and ``theta``. NaN if
    /// the Fisher information is singular.
    #[getter]
    fn covariance(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(single(&self.inner, "covariance", "segment(...)")?
            .covariance
            .clone())
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.chain_ladder.latest.clone())
    }

    /// Exposure per origin (Cape Cod), or ``None`` (LDF).
    #[getter]
    fn exposure(&self) -> Option<Vec<f64>> {
        self.inner.fits[0].exposure.as_ref()?;
        Some(by_origin(&self.inner, |f| {
            f.exposure.clone().unwrap_or_default()
        }))
    }

    /// Expected ultimate per origin, developed to infinity: fitted (LDF)
    /// or ``elr * exposure`` (Cape Cod).
    #[getter]
    fn expected_ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_ultimate.clone())
    }

    /// Ultimate per origin: the latest value plus the reserve.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    /// Reserve per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ClarkFit::reserves)
    }

    /// Process standard error per origin.
    #[getter]
    fn process_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.process_risk.clone())
    }

    /// Parameter standard error per origin.
    #[getter]
    fn parameter_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.parameter_risk.clone())
    }

    /// Standard error per origin: ``sqrt(process**2 + parameter**2)``.
    #[getter]
    fn standard_error(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.standard_error.clone())
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

    /// Process standard error of the total reserve.
    #[getter]
    fn total_process_risk(&self) -> PyResult<f64> {
        Ok(self.one("total_process_risk")?.total_process_risk)
    }

    /// Parameter standard error of the total reserve, with the covariance
    /// between origins.
    #[getter]
    fn total_parameter_risk(&self) -> PyResult<f64> {
        Ok(self.one("total_parameter_risk")?.total_parameter_risk)
    }

    /// Standard error of the total reserve.
    #[getter]
    fn total_standard_error(&self) -> PyResult<f64> {
        Ok(self.one("total_standard_error")?.total_standard_error)
    }

    /// Share of the expected ultimate developed by a development age.
    ///
    /// Parameters
    /// ----------
    /// age : float
    ///     Development age in months, before the shift to the average date
    ///     of loss; ``inf`` gives 1.
    ///
    /// Returns
    /// -------
    /// float
    fn growth(&self, age: f64) -> PyResult<f64> {
        Ok(single(&self.inner, "growth", "segment(...)")?.growth(age))
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``ultimate``, ``reserve``, (Cape Cod) ``exposure``,
    /// ``expected_ultimate``, ``process_risk``, ``parameter_risk``,
    /// ``standard_error`` and (Cape Cod) ``ulr``, the ultimate over the
    /// exposure (a loss ratio only when the exposure is premium; the
    /// fitted ``elr`` is a total). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the segment's total
    /// ``latest``, ``ultimate`` and ``reserve``, the ``process_risk``,
    /// ``parameter_risk`` and ``standard_error`` of its total reserve, and
    /// its ``omega``, ``theta``, ``scale`` and (Cape Cod) ``elr`` and
    /// ``ulr``, the total ultimate over the total exposure. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``.
    ///
    /// Returns
    /// -------
    /// ClarkFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        let method = if self.inner.fits[0].elr.is_some() {
            "cape_cod"
        } else {
            "ldf"
        };
        let se = match self.inner.fits.as_slice() {
            [one] => format!(", total_standard_error={:?}", one.total_standard_error),
            _ => String::new(),
        };
        format!(
            "ClarkFit({}method={method:?}, curve={:?}, origins={}, total_reserve={:?}{se})",
            segments_prefix(&self.inner),
            self.curve(),
            self.origins().len(),
            self.inner.total_reserve(),
        )
    }
}
