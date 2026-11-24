//! The tails: `TailConstant`, `TailCurve`, `TailBondy` and `TailLogLinear`, and the `tail=` argument.

use super::*;

/// A given tail factor, as chainladder-python's ``TailConstant``.
///
/// The factor applies from the attachment age to ultimate. Past the
/// attachment it is spread over the following periods as
/// ``1 + x * decay**k``, the last factor making up the difference; this
/// shapes the factors past the attachment, not the factor to ultimate. An
/// attachment before the oldest age replaces the estimated factors from
/// there.
///
/// Parameters
/// ----------
/// factor : float, default 1.0
///     Factor from the attachment age to ultimate; finite and positive.
/// decay : float, default 0.5
///     Share of each period's development kept in the next, from 0 to 1.
/// attachment_age : int, optional
///     Age in months the factor attaches at (the first age at or after
///     it); the oldest age by default. An age at or before the youngest
///     replaces every estimated factor (chainladder-python ignores such an
///     attachment).
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, TailConstant, Triangle
/// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 200.0]})
/// >>> fit = ChainLadder(tail=TailConstant(1.05)).fit(tri, "paid")
/// >>> fit.tail, round(fit.ultimate[1], 6)
/// (1.05, 315.0)
#[pyclass(name = "TailConstant", module = "prospicio.reserving", frozen)]
pub(crate) struct PyTailConstant {
    inner: TailConstant,
}

#[pymethods]
impl PyTailConstant {
    #[new]
    #[pyo3(signature = (factor = 1.0, decay = 0.5, attachment_age = None))]
    fn new(factor: f64, decay: f64, attachment_age: Option<Lag>) -> Self {
        Self {
            inner: TailConstant {
                factor,
                decay,
                attachment_age,
            },
        }
    }

    /// Factor from the attachment age to ultimate.
    #[getter]
    fn factor(&self) -> f64 {
        self.inner.factor
    }

    /// Share of each period's development kept in the next.
    #[getter]
    fn decay(&self) -> f64 {
        self.inner.decay
    }

    /// Age in months the factor attaches at; ``None`` is the oldest age.
    #[getter]
    fn attachment_age(&self) -> Option<Lag> {
        self.inner.attachment_age
    }

    fn __repr__(&self) -> String {
        tail_repr(&Tail::Constant(self.inner))
    }
}

/// A curve fitted to the estimated factors and extrapolated, as
/// chainladder-python's ``TailCurve``.
///
/// Factors above 1.00001 in the fit period are regressed by least squares:
/// ``ln(f - 1)`` on the 1-based development index ``k`` (exponential) or on
/// ``ln(k)`` (inverse power). The fitted curve replaces the factors from the
/// attachment age on and runs ``extrap_periods`` periods past the oldest
/// age. The Hoerl, Sherman and growth shapes are fitted to the same factors
/// as ``DevelopmentCurve`` fits them; chainladder-python has only the first
/// two.
///
/// Parameters
/// ----------
/// curve : {"exponential", "inverse_power", "hoerl", "sherman", "loglogistic", "weibull"}, default "exponential"
/// fit_period : tuple of (int or None, int or None), default (None, None)
///     Ages in months whose factors enter the fit: from the last age at or
///     before the first (inclusive) to the last age at or before the second
///     (exclusive), as chainladder-python reads them; ``None`` is
///     open-ended.
/// extrap_periods : int, default 100
///     Number of periods past the oldest age the curve is extrapolated.
/// attachment_age : int, optional
///     Age in months the curve attaches at (the first age at or after it);
///     the oldest age by default.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, TailCurve, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = ChainLadder(tail=TailCurve()).fit(tri, "values")
/// >>> 1.0 < fit.tail < 1.05
/// True
#[pyclass(name = "TailCurve", module = "prospicio.reserving", frozen)]
pub(crate) struct PyTailCurve {
    inner: TailCurve,
}

#[pymethods]
impl PyTailCurve {
    #[new]
    #[pyo3(signature = (curve = "exponential", fit_period = (None, None), extrap_periods = 100, attachment_age = None))]
    fn new(
        curve: &str,
        fit_period: (Option<Lag>, Option<Lag>),
        extrap_periods: usize,
        attachment_age: Option<Lag>,
    ) -> PyResult<Self> {
        let curve = curve_shape(curve, "curve")?;
        Ok(Self {
            inner: TailCurve {
                curve,
                fit_period,
                extrap_periods,
                attachment_age,
            },
        })
    }

    /// The curve fitted to ``f - 1``.
    #[getter]
    fn curve(&self) -> &'static str {
        curve_name(self.inner.curve)
    }

    /// Ages whose factors enter the fit, from (inclusive) and to
    /// (exclusive).
    #[getter]
    fn fit_period(&self) -> (Option<Lag>, Option<Lag>) {
        self.inner.fit_period
    }

    /// Number of periods past the oldest age the curve is extrapolated.
    #[getter]
    fn extrap_periods(&self) -> usize {
        self.inner.extrap_periods
    }

    /// Age in months the curve attaches at; ``None`` is the oldest age.
    #[getter]
    fn attachment_age(&self) -> Option<Lag> {
        self.inner.attachment_age
    }

    fn __repr__(&self) -> String {
        tail_repr(&Tail::Curve(self.inner))
    }
}

/// The Bondy tail, as chainladder-python's ``TailBondy``.
///
/// Each log factor from ``earliest_age`` on is taken as ``b`` times the one
/// before it, ``b`` fitted by least squares. The fitted factors are
/// ``f0 ** (b ** j)`` from the factor ``f0`` at ``earliest_age``, and those
/// past the next one multiply to the last fitted factor raised to
/// ``b / (1 - b)``. With the default ``earliest_age`` (the age of the last
/// factor) ``b`` is 1/2 and the tail repeats the last factor.
///
/// Parameters
/// ----------
/// earliest_age : int, optional
///     First age in months whose factor enters the fit (the last age at or
///     before it, as chainladder-python reads it); the age of the last
///     factor by default.
/// attachment_age : int, optional
///     The factor from this age (the last age at or before it) to the next
///     is kept and the fitted ones replace those after it; the age of the
///     last factor by default. Not before ``earliest_age``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, TailBondy, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2020, 2021, 2021, 2022],
/// ...     [12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 110.0, 170.0, 120.0],
/// ... )
/// >>> round(ChainLadder(tail=TailBondy()).fit(tri, "values").tail, 12)
/// 1.1
#[pyclass(name = "TailBondy", module = "prospicio.reserving", frozen)]
pub(crate) struct PyTailBondy {
    inner: TailBondy,
}

#[pymethods]
impl PyTailBondy {
    #[new]
    #[pyo3(signature = (earliest_age = None, attachment_age = None))]
    fn new(earliest_age: Option<Lag>, attachment_age: Option<Lag>) -> Self {
        Self {
            inner: TailBondy {
                earliest_age,
                attachment_age,
            },
        }
    }

    /// First age whose factor enters the fit; ``None`` is the age of the
    /// last factor.
    #[getter]
    fn earliest_age(&self) -> Option<Lag> {
        self.inner.earliest_age
    }

    /// Age after which the fitted factors replace the estimated ones;
    /// ``None`` is the age of the last factor.
    #[getter]
    fn attachment_age(&self) -> Option<Lag> {
        self.inner.attachment_age
    }

    fn __repr__(&self) -> String {
        tail_repr(&Tail::Bondy(self.inner))
    }
}

/// R ChainLadder's ``tail = TRUE`` rule (its ``tailfactor`` function).
///
/// When the third- and second-last factors multiply to more than 1.0001,
/// ``ln(f - 1)`` is regressed on the development index over the factors
/// above 1 and the next 100 extrapolated factors are multiplied; otherwise
/// the tail is 1. A tail above 2 is reset to 1, as R does.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Mack, TailLogLinear, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = Mack(tail=TailLogLinear()).fit(tri, "values")
/// >>> fit.tail > 1.0 and fit.standard_error[0] > 0.0
/// True
#[pyclass(name = "TailLogLinear", module = "prospicio.reserving", frozen)]
pub(crate) struct PyTailLogLinear;

#[pymethods]
impl PyTailLogLinear {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        tail_repr(&Tail::LogLinear)
    }
}

pub(crate) fn curve_name(c: CurveShape) -> &'static str {
    c.name()
}

/// A constant tail with the default decay and attachment, shown as its
/// factor.
fn plain_factor(tail: &Tail) -> Option<f64> {
    match tail {
        Tail::Constant(t) if t.decay == 0.5 && t.attachment_age.is_none() => Some(t.factor),
        _ => None,
    }
}

fn tail_repr(tail: &Tail) -> String {
    let age = |a: Option<Lag>| a.map_or("None".to_string(), |a| a.to_string());
    match tail {
        Tail::Constant(t) => format!(
            "TailConstant(factor={:?}, decay={:?}, attachment_age={})",
            t.factor,
            t.decay,
            age(t.attachment_age)
        ),
        Tail::Curve(t) => format!(
            "TailCurve(curve={:?}, fit_period=({}, {}), extrap_periods={}, attachment_age={})",
            curve_name(t.curve),
            age(t.fit_period.0),
            age(t.fit_period.1),
            t.extrap_periods,
            age(t.attachment_age)
        ),
        Tail::Bondy(t) => format!(
            "TailBondy(earliest_age={}, attachment_age={})",
            age(t.earliest_age),
            age(t.attachment_age)
        ),
        Tail::LogLinear => "TailLogLinear()".to_string(),
    }
}

/// The ``tail`` argument of ``ChainLadder``, ``Mack`` and the
/// expected-loss methods: ``None`` (no tail), a number (a constant factor)
/// or a tail estimator.
pub(crate) fn tail_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Tail> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(Tail::default());
    };
    if let Ok(t) = obj.extract::<PyRef<'_, PyTailConstant>>() {
        return Ok(Tail::Constant(t.inner));
    }
    if let Ok(t) = obj.extract::<PyRef<'_, PyTailCurve>>() {
        return Ok(Tail::Curve(t.inner));
    }
    if let Ok(t) = obj.extract::<PyRef<'_, PyTailBondy>>() {
        return Ok(Tail::Bondy(t.inner));
    }
    if obj.extract::<PyRef<'_, PyTailLogLinear>>().is_ok() {
        return Ok(Tail::LogLinear);
    }
    if !obj.is_instance_of::<PyBool>()
        && let Ok(factor) = obj.extract::<f64>()
    {
        return Ok(factor.into());
    }
    Err(PyTypeError::new_err(format!(
        "tail must be a number, TailConstant, TailCurve, TailBondy or TailLogLinear, got {}",
        obj.repr()?
    )))
}

/// The ``tail`` of a method as Python sees it: a plain
/// constant as its factor, any other tail as its estimator.
pub(crate) fn tail_object<'py>(py: Python<'py>, tail: &Tail) -> PyResult<Bound<'py, PyAny>> {
    if let Some(factor) = plain_factor(tail) {
        return Ok(PyFloat::new(py, factor).into_any());
    }
    Ok(match *tail {
        Tail::Constant(inner) => Bound::new(py, PyTailConstant { inner })?.into_any(),
        Tail::Curve(inner) => Bound::new(py, PyTailCurve { inner })?.into_any(),
        Tail::Bondy(inner) => Bound::new(py, PyTailBondy { inner })?.into_any(),
        Tail::LogLinear => Bound::new(py, PyTailLogLinear)?.into_any(),
    })
}

/// ``repr`` of a ``tail`` argument: a plain constant as its factor.
pub(crate) fn tail_arg_repr(tail: &Tail) -> String {
    plain_factor(tail).map_or_else(|| tail_repr(tail), |f| format!("{f:?}"))
}
