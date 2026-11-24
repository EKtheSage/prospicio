//! `prospicio.reserving` indexation by a calendar index and re-inflation of
//! a projection (issue #979, `docs/design/reserving-inflation.md`, PR 2),
//! over `prospicio_reserving::inflation`.
//!
//! `FutureCells.from_chain_ladder` takes a `ChainLadderFit` and, for a fit
//! over several segments, the key values of the one to use.

use prospicio_reserving::{
    CalendarIndex, FutureCell, FutureCells, Grain, Indexation, Month, Period, Reinflated,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyString};

use crate::odp_glm::PyOdpGlmFit;
use crate::reserving::{PyChainLadderFit, PyTriangle, err, pick, py_repr};

/// A period as a label: `"2020"`, `"2020H2"`, `"2020Q3"` or `"2020-07"`.
fn parse_period(text: &str) -> Option<Period> {
    let text = text.trim();
    let num = |s: &str| s.parse::<i32>().ok();
    let (grain, year, within) = if let Some((y, q)) = text.split_once('Q') {
        (
            Grain::Quarter,
            num(y)?,
            (num(q).filter(|q| (1..=4).contains(q))? - 1) * 3 + 1,
        )
    } else if let Some((y, h)) = text.split_once('H') {
        (
            Grain::Semester,
            num(y)?,
            (num(h).filter(|h| (1..=2).contains(h))? - 1) * 6 + 1,
        )
    } else if let Some((y, m)) = text.split_once('-') {
        (
            Grain::Month,
            num(y)?,
            num(m).filter(|m| (1..=12).contains(m))?,
        )
    } else {
        (Grain::Year, num(text)?, 1)
    };
    Some(Period::containing(
        Month::new(year, within as u8).ok()?,
        grain,
    ))
}

fn period_of(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Period> {
    if !obj.is_instance_of::<PyBool>()
        && let Ok(year) = obj.extract::<i32>()
    {
        return Ok(Period::year(year));
    }
    obj.extract::<String>()
        .ok()
        .and_then(|s| parse_period(&s))
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "{what} must be a period such as \"2020\", \"2020H2\", \"2020Q3\" or \
                 \"2020-07\" (or an integer year), got {}",
                py_repr(obj)
            ))
        })
}

fn bad_month(what: &str, text: &str) -> PyErr {
    PyValueError::new_err(format!("{what} must be a month \"YYYY-MM\", got {text:?}"))
}

/// A month as `"YYYY-MM"` or a date (`datetime.date`, pandas `Timestamp`);
/// an integer year is refused, as it would read as January.
fn month_of(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Month> {
    if obj.is_instance_of::<PyString>() {
        let text: String = obj.extract()?;
        return match text.trim().split_once('-') {
            Some((y, m)) => match (y.parse::<i32>(), m.parse::<u8>()) {
                (Ok(y), Ok(m)) => {
                    Month::new(y, m).map_err(|e| PyValueError::new_err(e.to_string()))
                }
                _ => Err(bad_month(what, &text)),
            },
            None => Err(bad_month(what, &text)),
        };
    } else if obj.hasattr("year")? && obj.hasattr("month")? {
        let year: i32 = obj.getattr("year")?.extract()?;
        let month: u8 = obj.getattr("month")?.extract()?;
        return Month::new(year, month).map_err(|e| PyValueError::new_err(e.to_string()));
    }
    Err(PyTypeError::new_err(format!(
        "{what} must be a month \"YYYY-MM\" or a date, got {}",
        py_repr(obj)
    )))
}

/// An index by calendar period: one value per period of one grain, from a
/// first period on, optionally extended past its last period at a constant
/// annual rate (CPI, wages, a claims-inflation index).
///
/// Parameters
/// ----------
/// first : str or int
///     The first period: ``"2020"`` (or the integer 2020), ``"2020H2"``,
///     ``"2020Q3"`` or ``"2020-07"``. Its grain is the index's grain.
/// values : list of float
///     The index in ``first``, then one value per period: finite and
///     positive.
/// future_rate : float, optional
///     The annual rate past the last period, compounded by months from the
///     last period's start: finite and above -1. Without it, a period
///     past the last is an error.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import CalendarIndex
/// >>> cpi = CalendarIndex("2020", [100.0, 104.0, 110.0], future_rate=0.05)
/// >>> cpi.value("2021-06")
/// 104.0
/// >>> round(cpi.value("2024-01"), 4)
/// 121.275
/// >>> cpi.periods
/// ['2020', '2021', '2022']
#[pyclass(name = "CalendarIndex", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCalendarIndex {
    inner: CalendarIndex,
}

#[pymethods]
impl PyCalendarIndex {
    #[new]
    #[pyo3(signature = (first, values, future_rate=None))]
    fn new(first: &Bound<'_, PyAny>, values: Vec<f64>, future_rate: Option<f64>) -> PyResult<Self> {
        let first = period_of(first, "first")?;
        let mut inner = CalendarIndex::new(first, values).map_err(err)?;
        if let Some(rate) = future_rate {
            inner = inner.with_future_rate(rate).map_err(err)?;
        }
        Ok(Self { inner })
    }

    /// The index of the period that contains a month.
    ///
    /// Parameters
    /// ----------
    /// month : str or date
    ///     ``"YYYY-MM"`` or a date.
    ///
    /// Returns
    /// -------
    /// float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For a month before the first period, or after the last one
    ///     when no future rate is set.
    fn value(&self, month: &Bound<'_, PyAny>) -> PyResult<f64> {
        self.inner.value(month_of(month, "month")?).map_err(err)
    }

    /// The given periods, one per value.
    #[getter]
    fn periods(&self) -> Vec<String> {
        self.inner
            .periods()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// The given values.
    #[getter]
    fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    /// The annual rate past the last period, or ``None``.
    #[getter]
    fn future_rate(&self) -> Option<f64> {
        self.inner.future_rate()
    }

    fn __getnewargs__(&self) -> (String, Vec<f64>, Option<f64>) {
        (
            self.inner.periods()[0].to_string(),
            self.inner.values().to_vec(),
            self.inner.future_rate(),
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyCalendarIndex>>()
            .is_ok_and(|o| o.inner == self.inner)
    }

    fn __repr__(&self) -> String {
        let rate = self
            .inner
            .future_rate()
            .map_or(String::new(), |r| format!(", future_rate={r:?}"));
        format!(
            "CalendarIndex(first={:?}, values={:?}{rate})",
            self.inner.periods()[0].to_string(),
            self.inner.values()
        )
    }
}

fn cells_of(inner: FutureCells) -> PyFutureCells {
    PyFutureCells { inner }
}

/// Projected future increments of one segment, one per origin and age, for
/// re-inflation.
///
/// Parameters
/// ----------
/// origins : list of str or int
///     The origin periods, such as ``["2020", "2021"]``.
/// origin : list of str or int
///     Each cell's origin, one of ``origins``.
/// age : list of int
///     Each cell's development age in months: 0 or more.
/// valuation : list of str or date
///     The month each cell's development period ends in (``"YYYY-MM"`` or
///     a date): its index period is the cell's payment period.
/// value : list of float
///     Each cell's increment, at the base price level.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import FutureCells
/// >>> cells = FutureCells(["2020"], ["2020", "2020"], [24, 36],
/// ...                     ["2021-12", "2022-12"], [50.0, 25.0])
/// >>> cells.reserves()
/// [75.0]
/// >>> cells.cells["age"]
/// [24, 36]
#[pyclass(name = "FutureCells", module = "prospicio.reserving", frozen)]
pub(crate) struct PyFutureCells {
    inner: FutureCells,
}

impl PyFutureCells {
    fn labels(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }
}

#[pymethods]
impl PyFutureCells {
    #[new]
    fn new(
        origins: Vec<Bound<'_, PyAny>>,
        origin: Vec<Bound<'_, PyAny>>,
        age: Vec<i64>,
        valuation: Vec<Bound<'_, PyAny>>,
        value: Vec<f64>,
    ) -> PyResult<Self> {
        let n = origin.len();
        if age.len() != n || valuation.len() != n || value.len() != n {
            return Err(PyValueError::new_err(format!(
                "origin, age, valuation and value must have one entry per cell: got {n}, {}, {} \
                 and {}",
                age.len(),
                valuation.len(),
                value.len()
            )));
        }
        let periods = origins
            .iter()
            .map(|o| period_of(o, "origins"))
            .collect::<PyResult<Vec<_>>>()?;
        let mut cells = Vec::with_capacity(n);
        for (k, label) in origin.iter().enumerate() {
            let cell_origin = period_of(label, "origin")?;
            let position = periods
                .iter()
                .position(|&q| q == cell_origin)
                .ok_or_else(|| {
                    PyValueError::new_err(format!("origin {cell_origin} is not one of the origins"))
                })?;
            let age = u32::try_from(age[k]).map_err(|_| {
                PyValueError::new_err(format!("age {} is not an age in months", age[k]))
            })?;
            cells.push(FutureCell {
                origin: position,
                age,
                valuation: month_of(&valuation[k], "valuation")?,
                value: value[k],
            });
        }
        Ok(cells_of(FutureCells {
            origins: periods,
            cells,
        }))
    }

    /// A chain ladder's future increments: at each age of the grid after an
    /// origin's latest one, the projected cumulative value less the one
    /// before, then the rest of the ultimate (the tail) as one increment
    /// one development period after the oldest age. The cells sum to the
    /// fit's reserves.
    ///
    /// Parameters
    /// ----------
    /// fit : ChainLadderFit
    /// segment : dict of str to object, optional
    ///     The key values choosing one segment of a fit over several, as
    ///     ``fit.segment``; without it the fit must have one segment.
    ///
    /// Returns
    /// -------
    /// FutureCells
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments and ``segment`` does not choose
    ///     exactly one.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import ChainLadder, FutureCells, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 110.0]}
    /// ... )
    /// >>> cells = FutureCells.from_chain_ladder(ChainLadder().fit(tri, "paid"))
    /// >>> cells.cells["origin"], cells.cells["age"]
    /// (['2021'], [24])
    #[staticmethod]
    #[pyo3(signature = (fit, segment=None))]
    fn from_chain_ladder(
        fit: PyRef<'_, PyChainLadderFit>,
        segment: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let one = pick(&fit.inner, segment)?;
        let [cl] = one.fits.as_slice() else {
            return Err(PyValueError::new_err(
                "the fit has several segments: choose one with segment={key: value}",
            ));
        };
        Ok(cells_of(FutureCells::from_chain_ladder(cl)))
    }

    /// An ODP GLM's future increments and their means.
    ///
    /// Parameters
    /// ----------
    /// fit : OdpGlmFit
    ///
    /// Returns
    /// -------
    /// FutureCells
    #[staticmethod]
    fn from_odp_glm(fit: PyRef<'_, PyOdpGlmFit>) -> PyResult<Self> {
        Ok(cells_of(
            FutureCells::from_odp_glm(&fit.inner).map_err(err)?,
        ))
    }

    /// The origin periods the cells refer to.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.labels()
    }

    /// The cells as a dict of columns: ``origin``, ``age``, ``valuation``
    /// (``"YYYY-MM"``) and ``value``.
    #[getter]
    fn cells<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let labels = self.labels();
        let c = &self.inner.cells;
        let out = PyDict::new(py);
        out.set_item(
            "origin",
            c.iter()
                .map(|x| labels[x.origin].clone())
                .collect::<Vec<_>>(),
        )?;
        out.set_item("age", c.iter().map(|x| x.age).collect::<Vec<_>>())?;
        out.set_item(
            "valuation",
            c.iter()
                .map(|x| x.valuation.to_string())
                .collect::<Vec<_>>(),
        )?;
        out.set_item("value", c.iter().map(|x| x.value).collect::<Vec<_>>())?;
        Ok(out)
    }

    /// The sum of the cells by origin.
    ///
    /// Returns
    /// -------
    /// list of float
    fn reserves(&self) -> Vec<f64> {
        self.inner.reserves()
    }

    fn __len__(&self) -> usize {
        self.inner.cells.len()
    }

    #[allow(clippy::type_complexity)]
    fn __getnewargs__(&self) -> (Vec<String>, Vec<String>, Vec<u32>, Vec<String>, Vec<f64>) {
        let labels = self.labels();
        let c = &self.inner.cells;
        (
            labels.clone(),
            c.iter().map(|x| labels[x.origin].clone()).collect(),
            c.iter().map(|x| x.age).collect(),
            c.iter().map(|x| x.valuation.to_string()).collect(),
            c.iter().map(|x| x.value).collect(),
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyFutureCells>>()
            .is_ok_and(|o| o.inner == self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "FutureCells(origins={:?}, cells={})",
            self.labels(),
            self.inner.cells.len()
        )
    }
}

/// A projection re-inflated by ``Indexation.reinflate``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import CalendarIndex, FutureCells, Indexation
/// >>> index = CalendarIndex("2020", [1.0, 1.1], future_rate=0.1)
/// >>> at_2020 = Indexation(index, "2020-01")
/// >>> cells = FutureCells(["2020"], ["2020"], [24], ["2021-12"], [100.0])
/// >>> out = at_2020.reinflate(cells)
/// >>> round(out.total_reserve, 6), out.total_base_reserve
/// (110.0, 100.0)
#[pyclass(name = "Reinflated", module = "prospicio.reserving", frozen)]
pub(crate) struct PyReinflated {
    inner: Reinflated,
}

#[pymethods]
impl PyReinflated {
    #[new]
    fn new(
        cells: PyRef<'_, PyFutureCells>,
        reserves: Vec<f64>,
        base_reserves: Vec<f64>,
    ) -> PyResult<Self> {
        let n = cells.inner.origins.len();
        if reserves.len() != n || base_reserves.len() != n {
            return Err(PyValueError::new_err(format!(
                "reserves and base_reserves need one value per origin ({n})"
            )));
        }
        Ok(Self {
            inner: Reinflated {
                cells: cells.inner.clone(),
                reserves,
                base_reserves,
            },
        })
    }

    fn __getnewargs__(&self) -> (PyFutureCells, Vec<f64>, Vec<f64>) {
        (
            cells_of(self.inner.cells.clone()),
            self.inner.reserves.clone(),
            self.inner.base_reserves.clone(),
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyReinflated>>()
            .is_ok_and(|o| o.inner == self.inner)
    }

    /// The re-inflated cells.
    #[getter]
    fn cells(&self) -> PyFutureCells {
        cells_of(self.inner.cells.clone())
    }

    /// The re-inflated reserve of each origin.
    #[getter]
    fn reserves(&self) -> Vec<f64> {
        self.inner.reserves.clone()
    }

    /// The reserve of each origin at the base level, before re-inflation.
    #[getter]
    fn base_reserves(&self) -> Vec<f64> {
        self.inner.base_reserves.clone()
    }

    /// The re-inflated total reserve.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    /// The total reserve at the base level.
    #[getter]
    fn total_base_reserve(&self) -> f64 {
        self.inner.total_base_reserve()
    }

    fn __repr__(&self) -> String {
        format!(
            "Reinflated(total_reserve={:?}, total_base_reserve={:?})",
            self.inner.total_reserve(),
            self.inner.total_base_reserve()
        )
    }
}

/// Restating a triangle at one price level by a ``CalendarIndex``, and
/// re-inflating a projection.
///
/// ``restate`` divides each increment of a column by ``I(p) / I(base)``,
/// ``p`` the index period that contains the cell's valuation month, in the
/// triangle's basis; ``inflate`` is its inverse. A restated triangle is
/// developed by any method; its projected future increments, as
/// ``FutureCells``, are taken back to the money of their payment periods by
/// ``reinflate``.
///
/// Parameters
/// ----------
/// index : CalendarIndex
/// base : str or date
///     The base price level: the index period that contains this month
///     (``"YYYY-MM"`` or a date).
///
/// Examples
/// --------
/// >>> from prospicio.reserving import CalendarIndex, Indexation, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 155.0, 110.0]},
/// ... )
/// >>> index = CalendarIndex("2020", [1.0, 1.1], future_rate=0.1)
/// >>> at_2020 = Indexation(index, "2020-01")
/// >>> restated = at_2020.restate(tri, "paid")
/// >>> restated.values[0][0]
/// [[100.0, 150.0], [100.0, nan]]
#[pyclass(name = "Indexation", module = "prospicio.reserving", frozen)]
pub(crate) struct PyIndexation {
    inner: Indexation,
}

#[pymethods]
impl PyIndexation {
    #[new]
    fn new(index: PyRef<'_, PyCalendarIndex>, base: &Bound<'_, PyAny>) -> PyResult<Self> {
        let base = month_of(base, "base")?;
        let inner = Indexation {
            index: index.inner.clone(),
            base,
        };
        // Fail now if the index does not cover the base period.
        inner.index.value(base).map_err(err)?;
        Ok(Self { inner })
    }

    /// The index.
    #[getter]
    fn index(&self) -> PyCalendarIndex {
        PyCalendarIndex {
            inner: self.inner.index.clone(),
        }
    }

    /// The base month, ``"YYYY-MM"``.
    #[getter]
    fn base(&self) -> String {
        self.inner.base.to_string()
    }

    /// ``I(p) / I(base)``, ``p`` the index period that contains ``month``:
    /// the factor that takes an amount at the base level to the money of
    /// ``p``.
    ///
    /// Parameters
    /// ----------
    /// month : str or date
    ///
    /// Returns
    /// -------
    /// float
    fn factor(&self, month: &Bound<'_, PyAny>) -> PyResult<f64> {
        self.inner.factor(month_of(month, "month")?).map_err(err)
    }

    /// A copy of the triangle with the increments of ``column`` restated at
    /// the base level, in every segment. Other columns are unchanged.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For an unknown column or a payment period the index does not
    ///     cover.
    fn restate(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyTriangle> {
        let (ix, data) = (&self.inner, &triangle.inner);
        let inner = py.detach(|| ix.restate(data, column)).map_err(err)?;
        Ok(PyTriangle { inner })
    }

    /// The inverse of ``restate``: the increments of ``column``, at the
    /// base level, taken to the money of their payment periods.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///
    /// Returns
    /// -------
    /// Triangle
    fn inflate(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyTriangle> {
        let (ix, data) = (&self.inner, &triangle.inner);
        let inner = py.detach(|| ix.inflate(data, column)).map_err(err)?;
        Ok(PyTriangle { inner })
    }

    /// Future increments at the base level re-inflated to the money of
    /// their payment periods, summed by origin.
    ///
    /// Parameters
    /// ----------
    /// cells : FutureCells
    ///
    /// Returns
    /// -------
    /// Reinflated
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the index does not cover a cell's payment period (set a
    ///     ``future_rate``).
    fn reinflate(&self, py: Python<'_>, cells: PyRef<'_, PyFutureCells>) -> PyResult<PyReinflated> {
        let (ix, data) = (&self.inner, &cells.inner);
        let inner = py.detach(|| ix.reinflate(data)).map_err(err)?;
        Ok(PyReinflated { inner })
    }

    fn __getnewargs__(&self) -> (PyCalendarIndex, String) {
        (self.index(), self.base())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyIndexation>>()
            .is_ok_and(|o| o.inner == self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Indexation(index={}, base={:?})",
            self.index().__repr__(),
            self.base()
        )
    }
}
