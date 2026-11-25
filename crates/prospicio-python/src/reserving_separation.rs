//! `prospicio.reserving.Separation`: Taylor's separation method, arithmetic
//! and geometric (issue #979, `docs/design/reserving-inflation.md`, PR 5),
//! over `prospicio_reserving::separation`.

use prospicio_reserving::{
    FutureInflation, Retransformation, Separation, SeparationFit, SeparationKind,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::reserving::{PyTriangle, err, pick};
use crate::reserving_inflation::{PyCalendarIndex, PyFutureCells, cells_of};

const KINDS: &str = "\"arithmetic\", \"geometric\" or \"geometric_lognormal\"";

fn kind_of(text: &str) -> PyResult<SeparationKind> {
    match text {
        "arithmetic" => Ok(SeparationKind::Arithmetic),
        "geometric" => Ok(SeparationKind::Geometric(Retransformation::GeometricMean)),
        "geometric_lognormal" => Ok(SeparationKind::Geometric(Retransformation::LognormalMean)),
        other => Err(PyValueError::new_err(format!(
            "kind must be {KINDS}, got {other:?}"
        ))),
    }
}

fn kind_name(kind: SeparationKind) -> &'static str {
    match kind {
        SeparationKind::Arithmetic => "arithmetic",
        SeparationKind::Geometric(Retransformation::GeometricMean) => "geometric",
        SeparationKind::Geometric(Retransformation::LognormalMean) => "geometric_lognormal",
    }
}

/// Taylor's (1977) separation method: increments over a normaliser are
/// ``r_j * lambda_(i+j)``, development proportions ``r_j`` (summing to 1)
/// times an index ``lambda_h`` per calendar period.
///
/// ``"arithmetic"`` solves Taylor's equations on the column and diagonal
/// sums (the marginal equations of a Poisson GLM on development and
/// calendar factors). ``"geometric"`` is least squares on the logarithms,
/// taken back to amounts by the geometric means; ``"geometric_lognormal"``
/// by the lognormal means (a factor ``exp(sigma^2 / 2)`` on every index,
/// which needs at least 3 origins). The geometric methods need every
/// increment positive. The triangle must be full: as many ages as origins,
/// the origin grain equal to the development grain, no hole.
///
/// The indices after the triangle's last calendar period are not fitted:
/// give exactly one of ``future_rate``, ``future_index`` and
/// ``trend_periods``. On a triangle already restated by an economic index
/// (``Indexation.restate``) the indices measure superimposed inflation.
///
/// Parameters
/// ----------
/// kind : {"arithmetic", "geometric", "geometric_lognormal"}, default "arithmetic"
/// normaliser : list of float, optional
///     ``n_i``, one per origin: finite and positive (claim counts, premium,
///     exposure). Without it every origin has 1.
/// future_rate : float, optional
///     A constant annual rate from the last fitted index, compounded by
///     months: above -1.
/// future_index : CalendarIndex, optional
///     The relative changes of an index from the triangle's last calendar
///     period, read at the last sub-period of each period.
/// trend_periods : int, optional
///     The exponential trend of the last ``trend_periods`` fitted indices,
///     continued from the last: at least 2, at most the number of calendar
///     periods.
///
/// Raises
/// ------
/// ValueError
///     For an unknown ``kind``, or when not exactly one of the three future
///     settings is given.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Separation, Triangle
/// >>> # Taylor (1977), Example 1: claims per claim incurred, cumulative.
/// >>> rows = [[50.4, 78.6, 87.6, 92.4], [58.0, 87.2, 96.9], [59.5, 92.7], [66.2]]
/// >>> tri = Triangle.from_long(
/// ...     [2001 + i for i, row in enumerate(rows) for _ in row],
/// ...     [12 * (j + 1) for row in rows for j in range(len(row))],
/// ...     {"s": [v for row in rows for v in row]},
/// ... )
/// >>> fit = Separation(future_rate=0.10).fit(tri, "s")
/// >>> [round(r, 4) for r in fit.proportions]
/// [0.5835, 0.2878, 0.0866, 0.0421]
/// >>> round(fit.indices[0], 1)
/// 86.4
#[pyclass(name = "Separation", module = "prospicio.reserving", frozen)]
pub(crate) struct PySeparation {
    inner: Separation,
}

#[pymethods]
impl PySeparation {
    #[new]
    #[pyo3(signature = (kind="arithmetic", normaliser=None, future_rate=None, future_index=None, trend_periods=None))]
    fn new(
        kind: &str,
        normaliser: Option<Vec<f64>>,
        future_rate: Option<f64>,
        future_index: Option<PyRef<'_, PyCalendarIndex>>,
        trend_periods: Option<i64>,
    ) -> PyResult<Self> {
        let kind = kind_of(kind)?;
        let future = match (future_rate, future_index, trend_periods) {
            (Some(rate), None, None) => FutureInflation::Rate(rate),
            (None, Some(index), None) => FutureInflation::Index(index.inner.clone()),
            (None, None, Some(periods)) => FutureInflation::Trend {
                periods: usize::try_from(periods).map_err(|_| {
                    PyValueError::new_err(format!(
                        "trend_periods must be a whole number of at least 2, got {periods}"
                    ))
                })?,
            },
            (None, None, None) => {
                return Err(PyValueError::new_err(
                    "give the future indices: one of future_rate, future_index or trend_periods",
                ));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "give only one of future_rate, future_index and trend_periods",
                ));
            }
        };
        Ok(Self {
            inner: Separation {
                kind,
                normaliser,
                future,
            },
        })
    }

    /// ``"arithmetic"``, ``"geometric"`` or ``"geometric_lognormal"``.
    #[getter]
    fn kind(&self) -> &'static str {
        kind_name(self.inner.kind)
    }

    /// The normaliser ``n_i``, or ``None`` for 1 in every origin.
    #[getter]
    fn normaliser(&self) -> Option<Vec<f64>> {
        self.inner.normaliser.clone()
    }

    /// The annual rate past the last period, or ``None``.
    #[getter]
    fn future_rate(&self) -> Option<f64> {
        match self.inner.future {
            FutureInflation::Rate(r) => Some(r),
            _ => None,
        }
    }

    /// The index the future follows, or ``None``.
    #[getter]
    fn future_index(&self) -> Option<PyCalendarIndex> {
        match &self.inner.future {
            FutureInflation::Index(i) => Some(PyCalendarIndex { inner: i.clone() }),
            _ => None,
        }
    }

    /// How many fitted indices the future trend uses, or ``None``.
    #[getter]
    fn trend_periods(&self) -> Option<usize> {
        match self.inner.future {
            FutureInflation::Trend { periods } => Some(periods),
            _ => None,
        }
    }

    /// Fits ``column`` of the triangle.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    /// segment : dict of str to object, optional
    ///     The key values choosing one segment of a triangle with several
    ///     (compared as ``str()`` of each value); a triangle with one needs
    ///     none.
    ///
    /// Returns
    /// -------
    /// SeparationFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For an unknown column, a triangle that is not full, a hole,
    ///     fewer than 2 origins, a normaliser of the wrong length or with a
    ///     value not finite and positive, a non-positive sum in the
    ///     recursion, a non-positive increment in a geometric method, fewer
    ///     than 3 origins for ``"geometric_lognormal"``, or a future that
    ///     cannot be formed (a rate at or below -1, an index that misses a
    ///     period, a trend over fewer than 2 or more than all periods).
    ///     Also when ``segment`` does not choose exactly one segment.
    #[pyo3(signature = (triangle, column, segment=None))]
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        segment: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PySeparationFit> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let fits = py.detach(|| model.fit_segments(tri, column)).map_err(err)?;
        let one = pick(&fits, segment)?;
        let [fit] = one.fits.as_slice() else {
            return Err(PyValueError::new_err(
                "the triangle has several segments: choose one with segment={key: value}",
            ));
        };
        Ok(PySeparationFit { inner: fit.clone() })
    }

    #[allow(clippy::type_complexity)]
    fn __getnewargs__(
        &self,
    ) -> (
        &'static str,
        Option<Vec<f64>>,
        Option<f64>,
        Option<PyCalendarIndex>,
        Option<usize>,
    ) {
        (
            self.kind(),
            self.normaliser(),
            self.future_rate(),
            self.future_index(),
            self.trend_periods(),
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PySeparation>>()
            .is_ok_and(|o| o.inner == self.inner)
    }

    fn __repr__(&self) -> String {
        let future = match &self.inner.future {
            FutureInflation::Rate(r) => format!("future_rate={r:?}"),
            FutureInflation::Index(_) => "future_index=CalendarIndex(...)".to_string(),
            FutureInflation::Trend { periods } => format!("trend_periods={periods}"),
        };
        let normaliser = if self.inner.normaliser.is_some() {
            ", normaliser=[...]"
        } else {
            ""
        };
        format!("Separation(kind={:?}{normaliser}, {future})", self.kind())
    }
}

/// A separation fit of one triangle segment.
///
/// Attributes
/// ----------
/// origins : list of str
///     The origin periods; ``normaliser``, ``fitted`` and ``reserves``
///     follow them.
/// ages : list of int
///     The development ages in months; ``proportions`` and the columns of
///     ``fitted`` follow them.
/// normaliser : list of float
///     ``n_i`` (1 when none was given).
/// proportions : list of float
///     ``r_j``, summing to 1.
/// calendar : list of str
///     The triangle's calendar periods, at the origins' grain.
/// indices : list of float
///     ``lambda_h``, one per calendar period.
/// future_indices : dict
///     The calendar periods after the triangle's last, up to the last
///     future cell's, as ``{"period": [...], "index": [...]}``.
/// fitted : list of list of float
///     ``n_i r_j lambda_(i+j)`` on the observed cells, ``None`` on the
///     future ones; one row per origin.
/// future : FutureCells
///     ``n_i r_j lambda_(i+j)`` on the future cells (at the money of their
///     payment periods; there is no tail past the oldest age).
/// reserves : list of float
///     The sum of each origin's future cells: the first origin's is 0.
/// log_variance : float or None
///     The geometric methods' residual variance on the log scale; ``None``
///     for the arithmetic method and when no degrees of freedom are left (as
///     on 2 origins).
/// total_reserve : float
#[pyclass(name = "SeparationFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PySeparationFit {
    inner: SeparationFit,
}

#[pymethods]
impl PySeparationFit {
    /// The origin periods; ``normaliser``, ``fitted`` and ``reserves`` follow them.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// The development ages in months; ``proportions`` and the columns of ``fitted`` follow them.
    #[getter]
    fn ages(&self) -> Vec<u32> {
        self.inner.ages.clone()
    }

    /// The normaliser ``n_i``, one per origin (1 when none was given).
    #[getter]
    fn normaliser(&self) -> Vec<f64> {
        self.inner.normaliser.clone()
    }

    /// The development proportions ``r_j``, one per age, summing to 1.
    #[getter]
    fn proportions(&self) -> Vec<f64> {
        self.inner.proportions.clone()
    }

    /// The triangle's calendar periods, at the origins' grain.
    #[getter]
    fn calendar(&self) -> Vec<String> {
        self.inner
            .calendar
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// The index ``lambda_h`` of each calendar period.
    #[getter]
    fn indices(&self) -> Vec<f64> {
        self.inner.indices.clone()
    }

    /// The periods after the triangle's last, as ``{"period": [...], "index": [...]}``.
    #[getter]
    fn future_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item(
            "period",
            self.inner
                .future_indices
                .iter()
                .map(|(p, _)| p.to_string())
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "index",
            self.inner
                .future_indices
                .iter()
                .map(|(_, v)| *v)
                .collect::<Vec<_>>(),
        )?;
        Ok(out)
    }

    /// ``n_i r_j lambda_(i+j)`` on the observed cells, ``None`` on the future ones.
    #[getter]
    fn fitted(&self) -> Vec<Vec<Option<f64>>> {
        self.inner.fitted.clone()
    }

    /// The future cells, ``n_i r_j lambda_(i+j)``, as ``FutureCells``.
    #[getter]
    fn future(&self) -> PyFutureCells {
        cells_of(self.inner.future.clone())
    }

    /// The sum of each origin's future cells; the first origin's is 0.
    #[getter]
    fn reserves(&self) -> Vec<f64> {
        self.inner.reserves.clone()
    }

    /// The geometric methods' residual variance on the log scale, or ``None``.
    #[getter]
    fn log_variance(&self) -> Option<f64> {
        self.inner.log_variance
    }

    /// The total reserve.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PySeparationFit>>()
            .is_ok_and(|o| o.inner == self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "SeparationFit(origins={}, total_reserve={:?})",
            self.inner.origins.len(),
            self.inner.total_reserve()
        )
    }
}
