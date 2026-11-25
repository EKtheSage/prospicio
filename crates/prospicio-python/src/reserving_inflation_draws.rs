//! `prospicio.reserving` stochastic re-inflation (issue #979,
//! `docs/design/reserving-inflation.md`, PR 8, decision 11), over
//! `prospicio_reserving::inflation_draws`: the `IndexPaths` class here, and
//! `Indexation.reinflate_draws` in `reserving_inflation.rs`, beside the
//! other `Indexation` methods.
//!
//! A `PredictiveDistribution` built in Python keys its origins by `str` or
//! `int`, never by a period, so `reinflate_draws` reads such an origin key
//! as a period (`"2020"`, `"2020Q3"`, `"2020-07"` or an integer year) and
//! gives the result back under the caller's own keys.

use std::collections::HashMap;

use prospicio_prob::{KeyValue, PredictiveDistribution};
use prospicio_reserving::{IndexPaths, Indexation, Period};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::distributions::PyPredictiveDistribution;
use crate::reserving::err;
use crate::reserving_inflation::{parse_period, period_of};
use crate::to_py;

/// One index path per simulation for the periods after a
/// ``CalendarIndex``'s last given period: index levels on the index's
/// scale, not rates. ``Indexation.reinflate_draws`` pairs simulation ``i``
/// of the draws with path ``i``.
///
/// Parameters
/// ----------
/// first : str or int
///     The first period of every path: ``"2023"`` (or the integer 2023),
///     ``"2023H1"``, ``"2023Q1"`` or ``"2023-01"``. Each level after the
///     first is one period of its grain later.
/// values : list of list of float or 2-D numpy array
///     One row per simulation, all of one length (at least 1): the index
///     levels from ``first`` on, finite and positive.
///
/// Raises
/// ------
/// ValueError
///     For no rows, empty or ragged rows, or a level that is not finite
///     and positive.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import IndexPaths
/// >>> paths = IndexPaths("2023", [[1.10, 1.20], [1.05, 1.08]])
/// >>> paths.n_sims, paths.n_periods
/// (2, 2)
/// >>> paths.periods
/// ['2023', '2024']
/// >>> paths.values[1]
/// [1.05, 1.08]
#[pyclass(name = "IndexPaths", module = "prospicio.reserving", frozen)]
pub(crate) struct PyIndexPaths {
    pub(crate) inner: IndexPaths,
}

#[pymethods]
impl PyIndexPaths {
    #[new]
    fn new(first: &Bound<'_, PyAny>, values: Vec<Vec<f64>>) -> PyResult<Self> {
        let first = period_of(first, "first")?;
        let width = values.first().map_or(0, Vec::len);
        if let Some(bad) = values.iter().position(|row| row.len() != width) {
            return Err(PyValueError::new_err(format!(
                "row {bad} of values has {} levels, expected {width} as in row 0",
                values[bad].len()
            )));
        }
        let n_sims = values.len();
        let inner =
            IndexPaths::new(first, n_sims, values.into_iter().flatten().collect()).map_err(err)?;
        Ok(Self { inner })
    }

    /// The first period of every path.
    #[getter]
    fn first(&self) -> String {
        self.inner.first().to_string()
    }

    /// The number of simulations (paths).
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims()
    }

    /// The number of periods of each path.
    #[getter]
    fn n_periods(&self) -> usize {
        self.inner.n_periods()
    }

    /// The periods of the paths, from ``first`` at its grain.
    #[getter]
    fn periods(&self) -> Vec<String> {
        self.inner
            .periods()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// The levels, one row per simulation.
    #[getter]
    fn values(&self) -> Vec<Vec<f64>> {
        self.inner
            .values()
            .chunks_exact(self.inner.n_periods())
            .map(<[f64]>::to_vec)
            .collect()
    }

    fn __getnewargs__(&self) -> (String, Vec<Vec<f64>>) {
        (self.first(), self.values())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyIndexPaths>>()
            .is_ok_and(|o| o.inner == self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "IndexPaths(first={:?}, n_sims={}, n_periods={})",
            self.first(),
            self.inner.n_sims(),
            self.inner.n_periods()
        )
    }
}

/// An origin key read as a period: a period label or an integer year.
fn origin_period(key: &KeyValue) -> Option<Period> {
    match key {
        KeyValue::Text(t) => parse_period(t),
        KeyValue::Int(year) => i32::try_from(*year).ok().map(Period::year),
        _ => None,
    }
}

/// The draws with their `str` and `int` origin keys as periods, or `None`
/// when there is no origin dimension or nothing to change. Two components
/// whose origins read as the same period (`"2020"` and `2020`) are an
/// error that names both keys.
fn with_period_origins(draws: &PredictiveDistribution) -> PyResult<Option<PredictiveDistribution>> {
    let Some(origin) = draws.dims().iter().position(|d| d == "origin") else {
        return Ok(None);
    };
    let mut changed = false;
    let mut components = Vec::with_capacity(draws.components().len());
    let mut seen = HashMap::with_capacity(draws.components().len());
    for (j, given) in draws.components().iter().enumerate() {
        let mut key = given.clone();
        if let Some(p) = origin_period(&key[origin]) {
            key[origin] = KeyValue::Period(p);
            changed = true;
        }
        if let Some(i) = seen.insert(key.clone(), j) {
            return Err(PyValueError::new_err(format!(
                "the components {} and {} both read as the origin {}",
                show_key(&draws.components()[i]),
                show_key(given),
                key[origin]
            )));
        }
        components.push(key);
    }
    if !changed {
        return Ok(None);
    }
    PredictiveDistribution::from_draws(
        draws.dims().to_vec(),
        components,
        draws.draw_matrix().to_vec(),
        draws.provenance().clone(),
    )
    .map(Some)
    .map_err(to_py)
}

/// A component key as the caller wrote it, a text value quoted.
fn show_key(key: &[KeyValue]) -> String {
    let parts: Vec<String> = key
        .iter()
        .map(|v| match v {
            KeyValue::Text(t) => format!("{t:?}"),
            other => other.to_string(),
        })
        .collect();
    format!("({})", parts.join(", "))
}

/// `Indexation::reinflate_draws` on draws built in Python or by a model,
/// the result under the input's own component keys.
pub(crate) fn reinflate_draws(
    py: Python<'_>,
    ix: &Indexation,
    draws: &PredictiveDistribution,
    paths: Option<&IndexPaths>,
) -> PyResult<PyPredictiveDistribution> {
    let inner = py.detach(|| -> PyResult<PredictiveDistribution> {
        let Some(read) = with_period_origins(draws)? else {
            return ix.reinflate_draws(draws, paths).map_err(err);
        };
        let out = ix.reinflate_draws(&read, paths).map_err(err)?;
        PredictiveDistribution::from_draws(
            draws.dims().to_vec(),
            draws.components().to_vec(),
            out.draw_matrix().to_vec(),
            out.provenance().clone(),
        )
        .map_err(to_py)
    })?;
    Ok(PyPredictiveDistribution { inner })
}
