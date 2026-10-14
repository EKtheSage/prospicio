//! `prospicio.distributions.Discrete` (Probability lane): a distribution on
//! finitely many values of either sign, over `prospicio_prob::Discrete`.

use prospicio_prob::discrete::RETURN_PERIODS;
use prospicio_prob::{Discrete, Distribution};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::to_py;

/// A distribution on finitely many values, of either sign and any spacing:
/// a row of a P&L (premium less loss less expenses) or any other signed
/// outcome, as the Python package ``aggregate``'s ``GridDistribution`` on an
/// irregular support.
///
/// Values may repeat and come in any order: equal values (to ``1e-12``) are
/// merged, zero probabilities dropped, and the probabilities rescaled to
/// sum to 1. Quantiles and TVaR follow ``aggregate`` exactly; ``tvar`` is
/// the upper tail even when the values are a payoff, whose adverse tail is
/// the low one.
///
/// Parameters
/// ----------
/// values : list of float
/// probs : list of float
///     Non-negative, with a positive total.
///
/// Raises
/// ------
/// ValueError
///     If the lengths differ, a value is not finite, a probability is
///     negative or none is positive.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Discrete
/// >>> # A margin: 9 less the loss of two claims of 1, 2 or 4.
/// >>> m = Discrete([7, 6, 5, 4, 3, 1], [0.25, 0.3, 0.09, 0.2, 0.12, 0.04])
/// >>> round(m.mean(), 12), m.quantile(0.01)
/// (5.2, 1.0)
/// >>> round(m.tvar(0.01), 6)
/// 5.242424
#[pyclass(name = "Discrete", module = "prospicio.distributions", frozen)]
pub(crate) struct PyDiscrete {
    pub(crate) inner: Discrete,
}

#[pymethods]
impl PyDiscrete {
    #[new]
    fn new(values: Vec<f64>, probs: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Discrete::new(values, probs).map_err(to_py)?,
        })
    }

    /// The distinct values, ascending.
    #[getter]
    fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    /// Their probabilities, summing to 1.
    #[getter]
    fn probs(&self) -> Vec<f64> {
        self.inner.probs().to_vec()
    }

    /// The mean.
    ///
    /// Returns
    /// -------
    /// float
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// The variance.
    ///
    /// Returns
    /// -------
    /// float
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// The standard deviation.
    ///
    /// Returns
    /// -------
    /// float
    fn std_dev(&self) -> f64 {
        self.inner.std_dev()
    }

    /// The skewness, or None when the variance is 0.
    ///
    /// Returns
    /// -------
    /// float or None
    fn skewness(&self) -> Option<f64> {
        self.inner.skewness()
    }

    /// ``P(X <= x)``.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(x)
    }

    /// ``P(X > x)``.
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn survival(&self, x: f64) -> f64 {
        self.inner.survival(x)
    }

    /// The probability of exactly ``x`` (to ``1e-12``).
    ///
    /// Parameters
    /// ----------
    /// x : float
    ///
    /// Returns
    /// -------
    /// float
    fn pmf(&self, x: f64) -> f64 {
        self.inner.pmf(x)
    }

    /// The lower quantile, the smallest value with ``F >= p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///     In ``[0, 1]``.
    ///
    /// Returns
    /// -------
    /// float
    fn quantile(&self, p: f64) -> PyResult<f64> {
        self.inner.quantile(p).map_err(to_py)
    }

    /// The upper quantile, the smallest value with ``F > p``; None at
    /// ``p = 1``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///     In ``[0, 1]``.
    ///
    /// Returns
    /// -------
    /// float or None
    fn quantile_upper(&self, p: f64) -> PyResult<Option<f64>> {
        self.inner.quantile_upper(p).map_err(to_py)
    }

    /// Tail value at risk at ``p``: the mean of the upper ``1 - p`` of the
    /// distribution, the upper tail even for a payoff.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///     In ``[0, 1]``.
    ///
    /// Returns
    /// -------
    /// float
    fn tvar(&self, p: f64) -> PyResult<f64> {
        self.inner.tvar(p).map_err(to_py)
    }

    /// The return-period table, as ``aggregate``'s ``tail_df``: each period
    /// ``T`` read from both tails, at ``P = 1 / T`` and ``P = 1 - 1 / T``. A
    /// payoff (a P&L) is read off the low rows, a loss off the high ones.
    ///
    /// Parameters
    /// ----------
    /// periods : list of float, optional
    ///     At least 1; ``aggregate``'s 2, 5, 10, 25, 50, 100, 200, 250, 500,
    ///     1000 by default.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Columns ``p``, ``period``, ``var``, ``tvar``, ``excess`` (VaR less
    ///     the mean) and ``var_to_mean`` (None near a zero mean).
    #[pyo3(signature = (periods=None))]
    fn tail<'py>(
        &self,
        py: Python<'py>,
        periods: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let periods = periods.unwrap_or_else(|| RETURN_PERIODS.to_vec());
        let rows = self.inner.tail(&periods).map_err(to_py)?;
        let d = PyDict::new(py);
        d.set_item("p", rows.iter().map(|r| r.p).collect::<Vec<_>>())?;
        d.set_item("period", rows.iter().map(|r| r.period).collect::<Vec<_>>())?;
        d.set_item("var", rows.iter().map(|r| r.var).collect::<Vec<_>>())?;
        d.set_item("tvar", rows.iter().map(|r| r.tvar).collect::<Vec<_>>())?;
        d.set_item("excess", rows.iter().map(|r| r.excess).collect::<Vec<_>>())?;
        d.set_item(
            "var_to_mean",
            rows.iter().map(|r| r.var_to_mean).collect::<Vec<_>>(),
        )?;
        Ok(d)
    }

    fn __len__(&self) -> usize {
        self.inner.values().len()
    }

    fn __repr__(&self) -> String {
        let v = self.inner.values();
        format!(
            "Discrete(n={}, from {} to {}, mean={})",
            v.len(),
            v[0],
            v[v.len() - 1],
            self.inner.mean()
        )
    }
}
