//! `prospicio.distributions` (Probability lane): claim counts beyond the
//! Poisson, negative binomial and binomial, over
//! `prospicio_prob::count_families`.

use prospicio_core::StreamRng;
use prospicio_prob::Counting;
use prospicio_prob::count_families::{
    CompoundPoisson, CountDist, EmpiricalCount, Logarithmic, MixedPoisson, Mixing, ZeroModified,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::distributions::{PyNegativeBinomial, PyPoisson};
use crate::to_py;

/// Any claim count as one CountDist, from a Poisson, NegativeBinomial,
/// Binomial or Count.
pub(crate) fn count_dist(obj: &Bound<'_, PyAny>) -> PyResult<CountDist> {
    if let Ok(n) = obj.extract::<PyRef<'_, PyPoisson>>() {
        return Ok(CountDist::Poisson(n.inner));
    }
    if let Ok(n) = obj.extract::<PyRef<'_, PyNegativeBinomial>>() {
        return Ok(CountDist::NegativeBinomial(n.inner));
    }
    if let Ok(n) = obj.extract::<PyRef<'_, crate::pareto::PyBinomial>>() {
        return Ok(CountDist::Binomial(n.inner));
    }
    if let Ok(n) = obj.extract::<PyRef<'_, PyCount>>() {
        return Ok(n.inner.clone());
    }
    Err(PyTypeError::new_err(
        "expected a Poisson, NegativeBinomial, Binomial or Count",
    ))
}

/// A claim count from one of the families beyond the Poisson, negative
/// binomial and binomial: zero-modified and zero-truncated counts, the
/// logarithmic, mixed Poisson counts (negative binomial, Delaporte,
/// Poisson-inverse Gaussian and its shifted version), Poisson-stopped sums
/// (Neyman type A, Pólya-Aeppli) and empirical counts. These are
/// ``aggregate``'s ``zm``, ``zt``, ``logarithmic``, ``mixed``,
/// ``neymana`` and ``dfreq``. Make one with a static constructor; every
/// aggregation function takes it like any other count.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Count, Poisson
/// >>> n = Count.zero_truncated(Poisson(2.0))
/// >>> n.pmf(0)
/// 0.0
/// >>> pig = Count.mixed_poisson(10.0, 0.5, mixing="inverse_gaussian")
/// >>> round(pig.variance(), 9)
/// 35.0
#[pyclass(name = "Count", module = "prospicio.distributions", frozen)]
pub(crate) struct PyCount {
    pub(crate) inner: CountDist,
}

#[pymethods]
impl PyCount {
    /// The base count with ``P(N = 0) = p0``; the other probabilities are
    /// scaled to sum to ``1 - p0``.
    ///
    /// Parameters
    /// ----------
    /// base : Poisson, NegativeBinomial, Binomial or Count
    /// p0 : float
    ///     In ``[0, 1)``.
    ///
    /// Returns
    /// -------
    /// Count
    #[staticmethod]
    fn zero_modified(base: &Bound<'_, PyAny>, p0: f64) -> PyResult<Self> {
        let z = ZeroModified::new(count_dist(base)?, p0).map_err(to_py)?;
        Ok(Self {
            inner: CountDist::ZeroModified(Box::new(z)),
        })
    }

    /// The base count conditioned on at least one claim.
    ///
    /// Parameters
    /// ----------
    /// base : Poisson, NegativeBinomial, Binomial or Count
    ///
    /// Returns
    /// -------
    /// Count
    #[staticmethod]
    fn zero_truncated(base: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::zero_modified(base, 0.0)
    }

    /// The logarithmic count on ``1, 2, ...``:
    /// ``P(N = k) = -p**k / (k log(1 - p))``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///     In ``(0, 1)``.
    ///
    /// Returns
    /// -------
    /// Count
    #[staticmethod]
    fn logarithmic(p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: CountDist::Logarithmic(Logarithmic::new(p).map_err(to_py)?),
        })
    }

    /// A Poisson count whose mean is ``mean * theta``, with the mixing
    /// variable ``theta = shift + (1 - shift) G`` of mean 1 and coefficient
    /// of variation ``cv``; variance ``mean + mean**2 cv**2``.
    ///
    /// Parameters
    /// ----------
    /// mean : float
    /// cv : float
    ///     Coefficient of variation of ``theta``, fixed part included.
    /// mixing : {"gamma", "inverse_gaussian"}, default "gamma"
    ///     The distribution of ``G``. Gamma gives the negative binomial
    ///     (``shift = 0``) or the Delaporte; inverse Gaussian the
    ///     Poisson-inverse Gaussian or its shifted version.
    /// shift : float, default 0.0
    ///     The fixed part of ``theta``, in ``[0, 1)``.
    ///
    /// Returns
    /// -------
    /// Count
    #[staticmethod]
    #[pyo3(signature = (mean, cv, mixing="gamma", shift=0.0))]
    fn mixed_poisson(mean: f64, cv: f64, mixing: &str, shift: f64) -> PyResult<Self> {
        let mixing = match mixing {
            "gamma" => Mixing::Gamma { cv },
            "inverse_gaussian" | "ig" => Mixing::InverseGaussian { cv },
            other => {
                return Err(PyValueError::new_err(format!(
                    "mixing must be \"gamma\" or \"inverse_gaussian\", not {other:?}"
                )));
            }
        };
        Ok(Self {
            inner: CountDist::MixedPoisson(MixedPoisson::new(mean, mixing, shift).map_err(to_py)?),
        })
    }

    /// A Poisson(``rate``) number of clusters, each a ``secondary`` count:
    /// Poisson secondaries give the Neyman type A.
    ///
    /// Parameters
    /// ----------
    /// rate : float
    /// secondary : Poisson, NegativeBinomial, Binomial or Count
    ///
    /// Returns
    /// -------
    /// Count
    #[staticmethod]
    fn compound_poisson(rate: f64, secondary: &Bound<'_, PyAny>) -> PyResult<Self> {
        let c = CompoundPoisson::new(rate, count_dist(secondary)?).map_err(to_py)?;
        Ok(Self {
            inner: CountDist::CompoundPoisson(Box::new(c)),
        })
    }

    /// An empirical count, ``P(N = k) = probs[k]``.
    ///
    /// Parameters
    /// ----------
    /// probs : list of float
    ///     Non-negative, summing to 1.
    ///
    /// Returns
    /// -------
    /// Count
    #[staticmethod]
    fn empirical(probs: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: CountDist::Empirical(EmpiricalCount::new(probs).map_err(to_py)?),
        })
    }

    /// ``P(N = k)``.
    ///
    /// Parameters
    /// ----------
    /// k : int
    ///
    /// Returns
    /// -------
    /// float
    fn pmf(&self, k: u64) -> f64 {
        self.inner.pmf(k)
    }

    /// ``P(N <= k)``.
    ///
    /// Parameters
    /// ----------
    /// k : int
    ///
    /// Returns
    /// -------
    /// float
    fn cdf(&self, k: u64) -> f64 {
        self.inner.cdf(k)
    }

    /// Mean of the claim count.
    ///
    /// Returns
    /// -------
    /// float
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Variance of the claim count.
    ///
    /// Returns
    /// -------
    /// float
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// Smallest ``k`` with ``P(N <= k) >= p``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// int
    fn quantile(&self, p: f64) -> PyResult<u64> {
        self.inner.quantile(p).map_err(to_py)
    }

    /// ``(a, b)`` of the ``(a, b, 1)`` class, or ``None`` when Panjer's
    /// recursion does not apply (use FFT).
    ///
    /// Returns
    /// -------
    /// tuple of (float, float) or None
    fn panjer_ab(&self) -> Option<(f64, f64)> {
        self.inner.panjer_ab()
    }

    /// ``n`` claim counts from stream ``stream`` of the generator keyed by
    /// ``seed``.
    ///
    /// Parameters
    /// ----------
    /// n : int
    /// seed : int
    /// stream : int, default 0
    ///
    /// Returns
    /// -------
    /// list of int
    #[pyo3(signature = (n, seed, stream = 0))]
    fn sample(&self, py: Python<'_>, n: usize, seed: u64, stream: u64) -> Vec<u64> {
        let inner = self.inner.clone();
        py.detach(|| inner.sample(&mut StreamRng::new(seed, stream), n))
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            CountDist::ZeroModified(z) => format!("Count(zero_modified, p0={:?})", z.p0()),
            CountDist::Logarithmic(l) => format!("Count.logarithmic({:?})", l.p()),
            CountDist::MixedPoisson(m) => format!(
                "Count.mixed_poisson({:?}, mixing={:?}, shift={:?})",
                m.lambda(),
                m.mixing(),
                m.shift()
            ),
            CountDist::CompoundPoisson(c) => {
                format!("Count(compound_poisson, rate={:?})", c.lambda())
            }
            CountDist::Empirical(e) => format!("Count.empirical({:?})", e.probs()),
            other => format!("Count({other:?})"),
        }
    }
}
