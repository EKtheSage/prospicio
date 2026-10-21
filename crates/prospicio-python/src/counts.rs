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
    /// mixing : {"gamma", "inverse_gaussian", "beta", "sichel", "sichel_gamma", "sichel_ig"}, default "gamma"
    ///     The distribution of ``G``. Gamma gives the negative binomial
    ///     (``shift = 0``) or the Delaporte; inverse Gaussian the
    ///     Poisson-inverse Gaussian or its shifted version. The others take
    ///     no shift, as ``aggregate``'s: ``"beta"``, the Poisson mean
    ///     ``max`` times a beta variable (``mixed beta cv max``);
    ///     ``"sichel"``, the generalized inverse Gaussian with the given
    ///     ``shape`` (``mixed sichel cv shape``; ``-0.5`` is the inverse
    ///     Gaussian) or, with ``skewness`` instead, the shape that gives that
    ///     skewness; ``"sichel_gamma"`` and ``"sichel_ig"``, the generalized
    ///     inverse Gaussian with the skewness of the Delaporte's or the
    ///     shifted inverse Gaussian's mixing with fixed part ``f``
    ///     (``mixed sichel.gamma cv f``, ``mixed sichel.ig cv f``).
    /// shift : float, default 0.0
    ///     The fixed part of ``theta``, in ``[0, 1)``.
    /// max : float, optional
    ///     The beta mixing's largest Poisson mean, above ``mean (1 + cv**2)``.
    /// shape : float, optional
    ///     The Sichel shape.
    /// skewness : float, optional
    ///     The Sichel mixing's skewness, for ``"sichel"`` without a shape.
    /// f : float, optional
    ///     The fixed part matched by ``"sichel_gamma"`` and ``"sichel_ig"``.
    ///
    /// Returns
    /// -------
    /// Count
    #[staticmethod]
    #[pyo3(signature = (mean, cv, mixing="gamma", shift=0.0, *, max=None, shape=None, skewness=None, f=None))]
    #[allow(clippy::too_many_arguments)]
    fn mixed_poisson(
        mean: f64,
        cv: f64,
        mixing: &str,
        shift: f64,
        max: Option<f64>,
        shape: Option<f64>,
        skewness: Option<f64>,
        f: Option<f64>,
    ) -> PyResult<Self> {
        let need = |v: Option<f64>, name: &str| {
            v.ok_or_else(|| PyValueError::new_err(format!("mixing {mixing:?} needs {name}")))
        };
        let mixing = match mixing {
            "gamma" => Mixing::Gamma { cv },
            "inverse_gaussian" | "ig" => Mixing::InverseGaussian { cv },
            "beta" => Mixing::Beta {
                cv,
                max: need(max, "max")?,
            },
            "sichel" => match (shape, skewness) {
                (Some(shape), None) => Mixing::Sichel { cv, shape },
                (None, Some(skewness)) => Mixing::SichelSkewness { cv, skewness },
                _ => {
                    return Err(PyValueError::new_err(
                        "mixing \"sichel\" needs one of shape or skewness",
                    ));
                }
            },
            "sichel_gamma" => Mixing::sichel_gamma(cv, need(f, "f")?),
            "sichel_ig" => Mixing::sichel_ig(cv, need(f, "f")?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "mixing must be \"gamma\", \"inverse_gaussian\", \"beta\", \"sichel\", \
                     \"sichel_gamma\" or \"sichel_ig\", not {other:?}"
                )));
            }
        };
        Ok(Self {
            inner: CountDist::MixedPoisson(MixedPoisson::new(mean, mixing, shift).map_err(to_py)?),
        })
    }

    /// The generalized inverse Gaussian a Sichel mixing solved, as a dict
    /// with ``shape``, ``omega`` and ``eta`` (``aggregate``'s ``lam``,
    /// ``mu / beta`` and ``mu``); ``None`` for any other count.
    ///
    /// Returns
    /// -------
    /// dict or None
    fn gig(&self) -> Option<std::collections::BTreeMap<&'static str, f64>> {
        match &self.inner {
            CountDist::MixedPoisson(m) => m.gig().map(|g| {
                [("shape", g.shape), ("omega", g.omega), ("eta", g.eta)]
                    .into_iter()
                    .collect()
            }),
            _ => None,
        }
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

    /// The number of claims by ``horizon`` when the waits between claims
    /// are independent draws from ``wait``: a renewal (Sparre Andersen)
    /// count, ``aggregate``'s ``years ... wait ...`` frequency. Exponential
    /// waits give the Poisson.
    ///
    /// The wait is put on a grid and the count read off by FFT; waits of
    /// zero arrive in clusters, and waits whose probabilities sum to less
    /// than 1 end the process. The grid is ``aggregate``'s by default.
    ///
    /// Parameters
    /// ----------
    /// wait : distribution or tuple of (list of float, list of float)
    ///     The wait's distribution, or its values and their probabilities
    ///     (``aggregate``'s ``dwait``), which may sum to less than 1.
    /// horizon : float
    ///     The period, in the wait's units.
    /// step : float, optional
    ///     The wait grid's step; give it with ``points``.
    /// points : int, optional
    ///     The wait grid's number of points.
    /// closed : bool, default False
    ///     With ``step``: whether waits summing to exactly ``horizon``
    ///     count in full (waits on the grid's lattice), not half.
    ///
    /// Returns
    /// -------
    /// Count
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Count, Gamma
    /// >>> n = Count.renewal(Gamma(2.0, 1.0), 5.0)
    /// >>> round(n.pmf(0), 6)  # the first wait is beyond 5: 6 e^-5
    /// 0.040428
    /// >>> round(Count.renewal(([1.0], [1.0]), 3.0).pmf(3), 12)  # waits of 1 by 3
    /// 1.0
    #[staticmethod]
    #[pyo3(signature = (wait, horizon, *, step=None, points=None, closed=false))]
    fn renewal(
        wait: &Bound<'_, PyAny>,
        horizon: f64,
        step: Option<f64>,
        points: Option<usize>,
        closed: bool,
    ) -> PyResult<Self> {
        use prospicio_aggregate::renewal::{Wait, WaitGrid, renewal_count};
        let grid = match (step, points) {
            (None, None) => None,
            (Some(step), Some(points)) => Some(WaitGrid {
                step,
                points,
                closed,
            }),
            _ => {
                return Err(PyValueError::new_err(
                    "give both step and points, or neither",
                ));
            }
        };
        let n = if let Ok((values, probs)) = wait.extract::<(Vec<f64>, Vec<f64>)>() {
            let w = Wait::Atoms {
                values: &values,
                probs: &probs,
            };
            renewal_count(&w, horizon, grid)
        } else {
            let d = crate::distributions::extract_dist(wait)?;
            renewal_count(&Wait::Law(&d), horizon, grid)
        }
        .map_err(to_py)?;
        Ok(Self {
            inner: CountDist::Empirical(n.count),
        })
    }

    /// A fixed count, ``N = n``, ``aggregate``'s ``fixed``.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///
    /// Returns
    /// -------
    /// Count
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Count
    /// >>> n = Count.fixed(3)
    /// >>> n.mean(), n.variance()
    /// (3.0, 0.0)
    #[staticmethod]
    fn fixed(n: u64) -> Self {
        Self {
            inner: CountDist::Empirical(EmpiricalCount::fixed(n)),
        }
    }

    /// A Bernoulli count, one claim with probability ``p`` in ``[0, 1)``,
    /// ``aggregate``'s ``bernoulli``.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// Count
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``p`` is outside ``[0, 1)``.
    #[staticmethod]
    fn bernoulli(p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: CountDist::Binomial(prospicio_prob::Binomial::bernoulli(p).map_err(to_py)?),
        })
    }

    /// A geometric count on ``0, 1, ...`` with this mean, a negative
    /// binomial with ``r = 1``, ``aggregate``'s ``geometric``.
    ///
    /// Parameters
    /// ----------
    /// mean : float
    ///
    /// Returns
    /// -------
    /// Count
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``mean`` is not finite and positive.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Count
    /// >>> round(Count.geometric(3.0).pmf(0), 12)
    /// 0.25
    #[staticmethod]
    fn geometric(mean: f64) -> PyResult<Self> {
        Ok(Self {
            inner: CountDist::NegativeBinomial(
                prospicio_prob::NegativeBinomial::geometric(mean).map_err(to_py)?,
            ),
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
