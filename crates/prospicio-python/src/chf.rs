//! `prospicio.distributions.invert_chf` and `ChfDistribution` (Probability
//! lane), over `prospicio_prob::chf`: a law's distribution on a grid from
//! its characteristic function (#779).

use std::sync::Arc;

use prospicio_prob::Distribution;
use prospicio_prob::chf::{
    BUCKET_SIZING_P, ChfDistribution, ChfFn, ChfLaw, ChfOptions, Complex64, UserChf, invert,
};
use pyo3::prelude::*;

use crate::distributions::{PyGrid, extract_dist};
use crate::normal::PyNormal;
use crate::to_py;

/// A law's bucket masses on a grid, from its characteristic function: what
/// ``invert_chf`` returns.
///
/// Bucket ``k`` holds the law's probability within half a step of
/// ``x_min + k * step`` (the rounding discretization). The masses are as
/// inverted, not renormalized; the statistics are those of the masses on
/// the window.
#[pyclass(name = "ChfDistribution", module = "prospicio.distributions", frozen)]
pub(crate) struct PyChfDistribution {
    inner: ChfDistribution,
}

#[pymethods]
impl PyChfDistribution {
    /// The first grid point.
    #[getter]
    fn x_min(&self) -> f64 {
        self.inner.x_min()
    }

    /// The bucket.
    #[getter]
    fn step(&self) -> f64 {
        self.inner.step()
    }

    /// The grid has ``2 ** log2`` buckets.
    #[getter]
    fn log2(&self) -> u32 {
        self.inner.report().log2
    }

    /// Whether the bucket was given rather than sized from the law's tails.
    #[getter]
    fn step_given(&self) -> bool {
        self.inner.report().step_given
    }

    /// The grid points.
    #[getter]
    fn xs(&self) -> Vec<f64> {
        self.inner.xs()
    }

    /// The bucket masses, as inverted.
    #[getter]
    fn masses(&self) -> Vec<f64> {
        self.inner.masses().to_vec()
    }

    /// ``|phi(pi / step)|`` of the continuous part: the frequencies the
    /// inversion drops. Above 1e-6 the masses ring; lower the step.
    #[getter]
    fn truncation_error(&self) -> f64 {
        self.inner.report().truncation_error
    }

    /// Whether ``truncation_error`` is above 1e-6.
    #[getter]
    fn is_truncated(&self) -> bool {
        self.inner.report().is_truncated()
    }

    /// The Chernoff bound on the law's mass below the window, which wraps
    /// onto its top; 0 on a grid whose left edge was given.
    #[getter]
    fn outside_below(&self) -> f64 {
        self.inner.report().outside_below
    }

    /// The Chernoff bound on the law's mass above the window, which wraps
    /// onto its bottom; 0 on a grid whose left edge was given.
    #[getter]
    fn outside_above(&self) -> f64 {
        self.inner.report().outside_above
    }

    /// The sum of the masses.
    #[getter]
    fn total_mass(&self) -> f64 {
        self.inner.report().total_mass
    }

    /// The law's mean (``nan`` for your own function).
    #[getter]
    fn law_mean(&self) -> f64 {
        self.inner.report().law_mean
    }

    /// The law's variance (``nan`` for your own function).
    #[getter]
    fn law_variance(&self) -> f64 {
        self.inner.report().law_variance
    }

    /// The mean of the masses on the window.
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// The variance of the masses on the window.
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// The standard deviation of the masses on the window.
    fn std_dev(&self) -> f64 {
        self.inner.std_dev()
    }

    /// The sum of the masses at grid points at or below ``x``.
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

    /// The first grid point whose cumulative mass reaches ``p``.
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

    /// The masses as a ``Grid`` from 0, negative round-off set to 0 and the
    /// sum rescaled to 1, for the compound and risk-measure functions.
    ///
    /// Returns
    /// -------
    /// Grid
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the window starts below 0.
    fn to_grid(&self) -> PyResult<PyGrid> {
        Ok(PyGrid {
            inner: self.inner.to_grid().map_err(to_py)?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.masses().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChfDistribution(x_min={:?}, step={:?}, log2={})",
            self.inner.x_min(),
            self.inner.step(),
            self.inner.report().log2
        )
    }
}

/// A Python callable of a list of frequencies returning their complex
/// values. A Python exception becomes the error message.
fn py_chf(f: Py<PyAny>) -> Arc<ChfFn> {
    Arc::new(move |t: &[f64]| {
        Python::attach(|py| {
            let out = f
                .call1(py, (t.to_vec(),))
                .map_err(|e| prospicio_core::Error::Data(e.to_string()))?;
            let mut values = Vec::with_capacity(t.len());
            for item in out
                .bind(py)
                .try_iter()
                .map_err(|e| prospicio_core::Error::Data(e.to_string()))?
            {
                let z = item
                    .and_then(|z| {
                        let re = z.getattr("real")?.extract::<f64>()?;
                        let im = z.getattr("imag")?.extract::<f64>()?;
                        Ok(Complex64::new(re, im))
                    })
                    .map_err(|e| prospicio_core::Error::Data(e.to_string()))?;
                values.push(z);
            }
            Ok(values)
        })
    })
}

/// A law's distribution on a grid, by inverting its characteristic
/// function: ``aggregate``'s ``AggregateCHF`` (``chfagg``).
///
/// The characteristic function is sampled on the FFT's frequency grid,
/// times ``sinc(t step / 2)``, and inverted once, so bucket ``k`` holds the
/// probability within half a step of ``x_min + k * step``. With ``step``
/// left out, the window is sized from the law's tails (Chernoff bounds) to
/// leave ``1 - bucket_sizing_p`` of the mass outside, half on each side,
/// and the step is rounded up to 1, 2, 4, 5 or 8 times a power of ten
/// (a power of two below 1); when the dropped frequencies are above 1e-6
/// the step moves down that ladder while that helps. A Tweedie's mass at 0
/// is inverted apart, so it does not ring.
///
/// Parameters
/// ----------
/// law : Normal, Gamma, Exponential, InverseGaussian, Tweedie, Affine or callable
///     A law with a closed-form characteristic function (an ``Affine`` of
///     one of them), or your own characteristic function: a callable of a
///     list of real frequencies ``t`` returning ``E exp(i t X)`` at each,
///     complex. Your own function needs ``step`` and ``x_min``, since it
///     says nothing of the law's tails.
/// log2 : int, default 16
///     The grid has ``2 ** log2`` buckets, 1 to 28.
/// step : float, optional
///     The bucket; sized from the tails when left out.
/// x_min : float, optional
///     The grid's first point, with ``step``; placed from the tails when
///     left out.
/// bucket_sizing_p : float, default 0.99999
///     The mass the window covers when it is sized from the tails.
///
/// Returns
/// -------
/// ChfDistribution
///
/// Raises
/// ------
/// ValueError
///     If an option is out of range, ``x_min`` is given without ``step``,
///     your own function is given without both, or it returns a value that
///     is not finite.
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Gamma, invert_chf
/// >>> d = invert_chf(Gamma(4.0, 25.0), log2=12)
/// >>> d.step, d.x_min, round(d.mean(), 6)
/// (0.25, 0.0, 100.0)
#[pyfunction]
#[pyo3(signature = (law, *, log2 = 16, step = None, x_min = None, bucket_sizing_p = BUCKET_SIZING_P))]
pub(crate) fn invert_chf(
    py: Python<'_>,
    law: &Bound<'_, PyAny>,
    log2: u32,
    step: Option<f64>,
    x_min: Option<f64>,
    bucket_sizing_p: f64,
) -> PyResult<PyChfDistribution> {
    let law = if let Ok(n) = law.extract::<PyRef<'_, PyNormal>>() {
        ChfLaw::Normal(n.inner)
    } else if let Ok(d) = extract_dist(law) {
        ChfLaw::from_dist(&d).map_err(to_py)?
    } else if law.is_callable() {
        ChfLaw::User(UserChf::new(py_chf(law.clone().unbind()), "chf", false))
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "law must be a Normal, Gamma, Exponential, InverseGaussian, Tweedie or Affine, \
             or a callable characteristic function",
        ));
    };
    let options = ChfOptions {
        log2,
        step,
        x_min,
        bucket_sizing_p,
    };
    let inner = py.detach(|| invert(&law, &options)).map_err(to_py)?;
    Ok(PyChfDistribution { inner })
}
