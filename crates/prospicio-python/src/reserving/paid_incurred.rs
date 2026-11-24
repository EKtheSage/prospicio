//! Merz and Wuthrich's paid-incurred chain, `PaidIncurredChain`.

use super::*;

/// Merz and Wüthrich's (2010) paid-incurred chain (design note
/// ``reserving-v02.md``, decision 32): a paid and an incurred column of one
/// square triangle combined into one ultimate per origin, in closed form.
///
/// Log paid link ratios (and the log first paid values) are normal with
/// means ``Phi_j`` and standard deviations ``sigma_j``, log incurred link
/// ratios normal with means ``Psi_j`` and standard deviations ``tau_j``,
/// and both reach the same ultimate. With normal priors on ``Phi`` and
/// ``Psi`` (non-informative by default) the posterior is normal, and each
/// origin's ultimate is predicted from its latest paid and latest incurred
/// value, with the square root of its conditional mean squared error of
/// prediction. As ``tau`` grows the prediction becomes Hertig's paid chain
/// ladder, as it shrinks the incurred chain ladder on geometric averages.
///
/// The defaults reproduce R ChainLadder's ``PaidIncurredChain``, except
/// that ChainLadder 0.2.21 leaves out of the posterior precision the terms
/// coupling ``Psi_0`` with ``Phi_1, ..., Phi_J``, which the newest origin's
/// incurred-paid ratio implies: ``convention="model"`` is the model's
/// posterior, ``convention="r_chainladder"`` ChainLadder's to its last
/// digits. On ChainLadder's USAA example the model's total reserve and
/// standard error are each about 0.5% above ChainLadder's.
///
/// Parameters
/// ----------
/// sigma : sequence of float, optional
///     Paid standard deviations, one per development age; ``None``
///     estimates them (``sigma_0`` from the log first paid values).
/// tau : sequence of float, optional
///     Incurred standard deviations, one per development age but the last;
///     ``None`` estimates them.
/// extrapolation : {"loglinear", "mack"}, default "loglinear"
///     How the last estimated ``sigma`` and ``tau``, each with one
///     observation, are filled in: a log-linear fit to the others (R's
///     rule), or Mack's ``min(s[J-1], s[J-2], s[J-1]**2 / s[J-2])``.
/// convention : {"model", "r_chainladder"}, default "model"
///     The model's posterior, or R ChainLadder's.
/// phi_mean, phi_sd : sequence of float, optional
///     Prior means and standard deviations of ``Phi_0, ..., Phi_J``, one
///     per development age; ``None`` for a non-informative prior. An
///     infinite standard deviation leaves that parameter non-informative.
/// psi_mean, psi_sd : sequence of float, optional
///     The same for ``Psi_0, ..., Psi_{J-1}``, one per age but the last.
///
/// Raises
/// ------
/// ValueError
///     If ``extrapolation`` or ``convention`` is unknown, or only one of a
///     prior's means and standard deviations is given.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import PaidIncurredChain, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     {
/// ...         "paid": [100, 150, 170, 180, 110, 168, 186, 125, 180, 130],
/// ...         "incurred": [200, 195, 185, 181, 215, 200, 192, 240, 230, 260],
/// ...     },
/// ... )
/// >>> fit = PaidIncurredChain().fit(tri, "paid", "incurred")
/// >>> fit.reserve[0], fit.total_std_err > 0
/// (0.0, True)
#[pyclass(name = "PaidIncurredChain", module = "prospicio.reserving", frozen)]
pub(crate) struct PyPaidIncurredChain {
    inner: PaidIncurredChain,
}

/// A prior's means and standard deviations, both or neither.
fn pic_prior(
    mean: Option<Vec<f64>>,
    sd: Option<Vec<f64>>,
    name: &str,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    match (mean, sd) {
        (None, None) => Ok((Vec::new(), Vec::new())),
        (Some(m), Some(s)) => Ok((m, s)),
        _ => Err(PyValueError::new_err(format!(
            "give both {name}_mean and {name}_sd, or neither"
        ))),
    }
}

fn pic_extrapolation_name(e: VarianceExtrapolation) -> &'static str {
    e.name()
}

fn pic_convention_name(c: PicConvention) -> &'static str {
    c.name()
}

#[pymethods]
impl PyPaidIncurredChain {
    #[new]
    #[pyo3(signature = (
        sigma = None,
        tau = None,
        extrapolation = "loglinear",
        convention = "model",
        phi_mean = None,
        phi_sd = None,
        psi_mean = None,
        psi_sd = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        sigma: Option<Vec<f64>>,
        tau: Option<Vec<f64>>,
        extrapolation: &str,
        convention: &str,
        phi_mean: Option<Vec<f64>>,
        phi_sd: Option<Vec<f64>>,
        psi_mean: Option<Vec<f64>>,
        psi_sd: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let extrapolation = VarianceExtrapolation::from_name(extrapolation).ok_or_else(|| {
            crate::unknown_name("extrapolation", extrapolation, VarianceExtrapolation::NAMES)
        })?;
        let convention = PicConvention::from_name(convention)
            .ok_or_else(|| crate::unknown_name("convention", convention, PicConvention::NAMES))?;
        let (phi_mean, phi_sd) = pic_prior(phi_mean, phi_sd, "phi")?;
        let (psi_mean, psi_sd) = pic_prior(psi_mean, psi_sd, "psi")?;
        Ok(Self {
            inner: PaidIncurredChain {
                priors: PicPriors {
                    phi_mean,
                    phi_sd,
                    psi_mean,
                    psi_sd,
                },
                sigma,
                tau,
                extrapolation,
                convention,
            },
        })
    }

    /// The given paid standard deviations, or ``None``.
    #[getter]
    fn sigma(&self) -> Option<Vec<f64>> {
        self.inner.sigma.clone()
    }

    /// The given incurred standard deviations, or ``None``.
    #[getter]
    fn tau(&self) -> Option<Vec<f64>> {
        self.inner.tau.clone()
    }

    /// How the last estimated variances are filled in.
    #[getter]
    fn extrapolation(&self) -> &'static str {
        pic_extrapolation_name(self.inner.extrapolation)
    }

    /// The posterior precision used.
    #[getter]
    fn convention(&self) -> &'static str {
        pic_convention_name(self.inner.convention)
    }

    /// Fits the paid and incurred columns of a single-segment triangle.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     As many origins as development ages (at least four), each origin
    ///     observed at exactly its ages up to the latest diagonal, every
    ///     value positive.
    /// paid, incurred : str
    ///     Names of the paid and incurred columns.
    ///
    /// Returns
    /// -------
    /// PaidIncurredFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For a triangle of another shape or with several segments, a value
    ///     that is not positive, settings of the wrong length or out of
    ///     range, or an estimated variance of zero.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        incurred: &str,
    ) -> PyResult<PyPaidIncurredFit> {
        let (m, tri) = (&self.inner, &triangle.inner);
        let inner = py.detach(|| m.fit(tri, paid, incurred)).map_err(err)?;
        Ok(PyPaidIncurredFit { inner })
    }

    fn __repr__(&self) -> String {
        let given =
            |v: &Option<Vec<f64>>| v.as_ref().map_or("None".to_string(), |v| format!("{v:?}"));
        format!(
            "PaidIncurredChain(sigma={}, tau={}, extrapolation={:?}, convention={:?})",
            given(&self.inner.sigma),
            given(&self.inner.tau),
            self.extrapolation(),
            self.convention()
        )
    }
}

/// A paid-incurred chain fitted by ``PaidIncurredChain.fit``. Per-origin
/// lists run over the triangle's origins; the oldest is taken as settled at
/// its last paid value, with no reserve and no error.
#[pyclass(name = "PaidIncurredFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyPaidIncurredFit {
    inner: PicFit,
}

#[pymethods]
impl PyPaidIncurredFit {
    /// Origin period of each per-origin value.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// Development ages in months, one per period.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.ages.clone()
    }

    /// Latest paid value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.inner.latest_paid.clone()
    }

    /// Latest incurred value per origin.
    #[getter]
    fn latest_incurred(&self) -> Vec<f64> {
        self.inner.latest_incurred.clone()
    }

    /// Predicted ultimate per origin.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        self.inner.ultimate.clone()
    }

    /// Reserve per origin: the ultimate less the latest paid value.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        self.inner.reserve.clone()
    }

    /// Weight of the incurred side in each origin's prediction.
    #[getter]
    fn incurred_weight(&self) -> Vec<f64> {
        self.inner.incurred_weight.clone()
    }

    /// Square root of each origin's conditional mean squared error of
    /// prediction.
    #[getter]
    fn std_err(&self) -> Vec<f64> {
        self.inner.std_err.clone()
    }

    /// Total ultimate, the oldest origin's last paid value included.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    /// Total reserve.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    /// Square root of the total's conditional mean squared error of
    /// prediction, with the covariance between origins.
    #[getter]
    fn total_std_err(&self) -> f64 {
        self.inner.total_std_err
    }

    /// Paid standard deviations used, one per development age.
    #[getter]
    fn sigma(&self) -> Vec<f64> {
        self.inner.sigma.clone()
    }

    /// Incurred standard deviations used, one per age but the last.
    #[getter]
    fn tau(&self) -> Vec<f64> {
        self.inner.tau.clone()
    }

    /// Posterior means of ``Phi_0, ..., Phi_J``.
    #[getter]
    fn phi(&self) -> Vec<f64> {
        self.inner.phi().to_vec()
    }

    /// Posterior means of ``Psi_0, ..., Psi_{J-1}``.
    #[getter]
    fn psi(&self) -> Vec<f64> {
        self.inner.psi().to_vec()
    }

    /// Posterior standard deviations of ``Phi_0, ..., Phi_J``.
    #[getter]
    fn phi_sd(&self) -> Vec<f64> {
        let n = self.inner.ages.len();
        self.inner.posterior_sd()[..n].to_vec()
    }

    /// Posterior standard deviations of ``Psi_0, ..., Psi_{J-1}``.
    #[getter]
    fn psi_sd(&self) -> Vec<f64> {
        let n = self.inner.ages.len();
        self.inner.posterior_sd()[n..].to_vec()
    }

    /// Posterior covariance of ``(Phi_0, ..., Phi_J, Psi_0, ...,
    /// Psi_{J-1})``, as a list of rows.
    #[getter]
    fn posterior_covariance(&self) -> Vec<Vec<f64>> {
        let dim = self.inner.posterior_mean.len();
        self.inner
            .posterior_covariance
            .chunks(dim)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// One row per origin: ``origin``, ``latest`` (paid),
    /// ``latest_incurred``, ``ultimate``, ``reserve``, ``incurred_weight``
    /// and ``std_err``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the total ``latest``, ``latest_incurred``, ``ultimate``,
    /// ``reserve`` and the ``std_err`` of the total. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per development age: ``sigma``, ``phi``, ``phi_sd``,
    /// ``tau``, ``psi`` and ``psi_sd`` (NaN at the last age for the
    /// incurred side). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    fn __repr__(&self) -> String {
        format!(
            "PaidIncurredFit(origins={}, total_reserve={:?}, total_std_err={:?})",
            self.inner.origins.len(),
            self.inner.total_reserve(),
            self.inner.total_std_err
        )
    }
}
