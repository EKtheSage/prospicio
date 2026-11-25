//! `prospicio.reserving.ApcGlm` and `ApcGlmFit`: the age-period-cohort
//! reserving GLM (issue #979, `docs/design/reserving-inflation.md`, PR 7),
//! over `prospicio_reserving::apc`.

use prospicio_reserving::{ApcEffects, ApcGlm, ApcGlmFit, Extrapolation, FutureCells};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::distributions::PyPredictiveDistribution;
use crate::models::family;
use crate::odp_glm::PyOdpGlmFit;
use crate::reserving::{PyTriangle, err};
use crate::reserving_inflation::{PyFutureCells, cells_of};

fn effects_of(text: &str) -> PyResult<ApcEffects> {
    match text.to_ascii_uppercase().as_str() {
        "APC" => Ok(ApcEffects::Apc),
        "AP" => Ok(ApcEffects::AgePeriod),
        "AC" => Ok(ApcEffects::AccidentAge),
        "PC" => Ok(ApcEffects::AccidentPeriod),
        _ => Err(PyValueError::new_err(format!(
            "effects must be \"APC\", \"AP\", \"AC\" or \"PC\", got {text:?}"
        ))),
    }
}

fn effects_name(effects: ApcEffects) -> &'static str {
    match effects {
        ApcEffects::Apc => "APC",
        ApcEffects::AgePeriod => "AP",
        ApcEffects::AccidentAge => "AC",
        ApcEffects::AccidentPeriod => "PC",
    }
}

fn extrapolation_of(text: &str) -> PyResult<Extrapolation> {
    match text.to_ascii_uppercase().as_str() {
        "I0" => Ok(Extrapolation::I0),
        "I1" => Ok(Extrapolation::I1),
        "I2" => Ok(Extrapolation::I2),
        _ => Err(PyValueError::new_err(format!(
            "extrapolation must be \"I0\", \"I1\" or \"I2\", got {text:?}"
        ))),
    }
}

fn extrapolation_name(rule: Extrapolation) -> &'static str {
    match rule {
        Extrapolation::I0 => "I0",
        Extrapolation::I1 => "I1",
        Extrapolation::I2 => "I2",
    }
}

/// The age-period-cohort (APC) reserving GLM: ``log mu_ij = delta +
/// alpha_i + beta_j + gamma_(i+j)`` on the increments of a full triangle,
/// with the over-dispersed Poisson error (or the gamma or Tweedie of
/// ``OdpGlm``). The calendar index is the sum of the accident and
/// development indices, so the effects are identified only up to linear
/// trends. The fit uses the canonical parameters of Kuang, Nielsen and
/// Nielsen (2008a): a level, two slopes and the second differences of each
/// effect, ``3k - 3`` parameters on a ``k`` by ``k`` triangle. Fitted
/// values and the deviance do not depend on the identification.
///
/// The future cells need the calendar effect past the latest diagonal.
/// ``extrapolation`` chooses one of the three rules of R ``apc``, each
/// independent of the identification: ``"I2"`` sets the future second
/// differences to 0 (the last calendar trend continues), ``"I1"`` continues
/// with the mean of the fitted first differences (the first left out, as R
/// ``apc`` does), and ``"I0"`` (R ``apc``'s default, at least 4 origins)
/// puts the future effects on the least-squares line through the effects of
/// the periods from the third on. The rules are linear in the fitted second differences, so the
/// prediction error covers the extrapolated effect.
///
/// The sub-models drop one effect: ``"AP"`` (development and calendar,
/// Taylor's separation model as a GLM), ``"AC"`` (accident and
/// development, the chain ladder, where the extrapolation does not apply)
/// and ``"PC"`` (accident and calendar).
///
/// The triangle must be full: as many ages as origins, the origin grain
/// equal to the development grain, no hole.
///
/// Parameters
/// ----------
/// effects : {"APC", "AP", "AC", "PC"}, default "APC"
///     The effects in the model, as Python and R ``apc`` name them; any
///     case.
/// extrapolation : {"I0", "I1", "I2"}, default "I0"
///     The rule for the calendar effect past the latest diagonal; any case.
/// family : {"poisson", "gamma", "tweedie"}, default "poisson"
///     The error distribution, with the log link and Pearson's dispersion.
/// power : float, optional
///     The Tweedie power, in (1, 2); needed for, and only for, ``"tweedie"``.
/// tolerance : float, default 1e-10
///     IRLS convergence tolerance, as ``OdpGlm``'s.
/// max_iterations : int, default 100
///     Maximum IRLS iterations, as ``OdpGlm``'s.
///
/// Raises
/// ------
/// ValueError
///     For an unknown ``effects``, ``extrapolation`` or ``family``, a
///     ``power`` given for a family other than the Tweedie, or a Tweedie
///     power missing or outside (1, 2).
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ApcGlm, Triangle
/// >>> rows = [[100.0, 60.0, 20.0, 5.0], [110.0, 70.0, 24.0], [125.0, 75.0], [140.0]]
/// >>> tri = Triangle.from_long(
/// ...     [2001 + i for i, row in enumerate(rows) for _ in row],
/// ...     [12 * (j + 1) for row in rows for j in range(len(row))],
/// ...     {"paid": [v for row in rows for v in row]},
/// ...     cumulative=False,
/// ... )
/// >>> fit = ApcGlm(extrapolation="i1").fit(tri, "paid")
/// >>> fit.effects, fit.extrapolation
/// ('APC', 'I1')
/// >>> len(fit.names), fit.names[:4]
/// (9, ['level', 'slope_age', 'slope_accident', 'dd_age[2]'])
#[pyclass(name = "ApcGlm", module = "prospicio.reserving", frozen)]
pub(crate) struct PyApcGlm {
    inner: ApcGlm,
}

#[pymethods]
impl PyApcGlm {
    #[new]
    #[pyo3(signature = (effects = "APC", extrapolation = "I0", family = "poisson", power = None, tolerance = 1e-10, max_iterations = 100))]
    fn new(
        effects: &str,
        extrapolation: &str,
        family: &str,
        power: Option<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) -> PyResult<Self> {
        let effects = effects_of(effects)?;
        let extrapolation = extrapolation_of(extrapolation)?;
        if !matches!(family, "poisson" | "gamma" | "tweedie") {
            return Err(PyValueError::new_err(format!(
                "family must be poisson, gamma or tweedie, got {family:?}"
            )));
        }
        if family != "tweedie" && power.is_some() {
            return Err(PyValueError::new_err(
                "power is the Tweedie's; leave it out for the poisson and gamma",
            ));
        }
        Ok(Self {
            inner: ApcGlm {
                family: self::family(family, None, power)?,
                effects,
                extrapolation,
                tolerance,
                max_iterations,
            },
        })
    }

    /// ``"APC"``, ``"AP"``, ``"AC"`` or ``"PC"``.
    #[getter]
    fn effects(&self) -> &'static str {
        effects_name(self.inner.effects)
    }

    /// ``"I0"``, ``"I1"`` or ``"I2"``.
    #[getter]
    fn extrapolation(&self) -> &'static str {
        extrapolation_name(self.inner.extrapolation)
    }

    /// The error distribution: ``"poisson"``, ``"gamma"`` or ``"tweedie"``.
    #[getter]
    fn family(&self) -> &'static str {
        self.inner.family.name()
    }

    /// The Tweedie power, or ``None`` for the other families.
    #[getter]
    fn power(&self) -> Option<f64> {
        match self.inner.family {
            prospicio_models::Family::Tweedie { power } => Some(power),
            _ => None,
        }
    }

    /// IRLS convergence tolerance.
    #[getter]
    fn tolerance(&self) -> f64 {
        self.inner.tolerance
    }

    /// Maximum IRLS iterations.
    #[getter]
    fn max_iterations(&self) -> usize {
        self.inner.max_iterations
    }

    /// Fits ``column`` of a single-segment triangle and projects the cells
    /// below the latest diagonal.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///
    /// Returns
    /// -------
    /// ApcGlmFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For an unknown column, a triangle with several segments (fit one
    ///     at a time), a triangle that is not full, fewer than 3 origins
    ///     (4 for ``"I0"`` with a calendar effect), a tolerance that is not
    ///     finite and positive, no iterations, an increment the family
    ///     cannot fit, or an IRLS that does not converge.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyApcGlmFit> {
        let (model, tri) = (self.inner, &triangle.inner);
        let inner = py.detach(|| model.fit(tri, column)).map_err(err)?;
        Ok(PyApcGlmFit { inner })
    }

    #[allow(clippy::type_complexity)]
    fn __getnewargs__(
        &self,
    ) -> (
        &'static str,
        &'static str,
        &'static str,
        Option<f64>,
        f64,
        usize,
    ) {
        (
            self.effects(),
            self.extrapolation(),
            self.family(),
            self.power(),
            self.inner.tolerance,
            self.inner.max_iterations,
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyApcGlm>>()
            .is_ok_and(|o| o.inner == self.inner)
    }

    fn __repr__(&self) -> String {
        let power = match self.power() {
            Some(p) => format!(", power={p:?}"),
            None => String::new(),
        };
        format!(
            "ApcGlm(effects={:?}, extrapolation={:?}, family={:?}{power})",
            self.effects(),
            self.extrapolation(),
            self.family()
        )
    }
}

/// A fitted APC GLM, from ``ApcGlm.fit``.
///
/// Each effect's periods are indexed from 0; the second difference at
/// ``l`` (``l = 2..k - 1``) is the change in the effect's slope at period
/// ``l``.
///
/// Attributes
/// ----------
/// effects : str
///     ``"APC"``, ``"AP"``, ``"AC"`` or ``"PC"``.
/// extrapolation : str
///     ``"I0"``, ``"I1"`` or ``"I2"``.
/// origins : list of str
///     The origin periods; ``fitted`` and ``reserves`` follow them.
/// ages : list of int
///     The development ages in months; the columns of ``fitted`` follow them.
/// level : float
///     The log mean of the first origin's first age.
/// age_slope : float
///     The step in the log mean from age 0 to age 1, at origin 0.
/// accident_slope : float
///     The step in the log mean from accident 0 to accident 1, at age 0.
/// age_second_differences : list of float
///     The development effect's second differences, ``l = 2..k - 1``;
///     empty without a development effect.
/// accident_second_differences : list of float
///     The accident effect's second differences; empty without it.
/// calendar_second_differences : list of float
///     The calendar effect's fitted second differences; empty without it.
/// future_calendar_second_differences : list of float
///     The two extrapolated calendar second differences, at ``l = k`` and
///     ``l = k + 1`` (all later ones are 0); zeros for ``"I2"`` and without
///     a calendar effect.
/// fitted : list of list of float
///     The fitted mean of each observed cell, one row per origin, ``None``
///     below the latest diagonal.
/// deviance : float
///     The GLM's deviance on the observed cells.
/// reserves : list of float
///     The sum of each origin's future means; the first origin's is 0
///     (no tail past the oldest age).
/// total_reserve : float
/// coefficients : list of float
///     The canonical coefficients, in the order of ``names``.
/// names : list of str
///     ``level``, ``slope_age``, ``slope_accident``, then ``dd_age[l]``,
///     ``dd_accident[l]`` and ``dd_calendar[l]`` for the effects in the
///     model.
/// covariance : list of list of float
///     The coefficients' covariance, as a list of rows.
/// dispersion : float
///     Pearson's dispersion ``phi``.
/// future : FutureCells
///     The future cells' means.
/// odp_fit : OdpGlmFit
///     The GLM on the canonical design, with ``residuals``,
///     ``predict_distribution`` and the rest of ``OdpGlmFit``.
#[pyclass(name = "ApcGlmFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyApcGlmFit {
    inner: ApcGlmFit,
}

#[pymethods]
impl PyApcGlmFit {
    /// ``"APC"``, ``"AP"``, ``"AC"`` or ``"PC"``.
    #[getter]
    fn effects(&self) -> &'static str {
        effects_name(self.inner.effects)
    }

    /// ``"I0"``, ``"I1"`` or ``"I2"``: the rule used for the future cells.
    #[getter]
    fn extrapolation(&self) -> &'static str {
        extrapolation_name(self.inner.extrapolation)
    }

    /// The origin periods; ``fitted`` and ``reserves`` follow them.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// The development ages in months; the columns of ``fitted`` follow them.
    #[getter]
    fn ages(&self) -> Vec<u32> {
        self.inner.ages.clone()
    }

    /// The log mean of the first origin's first age.
    #[getter]
    fn level(&self) -> f64 {
        self.inner.level
    }

    /// The step in the log mean from age 0 to age 1, at origin 0.
    #[getter]
    fn age_slope(&self) -> f64 {
        self.inner.age_slope
    }

    /// The step in the log mean from accident 0 to accident 1, at age 0.
    #[getter]
    fn accident_slope(&self) -> f64 {
        self.inner.accident_slope
    }

    /// The development effect's second differences, ``l = 2..k - 1``; empty without it.
    #[getter]
    fn age_second_differences(&self) -> Vec<f64> {
        self.inner.age_second_differences.clone()
    }

    /// The accident effect's second differences; empty without it.
    #[getter]
    fn accident_second_differences(&self) -> Vec<f64> {
        self.inner.accident_second_differences.clone()
    }

    /// The calendar effect's fitted second differences; empty without it.
    #[getter]
    fn calendar_second_differences(&self) -> Vec<f64> {
        self.inner.calendar_second_differences.clone()
    }

    /// The two extrapolated calendar second differences, at ``l = k`` and ``l = k + 1``.
    #[getter]
    fn future_calendar_second_differences(&self) -> Vec<f64> {
        self.inner.future_calendar_second_differences.to_vec()
    }

    /// The fitted mean of each observed cell, ``None`` below the latest diagonal.
    #[getter]
    fn fitted(&self) -> Vec<Vec<Option<f64>>> {
        self.inner.fitted.clone()
    }

    /// The GLM's deviance on the observed cells.
    #[getter]
    fn deviance(&self) -> f64 {
        self.inner.deviance
    }

    /// The sum of each origin's future means; the first origin's is 0.
    #[getter]
    fn reserves(&self) -> Vec<f64> {
        self.inner.reserves.clone()
    }

    /// The total reserve.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    /// The canonical coefficients, in the order of ``names``.
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.inner.coefficients().to_vec()
    }

    /// The canonical coefficients' names.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    /// The coefficients' covariance, as a list of rows.
    #[getter]
    fn covariance(&self) -> Vec<Vec<f64>> {
        let p = self.inner.coefficients().len();
        self.inner
            .covariance()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Pearson's dispersion ``phi``.
    #[getter]
    fn dispersion(&self) -> f64 {
        self.inner.fit.glm.dispersion()
    }

    /// The future cells' means, as ``FutureCells``.
    #[getter]
    fn future(&self) -> PyResult<PyFutureCells> {
        FutureCells::from_apc_glm(&self.inner)
            .map(cells_of)
            .map_err(err)
    }

    /// The GLM on the canonical design, as ``OdpGlmFit``: residuals,
    /// ``predict_distribution`` and the future design.
    #[getter]
    fn odp_fit(&self) -> PyOdpGlmFit {
        PyOdpGlmFit {
            inner: self.inner.fit.clone(),
        }
    }

    /// The reserves' prediction error by origin and in total, as
    /// ``OdpGlmFit.prediction_error``: the process error and the parameter
    /// error by the delta method. The extrapolated calendar effect is a
    /// fixed linear function of the fitted second differences, so its
    /// uncertainty is included.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``origin`` (the labels, then ``"Total"``), ``reserve``,
    ///     ``process_se``, ``parameter_se``, ``rmsep`` and ``cv``, as lists.
    fn prediction_error<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.odp_fit().prediction_error(py)
    }

    /// Joint draws of the future cells' increments, as
    /// ``OdpGlmFit.predict_distribution``: the same draws as
    /// ``fit.odp_fit.predict_distribution`` for the same arguments.
    ///
    /// Parameters
    /// ----------
    /// n_sims : int
    /// seed : int
    /// parameters : {"mean_preserving", "normal", "fixed"}, default "mean_preserving"
    ///     How the coefficients are drawn, as ``OdpGlmFit``'s.
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``n_sims`` is 0 or ``parameters`` is unknown.
    #[pyo3(signature = (n_sims, seed, parameters = "mean_preserving"))]
    fn predict_distribution(
        &self,
        py: Python<'_>,
        n_sims: usize,
        seed: u64,
        parameters: &str,
    ) -> PyResult<PyPredictiveDistribution> {
        self.odp_fit()
            .predict_distribution(py, n_sims, seed, parameters)
    }

    fn __repr__(&self) -> String {
        format!(
            "ApcGlmFit(effects={:?}, extrapolation={:?}, origins={}, total_reserve={:?})",
            self.effects(),
            self.extrapolation(),
            self.inner.origins.len(),
            self.inner.total_reserve()
        )
    }
}
