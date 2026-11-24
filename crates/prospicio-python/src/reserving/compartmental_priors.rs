//! The compartmental model's priors, `CompartmentalPriors` (`docs/design/reserving-compartmental.md`).

use super::*;

// ---------------------------------------------------------------------------
// Gesmann and Morris' compartmental model (docs/design/reserving-compartmental.md)
// ---------------------------------------------------------------------------

/// `(median, sd)` of a log-normal prior on a positive parameter.
fn positive_prior(name: &str, value: (f64, f64)) -> PyResult<PositivePrior> {
    let (median, sd) = value;
    if !(median.is_finite() && median > 0.0 && sd.is_finite() && sd > 0.0) {
        return Err(PyValueError::new_err(format!(
            "{name} must be (median, sd) with both finite and positive, got ({median}, {sd})"
        )));
    }
    Ok(PositivePrior::LogNormal { median, sd })
}

fn positive_prior_tuple(prior: PositivePrior) -> (f64, f64) {
    let PositivePrior::LogNormal { median, sd } = prior;
    (median, sd)
}

/// A half prior from ``("student_t", nu, scale)`` or ``("normal", scale)``.
fn half_prior(name: &str, value: &Bound<'_, PyTuple>) -> PyResult<HalfPrior> {
    let bad = || {
        PyValueError::new_err(format!(
            "{name} must be (\"student_t\", nu, scale) or (\"normal\", scale)"
        ))
    };
    let family: String = value.get_item(0).map_err(|_| bad())?.extract()?;
    let num = |i: usize| -> PyResult<f64> { value.get_item(i).map_err(|_| bad())?.extract() };
    match (family.to_ascii_lowercase().as_str(), value.len()) {
        ("student_t", 3) => Ok(HalfPrior::StudentT {
            nu: num(1)?,
            scale: num(2)?,
        }),
        ("normal", 2) => Ok(HalfPrior::Normal { scale: num(1)? }),
        _ => Err(bad()),
    }
}

fn half_prior_tuple<'py>(py: Python<'py>, prior: HalfPrior) -> PyResult<Bound<'py, PyTuple>> {
    match prior {
        HalfPrior::StudentT { nu, scale } => ("student_t", nu, scale).into_pyobject(py),
        HalfPrior::Normal { scale } => ("normal", scale).into_pyobject(py),
    }
}

/// A real prior from ``("student_t", nu, location, scale)`` or
/// ``("normal", mean, sd)``.
fn real_prior(name: &str, value: &Bound<'_, PyTuple>) -> PyResult<RealPrior> {
    let bad = || {
        PyValueError::new_err(format!(
            "{name} must be (\"student_t\", nu, location, scale) or (\"normal\", mean, sd)"
        ))
    };
    let family: String = value.get_item(0).map_err(|_| bad())?.extract()?;
    let num = |i: usize| -> PyResult<f64> { value.get_item(i).map_err(|_| bad())?.extract() };
    match (family.to_ascii_lowercase().as_str(), value.len()) {
        ("student_t", 4) => Ok(RealPrior::StudentT {
            nu: num(1)?,
            location: num(2)?,
            scale: num(3)?,
        }),
        ("normal", 3) => Ok(RealPrior::Normal {
            mean: num(1)?,
            sd: num(2)?,
        }),
        _ => Err(bad()),
    }
}

fn real_prior_tuple<'py>(py: Python<'py>, prior: RealPrior) -> PyResult<Bound<'py, PyTuple>> {
    match prior {
        RealPrior::StudentT {
            nu,
            location,
            scale,
        } => ("student_t", nu, location, scale).into_pyobject(py),
        RealPrior::Normal { mean, sd } => ("normal", mean, sd).into_pyobject(py),
    }
}

fn log_sigma_priors(value: &[Bound<'_, PyTuple>]) -> PyResult<[RealPrior; 2]> {
    if value.len() != 2 {
        return Err(PyValueError::new_err(
            "log_sigma must hold two priors: log sigma_os, then log sigma_pd",
        ));
    }
    Ok([
        real_prior("log_sigma[0]", &value[0])?,
        real_prior("log_sigma[1]", &value[1])?,
    ])
}

/// An optional half prior: ``None`` (no such effect) or a half prior's
/// tuple.
fn optional_half_prior(
    name: &str,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<HalfPrior>> {
    match value {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => {
            let t = v.cast::<PyTuple>().map_err(|_| {
                PyValueError::new_err(format!(
                    "{name} must be None, (\"student_t\", nu, scale) or (\"normal\", scale)"
                ))
            })?;
            Ok(Some(half_prior(name, t)?))
        }
    }
}

fn optional_half_prior_tuple<'py>(
    py: Python<'py>,
    prior: Option<HalfPrior>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    prior.map(|p| half_prior_tuple(py, p)).transpose()
}

/// The development-year effect sds on ``k_er``, ``k_p``, ``rlr`` and
/// ``rrf``: ``None`` (none) or four items, each ``None`` or a half prior.
fn development_effect_sds(value: Option<&Bound<'_, PyAny>>) -> PyResult<[Option<HalfPrior>; 4]> {
    let Some(v) = value.filter(|v| !v.is_none()) else {
        return Ok([None; 4]);
    };
    let bad = || {
        PyValueError::new_err(
            "development_effect_sd must be None or four priors, on k_er, k_p, rlr and rrf in \
             that order, each None or a half prior",
        )
    };
    let items: Vec<Bound<'_, PyAny>> = v.try_iter().map_err(|_| bad())?.collect::<PyResult<_>>()?;
    if items.len() != 4 {
        return Err(bad());
    }
    let mut out = [None; 4];
    for (i, (slot, item)) in out.iter_mut().zip(&items).enumerate() {
        *slot = optional_half_prior(&format!("development_effect_sd[{i}]"), Some(item))?;
    }
    Ok(out)
}

/// ``replace``'s value for an optional effect: ``None`` keeps it,
/// ``False`` removes it, a half prior's tuple sets it.
fn replace_effect(
    name: &str,
    value: Option<&Bound<'_, PyAny>>,
    current: Option<HalfPrior>,
) -> PyResult<Option<HalfPrior>> {
    match value {
        None => Ok(current),
        Some(v) if v.is_none() => Ok(current),
        Some(v) if v.is_instance_of::<PyBool>() => {
            if v.extract::<bool>()? {
                Err(PyValueError::new_err(format!(
                    "{name} must be False (no effect) or a half prior, not True"
                )))
            } else {
                Ok(None)
            }
        }
        Some(v) => optional_half_prior(name, Some(v)),
    }
}

/// The priors of ``Compartmental``, a required argument: the rates and
/// ratios are specific to a line and a book, so the library does not choose
/// them silently. ``CompartmentalPriors.gesmann_morris()`` is the paper's
/// model 1 and ``CompartmentalPriors.gesmann_morris_2()`` its model 2;
/// check any set with ``Compartmental.prior_predictive`` before fitting.
/// ``CompartmentalPriors.diffuse()`` is the starting point for a fit to paid alone
/// (``fitted="paid_only"``): it leaves the rates to the data. A fit to incurred alone
/// (``fitted="incurred_only"``) refuses a ``k_p`` prior with a log-scale sd above 0.5,
/// so give it a prior you believe, such as the paper's. Use a narrower prior than the
/// preset also when you hold information about the rates that is not in the data.
///
/// Each positive parameter ``θ`` (``k_er``, ``k_p``, ``rlr``, ``rrf``) is
/// ``median * exp(sd * o)`` with ``o ~ N(0, 1)``: a log-normal prior with
/// that median and log-scale sd, given as ``(median, sd)``. The effects
/// live on that ``o`` scale.
///
/// The hierarchy follows from the priors. ``RLR`` and ``RRF`` always vary
/// by origin, with correlated effects. ``ker_effect_sd`` and
/// ``kp_effect_sd`` add origin effects on the rates, uncorrelated with the
/// others, and ``development_effect_sd`` adds development-year effects on
/// any of the four parameters; ``None`` (model 1) means no such effect.
/// A development age with no observed cell takes a new effect from the
/// population in each simulation, as an origin with no cell does.
///
/// Parameters
/// ----------
/// k_er, k_p : tuple of (float, float)
///     ``(median, sd)`` of the reporting rate and the payment rate, per year.
/// rlr, rrf : tuple of (float, float)
///     ``(median, sd)`` of the population reported loss ratio and reserve
///     robustness factor (paid over reported).
/// rlr_effect_sd, rrf_effect_sd : tuple
///     Priors on the sds of the origin effects on the ``o`` scale:
///     ``("student_t", nu, scale)`` for a half Student-t or
///     ``("normal", scale)`` for a half normal.
/// lkj_eta : float
///     LKJ shape of the correlation of the two effects (1: uniform).
/// log_sigma : sequence of two tuples
///     Priors on ``log sigma_os`` and ``log sigma_pd``, in that order, each
///     ``("student_t", nu, location, scale)`` or ``("normal", mean, sd)``.
/// ker_effect_sd, kp_effect_sd : tuple, optional
///     Priors on the sds of origin effects on ``k_er`` and ``k_p``, in the
///     form of ``rlr_effect_sd``; ``None``: no such effect.
/// development_effect_sd : sequence of four, optional
///     Priors on the sds of development-year effects on ``k_er``, ``k_p``,
///     ``rlr`` and ``rrf``, in that order, each ``None`` (no effect) or a
///     half prior; ``None``: no development effect.
///
/// Raises
/// ------
/// ValueError
///     If a prior has the wrong form or a log-normal median or sd is not
///     finite and positive. The other settings are checked by the fit.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import CompartmentalPriors
/// >>> p = CompartmentalPriors.gesmann_morris()
/// >>> p.k_er, p.rlr_effect_sd, p.log_sigma[0]
/// ((3.0, 0.1), ('student_t', 10.0, 0.2), ('student_t', 1.0, 0.0, 1000.0))
/// >>> p.replace(k_er=(2.0, 0.3)).k_er
/// (2.0, 0.3)
/// >>> q = CompartmentalPriors.gesmann_morris_2()
/// >>> q.kp_effect_sd, q.development_effect_sd[2]
/// (('student_t', 10.0, 0.3), ('student_t', 10.0, 0.7))
/// >>> q.replace(kp_effect_sd=False).kp_effect_sd is None
/// True
#[pyclass(name = "CompartmentalPriors", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCompartmentalPriors {
    pub(crate) inner: CompartmentalPriors,
}

#[pymethods]
impl PyCompartmentalPriors {
    #[new]
    #[pyo3(signature = (
        *,
        k_er,
        k_p,
        rlr,
        rrf,
        rlr_effect_sd,
        rrf_effect_sd,
        lkj_eta,
        log_sigma,
        ker_effect_sd = None,
        kp_effect_sd = None,
        development_effect_sd = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        k_er: (f64, f64),
        k_p: (f64, f64),
        rlr: (f64, f64),
        rrf: (f64, f64),
        rlr_effect_sd: Bound<'_, PyTuple>,
        rrf_effect_sd: Bound<'_, PyTuple>,
        lkj_eta: f64,
        log_sigma: Vec<Bound<'_, PyTuple>>,
        ker_effect_sd: Option<Bound<'_, PyAny>>,
        kp_effect_sd: Option<Bound<'_, PyAny>>,
        development_effect_sd: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CompartmentalPriors {
                k_er: positive_prior("k_er", k_er)?,
                k_p: positive_prior("k_p", k_p)?,
                rlr: positive_prior("rlr", rlr)?,
                rrf: positive_prior("rrf", rrf)?,
                rlr_effect_sd: half_prior("rlr_effect_sd", &rlr_effect_sd)?,
                rrf_effect_sd: half_prior("rrf_effect_sd", &rrf_effect_sd)?,
                lkj_eta,
                log_sigma: log_sigma_priors(&log_sigma)?,
                ker_effect_sd: optional_half_prior("ker_effect_sd", ker_effect_sd.as_ref())?,
                kp_effect_sd: optional_half_prior("kp_effect_sd", kp_effect_sd.as_ref())?,
                development_effect_sd: development_effect_sds(development_effect_sd.as_ref())?,
            },
        })
    }

    /// Gesmann and Morris' (2020) case study, model 1: ``k_er`` (3, 0.1),
    /// ``k_p`` (1, 0.1), ``rlr`` (0.7, 0.2), ``rrf`` (0.8, 0.1), half
    /// Student-t(10) effect sds with scales 0.2 and 0.1, LKJ(1), and
    /// Student-t(1, 0, 1000) on each ``log sigma`` (effectively flat), with
    /// no effect on the rates and no development effect. On company 337 the
    /// data pull the rates 5 to 9 prior sds from these medians: the
    /// posterior is a compromise with the preset.
    ///
    /// Returns
    /// -------
    /// CompartmentalPriors
    #[staticmethod]
    fn gesmann_morris() -> Self {
        Self {
            inner: CompartmentalPriors::gesmann_morris(),
        }
    }

    /// A diffuse set for model 1: the paper's medians (``k_er`` 3, ``k_p`` 1,
    /// ``rlr`` 0.7, ``rrf`` 0.8) with a log-scale sd of 1 on each of the four,
    /// 5 to 10 times the paper's, and its effect sds and ``log sigma`` priors.
    /// Use it as the starting point for a fit to paid alone
    /// (``fitted="paid_only"``), where the paid cells pin the rates. It is
    /// refused for a fit to incurred alone (``fitted="incurred_only"``): no
    /// paid cell is fitted there, so the payment rate ``k_p`` and with it the
    /// paid run-off and the reserve follow the prior on ``k_p`` at every width
    /// (on company 337 -51,000 with an sd of about 150,000 at a log-scale sd of
    /// 1, 203,000 at 0.3), and a ``k_p`` prior with a log-scale sd above 0.5 is
    /// an error. For incurred alone replace ``k_p`` with a prior you believe
    /// (``replace``), for example one from the book's paid history or from a
    /// paid-and-incurred fit of a similar book; the paper's narrow prior put
    /// the reserve on company 337 26% above the fit to paid and outstanding.
    ///
    /// Returns
    /// -------
    /// CompartmentalPriors
    #[staticmethod]
    fn diffuse() -> Self {
        Self {
            inner: CompartmentalPriors::diffuse(),
        }
    }

    /// Gesmann and Morris' (2020) case study, model 2 (their code, sec.
    /// 7.2.3, where it and the text disagree): model 1's rate and ratio
    /// priors; origin effects on all four parameters (``RLR`` and ``RRF``
    /// correlated, LKJ(1)) and development-year effects on all four, every
    /// effect sd half Student-t(10) on the ``o`` scale with one scale per
    /// parameter, shared by its two groups: 0.3 for ``k_er`` and ``k_p``,
    /// 0.7 for ``rlr`` and 0.5 for ``rrf``; and ``("normal", log 0.2,
    /// 0.2)`` on each ``log sigma`` (the text gives a median of 0.1). Use
    /// it with ``Compartmental.model_2``.
    ///
    /// Returns
    /// -------
    /// CompartmentalPriors
    #[staticmethod]
    fn gesmann_morris_2() -> Self {
        Self {
            inner: CompartmentalPriors::gesmann_morris_2(),
        }
    }

    /// A copy with the given priors replaced, each in the constructor's
    /// form. For ``ker_effect_sd`` and ``kp_effect_sd``, ``None`` keeps the
    /// current prior and ``False`` removes the effect; a
    /// ``development_effect_sd`` replaces all four, its ``None`` items
    /// removing those effects.
    ///
    /// Returns
    /// -------
    /// CompartmentalPriors
    #[pyo3(signature = (
        *,
        k_er = None,
        k_p = None,
        rlr = None,
        rrf = None,
        rlr_effect_sd = None,
        rrf_effect_sd = None,
        lkj_eta = None,
        log_sigma = None,
        ker_effect_sd = None,
        kp_effect_sd = None,
        development_effect_sd = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn replace(
        &self,
        k_er: Option<(f64, f64)>,
        k_p: Option<(f64, f64)>,
        rlr: Option<(f64, f64)>,
        rrf: Option<(f64, f64)>,
        rlr_effect_sd: Option<Bound<'_, PyTuple>>,
        rrf_effect_sd: Option<Bound<'_, PyTuple>>,
        lkj_eta: Option<f64>,
        log_sigma: Option<Vec<Bound<'_, PyTuple>>>,
        ker_effect_sd: Option<Bound<'_, PyAny>>,
        kp_effect_sd: Option<Bound<'_, PyAny>>,
        development_effect_sd: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut p = self.inner;
        if let Some(v) = k_er {
            p.k_er = positive_prior("k_er", v)?;
        }
        if let Some(v) = k_p {
            p.k_p = positive_prior("k_p", v)?;
        }
        if let Some(v) = rlr {
            p.rlr = positive_prior("rlr", v)?;
        }
        if let Some(v) = rrf {
            p.rrf = positive_prior("rrf", v)?;
        }
        if let Some(v) = rlr_effect_sd {
            p.rlr_effect_sd = half_prior("rlr_effect_sd", &v)?;
        }
        if let Some(v) = rrf_effect_sd {
            p.rrf_effect_sd = half_prior("rrf_effect_sd", &v)?;
        }
        if let Some(v) = lkj_eta {
            p.lkj_eta = v;
        }
        if let Some(v) = log_sigma {
            p.log_sigma = log_sigma_priors(&v)?;
        }
        p.ker_effect_sd = replace_effect("ker_effect_sd", ker_effect_sd.as_ref(), p.ker_effect_sd)?;
        p.kp_effect_sd = replace_effect("kp_effect_sd", kp_effect_sd.as_ref(), p.kp_effect_sd)?;
        if let Some(v) = development_effect_sd.filter(|v| !v.is_none()) {
            p.development_effect_sd = development_effect_sds(Some(&v))?;
        }
        Ok(Self { inner: p })
    }

    /// ``(median, sd)`` of the reporting rate.
    #[getter]
    fn k_er(&self) -> (f64, f64) {
        positive_prior_tuple(self.inner.k_er)
    }

    /// ``(median, sd)`` of the payment rate.
    #[getter]
    fn k_p(&self) -> (f64, f64) {
        positive_prior_tuple(self.inner.k_p)
    }

    /// ``(median, sd)`` of the population reported loss ratio.
    #[getter]
    fn rlr(&self) -> (f64, f64) {
        positive_prior_tuple(self.inner.rlr)
    }

    /// ``(median, sd)`` of the population reserve robustness factor.
    #[getter]
    fn rrf(&self) -> (f64, f64) {
        positive_prior_tuple(self.inner.rrf)
    }

    /// Prior on the sd of the origin effects on ``RLR``.
    #[getter]
    fn rlr_effect_sd<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        half_prior_tuple(py, self.inner.rlr_effect_sd)
    }

    /// Prior on the sd of the origin effects on ``RRF``.
    #[getter]
    fn rrf_effect_sd<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        half_prior_tuple(py, self.inner.rrf_effect_sd)
    }

    /// LKJ shape of the correlation of the two effects.
    #[getter]
    fn lkj_eta(&self) -> f64 {
        self.inner.lkj_eta
    }

    /// Priors on ``log sigma_os`` and ``log sigma_pd``.
    #[getter]
    fn log_sigma<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        self.inner
            .log_sigma
            .iter()
            .map(|&p| real_prior_tuple(py, p))
            .collect()
    }

    /// Prior on the sd of the origin effects on ``k_er``; ``None``: no such
    /// effect.
    #[getter]
    fn ker_effect_sd<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        optional_half_prior_tuple(py, self.inner.ker_effect_sd)
    }

    /// Prior on the sd of the origin effects on ``k_p``; ``None``: no such
    /// effect.
    #[getter]
    fn kp_effect_sd<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        optional_half_prior_tuple(py, self.inner.kp_effect_sd)
    }

    /// Priors on the sds of the development-year effects on ``k_er``,
    /// ``k_p``, ``rlr`` and ``rrf``, each ``None`` when there is no such
    /// effect.
    #[getter]
    fn development_effect_sd<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Option<Bound<'py, PyTuple>>>> {
        self.inner
            .development_effect_sd
            .iter()
            .map(|&p| optional_half_prior_tuple(py, p))
            .collect()
    }

    fn __eq__(&self, other: PyRef<'_, Self>) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        let p = &self.inner;
        let half = |h: HalfPrior| match h {
            HalfPrior::StudentT { nu, scale } => format!("(\"student_t\", {nu:?}, {scale:?})"),
            HalfPrior::Normal { scale } => format!("(\"normal\", {scale:?})"),
        };
        let maybe = |h: Option<HalfPrior>| h.map_or_else(|| "None".to_string(), half);
        let real = |r: RealPrior| match r {
            RealPrior::StudentT {
                nu,
                location,
                scale,
            } => format!("(\"student_t\", {nu:?}, {location:?}, {scale:?})"),
            RealPrior::Normal { mean, sd } => format!("(\"normal\", {mean:?}, {sd:?})"),
        };
        let pos = |v: PositivePrior| format!("{:?}", positive_prior_tuple(v));
        let dev = if p.development_effect_sd.iter().all(Option::is_none) {
            "None".to_string()
        } else {
            let items: Vec<String> = p.development_effect_sd.iter().map(|&h| maybe(h)).collect();
            format!("[{}]", items.join(", "))
        };
        format!(
            "CompartmentalPriors(k_er={}, k_p={}, rlr={}, rrf={}, rlr_effect_sd={}, \
             rrf_effect_sd={}, lkj_eta={:?}, log_sigma=[{}, {}], ker_effect_sd={}, \
             kp_effect_sd={}, development_effect_sd={})",
            pos(p.k_er),
            pos(p.k_p),
            pos(p.rlr),
            pos(p.rrf),
            half(p.rlr_effect_sd),
            half(p.rrf_effect_sd),
            p.lkj_eta,
            real(p.log_sigma[0]),
            real(p.log_sigma[1]),
            maybe(p.ker_effect_sd),
            maybe(p.kp_effect_sd),
            dev,
        )
    }
}

/// A setting given as a string, with its accepted names in order.
pub(crate) fn named<T: Copy>(setting: &str, name: &str, values: &[(&str, T)]) -> PyResult<T> {
    let lower = name.to_ascii_lowercase();
    values
        .iter()
        .find(|(n, _)| *n == lower)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            let expected: Vec<String> = values.iter().map(|(n, _)| format!("{n:?}")).collect();
            PyValueError::new_err(format!(
                "unknown {setting} {name:?}; expected one of {}",
                expected.join(", ")
            ))
        })
}

/// The name of a setting's value.
pub(crate) fn name_of<T: PartialEq>(value: T, values: &[(&'static str, T)]) -> &'static str {
    values
        .iter()
        .find(|(_, v)| *v == value)
        .map_or("?", |(n, _)| n)
}
