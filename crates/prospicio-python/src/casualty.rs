//! `prospicio.pricing` casualty exposure rating (Aggregate lane):
//! `CasualtyProfile`, `IlfTable`, `ElfTable`, `CasualtyRating`,
//! `ParetoAnchor` and `poisson_tail` over `prospicio_pricing::casualty`.

use prospicio_pricing::casualty::{
    Alae, CasualtyCurve, CasualtyPolicy, CasualtyProfile, CasualtyRating, ElfTable, IlfTable,
    ParetoAnchor,
};
use prospicio_pricing::layer::XsLayer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::distributions::extract_severity;
use crate::to_py;

/// ``alae`` and its ``loading``: there is no default treatment.
fn alae(name: &str, loading: Option<f64>) -> PyResult<Alae> {
    let need = || {
        loading.ok_or_else(|| {
            PyValueError::new_err(format!("alae={name:?} needs a loading (0 for none)"))
        })
    };
    match name {
        "pro_rata" => Ok(Alae::ProRata(need()?)),
        "on_top" => Ok(Alae::OnTop(need()?)),
        "in_severity" => match loading {
            None => Ok(Alae::InSeverity),
            Some(_) => Err(PyValueError::new_err(
                "alae=\"in_severity\" takes no loading: the severity holds the ALAE",
            )),
        },
        _ => Err(PyValueError::new_err(format!(
            "alae must be \"pro_rata\", \"on_top\" or \"in_severity\", not {name:?}"
        ))),
    }
}

fn treaty(limit: f64, attachment: f64) -> PyResult<XsLayer> {
    XsLayer::new(limit, attachment).map_err(to_py)
}

/// An ILF table read as a limited expected value curve: linear between the
/// limits and from the origin to the first, checked concave, refused beyond
/// the last limit.
///
/// Parameters
/// ----------
/// limits : list of float
///     Positive and increasing.
/// factors : list of float
///     The increased limit factor at each limit.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import IlfTable
/// >>> IlfTable([100e3, 1e6, 2e6], [1.0, 1.6, 1.8]).ilf(1.5e6)
/// 1.7
#[pyclass(name = "IlfTable", module = "prospicio.pricing", frozen)]
pub(crate) struct PyIlfTable {
    inner: IlfTable,
}

#[pymethods]
impl PyIlfTable {
    #[new]
    fn new(limits: Vec<f64>, factors: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: IlfTable::new(&limits, &factors).map_err(to_py)?,
        })
    }

    /// The factor at ``limit``.
    fn ilf(&self, limit: f64) -> PyResult<f64> {
        self.inner.ilf(limit).map_err(to_py)
    }
}

/// An excess loss factor table, ``ELF(x) = E[(X - x)+] / E[X]``, read
/// between its limits as ``IlfTable`` is (``ELF(0) = 1``).
///
/// Parameters
/// ----------
/// limits : list of float
/// elfs : list of float
///     Decreasing, in ``(0, 1)``.
#[pyclass(name = "ElfTable", module = "prospicio.pricing", frozen)]
pub(crate) struct PyElfTable {
    inner: ElfTable,
}

#[pymethods]
impl PyElfTable {
    #[new]
    fn new(limits: Vec<f64>, elfs: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: ElfTable::new(&limits, &elfs).map_err(to_py)?,
        })
    }

    /// The factor at ``limit`` (0 at ``inf``).
    fn elf(&self, limit: f64) -> PyResult<f64> {
        self.inner.elf(limit).map_err(to_py)
    }
}

fn curve(obj: &Bound<'_, PyAny>) -> PyResult<CasualtyCurve> {
    if let Ok(t) = obj.cast::<PyIlfTable>() {
        return Ok(CasualtyCurve::Ilf(t.get().inner.clone()));
    }
    if let Ok(t) = obj.cast::<PyElfTable>() {
        return Ok(CasualtyCurve::Elf(t.get().inner.clone()));
    }
    extract_severity(obj)
        .map(CasualtyCurve::severity)
        .map_err(|_| PyTypeError::new_err("a curve is a severity, an IlfTable or an ElfTable"))
}

fn column<T: Clone>(v: Option<Vec<T>>, n: usize, name: &str, default: T) -> PyResult<Vec<T>> {
    match v {
        None => Ok(vec![default; n]),
        Some(v) if v.len() == n => Ok(v),
        Some(v) => Err(PyValueError::new_err(format!(
            "{name} has {} values for {n} policies",
            v.len()
        ))),
    }
}

/// A book of casualty policies, each rated on its own curve
/// (``docs/design/casualty-exposure.md``).
///
/// Policy ``i`` pays ``share * min((X - attachment)+, limit)`` for a
/// ground-up loss ``X`` per occurrence. A per-policy treaty ``limit`` xs
/// ``attachment`` cedes the part above its retention, and the share of the
/// policy's expected loss it takes is its exposure factor.
///
/// Parameters
/// ----------
/// curves : list
///     Severities, ``IlfTable`` or ``ElfTable``; each policy names one.
/// limit, premium, elr : list of float
///     One per policy: the policy limit above its attachment (``inf`` for
///     none, as in workers compensation), on-level premium and expected
///     loss ratio.
/// attachment, share : list of float, optional
///     Underlying attachment (0) and share of the layer written (1).
/// curve : list of int, optional
///     Each policy's index into ``curves`` (0).
/// tag : list of str or None, optional
///     A segment or line, for reporting and for leaving a segment out.
///
/// Examples
/// --------
/// Clark's workers compensation example, 750,000 xs 250,000:
///
/// >>> from prospicio.pricing import CasualtyProfile, ElfTable
/// >>> book = CasualtyProfile(
/// ...     [ElfTable([250e3, 1e6], [0.030, 0.006])], limit=[float("inf")],
/// ...     premium=[100e3], elr=[0.70])
/// >>> round(book.rate(750e3, 250e3, alae="in_severity").ceded, 6)
/// 1680.0
#[pyclass(name = "CasualtyProfile", module = "prospicio.pricing", frozen)]
pub(crate) struct PyCasualtyProfile {
    inner: CasualtyProfile,
}

#[pymethods]
impl PyCasualtyProfile {
    #[new]
    #[pyo3(signature = (curves, *, limit, premium, elr, attachment = None, share = None, curve = None, tag = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        curves: Vec<Bound<'_, PyAny>>,
        limit: Vec<f64>,
        premium: Vec<f64>,
        elr: Vec<f64>,
        attachment: Option<Vec<f64>>,
        share: Option<Vec<f64>>,
        curve: Option<Vec<usize>>,
        tag: Option<Vec<Option<String>>>,
    ) -> PyResult<Self> {
        let curves = curves
            .iter()
            .map(self::curve)
            .collect::<PyResult<Vec<_>>>()?;
        let n = limit.len();
        let premium = column(Some(premium), n, "premium", 0.0)?;
        let elr = column(Some(elr), n, "elr", 0.0)?;
        let attachment = column(attachment, n, "attachment", 0.0)?;
        let share = column(share, n, "share", 1.0)?;
        let index = column(curve, n, "curve", 0)?;
        let tag = column(tag, n, "tag", None)?;
        let policies = (0..n)
            .map(|i| {
                let p = CasualtyPolicy::new(limit[i], premium[i], elr[i])
                    .attachment(attachment[i])
                    .share(share[i])
                    .curve(index[i]);
                match &tag[i] {
                    Some(t) => p.tag(t.clone()),
                    None => p,
                }
            })
            .collect();
        Ok(Self {
            inner: CasualtyProfile::new(curves, policies).map_err(to_py)?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.policies().len()
    }

    /// Each policy's exposure factor for the treaty ``limit`` xs
    /// ``attachment``.
    ///
    /// Parameters
    /// ----------
    /// limit, attachment : float
    /// alae : {"pro_rata", "on_top", "in_severity"}
    ///     ``pro_rata``: ALAE in proportion to loss, the ceded loss times
    ///     ``1 + loading`` on a loss-only ELR. ``on_top``: the policy limit
    ///     caps loss only and ALAE is ``loading`` times it; the treaty's
    ///     retention and limit apply to loss and ALAE (Clark's second
    ///     formula). ``in_severity``: the curves are of loss and ALAE.
    /// loading : float, optional
    ///     Required with ``pro_rata`` and ``on_top``, refused otherwise.
    #[pyo3(signature = (limit, attachment, *, alae, loading = None))]
    fn exposure_factors(
        &self,
        limit: f64,
        attachment: f64,
        alae: &str,
        loading: Option<f64>,
    ) -> PyResult<Vec<f64>> {
        self.inner
            .exposure_factors(&treaty(limit, attachment)?, self::alae(alae, loading)?)
            .map_err(to_py)
    }

    /// The ILF form: each policy's ``premium * elr`` times its exposure
    /// factor, with the ALAE loading. Arguments as ``exposure_factors``.
    ///
    /// Returns
    /// -------
    /// CasualtyRating
    #[pyo3(signature = (limit, attachment, *, alae, loading = None))]
    fn rate(
        &self,
        limit: f64,
        attachment: f64,
        alae: &str,
        loading: Option<f64>,
    ) -> PyResult<PyCasualtyRating> {
        let inner = self
            .inner
            .rate(&treaty(limit, attachment)?, self::alae(alae, loading)?)
            .map_err(to_py)?;
        Ok(PyCasualtyRating { inner })
    }

    /// The frequency form: each policy's expected number of ground-up
    /// occurrences above ``threshold`` (``counts``) times the expected ceded
    /// amount given ``X > threshold`` under ``excess``, a severity of the
    /// losses above it (a Pareto from it, say). Fails when an exposed
    /// policy's cession starts below the threshold.
    ///
    /// Returns
    /// -------
    /// CasualtyRating
    #[pyo3(signature = (limit, attachment, *, alae, threshold, counts, excess, loading = None))]
    #[allow(clippy::too_many_arguments)]
    fn rate_by_frequency(
        &self,
        limit: f64,
        attachment: f64,
        alae: &str,
        threshold: f64,
        counts: Vec<f64>,
        excess: &Bound<'_, PyAny>,
        loading: Option<f64>,
    ) -> PyResult<PyCasualtyRating> {
        let excess = extract_severity(excess)?;
        let inner = self
            .inner
            .rate_by_frequency(
                &treaty(limit, attachment)?,
                self::alae(alae, loading)?,
                threshold,
                &counts,
                &excess,
            )
            .map_err(to_py)?;
        Ok(PyCasualtyRating { inner })
    }

    /// Each policy's expected ground-up occurrences above ``t`` a year,
    /// from its own curve: ``premium * elr / (share * [LEV(a + l) -
    /// LEV(a)]) * S(t)``, a table's ``S`` its slope just above ``t``.
    ///
    /// Returns
    /// -------
    /// list of float
    fn occurrences_above(&self, t: f64) -> PyResult<Vec<f64>> {
        self.inner.occurrences_above(t).map_err(to_py)
    }

    /// The frequency form at each alpha of a Pareto from ``threshold``,
    /// the counts held fixed. The central alpha and its source are yours to
    /// state beside the result; prospicio ships no benchmark.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Columns ``alpha``, ``ceded`` and ``loss_cost``.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Pareto
    /// >>> from prospicio.pricing import CasualtyProfile
    /// >>> book = CasualtyProfile([Pareto(1e6, 1.5)], limit=[10e6], premium=[1e6], elr=[0.6])
    /// >>> s = book.sweep(5e6, 5e6, alae="in_severity", threshold=1e6, counts=[1.0],
    /// ...                alphas=[1.0, 2.0])
    /// >>> round(s["ceded"][0] / s["ceded"][1], 2)
    /// 6.93
    #[pyo3(signature = (limit, attachment, *, alae, threshold, counts, alphas, loading = None))]
    #[allow(clippy::too_many_arguments)]
    fn sweep(
        &self,
        limit: f64,
        attachment: f64,
        alae: &str,
        threshold: f64,
        counts: Vec<f64>,
        alphas: Vec<f64>,
        loading: Option<f64>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let points = self
            .inner
            .sweep(
                &treaty(limit, attachment)?,
                self::alae(alae, loading)?,
                threshold,
                &counts,
                &alphas,
            )
            .map_err(to_py)?;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("alpha", points.iter().map(|p| p.alpha).collect::<Vec<_>>())?;
        d.set_item("ceded", points.iter().map(|p| p.ceded).collect::<Vec<_>>())?;
        d.set_item(
            "loss_cost",
            points.iter().map(|p| p.loss_cost).collect::<Vec<_>>(),
        )?;
        Ok(d.into_any().unbind())
    }

    /// The expected number of claims a year recorded at or above the
    /// ground-up size ``t``: a policy records an occurrence once it passes
    /// its attachment and caps it at ``a + l``, so it adds ``counts[i] *
    /// S(max(t, a))`` when ``t <= a + l``, ``S`` the survival of
    /// ``excess``.
    fn expected_claims(
        &self,
        t: f64,
        threshold: f64,
        counts: Vec<f64>,
        excess: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let excess = extract_severity(excess)?;
        self.inner
            .expected_claims(t, threshold, &counts, &excess)
            .map_err(to_py)
    }

    /// Expected against observed claims at each size, with the Poisson
    /// probability of a count at least as extreme in the direction
    /// observed.
    ///
    /// Parameters
    /// ----------
    /// sizes : list of float
    ///     Ground-up sizes, at or above ``threshold``.
    /// threshold, counts, excess
    ///     As in ``rate_by_frequency``.
    /// exposure_years : float
    ///     The experience period in the profile's years: each year's
    ///     on-level premium over the profile's, summed.
    /// observed : list of float
    ///     The trended ground-up losses (``PreparedLosses.trended``), a claim
    ///     closed at its limit at the limit.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Columns ``size``, ``expected``, ``observed`` and ``p_value``.
    #[pyo3(signature = (sizes, *, threshold, counts, excess, exposure_years, observed))]
    #[allow(clippy::too_many_arguments)]
    fn count_test(
        &self,
        py: Python<'_>,
        sizes: Vec<f64>,
        threshold: f64,
        counts: Vec<f64>,
        excess: &Bound<'_, PyAny>,
        exposure_years: f64,
        observed: Vec<f64>,
    ) -> PyResult<Py<PyAny>> {
        let excess = extract_severity(excess)?;
        let checks = self
            .inner
            .count_test(
                &sizes,
                threshold,
                &counts,
                &excess,
                exposure_years,
                &observed,
            )
            .map_err(to_py)?;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("size", checks.iter().map(|c| c.size).collect::<Vec<_>>())?;
        d.set_item(
            "expected",
            checks.iter().map(|c| c.expected).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "observed",
            checks.iter().map(|c| c.observed).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "p_value",
            checks.iter().map(|c| c.p_value).collect::<Vec<_>>(),
        )?;
        Ok(d.into_any().unbind())
    }
}

/// The treaty's expected loss, kept policy by policy, from
/// ``CasualtyProfile.rate`` or ``rate_by_frequency``.
#[pyclass(name = "CasualtyRating", module = "prospicio.pricing", frozen)]
pub(crate) struct PyCasualtyRating {
    inner: CasualtyRating,
}

#[pymethods]
impl PyCasualtyRating {
    /// The treaty's expected loss, with ALAE.
    #[getter]
    fn ceded(&self) -> f64 {
        self.inner.ceded()
    }

    /// Subject premium.
    #[getter]
    fn premium(&self) -> f64 {
        self.inner.premium()
    }

    /// Expected treaty loss over subject premium.
    #[getter]
    fn loss_cost(&self) -> f64 {
        self.inner.loss_cost()
    }

    /// One row per policy: columns ``factor`` and ``expected_loss`` (NaN
    /// in the frequency form), ``ceded``, ``premium`` and ``tag``.
    fn rows(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let r = self.inner.rows();
        let d = pyo3::types::PyDict::new(py);
        d.set_item("factor", r.iter().map(|x| x.factor).collect::<Vec<_>>())?;
        d.set_item(
            "expected_loss",
            r.iter().map(|x| x.expected_loss).collect::<Vec<_>>(),
        )?;
        d.set_item("ceded", r.iter().map(|x| x.ceded).collect::<Vec<_>>())?;
        d.set_item("premium", r.iter().map(|x| x.premium).collect::<Vec<_>>())?;
        d.set_item("tag", r.iter().map(|x| x.tag.clone()).collect::<Vec<_>>())?;
        Ok(d.into_any().unbind())
    }

    /// The rating with the policies tagged ``tag`` left out.
    fn without(&self, tag: &str) -> Self {
        Self {
            inner: self.inner.without(tag),
        }
    }

    /// Ceded loss and premium by tag, in first-seen order: columns ``tag``
    /// (``None`` for untagged), ``ceded`` and ``premium``.
    fn by_tag(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let g = self.inner.by_tag();
        let d = pyo3::types::PyDict::new(py);
        d.set_item("tag", g.iter().map(|x| x.0.clone()).collect::<Vec<_>>())?;
        d.set_item("ceded", g.iter().map(|x| x.1).collect::<Vec<_>>())?;
        d.set_item("premium", g.iter().map(|x| x.2).collect::<Vec<_>>())?;
        Ok(d.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "CasualtyRating(policies={}, ceded={}, loss_cost={})",
            self.inner.rows().len(),
            self.inner.ceded(),
            self.inner.loss_cost()
        )
    }
}

/// The Pareto above a threshold fitted to trended losses, with claims
/// closed at their policy limit censored: ``alpha = uncensored count /
/// sum ln(y / t)``, its standard error ``alpha / sqrt(n)``, and the number
/// of exceedances it rests on.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import ParetoAnchor
/// >>> a = ParetoAnchor.fit(1e6, [1.5e6, 2.5e6, 4e6, 2e6], censored=[False, False, False, True])
/// >>> (a.exceedances, a.censored)
/// (4, 1)
#[pyclass(name = "ParetoAnchor", module = "prospicio.pricing", frozen)]
pub(crate) struct PyParetoAnchor {
    inner: ParetoAnchor,
}

#[pymethods]
impl PyParetoAnchor {
    /// Fits the Pareto from ``threshold`` to the ``losses`` at or above it.
    ///
    /// Parameters
    /// ----------
    /// threshold : float
    /// losses : list of float
    /// censored : list of bool, optional
    ///     Where a loss was closed at its policy limit; none by default.
    #[staticmethod]
    #[pyo3(signature = (threshold, losses, censored = None))]
    fn fit(threshold: f64, losses: Vec<f64>, censored: Option<Vec<bool>>) -> PyResult<Self> {
        let censored = censored.unwrap_or_else(|| vec![false; losses.len()]);
        Ok(Self {
            inner: ParetoAnchor::fit(threshold, &losses, &censored).map_err(to_py)?,
        })
    }

    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[getter]
    fn standard_error(&self) -> f64 {
        self.inner.standard_error
    }

    /// Losses at or above the threshold.
    #[getter]
    fn exceedances(&self) -> usize {
        self.inner.exceedances
    }

    /// Of which censored at a policy limit.
    #[getter]
    fn censored(&self) -> usize {
        self.inner.censored
    }

    /// The alpha ``± z`` standard errors, the lower end kept above 0.
    fn interval(&self, z: f64) -> (f64, f64) {
        self.inner.interval(z)
    }

    fn __repr__(&self) -> String {
        format!(
            "ParetoAnchor(threshold={}, alpha={}, standard_error={}, exceedances={}, censored={})",
            self.inner.threshold,
            self.inner.alpha,
            self.inner.standard_error,
            self.inner.exceedances,
            self.inner.censored
        )
    }
}

/// The Poisson probability, at mean ``expected``, of a count at least as
/// far from it as ``observed`` on the side observed.
///
/// Examples
/// --------
/// >>> import math
/// >>> from prospicio.pricing import poisson_tail
/// >>> math.isclose(poisson_tail(3.0, 0), math.exp(-3.0))
/// True
#[pyfunction]
pub(crate) fn poisson_tail(expected: f64, observed: usize) -> PyResult<f64> {
    prospicio_pricing::casualty::poisson_tail(expected, observed).map_err(to_py)
}
