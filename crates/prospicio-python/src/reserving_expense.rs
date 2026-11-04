//! `prospicio.reserving` recoveries, ALAE and ULAE (issue #558,
//! `docs/design/reserving-v02.md`, decision 37), over
//! `prospicio_reserving::expense`.

use prospicio_core::{Lag, Period};
use prospicio_reserving::{
    Apriori, ChainLadder, RatioDevelopment, RatioDevelopmentFit, RatioFactors, Ulae, UlaeEstimate,
    UlaeInputs, UlaeWeights,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::reserving::{PyChainLadder, PyTriangle, age_values, apriori_by_origin, err, py_repr};

/// The ratio of two columns of one triangle, developed to ultimate and
/// applied to ultimate claims: Friedland's ratio approach to salvage and
/// subrogation (chapter 14) and to ALAE (chapter 16).
///
/// Each cell's ratio comes from the two cumulative columns; a cell whose
/// denominator is zero or missing has none. The latest ratio is developed to
/// ultimate, a selected ratio replaces it where the actuary judges otherwise,
/// and the numerator's ultimate is the ratio times the ultimate claims from
/// any method.
///
/// Parameters
/// ----------
/// development : ChainLadder, optional
///     How the ratio's factors are estimated, with its tail and any selected
///     ``ldf``. By default the simple average of every link ratio with no
///     tail, as Friedland averages ratios. Additive factors need
///     ``average="simple"`` and no tail or ``ldf``.
/// selected : dict, optional
///     An ultimate ratio by origin, as a label or integer year, which
///     replaces the projected one: ``{2008: 0.345}``.
/// additive : bool, default False
///     Average the differences between ages and add them to ultimate,
///     Friedland's chapter 16, Exhibit I, Sheets 9 and 10, instead of
///     multiplying age-to-age factors.
/// additive_selected : dict, optional
///     Selected differences by the age they start from, for ``additive``.
/// additive_tail : float, default 0.0
///     The addition past the oldest age, for ``additive``.
#[pyclass(name = "RatioDevelopment", module = "prospicio.reserving", frozen)]
pub(crate) struct PyRatioDevelopment {
    development: Option<Py<PyChainLadder>>,
    selected: Vec<(Period, f64)>,
    factors: RatioFactors,
}

#[pymethods]
impl PyRatioDevelopment {
    #[new]
    #[pyo3(signature = (
        development = None,
        *,
        selected = None,
        additive = false,
        additive_selected = None,
        additive_tail = 0.0,
    ))]
    fn new(
        development: Option<Py<PyChainLadder>>,
        selected: Option<&Bound<'_, PyDict>>,
        additive: bool,
        additive_selected: Option<&Bound<'_, PyAny>>,
        additive_tail: f64,
    ) -> PyResult<Self> {
        if !additive && (additive_selected.is_some() || additive_tail != 0.0) {
            return Err(PyValueError::new_err(
                "additive_selected and additive_tail need additive=True",
            ));
        }
        let selected = match selected {
            None => Vec::new(),
            Some(dict) => match apriori_by_origin(dict.iter().map(Ok))? {
                Apriori::ByOrigin(pairs) => pairs,
                _ => unreachable!("labels give values by origin"),
            },
        };
        let factors = if additive {
            RatioFactors::Additive {
                selected: match additive_selected {
                    None => Vec::new(),
                    Some(obj) => age_values(obj, "additive_selected")?,
                },
                tail: additive_tail,
            }
        } else {
            RatioFactors::Multiplicative
        };
        Ok(Self {
            development,
            selected,
            factors,
        })
    }

    /// Fits the ratio of ``numerator`` to ``denominator``, two columns of a
    /// single-segment triangle, and applies it to ``ultimate_claims``, one
    /// value per origin, oldest first.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     The triangle holding both columns.
    /// numerator : str
    ///     The cumulative column to project: received recoveries, paid ALAE.
    /// denominator : str
    ///     The cumulative column it is a ratio to: paid claims.
    /// ultimate_claims : sequence of float
    ///     The ultimate claims, from any method.
    ///
    /// Returns
    /// -------
    /// RatioDevelopmentFit
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        numerator: &str,
        denominator: &str,
        ultimate_claims: Vec<f64>,
    ) -> PyResult<PyRatioDevelopmentFit> {
        let tri = &triangle.inner;
        let defaults = RatioDevelopment::default();
        let (development, tail) = match &self.development {
            None => (defaults.development, defaults.tail),
            Some(method) => {
                let ChainLadder { development, tail } =
                    method.borrow(py).resolved(tri.origin_grain())?;
                (development, tail)
            }
        };
        let inner = RatioDevelopment {
            development,
            tail,
            factors: self.factors.clone(),
            selected: self.selected.clone(),
        }
        .fit(tri, numerator, denominator, &ultimate_claims)
        .map_err(err)?;
        Ok(PyRatioDevelopmentFit { inner })
    }

    fn __repr__(&self) -> String {
        let additive = matches!(self.factors, RatioFactors::Additive { .. });
        format!(
            "RatioDevelopment(additive={}, selected={})",
            if additive { "True" } else { "False" },
            self.selected.len()
        )
    }
}

/// A fitted ``RatioDevelopment``, per origin.
#[pyclass(name = "RatioDevelopmentFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyRatioDevelopmentFit {
    inner: RatioDevelopmentFit,
}

#[pymethods]
impl PyRatioDevelopmentFit {
    /// Origin of each per-origin value.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(Period::to_string).collect()
    }
    /// Development ages in months.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner.ages.clone()
    }
    /// The ratio triangle, one row per origin; ``None`` where a cell has no
    /// ratio.
    #[getter]
    fn ratios(&self) -> Vec<Vec<Option<f64>>> {
        self.inner.ratios.clone()
    }
    /// The averages of the ratio's age-to-age factors or differences.
    #[getter]
    fn averages(&self) -> Vec<f64> {
        self.inner.averages.clone()
    }
    /// The age-to-age factors or differences in use, after any selection.
    #[getter]
    fn factors(&self) -> Vec<f64> {
        self.inner.factors.clone()
    }
    /// Age-to-ultimate factors or additions, one per age.
    #[getter]
    fn cdf(&self) -> Vec<f64> {
        self.inner.cdf.clone()
    }
    /// The latest ratio per origin.
    #[getter]
    fn latest_ratio(&self) -> Vec<f64> {
        self.inner.latest_ratio.clone()
    }
    /// The latest ratio developed to ultimate.
    #[getter]
    fn projected_ratio(&self) -> Vec<f64> {
        self.inner.projected_ratio.clone()
    }
    /// The ultimate ratio in use: the projected one, or the selected one.
    #[getter]
    fn selected_ratio(&self) -> Vec<f64> {
        self.inner.selected_ratio.clone()
    }
    /// The ultimate claims the ratio applies to, as given.
    #[getter]
    fn ultimate_claims(&self) -> Vec<f64> {
        self.inner.ultimate_claims.clone()
    }
    /// The latest cumulative numerator per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.inner.latest.clone()
    }
    /// The numerator's ultimate: the selected ratio times the ultimate
    /// claims.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        self.inner.ultimate.clone()
    }
    /// Ultimate less latest; for recoveries, the recoverable.
    #[getter]
    fn unpaid(&self) -> Vec<f64> {
        self.inner.unpaid()
    }
    /// The sum of the ultimates.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }
    /// The sum of the unpaid amounts.
    #[getter]
    fn total_unpaid(&self) -> f64 {
        self.inner.total_unpaid()
    }
    fn __repr__(&self) -> String {
        format!(
            "RatioDevelopmentFit(total_ultimate={:?}, total_unpaid={:?})",
            self.inner.total_ultimate(),
            self.inner.total_unpaid()
        )
    }
}

/// An estimate of unpaid ULAE, from ``ulae_classical``, ``ulae_kittel`` or
/// ``ulae_conger_nolibos``.
#[pyclass(name = "UlaeEstimate", module = "prospicio.reserving", frozen)]
pub(crate) struct PyUlaeEstimate {
    inner: UlaeEstimate,
}

#[pymethods]
impl PyUlaeEstimate {
    /// The claims basis by calendar year.
    #[getter]
    fn basis(&self) -> Vec<f64> {
        self.inner.basis.clone()
    }
    /// Paid ULAE over the basis by calendar year; NaN at a zero basis.
    #[getter]
    fn ratios(&self) -> Vec<f64> {
        self.inner.ratios.clone()
    }
    /// Total paid ULAE over the total basis.
    #[getter]
    fn total_ratio(&self) -> f64 {
        self.inner.total_ratio
    }
    /// The ratio applied: the selected one, or ``total_ratio``.
    #[getter]
    fn ratio(&self) -> f64 {
        self.inner.ratio
    }
    /// Unpaid ULAE by row of ``case_outstanding`` and ``ibnr``.
    #[getter]
    fn unpaid(&self) -> Vec<f64> {
        self.inner.unpaid.clone()
    }
    /// Total unpaid ULAE.
    #[getter]
    fn total(&self) -> f64 {
        self.inner.total()
    }
    /// Conger and Nolibos' expected claims form, ``W* L - M``; ``None``
    /// without ``ultimate`` or for the other techniques.
    #[getter]
    fn expected_claims(&self) -> Option<f64> {
        self.inner.expected_claims
    }
    /// Conger and Nolibos' preferred form, ``W* (L - B)``; ``None`` without
    /// ``ultimate`` or for the other techniques. Under their simplification
    /// it loses the pure IBNR, and ``unpaid`` is the estimate.
    #[getter]
    fn bornhuetter_ferguson(&self) -> Option<f64> {
        self.inner.bornhuetter_ferguson
    }
    /// Conger and Nolibos' development form, ``M (L / B - 1)``; ``None``
    /// without ``ultimate`` or for the other techniques.
    #[getter]
    fn development(&self) -> Option<f64> {
        self.inner.development
    }
    fn __repr__(&self) -> String {
        format!(
            "UlaeEstimate(ratio={:?}, total={:?})",
            self.inner.ratio,
            self.inner.total()
        )
    }
}

/// Unpaid ULAE by the classical, or paid-to-paid, technique (Friedland,
/// chapter 17): calendar paid ULAE over calendar paid claims, applied to
/// ``1 - opening`` of case outstanding and all of the IBNR.
///
/// Mango and Allen's refinement is this technique with expected paid claims
/// as ``paid``.
///
/// Parameters
/// ----------
/// paid_ulae, paid : sequence of float
///     Paid ULAE and paid claims by calendar year. Claims include ALAE.
/// case_outstanding, ibnr : sequence of float
///     Case outstanding and total IBNR at the valuation date, one value for
///     the book or one per accident year.
/// ratio : float, optional
///     The selected ULAE ratio; by default the ratio of the totals.
/// opening : float, default 0.5
///     The share of a claim's ULAE sustained when it opens. An insurer
///     spending 25% opening and 75% closing applies 75% of the ratio to case
///     outstanding and 100% to IBNR (Friedland's erratum to page 388).
/// pure_ibnr : sequence of float, optional
///     The part of ``ibnr`` on claims not yet reported, which alone takes
///     the full ratio; the rest of the IBNR is treated as known claims.
///
/// Returns
/// -------
/// UlaeEstimate
#[pyfunction]
#[pyo3(signature = (paid_ulae, paid, case_outstanding, ibnr, *, ratio = None, opening = 0.5, pure_ibnr = None))]
pub(crate) fn ulae_classical(
    paid_ulae: Vec<f64>,
    paid: Vec<f64>,
    case_outstanding: Vec<f64>,
    ibnr: Vec<f64>,
    ratio: Option<f64>,
    opening: f64,
    pure_ibnr: Option<Vec<f64>>,
) -> PyResult<PyUlaeEstimate> {
    let inner = Ulae::Classical { ratio, opening }
        .estimate(&UlaeInputs {
            paid_ulae: &paid_ulae,
            paid: &paid,
            case_outstanding: &case_outstanding,
            ibnr: &ibnr,
            pure_ibnr: pure_ibnr.as_deref(),
            ..Default::default()
        })
        .map_err(err)?;
    Ok(PyUlaeEstimate { inner })
}

/// Unpaid ULAE by Kittel's refinement of the classical technique (Friedland,
/// chapter 17): calendar paid ULAE over the average of calendar paid and
/// calendar incurred claims, applied to half of case outstanding and all of
/// the IBNR.
///
/// Parameters
/// ----------
/// paid_ulae, paid, incurred : sequence of float
///     Paid ULAE, paid claims and incurred claims by calendar year. Incurred
///     claims are paid claims plus the change in case outstanding and IBNR.
/// case_outstanding, ibnr : sequence of float
///     Case outstanding and total IBNR at the valuation date, one value for
///     the book or one per accident year.
/// ratio : float, optional
///     The selected ULAE ratio; by default the ratio of the totals.
/// pure_ibnr : sequence of float, optional
///     The part of ``ibnr`` on claims not yet reported, which alone takes
///     the full ratio.
///
/// Returns
/// -------
/// UlaeEstimate
#[pyfunction]
#[pyo3(signature = (paid_ulae, paid, incurred, case_outstanding, ibnr, *, ratio = None, pure_ibnr = None))]
pub(crate) fn ulae_kittel(
    paid_ulae: Vec<f64>,
    paid: Vec<f64>,
    incurred: Vec<f64>,
    case_outstanding: Vec<f64>,
    ibnr: Vec<f64>,
    ratio: Option<f64>,
    pure_ibnr: Option<Vec<f64>>,
) -> PyResult<PyUlaeEstimate> {
    let inner = Ulae::Kittel { ratio }
        .estimate(&UlaeInputs {
            paid_ulae: &paid_ulae,
            paid: &paid,
            reported: &incurred,
            case_outstanding: &case_outstanding,
            ibnr: &ibnr,
            pure_ibnr: pure_ibnr.as_deref(),
            ..Default::default()
        })
        .map_err(err)?;
    Ok(PyUlaeEstimate { inner })
}

/// Unpaid ULAE by Conger and Nolibos' generalized approach (Friedland,
/// chapter 17): calendar paid ULAE over the claims basis ``B = U1 R + U2 P +
/// U3 C``, and ``W* (U1 (L - R) + U2 (L - P) + U3 (L - C))`` unpaid.
///
/// Their simplification passes the accident years' ultimate claims as
/// ``reported`` and ``U3 = 0``.
///
/// Parameters
/// ----------
/// paid_ulae, paid, reported : sequence of float
///     Paid ULAE ``M``, paid claims ``P`` and the ultimate cost of claims
///     reported ``R`` by calendar year.
/// case_outstanding, ibnr : sequence of float
///     Case outstanding and total IBNR at the valuation date, one value for
///     the book or one per accident year. Their sum is ``L - P(t)``.
/// weights : sequence of float
///     ``(U1, U2)`` or ``(U1, U2, U3)``, the shares spent opening,
///     maintaining and closing claims, summing to 1.
/// ratio : float, optional
///     The selected ULAE ratio ``W*``; by default the ratio of the totals.
/// pure_ibnr : sequence of float, optional
///     Claims not yet reported, ``L - R(t)``; needed unless ``U1`` is 0.
/// closed : sequence of float, optional
///     The ultimate cost of claims closed ``C`` by calendar year; needed
///     unless ``U3`` is 0.
/// unclosed : sequence of float, optional
///     The ultimate cost of claims not yet closed, ``L - C(t)``, one value
///     per row of ``case_outstanding``; needed unless ``U3`` is 0.
/// ultimate : sequence of float, optional
///     Ultimate claims ``L`` by accident year, over the same claims as the
///     calendar history, for ``expected_claims``,
///     ``bornhuetter_ferguson`` and ``development``.
///
/// Returns
/// -------
/// UlaeEstimate
#[pyfunction]
#[pyo3(signature = (
    paid_ulae,
    paid,
    reported,
    case_outstanding,
    ibnr,
    *,
    weights,
    ratio = None,
    pure_ibnr = None,
    closed = None,
    unclosed = None,
    ultimate = None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ulae_conger_nolibos(
    paid_ulae: Vec<f64>,
    paid: Vec<f64>,
    reported: Vec<f64>,
    case_outstanding: Vec<f64>,
    ibnr: Vec<f64>,
    weights: &Bound<'_, PyAny>,
    ratio: Option<f64>,
    pure_ibnr: Option<Vec<f64>>,
    closed: Option<Vec<f64>>,
    unclosed: Option<Vec<f64>>,
    ultimate: Option<Vec<f64>>,
) -> PyResult<PyUlaeEstimate> {
    let shares: Vec<f64> = weights.extract().map_err(|_| {
        PyTypeError::new_err(format!(
            "weights must be numbers (U1, U2) or (U1, U2, U3), got {}",
            py_repr(weights)
        ))
    })?;
    let weights = match shares[..] {
        [opening, maintaining] => UlaeWeights::new(opening, maintaining),
        [opening, maintaining, closing] => UlaeWeights {
            opening,
            maintaining,
            closing,
        },
        _ => {
            return Err(PyValueError::new_err(format!(
                "weights must be (U1, U2) or (U1, U2, U3), got {} values",
                shares.len()
            )));
        }
    };
    let inner = Ulae::CongerNolibos { weights, ratio }
        .estimate(&UlaeInputs {
            paid_ulae: &paid_ulae,
            paid: &paid,
            reported: &reported,
            closed: closed.as_deref().unwrap_or(&[]),
            case_outstanding: &case_outstanding,
            ibnr: &ibnr,
            pure_ibnr: pure_ibnr.as_deref(),
            unclosed: unclosed.as_deref().unwrap_or(&[]),
            ultimate: ultimate.as_deref().unwrap_or(&[]),
        })
        .map_err(err)?;
    Ok(PyUlaeEstimate { inner })
}
