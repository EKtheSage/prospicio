//! Python bindings for `prospicio_pricing::retro`: the insurance charge
//! and savings (Table M, Table L) and a retrospective rating plan.

use prospicio_pricing::retro::{InsuranceCharge, RetroPlan};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::distributions::PyGrid;
use crate::reinsurance::PyLossSensitivePremium;
use crate::to_py;

/// The insurance charge and savings of an aggregate loss distribution, at
/// any entry ratio: a Table M without the rows.
///
/// For aggregate losses ``A`` with mean ``E`` and an entry ratio ``r``,
/// the charge is ``phi(r) = E[max(A / E - r, 0)]`` and the savings
/// ``psi(r) = E[max(r - A / E, 0)]``, with ``phi(r) - psi(r) = 1 - r``.
///
/// Build one with ``InsuranceCharge.from_grid`` (an aggregate on a grid,
/// from ``fft`` or ``panjer``) or ``InsuranceCharge.from_sample``
/// (simulated or observed totals). With a per-occurrence limit, build it
/// from the limited aggregate (the limited Table M) and call ``table_l``
/// for a Table L.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import InsuranceCharge
/// >>> table = InsuranceCharge.from_sample([0.0, 50.0, 100.0, 250.0])
/// >>> table.expected
/// 100.0
/// >>> table.charge(1.0), table.savings(1.0)
/// (0.375, 0.375)
#[pyclass(name = "InsuranceCharge", module = "prospicio.pricing", frozen)]
pub(crate) struct PyInsuranceCharge {
    inner: InsuranceCharge,
}

#[pymethods]
impl PyInsuranceCharge {
    /// The table of an aggregate loss distribution on a grid.
    ///
    /// Parameters
    /// ----------
    /// aggregate : Grid
    ///     The aggregate losses, as ``fft`` or ``panjer`` return them.
    ///
    /// Returns
    /// -------
    /// InsuranceCharge
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the grid's mean is not positive.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.distributions import Grid
    /// >>> from prospicio.pricing import InsuranceCharge
    /// >>> agg = Grid(50.0, [0.25, 0.25, 0.25, 0.0, 0.0, 0.25])
    /// >>> InsuranceCharge.from_grid(agg).charge(0.5)
    /// 0.625
    #[staticmethod]
    fn from_grid(aggregate: PyRef<'_, PyGrid>) -> PyResult<Self> {
        Ok(Self {
            inner: InsuranceCharge::from_severity(aggregate.inner.clone()).map_err(to_py)?,
        })
    }

    /// The table of a sample of aggregate losses, each equally likely:
    /// simulated years, or the experience of a group of similar risks.
    ///
    /// Parameters
    /// ----------
    /// losses : list of float
    ///     The totals, in any order.
    ///
    /// Returns
    /// -------
    /// InsuranceCharge
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the sample is empty, holds a negative or non-finite loss, or
    ///     has mean zero.
    #[staticmethod]
    fn from_sample(losses: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: InsuranceCharge::from_sample(&losses).map_err(to_py)?,
        })
    }

    /// This table of a limited aggregate as a Table L: entry ratios become
    /// multiples of the unlimited expected loss, and the charge takes on
    /// the share of losses the per-occurrence limit removes.
    ///
    /// Parameters
    /// ----------
    /// unlimited_expected : float
    ///     The expected aggregate loss without the limit.
    ///
    /// Returns
    /// -------
    /// InsuranceCharge
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``unlimited_expected`` is below the limited mean or not
    ///     finite, or the table is already a Table L.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.pricing import InsuranceCharge
    /// >>> limited = InsuranceCharge.from_sample([110.0, 100.0, 0.0, 200.0])
    /// >>> table = limited.table_l(132.5)
    /// >>> round(table.loss_elimination, 6)
    /// 0.226415
    /// >>> table.charge(0.0)
    /// 1.0
    fn table_l(&self, unlimited_expected: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .table_l(unlimited_expected)
                .map_err(to_py)?,
        })
    }

    /// What the entry ratios multiply: the expected aggregate loss, and
    /// for a Table L the expected loss without the limit.
    #[getter]
    fn expected(&self) -> f64 {
        self.inner.expected()
    }

    /// A Table L's share of expected losses removed by the limit; 0 for
    /// any other table.
    #[getter]
    fn loss_elimination(&self) -> f64 {
        self.inner.loss_elimination()
    }

    /// The insurance charge ``phi(r)``.
    ///
    /// Parameters
    /// ----------
    /// entry_ratio : float
    ///
    /// Returns
    /// -------
    /// float
    fn charge(&self, entry_ratio: f64) -> f64 {
        self.inner.charge(entry_ratio)
    }

    /// The insurance savings ``psi(r) = phi(r) + r - 1``.
    ///
    /// Parameters
    /// ----------
    /// entry_ratio : float
    ///
    /// Returns
    /// -------
    /// float
    fn savings(&self, entry_ratio: f64) -> f64 {
        self.inner.savings(entry_ratio)
    }

    /// The table at the given entry ratios.
    ///
    /// Parameters
    /// ----------
    /// entry_ratios : list of float
    ///
    /// Returns
    /// -------
    /// dict
    ///     Columns ``entry_ratio``, ``charge`` and ``savings``, one value
    ///     per entry ratio, in their order.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.pricing import InsuranceCharge
    /// >>> t = InsuranceCharge.from_sample([0.0, 50.0, 100.0, 250.0]).table([0.0, 0.5, 1.0])
    /// >>> t["charge"], t["savings"]
    /// ([1.0, 0.625, 0.375], [0.0, 0.125, 0.375])
    fn table<'py>(&self, py: Python<'py>, entry_ratios: Vec<f64>) -> PyResult<Bound<'py, PyDict>> {
        let rows = self.inner.table(&entry_ratios);
        let d = PyDict::new(py);
        d.set_item("entry_ratio", entry_ratios)?;
        d.set_item("charge", rows.iter().map(|r| r.charge).collect::<Vec<_>>())?;
        d.set_item(
            "savings",
            rows.iter().map(|r| r.savings).collect::<Vec<_>>(),
        )?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "InsuranceCharge(expected={}, loss_elimination={})",
            self.inner.expected(),
            self.inner.loss_elimination()
        )
    }
}

/// A retrospective rating plan: the premium ``T (b + c L)`` for losses
/// ``L``, no less than a minimum and no more than a maximum.
///
/// Amounts are in money. A plan with a loss limit takes ``L`` as the
/// losses with each occurrence limited, and is balanced against a Table L.
///
/// Parameters
/// ----------
/// expected_loss : float
///     The expected losses ``E``, without any loss limit.
/// expense : float
///     Expenses and profit ``e``, excluding the premium tax.
/// loss_conversion : float, default 1.0
///     The loss conversion factor ``c``.
/// tax_multiplier : float, default 1.0
///     The tax multiplier ``T``.
/// minimum : float, default 0.0
///     The minimum premium ``H``.
/// maximum : float, default inf
///     The maximum premium ``G``.
///
/// Raises
/// ------
/// ValueError
///     Unless ``E``, ``c`` and ``T`` are positive and finite, ``e`` is
///     non-negative and ``0 <= H < G``.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import InsuranceCharge, RetroPlan
/// >>> losses = [0.0, 10.0, 30.0, 50.0, 70.0, 90.0, 120.0, 150.0, 200.0, 280.0]
/// >>> plan = RetroPlan(100.0, 30.0, loss_conversion=1.1, tax_multiplier=1.03,
/// ...                  minimum=80.0, maximum=220.0)
/// >>> balanced = plan.balance(InsuranceCharge.from_sample(losses))
/// >>> terms = plan.loss_sensitive_premium(balanced["basic"])
/// >>> mean = sum(terms.premium(x) for x in losses) / 10
/// >>> round(mean, 9) == round(plan.guaranteed_cost(), 9)
/// True
#[pyclass(name = "RetroPlan", module = "prospicio.pricing", frozen)]
pub(crate) struct PyRetroPlan {
    inner: RetroPlan,
}

#[pymethods]
impl PyRetroPlan {
    #[new]
    #[pyo3(signature = (expected_loss, expense, loss_conversion = 1.0, tax_multiplier = 1.0, minimum = 0.0, maximum = f64::INFINITY))]
    fn new(
        expected_loss: f64,
        expense: f64,
        loss_conversion: f64,
        tax_multiplier: f64,
        minimum: f64,
        maximum: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RetroPlan::new(
                expected_loss,
                expense,
                loss_conversion,
                tax_multiplier,
                minimum,
                maximum,
            )
            .map_err(to_py)?,
        })
    }

    /// The guaranteed-cost premium ``T (e + E)``.
    ///
    /// Returns
    /// -------
    /// float
    fn guaranteed_cost(&self) -> f64 {
        self.inner.guaranteed_cost()
    }

    /// The plan's premium terms under a basic premium, as a
    /// ``prospicio.reinsurance.LossSensitivePremium``: ``T (b + c L)`` for
    /// a year's losses ``L``, between the minimum and the maximum, with
    /// the tax multiplier folded into its basic premium and its loss
    /// conversion factor.
    ///
    /// Parameters
    /// ----------
    /// basic : float
    ///     The basic premium ``b``, before the tax multiplier.
    ///
    /// Returns
    /// -------
    /// prospicio.reinsurance.LossSensitivePremium
    ///     Its ``premium(losses)`` is the retrospective premium.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``basic`` is negative or not finite.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.pricing import RetroPlan
    /// >>> plan = RetroPlan(100.0, 30.0, 1.1, 1.03, minimum=80.0, maximum=220.0)
    /// >>> terms = plan.loss_sensitive_premium(25.0)
    /// >>> terms.premium(0.0), terms.premium(400.0)
    /// (80.0, 220.0)
    fn loss_sensitive_premium(&self, basic: f64) -> PyResult<PyLossSensitivePremium> {
        Ok(PyLossSensitivePremium {
            inner: self.inner.loss_sensitive_premium(basic).map_err(to_py)?,
        })
    }

    /// The basic premium at which the plan collects the guaranteed-cost
    /// premium on average.
    ///
    /// Parameters
    /// ----------
    /// charge : InsuranceCharge
    ///     The table of the plan's aggregate losses, on the plan's expected
    ///     losses: a Table M, or with a loss limit a Table L.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``basic``, ``entry_ratio_minimum`` and ``entry_ratio_maximum``
    ///     (where the minimum and the maximum bind; the first is negative
    ///     when the minimum never binds), ``net_insurance_charge`` and
    ///     ``expected_premium``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the table's expected loss is not the plan's, or the minimum
    ///     is at least the guaranteed-cost premium, or the maximum at most
    ///     it, or the minimum is high enough that only a negative basic
    ///     premium would balance the plan.
    fn balance<'py>(
        &self,
        py: Python<'py>,
        charge: PyRef<'_, PyInsuranceCharge>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let b = self.inner.balance(&charge.inner).map_err(to_py)?;
        let d = PyDict::new(py);
        d.set_item("basic", b.basic)?;
        d.set_item("entry_ratio_minimum", b.entry_ratio_minimum)?;
        d.set_item("entry_ratio_maximum", b.entry_ratio_maximum)?;
        d.set_item("net_insurance_charge", b.net_insurance_charge)?;
        d.set_item("expected_premium", b.expected_premium)?;
        Ok(d)
    }
}
