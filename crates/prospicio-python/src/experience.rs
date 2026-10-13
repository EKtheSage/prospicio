//! `prospicio.pricing` (Aggregate lane): experience rating from individual
//! large losses over `prospicio_pricing::experience`
//! (`docs/design/experience-rating.md`).

use prospicio_aggregate::Tower;
use prospicio_pricing::experience::{
    BurningCost, Experience, LargeLoss, Prepared, Trend, YearBasis,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::reinsurance::PyTower;
use crate::to_py;

/// A record of individual large losses, for experience rating.
///
/// Trend each loss to the prospective period with ``prepare``, then apply
/// each year's treaty and develop the layer totals with
/// ``PreparedLosses.burning_cost``. The steps follow Clark's *Basics of
/// Reinsurance Pricing* (CAS study note).
///
/// Parameters
/// ----------
/// ids : list of str
///     An identifier per loss.
/// accident : list of float
///     Each loss's accident date as a decimal year (see ``year_fraction``).
/// amount : list of float
///     Each loss's amount, incurred at the latest valuation.
/// threshold : float, default 0.0
///     The reporting threshold of the record, in nominal terms. With a
///     threshold, losses trended below it times the largest trend factor
///     are left out, so every year is complete on the same basis.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import LargeLossExperience
/// >>> from prospicio.reinsurance import Layer, Tower
/// >>> e = LargeLossExperience(["a", "b"], [1995.5, 1996.5], [150.0, 400.0])
/// >>> p = e.prepare(factors=[1.0, 1.0])
/// >>> tower = Tower([Layer("200x100", 200.0, 100.0)])
/// >>> b = p.burning_cost(tower, [1995, 1996], premium=[1000.0, 1000.0], development=[1.0, 1.5])
/// >>> b.to_dict()["developed"]
/// [[50.0], [300.0]]
/// >>> round(b.loss_cost[0], 4)
/// 0.175
#[pyclass(name = "LargeLossExperience", module = "prospicio.pricing", frozen)]
pub(crate) struct PyLargeLossExperience {
    inner: Experience,
}

#[pymethods]
impl PyLargeLossExperience {
    #[new]
    #[pyo3(signature = (ids, accident, amount, *, threshold = 0.0))]
    fn new(
        ids: Vec<String>,
        accident: Vec<f64>,
        amount: Vec<f64>,
        threshold: f64,
    ) -> PyResult<Self> {
        if ids.len() != accident.len() || ids.len() != amount.len() {
            return Err(PyValueError::new_err(format!(
                "{} ids, {} accident dates and {} amounts",
                ids.len(),
                accident.len(),
                amount.len()
            )));
        }
        let losses = ids
            .into_iter()
            .zip(accident)
            .zip(amount)
            .map(|((id, accident), amount)| LargeLoss {
                id,
                accident,
                amount,
            })
            .collect();
        let inner = Experience::new(losses)
            .and_then(|e| e.with_threshold(threshold))
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// The reporting threshold.
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold()
    }

    /// Trend every loss to the prospective period.
    ///
    /// Pass either an annual ``rate`` with the date ``to`` (the average
    /// accident date of the prospective period, a decimal year), for factors
    /// ``(1 + rate) ** (to - accident)``, or a factor per loss.
    ///
    /// Parameters
    /// ----------
    /// rate : float, optional
    /// to : float, optional
    /// factors : list of float, optional
    ///
    /// Returns
    /// -------
    /// PreparedLosses
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If neither or both of ``rate`` (with ``to``) and ``factors`` are
    ///     given.
    #[pyo3(signature = (*, rate = None, to = None, factors = None))]
    fn prepare(
        &self,
        rate: Option<f64>,
        to: Option<f64>,
        factors: Option<Vec<f64>>,
    ) -> PyResult<PyPreparedLosses> {
        let trend = match (rate, to, factors) {
            (Some(rate), Some(to), None) => Trend::Rate { rate, to },
            (None, None, Some(f)) => Trend::Factors(f),
            _ => {
                return Err(PyTypeError::new_err(
                    "pass rate and to, or factors, but not both",
                ));
            }
        };
        Ok(PyPreparedLosses {
            inner: self.inner.prepare(&trend).map_err(to_py)?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.losses().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "LargeLossExperience(losses={}, threshold={})",
            self.inner.losses().len(),
            self.inner.threshold()
        )
    }
}

/// Large losses trended to the prospective period, from
/// ``LargeLossExperience.prepare``.
#[pyclass(name = "PreparedLosses", module = "prospicio.pricing", frozen)]
pub(crate) struct PyPreparedLosses {
    inner: Prepared,
}

#[pymethods]
impl PyPreparedLosses {
    /// The trended size above which every year's record is complete: the
    /// threshold times the largest trend factor.
    #[getter]
    fn complete_above(&self) -> f64 {
        self.inner.complete_above()
    }

    /// The as-if loss to each layer by accident year, developed and divided
    /// by the premium.
    ///
    /// Parameters
    /// ----------
    /// towers : Tower or list of Tower
    ///     The treaty in force each year, or one tower for every year (to
    ///     price the prospective terms). Every tower needs the same number of
    ///     layers.
    /// years : list of int
    ///     The accident years to cost; a loss belongs to the year of its
    ///     accident date.
    /// premium : list of float
    ///     Each year's on-level subject premium.
    /// development : list of float
    ///     Each year's excess development factor for the layer losses.
    ///
    /// Returns
    /// -------
    /// BurningCost
    #[pyo3(signature = (towers, years, *, premium, development))]
    fn burning_cost(
        &self,
        towers: &Bound<'_, PyAny>,
        years: Vec<i32>,
        premium: Vec<f64>,
        development: Vec<f64>,
    ) -> PyResult<PyBurningCost> {
        if premium.len() != years.len() || development.len() != years.len() {
            return Err(PyValueError::new_err(format!(
                "{} years, {} premiums and {} development factors",
                years.len(),
                premium.len(),
                development.len()
            )));
        }
        let towers: Vec<Tower> = if let Ok(t) = towers.cast::<PyTower>() {
            vec![t.get().inner.clone(); years.len()]
        } else {
            towers
                .extract::<Vec<PyRef<'_, PyTower>>>()
                .map_err(|_| PyTypeError::new_err("towers must be a Tower or a list of Tower"))?
                .iter()
                .map(|t| t.inner.clone())
                .collect()
        };
        let basis: Vec<YearBasis> = years
            .iter()
            .zip(&premium)
            .zip(&development)
            .map(|((&year, &premium), &development)| YearBasis {
                year,
                premium,
                development,
            })
            .collect();
        Ok(PyBurningCost {
            inner: self.inner.burning_cost(&towers, &basis).map_err(to_py)?,
        })
    }

    /// The trended losses as columns: ``id``, ``year``, ``accident``,
    /// ``amount``, ``factor`` and ``trended``.
    ///
    /// Returns
    /// -------
    /// dict of str to list
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let l = self.inner.losses();
        let d = PyDict::new(py);
        d.set_item("id", l.iter().map(|x| x.id.clone()).collect::<Vec<_>>())?;
        d.set_item("year", l.iter().map(|x| x.year).collect::<Vec<_>>())?;
        d.set_item("accident", l.iter().map(|x| x.accident).collect::<Vec<_>>())?;
        d.set_item("amount", l.iter().map(|x| x.amount).collect::<Vec<_>>())?;
        d.set_item("factor", l.iter().map(|x| x.factor).collect::<Vec<_>>())?;
        d.set_item("trended", l.iter().map(|x| x.trended).collect::<Vec<_>>())?;
        Ok(d)
    }

    fn __len__(&self) -> usize {
        self.inner.losses().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PreparedLosses(losses={}, complete_above={})",
            self.inner.losses().len(),
            self.inner.complete_above()
        )
    }
}

/// The burning cost by accident year and overall, from
/// ``PreparedLosses.burning_cost``.
#[pyclass(name = "BurningCost", module = "prospicio.pricing", frozen)]
pub(crate) struct PyBurningCost {
    inner: BurningCost,
}

#[pymethods]
impl PyBurningCost {
    /// Each layer's developed losses over the premium, summed over the
    /// years: the premium-weighted loss cost.
    #[getter]
    fn loss_cost(&self) -> Vec<f64> {
        self.inner.loss_cost.clone()
    }

    /// Each layer's loss cost averaged over the years with equal weight.
    #[getter]
    fn mean_loss_cost(&self) -> Vec<f64> {
        self.inner.mean_loss_cost.clone()
    }

    /// The accident years.
    #[getter]
    fn years(&self) -> Vec<i32> {
        self.inner.years.iter().map(|y| y.year).collect()
    }

    /// The years as columns: ``year``, ``premium``, ``losses`` (the count
    /// through the tower), ``development``, and per layer ``layer_loss``,
    /// ``developed`` and ``loss_cost`` (a list per year, one value per
    /// layer).
    ///
    /// Returns
    /// -------
    /// dict of str to list
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let y = &self.inner.years;
        let d = PyDict::new(py);
        d.set_item("year", y.iter().map(|r| r.year).collect::<Vec<_>>())?;
        d.set_item("premium", y.iter().map(|r| r.premium).collect::<Vec<_>>())?;
        d.set_item("losses", y.iter().map(|r| r.losses).collect::<Vec<_>>())?;
        d.set_item(
            "development",
            y.iter().map(|r| r.development).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "layer_loss",
            y.iter().map(|r| r.layer_loss.clone()).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "developed",
            y.iter().map(|r| r.developed.clone()).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "loss_cost",
            y.iter().map(|r| r.loss_cost.clone()).collect::<Vec<_>>(),
        )?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "BurningCost(years={}, loss_cost={:?})",
            self.inner.years.len(),
            self.inner.loss_cost
        )
    }
}
