//! `prospicio.pricing` (Aggregate lane): experience rating from individual
//! large losses over `prospicio_pricing::experience`
//! (`docs/design/experience-rating.md`).

use prospicio_aggregate::Tower;
use prospicio_pricing::experience::{
    AsIf, BurningCost, ClaimValuation, Development, DevelopmentPattern, Experience, Fx,
    IndexClause, IndexKind, LargeLoss, Prepared, Trend, YearBasis,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::reinsurance::PyTower;
use crate::to_py;

/// A record of individual large losses, for experience rating.
///
/// Develop each known loss to ultimate (optional) and trend it to the
/// prospective period with ``prepare``, then apply each year's treaty and
/// develop the layer totals with ``PreparedLosses.burning_cost``. The steps
/// follow Clark's *Basics of Reinsurance Pricing* (CAS study note).
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
/// valuation : float, optional
///     The date the amounts were valued at, a decimal year; needed to
///     develop the losses by a ``DevelopmentPattern`` at their ages.
/// fx : float or list of float, optional
///     Converts the losses to the pricing currency before development and
///     trend: a stated rate, or a rate per loss (units of the pricing
///     currency per unit of the loss's). The threshold stays in the
///     reported currency.
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
    #[pyo3(signature = (ids, accident, amount, *, threshold = 0.0, valuation = None, fx = None))]
    fn new(
        ids: Vec<String>,
        accident: Vec<f64>,
        amount: Vec<f64>,
        threshold: f64,
        valuation: Option<f64>,
        fx: Option<&Bound<'_, PyAny>>,
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
        let mut inner = Experience::new(losses)
            .and_then(|e| e.with_threshold(threshold))
            .map_err(to_py)?;
        if let Some(v) = valuation {
            inner = inner.with_valuation(v).map_err(to_py)?;
        }
        if let Some(fx) = fx {
            let fx =
                if let Ok(r) = fx.extract::<f64>() {
                    Fx::Rate(r)
                } else {
                    Fx::PerLoss(fx.extract::<Vec<f64>>().map_err(|_| {
                        PyTypeError::new_err("fx must be a float or a list of floats")
                    })?)
                };
            inner = inner.with_fx(fx).map_err(to_py)?;
        }
        Ok(Self { inner })
    }

    /// The reporting threshold.
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold()
    }

    /// The valuation date, or ``None``.
    #[getter]
    fn valuation(&self) -> Option<f64> {
        self.inner.valuation()
    }

    /// Develop every loss to ultimate, if asked, and trend it to the
    /// prospective period.
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
    /// development : DevelopmentPattern or list of float, optional
    ///     Develops each known loss to ultimate (IBNER): a pattern, read at
    ///     each loss's age at the valuation date, or a factor per loss.
    ///     Then give ``burning_cost`` only the count IBNR factors, not an
    ///     excess development factor that already holds IBNER.
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
    #[pyo3(signature = (*, rate = None, to = None, factors = None, development = None))]
    fn prepare(
        &self,
        rate: Option<f64>,
        to: Option<f64>,
        factors: Option<Vec<f64>>,
        development: Option<&Bound<'_, PyAny>>,
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
        let inner = match development {
            None => self.inner.prepare(&trend),
            Some(d) => {
                let development = if let Ok(p) = d.cast::<PyDevelopmentPattern>() {
                    Development::Pattern(p.get().inner.clone())
                } else {
                    Development::Factors(d.extract::<Vec<f64>>().map_err(|_| {
                        PyTypeError::new_err(
                            "development must be a DevelopmentPattern or a list of floats",
                        )
                    })?)
                };
                self.inner.prepare_developed(&trend, &development)
            }
        };
        Ok(PyPreparedLosses {
            inner: inner.map_err(to_py)?,
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
    /// Every loss's trended amount, in the record's order.
    #[getter]
    fn trended(&self) -> Vec<f64> {
        self.inner.trended()
    }

    /// The record with the losses whose ids are in ``ids`` left out (an
    /// excluded segment, say), its completeness unchanged. Run
    /// ``burning_cost`` on both, with the segment's premium left out of the
    /// years' premiums too.
    ///
    /// Parameters
    /// ----------
    /// ids : list of str
    ///
    /// Returns
    /// -------
    /// PreparedLosses
    fn without(&self, ids: Vec<String>) -> Self {
        let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
        Self {
            inner: self.inner.without(&ids),
        }
    }

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
    /// development : list of float or DevelopmentPattern
    ///     Each year's development factor for the layer losses: Clark's
    ///     excess factor, or, when the losses were developed one by one in
    ///     ``prepare``, the count IBNR factor only. A count
    ///     ``DevelopmentPattern`` is read at each year's average age at the
    ///     record's valuation date.
    /// index : IndexClause, optional
    ///     An index clause on the treaty: each loss meets the retention and
    ///     limit moved by the clause at its own ratio in ``ratios``. Towers
    ///     with annual terms (an aggregate deductible or limit, a corridor)
    ///     or a surplus treaty are an error.
    /// ratios : list of float, optional
    ///     With ``index``, each loss's index ratio, in the record's order:
    ///     the index at its payment over the index at the treaty's base date.
    ///
    /// Returns
    /// -------
    /// BurningCost
    #[pyo3(signature = (towers, years, *, premium, development, index = None, ratios = None))]
    #[allow(clippy::too_many_arguments)]
    fn burning_cost(
        &self,
        towers: &Bound<'_, PyAny>,
        years: Vec<i32>,
        premium: Vec<f64>,
        development: &Bound<'_, PyAny>,
        index: Option<PyRef<'_, PyIndexClause>>,
        ratios: Option<Vec<f64>>,
    ) -> PyResult<PyBurningCost> {
        let development: Vec<f64> = if let Ok(p) = development.cast::<PyDevelopmentPattern>() {
            let valuation = self.inner.valuation().ok_or_else(|| {
                PyValueError::new_err("a count pattern needs the record's valuation date")
            })?;
            years
                .iter()
                .zip(&premium)
                .map(|(&y, &pr)| {
                    YearBasis::from_count_pattern(y, pr, &p.get().inner, valuation)
                        .map(|b| b.development)
                        .map_err(to_py)
                })
                .collect::<PyResult<_>>()?
        } else {
            development.extract().map_err(|_| {
                PyTypeError::new_err("development must be a list of floats or a DevelopmentPattern")
            })?
        };
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
        let inner = match (index, ratios) {
            (None, None) => self.inner.burning_cost(&towers, &basis),
            (Some(clause), Some(ratios)) => {
                self.inner
                    .burning_cost_indexed(&towers, &basis, &clause.inner, &ratios)
            }
            _ => {
                return Err(PyTypeError::new_err(
                    "pass index and ratios together, or neither",
                ));
            }
        };
        Ok(PyBurningCost {
            inner: inner.map_err(to_py)?,
        })
    }

    /// Each year's losses under the program in force that year, next to the
    /// year's trended losses under the proposed tower.
    ///
    /// The historical side takes every loss of the year as reported (in its
    /// own currency, untrended and undeveloped), so it can be checked
    /// against the cedant's recoveries; the proposed side is the as-if
    /// layer loss of ``burning_cost`` before development by year. The
    /// historical towers may differ in layers from year to year.
    ///
    /// Parameters
    /// ----------
    /// historical : list of Tower
    ///     The program in force in each of ``years``.
    /// years : list of int
    /// proposed : Tower
    ///
    /// Returns
    /// -------
    /// AsIf
    fn as_if(
        &self,
        historical: Vec<PyRef<'_, PyTower>>,
        years: Vec<i32>,
        proposed: PyRef<'_, PyTower>,
    ) -> PyResult<PyAsIf> {
        let towers: Vec<Tower> = historical.iter().map(|t| t.inner.clone()).collect();
        Ok(PyAsIf {
            inner: self
                .inner
                .as_if(&towers, &years, &proposed.inner)
                .map_err(to_py)?,
        })
    }

    /// The record's valuation date, or ``None``.
    #[getter]
    fn valuation(&self) -> Option<f64> {
        self.inner.valuation()
    }

    /// The trended losses as columns: ``id``, ``year``, ``accident``,
    /// ``amount``, ``fx``, ``development``, ``ultimate``, ``factor`` and
    /// ``trended``.
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
        d.set_item("fx", l.iter().map(|x| x.fx).collect::<Vec<_>>())?;
        d.set_item(
            "development",
            l.iter().map(|x| x.development).collect::<Vec<_>>(),
        )?;
        d.set_item("ultimate", l.iter().map(|x| x.ultimate).collect::<Vec<_>>())?;
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

/// Each year's cessions under its own program and under the proposed one,
/// from ``PreparedLosses.as_if``.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import LargeLossExperience
/// >>> from prospicio.reinsurance import Layer, Tower
/// >>> p = LargeLossExperience(["a"], [2019.5], [300.0]).prepare(factors=[1.2])
/// >>> then = Tower([Layer("150 xs 150", 150.0, 150.0)])
/// >>> now = Tower([Layer("200 xs 200", 200.0, 200.0)])
/// >>> a = p.as_if([then], [2019], now)
/// >>> a.to_dict()["historical"], [round(x, 6) for x in a.to_dict()["proposed"][0]]
/// ([[150.0]], [160.0])
#[pyclass(name = "AsIf", module = "prospicio.pricing", frozen)]
pub(crate) struct PyAsIf {
    inner: AsIf,
}

#[pymethods]
impl PyAsIf {
    /// The proposed tower's layer names.
    #[getter]
    fn proposed_layers(&self) -> Vec<String> {
        self.inner.proposed_layers.clone()
    }

    /// The years as columns: ``year``, ``reported_losses``,
    /// ``trended_losses``, and per year the lists ``historical_layers``,
    /// ``historical`` (one value per layer in force that year) and
    /// ``proposed`` (one per proposed layer).
    ///
    /// Returns
    /// -------
    /// dict of str to list
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let y = &self.inner.years;
        let d = PyDict::new(py);
        d.set_item("year", y.iter().map(|r| r.year).collect::<Vec<_>>())?;
        d.set_item(
            "reported_losses",
            y.iter().map(|r| r.reported_losses).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "trended_losses",
            y.iter().map(|r| r.trended_losses).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "historical_layers",
            y.iter()
                .map(|r| r.historical_layers.clone())
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "historical",
            y.iter().map(|r| r.historical.clone()).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "proposed",
            y.iter().map(|r| r.proposed.clone()).collect::<Vec<_>>(),
        )?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "AsIf(years={}, proposed_layers={:?})",
            self.inner.years.len(),
            self.inner.proposed_layers
        )
    }
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

/// Development factors to ultimate by age.
///
/// ``at(age)`` interpolates linearly in age between the given ages, takes
/// the first factor at younger ages and the last at older ones. Use one to
/// develop each known large loss to ultimate (IBNER) in
/// ``LargeLossExperience.prepare``, or as a count pattern for the claims not
/// yet reported in ``PreparedLosses.burning_cost``.
///
/// Parameters
/// ----------
/// ages : list of float
///     Strictly increasing ages, in years since the accident date.
/// factors : list of float
///     The factor to ultimate at each age.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import DevelopmentPattern
/// >>> p = DevelopmentPattern.from_history(
/// ...     ["a", "a", "b", "b"], [1.0, 2.0, 1.0, 2.0], [100.0, 150.0, 50.0, 50.0], ages=[1.0, 2.0]
/// ... )
/// >>> [round(f, 4) for f in p.factors]
/// [1.3333, 1.0]
/// >>> round(p.at(1.5), 4)
/// 1.1667
#[pyclass(name = "DevelopmentPattern", module = "prospicio.pricing", frozen)]
pub(crate) struct PyDevelopmentPattern {
    inner: DevelopmentPattern,
}

#[pymethods]
impl PyDevelopmentPattern {
    #[new]
    fn new(ages: Vec<f64>, factors: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: DevelopmentPattern::new(ages, factors).map_err(to_py)?,
        })
    }

    /// Estimated from each claim's valuations.
    ///
    /// The age-to-age factor from ``ages[k]`` to ``ages[k + 1]`` is the sum
    /// of the amounts at the later age over the sum at the earlier, over the
    /// claims valued at both with a positive earlier amount (a claim first
    /// seen later is IBNR, not IBNER). The factor at the last age is 1. With
    /// every claim valued at every age up to its latest, this is the
    /// volume-weighted chain ladder on the summed triangle.
    ///
    /// Parameters
    /// ----------
    /// ids : list of str
    ///     The claim of each valuation.
    /// age : list of float
    ///     The claim's age at each valuation, in years; valuations at an age
    ///     not in ``ages`` are ignored.
    /// amount : list of float
    ///     The claim's amount at each valuation.
    /// ages : list of float
    ///     The ages of the pattern.
    ///
    /// Returns
    /// -------
    /// DevelopmentPattern
    #[staticmethod]
    #[pyo3(signature = (ids, age, amount, *, ages))]
    fn from_history(
        ids: Vec<String>,
        age: Vec<f64>,
        amount: Vec<f64>,
        ages: Vec<f64>,
    ) -> PyResult<Self> {
        if ids.len() != age.len() || ids.len() != amount.len() {
            return Err(PyValueError::new_err(format!(
                "{} ids, {} ages and {} amounts",
                ids.len(),
                age.len(),
                amount.len()
            )));
        }
        let valuations: Vec<ClaimValuation> = ids
            .into_iter()
            .zip(age)
            .zip(amount)
            .map(|((id, age), amount)| ClaimValuation { id, age, amount })
            .collect();
        Ok(Self {
            inner: DevelopmentPattern::from_history(&valuations, &ages).map_err(to_py)?,
        })
    }

    /// The ages.
    #[getter]
    fn ages(&self) -> Vec<f64> {
        self.inner.ages().to_vec()
    }

    /// The factors to ultimate at each age.
    #[getter]
    fn factors(&self) -> Vec<f64> {
        self.inner.factors().to_vec()
    }

    /// The age-to-age factors.
    #[getter]
    fn age_to_age(&self) -> Vec<f64> {
        self.inner.age_to_age()
    }

    /// The factor to ultimate at ``age``.
    ///
    /// Parameters
    /// ----------
    /// age : float
    ///
    /// Returns
    /// -------
    /// float
    fn at(&self, age: f64) -> f64 {
        self.inner.at(age)
    }

    fn __repr__(&self) -> String {
        format!(
            "DevelopmentPattern(ages={:?}, factors={:?})",
            self.inner.ages(),
            self.inner.factors()
        )
    }
}

/// An index (stability) clause: the treaty's retention and limit move with
/// an index, by the ratio ``r`` of the index at a loss's payment to the
/// index at the treaty's base date, within a franchise.
///
/// With ``kind="full"``, nothing moves while ``|r - 1|`` is within the
/// franchise, and beyond it the terms move by all of ``r``, up or down.
/// With ``kind="severe"`` (a severe inflation clause), the terms move only
/// by the rise beyond the franchise, ``r / (1 + franchise)``, and never
/// down.
///
/// Parameters
/// ----------
/// franchise : float
///     A fraction, such as 0.1 for 10%.
/// kind : {"full", "severe"}, default "full"
///
/// Examples
/// --------
/// >>> from prospicio.pricing import IndexClause
/// >>> IndexClause(0.1).adjustment(1.25)
/// 1.25
/// >>> round(IndexClause(0.1, kind="severe").adjustment(1.21), 12)
/// 1.1
#[pyclass(name = "IndexClause", module = "prospicio.pricing", frozen)]
pub(crate) struct PyIndexClause {
    inner: IndexClause,
}

#[pymethods]
impl PyIndexClause {
    #[new]
    #[pyo3(signature = (franchise, kind = "full"))]
    fn new(franchise: f64, kind: &str) -> PyResult<Self> {
        let kind = match kind {
            "full" => IndexKind::Full,
            "severe" => IndexKind::Severe,
            other => {
                return Err(PyValueError::new_err(format!(
                    "kind must be \"full\" or \"severe\", not {other:?}"
                )));
            }
        };
        Ok(Self {
            inner: IndexClause::new(franchise, kind).map_err(to_py)?,
        })
    }

    /// The franchise.
    #[getter]
    fn franchise(&self) -> f64 {
        self.inner.franchise()
    }

    /// ``"full"`` or ``"severe"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind() {
            IndexKind::Full => "full",
            IndexKind::Severe => "severe",
        }
    }

    /// The factor the terms move by at index ratio ``ratio``.
    ///
    /// Parameters
    /// ----------
    /// ratio : float
    ///
    /// Returns
    /// -------
    /// float
    fn adjustment(&self, ratio: f64) -> f64 {
        self.inner.adjustment(ratio)
    }

    fn __repr__(&self) -> String {
        format!(
            "IndexClause(franchise={}, kind={:?})",
            self.inner.franchise(),
            self.kind()
        )
    }
}
