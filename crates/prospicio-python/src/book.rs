//! `prospicio.pricing` (Aggregate lane): a book of deals on one shared
//! simulation over `prospicio_pricing::book` (`docs/design/time-value.md`,
//! decisions 21 to 27).

use std::collections::HashMap;

use prospicio_pricing::book::{DealBook, DealRun, Tail};
use prospicio_pricing::cashflow::MonthlyPatterns;
use prospicio_prob::statistics::{Statistics, StatisticsGrid};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::cashflow::{PyCashflowTerms, PyCashflows};
use crate::distributions::PyPredictiveDistribution;
use crate::payout::{PyIrrResult, irr_options};
use crate::pricing::premium_rule;
use crate::reinsurance::PyTower;
use crate::risk::{PyDistortion, quantile_method, tvar_method};
use crate::to_py;

/// One deal's results in each simulation of a shared simulation: its
/// premium, its loss and its deductions (a commission, the brokerage, a
/// fee), each a list with one amount a simulation, and its inception in
/// months from 1 January of the book's first year.
///
/// ``deductions`` maps each deduction's name to its amounts, or to
/// ``(amounts, outside_nmbf)`` for one NMBF leaves out (a manager's fee).
/// The ``with_`` methods return a new run.
///
/// >>> from prospicio.pricing import DealRun
/// >>> run = DealRun("cat", [100.0, 100.0], [40.0, 150.0],
/// ...               deductions={"brokerage": [2.5, 2.5]}, inception=6)
/// >>> run.n_sims, run.inception
/// (2, 6)
#[pyclass(name = "DealRun", module = "prospicio.pricing", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyDealRun {
    inner: DealRun,
}

fn deductions(run: DealRun, given: Option<&Bound<'_, PyDict>>) -> PyResult<DealRun> {
    let Some(given) = given else {
        return Ok(run);
    };
    let mut run = run;
    for (name, value) in given.iter() {
        let name: String = name.extract()?;
        let (values, outside): (Vec<f64>, bool) = match value.extract::<Vec<f64>>() {
            Ok(v) => (v, false),
            Err(_) => value.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "deduction {name:?} must be a list of amounts or (amounts, outside_nmbf)"
                ))
            })?,
        };
        run = run.deduction(&name, values, outside).map_err(to_py)?;
    }
    Ok(run)
}

#[pymethods]
impl PyDealRun {
    #[new]
    #[pyo3(signature = (name, premium, loss, *, deductions = None, inception = 0))]
    fn new(
        name: &str,
        premium: Vec<f64>,
        loss: Vec<f64>,
        deductions: Option<&Bound<'_, PyDict>>,
        inception: usize,
    ) -> PyResult<Self> {
        let run = DealRun::new(name, premium, loss).map_err(to_py)?;
        let inner = self::deductions(run, deductions)?.inception(inception);
        Ok(PyDealRun { inner })
    }

    /// A layer's or deal's run from a result of ``tower``: its premium,
    /// ceded loss, commissions, brokerage and deductions, the amounts
    /// behind ``Tower.deal_ratios``, under the name ``name``.
    ///
    /// Each year's premium is the deposit (or swing-rated) premium plus the
    /// year's reinstatement premium from the tower, pro rata as to amount
    /// and, where the layer says so, as to time; ``reinstatement_premium=False``
    /// leaves it out, so the premium is the deposit in every year.
    ///
    /// >>> from prospicio.pricing import DealRun
    /// >>> from prospicio.reinsurance import Layer, Tower
    /// >>> cover = (Layer.stop_loss("cover", 120.0, 0.0)
    /// ...          .with_premium_rate(1.0, 100.0).with_brokerage(0.025))
    /// >>> tower = Tower([cover])
    /// >>> run = DealRun.from_tower("cover", tower, tower.apply_annual([40.0, 150.0]), "cover")
    /// >>> run.n_sims
    /// 2
    /// >>> from prospicio.aggregate import EventSet
    /// >>> cat = Layer("cat", 10.0, 10.0, premium=2.0, reinstatement_rates=[1.0])
    /// >>> tower = Tower([cat])
    /// >>> result = tower.apply(EventSet.from_years([[5.0], [15.0]]))
    /// >>> DealRun.from_tower("cat", tower, result, "cat").premium
    /// [2.0, 3.0]
    /// >>> DealRun.from_tower("cat", tower, result, "cat", reinstatement_premium=False).premium
    /// [2.0, 2.0]
    #[staticmethod]
    #[pyo3(signature = (name, tower, result, layer, *, inception = 0, reinstatement_premium = true))]
    fn from_tower(
        name: &str,
        tower: PyRef<'_, PyTower>,
        result: PyRef<'_, PyPredictiveDistribution>,
        layer: &str,
        inception: usize,
        reinstatement_premium: bool,
    ) -> PyResult<Self> {
        let inner = DealRun::from_tower_with(
            name,
            &tower.inner,
            &result.inner,
            layer,
            reinstatement_premium,
        )
        .map_err(to_py)?
        .inception(inception);
        Ok(PyDealRun { inner })
    }

    /// The run incepting at ``month``, counted from 1 January of the
    /// book's first year. It moves the deal's flows and patterns, not its
    /// losses.
    fn with_inception(&self, month: usize) -> Self {
        PyDealRun {
            inner: self.inner.clone().inception(month),
        }
    }

    /// The run with its monthly cumulative written, earned and paid
    /// shares, for ``DealBook.calendar``.
    fn with_patterns(&self, written: Vec<f64>, earned: Vec<f64>, paid: Vec<f64>) -> PyResult<Self> {
        let p = MonthlyPatterns::new(written, earned, paid).map_err(to_py)?;
        Ok(PyDealRun {
            inner: self.inner.clone().patterns(p),
        })
    }

    /// The run with its investor's flows from ``terms.run(...)``, one
    /// simulation for each of the deal's, and the terms' patterns.
    fn with_cashflows(
        &self,
        terms: PyRef<'_, PyCashflowTerms>,
        cashflows: PyRef<'_, PyCashflows>,
    ) -> PyResult<Self> {
        let inner = self
            .inner
            .clone()
            .cashflows(&terms.inner, &cashflows.inner)
            .map_err(to_py)?;
        Ok(PyDealRun { inner })
    }

    /// The run with its investor's net flows from any model: one list of
    /// monthly flows from inception a simulation, and the capital paid in
    /// during the term (``funded``) and at inception (``opening``).
    fn with_flows(
        &self,
        flows: Vec<Vec<f64>>,
        funded: Vec<f64>,
        opening: Vec<f64>,
    ) -> PyResult<Self> {
        let months = flows.first().map_or(0, Vec::len);
        if flows.iter().any(|r| r.len() != months) {
            return Err(PyValueError::new_err(
                "every row of flows needs the same months",
            ));
        }
        let inner = self
            .inner
            .clone()
            .flows(months, flows.concat(), funded, opening)
            .map_err(to_py)?;
        Ok(PyDealRun { inner })
    }

    /// The deal's name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().into()
    }

    /// The number of simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims()
    }

    /// The premium in each simulation.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner.premium().to_vec()
    }

    /// The loss in each simulation.
    #[getter]
    fn loss(&self) -> Vec<f64> {
        self.inner.loss().to_vec()
    }

    /// The inception month.
    #[getter]
    fn inception(&self) -> usize {
        self.inner.inception_month()
    }

    fn __repr__(&self) -> String {
        format!(
            "DealRun({:?}, n={}, inception={})",
            self.inner.name(),
            self.inner.n_sims(),
            self.inner.inception_month()
        )
    }
}

fn stats_dict<'py>(py: Python<'py>, s: &Statistics) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (name, value) in s.names.iter().zip(&s.values) {
        d.set_item(name, value)?;
    }
    Ok(d)
}

/// The book's investors' flows from ``DealBook.cashflows``.
#[pyclass(name = "BookCashflows", module = "prospicio.pricing", frozen)]
pub(crate) struct PyBookCashflows {
    inner: prospicio_pricing::book::BookCashflows,
}

#[pymethods]
impl PyBookCashflows {
    /// The book's months, month 0 at 1 January of its first year.
    #[getter]
    fn months(&self) -> usize {
        self.inner.months
    }

    /// The deals' names, in the book's order.
    #[getter]
    fn deals(&self) -> Vec<String> {
        self.inner.deals.clone()
    }

    /// The book's net flow in each month, one list a simulation: the
    /// deals' flows summed after their inceptions.
    #[getter]
    fn flows(&self) -> Vec<Vec<f64>> {
        let m = self.inner.months.max(1);
        self.inner.flows.chunks(m).map(<[f64]>::to_vec).collect()
    }

    /// Each deal's mean flow in each month of the book, by name.
    #[getter]
    fn mean_by_deal(&self) -> HashMap<String, Vec<f64>> {
        self.inner
            .deals
            .iter()
            .cloned()
            .zip(self.inner.mean_by_deal.iter().cloned())
            .collect()
    }

    /// The book's IRR in each simulation, on its summed flows.
    #[getter]
    fn irr(&self) -> PyIrrResult {
        PyIrrResult {
            inner: self.inner.irr.clone(),
        }
    }

    /// The book's NPV in each simulation, to month 0.
    #[getter]
    fn npv(&self) -> Vec<f64> {
        self.inner.npv.clone()
    }

    /// The book's MOIC in each simulation: one plus its summed flows over
    /// the capital funded and opened over every deal.
    #[getter]
    fn moic(&self) -> Vec<f64> {
        self.inner.moic.clone()
    }

    /// Each deal's own IRR in each simulation, by name.
    #[getter]
    fn irr_by_deal(&self) -> HashMap<String, PyIrrResult> {
        self.inner
            .deals
            .iter()
            .cloned()
            .zip(self.inner.irr_by_deal.iter().map(|inner| PyIrrResult {
                inner: inner.clone(),
            }))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "BookCashflows(deals={}, months={})",
            self.inner.deals.len(),
            self.inner.months
        )
    }
}

/// Several deals on one shared simulation: simulation ``j`` of every deal
/// is the same year, so the book adds them without reordering.
///
/// >>> from prospicio.pricing import DealBook, DealRun
/// >>> a = DealRun("a", [100.0, 100.0], [40.0, 80.0])
/// >>> b = DealRun("b", [300.0, 300.0], [150.0, 330.0])
/// >>> s = DealBook([a, b]).summary()
/// >>> [r for d, i, r in zip(s["deal"], s["item"], s["ratio"]) if (d, i) == ("total", "loss")]
/// [0.75]
#[pyclass(name = "DealBook", module = "prospicio.pricing", frozen)]
pub(crate) struct PyDealBook {
    inner: DealBook,
}

#[pymethods]
impl PyDealBook {
    #[new]
    fn new(deals: Vec<PyDealRun>) -> PyResult<Self> {
        let inner = DealBook::new(deals.into_iter().map(|d| d.inner).collect()).map_err(to_py)?;
        Ok(PyDealBook { inner })
    }

    /// The deals named in ``names``, in that order.
    fn select(&self, names: Vec<String>) -> PyResult<Self> {
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let inner = self.inner.select(&names).map_err(to_py)?;
        Ok(PyDealBook { inner })
    }

    /// The book with the deal ``name`` incepting at ``month`` instead.
    fn shifted(&self, name: &str, month: usize) -> PyResult<Self> {
        let inner = self.inner.shifted(name, month).map_err(to_py)?;
        Ok(PyDealBook { inner })
    }

    /// The deals' names, in order.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names().into_iter().map(String::from).collect()
    }

    /// The number of simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims()
    }

    /// The items of ``amounts``: ``premium``, ``loss``, each deduction,
    /// ``total_deductions``, ``nmbf`` and ``margin``.
    #[getter]
    fn items(&self) -> Vec<String> {
        self.inner.items()
    }

    /// Every item of every deal and of the book in each simulation, with
    /// dimensions ``["item", "deal"]``, the deals then ``"total"``.
    fn amounts(&self, py: Python<'_>) -> PyResult<PyPredictiveDistribution> {
        let book = &self.inner;
        let inner = py.detach(|| book.amounts()).map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    /// The mean of every item by deal and in total with its ratio to the
    /// mean premium, as columns ``deal``, ``item``, ``mean`` and ``ratio``.
    /// The book's ratios are sums over sums.
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let rows = self.inner.summary().map_err(to_py)?;
        let d = PyDict::new(py);
        d.set_item("deal", rows.iter().map(|r| r.0.clone()).collect::<Vec<_>>())?;
        d.set_item("item", rows.iter().map(|r| r.1.clone()).collect::<Vec<_>>())?;
        d.set_item("mean", rows.iter().map(|r| r.2).collect::<Vec<_>>())?;
        d.set_item("ratio", rows.iter().map(|r| r.3).collect::<Vec<_>>())?;
        Ok(d)
    }

    /// The book's statistics of ``item`` and, for each deal, its
    /// contribution at the book's ranks and its marginal effect (the book
    /// less the book without it), as a dict with ``"total"``,
    /// ``"contribution"`` and ``"marginal"`` (each deal's cells by name).
    /// ``tail="lower"`` reads low values as the risk, as for a margin.
    /// ``percentiles`` and the keywords set the grid as ``statistics``
    /// does.
    ///
    /// >>> from prospicio.pricing import DealBook, DealRun
    /// >>> a = DealRun("a", [10.0] * 4, [1.0, 2.0, 3.0, 9.0])
    /// >>> b = DealRun("b", [10.0] * 4, [4.0, 1.0, 2.0, 3.0])
    /// >>> d = DealBook([a, b]).distribution("loss", [], tvar_levels=[50.0])
    /// >>> d["total"]["TVaR 50"], d["contribution"]["a"]["TVaR 50"]
    /// (8.5, 6.0)
    #[pyo3(signature = (item, percentiles, *, tvar_levels = None, sd_multiples = None, quantile = "inverted_cdf", tvar = "exact", tail = "upper"))]
    #[allow(clippy::too_many_arguments)]
    fn distribution<'py>(
        &self,
        py: Python<'py>,
        item: &str,
        percentiles: Vec<f64>,
        tvar_levels: Option<Vec<f64>>,
        sd_multiples: Option<Vec<f64>>,
        quantile: &str,
        tvar: &str,
        tail: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut grid = StatisticsGrid::new(percentiles)
            .map_err(to_py)?
            .quantile_method(quantile_method(quantile)?)
            .tvar_method(tvar_method(tvar)?);
        if let Some(levels) = tvar_levels {
            grid = grid.tvar_levels(levels).map_err(to_py)?;
        }
        if let Some(k) = sd_multiples {
            grid = grid.sd_multiples(k).map_err(to_py)?;
        }
        let tail = match tail {
            "upper" => Tail::Upper,
            "lower" => Tail::Lower,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown tail {other:?}; use \"upper\" or \"lower\""
                )));
            }
        };
        let book = &self.inner;
        let out = py
            .detach(|| book.distribution(item, &grid, tail))
            .map_err(to_py)?;
        let d = PyDict::new(py);
        d.set_item("item", &out.item)?;
        d.set_item("total", stats_dict(py, &out.total)?)?;
        let contribution = PyDict::new(py);
        let marginal = PyDict::new(py);
        for c in &out.deals {
            contribution.set_item(&c.deal, stats_dict(py, &c.contribution)?)?;
            marginal.set_item(&c.deal, stats_dict(py, &c.marginal)?)?;
        }
        d.set_item("contribution", contribution)?;
        d.set_item("marginal", marginal)?;
        Ok(d)
    }

    /// The marginal price of each deal written last: the book's loss
    /// priced with the deal and without it, on the assets of the capital
    /// standard ``assets`` and by exactly one of ``cost_of_capital`` and
    /// ``distortion`` as ``price_portfolio`` prices a portfolio, and the
    /// differences. Under a cost of capital ``r`` the premium is
    /// ``(EL + r dA) / (1 + r)``. Under a subadditive standard it is at
    /// most the deal's allocated premium from ``price_portfolio``.
    ///
    /// Returns a dict of columns, one row a deal (those in ``deals``, all
    /// by default): ``deal``, ``expected_loss``, ``assets`` (``dA``),
    /// ``premium``, ``margin``, and the book's ``book_premium``,
    /// ``book_assets``, ``premium_without`` and ``assets_without``.
    ///
    /// >>> from prospicio.pricing import DealBook, DealRun
    /// >>> from prospicio.risk import Distortion
    /// >>> a = DealRun("a", [10.0] * 4, [1.0, 2.0, 3.0, 9.0])
    /// >>> b = DealRun("b", [10.0] * 4, [4.0, 1.0, 2.0, 3.0])
    /// >>> m = DealBook([a, b]).marginal_price(Distortion.tvar(0.5), cost_of_capital=0.1)
    /// >>> m["assets"], round(m["premium"][1], 6)
    /// ([5.0, 2.5], 2.5)
    #[pyo3(signature = (assets, *, cost_of_capital = None, distortion = None, deals = None))]
    fn marginal_price<'py>(
        &self,
        py: Python<'py>,
        assets: PyRef<'_, PyDistortion>,
        cost_of_capital: Option<f64>,
        distortion: Option<PyRef<'_, PyDistortion>>,
        deals: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rule = premium_rule(cost_of_capital, distortion)?;
        let names = deals.unwrap_or_else(|| self.names());
        let (book, a) = (&self.inner, assets.inner.clone());
        let rows = py
            .detach(|| {
                names
                    .iter()
                    .map(|n| book.marginal_price(n, &rule, &a))
                    .collect::<prospicio_core::Result<Vec<_>>>()
            })
            .map_err(to_py)?;
        let d = PyDict::new(py);
        let col = |f: &dyn Fn(&prospicio_pricing::book::MarginalPrice) -> f64| -> Vec<f64> {
            rows.iter().map(f).collect()
        };
        d.set_item("deal", names)?;
        d.set_item("expected_loss", col(&|m| m.expected_loss()))?;
        d.set_item("assets", col(&|m| m.assets()))?;
        d.set_item("premium", col(&|m| m.premium()))?;
        d.set_item("margin", col(&|m| m.margin()))?;
        d.set_item("book_premium", col(&|m| m.with.premium))?;
        d.set_item("book_assets", col(&|m| m.with.assets))?;
        d.set_item("premium_without", col(&|m| m.without.premium))?;
        d.set_item("assets_without", col(&|m| m.without.assets))?;
        Ok(d)
    }

    /// Price adequacy for each deal and the book: the premium charged and
    /// kept (less every deduction) beside the expected loss and margin, the
    /// allocated (Euler) and marginal assets and capital and the return on
    /// each, the technical and marginal premiums the rule gives, the target
    /// premium that returns the target on the allocated capital and the
    /// adequacy, kept over target. The book's losses are priced on the
    /// assets of the capital standard ``assets`` by exactly one of
    /// ``cost_of_capital`` and ``distortion``, as ``price_portfolio``
    /// prices a portfolio; the target return is the book's return at its
    /// technical price (the cost of capital, under one).
    ///
    /// Returns a dict of columns, one row a deal then ``"total"``:
    /// ``deal``, ``premium_charged``, ``premium_kept``, ``expected_loss``,
    /// ``margin``, ``allocated_assets``, ``allocated_capital``,
    /// ``return_on_allocated``, ``marginal_assets``, ``marginal_capital``,
    /// ``return_on_marginal``, ``technical_premium``, ``marginal_premium``,
    /// ``target_premium``, ``adequacy`` and ``target_return``.
    ///
    /// >>> from prospicio.pricing import DealBook, DealRun
    /// >>> from prospicio.risk import Distortion
    /// >>> a = DealRun("a", [10.0] * 4, [1.0, 2.0, 3.0, 9.0])
    /// >>> b = DealRun("b", [3.0] * 4, [4.0, 1.0, 2.0, 3.0])
    /// >>> r = DealBook([a, b]).adequacy(Distortion.tvar(0.5), cost_of_capital=0.1)
    /// >>> r["deal"], r["premium_kept"]
    /// (['a', 'b', 'total'], [10.0, 3.0, 13.0])
    /// >>> round(r["target_return"][0], 12)
    /// 0.1
    #[pyo3(signature = (assets, *, cost_of_capital = None, distortion = None))]
    fn adequacy<'py>(
        &self,
        py: Python<'py>,
        assets: PyRef<'_, PyDistortion>,
        cost_of_capital: Option<f64>,
        distortion: Option<PyRef<'_, PyDistortion>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rule = premium_rule(cost_of_capital, distortion)?;
        let (book, a) = (&self.inner, assets.inner.clone());
        let r = py.detach(|| book.adequacy(&rule, &a)).map_err(to_py)?;
        let rows = &r.rows;
        let col = |f: &dyn Fn(&prospicio_pricing::book::AdequacyRow) -> f64| -> Vec<f64> {
            rows.iter().map(f).collect()
        };
        let d = PyDict::new(py);
        d.set_item(
            "deal",
            rows.iter().map(|x| x.deal.clone()).collect::<Vec<_>>(),
        )?;
        d.set_item("premium_charged", col(&|x| x.premium_charged))?;
        d.set_item("premium_kept", col(&|x| x.premium_kept))?;
        d.set_item("expected_loss", col(&|x| x.expected_loss))?;
        d.set_item("margin", col(&|x| x.margin))?;
        d.set_item("allocated_assets", col(&|x| x.allocated_assets))?;
        d.set_item("allocated_capital", col(&|x| x.allocated_capital))?;
        d.set_item("return_on_allocated", col(&|x| x.return_on_allocated))?;
        d.set_item("marginal_assets", col(&|x| x.marginal_assets))?;
        d.set_item("marginal_capital", col(&|x| x.marginal_capital))?;
        d.set_item("return_on_marginal", col(&|x| x.return_on_marginal))?;
        d.set_item("technical_premium", col(&|x| x.technical_premium))?;
        d.set_item("marginal_premium", col(&|x| x.marginal_premium))?;
        d.set_item("target_premium", col(&|x| x.target_premium))?;
        d.set_item("adequacy", col(&|x| x.adequacy))?;
        d.set_item("target_return", vec![r.target_return; rows.len()])?;
        Ok(d)
    }

    /// The investors' flows on the book's calendar, each deal's moved to
    /// its inception and summed in each simulation, with the book's IRR,
    /// NPV at ``rate`` and MOIC on the summed flows. Every deal needs
    /// flows (``DealRun.with_cashflows`` or ``with_flows``). The IRR is
    /// searched as ``irr`` searches, from -95% to 100% a year by default: a
    /// simulation whose return is above the range has ``found`` false and
    /// ``root_above`` true in ``irr``, and ``extend=True`` (or a larger
    /// ``upper``) finds it, so leaving those simulations out does not bias
    /// the IRR statistics downwards.
    #[pyo3(signature = (rate = 0.0, *, lower = None, upper = None, points = 400, bisections = 8, newton_steps = 3, extend = false))]
    #[allow(clippy::too_many_arguments)]
    fn cashflows(
        &self,
        py: Python<'_>,
        rate: f64,
        lower: Option<f64>,
        upper: Option<f64>,
        points: usize,
        bisections: usize,
        newton_steps: usize,
        extend: bool,
    ) -> PyResult<PyBookCashflows> {
        let options = irr_options(lower, upper, points, bisections, newton_steps, extend);
        let book = &self.inner;
        let inner = py
            .detach(|| book.cashflows(rate, &options))
            .map_err(to_py)?;
        Ok(PyBookCashflows { inner })
    }

    /// The book by calendar year, with dimensions ``["item", "deal",
    /// "year"]``: written and earned premium and paid loss from each deal's
    /// patterns, and the investors' flows when every deal has them; book
    /// month ``m`` falls in ``start_year + m // 12``.
    fn calendar(&self, py: Python<'_>, start_year: i64) -> PyResult<PyPredictiveDistribution> {
        let book = &self.inner;
        let inner = py.detach(|| book.calendar(start_year)).map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.deals().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "DealBook({:?}, n={})",
            self.inner.names(),
            self.inner.n_sims()
        )
    }
}
