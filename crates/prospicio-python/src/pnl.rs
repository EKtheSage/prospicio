//! `prospicio.aggregate` (Aggregate lane): the P&L ledger, over
//! `prospicio_aggregate::pnl`.

use prospicio_aggregate::pnl::{
    Book as RsBook, Expense as RsExpense, ExpenseBasis, Face, Group as RsGroup, LADDER, Layout,
    Ledger as RsLedger, Leg as RsLeg, LegKind, ProgramLayer, Role, Row, RowKind, Source, Stats,
    derive_premium as rs_derive_premium, pnl_program as rs_pnl_program,
    round_premium as rs_round_premium, size_premium as rs_size_premium, snap, stack as rs_stack,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::aggregate::{AnyCount, PyBivariateReport};
use crate::discrete::PyDiscrete;
use crate::distributions::PyGrid;
use crate::reinsurance::{PyLossSensitivePremium, PyTower};
use crate::to_py;

fn kind_of(name: Option<&str>) -> PyResult<Option<LegKind>> {
    Ok(match name {
        None => None,
        Some("premium") => Some(LegKind::Premium),
        Some("loss") => Some(LegKind::Loss),
        Some("expense") => Some(LegKind::Expense),
        Some("recovery") => Some(LegKind::Recovery),
        Some("commission") => Some(LegKind::Commission),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "a leg's kind is premium, loss, expense, recovery or commission, not {other:?}"
            )));
        }
    })
}

/// One cash flow of a :class:`Group`: a constant, or a function of the
/// source's value, written as a positive magnitude; the group's role signs
/// it. A function is called once with all the source's values (a numpy
/// array when numpy is installed, else a list) and returns one value per
/// value, or a single number.
///
/// Parameters
/// ----------
/// label : str
///     Unique across the ledger.
/// value : float or callable
/// kind : str, optional
///     ``premium``, ``loss``, ``expense``, ``recovery`` or ``commission``.
/// step : float, optional
///     Put the leg's own distribution on a grid of this step (``aggregate``'s
///     ``bs``), scattering each value linearly to its two neighbours, which
///     keeps its mean. Only its marginal quantiles change; its mean, the
///     sums down the sheet and the scenario columns stay exact.
///     :meth:`Ledger.validation` compares the two means.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import Leg
/// >>> Leg("Loss", lambda x: x, kind="loss")
/// Leg("Loss")
#[pyclass(name = "Leg", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyLeg {
    label: String,
    value: Py<PyAny>,
    kind: Option<LegKind>,
    step: Option<f64>,
}

#[pymethods]
impl PyLeg {
    #[new]
    #[pyo3(signature = (label, value, kind=None, step=None))]
    fn new(
        label: String,
        value: Py<PyAny>,
        kind: Option<&str>,
        step: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            label,
            value,
            kind: kind_of(kind)?,
            step,
        })
    }

    /// The label.
    #[getter]
    fn label(&self) -> &str {
        &self.label
    }

    fn __repr__(&self) -> String {
        format!("Leg({:?})", self.label)
    }
}

impl PyLeg {
    fn evaluate(&self, py: Python<'_>, xs: &[f64]) -> PyResult<RsLeg> {
        let value = self.value.bind(py);
        let leg = if let Ok(c) = value.extract::<f64>() {
            RsLeg::constant(self.label.clone(), c)
        } else if value.is_callable() {
            let arg: Bound<'_, PyAny> = match py.import("numpy") {
                Ok(np) => np.call_method1("asarray", (xs.to_vec(),))?,
                Err(_) => xs.to_vec().into_pyobject(py)?.into_any(),
            };
            let out = value.call1((arg,))?;
            if let Ok(c) = out.extract::<f64>() {
                RsLeg::constant(self.label.clone(), c)
            } else {
                let out = if out.hasattr("tolist")? {
                    out.call_method0("tolist")?
                } else {
                    out
                };
                let v: Vec<f64> = out.extract().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "leg {:?}: the function must return a number or one number per value",
                        self.label
                    ))
                })?;
                RsLeg::values(self.label.clone(), v)
            }
        } else {
            return Err(PyTypeError::new_err(format!(
                "leg {:?}: the value must be a number or a function of the source's values",
                self.label
            )));
        };
        let leg = match self.kind {
            Some(k) => leg.kind(k),
            None => leg,
        };
        Ok(match self.step {
            Some(h) => leg.step(h),
            None => leg,
        })
    }
}

/// A book of the ledger: ``sell`` (the insurer of the subject business,
/// booking ``+consideration - obligation``) or ``buy`` (a cession, booking
/// the contra: ceded premium paid, recoveries and commission received).
///
/// Parameters
/// ----------
/// label : str
/// role : str
///     ``sell`` or ``buy``.
/// consideration : list of Leg, optional
/// obligation : list of Leg, optional
/// margin_label : str, optional
///     The label of a seller's own margin on a ledger that also buys
///     cover; ``Gross`` by default.
#[pyclass(name = "Group", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyGroup {
    label: String,
    role: Role,
    consideration: Vec<Py<PyLeg>>,
    obligation: Vec<Py<PyLeg>>,
    margin_label: Option<String>,
}

#[pymethods]
impl PyGroup {
    #[new]
    #[pyo3(signature = (label, role, consideration=None, obligation=None, margin_label=None))]
    fn new(
        label: String,
        role: &str,
        consideration: Option<Vec<Py<PyLeg>>>,
        obligation: Option<Vec<Py<PyLeg>>>,
        margin_label: Option<String>,
    ) -> PyResult<Self> {
        let role = match role {
            "sell" => Role::Sell,
            "buy" => Role::Buy,
            other => {
                return Err(PyValueError::new_err(format!(
                    "a group's role is sell or buy, not {other:?}"
                )));
            }
        };
        Ok(Self {
            label,
            role,
            consideration: consideration.unwrap_or_default(),
            obligation: obligation.unwrap_or_default(),
            margin_label,
        })
    }

    /// The label.
    #[getter]
    fn label(&self) -> &str {
        &self.label
    }

    fn __repr__(&self) -> String {
        format!(
            "Group({:?}, {:?}, {} consideration, {} obligation)",
            self.label,
            if self.role == Role::Sell {
                "sell"
            } else {
                "buy"
            },
            self.consideration.len(),
            self.obligation.len()
        )
    }
}

/// A source: a :class:`Grid`, a tuple ``(values, probs)``, or a joint
/// source ``(gross, ceded, probs)``.
fn source_of(source: &Bound<'_, PyAny>) -> PyResult<Source> {
    Ok(if let Ok(g) = source.extract::<PyRef<'_, PyGrid>>() {
        Source::from_grid(&g.inner).map_err(to_py)?
    } else if let Ok(t) = source.cast::<PyTuple>() {
        let col = |o: Bound<'_, PyAny>| -> PyResult<Vec<f64>> {
            if o.hasattr("tolist")? {
                o.call_method0("tolist")?.extract()
            } else {
                o.extract()
            }
        };
        match t.len() {
            2 => Source::new(col(t.get_item(0)?)?, col(t.get_item(1)?)?).map_err(to_py)?,
            3 => Source::joint(
                col(t.get_item(0)?)?,
                col(t.get_item(1)?)?,
                col(t.get_item(2)?)?,
            )
            .map_err(to_py)?,
            _ => {
                return Err(PyTypeError::new_err(
                    "a source tuple is (values, probs) or a joint (gross, ceded, probs)",
                ));
            }
        }
    } else {
        return Err(PyTypeError::new_err(
            "the source is a Grid, a tuple (values, probs) or a joint (gross, ceded, probs)",
        ));
    })
}

fn ladder_label(prefix: &str, q: f64) -> String {
    format!("{prefix}{:02.0}", q * 100.0)
}

/// A P&L ledger, as ``aggregate``'s ``PnL``: a source (a :class:`Grid`, or
/// values and their probabilities) and its groups, with every row (each
/// leg, each side's total, each group's result, running nets, tier and
/// net-of blocks, and with several groups the grand rows and the total
/// impact) a sum of signed legs at each atom, and its exact distribution.
///
/// The scenario columns ``k01`` to ``k99`` (``aggregate``'s ``κ``) are each
/// row's conditional mean given that the grand result equals its quantile
/// at that level, so each column is one state of the world and foots down
/// the sheet. Rows are in the payoff convention: ``k01`` is the bad year.
/// The source's probabilities are rescaled to sum to 1.
///
/// Parameters
/// ----------
/// source : Grid or tuple of (list of float, list of float)
/// groups : list of Group
/// name : str, default "ledger"
/// result_name : str, default "result"
/// spans : list of tuple of (str, int, int), optional
///     ``(label, lo, hi)``: groups ``lo`` to ``hi - 1`` with their own
///     subtotal block, and a net-of block when a later group follows.
/// force_tower : bool, default False
///     Present one group as a tower, with a step level.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import Group, Ledger, Leg
/// >>> book = Group("Book", "sell", [Leg("Premium", 10.0)], [Leg("Loss", lambda x: x)])
/// >>> pnl = Ledger(([2.0, 8.0], [0.5, 0.5]), [book], name="Book")
/// >>> pnl.result.mean()
/// 5.0
/// >>> s = pnl.sheet()
/// >>> s["label"], s["k01"]
/// (['Premium', 'Loss', 'Total'], [10.0, -8.0, 2.0])
#[pyclass(name = "Ledger", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyLedger {
    inner: RsLedger,
}

#[pymethods]
impl PyLedger {
    #[new]
    #[pyo3(signature = (source, groups, *, name="ledger".to_string(), result_name="result".to_string(), spans=None, force_tower=false))]
    fn new(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        groups: Vec<Py<PyGroup>>,
        name: String,
        result_name: String,
        spans: Option<Vec<(String, usize, usize)>>,
        force_tower: bool,
    ) -> PyResult<Self> {
        let source = source_of(source)?;
        let xs = source.values().to_vec();
        let mut rs_groups = Vec::with_capacity(groups.len());
        for g in &groups {
            let g = g.get();
            let mut rg = RsGroup::new(g.label.clone(), g.role);
            for leg in &g.consideration {
                rg = rg.consideration(leg.get().evaluate(py, &xs)?);
            }
            for leg in &g.obligation {
                rg = rg.obligation(leg.get().evaluate(py, &xs)?);
            }
            if let Some(m) = &g.margin_label {
                rg = rg.margin_label(m.clone());
            }
            rs_groups.push(rg);
        }
        let mut layout = Layout::new(name).result_name(result_name);
        for (label, lo, hi) in spans.unwrap_or_default() {
            layout = layout.span(label, lo, hi);
        }
        if force_tower {
            layout = layout.force_tower();
        }
        Ok(Self {
            inner: RsLedger::new(&source, rs_groups, layout).map_err(to_py)?,
        })
    }

    /// The ledger's name.
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// The rows' labels, in sheet order.
    #[getter]
    fn rows(&self) -> Vec<String> {
        self.inner.rows().iter().map(|r| r.label.clone()).collect()
    }

    /// The (grand) result's distribution.
    #[getter]
    fn result(&self) -> PyDiscrete {
        PyDiscrete {
            inner: self.inner.result().distribution().clone(),
        }
    }

    /// The distribution of the row labelled ``label``.
    ///
    /// Parameters
    /// ----------
    /// label : str
    ///
    /// Returns
    /// -------
    /// Discrete
    fn row(&self, label: &str) -> PyResult<PyDiscrete> {
        self.inner
            .row(label)
            .map(|r| PyDiscrete {
                inner: r.distribution().clone(),
            })
            .ok_or_else(|| PyValueError::new_err(format!("no row labelled {label:?}")))
    }

    /// The chance that the result is 0.
    #[getter]
    fn prob_eq_0(&self) -> f64 {
        self.inner.prob_eq_0()
    }

    /// The sheet, ``aggregate``'s ``economic_df``: every row with its mean,
    /// standard deviation, coefficient of variation, skewness and scenario
    /// columns ``k01`` to ``k99``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Columns ``step`` (on a tower), ``side``, ``label``, ``row`` (the
    ///     row's key), ``EX``, ``SD``, ``CV``, ``Skew`` and the scenarios.
    fn sheet<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = self.index(py)?;
        let rows = self.inner.rows();
        d.set_item(
            "EX",
            rows.iter().map(|r| snap(r.mean())).collect::<Vec<_>>(),
        )?;
        d.set_item("SD", rows.iter().map(|r| snap(r.sd())).collect::<Vec<_>>())?;
        d.set_item(
            "CV",
            rows.iter().map(|r| r.cv().map(snap)).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "Skew",
            rows.iter().map(|r| r.skew().map(snap)).collect::<Vec<_>>(),
        )?;
        for q in LADDER {
            d.set_item(
                ladder_label("k", q),
                self.inner.scenarios(q).map_err(to_py)?,
            )?;
        }
        Ok(d)
    }

    /// Each row's own quantiles, ``aggregate``'s ``economic_marginal_df``:
    /// ``P01`` to ``P99``, which do not add down the sheet.
    ///
    /// Returns
    /// -------
    /// dict
    fn marginal<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = self.index(py)?;
        for q in LADDER {
            let col: PyResult<Vec<f64>> = self
                .inner
                .rows()
                .iter()
                .map(|r| r.quantile(q).map(snap).map_err(to_py))
                .collect();
            d.set_item(ladder_label("P", q), col?)?;
        }
        Ok(d)
    }

    /// The card, ``aggregate``'s ``summary_df``: each block's consideration,
    /// obligation and margin (and on a tower each step's running net and
    /// the covers' total impact), with ``EX``, ``SD``, ``CV``, ``Skew``,
    /// ``P01``, ``Median`` and ``P99`` of each row's own distribution.
    ///
    /// Returns
    /// -------
    /// dict
    fn card<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let card = self.inner.card().map_err(to_py)?;
        let d = PyDict::new(py);
        if self.inner.is_tower() {
            d.set_item(
                "step",
                card.iter().map(|r| r.step.clone()).collect::<Vec<_>>(),
            )?;
        }
        d.set_item(
            "side",
            card.iter().map(|r| r.side.clone()).collect::<Vec<_>>(),
        )?;
        d.set_item("EX", card.iter().map(|r| r.ex).collect::<Vec<_>>())?;
        d.set_item("SD", card.iter().map(|r| r.sd).collect::<Vec<_>>())?;
        d.set_item("CV", card.iter().map(|r| r.cv).collect::<Vec<_>>())?;
        d.set_item("Skew", card.iter().map(|r| r.skew).collect::<Vec<_>>())?;
        d.set_item("P01", card.iter().map(|r| r.p01).collect::<Vec<_>>())?;
        d.set_item("Median", card.iter().map(|r| r.median).collect::<Vec<_>>())?;
        d.set_item("P99", card.iter().map(|r| r.p99).collect::<Vec<_>>())?;
        Ok(d)
    }

    /// The declared legs, ``aggregate``'s ``legs_df``: step, side, label,
    /// kind, ``EX`` and ``SD``.
    ///
    /// Returns
    /// -------
    /// dict
    fn legs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let legs: Vec<&Row> = self
            .inner
            .rows()
            .iter()
            .filter(|r| matches!(r.kind, RowKind::Leg { .. }))
            .collect();
        let d = PyDict::new(py);
        d.set_item(
            "step",
            legs.iter().map(|r| r.step.clone()).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "side",
            legs.iter().map(|r| r.side.name()).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "label",
            legs.iter().map(|r| r.label.clone()).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "kind",
            legs.iter()
                .map(|r| r.leg_kind.map(|k| k.name()))
                .collect::<Vec<_>>(),
        )?;
        d.set_item("EX", legs.iter().map(|r| r.mean()).collect::<Vec<_>>())?;
        d.set_item("SD", legs.iter().map(|r| snap(r.sd())).collect::<Vec<_>>())?;
        Ok(d)
    }

    /// Each rebucketed leg's exact mean against its grid's,
    /// ``aggregate``'s ``validation_df``. Empty when no leg has a ``step``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Columns ``label``, ``EX``, ``Est``, ``abs_err`` and ``rel_err``.
    fn validation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let v = self.inner.validation();
        let d = PyDict::new(py);
        d.set_item(
            "label",
            v.iter().map(|r| r.label.clone()).collect::<Vec<_>>(),
        )?;
        d.set_item("EX", v.iter().map(|r| r.ex).collect::<Vec<_>>())?;
        d.set_item("Est", v.iter().map(|r| r.est).collect::<Vec<_>>())?;
        d.set_item("abs_err", v.iter().map(|r| r.abs_err).collect::<Vec<_>>())?;
        d.set_item("rel_err", v.iter().map(|r| r.rel_err).collect::<Vec<_>>())?;
        Ok(d)
    }

    /// The ratio exhibit, ``aggregate``'s ``economic_ratios_df``: for each
    /// group, span, net-of block and ``All``, premium ``P``, loss ``L`` net
    /// of recoveries, expense ``E`` net of commission and the result ``M``
    /// (mean amounts, signed for the block's own view), the result's ``SD``,
    /// the ratios of means ``LR``, ``ER``, ``CR``, the means of ratios
    /// ``E_LR``, ``E_ER``, ``E_CR`` (where the premium never vanishes), and
    /// ``P_share`` and ``M_share`` of the first block. Legs without a kind
    /// count as premium on the consideration side and loss on the other.
    ///
    /// Returns
    /// -------
    /// dict
    fn ratios<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = self.inner.ratios();
        let d = PyDict::new(py);
        d.set_item("step", r.iter().map(|r| r.step.clone()).collect::<Vec<_>>())?;
        d.set_item("P", r.iter().map(|r| r.p).collect::<Vec<_>>())?;
        d.set_item("L", r.iter().map(|r| r.l).collect::<Vec<_>>())?;
        d.set_item("E", r.iter().map(|r| r.e).collect::<Vec<_>>())?;
        d.set_item("M", r.iter().map(|r| r.m).collect::<Vec<_>>())?;
        d.set_item("SD", r.iter().map(|r| r.sd).collect::<Vec<_>>())?;
        d.set_item("LR", r.iter().map(|r| r.lr).collect::<Vec<_>>())?;
        d.set_item("ER", r.iter().map(|r| r.er).collect::<Vec<_>>())?;
        d.set_item("CR", r.iter().map(|r| r.cr).collect::<Vec<_>>())?;
        d.set_item("E_LR", r.iter().map(|r| r.e_lr).collect::<Vec<_>>())?;
        d.set_item("E_ER", r.iter().map(|r| r.e_er).collect::<Vec<_>>())?;
        d.set_item("E_CR", r.iter().map(|r| r.e_cr).collect::<Vec<_>>())?;
        d.set_item("P_share", r.iter().map(|r| r.p_share).collect::<Vec<_>>())?;
        d.set_item("M_share", r.iter().map(|r| r.m_share).collect::<Vec<_>>())?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        let r = self.inner.result();
        format!(
            "Ledger({:?}, {} rows, result mean {}, sd {})",
            self.inner.name(),
            self.inner.rows().len(),
            r.mean(),
            r.sd()
        )
    }
}

impl PyLedger {
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let rows = self.inner.rows();
        let d = PyDict::new(py);
        if self.inner.is_tower() {
            d.set_item(
                "step",
                rows.iter().map(|r| r.step.clone()).collect::<Vec<_>>(),
            )?;
        }
        d.set_item(
            "side",
            rows.iter().map(|r| r.side.name()).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "label",
            rows.iter().map(|r| r.display.clone()).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "row",
            rows.iter().map(|r| r.label.clone()).collect::<Vec<_>>(),
        )?;
        Ok(d)
    }
}

/// Stack ledgers' results, ``aggregate``'s ``stack_marginal_pnls``: one row
/// per ledger with the mean, standard deviation, coefficient of variation,
/// skewness and quantiles ``P01`` to ``P99`` of its result, then one row per
/// impact, the difference of two stacked rows statistic by statistic.
///
/// Parameters
/// ----------
/// ledgers : dict of str to Ledger
///     In row order.
/// impacts : list of tuple of (str, str, str), optional
///     ``(label, target, base)``: the row ``target`` less the row ``base``.
///
/// Returns
/// -------
/// dict
///     Columns ``label``, ``EX``, ``SD``, ``CV``, ``Skew`` and ``P01`` to
///     ``P99``.
///
/// Raises
/// ------
/// ValueError
///     If an impact names a row that is not stacked.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import Group, Ledger, Leg, stack
/// >>> src = ([2.0, 8.0], [0.5, 0.5])
/// >>> a = Ledger(src, [Group("A", "sell", [Leg("P", 10.0)], [Leg("L", lambda x: x)])])
/// >>> b = Ledger(src, [Group("B", "sell", [Leg("P", 12.0)], [Leg("L", lambda x: x)])])
/// >>> stack({"a": a, "b": b}, [("b less a", "b", "a")])["EX"]
/// [5.0, 7.0, 2.0]
#[pyfunction]
#[pyo3(signature = (ledgers, impacts=None))]
pub(crate) fn stack<'py>(
    py: Python<'py>,
    ledgers: &Bound<'py, PyDict>,
    impacts: Option<Vec<(String, String, String)>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut held: Vec<(String, Py<PyLedger>)> = Vec::with_capacity(ledgers.len());
    for (k, v) in ledgers.iter() {
        held.push((k.extract()?, v.extract()?));
    }
    let refs: Vec<(&str, &RsLedger)> = held
        .iter()
        .map(|(k, l)| (k.as_str(), &l.get().inner))
        .collect();
    let impacts = impacts.unwrap_or_default();
    let imp: Vec<(&str, &str, &str)> = impacts
        .iter()
        .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
        .collect();
    let rows = rs_stack(&refs, &imp).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item(
        "label",
        rows.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>(),
    )?;
    let col = |f: fn(&Stats) -> Option<f64>| rows.iter().map(|(_, s)| f(s)).collect::<Vec<_>>();
    d.set_item("EX", col(|s| Some(s.ex)))?;
    d.set_item("SD", col(|s| Some(s.sd)))?;
    d.set_item("CV", col(|s| s.cv))?;
    d.set_item("Skew", col(|s| s.skew))?;
    for (k, q) in LADDER.iter().enumerate() {
        d.set_item(
            ladder_label("P", *q),
            rows.iter().map(|(_, s)| s.ladder[k]).collect::<Vec<_>>(),
        )?;
    }
    Ok(d)
}

fn basis_of(name: &str) -> PyResult<ExpenseBasis> {
    match name {
        "fixed" => Ok(ExpenseBasis::Fixed),
        "premium" => Ok(ExpenseBasis::Premium),
        "loss" => Ok(ExpenseBasis::Loss),
        other => Err(PyValueError::new_err(format!(
            "an expense basis is fixed, premium or loss, not {other:?}"
        ))),
    }
}

/// One expense leg of a :class:`Book`: an amount (``fixed``), a fraction of
/// the gross premium (``premium``) or of the gross loss (``loss``, loss
/// adjustment expense), or several such terms in one leg.
///
/// Unlabelled, a book's only expense is the leg ``expense``; with several,
/// a single-basis one is ``<basis> expense`` and a mixed one ``expense``,
/// with ``(2)``, ``(3)`` … added to a name already taken, as ``aggregate``.
///
/// Parameters
/// ----------
/// basis : str or list of tuple of (str, float)
///     ``fixed``, ``premium`` or ``loss``; or a list of ``(basis, value)``
///     terms.
/// value : float, optional
///     The amount or rate, with a single ``basis``.
/// label : str, optional
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import Expense
/// >>> Expense("premium", 0.25, label="Acq")
/// Expense("Acq", [("premium", 0.25)])
#[pyclass(
    name = "Expense",
    module = "prospicio.aggregate",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyExpense {
    inner: RsExpense,
}

#[pymethods]
impl PyExpense {
    #[new]
    #[pyo3(signature = (basis, value=None, *, label=None))]
    fn new(basis: &Bound<'_, PyAny>, value: Option<f64>, label: Option<String>) -> PyResult<Self> {
        let terms: Vec<(String, f64)> = if let Ok(b) = basis.extract::<String>() {
            let v = value
                .ok_or_else(|| PyTypeError::new_err("an expense with one basis needs a value"))?;
            vec![(b, v)]
        } else {
            if value.is_some() {
                return Err(PyTypeError::new_err(
                    "give either a basis and a value, or a list of (basis, value) terms",
                ));
            }
            basis.extract()?
        };
        if terms.is_empty() {
            return Err(PyValueError::new_err("an expense needs at least one term"));
        }
        let terms = terms
            .into_iter()
            .map(|(b, v)| Ok((basis_of(&b)?, v)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: RsExpense { label, terms },
        })
    }

    /// The label, or None.
    #[getter]
    fn label(&self) -> Option<String> {
        self.inner.label.clone()
    }

    /// The ``(basis, value)`` terms.
    #[getter]
    fn terms(&self) -> Vec<(&'static str, f64)> {
        self.inner
            .terms
            .iter()
            .map(|(b, v)| (b.name(), *v))
            .collect()
    }

    fn __repr__(&self) -> String {
        let terms: Vec<String> = self
            .inner
            .terms
            .iter()
            .map(|(b, v)| format!("({:?}, {v})", b.name()))
            .collect();
        match &self.inner.label {
            Some(l) => format!("Expense({l:?}, [{}])", terms.join(", ")),
            None => format!("Expense([{}])", terms.join(", ")),
        }
    }
}

/// The gross premium whose own expenses leave the technical premium
/// ``technical``, ``aggregate``'s ``derive premium``: ``(T + F) / (1 - r)``
/// with ``F`` the fixed amounts and ``r`` the premium rates.
///
/// Parameters
/// ----------
/// technical : float
/// expenses : list of Expense
///
/// Returns
/// -------
/// float
///
/// Raises
/// ------
/// ValueError
///     On a loss-basis term, or premium rates totalling 1 or more.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import Expense, derive_premium
/// >>> round(derive_premium(5.0, [Expense("premium", 0.1), Expense("fixed", 1.0)]), 12)
/// 6.666666666667
#[pyfunction]
pub(crate) fn derive_premium(technical: f64, expenses: Vec<PyExpense>) -> PyResult<f64> {
    let e: Vec<RsExpense> = expenses.into_iter().map(|e| e.inner).collect();
    rs_derive_premium(technical, &e).map_err(to_py)
}

/// A seller's book, ``aggregate``'s ``pnl`` and ``xpnl`` builders: a gross
/// premium, expenses and the labels of its legs, laid out over a source of
/// annual gross losses with the annual covers of a tower.
///
/// :meth:`ledger` gives one of three faces:
///
/// - ``consolidated`` (``pnl``): one group, the net position. With a tower
///   it books net premium (gross less ceded premiums) less the net loss
///   less the commission received less expenses, over the net loss; a
///   loss-basis expense books as its rate times the expected gross loss.
/// - ``walk`` (``xpnl``): the gross book, then one buy group per stage of
///   the tower, with the running net and the covers' total impact.
/// - ``peel-top-down`` and ``peel-bottom-up`` (``xpnl ... peel``): one buy
///   group per layer, with an ``All aggregate`` subtotal.
///
/// Every layer's terms are taken at the annual total. The builders book a
/// fixed premium and a flat ceding commission; a layer with loss-sensitive
/// terms is refused (write its legs with :class:`Leg`).
///
/// Parameters
/// ----------
/// name : str
///     The ledger's name.
/// premium : float
///     The gross premium.
/// expenses : list of Expense, optional
/// premium_label : str, optional
///     ``Premium`` by default.
/// loss_label : str, optional
///     ``Loss`` by default; also the gross book's margin on a walk.
/// step_label : str, optional
///     The gross book's step on a walk, ``Gross`` by default.
/// retro : prospicio.reinsurance.LossSensitivePremium, optional
///     Rate the book retrospectively (``aggregate``'s ``retro <collar>
///     premium``): the premium is these terms at the year's gross loss,
///     and ``premium`` (the collar's basic premium in ``aggregate``) is
///     what premium-basis expenses are taken on. A retro book buys no
///     cover here.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import Book, Expense
/// >>> from prospicio.reinsurance import Layer, Tower
/// >>> gross = ([2, 3, 4, 5, 6, 8], [0.25, 0.3, 0.09, 0.2, 0.12, 0.04])
/// >>> cover = Layer.stop_loss("agg 3 xs 5", 3.0, 5.0).with_deposit_premium(1.0)
/// >>> cover = cover.with_ceding_commission(0.1)
/// >>> book = Book("Ex3", 10.0, [Expense("premium", 0.1)])
/// >>> walk = book.ledger(gross, Tower([cover]), "walk")
/// >>> round(walk.result.mean(), 12), walk.sheet()["step"][-1]
/// (4.54, 'All')
#[pyclass(name = "Book", module = "prospicio.aggregate", frozen)]
pub(crate) struct PyBook {
    inner: RsBook,
}

#[pymethods]
impl PyBook {
    #[new]
    #[pyo3(signature = (name, premium, expenses=None, *, premium_label=None, loss_label=None, step_label=None, retro=None))]
    fn new(
        name: String,
        premium: f64,
        expenses: Option<Vec<PyExpense>>,
        premium_label: Option<String>,
        loss_label: Option<String>,
        step_label: Option<String>,
        retro: Option<PyRef<'_, PyLossSensitivePremium>>,
    ) -> Self {
        let mut b = RsBook::new(name, premium);
        if let Some(r) = retro {
            b = b.retro(r.inner);
        }
        for e in expenses.unwrap_or_default() {
            b = b.expense(e.inner);
        }
        if let Some(l) = premium_label {
            b = b.premium_label(l);
        }
        if let Some(l) = loss_label {
            b = b.loss_label(l);
        }
        if let Some(l) = step_label {
            b = b.step_label(l);
        }
        Self { inner: b }
    }

    /// The gross premium.
    #[getter]
    fn premium(&self) -> f64 {
        self.inner.gross_premium()
    }

    /// The ledger over the annual gross losses ``source``.
    ///
    /// Parameters
    /// ----------
    /// source : Grid or tuple of (list of float, list of float)
    /// tower : prospicio.reinsurance.Tower, optional
    ///     Annual covers; without one, the plain book.
    /// face : str, default "consolidated"
    ///     ``consolidated``, ``walk``, ``peel-top-down`` or
    ///     ``peel-bottom-up``.
    ///
    /// Returns
    /// -------
    /// Ledger
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A peel without a tower, a layer that needs each event, or one
    ///     with loss-sensitive terms.
    #[pyo3(signature = (source, tower=None, face="consolidated"))]
    fn ledger(
        &self,
        source: &Bound<'_, PyAny>,
        tower: Option<PyRef<'_, PyTower>>,
        face: &str,
    ) -> PyResult<PyLedger> {
        let face = face_of(face)?;
        let source = source_of(source)?;
        Ok(PyLedger {
            inner: self
                .inner
                .ledger(&source, tower.as_deref().map(|t| &t.inner), face)
                .map_err(to_py)?,
        })
    }

    /// The ledger with occurrence covers over the joint source ``joint``
    /// of the annual gross loss and the loss the ``occurrence`` tower cedes
    /// (:func:`occurrence_source`), then the aggregate covers of
    /// ``aggregate`` on the net of the occurrence covers, as
    /// ``aggregate``'s ``xpnl`` and ``pnl`` over its ``occ_bivariate``.
    ///
    /// The occurrence covers are one group on the walk, named after the
    /// first layer; a peel needs exactly one occurrence layer. The
    /// consolidated face is over the net loss at each atom.
    ///
    /// Parameters
    /// ----------
    /// joint : tuple of (list of float, list of float, list of float)
    ///     ``(gross, ceded, probs)``.
    /// occurrence : prospicio.reinsurance.Tower
    ///     Per-occurrence covers without annual terms.
    /// aggregate : prospicio.reinsurance.Tower, optional
    /// face : str, default "consolidated"
    ///
    /// Returns
    /// -------
    /// Ledger
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As :meth:`ledger`, an occurrence layer with annual terms, or a
    ///     peel through several occurrence layers.
    #[pyo3(signature = (joint, occurrence, aggregate=None, face="consolidated"))]
    fn occurrence_ledger(
        &self,
        joint: &Bound<'_, PyAny>,
        occurrence: PyRef<'_, PyTower>,
        aggregate: Option<PyRef<'_, PyTower>>,
        face: &str,
    ) -> PyResult<PyLedger> {
        let face = face_of(face)?;
        let joint = source_of(joint)?;
        Ok(PyLedger {
            inner: self
                .inner
                .occurrence_ledger(
                    &joint,
                    &occurrence.inner,
                    aggregate.as_deref().map(|t| &t.inner),
                    face,
                )
                .map_err(to_py)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("Book(premium={})", self.inner.gross_premium())
    }
}

/// A joint source's columns: gross, ceded, probabilities.
type JointColumns = (Vec<f64>, Vec<f64>, Vec<f64>);

fn face_of(face: &str) -> PyResult<Face> {
    match face {
        "consolidated" => Ok(Face::Consolidated),
        "walk" => Ok(Face::Walk),
        "peel-top-down" => Ok(Face::PeelTopDown),
        "peel-bottom-up" => Ok(Face::PeelBottomUp),
        other => Err(PyValueError::new_err(format!(
            "a face is consolidated, walk, peel-top-down or peel-bottom-up, not {other:?}"
        ))),
    }
}

/// The joint source of the annual gross loss and the loss ceded by the
/// per-occurrence covers of ``occurrence``, ``aggregate``'s
/// ``occ_bivariate(views=('gross', 'ceded'))``: each claim on the
/// severity's grid goes to ``(x, c(x))``, the cession split linearly
/// between grid points, compounded by a two-dimensional FFT on ``points``
/// points per axis; cells below ``1e-15`` of the total mass are dropped as
/// round-off.
///
/// Parameters
/// ----------
/// frequency : Poisson, NegativeBinomial, Binomial or a count
/// severity : Grid
/// occurrence : prospicio.reinsurance.Tower
///     Per-occurrence covers without annual terms.
/// points : tuple of (int, int)
///
/// Returns
/// -------
/// tuple of (tuple of (list of float, list of float, list of float), BivariateReport)
///     The joint source ``(gross, ceded, probs)`` for
///     :meth:`Book.occurrence_ledger`, and the FFT's report.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import occurrence_source
/// >>> from prospicio.distributions import Grid, Poisson
/// >>> from prospicio.reinsurance import Layer, Tower
/// >>> sev = Grid(1.0, [0.0, 0.5, 0.3, 0.0, 0.2])
/// >>> cover = Tower([Layer("occ 2 xs 2", 2.0, 2.0)])
/// >>> (gross, ceded, probs), _ = occurrence_source(Poisson(2.0), sev, cover, (64, 32))
/// >>> round(sum(c * p for c, p in zip(ceded, probs)), 12)
/// 0.8
#[pyfunction]
pub(crate) fn occurrence_source(
    py: Python<'_>,
    frequency: &Bound<'_, PyAny>,
    severity: PyRef<'_, PyGrid>,
    occurrence: PyRef<'_, PyTower>,
    points: (usize, usize),
) -> PyResult<(JointColumns, PyBivariateReport)> {
    let n = AnyCount::extract(frequency)?;
    let (sev, tower) = (severity.inner.clone(), occurrence.inner.clone());
    let (source, report) = py
        .detach(|| {
            prospicio_aggregate::pnl::occurrence_source(
                n.as_counting(),
                &sev,
                &tower,
                [points.0, points.1],
            )
        })
        .map_err(to_py)?;
    Ok((
        (
            source.values().to_vec(),
            source.second().unwrap_or_default().to_vec(),
            source.probs().to_vec(),
        ),
        PyBivariateReport { inner: report },
    ))
}

/// A premium rounded as ``aggregate`` writes one: to whole units above 100,
/// to cents at or below it, ties to even.
///
/// Parameters
/// ----------
/// premium : float
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import round_premium
/// >>> round_premium(714.12), round_premium(16.285)
/// (714.0, 16.29)
#[pyfunction]
pub(crate) fn round_premium(premium: f64) -> f64 {
    rs_round_premium(premium)
}

/// The premium sized from a target loss ratio, ``aggregate``'s
/// ``pnl_program`` without the ladder: the expected loss over
/// ``loss_ratio``, rounded by :func:`round_premium`.
///
/// Parameters
/// ----------
/// expected_loss : float
/// loss_ratio : float
///
/// Returns
/// -------
/// float
///
/// Raises
/// ------
/// ValueError
///     If either is not positive and finite.
#[pyfunction]
pub(crate) fn size_premium(expected_loss: f64, loss_ratio: f64) -> PyResult<f64> {
    rs_size_premium(expected_loss, loss_ratio).map_err(to_py)
}

/// The premium ladder of ``aggregate``'s ``pnl_program``, as numbers:
/// the net technical premium ``E_net / c_net``, each layer's deposit at
/// 100% placement ``round(E_j / c_j / s_j)`` (or the premium it states),
/// and the booked premium ``round((T_net + sum s_j d_j) / (1 - ER))``. The
/// book's own margin is ``(1 - c_net) T_net`` whatever the cover costs.
///
/// Parameters
/// ----------
/// net_loss : float
///     Expected net loss.
/// net_combined_ratio : float
/// expense_ratio : float
///     Gross expenses as a fraction of premium.
/// layers : list of dict, optional
///     Each with ``ceded_loss`` (expected, at the placed share), ``share``,
///     ``combined_ratio`` and optionally ``stated`` (a premium the contract
///     states, at the placed share, which is kept).
///
/// Returns
/// -------
/// dict
///     ``premium``, ``technical``, ``deposits`` and ``ceded_premium``.
///
/// Raises
/// ------
/// ValueError
///     A ratio that is not positive, an expense ratio outside ``[0, 1)``
///     or a share outside ``(0, 1]``.
///
/// Examples
/// --------
/// >>> from prospicio.aggregate import pnl_program
/// >>> p = pnl_program(406.0, 0.9, 0.25, [dict(ceded_loss=86.0, share=1.0, combined_ratio=0.6)])
/// >>> p["premium"], p["deposits"]
/// (792.0, [143.0])
#[pyfunction]
#[pyo3(signature = (net_loss, net_combined_ratio, expense_ratio, layers=None))]
pub(crate) fn pnl_program<'py>(
    py: Python<'py>,
    net_loss: f64,
    net_combined_ratio: f64,
    expense_ratio: f64,
    layers: Option<Vec<Bound<'py, PyDict>>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut ls = Vec::new();
    for d in layers.unwrap_or_default() {
        let get = |k: &str| -> PyResult<f64> {
            d.get_item(k)?
                .ok_or_else(|| PyValueError::new_err(format!("a program layer needs {k:?}")))?
                .extract()
        };
        let stated = match d.get_item("stated")? {
            Some(v) if !v.is_none() => Some(v.extract::<f64>()?),
            _ => None,
        };
        ls.push(ProgramLayer {
            ceded_loss: if stated.is_some() {
                d.get_item("ceded_loss")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.0)
            } else {
                get("ceded_loss")?
            },
            share: get("share")?,
            combined_ratio: if stated.is_some() {
                d.get_item("combined_ratio")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(1.0)
            } else {
                get("combined_ratio")?
            },
            stated,
        });
    }
    let p = rs_pnl_program(net_loss, net_combined_ratio, expense_ratio, &ls).map_err(to_py)?;
    let out = PyDict::new(py);
    out.set_item("premium", p.premium)?;
    out.set_item("technical", p.technical)?;
    out.set_item("deposits", p.deposits)?;
    out.set_item("ceded_premium", p.ceded_premium)?;
    Ok(out)
}
