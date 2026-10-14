//! `prospicio.aggregate` (Aggregate lane): the P&L ledger, over
//! `prospicio_aggregate::pnl`.

use prospicio_aggregate::pnl::{
    Group as RsGroup, LADDER, Layout, Ledger as RsLedger, Leg as RsLeg, LegKind, Role, Row,
    RowKind, Source, snap,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::discrete::PyDiscrete;
use crate::distributions::PyGrid;
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
}

#[pymethods]
impl PyLeg {
    #[new]
    #[pyo3(signature = (label, value, kind=None))]
    fn new(label: String, value: Py<PyAny>, kind: Option<&str>) -> PyResult<Self> {
        Ok(Self {
            label,
            value,
            kind: kind_of(kind)?,
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
        Ok(match self.kind {
            Some(k) => leg.kind(k),
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
        let source = if let Ok(g) = source.extract::<PyRef<'_, PyGrid>>() {
            Source::from_grid(&g.inner).map_err(to_py)?
        } else if let Ok(t) = source.cast::<PyTuple>() {
            if t.len() != 2 {
                return Err(PyTypeError::new_err("a source tuple is (values, probs)"));
            }
            let col = |o: Bound<'_, PyAny>| -> PyResult<Vec<f64>> {
                if o.hasattr("tolist")? {
                    o.call_method0("tolist")?.extract()
                } else {
                    o.extract()
                }
            };
            Source::new(col(t.get_item(0)?)?, col(t.get_item(1)?)?).map_err(to_py)?
        } else {
            return Err(PyTypeError::new_err(
                "the source is a Grid or a tuple (values, probs)",
            ));
        };
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
