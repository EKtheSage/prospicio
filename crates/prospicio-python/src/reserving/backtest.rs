//! The diagonal backtest, its candidates, and the calendar-year and development-correlation tests.

use super::*;

/// A generalized linear model of the cells of a triangle, for
/// ``diagonal_backtest``: the terms and the GLM, on the columns of the
/// backtest's cell table (``origin``, ``development`` and ``calendar``
/// factors; ``origin_index``, ``development_index``, ``calendar_index``,
/// ``age`` and ``exposure`` numbers; ``exposure`` is ``premium`` or 1).
/// The response is the incremental value of the column.
///
/// Parameters
/// ----------
/// terms : prospicio.models.Terms
///     The model's terms.
/// glm : prospicio.models.Glm
///     The family, link and dispersion.
/// name : str, default "glm"
///     The model's name in a ``diagonal_backtest`` given a list.
///
/// Examples
/// --------
/// >>> from prospicio.models import Glm, Terms
/// >>> from prospicio.reserving import GlmCandidate
/// >>> GlmCandidate.odp()
/// GlmCandidate(name='odp')
/// >>> GlmCandidate(Terms().intercept().factor("origin"), Glm("poisson", "log"), name="origin only").name
/// 'origin only'
#[pyclass(name = "GlmCandidate", module = "prospicio.reserving", frozen)]
pub(crate) struct PyGlmCandidate {
    inner: GlmCandidate,
}

#[pymethods]
impl PyGlmCandidate {
    #[new]
    #[pyo3(signature = (terms, glm, name = "glm"))]
    fn new(terms: PyRef<'_, PyTerms>, glm: PyRef<'_, PyGlm>, name: &str) -> Self {
        Self {
            inner: GlmCandidate {
                name: name.to_string(),
                terms: terms.inner.clone(),
                glm: glm.inner,
            },
        }
    }

    /// The over-dispersed Poisson GLM of the chain ladder: an intercept, an
    /// origin factor and a development factor, with the over-dispersed
    /// Poisson family. Its name is ``"odp"``.
    ///
    /// Returns
    /// -------
    /// GlmCandidate
    #[staticmethod]
    fn odp() -> Self {
        Self {
            inner: GlmCandidate {
                name: "odp".into(),
                terms: Terms::new()
                    .intercept()
                    .factor("origin")
                    .factor("development"),
                glm: Glm::over_dispersed_poisson(),
            },
        }
    }

    /// The name in the backtest.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// The model's terms.
    #[getter]
    fn terms(&self) -> PyTerms {
        PyTerms {
            inner: self.inner.terms.clone(),
        }
    }

    /// The model's family, link and dispersion.
    #[getter]
    fn glm(&self) -> PyGlm {
        PyGlm {
            inner: self.inner.glm,
        }
    }

    fn __repr__(&self) -> String {
        format!("GlmCandidate(name={})", quoted(&self.inner.name))
    }
}

/// `text` as a Python string literal in single quotes, for a `repr`.
pub(crate) fn quoted(text: &str) -> String {
    format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// `items` as a Python list of strings, for a `repr`.
fn quoted_list(items: &[String]) -> String {
    format!(
        "[{}]",
        items
            .iter()
            .map(|i| quoted(i))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// A model of a ``diagonal_backtest``, read from a Python object.
enum Candidate {
    Odp(BayesOdp),
    Meyers(Meyers),
    Glm(GlmCandidate),
    Compartmental(Box<CompartmentalCandidate>),
}

impl Candidate {
    fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(m) = obj.extract::<PyRef<'_, PyBayesOdp>>() {
            Ok(Self::Odp(m.inner.clone()))
        } else if let Ok(m) = obj.extract::<PyRef<'_, PyMeyers>>() {
            Ok(Self::Meyers(m.inner.clone()))
        } else if let Ok(m) = obj.extract::<PyRef<'_, PyGlmCandidate>>() {
            Ok(Self::Glm(m.inner.clone()))
        } else if let Ok(m) = obj.extract::<PyRef<'_, PyCompartmentalCandidate>>() {
            // A clone shares the fits the object keeps, so its
            // outstanding_backtest reuses what the backtest sampled.
            Ok(Self::Compartmental(Box::new(m.inner.clone())))
        } else {
            let kind = obj
                .get_type()
                .name()
                .map_or_else(|_| "?".into(), |n| n.to_string());
            Err(PyTypeError::new_err(format!(
                "a backtest model must be a BayesOdp, a Meyers, a GlmCandidate or a \
                 CompartmentalCandidate, got {kind}"
            )))
        }
    }

    fn default_name(&self) -> String {
        match self {
            Self::Odp(m) => m.name().to_string(),
            Self::Meyers(m) => m.name().to_string(),
            Self::Glm(m) => m.name.clone(),
            Self::Compartmental(m) => m.name().to_string(),
        }
    }

    fn into_named(self, name: String) -> NamedModel {
        match self {
            Self::Odp(m) => NamedModel::new(name, m),
            Self::Meyers(m) => NamedModel::new(name, m),
            Self::Glm(m) => NamedModel::new(name, m),
            Self::Compartmental(m) => NamedModel::new(name, *m),
        }
    }
}

/// The models of a backtest as named models: a dict's keys are the names,
/// a list's are the models' own names (``"bayes_odp"``, ``"meyers_crc"``,
/// ``"meyers_csr"``, ``"meyers_cay"``, a ``GlmCandidate``'s ``name``).
fn named_models(
    models: &Bound<'_, PyAny>,
) -> PyResult<(Vec<NamedModel>, bool, Vec<CompartmentalCandidate>)> {
    let mut named = Vec::new();
    let mut has_meyers = false;
    let mut compartmental = Vec::new();
    if let Ok(dict) = models.cast::<PyDict>() {
        for (key, value) in dict.iter() {
            let name: String = key
                .extract()
                .map_err(|_| PyTypeError::new_err("the keys of models must be model names"))?;
            let c = Candidate::from_py(&value)?;
            has_meyers |= matches!(c, Candidate::Meyers(_));
            if let Candidate::Compartmental(m) = &c {
                compartmental.push((**m).clone());
            }
            named.push(c.into_named(name));
        }
    } else {
        let iter = models.try_iter().map_err(|_| {
            PyTypeError::new_err("models must be a dict of name to model or a list of models")
        })?;
        for item in iter {
            let c = Candidate::from_py(&item?)?;
            has_meyers |= matches!(c, Candidate::Meyers(_));
            if let Candidate::Compartmental(m) = &c {
                compartmental.push((**m).clone());
            }
            let name = c.default_name();
            if named.iter().any(|m: &NamedModel| m.name == name) {
                return Err(PyValueError::new_err(format!(
                    "two models are named {}: pass a dict {{name: model}} to name each one",
                    quoted(&name)
                )));
            }
            named.push(c.into_named(name));
        }
    }
    if named.is_empty() {
        return Err(PyValueError::new_err("models must hold at least one model"));
    }
    Ok((named, has_meyers, compartmental))
}

/// The calendar period of each split of `bt`, as the frame labels it.
pub(crate) fn diagonal_labels(cells: &TriangleFrame, bt: &Backtest) -> Vec<String> {
    let labels = match cells.frame().column("calendar") {
        Some(Column::Categorical(c)) => Some(c),
        _ => None,
    };
    bt.scored_rows()
        .iter()
        .zip(bt.diagonals())
        .map(|(rows, &index)| match (labels, rows.first()) {
            (Some(c), Some(&r)) => c[r].clone(),
            _ => index.to_string(),
        })
        .collect()
}

/// A table as a pandas DataFrame, or as a dict of lists without pandas.
fn backtest_table<'py>(
    py: Python<'py>,
    columns: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    match py.import("pandas") {
        Ok(pandas) => pandas.call_method1("DataFrame", (&columns,)),
        Err(_) => Ok(columns.into_any()),
    }
}

/// Backtests models on the latest calendar diagonals of one measure column
/// of a single-segment triangle: for each of the last ``n_diagonals``
/// calendar periods, every model is fitted to the cells valued before it
/// and forecasts the cells valued in it, and the forecasts are scored
/// against what came in.
///
/// The table of cells has one row per origin and development cell, with
/// the incremental values of ``column`` as the response (a cumulative
/// triangle is differenced). The cells of each held-out diagonal are the
/// test set and the earlier ones the training set, so a model sees only
/// what was known. A held-out cell whose origin or development period has
/// no training cell is left out of the scoring of every model (see
/// ``excluded``).
///
/// A fit of a sampler (``BayesOdp``, ``Meyers``) whose diagnostics fail a
/// threshold is **still scored**: a study over many triangles wants every
/// split and a list of the doubtful ones, not an abort at the first. The
/// failed thresholds are in ``Backtest.warnings`` and
/// ``Backtest.diagnostics()``; a model that cannot be fitted at all raises
/// ``ValueError``.
///
/// Meyers' own statistic, the percentile of the outcome in the predictive
/// distribution of the ultimate, comes from a fit on the whole upper
/// triangle, not from a backtest: it is the ``cdf`` of the fit's
/// predictive total at the known outcome,
/// ``fit.ultimates.total().cdf(outcome)`` (``Sampled.cdf``). The
/// backtest's ``total_percentile`` is the same quantity for the held-out
/// diagonal's total, from the fit on the cells before it.
///
/// Parameters
/// ----------
/// triangle : Triangle
///     Cumulative, with one segment.
/// column : str
///     The measure column to forecast.
/// models : dict or list
///     A dict from name to model, or a list of models named by their
///     kind (``"bayes_odp"``, ``"meyers_crc"``, ``"meyers_csr"``,
///     ``"meyers_cay"``, ``"compartmental_gm1"``, ``"compartmental_gm2"``,
///     a ``GlmCandidate``'s ``name``); two models with one name are an
///     error. A model is a ``BayesOdp`` or a ``Meyers`` (the settings of
///     its own ``fit``, except that ``n_sims`` and ``seed`` are the
///     backtest's), a ``GlmCandidate`` or a ``CompartmentalCandidate``
///     (``Compartmental.candidate``, on this ``triangle`` and ``column``;
///     its ``outstanding_backtest`` scores the held-out outstanding on the
///     same fits).
/// n_diagonals : int
///     The number of latest calendar diagonals to hold out, oldest first.
/// premium : array-like, optional
///     The exposure of each origin, one value per origin. ``Meyers``
///     needs it; a ``GlmCandidate`` sees it as the ``exposure`` column.
/// n_sims : int, default 1000
///     Simulations per forecast.
/// seed : int, default 0
///     Seed of every fit and simulation; the same seed replays the scores.
/// interval : float, default 0.9
///     The central interval whose coverage is scored.
///
/// Returns
/// -------
/// Backtest
///
/// Raises
/// ------
/// ValueError
///     If ``models`` is empty or names two models alike; if ``Meyers`` is
///     given without ``premium``; if a ``CompartmentalCandidate`` was built
///     on another triangle or paid column than ``triangle`` and ``column``;
///     if the triangle has several segments or
///     lacks the column; if a diagonal cannot be held out; or if a model
///     cannot be fitted.
/// TypeError
///     If a model is not a ``BayesOdp``, a ``Meyers``, a
///     ``GlmCandidate`` or a ``CompartmentalCandidate``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import GlmCandidate, Triangle, diagonal_backtest
/// >>> paid = [[100.0, 150.0, 165.0, 170.0], [110.0, 170.0, 180.0],
/// ...         [120.0, 175.0], [130.0]]
/// >>> rows = [(2020 + w, 12 * (d + 1), v) for w, r in enumerate(paid)
/// ...         for d, v in enumerate(r)]
/// >>> origin, age, values = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(origin, age, {"paid": values})
/// >>> bt = diagonal_backtest(tri, "paid", [GlmCandidate.odp()], 1, n_sims=200)
/// >>> bt.models, bt.n_splits, bt.metrics[:2]
/// (['odp'], 1, ['crps', 'coverage'])
#[pyfunction]
#[pyo3(signature = (
    triangle,
    column,
    models,
    n_diagonals,
    *,
    premium = None,
    n_sims = 1000,
    seed = 0,
    interval = 0.9,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn diagonal_backtest(
    py: Python<'_>,
    triangle: PyRef<'_, PyTriangle>,
    column: &str,
    models: &Bound<'_, PyAny>,
    n_diagonals: usize,
    premium: Option<&Bound<'_, PyAny>>,
    n_sims: usize,
    seed: u64,
    interval: f64,
) -> PyResult<PyBacktest> {
    let (named, has_meyers, compartmental) = named_models(models)?;
    let premium = premium.map(|p| floats(p, "premium")).transpose()?;
    if has_meyers && premium.is_none() {
        return Err(PyValueError::new_err(
            "Meyers needs premium, one value per origin",
        ));
    }
    let tri = &triangle.inner;
    // A candidate reads its own triangle: it must be the backtest's.
    for c in &compartmental {
        if c.paid() != column || c.triangle() != tri {
            return Err(PyValueError::new_err(format!(
                "{} was built on another triangle or paid column than the backtest's: build \
                 it with Compartmental.candidate on this triangle and column {}",
                c.name(),
                quoted(column)
            )));
        }
    }
    let (inner, diagonals) = py
        .detach(|| {
            let cells = TriangleFrame::new(tri, column, premium.as_deref())?;
            let refs: Vec<&dyn TriangleModel> =
                named.iter().map(|m| m as &dyn TriangleModel).collect();
            let bt = run_backtest(&cells, &refs, n_diagonals, n_sims, seed, interval)?;
            let diagonals = diagonal_labels(&cells, &bt);
            Ok((bt, diagonals))
        })
        .map_err(err)?;
    Ok(PyBacktest { inner, diagonals })
}

/// The scores of a ``diagonal_backtest``: every model on every metric and
/// held-out diagonal, the diagnostics of each fit and the pointwise
/// held-out log densities.
///
/// The metrics (``metrics``), each scored per diagonal over the held-out
/// cells that every model fits:
///
/// - ``crps``: the mean over the cells of the continuous ranked
///   probability score of the forecast distribution at the cell's value.
/// - ``coverage``: the share of cells inside the central ``interval`` of
///   their forecast draws.
/// - ``actual_vs_expected``: the diagonal total over the forecast mean of
///   the diagonal total.
/// - ``total_crps``: the CRPS of the diagonal total from the draws summed
///   per simulation.
/// - ``mean_log_density``: the mean over the cells of the model's held-out
///   log predictive density (NaN for a model that gives none).
/// - ``total_percentile``: the share of the simulated diagonal totals at
///   or below the actual total, the probability integral transform of the
///   diagonal: uniform over many triangles for a calibrated model.
///
/// The tables (``scores()``, ``mean()``, ``diagnostics()``) are pandas
/// DataFrames when pandas is installed and dicts of lists otherwise.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import GlmCandidate, Triangle, diagonal_backtest
/// >>> paid = [[100.0, 150.0, 165.0, 170.0], [110.0, 170.0, 180.0],
/// ...         [120.0, 175.0], [130.0]]
/// >>> rows = [(2020 + w, 12 * (d + 1), v) for w, r in enumerate(paid)
/// ...         for d, v in enumerate(r)]
/// >>> origin, age, values = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(origin, age, {"paid": values})
/// >>> bt = diagonal_backtest(tri, "paid", [GlmCandidate.odp()], 1, n_sims=200)
/// >>> bt
/// Backtest(models=['odp'], diagonals=['2023'], metrics=6, warnings=0)
/// >>> bt.warnings
/// []
#[pyclass(name = "Backtest", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBacktest {
    pub(crate) inner: Backtest,
    pub(crate) diagonals: Vec<String>,
}

#[pymethods]
impl PyBacktest {
    /// Model names, in the order given.
    #[getter]
    fn models(&self) -> Vec<String> {
        self.inner.models().to_vec()
    }

    /// Metric names, in the order of ``scores()``: ``crps``,
    /// ``coverage``, ``actual_vs_expected``, ``total_crps``,
    /// ``mean_log_density`` and ``total_percentile``.
    #[getter]
    fn metrics(&self) -> Vec<String> {
        self.inner
            .metrics()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Number of held-out diagonals.
    #[getter]
    fn n_splits(&self) -> usize {
        self.inner.n_splits()
    }

    /// The calendar period held out in each split (labelled at the
    /// triangle's development grain, as the ``calendar`` column of the
    /// cells), oldest first.
    #[getter]
    fn diagonals(&self) -> Vec<String> {
        self.diagonals.clone()
    }

    /// Every score as a long table with one row per model, diagonal and
    /// metric: columns ``model``, ``diagonal``, ``metric`` and ``value``.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame or dict
    fn scores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let bt = &self.inner;
        let (mut model, mut diagonal, mut metric, mut value) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (m, name) in bt.models().iter().enumerate() {
            for (k, label) in self.diagonals.iter().enumerate() {
                for (j, metric_name) in bt.metrics().iter().enumerate() {
                    model.push(name.clone());
                    diagonal.push(label.clone());
                    metric.push((*metric_name).to_string());
                    value.push(bt.split_scores(m, j)[k]);
                }
            }
        }
        let out = PyDict::new(py);
        out.set_item("model", model)?;
        out.set_item("diagonal", diagonal)?;
        out.set_item("metric", metric)?;
        out.set_item("value", value)?;
        backtest_table(py, out)
    }

    /// The mean of each metric over the diagonals, one row per model:
    /// the column ``model`` and one column per metric. A metric that is NaN
    /// on a diagonal (``mean_log_density`` of a model with no density) is
    /// NaN in the mean.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame or dict
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let bt = &self.inner;
        let out = PyDict::new(py);
        out.set_item("model", bt.models().to_vec())?;
        for (j, metric) in bt.metrics().iter().enumerate() {
            let means: Vec<f64> = (0..bt.models().len()).map(|m| bt.mean(m, j)).collect();
            out.set_item(*metric, means)?;
        }
        backtest_table(py, out)
    }

    /// Per diagonal, the number of held-out cells left out of the scoring
    /// of every model because their origin or development period has no
    /// training cell that every model fits on.
    #[getter]
    fn excluded(&self) -> Vec<usize> {
        self.inner.excluded().to_vec()
    }

    /// Per diagonal, the rows of the cell table that were scored (origin
    /// by development, row-major, counted from 0; R's ``scored_rows``
    /// count from 1), in the order of
    /// ``log_densities``.
    #[getter]
    fn scored_rows(&self) -> Vec<Vec<usize>> {
        self.inner.scored_rows().to_vec()
    }

    /// Each model's held-out log predictive density of every scored row,
    /// diagonals in order, by model name; ``None`` for a model that gives
    /// none. Every model's list is for the same rows, so the lists are
    /// the input of the stacking functions.
    #[getter]
    fn log_densities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (name, lpd) in self.inner.models().iter().zip(self.inner.log_densities()) {
            out.set_item(name, lpd.clone())?;
        }
        Ok(out)
    }

    /// The sampler diagnostics of each fit, one row per model and
    /// diagonal: ``model``, ``diagonal``, ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``n_warnings``; a row for
    /// every model and diagonal, models in order. A model with no sampler
    /// (the GLM) has NaN in the three float columns and ``None`` in the
    /// two count columns (NaN in a DataFrame). A fit with warnings is
    /// scored like any other.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame or dict
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let bt = &self.inner;
        let (mut model, mut diagonal) = (Vec::new(), Vec::new());
        let (mut rhat, mut bulk, mut tail) = (Vec::new(), Vec::new(), Vec::new());
        let (mut divergences, mut n_warnings) = (Vec::new(), Vec::new());
        for (m, name) in bt.models().iter().enumerate() {
            for (k, d) in bt.diagnostics(m).iter().enumerate() {
                model.push(name.clone());
                diagonal.push(self.diagonals[k].clone());
                rhat.push(d.as_ref().map_or(f64::NAN, |d| d.max_rhat));
                bulk.push(d.as_ref().map_or(f64::NAN, |d| d.min_ess_bulk));
                tail.push(d.as_ref().map_or(f64::NAN, |d| d.min_ess_tail));
                divergences.push(d.as_ref().map(|d| d.divergences));
                n_warnings.push(d.as_ref().map(|d| d.warnings.len()));
            }
        }
        let out = PyDict::new(py);
        out.set_item("model", model)?;
        out.set_item("diagonal", diagonal)?;
        out.set_item("max_rhat", rhat)?;
        out.set_item("min_ess_bulk", bulk)?;
        out.set_item("min_ess_tail", tail)?;
        out.set_item("divergences", divergences)?;
        out.set_item("n_warnings", n_warnings)?;
        backtest_table(py, out)
    }

    /// Every failed diagnostic threshold as ``(model, diagonal,
    /// message)``, in model then diagonal order; empty when every fit
    /// passed.
    #[getter]
    fn warnings(&self) -> Vec<(String, String, String)> {
        self.inner
            .warnings()
            .into_iter()
            .map(|(m, k, message)| {
                (
                    self.inner.models()[m].clone(),
                    self.diagonals[k].clone(),
                    message,
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Backtest(models={}, diagonals={}, metrics={}, warnings={})",
            quoted_list(self.inner.models()),
            quoted_list(&self.diagonals),
            self.inner.metrics().len(),
            self.inner.warnings().len(),
        )
    }
}

/// Mack's test for calendar-year effects (Mack 1994, appendix H; R
/// ChainLadder's ``cyEffTest``), from ``calendar_year_effect`` or
/// ``MackFit.calendar_year_effect``.
///
/// At each age the link ratios are split at their median into small and
/// large ones (one equal to the median is in neither); each calendar
/// diagonal counts its small ones ``S_j`` and large ones ``L_j``, and
/// ``z``, the sum of ``min(S_j, L_j)``, is compared with the normal
/// interval ``expected ± q sqrt(variance)``, ``q`` the standard normal's
/// ``(1 + ci) / 2`` quantile. A diagonal where small (or large) factors
/// prevail lowers ``z``.
#[pyclass(name = "CalendarYearTest", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCalendarYearTest {
    pub(crate) inner: CalendarYearTest,
}

#[pymethods]
impl PyCalendarYearTest {
    /// ``Z``, the sum of each diagonal's ``min(S_j, L_j)``.
    #[getter]
    fn z(&self) -> f64 {
        self.inner.z
    }

    /// ``E[Z]`` under no calendar-year effect.
    #[getter]
    fn expected(&self) -> f64 {
        self.inner.expected
    }

    /// ``Var[Z]`` under no calendar-year effect.
    #[getter]
    fn variance(&self) -> f64 {
        self.inner.variance
    }

    /// The interval's confidence level.
    #[getter]
    fn ci(&self) -> f64 {
        self.inner.ci
    }

    /// The interval's lower end.
    #[getter]
    fn lower(&self) -> f64 {
        self.inner.lower
    }

    /// The interval's upper end.
    #[getter]
    fn upper(&self) -> f64 {
        self.inner.upper
    }

    /// Whether ``z`` lies within the interval (ends included): no
    /// calendar-year effect is detected. R prints the opposite as
    /// "Calendar Year Effect: TRUE".
    #[getter]
    fn passes(&self) -> bool {
        self.inner.passes()
    }

    /// Every calendar diagonal with at least two link ratios, oldest
    /// first, as a list of dicts with ``valuation`` (the year and month of
    /// its later cells, as ``"1983-12"``), ``small``, ``large``, ``z``,
    /// ``n`` (``small + large``), ``m`` (``(n - 1) // 2``), ``expected``
    /// and ``variance``: R's ``test_table``.
    #[getter]
    fn diagonals<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .diagonals
            .iter()
            .map(|d| {
                let row = PyDict::new(py);
                row.set_item("valuation", d.valuation.to_string())?;
                row.set_item("small", d.small)?;
                row.set_item("large", d.large)?;
                row.set_item("z", d.z)?;
                row.set_item("n", d.n)?;
                row.set_item("m", d.m)?;
                row.set_item("expected", d.expected)?;
                row.set_item("variance", d.variance)?;
                Ok(row)
            })
            .collect()
    }

    /// ``diagonals`` as a table, one row per diagonal. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let d = &self.inner.diagonals;
        let out = PyDict::new(py);
        let valuation: Vec<String> = d.iter().map(|d| d.valuation.to_string()).collect();
        out.set_item("valuation", valuation)?;
        out.set_item("small", d.iter().map(|d| d.small).collect::<Vec<_>>())?;
        out.set_item("large", d.iter().map(|d| d.large).collect::<Vec<_>>())?;
        out.set_item("z", d.iter().map(|d| d.z).collect::<Vec<_>>())?;
        out.set_item("n", d.iter().map(|d| d.n).collect::<Vec<_>>())?;
        out.set_item("m", d.iter().map(|d| d.m).collect::<Vec<_>>())?;
        out.set_item("expected", d.iter().map(|d| d.expected).collect::<Vec<_>>())?;
        out.set_item("variance", d.iter().map(|d| d.variance).collect::<Vec<_>>())?;
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    fn __repr__(&self) -> String {
        let t = &self.inner;
        format!(
            "CalendarYearTest(z={:?}, expected={:?}, variance={:?}, ci={:?}, lower={:?}, \
             upper={:?}, passes={})",
            t.z,
            t.expected,
            t.variance,
            t.ci,
            t.lower,
            t.upper,
            if t.passes() { "True" } else { "False" }
        )
    }
}

/// Mack's test for correlation between adjacent development factors
/// (Mack 1994, appendix G; R ChainLadder's ``dfCorTest``), from
/// ``development_correlation`` or ``MackFit.development_correlation``.
///
/// ``t`` is the weighted mean of the Spearman rank correlations of each
/// pair of adjacent link-ratio columns over the origins they share
/// (weight ``n - 1``), with variance ``1 / sum(weights)``, Mack's
/// ``1 / ((I - 2)(I - 3) / 2)`` on a full triangle, compared with
/// ``± q sqrt(variance)``, ``q`` the standard normal's ``(1 + ci) / 2``
/// quantile.
#[pyclass(
    name = "DevelopmentCorrelationTest",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyDevelopmentCorrelationTest {
    pub(crate) inner: DevelopmentCorrelationTest,
}

#[pymethods]
impl PyDevelopmentCorrelationTest {
    /// ``T``, the weighted mean of the pairs' Spearman correlations.
    #[getter]
    fn t(&self) -> f64 {
        self.inner.t
    }

    /// ``Var[T]``; ``E[T]`` is 0.
    #[getter]
    fn variance(&self) -> f64 {
        self.inner.variance
    }

    /// The interval's confidence level.
    #[getter]
    fn ci(&self) -> f64 {
        self.inner.ci
    }

    /// The interval's lower end, ``-upper``.
    #[getter]
    fn lower(&self) -> f64 {
        self.inner.lower
    }

    /// The interval's upper end.
    #[getter]
    fn upper(&self) -> f64 {
        self.inner.upper
    }

    /// Whether ``t`` lies within the interval (ends included): no
    /// correlation between adjacent development factors is detected.
    #[getter]
    fn passes(&self) -> bool {
        self.inner.passes()
    }

    /// Every pair of adjacent link columns with two origins in common,
    /// youngest first, as a list of dicts with ``age`` (the age in months
    /// at the start of the first link), ``n`` (the origins in common),
    /// ``t`` (Spearman's correlation with average ranks for ties; NaN if a
    /// column's ranks are all tied) and ``weight`` (``n - 1``, or 0 where
    /// ``t`` is NaN).
    #[getter]
    fn pairs<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .pairs
            .iter()
            .map(|p| {
                let row = PyDict::new(py);
                row.set_item("age", p.age)?;
                row.set_item("n", p.n)?;
                row.set_item("t", p.t)?;
                row.set_item("weight", p.weight)?;
                Ok(row)
            })
            .collect()
    }

    /// ``pairs`` as a table, one row per pair. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let p = &self.inner.pairs;
        let out = PyDict::new(py);
        out.set_item("age", p.iter().map(|p| p.age).collect::<Vec<_>>())?;
        out.set_item("n", p.iter().map(|p| p.n).collect::<Vec<_>>())?;
        out.set_item("t", p.iter().map(|p| p.t).collect::<Vec<_>>())?;
        out.set_item("weight", p.iter().map(|p| p.weight).collect::<Vec<_>>())?;
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    fn __repr__(&self) -> String {
        let t = &self.inner;
        format!(
            "DevelopmentCorrelationTest(t={:?}, variance={:?}, ci={:?}, lower={:?}, \
             upper={:?}, passes={})",
            t.t,
            t.variance,
            t.ci,
            t.lower,
            t.upper,
            if t.passes() { "True" } else { "False" }
        )
    }
}

/// Mack's calendar-year effect test (Mack 1994, appendix H; R
/// ChainLadder's ``cyEffTest``) on every observed link ratio of one
/// column of a single-segment triangle. ``MackFit.calendar_year_effect``
/// tests only the link ratios a fit averaged.
///
/// Parameters
/// ----------
/// triangle : Triangle
/// column : str
/// ci : float, default 0.95
///     The interval's confidence level, Mack's and R's 95%.
///
/// Returns
/// -------
/// CalendarYearTest
///
/// Raises
/// ------
/// ValueError
///     If the triangle has several segments or lacks the column, ``ci`` is
///     not strictly between 0 and 1, or no calendar diagonal has two link
///     ratios (fewer than three ages).
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Triangle, calendar_year_effect
/// >>> paid = [[100.0, 160.0, 180.0, 186.0, 188.0], [110.0, 170.0, 195.0, 200.0],
/// ...         [120.0, 200.0, 220.0], [130.0, 205.0], [140.0]]
/// >>> rows = [(2020 + w, 12 * (d + 1), v) for w, r in enumerate(paid)
/// ...         for d, v in enumerate(r)]
/// >>> origin, age, values = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(origin, age, {"paid": values})
/// >>> cy = calendar_year_effect(tri, "paid")
/// >>> cy.lower <= cy.expected <= cy.upper
/// True
#[pyfunction]
#[pyo3(signature = (triangle, column, ci = CALENDAR_YEAR_CI))]
pub(crate) fn calendar_year_effect(
    triangle: PyRef<'_, PyTriangle>,
    column: &str,
    ci: f64,
) -> PyResult<PyCalendarYearTest> {
    let inner = run_calendar_year_effect(&triangle.inner, column, ci).map_err(err)?;
    Ok(PyCalendarYearTest { inner })
}

/// Mack's test for correlation between adjacent development factors
/// (Mack 1994, appendix G; R ChainLadder's ``dfCorTest``) on every
/// observed link ratio of one column of a single-segment triangle.
/// ``MackFit.development_correlation`` tests only the link ratios a fit
/// averaged.
///
/// Parameters
/// ----------
/// triangle : Triangle
/// column : str
/// ci : float, default 0.5
///     The interval's confidence level, Mack's and R's 50%.
///
/// Returns
/// -------
/// DevelopmentCorrelationTest
///
/// Raises
/// ------
/// ValueError
///     If the triangle has several segments or lacks the column, ``ci`` is
///     not strictly between 0 and 1, or no pair of adjacent link columns
///     has two origins in common and untied ranks (fewer than four ages on
///     a square triangle).
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Triangle, development_correlation
/// >>> paid = [[100.0, 160.0, 180.0, 186.0, 188.0], [110.0, 170.0, 195.0, 200.0],
/// ...         [120.0, 200.0, 220.0], [130.0, 205.0], [140.0]]
/// >>> rows = [(2020 + w, 12 * (d + 1), v) for w, r in enumerate(paid)
/// ...         for d, v in enumerate(r)]
/// >>> origin, age, values = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(origin, age, {"paid": values})
/// >>> dc = development_correlation(tri, "paid")
/// >>> [p["age"] for p in dc.pairs]
/// [12, 24]
#[pyfunction]
#[pyo3(signature = (triangle, column, ci = DEVELOPMENT_CORRELATION_CI))]
pub(crate) fn development_correlation(
    triangle: PyRef<'_, PyTriangle>,
    column: &str,
    ci: f64,
) -> PyResult<PyDevelopmentCorrelationTest> {
    let inner = run_development_correlation(&triangle.inner, column, ci).map_err(err)?;
    Ok(PyDevelopmentCorrelationTest { inner })
}
