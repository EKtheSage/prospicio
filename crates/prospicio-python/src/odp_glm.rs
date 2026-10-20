//! `prospicio.reserving.OdpGlm` and `OdpGlmFit` (issue #333), over
//! `prospicio_reserving::OdpGlm`: the ODP model as a quasi-Poisson GLM on a
//! triangle, with the reserves' analytic prediction error, residuals and
//! custom designs.

use prospicio_glm::ParameterDraws;
use prospicio_models::Column;
use prospicio_reserving::{OdpGlm, OdpGlmFit, ResidualGroup, SumPredictionError, TriangleFrame};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::distributions::PyPredictiveDistribution;
use crate::models::{PyDesign, PyGlmFit};
use crate::reserving::PyTriangle;

fn err(e: prospicio_reserving::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The over-dispersed Poisson (ODP) reserving model as a quasi-Poisson GLM:
/// ``ln E[X] = c + a[origin] + b[development]`` on the incremental values,
/// ``Var X = phi E[X]``. Its future cells summed by origin are the
/// volume-weighted chain-ladder reserves. ``fit`` takes another design on
/// the triangle's cells too (``OdpGlm.cells``): fewer parameters, calendar
/// trends, interactions (Taylor and McGuire, CAS Monograph 3, Chapter 7).
///
/// Parameters
/// ----------
/// tolerance : float, default 1e-10
///     IRLS convergence tolerance.
/// max_iterations : int, default 100
///     Maximum IRLS iterations.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import OdpGlm, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2020, 2021, 2021, 2022],
/// ...     [12, 24, 36, 12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 165.0, 110.0, 170.0, 120.0]},
/// ... )
/// >>> fit = OdpGlm().fit(tri, "paid")
/// >>> round(fit.total_reserve, 4)
/// 98.1429
#[pyclass(name = "OdpGlm", module = "prospicio.reserving", frozen)]
pub(crate) struct PyOdpGlm {
    inner: OdpGlm,
}

#[pymethods]
impl PyOdpGlm {
    #[new]
    #[pyo3(signature = (tolerance = 1e-10, max_iterations = 100))]
    fn new(tolerance: f64, max_iterations: usize) -> Self {
        Self {
            inner: OdpGlm {
                tolerance,
                max_iterations,
            },
        }
    }

    /// IRLS convergence tolerance.
    #[getter]
    fn tolerance(&self) -> f64 {
        self.inner.tolerance
    }

    /// Maximum IRLS iterations.
    #[getter]
    fn max_iterations(&self) -> usize {
        self.inner.max_iterations
    }

    /// Fits the GLM to the incremental values of one measure column of a
    /// single-segment triangle and predicts the cells below the latest
    /// diagonal.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    /// design : prospicio.models.Design, optional
    ///     The model's columns, one row per cell of ``OdpGlm.cells(triangle,
    ///     column)`` in its order, intercept included if wanted; its offset
    ///     and weights are used too. By default the cross-classified model,
    ///     an intercept and origin and development factors. A
    ///     ``prospicio.models.Terms`` coding of the cells gives one
    ///     (``terms.fit(cells).design(cells)``), or build the columns from
    ///     the cells' indices.
    ///
    /// Returns
    /// -------
    /// OdpGlmFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the triangle has several segments, a past cell has no
    ///     increment, or the GLM does not converge (a collinear design among
    ///     them); without ``design``, if an origin or age has no observed
    ///     increment; with it, if it does not have a row per cell or a
    ///     column is zero on every observed cell.
    ///
    /// Examples
    /// --------
    /// The origins' effect as a straight line in their index instead of a
    /// level each:
    ///
    /// >>> from prospicio.models import Design
    /// >>> from prospicio.reserving import OdpGlm, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020, 2020, 2021, 2021, 2022],
    /// ...     [12, 24, 36, 12, 24, 12],
    /// ...     {"paid": [100.0, 150.0, 165.0, 110.0, 170.0, 120.0]},
    /// ... )
    /// >>> cells = OdpGlm.cells(tri, "paid")
    /// >>> j = cells["development_index"]
    /// >>> design = Design(
    /// ...     [[1.0] * len(j), cells["origin_index"], [float(x == 1) for x in j],
    /// ...      [float(x == 2) for x in j]],
    /// ...     ["a0", "slope", "b2", "b3"],
    /// ... )
    /// >>> fit = OdpGlm().fit(tri, "paid", design=design)
    /// >>> fit.glm.names
    /// ['a0', 'slope', 'b2', 'b3']
    #[pyo3(signature = (triangle, column, design = None))]
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        design: Option<PyRef<'_, PyDesign>>,
    ) -> PyResult<PyOdpGlmFit> {
        let (model, tri) = (self.inner, &triangle.inner);
        let design = design.map(|d| d.inner.clone());
        let inner = py
            .detach(|| match &design {
                Some(d) => model.fit_design(tri, column, d),
                None => model.fit(tri, column),
            })
            .map_err(err)?;
        Ok(PyOdpGlmFit { inner })
    }

    /// The cells of a single-segment triangle, one per origin and
    /// development age, origin-major: the rows a ``design`` for ``fit``
    /// has, in that order.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///
    /// Returns
    /// -------
    /// dict
    ///     Lists of equal length: ``origin``, ``development`` (the age in
    ///     months) and ``calendar`` as labels; ``origin_index``,
    ///     ``development_index`` and ``calendar_index`` (0-based, the last
    ///     from the earliest valuation), ``age`` in months and
    ///     ``exposure`` (1) as numbers; ``observed`` and ``future``, whether
    ///     the cell is fitted or predicted (a cell after a hole is
    ///     neither).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the triangle has several segments or no such column.
    #[staticmethod]
    fn cells<'py>(
        py: Python<'py>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let cells = TriangleFrame::new(&triangle.inner, column, None).map_err(err)?;
        let out = PyDict::new(py);
        for name in [
            "origin",
            "development",
            "calendar",
            "origin_index",
            "development_index",
            "calendar_index",
            "age",
            "exposure",
        ] {
            match cells.frame().column(name) {
                Some(Column::Numeric(v)) => out.set_item(name, v.clone())?,
                Some(Column::Categorical(v)) => out.set_item(name, v.clone())?,
                None => unreachable!("a triangle frame has column {name}"),
            }
        }
        let mark = |rows: &[usize]| {
            let mut v = vec![false; cells.n_rows()];
            for &r in rows {
                v[r] = true;
            }
            v
        };
        out.set_item("observed", mark(cells.observed()))?;
        out.set_item("future", mark(cells.future()))?;
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!(
            "OdpGlm(tolerance={:?}, max_iterations={})",
            self.inner.tolerance, self.inner.max_iterations
        )
    }
}

fn error_dict<'py>(py: Python<'py>, e: &SumPredictionError) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("reserve", e.total)?;
    out.set_item("process_se", e.process_variance.sqrt())?;
    out.set_item("parameter_se", e.parameter_variance.sqrt())?;
    out.set_item("rmsep", e.rmsep())?;
    Ok(out)
}

/// A fitted ODP GLM, from ``OdpGlm.fit``: the GLM, the future cells' means,
/// the reserves and their prediction error.
#[pyclass(name = "OdpGlmFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyOdpGlmFit {
    inner: OdpGlmFit,
}

#[pymethods]
impl PyOdpGlmFit {
    /// The quasi-Poisson GLM on the observed increments: coefficients,
    /// standard errors, covariance and Pearson's dispersion. Without a
    /// ``design``, its columns are ``(Intercept)``, ``origin[<period>]`` and
    /// ``development[<age>]``, with the first origin and the first age as
    /// references; with one, the design's.
    #[getter]
    fn glm(&self) -> PyGlmFit {
        PyGlmFit {
            inner: self.inner.glm.clone(),
        }
    }

    /// Pearson's dispersion ``phi``.
    #[getter]
    fn dispersion(&self) -> f64 {
        self.inner.glm.dispersion()
    }

    /// Origin periods of the triangle.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// Reserve of each origin, aligned with ``origins``: its future cells'
    /// fitted means summed, 0 for a fully developed origin.
    #[getter]
    fn reserves(&self) -> Vec<f64> {
        self.inner.reserves.clone()
    }

    /// Sum of the reserves.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    /// The future cells, as ``(origin, age in months)``, origin-major.
    #[getter]
    fn future(&self) -> Vec<(String, u32)> {
        self.inner
            .future
            .iter()
            .map(|(o, a)| (o.to_string(), *a))
            .collect()
    }

    /// Fitted mean of each future cell, aligned with ``future``.
    #[getter]
    fn future_means(&self) -> Vec<f64> {
        self.inner.future_means.clone()
    }

    /// Design of the future cells, rows aligned with ``future``: what
    /// ``glm`` predicts from.
    #[getter]
    fn future_design(&self) -> PyDesign {
        PyDesign {
            inner: self.inner.future_design().clone(),
        }
    }

    /// The reserves' prediction error by origin and in total (Taylor and
    /// McGuire, CAS Monograph 3, (5-11) to (5-13), Table 5-4): the process
    /// error ``phi * sum(mu)`` of the future cells, the parameter error
    /// ``g' Sigma g`` by the delta method with ``g = sum(mu * x)``, and the
    /// root mean square error of prediction. The total's exceeds the
    /// origins' combined as if independent, because every origin shares
    /// the development factors.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``origin`` (the labels, then ``"Total"``), ``reserve``,
    ///     ``process_se``, ``parameter_se``, ``rmsep`` and ``cv``
    ///     (``rmsep / reserve``, ``nan`` for a zero reserve), as lists.
    ///     ``pandas.DataFrame(fit.prediction_error())`` lays it out as
    ///     Table 5-4.
    fn prediction_error<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let error = self.inner.prediction_error().map_err(err)?;
        let rows: Vec<&SumPredictionError> = error.by_origin.iter().chain([&error.total]).collect();
        let mut origin = self.origins();
        origin.push("Total".into());
        let col = |f: fn(&SumPredictionError) -> f64| rows.iter().map(|e| f(e)).collect::<Vec<_>>();
        let out = PyDict::new(py);
        out.set_item("origin", origin)?;
        out.set_item("reserve", col(|e| e.total))?;
        out.set_item("process_se", col(|e| e.process_variance.sqrt()))?;
        out.set_item("parameter_se", col(|e| e.parameter_variance.sqrt()))?;
        out.set_item("rmsep", col(SumPredictionError::rmsep))?;
        out.set_item(
            "cv",
            col(|e| {
                if e.total == 0.0 {
                    f64::NAN
                } else {
                    e.rmsep() / e.total
                }
            }),
        )?;
        Ok(out)
    }

    /// The residuals of every observed cell (Taylor and McGuire, CAS
    /// Monograph 3, Chapter 6). A cell with leverage 1, the only
    /// observation of its origin or age, is fitted exactly and has no
    /// standardized residual (``nan``).
    ///
    /// Returns
    /// -------
    /// dict
    ///     One list per column, a row per observed cell, origin-major:
    ///     ``origin``, ``development`` (age in months), ``calendar`` (the
    ///     calendar period's label), ``calendar_index`` (0 for the earliest
    ///     valuation), ``actual``, ``fitted``, ``actual_to_fitted`` (the
    ///     heat map's cell), ``pearson``, ``deviance``, ``leverage`` (the
    ///     hat value), ``standardized_deviance`` (R's ``rstandard``) and
    ///     ``standardized_pearson``.
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = py.detach(|| self.inner.residuals()).map_err(err)?;
        let out = PyDict::new(py);
        out.set_item(
            "origin",
            r.origin.iter().map(ToString::to_string).collect::<Vec<_>>(),
        )?;
        out.set_item("development", r.development)?;
        out.set_item(
            "calendar",
            r.calendar.iter().map(|c| c.1.clone()).collect::<Vec<_>>(),
        )?;
        out.set_item(
            "calendar_index",
            r.calendar.iter().map(|c| c.0).collect::<Vec<_>>(),
        )?;
        out.set_item("actual", r.actual)?;
        out.set_item("fitted", r.fitted)?;
        out.set_item("actual_to_fitted", r.actual_to_fitted)?;
        out.set_item("pearson", r.pearson)?;
        out.set_item("deviance", r.deviance)?;
        out.set_item("leverage", r.leverage)?;
        out.set_item("standardized_deviance", r.standardized_deviance)?;
        out.set_item("standardized_pearson", r.standardized_pearson)?;
        Ok(out)
    }

    /// The standardized deviance residuals grouped by origin, development
    /// age or calendar period: a trend in the calendar means points to a
    /// calendar effect the model lacks, a spread that falls with age to a
    /// variance that is not ``phi * mu``.
    ///
    /// Parameters
    /// ----------
    /// by : {"development", "origin", "calendar"}, default "development"
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``label``, ``n`` (cells with a defined residual), ``mean`` and
    ///     ``sd`` (divisor ``n - 1``; ``nan`` below 2 cells), as lists.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     ``by`` is unknown.
    #[pyo3(signature = (by = "development"))]
    fn residual_summary<'py>(&self, py: Python<'py>, by: &str) -> PyResult<Bound<'py, PyDict>> {
        let group = match by {
            "development" => ResidualGroup::Development,
            "origin" => ResidualGroup::Origin,
            "calendar" => ResidualGroup::Calendar,
            _ => {
                return Err(PyValueError::new_err(
                    "by must be \"development\", \"origin\" or \"calendar\"",
                ));
            }
        };
        let rows = py
            .detach(|| self.inner.residuals())
            .map_err(err)?
            .summary(group);
        let out = PyDict::new(py);
        out.set_item(
            "label",
            rows.iter().map(|g| g.label.clone()).collect::<Vec<_>>(),
        )?;
        out.set_item("n", rows.iter().map(|g| g.n).collect::<Vec<_>>())?;
        out.set_item("mean", rows.iter().map(|g| g.mean).collect::<Vec<_>>())?;
        out.set_item("sd", rows.iter().map(|g| g.sd).collect::<Vec<_>>())?;
        Ok(out)
    }

    /// The total reserve's prediction error: ``reserve``, ``process_se``,
    /// ``parameter_se`` and ``rmsep``.
    ///
    /// Returns
    /// -------
    /// dict
    fn total_prediction_error<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let error = self.inner.prediction_error().map_err(err)?;
        error_dict(py, &error.total)
    }

    /// Mean of each future cell under normal parameter draws,
    /// ``exp(eta + v / 2)``: the fitted mean times the lognormal bias of
    /// parameter uncertainty on the log scale.
    ///
    /// Returns
    /// -------
    /// list of float
    fn predictive_means(&self) -> Vec<f64> {
        self.inner.predictive_means()
    }

    /// Joint draws of the future cells' increments, with dimensions
    /// ``["origin", "development"]``. Each simulation draws the coefficients
    /// from their normal approximation, then each cell as
    /// ``phi * Poisson(mu / phi)``. ``aggregate(["origin"])`` gives the
    /// reserves by origin.
    ///
    /// Parameters
    /// ----------
    /// n_sims : int
    /// seed : int
    /// parameters : {"mean_preserving", "normal", "fixed"}, default "mean_preserving"
    ///     ``"mean_preserving"`` shifts each cell's linear predictor by
    ///     ``-v / 2`` so the draws average the fitted means (the chain
    ///     ladder's); ``"normal"`` leaves it (means ``predictive_means()``);
    ///     ``"fixed"`` keeps the fitted coefficients (process error only).
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``n_sims`` is 0, the triangle is fully developed, or
    ///     ``parameters`` is unknown.
    #[pyo3(signature = (n_sims, seed, parameters = "mean_preserving"))]
    fn predict_distribution(
        &self,
        py: Python<'_>,
        n_sims: usize,
        seed: u64,
        parameters: &str,
    ) -> PyResult<PyPredictiveDistribution> {
        let parameters = match parameters {
            "mean_preserving" => ParameterDraws::MeanPreserving,
            "normal" => ParameterDraws::Normal,
            "fixed" => ParameterDraws::Fixed,
            _ => {
                return Err(PyValueError::new_err(
                    "parameters must be \"mean_preserving\", \"normal\" or \"fixed\"",
                ));
            }
        };
        let fit = &self.inner;
        let inner = py
            .detach(|| fit.predict_distribution_with(n_sims, seed, parameters))
            .map_err(err)?;
        Ok(PyPredictiveDistribution { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "OdpGlmFit(origins={}, total_reserve={}, dispersion={})",
            self.inner.origins.len(),
            self.inner.total_reserve(),
            self.inner.glm.dispersion()
        )
    }
}
