//! Python bindings for `prospicio_esg` (the economic scenario generator)
//! and `prospicio_aggregate::dfa` (the multi-year projection on its paths),
//! `docs/design/esg-dfa.md` PR 5. Paths come back as NumPy matrices of
//! simulations by steps (the scenarios) or by years (the projection).

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use prospicio_aggregate::dfa::{
    BondBucket, Breach, CapitalBasis, CapitalRule, Cycle, Estimate, Line, LossDependence,
    Projection, ProjectionPaths, RequiredCapital, Tax,
};
use prospicio_esg::{
    Cir, Curve, Esg, HullWhite, Index, Inflation, InitialCurve, Scenarios, ShortRate, Step,
    Superimposed, Svensson, Vasicek,
};
use prospicio_prob::copula::Archimedean;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::arrays::Matrix;
use crate::{to_py, unknown_name};

/// A row-major buffer as a `rows` by `cols` NumPy matrix.
fn matrix<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    PyArray1::from_vec(py, values).reshape([rows, cols])
}

fn step_named(name: &str) -> PyResult<Step> {
    Step::from_name(name).ok_or_else(|| unknown_name("step", name, Step::NAMES))
}

fn index_named(name: &str) -> PyResult<Index> {
    Index::from_name(name).ok_or_else(|| unknown_name("index", name, Index::NAMES))
}

fn line_named(name: &str) -> PyResult<Line> {
    Line::from_name(name).ok_or_else(|| unknown_name("line", name, Line::NAMES))
}

/// The Vasicek short rate, ``dr = a (b - r) dt + sigma dW``.
///
/// Parameters
/// ----------
/// a : float
///     The speed of mean reversion, positive.
/// b : float
///     The long-run mean.
/// sigma : float
///     The volatility, positive.
/// r0 : float
///     The rate today.
///
/// Examples
/// --------
/// >>> from prospicio.esg import Vasicek
/// >>> v = Vasicek(0.3, 0.04, 0.01, 0.02)
/// >>> round(v.mean(1.0), 6)
/// 0.025184
#[pyclass(name = "Vasicek", module = "prospicio.esg", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyVasicek {
    inner: Vasicek,
}

#[pymethods]
impl PyVasicek {
    #[new]
    fn new(a: f64, b: f64, sigma: f64, r0: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Vasicek::new(a, b, sigma, r0).map_err(to_py)?,
        })
    }

    /// The mean of the rate at ``t`` years.
    fn mean(&self, t: f64) -> f64 {
        self.inner.mean(t)
    }

    /// The variance of the rate at ``t`` years.
    fn variance(&self, t: f64) -> f64 {
        self.inner.variance(t)
    }

    /// The price of a zero-coupon bond of maturity ``tau`` when the rate is
    /// ``r``.
    fn bond_price(&self, r: f64, tau: f64) -> f64 {
        self.inner.bond_price(r, tau)
    }

    fn __repr__(&self) -> String {
        let v = &self.inner;
        format!(
            "Vasicek(a={}, b={}, sigma={}, r0={})",
            v.a, v.b, v.sigma, v.r0
        )
    }
}

/// The Cox-Ingersoll-Ross short rate, ``dr = a (b - r) dt + sigma sqrt(r) dW``.
///
/// Parameters
/// ----------
/// a, b, sigma, r0 : float
///     As for :class:`Vasicek`; ``r0`` non-negative.
///
/// Examples
/// --------
/// >>> from prospicio.esg import Cir
/// >>> Cir(0.4, 0.03, 0.08, 0.02).feller()
/// True
#[pyclass(name = "Cir", module = "prospicio.esg", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyCir {
    inner: Cir,
}

#[pymethods]
impl PyCir {
    #[new]
    fn new(a: f64, b: f64, sigma: f64, r0: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Cir::new(a, b, sigma, r0).map_err(to_py)?,
        })
    }

    /// Whether ``2 a b >= sigma^2``, so the rate stays positive.
    fn feller(&self) -> bool {
        self.inner.feller()
    }

    /// The mean of the rate at ``t`` years.
    fn mean(&self, t: f64) -> f64 {
        self.inner.mean(t)
    }

    /// The variance of the rate at ``t`` years.
    fn variance(&self, t: f64) -> f64 {
        self.inner.variance(t)
    }

    /// The price of a zero-coupon bond of maturity ``tau`` at rate ``r``.
    fn bond_price(&self, r: f64, tau: f64) -> f64 {
        self.inner.bond_price(r, tau)
    }
}

/// A Nelson-Siegel-Svensson curve of continuously compounded spot rates.
///
/// Parameters
/// ----------
/// beta : list of float
///     ``beta0`` to ``beta3``.
/// tau : list of float
///     ``tau1`` and ``tau2``, positive.
///
/// Examples
/// --------
/// >>> from prospicio.esg import Svensson
/// >>> c = Svensson([0.04, -0.02, 0.01, 0.0], [2.0, 8.0])
/// >>> round(c.spot(30.0), 4)
/// 0.0393
#[pyclass(name = "Svensson", module = "prospicio.esg", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PySvensson {
    inner: Svensson,
}

#[pymethods]
impl PySvensson {
    #[new]
    fn new(beta: [f64; 4], tau: [f64; 2]) -> PyResult<Self> {
        Ok(Self {
            inner: Svensson::new(beta, tau).map_err(to_py)?,
        })
    }

    /// Fit the curve to spot yields by maturity (least squares).
    #[staticmethod]
    fn fit(maturities: Vec<f64>, yields: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: Svensson::fit(&maturities, &yields).map_err(to_py)?,
        })
    }

    /// The parameters ``beta0`` to ``beta3``.
    #[getter]
    fn beta(&self) -> [f64; 4] {
        self.inner.beta
    }

    /// The parameters ``tau1`` and ``tau2``.
    #[getter]
    fn tau(&self) -> [f64; 2] {
        self.inner.tau
    }

    /// The spot rate at maturity ``m`` years.
    fn spot(&self, m: f64) -> f64 {
        self.inner.spot(m)
    }

    /// The instantaneous forward rate at ``m`` years.
    fn forward(&self, m: f64) -> f64 {
        self.inner.forward(m)
    }

    /// The discount factor at ``m`` years.
    fn discount(&self, m: f64) -> f64 {
        self.inner.discount(m)
    }
}

/// The Hull-White one-factor short rate, fitted to an initial curve.
///
/// Parameters
/// ----------
/// a : float
///     The speed of mean reversion, positive.
/// sigma : float
///     The volatility, positive.
/// maturities, rates : list of float, optional
///     The initial curve as annually compounded spot rates by maturity.
/// svensson : Svensson, optional
///     The initial curve as a Svensson curve, instead of ``maturities``
///     and ``rates``.
///
/// Examples
/// --------
/// >>> from prospicio.esg import HullWhite
/// >>> hw = HullWhite(0.1, 0.01, maturities=[1.0, 10.0], rates=[0.03, 0.04])
/// >>> round(hw.discount(1.0), 6)
/// 0.970874
#[pyclass(name = "HullWhite", module = "prospicio.esg", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyHullWhite {
    inner: HullWhite,
}

#[pymethods]
impl PyHullWhite {
    #[new]
    #[pyo3(signature = (a, sigma, maturities = None, rates = None, svensson = None))]
    fn new(
        a: f64,
        sigma: f64,
        maturities: Option<Vec<f64>>,
        rates: Option<Vec<f64>>,
        svensson: Option<PySvensson>,
    ) -> PyResult<Self> {
        let initial = match (maturities, rates, svensson) {
            (Some(m), Some(r), None) => InitialCurve::spot(m, r).map_err(to_py)?,
            (None, None, Some(s)) => InitialCurve::Svensson(s.inner),
            _ => {
                return Err(PyValueError::new_err(
                    "give the initial curve as maturities and rates, or as svensson",
                ));
            }
        };
        Ok(Self {
            inner: HullWhite::new(a, sigma, initial).map_err(to_py)?,
        })
    }

    /// The initial curve's discount factor at ``t`` years.
    fn discount(&self, t: f64) -> f64 {
        self.inner.initial.discount(t)
    }

    /// The initial curve's instantaneous forward rate at ``t`` years.
    fn forward(&self, t: f64) -> f64 {
        self.inner.initial.forward(t)
    }
}

/// Economic inflation, an Ornstein-Uhlenbeck rate,
/// ``di = kappa (mean - i) dt + sigma dW``.
///
/// Parameters
/// ----------
/// kappa : float
///     The speed of mean reversion, positive.
/// mean : float
///     The long-run rate.
/// sigma : float
///     The volatility, non-negative.
/// i0 : float
///     The rate today.
///
/// Examples
/// --------
/// >>> from prospicio.esg import Inflation
/// >>> round(Inflation(0.5, 0.025, 0.01, 0.03).index_mean(1.0), 6) > 1.0
/// True
#[pyclass(name = "Inflation", module = "prospicio.esg", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyInflation {
    inner: Inflation,
}

#[pymethods]
impl PyInflation {
    #[new]
    fn new(kappa: f64, mean: f64, sigma: f64, i0: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Inflation::new(kappa, mean, sigma, i0).map_err(to_py)?,
        })
    }

    /// The mean of the rate at ``t`` years.
    fn mean(&self, t: f64) -> f64 {
        self.inner.mean(t)
    }

    /// The variance of the rate at ``t`` years.
    fn variance(&self, t: f64) -> f64 {
        self.inner.variance(t)
    }

    /// The variance of the rate in the long run.
    fn stationary_variance(&self) -> f64 {
        self.inner.stationary_variance()
    }

    /// The mean of the price index at ``t`` years.
    fn index_mean(&self, t: f64) -> f64 {
        self.inner.index_mean(t)
    }

    /// The mean of the log price index at ``t`` years.
    fn log_index_mean(&self, t: f64) -> f64 {
        self.inner.log_index_mean(t)
    }

    /// The variance of the log price index at ``t`` years.
    fn log_index_variance(&self, t: f64) -> f64 {
        self.inner.log_index_variance(t)
    }
}

/// Superimposed claims inflation above the economic rate, an AR(1) by year,
/// ``s_t = mean + persistence (s_(t-1) - mean) + sigma e_t``.
///
/// Parameters
/// ----------
/// mean : float
///     The long-run excess.
/// persistence : float
///     In ``(-1, 1)``.
/// sigma : float
///     The shock's standard deviation, non-negative.
///
/// Examples
/// --------
/// >>> from prospicio.esg import Superimposed
/// >>> round(Superimposed(0.01, 0.5, 0.01).stationary_variance(), 8)
/// 0.00013333
#[pyclass(
    name = "Superimposed",
    module = "prospicio.esg",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PySuperimposed {
    inner: Superimposed,
}

#[pymethods]
impl PySuperimposed {
    #[new]
    fn new(mean: f64, persistence: f64, sigma: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Superimposed::new(mean, persistence, sigma).map_err(to_py)?,
        })
    }

    /// The variance of the excess after ``years`` years.
    fn variance(&self, years: usize) -> f64 {
        self.inner.variance(years)
    }

    /// The variance of the excess in the long run.
    fn stationary_variance(&self) -> f64 {
        self.inner.stationary_variance()
    }
}

/// A short rate model: Vasicek, Cir or HullWhite.
#[derive(FromPyObject)]
enum RateArg {
    Vasicek(PyVasicek),
    Cir(PyCir),
    HullWhite(PyHullWhite),
}

impl RateArg {
    fn short_rate(self) -> ShortRate {
        match self {
            RateArg::Vasicek(v) => v.inner.into(),
            RateArg::Cir(c) => c.inner.into(),
            RateArg::HullWhite(h) => h.inner.into(),
        }
    }
}

/// An economic scenario generator: a short rate, and optionally economic
/// and superimposed inflation with correlated shocks
/// (``docs/design/esg-dfa.md``).
///
/// Parameters
/// ----------
/// rate : Vasicek, Cir or HullWhite
///     The short rate model.
/// inflation : Inflation, optional
/// superimposed : Superimposed, optional
/// correlation : list of list of float, optional
///     The correlation of the shocks, in the order short rate, inflation,
///     superimposed, for the models given; independent by default.
/// step : str, default "monthly"
///     ``"monthly"``, ``"quarterly"`` or ``"annual"``.
/// maturities : list of float, optional
///     The maturities of the curve reported at each step.
///
/// Examples
/// --------
/// >>> from prospicio.esg import Esg, Inflation, Vasicek
/// >>> esg = Esg(Vasicek(0.3, 0.04, 0.01, 0.02), inflation=Inflation(0.5, 0.025, 0.01, 0.03),
/// ...           correlation=[[1.0, 0.4], [0.4, 1.0]], step="annual")
/// >>> sc = esg.simulate(5, 100, seed=1)
/// >>> sc.short_rates().shape
/// (100, 6)
#[pyclass(name = "Esg", module = "prospicio.esg", frozen)]
pub(crate) struct PyEsg {
    inner: Esg,
}

#[pymethods]
impl PyEsg {
    #[new]
    #[pyo3(signature = (rate, inflation = None, superimposed = None, correlation = None, step = "monthly", maturities = None))]
    fn new(
        rate: RateArg,
        inflation: Option<PyInflation>,
        superimposed: Option<PySuperimposed>,
        correlation: Option<Vec<Vec<f64>>>,
        step: &str,
        maturities: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let mut esg = Esg::new(rate.short_rate()).with_step(step_named(step)?);
        if let Some(i) = inflation {
            esg = esg.with_inflation(i.inner);
        }
        if let Some(s) = superimposed {
            esg = esg.with_superimposed(s.inner);
        }
        if let Some(c) = correlation {
            esg = esg.with_correlation(c);
        }
        if let Some(m) = maturities {
            esg = esg.with_maturities(m);
        }
        Ok(Self { inner: esg })
    }

    /// Simulate ``sims`` paths of ``years`` years from ``seed``.
    ///
    /// Path ``s`` draws from its own stream, so a path is the same whatever
    /// the number of paths and threads.
    fn simulate(
        &self,
        py: Python<'_>,
        years: usize,
        sims: usize,
        seed: u64,
    ) -> PyResult<PyScenarios> {
        let inner = py
            .detach(|| self.inner.simulate(years, sims, seed))
            .map_err(to_py)?;
        Ok(PyScenarios { inner })
    }
}

/// Simulated economic paths, from :meth:`Esg.simulate`.
///
/// Every matrix has one row per path and one column per step from today
/// (``n_steps + 1``), except :meth:`superimposed`, by year.
#[pyclass(name = "Scenarios", module = "prospicio.esg", frozen)]
pub(crate) struct PyScenarios {
    pub(crate) inner: Scenarios,
}

impl PyScenarios {
    fn by_step<'py>(
        &self,
        py: Python<'py>,
        f: impl Fn(usize, usize) -> prospicio_core::Result<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, k) = (self.inner.n_sims(), self.inner.n_steps() + 1);
        let mut out = Vec::with_capacity(n * k);
        for s in 0..n {
            for j in 0..k {
                out.push(f(s, j).map_err(to_py)?);
            }
        }
        matrix(py, out, n, k)
    }
}

#[pymethods]
impl PyScenarios {
    /// The number of paths.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims()
    }

    /// The number of steps after today.
    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps()
    }

    /// The step's name.
    #[getter]
    fn step(&self) -> &'static str {
        self.inner.step().name()
    }

    /// The time in years of each step, from 0.
    fn times(&self) -> Vec<f64> {
        (0..=self.inner.n_steps())
            .map(|k| self.inner.time(k))
            .collect()
    }

    /// The maturities of the reported curves.
    fn maturities(&self) -> Vec<f64> {
        self.inner.maturities().to_vec()
    }

    /// The short rate by path and step.
    fn short_rates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.by_step(py, |s, k| self.inner.short_rate(s, k))
    }

    /// The stochastic discount factor ``exp(-∫ r)`` by path and step.
    fn deflators<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.by_step(py, |s, k| self.inner.deflator(s, k))
    }

    /// The price of the zero-coupon bond of the ``j``-th maturity by path
    /// and step.
    fn bond_prices<'py>(&self, py: Python<'py>, j: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.by_step(py, |s, k| self.inner.bond_price(s, k, j))
    }

    /// The annually compounded spot curve of path ``sim`` at step ``k``:
    /// a dict of ``maturities`` and ``rates``.
    fn curve<'py>(&self, py: Python<'py>, sim: usize, k: usize) -> PyResult<Bound<'py, PyDict>> {
        let c: Curve = self.inner.curve(sim, k).map_err(to_py)?;
        let d = PyDict::new(py);
        d.set_item("maturities", c.maturities)?;
        d.set_item("rates", c.rates)?;
        Ok(d)
    }

    /// The economic inflation rate by path and step.
    fn inflation_rates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.by_step(py, |s, k| self.inner.inflation_rate(s, k))
    }

    /// The level of ``index`` (``"claims"`` or ``"price"``) by path and
    /// step, 1 today.
    #[pyo3(signature = (index = "claims"))]
    fn index_levels<'py>(
        &self,
        py: Python<'py>,
        index: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let index = index_named(index)?;
        self.by_step(py, |s, k| self.inner.index_level(index, s, k))
    }

    /// The superimposed excess by path and year (years 1 to the horizon).
    fn superimposed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = self.inner.n_sims();
        let years = self.inner.n_steps() / self.inner.step().per_year();
        let mut out = Vec::with_capacity(n * years);
        for s in 0..n {
            for t in 1..=years {
                out.push(self.inner.superimposed(s, t).map_err(to_py)?);
            }
        }
        matrix(py, out, n, years)
    }
}

/// An underwriting cycle: an AR(2) on the log rate adequacy that scales each
/// year's premium, ``x_t - mean = phi1 (x_(t-1) - mean) + phi2 (x_(t-2) -
/// mean) + sigma e_t``, from ``x_0 = x_(-1) = x0``.
///
/// Examples
/// --------
/// >>> from prospicio.esg import Cycle
/// >>> round(Cycle(0.0, 1.2, -0.6, 0.05, 0.0).period(), 3)
/// 9.176
#[pyclass(name = "Cycle", module = "prospicio.esg", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyCycle {
    inner: Cycle,
}

#[pymethods]
impl PyCycle {
    #[new]
    fn new(mean: f64, phi1: f64, phi2: f64, sigma: f64, x0: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Cycle::new(mean, phi1, phi2, sigma, x0).map_err(to_py)?,
        })
    }

    /// The cycle's period in years, ``None`` when it does not oscillate.
    fn period(&self) -> Option<f64> {
        self.inner.period()
    }

    /// The mean of ``x_t``.
    fn mean_at(&self, t: usize) -> f64 {
        self.inner.mean_at(t)
    }

    /// The variance of ``x_t``.
    fn variance_at(&self, t: usize) -> f64 {
        self.inner.variance_at(t)
    }

    /// The variance in the long run.
    fn stationary_variance(&self) -> f64 {
        self.inner.stationary_variance()
    }
}

/// A copula family for the losses' rank dependence on the index growth.
fn dependence(family: &str, parameter: f64, nu: Option<f64>) -> PyResult<LossDependence> {
    Ok(match family {
        "gaussian" => LossDependence::Gaussian { rho: parameter },
        "t" | "student_t" => LossDependence::StudentT {
            rho: parameter,
            nu: nu.ok_or_else(|| PyValueError::new_err("a t copula needs nu"))?,
        },
        other => match Archimedean::from_name(other) {
            Some(family) => LossDependence::Archimedean {
                family,
                theta: parameter,
            },
            None => {
                return Err(PyValueError::new_err(format!(
                    "dependence must be \"gaussian\", \"t\" or an Archimedean family ({}), got {other:?}",
                    Archimedean::NAMES
                        .iter()
                        .map(|(n, _)| format!("{n:?}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
        },
    })
}

/// A company's multi-year projection along economic scenarios
/// (``docs/design/esg-dfa.md``): premium and expenses at the start of each
/// year, the year's new losses scaled by the path's index and paid on a
/// pattern, the opening reserves' run-off, investment income on cash and
/// bond buckets, tax, dividends and injections.
///
/// Parameters
/// ----------
/// losses : numpy.ndarray or list of list of float
///     The new losses in base money, one row per path, one column per year.
/// opening_assets : float
/// premiums : list of float, optional
///     The premium of each year; none by default.
/// expense_ratio : float, default 0
/// opening_reserves : list of float, optional
///     The opening reserves' payments in base money, at the end of each year
///     from the first.
/// payout : list of float, optional
///     The share of a year's losses paid at the end of that year and each
///     later one; all in the year by default.
/// bonds : list of (float, float), optional
///     Bond buckets as ``(share, duration)``; the rest is cash.
/// index : str, default "claims"
///     The index that scales the losses: ``"claims"`` or ``"price"``.
/// cycle : Cycle, optional
/// cycle_seed : int, default 0
/// tax_rate : float, optional
/// carry_forward : bool, default True
///     Carry a tax loss forward against later profits.
/// dividend_above, inject_below : float, optional
///     Capital rule ratios of ``capital_basis``.
/// capital_basis : str, default "premium"
///     ``"premium"`` or ``"reserves"``.
/// dependence : str, optional
///     The losses' rank dependence on the index growth: ``"gaussian"``,
///     ``"t"``, or an Archimedean family (``"clayton"``, ``"gumbel"``,
///     ``"frank"``, ``"joe"``); independent by default.
/// dependence_parameter : float, default 0
///     The copula's correlation or ``theta``.
/// dependence_nu : float, optional
///     The t copula's degrees of freedom.
/// dependence_seed : int, default 0
///
/// Examples
/// --------
/// >>> from prospicio.esg import Projection
/// >>> p = Projection([[60.0, 70.0, 80.0], [50.0, 120.0, 40.0]], 200.0,
/// ...                premiums=[100.0] * 3, expense_ratio=0.3, payout=[0.5, 0.5])
/// >>> paths = p.run()
/// >>> paths.values("result")[1].round(6).tolist()
/// [20.0, -50.0, 30.0]
#[pyclass(name = "Projection", module = "prospicio.esg", frozen)]
pub(crate) struct PyProjection {
    inner: Projection,
}

#[pymethods]
impl PyProjection {
    #[new]
    #[pyo3(signature = (losses, opening_assets, premiums = None, expense_ratio = 0.0, opening_reserves = None, payout = None, bonds = None, index = "claims", cycle = None, cycle_seed = 0, tax_rate = None, carry_forward = true, dividend_above = None, inject_below = None, capital_basis = "premium", dependence = None, dependence_parameter = 0.0, dependence_nu = None, dependence_seed = 0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        losses: Matrix<'_>,
        opening_assets: f64,
        premiums: Option<Vec<f64>>,
        expense_ratio: f64,
        opening_reserves: Option<Vec<f64>>,
        payout: Option<Vec<f64>>,
        bonds: Option<Vec<(f64, f64)>>,
        index: &str,
        cycle: Option<PyCycle>,
        cycle_seed: u64,
        tax_rate: Option<f64>,
        carry_forward: bool,
        dividend_above: Option<f64>,
        inject_below: Option<f64>,
        capital_basis: &str,
        dependence: Option<&str>,
        dependence_parameter: f64,
        dependence_nu: Option<f64>,
        dependence_seed: u64,
    ) -> PyResult<Self> {
        let (rows, years, values) = match losses {
            Matrix::Rows(r) => {
                let years = r.first().map_or(0, Vec::len);
                let n = r.len();
                (n, years, Matrix::Rows(r).row_major(years, "year")?)
            }
            Matrix::Array(a) => {
                let (n, years) = {
                    let v = a.as_array();
                    v.dim()
                };
                (n, years, Matrix::Array(a).row_major(years, "year")?)
            }
        };
        let mut p = Projection::new(years, opening_assets, values, rows)
            .map_err(to_py)?
            .with_premiums(premiums.unwrap_or_else(|| vec![0.0; years]))
            .with_expense_ratio(expense_ratio)
            .with_index(index_named(index)?);
        if let Some(r) = opening_reserves {
            p = p.with_opening_reserves(r);
        }
        if let Some(pattern) = payout {
            p = p.with_payout(pattern);
        }
        if let Some(b) = bonds {
            p = p.with_bonds(
                b.into_iter()
                    .map(|(share, duration)| BondBucket { share, duration })
                    .collect(),
            );
        }
        if let Some(c) = cycle {
            p = p.with_cycle(c.inner, cycle_seed);
        }
        if let Some(rate) = tax_rate {
            p = p.with_tax(Tax {
                rate,
                carry_forward,
            });
        }
        if dividend_above.is_some() || inject_below.is_some() {
            let basis = CapitalBasis::from_name(capital_basis)
                .ok_or_else(|| unknown_name("capital_basis", capital_basis, CapitalBasis::NAMES))?;
            p = p.with_capital(CapitalRule {
                basis,
                dividend_above,
                inject_below,
            });
        }
        if let Some(family) = dependence {
            p = p.with_loss_dependence(
                self::dependence(family, dependence_parameter, dependence_nu)?,
                dependence_seed,
            );
        }
        Ok(Self { inner: p })
    }

    /// Run the projection along ``scenarios`` (path ``s`` on scenario path
    /// ``s``), or without an economy when ``None``: no investment return and
    /// an index of 1.
    #[pyo3(signature = (scenarios = None))]
    fn run(&self, py: Python<'_>, scenarios: Option<&PyScenarios>) -> PyResult<PyProjectionPaths> {
        let inner = py
            .detach(|| match scenarios {
                Some(sc) => self.inner.run(&sc.inner),
                None => self.inner.run_without_economy(),
            })
            .map_err(to_py)?;
        Ok(PyProjectionPaths { inner })
    }
}

fn estimate<'py>(py: Python<'py>, e: Estimate) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("probability", e.probability)?;
    d.set_item("standard_error", e.standard_error)?;
    Ok(d)
}

/// A projection's paths: each line by path and year, ruin and breach.
#[pyclass(name = "ProjectionPaths", module = "prospicio.esg", frozen)]
pub(crate) struct PyProjectionPaths {
    inner: ProjectionPaths,
}

#[pymethods]
impl PyProjectionPaths {
    /// The names of the lines, for :meth:`values`.
    #[staticmethod]
    fn lines() -> Vec<&'static str> {
        Line::NAMES.iter().map(|(n, _)| *n).collect()
    }

    /// The number of paths.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims()
    }

    /// The years projected.
    #[getter]
    fn years(&self) -> usize {
        self.inner.years()
    }

    /// The surplus today, assets less the opening reserves.
    #[getter]
    fn opening_surplus(&self) -> f64 {
        self.inner.opening_surplus()
    }

    /// A line (as ``"surplus"``, from :meth:`lines`) by path and year.
    fn values<'py>(&self, py: Python<'py>, line: &str) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let v = self.inner.values(line_named(line)?).to_vec();
        matrix(py, v, self.inner.n_sims(), self.inner.years())
    }

    /// The probability of ruin (the surplus before capital below zero at a
    /// year end) within ``horizon`` years, with its standard error: a dict
    /// of ``probability`` and ``standard_error``.
    fn ruin<'py>(&self, py: Python<'py>, horizon: usize) -> PyResult<Bound<'py, PyDict>> {
        estimate(py, self.inner.ruin(horizon).map_err(to_py)?)
    }

    /// The probability that the surplus before capital falls below the
    /// required capital at a year end within ``horizon`` years.
    ///
    /// ``capital`` is a fixed amount, a list of amounts by year, or a pair
    /// ``("premium", ratio)`` or ``("reserves", ratio)``. Returns a dict of
    /// ``probability``, ``standard_error`` and ``shortfall`` (the mean
    /// largest shortfall on the paths that breach, ``None`` when none do).
    fn breach<'py>(
        &self,
        py: Python<'py>,
        horizon: usize,
        capital: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let required = if let Ok(x) = capital.extract::<f64>() {
            RequiredCapital::Fixed(x)
        } else if let Ok((basis, ratio)) = capital.extract::<(String, f64)>() {
            match CapitalBasis::from_name(&basis) {
                Some(CapitalBasis::Premium) => RequiredCapital::PremiumRatio(ratio),
                Some(CapitalBasis::Reserves) => RequiredCapital::ReserveRatio(ratio),
                None => return Err(unknown_name("capital", &basis, CapitalBasis::NAMES)),
            }
        } else if let Ok(v) = capital.extract::<Vec<f64>>() {
            RequiredCapital::ByYear(v)
        } else {
            return Err(PyValueError::new_err(
                "capital must be an amount, a list of amounts by year, or (basis, ratio)",
            ));
        };
        let b: Breach = self.inner.breach(horizon, &required).map_err(to_py)?;
        let d = estimate(py, b.estimate)?;
        d.set_item("shortfall", b.shortfall)?;
        Ok(d)
    }
}
