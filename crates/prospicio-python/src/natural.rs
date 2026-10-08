//! `prospicio.pricing` (Aggregate lane): natural allocation of a
//! portfolio held as the conditional expectations of its units given the
//! total, over `prospicio_pricing::natural`.

use prospicio_pricing::natural::{Allocation, NaturalPrice, Pentagon, Portfolio, Quantity, Target};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::distributions::{PyGrid, PyPredictiveDistribution};
use crate::risk::{PyDistortion, discrete_of, family_from};
use crate::to_py;
use prospicio_pricing::classical::{self, Kind, Principle};

/// Loss, margin, premium, capital and assets, with ``P = L + M`` and
/// ``a = P + Q``.
///
/// Build one from any three known quantities with ``Pentagon.solve``.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import Pentagon
/// >>> p = Pentagon.solve(loss=46.6, assets=100.0, return_on_capital=0.15)
/// >>> round(p.premium, 6)
/// 53.565217
#[pyclass(
    name = "Pentagon",
    module = "prospicio.pricing",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyPentagon {
    inner: Pentagon,
}

#[pymethods]
impl PyPentagon {
    /// The pentagon fixed by exactly three of its quantities.
    ///
    /// Parameters
    /// ----------
    /// loss, margin, premium, capital, assets : float, optional
    /// loss_ratio, premium_to_capital, return_on_capital : float, optional
    ///
    /// Returns
    /// -------
    /// Pentagon
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Unless exactly three are given and they determine the rest.
    #[staticmethod]
    #[pyo3(signature = (*, loss=None, margin=None, premium=None, capital=None, assets=None,
                        loss_ratio=None, premium_to_capital=None, return_on_capital=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        loss: Option<f64>,
        margin: Option<f64>,
        premium: Option<f64>,
        capital: Option<f64>,
        assets: Option<f64>,
        loss_ratio: Option<f64>,
        premium_to_capital: Option<f64>,
        return_on_capital: Option<f64>,
    ) -> PyResult<Self> {
        let given: Vec<(Quantity, f64)> = [
            (Quantity::Loss, loss),
            (Quantity::Margin, margin),
            (Quantity::Premium, premium),
            (Quantity::Capital, capital),
            (Quantity::Assets, assets),
            (Quantity::LossRatio, loss_ratio),
            (Quantity::PremiumToCapital, premium_to_capital),
            (Quantity::ReturnOnCapital, return_on_capital),
        ]
        .into_iter()
        .filter_map(|(q, v)| v.map(|v| (q, v)))
        .collect();
        let known: [(Quantity, f64); 3] = given
            .try_into()
            .map_err(|_| PyValueError::new_err("give exactly three quantities"))?;
        Ok(Self {
            inner: Pentagon::solve(known).map_err(to_py)?,
        })
    }

    /// Expected loss paid, ``L``.
    #[getter]
    fn loss(&self) -> f64 {
        self.inner.loss
    }

    /// Margin ``M = P - L``.
    #[getter]
    fn margin(&self) -> f64 {
        self.inner.margin
    }

    /// Premium ``P``.
    #[getter]
    fn premium(&self) -> f64 {
        self.inner.premium
    }

    /// Capital ``Q = a - P``.
    #[getter]
    fn capital(&self) -> f64 {
        self.inner.capital
    }

    /// Assets ``a``.
    #[getter]
    fn assets(&self) -> f64 {
        self.inner.assets
    }

    /// Loss ratio ``L / P``.
    #[getter]
    fn loss_ratio(&self) -> f64 {
        self.inner.loss_ratio()
    }

    /// Premium leverage ``P / Q``.
    #[getter]
    fn premium_to_capital(&self) -> f64 {
        self.inner.premium_to_capital()
    }

    /// Return on capital ``M / Q``.
    #[getter]
    fn return_on_capital(&self) -> f64 {
        self.inner.return_on_capital()
    }

    /// Discount ``M / (a - L)``; ``P = (1 - discount) L + discount a``.
    #[getter]
    fn discount(&self) -> f64 {
        self.inner.discount()
    }

    fn __repr__(&self) -> String {
        let p = &self.inner;
        format!(
            "Pentagon(loss={}, margin={}, premium={}, capital={}, assets={})",
            p.loss, p.margin, p.premium, p.capital, p.assets
        )
    }
}

/// A portfolio's price and its natural allocation, from ``Portfolio.price``.
#[pyclass(name = "NaturalPrice", module = "prospicio.pricing", frozen)]
pub(crate) struct PyNaturalPrice {
    inner: NaturalPrice,
}

#[pymethods]
impl PyNaturalPrice {
    /// Unit names.
    #[getter]
    fn units(&self) -> Vec<String> {
        self.inner.units.clone()
    }

    /// Each unit's share, a list of Pentagon; the amounts add up to ``total``.
    #[getter]
    fn allocated(&self) -> Vec<PyPentagon> {
        self.inner
            .allocated
            .iter()
            .map(|&inner| PyPentagon { inner })
            .collect()
    }

    /// The portfolio, a Pentagon.
    #[getter]
    fn total(&self) -> PyPentagon {
        PyPentagon {
            inner: self.inner.total,
        }
    }

    /// The table as a dict of columns: ``unit`` (the units, then
    /// ``"total"``), ``loss``, ``margin``, ``premium``, ``capital``,
    /// ``assets``, ``loss_ratio``, ``premium_to_capital`` and
    /// ``return_on_capital``. ``pandas.DataFrame(price.to_dict())`` makes
    /// a frame of it.
    ///
    /// Returns
    /// -------
    /// dict of str to list
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let rows: Vec<&Pentagon> = self
            .inner
            .allocated
            .iter()
            .chain([&self.inner.total])
            .collect();
        let mut names = self.inner.units.clone();
        names.push("total".into());
        let d = pyo3::types::PyDict::new(py);
        d.set_item("unit", names)?;
        let col = |f: fn(&Pentagon) -> f64| rows.iter().map(|p| f(p)).collect::<Vec<f64>>();
        d.set_item("loss", col(|p| p.loss))?;
        d.set_item("margin", col(|p| p.margin))?;
        d.set_item("premium", col(|p| p.premium))?;
        d.set_item("capital", col(|p| p.capital))?;
        d.set_item("assets", col(|p| p.assets))?;
        d.set_item("loss_ratio", col(Pentagon::loss_ratio))?;
        d.set_item("premium_to_capital", col(Pentagon::premium_to_capital))?;
        d.set_item("return_on_capital", col(Pentagon::return_on_capital))?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "NaturalPrice(units={:?}, premium={}, assets={})",
            self.inner.units, self.inner.total.premium, self.inner.total.assets
        )
    }
}

/// A portfolio as the distribution of its total and each unit's
/// conditional expectation given the total, ``kappa_i(x) = E[X_i | X = x]``:
/// the representation of Mildenhall and Major (*Pricing Insurance Risk*,
/// 2022) and CAS Monograph 15, for pricing with limited liability and the
/// natural allocation.
///
/// Parameters
/// ----------
/// units : list of str
/// rows : list of list of float
///     Each scenario's loss by unit; non-negative.
/// probs : list of float, optional
///     The scenarios' probabilities; equal when omitted.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import Portfolio
/// >>> from prospicio.risk import Distortion
/// >>> rows = [[15, 7, 0], [15, 13, 0], [5, 20, 11], [7, 33, 0], [13, 20, 7],
/// ...         [5, 27, 8], [15, 16, 9], [26, 19, 10], [17, 8, 40], [16, 20, 64]]
/// >>> insco = Portfolio(["A", "B", "C"], rows)
/// >>> price = insco.price(Distortion.ccoc(0.15), assets=100)
/// >>> round(price.total.premium, 6)
/// 53.565217
/// >>> [round(u.assets, 6) for u in price.allocated]
/// [16.0, 20.0, 64.0]
#[pyclass(name = "Portfolio", module = "prospicio.pricing", frozen)]
pub(crate) struct PyPortfolio {
    inner: Portfolio,
}

fn allocation(name: &str) -> PyResult<Allocation> {
    match name {
        "linear" => Ok(Allocation::Linear),
        "lifted" => Ok(Allocation::Lifted),
        other => Err(PyValueError::new_err(format!(
            "allocation must be \"linear\" or \"lifted\", not {other:?}"
        ))),
    }
}

impl PyPortfolio {
    fn level(&self, assets: Option<f64>, p: Option<f64>) -> PyResult<f64> {
        match (assets, p) {
            (Some(a), None) => Ok(a),
            (None, Some(p)) => self.inner.assets(p).map_err(to_py),
            (None, None) => Ok(self.inner.max()),
            (Some(_), Some(_)) => Err(PyValueError::new_err("give assets or p, not both")),
        }
    }
}

#[pymethods]
impl PyPortfolio {
    #[new]
    #[pyo3(signature = (units, rows, probs=None))]
    fn new(units: Vec<String>, rows: Vec<Vec<f64>>, probs: Option<Vec<f64>>) -> PyResult<Self> {
        Ok(Self {
            inner: Portfolio::from_rows(units, &rows, probs.as_deref()).map_err(to_py)?,
        })
    }

    /// From a joint simulation: each component is a unit, each simulation
    /// an equally likely scenario.
    ///
    /// Parameters
    /// ----------
    /// pd : PredictiveDistribution
    ///
    /// Returns
    /// -------
    /// Portfolio
    #[staticmethod]
    fn from_predictive(pd: PyRef<'_, PyPredictiveDistribution>) -> PyResult<Self> {
        Ok(Self {
            inner: Portfolio::from_predictive(&pd.inner).map_err(to_py)?,
        })
    }

    /// From independent units, each a Grid with the same step; the total
    /// and the conditional expectations are computed by FFT.
    ///
    /// Parameters
    /// ----------
    /// units : list of str
    /// grids : list of Grid
    ///
    /// Returns
    /// -------
    /// Portfolio
    #[staticmethod]
    fn from_independent(units: Vec<String>, grids: Vec<PyRef<'_, PyGrid>>) -> PyResult<Self> {
        let grids: Vec<_> = grids.iter().map(|g| g.inner.clone()).collect();
        Ok(Self {
            inner: Portfolio::from_independent(units, &grids).map_err(to_py)?,
        })
    }

    /// Unit names.
    #[getter]
    fn units(&self) -> Vec<String> {
        self.inner.units().to_vec()
    }

    /// The distinct totals, ascending.
    #[getter]
    fn totals(&self) -> Vec<f64> {
        self.inner.totals().to_vec()
    }

    /// The probability of each total.
    #[getter]
    fn probs(&self) -> Vec<f64> {
        self.inner.probs().to_vec()
    }

    /// A unit's conditional expectation at each total.
    ///
    /// Parameters
    /// ----------
    /// unit : str
    ///
    /// Returns
    /// -------
    /// list of float
    fn kappa(&self, unit: &str) -> PyResult<Vec<f64>> {
        let i = self
            .inner
            .units()
            .iter()
            .position(|u| u == unit)
            .ok_or_else(|| PyValueError::new_err(format!("no unit {unit:?}")))?;
        Ok(self.inner.kappa(i))
    }

    /// Each unit's expected loss.
    ///
    /// Returns
    /// -------
    /// list of float
    fn expected(&self) -> Vec<f64> {
        self.inner.expected()
    }

    /// Assets at the capital standard ``p``: the lower ``p`` quantile of
    /// the total.
    ///
    /// Parameters
    /// ----------
    /// p : float
    ///
    /// Returns
    /// -------
    /// float
    fn assets(&self, p: f64) -> PyResult<f64> {
        self.inner.assets(p).map_err(to_py)
    }

    /// The premium of the loss capped at the assets, and its natural
    /// allocation to the units.
    ///
    /// Parameters
    /// ----------
    /// distortion : Distortion
    /// assets : float, optional
    ///     The asset level; or give ``p``. The largest total by default.
    /// p : float, optional
    ///     The capital standard: assets at the total's lower ``p`` quantile.
    /// allocation : {"linear", "lifted"}, default "linear"
    ///     The unit share of the assets above the asset level: expected
    ///     (linear) or distorted (lifted).
    ///
    /// Returns
    /// -------
    /// NaturalPrice
    #[pyo3(signature = (distortion, assets=None, p=None, allocation="linear"))]
    fn price(
        &self,
        distortion: PyRef<'_, PyDistortion>,
        assets: Option<f64>,
        p: Option<f64>,
        allocation: &str,
    ) -> PyResult<PyNaturalPrice> {
        let a = self.level(assets, p)?;
        let method = self::allocation(allocation)?;
        Ok(PyNaturalPrice {
            inner: self
                .inner
                .price(&distortion.inner, a, method)
                .map_err(to_py)?,
        })
    }

    /// The member of a distortion family that prices the loss capped at
    /// the assets at a target: give exactly one of ``premium``,
    /// ``return_on_capital`` and ``loss_ratio``.
    ///
    /// Parameters
    /// ----------
    /// family : str
    ///     As for ``prospicio.risk.calibrate``.
    /// assets : float, optional
    /// p : float, optional
    /// premium, return_on_capital, loss_ratio : float, optional
    /// r0 : float, default 0.0
    ///
    /// Returns
    /// -------
    /// Distortion
    #[pyo3(signature = (family, assets=None, p=None, *, premium=None, return_on_capital=None,
                        loss_ratio=None, r0=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn calibrate(
        &self,
        family: &str,
        assets: Option<f64>,
        p: Option<f64>,
        premium: Option<f64>,
        return_on_capital: Option<f64>,
        loss_ratio: Option<f64>,
        r0: f64,
    ) -> PyResult<PyDistortion> {
        let a = self.level(assets, p)?;
        let target = match (premium, return_on_capital, loss_ratio) {
            (Some(v), None, None) => Target::Premium(v),
            (None, Some(v), None) => Target::ReturnOnCapital(v),
            (None, None, Some(v)) => Target::LossRatio(v),
            _ => {
                return Err(PyValueError::new_err(
                    "give exactly one of premium, return_on_capital and loss_ratio",
                ));
            }
        };
        let family = family_from(family, r0)?;
        Ok(PyDistortion {
            inner: self.inner.calibrate(family, a, target).map_err(to_py)?,
        })
    }

    /// Bodoff's percentile layer of capital: each unit's share of the
    /// assets.
    ///
    /// Parameters
    /// ----------
    /// assets : float, optional
    /// p : float, optional
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (assets=None, p=None))]
    fn bodoff(&self, assets: Option<f64>, p: Option<f64>) -> PyResult<Vec<f64>> {
        Ok(self.inner.bodoff(self.level(assets, p)?))
    }

    /// The expected policyholder deficit ratio with these assets: in
    /// total, and by unit under equal priority.
    ///
    /// Parameters
    /// ----------
    /// assets : float
    ///
    /// Returns
    /// -------
    /// tuple of (float, list of float)
    fn epd(&self, assets: f64) -> (f64, Vec<f64>) {
        self.inner.epd(assets)
    }

    /// The range of each unit's premium over every distortion that prices
    /// the total, capped at the assets, at ``premium`` (linear allocation).
    /// The extremes are BiTVaR distortions, found exactly.
    ///
    /// Parameters
    /// ----------
    /// premium : float
    /// assets : float, optional
    /// p : float, optional
    ///
    /// Returns
    /// -------
    /// list of dict
    ///     One per unit: ``unit``, ``lower``, ``upper``, and the Distortion
    ///     giving each, ``lower_distortion`` and ``upper_distortion``.
    #[pyo3(signature = (premium, assets=None, p=None))]
    fn premium_bounds<'py>(
        &self,
        py: Python<'py>,
        premium: f64,
        assets: Option<f64>,
        p: Option<f64>,
    ) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
        let a = self.level(assets, p)?;
        let bounds = self.inner.premium_bounds(premium, a).map_err(to_py)?;
        bounds
            .into_iter()
            .zip(self.inner.units())
            .map(|(b, unit)| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("unit", unit)?;
                d.set_item("lower", b.lower)?;
                d.set_item("upper", b.upper)?;
                d.set_item(
                    "lower_distortion",
                    PyDistortion {
                        inner: b.lower_distortion,
                    },
                )?;
                d.set_item(
                    "upper_distortion",
                    PyDistortion {
                        inner: b.upper_distortion,
                    },
                )?;
                Ok(d)
            })
            .collect()
    }

    /// The smallest assets whose total EPD ratio is at most ``epd``.
    ///
    /// Parameters
    /// ----------
    /// epd : float
    ///
    /// Returns
    /// -------
    /// float
    fn assets_for_epd(&self, epd: f64) -> PyResult<f64> {
        self.inner.assets_for_epd(epd).map_err(to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Portfolio(units={:?}, totals={})",
            self.inner.units(),
            self.inner.totals().len()
        )
    }
}

fn principle(name: &str, loading: f64, q: f64) -> PyResult<Principle> {
    Ok(match kind(name, q)? {
        Kind::ExpectedValue => Principle::ExpectedValue(loading),
        Kind::Variance => Principle::Variance(loading),
        Kind::StandardDeviation => Principle::StandardDeviation(loading),
        Kind::SemiVariance => Principle::SemiVariance(loading),
        Kind::Exponential => Principle::Exponential(loading),
        Kind::Esscher => Principle::Esscher(loading),
        Kind::Dutch => Principle::Dutch(loading),
        Kind::Fischer { q } => Principle::Fischer { theta: loading, q },
        Kind::Var => Principle::Var(loading),
    })
}

fn kind(name: &str, q: f64) -> PyResult<Kind> {
    Ok(match name {
        "expected_value" => Kind::ExpectedValue,
        "variance" => Kind::Variance,
        "standard_deviation" => Kind::StandardDeviation,
        "semi_variance" => Kind::SemiVariance,
        "exponential" => Kind::Exponential,
        "esscher" => Kind::Esscher,
        "dutch" => Kind::Dutch,
        "fischer" => Kind::Fischer { q },
        "var" => Kind::Var,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown premium principle {other:?}"
            )));
        }
    })
}

fn loading_of(p: Principle) -> f64 {
    match p {
        Principle::ExpectedValue(t)
        | Principle::Variance(t)
        | Principle::StandardDeviation(t)
        | Principle::SemiVariance(t)
        | Principle::Exponential(t)
        | Principle::Esscher(t)
        | Principle::Dutch(t)
        | Principle::Var(t) => t,
        Principle::Fischer { theta, .. } => theta,
    }
}

/// The premium of a distribution under a classical premium principle.
///
/// Parameters
/// ----------
/// principle : str
///     ``"expected_value"`` (``(1 + t) mean``), ``"variance"``
///     (``mean + t var``), ``"standard_deviation"``, ``"semi_variance"``
///     (``mean + t E[(X - mean)+**2]``), ``"exponential"``
///     (``log E[exp(t X)] / t``), ``"esscher"``
///     (``E[X exp(t X)] / E[exp(t X)]``), ``"dutch"``
///     (``mean + t E[(X - mean)+]``), ``"fischer"``
///     (``mean + t E[(X - mean)+**q]**(1/q)``) or ``"var"`` (the lower ``t``
///     quantile).
/// dist : Sampled, Grid or PredictiveDistribution
///     A predictive distribution is priced on its total.
/// loading : float
/// q : float, default 2.0
///     The Fischer power.
///
/// Returns
/// -------
/// float
///
/// Examples
/// --------
/// >>> from prospicio.distributions import Sampled
/// >>> from prospicio.pricing import classical_premium
/// >>> classical_premium("standard_deviation", Sampled([0.0, 10.0]), 0.2)
/// 6.0
#[pyfunction]
#[pyo3(signature = (principle, dist, loading, q=2.0))]
pub(crate) fn classical_premium(
    principle: &str,
    dist: &Bound<'_, PyAny>,
    loading: f64,
    q: f64,
) -> PyResult<f64> {
    let (x, p) = discrete_of(dist, None)?;
    self::principle(principle, loading, q)?
        .premium(&x, &p)
        .map_err(to_py)
}

/// The loading of a classical premium principle that gives ``premium``;
/// see ``classical_premium``.
///
/// Parameters
/// ----------
/// principle : str
/// dist : Sampled, Grid or PredictiveDistribution
/// premium : float
/// q : float, default 2.0
///
/// Returns
/// -------
/// float
#[pyfunction]
#[pyo3(signature = (principle, dist, premium, q=2.0))]
pub(crate) fn calibrate_classical(
    principle: &str,
    dist: &Bound<'_, PyAny>,
    premium: f64,
    q: f64,
) -> PyResult<f64> {
    let (x, p) = discrete_of(dist, None)?;
    let p = classical::calibrate(kind(principle, q)?, &x, &p, premium).map_err(to_py)?;
    Ok(loading_of(p))
}
