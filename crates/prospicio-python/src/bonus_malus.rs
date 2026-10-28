//! `prospicio.pricing.BonusMalus` (Aggregate lane), over
//! `prospicio_pricing::bonus_malus`.

use prospicio_pricing::bonus_malus::{BonusMalus, BonusRule, Portfolio};
use prospicio_prob::count_families::Mixing;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::to_py;

/// A bonus-malus or no-claim-discount scale as a Markov chain.
///
/// A policyholder sits on one of the levels ``0 .. levels - 1``, 0 the
/// best, and moves each year by the number of claims. Under Poisson claims
/// with mean ``lam`` the levels are a Markov chain; over a mixed Poisson
/// portfolio (a priori classes with frequencies ``lambdas`` and a residual
/// heterogeneity ``Theta`` of mean 1, gamma or inverse Gaussian with
/// coefficient of variation ``cv``) the scale has a stationary distribution
/// of levels and Norberg's optimal relativities, ``E[Theta | level]``,
/// which balance to 1 over it.
///
/// Build one with ``BonusMalus.rule`` or ``BonusMalus.table``.
///
/// Examples
/// --------
/// >>> from prospicio.pricing import BonusMalus
/// >>> # -1/+top on six levels: any claim sends the driver to level 5.
/// >>> scale = BonusMalus.rule(6, 5, down=1, up_per_claim=5)
/// >>> r = scale.optimal_relativities(0.1, cv=1.0)
/// >>> round(r[0], 6), round(r[5], 6)
/// (0.666667, 1.909091)
/// >>> pi = scale.stationary(0.1, cv=1.0)
/// >>> round(sum(p * x for p, x in zip(pi, r)), 12)
/// 1.0
#[pyclass(name = "BonusMalus", module = "prospicio.pricing", frozen)]
pub(crate) struct PyBonusMalus {
    inner: BonusMalus,
}

fn portfolio(
    lambdas: &Bound<'_, PyAny>,
    cv: f64,
    mixing: &str,
    weights: Option<Vec<f64>>,
) -> PyResult<Portfolio> {
    let lambdas: Vec<f64> = match lambdas.extract::<f64>() {
        Ok(l) => vec![l],
        Err(_) => lambdas.extract()?,
    };
    let weights = weights.unwrap_or_else(|| vec![1.0; lambdas.len()]);
    let mixing = match mixing {
        "gamma" => Mixing::Gamma { cv },
        "inverse_gaussian" | "ig" => Mixing::InverseGaussian { cv },
        other => {
            return Err(PyValueError::new_err(format!(
                "mixing must be \"gamma\" or \"inverse_gaussian\", not {other:?}"
            )));
        }
    };
    Portfolio::classes(weights, lambdas, mixing).map_err(to_py)
}

#[pymethods]
impl PyBonusMalus {
    /// A scale of ``levels`` levels, newcomers on ``start``, moving down
    /// ``down`` levels after a claim-free year and up ``up_per_claim``
    /// levels for each claim, capped at both ends. The -1/+top scale is
    /// ``down=1, up_per_claim=levels - 1``.
    ///
    /// Parameters
    /// ----------
    /// levels : int
    ///     At least 2.
    /// start : int
    /// down : int, default 1
    /// up_per_claim : int
    ///     At least 1.
    ///
    /// Returns
    /// -------
    /// BonusMalus
    #[staticmethod]
    #[pyo3(signature = (levels, start, *, down = 1, up_per_claim))]
    fn rule(levels: usize, start: usize, down: usize, up_per_claim: usize) -> PyResult<Self> {
        Ok(Self {
            inner: BonusMalus::rule(levels, start, BonusRule { down, up_per_claim })
                .map_err(to_py)?,
        })
    }

    /// A scale from its table: ``table[l][k]`` is the level after ``k``
    /// claims from level ``l``, one row per level, every row the same
    /// length; the last column is for that many claims or more.
    ///
    /// Parameters
    /// ----------
    /// start : int
    /// table : list of list of int
    ///
    /// Returns
    /// -------
    /// BonusMalus
    #[staticmethod]
    fn table(start: usize, table: Vec<Vec<usize>>) -> PyResult<Self> {
        Ok(Self {
            inner: BonusMalus::table(start, table).map_err(to_py)?,
        })
    }

    /// Number of levels.
    #[getter]
    fn levels(&self) -> usize {
        self.inner.levels()
    }

    /// The level a newcomer starts on.
    #[getter]
    fn start(&self) -> usize {
        self.inner.start()
    }

    /// The table of next levels by level and claim count.
    #[getter]
    fn transitions(&self) -> Vec<Vec<usize>> {
        self.inner.transitions().to_vec()
    }

    /// The level after ``claims`` claims from ``level``.
    ///
    /// Parameters
    /// ----------
    /// level : int
    /// claims : int
    ///
    /// Returns
    /// -------
    /// int
    fn next_level(&self, level: usize, claims: usize) -> PyResult<usize> {
        if level >= self.inner.levels() {
            return Err(PyValueError::new_err(format!(
                "level {level} is not on a scale of {} levels",
                self.inner.levels()
            )));
        }
        Ok(self.inner.next_level(level, claims))
    }

    /// The one-year transition matrix under Poisson claims with mean
    /// ``lam``: row ``l``, column ``j`` is the probability of moving from
    /// level ``l`` to ``j``.
    ///
    /// Parameters
    /// ----------
    /// lam : float
    ///
    /// Returns
    /// -------
    /// list of list of float
    fn transition_matrix(&self, lam: f64) -> PyResult<Vec<Vec<f64>>> {
        let m = self.inner.transition_matrix(lam).map_err(to_py)?;
        Ok(m.chunks(self.inner.levels()).map(<[f64]>::to_vec).collect())
    }

    /// The stationary distribution of the levels at frequency ``lam``.
    ///
    /// Parameters
    /// ----------
    /// lam : float
    ///
    /// Returns
    /// -------
    /// list of float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If it is not unique (unreachable or absorbing levels).
    fn stationary_at(&self, lam: f64) -> PyResult<Vec<f64>> {
        self.inner.stationary_at(lam).map_err(to_py)
    }

    /// The distribution of the levels after ``years`` years from the start
    /// level, at frequency ``lam``.
    ///
    /// Parameters
    /// ----------
    /// lam : float
    /// years : int
    ///
    /// Returns
    /// -------
    /// list of float
    fn distribution_after(&self, lam: f64, years: usize) -> PyResult<Vec<f64>> {
        self.inner.distribution_after(lam, years).map_err(to_py)
    }

    /// The years from the start level until the distribution of the levels
    /// is within ``tolerance`` of the stationary one in total variation.
    ///
    /// Parameters
    /// ----------
    /// lam : float
    /// tolerance : float, default 0.01
    ///
    /// Returns
    /// -------
    /// int or None
    ///     None past 100,000 years.
    #[pyo3(signature = (lam, tolerance = 0.01))]
    fn years_to_stationarity(&self, lam: f64, tolerance: f64) -> PyResult<Option<usize>> {
        self.inner
            .years_to_stationarity(lam, tolerance)
            .map_err(to_py)
    }

    /// The stationary mean relativity ``sum_l pi_l(lam) r_l``.
    ///
    /// Parameters
    /// ----------
    /// lam : float
    /// relativities : list of float
    ///     One per level.
    ///
    /// Returns
    /// -------
    /// float
    fn mean_relativity(&self, lam: f64, relativities: Vec<f64>) -> PyResult<f64> {
        self.inner
            .mean_relativity(lam, &relativities)
            .map_err(to_py)
    }

    /// Loimaranta's efficiency ``d ln r(lam) / d ln lam`` of the scale with
    /// ``relativities``, exact.
    ///
    /// Parameters
    /// ----------
    /// lam : float
    /// relativities : list of float
    ///
    /// Returns
    /// -------
    /// float
    fn efficiency(&self, lam: f64, relativities: Vec<f64>) -> PyResult<f64> {
        self.inner.efficiency(lam, &relativities).map_err(to_py)
    }

    /// The stationary distribution of the levels over a mixed Poisson
    /// portfolio.
    ///
    /// Parameters
    /// ----------
    /// lambdas : float or list of float
    ///     The a priori classes' frequencies.
    /// cv : float
    ///     The coefficient of variation of ``Theta``.
    /// mixing : {"gamma", "inverse_gaussian"}, default "gamma"
    /// weights : list of float, optional
    ///     The classes' weights; equal by default.
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (lambdas, cv, mixing = "gamma", weights = None))]
    fn stationary(
        &self,
        py: Python<'_>,
        lambdas: &Bound<'_, PyAny>,
        cv: f64,
        mixing: &str,
        weights: Option<Vec<f64>>,
    ) -> PyResult<Vec<f64>> {
        let p = portfolio(lambdas, cv, mixing, weights)?;
        py.detach(|| self.inner.stationary(&p)).map_err(to_py)
    }

    /// Norberg's optimal relativities under quadratic loss,
    /// ``E[Theta | level]`` over the stationary portfolio; a level it never
    /// reaches gets NaN.
    ///
    /// Parameters
    /// ----------
    /// lambdas : float or list of float
    /// cv : float
    /// mixing : {"gamma", "inverse_gaussian"}, default "gamma"
    /// weights : list of float, optional
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (lambdas, cv, mixing = "gamma", weights = None))]
    fn optimal_relativities(
        &self,
        py: Python<'_>,
        lambdas: &Bound<'_, PyAny>,
        cv: f64,
        mixing: &str,
        weights: Option<Vec<f64>>,
    ) -> PyResult<Vec<f64>> {
        let p = portfolio(lambdas, cv, mixing, weights)?;
        py.detach(|| self.inner.optimal_relativities(&p))
            .map_err(to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "BonusMalus(levels={}, start={})",
            self.inner.levels(),
            self.inner.start()
        )
    }
}
