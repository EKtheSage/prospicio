//! `prospicio.credibility`: classical credibility, the limited fluctuation
//! standards, the Bühlmann–Straub model and Hachemeister's regression
//! model (Aggregate lane), and Bayesian
//! hierarchical credibility (Models lane; `docs/design/credibility.md`).

use prospicio_bayes::credibility::{
    CredibilityPriors, HierarchicalCredibility, HierarchicalCredibilityFit, Sampler, ScalePrior,
};
use prospicio_pricing::credibility::{
    BuhlmannStraub, BuhlmannStraubFit, Hachemeister, HachemeisterFit, LimitedFluctuation,
    StructureEstimator,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::distributions::PyPredictiveDistribution;
use crate::to_py;

/// Limited fluctuation (classical) credibility: the expected number of
/// claims for the observed value to lie within ``k`` of its mean with
/// probability ``p``, under a Poisson count and the normal approximation.
///
/// With ``z = Phi^-1((1 + p) / 2)`` and ``n_F = (z / k)**2``, the full
/// standards in claims are ``n_F`` for the frequency, ``n_F * cv**2`` for
/// the severity and ``n_F * (1 + cv**2)`` for the pure premium; partial
/// credibility is ``min(1, sqrt(n / standard))``.
///
/// Parameters
/// ----------
/// p : float, default 0.90
/// k : float, default 0.05
///
/// Raises
/// ------
/// ValueError
///     If ``p`` is not in ``(0, 1)`` or ``k`` is not positive.
///
/// Examples
/// --------
/// >>> from prospicio.credibility import LimitedFluctuation
/// >>> lf = LimitedFluctuation(0.90, 0.05)
/// >>> round(lf.frequency_standard())
/// 1082
/// >>> lf.p, lf.k
/// (0.9, 0.05)
#[pyclass(name = "LimitedFluctuation", module = "prospicio.credibility", frozen)]
pub(crate) struct PyLimitedFluctuation {
    inner: LimitedFluctuation,
}

#[pymethods]
impl PyLimitedFluctuation {
    #[new]
    #[pyo3(signature = (p = 0.90, k = 0.05))]
    fn new(p: f64, k: f64) -> PyResult<Self> {
        Ok(Self {
            inner: LimitedFluctuation::new(p, k).map_err(to_py)?,
        })
    }

    /// The probability ``p`` of lying within ``k`` of the mean.
    #[getter]
    fn p(&self) -> f64 {
        self.inner.p()
    }

    /// The tolerance ``k``, relative to the mean.
    #[getter]
    fn k(&self) -> f64 {
        self.inner.k()
    }

    /// Full standard for the frequency, in expected claims.
    ///
    /// Returns
    /// -------
    /// float
    fn frequency_standard(&self) -> f64 {
        self.inner.frequency_standard()
    }

    /// Full standard for the severity, in claims.
    ///
    /// Parameters
    /// ----------
    /// cv : float
    ///     The severity's coefficient of variation.
    ///
    /// Returns
    /// -------
    /// float
    fn severity_standard(&self, cv: f64) -> f64 {
        self.inner.severity_standard(cv)
    }

    /// Full standard for the pure premium, in expected claims.
    ///
    /// Parameters
    /// ----------
    /// cv : float
    ///     The severity's coefficient of variation.
    ///
    /// Returns
    /// -------
    /// float
    fn pure_premium_standard(&self, cv: f64) -> f64 {
        self.inner.pure_premium_standard(cv)
    }

    /// Credibility ``min(1, sqrt(n / standard))`` of ``n`` claims.
    ///
    /// Parameters
    /// ----------
    /// n : float
    /// standard : float
    ///
    /// Returns
    /// -------
    /// float
    fn credibility(&self, n: f64, standard: f64) -> f64 {
        self.inner.credibility(n, standard)
    }

    fn __repr__(&self) -> String {
        format!(
            "LimitedFluctuation(p={}, k={}, frequency_standard={:.2})",
            self.inner.p(),
            self.inner.k(),
            self.inner.frequency_standard()
        )
    }
}

/// A fitted Bühlmann–Straub model, from ``buhlmann_straub``.
#[pyclass(name = "BuhlmannStraubFit", module = "prospicio.credibility", frozen)]
pub(crate) struct PyBuhlmannStraubFit {
    inner: BuhlmannStraubFit,
}

#[pymethods]
impl PyBuhlmannStraubFit {
    /// The credibility-weighted collective mean.
    #[getter]
    fn collective_mean(&self) -> f64 {
        self.inner.collective_mean
    }

    /// The between-group variance ``a`` (may be negative; the factors are
    /// then 0).
    #[getter]
    fn between_variance(&self) -> f64 {
        self.inner.between_variance
    }

    /// The within-group variance ``s2``.
    #[getter]
    fn within_variance(&self) -> f64 {
        self.inner.within_variance
    }

    /// Bühlmann's ``k = s2 / a`` (``inf`` when ``a <= 0``).
    #[getter]
    fn k(&self) -> f64 {
        self.inner.k()
    }

    /// Each group's weighted mean.
    #[getter]
    fn group_means(&self) -> Vec<f64> {
        self.inner.group_means.clone()
    }

    /// Each group's total weight.
    #[getter]
    fn group_weights(&self) -> Vec<f64> {
        self.inner.group_weights.clone()
    }

    /// Each group's credibility factor.
    #[getter]
    fn credibility(&self) -> Vec<f64> {
        self.inner.credibility.clone()
    }

    /// Each group's credibility premium.
    #[getter]
    fn premiums(&self) -> Vec<f64> {
        self.inner.premiums.clone()
    }

    /// Iterations of the iterative estimator (0 for the unbiased one).
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "BuhlmannStraubFit(collective_mean={:?}, between_variance={:?}, within_variance={:?})",
            self.inner.collective_mean, self.inner.between_variance, self.inner.within_variance
        )
    }
}

/// The Bühlmann–Straub credibility model, as R's ``actuar::cm(~group)``:
/// group ``i`` in period ``j`` has ratio ``X_ij`` (loss per unit of
/// exposure) with weight ``w_ij``. Without weights it is Bühlmann's model.
///
/// Parameters
/// ----------
/// ratios : list of list of float
///     One row per group, one column per period; NaN for a missing cell.
/// weights : list of list of float, optional
///     The same shape, NaN in the same places; equal weights if omitted.
/// estimator : {"unbiased", "iterative"}, default "unbiased"
///     The between variance: Bühlmann and Straub's unbiased estimator
///     (``actuar``'s ``"Buhlmann-Gisler"`` and ``"Ohlsson"``), or Bichsel
///     and Straub's iterative one.
/// tol : float, default 1.49e-8
///     Relative tolerance of the iterative estimator.
/// max_iter : int, default 100
///
/// Returns
/// -------
/// BuhlmannStraubFit
///
/// Raises
/// ------
/// ValueError
///     For fewer than two groups, too few observations, mismatched shapes
///     or missing cells, or missing ratios without weights.
///
/// Examples
/// --------
/// >>> from prospicio.credibility import buhlmann_straub
/// >>> fit = buhlmann_straub([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0]])
/// >>> fit.within_variance
/// 4.0
#[pyfunction]
#[pyo3(signature = (ratios, weights = None, estimator = "unbiased", tol = 1.4901161193847656e-8, max_iter = 100))]
pub(crate) fn buhlmann_straub(
    ratios: Vec<Vec<f64>>,
    weights: Option<Vec<Vec<f64>>>,
    estimator: &str,
    tol: f64,
    max_iter: usize,
) -> PyResult<PyBuhlmannStraubFit> {
    let inner = BuhlmannStraub::new(structure_estimator(estimator, tol, max_iter)?)
        .fit(&ratios, weights.as_deref())
        .map_err(to_py)?;
    Ok(PyBuhlmannStraubFit { inner })
}

fn structure_estimator(estimator: &str, tol: f64, max_iter: usize) -> PyResult<StructureEstimator> {
    match estimator {
        "unbiased" => Ok(StructureEstimator::Unbiased),
        "iterative" => Ok(StructureEstimator::Iterative { tol, max_iter }),
        other => Err(PyValueError::new_err(format!(
            "estimator must be \"unbiased\" or \"iterative\", got {other:?}"
        ))),
    }
}

/// A fitted Hachemeister model, from ``hachemeister``.
#[pyclass(name = "HachemeisterFit", module = "prospicio.credibility", frozen)]
pub(crate) struct PyHachemeisterFit {
    inner: HachemeisterFit,
}

#[pymethods]
impl PyHachemeisterFit {
    /// The collective coefficients, one per design column.
    #[getter]
    fn collective(&self) -> Vec<f64> {
        self.inner.collective.clone()
    }

    /// Each risk's credibility coefficients (the collective for a risk
    /// with no weight).
    #[getter]
    fn coefficients(&self) -> Vec<Vec<f64>> {
        self.inner.coefficients.clone()
    }

    /// Each risk's own weighted least squares coefficients (0 for a risk
    /// with no weight).
    #[getter]
    fn individual(&self) -> Vec<Vec<f64>> {
        self.inner.individual.clone()
    }

    /// Each risk's credibility matrix, in the basis fitted (the
    /// orthogonalised design when the intercept is adjusted).
    #[getter]
    fn credibility(&self) -> Vec<Vec<Vec<f64>>> {
        self.inner.credibility.clone()
    }

    /// The between-risk covariance of the coefficients, in the basis
    /// fitted (diagonal when the intercept is adjusted).
    #[getter]
    fn between(&self) -> Vec<Vec<f64>> {
        self.inner.between.clone()
    }

    /// The within-risk variance ``s2``, the mean of the risks' residual
    /// variances.
    #[getter]
    fn within_variance(&self) -> f64 {
        self.inner.within_variance
    }

    /// Each risk's total weight.
    #[getter]
    fn risk_weights(&self) -> Vec<f64> {
        self.inner.risk_weights.clone()
    }

    /// When the intercept is adjusted, the upper triangular ``R`` with the
    /// design ``Y = Q R``: a coefficient ``c`` in the basis fitted is
    /// ``R^-1 c`` in the design's. ``None`` at the origin.
    #[getter]
    fn transition(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.transition.clone()
    }

    /// Iterations of the iterative estimator.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    /// Each risk's credibility premium at the design row ``x``.
    ///
    /// Parameters
    /// ----------
    /// x : list of float
    ///     One value per design column, such as ``[1.0, 13.0]`` for the
    ///     next quarter of a trend line over twelve.
    ///
    /// Returns
    /// -------
    /// list of float
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``x`` does not have one value per design column.
    fn predict(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.predict(&x).map_err(to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "HachemeisterFit(collective={:?}, within_variance={:?})",
            self.inner.collective, self.inner.within_variance
        )
    }
}

/// Hachemeister's regression credibility model, as R's ``actuar::cm(~risk,
/// regformula = ...)``: risk ``i`` in period ``t`` has ratio ``X_it`` with
/// weight ``w_it`` and a mean ``design[t] @ beta_i`` on a design shared by
/// the risks (a trend line ``[1, t]``, say). Each risk's own weighted least
/// squares fit is shrunk towards the collective by a credibility matrix.
///
/// At the origin the matrices are full and the between covariance is
/// Hachemeister's iterative estimator whatever ``estimator`` says (as
/// ``actuar``; on near-collinear coefficients its iteration may wander in
/// the last digits). With ``adjust_intercept`` (``actuar``'s
/// ``adj.intercept``) the design is orthogonalised under the collective
/// weights, the matrices are diagonal and each diagonal is estimated as
/// Bühlmann–Straub's by ``estimator``; the design's first column must
/// then be constant.
///
/// Parameters
/// ----------
/// ratios : list of list of float
///     One row per risk, one column per period; NaN for a missing cell.
/// design : list of list of float
///     One row per period, one column per coefficient.
/// weights : list of list of float, optional
///     The shape of ``ratios``, NaN in the same places; equal if omitted.
/// estimator : {"unbiased", "iterative"}, default "unbiased"
/// adjust_intercept : bool, default False
/// tol : float, default 1.49e-8
///     Relative tolerance of the iterations.
/// max_iter : int, default 100
///
/// Returns
/// -------
/// HachemeisterFit
///
/// Raises
/// ------
/// ValueError
///     For fewer than two risks with weight, mismatched shapes or missing
///     cells, a risk whose observed cells do not identify its coefficients,
///     linearly dependent design columns, or a singular system.
///
/// Examples
/// --------
/// >>> from prospicio.credibility import hachemeister
/// >>> x = [[10.0, 12.0, 13.0, 15.0, 16.0],
/// ...      [8.0, 8.5, 9.5, 9.0, 10.0],
/// ...      [14.0, 13.0, 16.0, 17.0, 19.0]]
/// >>> design = [[1.0, float(t)] for t in range(1, 6)]
/// >>> fit = hachemeister(x, design, adjust_intercept=True)
/// >>> nxt = fit.predict([1.0, 6.0])
/// >>> nxt[1] < nxt[0] < nxt[2]
/// True
#[pyfunction]
#[pyo3(signature = (ratios, design, weights = None, estimator = "unbiased", adjust_intercept = false, tol = 1.4901161193847656e-8, max_iter = 100))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn hachemeister(
    py: Python<'_>,
    ratios: Vec<Vec<f64>>,
    design: Vec<Vec<f64>>,
    weights: Option<Vec<Vec<f64>>>,
    estimator: &str,
    adjust_intercept: bool,
    tol: f64,
    max_iter: usize,
) -> PyResult<PyHachemeisterFit> {
    let model = Hachemeister::new(
        structure_estimator(estimator, tol, max_iter)?,
        adjust_intercept,
    );
    let inner = py
        .detach(|| model.fit(&ratios, weights.as_deref(), &design))
        .map_err(to_py)?;
    Ok(PyHachemeisterFit { inner })
}

/// A prior on a standard deviation from Python: ``("half_normal", scale)``
/// or ``("half_cauchy", scale)``.
fn scale_prior(p: (String, f64)) -> PyResult<ScalePrior> {
    match p.0.as_str() {
        "half_normal" => Ok(ScalePrior::HalfNormal(p.1)),
        "half_cauchy" => Ok(ScalePrior::HalfCauchy(p.1)),
        other => Err(PyValueError::new_err(format!(
            "a scale prior is (\"half_normal\" or \"half_cauchy\", scale), got {other:?}"
        ))),
    }
}

fn scale_prior_tuple(p: &ScalePrior) -> (&'static str, f64) {
    match *p {
        ScalePrior::HalfNormal(s) => ("half_normal", s),
        ScalePrior::HalfCauchy(s) => ("half_cauchy", s),
    }
}

/// Bayesian hierarchical credibility, sampled with NUTS: Bühlmann–Straub's
/// model, or Jewell's hierarchy of it, as a normal random-effects model.
///
/// Group ``i`` in period ``j`` has ratio ``X_ij ~ N(theta_i, sigma**2 /
/// w_ij)``; each group's risk premium ``theta_i`` is normal about its node
/// at the level above (``levels``), each node about its own parent, and
/// the top about the collective mean ``mu``. The means are integrated out
/// exactly, so NUTS samples only the standard deviations; given them, each
/// premium is the credibility formula ``Z * M + (1 - Z) * E[parent]``, and
/// the Bayesian premium averages it over their posterior.
///
/// Parameters
/// ----------
/// ratios : list of list of float
///     One row per group, one column per period; NaN for a missing cell.
/// weights : list of list of float
///     The same shape, NaN in the same places.
/// levels : list of list of int, optional
///     The hierarchy above the groups, outermost level first:
///     ``levels[k][i]`` is group ``i``'s node (from 0) at level ``k``.
///     Omitted for Bühlmann–Straub's model.
/// collective_mean : (float, float), optional
///     ``(mean, sd)`` of a normal prior on ``mu``; flat if omitted.
/// within_prior : (str, float), optional
///     ``("half_normal", scale)`` or ``("half_cauchy", scale)`` on
///     ``sigma``; half-normal at five times the pooled within standard
///     deviation if omitted.
/// between_priors : list of (str, float), optional
///     One per level, outermost first and the groups' last; half-normal at
///     five times each level's spread if omitted.
/// chains, tune, draws, seed : int
///     NUTS settings.
///
/// Returns
/// -------
/// HierarchicalCredibilityFit
///
/// Examples
/// --------
/// >>> from prospicio.credibility import hierarchical_credibility
/// >>> x = [[1.0, 2.0, 3.0, 2.0], [4.0, 5.0, 6.0, 5.0], [2.0, 3.0, 4.0, 3.0]]
/// >>> fit = hierarchical_credibility(x, [[1.0] * 4] * 3, chains=2, tune=300, draws=300)
/// >>> fit.collective_mean < fit.premiums[1] < 5.0
/// True
#[pyfunction]
#[pyo3(signature = (ratios, weights, levels = None, collective_mean = None, within_prior = None, between_priors = None, chains = 4, tune = 1000, draws = 1000, seed = 0))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn hierarchical_credibility(
    py: Python<'_>,
    ratios: Vec<Vec<f64>>,
    weights: Vec<Vec<f64>>,
    levels: Option<Vec<Vec<usize>>>,
    collective_mean: Option<(f64, f64)>,
    within_prior: Option<(String, f64)>,
    between_priors: Option<Vec<(String, f64)>>,
    chains: usize,
    tune: usize,
    draws: usize,
    seed: u64,
) -> PyResult<PyHierarchicalCredibilityFit> {
    let priors = CredibilityPriors {
        collective_mean,
        within: within_prior.map(scale_prior).transpose()?,
        between: between_priors
            .unwrap_or_default()
            .into_iter()
            .map(scale_prior)
            .collect::<PyResult<Vec<_>>>()?,
    };
    let spec = HierarchicalCredibility::default()
        .priors(priors)
        .sampler(Sampler {
            chains,
            tune,
            draws,
            seed,
            ..Sampler::default()
        });
    let levels = levels.unwrap_or_default();
    let inner = py
        .detach(|| spec.fit(&ratios, &weights, &levels))
        .map_err(to_py)?;
    Ok(PyHierarchicalCredibilityFit { inner })
}

/// The credibility premiums at given structure parameters: Bühlmann–Straub's
/// with one level, Bühlmann–Gisler's hierarchical ones (as ``actuar::cm``)
/// with more, which are the posterior means of the normal model with the
/// variances known.
///
/// Parameters
/// ----------
/// ratios, weights, levels
///     As for ``hierarchical_credibility``.
/// within_variance : float
///     ``s**2``.
/// between_variances : list of float
///     One per level, outermost first and the groups' last (``a`` with one
///     level; ``b`` then ``a`` with two).
/// collective_mean : (float, float), optional
///     ``(mean, sd)`` of a normal prior on ``mu``; flat if omitted.
///
/// Returns
/// -------
/// dict
///     ``premiums``, ``credibility``, ``level_premiums``,
///     ``level_credibility`` and ``collective_mean``.
///
/// Examples
/// --------
/// >>> from prospicio.credibility import credibility_premiums
/// >>> p = credibility_premiums([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0]], [[1.0] * 3] * 2,
/// ...                          within_variance=4.0, between_variances=[8.0 - 4.0 / 3.0])
/// >>> round(p["collective_mean"], 12)
/// 4.0
#[pyfunction]
#[pyo3(signature = (ratios, weights, within_variance, between_variances, levels = None, collective_mean = None))]
pub(crate) fn credibility_premiums<'py>(
    py: Python<'py>,
    ratios: Vec<Vec<f64>>,
    weights: Vec<Vec<f64>>,
    within_variance: f64,
    between_variances: Vec<f64>,
    levels: Option<Vec<Vec<usize>>>,
    collective_mean: Option<(f64, f64)>,
) -> PyResult<Bound<'py, PyDict>> {
    let p = prospicio_bayes::credibility::credibility_premiums(
        &ratios,
        &weights,
        &levels.unwrap_or_default(),
        within_variance,
        &between_variances,
        collective_mean,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("premiums", p.premiums)?;
    d.set_item("credibility", p.credibility)?;
    d.set_item("level_premiums", p.level_premiums)?;
    d.set_item("level_credibility", p.level_credibility)?;
    d.set_item("collective_mean", p.collective_mean)?;
    Ok(d)
}

/// A fitted Bayesian hierarchical credibility model, from
/// ``hierarchical_credibility``.
#[pyclass(
    name = "HierarchicalCredibilityFit",
    module = "prospicio.credibility",
    frozen
)]
pub(crate) struct PyHierarchicalCredibilityFit {
    inner: HierarchicalCredibilityFit,
}

#[pymethods]
impl PyHierarchicalCredibilityFit {
    /// Posterior mean of each group's risk premium.
    #[getter]
    fn premiums(&self) -> Vec<f64> {
        self.inner.premiums.clone()
    }

    /// Posterior mean of each group's credibility factor.
    #[getter]
    fn credibility(&self) -> Vec<f64> {
        self.inner.credibility.clone()
    }

    /// Posterior mean of each node's mean, level by level (outermost first).
    #[getter]
    fn level_premiums(&self) -> Vec<Vec<f64>> {
        self.inner.level_premiums.clone()
    }

    /// Posterior mean of the collective mean.
    #[getter]
    fn collective_mean(&self) -> f64 {
        self.inner.collective_mean
    }

    /// Each group's weighted mean.
    #[getter]
    fn group_means(&self) -> Vec<f64> {
        self.inner.group_means.clone()
    }

    /// Each group's total weight.
    #[getter]
    fn group_weights(&self) -> Vec<f64> {
        self.inner.group_weights.clone()
    }

    /// The priors used, with the defaults filled in: a dict with
    /// ``collective_mean``, ``within`` and ``between``.
    #[getter]
    fn priors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let p = &self.inner.priors;
        let d = PyDict::new(py);
        d.set_item("collective_mean", p.collective_mean)?;
        d.set_item("within", p.within.as_ref().map(scale_prior_tuple))?;
        d.set_item(
            "between",
            p.between.iter().map(scale_prior_tuple).collect::<Vec<_>>(),
        )?;
        Ok(d)
    }

    /// Divergent transitions over all chains.
    #[getter]
    fn divergences(&self) -> usize {
        self.inner.divergences()
    }

    /// Posterior summaries of ``within_variance`` and
    /// ``between_variance[l]`` (outermost level first): one dict each with
    /// ``name``, ``mean``, ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``,
    /// ``ess_bulk`` and ``ess_tail``.
    ///
    /// Returns
    /// -------
    /// list of dict
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .summary()
            .map_err(to_py)?
            .into_iter()
            .map(|s| {
                let d = PyDict::new(py);
                d.set_item("name", s.name)?;
                d.set_item("mean", s.mean)?;
                d.set_item("sd", s.sd)?;
                d.set_item("q05", s.q05)?;
                d.set_item("q50", s.q50)?;
                d.set_item("q95", s.q95)?;
                d.set_item("rhat", s.rhat)?;
                d.set_item("ess_bulk", s.ess_bulk)?;
                d.set_item("ess_tail", s.ess_tail)?;
                Ok(d)
            })
            .collect()
    }

    /// Posterior predictive draws by group (dimension ``group``): with
    /// ``weights``, each group's next-period ratio ``N(theta_i, sigma**2 /
    /// w_i)``; without, its risk premium ``theta_i``.
    ///
    /// Parameters
    /// ----------
    /// weights : list of float, optional
    ///     Each group's weight next period.
    /// n_sims : int, default 10000
    /// seed : int, default 0
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution
    #[pyo3(signature = (weights = None, n_sims = 10_000, seed = 0))]
    fn predictive(
        &self,
        py: Python<'_>,
        weights: Option<Vec<f64>>,
        n_sims: usize,
        seed: u64,
    ) -> PyResult<PyPredictiveDistribution> {
        let fit = &self.inner;
        let inner = py
            .detach(|| fit.predictive(weights.as_deref(), n_sims, seed))
            .map_err(to_py)?;
        Ok(PyPredictiveDistribution { inner })
    }
}
