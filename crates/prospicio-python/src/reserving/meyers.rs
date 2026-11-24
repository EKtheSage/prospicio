//! Meyers' models, `Meyers`, and `NewYear`.

use super::*;

fn meyers_model(name: &str) -> PyResult<MeyersModel> {
    MeyersModel::from_name(name)
        .ok_or_else(|| crate::unknown_name("model", name, MeyersModel::NAMES))
}

/// Meyers' (2019, CAS Monograph 8) log-normal models of a cumulative
/// triangle ``C`` with premium ``P``, for origins ``w = 1..n`` and ages
/// ``d = 1..m``: ``log C[w, d]`` is normal with mean
/// ``log P[w] + logelr + alpha[w] + beta[d] g[w]`` (``alpha[1] = 0``,
/// ``beta[m] = 0``) and sd ``sigma[d]``, decreasing with age. ``g[w]`` is 1
/// for the correlated chain ladder without its correlation between origins
/// (``"crc"``) and ``(1 - gamma)^(w - 1)`` for the changing settlement rate
/// (``"csr"``). The correlated accident year model (``"cay"``) keeps
/// ``g[w] = 1`` and adds ``rho`` times the previous origin's residual at
/// the same age, ``log C[w - 1, d]`` less its mean, to the mean of every
/// origin after the first, so neighbouring origins are correlated; Meyers
/// fits it to incurred data. The priors are those of Meyers' Stan code:
/// ``logelr`` normal with mean ``logelr_mean`` and sd ``logelr_sd``,
/// ``alpha`` and ``beta`` normal around 0, ``gamma`` normal around 0 with
/// sd ``gamma_sd``, ``(1 - rho) / 2`` beta with both shapes ``rho_shape``,
/// and each step of ``sigma[d]^2`` uniform on ``(1 - exp(-1e-5), 1)``.
///
/// The ultimate is the value at the oldest age (no tail): each posterior
/// draw gives each open origin a log-normal ultimate; an origin observed at
/// the oldest age keeps its value; under ``"cay"`` the ultimates are drawn
/// origin by origin, each from the previous origin's residual at the oldest
/// age, observed for a closed origin and simulated otherwise.
/// The reserve is the ultimate less the latest value of the fitted column
/// (an outstanding reserve on paid data, IBNR on incurred data) and can be
/// negative. Every observed cell must be positive, and under ``"cay"``
/// every observed cell after the first origin needs the previous origin's
/// cell at the same age.
///
/// One ``seed`` replays a fit exactly, whatever the number of threads: the
/// chains and the simulations each take a seed derived from it.
///
/// Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
/// per chain, any divergent transition) do not fail the fit: ``fit``
/// raises one ``UserWarning`` listing them, and they stay in
/// ``MeyersFit.diagnostics``. With the defaults, all 120 fits of a seed
/// sweep on Meyers' group 353 data (paid and incurred, CRC and CSR, 30
/// seeds) passed every threshold, and CAY on incurred passed at 199 of 200
/// seeds (one divergent transition in 8,000 draws at the other).
///
/// Parameters
/// ----------
/// model : {"crc", "csr", "cay"}, default "crc"
///     The correlated chain ladder without its correlation between origins,
///     the changing settlement rate model, or the correlated accident year
///     model.
/// logelr_mean : float, default -0.4
///     Mean of the normal prior on ``logelr``, the log expected loss ratio.
/// logelr_sd : float, default 3.162
///     Standard deviation of the normal prior on ``logelr`` (3.162 is
///     Meyers' rounding of ``sqrt(10)``).
/// origin_sd : float, default 3.162
///     Standard deviation of the normal prior on each origin effect
///     ``alpha[w]``, ``w >= 2``, centred on 0.
/// development_sd : float, default 3.162
///     Standard deviation of the normal prior on each development effect
///     ``beta[d]``, ``d < m``, centred on 0.
/// gamma_sd : float, default 0.05
///     Standard deviation of the normal prior on ``gamma``, centred on 0
///     (``"csr"`` only).
/// rho_shape : float, default 2.0
///     Both shapes of the beta prior on ``(1 - rho) / 2`` (``"cay"``
///     only); Meyers' 2 makes the prior density of ``rho`` proportional to
///     ``1 - rho^2``.
/// chains, tune, draws : int, default 4, 1000, 2000
///     NUTS chains, warm-up draws per chain and kept draws per chain; with
///     1,000 kept draws the late ``sigma[d]`` sometimes miss the R-hat
///     threshold.
/// n_sims : int, default 10000
///     Number of predictive simulations, independent of the number of
///     posterior draws; positive.
/// seed : int, default 0
///     Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
/// target_accept : float, default 0.95
///     Target mean acceptance rate of the step-size adaptation; at NUTS's
///     usual 0.8 the paid fits of group 353 leave divergent transitions.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
/// dependence : {"independent", "rank_correlation", "synchronized"}, default "independent"
///     How the segments of a multi-segment triangle depend on each other
///     in ``fit_segments`` (``fit`` has one segment and does not use it),
///     as ``BayesOdp``'s: each segment's own posterior and simulations, with
///     seeds derived from ``seed`` and the segment's position; or the
///     segments' simulations reordered as whole rows (Iman–Conover on the
///     segments' total reserves, each segment's ultimates moving with its
///     reserves) to the Spearman matrix ``spearman`` and then put in a
///     random order. ``"synchronized"`` is accepted here, as the bootstraps
///     take it, but ``fit_segments`` raises: a posterior has no residuals to
///     share.
/// spearman : list of list of float, optional
///     With ``dependence="rank_correlation"`` only: Spearman's rho between
///     the segments' total reserves, as ``OdpBootstrap``'s.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero, ``model`` or ``dependence`` is unknown, or
///     ``spearman`` is missing for ``"rank_correlation"``, given for another
///     dependence or not square.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> import warnings
/// >>> from prospicio.reserving import Meyers, Triangle
/// >>> paid = [[952, 1529, 2813, 3647, 3724, 3832], [849, 1564, 2202, 2432, 2468],
/// ...         [983, 2211, 2830, 3832], [1657, 2685, 3169], [932, 1940], [1162]]
/// >>> premium = [5812, 4908, 5454, 5165, 5214, 5230]
/// >>> rows = [(1988 + w, 12 * (d + 1), float(v), float(premium[w]))
/// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
/// >>> origin, age, values, prem = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(origin, age, {"paid": values, "premium": prem})
/// >>> with warnings.catch_warnings():
/// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
/// ...     fit = Meyers("csr", tune=300, draws=300, n_sims=2000, seed=1).fit(
/// ...         tri, "paid", "premium")
/// >>> fit.reserves.components()
/// [('1988',), ('1989',), ('1990',), ('1991',), ('1992',), ('1993',)]
/// >>> fit.parameter_names[0], fit.parameter_names[-1]
/// ('logelr', 'sigma[6]')
/// >>> fit.premium[:2]
/// [5812.0, 4908.0]
#[pyclass(name = "Meyers", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMeyers {
    pub(crate) inner: Meyers,
}

#[pymethods]
impl PyMeyers {
    #[new]
    #[pyo3(signature = (
        model = "crc",
        logelr_mean = -0.4,
        logelr_sd = 3.162,
        origin_sd = 3.162,
        development_sd = 3.162,
        gamma_sd = 0.05,
        rho_shape = 2.0,
        chains = 4,
        tune = 1000,
        draws = 2000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.95,
        max_depth = 10,
        dependence = "independent",
        spearman = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &str,
        logelr_mean: f64,
        logelr_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        gamma_sd: f64,
        rho_shape: f64,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        dependence: &str,
        spearman: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: Meyers {
                model: meyers_model(model)?,
                logelr_mean,
                logelr_sd,
                origin_sd,
                development_sd,
                gamma_sd,
                rho_shape,
                sampler: BayesSampler {
                    chains,
                    tune,
                    draws,
                    seed: 0,
                    target_accept,
                    max_depth,
                },
                n_sims,
                seed,
                dependence: segment_dependence(dependence, spearman)?,
            },
        })
    }

    /// The model: ``"crc"``, ``"csr"`` or ``"cay"``.
    #[getter]
    fn model(&self) -> &'static str {
        self.inner.model.name()
    }

    /// Mean of the prior on ``logelr``.
    #[getter]
    fn logelr_mean(&self) -> f64 {
        self.inner.logelr_mean
    }

    /// Standard deviation of the prior on ``logelr``.
    #[getter]
    fn logelr_sd(&self) -> f64 {
        self.inner.logelr_sd
    }

    /// Standard deviation of each origin effect's prior.
    #[getter]
    fn origin_sd(&self) -> f64 {
        self.inner.origin_sd
    }

    /// Standard deviation of each development effect's prior.
    #[getter]
    fn development_sd(&self) -> f64 {
        self.inner.development_sd
    }

    /// Standard deviation of the prior on ``gamma`` (``"csr"`` only).
    #[getter]
    fn gamma_sd(&self) -> f64 {
        self.inner.gamma_sd
    }

    /// Both shapes of the beta prior on ``(1 - rho) / 2`` (``"cay"`` only).
    #[getter]
    fn rho_shape(&self) -> f64 {
        self.inner.rho_shape
    }

    /// Number of NUTS chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.sampler.chains
    }

    /// Warm-up draws per chain.
    #[getter]
    fn tune(&self) -> usize {
        self.inner.sampler.tune
    }

    /// Kept draws per chain.
    #[getter]
    fn draws(&self) -> usize {
        self.inner.sampler.draws
    }

    /// Number of predictive simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the chains and of the simulations.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Target mean acceptance rate.
    #[getter]
    fn target_accept(&self) -> f64 {
        self.inner.sampler.target_accept
    }

    /// Largest NUTS tree depth.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.sampler.max_depth
    }

    /// How segments depend on each other in ``fit_segments``:
    /// ``"independent"``, ``"rank_correlation"`` or ``"synchronized"``.
    #[getter]
    fn dependence(&self) -> &'static str {
        dependence_name(&self.inner.dependence)
    }

    /// The Spearman matrix of ``dependence="rank_correlation"``; ``None``
    /// otherwise.
    #[getter]
    fn spearman(&self) -> Option<Vec<Vec<f64>>> {
        dependence_spearman(&self.inner.dependence)
    }

    /// Samples the posterior of one measure column of a single-segment
    /// cumulative triangle, with each origin's premium from another column
    /// (its latest value, as ``CapeCod`` reads exposure), and simulates the
    /// ultimates and reserves. Every origin must be observed from the first
    /// age up to its latest, every observed cell must be positive, and
    /// there must be more observed cells than parameters. Raises one
    /// ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses: paid, or case incurred.
    /// premium : str
    ///     The premium of each origin.
    ///
    /// Returns
    /// -------
    /// MeyersFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a column is missing, the triangle has several segments or an
    ///     origin has a gap; if a premium is missing or not positive; if a
    ///     prior standard deviation or ``rho_shape`` is not positive and
    ///     finite, ``logelr_mean`` is not finite or a sampler setting is
    ///     invalid; if an observed cell is not positive; if there are no more
    ///     observed cells than parameters; or, under ``"cay"``, if a cell
    ///     lacks the previous origin's cell at the same age.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        premium: &str,
    ) -> PyResult<PyMeyersFit> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py.detach(|| model.fit(tri, column, premium)).map_err(err)?;
        warn_diagnostics(py, "Meyers", &inner.diagnostics.warnings)?;
        Ok(PyMeyersFit { inner })
    }

    /// Samples the posterior of one measure column in every segment of a
    /// cumulative triangle, each segment with its own posterior (the same
    /// model and settings, sampled independently) and its premium from its
    /// own ``premium`` column as ``fit`` reads it, into one joint
    /// distribution of the ultimates and one of the reserves, the segments
    /// depending on each other as ``dependence`` says. Segment ``k`` (in
    /// index order) takes seeds derived from ``seed`` and ``k`` for its
    /// chains and its simulations, so independent segments share no random
    /// numbers, and a segment's draws differ from ``fit`` of that segment
    /// alone. Raises one ``UserWarning`` listing the failed diagnostics of
    /// every segment, each message prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    ///     The losses: paid, or case incurred.
    /// premium : str
    ///     The premium of each origin.
    ///
    /// Returns
    /// -------
    /// MeyersFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit`` for any segment (the message names the segment); if
    ///     ``dependence`` is ``"synchronized"`` (before any sampling); or if
    ///     ``spearman`` does not fit the segments.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Meyers, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
    /// ...         [983, 2211, 2830], [1657, 2685], [932]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214]
    /// >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * premium[w])
    /// ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> lob, origin, age, values, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(
    /// ...     origin, age, {"paid": values, "premium": prem}, keys={"lob": lob})
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     fits = Meyers(tune=200, draws=200, n_sims=1000).fit_segments(
    /// ...         tri, "paid", "premium")
    /// >>> fits.index, fits.reserves.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> fits.segment(lob="Home").premium[:2]
    /// [11624.0, 9816.0]
    fn fit_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        premium: &str,
    ) -> PyResult<PyMeyersFits> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.fit_segments(tri, column, premium))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.diagnostics);
        warn_diagnostics(py, "Meyers", &warnings)?;
        Ok(PyMeyersFits { inner })
    }

    /// The one-year view of one measure column of a single-segment
    /// cumulative triangle with premium: the claims development result over
    /// the coming year, by Pareto smoothed importance sampling of the
    /// posterior (decision 19 of ``docs/design/reserving-bayes.md``). The
    /// posterior and the lifetime ultimates and reserves are ``fit``'s for
    /// the same settings. Each simulation picks a posterior draw, draws the
    /// cumulative values of the year's cells from it (the cells valued in
    /// the next twelve months, as in ``OdpBootstrap.one_year``), reweights
    /// every posterior draw by the likelihood of those values, and smooths
    /// the weights (one Pareto ``k`` per simulation). Origin ``w``'s claims
    /// development result is
    ///
    /// ``CDR[w] = E[U[w] | D0] - E[U[w] | D0, C]``
    ///
    /// with ``U`` the ultimate (the value at the oldest age) and each
    /// expectation the (weighted) posterior mean of the ultimate's expected
    /// value given a draw; an origin the year takes to the oldest age has
    /// its value there instead of the second term. The mean is about zero.
    /// A simulation whose ``k`` is above ``k_threshold`` is flagged and
    /// handled as ``refit`` says: by default moment-matched, and refitted
    /// with NUTS only if its ``k`` stays above the threshold. On Meyers'
    /// group 353 nearly every simulation is flagged and matching rescues
    /// about nine in ten, so 10,000 simulations take 4 to 7 minutes on 16
    /// threads in a release build, and ``"refit_flagged"`` about 6.7 times
    /// as long. The view therefore runs at most 2,000 simulations by
    /// default (about a fifth of that time), whatever the model's
    /// ``n_sims`` is above that; ``n_sims`` here sets the count. Raises one ``UserWarning``
    /// listing the failed diagnostics of the fit, and another if any refit
    /// failed its diagnostics.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses: paid, or case incurred.
    /// premium : str
    ///     The premium of each origin, as in ``fit``.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations. By default this model's
    ///     ``n_sims``, at most 2,000: nearly every simulation of these
    ///     models is flagged and costs a moment match or a refit. The
    ///     lifetime view in ``fit`` keeps ``n_sims``; a smaller count
    ///     gives the first simulations of a larger one.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    ///     With ``fit`` a ``MeyersFit``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Meyers, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724, 3832], [849, 1564, 2202, 2432, 2468],
    /// ...         [983, 2211, 2830, 3832], [1657, 2685, 3169], [932, 1940], [1162]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214, 5230]
    /// >>> rows = [(1988 + w, 12 * (d + 1), v, premium[w])
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> origin, age, values, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(origin, age, {"paid": values, "premium": prem})
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     view = Meyers(tune=300, draws=300, n_sims=1000, seed=1).one_year(
    /// ...         tri, "paid", "premium", refit="reweight_only")
    /// >>> type(view.fit).__name__, view.cdr.components()[:2]
    /// ('MeyersFit', [('1988',), ('1989',)])
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (triangle, column, premium, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        premium: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year(tri, column, premium, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "Meyers", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "Meyers",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::Meyers(Box::new(inner)),
        })
    }

    /// The one-year view of one measure column in every segment of a
    /// cumulative triangle with premium: each segment's posterior and
    /// simulations as in ``fit_segments`` (segment ``k``'s own seeds), its
    /// view as in ``one_year``, into one joint claims development result,
    /// the segments depending on each other as ``dependence`` says:
    /// ``"independent"``, or ``"rank_correlation"`` by Iman–Conover on the
    /// segments' total claims development results. Raises one
    /// ``UserWarning`` listing the failed diagnostics of every segment's
    /// fit, and another if any refit failed its diagnostics, each message
    /// prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    /// premium : str
    /// k_threshold : float, default 0.7
    ///     As in ``one_year``.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     As in ``one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``, for each segment: ``k`` segments
    ///     refit at most ``k`` times this many.
    /// n_sims : int, optional
    ///     As in ``one_year``.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFits
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``one_year`` and ``fit_segments``, including
    ///     ``dependence="synchronized"``.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Meyers, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
    /// ...         [983, 2211, 2830], [1657, 2685], [932]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214]
    /// >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * premium[w])
    /// ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> lob, origin, age, values, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(
    /// ...     origin, age, {"paid": values, "premium": prem}, keys={"lob": lob})
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     views = Meyers(tune=200, draws=200, n_sims=500).one_year_segments(
    /// ...         tri, "paid", "premium", refit="reweight_only")
    /// >>> views.index, views.cdr.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> views.segment(lob="Home").premium[:2]
    /// [11624.0, 9816.0]
    #[pyo3(signature = (triangle, column, premium, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        premium: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFits> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year_segments(tri, column, premium, &settings))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.fit.diagnostics);
        warn_diagnostics(py, "Meyers", &warnings)?;
        warn_refits(py, "Meyers", &segment_refits(&inner.segments))?;
        Ok(PyBayesOneYearFits {
            inner: BayesOneYearFitsOf::Meyers(inner),
        })
    }

    fn __repr__(&self) -> String {
        let (m, s) = (&self.inner, &self.inner.sampler);
        format!(
            "Meyers(model={:?}, logelr_mean={:?}, logelr_sd={:?}, origin_sd={:?}, \
             development_sd={:?}, gamma_sd={:?}, rho_shape={:?}, chains={}, tune={}, \
             draws={}, n_sims={}, seed={}{})",
            m.model.name(),
            m.logelr_mean,
            m.logelr_sd,
            m.origin_sd,
            m.development_sd,
            m.gamma_sd,
            m.rho_shape,
            s.chains,
            s.tune,
            s.draws,
            m.n_sims,
            m.seed,
            dependence_repr(&m.dependence),
        )
    }
}

/// The posterior predictive of a new underwriting year, from
/// ``BayesBfNormalisedFit.new_year`` or ``MeyersFit.new_year``.
///
/// Row ``i`` of ``loss_ratio``, ``ultimate`` and every pattern of
/// ``patterns`` is simulation ``i``.
#[pyclass(name = "NewYear", module = "prospicio.reserving", frozen)]
pub(crate) struct PyNewYear {
    pub(crate) inner: NewYear,
}

#[pymethods]
impl PyNewYear {
    /// The new origin period, the one after the triangle's newest.
    #[getter]
    fn origin(&self) -> String {
        self.inner.origin.to_string()
    }

    /// The new year's premium.
    #[getter]
    fn premium(&self) -> f64 {
        self.inner.premium
    }

    /// The ultimate loss ratio of the new year: dimension ``"origin"``,
    /// one component, one row per simulation.
    #[getter]
    fn loss_ratio(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.loss_ratio.clone(),
        }
    }

    /// The ultimate loss of the new year, ``loss_ratio`` times
    /// ``premium``, with the same layout.
    #[getter]
    fn ultimate(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.ultimate.clone(),
        }
    }

    /// The development pattern of each simulation's posterior draw; its
    /// ``monthly`` is the new year's emergence by month.
    ///
    /// Returns
    /// -------
    /// DevelopmentPatterns
    #[getter]
    fn patterns(&self) -> PyDevelopmentPatterns {
        PyDevelopmentPatterns {
            inner: self.inner.patterns.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NewYear(origin={}, premium={}, n_sims={})",
            self.inner.origin,
            self.inner.premium,
            self.inner.patterns.n_sims()
        )
    }
}

/// A fitted ``Meyers``: the posterior of ``logelr``, ``alpha[2..n]``,
/// ``beta[1..m-1]``, ``gamma`` (``"csr"`` only) or ``rho`` (``"cay"``
/// only), and ``sigma[1..m]``, its
/// MCMC diagnostics, and the posterior predictive distributions of the
/// ultimates and reserves by origin. The frames' ``latest``, ``ultimate``
/// and ``reserve`` are the volume-weighted chain ladder's
/// (``chain_ladder``); ``mean`` and ``std_dev`` are the posterior
/// predictive reserve's, and ``ultimate_mean`` and ``ultimate_std_dev``
/// the posterior predictive ultimate's.
#[pyclass(name = "MeyersFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMeyersFit {
    pub(crate) inner: MeyersFit,
}

#[pymethods]
impl PyMeyersFit {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Meyers' models simulate no path, so each reserve is spread over its
    /// simulation's pattern.
    ///
    /// Parameters
    /// ----------
    /// basis : {"paid", "incurred"}
    ///     What the fit's column is, stated by the caller (no default), so
    ///     that a fit on incurred claims is not discounted by its reporting
    ///     pattern by mistake. ``"paid"`` pays each simulation's reserve
    ///     along that simulation's own pattern. ``"incurred"`` pays each
    ///     reserve (the IBNR) plus the origin's case reserve (its latest
    ///     incurred less ``latest_paid``) along ``paid_pattern``.
    /// rate : float, optional
    ///     A flat annual effective rate. Give ``rate`` or ``curve``.
    /// curve : DiscountCurve, optional
    ///     Spot rates by maturity.
    /// timing : {"mid", "end"}, default "mid"
    ///     Payments at the middle or the end of each development period.
    /// paid_pattern : list of float, optional
    ///     For ``basis="incurred"``: the cumulative paid share at each of
    ///     the fit's pattern ages (with the tail's age when the fit has a
    ///     tail), ending at 1.
    /// latest_paid : list of float, optional
    ///     For ``basis="incurred"``: each origin's latest paid, one per
    ///     component of ``reserves`` in their order.
    ///
    /// Returns
    /// -------
    /// DiscountedReserves
    #[pyo3(signature = (
        *,
        basis,
        rate = None,
        curve = None,
        timing = "mid",
        paid_pattern = None,
        latest_paid = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn discounted(
        &self,
        py: Python<'_>,
        basis: &str,
        rate: Option<f64>,
        curve: Option<PyRef<'_, PyDiscountCurve>>,
        timing: &str,
        paid_pattern: Option<Vec<f64>>,
        latest_paid: Option<Vec<f64>>,
    ) -> PyResult<PyDiscountedReserves> {
        let args = DiscountArgs {
            basis,
            rate,
            curve,
            timing,
            paid_pattern,
            latest_paid,
        };
        discounted(py, &self.inner, args)
    }

    /// The posterior predictive of a new underwriting year, the origin
    /// after the triangle's newest, for pricing a year not yet written.
    ///
    /// Each simulation takes a posterior draw and from it the new origin's
    /// level ``logelr + alpha``, then its ultimate, log-normal with sd
    /// ``sigma[m]`` (in CAY, plus ``rho`` times the newest origin's
    /// residual). Meyers' models have no hyperparameter for the origin
    /// effects, so ``level`` says how the new one is drawn.
    ///
    /// Parameters
    /// ----------
    /// premium : float
    ///     The new year's premium, finite and positive. The loss ratio does
    ///     not depend on it.
    /// n_sims : int
    ///     Number of simulations.
    /// seed : int
    ///     Seed of the simulations.
    /// level : str
    ///     ``"exchangeable"`` (the default): the new origin is like the
    ///     triangle's, its level a Student t draw from a normal fitted to the
    ///     draw's origin levels (needs three origins). ``"prior"``: Meyers'
    ///     own prior ``alpha ~ N(0, origin_sd**2)``, too wide to price on at
    ///     his ``origin_sd`` of 3.162.
    ///
    /// Returns
    /// -------
    /// NewYear
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``premium`` is not positive, ``n_sims`` is 0, ``level`` is
    ///     unknown, or the triangle has fewer than three origins under
    ///     ``"exchangeable"``.
    #[pyo3(signature = (premium, n_sims = 10_000, seed = 0, level = "exchangeable"))]
    fn new_year(
        &self,
        py: Python<'_>,
        premium: f64,
        n_sims: usize,
        seed: u64,
        level: &str,
    ) -> PyResult<PyNewYear> {
        let level = match level {
            "exchangeable" => NewOriginLevel::Exchangeable,
            "prior" => NewOriginLevel::Prior,
            other => {
                return Err(PyValueError::new_err(format!(
                    "level must be \"exchangeable\" or \"prior\", not {other:?}"
                )));
            }
        };
        let fit = &self.inner;
        let inner = py
            .detach(|| fit.new_year_with_level(premium, level, n_sims, seed))
            .map_err(err)?;
        Ok(PyNewYear { inner })
    }

    /// The model: ``"crc"``, ``"csr"`` or ``"cay"``.
    #[getter]
    fn model(&self) -> &'static str {
        self.inner.model().name()
    }

    /// The volume-weighted chain ladder of the fitted column, whose latest
    /// values the reserves are measured from.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    /// Origin period of each component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .chain_ladder
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.chain_ladder.development.development.clone()
    }

    /// The premium of each origin.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner.premium.clone()
    }

    /// Each simulation's development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine: the shares
    /// ``exp(beta_d g)`` of the posterior draw the simulation took, the
    /// median value at each age over the median at the oldest age of an
    /// origin after the newest (``g`` its speed-up ``(1 - gamma)^n`` in CSR,
    /// 1 otherwise). No tail: each pattern ends at 1 at the oldest age.
    /// Nothing keeps ``beta_d`` rising or below 0, so a share can fall or
    /// pass 1 (on Meyers' group 353 data it falls in nearly every
    /// simulation), which ``monthly`` refuses unless
    /// ``monotone="running_max"``.
    ///
    /// Row ``i`` of every pattern is simulation ``i`` of ``reserves``,
    /// ``ultimates`` and ``ultimate_draws``.
    ///
    /// Returns
    /// -------
    /// DevelopmentPatterns
    #[getter]
    fn patterns(&self) -> PyDevelopmentPatterns {
        PyDevelopmentPatterns {
            inner: self.inner.patterns.clone(),
        }
    }

    /// Joint posterior predictive distribution of the ultimate (the value
    /// at the oldest age) by origin: dimension ``"origin"``, one component
    /// per origin, one row per simulation. An origin observed at the
    /// oldest age keeps its value.
    #[getter]
    fn ultimates(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.ultimates.clone(),
        }
    }

    /// The ultimates less the latest values, simulation by simulation,
    /// laid out as ``ultimates`` and as ``OdpBootstrapFit.reserves``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Each simulation's ultimate, for a tower, a deal book or the cash-flow
    /// engine: index ``i`` is simulation ``i`` for every origin and every
    /// call, the same as row ``i`` of ``reserves`` and ``ultimates``, and
    /// the same again when the fit is run with the same seed.
    ///
    /// The ultimate is the origin's latest value plus its reserve in that
    /// simulation, which is the simulated value at the oldest age itself (``ultimates``).
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     The component key (``(2021,)``) or the origin alone; an origin period by its label
    ///     (``"2021"``, ``2021``, ``"2021Q3"``). ``None`` (the default) gives
    ///     the total over every component, summed simulation by simulation.
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (origin = None))]
    fn ultimate_draws(&self, origin: Option<OriginArg>) -> PyResult<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// Each simulation's ultimate over ``premium``: the loss ratio a
    /// contract on the loss ratio reads, in the order of
    /// ``ultimate_draws``. Multiply by the premium for
    /// ``Tower.apply_annual``.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     As in ``ultimate_draws``; ``None`` for the total.
    /// premium : float
    ///     The subject premium of the origin (of every origin for the
    ///     total), finite and positive.
    ///
    /// Returns
    /// -------
    /// list of float
    fn loss_ratio_draws(&self, origin: Option<OriginArg>, premium: f64) -> PyResult<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold; empty when every threshold holds).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = &self.inner.diagnostics;
        let out = PyDict::new(py);
        out.set_item("max_rhat", d.max_rhat)?;
        out.set_item("min_ess_bulk", d.min_ess_bulk)?;
        out.set_item("min_ess_tail", d.min_ess_tail)?;
        out.set_item("divergences", d.divergences)?;
        out.set_item("warnings", d.warnings.clone())?;
        Ok(out)
    }

    /// Parameter names: ``logelr``, ``alpha[2]`` to ``alpha[n]``,
    /// ``beta[1]`` to ``beta[m-1]``, ``gamma`` (``"csr"`` only) or ``rho``
    /// (``"cay"`` only), and ``sigma[1]`` to ``sigma[m]``.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, one row per draw (chain by chain), one column per
    /// parameter in the order of ``parameter_names``.
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        let p = self.inner.posterior.dim().max(1);
        self.inner
            .posterior
            .values()
            .chunks(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// Number of chains.
    #[getter]
    fn chains(&self) -> usize {
        self.inner.posterior.chains()
    }

    /// Posterior summary, one row per parameter: ``parameter``, ``mean``,
    /// ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
    /// ``ess_tail``, the fields of ``BayesGlmFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        let out = PyDict::new(py);
        out.set_item(
            "parameter",
            s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
        )?;
        out.set_item("mean", col(|p| p.mean))?;
        out.set_item("sd", col(|p| p.sd))?;
        out.set_item("q05", col(|p| p.q05))?;
        out.set_item("q50", col(|p| p.q50))?;
        out.set_item("q95", col(|p| p.q95))?;
        out.set_item("rhat", col(|p| p.rhat))?;
        out.set_item("ess_bulk", col(|p| p.ess_bulk))?;
        out.set_item("ess_tail", col(|p| p.ess_tail))?;
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    /// PSIS-LOO of the observed cells, with each cell's relative
    /// efficiency from the chains, as ``BayesGlmFit.loo``. The pointwise
    /// term is Meyers', the normal density of ``log C[w, d]``, so the
    /// result is on the log scale and comparable only between ``"crc"``,
    /// ``"csr"`` and ``"cay"`` fits of the same column (Meyers 2019,
    /// Tables 6.4, 7.3 and 8.3). Under ``"cay"`` each cell's density is
    /// conditional on the previous origin's cell, as in Meyers' code.
    ///
    /// Returns
    /// -------
    /// Elpd
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per origin: ``origin``, the chain ladder's ``latest``,
    /// ``ultimate`` and ``reserve``, the ``premium``, ``ulr`` (the chain
    /// ladder's ultimate over the premium), the ``mean`` and
    /// ``std_dev`` of the posterior predictive reserve, the
    /// ``ultimate_mean`` and ``ultimate_std_dev`` of the posterior
    /// predictive ultimate, and ``ulr_mean``, the ``ultimate_mean`` over
    /// the premium. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the chain ladder's totals, the total ``premium`` and
    /// ``ulr``, the ``mean`` and ``std_dev`` of the posterior predictive
    /// total reserve, the ``ultimate_mean`` and ``ultimate_std_dev`` of the
    /// total ultimate, and ``ulr_mean``, the total ``ultimate_mean`` over
    /// the total premium. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladder's development factors, one row per age, as
    /// ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    fn __repr__(&self) -> String {
        format!(
            "MeyersFit(model={:?}, origins={}, draws={}, n_sims={}, warnings={})",
            self.inner.model().name(),
            self.inner.chain_ladder.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// ``Meyers.fit_segments``: each segment's own posterior and the joint
/// posterior predictive distributions of the ultimates and the reserves,
/// laid out as ``BayesOdpFits``.
///
/// ``ultimates`` and ``reserves`` have the triangle's keys and ``"origin"``
/// as dimensions, so ``reserves.aggregate(["lob"])`` keeps the dependence
/// between segments. Per-origin lists (``origins``, ``premium``) run over
/// the origins of each segment in turn, like the rows of ``to_frame()``
/// and the components of ``reserves``. The posterior of one segment
/// (``diagnostics``, ``parameter_names``, ``draws``, ``chains``,
/// ``summary()`` and ``loo()``) needs a single-segment fit: for several
/// segments use ``segment(...)`` or ``diagnostics_frame()``.
#[pyclass(name = "MeyersFits", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMeyersFits {
    inner: MeyersFits,
}

impl PyMeyersFits {
    fn one(&self, field: &str, instead: &str) -> PyResult<&MeyersSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyMeyersFits {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Meyers' models simulate no path, so each reserve is spread over its
    /// simulation's pattern, each segment by its own.
    ///
    /// Parameters
    /// ----------
    /// basis : {"paid", "incurred"}
    ///     What the fit's column is, stated by the caller (no default), so
    ///     that a fit on incurred claims is not discounted by its reporting
    ///     pattern by mistake. ``"paid"`` pays each simulation's reserve
    ///     along that simulation's own pattern. ``"incurred"`` pays each
    ///     reserve (the IBNR) plus the origin's case reserve (its latest
    ///     incurred less ``latest_paid``) along ``paid_pattern``.
    /// rate : float, optional
    ///     A flat annual effective rate. Give ``rate`` or ``curve``.
    /// curve : DiscountCurve, optional
    ///     Spot rates by maturity.
    /// timing : {"mid", "end"}, default "mid"
    ///     Payments at the middle or the end of each development period.
    /// paid_pattern : list of float, optional
    ///     For ``basis="incurred"``: the cumulative paid share at each of
    ///     the fit's pattern ages (with the tail's age when the fit has a
    ///     tail), ending at 1.
    /// latest_paid : list of float, optional
    ///     For ``basis="incurred"``: each origin's latest paid, one per
    ///     component of ``reserves`` in their order.
    ///
    /// Returns
    /// -------
    /// DiscountedReserves
    #[pyo3(signature = (
        *,
        basis,
        rate = None,
        curve = None,
        timing = "mid",
        paid_pattern = None,
        latest_paid = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn discounted(
        &self,
        py: Python<'_>,
        basis: &str,
        rate: Option<f64>,
        curve: Option<PyRef<'_, PyDiscountCurve>>,
        timing: &str,
        paid_pattern: Option<Vec<f64>>,
        latest_paid: Option<Vec<f64>>,
    ) -> PyResult<PyDiscountedReserves> {
        let args = DiscountArgs {
            basis,
            rate,
            curve,
            timing,
            paid_pattern,
            latest_paid,
        };
        discounted(py, &self.inner, args)
    }

    /// The model: ``"crc"``, ``"csr"`` or ``"cay"``.
    #[getter]
    fn model(&self) -> &'static str {
        self.inner.segments.fits[0].model().name()
    }

    /// The volume-weighted chain ladder of every segment's fitted column,
    /// whose latest values the reserves are measured from.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.segments.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner.segments)
    }

    /// Origin period of each per-origin value and component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner.segments)
    }

    /// Development ages in months of the first segment.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.segments.fits[0]
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// The premium of each segment's origins, in the order of ``origins``.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.premium.iter().copied())
            .collect()
    }

    /// Each simulation's development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine: the shares
    /// ``exp(beta_d g)`` of the posterior draw the simulation took, the
    /// median value at each age over the median at the oldest age of an
    /// origin after the newest (``g`` its speed-up ``(1 - gamma)^n`` in CSR,
    /// 1 otherwise). No tail: each pattern ends at 1 at the oldest age.
    /// Nothing keeps ``beta_d`` rising or below 0, so a share can fall or
    /// pass 1 (on Meyers' group 353 data it falls in nearly every
    /// simulation), which ``monthly`` refuses unless
    /// ``monotone="running_max"``.
    ///
    /// Row ``i`` of every pattern is simulation ``i`` of ``reserves``,
    /// ``ultimates`` and ``ultimate_draws``.
    ///
    /// Returns
    /// -------
    /// DevelopmentPatterns
    #[getter]
    fn patterns(&self) -> PyDevelopmentPatterns {
        PyDevelopmentPatterns {
            inner: self.inner.patterns.clone(),
        }
    }

    /// Joint posterior predictive distribution of the ultimate (the value
    /// at the oldest age) by segment and origin: the triangle's keys and
    /// ``"origin"`` are its dimensions, one component per segment and
    /// origin, one row per simulation.
    #[getter]
    fn ultimates(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.ultimates.clone(),
        }
    }

    /// The ultimates less the latest values, simulation by simulation,
    /// laid out as ``ultimates``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Each simulation's ultimate, for a tower, a deal book or the cash-flow
    /// engine: index ``i`` is simulation ``i`` for every origin and every
    /// call, the same as row ``i`` of ``reserves`` and ``ultimates``, and
    /// the same again when the fit is run with the same seed.
    ///
    /// The ultimate is the origin's latest value plus its reserve in that
    /// simulation, which is the simulated value at the oldest age itself (``ultimates``).
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     The component key, the segment's key values then the origin (``("Auto", 2021)``); an origin period by its label
    ///     (``"2021"``, ``2021``, ``"2021Q3"``). ``None`` (the default) gives
    ///     the total over every component, summed simulation by simulation.
    ///
    /// Returns
    /// -------
    /// list of float
    #[pyo3(signature = (origin = None))]
    fn ultimate_draws(&self, origin: Option<OriginArg>) -> PyResult<Vec<f64>> {
        ultimate_draws_of(&self.inner, origin)
    }

    /// Each simulation's ultimate over ``premium``: the loss ratio a
    /// contract on the loss ratio reads, in the order of
    /// ``ultimate_draws``. Multiply by the premium for
    /// ``Tower.apply_annual``.
    ///
    /// Parameters
    /// ----------
    /// origin : tuple, str, int or None
    ///     As in ``ultimate_draws``; ``None`` for the total.
    /// premium : float
    ///     The subject premium of the origin (of every origin for the
    ///     total), finite and positive.
    ///
    /// Returns
    /// -------
    /// list of float
    fn loss_ratio_draws(&self, origin: Option<OriginArg>, premium: f64) -> PyResult<Vec<f64>> {
        loss_ratio_draws_of(&self.inner, origin, premium)
    }

    /// MCMC diagnostics of a single-segment fit, as
    /// ``MeyersFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(
            py,
            &self.one("diagnostics", "diagnostics_frame()")?.diagnostics,
        )
    }

    /// Parameter names of a single-segment fit, as
    /// ``MeyersFit.parameter_names``.
    #[getter]
    fn parameter_names(&self) -> PyResult<Vec<String>> {
        Ok(self.one("parameter_names", "")?.posterior.names().to_vec())
    }

    /// Posterior draws of a single-segment fit, as ``MeyersFit.draws``.
    #[getter]
    fn draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let posterior = &self.one("draws", "")?.posterior;
        let p = posterior.dim().max(1);
        Ok(posterior.values().chunks(p).map(<[f64]>::to_vec).collect())
    }

    /// Number of chains of a single-segment fit.
    #[getter]
    fn chains(&self) -> PyResult<usize> {
        Ok(self.one("chains", "")?.posterior.chains())
    }

    /// Posterior summary of a single-segment fit, as
    /// ``MeyersFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let one = self.one("summary()", "")?;
        let s = py.detach(|| one.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of a single-segment fit's observed cells on Meyers' log
    /// scale, as ``MeyersFit.loo``.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments.
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let one = self.one("loo()", "")?;
        let l = py.detach(|| one.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per segment: the key columns, then ``max_rhat``,
    /// ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
    /// ``warnings`` (the segment's messages joined by ``"; "``, empty when
    /// every threshold holds). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn diagnostics_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        segment_diagnostics_frame(py, &self.inner.segments, |s| &s.diagnostics)
    }

    /// One row per segment and origin: the key columns, ``origin``, the
    /// chain ladder's ``latest``, ``ultimate`` and ``reserve``, the
    /// ``premium``, ``ulr`` (the chain ladder's ultimate over the premium),
    /// the ``mean`` and ``std_dev`` of the posterior predictive reserve,
    /// the ``ultimate_mean`` and ``ultimate_std_dev`` of the posterior
    /// predictive ultimate, and ``ulr_mean``, the ``ultimate_mean`` over the
    /// premium. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the chain ladder's totals,
    /// the total ``premium`` and ``ulr``, the ``mean`` and ``std_dev`` of
    /// the segment's posterior predictive total reserve, and the
    /// ``ultimate_mean`` and ``ultimate_std_dev`` of its total ultimate,
    /// and ``ulr_mean``, the total ``ultimate_mean`` over the total premium.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The chain ladders' development factors, one row per segment and
    /// age, as ``ChainLadderFit.development_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint ultimates and
    /// reserves (same dimensions).
    ///
    /// Returns
    /// -------
    /// MeyersFits
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        let segments = &self.inner.segments;
        let warnings: usize = segments
            .fits
            .iter()
            .map(|f| f.diagnostics.warnings.len())
            .sum();
        format!(
            "MeyersFits(model={:?}, {}origins={}, n_sims={}, warnings={warnings})",
            self.model(),
            segments_prefix(segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}
