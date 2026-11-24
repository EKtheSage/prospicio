//! The Bayesian chain ladder, `BayesOdp`, and its fits.

use super::*;

/// The Bayesian chain ladder of England and Verrall (2002, 2006): the
/// over-dispersed Poisson model of a cumulative triangle's incremental
/// values ``X`` with mean ``exp(c + a[w] + b[d])`` (``a[1] = b[1] = 0``),
/// normal priors on the log scale, and the chain ladder's Pearson scale
/// ``phi`` (``OdpBootstrapFit.scale``) plugged in. The posterior is sampled
/// with NUTS, and each draw projects the future cells up to the oldest
/// observed age (no tail) with the process error of ``process``, as
/// ``OdpBootstrap`` does. Every origin's and every development column's
/// observed increments must sum to more than zero; negative cells are
/// allowed.
///
/// One ``seed`` replays a fit exactly, whatever the number of threads: the
/// chains and the simulations each take a seed derived from it.
///
/// Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
/// per chain, any divergent transition) do not fail the fit: ``fit``
/// raises one ``UserWarning`` listing them, and they stay in
/// ``BayesOdpFit.diagnostics``.
///
/// Parameters
/// ----------
/// intercept_sd : float, default 10.0
///     Standard deviation of the normal prior on the intercept ``c``,
///     centred on the log of the mean observed increment.
/// origin_sd : float, default 10.0
///     Standard deviation of the normal prior on each origin effect
///     ``a[w]``, ``w >= 2``, centred on 0.
/// development_sd : float, default 10.0
///     Standard deviation of the normal prior on each development effect
///     ``b[d]``, ``d >= 2``, centred on 0.
/// process : {"gamma", "none"}, default "gamma"
///     Process error on each simulated future increment: Gamma with mean
///     ``mu`` and variance ``phi * mu``, or none for parameter error only.
/// chains, tune, draws : int, default 4, 1000, 1000
///     NUTS chains, warm-up draws per chain and kept draws per chain.
/// n_sims : int, default 10000
///     Number of predictive simulations, independent of the number of
///     posterior draws; positive.
/// seed : int, default 0
///     Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
/// target_accept : float, default 0.95
///     Target mean acceptance rate of the step-size adaptation; above
///     NUTS's usual 0.8, which on triangles such as RAA usually leaves a
///     few divergent transitions.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
/// dependence : {"independent", "rank_correlation", "synchronized"}, default "independent"
///     How the segments of a multi-segment triangle depend on each other
///     in ``fit_segments`` (``fit`` has one segment and does not use it):
///     each segment's own posterior and simulations, with seeds derived
///     from ``seed`` and the segment's position so that the segments share
///     no random numbers; or the segments' simulations reordered as whole
///     rows (Iman–Conover on the segment totals) to the Spearman matrix
///     ``spearman`` and then put in a random order, as
///     ``OdpBootstrap``'s. ``"synchronized"`` is accepted here, as the
///     bootstraps take it, but ``fit_segments`` raises: it resamples the
///     residuals of the same cells in every segment, and a posterior has no
///     residuals to share.
/// spearman : list of list of float, optional
///     With ``dependence="rank_correlation"`` only: Spearman's rho between
///     the segments' totals, as ``OdpBootstrap``'s.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero, ``process`` or ``dependence`` is unknown, or
///     ``spearman`` is missing for ``"rank_correlation"``, given for another
///     dependence or not square.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BayesOdp, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = BayesOdp(n_sims=2000, seed=1).fit(tri, "values")
/// >>> fit.reserves.components()
/// [('2020',), ('2021',), ('2022',), ('2023',)]
/// >>> fit.reserves.mean() > 0
/// True
/// >>> fit.parameter_names[:2]
/// ['c', 'a[2]']
#[pyclass(name = "BayesOdp", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOdp {
    pub(crate) inner: BayesOdp,
}

#[pymethods]
impl PyBayesOdp {
    #[new]
    #[pyo3(signature = (
        intercept_sd = 10.0,
        origin_sd = 10.0,
        development_sd = 10.0,
        process = "gamma",
        chains = 4,
        tune = 1000,
        draws = 1000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.95,
        max_depth = 10,
        dependence = "independent",
        spearman = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        intercept_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        process: &str,
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
            inner: BayesOdp {
                intercept_sd,
                origin_sd,
                development_sd,
                process: process_distribution(process)?,
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

    /// Standard deviation of the intercept's prior.
    #[getter]
    fn intercept_sd(&self) -> f64 {
        self.inner.intercept_sd
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

    /// Process error: ``"gamma"`` or ``"none"``.
    #[getter]
    fn process(&self) -> &'static str {
        process_name(self.inner.process)
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
    /// cumulative triangle and simulates its reserves. Every origin must be
    /// observed from the first age up to its latest, there must be more
    /// observed cells than parameters, and every origin's and every
    /// development column's increments must sum to more than zero. Raises
    /// one ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///
    /// Returns
    /// -------
    /// BayesOdpFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``OdpBootstrap.fit``; if the triangle has several segments; if
    ///     a prior standard deviation is not positive and finite or a
    ///     sampler setting is invalid; or if an origin's or a development
    ///     column's increments do not sum to more than zero.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyBayesOdpFit> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py.detach(|| model.fit(tri, column)).map_err(err)?;
        warn_diagnostics(py, "BayesOdp", &inner.diagnostics.warnings)?;
        Ok(PyBayesOdpFit { inner })
    }

    /// Samples the posterior of one measure column in every segment of a
    /// cumulative triangle, each segment with its own posterior (the same
    /// model and settings, sampled independently), into one joint
    /// distribution of the reserves, the segments depending on each other
    /// as ``dependence`` says. Segment ``k`` (in index order) takes seeds
    /// derived from ``seed`` and ``k`` for its chains and its simulations,
    /// so independent segments share no random numbers, and a segment's
    /// draws differ from ``fit`` of that segment alone. Raises one
    /// ``UserWarning`` listing the failed diagnostics of every segment,
    /// each message prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    ///
    /// Returns
    /// -------
    /// BayesOdpFits
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
    /// >>> from prospicio.reserving import BayesOdp, Triangle
    /// >>> paid = [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]
    /// >>> tri = Triangle.from_long(
    /// ...     ([2020] * 4 + [2021] * 3 + [2022] * 2 + [2023]) * 2,
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12] * 2,
    /// ...     {"paid": paid + [2 * v for v in paid]},
    /// ...     keys={"lob": ["Auto"] * 10 + ["Home"] * 10},
    /// ... )
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     fits = BayesOdp(tune=300, draws=300, n_sims=1000).fit_segments(tri, "paid")
    /// >>> fits.index, fits.reserves.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> fits.segment(lob="Home").scale > 0
    /// True
    fn fit_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyBayesOdpFits> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py.detach(|| model.fit_segments(tri, column)).map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.diagnostics);
        warn_diagnostics(py, "BayesOdp", &warnings)?;
        Ok(PyBayesOdpFits { inner })
    }

    /// The one-year view of one measure column of a single-segment
    /// cumulative triangle: the claims development result over the coming
    /// year, by Pareto smoothed importance sampling of the posterior
    /// (decision 16 of ``docs/design/reserving-bayes.md``). The posterior
    /// and the lifetime reserves are ``fit``'s for the same settings. Each
    /// simulation picks a posterior draw, draws the year's increments from
    /// it with ``process`` error (the cells valued in the next twelve
    /// months, as in ``OdpBootstrap.one_year``), reweights every posterior
    /// draw by the year's quasi likelihood with ``scale`` and the priors
    /// unchanged, and smooths the weights (one Pareto ``k`` per
    /// simulation). Origin ``w``'s claims development result is
    ///
    /// ``CDR[w] = E[R[w] | D0] - (X[w] + E[R'[w] | D0, X])``
    ///
    /// with ``R`` the expected run-off at the opening, ``X`` the year's
    /// payments and ``R'`` the expected run-off after the year, each
    /// expectation the (weighted) mean over the posterior draws; its mean
    /// is about zero. A simulation whose ``k`` is above ``k_threshold`` is
    /// flagged. With ``refit="moment_match_then_refit"`` (the default) its
    /// posterior draws are first moved by affine maps to the moments the
    /// weights estimate (moment matching, as R's
    /// ``loo::loo_moment_match``), and only a simulation whose ``k`` stays
    /// above ``k_threshold`` is refitted with NUTS on the triangle extended
    /// by its year, with this model's sampler. 14% to 26% of the
    /// simulations are flagged on RAA and GenIns; matching rescues nearly
    /// all of them, within 0.04 to 0.06 sd of their refits, at about 1 ms
    /// each against about 26 ms for a refit, so a default run of 10,000
    /// simulations takes a few seconds in a release build.
    /// ``"refit_flagged"`` refits every flagged simulation (one to two
    /// minutes there), and ``"reweight_only"`` takes about a second and
    /// leaves the flagged simulations' reweighting as it is. Every cell up to the end of the
    /// year is observed and conditioned on, also for an origin that lags
    /// the valuation. Raises one ``UserWarning`` listing the failed
    /// diagnostics of the fit, and another if any refit failed its
    /// diagnostics.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted (the threshold of Vehtari et al. 2024 for a few
    ///     thousand draws).
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     What happens to a flagged simulation: moment-matched, and
    ///     refitted only if its ``k`` stays above ``k_threshold``
    ///     (``moment_matched`` lists the rescued); refitted with NUTS
    ///     (``"refit_flagged"``); or reweighted anyway (fast, biased where
    ///     ``k`` is high); ``"refit_all"`` refits every simulation, to
    ///     check the reweighting.
    /// max_refits : int, optional
    ///     The most simulations refitted with NUTS: the first this many due
    ///     for a refit, in simulation order; the others are reweighted as
    ///     they are (as unreliable as their ``k`` says), listed in
    ///     ``unrefitted`` and warned about. ``None`` (the default) refits
    ///     every one, ``0`` none.
    /// n_sims : int, optional
    ///     The number of one-year simulations; by default this model's
    ///     ``n_sims``. A smaller count gives the first simulations of a
    ///     larger one.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import BayesOdp, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    /// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    /// ... )
    /// >>> view = BayesOdp(n_sims=1000, seed=1).one_year(tri, "values", refit="reweight_only")
    /// >>> view.cdr.components()
    /// [('2020',), ('2021',), ('2022',), ('2023',)]
    /// >>> len(view.pareto_k), view.refitted
    /// (1000, [])
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (triangle, column, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year(tri, column, &settings))
            .map_err(err)?;
        warn_diagnostics(py, "BayesOdp", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "BayesOdp",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::Odp(inner),
        })
    }

    /// The one-year view of one measure column in every segment of a
    /// cumulative triangle: each segment's posterior and simulations as in
    /// ``fit_segments`` (segment ``k``'s own seeds), reweighted and
    /// refitted as in ``one_year``, into one joint claims development
    /// result, the segments depending on each other as ``dependence`` says:
    /// ``"independent"``, or ``"rank_correlation"`` by Iman–Conover on the
    /// segments' total claims development results, as the bootstraps'
    /// one-year view does. Raises one ``UserWarning`` listing the failed
    /// diagnostics of every segment's fit, and another if any refit failed
    /// its diagnostics, each message prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
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
    /// >>> from prospicio.reserving import BayesOdp, Triangle
    /// >>> paid = [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]
    /// >>> tri = Triangle.from_long(
    /// ...     ([2020] * 4 + [2021] * 3 + [2022] * 2 + [2023]) * 2,
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12] * 2,
    /// ...     {"paid": paid + [2 * v for v in paid]},
    /// ...     keys={"lob": ["Auto"] * 10 + ["Home"] * 10},
    /// ... )
    /// >>> model = BayesOdp(tune=300, draws=300, n_sims=500)
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     views = model.one_year_segments(tri, "paid", refit="reweight_only")
    /// >>> views.index, views.cdr.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> len(views.segment(lob="Home").pareto_k)
    /// 500
    #[pyo3(signature = (triangle, column, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFits> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year_segments(tri, column, &settings))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.fit.diagnostics);
        warn_diagnostics(py, "BayesOdp", &warnings)?;
        warn_refits(py, "BayesOdp", &segment_refits(&inner.segments))?;
        Ok(PyBayesOneYearFits {
            inner: BayesOneYearFitsOf::Odp(inner),
        })
    }

    fn __repr__(&self) -> String {
        let s = &self.inner.sampler;
        format!(
            "BayesOdp(intercept_sd={:?}, origin_sd={:?}, development_sd={:?}, process={:?}, \
             chains={}, tune={}, draws={}, n_sims={}, seed={}{})",
            self.inner.intercept_sd,
            self.inner.origin_sd,
            self.inner.development_sd,
            self.process(),
            s.chains,
            s.tune,
            s.draws,
            self.inner.n_sims,
            self.inner.seed,
            dependence_repr(&self.inner.dependence),
        )
    }
}

/// A fitted ``BayesOdp``: the posterior of ``c``, ``a[2..n]`` and
/// ``b[2..m]``, its MCMC diagnostics, and the posterior predictive
/// distribution of the reserves, laid out as ``OdpBootstrapFit.reserves``.
/// The frames' ``latest``, ``ultimate`` and ``reserve`` are the
/// volume-weighted chain ladder's (``chain_ladder``); ``mean`` and
/// ``std_dev`` are the posterior predictive reserve's.
#[pyclass(name = "BayesOdpFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOdpFit {
    pub(crate) inner: BayesOdpFit,
}

#[pymethods]
impl PyBayesOdpFit {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Each simulation's reserve is spread over the pattern of its own
    /// posterior draw.
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

    /// Each simulation's development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine.
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
    /// The volume-weighted chain ladder of the triangle, whose latest
    /// values and Pearson scale the model uses.
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

    /// Origin period of each reserve component.
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

    /// The plugged-in scale ``phi``, the chain ladder's Pearson scale (the
    /// ODP bootstrap's).
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// Joint posterior predictive distribution of the reserve (the sum of
    /// future incremental values) by origin: dimension ``"origin"``, one
    /// component per origin, one row per simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Joint distribution of the ultimate by origin: ``reserves`` with
    /// each origin's latest value added in every simulation, the same
    /// dimensions, components and rows, so ``join``, ``select`` and
    /// ``reorder_groups`` keep each simulation's origins together.
    #[getter]
    fn ultimates(&self) -> PyResult<PyPredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// Each simulation's ultimate, for a tower, a deal book or the cash-flow
    /// engine: index ``i`` is simulation ``i`` for every origin and every
    /// call, the same as row ``i`` of ``reserves`` and ``ultimates``, and
    /// the same again when the fit is run with the same seed.
    ///
    /// The ultimate is the origin's latest value plus its reserve in that
    /// simulation.
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

    /// Parameter names: ``c``, ``a[2]`` to ``a[n]``, ``b[2]`` to ``b[m]``.
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

    /// PSIS-LOO of the observed increments, with each cell's relative
    /// efficiency from the chains, as ``BayesGlmFit.loo``. The pointwise
    /// term is the normalised over-dispersed Poisson density at the cell's
    /// mean and ``scale``, so compare the result only with models of the
    /// same incremental values.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If an observed increment is negative (the density does not
    ///     exist there).
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l))
    }

    /// One row per origin: ``origin``, the chain ladder's ``latest``,
    /// ``ultimate`` and ``reserve``, and the ``mean`` and ``std_dev`` of
    /// the posterior predictive reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the chain ladder's totals, the ``scale``, and the ``mean``
    /// and ``std_dev`` of the posterior predictive total reserve. Needs
    /// pandas.
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
            "BayesOdpFit(origins={}, draws={}, n_sims={}, scale={:?}, warnings={})",
            self.inner.chain_ladder.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.scale,
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// ``BayesOdp.fit_segments``: each segment's own posterior and the joint
/// posterior predictive distribution of the reserves, laid out as
/// ``OdpBootstrapFit``.
///
/// ``reserves`` has the triangle's keys and ``"origin"`` as dimensions, so
/// ``reserves.aggregate(["lob"])`` keeps the dependence between segments.
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()`` and the components of ``reserves``. The
/// posterior of one segment (``scale``, ``diagnostics``,
/// ``parameter_names``, ``draws``, ``chains``, ``summary()`` and
/// ``loo()``) needs a single-segment fit: for several segments use
/// ``segment(...)``, ``totals_frame()`` (the scales) or
/// ``diagnostics_frame()``.
#[pyclass(name = "BayesOdpFits", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesOdpFits {
    inner: BayesOdpFits,
}

impl_one_segment!(PyBayesOdpFits => BayesOdpSegment);

#[pymethods]
impl PyBayesOdpFits {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Each segment is spread over its own posterior patterns, keeping the
    /// simulations together across segments.
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

    /// Each simulation's development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine.
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
    /// The volume-weighted chain ladder of every segment, whose latest
    /// values and Pearson scales the model uses.
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

    /// Origin period of each per-origin value and reserve component.
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

    /// The plugged-in scale ``phi`` of a single-segment fit.
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        Ok(self.one("scale", "totals_frame()")?.scale)
    }

    /// Joint posterior predictive distribution of the reserve by segment
    /// and origin: the triangle's keys and ``"origin"`` are its dimensions,
    /// one component per segment and origin, one row per simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Joint distribution of the ultimate by segment and origin: ``reserves`` with
    /// each origin's latest value added in every simulation, the same
    /// dimensions, components and rows, so ``join``, ``select`` and
    /// ``reorder_groups`` keep each simulation's origins together.
    #[getter]
    fn ultimates(&self) -> PyResult<PyPredictiveDistribution> {
        ultimates_of(&self.inner)
    }

    /// Each simulation's ultimate, for a tower, a deal book or the cash-flow
    /// engine: index ``i`` is simulation ``i`` for every origin and every
    /// call, the same as row ``i`` of ``reserves`` and ``ultimates``, and
    /// the same again when the fit is run with the same seed.
    ///
    /// The ultimate is the origin's latest value plus its reserve in that
    /// simulation.
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
    /// ``BayesOdpFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(
            py,
            &self.one("diagnostics", "diagnostics_frame()")?.diagnostics,
        )
    }

    /// Parameter names of a single-segment fit, as
    /// ``BayesOdpFit.parameter_names``.
    #[getter]
    fn parameter_names(&self) -> PyResult<Vec<String>> {
        Ok(self.one("parameter_names", "")?.posterior.names().to_vec())
    }

    /// Posterior draws of a single-segment fit, as ``BayesOdpFit.draws``.
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
    /// ``BayesOdpFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let one = self.one("summary()", "")?;
        let s = py.detach(|| one.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of a single-segment fit's observed increments, as
    /// ``BayesOdpFit.loo``.
    ///
    /// Returns
    /// -------
    /// Elpd
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments, or an observed increment is
    ///     negative.
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
    /// chain ladder's ``latest``, ``ultimate`` and ``reserve``, and the
    /// ``mean`` and ``std_dev`` of the posterior predictive reserve. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the chain ladder's totals,
    /// the ``scale``, and the ``mean`` and ``std_dev`` of the segment's
    /// posterior predictive total reserve. Needs pandas.
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
    /// ``ChainLadderFit.segment``, with its part of the joint reserves
    /// (same dimensions).
    ///
    /// Returns
    /// -------
    /// BayesOdpFits
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
        let scale = match segments.fits.as_slice() {
            [one] => format!(", scale={:?}", one.scale),
            _ => String::new(),
        };
        let warnings: usize = segments
            .fits
            .iter()
            .map(|f| f.diagnostics.warnings.len())
            .sum();
        format!(
            "BayesOdpFits({}origins={}, n_sims={}{scale}, warnings={warnings})",
            segments_prefix(segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}
