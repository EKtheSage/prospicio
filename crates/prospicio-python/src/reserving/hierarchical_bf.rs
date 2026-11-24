//! The hierarchical Bayesian Bornhuetter-Ferguson, `HierarchicalBf`.

use super::*;

/// The programs of a dict of name to one-segment `Triangle`, stacked into
/// one triangle with a segment per program under the key `program`, through
/// each triangle's long table (`HierarchicalBf.fit`).
fn stack_programs(programs: &Bound<'_, PyDict>, columns: &[&str]) -> PyResult<Triangle> {
    if programs.is_empty() {
        return Err(PyValueError::new_err("no programs: the dict is empty"));
    }
    let mut named: Vec<(String, PyRef<'_, PyTriangle>)> = Vec::with_capacity(programs.len());
    for (key, value) in programs.iter() {
        let name: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("program names must be strings"))?;
        let t = value
            .extract::<PyRef<'_, PyTriangle>>()
            .map_err(|_| PyTypeError::new_err(format!("program {name:?} is not a Triangle")))?;
        named.push((name, t));
    }
    // The core's rules (#838), the same in R.
    let refs: Vec<(&str, &Triangle)> = named.iter().map(|(n, t)| (n.as_str(), &t.inner)).collect();
    Triangle::stack(&refs, "program", columns).map_err(err)
}

/// The triangle of `HierarchicalBf.fit` and `.one_year`: a `Triangle` with
/// a segment per program, or a dict of program name to one-segment
/// `Triangle`, stacked.
fn programs_triangle(
    triangle: &Bound<'_, PyAny>,
    column: &str,
    exposure: &str,
) -> PyResult<Triangle> {
    if let Ok(programs) = triangle.cast::<PyDict>() {
        stack_programs(programs, &[column, exposure])
    } else if let Ok(t) = triangle.extract::<PyRef<'_, PyTriangle>>() {
        Ok(t.inner.clone())
    } else {
        Err(PyTypeError::new_err(
            "triangle must be a Triangle or a dict of program name to Triangle",
        ))
    }
}

/// The hierarchical Bayesian Bornhuetter-Ferguson: several programs, one
/// segment of a triangle each, pooled under an industry hyperprior on the
/// loss ratio, with the development pattern partially pooled
/// (``docs/design/reserving-bayes.md``, decision 30).
///
/// Each origin's cumulative-value increments are the over-dispersed Poisson
/// of ``BayesBfNormalised`` with mean ``U[c, w] * p[c, d]``, the origin's
/// expected ultimate times the share of it that falls at age ``d``. The
/// industry loss ratio ``L`` has a lognormal prior with mean ``apriori`` and
/// coefficient of variation ``level_cv`` (flat on ``log L`` when
/// ``level_cv`` is ``None``, so the programs alone set the level). Program
/// ``c`` has loss ratio ``L * s[c]`` with ``log s[c]`` normal with sd
/// ``tau`` and ``E[s[c]] = 1``, ``tau`` half-normal with scale
/// ``tau_scale``; its origins' expected ultimates are gamma with mean
/// ``L * s[c] * exposure`` and coefficient of variation ``cv``. The shared
/// log pattern ``beta[d]`` is normal with sd ``development_sd`` (or the
/// ``pattern_prior``), and each program's own pattern departs from it by
/// ``omega`` times a standard normal per age, ``omega`` half-normal with
/// scale ``omega_scale``. A thin program's young ages therefore move with
/// its own data, its later ages follow the shared pattern, and its loss
/// ratio is shrunk towards ``L`` by as much as the spread between the
/// programs supports.
///
/// The scale is plugged in per unit of premium, pooled over the programs
/// (``scale_per_premium``), so a program of two origins, which has no
/// degree of freedom of its own, still has a scale. The panel together must
/// have a degree of freedom. Failed MCMC diagnostics (R-hat above 1.01,
/// bulk or tail ESS below 100 per chain, any divergent transition) do not
/// fail the fit: ``fit`` raises one ``UserWarning`` listing them, and they
/// stay in ``HierarchicalBfFit.diagnostics``. One ``seed`` replays a fit
/// exactly. The one-year view is not available yet.
///
/// Parameters
/// ----------
/// apriori : float, default 1.0
///     Prior mean of the industry loss ratio ``L``; positive and finite.
/// level_cv : float, optional
///     Coefficient of variation of the lognormal prior on ``L``; ``None``
///     (the default) is flat on ``log L``.
/// cv : float, default 0.1
///     Coefficient of variation of each origin's expected ultimate about
///     its program's level; fixed, finite and positive.
/// development_sd : float, default 10.0
///     Standard deviation of the normal prior on each shared ``beta[d]``;
///     unused with a ``pattern_prior``.
/// tau_scale : float, default 0.3
///     Scale of the half-normal prior on ``tau``, the spread of the
///     programs' log loss ratios (a median spread of about 20%).
/// omega_scale : float, default 0.5
///     Scale of the half-normal prior on ``omega``, the spread of the
///     programs' log share ratios.
/// process : {"gamma", "none"}, default "gamma"
///     Process error on each simulated future increment.
/// chains, tune, draws : int, default 4, 1000, 1000
///     NUTS chains, warm-up draws per chain and kept draws per chain.
/// n_sims : int, default 10000
///     Number of predictive simulations; positive.
/// seed : int, default 0
///     Seed of the chains and of the simulations.
/// target_accept : float, default 0.99
///     Target mean acceptance rate of the step-size adaptation.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
/// pattern_prior : PatternPrior, optional
///     An informative prior on the shared pattern, for example
///     ``IndustryPrior.pattern``; its ages must be the programs' ages.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero or ``process`` is unknown. ``fit`` raises it
///     when falling columns reach ``1 / omega_scale`` in the programs'
///     scales, jointly over the programs, or when a program's falling
///     columns exceed its first age: such a panel has no posterior.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import HierarchicalBf, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2021, 2021, 2021, 2022, 2022, 2023, 2022, 2022, 2023],
/// ...     [12, 24, 36, 12, 24, 12, 12, 24, 12],
/// ...     {
/// ...         "paid": [100.0, 150.0, 170.0, 110.0, 172.0, 130.0, 55.0, 80.0, 60.0],
/// ...         "premium": [250.0] * 3 + [270.0] * 2 + [290.0] + [120.0] * 2 + [130.0],
/// ...     },
/// ...     keys={"program": ["a"] * 6 + ["b"] * 3},
/// ... )
/// >>> import warnings
/// >>> warnings.simplefilter("ignore")  # two tiny programs: the sampler is short
/// >>> model = HierarchicalBf(apriori=0.7, level_cv=0.2, chains=2, tune=200, draws=200, n_sims=500, seed=1)
/// >>> fit = model.fit(tri, "paid", "premium")
/// >>> [p.label for p in fit.programs]
/// ['a', 'b']
/// >>> fit.reserves.dims
/// ['program', 'origin']
#[pyclass(name = "HierarchicalBf", module = "prospicio.reserving", frozen)]
pub(crate) struct PyHierarchicalBf {
    inner: HierarchicalBf,
    pattern_prior: Option<PatternPrior>,
}

#[pymethods]
impl PyHierarchicalBf {
    #[new]
    #[pyo3(signature = (
        apriori = 1.0,
        level_cv = None,
        cv = 0.1,
        development_sd = 10.0,
        tau_scale = 0.3,
        omega_scale = 0.5,
        process = "gamma",
        chains = 4,
        tune = 1000,
        draws = 1000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.99,
        max_depth = 10,
        pattern_prior = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: f64,
        level_cv: Option<f64>,
        cv: f64,
        development_sd: f64,
        tau_scale: f64,
        omega_scale: f64,
        process: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        pattern_prior: Option<PyRef<'_, PyPatternPrior>>,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: HierarchicalBf {
                apriori,
                level_cv,
                cv,
                development_sd,
                tau_scale,
                omega_scale,
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
            },
            pattern_prior: pattern_prior.map(|p| p.inner.clone()),
        })
    }

    /// Prior mean of the industry loss ratio.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// Coefficient of variation of the industry loss ratio's prior, or
    /// ``None`` for a flat prior on its log.
    #[getter]
    fn level_cv(&self) -> Option<f64> {
        self.inner.level_cv
    }

    /// Coefficient of variation of each origin's expected ultimate.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv
    }

    /// Standard deviation of each shared development effect's prior.
    #[getter]
    fn development_sd(&self) -> f64 {
        self.inner.development_sd
    }

    /// Scale of the half-normal prior on ``tau``.
    #[getter]
    fn tau_scale(&self) -> f64 {
        self.inner.tau_scale
    }

    /// Scale of the half-normal prior on ``omega``.
    #[getter]
    fn omega_scale(&self) -> f64 {
        self.inner.omega_scale
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

    /// The prior on the shared pattern, or ``None`` for the vague
    /// ``N(0, development_sd**2)``.
    #[getter]
    fn pattern_prior(&self) -> Option<PyPatternPrior> {
        self.pattern_prior
            .clone()
            .map(|inner| PyPatternPrior { inner })
    }

    /// Samples the posterior of one loss column of a cumulative triangle
    /// with a segment per program, with each origin's exposure from
    /// another column, and simulates every program's reserves jointly.
    /// Every origin must be observed from the first age up to its latest.
    /// A triangle with one segment is one program. Raises one
    /// ``UserWarning`` listing the failed diagnostics, if any.
    ///
    /// Instead of one triangle, ``triangle`` may be a dict of program name
    /// to a ``Triangle`` with one segment. The dict is stacked into a
    /// triangle with a segment per program under the key ``"program"``
    /// (any keys the triangles carry are dropped). The names label the
    /// programs, which run in the sorted order of their names, as the
    /// segments of a triangle do, not in the dict's order. The triangles
    /// must all have the two columns, the same grains and the same
    /// cumulative state. The fit is then the same, draw for draw, as the
    /// fit of the same panel given as one triangle with that key.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle or dict of str to Triangle
    ///     Cumulative, with a segment per program, all on the same ages; or
    ///     the programs by name, each a triangle with one segment.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure.
    ///
    /// Returns
    /// -------
    /// HierarchicalBfFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the triangle is not cumulative or has an origin with a gap or
    ///     without exposure, if a setting is invalid, if the programs
    ///     together have no degree of freedom, or if a development column
    ///     or origin sums to too little (the error names it). For a dict:
    ///     if it is empty, a name is empty, a triangle has several
    ///     segments or lacks a column, or the grains or cumulative state
    ///     differ.
    /// TypeError
    ///     If a dict's key is not a string or its value not a ``Triangle``.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: &Bound<'_, PyAny>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyHierarchicalBfFit> {
        let tri = programs_triangle(triangle, column, exposure)?;
        let (model, prior) = (&self.inner, &self.pattern_prior);
        let inner = py
            .detach(|| match prior {
                Some(p) => model.fit_with_pattern_prior(&tri, column, exposure, p),
                None => model.fit(&tri, column, exposure),
            })
            .map_err(err)?;
        warn_diagnostics(py, "HierarchicalBf", &inner.diagnostics.warnings)?;
        Ok(PyHierarchicalBfFit { inner })
    }

    /// The one-year view of the fit over every program at once: the
    /// claims development result of each program and origin over the
    /// coming year, by importance reweighting of the posterior by the
    /// year's cells, as ``BayesBfNormalised.one_year`` does for one
    /// triangle (``docs/design/reserving-bayes.md``, decision 30). It is
    /// one view over the stacked programs and not one per program, because
    /// a program's next diagonal moves the industry level, the spreads and
    /// the shared pattern, and through them the other programs' closing
    /// reserves. A refit of a flagged simulation samples the whole panel.
    /// Raises ``UserWarning`` for the fit's failed diagnostics, for refits
    /// whose diagnostics failed, for simulations left unrefitted by
    /// ``max_refits``, and for each of the view's ``warnings``: programs
    /// whose next diagonal falls at development ages they have never
    /// observed, which make the view expensive (nearly every simulation is
    /// flagged and refitted) or, with ``refit="reweight_only"``,
    /// approximate.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle or dict of str to Triangle
    ///     As in ``fit``.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, optional
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    ///     ``None`` (the default) is ``"moment_match_then_refit"`` with the
    ///     model's default sampler of the refits (``light_refit``).
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations; by default this model's
    ///     ``n_sims``, at most 2,000. A smaller count gives the first
    ///     simulations of a larger one.
    /// light_refit : bool or tuple, optional
    ///     The sampler of the refits: a shorter, warm-started NUTS run
    ///     instead of the fit's own. ``None`` (the default) is the model's
    ///     own when ``refit`` is ``None``, the light refit of ``True`` below,
    ///     and the fit's sampler, with its pilot run, when ``refit`` is given.
    ///     ``True`` takes 4 chains of 150 warm-up and 250 kept draws, a
    ///     tuple ``(chains, tune, draws)`` its own, and ``False`` turns a default
    ///     light refit off. Each refit scales its sampler by the opening
    ///     posterior and starts at its mean, with no pilot run, and warns when
    ///     its closing reserve has R-hat above 1.05, fewer than 100 effective
    ///     draws or a divergence.
    ///
    /// Returns
    /// -------
    /// HierarchicalBfOneYearFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    /// TypeError
    ///     As ``fit``.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import HierarchicalBf, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2021, 2021, 2021, 2022, 2022, 2023, 2022, 2022, 2023],
    /// ...     [12, 24, 36, 12, 24, 12, 12, 24, 12],
    /// ...     {
    /// ...         "paid": [100.0, 150.0, 170.0, 110.0, 172.0, 130.0, 55.0, 80.0, 60.0],
    /// ...         "premium": [250.0] * 3 + [270.0] * 2 + [290.0] + [120.0] * 2 + [130.0],
    /// ...     },
    /// ...     keys={"program": ["a"] * 6 + ["b"] * 3},
    /// ... )
    /// >>> import warnings
    /// >>> warnings.simplefilter("ignore")  # two tiny programs: the sampler is short
    /// >>> model = HierarchicalBf(apriori=0.7, level_cv=0.2, chains=2, tune=200, draws=200, n_sims=500, seed=1)
    /// >>> view = model.one_year(tri, "paid", "premium", refit="reweight_only")
    /// >>> view.program
    /// ['a', 'a', 'a', 'b', 'b']
    /// >>> view.cdr.dims
    /// ['program', 'origin']
    /// >>> len(view.pareto_k)
    /// 500
    #[pyo3(signature = (triangle, column, exposure, k_threshold = 0.7, refit = None, max_refits = None, n_sims = None, light_refit = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: &Bound<'_, PyAny>,
        column: &str,
        exposure: &str,
        k_threshold: f64,
        refit: Option<&str>,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
        light_refit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyHierarchicalBfOneYearFit> {
        let mut settings = one_year_settings(k_threshold, refit, max_refits, n_sims)?;
        (settings.light_refit, settings.no_light_refit) = light_refit_arg(light_refit)?;
        let tri = programs_triangle(triangle, column, exposure)?;
        let (model, prior) = (&self.inner, &self.pattern_prior);
        let inner = py
            .detach(|| match prior {
                Some(p) => model.one_year_with_pattern_prior(&tri, column, exposure, &settings, p),
                None => model.one_year(&tri, column, exposure, &settings),
            })
            .map_err(err)?;
        warn_diagnostics(py, "HierarchicalBf", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "HierarchicalBf",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        for message in &inner.warnings {
            let message = std::ffi::CString::new(format!("HierarchicalBf: {message}"))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyHierarchicalBfOneYearFit { inner })
    }

    fn __repr__(&self) -> String {
        let s = &self.inner.sampler;
        format!(
            "HierarchicalBf(apriori={:?}, level_cv={}, cv={:?}, development_sd={:?}, \
             tau_scale={:?}, omega_scale={:?}, process={:?}, chains={}, tune={}, draws={}, \
             n_sims={}, seed={}{})",
            self.inner.apriori,
            self.inner
                .level_cv
                .map_or_else(|| "None".to_string(), |c| format!("{c:?}")),
            self.inner.cv,
            self.inner.development_sd,
            self.inner.tau_scale,
            self.inner.omega_scale,
            self.process(),
            s.chains,
            s.tune,
            s.draws,
            self.inner.n_sims,
            self.inner.seed,
            if self.pattern_prior.is_some() {
                ", pattern_prior=PatternPrior(...)"
            } else {
                ""
            },
        )
    }
}

/// A fitted ``HierarchicalBf``: the posterior of the industry loss ratio
/// ``level``, the spreads ``tau`` and ``omega``, and every program's
/// effects, its MCMC diagnostics, and the joint posterior predictive
/// distribution of every program's reserves by origin (``reserves``, with
/// the triangle's keys and ``"origin"`` as dimensions, paired by posterior
/// draw because the programs share parameters). ``programs`` holds what
/// the fit says about each program.
#[pyclass(name = "HierarchicalBfFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyHierarchicalBfFit {
    inner: HierarchicalBfFit,
}

#[pymethods]
impl PyHierarchicalBfFit {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// The hierarchical BF has no pattern per simulation: this raises
    /// ``NotSupportedError``.
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

    /// The programs, in the triangle's segment order.
    ///
    /// Returns
    /// -------
    /// list of HierarchicalBfProgram
    #[getter]
    fn programs(&self) -> Vec<PyHierarchicalBfProgram> {
        self.inner
            .programs
            .iter()
            .map(|p| PyHierarchicalBfProgram { inner: p.clone() })
            .collect()
    }

    /// The development ages of every program's pattern, in months.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner.ages.clone()
    }

    /// The pooled scale per unit of premium, ``kappa``: a program's scale
    /// is ``kappa`` times its mean premium per origin.
    #[getter]
    fn scale_per_premium(&self) -> f64 {
        self.inner.scale_per_premium
    }

    /// Joint posterior predictive distribution of the reserve (the sum of
    /// future incremental values) by program and origin: one component per
    /// program and origin, one row per simulation. Each program's part is
    /// its ``reserves``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
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

    /// Parameter names, in the order of the columns of ``draws``:
    /// ``log_level``, ``log_tau``, ``z[c]``, ``alpha[c,w]``, ``beta[d]``,
    /// ``log_omega`` and ``delta[c,d]``.
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

    /// Posterior summary, one row per sampled parameter: ``parameter``,
    /// ``mean``, ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk``
    /// and ``ess_tail``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// The posterior of the hyperparameters on their natural scale, in the
    /// columns of ``summary``: ``level`` (the industry loss ratio), ``tau``
    /// (the spread of the programs' loss ratios) and ``omega`` (the spread
    /// of their patterns). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn hyperparameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.hyperparameters()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// One row per program: ``program``, ``origins`` (their number),
    /// ``exposure``, ``latest``, ``prior_ultimate`` and
    /// ``expected_ultimate`` (summed over the origins), ``scale``,
    /// ``loss_ratio``, ``own_loss_ratio``, ``shrinkage`` and the ``mean``
    /// and ``std_dev`` of the program's posterior predictive reserve.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // The core's table (#838), the same in R; origins is a count.
        let frame = table_frame(py, self.inner.totals())?;
        let origins = frame
            .get_item("origins")?
            .call_method1("astype", ("int64",))?;
        frame.set_item("origins", origins)?;
        Ok(frame)
    }

    fn __repr__(&self) -> String {
        format!(
            "HierarchicalBfFit(programs={}, ages={}, draws={}, n_sims={}, warnings={})",
            self.inner.programs.len(),
            self.inner.ages.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// One program of a ``HierarchicalBfFit``. Per-origin lists run over the
/// program's origins in order (``origins``).
#[pyclass(name = "HierarchicalBfProgram", module = "prospicio.reserving", frozen)]
pub(crate) struct PyHierarchicalBfProgram {
    inner: HierarchicalBfProgram,
}

#[pymethods]
impl PyHierarchicalBfProgram {
    /// The program's segment label: the key values joined by ``" / "``.
    #[getter]
    fn label(&self) -> String {
        self.inner.label.to_string()
    }

    /// The program's key values, one per key of the triangle.
    #[getter]
    fn key(&self) -> Vec<String> {
        self.inner.label.parts().to_vec()
    }

    /// Origin period of each origin.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// Exposure per origin: the exposure column's latest observed value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        self.inner.exposure.clone()
    }

    /// The latest observed cumulative value of each origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.inner.latest.clone()
    }

    /// The hyperprior's mean of each origin's expected ultimate,
    /// ``apriori * exposure``.
    #[getter]
    fn prior_ultimate(&self) -> Vec<f64> {
        self.inner.prior_ultimate.clone()
    }

    /// The plugged-in scale, ``scale_per_premium`` times the program's mean
    /// premium per origin.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// The posterior mean of each origin's expected ultimate.
    #[getter]
    fn expected_ultimate(&self) -> Vec<f64> {
        self.inner.expected_ultimate.clone()
    }

    /// The posterior mean of the program's share at each of the fit's
    /// ``ages``; the shares sum to 1.
    #[getter]
    fn pattern(&self) -> Vec<f64> {
        self.inner.pattern.clone()
    }

    /// The posterior mean of the program's loss ratio ``L * s``.
    #[getter]
    fn loss_ratio(&self) -> f64 {
        self.inner.loss_ratio
    }

    /// The loss ratio of the program's own chain ladder (its ultimates over
    /// its premium, with the factors pooled over every program past the
    /// ages it observes).
    #[getter]
    fn own_loss_ratio(&self) -> f64 {
        self.inner.own_loss_ratio
    }

    /// How far the program stays from the collective on the log scale: 1
    /// keeps its own level and 0 takes the collective's; ``nan`` when its
    /// own loss ratio is not positive.
    #[getter]
    fn shrinkage(&self) -> f64 {
        self.inner.shrinkage
    }

    /// The program's part of the fit's ``reserves``, by origin, paired
    /// with it simulation by simulation.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HierarchicalBfProgram(label='{}', origins={}, loss_ratio={:?}, own_loss_ratio={:?}, \
             shrinkage={:?})",
            self.inner.label,
            self.inner.origins.len(),
            self.inner.loss_ratio,
            self.inner.own_loss_ratio,
            self.inner.shrinkage,
        )
    }
}

/// The one-year view of a ``HierarchicalBf`` over every program at once,
/// from ``HierarchicalBf.one_year``: the lifetime fit, the opening reserve,
/// the claims development result over the coming year and how reliable each
/// simulation's importance reweighting was.
///
/// It is one view over the stacked programs, not one per program, because
/// a program's next diagonal moves the industry level, the spreads and the
/// shared pattern, and through them the other programs' closing reserves.
/// ``cdr`` is the joint distribution of the claims development result by
/// program and origin (the triangle's keys and ``"origin"`` as dimensions),
/// the opening reserve less the year's payment and the posterior mean
/// reserve after the year, so a negative value is an adverse development
/// and ``cdr.aggregate(["program"])`` is each program's result with the
/// dependence between programs kept. Per-origin lists run over every
/// program's origins in turn, program by program, like the components of
/// ``cdr``. ``pareto_k`` has one Pareto ``k`` per simulation;
/// ``flagged_share`` is the share above ``k_threshold``, whether or not
/// they were refitted, ``refitted`` lists the simulations refitted with
/// NUTS and ``moment_matched`` those moment matching rescued.
#[pyclass(
    name = "HierarchicalBfOneYearFit",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyHierarchicalBfOneYearFit {
    inner: BayesOneYearFit<HierarchicalBfFit>,
}

#[pymethods]
impl PyHierarchicalBfOneYearFit {
    /// The model fitted to the observed triangle, with its posterior and
    /// lifetime reserves, as ``HierarchicalBf.fit`` gives it.
    ///
    /// Returns
    /// -------
    /// HierarchicalBfFit
    #[getter]
    fn fit(&self) -> PyHierarchicalBfFit {
        PyHierarchicalBfFit {
            inner: self.inner.fit.clone(),
        }
    }

    /// The program of each per-origin value and CDR component: the
    /// segment label, repeated over its origins.
    #[getter]
    fn program(&self) -> Vec<String> {
        self.inner
            .fit
            .programs
            .iter()
            .flat_map(|p| std::iter::repeat_n(p.label.to_string(), p.origins.len()))
            .collect()
    }

    /// Origin period of each per-origin value and CDR component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .fit
            .programs
            .iter()
            .flat_map(|p| p.origins.iter().map(ToString::to_string))
            .collect()
    }

    /// Latest observed value per program and origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.inner
            .fit
            .programs
            .iter()
            .flat_map(|p| p.latest.iter().copied())
            .collect()
    }

    /// The latest value plus the opening reserve, per program and origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        self.inner.opening_ultimate.clone()
    }

    /// Each origin's posterior mean reserve on the observed triangle: the
    /// mean over the posterior draws of the expected future increments.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        self.inner.opening_reserve.clone()
    }

    /// Joint distribution of the claims development result by program and
    /// origin: one component per program and origin, one row per
    /// simulation.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    /// Each simulation's Pareto ``k``, in the order of the simulations
    /// (``inf`` when too few weights differ to fit the tail).
    #[getter]
    fn pareto_k(&self) -> Vec<f64> {
        self.inner.pareto_k.clone()
    }

    /// The simulations refitted with NUTS, in order (0-based).
    #[getter]
    fn refitted(&self) -> Vec<usize> {
        self.inner.refitted.clone()
    }

    /// The simulations moment matching rescued (their ``k`` brought to at
    /// most ``k_threshold``), in order (0-based); empty unless ``refit``
    /// was ``"moment_match_then_refit"``, when each flagged simulation is
    /// in this, in ``refitted`` or in ``unrefitted``.
    #[getter]
    fn moment_matched(&self) -> Vec<usize> {
        self.inner.moment_matched.clone()
    }

    /// The simulations due for a refit beyond ``max_refits``, reweighted
    /// instead, in order (0-based); empty without ``max_refits``.
    #[getter]
    fn unrefitted(&self) -> Vec<usize> {
        self.inner.unrefitted.clone()
    }

    /// How many of the refits had a diagnostic warning (R-hat, ESS or a
    /// divergence).
    #[getter]
    fn refit_warnings(&self) -> usize {
        self.inner.refit_warnings
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        self.inner.k_threshold
    }

    /// The share of simulations whose ``k`` is above ``k_threshold`` (or
    /// NaN), whether or not they were refitted.
    #[getter]
    fn flagged_share(&self) -> f64 {
        self.inner.flagged_share()
    }

    /// What else the view warns of, one message each (``one_year`` raises
    /// each as a ``UserWarning``): the programs whose next diagonal falls at
    /// development ages they have never observed, which make the view
    /// expensive (or, reweighting only, approximate). Empty when the
    /// programs share their ages.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings.clone()
    }

    /// One row per program and origin: ``program``, ``origin``, ``latest``,
    /// ``opening_ultimate``, ``opening_reserve`` and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per program: the ``latest`` value, ``opening_ultimate`` and
    /// ``opening_reserve`` summed over its origins, and the ``cdr_mean``
    /// and ``cdr_std_dev`` of the program's claims development result (its
    /// origins' sum), with the ``flagged_share`` of the whole view. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    fn __repr__(&self) -> String {
        format!(
            "HierarchicalBfOneYearFit(programs={}, origins={}, n_sims={}, flagged_share={:?}, \
             refitted={})",
            self.inner.fit.programs.len(),
            self.inner.opening_reserve.len(),
            self.inner.cdr.n_sims(),
            self.inner.flagged_share(),
            self.inner.refitted.len(),
        )
    }
}
