//! The Bayesian Bornhuetter-Ferguson with its prior on the ultimate, `BayesBfNormalised`.

use super::*;

/// The Bayesian Bornhuetter–Ferguson whose prior sits on the ultimate: the
/// over-dispersed Poisson model of a cumulative triangle's incremental
/// values ``X`` with mean ``U[w] * p[d]``, the origin's expected ultimate
/// ``U[w]`` times the share ``p[d]`` of it that falls at age ``d``, the
/// shares summing to 1 (``p[d] = exp(beta[d]) / sum(exp(beta))`` with
/// ``beta[1] = 0``). The informative gamma prior is on ``U[w]`` itself,
/// with mean ``apriori * exposure[w]`` and coefficient of variation
/// ``cv``, so ``apriori`` is the expected ultimate loss ratio and ``cv``
/// says how firmly it is held, as in the classical
/// ``BornhuetterFerguson``. The shares have a vague normal prior on their
/// log ratios to the first age's, and the chain ladder's Pearson scale
/// ``phi`` (``OdpBootstrapFit.scale``) is plugged in. The posterior is
/// sampled with NUTS, and each draw projects the future cells up to the
/// oldest observed age (no tail) with the process error of ``process``, as
/// ``BayesOdp`` does.
///
/// ``BayesBf`` (England, Verrall and Wüthrich's Model 3.1) instead
/// leaves the pattern unnormalised and puts its prior on ``mu[w]``, which
/// is not the ultimate: there the level of ``apriori`` is cancelled by the
/// pattern and barely moves the reserve. Use this model when the
/// a-priori loss ratio is a judgment about the ultimate; use ``BayesBf``
/// to match England, Verrall and Wüthrich's published results.
///
/// The prior's strength moves the reserve between two limits. With
/// ``cv=float("inf")`` (a flat prior on ``log U[w]``) the model is the
/// Bayesian chain ladder, and the reserves agree with ``BayesOdp``'s within
/// Monte Carlo error. As ``cv`` falls to 0, ``U[w]`` is fixed at the prior
/// ultimate ``m[w]`` and the shares are ``S[d] / (M[d] + lambda)`` (``S[d]``
/// the observed increments of column ``d``, ``M[d]`` the prior ultimates of
/// the origins observed there, ``lambda`` such that they sum to 1). When
/// the prior ultimates are the chain ladder's, ``lambda`` is 0 and the
/// reserve is exactly the classical ``BornhuetterFerguson`` reserve
/// ``(1 - 1 / cdf) * m[w]``; otherwise the shares move with the ultimates,
/// where the classical method keeps the chain ladder's pattern.
///
/// Every development column's observed increments must sum to more than
/// zero, and every origin's to more than ``-phi / cv**2``; negative cells
/// are allowed. The error names the age or origin, its number of observed
/// cells, their sum and the remedies. A column that sums to zero or less
/// has no log-linear mean, since every mean is an ultimate times a
/// positive share; the chain ladder accepts the same triangle, and
/// ``BayesOdp`` and ``BayesBf`` refuse it too. To drop a sparse last age,
/// keep the ages before it with ``Triangle.max_age`` (``tri.max_age(108)``
/// for a last age of 120 months) and carry the movement left out as a tail
/// outside the model. A falling first age is dropped by rebuilding the
/// triangle without its rows (``Triangle.from_frame`` on the rows of
/// ``tri.to_frame()`` at the other ages); a falling middle age has no
/// remedy in this model. For an origin, a finite ``cv`` below the bound
/// the error prints makes its posterior proper. One ``seed`` replays a fit
/// exactly, whatever the number of threads. Failed MCMC diagnostics (R-hat
/// above 1.01, bulk or tail ESS below 100 per chain, any divergent
/// transition) do not fail the fit: ``fit`` raises one ``UserWarning``
/// listing them, and they stay in ``BayesBfNormalisedFit.diagnostics``.
///
/// Parameters
/// ----------
/// apriori : float, default 1.0
///     Prior mean of each origin's expected ultimate ``U[w]`` per unit of
///     exposure, the expected ultimate loss ratio; positive and finite.
/// cv : float, default 0.1
///     Coefficient of variation of the gamma prior on each ``U[w]`` (shape
///     ``1 / cv**2``); positive, ``float("inf")`` for a flat prior.
/// development_sd : float, default 10.0
///     Standard deviation of the normal prior on each ``beta[d]`` (the log
///     ratio of the share at age ``d`` to the first age's), centred on 0.
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
///     Target mean acceptance rate of the step-size adaptation.
/// max_depth : int, default 10
///     Largest NUTS tree depth.
/// common_cv : float, optional
///     The coefficient of variation of one gamma multiplier (mean 1) common
///     to all origins' prior ultimates; ``None`` has none.
/// pattern_prior : PatternPrior, optional
///     An informative prior on the pattern: each ``beta[d]`` is normal with
///     the prior's mean and sd in place of ``N(0, development_sd**2)``, for
///     example ``IndustryPrior.pattern``. Its ages must be the fitted
///     triangle's (``PatternPrior.truncated`` or ``IndustryPrior.at_age``
///     cut a longer one). ``None`` (the default) keeps the vague prior.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero or ``process`` is unknown.
/// OverflowError
///     If an integer setting is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import BayesBfNormalised, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     {
/// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ...         "premium": [200.0] * 4 + [220.0] * 3 + [240.0] * 2 + [260.0],
/// ...     },
/// ... )
/// >>> fit = BayesBfNormalised(apriori=0.85, cv=0.1, n_sims=2000, seed=1).fit(tri, "paid", "premium")
/// >>> fit.reserves.components()
/// [('2020',), ('2021',), ('2022',), ('2023',)]
/// >>> fit.prior_ultimate[3] == 0.85 * 260.0
/// True
/// >>> fit.parameter_names[:2]
/// ['alpha[1]', 'alpha[2]']
#[pyclass(name = "BayesBfNormalised", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesBfNormalised {
    pub(crate) inner: BayesBfNormalised,
    /// The pattern prior its fits take, if any (not a setting of the Rust
    /// model, which takes it at the fit: decision 29).
    pub(crate) pattern_prior: Option<PatternPrior>,
}

#[pymethods]
impl PyBayesBfNormalised {
    #[new]
    #[pyo3(signature = (
        apriori = 1.0,
        cv = 0.1,
        development_sd = 10.0,
        process = "gamma",
        chains = 4,
        tune = 1000,
        draws = 1000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.95,
        max_depth = 10,
        common_cv = None,
        pattern_prior = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        apriori: f64,
        cv: f64,
        development_sd: f64,
        process: &str,
        chains: usize,
        tune: usize,
        draws: usize,
        n_sims: usize,
        seed: u64,
        target_accept: f64,
        max_depth: u64,
        common_cv: Option<f64>,
        pattern_prior: Option<PyRef<'_, PyPatternPrior>>,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: BayesBfNormalised {
                apriori,
                cv,
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
                common_cv,
            },
            pattern_prior: pattern_prior.map(|p| p.inner.clone()),
        })
    }

    /// The coefficient of variation of the multiplier common to all
    /// origins' prior ultimates, or ``None`` if there is none.
    #[getter]
    fn common_cv(&self) -> Option<f64> {
        self.inner.common_cv
    }

    /// The prior on the pattern, or ``None`` for the vague
    /// ``N(0, development_sd**2)``.
    #[getter]
    fn pattern_prior(&self) -> Option<PyPatternPrior> {
        self.pattern_prior
            .clone()
            .map(|inner| PyPatternPrior { inner })
    }

    /// Prior mean of each origin's expected ultimate per unit of exposure.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// Coefficient of variation of each origin's prior; ``inf`` is flat.
    #[getter]
    fn cv(&self) -> f64 {
        self.inner.cv
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

    /// Samples the posterior of one loss column of a single-segment
    /// cumulative triangle, with each origin's exposure from another
    /// column, and simulates its reserves. Every origin must be observed
    /// from the first age up to its latest and there must be more observed
    /// cells than chain-ladder parameters. Raises one ``UserWarning``
    /// listing the failed diagnostics, if any.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure, as for ``BornhuetterFerguson``.
    ///
    /// Returns
    /// -------
    /// BayesBfNormalisedFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``OdpBootstrap.fit``; if the triangle has several segments; if
    ///     an origin has no observed, finite, positive exposure; if
    ///     ``apriori``, ``cv`` or ``development_sd`` is invalid or a sampler
    ///     setting is; if a development column's increments do not sum to
    ///     more than zero; or if an origin's sum to ``-phi / cv**2`` or
    ///     less.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyBayesBfNormalisedFit> {
        let (model, tri, prior) = (&self.inner, &triangle.inner, &self.pattern_prior);
        let inner = py
            .detach(|| match prior {
                Some(p) => model.fit_with_pattern_prior(tri, column, exposure, p),
                None => model.fit(tri, column, exposure),
            })
            .map_err(err)?;
        let warnings = &inner.diagnostics.warnings;
        if !warnings.is_empty() {
            let message = format!(
                "BayesBfNormalised: the MCMC diagnostics failed; check them before relying on the fit: {}",
                warnings.join("; ")
            );
            let message = std::ffi::CString::new(message)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyBayesBfNormalisedFit { inner })
    }

    /// The one-year view of one loss column of a single-segment cumulative
    /// triangle, with each origin's exposure from another column: the
    /// claims development result over the coming year, by Pareto smoothed
    /// importance sampling of ``fit``'s posterior (decisions 16 and 17 of
    /// ``docs/design/reserving-bayes.md``), exactly as
    /// ``BayesOdp.one_year`` with this model's means
    /// ``exp(alpha[w]) * p[d]``. Each simulation draws the year's increments
    /// from a posterior draw with ``process`` error, reweights every
    /// posterior draw by the year's quasi likelihood (``scale``, the
    /// priors and the prior ultimates unchanged), and takes each origin's
    /// claims development result as its posterior mean reserve at the
    /// opening less its payment in the year and its weighted posterior mean
    /// reserve after it. A simulation whose Pareto ``k`` is above
    /// ``k_threshold`` is handled as ``refit`` says: by default
    /// moment-matched, and refitted with NUTS on the extended triangle
    /// only if matching cannot bring its ``k`` to the threshold. A flat
    /// prior (``cv=inf``) gives about ``BayesOdp.one_year``'s view; a
    /// tighter prior holds each origin's ultimate more firmly and gives a
    /// narrower one. Raises one ``UserWarning`` listing the failed
    /// diagnostics of the fit, and another if any refit failed its
    /// diagnostics.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column, as in ``fit``.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, default "moment_match_then_refit"
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations; by default this model's
    ///     ``n_sims``. A smaller count gives the first simulations of a
    ///     larger one.
    ///
    /// Returns
    /// -------
    /// BayesOneYearFit
    ///     With ``fit`` a ``BayesBfNormalisedFit``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import BayesBfNormalised, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    /// ...     {
    /// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    /// ...         "premium": [200.0] * 4 + [220.0] * 3 + [240.0] * 2 + [260.0],
    /// ...     },
    /// ... )
    /// >>> model = BayesBfNormalised(apriori=0.85, cv=0.1, n_sims=1000, seed=1)
    /// >>> view = model.one_year(tri, "paid", "premium", refit="reweight_only")
    /// >>> view.cdr.components()
    /// [('2020',), ('2021',), ('2022',), ('2023',)]
    /// >>> type(view.fit).__name__, len(view.pareto_k)
    /// ('BayesBfNormalisedFit', 1000)
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (triangle, column, exposure, k_threshold = 0.7, refit = "moment_match_then_refit", max_refits = None, n_sims = None))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
        k_threshold: f64,
        refit: &str,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
    ) -> PyResult<PyBayesOneYearFit> {
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, n_sims)?;
        let (model, tri, prior) = (&self.inner, &triangle.inner, &self.pattern_prior);
        let inner = py
            .detach(|| match prior {
                Some(p) => model.one_year_with_pattern_prior(tri, column, exposure, &settings, p),
                None => model.one_year(tri, column, exposure, &settings),
            })
            .map_err(err)?;
        warn_diagnostics(py, "BayesBfNormalised", &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "BayesBfNormalised",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        Ok(PyBayesOneYearFit {
            inner: BayesOneYearOf::BfNormalised(inner),
        })
    }

    fn __repr__(&self) -> String {
        let s = &self.inner.sampler;
        format!(
            "BayesBfNormalised(apriori={:?}, cv={:?}, development_sd={:?}, process={:?}, chains={}, \
             tune={}, draws={}, n_sims={}, seed={}{}{})",
            self.inner.apriori,
            self.inner.cv,
            self.inner.development_sd,
            self.process(),
            s.chains,
            s.tune,
            s.draws,
            self.inner.n_sims,
            self.inner.seed,
            self.inner
                .common_cv
                .map_or_else(String::new, |c| format!(", common_cv={c:?}")),
            if self.pattern_prior.is_some() {
                ", pattern_prior=PatternPrior(...)"
            } else {
                ""
            },
        )
    }
}

/// A fitted ``BayesBfNormalised``: the posterior of ``alpha[1..n]`` (the log
/// expected ultimates) and ``beta[2..m]`` (the log ratios of the shares to
/// the first age's), its MCMC diagnostics, and the
/// posterior predictive distribution of the reserves, laid out as
/// ``BayesOdpFit.reserves``. The frames' ``latest``, ``ultimate`` and
/// ``reserve`` are the volume-weighted chain ladder's (``chain_ladder``),
/// with each origin's ``exposure`` and ``prior_ultimate``; ``mean`` and
/// ``std_dev`` are the posterior predictive reserve's.
#[pyclass(name = "BayesBfNormalisedFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyBayesBfNormalisedFit {
    pub(crate) inner: BayesBfNormalisedFit,
}

#[pymethods]
impl PyBayesBfNormalisedFit {
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

    /// The posterior predictive of a new underwriting year, the origin
    /// after the triangle's newest, for pricing a year not yet written.
    ///
    /// Each simulation takes a posterior draw and from it the new origin's
    /// expected ultimate, a gamma with cv ``cv`` and mean ``s * apriori *
    /// premium`` (``s`` the draw's common multiplier, 1 without
    /// ``common_cv``), then its ultimate with the fit's process error. The
    /// data reach the new year only through ``s``: without ``common_cv``
    /// the new year is the prior, ``Gamma(1 / cv**2, apriori)`` in loss
    /// ratio, whatever the triangle says; with a vague ``common_cv`` (say
    /// 10) the triangle sets its level.
    ///
    /// Parameters
    /// ----------
    /// premium : float
    ///     The new year's premium (its exposure), finite and positive.
    /// n_sims : int
    ///     Number of simulations.
    /// seed : int
    ///     Seed of the simulations.
    ///
    /// Returns
    /// -------
    /// NewYear
    ///     ``loss_ratio`` and ``ultimate`` by simulation, and ``patterns``,
    ///     the development pattern of each simulation's posterior draw, whose
    ///     ``monthly`` is the new year's emergence by month.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``premium`` is not positive, ``n_sims`` is 0, or the fit's
    ///     ``cv`` is infinite (a flat prior gives a new origin no proper
    ///     predictive).
    #[pyo3(signature = (premium, n_sims = 10_000, seed = 0))]
    fn new_year(
        &self,
        py: Python<'_>,
        premium: f64,
        n_sims: usize,
        seed: u64,
    ) -> PyResult<PyNewYear> {
        let fit = &self.inner;
        let inner = py
            .detach(|| fit.new_year(premium, n_sims, seed))
            .map_err(err)?;
        Ok(PyNewYear { inner })
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

    /// The plugged-in scale ``phi``, the chain ladder's Pearson scale.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    /// Exposure per origin: the exposure column's latest observed value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        self.inner.exposure.clone()
    }

    /// The prior mean of each origin's expected ultimate,
    /// ``apriori * exposure``.
    #[getter]
    fn prior_ultimate(&self) -> Vec<f64> {
        self.inner.prior_ultimate.clone()
    }

    /// The posterior mean of each age's share ``p[d]`` of the ultimate, in
    /// development order; the shares sum to 1.
    #[getter]
    fn pattern(&self) -> Vec<f64> {
        self.inner.pattern()
    }

    /// The posterior mean of each origin's expected ultimate
    /// ``exp(alpha[w])``, the quantity that ``apriori`` and ``cv`` put the
    /// prior on (the reserve's process error and the cells already paid are
    /// not in it).
    #[getter]
    fn expected_ultimate(&self) -> Vec<f64> {
        self.inner.expected_ultimate()
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

    /// Parameter names: ``alpha[1]`` to ``alpha[n]``, ``beta[2]`` to
    /// ``beta[m]`` (the first age is the corner, ``beta[1] = 0``).
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
    /// ``ess_tail``, as ``BayesOdpFit.summary``. Needs pandas.
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

    /// PSIS-LOO of the observed increments, as ``BayesOdpFit.loo``: the
    /// pointwise term is the normalised over-dispersed Poisson density at
    /// the cell's mean and ``scale``, so the result compares with
    /// ``BayesOdp``'s on the same column.
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
    /// ``ultimate`` and ``reserve``, the ``exposure``, ``prior_ultimate``
    /// and ``ulr`` (the chain ladder's ultimate over the exposure, a loss
    /// ratio only when the exposure is premium), the ``mean`` and
    /// ``std_dev`` of the posterior predictive reserve, ``ultimate_mean``
    /// (``latest`` plus ``mean``, the posterior mean ultimate) and
    /// ``ulr_mean`` (``ultimate_mean`` over the exposure). Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the chain ladder's totals, the total ``exposure`` and
    /// ``prior_ultimate``, the ``scale``, the total ``ulr``, the ``mean``
    /// and ``std_dev`` of the posterior predictive total reserve, and the
    /// total ``ultimate_mean`` and ``ulr_mean`` (over the total exposure).
    /// Needs pandas.
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
            "BayesBfNormalisedFit(origins={}, draws={}, n_sims={}, scale={:?}, warnings={})",
            self.inner.chain_ladder.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.scale,
            self.inner.diagnostics.warnings.len(),
        )
    }
}
