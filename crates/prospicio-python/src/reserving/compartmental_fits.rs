//! The compartmental model's fits, one-year views and backtest candidate.

use super::*;

/// A ``Compartmental`` for ``diagonal_backtest``, from
/// ``Compartmental.candidate``: the model's settings with the triangle and
/// the paid, outstanding (or incurred) and premium columns it reads.
///
/// Put it in the ``models`` of ``diagonal_backtest`` (a list, or a dict
/// that names it) to score the paid increments of the held-out diagonals
/// beside the other models, and call ``outstanding_backtest`` for the
/// second table, the held-out outstanding of the same diagonals. Both
/// tables use the same fits (one run of the sampler per split, kept by the
/// candidate), whichever is asked for first.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
/// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
/// ...         [983, 2211, 2830], [1657, 2685], [932]]
/// >>> incurred = [[1500, 1400, 900, 400, 100], [1300, 1200, 800, 300],
/// ...             [1400, 1100, 700], [1200, 900], [1100]]
/// >>> premium = [5812, 4908, 5454, 5165, 5214]
/// >>> rows = [(1988 + w, 12 * (d + 1), v, incurred[w][d] + v, premium[w])
/// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
/// >>> origin, age, values, inc, prem = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(
/// ...     origin, age, {"paid": values, "incurred": inc, "premium": prem})
/// >>> model = Compartmental(CompartmentalPriors.gesmann_morris())
/// >>> cand = model.candidate(tri, "paid", "premium", incurred="incurred")
/// >>> cand.name, cand.paid, cand.premium, cand.outstanding_from
/// ('compartmental_gm1', 'paid', 'premium', 'incurred')
#[pyclass(
    name = "CompartmentalCandidate",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyCompartmentalCandidate {
    pub(crate) inner: CompartmentalCandidate,
}

#[pymethods]
impl PyCompartmentalCandidate {
    /// The model's name in a backtest: ``"compartmental_gm1"`` for model 1,
    /// ``"compartmental_gm2"`` for model 2, ``"compartmental"`` otherwise.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// The cumulative paid column, the backtest's ``column``.
    #[getter]
    fn paid(&self) -> String {
        self.inner.paid().to_string()
    }

    /// The premium column.
    #[getter]
    fn premium(&self) -> String {
        self.inner.premium().to_string()
    }

    /// The column the outstanding comes from.
    #[getter]
    fn outstanding(&self) -> String {
        match self.inner.outstanding() {
            Outstanding::Column(c) | Outstanding::FromIncurred(c) => c.to_string(),
        }
    }

    /// ``"column"`` when ``outstanding`` is a column of outstanding
    /// amounts, ``"incurred"`` when it is a column of incurred amounts less
    /// paid.
    #[getter]
    fn outstanding_from(&self) -> &'static str {
        match self.inner.outstanding() {
            Outstanding::Column(_) => "column",
            Outstanding::FromIncurred(_) => "incurred",
        }
    }

    /// Scores the held-out **outstanding** of the latest ``n_diagonals``
    /// calendar diagonals, a second table beside ``diagonal_backtest``'s
    /// paid one: for each, the model is fitted to the cells valued before
    /// it (the outstanding of the held-out diagonal is always withheld) and
    /// forecasts the outstanding of the diagonal's cells, scored by the
    /// backtest's metrics. The splits, ``n_sims``, ``seed`` and
    /// ``interval`` are ``diagonal_backtest``'s, so a backtest with the
    /// same arguments and this candidate (or a copy of it in its
    /// ``models``) has fitted the model already and the fits are reused.
    /// The one model of the table is named after the candidate with
    /// ``_outstanding``. A held-out cell without an outstanding in the
    /// triangle, or without a fitted origin or development period, is
    /// counted in ``excluded``.
    ///
    /// Parameters
    /// ----------
    /// n_diagonals : int
    ///     The number of latest calendar diagonals to hold out.
    /// n_sims : int, default 1000
    ///     Simulations per forecast.
    /// seed : int, default 0
    ///     Seed of every fit and simulation.
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
    ///     If a diagonal cannot be held out, the model cannot be fitted or
    ///     a setting is invalid.
    #[pyo3(signature = (n_diagonals, *, n_sims = 1000, seed = 0, interval = 0.9))]
    fn outstanding_backtest(
        &self,
        py: Python<'_>,
        n_diagonals: usize,
        n_sims: usize,
        seed: u64,
        interval: f64,
    ) -> PyResult<PyBacktest> {
        let candidate = &self.inner;
        let (inner, diagonals) = py
            .detach(|| {
                let cells = TriangleFrame::new(candidate.triangle(), candidate.paid(), None)?;
                let bt =
                    candidate.outstanding_backtest(&cells, n_diagonals, n_sims, seed, interval)?;
                let diagonals = diagonal_labels(&cells, &bt);
                Ok((bt, diagonals))
            })
            .map_err(err)?;
        Ok(PyBacktest { inner, diagonals })
    }

    fn __repr__(&self) -> String {
        format!(
            "CompartmentalCandidate(name={}, paid={}, premium={})",
            quoted(self.inner.name()),
            quoted(self.inner.paid()),
            quoted(self.inner.premium())
        )
    }
}

/// Where a compartmental fit's outstanding comes from: exactly one of a
/// column of outstanding amounts and a column of incurred amounts, or
/// under ``fitted="paid_only"``, which does not read it, at most one.
pub(crate) fn compartmental_outstanding<'a>(
    fitted: Fitted,
    outstanding: Option<&'a str>,
    incurred: Option<&'a str>,
) -> PyResult<Outstanding<'a>> {
    match (outstanding, incurred) {
        (Some(c), None) => Ok(Outstanding::Column(c)),
        (None, Some(c)) => Ok(Outstanding::FromIncurred(c)),
        // Not read under paid alone.
        (None, None) if fitted == Fitted::PaidOnly => Ok(Outstanding::Column("")),
        _ => Err(PyValueError::new_err(
            "give exactly one of outstanding (a column of outstanding amounts) and \
             incurred (a column of incurred amounts); under fitted=\"paid_only\" neither \
             is needed",
        )),
    }
}

/// The one-year view of a ``Compartmental`` model, from
/// ``Compartmental.one_year``: the lifetime fit, the opening reserve, the
/// claims development result over the coming year and how reliable each
/// simulation's importance reweighting was, by the premium column's
/// origins.
///
/// ``cdr`` is the joint distribution of the claims development result by
/// origin (dimension ``"origin"``): the opening reserve less the year's
/// paid and the posterior mean reserve after the year; a negative value
/// is an adverse development. ``pareto_k`` holds one Pareto ``k`` per
/// simulation, ``flagged_share`` the share above ``k_threshold``;
/// ``moment_matched``, ``refitted`` and ``unrefitted`` list what happened
/// to the flagged simulations; ``approximate`` says why the view is
/// approximate when it reweighted them only by its model's default.
#[pyclass(
    name = "CompartmentalOneYearFit",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyCompartmentalOneYearFit {
    pub(crate) inner: BayesOneYearFit<CompartmentalFit>,
}

#[pymethods]
impl PyCompartmentalOneYearFit {
    /// The model fitted to the observed triangle, as ``Compartmental.fit``
    /// gives it.
    #[getter]
    fn fit(&self) -> PyCompartmentalFit {
        PyCompartmentalFit {
            inner: self.inner.fit.clone(),
        }
    }

    /// Origin period of each component, those of the premium column.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .fit
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Each origin's latest observed paid (0 with none).
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        self.inner.fit.latest_paid.clone()
    }

    /// The latest paid plus the opening reserve, per origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        self.inner.opening_ultimate.clone()
    }

    /// Each origin's posterior mean reserve on the observed triangle: the
    /// expected paid to the square's last age less the latest paid.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        self.inner.opening_reserve.clone()
    }

    /// Joint distribution of the claims development result by origin.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    /// Each simulation's Pareto ``k``, in order.
    #[getter]
    fn pareto_k(&self) -> Vec<f64> {
        self.inner.pareto_k.clone()
    }

    /// The simulations refitted with NUTS, in order (0-based).
    #[getter]
    fn refitted(&self) -> Vec<usize> {
        self.inner.refitted.clone()
    }

    /// The simulations moment matching rescued, in order (0-based).
    #[getter]
    fn moment_matched(&self) -> Vec<usize> {
        self.inner.moment_matched.clone()
    }

    /// The simulations due for a refit beyond ``max_refits``, reweighted
    /// instead, in order (0-based).
    #[getter]
    fn unrefitted(&self) -> Vec<usize> {
        self.inner.unrefitted.clone()
    }

    /// Why the view is approximate, or ``None``: set when ``refit`` was
    /// left to a model with development effects (model 2) whose default
    /// light refit was turned off (``light_refit=False``), so that it
    /// reweights the flagged simulations only.
    #[getter]
    fn approximate(&self) -> Option<String> {
        self.inner.approximate.clone()
    }

    /// How many of the refits had a diagnostic warning.
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

    /// One row per origin: ``origin``, ``premium``, ``latest_paid``,
    /// ``opening_ultimate``, ``opening_reserve``, and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the totals of ``to_frame``'s columns, the
    /// ``flagged_share``, and the ``cdr_mean`` and ``cdr_std_dev`` of the
    /// total claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    fn __repr__(&self) -> String {
        format!(
            "CompartmentalOneYearFit(origins={}, n_sims={}, flagged_share={:.3}, \
             moment_matched={}, refitted={}, unrefitted={})",
            self.inner.fit.origins.len(),
            self.inner.pareto_k.len(),
            self.inner.flagged_share(),
            self.inner.moment_matched.len(),
            self.inner.refitted.len(),
            self.inner.unrefitted.len(),
        )
    }
}

/// A fitted ``Compartmental``: the posterior, its MCMC diagnostics, and the
/// posterior predictive distributions of the reserves and of the future
/// paid and outstanding cells, all from the same simulations.
#[pyclass(name = "CompartmentalFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCompartmentalFit {
    pub(crate) inner: CompartmentalFit,
}

#[pymethods]
impl PyCompartmentalFit {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Paid by the simulated paid cells, each increment at its own age
    /// (process timing included), not spread; ``basis`` must be ``"paid"``,
    /// and a fit on incurred claims only is refused.
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

    /// Origin period of each component, those of the premium column.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// Each origin's premium.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner.premium.clone()
    }

    /// Each origin's latest observed paid (0 with none).
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        self.inner.latest_paid.clone()
    }

    /// The origins with no observed cell, whose effects each simulation
    /// draws from the population.
    #[getter]
    fn population_origins(&self) -> Vec<String> {
        self.inner
            .population_origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Joint posterior predictive distribution of each origin's reserve, the
    /// paid at the square's last age less the latest paid: dimension
    /// ``"origin"``, laid out as ``OdpBootstrapFit.reserves``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Joint distribution of the ultimate by origin: ``reserves`` with
    /// each origin's latest paid added in every simulation, the same
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
    /// The ultimate is the origin's latest paid plus its reserve in that
    /// simulation: the paid at the square's last age, on the paid basis.
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

    /// Posterior predictive cumulative paid of each future cell of the
    /// square: dimensions ``"origin"`` and ``"age"`` (months).
    #[getter]
    fn paid(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.paid.clone(),
        }
    }

    /// Posterior predictive outstanding (or incurred, the second response)
    /// of each future cell of the square, laid out as ``paid``; ``None``
    /// under ``fitted="paid_only"``.
    #[getter]
    fn outstanding(&self) -> Option<PyPredictiveDistribution> {
        self.inner
            .outstanding
            .clone()
            .map(|inner| PyPredictiveDistribution { inner })
    }

    /// Each simulation's paid development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine: at each age of the
    /// square (to the last age, where the reserves end), the expected
    /// cumulative paid of the ODE for the posterior draw the simulation
    /// took, for an origin with nothing observed (the population's rates,
    /// with each observed age's development effects and none at an age with
    /// no observed cell), over that at the last age. No tail: each pattern
    /// ends at 1 at the square's last age. The paid curve rises, and so does
    /// a running sum of positive incremental means, so with cumulative paid
    /// and no development effects (model 1) or incremental paid (model 2) a
    /// share never falls.
    ///
    /// Row ``i`` of every pattern is simulation ``i`` of ``reserves``,
    /// ``paid``, ``outstanding`` and ``ultimate_draws``.
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

    /// Each origin's posterior mean ultimate loss ratio ``RLR_i RRF_i``,
    /// without process error past the square.
    #[getter]
    fn ultimate_loss_ratio(&self) -> Vec<f64> {
        self.inner.ultimate_loss_ratio.clone()
    }

    /// The number of posterior draws whose population rates have
    /// ``k_er <= k_p`` (0 under ``rate_order="ordered"``).
    #[getter]
    fn rate_swaps(&self) -> usize {
        self.inner.rate_swaps
    }

    /// With effects on the rates (each cell its own ``k_er`` and ``k_p``),
    /// the share of posterior draws and observed cells with
    /// ``k_er <= k_p``: a diagnostic, not a warning, since each cell's
    /// means are invariant under the swap on their own. ``None`` when the
    /// rates are shared.
    #[getter]
    fn cell_rate_swaps(&self) -> Option<f64> {
        self.inner.cell_rate_swaps
    }

    /// MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
    /// ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
    /// failed threshold and one for rate swaps; empty when all is well).
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(py, &self.inner.diagnostics)
    }

    /// Parameter names: on the natural scale ``k_er``, ``k_p``, ``RLR``,
    /// ``RRF`` (the population values, at a zero effect), ``sigma_os``,
    /// ``sigma_pd``, ``RLR[o]`` and ``RRF[o]`` for each origin ``o`` with
    /// an observed cell (at a zero development effect), with origin effects
    /// on the rates ``k_er[o]`` and ``k_p[o]`` too, and with development
    /// effects ``dev_ker[a]``, ``dev_kp[a]``, ``dev_RLR[a]`` and
    /// ``dev_RRF[a]``, each development age ``a`` (months) with an observed
    /// cell's effect on the ``o`` scale; then on the scale the paper prints
    /// ``b_ker``, ``b_kp``, ``b_RLR``, ``b_RRF``, ``tau_RLR``,
    /// ``tau_RRF`` (and ``tau_ker``, ``tau_kp``), ``rho`` (and
    /// ``tau_dev_ker``, ``tau_dev_kp``, ``tau_dev_RLR``, ``tau_dev_RRF``),
    /// ``log_sigma_os`` and ``log_sigma_pd``. Each name in parentheses is
    /// there only when the priors have that effect.
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

    /// Posterior summary, one row per parameter, as ``MeyersFit.summary``.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fit = &self.inner;
        let s = py.detach(|| fit.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of the observed cells, outstanding cells first and then
    /// paid cells, with each cell's relative efficiency from the chains.
    /// The pointwise term is the full log density of each response, normal
    /// or log-normal (with its ``-log y`` term), as brms' ``log_lik``, on
    /// the response's scale: amounts or loss ratios, cumulative or
    /// incremental paid. Compare it only with fits of the same responses.
    /// ``loo_by_type`` splits the elpd.
    ///
    /// Returns
    /// -------
    /// Elpd
    fn loo(&self, py: Python<'_>) -> PyResult<PyElpd> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l.loo))
    }

    /// The elpd of ``loo`` split by type: a dict with ``n_outstanding``
    /// (the number of outstanding cells, which come first in the pointwise
    /// values), ``elpd_outstanding`` and ``elpd_paid``.
    ///
    /// Returns
    /// -------
    /// dict
    fn loo_by_type<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let fit = &self.inner;
        let l = py.detach(|| fit.loo()).map_err(err)?;
        let out = PyDict::new(py);
        out.set_item("n_outstanding", l.n_outstanding)?;
        out.set_item("elpd_outstanding", l.elpd_outstanding)?;
        out.set_item("elpd_paid", l.elpd_paid)?;
        Ok(out)
    }

    /// One row per origin: ``origin``, ``premium``, ``latest_paid``, the
    /// ``mean`` and ``std_dev`` of the posterior predictive reserve, and the
    /// posterior mean ``ultimate_loss_ratio``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row: the total ``premium`` and ``latest_paid``, and the ``mean``
    /// and ``std_dev`` of the posterior predictive total reserve. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per future paid cell: ``origin``, ``development`` (age in
    /// months), and the ``mean``, ``std_dev``, ``q05``, ``q50`` and ``q95``
    /// of the posterior predictive cumulative paid. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn paid_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.paid_table().map_err(err)?)
    }

    /// The same for the future outstanding cells. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Under ``fitted="paid_only"``, which has no outstanding cells.
    fn outstanding_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.outstanding_table().map_err(err)?)
    }

    fn __repr__(&self) -> String {
        format!(
            "CompartmentalFit(origins={}, draws={}, n_sims={}, rate_swaps={}, warnings={})",
            self.inner.origins.len(),
            self.inner.posterior.n_draws(),
            self.inner.reserves.n_sims(),
            self.inner.rate_swaps,
            self.inner.diagnostics.warnings.len(),
        )
    }
}

/// ``Compartmental.fit_segments``: each segment's own posterior and the
/// joint posterior predictive distributions of the reserves and of the
/// future paid and outstanding cells.
///
/// ``reserves``, ``paid`` and ``outstanding`` have the triangle's keys as
/// their first dimensions, so ``reserves.aggregate(["lob"])`` keeps the
/// dependence between segments. Per-origin lists (``origins``,
/// ``premium``, ``latest_paid``) run over the origins of each segment in
/// turn, like the rows of ``to_frame()`` and the components of
/// ``reserves``. The posterior of one segment (``diagnostics``,
/// ``parameter_names``, ``draws``, ``chains``, ``summary()``, ``loo()``)
/// needs a single-segment fit: for several segments use ``segment(...)`` or
/// ``diagnostics_frame()``.
#[pyclass(name = "CompartmentalFits", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCompartmentalFits {
    pub(crate) inner: CompartmentalFits,
}

impl PyCompartmentalFits {
    fn one(&self, field: &str, instead: &str) -> PyResult<&CompartmentalSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyCompartmentalFits {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Paid by each segment's simulated paid cells, not spread; ``basis``
    /// must be ``"paid"``.
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

    /// Origin period of each per-origin value and component, those of each
    /// segment's premium column.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.origins.iter().map(ToString::to_string))
            .collect()
    }

    /// The premium of each segment's origins, in the order of ``origins``.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.premium.iter().copied())
            .collect()
    }

    /// Each origin's latest observed paid, in the order of ``origins``.
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.latest_paid.iter().copied())
            .collect()
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
    /// each origin's latest paid added in every simulation, the same
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
    /// The ultimate is the origin's latest paid plus its reserve in that
    /// simulation: the paid at the square's last age, on the paid basis.
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

    /// Posterior predictive cumulative paid of each future cell of every
    /// segment: dimensions the keys, ``"origin"`` and ``"age"`` (months),
    /// from the same simulations as ``reserves``.
    #[getter]
    fn paid(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.paid.clone(),
        }
    }

    /// Posterior predictive outstanding (or incurred) of each future cell of
    /// every segment, laid out as ``paid``; ``None`` under
    /// ``fitted="paid_only"``.
    #[getter]
    fn outstanding(&self) -> Option<PyPredictiveDistribution> {
        self.inner
            .outstanding
            .clone()
            .map(|inner| PyPredictiveDistribution { inner })
    }

    /// Each simulation's paid development pattern, to pair with the
    /// simulation's loss ratio in the cash-flow engine: at each age of the
    /// square (to the last age, where the reserves end), the expected
    /// cumulative paid of the ODE for the posterior draw the simulation
    /// took, for an origin with nothing observed (the population's rates,
    /// with each observed age's development effects and none at an age with
    /// no observed cell), over that at the last age. No tail: each pattern
    /// ends at 1 at the square's last age. The paid curve rises, and so does
    /// a running sum of positive incremental means, so with cumulative paid
    /// and no development effects (model 1) or incremental paid (model 2) a
    /// share never falls.
    ///
    /// Row ``i`` of every pattern is simulation ``i`` of ``reserves``,
    /// ``paid``, ``outstanding`` and ``ultimate_draws``.
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

    /// The number of posterior draws with ``k_er <= k_p`` in each segment,
    /// in index order.
    #[getter]
    fn rate_swaps(&self) -> Vec<usize> {
        self.inner
            .segments
            .fits
            .iter()
            .map(|s| s.fit.rate_swaps)
            .collect()
    }

    /// MCMC diagnostics of a single-segment fit, as
    /// ``CompartmentalFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(
            py,
            &self
                .one("diagnostics", "diagnostics_frame()")?
                .fit
                .diagnostics,
        )
    }

    /// Parameter names of a single-segment fit, as
    /// ``CompartmentalFit.parameter_names``.
    #[getter]
    fn parameter_names(&self) -> PyResult<Vec<String>> {
        Ok(self
            .one("parameter_names", "")?
            .fit
            .posterior
            .names()
            .to_vec())
    }

    /// Posterior draws of a single-segment fit, as
    /// ``CompartmentalFit.draws``.
    #[getter]
    fn draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let posterior = &self.one("draws", "")?.fit.posterior;
        let p = posterior.dim().max(1);
        Ok(posterior.values().chunks(p).map(<[f64]>::to_vec).collect())
    }

    /// Number of chains of a single-segment fit.
    #[getter]
    fn chains(&self) -> PyResult<usize> {
        Ok(self.one("chains", "")?.fit.posterior.chains())
    }

    /// Posterior summary of a single-segment fit, as
    /// ``CompartmentalFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let one = &self.one("summary()", "")?.fit;
        let s = py.detach(|| one.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of a single-segment fit's observed cells, as
    /// ``CompartmentalFit.loo``.
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
        let one = &self.one("loo()", "")?.fit;
        let l = py.detach(|| one.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l.loo))
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
        segment_diagnostics_frame(py, &self.inner.segments, |s| &s.fit.diagnostics)
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``premium``, ``latest_paid``, the ``mean`` and ``std_dev`` of the
    /// posterior predictive reserve, and the posterior mean
    /// ``ultimate_loss_ratio``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the total ``premium`` and
    /// ``latest_paid``, and the ``mean`` and ``std_dev`` of the segment's
    /// posterior predictive total reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per segment and future paid cell, as
    /// ``CompartmentalFit.paid_frame``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn paid_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.paid_table().map_err(err)?)
    }

    /// The same for the future outstanding cells. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Under ``fitted="paid_only"``, which has no outstanding cells.
    fn outstanding_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.outstanding_table().map_err(err)?)
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint
    /// distributions (same dimensions).
    ///
    /// Returns
    /// -------
    /// CompartmentalFits
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
            .map(|s| s.fit.diagnostics.warnings.len())
            .sum();
        format!(
            "CompartmentalFits({}origins={}, n_sims={}, warnings={warnings})",
            segments_prefix(segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}

/// The one-year view of a ``Compartmental`` in every segment of a
/// triangle, from ``Compartmental.one_year_segments``: each segment's
/// posterior, opening reserve and Pareto ``k``, and the joint claims
/// development result across segments.
///
/// ``cdr`` is one joint distribution of the claims development result with
/// the triangle's keys and ``"origin"`` as dimensions, so
/// ``cdr.aggregate(["lob"])`` keeps the dependence between segments.
/// Per-origin lists (``origins``, ``premium``, ``latest_paid``,
/// ``opening_ultimate``, ``opening_reserve``) run over the origins of each
/// segment in turn, like the rows of ``to_frame()`` and the components of
/// ``cdr``. ``diagnostics``, ``parameter_names``, ``draws``, ``chains``,
/// ``summary()``, ``loo()``, ``pareto_k``, ``moment_matched``, ``refitted``,
/// ``unrefitted``, ``refit_warnings`` and ``flagged_share`` are one
/// segment's and need a single-segment fit: for several segments use
/// ``segment(...)``, ``totals_frame()`` (which has the flagged shares) or
/// ``diagnostics_frame()``. ``approximate`` says why the views are
/// approximate when they reweighted the flagged simulations only by their
/// model's default (``light_refit=False``).
#[pyclass(
    name = "CompartmentalOneYearFits",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyCompartmentalOneYearFits {
    pub(crate) inner: CompartmentalOneYearFits,
}

impl PyCompartmentalOneYearFits {
    fn one(
        &self,
        field: &str,
        instead: &str,
    ) -> PyResult<&BayesOneYearSegment<CompartmentalSegment>> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyCompartmentalOneYearFits {
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

    /// Origin period of each per-origin value and component, those of each
    /// segment's premium column.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.fit.fit.origins.iter().map(ToString::to_string))
            .collect()
    }

    /// The premium of each segment's origins, in the order of ``origins``.
    #[getter]
    fn premium(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.fit.fit.premium.clone())
    }

    /// Each origin's latest observed paid, in the order of ``origins``.
    #[getter]
    fn latest_paid(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.fit.fit.latest_paid.clone())
    }

    /// The latest paid plus the opening reserve, per origin.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.opening_ultimate.clone())
    }

    /// Each origin's posterior mean reserve on the observed triangle.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        by_origin(&self.inner.segments, |s| s.opening_reserve.clone())
    }

    /// Joint distribution of the claims development result by segment and
    /// origin: the triangle's keys and ``"origin"`` are its dimensions, one
    /// component per segment and origin, one row per simulation.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    /// Why the views are approximate, or ``None``: set when ``refit`` was
    /// left to a model with development effects (model 2) whose default
    /// light refit was turned off (``light_refit=False``), so that it
    /// reweights the flagged simulations only.
    #[getter]
    fn approximate(&self) -> Option<String> {
        self.inner.approximate.clone()
    }

    /// MCMC diagnostics of a single-segment fit's posterior, as
    /// ``CompartmentalFit.diagnostics``.
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        diagnostics_dict(
            py,
            &self
                .one("diagnostics", "diagnostics_frame()")?
                .fit
                .fit
                .diagnostics,
        )
    }

    /// Parameter names of a single-segment fit.
    #[getter]
    fn parameter_names(&self) -> PyResult<Vec<String>> {
        Ok(self
            .one("parameter_names", "")?
            .fit
            .fit
            .posterior
            .names()
            .to_vec())
    }

    /// Posterior draws of a single-segment fit, one list per draw.
    #[getter]
    fn draws(&self) -> PyResult<Vec<Vec<f64>>> {
        let posterior = &self.one("draws", "")?.fit.fit.posterior;
        let p = posterior.dim().max(1);
        Ok(posterior.values().chunks(p).map(<[f64]>::to_vec).collect())
    }

    /// Number of chains of a single-segment fit.
    #[getter]
    fn chains(&self) -> PyResult<usize> {
        Ok(self.one("chains", "")?.fit.fit.posterior.chains())
    }

    /// Each simulation's Pareto ``k`` in a single-segment fit.
    #[getter]
    fn pareto_k(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("pareto_k", "")?.pareto_k.clone())
    }

    /// The simulations of a single-segment fit refitted with NUTS
    /// (0-based).
    #[getter]
    fn refitted(&self) -> PyResult<Vec<usize>> {
        Ok(self.one("refitted", "")?.refitted.clone())
    }

    /// The simulations of a single-segment fit that moment matching
    /// rescued (0-based).
    #[getter]
    fn moment_matched(&self) -> PyResult<Vec<usize>> {
        Ok(self.one("moment_matched", "")?.moment_matched.clone())
    }

    /// The simulations of a single-segment fit due for a refit beyond
    /// ``max_refits``, reweighted instead (0-based).
    #[getter]
    fn unrefitted(&self) -> PyResult<Vec<usize>> {
        Ok(self.one("unrefitted", "")?.unrefitted.clone())
    }

    /// How many of a single-segment fit's refits had a diagnostic warning.
    #[getter]
    fn refit_warnings(&self) -> PyResult<usize> {
        Ok(self.one("refit_warnings", "")?.refit_warnings)
    }

    /// The Pareto ``k`` above which a simulation is flagged.
    #[getter]
    fn k_threshold(&self) -> f64 {
        self.inner.segments.fits[0].k_threshold
    }

    /// The share of a single-segment fit's simulations whose ``k`` is
    /// above ``k_threshold`` (or NaN).
    #[getter]
    fn flagged_share(&self) -> PyResult<f64> {
        Ok(self.one("flagged_share", "totals_frame()")?.flagged_share())
    }

    /// Posterior summary of a single-segment fit, as
    /// ``CompartmentalFit.summary``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let one = &self.one("summary()", "")?.fit.fit;
        let s = py.detach(|| one.summary()).map_err(err)?;
        summary_frame(py, &s)
    }

    /// PSIS-LOO of a single-segment fit's observed cells, as
    /// ``CompartmentalFit.loo``.
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
        let one = &self.one("loo()", "")?.fit.fit;
        let l = py.detach(|| one.loo()).map_err(err)?;
        Ok(PyElpd::from_loo(l.loo))
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
        segment_diagnostics_frame(py, &self.inner.segments, |s| &s.fit.fit.diagnostics)
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``premium``, ``latest_paid``, ``opening_ultimate``,
    /// ``opening_reserve``, and the ``cdr_mean`` and ``cdr_std_dev`` of the
    /// claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the totals of ``to_frame()``'s
    /// columns, the ``flagged_share``, and the ``cdr_mean`` and
    /// ``cdr_std_dev`` of the segment's total claims development result.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The one-year view of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint claims
    /// development result (same dimensions).
    ///
    /// Returns
    /// -------
    /// CompartmentalOneYearFits
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
        let detail = match segments.fits.as_slice() {
            [one] => format!(
                ", flagged_share={:?}, refitted={}",
                one.flagged_share(),
                one.refitted.len()
            ),
            _ => String::new(),
        };
        format!(
            "CompartmentalOneYearFits({}origins={}, n_sims={}{detail})",
            segments_prefix(segments),
            self.origins().len(),
            self.inner.cdr.n_sims(),
        )
    }
}
