//! Gesmann and Morris' compartmental model, `Compartmental`.

use super::*;

const LIKELIHOODS: [(&str, CompartmentalLikelihood); 2] = [
    ("normal", CompartmentalLikelihood::Normal),
    ("lognormal", CompartmentalLikelihood::LogNormal),
];
const SCALES: [(&str, CompartmentalScale); 2] = [
    ("amount", CompartmentalScale::Amount),
    ("loss_ratio", CompartmentalScale::LossRatio),
];
const PAID_RESPONSES: [(&str, PaidResponse); 2] = [
    ("cumulative", PaidResponse::Cumulative),
    ("incremental", PaidResponse::Incremental),
];
const FITTED: [(&str, Fitted); 4] = [
    ("paid_and_outstanding", Fitted::PaidAndOutstanding),
    ("paid_and_incurred", Fitted::PaidAndIncurred),
    ("paid_only", Fitted::PaidOnly),
    ("incurred_only", Fitted::IncurredOnly),
];
const RATE_ORDERS: [(&str, RateOrder); 2] =
    [("free", RateOrder::Free), ("ordered", RateOrder::Ordered)];
const DEVELOPMENT_LEVELS: [(&str, DevelopmentLevels); 2] = [
    ("per_age", DevelopmentLevels::PerAge),
    ("by_year", DevelopmentLevels::ByYear),
];
const STARTS: [(&str, CompartmentalStart); 2] = [
    ("data", CompartmentalStart::Data),
    ("prior_centre", CompartmentalStart::PriorCentre),
];

/// Gesmann and Morris' (2020) hierarchical compartmental reserving model
/// with one outstanding compartment, their models 1 and 2
/// (``docs/design/reserving-compartmental.md``). Claims flow from exposure
/// into outstanding as they are reported, at rate ``k_er``, and from
/// outstanding into paid as they settle, at rate ``k_p``. For origin ``i``
/// with premium ``P_i`` and development time ``t`` in years, with
/// ``g(t) = (exp(-k_p t) - exp(-k_er t)) / (k_er - k_p)``::
///
///     OS_i(t) = P_i RLR_i k_er g(t)
///     PD_i(t) = P_i RLR_i RRF_i [(1 - exp(-k_p t)) - k_p g(t)]
///
/// so ``RLR_i`` is the reported loss ratio, ``RRF_i`` the reserve
/// robustness factor (paid over reported) and ``RLR_i RRF_i`` the ultimate
/// loss ratio. ``RLR`` and ``RRF`` vary by origin with correlated effects
/// (LKJ prior on the correlation) on the priors' ``o`` scale; the priors
/// can add origin effects on the rates and development-year effects on any
/// of the four (``CompartmentalPriors``), each cell then taking its own
/// parameters. NUTS samples the posterior on a pilot run's dense scale,
/// with the gradient from dual numbers.
///
/// Model 1 (the defaults) has normal outstanding and cumulative paid
/// amounts about these curves, each type with its own sigma. Model 2
/// (``Compartmental.model_2``, with
/// ``CompartmentalPriors.gesmann_morris_2()``) has log-normal outstanding
/// and incremental paid loss ratios to premium; every combination of
/// ``likelihood``, ``scale`` and ``paid_response`` is built. Under
/// ``likelihood="lognormal"`` every response must be positive, and under
/// ``paid_response="incremental"`` each paid cell needs the paid at the
/// age before; ``fit`` raises ``ValueError`` naming the cell otherwise.
/// ``fitted="paid_and_incurred"`` fits paid and incurred, with incurred in place of
/// outstanding as the second response: the mean of an incurred cell is ``RLR_i (OS(t) +
/// RRF_i PD(t))`` times premium (or as a loss ratio), with the cumulative ``PD`` even
/// under ``paid_response="incremental"``, and ``log_sigma_os`` is the incurred sigma.
/// Give the incurred column as ``incurred=``, or an outstanding column as
/// ``outstanding=`` and the incurred is outstanding plus paid where both are observed.
/// The fit's ``outstanding`` then holds the future incurred cells.
///
/// ``fitted="paid_only"`` and ``fitted="incurred_only"`` fit one response alone and need
/// ``rate_order="ordered"`` (``ValueError`` under ``"free"``): from one response the means
/// do not change under the rate swap below, and from paid alone only ``RLR x RRF`` is
/// identified, so the priors carry the split of ``RLR`` and ``RRF``. The model then has one
/// sigma, ``sigma_pd`` or ``sigma_os``. Under ``"paid_only"`` the second column is not read
/// (``outstanding=`` and ``incurred=`` may both be left out) and the fit has no
/// ``outstanding`` (``None``). Under ``"incurred_only"`` the incurred is given as for
/// ``"paid_and_incurred"`` and the paid column supplies each origin's latest paid only: the
/// reserve is the expected paid at the last age less the latest paid, and the future paid
/// cells are expected values, without noise, since no paid cell is fitted.
///
/// ``CompartmentalPriors.diffuse()`` is the starting point for ``"paid_only"``. Under
/// ``"incurred_only"`` no paid cell is fitted, so the incurred does not identify the payment
/// rate ``k_p`` and its prior sets the paid run-off and the reserve at every width (on company
/// 337 the paper's narrow prior put the reserve 26% above the fit to paid and outstanding):
/// the fit refuses a ``k_p`` prior with a log-scale sd above 0.5, so the diffuse preset is an
/// error there, and you give a ``k_p`` prior you believe, for example from the book's paid
/// history or from a paid-and-incurred fit of a similar book.
///
/// An origin with a premium and no observed cell (the papers' training
/// triangle has one) draws its effects from the population in each
/// simulation, and so does a development age with no observed cell. The
/// reserve is the paid at the square's last age less the latest paid; the
/// square runs for as many origin periods as the premium column has
/// origins, in development periods (40 quarterly ages for ten annual
/// origins). The development grain may be finer than the origin grain
/// (the triangle's own grains), and each cell's development time is its
/// age in years on every grain, so the rates are per year and an
/// incremental paid cell is the increment over one development period.
/// The coming year of ``one_year`` is the next twelve months of calendar
/// time, up to four quarterly cells per origin. One ``seed`` replays a fit
/// exactly: the chains and the simulations each take a seed derived from
/// it.
///
/// The means are exactly invariant under swapping the two rates (with
/// ``RLR`` and ``RRF`` rescaled), so only the priors tell the two modes
/// apart. Under ``rate_order="free"`` (the paper's priors) the fit warns
/// when any posterior draw has population rates with ``k_er <= k_p``;
/// ``rate_order="ordered"`` samples ``k_er = k_p + exp(delta)``, the same
/// priors restricted to ``k_er > k_p``. With effects on the rates,
/// ``CompartmentalFit.cell_rate_swaps`` reports the share of draws and
/// cells with ``k_er <= k_p``.
///
/// Fitting sub-annual data
/// -----------------------
/// Quarterly or monthly development cells are allowed. What follows is from
/// a study on simulated books (design note decision 14,
/// ``knowledge/findings/compartmental-subannual.md``); no real sub-annual
/// triangle was available.
///
/// * Rate priors: the rates are per year on every grain, and the paper's
///   presets are kept as they are. Finer cells pull the preset towards the
///   truth; a line whose first year runs much faster or slower than the
///   preset's ``k_er`` of about 3 needs priors of its own.
/// * Development levels (model 2): ``development_levels="per_age"`` or
///   ``"by_year"``, a choice to make with ``loo``. Where effects move by
///   year, by-year levels agree with per-age ones on the reserve (0.10 sd
///   apart on average in the study), win elpd in 30 of 30 quarterly books and
///   on 10 monthly books under a short sampler took 89 s against 126 s (rough
///   times) with a worst R-hat of 1.07 against 1.62, though a median bulk ESS
///   of 101 against 114; where effects move within a year they lose 361 of
///   elpd in 30 of 30 books and widen the reserve's sd by 59%.
/// * Seasonality: there is no seasonal term. Model 1 reads a season as noise
///   (the interval widens, by up to 54% in the sd for a 30% season, and the
///   mean can sit nearly 1 sd off the outcome), and model 2's per-age levels
///   do not absorb a season unless every origin starts in the same calendar
///   quarter. A calendar-quarter factor is not built.
/// * Cumulative paid: the interval narrows as the cells get finer, right where
///   the noise of the cells is independent and too narrow where it persists.
///   In the study's world with persistent payment noise, at one persistence
///   chosen by hand and fitted to no book (``PERSIST_PHI`` 25, ``PERSIST_RHO``
///   0.95; a smaller fall is expected with less, untested), the 90% interval
///   covered 88%, 70% and 56% at annual, quarterly and monthly cells and the
///   RMSE rose from 1,623 to 2,033. A fit with cumulative paid on cells finer
///   than a year therefore warns, asking you to check the residuals; it also
///   warns when
///   the fit is calibrated, since the data do not say which world they are
///   in. ``paid_response="incremental"`` keeps the interval but, in that
///   world, ran 1.3 to 2 times too wide, with a worse RMSE at quarterly and
///   monthly grain and a failing Meyers' D at annual grain. Neither is chosen
///   for you.
/// * Earned exposure: all of an origin's exposure starts at age 0, so
///   sub-annual development of annual origins fits an exposure the model does
///   not have; use quarterly origins where the data allow. On simulated books
///   with earned premium annual origins failed Meyers' D at annual, quarterly
///   and monthly development, and quarterly origins of the same books passed
///   narrowly (17.8 against a critical 19.2, on a short sampler, with 3 books
///   at the default sampler); that comparison also changes the number of
///   origins and the effects' variance. A setting that spreads the exposure
///   over the origin's period is not built.
///
/// Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
/// per chain, any divergent transition) and rate swaps do not fail the
/// fit: ``fit`` raises one ``UserWarning`` listing them, and they stay in
/// ``CompartmentalFit.diagnostics``. With the defaults, 194 of 200 seeds on
/// company 337 passed every threshold for model 1, and with
/// ``model_2``'s sampler 195 of 200 for model 2; each of the others just
/// missed one.
///
/// Parameters
/// ----------
/// priors : CompartmentalPriors
///     The priors; required.
/// likelihood : str
///     ``"normal"`` (model 1) or ``"lognormal"`` (model 2).
/// scale : str
///     The response scale: ``"amount"`` (model 1) or ``"loss_ratio"``,
///     amounts over the origin's premium (model 2).
/// paid_response : str
///     ``"cumulative"`` (model 1) or ``"incremental"`` (model 2).
/// fitted : str
///     ``"paid_and_outstanding"``, ``"paid_and_incurred"`` (incurred replaces
///     outstanding as the second response, as above), ``"paid_only"`` or
///     ``"incurred_only"`` (one response alone, with ``rate_order="ordered"``).
/// rate_order : str
///     ``"free"`` or ``"ordered"``, as above.
/// development_levels : str
///     The width of a development level, for priors with development
///     effects (model 2): ``"per_age"`` (default), one level for each age, or
///     ``"by_year"``, one for each development year (counted from the
///     origin's start: ages 1 to 12 months, 13 to 24 and so on). Choose with
///     ``loo``; see "Fitting sub-annual data" above. The same fit at annual
///     grain. ``ValueError`` for ``"by_year"`` with priors that have no
///     development effects (model 1).
/// start : str
///     Where the pilot run starts: ``"data"``, a least-squares fit of the
///     population curve (the prior centre if that fit fails, and under one
///     response if its log posterior is below the prior centre's), or
///     ``"prior_centre"``.
/// chains, tune, draws : int
///     NUTS chains, warm-up draws per chain and kept draws per chain.
/// n_sims : int
///     Number of predictive simulations; positive.
/// seed : int
///     Seed of the chains and of the simulations.
/// target_accept : float
///     Target mean acceptance rate (the paper's ``adapt_delta``, 0.99).
/// max_depth : int
///     Largest NUTS tree depth.
///
/// Examples
/// --------
/// >>> import warnings
/// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
/// >>> paid = [[9.6, 22.8, 33.3, 40.3, 45.1], [7.9, 19.5, 29.6, 36.8],
/// ...         [8.7, 24.3, 35.4], [13.3, 33.0], [11.4]]
/// >>> incurred = [[62.7, 64.0, 65.6, 65.3, 62.3], [62.1, 56.7, 56.6, 56.8],
/// ...             [64.0, 61.5, 63.2], [73.9, 75.1], [77.1]]
/// >>> premium = [104.4, 88.9, 86.0, 99.3, 104.9]
/// >>> rows = [(1988 + w, 12 * (d + 1), paid[w][d], incurred[w][d], premium[w])
/// ...         for w in range(5) for d in range(len(paid[w]))]
/// >>> origin, age, pd, inc, prem = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(
/// ...     origin, age, {"paid": pd, "incurred": inc, "premium": prem})
/// >>> model = Compartmental(CompartmentalPriors.gesmann_morris(), chains=2,
/// ...                       tune=150, draws=150, target_accept=0.9, n_sims=1000)
/// >>> # Check the priors first: the mean paid of 1988 at age 60 over its premium.
/// >>> paths = model.prior_predictive(tri, "premium", n_sims=1000)
/// >>> ratio = paths["paid"].marginal(("1988", 60))
/// >>> 0.2 < ratio.quantile(0.05) / 104.4 and ratio.quantile(0.95) / 104.4 < 1.2
/// True
/// >>> with warnings.catch_warnings():
/// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
/// ...     fit = model.fit(tri, "paid", "premium", incurred="incurred")
/// >>> fit.reserves.dims, fit.parameter_names[0]
/// (['origin'], 'k_er')
#[pyclass(name = "Compartmental", module = "prospicio.reserving", frozen)]
pub(crate) struct PyCompartmental {
    inner: Compartmental,
}

#[pymethods]
impl PyCompartmental {
    #[new]
    #[pyo3(signature = (
        priors,
        *,
        likelihood = "normal",
        scale = "amount",
        paid_response = "cumulative",
        fitted = "paid_and_outstanding",
        rate_order = "free",
        development_levels = "per_age",
        start = "data",
        chains = 4,
        tune = 1000,
        draws = 2000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.99,
        max_depth = 10,
        dependence = "independent",
        spearman = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        priors: PyRef<'_, PyCompartmentalPriors>,
        likelihood: &str,
        scale: &str,
        paid_response: &str,
        fitted: &str,
        rate_order: &str,
        development_levels: &str,
        start: &str,
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
        let inner = Compartmental {
            priors: priors.inner,
            likelihood: named("likelihood", likelihood, &LIKELIHOODS)?,
            scale: named("scale", scale, &SCALES)?,
            paid_response: named("paid_response", paid_response, &PAID_RESPONSES)?,
            fitted: named("fitted", fitted, &FITTED)?,
            rate_order: named("rate_order", rate_order, &RATE_ORDERS)?,
            development_levels: named(
                "development_levels",
                development_levels,
                &DEVELOPMENT_LEVELS,
            )?,
            start: named("start", start, &STARTS)?,
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
        };
        // Refuse a combination of settings the model does not support when
        // it is built, not when it is fitted (#580).
        inner.check_supported().map_err(err)?;
        Ok(Self { inner })
    }

    /// Gesmann and Morris' model 2 with ``priors`` (the paper's are
    /// ``CompartmentalPriors.gesmann_morris_2()``): the log-normal
    /// likelihood on loss ratios with incremental paid, and a longer, finer
    /// sampler than the constructor's, four chains of 2,000 warm-up and
    /// 8,000 kept draws with target acceptance 0.999. With 87 parameters,
    /// effect sds whose posteriors pile up near 0 and a slowly mixing
    /// ``k_p``, model 1's sampler left divergent transitions or an R-hat
    /// above 1.01 on most seeds; this one passes every threshold on 195 of
    /// 200 seeds on company 337. A fit takes about a minute. The other
    /// arguments are the constructor's.
    ///
    /// Returns
    /// -------
    /// Compartmental
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Compartmental, CompartmentalPriors
    /// >>> m = Compartmental.model_2(CompartmentalPriors.gesmann_morris_2())
    /// >>> m.likelihood, m.scale, m.paid_response, m.tune, m.draws, m.target_accept
    /// ('lognormal', 'loss_ratio', 'incremental', 2000, 8000, 0.999)
    #[staticmethod]
    #[pyo3(signature = (
        priors,
        *,
        rate_order = "free",
        development_levels = "per_age",
        start = "data",
        chains = 4,
        tune = 2000,
        draws = 8000,
        n_sims = 10_000,
        seed = 0,
        target_accept = 0.999,
        max_depth = 10,
        dependence = "independent",
        spearman = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn model_2(
        priors: PyRef<'_, PyCompartmentalPriors>,
        rate_order: &str,
        development_levels: &str,
        start: &str,
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
        Self::new(
            priors,
            "lognormal",
            "loss_ratio",
            "incremental",
            "paid_and_outstanding",
            rate_order,
            development_levels,
            start,
            chains,
            tune,
            draws,
            n_sims,
            seed,
            target_accept,
            max_depth,
            dependence,
            spearman,
        )
    }

    /// The priors.
    #[getter]
    fn priors(&self) -> PyCompartmentalPriors {
        PyCompartmentalPriors {
            inner: self.inner.priors,
        }
    }

    /// The likelihood: ``"normal"`` or ``"lognormal"``.
    #[getter]
    fn likelihood(&self) -> &'static str {
        name_of(self.inner.likelihood, &LIKELIHOODS)
    }

    /// The response scale: ``"amount"`` or ``"loss_ratio"``.
    #[getter]
    fn scale(&self) -> &'static str {
        name_of(self.inner.scale, &SCALES)
    }

    /// The paid response: ``"cumulative"`` or ``"incremental"``.
    #[getter]
    fn paid_response(&self) -> &'static str {
        name_of(self.inner.paid_response, &PAID_RESPONSES)
    }

    /// The responses fitted.
    #[getter]
    fn fitted(&self) -> &'static str {
        name_of(self.inner.fitted, &FITTED)
    }

    /// How the rates are kept apart: ``"free"`` or ``"ordered"``.
    #[getter]
    fn rate_order(&self) -> &'static str {
        name_of(self.inner.rate_order, &RATE_ORDERS)
    }

    /// The width of a development level: ``"per_age"`` or ``"by_year"``.
    #[getter]
    fn development_levels(&self) -> &'static str {
        name_of(self.inner.development_levels, &DEVELOPMENT_LEVELS)
    }

    /// Where the pilot run starts: ``"data"`` or ``"prior_centre"``.
    #[getter]
    fn start(&self) -> &'static str {
        name_of(self.inner.start, &STARTS)
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

    /// How ``fit_segments`` joins the segments: ``"independent"`` or
    /// ``"rank_correlation"`` (``"synchronized"`` is refused by
    /// ``fit_segments``). ``fit`` ignores it.
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

    /// Samples the posterior of paid and outstanding of a single-segment
    /// cumulative triangle, with each origin's premium from another column
    /// (its latest value), and simulates the future cells of the square and
    /// the reserves. Outstanding comes from ``outstanding``, a column of
    /// outstanding amounts, or from ``incurred``, a column of incurred
    /// amounts less paid cell by cell (the papers' outstanding on company
    /// 337 is incurred with bulk and IBNR less paid); give exactly one, or
    /// neither under ``fitted="paid_only"``, which does not read it.
    /// Raises one ``UserWarning`` listing the failed diagnostics and any
    /// rate swap.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// paid : str
    ///     Cumulative paid amounts.
    /// premium : str
    ///     The premium of each origin; the fit's origins are this column's.
    /// outstanding : str, optional
    ///     A column of outstanding amounts.
    /// incurred : str, optional
    ///     A column of incurred amounts.
    ///
    /// Returns
    /// -------
    /// CompartmentalFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``fitted`` is ``"paid_only"`` or ``"incurred_only"`` and
    ///     ``rate_order`` is not ``"ordered"``; if not exactly one of
    ///     ``outstanding`` and ``incurred`` is given (at most one under
    ///     ``"paid_only"``); if a column is
    ///     missing, the triangle has several segments, or a premium is
    ///     missing or not positive; if a prior or sampler setting is
    ///     invalid; or if there are no more observed cells than parameters.
    #[pyo3(signature = (triangle, paid, premium, *, outstanding = None, incurred = None))]
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
    ) -> PyResult<PyCompartmentalFit> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.fit(tri, paid, os, premium))
            .map_err(err)?;
        warn_compartmental(py, &inner.diagnostics.warnings)?;
        Ok(PyCompartmentalFit { inner })
    }

    /// Samples the posterior of paid and outstanding in every segment of a
    /// cumulative triangle, each segment with its own posterior (the same
    /// model and settings, sampled independently) and its premium from its
    /// own ``premium`` column as ``fit`` reads it, into joint distributions
    /// of the reserves and of the future paid and outstanding cells, the
    /// segments depending on each other as ``dependence`` says. Under
    /// ``"rank_correlation"`` each segment's reserves, future paid cells and
    /// future outstanding cells are reordered together, ranked on the
    /// reserves, so every output of a segment stays consistent. Segment
    /// ``k`` (in index order) takes seeds derived from ``seed`` and ``k``
    /// for its chains and its simulations, so independent segments share no
    /// random numbers, and a segment's draws differ from ``fit`` of that
    /// segment alone. Raises one ``UserWarning`` listing the failed
    /// diagnostics and any rate swap of every segment, each message
    /// prefixed with its segment's label.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// paid : str
    ///     Cumulative paid amounts.
    /// premium : str
    ///     The premium of each origin.
    /// outstanding : str, optional
    ///     A column of outstanding amounts.
    /// incurred : str, optional
    ///     A column of incurred amounts.
    ///
    /// Returns
    /// -------
    /// CompartmentalFits
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
    /// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
    /// ...         [983, 2211, 2830], [1657, 2685], [932]]
    /// >>> case = [[1500, 1400, 900, 400, 100], [1300, 1200, 800, 300],
    /// ...         [1400, 1100, 700], [1200, 900], [1100]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214]
    /// >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * case[w][d], k * premium[w])
    /// ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> lob, origin, age, values, os, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(
    /// ...     origin, age,
    /// ...     {"paid": values, "case": os, "premium": prem}, keys={"lob": lob})
    /// >>> model = Compartmental(
    /// ...     CompartmentalPriors.gesmann_morris(), tune=100, draws=100, n_sims=200,
    /// ...     chains=2)
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     fits = model.fit_segments(tri, "paid", "premium", outstanding="case")
    /// >>> fits.index, fits.reserves.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    #[pyo3(signature = (triangle, paid, premium, *, outstanding = None, incurred = None))]
    fn fit_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
    ) -> PyResult<PyCompartmentalFits> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.fit_segments(tri, paid, os, premium))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.fit.diagnostics);
        warn_compartmental(py, &warnings)?;
        Ok(PyCompartmentalFits { inner })
    }

    /// The one-year view: the claims development result over the coming
    /// year, by Pareto smoothed importance sampling of the posterior
    /// (decision 20 of ``docs/design/reserving-bayes.md``). The posterior
    /// and the lifetime reserves are ``fit``'s for the same settings. Each
    /// simulation picks a posterior draw, draws the paid and outstanding of
    /// the cells valued in the next twelve months from it (every origin's
    /// next diagonal), reweights every posterior draw by the likelihood of
    /// those cells, and smooths the weights (one Pareto ``k`` per
    /// simulation). Origin ``w``'s claims development result is the
    /// expected paid to the square's last age given the observed cells,
    /// less the year's paid and the same expectation given the year too;
    /// its mean is about zero. An origin with no observed cell (1997 on
    /// the papers' triangle) and, with development effects (model 2), an
    /// age with none take their effects as part of each posterior draw, so
    /// the year's cells update them. A simulation whose ``k`` is above
    /// ``k_threshold`` is flagged and handled as ``refit`` says. On company
    /// 337 about four in five of model 1's simulations are flagged and all
    /// of model 2's; for model 1 moment matching rescues about nineteen in
    /// twenty of them, for model 2 none, and a refit takes as long as a fit
    /// (for model 2 about 31 s of wall time with 8 running side by side).
    /// So by default (``refit=None``, the user's decisions of 2026-10-16 and
    /// 2026-10-21) model 1 moment-matches its flagged simulations and refits
    /// with NUTS only those whose ``k`` stays above the threshold, and a
    /// model with development effects (model 2) refits them with the light
    /// refit (``light_refit``): a warm-started NUTS run of 4 chains, 150
    /// warm-up and 250 kept draws, within about 0.03 sd of the total claims
    /// development result of a full refit, at about 20 times the run time
    /// of reweighting only (about 30 minutes for the default 2,000
    /// simulations on company 337). ``light_refit=False`` turns it off:
    /// model 2 then reweights its flagged simulations only, which is
    /// approximate (about 0.4 sd off a refit per simulation) and warns so
    /// (``approximate``). An explicit ``refit`` is used as it is, with the
    /// fit's own sampler unless ``light_refit`` is given, with
    /// ``max_refits`` as a budget. The view runs at most 2,000 simulations
    /// by default (``n_sims``).
    ///
    /// Raises one ``UserWarning`` listing the failed diagnostics of the
    /// fit, another if any refit failed its diagnostics, another if
    /// ``max_refits`` left simulations unrefitted, and another if the
    /// view is approximate (model 2 with ``light_refit=False``).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// paid : str
    /// premium : str
    /// outstanding : str, optional
    /// incurred : str, optional
    ///     As in ``fit``.
    /// k_threshold : float, default 0.7
    ///     The Pareto ``k`` above which a simulation's importance weights
    ///     are not trusted.
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, optional
    ///     What happens to a flagged simulation, as in ``BayesOdp.one_year``.
    ///     ``None`` (the default) takes the model's own:
    ///     ``"moment_match_then_refit"`` without development effects (model
    ///     1), ``"refit_flagged"`` with the light refit with them (model 2;
    ///     ``light_refit=False`` makes it ``"reweight_only"``, with a
    ///     warning).
    /// max_refits : int, optional
    ///     As in ``BayesOdp.one_year``.
    /// n_sims : int, optional
    ///     The number of one-year simulations. By default this model's
    ///     ``n_sims``, at most 2,000: nearly every simulation of these
    ///     models is flagged. The lifetime view in ``fit`` keeps
    ///     ``n_sims``; a smaller count gives the first simulations of a
    ///     larger one.
    /// light_refit : bool or tuple, optional
    ///     The sampler of the refits: a shorter, warm-started NUTS run
    ///     instead of the fit's own. ``None`` (the default) is the model's
    ///     own: the light refit for model 2 when ``refit`` is ``None``, the
    ///     fit's own sampler otherwise. ``True`` takes 4 chains of 150
    ///     warm-up and 250 kept draws, a tuple ``(chains, tune, draws)`` its
    ///     own. ``False`` turns model 2's default off: with ``refit=None``
    ///     the view reweights only and is approximate, with a warning. Each
    ///     refit scales its sampler by the opening posterior and starts
    ///     there, with no pilot run. On company 337 it comes within about
    ///     0.03 sd of the total claims development result of a full refit
    ///     of model 2, at about a thirtieth of the cost. A refit warns when
    ///     its closing reserve has R-hat above 1.05, fewer than 100
    ///     effective draws or a divergence.
    ///
    /// Returns
    /// -------
    /// CompartmentalOneYearFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``fit``; if ``k_threshold`` is NaN or ``refit`` is unknown.
    ///
    /// Examples
    /// --------
    /// >>> import warnings
    /// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
    /// >>> paid = [[9.6, 22.8, 33.3, 40.3, 45.1], [7.9, 19.5, 29.6, 36.8],
    /// ...         [8.7, 24.3, 35.4], [13.3, 33.0], [11.4]]
    /// >>> incurred = [[62.7, 64.0, 65.6, 65.3, 62.3], [62.1, 56.7, 56.6, 56.8],
    /// ...             [64.0, 61.5, 63.2], [73.9, 75.1], [77.1]]
    /// >>> premium = [104.4, 88.9, 86.0, 99.3, 104.9]
    /// >>> rows = [(1988 + w, 12 * (d + 1), v, incurred[w][d], premium[w])
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> origin, age, pd, inc, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(origin, age, {"paid": pd, "incurred": inc, "premium": prem})
    /// >>> model = Compartmental(CompartmentalPriors.gesmann_morris(), chains=2, tune=150,
    /// ...                       draws=150, target_accept=0.9, n_sims=500)
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     view = model.one_year(tri, "paid", "premium", incurred="incurred",
    /// ...                           refit="reweight_only")
    /// >>> view.cdr.components()[:2]
    /// [('1988',), ('1989',)]
    /// >>> view.cdr.variance() < view.fit.reserves.variance()
    /// True
    #[pyo3(signature = (
        triangle,
        paid,
        premium,
        *,
        outstanding = None,
        incurred = None,
        k_threshold = 0.7,
        refit = None,
        max_refits = None,
        n_sims = None,
        light_refit = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
        k_threshold: f64,
        refit: Option<&str>,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
        light_refit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyCompartmentalOneYearFit> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        let mut settings = one_year_settings(k_threshold, refit, max_refits, n_sims)?;
        (settings.light_refit, settings.no_light_refit) = light_refit_arg(light_refit)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year(tri, paid, os, premium, &settings))
            .map_err(err)?;
        warn_compartmental(py, &inner.fit.diagnostics.warnings)?;
        warn_refits(
            py,
            "Compartmental",
            &[(
                String::new(),
                inner.refitted.len(),
                inner.refit_warnings,
                inner.unrefitted.len(),
            )],
        )?;
        if let Some(message) = &inner.approximate {
            let message = std::ffi::CString::new(format!("Compartmental: {message}"))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyCompartmentalOneYearFit { inner })
    }

    /// The one-year view of every segment of a triangle: each segment's
    /// view as ``one_year`` gives it, on the segment's own seeds as in
    /// ``fit_segments`` (a segment's simulations differ from
    /// ``one_year`` of the segment alone), and one joint claims development
    /// result across segments, which depend on each other as ``dependence``
    /// says: ``"independent"`` as drawn, ``"rank_correlation"`` by
    /// Iman-Conover on the segments' total claims development results.
    /// Everything else is as ``one_year``: the refit default of the model
    /// (moment matching for model 1, the light refit for model 2, or
    /// reweighting only with a warning with ``light_refit=False``), at most 2,000 simulations by default, and ``max_refits``
    /// as a budget of each segment, so ``k`` segments refit at most ``k``
    /// times this many. Raises one ``UserWarning`` listing the failed
    /// diagnostics of each segment's fit (with the segment's label), another
    /// if any refit failed its diagnostics, another if ``max_refits`` left
    /// simulations unrefitted, and another if the view is approximate.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// paid : str
    /// premium : str
    /// outstanding : str, optional
    /// incurred : str, optional
    ///     As in ``fit_segments``.
    /// k_threshold : float, default 0.7
    /// refit : {"moment_match_then_refit", "refit_flagged", "reweight_only", "refit_all"}, optional
    /// max_refits : int, optional
    /// n_sims : int, optional
    /// light_refit : bool or tuple, optional
    ///     As in ``one_year``.
    ///
    /// Returns
    /// -------
    /// CompartmentalOneYearFits
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
    /// >>> from prospicio.reserving import Compartmental, CompartmentalPriors, Triangle
    /// >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
    /// ...         [983, 2211, 2830], [1657, 2685], [932]]
    /// >>> case = [[1500, 1400, 900, 400, 100], [1300, 1200, 800, 300],
    /// ...         [1400, 1100, 700], [1200, 900], [1100]]
    /// >>> premium = [5812, 4908, 5454, 5165, 5214]
    /// >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * case[w][d], k * premium[w])
    /// ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
    /// ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    /// >>> lob, origin, age, values, os, prem = map(list, zip(*rows))
    /// >>> tri = Triangle.from_long(
    /// ...     origin, age,
    /// ...     {"paid": values, "case": os, "premium": prem}, keys={"lob": lob})
    /// >>> model = Compartmental(
    /// ...     CompartmentalPriors.gesmann_morris(), tune=100, draws=100, n_sims=200,
    /// ...     chains=2)
    /// >>> with warnings.catch_warnings():
    /// ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    /// ...     views = model.one_year_segments(
    /// ...         tri, "paid", "premium", outstanding="case", refit="reweight_only")
    /// >>> views.index, views.cdr.dims
    /// (['Auto', 'Home'], ['lob', 'origin'])
    /// >>> views.segment(lob="Home").premium[:2]
    /// [11624.0, 9816.0]
    #[pyo3(signature = (
        triangle,
        paid,
        premium,
        *,
        outstanding = None,
        incurred = None,
        k_threshold = 0.7,
        refit = None,
        max_refits = None,
        n_sims = None,
        light_refit = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn one_year_segments(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
        k_threshold: f64,
        refit: Option<&str>,
        max_refits: Option<usize>,
        n_sims: Option<usize>,
        light_refit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyCompartmentalOneYearFits> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        let mut settings = one_year_settings(k_threshold, refit, max_refits, n_sims)?;
        (settings.light_refit, settings.no_light_refit) = light_refit_arg(light_refit)?;
        let (model, tri) = (&self.inner, &triangle.inner);
        let inner = py
            .detach(|| model.one_year_segments(tri, paid, os, premium, &settings))
            .map_err(err)?;
        let warnings = segment_warnings(&inner.segments, |s| &s.fit.fit.diagnostics);
        warn_compartmental(py, &warnings)?;
        warn_refits(py, "Compartmental", &segment_refits(&inner.segments))?;
        if let Some(message) = &inner.approximate {
            let message = std::ffi::CString::new(format!("Compartmental: {message}"))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            PyErr::warn(py, category.as_any(), &message, 1)?;
        }
        Ok(PyCompartmentalOneYearFits { inner })
    }

    /// Paths of the mean outstanding and cumulative paid of each origin of
    /// the premium column at each age of the square, drawn from the priors
    /// alone (Gesmann and Morris, sec. 3.4): a check of a prior set before
    /// fitting. The paths are means, without the likelihood's noise, which
    /// under the preset's effectively flat prior on ``log sigma`` would
    /// swamp them. The seed is derived from ``seed``.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with one segment.
    /// premium : str
    ///     The premium of each origin.
    /// n_sims : int
    ///     Number of paths; positive.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{"outstanding": PredictiveDistribution, "paid":
    ///     PredictiveDistribution}``, each with dimensions ``origin`` and
    ///     ``age`` (months).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a prior setting is invalid, ``n_sims`` is 0, or the premium
    ///     column is missing or not positive.
    #[pyo3(signature = (triangle, premium, n_sims = 1000))]
    fn prior_predictive<'py>(
        &self,
        py: Python<'py>,
        triangle: PyRef<'_, PyTriangle>,
        premium: &str,
        n_sims: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (model, tri) = (&self.inner, &triangle.inner);
        let paths = py
            .detach(|| model.prior_predictive(tri, premium, n_sims))
            .map_err(err)?;
        let out = PyDict::new(py);
        out.set_item(
            "outstanding",
            PyPredictiveDistribution {
                inner: paths.outstanding,
            },
        )?;
        out.set_item("paid", PyPredictiveDistribution { inner: paths.paid })?;
        Ok(out)
    }

    /// This model as a candidate of ``diagonal_backtest``, with the
    /// columns it reads: the cumulative ``paid`` column (the backtest's
    /// ``column``), the outstanding (from ``outstanding`` or from
    /// ``incurred``, as ``fit``) and each origin's ``premium``. The
    /// backtest fits the model to the cells valued before each held-out
    /// diagonal and forecasts the paid increments of the diagonal; the
    /// candidate's ``outstanding_backtest`` scores the held-out outstanding
    /// of the same diagonals on the **same fits**. The settings are
    /// ``fit``'s, except that the backtest's ``n_sims`` and ``seed``
    /// replace ``n_sims`` and ``seed``, and the sampler runs once per
    /// split (the whole ``chains`` x ``draws`` posterior of ``fit``, so
    /// size them for the number of diagonals).
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     The cumulative triangle with one segment. It must be the one
    ///     given to ``diagonal_backtest``, with ``paid`` as its ``column``.
    /// paid : str
    ///     Cumulative paid amounts.
    /// premium : str
    ///     The premium of each origin.
    /// outstanding : str, optional
    ///     A column of outstanding amounts.
    /// incurred : str, optional
    ///     A column of incurred amounts.
    ///
    /// Returns
    /// -------
    /// CompartmentalCandidate
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If not exactly one of ``outstanding`` and ``incurred`` is given
    ///     (at most one under ``fitted="paid_only"``).
    #[pyo3(signature = (triangle, paid, premium, *, outstanding = None, incurred = None))]
    fn candidate(
        &self,
        triangle: PyRef<'_, PyTriangle>,
        paid: &str,
        premium: &str,
        outstanding: Option<&str>,
        incurred: Option<&str>,
    ) -> PyResult<PyCompartmentalCandidate> {
        let os = compartmental_outstanding(self.inner.fitted, outstanding, incurred)?;
        Ok(PyCompartmentalCandidate {
            inner: CompartmentalCandidate::new(
                self.inner.clone(),
                &triangle.inner,
                paid,
                os,
                premium,
            ),
        })
    }

    fn __repr__(&self) -> String {
        let (m, s) = (&self.inner, &self.inner.sampler);
        format!(
            "Compartmental(likelihood={:?}, scale={:?}, paid_response={:?}, fitted={:?}, \
             rate_order={:?}, development_levels={:?}, start={:?}, chains={}, tune={}, \
             draws={}, n_sims={}, seed={})",
            name_of(m.likelihood, &LIKELIHOODS),
            name_of(m.scale, &SCALES),
            name_of(m.paid_response, &PAID_RESPONSES),
            name_of(m.fitted, &FITTED),
            name_of(m.rate_order, &RATE_ORDERS),
            name_of(m.development_levels, &DEVELOPMENT_LEVELS),
            name_of(m.start, &STARTS),
            s.chains,
            s.tune,
            s.draws,
            m.n_sims,
            m.seed,
        )
    }
}
