//! The Mack bootstrap, `MackBootstrap`.

use super::*;

/// The method of ``OdpBootstrap.one_year``: a chain ladder without
/// exposure, or an expected-loss method with its exposure column.
pub(crate) fn one_year_method(
    method: &Bound<'_, PyAny>,
    exposure: Option<String>,
    origin: Grain,
) -> PyResult<OneYearMethod> {
    if let Ok(m) = method.extract::<PyRef<'_, PyChainLadder>>() {
        if exposure.is_some() {
            return Err(PyValueError::new_err(
                "ChainLadder takes no exposure column",
            ));
        }
        return Ok(OneYearMethod::ChainLadder(m.resolved(origin)?));
    }
    let needs_exposure = |name: &str| {
        exposure.clone().ok_or_else(|| {
            PyValueError::new_err(format!(
                "{name} needs an exposure column: pass exposure=..."
            ))
        })
    };
    if let Ok(m) = method.extract::<PyRef<'_, PyExpectedLoss>>() {
        return Ok(OneYearMethod::ExpectedLoss(
            m.resolved(origin)?,
            needs_exposure("ExpectedLoss")?,
        ));
    }
    if let Ok(m) = method.extract::<PyRef<'_, PyBornhuetterFerguson>>() {
        return Ok(OneYearMethod::BornhuetterFerguson(
            m.resolved(origin)?,
            needs_exposure("BornhuetterFerguson")?,
        ));
    }
    if let Ok(m) = method.extract::<PyRef<'_, PyBenktander>>() {
        return Ok(OneYearMethod::Benktander(
            m.resolved(origin)?,
            needs_exposure("Benktander")?,
        ));
    }
    if let Ok(m) = method.extract::<PyRef<'_, PyCapeCod>>() {
        return Ok(OneYearMethod::CapeCod(
            m.resolved(origin)?,
            needs_exposure("CapeCod")?,
        ));
    }
    Err(PyTypeError::new_err(format!(
        "method must be ChainLadder, ExpectedLoss, BornhuetterFerguson, Benktander or \
         CapeCod, got {}",
        method.repr()?
    )))
}

/// Mack's bootstrap for the lifetime and one-year views (England, Verrall
/// and Wüthrich 2019, Appendix 1): the scaled bias-adjusted residuals of
/// the link ratios are resampled into pseudo factors, and each future
/// cumulative value, to the last age (``fit``) or over the coming year
/// (``one_year``), is drawn from the one before ``C`` (the observed latest
/// value for the first) with mean ``f* C`` and Mack's variance
/// ``sigma**2 * abs(C)**(2 - alpha)``. The lifetime view's standard
/// deviation approximates Mack's analytic standard error. Beside
/// ``OdpBootstrap`` (variance ``scale`` times the mean increment), it gives
/// the one-year view under Mack's process: with the volume-weighted chain
/// ladder and no tail, its standard deviation is
/// ``MackFit.claims_development_result()``'s (Merz and Wüthrich) within
/// Monte Carlo error. The residuals are centred by default
/// (``centre_residuals``), so the mean CDR is Merz and Wüthrich's zero and
/// the lifetime mean reserve the chain ladder's, and EVW's Table 4 expected
/// reserves agree. Uncentred, as EVW's Appendix 1 is written, the pool's
/// non-zero mean biases the pseudo factors: the mean CDR is about -0.2
/// (RAA), -0.04 (GenIns) and +0.18 (ABC) times its standard deviation, the
/// lifetime mean reserve about +17%, +0.7% and -0.8% off the chain
/// ladder's, and the one-year standard deviation up to 1.3% wide on RAA.
/// Simulation ``i`` uses random stream ``i`` of ``seed`` for every segment
/// in turn.
///
/// Parameters
/// ----------
/// n_sims : int, default 10000
///     Number of simulations; positive.
/// seed : int, default 0
///     Seed of the simulation streams, from 0 to ``2**64 - 1``.
/// process : {"gamma", "lognormal", "residuals", "normal", "none"}, default "gamma"
///     Process error on each next cumulative value: Gamma or lognormal
///     (negated for a negative mean) or normal, with Mack's mean and
///     variance; the mean plus a resampled residual times the standard
///     deviation, which carries the residuals' mean and variance; or none
///     for parameter error only.
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
///     How Mack's model averages the link ratios (its ``alpha``).
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
///     How a sigma behind a single link ratio is filled in.
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``. The bootstrap keeps the link ratios that are
///     kept at the observed triangle (the drops are not ranked again in
///     each simulation): only they have residuals and enter the pseudo
///     factors, so Mack's standard error under the same settings is its
///     standard deviation's reference. In ``one_year``, a bootstrap that
///     sets none of them takes ``method``'s; one that sets its own must set
///     the same as ``method``, else a ``ValueError`` names both. The mean
///     of the claims development result is then not zero: the window moves
///     and the drops are ranked again on the closing triangle, as the next
///     year's reserving does.
/// centre_residuals : bool, default True
///     Subtract the residuals' mean before resampling them, so that the
///     pseudo factors are unbiased, the mean CDR is about zero and the
///     lifetime mean reserve is the chain ladder's (without a tail or with
///     a constant one). ``False`` resamples them uncentred, as EVW's
///     Appendix 1 is written.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     Development past the oldest age in the lifetime view (``fit``), as
///     ``Mack``'s: one more step, to ultimate, with the process error of
///     the tail's sigma (Mack 1999). An estimated tail (``TailCurve``,
///     ``TailBondy``, ``TailLogLinear``) is refitted on each simulation's
///     pseudo factors, which gives its parameter error; a constant one is
///     drawn from the lognormal with the factor as mean and the tail's
///     standard error. No tail by default. ``one_year`` takes the step for
///     the origins whose step falls in the coming year, as
///     ``OdpBootstrap``'s does; ``method``'s tail must then be this one.
/// tail_sigma : float, optional
///     The tail's sigma (R's ``tail.sigma``); extrapolated if not given.
///     Unused when the tail factor is 1.
/// tail_std_err : float, optional
///     The constant tail factor's standard error (R's ``tail.se``);
///     extrapolated if not given, unlike ``OdpBootstrap``, which then keeps
///     the factor fixed. Unused when the tail factor is 1; an error with an
///     estimated tail.
/// dependence : {"independent", "synchronized", "rank_correlation"}, default "independent"
///     How the segments of a multi-segment triangle depend on each other
///     in ``fit`` and ``one_year``: each resamples its own residuals
///     independently; every segment resamples the link-ratio residuals of
///     the same origins and ages in each simulation (the synchronous bootstrap of
///     Taylor and McGuire 2007 and Kirschner, Kerley and Isaacs 2008), so
///     the lines' parameter error takes the correlation of their residuals
///     and process error stays independent, which needs the same origins,
///     ages and observed cells in every segment; or each is bootstrapped
///     independently and the segments' simulations are reordered as whole
///     rows (Iman–Conover on the segment totals) to the Spearman matrix
///     ``spearman`` and then put in a random order, so that row ``i`` is
///     no longer simulation ``i`` but any subset of rows is a fair sample.
/// spearman : list of list of float, optional
///     With ``dependence="rank_correlation"`` only: Spearman's rho between
///     the segments' totals, one row and column per segment in index
///     order, symmetric with a unit diagonal. It is converted to the normal
///     scores' correlation ``2 sin(pi rho / 6)``, which must be positive
///     definite.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``. Mack's sigmas
///     and residuals come from the triangle's own estimate; each simulation
///     then uses ``w * selected + (1 - w) * pseudo`` at the selected ages,
///     so a fully selected factor has no parameter error and a blended one
///     ``1 - w`` times the pseudo factor's deviation, as ``Mack`` charges
///     it; the process error is unchanged. A supplied pattern lowers the
///     reported uncertainty. Each origin is projected from its observed
///     latest value, so with every factor selected only the process error
///     is left (unlike ``OdpBootstrap``, which keeps the origins' level
///     error). With no ``tail``, a cumulative pattern to the oldest age
///     implies a constant tail (``ChainLadderFit.pattern_tail``), the same
///     in every simulation unless ``tail_std_err`` is given. In
///     ``one_year``, a bootstrap without a pattern takes ``method``'s
///     (``bootstrap_ldf``); one with a pattern needs ``method`` to select
///     the same factors.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero, a setting is unknown, or ``spearman`` is
///     missing for ``"rank_correlation"``, given for another dependence or
///     not square.
/// TypeError
///     If ``tail`` is not a number or a tail estimator.
/// OverflowError
///     If ``n_sims`` or ``seed`` is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, Mack, MackBootstrap, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = MackBootstrap(n_sims=2000, seed=42).one_year(tri, "values", ChainLadder())
/// >>> fit.model
/// 'mack'
/// >>> fit.cdr.variance() ** 0.5 < Mack().fit(tri, "values").total_standard_error
/// True
#[pyclass(name = "MackBootstrap", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMackBootstrap {
    inner: MackBootstrap,
    ldf_basis: LdfBasis,
}

fn mack_process(name: &str) -> PyResult<MackProcess> {
    MackProcess::from_name(name)
        .ok_or_else(|| crate::unknown_name("process", name, MackProcess::NAMES))
}

fn mack_process_name(p: MackProcess) -> &'static str {
    p.name()
}

#[pymethods]
impl PyMackBootstrap {
    #[new]
    #[pyo3(signature = (
        n_sims = 10_000,
        seed = 0,
        process = "gamma",
        average = "volume",
        sigma_interpolation = "log-linear",
        centre_residuals = true,
        tail = None,
        tail_sigma = None,
        tail_std_err = None,
        dependence = "independent",
        spearman = None,
        ldf = None,
        ldf_kind = None,
        ldf_weight = None,
        n_periods = None,
        drop_high = 0,
        drop_low = 0,
        ldf_origin = None,
        ldf_underwriting = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_sims: usize,
        seed: u64,
        process: &str,
        average: &str,
        sigma_interpolation: &str,
        centre_residuals: bool,
        tail: Option<&Bound<'_, PyAny>>,
        tail_sigma: Option<f64>,
        tail_std_err: Option<f64>,
        dependence: &str,
        spearman: Option<Vec<Vec<f64>>>,
        ldf: Option<&Bound<'_, PyAny>>,
        ldf_kind: Option<&str>,
        ldf_weight: Option<&Bound<'_, PyAny>>,
        n_periods: Option<usize>,
        drop_high: usize,
        drop_low: usize,
        ldf_origin: Option<&str>,
        ldf_underwriting: Option<PyRef<'_, PyUnderwritingBasis>>,
    ) -> PyResult<Self> {
        let ldf_basis = ldf_basis_arg(ldf_origin, ldf_underwriting.as_deref(), ldf)?;
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            ldf_basis,
            inner: MackBootstrap {
                n_sims,
                seed,
                process: mack_process(process)?,
                development: development(
                    average,
                    sigma_interpolation,
                    selected_pattern(ldf, ldf_kind, ldf_weight)?,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
                centre_residuals,
                tail: tail_arg(tail)?,
                tail_sigma,
                tail_std_err,
                dependence: segment_dependence(dependence, spearman)?,
                one_year_pattern: OneYearPattern::Method,
            },
        })
    }

    /// Number of simulations.
    #[getter]
    fn n_sims(&self) -> usize {
        self.inner.n_sims
    }

    /// Seed of the simulation streams.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Process error: ``"gamma"``, ``"lognormal"``, ``"residuals"``,
    /// ``"normal"`` or ``"none"``.
    #[getter]
    fn process(&self) -> &'static str {
        mack_process_name(self.inner.process)
    }

    /// How Mack's model averages the link ratios.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.development.average)
    }

    /// How a sigma behind a single link ratio is filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.development.sigma_interpolation)
    }

    /// Number of latest diagonals averaged; ``None`` uses every link ratio.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.development.n_periods
    }

    /// Highest link ratios left out at each age.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.development.drop_high
    }

    /// Lowest link ratios left out at each age.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.development.drop_low
    }

    /// Whether the residuals are centred before they are resampled.
    #[getter]
    fn centre_residuals(&self) -> bool {
        self.inner.centre_residuals
    }

    /// The lifetime view's tail: a constant factor as a number, otherwise
    /// its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.tail)
    }

    /// The given tail sigma, or ``None`` to extrapolate it.
    #[getter]
    fn tail_sigma(&self) -> Option<f64> {
        self.inner.tail_sigma
    }

    /// The given standard error of a constant tail factor, or ``None`` to
    /// extrapolate it.
    #[getter]
    fn tail_std_err(&self) -> Option<f64> {
        self.inner.tail_std_err
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.development.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.development.pattern.as_ref())
    }

    /// The origin grain ``ldf`` was measured on, averaged over the fitted
    /// origins' periods at fit time; ``None`` reads it at the origin's age.
    #[getter]
    fn ldf_origin(&self) -> Option<&'static str> {
        self.ldf_basis.origin.map(grain_name)
    }

    /// The underwriting basis ``ldf`` is converted to at fit time
    /// (``UnderwritingBasis``), or ``None``.
    #[getter]
    fn ldf_underwriting(&self) -> Option<PyUnderwritingBasis> {
        self.ldf_basis
            .underwriting
            .clone()
            .map(|inner| PyUnderwritingBasis { inner })
    }

    /// The weight on the selected factors: a number, a dict by age, or
    /// ``None`` without a pattern.
    #[getter]
    fn ldf_weight<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        pattern_weight(py, self.inner.development.pattern.as_ref())
    }

    /// How the segments depend on each other: ``"independent"``,
    /// ``"synchronized"`` or ``"rank_correlation"``.
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

    /// The lifetime view: bootstraps one measure column in every segment of
    /// a cumulative triangle, each with its own Mack model and residuals,
    /// into one joint distribution of the reserves (EVW's Appendix 1), the
    /// segments depending on each other as ``dependence`` says. Each
    /// simulation resamples the residuals into pseudo factors and draws
    /// every cumulative value from the latest observed one to the last age,
    /// each from the one before, with Mack's mean and variance; an origin's
    /// reserve is its last drawn value less its latest. With a ``tail``
    /// each origin takes one more step, to ultimate, with the tail's sigma,
    /// the tail factor refitted (an estimated tail) or drawn (a constant
    /// one) per simulation. The standard deviation approximates Mack's
    /// analytic standard error (``Mack.fit``, with the same tail; for an
    /// estimated tail the parameter error is the refit's, not Mack's
    /// extrapolated one); the mean is the chain ladder's reserve with
    /// ``centre_residuals`` (the default), without a tail or with a
    /// constant one, about 17% above it on RAA without centring.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments and any development
    ///     grain, every origin observed from the first age to its latest
    ///     with no negative value.
    /// column : str
    /// cells : bool, default False
    ///     Whether to keep each simulation's future incremental values in
    ///     the fit's ``cells``, for ``Indexation.reinflate_draws``. The
    ///     reserves and patterns are the same, bit for bit, with and
    ///     without them, since keeping them draws no other random number.
    ///     They take ``n_sims`` times the number of future cells in memory
    ///     (about 4 MB on RAA at 10,000 simulations, 66 MB on a 40 by 40
    ///     quarterly triangle), so they are kept only when asked for.
    ///
    /// Returns
    /// -------
    /// MackBootstrapFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``Mack.fit``, if an origin has a gap before its latest age, if
    ///     a cumulative value is negative, if synchronized segments differ in
    ///     their origins, ages or observed cells, if ``spearman`` does not
    ///     fit the segments, if ``tail_std_err`` is given with an estimated
    ///     tail, or if the tail cannot be refitted on some simulation's
    ///     pseudo factors (the message counts them and gives one); with
    ///     ``cells=True``, if no segment has a future cell (a fully
    ///     developed triangle with no ``tail``).
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import MackBootstrap, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    /// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    /// ... )
    /// >>> fit = MackBootstrap(n_sims=2000, seed=42).fit(tri, "values")
    /// >>> fit.reserves.components()
    /// [('2020',), ('2021',), ('2022',), ('2023',)]
    /// >>> abs(fit.reserves.mean() / fit.chain_ladder.total_reserve - 1) < 0.05
    /// True
    #[pyo3(signature = (triangle, column, *, cells = false))]
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        cells: bool,
    ) -> PyResult<PyMackBootstrapFit> {
        let tri = &triangle.inner;
        let boot = &self.resolved(tri.origin_grain())?;
        let inner = py
            .detach(|| {
                if cells {
                    boot.fit_segments_with_cells(tri, column)
                } else {
                    boot.fit_segments(tri, column)
                }
            })
            .map_err(err)?;
        Ok(PyMackBootstrapFit { inner })
    }

    /// The one-year view of any reserving method under Mack's process, as
    /// ``OdpBootstrap.one_year``: each simulation draws the cells of the
    /// coming twelve months from Mack's bootstrap, each from the one before
    /// with the same pseudo factors, appends them to the triangle, refits
    /// ``method`` and records ``CDR = opening ultimate - closing
    /// ultimate``. Without a bootstrap ``tail``, development past the
    /// oldest age moves only through ``method``'s refitted tail; with one,
    /// which must be ``method``'s, the origins whose tail step falls in
    /// the coming year take Mack's step to ultimate and are closed, as in
    /// ``OdpBootstrap.one_year``.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments and any development
    ///     grain, every origin observed from the first age to its latest
    ///     with no negative value.
    /// column : str
    /// method : ChainLadder, ExpectedLoss, BornhuetterFerguson, Benktander or CapeCod
    ///     The method refitted at the start and at the end of the year.
    /// exposure : str, optional
    ///     The exposure column; required by the expected-loss methods, not
    ///     taken by ``ChainLadder``.
    /// bootstrap_ldf : bool, default True
    ///     As ``OdpBootstrap.one_year``: without an ``ldf`` of its own, the
    ///     bootstrap takes ``method``'s selected pattern and reads it as
    ///     ``method`` does (no implied tail when ``method`` has a ``tail``);
    ///     ``False`` simulates from the triangle's own factors.
    ///
    /// Returns
    /// -------
    /// OneYearFit
    ///     With ``model == "mack"``.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``method`` is not one of the classes above.
    /// ValueError
    ///     As ``OdpBootstrap.one_year`` and ``Mack.fit``, if a cumulative
    ///     value is negative, and if the bootstrap has a ``tail`` that is
    ///     not ``method``'s.
    #[pyo3(signature = (triangle, column, method, exposure = None, bootstrap_ldf = true))]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        method: &Bound<'_, PyAny>,
        exposure: Option<String>,
        bootstrap_ldf: bool,
    ) -> PyResult<PyOneYearFit> {
        let origin = triangle.inner.origin_grain();
        let method = one_year_method(method, exposure, origin)?;
        let mut boot = self.resolved(origin)?;
        boot.one_year_pattern =
            one_year_pattern(boot.development.pattern.is_some(), bootstrap_ldf)?;
        let tri = &triangle.inner;
        let inner = py
            .detach(|| boot.one_year_segments(tri, column, &method))
            .map_err(err)?;
        Ok(PyOneYearFit {
            inner: OneYear::Mack(inner),
        })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl_resolved!(PyMackBootstrap => MackBootstrap, pattern development.pattern);

impl PyMackBootstrap {
    fn base_repr(&self) -> String {
        let given =
            |name: &str, v: Option<f64>| v.map_or(String::new(), |v| format!(", {name}={v:?}"));
        format!(
            "MackBootstrap(n_sims={}, seed={}, process={:?}, average={:?}, \
             sigma_interpolation={:?}, centre_residuals={}, tail={}{}{}{}{})",
            self.inner.n_sims,
            self.inner.seed,
            self.process(),
            self.average(),
            self.sigma_interpolation(),
            if self.inner.centre_residuals {
                "True"
            } else {
                "False"
            },
            tail_arg_repr(&self.inner.tail),
            given("tail_sigma", self.inner.tail_sigma),
            given("tail_std_err", self.inner.tail_std_err),
            dependence_repr(&self.inner.dependence),
            development_arg_repr(&self.inner.development),
        )
    }
}

/// A fitted bootstrap of Mack's model, the lifetime view, of every
/// segment (``MackBootstrap.fit``).
///
/// ``reserves`` is one joint distribution with the triangle's keys and
/// ``"origin"`` as dimensions, so ``reserves.aggregate(["lob"])`` keeps the
/// dependence between segments. Per-origin lists run over the origins of
/// each segment in turn, like the rows of ``to_frame()`` and the
/// components of ``reserves``. ``residuals`` needs a single-segment fit;
/// for several segments use ``segment(...)``.
#[pyclass(name = "MackBootstrapFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMackBootstrapFit {
    inner: MackBootstrapFits,
}

#[pymethods]
impl PyMackBootstrapFit {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Each simulation's reserve is spread over that simulation's pattern:
    /// the rises of its running maximum after each origin's latest age, the
    /// tail's share one period past the oldest age.
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
    /// The chain ladder of Mack's model (its averaging), with the
    /// bootstrap's tail: the reserves the bootstrap's mean equals with
    /// ``centre_residuals=True``, the default, without a tail or with a
    /// constant one (an estimated tail's refit has its own mean).
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.segments.map(|s| s.mack.chain_ladder.clone()),
        }
    }

    /// Mack's model on the observed triangle, with the bootstrap's tail:
    /// the factors and sigmas the simulation uses, and the analytic
    /// standard errors the simulated standard deviations approximate (with
    /// an estimated tail, up to the refit's own parameter error).
    #[getter]
    fn mack(&self) -> PyMackFit {
        PyMackFit {
            inner: self.inner.segments.map(|s| s.mack.clone()),
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

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.segments.fits[0]
            .mack
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// Mack's scaled bias-adjusted residuals of the link ratios,
    /// ``[origin][development]``, ``[o][k]`` the link from age ``k`` to
    /// ``k + 1``; ``nan`` where there is none, from a zero, or behind a
    /// factor with a single link ratio or a zero sigma. Never centred.
    #[getter]
    fn residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        let flat = &single(&self.inner.segments, "residuals", "")?.residuals;
        let n_dev = self.development().len();
        Ok(flat.chunks(n_dev.max(1)).map(<[f64]>::to_vec).collect())
    }

    /// Joint distribution of the reserve (each origin's last simulated
    /// cumulative value less its latest) by segment and origin: the
    /// triangle's keys and ``"origin"`` are its dimensions, one component
    /// per segment and origin, one row per simulation. Its ``mean`` and
    /// ``quantile`` describe the total reserve. Columns of
    /// ``draw_matrix()`` follow ``origins``.
    #[getter]
    fn reserves(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// Each simulation's future incremental values, or ``None`` unless the
    /// fit was made with ``fit(cells=True)``: each simulated cumulative
    /// value less the one before it, starting from the observed latest
    /// value, so that they are payments. The dimensions are the triangle's
    /// keys, ``"origin"`` and ``"development"`` (the age in months at the
    /// end of the cell's development period); one component per future
    /// cell, each origin's cells after its latest age in age order,
    /// segment by segment as ``reserves``; one row per simulation, in the
    /// reserves' order (under ``dependence="rank_correlation"`` each
    /// segment's cells move with its reserves). With a ``tail`` each origin
    /// has one more cell, the ultimate less the value at the oldest age, at
    /// the oldest age plus one development period, kept in every
    /// simulation even when it is 0. A fully developed origin with no tail
    /// has no cell. Each origin's cells sum to its reserve in every
    /// simulation to round-off, so ``cells.aggregate(keys + ["origin"])``
    /// gives the reserves. ``Indexation.reinflate_draws`` takes them as
    /// they are.
    ///
    /// Returns
    /// -------
    /// PredictiveDistribution or None
    #[getter]
    fn cells(&self) -> Option<PyPredictiveDistribution> {
        self.inner
            .cells
            .clone()
            .map(|inner| PyPredictiveDistribution { inner })
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

    /// One row per segment and origin: the key columns, ``origin``, the
    /// chain ladder's ``latest``, ``ultimate`` and ``reserve``, and the
    /// ``mean`` and ``std_dev`` of the bootstrapped reserve. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the chain ladder's totals, and
    /// the ``mean`` and ``std_dev`` of the segment's bootstrapped total
    /// reserve. Needs pandas.
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

    /// The bootstrap of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint reserves
    /// (same dimensions) and of ``cells``. Its ``cells`` are ``None`` when
    /// the fit kept none, and also when it kept them but this segment has
    /// no future cell (fully developed, with no tail).
    ///
    /// Returns
    /// -------
    /// MackBootstrapFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MackBootstrapFit({}origins={}, n_sims={})",
            segments_prefix(&self.inner.segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}
