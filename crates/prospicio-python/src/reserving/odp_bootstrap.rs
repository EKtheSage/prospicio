//! The ODP bootstrap, `OdpBootstrap`.

use super::*;

/// Over-dispersed Poisson bootstrap of the chain ladder (England and
/// Verrall 2002), as R ChainLadder's ``BootChainLadder``: adjusted Pearson
/// residuals of the volume-weighted chain ladder are resampled into pseudo
/// triangles, each is re-projected, and process error is added to every
/// future incremental value. Simulation ``i`` uses random stream ``i`` of
/// ``seed`` for every segment in turn, so results do not depend on the
/// number of threads.
///
/// Parameters
/// ----------
/// n_sims : int, default 10000
///     Number of simulations; positive.
/// seed : int, default 0
///     Seed of the simulation streams, from 0 to ``2**64 - 1``. R accepts
///     seeds below ``2**53``; a seed in both ranges gives the same draws.
/// process : {"gamma", "none"}, default "gamma"
///     Process error on each simulated future incremental value: Gamma with
///     the expected value as mean and variance ``scale * |mean|`` (R's
///     ``process.distr = "gamma"``), or none for parameter error only.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     Development past the oldest age in the lifetime view (``fit``), as
///     ``ChainLadder``'s; no tail by default, as R's ``BootChainLadder``.
///     An estimated tail (``TailCurve``, ``TailBondy``, ``TailLogLinear``)
///     is refitted on each simulation's pseudo factors, which gives its
///     parameter error; a constant one is fixed unless ``tail_std_err`` is
///     given. The step from the oldest age to ultimate is one more future
///     increment, with mean the pseudo value at the oldest age times the
///     tail factor less 1 and the same process error. ``one_year`` values
///     that step one development period after the oldest age and takes it
///     for the origins whose step falls in the coming year, closing them;
///     ``method``'s tail must then be this one. R's rule
///     (``TailLogLinear``) refitted on the ODP's pseudo factors is often
///     exactly 1 (5.7% of simulations on RAA) and on average high (RAA's
///     oldest origin's mean reserve 78% above the plug-in); a constant tail
///     with ``tail_std_err`` has neither.
/// tail_std_err : float, optional
///     Standard error of a constant tail factor: each simulation draws the
///     factor from the lognormal with the factor as mean and this standard
///     deviation. Not given, the factor is fixed, unlike
///     ``MackBootstrap``, which extrapolates Mack's standard error. Unused
///     when the factor is 1; an error with an estimated tail.
/// dependence : {"independent", "synchronized", "rank_correlation"}, default "independent"
///     How the segments of a multi-segment triangle depend on each other
///     in ``fit`` and ``one_year``: each resamples its own residuals
///     independently; every segment resamples the residuals of the same
///     origins and ages in each simulation (the synchronous bootstrap of
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
///     A selected development pattern, as in ``ChainLadder``. The residuals
///     and the scale come from the triangle's own volume-weighted factors;
///     each simulation then uses ``w * selected + (1 - w) * pseudo`` at
///     the selected ages, so a fully selected factor is the same in every
///     simulation (no parameter error) and a blended one moves by ``1 - w``
///     times the pseudo factor's deviation. The origins' level still moves
///     (each simulation projects from its pseudo latest values), so even a
///     fully selected pattern keeps part of the spread: with every factor
///     selected and no process error, about 73% of the unblended standard
///     deviation on RAA. This is the ODP's parameter error in the origins'
///     levels, which a development pattern does not fix, and it is kept on
///     purpose; ``MackBootstrap`` projects from the observed latest values
///     and keeps only the process error. A supplied pattern lowers the
///     reported uncertainty. With no ``tail``, a cumulative pattern to the
///     oldest age implies a constant tail (``ChainLadderFit.pattern_tail``),
///     the same in every simulation unless ``tail_std_err`` is given. In
///     ``one_year``, a bootstrap without a pattern takes ``method``'s
///     (``bootstrap_ldf``); one with a pattern needs ``method`` to select the
///     same factors, and the tail its pattern implies to be ``method``'s.
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``, for the volume-weighted chain ladder the
///     bootstrap fits (it has no average setting). The fitted values are
///     back-cast with the factors under the exclusions; an incremental cell
///     has a residual only if the link ratio it closes is kept (the first
///     age's cells always do), and the pseudo factors average the kept
///     links only. The drops are fixed on the observed triangle: the
///     simulated triangles are not ranked again. The scale is the sum of
///     the kept squared residuals over ``N - p`` (chainladder-python's
///     rule), which is biased low under exclusions: on RAA with
///     ``n_periods=3`` it is about 70% of the plain scale (694 against
///     984), and the standard deviations are correspondingly narrow. In
///     ``one_year``, a bootstrap that sets none of them takes ``method``'s;
///     one that sets its own must set the same as ``method``, else a
///     ``ValueError`` names both. The mean of the claims development
///     result is then not zero: the window moves and the drops are ranked
///     again on the closing triangle, as the next year's reserving does.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero, ``process`` or ``dependence`` is unknown, or
///     ``spearman`` is missing for ``"rank_correlation"``, given for another
///     dependence or not square; if ``n_periods`` is zero.
/// TypeError
///     If ``tail`` is not a number or a tail estimator.
/// OverflowError
///     If ``n_sims`` or ``seed`` is negative or too large.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import OdpBootstrap, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = OdpBootstrap(n_sims=2000, seed=42).fit(tri, "values")
/// >>> fit.reserves.components()
/// [('2020',), ('2021',), ('2022',), ('2023',)]
/// >>> fit.reserves.mean() > 0
/// True
#[pyclass(name = "OdpBootstrap", module = "prospicio.reserving", frozen)]
pub(crate) struct PyOdpBootstrap {
    inner: OdpBootstrap,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyOdpBootstrap {
    #[new]
    #[pyo3(signature = (
        n_sims = 10_000,
        seed = 0,
        process = "gamma",
        tail = None,
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
        tail: Option<&Bound<'_, PyAny>>,
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
            inner: OdpBootstrap {
                n_sims,
                seed,
                process: process_distribution(process)?,
                tail: tail_arg(tail)?,
                tail_std_err,
                dependence: segment_dependence(dependence, spearman)?,
                pattern: selected_pattern(ldf, ldf_kind, ldf_weight)?,
                one_year_pattern: OneYearPattern::Method,
                n_periods,
                drop_high,
                drop_low,
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

    /// Process error: ``"gamma"`` or ``"none"``.
    #[getter]
    fn process(&self) -> &'static str {
        process_name(self.inner.process)
    }

    /// The lifetime view's tail: a constant factor as a number, otherwise
    /// its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.tail)
    }

    /// The standard error a constant tail factor is drawn with, or
    /// ``None`` to keep it fixed.
    #[getter]
    fn tail_std_err(&self) -> Option<f64> {
        self.inner.tail_std_err
    }

    /// The selected pattern by age, as given; ``None`` without one.
    #[getter]
    fn ldf<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        pattern_values(py, self.inner.pattern.as_ref())
    }

    /// How ``ldf`` is expressed: ``"ldf"``, ``"cdf"`` or
    /// ``"percent_developed"``; ``None`` without a pattern.
    #[getter]
    fn ldf_kind(&self) -> Option<&'static str> {
        pattern_kind_of(self.inner.pattern.as_ref())
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
        pattern_weight(py, self.inner.pattern.as_ref())
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

    /// Link ratios that start before the latest ``n_periods`` calendar
    /// diagonals are left out; ``None`` keeps all.
    #[getter]
    fn n_periods(&self) -> Option<usize> {
        self.inner.n_periods
    }

    /// How many of the highest link ratios at each age are left out.
    #[getter]
    fn drop_high(&self) -> usize {
        self.inner.drop_high
    }

    /// How many of the lowest link ratios at each age are left out.
    #[getter]
    fn drop_low(&self) -> usize {
        self.inner.drop_low
    }

    /// Bootstraps one measure column in every segment of a cumulative
    /// triangle, each with its own residuals and scale, into one joint
    /// distribution of the reserves, the segments depending on each other
    /// as ``dependence`` says. Every origin must be observed from the first
    /// age up to its latest.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments.
    /// column : str
    ///
    /// Returns
    /// -------
    /// OdpBootstrapFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, and if an origin has a gap before its
    ///     latest age, a segment has too few observed cells for the
    ///     degrees of freedom to be positive, synchronized segments differ
    ///     in their origins, ages or observed cells, or ``spearman`` does
    ///     not fit the segments; with an estimated ``tail``, if it cannot be
    ///     refitted on some simulation's pseudo factors (the message counts
    ///     them and gives one).
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyOdpBootstrapFit> {
        let tri = &triangle.inner;
        let boot = &self.resolved(tri.origin_grain())?;
        let inner = py.detach(|| boot.fit_segments(tri, column)).map_err(err)?;
        Ok(PyOdpBootstrapFit { inner })
    }

    /// The one-year view of any reserving method: the claims development
    /// result over the coming year, by re-reserving on the bootstrap
    /// ("actuary in the box"). The coming year is every cell valued in the
    /// twelve months after the segment's valuation: one per origin for an
    /// annual development grain, four for a quarterly one (fewer for an
    /// origin that reaches the last age). Each simulation resamples the
    /// residuals for the volume-weighted factors, projects the increments
    /// of those cells in turn from the origin's resampled latest value with
    /// the bootstrap's process error, as ``fit`` projects, adds them to the
    /// observed latest value, appends the cells to the triangle, refits
    /// ``method`` and records ``CDR = opening ultimate - closing
    /// ultimate``; a negative CDR is an adverse development. An origin whose
    /// remaining cells all fall in the year thus has its lifetime bootstrap
    /// reserve as its one-year view; an origin at the last age gets no new
    /// cell; an origin short of the latest diagonal develops from its own
    /// latest cell, and only the year's cells are appended. Unlike
    /// ``MackFit.claims_development_result()`` (Merz and Wüthrich), any
    /// averaging, tail and development grain are allowed.
    ///
    /// Without a bootstrap ``tail``, development past the oldest age moves
    /// only through ``method``'s refitted tail. With one, which must be
    /// ``method``'s tail, the tail is drawn in each simulation as ``fit``
    /// draws it, and its step to ultimate is valued one development period
    /// after the oldest age: an origin whose step falls in the coming year
    /// (with an annual grain, one already at the oldest age) takes it and
    /// is closed at that simulated ultimate. The step is the whole tail: a
    /// curve or a decayed constant tail is taken as the product of its
    /// factors in that one step, so an origin that takes it has its
    /// lifetime tail risk in the coming year, an upper bound on its
    /// one-year risk.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments and any development
    ///     grain.
    /// column : str
    /// method : ChainLadder, ExpectedLoss, BornhuetterFerguson, Benktander or CapeCod
    ///     The method refitted at the start and at the end of the year.
    /// exposure : str, optional
    ///     The exposure column; required by the expected-loss methods, not
    ///     taken by ``ChainLadder``. Its latest value per origin is kept for
    ///     the end of the year.
    /// bootstrap_ldf : bool, default True
    ///     With a bootstrap that has no ``ldf`` of its own, whether it takes
    ///     ``method``'s selected pattern, so that each simulation's pseudo
    ///     factors are blended with it as the method's factors are (as R's
    ///     ``odp_one_year(bootstrap_ldf = TRUE)`` and the Rust default,
    ///     ``OneYearPattern::Method``). ``False`` simulates from the
    ///     triangle's own factors and lets the pattern enter only when the
    ///     method re-reserves; it is an error with a bootstrap that has its
    ///     own ``ldf``. A bootstrap's own ``ldf`` must select the same factors
    ///     and weights as ``method``'s, in any order or kind; and when it
    ///     implies a tail (a cumulative pattern to the oldest age, no
    ///     ``tail``), ``method`` must use the same tail. The ``ldf`` taken
    ///     from ``method`` is read as ``method`` reads it: with no ``tail``
    ///     on either, a cumulative pattern to the oldest age implies the
    ///     same tail for both; when ``method`` has a ``tail``, to which it
    ///     rescales the pattern's level, the pattern implies no tail for
    ///     the bootstrap either.
    ///
    /// Returns
    /// -------
    /// OneYearFit
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``method`` is not one of the classes above.
    /// ValueError
    ///     As ``fit`` and the method's own ``fit``; if ``exposure`` is
    ///     missing for an expected-loss method or given for
    ///     ``ChainLadder``; if the bootstrap has a ``tail`` that is not the
    ///     one ``method`` uses (its own, or the one its ``ldf`` implies), or
    ///     an ``ldf`` that selects other factors or weights
    ///     than ``method``'s; or if the refit (or the bootstrap's tail refit)
    ///     fails in any simulation (the message counts them and gives one).
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import BornhuetterFerguson, OdpBootstrap, Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    /// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    /// ...     {
    /// ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    /// ...         "premium": [250.0] * 4 + [260.0] * 3 + [270.0] * 2 + [280.0],
    /// ...     },
    /// ... )
    /// >>> boot = OdpBootstrap(n_sims=2000, seed=42)
    /// >>> fit = boot.one_year(tri, "paid", BornhuetterFerguson(apriori=0.7), exposure="premium")
    /// >>> fit.cdr.components()
    /// [('2020',), ('2021',), ('2022',), ('2023',)]
    /// >>> fit.opening_reserve[0]
    /// 0.0
    /// >>> fit.cdr.variance() < boot.fit(tri, "paid").reserves.variance()
    /// True
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
        boot.one_year_pattern = one_year_pattern(boot.pattern.is_some(), bootstrap_ldf)?;
        let tri = &triangle.inner;
        let inner = py
            .detach(|| boot.one_year_segments(tri, column, &method))
            .map_err(err)?;
        Ok(PyOneYearFit {
            inner: OneYear::Odp(inner),
        })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl PyOdpBootstrap {
    /// The method with its pattern converted as ``ldf_origin`` and
    /// ``ldf_underwriting`` say, for
    /// origins of `origin`.
    fn resolved(&self, origin: Grain) -> PyResult<OdpBootstrap> {
        let mut m = self.inner.clone();
        m.pattern = averaged(&m.pattern, &self.ldf_basis, origin)?;
        Ok(m)
    }

    fn base_repr(&self) -> String {
        let std_err = self
            .inner
            .tail_std_err
            .map_or(String::new(), |v| format!(", tail_std_err={v:?}"));
        format!(
            "OdpBootstrap(n_sims={}, seed={}, process={:?}, tail={}{std_err}{}{}{})",
            self.inner.n_sims,
            self.inner.seed,
            self.process(),
            tail_arg_repr(&self.inner.tail),
            dependence_repr(&self.inner.dependence),
            exclusions_arg_repr(
                self.inner.n_periods,
                self.inner.drop_high,
                self.inner.drop_low,
            ),
            pattern_arg_repr(self.inner.pattern.as_ref()),
        )
    }
}

/// A fitted ODP bootstrap of every segment.
///
/// ``reserves`` is one joint distribution with the triangle's keys and
/// ``"origin"`` as dimensions, so ``reserves.aggregate(["lob"])`` keeps the
/// dependence between segments. Per-origin lists run over the origins of
/// each segment in turn, like the rows of ``to_frame()`` and the
/// components of ``reserves``. ``fitted``, ``residuals`` and ``scale``
/// need a single-segment fit; for several segments use ``segment(...)`` or
/// ``totals_frame()``. ``fitted`` and ``residuals`` are nested lists
/// indexed ``[origin][development]``, like one segment of
/// ``Triangle.values``, with ``nan`` where the triangle is not observed.
#[pyclass(name = "OdpBootstrapFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyOdpBootstrapFit {
    inner: OdpBootstrapFits,
}

impl PyOdpBootstrapFit {
    /// A row-major origin x development vector as nested lists.
    fn grid(&self, flat: &[f64]) -> Vec<Vec<f64>> {
        let n_dev = self.development().len();
        flat.chunks(n_dev.max(1)).map(<[f64]>::to_vec).collect()
    }

    fn one(&self, field: &str, instead: &str) -> PyResult<&OdpBootstrapSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[pymethods]
impl PyOdpBootstrapFit {
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
    /// The deterministic volume-weighted chain ladder the bootstrap is
    /// centred on, with the bootstrap's tail.
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

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.segments.fits[0]
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// Fitted incremental values, ``[origin][development]``.
    #[getter]
    fn fitted(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(self.grid(&self.one("fitted", "")?.fitted))
    }

    /// Adjusted Pearson residuals ``(x - m) / sqrt(|m|) * sqrt(n / (n - p))``,
    /// ``[origin][development]``; ``nan`` where not observed or where the
    /// fitted value is zero.
    #[getter]
    fn residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(self.grid(&self.one("residuals", "")?.residuals))
    }

    /// The scale parameter ``phi``: the sum of squared unadjusted residuals
    /// over the degrees of freedom ``n - p``.
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        Ok(self.one("scale", "totals_frame()")?.scale)
    }

    /// Joint distribution of the reserve (the sum of future incremental
    /// values) by segment and origin: the triangle's keys and ``"origin"``
    /// are its dimensions, one component per segment and origin, one row
    /// per simulation. Its ``mean`` and ``quantile`` describe the total
    /// reserve. Columns of ``draw_matrix()`` follow ``origins``.
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

    /// One row per segment: the key columns, the chain ladder's totals, the
    /// ``scale``, and the ``mean`` and ``std_dev`` of the segment's
    /// bootstrapped total reserve. Needs pandas.
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
    /// (same dimensions).
    ///
    /// Returns
    /// -------
    /// OdpBootstrapFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    fn __repr__(&self) -> String {
        let scale = match self.inner.segments.fits.as_slice() {
            [one] => format!(", scale={:?}", one.scale),
            _ => String::new(),
        };
        format!(
            "OdpBootstrapFit({}origins={}, n_sims={}{scale})",
            segments_prefix(&self.inner.segments),
            self.origins().len(),
            self.inner.reserves.n_sims(),
        )
    }
}

/// Which factors a bootstrap simulates the one-year view with: with
/// `bootstrap_ldf`, the method's pattern when it has none of its own (the
/// Rust default, as R's); without, the triangle's own factors. `own` is
/// whether the bootstrap has a pattern of its own.
pub(crate) fn one_year_pattern(own: bool, bootstrap_ldf: bool) -> PyResult<OneYearPattern> {
    match (own, bootstrap_ldf) {
        (_, true) => Ok(OneYearPattern::Method),
        (false, false) => Ok(OneYearPattern::Estimated),
        (true, false) => Err(PyValueError::new_err(
            "bootstrap_ldf=False takes a bootstrap without ldf: the bootstrap has its own",
        )),
    }
}
