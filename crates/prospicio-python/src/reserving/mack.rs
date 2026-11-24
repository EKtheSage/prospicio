//! Mack's model, `Mack` and `MackFit`, and `ClaimsDevelopmentResult`.

use super::*;

/// Mack's distribution-free chain ladder: the chain-ladder projection plus
/// the standard error of each origin's reserve and of the total, split into
/// process and parameter risk (Mack 1993, 1999).
///
/// A tail other than 1 is one more development step, from the oldest age
/// to ultimate, with its own sigma and standard error, as R ChainLadder's
/// ``MackChainLadder(tail = ...)``; unless given, both are extrapolated
/// log-linearly. Every origin, the oldest included, carries the tail's risk.
/// A tail below 1 follows chainladder-python: it scales the ultimates and
/// carries the risk read where a tail of 1.001 would be. R's
/// ``MackChainLadder`` ignores a tail below 1 altogether.
///
/// Parameters
/// ----------
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// n_periods, drop_high, drop_low : optional
///     As ``ChainLadder``. Mack's standard errors follow the link ratios
///     that are kept (R's ``MackChainLadder(weights = ...)``).
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
/// tail_sigma : float, optional
///     The tail's sigma (R's ``tail.sigma``); extrapolated if not given.
///     Unused when the tail factor is 1.
/// tail_std_err : float, optional
///     The tail factor's standard error (R's ``tail.se``); extrapolated if
///     not given. Unused when the tail factor is 1.
/// ldf, ldf_kind, ldf_weight, ldf_origin, ldf_underwriting : optional
///     A selected development pattern, as in ``ChainLadder``. A selected
///     factor is taken as known: it has no parameter error, and a blended
///     one ``(1 - w)`` times the estimate's standard error, so the
///     parameter risk is ``(1 - w)`` times Mack's on those links (none for
///     a fully selected pattern). The process risk keeps the triangle's
///     sigmas, around the factors used. A supplied pattern therefore lowers
///     the reported standard errors, and a wrong benchmark is not reflected
///     in them. ``claims_development_result()`` takes the pattern too. With
///     no ``tail``, a cumulative pattern to the oldest age implies a constant
///     tail (``ChainLadderFit.pattern_tail``) with no parameter error
///     (``tail_std_err`` 0 unless given) and its sigma extrapolated, as a
///     constant tail without ``tail_sigma``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Mack, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = Mack().fit(tri, "values")
/// >>> fit.total_standard_error > 0 and fit.standard_error[0] == 0
/// True
#[pyclass(name = "Mack", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMack {
    inner: Mack,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyMack {
    #[new]
    #[pyo3(signature = (
        average = "volume",
        sigma_interpolation = "log-linear",
        tail = None,
        tail_sigma = None,
        tail_std_err = None,
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
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
        tail_sigma: Option<f64>,
        tail_std_err: Option<f64>,
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
        let pattern = selected_pattern(ldf, ldf_kind, ldf_weight)?;
        Ok(Self {
            ldf_basis,
            inner: Mack {
                development: development(
                    average,
                    sigma_interpolation,
                    pattern,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
                tail: tail_arg(tail)?,
                tail_sigma,
                tail_std_err,
            },
        })
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.development.average)
    }

    /// How unestimable variance parameters are filled in.
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

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.tail)
    }

    /// The given tail sigma, or ``None`` to extrapolate it.
    #[getter]
    fn tail_sigma(&self) -> Option<f64> {
        self.inner.tail_sigma
    }

    /// The given standard error of the tail factor, or ``None`` to
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

    /// Fits one measure column in every segment of a triangle, each on its
    /// own.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///
    /// Returns
    /// -------
    /// MackFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, and if the triangle has fewer than three
    ///     ages, a variance parameter can be neither estimated nor
    ///     interpolated, or the tail's sigma or standard error can neither be
    ///     extrapolated nor is given.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyMackFit> {
        let tri = &triangle.inner;
        let mack = self.resolved(tri.origin_grain())?;
        let inner = py.detach(|| mack.fit_segments(tri, column)).map_err(err)?;
        Ok(PyMackFit { inner })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl PyMack {
    /// The method with its pattern converted as ``ldf_origin`` and
    /// ``ldf_underwriting`` say, for
    /// origins of `origin`.
    fn resolved(&self, origin: Grain) -> PyResult<Mack> {
        let mut m = self.inner.clone();
        m.development.pattern = averaged(&m.development.pattern, &self.ldf_basis, origin)?;
        Ok(m)
    }

    fn base_repr(&self) -> String {
        let given =
            |name: &str, v: Option<f64>| v.map_or(String::new(), |v| format!(", {name}={v:?}"));
        format!(
            "Mack(average={:?}, sigma_interpolation={:?}, tail={}{}{}{})",
            self.average(),
            self.sigma_interpolation(),
            tail_arg_repr(&self.inner.tail),
            given("tail_sigma", self.inner.tail_sigma),
            given("tail_std_err", self.inner.tail_std_err),
            development_arg_repr(&self.inner.development),
        )
    }
}

/// A fitted Mack model of every segment: the chain-ladder fields, plus
/// standard errors of each origin's reserve and of each segment's total.
///
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()``. Per-age lists, the tail and the totals'
/// standard errors need a single-segment fit; for several segments use
/// ``development_frame()``, ``totals_frame()`` (the totals' standard errors
/// and the tail) or ``segment(...)``.
/// ``total_ultimate`` and ``total_reserve`` sum over every segment.
#[pyclass(name = "MackFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyMackFit {
    pub(crate) inner: SegmentFits<MackFit>,
}

impl PyMackFit {
    /// The chain ladder of a single-segment fit, for a `field` found in the
    /// long table `instead` (if any) of a fit with several segments.
    fn one(&self, field: &str, instead: &str) -> PyResult<&ChainLadderFit> {
        Ok(&single(&self.inner, field, instead)?.chain_ladder)
    }
}

#[pymethods]
impl PyMackFit {
    /// The unpaid claims discounted simulation by simulation, by origin and
    /// in total, for the IFRS 17 liability for incurred claims and its risk
    /// adjustment (``DiscountedReserves``, ``docs/design/reserving-ifrs17.md``).
    ///
    /// Analytic Mack has no simulations to discount: this raises
    /// ``NotSupportedError``; discount a ``MackBootstrap`` fit instead.
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
        let (_, first) = self.inner.iter().next().expect("a fit has a segment");
        discounted(py, first, args)
    }

    /// The underlying chain-ladder projection.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.map(|m| m.chain_ladder.clone()),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.key_names.clone()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        segment_labels(py, &self.inner)
    }

    /// Origin period of each per-origin value.
    #[getter]
    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner)
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.fits[0]
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// Selected age-to-age factors, as ``ChainLadderFit.ldf``.
    #[getter]
    fn ldf(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("ldf", "development_frame()")?.ldf().to_vec())
    }

    /// Age-to-ultimate factors, including the tail.
    #[getter]
    fn cdf(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("cdf", "development_frame()")?.cdf.clone())
    }

    /// Tail factor from the oldest age to ultimate.
    #[getter]
    fn tail(&self) -> PyResult<f64> {
        Ok(self.one("tail", "totals_frame()")?.tail.factor)
    }

    /// Factors as estimated, as ``ChainLadderFit.estimated_ldf``.
    #[getter]
    fn estimated_ldf(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("estimated_ldf", "")?.development.ldf.clone())
    }

    /// Standard error of each estimated factor, as
    /// ``ChainLadderFit.estimated_std_err``.
    #[getter]
    fn estimated_std_err(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("estimated_std_err", "development_frame()")?
            .development
            .std_err
            .clone())
    }

    /// The selected pattern's factor of each link, as
    /// ``ChainLadderFit.selected_ldf``.
    #[getter]
    fn selected_ldf(&self) -> PyResult<Vec<Option<f64>>> {
        Ok(self
            .one("selected_ldf", "development_frame()")?
            .development
            .selected
            .clone())
    }

    /// The weight on each link's selected factor, as
    /// ``ChainLadderFit.ldf_weight``.
    #[getter]
    fn ldf_weight(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("ldf_weight", "development_frame()")?
            .development
            .weight
            .clone())
    }

    /// The factors before the tail, as ``ChainLadderFit.blended_ldf``.
    #[getter]
    fn blended_ldf(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("blended_ldf", "development_frame()")?
            .development
            .blended_ldf())
    }

    /// The rescale of a cumulative pattern, as
    /// ``ChainLadderFit.pattern_rescale``.
    #[getter]
    fn pattern_rescale(&self) -> PyResult<Option<f64>> {
        Ok(self
            .one("pattern_rescale", "totals_frame()")?
            .pattern_rescale)
    }

    /// The tail a cumulative pattern implies, as
    /// ``ChainLadderFit.pattern_tail``.
    #[getter]
    fn pattern_tail(&self) -> PyResult<Option<f64>> {
        Ok(self.one("pattern_tail", "totals_frame()")?.pattern_tail)
    }

    /// Age from which ``ldf`` holds the tail's factors, as
    /// ``ChainLadderFit.tail_attachment_age``.
    #[getter]
    fn tail_attachment_age(&self) -> PyResult<Lag> {
        attachment_age(self.one("tail_attachment_age", "")?)
    }

    /// Factors past the oldest age, as ``ChainLadderFit.tail_ldf``.
    #[getter]
    fn tail_ldf(&self) -> PyResult<Vec<f64>> {
        let cl = self.one("tail_ldf", "")?;
        Ok(cl.tail.ldf[cl.development.ldf.len()..].to_vec())
    }

    /// The tail's sigma used in the process risk: given, or extrapolated
    /// log-linearly; 0 without a tail (a factor of 1).
    #[getter]
    fn tail_sigma(&self) -> PyResult<f64> {
        Ok(self.one("tail_sigma", "totals_frame()")?.tail.sigma)
    }

    /// The tail factor's standard error used in the parameter risk: given,
    /// or extrapolated log-linearly; 0 without a tail (a factor of 1).
    #[getter]
    fn tail_std_err(&self) -> PyResult<f64> {
        Ok(self.one("tail_std_err", "totals_frame()")?.tail.std_err)
    }

    /// Variance parameter of each factor.
    #[getter]
    fn sigma(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("sigma", "development_frame()")?
            .development
            .sigma
            .clone())
    }

    /// Standard error of each factor used before the tail, as
    /// ``ChainLadderFit.std_err``: the parameter risk's.
    #[getter]
    fn std_err(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("std_err", "development_frame()")?
            .development
            .blended_std_err())
    }

    /// Number of link ratios behind each estimated factor, as
    /// ``ChainLadderFit.n_links``.
    #[getter]
    fn n_links(&self) -> PyResult<Vec<usize>> {
        Ok(self
            .one("n_links", "development_frame()")?
            .development
            .n_links
            .clone())
    }

    /// Whether ``n_periods`` or the drops left out a link ratio at some age.
    #[getter]
    fn excluded(&self) -> PyResult<bool> {
        Ok(self
            .one("excluded", "development_frame()")?
            .development
            .excluded)
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.chain_ladder.latest.clone())
    }

    /// Projected ultimate per origin.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.chain_ladder.ultimate.clone())
    }

    /// Reserve per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.chain_ladder.reserves())
    }

    /// Total ultimate across segments and origins.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    /// Total reserve across segments and origins.
    #[getter]
    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    /// Process standard error per origin.
    #[getter]
    fn process_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.process_risk.clone())
    }

    /// Parameter (estimation) standard error per origin.
    #[getter]
    fn parameter_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.parameter_risk.clone())
    }

    /// Mack standard error per origin: ``sqrt(process**2 + parameter**2)``.
    #[getter]
    fn standard_error(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.standard_error.clone())
    }

    /// Process standard error of the total reserve.
    #[getter]
    fn total_process_risk(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "total_process_risk", "totals_frame()")?.total_process_risk)
    }

    /// Parameter standard error of the total reserve, including the
    /// correlation between origins that share estimated factors.
    #[getter]
    fn total_parameter_risk(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "total_parameter_risk", "totals_frame()")?.total_parameter_risk)
    }

    /// Mack standard error of the total reserve.
    #[getter]
    fn total_standard_error(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "total_standard_error", "totals_frame()")?.total_standard_error)
    }

    /// Coefficient of variation of the total reserve.
    #[getter]
    fn total_cv(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "total_cv", "totals_frame()")?.total_cv())
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``ultimate``, ``reserve``, ``process_risk``,
    /// ``parameter_risk`` and ``standard_error``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the segment's total
    /// ``latest``, ``ultimate`` and ``reserve``, and the ``process_risk``,
    /// ``parameter_risk`` and ``standard_error`` of its total reserve.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per segment and age, as ``ChainLadderFit.development_frame``.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// Merz and Wüthrich's (2008) one-year view: the standard error of the
    /// claims development result of each origin and in total, in the next
    /// calendar year and in every later one, as R ChainLadder's
    /// ``CDR(MackChainLadder(x), dev = "all")``.
    ///
    /// The method form of the ``cdr`` property, kept as its alias.
    ///
    /// With a selected pattern (``Mack(ldf=...)``), each factor used varies
    /// with ``1 - w`` times the estimate's parameter error, so a fully
    /// selected link adds process error only; with weights strictly
    /// between 0 and 1 the calendar years' CDRs are correlated, and their
    /// standard errors need not add up to Mack's
    /// (``docs/design/reserving-v02.md``, decision 12).
    ///
    /// Returns
    /// -------
    /// ClaimsDevelopmentResult
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments (use ``segment(...)``), the
    ///     factors are not volume-weighted, the fit has a tail (a factor
    ///     other than 1, or one that replaces estimated factors), the fit
    ///     excludes link ratios (``n_periods``, ``drop_high`` or
    ///     ``drop_low``; use ``MackBootstrap.one_year``, which follows
    ///     them), or the latest values do not lie on one calendar diagonal
    ///     with one new origin per period.
    fn claims_development_result(&self) -> PyResult<PyClaimsDevelopmentResult> {
        let fit = single(&self.inner, "claims_development_result", "")?;
        let inner = fit.claims_development_result().map_err(err)?;
        Ok(PyClaimsDevelopmentResult { inner })
    }

    /// Mack's calendar-year effect test (Mack 1994, appendix H; R
    /// ChainLadder's ``cyEffTest``) on the link ratios this fit averaged:
    /// its column's, without those ``n_periods``, ``drop_high`` and
    /// ``drop_low`` exclude. A selected pattern (``ldf``) changes no link
    /// ratio and so not the test (``docs/design/reserving-v02.md``,
    /// decision 27).
    ///
    /// Parameters
    /// ----------
    /// ci : float, default 0.95
    ///     The interval's confidence level, Mack's and R's 95%.
    ///
    /// Returns
    /// -------
    /// CalendarYearTest
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments (use ``segment(...)``),
    ///     ``ci`` is not strictly between 0 and 1, or no calendar diagonal
    ///     has two link ratios.
    #[pyo3(signature = (ci = CALENDAR_YEAR_CI))]
    fn calendar_year_effect(&self, ci: f64) -> PyResult<PyCalendarYearTest> {
        let fit = single(&self.inner, "calendar_year_effect", "")?;
        let inner = fit.calendar_year_effect(ci).map_err(err)?;
        Ok(PyCalendarYearTest { inner })
    }

    /// Mack's test for correlation between adjacent development factors
    /// (Mack 1994, appendix G; R ChainLadder's ``dfCorTest``) on the link
    /// ratios this fit averaged, as ``calendar_year_effect`` reads them.
    ///
    /// Parameters
    /// ----------
    /// ci : float, default 0.5
    ///     The interval's confidence level, Mack's and R's 50%.
    ///
    /// Returns
    /// -------
    /// DevelopmentCorrelationTest
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments (use ``segment(...)``),
    ///     ``ci`` is not strictly between 0 and 1, or no pair of adjacent
    ///     link columns has two origins in common and untied ranks.
    #[pyo3(signature = (ci = DEVELOPMENT_CORRELATION_CI))]
    fn development_correlation(&self, ci: f64) -> PyResult<PyDevelopmentCorrelationTest> {
        let fit = single(&self.inner, "development_correlation", "")?;
        let inner = fit.development_correlation(ci).map_err(err)?;
        Ok(PyDevelopmentCorrelationTest { inner })
    }

    /// Merz and Wüthrich's one-year view, as a property: the same
    /// ``ClaimsDevelopmentResult`` that ``claims_development_result()``
    /// returns (that method stays as an alias), named as ``OneYearFit.cdr``
    /// names the claims development result. This one holds the analytic
    /// standard errors; ``OneYearFit.cdr`` is the simulated distribution.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``claims_development_result()``.
    ///
    /// Returns
    /// -------
    /// ClaimsDevelopmentResult
    #[getter]
    fn cdr(&self) -> PyResult<PyClaimsDevelopmentResult> {
        self.claims_development_result()
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``.
    ///
    /// Returns
    /// -------
    /// MackFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        let se = match self.inner.fits.as_slice() {
            [one] => format!(", total_standard_error={:?}", one.total_standard_error),
            _ => String::new(),
        };
        format!(
            "MackFit({}origins={}, total_reserve={:?}{se})",
            segments_prefix(&self.inner),
            self.origins().len(),
            self.inner.total_reserve(),
        )
    }
}

/// Merz and Wüthrich's (2008) one-year view of a Mack fit: standard errors
/// of the claims development result (CDR), the change in the chain-ladder
/// ultimate over a calendar year, per origin and in total. The total
/// includes the covariance between origins. Year ``k`` of the run-off is
/// R ChainLadder's ``CDR(k)S.E.``; summed in square over the years, the
/// run-off gives back Mack's standard error.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import Mack, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> mack = Mack().fit(tri, "values")
/// >>> cdr = mack.claims_development_result()
/// >>> len(cdr.by_calendar_year), cdr.one_year_standard_error[0]
/// (3, 0.0)
/// >>> abs(cdr.total_run_off_standard_error - mack.total_standard_error) < 1e-9
/// True
#[pyclass(
    name = "ClaimsDevelopmentResult",
    module = "prospicio.reserving",
    frozen
)]
pub(crate) struct PyClaimsDevelopmentResult {
    inner: ClaimsDevelopmentResult,
}

#[pymethods]
impl PyClaimsDevelopmentResult {
    /// Origin periods, oldest first.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// Standard error of each origin's CDR in the next calendar year, R's
    /// ``CDR(1)S.E.``.
    #[getter]
    fn one_year_standard_error(&self) -> Vec<f64> {
        self.inner.one_year_standard_error.clone()
    }

    /// Standard error of the total CDR in the next calendar year.
    #[getter]
    fn total_one_year_standard_error(&self) -> f64 {
        self.inner.total_one_year_standard_error
    }

    /// Standard error of each origin's CDR in each future calendar year:
    /// ``by_calendar_year[k - 1][i]`` is year ``k`` (R's ``CDR(k)S.E.``) of
    /// origin ``i``, zero once the origin is fully developed. One year per
    /// age-to-age factor.
    #[getter]
    fn by_calendar_year(&self) -> Vec<Vec<f64>> {
        self.inner.by_calendar_year.clone()
    }

    /// Standard error of the total CDR in each future calendar year.
    #[getter]
    fn total_by_calendar_year(&self) -> Vec<f64> {
        self.inner.total_by_calendar_year.clone()
    }

    /// Standard error of each origin's full run-off, the square root of
    /// the sum of its yearly mean squared errors; equals Mack's.
    #[getter]
    fn run_off_standard_error(&self) -> Vec<f64> {
        self.inner.run_off_standard_error()
    }

    /// Standard error of the total full run-off; equals Mack's.
    #[getter]
    fn total_run_off_standard_error(&self) -> f64 {
        self.inner.total_run_off_standard_error()
    }

    /// One row per origin: ``origin``, then ``cdr_1``, ``cdr_2``, ... the
    /// standard error of the CDR in each future calendar year (R's
    /// ``CDR(k)S.E.``), and ``run_off``, that of the full run-off. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut values: Vec<(String, Vec<f64>)> = self
            .inner
            .by_calendar_year
            .iter()
            .enumerate()
            .map(|(t, year)| (format!("cdr_{}", t + 1), year.clone()))
            .collect();
        values.push(("run_off".into(), self.inner.run_off_standard_error()));
        let table = FitTable {
            keys: Vec::new(),
            origin: Some(self.inner.origins.clone()),
            age: None,
            values,
        };
        table_frame(py, table)
    }

    fn __repr__(&self) -> String {
        format!(
            "ClaimsDevelopmentResult(origins={}, total_one_year_standard_error={:?}, \
             total_run_off_standard_error={:?})",
            self.inner.origins.len(),
            self.inner.total_one_year_standard_error,
            self.inner.total_run_off_standard_error()
        )
    }
}

/// The chain ladder an expected-loss method estimates its development
/// pattern with.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pattern(
    average: &str,
    sigma_interpolation: &str,
    tail: Option<&Bound<'_, PyAny>>,
    ldf: Option<&Bound<'_, PyAny>>,
    ldf_kind: Option<&str>,
    ldf_weight: Option<&Bound<'_, PyAny>>,
    n_periods: Option<usize>,
    drop_high: usize,
    drop_low: usize,
) -> PyResult<ChainLadder> {
    let pattern = selected_pattern(ldf, ldf_kind, ldf_weight)?;
    Ok(ChainLadder {
        development: development(
            average,
            sigma_interpolation,
            pattern,
            n_periods,
            drop_high,
            drop_low,
        )?,
        tail: tail_arg(tail)?,
    })
}

/// The ``repr`` arguments describing a method's development pattern.
pub(crate) fn pattern_repr(cl: &ChainLadder) -> String {
    format!(
        "average={:?}, sigma_interpolation={:?}, tail={}{}",
        average_name(cl.development.average),
        sigma_interpolation_name(cl.development.sigma_interpolation),
        tail_arg_repr(&cl.tail),
        development_arg_repr(&cl.development)
    )
}
