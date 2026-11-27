//! The chain ladder, `ChainLadder` and `ChainLadderFit`, and the helpers of segmented fits.

use super::*;

/// The chain-ladder method: each origin's latest value projected to
/// ultimate with age-to-age factors estimated from the triangle and a tail.
///
/// Parameters
/// ----------
/// average : {"volume", "simple", "regression", "geometric"}, default "volume"
///     How link ratios are averaged into one factor per age: volume
///     weighted, their mean, or least squares through the origin (Mack's
///     ``alpha`` of 1, 0 and 2), or their geometric mean. A geometric
///     average estimates the factors only: ``Mack`` and the bootstraps
///     reject it, since it has no variance model.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
///     How a variance parameter with a single link ratio is filled in.
/// n_periods : int, optional
///     Average only the link ratios that start in the latest ``n_periods``
///     calendar diagonals of the triangle; an age with fewer uses all of
///     them. All of them by default.
/// drop_high : int, default 0
///     Leave out this many of the highest link ratios at each age, among
///     those ``n_periods`` keeps. Skipped at an age where it would leave
///     none.
/// drop_low : int, default 0
///     As ``drop_high``, for the lowest link ratios.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     Development past the oldest age: a number is a constant factor from
///     the oldest age to ultimate. No tail (a factor of 1) by default.
/// ldf : dict, optional
///     A selected development pattern, from age in months to value, which
///     replaces (or, with ``ldf_weight``, blends with) the factors
///     estimated from the triangle at the ages it gives; the other ages
///     stay estimated, so a benchmark for the immature ages only is a
///     partial pattern. As ``ldf_kind`` says, the values are age-to-age
///     factors, or factors to ultimate or percents developed, which select
///     the factors between the first age and the last (``cdf[a] / cdf[next
///     age]``); one factor to ultimate (or percent developed) is the tail
///     from its age, so it is given at the oldest age, with no ``tail`` and
///     ``ldf_weight`` 1, and on a triangle observed at one age it is the
///     whole projection. When every age given is a development
///     age of the triangle, an age-to-age factor links its age to the
///     triangle's next age (any age but the oldest: development past it is
///     the tail's), and a development age the cumulative values skip is
///     interpolated: between two given ages ``b < c``, ``log(cdf)`` is
///     linear in the age, so the factor from ``a`` to the next age ``a'``
///     is ``(cdf[b] / cdf[c]) ** ((a' - a) / (c - b))`` (percents
///     developed are converted to factors to ultimate first). Otherwise the
///     pattern is read on its own ages: an age-to-age factor runs to the
///     next age given (the last one by the spacing before it), a link of
///     the triangle within the pattern's span is selected with ``log(cdf)``
///     linear between the pattern's ages, and what the pattern gives past
///     the triangle's oldest age folds into a constant tail there, at full
///     weight. So a monthly pattern applies to an annual triangle. Adding
///     one off-grid age to a sparse age-to-age pattern changes how its
///     other factors are read: each then runs to the pattern's next age,
///     not the triangle's. On RAA, ``{12: 1.5, 18: 1.1, 36: 1.2}`` selects
///     1.5484 from 12 to 24, where ``{12: 1.5, 36: 1.2}`` selects 1.5, and
///     nothing warns of the change. Their level at their last age selects
///     nothing. With no ``tail`` and a last age that is the oldest, the
///     level is the tail: a constant tail factor equal to the factor to
///     ultimate there (1 over the percent developed), with no parameter
///     error, as chainladder-python's ``DevelopmentConstant`` keeps it
///     (``ChainLadderFit.pattern_tail``). Otherwise the level is rescaled
///     to the factors and tail in use (``ChainLadderFit.pattern_rescale``).
///     A selected factor at or after a tail's attachment age is an error.
/// ldf_kind : {"ldf", "cdf", "percent_developed"}, optional
///     How ``ldf`` is expressed; required with it, since age-to-age factors
///     and factors to ultimate cannot be told apart. Factors to ultimate
///     and percents developed need only be finite and positive, in any
///     order.
/// ldf_weight : float or dict, optional
///     The credibility ``w`` on each selected factor, from 0 to 1: the
///     factor used is ``w * selected + (1 - w) * estimated``. A number
///     applies to every selected factor, a dict from age (the age that
///     starts the factor's link) to weight to those listed, the others
///     fully selected. Not given, every selected factor is used as is
///     (``w = 1``).
/// ldf_origin : {"M", "Q", "S", "Y"}, optional
///     The origin grain ``ldf`` was measured on, when it is not the
///     triangle's: an accident-month pattern is ``"M"``. At fit time the
///     pattern is averaged over the sub-periods of each origin period, with
///     the exposure even through the period, into percents developed at
///     every month (Rust's ``SelectedPattern::averaged_over_origins``). The
///     ultimates are exact for origins up to the pattern's last age; past
///     it the development of the period's later sub-periods is deferred to
///     the period's last age, which overstates the ultimate slightly. An
///     estimated tail is then fitted on monthly factors, so a constant tail
///     taken from the monthly pattern's own tail is the safer choice. The
///     triangle's origin grain must be a whole multiple of it; a weight by
///     age (``ldf_weight`` a dict, applied before the averaging; the
///     averaged pattern cannot be weighted by age here) and a single
///     age-to-age factor are errors; the triangle's own grain leaves the
///     pattern as given. Requires ``ldf``. With ``ldf_underwriting`` it is
///     the grain of the accident periods ``ldf`` is converted from instead.
/// ldf_underwriting : UnderwritingBasis, optional
///     Converts ``ldf``, an accident-period pattern, to the pattern of the
///     triangle's underwriting periods at fit time: an underwriting year's
///     share developed at age ``a`` from its start is ``sum_t e(t) p(a -
///     t)``, with ``e`` its exposure by accident month (the writing
///     convolved with even earning over the policy term,
///     ``UnderwritingBasis.exposure``) and ``p`` an accident month's (Rust's
///     ``SelectedPattern::to_underwriting_basis``). ``ldf`` is on accident
///     periods of ``ldf_origin``, or of the triangle's origin grain without
///     it: an accident-year pattern on underwriting-year origins needs no
///     ``ldf_origin``. An accident-month pattern (``ldf_origin="M"``)
///     converts under any writing and term; a pattern of longer accident
///     periods needs a policy term that is a whole multiple of them or
///     writing even within each of them. Given at every month from 1 it is
///     exact; at other ages (annual ones, say) it is read between them as
///     ``UnderwritingBasis.reading`` says: by default the Weibull fitted
///     through the period averages and corrected to the nearest monotone
///     accident-month pattern whose averages are the given values exactly,
///     within 0.2% of a simulated parallelogram book at every age, or by
///     the rule above with ``reading="ln-linear"`` (31% too much developed
///     at 12 months on that book). The result is longer than
///     ``ldf`` by the writing months plus the policy term less the accident
///     period (12 months for an accident-year pattern under the default,
///     23 for an accident-month one), and a constant tail from ``ldf``'s
///     own tail is the safer choice there. The writing must be no longer
///     than the triangle's origin period (the default's 12 months suit
///     underwriting years only; on underwriting quarters give
///     ``UnderwritingBasis([1, 1, 1])``), or the fit is an error. Requires
///     ``ldf``.
///
/// A selected factor carries no parameter error, and a blended one
/// ``(1 - w)`` times the estimate's, in ``Mack`` and the bootstraps; the
/// process error stays the triangle's. A supplied pattern therefore lowers
/// the reported uncertainty, and a wrong benchmark is not reflected in it.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, Triangle
/// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 200.0]})
/// >>> fit = ChainLadder().fit(tri, "paid")
/// >>> fit.ldf, fit.ultimate, fit.total_reserve
/// ([1.5], [150.0, 300.0], 100.0)
///
/// A benchmark for the immature ages only: 50% developed at 12 months and
/// 80% at 24, so the factor from 12 to 24 is 1.6; the factor from 24 to 36
/// stays the triangle's.
///
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2020, 2021, 2021, 2022],
/// ...     [12, 24, 36, 12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 165.0, 110.0, 170.0, 120.0]},
/// ... )
/// >>> bench = ChainLadder(ldf={12: 0.5, 24: 0.8}, ldf_kind="percent_developed").fit(tri, "paid")
/// >>> [round(f, 4) for f in bench.ldf], [round(f, 4) for f in bench.estimated_ldf]
/// ([1.6, 1.1], [1.5238, 1.1])
/// >>> round(bench.ultimate[2], 2), round(bench.pattern_rescale, 4)
/// (211.2, 1.1364)
///
/// A pattern to the oldest age with no tail keeps its level as the tail:
/// 95% developed at 36 months is a tail of 1 / 0.95.
///
/// >>> full = ChainLadder(ldf={12: 0.5, 24: 0.8, 36: 0.95}, ldf_kind="percent_developed")
/// >>> fit = full.fit(tri, "paid")
/// >>> round(fit.pattern_tail, 4), round(fit.ultimate[2], 2)
/// (1.0526, 240.0)
#[pyclass(name = "ChainLadder", module = "prospicio.reserving", frozen)]
pub(crate) struct PyChainLadder {
    inner: ChainLadder,
    ldf_basis: LdfBasis,
}

#[pymethods]
impl PyChainLadder {
    #[new]
    #[pyo3(signature = (
        average = "volume",
        sigma_interpolation = "log-linear",
        tail = None,
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
            inner: ChainLadder {
                development: development(
                    average,
                    sigma_interpolation,
                    pattern,
                    n_periods,
                    drop_high,
                    drop_low,
                )?,
                tail: tail_arg(tail)?,
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
    ///     Cumulative or incremental, with any number of segments.
    /// column : str
    ///
    /// Returns
    /// -------
    /// ChainLadderFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the column is unknown, a factor cannot be estimated or the tail
    ///     cannot be fitted (a constant that is not positive, a curve with
    ///     fewer than two factors above 1 to fit); with keys, the message
    ///     names the segment.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyChainLadderFit> {
        let tri = &triangle.inner;
        let cl = self.resolved(tri.origin_grain())?;
        let inner = py.detach(|| cl.fit_segments(tri, column)).map_err(err)?;
        Ok(PyChainLadderFit { inner })
    }

    /// Projects a latest diagonal, in every segment, with the fully
    /// selected pattern ``ldf`` (``ldf_weight`` 1, no ``n_periods`` or
    /// drops) or with the development of a fit (``development``):
    /// ``docs/design/reserving-v02.md``, decisions 15 to 19. Each origin's
    /// latest value is multiplied by the pattern's factor to ultimate at
    /// its own age, read ``log(cdf)``-linearly between the pattern's ages,
    /// and the tail. The fit's development axis is the grid the diagonal is
    /// placed on, which spans the pattern's ages as well as the
    /// diagonal's; its origins run oldest first over every period from the
    /// earliest to the latest. An origin older than the pattern's end is
    /// developed as the diagonal's ``past_pattern_end`` says.
    ///
    /// Parameters
    /// ----------
    /// diagonal : LatestDiagonal
    /// development : ChainLadderFit, optional
    ///     A fit whose factors and tail develop the diagonal, each segment by
    ///     the source segment of the same label (a single-segment source
    ///     develops every segment): chainladder-python's
    ///     ``fit(source).predict(target)``. For another method's fit, pass
    ///     its ``.chain_ladder``. The method itself then takes no ``ldf``,
    ///     no ``tail`` other than 1 and no ``n_periods``, ``drop_high`` or
    ///     ``drop_low``. The source's factors are read at each origin's own
    ///     age, as fitted, with no ``ldf_origin``: a source fitted on
    ///     another origin grain (accident quarters for a diagonal of
    ///     accident years, say) goes through ``to_chain_ladder()`` instead,
    ///     with ``ldf_origin`` added to the arguments it gives, one segment
    ///     at a time.
    /// segment_map : dict, optional
    ///     With ``development``, the source segment of each segment of the
    ///     diagonal, as ``{target: source}`` with labels written as
    ///     ``index`` writes them (``"Total"``, a str, or a tuple); every
    ///     segment of the diagonal listed once.
    ///
    /// Returns
    /// -------
    /// ChainLadderFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If there is no pattern, it is not fully selected, an origin is
    ///     younger than its first age, or older than its end with
    ///     ``past_pattern_end="reject"``; if the map leaves a segment
    ///     without a source; with keys, the message names the segment.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import ChainLadder, LatestDiagonal, Triangle
    /// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 200.0]})
    /// >>> own = ChainLadder().fit(tri, "paid")
    /// >>> d = LatestDiagonal.from_triangle(tri, "paid")
    /// >>> ChainLadder().fit_diagonal(d, development=own).ultimate
    /// [150.0, 300.0]
    #[pyo3(signature = (diagonal, development = None, segment_map = None))]
    fn fit_diagonal(
        &self,
        py: Python<'_>,
        diagonal: PyRef<'_, PyLatestDiagonal>,
        development: Option<&Bound<'_, PyAny>>,
        segment_map: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyChainLadderFit> {
        let d = &diagonal.inner;
        let m = self.resolved(d.origin_grain())?;
        let source = diagonal_development(d, development, segment_map)?;
        let inner = py
            .detach(|| match &source {
                None => m.fit_diagonal_segments(d),
                Some(s) => m.fit_diagonal_segments_from(d, s),
            })
            .map_err(err)?;
        Ok(PyChainLadderFit { inner })
    }

    fn __repr__(&self) -> String {
        with_ldf_basis(self.base_repr(), &self.ldf_basis)
    }
}

impl PyChainLadder {
    /// The method with its pattern converted as ``ldf_origin`` and
    /// ``ldf_underwriting`` say, for
    /// origins of `origin`.
    pub(crate) fn resolved(&self, origin: Grain) -> PyResult<ChainLadder> {
        chain_ladder_on(&self.inner, &self.ldf_basis, origin)
    }

    fn base_repr(&self) -> String {
        format!(
            "ChainLadder(average={:?}, sigma_interpolation={:?}, tail={}{})",
            self.average(),
            self.sigma_interpolation(),
            tail_arg_repr(&self.inner.tail),
            development_arg_repr(&self.inner.development)
        )
    }
}

/// `impl_one_segment!(PyX => Segment)`: `PyX::one(field, instead)`, the
/// one segment of a fit over segments (`self.inner.segments`) for a getter
/// that needs a single-segment fit, or [`single`]'s error.
macro_rules! impl_one_segment {
    ($ty:ty => $segment:ty) => {
        impl $ty {
            fn one(&self, field: &str, instead: &str) -> PyResult<&$segment> {
                single(&self.inner.segments, field, instead)
            }
        }
    };
}
pub(crate) use impl_one_segment;

/// The one fit of a single-segment result, or an error naming what to use
/// instead: the long table `instead` (if any) or `segment(...)`.
pub(crate) fn single<'a, T>(
    fits: &'a SegmentFits<T>,
    field: &str,
    instead: &str,
) -> PyResult<&'a T> {
    match fits.fits.as_slice() {
        [one] => Ok(one),
        _ => {
            let table = if instead.is_empty() {
                String::new()
            } else {
                format!("{instead} or ")
            };
            Err(PyValueError::new_err(format!(
                "{field} needs a single-segment fit, and this one has {} segments; \
                 use {table}segment(...)",
                fits.len()
            )))
        }
    }
}

/// Age from which a fit's selected factors are the tail's.
pub(crate) fn attachment_age(fit: &ChainLadderFit) -> PyResult<Lag> {
    fit.development
        .development
        .get(fit.tail.attachment)
        .copied()
        .ok_or_else(|| PyValueError::new_err("the fit has no development ages"))
}

/// Per-origin values of every segment, in the order of the long rows.
pub(crate) fn by_origin<T>(fits: &SegmentFits<T>, f: impl Fn(&T) -> Vec<f64>) -> Vec<f64> {
    fits.fits.iter().flat_map(f).collect()
}

/// Origin labels of every segment, in the order of the long rows.
pub(crate) fn origin_labels<T: ReserveFit>(fits: &SegmentFits<T>) -> Vec<String> {
    fits.fits
        .iter()
        .flat_map(|f| f.chain_ladder().origins.iter().map(ToString::to_string))
        .collect()
}

/// Labels of the segments, as ``Triangle.index``.
pub(crate) fn segment_labels<'py, T>(
    py: Python<'py>,
    fits: &SegmentFits<T>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let n_keys = fits.key_names.len();
    fits.labels
        .iter()
        .map(|l| label_to_py(py, n_keys, l))
        .collect()
}

/// Key values choosing one segment, compared as ``str()`` of each value.
pub(crate) fn segment_keys(keys: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, String)>> {
    match keys {
        None => Ok(Vec::new()),
        Some(keys) => keys
            .iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, v.str()?.to_string())))
            .collect(),
    }
}

pub(crate) fn pick<T: Clone>(
    fits: &SegmentFits<T>,
    keys: Option<&Bound<'_, PyDict>>,
) -> PyResult<SegmentFits<T>> {
    let keys = segment_keys(keys)?;
    let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    fits.segment(&keys).map_err(err)
}

/// A long result table as a pandas DataFrame: the key columns, then
/// ``origin`` (str) or ``development`` (age), then the values.
pub(crate) fn table_frame<'py>(py: Python<'py>, table: FitTable) -> PyResult<Bound<'py, PyAny>> {
    let mut names: Vec<String> = Vec::new();
    if table.origin.is_some() {
        names.push("origin".into());
    }
    if table.age.is_some() {
        names.push("development".into());
    }
    names.extend(table.values.iter().map(|(n, _)| n.clone()));
    if let Some((k, _)) = table.keys.iter().find(|(k, _)| names.contains(k)) {
        return Err(PyValueError::new_err(format!(
            "key {k:?} clashes with the {k:?} column of the result"
        )));
    }
    let out = PyDict::new(py);
    for (name, values) in table.keys {
        out.set_item(name, values)?;
    }
    if let Some(origin) = table.origin {
        let labels: Vec<String> = origin.iter().map(ToString::to_string).collect();
        out.set_item("origin", labels)?;
    }
    if let Some(age) = table.age {
        out.set_item("development", age)?;
    }
    for (name, values) in table.values {
        out.set_item(name, values)?;
    }
    py.import("pandas")?.call_method1("DataFrame", (out,))
}

/// The ``repr`` prefix naming the number of segments when there are several.
pub(crate) fn segments_prefix<T>(fits: &SegmentFits<T>) -> String {
    if fits.len() > 1 {
        format!("segments={}, ", fits.len())
    } else {
        String::new()
    }
}

/// A fitted chain-ladder projection of every segment of a triangle column.
///
/// Per-origin lists (``origins``, ``latest``, ``ultimate``, ``reserve``)
/// run over the origins of each segment in turn, like the rows of
/// ``to_frame()``, so a single-segment fit has one value per origin.
/// Per-age lists (``ldf``, ``cdf``, ``sigma``, ``std_err``) and the tail
/// need a single-segment fit; for several segments use
/// ``development_frame()`` (per age), ``totals_frame()`` (``tail``,
/// ``tail_sigma``, ``tail_std_err``) or ``segment(...)``.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021] * 2,
/// ...     [12, 24, 12] * 2,
/// ...     {"paid": [100.0, 150.0, 200.0, 10.0, 20.0, 30.0]},
/// ...     keys={"lob": ["Auto"] * 3 + ["Home"] * 3},
/// ... )
/// >>> fit = ChainLadder().fit(tri, "paid")
/// >>> fit.index, fit.reserve
/// (['Auto', 'Home'], [0.0, 100.0, 0.0, 30.0])
/// >>> fit.segment(lob="Home").ldf
/// [2.0]
#[pyclass(name = "ChainLadderFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyChainLadderFit {
    pub(crate) inner: SegmentFits<ChainLadderFit>,
}

#[pymethods]
impl PyChainLadderFit {
    /// This fit, every segment, as a JSON document that ``from_json`` reads
    /// back exactly: every number bit for bit, as ``Triangle.to_json``
    /// does. Pickling uses it too.
    ///
    /// Returns
    /// -------
    /// str
    ///
    /// Examples
    /// --------
    /// >>> import pickle
    /// >>> from prospicio.reserving import ChainLadder, ChainLadderFit, Triangle
    /// >>> tri = Triangle.from_long(origin=[2020, 2020, 2021], development=[12, 24, 12],
    /// ...                          values={"paid": [100.0, 150.0, 110.0]})
    /// >>> fit = ChainLadder().fit(tri, "paid")
    /// >>> ChainLadderFit.from_json(fit.to_json()).ultimate == fit.ultimate
    /// True
    /// >>> pickle.loads(pickle.dumps(fit)).to_json() == fit.to_json()
    /// True
    fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// A fit from a document written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// ChainLadderFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the text is not such a document, or is from a newer version.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        SegmentFits::<ChainLadderFit>::from_json(text)
            .map(|inner| Self { inner })
            .map_err(err)
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let from_json = slf.get_type().getattr("from_json")?;
        Ok((from_json, (slf.get().inner.to_json(),)))
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
        self.inner.fits[0].development.development.clone()
    }

    /// The fit frozen as a ``ChainLadder``: its factors in use (``ldf``) as
    /// fully selected age-to-age factors at its ages, and its tail factor
    /// as a constant tail (``docs/design/reserving-v02.md``, decision 17).
    /// Applied to the fit's own latest diagonal it projects as the fit did,
    /// the factors to the bit; to another book's
    /// (``LatestDiagonal.from_triangle``) it is chainladder-python's
    /// ``fit(source).predict(book)``. Its ``ldf``, ``ldf_kind`` and ``tail``
    /// unpack into any method's arguments, as
    /// ``BornhuetterFerguson(apriori=0.6, ldf=cl.ldf, ldf_kind=cl.ldf_kind,
    /// tail=cl.tail)``. It does not carry sigma, the standard errors or the
    /// tail's form. Needs a single-segment fit: for several segments pass
    /// the fit as ``development`` to ``fit_diagonal``, or use
    /// ``segment(...)``.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import ChainLadder, LatestDiagonal, Triangle
    /// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 200.0]})
    /// >>> cl = ChainLadder(tail=1.1).fit(tri, "paid").to_chain_ladder()
    /// >>> cl.ldf, cl.ldf_kind, cl.tail
    /// ({12: 1.5}, 'ldf', 1.1)
    fn to_chain_ladder(&self) -> PyResult<PyChainLadder> {
        let f = single(&self.inner, "to_chain_ladder", "")?;
        Ok(PyChainLadder {
            inner: f.to_chain_ladder(),
            ldf_basis: LdfBasis::default(),
        })
    }

    /// Selected age-to-age factors, which the projection uses: the
    /// estimated ones (or a selected pattern's, ``blended_ldf``), replaced
    /// by the tail's from its attachment age. Factor ``k`` links age ``k``
    /// to ``k + 1``.
    #[getter]
    fn ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "ldf", "development_frame()")?;
        Ok(f.ldf().to_vec())
    }

    /// Age-to-ultimate factors, one per age, including the tail.
    #[getter]
    fn cdf(&self) -> PyResult<Vec<f64>> {
        Ok(single(&self.inner, "cdf", "development_frame()")?
            .cdf
            .clone())
    }

    /// Variance parameter of each factor, with unestimable ones
    /// interpolated (``nan`` where that is impossible).
    #[getter]
    fn sigma(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "sigma", "development_frame()")?;
        Ok(f.development.sigma.clone())
    }

    /// Standard error of each factor used before the tail: ``(1 -
    /// ldf_weight)`` times ``estimated_std_err``, so 0 where a selected
    /// pattern replaces the estimate.
    #[getter]
    fn std_err(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "std_err", "development_frame()")?;
        Ok(f.development.blended_std_err())
    }

    /// Number of link ratios behind each estimated factor, after
    /// ``n_periods`` and the drops; factor ``k`` links age ``k`` to ``k + 1``.
    #[getter]
    fn n_links(&self) -> PyResult<Vec<usize>> {
        let f = single(&self.inner, "n_links", "development_frame()")?;
        Ok(f.development.n_links.clone())
    }

    /// Whether ``n_periods`` or the drops left out a link ratio at some age.
    #[getter]
    fn excluded(&self) -> PyResult<bool> {
        let f = single(&self.inner, "excluded", "development_frame()")?;
        Ok(f.development.excluded)
    }

    /// Age-to-age factors as estimated from the triangle, before a selected
    /// pattern or the tail replaced any.
    #[getter]
    fn estimated_ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "estimated_ldf", "")?;
        Ok(f.development.ldf.clone())
    }

    /// Standard error of each estimated factor.
    #[getter]
    fn estimated_std_err(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "estimated_std_err", "development_frame()")?;
        Ok(f.development.std_err.clone())
    }

    /// The selected pattern's age-to-age factor of each link, after
    /// conversion from ``ldf_kind``; ``None`` where it selects none.
    #[getter]
    fn selected_ldf(&self) -> PyResult<Vec<Option<f64>>> {
        let f = single(&self.inner, "selected_ldf", "development_frame()")?;
        Ok(f.development.selected.clone())
    }

    /// The weight on each link's selected factor; 0 where none is
    /// selected.
    #[getter]
    fn ldf_weight(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "ldf_weight", "development_frame()")?;
        Ok(f.development.weight.clone())
    }

    /// The factors before the tail: ``w * selected + (1 - w) * estimated``
    /// where a factor is selected, the estimate elsewhere.
    #[getter]
    fn blended_ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "blended_ldf", "development_frame()")?;
        Ok(f.development.blended_ldf())
    }

    /// For a pattern given as factors to ultimate or percents developed,
    /// the factor that multiplies the given percents developed (and divides
    /// the given factors to ultimate) to reconcile the pattern's level at
    /// its last age with the factors and tail in use: ``cdf_given /
    /// cdf_used`` there, so that a pattern given to the oldest age with a
    /// ``tail``, rescaled, times the tail is 100%. 1 for a pattern to the
    /// oldest age with no tail, whose level is the tail (``pattern_tail``).
    /// At earlier ages the rescaled values are the projection's only where
    /// every link is fully selected; with ``ldf_weight`` below 1, read
    /// ``cdf`` for the factors in use. ``None`` for an age-to-age pattern
    /// or none.
    #[getter]
    fn pattern_rescale(&self) -> PyResult<Option<f64>> {
        Ok(single(&self.inner, "pattern_rescale", "totals_frame()")?.pattern_rescale)
    }

    /// The tail a pattern of factors to ultimate or percents developed
    /// implies: with no ``tail`` and a pattern to the oldest age, its
    /// factor to ultimate there (1 over its percent developed), used as a
    /// constant tail with no parameter error (``tail_std_err`` 0, its sigma
    /// extrapolated as a constant tail's). ``None`` with a ``tail``, for a
    /// pattern that stops before the oldest age, an age-to-age pattern, or
    /// none.
    #[getter]
    fn pattern_tail(&self) -> PyResult<Option<f64>> {
        Ok(single(&self.inner, "pattern_tail", "totals_frame()")?.pattern_tail)
    }

    /// Age from which ``ldf`` holds the tail's factors rather than the
    /// estimated ones; the oldest age when the tail replaced none.
    #[getter]
    fn tail_attachment_age(&self) -> PyResult<Lag> {
        attachment_age(single(&self.inner, "tail_attachment_age", "")?)
    }

    /// Tail factor from the oldest age to ultimate.
    #[getter]
    fn tail(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "tail", "totals_frame()")?.tail.factor)
    }

    /// Factors past the oldest age, which multiply to ``tail``: one per
    /// development period of the following year and one to ultimate, as
    /// chainladder-python's ``ldf_`` (a single factor for
    /// ``TailLogLinear``).
    #[getter]
    fn tail_ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "tail_ldf", "")?;
        Ok(f.tail.ldf[f.development.ldf.len()..].to_vec())
    }

    /// The tail's variance parameter, extrapolated log-linearly; 0 without
    /// a tail (a factor of 1), ``nan`` if it cannot be extrapolated. A tail
    /// below 1 is read where a tail of 1.001 would be, as chainladder-python
    /// does.
    #[getter]
    fn tail_sigma(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "tail_sigma", "totals_frame()")?
            .tail
            .sigma)
    }

    /// Standard error of the tail factor, extrapolated log-linearly.
    #[getter]
    fn tail_std_err(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "tail_std_err", "totals_frame()")?
            .tail
            .std_err)
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.latest.clone())
    }

    /// Projected ultimate per origin.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    /// Reserve (ultimate minus latest) per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ChainLadderFit::reserves)
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

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``ultimate`` and ``reserve``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the segment's total
    /// ``latest``, ``ultimate`` and ``reserve``, and its ``tail``,
    /// ``tail_sigma`` and ``tail_std_err``; with a selected pattern, its
    /// ``pattern_rescale`` and ``pattern_tail`` too (``nan`` where there is
    /// none). Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per segment and age: the key columns, ``development``,
    /// ``ldf`` (the selected factor to the next age), ``cdf`` (to ultimate,
    /// with the tail), ``sigma`` and ``std_err``; the oldest age has ``nan``
    /// for ``ldf``, ``sigma`` and ``std_err``, and the tail factor as its
    /// ``cdf``. With a selected pattern, also ``estimated_ldf``,
    /// ``selected_ldf`` (``nan`` where none), ``ldf_weight``,
    /// ``blended_ldf`` and ``estimated_std_err``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.development_table())
    }

    /// The fit of one segment, chosen by key values (compared as ``str()``
    /// of each value). Keys not named may take any value, so a fit with one
    /// segment needs none.
    ///
    /// Returns
    /// -------
    /// ChainLadderFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a key or value is unknown, or the choice matches several
    ///     segments.
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ChainLadderFit({}origins={}, total_ultimate={:?}, total_reserve={:?})",
            segments_prefix(&self.inner),
            self.origins().len(),
            self.inner.total_ultimate(),
            self.inner.total_reserve()
        )
    }
}
