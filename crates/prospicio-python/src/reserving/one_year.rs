//! The one-year view of either bootstrap, `OneYearFit`.

use super::*;

/// A one-year view from either bootstrap.
pub(crate) enum OneYear {
    Odp(OneYearFits),
    Mack(OneYearFits<MackBootstrapSegment>),
}

/// `$body` on whichever one-year view `$fit` holds, bound to `$f`.
macro_rules! each_one_year {
    ($fit:expr, $f:ident => $body:expr) => {
        match $fit {
            OneYear::Odp($f) => $body,
            OneYear::Mack($f) => $body,
        }
    };
}

/// The simulated one-year view of every segment, from
/// ``OdpBootstrap.one_year`` or ``MackBootstrap.one_year`` (``model``
/// says which).
///
/// ``cdr`` is one joint distribution of the claims development result
/// with the triangle's keys and ``"origin"`` as dimensions, so
/// ``cdr.aggregate(["lob"])`` keeps the dependence between segments, and
/// ``cdr.quantile(0.005)`` is minus the one-year value at risk at 99.5%.
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()`` and the components of ``cdr``. ``fitted``,
/// ``residuals`` and ``scale`` need a single-segment fit; for several
/// segments use ``segment(...)`` or ``totals_frame()``. ``fitted`` and
/// ``scale`` are the ODP bootstrap's (``OdpBootstrapFit``'s), ``mack`` is
/// Mack's bootstrap's model (a ``MackFit`` of every segment, as
/// ``chain_ladder``), and ``residuals`` are either's.
#[pyclass(name = "OneYearFit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyOneYearFit {
    pub(crate) inner: OneYear,
}

impl PyOneYearFit {
    /// A row-major origin x development vector as nested lists.
    fn grid(&self, flat: &[f64]) -> Vec<Vec<f64>> {
        let n_dev = self.development().len();
        flat.chunks(n_dev.max(1)).map(<[f64]>::to_vec).collect()
    }

    /// The ODP bootstrap of a single-segment ODP fit, for `field`.
    fn odp(&self, field: &str, instead: &str) -> PyResult<&OdpBootstrapSegment> {
        match &self.inner {
            OneYear::Odp(f) => Ok(&single(&f.segments, field, instead)?.bootstrap),
            OneYear::Mack(_) => Err(PyValueError::new_err(format!(
                "{field} is the ODP bootstrap's; this one-year view is Mack's"
            ))),
        }
    }
}

#[pymethods]
impl PyOneYearFit {
    /// The bootstrap's model: ``"odp"`` or ``"mack"``.
    #[getter]
    fn model(&self) -> &'static str {
        match self.inner {
            OneYear::Odp(_) => "odp",
            OneYear::Mack(_) => "mack",
        }
    }

    /// The bootstrap's chain ladder: the factors the simulated next cells
    /// develop with (volume-weighted for the ODP, Mack's averaging for
    /// Mack's).
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: each_one_year!(&self.inner, f => f
                .segments
                .map(|s| s.bootstrap.chain_ladder().clone())),
        }
    }

    /// Mack's model behind ``MackBootstrap.one_year``, with its lifetime
    /// standard errors and ``claims_development_result()``.
    #[getter]
    fn mack(&self) -> PyResult<PyMackFit> {
        match &self.inner {
            OneYear::Mack(f) => Ok(PyMackFit {
                inner: f.segments.map(|s| s.bootstrap.mack.clone()),
            }),
            OneYear::Odp(_) => Err(PyValueError::new_err(
                "mack is Mack's bootstrap's model; this one-year view is the ODP's",
            )),
        }
    }

    /// Names of the triangle's key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        each_one_year!(&self.inner, f => f.segments.key_names.clone())
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        each_one_year!(&self.inner, f => segment_labels(py, &f.segments))
    }

    /// Origin period of each per-origin value and CDR component.
    #[getter]
    fn origins(&self) -> Vec<String> {
        each_one_year!(&self.inner, f => origin_labels(&f.segments))
    }

    /// Latest observed value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        each_one_year!(&self.inner, f => by_origin(&f.segments, |s| {
            s.bootstrap.chain_ladder().latest.clone()
        }))
    }

    /// The method's ultimate per origin on the observed triangle.
    #[getter]
    fn opening_ultimate(&self) -> Vec<f64> {
        each_one_year!(&self.inner, f => by_origin(&f.segments, |s| s.opening_ultimate.clone()))
    }

    /// The opening ultimate less the latest value, per origin.
    #[getter]
    fn opening_reserve(&self) -> Vec<f64> {
        each_one_year!(&self.inner, f => by_origin(&f.segments, |s| s.opening_reserve.clone()))
    }

    /// Development ages in months.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        each_one_year!(&self.inner, f => f.segments.fits[0]
            .bootstrap
            .chain_ladder()
            .development
            .development
            .clone())
    }

    /// The ODP bootstrap's fitted incremental values,
    /// ``[origin][development]``.
    #[getter]
    fn fitted(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(self.grid(&self.odp("fitted", "")?.fitted))
    }

    /// The residuals the bootstrap resamples, ``[origin][development]``:
    /// the ODP's adjusted Pearson residuals, as
    /// ``OdpBootstrapFit.residuals``, or Mack's scaled bias-adjusted
    /// residuals of the link ratios, ``[o][k]`` the link from age ``k`` to
    /// ``k + 1`` (``nan`` where there is none, from a zero, or behind a
    /// factor with a single link ratio).
    #[getter]
    fn residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        let flat = match &self.inner {
            OneYear::Odp(f) => &single(&f.segments, "residuals", "")?.bootstrap.residuals,
            OneYear::Mack(f) => &single(&f.segments, "residuals", "")?.bootstrap.residuals,
        };
        Ok(self.grid(flat))
    }

    /// The ODP bootstrap's scale parameter ``phi``.
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        Ok(self.odp("scale", "totals_frame()")?.scale)
    }

    /// Joint distribution of the claims development result (opening less
    /// closing ultimate) by segment and origin: the triangle's keys and
    /// ``"origin"`` are its dimensions, one component per segment and
    /// origin, one row per simulation. Its ``mean``, ``variance`` and
    /// ``quantile`` describe the total. Columns of ``draw_matrix()`` follow
    /// ``origins``.
    #[getter]
    fn cdr(&self) -> PyPredictiveDistribution {
        PyPredictiveDistribution {
            inner: each_one_year!(&self.inner, f => f.cdr.clone()),
        }
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, the method's ``opening_ultimate`` and
    /// ``opening_reserve``, and the ``cdr_mean`` and ``cdr_std_dev`` of the
    /// simulated claims development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_one_year!(&self.inner, f => f.to_long()))
    }

    /// One row per segment: the key columns, the totals of ``to_frame()``'s
    /// columns, the ODP bootstrap's ``scale`` (not for Mack's), and the
    /// ``cdr_mean`` and ``cdr_std_dev`` of the segment's total claims
    /// development result. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, each_one_year!(&self.inner, f => f.totals()))
    }

    /// The bootstrap's chain ladders' development factors, one row per
    /// segment and age, as ``ChainLadderFit.development_frame``. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn development_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(
            py,
            each_one_year!(&self.inner, f => f.segments.development_table()),
        )
    }

    /// The one-year view of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``, with its part of the joint claims
    /// development result (same dimensions).
    ///
    /// Returns
    /// -------
    /// OneYearFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let inner = match &self.inner {
            OneYear::Odp(f) => OneYear::Odp(f.segment(&keys).map_err(err)?),
            OneYear::Mack(f) => OneYear::Mack(f.segment(&keys).map_err(err)?),
        };
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        let (prefix, n_sims) = each_one_year!(&self.inner, f => (
            segments_prefix(&f.segments),
            f.cdr.n_sims(),
        ));
        let detail = match &self.inner {
            OneYear::Odp(f) => match f.segments.fits.as_slice() {
                [one] => format!(", scale={:?}", one.bootstrap.scale),
                _ => String::new(),
            },
            OneYear::Mack(_) => ", model=\"mack\"".to_string(),
        };
        format!(
            "OneYearFit({prefix}origins={}, n_sims={n_sims}{detail})",
            self.origins().len(),
        )
    }
}
