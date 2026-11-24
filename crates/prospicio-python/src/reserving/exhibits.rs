//! The exhibits: `DevelopmentExhibit`, `UltimateExhibit` and `SelectedUltimates`.

use super::*;

/// A table with one row per label of `index` (named `index_name`) and the
/// given columns: a pandas DataFrame with pandas installed, a dict of lists
/// (the index first) otherwise.
fn labelled_frame<'py>(
    py: Python<'py>,
    index_name: &str,
    index: Vec<String>,
    columns: Vec<(String, Vec<f64>)>,
) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);
    out.set_item(index_name, index)?;
    for (name, values) in columns {
        out.set_item(name, values)?;
    }
    match pandas(py)? {
        Some(pd) => {
            let frame = pd.call_method1("DataFrame", (out,))?;
            frame.call_method1("set_index", (index_name,))
        }
        None => Ok(out.into_any()),
    }
}

/// Friedland's development exhibit (chapter 7, Exhibit I, sheets 1 and 2)
/// of one column of a single-segment triangle: the link ratios, each named
/// average of them, the selected factors, the tail, the factors to ultimate
/// and the percents developed (``docs/design/reserving-v02.md``, decision
/// 34).
///
/// The selected factors and factors to ultimate are those of the chain
/// ladder ``selection`` fitted to the column, so a selected pattern
/// (``ldf=``), the exclusions and the tail apply as in ``ChainLadder``; the
/// exhibit rounds nothing. Friedland multiplies by factors to ultimate
/// rounded to three decimals: give those with ``ldf_kind="cdf"`` to
/// reproduce her projections.
///
/// Parameters
/// ----------
/// triangle : Triangle
/// column : str
/// selection : ChainLadder, optional
///     The chain ladder whose factors are selected; ``ChainLadder()`` (the
///     volume-weighted average of every year, no tail) by default.
/// averages : dict of str to ChainLadder, optional
///     The averages to show, each the development of a ``ChainLadder``
///     (its ``average``, ``n_periods``, ``drop_high`` and ``drop_low``; its
///     tail is not used). By default Friedland's six: ``"simple latest
///     5"``, ``"simple latest 3"``, ``"medial latest 5x1"``, ``"volume
///     latest 5"``, ``"volume latest 3"`` and ``"geometric latest 4"``.
///
/// Raises
/// ------
/// ValueError
///     If the triangle has several segments or lacks the column, an
///     average cannot be estimated (it is named), or the selection cannot
///     be fitted.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import ChainLadder, DevelopmentExhibit, Triangle
/// >>> paid = [[100.0, 160.0, 180.0], [110.0, 170.0], [120.0]]
/// >>> rows = [(2020 + w, 12 * (d + 1), v) for w, r in enumerate(paid)
/// ...         for d, v in enumerate(r)]
/// >>> origin, age, values = map(list, zip(*rows))
/// >>> tri = Triangle.from_long(origin, age, {"paid": values})
/// >>> ex = DevelopmentExhibit(tri, "paid", ChainLadder(tail=1.05))
/// >>> ex.link_labels
/// ['12-24', '24-36', '36-ult']
/// >>> round(ex.cdf[0], 4), ex.tail
/// (1.8563, 1.05)
#[pyclass(name = "DevelopmentExhibit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyDevelopmentExhibit {
    inner: DevelopmentExhibit,
}

#[pymethods]
impl PyDevelopmentExhibit {
    #[new]
    #[pyo3(signature = (triangle, column, selection = None, averages = None))]
    fn new(
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        selection: Option<PyRef<'_, PyChainLadder>>,
        averages: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let grain = triangle.inner.origin_grain();
        let selection = match selection {
            Some(s) => s.resolved(grain)?,
            None => ChainLadder::default(),
        };
        let averages = match averages {
            None => friedland_averages(),
            Some(map) => {
                let mut out = Vec::new();
                for item in map.call_method0("items")?.try_iter()? {
                    let (name, method): (String, PyRef<'_, PyChainLadder>) = item?.extract()?;
                    out.push((name, method.resolved(grain)?.development));
                }
                out
            }
        };
        let inner =
            DevelopmentExhibit::with_averages(&triangle.inner, column, &selection, &averages)
                .map_err(err)?;
        Ok(Self { inner })
    }

    /// Origin labels, oldest first.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// Development ages in months.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner.ages.clone()
    }

    /// The label of each link, ``"12-24"`` and so on, then ``"120-ult"``
    /// for the tail.
    #[getter]
    fn link_labels(&self) -> Vec<String> {
        self.inner.link_labels()
    }

    /// Cumulative values by origin, then age; ``nan`` where not observed.
    #[getter]
    fn values(&self) -> Vec<Vec<f64>> {
        self.inner.values.clone()
    }

    /// Link ratios by origin, then link; ``nan`` where not observed.
    #[getter]
    fn link_ratios(&self) -> Vec<Vec<f64>> {
        self.inner.link_ratios.clone()
    }

    /// Each average's factor per link, by name, in the order given.
    #[getter]
    fn averages<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (name, factors) in &self.inner.averages {
            out.set_item(name, factors.clone())?;
        }
        Ok(out)
    }

    /// The age-to-age factors the projection uses, per link.
    #[getter]
    fn selected(&self) -> Vec<f64> {
        self.inner.selected.clone()
    }

    /// The factor from the oldest age to ultimate.
    #[getter]
    fn tail(&self) -> f64 {
        self.inner.tail
    }

    /// The factor to ultimate at each age; the last is the tail.
    #[getter]
    fn cdf(&self) -> Vec<f64> {
        self.inner.cdf.clone()
    }

    /// The share of the ultimate developed at each age, ``1 / cdf``.
    #[getter]
    fn percent_developed(&self) -> Vec<f64> {
        self.inner.percent_developed.clone()
    }

    /// Friedland's parts 2 to 4 as tables.
    ///
    /// Returns
    /// -------
    /// tuple of pandas.DataFrame or dict
    ///     The link ratios (one row per origin, a column per link label
    ///     without the tail) and the factors (one row per average, then
    ///     ``"selected"``, ``"cdf"`` and ``"percent_developed"``, a column
    ///     per link label with the tail: an average is ``nan`` there, the
    ///     selected factor is the tail). Dicts of lists without pandas.
    fn to_frames<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let ex = &self.inner;
        let labels = ex.link_labels();
        let n_links = labels.len() - 1;
        let links = labels[..n_links]
            .iter()
            .enumerate()
            .map(|(k, l)| (l.clone(), ex.link_ratios.iter().map(|r| r[k]).collect()))
            .collect();
        let ratios = labelled_frame(py, "origin", self.origins(), links)?;
        let mut rows: Vec<(String, Vec<f64>)> = ex
            .averages
            .iter()
            .map(|(n, f)| {
                let mut f = f.clone();
                f.push(f64::NAN);
                (n.clone(), f)
            })
            .collect();
        let mut selected = ex.selected.clone();
        selected.push(ex.tail);
        rows.push(("selected".into(), selected));
        rows.push(("cdf".into(), ex.cdf.clone()));
        rows.push(("percent_developed".into(), ex.percent_developed.clone()));
        let names: Vec<String> = rows.iter().map(|(n, _)| n.clone()).collect();
        let columns = labels
            .iter()
            .enumerate()
            .map(|(k, l)| (l.clone(), rows.iter().map(|(_, r)| r[k]).collect()))
            .collect();
        let factors = labelled_frame(py, "factor", names, columns)?;
        Ok((ratios, factors))
    }

    fn __repr__(&self) -> String {
        format!(
            "DevelopmentExhibit({} origins, {} ages, {} averages, tail={})",
            self.inner.origins.len(),
            self.inner.ages.len(),
            self.inner.averages.len(),
            self.inner.tail
        )
    }
}

/// The weights of a selection of ultimates from Python: ``weights`` a
/// method name, or a mapping (a dict or a DataFrame) of method name to one
/// weight for every origin or a weight per origin; or ``by_age`` a mapping
/// of age in months to a mapping of method name to weight.
fn method_weights(
    n_origins: usize,
    weights: Option<&Bound<'_, PyAny>>,
    by_age: Option<&Bound<'_, PyAny>>,
) -> PyResult<MethodWeights> {
    match (weights, by_age) {
        (Some(w), None) => {
            if let Ok(name) = w.extract::<String>() {
                return Ok(MethodWeights::Single(name));
            }
            let mut out = Vec::new();
            for item in w.call_method0("items")?.try_iter()? {
                let (name, value): (String, Bound<'_, PyAny>) = item?.extract()?;
                let values = match value.extract::<f64>() {
                    Ok(x) => vec![x; n_origins],
                    Err(_) => plain(&value)?.extract::<Vec<f64>>()?,
                };
                out.push((name, values));
            }
            Ok(MethodWeights::ByOrigin(out))
        }
        (None, Some(b)) => {
            let mut rows: Vec<(Lag, Vec<(String, f64)>)> = Vec::new();
            for item in b.call_method0("items")?.try_iter()? {
                let (age, row): (Lag, Bound<'_, PyAny>) = item?.extract()?;
                let mut methods = Vec::new();
                for entry in row.call_method0("items")?.try_iter()? {
                    methods.push(entry?.extract::<(String, f64)>()?);
                }
                rows.push((age, methods));
            }
            rows.sort_by_key(|(age, _)| *age);
            Ok(MethodWeights::ByAge(rows))
        }
        (Some(_), Some(_)) => Err(PyValueError::new_err("give weights or by_age, not both")),
        (None, None) => Err(PyValueError::new_err("give weights or by_age")),
    }
}

/// A summary of each method's ultimate per origin beside the claims to
/// date, from which IBNR (ultimate less reported), the total unpaid
/// (ultimate less paid) and the case outstanding follow: Friedland's
/// chapter 7, Exhibit I, sheet 4, and the summaries of chapter 15, Exhibit
/// I (``docs/design/reserving-v02.md``, decision 34). ``select`` weights the
/// methods' ultimates into the selected ultimate per origin.
///
/// Build it from a triangle with ``UltimateExhibit.from_triangle`` or from
/// the claims to date, then add each method with ``with_method``, which
/// returns a new exhibit.
///
/// Parameters
/// ----------
/// origin : array-like
///     Any date in each origin period, or integer years.
/// age : array-like of int
///     Each origin's age in months at the valuation date.
/// reported, paid : array-like of float, optional
///     Claims to date per origin.
/// origin_grain : str, default "Y"
///     ``"Y"``, ``"S"``, ``"Q"`` or ``"M"``.
///
/// Raises
/// ------
/// ValueError
///     If the columns differ in length, an origin is given twice, or a
///     value to date is not finite.
///
/// Examples
/// --------
/// >>> from prospicio.reserving import UltimateExhibit
/// >>> ex = UltimateExhibit([2007, 2008], [24, 12], reported=[31732.0, 18632.0],
/// ...                      paid=[11865.0, 3409.0])
/// >>> ex = ex.with_method("bf reported", [40300.0, 36842.0])
/// >>> ex = ex.with_method("frequency-severity", [30512.0, 30140.0])
/// >>> sel = ex.select({"bf reported": [1.0, 0.5], "frequency-severity": [0.0, 0.5]})
/// >>> sel.ultimate, sel.ibnr
/// ([40300.0, 33491.0], [8568.0, 14859.0])
#[pyclass(name = "UltimateExhibit", module = "prospicio.reserving", frozen)]
pub(crate) struct PyUltimateExhibit {
    inner: UltimateExhibit,
}

#[pymethods]
impl PyUltimateExhibit {
    #[new]
    #[pyo3(signature = (origin, age, reported = None, paid = None, origin_grain = "Y"))]
    fn new(
        origin: &Bound<'_, PyAny>,
        age: &Bound<'_, PyAny>,
        reported: Option<&Bound<'_, PyAny>>,
        paid: Option<&Bound<'_, PyAny>>,
        origin_grain: &str,
    ) -> PyResult<Self> {
        let g = grain(origin_grain)?;
        let origins = months(origin, "origin")?
            .into_iter()
            .map(|m| Period::containing(m, g))
            .collect();
        let floats = |v: Option<&Bound<'_, PyAny>>| -> PyResult<Option<Vec<f64>>> {
            v.map(|v| plain(v)?.extract()).transpose()
        };
        let ages: Vec<Lag> = plain(age)?.extract()?;
        let inner =
            UltimateExhibit::new(origins, ages, floats(reported)?, floats(paid)?).map_err(err)?;
        Ok(Self { inner })
    }

    /// The exhibit of a single-segment triangle's latest reported and paid
    /// values (either may be left out); each origin's age is its latest age
    /// in the columns given.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// reported, paid : str, optional
    ///     The columns of reported and paid claims; give at least one.
    ///
    /// Returns
    /// -------
    /// UltimateExhibit
    ///     With no method yet.
    #[staticmethod]
    #[pyo3(signature = (triangle, reported = None, paid = None))]
    fn from_triangle(
        triangle: PyRef<'_, PyTriangle>,
        reported: Option<&str>,
        paid: Option<&str>,
    ) -> PyResult<Self> {
        let inner = UltimateExhibit::from_triangle(&triangle.inner, reported, paid).map_err(err)?;
        Ok(Self { inner })
    }

    /// The exhibit with one more method.
    ///
    /// Parameters
    /// ----------
    /// name : str
    /// ultimate : fit or array-like of float
    ///     A fit of one segment with ``origins`` and ``ultimate`` (a
    ///     ``ChainLadderFit``, ``MackFit``, ``ExpectedLossFit``, ``CapeCodFit``
    ///     or ``ClarkFit``), whose origins must be the exhibit's, or the
    ///     ultimate per origin, ``nan`` where the method gives none: for a
    ///     bootstrap or a Bayesian fit, its mean ultimate per origin.
    ///
    /// Returns
    /// -------
    /// UltimateExhibit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the name is empty or already used, the fit's origins are not
    ///     the exhibit's, or the number of values is wrong.
    /// TypeError
    ///     If a fit (an object with ``origins``) has no ``ultimate``.
    fn with_method(&self, name: &str, ultimate: &Bound<'_, PyAny>) -> PyResult<Self> {
        let is_fit = ultimate.hasattr("origins")?;
        if is_fit && !ultimate.hasattr("ultimate")? {
            return Err(PyTypeError::new_err(format!(
                "the fit {name:?} has no ultimate per origin; give its mean ultimate per \
                 origin as a sequence of numbers"
            )));
        }
        let values: Vec<f64> = if is_fit {
            let origins: Vec<String> = ultimate.getattr("origins")?.extract()?;
            if origins != self.origins() {
                return Err(PyValueError::new_err(format!(
                    "the fit {name:?} has the origins {origins:?}, the exhibit {:?}",
                    self.origins()
                )));
            }
            plain(&ultimate.getattr("ultimate")?)?.extract()?
        } else {
            plain(ultimate)?.extract()?
        };
        let inner = self.inner.clone().with_method(name, values).map_err(err)?;
        Ok(Self { inner })
    }

    /// Origin labels, oldest first.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// Each origin's age in months at the valuation date.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner.ages.clone()
    }

    /// Reported claims to date, or ``None``.
    #[getter]
    fn reported(&self) -> Option<Vec<f64>> {
        self.inner.reported.clone()
    }

    /// Paid claims to date, or ``None``.
    #[getter]
    fn paid(&self) -> Option<Vec<f64>> {
        self.inner.paid.clone()
    }

    /// Reported less paid, or ``None`` unless both are given.
    #[getter]
    fn case_outstanding(&self) -> Option<Vec<f64>> {
        self.inner.case_outstanding()
    }

    /// The names of the methods, in the order added.
    #[getter]
    fn methods(&self) -> Vec<String> {
        self.inner.methods.iter().map(|(n, _)| n.clone()).collect()
    }

    /// The method ``name``'s ultimate per origin.
    fn ultimate(&self, name: &str) -> PyResult<Vec<f64>> {
        self.inner
            .ultimate(name)
            .map(<[f64]>::to_vec)
            .ok_or_else(|| PyValueError::new_err(format!("the exhibit has no method {name:?}")))
    }

    /// The method ``name``'s IBNR per origin, its ultimate less reported.
    fn ibnr(&self, name: &str) -> PyResult<Vec<f64>> {
        self.inner.ibnr(name).map_err(err)
    }

    /// The method ``name``'s total unpaid per origin, its ultimate less paid.
    fn unpaid(&self, name: &str) -> PyResult<Vec<f64>> {
        self.inner.unpaid(name).map_err(err)
    }

    /// One of Friedland's summaries: one row per origin and a column per
    /// method.
    ///
    /// Parameters
    /// ----------
    /// quantity : {"ultimate", "ibnr", "unpaid"}, default "ultimate"
    ///     The ultimates (after the ``reported`` and ``paid`` columns that
    ///     are given), the IBNR or the total unpaid.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame or dict
    ///     Indexed by origin; a dict of lists without pandas.
    #[pyo3(signature = (quantity = "ultimate"))]
    fn to_frame<'py>(&self, py: Python<'py>, quantity: &str) -> PyResult<Bound<'py, PyAny>> {
        let table = match quantity {
            "ultimate" => self.inner.ultimates_table(),
            "ibnr" => self.inner.ibnr_table().map_err(err)?,
            "unpaid" => self.inner.unpaid_table().map_err(err)?,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "quantity must be \"ultimate\", \"ibnr\" or \"unpaid\", got {quantity:?}"
                )));
            }
        };
        labelled_frame(py, "origin", self.origins(), table.values)
    }

    /// The selected ultimate per origin, a weighting of the methods'
    /// ultimates, with its IBNR and unpaid.
    ///
    /// Each origin's weights must be at least 0 and sum to 1 (within
    /// 1e-9); they are not rescaled, unlike chainladder-python's
    /// ``VotingChainladder``, which divides by their sum. A method left out has weight 0,
    /// and a method with weight 0 may have no ultimate (``nan``) there.
    ///
    /// Parameters
    /// ----------
    /// weights : str or mapping, optional
    ///     A method name (that method for every origin), or a mapping (a
    ///     dict or a DataFrame) of method name to one weight for every
    ///     origin or a weight per origin.
    /// by_age : mapping, optional
    ///     Instead of ``weights``, a mapping of age in months to a mapping
    ///     of method name to weight: each origin takes the row of the
    ///     oldest age not past its own age, so ``{12: {"bf": 1.0}, 48:
    ///     {"cl": 1.0}}`` selects Bornhuetter-Ferguson before 48 months and
    ///     the chain ladder from 48 months on.
    ///
    /// Returns
    /// -------
    /// SelectedUltimates
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If both or neither of ``weights`` and ``by_age`` are given, a
    ///     method is unknown or named twice, an origin's weights do not sum
    ///     to 1 or include a negative one, a method with a positive weight
    ///     has no ultimate there, or an origin is younger than the first
    ///     age of ``by_age``.
    #[pyo3(signature = (weights = None, by_age = None))]
    fn select(
        &self,
        weights: Option<&Bound<'_, PyAny>>,
        by_age: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PySelectedUltimates> {
        let w = method_weights(self.inner.n_origins(), weights, by_age)?;
        let inner = self.inner.select(&UltimateSelection::new(w)).map_err(err)?;
        Ok(PySelectedUltimates { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "UltimateExhibit({} origins, methods={:?})",
            self.inner.n_origins(),
            self.methods()
        )
    }
}

/// The selected ultimate per origin, with its IBNR and unpaid and the
/// weights that gave it (``UltimateExhibit.select``).
#[pyclass(name = "SelectedUltimates", module = "prospicio.reserving", frozen)]
pub(crate) struct PySelectedUltimates {
    inner: SelectedUltimates,
}

#[pymethods]
impl PySelectedUltimates {
    /// Origin labels, oldest first.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    /// Each origin's age in months at the valuation date.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner.ages.clone()
    }

    /// The selected ultimate per origin.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        self.inner.ultimate.clone()
    }

    /// The selected ultimate less reported, or ``None`` without reported
    /// claims.
    #[getter]
    fn ibnr(&self) -> Option<Vec<f64>> {
        self.inner.ibnr()
    }

    /// The selected ultimate less paid, or ``None`` without paid claims.
    #[getter]
    fn unpaid(&self) -> Option<Vec<f64>> {
        self.inner.unpaid()
    }

    /// Every method's weight per origin, by name.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (name, w) in &self.inner.weights {
            out.set_item(name, w.clone())?;
        }
        Ok(out)
    }

    /// The total selected ultimate.
    #[getter]
    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    /// The total IBNR, or ``None`` without reported claims.
    #[getter]
    fn total_ibnr(&self) -> Option<f64> {
        self.inner.total_ibnr()
    }

    /// The total unpaid, or ``None`` without paid claims.
    #[getter]
    fn total_unpaid(&self) -> Option<f64> {
        self.inner.total_unpaid()
    }

    /// One row per origin: ``reported`` and ``paid`` when given,
    /// ``ultimate``, ``ibnr`` and ``unpaid`` when defined, then each
    /// method's weight as ``weight_<method>``; with ``total=True`` a last
    /// row ``"total"`` with the sums (``nan`` weights).
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame or dict
    ///     Indexed by origin; a dict of lists without pandas.
    #[pyo3(signature = (total = false))]
    fn to_frame<'py>(&self, py: Python<'py>, total: bool) -> PyResult<Bound<'py, PyAny>> {
        let mut index = self.origins();
        let mut columns = self.inner.to_table().values;
        if total {
            index.push("total".into());
            for (name, values) in &mut columns {
                let sum = if name.starts_with("weight_") {
                    f64::NAN
                } else {
                    values.iter().sum()
                };
                values.push(sum);
            }
        }
        labelled_frame(py, "origin", index, columns)
    }

    fn __repr__(&self) -> String {
        format!(
            "SelectedUltimates({} origins, total_ultimate={})",
            self.inner.origins.len(),
            self.inner.total_ultimate()
        )
    }
}
