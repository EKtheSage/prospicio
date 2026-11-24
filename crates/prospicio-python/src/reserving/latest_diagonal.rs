//! Latest diagonals, `LatestDiagonal` (`reserving-v02.md`, decisions 15 to 19).

use super::*;

pub(crate) fn past_pattern_end(name: &str) -> PyResult<PastPatternEnd> {
    PastPatternEnd::from_name(name)
        .ok_or_else(|| crate::unknown_name("past_pattern_end", name, PastPatternEnd::NAMES))
}

fn past_pattern_end_name(rule: PastPatternEnd) -> &'static str {
    rule.name()
}

/// Key columns of a dict from key name to column, in its order.
fn key_columns(keys: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<(String, Vec<String>)>> {
    let Some(keys) = keys.filter(|k| !k.is_none()) else {
        return Ok(Vec::new());
    };
    let dict = keys
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("keys must be a dict of key name to column"))?;
    dict.iter()
        .map(|(k, v)| {
            let name: String = k
                .extract()
                .map_err(|_| PyTypeError::new_err("key names must be strings"))?;
            let column = key_values(&v, &name)?;
            Ok((name, column))
        })
        .collect()
}

/// One valuation month: a date (``datetime.date``, a pandas ``Timestamp``
/// or a numpy ``datetime64``), not an integer year, which would read as
/// January.
fn valuation_month(obj: &Bound<'_, PyAny>) -> PyResult<Month> {
    if obj.is_instance_of::<PyBool>()
        || obj.is_instance_of::<PyInt>()
        || (obj.hasattr("__index__")? && !obj.hasattr("dtype")?)
    {
        return Err(PyTypeError::new_err(format!(
            "give the valuation month as a date, got {}: an integer year would read as January",
            py_repr(obj)
        )));
    }
    if obj.hasattr("dtype")? {
        let flat = obj
            .py()
            .import("numpy")?
            .call_method1("asarray", (obj,))?
            .call_method1("reshape", (-1,))?;
        let kind: String = flat.getattr("dtype")?.getattr("kind")?.extract()?;
        if kind != "M" {
            return Err(PyTypeError::new_err(format!(
                "give the valuation month as a date, got {}",
                py_repr(obj)
            )));
        }
        let all = months(&flat, "valuation")?;
        return match all.as_slice() {
            [one] => Ok(*one),
            _ => Err(PyValueError::new_err("valuation must be a single date")),
        };
    }
    month_of(obj, "valuation")
}

/// What `LatestDiagonal`'s constructors read: the long table's columns.
struct DiagonalArgs {
    keys: Vec<(String, Vec<String>)>,
    origin: Vec<Month>,
    age: DiagonalArgAge,
    latest: Vec<f64>,
    exposure: Option<Vec<f64>>,
    origin_grain: Grain,
    past_pattern_end: PastPatternEnd,
}

enum DiagonalArgAge {
    Ages(Vec<Lag>),
    Valuation(Month),
    Valuations(Vec<Month>),
}

impl DiagonalArgs {
    fn build(&self, py: Python<'_>) -> PyResult<PyLatestDiagonal> {
        let key_values: Vec<Vec<&str>> = self
            .keys
            .iter()
            .map(|(_, v)| v.iter().map(String::as_str).collect())
            .collect();
        let keys: Vec<(&str, &[&str])> = self
            .keys
            .iter()
            .zip(&key_values)
            .map(|((n, _), v)| (n.as_str(), v.as_slice()))
            .collect();
        let long = LongDiagonal {
            keys: &keys,
            origin: &self.origin,
            age: match &self.age {
                DiagonalArgAge::Ages(a) => DiagonalAge::Ages(a),
                DiagonalArgAge::Valuation(v) => DiagonalAge::Valuation(*v),
                DiagonalArgAge::Valuations(v) => DiagonalAge::Valuations(v),
            },
            latest: &self.latest,
            exposure: self.exposure.as_deref(),
            origin_grain: self.origin_grain,
        };
        let rule = self.past_pattern_end;
        let inner = py
            .detach(|| LatestDiagonals::from_long(&long))
            .map_err(err)?
            .with_past_pattern_end(rule);
        Ok(PyLatestDiagonal { inner })
    }
}

/// A latest diagonal on its own, one value per origin at its own age, as a
/// bordereau holds it, in one or several segments; projected with a
/// selected pattern from elsewhere by the ``fit_diagonal`` of
/// ``ChainLadder``, ``ExpectedLoss``, ``BornhuetterFerguson``,
/// ``Benktander`` and ``CapeCod`` (``docs/design/reserving-v02.md``,
/// decisions 15 to 19).
///
/// Each origin is given once per segment, at a positive age, with its
/// latest cumulative value (0 for an origin with nothing reported yet).
/// The ages may be off the pattern's grid (18 months against an annual
/// pattern): the fit reads them on the pattern's own ages, ``log(cdf)``
/// linear between them. An origin older than the pattern's end is an error
/// unless ``past_pattern_end`` says how to develop it.
///
/// Segments are named by key columns, as a ``Triangle``'s are, and sorted
/// by label; within a segment the origins keep the order given. Rows run
/// segment by segment in that order, as ``to_frame()`` lists them.
///
/// Parameters
/// ----------
/// origin : array-like
///     Any date in each row's origin period, or integer years.
/// age : array-like
///     Each row's age in months: from the start of the origin period to
///     the end of the valuation month (an accident year 2024 at December
///     2024 is 12 months).
/// latest : array-like
///     Each row's latest cumulative value.
/// exposure : array-like, optional
///     Each row's exposure (premium), for the expected-loss methods.
/// keys : dict of str to array-like, optional
///     Key columns by name, as ``Triangle.from_long`` takes them.
/// origin_grain : {"Y", "S", "Q", "M"}, default "Y"
/// past_pattern_end : {"reject", "no_development", "tail"}, default "reject"
///     How an origin older than the pattern's end (its last age; for
///     age-to-age factors, the end of their span) is developed:
///     ``"reject"`` is an error that names it; ``"no_development"`` a
///     factor of 1 there, with no ``tail`` (a cumulative pattern's level
///     at its end is then rescaled to 100% for every origin, so with a
///     level other than 100% the young origins move too); ``"tail"`` the
///     method's ``tail`` continued from the pattern's end at one step per
///     origin period, a constant (``attachment_age`` unset) or one
///     estimated on the pattern's own factors.
///
/// Examples
/// --------
/// >>> import datetime
/// >>> from prospicio.reserving import BornhuetterFerguson, ChainLadder, LatestDiagonal
/// >>> d = LatestDiagonal.at_valuation(
/// ...     [2022, 2023, 2024], datetime.date(2024, 12, 31), [900.0, 600.0, 150.0],
/// ...     exposure=[1000.0, 1000.0, 1000.0],
/// ... )
/// >>> d.ages
/// [36, 24, 12]
/// >>> cl = ChainLadder(ldf={12: 2.0, 24: 1.25, 36: 1.0}, ldf_kind="cdf")
/// >>> cl.fit_diagonal(d).ultimate
/// [900.0, 750.0, 300.0]
/// >>> bf = BornhuetterFerguson(apriori=0.7, ldf={12: 2.0, 24: 1.25, 36: 1.0}, ldf_kind="cdf")
/// >>> round(bf.fit_diagonal(d).ultimate[2], 9)
/// 500.0
#[pyclass(name = "LatestDiagonal", module = "prospicio.reserving", frozen)]
pub(crate) struct PyLatestDiagonal {
    pub(crate) inner: LatestDiagonals,
}

#[pymethods]
impl PyLatestDiagonal {
    #[new]
    #[pyo3(signature = (origin, age, latest, exposure = None, keys = None, origin_grain = "Y", past_pattern_end = "reject"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        origin: &Bound<'_, PyAny>,
        age: &Bound<'_, PyAny>,
        latest: &Bound<'_, PyAny>,
        exposure: Option<&Bound<'_, PyAny>>,
        keys: Option<&Bound<'_, PyAny>>,
        origin_grain: &str,
        past_pattern_end: &str,
    ) -> PyResult<Self> {
        DiagonalArgs {
            keys: key_columns(keys)?,
            origin: months(origin, "origin")?,
            age: DiagonalArgAge::Ages(ages(age)?),
            latest: floats(latest, "latest")?,
            exposure: exposure
                .filter(|e| !e.is_none())
                .map(|e| floats(e, "exposure"))
                .transpose()?,
            origin_grain: grain(origin_grain)?,
            past_pattern_end: self::past_pattern_end(past_pattern_end)?,
        }
        .build(py)
    }

    /// A diagonal valued at one month: each origin's age is the months from
    /// its start to the end of the valuation month (an accident year 2024
    /// at June 2024 is six months).
    ///
    /// Parameters
    /// ----------
    /// origin : array-like
    ///     Any date in each row's origin period, or integer years.
    /// valuation : date
    ///     Any date in the valuation month (``datetime.date``, a pandas
    ///     ``Timestamp`` or a numpy ``datetime64``); an integer is an error,
    ///     since a year would read as January.
    /// latest, exposure, keys, origin_grain, past_pattern_end
    ///     As for the constructor.
    ///
    /// Returns
    /// -------
    /// LatestDiagonal
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``valuation`` is not a date.
    /// ValueError
    ///     If the valuation is before an origin starts (naming the row), an
    ///     origin is given twice in a segment, or a column has another
    ///     length.
    #[staticmethod]
    #[pyo3(signature = (origin, valuation, latest, exposure = None, keys = None, origin_grain = "Y", past_pattern_end = "reject"))]
    #[allow(clippy::too_many_arguments)]
    fn at_valuation(
        py: Python<'_>,
        origin: &Bound<'_, PyAny>,
        valuation: &Bound<'_, PyAny>,
        latest: &Bound<'_, PyAny>,
        exposure: Option<&Bound<'_, PyAny>>,
        keys: Option<&Bound<'_, PyAny>>,
        origin_grain: &str,
        past_pattern_end: &str,
    ) -> PyResult<Self> {
        DiagonalArgs {
            keys: key_columns(keys)?,
            origin: months(origin, "origin")?,
            age: DiagonalArgAge::Valuation(valuation_month(valuation)?),
            latest: floats(latest, "latest")?,
            exposure: exposure
                .filter(|e| !e.is_none())
                .map(|e| floats(e, "exposure"))
                .transpose()?,
            origin_grain: grain(origin_grain)?,
            past_pattern_end: self::past_pattern_end(past_pattern_end)?,
        }
        .build(py)
    }

    /// A diagonal from a data frame, one row per segment and origin.
    ///
    /// Columns are looked up with ``data[name]``, so a pandas or Polars
    /// DataFrame works, as does a dict of columns.
    ///
    /// Parameters
    /// ----------
    /// data : DataFrame or dict
    /// origin : str
    ///     Name of the origin column (dates or integer years).
    /// latest : str
    ///     Name of the latest-value column.
    /// age : str, optional
    ///     Name of the age column, in months.
    /// valuation : date or str, optional
    ///     The valuation month of every row (a date), or the name of a
    ///     column of each row's valuation date. Give exactly one of ``age``
    ///     and ``valuation``.
    /// exposure : str, optional
    ///     Name of the exposure column.
    /// keys : str or list of str, optional
    ///     Names of the key columns.
    /// origin_grain, past_pattern_end
    ///     As for the constructor.
    ///
    /// Returns
    /// -------
    /// LatestDiagonal
    ///
    /// Examples
    /// --------
    /// >>> import datetime
    /// >>> from prospicio.reserving import LatestDiagonal
    /// >>> df = {
    /// ...     "lob": ["b", "a", "a"],
    /// ...     "year": [2024, 2023, 2024],
    /// ...     "paid": [50.0, 400.0, 150.0],
    /// ... }
    /// >>> d = LatestDiagonal.from_frame(
    /// ...     df, "year", "paid", valuation=datetime.date(2024, 12, 31), keys="lob"
    /// ... )
    /// >>> d.index, d.ages
    /// (['a', 'b'], [24, 12, 12])
    #[staticmethod]
    #[pyo3(signature = (data, origin, latest, age = None, valuation = None, exposure = None, keys = None, origin_grain = "Y", past_pattern_end = "reject"))]
    #[allow(clippy::too_many_arguments)]
    fn from_frame(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        origin: &str,
        latest: &str,
        age: Option<&str>,
        valuation: Option<&Bound<'_, PyAny>>,
        exposure: Option<&str>,
        keys: Option<&Bound<'_, PyAny>>,
        origin_grain: &str,
        past_pattern_end: &str,
    ) -> PyResult<Self> {
        let valuation = valuation.filter(|v| !v.is_none());
        let age = match (age, valuation) {
            (Some(name), None) => DiagonalArgAge::Ages(ages(&data.get_item(name)?)?),
            (None, Some(v)) if v.is_instance_of::<PyString>() => {
                let name: String = v.extract()?;
                DiagonalArgAge::Valuations(months(&data.get_item(name)?, "valuation")?)
            }
            (None, Some(v)) => DiagonalArgAge::Valuation(valuation_month(v)?),
            _ => {
                return Err(PyValueError::new_err(
                    "give exactly one of age (a column of ages) and valuation (a date, or a \
                     column of valuation dates)",
                ));
            }
        };
        let keys = match keys.filter(|k| !k.is_none()) {
            None => Vec::new(),
            Some(keys) => names(keys)?
                .into_iter()
                .map(|name| {
                    let column = key_values(&data.get_item(&name)?, &name)?;
                    Ok((name, column))
                })
                .collect::<PyResult<_>>()?,
        };
        DiagonalArgs {
            keys,
            origin: months(&data.get_item(origin)?, "origin")?,
            age,
            latest: floats(&data.get_item(latest)?, latest)?,
            exposure: exposure
                .map(|e| floats(&data.get_item(e)?, e))
                .transpose()?,
            origin_grain: grain(origin_grain)?,
            past_pattern_end: self::past_pattern_end(past_pattern_end)?,
        }
        .build(py)
    }

    /// The latest diagonal of a triangle column, in every segment: each
    /// origin at the age of its last observed cell, with its cumulative
    /// value there, as chainladder-python's ``latest_diagonal``. With
    /// ``exposure``, each origin also takes that column's last observed
    /// value. This is how a fit is applied to another book:
    /// ``source.to_chain_ladder().fit_diagonal(LatestDiagonal.from_triangle(book, "paid"))``
    /// is chainladder-python's ``fit(source).predict(book)``.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     With any number of segments.
    /// column : str
    /// exposure : str, optional
    /// past_pattern_end : {"reject", "no_development", "tail"}, default "reject"
    ///
    /// Returns
    /// -------
    /// LatestDiagonal
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a column is unknown, an origin has no observed cell, or an
    ///     exposure is not finite and positive; with keys, the message names
    ///     the segment.
    #[staticmethod]
    #[pyo3(signature = (triangle, column, exposure = None, past_pattern_end = "reject"))]
    fn from_triangle(
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: Option<&str>,
        past_pattern_end: &str,
    ) -> PyResult<Self> {
        let rule = self::past_pattern_end(past_pattern_end)?;
        let inner = LatestDiagonals::from_triangle(&triangle.inner, column, exposure)
            .map_err(err)?
            .with_past_pattern_end(rule);
        Ok(Self { inner })
    }

    /// The diagonal with an exposure per row, in the row order of
    /// ``to_frame()``: the segments in order and, within one, its origins
    /// in the order given.
    fn with_exposure(&self, exposure: &Bound<'_, PyAny>) -> PyResult<Self> {
        let exposure = floats(exposure, "exposure")?;
        let inner = self.inner.clone().with_exposure(&exposure).map_err(err)?;
        Ok(Self { inner })
    }

    /// The diagonal with another rule for origins older than the pattern's
    /// end: ``"reject"``, ``"no_development"`` or ``"tail"``.
    fn with_past_pattern_end(&self, rule: &str) -> PyResult<Self> {
        let rule = past_pattern_end(rule)?;
        Ok(Self {
            inner: self.inner.clone().with_past_pattern_end(rule),
        })
    }

    /// The one segment whose keys have the given values, as
    /// ``ChainLadderFit.segment``.
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(Self {
            inner: self.inner.segment(&keys).map_err(err)?,
        })
    }

    /// Names of the key columns; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.key_names().to_vec()
    }

    /// Label of each segment, as ``Triangle.index``.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let n_keys = self.inner.key_names().len();
        self.inner
            .labels()
            .iter()
            .map(|l| label_to_py(py, n_keys, l))
            .collect()
    }

    /// Origin grain: ``"Y"``, ``"S"``, ``"Q"`` or ``"M"``.
    #[getter]
    fn origin_grain(&self) -> &'static str {
        grain_name(self.inner.origin_grain())
    }

    /// Origin period of each row.
    #[getter]
    fn origins(&self) -> Vec<String> {
        let grain = self.inner.origin_grain();
        self.inner
            .diagonals()
            .iter()
            .flat_map(|d| d.origins().iter())
            .map(|&m| Period::containing(m, grain).to_string())
            .collect()
    }

    /// Age of each row in months.
    #[getter]
    fn ages(&self) -> Vec<Lag> {
        self.inner
            .diagonals()
            .iter()
            .flat_map(|d| d.ages().iter().copied())
            .collect()
    }

    /// Latest value of each row.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        self.inner
            .diagonals()
            .iter()
            .flat_map(|d| d.latest().iter().copied())
            .collect()
    }

    /// Exposure of each row; ``None`` when not given.
    #[getter]
    fn exposure(&self) -> Option<Vec<f64>> {
        self.inner
            .diagonals()
            .iter()
            .map(|d| d.exposure().map(<[f64]>::to_vec))
            .collect::<Option<Vec<_>>>()
            .map(|e| e.concat())
    }

    /// The rule for origins older than the pattern's end.
    #[getter]
    fn past_pattern_end(&self) -> &'static str {
        past_pattern_end_name(self.inner.diagonals()[0].past_pattern_end())
    }

    /// The rows as a pandas DataFrame: the key columns, then ``origin``,
    /// ``age``, ``latest`` and, when given, ``exposure``.
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = PyDict::new(py);
        for (k, name) in self.inner.key_names().iter().enumerate() {
            let values: Vec<String> = self
                .inner
                .iter()
                .flat_map(|(l, d)| std::iter::repeat_n(l.parts()[k].clone(), d.origins().len()))
                .collect();
            out.set_item(name, values)?;
        }
        out.set_item("origin", self.origins())?;
        out.set_item("age", self.ages())?;
        out.set_item("latest", self.latest())?;
        if let Some(e) = self.exposure() {
            out.set_item("exposure", e)?;
        }
        py.import("pandas")?.call_method1("DataFrame", (out,))
    }

    fn __repr__(&self) -> String {
        let segments = if self.inner.len() > 1 {
            format!("segments={}, ", self.inner.len())
        } else {
            String::new()
        };
        format!(
            "LatestDiagonal({segments}origins={}, origin_grain={:?}, exposure={}, past_pattern_end={:?})",
            self.ages().len(),
            self.origin_grain(),
            if self.exposure().is_some() {
                "True"
            } else {
                "False"
            },
            self.past_pattern_end()
        )
    }
}

/// A segment label as ``index`` writes it: ``"Total"`` without keys, a str
/// with one key, a tuple with several.
fn label_from_py(obj: &Bound<'_, PyAny>, n_keys: usize) -> PyResult<Label> {
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let parts: Vec<String> = tuple
            .iter()
            .map(|p| Ok(p.str()?.to_string()))
            .collect::<PyResult<_>>()?;
        return Ok(Label::new(parts));
    }
    let s = obj.str()?.to_string();
    Ok(if n_keys == 0 && s == "Total" {
        Label::default()
    } else {
        Label::from(s)
    })
}

/// The development of each segment of `diagonal` from the ``development``
/// and ``segment_map`` arguments of ``fit_diagonal``: ``None`` without
/// ``development``.
pub(crate) fn diagonal_development(
    diagonal: &LatestDiagonals,
    development: Option<&Bound<'_, PyAny>>,
    segment_map: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<SegmentFits<ChainLadder>>> {
    let segment_map = segment_map.filter(|m| !m.is_none());
    let Some(development) = development.filter(|d| !d.is_none()) else {
        if segment_map.is_some() {
            return Err(PyValueError::new_err(
                "segment_map is given without development",
            ));
        }
        return Ok(None);
    };
    let source = development
        .extract::<PyRef<'_, PyChainLadderFit>>()
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "development must be a ChainLadderFit; for another method's fit, pass its \
                 .chain_ladder, got {}",
                py_repr(development)
            ))
        })?;
    let map = match segment_map {
        None => SegmentMap::ByLabel,
        Some(m) => {
            let dict = m.cast::<PyDict>().map_err(|_| {
                PyTypeError::new_err(
                    "segment_map must be a dict from a segment of the diagonal to a segment of \
                     the development",
                )
            })?;
            let (n_target, n_source) = (diagonal.key_names().len(), source.inner.key_names.len());
            SegmentMap::Pairs(
                dict.iter()
                    .map(|(t, s)| Ok((label_from_py(&t, n_target)?, label_from_py(&s, n_source)?)))
                    .collect::<PyResult<_>>()?,
            )
        }
    };
    diagonal
        .developments(&source.inner, &map)
        .map(Some)
        .map_err(err)
}
