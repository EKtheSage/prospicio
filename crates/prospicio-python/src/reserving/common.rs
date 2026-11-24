//! Shared glue: the `NotSupportedError` exception, error mapping, option names and their parsers, and the array-likes read from Python.

use super::*;

pyo3::create_exception!(
    prospicio,
    NotSupportedError,
    PyValueError,
    "A setting, or a combination of settings, that the method it was given \
     to does not support, raised when the setting is chosen where it can be \
     (the Rust error `NotSupported`, `reserving-v02.md` decision 29). A \
     subclass of ``ValueError``, so code that catches ``ValueError`` still \
     catches it."
);

/// A reserving error as a Python exception: [`NotSupportedError`] for a
/// setting the method does not support
/// ([`prospicio_reserving::Error::is_not_supported`]), `ValueError`
/// otherwise.
pub(crate) fn err(e: prospicio_reserving::Error) -> PyErr {
    if e.is_not_supported() {
        NotSupportedError::new_err(e.to_string())
    } else {
        PyValueError::new_err(e.to_string())
    }
}

/// An origin as Python names it: a key tuple, one label (a string or an
/// integer), or ``None`` for the total.
#[derive(FromPyObject)]
pub(crate) enum OriginArg {
    One(KeyArg),
    Key(Vec<KeyArg>),
}

impl OriginArg {
    fn key(self) -> ComponentKey {
        match self {
            OriginArg::One(v) => key_from_py(vec![v]),
            OriginArg::Key(k) => key_from_py(k),
        }
    }
}

/// [`UltimateDraws::ultimates`] for Python.
pub(crate) fn ultimates_of(fit: &impl UltimateDraws) -> PyResult<PyPredictiveDistribution> {
    Ok(PyPredictiveDistribution {
        inner: fit.ultimates().map_err(err)?,
    })
}

/// One origin's ultimate draws, or the total's with no origin.
pub(crate) fn ultimate_draws_of(
    fit: &impl UltimateDraws,
    origin: Option<OriginArg>,
) -> PyResult<Vec<f64>> {
    match origin {
        None => fit.total_ultimate_draws(),
        Some(o) => fit.ultimate_draws(&o.key()),
    }
    .map_err(err)
}

/// One origin's loss ratio draws, or the total's with no origin.
pub(crate) fn loss_ratio_draws_of(
    fit: &impl UltimateDraws,
    origin: Option<OriginArg>,
    premium: f64,
) -> PyResult<Vec<f64>> {
    match origin {
        None => fit.total_loss_ratio_draws(premium),
        Some(o) => fit.loss_ratio_draws(&o.key(), premium),
    }
    .map_err(err)
}

pub(crate) fn grain(name: &str) -> PyResult<Grain> {
    Grain::from_name(&name.trim().to_ascii_lowercase())
        .ok_or_else(|| crate::unknown_name("grain", name, Grain::NAMES))
}

pub(crate) fn average(name: &str) -> PyResult<Average> {
    Average::from_name(name).ok_or_else(|| crate::unknown_name("average", name, Average::NAMES))
}

pub(crate) fn average_name(a: Average) -> &'static str {
    a.name()
}

pub(crate) fn sigma_interpolation(name: &str) -> PyResult<SigmaInterpolation> {
    SigmaInterpolation::from_name(name)
        .ok_or_else(|| crate::unknown_name("sigma_interpolation", name, SigmaInterpolation::NAMES))
}

pub(crate) fn process_distribution(name: &str) -> PyResult<ProcessDistribution> {
    match name {
        "gamma" => Ok(ProcessDistribution::Gamma),
        "none" => Ok(ProcessDistribution::None),
        _ => Err(PyValueError::new_err(format!(
            "process must be \"gamma\" or \"none\", got {name:?}"
        ))),
    }
}

pub(crate) fn process_name(p: ProcessDistribution) -> &'static str {
    match p {
        ProcessDistribution::Gamma => "gamma",
        ProcessDistribution::None => "none",
    }
}

/// The dependence between segments named `name`, with the Spearman matrix
/// that `"rank_correlation"` needs and the others refuse.
pub(crate) fn segment_dependence(
    name: &str,
    spearman: Option<Vec<Vec<f64>>>,
) -> PyResult<SegmentDependence> {
    match (name, spearman) {
        ("independent", None) => Ok(SegmentDependence::Independent),
        ("synchronized", None) => Ok(SegmentDependence::Synchronized),
        ("rank_correlation", Some(rows)) => {
            if rows.iter().any(|r| r.len() != rows.len()) {
                return Err(PyValueError::new_err(
                    "spearman must be a square matrix, one row and column per segment",
                ));
            }
            Ok(SegmentDependence::RankCorrelation {
                spearman: rows.into_iter().flatten().collect(),
            })
        }
        ("rank_correlation", None) => Err(PyValueError::new_err(
            "dependence=\"rank_correlation\" needs a spearman matrix",
        )),
        ("independent" | "synchronized", Some(_)) => Err(PyValueError::new_err(
            "spearman is taken only with dependence=\"rank_correlation\"",
        )),
        _ => Err(PyValueError::new_err(format!(
            "dependence must be \"independent\", \"synchronized\" or \"rank_correlation\", \
             got {name:?}"
        ))),
    }
}

pub(crate) fn dependence_name(d: &SegmentDependence) -> &'static str {
    match d {
        SegmentDependence::Independent => "independent",
        SegmentDependence::Synchronized => "synchronized",
        SegmentDependence::RankCorrelation { .. } => "rank_correlation",
    }
}

/// The Spearman matrix of a rank correlation, as rows; `None` otherwise.
pub(crate) fn dependence_spearman(d: &SegmentDependence) -> Option<Vec<Vec<f64>>> {
    match d {
        SegmentDependence::RankCorrelation { spearman } => {
            let n = (spearman.len() as f64).sqrt().round() as usize;
            Some(spearman.chunks(n.max(1)).map(<[f64]>::to_vec).collect())
        }
        _ => None,
    }
}

/// `, dependence=...` for a bootstrap's repr; empty when independent.
pub(crate) fn dependence_repr(d: &SegmentDependence) -> String {
    match dependence_spearman(d) {
        None if *d == SegmentDependence::Independent => String::new(),
        None => format!(", dependence={:?}", dependence_name(d)),
        Some(m) => format!(", dependence={:?}, spearman={m:?}", dependence_name(d)),
    }
}

pub(crate) fn sigma_interpolation_name(s: SigmaInterpolation) -> &'static str {
    s.name()
}

pub(crate) fn development(
    average_: &str,
    sigma_interpolation_: &str,
    pattern: Option<SelectedPattern>,
    n_periods: Option<usize>,
    drop_high: usize,
    drop_low: usize,
) -> PyResult<Development> {
    Ok(Development {
        average: average(average_)?,
        sigma_interpolation: sigma_interpolation(sigma_interpolation_)?,
        pattern,
        n_periods,
        drop_high,
        drop_low,
    })
}

fn pattern_kind(name: &str) -> PyResult<PatternKind> {
    PatternKind::from_name(name)
        .ok_or_else(|| crate::unknown_name("ldf_kind", name, PatternKind::NAMES))
}

pub(crate) fn pattern_kind_name(kind: PatternKind) -> &'static str {
    kind.name()
}

pub(crate) fn py_repr(obj: &Bound<'_, PyAny>) -> String {
    obj.repr().map_or_else(|_| "?".into(), |r| r.to_string())
}

/// `(age, value)` pairs of a dict (or any mapping, a pandas Series too)
/// from age in months to a number.
pub(crate) fn age_values(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<(Lag, f64)>> {
    if obj.is_instance_of::<PyString>() || !obj.hasattr("items")? {
        return Err(PyTypeError::new_err(format!(
            "{what} must be a dict from age in months to a number, got {}",
            py_repr(obj)
        )));
    }
    let mut out = Vec::new();
    for item in obj.call_method0("items")?.try_iter()? {
        let (key, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
        let age: i64 = if key.is_instance_of::<PyBool>() {
            None
        } else {
            key.extract().ok()
        }
        .ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{what}: an age must be a whole number of months, got {}",
                py_repr(&key)
            ))
        })?;
        let age = Lag::try_from(age)
            .map_err(|_| PyValueError::new_err(format!("{what}: {age} is not an age in months")))?;
        let value: f64 = if value.is_instance_of::<PyBool>() {
            None
        } else {
            value.extract().ok()
        }
        .ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{what}: the value at age {age} must be a number, got {}",
                py_repr(&value)
            ))
        })?;
        out.push((age, value));
    }
    Ok(out)
}

/// The selected pattern of the ``ldf``, ``ldf_kind`` and ``ldf_weight``
/// arguments: ``None`` without ``ldf``. The kind is required: age-to-age
/// factors and factors to ultimate cannot be told apart.
pub(crate) fn selected_pattern(
    ldf: Option<&Bound<'_, PyAny>>,
    ldf_kind: Option<&str>,
    ldf_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<SelectedPattern>> {
    let ldf_weight = ldf_weight.filter(|o| !o.is_none());
    let Some(ldf) = ldf.filter(|o| !o.is_none()) else {
        if ldf_kind.is_some() {
            return Err(PyValueError::new_err("ldf_kind is given without ldf"));
        }
        if ldf_weight.is_some() {
            return Err(PyValueError::new_err("ldf_weight is given without ldf"));
        }
        return Ok(None);
    };
    let Some(kind) = ldf_kind else {
        return Err(PyValueError::new_err(
            "ldf needs ldf_kind: \"ldf\" (age-to-age factors), \"cdf\" (factors to \
             ultimate) or \"percent_developed\"; age-to-age factors and factors to \
             ultimate cannot be told apart",
        ));
    };
    let pattern = SelectedPattern::new(pattern_kind(kind)?, age_values(ldf, "ldf")?);
    let weight = match ldf_weight {
        None => PatternWeight::default(),
        Some(w) if w.hasattr("items")? => PatternWeight::ByAge(age_values(w, "ldf_weight")?),
        Some(w) if !w.is_instance_of::<PyBool>() => {
            PatternWeight::All(w.extract().map_err(|_| {
                PyTypeError::new_err(format!(
                    "ldf_weight must be a number or a dict from age in months to a number, got {}",
                    py_repr(w)
                ))
            })?)
        }
        Some(w) => {
            return Err(PyTypeError::new_err(format!(
                "ldf_weight must be a number or a dict from age in months to a number, got {}",
                py_repr(w)
            )));
        }
    };
    Ok(Some(pattern.with_weight(weight)))
}

/// A dict from age in months to value.
pub(crate) fn age_dict<'py>(py: Python<'py>, pairs: &[(Lag, f64)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for &(age, v) in pairs {
        d.set_item(age, v)?;
    }
    Ok(d)
}

/// A method's ``ldf`` as Python sees it: a dict from age to value, or
/// ``None`` without a pattern.
pub(crate) fn pattern_values<'py>(
    py: Python<'py>,
    pattern: Option<&SelectedPattern>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    pattern.map(|p| age_dict(py, &p.values)).transpose()
}

/// A method's ``ldf_kind``, ``None`` without a pattern.
pub(crate) fn pattern_kind_of(pattern: Option<&SelectedPattern>) -> Option<&'static str> {
    pattern.map(|p| pattern_kind_name(p.kind))
}

/// A method's ``ldf_weight``: a float, a dict from age to weight, or
/// ``None`` without a pattern.
pub(crate) fn pattern_weight<'py>(
    py: Python<'py>,
    pattern: Option<&SelectedPattern>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    pattern
        .map(|p| match &p.weight {
            PatternWeight::All(w) => Ok(PyFloat::new(py, *w).into_any()),
            PatternWeight::ByAge(by_age) => Ok(age_dict(py, by_age)?.into_any()),
        })
        .transpose()
}

/// `, n_periods=..., drop_high=..., drop_low=...` for the exclusions that are
/// set, then the selected pattern; empty at the defaults.
pub(crate) fn development_arg_repr(d: &Development) -> String {
    let mut out = exclusions_arg_repr(d.n_periods, d.drop_high, d.drop_low);
    out.push_str(&pattern_arg_repr(d.pattern.as_ref()));
    out
}

/// The exclusions that are set, as `, n_periods=..., drop_high=...,
/// drop_low=...`; empty at the defaults.
pub(crate) fn exclusions_arg_repr(
    n_periods: Option<usize>,
    drop_high: usize,
    drop_low: usize,
) -> String {
    let mut out = String::new();
    if let Some(n) = n_periods {
        out.push_str(&format!(", n_periods={n}"));
    }
    if drop_high != 0 {
        out.push_str(&format!(", drop_high={drop_high}"));
    }
    if drop_low != 0 {
        out.push_str(&format!(", drop_low={drop_low}"));
    }
    out
}

/// `, ldf={...}, ldf_kind=..., ldf_weight=...` for a method's repr; empty
/// without a pattern, and no weight when it is the default.
pub(crate) fn pattern_arg_repr(pattern: Option<&SelectedPattern>) -> String {
    let Some(p) = pattern else {
        return String::new();
    };
    let map = |pairs: &[(Lag, f64)]| {
        let items: Vec<String> = pairs.iter().map(|(a, v)| format!("{a}: {v:?}")).collect();
        format!("{{{}}}", items.join(", "))
    };
    let weight = match &p.weight {
        PatternWeight::All(w) if *w == 1.0 => String::new(),
        PatternWeight::All(w) => format!(", ldf_weight={w:?}"),
        PatternWeight::ByAge(by_age) => format!(", ldf_weight={}", map(by_age)),
    };
    format!(
        ", ldf={}, ldf_kind={:?}{weight}",
        map(&p.values),
        pattern_kind_name(p.kind)
    )
}

/// A column as a plain Python sequence: numpy arrays and pandas or Polars
/// series become lists.
pub(crate) fn plain<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if obj.hasattr("tolist")? {
        obj.call_method0("tolist")
    } else {
        Ok(obj.clone())
    }
}

pub(crate) fn month_of(item: &Bound<'_, PyAny>, what: &str) -> PyResult<Month> {
    // Python ints and integer scalars such as `numpy.int64` (via `__index__`).
    if !item.is_instance_of::<PyBool>()
        && (item.is_instance_of::<PyInt>() || item.hasattr("__index__")?)
    {
        return Ok(Month::january(item.extract()?));
    }
    if item.hasattr("year")? && item.hasattr("month")? {
        let year: i32 = item.getattr("year")?.extract()?;
        let month: u8 = item.getattr("month")?.extract()?;
        return Month::new(year, month).map_err(to_py);
    }
    Err(PyTypeError::new_err(format!(
        "{what} must hold dates or integer years, got {}",
        item.repr()?
    )))
}

/// Months of a column of dates (numpy `datetime64`, `datetime.date`,
/// pandas `Timestamp`) or integer years.
pub(crate) fn months(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<Month>> {
    let py = obj.py();
    if obj.hasattr("dtype")? {
        // A numpy array or a pandas/Polars series: convert datetimes in bulk.
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (obj,))?;
        let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
        if kind == "M" {
            let ordinals: Vec<i64> = arr
                .call_method1("astype", ("datetime64[M]",))?
                .call_method1("astype", ("int64",))?
                .call_method0("tolist")?
                .extract()?;
            return ordinals
                .into_iter()
                .map(|o| {
                    if o == i64::MIN {
                        Err(PyValueError::new_err(format!("{what} has a missing date")))
                    } else {
                        Ok(Month::january(1970).add_months(o))
                    }
                })
                .collect();
        }
        return arr
            .call_method0("tolist")?
            .try_iter()?
            .map(|item| month_of(&item?, what))
            .collect();
    }
    obj.try_iter()?.map(|item| month_of(&item?, what)).collect()
}

pub(crate) fn ages(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Lag>> {
    let raw: Vec<i64> = plain(obj)?
        .extract()
        .map_err(|_| PyTypeError::new_err("development must hold integer ages in months"))?;
    raw.into_iter()
        .map(|a| {
            Lag::try_from(a).map_err(|_| {
                PyValueError::new_err(format!("development age {a} is not a positive month count"))
            })
        })
        .collect()
}

pub(crate) fn floats(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    plain(obj)?
        .extract()
        .map_err(|_| PyTypeError::new_err(format!("column {name:?} must hold numbers")))
}

/// Values of the key column `name` as strings (`str()` of each value).
/// `None`, float NaN and pandas' `NA` and `NaT` are missing values, which a
/// key may not have.
pub(crate) fn key_values(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<String>> {
    let missing_error = || PyValueError::new_err(format!("key column {name:?} has missing values"));
    // A pandas series knows its own missing values, whatever its dtype.
    if obj.hasattr("isna")? && obj.call_method0("isna")?.call_method0("any")?.is_truthy()? {
        return Err(missing_error());
    }
    plain(obj)?
        .try_iter()?
        .map(|item| {
            let item = item?;
            let missing = item.is_none()
                || (item.is_instance_of::<PyFloat>() && item.extract::<f64>()?.is_nan())
                || matches!(
                    item.get_type().name()?.extract::<String>()?.as_str(),
                    "NAType" | "NaTType"
                );
            if missing {
                return Err(missing_error());
            }
            Ok(item.str()?.to_string())
        })
        .collect()
}

/// A label as Python sees it: ``"Total"`` without keys, a str with one key,
/// a tuple with several.
pub(crate) fn label_to_py<'py>(
    py: Python<'py>,
    n_keys: usize,
    label: &Label,
) -> PyResult<Bound<'py, PyAny>> {
    match (n_keys, label.parts()) {
        (0, _) => Ok(PyString::new(py, &label.to_string()).into_any()),
        (1, [one]) => Ok(PyString::new(py, one).into_any()),
        (_, parts) => Ok(PyTuple::new(py, parts)?.into_any()),
    }
}

/// Key values to select: one value (a string, number, date, ...) or any
/// other iterable of values (a list, tuple, set, NumPy array or pandas
/// Series); each compared as ``str()`` of it, as key columns are stored.
pub(crate) fn selection_values(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyBytes>() {
        return Ok(vec![obj.str()?.to_string()]);
    }
    match obj.try_iter() {
        Ok(items) => items.map(|item| Ok(item?.str()?.to_string())).collect(),
        Err(_) => Ok(vec![obj.str()?.to_string()]),
    }
}

/// The `FileLayout` of `Triangle.read_csv` and `read_excel`: long when
/// `development` is given, wide otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn file_layout(
    origin: &str,
    development: Option<&str>,
    columns: Option<&Bound<'_, PyAny>>,
    keys: Option<&Bound<'_, PyAny>>,
    ages: Option<Vec<Lag>>,
    origin_grain: &str,
    development_grain: &str,
    cumulative: bool,
    development_is_valuation: bool,
) -> PyResult<FileLayout> {
    let columns = columns.map(names).transpose()?.unwrap_or_default();
    let layout = match development {
        Some(development) => {
            if ages.is_some() {
                return Err(PyValueError::new_err(
                    "ages name the columns of a wide layout; give development=None",
                ));
            }
            FileLayout::long(origin, development, columns)
        }
        None => {
            if development_is_valuation {
                return Err(PyValueError::new_err(
                    "development_is_valuation needs a long layout: give the development column",
                ));
            }
            let column = match columns.as_slice() {
                [] => "values".to_string(),
                [one] => one.clone(),
                _ => {
                    return Err(PyValueError::new_err(
                        "a wide layout has one measure: give columns as one name",
                    ));
                }
            };
            let layout = FileLayout::wide(origin, column);
            match ages {
                Some(ages) => layout.ages(ages),
                None => layout,
            }
        }
    };
    let keys = keys.map(names).transpose()?.unwrap_or_default();
    Ok(layout
        .keys(keys)
        .grains(grain(origin_grain)?, grain(development_grain)?)
        .cumulative(cumulative)
        .development_is_valuation(development_is_valuation))
}

pub(crate) fn names(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(name) = obj.extract::<String>() {
        Ok(vec![name])
    } else {
        obj.extract()
    }
}

/// Measure columns from a dict of name to column, or one unnamed column
/// (named ``"values"``).
pub(crate) fn value_columns(values: &Bound<'_, PyAny>) -> PyResult<Vec<(String, Vec<f64>)>> {
    if let Ok(dict) = values.cast::<PyDict>() {
        dict.iter()
            .map(|(k, v)| {
                let name: String = k
                    .extract()
                    .map_err(|_| PyTypeError::new_err("value column names must be strings"))?;
                let column = floats(&v, &name)?;
                Ok((name, column))
            })
            .collect()
    } else {
        Ok(vec![("values".to_string(), floats(values, "values")?)])
    }
}

pub(crate) struct LongArgs {
    origin: Vec<Month>,
    development: Vec<Lag>,
    valuations: Vec<Month>,
    development_is_valuation: bool,
    values: Vec<(String, Vec<f64>)>,
    keys: Vec<(String, Vec<String>)>,
    origin_grain: Grain,
    development_grain: Grain,
    cumulative: bool,
}

impl LongArgs {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        origin: &Bound<'_, PyAny>,
        development: &Bound<'_, PyAny>,
        development_is_valuation: bool,
        values: Vec<(String, Vec<f64>)>,
        keys: Vec<(String, Vec<String>)>,
        origin_grain: &str,
        development_grain: &str,
        cumulative: bool,
    ) -> PyResult<Self> {
        let (development, valuations) = if development_is_valuation {
            (Vec::new(), months(development, "development")?)
        } else {
            (ages(development)?, Vec::new())
        };
        Ok(Self {
            origin: months(origin, "origin")?,
            development,
            valuations,
            development_is_valuation,
            values,
            keys,
            origin_grain: grain(origin_grain)?,
            development_grain: grain(development_grain)?,
            cumulative,
        })
    }

    pub(crate) fn build(&self, py: Python<'_>) -> PyResult<PyTriangle> {
        let values: Vec<(&str, &[f64])> = self
            .values
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
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
        let long = Long {
            keys: &keys,
            origin: &self.origin,
            development: if self.development_is_valuation {
                DevelopmentColumn::Valuation(&self.valuations)
            } else {
                DevelopmentColumn::Age(&self.development)
            },
            values: &values,
            origin_grain: self.origin_grain,
            development_grain: self.development_grain,
            cumulative: self.cumulative,
        };
        let inner = py.detach(|| Triangle::from_long(&long)).map_err(err)?;
        Ok(PyTriangle { inner })
    }
}
