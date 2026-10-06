//! `actuarialrs.reserving` (Reserving lane): the loss triangle, the chain
//! ladder, Mack's model and the ODP bootstrap over `act_reserving`
//! (`docs/design/triangle.md`).
//!
//! Long tables come in as array-likes (lists, numpy arrays, pandas or
//! Polars columns) and go out as dicts of lists; numpy and pandas are used
//! when the caller passes them but are not required.

use act_core::{Grain, Lag, Month};
use act_reserving::{
    Average, ChainLadder, ChainLadderFit, ClaimsDevelopmentResult, Development, DevelopmentColumn,
    FitTable, Label, Long, Mack, MackFit, OdpBootstrap, OdpBootstrapFits, OdpBootstrapSegment,
    ProcessDistribution, ReserveFit, SegmentFits, SigmaInterpolation, Triangle, view,
};
use pyo3::exceptions::{PyImportError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDate, PyDelta, PyDict, PyFloat, PyInt, PyModule, PyString, PyTuple,
};

use crate::distributions::PyPredictiveDistribution;
use crate::to_py;

fn err(e: act_reserving::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn grain(name: &str) -> PyResult<Grain> {
    match name.to_ascii_uppercase().as_str() {
        "M" => Ok(Grain::Month),
        "Q" => Ok(Grain::Quarter),
        "S" => Ok(Grain::Semester),
        "Y" => Ok(Grain::Year),
        _ => Err(PyValueError::new_err(format!(
            "grain must be \"M\", \"Q\", \"S\" or \"Y\", got {name:?}"
        ))),
    }
}

fn average(name: &str) -> PyResult<Average> {
    match name {
        "volume" => Ok(Average::Volume),
        "simple" => Ok(Average::Simple),
        "regression" => Ok(Average::Regression),
        _ => Err(PyValueError::new_err(format!(
            "average must be \"volume\", \"simple\" or \"regression\", got {name:?}"
        ))),
    }
}

fn average_name(a: Average) -> &'static str {
    match a {
        Average::Volume => "volume",
        Average::Simple => "simple",
        Average::Regression => "regression",
    }
}

fn sigma_interpolation(name: &str) -> PyResult<SigmaInterpolation> {
    match name {
        "log-linear" => Ok(SigmaInterpolation::LogLinear),
        "mack" => Ok(SigmaInterpolation::Mack),
        _ => Err(PyValueError::new_err(format!(
            "sigma_interpolation must be \"log-linear\" or \"mack\", got {name:?}"
        ))),
    }
}

fn process_distribution(name: &str) -> PyResult<ProcessDistribution> {
    match name {
        "gamma" => Ok(ProcessDistribution::Gamma),
        "none" => Ok(ProcessDistribution::None),
        _ => Err(PyValueError::new_err(format!(
            "process must be \"gamma\" or \"none\", got {name:?}"
        ))),
    }
}

fn process_name(p: ProcessDistribution) -> &'static str {
    match p {
        ProcessDistribution::Gamma => "gamma",
        ProcessDistribution::None => "none",
    }
}

fn sigma_interpolation_name(s: SigmaInterpolation) -> &'static str {
    match s {
        SigmaInterpolation::LogLinear => "log-linear",
        SigmaInterpolation::Mack => "mack",
    }
}

fn development(average_: &str, sigma_interpolation_: &str) -> PyResult<Development> {
    Ok(Development {
        average: average(average_)?,
        sigma_interpolation: sigma_interpolation(sigma_interpolation_)?,
    })
}

/// A column as a plain Python sequence: numpy arrays and pandas or Polars
/// series become lists.
fn plain<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if obj.hasattr("tolist")? {
        obj.call_method0("tolist")
    } else {
        Ok(obj.clone())
    }
}

fn month_of(item: &Bound<'_, PyAny>, what: &str) -> PyResult<Month> {
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
fn months(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<Month>> {
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

fn ages(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Lag>> {
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

fn floats(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    plain(obj)?
        .extract()
        .map_err(|_| PyTypeError::new_err(format!("column {name:?} must hold numbers")))
}

/// Values of the key column `name` as strings (`str()` of each value).
/// `None`, float NaN and pandas' `NA` and `NaT` are missing values, which a
/// key may not have.
fn key_values(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<String>> {
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
fn label_to_py<'py>(py: Python<'py>, n_keys: usize, label: &Label) -> PyResult<Bound<'py, PyAny>> {
    match (n_keys, label.parts()) {
        (0, _) => Ok(PyString::new(py, &label.to_string()).into_any()),
        (1, [one]) => Ok(PyString::new(py, one).into_any()),
        (_, parts) => Ok(PyTuple::new(py, parts)?.into_any()),
    }
}

/// Key values to select: one value (a string, number, date, ...) or any
/// other iterable of values (a list, tuple, set, NumPy array or pandas
/// Series); each compared as ``str()`` of it, as key columns are stored.
fn selection_values(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyBytes>() {
        return Ok(vec![obj.str()?.to_string()]);
    }
    match obj.try_iter() {
        Ok(items) => items.map(|item| Ok(item?.str()?.to_string())).collect(),
        Err(_) => Ok(vec![obj.str()?.to_string()]),
    }
}

fn names(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(name) = obj.extract::<String>() {
        Ok(vec![name])
    } else {
        obj.extract()
    }
}

/// Measure columns from a dict of name to column, or one unnamed column
/// (named ``"values"``).
fn value_columns(values: &Bound<'_, PyAny>) -> PyResult<Vec<(String, Vec<f64>)>> {
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

struct LongArgs {
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
    fn new(
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

    fn build(&self, py: Python<'_>) -> PyResult<PyTriangle> {
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

/// A loss triangle with four axes: index (segment), column (measure), origin
/// and development age, in chainladder-python's order.
///
/// Segments are named by key columns such as ``"lob"`` and ``"state"``:
/// ``keys`` gives their names and ``index`` one label per segment. A
/// triangle without keys has one segment, ``"Total"``.
///
/// Build one from a long table with ``from_long`` or ``from_frame``. Ages
/// are whole months from the start of the origin period, so age 12 on a
/// 2021 accident year is valued at December 2021. Cells that were not
/// observed are ``nan`` in ``values``; an observed zero stays zero.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import Triangle
/// >>> tri = Triangle.from_long(
/// ...     origin=[2020, 2020, 2021],
/// ...     development=[12, 24, 12],
/// ...     values={"paid": [100.0, 150.0, 110.0]},
/// ... )
/// >>> tri.shape
/// (1, 1, 2, 2)
/// >>> tri.origins, tri.development, tri.valuation
/// (['2020', '2021'], [12, 24], datetime.date(2021, 12, 31))
/// >>> tri.values[0][0]
/// [[100.0, 150.0], [110.0, nan]]
#[pyclass(name = "Triangle", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyTriangle {
    inner: Triangle,
}

fn wrap(inner: Triangle) -> PyTriangle {
    PyTriangle { inner }
}

/// The last day of `month` as a ``datetime.date``.
fn month_end<'py>(py: Python<'py>, month: Month) -> PyResult<Bound<'py, PyDate>> {
    // The first of the next month, minus one day.
    let next = month.add_months(1);
    let first_of_next = PyDate::new(py, next.year(), next.month(), 1)?;
    let one_day = PyDelta::new(py, 1, 0, 0, false)?;
    first_of_next
        .call_method1("__sub__", (one_day,))?
        .cast_into::<PyDate>()
        .map_err(Into::into)
}

/// pandas when it is installed, so tables come out as DataFrames; the
/// tables are dicts of lists otherwise.
fn pandas(py: Python<'_>) -> PyResult<Option<Bound<'_, PyModule>>> {
    match py.import("pandas") {
        Ok(module) => Ok(Some(module)),
        Err(e) if e.is_instance_of::<PyImportError>(py) => Ok(None),
        Err(e) => Err(e),
    }
}

#[pymethods]
impl PyTriangle {
    /// Builds a triangle from the columns of a long table, one row per
    /// (keys, origin, development).
    ///
    /// Origins span every period from the earliest to the latest row and
    /// ages every development period from the youngest to the oldest. Rows
    /// with the same (keys, origin, age) are summed; ``nan`` values are
    /// missing. Incremental input treats a missing row as a period without
    /// movement, as chainladder-python does.
    ///
    /// Parameters
    /// ----------
    /// origin : array-like
    ///     Any date in each row's origin period (numpy ``datetime64``,
    ///     ``datetime.date``, pandas ``Timestamp``), or integer years.
    /// development : array-like
    ///     Development age of each row in months (12, 24, ...), or its
    ///     valuation date when ``development_is_valuation`` is true.
    /// values : dict of str to array-like, or array-like
    ///     Measure columns by name. A single array-like is one column named
    ///     ``"values"``.
    /// keys : dict of str to array-like, optional
    ///     Key columns by name, such as ``{"lob": [...], "state": [...]}``,
    ///     in key order. Values are stored as strings (``str()`` of each);
    ///     ``None`` and ``nan`` are not allowed. Each distinct combination is
    ///     a segment. By default every row is in one segment, ``"Total"``.
    /// origin_grain : {"Y", "S", "Q", "M"}, default "Y"
    ///     Length of an origin period.
    /// development_grain : {"Y", "S", "Q", "M"}, default "Y"
    ///     Spacing of development ages; must divide the origin grain.
    /// cumulative : bool, default True
    ///     Whether the values are cumulative (otherwise incremental).
    /// development_is_valuation : bool, default False
    ///     Whether ``development`` holds valuation dates instead of ages.
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If columns differ in length, an age is not on the development
    ///     grid, a value is infinite, the grains are incompatible, a key name
    ///     is repeated or also a value column, or a key has missing values.
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     origin=[2020, 2020, 2021, 2020],
    /// ...     development=[12, 24, 12, 12],
    /// ...     values={"paid": [100.0, 150.0, 110.0, 50.0]},
    /// ...     keys={"lob": ["Auto", "Auto", "Auto", "Home"], "state": ["CA", "CA", "CA", "NY"]},
    /// ... )
    /// >>> tri.keys, tri.index
    /// (['lob', 'state'], [('Auto', 'CA'), ('Home', 'NY')])
    ///
    /// Valuation dates instead of ages:
    ///
    /// >>> import datetime
    /// >>> from actuarialrs.reserving import Triangle
    /// >>> d = datetime.date
    /// >>> tri = Triangle.from_long(
    /// ...     origin=[d(2021, 2, 1), d(2021, 2, 1), d(2021, 5, 1)],
    /// ...     development=[d(2021, 3, 31), d(2021, 6, 30), d(2021, 6, 30)],
    /// ...     values=[10.0, 25.0, 7.0],
    /// ...     origin_grain="Q",
    /// ...     development_grain="Q",
    /// ...     development_is_valuation=True,
    /// ... )
    /// >>> tri.origins, tri.development
    /// (['2021Q1', '2021Q2'], [3, 6])
    #[staticmethod]
    #[pyo3(signature = (origin, development, values, keys = None, origin_grain = "Y", development_grain = "Y", cumulative = true, development_is_valuation = false))]
    #[allow(clippy::too_many_arguments)]
    fn from_long(
        py: Python<'_>,
        origin: &Bound<'_, PyAny>,
        development: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        keys: Option<&Bound<'_, PyAny>>,
        origin_grain: &str,
        development_grain: &str,
        cumulative: bool,
        development_is_valuation: bool,
    ) -> PyResult<Self> {
        let keys = match keys {
            None => Vec::new(),
            Some(keys) => {
                let dict = keys.cast::<PyDict>().map_err(|_| {
                    PyTypeError::new_err("keys must be a dict of key name to column")
                })?;
                dict.iter()
                    .map(|(k, v)| {
                        let name: String = k
                            .extract()
                            .map_err(|_| PyTypeError::new_err("key names must be strings"))?;
                        let column = key_values(&v, &name)?;
                        Ok((name, column))
                    })
                    .collect::<PyResult<_>>()?
            }
        };
        LongArgs::new(
            origin,
            development,
            development_is_valuation,
            value_columns(values)?,
            keys,
            origin_grain,
            development_grain,
            cumulative,
        )?
        .build(py)
    }

    /// Builds a triangle from a data frame in long format.
    ///
    /// Columns are looked up with ``data[name]``, so a pandas or Polars
    /// DataFrame works, as does a dict of columns.
    ///
    /// Parameters
    /// ----------
    /// data : DataFrame or dict
    /// origin : str
    ///     Name of the origin column (dates or integer years).
    /// development : str
    ///     Name of the development column (ages in months, or valuation
    ///     dates when ``development_is_valuation`` is true).
    /// columns : str or list of str
    ///     Names of the measure columns.
    /// keys : str or list of str, optional
    ///     Names of the key columns, such as ``["lob", "state"]``. By
    ///     default every row is in one segment, ``"Total"``.
    /// origin_grain : {"Y", "S", "Q", "M"}, default "Y"
    /// development_grain : {"Y", "S", "Q", "M"}, default "Y"
    /// cumulative : bool, default True
    /// development_is_valuation : bool, default False
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As for ``from_long``.
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reserving import Triangle
    /// >>> df = {
    /// ...     "lob": ["Auto", "Auto", "Auto", "Home"],
    /// ...     "year": [2020, 2020, 2021, 2020],
    /// ...     "age": [12, 24, 12, 12],
    /// ...     "paid": [100.0, 150.0, 110.0, 50.0],
    /// ... }
    /// >>> tri = Triangle.from_frame(df, "year", "age", "paid", keys="lob")
    /// >>> tri.keys, tri.index, tri.shape
    /// (['lob'], ['Auto', 'Home'], (2, 1, 2, 2))
    #[staticmethod]
    #[pyo3(signature = (data, origin, development, columns, keys = None, origin_grain = "Y", development_grain = "Y", cumulative = true, development_is_valuation = false))]
    #[allow(clippy::too_many_arguments)]
    fn from_frame(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        origin: &str,
        development: &str,
        columns: &Bound<'_, PyAny>,
        keys: Option<&Bound<'_, PyAny>>,
        origin_grain: &str,
        development_grain: &str,
        cumulative: bool,
        development_is_valuation: bool,
    ) -> PyResult<Self> {
        let values = names(columns)?
            .into_iter()
            .map(|name| {
                let column = floats(&data.get_item(&name)?, &name)?;
                Ok((name, column))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let keys = match keys {
            None => Vec::new(),
            Some(keys) => names(keys)?
                .into_iter()
                .map(|name| {
                    let column = key_values(&data.get_item(&name)?, &name)?;
                    Ok((name, column))
                })
                .collect::<PyResult<_>>()?,
        };
        LongArgs::new(
            &data.get_item(origin)?,
            &data.get_item(development)?,
            development_is_valuation,
            values,
            keys,
            origin_grain,
            development_grain,
            cumulative,
        )?
        .build(py)
    }

    /// Axis lengths: ``(index, column, origin, development)``.
    #[getter]
    fn shape(&self) -> (usize, usize, usize, usize) {
        let [i, c, o, d] = self.inner.shape();
        (i, c, o, d)
    }

    /// Names of the key columns, in key order; empty without keys.
    #[getter]
    fn keys(&self) -> Vec<String> {
        self.inner.key_names().to_vec()
    }

    /// Segment labels: a str each with one key, a tuple of key values with
    /// several, and ``["Total"]`` without keys.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let n_keys = self.inner.key_names().len();
        self.inner
            .index()
            .iter()
            .map(|l| label_to_py(py, n_keys, l))
            .collect()
    }

    /// Measure column names.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.columns().to_vec()
    }

    /// Origin periods, oldest first: ``"2021"``, ``"2021H1"``,
    /// ``"2021Q3"`` or ``"2021-07"`` by grain.
    #[getter]
    fn origins(&self) -> Vec<String> {
        self.inner.origins().iter().map(|p| p.to_string()).collect()
    }

    /// Origin grain: ``"Y"``, ``"S"``, ``"Q"`` or ``"M"``.
    #[getter]
    fn origin_grain(&self) -> String {
        self.inner.origin_grain().to_string()
    }

    /// Development ages in months, youngest first.
    #[getter]
    fn development(&self) -> Vec<Lag> {
        self.inner.development().to_vec()
    }

    /// Development grain: ``"Y"``, ``"S"``, ``"Q"`` or ``"M"``.
    #[getter]
    fn development_grain(&self) -> String {
        self.inner.development_grain().to_string()
    }

    /// Valuation date of the latest diagonal: the last day of its month,
    /// as a ``datetime.date``.
    #[getter]
    fn valuation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDate>> {
        month_end(py, self.inner.valuation())
    }

    /// Whether the values are cumulative (otherwise incremental).
    #[getter]
    fn is_cumulative(&self) -> bool {
        self.inner.is_cumulative()
    }

    /// Values as nested lists indexed ``[index][column][origin][development]``,
    /// ``nan`` where unobserved. ``numpy.asarray`` gives the 4-D array.
    #[getter]
    fn values(&self) -> Vec<Vec<Vec<Vec<f64>>>> {
        let [ni, nc, no, nd] = self.inner.shape();
        (0..ni)
            .map(|i| {
                (0..nc)
                    .map(|c| {
                        (0..no)
                            .map(|o| {
                                (0..nd)
                                    .map(|d| self.inner.get(i, c, o, d).unwrap_or(f64::NAN))
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    /// The triangle as a long table: a dict of equal-length lists with one
    /// entry per key column (by name), ``"origin"`` (start of the origin
    /// period, a ``datetime.date``), ``"development"`` (age in months) and
    /// one per measure column, with a row per (segment, origin, age) that
    /// has an observed measure. It feeds back into ``from_frame`` (with
    /// ``keys=tri.keys``) or ``pandas.DataFrame``.
    ///
    /// Returns
    /// -------
    /// dict of str to list
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a key or measure column is named ``origin`` or
    ///     ``development``.
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020], [12, 12], {"paid": [1.0, 2.0]}, keys={"lob": ["Auto", "Home"]}
    /// ... )
    /// >>> long = tri.to_long()
    /// >>> list(long), long["lob"]
    /// (['lob', 'origin', 'development', 'paid'], ['Auto', 'Home'])
    fn to_long<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if let Some(c) = self
            .inner
            .key_names()
            .iter()
            .chain(self.inner.columns())
            .find(|c| ["origin", "development"].contains(&c.as_str()))
        {
            return Err(PyValueError::new_err(format!(
                "column {c:?} clashes with the {c:?} column of the long table"
            )));
        }
        let long = self.inner.to_long();
        let date = py.import("datetime")?.getattr("date")?;
        let out = PyDict::new(py);
        let origin = long
            .origin
            .iter()
            .map(|m| date.call1((m.year(), m.month(), 1)))
            .collect::<PyResult<Vec<_>>>()?;
        for (name, values) in long.keys {
            out.set_item(name, values)?;
        }
        out.set_item("origin", origin)?;
        out.set_item("development", long.development)?;
        for (name, values) in long.values {
            out.set_item(name, values)?;
        }
        Ok(out)
    }

    /// The long table of ``to_long`` as a pandas DataFrame. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let long = self.to_long(py)?;
        py.import("pandas")?.call_method1("DataFrame", (long,))
    }

    /// Incremental values: each observed value minus the previous observed
    /// value in its row.
    ///
    /// Returns
    /// -------
    /// Triangle
    fn to_incremental(&self) -> Self {
        wrap(self.inner.to_incremental())
    }

    /// Cumulative values: running sums of the observed increments.
    ///
    /// Returns
    /// -------
    /// Triangle
    fn to_cumulative(&self) -> Self {
        wrap(self.inner.to_cumulative())
    }

    /// The latest observed value of each origin, as nested lists indexed
    /// ``[index][column][origin]``, ``nan`` for an origin with no value.
    ///
    /// Returns
    /// -------
    /// list of list of list of float
    fn latest_diagonal(&self) -> Vec<Vec<Vec<f64>>> {
        let diagonal = self.inner.latest_diagonal();
        let [ni, nc, no] = diagonal.shape();
        (0..ni)
            .map(|i| {
                (0..nc)
                    .map(|c| {
                        (0..no)
                            .map(|o| diagonal.get(i, c, o).map_or(f64::NAN, |(_, v)| v))
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    /// Age-to-age link ratios of the cumulative values. Development
    /// position ``d`` holds the ratio from age ``d`` to age ``d + 1``,
    /// observed where both ages are observed and the earlier value is not
    /// zero.
    ///
    /// Returns
    /// -------
    /// Triangle
    fn link_ratios(&self) -> Self {
        wrap(self.inner.link_ratios())
    }

    /// The segments whose key values match, and the measure columns named.
    ///
    /// Each keyword names a key and gives one value or a list of values to
    /// keep; segments must match every keyword, and keep their order.
    /// Values are compared as strings, as keys are stored (``str()`` of a
    /// value). A key named ``columns`` cannot be selected this way.
    ///
    /// Parameters
    /// ----------
    /// columns : str or list of str, optional
    ///     Measure columns to keep, in this order. By default every column.
    /// **keys : value or list of values
    ///     For example ``lob="Auto"`` or ``state=["CA", "NY"]``; any
    ///     iterable that is not a string (a tuple, set, NumPy array or
    ///     pandas Series) is a list of values.
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a key, value or column is unknown or given twice, a list of
    ///     values is empty, or no segment matches.
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020, 2020],
    /// ...     [12, 12, 12],
    /// ...     {"paid": [1.0, 2.0, 3.0], "incurred": [2.0, 3.0, 4.0]},
    /// ...     keys={"lob": ["Auto", "Auto", "Home"], "state": ["CA", "NY", "NY"]},
    /// ... )
    /// >>> tri.select(state="NY").index
    /// [('Auto', 'NY'), ('Home', 'NY')]
    /// >>> tri.select(lob="Auto", state=["CA", "NY"], columns="paid").shape
    /// (2, 1, 1, 1)
    #[pyo3(signature = (columns = None, **keys))]
    fn select(
        &self,
        columns: Option<&Bound<'_, PyAny>>,
        keys: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let conditions: Vec<(String, Vec<String>)> = match keys {
            None => Vec::new(),
            Some(keys) => keys
                .iter()
                .map(|(k, v)| Ok((k.extract::<String>()?, selection_values(&v)?)))
                .collect::<PyResult<_>>()?,
        };
        let values: Vec<Vec<&str>> = conditions
            .iter()
            .map(|(_, v)| v.iter().map(String::as_str).collect())
            .collect();
        let conditions: Vec<(&str, &[&str])> = conditions
            .iter()
            .zip(&values)
            .map(|((k, _), v)| (k.as_str(), v.as_slice()))
            .collect();
        let mut out = self.inner.select(&conditions).map_err(err)?;
        if let Some(columns) = columns {
            let columns = names(columns)?;
            let columns: Vec<&str> = columns.iter().map(String::as_str).collect();
            out = out.select_columns(&columns).map_err(err)?;
        }
        Ok(wrap(out))
    }

    /// Sums the segments that share the values of ``keys``, dropping the
    /// other keys.
    ///
    /// Cumulative values are summed cell by cell, and a cell is observed if
    /// any segment in the group observes it. An incremental triangle is
    /// summed as cumulative values and returned incremental.
    ///
    /// Parameters
    /// ----------
    /// keys : str or list of str
    ///     The keys to keep, in the order the result has them. ``[]`` sums
    ///     every segment into one, labelled ``"Total"``.
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a key is unknown or named twice.
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020, 2020],
    /// ...     [12, 12, 12],
    /// ...     {"paid": [1.0, 2.0, 3.0]},
    /// ...     keys={"lob": ["Auto", "Auto", "Home"], "state": ["CA", "NY", "NY"]},
    /// ... )
    /// >>> by_lob = tri.group_by("lob")
    /// >>> by_lob.index, by_lob.to_long()["paid"]
    /// (['Auto', 'Home'], [3.0, 3.0])
    /// >>> tri.group_by([]).to_long()["paid"]
    /// [6.0]
    fn group_by(&self, keys: &Bound<'_, PyAny>) -> PyResult<Self> {
        let keys = names(keys)?;
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        self.inner.group_by(&keys).map(wrap).map_err(err)
    }

    /// The triangle at a coarser origin and/or development grain.
    ///
    /// Parameters
    /// ----------
    /// origin_grain : {"Y", "S", "Q", "M"}
    /// development_grain : {"Y", "S", "Q", "M"}, optional
    ///     By default ``origin_grain``.
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a grain is finer than the current one, or the development
    ///     grain does not divide the origin grain.
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reserving import Triangle
    /// >>> q = Triangle.from_long(
    /// ...     [2020, 2020], [3, 6], [1.0, 2.0], origin_grain="Q", development_grain="Q"
    /// ... )
    /// >>> y = q.grain("Y")
    /// >>> y.origins, y.development_grain
    /// (['2020'], 'Y')
    #[pyo3(signature = (origin_grain, development_grain = None))]
    fn grain(&self, origin_grain: &str, development_grain: Option<&str>) -> PyResult<Self> {
        let origin = grain(origin_grain)?;
        let dev = match development_grain {
            Some(g) => grain(g)?,
            None => origin,
        };
        self.inner.grain(origin, dev).map(wrap).map_err(err)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyTriangle>>()
            .is_ok_and(|o| o.inner == self.inner)
    }

    /// One segment and measure as an origin × development table.
    ///
    /// Parameters
    /// ----------
    /// column : str, optional
    ///     The measure; may be left out when the triangle has one column.
    /// **keys : value
    ///     One value per key, such as ``lob="Auto"``, compared as ``str()``
    ///     of it. Keys not named may take any value, but the choice must
    ///     leave one segment; a triangle with one segment needs none. A key
    ///     named ``column`` cannot be chosen this way (``select`` it first).
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame or dict
    ///     With pandas installed, a DataFrame with the origin labels as its
    ///     index (named ``origin``), the ages in months as its columns
    ///     (named ``development``) and ``nan`` where a cell is not observed.
    ///     Without pandas, a dict of lists as the other tables of this
    ///     module: ``"origin"``, then one list per age keyed by the age.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a key, value or column is unknown, the keys match several
    ///     segments, or the column is left out and there are several.
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020, 2021, 2020],
    /// ...     [12, 24, 12, 12],
    /// ...     {"paid": [100.0, 150.0, 110.0, 50.0]},
    /// ...     keys={"lob": ["Auto", "Auto", "Auto", "Home"]},
    /// ... )
    /// >>> v = tri.view(lob="Auto")
    /// >>> list(v.index), list(v.columns), float(v.loc["2021", 12])
    /// (['2020', '2021'], [12, 24], 110.0)
    #[pyo3(signature = (column = None, **keys))]
    fn view<'py>(
        &self,
        py: Python<'py>,
        column: Option<&str>,
        keys: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let keys = segment_keys(keys)?;
        let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let view = self.inner.view(&keys, column).map_err(err)?;
        let origins: Vec<String> = view.origins.iter().map(ToString::to_string).collect();
        let (no, nd) = (view.origins.len(), view.development.len());
        let cell = |o: usize, d: usize| view.get(o, d).unwrap_or(f64::NAN);
        match pandas(py)? {
            Some(pd) => {
                let rows: Vec<Vec<f64>> = (0..no)
                    .map(|o| (0..nd).map(|d| cell(o, d)).collect())
                    .collect();
                let index_args = PyDict::new(py);
                index_args.set_item("name", "origin")?;
                let index = pd.call_method("Index", (origins,), Some(&index_args))?;
                let columns_args = PyDict::new(py);
                columns_args.set_item("name", "development")?;
                let columns =
                    pd.call_method("Index", (view.development.clone(),), Some(&columns_args))?;
                let args = PyDict::new(py);
                args.set_item("index", index)?;
                args.set_item("columns", columns)?;
                pd.call_method("DataFrame", (rows,), Some(&args))
            }
            None => {
                let out = PyDict::new(py);
                out.set_item("origin", origins)?;
                for (d, age) in view.development.iter().enumerate() {
                    let values: Vec<f64> = (0..no).map(|o| cell(o, d)).collect();
                    out.set_item(age, values)?;
                }
                Ok(out.into_any())
            }
        }
    }

    /// One row per segment and measure: the key values, ``column``,
    /// ``n_origins`` (origins with an observed value), ``first_origin`` and
    /// ``last_origin`` of those, ``valuation`` (the last day of the latest
    /// valuation with an observed value), ``latest`` (the sum over origins of
    /// the latest cumulative value, so for an incremental triangle the sum
    /// of every increment) and ``cumulative``. An origin or valuation is
    /// missing (``None``, which pandas may show as ``NaN``) when the segment
    /// has no observed value of the measure.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame or dict
    ///     A DataFrame with pandas installed, a dict of lists otherwise.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a key has the name of one of the summary's columns.
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020, 2021, 2020],
    /// ...     [12, 24, 12, 12],
    /// ...     {"paid": [100.0, 150.0, 110.0, 50.0]},
    /// ...     keys={"lob": ["Auto", "Auto", "Auto", "Home"]},
    /// ... )
    /// >>> s = tri.summary()
    /// >>> s["lob"].tolist(), s["n_origins"].tolist(), s["latest"].tolist()
    /// (['Auto', 'Home'], [2, 1], [260.0, 50.0])
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        const COLUMNS: [&str; 7] = [
            "column",
            "n_origins",
            "first_origin",
            "last_origin",
            "valuation",
            "latest",
            "cumulative",
        ];
        if let Some(k) = self
            .inner
            .key_names()
            .iter()
            .find(|k| COLUMNS.contains(&k.as_str()))
        {
            return Err(PyValueError::new_err(format!(
                "key {k:?} clashes with the {k:?} column of the summary"
            )));
        }
        let summary = self.inner.summary();
        let rows = &summary.rows;
        let out = PyDict::new(py);
        for (k, name) in summary.key_names.iter().enumerate() {
            let values: Vec<&str> = rows.iter().map(|r| r.label.parts()[k].as_str()).collect();
            out.set_item(name, values)?;
        }
        let label = |p: Option<act_core::Period>| p.map(|p| p.to_string());
        out.set_item(
            "column",
            rows.iter().map(|r| r.column.as_str()).collect::<Vec<_>>(),
        )?;
        out.set_item(
            "n_origins",
            rows.iter().map(|r| r.n_origins).collect::<Vec<_>>(),
        )?;
        out.set_item(
            "first_origin",
            rows.iter()
                .map(|r| label(r.first_origin))
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "last_origin",
            rows.iter()
                .map(|r| label(r.last_origin))
                .collect::<Vec<_>>(),
        )?;
        let valuation = rows
            .iter()
            .map(|r| r.valuation.map(|m| month_end(py, m)).transpose())
            .collect::<PyResult<Vec<_>>>()?;
        out.set_item("valuation", valuation)?;
        out.set_item("latest", rows.iter().map(|r| r.latest).collect::<Vec<_>>())?;
        out.set_item("cumulative", vec![summary.cumulative; rows.len()])?;
        match pandas(py)? {
            Some(pd) => pd.call_method1("DataFrame", (out,)),
            None => Ok(out.into_any()),
        }
    }

    /// The printout as text: the origin × development grid for a triangle
    /// with one segment and one measure (as ``view``), otherwise the
    /// ``summary`` table. Numbers are rounded for reading; ``view`` and
    /// ``summary`` give exact values.
    ///
    /// Parameters
    /// ----------
    /// max_rows : int, default 20
    ///     Rows shown before the middle ones are left out; 0 for no limit.
    /// max_cols : int, default 12
    ///     Development ages shown before the middle ones are left out; 0
    ///     for no limit.
    ///
    /// Returns
    /// -------
    /// str
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reserving import Triangle
    /// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [1000.0, 1500.0, 1100.0]})
    /// >>> print(tri.to_string())
    /// Triangle: paid (cumulative, valuation 2021-12)
    ///          12     24
    /// 2020  1,000  1,500
    /// 2021  1,100
    #[pyo3(signature = (max_rows = view::MAX_ROWS, max_cols = view::MAX_COLS))]
    fn to_string(&self, max_rows: usize, max_cols: usize) -> String {
        self.inner.to_text(max_rows, max_cols)
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }

    fn _repr_html_(&self) -> String {
        self.inner.to_html(view::MAX_ROWS, view::MAX_COLS)
    }
}

/// The chain-ladder method: each origin's latest value projected to
/// ultimate with age-to-age factors estimated from the triangle and a tail
/// factor.
///
/// Parameters
/// ----------
/// average : {"volume", "simple", "regression"}, default "volume"
///     How link ratios are averaged into one factor per age: volume
///     weighted, their mean, or least squares through the origin (Mack's
///     ``alpha`` of 1, 0 and 2).
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
///     How a variance parameter with a single link ratio is filled in.
/// tail : float, default 1.0
///     Factor from the oldest age to ultimate.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import ChainLadder, Triangle
/// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 200.0]})
/// >>> fit = ChainLadder().fit(tri, "paid")
/// >>> fit.ldf, fit.ultimate, fit.total_reserve
/// ([1.5], [150.0, 300.0], 100.0)
#[pyclass(name = "ChainLadder", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyChainLadder {
    inner: ChainLadder,
}

#[pymethods]
impl PyChainLadder {
    #[new]
    #[pyo3(signature = (average = "volume", sigma_interpolation = "log-linear", tail = 1.0))]
    fn new(average: &str, sigma_interpolation: &str, tail: f64) -> PyResult<Self> {
        Ok(Self {
            inner: ChainLadder {
                development: development(average, sigma_interpolation)?,
                tail,
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

    /// Tail factor.
    #[getter]
    fn tail(&self) -> f64 {
        self.inner.tail
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
    ///     is not positive; with keys, the message names the segment.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyChainLadderFit> {
        let (cl, tri) = (self.inner, &triangle.inner);
        let inner = py.detach(|| cl.fit_segments(tri, column)).map_err(err)?;
        Ok(PyChainLadderFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ChainLadder(average={:?}, sigma_interpolation={:?}, tail={:?})",
            self.average(),
            self.sigma_interpolation(),
            self.inner.tail
        )
    }
}

/// The one fit of a single-segment result, or an error naming what to use
/// instead: `instead`, if there is an alternative, or `segment(...)`.
fn single<'a, T>(fits: &'a SegmentFits<T>, field: &str, instead: Option<&str>) -> PyResult<&'a T> {
    match fits.fits.as_slice() {
        [one] => Ok(one),
        _ => Err(PyValueError::new_err(format!(
            "{field} needs a single-segment fit, and this one has {} segments; \
             use {}segment(...)",
            fits.len(),
            instead.map_or(String::new(), |i| format!("{i} or ")),
        ))),
    }
}

/// Per-origin values of every segment, in the order of the long rows.
fn by_origin<T>(fits: &SegmentFits<T>, f: impl Fn(&T) -> Vec<f64>) -> Vec<f64> {
    fits.fits.iter().flat_map(f).collect()
}

/// Origin labels of every segment, in the order of the long rows.
fn origin_labels<T: ReserveFit>(fits: &SegmentFits<T>) -> Vec<String> {
    fits.fits
        .iter()
        .flat_map(|f| f.chain_ladder().origins.iter().map(ToString::to_string))
        .collect()
}

/// Labels of the segments, as ``Triangle.index``.
fn segment_labels<'py, T>(
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
fn segment_keys(keys: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, String)>> {
    match keys {
        None => Ok(Vec::new()),
        Some(keys) => keys
            .iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, v.str()?.to_string())))
            .collect(),
    }
}

fn pick<T: Clone>(
    fits: &SegmentFits<T>,
    keys: Option<&Bound<'_, PyDict>>,
) -> PyResult<SegmentFits<T>> {
    let keys = segment_keys(keys)?;
    let keys: Vec<(&str, &str)> = keys.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    fits.segment(&keys).map_err(err)
}

/// A long result table as a pandas DataFrame: the key columns, then
/// ``origin`` (str) or ``development`` (age), then the values.
fn table_frame<'py>(py: Python<'py>, table: FitTable) -> PyResult<Bound<'py, PyAny>> {
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
fn segments_prefix<T>(fits: &SegmentFits<T>) -> String {
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
/// Per-age lists (``ldf``, ``cdf``, ``sigma``, ``std_err``) need a
/// single-segment fit; for several segments use ``development_frame()`` or
/// ``segment(...)``.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import ChainLadder, Triangle
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
#[pyclass(name = "ChainLadderFit", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyChainLadderFit {
    inner: SegmentFits<ChainLadderFit>,
}

#[pymethods]
impl PyChainLadderFit {
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

    /// Age-to-age factors; factor ``k`` links age ``k`` to ``k + 1``.
    #[getter]
    fn ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "ldf", Some("development_frame()"))?;
        Ok(f.development.ldf.clone())
    }

    /// Age-to-ultimate factors, one per age, including the tail.
    #[getter]
    fn cdf(&self) -> PyResult<Vec<f64>> {
        Ok(single(&self.inner, "cdf", Some("development_frame()"))?
            .cdf
            .clone())
    }

    /// Variance parameter of each factor, with unestimable ones
    /// interpolated (``nan`` where that is impossible).
    #[getter]
    fn sigma(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "sigma", Some("development_frame()"))?;
        Ok(f.development.sigma.clone())
    }

    /// Standard error of each factor.
    #[getter]
    fn std_err(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "std_err", Some("development_frame()"))?;
        Ok(f.development.std_err.clone())
    }

    /// Tail factor.
    #[getter]
    fn tail(&self) -> f64 {
        self.inner.fits[0].tail
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

    /// One row per segment: the key columns and the segment's total
    /// ``latest``, ``ultimate`` and ``reserve``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// One row per segment and age: the key columns, ``development``,
    /// ``ldf`` (to the next age), ``cdf`` (to ultimate, with the tail),
    /// ``sigma`` and ``std_err``; the oldest age has ``nan`` for ``ldf``,
    /// ``sigma`` and ``std_err``. Needs pandas.
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

/// Mack's distribution-free chain ladder: the chain-ladder projection plus
/// the standard error of each origin's reserve and of the total, split into
/// process and parameter risk (Mack 1993, 1999). No tail factor.
///
/// Parameters
/// ----------
/// average : {"volume", "simple", "regression"}, default "volume"
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import Mack, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = Mack().fit(tri, "values")
/// >>> fit.total_standard_error > 0 and fit.standard_error[0] == 0
/// True
#[pyclass(name = "Mack", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyMack {
    inner: Mack,
}

#[pymethods]
impl PyMack {
    #[new]
    #[pyo3(signature = (average = "volume", sigma_interpolation = "log-linear"))]
    fn new(average: &str, sigma_interpolation: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Mack {
                development: development(average, sigma_interpolation)?,
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
    ///     ages or a variance parameter can be neither estimated nor
    ///     interpolated.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyMackFit> {
        let (mack, tri) = (self.inner, &triangle.inner);
        let inner = py.detach(|| mack.fit_segments(tri, column)).map_err(err)?;
        Ok(PyMackFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "Mack(average={:?}, sigma_interpolation={:?})",
            self.average(),
            self.sigma_interpolation()
        )
    }
}

/// A fitted Mack model of every segment: the chain-ladder fields, plus
/// standard errors of each origin's reserve and of each segment's total.
///
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()``. Per-age lists and the totals' standard errors
/// need a single-segment fit; for several segments use
/// ``development_frame()``, ``totals_frame()`` or ``segment(...)``.
/// ``total_ultimate`` and ``total_reserve`` sum over every segment.
#[pyclass(name = "MackFit", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyMackFit {
    inner: SegmentFits<MackFit>,
}

impl PyMackFit {
    /// The chain ladder of a single-segment fit, for a per-age `field`.
    fn one(&self, field: &str) -> PyResult<&ChainLadderFit> {
        Ok(&single(&self.inner, field, Some("development_frame()"))?.chain_ladder)
    }
}

#[pymethods]
impl PyMackFit {
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

    /// Age-to-age factors.
    #[getter]
    fn ldf(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("ldf")?.development.ldf.clone())
    }

    /// Age-to-ultimate factors.
    #[getter]
    fn cdf(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("cdf")?.cdf.clone())
    }

    /// Variance parameter of each factor.
    #[getter]
    fn sigma(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("sigma")?.development.sigma.clone())
    }

    /// Standard error of each factor.
    #[getter]
    fn std_err(&self) -> PyResult<Vec<f64>> {
        Ok(self.one("std_err")?.development.std_err.clone())
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
        Ok(single(&self.inner, "total_process_risk", Some("totals_frame()"))?.total_process_risk)
    }

    /// Parameter standard error of the total reserve, including the
    /// correlation between origins that share estimated factors.
    #[getter]
    fn total_parameter_risk(&self) -> PyResult<f64> {
        Ok(
            single(&self.inner, "total_parameter_risk", Some("totals_frame()"))?
                .total_parameter_risk,
        )
    }

    /// Mack standard error of the total reserve.
    #[getter]
    fn total_standard_error(&self) -> PyResult<f64> {
        Ok(
            single(&self.inner, "total_standard_error", Some("totals_frame()"))?
                .total_standard_error,
        )
    }

    /// Coefficient of variation of the total reserve.
    #[getter]
    fn total_cv(&self) -> PyResult<f64> {
        Ok(single(&self.inner, "total_cv", Some("totals_frame()"))?.total_cv())
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
    /// Returns
    /// -------
    /// ClaimsDevelopmentResult
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments (use ``segment(...)``), the
    ///     factors are not volume-weighted, or the latest values do not lie
    ///     on one calendar diagonal with one new origin per period.
    fn claims_development_result(&self) -> PyResult<PyClaimsDevelopmentResult> {
        let fit = single(&self.inner, "claims_development_result", None)?;
        let inner = fit.claims_development_result().map_err(err)?;
        Ok(PyClaimsDevelopmentResult { inner })
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
/// >>> from actuarialrs.reserving import Mack, Triangle
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
    module = "actuarialrs.reserving",
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
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero or ``process`` is unknown.
/// OverflowError
///     If ``n_sims`` or ``seed`` is negative or too large.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import OdpBootstrap, Triangle
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
#[pyclass(name = "OdpBootstrap", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyOdpBootstrap {
    inner: OdpBootstrap,
}

#[pymethods]
impl PyOdpBootstrap {
    #[new]
    #[pyo3(signature = (n_sims = 10_000, seed = 0, process = "gamma"))]
    fn new(n_sims: usize, seed: u64, process: &str) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: OdpBootstrap {
                n_sims,
                seed,
                process: process_distribution(process)?,
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

    /// Bootstraps one measure column in every segment of a cumulative
    /// triangle, each with its own residuals and scale, into one joint
    /// distribution of the reserves. Every origin must be observed from the
    /// first age up to its latest.
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
    ///     latest age or a segment has too few observed cells for the
    ///     degrees of freedom to be positive.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyOdpBootstrapFit> {
        let (boot, tri) = (self.inner, &triangle.inner);
        let inner = py.detach(|| boot.fit_segments(tri, column)).map_err(err)?;
        Ok(PyOdpBootstrapFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "OdpBootstrap(n_sims={}, seed={}, process={:?})",
            self.inner.n_sims,
            self.inner.seed,
            self.process()
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
#[pyclass(name = "OdpBootstrapFit", module = "actuarialrs.reserving", frozen)]
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
        single(&self.inner.segments, field, Some(instead))
    }
}

#[pymethods]
impl PyOdpBootstrapFit {
    /// The deterministic volume-weighted chain ladder the bootstrap is
    /// centred on.
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
        Ok(self.grid(&self.one("fitted", "segment(...)")?.fitted))
    }

    /// Adjusted Pearson residuals ``(x - m) / sqrt(|m|) * sqrt(n / (n - p))``,
    /// ``[origin][development]``; ``nan`` where not observed or where the
    /// fitted value is zero.
    #[getter]
    fn residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(self.grid(&self.one("residuals", "segment(...)")?.residuals))
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
