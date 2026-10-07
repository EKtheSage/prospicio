//! `actuarialrs.reserving` (Reserving lane): the loss triangle, the chain
//! ladder, Mack's model, the expected-loss methods, Clark's growth curves
//! and the ODP bootstrap over `act_reserving` (`docs/design/triangle.md`,
//! `docs/design/reserving-v02.md`).
//!
//! Long tables come in as array-likes (lists, numpy arrays, pandas or
//! Polars columns) and go out as dicts of lists; numpy and pandas are used
//! when the caller passes them but are not required.

use act_core::{Grain, Lag, Month};
use act_reserving::{
    Average, Benktander, BornhuetterFerguson, CapeCod, CapeCodFit, ChainLadder, ChainLadderFit,
    ClaimsDevelopmentResult, ClarkCapeCod, ClarkFit, ClarkLdf, CurveShape, Development,
    DevelopmentColumn, ExpectedLoss, ExpectedLossFit, FitTable, GrowthCurve, Label, Long, Mack,
    MackBootstrap, MackBootstrapSegment, MackFit, MackProcess, OdpBootstrap, OdpBootstrapFits,
    OdpBootstrapSegment, OneYearFits, OneYearMethod, ProcessDistribution, ReserveFit, SegmentFits,
    SigmaInterpolation, Tail, TailBondy, TailConstant, TailCurve, Triangle, view,
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

/// A given tail factor, as chainladder-python's ``TailConstant``.
///
/// The factor applies from the attachment age to ultimate. Past the
/// attachment it is spread over the following periods as
/// ``1 + x * decay**k``, the last factor making up the difference; this
/// shapes the factors past the attachment, not the factor to ultimate. An
/// attachment before the oldest age replaces the estimated factors from
/// there.
///
/// Parameters
/// ----------
/// factor : float, default 1.0
///     Factor from the attachment age to ultimate; finite and positive.
/// decay : float, default 0.5
///     Share of each period's development kept in the next, from 0 to 1.
/// attachment_age : int, optional
///     Age in months the factor attaches at (the first age at or after
///     it); the oldest age by default. An age at or before the youngest
///     replaces every estimated factor (chainladder-python ignores such an
///     attachment).
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import ChainLadder, TailConstant, Triangle
/// >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 200.0]})
/// >>> fit = ChainLadder(tail=TailConstant(1.05)).fit(tri, "paid")
/// >>> fit.tail, round(fit.ultimate[1], 6)
/// (1.05, 315.0)
#[pyclass(name = "TailConstant", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyTailConstant {
    inner: TailConstant,
}

#[pymethods]
impl PyTailConstant {
    #[new]
    #[pyo3(signature = (factor = 1.0, decay = 0.5, attachment_age = None))]
    fn new(factor: f64, decay: f64, attachment_age: Option<Lag>) -> Self {
        Self {
            inner: TailConstant {
                factor,
                decay,
                attachment_age,
            },
        }
    }

    /// Factor from the attachment age to ultimate.
    #[getter]
    fn factor(&self) -> f64 {
        self.inner.factor
    }

    /// Share of each period's development kept in the next.
    #[getter]
    fn decay(&self) -> f64 {
        self.inner.decay
    }

    /// Age in months the factor attaches at; ``None`` is the oldest age.
    #[getter]
    fn attachment_age(&self) -> Option<Lag> {
        self.inner.attachment_age
    }

    fn __repr__(&self) -> String {
        tail_repr(&Tail::Constant(self.inner))
    }
}

/// A curve fitted to the estimated factors and extrapolated, as
/// chainladder-python's ``TailCurve``.
///
/// Factors above 1.00001 in the fit period are regressed by least squares:
/// ``ln(f - 1)`` on the 1-based development index ``k`` (exponential) or on
/// ``ln(k)`` (inverse power). The fitted curve replaces the factors from the
/// attachment age on and runs ``extrap_periods`` periods past the oldest
/// age.
///
/// Parameters
/// ----------
/// curve : {"exponential", "inverse_power"}, default "exponential"
/// fit_period : tuple of (int or None, int or None), default (None, None)
///     Ages in months whose factors enter the fit: from the last age at or
///     before the first (inclusive) to the last age at or before the second
///     (exclusive), as chainladder-python reads them; ``None`` is
///     open-ended.
/// extrap_periods : int, default 100
///     Number of periods past the oldest age the curve is extrapolated.
/// attachment_age : int, optional
///     Age in months the curve attaches at (the first age at or after it);
///     the oldest age by default.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import ChainLadder, TailCurve, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = ChainLadder(tail=TailCurve()).fit(tri, "values")
/// >>> 1.0 < fit.tail < 1.05
/// True
#[pyclass(name = "TailCurve", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyTailCurve {
    inner: TailCurve,
}

#[pymethods]
impl PyTailCurve {
    #[new]
    #[pyo3(signature = (curve = "exponential", fit_period = (None, None), extrap_periods = 100, attachment_age = None))]
    fn new(
        curve: &str,
        fit_period: (Option<Lag>, Option<Lag>),
        extrap_periods: usize,
        attachment_age: Option<Lag>,
    ) -> PyResult<Self> {
        let curve = match curve {
            "exponential" => CurveShape::Exponential,
            "inverse_power" => CurveShape::InversePower,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "curve must be \"exponential\" or \"inverse_power\", got {curve:?}"
                )));
            }
        };
        Ok(Self {
            inner: TailCurve {
                curve,
                fit_period,
                extrap_periods,
                attachment_age,
            },
        })
    }

    /// The curve fitted to ``f - 1``.
    #[getter]
    fn curve(&self) -> &'static str {
        curve_name(self.inner.curve)
    }

    /// Ages whose factors enter the fit, from (inclusive) and to
    /// (exclusive).
    #[getter]
    fn fit_period(&self) -> (Option<Lag>, Option<Lag>) {
        self.inner.fit_period
    }

    /// Number of periods past the oldest age the curve is extrapolated.
    #[getter]
    fn extrap_periods(&self) -> usize {
        self.inner.extrap_periods
    }

    /// Age in months the curve attaches at; ``None`` is the oldest age.
    #[getter]
    fn attachment_age(&self) -> Option<Lag> {
        self.inner.attachment_age
    }

    fn __repr__(&self) -> String {
        tail_repr(&Tail::Curve(self.inner))
    }
}

/// The Bondy tail, as chainladder-python's ``TailBondy``.
///
/// Each log factor from ``earliest_age`` on is taken as ``b`` times the one
/// before it, ``b`` fitted by least squares. The fitted factors are
/// ``f0 ** (b ** j)`` from the factor ``f0`` at ``earliest_age``, and those
/// past the next one multiply to the last fitted factor raised to
/// ``b / (1 - b)``. With the default ``earliest_age`` (the age of the last
/// factor) ``b`` is 1/2 and the tail repeats the last factor.
///
/// Parameters
/// ----------
/// earliest_age : int, optional
///     First age in months whose factor enters the fit (the last age at or
///     before it, as chainladder-python reads it); the age of the last
///     factor by default.
/// attachment_age : int, optional
///     The factor from this age (the last age at or before it) to the next
///     is kept and the fitted ones replace those after it; the age of the
///     last factor by default. Not before ``earliest_age``.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import ChainLadder, TailBondy, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2020, 2021, 2021, 2022],
/// ...     [12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 110.0, 170.0, 120.0],
/// ... )
/// >>> round(ChainLadder(tail=TailBondy()).fit(tri, "values").tail, 12)
/// 1.1
#[pyclass(name = "TailBondy", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyTailBondy {
    inner: TailBondy,
}

#[pymethods]
impl PyTailBondy {
    #[new]
    #[pyo3(signature = (earliest_age = None, attachment_age = None))]
    fn new(earliest_age: Option<Lag>, attachment_age: Option<Lag>) -> Self {
        Self {
            inner: TailBondy {
                earliest_age,
                attachment_age,
            },
        }
    }

    /// First age whose factor enters the fit; ``None`` is the age of the
    /// last factor.
    #[getter]
    fn earliest_age(&self) -> Option<Lag> {
        self.inner.earliest_age
    }

    /// Age after which the fitted factors replace the estimated ones;
    /// ``None`` is the age of the last factor.
    #[getter]
    fn attachment_age(&self) -> Option<Lag> {
        self.inner.attachment_age
    }

    fn __repr__(&self) -> String {
        tail_repr(&Tail::Bondy(self.inner))
    }
}

/// R ChainLadder's ``tail = TRUE`` rule (its ``tailfactor`` function).
///
/// When the third- and second-last factors multiply to more than 1.0001,
/// ``ln(f - 1)`` is regressed on the development index over the factors
/// above 1 and the next 100 extrapolated factors are multiplied; otherwise
/// the tail is 1. A tail above 2 is reset to 1, as R does.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import Mack, TailLogLinear, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
/// ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
/// ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
/// ... )
/// >>> fit = Mack(tail=TailLogLinear()).fit(tri, "values")
/// >>> fit.tail > 1.0 and fit.standard_error[0] > 0.0
/// True
#[pyclass(name = "TailLogLinear", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyTailLogLinear;

#[pymethods]
impl PyTailLogLinear {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        tail_repr(&Tail::LogLinear)
    }
}

fn curve_name(c: CurveShape) -> &'static str {
    match c {
        CurveShape::Exponential => "exponential",
        CurveShape::InversePower => "inverse_power",
    }
}

/// A constant tail with the default decay and attachment, shown as its
/// factor.
fn plain_factor(tail: &Tail) -> Option<f64> {
    match tail {
        Tail::Constant(t) if t.decay == 0.5 && t.attachment_age.is_none() => Some(t.factor),
        _ => None,
    }
}

fn tail_repr(tail: &Tail) -> String {
    let age = |a: Option<Lag>| a.map_or("None".to_string(), |a| a.to_string());
    match tail {
        Tail::Constant(t) => format!(
            "TailConstant(factor={:?}, decay={:?}, attachment_age={})",
            t.factor,
            t.decay,
            age(t.attachment_age)
        ),
        Tail::Curve(t) => format!(
            "TailCurve(curve={:?}, fit_period=({}, {}), extrap_periods={}, attachment_age={})",
            curve_name(t.curve),
            age(t.fit_period.0),
            age(t.fit_period.1),
            t.extrap_periods,
            age(t.attachment_age)
        ),
        Tail::Bondy(t) => format!(
            "TailBondy(earliest_age={}, attachment_age={})",
            age(t.earliest_age),
            age(t.attachment_age)
        ),
        Tail::LogLinear => "TailLogLinear()".to_string(),
    }
}

/// The ``tail`` argument of ``ChainLadder``, ``Mack`` and the
/// expected-loss methods: ``None`` (no tail), a number (a constant factor)
/// or a tail estimator.
fn tail_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Tail> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(Tail::default());
    };
    if let Ok(t) = obj.extract::<PyRef<'_, PyTailConstant>>() {
        return Ok(Tail::Constant(t.inner));
    }
    if let Ok(t) = obj.extract::<PyRef<'_, PyTailCurve>>() {
        return Ok(Tail::Curve(t.inner));
    }
    if let Ok(t) = obj.extract::<PyRef<'_, PyTailBondy>>() {
        return Ok(Tail::Bondy(t.inner));
    }
    if obj.extract::<PyRef<'_, PyTailLogLinear>>().is_ok() {
        return Ok(Tail::LogLinear);
    }
    if !obj.is_instance_of::<PyBool>() {
        if let Ok(factor) = obj.extract::<f64>() {
            return Ok(factor.into());
        }
    }
    Err(PyTypeError::new_err(format!(
        "tail must be a number, TailConstant, TailCurve, TailBondy or TailLogLinear, got {}",
        obj.repr()?
    )))
}

/// The ``tail`` of a method as Python sees it: a plain
/// constant as its factor, any other tail as its estimator.
fn tail_object<'py>(py: Python<'py>, tail: &Tail) -> PyResult<Bound<'py, PyAny>> {
    if let Some(factor) = plain_factor(tail) {
        return Ok(PyFloat::new(py, factor).into_any());
    }
    Ok(match *tail {
        Tail::Constant(inner) => Bound::new(py, PyTailConstant { inner })?.into_any(),
        Tail::Curve(inner) => Bound::new(py, PyTailCurve { inner })?.into_any(),
        Tail::Bondy(inner) => Bound::new(py, PyTailBondy { inner })?.into_any(),
        Tail::LogLinear => Bound::new(py, PyTailLogLinear)?.into_any(),
    })
}

/// ``repr`` of a ``tail`` argument: a plain constant as its factor.
fn tail_arg_repr(tail: &Tail) -> String {
    plain_factor(tail).map_or_else(|| tail_repr(tail), |f| format!("{f:?}"))
}

/// The chain-ladder method: each origin's latest value projected to
/// ultimate with age-to-age factors estimated from the triangle and a tail.
///
/// Parameters
/// ----------
/// average : {"volume", "simple", "regression"}, default "volume"
///     How link ratios are averaged into one factor per age: volume
///     weighted, their mean, or least squares through the origin (Mack's
///     ``alpha`` of 1, 0 and 2).
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
///     How a variance parameter with a single link ratio is filled in.
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     Development past the oldest age: a number is a constant factor from
///     the oldest age to ultimate. No tail (a factor of 1) by default.
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
    #[pyo3(signature = (average = "volume", sigma_interpolation = "log-linear", tail = None))]
    fn new(
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ChainLadder {
                development: development(average, sigma_interpolation)?,
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

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.tail)
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
        let (cl, tri) = (self.inner, &triangle.inner);
        let inner = py.detach(|| cl.fit_segments(tri, column)).map_err(err)?;
        Ok(PyChainLadderFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ChainLadder(average={:?}, sigma_interpolation={:?}, tail={})",
            self.average(),
            self.sigma_interpolation(),
            tail_arg_repr(&self.inner.tail)
        )
    }
}

/// The one fit of a single-segment result, or an error naming what to use
/// instead: the long table `instead` (if any) or `segment(...)`.
fn single<'a, T>(fits: &'a SegmentFits<T>, field: &str, instead: &str) -> PyResult<&'a T> {
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
fn attachment_age(fit: &ChainLadderFit) -> PyResult<Lag> {
    fit.development
        .development
        .get(fit.tail.attachment)
        .copied()
        .ok_or_else(|| PyValueError::new_err("the fit has no development ages"))
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
/// Per-age lists (``ldf``, ``cdf``, ``sigma``, ``std_err``) and the tail
/// need a single-segment fit; for several segments use
/// ``development_frame()`` (per age), ``totals_frame()`` (``tail``,
/// ``tail_sigma``, ``tail_std_err``) or ``segment(...)``.
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

    /// Selected age-to-age factors, which the projection uses: the
    /// estimated ones, replaced by the tail's from its attachment age.
    /// Factor ``k`` links age ``k`` to ``k + 1``.
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

    /// Standard error of each factor.
    #[getter]
    fn std_err(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "std_err", "development_frame()")?;
        Ok(f.development.std_err.clone())
    }

    /// Age-to-age factors as estimated, before the tail replaced any.
    #[getter]
    fn estimated_ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "estimated_ldf", "")?;
        Ok(f.development.ldf.clone())
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
    /// ``tail_sigma`` and ``tail_std_err``. Needs pandas.
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
    /// ``cdf``. Needs pandas.
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
/// average : {"volume", "simple", "regression"}, default "volume"
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
/// tail_sigma : float, optional
///     The tail's sigma (R's ``tail.sigma``); extrapolated if not given.
///     Unused when the tail factor is 1.
/// tail_std_err : float, optional
///     The tail factor's standard error (R's ``tail.se``); extrapolated if
///     not given. Unused when the tail factor is 1.
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
    #[pyo3(signature = (
        average = "volume",
        sigma_interpolation = "log-linear",
        tail = None,
        tail_sigma = None,
        tail_std_err = None,
    ))]
    fn new(
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
        tail_sigma: Option<f64>,
        tail_std_err: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Mack {
                development: development(average, sigma_interpolation)?,
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
        let (mack, tri) = (self.inner, &triangle.inner);
        let inner = py.detach(|| mack.fit_segments(tri, column)).map_err(err)?;
        Ok(PyMackFit { inner })
    }

    fn __repr__(&self) -> String {
        let given =
            |name: &str, v: Option<f64>| v.map_or(String::new(), |v| format!(", {name}={v:?}"));
        format!(
            "Mack(average={:?}, sigma_interpolation={:?}, tail={}{}{})",
            self.average(),
            self.sigma_interpolation(),
            tail_arg_repr(&self.inner.tail),
            given("tail_sigma", self.inner.tail_sigma),
            given("tail_std_err", self.inner.tail_std_err),
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
#[pyclass(name = "MackFit", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyMackFit {
    inner: SegmentFits<MackFit>,
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

    /// Standard error of each factor.
    #[getter]
    fn std_err(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .one("std_err", "development_frame()")?
            .development
            .std_err
            .clone())
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
    /// Returns
    /// -------
    /// ClaimsDevelopmentResult
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the fit has several segments (use ``segment(...)``), the
    ///     factors are not volume-weighted, the fit has a tail (a factor
    ///     other than 1, or one that replaces estimated factors), or the
    ///     latest values do not lie on one calendar diagonal with one new
    ///     origin per period.
    fn claims_development_result(&self) -> PyResult<PyClaimsDevelopmentResult> {
        let fit = single(&self.inner, "claims_development_result", "")?;
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

/// The chain ladder an expected-loss method estimates its development
/// pattern with.
fn pattern(
    average: &str,
    sigma_interpolation: &str,
    tail: Option<&Bound<'_, PyAny>>,
) -> PyResult<ChainLadder> {
    Ok(ChainLadder {
        development: development(average, sigma_interpolation)?,
        tail: tail_arg(tail)?,
    })
}

/// The ``repr`` arguments describing a method's development pattern.
fn pattern_repr(cl: &ChainLadder) -> String {
    format!(
        "average={:?}, sigma_interpolation={:?}, tail={}",
        average_name(cl.development.average),
        sigma_interpolation_name(cl.development.sigma_interpolation),
        tail_arg_repr(&cl.tail)
    )
}

/// The expected loss ratio method: each origin's ultimate is ``apriori``
/// times its exposure, whatever has been observed. The chain ladder is
/// still fitted for the development pattern the fit reports.
///
/// The exposure is a measure column of the same triangle (premium, say):
/// each origin's latest observed cumulative value in the segment fitted.
///
/// Parameters
/// ----------
/// apriori : float, default 1.0
///     Expected loss ratio: the ultimate per unit of exposure; positive.
/// average : {"volume", "simple", "regression"}, default "volume"
///     How link ratios are averaged, as in ``ChainLadder``.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import ExpectedLoss, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = ExpectedLoss(apriori=0.5).fit(tri, "paid", "premium")
/// >>> fit.ultimate, fit.reserve
/// ([125.0, 200.0], [-25.0, 0.0])
#[pyclass(name = "ExpectedLoss", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyExpectedLoss {
    inner: ExpectedLoss,
}

#[pymethods]
impl PyExpectedLoss {
    #[new]
    #[pyo3(signature = (apriori = 1.0, average = "volume", sigma_interpolation = "log-linear", tail = None))]
    fn new(
        apriori: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ExpectedLoss {
                apriori,
                chain_ladder: pattern(average, sigma_interpolation, tail)?,
            },
        })
    }

    /// Expected loss ratio.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.chain_ladder.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.chain_ladder.development.sigma_interpolation)
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.chain_ladder.tail)
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure (an incremental triangle's is cumulated).
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, if ``apriori`` is not positive, or if an
    ///     origin has no observed, finite, positive exposure (the message
    ///     names it and, with keys, its segment).
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyExpectedLossFit> {
        let (m, tri) = (self.inner, &triangle.inner);
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ExpectedLoss(apriori={:?}, {})",
            self.inner.apriori,
            pattern_repr(&self.inner.chain_ladder)
        )
    }
}

/// The Bornhuetter–Ferguson method: each origin's latest value plus the
/// expected loss ``apriori * exposure`` times the share still to develop,
/// ``1 - 1 / cdf``, as chainladder-python's ``BornhuetterFerguson``.
///
/// The exposure is a measure column of the same triangle (premium, say):
/// each origin's latest observed cumulative value in the segment fitted.
///
/// Parameters
/// ----------
/// apriori : float, default 1.0
///     Expected loss ratio: the expected ultimate per unit of exposure;
///     positive.
/// average : {"volume", "simple", "regression"}, default "volume"
///     How link ratios are averaged, as in ``ChainLadder``.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import BornhuetterFerguson, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = BornhuetterFerguson(apriori=0.5).fit(tri, "paid", "premium")
/// >>> [round(u, 2) for u in fit.ultimate]
/// [150.0, 266.67]
#[pyclass(name = "BornhuetterFerguson", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyBornhuetterFerguson {
    inner: BornhuetterFerguson,
}

#[pymethods]
impl PyBornhuetterFerguson {
    #[new]
    #[pyo3(signature = (apriori = 1.0, average = "volume", sigma_interpolation = "log-linear", tail = None))]
    fn new(
        apriori: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: BornhuetterFerguson {
                apriori,
                chain_ladder: pattern(average, sigma_interpolation, tail)?,
            },
        })
    }

    /// Expected loss ratio.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.chain_ladder.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.chain_ladder.development.sigma_interpolation)
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.chain_ladder.tail)
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure (an incremental triangle's is cumulated).
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ExpectedLoss.fit``.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyExpectedLossFit> {
        let (m, tri) = (self.inner, &triangle.inner);
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "BornhuetterFerguson(apriori={:?}, {})",
            self.inner.apriori,
            pattern_repr(&self.inner.chain_ladder)
        )
    }
}

/// The Benktander (iterated Bornhuetter–Ferguson) method: starting from
/// ``U(0) = apriori * exposure``, ``U(k) = latest + (1 - 1 / cdf) * U(k-1)``
/// for ``n_iters`` steps, as chainladder-python's ``Benktander``.
/// ``n_iters=0`` is the expected loss method, 1 is Bornhuetter–Ferguson,
/// and many iterations approach the chain ladder. The steps are summed in
/// closed form, so a large ``n_iters`` is cheap; where an origin's ``cdf``
/// is below 1/2 they diverge instead.
///
/// Parameters
/// ----------
/// apriori : float, default 1.0
///     Expected loss ratio of the starting ultimate; positive.
/// n_iters : int, default 1
///     Number of Bornhuetter–Ferguson steps.
/// average : {"volume", "simple", "regression"}, default "volume"
///     How link ratios are averaged, as in ``ChainLadder``.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import Benktander, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = Benktander(apriori=0.5, n_iters=2).fit(tri, "paid", "premium")
/// >>> [round(u, 2) for u in fit.ultimate]
/// [150.0, 288.89]
#[pyclass(name = "Benktander", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyBenktander {
    inner: Benktander,
}

#[pymethods]
impl PyBenktander {
    #[new]
    #[pyo3(signature = (apriori = 1.0, n_iters = 1, average = "volume", sigma_interpolation = "log-linear", tail = None))]
    fn new(
        apriori: f64,
        n_iters: usize,
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Benktander {
                apriori,
                n_iters,
                chain_ladder: pattern(average, sigma_interpolation, tail)?,
            },
        })
    }

    /// Expected loss ratio of the starting ultimate.
    #[getter]
    fn apriori(&self) -> f64 {
        self.inner.apriori
    }

    /// Number of Bornhuetter–Ferguson steps.
    #[getter]
    fn n_iters(&self) -> usize {
        self.inner.n_iters
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.chain_ladder.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.chain_ladder.development.sigma_interpolation)
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.chain_ladder.tail)
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure (an incremental triangle's is cumulated).
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ExpectedLoss.fit``.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyExpectedLossFit> {
        let (m, tri) = (self.inner, &triangle.inner);
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyExpectedLossFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "Benktander(apriori={:?}, n_iters={}, {})",
            self.inner.apriori,
            self.inner.n_iters,
            pattern_repr(&self.inner.chain_ladder)
        )
    }
}

/// The Cape Cod (Stanard–Bühlmann) method: Bornhuetter–Ferguson with each
/// origin's apriori estimated from the triangle, as chainladder-python's
/// ``CapeCod``.
///
/// Origin ``j``'s used-up exposure is ``exposure[j] / cdf[j]`` and its
/// latest value is trended to the triangle's valuation by
/// ``(1 + trend) ** (months / 12)``, the months running from the end of
/// the origin period. Origin ``i``'s trended apriori is the sum of the
/// trended latest values weighted by ``decay ** abs(i - j)`` over the same
/// weighted sum of used-up exposures; dividing by its own trend factor
/// gives the apriori of its Bornhuetter–Ferguson ultimate.
///
/// Parameters
/// ----------
/// trend : float, default 0.0
///     Annual trend of the loss ratio; above -1.
/// decay : float, default 1.0
///     Weight of an origin ``n`` periods away, ``decay ** n``; from 0 to 1.
///     With 1 every origin shares one loss ratio.
/// average : {"volume", "simple", "regression"}, default "volume"
///     How link ratios are averaged, as in ``ChainLadder``.
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
/// tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
///     As ``ChainLadder``; no tail by default.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import CapeCod, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = CapeCod().fit(tri, "paid", "premium")
/// >>> [round(a, 4) for a in fit.apriori], [round(u, 2) for u in fit.ultimate]
/// ([0.6774, 0.6774], [150.0, 290.32])
#[pyclass(name = "CapeCod", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyCapeCod {
    inner: CapeCod,
}

#[pymethods]
impl PyCapeCod {
    #[new]
    #[pyo3(signature = (trend = 0.0, decay = 1.0, average = "volume", sigma_interpolation = "log-linear", tail = None))]
    fn new(
        trend: f64,
        decay: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CapeCod {
                trend,
                decay,
                chain_ladder: pattern(average, sigma_interpolation, tail)?,
            },
        })
    }

    /// Annual trend of the loss ratio.
    #[getter]
    fn trend(&self) -> f64 {
        self.inner.trend
    }

    /// Weight of an origin one period away.
    #[getter]
    fn decay(&self) -> f64 {
        self.inner.decay
    }

    /// How link ratios are averaged.
    #[getter]
    fn average(&self) -> &'static str {
        average_name(self.inner.chain_ladder.development.average)
    }

    /// How unestimable variance parameters are filled in.
    #[getter]
    fn sigma_interpolation(&self) -> &'static str {
        sigma_interpolation_name(self.inner.chain_ladder.development.sigma_interpolation)
    }

    /// The tail: a constant factor as a number, otherwise its estimator.
    #[getter]
    fn tail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tail_object(py, &self.inner.chain_ladder.tail)
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure and apriori. Trend runs to the triangle's valuation.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed cumulative
    ///     value is its exposure (an incremental triangle's is cumulated).
    ///
    /// Returns
    /// -------
    /// CapeCodFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, if ``trend`` or ``decay`` is out of
    ///     range, or if an origin has no observed, finite, positive exposure.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyCapeCodFit> {
        let (m, tri) = (self.inner, &triangle.inner);
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyCapeCodFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "CapeCod(trend={:?}, decay={:?}, {})",
            self.inner.trend,
            self.inner.decay,
            pattern_repr(&self.inner.chain_ladder)
        )
    }
}

/// A fitted expected-loss method (``ExpectedLoss``,
/// ``BornhuetterFerguson`` or ``Benktander``) of every segment of a
/// triangle column.
///
/// Per-origin lists (``origins``, ``latest``, ``exposure``, ``apriori``,
/// ``ultimate``, ``reserve``) run over the origins of each segment in turn,
/// like the rows of ``to_frame()``. ``ultimate`` and ``reserve`` are this
/// method's; ``chain_ladder`` holds the chain ladder's. Per-age lists need
/// a single-segment fit; for several segments use ``development_frame()``
/// or ``segment(...)``.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import BornhuetterFerguson, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021] * 2,
/// ...     [12, 24, 12] * 2,
/// ...     {"paid": [100.0, 150.0, 200.0, 10.0, 20.0, 30.0],
/// ...      "premium": [250.0, 250.0, 400.0, 500.0, 500.0, 800.0]},
/// ...     keys={"lob": ["Auto"] * 3 + ["Home"] * 3},
/// ... )
/// >>> fit = BornhuetterFerguson(apriori=0.5).fit(tri, "paid", "premium")
/// >>> fit.exposure, fit.segment(lob="Home").ultimate
/// ([250.0, 400.0, 500.0, 800.0], [20.0, 230.0])
#[pyclass(name = "ExpectedLossFit", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyExpectedLossFit {
    inner: SegmentFits<ExpectedLossFit>,
}

#[pymethods]
impl PyExpectedLossFit {
    /// The underlying chain-ladder projection, with the chain ladder's
    /// ultimate.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.map(|f| f.chain_ladder.clone()),
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

    /// Age-to-age factors; factor ``k`` links age ``k`` to ``k + 1``.
    #[getter]
    fn ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "ldf", "development_frame()")?;
        Ok(f.chain_ladder.development.ldf.clone())
    }

    /// Age-to-ultimate factors, one per age, including the tail.
    #[getter]
    fn cdf(&self) -> PyResult<Vec<f64>> {
        Ok(single(&self.inner, "cdf", "development_frame()")?
            .chain_ladder
            .cdf
            .clone())
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.chain_ladder.latest.clone())
    }

    /// Exposure per origin: the exposure column's latest observed cumulative
    /// value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.exposure.clone())
    }

    /// Expected loss ratio applied per origin.
    #[getter]
    fn apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.apriori.clone())
    }

    /// This method's ultimate per origin.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    /// Reserve (ultimate minus latest) per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ExpectedLossFit::reserves)
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
    /// ``latest``, ``ultimate``, ``reserve``, ``exposure`` and ``apriori``.
    /// Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns and the segment's total
    /// ``latest``, ``ultimate``, ``reserve`` and ``exposure``. Needs pandas.
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

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``.
    ///
    /// Returns
    /// -------
    /// ExpectedLossFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ExpectedLossFit({}origins={}, total_ultimate={:?}, total_reserve={:?})",
            segments_prefix(&self.inner),
            self.origins().len(),
            self.inner.total_ultimate(),
            self.inner.total_reserve()
        )
    }
}

/// A fitted Cape Cod of every segment of a triangle column: the fields of
/// ``ExpectedLossFit``, with ``apriori`` the detrended loss ratio applied
/// to each origin (chainladder-python's ``detrended_apriori_``), plus
/// ``trended_apriori`` before detrending (its ``apriori_``).
///
/// Per-origin lists run over the origins of each segment in turn, like the
/// rows of ``to_frame()``.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import CapeCod, Triangle
/// >>> tri = Triangle.from_long(
/// ...     [2020, 2020, 2021], [12, 24, 12],
/// ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
/// ... )
/// >>> fit = CapeCod(trend=0.1).fit(tri, "paid", "premium")
/// >>> round(fit.trended_apriori[0] / fit.apriori[0], 10)
/// 1.1
#[pyclass(name = "CapeCodFit", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyCapeCodFit {
    inner: SegmentFits<CapeCodFit>,
}

#[pymethods]
impl PyCapeCodFit {
    /// The expected-loss fit: ultimates, exposures and the detrended
    /// apriori.
    #[getter]
    fn expected_loss(&self) -> PyExpectedLossFit {
        PyExpectedLossFit {
            inner: self.inner.map(|f| f.expected_loss.clone()),
        }
    }

    /// The underlying chain-ladder projection, with the chain ladder's
    /// ultimate.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.map(|f| f.expected_loss.chain_ladder.clone()),
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
            .expected_loss
            .chain_ladder
            .development
            .development
            .clone()
    }

    /// Age-to-age factors; factor ``k`` links age ``k`` to ``k + 1``.
    #[getter]
    fn ldf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "ldf", "development_frame()")?;
        Ok(f.expected_loss.chain_ladder.development.ldf.clone())
    }

    /// Age-to-ultimate factors, one per age, including the tail.
    #[getter]
    fn cdf(&self) -> PyResult<Vec<f64>> {
        let f = single(&self.inner, "cdf", "development_frame()")?;
        Ok(f.expected_loss.chain_ladder.cdf.clone())
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_loss.chain_ladder.latest.clone())
    }

    /// Exposure per origin: the exposure column's latest observed cumulative
    /// value.
    #[getter]
    fn exposure(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_loss.exposure.clone())
    }

    /// Detrended expected loss ratio applied per origin.
    #[getter]
    fn apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_loss.apriori.clone())
    }

    /// Expected loss ratio per origin at the valuation's cost level, before
    /// detrending.
    #[getter]
    fn trended_apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.trended_apriori.clone())
    }

    /// Cape Cod ultimate per origin.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_loss.ultimate.clone())
    }

    /// Reserve (ultimate minus latest) per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, CapeCodFit::reserves)
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
    /// ``latest``, ``ultimate``, ``reserve``, ``exposure``, ``apriori`` and
    /// ``trended_apriori``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns and the segment's total
    /// ``latest``, ``ultimate``, ``reserve`` and ``exposure``. Needs pandas.
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

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``.
    ///
    /// Returns
    /// -------
    /// CapeCodFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CapeCodFit({}origins={}, total_ultimate={:?}, total_reserve={:?})",
            segments_prefix(&self.inner),
            self.origins().len(),
            self.inner.total_ultimate(),
            self.inner.total_reserve()
        )
    }
}

/// The growth curve named `name`.
fn growth_curve(name: &str) -> PyResult<GrowthCurve> {
    match name {
        "loglogistic" => Ok(GrowthCurve::LogLogistic),
        "weibull" => Ok(GrowthCurve::Weibull),
        other => Err(PyValueError::new_err(format!(
            "unknown growth curve {other:?}; expected \"loglogistic\" or \"weibull\""
        ))),
    }
}

fn growth_curve_name(curve: GrowthCurve) -> &'static str {
    match curve {
        GrowthCurve::LogLogistic => "loglogistic",
        GrowthCurve::Weibull => "weibull",
    }
}

/// Clark's LDF method (Clark 2003), as R ChainLadder's ``ClarkLDF``: each
/// origin's expected ultimate and a growth curve are fitted to the
/// incremental losses by over-dispersed Poisson maximum likelihood, with
/// ages measured from the average date of loss (the middle of the origin
/// period, R's ``adol = TRUE``).
///
/// The ultimate is the latest value developed by the fitted curve to
/// ``max_age``. Process risk is the scale times the fitted reserve, and
/// parameter risk the delta method on the parameters' covariance, the
/// scale times the inverse Fisher information.
///
/// Parameters
/// ----------
/// curve : {"loglogistic", "weibull"}, default "loglogistic"
///     The growth curve ``G``: ``x**omega / (x**omega + theta**omega)`` or
///     ``1 - exp(-(x / theta)**omega)``.
/// max_age : float, optional
///     Age in months at which development stops; at least the triangle's
///     last age. ``None`` develops to infinity.
///
/// Raises
/// ------
/// ValueError
///     If ``curve`` is unknown.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import ClarkLdf, Triangle
/// >>> rows = [[110.0, 290.0, 370.0, 420.0, 440.0], [95.0, 300.0, 390.0, 425.0],
/// ...         [130.0, 320.0, 410.0], [105.0, 305.0], [120.0]]
/// >>> tri = Triangle.from_long(
/// ...     [2020 + i for i, row in enumerate(rows) for _ in row],
/// ...     [12 * (d + 1) for row in rows for d in range(len(row))],
/// ...     [v for row in rows for v in row],
/// ... )
/// >>> fit = ClarkLdf(curve="weibull", max_age=120).fit(tri, "values")
/// >>> fit.omega > 0 and fit.total_standard_error > fit.total_process_risk
/// True
/// >>> round(fit.ultimate[2] * fit.growth(36) / fit.growth(120), 6)
/// 410.0
#[pyclass(name = "ClarkLdf", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyClarkLdf {
    inner: ClarkLdf,
}

#[pymethods]
impl PyClarkLdf {
    #[new]
    #[pyo3(signature = (curve = "loglogistic", max_age = None))]
    fn new(curve: &str, max_age: Option<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: ClarkLdf {
                curve: growth_curve(curve)?,
                max_age,
            },
        })
    }

    /// The growth curve.
    #[getter]
    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.curve)
    }

    /// Age in months at which development stops; ``None`` for infinity.
    #[getter]
    fn max_age(&self) -> Option<f64> {
        self.inner.max_age
    }

    /// Fits one loss column in every segment of a triangle, each on its
    /// own.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///
    /// Returns
    /// -------
    /// ClarkFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ChainLadder.fit``, if the triangle has fewer than four ages,
    ///     ``max_age`` is before its last age, an origin's latest value is
    ///     not positive, or the likelihood search does not converge.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
    ) -> PyResult<PyClarkFit> {
        let (m, tri) = (self.inner, &triangle.inner);
        let inner = py.detach(|| m.fit_segments(tri, column)).map_err(err)?;
        Ok(PyClarkFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ClarkLdf(curve={:?}, max_age={})",
            self.curve(),
            max_age_repr(self.inner.max_age)
        )
    }
}

fn max_age_repr(max_age: Option<f64>) -> String {
    max_age.map_or_else(|| "None".to_string(), |m| format!("{m:?}"))
}

/// Clark's Cape Cod method (Clark 2003), as R ChainLadder's
/// ``ClarkCapeCod``: one expected loss ratio times each origin's exposure
/// and a growth curve are fitted to the incremental losses by
/// over-dispersed Poisson maximum likelihood, with ages measured from the
/// average date of loss.
///
/// The reserve is the fitted ``elr * exposure * (G(max_age) - G(age))``;
/// process and parameter risk are as in ``ClarkLdf``.
///
/// Parameters
/// ----------
/// curve : {"loglogistic", "weibull"}, default "loglogistic"
///     The growth curve, as in ``ClarkLdf``.
/// max_age : float, optional
///     Age in months at which development stops; at least the triangle's
///     last age. ``None`` develops to infinity.
///
/// Raises
/// ------
/// ValueError
///     If ``curve`` is unknown.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import ClarkCapeCod, Triangle
/// >>> rows = [[110.0, 290.0, 370.0, 420.0, 440.0], [95.0, 300.0, 390.0, 425.0],
/// ...         [130.0, 320.0, 410.0], [105.0, 305.0], [120.0]]
/// >>> tri = Triangle.from_long(
/// ...     [2020 + i for i, row in enumerate(rows) for _ in row],
/// ...     [12 * (d + 1) for row in rows for d in range(len(row))],
/// ...     {"paid": [v for row in rows for v in row],
/// ...      "premium": [800.0 for row in rows for _ in row]},
/// ... )
/// >>> fit = ClarkCapeCod().fit(tri, "paid", "premium")
/// >>> 0 < fit.elr < 1 and fit.expected_ultimate == [fit.elr * 800.0] * 5
/// True
#[pyclass(name = "ClarkCapeCod", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyClarkCapeCod {
    inner: ClarkCapeCod,
}

#[pymethods]
impl PyClarkCapeCod {
    #[new]
    #[pyo3(signature = (curve = "loglogistic", max_age = None))]
    fn new(curve: &str, max_age: Option<f64>) -> PyResult<Self> {
        Ok(Self {
            inner: ClarkCapeCod {
                curve: growth_curve(curve)?,
                max_age,
            },
        })
    }

    /// The growth curve.
    #[getter]
    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.curve)
    }

    /// Age in months at which development stops; ``None`` for infinity.
    #[getter]
    fn max_age(&self) -> Option<f64> {
        self.inner.max_age
    }

    /// Fits one loss column in every segment of a triangle, each with its
    /// own exposure.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    /// column : str
    ///     The losses to project.
    /// exposure : str
    ///     The exposure column; each origin's latest observed value is its
    ///     exposure.
    ///
    /// Returns
    /// -------
    /// ClarkFit
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As ``ClarkLdf.fit``, and if an origin has no observed, finite,
    ///     positive exposure.
    fn fit(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        exposure: &str,
    ) -> PyResult<PyClarkFit> {
        let (m, tri) = (self.inner, &triangle.inner);
        let inner = py
            .detach(|| m.fit_segments(tri, column, exposure))
            .map_err(err)?;
        Ok(PyClarkFit { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ClarkCapeCod(curve={:?}, max_age={})",
            self.curve(),
            max_age_repr(self.inner.max_age)
        )
    }
}

/// A fitted Clark LDF or Cape Cod model of every segment of a triangle
/// column.
///
/// Per-origin lists (``origins``, ``latest``, ``expected_ultimate``,
/// ``ultimate``, ``reserve`` and the standard errors) run over the origins
/// of each segment in turn, like the rows of ``to_frame()``. The fitted
/// parameters and the standard errors of the total need a single-segment
/// fit; for several segments use ``totals_frame()`` or ``segment(...)``.
/// ``total_ultimate`` and ``total_reserve`` sum over every segment.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import ClarkLdf, Triangle
/// >>> rows = [[110.0, 290.0, 370.0, 420.0, 440.0], [95.0, 300.0, 390.0, 425.0],
/// ...         [130.0, 320.0, 410.0], [105.0, 305.0], [120.0]]
/// >>> tri = Triangle.from_long(
/// ...     [2020 + i for i, row in enumerate(rows) for _ in row],
/// ...     [12 * (d + 1) for row in rows for d in range(len(row))],
/// ...     [v for row in rows for v in row],
/// ... )
/// >>> fit = ClarkLdf().fit(tri, "values")
/// >>> len(fit.covariance), fit.elr, fit.growth(float("inf"))
/// (7, None, 1.0)
#[pyclass(name = "ClarkFit", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyClarkFit {
    inner: SegmentFits<ClarkFit>,
}

impl PyClarkFit {
    /// The fit of a single-segment result, for `field`.
    fn one(&self, field: &str) -> PyResult<&ClarkFit> {
        single(&self.inner, field, "totals_frame()")
    }
}

#[pymethods]
impl PyClarkFit {
    /// The volume-weighted chain ladder of the same column, with the chain
    /// ladder's ultimate.
    #[getter]
    fn chain_ladder(&self) -> PyChainLadderFit {
        PyChainLadderFit {
            inner: self.inner.map(|f| f.chain_ladder.clone()),
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

    /// The growth curve.
    #[getter]
    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.fits[0].curve)
    }

    /// Age in months at which development stops; ``None`` for infinity.
    #[getter]
    fn max_age(&self) -> Option<f64> {
        self.inner.fits[0].max_age
    }

    /// Fitted shape of the growth curve.
    #[getter]
    fn omega(&self) -> PyResult<f64> {
        Ok(self.one("omega")?.omega)
    }

    /// Fitted scale of the growth curve, in months.
    #[getter]
    fn theta(&self) -> PyResult<f64> {
        Ok(self.one("theta")?.theta)
    }

    /// Expected loss ratio (Cape Cod), or ``None`` (LDF).
    #[getter]
    fn elr(&self) -> PyResult<Option<f64>> {
        if self.inner.fits[0].elr.is_none() {
            return Ok(None);
        }
        Ok(self.one("elr")?.elr)
    }

    /// Length of the origin period in months; ages are shifted by half of
    /// it to the average date of loss.
    #[getter]
    fn origin_width(&self) -> f64 {
        self.inner.fits[0].origin_width
    }

    /// Number of observed incremental values fitted; ``scale`` divides by
    /// this less the number of parameters.
    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        Ok(single(&self.inner, "n_observations", "segment(...)")?.n_observations)
    }

    /// Over-dispersion ``sigma**2``: squared Pearson residuals over the
    /// observed incremental values less the number of parameters.
    #[getter]
    fn scale(&self) -> PyResult<f64> {
        Ok(self.one("scale")?.scale)
    }

    /// Covariance of the parameters: the expected ultimates (LDF) or the
    /// expected loss ratio (Cape Cod), then ``omega`` and ``theta``. NaN if
    /// the Fisher information is singular.
    #[getter]
    fn covariance(&self) -> PyResult<Vec<Vec<f64>>> {
        Ok(single(&self.inner, "covariance", "segment(...)")?
            .covariance
            .clone())
    }

    /// Latest observed cumulative value per origin.
    #[getter]
    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.chain_ladder.latest.clone())
    }

    /// Exposure per origin (Cape Cod), or ``None`` (LDF).
    #[getter]
    fn exposure(&self) -> Option<Vec<f64>> {
        self.inner.fits[0].exposure.as_ref()?;
        Some(by_origin(&self.inner, |f| {
            f.exposure.clone().unwrap_or_default()
        }))
    }

    /// Expected ultimate per origin, developed to infinity: fitted (LDF)
    /// or ``elr * exposure`` (Cape Cod).
    #[getter]
    fn expected_ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_ultimate.clone())
    }

    /// Ultimate per origin: the latest value plus the reserve.
    #[getter]
    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    /// Reserve per origin.
    #[getter]
    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ClarkFit::reserves)
    }

    /// Process standard error per origin.
    #[getter]
    fn process_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.process_risk.clone())
    }

    /// Parameter standard error per origin.
    #[getter]
    fn parameter_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.parameter_risk.clone())
    }

    /// Standard error per origin: ``sqrt(process**2 + parameter**2)``.
    #[getter]
    fn standard_error(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.standard_error.clone())
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

    /// Process standard error of the total reserve.
    #[getter]
    fn total_process_risk(&self) -> PyResult<f64> {
        Ok(self.one("total_process_risk")?.total_process_risk)
    }

    /// Parameter standard error of the total reserve, with the covariance
    /// between origins.
    #[getter]
    fn total_parameter_risk(&self) -> PyResult<f64> {
        Ok(self.one("total_parameter_risk")?.total_parameter_risk)
    }

    /// Standard error of the total reserve.
    #[getter]
    fn total_standard_error(&self) -> PyResult<f64> {
        Ok(self.one("total_standard_error")?.total_standard_error)
    }

    /// Share of the expected ultimate developed by a development age.
    ///
    /// Parameters
    /// ----------
    /// age : float
    ///     Development age in months, before the shift to the average date
    ///     of loss; ``inf`` gives 1.
    ///
    /// Returns
    /// -------
    /// float
    fn growth(&self, age: f64) -> PyResult<f64> {
        Ok(single(&self.inner, "growth", "segment(...)")?.growth(age))
    }

    /// One row per segment and origin: the key columns, ``origin``,
    /// ``latest``, ``ultimate``, ``reserve``, (Cape Cod) ``exposure``,
    /// ``expected_ultimate``, ``process_risk``, ``parameter_risk`` and
    /// ``standard_error``. Needs pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.to_long())
    }

    /// One row per segment: the key columns, the segment's total
    /// ``latest``, ``ultimate`` and ``reserve``, the ``process_risk``,
    /// ``parameter_risk`` and ``standard_error`` of its total reserve, and
    /// its ``omega``, ``theta``, ``scale`` and (Cape Cod) ``elr``. Needs
    /// pandas.
    ///
    /// Returns
    /// -------
    /// pandas.DataFrame
    fn totals_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        table_frame(py, self.inner.totals())
    }

    /// The fit of one segment, chosen by key values as
    /// ``ChainLadderFit.segment``.
    ///
    /// Returns
    /// -------
    /// ClarkFit
    #[pyo3(signature = (**keys))]
    fn segment(&self, keys: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys)?,
        })
    }

    fn __repr__(&self) -> String {
        let method = if self.inner.fits[0].elr.is_some() {
            "cape_cod"
        } else {
            "ldf"
        };
        let se = match self.inner.fits.as_slice() {
            [one] => format!(", total_standard_error={:?}", one.total_standard_error),
            _ => String::new(),
        };
        format!(
            "ClarkFit({}method={method:?}, curve={:?}, origins={}, total_reserve={:?}{se})",
            segments_prefix(&self.inner),
            self.curve(),
            self.origins().len(),
            self.inner.total_reserve(),
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

    /// The one-year view of any reserving method: the claims development
    /// result over the coming year, by re-reserving on the bootstrap
    /// ("actuary in the box"). Each simulation resamples the residuals for
    /// the volume-weighted factors, projects every origin's next increment
    /// from its resampled latest value with the bootstrap's process error,
    /// as ``fit`` projects, adds it to the observed latest value, appends
    /// it to the triangle, refits ``method`` and records ``CDR = opening
    /// ultimate - closing ultimate``; a negative CDR is an adverse
    /// development. An origin with one cell left thus has its lifetime
    /// bootstrap reserve as its one-year view; an origin at the last age
    /// gets no new cell. Unlike ``MackFit.claims_development_result()``
    /// (Merz and Wüthrich), any averaging and tail are allowed.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments, an annual development
    ///     grain, and every origin short of the last age on its segment's
    ///     latest diagonal.
    /// column : str
    /// method : ChainLadder, ExpectedLoss, BornhuetterFerguson, Benktander or CapeCod
    ///     The method refitted at the start and at the end of the year.
    /// exposure : str, optional
    ///     The exposure column; required by the expected-loss methods, not
    ///     taken by ``ChainLadder``. Its latest value per origin is kept for
    ///     the end of the year.
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
    ///     As ``fit`` and the method's own ``fit``; if the development
    ///     grain is not a year or an origin short of the last age lags its
    ///     segment's latest diagonal; if ``exposure`` is missing for an
    ///     expected-loss method or given for ``ChainLadder``; or if the
    ///     refit fails in any simulation (the message counts them and gives
    ///     one).
    ///
    /// Examples
    /// --------
    /// >>> from actuarialrs.reserving import BornhuetterFerguson, OdpBootstrap, Triangle
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
    #[pyo3(signature = (triangle, column, method, exposure = None))]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        method: &Bound<'_, PyAny>,
        exposure: Option<String>,
    ) -> PyResult<PyOneYearFit> {
        let method = one_year_method(method, exposure)?;
        let (boot, tri) = (self.inner, &triangle.inner);
        let inner = py
            .detach(|| boot.one_year_segments(tri, column, &method))
            .map_err(err)?;
        Ok(PyOneYearFit {
            inner: OneYear::Odp(inner),
        })
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
        single(&self.inner.segments, field, instead)
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

/// The method of ``OdpBootstrap.one_year``: a chain ladder without
/// exposure, or an expected-loss method with its exposure column.
fn one_year_method(method: &Bound<'_, PyAny>, exposure: Option<String>) -> PyResult<OneYearMethod> {
    if let Ok(m) = method.extract::<PyRef<'_, PyChainLadder>>() {
        if exposure.is_some() {
            return Err(PyValueError::new_err(
                "ChainLadder takes no exposure column",
            ));
        }
        return Ok(OneYearMethod::ChainLadder(m.inner));
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
            m.inner,
            needs_exposure("ExpectedLoss")?,
        ));
    }
    if let Ok(m) = method.extract::<PyRef<'_, PyBornhuetterFerguson>>() {
        return Ok(OneYearMethod::BornhuetterFerguson(
            m.inner,
            needs_exposure("BornhuetterFerguson")?,
        ));
    }
    if let Ok(m) = method.extract::<PyRef<'_, PyBenktander>>() {
        return Ok(OneYearMethod::Benktander(
            m.inner,
            needs_exposure("Benktander")?,
        ));
    }
    if let Ok(m) = method.extract::<PyRef<'_, PyCapeCod>>() {
        return Ok(OneYearMethod::CapeCod(m.inner, needs_exposure("CapeCod")?));
    }
    Err(PyTypeError::new_err(format!(
        "method must be ChainLadder, ExpectedLoss, BornhuetterFerguson, Benktander or \
         CapeCod, got {}",
        method.repr()?
    )))
}

/// Mack's bootstrap for the one-year view (England, Verrall and Wüthrich
/// 2019, Appendix 1): the scaled bias-adjusted residuals of the link
/// ratios are resampled into pseudo factors, and every origin's next
/// cumulative value is drawn from its observed latest value ``C`` with mean
/// ``f* C`` and Mack's variance ``sigma**2 * C**(2 - alpha)``. Beside
/// ``OdpBootstrap`` (variance ``scale`` times the mean increment), it gives
/// the one-year view under Mack's process: with the volume-weighted chain
/// ladder and no tail, ``MackFit.claims_development_result()`` (Merz and
/// Wüthrich) within Monte Carlo error. Simulation ``i`` uses random stream
/// ``i`` of ``seed`` for every segment in turn.
///
/// Parameters
/// ----------
/// n_sims : int, default 10000
///     Number of simulations; positive.
/// seed : int, default 0
///     Seed of the simulation streams, from 0 to ``2**64 - 1``.
/// process : {"gamma", "lognormal", "residuals", "normal", "none"}, default "gamma"
///     Process error on each next cumulative value, all with Mack's mean
///     and variance: Gamma or lognormal (negated for a negative mean), the
///     mean plus a resampled residual times the standard deviation, normal,
///     or none for parameter error only.
/// average : {"volume", "simple", "regression"}, default "volume"
///     How Mack's model averages the link ratios (its ``alpha``).
/// sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
///     How a sigma behind a single link ratio is filled in.
///
/// Raises
/// ------
/// ValueError
///     If ``n_sims`` is zero or a setting is unknown.
/// OverflowError
///     If ``n_sims`` or ``seed`` is negative or too large.
///
/// Examples
/// --------
/// >>> from actuarialrs.reserving import ChainLadder, Mack, MackBootstrap, Triangle
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
#[pyclass(name = "MackBootstrap", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyMackBootstrap {
    inner: MackBootstrap,
}

fn mack_process(name: &str) -> PyResult<MackProcess> {
    match name {
        "gamma" => Ok(MackProcess::Gamma),
        "lognormal" => Ok(MackProcess::Lognormal),
        "residuals" => Ok(MackProcess::Residuals),
        "normal" => Ok(MackProcess::Normal),
        "none" => Ok(MackProcess::None),
        _ => Err(PyValueError::new_err(format!(
            "process must be \"gamma\", \"lognormal\", \"residuals\", \"normal\" or \"none\", \
             got {name:?}"
        ))),
    }
}

fn mack_process_name(p: MackProcess) -> &'static str {
    match p {
        MackProcess::Gamma => "gamma",
        MackProcess::Lognormal => "lognormal",
        MackProcess::Residuals => "residuals",
        MackProcess::Normal => "normal",
        MackProcess::None => "none",
    }
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
    ))]
    fn new(
        n_sims: usize,
        seed: u64,
        process: &str,
        average: &str,
        sigma_interpolation: &str,
    ) -> PyResult<Self> {
        if n_sims == 0 {
            return Err(PyValueError::new_err("n_sims must be positive"));
        }
        Ok(Self {
            inner: MackBootstrap {
                n_sims,
                seed,
                process: mack_process(process)?,
                development: development(average, sigma_interpolation)?,
                centre_residuals: false,
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

    /// The one-year view of any reserving method under Mack's process, as
    /// ``OdpBootstrap.one_year``: each simulation draws the next diagonal
    /// from Mack's bootstrap, appends it to the triangle, refits ``method``
    /// and records ``CDR = opening ultimate - closing ultimate``. Mack's
    /// model has no tail here: development past the oldest age moves only
    /// through ``method``'s refitted tail.
    ///
    /// Parameters
    /// ----------
    /// triangle : Triangle
    ///     Cumulative, with any number of segments, an annual development
    ///     grain, every origin observed from the first age to its latest
    ///     with no negative value, and every origin short of the last age on
    ///     its segment's latest diagonal.
    /// column : str
    /// method : ChainLadder, ExpectedLoss, BornhuetterFerguson, Benktander or CapeCod
    ///     The method refitted at the start and at the end of the year.
    /// exposure : str, optional
    ///     The exposure column; required by the expected-loss methods, not
    ///     taken by ``ChainLadder``.
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
    ///     As ``OdpBootstrap.one_year`` and ``Mack.fit``, and if a
    ///     cumulative value is negative.
    #[pyo3(signature = (triangle, column, method, exposure = None))]
    fn one_year(
        &self,
        py: Python<'_>,
        triangle: PyRef<'_, PyTriangle>,
        column: &str,
        method: &Bound<'_, PyAny>,
        exposure: Option<String>,
    ) -> PyResult<PyOneYearFit> {
        let method = one_year_method(method, exposure)?;
        let (boot, tri) = (self.inner, &triangle.inner);
        let inner = py
            .detach(|| boot.one_year_segments(tri, column, &method))
            .map_err(err)?;
        Ok(PyOneYearFit {
            inner: OneYear::Mack(inner),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MackBootstrap(n_sims={}, seed={}, process={:?}, average={:?}, \
             sigma_interpolation={:?})",
            self.inner.n_sims,
            self.inner.seed,
            self.process(),
            self.average(),
            self.sigma_interpolation(),
        )
    }
}

/// A one-year view from either bootstrap.
enum OneYear {
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
/// ``residuals``, ``scale`` and ``mack`` need a single-segment fit; for
/// several segments use ``segment(...)`` or ``totals_frame()``. ``fitted``
/// and ``scale`` are the ODP bootstrap's (``OdpBootstrapFit``'s), ``mack``
/// is Mack's bootstrap's model, and ``residuals`` are either's.
#[pyclass(name = "OneYearFit", module = "actuarialrs.reserving", frozen)]
pub(crate) struct PyOneYearFit {
    inner: OneYear,
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
