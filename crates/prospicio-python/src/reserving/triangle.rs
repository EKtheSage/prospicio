//! The loss triangle, `Triangle`.

use super::*;

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
/// >>> from prospicio.reserving import Triangle
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
#[pyclass(name = "Triangle", module = "prospicio.reserving", frozen)]
pub(crate) struct PyTriangle {
    pub(crate) inner: Triangle,
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
pub(crate) fn pandas(py: Python<'_>) -> PyResult<Option<Bound<'_, PyModule>>> {
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
    /// >>> from prospicio.reserving import Triangle
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
    /// >>> from prospicio.reserving import Triangle
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
    /// >>> from prospicio.reserving import Triangle
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

    /// Reads a triangle from a CSV file, in a long or a wide layout.
    ///
    /// Long (``development`` given): one row per (keys, origin,
    /// development) cell, as ``from_frame`` takes it; rows with the same
    /// cell are summed. Wide (``development=None``): one row per (keys,
    /// origin), and every other column is a development age, named by its
    /// age in months (``12``, ``24``, ...) unless ``ages`` gives them; a
    /// repeated origin is an error.
    ///
    /// How cells are written: an origin is a year (``1988``), a date
    /// (``2021-07-15`` or ``2021-07``) or a period label as ``origins``
    /// prints it (``2021Q3``, ``2021H2``); an age is a whole number of
    /// months; a valuation (``development_is_valuation``) is a year (its
    /// December), a date or a period label (its last month); a value is a
    /// number, and an empty cell, ``NA`` or ``NaN`` is not observed. Errors
    /// name the file line and column. The file is read in Rust, so pandas is
    /// not needed.
    ///
    /// Parameters
    /// ----------
    /// path : str or os.PathLike
    /// origin : str
    ///     Name of the origin column.
    /// development : str, optional
    ///     Name of the development column (long layout). ``None`` reads a
    ///     wide layout.
    /// columns : str or list of str, optional
    ///     Long: the measure columns, by default every column that is not
    ///     the origin, the development or a key. Wide: the name of the one
    ///     measure, by default ``"values"``.
    /// keys : str or list of str, optional
    ///     Names of the key columns; each distinct combination is a segment.
    /// ages : list of int, optional
    ///     Wide only: the ages in months of the age columns, in column order.
    /// origin_grain : {"Y", "S", "Q", "M"}, default "Y"
    /// development_grain : {"Y", "S", "Q", "M"}, default "Y"
    /// cumulative : bool, default True
    /// development_is_valuation : bool, default False
    ///     Long only: whether ``development`` holds valuations, not ages.
    /// comment : str, optional
    ///     A character that starts a comment line, such as ``"#"``.
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the file cannot be read, a column is missing or a cell cannot
    ///     be read (naming its line and column), or as for ``from_long``.
    ///
    /// Examples
    /// --------
    /// >>> import os, tempfile
    /// >>> from prospicio.reserving import Triangle
    /// >>> path = os.path.join(tempfile.mkdtemp(), "paid.csv")
    /// >>> with open(path, "w") as f:
    /// ...     _ = f.write("year,12,24\n2020,100,150\n2021,110,\n")
    /// >>> tri = Triangle.read_csv(path, "year", columns="paid")
    /// >>> tri.shape, tri.columns, tri.development
    /// ((1, 1, 2, 2), ['paid'], [12, 24])
    #[staticmethod]
    #[pyo3(signature = (path, origin, development = None, columns = None, keys = None, ages = None, origin_grain = "Y", development_grain = "Y", cumulative = true, development_is_valuation = false, comment = None))]
    #[allow(clippy::too_many_arguments)]
    fn read_csv(
        py: Python<'_>,
        path: std::path::PathBuf,
        origin: &str,
        development: Option<&str>,
        columns: Option<&Bound<'_, PyAny>>,
        keys: Option<&Bound<'_, PyAny>>,
        ages: Option<Vec<Lag>>,
        origin_grain: &str,
        development_grain: &str,
        cumulative: bool,
        development_is_valuation: bool,
        comment: Option<&str>,
    ) -> PyResult<Self> {
        let comment = match comment {
            None => None,
            Some(c) if c.len() == 1 && c.is_ascii() => Some(c.as_bytes()[0]),
            Some(c) => {
                return Err(PyValueError::new_err(format!(
                    "comment must be one ASCII character, got {c:?}"
                )));
            }
        };
        let layout = file_layout(
            origin,
            development,
            columns,
            keys,
            ages,
            origin_grain,
            development_grain,
            cumulative,
            development_is_valuation,
        )?
        .comment(comment);
        py.detach(|| Triangle::read_csv(&path, &layout))
            .map(wrap)
            .map_err(err)
    }

    /// Reads a triangle from a sheet of an Excel workbook (``.xlsx``,
    /// ``.xlsm``, ``.xlsb``, ``.xls``) or an OpenDocument spreadsheet
    /// (``.ods``), in a long or a wide layout as ``read_csv`` reads them.
    ///
    /// The sheet's used range starts with the header row. Number cells are
    /// numbers, date cells are dates (an origin or a valuation) and text
    /// cells are read as in a CSV file. Errors name the cell (``C7``) and
    /// the sheet. The workbook is read in Rust (calamine), so neither
    /// openpyxl nor pandas is needed.
    ///
    /// Parameters
    /// ----------
    /// path : str or os.PathLike
    /// origin, development, columns, keys, ages
    ///     As for ``read_csv``.
    /// sheet : str, optional
    ///     The sheet's name; the first sheet by default.
    /// origin_grain, development_grain, cumulative, development_is_valuation
    ///     As for ``read_csv``.
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     As for ``read_csv``, or if the workbook has no such sheet.
    #[staticmethod]
    #[pyo3(signature = (path, origin, development = None, columns = None, keys = None, ages = None, sheet = None, origin_grain = "Y", development_grain = "Y", cumulative = true, development_is_valuation = false))]
    #[allow(clippy::too_many_arguments)]
    fn read_excel(
        py: Python<'_>,
        path: std::path::PathBuf,
        origin: &str,
        development: Option<&str>,
        columns: Option<&Bound<'_, PyAny>>,
        keys: Option<&Bound<'_, PyAny>>,
        ages: Option<Vec<Lag>>,
        sheet: Option<&str>,
        origin_grain: &str,
        development_grain: &str,
        cumulative: bool,
        development_is_valuation: bool,
    ) -> PyResult<Self> {
        let layout = file_layout(
            origin,
            development,
            columns,
            keys,
            ages,
            origin_grain,
            development_grain,
            cumulative,
            development_is_valuation,
        )?;
        py.detach(|| Triangle::read_excel(&path, sheet, &layout))
            .map(wrap)
            .map_err(err)
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
    /// >>> from prospicio.reserving import Triangle
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
    /// >>> from prospicio.reserving import Triangle
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
    /// >>> from prospicio.reserving import Triangle
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
    /// >>> from prospicio.reserving import Triangle
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

    /// The triangle with the development ages up to ``age`` months, as
    /// chainladder-python's ``tri[tri.development <= age]``.
    ///
    /// Every age at or before ``age`` is kept, in every segment, measure
    /// and origin; the later ages are left out, not folded into a tail.
    /// ``age`` need not be one of the triangle's ages, and one at or past
    /// the last age returns the triangle unchanged. The origins, measures
    /// (a premium column among them), grains and basis (cumulative or
    /// incremental) are kept; the valuation becomes the latest valuation of
    /// a kept observed cell. An origin observed only at dropped ages keeps
    /// an empty row, which the fits trim when it is the first or the last
    /// origin and refuse between observed origins. On a cumulative triangle
    /// where every origin keeps an observed cell, the result equals
    /// ``Triangle.from_frame`` on the rows of ``to_frame()`` at the kept
    /// ages (an origin left without cells is not in those rows); on an
    /// incremental one that rebuild brings the dropped ages back as zero
    /// increments, and this method does not.
    ///
    /// Parameters
    /// ----------
    /// age : int
    ///     The last age to keep, in months.
    ///
    /// Returns
    /// -------
    /// Triangle
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``age`` is before the first age, so that nothing is kept, or
    ///     no observed cell is left.
    /// OverflowError
    ///     If ``age`` is negative.
    ///
    /// Examples
    /// --------
    /// >>> from prospicio.reserving import Triangle
    /// >>> tri = Triangle.from_long(
    /// ...     [2020, 2020, 2020, 2021, 2021, 2022], [12, 24, 36, 12, 24, 12],
    /// ...     [100.0, 150.0, 145.0, 110.0, 170.0, 120.0],
    /// ... )
    /// >>> tri.max_age(24).development
    /// [12, 24]
    /// >>> tri.max_age(30) == tri.max_age(24), tri.max_age(36) == tri
    /// (True, True)
    fn max_age(&self, age: u32) -> PyResult<Self> {
        self.inner.max_age(age).map(wrap).map_err(err)
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
    /// >>> from prospicio.reserving import Triangle
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
    /// >>> from prospicio.reserving import Triangle
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
        let label = |p: Option<prospicio_core::Period>| p.map(|p| p.to_string());
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
    /// >>> from prospicio.reserving import Triangle
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
