//! Triangles read from CSV files and Excel workbooks (the `io` feature).
//!
//! A file holds a triangle in one of two layouts, described by a
//! [`FileLayout`]:
//!
//! - **Long**: one row per (keys, origin, development) cell, with one or
//!   more measure columns, the shape of a claims extract and of
//!   [`Triangle::from_long`].
//! - **Wide**: one row per (keys, origin), with one column per development
//!   age, the shape of a triangle printed in a workbook.
//!
//! How cells are written (`docs/design/triangle.md`, decision 6):
//!
//! - An **origin** is a year (`1988`, or a whole number such as `1988.0`),
//!   a date (`2021-07-15`, `2021-07`, an Excel date cell) or a period label
//!   as the triangle prints it (`2021Q3`, `2021H2`). Any month inside the
//!   origin period will do.
//! - A **development** cell is an age in whole months (`12`, `24.0`), or,
//!   with [`FileLayout::development_is_valuation`], a valuation: a year
//!   (meaning its December, as Schedule P's `DevelopmentYear`), a date or a
//!   period label (meaning its last month).
//! - A wide layout's age columns are named by their age in months (`12`,
//!   `24`, ...) unless [`FileLayout::ages`] gives the ages.
//! - A **value** is a number. An empty cell, `NA`, `NaN` or `nan` is not
//!   observed; anything else that is not a finite number is an error.
//! - A **key** is text; a number is written as it reads (`353`, not
//!   `353.0`).
//!
//! Errors name the file line (CSV) or the cell (Excel) and the column. Rows
//! whose cells are all empty are skipped, and so are columns with no name
//! and no values (a header ending in a delimiter); a column with values and
//! no name is an error. In a long file, rows with the same (keys, origin,
//! development) are summed, as in [`Triangle::from_long`]; in a wide file
//! two rows in the same (keys, origin period), or two columns at the same
//! age, are an error. A file with no observed value is an error.
//!
//! ```
//! use prospicio_reserving::{FileLayout, Triangle};
//!
//! let long = "year,age,paid\n2020,12,100\n2020,24,150\n2021,12,110\n";
//! let tri = Triangle::from_csv(long.as_bytes(), &FileLayout::long("year", "age", ["paid"]))?;
//! assert_eq!(tri.shape(), [1, 1, 2, 2]);
//!
//! let wide = "year,12,24\n2020,100,150\n2021,110,\n";
//! let same = Triangle::from_csv(wide.as_bytes(), &FileLayout::wide("year", "paid"))?;
//! assert_eq!(same, tri);
//! # Ok::<(), prospicio_reserving::Error>(())
//! ```

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use prospicio_core::{Grain, Lag, Month, Period};

use crate::error::{Error, Result};
use crate::triangle::{DevelopmentColumn, Long, Triangle};

/// Where the columns of a triangle are in a file, and how to read them.
///
/// Build one with [`FileLayout::long`] or [`FileLayout::wide`], then set
/// the keys, grains and other settings; the defaults are no keys, annual
/// origins and development, cumulative values, ages (not valuations) and
/// no comment character.
#[derive(Debug, Clone, PartialEq)]
pub struct FileLayout {
    origin: String,
    shape: Shape,
    keys: Vec<String>,
    origin_grain: Grain,
    development_grain: Grain,
    cumulative: bool,
    development_is_valuation: bool,
    ages: Option<Vec<Lag>>,
    comment: Option<u8>,
}

#[derive(Debug, Clone, PartialEq)]
enum Shape {
    Long {
        development: String,
        columns: Vec<String>,
    },
    Wide {
        column: String,
    },
}

impl FileLayout {
    /// A long layout: the `origin` and `development` columns and the
    /// measure `columns`. No columns means every column that is not the
    /// origin, the development or a key.
    pub fn long<S: Into<String>>(
        origin: impl Into<String>,
        development: impl Into<String>,
        columns: impl IntoIterator<Item = S>,
    ) -> Self {
        Self::new(
            origin.into(),
            Shape::Long {
                development: development.into(),
                columns: columns.into_iter().map(Into::into).collect(),
            },
        )
    }

    /// A wide layout: the `origin` column, and every column that is not the
    /// origin or a key is a development age of the one measure, named
    /// `column` in the triangle.
    pub fn wide(origin: impl Into<String>, column: impl Into<String>) -> Self {
        Self::new(
            origin.into(),
            Shape::Wide {
                column: column.into(),
            },
        )
    }

    fn new(origin: String, shape: Shape) -> Self {
        Self {
            origin,
            shape,
            keys: Vec::new(),
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
            development_is_valuation: false,
            ages: None,
            comment: None,
        }
    }

    /// The key columns, in key order; each distinct combination of their
    /// values is a segment.
    pub fn keys<S: Into<String>>(mut self, keys: impl IntoIterator<Item = S>) -> Self {
        self.keys = keys.into_iter().map(Into::into).collect();
        self
    }

    /// The length of an origin period and the spacing of the ages.
    pub fn grains(mut self, origin: Grain, development: Grain) -> Self {
        self.origin_grain = origin;
        self.development_grain = development;
        self
    }

    /// Whether the values are cumulative (the default) or incremental.
    pub fn cumulative(mut self, cumulative: bool) -> Self {
        self.cumulative = cumulative;
        self
    }

    /// Whether a long layout's development column holds valuations instead
    /// of ages. A wide layout with this set is an error when read.
    pub fn development_is_valuation(mut self, valuation: bool) -> Self {
        self.development_is_valuation = valuation;
        self
    }

    /// The ages in months of a wide layout's age columns, in column order,
    /// in place of their names. A long layout with ages is an error when
    /// read.
    pub fn ages(mut self, ages: impl IntoIterator<Item = Lag>) -> Self {
        self.ages = Some(ages.into_iter().collect());
        self
    }

    /// A CSV file's comment character: lines starting with it are skipped
    /// (`b'#'` for the files in `validation/data/`). Excel ignores it.
    pub fn comment(mut self, comment: Option<u8>) -> Self {
        self.comment = comment;
        self
    }
}

/// One cell of a file, before it is read as an origin, age, key or value.
#[derive(Debug, Clone, PartialEq)]
enum Cell {
    Empty,
    Number(f64),
    Text(String),
    Date(Month),
}

/// Where a table came from, to name a cell in an error.
#[derive(Debug, Clone)]
enum Source {
    /// A CSV file: rows carry their line numbers.
    Csv,
    /// An Excel sheet whose range starts at this 0-based (row, column).
    Excel { sheet: String, first_col: u32 },
}

/// A table read from a file: the header and the rows, each with its line
/// (CSV) or 1-based sheet row (Excel).
struct Table {
    source: Source,
    header: Vec<String>,
    rows: Vec<(u64, Vec<Cell>)>,
}

impl Table {
    /// "line 7, column \"paid\"" or "cell C7 (column \"paid\") of sheet \"Data\"".
    fn at(&self, row: usize, col: usize) -> String {
        let line = self.rows[row].0;
        let name = &self.header[col];
        match &self.source {
            Source::Csv => format!("line {line}, column \"{name}\""),
            Source::Excel { sheet, first_col } => format!(
                "cell {}{line} (column \"{name}\") of sheet \"{sheet}\"",
                column_letters(first_col + col as u32)
            ),
        }
    }

    fn column(&self, name: &str) -> Result<usize> {
        self.header.iter().position(|h| h == name).ok_or_else(|| {
            Error::File(format!(
                "no column named \"{name}\"; the columns are {}",
                self.header
                    .iter()
                    .map(|h| format!("\"{h}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
    }
}

/// Excel's column letters of a 0-based column: 0 is `A`, 26 is `AA`.
fn column_letters(col: u32) -> String {
    let mut n = col as u64 + 1;
    let mut out = Vec::new();
    while n > 0 {
        let r = ((n - 1) % 26) as u8;
        out.push(b'A' + r);
        n = (n - 1) / 26;
    }
    out.reverse();
    String::from_utf8(out).expect("ASCII letters")
}

fn csv_table<R: Read>(mut reader: R, comment: Option<u8>) -> Result<Table> {
    // A spreadsheet's CSV export often starts with a UTF-8 byte order mark,
    // which would hide a comment character on the first line.
    let mut text = Vec::new();
    reader
        .read_to_end(&mut text)
        .map_err(|e| Error::File(e.to_string()))?;
    let text = text.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&text);
    let mut csv = csv::ReaderBuilder::new()
        .has_headers(true)
        // A line of only spaces trims to one empty field: it is skipped with
        // the other empty rows, and `build` names a ragged row.
        .flexible(true)
        .trim(csv::Trim::All)
        .comment(comment)
        .from_reader(text);
    // The file line of a byte offset. The csv crate's own line count skips
    // blank lines and counts a Windows line ending twice, so count the line
    // breaks before the record's first byte that is not a line ending: a
    // `\n`, or a `\r` not followed by `\n` (old Mac files end lines so).
    let newlines: Vec<usize> = (0..text.len())
        .filter(|&k| text[k] == b'\n' || (text[k] == b'\r' && text.get(k + 1) != Some(&b'\n')))
        .collect();
    let line_at = |byte: u64| {
        let mut at = byte as usize;
        while at < text.len() && matches!(text[at], b'\r' | b'\n') {
            at += 1;
        }
        newlines.partition_point(|&k| k < at) as u64 + 1
    };
    let csv_error = |e: csv::Error| {
        let line = e.position().map(|p| line_at(p.byte()));
        Error::File(match line {
            Some(line) => format!("line {line}: {e}"),
            None => e.to_string(),
        })
    };
    let header: Vec<String> = csv
        .headers()
        .map_err(csv_error)?
        .iter()
        .map(str::to_string)
        .collect();
    let mut rows = Vec::new();
    for record in csv.records() {
        let record = record.map_err(csv_error)?;
        let line = record.position().map_or(0, |p| line_at(p.byte()));
        let cells = record
            .iter()
            .map(|field| {
                if field.is_empty() {
                    Cell::Empty
                } else {
                    Cell::Text(field.to_string())
                }
            })
            .collect();
        rows.push((line, cells));
    }
    Ok(Table {
        source: Source::Csv,
        header,
        rows,
    })
}

fn excel_table(path: &Path, sheet: Option<&str>) -> Result<Table> {
    use calamine::{Data, Reader, open_workbook_auto};

    let shown = path.display();
    let mut book = open_workbook_auto(path).map_err(|e| Error::File(format!("{shown}: {e}")))?;
    let names = book.sheet_names();
    let sheet = match sheet {
        Some(s) if names.iter().any(|n| n == s) => s.to_string(),
        Some(s) => {
            return Err(Error::File(format!(
                "{shown} has no sheet \"{s}\"; its sheets are {}",
                names
                    .iter()
                    .map(|n| format!("\"{n}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        None => names
            .first()
            .cloned()
            .ok_or_else(|| Error::File(format!("{shown} has no sheets")))?,
    };
    let range = book
        .worksheet_range(&sheet)
        .map_err(|e| Error::File(format!("{shown}, sheet \"{sheet}\": {e}")))?;
    let Some((first_row, first_col)) = range.start() else {
        return Err(Error::File(format!(
            "sheet \"{sheet}\" of {shown} is empty"
        )));
    };
    let at = |row: u32, col: usize| {
        format!(
            "cell {}{} of sheet \"{sheet}\"",
            column_letters(first_col + col as u32),
            row + 1
        )
    };
    let mut rows = Vec::new();
    let mut header = None;
    for (k, row) in range.rows().enumerate() {
        let sheet_row = first_row + k as u32;
        let cells = row
            .iter()
            .enumerate()
            .map(|(col, data)| {
                Ok(match data {
                    Data::Empty => Cell::Empty,
                    Data::Int(i) => Cell::Number(*i as f64),
                    Data::Float(x) => Cell::Number(*x),
                    Data::String(s) | Data::DateTimeIso(s) => {
                        let s = s.trim();
                        if s.is_empty() {
                            Cell::Empty
                        } else {
                            Cell::Text(s.to_string())
                        }
                    }
                    Data::DateTime(d) if d.is_datetime() => {
                        let (year, month, ..) = d.to_ymd_hms_milli();
                        Cell::Date(Month::new(year as i32, month)?)
                    }
                    Data::DateTime(_) | Data::DurationIso(_) => {
                        return Err(Error::File(format!(
                            "{} is a duration, not a number or a date",
                            at(sheet_row, col)
                        )));
                    }
                    Data::Bool(b) => {
                        return Err(Error::File(format!(
                            "{} is the boolean {b}, not a number",
                            at(sheet_row, col)
                        )));
                    }
                    Data::Error(e) => {
                        return Err(Error::File(format!(
                            "{} holds the Excel error {e}",
                            at(sheet_row, col)
                        )));
                    }
                })
            })
            .collect::<Result<Vec<Cell>>>()?;
        if header.is_none() {
            header = Some(
                cells
                    .iter()
                    .map(|c| match c {
                        Cell::Empty => String::new(),
                        Cell::Number(x) => number_text(*x),
                        Cell::Text(s) => s.clone(),
                        Cell::Date(m) => m.to_string(),
                    })
                    .collect::<Vec<_>>(),
            );
        } else {
            rows.push((u64::from(sheet_row) + 1, cells));
        }
    }
    Ok(Table {
        source: Source::Excel { sheet, first_col },
        header: header.unwrap_or_default(),
        rows,
    })
}

/// A number as a key or header: `353`, not `353.0`.
fn number_text(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        x.to_string()
    }
}

/// A whole number written as text (`1988`, `1988.0`, `12`).
fn whole(text: &str) -> Option<i64> {
    if let Ok(i) = text.parse::<i64>() {
        return Some(i);
    }
    let x: f64 = text.parse().ok()?;
    (x.is_finite() && x.fract() == 0.0 && x.abs() < 1e15).then_some(x as i64)
}

/// A period label or date written as text: its first and last months.
/// `2021` is the year, `2021Q3` and `2021H2` the quarter and half,
/// `2021-07` and `2021-07-15` (with any time after it) the month.
fn period_text(text: &str) -> Option<(Month, Month)> {
    if let Some(year) = whole(text) {
        let year = i32::try_from(year).ok()?;
        return Some((Month::january(year), Month::new(year, 12).ok()?));
    }
    for (letter, size) in [('Q', 3u8), ('H', 6u8)] {
        if let Some((year, part)) = text.split_once(letter) {
            let year: i32 = year.parse().ok()?;
            let part: u8 = part.parse().ok()?;
            if part == 0 || part > 12 / size {
                return None;
            }
            let start = Month::new(year, (part - 1) * size + 1).ok()?;
            return Some((start, start.add_months(i64::from(size) - 1)));
        }
    }
    // A date: YYYY-MM or YYYY-MM-DD, then optionally 'T' or ' ' and a time.
    let date = match text.as_bytes().get(10) {
        Some(b'T' | b' ') => &text[..10],
        _ => text,
    };
    let mut parts = date.split('-');
    let year: i32 = parts.next()?.parse().ok()?;
    let month_text = parts.next()?;
    if month_text.len() != 2 {
        return None;
    }
    let month = Month::new(year, month_text.parse().ok()?).ok()?;
    match parts.next() {
        None => {}
        Some(day) if day.len() == 2 && matches!(day.parse::<u8>(), Ok(1..=31)) => {}
        Some(_) => return None,
    }
    if parts.next().is_some() {
        return None;
    }
    Some((month, month))
}

/// What a value cell reads as: a finite number or missing (NaN).
fn value(cell: &Cell) -> std::result::Result<f64, String> {
    match cell {
        Cell::Empty => Ok(f64::NAN),
        Cell::Number(x) if x.is_finite() => Ok(*x),
        Cell::Number(x) => Err(format!("{x} is not a finite number")),
        Cell::Text(s) if matches!(s.as_str(), "NA" | "NaN" | "nan") => Ok(f64::NAN),
        Cell::Text(s) => match s.parse::<f64>() {
            Ok(x) if x.is_finite() => Ok(x),
            _ => Err(format!("\"{s}\" is not a number")),
        },
        Cell::Date(m) => Err(format!("the date {m} is not a number")),
    }
}

/// The first or last month of the period an origin or valuation cell names.
fn month(cell: &Cell, last: bool) -> std::result::Result<Month, String> {
    let pick = |(first, end): (Month, Month)| if last { end } else { first };
    match cell {
        Cell::Empty => Err("is empty".into()),
        Cell::Date(m) => Ok(*m),
        Cell::Number(x) => {
            let year = (x.is_finite() && x.fract() == 0.0)
                .then_some(*x as i64)
                .and_then(|y| i32::try_from(y).ok())
                .ok_or_else(|| format!("{x} is not a year, a date or a period"))?;
            period_text(&year.to_string())
                .map(pick)
                .ok_or_else(|| format!("{x} is not a year"))
        }
        Cell::Text(s) => period_text(s)
            .map(pick)
            .ok_or_else(|| format!("\"{s}\" is not a year, a date or a period")),
    }
}

/// An age in whole months.
fn age(cell: &Cell) -> std::result::Result<Lag, String> {
    let text = match cell {
        Cell::Empty => return Err("is empty".into()),
        Cell::Number(x) => number_text(*x),
        Cell::Text(s) => s.clone(),
        Cell::Date(m) => return Err(format!("the date {m} is not an age in months")),
    };
    whole(&text)
        .filter(|&a| a > 0)
        .and_then(|a| Lag::try_from(a).ok())
        .ok_or_else(|| format!("\"{text}\" is not an age in whole months"))
}

fn key(cell: &Cell) -> std::result::Result<String, String> {
    match cell {
        Cell::Empty => Err("is empty".into()),
        Cell::Number(x) => Ok(number_text(*x)),
        Cell::Text(s) => Ok(s.clone()),
        Cell::Date(m) => Ok(m.to_string()),
    }
}

/// A cell's problem as a file error: `at` names the cell.
fn cell_error(at: String, why: String) -> Error {
    if why == "is empty" {
        Error::File(format!("{at} is empty"))
    } else {
        Error::File(format!("{at}: {why}"))
    }
}

/// The triangle a table holds under `layout`.
fn build(table: &Table, layout: &FileLayout) -> Result<Triangle> {
    // A column with no name and no values, as a header ending in a
    // delimiter writes it, is not part of the table; a column with values
    // and no name is an error.
    let unnamed: Vec<usize> = (0..table.header.len())
        .filter(|&c| table.header[c].is_empty())
        .collect();
    for &c in &unnamed {
        if table
            .rows
            .iter()
            .any(|(_, cells)| cells.get(c).is_some_and(|x| *x != Cell::Empty))
        {
            return Err(Error::File(format!(
                "column {} has values but no name in the header",
                c + 1
            )));
        }
    }
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for (k, name) in table.header.iter().enumerate() {
        if name.is_empty() {
            continue;
        }
        if let Some(first) = seen.insert(name.as_str(), k) {
            return Err(Error::File(format!(
                "columns {} and {} are both named \"{name}\"",
                first + 1,
                k + 1
            )));
        }
    }
    let n_cols = table.header.len();
    let origin_col = table.column(&layout.origin)?;
    let key_cols: Vec<usize> = layout
        .keys
        .iter()
        .map(|k| table.column(k))
        .collect::<Result<_>>()?;

    let rows: Vec<usize> = (0..table.rows.len())
        .filter(|&r| table.rows[r].1.iter().any(|c| *c != Cell::Empty))
        .collect();
    for &r in &rows {
        let found = table.rows[r].1.len();
        if found != n_cols {
            return Err(Error::File(format!(
                "{} has {found} cells, the header {n_cols}",
                match &table.source {
                    Source::Csv => format!("line {}", table.rows[r].0),
                    Source::Excel { sheet, .. } =>
                        format!("row {} of sheet \"{sheet}\"", table.rows[r].0),
                },
            )));
        }
    }
    let cell = |r: usize, c: usize| &table.rows[r].1[c];
    let origin_of = |r: usize| {
        month(cell(r, origin_col), false).map_err(|why| cell_error(table.at(r, origin_col), why))
    };
    let keys_of = |r: usize| {
        key_cols
            .iter()
            .map(|&c| key(cell(r, c)).map_err(|why| cell_error(table.at(r, c), why)))
            .collect::<Result<Vec<String>>>()
    };

    // The long table, with each long row's source cell for errors.
    let mut origin = Vec::new();
    let mut key_values: Vec<Vec<String>> = vec![Vec::new(); key_cols.len()];
    let mut where_row: Vec<(usize, usize)> = Vec::new();
    let names: Vec<String>;
    let mut values: Vec<Vec<f64>>;
    let mut lags: Vec<Lag> = Vec::new();
    let mut valuations: Vec<Month> = Vec::new();
    let valuation = layout.development_is_valuation;

    match &layout.shape {
        Shape::Long {
            development,
            columns,
        } => {
            if layout.ages.is_some() {
                return Err(Error::File(
                    "ages name the columns of a wide layout; a long layout has a development column"
                        .into(),
                ));
            }
            let dev_col = table.column(development)?;
            let value_cols: Vec<usize> = if columns.is_empty() {
                (0..n_cols)
                    .filter(|c| {
                        *c != origin_col
                            && *c != dev_col
                            && !key_cols.contains(c)
                            && !unnamed.contains(c)
                    })
                    .collect()
            } else {
                columns
                    .iter()
                    .map(|c| table.column(c))
                    .collect::<Result<_>>()?
            };
            if value_cols.is_empty() {
                return Err(Error::File("the file has no measure columns".into()));
            }
            names = value_cols
                .iter()
                .map(|&c| table.header[c].clone())
                .collect();
            values = vec![Vec::with_capacity(rows.len()); value_cols.len()];
            for &r in &rows {
                origin.push(origin_of(r)?);
                for (k, v) in keys_of(r)?.into_iter().enumerate() {
                    key_values[k].push(v);
                }
                let dev = cell(r, dev_col);
                let at_dev = || table.at(r, dev_col);
                if valuation {
                    valuations.push(month(dev, true).map_err(|why| cell_error(at_dev(), why))?);
                } else {
                    lags.push(age(dev).map_err(|why| cell_error(at_dev(), why))?);
                }
                for (k, &c) in value_cols.iter().enumerate() {
                    values[k]
                        .push(value(cell(r, c)).map_err(|why| cell_error(table.at(r, c), why))?);
                }
                where_row.push((r, dev_col));
            }
        }
        Shape::Wide { column } => {
            if valuation {
                return Err(Error::File(
                    "a wide layout's columns are ages; valuations need a long layout".into(),
                ));
            }
            let age_cols: Vec<usize> = (0..n_cols)
                .filter(|c| *c != origin_col && !key_cols.contains(c) && !unnamed.contains(c))
                .collect();
            if age_cols.is_empty() {
                return Err(Error::File("the file has no age columns".into()));
            }
            let ages: Vec<Lag> = match &layout.ages {
                Some(ages) if ages.len() == age_cols.len() => ages.clone(),
                Some(ages) => {
                    return Err(Error::File(format!(
                        "{} ages given for {} age columns",
                        ages.len(),
                        age_cols.len()
                    )));
                }
                None => age_cols
                    .iter()
                    .map(|&c| {
                        let name = &table.header[c];
                        age(&Cell::Text(name.clone())).map_err(|_| {
                            Error::File(format!(
                                "column \"{name}\" of a wide layout is not an age in months; \
                                 name the age columns 12, 24, ... or give the ages"
                            ))
                        })
                    })
                    .collect::<Result<_>>()?,
            };
            // Two columns at one age (`12` and `12.0`, or a repeated given
            // age) would be summed into one cell.
            let mut first_col: HashMap<Lag, usize> = HashMap::new();
            for (&c, &a) in age_cols.iter().zip(&ages) {
                if let Some(before) = first_col.insert(a, c) {
                    return Err(Error::File(format!(
                        "columns \"{}\" and \"{}\" both hold age {a}",
                        table.header[before], table.header[c],
                    )));
                }
            }
            names = vec![column.clone()];
            values = vec![Vec::new()];
            // Two rows in the same origin period (2021-01 and 2021-02 under
            // annual origins) are the same origin.
            let mut first_row: HashMap<(Vec<String>, Period), usize> = HashMap::new();
            for &r in &rows {
                let o = origin_of(r)?;
                let k = keys_of(r)?;
                let period = Period::containing(o, layout.origin_grain);
                if let Some(before) = first_row.insert((k.clone(), period), r) {
                    return Err(Error::File(format!(
                        "{} and {} hold the same origin {period}",
                        table.at(before, origin_col),
                        table.at(r, origin_col),
                    )));
                }
                for (&c, &a) in age_cols.iter().zip(&ages) {
                    origin.push(o);
                    for (kk, v) in k.iter().enumerate() {
                        key_values[kk].push(v.clone());
                    }
                    lags.push(a);
                    values[0]
                        .push(value(cell(r, c)).map_err(|why| cell_error(table.at(r, c), why))?);
                    where_row.push((r, c));
                }
            }
        }
    }
    if origin.is_empty() {
        return Err(Error::File("the file has no rows".into()));
    }

    let key_refs: Vec<Vec<&str>> = key_values
        .iter()
        .map(|v| v.iter().map(String::as_str).collect())
        .collect();
    let key_columns: Vec<(&str, &[&str])> = layout
        .keys
        .iter()
        .zip(&key_refs)
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    let value_columns: Vec<(&str, &[f64])> = names
        .iter()
        .zip(&values)
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    let development = if valuation {
        DevelopmentColumn::Valuation(&valuations)
    } else {
        DevelopmentColumn::Age(&lags)
    };
    Triangle::from_long(&Long {
        keys: &key_columns,
        origin: &origin,
        development,
        values: &value_columns,
        origin_grain: layout.origin_grain,
        development_grain: layout.development_grain,
        cumulative: layout.cumulative,
    })
    .map_err(|e| match e {
        // Rows of the long table are not lines of the file: name the cell.
        Error::NonPositiveAge { row } => {
            let (r, c) = where_row[row];
            Error::File(format!(
                "{}: the valuation is before the origin starts",
                table.at(r, c)
            ))
        }
        Error::OffGrid { row, age } => {
            let (r, c) = where_row[row];
            Error::File(format!(
                "{}: age {age} is not on the development grid (a whole number of {} periods from the youngest age)",
                table.at(r, c),
                layout.development_grain
            ))
        }
        Error::Empty => Error::File("the file has no observed values".into()),
        other => other,
    })
}

impl Triangle {
    /// Reads a triangle from CSV text: a header row, then one row per cell
    /// (long layout) or per origin (wide layout). Fields are trimmed. See
    /// the [module documentation](crate::io) for how cells are written.
    ///
    /// The reader can be a file, a buffer or a network stream; this does no
    /// file access of its own.
    pub fn from_csv<R: Read>(reader: R, layout: &FileLayout) -> Result<Self> {
        build(&csv_table(reader, layout.comment)?, layout)
    }

    /// Reads a triangle from the CSV file at `path`; see
    /// [`Triangle::from_csv`].
    pub fn read_csv(path: impl AsRef<Path>, layout: &FileLayout) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| Error::File(format!("{}: {e}", path.display())))?;
        Self::from_csv(std::io::BufReader::new(file), layout).map_err(|e| match e {
            Error::File(why) => Error::File(format!("{}: {why}", path.display())),
            other => other,
        })
    }

    /// Reads a triangle from a sheet of the workbook at `path` (`.xlsx`,
    /// `.xlsm`, `.xlsb`, `.xls` or `.ods`), the first sheet when `sheet` is
    /// `None`. The sheet's used range starts with the header row. Number
    /// cells are numbers, date cells dates (an origin or valuation) and text
    /// cells are read as in a CSV file.
    pub fn read_excel(
        path: impl AsRef<Path>,
        sheet: Option<&str>,
        layout: &FileLayout,
    ) -> Result<Self> {
        let path = path.as_ref();
        build(&excel_table(path, sheet)?, layout).map_err(|e| match e {
            Error::File(why) if !why.starts_with(&path.display().to_string()) => {
                Error::File(format!("{}: {why}", path.display()))
            }
            other => other,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::raa;

    fn file_error(r: Result<Triangle>) -> String {
        match r {
            Err(Error::File(why)) => why,
            other => panic!("expected a file error, got {other:?}"),
        }
    }

    /// RAA written out long, from the triangle's own long table.
    fn raa_long_csv() -> String {
        let long = raa().to_long();
        let mut out = String::from("origin,development,values\n");
        for (k, o) in long.origin.iter().enumerate() {
            out += &format!(
                "{},{},{}\n",
                o.year(),
                long.development[k],
                long.values[0].1[k]
            );
        }
        out
    }

    /// RAA written out wide: one row per origin, empty below the diagonal.
    fn raa_wide_csv() -> String {
        let tri = raa();
        let mut out = String::from("origin");
        for a in tri.development() {
            out += &format!(",{a}");
        }
        out.push('\n');
        for (o, p) in tri.origins().iter().enumerate() {
            out += &p.to_string();
            for d in 0..tri.development().len() {
                out.push(',');
                if let Some(v) = tri.get(0, 0, o, d) {
                    out += &v.to_string();
                }
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn raa_round_trips_through_both_layouts() {
        let long = Triangle::from_csv(
            raa_long_csv().as_bytes(),
            &FileLayout::long("origin", "development", ["values"]),
        )
        .unwrap();
        assert_eq!(long, raa());
        let wide = Triangle::from_csv(
            raa_wide_csv().as_bytes(),
            &FileLayout::wide("origin", "values"),
        )
        .unwrap();
        assert_eq!(wide, raa());
    }

    #[test]
    fn read_csv_reads_a_file_with_a_bom_and_comments() {
        let dir = std::env::temp_dir().join(format!("prospicio-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("raa.csv");
        let text = format!("\u{feff}# RAA, cumulative paid\n{}", raa_long_csv());
        std::fs::write(&path, text).unwrap();
        let layout =
            FileLayout::long("origin", "development", Vec::<String>::new()).comment(Some(b'#'));
        let tri = Triangle::read_csv(&path, &layout).unwrap();
        assert_eq!(tri, raa());
        let missing = file_error(Triangle::read_csv(dir.join("none.csv"), &layout));
        assert!(missing.contains("none.csv"), "{missing}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn keys_valuations_and_periods() {
        let text = "\
company,ay,dy,paid,incurred
353,1988,1988,10,20
353,1988,1989,15,22
353,1989,1989,12,
388,1988,1988,7,NA
";
        let tri = Triangle::from_csv(
            text.as_bytes(),
            &FileLayout::long("ay", "dy", Vec::<String>::new())
                .keys(["company"])
                .development_is_valuation(true),
        )
        .unwrap();
        assert_eq!(tri.key_names(), ["company"]);
        assert_eq!(tri.columns(), ["paid", "incurred"]);
        assert_eq!(tri.shape(), [2, 2, 2, 2]);
        assert_eq!(tri.development(), [12, 24]);
        assert_eq!(tri.get(0, 0, 0, 1), Some(15.0));
        assert_eq!(tri.get(0, 1, 1, 0), None);
        assert_eq!(tri.get(1, 1, 0, 0), None);
        assert_eq!(tri.index()[1].parts(), ["388"]);

        // Quarterly origins written as labels and dates.
        let text = "q,age,paid\n2021Q1,3,1\n2021-04-15,3,2\n2021-01,6,3\n";
        let tri = Triangle::from_csv(
            text.as_bytes(),
            &FileLayout::long("q", "age", ["paid"]).grains(Grain::Quarter, Grain::Quarter),
        )
        .unwrap();
        assert_eq!(tri.shape(), [1, 1, 2, 2]);
        assert_eq!(tri.get(0, 0, 0, 1), Some(3.0));
        assert_eq!(tri.get(0, 0, 1, 0), Some(2.0));
    }

    #[test]
    fn wide_ages_can_be_given_and_incremental_values_fill() {
        let text = "year,first,second\n2020,100,50\n2021,110,\n";
        let tri = Triangle::from_csv(
            text.as_bytes(),
            &FileLayout::wide("year", "paid")
                .ages([12, 24])
                .cumulative(false),
        )
        .unwrap();
        assert!(!tri.is_cumulative());
        assert_eq!(tri.to_cumulative().get(0, 0, 0, 1), Some(150.0));
    }

    #[test]
    fn errors_name_the_line_and_column() {
        let long = |text: &str| {
            file_error(Triangle::from_csv(
                text.as_bytes(),
                &FileLayout::long("year", "age", ["paid"]),
            ))
        };
        let e = long("year,age,paid\n2020,12,100\n2020,24,1O0\n");
        assert_eq!(e, "line 3, column \"paid\": \"1O0\" is not a number");
        // Windows line endings count the same lines.
        let e = long("year,age,paid\r\n2020,12,100\r\n2020,24,1O0\r\n");
        assert_eq!(e, "line 3, column \"paid\": \"1O0\" is not a number");
        // A blank line is skipped and still counted.
        let e = long("year,age,paid\n\n2020,24,1O0\n");
        assert_eq!(e, "line 3, column \"paid\": \"1O0\" is not a number");
        let e = long("year,age,paid\n2020,12,inf\n");
        assert!(e.contains("line 2, column \"paid\""), "{e}");
        let e = long("year,age,paid\n2020,1.5,100\n");
        assert_eq!(
            e,
            "line 2, column \"age\": \"1.5\" is not an age in whole months"
        );
        let e = long("year,age,paid\n,12,100\n");
        assert_eq!(e, "line 2, column \"year\" is empty");
        let e = long("year,age,paid\nlast year,12,100\n");
        assert!(e.contains("is not a year, a date or a period"), "{e}");
        let e = long("year,dev,paid\n2020,12,100\n");
        assert_eq!(
            e,
            "no column named \"age\"; the columns are \"year\", \"dev\", \"paid\""
        );
        let e = long("year,age,paid\n2020,12,100\n2020,18,100\n");
        assert!(
            e.starts_with("line 3, column \"age\": age 18 is not on"),
            "{e}"
        );
        let e = long("year,age,paid,paid\n2020,12,100,1\n");
        assert_eq!(e, "columns 3 and 4 are both named \"paid\"");
        let e = long("year,age,paid\n2020,12,100,5\n");
        assert!(e.starts_with("line 2"), "{e}");
        let e = long("year,age,paid\n");
        assert_eq!(e, "the file has no rows");

        let wide = |text: &str| {
            file_error(Triangle::from_csv(
                text.as_bytes(),
                &FileLayout::wide("year", "paid"),
            ))
        };
        let e = wide("year,12,24\n2020,1,2\n2021,3,\n2020,5,6\n");
        assert_eq!(
            e,
            "line 2, column \"year\" and line 4, column \"year\" hold the same origin 2020"
        );
        // Months of one origin year are the same origin.
        let e = wide("year,12\n2021-01,1\n2021-02,2\n");
        assert!(e.ends_with("hold the same origin 2021"), "{e}");
        let e = wide("year,12,total\n2020,1,2\n");
        assert!(e.starts_with("column \"total\" of a wide layout"), "{e}");

        // Long files sum repeated cells, as from_long does.
        let tri = Triangle::from_csv(
            "year,age,paid\n2020,12,100\n2020,12,5\n".as_bytes(),
            &FileLayout::long("year", "age", ["paid"]),
        )
        .unwrap();
        assert_eq!(tri.get(0, 0, 0, 0), Some(105.0));
    }

    #[test]
    fn cells_read_as_written() {
        assert_eq!(column_letters(0), "A");
        assert_eq!(column_letters(25), "Z");
        assert_eq!(column_letters(26), "AA");
        assert_eq!(column_letters(701), "ZZ");
        assert_eq!(column_letters(702), "AAA");
        let m = |y, mo| Month::new(y, mo).unwrap();
        let first = |s: &str| month(&Cell::Text(s.into()), false);
        let last = |s: &str| month(&Cell::Text(s.into()), true);
        assert_eq!(first("1988"), Ok(m(1988, 1)));
        assert_eq!(last("1988"), Ok(m(1988, 12)));
        assert_eq!(first("1988.0"), Ok(m(1988, 1)));
        assert_eq!(last("2021Q3"), Ok(m(2021, 9)));
        assert_eq!(first("2021H2"), Ok(m(2021, 7)));
        assert_eq!(last("2021-07-15"), Ok(m(2021, 7)));
        assert_eq!(first("2021-07-15 00:00:00"), Ok(m(2021, 7)));
        assert_eq!(first("2021-07-15T00:00:00"), Ok(m(2021, 7)));
        assert_eq!(first("2021-07"), Ok(m(2021, 7)));
        assert!(first("2021Q5").is_err());
        assert!(first("2021-13").is_err());
        assert!(first("2021-7").is_err());
        assert!(first("2021-07-32").is_err());
        assert_eq!(month(&Cell::Number(1990.0), true), Ok(m(1990, 12)));
        assert!(month(&Cell::Number(1990.5), false).is_err());
        assert_eq!(key(&Cell::Number(353.0)), Ok("353".into()));
        assert_eq!(age(&Cell::Number(24.0)), Ok(24));
        assert!(age(&Cell::Text("0".into())).is_err());
        assert!(value(&Cell::Text("NaN".into())).unwrap().is_nan());
        assert!(value(&Cell::Empty).unwrap().is_nan());
        assert_eq!(value(&Cell::Text("-1.5e3".into())), Ok(-1500.0));
    }

    #[test]
    fn a_line_of_spaces_is_an_empty_row() {
        let tri = Triangle::from_csv(
            "year,age,paid\n2020,12,100\n   \n2021,12,5\n".as_bytes(),
            &FileLayout::long("year", "age", ["paid"]),
        )
        .unwrap();
        assert_eq!(tri.shape(), [1, 1, 2, 1]);
        // A ragged row is still named, with its own line.
        let e = file_error(Triangle::from_csv(
            "year,age,paid\n2020,12\n".as_bytes(),
            &FileLayout::long("year", "age", ["paid"]),
        ));
        assert_eq!(e, "line 2 has 2 cells, the header 3");
    }

    #[test]
    fn an_unnamed_empty_column_is_dropped() {
        // A header and rows ending in a delimiter, as spreadsheets export.
        let tri = Triangle::from_csv(
            "year,age,paid,\n2020,12,1,\n2020,24,2,\n".as_bytes(),
            &FileLayout::long("year", "age", Vec::<String>::new()),
        )
        .unwrap();
        assert_eq!(tri.columns(), ["paid"]);
        let tri = Triangle::from_csv(
            "year,12,24,,\n2020,1,2,,\n2021,3,,,\n".as_bytes(),
            &FileLayout::wide("year", "paid"),
        )
        .unwrap();
        assert_eq!(tri.development(), [12, 24]);
        assert_eq!(tri.get(0, 0, 0, 1), Some(2.0));
        // Given ages count only the named age columns.
        let tri = Triangle::from_csv(
            "year,first,second,\n2020,1,2,\n".as_bytes(),
            &FileLayout::wide("year", "paid").ages([12, 24]),
        )
        .unwrap();
        assert_eq!(tri.development(), [12, 24]);
        // An unnamed column with values is an error.
        let e = file_error(Triangle::from_csv(
            "year,age,paid,\n2020,12,1,7\n".as_bytes(),
            &FileLayout::long("year", "age", Vec::<String>::new()),
        ));
        assert_eq!(e, "column 4 has values but no name in the header");
    }

    #[test]
    fn two_wide_columns_at_one_age_are_an_error() {
        let e = file_error(Triangle::from_csv(
            "year,first,second\n2020,100,150\n".as_bytes(),
            &FileLayout::wide("year", "paid").ages([12, 12]),
        ));
        assert_eq!(e, "columns \"first\" and \"second\" both hold age 12");
        let e = file_error(Triangle::from_csv(
            "year,12,12.0\n2020,100,150\n".as_bytes(),
            &FileLayout::wide("year", "paid"),
        ));
        assert_eq!(e, "columns \"12\" and \"12.0\" both hold age 12");
    }

    #[test]
    fn old_mac_line_endings_count_lines() {
        let e = file_error(Triangle::from_csv(
            "year,age,paid\r2020,12,1\r2020,24,x\r".as_bytes(),
            &FileLayout::long("year", "age", ["paid"]),
        ));
        assert_eq!(e, "line 3, column \"paid\": \"x\" is not a number");
    }

    #[test]
    fn a_file_with_no_observed_value_names_the_file() {
        let e = file_error(Triangle::from_csv(
            "year,12,24\n2020,,\n".as_bytes(),
            &FileLayout::wide("year", "paid"),
        ));
        assert_eq!(e, "the file has no observed values");
        let dir = std::env::temp_dir().join(format!("prospicio-io-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blank.csv");
        std::fs::write(&path, "year,12,24\n2020,NA,\n").unwrap();
        let e = file_error(Triangle::read_csv(&path, &FileLayout::wide("year", "paid")));
        assert!(
            e.contains("blank.csv") && e.ends_with("the file has no observed values"),
            "{e}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
