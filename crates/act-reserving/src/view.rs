//! Views for reading a triangle (decision 5 of `docs/design/triangle.md`):
//! one segment and measure as an origin × development grid, one summary row
//! per segment and measure, and the text and HTML printouts the Python and
//! R bindings show, so both print the same numbers.

use std::fmt::{self, Write as _};

use act_core::{Lag, Month, Period};

use crate::error::{Error, Result};
use crate::segments::find_segment;
use crate::triangle::{Label, Triangle};

/// Origins (rows) a printout shows before it truncates.
pub const MAX_ROWS: usize = 20;
/// Development ages (columns) a printout shows before it truncates.
pub const MAX_COLS: usize = 12;

/// One segment and measure of a triangle as an origin × development grid.
#[derive(Debug, Clone, PartialEq)]
pub struct TriangleView {
    /// Key names of the triangle.
    pub key_names: Vec<String>,
    /// The segment's label, one part per key.
    pub label: Label,
    /// The measure column.
    pub column: String,
    /// Origin periods, oldest first.
    pub origins: Vec<Period>,
    /// Development ages in months, youngest first.
    pub development: Vec<Lag>,
    /// Values row-major over origin × development, `None` where not
    /// observed.
    pub values: Vec<Option<f64>>,
    /// Whether the values are cumulative.
    pub cumulative: bool,
    /// Valuation month of the triangle.
    pub valuation: Month,
}

impl TriangleView {
    /// The value at `origin`, `dev`, if observed.
    pub fn get(&self, origin: usize, dev: usize) -> Option<f64> {
        if origin >= self.origins.len() || dev >= self.development.len() {
            return None;
        }
        self.values[origin * self.development.len() + dev]
    }

    /// The first line of the printout: measure, segment and form.
    fn title(&self) -> String {
        let mut out = format!("Triangle: {}", self.column);
        for (k, v) in self.key_names.iter().zip(self.label.parts()) {
            write!(out, ", {k}={v}").unwrap();
        }
        write!(
            out,
            " ({}, valuation {})",
            form(self.cumulative),
            self.valuation
        )
        .unwrap();
        out
    }

    /// The header row, origin labels and cells as text, with at most
    /// `max_rows` origins and `max_cols` ages (0 for no limit), and a note
    /// when some are left out.
    fn cells(&self, max_rows: usize, max_cols: usize) -> (Grid, Option<String>) {
        let rows = shown(self.origins.len(), max_rows);
        let cols = shown(self.development.len(), max_cols);
        let decimals = decimals(self.values.iter().flatten().copied());
        let mut header = vec![String::new()];
        header.extend(cols.iter().map(|c| match c {
            Some(d) => self.development[*d].to_string(),
            None => ELLIPSIS.to_string(),
        }));
        let body = rows
            .iter()
            .map(|r| {
                let Some(o) = *r else {
                    return vec![ELLIPSIS.to_string(); cols.len() + 1];
                };
                let mut row = vec![self.origins[o].to_string()];
                row.extend(cols.iter().map(|c| {
                    match c {
                        Some(d) => self
                            .get(o, *d)
                            .map_or_else(String::new, |v| number(v, decimals)),
                        None => ELLIPSIS.to_string(),
                    }
                }));
                row
            })
            .collect();
        let truncated = rows.len() < self.origins.len() || cols.len() < self.development.len();
        let note = truncated.then(|| {
            format!(
                "[{} origins x {} ages]",
                self.origins.len(),
                self.development.len()
            )
        });
        (
            Grid {
                header,
                body,
                left: 1,
            },
            note,
        )
    }

    /// The grid as text: a title line, then the origins down and the ages
    /// across, blank where a cell is not observed. Long or wide grids show
    /// their first and last `max_rows` origins and `max_cols` ages around
    /// a `...` (0 for no limit).
    pub fn to_text(&self, max_rows: usize, max_cols: usize) -> String {
        let (grid, note) = self.cells(max_rows, max_cols);
        let mut out = self.title();
        out.push('\n');
        out.push_str(&grid.text());
        if let Some(note) = note {
            out.push('\n');
            out.push_str(&note);
        }
        out
    }

    /// The grid as an HTML table, truncated as [`to_text`](Self::to_text).
    pub fn to_html(&self, max_rows: usize, max_cols: usize) -> String {
        let (grid, note) = self.cells(max_rows, max_cols);
        grid.html(&self.title(), note.as_deref())
    }
}

impl fmt::Display for TriangleView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text(MAX_ROWS, MAX_COLS))
    }
}

/// One segment and measure of a triangle, summarised.
#[derive(Debug, Clone, PartialEq)]
pub struct SummaryRow {
    /// The segment's label, one part per key.
    pub label: Label,
    /// The measure column.
    pub column: String,
    /// Origins with at least one observed value.
    pub n_origins: usize,
    /// The oldest of those origins, `None` if there are none.
    pub first_origin: Option<Period>,
    /// The latest of those origins.
    pub last_origin: Option<Period>,
    /// The latest valuation month with an observed value.
    pub valuation: Option<Month>,
    /// Sum over origins of the latest cumulative value: the latest
    /// diagonal's total for a cumulative triangle, the sum of every
    /// increment for an incremental one; 0 with nothing observed.
    pub latest: f64,
}

/// One [`SummaryRow`] per segment and measure of a triangle, segments in
/// index order and measures in column order within each.
#[derive(Debug, Clone, PartialEq)]
pub struct TriangleSummary {
    /// Key names of the triangle.
    pub key_names: Vec<String>,
    /// Number of segments.
    pub n_segments: usize,
    /// Number of measure columns.
    pub n_columns: usize,
    /// Whether the values are cumulative.
    pub cumulative: bool,
    /// Valuation month of the triangle.
    pub valuation: Month,
    /// The rows.
    pub rows: Vec<SummaryRow>,
}

impl TriangleSummary {
    fn title(&self) -> String {
        let plural = |n: usize, what: &str| format!("{n} {what}{}", if n == 1 { "" } else { "s" });
        let mut out = format!(
            "Triangle: {} x {}",
            plural(self.n_segments, "segment"),
            plural(self.n_columns, "column")
        );
        if !self.key_names.is_empty() {
            write!(out, ", keys {}", self.key_names.join(", ")).unwrap();
        }
        write!(
            out,
            " ({}, valuation {})",
            form(self.cumulative),
            self.valuation
        )
        .unwrap();
        out
    }

    fn cells(&self, max_rows: usize) -> (Grid, Option<String>) {
        let mut header = self.key_names.clone();
        header.extend(
            [
                "column",
                "n_origins",
                "first_origin",
                "last_origin",
                "valuation",
                "latest",
            ]
            .map(String::from),
        );
        let decimals = decimals(self.rows.iter().map(|r| r.latest));
        let opt = |p: Option<String>| p.unwrap_or_default();
        let width = header.len();
        let rows = shown(self.rows.len(), max_rows);
        let body = rows
            .iter()
            .map(|r| {
                let Some(r) = *r else {
                    return vec![ELLIPSIS.to_string(); width];
                };
                let row = &self.rows[r];
                let mut cells: Vec<String> = row.label.parts().to_vec();
                cells.extend([
                    row.column.clone(),
                    row.n_origins.to_string(),
                    opt(row.first_origin.map(|p| p.to_string())),
                    opt(row.last_origin.map(|p| p.to_string())),
                    opt(row.valuation.map(|m| m.to_string())),
                    number(row.latest, decimals),
                ]);
                cells
            })
            .collect();
        let note = (rows.len() < self.rows.len()).then(|| format!("[{} rows]", self.rows.len()));
        (
            Grid {
                header,
                body,
                left: self.key_names.len() + 1,
            },
            note,
        )
    }

    /// The summary as text: a title line, then a table with one row per
    /// segment and measure, showing the first and last `max_rows` rows
    /// around a `...` when there are more (0 for no limit).
    pub fn to_text(&self, max_rows: usize) -> String {
        let (grid, note) = self.cells(max_rows);
        let mut out = self.title();
        out.push('\n');
        out.push_str(&grid.text());
        if let Some(note) = note {
            out.push('\n');
            out.push_str(&note);
        }
        out
    }

    /// The summary as an HTML table, truncated as
    /// [`to_text`](Self::to_text).
    pub fn to_html(&self, max_rows: usize) -> String {
        let (grid, note) = self.cells(max_rows);
        grid.html(&self.title(), note.as_deref())
    }
}

impl fmt::Display for TriangleSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text(MAX_ROWS))
    }
}

impl Triangle {
    /// One segment and measure as an origin × development grid.
    ///
    /// `keys` choose the segment as `(key, value)` pairs; keys not named
    /// may take any value, so a triangle with one segment needs none.
    /// `column` names the measure, and may be `None` when there is only
    /// one.
    ///
    /// Errors: an unknown or repeated key, a value no segment has, a choice
    /// that matches several segments or none, an unknown column, or no
    /// column named when there are several.
    ///
    /// ```
    /// use act_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
    ///
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[("lob", &["Auto", "Auto", "Home"])],
    ///     origin: &[2020, 2020, 2021].map(Month::january),
    ///     development: DevelopmentColumn::Age(&[12, 24, 12]),
    ///     values: &[("paid", &[100.0, 150.0, 40.0])],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })
    /// .unwrap();
    /// let auto = tri.view(&[("lob", "Auto")], None).unwrap();
    /// assert_eq!(auto.get(0, 1), Some(150.0));
    /// assert_eq!(auto.get(1, 0), None);
    /// assert!(auto.to_string().starts_with("Triangle: paid, lob=Auto"));
    /// assert!(tri.view(&[], None).is_err()); // two segments
    /// ```
    pub fn view(&self, keys: &[(&str, &str)], column: Option<&str>) -> Result<TriangleView> {
        let index = find_segment(self.key_names(), self.index(), keys)?;
        let c = match column {
            Some(name) => self.column_position(name)?,
            None if self.columns().len() == 1 => 0,
            None => return Err(Error::MultipleColumns(self.columns().len())),
        };
        let [_, _, no, nd] = self.shape();
        let values = (0..no)
            .flat_map(|o| (0..nd).map(move |d| (o, d)))
            .map(|(o, d)| self.get(index, c, o, d))
            .collect();
        Ok(TriangleView {
            key_names: self.key_names().to_vec(),
            label: self.index()[index].clone(),
            column: self.columns()[c].clone(),
            origins: self.origins(),
            development: self.development().to_vec(),
            values,
            cumulative: self.is_cumulative(),
            valuation: self.valuation(),
        })
    }

    /// One row per segment and measure: its origins with an observed value
    /// (how many, the first and the last), its latest valuation and its
    /// latest cumulative total.
    ///
    /// ```
    /// use act_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
    ///
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[("lob", &["Auto", "Auto", "Auto"])],
    ///     origin: &[2020, 2020, 2021].map(Month::january),
    ///     development: DevelopmentColumn::Age(&[12, 24, 12]),
    ///     values: &[("paid", &[100.0, 150.0, 40.0])],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })
    /// .unwrap();
    /// let rows = tri.summary().rows;
    /// assert_eq!(rows[0].n_origins, 2);
    /// assert_eq!(rows[0].latest, 190.0);
    /// ```
    pub fn summary(&self) -> TriangleSummary {
        let [ni, nc, _, nd] = self.shape();
        let origins = self.origins();
        let mut rows = Vec::with_capacity(ni * nc);
        for i in 0..ni {
            for c in 0..nc {
                let mut row = SummaryRow {
                    label: self.index()[i].clone(),
                    column: self.columns()[c].clone(),
                    n_origins: 0,
                    first_origin: None,
                    last_origin: None,
                    valuation: None,
                    latest: 0.0,
                };
                for (o, &origin) in origins.iter().enumerate() {
                    let observed: Vec<(usize, f64)> = (0..nd)
                        .filter_map(|d| Some((d, self.get(i, c, o, d)?)))
                        .collect();
                    let Some(&(last, value)) = observed.last() else {
                        continue;
                    };
                    row.n_origins += 1;
                    row.first_origin.get_or_insert(origin);
                    row.last_origin = Some(origin);
                    let valuation = self.valuation_of(o, last);
                    row.valuation = Some(row.valuation.map_or(valuation, |v| v.max(valuation)));
                    row.latest += if self.is_cumulative() {
                        value
                    } else {
                        observed.iter().map(|&(_, v)| v).sum()
                    };
                }
                rows.push(row);
            }
        }
        TriangleSummary {
            key_names: self.key_names().to_vec(),
            n_segments: ni,
            n_columns: nc,
            cumulative: self.is_cumulative(),
            valuation: self.valuation(),
            rows,
        }
    }

    /// The printout: the grid of [`view`](Self::view) for a triangle with
    /// one segment and one measure, otherwise the [`summary`](Self::summary)
    /// table; truncated to `max_rows` rows and `max_cols` ages (0 for no
    /// limit).
    pub fn to_text(&self, max_rows: usize, max_cols: usize) -> String {
        match self.single_view() {
            Some(view) => view.to_text(max_rows, max_cols),
            None => self.summary().to_text(max_rows),
        }
    }

    /// The printout of [`to_text`](Self::to_text) as an HTML table.
    pub fn to_html(&self, max_rows: usize, max_cols: usize) -> String {
        match self.single_view() {
            Some(view) => view.to_html(max_rows, max_cols),
            None => self.summary().to_html(max_rows),
        }
    }

    fn single_view(&self) -> Option<TriangleView> {
        let [ni, nc, _, _] = self.shape();
        (ni == 1 && nc == 1).then(|| self.view(&[], None).expect("one segment and column"))
    }
}

impl fmt::Display for Triangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text(MAX_ROWS, MAX_COLS))
    }
}

const ELLIPSIS: &str = "...";

fn form(cumulative: bool) -> &'static str {
    if cumulative {
        "cumulative"
    } else {
        "incremental"
    }
}

/// Positions shown out of `n` with at most `max` (0 for no limit): all of
/// them, or the first and last halves around a `None` for the gap.
fn shown(n: usize, max: usize) -> Vec<Option<usize>> {
    if max == 0 || n <= max {
        return (0..n).map(Some).collect();
    }
    let head = max.div_ceil(2);
    let tail = max / 2;
    (0..head)
        .map(Some)
        .chain(std::iter::once(None))
        .chain((n - tail..n).map(Some))
        .collect()
}

/// Decimals to print a set of values with: none when every value is whole,
/// otherwise by the size of a typical (median non-zero) value, so amounts
/// print as whole numbers and link ratios with three decimals.
fn decimals(values: impl Iterator<Item = f64>) -> usize {
    let mut sizes: Vec<f64> = Vec::new();
    let mut whole = true;
    for v in values {
        whole &= v.fract() == 0.0;
        if v != 0.0 {
            sizes.push(v.abs());
        }
    }
    if whole || sizes.is_empty() {
        return 0;
    }
    sizes.sort_by(f64::total_cmp);
    match sizes[sizes.len() / 2] {
        x if x >= 1000.0 => 0,
        x if x >= 10.0 => 2,
        x if x >= 1.0 => 3,
        x if x >= 0.01 => 4,
        _ => 6,
    }
}

/// `value` with `decimals` decimals and thousands separators.
fn number(value: f64, decimals: usize) -> String {
    let text = format!("{:.*}", decimals, value.abs());
    let (int, frac) = match text.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (text.as_str(), None),
    };
    let mut out = String::new();
    // A value that rounds to zero prints without a sign.
    if value < 0.0 && text.chars().any(|c| ('1'..='9').contains(&c)) {
        out.push('-');
    }
    for (k, ch) in int.chars().enumerate() {
        if k > 0 && (int.len() - k) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    if let Some(frac) = frac {
        out.push('.');
        out.push_str(frac);
    }
    out
}

/// A table of text cells: the first `left` columns are labels (left
/// aligned), the rest numbers (right aligned).
struct Grid {
    header: Vec<String>,
    body: Vec<Vec<String>>,
    left: usize,
}

impl Grid {
    fn text(&self) -> String {
        let width = |s: &String| s.chars().count();
        let widths: Vec<usize> = (0..self.header.len())
            .map(|j| {
                std::iter::once(&self.header)
                    .chain(&self.body)
                    .map(|row| width(&row[j]))
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let line = |row: &[String]| {
            let mut out = String::new();
            for (j, cell) in row.iter().enumerate() {
                if j > 0 {
                    out.push_str("  ");
                }
                let pad = " ".repeat(widths[j] - width(cell));
                if j < self.left {
                    out.push_str(cell);
                    out.push_str(&pad);
                } else {
                    out.push_str(&pad);
                    out.push_str(cell);
                }
            }
            out.trim_end().to_string()
        };
        std::iter::once(line(&self.header))
            .chain(self.body.iter().map(|row| line(row)))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn html(&self, title: &str, note: Option<&str>) -> String {
        let mut out = String::from("<table class=\"actuarialrs-triangle\">\n<caption>");
        out.push_str(&escape(title));
        if let Some(note) = note {
            out.push(' ');
            out.push_str(&escape(note));
        }
        out.push_str("</caption>\n<thead><tr>");
        for cell in &self.header {
            write!(out, "<th>{}</th>", escape(cell)).unwrap();
        }
        out.push_str("</tr></thead>\n<tbody>\n");
        for row in &self.body {
            out.push_str("<tr>");
            for (j, cell) in row.iter().enumerate() {
                if j < self.left {
                    write!(out, "<th style=\"text-align: left\">{}</th>", escape(cell)).unwrap();
                } else {
                    write!(out, "<td style=\"text-align: right\">{}</td>", escape(cell)).unwrap();
                }
            }
            out.push_str("</tr>\n");
        }
        out.push_str("</tbody>\n</table>");
        out
    }
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::{RAA, annual};
    use crate::{DevelopmentColumn, Grain, Long};

    /// Two lobs × two states, paid and incurred, origins 2020-2021.
    fn two_keys() -> Triangle {
        let lob = ["Auto", "Auto", "Auto", "Auto", "Home", "Home", "Home"];
        let state = ["CA", "CA", "CA", "NY", "CA", "NY", "NY"];
        let origin = [2020, 2020, 2021, 2020, 2020, 2020, 2021].map(Month::january);
        let paid = [100.0, 150.0, 110.0, 50.0, 30.0, 20.0, f64::NAN];
        let incurred = [120.0, 160.0, 130.0, 60.0, 35.0, 25.0, 15.0];
        Triangle::from_long(&Long {
            keys: &[("lob", &lob), ("state", &state)],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 12, 12, 12, 12, 12]),
            values: &[("paid", &paid), ("incurred", &incurred)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn view_matches_get() {
        let tri = two_keys();
        let [ni, nc, no, nd] = tri.shape();
        for i in 0..ni {
            let parts = tri.index()[i].parts();
            let keys = [("lob", parts[0].as_str()), ("state", parts[1].as_str())];
            for c in 0..nc {
                let v = tri.view(&keys, Some(&tri.columns()[c])).unwrap();
                assert_eq!(v.label, tri.index()[i]);
                for o in 0..no {
                    for d in 0..nd {
                        assert_eq!(v.get(o, d), tri.get(i, c, o, d));
                    }
                }
            }
        }
    }

    #[test]
    fn view_errors() {
        let tri = two_keys();
        assert_eq!(
            tri.view(&[("lob", "Auto")], Some("paid")),
            Err(Error::AmbiguousSegment(2))
        );
        assert_eq!(
            tri.view(&[("lob", "Auto"), ("state", "CA")], None),
            Err(Error::MultipleColumns(2))
        );
        assert_eq!(
            tri.view(&[("lob", "Auto"), ("state", "CA")], Some("x")),
            Err(Error::UnknownColumn("x".into()))
        );
        assert!(matches!(
            tri.view(&[("lob", "Boat")], Some("paid")),
            Err(Error::UnknownKeyValue { .. })
        ));
        assert!(matches!(
            tri.view(&[("line", "Auto")], Some("paid")),
            Err(Error::UnknownKey(_))
        ));
    }

    #[test]
    fn summary_rows_of_two_keys_and_two_measures() {
        let tri = two_keys();
        let s = tri.summary();
        assert_eq!((s.n_segments, s.n_columns), (4, 2));
        let got: Vec<(String, &str, usize, f64)> = s
            .rows
            .iter()
            .map(|r| {
                (
                    r.label.to_string(),
                    r.column.as_str(),
                    r.n_origins,
                    r.latest,
                )
            })
            .collect();
        assert_eq!(
            got,
            [
                ("Auto / CA".into(), "paid", 2, 260.0),
                ("Auto / CA".into(), "incurred", 2, 290.0),
                ("Auto / NY".into(), "paid", 1, 50.0),
                ("Auto / NY".into(), "incurred", 1, 60.0),
                ("Home / CA".into(), "paid", 1, 30.0),
                ("Home / CA".into(), "incurred", 1, 35.0),
                ("Home / NY".into(), "paid", 1, 20.0),
                ("Home / NY".into(), "incurred", 2, 40.0),
            ]
        );
        let home_ny_paid = &s.rows[6];
        assert_eq!(home_ny_paid.first_origin, Some(Period::year(2020)));
        assert_eq!(home_ny_paid.last_origin, Some(Period::year(2020)));
        assert_eq!(home_ny_paid.valuation, Some(Month::new(2020, 12).unwrap()));
        assert_eq!(s.rows[1].valuation, Some(Month::new(2021, 12).unwrap()));
        // The latest total is the latest diagonal's sum.
        let diagonal = tri.latest_diagonal();
        for (k, row) in s.rows.iter().enumerate() {
            let total: f64 = diagonal
                .values(k / 2, k % 2)
                .iter()
                .filter(|v| !v.is_nan())
                .sum();
            assert_eq!(row.latest, total);
        }
    }

    #[test]
    fn incremental_summary_sums_increments() {
        let tri = two_keys();
        let inc = tri.to_incremental().summary();
        let cum = tri.summary();
        for (a, b) in inc.rows.iter().zip(&cum.rows) {
            assert!((a.latest - b.latest).abs() < 1e-9);
            assert_eq!(a.n_origins, b.n_origins);
        }
        assert!(!inc.cumulative);
        assert!(inc.to_text(0).contains("incremental"));
    }

    #[test]
    fn printout_of_one_segment_is_the_grid() {
        let raa = annual(1981, &RAA);
        let text = raa.to_string();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "Triangle: values (cumulative, valuation 1990-12)");
        assert_eq!(lines.len(), 12);
        let header: Vec<&str> = lines[1].split_whitespace().collect();
        assert_eq!(header[0], "12");
        assert_eq!(header[9], "120");
        let first: Vec<&str> = lines[2].split_whitespace().collect();
        assert_eq!(first[..3], ["1981", "5,012", "8,269"]);
        let last: Vec<&str> = lines[11].split_whitespace().collect();
        assert_eq!(last, ["1990", "2,063"]);
    }

    #[test]
    fn printout_of_several_segments_is_the_summary() {
        let text = two_keys().to_string();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("Triangle: 4 segments x 2 columns, keys lob, state"));
        let header: Vec<&str> = lines[1].split_whitespace().collect();
        assert_eq!(
            header,
            [
                "lob",
                "state",
                "column",
                "n_origins",
                "first_origin",
                "last_origin",
                "valuation",
                "latest"
            ]
        );
        let row: Vec<&str> = lines[2].split_whitespace().collect();
        assert_eq!(
            row,
            ["Auto", "CA", "paid", "2", "2020", "2021", "2021-12", "260"]
        );
        assert_eq!(lines.len(), 10);
    }

    #[test]
    fn large_triangles_are_truncated() {
        let rows: Vec<Vec<f64>> = (0..40)
            .map(|k| (0..40 - k).map(|d| 1.5 * (d + 1) as f64).collect())
            .collect();
        let rows: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let tri = annual(1981, &rows);
        let text = tri.to_text(MAX_ROWS, MAX_COLS);
        let lines: Vec<&str> = text.lines().collect();
        // Title, header, 20 origins, the gap and the note.
        assert_eq!(lines.len(), 1 + 1 + MAX_ROWS + 1 + 1);
        assert_eq!(lines.last(), Some(&"[40 origins x 40 ages]"));
        assert!(lines.iter().any(|l| l.trim_start().starts_with("...")));
        assert_eq!(lines[1].split_whitespace().count(), MAX_COLS + 1);
        assert!(lines[2].contains("1.50"));
        // No limit shows everything.
        assert_eq!(tri.to_text(0, 0).lines().count(), 42);
        assert!(
            tri.to_html(MAX_ROWS, MAX_COLS)
                .contains("[40 origins x 40 ages]")
        );
    }

    #[test]
    fn empty_and_holey_segments_print() {
        // Home has no paid values at all; Auto has a hole at age 24.
        let tri = Triangle::from_long(&Long {
            keys: &[("lob", &["Auto", "Auto", "Home"])],
            origin: &[Month::january(2020); 3],
            development: DevelopmentColumn::Age(&[12, 36, 12]),
            values: &[("paid", &[1.0, 3.0, f64::NAN])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let s = tri.summary();
        assert_eq!(s.rows[1].n_origins, 0);
        assert_eq!(s.rows[1].first_origin, None);
        assert_eq!(s.rows[1].latest, 0.0);
        assert!(tri.to_string().contains("Home"));
        let home = tri.view(&[("lob", "Home")], None).unwrap();
        assert!(home.values.iter().all(Option::is_none));
        assert!(home.to_string().contains("2020"));
        let auto = tri.view(&[("lob", "Auto")], None).unwrap().to_string();
        let row: Vec<&str> = auto.lines().nth(2).unwrap().split_whitespace().collect();
        assert_eq!(row, ["2020", "1", "3"]);
    }

    #[test]
    fn html_escapes_labels() {
        let tri = Triangle::from_long(&Long {
            keys: &[("lob", &["<b>"])],
            origin: &[Month::january(2020)],
            development: DevelopmentColumn::Age(&[12]),
            values: &[("paid", &[1.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let html = tri.to_html(MAX_ROWS, MAX_COLS);
        assert!(html.contains("lob=&lt;b&gt;"));
        assert!(!html.contains("<b>"));
    }

    #[test]
    fn numbers() {
        assert_eq!(number(1234567.0, 0), "1,234,567");
        assert_eq!(number(-1234.5, 2), "-1,234.50");
        assert_eq!(number(-0.0001, 2), "0.00");
        assert_eq!(number(999.0, 0), "999");
        assert_eq!(decimals([1.0, 2.0].into_iter()), 0);
        assert_eq!(decimals([1.25, 2.0].into_iter()), 3);
        assert_eq!(decimals([0.5, 0.25].into_iter()), 4);
        assert_eq!(decimals([1.5, 1.2, 40.0].into_iter()), 3);
        assert_eq!(decimals([12.5].into_iter()), 2);
        assert_eq!(decimals([1234.5].into_iter()), 0);
        assert_eq!(decimals(std::iter::empty()), 0);
    }
}
