//! The loss triangle: index × column × origin × development, dense and masked.
//!
//! Axes and semantics follow chainladder-python. See
//! `docs/design/triangle.md`.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{Error, Result};
use act_core::{Grain, Lag, Month, Period};

/// A position on the index axis (a segment such as a company or line of
/// business): one value per key of the triangle, in key order. A triangle
/// without keys has one segment with an empty label, displayed as `Total`.
///
/// ```
/// use act_reserving::Label;
///
/// let l = Label::new(["Auto", "CA"]);
/// assert_eq!(l.parts(), ["Auto", "CA"]);
/// assert_eq!(l.to_string(), "Auto / CA");
/// assert!(Label::default().parts().is_empty());
/// assert_eq!(Label::default().to_string(), "Total");
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label(Vec<String>);

impl Label {
    /// A label with the given parts.
    pub fn new<S: Into<String>>(parts: impl IntoIterator<Item = S>) -> Self {
        Self(parts.into_iter().map(Into::into).collect())
    }

    /// The label's values, one per key, in key order.
    pub fn parts(&self) -> &[String] {
        &self.0
    }
}

impl From<&str> for Label {
    fn from(s: &str) -> Self {
        Self(vec![s.to_string()])
    }
}

impl From<String> for Label {
    fn from(s: String) -> Self {
        Self(vec![s])
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            f.write_str("Total")
        } else {
            f.write_str(&self.0.join(" / "))
        }
    }
}

/// The development column of a long table: ages, or valuation months.
#[derive(Debug, Clone, Copy)]
pub enum DevelopmentColumn<'a> {
    /// Months from the start of the origin period (12, 24, ...).
    Age(&'a [Lag]),
    /// Valuation month of each row; ages are derived from it.
    Valuation(&'a [Month]),
}

impl DevelopmentColumn<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Age(a) => a.len(),
            Self::Valuation(v) => v.len(),
        }
    }
}

/// A long table borrowed from the caller: one row per (keys, origin,
/// development) with one or more measure columns. This is the shape of a
/// claims extract and of the Arrow tables the bindings pass in.
#[derive(Debug, Clone, Copy)]
pub struct Long<'a> {
    /// Key columns as `(name, values)`, one value per row, such as
    /// `("lob", ...)` and `("state", ...)`. Each distinct combination of
    /// key values is a segment. No keys (`&[]`) puts every row in one
    /// segment with an empty label.
    pub keys: &'a [(&'a str, &'a [&'a str])],
    /// Any month inside each row's origin period.
    pub origin: &'a [Month],
    /// Development age or valuation of each row.
    pub development: DevelopmentColumn<'a>,
    /// Measure columns as `(name, values)`. NaN marks a missing value.
    pub values: &'a [(&'a str, &'a [f64])],
    /// Length of an origin period.
    pub origin_grain: Grain,
    /// Spacing of development ages. Must divide the origin grain.
    pub development_grain: Grain,
    /// Whether the values are cumulative (otherwise incremental).
    pub cumulative: bool,
}

/// A long table owned by the caller, as returned by [`Triangle::to_long`].
#[derive(Debug, Clone, PartialEq)]
pub struct LongTable {
    /// Key columns as `(name, values)`, in the triangle's key order; empty
    /// for a triangle without keys.
    pub keys: Vec<(String, Vec<String>)>,
    /// Start month of each row's origin period.
    pub origin: Vec<Month>,
    pub development: Vec<Lag>,
    /// Measure columns; NaN where a measure is not observed on a row.
    pub values: Vec<(String, Vec<f64>)>,
}

/// A loss triangle with four axes, in chainladder-python's order:
/// index (segment) × column (measure) × origin × development (age).
///
/// Values are stored densely with a separate observation mask, so a zero is
/// a valid observation and a hole in the middle of a row is representable.
/// The development axis always holds ages in months; valuation dates are
/// derived ([`Triangle::dev_to_val`]).
///
/// ```
/// use act_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
///
/// let origin = [2020, 2020, 2021].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[("lob", &["Auto", "Auto", "Home"])],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 12]),
///     values: &[("paid", &[100.0, 150.0, 110.0])],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })
/// .unwrap();
/// assert_eq!(tri.key_names(), ["lob"]);
/// assert_eq!(tri.shape(), [2, 1, 2, 2]);
/// assert_eq!(tri.index()[1].parts(), ["Home"]);
/// assert_eq!(tri.get(0, 0, 0, 1), Some(150.0));
/// assert_eq!(tri.get(1, 0, 1, 0), Some(110.0));
/// assert_eq!(tri.get(1, 0, 0, 0), None);
/// assert_eq!(tri.valuation(), Month::new(2021, 12).unwrap());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Triangle {
    values: Vec<f64>,
    mask: Vec<bool>,
    shape: [usize; 4],
    /// Names of the key columns; each index label has one part per key.
    keys: Vec<String>,
    index: Vec<Label>,
    columns: Vec<String>,
    /// Start month of each origin period, contiguous at `origin_grain`.
    origins: Vec<Month>,
    origin_grain: Grain,
    /// Ages in months, contiguous at `development_grain`.
    development: Vec<Lag>,
    development_grain: Grain,
    valuation: Month,
    cumulative: bool,
}

impl Triangle {
    /// Builds a triangle from a long table.
    ///
    /// Origins span every period from the earliest to the latest row, and
    /// ages every development period from the youngest to the oldest.
    /// Rows with the same (keys, origin, age) are summed. NaN values are
    /// treated as missing. Key names must differ from each other and from
    /// the measure names.
    ///
    /// Cumulative input: periods with no rows are unobserved. Incremental
    /// input: as in chainladder-python, a missing row is a period without
    /// movement, so in every (index, column, origin) with any value, cells
    /// up to the valuation are zero increments, and ages run to the oldest
    /// origin's age at the valuation.
    pub fn from_long(long: &Long<'_>) -> Result<Self> {
        let n = long.origin.len();
        if n == 0 || long.values.is_empty() {
            return Err(Error::Empty);
        }
        if !long.origin_grain.is_multiple_of(long.development_grain) {
            return Err(Error::InvalidGrain(
                "development grain must divide the origin grain",
            ));
        }
        let check_len = |column: &str, found: usize| {
            if found == n {
                Ok(())
            } else {
                Err(Error::LengthMismatch {
                    column: column.to_string(),
                    expected: n,
                    found,
                })
            }
        };
        let mut keys: Vec<String> = Vec::with_capacity(long.keys.len());
        for (name, values) in long.keys {
            check_len(name, values.len())?;
            if keys.iter().any(|k| k == name) {
                return Err(Error::DuplicateKey(name.to_string()));
            }
            keys.push(name.to_string());
        }
        check_len("development", long.development.len())?;
        let mut columns: Vec<String> = Vec::with_capacity(long.values.len());
        for (name, values) in long.values {
            check_len(name, values.len())?;
            if columns.iter().any(|c| c == name) {
                return Err(Error::DuplicateColumn(name.to_string()));
            }
            if let Some(row) = values.iter().position(|v| v.is_infinite()) {
                return Err(Error::NonFinite {
                    column: name.to_string(),
                    row,
                });
            }
            if keys.iter().any(|k| k == name) {
                return Err(Error::KeyClash(name.to_string()));
            }
            columns.push(name.to_string());
        }

        let starts: Vec<Month> = long
            .origin
            .iter()
            .map(|m| m.floor(long.origin_grain))
            .collect();
        let ages: Vec<i64> = match long.development {
            DevelopmentColumn::Age(ages) => ages.iter().map(|&a| a as i64).collect(),
            DevelopmentColumn::Valuation(vals) => vals
                .iter()
                .zip(&starts)
                .map(|(v, s)| v.months_since(*s) + 1)
                .collect(),
        };
        if let Some(row) = ages.iter().position(|&a| a <= 0) {
            return Err(Error::NonPositiveAge { row });
        }

        let row_labels: Vec<Label> = (0..n)
            .map(|row| Label::new(long.keys.iter().map(|(_, values)| values[row])))
            .collect();
        let mut index = row_labels.clone();
        index.sort();
        index.dedup();

        let first = *starts.iter().min().expect("n > 0");
        let last = *starts.iter().max().expect("n > 0");
        let step = long.origin_grain.months() as i64;
        let n_origins = (last.months_since(first) / step + 1) as usize;
        let origins: Vec<Month> = (0..n_origins)
            .map(|k| first.add_months(k as i64 * step))
            .collect();

        let has_value = |row: usize| long.values.iter().any(|(_, v)| !v[row].is_nan());
        let valuation = (0..n)
            .filter(|&row| has_value(row))
            .map(|row| starts[row].add_months(ages[row] - 1))
            .max()
            .ok_or(Error::Empty)?;

        let youngest = *ages.iter().min().expect("n > 0");
        let mut oldest = *ages.iter().max().expect("n > 0");
        let dev_step = long.development_grain.months() as i64;
        for (row, &age) in ages.iter().enumerate() {
            if (age - youngest) % dev_step != 0 {
                return Err(Error::OffGrid {
                    row,
                    age: age as u32,
                });
            }
        }
        if !long.cumulative {
            // Missing incremental rows are periods without movement, so the
            // grid runs to the oldest origin's age at the valuation.
            let at_valuation = valuation.months_since(first) + 1;
            oldest = oldest.max(youngest + (at_valuation - youngest) / dev_step * dev_step);
        }
        let n_dev = ((oldest - youngest) / dev_step + 1) as usize;
        let development: Vec<Lag> = (0..n_dev)
            .map(|k| (youngest + k as i64 * dev_step) as Lag)
            .collect();

        let shape = [index.len(), columns.len(), n_origins, n_dev];
        let size = shape.iter().product();
        let mut tri = Self {
            values: vec![0.0; size],
            mask: vec![false; size],
            shape,
            keys,
            index,
            columns,
            origins,
            origin_grain: long.origin_grain,
            development,
            development_grain: long.development_grain,
            valuation,
            cumulative: long.cumulative,
        };

        for row in 0..n {
            let i = tri
                .index
                .binary_search(&row_labels[row])
                .expect("label collected above");
            let o = (starts[row].months_since(first) / step) as usize;
            let d = ((ages[row] - youngest) / dev_step) as usize;
            for (c, (_, values)) in long.values.iter().enumerate() {
                let v = values[row];
                if v.is_nan() {
                    continue;
                }
                let at = tri.offset(i, c, o, d);
                tri.values[at] += v;
                tri.mask[at] = true;
            }
        }
        if !long.cumulative {
            tri.fill_missing_increments();
        }
        Ok(tri)
    }

    /// Marks every unobserved cell up to the valuation as a zero increment,
    /// in each (index, column, origin) row that has any observation, as
    /// chainladder-python does: a claims extract has no row for a period
    /// without payments.
    fn fill_missing_increments(&mut self) {
        let [ni, nc, no, nd] = self.shape;
        for i in 0..ni {
            for c in 0..nc {
                for o in 0..no {
                    let row = self.offset(i, c, o, 0);
                    if !self.mask[row..row + nd].contains(&true) {
                        continue;
                    }
                    for d in 0..nd {
                        if !self.mask[row + d] && self.valuation_of(o, d) <= self.valuation {
                            self.mask[row + d] = true;
                        }
                    }
                }
            }
        }
    }

    /// The triangle as a long table: one row per (segment, origin, age) with
    /// at least one observed measure, in axis order, with the segment's key
    /// values as named columns.
    pub fn to_long(&self) -> LongTable {
        let [ni, nc, no, nd] = self.shape;
        let mut out = LongTable {
            keys: self.keys.iter().map(|k| (k.clone(), Vec::new())).collect(),
            origin: Vec::new(),
            development: Vec::new(),
            values: self
                .columns
                .iter()
                .map(|c| (c.clone(), Vec::new()))
                .collect(),
        };
        for i in 0..ni {
            for o in 0..no {
                for d in 0..nd {
                    if !(0..nc).any(|c| self.mask[self.offset(i, c, o, d)]) {
                        continue;
                    }
                    for ((_, column), part) in out.keys.iter_mut().zip(self.index[i].parts()) {
                        column.push(part.clone());
                    }
                    out.origin.push(self.origins[o]);
                    out.development.push(self.development[d]);
                    for (c, (_, column)) in out.values.iter_mut().enumerate() {
                        column.push(self.get(i, c, o, d).unwrap_or(f64::NAN));
                    }
                }
            }
        }
        out
    }

    fn offset(&self, i: usize, c: usize, o: usize, d: usize) -> usize {
        let [_, nc, no, nd] = self.shape;
        ((i * nc + c) * no + o) * nd + d
    }

    /// Axis lengths: `[index, column, origin, development]`.
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    /// Names of the key columns, in the order supplied to
    /// [`Triangle::from_long`]; empty for a triangle without keys.
    pub fn key_names(&self) -> &[String] {
        &self.keys
    }

    /// Labels of the index axis, one part per key, sorted.
    pub fn index(&self) -> &[Label] {
        &self.index
    }

    /// Names of the measure columns, in the order supplied.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Position of the column named `name`.
    pub fn column_position(&self, name: &str) -> Result<usize> {
        self.columns
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| Error::UnknownColumn(name.to_string()))
    }

    /// The origin periods, oldest first.
    pub fn origins(&self) -> Vec<Period> {
        self.origins
            .iter()
            .map(|&m| Period::containing(m, self.origin_grain))
            .collect()
    }

    /// Length of an origin period.
    pub fn origin_grain(&self) -> Grain {
        self.origin_grain
    }

    /// Development ages in months, youngest first.
    pub fn development(&self) -> &[Lag] {
        &self.development
    }

    /// Spacing of the development ages.
    pub fn development_grain(&self) -> Grain {
        self.development_grain
    }

    /// Month of the latest diagonal (the latest observed valuation).
    pub fn valuation(&self) -> Month {
        self.valuation
    }

    /// Whether the values are cumulative (otherwise incremental).
    pub fn is_cumulative(&self) -> bool {
        self.cumulative
    }

    /// The value at the given axis positions, if observed.
    pub fn get(&self, index: usize, column: usize, origin: usize, dev: usize) -> Option<f64> {
        let [ni, nc, no, nd] = self.shape;
        if index >= ni || column >= nc || origin >= no || dev >= nd {
            return None;
        }
        let at = self.offset(index, column, origin, dev);
        self.mask[at].then_some(self.values[at])
    }

    /// Valuation month of the cell at `origin`, `dev`.
    pub fn valuation_of(&self, origin: usize, dev: usize) -> Month {
        self.origins[origin].add_months(self.development[dev] as i64 - 1)
    }

    /// Applies `f` to each (index, column, origin) row of values and mask.
    fn map_rows(&self, mut f: impl FnMut(&mut [f64], &[bool])) -> Self {
        let mut out = self.clone();
        let nd = self.shape[3];
        for (values, mask) in out
            .values
            .chunks_exact_mut(nd)
            .zip(self.mask.chunks_exact(nd))
        {
            f(values, mask);
        }
        out
    }

    /// Incremental values: each observed value minus the previous observed
    /// value in its row. Unobserved cells stay unobserved, so the increment
    /// after a hole covers the whole gap.
    pub fn to_incremental(&self) -> Self {
        if !self.cumulative {
            return self.clone();
        }
        let mut out = self.map_rows(|values, mask| {
            let mut previous = 0.0;
            for (v, &seen) in values.iter_mut().zip(mask) {
                if seen {
                    let current = *v;
                    *v -= previous;
                    previous = current;
                }
            }
        });
        out.cumulative = false;
        out
    }

    /// Cumulative values: running sums of the observed increments.
    pub fn to_cumulative(&self) -> Self {
        if self.cumulative {
            return self.clone();
        }
        let mut out = self.map_rows(|values, mask| {
            let mut total = 0.0;
            for (v, &seen) in values.iter_mut().zip(mask) {
                if seen {
                    total += *v;
                    *v = total;
                }
            }
        });
        out.cumulative = true;
        out
    }

    /// The latest observed value of every (index, column, origin).
    pub fn latest_diagonal(&self) -> Diagonal {
        let nd = self.shape[3];
        let cells = self
            .values
            .chunks_exact(nd)
            .zip(self.mask.chunks_exact(nd))
            .map(|(values, mask)| {
                let d = mask.iter().rposition(|&seen| seen)?;
                Some((self.development[d], values[d]))
            })
            .collect();
        Diagonal {
            cells,
            shape: [self.shape[0], self.shape[1], self.shape[2]],
        }
    }

    /// Age-to-age link ratios of the cumulative values. Development position
    /// `d` holds the ratio from age `d` to age `d + 1`, observed only where
    /// both ages are observed and the earlier value is non-zero. The result
    /// is marked as not cumulative.
    pub fn link_ratios(&self) -> Self {
        let cum = self.to_cumulative();
        let [ni, nc, no, nd] = self.shape;
        let nl = nd.saturating_sub(1);
        let mut out = Self {
            values: vec![0.0; ni * nc * no * nl],
            mask: vec![false; ni * nc * no * nl],
            shape: [ni, nc, no, nl],
            development: self.development[..nl].to_vec(),
            cumulative: false,
            ..self.clone()
        };
        for i in 0..ni {
            for c in 0..nc {
                for o in 0..no {
                    for d in 0..nl {
                        let (Some(from), Some(to)) = (cum.get(i, c, o, d), cum.get(i, c, o, d + 1))
                        else {
                            continue;
                        };
                        if from != 0.0 {
                            let at = out.offset(i, c, o, d);
                            out.values[at] = to / from;
                            out.mask[at] = true;
                        }
                    }
                }
            }
        }
        out
    }

    /// The segments whose key values satisfy every condition: each
    /// `(key, values)` keeps segments whose value of `key` is one of
    /// `values`, and conditions combine with AND. Segments keep their order;
    /// no conditions keep every segment.
    ///
    /// Errors: a key that does not exist or is named twice, a value that no
    /// segment has or that is given twice, an empty value list, and a
    /// selection that matches no segment.
    ///
    /// ```
    /// use act_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
    ///
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[("lob", &["Auto", "Auto", "Home"]), ("state", &["CA", "NY", "NY"])],
    ///     origin: &[Month::january(2020); 3],
    ///     development: DevelopmentColumn::Age(&[12, 12, 12]),
    ///     values: &[("paid", &[1.0, 2.0, 3.0])],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })
    /// .unwrap();
    /// let ny = tri.select(&[("state", &["NY"])]).unwrap();
    /// assert_eq!(ny.shape()[0], 2);
    /// let auto_ny = tri.select(&[("lob", &["Auto"]), ("state", &["NY"])]).unwrap();
    /// assert_eq!(auto_ny.get(0, 0, 0, 0), Some(2.0));
    /// ```
    pub fn select(&self, conditions: &[(&str, &[&str])]) -> Result<Self> {
        let mut seen: Vec<usize> = Vec::with_capacity(conditions.len());
        for (key, values) in conditions {
            let k = self.key_position(key)?;
            if seen.contains(&k) {
                return Err(Error::DuplicateKey(key.to_string()));
            }
            seen.push(k);
            if values.is_empty() {
                return Err(Error::EmptySelection);
            }
            for (n, value) in values.iter().enumerate() {
                if values[..n].contains(value) {
                    return Err(Error::DuplicateKeyValue {
                        key: key.to_string(),
                        value: value.to_string(),
                    });
                }
                if !self.index.iter().any(|l| l.parts()[k] == *value) {
                    return Err(Error::UnknownKeyValue {
                        key: key.to_string(),
                        value: value.to_string(),
                    });
                }
            }
        }
        let index_pos: Vec<usize> = (0..self.shape[0])
            .filter(|&i| {
                let parts = self.index[i].parts();
                conditions
                    .iter()
                    .zip(&seen)
                    .all(|((_, values), &k)| values.contains(&parts[k].as_str()))
            })
            .collect();
        if index_pos.is_empty() {
            return Err(Error::NoSegments);
        }
        let column_pos: Vec<usize> = (0..self.shape[1]).collect();
        Ok(self.take(&index_pos, &column_pos))
    }

    /// The named measure columns, in the order given. Naming a column twice,
    /// one that does not exist, or none is an error.
    pub fn select_columns(&self, columns: &[&str]) -> Result<Self> {
        if columns.is_empty() {
            return Err(Error::EmptySelection);
        }
        let column_pos: Vec<usize> = columns
            .iter()
            .map(|n| self.column_position(n))
            .collect::<Result<_>>()?;
        if let Some(k) = (1..column_pos.len()).find(|&k| column_pos[..k].contains(&column_pos[k])) {
            return Err(Error::DuplicateColumn(self.columns[column_pos[k]].clone()));
        }
        let index_pos: Vec<usize> = (0..self.shape[0]).collect();
        Ok(self.take(&index_pos, &column_pos))
    }

    /// Sums the segments that share the values of `keys`, dropping the other
    /// keys. The result has `keys` as its keys, in the order given, and its
    /// segments sorted by them; `&[]` sums everything into one segment
    /// without keys.
    ///
    /// Cumulative values are summed cell by cell, and a cell is observed if
    /// any of its members is. An incremental triangle is summed as
    /// cumulative values and returned incremental. Unknown or repeated keys
    /// are an error.
    ///
    /// ```
    /// use act_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
    ///
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[("lob", &["Auto", "Auto", "Home"]), ("state", &["CA", "NY", "NY"])],
    ///     origin: &[Month::january(2020); 3],
    ///     development: DevelopmentColumn::Age(&[12, 12, 12]),
    ///     values: &[("paid", &[1.0, 2.0, 3.0])],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })
    /// .unwrap();
    /// let by_lob = tri.group_by(&["lob"]).unwrap();
    /// assert_eq!(by_lob.key_names(), ["lob"]);
    /// assert_eq!(by_lob.get(0, 0, 0, 0), Some(3.0));
    /// assert_eq!(tri.group_by(&[]).unwrap().get(0, 0, 0, 0), Some(6.0));
    /// ```
    pub fn group_by(&self, keys: &[&str]) -> Result<Self> {
        let mut positions: Vec<usize> = Vec::with_capacity(keys.len());
        for key in keys {
            let k = self.key_position(key)?;
            if positions.contains(&k) {
                return Err(Error::DuplicateKey(key.to_string()));
            }
            positions.push(k);
        }
        let source = self.to_cumulative();
        let mut groups: BTreeMap<Label, Vec<usize>> = BTreeMap::new();
        for (i, label) in self.index.iter().enumerate() {
            let parts = positions.iter().map(|&k| label.parts()[k].clone());
            groups.entry(Label::new(parts)).or_default().push(i);
        }
        let [_, nc, no, nd] = self.shape;
        let block = nc * no * nd;
        let mut values = vec![0.0; groups.len() * block];
        let mut mask = vec![false; groups.len() * block];
        for (g, members) in groups.values().enumerate() {
            for &i in members {
                let from = source.offset(i, 0, 0, 0);
                for at in 0..block {
                    if source.mask[from + at] {
                        values[g * block + at] += source.values[from + at];
                        mask[g * block + at] = true;
                    }
                }
            }
        }
        let out = Self {
            values,
            mask,
            shape: [groups.len(), nc, no, nd],
            keys: keys.iter().map(|k| k.to_string()).collect(),
            index: groups.into_keys().collect(),
            ..source
        };
        Ok(if self.cumulative {
            out
        } else {
            out.to_incremental()
        })
    }

    /// Position of the key column named `name`.
    fn key_position(&self, name: &str) -> Result<usize> {
        self.keys
            .iter()
            .position(|k| k == name)
            .ok_or_else(|| Error::UnknownKey(name.to_string()))
    }

    /// The given index positions and columns, in that order.
    fn take(&self, index_pos: &[usize], column_pos: &[usize]) -> Self {
        let [_, _, no, nd] = self.shape;
        let block = no * nd;
        let mut values = Vec::with_capacity(index_pos.len() * column_pos.len() * block);
        let mut mask = Vec::with_capacity(values.capacity());
        for &i in index_pos {
            for &c in column_pos {
                let at = self.offset(i, c, 0, 0);
                values.extend_from_slice(&self.values[at..at + block]);
                mask.extend_from_slice(&self.mask[at..at + block]);
            }
        }
        Self {
            values,
            mask,
            shape: [index_pos.len(), column_pos.len(), no, nd],
            index: index_pos.iter().map(|&i| self.index[i].clone()).collect(),
            columns: column_pos
                .iter()
                .map(|&c| self.columns[c].clone())
                .collect(),
            ..self.clone()
        }
    }

    /// A calendar view: the development axis as valuation months.
    pub fn dev_to_val(&self) -> CalendarView<'_> {
        let mut valuations: Vec<Month> = (0..self.shape[2])
            .flat_map(|o| (0..self.shape[3]).map(move |d| (o, d)))
            .map(|(o, d)| self.valuation_of(o, d))
            .filter(|&v| v <= self.valuation)
            .collect();
        valuations.sort();
        valuations.dedup();
        CalendarView {
            triangle: self,
            valuations,
        }
    }

    /// Coarsens the origin and development periods, as chainladder-python's
    /// `grain()`. Both new grains must be multiples of the current ones, and
    /// the development grain must divide the origin grain.
    ///
    /// The cumulative value of a new origin at a valuation is the sum of its
    /// sub-origins' increments valued by then, so a sub-origin with no rows
    /// or a later start contributes nothing rather than masking the cell. A
    /// cell is observed if any of those increments is. Valuations are kept
    /// every development period back from the triangle's valuation, so a
    /// partial latest period keeps its exact latest diagonal and ages are
    /// measured from the new origin start.
    pub fn grain(&self, origin_grain: Grain, development_grain: Grain) -> Result<Self> {
        if !origin_grain.is_multiple_of(self.origin_grain) {
            return Err(Error::InvalidGrain(
                "origin grain must be a multiple of the current origin grain",
            ));
        }
        if !development_grain.is_multiple_of(self.development_grain) {
            return Err(Error::InvalidGrain(
                "development grain must be a multiple of the current development grain",
            ));
        }
        let inc = self.to_incremental();
        let [ni, nc, no, nd] = self.shape;
        let dev_step = development_grain.months() as i64;

        // (index, new origin start) -> observed increments as (valuation,
        // column, value).
        type Increments = Vec<(Month, usize, f64)>;
        let mut increments: BTreeMap<(usize, Month), Increments> = BTreeMap::new();
        for i in 0..ni {
            for o in 0..no {
                let new_origin = self.origins[o].floor(origin_grain);
                for c in 0..nc {
                    for d in 0..nd {
                        if let Some(v) = inc.get(i, c, o, d) {
                            increments.entry((i, new_origin)).or_default().push((
                                self.valuation_of(o, d),
                                c,
                                v,
                            ));
                        }
                    }
                }
            }
        }

        let mut index: Vec<usize> = Vec::new();
        let mut origin = Vec::new();
        let mut ages = Vec::new();
        let mut columns: Vec<Vec<f64>> = vec![Vec::new(); nc];
        for ((i, new_origin), cells) in increments {
            let mut valuation = self.valuation;
            while valuation >= new_origin {
                let mut sums = vec![f64::NAN; nc];
                for &(v, c, x) in &cells {
                    if v <= valuation {
                        sums[c] = if sums[c].is_nan() { x } else { sums[c] + x };
                    }
                }
                index.push(i);
                origin.push(new_origin);
                ages.push((valuation.months_since(new_origin) + 1) as Lag);
                for (column, sum) in columns.iter_mut().zip(sums) {
                    column.push(sum);
                }
                valuation = valuation.add_months(-dev_step);
            }
        }
        let values: Vec<(&str, &[f64])> = self
            .columns
            .iter()
            .zip(&columns)
            .map(|(name, v)| (name.as_str(), v.as_slice()))
            .collect();
        let key_values: Vec<Vec<&str>> = (0..self.keys.len())
            .map(|k| {
                index
                    .iter()
                    .map(|&i| self.index[i].parts()[k].as_str())
                    .collect()
            })
            .collect();
        let keys: Vec<(&str, &[&str])> = self
            .keys
            .iter()
            .zip(&key_values)
            .map(|(name, v)| (name.as_str(), v.as_slice()))
            .collect();
        let out = Self::from_long(&Long {
            keys: &keys,
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &values,
            origin_grain,
            development_grain,
            cumulative: true,
        })?;
        Ok(if self.cumulative {
            out
        } else {
            out.to_incremental()
        })
    }

    /// The (origin × development) cells of `column` in the only index
    /// position, as cumulative values. Reserving methods fit on this.
    pub(crate) fn segment(&self, column: &str) -> Result<Segment> {
        if self.shape[0] != 1 {
            return Err(Error::MultipleSegments(self.shape[0]));
        }
        Ok(self.segments(column)?.pop().expect("one index position"))
    }

    /// The cells of `column` in every index position, in index order, as
    /// cumulative values. Each segment is trimmed to the origins and ages
    /// it observes (from its first to its last), so a line that starts
    /// later or has fewer ages than the triangle fits on its own range.
    pub(crate) fn segments(&self, column: &str) -> Result<Vec<Segment>> {
        let c = self.column_position(column)?;
        let cum = self.to_cumulative();
        let [ni, _, no, nd] = self.shape;
        let origins = self.origins();
        (0..ni)
            .map(|i| {
                let seen = |o: usize, d: usize| cum.get(i, c, o, d).is_some();
                let rows: Vec<usize> = (0..no).filter(|&o| (0..nd).any(|d| seen(o, d))).collect();
                let cols: Vec<usize> = (0..nd).filter(|&d| (0..no).any(|o| seen(o, d))).collect();
                let (Some(&o0), Some(&o1), Some(&d0), Some(&d1)) =
                    (rows.first(), rows.last(), cols.first(), cols.last())
                else {
                    return Err(Error::EmptyOrigin(format!(
                        "every origin of {} in {}",
                        column, self.index[i]
                    )));
                };
                Ok(Segment {
                    cells: (o0..=o1)
                        .flat_map(|o| (d0..=d1).map(move |d| (o, d)))
                        .map(|(o, d)| cum.get(i, c, o, d))
                        .collect(),
                    n_origins: o1 - o0 + 1,
                    n_dev: d1 - d0 + 1,
                    origins: origins[o0..=o1].to_vec(),
                    ages: self.development[d0..=d1].to_vec(),
                    origin_offset: o0,
                    dev_offset: d0,
                })
            })
            .collect()
    }
}

/// The latest diagonal: per (index, column, origin), the age and value of
/// the latest observation.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagonal {
    cells: Vec<Option<(Lag, f64)>>,
    shape: [usize; 3],
}

impl Diagonal {
    /// Axis lengths: `[index, column, origin]`.
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Age and value of the latest observation, if the row has any.
    pub fn get(&self, index: usize, column: usize, origin: usize) -> Option<(Lag, f64)> {
        let [ni, nc, no] = self.shape;
        if index >= ni || column >= nc || origin >= no {
            return None;
        }
        self.cells[(index * nc + column) * no + origin]
    }

    /// Latest values of one (index, column), by origin; 0 where a row has
    /// no observation.
    pub fn values(&self, index: usize, column: usize) -> Vec<f64> {
        (0..self.shape[2])
            .map(|o| self.get(index, column, o).map_or(0.0, |(_, v)| v))
            .collect()
    }
}

/// A triangle seen by valuation month instead of age.
#[derive(Debug, Clone)]
pub struct CalendarView<'a> {
    triangle: &'a Triangle,
    valuations: Vec<Month>,
}

impl<'a> CalendarView<'a> {
    /// The distinct valuation months up to the triangle's valuation, oldest
    /// first.
    pub fn valuations(&self) -> &[Month] {
        &self.valuations
    }

    /// Value of `(index, column, origin)` at the valuation in position `val`.
    pub fn get(&self, index: usize, column: usize, origin: usize, val: usize) -> Option<f64> {
        let t = self.triangle;
        let valuation = *self.valuations.get(val)?;
        let start = *t.origins.get(origin)?;
        let age = valuation.months_since(start) + 1;
        let d = t.development.iter().position(|&a| a as i64 == age)?;
        t.get(index, column, origin, d)
    }

    /// Back to the development view.
    pub fn val_to_dev(self) -> &'a Triangle {
        self.triangle
    }
}

/// Cumulative cells of one (index, column), row-major over origin × age.
#[derive(Debug, Clone)]
pub(crate) struct Segment {
    cells: Vec<Option<f64>>,
    pub(crate) n_origins: usize,
    pub(crate) n_dev: usize,
    pub(crate) origins: Vec<Period>,
    /// Ages of the segment's development positions.
    pub(crate) ages: Vec<Lag>,
    /// Triangle origin position of the segment's first origin.
    pub(crate) origin_offset: usize,
    /// Triangle development position of the segment's first age.
    pub(crate) dev_offset: usize,
}

impl Segment {
    pub(crate) fn get(&self, origin: usize, dev: usize) -> Option<f64> {
        self.cells[origin * self.n_dev + dev]
    }

    /// Latest observed development position and value of `origin`.
    pub(crate) fn latest(&self, origin: usize) -> Result<(usize, f64)> {
        (0..self.n_dev)
            .rev()
            .find_map(|d| Some((d, self.get(origin, d)?)))
            .ok_or_else(|| Error::EmptyOrigin(self.origins[origin].to_string()))
    }

    /// `(from, to)` values of every origin observed at both `dev` and
    /// `dev + 1`.
    pub(crate) fn link_pairs(&self, dev: usize) -> Vec<(f64, f64)> {
        (0..self.n_origins)
            .filter_map(|o| Some((self.get(o, dev)?, self.get(o, dev + 1)?)))
            .collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// RAA cumulative triangle (Mack 1993), origins 1981-1990.
    pub(crate) const RAA: [&[f64]; 10] = [
        &[
            5012.0, 8269.0, 10907.0, 11805.0, 13539.0, 16181.0, 18009.0, 18608.0, 18662.0, 18834.0,
        ],
        &[
            106.0, 4285.0, 5396.0, 10666.0, 13782.0, 15599.0, 15496.0, 16169.0, 16704.0,
        ],
        &[
            3410.0, 8992.0, 13873.0, 16141.0, 18735.0, 22214.0, 22863.0, 23466.0,
        ],
        &[5655.0, 11555.0, 15766.0, 21266.0, 23425.0, 26083.0, 27067.0],
        &[1092.0, 9565.0, 15836.0, 22169.0, 25955.0, 26180.0],
        &[1513.0, 6445.0, 11702.0, 12935.0, 15852.0],
        &[557.0, 4020.0, 10946.0, 12314.0],
        &[1351.0, 6947.0, 13112.0],
        &[3133.0, 5395.0],
        &[2063.0],
    ];

    /// An annual triangle from rows of cumulative values, origin years
    /// starting at `first_year`.
    pub(crate) fn annual(first_year: i32, rows: &[&[f64]]) -> Triangle {
        let mut origin = Vec::new();
        let mut ages = Vec::new();
        let mut values = Vec::new();
        for (k, row) in rows.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                origin.push(Month::january(first_year + k as i32));
                ages.push(12 * (d as Lag + 1));
                values.push(v);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("values", &values)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    pub(crate) fn raa() -> Triangle {
        annual(1981, &RAA)
    }

    fn m(year: i32, month: u8) -> Month {
        Month::new(year, month).unwrap()
    }

    #[test]
    fn raa_shape_and_axes() {
        let t = raa();
        assert_eq!(t.shape(), [1, 1, 10, 10]);
        assert_eq!(t.development(), [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]);
        assert_eq!(t.origins()[0], Period::year(1981));
        assert_eq!(t.origins()[9].to_string(), "1990");
        assert_eq!(t.valuation(), m(1990, 12));
        assert_eq!(t.index(), [Label::default()]);
        assert!(t.key_names().is_empty());
        assert_eq!(t.index()[0].to_string(), "Total");
        assert!(t.is_cumulative());
    }

    #[test]
    fn latest_diagonal_of_raa() {
        let diag = raa().latest_diagonal();
        let latest = diag.values(0, 0);
        assert_eq!(latest.iter().sum::<f64>(), 160_987.0);
        assert_eq!(diag.get(0, 0, 0), Some((120, 18834.0)));
        assert_eq!(diag.get(0, 0, 9), Some((12, 2063.0)));
        assert_eq!(diag.get(0, 0, 10), None);
    }

    #[test]
    fn incremental_round_trips_with_holes() {
        let origin = [2020, 2020, 2020, 2021].map(Month::january);
        let t = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 36, 48, 12]),
            values: &[("paid", &[10.0, 25.0, 25.0, 7.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        assert_eq!(t.get(0, 0, 0, 1), None);
        let inc = t.to_incremental();
        assert!(!inc.is_cumulative());
        assert_eq!(inc.get(0, 0, 0, 2), Some(15.0));
        assert_eq!(inc.get(0, 0, 0, 3), Some(0.0));
        assert_eq!(inc.get(0, 0, 0, 1), None);
        assert_eq!(inc.to_cumulative(), t);
        let raa = raa();
        assert_eq!(raa.to_incremental().to_cumulative(), raa);
    }

    #[test]
    fn link_ratios_mask_missing_and_zero() {
        let t = annual(2020, &[&[0.0, 5.0, 10.0], &[4.0, 6.0]]);
        let lr = t.link_ratios();
        assert_eq!(lr.shape(), [1, 1, 2, 2]);
        assert_eq!(lr.development(), [12, 24]);
        assert_eq!(lr.get(0, 0, 0, 0), None);
        assert_eq!(lr.get(0, 0, 0, 1), Some(2.0));
        assert_eq!(lr.get(0, 0, 1, 0), Some(1.5));
        assert_eq!(lr.get(0, 0, 1, 1), None);
    }

    #[test]
    fn from_long_by_valuation_with_segments_and_duplicates() {
        let origin = [m(2020, 5), m(2020, 2), m(2020, 3), m(2020, 1)];
        let valuation = [m(2020, 6), m(2020, 3), m(2020, 3), m(2020, 6)];
        let t = Triangle::from_long(&Long {
            keys: &[
                ("lob", &["B", "A", "A", "A"]),
                ("state", &["x", "y", "y", "y"]),
            ],
            origin: &origin,
            development: DevelopmentColumn::Valuation(&valuation),
            values: &[("paid", &[1.0, 2.0, 3.0, f64::NAN]), ("count", &[1.0; 4])],
            origin_grain: Grain::Quarter,
            development_grain: Grain::Quarter,
            cumulative: true,
        })
        .unwrap();
        assert_eq!(t.shape(), [2, 2, 2, 2]);
        assert_eq!(t.index()[0], Label::new(["A", "y"]));
        assert_eq!(t.development(), [3, 6]);
        assert_eq!(t.origins()[1].to_string(), "2020Q2");
        // Two rows at 2020Q1, age 3 are summed.
        assert_eq!(t.get(0, 0, 0, 0), Some(5.0));
        // NaN paid is missing, but the row's count is observed.
        assert_eq!(t.get(0, 0, 0, 1), None);
        assert_eq!(t.get(0, 1, 0, 1), Some(1.0));
        assert_eq!(t.get(1, 0, 1, 0), Some(1.0));
        assert_eq!(t.valuation(), m(2020, 6));
    }

    /// Rebuilds a triangle from its own long table.
    fn from_long_table(long: &LongTable, grain: Grain) -> Result<Triangle> {
        let key_values: Vec<Vec<&str>> = long
            .keys
            .iter()
            .map(|(_, v)| v.iter().map(String::as_str).collect())
            .collect();
        let keys: Vec<(&str, &[&str])> = long
            .keys
            .iter()
            .zip(&key_values)
            .map(|((n, _), v)| (n.as_str(), v.as_slice()))
            .collect();
        let values: Vec<(&str, &[f64])> = long
            .values
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        Triangle::from_long(&Long {
            keys: &keys,
            origin: &long.origin,
            development: DevelopmentColumn::Age(&long.development),
            values: &values,
            origin_grain: grain,
            development_grain: grain,
            cumulative: true,
        })
    }

    #[test]
    fn to_long_round_trips() {
        let t = raa();
        let long = t.to_long();
        assert_eq!(long.origin.len(), 55);
        assert!(long.keys.is_empty());
        assert_eq!(from_long_table(&long, Grain::Year).unwrap(), t);
    }

    /// Two lines by two states, two measures: paid and incurred.
    fn multi_key() -> Triangle {
        let origin = [2020, 2020, 2021, 2020, 2020, 2021].map(Month::january);
        Triangle::from_long(&Long {
            keys: &[
                ("lob", &["Home", "Home", "Home", "Auto", "Auto", "Auto"]),
                ("state", &["NY", "NY", "NY", "CA", "CA", "TX"]),
            ],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 12, 12, 24, 12]),
            values: &[
                ("paid", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ("incurred", &[2.0, 3.0, 4.0, 5.0, 6.0, f64::NAN]),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn named_keys_round_trip() {
        let t = multi_key();
        assert_eq!(t.key_names(), ["lob", "state"]);
        assert_eq!(
            t.index(),
            [
                Label::new(["Auto", "CA"]),
                Label::new(["Auto", "TX"]),
                Label::new(["Home", "NY"])
            ]
        );
        assert_eq!(t.shape(), [3, 2, 2, 2]);
        assert_eq!(t.get(1, 0, 1, 0), Some(6.0));
        assert_eq!(t.get(1, 1, 1, 0), None);
        let long = t.to_long();
        assert_eq!(long.keys[0].0, "lob");
        assert_eq!(long.keys[1].0, "state");
        assert_eq!(
            long.keys[0].1,
            ["Auto", "Auto", "Auto", "Home", "Home", "Home"]
        );
        assert_eq!(long.keys[1].1, ["CA", "CA", "TX", "NY", "NY", "NY"]);
        assert_eq!(long.values[0].1, [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
        assert_eq!(from_long_table(&long, Grain::Year).unwrap(), t);
        // Keys survive selection and a grain change.
        let s = t
            .select(&[("lob", &["Home"]), ("state", &["NY"])])
            .unwrap()
            .select_columns(&["incurred"])
            .unwrap();
        assert_eq!(s.key_names(), ["lob", "state"]);
        assert_eq!(s.to_long().keys[1].1, ["NY"; 3]);
        let g = t.grain(Grain::Year, Grain::Year).unwrap();
        assert_eq!(g, t);
    }

    #[test]
    fn rejects_bad_keys() {
        let origin = [Month::january(2020); 2];
        let build = |keys: &[(&str, &[&str])]| {
            Triangle::from_long(&Long {
                keys,
                origin: &origin,
                development: DevelopmentColumn::Age(&[12, 24]),
                values: &[("paid", &[1.0, 2.0])],
                origin_grain: Grain::Year,
                development_grain: Grain::Year,
                cumulative: true,
            })
        };
        assert_eq!(
            build(&[("lob", &["A", "A"]), ("lob", &["B", "B"])]),
            Err(Error::DuplicateKey("lob".into()))
        );
        assert_eq!(
            build(&[("paid", &["A", "A"])]),
            Err(Error::KeyClash("paid".into()))
        );
        assert_eq!(
            build(&[("lob", &["A"])]),
            Err(Error::LengthMismatch {
                column: "lob".into(),
                expected: 2,
                found: 1
            })
        );
        assert_eq!(build(&[("lob", &["A", "A"])]).unwrap().shape()[0], 1);
    }

    #[test]
    fn rejects_bad_long_tables() {
        let origin = [Month::january(2020); 2];
        let base = |dev: &[Lag], paid: &[f64]| {
            Triangle::from_long(&Long {
                keys: &[],
                origin: &origin,
                development: DevelopmentColumn::Age(dev),
                values: &[("paid", paid)],
                origin_grain: Grain::Year,
                development_grain: Grain::Year,
                cumulative: true,
            })
        };
        assert!(matches!(
            base(&[12], &[1.0, 2.0]),
            Err(Error::LengthMismatch { .. })
        ));
        assert_eq!(
            base(&[12, 0], &[1.0, 2.0]),
            Err(Error::NonPositiveAge { row: 1 })
        );
        assert_eq!(
            base(&[12, 18], &[1.0, 2.0]),
            Err(Error::OffGrid { row: 1, age: 18 })
        );
        assert!(matches!(
            base(&[12, 24], &[1.0, f64::INFINITY]),
            Err(Error::NonFinite { row: 1, .. })
        ));
        assert_eq!(base(&[12, 24], &[f64::NAN; 2]), Err(Error::Empty));
    }

    #[test]
    fn select_keeps_segments_and_columns() {
        let origin = [Month::january(2020); 2];
        let t = Triangle::from_long(&Long {
            keys: &[("lob", &["A", "B"])],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 12]),
            values: &[("paid", &[1.0, 2.0]), ("incurred", &[3.0, 4.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let s = t
            .select(&[("lob", &["B"])])
            .unwrap()
            .select_columns(&["incurred", "paid"])
            .unwrap();
        assert_eq!(s.shape(), [1, 2, 1, 1]);
        assert_eq!(s.columns(), ["incurred", "paid"]);
        assert_eq!(s.get(0, 0, 0, 0), Some(4.0));
        assert_eq!(s.get(0, 1, 0, 0), Some(2.0));
        assert_eq!(t.select(&[]).unwrap(), t);
        assert_eq!(
            t.select_columns(&["reported"]),
            Err(Error::UnknownColumn("reported".into()))
        );
    }

    #[test]
    fn select_combines_conditions() {
        let t = multi_key();
        let auto = t.select(&[("lob", &["Auto"])]).unwrap();
        assert_eq!(
            auto.index(),
            [Label::new(["Auto", "CA"]), Label::new(["Auto", "TX"])]
        );
        // Segments keep their order whatever the order of the values.
        let both = t.select(&[("state", &["NY", "CA"])]).unwrap();
        assert_eq!(
            both.index(),
            [Label::new(["Auto", "CA"]), Label::new(["Home", "NY"])]
        );
        let one = t
            .select(&[("lob", &["Auto", "Home"]), ("state", &["TX"])])
            .unwrap();
        assert_eq!(one.index(), [Label::new(["Auto", "TX"])]);
        assert_eq!(one.key_names(), ["lob", "state"]);
        assert_eq!(one.get(0, 0, 1, 0), Some(6.0));
    }

    #[test]
    fn select_rejects_bad_conditions() {
        let t = multi_key();
        assert_eq!(
            t.select(&[("line", &["Auto"])]),
            Err(Error::UnknownKey("line".into()))
        );
        assert_eq!(
            t.select(&[("lob", &["Boat"])]),
            Err(Error::UnknownKeyValue {
                key: "lob".into(),
                value: "Boat".into()
            })
        );
        assert_eq!(
            t.select(&[("lob", &["Auto", "Auto"])]),
            Err(Error::DuplicateKeyValue {
                key: "lob".into(),
                value: "Auto".into()
            })
        );
        assert_eq!(
            t.select(&[("lob", &["Auto"]), ("lob", &["Home"])]),
            Err(Error::DuplicateKey("lob".into()))
        );
        assert_eq!(t.select(&[("lob", &[])]), Err(Error::EmptySelection));
        // Both values exist, but not together.
        assert_eq!(
            t.select(&[("lob", &["Home"]), ("state", &["CA"])]),
            Err(Error::NoSegments)
        );
        assert_eq!(t.select_columns(&[]), Err(Error::EmptySelection));
    }

    #[test]
    fn group_by_sums_other_keys() {
        let t = multi_key();
        let lob = t.group_by(&["lob"]).unwrap();
        assert_eq!(lob.key_names(), ["lob"]);
        assert_eq!(lob.index(), [Label::from("Auto"), Label::from("Home")]);
        assert_eq!(lob.columns(), t.columns());
        // Auto paid: CA 2020 [4, 5], TX 2021 [6].
        assert_eq!(lob.get(0, 0, 0, 0), Some(4.0));
        assert_eq!(lob.get(0, 0, 0, 1), Some(5.0));
        assert_eq!(lob.get(0, 0, 1, 0), Some(6.0));
        // TX incurred is missing, so Auto 2021 incurred is unobserved.
        assert_eq!(lob.get(0, 1, 1, 0), None);
        // Keys follow the order given.
        let swapped = t.group_by(&["state", "lob"]).unwrap();
        assert_eq!(swapped.key_names(), ["state", "lob"]);
        assert_eq!(swapped.index()[0], Label::new(["CA", "Auto"]));
        // No keys: one total segment, equal to grouping the groups.
        let total = t.group_by(&[]).unwrap();
        assert_eq!(total.index(), [Label::default()]);
        assert!(total.key_names().is_empty());
        assert_eq!(total.get(0, 0, 0, 0), Some(5.0));
        assert_eq!(total.get(0, 0, 1, 0), Some(9.0));
        assert_eq!(lob.group_by(&[]).unwrap(), total);
        // Grouping by every key changes nothing.
        assert_eq!(t.group_by(&["lob", "state"]).unwrap(), t);
    }

    #[test]
    fn group_by_incremental_sums_cumulative_values() {
        let t = multi_key();
        let inc = t.to_incremental().group_by(&["lob"]).unwrap();
        assert!(!inc.is_cumulative());
        assert_eq!(inc, t.group_by(&["lob"]).unwrap().to_incremental());

        // Members observed to different ages in one origin: A to 24 months,
        // B to 12. The group's cumulative values are 1 + 10 = 11 at 12 and
        // 3 at 24 (only A is observed), so its increment at 24 is 3 - 11,
        // not A's own increment 3 - 1.
        let t = Triangle::from_long(&Long {
            keys: &[("seg", &["A", "A", "B"])],
            origin: &[Month::january(2020); 3],
            development: DevelopmentColumn::Age(&[12, 24, 12]),
            values: &[("paid", &[1.0, 3.0, 10.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let summed = t.to_incremental().group_by(&[]).unwrap();
        assert_eq!(summed.get(0, 0, 0, 0), Some(11.0));
        assert_eq!(summed.get(0, 0, 0, 1), Some(-8.0));
        assert_eq!(summed, t.group_by(&[]).unwrap().to_incremental());
    }

    #[test]
    fn group_by_rejects_bad_keys() {
        let t = multi_key();
        assert_eq!(t.group_by(&["line"]), Err(Error::UnknownKey("line".into())));
        assert_eq!(
            t.group_by(&["lob", "lob"]),
            Err(Error::DuplicateKey("lob".into()))
        );
    }

    #[test]
    fn calendar_view() {
        let t = annual(2020, &[&[1.0, 2.0], &[3.0]]);
        let cal = t.dev_to_val();
        assert_eq!(cal.valuations(), [m(2020, 12), m(2021, 12)]);
        assert_eq!(cal.get(0, 0, 0, 1), Some(2.0));
        assert_eq!(cal.get(0, 0, 1, 1), Some(3.0));
        assert_eq!(cal.get(0, 0, 1, 0), None);
        assert_eq!(cal.val_to_dev(), &t);
    }

    /// Quarterly origins and ages, valued at 2021Q2: two full years of
    /// quarters where each quarter's cumulative is 1 per elapsed quarter.
    fn quarterly() -> Triangle {
        let mut origin = Vec::new();
        let mut ages = Vec::new();
        let mut values = Vec::new();
        let valuation = m(2021, 6);
        for q in 0..6 {
            let start = m(2020, 1).add_months(3 * q);
            let mut age = 3;
            while start.add_months(age - 1) <= valuation {
                origin.push(start);
                ages.push(age as Lag);
                values.push((age / 3) as f64);
                age += 3;
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &values)],
            origin_grain: Grain::Quarter,
            development_grain: Grain::Quarter,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn grain_to_annual_keeps_partial_latest_diagonal() {
        let q = quarterly();
        assert_eq!(q.shape(), [1, 1, 6, 6]);
        let y = q.grain(Grain::Year, Grain::Year).unwrap();
        // chainladder-python 0.10.1: grain("OYDY") on the same data gives
        // ages 6 and 18 with rows [3, 18] and [3, NaN].
        // Valued at June: ages are 6 and 18 months from each year's start.
        assert_eq!(y.development(), [6, 18]);
        assert_eq!(y.valuation(), m(2021, 6));
        // 2020 at June 2020: Q1 has 2 quarters, Q2 has 1.
        assert_eq!(y.get(0, 0, 0, 0), Some(3.0));
        // 2020 at June 2021: 6 + 5 + 4 + 3 quarters elapsed.
        assert_eq!(y.get(0, 0, 0, 1), Some(18.0));
        assert_eq!(y.get(0, 0, 1, 0), Some(3.0));
        assert_eq!(y.get(0, 0, 1, 1), None);
    }

    #[test]
    fn grain_to_annual_origin_quarterly_development() {
        let y = quarterly().grain(Grain::Year, Grain::Quarter).unwrap();
        // chainladder-python 0.10.1: grain("OYDQ") on the same data.
        assert_eq!(y.development(), [3, 6, 9, 12, 15, 18]);
        let row = |o| (0..6).map(|d| y.get(0, 0, o, d)).collect::<Vec<_>>();
        assert_eq!(row(0), [1.0, 3.0, 6.0, 10.0, 14.0, 18.0].map(Some).to_vec());
        assert_eq!(
            row(1),
            [Some(1.0), Some(3.0), None, None, None, None].to_vec()
        );
        let inc = quarterly().to_incremental();
        assert_eq!(
            inc.grain(Grain::Year, Grain::Quarter).unwrap(),
            y.to_incremental()
        );
    }

    #[test]
    fn grain_rejects_refinement() {
        assert!(matches!(
            raa().grain(Grain::Quarter, Grain::Quarter),
            Err(Error::InvalidGrain(_))
        ));
        let t = quarterly();
        assert!(matches!(
            t.grain(Grain::Quarter, Grain::Year),
            Err(Error::InvalidGrain(_))
        ));
    }

    #[test]
    fn segment_requires_single_index() {
        let t = raa();
        let s = t.segment("values").unwrap();
        assert_eq!(s.latest(9).unwrap(), (0, 2063.0));
        assert_eq!(s.link_pairs(8), vec![(18662.0, 18834.0)]);
        assert_eq!(
            t.segment("paid").unwrap_err(),
            Error::UnknownColumn("paid".into())
        );
    }

    /// Quarterly origins and ages valued at 2020-12, cumulative 1 per elapsed
    /// quarter, for the given (segment, 2020 quarters present).
    fn sparse_quarterly(segments: &[(&str, &[i64])]) -> Triangle {
        let (mut index, mut origin, mut ages, mut values) = (vec![], vec![], vec![], vec![]);
        for (label, quarters) in segments {
            for &q in *quarters {
                let start = m(2020, 1).add_months(3 * q);
                for k in 1..=(4 - q) {
                    index.push(*label);
                    origin.push(start);
                    ages.push(3 * k as Lag);
                    values.push(k as f64);
                }
            }
        }
        Triangle::from_long(&Long {
            keys: &[("segment", &index)],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &values)],
            origin_grain: Grain::Quarter,
            development_grain: Grain::Quarter,
            cumulative: true,
        })
        .unwrap()
    }

    #[test]
    fn grain_with_an_empty_sub_origin() {
        // chainladder-python 0.10.1 grain("OYDY") gives 7: Q2 has no rows.
        let y = sparse_quarterly(&[("A", &[0, 2, 3])])
            .grain(Grain::Year, Grain::Year)
            .unwrap();
        assert_eq!(y.get(0, 0, 0, 0), Some(7.0));
    }

    #[test]
    fn grain_with_a_segment_that_starts_later() {
        // chainladder-python 0.10.1: B gives 3 for OYDY, [nan, nan, 1, 3]
        // for OYDQ.
        let t = sparse_quarterly(&[("A", &[0, 1, 2, 3]), ("B", &[2, 3])]);
        let yy = t.grain(Grain::Year, Grain::Year).unwrap();
        assert_eq!(yy.get(0, 0, 0, 0), Some(10.0));
        assert_eq!(yy.get(1, 0, 0, 0), Some(3.0));
        let yq = t.grain(Grain::Year, Grain::Quarter).unwrap();
        let b: Vec<_> = (0..4).map(|d| yq.get(1, 0, 0, d)).collect();
        assert_eq!(b, [None, None, Some(1.0), Some(3.0)]);
    }

    #[test]
    fn missing_incremental_rows_are_zero_increments() {
        let origin = [2018, 2018, 2018, 2019, 2019, 2020, 2020, 2021].map(Month::january);
        let t = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12, 24, 12]),
            values: &[(
                "paid",
                &[100.0, 50.0, 25.0, 100.0, 60.0, 100.0, 40.0, 100.0],
            )],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: false,
        })
        .unwrap();
        // The grid reaches 2018's age at the 2021 valuation.
        assert_eq!(t.development(), [12, 24, 36, 48]);
        let cum = t.to_cumulative();
        assert_eq!(cum.get(0, 0, 0, 3), Some(175.0));
        assert_eq!(cum.get(0, 0, 1, 2), Some(160.0));
        assert_eq!(cum.get(0, 0, 1, 3), None);
        // chainladder-python 0.10.1 Chainladder().fit(...).ultimate_.
        let cl = crate::ChainLadder::default().fit(&t, "paid").unwrap();
        for (got, want) in cl.ultimate.iter().zip([175.0, 160.0, 151.29, 162.10]) {
            assert!((got - want).abs() < 0.01, "{:?}", cl.ultimate);
        }
    }

    #[test]
    fn select_columns_rejects_duplicates() {
        let t = sparse_quarterly(&[("A", &[0]), ("B", &[0])]);
        assert_eq!(
            t.select_columns(&["paid", "paid"]),
            Err(Error::DuplicateColumn("paid".into()))
        );
    }
}
