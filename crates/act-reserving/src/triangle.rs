//! The loss triangle: index × column × origin × development, dense and masked.
//!
//! Axes and semantics follow chainladder-python. See
//! `docs/design/triangle.md`.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{Error, Result};
use crate::period::{Grain, Lag, Month, Period};

/// A position on the index axis (a segment such as a company or line of
/// business). Labels have one or more parts, like a pandas `MultiIndex` row.
///
/// ```
/// use act_reserving::Label;
///
/// let l = Label::new(["Auto", "CA"]);
/// assert_eq!(l.parts(), ["Auto", "CA"]);
/// assert_eq!(l.to_string(), "Auto / CA");
/// assert_eq!(Label::from("Total").parts(), ["Total"]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label(Vec<String>);

impl Label {
    /// A label with the given parts.
    pub fn new<S: Into<String>>(parts: impl IntoIterator<Item = S>) -> Self {
        Self(parts.into_iter().map(Into::into).collect())
    }

    /// The label's parts, outermost first.
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
        f.write_str(&self.0.join(" / "))
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

/// A long table borrowed from the caller: one row per (index, origin,
/// development) with one or more measure columns. This is the shape of a
/// claims extract and of the Arrow tables the bindings pass in.
#[derive(Debug, Clone, Copy)]
pub struct Long<'a> {
    /// Segment of each row; `None` puts every row in one segment, `Total`.
    pub index: Option<&'a [Label]>,
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
    pub index: Vec<Label>,
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
///     index: None,
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 12]),
///     values: &[("paid", &[100.0, 150.0, 110.0])],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })
/// .unwrap();
/// assert_eq!(tri.shape(), [1, 1, 2, 2]);
/// assert_eq!(tri.get(0, 0, 1, 0), Some(110.0));
/// assert_eq!(tri.get(0, 0, 1, 1), None);
/// assert_eq!(tri.valuation(), Month::new(2021, 12).unwrap());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Triangle {
    values: Vec<f64>,
    mask: Vec<bool>,
    shape: [usize; 4],
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
    /// Rows with the same (index, origin, age) are summed. NaN values are
    /// treated as missing.
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
        if let Some(index) = long.index {
            check_len("index", index.len())?;
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

        let default_label = Label::from("Total");
        let label_of = |row: usize| long.index.map_or(&default_label, |ix| &ix[row]);
        let mut index: Vec<Label> = (0..n).map(|r| label_of(r).clone()).collect();
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
                .binary_search(label_of(row))
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

    /// The triangle as a long table: one row per (index, origin, age) with
    /// at least one observed measure, in axis order.
    pub fn to_long(&self) -> LongTable {
        let [ni, nc, no, nd] = self.shape;
        let mut out = LongTable {
            index: Vec::new(),
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
                    out.index.push(self.index[i].clone());
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

    /// Labels of the index axis: sorted by [`Triangle::from_long`], in the
    /// requested order after [`Triangle::slice`].
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
            .ok_or_else(|| Error::UnknownLabel(name.to_string()))
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

    /// A triangle with only the named index positions and columns, in the
    /// order given. `None` keeps an axis whole. Naming a label or column
    /// twice is an error.
    pub fn slice(&self, index: Option<&[Label]>, columns: Option<&[&str]>) -> Result<Self> {
        let index_pos: Vec<usize> = match index {
            None => (0..self.shape[0]).collect(),
            Some(labels) => labels
                .iter()
                .map(|l| {
                    self.index
                        .iter()
                        .position(|x| x == l)
                        .ok_or_else(|| Error::UnknownLabel(l.to_string()))
                })
                .collect::<Result<_>>()?,
        };
        let column_pos: Vec<usize> = match columns {
            None => (0..self.shape[1]).collect(),
            Some(names) => names
                .iter()
                .map(|n| self.column_position(n))
                .collect::<Result<_>>()?,
        };
        if index_pos.is_empty() || column_pos.is_empty() {
            return Err(Error::Empty);
        }
        if let Some(k) = (1..index_pos.len()).find(|&k| index_pos[..k].contains(&index_pos[k])) {
            return Err(Error::DuplicateLabel(self.index[index_pos[k]].to_string()));
        }
        if let Some(k) = (1..column_pos.len()).find(|&k| column_pos[..k].contains(&column_pos[k])) {
            return Err(Error::DuplicateColumn(self.columns[column_pos[k]].clone()));
        }
        let [_, _, no, nd] = self.shape;
        let block = no * nd;
        let mut values = Vec::with_capacity(index_pos.len() * column_pos.len() * block);
        let mut mask = Vec::with_capacity(values.capacity());
        for &i in &index_pos {
            for &c in &column_pos {
                let at = self.offset(i, c, 0, 0);
                values.extend_from_slice(&self.values[at..at + block]);
                mask.extend_from_slice(&self.mask[at..at + block]);
            }
        }
        Ok(Self {
            values,
            mask,
            shape: [index_pos.len(), column_pos.len(), no, nd],
            index: index_pos.iter().map(|&i| self.index[i].clone()).collect(),
            columns: column_pos
                .iter()
                .map(|&c| self.columns[c].clone())
                .collect(),
            ..self.clone()
        })
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
        let mut increments: BTreeMap<(usize, Month), Vec<(Month, usize, f64)>> = BTreeMap::new();
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

        let mut index = Vec::new();
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
                index.push(self.index[i].clone());
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
        let out = Self::from_long(&Long {
            index: Some(&index),
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
        let c = self.column_position(column)?;
        let cum = self.to_cumulative();
        let [_, _, no, nd] = self.shape;
        let cells = (0..no)
            .flat_map(|o| (0..nd).map(move |d| (o, d)))
            .map(|(o, d)| cum.get(0, c, o, d))
            .collect();
        Ok(Segment {
            cells,
            n_origins: no,
            n_dev: nd,
            origins: self.origins(),
        })
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
            index: None,
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
        assert_eq!(t.index(), [Label::from("Total")]);
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
            index: None,
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
        let index = [
            Label::new(["B", "x"]),
            Label::new(["A", "y"]),
            Label::new(["A", "y"]),
            Label::new(["A", "y"]),
        ];
        let origin = [m(2020, 5), m(2020, 2), m(2020, 3), m(2020, 1)];
        let valuation = [m(2020, 6), m(2020, 3), m(2020, 3), m(2020, 6)];
        let t = Triangle::from_long(&Long {
            index: Some(&index),
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

    #[test]
    fn to_long_round_trips() {
        let t = raa();
        let long = t.to_long();
        assert_eq!(long.origin.len(), 55);
        let values: Vec<(&str, &[f64])> = long
            .values
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        let back = Triangle::from_long(&Long {
            index: Some(&long.index),
            origin: &long.origin,
            development: DevelopmentColumn::Age(&long.development),
            values: &values,
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn rejects_bad_long_tables() {
        let origin = [Month::january(2020); 2];
        let base = |dev: &[Lag], paid: &[f64]| {
            Triangle::from_long(&Long {
                index: None,
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
    fn slice_selects_segments_and_columns() {
        let index = [Label::from("A"), Label::from("B")];
        let origin = [Month::january(2020); 2];
        let t = Triangle::from_long(&Long {
            index: Some(&index),
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 12]),
            values: &[("paid", &[1.0, 2.0]), ("incurred", &[3.0, 4.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let s = t
            .slice(Some(&[Label::from("B")]), Some(&["incurred", "paid"]))
            .unwrap();
        assert_eq!(s.shape(), [1, 2, 1, 1]);
        assert_eq!(s.columns(), ["incurred", "paid"]);
        assert_eq!(s.get(0, 0, 0, 0), Some(4.0));
        assert_eq!(s.get(0, 1, 0, 0), Some(2.0));
        assert_eq!(
            t.slice(None, Some(&["reported"])),
            Err(Error::UnknownLabel("reported".into()))
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
            index: None,
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
            Error::UnknownLabel("paid".into())
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
                    index.push(Label::from(*label));
                    origin.push(start);
                    ages.push(3 * k as Lag);
                    values.push(k as f64);
                }
            }
        }
        Triangle::from_long(&Long {
            index: Some(&index),
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
            index: None,
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
    fn slice_rejects_duplicates() {
        let t = sparse_quarterly(&[("A", &[0]), ("B", &[0])]);
        let a = Label::from("A");
        assert_eq!(
            t.slice(Some(&[a.clone(), a]), None),
            Err(Error::DuplicateLabel("A".into()))
        );
        assert_eq!(
            t.slice(None, Some(&["paid", "paid"])),
            Err(Error::DuplicateColumn("paid".into()))
        );
    }
}
