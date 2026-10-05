//! The bridge from a triangle to the models of `act-models`: one row per
//! origin × development cell, with the incremental value and features.
//!
//! Models never see a [`Triangle`] (`docs/design/models.md`, "Fitting a
//! triangle with several models"). A [`TriangleFrame`] gives them an
//! [`act_models::Frame`] of features, the response, the observed (training)
//! rows, the future (prediction) rows below the latest diagonal, and
//! calendar-diagonal splits for backtesting.

use act_core::{Lag, Period};
use act_models::resample::{Split, time_ordered};
use act_models::{Column, Frame};

use crate::error::{Error, Result};
use crate::triangle::Triangle;

/// One row per origin × development cell of a single-segment triangle.
///
/// Feature columns, for every row:
///
/// | Column | Kind | Value |
/// |---|---|---|
/// | `origin` | factor | origin period, e.g. `"1981"` |
/// | `development` | factor | age in months, e.g. `"12"` |
/// | `calendar` | factor | calendar period of the valuation, at the development grain |
/// | `origin_index` | number | 0-based origin position |
/// | `development_index` | number | 0-based development position |
/// | `calendar_index` | number | 0-based calendar period, from the earliest valuation |
/// | `age` | number | age in months |
/// | `exposure` | number | the origin's exposure (1 if none was given) |
///
/// The response is the incremental value of `column`. A cell is observed
/// (a training row) when it and the previous age are observed, so the
/// increment covers one development period; a cell valued after the
/// triangle's valuation is a future (prediction) row; any other cell (one
/// after a hole) is neither.
///
/// ```
/// use act_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle, TriangleFrame};
///
/// let origin = [2020, 2020, 2021].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 12]),
///     values: &[("paid", &[100.0, 150.0, 110.0])],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let cells = TriangleFrame::new(&tri, "paid", None)?;
/// assert_eq!(cells.n_rows(), 4);
/// assert_eq!(cells.observed(), [0, 1, 2]);
/// assert_eq!(cells.future(), [3]);
/// assert_eq!(cells.response()[1], 50.0);
/// assert!(cells.response()[3].is_nan());
/// # Ok::<(), act_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TriangleFrame {
    frame: Frame,
    response: Vec<f64>,
    observed: Vec<usize>,
    future: Vec<usize>,
    origin: Vec<usize>,
    development: Vec<usize>,
    calendar: Vec<i64>,
    origins: Vec<Period>,
    ages: Vec<Lag>,
}

impl TriangleFrame {
    /// The cells of `column` in a single-segment triangle. `exposure`, if
    /// given, has one value per origin.
    pub fn new(triangle: &Triangle, column: &str, exposure: Option<&[f64]>) -> Result<Self> {
        let segment = triangle.segment(column)?;
        let (no, nd) = (segment.n_origins, segment.n_dev);
        if let Some(e) = exposure {
            if e.len() != no {
                return Err(Error::LengthMismatch {
                    column: "exposure".into(),
                    expected: no,
                    found: e.len(),
                });
            }
            if let Some(&bad) = e.iter().find(|x| !x.is_finite() || **x <= 0.0) {
                return Err(Error::Core(act_core::Error::InvalidParameter {
                    name: "exposure",
                    value: bad,
                    reason: "must be finite and positive",
                }));
            }
        }
        let ages = triangle.development().to_vec();
        let grain = triangle.development_grain();
        let step = grain.months() as i64;
        let first_valuation = (0..no)
            .flat_map(|o| (0..nd).map(move |d| (o, d)))
            .map(|(o, d)| triangle.valuation_of(o, d))
            .min()
            .expect("a triangle has at least one cell");

        let n = no * nd;
        let mut origin = Vec::with_capacity(n);
        let mut development = Vec::with_capacity(n);
        let mut calendar = Vec::with_capacity(n);
        let mut response = Vec::with_capacity(n);
        let (mut observed, mut future) = (Vec::new(), Vec::new());
        let (mut origin_label, mut dev_label, mut cal_label) = (vec![], vec![], vec![]);
        for o in 0..no {
            for (d, age) in ages.iter().enumerate() {
                let row = o * nd + d;
                let valuation = triangle.valuation_of(o, d);
                origin.push(o);
                development.push(d);
                calendar.push(valuation.months_since(first_valuation).div_euclid(step));
                origin_label.push(segment.origins[o].to_string());
                dev_label.push(age.to_string());
                cal_label.push(Period::containing(valuation, grain).to_string());
                let previous = if d == 0 {
                    Some(0.0)
                } else {
                    segment.get(o, d - 1)
                };
                match (segment.get(o, d), previous) {
                    (Some(value), Some(before)) => {
                        response.push(value - before);
                        observed.push(row);
                    }
                    _ => {
                        response.push(f64::NAN);
                        if valuation > triangle.valuation() {
                            future.push(row);
                        }
                    }
                }
            }
        }

        let numbers = |v: &[usize]| Column::Numeric(v.iter().map(|&x| x as f64).collect());
        let exposure_column: Vec<f64> = origin
            .iter()
            .map(|&o| exposure.map_or(1.0, |e| e[o]))
            .collect();
        let frame = Frame::new(vec![
            ("origin".into(), Column::Categorical(origin_label)),
            ("development".into(), Column::Categorical(dev_label)),
            ("calendar".into(), Column::Categorical(cal_label)),
            ("origin_index".into(), numbers(&origin)),
            ("development_index".into(), numbers(&development)),
            (
                "calendar_index".into(),
                Column::Numeric(calendar.iter().map(|&c| c as f64).collect()),
            ),
            (
                "age".into(),
                Column::Numeric(development.iter().map(|&d| ages[d] as f64).collect()),
            ),
            ("exposure".into(), Column::Numeric(exposure_column)),
        ])?;
        Ok(Self {
            frame,
            response,
            observed,
            future,
            origin,
            development,
            calendar,
            origins: segment.origins.clone(),
            ages,
        })
    }

    /// Number of rows (cells).
    pub fn n_rows(&self) -> usize {
        self.response.len()
    }

    /// Features of every row.
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// Features of the given rows, in that order.
    pub fn select(&self, rows: &[usize]) -> Result<Frame> {
        let names = [
            "origin",
            "development",
            "calendar",
            "origin_index",
            "development_index",
            "calendar_index",
            "age",
            "exposure",
        ];
        let columns = names
            .iter()
            .map(|&name| {
                let column = match self.frame.column(name).expect("built in new") {
                    Column::Numeric(v) => Column::Numeric(rows.iter().map(|&r| v[r]).collect()),
                    Column::Categorical(v) => {
                        Column::Categorical(rows.iter().map(|&r| v[r].clone()).collect())
                    }
                };
                (name.to_string(), column)
            })
            .collect();
        Ok(Frame::new(columns)?)
    }

    /// Incremental value of every row; NaN where the row is not observed.
    pub fn response(&self) -> &[f64] {
        &self.response
    }

    /// Response of the given rows, in that order.
    pub fn response_of(&self, rows: &[usize]) -> Vec<f64> {
        rows.iter().map(|&r| self.response[r]).collect()
    }

    /// Observed rows: the training data.
    pub fn observed(&self) -> &[usize] {
        &self.observed
    }

    /// Rows valued after the triangle's valuation: the cells to predict.
    pub fn future(&self) -> &[usize] {
        &self.future
    }

    /// Origin position of `row`.
    pub fn origin_of(&self, row: usize) -> usize {
        self.origin[row]
    }

    /// Development position of `row`.
    pub fn development_of(&self, row: usize) -> usize {
        self.development[row]
    }

    /// Calendar period index of `row`.
    pub fn calendar_of(&self, row: usize) -> i64 {
        self.calendar[row]
    }

    /// The triangle's origin periods.
    pub fn origins(&self) -> &[Period] {
        &self.origins
    }

    /// The triangle's development ages.
    pub fn ages(&self) -> &[Lag] {
        &self.ages
    }

    /// Calendar-diagonal backtest splits over the observed rows: for each of
    /// the latest `n_diagonals` calendar periods, train on observed rows
    /// valued before it and test on its observed rows. Row numbers index
    /// this frame.
    pub fn diagonal_splits(&self, n_diagonals: usize) -> Result<Vec<Split>> {
        let periods: Vec<i64> = self.observed.iter().map(|&r| self.calendar[r]).collect();
        let splits = time_ordered(&periods, n_diagonals)?;
        let to_rows = |positions: Vec<usize>| -> Vec<usize> {
            positions.into_iter().map(|p| self.observed[p]).collect()
        };
        Ok(splits
            .into_iter()
            .map(|s| Split {
                train: to_rows(s.train),
                test: to_rows(s.test),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::{annual, raa};
    use crate::{DevelopmentColumn, Grain, Long, Month};

    fn labels<'a>(frame: &'a Frame, name: &str) -> &'a [String] {
        match frame.column(name) {
            Some(Column::Categorical(v)) => v,
            other => panic!("{name}: {other:?}"),
        }
    }

    fn numbers<'a>(frame: &'a Frame, name: &str) -> &'a [f64] {
        match frame.column(name) {
            Some(Column::Numeric(v)) => v,
            other => panic!("{name}: {other:?}"),
        }
    }

    #[test]
    fn raa_cells() {
        let cells = TriangleFrame::new(&raa(), "values", None).unwrap();
        assert_eq!(cells.n_rows(), 100);
        assert_eq!(cells.observed().len(), 55);
        assert_eq!(cells.future().len(), 45);
        // Observed increments sum to the latest diagonal.
        let total: f64 = cells.observed().iter().map(|&r| cells.response()[r]).sum();
        assert_eq!(total, 160_987.0);
        // 1982 at 24 months: 4285 - 106.
        assert_eq!(cells.response()[11], 4179.0);
        let frame = cells.frame();
        assert_eq!(labels(frame, "origin")[11], "1982");
        assert_eq!(labels(frame, "development")[11], "24");
        assert_eq!(labels(frame, "calendar")[11], "1983");
        assert_eq!(numbers(frame, "calendar_index")[11], 2.0);
        assert_eq!(numbers(frame, "age")[11], 24.0);
        assert_eq!(numbers(frame, "exposure")[11], 1.0);
        // The latest diagonal is calendar 1990, index 9.
        assert!(cells.future().iter().all(|&r| cells.calendar_of(r) > 9));
    }

    #[test]
    fn diagonal_splits_hold_out_latest_diagonals() {
        let cells = TriangleFrame::new(&raa(), "values", None).unwrap();
        let splits = cells.diagonal_splits(2).unwrap();
        assert_eq!(splits.len(), 2);
        // Calendar 1989 (9 cells) then 1990 (10 cells).
        assert_eq!(splits[0].test.len(), 9);
        assert_eq!(splits[1].test.len(), 10);
        assert_eq!(splits[1].train.len(), 45);
        for s in &splits {
            let t = cells.calendar_of(s.test[0]);
            assert!(s.test.iter().all(|&r| cells.calendar_of(r) == t));
            assert!(s.train.iter().all(|&r| cells.calendar_of(r) < t));
        }
        assert!(cells.diagonal_splits(10).is_err());
    }

    #[test]
    fn select_and_exposure() {
        let tri = annual(2020, &[&[10.0, 15.0], &[20.0]]);
        let cells = TriangleFrame::new(&tri, "values", Some(&[2.0, 4.0])).unwrap();
        let picked = cells.select(&[3, 0]).unwrap();
        assert_eq!(labels(&picked, "origin"), ["2021", "2020"]);
        assert_eq!(numbers(&picked, "exposure"), [4.0, 2.0]);
        assert_eq!(cells.response_of(&[1, 0]), [5.0, 10.0]);
        assert!(matches!(
            TriangleFrame::new(&tri, "values", Some(&[1.0])),
            Err(Error::LengthMismatch { .. })
        ));
        assert!(TriangleFrame::new(&tri, "values", Some(&[1.0, 0.0])).is_err());
    }

    #[test]
    fn cell_after_a_hole_is_neither_observed_nor_future() {
        let origin = [2019, 2019, 2019, 2020, 2020, 2021].map(Month::january);
        let tri = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 36, 12, 36, 12]),
            values: &[("paid", &[1.0, 2.0, 3.0, 1.0, 3.0, 1.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let cells = TriangleFrame::new(&tri, "paid", None).unwrap();
        // 2020 at 36 follows the hole at 24. It also sets the valuation to
        // 2022-12, so 2021 at 24 (valued 2022-12) is a missing past cell.
        for row in [5, 7] {
            assert!(!cells.observed().contains(&row));
            assert!(!cells.future().contains(&row));
        }
        assert_eq!(cells.future(), [8]);
    }

    #[test]
    fn quarterly_calendar_labels() {
        let origin = [2020, 2020, 2020].map(|y| Month::new(y, 1).unwrap());
        let tri = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[3, 6, 9]),
            values: &[("paid", &[1.0, 2.0, 3.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Quarter,
            cumulative: true,
        })
        .unwrap();
        let cells = TriangleFrame::new(&tri, "paid", None).unwrap();
        assert_eq!(
            labels(cells.frame(), "calendar"),
            ["2020Q1", "2020Q2", "2020Q3"]
        );
        assert_eq!(numbers(cells.frame(), "calendar_index"), [0.0, 1.0, 2.0]);
    }
}
