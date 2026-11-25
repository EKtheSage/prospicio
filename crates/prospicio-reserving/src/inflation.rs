//! Calendar-period indexation of a triangle (`reserving-inflation.md`, PR 1,
//! issue #979).
//!
//! An index by calendar period ([`CalendarIndex`]: CPI, wages, a
//! claims-inflation index) restates each increment of a triangle at the
//! price level of one base period ([`Indexation::restate`]):
//!
//! ```text
//! x*_ij = x_ij I(base) / I(p(i, j))
//! ```
//!
//! where `p(i, j)` is the index period that contains the cell's valuation
//! month ([`Triangle::valuation_of`], decision 1). The restated triangle is
//! developed by any method; its projected future increments, collected as
//! [`FutureCells`], are re-inflated to the money of their payment periods
//! ([`Indexation::reinflate`]), the index past its last given period
//! extended at a constant annual rate (decision 4). This is the
//! re-inflation of Taylor (1977, equation 8), whose 10% projection of
//! Example 1 the validation tests reproduce.
//!
//! Decisions (`reserving-inflation.md`): the calendar is a `Month` and the
//! index has its own grain (1); indexation acts on increments, whatever the
//! triangle's basis (2); the base is explicit (3); a constant future rate
//! (4); a chain ladder's tail increment is paid one development period
//! after the oldest age (5); re-inflation takes future cells, not a fit (6).

use crate::chain_ladder::ChainLadderFit;
use crate::error::{Error, Result};
use crate::odp_glm::OdpGlmFit;
use crate::triangle::Triangle;
use prospicio_core::{Lag, Month, Period};
use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};

fn inflation_err<T>(why: impl Into<String>) -> Result<T> {
    Err(Error::Inflation(why.into()))
}

/// An index by calendar period: one value per period of one grain, from a
/// first period on, optionally extended past its last period at a constant
/// annual rate.
///
/// ```
/// use prospicio_reserving::{CalendarIndex, Month, Period};
///
/// let cpi = CalendarIndex::new(Period::year(2020), vec![100.0, 104.0, 110.0])?
///     .with_future_rate(0.05)?;
/// assert_eq!(cpi.value(Month::new(2021, 6)?)?, 104.0);
/// // Two years past 2022 at 5% a year.
/// assert!((cpi.value(Month::january(2024))? - 110.0 * 1.05_f64.powi(2)).abs() < 1e-9);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CalendarIndex {
    first: Period,
    values: Vec<f64>,
    future_rate: Option<f64>,
}

impl CalendarIndex {
    /// `values[0]` is the index in `first`, then one value per period of
    /// `first`'s grain.
    ///
    /// Fails if `values` is empty or a value is not finite and positive.
    pub fn new(first: Period, values: Vec<f64>) -> Result<Self> {
        if values.is_empty() {
            return inflation_err("an index needs at least one value");
        }
        if let Some(&bad) = values.iter().find(|v| !v.is_finite() || **v <= 0.0) {
            return Err(Error::InvalidSetting {
                name: "index",
                value: bad,
                expected: "finite and positive values",
            });
        }
        Ok(Self {
            first,
            values,
            future_rate: None,
        })
    }

    /// The index extended past its last period at the annual `rate`,
    /// compounded by months from the last period's start: a period `m`
    /// months later has `last * (1 + rate)^(m / 12)`.
    ///
    /// Fails unless `rate` is finite and above -1.
    pub fn with_future_rate(self, rate: f64) -> Result<Self> {
        if !rate.is_finite() || rate <= -1.0 {
            return Err(Error::InvalidSetting {
                name: "future_rate",
                value: rate,
                expected: "finite and above -1",
            });
        }
        Ok(Self {
            future_rate: Some(rate),
            ..self
        })
    }

    /// The index of the period that contains `month`.
    ///
    /// Fails for a month before the first period, or after the last one
    /// when no future rate is set.
    pub fn value(&self, month: Month) -> Result<f64> {
        let grain = self.first.grain();
        let start = Period::containing(month, grain).start();
        let months = start.months_since(self.first.start());
        if months < 0 {
            return inflation_err(format!("the index starts in {}, after {month}", self.first));
        }
        let position = months / grain.months() as i64;
        let last = self.values.len() as i64 - 1;
        if position <= last {
            return Ok(self.values[position as usize]);
        }
        match self.future_rate {
            Some(rate) => {
                let months_past = (position - last) * grain.months() as i64;
                Ok(self.values[last as usize] * (1.0 + rate).powf(months_past as f64 / 12.0))
            }
            None => inflation_err(format!(
                "the index ends in {} and has no future rate for {month}",
                self.periods()[last as usize]
            )),
        }
    }

    /// The given periods, one per value.
    pub fn periods(&self) -> Vec<Period> {
        let step = self.first.grain().months() as i64;
        (0..self.values.len() as i64)
            .map(|k| {
                Period::containing(self.first.start().add_months(k * step), self.first.grain())
            })
            .collect()
    }

    /// The given values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The annual rate past the last period, if set.
    pub fn future_rate(&self) -> Option<f64> {
        self.future_rate
    }
}

/// Restating a triangle at one price level by a [`CalendarIndex`], and
/// re-inflating a projection.
///
/// ```
/// use prospicio_reserving::{
///     CalendarIndex, ChainLadder, DevelopmentColumn, FutureCells, Grain, Indexation, Long,
///     Month, Period, Triangle,
/// };
///
/// // Payments of 100, 50 and 25 at 2020 prices, inflated at 10% a year.
/// let paid = [100.0, 155.0, 185.25, 110.0, 170.5, 121.0];
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &[2020, 2020, 2020, 2021, 2021, 2022].map(Month::january),
///     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
///     values: &[("paid", &paid)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let index = CalendarIndex::new(Period::year(2020), vec![1.0, 1.1, 1.21])?
///     .with_future_rate(0.10)?;
/// let at_2020 = Indexation { index, base: Month::january(2020) };
/// let restated = at_2020.restate(&tri, "paid")?;
/// // Every origin's increments are 100, 50, 25 at 2020 prices.
/// assert!((restated.get(0, 0, 2, 0).unwrap() - 100.0).abs() < 1e-9);
/// let fit = ChainLadder::default().fit(&restated, "paid")?;
/// let reinflated = at_2020.reinflate(&FutureCells::from_chain_ladder(&fit))?;
/// // The 2022 origin pays 50 * 1.1^3 in 2023 and 25 * 1.1^4 in 2024.
/// let want = 50.0 * 1.1_f64.powi(3) + 25.0 * 1.1_f64.powi(4);
/// assert!((reinflated.reserves[2] - want).abs() < 1e-9);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Indexation {
    /// The index by calendar period.
    pub index: CalendarIndex,
    /// The base price level: the index period that contains this month.
    pub base: Month,
}

impl Indexation {
    /// `I(p) / I(base)`, `p` the index period that contains `month`: the
    /// factor that takes an amount at the base level to the money of `p`.
    ///
    /// Fails if the index has no value for either period.
    pub fn factor(&self, month: Month) -> Result<f64> {
        Ok(self.index.value(month)? / self.index.value(self.base)?)
    }

    /// A copy of `triangle` with the increments of `column` restated at
    /// the base level, each divided by the [`factor`](Self::factor) of its
    /// cell's valuation month, in the triangle's basis (a cumulative
    /// triangle is differenced, restated and cumulated again), in every
    /// segment. Other columns are unchanged. An increment after a hole
    /// covers several periods and is restated at its own cell's period.
    ///
    /// Fails for an unknown column or a payment period the index does not
    /// cover.
    pub fn restate(&self, triangle: &Triangle, column: &str) -> Result<Triangle> {
        let base = self.index.value(self.base)?;
        triangle.scale_increments(column, |month| Ok(base / self.index.value(month)?))
    }

    /// The inverse of [`restate`](Self::restate): the increments of
    /// `column`, at the base level, taken to the money of their payment
    /// periods.
    ///
    /// Fails as [`restate`](Self::restate).
    pub fn inflate(&self, triangle: &Triangle, column: &str) -> Result<Triangle> {
        triangle.scale_increments(column, |month| self.factor(month))
    }

    /// Future increments at the base level re-inflated to the money of
    /// their payment periods, and summed by origin.
    ///
    /// Fails if the index does not cover a cell's payment period (set a
    /// [future rate](CalendarIndex::with_future_rate)).
    pub fn reinflate(&self, cells: &FutureCells) -> Result<Reinflated> {
        let n = cells.origins.len();
        let mut reserves = vec![0.0; n];
        let mut base_reserves = vec![0.0; n];
        let mut out = Vec::with_capacity(cells.cells.len());
        for cell in &cells.cells {
            if cell.origin >= n {
                return inflation_err(format!(
                    "a future cell's origin position {} is past the {n} origins",
                    cell.origin
                ));
            }
            let value = cell.value * self.factor(cell.valuation)?;
            reserves[cell.origin] += value;
            base_reserves[cell.origin] += cell.value;
            out.push(FutureCell { value, ..*cell });
        }
        Ok(Reinflated {
            cells: FutureCells {
                origins: cells.origins.clone(),
                cells: out,
            },
            reserves,
            base_reserves,
        })
    }

    /// Simulated future increments at the base level re-inflated to the
    /// money of their payment periods, one index path per simulation
    /// (decision 11).
    ///
    /// `draws` is keyed by cell: an `origin` dimension (a period, or a
    /// year as an integer or a label) and an age in months, named
    /// `development` (as [`OdpGlmFit::predict_distribution`]) or `age` (as
    /// the compartmental model's paths); other dimensions (a line) are
    /// kept. `paths` holds
    /// one [`CalendarIndex`] per simulation, simulation `i`'s increments
    /// scaled by `I_i(p) / I_i(base)`, or is empty for [`index`](Self::index)
    /// in every simulation, which matches [`reinflate`](Self::reinflate) on
    /// each simulation. The result has the same components and simulation
    /// order; [`aggregate`](PredictiveDistribution::aggregate) by `origin`
    /// gives the re-inflated reserves.
    ///
    /// ```
    /// use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
    /// use prospicio_reserving::{CalendarIndex, Indexation, Month, Period};
    ///
    /// // One cell, origin 2022 at 24 months (paid in 2023), two simulations.
    /// let draws = PredictiveDistribution::from_draws(
    ///     vec!["origin".into(), "development".into()],
    ///     vec![vec![KeyValue::Period(Period::year(2022)), KeyValue::Int(24)]],
    ///     vec![100.0, 200.0],
    ///     Provenance::new("example"),
    /// )?;
    /// let history = |rate| CalendarIndex::new(Period::year(2022), vec![1.0])?.with_future_rate(rate);
    /// let at_2022 = Indexation { index: history(0.05)?, base: Month::january(2022) };
    /// // Simulation 0 sees 10% inflation in 2023, simulation 1 sees none.
    /// let paths = [history(0.10)?, history(0.0)?];
    /// let out = at_2022.reinflate_draws(&draws, &paths)?;
    /// assert!((out.row(0).unwrap()[0] - 110.0).abs() < 1e-9);
    /// assert_eq!(out.row(1).unwrap()[0], 200.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Fails with [`Error::Inflation`] if `paths` is neither empty nor one
    /// per simulation, the draws have no origin or age dimension, a key is
    /// not a period or year and an age in months, or a path does not cover
    /// the base or a cell's payment period.
    pub fn reinflate_draws(
        &self,
        draws: &PredictiveDistribution,
        paths: &[CalendarIndex],
    ) -> Result<PredictiveDistribution> {
        let n_sims = draws.n_sims();
        if !paths.is_empty() && paths.len() != n_sims {
            return inflation_err(format!(
                "{} index paths for {n_sims} simulations: give one per simulation, or none",
                paths.len()
            ));
        }
        let dims = draws.dims();
        let position = |names: &[&str]| dims.iter().position(|d| names.contains(&d.as_str()));
        let Some(o) = position(&["origin"]) else {
            return inflation_err("the draws have no \"origin\" dimension");
        };
        let Some(a) = position(&["development", "age"]) else {
            return inflation_err("the draws have no \"development\" or \"age\" dimension");
        };
        let valuations = draws
            .components()
            .iter()
            .map(|key| {
                let origin = match &key[o] {
                    KeyValue::Period(p) => *p,
                    KeyValue::Int(year) => match i32::try_from(*year) {
                        Ok(y) => Period::year(y),
                        Err(_) => {
                            return inflation_err(format!("origin year {year} is out of range"));
                        }
                    },
                    KeyValue::Text(label) => match label.trim().parse::<i32>() {
                        Ok(y) => Period::year(y),
                        Err(_) => {
                            return inflation_err(format!(
                                "origin {label:?} is not a period or a year"
                            ));
                        }
                    },
                };
                let age = match &key[a] {
                    KeyValue::Int(m) if *m >= 1 => *m,
                    other => {
                        return inflation_err(format!(
                            "age {other} is not a positive number of months"
                        ));
                    }
                };
                Ok(origin.start().add_months(age - 1))
            })
            .collect::<Result<Vec<Month>>>()?;
        let factors = |index: &CalendarIndex| -> Result<Vec<f64>> {
            let base = index.value(self.base)?;
            valuations
                .iter()
                .map(|&m| Ok(index.value(m)? / base))
                .collect()
        };
        let fixed = if paths.is_empty() {
            Some(factors(&self.index)?)
        } else {
            None
        };
        let width = valuations.len();
        let mut out = draws.draw_matrix().to_vec();
        for (i, row) in out.chunks_exact_mut(width).enumerate() {
            let path;
            let f = match &fixed {
                Some(f) => f,
                None => {
                    path = factors(&paths[i])?;
                    &path
                }
            };
            for (x, k) in row.iter_mut().zip(f) {
                *x *= k;
            }
        }
        let provenance = Provenance::new("reinflate_draws")
            .param("base", self.base)
            .param("paths", if paths.is_empty() { 1 } else { paths.len() })
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"))
            .draws_from(draws.provenance());
        Ok(PredictiveDistribution::from_draws(
            dims.to_vec(),
            draws.components().to_vec(),
            out,
            provenance,
        )?)
    }
}

/// One projected future increment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FutureCell {
    /// Position of the cell's origin in [`FutureCells::origins`].
    pub origin: usize,
    /// Development age in months.
    pub age: Lag,
    /// The month the cell's development period ends in, whose index
    /// period is its payment period.
    pub valuation: Month,
    /// The increment.
    pub value: f64,
}

/// Projected future increments of one segment, one per origin and age,
/// for re-inflation (decision 6).
#[derive(Debug, Clone, PartialEq)]
pub struct FutureCells {
    /// The origins the cells' positions refer to.
    pub origins: Vec<Period>,
    /// The cells, origin by origin, ages increasing.
    pub cells: Vec<FutureCell>,
}

impl FutureCells {
    /// A chain ladder's future increments: at each age of the grid after
    /// an origin's latest one, the projected cumulative value less the one
    /// before; then the rest of the ultimate (the tail) as one increment
    /// one development period after the oldest age (decision 5), where the
    /// tail adds anything (more than round-off). The cells sum to
    /// [`ChainLadderFit::reserves`]. On a grid of one age the development
    /// period is taken as the origin's.
    pub fn from_chain_ladder(fit: &ChainLadderFit) -> Self {
        let ages = &fit.development.development;
        let step = match ages[..] {
            [.., before, last] => last - before,
            _ => fit.origins.first().map_or(12, |p| p.grain().months()),
        };
        let mut cells = Vec::new();
        for (o, origin) in fit.origins.iter().enumerate() {
            let projection = fit.projection(o);
            let start = fit.latest_position[o];
            let cell = |age: Lag, value: f64| FutureCell {
                origin: o,
                age,
                valuation: origin.start().add_months(age as i64 - 1),
                value,
            };
            for (k, pair) in projection.windows(2).enumerate() {
                cells.push(cell(ages[start + k + 1], pair[1] - pair[0]));
            }
            let last = projection[projection.len() - 1];
            // The ultimate is the latest times the cdf, multiplied in the
            // other order, so a fit without a tail leaves round-off here.
            let tail = fit.ultimate[o] - last;
            if tail.abs() > 1e-12 * fit.ultimate[o].abs().max(1.0) {
                let oldest = ages.last().copied().unwrap_or(0);
                cells.push(cell(oldest + step, tail));
            }
        }
        Self {
            origins: fit.origins.clone(),
            cells,
        }
    }

    /// An [`OdpGlmFit`]'s future cells and their means.
    ///
    /// Fails with [`Error::Inflation`] if a future cell's origin is not
    /// among the fit's origins, which a fit from
    /// [`OdpGlm`](crate::OdpGlm) never has.
    pub fn from_odp_glm(fit: &OdpGlmFit) -> Result<Self> {
        let cells = fit
            .future
            .iter()
            .zip(&fit.future_means)
            .map(|(&(period, age), &value)| {
                let Some(origin) = fit.origins.iter().position(|&p| p == period) else {
                    return inflation_err(format!(
                        "a future cell's origin {period} is not one of the fit's origins"
                    ));
                };
                Ok(FutureCell {
                    origin,
                    age,
                    valuation: period.start().add_months(age as i64 - 1),
                    value,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            origins: fit.origins.clone(),
            cells,
        })
    }

    /// The sum of the cells by origin; a cell whose origin position is past
    /// the origins is left out.
    pub fn reserves(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.origins.len()];
        for cell in &self.cells {
            if let Some(total) = out.get_mut(cell.origin) {
                *total += cell.value;
            }
        }
        out
    }
}

/// A projection re-inflated by [`Indexation::reinflate`].
#[derive(Debug, Clone, PartialEq)]
pub struct Reinflated {
    /// The re-inflated cells.
    pub cells: FutureCells,
    /// The re-inflated reserve of each origin of `cells.origins`.
    pub reserves: Vec<f64>,
    /// The reserve of each origin at the base level, before re-inflation.
    pub base_reserves: Vec<f64>,
}

impl Reinflated {
    /// The re-inflated total reserve.
    pub fn total_reserve(&self) -> f64 {
        self.reserves.iter().sum()
    }

    /// The total reserve at the base level.
    pub fn total_base_reserve(&self) -> f64 {
        self.base_reserves.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain_ladder::ChainLadder;
    use crate::odp_glm::OdpGlm;
    use crate::triangle::tests::{annual, genins};
    use prospicio_core::Grain;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1.0)
    }

    /// Increments `r_j (1 + f)^(i + j)` of a `rows` by `rows` triangle,
    /// cumulated, origins from 2001.
    fn inflated(r: &[f64], f: f64) -> Triangle {
        let k = r.len();
        let rows: Vec<Vec<f64>> = (0..k)
            .map(|i| {
                let mut total = 0.0;
                (0..k - i)
                    .map(|j| {
                        total += r[j] * (1.0 + f).powi((i + j) as i32);
                        total
                    })
                    .collect()
            })
            .collect();
        let rows: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        annual(2001, &rows)
    }

    /// The same cells, mask and basis, values to 1e-12 relative.
    fn same(got: &Triangle, want: &Triangle) {
        assert_eq!(got.shape(), want.shape());
        assert_eq!(got.is_cumulative(), want.is_cumulative());
        let [_, _, no, nd] = want.shape();
        for o in 0..no {
            for d in 0..nd {
                match (got.get(0, 0, o, d), want.get(0, 0, o, d)) {
                    (Some(a), Some(b)) => assert!(close(a, b, 1e-12), "{o} {d}: {a} {b}"),
                    (None, None) => {}
                    other => panic!("mask differs at {o} {d}: {other:?}"),
                }
            }
        }
    }

    fn yearly(first: i32, rate: f64, n: usize) -> CalendarIndex {
        CalendarIndex::new(
            Period::year(first),
            (0..n).map(|k| (1.0 + rate).powi(k as i32)).collect(),
        )
        .unwrap()
    }

    #[test]
    fn the_index_reads_its_periods_and_extends_at_the_rate() {
        let q = CalendarIndex::new(
            Period::containing(Month::new(2020, 4).unwrap(), Grain::Quarter),
            vec![100.0, 101.0, 103.0],
        )
        .unwrap();
        assert_eq!(q.value(Month::new(2020, 6).unwrap()).unwrap(), 100.0);
        assert_eq!(q.value(Month::new(2020, 12).unwrap()).unwrap(), 103.0);
        assert!(q.value(Month::new(2020, 3).unwrap()).is_err());
        assert!(q.value(Month::new(2021, 1).unwrap()).is_err());
        let q = q.with_future_rate(0.08).unwrap();
        let want = 103.0 * 1.08_f64.powf(0.25);
        assert!(close(
            q.value(Month::new(2021, 2).unwrap()).unwrap(),
            want,
            1e-15
        ));
        assert_eq!(
            q.periods()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            ["2020Q2", "2020Q3", "2020Q4"]
        );
        assert!(CalendarIndex::new(Period::year(2020), vec![]).is_err());
        assert!(CalendarIndex::new(Period::year(2020), vec![1.0, 0.0]).is_err());
        assert!(CalendarIndex::new(Period::year(2020), vec![f64::NAN]).is_err());
        assert!(yearly(2020, 0.0, 2).with_future_rate(-1.0).is_err());
        assert!(
            yearly(2020, 0.0, 2)
                .with_future_rate(f64::INFINITY)
                .is_err()
        );
    }

    #[test]
    fn a_unit_index_changes_nothing_and_inflate_undoes_restate() {
        let tri = genins();
        let unit = Indexation {
            index: yearly(2001, 0.0, 10),
            base: Month::january(2010),
        };
        same(&unit.restate(&tri, "values").unwrap(), &tri);
        let cpi = Indexation {
            index: CalendarIndex::new(
                Period::year(2001),
                vec![
                    100.0, 103.0, 101.5, 108.0, 112.0, 111.0, 118.0, 125.0, 131.0, 140.0,
                ],
            )
            .unwrap(),
            base: Month::new(2006, 7).unwrap(),
        };
        let restated = cpi.restate(&tri, "values").unwrap();
        assert!(restated.is_cumulative());
        same(&cpi.inflate(&restated, "values").unwrap(), &tri);
        // The first cell is paid in 2001: restated by 111 / 100.
        let want = tri.get(0, 0, 0, 0).unwrap() * 111.0 / 100.0;
        assert!(close(restated.get(0, 0, 0, 0).unwrap(), want, 1e-14));
    }

    #[test]
    fn a_change_of_base_scales_every_cell_alike() {
        let tri = genins();
        let index = yearly(2001, 0.07, 10);
        let at = |year| Indexation {
            index: index.clone(),
            base: Month::january(year),
        };
        let a = at(2003).restate(&tri, "values").unwrap();
        let b = at(2010).restate(&tri, "values").unwrap();
        let scale = 1.07_f64.powi(7);
        for o in 0..10 {
            for d in 0..10 - o {
                let (a, b) = (a.get(0, 0, o, d).unwrap(), b.get(0, 0, o, d).unwrap());
                assert!(close(b, a * scale, 1e-12));
            }
        }
    }

    #[test]
    fn a_constant_rate_comes_out_and_goes_back_in() {
        let r = [0.5, 0.3, 0.15, 0.05];
        let f = 0.06;
        let tri = inflated(&r, f);
        let ix = Indexation {
            index: yearly(2001, f, 4).with_future_rate(f).unwrap(),
            base: Month::january(2001),
        };
        let restated = ix.restate(&tri, "values").unwrap().to_incremental();
        for o in 0..4 {
            for (d, rd) in r.iter().enumerate().take(4 - o) {
                assert!(close(restated.get(0, 0, o, d).unwrap(), *rd, 1e-13));
            }
        }
        let fit = ChainLadder::default()
            .fit(&ix.restate(&tri, "values").unwrap(), "values")
            .unwrap();
        let cells = FutureCells::from_chain_ladder(&fit);
        assert_eq!(cells.cells.len(), 6);
        let out = ix.reinflate(&cells).unwrap();
        for cell in &out.cells.cells {
            let (i, j) = (cell.origin, (cell.age / 12 - 1) as usize);
            let want = r[j] * (1.0 + f).powi((i + j) as i32);
            assert!(close(cell.value, want, 1e-12), "{i} {j}");
        }
        let base: f64 = (1..4).flat_map(|i| (4 - i..4).map(move |j| r[j])).sum();
        assert!(close(out.total_base_reserve(), base, 1e-12));
    }

    #[test]
    fn chain_ladder_cells_sum_to_its_reserves_with_a_tail() {
        let tri = genins();
        let fit = ChainLadder {
            tail: crate::tail::Tail::from(1.05),
            ..ChainLadder::default()
        }
        .fit(&tri, "values")
        .unwrap();
        let cells = FutureCells::from_chain_ladder(&fit);
        for (got, want) in cells.reserves().iter().zip(fit.reserves()) {
            assert!(close(*got, want, 1e-12));
        }
        // Each origin's tail is one cell at age 132, paid in its 11th year.
        let first = cells.cells.iter().find(|c| c.origin == 0).unwrap();
        assert_eq!(first.age, 132);
        assert_eq!(first.valuation, Month::new(2011, 12).unwrap());
    }

    #[test]
    fn no_tail_needs_no_index_past_the_grid() {
        let tri = genins();
        let fit = ChainLadder::default().fit(&tri, "values").unwrap();
        let cells = FutureCells::from_chain_ladder(&fit);
        assert_eq!(cells.cells.len(), 45);
        assert!(cells.cells.iter().all(|c| c.age <= 120));
        let through_2019 = Indexation {
            index: yearly(2001, 0.05, 19),
            base: Month::january(2010),
        };
        assert!(through_2019.reinflate(&cells).is_ok());
    }

    #[test]
    fn odp_glm_cells_sum_to_its_reserves() {
        let fit = OdpGlm::default().fit(&genins(), "values").unwrap();
        let cells = FutureCells::from_odp_glm(&fit).unwrap();
        for (got, want) in cells.reserves().iter().zip(&fit.reserves) {
            assert!(close(*got, *want, 1e-12));
        }
    }

    /// One annual origin (2020) of `paid` and `count`, one row per age.
    fn one_origin(paid: &[f64], count: &[f64], cumulative: bool) -> Triangle {
        let n = paid.len();
        Triangle::from_long(&crate::Long {
            keys: &[],
            origin: &vec![Month::january(2020); n],
            development: crate::DevelopmentColumn::Age(
                &(1..=n as Lag).map(|k| 12 * k).collect::<Vec<_>>(),
            ),
            values: &[("paid", paid), ("count", count)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative,
        })
        .unwrap()
    }

    /// 1.0, 1.1, 1.21 for 2020 to 2022, at the 2020 level.
    fn tenth() -> Indexation {
        Indexation {
            index: CalendarIndex::new(Period::year(2020), vec![1.0, 1.1, 1.21]).unwrap(),
            base: Month::january(2020),
        }
    }

    #[test]
    fn an_incremental_triangle_is_restated_cell_by_cell() {
        // Increments 100, 55, 30.25 are 100, 50, 25 at 2020 prices.
        let tri = one_origin(&[100.0, 55.0, 30.25], &[1.0, 2.0, 3.0], false);
        let restated = tenth().restate(&tri, "paid").unwrap();
        assert!(!restated.is_cumulative());
        for (d, want) in [100.0, 50.0, 25.0].into_iter().enumerate() {
            assert!(close(restated.get(0, 0, 0, d).unwrap(), want, 1e-14), "{d}");
            // The other column is as it was.
            assert_eq!(restated.get(0, 1, 0, d), tri.get(0, 1, 0, d));
        }
        same(&tenth().inflate(&restated, "paid").unwrap(), &tri);
    }

    #[test]
    fn an_increment_after_a_hole_is_scaled_at_its_own_cells_month() {
        // Cumulative 100, (missing), 300: the third increment, 200, covers
        // two periods and is restated at the third cell's, 2022, level.
        let tri = one_origin(&[100.0, f64::NAN, 300.0], &[1.0, 2.0, 3.0], true);
        assert_eq!(tri.get(0, 0, 0, 1), None);
        let restated = tenth().restate(&tri, "paid").unwrap();
        assert_eq!(restated.get(0, 0, 0, 1), None);
        assert!(close(restated.get(0, 0, 0, 0).unwrap(), 100.0, 1e-14));
        assert!(close(
            restated.get(0, 0, 0, 2).unwrap(),
            100.0 + 200.0 / 1.21,
            1e-14
        ));
        same(&tenth().inflate(&restated, "paid").unwrap(), &tri);
    }

    #[test]
    fn every_segment_is_restated_and_other_columns_are_untouched() {
        let a = one_origin(&[100.0, 155.0], &[1.0, 2.0], true);
        let b = one_origin(&[200.0, 310.0], &[5.0, 6.0], true);
        let stacked =
            Triangle::stack(&[("A", &a), ("B", &b)], "program", &["paid", "count"]).unwrap();
        let restated = tenth().restate(&stacked, "paid").unwrap();
        assert_eq!(restated.shape(), stacked.shape());
        for (i, first) in [(0, 100.0), (1, 200.0)] {
            assert!(close(restated.get(i, 0, 0, 0).unwrap(), first, 1e-14));
            assert!(close(restated.get(i, 0, 0, 1).unwrap(), 1.5 * first, 1e-14));
            for d in 0..2 {
                assert_eq!(restated.get(i, 1, 0, d), stacked.get(i, 1, 0, d));
            }
        }
        same_all(&tenth().inflate(&restated, "paid").unwrap(), &stacked);
    }

    /// Every segment and column of `want`, values to 1e-12 relative.
    fn same_all(got: &Triangle, want: &Triangle) {
        assert_eq!(got.shape(), want.shape());
        let [ni, nc, no, nd] = want.shape();
        for (i, c, o, d) in (0..ni).flat_map(|i| {
            (0..nc).flat_map(move |c| (0..no).flat_map(move |o| (0..nd).map(move |d| (i, c, o, d))))
        }) {
            match (got.get(i, c, o, d), want.get(i, c, o, d)) {
                (Some(a), Some(b)) => assert!(close(a, b, 1e-12), "{i} {c} {o} {d}"),
                (None, None) => {}
                other => panic!("mask differs at {i} {c} {o} {d}: {other:?}"),
            }
        }
    }

    #[test]
    fn a_period_the_index_misses_is_an_error() {
        let tri = genins();
        let late = Indexation {
            index: yearly(2002, 0.05, 9),
            base: Month::january(2005),
        };
        assert!(matches!(
            late.restate(&tri, "values"),
            Err(Error::Inflation(_))
        ));
        let fit = ChainLadder::default().fit(&tri, "values").unwrap();
        let short = Indexation {
            index: yearly(2001, 0.05, 10),
            base: Month::january(2010),
        };
        assert!(
            short
                .reinflate(&FutureCells::from_chain_ladder(&fit))
                .is_err()
        );
        assert!(matches!(
            short.restate(&tri, "nope"),
            Err(Error::UnknownColumn(_))
        ));
    }
}
