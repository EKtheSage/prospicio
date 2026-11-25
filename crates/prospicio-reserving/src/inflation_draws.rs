//! Stochastic re-inflation of simulated future cells (`reserving-inflation.md`,
//! PR 8, decision 11, issue #979).
//!
//! [`Indexation::reinflate`] takes one projection's future cells to the
//! money of their payment periods. [`Indexation::reinflate_draws`] does the
//! same to every simulation of a [`PredictiveDistribution`] of future
//! cells, either with the index's own future rate or with one index path
//! per simulation ([`IndexPaths`]), so that future inflation is uncertain
//! too. The paths are an input: this module does not generate them.

use crate::error::{Error, Result};
use crate::inflation::Indexation;
use prospicio_core::{Month, Period};
use prospicio_prob::{KeyValue, PredictiveDistribution};

fn inflation_err<T>(why: impl Into<String>) -> Result<T> {
    Err(Error::Inflation(why.into()))
}

/// One index path per simulation for the periods after a
/// [`CalendarIndex`](crate::CalendarIndex)'s last given period: index
/// levels on the index's scale, not rates.
///
/// ```
/// use prospicio_reserving::{IndexPaths, Period};
///
/// // Two simulations of 2023 and 2024.
/// let paths = IndexPaths::new(Period::year(2023), 2, vec![1.10, 1.20, 1.05, 1.08])?;
/// assert_eq!(paths.n_periods(), 2);
/// assert_eq!(paths.path(1), Some(&[1.05, 1.08][..]));
/// assert_eq!(paths.periods(), [Period::year(2023), Period::year(2024)]);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IndexPaths {
    first: Period,
    n_sims: usize,
    values: Vec<f64>,
}

impl IndexPaths {
    /// Paths from `first` on, one period of `first`'s grain a value.
    /// `values` is simulation-major: `values[i * n_periods + k]` is the
    /// level of period `k` in simulation `i`, where `n_periods` is
    /// `values.len() / n_sims`.
    ///
    /// Fails unless `n_sims` is positive, `values.len()` is a positive
    /// multiple of `n_sims`, and every value is finite and positive.
    pub fn new(first: Period, n_sims: usize, values: Vec<f64>) -> Result<Self> {
        if n_sims == 0 {
            return inflation_err("index paths need at least one simulation");
        }
        if values.is_empty() || !values.len().is_multiple_of(n_sims) {
            return inflation_err(format!(
                "{} path values are not a positive multiple of {n_sims} simulations",
                values.len()
            ));
        }
        if let Some(&bad) = values.iter().find(|v| !v.is_finite() || **v <= 0.0) {
            return Err(Error::InvalidSetting {
                name: "paths",
                value: bad,
                expected: "finite and positive values",
            });
        }
        Ok(Self {
            first,
            n_sims,
            values,
        })
    }

    /// The first period of every path.
    pub fn first(&self) -> Period {
        self.first
    }

    /// The number of simulations (paths).
    pub fn n_sims(&self) -> usize {
        self.n_sims
    }

    /// The number of periods of each path.
    pub fn n_periods(&self) -> usize {
        self.values.len() / self.n_sims
    }

    /// The periods of the paths, from [`first`](Self::first) at its grain.
    pub fn periods(&self) -> Vec<Period> {
        let step = self.first.grain().months() as i64;
        (0..self.n_periods() as i64)
            .map(|k| {
                Period::containing(self.first.start().add_months(k * step), self.first.grain())
            })
            .collect()
    }

    /// Every path's levels, simulation-major.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The levels of simulation `sim`, or `None` past the last one.
    pub fn path(&self, sim: usize) -> Option<&[f64]> {
        let n = self.n_periods();
        self.values.get(sim * n..(sim + 1) * n)
    }
}

/// How one cell's factor is found in each simulation.
#[derive(Debug, Clone, Copy)]
enum Factor {
    /// The same factor in every simulation.
    Fixed(f64),
    /// The level at this position of each simulation's path, over the
    /// base level.
    Path(usize),
}

impl Indexation {
    /// Simulated future increments at the base level re-inflated to the
    /// money of their payment periods, simulation by simulation.
    ///
    /// `draws` holds one component per future cell. Its dimensions must
    /// include `origin`, each key a [`KeyValue::Period`], and exactly one of
    /// `development` or `age`, each key a [`KeyValue::Int`]: the age in
    /// months at the end of the cell's development period. The cell is paid
    /// in the month `age - 1` months after the origin's first month, as in
    /// [`FutureCells::from_odp_glm`](crate::FutureCells::from_odp_glm).
    /// Other dimensions (a line of business) pass through untouched.
    ///
    /// With `paths` `None`, each cell of every simulation is multiplied by
    /// the [`factor`](Self::factor) of its payment month, which gives
    /// [`reinflate`](Self::reinflate) applied to each simulation's cells:
    /// the index's future rate where it has one.
    ///
    /// With `paths`, simulation `i` of `draws` takes path `i`. The paths
    /// replace the future rate: a payment period inside the given index
    /// takes its given value, one from the paths' first period on takes
    /// that simulation's level, and one past the paths' last period is an
    /// error. The paths must start in the period right after the index's
    /// last given one, at the index's grain, and the base must be one of
    /// the given periods, so that the base level is the same in every
    /// simulation.
    ///
    /// The result has the same dimensions, components (in the same order)
    /// and simulations, and the input's provenance with a `reinflated`
    /// parameter. Its `aggregate(&["origin"])`
    /// ([`PredictiveDistribution::aggregate`]) gives the re-inflated
    /// reserves by origin, of the origins with a future cell.
    ///
    /// Which draws can be passed:
    /// [`OdpGlmFit::predict_distribution`](crate::OdpGlmFit::predict_distribution)
    /// and [`ApcGlmFit::predict_distribution`](crate::ApcGlmFit::predict_distribution)
    /// give future increments with dimensions `origin` and `development`.
    /// The bootstraps' reserve draws are by origin only (`origin`, after the
    /// triangle's segment keys if it has any), without cells, and are
    /// rejected for their missing age.
    /// The compartmental model's `paid` draws (dimensions `origin` and
    /// `age`) are cumulative paid, under incremental paid too, so they pass
    /// these checks but would be re-inflated wrongly: difference them by
    /// origin first (the first future cell less the origin's latest paid).
    ///
    /// Fails with [`Error::Inflation`] for a missing or mistyped origin or
    /// age, both `development` and `age`, an age below 1, paths that do not
    /// start right after the index or have another number of simulations,
    /// a payment period the index (or the paths) do not cover, or with
    /// paths a base outside the given periods.
    ///
    /// ```
    /// use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
    /// use prospicio_reserving::{CalendarIndex, IndexPaths, Indexation, Month, Period};
    ///
    /// // Origin 2022's cells at ages 24 and 36, paid in 2023 and 2024, at
    /// // the 2022 level, in two simulations.
    /// let cell = |age| vec![KeyValue::Period(Period::year(2022)), KeyValue::Int(age)];
    /// let draws = PredictiveDistribution::from_draws(
    ///     vec!["origin".into(), "development".into()],
    ///     vec![cell(24), cell(36)],
    ///     vec![100.0, 50.0, 120.0, 40.0],
    ///     Provenance::new("example"),
    /// )
    /// .unwrap();
    /// let at_2022 = Indexation {
    ///     index: CalendarIndex::new(Period::year(2021), vec![1.0, 1.1])?,
    ///     base: Month::january(2022),
    /// };
    /// let paths = IndexPaths::new(Period::year(2023), 2, vec![1.21, 1.331, 1.155, 1.21])?;
    /// let out = at_2022.reinflate_draws(&draws, Some(&paths))?;
    /// assert!((out.row(0).unwrap()[1] - 50.0 * 1.331 / 1.1).abs() < 1e-9);
    /// assert!((out.row(1).unwrap()[0] - 120.0 * 1.155 / 1.1).abs() < 1e-9);
    /// # Ok::<(), prospicio_reserving::Error>(())
    /// ```
    pub fn reinflate_draws(
        &self,
        draws: &PredictiveDistribution,
        paths: Option<&IndexPaths>,
    ) -> Result<PredictiveDistribution> {
        let months = payment_months(draws)?;
        let (factors, base) = match paths {
            None => (
                months
                    .iter()
                    .map(|&m| Ok(Factor::Fixed(self.factor(m)?)))
                    .collect::<Result<Vec<_>>>()?,
                1.0,
            ),
            Some(paths) => self.path_factors(&months, paths, draws.n_sims())?,
        };
        let n = draws.n_components();
        let mut out = Vec::with_capacity(draws.draw_matrix().len());
        for (sim, row) in draws.draw_matrix().chunks_exact(n).enumerate() {
            let path = paths.and_then(|p| p.path(sim)).unwrap_or(&[]);
            out.extend(row.iter().zip(&factors).map(|(x, factor)| match *factor {
                Factor::Fixed(f) => x * f,
                Factor::Path(k) => x * (path[k] / base),
            }));
        }
        let record = match (paths, self.index.future_rate()) {
            (Some(p), _) => format!(
                "base {}; {} index paths from {}",
                self.base,
                p.n_sims(),
                p.first()
            ),
            (None, Some(rate)) => format!("base {}; future rate {rate}", self.base),
            (None, None) => format!("base {}; given index", self.base),
        };
        Ok(PredictiveDistribution::from_draws(
            draws.dims().to_vec(),
            draws.components().to_vec(),
            out,
            draws.provenance().clone().param("reinflated", record),
        )?)
    }

    /// Each cell's factor with `paths`, and the base level, after checking
    /// the paths against the index and the draws.
    fn path_factors(
        &self,
        months: &[Month],
        paths: &IndexPaths,
        n_sims: usize,
    ) -> Result<(Vec<Factor>, f64)> {
        let given = self.index.values();
        let periods = self.index.periods();
        let (first, last) = (periods[0], periods[periods.len() - 1]);
        let grain = first.grain();
        if paths.first().grain() != grain {
            return inflation_err(format!(
                "the paths start in {}, at another grain than the index's ({first})",
                paths.first()
            ));
        }
        let next = Period::containing(last.start().add_months(grain.months() as i64), grain);
        if paths.first() != next {
            return inflation_err(format!(
                "the paths start in {}, not in {next}, the period after the index's last given one",
                paths.first()
            ));
        }
        if paths.n_sims() != n_sims {
            return inflation_err(format!(
                "{} index paths for {n_sims} simulations",
                paths.n_sims()
            ));
        }
        let position = |month: Month| {
            Period::containing(month, grain)
                .start()
                .months_since(first.start())
                .div_euclid(grain.months() as i64)
        };
        let n_given = given.len() as i64;
        let base = position(self.base);
        if !(0..n_given).contains(&base) {
            return inflation_err(format!(
                "the base {} is outside the given index ({first} to {last}), so its level would differ between paths",
                self.base
            ));
        }
        let base = given[base as usize];
        let factors = months
            .iter()
            .map(|&m| {
                let p = position(m);
                if p < 0 {
                    inflation_err(format!("the index starts in {first}, after {m}"))
                } else if p < n_given {
                    Ok(Factor::Fixed(given[p as usize] / base))
                } else if p - n_given < paths.n_periods() as i64 {
                    Ok(Factor::Path((p - n_given) as usize))
                } else {
                    let end = paths.periods()[paths.n_periods() - 1];
                    inflation_err(format!("the paths end in {end}, before {m}"))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((factors, base))
    }
}

/// The payment month of each component of `draws`: its origin's first
/// month plus its age less one month.
fn payment_months(draws: &PredictiveDistribution) -> Result<Vec<Month>> {
    let dims = draws.dims();
    let find = |name: &str| dims.iter().position(|d| d == name);
    let Some(origin) = find("origin") else {
        return inflation_err(format!(
            "the draws have no origin dimension (dimensions {dims:?}): re-inflation needs future cells by origin and age"
        ));
    };
    let age = match (find("development"), find("age")) {
        (Some(a), None) | (None, Some(a)) => a,
        (Some(_), Some(_)) => {
            return inflation_err("the draws have both a development and an age dimension");
        }
        (None, None) => {
            return inflation_err(format!(
                "the draws have no development or age dimension (dimensions {dims:?}): reserve draws by origin have no cells to re-inflate"
            ));
        }
    };
    draws
        .components()
        .iter()
        .map(|key| {
            let period = match &key[origin] {
                KeyValue::Period(p) => *p,
                other => {
                    return inflation_err(format!("the origin key {other} is not a period"));
                }
            };
            let months = match &key[age] {
                KeyValue::Int(a) if *a >= 1 => *a,
                KeyValue::Int(a) => {
                    return inflation_err(format!(
                        "the age {a} is not a positive number of months"
                    ));
                }
                other => {
                    return inflation_err(format!(
                        "the {} key {other} is not an age in months",
                        dims[age]
                    ));
                }
            };
            Ok(period.start().add_months(months - 1))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inflation::{CalendarIndex, FutureCell, FutureCells};
    use crate::odp_glm::OdpGlm;
    use crate::triangle::tests::genins;
    use prospicio_core::Grain;
    use prospicio_prob::Provenance;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1.0)
    }

    fn origin(year: i32) -> KeyValue {
        KeyValue::Period(Period::year(year))
    }

    /// The future cells of origins 2021 and 2022 of a triangle valued at
    /// the end of 2022: (2021, 36), (2022, 24), (2022, 36).
    const CELLS: [(i32, i64); 3] = [(2021, 36), (2022, 24), (2022, 36)];

    /// Three simulations of [`CELLS`].
    const DRAWS: [f64; 9] = [10.0, 30.0, 12.0, 8.0, 25.0, 9.0, 13.0, 41.0, 15.0];

    fn draws() -> PredictiveDistribution {
        PredictiveDistribution::from_draws(
            vec!["origin".into(), "development".into()],
            CELLS
                .iter()
                .map(|&(o, a)| vec![origin(o), KeyValue::Int(a)])
                .collect(),
            DRAWS.to_vec(),
            Provenance::new("test"),
        )
        .unwrap()
    }

    /// 2020 to 2022 annual, base 2021.
    fn index() -> Indexation {
        Indexation {
            index: CalendarIndex::new(Period::year(2020), vec![100.0, 104.0, 109.0]).unwrap(),
            base: Month::new(2021, 7).unwrap(),
        }
    }

    /// Simulation `sim` of [`DRAWS`] as future cells.
    fn cells_of(sim: usize) -> FutureCells {
        let origins = vec![Period::year(2021), Period::year(2022)];
        let cells = CELLS
            .iter()
            .enumerate()
            .map(|(j, &(o, a))| FutureCell {
                origin: (o - 2021) as usize,
                age: a as u32,
                valuation: Period::year(o).start().add_months(a - 1),
                value: DRAWS[3 * sim + j],
            })
            .collect();
        FutureCells { origins, cells }
    }

    #[test]
    fn index_paths_read_their_simulations_and_check_their_inputs() {
        let q = Period::containing(Month::new(2023, 4).unwrap(), Grain::Quarter);
        let paths = IndexPaths::new(q, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(paths.first(), q);
        assert_eq!((paths.n_sims(), paths.n_periods()), (2, 3));
        assert_eq!(paths.path(1), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(paths.path(2), None);
        assert_eq!(paths.values().len(), 6);
        assert_eq!(
            paths
                .periods()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            ["2023Q2", "2023Q3", "2023Q4"]
        );
        let year = Period::year(2023);
        assert!(matches!(
            IndexPaths::new(year, 0, vec![1.0]),
            Err(Error::Inflation(_))
        ));
        assert!(matches!(
            IndexPaths::new(year, 2, vec![]),
            Err(Error::Inflation(_))
        ));
        assert!(matches!(
            IndexPaths::new(year, 2, vec![1.0, 2.0, 3.0]),
            Err(Error::Inflation(_))
        ));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                IndexPaths::new(year, 2, vec![1.0, bad]),
                Err(Error::InvalidSetting { .. })
            ));
        }
    }

    /// The doc example of [`IndexPaths`] and of
    /// [`Indexation::reinflate_draws`], as unit tests.
    #[test]
    fn the_doc_examples_hold() {
        let paths = IndexPaths::new(Period::year(2023), 2, vec![1.10, 1.20, 1.05, 1.08]).unwrap();
        assert_eq!(paths.n_periods(), 2);
        assert_eq!(paths.path(1), Some(&[1.05, 1.08][..]));
        assert_eq!(paths.periods(), [Period::year(2023), Period::year(2024)]);

        let cell = |age| vec![KeyValue::Period(Period::year(2022)), KeyValue::Int(age)];
        let draws = PredictiveDistribution::from_draws(
            vec!["origin".into(), "development".into()],
            vec![cell(24), cell(36)],
            vec![100.0, 50.0, 120.0, 40.0],
            Provenance::new("example"),
        )
        .unwrap();
        let at_2022 = Indexation {
            index: CalendarIndex::new(Period::year(2021), vec![1.0, 1.1]).unwrap(),
            base: Month::january(2022),
        };
        let paths = IndexPaths::new(Period::year(2023), 2, vec![1.21, 1.331, 1.155, 1.21]).unwrap();
        let out = at_2022.reinflate_draws(&draws, Some(&paths)).unwrap();
        assert!((out.row(0).unwrap()[1] - 50.0 * 1.331 / 1.1).abs() < 1e-9);
        assert!((out.row(1).unwrap()[0] - 120.0 * 1.155 / 1.1).abs() < 1e-9);
    }

    #[test]
    fn without_paths_each_simulation_is_reinflate_of_its_cells() {
        let ix = Indexation {
            index: index().index.with_future_rate(0.06).unwrap(),
            ..index()
        };
        let out = ix.reinflate_draws(&draws(), None).unwrap();
        assert_eq!(out.dims(), draws().dims());
        assert_eq!(out.components(), draws().components());
        assert_eq!(out.n_sims(), 3);
        for sim in 0..3 {
            let want = ix.reinflate(&cells_of(sim)).unwrap();
            for (got, cell) in out.row(sim).unwrap().iter().zip(&want.cells.cells) {
                assert_eq!(*got, cell.value, "{sim}");
            }
        }
        // (2021, 36) is paid in 2023, one year past the index at 6%.
        let want = 10.0 * 109.0 * 1.06 / 104.0;
        assert!(close(out.row(0).unwrap()[0], want, 1e-14));
        assert!(
            out.provenance()
                .parameters
                .iter()
                .any(|(k, v)| k == "reinflated" && v.contains("future rate 0.06"))
        );
        // Without a future rate the 2023 payments are not covered.
        assert!(matches!(
            index().reinflate_draws(&draws(), None),
            Err(Error::Inflation(_))
        ));
    }

    #[test]
    fn paths_at_the_future_rate_reproduce_no_paths() {
        let rate = 0.045;
        let ix = Indexation {
            index: index().index.with_future_rate(rate).unwrap(),
            ..index()
        };
        let path: Vec<f64> = (1..=3).map(|k| 109.0 * (1.0 + rate).powi(k)).collect();
        let paths = IndexPaths::new(Period::year(2023), 3, path.repeat(3)).unwrap();
        let with = ix.reinflate_draws(&draws(), Some(&paths)).unwrap();
        let without = ix.reinflate_draws(&draws(), None).unwrap();
        for (a, b) in with.draw_matrix().iter().zip(without.draw_matrix()) {
            assert!(close(*a, *b, 1e-14), "{a} {b}");
        }
    }

    #[test]
    fn each_simulation_takes_its_own_path() {
        // 2023 and 2024: simulation i has 2023 at 110 + i, 2024 at 120 + 2i.
        let levels: Vec<f64> = (0..3)
            .flat_map(|i| [110.0 + i as f64, 120.0 + 2.0 * i as f64])
            .collect();
        let paths = IndexPaths::new(Period::year(2023), 3, levels.clone()).unwrap();
        let out = index().reinflate_draws(&draws(), Some(&paths)).unwrap();
        for sim in 0..3 {
            let row = out.row(sim).unwrap();
            let (y23, y24) = (levels[2 * sim], levels[2 * sim + 1]);
            // (2021, 36) and (2022, 24) are paid in 2023, (2022, 36) in 2024.
            let want = [
                DRAWS[3 * sim] * y23 / 104.0,
                DRAWS[3 * sim + 1] * y23 / 104.0,
                DRAWS[3 * sim + 2] * y24 / 104.0,
            ];
            for (g, w) in row.iter().zip(want) {
                assert!(close(*g, w, 1e-15), "{sim}: {g} {w}");
            }
        }
    }

    #[test]
    fn a_payment_inside_the_given_index_takes_its_given_value() {
        // The index runs to 2023, so the 2023 payments use 112 in every
        // simulation and only (2022, 36), paid in 2024, takes the paths.
        let ix = Indexation {
            index: CalendarIndex::new(Period::year(2020), vec![100.0, 104.0, 109.0, 112.0])
                .unwrap(),
            base: Month::january(2020),
        };
        let paths = IndexPaths::new(Period::year(2024), 3, vec![115.0, 118.0, 121.0]).unwrap();
        let out = ix.reinflate_draws(&draws(), Some(&paths)).unwrap();
        for sim in 0..3 {
            let row = out.row(sim).unwrap();
            assert!(close(row[0], DRAWS[3 * sim] * 1.12, 1e-15));
            assert!(close(row[1], DRAWS[3 * sim + 1] * 1.12, 1e-15));
            let level = [115.0, 118.0, 121.0][sim];
            assert!(close(row[2], DRAWS[3 * sim + 2] * level / 100.0, 1e-15));
        }
    }

    #[test]
    fn other_dimensions_pass_through() {
        let keys: Vec<Vec<KeyValue>> = ["motor", "home"]
            .iter()
            .flat_map(|lob| {
                CELLS
                    .iter()
                    .map(move |&(o, a)| vec![KeyValue::from(*lob), origin(o), KeyValue::Int(a)])
            })
            .collect();
        // Home's draws are twice motor's.
        let values: Vec<f64> = DRAWS
            .chunks(3)
            .flat_map(|row| {
                let mut both = row.to_vec();
                both.extend(row.iter().map(|x| 2.0 * x));
                both
            })
            .collect();
        let stacked = PredictiveDistribution::from_draws(
            vec!["lob".into(), "origin".into(), "age".into()],
            keys,
            values,
            Provenance::new("test"),
        )
        .unwrap();
        let ix = Indexation {
            index: index().index.with_future_rate(0.03).unwrap(),
            ..index()
        };
        let out = ix.reinflate_draws(&stacked, None).unwrap();
        assert_eq!(out.dims(), stacked.dims());
        assert_eq!(out.components(), stacked.components());
        let one = ix.reinflate_draws(&draws(), None).unwrap();
        for sim in 0..3 {
            let (row, want) = (out.row(sim).unwrap(), one.row(sim).unwrap());
            for j in 0..3 {
                assert!(close(row[j], want[j], 1e-15));
                assert!(close(row[3 + j], 2.0 * want[j], 1e-15));
            }
        }
        // By origin: motor and home summed.
        let by_origin = out.aggregate(&["origin"]).unwrap();
        assert_eq!(by_origin.dims(), ["origin"]);
        let o2022 = by_origin.component_index(&vec![origin(2022)]).unwrap();
        let first = one.row(0).unwrap();
        assert!(close(
            by_origin.row(0).unwrap()[o2022],
            3.0 * (first[1] + first[2]),
            1e-14
        ));
    }

    /// GenIns' ODP GLM draws: with paths, each simulation's cells are
    /// [`Indexation::reinflate`] under an index extended by that
    /// simulation's path as given values, cell by cell and by origin; and
    /// the reserve's mean under the future rate is the re-inflated fitted
    /// reserve, to simulation noise.
    #[test]
    fn odp_glm_draws_on_genins() {
        let fit = OdpGlm::default().fit(&genins(), "values").unwrap();
        let n_sims = 2_000;
        let pd = fit.predict_distribution(n_sims, 17).unwrap();
        let given: Vec<f64> = (0..10).map(|k| 100.0 * 1.04_f64.powi(k)).collect();
        let at_2006 = Indexation {
            index: CalendarIndex::new(Period::year(2001), given.clone())
                .unwrap()
                .with_future_rate(0.05)
                .unwrap(),
            base: Month::new(2006, 3).unwrap(),
        };
        // Nine future years, 2011 to 2019; each simulation its own rate.
        let path_of = |sim: usize| -> Vec<f64> {
            let rate = 0.01 + 0.06 * (sim % 7) as f64 / 6.0;
            (1..=9).map(|k| given[9] * (1.0 + rate).powi(k)).collect()
        };
        let levels: Vec<f64> = (0..n_sims).flat_map(path_of).collect();
        let paths = IndexPaths::new(Period::year(2011), n_sims, levels).unwrap();
        let out = at_2006.reinflate_draws(&pd, Some(&paths)).unwrap();
        let by_origin = out.aggregate(&["origin"]).unwrap();
        let template = FutureCells::from_odp_glm(&fit).unwrap();
        for sim in [0, 1, 5, 6, 999, n_sims - 1] {
            let extended = Indexation {
                index: CalendarIndex::new(
                    Period::year(2001),
                    [given.clone(), path_of(sim)].concat(),
                )
                .unwrap(),
                base: at_2006.base,
            };
            let mut cells = template.clone();
            for (cell, &x) in cells.cells.iter_mut().zip(pd.row(sim).unwrap()) {
                cell.value = x;
            }
            let want = extended.reinflate(&cells).unwrap();
            for (got, cell) in out.row(sim).unwrap().iter().zip(&want.cells.cells) {
                assert!(close(*got, cell.value, 1e-13), "{sim}");
            }
            for (o, period) in want.cells.origins.iter().enumerate() {
                match by_origin.component_index(&vec![KeyValue::Period(*period)]) {
                    Some(j) => assert!(close(
                        by_origin.row(sim).unwrap()[j],
                        want.reserves[o],
                        1e-12
                    )),
                    None => assert_eq!(want.reserves[o], 0.0),
                }
            }
        }

        let at_rate = at_2006.reinflate_draws(&pd, None).unwrap();
        let totals: Vec<f64> = (0..n_sims)
            .map(|s| at_rate.row(s).unwrap().iter().sum())
            .collect();
        let mean = totals.iter().sum::<f64>() / n_sims as f64;
        let sd =
            (totals.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n_sims - 1) as f64).sqrt();
        let se = sd / (n_sims as f64).sqrt();
        let want = at_2006.reinflate(&template).unwrap().total_reserve();
        assert!((mean - want).abs() < 4.0 * se, "{mean} {want} {se}");
        // Not the base-level reserve: the index moves it by more than the noise.
        assert!((fit.total_reserve() - want).abs() > 8.0 * se);
    }

    fn err(result: Result<PredictiveDistribution>) -> String {
        match result {
            Err(Error::Inflation(why)) => why,
            other => panic!("expected an inflation error, got {other:?}"),
        }
    }

    fn keyed(dims: &[&str], keys: Vec<Vec<KeyValue>>) -> PredictiveDistribution {
        let n = keys.len();
        PredictiveDistribution::from_draws(
            dims.iter().map(|d| d.to_string()).collect(),
            keys,
            vec![1.0; n],
            Provenance::new("test"),
        )
        .unwrap()
    }

    #[test]
    fn draws_without_cells_or_with_mistyped_keys_are_errors() {
        let ix = Indexation {
            index: index().index.with_future_rate(0.0).unwrap(),
            ..index()
        };
        let no_origin = keyed(&["development"], vec![vec![KeyValue::Int(24)]]);
        assert!(err(ix.reinflate_draws(&no_origin, None)).contains("no origin"));
        // A bootstrap's reserves: by origin only.
        let by_origin = keyed(&["origin"], vec![vec![origin(2021)]]);
        assert!(err(ix.reinflate_draws(&by_origin, None)).contains("no development or age"));
        let both = keyed(
            &["origin", "development", "age"],
            vec![vec![origin(2021), KeyValue::Int(24), KeyValue::Int(24)]],
        );
        assert!(err(ix.reinflate_draws(&both, None)).contains("both"));
        let text_origin = keyed(
            &["origin", "age"],
            vec![vec![KeyValue::from("2021"), KeyValue::Int(24)]],
        );
        assert!(err(ix.reinflate_draws(&text_origin, None)).contains("not a period"));
        let text_age = keyed(
            &["origin", "age"],
            vec![vec![origin(2021), KeyValue::from("24")]],
        );
        assert!(err(ix.reinflate_draws(&text_age, None)).contains("not an age"));
        let zero_age = keyed(
            &["origin", "age"],
            vec![vec![origin(2021), KeyValue::Int(0)]],
        );
        assert!(err(ix.reinflate_draws(&zero_age, None)).contains("not a positive"));
        // Paid before the index starts, with and without paths.
        let early = keyed(
            &["origin", "age"],
            vec![vec![origin(2018), KeyValue::Int(12)]],
        );
        assert!(err(ix.reinflate_draws(&early, None)).contains("starts in 2020"));
        let one = IndexPaths::new(Period::year(2023), 1, vec![110.0]).unwrap();
        assert!(err(index().reinflate_draws(&early, Some(&one))).contains("starts in 2020"));
    }

    #[test]
    fn paths_that_do_not_fit_the_index_or_the_draws_are_errors() {
        let year = |y, n: usize| IndexPaths::new(Period::year(y), 3, vec![110.0; 3 * n]).unwrap();
        // The index ends in 2022: the paths must start in 2023.
        for first in [2022, 2024] {
            let why = err(index().reinflate_draws(&draws(), Some(&year(first, 2))));
            assert!(why.contains("not in 2023"), "{why}");
        }
        let quarterly = IndexPaths::new(
            Period::containing(Month::january(2023), Grain::Quarter),
            3,
            vec![110.0; 24],
        )
        .unwrap();
        assert!(err(index().reinflate_draws(&draws(), Some(&quarterly))).contains("another grain"));
        let two = IndexPaths::new(Period::year(2023), 2, vec![110.0; 4]).unwrap();
        assert!(err(index().reinflate_draws(&draws(), Some(&two))).contains("2 index paths for 3"));
        // One period of paths: (2022, 36), paid in 2024, is past them.
        let why = err(index().reinflate_draws(&draws(), Some(&year(2023, 1))));
        assert!(why.contains("end in 2023"), "{why}");
        // A base past the given index: the future rate gives its level
        // without paths; with paths it would differ between them.
        let late_base = Indexation {
            index: index().index.with_future_rate(0.05).unwrap(),
            base: Month::january(2023),
        };
        assert!(late_base.reinflate_draws(&draws(), None).is_ok());
        let why = err(late_base.reinflate_draws(&draws(), Some(&year(2023, 2))));
        assert!(why.contains("outside"), "{why}");
        // With paths, a future rate on the index is not used.
        let with_rate = Indexation {
            index: index().index.with_future_rate(0.5).unwrap(),
            ..index()
        };
        let a = with_rate
            .reinflate_draws(&draws(), Some(&year(2023, 2)))
            .unwrap();
        let b = index()
            .reinflate_draws(&draws(), Some(&year(2023, 2)))
            .unwrap();
        assert_eq!(a.draw_matrix(), b.draw_matrix());
    }
}
