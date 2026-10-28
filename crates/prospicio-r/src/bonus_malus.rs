//! Aggregate lane: bonus-malus scales (`prospicio_pricing::bonus_malus`)
//! for R's `bonus_malus.R`.

use extendr_api::Result;
use extendr_api::prelude::*;
use prospicio_pricing::bonus_malus::{BonusMalus, BonusRule, Portfolio};
use prospicio_prob::count_families::Mixing;

use crate::{to_r, whole};

/// A bonus-malus scale.
#[extendr]
pub(crate) struct BonusMalusR {
    inner: BonusMalus,
}

fn level(x: f64, name: &str) -> Result<usize> {
    Ok(whole(x, name)? as usize)
}

#[extendr]
impl BonusMalusR {
    fn rule(levels: f64, start: f64, down: f64, up_per_claim: f64) -> Result<Self> {
        let rule = BonusRule {
            down: level(down, "down")?,
            up_per_claim: level(up_per_claim, "up_per_claim")?,
        };
        Ok(Self {
            inner: BonusMalus::rule(level(levels, "levels")?, level(start, "start")?, rule)
                .map_err(to_r)?,
        })
    }

    /// `table` row-major, `cols` columns; levels 0-based.
    fn table(start: f64, table: &[f64], cols: f64) -> Result<Self> {
        let cols = level(cols, "cols")?;
        if cols == 0 || table.len() % cols != 0 {
            return Err(Error::Other("the table is not a matrix".into()));
        }
        let rows = table
            .chunks(cols)
            .map(|r| {
                r.iter()
                    .map(|&v| level(v, "table"))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inner: BonusMalus::table(level(start, "start")?, rows).map_err(to_r)?,
        })
    }

    fn levels(&self) -> f64 {
        self.inner.levels() as f64
    }

    fn start(&self) -> f64 {
        self.inner.start() as f64
    }

    /// Row-major, 0-based levels.
    fn transitions(&self) -> Vec<f64> {
        self.inner
            .transitions()
            .iter()
            .flatten()
            .map(|&v| v as f64)
            .collect()
    }

    fn cols(&self) -> f64 {
        self.inner.transitions()[0].len() as f64
    }

    /// Row-major.
    fn transition_matrix(&self, lambda: f64) -> Result<Vec<f64>> {
        self.inner.transition_matrix(lambda).map_err(to_r)
    }

    fn stationary_at(&self, lambda: f64) -> Result<Vec<f64>> {
        self.inner.stationary_at(lambda).map_err(to_r)
    }

    fn distribution_after(&self, lambda: f64, years: f64) -> Result<Vec<f64>> {
        self.inner
            .distribution_after(lambda, level(years, "years")?)
            .map_err(to_r)
    }

    /// NA past 100,000 years.
    fn years_to_stationarity(&self, lambda: f64, tolerance: f64) -> Result<f64> {
        Ok(self
            .inner
            .years_to_stationarity(lambda, tolerance)
            .map_err(to_r)?
            .map_or(f64::NAN, |n| n as f64))
    }

    fn mean_relativity(&self, lambda: f64, relativities: &[f64]) -> Result<f64> {
        self.inner
            .mean_relativity(lambda, relativities)
            .map_err(to_r)
    }

    fn efficiency(&self, lambda: f64, relativities: &[f64]) -> Result<f64> {
        self.inner.efficiency(lambda, relativities).map_err(to_r)
    }

    /// The stationary distribution and optimal relativities over the
    /// portfolio.
    fn portfolio(&self, lambdas: &[f64], weights: &[f64], cv: f64, mixing: &str) -> Result<List> {
        let mixing = match mixing {
            "gamma" => Mixing::Gamma { cv },
            "inverse_gaussian" => Mixing::InverseGaussian { cv },
            other => {
                return Err(Error::Other(format!(
                    "mixing must be \"gamma\" or \"inverse_gaussian\", not {other:?}"
                )));
            }
        };
        let p = Portfolio::classes(weights.to_vec(), lambdas.to_vec(), mixing).map_err(to_r)?;
        Ok(list!(
            stationary = self.inner.stationary(&p).map_err(to_r)?,
            relativity = self.inner.optimal_relativities(&p).map_err(to_r)?
        ))
    }
}

extendr_module! {
    mod bonus_malus;
    impl BonusMalusR;
}
