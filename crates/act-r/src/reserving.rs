//! Reserving lane: wrappers over `act_reserving` (the Triangle, chain
//! ladder, Mack and the ODP bootstrap, `docs/design/triangle.md`) for the R
//! `reserving.R` API.
//!
//! Arrays cross the boundary flattened row-major over the Rust axes
//! (index, column, origin, development) with NaN where a cell is not
//! observed; R reshapes them. Months come in as (year, month) integer pairs.

use act_reserving::{
    Average, ChainLadder, ChainLadderFit as ChainLadderInner, Development, DevelopmentColumn,
    Grain, Label, Lag, Long, Mack, MackFit as MackInner, Month, OdpBootstrap,
    OdpBootstrapFit as OdpBootstrapInner, ProcessDistribution, SigmaInterpolation,
    Triangle as TriangleInner,
};
use extendr_api::prelude::*;
use extendr_api::{Error, Result};

use crate::distributions::PredictiveDistribution;
use crate::whole;

fn to_r(e: act_reserving::Error) -> Error {
    Error::Other(e.to_string())
}

fn grain(name: &str, arg: &str) -> Result<Grain> {
    match name {
        "M" => Ok(Grain::Month),
        "Q" => Ok(Grain::Quarter),
        "S" => Ok(Grain::Semester),
        "Y" => Ok(Grain::Year),
        _ => Err(Error::Other(format!(
            "{arg} must be \"M\", \"Q\", \"S\" or \"Y\", got \"{name}\""
        ))),
    }
}

fn development(average: &str, sigma_interpolation: &str) -> Result<Development> {
    let average = match average {
        "volume" => Average::Volume,
        "simple" => Average::Simple,
        "regression" => Average::Regression,
        _ => {
            return Err(Error::Other(format!(
                "average must be \"volume\", \"simple\" or \"regression\", got \"{average}\""
            )));
        }
    };
    let sigma_interpolation = match sigma_interpolation {
        "log-linear" => SigmaInterpolation::LogLinear,
        "mack" => SigmaInterpolation::Mack,
        _ => {
            return Err(Error::Other(format!(
                "sigma_interpolation must be \"log-linear\" or \"mack\", got \"{sigma_interpolation}\""
            )));
        }
    };
    Ok(Development {
        average,
        sigma_interpolation,
    })
}

fn months(years: &[i32], months: &[i32], arg: &str) -> Result<Vec<Month>> {
    if years.len() != months.len() {
        return Err(Error::Other(format!(
            "{arg}: {} years but {} months",
            years.len(),
            months.len()
        )));
    }
    years
        .iter()
        .zip(months)
        .map(|(&y, &m)| {
            let m = u8::try_from(m)
                .map_err(|_| Error::Other(format!("{arg}: month {m} is not in 1..12")))?;
            Month::new(y, m).map_err(crate::to_r)
        })
        .collect()
}

fn age(x: f64) -> Result<Lag> {
    Lag::try_from(whole(x, "development")?)
        .map_err(|_| Error::Other(format!("development age {x} is too large")))
}

/// Key columns as a named list of character vectors.
fn key_columns(keys: Vec<(String, Vec<String>)>) -> List {
    let (names, values): (Vec<String>, Vec<Vec<String>>) = keys.into_iter().unzip();
    List::from_names_and_values(names, values).expect("one name per key")
}

/// A loss triangle: index × column × origin × development.
#[extendr]
pub(crate) struct Triangle {
    inner: TriangleInner,
}

impl From<TriangleInner> for Triangle {
    fn from(inner: TriangleInner) -> Self {
        Self { inner }
    }
}

#[extendr]
impl Triangle {
    /// Builds a triangle from a long table. `keys` holds the key columns
    /// named `key_names`, column by column; `values` the value columns,
    /// column by column. Development is `ages`, or the valuation months
    /// when `development_is_valuation`.
    #[allow(clippy::too_many_arguments)]
    fn from_long(
        key_names: Vec<String>,
        keys: Vec<String>,
        origin_year: &[i32],
        origin_month: &[i32],
        ages: &[f64],
        valuation_year: &[i32],
        valuation_month: &[i32],
        development_is_valuation: bool,
        names: Vec<String>,
        values: &[f64],
        origin_grain: &str,
        development_grain: &str,
        cumulative: bool,
    ) -> Result<Self> {
        let origin = months(origin_year, origin_month, "origin")?;
        let n = origin.len();
        if keys.len() != n * key_names.len() {
            return Err(Error::Other(format!(
                "keys has {} entries, expected {n} rows times {} keys",
                keys.len(),
                key_names.len()
            )));
        }
        let key_values: Vec<Vec<&str>> = (0..key_names.len())
            .map(|k| {
                keys[k * n..(k + 1) * n]
                    .iter()
                    .map(String::as_str)
                    .collect()
            })
            .collect();
        let key_columns: Vec<(&str, &[&str])> = key_names
            .iter()
            .zip(&key_values)
            .map(|(name, v)| (name.as_str(), v.as_slice()))
            .collect();
        if values.len() != n * names.len() {
            return Err(Error::Other(format!(
                "values has {} entries, expected {n} rows times {} columns",
                values.len(),
                names.len()
            )));
        }
        let columns: Vec<(&str, &[f64])> = names
            .iter()
            .enumerate()
            .map(|(c, name)| (name.as_str(), &values[c * n..(c + 1) * n]))
            .collect();
        let valuations;
        let lags: Vec<Lag>;
        let development = if development_is_valuation {
            valuations = months(valuation_year, valuation_month, "development")?;
            DevelopmentColumn::Valuation(&valuations)
        } else {
            lags = ages.iter().map(|&a| age(a)).collect::<Result<_>>()?;
            DevelopmentColumn::Age(&lags)
        };
        let inner = TriangleInner::from_long(&Long {
            keys: &key_columns,
            origin: &origin,
            development,
            values: &columns,
            origin_grain: grain(origin_grain, "origin_grain")?,
            development_grain: grain(development_grain, "development_grain")?,
            cumulative,
        })
        .map_err(to_r)?;
        Ok(Self { inner })
    }

    fn shape(&self) -> Vec<i32> {
        self.inner.shape().iter().map(|&n| n as i32).collect()
    }

    /// Names of the key columns.
    fn keys(&self) -> Vec<String> {
        self.inner.key_names().to_vec()
    }

    /// Index labels as a named list of key columns, one row per label.
    fn index(&self) -> List {
        let index = self.inner.index();
        key_columns(
            self.inner
                .key_names()
                .iter()
                .enumerate()
                .map(|(k, name)| {
                    let values = index.iter().map(|l| l.parts()[k].clone()).collect();
                    (name.clone(), values)
                })
                .collect(),
        )
    }

    /// Index labels with their parts joined by " / " ("Total" without keys).
    fn index_names(&self) -> Vec<String> {
        self.inner.index().iter().map(Label::to_string).collect()
    }

    fn columns(&self) -> Vec<String> {
        self.inner.columns().to_vec()
    }

    fn origins(&self) -> Vec<String> {
        self.inner
            .origins()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    fn development(&self) -> Vec<i32> {
        self.inner.development().iter().map(|&a| a as i32).collect()
    }

    fn origin_grain(&self) -> String {
        self.inner.origin_grain().to_string()
    }

    fn development_grain(&self) -> String {
        self.inner.development_grain().to_string()
    }

    /// Valuation month as "YYYY-MM".
    fn valuation(&self) -> String {
        self.inner.valuation().to_string()
    }

    fn is_cumulative(&self) -> bool {
        self.inner.is_cumulative()
    }

    /// Values, row-major over (index, column, origin, development).
    fn values(&self) -> Vec<f64> {
        let [ni, nc, no, nd] = self.inner.shape();
        let mut out = Vec::with_capacity(ni * nc * no * nd);
        for i in 0..ni {
            for c in 0..nc {
                for o in 0..no {
                    for d in 0..nd {
                        out.push(self.inner.get(i, c, o, d).unwrap_or(f64::NAN));
                    }
                }
            }
        }
        out
    }

    /// The long table: the key columns (a named list), origin start (year,
    /// month), age and the value columns (NaN where a measure is not
    /// observed on a row).
    fn to_long(&self) -> List {
        let long = self.inner.to_long();
        let values = List::from_values(long.values.iter().map(|(_, v)| v.clone()));
        let names: Vec<String> = long.values.iter().map(|(n, _)| n.clone()).collect();
        list!(
            keys = key_columns(long.keys),
            origin_year = long.origin.iter().map(|m| m.year()).collect::<Vec<i32>>(),
            origin_month = long
                .origin
                .iter()
                .map(|m| m.month() as i32)
                .collect::<Vec<i32>>(),
            development = long
                .development
                .iter()
                .map(|&a| a as i32)
                .collect::<Vec<i32>>(),
            names = names,
            values = values
        )
    }

    fn to_incremental(&self) -> Self {
        self.inner.to_incremental().into()
    }

    fn to_cumulative(&self) -> Self {
        self.inner.to_cumulative().into()
    }

    /// Latest values, row-major over (index, column, origin); NaN where an
    /// origin has no observation.
    fn latest_diagonal(&self) -> Vec<f64> {
        let diagonal = self.inner.latest_diagonal();
        let [ni, nc, no] = diagonal.shape();
        let mut out = Vec::with_capacity(ni * nc * no);
        for i in 0..ni {
            for c in 0..nc {
                for o in 0..no {
                    out.push(diagonal.get(i, c, o).map_or(f64::NAN, |(_, v)| v));
                }
            }
        }
        out
    }

    fn link_ratios(&self) -> Self {
        self.inner.link_ratios().into()
    }

    /// Keeps the segments whose value of `keys[k]` is in `values[[k]]` (a
    /// character vector) for every `k`, then the named columns (`NULL`
    /// keeps every column).
    fn select(
        &self,
        keys: Vec<String>,
        values: List,
        columns: Nullable<Vec<String>>,
    ) -> Result<Self> {
        let values: Vec<Vec<String>> = values
            .values()
            .map(|v| {
                Vec::<String>::try_from(v)
                    .map_err(|_| Error::Other("key values must be character".into()))
            })
            .collect::<Result<_>>()?;
        if values.len() != keys.len() {
            return Err(Error::Other("one set of values is needed per key".into()));
        }
        let values: Vec<Vec<&str>> = values
            .iter()
            .map(|v| v.iter().map(String::as_str).collect())
            .collect();
        let conditions: Vec<(&str, &[&str])> = keys
            .iter()
            .zip(&values)
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        let mut inner = self.inner.select(&conditions).map_err(to_r)?;
        if let Nullable::NotNull(columns) = columns {
            let columns: Vec<&str> = columns.iter().map(String::as_str).collect();
            inner = inner.select_columns(&columns).map_err(to_r)?;
        }
        Ok(inner.into())
    }

    /// Sums the segments that share the values of `keys`.
    fn group_by(&self, keys: Vec<String>) -> Result<Self> {
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        Ok(self.inner.group_by(&keys).map_err(to_r)?.into())
    }

    fn grain(&self, origin_grain: &str, development_grain: &str) -> Result<Self> {
        let inner = self
            .inner
            .grain(
                grain(origin_grain, "origin_grain")?,
                grain(development_grain, "development_grain")?,
            )
            .map_err(to_r)?;
        Ok(inner.into())
    }

    fn chain_ladder(
        &self,
        column: &str,
        average: &str,
        sigma_interpolation: &str,
        tail: f64,
    ) -> Result<ChainLadderFit> {
        let inner = ChainLadder {
            development: development(average, sigma_interpolation)?,
            tail,
        }
        .fit(&self.inner, column)
        .map_err(to_r)?;
        Ok(ChainLadderFit { inner })
    }

    fn mack(&self, column: &str, average: &str, sigma_interpolation: &str) -> Result<MackFit> {
        let inner = Mack {
            development: development(average, sigma_interpolation)?,
        }
        .fit(&self.inner, column)
        .map_err(to_r)?;
        Ok(MackFit { inner })
    }

    fn odp_bootstrap(
        &self,
        column: &str,
        n_sims: f64,
        seed: f64,
        process: &str,
    ) -> Result<OdpBootstrapFit> {
        let n_sims = whole(n_sims, "n_sims")? as usize;
        if n_sims == 0 {
            return Err(Error::Other("n_sims must be positive".into()));
        }
        let process = match process {
            "gamma" => ProcessDistribution::Gamma,
            "none" => ProcessDistribution::None,
            _ => {
                return Err(Error::Other(format!(
                    "process must be \"gamma\" or \"none\", got \"{process}\""
                )));
            }
        };
        let inner = OdpBootstrap {
            n_sims,
            seed: whole(seed, "seed")?,
            process,
        }
        .fit(&self.inner, column)
        .map_err(to_r)?;
        Ok(OdpBootstrapFit { inner })
    }
}

/// A fitted chain ladder.
#[extendr]
pub(crate) struct ChainLadderFit {
    inner: ChainLadderInner,
}

#[extendr]
impl ChainLadderFit {
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    fn development(&self) -> Vec<i32> {
        let ages = &self.inner.development.development;
        ages.iter().map(|&a| a as i32).collect()
    }

    fn ldf(&self) -> Vec<f64> {
        self.inner.development.ldf.clone()
    }

    fn sigma(&self) -> Vec<f64> {
        self.inner.development.sigma.clone()
    }

    fn std_err(&self) -> Vec<f64> {
        self.inner.development.std_err.clone()
    }

    fn alpha(&self) -> f64 {
        self.inner.development.alpha
    }

    fn tail(&self) -> f64 {
        self.inner.tail
    }

    fn cdf(&self) -> Vec<f64> {
        self.inner.cdf.clone()
    }

    fn latest(&self) -> Vec<f64> {
        self.inner.latest.clone()
    }

    fn ultimate(&self) -> Vec<f64> {
        self.inner.ultimate.clone()
    }

    fn reserve(&self) -> Vec<f64> {
        self.inner.reserves()
    }

    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }
}

/// A fitted Mack chain ladder.
#[extendr]
pub(crate) struct MackFit {
    inner: MackInner,
}

#[extendr]
impl MackFit {
    /// The underlying chain-ladder projection.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.chain_ladder.clone(),
        }
    }

    fn process_risk(&self) -> Vec<f64> {
        self.inner.process_risk.clone()
    }

    fn parameter_risk(&self) -> Vec<f64> {
        self.inner.parameter_risk.clone()
    }

    fn standard_error(&self) -> Vec<f64> {
        self.inner.standard_error.clone()
    }

    fn total_process_risk(&self) -> f64 {
        self.inner.total_process_risk
    }

    fn total_parameter_risk(&self) -> f64 {
        self.inner.total_parameter_risk
    }

    fn total_standard_error(&self) -> f64 {
        self.inner.total_standard_error
    }

    fn total_cv(&self) -> f64 {
        self.inner.total_cv()
    }
}

/// A fitted ODP bootstrap. `fitted` and `residuals` are row-major over
/// origin x development, NaN where not observed.
#[extendr]
pub(crate) struct OdpBootstrapFit {
    inner: OdpBootstrapInner,
}

#[extendr]
impl OdpBootstrapFit {
    /// The chain ladder the bootstrap is centred on.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.chain_ladder.clone(),
        }
    }

    fn fitted(&self) -> Vec<f64> {
        self.inner.fitted.clone()
    }

    fn residuals(&self) -> Vec<f64> {
        self.inner.residuals.clone()
    }

    fn scale(&self) -> f64 {
        self.inner.scale
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }
}

extendr_module! {
    mod reserving;
    impl Triangle;
    impl ChainLadderFit;
    impl MackFit;
    impl OdpBootstrapFit;
}
