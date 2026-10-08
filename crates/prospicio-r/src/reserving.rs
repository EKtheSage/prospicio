//! Reserving lane: wrappers over `prospicio_reserving` (the Triangle, chain
//! ladder, Mack with its one-year view, tails, the expected-loss methods,
//! Clark's growth curves, and the ODP and Mack bootstraps with their simulated one-year
//! view, `docs/design/triangle.md`, `docs/design/reserving-v02.md`) for the
//! R `reserving.R` API.
//!
//! Arrays cross the boundary flattened row-major over the Rust axes
//! (index, column, origin, development) with NaN where a cell is not
//! observed; R reshapes them. Months come in as (year, month) integer pairs.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_reserving::{
    Average, Benktander, BornhuetterFerguson, CapeCod, CapeCodFit as CapeCodInner, ChainLadder,
    ChainLadderFit as ChainLadderInner, ClaimsDevelopmentResult as ClaimsDevelopmentInner,
    ClarkCapeCod, ClarkFit as ClarkInner, ClarkLdf, CurveShape, Development, DevelopmentColumn,
    ExpectedLoss, ExpectedLossFit as ExpectedLossInner, FitTable, Grain, GrowthCurve, Label, Lag,
    Long, Mack, MackBootstrap, MackBootstrapFits, MackBootstrapSegment, MackFit as MackInner,
    MackProcess, Month, OdpBootstrap, OdpBootstrapFits, OneYearFits, OneYearMethod,
    ProcessDistribution, ReserveFit, SegmentFits, SigmaInterpolation, Tail, TailBondy,
    TailConstant, TailCurve, Triangle as TriangleInner,
};

use crate::distributions::PredictiveDistribution;
use crate::whole;

fn to_r(e: prospicio_reserving::Error) -> Error {
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

/// The chain ladder that gives a method its development pattern.
fn pattern(average: &str, sigma_interpolation: &str, tail: Robj) -> Result<ChainLadder> {
    Ok(ChainLadder {
        development: development(average, sigma_interpolation)?,
        tail: tail_arg(&tail)?,
    })
}

/// The method of `odp_one_year()` and `mack_one_year()`, by name, with
/// the settings it reads and `chain_ladder` its development pattern.
fn one_year_method(
    method: &str,
    exposure: Nullable<String>,
    apriori: f64,
    n_iters: f64,
    trend: f64,
    decay: f64,
    chain_ladder: ChainLadder,
) -> Result<OneYearMethod> {
    let exposure = match exposure {
        Nullable::NotNull(e) => Some(e),
        Nullable::Null => None,
    };
    let needs = |e: Option<String>| {
        e.ok_or_else(|| {
            Error::Other(format!(
                "method \"{method}\" needs an exposure column: pass exposure = ..."
            ))
        })
    };
    Ok(match method {
        "chain_ladder" => {
            if exposure.is_some() {
                return Err(Error::Other(
                    "method \"chain_ladder\" takes no exposure column".into(),
                ));
            }
            OneYearMethod::ChainLadder(chain_ladder)
        }
        "expected_loss" => OneYearMethod::ExpectedLoss(
            ExpectedLoss {
                apriori,
                chain_ladder,
            },
            needs(exposure)?,
        ),
        "bornhuetter_ferguson" => OneYearMethod::BornhuetterFerguson(
            BornhuetterFerguson {
                apriori,
                chain_ladder,
            },
            needs(exposure)?,
        ),
        "benktander" => {
            let n_iters = usize::try_from(whole(n_iters, "n_iters")?)
                .map_err(|_| Error::Other(format!("n_iters {n_iters} is too large")))?;
            OneYearMethod::Benktander(
                Benktander {
                    apriori,
                    n_iters,
                    chain_ladder,
                },
                needs(exposure)?,
            )
        }
        "cape_cod" => OneYearMethod::CapeCod(
            CapeCod {
                trend,
                decay,
                chain_ladder,
            },
            needs(exposure)?,
        ),
        _ => {
            return Err(Error::Other(format!(
                "method must be \"chain_ladder\", \"expected_loss\", \
                 \"bornhuetter_ferguson\", \"benktander\" or \"cape_cod\", got \"{method}\""
            )));
        }
    })
}

/// Mack's bootstrap of `mack_bootstrap()` and `mack_one_year()`.
fn mack_bootstrap(
    n_sims: f64,
    seed: f64,
    process: &str,
    average: &str,
    sigma_interpolation: &str,
    centre_residuals: bool,
) -> Result<MackBootstrap> {
    let n_sims = whole(n_sims, "n_sims")? as usize;
    if n_sims == 0 {
        return Err(Error::Other("n_sims must be positive".into()));
    }
    let process = match process {
        "gamma" => MackProcess::Gamma,
        "lognormal" => MackProcess::Lognormal,
        "residuals" => MackProcess::Residuals,
        "normal" => MackProcess::Normal,
        "none" => MackProcess::None,
        _ => {
            return Err(Error::Other(format!(
                "process must be \"gamma\", \"lognormal\", \"residuals\", \"normal\" or \
                 \"none\", got \"{process}\""
            )));
        }
    };
    Ok(MackBootstrap {
        n_sims,
        seed: whole(seed, "seed")?,
        process,
        development: development(average, sigma_interpolation)?,
        centre_residuals,
    })
}

/// The ODP bootstrap of `odp_bootstrap()` and `odp_one_year()`.
fn bootstrap(n_sims: f64, seed: f64, process: &str) -> Result<OdpBootstrap> {
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
    Ok(OdpBootstrap {
        n_sims,
        seed: whole(seed, "seed")?,
        process,
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

fn growth_curve(name: &str) -> Result<GrowthCurve> {
    match name {
        "loglogistic" => Ok(GrowthCurve::LogLogistic),
        "weibull" => Ok(GrowthCurve::Weibull),
        _ => Err(Error::Other(format!(
            "curve must be \"loglogistic\" or \"weibull\", got \"{name}\""
        ))),
    }
}

fn growth_curve_name(curve: GrowthCurve) -> &'static str {
    match curve {
        GrowthCurve::LogLogistic => "loglogistic",
        GrowthCurve::Weibull => "weibull",
    }
}

/// Clark's `max_age`: R passes `Inf` to develop to infinity.
fn clark_max_age(x: f64) -> Option<f64> {
    (x != f64::INFINITY).then_some(x)
}

fn age(x: f64) -> Result<Lag> {
    Lag::try_from(whole(x, "development")?)
        .map_err(|_| Error::Other(format!("development age {x} is too large")))
}

/// An optional age in months named `name`; `NULL` is not given.
fn optional_age(x: Nullable<f64>, name: &str) -> Result<Option<Lag>> {
    match x {
        Nullable::NotNull(a) => Lag::try_from(whole(a, name)?)
            .map(Some)
            .map_err(|_| Error::Other(format!("{name} {a} is too large"))),
        Nullable::Null => Ok(None),
    }
}

fn optional(x: Nullable<f64>) -> Option<f64> {
    match x {
        Nullable::NotNull(v) => Some(v),
        Nullable::Null => None,
    }
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

    /// One segment (chosen by `keys` = `values`, one value each) and
    /// measure (`NULL`: the only one) as `origins`, `development` and
    /// `values` row-major over origin × development, NaN where unobserved.
    fn view(
        &self,
        keys: Vec<String>,
        values: Vec<String>,
        column: Nullable<String>,
    ) -> Result<List> {
        let column = match &column {
            Nullable::NotNull(c) => Some(c.as_str()),
            Nullable::Null => None,
        };
        let view = self
            .inner
            .view(&choice(&keys, &values)?, column)
            .map_err(to_r)?;
        let (no, nd) = (view.origins.len(), view.development.len());
        let cells: Vec<f64> = (0..no)
            .flat_map(|o| (0..nd).map(move |d| (o, d)))
            .map(|(o, d)| view.get(o, d).unwrap_or(f64::NAN))
            .collect();
        Ok(list!(
            origins = view
                .origins
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            development = view
                .development
                .iter()
                .map(|&a| a as i32)
                .collect::<Vec<_>>(),
            values = cells
        ))
    }

    /// One row per segment and measure: the key columns (a named list),
    /// `column`, `n_origins`, `first_origin` and `last_origin` (period
    /// labels), `valuation` ("YYYY-MM"), with "" where the segment has no
    /// observed value, `latest` and `cumulative`.
    fn summary(&self) -> List {
        let summary = self.inner.summary();
        let rows = &summary.rows;
        let keys = summary
            .key_names
            .iter()
            .enumerate()
            .map(|(k, name)| {
                let values = rows.iter().map(|r| r.label.parts()[k].clone()).collect();
                (name.clone(), values)
            })
            .collect();
        let text = |x: Option<String>| x.unwrap_or_default();
        list!(
            keys = key_columns(keys),
            column = rows.iter().map(|r| r.column.clone()).collect::<Vec<_>>(),
            n_origins = rows.iter().map(|r| r.n_origins as i32).collect::<Vec<_>>(),
            first_origin = rows
                .iter()
                .map(|r| text(r.first_origin.map(|p| p.to_string())))
                .collect::<Vec<_>>(),
            last_origin = rows
                .iter()
                .map(|r| text(r.last_origin.map(|p| p.to_string())))
                .collect::<Vec<_>>(),
            valuation = rows
                .iter()
                .map(|r| text(r.valuation.map(|m| m.to_string())))
                .collect::<Vec<_>>(),
            latest = rows.iter().map(|r| r.latest).collect::<Vec<_>>(),
            cumulative = summary.cumulative
        )
    }

    /// The printout: the grid of one segment and measure, otherwise the
    /// summary table, with at most `max_rows` rows and `max_cols` ages (0
    /// for no limit).
    fn to_text(&self, max_rows: f64, max_cols: f64) -> Result<String> {
        let max_rows = whole(max_rows, "max_rows")? as usize;
        let max_cols = whole(max_cols, "max_cols")? as usize;
        Ok(self.inner.to_text(max_rows, max_cols))
    }

    /// `tail` is a `ReservingTail`.
    fn chain_ladder(
        &self,
        column: &str,
        average: &str,
        sigma_interpolation: &str,
        tail: Robj,
    ) -> Result<ChainLadderFit> {
        let inner = pattern(average, sigma_interpolation, tail)?
            .fit_segments(&self.inner, column)
            .map_err(to_r)?;
        Ok(ChainLadderFit { inner })
    }

    /// `tail` is a `ReservingTail`; `tail_sigma` and `tail_std_err` are
    /// `NULL` to extrapolate them.
    fn mack(
        &self,
        column: &str,
        average: &str,
        sigma_interpolation: &str,
        tail: Robj,
        tail_sigma: Nullable<f64>,
        tail_std_err: Nullable<f64>,
    ) -> Result<MackFit> {
        let inner = Mack {
            development: development(average, sigma_interpolation)?,
            tail: tail_arg(&tail)?,
            tail_sigma: optional(tail_sigma),
            tail_std_err: optional(tail_std_err),
        }
        .fit_segments(&self.inner, column)
        .map_err(to_r)?;
        Ok(MackFit { inner })
    }

    /// The expected loss ratio method on loss `column` with exposure from
    /// the `exposure` column; the chain ladder gives the development
    /// pattern reported.
    fn expected_loss(
        &self,
        column: &str,
        exposure: &str,
        apriori: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Robj,
    ) -> Result<ExpectedLossFit> {
        let inner = ExpectedLoss {
            apriori,
            chain_ladder: pattern(average, sigma_interpolation, tail)?,
        }
        .fit_segments(&self.inner, column, exposure)
        .map_err(to_r)?;
        Ok(ExpectedLossFit { inner })
    }

    fn bornhuetter_ferguson(
        &self,
        column: &str,
        exposure: &str,
        apriori: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Robj,
    ) -> Result<ExpectedLossFit> {
        let inner = BornhuetterFerguson {
            apriori,
            chain_ladder: pattern(average, sigma_interpolation, tail)?,
        }
        .fit_segments(&self.inner, column, exposure)
        .map_err(to_r)?;
        Ok(ExpectedLossFit { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn benktander(
        &self,
        column: &str,
        exposure: &str,
        apriori: f64,
        n_iters: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Robj,
    ) -> Result<ExpectedLossFit> {
        let n_iters = usize::try_from(whole(n_iters, "n_iters")?)
            .map_err(|_| Error::Other(format!("n_iters {n_iters} is too large")))?;
        let inner = Benktander {
            apriori,
            n_iters,
            chain_ladder: pattern(average, sigma_interpolation, tail)?,
        }
        .fit_segments(&self.inner, column, exposure)
        .map_err(to_r)?;
        Ok(ExpectedLossFit { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn cape_cod(
        &self,
        column: &str,
        exposure: &str,
        trend: f64,
        decay: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Robj,
    ) -> Result<CapeCodFit> {
        let inner = CapeCod {
            trend,
            decay,
            chain_ladder: pattern(average, sigma_interpolation, tail)?,
        }
        .fit_segments(&self.inner, column, exposure)
        .map_err(to_r)?;
        Ok(CapeCodFit { inner })
    }

    fn odp_bootstrap(
        &self,
        column: &str,
        n_sims: f64,
        seed: f64,
        process: &str,
    ) -> Result<OdpBootstrapFit> {
        let inner = bootstrap(n_sims, seed, process)?
            .fit_segments(&self.inner, column)
            .map_err(to_r)?;
        Ok(OdpBootstrapFit { inner })
    }

    /// The lifetime view under Mack's bootstrap: each origin's reserve
    /// simulated to the last age with `process` ("gamma", "lognormal",
    /// "residuals", "normal" or "none"), Mack's model averaged as `average`
    /// with `sigma_interpolation`, and the residuals centred before
    /// resampling if `centre_residuals`.
    #[allow(clippy::too_many_arguments)]
    fn mack_bootstrap(
        &self,
        column: &str,
        n_sims: f64,
        seed: f64,
        process: &str,
        average: &str,
        sigma_interpolation: &str,
        centre_residuals: bool,
    ) -> Result<MackBootstrapFit> {
        let inner = mack_bootstrap(
            n_sims,
            seed,
            process,
            average,
            sigma_interpolation,
            centre_residuals,
        )?
        .fit_segments(&self.inner, column)
        .map_err(to_r)?;
        Ok(MackBootstrapFit { inner })
    }

    /// The one-year view of `method` ("chain_ladder", "expected_loss",
    /// "bornhuetter_ferguson", "benktander" or "cape_cod") by re-reserving
    /// on the ODP bootstrap. `exposure` is `NULL` for the chain ladder and
    /// names the exposure column of the other methods; each method reads
    /// only its own settings (`apriori`, `n_iters`, `trend`, `decay`).
    #[allow(clippy::too_many_arguments)]
    fn odp_one_year(
        &self,
        column: &str,
        method: &str,
        exposure: Nullable<String>,
        apriori: f64,
        n_iters: f64,
        trend: f64,
        decay: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Robj,
        n_sims: f64,
        seed: f64,
        process: &str,
    ) -> Result<OneYearFit> {
        let method = one_year_method(
            method,
            exposure,
            apriori,
            n_iters,
            trend,
            decay,
            pattern(average, sigma_interpolation, tail)?,
        )?;
        let inner = bootstrap(n_sims, seed, process)?
            .one_year_segments(&self.inner, column, &method)
            .map_err(to_r)?;
        Ok(OneYearFit {
            inner: OneYear::Odp(inner),
        })
    }

    /// The one-year view of `method`, as `odp_one_year`, under Mack's
    /// process: Mack's bootstrap with `process` ("gamma", "lognormal",
    /// "residuals", "normal" or "none"), Mack's model averaged as
    /// `mack_average` with `mack_sigma_interpolation`, and the residuals
    /// centred before resampling if `centre_residuals`.
    #[allow(clippy::too_many_arguments)]
    fn mack_one_year(
        &self,
        column: &str,
        method: &str,
        exposure: Nullable<String>,
        apriori: f64,
        n_iters: f64,
        trend: f64,
        decay: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Robj,
        n_sims: f64,
        seed: f64,
        process: &str,
        mack_average: &str,
        mack_sigma_interpolation: &str,
        centre_residuals: bool,
    ) -> Result<OneYearFit> {
        let method = one_year_method(
            method,
            exposure,
            apriori,
            n_iters,
            trend,
            decay,
            pattern(average, sigma_interpolation, tail)?,
        )?;
        let inner = mack_bootstrap(
            n_sims,
            seed,
            process,
            mack_average,
            mack_sigma_interpolation,
            centre_residuals,
        )?
        .one_year_segments(&self.inner, column, &method)
        .map_err(to_r)?;
        Ok(OneYearFit {
            inner: OneYear::Mack(inner),
        })
    }

    /// Clark's LDF method; `max_age` is `Inf` to develop to infinity.
    fn clark_ldf(&self, column: &str, curve: &str, max_age: f64) -> Result<ClarkFit> {
        let inner = ClarkLdf {
            curve: growth_curve(curve)?,
            max_age: clark_max_age(max_age),
        }
        .fit_segments(&self.inner, column)
        .map_err(to_r)?;
        Ok(ClarkFit { inner })
    }

    /// Clark's Cape Cod method, with each origin's exposure the latest
    /// value of column `exposure`.
    fn clark_cape_cod(
        &self,
        column: &str,
        exposure: &str,
        curve: &str,
        max_age: f64,
    ) -> Result<ClarkFit> {
        let inner = ClarkCapeCod {
            curve: growth_curve(curve)?,
            max_age: clark_max_age(max_age),
        }
        .fit_segments(&self.inner, column, exposure)
        .map_err(to_r)?;
        Ok(ClarkFit { inner })
    }
}

/// How development past the oldest age is estimated: a constant factor, a
/// curve fitted to the factors, Bondy's rule or R ChainLadder's log-linear
/// rule. An age that is not given is `NULL`.
#[extendr]
pub(crate) struct ReservingTail {
    inner: Tail,
}

fn tail_arg(tail: &Robj) -> Result<Tail> {
    <&ReservingTail>::try_from(tail)
        .map(|t| t.inner)
        .map_err(|_| Error::Other("tail must be a tail estimator".into()))
}

fn curve_name(curve: CurveShape) -> &'static str {
    match curve {
        CurveShape::Exponential => "exponential",
        CurveShape::InversePower => "inverse_power",
    }
}

#[extendr]
impl ReservingTail {
    fn constant(factor: f64, decay: f64, attachment_age: Nullable<f64>) -> Result<Self> {
        Ok(Self {
            inner: Tail::Constant(TailConstant {
                factor,
                decay,
                attachment_age: optional_age(attachment_age, "attachment_age")?,
            }),
        })
    }

    /// `curve` is "exponential" or "inverse_power"; the fit period runs
    /// from `fit_from` (inclusive) to `fit_to` (exclusive).
    fn curve(
        curve: &str,
        fit_from: Nullable<f64>,
        fit_to: Nullable<f64>,
        extrap_periods: f64,
        attachment_age: Nullable<f64>,
    ) -> Result<Self> {
        let curve = match curve {
            "exponential" => CurveShape::Exponential,
            "inverse_power" => CurveShape::InversePower,
            _ => {
                return Err(Error::Other(format!(
                    "curve must be \"exponential\" or \"inverse_power\", got \"{curve}\""
                )));
            }
        };
        let extrap_periods = usize::try_from(whole(extrap_periods, "extrap_periods")?)
            .map_err(|_| Error::Other("extrap_periods is too large".into()))?;
        Ok(Self {
            inner: Tail::Curve(TailCurve {
                curve,
                fit_period: (
                    optional_age(fit_from, "fit_period")?,
                    optional_age(fit_to, "fit_period")?,
                ),
                extrap_periods,
                attachment_age: optional_age(attachment_age, "attachment_age")?,
            }),
        })
    }

    fn bondy(earliest_age: Nullable<f64>, attachment_age: Nullable<f64>) -> Result<Self> {
        Ok(Self {
            inner: Tail::Bondy(TailBondy {
                earliest_age: optional_age(earliest_age, "earliest_age")?,
                attachment_age: optional_age(attachment_age, "attachment_age")?,
            }),
        })
    }

    fn log_linear() -> Self {
        Self {
            inner: Tail::LogLinear,
        }
    }

    /// "constant", "curve", "bondy" or "log_linear".
    fn kind(&self) -> &'static str {
        match self.inner {
            Tail::Constant(_) => "constant",
            Tail::Curve(_) => "curve",
            Tail::Bondy(_) => "bondy",
            Tail::LogLinear => "log_linear",
        }
    }

    /// The parameters of this kind as a named list, with `NULL` for an age
    /// not given; the curve's fit period is `fit_from` and `fit_to`.
    fn params(&self) -> List {
        let age = |a: Option<Lag>| -> Robj {
            match a {
                Some(a) => (a as i32).into(),
                None => ().into(),
            }
        };
        match self.inner {
            Tail::Constant(t) => list!(
                factor = t.factor,
                decay = t.decay,
                attachment_age = age(t.attachment_age)
            ),
            Tail::Curve(t) => list!(
                curve = curve_name(t.curve),
                fit_from = age(t.fit_period.0),
                fit_to = age(t.fit_period.1),
                extrap_periods = t.extrap_periods as f64,
                attachment_age = age(t.attachment_age)
            ),
            Tail::Bondy(t) => list!(
                earliest_age = age(t.earliest_age),
                attachment_age = age(t.attachment_age)
            ),
            Tail::LogLinear => List::new(0),
        }
    }
}

/// The one fit of a single-segment result, or an error naming what to use
/// instead: the long table `instead` (if any) or `segment()`.
fn single<'a, T>(fits: &'a SegmentFits<T>, field: &str, instead: &str) -> Result<&'a T> {
    match fits.fits.as_slice() {
        [one] => Ok(one),
        _ => {
            let table = if instead.is_empty() {
                String::new()
            } else {
                format!("{instead} or ")
            };
            Err(Error::Other(format!(
                "{field} needs a single-segment fit, and this one has {} segments; \
                 use {table}segment()",
                fits.len()
            )))
        }
    }
}

/// Per-origin values of every segment, in the order of the long rows.
fn by_origin<T>(fits: &SegmentFits<T>, f: impl Fn(&T) -> Vec<f64>) -> Vec<f64> {
    fits.fits.iter().flat_map(f).collect()
}

fn origin_labels<T: ReserveFit>(fits: &SegmentFits<T>) -> Vec<String> {
    fits.fits
        .iter()
        .flat_map(|f| f.chain_ladder().origins.iter().map(ToString::to_string))
        .collect()
}

/// The segment label (parts joined by " / ") of each per-origin value.
fn row_segments<T: ReserveFit>(fits: &SegmentFits<T>) -> Vec<String> {
    fits.iter()
        .flat_map(|(l, f)| std::iter::repeat_n(l.to_string(), f.chain_ladder().origins.len()))
        .collect()
}

fn ages<T: ReserveFit>(fits: &SegmentFits<T>) -> Vec<i32> {
    let dev = &fits.fits[0].chain_ladder().development.development;
    dev.iter().map(|&a| a as i32).collect()
}

/// The keys choosing one segment as `(key, value)` pairs.
fn choice<'a>(keys: &'a [String], values: &'a [String]) -> Result<Vec<(&'a str, &'a str)>> {
    if keys.len() != values.len() {
        return Err(Error::Other("one value is needed per key".into()));
    }
    Ok(keys
        .iter()
        .zip(values)
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect())
}

fn pick<T: Clone>(
    fits: &SegmentFits<T>,
    keys: Vec<String>,
    values: Vec<String>,
) -> Result<SegmentFits<T>> {
    fits.segment(&choice(&keys, &values)?).map_err(to_r)
}

/// A long result table: the key columns (a named list), `origin` (labels)
/// or `development` (ages), `NULL` where the table has none, and the value
/// columns with their names.
fn fit_table(table: FitTable) -> List {
    let origin: Robj = match table.origin {
        Some(o) => o.iter().map(ToString::to_string).collect::<Vec<_>>().into(),
        None => ().into(),
    };
    let development: Robj = match table.age {
        Some(a) => a.iter().map(|&a| a as i32).collect::<Vec<_>>().into(),
        None => ().into(),
    };
    let names: Vec<String> = table.values.iter().map(|(n, _)| n.clone()).collect();
    let values = List::from_values(table.values.into_iter().map(|(_, v)| v));
    list!(
        keys = key_columns(table.keys),
        origin = origin,
        development = development,
        names = names,
        values = values
    )
}

/// A chain ladder fitted to every segment of a triangle column. Per-origin
/// vectors run over the origins of each segment in turn.
#[extendr]
pub(crate) struct ChainLadderFit {
    inner: SegmentFits<ChainLadderInner>,
}

#[extendr]
impl ChainLadderFit {
    fn n_segments(&self) -> i32 {
        self.inner.len() as i32
    }

    fn keys(&self) -> Vec<String> {
        self.inner.key_names.clone()
    }

    fn row_segments(&self) -> Vec<String> {
        row_segments(&self.inner)
    }

    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner)
    }

    fn development(&self) -> Vec<i32> {
        ages(&self.inner)
    }

    /// The selected factors within the triangle: the estimated ones,
    /// replaced by the tail's from its attachment age.
    fn ldf(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "ldf", "development_frame()")?;
        Ok(f.ldf().to_vec())
    }

    fn sigma(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "sigma", "development_frame()")?;
        Ok(f.development.sigma.clone())
    }

    fn std_err(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "std_err", "development_frame()")?;
        Ok(f.development.std_err.clone())
    }

    fn alpha(&self) -> f64 {
        self.inner.fits[0].development.alpha
    }

    /// The factors as estimated, before the tail replaced any.
    fn estimated_ldf(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "estimated_ldf", "")?;
        Ok(f.development.ldf.clone())
    }

    /// Age from which `ldf` holds the tail's factors; the oldest age when
    /// the tail replaced none.
    fn tail_attachment_age(&self) -> Result<i32> {
        let f = single(&self.inner, "tail_attachment_age", "")?;
        f.development
            .development
            .get(f.tail.attachment)
            .map(|&a| a as i32)
            .ok_or_else(|| Error::Other("the fit has no development ages".into()))
    }

    fn tail(&self) -> Result<f64> {
        Ok(single(&self.inner, "tail", "totals_frame()")?.tail.factor)
    }

    /// Factors past the oldest age, which multiply to the tail.
    fn tail_ldf(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "tail_ldf", "")?;
        Ok(f.tail.ldf[f.development.ldf.len()..].to_vec())
    }

    fn tail_sigma(&self) -> Result<f64> {
        Ok(single(&self.inner, "tail_sigma", "totals_frame()")?
            .tail
            .sigma)
    }

    fn tail_std_err(&self) -> Result<f64> {
        Ok(single(&self.inner, "tail_std_err", "totals_frame()")?
            .tail
            .std_err)
    }

    fn cdf(&self) -> Result<Vec<f64>> {
        Ok(single(&self.inner, "cdf", "development_frame()")?
            .cdf
            .clone())
    }

    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.latest.clone())
    }

    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ChainLadderInner::reserves)
    }

    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    /// The fit of the one segment whose `keys` have `values`.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys, values)?,
        })
    }
}

/// A Mack chain ladder fitted to every segment of a triangle column.
#[extendr]
pub(crate) struct MackFit {
    inner: SegmentFits<MackInner>,
}

#[extendr]
impl MackFit {
    /// The underlying chain-ladder projection.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.map(|m| m.chain_ladder.clone()),
        }
    }

    fn process_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.process_risk.clone())
    }

    fn parameter_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.parameter_risk.clone())
    }

    fn standard_error(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.standard_error.clone())
    }

    fn total_process_risk(&self) -> Result<f64> {
        Ok(single(&self.inner, "total_process_risk", "totals_frame()")?.total_process_risk)
    }

    fn total_parameter_risk(&self) -> Result<f64> {
        Ok(single(&self.inner, "total_parameter_risk", "totals_frame()")?.total_parameter_risk)
    }

    fn total_standard_error(&self) -> Result<f64> {
        Ok(single(&self.inner, "total_standard_error", "totals_frame()")?.total_standard_error)
    }

    fn total_cv(&self) -> Result<f64> {
        Ok(single(&self.inner, "total_cv", "totals_frame()")?.total_cv())
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys, values)?,
        })
    }

    /// Merz and Wüthrich's one-year view of a single-segment fit.
    fn claims_development_result(&self) -> Result<ClaimsDevelopmentResult> {
        let fit = single(&self.inner, "claims_development_result", "")?;
        Ok(ClaimsDevelopmentResult {
            inner: fit.claims_development_result().map_err(to_r)?,
        })
    }
}

/// Merz and Wüthrich's (2008) one-year view of a Mack fit: standard errors
/// of the claims development result per origin and in total, in the next
/// calendar year and in each later one. `by_calendar_year` is row-major
/// over calendar year x origin.
#[extendr]
pub(crate) struct ClaimsDevelopmentResult {
    inner: ClaimsDevelopmentInner,
}

#[extendr]
impl ClaimsDevelopmentResult {
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    fn one_year_standard_error(&self) -> Vec<f64> {
        self.inner.one_year_standard_error.clone()
    }

    fn total_one_year_standard_error(&self) -> f64 {
        self.inner.total_one_year_standard_error
    }

    fn by_calendar_year(&self) -> Vec<f64> {
        self.inner.by_calendar_year.concat()
    }

    fn total_by_calendar_year(&self) -> Vec<f64> {
        self.inner.total_by_calendar_year.clone()
    }

    fn run_off_standard_error(&self) -> Vec<f64> {
        self.inner.run_off_standard_error()
    }

    fn total_run_off_standard_error(&self) -> f64 {
        self.inner.total_run_off_standard_error()
    }
}

/// An expected-loss method (expected loss, Bornhuetter-Ferguson or
/// Benktander) fitted to every segment of a triangle column, each with its
/// exposure from another column of the same triangle.
#[extendr]
pub(crate) struct ExpectedLossFit {
    inner: SegmentFits<ExpectedLossInner>,
}

#[extendr]
impl ExpectedLossFit {
    /// The chain ladder that gives the development pattern.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.map(|f| f.chain_ladder.clone()),
        }
    }

    fn exposure(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.exposure.clone())
    }

    fn apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.apriori.clone())
    }

    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ExpectedLossInner::reserves)
    }

    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys, values)?,
        })
    }
}

/// A Cape Cod fitted to every segment of a triangle column: the
/// Bornhuetter-Ferguson fit on the detrended apriori, and the trended
/// apriori it came from.
#[extendr]
pub(crate) struct CapeCodFit {
    inner: SegmentFits<CapeCodInner>,
}

#[extendr]
impl CapeCodFit {
    /// The Bornhuetter-Ferguson fit on the detrended apriori.
    fn expected_loss(&self) -> ExpectedLossFit {
        ExpectedLossFit {
            inner: self.inner.map(|f| f.expected_loss.clone()),
        }
    }

    fn trended_apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.trended_apriori.clone())
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys, values)?,
        })
    }
}

/// An ODP bootstrap of every segment of a triangle column. `fitted` and
/// `residuals` are row-major over origin x development, NaN where not
/// observed.
#[extendr]
pub(crate) struct OdpBootstrapFit {
    inner: OdpBootstrapFits,
}

#[extendr]
impl OdpBootstrapFit {
    /// The chain ladder the bootstrap is centred on.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    fn fitted(&self) -> Result<Vec<f64>> {
        let s = single(&self.inner.segments, "fitted", "")?;
        Ok(s.fitted.clone())
    }

    fn residuals(&self) -> Result<Vec<f64>> {
        let s = single(&self.inner.segments, "residuals", "")?;
        Ok(s.residuals.clone())
    }

    fn scale(&self) -> Result<f64> {
        Ok(single(&self.inner.segments, "scale", "totals_frame()")?.scale)
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: self.inner.segment(&choice(&keys, &values)?).map_err(to_r)?,
        })
    }
}

/// Mack's bootstrap, the lifetime view, of every segment of a triangle
/// column. `residuals` is row-major over origin x development, NaN where
/// there is no residual.
#[extendr]
pub(crate) struct MackBootstrapFit {
    inner: MackBootstrapFits,
}

#[extendr]
impl MackBootstrapFit {
    /// The chain ladder of Mack's model.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.segments.map(|s| s.mack.chain_ladder.clone()),
        }
    }

    /// Mack's model on the observed triangle, without a tail.
    fn mack(&self) -> MackFit {
        MackFit {
            inner: self.inner.segments.map(|s| s.mack.clone()),
        }
    }

    fn residuals(&self) -> Result<Vec<f64>> {
        let s = single(&self.inner.segments, "residuals", "")?;
        Ok(s.residuals.clone())
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: self.inner.segment(&choice(&keys, &values)?).map_err(to_r)?,
        })
    }
}

/// The simulated one-year view of every segment of a triangle column:
/// each segment's bootstrap and opening ultimate, and the joint claims
/// development result, from the ODP bootstrap or Mack's.
#[extendr]
pub(crate) struct OneYearFit {
    inner: OneYear,
}

/// A one-year view from either bootstrap.
enum OneYear {
    Odp(OneYearFits),
    Mack(OneYearFits<MackBootstrapSegment>),
}

/// `$body` on whichever one-year view `$fit` holds, bound to `$f`.
macro_rules! each_one_year {
    ($fit:expr, $f:ident => $body:expr) => {
        match $fit {
            OneYear::Odp($f) => $body,
            OneYear::Mack($f) => $body,
        }
    };
}

#[extendr]
impl OneYearFit {
    /// "odp" or "mack".
    fn model(&self) -> &'static str {
        match self.inner {
            OneYear::Odp(_) => "odp",
            OneYear::Mack(_) => "mack",
        }
    }

    /// The bootstrap's chain ladder: volume-weighted for the ODP, Mack's
    /// averaging for Mack's.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: each_one_year!(&self.inner, f => f
                .segments
                .map(|s| s.bootstrap.chain_ladder().clone())),
        }
    }

    /// Mack's model behind `mack_one_year()`.
    fn mack(&self) -> Result<MackFit> {
        match &self.inner {
            OneYear::Mack(f) => Ok(MackFit {
                inner: f.segments.map(|s| s.bootstrap.mack.clone()),
            }),
            OneYear::Odp(_) => Err(Error::Other(
                "mack is Mack's bootstrap's model; this one-year view is the ODP's".into(),
            )),
        }
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        each_one_year!(&self.inner, f => by_origin(&f.segments, |s| s.opening_ultimate.clone()))
    }

    fn opening_reserve(&self) -> Vec<f64> {
        each_one_year!(&self.inner, f => by_origin(&f.segments, |s| s.opening_reserve.clone()))
    }

    fn scale(&self) -> Result<f64> {
        match &self.inner {
            OneYear::Odp(f) => Ok(single(&f.segments, "scale", "totals_frame()")?
                .bootstrap
                .scale),
            OneYear::Mack(_) => Err(Error::Other(
                "scale is the ODP bootstrap's; this one-year view is Mack's".into(),
            )),
        }
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: each_one_year!(&self.inner, f => f.cdr.clone()),
        }
    }

    fn long_table(&self) -> List {
        fit_table(each_one_year!(&self.inner, f => f.to_long()))
    }

    fn totals_table(&self) -> List {
        fit_table(each_one_year!(&self.inner, f => f.totals()))
    }

    fn development_table(&self) -> List {
        fit_table(each_one_year!(&self.inner, f => f.segments.development_table()))
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let keys = choice(&keys, &values)?;
        let inner = match &self.inner {
            OneYear::Odp(f) => OneYear::Odp(f.segment(&keys).map_err(to_r)?),
            OneYear::Mack(f) => OneYear::Mack(f.segment(&keys).map_err(to_r)?),
        };
        Ok(Self { inner })
    }
}

/// A Clark LDF or Cape Cod fit of every segment of a triangle column.
/// Per-origin vectors run over the origins of each segment in turn.
#[extendr]
pub(crate) struct ClarkFit {
    inner: SegmentFits<ClarkInner>,
}

impl ClarkFit {
    /// The fit of a single-segment result, for `field`.
    fn one(&self, field: &str) -> Result<&ClarkInner> {
        single(&self.inner, field, "totals_frame()")
    }
}

#[extendr]
impl ClarkFit {
    /// The volume-weighted chain ladder of the same column.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.map(|f| f.chain_ladder.clone()),
        }
    }

    /// "ldf" or "cape_cod".
    fn method(&self) -> &'static str {
        if self.inner.fits[0].elr.is_some() {
            "cape_cod"
        } else {
            "ldf"
        }
    }

    fn curve(&self) -> &'static str {
        growth_curve_name(self.inner.fits[0].curve)
    }

    /// `Inf` when development runs to infinity.
    fn max_age(&self) -> f64 {
        self.inner.fits[0].max_age.unwrap_or(f64::INFINITY)
    }

    fn omega(&self) -> Result<f64> {
        Ok(self.one("omega")?.omega)
    }

    fn theta(&self) -> Result<f64> {
        Ok(self.one("theta")?.theta)
    }

    /// The expected loss ratio; NaN for the LDF method.
    fn elr(&self) -> Result<f64> {
        if self.inner.fits[0].elr.is_none() {
            return Ok(f64::NAN);
        }
        Ok(self.one("elr")?.elr.unwrap_or(f64::NAN))
    }

    /// Length of the origin period in months.
    fn origin_width(&self) -> f64 {
        self.inner.fits[0].origin_width
    }

    /// Number of observed incremental values fitted.
    fn n_observations(&self) -> Result<i32> {
        let f = single(&self.inner, "n_observations", "segment()")?;
        Ok(f.n_observations as i32)
    }

    fn scale(&self) -> Result<f64> {
        Ok(self.one("scale")?.scale)
    }

    /// The parameter covariance, row-major.
    fn covariance(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "covariance", "segment()")?;
        Ok(f.covariance.concat())
    }

    /// Exposure per origin; empty for the LDF method.
    fn exposure(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.exposure.clone().unwrap_or_default())
    }

    fn expected_ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.expected_ultimate.clone())
    }

    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ClarkInner::reserves)
    }

    fn process_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.process_risk.clone())
    }

    fn parameter_risk(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.parameter_risk.clone())
    }

    fn standard_error(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.standard_error.clone())
    }

    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    fn total_process_risk(&self) -> Result<f64> {
        Ok(self.one("total_process_risk")?.total_process_risk)
    }

    fn total_parameter_risk(&self) -> Result<f64> {
        Ok(self.one("total_parameter_risk")?.total_parameter_risk)
    }

    fn total_standard_error(&self) -> Result<f64> {
        Ok(self.one("total_standard_error")?.total_standard_error)
    }

    /// Share of the expected ultimate developed by each development age
    /// (months; `Inf` gives 1).
    fn growth(&self, age: &[f64]) -> Result<Vec<f64>> {
        let f = single(&self.inner, "growth", "segment()")?;
        Ok(age.iter().map(|&a| f.growth(a)).collect())
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys, values)?,
        })
    }
}

extendr_module! {
    mod reserving;
    impl Triangle;
    impl ReservingTail;
    impl ChainLadderFit;
    impl MackFit;
    impl ExpectedLossFit;
    impl CapeCodFit;
    impl ClaimsDevelopmentResult;
    impl OdpBootstrapFit;
    impl MackBootstrapFit;
    impl OneYearFit;
    impl ClarkFit;
}
