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
    MackProcess, Month, OdpBootstrap, OdpBootstrapFits, OneYearFits, OneYearMethod, OneYearPattern,
    PatternKind, PatternWeight, ProcessDistribution, ReserveFit, SegmentDependence, SegmentFits,
    SelectedPattern, SigmaInterpolation, Tail, TailBondy, TailConstant, TailCurve,
    Triangle as TriangleInner,
};

use prospicio_bayes::nuts::{ParamSummary, Sampler as BayesSampler};
use prospicio_reserving::bayes::{
    BayesOdpFits as BayesOdpFitsInner, BayesOdpSegment, BayesOneYear,
    BayesOneYearFit as BayesOneYearInner, BayesOneYearFits as BayesOneYearFitsInner, Compartmental,
    CompartmentalFit as CompartmentalInner, CompartmentalLikelihood, CompartmentalPriors,
    CompartmentalStart, Fitted, HalfPrior, MeyersFits as MeyersFitsInner, MeyersSegment,
    OneYearRefit, Outstanding, PaidResponse, PositivePrior, RateOrder, RealPrior,
    Scale as CompartmentalScale,
};
use prospicio_reserving::{
    BayesBf, BayesBfFit as BayesBfInner, BayesOdp, BayesOdpFit as BayesOdpInner, Diagnostics,
    Meyers, MeyersFit as MeyersInner, MeyersModel,
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

fn development(
    average: &str,
    sigma_interpolation: &str,
    pattern: Option<SelectedPattern>,
) -> Result<Development> {
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
        pattern,
    })
}

/// A selected pattern as `ldf_pattern()` in R encodes it: `NULL`, or a
/// list of `kind` ("ldf", "cdf" or "percent_developed"), `ages` and
/// `values`, and `weights` with `weight_ages` (`NULL` for one weight on
/// every selected factor).
fn selected_pattern(pattern: &Robj) -> Result<Option<SelectedPattern>> {
    if pattern.is_null() {
        return Ok(None);
    }
    let list = List::try_from(pattern)
        .map_err(|_| Error::Other("the selected pattern must be a list".into()))?;
    let mut kind = None;
    let (mut ages, mut values) = (Vec::new(), Vec::new());
    let (mut weight_ages, mut weights) = (None, vec![1.0]);
    for (name, value) in list.iter() {
        match name {
            "kind" => kind = value.as_str().map(str::to_owned),
            "ages" => ages = value.as_real_vector().unwrap_or_default(),
            "values" => values = value.as_real_vector().unwrap_or_default(),
            "weight_ages" => weight_ages = value.as_real_vector(),
            "weights" => weights = value.as_real_vector().unwrap_or_default(),
            _ => {}
        }
    }
    let kind = match kind.as_deref() {
        Some("ldf") => PatternKind::AgeToAge,
        Some("cdf") => PatternKind::AgeToUltimate,
        Some("percent_developed") => PatternKind::PercentDeveloped,
        _ => {
            return Err(Error::Other(
                "ldf_kind must be \"ldf\", \"cdf\" or \"percent_developed\"".into(),
            ));
        }
    };
    let pairs = |ages: &[f64], values: &[f64], what: &str| -> Result<Vec<(Lag, f64)>> {
        if ages.len() != values.len() {
            return Err(Error::Other(format!(
                "{what}: {} ages but {} values",
                ages.len(),
                values.len()
            )));
        }
        ages.iter()
            .zip(values)
            .map(|(&a, &v)| Ok((age(a)?, v)))
            .collect()
    };
    let weight = match weight_ages {
        None => match weights.as_slice() {
            [w] => PatternWeight::All(*w),
            _ => return Err(Error::Other("ldf_weight must be a single number".into())),
        },
        Some(weight_ages) => PatternWeight::ByAge(pairs(&weight_ages, &weights, "ldf_weight")?),
    };
    Ok(Some(
        SelectedPattern::new(kind, pairs(&ages, &values, "ldf")?).with_weight(weight),
    ))
}

/// The chain ladder that gives a method its development pattern, with the
/// selected pattern `ldf` (as `selected_pattern` reads it).
fn pattern(
    average: &str,
    sigma_interpolation: &str,
    tail: Robj,
    ldf: &Robj,
) -> Result<ChainLadder> {
    Ok(ChainLadder {
        development: development(average, sigma_interpolation, selected_pattern(ldf)?)?,
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

/// The dependence between segments named `dependence` ("independent",
/// "synchronized" or "rank_correlation"), with the row-major Spearman
/// matrix that "rank_correlation" needs and the others refuse.
fn segment_dependence(dependence: &str, spearman: Nullable<Vec<f64>>) -> Result<SegmentDependence> {
    match (dependence, spearman) {
        ("independent", Nullable::Null) => Ok(SegmentDependence::Independent),
        ("synchronized", Nullable::Null) => Ok(SegmentDependence::Synchronized),
        ("rank_correlation", Nullable::NotNull(spearman)) => {
            Ok(SegmentDependence::RankCorrelation { spearman })
        }
        ("rank_correlation", Nullable::Null) => Err(Error::Other(
            "dependence = \"rank_correlation\" needs a spearman matrix".into(),
        )),
        ("independent" | "synchronized", Nullable::NotNull(_)) => Err(Error::Other(
            "spearman is taken only with dependence = \"rank_correlation\"".into(),
        )),
        _ => Err(Error::Other(format!(
            "dependence must be \"independent\", \"synchronized\" or \"rank_correlation\", \
             got \"{dependence}\""
        ))),
    }
}

/// Mack's bootstrap of `mack_bootstrap()` and `mack_one_year()`.
#[allow(clippy::too_many_arguments)]
fn mack_bootstrap(
    n_sims: f64,
    seed: f64,
    process: &str,
    average: &str,
    sigma_interpolation: &str,
    centre_residuals: bool,
    dependence: SegmentDependence,
    pattern: Option<SelectedPattern>,
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
        development: development(average, sigma_interpolation, pattern)?,
        centre_residuals,
        dependence,
        ..Default::default()
    })
}

/// The ODP bootstrap of `odp_bootstrap()` and `odp_one_year()`.
fn bootstrap(
    n_sims: f64,
    seed: f64,
    process: &str,
    dependence: SegmentDependence,
) -> Result<OdpBootstrap> {
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
        dependence,
        ..Default::default()
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

    /// `tail` is a `ReservingTail`; `ldf` a selected pattern as
    /// `ldf_pattern()` in R encodes it, or `NULL`.
    fn chain_ladder(
        &self,
        column: &str,
        average: &str,
        sigma_interpolation: &str,
        tail: Robj,
        ldf: Robj,
    ) -> Result<ChainLadderFit> {
        let inner = pattern(average, sigma_interpolation, tail, &ldf)?
            .fit_segments(&self.inner, column)
            .map_err(to_r)?;
        Ok(ChainLadderFit { inner })
    }

    /// `tail` is a `ReservingTail`; `tail_sigma` and `tail_std_err` are
    /// `NULL` to extrapolate them; `ldf` as in `chain_ladder`.
    #[allow(clippy::too_many_arguments)]
    fn mack(
        &self,
        column: &str,
        average: &str,
        sigma_interpolation: &str,
        tail: Robj,
        tail_sigma: Nullable<f64>,
        tail_std_err: Nullable<f64>,
        ldf: Robj,
    ) -> Result<MackFit> {
        let inner = Mack {
            development: development(average, sigma_interpolation, selected_pattern(&ldf)?)?,
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
    #[allow(clippy::too_many_arguments)]
    fn expected_loss(
        &self,
        column: &str,
        exposure: &str,
        apriori: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Robj,
        ldf: Robj,
    ) -> Result<ExpectedLossFit> {
        let inner = ExpectedLoss {
            apriori,
            chain_ladder: pattern(average, sigma_interpolation, tail, &ldf)?,
        }
        .fit_segments(&self.inner, column, exposure)
        .map_err(to_r)?;
        Ok(ExpectedLossFit { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn bornhuetter_ferguson(
        &self,
        column: &str,
        exposure: &str,
        apriori: f64,
        average: &str,
        sigma_interpolation: &str,
        tail: Robj,
        ldf: Robj,
    ) -> Result<ExpectedLossFit> {
        let inner = BornhuetterFerguson {
            apriori,
            chain_ladder: pattern(average, sigma_interpolation, tail, &ldf)?,
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
        ldf: Robj,
    ) -> Result<ExpectedLossFit> {
        let n_iters = usize::try_from(whole(n_iters, "n_iters")?)
            .map_err(|_| Error::Other(format!("n_iters {n_iters} is too large")))?;
        let inner = Benktander {
            apriori,
            n_iters,
            chain_ladder: pattern(average, sigma_interpolation, tail, &ldf)?,
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
        ldf: Robj,
    ) -> Result<CapeCodFit> {
        let inner = CapeCod {
            trend,
            decay,
            chain_ladder: pattern(average, sigma_interpolation, tail, &ldf)?,
        }
        .fit_segments(&self.inner, column, exposure)
        .map_err(to_r)?;
        Ok(CapeCodFit { inner })
    }

    /// The ODP bootstrap of every segment, the segments depending on each
    /// other as `dependence` ("independent", "synchronized" or
    /// "rank_correlation" with the row-major `spearman` matrix) says.
    /// `tail` is a `ReservingTail`, developed past the oldest age in every
    /// simulation; `tail_std_err` is `NULL` to keep a constant tail fixed.
    #[allow(clippy::too_many_arguments)]
    fn odp_bootstrap(
        &self,
        column: &str,
        n_sims: f64,
        seed: f64,
        process: &str,
        tail: Robj,
        tail_std_err: Nullable<f64>,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
        ldf: Robj,
    ) -> Result<OdpBootstrapFit> {
        let dependence = segment_dependence(dependence, spearman)?;
        let boot = OdpBootstrap {
            tail: tail_arg(&tail)?,
            tail_std_err: optional(tail_std_err),
            pattern: selected_pattern(&ldf)?,
            ..bootstrap(n_sims, seed, process, dependence)?
        };
        let inner = boot.fit_segments(&self.inner, column).map_err(to_r)?;
        Ok(OdpBootstrapFit { inner })
    }

    /// The Bayesian ODP chain ladder of a single-segment triangle column:
    /// normal priors with sds `intercept_sd`, `origin_sd` and
    /// `development_sd` on the log scale, `process` error ("gamma" or
    /// "none"), NUTS with `chains`, `tune` and `draws` per chain,
    /// `target_accept` and `max_depth`, and `n_sims` predictive
    /// simulations, all replayed from `seed`.
    #[allow(clippy::too_many_arguments)]
    fn bayes_odp(
        &self,
        column: &str,
        intercept_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        process: &str,
        chains: f64,
        tune: f64,
        draws: f64,
        n_sims: f64,
        seed: f64,
        target_accept: f64,
        max_depth: f64,
    ) -> Result<BayesOdpFit> {
        let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
        let model = BayesOdp {
            intercept_sd,
            origin_sd,
            development_sd,
            process: boot.process,
            sampler: BayesSampler {
                chains: whole(chains, "chains")? as usize,
                tune: whole(tune, "tune")? as usize,
                draws: whole(draws, "draws")? as usize,
                seed: 0,
                target_accept,
                max_depth: whole(max_depth, "max_depth")?,
            },
            n_sims: boot.n_sims,
            seed: boot.seed,
            ..BayesOdp::default()
        };
        let inner = model.fit(&self.inner, column).map_err(to_r)?;
        Ok(BayesOdpFit { inner })
    }

    /// The Bayesian ODP chain ladder of every segment of a triangle column,
    /// with the settings of `bayes_odp`, each segment's own posterior, and
    /// the segments depending on each other as `dependence`
    /// ("independent", "rank_correlation" with the row-major `spearman`
    /// matrix, or "synchronized", an error) says.
    #[allow(clippy::too_many_arguments)]
    fn bayes_odp_segments(
        &self,
        column: &str,
        intercept_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        process: &str,
        chains: f64,
        tune: f64,
        draws: f64,
        n_sims: f64,
        seed: f64,
        target_accept: f64,
        max_depth: f64,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
    ) -> Result<BayesOdpFits> {
        let dependence = segment_dependence(dependence, spearman)?;
        let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
        let model = BayesOdp {
            intercept_sd,
            origin_sd,
            development_sd,
            process: boot.process,
            sampler: BayesSampler {
                chains: whole(chains, "chains")? as usize,
                tune: whole(tune, "tune")? as usize,
                draws: whole(draws, "draws")? as usize,
                seed: 0,
                target_accept,
                max_depth: whole(max_depth, "max_depth")?,
            },
            n_sims: boot.n_sims,
            seed: boot.seed,
            dependence,
        };
        let inner = model.fit_segments(&self.inner, column).map_err(to_r)?;
        Ok(BayesOdpFits { inner })
    }

    /// The one-year view of `bayes_odp` on a single-segment triangle
    /// column by PSIS reweighting: `priors` are `intercept_sd`, `origin_sd`
    /// and `development_sd`, `run` is `chains`, `tune`, `draws`, `n_sims`,
    /// `seed`, `target_accept` and `max_depth`; a simulation whose Pareto
    /// k is above `k_threshold` is handled as `refit`
    /// ("moment_match_then_refit", "refit_flagged", "reweight_only" or
    /// "refit_all") says, refitting at most `max_refits` (`NULL` for no
    /// limit).
    #[allow(clippy::too_many_arguments)]
    fn bayes_odp_one_year(
        &self,
        column: &str,
        priors: Vec<f64>,
        process: &str,
        run: Vec<f64>,
        k_threshold: f64,
        refit: &str,
        max_refits: Nullable<f64>,
    ) -> Result<BayesOneYearFit> {
        let model = bayes_odp_model(&priors, process, &run, SegmentDependence::Independent)?;
        let settings = one_year_settings(k_threshold, refit, max_refits)?;
        let inner = model
            .one_year(&self.inner, column, &settings)
            .map_err(to_r)?;
        Ok(BayesOneYearFit {
            inner: BayesOneYearOf::Odp(inner),
        })
    }

    /// The one-year view of `bayes_odp_segments` on every segment of a
    /// triangle column, with the settings of `bayes_odp_one_year` and the
    /// segments depending on each other as `dependence` ("independent",
    /// "rank_correlation" with the row-major `spearman` matrix, or
    /// "synchronized", an error) says.
    #[allow(clippy::too_many_arguments)]
    fn bayes_odp_one_year_segments(
        &self,
        column: &str,
        priors: Vec<f64>,
        process: &str,
        run: Vec<f64>,
        k_threshold: f64,
        refit: &str,
        max_refits: Nullable<f64>,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
    ) -> Result<BayesOneYearFits> {
        let dependence = segment_dependence(dependence, spearman)?;
        let model = bayes_odp_model(&priors, process, &run, dependence)?;
        let settings = one_year_settings(k_threshold, refit, max_refits)?;
        let inner = model
            .one_year_segments(&self.inner, column, &settings)
            .map_err(to_r)?;
        Ok(BayesOneYearFits {
            inner: BayesOneYearFitsOf::Odp(inner),
        })
    }

    /// Meyers' CRC, CSR or CAY model (`model`, "crc", "csr" or "cay") of a
    /// single-segment triangle column with the premium of each origin from
    /// the `premium` column: normal priors on `logelr` (mean `logelr_mean`,
    /// sd `logelr_sd`), the origin and development effects and `gamma`, a
    /// beta prior with both shapes `rho_shape` on `(1 - rho) / 2`, NUTS with
    /// `chains`, `tune` and `draws` per chain, `target_accept` and
    /// `max_depth`, and `n_sims` predictive simulations, all replayed from
    /// `seed`.
    #[allow(clippy::too_many_arguments)]
    fn meyers(
        &self,
        column: &str,
        premium: &str,
        model: &str,
        logelr_mean: f64,
        logelr_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        gamma_sd: f64,
        rho_shape: f64,
        chains: f64,
        tune: f64,
        draws: f64,
        n_sims: f64,
        seed: f64,
        target_accept: f64,
        max_depth: f64,
    ) -> Result<MeyersFit> {
        let meyers = meyers_settings(
            model,
            [
                logelr_mean,
                logelr_sd,
                origin_sd,
                development_sd,
                gamma_sd,
                rho_shape,
            ],
            [chains, tune, draws, n_sims, seed, target_accept, max_depth],
            SegmentDependence::Independent,
        )?;
        let inner = meyers.fit(&self.inner, column, premium).map_err(to_r)?;
        Ok(MeyersFit { inner })
    }

    /// Meyers' model of every segment of a triangle column, with the
    /// settings of `meyers`, each segment's own posterior, and the segments
    /// depending on each other as `dependence` ("independent",
    /// "rank_correlation" with the row-major `spearman` matrix, or
    /// "synchronized", an error) says.
    #[allow(clippy::too_many_arguments)]
    fn meyers_segments(
        &self,
        column: &str,
        premium: &str,
        model: &str,
        logelr_mean: f64,
        logelr_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        gamma_sd: f64,
        rho_shape: f64,
        chains: f64,
        tune: f64,
        draws: f64,
        n_sims: f64,
        seed: f64,
        target_accept: f64,
        max_depth: f64,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
    ) -> Result<MeyersFits> {
        let dependence = segment_dependence(dependence, spearman)?;
        let meyers = meyers_settings(
            model,
            [
                logelr_mean,
                logelr_sd,
                origin_sd,
                development_sd,
                gamma_sd,
                rho_shape,
            ],
            [chains, tune, draws, n_sims, seed, target_accept, max_depth],
            dependence,
        )?;
        let inner = meyers
            .fit_segments(&self.inner, column, premium)
            .map_err(to_r)?;
        Ok(MeyersFits { inner })
    }

    /// The one-year view of `meyers` on a single-segment triangle column
    /// by PSIS reweighting: `priors` are `logelr_mean`, `logelr_sd`,
    /// `origin_sd`, `development_sd`, `gamma_sd` and `rho_shape`, `run` is
    /// `chains`, `tune`, `draws`, `n_sims`, `seed`, `target_accept` and
    /// `max_depth`; a simulation whose Pareto k is above `k_threshold` is
    /// handled as `refit` ("moment_match_then_refit", "refit_flagged",
    /// "reweight_only" or "refit_all") says.
    #[allow(clippy::too_many_arguments)]
    fn meyers_one_year(
        &self,
        column: &str,
        premium: &str,
        model: &str,
        priors: Vec<f64>,
        run: Vec<f64>,
        k_threshold: f64,
        refit: &str,
        max_refits: Nullable<f64>,
    ) -> Result<BayesOneYearFit> {
        let meyers = meyers_settings(
            model,
            fixed::<6>(&priors, "priors")?,
            fixed::<7>(&run, "run")?,
            SegmentDependence::Independent,
        )?;
        let settings = one_year_settings(k_threshold, refit, max_refits)?;
        let inner = meyers
            .one_year(&self.inner, column, premium, &settings)
            .map_err(to_r)?;
        Ok(BayesOneYearFit {
            inner: BayesOneYearOf::Meyers(Box::new(inner)),
        })
    }

    /// The one-year view of `meyers_segments` on every segment of a
    /// triangle column, with the settings of `meyers_one_year` and the
    /// segments depending on each other as `dependence` ("independent",
    /// "rank_correlation" with the row-major `spearman` matrix, or
    /// "synchronized", an error) says.
    #[allow(clippy::too_many_arguments)]
    fn meyers_one_year_segments(
        &self,
        column: &str,
        premium: &str,
        model: &str,
        priors: Vec<f64>,
        run: Vec<f64>,
        k_threshold: f64,
        refit: &str,
        max_refits: Nullable<f64>,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
    ) -> Result<BayesOneYearFits> {
        let dependence = segment_dependence(dependence, spearman)?;
        let meyers = meyers_settings(
            model,
            fixed::<6>(&priors, "priors")?,
            fixed::<7>(&run, "run")?,
            dependence,
        )?;
        let settings = one_year_settings(k_threshold, refit, max_refits)?;
        let inner = meyers
            .one_year_segments(&self.inner, column, premium, &settings)
            .map_err(to_r)?;
        Ok(BayesOneYearFits {
            inner: BayesOneYearFitsOf::Meyers(inner),
        })
    }

    /// Gesmann and Morris' compartmental model of a single-segment
    /// triangle: paid from the `paid` column, outstanding from
    /// the `outstanding` column ("column") or as the `outstanding` column
    /// of incurred less paid ("incurred"), each origin's premium from the
    /// `premium` column. `priors` and `families` are encoded as
    /// `compartmental_priors_values()` in R encodes them; `settings` are the
    /// likelihood, scale, paid response, fitted responses, rate order and
    /// start; `run` is `chains`, `tune`, `draws`, `n_sims`, `seed`,
    /// `target_accept` and `max_depth`.
    #[allow(clippy::too_many_arguments)]
    fn compartmental(
        &self,
        paid: &str,
        outstanding: &str,
        outstanding_from: &str,
        premium: &str,
        priors: Vec<f64>,
        families: Vec<String>,
        settings: Vec<String>,
        run: Vec<f64>,
    ) -> Result<CompartmentalFit> {
        let model = compartmental_settings(&priors, &families, &settings, &run)?;
        let os = compartmental_outstanding(outstanding, outstanding_from)?;
        let inner = model.fit(&self.inner, paid, os, premium).map_err(to_r)?;
        Ok(CompartmentalFit { inner })
    }

    /// The one-year view of `compartmental` by PSIS reweighting, with the
    /// arguments of `compartmental`; a simulation whose Pareto k is above
    /// `k_threshold` is handled as `refit` says, refitting at most
    /// `max_refits` (`NULL` for no limit).
    #[allow(clippy::too_many_arguments)]
    fn compartmental_one_year(
        &self,
        paid: &str,
        outstanding: &str,
        outstanding_from: &str,
        premium: &str,
        priors: Vec<f64>,
        families: Vec<String>,
        settings: Vec<String>,
        run: Vec<f64>,
        k_threshold: f64,
        refit: &str,
        max_refits: Nullable<f64>,
    ) -> Result<CompartmentalOneYearFit> {
        let model = compartmental_settings(&priors, &families, &settings, &run)?;
        let os = compartmental_outstanding(outstanding, outstanding_from)?;
        let view = one_year_settings(k_threshold, refit, max_refits)?;
        let inner = model
            .one_year(&self.inner, paid, os, premium, &view)
            .map_err(to_r)?;
        Ok(CompartmentalOneYearFit { inner })
    }

    /// Paths of the mean outstanding and cumulative paid drawn from the
    /// priors alone: `list(outstanding, paid)`, two predictive
    /// distributions with dimensions `origin` and `age`. The arguments are
    /// those of `compartmental`.
    fn compartmental_prior_predictive(
        &self,
        premium: &str,
        priors: Vec<f64>,
        families: Vec<String>,
        settings: Vec<String>,
        run: Vec<f64>,
    ) -> Result<List> {
        let model = compartmental_settings(&priors, &families, &settings, &run)?;
        let paths = model
            .prior_predictive(&self.inner, premium, model.n_sims)
            .map_err(to_r)?;
        Ok(list!(
            outstanding = PredictiveDistribution {
                inner: paths.outstanding
            },
            paid = PredictiveDistribution { inner: paths.paid }
        ))
    }

    /// The Bayesian Bornhuetter-Ferguson of a single-segment triangle
    /// column with each origin's exposure from the `exposure` column: a
    /// gamma prior with mean `apriori` times the exposure and coefficient
    /// of variation `cv` (infinite for a flat prior) on each origin's row
    /// parameter (its expected ultimate divided by the sum of the
    /// unnormalised pattern), a normal prior with sd `development_sd` on each
    /// log development effect, `process` error ("gamma" or "none"), NUTS
    /// with `chains`, `tune` and `draws` per chain, `target_accept` and
    /// `max_depth`, and `n_sims` predictive simulations, all replayed from
    /// `seed`.
    #[allow(clippy::too_many_arguments)]
    fn bayes_bf(
        &self,
        column: &str,
        exposure: &str,
        apriori: f64,
        cv: f64,
        development_sd: f64,
        process: &str,
        chains: f64,
        tune: f64,
        draws: f64,
        n_sims: f64,
        seed: f64,
        target_accept: f64,
        max_depth: f64,
    ) -> Result<BayesBfFit> {
        let model = bayes_bf_model(
            &[apriori, cv, development_sd],
            process,
            &[chains, tune, draws, n_sims, seed, target_accept, max_depth],
        )?;
        let inner = model.fit(&self.inner, column, exposure).map_err(to_r)?;
        Ok(BayesBfFit { inner })
    }

    /// The one-year view of `bayes_bf` on a single-segment triangle column
    /// by PSIS reweighting: `priors` are `apriori`, `cv` and
    /// `development_sd`, `run` is `chains`, `tune`, `draws`, `n_sims`,
    /// `seed`, `target_accept` and `max_depth`; a simulation whose Pareto
    /// k is above `k_threshold` is handled as `refit`
    /// ("moment_match_then_refit", "refit_flagged", "reweight_only" or
    /// "refit_all") says.
    #[allow(clippy::too_many_arguments)]
    fn bayes_bf_one_year(
        &self,
        column: &str,
        exposure: &str,
        priors: Vec<f64>,
        process: &str,
        run: Vec<f64>,
        k_threshold: f64,
        refit: &str,
        max_refits: Nullable<f64>,
    ) -> Result<BayesOneYearFit> {
        let model = bayes_bf_model(&priors, process, &run)?;
        let settings = one_year_settings(k_threshold, refit, max_refits)?;
        let inner = model
            .one_year(&self.inner, column, exposure, &settings)
            .map_err(to_r)?;
        Ok(BayesOneYearFit {
            inner: BayesOneYearOf::Bf(inner),
        })
    }

    /// The lifetime view under Mack's bootstrap: each origin's reserve
    /// simulated to the last age with `process` ("gamma", "lognormal",
    /// "residuals", "normal" or "none"), Mack's model averaged as `average`
    /// with `sigma_interpolation`, the residuals centred before resampling
    /// if `centre_residuals`, and the segments depending on each other as
    /// `dependence` and `spearman` say (as `odp_bootstrap`); then to
    /// ultimate with `tail` (a `ReservingTail`), whose `tail_sigma` and
    /// `tail_std_err` are `NULL` to extrapolate them.
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
        tail: Robj,
        tail_sigma: Nullable<f64>,
        tail_std_err: Nullable<f64>,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
        ldf: Robj,
    ) -> Result<MackBootstrapFit> {
        let boot = MackBootstrap {
            tail: tail_arg(&tail)?,
            tail_sigma: optional(tail_sigma),
            tail_std_err: optional(tail_std_err),
            ..mack_bootstrap(
                n_sims,
                seed,
                process,
                average,
                sigma_interpolation,
                centre_residuals,
                segment_dependence(dependence, spearman)?,
                selected_pattern(&ldf)?,
            )?
        };
        let inner = boot.fit_segments(&self.inner, column).map_err(to_r)?;
        Ok(MackBootstrapFit { inner })
    }

    /// The one-year view of `method` ("chain_ladder", "expected_loss",
    /// "bornhuetter_ferguson", "benktander" or "cape_cod") by re-reserving
    /// on the ODP bootstrap. `exposure` is `NULL` for the chain ladder and
    /// names the exposure column of the other methods; each method reads
    /// only its own settings (`apriori`, `n_iters`, `trend`, `decay`). The
    /// segments depend on each other as `dependence` and `spearman` say (as
    /// `odp_bootstrap`). With `bootstrap_tail`, the bootstrap takes
    /// `method`'s `tail` too, with `tail_std_err` (`NULL` to keep a constant
    /// tail fixed), and the origins whose tail step falls in the coming year
    /// take it.
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
        bootstrap_tail: bool,
        tail_std_err: Nullable<f64>,
        n_sims: f64,
        seed: f64,
        process: &str,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
        ldf: Robj,
        bootstrap_ldf: bool,
    ) -> Result<OneYearFit> {
        let boot_tail = if bootstrap_tail {
            tail_arg(&tail)?
        } else {
            Tail::default()
        };
        let method = one_year_method(
            method,
            exposure,
            apriori,
            n_iters,
            trend,
            decay,
            pattern(average, sigma_interpolation, tail, &ldf)?,
        )?;
        let dependence = segment_dependence(dependence, spearman)?;
        let inner = OdpBootstrap {
            tail: boot_tail,
            tail_std_err: optional(tail_std_err),
            one_year_pattern: one_year_pattern(bootstrap_ldf),
            ..bootstrap(n_sims, seed, process, dependence)?
        }
        .one_year_segments(&self.inner, column, &method)
        .map_err(to_r)?;
        Ok(OneYearFit {
            inner: OneYear::Odp(inner),
        })
    }

    /// The one-year view of `method`, as `odp_one_year`, under Mack's
    /// process: Mack's bootstrap with `process` ("gamma", "lognormal",
    /// "residuals", "normal" or "none"), Mack's model averaged as
    /// `mack_average` with `mack_sigma_interpolation`, the residuals centred
    /// before resampling if `centre_residuals`, and the segments depending
    /// on each other as `dependence` and `spearman` say. With
    /// `bootstrap_tail`, Mack's bootstrap takes `method`'s `tail` too, its
    /// `tail_sigma` and `tail_std_err` `NULL` to extrapolate them.
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
        bootstrap_tail: bool,
        tail_sigma: Nullable<f64>,
        tail_std_err: Nullable<f64>,
        n_sims: f64,
        seed: f64,
        process: &str,
        mack_average: &str,
        mack_sigma_interpolation: &str,
        centre_residuals: bool,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
        ldf: Robj,
        bootstrap_ldf: bool,
    ) -> Result<OneYearFit> {
        let boot_tail = if bootstrap_tail {
            tail_arg(&tail)?
        } else {
            Tail::default()
        };
        let method = one_year_method(
            method,
            exposure,
            apriori,
            n_iters,
            trend,
            decay,
            pattern(average, sigma_interpolation, tail, &ldf)?,
        )?;
        let inner = MackBootstrap {
            tail: boot_tail,
            tail_sigma: optional(tail_sigma),
            tail_std_err: optional(tail_std_err),
            one_year_pattern: one_year_pattern(bootstrap_ldf),
            ..mack_bootstrap(
                n_sims,
                seed,
                process,
                mack_average,
                mack_sigma_interpolation,
                centre_residuals,
                segment_dependence(dependence, spearman)?,
                None,
            )?
        }
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

    /// Standard errors of the factors used before the tail: `1 - weight`
    /// times the estimate's.
    fn std_err(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "std_err", "development_frame()")?;
        Ok(f.development.blended_std_err())
    }

    fn estimated_std_err(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "estimated_std_err", "development_frame()")?;
        Ok(f.development.std_err.clone())
    }

    /// The selected pattern's factor of each link; NaN where none.
    fn selected_ldf(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "selected_ldf", "development_frame()")?;
        Ok(f.development
            .selected
            .iter()
            .map(|s| s.unwrap_or(f64::NAN))
            .collect())
    }

    fn ldf_weight(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "ldf_weight", "development_frame()")?;
        Ok(f.development.weight.clone())
    }

    fn blended_ldf(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "blended_ldf", "development_frame()")?;
        Ok(f.development.blended_ldf())
    }

    /// The rescale of a cumulative pattern; NaN for an age-to-age pattern
    /// or none.
    fn pattern_rescale(&self) -> Result<f64> {
        let f = single(&self.inner, "pattern_rescale", "totals_frame()")?;
        Ok(f.pattern_rescale.unwrap_or(f64::NAN))
    }

    /// The tail a cumulative pattern implies; NaN where it implies none.
    fn pattern_tail(&self) -> Result<f64> {
        let f = single(&self.inner, "pattern_tail", "totals_frame()")?;
        Ok(f.pattern_tail.unwrap_or(f64::NAN))
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
    /// The chain ladder the bootstrap is centred on, with the bootstrap's
    /// tail.
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

    /// Mack's model on the observed triangle, with the bootstrap's tail.
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

/// A Bayesian ODP chain ladder of a single-segment triangle column.
#[extendr]
pub(crate) struct BayesOdpFit {
    inner: BayesOdpInner,
}

#[extendr]
impl BayesOdpFit {
    /// The volume-weighted chain ladder the model takes its latest values
    /// and scale from.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    fn scale(&self) -> f64 {
        self.inner.scale
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// `list(max_rhat, min_ess_bulk, min_ess_tail, divergences, warnings)`.
    fn diagnostics(&self) -> List {
        let d = &self.inner.diagnostics;
        list!(
            max_rhat = d.max_rhat,
            min_ess_bulk = d.min_ess_bulk,
            min_ess_tail = d.min_ess_tail,
            divergences = d.divergences as f64,
            warnings = d.warnings.clone()
        )
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, row-major `n_draws x n_parameters`, chain by chain.
    fn draws(&self) -> Vec<f64> {
        self.inner.posterior.values().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.posterior.chains() as f64
    }

    /// The posterior summary, columns as `bayes_glm_model@summary`.
    fn summary(&self) -> Result<List> {
        let s = self.inner.summary().map_err(to_r)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        Ok(list!(
            parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            mean = col(|p| p.mean),
            sd = col(|p| p.sd),
            q05 = col(|p| p.q05),
            q50 = col(|p| p.q50),
            q95 = col(|p| p.q95),
            rhat = col(|p| p.rhat),
            ess_bulk = col(|p| p.ess_bulk),
            ess_tail = col(|p| p.ess_tail)
        ))
    }

    /// PSIS-LOO of the observed increments: `list(estimates, pointwise,
    /// pareto_k, k_threshold)`, as `BayesGlmFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.inner.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
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

    /// The fit itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

/// The `BayesOdp` of the one-year views: `priors` are `intercept_sd`,
/// `origin_sd` and `development_sd`; `run` is `chains`, `tune`, `draws`,
/// `n_sims`, `seed`, `target_accept` and `max_depth`.
fn bayes_odp_model(
    priors: &[f64],
    process: &str,
    run: &[f64],
    dependence: SegmentDependence,
) -> Result<BayesOdp> {
    let (&[intercept_sd, origin_sd, development_sd], &[chains, tune, draws, n_sims, seed, ta, md]) =
        (priors, run)
    else {
        return Err(Error::Other(
            "bayes_odp_model takes 3 priors and 7 run settings".into(),
        ));
    };
    let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
    Ok(BayesOdp {
        intercept_sd,
        origin_sd,
        development_sd,
        process: boot.process,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept: ta,
            max_depth: whole(md, "max_depth")?,
        },
        n_sims: boot.n_sims,
        seed: boot.seed,
        dependence,
    })
}

/// The `BayesBf` of `bayes_bf` and `bayes_bf_one_year`: `priors` are
/// `apriori`, `cv` and `development_sd`, `run` is `chains`, `tune`,
/// `draws`, `n_sims`, `seed`, `target_accept` and `max_depth`.
fn bayes_bf_model(priors: &[f64], process: &str, run: &[f64]) -> Result<BayesBf> {
    let (&[apriori, cv, development_sd], &[chains, tune, draws, n_sims, seed, ta, md]) =
        (priors, run)
    else {
        return Err(Error::Other(
            "bayes_bf_model takes 3 priors and 7 run settings".into(),
        ));
    };
    let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
    Ok(BayesBf {
        apriori,
        cv,
        development_sd,
        process: boot.process,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept: ta,
            max_depth: whole(md, "max_depth")?,
        },
        n_sims: boot.n_sims,
        seed: boot.seed,
    })
}

/// The settings of a Bayesian one-year view: `k_threshold`, the `refit`
/// policy by name and the most refits (`NULL` for no limit).
fn one_year_settings(
    k_threshold: f64,
    refit: &str,
    max_refits: Nullable<f64>,
) -> Result<BayesOneYear> {
    let refit = match refit {
        "refit_flagged" => OneYearRefit::RefitFlagged,
        "reweight_only" => OneYearRefit::ReweightOnly,
        "refit_all" => OneYearRefit::RefitAll,
        "moment_match_then_refit" => OneYearRefit::MomentMatchThenRefit,
        _ => {
            return Err(Error::Other(format!(
                "refit must be \"moment_match_then_refit\", \"refit_flagged\", \
                 \"reweight_only\" or \"refit_all\", got \"{refit}\""
            )));
        }
    };
    let max_refits = match max_refits {
        Nullable::NotNull(m) => Some(whole(m, "max_refits")? as usize),
        Nullable::Null => None,
    };
    Ok(BayesOneYear {
        k_threshold,
        refit,
        max_refits,
    })
}

/// The one-year view of a Bayesian model on a single-segment triangle
/// column: the lifetime fit, the opening reserve, the claims development
/// result and each simulation's Pareto k.
#[extendr]
pub(crate) struct BayesOneYearFit {
    inner: BayesOneYearOf,
}

/// The one-year view of each Bayesian model with one, by its lifetime fit.
#[derive(Clone)]
enum BayesOneYearOf {
    Odp(BayesOneYearInner<BayesOdpInner>),
    Bf(BayesOneYearInner<BayesBfInner>),
    Meyers(Box<BayesOneYearInner<MeyersInner>>),
}

/// Evaluates `$body` with `$f` bound to the view, whatever its model.
macro_rules! each_bayes_one_year {
    ($view:expr, $f:ident => $body:expr) => {
        match $view {
            BayesOneYearOf::Odp($f) => $body,
            BayesOneYearOf::Bf($f) => $body,
            BayesOneYearOf::Meyers($f) => $body,
        }
    };
}

#[extendr]
impl BayesOneYearFit {
    /// The model behind the view: "bayes_odp", "bayes_bf" or "meyers".
    fn model(&self) -> &'static str {
        match &self.inner {
            BayesOneYearOf::Odp(_) => "bayes_odp",
            BayesOneYearOf::Bf(_) => "bayes_bf",
            BayesOneYearOf::Meyers(_) => "meyers",
        }
    }

    /// The lifetime fit, as `bayes_odp` gives it; an error for another
    /// model.
    fn fit(&self) -> Result<BayesOdpFit> {
        match &self.inner {
            BayesOneYearOf::Odp(f) => Ok(BayesOdpFit {
                inner: f.fit.clone(),
            }),
            _ => Err(Error::Other(format!(
                "the one-year view is {}'s, not bayes_odp's",
                self.model()
            ))),
        }
    }

    /// The lifetime fit, as `bayes_bf` gives it; an error for another
    /// model.
    fn bf_fit(&self) -> Result<BayesBfFit> {
        match &self.inner {
            BayesOneYearOf::Bf(f) => Ok(BayesBfFit {
                inner: f.fit.clone(),
            }),
            _ => Err(Error::Other(format!(
                "the one-year view is {}'s, not bayes_bf's",
                self.model()
            ))),
        }
    }

    /// The lifetime fit, as `meyers` gives it; an error for another model.
    fn meyers_fit(&self) -> Result<MeyersFit> {
        match &self.inner {
            BayesOneYearOf::Meyers(f) => Ok(MeyersFit {
                inner: f.fit.clone(),
            }),
            _ => Err(Error::Other(format!(
                "the one-year view is {}'s, not meyers'",
                self.model()
            ))),
        }
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.opening_ultimate.clone())
    }

    fn opening_reserve(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.opening_reserve.clone())
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: each_bayes_one_year!(&self.inner, f => f.cdr.clone()),
        }
    }

    fn pareto_k(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f.pareto_k.clone())
    }

    /// The simulations refitted with NUTS, 1-based.
    fn refitted(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    /// The simulations moment matching rescued, 1-based.
    fn moment_matched(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    /// The simulations due for a refit beyond `max_refits`, reweighted
    /// instead, 1-based.
    fn unrefitted(&self) -> Vec<f64> {
        each_bayes_one_year!(&self.inner, f => f
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect())
    }

    fn refit_warnings(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.refit_warnings as f64)
    }

    fn k_threshold(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.k_threshold)
    }

    fn flagged_share(&self) -> f64 {
        each_bayes_one_year!(&self.inner, f => f.flagged_share())
    }

    fn long_table(&self) -> List {
        fit_table(each_bayes_one_year!(&self.inner, f => f.to_long()))
    }

    fn totals_table(&self) -> List {
        fit_table(each_bayes_one_year!(&self.inner, f => f.totals()))
    }

    fn development_table(&self) -> List {
        fit_table(each_bayes_one_year!(&self.inner, f => f.fit.development_table()))
    }

    /// The fit itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

/// The one-year view of a Bayesian model on every segment of a triangle
/// column: each segment's posterior, opening reserve and Pareto k, and the
/// joint claims development result.
#[extendr]
pub(crate) struct BayesOneYearFits {
    inner: BayesOneYearFitsOf,
}

/// The one-year view over segments of each Bayesian model with one, by
/// its segment fit.
enum BayesOneYearFitsOf {
    Odp(BayesOneYearFitsInner<BayesOdpSegment>),
    Meyers(BayesOneYearFitsInner<MeyersSegment>),
}

/// Evaluates `$body` with `$f` bound to the views over segments, whatever
/// the model.
macro_rules! each_bayes_one_years {
    ($views:expr, $f:ident => $body:expr) => {
        match $views {
            BayesOneYearFitsOf::Odp($f) => $body,
            BayesOneYearFitsOf::Meyers($f) => $body,
        }
    };
}

#[extendr]
impl BayesOneYearFits {
    /// The model behind the view: "bayes_odp" or "meyers".
    fn model(&self) -> &'static str {
        match &self.inner {
            BayesOneYearFitsOf::Odp(_) => "bayes_odp",
            BayesOneYearFitsOf::Meyers(_) => "meyers",
        }
    }

    /// The volume-weighted chain ladder of every segment.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: each_bayes_one_years!(&self.inner, f => f
                .segments
                .map(|s| s.fit.chain_ladder.clone())),
        }
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .opening_ultimate
            .clone()))
    }

    fn opening_reserve(&self) -> Vec<f64> {
        each_bayes_one_years!(&self.inner, f => by_origin(&f.segments, |s| s
            .opening_reserve
            .clone()))
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: each_bayes_one_years!(&self.inner, f => f.cdr.clone()),
        }
    }

    /// The plugged-in scale of a single-segment `bayes_odp` view.
    fn scale(&self) -> Result<f64> {
        match &self.inner {
            BayesOneYearFitsOf::Odp(f) => {
                Ok(single(&f.segments, "scale", "totals_frame()")?.fit.scale)
            }
            BayesOneYearFitsOf::Meyers(_) => Err(Error::Other(
                "scale is the ODP's plugged-in dispersion; Meyers' models have none".into(),
            )),
        }
    }

    /// The premium of each origin of a single-segment `meyers` view.
    fn premium(&self) -> Result<Vec<f64>> {
        match &self.inner {
            BayesOneYearFitsOf::Meyers(f) => Ok(single(&f.segments, "premium", "totals_frame()")?
                .fit
                .premium
                .clone()),
            BayesOneYearFitsOf::Odp(_) => Err(Error::Other(
                "premium is Meyers' input; bayes_odp has none".into(),
            )),
        }
    }

    fn pareto_k(&self) -> Result<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "pareto_k", "")?
            .pareto_k
            .clone()))
    }

    /// The simulations of a single-segment fit refitted with NUTS, 1-based.
    fn refitted(&self) -> Result<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "refitted", "")?
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()))
    }

    /// The simulations of a single-segment fit that moment matching
    /// rescued, 1-based.
    fn moment_matched(&self) -> Result<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "moment_matched", "")?
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()))
    }

    fn unrefitted(&self) -> Result<Vec<f64>> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "unrefitted", "")?
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()))
    }

    fn refit_warnings(&self) -> Result<f64> {
        each_bayes_one_years!(&self.inner, f => Ok(single(&f.segments, "refit_warnings", "")?
            .refit_warnings as f64))
    }

    /// Each segment's number of refits and of refits with a diagnostic
    /// warning, for the warning of the one-year views over segments.
    fn refit_counts(&self) -> List {
        each_bayes_one_years!(&self.inner, f => {
            let fits = &f.segments.fits;
            list!(
                refits = fits
                    .iter()
                    .map(|s| s.refitted.len() as f64)
                    .collect::<Vec<_>>(),
                warned = fits
                    .iter()
                    .map(|s| s.refit_warnings as f64)
                    .collect::<Vec<_>>(),
                unrefitted = fits
                    .iter()
                    .map(|s| s.unrefitted.len() as f64)
                    .collect::<Vec<_>>()
            )
        })
    }

    fn k_threshold(&self) -> f64 {
        each_bayes_one_years!(&self.inner, f => f.segments.fits[0].k_threshold)
    }

    fn flagged_share(&self) -> Result<f64> {
        each_bayes_one_years!(&self.inner, f => Ok(single(
            &f.segments,
            "flagged_share",
            "totals_frame()"
        )?
        .flagged_share()))
    }

    /// One row per segment: `list(table, warnings)`, as
    /// `segment_diagnostics`.
    fn diagnostics_table(&self) -> List {
        match &self.inner {
            BayesOneYearFitsOf::Odp(f) => {
                segment_diagnostics(&f.segments, f.totals(), |s| &s.fit.diagnostics)
            }
            BayesOneYearFitsOf::Meyers(f) => {
                segment_diagnostics(&f.segments, f.totals(), |s| &s.fit.diagnostics)
            }
        }
    }

    fn long_table(&self) -> List {
        fit_table(each_bayes_one_years!(&self.inner, f => f.to_long()))
    }

    fn totals_table(&self) -> List {
        fit_table(each_bayes_one_years!(&self.inner, f => f.totals()))
    }

    fn development_table(&self) -> List {
        fit_table(each_bayes_one_years!(&self.inner, f => f.segments.development_table()))
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let keys = choice(&keys, &values)?;
        let inner = match &self.inner {
            BayesOneYearFitsOf::Odp(f) => BayesOneYearFitsOf::Odp(f.segment(&keys).map_err(to_r)?),
            BayesOneYearFitsOf::Meyers(f) => {
                BayesOneYearFitsOf::Meyers(f.segment(&keys).map_err(to_r)?)
            }
        };
        Ok(Self { inner })
    }
}

/// The `Meyers` of `meyers` and `meyers_segments`: `priors` are
/// `logelr_mean`, `logelr_sd`, `origin_sd`, `development_sd`, `gamma_sd` and
/// `rho_shape`; `run` is `chains`, `tune`, `draws`, `n_sims`, `seed`,
/// `target_accept` and `max_depth`.
fn meyers_settings(
    model: &str,
    priors: [f64; 6],
    run: [f64; 7],
    dependence: SegmentDependence,
) -> Result<Meyers> {
    let model = match model {
        "crc" => MeyersModel::Crc,
        "csr" => MeyersModel::Csr,
        "cay" => MeyersModel::Cay,
        _ => {
            return Err(Error::Other(format!(
                "model must be \"crc\", \"csr\" or \"cay\", got \"{model}\""
            )));
        }
    };
    let [
        logelr_mean,
        logelr_sd,
        origin_sd,
        development_sd,
        gamma_sd,
        rho_shape,
    ] = priors;
    let [chains, tune, draws, n_sims, seed, target_accept, max_depth] = run;
    let n_sims = whole(n_sims, "n_sims")? as usize;
    if n_sims == 0 {
        return Err(Error::Other("n_sims must be positive".into()));
    }
    Ok(Meyers {
        model,
        logelr_mean,
        logelr_sd,
        origin_sd,
        development_sd,
        gamma_sd,
        rho_shape,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept,
            max_depth: whole(max_depth, "max_depth")?,
        },
        n_sims,
        seed: whole(seed, "seed")?,
        dependence,
    })
}

/// One row per segment: `list(table, warnings)`, `table` a fit table with
/// the key columns of `totals` and `max_rhat`, `min_ess_bulk`,
/// `min_ess_tail` and `divergences`, `warnings` each segment's messages
/// joined by "; ".
fn segment_diagnostics<T>(
    segments: &SegmentFits<T>,
    totals: FitTable,
    diag: fn(&T) -> &Diagnostics,
) -> List {
    let fits = &segments.fits;
    let col = |g: fn(&Diagnostics) -> f64| fits.iter().map(|f| g(diag(f))).collect::<Vec<f64>>();
    let table = FitTable {
        keys: totals.keys,
        origin: None,
        age: None,
        values: vec![
            ("max_rhat".into(), col(|d| d.max_rhat)),
            ("min_ess_bulk".into(), col(|d| d.min_ess_bulk)),
            ("min_ess_tail".into(), col(|d| d.min_ess_tail)),
            ("divergences".into(), col(|d| d.divergences as f64)),
        ],
    };
    let warnings: Vec<String> = fits.iter().map(|f| diag(f).warnings.join("; ")).collect();
    list!(table = fit_table(table), warnings = warnings)
}

/// A posterior summary as a list of columns, as `bayes_glm_model@summary`.
fn summary_list(s: &[ParamSummary]) -> List {
    let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
    list!(
        parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
        mean = col(|p| p.mean),
        sd = col(|p| p.sd),
        q05 = col(|p| p.q05),
        q50 = col(|p| p.q50),
        q95 = col(|p| p.q95),
        rhat = col(|p| p.rhat),
        ess_bulk = col(|p| p.ess_bulk),
        ess_tail = col(|p| p.ess_tail)
    )
}

/// The Bayesian ODP chain ladder of every segment of a triangle column:
/// each segment's posterior and the joint reserves. The posterior of one
/// segment needs a single-segment fit.
#[extendr]
pub(crate) struct BayesOdpFits {
    inner: BayesOdpFitsInner,
}

impl BayesOdpFits {
    fn one(&self, field: &str, instead: &str) -> Result<&BayesOdpSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[extendr]
impl BayesOdpFits {
    /// The volume-weighted chain ladder of every segment.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    fn scale(&self) -> Result<f64> {
        Ok(self.one("scale", "totals_frame()")?.scale)
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// One row per segment: `list(table, warnings)`, as
    /// `segment_diagnostics`.
    fn diagnostics_table(&self) -> List {
        segment_diagnostics(&self.inner.segments, self.inner.totals(), |s| {
            &s.diagnostics
        })
    }

    fn parameter_names(&self) -> Result<Vec<String>> {
        Ok(self.one("draws", "")?.posterior.names().to_vec())
    }

    /// Posterior draws of a single-segment fit, row-major `n_draws x
    /// n_parameters`, chain by chain.
    fn draws(&self) -> Result<Vec<f64>> {
        Ok(self.one("draws", "")?.posterior.values().to_vec())
    }

    fn chains(&self) -> Result<f64> {
        Ok(self.one("draws", "")?.posterior.chains() as f64)
    }

    fn summary(&self) -> Result<List> {
        let s = self.one("summary", "")?.summary().map_err(to_r)?;
        Ok(summary_list(&s))
    }

    /// PSIS-LOO of a single-segment fit's observed increments, as
    /// `BayesOdpFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.one("bayes_loo()", "")?.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
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

/// Meyers' CRC, CSR or CAY model of a single-segment triangle column.
#[extendr]
pub(crate) struct MeyersFit {
    inner: MeyersInner,
}

#[extendr]
impl MeyersFit {
    /// "crc", "csr" or "cay".
    fn model(&self) -> &'static str {
        self.inner.model().name()
    }

    /// The volume-weighted chain ladder the reserves are measured from.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    fn premium(&self) -> Vec<f64> {
        self.inner.premium.clone()
    }

    fn ultimates(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.ultimates.clone(),
        }
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// `list(max_rhat, min_ess_bulk, min_ess_tail, divergences, warnings)`.
    fn diagnostics(&self) -> List {
        let d = &self.inner.diagnostics;
        list!(
            max_rhat = d.max_rhat,
            min_ess_bulk = d.min_ess_bulk,
            min_ess_tail = d.min_ess_tail,
            divergences = d.divergences as f64,
            warnings = d.warnings.clone()
        )
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, row-major `n_draws x n_parameters`, chain by chain.
    fn draws(&self) -> Vec<f64> {
        self.inner.posterior.values().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.posterior.chains() as f64
    }

    /// The posterior summary, columns as `bayes_glm_model@summary`.
    fn summary(&self) -> Result<List> {
        let s = self.inner.summary().map_err(to_r)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        Ok(list!(
            parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            mean = col(|p| p.mean),
            sd = col(|p| p.sd),
            q05 = col(|p| p.q05),
            q50 = col(|p| p.q50),
            q95 = col(|p| p.q95),
            rhat = col(|p| p.rhat),
            ess_bulk = col(|p| p.ess_bulk),
            ess_tail = col(|p| p.ess_tail)
        ))
    }

    /// PSIS-LOO of the observed cells on the log scale: `list(estimates,
    /// pointwise, pareto_k, k_threshold)`, as `BayesGlmFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.inner.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
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

    /// The fit itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

/// A Bayesian Bornhuetter-Ferguson of a single-segment triangle column.
#[extendr]
pub(crate) struct BayesBfFit {
    inner: BayesBfInner,
}

#[extendr]
impl BayesBfFit {
    /// The volume-weighted chain ladder the model takes its latest values
    /// and scale from.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: SegmentFits {
                key_names: Vec::new(),
                labels: vec![Label::new(Vec::<String>::new())],
                fits: vec![self.inner.chain_ladder.clone()],
            },
        }
    }

    fn scale(&self) -> f64 {
        self.inner.scale
    }

    fn exposure(&self) -> Vec<f64> {
        self.inner.exposure.clone()
    }

    fn prior_ultimate(&self) -> Vec<f64> {
        self.inner.prior_ultimate.clone()
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// `list(max_rhat, min_ess_bulk, min_ess_tail, divergences, warnings)`.
    fn diagnostics(&self) -> List {
        let d = &self.inner.diagnostics;
        list!(
            max_rhat = d.max_rhat,
            min_ess_bulk = d.min_ess_bulk,
            min_ess_tail = d.min_ess_tail,
            divergences = d.divergences as f64,
            warnings = d.warnings.clone()
        )
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, row-major `n_draws x n_parameters`, chain by chain.
    fn draws(&self) -> Vec<f64> {
        self.inner.posterior.values().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.posterior.chains() as f64
    }

    /// The posterior summary, columns as `bayes_glm_model@summary`.
    fn summary(&self) -> Result<List> {
        let s = self.inner.summary().map_err(to_r)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        Ok(list!(
            parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            mean = col(|p| p.mean),
            sd = col(|p| p.sd),
            q05 = col(|p| p.q05),
            q50 = col(|p| p.q50),
            q95 = col(|p| p.q95),
            rhat = col(|p| p.rhat),
            ess_bulk = col(|p| p.ess_bulk),
            ess_tail = col(|p| p.ess_tail)
        ))
    }

    /// PSIS-LOO of the observed increments: `list(estimates, pointwise,
    /// pareto_k, k_threshold)`, as `BayesGlmFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.inner.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
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

    /// The fit itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

/// Meyers' model of every segment of a triangle column: each segment's
/// posterior and the joint ultimates and reserves. The posterior of one
/// segment needs a single-segment fit.
#[extendr]
pub(crate) struct MeyersFits {
    inner: MeyersFitsInner,
}

impl MeyersFits {
    fn one(&self, field: &str, instead: &str) -> Result<&MeyersSegment> {
        single(&self.inner.segments, field, instead)
    }
}

#[extendr]
impl MeyersFits {
    /// "crc", "csr" or "cay".
    fn model(&self) -> &'static str {
        self.inner.segments.fits[0].model().name()
    }

    /// The volume-weighted chain ladder of every segment.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.segments.map(|s| s.chain_ladder.clone()),
        }
    }

    /// Each segment's premium by origin, in the order of the origins.
    fn premium(&self) -> Vec<f64> {
        self.inner
            .segments
            .fits
            .iter()
            .flat_map(|s| s.premium.iter().copied())
            .collect()
    }

    fn ultimates(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.ultimates.clone(),
        }
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    /// One row per segment: `list(table, warnings)`, as
    /// `segment_diagnostics`.
    fn diagnostics_table(&self) -> List {
        segment_diagnostics(&self.inner.segments, self.inner.totals(), |s| {
            &s.diagnostics
        })
    }

    fn parameter_names(&self) -> Result<Vec<String>> {
        Ok(self.one("draws", "")?.posterior.names().to_vec())
    }

    /// Posterior draws of a single-segment fit, row-major `n_draws x
    /// n_parameters`, chain by chain.
    fn draws(&self) -> Result<Vec<f64>> {
        Ok(self.one("draws", "")?.posterior.values().to_vec())
    }

    fn chains(&self) -> Result<f64> {
        Ok(self.one("draws", "")?.posterior.chains() as f64)
    }

    fn summary(&self) -> Result<List> {
        let s = self.one("summary", "")?.summary().map_err(to_r)?;
        Ok(summary_list(&s))
    }

    /// PSIS-LOO of a single-segment fit's observed cells on the log scale,
    /// as `MeyersFit$loo()`.
    fn loo(&self) -> Result<List> {
        let l = self.one("bayes_loo()", "")?.loo().map_err(to_r)?;
        let e = &l.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.pareto_k,
            k_threshold = l.k_threshold
        ))
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

/// The `CompartmentalPriors` of `priors` (31 numbers) and `families` (10
/// names): `k_er`, `k_p`, `rlr` and `rrf` as (median, sd); the half priors
/// on the effect sds of `RLR` and `RRF` as (nu, scale) for "student_t" or
/// (scale, NaN) for "normal"; `lkj_eta`; the priors on `log sigma_os` and
/// `log sigma_pd` as (nu, location, scale) for "student_t" or (mean, sd,
/// NaN) for "normal"; then the optional half priors on the origin effect
/// sds of `k_er` and `k_p` and on the development effect sds of `k_er`,
/// `k_p`, `rlr` and `rrf`, each two numbers with family "none" (NaN, NaN)
/// for no such effect.
fn compartmental_priors(priors: &[f64], families: &[String]) -> Result<CompartmentalPriors> {
    if priors.len() != 31 || families.len() != 10 {
        return Err(Error::Other(
            "priors must hold 31 numbers and 10 family names".into(),
        ));
    }
    let positive = |i: usize| PositivePrior::LogNormal {
        median: priors[i],
        sd: priors[i + 1],
    };
    let half = |family: &str, i: usize, arg: &str| -> Result<HalfPrior> {
        match family {
            "student_t" => Ok(HalfPrior::StudentT {
                nu: priors[i],
                scale: priors[i + 1],
            }),
            "normal" => Ok(HalfPrior::Normal { scale: priors[i] }),
            _ => Err(Error::Other(format!(
                "{arg} family must be \"student_t\" or \"normal\", got \"{family}\""
            ))),
        }
    };
    let real = |family: &str, i: usize, arg: &str| -> Result<RealPrior> {
        match family {
            "student_t" => Ok(RealPrior::StudentT {
                nu: priors[i],
                location: priors[i + 1],
                scale: priors[i + 2],
            }),
            "normal" => Ok(RealPrior::Normal {
                mean: priors[i],
                sd: priors[i + 1],
            }),
            _ => Err(Error::Other(format!(
                "{arg} family must be \"student_t\" or \"normal\", got \"{family}\""
            ))),
        }
    };
    let optional = |j: usize, arg: &str| -> Result<Option<HalfPrior>> {
        let family = &families[4 + j];
        if family == "none" {
            Ok(None)
        } else {
            half(family, 19 + 2 * j, arg).map(Some)
        }
    };
    Ok(CompartmentalPriors {
        k_er: positive(0),
        k_p: positive(2),
        rlr: positive(4),
        rrf: positive(6),
        rlr_effect_sd: half(&families[0], 8, "rlr_effect_sd")?,
        rrf_effect_sd: half(&families[1], 10, "rrf_effect_sd")?,
        lkj_eta: priors[12],
        log_sigma: [
            real(&families[2], 13, "log_sigma[[1]]")?,
            real(&families[3], 16, "log_sigma[[2]]")?,
        ],
        ker_effect_sd: optional(0, "ker_effect_sd")?,
        kp_effect_sd: optional(1, "kp_effect_sd")?,
        development_effect_sd: [
            optional(2, "development_effect_sd$k_er")?,
            optional(3, "development_effect_sd$k_p")?,
            optional(4, "development_effect_sd$rlr")?,
            optional(5, "development_effect_sd$rrf")?,
        ],
    })
}

/// The `Compartmental` of `compartmental` and
/// `compartmental_prior_predictive`.
fn compartmental_settings(
    priors: &[f64],
    families: &[String],
    settings: &[String],
    run: &[f64],
) -> Result<Compartmental> {
    let [likelihood, scale, paid_response, fitted, rate_order, start] = settings else {
        return Err(Error::Other("settings must hold 6 names".into()));
    };
    let &[chains, tune, draws, n_sims, seed, target_accept, max_depth] = run else {
        return Err(Error::Other("run must hold 7 numbers".into()));
    };
    let bad = |arg: &str, value: &str, expected: &str| {
        Err(Error::Other(format!(
            "{arg} must be {expected}, got \"{value}\""
        )))
    };
    let likelihood = match likelihood.as_str() {
        "normal" => CompartmentalLikelihood::Normal,
        "lognormal" => CompartmentalLikelihood::LogNormal,
        v => return bad("likelihood", v, "\"normal\" or \"lognormal\""),
    };
    let scale = match scale.as_str() {
        "amount" => CompartmentalScale::Amount,
        "loss_ratio" => CompartmentalScale::LossRatio,
        v => return bad("scale", v, "\"amount\" or \"loss_ratio\""),
    };
    let paid_response = match paid_response.as_str() {
        "cumulative" => PaidResponse::Cumulative,
        "incremental" => PaidResponse::Incremental,
        v => return bad("paid_response", v, "\"cumulative\" or \"incremental\""),
    };
    let fitted = match fitted.as_str() {
        "paid_and_outstanding" => Fitted::PaidAndOutstanding,
        "paid_and_incurred" => Fitted::PaidAndIncurred,
        "paid_only" => Fitted::PaidOnly,
        "incurred_only" => Fitted::IncurredOnly,
        v => {
            return bad(
                "fitted",
                v,
                "\"paid_and_outstanding\", \"paid_and_incurred\", \"paid_only\" or \"incurred_only\"",
            );
        }
    };
    let rate_order = match rate_order.as_str() {
        "free" => RateOrder::Free,
        "ordered" => RateOrder::Ordered,
        v => return bad("rate_order", v, "\"free\" or \"ordered\""),
    };
    let start = match start.as_str() {
        "data" => CompartmentalStart::Data,
        "prior_centre" => CompartmentalStart::PriorCentre,
        v => return bad("start", v, "\"data\" or \"prior_centre\""),
    };
    let n_sims = whole(n_sims, "n_sims")? as usize;
    if n_sims == 0 {
        return Err(Error::Other("n_sims must be positive".into()));
    }
    Ok(Compartmental {
        priors: compartmental_priors(priors, families)?,
        likelihood,
        scale,
        paid_response,
        fitted,
        rate_order,
        start,
        sampler: BayesSampler {
            chains: whole(chains, "chains")? as usize,
            tune: whole(tune, "tune")? as usize,
            draws: whole(draws, "draws")? as usize,
            seed: 0,
            target_accept,
            max_depth: whole(max_depth, "max_depth")?,
        },
        n_sims,
        seed: whole(seed, "seed")?,
    })
}

/// Where a compartmental fit's outstanding comes from: the `outstanding`
/// column itself ("column") or incurred less paid ("incurred").
fn compartmental_outstanding<'a>(
    outstanding: &'a str,
    outstanding_from: &str,
) -> Result<Outstanding<'a>> {
    match outstanding_from {
        "column" => Ok(Outstanding::Column(outstanding)),
        "incurred" => Ok(Outstanding::FromIncurred(outstanding)),
        _ => Err(Error::Other(format!(
            "outstanding_from must be \"column\" or \"incurred\", got \"{outstanding_from}\""
        ))),
    }
}

/// The one-year view of a compartmental model: the lifetime fit, the
/// opening reserve, the claims development result and each simulation's
/// Pareto k, by the premium column's origins.
#[extendr]
pub(crate) struct CompartmentalOneYearFit {
    inner: BayesOneYearInner<CompartmentalInner>,
}

#[extendr]
impl CompartmentalOneYearFit {
    fn fit(&self) -> CompartmentalFit {
        CompartmentalFit {
            inner: self.inner.fit.clone(),
        }
    }

    fn origins(&self) -> Vec<String> {
        self.inner
            .fit
            .origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    fn latest_paid(&self) -> Vec<f64> {
        self.inner.fit.latest_paid.clone()
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        self.inner.opening_ultimate.clone()
    }

    fn opening_reserve(&self) -> Vec<f64> {
        self.inner.opening_reserve.clone()
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.cdr.clone(),
        }
    }

    fn pareto_k(&self) -> Vec<f64> {
        self.inner.pareto_k.clone()
    }

    /// The simulations refitted with NUTS, 1-based.
    fn refitted(&self) -> Vec<f64> {
        self.inner
            .refitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    /// The simulations moment matching rescued, 1-based.
    fn moment_matched(&self) -> Vec<f64> {
        self.inner
            .moment_matched
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    /// The simulations due for a refit beyond `max_refits`, reweighted
    /// instead, 1-based.
    fn unrefitted(&self) -> Vec<f64> {
        self.inner
            .unrefitted
            .iter()
            .map(|&i| (i + 1) as f64)
            .collect()
    }

    fn refit_warnings(&self) -> f64 {
        self.inner.refit_warnings as f64
    }

    fn k_threshold(&self) -> f64 {
        self.inner.k_threshold
    }

    fn flagged_share(&self) -> f64 {
        self.inner.flagged_share()
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    /// The view itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
        })
    }
}

/// A fitted Gesmann and Morris compartmental model of a single-segment triangle.
#[extendr]
pub(crate) struct CompartmentalFit {
    inner: CompartmentalInner,
}

#[extendr]
impl CompartmentalFit {
    fn origins(&self) -> Vec<String> {
        self.inner.origins.iter().map(ToString::to_string).collect()
    }

    fn premium(&self) -> Vec<f64> {
        self.inner.premium.clone()
    }

    fn latest_paid(&self) -> Vec<f64> {
        self.inner.latest_paid.clone()
    }

    fn population_origins(&self) -> Vec<String> {
        self.inner
            .population_origins
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    fn reserves(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.reserves.clone(),
        }
    }

    fn paid(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.paid.clone(),
        }
    }

    fn outstanding(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.outstanding.clone(),
        }
    }

    fn ultimate_loss_ratio(&self) -> Vec<f64> {
        self.inner.ultimate_loss_ratio.clone()
    }

    fn rate_swaps(&self) -> f64 {
        self.inner.rate_swaps as f64
    }

    /// The share of draws and observed cells with `k_er <= k_p` when the
    /// rates vary by cell; NA when they are shared.
    fn cell_rate_swaps(&self) -> Rfloat {
        self.inner
            .cell_rate_swaps
            .map_or(Rfloat::na(), Rfloat::from)
    }

    /// `list(max_rhat, min_ess_bulk, min_ess_tail, divergences, warnings)`.
    fn diagnostics(&self) -> List {
        let d = &self.inner.diagnostics;
        list!(
            max_rhat = d.max_rhat,
            min_ess_bulk = d.min_ess_bulk,
            min_ess_tail = d.min_ess_tail,
            divergences = d.divergences as f64,
            warnings = d.warnings.clone()
        )
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.posterior.names().to_vec()
    }

    /// Posterior draws, row-major `n_draws x n_parameters`, chain by chain.
    fn draws(&self) -> Vec<f64> {
        self.inner.posterior.values().to_vec()
    }

    fn chains(&self) -> f64 {
        self.inner.posterior.chains() as f64
    }

    /// The posterior summary, columns as `bayes_glm_model@summary`.
    fn summary(&self) -> Result<List> {
        let s = self.inner.summary().map_err(to_r)?;
        let col = |g: fn(&ParamSummary) -> f64| s.iter().map(g).collect::<Vec<f64>>();
        Ok(list!(
            parameter = s.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            mean = col(|p| p.mean),
            sd = col(|p| p.sd),
            q05 = col(|p| p.q05),
            q50 = col(|p| p.q50),
            q95 = col(|p| p.q95),
            rhat = col(|p| p.rhat),
            ess_bulk = col(|p| p.ess_bulk),
            ess_tail = col(|p| p.ess_tail)
        ))
    }

    /// PSIS-LOO of the observed cells, outstanding first: `list(estimates,
    /// pointwise, pareto_k, k_threshold, n_outstanding, elpd_outstanding,
    /// elpd_paid)`.
    fn loo(&self) -> Result<List> {
        let l = self.inner.loo().map_err(to_r)?;
        let e = &l.loo.estimate;
        Ok(list!(
            estimates = vec![e.elpd, e.se, e.p, e.ic],
            pointwise = e.pointwise.clone(),
            pareto_k = l.loo.pareto_k,
            k_threshold = l.loo.k_threshold,
            n_outstanding = l.n_outstanding as f64,
            elpd_outstanding = l.elpd_outstanding,
            elpd_paid = l.elpd_paid
        ))
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn paid_table(&self) -> Result<List> {
        Ok(fit_table(self.inner.paid_table().map_err(to_r)?))
    }

    fn outstanding_table(&self) -> Result<List> {
        Ok(fit_table(self.inner.outstanding_table().map_err(to_r)?))
    }

    /// The fit itself, as for any single-segment fit without keys: a
    /// condition on a key is an error.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let one = SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![()],
        };
        pick(&one, keys, values)?;
        Ok(Self {
            inner: self.inner.clone(),
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
    impl BayesOdpFit;
    impl BayesOdpFits;
    impl BayesOneYearFit;
    impl BayesOneYearFits;
    impl MeyersFit;
    impl MeyersFits;
    impl CompartmentalFit;
    impl CompartmentalOneYearFit;
    impl BayesBfFit;
    impl MackBootstrapFit;
    impl OneYearFit;
    impl ClarkFit;
}

/// Which factors a one-year bootstrap simulates: with `bootstrap_ldf` the
/// method's pattern (the Rust default), without it the triangle's own.
fn one_year_pattern(bootstrap_ldf: bool) -> OneYearPattern {
    if bootstrap_ldf {
        OneYearPattern::Method
    } else {
        OneYearPattern::Estimated
    }
}

/// `values` as an array of `N`, or an error naming `name`.
fn fixed<const N: usize>(values: &[f64], name: &str) -> Result<[f64; N]> {
    values
        .try_into()
        .map_err(|_| Error::Other(format!("{name} must have {N} values, got {}", values.len())))
}
