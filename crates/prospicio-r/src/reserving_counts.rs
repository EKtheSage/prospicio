//! Methods on claim counts (issues #554 and #555,
//! `docs/design/reserving-v02.md`, decisions 35 and 36): `frequency_severity()`,
//! `disposal_method()`, `case_outstanding()` and the Berquist-Sherman
//! adjustments, over `prospicio_reserving::{frequency_severity,
//! berquist_sherman}`.
//!
//! The R side encodes each choice as a named list with a `kind`
//! (`reserving_counts.R`); fits come back as plain numbers, the cell tables
//! flattened row by row, and the R side holds them in S7 objects.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_core::{Lag, Period};
use prospicio_reserving::{
    BerquistSherman, CaseAdequacy, CaseOutstanding, CaseOutstandingFit, CaseTrend, ChainLadder,
    CountProjection, Disposal, DisposalFit, FrequencySeverity, FrequencySeverityFit, Interpolation,
    OpenCounts, Selection, SettlementRate, SeverityProjection, TrendFit, Trended,
    Triangle as TriangleInner, UltimateCounts,
};

use crate::reserving::{Triangle, pattern, to_r};
use crate::whole;

fn triangle_of(triangle: &Robj) -> Result<&Triangle> {
    <&Triangle>::try_from(triangle).map_err(|_| Error::Other("triangle must be a triangle".into()))
}

fn field(list: &Robj, name: &str) -> Result<Robj> {
    list.dollar(name)
}

fn text(list: &Robj, name: &str) -> Result<String> {
    field(list, name)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| Error::Other(format!("{name} must be text")))
}

fn numbers(list: &Robj, name: &str) -> Result<Vec<f64>> {
    let v = field(list, name)?;
    if v.is_null() {
        return Ok(Vec::new());
    }
    v.as_real_vector()
        .ok_or_else(|| Error::Other(format!("{name} must be numeric")))
}

fn number(list: &Robj, name: &str) -> Result<f64> {
    field(list, name)?
        .as_real()
        .ok_or_else(|| Error::Other(format!("{name} must be a number")))
}

fn lag(x: f64, what: &str) -> Result<Lag> {
    whole(x, what)
        .and_then(|a| Lag::try_from(a).map_err(|_| Error::Other(format!("{what} is too large"))))
}

fn count(x: f64, what: &str) -> Result<usize> {
    whole(x, what).map(|a| a as usize)
}

fn optional_lag(x: f64, what: &str) -> Result<Option<Lag>> {
    if x.is_nan() {
        Ok(None)
    } else {
        lag(x, what).map(Some)
    }
}

fn selection(x: &Robj) -> Result<Selection> {
    match text(x, "kind")?.as_str() {
        "average" => {
            let n = number(x, "n_periods")?;
            Ok(Selection::Average {
                n_periods: if n.is_nan() {
                    None
                } else {
                    Some(count(n, "n_periods")?)
                },
                drop_high: count(number(x, "drop_high")?, "drop_high")?,
                drop_low: count(number(x, "drop_low")?, "drop_low")?,
            })
        }
        "latest_diagonal" => Ok(Selection::LatestDiagonal),
        "given" => {
            let ages = numbers(x, "ages")?;
            let values = numbers(x, "values")?;
            let pairs = ages
                .iter()
                .zip(values)
                .map(|(&a, v)| Ok((lag(a, "an age")?, v)))
                .collect::<Result<Vec<_>>>()?;
            Ok(Selection::Given(pairs))
        }
        other => Err(Error::Other(format!("unknown selection {other:?}"))),
    }
}

fn method(x: &Robj, tri: &TriangleInner) -> Result<ChainLadder> {
    pattern(
        &text(x, "average")?,
        &text(x, "sigma")?,
        &field(x, "excl")?,
        field(x, "tail")?,
        &field(x, "ldf")?,
        tri.origin_grain(),
    )
}

fn ultimate_counts(x: &Robj, tri: &TriangleInner) -> Result<UltimateCounts> {
    match text(x, "kind")?.as_str() {
        "given" => Ok(UltimateCounts::Given(numbers(x, "values")?)),
        "development" => Ok(UltimateCounts::Development {
            column: text(x, "column")?,
            method: method(x, tri)?,
        }),
        "mean" => {
            let parts = field(x, "parts")?;
            let parts = parts
                .as_list()
                .ok_or_else(|| Error::Other("parts must be a list".into()))?;
            let parts = parts
                .values()
                .map(|p| ultimate_counts(&p, tri))
                .collect::<Result<Vec<_>>>()?;
            Ok(UltimateCounts::Mean(parts))
        }
        other => Err(Error::Other(format!("unknown ultimate counts {other:?}"))),
    }
}

fn origin_labelled(label: &str, tri: &TriangleInner, what: &str) -> Result<Period> {
    tri.origins()
        .iter()
        .find(|p| p.to_string() == label)
        .copied()
        .ok_or_else(|| Error::Other(format!("{what}: {label} is not an origin of the triangle")))
}

fn trended(x: &Robj, tri: &TriangleInner) -> Result<Trended> {
    Ok(Trended {
        selected: number(x, "selected")?,
        base: origin_labelled(&text(x, "base")?, tri, "trended()'s base")?,
        trend: number(x, "trend")?,
        factors: numbers(x, "factors")?,
    })
}

/// Rows flattened row by row, a missing cell as NaN.
fn flat(rows: &[Vec<Option<f64>>]) -> Vec<f64> {
    rows.iter()
        .flat_map(|r| r.iter().map(|c| c.unwrap_or(f64::NAN)))
        .collect()
}

fn labels(origins: &[Period]) -> Vec<String> {
    origins.iter().map(ToString::to_string).collect()
}

fn ages(ages: &[Lag]) -> Vec<f64> {
    ages.iter().map(|&a| f64::from(a)).collect()
}

fn frequency_severity_list(fit: &FrequencySeverityFit) -> List {
    list!(
        origins = labels(&fit.origins),
        latest = fit.latest.clone(),
        latest_severity = fit.latest_severity.clone(),
        ultimate_counts = fit.ultimate_counts.clone(),
        ultimate_severity = fit.ultimate_severity.clone(),
        ultimate = fit.ultimate.clone(),
        reserve = fit.reserves()
    )
}

/// `frequency_severity()`.
#[extendr]
fn frequency_severity_rust(
    triangle: Robj,
    amounts: &str,
    counts: &str,
    ultimate: Robj,
    severity: Robj,
) -> Result<List> {
    let tri = &triangle_of(&triangle)?.inner;
    let ultimate_counts = match text(&ultimate, "kind")?.as_str() {
        "frequency" => CountProjection::Frequency {
            exposure: text(&ultimate, "exposure")?,
            exposure_factors: numbers(&ultimate, "exposure_factors")?,
            frequency: trended(&field(&ultimate, "frequency")?, tri)?,
        },
        _ => CountProjection::Counts(ultimate_counts(&ultimate, tri)?),
    };
    let severity = match text(&severity, "kind")?.as_str() {
        "trended" => SeverityProjection::Trended(trended(&severity, tri)?),
        _ => SeverityProjection::Development(method(&severity, tri)?),
    };
    let fit = FrequencySeverity {
        amounts: amounts.into(),
        counts: counts.into(),
        ultimate_counts,
        severity,
    }
    .fit(tri)
    .map_err(to_r)?;
    Ok(frequency_severity_list(&fit))
}

fn disposal_list(fit: &DisposalFit) -> List {
    list!(
        origins = labels(&fit.origins),
        ages = ages(&fit.ages),
        ultimate_counts = fit.ultimate_counts.clone(),
        disposal_rates = flat(&fit.disposal_rates),
        selected_rates = fit.selected_rates.clone(),
        severities = flat(&fit.severities),
        trended_severities = flat(&fit.trended_severities),
        selected_severities = fit.selected_severities.clone(),
        incremental_counts = flat(&fit.incremental_counts),
        projected_severities = flat(&fit.projected_severities),
        incremental_paid = flat(&fit.incremental_paid),
        latest = fit.latest.clone(),
        ultimate = fit.ultimate.clone(),
        reserve = fit.reserves()
    )
}

/// `disposal_method()`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn disposal_rust(
    triangle: Robj,
    closed_counts: &str,
    paid: &str,
    ultimate: Robj,
    disposal_rates: Robj,
    severities: Robj,
    tail_severity_from: f64,
    trend: f64,
    base: Robj,
    factors: Vec<f64>,
) -> Result<List> {
    let tri = &triangle_of(&triangle)?.inner;
    let base = match base.as_str() {
        Some(label) => Some(origin_labelled(label, tri, "base")?),
        None => None,
    };
    let fit = Disposal {
        closed_counts: closed_counts.into(),
        paid: paid.into(),
        ultimate_counts: ultimate_counts(&ultimate, tri)?,
        disposal_rates: selection(&disposal_rates)?,
        severities: selection(&severities)?,
        tail_severity_from: optional_lag(tail_severity_from, "tail_severity_from")?,
        trend,
        base,
        factors,
    }
    .fit(tri)
    .map_err(to_r)?;
    Ok(disposal_list(&fit))
}

fn case_outstanding_list(fit: &CaseOutstandingFit) -> List {
    list!(
        origins = labels(&fit.origins),
        ages = ages(&fit.ages),
        paid_on_case = flat(&fit.paid_on_case),
        case_on_case = flat(&fit.case_on_case),
        selected_paid_on_case = fit.selected_paid_on_case.clone(),
        selected_case_on_case = fit.selected_case_on_case.clone(),
        case_outstanding = flat(&fit.case_outstanding),
        incremental_paid = flat(&fit.incremental_paid),
        latest_paid = fit.latest_paid.clone(),
        latest_incurred = fit.latest_incurred.clone(),
        ultimate = fit.ultimate.clone(),
        ibnr = fit.ibnr(),
        reserve = fit.reserves()
    )
}

/// `case_outstanding()`.
#[extendr]
fn case_outstanding_rust(
    triangle: Robj,
    paid: &str,
    incurred: &str,
    paid_on_case: Robj,
    case_on_case: Robj,
    paid_to_ultimate: f64,
) -> Result<List> {
    let tri = &triangle_of(&triangle)?.inner;
    let fit = CaseOutstanding {
        paid: paid.into(),
        incurred: incurred.into(),
        paid_on_case: selection(&paid_on_case)?,
        case_on_case: selection(&case_on_case)?,
        paid_to_ultimate,
    }
    .fit(tri)
    .map_err(to_r)?;
    Ok(case_outstanding_list(&fit))
}

fn case_adequacy(
    incurred: &str,
    paid: &str,
    open_counts: &Robj,
    trend: &Robj,
    max_age: f64,
) -> Result<CaseAdequacy> {
    let open_counts = match text(open_counts, "kind")?.as_str() {
        "column" => OpenCounts::Column(text(open_counts, "column")?),
        _ => OpenCounts::ReportedLessClosed {
            reported: text(open_counts, "reported")?,
            closed: text(open_counts, "closed")?,
        },
    };
    let trend = match text(trend, "kind")?.as_str() {
        "given" => CaseTrend::Given(number(trend, "rate")?),
        "average_case" => CaseTrend::AverageCase,
        _ => CaseTrend::Fitted {
            amount: text(trend, "amount")?,
            counts: text(trend, "counts")?,
        },
    };
    Ok(CaseAdequacy {
        incurred: incurred.into(),
        paid: paid.into(),
        open_counts,
        trend,
        max_age: optional_lag(max_age, "max_age")?,
    })
}

fn settlement(
    tri: &TriangleInner,
    paid: &str,
    closed_counts: &str,
    ultimate: &Robj,
    disposal_rates: &Robj,
    interpolation: &str,
) -> Result<SettlementRate> {
    Ok(SettlementRate {
        paid: paid.into(),
        closed_counts: closed_counts.into(),
        ultimate_counts: ultimate_counts(ultimate, tri)?,
        disposal_rates: selection(disposal_rates)?,
        interpolation: match interpolation {
            "linear" => Interpolation::Linear,
            _ => Interpolation::Exponential,
        },
    })
}

/// `case_adequacy_adjust()`.
#[extendr]
fn case_adequacy_rust(
    triangle: Robj,
    incurred: &str,
    paid: &str,
    open_counts: Robj,
    trend: Robj,
    max_age: f64,
) -> Result<Triangle> {
    let tri = &triangle_of(&triangle)?.inner;
    let inner = case_adequacy(incurred, paid, &open_counts, &trend, max_age)?
        .adjust(tri)
        .map_err(to_r)?;
    Ok(Triangle { inner })
}

/// `average_case()`.
#[extendr]
fn average_case_rust(
    triangle: Robj,
    incurred: &str,
    paid: &str,
    open_counts: Robj,
    trend: Robj,
    max_age: f64,
) -> Result<Triangle> {
    let tri = &triangle_of(&triangle)?.inner;
    let inner = case_adequacy(incurred, paid, &open_counts, &trend, max_age)?
        .average_case(tri)
        .map_err(to_r)?;
    Ok(Triangle { inner })
}

/// `settlement_rate_adjust()`.
#[extendr]
fn settlement_rate_rust(
    triangle: Robj,
    paid: &str,
    closed_counts: &str,
    ultimate: Robj,
    disposal_rates: Robj,
    interpolation: &str,
) -> Result<Triangle> {
    let tri = &triangle_of(&triangle)?.inner;
    let inner = settlement(
        tri,
        paid,
        closed_counts,
        &ultimate,
        &disposal_rates,
        interpolation,
    )?
    .adjust(tri)
    .map_err(to_r)?;
    Ok(Triangle { inner })
}

/// `berquist_sherman_adjust()`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn berquist_sherman_rust(
    triangle: Robj,
    incurred: &str,
    paid: &str,
    open_counts: Robj,
    closed_counts: &str,
    trend: Robj,
    ultimate: Robj,
    max_age: f64,
    disposal_rates: Robj,
    interpolation: &str,
) -> Result<Triangle> {
    let tri = &triangle_of(&triangle)?.inner;
    let inner = BerquistSherman {
        case_adequacy: case_adequacy(incurred, paid, &open_counts, &trend, max_age)?,
        settlement_rate: settlement(
            tri,
            paid,
            closed_counts,
            &ultimate,
            &disposal_rates,
            interpolation,
        )?,
    }
    .adjust(tri)
    .map_err(to_r)?;
    Ok(Triangle { inner })
}

fn trend_values(fit: Option<TrendFit>) -> [f64; 3] {
    match fit {
        Some(f) => [f.rate, f.r_squared, f.n as f64],
        None => [f64::NAN; 3],
    }
}

/// `exponential_trend()`: rate, r squared and n, NaN without a fit.
#[extendr]
fn exponential_trend_rust(x: Vec<f64>, y: Vec<f64>) -> Result<Vec<f64>> {
    if x.len() != y.len() {
        return Err(Error::Other("x and y need the same length".into()));
    }
    let points: Vec<(f64, f64)> = x.into_iter().zip(y).collect();
    Ok(trend_values(prospicio_reserving::exponential_trend(&points)).to_vec())
}

/// `trend_by_age()`.
#[extendr]
fn trend_by_age_rust(triangle: Robj, column: &str) -> Result<List> {
    let tri = &triangle_of(&triangle)?.inner;
    let fits = prospicio_reserving::trend_by_age(tri, column).map_err(to_r)?;
    let age: Vec<f64> = fits.iter().map(|(a, _)| f64::from(*a)).collect();
    let values: Vec<[f64; 3]> = fits.iter().map(|(_, f)| trend_values(*f)).collect();
    Ok(list!(
        age = age,
        rate = values.iter().map(|v| v[0]).collect::<Vec<_>>(),
        r_squared = values.iter().map(|v| v[1]).collect::<Vec<_>>(),
        n = values.iter().map(|v| v[2]).collect::<Vec<_>>()
    ))
}

/// `pooled_trend()`: rate, r squared and n, NaN without a fit.
#[extendr]
fn pooled_trend_rust(triangle: Robj, column: &str) -> Result<Vec<f64>> {
    let tri = &triangle_of(&triangle)?.inner;
    let fit = prospicio_reserving::pooled_trend(tri, column).map_err(to_r)?;
    Ok(trend_values(fit).to_vec())
}

/// `ratio_column()`.
#[extendr]
fn ratio_column_rust(
    triangle: Robj,
    numerator: &str,
    denominator: &str,
    name: &str,
) -> Result<Triangle> {
    let tri = &triangle_of(&triangle)?.inner;
    let inner =
        prospicio_reserving::ratio_column(tri, numerator, denominator, name).map_err(to_r)?;
    Ok(Triangle { inner })
}

/// `case_development_factor()`.
#[extendr]
fn case_development_factor_rust(reported_cdf: f64, paid_cdf: f64) -> Result<f64> {
    prospicio_reserving::case_development_factor(reported_cdf, paid_cdf).map_err(to_r)
}

extendr_module! {
    mod reserving_counts;
    fn frequency_severity_rust;
    fn disposal_rust;
    fn case_outstanding_rust;
    fn case_adequacy_rust;
    fn average_case_rust;
    fn settlement_rate_rust;
    fn berquist_sherman_rust;
    fn exponential_trend_rust;
    fn trend_by_age_rust;
    fn pooled_trend_rust;
    fn ratio_column_rust;
    fn case_development_factor_rust;
}
