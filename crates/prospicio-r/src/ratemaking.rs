//! Aggregate lane: ratemaking for the R `ratemaking.R` API, premium at the
//! current rate level by the parallelogram method and trend
//! (`docs/design/ratemaking.md`).

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_pricing::ratemaking::{
    Applies, BeforeSchedule, ExpenseCategory, LossRatioExperience, LossRatioYear, Period,
    PeriodKind, Provisions, PurePremiumExperience, PurePremiumYear, RateChange, RateHistory,
    RateLevel, RateSchedule, ScheduleTrend, TrendBasis, TrendFit, TrendItem, TrendKind,
    credibility_weighted, trend_factor, trended_present_rates,
};

use crate::to_r;

/// Year fractions of dates on the half-month grid.
#[extendr]
fn year_fractions(year: &[i32], month: &[i32], day: &[i32]) -> Result<Vec<f64>> {
    if month.len() != year.len() || day.len() != year.len() {
        return Err(Error::Other(
            "year, month and day must have the same length".into(),
        ));
    }
    year.iter()
        .zip(month)
        .zip(day)
        .map(|((&y, &m), &d)| {
            let (m, d) = (
                u8::try_from(m).map_err(|_| Error::Other(format!("no month {m}")))?,
                u8::try_from(d).map_err(|_| Error::Other(format!("no day {d}")))?,
            );
            prospicio_pricing::ratemaking::year_fraction(y, m, d).map_err(to_r)
        })
        .collect()
}

fn period(kind: &str, start: f64, end: f64) -> Result<Period> {
    let kind = PeriodKind::from_name(kind)
        .ok_or_else(|| crate::unknown_name("period", kind, PeriodKind::NAMES))?;
    Period::new(kind, start, end).map_err(to_r)
}

/// `c(average written date, average earned date)` of a period.
#[extendr]
fn period_dates(kind: &str, start: f64, end: f64, term: f64) -> Result<Vec<f64>> {
    let p = period(kind, start, end)?;
    Ok(vec![
        p.average_written_date(term),
        p.average_earned_date(term),
    ])
}

fn applies(values: &[String]) -> Result<Vec<Applies>> {
    values
        .iter()
        .map(|a| {
            Applies::from_name(a).ok_or_else(|| crate::unknown_name("applies", a, Applies::NAMES))
        })
        .collect()
}

/// The parallelogram method for one period. `values` are changes, or
/// cumulative levels when `levels` is true.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rate_level_rust(
    effective: &[f64],
    values: &[f64],
    applies_to: Vec<String>,
    levels: bool,
    kind: &str,
    start: f64,
    end: f64,
    term: f64,
) -> Result<List> {
    if effective.len() != values.len() || applies_to.len() != values.len() {
        return Err(Error::Other(
            "effective, the changes and applies must have the same length".into(),
        ));
    }
    let applies = applies(&applies_to)?;
    let history = if levels {
        RateHistory::from_levels(effective, values, &applies)
    } else {
        let changes: Vec<RateChange> = effective
            .iter()
            .zip(values)
            .zip(&applies)
            .map(|((&effective, &change), &applies)| RateChange {
                effective,
                change,
                applies,
            })
            .collect();
        RateHistory::new(&changes)
    }
    .map_err(to_r)?;
    let level: RateLevel = history
        .rate_level(&period(kind, start, end)?, term)
        .map_err(to_r)?;
    let p = &level.portions;
    Ok(list!(
        written_changes = p
            .iter()
            .map(|x| x.written_changes as f64)
            .collect::<Vec<_>>(),
        in_force_changes = p
            .iter()
            .map(|x| x.in_force_changes as f64)
            .collect::<Vec<_>>(),
        level = p.iter().map(|x| x.level).collect::<Vec<_>>(),
        share = p.iter().map(|x| x.share).collect::<Vec<_>>(),
        average_level = level.average_level,
        current_level = level.current_level,
        on_level_factor = level.on_level_factor()
    ))
}

/// A least-squares trend: `c(intercept, slope, r_squared, annual_change)`.
#[extendr]
fn trend_fit_rust(times: &[f64], values: &[f64], kind: &str) -> Result<Vec<f64>> {
    let kind = match kind {
        "exponential" => TrendKind::Exponential,
        "linear" => TrendKind::Linear,
        other => {
            return Err(Error::Other(format!(
                "kind must be \"exponential\" or \"linear\", got {other}"
            )));
        }
    };
    let fit = match kind {
        TrendKind::Exponential => TrendFit::exponential(times, values),
        TrendKind::Linear => TrendFit::linear(times, values),
    }
    .map_err(to_r)?;
    Ok(vec![
        fit.intercept,
        fit.slope,
        fit.r_squared,
        fit.annual_change(),
    ])
}

/// Trend factors by a calendar-year rate schedule:
/// `list(year, start, target, factor)`. `cutoff` is 0 for complete years.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn schedule_trend_rust(
    years: &[i32],
    trend_years: &[f64],
    trend_rates: &[f64],
    inception: f64,
    basis: &str,
    kind: &str,
    cutoff: i32,
    before: &str,
) -> Result<List> {
    let schedule = RateSchedule::from_columns(trend_years, trend_rates).map_err(to_r)?;
    let basis = match basis {
        "RAD" => TrendBasis::RisksAttaching,
        "LOD" => TrendBasis::LossesOccurring,
        other => {
            return Err(Error::Other(format!(
                "basis must be \"RAD\" or \"LOD\", got {other}"
            )));
        }
    };
    let item = match kind {
        "loss" => TrendItem::Loss,
        "premium" => TrendItem::Premium,
        other => {
            return Err(Error::Other(format!(
                "kind must be \"loss\" or \"premium\", got {other}"
            )));
        }
    };
    let before = match before {
        "latest" => BeforeSchedule::Latest,
        "earliest" => BeforeSchedule::Earliest,
        other => {
            return Err(Error::Other(format!(
                "before must be \"latest\" or \"earliest\", got {other}"
            )));
        }
    };
    let mut trend = ScheduleTrend::new(inception)
        .basis(basis)
        .item(item)
        .before(before);
    if cutoff != 0 {
        trend.cutoff = Some(
            u32::try_from(cutoff)
                .map_err(|_| Error::Other(format!("cutoff must be 1 to 12, got {cutoff}")))?,
        );
    }
    let rows = trend.factors(&schedule, years).map_err(to_r)?;
    Ok(list!(
        year = rows.iter().map(|r| r.year).collect::<Vec<_>>(),
        start = rows.iter().map(|r| r.start).collect::<Vec<_>>(),
        target = rows.iter().map(|r| r.target).collect::<Vec<_>>(),
        factor = rows.iter().map(|r| r.factor).collect::<Vec<_>>()
    ))
}

/// The running premium-weighted loss ratio.
#[extendr]
fn cumulative_loss_ratio_rust(loss_ratios: &[f64], premiums: &[f64]) -> Result<Vec<f64>> {
    prospicio_pricing::ratemaking::cumulative_loss_ratio(loss_ratios, premiums).map_err(to_r)
}

fn same_length(columns: &[&[f64]]) -> Result<usize> {
    let n = columns[0].len();
    if columns.iter().any(|c| c.len() != n) {
        return Err(Error::Other(
            "every column must have one value per year".into(),
        ));
    }
    Ok(n)
}

/// The loss ratio method's experience: per-year projected premium, losses
/// and loss ratio, and the totals.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn loss_ratio_experience_rust(
    earned_premium: &[f64],
    losses: &[f64],
    on_level_factor: &[f64],
    premium_trend: &[f64],
    development: &[f64],
    loss_trend: &[f64],
    lae_factor: &[f64],
) -> Result<List> {
    let n = same_length(&[
        earned_premium,
        losses,
        on_level_factor,
        premium_trend,
        development,
        loss_trend,
        lae_factor,
    ])?;
    let years: Vec<LossRatioYear> = (0..n)
        .map(|i| LossRatioYear {
            earned_premium: earned_premium[i],
            on_level_factor: on_level_factor[i],
            premium_trend: premium_trend[i],
            losses: losses[i],
            development: development[i],
            loss_trend: loss_trend[i],
            lae_factor: lae_factor[i],
        })
        .collect();
    let e = LossRatioExperience::new(&years).map_err(to_r)?;
    Ok(list!(
        projected_premium_by_year = years
            .iter()
            .map(|y| y.projected_premium())
            .collect::<Vec<_>>(),
        projected_losses_by_year = years
            .iter()
            .map(|y| y.projected_losses())
            .collect::<Vec<_>>(),
        loss_ratio_by_year = years.iter().map(|y| y.loss_ratio()).collect::<Vec<_>>(),
        projected_premium = e.projected_premium(),
        projected_losses = e.projected_losses(),
        loss_ratio = e.loss_ratio()
    ))
}

/// The pure premium method's experience: per-year projected losses and
/// pure premium, and the totals.
#[extendr]
fn pure_premium_experience_rust(
    exposure: &[f64],
    losses: &[f64],
    development: &[f64],
    loss_trend: &[f64],
    lae_factor: &[f64],
) -> Result<List> {
    let n = same_length(&[exposure, losses, development, loss_trend, lae_factor])?;
    let years: Vec<PurePremiumYear> = (0..n)
        .map(|i| PurePremiumYear {
            exposure: exposure[i],
            losses: losses[i],
            development: development[i],
            loss_trend: loss_trend[i],
            lae_factor: lae_factor[i],
        })
        .collect();
    let e = PurePremiumExperience::new(&years).map_err(to_r)?;
    Ok(list!(
        projected_losses_by_year = years
            .iter()
            .map(|y| y.projected_losses())
            .collect::<Vec<_>>(),
        pure_premium_by_year = years.iter().map(|y| y.pure_premium()).collect::<Vec<_>>(),
        exposure = e.exposure(),
        projected_losses = e.projected_losses(),
        pure_premium = e.pure_premium()
    ))
}

/// `c(permissible loss ratio, indicated change, indicated rate)` for one
/// set of provisions.
#[extendr]
fn indication_rust(
    variable: f64,
    profit: f64,
    fixed_ratio: f64,
    loss_ratio: f64,
    pure_premium: f64,
    fixed_per_exposure: f64,
) -> Result<Vec<f64>> {
    let p = Provisions::new(variable, profit).map_err(to_r)?;
    Ok(vec![
        p.permissible_loss_ratio(fixed_ratio),
        p.loss_ratio_change(loss_ratio, fixed_ratio),
        p.pure_premium_rate(pure_premium, fixed_per_exposure),
    ])
}

/// `c(fixed, variable)` expense ratios of the premium-based projection.
#[extendr]
fn premium_based_expenses_rust(ratio: &[f64], fixed_share: &[f64]) -> Result<Vec<f64>> {
    same_length(&[ratio, fixed_share])?;
    let categories: Vec<ExpenseCategory> = ratio
        .iter()
        .zip(fixed_share)
        .map(|(&ratio, &fixed_share)| ExpenseCategory { ratio, fixed_share })
        .collect();
    let (fixed, variable) =
        prospicio_pricing::ratemaking::premium_based_expenses(&categories).map_err(to_r)?;
    Ok(vec![fixed, variable])
}

/// `f` over its arguments recycled to the longest, as R's arithmetic
/// recycles; a zero-length argument gives a zero-length result.
fn recycled<const N: usize>(args: [&[f64]; N], f: impl Fn([f64; N]) -> f64) -> Vec<f64> {
    if args.iter().any(|a| a.is_empty()) {
        return Vec::new();
    }
    let n = args.iter().map(|a| a.len()).max().unwrap_or(0);
    (0..n)
        .map(|i| f(std::array::from_fn(|j| args[j][i % args[j].len()])))
        .collect()
}

/// [`trend_factor`] element by element, recycled.
#[extendr]
fn trend_factor_rust(rate: &[f64], years: &[f64]) -> Vec<f64> {
    recycled([rate, years], |[r, y]| trend_factor(r, y))
}

/// [`credibility_weighted`] element by element, recycled.
#[extendr]
fn credibility_weighted_rust(
    estimate: &[f64],
    credibility: &[f64],
    complement: &[f64],
) -> Vec<f64> {
    recycled([estimate, credibility, complement], |[e, z, c]| {
        credibility_weighted(e, z, c)
    })
}

/// [`trended_present_rates`] element by element, recycled.
#[extendr]
fn trended_present_rates_rust(
    prior_indicated: &[f64],
    prior_implemented: &[f64],
    trend: &[f64],
) -> Vec<f64> {
    recycled([prior_indicated, prior_implemented, trend], |[i, m, t]| {
        trended_present_rates(i, m, t)
    })
}

extendr_module! {
    mod ratemaking;
    fn trend_factor_rust;
    fn credibility_weighted_rust;
    fn trended_present_rates_rust;
    fn loss_ratio_experience_rust;
    fn pure_premium_experience_rust;
    fn indication_rust;
    fn premium_based_expenses_rust;
    fn year_fractions;
    fn period_dates;
    fn rate_level_rust;
    fn trend_fit_rust;
    fn schedule_trend_rust;
    fn cumulative_loss_ratio_rust;
}
