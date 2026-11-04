//! Recoveries, ALAE and ULAE (issue #558, `docs/design/reserving-v02.md`,
//! decision 37): `ratio_development()` and `ulae_estimate()`, over
//! `prospicio_reserving::expense`.
//!
//! Fits come back as plain numbers, the ratio triangle flattened row by row,
//! and the R side (`reserving_expense.R`) holds them in S7 objects.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_core::Lag;
use prospicio_reserving::{Apriori, RatioDevelopment, RatioFactors, Ulae, UlaeInputs, UlaeWeights};

use crate::reserving::{Triangle, pattern, to_r};
use crate::whole;

/// `ratio_development()`. `selected` is a numeric vector named by origin, or
/// empty; `additive_ages` and `additive_values` are the selected differences.
/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn ratio_development_rust(
    triangle: Robj,
    numerator: &str,
    denominator: &str,
    ultimate_claims: Vec<f64>,
    average: &str,
    sigma: &str,
    excl: Robj,
    tail: Robj,
    ldf: Robj,
    selected: Robj,
    additive: bool,
    additive_ages: Vec<f64>,
    additive_values: Vec<f64>,
    additive_tail: f64,
) -> Result<List> {
    let tri = &<&Triangle>::try_from(&triangle)
        .map_err(|_| Error::Other("triangle must be a triangle".into()))?
        .inner;
    let cl = pattern(average, sigma, &excl, tail, &ldf, tri.origin_grain())?;
    let selected = match selected.as_real_vector() {
        Some(values) if !values.is_empty() => {
            let names: Vec<String> = selected
                .names()
                .ok_or_else(|| Error::Other("selected must be named by origin".into()))?
                .map(str::to_string)
                .collect();
            match Apriori::from_labels(names.iter().zip(values)).map_err(to_r)? {
                Apriori::ByOrigin(pairs) => pairs,
                _ => unreachable!("labels give values by origin"),
            }
        }
        _ => Vec::new(),
    };
    let factors = if additive {
        RatioFactors::Additive {
            selected: additive_ages
                .iter()
                .zip(additive_values)
                .map(|(&age, value)| {
                    let age = whole(age, "an age of additive_selected")?;
                    let age = Lag::try_from(age).map_err(|_| {
                        Error::Other("an age of additive_selected is too large".into())
                    })?;
                    Ok((age, value))
                })
                .collect::<Result<Vec<_>>>()?,
            tail: additive_tail,
        }
    } else {
        RatioFactors::Multiplicative
    };
    let fit = RatioDevelopment {
        development: cl.development,
        tail: cl.tail,
        factors,
        selected,
    }
    .fit(tri, numerator, denominator, &ultimate_claims)
    .map_err(to_r)?;
    let ratios: Vec<f64> = fit
        .ratios
        .iter()
        .flatten()
        .map(|r| r.unwrap_or(f64::NAN))
        .collect();
    let ages: Vec<f64> = fit.ages.iter().map(|&a| f64::from(a)).collect();
    let unpaid = fit.unpaid();
    Ok(list!(
        origins = fit
            .origins
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ages = ages,
        ratios = ratios,
        averages = fit.averages,
        factors = fit.factors,
        cdf = fit.cdf,
        latest_ratio = fit.latest_ratio,
        projected_ratio = fit.projected_ratio,
        selected_ratio = fit.selected_ratio,
        ultimate_claims = fit.ultimate_claims,
        latest = fit.latest,
        ultimate = fit.ultimate,
        unpaid = unpaid
    ))
}

/// `ulae_estimate()`. `ratio` is NaN for none; `pure_ibnr` is NULL for none,
/// and the other optional vectors are empty.
/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn ulae_rust(
    method: &str,
    paid_ulae: Vec<f64>,
    paid: Vec<f64>,
    reported: Vec<f64>,
    closed: Vec<f64>,
    case_outstanding: Vec<f64>,
    ibnr: Vec<f64>,
    pure_ibnr: Nullable<Vec<f64>>,
    unclosed: Vec<f64>,
    ultimate: Vec<f64>,
    ratio: f64,
    opening: f64,
    weights: Vec<f64>,
) -> Result<List> {
    let ratio = if ratio.is_nan() { None } else { Some(ratio) };
    let model = match method {
        "classical" => Ulae::Classical { ratio, opening },
        "kittel" => Ulae::Kittel { ratio },
        "conger_nolibos" => {
            let [opening, maintaining, closing] = weights[..] else {
                return Err(Error::Other("weights must be three shares".into()));
            };
            Ulae::CongerNolibos {
                weights: UlaeWeights {
                    opening,
                    maintaining,
                    closing,
                },
                ratio,
            }
        }
        other => return Err(Error::Other(format!("unknown method {other:?}"))),
    };
    let pure = match pure_ibnr {
        Nullable::NotNull(p) => Some(p),
        Nullable::Null => None,
    };
    let fit = model
        .estimate(&UlaeInputs {
            paid_ulae: &paid_ulae,
            paid: &paid,
            reported: &reported,
            closed: &closed,
            case_outstanding: &case_outstanding,
            ibnr: &ibnr,
            pure_ibnr: pure.as_deref(),
            unclosed: &unclosed,
            ultimate: &ultimate,
        })
        .map_err(to_r)?;
    let total = fit.total();
    let na = |v: Option<f64>| v.unwrap_or(f64::NAN);
    Ok(list!(
        basis = fit.basis,
        ratios = fit.ratios,
        total_ratio = fit.total_ratio,
        ratio = fit.ratio,
        unpaid = fit.unpaid,
        total = total,
        expected_claims = na(fit.expected_claims),
        bornhuetter_ferguson = na(fit.bornhuetter_ferguson),
        development = na(fit.development)
    ))
}

extendr_module! {
    mod reserving_expense;
    fn ratio_development_rust;
    fn ulae_rust;
}
