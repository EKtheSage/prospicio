//! `discounted_reserves()`, `risk_adjustment()` and `discount_curve()`
//! (issue #553, `docs/design/reserving-ifrs17.md`, PR 2): a fit's unpaid
//! claims discounted simulation by simulation and the IFRS 17 risk
//! adjustment on them, over `prospicio_reserving::discount`.
//!
//! The R side holds the curve as plain numbers in an S7 object and passes
//! its maturities and rates; the fit comes as its external pointer, which
//! [`discounted_reserves_rust`] matches against each supported fit.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_reserving::discount::{
    Capital, CapitalCharge, DiscountCurve, Discountable, DiscountedReserves as Inner, Discounting,
    PaymentBasis, PaymentTiming, RiskAdjustment, RiskAdjustmentResult, SegmentRiskAdjustment,
    discounted_reserves,
};

use crate::distributions::PredictiveDistribution;
use crate::reserving::{
    BayesBfFit, BayesBfNormalisedFit, BayesOdpFit, BayesOdpFits, CompartmentalFit,
    CompartmentalFits, HierarchicalBfFit, MackBootstrapFit, MackFit, MeyersFit, MeyersFits,
    OdpBootstrapFit, to_r,
};

const TIMINGS: [(&str, PaymentTiming); 2] = [
    ("mid", PaymentTiming::MidPeriod),
    ("end", PaymentTiming::EndOfPeriod),
];

const CHARGES: [(&str, CapitalCharge); 2] = [
    ("end", CapitalCharge::EndOfYear),
    ("start", CapitalCharge::StartOfYear),
];

fn named<T: Copy>(arg: &str, got: &str, names: &[(&str, T)]) -> Result<T> {
    names
        .iter()
        .find(|(n, _)| *n == got)
        .map(|(_, v)| *v)
        .ok_or_else(|| crate::unknown_name(arg, got, names))
}

fn curve(maturities: &[f64], rates: &[f64]) -> Result<DiscountCurve> {
    DiscountCurve::spot(maturities, rates).map_err(to_r)
}

/// The discount factor of the curve at each of `t` years; fails for a curve
/// that `DiscountCurve::spot` refuses, so `discount_curve()` checks with it.
#[extendr]
fn discount_factor_rust(maturities: Vec<f64>, rates: Vec<f64>, t: Vec<f64>) -> Result<Vec<f64>> {
    let c = curve(&maturities, &rates)?;
    Ok(t.iter().map(|&t| c.factor(t)).collect())
}

/// The spot rate of the curve at each of `t` years.
#[extendr]
fn discount_rate_rust(maturities: Vec<f64>, rates: Vec<f64>, t: Vec<f64>) -> Result<Vec<f64>> {
    let c = curve(&maturities, &rates)?;
    Ok(t.iter().map(|&t| c.rate(t)).collect())
}

fn run<F: Discountable + ?Sized>(
    fit: &F,
    discounting: &Discounting,
    basis: &PaymentBasis,
) -> Result<DiscountedReserves> {
    Ok(DiscountedReserves {
        inner: discounted_reserves(fit, discounting, basis).map_err(to_r)?,
    })
}

/// A fit's unpaid claims discounted. `fit` is the fit's external pointer;
/// `basis` is "paid" or "incurred" (with `paid_pattern` and `latest_paid`);
/// `timing` is "mid" or "end".
#[extendr]
fn discounted_reserves_rust(
    fit: Robj,
    basis: &str,
    maturities: Vec<f64>,
    rates: Vec<f64>,
    timing: &str,
    paid_pattern: Nullable<Vec<f64>>,
    latest_paid: Nullable<Vec<f64>>,
) -> Result<DiscountedReserves> {
    let discounting = Discounting {
        curve: curve(&maturities, &rates)?,
        timing: named("timing", timing, &TIMINGS)?,
    };
    let basis = match (basis, paid_pattern, latest_paid) {
        ("paid", Nullable::Null, Nullable::Null) => PaymentBasis::Paid,
        ("paid", _, _) => {
            return Err(Error::Other(
                "paid_pattern and latest_paid are for basis = \"incurred\"; a fit on paid \
                 claims is paid along its own patterns"
                    .into(),
            ));
        }
        ("incurred", Nullable::NotNull(paid_pattern), Nullable::NotNull(latest_paid)) => {
            PaymentBasis::Incurred {
                paid_pattern,
                latest_paid,
            }
        }
        ("incurred", _, _) => {
            return Err(Error::Other(
                "basis = \"incurred\" needs paid_pattern (the cumulative paid share at each of \
                 the fit's pattern ages) and latest_paid (one per component of the reserves)"
                    .into(),
            ));
        }
        (other, _, _) => {
            return Err(Error::Other(format!(
                "basis must be \"paid\" or \"incurred\", got {other:?}"
            )));
        }
    };
    let d = &discounting;
    let b = &basis;
    if let Ok(f) = <&OdpBootstrapFit>::try_from(&fit) {
        return run(&f.inner, d, b);
    }
    if let Ok(f) = <&MackBootstrapFit>::try_from(&fit) {
        return run(&f.inner, d, b);
    }
    if let Ok(f) = <&MackFit>::try_from(&fit) {
        let (_, first) = f
            .inner
            .iter()
            .next()
            .ok_or_else(|| Error::Other("the Mack fit has no segment".into()))?;
        return run(first, d, b);
    }
    if let Ok(f) = <&BayesOdpFit>::try_from(&fit) {
        return run(&f.inner, d, b);
    }
    if let Ok(f) = <&BayesOdpFits>::try_from(&fit) {
        return run(&f.inner, d, b);
    }
    if let Ok(f) = <&BayesBfFit>::try_from(&fit) {
        return run(&f.inner, d, b);
    }
    if let Ok(f) = <&BayesBfNormalisedFit>::try_from(&fit) {
        return run(&f.inner, d, b);
    }
    if let Ok(f) = <&HierarchicalBfFit>::try_from(&fit) {
        return run(&f.inner, d, b);
    }
    if let Ok(f) = <&MeyersFit>::try_from(&fit) {
        return run(&f.inner, d, b);
    }
    if let Ok(f) = <&MeyersFits>::try_from(&fit) {
        return run(&f.inner, d, b);
    }
    if let Ok(f) = <&CompartmentalFit>::try_from(&fit) {
        return run(&f.inner, d, b);
    }
    if let Ok(f) = <&CompartmentalFits>::try_from(&fit) {
        return run(&f.inner, d, b);
    }
    Err(Error::Other(
        "fit must be a stochastic reserving fit: odp_bootstrap(), mack_bootstrap(), \
         bayes_odp(), bayes_bf(), bayes_bf_normalised(), meyers(), compartmental() or their \
         fits over segments"
            .into(),
    ))
}

fn optional(x: Option<f64>) -> Robj {
    x.map_or_else(|| Robj::from(()), Robj::from)
}

fn result_list(r: &RiskAdjustmentResult) -> Result<List> {
    let segments: Vec<Robj> = r
        .segments
        .iter()
        .map(|s: &SegmentRiskAdjustment| {
            List::from_names_and_values(
                [
                    "label",
                    "amount",
                    "best_estimate",
                    "implied_confidence",
                    "capital",
                ],
                [
                    Robj::from(s.label.clone()),
                    Robj::from(s.result.amount),
                    Robj::from(s.result.best_estimate),
                    Robj::from(s.result.implied_confidence),
                    optional(s.result.capital),
                ],
            )
            .map(Robj::from)
        })
        .collect::<Result<_>>()?;
    List::from_names_and_values(
        [
            "amount",
            "best_estimate",
            "implied_confidence",
            "capital",
            "segments",
        ],
        [
            Robj::from(r.amount),
            Robj::from(r.best_estimate),
            Robj::from(r.implied_confidence),
            optional(r.capital),
            Robj::from(List::from_values(segments)),
        ],
    )
}

/// The result of `discounted_reserves()`.
#[extendr]
pub(crate) struct DiscountedReserves {
    inner: Inner,
}

#[extendr]
impl DiscountedReserves {
    fn unpaid(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.unpaid.clone(),
        }
    }

    fn discounted(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: self.inner.discounted.clone(),
        }
    }

    fn best_estimate(&self) -> f64 {
        self.inner.best_estimate
    }

    fn runoff(&self) -> Vec<f64> {
        self.inner.runoff.clone()
    }

    fn discount_factors(&self) -> Vec<f64> {
        self.inner.discount_factors.clone()
    }

    fn maturities(&self) -> Vec<f64> {
        self.inner.discounting().curve.maturities().to_vec()
    }

    fn rates(&self) -> Vec<f64> {
        self.inner.discounting().curve.rates().to_vec()
    }

    fn timing(&self) -> &'static str {
        match self.inner.discounting().timing {
            PaymentTiming::MidPeriod => "mid",
            PaymentTiming::EndOfPeriod => "end",
        }
    }

    /// One character vector per segment; an empty list for one segment.
    fn segment_labels(&self) -> List {
        List::from_values(self.inner.segment_labels())
    }

    /// Segment `i`'s run-off, `i` counted from 1.
    fn segment_runoff(&self, i: f64) -> Result<Vec<f64>> {
        let k = crate::whole(i, "i")? as usize;
        k.checked_sub(1)
            .and_then(|k| self.inner.segment_runoff(k))
            .map(<[f64]>::to_vec)
            .ok_or_else(|| Error::Other(format!("no segment {i}")))
    }

    /// `method` is "confidence" or "tvar" (with `p`) or "cost_of_capital"
    /// (with `rate`, `capital_kind` "var", "tvar" or "amount", `capital` and
    /// `charge` "end" or "start").
    fn risk_adjustment(
        &self,
        method: &str,
        p: Nullable<f64>,
        rate: Nullable<f64>,
        capital_kind: Nullable<String>,
        capital: Nullable<f64>,
        charge: &str,
    ) -> Result<List> {
        let charge = named("charge", charge, &CHARGES)?;
        let p = p.into_option();
        let method = match method {
            "confidence" | "tvar" => {
                let p = p.ok_or_else(|| Error::Other(format!("method = {method:?} needs p")))?;
                if method == "confidence" {
                    RiskAdjustment::ConfidenceLevel(p)
                } else {
                    RiskAdjustment::Tvar(p)
                }
            }
            "cost_of_capital" => {
                let rate = rate.into_option().ok_or_else(|| {
                    Error::Other("method = \"cost_of_capital\" needs rate".into())
                })?;
                let k = capital.into_option().ok_or_else(|| {
                    Error::Other(
                        "method = \"cost_of_capital\" needs capital: c(var = q), c(tvar = q) or \
                         an amount"
                            .into(),
                    )
                })?;
                let capital = match capital_kind.into_option().as_deref() {
                    Some("var") => Capital::Var(k),
                    Some("tvar") => Capital::Tvar(k),
                    Some("amount") | None => Capital::Amount(k),
                    Some(other) => {
                        return Err(Error::Other(format!(
                            "capital must be named \"var\", \"tvar\" or \"amount\", got {other:?}"
                        )));
                    }
                };
                RiskAdjustment::CostOfCapital {
                    rate,
                    capital,
                    charge,
                }
            }
            other => {
                return Err(Error::Other(format!(
                    "method must be \"confidence\", \"tvar\" or \"cost_of_capital\", got {other:?}"
                )));
            }
        };
        result_list(&self.inner.risk_adjustment(&method).map_err(to_r)?)
    }
}

extendr_module! {
    mod reserving_discount;
    fn discount_factor_rust;
    fn discount_rate_rust;
    fn discounted_reserves_rust;
    impl DiscountedReserves;
}
