//! A collateralised deal's monthly cash flows (Aggregate lane) over
//! `prospicio_pricing::cashflow` (`docs/design/time-value.md`, decisions
//! 13 to 20). `R/cashflow.R` builds the arguments.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_pricing::cashflow::{
    Adjustment, AdjustmentBase, Base, Buffer, CashflowTerms, Dates, FeeBasis, Funding, FundingRule,
    FundingSource, Interest, InterestBasis, MonthlyPatterns, ObligationReference, Operating,
    PerformanceFee, PremiumFlow, Requirement, Term, Timing,
};
use prospicio_pricing::returns::TieredFee;

use crate::{to_r, whole};

fn bad(what: &str, value: &str, choices: &str) -> Error {
    Error::Other(format!("unknown {what} {value:?}; use {choices}"))
}

fn requirement(bases: &[String], loads: &[f64], floor: f64, limit: &[f64]) -> Result<Requirement> {
    if bases.len() != loads.len() {
        return Err(Error::Other(format!(
            "{} bases for {} loads",
            bases.len(),
            loads.len()
        )));
    }
    let terms = bases
        .iter()
        .zip(loads)
        .map(|(b, &l)| {
            let base = match b.as_str() {
                "unearned-premium" => Base::UnearnedPremium,
                "reserve" => Base::Reserve,
                "outstanding" => Base::Outstanding,
                "written" => Base::Written,
                "earned" => Base::Earned,
                "fixed" => Base::Fixed,
                other => {
                    return Err(bad(
                        "base",
                        other,
                        "\"unearned-premium\", \"reserve\", \"outstanding\", \"written\", \
                         \"earned\" or \"fixed\"",
                    ));
                }
            };
            Ok(Term::new(base, l))
        })
        .collect::<Result<_>>()?;
    let r = Requirement::new(terms).with_floor(floor);
    match limit {
        [] => Ok(r),
        [trigger, cap] => Ok(r.with_limit(*trigger, *cap)),
        _ => Err(Error::Other("limit is c(trigger, cap)".into())),
    }
}

/// Months from `R/cashflow.R`'s `cashflow_dates()`: a kind and its values.
fn dates(kind: &str, values: &[f64], name: &str) -> Result<Dates> {
    match (kind, values) {
        ("monthly", _) => Ok(Dates::monthly()),
        ("quarterly", _) => Ok(Dates::quarterly()),
        ("every", [months, first]) => Ok(Dates::Every {
            months: whole(*months, name)? as usize,
            first: whole(*first, name)? as usize,
        }),
        ("months", m) => Ok(Dates::Months(
            m.iter()
                .map(|&x| whole(x, name).map(|v| v as usize))
                .collect::<Result<_>>()?,
        )),
        (other, _) => Err(Error::Other(format!("{name}: unknown dates {other:?}"))),
    }
}

fn timing(kind: &str, values: &[f64], name: &str) -> Result<Timing> {
    match kind {
        "monthly" => Ok(Timing::Monthly),
        "at-commutation" => Ok(Timing::AtCommutation),
        _ => Ok(Timing::At(dates(kind, values, name)?)),
    }
}

/// A collateralised deal's terms; `R/cashflow.R`'s `cashflow_terms()`
/// documents the arguments.
#[extendr]
pub(crate) struct Cashflow {
    pub(crate) inner: CashflowTerms,
}

#[extendr]
impl Cashflow {
    #[allow(clippy::too_many_arguments)]
    fn new(
        premium: f64,
        commutation_month: f64,
        written: Vec<f64>,
        earned: Vec<f64>,
        paid: Vec<f64>,
        bases: Vec<String>,
        loads: &[f64],
        floor: f64,
        limit: &[f64],
        commutation_bases: Vec<String>,
        commutation_loads: &[f64],
        commutation_floor: f64,
        commutation_limit: &[f64],
        collected_rate: f64,
        lag: f64,
        adjustment_month: &[f64],
        adjustment_rate: &[f64],
        adjustment_less_commission: &[f64],
        adjustment_base: Vec<String>,
        funding_kind: &str,
        funding_values: &[f64],
        funding_rule: &str,
        premium_offset: f64,
        reference: &str,
        funding_source: &str,
        release_kind: &str,
        release_values: &[f64],
        annual_yield: f64,
        interest_basis: &str,
        opening_fixed: f64,
        opening_share: f64,
        expense_rate: f64,
        buffer_months: f64,
        buffer_paid: bool,
        buffer_expenses: bool,
        equity_kind: &str,
        equity_values: &[f64],
        fee_breakpoints: Vec<f64>,
        fee_slopes: Vec<f64>,
        fee_basis: &str,
    ) -> Result<Self> {
        let patterns = MonthlyPatterns::new(written, earned, paid).map_err(to_r)?;
        let mut t = CashflowTerms::new(
            premium,
            whole(commutation_month, "commutation_month")? as usize,
            patterns,
            requirement(&bases, loads, floor, limit)?,
        )
        .map_err(to_r)?;
        t.commutation_requirement = requirement(
            &commutation_bases,
            commutation_loads,
            commutation_floor,
            commutation_limit,
        )?;
        let n = adjustment_month.len();
        if [
            adjustment_rate.len(),
            adjustment_less_commission.len(),
            adjustment_base.len(),
        ]
        .iter()
        .any(|&k| k != n)
        {
            return Err(Error::Other(
                "the adjustments' columns differ in length".into(),
            ));
        }
        let adjustments = (0..n)
            .map(|i| {
                Ok(Adjustment {
                    month: whole(adjustment_month[i], "an adjustment's month")? as usize,
                    written_rate: adjustment_rate[i],
                    less_commission: adjustment_less_commission[i] != 0.0,
                    base: match adjustment_base[i].as_str() {
                        "ultimate" => AdjustmentBase::Ultimate,
                        "to-date" => AdjustmentBase::ToDate,
                        other => {
                            return Err(bad(
                                "adjustment base",
                                other,
                                "\"ultimate\" or \"to-date\"",
                            ));
                        }
                    },
                })
            })
            .collect::<Result<_>>()?;
        t.premium_flow = PremiumFlow {
            collected_rate,
            lag: whole(lag, "lag")? as usize,
            adjustments,
        };
        let rule = match funding_rule {
            "shortfall" => FundingRule::Shortfall,
            "obligation-increase" => FundingRule::ObligationIncrease {
                premium_offset,
                reference: match reference {
                    "previous-date" => ObligationReference::PreviousDate,
                    "running-max" => ObligationReference::RunningMax,
                    other => {
                        return Err(bad(
                            "reference",
                            other,
                            "\"previous-date\" or \"running-max\"",
                        ));
                    }
                },
            },
            other => {
                return Err(bad(
                    "funding rule",
                    other,
                    "\"shortfall\" or \"obligation-increase\"",
                ));
            }
        };
        let source = match funding_source {
            "investor" => FundingSource::Investor,
            "operating-first" => FundingSource::OperatingFirst,
            other => {
                return Err(bad(
                    "funding source",
                    other,
                    "\"investor\" or \"operating-first\"",
                ));
            }
        };
        t.funding = Funding {
            dates: dates(funding_kind, funding_values, "funding_dates")?,
            rule,
            source,
        };
        t.collateral_release = timing(release_kind, release_values, "collateral_release")?;
        t.interest = Interest {
            annual_yield,
            basis: match interest_basis {
                "average" => InterestBasis::Average,
                "opening" => InterestBasis::Opening,
                "closing" => InterestBasis::Closing,
                other => {
                    return Err(bad(
                        "interest basis",
                        other,
                        "\"average\", \"opening\" or \"closing\"",
                    ));
                }
            },
        };
        let fee = if fee_breakpoints.is_empty() && fee_slopes.is_empty() {
            None
        } else {
            Some(PerformanceFee {
                tiers: TieredFee::new(fee_breakpoints, fee_slopes).map_err(to_r)?,
                basis: match fee_basis {
                    "profit" => FeeBasis::Profit,
                    "profit-over-premium" => FeeBasis::ProfitOverPremium,
                    "profit-over-capital" => FeeBasis::ProfitOverCapital,
                    other => {
                        return Err(bad(
                            "fee basis",
                            other,
                            "\"profit\", \"profit-over-premium\" or \"profit-over-capital\"",
                        ));
                    }
                },
            })
        };
        t.operating = Operating {
            opening_fixed,
            opening_share,
            expense_rate,
            buffer: Buffer {
                months: whole(buffer_months, "buffer_months")? as usize,
                paid: buffer_paid,
                expenses: buffer_expenses,
            },
            release: timing(equity_kind, equity_values, "equity_release")?,
            fee,
        };
        // A trace runs the terms' checks, so a bad rule is reported here.
        t.trace(0.0, 0.0).map_err(to_r)?;
        Ok(Cashflow { inner: t })
    }

    /// Each simulation's flows (row-major, one row a simulation) and its
    /// totals; `requirement` empty for the terms' own requirement, or
    /// row-major with `commutation_month + buffer_months` columns.
    fn run(&self, u: &[f64], commission: &[f64], requirement: &[f64]) -> Result<List> {
        let out = if requirement.is_empty() {
            self.inner.run(u, commission)
        } else {
            self.inner.run_with_requirement(u, commission, requirement)
        }
        .map_err(to_r)?;
        Ok(list!(
            flows = out.flows().to_vec(),
            months = out.months() as i32,
            funded = out.funded().to_vec(),
            opening = out.opening().to_vec(),
            released = out.released().to_vec(),
            distributed = out.distributed().to_vec(),
            fee = out.fee().to_vec(),
            commutation = out.commutation().to_vec()
        ))
    }

    /// Every account line of one simulation, month by month.
    fn trace(&self, u: f64, commission: f64) -> Result<List> {
        let t = self.inner.trace(u, commission).map_err(to_r)?;
        let error = t.conservation_error();
        Ok(list!(
            requirement = t.requirement,
            premium = t.premium,
            adjustment = t.adjustment,
            paid = t.paid,
            funding_investor = t.funding_investor,
            funding_operating = t.funding_operating,
            interest = t.interest,
            collateral_release = t.collateral_release,
            collateral_balance = t.collateral_balance,
            expenses = t.expenses,
            equity_release = t.equity_release,
            operating_balance = t.operating_balance,
            investor = t.investor,
            commutation_paid = t.commutation_paid,
            fee = t.fee,
            conservation_error = error
        ))
    }

    fn buffer_months(&self) -> i32 {
        self.inner.operating.buffer.months as i32
    }

    fn commutation_month(&self) -> i32 {
        self.inner.commutation_month as i32
    }

    fn premium(&self) -> f64 {
        self.inner.premium
    }
}

extendr_module! {
    mod cashflow;
    impl Cashflow;
}
