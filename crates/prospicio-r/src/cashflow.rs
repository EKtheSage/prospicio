//! A collateralised deal's monthly cash flows (Aggregate lane) over
//! `prospicio_pricing::cashflow` (`docs/design/time-value.md`, decisions
//! 13 to 20), and the deal valued after binding (decision 34).
//! `R/cashflow.R` builds the arguments.

use extendr_api::prelude::*;
use extendr_api::{Error, Result};
use prospicio_pricing::cashflow::{
    ActualState, Adjustment, AdjustmentBase, Base, Buffer, CashflowTerms, Conditioning, Dates,
    FeeBasis, Funding, FundingRule, FundingSource, Interest, InterestBasis, MonthlyPatterns,
    ObligationReference, Operating, PatternDraws, PerformanceFee, PremiumFlow, Requirement, Term,
    Timing, Trace, Valuation, condition_on_paid, deal_ave,
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

fn conditioning(name: &str) -> Result<Conditioning> {
    match name {
        "bornhuetter-ferguson" => Ok(Conditioning::BornhuetterFerguson),
        "truncation" => Ok(Conditioning::Truncation),
        other => Err(bad(
            "conditioning",
            other,
            "\"bornhuetter-ferguson\" or \"truncation\"",
        )),
    }
}

/// An actual state from `R/cashflow.R`'s `cashflow_valuation()`: the six
/// amounts `c(premium_collected, paid, collateral, funded, released,
/// operating)`, the flows when `has_flows`, and the funding reference
/// (empty for none).
fn actual_state(
    month: f64,
    amounts: &[f64],
    flows: &[f64],
    has_flows: bool,
    reference: &[f64],
) -> Result<ActualState> {
    let [
        premium_collected,
        paid,
        collateral,
        funded,
        released,
        operating,
    ] = amounts
    else {
        return Err(Error::Other("an actual state has six amounts".into()));
    };
    Ok(ActualState {
        month: whole(month, "month")? as usize,
        premium_collected: *premium_collected,
        paid: *paid,
        collateral: *collateral,
        funded: *funded,
        released: *released,
        operating: *operating,
        flows: has_flows.then(|| flows.to_vec()),
        funding_reference: reference.first().copied(),
    })
}

fn patterns(paid: &[f64], n_patterns: f64, index: &[f64]) -> Result<Option<PatternDraws>> {
    if paid.is_empty() {
        if !index.is_empty() {
            return Err(Error::Other("pattern_index needs paid_patterns".into()));
        }
        return Ok(None);
    }
    let k = whole(n_patterns, "the number of paid patterns")? as usize;
    if k == 0 || paid.len() % k != 0 {
        return Err(Error::Other(format!(
            "{} paid values do not split into {k} patterns",
            paid.len()
        )));
    }
    let patterns: Vec<Vec<f64>> = paid.chunks(paid.len() / k).map(<[f64]>::to_vec).collect();
    let draws = if index.is_empty() {
        PatternDraws::new(patterns)
    } else {
        let index = index
            .iter()
            .map(|&i| whole(i, "pattern_index").map(|i| i as usize))
            .collect::<Result<Vec<_>>>()?;
        PatternDraws::indexed(patterns, index)
    }
    .map_err(to_r)?;
    Ok(Some(draws))
}

fn trace_list(t: Trace) -> List {
    let error = t.conservation_error();
    list!(
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
        funding_reference = t.funding_reference,
        opening_balance = t.opening_balance,
        commutation_paid = t.commutation_paid,
        fee = t.fee,
        conservation_error = error
    )
}

/// Each simulation's ultimate loss ratio conditioned on the loss paid to
/// date; `R/cashflow.R`'s `condition_on_paid()` documents the arguments.
#[extendr]
fn cashflow_condition_on_paid(
    u: &[f64],
    premium: f64,
    paid: f64,
    share_paid: f64,
    conditioning: &str,
) -> Result<List> {
    let how = self::conditioning(conditioning)?;
    let c = condition_on_paid(u, premium, paid, share_paid, how).map_err(to_r)?;
    let kept: Vec<f64> = c.kept.iter().map(|&j| j as f64).collect();
    Ok(list!(kept = kept, ultimate = c.ultimate))
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
        deduct_commission: bool,
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
            deduct_commission,
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
    /// row-major with `commutation_month + buffer_months` columns. `paid`
    /// empty for the terms' own paid pattern, or `n_patterns` patterns
    /// row-major, each simulation's `index` from 0 (empty for one pattern a
    /// simulation, in order).
    fn run(
        &self,
        u: &[f64],
        commission: &[f64],
        requirement: &[f64],
        paid: &[f64],
        n_patterns: f64,
        index: &[f64],
    ) -> Result<List> {
        let out = if let Some(draws) = patterns(paid, n_patterns, index)? {
            if !requirement.is_empty() {
                return Err(Error::Other(
                    "give paid patterns or a months requirement, not both".into(),
                ));
            }
            self.inner.run_with_patterns(u, commission, &draws)
        } else if requirement.is_empty() {
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
            commutation = out.commutation().to_vec(),
            paid_first = out.paid_first()
        ))
    }

    /// Every account line of one simulation, month by month; `paid` empty
    /// for the terms' own paid pattern.
    fn trace(&self, u: f64, commission: f64, paid: &[f64]) -> Result<List> {
        let t = if paid.is_empty() {
            self.inner.trace(u, commission)
        } else {
            self.inner.trace_with_pattern(u, commission, paid)
        }
        .map_err(to_r)?;
        Ok(trace_list(t))
    }

    /// The actual state of one simulated path at the start of `month`:
    /// `amounts` as `cashflow_valuation()` passes them, the flows and the
    /// funding reference.
    fn actual_state(&self, u: f64, month: f64, commission: f64, paid: &[f64]) -> Result<List> {
        let t = if paid.is_empty() {
            self.inner.trace(u, commission)
        } else {
            self.inner.trace_with_pattern(u, commission, paid)
        }
        .map_err(to_r)?;
        let s = ActualState::from_trace(&t, whole(month, "month")? as usize).map_err(to_r)?;
        Ok(list!(
            premium_collected = s.premium_collected,
            paid = s.paid,
            collateral = s.collateral,
            funded = s.funded,
            released = s.released,
            operating = s.operating,
            flows = s.flows.unwrap_or_default(),
            funding_reference = s.funding_reference.unwrap_or(0.0)
        ))
    }

    /// The valuation's share paid and net asset value, checking the state.
    fn valuation(
        &self,
        month: f64,
        amounts: &[f64],
        flows: &[f64],
        has_flows: bool,
        reference: &[f64],
    ) -> Result<List> {
        let v = self.valued(month, amounts, flows, has_flows, reference)?;
        Ok(list!(share_paid = v.share_paid(), nav = v.nav()))
    }

    /// Each simulation's flows from the valuation month, conditioned on
    /// the actual paid; `kept` from 0.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        month: f64,
        amounts: &[f64],
        flows: &[f64],
        has_flows: bool,
        reference: &[f64],
        u: &[f64],
        commission: &[f64],
        conditioning: &str,
    ) -> Result<List> {
        let v = self.valued(month, amounts, flows, has_flows, reference)?;
        let f = v
            .run(u, commission, self::conditioning(conditioning)?)
            .map_err(to_r)?;
        let since = if has_flows {
            Robj::from(f.since_inception_rows().map_err(to_r)?)
        } else {
            Robj::from(())
        };
        let since_moic = if has_flows {
            Robj::from(f.since_inception_moic().map_err(to_r)?)
        } else {
            Robj::from(())
        };
        let out = f.flows();
        let kept: Vec<f64> = f.kept().iter().map(|&j| j as f64).collect();
        Ok(list!(
            kept = kept,
            ultimate = f.ultimate().to_vec(),
            nav = f.nav(),
            flows = out.flows().to_vec(),
            months = out.months() as i32,
            forward_rows = f.forward_rows(),
            since_inception_rows = since,
            forward_moic = f.forward_moic().map_err(to_r)?,
            since_inception_moic = since_moic,
            funded = out.funded().to_vec(),
            released = out.released().to_vec(),
            distributed = out.distributed().to_vec(),
            fee = out.fee().to_vec(),
            commutation = out.commutation().to_vec()
        ))
    }

    /// Every account line of one simulation from the valuation month.
    #[allow(clippy::too_many_arguments)]
    fn valuation_trace(
        &self,
        month: f64,
        amounts: &[f64],
        flows: &[f64],
        has_flows: bool,
        reference: &[f64],
        u: f64,
        commission: f64,
        conditioning: &str,
    ) -> Result<List> {
        let v = self.valued(month, amounts, flows, has_flows, reference)?;
        let t = v
            .trace(u, commission, self::conditioning(conditioning)?)
            .map_err(to_r)?;
        Ok(trace_list(t))
    }

    /// Actual against expected paid by month; `paid`, `n_patterns` and
    /// `index` as `run`'s.
    fn deal_ave(
        &self,
        actual: &[f64],
        u: &[f64],
        paid: &[f64],
        n_patterns: f64,
        index: &[f64],
        levels: &[f64],
    ) -> Result<List> {
        let draws = patterns(paid, n_patterns, index)?;
        let r = deal_ave(actual, &self.inner, u, draws.as_ref(), levels).map_err(to_r)?;
        let month: Vec<f64> = r.month.iter().map(|&m| m as f64).collect();
        Ok(list!(
            month = month,
            actual = r.actual,
            expected = r.expected,
            ratio = r.ratio,
            percentile = r.percentile,
            quantiles = r.quantiles
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

impl Cashflow {
    fn valued(
        &self,
        month: f64,
        amounts: &[f64],
        flows: &[f64],
        has_flows: bool,
        reference: &[f64],
    ) -> Result<Valuation> {
        let state = actual_state(month, amounts, flows, has_flows, reference)?;
        self.inner.valuation(state).map_err(to_r)
    }
}

extendr_module! {
    mod cashflow;
    fn cashflow_condition_on_paid;
    impl Cashflow;
}
