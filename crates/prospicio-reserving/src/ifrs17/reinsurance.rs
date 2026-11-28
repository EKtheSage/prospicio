//! Reinsurance contracts held (paragraphs 60 to 70A, B119C to B119F,
//! `reserving-ifrs17-measurement.md` section 6): a group measured like a
//! GMM group with the signs reversed, as an asset.
//!
//! A [`ReinsuranceHeld`] is recognised from its expected ceded cash flows
//! (premiums paid to the reinsurer, recoveries from it), the RA for the risk
//! it transfers (paragraph 64) and the locked-in curve. Its three
//! differences from a group of insurance contracts issued:
//!
//! * **No loss at recognition** (paragraph 65): the net cost or net gain of
//!   buying the cover is the CSM, positive (a net cost, deferred) or
//!   negative (a net gain, deferred). A net cost for events before the
//!   purchase is an expense at once (paragraph 65A,
//!   [`ReinsuranceOptions::retroactive`]).
//! * **The loss-recovery component** (paragraphs 66A, 66B, B119C to B119F):
//!   when an underlying group is onerous at recognition, the CSM is adjusted
//!   and income recognised by the underlying loss times the share of its
//!   claims the cover recovers; later losses and reversals on the
//!   underlying group are matched the same way, in profit or loss rather
//!   than the CSM (paragraph 66(c)(i)). The component is the share of the
//!   underlying loss component, period by period; its release, which
//!   follows the underlying release, is excluded from the allocation of
//!   premiums paid and from the amounts recovered.
//! * **Non-performance** (paragraph 63): the present value of the
//!   recoveries is reduced for the reinsurer's risk of default
//!   ([`NonPerformance`]), and every change in that reduction is in profit
//!   or loss, not the CSM (paragraph 67).
//!
//! Each period: the CSM accretes at the locked-in rate; the changes for
//! future service (at the locked-in curve) adjust it, less the part matched
//! to the underlying groups' losses and reversals; it is released by
//! coverage units, the release an expense when the CSM is a net cost and
//! income when a net gain. The present value is measured at the current
//! curve when one is given, the difference a finance effect. Recoveries are
//! received in the period they are incurred, so the asset for incurred
//! claims is nil at each period end, as the GMM part assumes; there is no
//! OCI option for the reinsurance group yet.
//!
//! Signs: the asset and every reconciliation row are positive when they add
//! to the asset; profit or loss lines are positive amounts with the sign of
//! their name.
//!
//! No IASB example could be read from a cloud session (Examples 11 and 12
//! are beyond the reachable text), so the tests are worked by hand.
//!
//! ```
//! use prospicio_reserving::DiscountCurve;
//! use prospicio_reserving::ifrs17::reinsurance::{
//!     CededCashFlows, ReinsuranceHeld, ReinsuranceInputs, ReinsuranceOptions,
//! };
//!
//! // A three-year cover: premium 300 paid at once, recoveries of 90 at the
//! // end of each year, RA 15, no discounting. The net cost 300 - 270 - 15
//! // = 15 is the CSM, released evenly.
//! let ceded = CededCashFlows::new(vec![300.0, 0.0, 0.0], vec![90.0; 3]);
//! let curve = DiscountCurve::flat(0.0)?;
//! let mut held = ReinsuranceHeld::recognise(
//!     ceded, 15.0, curve, ReinsuranceOptions::default(), &[],
//! )?;
//! assert_eq!(held.balances().csm, 15.0);
//! assert_eq!(held.balances().asset(), 0.0);
//!
//! let year = held.roll_forward(
//!     &ReinsuranceInputs::new(held.as_expected(), CededCashFlows::new(vec![0.0; 2], vec![90.0; 2]), 10.0),
//!     &[],
//! )?;
//! assert_eq!(year.closing.csm, 10.0);
//! // Allocation of premiums: 90 expected recoveries, 5 of RA, 5 of CSM.
//! assert_eq!(year.allocation.total(), 100.0);
//! assert_eq!(year.net_expense(), 10.0);
//! # Ok::<(), prospicio_reserving::Error>(())
//! ```

use super::{
    CashFlowTiming, CoverageLine, Group, GroupPeriod, NIL, check_ra, check_units, invalid,
};
use crate::discount::DiscountCurve;
use crate::error::{Error, Result};

/// The expected ceded cash flows of a reinsurance group as at one valuation
/// date, by future period: element 0 is the next period.
///
/// `premiums` are paid to the reinsurer, net of ceding commissions that do
/// not depend on claims (paragraph 86); `recoveries` are the amounts
/// expected from it before any allowance for its default. Both streams have
/// one length.
#[derive(Debug, Clone, PartialEq)]
pub struct CededCashFlows {
    /// Premiums paid to the reinsurer.
    pub premiums: Vec<f64>,
    /// Recoveries from the reinsurer, before non-performance.
    pub recoveries: Vec<f64>,
    /// When in each period the premiums are paid (start by default).
    pub premium_timing: CashFlowTiming,
    /// When in each period the recoveries are received (end by default).
    pub recovery_timing: CashFlowTiming,
}

impl CededCashFlows {
    /// Premiums at the start and recoveries at the end of each period.
    pub fn new(premiums: Vec<f64>, recoveries: Vec<f64>) -> Self {
        Self {
            premiums,
            recoveries,
            premium_timing: CashFlowTiming::StartOfPeriod,
            recovery_timing: CashFlowTiming::EndOfPeriod,
        }
    }

    /// Set when the premiums and the recoveries fall within each period.
    pub fn with_timing(mut self, premiums: CashFlowTiming, recoveries: CashFlowTiming) -> Self {
        self.premium_timing = premiums;
        self.recovery_timing = recoveries;
        self
    }

    /// The number of future periods.
    pub fn periods(&self) -> usize {
        self.premiums.len()
    }

    fn validate(&self) -> Result<()> {
        if self.recoveries.len() != self.premiums.len() {
            return Err(Error::LengthMismatch {
                column: "recoveries".into(),
                expected: self.premiums.len(),
                found: self.recoveries.len(),
            });
        }
        for (name, stream) in [
            ("premiums", &self.premiums),
            ("recoveries", &self.recoveries),
        ] {
            if let Some(k) = stream.iter().position(|v| !v.is_finite()) {
                return Err(invalid(format!(
                    "the expected ceded {name} of future period {} are not finite",
                    k + 1
                )));
            }
        }
        Ok(())
    }

    fn at(stream: &[f64], k: usize) -> f64 {
        stream.get(k).copied().unwrap_or(0.0)
    }
}

/// The reinsurer's risk of non-performance (paragraph 63): a constant annual
/// probability of default and the share of a recovery lost when it
/// defaults. A recovery due `τ` years after the valuation date is reduced by
/// `(1 − (1 − p)^τ) × loss_given_default`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonPerformance {
    /// The annual probability of default, in `[0, 1)`.
    pub default_probability: f64,
    /// The share of a recovery lost on default, in `[0, 1]`.
    pub loss_given_default: f64,
}

impl NonPerformance {
    /// Fails when a probability or share is outside its range.
    pub fn new(default_probability: f64, loss_given_default: f64) -> Result<Self> {
        let np = Self {
            default_probability,
            loss_given_default,
        };
        np.validate()?;
        Ok(np)
    }

    fn validate(&self) -> Result<()> {
        if !(self.default_probability >= 0.0 && self.default_probability < 1.0) {
            return Err(invalid(format!(
                "the default probability {} is not in [0, 1)",
                self.default_probability
            )));
        }
        if !(self.loss_given_default >= 0.0 && self.loss_given_default <= 1.0) {
            return Err(invalid(format!(
                "the loss given default {} is not in [0, 1]",
                self.loss_given_default
            )));
        }
        Ok(())
    }

    /// The share of a recovery due in `tau` years expected to be lost.
    fn loss(&self, tau: f64) -> f64 {
        if tau <= 0.0 {
            return 0.0;
        }
        (1.0 - (1.0 - self.default_probability).powf(tau)) * self.loss_given_default
    }
}

/// How a reinsurance group is measured.
#[derive(Debug, Clone, PartialEq)]
pub struct ReinsuranceOptions {
    /// The coverage units of each period of coverage received (B119); one
    /// per period when `None`.
    pub coverage_units: Option<Vec<f64>>,
    /// The length of a period in years.
    pub period_years: f64,
    /// The cover is for events before its purchase: a net cost at
    /// recognition is an expense at once (paragraph 65A).
    pub retroactive: bool,
    /// The reinsurer's risk of non-performance at recognition; none when
    /// `None`.
    pub non_performance: Option<NonPerformance>,
}

impl Default for ReinsuranceOptions {
    fn default() -> Self {
        Self {
            coverage_units: None,
            period_years: 1.0,
            retroactive: false,
            non_performance: None,
        }
    }
}

/// The ceded cash flows of one period as they happened.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CededActual {
    /// Premiums paid to the reinsurer.
    pub premiums: f64,
    /// The part of `premiums` beyond those expected that pays for future
    /// coverage, an adjustment of the CSM (B96(a)).
    pub premiums_for_future_service: f64,
    /// Recoveries received.
    pub recoveries: f64,
}

/// What happened in one period and the view at its end.
#[derive(Debug, Clone, PartialEq)]
pub struct ReinsuranceInputs {
    /// The ceded cash flows of the period.
    pub actual: CededActual,
    /// The expected ceded cash flows as at the end of the period.
    pub expected: CededCashFlows,
    /// The RA at the end of the period.
    pub risk_adjustment: f64,
    /// The RA released for the period's coverage; by coverage units when
    /// `None`, the rest of its change being future service.
    pub risk_adjustment_released: Option<f64>,
    /// Revised coverage units of the later periods.
    pub future_coverage_units: Option<Vec<f64>>,
    /// The current curve at the end of the period, maturities from then.
    pub current_curve: Option<DiscountCurve>,
    /// The reinsurer's risk of non-performance at the end of the period; as
    /// before when `None`.
    pub non_performance: Option<NonPerformance>,
}

impl ReinsuranceInputs {
    /// The period's cash flows, the view at its end and the RA.
    pub fn new(actual: CededActual, expected: CededCashFlows, risk_adjustment: f64) -> Self {
        Self {
            actual,
            expected,
            risk_adjustment,
            risk_adjustment_released: None,
            future_coverage_units: None,
            current_curve: None,
            non_performance: None,
        }
    }

    /// Measure the closing present value at a current curve.
    pub fn with_current_curve(mut self, curve: DiscountCurve) -> Self {
        self.current_curve = Some(curve);
        self
    }

    /// The reinsurer's risk of non-performance as revised at the period end.
    pub fn with_non_performance(mut self, non_performance: NonPerformance) -> Self {
        self.non_performance = Some(non_performance);
        self
    }

    /// The RA released for the period's coverage.
    pub fn with_risk_adjustment_released(mut self, released: f64) -> Self {
        self.risk_adjustment_released = Some(released);
        self
    }

    /// Revised coverage units of the later periods.
    pub fn with_future_coverage_units(mut self, units: Vec<f64>) -> Self {
        self.future_coverage_units = Some(units);
        self
    }
}

/// The balances of a reinsurance group, as an asset.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ReinsuranceBalances {
    /// The present value of the future recoveries less the future premiums,
    /// before non-performance, at the current curve.
    pub present_value: f64,
    /// The same at the locked-in curve.
    pub present_value_locked_in: f64,
    /// The reduction of the present value for the reinsurer's risk of
    /// non-performance, a positive amount.
    pub non_performance: f64,
    /// The RA for the risk transferred.
    pub risk_adjustment: f64,
    /// The CSM: a net cost when positive, a net gain when negative.
    pub csm: f64,
    /// The loss-recovery component, a part of the asset for remaining
    /// coverage.
    pub loss_recovery: f64,
}

impl ReinsuranceBalances {
    /// The fulfilment cash flows.
    pub fn fulfilment_cash_flows(&self) -> f64 {
        self.present_value - self.non_performance + self.risk_adjustment
    }

    /// The asset: the fulfilment cash flows and the CSM.
    pub fn asset(&self) -> f64 {
        self.fulfilment_cash_flows() + self.csm
    }
}

/// A line of the reconciliation by measurement component of a reinsurance
/// group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CededLine {
    /// The balances at the start of the period.
    Opening,
    /// Contracts recognised in the period.
    NewContracts,
    /// Changes that relate to future service.
    FutureService,
    /// The CSM adjusted for income on onerous underlying groups after
    /// recognition (paragraph 66(c)(i)).
    LossRecovery,
    /// The CSM recognised in profit or loss.
    CsmRecognised,
    /// The change in the RA for the risk that expired.
    RiskExpired,
    /// Experience adjustments for current service.
    ExperienceAdjustments,
    /// The change in the reinsurer's risk of non-performance (paragraph 67).
    NonPerformance,
    /// Finance income: the unwind of the present value and the accretion of
    /// the CSM.
    FinanceIncome,
    /// Premiums paid to the reinsurer.
    PremiumsPaid,
    /// Recoveries received.
    RecoveriesReceived,
    /// The balances at the end of the period.
    Closing,
}

impl CededLine {
    /// The line's name, as the front ends report it (`"opening"`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Opening => "opening",
            Self::NewContracts => "new_contracts",
            Self::FutureService => "future_service",
            Self::LossRecovery => "loss_recovery",
            Self::CsmRecognised => "csm_recognised",
            Self::RiskExpired => "risk_expired",
            Self::ExperienceAdjustments => "experience_adjustments",
            Self::NonPerformance => "non_performance",
            Self::FinanceIncome => "finance_income",
            Self::PremiumsPaid => "premiums_paid",
            Self::RecoveriesReceived => "recoveries_received",
            Self::Closing => "closing",
        }
    }
}

/// One row of the reconciliation by measurement component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CededRow {
    /// Which line.
    pub line: CededLine,
    /// The present value before non-performance.
    pub present_value: f64,
    /// The non-performance column, negative when it reduces the asset.
    pub non_performance: f64,
    /// The RA.
    pub risk_adjustment: f64,
    /// The CSM.
    pub csm: f64,
}

impl CededRow {
    /// The row's effect on the asset.
    pub fn total(&self) -> f64 {
        self.present_value + self.non_performance + self.risk_adjustment + self.csm
    }
}

/// The allocation of the premiums paid (paragraph 86), an expense.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Allocation {
    /// The recoveries expected for the period, less the release of the
    /// loss-recovery component (B119F).
    pub expected_recoveries: f64,
    /// The RA released.
    pub risk_adjustment_released: f64,
    /// The CSM released: positive for a net cost, negative for a net gain.
    pub csm_released: f64,
    /// Premiums paid beyond those expected for the period's coverage.
    pub premium_experience: f64,
}

impl Allocation {
    /// The allocation of premiums paid.
    pub fn total(&self) -> f64 {
        self.expected_recoveries
            + self.risk_adjustment_released
            + self.csm_released
            + self.premium_experience
    }
}

/// The amounts recovered from the reinsurer (paragraph 86), income.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Recovered {
    /// Recoveries of claims incurred, less the release of the
    /// loss-recovery component.
    pub claims: f64,
    /// Income on the losses of onerous underlying groups (paragraph 66A), at
    /// recognition and after.
    pub losses: f64,
    /// The reversal of that income as the underlying losses reverse.
    pub loss_reversals: f64,
    /// The change in the reinsurer's risk of non-performance, negative when
    /// the risk grows.
    pub non_performance: f64,
}

impl Recovered {
    /// The amounts recovered.
    pub fn total(&self) -> f64 {
        self.claims + self.losses - self.loss_reversals + self.non_performance
    }
}

/// The movement of the loss-recovery component in a period.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LossRecoveryMovement {
    /// At the start of the period.
    pub opening: f64,
    /// Set up for the underlying groups' losses.
    pub losses: f64,
    /// Reversed with the underlying groups' reversals.
    pub reversals: f64,
    /// Its share of the underlying finance expense.
    pub finance: f64,
    /// Released with the underlying loss components.
    pub released: f64,
    /// At the end of the period.
    pub closing: f64,
}

/// What a period of a reinsurance group reports.
#[derive(Debug, Clone, PartialEq)]
pub struct ReinsurancePeriod {
    /// The period's number, from 1.
    pub period: usize,
    /// The reconciliation by measurement component (paragraph 101).
    pub by_component: Vec<CededRow>,
    /// The loss-recovery component.
    pub loss_recovery: LossRecoveryMovement,
    /// The allocation of premiums paid.
    pub allocation: Allocation,
    /// The amounts recovered.
    pub recovered: Recovered,
    /// A net cost for past events, expensed at recognition (paragraph 65A).
    pub past_events_cost: f64,
    /// Finance income: the unwind of the present value, the current curve's
    /// effect and the accretion of the CSM.
    pub finance_income: f64,
    /// The balances at the end of the period.
    pub closing: ReinsuranceBalances,
}

impl ReinsurancePeriod {
    /// The net expense from the reinsurance (paragraph 86): the allocation of
    /// premiums paid and any past-events cost, less the amounts recovered.
    pub fn net_expense(&self) -> f64 {
        self.allocation.total() + self.past_events_cost - self.recovered.total()
    }

    /// The profit or loss: finance income less the net expense.
    pub fn profit(&self) -> f64 {
        self.finance_income - self.net_expense()
    }

    /// A row of the reconciliation.
    pub fn component(&self, line: CededLine) -> Option<&CededRow> {
        self.by_component.iter().find(|r| r.line == line)
    }
}

/// A group of reinsurance contracts held.
#[derive(Debug, Clone, PartialEq)]
pub struct ReinsuranceHeld {
    pub(super) curve: DiscountCurve,
    pub(super) options: ReinsuranceOptions,
    pub(super) period: usize,
    pub(super) view: CededCashFlows,
    pub(super) non_performance: Option<NonPerformance>,
    pub(super) balances: ReinsuranceBalances,
    pub(super) coverage_units: Vec<f64>,
    /// The share of each underlying group's claims recovered.
    pub(super) shares: Vec<f64>,
    /// The income recognised at recognition for each underlying group.
    pub(super) recognised_losses: Vec<f64>,
    pub(super) recognised: Option<(ReinsuranceBalances, f64)>,
}

impl ReinsuranceHeld {
    /// Recognise a reinsurance group from its expected ceded cash flows, the
    /// RA for the risk transferred and the curve at recognition.
    ///
    /// `underlying` pairs each underlying group with the share of its claims
    /// the cover is expected to recover (B119D). A group recognised at the
    /// same time (not yet rolled forward) that is onerous gives income of
    /// its loss component times the share (paragraph 66A); later losses come
    /// through [`roll_forward`](Self::roll_forward).
    ///
    /// Fails when the cash flows are not finite or of different lengths,
    /// the RA is negative, a share is outside `[0, 1]`, or the coverage
    /// units do not fit.
    pub fn recognise(
        expected: CededCashFlows,
        risk_adjustment: f64,
        curve: DiscountCurve,
        options: ReinsuranceOptions,
        underlying: &[(&Group, f64)],
    ) -> Result<Self> {
        expected.validate()?;
        check_ra(risk_adjustment)?;
        if !(options.period_years.is_finite() && options.period_years > 0.0) {
            return Err(Error::InvalidSetting {
                name: "period_years",
                value: options.period_years,
                expected: "finite and positive",
            });
        }
        if let Some(np) = &options.non_performance {
            np.validate()?;
        }
        let coverage_units = match &options.coverage_units {
            Some(units) => units.clone(),
            None => vec![1.0; expected.periods()],
        };
        check_units(&coverage_units, "coverage_units")?;
        if coverage_units.iter().sum::<f64>() <= 0.0 {
            return Err(invalid("the reinsurance provides no coverage units"));
        }
        let mut shares = Vec::with_capacity(underlying.len());
        let mut recognised_losses = Vec::with_capacity(underlying.len());
        for (k, (group, share)) in underlying.iter().enumerate() {
            check_share(*share, k)?;
            shares.push(*share);
            let loss = if group.period() == 0 {
                group.balances().loss_component
            } else {
                0.0
            };
            recognised_losses.push(share * loss);
        }
        let mut held = Self {
            curve,
            non_performance: options.non_performance,
            options,
            period: 0,
            view: expected,
            balances: ReinsuranceBalances::default(),
            coverage_units,
            shares,
            recognised_losses,
            recognised: None,
        };
        let pv = held.present_value(&held.view, 0.0);
        let npr = held.non_performance_at(&held.view, 0.0, held.non_performance, None);
        let recovery: f64 = held.recognised_losses.iter().sum();
        let fcf = pv - npr + risk_adjustment;
        let mut csm = recovery - fcf;
        let mut past = 0.0;
        if held.options.retroactive && csm > 0.0 {
            past = csm;
            csm = 0.0;
        }
        let balances = ReinsuranceBalances {
            present_value: pv,
            present_value_locked_in: pv,
            non_performance: npr,
            risk_adjustment,
            csm,
            loss_recovery: recovery,
        };
        held.balances = balances;
        held.recognised = Some((balances, past));
        Ok(held)
    }

    /// The balances after the last period rolled forward.
    pub fn balances(&self) -> ReinsuranceBalances {
        self.balances
    }

    /// The periods rolled forward.
    pub fn period(&self) -> usize {
        self.period
    }

    /// The expected ceded cash flows as at the last period end.
    pub fn expected(&self) -> &CededCashFlows {
        &self.view
    }

    /// The next period's ceded cash flows as expected, before default.
    pub fn as_expected(&self) -> CededActual {
        CededActual {
            premiums: CededCashFlows::at(&self.view.premiums, 0),
            premiums_for_future_service: 0.0,
            recoveries: CededCashFlows::at(&self.view.recoveries, 0),
        }
    }

    fn factor(&self, t: f64) -> f64 {
        self.curve.factor(t)
    }

    /// A view's flows to the cedant (recoveries in, premiums out) with their
    /// times in years, for a view at `at`.
    fn flows<'a>(
        &self,
        view: &'a CededCashFlows,
        at: f64,
    ) -> impl Iterator<Item = (f64, f64)> + 'a {
        let l = self.options.period_years;
        let (tp, tr) = (view.premium_timing.offset(), view.recovery_timing.offset());
        (0..view.periods()).flat_map(move |k| {
            let s = at + k as f64 * l;
            [
                (s + tp * l, -view.premiums[k]),
                (s + tr * l, view.recoveries[k]),
            ]
        })
    }

    /// The present value at `at` of a view taken at `at`, at the locked-in
    /// curve.
    fn present_value(&self, view: &CededCashFlows, at: f64) -> f64 {
        self.flows(view, at)
            .map(|(u, cf)| cf * self.factor(u) / self.factor(at))
            .sum()
    }

    /// The same at a current curve whose maturities run from `at`.
    fn present_value_at(&self, view: &CededCashFlows, at: f64, curve: &DiscountCurve) -> f64 {
        self.flows(view, at)
            .map(|(u, cf)| cf * curve.factor(u - at))
            .sum()
    }

    /// The reduction for non-performance of a view's recoveries at `at`.
    fn non_performance_at(
        &self,
        view: &CededCashFlows,
        at: f64,
        np: Option<NonPerformance>,
        curve: Option<&DiscountCurve>,
    ) -> f64 {
        let Some(np) = np else { return 0.0 };
        let l = self.options.period_years;
        (0..view.periods())
            .map(|k| {
                let tau = (k as f64 + view.recovery_timing.offset()) * l;
                let df = match curve {
                    Some(c) => c.factor(tau),
                    None => self.factor(at + tau) / self.factor(at),
                };
                view.recoveries[k] * np.loss(tau) * df
            })
            .sum()
    }

    /// Roll the group forward over one period, with the same period of each
    /// underlying group, in the order given at recognition.
    ///
    /// Fails when the inputs are not finite, the RA is negative, the
    /// coverage has ended, the revised units do not fit, or the underlying
    /// periods are not one per underlying group.
    pub fn roll_forward(
        &mut self,
        inputs: &ReinsuranceInputs,
        underlying: &[&GroupPeriod],
    ) -> Result<ReinsurancePeriod> {
        inputs.expected.validate()?;
        check_ra(inputs.risk_adjustment)?;
        if let Some(np) = &inputs.non_performance {
            np.validate()?;
        }
        if underlying.len() != self.shares.len() {
            return Err(invalid(format!(
                "{} underlying periods for {} underlying groups",
                underlying.len(),
                self.shares.len()
            )));
        }
        let actual = inputs.actual;
        for (name, v) in [
            ("premiums", actual.premiums),
            (
                "premiums_for_future_service",
                actual.premiums_for_future_service,
            ),
            ("recoveries", actual.recoveries),
        ] {
            if !v.is_finite() {
                return Err(invalid(format!("the actual ceded {name} are not finite")));
            }
        }
        let Some((&cu, later)) = self.coverage_units.split_first() else {
            return Err(invalid("the reinsurance coverage has ended"));
        };
        let future_units = match &inputs.future_coverage_units {
            Some(units) => {
                check_units(units, "future_coverage_units")?;
                units.clone()
            }
            None => later.to_vec(),
        };
        let later_units: f64 = future_units.iter().sum();
        if cu + later_units <= 0.0 {
            return Err(invalid(
                "no coverage units are left for this period or later ones",
            ));
        }

        let l = self.options.period_years;
        let s = self.period as f64 * l;
        let e = s + l;
        let growth = self.factor(s) / self.factor(e) - 1.0;
        let first = self.period == 0;
        let opening = if first {
            ReinsuranceBalances::default()
        } else {
            self.balances
        };
        let start = self.balances;
        let new = self.recognised.take();

        // The loss-recovery component, from the underlying loss components.
        let mut lr = LossRecoveryMovement {
            opening: opening.loss_recovery,
            ..Default::default()
        };
        let mut matched = 0.0;
        for (k, p) in underlying.iter().enumerate() {
            let share = self.shares[k];
            let lc = |line| p.coverage(line).map_or(0.0, |r| r.loss_component);
            let losses = share * lc(CoverageLine::OnerousLosses);
            let reversals = -share * lc(CoverageLine::LossReversals);
            lr.losses += losses;
            lr.reversals += reversals;
            lr.finance += share * lc(CoverageLine::FinanceExpenses);
            lr.released += -share * lc(CoverageLine::IncurredClaims);
            let at_recognition = if first {
                self.recognised_losses[k]
            } else {
                0.0
            };
            matched += losses - at_recognition - reversals;
        }
        lr.closing = lr.opening + lr.losses - lr.reversals + lr.finance - lr.released;
        if lr.closing.abs() < NIL {
            lr.closing = 0.0;
        }

        // The expected flows of the period, and the unwind at the locked-in
        // curve.
        let exp_premiums = CededCashFlows::at(&self.view.premiums, 0);
        let exp_recoveries = CededCashFlows::at(&self.view.recoveries, 0);
        let finance_locked_in = start.present_value_locked_in * growth
            - self
                .flows(&self.view, s)
                .take(2)
                .map(|(u, cf)| cf * (self.factor(u) / self.factor(e) - 1.0))
                .sum::<f64>();
        let accretion = start.csm * growth;

        let ra_released = inputs
            .risk_adjustment_released
            .unwrap_or(start.risk_adjustment * cu / (cu + later_units));
        if !ra_released.is_finite() {
            return Err(invalid("the RA released is not finite"));
        }

        // Cash and experience: recoveries in, premiums out.
        let cash = actual.premiums - actual.recoveries;
        let premium_current = actual.premiums - exp_premiums - actual.premiums_for_future_service;
        let experience = (actual.recoveries - exp_recoveries) - premium_current;

        // Future service at the locked-in curve; the current curve's effect
        // is finance.
        let closing_locked_in = self.present_value(&inputs.expected, e);
        let future_pv = closing_locked_in
            - (start.present_value_locked_in + finance_locked_in + cash + experience);
        let closing_pv = match &inputs.current_curve {
            Some(curve) => self.present_value_at(&inputs.expected, e, curve),
            None => closing_locked_in,
        };
        let finance_pv = closing_pv - (start.present_value + future_pv + cash + experience);
        let future_ra = inputs.risk_adjustment - (start.risk_adjustment - ra_released);

        // Non-performance: every change in profit or loss (paragraph 67).
        let np = inputs.non_performance.or(self.non_performance);
        let closing_npr =
            self.non_performance_at(&inputs.expected, e, np, inputs.current_curve.as_ref());
        let npr_change = closing_npr - start.non_performance;

        // The CSM: accretion, future service less the part matched to the
        // underlying losses, then the release by coverage units.
        let change = future_pv + future_ra;
        let future_service_csm = -change;
        let loss_recovery_csm = matched;
        let mut csm = start.csm + accretion + future_service_csm + loss_recovery_csm;
        let csm_released = csm * cu / (cu + later_units);
        csm -= csm_released;

        let closing = ReinsuranceBalances {
            present_value: closing_pv,
            present_value_locked_in: closing_locked_in,
            non_performance: closing_npr,
            risk_adjustment: inputs.risk_adjustment,
            csm,
            loss_recovery: lr.closing,
        };

        let allocation = Allocation {
            expected_recoveries: exp_recoveries - lr.released,
            risk_adjustment_released: ra_released,
            csm_released,
            premium_experience: premium_current,
        };
        let recovered = Recovered {
            claims: actual.recoveries - lr.released,
            losses: lr.losses,
            loss_reversals: lr.reversals,
            non_performance: -npr_change,
        };
        let (new_balances, past_events_cost) = match new {
            Some((b, past)) => (Some(b), past),
            None => (None, 0.0),
        };

        let row = |line, pv, npr, ra, csm| CededRow {
            line,
            present_value: pv,
            non_performance: npr,
            risk_adjustment: ra,
            csm,
        };
        let mut by_component = vec![row(
            CededLine::Opening,
            opening.present_value,
            -opening.non_performance,
            opening.risk_adjustment,
            opening.csm,
        )];
        if let Some(b) = new_balances {
            by_component.push(row(
                CededLine::NewContracts,
                b.present_value,
                -b.non_performance,
                b.risk_adjustment,
                b.csm,
            ));
        }
        by_component.extend([
            row(
                CededLine::FutureService,
                future_pv,
                0.0,
                future_ra,
                future_service_csm,
            ),
            row(CededLine::LossRecovery, 0.0, 0.0, 0.0, loss_recovery_csm),
            row(CededLine::CsmRecognised, 0.0, 0.0, 0.0, -csm_released),
            row(CededLine::RiskExpired, 0.0, 0.0, -ra_released, 0.0),
            row(CededLine::ExperienceAdjustments, experience, 0.0, 0.0, 0.0),
            row(CededLine::NonPerformance, 0.0, -npr_change, 0.0, 0.0),
            row(CededLine::FinanceIncome, finance_pv, 0.0, 0.0, accretion),
            row(CededLine::PremiumsPaid, actual.premiums, 0.0, 0.0, 0.0),
            row(
                CededLine::RecoveriesReceived,
                -actual.recoveries,
                0.0,
                0.0,
                0.0,
            ),
            row(
                CededLine::Closing,
                closing.present_value,
                -closing.non_performance,
                closing.risk_adjustment,
                closing.csm,
            ),
        ]);

        self.period += 1;
        self.view = inputs.expected.clone();
        self.non_performance = np;
        self.balances = closing;
        self.coverage_units = future_units;

        Ok(ReinsurancePeriod {
            period: self.period,
            by_component,
            loss_recovery: lr,
            allocation,
            recovered,
            past_events_cost,
            finance_income: finance_pv + accretion,
            closing,
        })
    }
}

fn check_share(share: f64, k: usize) -> Result<()> {
    if share.is_finite() && (0.0..=1.0).contains(&share) {
        Ok(())
    } else {
        Err(invalid(format!(
            "the share of underlying group {k} recovered is {share}, not in [0, 1]"
        )))
    }
}
