//! IFRS 17 measurement of a group of insurance contracts
//! (`docs/design/reserving-ifrs17-measurement.md`).
//!
//! This first part is the general measurement model (GMM, paragraphs 32 to
//! 52): a [`Group`] is recognised from its expected cash flows, its risk
//! adjustment for non-financial risk (RA) and a locked-in discount curve,
//! then rolled forward one reporting period at a time from what happened in
//! the period and the expected cash flows as at its end. Each period gives a
//! [`GroupPeriod`]: the contractual service margin (CSM) and the loss
//! component, the two reconciliations of paragraphs 100 and 101, and the
//! profit or loss lines (insurance revenue, insurance service expenses,
//! insurance finance expenses).
//!
//! The library does the accounting, not the projection (decision 2): the
//! expected cash flows are the user's, by future period. Within a period:
//!
//! 1. the CSM accretes at the locked-in rate (paragraph 44(b));
//! 2. the loss component takes its share of the period's release of
//!    expected claims and expenses, of the RA released and of the finance
//!    expense: the loss component over the present value of the future
//!    claims and expenses (investment components excluded) plus the RA at
//!    the start of the period (paragraphs 50(a) and 51, IFRS 17 Example 8,
//!    decision 5);
//! 3. the changes that relate to future service (B96: the revised present
//!    value at the locked-in curve, the RA for future service, experience
//!    adjustments of premiums for future service and of acquisition cash
//!    flows) adjust the CSM; an unfavourable change beyond the CSM is a loss
//!    that starts or adds to the loss component (paragraph 48), and a
//!    favourable one reverses the loss component before it rebuilds the CSM
//!    (paragraph 50(b));
//! 4. the CSM is released by coverage units, `cu / (cu + future cu)`, after
//!    steps 1 and 3 (B119; undiscounted unless
//!    [`GroupOptions::discount_coverage_units`]).
//!
//! The present value is measured at the current curve of each period end
//! ([`PeriodInputs::with_current_curve`]; the locked-in curve when none is
//! given), and the changes that adjust the CSM at the locked-in curve: the
//! difference, and the effect of changes in financial assumptions
//! ([`PeriodInputs::with_financial_revision`]), are insurance finance income
//! or expense (B97(a)). With [`GroupOptions::oci`] profit or loss takes the
//! finance expense at the locked-in curve on the cash flows as revised
//! (B131, B132(a)(i)) and OCI the rest, so that the accumulated OCI is the
//! locked-in present value less the current one; the effective yield and
//! projected crediting rate allocations (B132(a)(ii)) are not built. The
//! CSM accretes at [`GroupOptions::accretion_curve`] when the rates for
//! nominal cash flows differ (B72(b)). The loss component's share of the
//! finance expense is taken at the locked-in curve.
//!
//! What this part leaves to the later ones (the design note's PRs 3 to 5):
//! the RA's change is all
//! insurance service result (paragraph 81 not disaggregated); claims and
//! expenses are paid in the period they are incurred, so the liability for
//! incurred claims (LIC) is nil at each period end, as in the IASB's
//! examples; contracts are recognised at once, not added during the
//! cohort's year; and there is no premium allocation approach, reinsurance
//! held or JSON state.
//!
//! Signs: the liability and every reconciliation row are positive when they
//! add to the liability; profit or loss lines are positive amounts with
//! the sign of their name (revenue, expenses, finance expenses).
//!
//! ```
//! use prospicio_reserving::DiscountCurve;
//! use prospicio_reserving::ifrs17::{ExpectedCashFlows, Group, GroupOptions, PeriodInputs};
//!
//! // IFRS 17 Example 1A: premium 900 at once, claims of 200 at the end of
//! // each of three years, RA 120, 5%.
//! let cash = ExpectedCashFlows::new(3)
//!     .with_premiums(vec![900.0, 0.0, 0.0])
//!     .with_claims(vec![200.0; 3]);
//! let curve = DiscountCurve::flat(0.05)?;
//! let mut group = Group::recognise(cash, 120.0, curve, GroupOptions::default())?;
//! assert_eq!(group.balances().csm.round(), 235.0);
//!
//! // Year 1 as expected (Example 2): the CSM accretes 12 and releases 82.
//! let year = group.roll_forward(&PeriodInputs::new(
//!     group.as_expected(),
//!     ExpectedCashFlows::new(2).with_claims(vec![200.0; 2]),
//!     80.0,
//! ))?;
//! assert_eq!(year.closing.csm.round(), 165.0);
//! assert_eq!(year.closing.liability().round(), 617.0);
//! # Ok::<(), prospicio_reserving::Error>(())
//! ```

use crate::discount::DiscountCurve;
use crate::error::{Error, Result};

/// The share of the loss component below which it counts as nil.
const NIL: f64 = 1e-9;

fn invalid(why: impl Into<String>) -> Error {
    Error::Ifrs17(why.into())
}

/// When in a period a stream of cash flows falls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CashFlowTiming {
    /// At the start of the period (premiums received in advance, acquisition
    /// cash flows paid at issue).
    StartOfPeriod,
    /// At the middle of the period.
    MidPeriod,
    /// At the end of the period (the IASB examples' claims).
    EndOfPeriod,
}

impl CashFlowTiming {
    /// The fraction of a period after its start at which the flows fall.
    fn offset(self) -> f64 {
        match self {
            Self::StartOfPeriod => 0.0,
            Self::MidPeriod => 0.5,
            Self::EndOfPeriod => 1.0,
        }
    }
}

/// The expected future cash flows of a group as at one valuation date, by
/// future period: element 0 is the next period.
///
/// Every amount is a probability-weighted expected value, undiscounted.
/// `claims` include any investment component, which `investment_component`
/// then names (paragraph 85: amounts repaid in every circumstance, excluded
/// from insurance revenue and insurance service expenses). The streams must
/// have one length.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectedCashFlows {
    /// Premiums to be received (inflows).
    pub premiums: Vec<f64>,
    /// Claims to be paid, investment components included.
    pub claims: Vec<f64>,
    /// Other fulfilment expenses to be paid (claims handling, maintenance).
    pub expenses: Vec<f64>,
    /// Insurance acquisition cash flows to be paid.
    pub acquisition: Vec<f64>,
    /// The part of `claims` that is an investment component.
    pub investment_component: Vec<f64>,
    /// When in a period the premiums fall (default: the start).
    pub premium_timing: CashFlowTiming,
    /// When in a period claims and expenses fall (default: the end).
    pub claim_timing: CashFlowTiming,
    /// When in a period acquisition cash flows fall (default: the start).
    pub acquisition_timing: CashFlowTiming,
}

impl ExpectedCashFlows {
    /// `periods` future periods of nil cash flows, premiums and acquisition
    /// cash flows at the start of a period and claims and expenses at its
    /// end, as in the IASB's examples.
    pub fn new(periods: usize) -> Self {
        Self {
            premiums: vec![0.0; periods],
            claims: vec![0.0; periods],
            expenses: vec![0.0; periods],
            acquisition: vec![0.0; periods],
            investment_component: vec![0.0; periods],
            premium_timing: CashFlowTiming::StartOfPeriod,
            claim_timing: CashFlowTiming::EndOfPeriod,
            acquisition_timing: CashFlowTiming::StartOfPeriod,
        }
    }

    /// With these premiums by future period.
    pub fn with_premiums(mut self, premiums: Vec<f64>) -> Self {
        self.premiums = premiums;
        self
    }

    /// With these claims by future period (investment components included).
    pub fn with_claims(mut self, claims: Vec<f64>) -> Self {
        self.claims = claims;
        self
    }

    /// With these other fulfilment expenses by future period.
    pub fn with_expenses(mut self, expenses: Vec<f64>) -> Self {
        self.expenses = expenses;
        self
    }

    /// With these acquisition cash flows by future period.
    pub fn with_acquisition(mut self, acquisition: Vec<f64>) -> Self {
        self.acquisition = acquisition;
        self
    }

    /// With these investment components (parts of `claims`) by future period.
    pub fn with_investment_component(mut self, investment_component: Vec<f64>) -> Self {
        self.investment_component = investment_component;
        self
    }

    /// With these timings of premiums, of claims and expenses, and of
    /// acquisition cash flows.
    pub fn with_timing(
        mut self,
        premiums: CashFlowTiming,
        claims: CashFlowTiming,
        acquisition: CashFlowTiming,
    ) -> Self {
        self.premium_timing = premiums;
        self.claim_timing = claims;
        self.acquisition_timing = acquisition;
        self
    }

    /// The number of future periods.
    pub fn periods(&self) -> usize {
        self.premiums.len()
    }

    fn validate(&self) -> Result<()> {
        let n = self.premiums.len();
        for (name, stream) in [
            ("claims", &self.claims),
            ("expenses", &self.expenses),
            ("acquisition", &self.acquisition),
            ("investment_component", &self.investment_component),
        ] {
            if stream.len() != n {
                return Err(Error::LengthMismatch {
                    column: name.into(),
                    expected: n,
                    found: stream.len(),
                });
            }
        }
        for (name, stream) in [
            ("premiums", &self.premiums),
            ("claims", &self.claims),
            ("expenses", &self.expenses),
            ("acquisition", &self.acquisition),
            ("investment_component", &self.investment_component),
        ] {
            if let Some(k) = stream.iter().position(|v| !v.is_finite()) {
                return Err(invalid(format!(
                    "the expected {name} of future period {} are not finite",
                    k + 1
                )));
            }
        }
        Ok(())
    }

    /// The amount of a stream in future period `k`, nil beyond the horizon.
    fn at(stream: &[f64], k: usize) -> f64 {
        stream.get(k).copied().unwrap_or(0.0)
    }
}

/// The cash flows of one period as they happened.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ActualCashFlows {
    /// Premiums received.
    pub premiums: f64,
    /// The part of the difference between the premiums received and those
    /// expected that relates to future service, which adjusts the CSM (B96(a));
    /// the rest of the difference relates to current or past service and is
    /// insurance revenue (B124(a)(iv)). Nil by default.
    pub premiums_for_future_service: f64,
    /// Claims incurred and paid, investment components included.
    pub claims: f64,
    /// Other fulfilment expenses incurred and paid.
    pub expenses: f64,
    /// Insurance acquisition cash flows paid; a difference from those
    /// expected adjusts the CSM (B96(a)).
    pub acquisition: f64,
    /// The part of `claims` that is an investment component.
    pub investment_component: f64,
}

/// What happened in one period and the view at its end.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodInputs {
    /// The cash flows of the period.
    pub actual: ActualCashFlows,
    /// The expected future cash flows as at the end of the period, element 0
    /// the next period.
    pub expected: ExpectedCashFlows,
    /// The RA as at the end of the period.
    pub risk_adjustment: f64,
    /// The RA released for the risk that expired in the period (current
    /// service); the rest of the RA's change relates to future service.
    /// `None` releases the opening RA by the period's coverage units,
    /// `RA · cu / (cu + future cu)`, undiscounted.
    pub risk_adjustment_released: Option<f64>,
    /// The coverage units expected in the periods after this one, when they
    /// are revised; `None` keeps those set at recognition.
    pub future_coverage_units: Option<Vec<f64>>,
    /// The acquisition cash flows recovered in the period (B125); `None`
    /// allocates those not yet recovered by the period's coverage units,
    /// undiscounted.
    pub acquisition_recovered: Option<f64>,
    /// The current discount curve at the end of the period, maturities from
    /// that date (paragraph 36, B72(a)); `None` measures at the locked-in
    /// curve.
    pub current_curve: Option<DiscountCurve>,
    /// The view at the start of the period revised for the changes in
    /// financial assumptions made in it (expected returns credited, an
    /// index): the opening view's periods, element 0 this period. Its effect
    /// on the present value is insurance finance income or expense, not a
    /// change for future service (B97(a), B128); the period's expected cash
    /// flows are read from it. `None` keeps the opening view.
    pub financial_revision: Option<ExpectedCashFlows>,
}

impl PeriodInputs {
    /// The period's cash flows, the expected cash flows and the RA at its end;
    /// the RA released, the coverage units and the recovery of acquisition
    /// cash flows by their defaults.
    pub fn new(actual: ActualCashFlows, expected: ExpectedCashFlows, risk_adjustment: f64) -> Self {
        Self {
            actual,
            expected,
            risk_adjustment,
            risk_adjustment_released: None,
            future_coverage_units: None,
            acquisition_recovered: None,
            current_curve: None,
            financial_revision: None,
        }
    }

    /// With the current curve at the end of the period, maturities from it.
    pub fn with_current_curve(mut self, curve: DiscountCurve) -> Self {
        self.current_curve = Some(curve);
        self
    }

    /// With the opening view revised for the period's changes in financial
    /// assumptions.
    pub fn with_financial_revision(mut self, revised: ExpectedCashFlows) -> Self {
        self.financial_revision = Some(revised);
        self
    }

    /// With the RA released for current service given.
    pub fn with_risk_adjustment_released(mut self, released: f64) -> Self {
        self.risk_adjustment_released = Some(released);
        self
    }

    /// With the coverage units of the later periods revised.
    pub fn with_future_coverage_units(mut self, units: Vec<f64>) -> Self {
        self.future_coverage_units = Some(units);
        self
    }

    /// With the acquisition cash flows recovered in the period given.
    pub fn with_acquisition_recovered(mut self, recovered: f64) -> Self {
        self.acquisition_recovered = Some(recovered);
        self
    }
}

/// The settings of a group.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupOptions {
    /// The coverage units provided in each period of the coverage, from the
    /// first; `None` gives one unit to each period of the expected cash
    /// flows at recognition (the passage of time).
    pub coverage_units: Option<Vec<f64>>,
    /// Discount the future coverage units at the locked-in curve when the CSM
    /// is released (B119 allows either; default: undiscounted, decision 5).
    pub discount_coverage_units: bool,
    /// The length of a period in years (1 for annual reporting, 0.25 for
    /// quarterly), the step of the curve between period ends.
    pub period_years: f64,
    /// Disaggregate insurance finance income or expense between profit or
    /// loss and other comprehensive income (paragraph 88(b)): profit or loss
    /// takes the finance expense at the locked-in curve (B131, B72(e)(i)),
    /// OCI the rest, so that the accumulated OCI is the present value at the
    /// locked-in curve less that at the current one. Default: all in profit
    /// or loss (paragraph 88(a)).
    pub oci: bool,
    /// The curve at recognition that accretes the CSM, for cash flows that
    /// do not vary with the returns on underlying items (B72(b)), when it
    /// differs from the locked-in curve that measures the changes adjusting
    /// the CSM (B72(c)); `None` uses the locked-in curve.
    pub accretion_curve: Option<DiscountCurve>,
}

impl Default for GroupOptions {
    fn default() -> Self {
        Self {
            coverage_units: None,
            discount_coverage_units: false,
            period_years: 1.0,
            oci: false,
            accretion_curve: None,
        }
    }
}

/// The balances of a group at a period end (or at recognition).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Balances {
    /// The present value of the future cash flows, outflows less inflows, at
    /// the current curve (the locked-in curve when no current one is given).
    pub present_value: f64,
    /// The same present value at the locked-in curve.
    pub present_value_locked_in: f64,
    /// The RA for non-financial risk.
    pub risk_adjustment: f64,
    /// The contractual service margin.
    pub csm: f64,
    /// The loss component of the liability for remaining coverage, a part
    /// of `present_value + risk_adjustment`.
    pub loss_component: f64,
}

impl Balances {
    /// The fulfilment cash flows, `present_value + risk_adjustment`.
    pub fn fulfilment_cash_flows(&self) -> f64 {
        self.present_value + self.risk_adjustment
    }

    /// The liability for remaining coverage, `present_value +
    /// risk_adjustment + csm` (the LIC is nil in this part).
    pub fn liability(&self) -> f64 {
        self.present_value + self.risk_adjustment + self.csm
    }

    /// The liability for remaining coverage excluding the loss component.
    pub fn excluding_loss_component(&self) -> f64 {
        self.liability() - self.loss_component
    }

    /// The effect of the current curve on the present value, the locked-in
    /// present value less the current one: the accumulated OCI of a group
    /// that takes the OCI option (paragraph 88(b)).
    pub fn accumulated_oci(&self) -> f64 {
        self.present_value_locked_in - self.present_value
    }
}

/// A line of the reconciliation by measurement component (paragraph 101).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentLine {
    /// The balances at the start of the period.
    Opening,
    /// Contracts recognised in the period (paragraph 104(a)(iii)).
    NewContracts,
    /// Changes that relate to future service: in estimates that adjust the
    /// CSM, and those that do not (losses on onerous groups and their
    /// reversals; paragraph 104(a)(i) and (ii)).
    FutureService,
    /// The CSM recognised in profit or loss (paragraph 104(b)(i)).
    CsmRecognised,
    /// The change in the RA for the risk that expired (paragraph 104(b)(ii)).
    RiskExpired,
    /// Experience adjustments for current service (paragraph 104(b)(iii)).
    ExperienceAdjustments,
    /// Insurance finance expenses: the unwind of the present value and the
    /// accretion of the CSM.
    FinanceExpenses,
    /// Premiums received.
    PremiumsReceived,
    /// Claims and other fulfilment expenses paid, investment components
    /// included.
    ClaimsAndExpensesPaid,
    /// Insurance acquisition cash flows paid.
    AcquisitionPaid,
    /// The balances at the end of the period.
    Closing,
}

/// One row of the reconciliation by measurement component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComponentRow {
    /// Which line.
    pub line: ComponentLine,
    /// The present value of future cash flows.
    pub present_value: f64,
    /// The RA.
    pub risk_adjustment: f64,
    /// The CSM.
    pub csm: f64,
}

impl ComponentRow {
    /// The row's effect on the liability.
    pub fn total(&self) -> f64 {
        self.present_value + self.risk_adjustment + self.csm
    }
}

/// A line of the reconciliation by remaining coverage and incurred claims
/// (paragraph 100).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageLine {
    /// The balances at the start of the period.
    Opening,
    /// Insurance revenue (paragraph 103(a)).
    InsuranceRevenue,
    /// Claims and other expenses incurred, less the loss component's share
    /// of the expected release (paragraph 103(b)(i)).
    IncurredClaims,
    /// Amortisation of insurance acquisition cash flows (paragraph 103(b)(ii)).
    AcquisitionAmortisation,
    /// Losses on onerous groups (paragraph 103(b)(iv)), new contracts
    /// included.
    OnerousLosses,
    /// Reversals of losses on onerous groups.
    LossReversals,
    /// Investment components, moved from the remaining coverage to the
    /// incurred claims (paragraph 103(c)).
    InvestmentComponents,
    /// Insurance finance expenses.
    FinanceExpenses,
    /// Premiums received.
    PremiumsReceived,
    /// Insurance acquisition cash flows paid.
    AcquisitionPaid,
    /// Claims and other fulfilment expenses paid, investment components
    /// included.
    ClaimsAndExpensesPaid,
    /// The balances at the end of the period.
    Closing,
}

/// One row of the reconciliation by remaining coverage and incurred claims.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoverageRow {
    /// Which line.
    pub line: CoverageLine,
    /// The liability for remaining coverage excluding the loss component.
    pub excluding_loss_component: f64,
    /// The loss component.
    pub loss_component: f64,
    /// The liability for incurred claims.
    pub incurred_claims: f64,
}

impl CoverageRow {
    /// The row's effect on the liability.
    pub fn total(&self) -> f64 {
        self.excluding_loss_component + self.loss_component + self.incurred_claims
    }
}

/// Insurance revenue by what it releases (B121 to B124).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Revenue {
    /// The claims and expenses expected for the period, investment
    /// components excluded, less the loss component's share.
    pub expected_claims_and_expenses: f64,
    /// The RA released for the risk that expired, less the loss component's
    /// share.
    pub risk_adjustment_released: f64,
    /// The CSM recognised for the service of the period.
    pub csm_released: f64,
    /// The recovery of insurance acquisition cash flows (B125).
    pub acquisition_recovered: f64,
    /// Experience adjustments of premiums for current or past service.
    pub premium_experience: f64,
}

impl Revenue {
    /// The insurance revenue of the period.
    pub fn total(&self) -> f64 {
        self.expected_claims_and_expenses
            + self.risk_adjustment_released
            + self.csm_released
            + self.acquisition_recovered
            + self.premium_experience
    }
}

/// One period of a group: its reconciliations and its profit or loss.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupPeriod {
    /// The period, from 1.
    pub period: usize,
    /// The reconciliation by measurement component (paragraph 101), from
    /// `Opening` to `Closing`.
    pub by_component: Vec<ComponentRow>,
    /// The reconciliation by remaining coverage and incurred claims
    /// (paragraph 100), from `Opening` to `Closing`.
    pub by_coverage: Vec<CoverageRow>,
    /// Insurance revenue by what it releases.
    pub revenue: Revenue,
    /// Claims and other expenses incurred, investment components excluded,
    /// less the loss component's share of the expected release.
    pub incurred_claims: f64,
    /// Amortisation of insurance acquisition cash flows, equal to their
    /// recovery in revenue.
    pub acquisition_amortisation: f64,
    /// Losses on onerous groups recognised in the period, at recognition
    /// included.
    pub onerous_losses: f64,
    /// Reversals of such losses.
    pub loss_reversals: f64,
    /// Insurance finance expenses: the unwind of the present value and the
    /// accretion of the CSM, the effect of changes in the current curve and
    /// in financial assumptions included (B97(a)), in profit or loss and OCI
    /// together.
    pub finance_expenses: f64,
    /// The part of `finance_expenses` in other comprehensive income under
    /// the OCI option, nil without it.
    pub finance_expenses_in_oci: f64,
    /// The loss component's share of the period's release, finance expense
    /// included (nil when the group is not onerous at the start of the
    /// period).
    pub loss_component_share: f64,
    /// The balances at the end of the period.
    pub closing: Balances,
}

impl GroupPeriod {
    /// Insurance revenue.
    pub fn insurance_revenue(&self) -> f64 {
        self.revenue.total()
    }

    /// Insurance service expenses: incurred claims, acquisition amortisation,
    /// and losses less reversals.
    pub fn insurance_service_expenses(&self) -> f64 {
        self.incurred_claims + self.acquisition_amortisation + self.onerous_losses
            - self.loss_reversals
    }

    /// Insurance service result, revenue less service expenses.
    pub fn insurance_service_result(&self) -> f64 {
        self.insurance_revenue() - self.insurance_service_expenses()
    }

    /// Insurance finance expenses in profit or loss.
    pub fn finance_expenses_in_profit(&self) -> f64 {
        self.finance_expenses - self.finance_expenses_in_oci
    }

    /// Profit or loss of the group: the service result less the finance
    /// expenses in profit or loss.
    pub fn profit(&self) -> f64 {
        self.insurance_service_result() - self.finance_expenses_in_profit()
    }

    /// Other comprehensive income of the group, the finance expenses in OCI
    /// with their sign turned.
    pub fn other_comprehensive_income(&self) -> f64 {
        -self.finance_expenses_in_oci
    }

    /// The reconciliation row by measurement component of a line.
    pub fn component(&self, line: ComponentLine) -> Option<&ComponentRow> {
        self.by_component.iter().find(|r| r.line == line)
    }

    /// The reconciliation row by remaining coverage of a line.
    pub fn coverage(&self, line: CoverageLine) -> Option<&CoverageRow> {
        self.by_coverage.iter().find(|r| r.line == line)
    }
}

/// A group of insurance contracts under the general measurement model.
#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    curve: DiscountCurve,
    options: GroupOptions,
    /// Periods rolled forward.
    period: usize,
    /// The expected cash flows as at the last period end (or recognition).
    view: ExpectedCashFlows,
    balances: Balances,
    /// The coverage units of the remaining periods, the next first.
    coverage_units: Vec<f64>,
    /// Acquisition cash flows paid to date, and recovered in revenue to date.
    acquisition_paid: f64,
    acquisition_recovered: f64,
    /// The balances recognised, reported in the first period.
    recognised: Option<Balances>,
}

impl Group {
    /// Recognise a group (paragraphs 38 and 47): its expected cash flows from
    /// the first period of coverage, its RA, and the curve at recognition,
    /// which stays the locked-in curve.
    ///
    /// A net inflow gives a CSM; a net outflow is a loss, the loss component.
    /// Fails when the cash flows are not finite or of different lengths,
    /// when the RA is negative, or when the coverage units do not fit.
    pub fn recognise(
        expected: ExpectedCashFlows,
        risk_adjustment: f64,
        curve: DiscountCurve,
        options: GroupOptions,
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
        let coverage_units = match &options.coverage_units {
            Some(units) => units.clone(),
            None => vec![1.0; expected.periods()],
        };
        check_units(&coverage_units, "coverage_units")?;
        if coverage_units.iter().sum::<f64>() <= 0.0 {
            return Err(invalid("the group provides no coverage units"));
        }
        let mut group = Self {
            curve,
            options,
            period: 0,
            view: expected,
            balances: Balances::default(),
            coverage_units,
            acquisition_paid: 0.0,
            acquisition_recovered: 0.0,
            recognised: None,
        };
        let pv = group.present_value(&group.view, 0.0);
        let fcf = pv + risk_adjustment;
        let balances = Balances {
            present_value: pv,
            present_value_locked_in: pv,
            risk_adjustment,
            csm: (-fcf).max(0.0),
            loss_component: fcf.max(0.0),
        };
        group.balances = balances;
        group.recognised = Some(balances);
        Ok(group)
    }

    /// The balances after the last period rolled forward (at recognition
    /// before the first).
    pub fn balances(&self) -> Balances {
        self.balances
    }

    /// The periods rolled forward.
    pub fn period(&self) -> usize {
        self.period
    }

    /// The expected cash flows as at the last period end.
    pub fn expected(&self) -> &ExpectedCashFlows {
        &self.view
    }

    /// The next period's cash flows as expected, for a period in which
    /// everything happens as expected.
    pub fn as_expected(&self) -> ActualCashFlows {
        let v = &self.view;
        ActualCashFlows {
            premiums: ExpectedCashFlows::at(&v.premiums, 0),
            premiums_for_future_service: 0.0,
            claims: ExpectedCashFlows::at(&v.claims, 0),
            expenses: ExpectedCashFlows::at(&v.expenses, 0),
            acquisition: ExpectedCashFlows::at(&v.acquisition, 0),
            investment_component: ExpectedCashFlows::at(&v.investment_component, 0),
        }
    }

    /// The discount factor at `t` years of the locked-in curve.
    fn factor(&self, t: f64) -> f64 {
        self.curve.factor(t)
    }

    /// The present value at `at` of a view taken at `at`, at a current curve
    /// whose maturities run from `at`.
    fn present_value_at(&self, view: &ExpectedCashFlows, at: f64, curve: &DiscountCurve) -> f64 {
        self.flows(view, at)
            .map(|(u, cf)| cf * curve.factor(u - at))
            .sum()
    }

    /// The present value at time `at` (years) of a view taken at `at`.
    fn present_value(&self, view: &ExpectedCashFlows, at: f64) -> f64 {
        self.flows(view, at)
            .map(|(u, cf)| cf * self.factor(u) / self.factor(at))
            .sum()
    }

    /// The present value at `at` of a view's future claims and expenses,
    /// investment components excluded: the base of the loss component's share.
    fn service_outflows(&self, view: &ExpectedCashFlows, at: f64) -> f64 {
        let l = self.options.period_years;
        (0..view.periods())
            .map(|k| {
                let u = at + (k as f64 + view.claim_timing.offset()) * l;
                (view.claims[k] + view.expenses[k] - view.investment_component[k]) * self.factor(u)
                    / self.factor(at)
            })
            .sum()
    }

    /// A view's net outflows with their times in years, for a view at `at`.
    fn flows<'a>(
        &self,
        view: &'a ExpectedCashFlows,
        at: f64,
    ) -> impl Iterator<Item = (f64, f64)> + 'a {
        let l = self.options.period_years;
        let (tp, tc, ta) = (
            view.premium_timing.offset(),
            view.claim_timing.offset(),
            view.acquisition_timing.offset(),
        );
        (0..view.periods()).flat_map(move |k| {
            let s = at + k as f64 * l;
            [
                (s + tp * l, -view.premiums[k]),
                (s + tc * l, view.claims[k] + view.expenses[k]),
                (s + ta * l, view.acquisition[k]),
            ]
        })
    }

    /// Roll the group forward over one period: what happened in it and the
    /// view at its end.
    ///
    /// Fails when the inputs are not finite, the RA is negative, the
    /// coverage has ended, or the revised coverage units do not fit.
    pub fn roll_forward(&mut self, inputs: &PeriodInputs) -> Result<GroupPeriod> {
        inputs.expected.validate()?;
        check_ra(inputs.risk_adjustment)?;
        if let Some(revised) = &inputs.financial_revision {
            revised.validate()?;
            if revised.periods() != self.view.periods() {
                return Err(invalid(format!(
                    "the financial revision has {} periods, the opening view {}",
                    revised.periods(),
                    self.view.periods()
                )));
            }
        }
        let actual = inputs.actual;
        for (name, v) in [
            ("premiums", actual.premiums),
            (
                "premiums_for_future_service",
                actual.premiums_for_future_service,
            ),
            ("claims", actual.claims),
            ("expenses", actual.expenses),
            ("acquisition", actual.acquisition),
            ("investment_component", actual.investment_component),
        ] {
            if !v.is_finite() {
                return Err(invalid(format!("the actual {name} are not finite")));
            }
        }
        let Some((&cu, later)) = self.coverage_units.split_first() else {
            return Err(invalid("the group's coverage has ended"));
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
        let accretion_growth = match &self.options.accretion_curve {
            Some(c) => c.factor(s) / c.factor(e) - 1.0,
            None => growth,
        };
        // The period's expected flows come from the view revised for
        // financial assumptions, whose effect is a finance effect.
        let view = inputs.financial_revision.as_ref().unwrap_or(&self.view);

        // Opening, and the balances recognised in the first period.
        let opening = if self.period == 0 {
            Balances::default()
        } else {
            self.balances
        };
        let start = self.balances;
        let new = self.recognised.take();
        let revision = match &inputs.financial_revision {
            Some(revised) => self.present_value(revised, s) - start.present_value_locked_in,
            None => 0.0,
        };
        let start_locked_in = start.present_value_locked_in + revision;

        // The opening view's flows in this period, and their unwind.
        let l0 = |stream: &[f64]| ExpectedCashFlows::at(stream, 0);
        let exp_premiums = l0(&view.premiums);
        let exp_claims = l0(&view.claims) + l0(&view.expenses);
        let exp_investment = l0(&view.investment_component);
        let in_period: Vec<(f64, f64)> = self
            .flows(view, s)
            .take(3)
            .filter(|&(_, cf)| cf != 0.0)
            .collect();
        let finance_locked_in = start_locked_in * growth
            - in_period
                .iter()
                .map(|&(u, cf)| cf * (self.factor(u) / self.factor(e) - 1.0))
                .sum::<f64>();
        let accretion = start.csm * accretion_growth;

        // The RA released, and the acquisition cash flows recovered.
        let ra_released = inputs
            .risk_adjustment_released
            .unwrap_or(start.risk_adjustment * cu / (cu + later_units));
        if !ra_released.is_finite() {
            return Err(invalid("the RA released is not finite"));
        }
        let acquisition_total = self.acquisition_paid
            + actual.acquisition
            + inputs.expected.acquisition.iter().sum::<f64>();
        let acquisition_recovered = inputs
            .acquisition_recovered
            .unwrap_or((acquisition_total - self.acquisition_recovered) * cu / (cu + later_units));
        if !acquisition_recovered.is_finite() {
            return Err(invalid(
                "the acquisition cash flows recovered are not finite",
            ));
        }

        // The loss component's share of the release (paragraphs 50(a), 51).
        let release = exp_claims - exp_investment;
        let base = self.service_outflows(view, s) + start.risk_adjustment;
        let share = if start.loss_component > NIL && base > 0.0 {
            (start.loss_component / base).min(1.0)
        } else {
            0.0
        };
        // The finance expense allocated is the unwind of the base, so that the
        // loss component runs off with the base by the end of the coverage.
        let base_unwind = self.service_outflows(view, s) * growth
            - (exp_claims - exp_investment)
                * (self.factor(s + view.claim_timing.offset() * l) / self.factor(e) - 1.0);
        let lc_finance = share * base_unwind;
        let mut lc_release = share * (release + ra_released);
        let lc_after = start.loss_component + lc_finance - lc_release;
        if lc_after < 0.0 {
            lc_release += lc_after;
        }
        let lc_after = (start.loss_component + lc_finance - lc_release).max(0.0);

        // Cash flows and experience adjustments.
        let paid = actual.claims + actual.expenses;
        let cash = actual.premiums - paid - actual.acquisition;
        let premium_difference = actual.premiums - exp_premiums;
        let premium_current = premium_difference - actual.premiums_for_future_service;
        let experience = (paid - exp_claims) - premium_current;

        // Changes for future service, at the locked-in curve (B96); the
        // current curve's effect on them, and on everything else, is finance
        // (B97(a)): the finance row is what the current measurement leaves.
        let closing_locked_in = self.present_value(&inputs.expected, e);
        let future_pv =
            closing_locked_in - (start_locked_in + finance_locked_in + cash + experience);
        let closing_pv = match &inputs.current_curve {
            Some(curve) => self.present_value_at(&inputs.expected, e, curve),
            None => closing_locked_in,
        };
        let finance_pv = closing_pv - (start.present_value + future_pv + cash + experience);
        let finance_pv_in_oci = if self.options.oci {
            finance_pv - (finance_locked_in + revision)
        } else {
            0.0
        };
        let future_ra = inputs.risk_adjustment - (start.risk_adjustment - ra_released);
        let change = future_pv + future_ra;
        let csm_before = start.csm + accretion;
        let (mut csm, mut lc) = (csm_before, lc_after);
        let (mut loss, mut reversal) = (0.0, 0.0);
        if change > 0.0 {
            if lc > NIL {
                lc += change;
                loss = change;
            } else if change > csm {
                loss = change - csm;
                lc = loss;
                csm = 0.0;
            } else {
                csm -= change;
            }
        } else {
            let gain = -change;
            reversal = gain.min(lc);
            lc -= reversal;
            csm += gain - reversal;
        }

        // The CSM released for the service of the period (B119).
        let weights: f64 = if self.options.discount_coverage_units {
            future_units
                .iter()
                .enumerate()
                .map(|(k, u)| u * self.factor(e + (k + 1) as f64 * l) / self.factor(e))
                .sum()
        } else {
            later_units
        };
        let csm_adjustment = csm - csm_before;
        let csm_released = csm * cu / (cu + weights);
        csm -= csm_released;

        // The loss component is a part of the fulfilment cash flows and never
        // exceeds them; what is left beyond them is released with the rest.
        let excess = lc - closing_pv_ra(closing_pv, inputs.risk_adjustment);
        if lc > NIL && excess > 0.0 {
            lc -= excess;
            lc_release += excess;
        }

        let closing = Balances {
            present_value: closing_pv,
            present_value_locked_in: closing_locked_in,
            risk_adjustment: inputs.risk_adjustment,
            csm,
            loss_component: if lc > NIL { lc } else { 0.0 },
        };
        let new_loss = new.map_or(0.0, |b| b.loss_component);
        let finance = finance_pv + accretion;
        let revenue = Revenue {
            expected_claims_and_expenses: release - lc_release * release_part(release, ra_released),
            risk_adjustment_released: ra_released
                - lc_release * (1.0 - release_part(release, ra_released)),
            csm_released,
            acquisition_recovered,
            premium_experience: premium_current,
        };
        let incurred = paid - actual.investment_component - lc_release;

        let mut by_component = vec![row(
            ComponentLine::Opening,
            opening.present_value,
            opening.risk_adjustment,
            opening.csm,
        )];
        if let Some(b) = new {
            by_component.push(row(
                ComponentLine::NewContracts,
                b.present_value,
                b.risk_adjustment,
                b.csm,
            ));
        }
        by_component.extend([
            row(
                ComponentLine::FutureService,
                future_pv,
                future_ra,
                csm_adjustment,
            ),
            row(ComponentLine::CsmRecognised, 0.0, 0.0, -csm_released),
            row(ComponentLine::RiskExpired, 0.0, -ra_released, 0.0),
            row(ComponentLine::ExperienceAdjustments, experience, 0.0, 0.0),
            row(ComponentLine::FinanceExpenses, finance_pv, 0.0, accretion),
            row(ComponentLine::PremiumsReceived, actual.premiums, 0.0, 0.0),
            row(ComponentLine::ClaimsAndExpensesPaid, -paid, 0.0, 0.0),
            row(
                ComponentLine::AcquisitionPaid,
                -actual.acquisition,
                0.0,
                0.0,
            ),
            row(
                ComponentLine::Closing,
                closing.present_value,
                closing.risk_adjustment,
                closing.csm,
            ),
        ]);

        let cov = |line, x, lc, lic| CoverageRow {
            line,
            excluding_loss_component: x,
            loss_component: lc,
            incurred_claims: lic,
        };
        let by_coverage = vec![
            cov(
                CoverageLine::Opening,
                opening.excluding_loss_component(),
                opening.loss_component,
                0.0,
            ),
            cov(CoverageLine::InsuranceRevenue, -revenue.total(), 0.0, 0.0),
            cov(
                CoverageLine::IncurredClaims,
                0.0,
                -lc_release,
                paid - actual.investment_component,
            ),
            cov(
                CoverageLine::AcquisitionAmortisation,
                acquisition_recovered,
                0.0,
                0.0,
            ),
            cov(CoverageLine::OnerousLosses, 0.0, new_loss + loss, 0.0),
            cov(CoverageLine::LossReversals, 0.0, -reversal, 0.0),
            cov(
                CoverageLine::InvestmentComponents,
                -actual.investment_component,
                0.0,
                actual.investment_component,
            ),
            cov(
                CoverageLine::FinanceExpenses,
                finance - lc_finance,
                lc_finance,
                0.0,
            ),
            cov(CoverageLine::PremiumsReceived, actual.premiums, 0.0, 0.0),
            cov(CoverageLine::AcquisitionPaid, -actual.acquisition, 0.0, 0.0),
            cov(CoverageLine::ClaimsAndExpensesPaid, 0.0, 0.0, -paid),
            cov(
                CoverageLine::Closing,
                closing.excluding_loss_component(),
                closing.loss_component,
                0.0,
            ),
        ];

        self.period += 1;
        self.view = inputs.expected.clone();
        self.balances = closing;
        self.coverage_units = future_units;
        self.acquisition_paid += actual.acquisition;
        self.acquisition_recovered += acquisition_recovered;

        Ok(GroupPeriod {
            period: self.period,
            by_component,
            by_coverage,
            revenue,
            incurred_claims: incurred,
            acquisition_amortisation: acquisition_recovered,
            onerous_losses: new_loss + loss,
            loss_reversals: reversal,
            finance_expenses: finance,
            finance_expenses_in_oci: finance_pv_in_oci,
            loss_component_share: share,
            closing,
        })
    }
}

/// The part of the loss component's release that falls on the expected
/// claims and expenses rather than on the RA.
fn release_part(release: f64, ra_released: f64) -> f64 {
    let total = release + ra_released;
    if total == 0.0 { 1.0 } else { release / total }
}

fn closing_pv_ra(present_value: f64, risk_adjustment: f64) -> f64 {
    (present_value + risk_adjustment).max(0.0)
}

fn row(line: ComponentLine, present_value: f64, risk_adjustment: f64, csm: f64) -> ComponentRow {
    ComponentRow {
        line,
        present_value,
        risk_adjustment,
        csm,
    }
}

fn check_ra(ra: f64) -> Result<()> {
    if ra.is_finite() && ra >= 0.0 {
        Ok(())
    } else {
        Err(invalid(format!(
            "the RA {ra} is not finite and non-negative"
        )))
    }
}

fn check_units(units: &[f64], name: &str) -> Result<()> {
    if let Some(k) = units.iter().position(|u| !(u.is_finite() && *u >= 0.0)) {
        return Err(invalid(format!(
            "{name}[{k}] is {}, not finite and non-negative",
            units[k]
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
