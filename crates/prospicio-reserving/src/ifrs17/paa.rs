//! The premium allocation approach (paragraphs 53 to 59,
//! `reserving-ifrs17-measurement.md` section 5): a simplified liability for
//! remaining coverage (LRC) from premiums received less revenue, and the
//! liability for incurred claims (LIC) at its fulfilment cash flows.
//!
//! Each period, in this order:
//!
//! 1. premiums received add to the LRC, and acquisition cash flows paid
//!    reduce it, unless they are expensed when paid (paragraph 59(a));
//! 2. insurance revenue is the expected premium receipts not yet
//!    recognised, allocated by the revenue pattern's weights over the
//!    remaining coverage (B126; the passage of time is equal weights), and
//!    the acquisition cash flows are amortised on the same pattern;
//! 3. claims incurred in the period enter the LIC at their fulfilment cash
//!    flows; the LIC's opening present value unwinds at the LIC curve (an
//!    insurance finance expense); paid claims leave it; what the closing
//!    LIC leaves after these is a change for past service (paragraph
//!    103(b)(iii));
//! 4. when facts and circumstances indicate the group is onerous
//!    ([`PaaInputs::with_remaining_fulfilment`]), the loss component is the
//!    GMM fulfilment cash flows of the remaining coverage less the LRC
//!    excluding it (paragraph 57); its change is a loss or a reversal.
//!    Without that input the loss component runs off with the revenue
//!    pattern, as a reversal.
//!
//! No significant financing component is assumed, so the LRC does not
//! accrete (paragraph 56); the LIC is discounted at
//! [`PaaOptions::lic_curve`] when given, not when the claims are paid within
//! a year (paragraph 59(b)). Its present value and RA can come from a
//! reserving fit: [`Lic::from_reserves`].
//!
//! No IASB example could be read for it from a cloud session (Example 10
//! is beyond the reachable text), so the tests are worked by hand.
//!
//! ```
//! use prospicio_reserving::ifrs17::paa::{Lic, PaaGroup, PaaInputs, PaaOptions};
//!
//! // A two-year cover, premium 1,000 at once, acquisition 100, both
//! // recognised evenly; claims of 300 incurred in year 1, 200 of them paid.
//! let mut group = PaaGroup::recognise(1_000.0, 100.0, PaaOptions::evenly(2))?;
//! let year = group.roll_forward(
//!     &PaaInputs::new(1_000.0, 100.0, 300.0, 200.0, Lic::new(100.0, 10.0)),
//! )?;
//! assert_eq!(year.insurance_revenue, 500.0);
//! // 300 incurred, 10 of RA for past service, 50 amortised.
//! assert_eq!(year.insurance_service_expenses(), 360.0);
//! assert_eq!(year.closing.lrc_excluding_loss_component, 450.0);
//! # Ok::<(), prospicio_reserving::Error>(())
//! ```

use super::{CoverageLine, CoverageRow, invalid};
use crate::discount::{DiscountCurve, DiscountedReserves, RiskAdjustmentResult};
use crate::error::Result;

/// The liability for incurred claims: the present value of their future
/// payments and their RA.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Lic {
    /// The present value of the incurred claims' future payments (their
    /// undiscounted amount when they are not discounted).
    pub present_value: f64,
    /// The RA for non-financial risk on them.
    pub risk_adjustment: f64,
}

impl Lic {
    /// A LIC of this present value and RA.
    pub fn new(present_value: f64, risk_adjustment: f64) -> Self {
        Self {
            present_value,
            risk_adjustment,
        }
    }

    /// The LIC from a reserving fit's discounted unpaid claims and its RA
    /// ([`DiscountedReserves::risk_adjustment`]): their best estimate and
    /// the RA's amount.
    pub fn from_reserves(reserves: &DiscountedReserves, ra: &RiskAdjustmentResult) -> Self {
        Self::new(reserves.best_estimate, ra.amount)
    }

    /// The LIC's fulfilment cash flows.
    pub fn total(&self) -> f64 {
        self.present_value + self.risk_adjustment
    }
}

/// The settings of a group under the PAA.
#[derive(Debug, Clone, PartialEq)]
pub struct PaaOptions {
    /// The weights of the expected premium receipts by period of coverage,
    /// from the first (B126): equal weights for the passage of time, or the
    /// expected pattern of incurred claims.
    pub revenue_pattern: Vec<f64>,
    /// Expense the acquisition cash flows when paid (paragraph 59(a), a
    /// coverage of a year or less) instead of amortising them.
    pub expense_acquisition: bool,
    /// The curve that discounts the LIC, maturities from each period end;
    /// its unwind over a period is the LIC's finance expense. `None` when
    /// the LIC is not discounted.
    pub lic_curve: Option<DiscountCurve>,
    /// The length of a period in years.
    pub period_years: f64,
}

impl PaaOptions {
    /// `periods` periods of a year with equal weights, acquisition
    /// amortised, the LIC undiscounted.
    pub fn evenly(periods: usize) -> Self {
        Self {
            revenue_pattern: vec![1.0; periods],
            expense_acquisition: false,
            lic_curve: None,
            period_years: 1.0,
        }
    }
}

/// What happened in one period under the PAA.
#[derive(Debug, Clone, PartialEq)]
pub struct PaaInputs {
    /// Premiums received.
    pub premiums_received: f64,
    /// Acquisition cash flows paid.
    pub acquisition_paid: f64,
    /// The fulfilment cash flows of the claims and expenses incurred in the
    /// period, at the period end.
    pub claims_incurred: f64,
    /// Claims and expenses paid.
    pub claims_paid: f64,
    /// The LIC at the end of the period.
    pub lic: Lic,
    /// The expected premium receipts of the whole coverage, when revised;
    /// `None` keeps them.
    pub expected_premiums: Option<f64>,
    /// The GMM fulfilment cash flows of the remaining coverage (future
    /// outflows and their RA less future premiums), when facts and
    /// circumstances indicate the group is onerous (paragraph 57).
    pub remaining_fulfilment: Option<f64>,
}

impl PaaInputs {
    /// A period's premiums, acquisition cash flows, claims incurred and
    /// paid, and the closing LIC.
    pub fn new(
        premiums_received: f64,
        acquisition_paid: f64,
        claims_incurred: f64,
        claims_paid: f64,
        lic: Lic,
    ) -> Self {
        Self {
            premiums_received,
            acquisition_paid,
            claims_incurred,
            claims_paid,
            lic,
            expected_premiums: None,
            remaining_fulfilment: None,
        }
    }

    /// With the expected premium receipts revised.
    pub fn with_expected_premiums(mut self, expected: f64) -> Self {
        self.expected_premiums = Some(expected);
        self
    }

    /// With the onerous test's fulfilment cash flows of the remaining
    /// coverage.
    pub fn with_remaining_fulfilment(mut self, fulfilment: f64) -> Self {
        self.remaining_fulfilment = Some(fulfilment);
        self
    }
}

/// The balances of a PAA group at a period end.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PaaBalances {
    /// The LRC excluding the loss component.
    pub lrc_excluding_loss_component: f64,
    /// The loss component.
    pub loss_component: f64,
    /// The LIC.
    pub lic: Lic,
}

impl PaaBalances {
    /// The liability for remaining coverage, the loss component included.
    pub fn lrc(&self) -> f64 {
        self.lrc_excluding_loss_component + self.loss_component
    }

    /// The group's liability, LRC and LIC.
    pub fn liability(&self) -> f64 {
        self.lrc() + self.lic.total()
    }
}

/// One period of a PAA group.
#[derive(Debug, Clone, PartialEq)]
pub struct PaaPeriod {
    /// The period, from 1.
    pub period: usize,
    /// The reconciliation by remaining coverage and incurred claims
    /// (paragraph 100), from `Opening` to `Closing`; the incurred claims'
    /// column is the LIC, its RA included.
    pub by_coverage: Vec<CoverageRow>,
    /// Insurance revenue.
    pub insurance_revenue: f64,
    /// Claims and expenses incurred in the period.
    pub incurred_claims: f64,
    /// Changes in the LIC for past service.
    pub past_service: f64,
    /// Acquisition cash flows amortised, or expensed when paid.
    pub acquisition_expense: f64,
    /// Losses on the onerous group.
    pub onerous_losses: f64,
    /// Reversals of those losses.
    pub loss_reversals: f64,
    /// Insurance finance expenses: the LIC's unwind.
    pub finance_expenses: f64,
    /// The balances at the end of the period.
    pub closing: PaaBalances,
}

impl PaaPeriod {
    /// Insurance service expenses.
    pub fn insurance_service_expenses(&self) -> f64 {
        self.incurred_claims + self.past_service + self.acquisition_expense + self.onerous_losses
            - self.loss_reversals
    }

    /// Insurance service result.
    pub fn insurance_service_result(&self) -> f64 {
        self.insurance_revenue - self.insurance_service_expenses()
    }

    /// Profit or loss: the service result less finance expenses.
    pub fn profit(&self) -> f64 {
        self.insurance_service_result() - self.finance_expenses
    }

    /// The reconciliation row of a line.
    pub fn coverage(&self, line: CoverageLine) -> Option<&CoverageRow> {
        self.by_coverage.iter().find(|r| r.line == line)
    }
}

/// A group of contracts under the premium allocation approach.
#[derive(Debug, Clone, PartialEq)]
pub struct PaaGroup {
    pub(super) options: PaaOptions,
    pub(super) expected_premiums: f64,
    pub(super) acquisition: f64,
    pub(super) period: usize,
    pub(super) revenue_to_date: f64,
    pub(super) amortised_to_date: f64,
    pub(super) balances: PaaBalances,
}

impl PaaGroup {
    /// Recognise a group with its expected premium receipts and acquisition
    /// cash flows over the whole coverage. The LRC starts at nil: premiums
    /// and acquisition cash flows enter it as they are received and paid.
    ///
    /// Fails when the amounts are not finite, the pattern is empty, has a
    /// negative or non-finite weight or sums to nil, or the period length
    /// is not positive.
    pub fn recognise(
        expected_premiums: f64,
        acquisition: f64,
        options: PaaOptions,
    ) -> Result<Self> {
        if !(expected_premiums.is_finite() && acquisition.is_finite()) {
            return Err(invalid(
                "the expected premiums and acquisition cash flows must be finite",
            ));
        }
        let w = &options.revenue_pattern;
        if w.is_empty()
            || w.iter().any(|x| !(x.is_finite() && *x >= 0.0))
            || w.iter().sum::<f64>() <= 0.0
        {
            return Err(invalid(
                "the revenue pattern needs non-negative finite weights with a positive sum",
            ));
        }
        if !(options.period_years.is_finite() && options.period_years > 0.0) {
            return Err(invalid("the period length must be finite and positive"));
        }
        Ok(Self {
            options,
            expected_premiums,
            acquisition,
            period: 0,
            revenue_to_date: 0.0,
            amortised_to_date: 0.0,
            balances: PaaBalances::default(),
        })
    }

    /// The balances after the last period rolled forward.
    pub fn balances(&self) -> PaaBalances {
        self.balances
    }

    /// Roll the group forward over one period.
    ///
    /// Fails when the inputs are not finite or the coverage has ended.
    pub fn roll_forward(&mut self, inputs: &PaaInputs) -> Result<PaaPeriod> {
        let pattern = &self.options.revenue_pattern;
        if self.period >= pattern.len() {
            return Err(invalid("the group's coverage has ended"));
        }
        let values = [
            inputs.premiums_received,
            inputs.acquisition_paid,
            inputs.claims_incurred,
            inputs.claims_paid,
            inputs.lic.present_value,
            inputs.lic.risk_adjustment,
            inputs.expected_premiums.unwrap_or(0.0),
            inputs.remaining_fulfilment.unwrap_or(0.0),
        ];
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("the PAA inputs must be finite"));
        }
        if inputs.lic.risk_adjustment < 0.0 {
            return Err(invalid("the LIC's RA must not be negative"));
        }
        if let Some(expected) = inputs.expected_premiums {
            self.expected_premiums = expected;
        }
        let remaining: f64 = pattern[self.period..].iter().sum();
        let share = if remaining > 0.0 {
            pattern[self.period] / remaining
        } else {
            1.0
        };
        let opening = self.balances;

        // Revenue and amortisation on the pattern over what remains.
        let revenue = (self.expected_premiums - self.revenue_to_date) * share;
        let amortised = if self.options.expense_acquisition {
            0.0
        } else {
            (self.acquisition - self.amortised_to_date) * share
        };
        let acquisition_expense = if self.options.expense_acquisition {
            inputs.acquisition_paid
        } else {
            amortised
        };
        let acquisition_in_lrc = if self.options.expense_acquisition {
            0.0
        } else {
            inputs.acquisition_paid
        };
        let lrc = opening.lrc_excluding_loss_component + inputs.premiums_received
            - acquisition_in_lrc
            + amortised
            - revenue;

        // The LIC.
        let l = self.options.period_years;
        let finance = match &self.options.lic_curve {
            Some(curve) => opening.lic.present_value * (1.0 / curve.factor(l) - 1.0),
            None => 0.0,
        };
        let past_service = inputs.lic.total()
            - (opening.lic.total() + inputs.claims_incurred + finance - inputs.claims_paid);

        // The loss component (paragraph 57).
        let lc = match inputs.remaining_fulfilment {
            Some(fcf) => (fcf - lrc).max(0.0),
            None => opening.loss_component * (1.0 - share),
        };
        let change = lc - opening.loss_component;
        let (losses, reversals) = if change >= 0.0 {
            (change, 0.0)
        } else {
            (0.0, -change)
        };

        let closing = PaaBalances {
            lrc_excluding_loss_component: lrc,
            loss_component: lc,
            lic: inputs.lic,
        };
        let cov = |line, x, lc, lic| CoverageRow {
            line,
            excluding_loss_component: x,
            loss_component: lc,
            incurred_claims: lic,
        };
        let by_coverage = vec![
            cov(
                CoverageLine::Opening,
                opening.lrc_excluding_loss_component,
                opening.loss_component,
                opening.lic.total(),
            ),
            cov(CoverageLine::InsuranceRevenue, -revenue, 0.0, 0.0),
            cov(
                CoverageLine::IncurredClaims,
                0.0,
                0.0,
                inputs.claims_incurred,
            ),
            cov(CoverageLine::PastService, 0.0, 0.0, past_service),
            cov(CoverageLine::AcquisitionAmortisation, amortised, 0.0, 0.0),
            cov(CoverageLine::OnerousLosses, 0.0, losses, 0.0),
            cov(CoverageLine::LossReversals, 0.0, -reversals, 0.0),
            cov(CoverageLine::FinanceExpenses, 0.0, 0.0, finance),
            cov(
                CoverageLine::PremiumsReceived,
                inputs.premiums_received,
                0.0,
                0.0,
            ),
            cov(CoverageLine::AcquisitionPaid, -acquisition_in_lrc, 0.0, 0.0),
            cov(
                CoverageLine::ClaimsAndExpensesPaid,
                0.0,
                0.0,
                -inputs.claims_paid,
            ),
            cov(
                CoverageLine::Closing,
                closing.lrc_excluding_loss_component,
                closing.loss_component,
                closing.lic.total(),
            ),
        ];

        self.period += 1;
        self.revenue_to_date += revenue;
        self.amortised_to_date += amortised;
        self.balances = closing;
        Ok(PaaPeriod {
            period: self.period,
            by_coverage,
            insurance_revenue: revenue,
            incurred_claims: inputs.claims_incurred,
            past_service,
            acquisition_expense,
            onerous_losses: losses,
            loss_reversals: reversals,
            finance_expenses: finance,
            closing,
        })
    }
}
