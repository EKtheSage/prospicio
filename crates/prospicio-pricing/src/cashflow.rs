//! A monthly cash-flow engine over simulations (#479): a collateralised
//! cover's trust (collateral) account and the investor's operating account,
//! month by month, for each simulation's ultimate loss ratio and
//! commission rate. See `docs/design/time-value.md`, decisions 13 to 20.
//!
//! [`CashflowTerms`] holds the deal: the premium, monthly cumulative
//! written, earned and paid patterns ([`MonthlyPatterns`]), the commutation
//! month and one rule per account. Every rule is a parameter:
//!
//! - the collateral required each month ([`Requirement`]: loads on named
//!   bases, a floor and a switch to the limit), and a second requirement at
//!   commutation;
//! - the premium paid into the trust ([`PremiumFlow`]: a collected rate, a
//!   lag and adjustments);
//! - when and how capital is called ([`Funding`]: the dates, the rule and
//!   which account pays);
//! - when the excess over the requirement leaves the trust ([`Timing`]) and
//!   the interest it earns ([`Interest`]);
//! - the operating account ([`Operating`]: opening capital, expenses on the
//!   requirement, a buffer, when equity is released, a fee at
//!   commutation).
//!
//! [`CashflowTerms::run`] gives each simulation's monthly investor flows
//! and totals ([`Cashflows`]); its flows at months `0..T` are what
//! [`npv_rows`] and [`irr_rows`] take with [`TimeUnit::Months`].
//! [`CashflowTerms::trace`] gives every account line of one simulation, to
//! check a deal month by month.
//!
//! Over the term, money is conserved: what the investor gets back net of
//! what it put in, plus the fee, the claims paid, the expenses and the
//! commutation payment, equals the premium and adjustments paid into the
//! trust plus its interest ([`Trace::conservation_error`]).

use prospicio_core::{Error, Result};

use crate::payout::{Interpolation, PaymentPattern};
use crate::returns::{Irr, IrrOptions, TieredFee, TimeUnit, irr_rows, npv_rows};

/// Monthly cumulative shares of the premium written and earned and of the
/// ultimate loss paid, each at the end of month `t = 0, 1, ...` from
/// inception. All three have the same length, which must reach the
/// commutation month plus the operating account's buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct MonthlyPatterns {
    /// The share of the premium written by the end of each month.
    pub written: Vec<f64>,
    /// The share of the premium earned by the end of each month.
    pub earned: Vec<f64>,
    /// The share of the ultimate loss paid by the end of each month.
    pub paid: Vec<f64>,
}

impl MonthlyPatterns {
    /// The earned and paid patterns from a written one, a policy term and a
    /// payment pattern by development age (#601). The premium written in
    /// month `s` is written at its start and earned evenly over the
    /// `policy_months` months from it, so the share earned by the end of
    /// month `t` is `Σ_s ΔW_s min((t − s + 1) / policy_months, 1)`. Losses
    /// occur as premium is earned, in the middle of their month, and are
    /// paid by `paid`, measured from the accident date and read between
    /// its times as `how` says ([`PaymentPattern::paid_by`]): the share of
    /// the ultimate loss paid by the end of month `t` is
    /// `Σ_e ΔE_e paid_by((t − e + ½) / 12)`.
    ///
    /// All premium written at inception (`written` all 1) is a single
    /// policy; written evenly over a year, policies written through it.
    ///
    /// ```
    /// use prospicio_pricing::cashflow::MonthlyPatterns;
    /// use prospicio_pricing::payout::{Interpolation, PaymentPattern};
    ///
    /// // One annual policy at inception, paid evenly over three years from
    /// // the accident date.
    /// let paid = PaymentPattern::new(vec![1.0, 2.0, 3.0], vec![1.0 / 3.0; 3]).unwrap();
    /// let p = MonthlyPatterns::from_annual(vec![1.0; 60], 12, &paid, Interpolation::Linear)
    ///     .unwrap();
    /// assert!((p.earned[5] - 0.5).abs() < 1e-15 && p.earned[11] == 1.0);
    /// // Half a month of the first month's losses, over 36 months to pay.
    /// assert!((p.paid[0] - 0.5 / 36.0 / 12.0).abs() < 1e-15);
    /// assert!((p.paid[59] - 1.0).abs() < 1e-12);
    /// ```
    pub fn from_annual(
        written: Vec<f64>,
        policy_months: usize,
        paid: &PaymentPattern,
        how: Interpolation,
    ) -> Result<Self> {
        if policy_months == 0 {
            return Err(Error::InvalidParameter {
                name: "policy_months",
                value: 0.0,
                reason: "must be at least 1",
            });
        }
        if written.is_empty() || written.iter().any(|w| !w.is_finite()) {
            return Err(Error::Data(
                "the written pattern must be finite, with at least one month".into(),
            ));
        }
        let n = written.len();
        let increments = |v: &[f64]| -> Vec<f64> {
            (0..v.len())
                .map(|t| v[t] - if t == 0 { 0.0 } else { v[t - 1] })
                .collect()
        };
        let dw = increments(&written);
        let earned: Vec<f64> = (0..n)
            .map(|t| {
                (0..=t)
                    .map(|s| dw[s] * ((t - s + 1) as f64 / policy_months as f64).min(1.0))
                    .sum()
            })
            .collect();
        let de = increments(&earned);
        // The share paid of a month's losses after k whole months.
        let lag: Vec<f64> = (0..n)
            .map(|k| paid.paid_by((k as f64 + 0.5) / 12.0, how))
            .collect();
        let paid_share: Vec<f64> = (0..n)
            .map(|t| (0..=t).map(|e| de[e] * lag[t - e]).sum())
            .collect();
        Self::new(written, earned, paid_share)
    }

    /// From the three cumulative patterns, finite and of one length.
    pub fn new(written: Vec<f64>, earned: Vec<f64>, paid: Vec<f64>) -> Result<Self> {
        let p = MonthlyPatterns {
            written,
            earned,
            paid,
        };
        p.check()?;
        Ok(p)
    }

    /// The number of months.
    pub fn len(&self) -> usize {
        self.written.len()
    }

    /// Whether there are no months.
    pub fn is_empty(&self) -> bool {
        self.written.is_empty()
    }

    fn check(&self) -> Result<()> {
        let n = self.written.len();
        if n == 0 || self.earned.len() != n || self.paid.len() != n {
            return Err(Error::Data(format!(
                "written, earned and paid patterns have {}, {} and {} months; they need one \
                 positive length",
                n,
                self.earned.len(),
                self.paid.len()
            )));
        }
        if self
            .written
            .iter()
            .chain(&self.earned)
            .chain(&self.paid)
            .any(|x| !x.is_finite())
        {
            return Err(Error::Data("patterns must be finite".into()));
        }
        Ok(())
    }
}

/// What a [`Requirement`] term is a load on, at the end of month `t`, with
/// `P` the premium, `u` the simulation's ultimate loss ratio and `W`, `E`
/// and `C` the premium written, the premium earned and the loss paid to
/// date.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Base {
    /// The unearned premium `W − E`.
    UnearnedPremium,
    /// The reserve on earned premium `u E − C`.
    ///
    /// Early in the term it is small, since little premium is earned: a
    /// requirement on it alone releases the premium collected ahead of
    /// earning to the investor at once, who then pays the claims later, so
    /// the investor's flows start positive, can have several internal rates
    /// of return ([`Irr::sign_changes`] above 1) and the one found is
    /// usually meaningless ([`Cashflows::paid_first`] flags such runs). A
    /// collateralised fund usually wants [`Outstanding`](Self::Outstanding),
    /// with [`UnearnedPremium`](Self::UnearnedPremium) beside it.
    Reserve,
    /// The outstanding ultimate loss `u P − C`, the usual base of a
    /// collateralised fund's requirement.
    Outstanding,
    /// The premium written `W`.
    Written,
    /// The premium earned `E`.
    Earned,
    /// One unit: the load is an amount.
    Fixed,
}

/// One term of a [`Requirement`]: `load × base`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Term {
    /// The base the load applies to.
    pub base: Base,
    /// The load, or the amount for [`Base::Fixed`].
    pub load: f64,
}

impl Term {
    /// `load × base`.
    pub fn new(base: Base, load: f64) -> Self {
        Term { base, load }
    }
}

/// A switch to the limit: when `trigger × u ≥ cap`, the requirement is the
/// limit not yet paid, `cap × W − C`, in place of its terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimitSwitch {
    /// The load on the ultimate loss ratio that is compared with the cap.
    pub trigger: f64,
    /// The cap on the loss ratio, the limit as a share of written premium.
    pub cap: f64,
}

/// The collateral required at the end of a month: the sum of its terms,
/// or the limit not yet paid when a [`LimitSwitch`] fires, never below the
/// floor.
#[derive(Debug, Clone, PartialEq)]
pub struct Requirement {
    /// The loaded bases, summed.
    pub terms: Vec<Term>,
    /// The least requirement.
    pub floor: f64,
    /// An optional switch to the limit.
    pub limit: Option<LimitSwitch>,
}

impl Requirement {
    /// The sum of `terms`, floored at 0, with no limit switch.
    pub fn new(terms: Vec<Term>) -> Self {
        Requirement {
            terms,
            floor: 0.0,
            limit: None,
        }
    }

    /// The same with a switch to the limit `cap × W − C` when
    /// `trigger × u ≥ cap`.
    pub fn with_limit(mut self, trigger: f64, cap: f64) -> Self {
        self.limit = Some(LimitSwitch { trigger, cap });
        self
    }

    /// The same with another floor.
    pub fn with_floor(mut self, floor: f64) -> Self {
        self.floor = floor;
        self
    }

    fn check(&self, name: &'static str) -> Result<()> {
        let finite = self.terms.iter().all(|t| t.load.is_finite())
            && self.floor.is_finite()
            && self
                .limit
                .is_none_or(|l| l.trigger.is_finite() && l.cap.is_finite());
        if !finite {
            return Err(Error::Data(format!(
                "{name}: loads, floor, trigger and cap must be finite"
            )));
        }
        Ok(())
    }

    fn at(&self, s: &Month) -> f64 {
        let r = match self.limit {
            Some(l) if l.trigger * s.u >= l.cap => l.cap * s.written - s.paid,
            _ => self
                .terms
                .iter()
                .map(|t| {
                    t.load
                        * match t.base {
                            Base::UnearnedPremium => s.written - s.earned,
                            Base::Reserve => s.u * s.earned - s.paid,
                            Base::Outstanding => s.u * s.premium - s.paid,
                            Base::Written => s.written,
                            Base::Earned => s.earned,
                            Base::Fixed => 1.0,
                        }
                })
                .sum(),
        };
        r.max(self.floor)
    }
}

/// The amounts to date at the end of a month, for one simulation.
struct Month {
    premium: f64,
    u: f64,
    written: f64,
    earned: f64,
    paid: f64,
}

/// A set of months from inception, `0` the first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dates {
    /// Every `months` months from month `first`: `first`, `first + months`,
    /// ...
    Every {
        /// The interval, at least 1.
        months: usize,
        /// The first month.
        first: usize,
    },
    /// The months listed.
    Months(Vec<usize>),
}

impl Dates {
    /// Every month.
    pub fn monthly() -> Self {
        Dates::Every {
            months: 1,
            first: 0,
        }
    }

    /// Every three months from month 0.
    pub fn quarterly() -> Self {
        Dates::Every {
            months: 3,
            first: 0,
        }
    }

    fn contains(&self, t: usize) -> bool {
        match self {
            Dates::Every { months, first } => t >= *first && (t - first) % months == 0,
            Dates::Months(m) => m.contains(&t),
        }
    }

    fn check(&self, name: &'static str) -> Result<()> {
        if let Dates::Every { months: 0, .. } = self {
            return Err(Error::Data(format!("{name}: an interval of 0 months")));
        }
        Ok(())
    }
}

/// When an account releases what it holds above what it must keep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Timing {
    /// At the end of every month.
    Monthly,
    /// At the end of the months given.
    At(Dates),
    /// Only at commutation.
    AtCommutation,
}

impl Timing {
    fn now(&self, t: usize) -> bool {
        match self {
            Timing::Monthly => true,
            Timing::At(d) => d.contains(t),
            Timing::AtCommutation => false,
        }
    }

    fn check(&self, name: &'static str) -> Result<()> {
        match self {
            Timing::At(d) => d.check(name),
            _ => Ok(()),
        }
    }
}

/// The obligation a [`FundingRule::ObligationIncrease`] compares with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObligationReference {
    /// The obligation at the previous funding date.
    PreviousDate,
    /// The highest obligation at any earlier funding date.
    RunningMax,
}

/// How much capital a funding date calls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FundingRule {
    /// The rise in the obligation `S_t − premium_offset × W_{t−lag}`, the
    /// requirement net of the premium collected to date (lagged by the
    /// [`PremiumFlow`]'s lag), over the reference; never negative.
    ObligationIncrease {
        /// The share of written premium set against the requirement.
        premium_offset: f64,
        /// The obligation the rise is measured from.
        reference: ObligationReference,
    },
    /// What the trust lacks: the requirement less the balance after the
    /// month's premium, adjustments and claims; never negative.
    Shortfall,
}

/// Which account a capital call is paid from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FundingSource {
    /// The investor pays every call.
    Investor,
    /// The operating account pays what it holds at the start of the month,
    /// the investor the rest.
    OperatingFirst,
}

/// When and how capital is paid into the trust.
#[derive(Debug, Clone, PartialEq)]
pub struct Funding {
    /// The months on which capital may be called.
    pub dates: Dates,
    /// How much a date calls.
    pub rule: FundingRule,
    /// Which account pays.
    pub source: FundingSource,
}

impl Default for Funding {
    /// Every month, the shortfall, from the investor.
    fn default() -> Self {
        Funding {
            dates: Dates::monthly(),
            rule: FundingRule::Shortfall,
            source: FundingSource::Investor,
        }
    }
}

/// What the trust's interest is earned on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterestBasis {
    /// The average of the month's opening and closing balances.
    #[default]
    Average,
    /// The opening balance.
    Opening,
    /// The closing balance, before the month's release.
    Closing,
}

/// The trust's interest: `annual_yield / 12` a month on the basis.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Interest {
    /// The annual yield.
    pub annual_yield: f64,
    /// The balance it is earned on.
    pub basis: InterestBasis,
}

/// What an [`Adjustment`] reads the written and earned premium at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdjustmentBase {
    /// The last month of the patterns: the ultimate written and earned.
    #[default]
    Ultimate,
    /// The adjustment's month.
    ToDate,
}

/// A premium adjustment paid into the trust at the end of `month`:
/// `written_rate × W − c × E` with `c` the simulation's commission rate
/// when `less_commission`, `W` and `E` read as `base` says. A negative
/// amount is paid out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adjustment {
    /// The month it is paid.
    pub month: usize,
    /// The rate on written premium.
    pub written_rate: f64,
    /// Whether the commission on earned premium is taken off.
    pub less_commission: bool,
    /// When the written and earned premium are read.
    pub base: AdjustmentBase,
}

/// The premium paid into the trust: `collected_rate × ΔW_{t−lag}` each
/// month, or `(collected_rate − c) × ΔW_{t−lag}` with `c` the simulation's
/// commission rate when `deduct_commission`, and the adjustments.
///
/// Without `deduct_commission` the commission rate given to
/// [`CashflowTerms::run`] does not reduce the premium paid in: it is read
/// only by adjustments with `less_commission` and by the fee. A flat
/// commission `c` is then `collected_rate = 1 − c`; a commission that
/// differs by simulation, such as a sliding scale's, needs
/// `deduct_commission`.
#[derive(Debug, Clone, PartialEq)]
pub struct PremiumFlow {
    /// The share of each month's written premium paid in.
    pub collected_rate: f64,
    /// Whether each simulation's commission rate is taken off the share
    /// collected.
    pub deduct_commission: bool,
    /// Months from writing to collection.
    pub lag: usize,
    /// Adjustments at given months.
    pub adjustments: Vec<Adjustment>,
}

impl Default for PremiumFlow {
    /// All written premium collected in the month written.
    fn default() -> Self {
        PremiumFlow {
            collected_rate: 1.0,
            deduct_commission: false,
            lag: 0,
            adjustments: Vec::new(),
        }
    }
}

/// What the operating account keeps back before releasing equity: the
/// claims paid and the expenses of the next `months` months, each when
/// its flag is set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Buffer {
    /// The months ahead.
    pub months: usize,
    /// Whether the claims paid in them count.
    pub paid: bool,
    /// Whether the expenses in them count.
    pub expenses: bool,
}

/// What a performance fee's [`TieredFee`] reads, with the profit the
/// investor's inflows before the fee less its outflows; the fee is the
/// tiered amount times the divisor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeeBasis {
    /// The profit.
    Profit,
    /// The profit over the premium.
    ProfitOverPremium,
    /// The profit over the capital funded plus the opening capital (no fee
    /// when that is not positive).
    ProfitOverCapital,
}

/// A fee paid out of the operating account at commutation.
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceFee {
    /// The tiers.
    pub tiers: TieredFee,
    /// What they read.
    pub basis: FeeBasis,
}

/// The investor's operating account.
#[derive(Debug, Clone, PartialEq)]
pub struct Operating {
    /// The opening capital's fixed part, paid in at month 0.
    pub opening_fixed: f64,
    /// The opening capital's share of the premium.
    pub opening_share: f64,
    /// The annual expense rate on the month's requirement (the cost of
    /// collateral, account maintenance), charged a twelfth a month.
    pub expense_rate: f64,
    /// What is kept back before equity is released.
    pub buffer: Buffer,
    /// When equity above the buffer is released to the investor.
    pub release: Timing,
    /// A fee at commutation, if any.
    pub fee: Option<PerformanceFee>,
}

impl Default for Operating {
    /// No opening capital, no expenses, no buffer, equity released
    /// monthly, no fee.
    fn default() -> Self {
        Operating {
            opening_fixed: 0.0,
            opening_share: 0.0,
            expense_rate: 0.0,
            buffer: Buffer::default(),
            release: Timing::Monthly,
            fee: None,
        }
    }
}

/// A collateralised deal's monthly cash flows: its terms and one rule per
/// account, each a public field with the default the constructor sets.
///
/// Month `t` runs from `0` to the commutation month `T − 1`. In each month
/// the trust takes the capital called, the premium collected and any
/// adjustment, pays the month's claims, earns interest and releases what it
/// holds above the requirement to the operating account. The operating
/// account pays the expenses on the requirement and releases to the
/// investor what it holds above its buffer. In month `T − 1` the
/// requirement is the commutation requirement, which is paid to the cedant;
/// the trust's balance above it (or below it, which the operating account
/// makes good) goes to the operating account, which pays the fee and
/// distributes the rest.
///
/// ```
/// use prospicio_pricing::cashflow::{Base, CashflowTerms, MonthlyPatterns, Requirement, Term};
///
/// // A premium of 100 written and collected at once, earned and paid
/// // evenly over three months; collateral is the reserve on earned
/// // premium, the deal commutes after month 2 at the outstanding loss.
/// let patterns = MonthlyPatterns::new(
///     vec![1.0; 3],
///     vec![1.0 / 3.0, 2.0 / 3.0, 1.0],
///     vec![1.0 / 3.0, 2.0 / 3.0, 1.0],
/// ).unwrap();
/// let mut terms = CashflowTerms::new(
///     100.0, 3, patterns, Requirement::new(vec![Term::new(Base::Reserve, 1.0)]),
/// ).unwrap();
/// terms.commutation_requirement = Requirement::new(vec![Term::new(Base::Outstanding, 1.0)]);
/// // At a loss ratio of 60% each month pays 20 and the reserve is 0: the
/// // trust releases the 100 collected less the claims paid, 80 at month 0.
/// let round = |v: &[f64]| v.iter().map(|x| (x * 1e9).round() / 1e9).collect::<Vec<_>>();
/// let out = terms.run(&[0.6], &[0.0]).unwrap();
/// assert_eq!(round(out.flows()), [80.0, -20.0, -20.0]);
/// // At 150% the reserve is 0 too, and the investor funds each month's 50
/// // of claims beyond the premium.
/// let out = terms.run(&[1.5], &[0.0]).unwrap();
/// assert_eq!(round(out.flows()), [50.0, -50.0, -50.0]);
/// assert_eq!(round(out.funded()), [100.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CashflowTerms {
    /// The premium `P`.
    pub premium: f64,
    /// The commutation month count `T`: the last month is `T − 1`.
    pub commutation_month: usize,
    /// The monthly written, earned and paid patterns.
    pub patterns: MonthlyPatterns,
    /// The collateral required in months `0..T − 1` (and in the buffer's
    /// months after).
    pub requirement: Requirement,
    /// The collateral required at commutation, paid to the cedant.
    pub commutation_requirement: Requirement,
    /// The premium paid into the trust.
    pub premium_flow: PremiumFlow,
    /// Capital calls.
    pub funding: Funding,
    /// When the trust releases its excess over the requirement.
    pub collateral_release: Timing,
    /// The trust's interest.
    pub interest: Interest,
    /// The operating account.
    pub operating: Operating,
}

impl CashflowTerms {
    /// A deal with premium `premium`, commuting after month
    /// `commutation_month − 1`, and the default rules: the commutation
    /// requirement the same as `requirement`, all premium collected in the
    /// month written, the shortfall called from the investor every month,
    /// the excess released monthly, no interest, and an operating account
    /// with no opening capital, expenses, buffer or fee that releases
    /// monthly.
    pub fn new(
        premium: f64,
        commutation_month: usize,
        patterns: MonthlyPatterns,
        requirement: Requirement,
    ) -> Result<Self> {
        let terms = CashflowTerms {
            premium,
            commutation_month,
            patterns,
            commutation_requirement: requirement.clone(),
            requirement,
            premium_flow: PremiumFlow::default(),
            funding: Funding::default(),
            collateral_release: Timing::Monthly,
            interest: Interest::default(),
            operating: Operating::default(),
        };
        terms.check()?;
        Ok(terms)
    }

    fn check(&self) -> Result<()> {
        if !(self.premium.is_finite() && self.premium > 0.0) {
            return Err(Error::InvalidParameter {
                name: "premium",
                value: self.premium,
                reason: "must be finite and positive",
            });
        }
        if self.commutation_month == 0 {
            return Err(Error::Data(
                "the commutation month must be at least 1".into(),
            ));
        }
        self.patterns.check()?;
        let needed = self.commutation_month + self.operating.buffer.months;
        if self.patterns.len() < needed {
            return Err(Error::Data(format!(
                "the patterns have {} months; commutation after {} months and a buffer of {} \
                 need {needed}",
                self.patterns.len(),
                self.commutation_month,
                self.operating.buffer.months
            )));
        }
        self.requirement.check("requirement")?;
        self.commutation_requirement
            .check("commutation requirement")?;
        let f = &self.premium_flow;
        if !f.collected_rate.is_finite()
            || f.adjustments.iter().any(|a| !a.written_rate.is_finite())
        {
            return Err(Error::Data(
                "the collected rate and adjustment rates must be finite".into(),
            ));
        }
        if let Some(a) = f
            .adjustments
            .iter()
            .find(|a| a.month >= self.commutation_month)
        {
            return Err(Error::Data(format!(
                "an adjustment at month {} is not before commutation (months 0 to {})",
                a.month,
                self.commutation_month - 1
            )));
        }
        self.funding.dates.check("funding dates")?;
        if let FundingRule::ObligationIncrease { premium_offset, .. } = self.funding.rule {
            if !premium_offset.is_finite() {
                return Err(Error::InvalidParameter {
                    name: "premium_offset",
                    value: premium_offset,
                    reason: "must be finite",
                });
            }
        }
        self.collateral_release.check("collateral release")?;
        self.operating.release.check("equity release")?;
        let o = &self.operating;
        if ![
            self.interest.annual_yield,
            o.opening_fixed,
            o.opening_share,
            o.expense_rate,
        ]
        .iter()
        .all(|x| x.is_finite())
        {
            return Err(Error::Data(
                "the yield, opening capital and expense rate must be finite".into(),
            ));
        }
        Ok(())
    }

    /// Each simulation's cash flows, from its ultimate loss ratio `u[j]`
    /// and commission rate `commission[j]`. The commission reduces the
    /// premium paid in only under [`PremiumFlow::deduct_commission`].
    ///
    /// Row `j` of the result is simulation `j` of `u` and `commission`,
    /// which must come from the same simulation `j` as everything else in
    /// the deal: a reserving fit's `loss_ratio_draws`
    /// (`prospicio_reserving::UltimateDraws`) and a tower's terms on the
    /// same draws keep that order.
    pub fn run(&self, u: &[f64], commission: &[f64]) -> Result<Cashflows> {
        self.check()?;
        check_sims(u, commission)?;
        let t_n = self.commutation_month;
        let mut out = Cashflows::empty(t_n, u.len());
        let req = |_: usize, m: &Month| self.requirement.at(m);
        for (&u, &c) in u.iter().zip(commission) {
            out.push(self.simulate(u, c, &req, None));
        }
        Ok(out)
    }

    /// [`run`](Self::run) with each simulation's requirement given in
    /// months `0..T + B` (`B` the buffer's months), one simulation a row of
    /// a row-major matrix, in place of [`requirement`](Self::requirement).
    /// The commutation requirement is still read from its terms.
    pub fn run_with_requirement(
        &self,
        u: &[f64],
        commission: &[f64],
        requirement: &[f64],
    ) -> Result<Cashflows> {
        self.check()?;
        check_sims(u, commission)?;
        let width = self.commutation_month + self.operating.buffer.months;
        if requirement.len() != width * u.len() {
            return Err(Error::Data(format!(
                "{} requirement values for {} simulations of {width} months",
                requirement.len(),
                u.len()
            )));
        }
        if requirement.iter().any(|x| !x.is_finite()) {
            return Err(Error::Data("the requirement must be finite".into()));
        }
        let mut out = Cashflows::empty(self.commutation_month, u.len());
        for (j, (&u, &c)) in u.iter().zip(commission).enumerate() {
            let row = &requirement[j * width..(j + 1) * width];
            let req = |t: usize, _: &Month| row[t];
            out.push(self.simulate(u, c, &req, None));
        }
        Ok(out)
    }

    /// Every account line of one simulation, month by month.
    pub fn trace(&self, u: f64, commission: f64) -> Result<Trace> {
        self.check()?;
        check_sims(&[u], &[commission])?;
        let mut trace = Trace::new(self.commutation_month);
        let req = |_: usize, m: &Month| self.requirement.at(m);
        self.simulate(u, commission, &req, Some(&mut trace));
        Ok(trace)
    }

    fn month(&self, t: usize, u: f64) -> Month {
        let p = &self.patterns;
        Month {
            premium: self.premium,
            u,
            written: self.premium * p.written[t],
            earned: self.premium * p.earned[t],
            paid: u * self.premium * p.paid[t],
        }
    }

    fn simulate(
        &self,
        u: f64,
        c: f64,
        req: &dyn Fn(usize, &Month) -> f64,
        mut trace: Option<&mut Trace>,
    ) -> Sim {
        let t_n = self.commutation_month;
        let pr = self.premium;
        let pat = &self.patterns;
        let last = pat.len() - 1;
        let at = |v: &[f64], t: isize| if t < 0 { 0.0 } else { v[t as usize] };
        let written_inc = |t: isize| pr * (at(&pat.written, t) - at(&pat.written, t - 1));
        let paid_inc =
            |t: usize| u * pr * (at(&pat.paid, t as isize) - at(&pat.paid, t as isize - 1));
        let lag = self.premium_flow.lag as isize;
        let monthly_rate = self.interest.annual_yield / 12.0;
        let expense = |s: f64| self.operating.expense_rate / 12.0 * s;
        let buffer_months = self.operating.buffer.months;
        // The regular requirement over the term and the buffer's months.
        let requirement: Vec<f64> = (0..t_n + buffer_months)
            .map(|t| req(t, &self.month(t, u)))
            .collect();
        let commutation = self.commutation_requirement.at(&self.month(t_n - 1, u));

        let opening = self.operating.opening_fixed + self.operating.opening_share * pr;
        let mut collateral = 0.0;
        let mut operating = opening;
        let mut reference = 0.0_f64;
        let mut sim = Sim {
            flows: vec![0.0; t_n],
            funded: 0.0,
            opening,
            released: 0.0,
            distributed: 0.0,
            fee: 0.0,
            commutation,
        };
        sim.flows[0] = -opening;

        for t in 0..t_n {
            let final_month = t + 1 == t_n;
            let s = if final_month {
                commutation
            } else {
                requirement[t]
            };
            let start = collateral;
            let flow = &self.premium_flow;
            let collected = if flow.deduct_commission {
                flow.collected_rate - c
            } else {
                flow.collected_rate
            };
            let premium_in = collected * written_inc(t as isize - lag);
            let adjustment: f64 = self
                .premium_flow
                .adjustments
                .iter()
                .filter(|a| a.month == t)
                .map(|a| {
                    let k = match a.base {
                        AdjustmentBase::Ultimate => last,
                        AdjustmentBase::ToDate => t,
                    };
                    let less = if a.less_commission {
                        c * pr * pat.earned[k]
                    } else {
                        0.0
                    };
                    a.written_rate * pr * pat.written[k] - less
                })
                .sum();
            let paid = paid_inc(t);
            let before = start + premium_in + adjustment - paid;
            let call = if self.funding.dates.contains(t) {
                match self.funding.rule {
                    FundingRule::Shortfall => (s - before).max(0.0),
                    FundingRule::ObligationIncrease {
                        premium_offset,
                        reference: kind,
                    } => {
                        let ob = s - premium_offset * pr * at(&pat.written, t as isize - lag);
                        let call = (ob - reference).max(0.0);
                        reference = match kind {
                            ObligationReference::PreviousDate => ob,
                            ObligationReference::RunningMax => reference.max(ob),
                        };
                        call
                    }
                }
            } else {
                0.0
            };
            let from_operating = match self.funding.source {
                FundingSource::Investor => 0.0,
                FundingSource::OperatingFirst => call.min(operating.max(0.0)),
            };
            let from_investor = call - from_operating;
            operating -= from_operating;
            let end = before + call;
            let interest = monthly_rate
                * match self.interest.basis {
                    InterestBasis::Average => 0.5 * (start + end),
                    InterestBasis::Opening => start,
                    InterestBasis::Closing => end,
                };
            collateral = end + interest;
            let release = if final_month {
                collateral - s
            } else if self.collateral_release.now(t) {
                (collateral - s).max(0.0)
            } else {
                0.0
            };
            collateral -= release;
            let commutation_paid = if final_month { collateral } else { 0.0 };
            collateral -= commutation_paid;

            let expenses = expense(s);
            operating += release - expenses;
            let equity = if final_month {
                0.0
            } else if self.operating.release.now(t) {
                let b = &self.operating.buffer;
                let keep: f64 = (t + 1..=t + b.months)
                    .map(|k| {
                        let mut x = 0.0;
                        if b.paid {
                            x += paid_inc(k);
                        }
                        if b.expenses {
                            x += expense(requirement[k]);
                        }
                        x
                    })
                    .sum();
                (operating - keep).max(0.0)
            } else {
                0.0
            };
            operating -= equity;
            sim.funded += from_investor;
            sim.released += equity;
            sim.flows[t] += equity - from_investor;

            let (fee, distributed) = if final_month {
                let profit = sim.released + operating - sim.funded - opening;
                let fee = self.operating.fee.as_ref().map_or(0.0, |f| {
                    let capital = sim.funded + opening;
                    match f.basis {
                        FeeBasis::Profit => f.tiers.fee(profit),
                        FeeBasis::ProfitOverPremium => f.tiers.fee(profit / pr) * pr,
                        FeeBasis::ProfitOverCapital if capital > 0.0 => {
                            f.tiers.fee(profit / capital) * capital
                        }
                        FeeBasis::ProfitOverCapital => 0.0,
                    }
                });
                let d = operating - fee;
                operating = 0.0;
                (fee, d)
            } else {
                (0.0, 0.0)
            };
            sim.fee += fee;
            sim.distributed += distributed;
            sim.flows[t] += distributed;

            if let Some(tr) = trace.as_deref_mut() {
                tr.requirement[t] = s;
                tr.premium[t] = premium_in;
                tr.adjustment[t] = adjustment;
                tr.paid[t] = paid;
                tr.funding_investor[t] = from_investor;
                tr.funding_operating[t] = from_operating;
                tr.interest[t] = interest;
                tr.collateral_release[t] = release;
                tr.collateral_balance[t] = collateral;
                tr.expenses[t] = expenses;
                tr.equity_release[t] = equity + distributed;
                tr.operating_balance[t] = operating;
                tr.investor[t] = sim.flows[t];
                tr.commutation_paid += commutation_paid;
                tr.fee += fee;
            }
        }
        sim
    }
}

fn check_sims(u: &[f64], commission: &[f64]) -> Result<()> {
    if u.len() != commission.len() || u.is_empty() {
        return Err(Error::Data(format!(
            "{} loss ratios and {} commission rates; one each per simulation, at least one",
            u.len(),
            commission.len()
        )));
    }
    if u.iter().chain(commission).any(|x| !x.is_finite()) {
        return Err(Error::Data(
            "loss ratios and commission rates must be finite".into(),
        ));
    }
    Ok(())
}

struct Sim {
    flows: Vec<f64>,
    funded: f64,
    opening: f64,
    released: f64,
    distributed: f64,
    fee: f64,
    commutation: f64,
}

/// Each simulation's monthly investor flows and totals from
/// [`CashflowTerms::run`].
#[derive(Debug, Clone, PartialEq)]
pub struct Cashflows {
    months: usize,
    flows: Vec<f64>,
    funded: Vec<f64>,
    opening: Vec<f64>,
    released: Vec<f64>,
    distributed: Vec<f64>,
    fee: Vec<f64>,
    commutation: Vec<f64>,
}

impl Cashflows {
    fn empty(months: usize, n: usize) -> Self {
        Cashflows {
            months,
            flows: Vec::with_capacity(months * n),
            funded: Vec::with_capacity(n),
            opening: Vec::with_capacity(n),
            released: Vec::with_capacity(n),
            distributed: Vec::with_capacity(n),
            fee: Vec::with_capacity(n),
            commutation: Vec::with_capacity(n),
        }
    }

    fn push(&mut self, s: Sim) {
        self.flows.extend(s.flows);
        self.funded.push(s.funded);
        self.opening.push(s.opening);
        self.released.push(s.released);
        self.distributed.push(s.distributed);
        self.fee.push(s.fee);
        self.commutation.push(s.commutation);
    }

    /// The number of months `T`, the width of a row of [`flows`](Self::flows).
    pub fn months(&self) -> usize {
        self.months
    }

    /// The number of simulations.
    pub fn len(&self) -> usize {
        self.funded.len()
    }

    /// Whether there are no simulations.
    pub fn is_empty(&self) -> bool {
        self.funded.is_empty()
    }

    /// The investor's net flow in each month `0..T`, one simulation a row
    /// of a row-major matrix: equity released and distributed less the
    /// capital it paid, the opening capital at month 0.
    pub fn flows(&self) -> &[f64] {
        &self.flows
    }

    /// The month offsets `0..T`.
    pub fn month_offsets(&self) -> Vec<f64> {
        (0..self.months).map(|t| t as f64).collect()
    }

    /// The capital each simulation's investor paid into the trust,
    /// without the opening capital.
    pub fn funded(&self) -> &[f64] {
        &self.funded
    }

    /// The opening capital.
    pub fn opening(&self) -> &[f64] {
        &self.opening
    }

    /// The equity released before commutation.
    pub fn released(&self) -> &[f64] {
        &self.released
    }

    /// What was distributed at commutation, after the fee.
    pub fn distributed(&self) -> &[f64] {
        &self.distributed
    }

    /// The performance fee.
    pub fn fee(&self) -> &[f64] {
        &self.fee
    }

    /// The commutation requirement paid to the cedant.
    pub fn commutation(&self) -> &[f64] {
        &self.commutation
    }

    /// Whether each simulation's first nonzero investor flow is positive:
    /// the investor is paid before putting money in, so the flows can have
    /// several internal rates of return and the one found is usually
    /// meaningless (see [`Base::Reserve`]).
    pub fn paid_first(&self) -> Vec<bool> {
        if self.months == 0 {
            return vec![false; self.len()];
        }
        self.flows
            .chunks(self.months)
            .map(|row| row.iter().find(|&&x| x != 0.0).is_some_and(|&x| x > 0.0))
            .collect()
    }

    /// Each simulation's net present value at the annual `rate`, the months
    /// read as twelfths of a year.
    pub fn npv(&self, rate: f64) -> Result<Vec<f64>> {
        npv_rows(&self.flows, &self.years(), rate)
    }

    /// Each simulation's internal rate of return, the months read as
    /// twelfths of a year.
    pub fn irr(&self, options: &IrrOptions) -> Result<Vec<Irr>> {
        irr_rows(&self.flows, &self.years(), options)
    }

    fn years(&self) -> Vec<f64> {
        TimeUnit::Months.years(&self.month_offsets())
    }
}

/// Every account line of one simulation from [`CashflowTerms::trace`], one
/// value a month.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Trace {
    /// The collateral required (the commutation requirement in the last
    /// month).
    pub requirement: Vec<f64>,
    /// The premium collected into the trust.
    pub premium: Vec<f64>,
    /// Premium adjustments into the trust.
    pub adjustment: Vec<f64>,
    /// Claims paid from the trust.
    pub paid: Vec<f64>,
    /// Capital called from the investor.
    pub funding_investor: Vec<f64>,
    /// Capital called from the operating account.
    pub funding_operating: Vec<f64>,
    /// The trust's interest.
    pub interest: Vec<f64>,
    /// What the trust released to the operating account.
    pub collateral_release: Vec<f64>,
    /// The trust's closing balance.
    pub collateral_balance: Vec<f64>,
    /// Expenses paid from the operating account.
    pub expenses: Vec<f64>,
    /// Equity released or distributed to the investor.
    pub equity_release: Vec<f64>,
    /// The operating account's closing balance.
    pub operating_balance: Vec<f64>,
    /// The investor's net flow.
    pub investor: Vec<f64>,
    /// The commutation requirement paid to the cedant.
    pub commutation_paid: f64,
    /// The performance fee.
    pub fee: f64,
}

impl Trace {
    fn new(months: usize) -> Self {
        let z = || vec![0.0; months];
        Trace {
            requirement: z(),
            premium: z(),
            adjustment: z(),
            paid: z(),
            funding_investor: z(),
            funding_operating: z(),
            interest: z(),
            collateral_release: z(),
            collateral_balance: z(),
            expenses: z(),
            equity_release: z(),
            operating_balance: z(),
            investor: z(),
            commutation_paid: 0.0,
            fee: 0.0,
        }
    }

    /// The premium, adjustments and interest paid into the trust less what
    /// left the deal: the investor's net flows, the fee, the claims, the
    /// expenses and the commutation payment. Zero up to rounding.
    pub fn conservation_error(&self) -> f64 {
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        sum(&self.premium) + sum(&self.adjustment) + sum(&self.interest)
            - sum(&self.investor)
            - self.fee
            - sum(&self.paid)
            - sum(&self.expenses)
            - self.commutation_paid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deal written over 12 months, earned over 36 and paid over 48,
    /// commuting after month 30 with premium still unearned.
    fn patterns() -> MonthlyPatterns {
        let n = 60;
        let ramp = |m: f64| (0..n).map(|t| ((t + 1) as f64 / m).min(1.0)).collect();
        MonthlyPatterns::new(ramp(12.0), ramp(36.0), ramp(48.0)).unwrap()
    }

    /// The issue's chain with every rule set.
    fn full() -> CashflowTerms {
        let requirement = Requirement::new(vec![
            Term::new(Base::UnearnedPremium, 0.3),
            Term::new(Base::Reserve, 1.2),
        ])
        .with_limit(1.2, 1.5);
        let mut t = CashflowTerms::new(100.0, 30, patterns(), requirement).unwrap();
        t.commutation_requirement = Requirement::new(vec![
            Term::new(Base::UnearnedPremium, 0.3),
            Term::new(Base::Outstanding, 1.0),
        ]);
        t.premium_flow = PremiumFlow {
            collected_rate: 0.7,
            deduct_commission: false,
            lag: 2,
            adjustments: vec![Adjustment {
                month: 18,
                written_rate: 0.1,
                less_commission: true,
                base: AdjustmentBase::Ultimate,
            }],
        };
        t.funding = Funding {
            dates: Dates::quarterly(),
            rule: FundingRule::ObligationIncrease {
                premium_offset: 0.7,
                reference: ObligationReference::RunningMax,
            },
            source: FundingSource::Investor,
        };
        t.interest = Interest {
            annual_yield: 0.04,
            basis: InterestBasis::Average,
        };
        t.operating = Operating {
            opening_fixed: 5.0,
            opening_share: 0.1,
            expense_rate: 0.02,
            buffer: Buffer {
                months: 12,
                paid: true,
                expenses: true,
            },
            release: Timing::Monthly,
            fee: Some(PerformanceFee {
                tiers: TieredFee::new(vec![0.0, 0.1], vec![0.1, 0.2]).unwrap(),
                basis: FeeBasis::ProfitOverCapital,
            }),
        };
        t
    }

    fn variants() -> Vec<CashflowTerms> {
        let base = full();
        let mut v = vec![base.clone()];
        let mut a = base.clone();
        a.funding.rule = FundingRule::Shortfall;
        a.funding.source = FundingSource::OperatingFirst;
        v.push(a);
        let mut b = base.clone();
        b.funding.dates = Dates::Months(vec![0, 7, 20]);
        b.funding.rule = FundingRule::ObligationIncrease {
            premium_offset: 0.5,
            reference: ObligationReference::PreviousDate,
        };
        b.collateral_release = Timing::AtCommutation;
        b.interest.basis = InterestBasis::Opening;
        v.push(b);
        let mut c = base.clone();
        c.collateral_release = Timing::At(Dates::quarterly());
        c.operating.release = Timing::At(Dates::Every {
            months: 6,
            first: 5,
        });
        c.interest.basis = InterestBasis::Closing;
        c.operating.fee = Some(PerformanceFee {
            tiers: TieredFee::new(vec![10.0], vec![0.25]).unwrap(),
            basis: FeeBasis::Profit,
        });
        c.premium_flow.adjustments[0].base = AdjustmentBase::ToDate;
        v.push(c);
        let mut d = base;
        d.operating.release = Timing::AtCommutation;
        d.operating.buffer = Buffer::default();
        d.requirement = d.requirement.with_floor(2.0);
        d.operating.fee = Some(PerformanceFee {
            tiers: TieredFee::new(vec![0.0], vec![0.1]).unwrap(),
            basis: FeeBasis::ProfitOverPremium,
        });
        v.push(d);
        v
    }

    #[test]
    fn money_is_conserved_under_every_rule() {
        for (i, terms) in variants().iter().enumerate() {
            for &u in &[0.0, 0.4, 0.8, 1.1, 1.3, 2.0] {
                for &c in &[0.0, 0.25] {
                    let tr = terms.trace(u, c).unwrap();
                    assert!(
                        tr.conservation_error().abs() < 1e-9,
                        "variant {i}, u {u}, c {c}: {}",
                        tr.conservation_error()
                    );
                    // The trust is empty after commutation and the
                    // operating account after distribution.
                    assert!(tr.collateral_balance.last().unwrap().abs() < 1e-9);
                    assert!(tr.operating_balance.last().unwrap().abs() < 1e-9);
                    // The run gives the trace's investor flows.
                    let run = terms.run(&[u], &[c]).unwrap();
                    assert_eq!(run.flows(), tr.investor.as_slice());
                    assert!((run.fee()[0] - tr.fee).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn the_trust_holds_the_requirement_between_releases() {
        let terms = full();
        let tr = terms.trace(0.8, 0.2).unwrap();
        for t in 0..terms.commutation_month - 1 {
            // Released monthly, the trust ends each month at its
            // requirement or below it between funding dates.
            assert!(
                tr.collateral_balance[t] <= tr.requirement[t] + 1e-9,
                "month {t}"
            );
        }
    }

    #[test]
    fn the_limit_switch_reads_the_limit_not_yet_paid() {
        let terms = full();
        // 1.2 × 1.3 = 1.56 ≥ 1.5: the requirement is 1.5 W − C.
        let tr = terms.trace(1.3, 0.0).unwrap();
        let t = 5;
        let w = 100.0 * 6.0 / 12.0;
        let c = 1.3 * 100.0 * 6.0 / 48.0;
        assert!((tr.requirement[t] - (1.5 * w - c)).abs() < 1e-12);
        // 1.2 × 1.2 = 1.44 < 1.5: the loaded terms.
        let tr = terms.trace(1.2, 0.0).unwrap();
        let e = 100.0 * 6.0 / 36.0;
        let c = 1.2 * 100.0 * 6.0 / 48.0;
        let want = 0.3 * (w - e) + 1.2 * (1.2 * e - c);
        assert!((tr.requirement[t] - want).abs() < 1e-12);
    }

    #[test]
    fn the_obligation_rule_lets_the_trust_run_below_zero() {
        // The obligation nets the premium collected but not the claims
        // paid, so the trust pays claims nobody funded; the shortfall rule
        // keeps it at the requirement.
        let low = |t: &CashflowTerms| {
            let tr = t.trace(0.9, 0.2).unwrap();
            tr.collateral_balance
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min)
        };
        let mut terms = full();
        let obligation = low(&terms);
        assert!((obligation + 8.885288354521).abs() < 1e-9, "{obligation}");
        terms.funding.rule = FundingRule::Shortfall;
        assert!(low(&terms) >= -1e-9);
    }

    #[test]
    fn capital_is_called_on_funding_dates_only() {
        let terms = full();
        let tr = terms.trace(1.1, 0.0).unwrap();
        for (t, f) in tr.funding_investor.iter().enumerate() {
            if t % 3 != 0 {
                assert_eq!(*f, 0.0, "month {t}");
            }
        }
        assert!(tr.funding_investor.iter().sum::<f64>() > 0.0);
    }

    #[test]
    fn the_running_max_never_calls_more_than_the_previous_date() {
        let mut a = full();
        let tr_max = a.trace(1.1, 0.0).unwrap();
        a.funding.rule = FundingRule::ObligationIncrease {
            premium_offset: 0.7,
            reference: ObligationReference::PreviousDate,
        };
        let tr_prev = a.trace(1.1, 0.0).unwrap();
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        assert!(sum(&tr_max.funding_investor) <= sum(&tr_prev.funding_investor) + 1e-9);
    }

    #[test]
    fn negative_controls_move_the_result() {
        // Each of the issue's negative controls changes the flows.
        let base = full();
        let flows = |t: &CashflowTerms| t.run(&[0.9], &[0.2]).unwrap().flows().to_vec();
        let reference = flows(&base);
        let mut monthly = base.clone();
        monthly.funding.dates = Dates::monthly();
        let mut no_buffer = base.clone();
        no_buffer.operating.buffer = Buffer::default();
        let mut opening_interest = base.clone();
        opening_interest.interest.basis = InterestBasis::Opening;
        let mut loaded = base.clone();
        loaded.commutation_requirement.terms[1].load = 1.1;
        let mut no_uep = base.clone();
        no_uep.commutation_requirement.terms.remove(0);
        for (name, t) in [
            ("monthly funding", monthly),
            ("no buffer", no_buffer),
            ("interest on the opening balance", opening_interest),
            ("a load on the commutation reserve", loaded),
            ("no UEP at commutation", no_uep),
        ] {
            assert_ne!(flows(&t), reference, "{name}");
        }
        // A buffer only delays releases: without one the investor gets its
        // money earlier, so its NPV at a positive rate is no lower.
        let mut nb = base.clone();
        nb.operating.buffer = Buffer::default();
        let npv = |t: &CashflowTerms| t.run(&[0.9], &[0.2]).unwrap().npv(0.1).unwrap()[0];
        assert!(npv(&nb) >= npv(&base) - 1e-9);
    }

    #[test]
    fn the_flows_feed_the_return_metrics() {
        let mut terms = full();
        terms.funding.rule = FundingRule::Shortfall;
        let out = terms.run(&[0.5, 0.9, 1.4], &[0.2, 0.2, 0.2]).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out.flows().len(), 3 * 30);
        // A better loss ratio is worth more.
        let at5 = out.npv(0.05).unwrap();
        assert!(at5[0] > at5[1] && at5[1] > at5[2], "{at5:?}");
        // Each IRR found prices the flows at zero.
        let irr = out.irr(&IrrOptions::default()).unwrap();
        assert!(irr.iter().any(|r| r.found));
        for r in irr.iter().filter(|r| r.found) {
            assert!(r.npv.abs() < 1e-6);
        }
        let npv = out.npv(0.0).unwrap();
        for (j, (v, row)) in npv.iter().zip(out.flows().chunks(30)).enumerate() {
            assert!((v - row.iter().sum::<f64>()).abs() < 1e-9);
            let net = out.released()[j] + out.distributed()[j] - out.funded()[j] - out.opening()[j];
            assert!((v - net).abs() < 1e-9);
        }
    }

    #[test]
    fn a_given_requirement_replaces_the_terms() {
        let terms = full();
        let width = 30 + 12;
        let tr = terms.trace(0.9, 0.2).unwrap();
        // Rebuild the term requirement as given rows: the same flows.
        let given: Vec<f64> = (0..width)
            .map(|t| terms.requirement.at(&terms.month(t, 0.9)))
            .collect();
        let out = terms.run_with_requirement(&[0.9], &[0.2], &given).unwrap();
        assert_eq!(out.flows(), tr.investor.as_slice());
        assert!(
            terms
                .run_with_requirement(&[0.9], &[0.2], &given[1..])
                .is_err()
        );
    }

    #[test]
    fn refuses_what_it_cannot_run() {
        let r = Requirement::new(vec![Term::new(Base::Reserve, 1.0)]);
        assert!(CashflowTerms::new(0.0, 3, patterns(), r.clone()).is_err());
        assert!(CashflowTerms::new(100.0, 0, patterns(), r.clone()).is_err());
        assert!(CashflowTerms::new(100.0, 61, patterns(), r.clone()).is_err());
        let mut t = full();
        t.operating.buffer.months = 31;
        assert!(
            t.run(&[0.5], &[0.0]).is_err(),
            "patterns too short for the buffer"
        );
        let mut t = full();
        t.premium_flow.adjustments[0].month = 30;
        assert!(t.run(&[0.5], &[0.0]).is_err(), "adjustment at commutation");
        let mut t = full();
        t.funding.dates = Dates::Every {
            months: 0,
            first: 0,
        };
        assert!(t.run(&[0.5], &[0.0]).is_err());
        assert!(full().run(&[0.5], &[]).is_err());
        assert!(full().run(&[f64::NAN], &[0.0]).is_err());
        assert!(MonthlyPatterns::new(vec![1.0], vec![1.0, 1.0], vec![1.0]).is_err());
    }

    #[test]
    fn a_reserve_rule_alone_pays_the_investor_first() {
        // The module example's terms: the trust releases the premium
        // collected ahead of earning at once.
        let p = MonthlyPatterns::new(
            vec![1.0, 1.0, 1.0],
            vec![1.0 / 3.0, 2.0 / 3.0, 1.0],
            vec![1.0 / 3.0, 2.0 / 3.0, 1.0],
        )
        .unwrap();
        let reserve = Requirement::new(vec![Term::new(Base::Reserve, 1.0)]);
        let mut terms = CashflowTerms::new(100.0, 3, p.clone(), reserve).unwrap();
        terms.commutation_requirement = Requirement::new(vec![Term::new(Base::Outstanding, 1.0)]);
        let run = terms.run(&[0.6, 1.5], &[0.0, 0.0]).unwrap();
        // The doc example's flows, unchanged: [80, -20, -20] and [50, -50, -50].
        assert!((run.flows()[0] - 80.0).abs() < 1e-9);
        assert!((run.flows()[3] - 50.0).abs() < 1e-9);
        assert_eq!(run.paid_first(), [true, true]);
        // On the outstanding loss the investor puts money in first.
        let outstanding = Requirement::new(vec![Term::new(Base::Outstanding, 1.0)]);
        let terms = CashflowTerms::new(100.0, 3, p, outstanding).unwrap();
        assert!(!terms.run(&[1.2], &[0.0]).unwrap().paid_first()[0]);
    }

    #[test]
    fn a_deducted_commission_is_a_lower_collected_rate() {
        let base = full();
        let mut deducted = base.clone();
        deducted.premium_flow.deduct_commission = true;
        let u = [0.4, 0.7, 1.1];
        let c = [0.2, 0.25, 0.3];
        let with = deducted.run(&u, &c).unwrap();
        for j in 0..u.len() {
            let mut flat = base.clone();
            flat.premium_flow.collected_rate = base.premium_flow.collected_rate - c[j];
            // The commission still reaches the adjustment and the fee, so
            // the comparison run keeps it.
            let one = flat.run(&u[j..=j], &c[j..=j]).unwrap();
            let m = one.months();
            for (a, b) in with.flows()[j * m..(j + 1) * m].iter().zip(one.flows()) {
                assert!((a - b).abs() < 1e-9, "{a} against {b}");
            }
        }
        // The default leaves every result as it was.
        assert!(!PremiumFlow::default().deduct_commission);
        let plain = base.run(&u, &c).unwrap();
        assert!(plain.flows() != with.flows());
    }

    #[test]
    fn monthly_patterns_from_an_annual_payment_pattern() {
        // Written evenly over a year, twelve-month policies: earned by the
        // end of month t is the written share's convolution with the term.
        let n = 72;
        let written: Vec<f64> = (0..n).map(|t| ((t + 1) as f64 / 12.0).min(1.0)).collect();
        let paid = PaymentPattern::from_development(&[1.0, 2.0, 3.0], &[2.5, 1.25, 1.0]).unwrap();
        let p = MonthlyPatterns::from_annual(written.clone(), 12, &paid, Interpolation::Linear)
            .unwrap();
        // Month 0 writes 1/12, earned 1/12 of it.
        assert!((p.earned[0] - 1.0 / 144.0).abs() < 1e-15);
        // Month 11: each month's writing earned (12 - s) / 12 of it.
        let e11: f64 = (0..12).map(|s| (12 - s) as f64 / 144.0).sum();
        assert!((p.earned[11] - e11).abs() < 1e-15);
        assert!((p.earned[22] - 1.0).abs() < 1e-15);
        // Paid by a hand sum over the accident months.
        let t = 30;
        let de: Vec<f64> = (0..n)
            .map(|k| p.earned[k] - if k == 0 { 0.0 } else { p.earned[k - 1] })
            .collect();
        let want: f64 = (0..=t)
            .map(|e| {
                de[e] * paid.paid_by((t - e) as f64 / 12.0 + 1.0 / 24.0, Interpolation::Linear)
            })
            .sum();
        assert!((p.paid[t] - want).abs() < 1e-15);
        assert!(p.paid.windows(2).all(|w| w[1] >= w[0] - 1e-15));
        assert!((p.paid[n - 1] - 1.0).abs() < 1e-12);
        // It runs in the engine.
        let req = Requirement::new(vec![Term::new(Base::Outstanding, 1.0)]);
        let terms = CashflowTerms::new(100.0, 48, p, req).unwrap();
        assert!(terms.run(&[0.7], &[0.0]).is_ok());
        assert!(MonthlyPatterns::from_annual(written, 0, &paid, Interpolation::Linear).is_err());
    }
}
