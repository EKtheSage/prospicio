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
//! After binding, [`CashflowTerms::valuation`] takes the deal's actual
//! accounts at a month ([`ActualState`]) and [`Valuation::run`] simulates
//! the rest of the term from them, each simulation's ultimate conditioned
//! on the loss paid to date ([`condition_on_paid`]), with the return from
//! the valuation month and since inception ([`Forward`]); [`deal_ave`]
//! compares the paid to date with what the priced simulations expected
//! (#603, `time-value.md` decision 34).
//!
//! Over the term, money is conserved: what the investor gets back net of
//! what it put in, plus the fee, the claims paid, the expenses and the
//! commutation payment, equals the premium and adjustments paid into the
//! trust plus its interest ([`Trace::conservation_error`]).

use prospicio_core::{Error, Result};

use crate::payout::{Interpolation, PaymentPattern};
use prospicio_prob::statistics::QuantileMethod;

use crate::returns::{Irr, IrrOptions, TieredFee, TimeUnit, irr_rows, moic, npv_rows};

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

/// Paid patterns that differ by simulation, for
/// [`CashflowTerms::run_with_patterns`] (#600): `K` monthly cumulative
/// paid patterns and, for each simulation, the pattern it pays by. One
/// pattern a simulation is [`new`](Self::new); a few patterns shared by
/// many simulations, to save memory, are [`indexed`](Self::indexed).
///
/// Each pattern is finite and non-decreasing and all end at the same
/// share; against the terms they need the terms' length and to end at the
/// terms' paid share, so that a simulation's ultimate loss ratio means the
/// same whichever pattern pays it.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternDraws {
    patterns: Vec<Vec<f64>>,
    index: Vec<usize>,
}

impl PatternDraws {
    /// One paid pattern per simulation, in simulation order.
    pub fn new(patterns: Vec<Vec<f64>>) -> Result<Self> {
        let index = (0..patterns.len()).collect();
        Self::indexed(patterns, index)
    }

    /// `patterns` shared by the simulations, simulation `j` paying by
    /// `patterns[index[j]]`.
    pub fn indexed(patterns: Vec<Vec<f64>>, index: Vec<usize>) -> Result<Self> {
        let Some(first) = patterns.first() else {
            return Err(Error::Data("at least one paid pattern is needed".into()));
        };
        let (n, end) = (first.len(), first.last().copied().unwrap_or(f64::NAN));
        for (k, p) in patterns.iter().enumerate() {
            if p.len() != n || n == 0 {
                return Err(Error::Data(format!(
                    "paid pattern {k} has {} months and pattern 0 has {n}; they need one \
                     positive length",
                    p.len()
                )));
            }
            if p.iter().any(|x| !x.is_finite()) {
                return Err(Error::Data(format!("paid pattern {k} is not finite")));
            }
            if let Some(t) = (1..n).find(|&t| p[t] < p[t - 1]) {
                return Err(Error::Data(format!(
                    "paid pattern {k} falls in month {t}; a cumulative pattern does not decrease"
                )));
            }
            if (p[n - 1] - end).abs() > 1e-9 {
                return Err(Error::Data(format!(
                    "paid pattern {k} ends at {} and pattern 0 at {end}; they need one ultimate \
                     share",
                    p[n - 1]
                )));
            }
        }
        if let Some(&bad) = index.iter().find(|&&i| i >= patterns.len()) {
            return Err(Error::Data(format!(
                "pattern index {bad} is past the {} patterns",
                patterns.len()
            )));
        }
        Ok(Self { patterns, index })
    }

    /// The number of simulations.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether there are no simulations.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// The paid pattern of simulation `j`.
    pub fn pattern(&self, j: usize) -> &[f64] {
        &self.patterns[self.index[j]]
    }

    /// The distinct patterns.
    pub fn patterns(&self) -> &[Vec<f64>] {
        &self.patterns
    }

    /// Each simulation's pattern, as an index into
    /// [`patterns`](Self::patterns).
    pub fn index(&self) -> &[usize] {
        &self.index
    }

    /// Fails unless there are `n_sims` simulations and the patterns have
    /// the length of `terms` and end at its paid share, to 1e-9.
    pub fn check_against(&self, terms: &MonthlyPatterns, n_sims: usize) -> Result<()> {
        if self.len() != n_sims {
            return Err(Error::Data(format!(
                "{} paid patterns for {n_sims} simulations",
                self.len()
            )));
        }
        let p = &self.patterns[0];
        if p.len() != terms.len() {
            return Err(Error::Data(format!(
                "the paid patterns have {} months and the terms' patterns {}",
                p.len(),
                terms.len()
            )));
        }
        let end = terms.paid[terms.len() - 1];
        if (p[p.len() - 1] - end).abs() > 1e-9 {
            return Err(Error::Data(format!(
                "the paid patterns end at {} and the terms' paid pattern at {end}",
                p[p.len() - 1]
            )));
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
            let path = PaidPath::new(u, self.premium, &self.patterns.paid);
            out.push(self.simulate(&path, c, &req, None, None));
        }
        Ok(out)
    }

    /// [`run`](Self::run) with each simulation's paid pattern drawn from
    /// `paid` (#600): simulation `j` pays its ultimate loss by
    /// `paid.pattern(j)`, so the timing of payments, and with it the
    /// collateral held and released, differs by simulation. The written
    /// and earned patterns stay those of the terms. Identical patterns
    /// equal to the terms' paid pattern reproduce [`run`](Self::run)
    /// exactly.
    ///
    /// Fails unless `paid` has one index per simulation and patterns of
    /// the terms' length ending at the terms' paid share
    /// ([`PatternDraws::check_against`]).
    pub fn run_with_patterns(
        &self,
        u: &[f64],
        commission: &[f64],
        paid: &PatternDraws,
    ) -> Result<Cashflows> {
        self.check()?;
        check_sims(u, commission)?;
        paid.check_against(&self.patterns, u.len())?;
        let mut out = Cashflows::empty(self.commutation_month, u.len());
        let req = |_: usize, m: &Month| self.requirement.at(m);
        for (j, (&u, &c)) in u.iter().zip(commission).enumerate() {
            let path = PaidPath::new(u, self.premium, paid.pattern(j));
            out.push(self.simulate(&path, c, &req, None, None));
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
            let path = PaidPath::new(u, self.premium, &self.patterns.paid);
            out.push(self.simulate(&path, c, &req, None, None));
        }
        Ok(out)
    }

    /// Every account line of one simulation, month by month.
    pub fn trace(&self, u: f64, commission: f64) -> Result<Trace> {
        self.check()?;
        check_sims(&[u], &[commission])?;
        let mut trace = Trace::new(self.commutation_month);
        let req = |_: usize, m: &Month| self.requirement.at(m);
        let path = PaidPath::new(u, self.premium, &self.patterns.paid);
        self.simulate(&path, commission, &req, None, Some(&mut trace));
        Ok(trace)
    }

    /// [`trace`](Self::trace) of one simulation paid by its own pattern
    /// `paid`, checked as each of [`PatternDraws`] is.
    pub fn trace_with_pattern(&self, u: f64, commission: f64, paid: &[f64]) -> Result<Trace> {
        self.check()?;
        check_sims(&[u], &[commission])?;
        PatternDraws::new(vec![paid.to_vec()])?.check_against(&self.patterns, 1)?;
        let mut trace = Trace::new(self.commutation_month);
        let req = |_: usize, m: &Month| self.requirement.at(m);
        let path = PaidPath::new(u, self.premium, paid);
        self.simulate(&path, commission, &req, None, Some(&mut trace));
        Ok(trace)
    }

    fn month(&self, t: usize, path: &PaidPath) -> Month {
        let p = &self.patterns;
        Month {
            premium: self.premium,
            u: path.u,
            written: self.premium * p.written[t],
            earned: self.premium * p.earned[t],
            paid: path.cum(t),
        }
    }

    /// One simulation from inception, or from a valuation month when
    /// `start` holds the accounts at its start.
    fn simulate(
        &self,
        path: &PaidPath,
        c: f64,
        req: &dyn Fn(usize, &Month) -> f64,
        start: Option<&Start>,
        mut trace: Option<&mut Trace>,
    ) -> Sim {
        let t_n = self.commutation_month;
        let v = start.map_or(0, |s| s.month);
        let pr = self.premium;
        let pat = &self.patterns;
        let last = pat.len() - 1;
        let at = |v: &[f64], t: isize| if t < 0 { 0.0 } else { v[t as usize] };
        let written_inc = |t: isize| pr * (at(&pat.written, t) - at(&pat.written, t - 1));
        let paid_inc = |t: usize| path.inc(t);
        let lag = self.premium_flow.lag as isize;
        let monthly_rate = self.interest.annual_yield / 12.0;
        let expense = |s: f64| self.operating.expense_rate / 12.0 * s;
        let buffer_months = self.operating.buffer.months;
        // The regular requirement over the term and the buffer's months.
        let requirement: Vec<f64> = (0..t_n + buffer_months)
            .map(|t| req(t, &self.month(t, path)))
            .collect();
        let commutation = self.commutation_requirement.at(&self.month(t_n - 1, path));

        let opening = self.operating.opening_fixed + self.operating.opening_share * pr;
        let mut collateral = start.map_or(0.0, |s| s.collateral);
        let mut operating = start.map_or(opening, |s| s.operating);
        let mut reference = start.map_or(0.0, |s| s.reference);
        let mut sim = Sim {
            flows: vec![0.0; t_n - v],
            funded: start.map_or(0.0, |s| s.funded),
            opening,
            released: start.map_or(0.0, |s| s.released),
            distributed: 0.0,
            fee: 0.0,
            commutation,
        };
        if start.is_none() {
            sim.flows[0] = -opening;
        }
        let arrears = start.map_or(0.0, |s| s.arrears);
        if let Some(tr) = trace.as_deref_mut() {
            tr.opening_balance = start.map_or(0.0, |s| s.collateral + s.operating);
        }

        for t in v..t_n {
            let i = t - v;
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
            let mut premium_in = collected * written_inc(t as isize - lag);
            if t == v {
                premium_in += arrears;
            }
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
            sim.flows[i] += equity - from_investor;

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
            sim.flows[i] += distributed;

            if let Some(tr) = trace.as_deref_mut() {
                tr.requirement[i] = s;
                tr.premium[i] = premium_in;
                tr.adjustment[i] = adjustment;
                tr.paid[i] = paid;
                tr.funding_investor[i] = from_investor;
                tr.funding_operating[i] = from_operating;
                tr.interest[i] = interest;
                tr.collateral_release[i] = release;
                tr.collateral_balance[i] = collateral;
                tr.expenses[i] = expenses;
                tr.equity_release[i] = equity + distributed;
                tr.operating_balance[i] = operating;
                tr.investor[i] = sim.flows[i];
                tr.funding_reference[i] = reference;
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

/// A simulation's cumulative paid loss: `u P S(t)` from inception, or,
/// after a valuation at month `v`, the actual paid `A` to its start and
/// `A + scale × (S(t) − S(v − 1))` after it.
struct PaidPath<'a> {
    /// The ultimate loss ratio the requirement reads.
    u: f64,
    premium: f64,
    pattern: &'a [f64],
    /// The valuation month, the actual paid and the scale.
    from: Option<(usize, f64, f64)>,
}

impl<'a> PaidPath<'a> {
    fn new(u: f64, premium: f64, pattern: &'a [f64]) -> Self {
        PaidPath {
            u,
            premium,
            pattern,
            from: None,
        }
    }

    fn at(&self, t: isize) -> f64 {
        if t < 0 { 0.0 } else { self.pattern[t as usize] }
    }

    /// The loss paid by the end of month `t`.
    fn cum(&self, t: usize) -> f64 {
        match self.from {
            None => self.u * self.premium * self.pattern[t],
            Some((v, a, _)) if t < v => a,
            Some((v, a, scale)) => a + scale * (self.pattern[t] - self.pattern[v - 1]),
        }
    }

    /// The loss paid in month `t`.
    fn inc(&self, t: usize) -> f64 {
        let step = self.at(t as isize) - self.at(t as isize - 1);
        match self.from {
            None => self.u * self.premium * step,
            Some((v, _, _)) if t < v => 0.0,
            Some((_, _, scale)) => scale * step,
        }
    }
}

/// The accounts at the start of a valuation month.
struct Start {
    month: usize,
    collateral: f64,
    operating: f64,
    funded: f64,
    released: f64,
    reference: f64,
    /// Premium the schedule collected before the month less what the trust
    /// received, collected in the month.
    arrears: f64,
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
    /// The obligation a [`FundingRule::ObligationIncrease`] compares with
    /// after each month (0 under [`FundingRule::Shortfall`]), what
    /// [`ActualState::funding_reference`] carries into a valuation.
    pub funding_reference: Vec<f64>,
    /// The trust and operating balances the trace starts from: 0 from
    /// inception, the actual balances from a valuation month.
    pub opening_balance: f64,
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
            funding_reference: z(),
            opening_balance: 0.0,
            commutation_paid: 0.0,
            fee: 0.0,
        }
    }

    /// The premium, adjustments and interest paid into the trust, and the
    /// balances held at the start, less what left the deal: the investor's
    /// net flows, the fee, the claims, the expenses and the commutation
    /// payment. Zero up to rounding.
    pub fn conservation_error(&self) -> f64 {
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        sum(&self.premium) + sum(&self.adjustment) + sum(&self.interest) + self.opening_balance
            - sum(&self.investor)
            - self.fee
            - sum(&self.paid)
            - sum(&self.expenses)
            - self.commutation_paid
    }
}

/// How a valuation conditions each simulation's ultimate loss on the loss
/// actually paid to date `A`, with `S` the share of the paid pattern
/// reached by then (#603, `time-value.md` decision 34, item M1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Conditioning {
    /// Every simulation is kept, and its ultimate is the actual paid plus
    /// its own unpaid, `A + u_j P (1 − S)`, as the Bornhuetter-Ferguson
    /// method takes it: future payments do not depend on past ones. The
    /// forward distribution narrows as the deal runs off.
    #[default]
    BornhuetterFerguson,
    /// Only the simulations whose ultimate `u_j P` is at least `A` are
    /// kept, unchanged; what each has still to pay, `u_j P − A`, follows
    /// the rest of the pattern. Late in a deal it can drop most
    /// simulations.
    Truncation,
}

/// The simulations kept by [`condition_on_paid`] and their ultimate loss
/// ratios.
#[derive(Debug, Clone, PartialEq)]
pub struct Conditioned {
    /// The simulations kept, by their index in the input, in order.
    pub kept: Vec<usize>,
    /// Each kept simulation's ultimate loss ratio given the paid to date,
    /// `U_j / P`.
    pub ultimate: Vec<f64>,
}

/// Each simulation's ultimate loss ratio `u[j]` conditioned on the loss
/// `paid` to date when the pattern has reached `share_paid`, as `how`
/// says: under [`Conditioning::BornhuetterFerguson`] every simulation, at
/// `paid / premium + u_j (1 − share_paid)`; under
/// [`Conditioning::Truncation`] those with `u_j premium ≥ paid`, at `u_j`.
///
/// ```
/// use prospicio_pricing::cashflow::{Conditioning, condition_on_paid};
///
/// // 30 paid of a premium of 100 when the pattern is 40% paid.
/// let bf = condition_on_paid(&[0.5, 0.8], 100.0, 30.0, 0.4, Conditioning::BornhuetterFerguson)
///     .unwrap();
/// assert_eq!(bf.kept, [0, 1]);
/// assert!((bf.ultimate[0] - 0.6).abs() < 1e-15 && (bf.ultimate[1] - 0.78).abs() < 1e-15);
/// let cut = condition_on_paid(&[0.2, 0.8], 100.0, 30.0, 0.4, Conditioning::Truncation).unwrap();
/// assert_eq!((cut.kept, cut.ultimate), (vec![1], vec![0.8]));
/// ```
pub fn condition_on_paid(
    u: &[f64],
    premium: f64,
    paid: f64,
    share_paid: f64,
    how: Conditioning,
) -> Result<Conditioned> {
    if !(premium.is_finite() && premium > 0.0) {
        return Err(Error::InvalidParameter {
            name: "premium",
            value: premium,
            reason: "must be finite and positive",
        });
    }
    if !paid.is_finite() || !share_paid.is_finite() || u.iter().any(|x| !x.is_finite()) {
        return Err(Error::Data(
            "the loss ratios, the paid to date and the share paid must be finite".into(),
        ));
    }
    let (kept, ultimate): (Vec<usize>, Vec<f64>) = match how {
        Conditioning::BornhuetterFerguson => u
            .iter()
            .enumerate()
            .map(|(j, &u)| (j, paid / premium + u * (1.0 - share_paid)))
            .unzip(),
        Conditioning::Truncation => u
            .iter()
            .enumerate()
            .filter(|&(_, &u)| u * premium >= paid)
            .map(|(j, &u)| (j, u))
            .unzip(),
    };
    if kept.is_empty() {
        return Err(Error::Data(format!(
            "no simulation's ultimate loss reaches the {paid} paid to date, so truncation keeps \
             none"
        )));
    }
    Ok(Conditioned { kept, ultimate })
}

/// A bound deal's actual accounts at a valuation (#603, `time-value.md`
/// decision 34, item M2): each amount to the end of month `month − 1`, so
/// the forward run starts at month `month`.
///
/// A valuation month after an adjustment month is refused, and the fee
/// crystallises only at commutation, so every amount the engine carries
/// from month to month is here.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ActualState {
    /// The valuation month `v`, the first month the forward run simulates.
    pub month: usize,
    /// The premium paid into the trust to date. What the deal's schedule
    /// would have collected by then and the trust has not received (or
    /// received beyond it) is collected (or paid back) in month `v`.
    pub premium_collected: f64,
    /// The losses paid from the trust to date, `A`.
    pub paid: f64,
    /// The trust's balance.
    pub collateral: f64,
    /// The capital the investor has paid into the trust, without the
    /// opening capital.
    pub funded: f64,
    /// The equity released to the investor.
    pub released: f64,
    /// The operating account's balance.
    pub operating: f64,
    /// The investor's actual flows in months `0..v` (the opening capital
    /// in month 0), for the return since inception; without them only the
    /// forward return is given.
    pub flows: Option<Vec<f64>>,
    /// Under [`FundingRule::ObligationIncrease`], the obligation its next
    /// call is measured from ([`Trace::funding_reference`]); required
    /// under that rule after month 0.
    pub funding_reference: Option<f64>,
}

impl ActualState {
    /// The state of a simulated deal at the start of month `month`, from
    /// its [`trace`](CashflowTerms::trace) from inception: to value one
    /// path of a simulation as if it had happened.
    pub fn from_trace(trace: &Trace, month: usize) -> Result<Self> {
        if trace.opening_balance != 0.0 {
            return Err(Error::Data(
                "the trace starts from a valuation; an actual state needs one from inception"
                    .into(),
            ));
        }
        let n = trace.investor.len();
        if month >= n {
            return Err(Error::Data(format!(
                "a valuation at month {month} of a trace of {n} months leaves nothing to run"
            )));
        }
        if month == 0 {
            return Ok(ActualState {
                flows: Some(Vec::new()),
                ..ActualState::default()
            });
        }
        let sum = |v: &[f64]| v[..month].iter().sum::<f64>();
        Ok(ActualState {
            month,
            premium_collected: sum(&trace.premium),
            paid: sum(&trace.paid),
            collateral: trace.collateral_balance[month - 1],
            funded: sum(&trace.funding_investor),
            released: sum(&trace.equity_release),
            operating: trace.operating_balance[month - 1],
            flows: Some(trace.investor[..month].to_vec()),
            funding_reference: Some(trace.funding_reference[month - 1]),
        })
    }

    fn is_empty(&self) -> bool {
        [
            self.premium_collected,
            self.paid,
            self.collateral,
            self.funded,
            self.released,
            self.operating,
        ]
        .iter()
        .all(|&x| x == 0.0)
            && self.flows.as_ref().is_none_or(|f| f.is_empty())
            && self.funding_reference.is_none_or(|r| r == 0.0)
    }
}

/// A deal's terms with its actual state at a valuation month, from
/// [`CashflowTerms::valuation`] (#603).
#[derive(Debug, Clone, PartialEq)]
pub struct Valuation {
    terms: CashflowTerms,
    state: ActualState,
}

impl CashflowTerms {
    /// The deal valued at `state.month` from its actual accounts (#603,
    /// `time-value.md` decision 34): [`Valuation::run`] simulates the
    /// months from there to commutation, each simulation conditioned on
    /// the actual paid to date. Valuation is at a month end, the engine's
    /// step (item M6).
    ///
    /// Fails when nothing is left to run (`month` at or past the
    /// commutation month), at month 0 unless the state is empty, after an
    /// adjustment month, under [`FundingRule::ObligationIncrease`] after
    /// month 0 without a funding reference, or when the actual flows are
    /// not one a month to the valuation month.
    pub fn valuation(&self, state: ActualState) -> Result<Valuation> {
        self.check()?;
        let v = state.month;
        if v >= self.commutation_month {
            return Err(Error::Data(format!(
                "the deal commutes in month {}; a valuation at month {v} leaves nothing to run",
                self.commutation_month - 1
            )));
        }
        let amounts = [
            state.premium_collected,
            state.paid,
            state.collateral,
            state.funded,
            state.released,
            state.operating,
            state.funding_reference.unwrap_or(0.0),
        ];
        if amounts.iter().any(|x| !x.is_finite())
            || state.flows.iter().flatten().any(|x| !x.is_finite())
        {
            return Err(Error::Data("the actual state must be finite".into()));
        }
        if v == 0 && !state.is_empty() {
            return Err(Error::Data(
                "a valuation at month 0 is inception: the actual state must be empty".into(),
            ));
        }
        if let Some(f) = &state.flows {
            if f.len() != v {
                return Err(Error::Data(format!(
                    "{} actual flows for a valuation at month {v}; one a month in 0..{v}",
                    f.len()
                )));
            }
        }
        if let Some(a) = self.premium_flow.adjustments.iter().find(|a| a.month < v) {
            return Err(Error::Data(format!(
                "a premium adjustment was made in month {}, before the valuation month {v}; a \
                 valuation after an adjustment is not supported",
                a.month
            )));
        }
        if v > 0
            && matches!(self.funding.rule, FundingRule::ObligationIncrease { .. })
            && state.funding_reference.is_none()
        {
            return Err(Error::Data(
                "under the obligation-increase funding rule a valuation needs the funding \
                 reference, the obligation the next call is measured from"
                    .into(),
            ));
        }
        Ok(Valuation {
            terms: self.clone(),
            state,
        })
    }
}

impl Valuation {
    /// The deal's terms.
    pub fn terms(&self) -> &CashflowTerms {
        &self.terms
    }

    /// The actual state.
    pub fn state(&self) -> &ActualState {
        &self.state
    }

    /// The valuation month `v`.
    pub fn month(&self) -> usize {
        self.state.month
    }

    /// The share of the paid pattern reached by the end of month `v − 1`,
    /// `S` (0 at month 0).
    pub fn share_paid(&self) -> f64 {
        match self.state.month {
            0 => 0.0,
            v => self.terms.patterns.paid[v - 1],
        }
    }

    /// The investor's net asset value at the valuation: the trust's and
    /// the operating account's balances, which the forward return takes
    /// as the investment at the valuation month (item M3).
    pub fn nav(&self) -> f64 {
        self.state.collateral + self.state.operating
    }

    /// Each simulation's cash flows from the valuation month to
    /// commutation, its ultimate loss ratio `u[j]` conditioned on the
    /// actual paid as `how` says ([`condition_on_paid`]) and its loss still
    /// to pay following the rest of the terms' paid pattern. The trust and
    /// operating account open at the actual balances, the premium already
    /// collected is not collected again, and the fee reads the totals since
    /// inception.
    ///
    /// At month 0 with an empty state the flows are [`CashflowTerms::run`]'s
    /// exactly.
    pub fn run(&self, u: &[f64], commission: &[f64], how: Conditioning) -> Result<Forward> {
        check_sims(u, commission)?;
        let terms = &self.terms;
        let s = &self.state;
        let v = s.month;
        let pr = terms.premium;
        let share = self.share_paid();
        let cond = condition_on_paid(u, pr, s.paid, share, how)?;
        let req = |_: usize, m: &Month| terms.requirement.at(m);
        let mut out = Cashflows::empty(terms.commutation_month - v, cond.kept.len());
        for (&j, &ult) in cond.kept.iter().zip(&cond.ultimate) {
            out.push(self.simulate_one(u[j], ult, commission[j], how, &req, None));
        }
        Ok(Forward {
            month: v,
            kept: cond.kept,
            ultimate: cond.ultimate,
            flows: out,
            nav: self.nav(),
            funded_before: s.funded,
            actual: s.flows.clone(),
        })
    }

    /// Every account line of one simulation from the valuation month,
    /// conditioned as [`run`](Self::run) conditions it: one value a month
    /// in months `v..T`, with [`Trace::opening_balance`] the actual
    /// balances. Fails when truncation drops the simulation.
    pub fn trace(&self, u: f64, commission: f64, how: Conditioning) -> Result<Trace> {
        check_sims(&[u], &[commission])?;
        let terms = &self.terms;
        let cond = condition_on_paid(&[u], terms.premium, self.state.paid, self.share_paid(), how)?;
        let req = |_: usize, m: &Month| terms.requirement.at(m);
        let mut trace = Trace::new(terms.commutation_month - self.state.month);
        self.simulate_one(u, cond.ultimate[0], commission, how, &req, Some(&mut trace));
        Ok(trace)
    }

    fn simulate_one(
        &self,
        u: f64,
        ultimate: f64,
        c: f64,
        how: Conditioning,
        req: &dyn Fn(usize, &Month) -> f64,
        trace: Option<&mut Trace>,
    ) -> Sim {
        let terms = &self.terms;
        let s = &self.state;
        let v = s.month;
        let pr = terms.premium;
        let pattern = &terms.patterns.paid;
        if v == 0 {
            return terms.simulate(&PaidPath::new(ultimate, pr, pattern), c, req, None, trace);
        }
        let rest = 1.0 - self.share_paid();
        let scale = match how {
            Conditioning::BornhuetterFerguson => u * pr,
            Conditioning::Truncation if rest > 0.0 => (u * pr - s.paid) / rest,
            Conditioning::Truncation => 0.0,
        };
        let path = PaidPath {
            u: ultimate,
            premium: pr,
            pattern,
            from: Some((v, s.paid, scale)),
        };
        // The premium the schedule collected in months 0..v.
        let flow = &terms.premium_flow;
        let collected = if flow.deduct_commission {
            flow.collected_rate - c
        } else {
            flow.collected_rate
        };
        let written = |t: isize| {
            if t < 0 {
                0.0
            } else {
                terms.patterns.written[t as usize]
            }
        };
        let scheduled = collected * pr * written(v as isize - 1 - flow.lag as isize);
        let start = Start {
            month: v,
            collateral: s.collateral,
            operating: s.operating,
            funded: s.funded,
            released: s.released,
            reference: s.funding_reference.unwrap_or(0.0),
            arrears: scheduled - s.premium_collected,
        };
        terms.simulate(&path, c, req, Some(&start), trace)
    }
}

/// The forward cash flows of a [`Valuation`], one row a simulation kept,
/// with the returns from the valuation month and since inception (item
/// M3).
#[derive(Debug, Clone, PartialEq)]
pub struct Forward {
    month: usize,
    kept: Vec<usize>,
    ultimate: Vec<f64>,
    flows: Cashflows,
    nav: f64,
    funded_before: f64,
    actual: Option<Vec<f64>>,
}

impl Forward {
    /// The valuation month `v`.
    pub fn month(&self) -> usize {
        self.month
    }

    /// The simulations kept, by their index in the loss ratios given.
    pub fn kept(&self) -> &[usize] {
        &self.kept
    }

    /// Each kept simulation's ultimate loss ratio given the paid to date.
    pub fn ultimate(&self) -> &[f64] {
        &self.ultimate
    }

    /// The investor's flows in months `v..T`, one kept simulation a row
    /// (offset 0 is month `v`), with the totals since inception: the
    /// capital funded, the equity released, the fee and the commutation
    /// payment.
    pub fn flows(&self) -> &Cashflows {
        &self.flows
    }

    /// The net asset value taken as the investment at the valuation month.
    pub fn nav(&self) -> f64 {
        self.nav
    }

    /// The forward flows with the net asset value paid in at their first
    /// month, as [`CashflowTerms::run`] puts the opening capital in month
    /// 0: one row a kept simulation, months `v..T`. At month 0 they are the
    /// run's flows.
    pub fn forward_rows(&self) -> Vec<f64> {
        let m = self.flows.months;
        let mut rows = self.flows.flows.clone();
        if self.month > 0 {
            for row in rows.chunks_mut(m) {
                row[0] -= self.nav;
            }
        }
        rows
    }

    /// The actual flows in months `0..v` followed by each kept simulation's
    /// forward flows: one row a kept simulation, months `0..T`. Fails
    /// without the actual flows ([`ActualState::flows`]).
    pub fn since_inception_rows(&self) -> Result<Vec<f64>> {
        let Some(actual) = &self.actual else {
            return Err(Error::Data(
                "the return since inception needs the actual flows to date".into(),
            ));
        };
        let m = self.flows.months;
        Ok(self
            .flows
            .flows
            .chunks(m)
            .flat_map(|row| actual.iter().chain(row).copied())
            .collect())
    }

    /// Each kept simulation's net present value from the valuation month at
    /// the annual `rate`.
    pub fn forward_npv(&self, rate: f64) -> Result<Vec<f64>> {
        npv_rows(&self.forward_rows(), &years(self.flows.months), rate)
    }

    /// Each kept simulation's internal rate of return from the valuation
    /// month.
    pub fn forward_irr(&self, options: &IrrOptions) -> Result<Vec<Irr>> {
        irr_rows(&self.forward_rows(), &years(self.flows.months), options)
    }

    /// Each kept simulation's multiple on the capital from the valuation
    /// month, `1 + net / (NAV + funded after it)`, the net the sum of its
    /// forward row.
    pub fn forward_moic(&self) -> Result<Vec<f64>> {
        let m = self.flows.months;
        self.forward_rows()
            .chunks(m)
            .enumerate()
            .map(|(j, row)| {
                let funded = self.flows.funded[j] - self.funded_before;
                let base = if self.month == 0 {
                    self.flows.opening[j]
                } else {
                    self.nav
                };
                moic(row.iter().sum(), funded, base)
            })
            .collect()
    }

    /// Each kept simulation's net present value since inception at the
    /// annual `rate`.
    pub fn since_inception_npv(&self, rate: f64) -> Result<Vec<f64>> {
        let months = self.month + self.flows.months;
        npv_rows(&self.since_inception_rows()?, &years(months), rate)
    }

    /// Each kept simulation's internal rate of return since inception.
    pub fn since_inception_irr(&self, options: &IrrOptions) -> Result<Vec<Irr>> {
        let months = self.month + self.flows.months;
        irr_rows(&self.since_inception_rows()?, &years(months), options)
    }

    /// Each kept simulation's multiple on invested capital since inception,
    /// `1 + net / (funded + opening)`.
    pub fn since_inception_moic(&self) -> Result<Vec<f64>> {
        let months = self.month + self.flows.months;
        self.since_inception_rows()?
            .chunks(months)
            .enumerate()
            .map(|(j, row)| {
                moic(
                    row.iter().sum(),
                    self.flows.funded[j],
                    self.flows.opening[j],
                )
            })
            .collect()
    }
}

fn years(months: usize) -> Vec<f64> {
    let offsets: Vec<f64> = (0..months).map(|t| t as f64).collect();
    TimeUnit::Months.years(&offsets)
}

/// Actual against expected paid loss by month for a bound deal, from
/// [`deal_ave`] (#603, items M4 and M5; the columns named as
/// `prospicio_models::monitor` names them).
#[derive(Debug, Clone, PartialEq)]
pub struct AvEReport {
    /// The months `0..n`, one per actual value.
    pub month: Vec<usize>,
    /// The actual cumulative paid by the end of each month.
    pub actual: Vec<f64>,
    /// The mean over the simulations of the expected cumulative paid,
    /// `u_j P S_j(t)`.
    pub expected: Vec<f64>,
    /// `actual / expected`.
    pub ratio: Vec<f64>,
    /// The actual's percentile among the simulations' expected paid: the
    /// share at or below it, ties counted half.
    pub percentile: Vec<f64>,
    /// The percentile levels of [`quantiles`](Self::quantiles).
    pub levels: Vec<f64>,
    /// The expected paid at each level (numpy's linear quantile), one row a
    /// month, one column a level.
    pub quantiles: Vec<f64>,
}

/// The actual cumulative paid loss `actual[t]` at the end of each month
/// `t = 0, 1, ...` against what the priced simulations expected: each
/// simulation's `u_j P S_j(t)`, with `S_j` the terms' paid pattern or, when
/// `paid` is given, the simulation's own (#600). The expected paid comes
/// from the priced simulations, not conditioned ones (item M4).
///
/// A reported loss works the same way: give the reported pattern as `paid`
/// (one pattern for every simulation, [`PatternDraws::indexed`]) and the
/// actual reported loss.
///
/// ```
/// use prospicio_pricing::cashflow::{
///     Base, CashflowTerms, MonthlyPatterns, Requirement, Term, deal_ave,
/// };
///
/// let p = MonthlyPatterns::new(vec![1.0; 4], vec![1.0; 4], vec![0.25, 0.5, 0.75, 1.0]).unwrap();
/// let req = Requirement::new(vec![Term::new(Base::Outstanding, 1.0)]);
/// let terms = CashflowTerms::new(100.0, 4, p, req).unwrap();
/// // Simulated loss ratios 0.4, 0.6 and 0.8; 15 paid by month 0, 45 by month 1.
/// let r = deal_ave(&[15.0, 45.0], &terms, &[0.4, 0.6, 0.8], None, &[0.5]).unwrap();
/// assert_eq!(r.expected, [15.0, 30.0]);
/// // At month 0 the actual ties the middle simulation, at month 1 it is
/// // above all three.
/// assert_eq!(r.percentile, [0.5, 1.0]);
/// assert_eq!((r.ratio[1], r.quantiles[1]), (1.5, 30.0));
/// ```
pub fn deal_ave(
    actual: &[f64],
    terms: &CashflowTerms,
    u: &[f64],
    paid: Option<&PatternDraws>,
    levels: &[f64],
) -> Result<AvEReport> {
    terms.check()?;
    let n = u.len();
    if n == 0 || u.iter().any(|x| !x.is_finite()) {
        return Err(Error::Data(
            "deal_ave needs at least one finite loss ratio".into(),
        ));
    }
    if actual.iter().any(|x| !x.is_finite()) {
        return Err(Error::Data("the actual paid must be finite".into()));
    }
    if let Some(&bad) = levels.iter().find(|&&p| !(0.0..=1.0).contains(&p)) {
        return Err(Error::InvalidParameter {
            name: "levels",
            value: bad,
            reason: "percentile levels must be in [0, 1]",
        });
    }
    let width = match paid {
        Some(d) => {
            if d.len() != n {
                return Err(Error::Data(format!(
                    "{} paid patterns for {n} simulations",
                    d.len()
                )));
            }
            d.patterns()[0].len()
        }
        None => terms.patterns.len(),
    };
    if actual.len() > width {
        return Err(Error::Data(format!(
            "{} months of actual paid and patterns of {width} months",
            actual.len()
        )));
    }
    let pattern = |j: usize| match paid {
        Some(d) => d.pattern(j),
        None => terms.patterns.paid.as_slice(),
    };
    let pr = terms.premium;
    let mut report = AvEReport {
        month: (0..actual.len()).collect(),
        actual: actual.to_vec(),
        expected: Vec::with_capacity(actual.len()),
        ratio: Vec::with_capacity(actual.len()),
        percentile: Vec::with_capacity(actual.len()),
        levels: levels.to_vec(),
        quantiles: Vec::with_capacity(actual.len() * levels.len()),
    };
    let mut e = vec![0.0; n];
    for (t, &a) in actual.iter().enumerate() {
        for (j, x) in e.iter_mut().enumerate() {
            *x = u[j] * pr * pattern(j)[t];
        }
        let mean = e.iter().sum::<f64>() / n as f64;
        let below = e.iter().filter(|&&x| x < a).count() as f64;
        let ties = e.iter().filter(|&&x| x == a).count() as f64;
        report.expected.push(mean);
        report.ratio.push(a / mean);
        report.percentile.push((below + 0.5 * ties) / n as f64);
        e.sort_by(f64::total_cmp);
        for &p in levels {
            report
                .quantiles
                .push(QuantileMethod::Linear.quantile(&e, p)?);
        }
    }
    Ok(report)
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

    /// A fast and a slow paid pattern of the terms' length, both reaching
    /// the terms' ultimate share by the last month.
    fn fast_and_slow() -> (Vec<f64>, Vec<f64>) {
        let n = patterns().len();
        let ramp = |m: f64| (0..n).map(|t| ((t + 1) as f64 / m).min(1.0)).collect();
        (ramp(24.0), ramp(56.0))
    }

    #[test]
    fn the_terms_pattern_in_every_simulation_reproduces_run() {
        let us = [0.0, 0.4, 0.8, 1.3, 2.0];
        let cs = [0.0, 0.25, 0.1, 0.0, 0.3];
        for terms in variants() {
            let draws = PatternDraws::indexed(vec![terms.patterns.paid.clone()], vec![0; us.len()])
                .unwrap();
            let a = terms.run(&us, &cs).unwrap();
            let b = terms.run_with_patterns(&us, &cs, &draws).unwrap();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn a_mixture_of_two_patterns_is_two_runs_recombined() {
        let (fast, slow) = fast_and_slow();
        let us = [0.4, 0.9, 1.3, 0.7, 2.0, 0.0];
        let cs = [0.0, 0.2, 0.1, 0.0, 0.3, 0.1];
        let index = vec![0, 1, 1, 0, 1, 0];
        for terms in variants() {
            let draws =
                PatternDraws::indexed(vec![fast.clone(), slow.clone()], index.clone()).unwrap();
            let mixed = terms.run_with_patterns(&us, &cs, &draws).unwrap();
            let months = mixed.months();
            for (k, pattern) in [&fast, &slow].into_iter().enumerate() {
                let mut alone = terms.clone();
                alone.patterns.paid = pattern.clone();
                let sims: Vec<usize> = (0..us.len()).filter(|&j| index[j] == k).collect();
                let u: Vec<f64> = sims.iter().map(|&j| us[j]).collect();
                let c: Vec<f64> = sims.iter().map(|&j| cs[j]).collect();
                let run = alone.run(&u, &c).unwrap();
                for (i, &j) in sims.iter().enumerate() {
                    assert_eq!(
                        &mixed.flows()[j * months..(j + 1) * months],
                        &run.flows()[i * months..(i + 1) * months]
                    );
                    assert_eq!(mixed.funded()[j], run.funded()[i]);
                    assert_eq!(mixed.fee()[j], run.fee()[i]);
                }
            }
            // The pattern moves the result: a slow year differs from a fast one.
            let one = |p: &Vec<f64>| {
                let d = PatternDraws::new(vec![p.clone()]).unwrap();
                terms.run_with_patterns(&[1.3], &[0.0], &d).unwrap()
            };
            assert_ne!(one(&fast).flows(), one(&slow).flows());
        }
    }

    #[test]
    fn money_is_conserved_with_a_simulation_pattern() {
        let (fast, slow) = fast_and_slow();
        for terms in variants() {
            for p in [&fast, &slow] {
                for &u in &[0.4, 1.3, 2.0] {
                    let tr = terms.trace_with_pattern(u, 0.1, p).unwrap();
                    assert!(tr.conservation_error().abs() < 1e-9);
                    assert!(tr.collateral_balance.last().unwrap().abs() < 1e-9);
                    let d = PatternDraws::new(vec![p.clone()]).unwrap();
                    let run = terms.run_with_patterns(&[u], &[0.1], &d).unwrap();
                    assert_eq!(run.flows(), tr.investor.as_slice());
                }
            }
        }
    }

    fn lognormal(seed: u64, stream: u64, n: usize, mu: f64, sigma: f64) -> Vec<f64> {
        let mut rng = prospicio_core::StreamRng::new(seed, stream);
        (0..n)
            .map(|_| (mu + sigma * prospicio_math::special::norm_quantile(rng.next_open01())).exp())
            .collect()
    }

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn a_valuation_at_month_0_is_the_run() {
        let us = [0.0, 0.4, 0.8, 1.3, 2.0];
        let cs = [0.0, 0.25, 0.1, 0.0, 0.3];
        for terms in variants() {
            let run = terms.run(&us, &cs).unwrap();
            let val = terms.valuation(ActualState::default()).unwrap();
            assert_eq!(val.nav(), 0.0);
            for how in [Conditioning::BornhuetterFerguson, Conditioning::Truncation] {
                let f = val.run(&us, &cs, how).unwrap();
                assert_eq!(f.kept(), [0, 1, 2, 3, 4]);
                assert_eq!(f.flows(), &run);
                assert_eq!(f.forward_rows(), run.flows());
                let empty = ActualState {
                    flows: Some(vec![]),
                    ..ActualState::default()
                };
                let f = terms.valuation(empty).unwrap().run(&us, &cs, how).unwrap();
                assert_eq!(f.since_inception_rows().unwrap(), run.flows());
                assert_eq!(f.forward_npv(0.08).unwrap(), run.npv(0.08).unwrap());
                assert_eq!(f.forward_moic().unwrap(), f.since_inception_moic().unwrap());
            }
        }
    }

    #[test]
    fn valuing_a_path_at_its_own_loss_ratio_gives_its_flows() {
        // Under either conditioning a simulation at the path's own loss
        // ratio pays what the path pays, so the forward run is the rest of
        // the path and the since-inception flows are the whole path.
        for (i, terms) in variants().iter().enumerate() {
            for &u in &[0.0, 0.4, 0.9, 1.3, 2.0] {
                let c = 0.2;
                let path = terms.trace(u, c).unwrap();
                for v in [1, 6, 12, 18] {
                    let state = ActualState::from_trace(&path, v).unwrap();
                    let val = terms.valuation(state).unwrap();
                    for how in [Conditioning::BornhuetterFerguson, Conditioning::Truncation] {
                        let f = val.run(&[u], &[c], how).unwrap();
                        assert!(
                            close(f.flows().flows(), &path.investor[v..], 1e-9),
                            "variant {i}, u {u}, month {v}, {how:?}"
                        );
                        assert!(close(
                            &f.since_inception_rows().unwrap(),
                            &path.investor,
                            1e-9
                        ));
                        assert!((f.flows().fee()[0] - path.fee).abs() < 1e-9);
                        let tr = val.trace(u, c, how).unwrap();
                        assert!(
                            tr.conservation_error().abs() < 1e-9,
                            "variant {i}, month {v}"
                        );
                        assert!(close(
                            &tr.collateral_balance,
                            &path.collateral_balance[v..],
                            1e-9
                        ));
                        assert!(tr.operating_balance.last().unwrap().abs() < 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn money_is_conserved_from_a_valuation_for_any_simulation() {
        for terms in variants() {
            let path = terms.trace(0.9, 0.2).unwrap();
            let val = terms
                .valuation(ActualState::from_trace(&path, 12).unwrap())
                .unwrap();
            for &u in &[0.0, 0.5, 1.1, 2.0] {
                for how in [Conditioning::BornhuetterFerguson, Conditioning::Truncation] {
                    let Ok(tr) = val.trace(u, 0.1, how) else {
                        // Truncation drops a simulation below the paid.
                        assert_eq!(how, Conditioning::Truncation);
                        assert!(u * 100.0 < path.paid[..12].iter().sum::<f64>());
                        continue;
                    };
                    assert!(tr.conservation_error().abs() < 1e-9);
                    assert!(tr.collateral_balance.last().unwrap().abs() < 1e-9);
                    assert!(tr.operating_balance.last().unwrap().abs() < 1e-9);
                    // The losses paid from the valuation are the conditioned
                    // ultimate's rest along the pattern.
                    let a: f64 = path.paid[..12].iter().sum();
                    let s = terms.patterns.paid[11];
                    let e = terms.patterns.paid[29];
                    let want = match how {
                        Conditioning::BornhuetterFerguson => u * 100.0 * (e - s),
                        Conditioning::Truncation => (u * 100.0 - a) * (e - s) / (1.0 - s),
                    };
                    assert!((tr.paid.iter().sum::<f64>() - want).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn premium_short_of_the_schedule_is_collected_at_valuation() {
        let terms = full();
        let path = terms.trace(0.9, 0.2).unwrap();
        let mut state = ActualState::from_trace(&path, 6).unwrap();
        let val = terms.valuation(state.clone()).unwrap();
        let on_time = val
            .trace(0.9, 0.2, Conditioning::BornhuetterFerguson)
            .unwrap();
        state.premium_collected -= 5.0;
        state.collateral -= 5.0;
        let late = terms
            .valuation(state)
            .unwrap()
            .trace(0.9, 0.2, Conditioning::BornhuetterFerguson)
            .unwrap();
        assert!((late.premium[0] - on_time.premium[0] - 5.0).abs() < 1e-12);
        assert!((late.premium[1] - on_time.premium[1]).abs() < 1e-12);
        assert!(late.conservation_error().abs() < 1e-9);
    }

    #[test]
    fn truncation_drops_what_has_paid_less_than_the_actual() {
        let mut terms = full();
        terms.premium_flow.adjustments.clear();
        let path = terms.trace(1.0, 0.2).unwrap();
        let state = ActualState::from_trace(&path, 24).unwrap();
        let a = state.paid;
        let val = terms.valuation(state).unwrap();
        let us = [0.3, 0.45, 0.55, 1.0, 1.5];
        let f = val.run(&us, &[0.2; 5], Conditioning::Truncation).unwrap();
        let kept: Vec<usize> = (0..5).filter(|&j| us[j] * 100.0 >= a).collect();
        assert_eq!(f.kept(), kept.as_slice());
        assert_eq!(f.flows().len(), kept.len());
        let bf = val
            .run(&us, &[0.2; 5], Conditioning::BornhuetterFerguson)
            .unwrap();
        assert_eq!(bf.kept().len(), 5);
        // Every BF ultimate is at least the paid to date.
        assert!(bf.ultimate().iter().all(|&x| x * 100.0 >= a - 1e-12));
        assert!(val.run(&[0.1], &[0.2], Conditioning::Truncation).is_err());
    }

    #[test]
    fn the_forward_return_starts_from_the_net_asset_value() {
        let mut terms = full();
        terms.funding.rule = FundingRule::Shortfall;
        let path = terms.trace(0.7, 0.2).unwrap();
        let state = ActualState::from_trace(&path, 12).unwrap();
        let nav = state.collateral + state.operating;
        let val = terms.valuation(state).unwrap();
        let f = val
            .run(
                &[0.5, 0.7, 1.2],
                &[0.2; 3],
                Conditioning::BornhuetterFerguson,
            )
            .unwrap();
        assert_eq!(f.nav(), nav);
        let m = f.flows().months();
        assert_eq!(m, 18);
        let rows = f.forward_rows();
        for j in 0..3 {
            let raw = &f.flows().flows()[j * m..(j + 1) * m];
            assert_eq!(rows[j * m], raw[0] - nav);
            assert_eq!(&rows[j * m + 1..(j + 1) * m], &raw[1..]);
        }
        // A better loss ratio is worth more, from now and since inception.
        let fwd = f.forward_npv(0.05).unwrap();
        let all = f.since_inception_npv(0.05).unwrap();
        assert!(fwd[0] > fwd[1] && fwd[1] > fwd[2], "{fwd:?}");
        assert!(all[0] > all[1] && all[1] > all[2], "{all:?}");
        let moic = f.since_inception_moic().unwrap();
        assert!(moic[0] > moic[2]);
        let irr = f.forward_irr(&IrrOptions::default()).unwrap();
        for r in irr.iter().filter(|r| r.found) {
            assert!(r.npv.abs() < 1e-6);
        }
        assert_eq!(
            f.since_inception_irr(&IrrOptions::default()).unwrap().len(),
            3
        );
    }

    #[test]
    fn refuses_valuations_it_cannot_run() {
        let terms = full();
        let path = terms.trace(0.9, 0.2).unwrap();
        // At or past commutation.
        assert!(ActualState::from_trace(&path, 30).is_err());
        let mut late = ActualState::from_trace(&path, 12).unwrap();
        late.month = 30;
        assert!(terms.valuation(late).is_err());
        // After the adjustment at month 18.
        let after = ActualState::from_trace(&path, 19).unwrap();
        assert!(terms.valuation(after).is_err());
        assert!(
            terms
                .valuation(ActualState::from_trace(&path, 18).unwrap())
                .is_ok()
        );
        // Month 0 with something in it.
        let busy = ActualState {
            paid: 1.0,
            ..ActualState::default()
        };
        assert!(terms.valuation(busy).is_err());
        // The obligation rule needs its reference.
        let mut no_ref = ActualState::from_trace(&path, 6).unwrap();
        no_ref.funding_reference = None;
        assert!(terms.valuation(no_ref.clone()).is_err());
        let mut shortfall = terms.clone();
        shortfall.funding.rule = FundingRule::Shortfall;
        assert!(shortfall.valuation(no_ref.clone()).is_ok());
        // Actual flows one a month to the valuation.
        let mut flows = ActualState::from_trace(&path, 6).unwrap();
        flows.flows = Some(vec![0.0; 5]);
        assert!(terms.valuation(flows).is_err());
        let mut nan = ActualState::from_trace(&path, 6).unwrap();
        nan.collateral = f64::NAN;
        assert!(terms.valuation(nan).is_err());
        // The return since inception needs the actual flows.
        let mut blind = ActualState::from_trace(&path, 6).unwrap();
        blind.flows = None;
        let f = terms
            .valuation(blind)
            .unwrap()
            .run(&[0.9], &[0.2], Conditioning::BornhuetterFerguson)
            .unwrap();
        assert!(f.since_inception_rows().is_err());
        assert!(f.forward_npv(0.05).is_ok());
        // A trace from a valuation is not a path from inception.
        let val = terms
            .valuation(ActualState::from_trace(&path, 6).unwrap())
            .unwrap();
        let tr = val
            .trace(0.9, 0.2, Conditioning::BornhuetterFerguson)
            .unwrap();
        assert!(ActualState::from_trace(&tr, 3).is_err());
        assert!(
            condition_on_paid(&[0.5], 0.0, 1.0, 0.5, Conditioning::BornhuetterFerguson).is_err()
        );
    }

    /// The Kolmogorov-Smirnov distance of values in [0, 1] from the
    /// uniform.
    fn ks_uniform(mut x: Vec<f64>) -> f64 {
        x.sort_by(f64::total_cmp);
        let n = x.len() as f64;
        x.iter()
            .enumerate()
            .map(|(i, &v)| (v - i as f64 / n).abs().max(((i + 1) as f64 / n - v).abs()))
            .fold(0.0, f64::max)
    }

    #[test]
    fn the_actual_percentile_is_uniform_over_paths() {
        // Paths drawn from the priced distribution: the actual's percentile
        // among the priced simulations is uniform at every month, and the
        // realised return's percentile among the conditioned forward
        // returns is uniform too, so the forward distribution brackets the
        // realised return as often as it says.
        let mut terms = full();
        terms.funding.rule = FundingRule::Shortfall;
        terms.premium_flow.adjustments.clear();
        let sims = lognormal(7, 0, 2_000, -0.3, 0.35);
        let paths = lognormal(7, 1, 300, -0.3, 0.35);
        let crit = 1.63 / (paths.len() as f64).sqrt(); // 1% critical value
        let mut ave = Vec::new();
        let mut realised = Vec::new();
        let c = vec![0.2; sims.len()];
        for &u in &paths {
            let path = terms.trace(u, 0.2).unwrap();
            let actual: Vec<f64> = path
                .paid
                .iter()
                .scan(0.0, |s, x| {
                    *s += x;
                    Some(*s)
                })
                .collect();
            let r = deal_ave(&actual[..12], &terms, &sims, None, &[]).unwrap();
            ave.push(r.percentile[11]);
            let val = terms
                .valuation(ActualState::from_trace(&path, 12).unwrap())
                .unwrap();
            let f = val
                .run(&sims, &c, Conditioning::BornhuetterFerguson)
                .unwrap();
            let npv = f.since_inception_npv(0.05).unwrap();
            let own: f64 = path
                .investor
                .iter()
                .enumerate()
                .map(|(t, x)| x / 1.05_f64.powf(t as f64 / 12.0))
                .sum();
            let below = npv.iter().filter(|&&x| x < own).count() as f64;
            realised.push(below / npv.len() as f64);
        }
        let d_ave = ks_uniform(ave);
        let d_npv = ks_uniform(realised);
        assert!(d_ave < crit, "{d_ave} against {crit}");
        assert!(d_npv < crit, "{d_npv} against {crit}");
    }

    #[test]
    fn deal_ave_reads_each_simulations_pattern() {
        let terms = full();
        let (fast, slow) = fast_and_slow();
        let us = [0.5, 1.0];
        let draws = PatternDraws::new(vec![fast.clone(), slow.clone()]).unwrap();
        let actual = [3.0, 6.0, 9.0];
        let r = deal_ave(&actual, &terms, &us, Some(&draws), &[0.0, 1.0]).unwrap();
        for t in 0..3 {
            let e = [0.5 * 100.0 * fast[t], 100.0 * slow[t]];
            assert!((r.expected[t] - 0.5 * (e[0] + e[1])).abs() < 1e-12);
            assert_eq!(r.quantiles[2 * t], e[0].min(e[1]));
            assert_eq!(r.quantiles[2 * t + 1], e[0].max(e[1]));
        }
        assert!(deal_ave(&actual, &terms, &[0.5], Some(&draws), &[]).is_err());
        assert!(deal_ave(&actual, &terms, &us, None, &[1.5]).is_err());
        assert!(deal_ave(&[0.0; 61], &terms, &us, None, &[]).is_err());
    }

    #[test]
    fn the_forward_distribution_narrows_as_the_deal_runs_off() {
        // The finding in knowledge/findings/deal-valuation-conditioning.md:
        // under Bornhuetter-Ferguson conditioning the sd of the since-
        // inception NPV falls with the share paid; truncation keeps its
        // simulations whole and drops more as the deal runs off.
        let mut terms = full();
        terms.funding.rule = FundingRule::Shortfall;
        terms.premium_flow.adjustments.clear();
        let sims = lognormal(11, 0, 4_000, -0.3, 0.35);
        let c = vec![0.2; sims.len()];
        let path = terms.trace(0.9, 0.2).unwrap();
        let sd = |x: &[f64]| {
            let m = x.iter().sum::<f64>() / x.len() as f64;
            (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
        };
        let mut bf_sd = Vec::new();
        let mut kept = Vec::new();
        for v in [0, 6, 12, 18, 24, 29] {
            let state = ActualState::from_trace(&path, v).unwrap();
            let val = terms.valuation(state).unwrap();
            let f = val
                .run(&sims, &c, Conditioning::BornhuetterFerguson)
                .unwrap();
            bf_sd.push(sd(&f.since_inception_npv(0.05).unwrap()));
            let t = val.run(&sims, &c, Conditioning::Truncation).unwrap();
            kept.push(t.kept().len() as f64 / sims.len() as f64);
        }
        assert!(bf_sd.windows(2).all(|w| w[1] < w[0]), "{bf_sd:?}");
        assert!(kept.windows(2).all(|w| w[1] <= w[0]), "{kept:?}");
        // Close to the unpaid share 1 - S(v - 1) of a 48-month pattern:
        // 1, 0.875, 0.75, 0.625, 0.5, 0.396.
        let ratio: Vec<f64> = bf_sd.iter().map(|s| s / bf_sd[0]).collect();
        let want = [1.0, 0.8798, 0.7520, 0.6177, 0.4879, 0.3807];
        assert!(close(&ratio, &want, 1e-4), "{ratio:?}");
        let want = [1.0, 1.0, 0.9995, 0.9855, 0.92025, 0.802];
        assert!(close(&kept, &want, 1e-12), "{kept:?}");
    }

    #[test]
    fn refuses_patterns_that_do_not_fit() {
        let terms = full();
        let (fast, _) = fast_and_slow();
        let n = fast.len();
        let mut falls = fast.clone();
        falls[3] = falls[2] - 0.01;
        assert!(PatternDraws::new(vec![falls]).is_err());
        let mut short_end = fast.clone();
        short_end[n - 1] = 0.9;
        assert!(PatternDraws::new(vec![fast.clone(), short_end.clone()]).is_err());
        assert!(PatternDraws::new(vec![fast[..n - 1].to_vec(), fast.clone()]).is_err());
        assert!(PatternDraws::indexed(vec![fast.clone()], vec![1]).is_err());
        assert!(PatternDraws::new(vec![]).is_err());
        let two = PatternDraws::new(vec![fast.clone(), fast.clone()]).unwrap();
        assert!(terms.run_with_patterns(&[0.5], &[0.0], &two).is_err());
        let short = PatternDraws::new(vec![fast[..n - 1].to_vec()]).unwrap();
        assert!(terms.run_with_patterns(&[0.5], &[0.0], &short).is_err());
        let low = PatternDraws::new(vec![fast.iter().map(|x| 0.9 * x).collect()]).unwrap();
        assert!(terms.run_with_patterns(&[0.5], &[0.0], &low).is_err());
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
            .map(|t| {
                terms
                    .requirement
                    .at(&terms.month(t, &PaidPath::new(0.9, 100.0, &terms.patterns.paid)))
            })
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
