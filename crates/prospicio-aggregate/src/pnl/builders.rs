//! Builders: the ledger of a book (premium less its annual loss and
//! expenses) and of the annual covers it buys, as `aggregate`'s `pnl` and
//! `xpnl` builders over an aggregate (annual) program
//! (`docs/design/pnl.md`, PR 4).
//!
//! A [`Book`] is the seller: its premium, its expenses and the labels of
//! its legs. [`Book::ledger`] lays it out over a source of annual gross
//! losses in one of three [`Face`]s:
//!
//! - [`Face::Consolidated`], `pnl`: one group, the net position. Without a
//!   tower it is premium less loss less expenses over the gross; with one,
//!   net premium (gross less the ceded premiums) less the net loss less the
//!   commission received (booked as a negative obligation) less expenses,
//!   over the net loss at each atom. A loss-basis expense books as its
//!   rate times the atom's gross loss, so the margin is the walk's net
//!   margin atom by atom ([`Book::expense_at_expected_loss`] books it at
//!   the expected gross loss instead).
//! - [`Face::Walk`], `xpnl`: the gross book, then one buy group per stage
//!   of the tower, every leg a function of the gross loss, with the
//!   running net and the covers' total impact. Without a tower it is the
//!   gross book alone as a one-step walk.
//! - [`Face::PeelTopDown`] and [`Face::PeelBottomUp`], `xpnl ... peel`:
//!   the walk with one buy group per layer (the highest first, or the
//!   lowest, within each stage) and an `All aggregate` subtotal.
//!
//! Every layer's terms are taken at the annual total, as
//! [`Tower::on_annual_grid`] does: its ceded loss after its aggregate
//! deductible, limit and corridor, at its share. Its premium (swing rated,
//! or fixed), flat commission, sliding-scale commission and profit
//! commission are legs at each atom, from the layer's own functions at its
//! annual ceded loss. Reinstatement premiums are not booked; a layer with
//! them is refused, and its legs can be written by hand with [`Leg::of`].
//! [`pnl_program`] prices the book and its covers from target combined
//! ratios.

use prospicio_core::{Error, Result};

use super::{Group, Layout, Ledger, Leg, LegKind, Role, Source};
use crate::bivariate::{BivariateReport, JointGrid, fft_bivariate};
use crate::reinsurance::{Commission, Layer, LossSensitivePremium, Tower};
use prospicio_prob::{Counting, Grid};

/// What an expense is a fraction of, `aggregate`'s expense bases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpenseBasis {
    /// An amount.
    Fixed,
    /// A fraction of the gross premium.
    Premium,
    /// A fraction of the gross loss: loss adjustment expense, stochastic
    /// where the gross loss is on the source.
    Loss,
}

impl ExpenseBasis {
    /// `fixed`, `premium` or `loss`.
    pub fn name(self) -> &'static str {
        match self {
            ExpenseBasis::Fixed => "fixed",
            ExpenseBasis::Premium => "premium",
            ExpenseBasis::Loss => "loss",
        }
    }
}

/// One expense leg: one or more terms (`aggregate`'s terms joined by
/// `and`), with an optional label.
///
/// Unlabelled, a book's only expense is the leg `expense`; with several, a
/// single-basis one is `<basis> expense` and a mixed one `expense`, with
/// `(2)`, `(3)` … added to a name already taken.
#[derive(Debug, Clone, PartialEq)]
pub struct Expense {
    /// The leg's label, if given.
    pub label: Option<String>,
    /// The terms: a basis and an amount or rate.
    pub terms: Vec<(ExpenseBasis, f64)>,
}

impl Expense {
    /// An expense of one term.
    pub fn new(basis: ExpenseBasis, value: f64) -> Self {
        Self {
            label: None,
            terms: vec![(basis, value)],
        }
    }

    /// A fixed amount.
    pub fn fixed(amount: f64) -> Self {
        Self::new(ExpenseBasis::Fixed, amount)
    }

    /// A fraction of the gross premium.
    pub fn premium(rate: f64) -> Self {
        Self::new(ExpenseBasis::Premium, rate)
    }

    /// A fraction of the gross loss.
    pub fn loss(rate: f64) -> Self {
        Self::new(ExpenseBasis::Loss, rate)
    }

    /// Adds a term to the same leg.
    pub fn and(mut self, basis: ExpenseBasis, value: f64) -> Self {
        self.terms.push((basis, value));
        self
    }

    /// Labels the leg.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// `(fixed amount plus premium share, loss rate)` at gross premium
    /// `premium`.
    fn split(&self, premium: f64) -> (f64, f64) {
        let (mut scalar, mut rate) = (0.0, 0.0);
        for &(basis, v) in &self.terms {
            match basis {
                ExpenseBasis::Fixed => scalar += v,
                ExpenseBasis::Premium => scalar += v * premium,
                ExpenseBasis::Loss => rate += v,
            }
        }
        (scalar, rate)
    }

    fn default_name(&self) -> String {
        let first = self.terms[0].0;
        if self.terms.iter().all(|(b, _)| *b == first) {
            format!("{} expense", first.name())
        } else {
            "expense".into()
        }
    }
}

/// The gross premium whose own expenses leave the technical premium
/// `technical`, `aggregate`'s `derive premium`: `(T + F) / (1 − r)` with
/// `F` the fixed amounts and `r` the premium rates of every expense.
///
/// ```
/// use prospicio_aggregate::pnl::{Expense, derive_premium};
///
/// // 10% of premium and 1 fixed leave 5: (5 + 1) / 0.9.
/// let p = derive_premium(5.0, &[Expense::premium(0.1), Expense::fixed(1.0)]).unwrap();
/// assert!((p - 6.0 / 0.9).abs() < 1e-12);
/// ```
///
/// # Errors
///
/// A loss-basis term (the loss is not known in advance), premium rates
/// totalling 1 or more, or a value that is not finite.
pub fn derive_premium(technical: f64, expenses: &[Expense]) -> Result<f64> {
    let (mut fixed, mut rate) = (0.0, 0.0);
    for e in expenses {
        for &(basis, v) in &e.terms {
            if !v.is_finite() {
                return Err(Error::Data(format!(
                    "an expense term must be finite, not {v}"
                )));
            }
            match basis {
                ExpenseBasis::Fixed => fixed += v,
                ExpenseBasis::Premium => rate += v,
                ExpenseBasis::Loss => {
                    return Err(Error::Data(
                        "a premium cannot be grossed up for a loss-basis expense; use fixed or \
                         premium expenses"
                            .into(),
                    ));
                }
            }
        }
    }
    if rate >= 1.0 {
        return Err(Error::Data(format!(
            "the premium expense rates total {rate}, at or above 1; no premium covers them"
        )));
    }
    Ok((technical + fixed) / (1.0 - rate))
}

/// A premium rounded as `aggregate` writes one: to whole units above 100,
/// to cents at or below it, ties to even (Python's `round`).
///
/// ```
/// use prospicio_aggregate::pnl::round_premium;
///
/// assert_eq!(round_premium(714.12), 714.0);
/// assert_eq!(round_premium(16.285), 16.29);
/// assert_eq!(round_premium(146.5), 146.0);
/// ```
pub fn round_premium(premium: f64) -> f64 {
    if premium > 100.0 {
        premium.round_ties_even()
    } else {
        // Formatting rounds the exact binary value to cents, ties to even,
        // as Python's round(x, 2).
        format!("{premium:.2}").parse().unwrap_or(premium)
    }
}

/// The premium sized from a target loss ratio, `aggregate`'s
/// `pnl_program` without the ladder: the expected loss over `loss_ratio`,
/// rounded by [`round_premium`].
///
/// # Errors
///
/// An expected loss or loss ratio that is not positive and finite.
pub fn size_premium(expected_loss: f64, loss_ratio: f64) -> Result<f64> {
    if !(expected_loss.is_finite() && expected_loss > 0.0) {
        return Err(Error::Data(format!(
            "a premium is sized from a positive expected loss, not {expected_loss}"
        )));
    }
    if !(loss_ratio.is_finite() && loss_ratio > 0.0) {
        return Err(Error::Data(format!(
            "the loss ratio must be positive and finite, not {loss_ratio}"
        )));
    }
    Ok(round_premium(expected_loss / loss_ratio))
}

/// One layer of [`pnl_program`]: its expected ceded loss at its placed
/// share, the share, its target combined ratio (expected ceded loss over
/// ceded premium), its premium if the contract already states one (in
/// currency, at the placed share), which the ladder then keeps, and whether
/// it is on the aggregate tier (which inures behind the occurrence tier, so
/// its rate is quoted on the gross premium less the occurrence premiums).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgramLayer {
    /// Expected ceded loss, at the placed share.
    pub ceded_loss: f64,
    /// The placed share, in `(0, 1]`.
    pub share: f64,
    /// Target combined ratio: expected ceded loss over ceded premium.
    pub combined_ratio: f64,
    /// A premium the contract already states, at the placed share.
    pub stated: Option<f64>,
    /// On the aggregate tier rather than the occurrence tier.
    pub aggregate: bool,
}

impl ProgramLayer {
    /// An occurrence layer the ladder prices.
    pub fn occurrence(ceded_loss: f64, share: f64, combined_ratio: f64) -> Self {
        Self {
            ceded_loss,
            share,
            combined_ratio,
            stated: None,
            aggregate: false,
        }
    }

    /// An aggregate layer the ladder prices.
    pub fn aggregate(ceded_loss: f64, share: f64, combined_ratio: f64) -> Self {
        Self {
            aggregate: true,
            ..Self::occurrence(ceded_loss, share, combined_ratio)
        }
    }

    /// The layer with a premium its contract states (at the placed share),
    /// which the ladder keeps.
    pub fn stated(mut self, premium: f64) -> Self {
        self.stated = Some(premium);
        self
    }
}

/// What [`pnl_program`] prices: the gross premium to book and each layer's
/// deposit at 100% placement, and the same as a rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// The gross premium, rounded by [`round_premium`] (unrounded when an
    /// engine's stated premium is only grossed up, `aggregate`'s `derive
    /// premium`).
    pub premium: f64,
    /// The technical premium before expenses: the net technical premium
    /// plus the ceded premiums.
    pub technical: f64,
    /// Each layer's deposit at 100% placement (a stated premium over its
    /// share).
    pub deposits: Vec<f64>,
    /// Each ladder-priced layer's deposit respelled as a rate, to eight
    /// significant figures, `aggregate`'s `premium_style='rate'`: on the
    /// booked premium for an occurrence layer, on the booked premium less
    /// the occurrence premiums for an aggregate one (each resolved from its
    /// rate). `None` for a layer that states its premium.
    pub rates: Vec<Option<f64>>,
    /// The ceded premium of all layers, at their shares.
    pub ceded_premium: f64,
}

/// The premium ladder of `aggregate`'s `pnl_program`, as numbers: price
/// the net book at `net_combined_ratio` and each layer at its own, then
/// gross up once for a premium-basis `expense_ratio`.
///
/// With `E_net` the expected net loss and `E_j` layer `j`'s expected ceded
/// loss at share `s_j`: the net technical premium `T_net = E_net / c_net`,
/// each layer's deposit `d_j = round(E_j / c_j / s_j)` (quoted at 100%, as
/// a contract states it), the technical premium `TP = T_net + Σ s_j d_j`,
/// and the booked premium `P = round(TP / (1 − ER))`. The book's own
/// margin is then `(1 − c_net) T_net` whatever the cover costs. Layers
/// stating their own premium keep it. Each priced deposit is also given as
/// a rate ([`Program::rates`]). An engine that states its own technical
/// premium is priced by [`pnl_program_stated`]. `aggregate` writes the
/// result as DecL.
///
/// ```
/// use prospicio_aggregate::pnl::{ProgramLayer, pnl_program};
///
/// let layer = ProgramLayer::occurrence(86.0, 1.0, 0.6);
/// let p = pnl_program(406.0, 0.9, 0.25, &[layer]).unwrap();
/// assert_eq!(p.deposits, vec![143.0]); // 86 / 0.6, rounded
/// // (406 / 0.9 + 143) / (1 − 0.25) = 792.15, booked as 792.
/// assert_eq!(p.premium, 792.0);
/// assert_eq!(p.rates, vec![Some(0.18055556)]); // 143 / 792
/// ```
///
/// # Errors
///
/// A ratio that is not positive and finite, an expense ratio outside
/// `[0, 1)`, a share outside `(0, 1]`, or a rate that cannot be quoted (a
/// subject premium at or below zero).
pub fn pnl_program(
    net_loss: f64,
    net_combined_ratio: f64,
    expense_ratio: f64,
    layers: &[ProgramLayer],
) -> Result<Program> {
    positive("the net combined ratio", net_combined_ratio)?;
    positive("the expected net loss", net_loss)?;
    ladder(
        Some(net_loss / net_combined_ratio),
        None,
        expense_ratio,
        layers,
    )
}

/// The premium ladder for an engine that states its own technical premium
/// `stated` (`aggregate`'s `premium at lr` exposure): the stated premium
/// takes the net book's place, so `TP = stated + Σ s_j d_j` and `P =
/// round(TP / (1 − ER))`. With no layer to pay for, the premium is the
/// stated one grossed up and left unrounded, as `aggregate`'s `derive
/// premium`.
///
/// ```
/// use prospicio_aggregate::pnl::pnl_program_stated;
///
/// let p = pnl_program_stated(1000.0, 0.25, &[]).unwrap();
/// assert!((p.premium - 1000.0 / 0.75).abs() < 1e-12);
/// ```
///
/// # Errors
///
/// As [`pnl_program`], and a stated premium that is not positive.
pub fn pnl_program_stated(
    stated: f64,
    expense_ratio: f64,
    layers: &[ProgramLayer],
) -> Result<Program> {
    positive("the stated technical premium", stated)?;
    ladder(None, Some(stated), expense_ratio, layers)
}

fn positive(name: &str, v: f64) -> Result<()> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(Error::Data(format!(
            "{name} must be positive and finite, not {v}"
        )))
    }
}

/// A number to eight significant figures, as Python's `f"{x:.8g}"`.
fn eight_figures(x: f64) -> f64 {
    format!("{x:.7e}").parse().unwrap_or(x)
}

fn ladder(
    net_technical: Option<f64>,
    stated: Option<f64>,
    expense_ratio: f64,
    layers: &[ProgramLayer],
) -> Result<Program> {
    let mut deposits = Vec::with_capacity(layers.len());
    let mut ceded_premium = 0.0;
    for l in layers {
        if !(l.share > 0.0 && l.share <= 1.0) {
            return Err(Error::Data(format!(
                "a layer's share is in (0, 1], not {}",
                l.share
            )));
        }
        match l.stated {
            Some(p) => {
                ceded_premium += p;
                deposits.push(p / l.share);
            }
            None => {
                positive("a layer's combined ratio", l.combined_ratio)?;
                let d = round_premium(l.ceded_loss / l.combined_ratio / l.share);
                ceded_premium += l.share * d;
                deposits.push(d);
            }
        }
    }
    let expenses = [Expense::premium(expense_ratio)];
    let (technical, premium) = match (net_technical, stated) {
        (Some(t), _) => {
            let technical = t + ceded_premium;
            (
                technical,
                round_premium(derive_premium(technical, &expenses)?),
            )
        }
        (None, Some(s)) if ceded_premium == 0.0 => (s, derive_premium(s, &expenses)?),
        (None, Some(s)) => {
            let technical = s + ceded_premium;
            (
                technical,
                round_premium(derive_premium(technical, &expenses)?),
            )
        }
        (None, None) => unreachable!("the ladder needs a net or a stated premium"),
    };
    // The rate respelling: occurrence rates on the booked premium, then
    // aggregate rates on it less the occurrence premiums as they resolve.
    let mut rates = vec![None; layers.len()];
    let mut base = premium;
    for tier in [false, true] {
        let mut resolved = 0.0;
        for (j, l) in layers.iter().enumerate() {
            if l.aggregate != tier {
                continue;
            }
            match l.stated {
                Some(p) => resolved += p,
                None => {
                    if base <= 0.0 {
                        return Err(Error::Data(format!(
                            "layer {} cannot be quoted as a rate: its subject premium is {base}",
                            j + 1
                        )));
                    }
                    let rate = eight_figures(deposits[j] / base);
                    resolved += l.share * rate * base;
                    rates[j] = Some(rate);
                }
            }
        }
        base -= resolved;
    }
    Ok(Program {
        premium,
        technical,
        deposits,
        rates,
        ceded_premium,
    })
}

/// How [`Book::ledger`] lays the book and its covers out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    /// One group, the net position (`aggregate`'s `pnl`).
    Consolidated,
    /// The gross book, then one group per stage of the tower (`xpnl`).
    Walk,
    /// The gross book, then one group per layer, the highest first within
    /// each stage (`xpnl ... peel top-down`).
    PeelTopDown,
    /// The gross book, then one group per layer, the lowest first within
    /// each stage (`xpnl ... peel bottom-up`).
    PeelBottomUp,
}

/// The seller's book: a premium, expenses and the labels of its legs.
///
/// ```
/// use prospicio_aggregate::pnl::{Book, Expense, Face, Source};
/// use prospicio_aggregate::{Layer, Tower};
///
/// // Example 1's loss, 10 of premium, a 10% premium expense and an
/// // aggregate cover of 3 xs 5 bought for 1 with a 10% ceding commission.
/// let gross = Source::new(
///     vec![2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
///     vec![0.25, 0.3, 0.09, 0.2, 0.12, 0.04],
/// )
/// .unwrap();
/// let cover = Layer::stop_loss("agg 3 xs 5", 3.0, 5.0).unwrap()
///     .deposit_premium(1.0).unwrap()
///     .ceding_commission(0.1).unwrap();
/// let tower = Tower::new(vec![cover]).unwrap();
/// let book = Book::new("Ex3", 10.0).expense(Expense::premium(0.1));
/// let walk = book.ledger(&gross, Some(&tower), Face::Walk).unwrap();
/// let net = book.ledger(&gross, Some(&tower), Face::Consolidated).unwrap();
/// assert!((walk.result().mean() - 4.54).abs() < 1e-12);
/// assert!((net.result().mean() - 4.54).abs() < 1e-12);
/// assert!((walk.row("agg 3 xs 5 recovery").unwrap().mean() - 0.24).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Book {
    name: String,
    premium: f64,
    premium_label: Option<String>,
    loss_label: Option<String>,
    step_label: Option<String>,
    expenses: Vec<Expense>,
    retro: Option<LossSensitivePremium>,
    expense_at_mean: bool,
}

impl Book {
    /// A book named `name` (the ledger's name) with gross premium
    /// `premium`.
    pub fn new(name: impl Into<String>, premium: f64) -> Self {
        Self {
            name: name.into(),
            premium,
            premium_label: None,
            loss_label: None,
            step_label: None,
            expenses: Vec::new(),
            retro: None,
            expense_at_mean: false,
        }
    }

    /// The premium leg's label, `Premium` by default (`<label> (net)` on
    /// the consolidated face with a tower, else `net premium`).
    pub fn premium_label(mut self, label: impl Into<String>) -> Self {
        self.premium_label = Some(label.into());
        self
    }

    /// The loss leg's label, `Loss` by default (`<label> (net)` on the
    /// consolidated face with a tower); it also labels the gross book's
    /// margin on a walk, `Gross` by default.
    pub fn loss_label(mut self, label: impl Into<String>) -> Self {
        self.loss_label = Some(label.into());
        self
    }

    /// The gross book's step on a walk, `Gross` by default.
    pub fn step_label(mut self, label: impl Into<String>) -> Self {
        self.step_label = Some(label.into());
        self
    }

    /// Adds an expense leg.
    pub fn expense(mut self, expense: Expense) -> Self {
        self.expenses.push(expense);
        self
    }

    /// Books a loss-basis expense on the consolidated face at the expected
    /// gross loss, `rate × E[gross]`, as `aggregate`'s guaranteed-cost and
    /// occurrence builders do, rather than at each atom's gross loss (the
    /// default, as its reinstatement builder). The mean is the same; the
    /// default makes the consolidated margin the walk's net margin atom by
    /// atom.
    pub fn expense_at_expected_loss(mut self) -> Self {
        self.expense_at_mean = true;
        self
    }

    /// Rates the book retrospectively, `aggregate`'s `retro <collar>
    /// premium`: the premium is `terms` at the year's gross loss, and the
    /// gross premium given to [`Book::new`] (the collar's basic premium in
    /// `aggregate`) is what premium-basis expenses are taken on. A
    /// retrospectively rated book buys no cover in these builders.
    pub fn retro(mut self, terms: LossSensitivePremium) -> Self {
        self.retro = Some(terms);
        self
    }

    /// The gross premium.
    pub fn gross_premium(&self) -> f64 {
        self.premium
    }

    /// The ledger of the book over the annual gross losses `gross`, with
    /// the annual covers of `tower` if given, laid out as `face`.
    ///
    /// # Errors
    ///
    /// A premium or expense that is not finite, an expense with no term, a
    /// peel without a tower, a layer that needs each event (a surplus
    /// treaty, or reinstatements pro rata as to time), a layer with
    /// loss-sensitive terms, or a ledger the kernel refuses.
    pub fn ledger(&self, gross: &Source, tower: Option<&Tower>, face: Face) -> Result<Ledger> {
        self.validate()?;
        match tower {
            None => self.plain(gross, face),
            Some(_) if self.retro.is_some() => Err(Error::Data(
                "a retrospectively rated book buys no cover here: retro rating varies the gross \
                 premium with the gross loss, as aggregate's retro"
                    .into(),
            )),
            Some(t) => {
                check_tower(t)?;
                self.covered(gross, None, Some(t), face)
            }
        }
    }

    /// The ledger of the book with occurrence covers over the joint source
    /// `joint` of the annual gross loss and the annual loss ceded by the
    /// `occurrence` tower ([`occurrence_source`]), then the aggregate
    /// covers of `aggregate`, if given, on the net of the occurrence
    /// covers, as `aggregate`'s `xpnl` and `pnl` over its `occ_bivariate`.
    ///
    /// The ceded aggregate is not a function of the gross aggregate (the
    /// claim count decouples them), so the walk needs the joint: the
    /// occurrence tier is one group (the joint's second coordinate is its
    /// total cession) and each aggregate cover recovers its cession of
    /// `max(gross − ceded, 0)`. On the walk the occurrence covers are one
    /// group named after the first layer; a peel needs exactly one
    /// occurrence layer, as `aggregate`'s per-atom peel (its stitched
    /// route for several is not built). The consolidated face is over the
    /// net loss at each atom.
    ///
    /// An occurrence tier of one layer may carry annual terms and paid
    /// reinstatements, applied to the joint's second coordinate, its
    /// unlimited annual cession `R`: the recovery is `share ×
    /// after_terms(R / share)` and the premium the deposit plus the
    /// reinstatement premium on that layer loss, `aggregate`'s `A(R)` and
    /// `D + h(R)`, so the cover's premium varies with the year.
    ///
    /// # Errors
    ///
    /// As [`Book::ledger`], and a source that is not joint, annual terms or
    /// reinstatements on an occurrence tier of several layers, loss-sensitive
    /// terms on an occurrence layer, or a peel through more than one
    /// occurrence layer.
    pub fn occurrence_ledger(
        &self,
        joint: &Source,
        occurrence: &Tower,
        aggregate: Option<&Tower>,
        face: Face,
    ) -> Result<Ledger> {
        self.validate()?;
        if self.retro.is_some() {
            return Err(Error::Data(
                "a retrospectively rated book buys no cover here".into(),
            ));
        }
        if joint.second().is_none() {
            return Err(Error::Data(
                "occurrence covers need the joint source of the gross and ceded loss \
                 (occurrence_source)"
                    .into(),
            ));
        }
        check_events(occurrence)?;
        check_occurrence(occurrence)?;
        if let Some(t) = aggregate {
            check_tower(t)?;
        }
        self.covered(joint, Some(occurrence), aggregate, face)
    }

    fn validate(&self) -> Result<()> {
        if !self.premium.is_finite() {
            return Err(Error::Data(format!(
                "the premium must be finite, not {}",
                self.premium
            )));
        }
        for e in &self.expenses {
            if e.terms.is_empty() {
                return Err(Error::Data("an expense needs at least one term".into()));
            }
            if let Some((_, v)) = e.terms.iter().find(|(_, v)| !v.is_finite()) {
                return Err(Error::Data(format!(
                    "an expense term must be finite, not {v}"
                )));
            }
        }
        Ok(())
    }

    fn prem_key(&self) -> String {
        self.premium_label
            .clone()
            .unwrap_or_else(|| "Premium".into())
    }

    fn loss_key(&self) -> String {
        self.loss_label.clone().unwrap_or_else(|| "Loss".into())
    }

    fn margin_key(&self) -> String {
        self.loss_label.clone().unwrap_or_else(|| "Gross".into())
    }

    /// One obligation leg per expense, `aggregate`'s `_expense_legs`: a
    /// loss-basis term is `rate × x` at the source's value, or at `gross[k]`
    /// at each atom when the source is the net loss (the consolidated
    /// face). Zero legs are left out.
    fn expense_legs(&self, gross: Option<&[f64]>, taken: &[String]) -> Vec<Leg> {
        let mut used: Vec<String> = taken.to_vec();
        let lone_unlabelled = self.expenses.len() == 1 && self.expenses[0].label.is_none();
        let mut legs = Vec::new();
        for e in &self.expenses {
            let name = match &e.label {
                Some(l) => l.clone(),
                None if lone_unlabelled => "expense".into(),
                None => e.default_name(),
            };
            let (mut key, mut n) = (name.clone(), 2);
            while used.contains(&key) {
                key = format!("{name} ({n})");
                n += 1;
            }
            used.push(key.clone());
            let (scalar, rate) = e.split(self.premium);
            let leg = match gross {
                _ if rate == 0.0 && scalar == 0.0 => continue,
                _ if rate == 0.0 => Leg::constant(key, scalar),
                None => Leg::of(key, move |x| rate * x + scalar),
                Some(g) => Leg::values(key, g.iter().map(|x| rate * x + scalar).collect()),
            };
            legs.push(leg.kind(LegKind::Expense));
        }
        legs
    }

    /// The gross premium leg: fixed, or the retro premium at the gross loss.
    fn premium_leg(&self, label: String) -> Leg {
        match self.retro {
            Some(terms) => Leg::of(label, move |x| terms.premium(x)),
            None => Leg::constant(label, self.premium),
        }
        .kind(LegKind::Premium)
    }

    /// The gross book's group.
    fn gross_group(&self, label: String) -> Group {
        let (prem, loss) = (self.prem_key(), self.loss_key());
        let mut g = Group::sell(label)
            .consideration(self.premium_leg(prem.clone()))
            .obligation(Leg::of(loss.clone(), |x| x).kind(LegKind::Loss))
            .margin_label(self.margin_key());
        for leg in self.expense_legs(None, &[prem, loss]) {
            g = g.obligation(leg);
        }
        g
    }

    fn plain(&self, gross: &Source, face: Face) -> Result<Ledger> {
        match face {
            Face::Consolidated => {
                let (prem, loss) = (self.prem_key(), self.loss_key());
                let mut obligation = vec![Leg::of(loss.clone(), |x| x).kind(LegKind::Loss)];
                obligation.extend(self.expense_legs(None, &[prem.clone(), loss]));
                Ledger::single(
                    gross,
                    self.name.clone(),
                    Role::Sell,
                    vec![self.premium_leg(prem)],
                    obligation,
                    "margin",
                )
            }
            Face::Walk => {
                let step = self.step_label.clone().unwrap_or_else(|| "Gross".into());
                Ledger::new(
                    gross,
                    vec![self.gross_group(step)],
                    Layout::new(self.name.clone())
                        .result_name("margin")
                        .force_tower(),
                )
            }
            Face::PeelTopDown | Face::PeelBottomUp => Err(Error::Data(
                "a peel needs a tower: there are no layers to peel".into(),
            )),
        }
    }

    /// The book with occurrence covers (on a joint source) and aggregate
    /// covers, laid out as `face`.
    fn covered(
        &self,
        source: &Source,
        occ: Option<&Tower>,
        agg: Option<&Tower>,
        face: Face,
    ) -> Result<Ledger> {
        let xs = source.values();
        // The occurrence tier's cash flows from the joint's unlimited annual
        // cession: its recovery after any annual terms, and its premium with
        // any reinstatement premium.
        let occ_flows = occ.map(|t| occurrence_flows(t, source.second().unwrap_or_default()));
        let occ_ceded: Vec<f64> = match &occ_flows {
            Some(f) => f.recovery.clone(),
            None => vec![0.0; xs.len()],
        };
        // The aggregate covers' subject: the gross, or its net of the
        // occurrence covers (clipped at 0 off the support, as aggregate).
        let subject: Vec<f64> = match occ {
            Some(_) => xs
                .iter()
                .zip(&occ_ceded)
                .map(|(l, c)| (l - c).max(0.0))
                .collect(),
            None => xs.to_vec(),
        };
        let agg_ceded = agg.map(|t| ceded_at(t, &subject));
        if face == Face::Consolidated {
            let net: Vec<f64> = (0..xs.len())
                .map(|k| {
                    let a: f64 = agg_ceded.as_ref().map_or(0.0, |c| c[k].iter().sum());
                    xs[k] - occ_ceded[k] - a
                })
                .collect();
            let mut covers = Vec::new();
            if let Some(f) = occ_flows {
                covers.push(f);
            }
            if let (Some(t), Some(ceded)) = (agg, &agg_ceded) {
                let all: Vec<usize> = (0..t.layers.len()).collect();
                covers.push(Flows::of(t, &all, ceded));
            }
            return self.consolidated(source, net, &covers);
        }
        let mut taken: Vec<String> = Vec::new();
        let step = self.step_label.clone().unwrap_or_else(|| "Gross".into());
        let mut groups = vec![self.gross_group(step)];
        let mut spans = Vec::new();
        if let Some(t) = occ {
            let idx: Vec<usize> = (0..t.layers.len())
                .filter(|&i| t.layers[i].share != 0.0)
                .collect();
            if idx.is_empty() {
                return Err(Error::Data(
                    "every occurrence layer has a zero share; there is no cover to walk through"
                        .into(),
                ));
            }
            if face != Face::Walk && idx.len() > 1 {
                return Err(Error::Data(
                    "a peel through several occurrence layers needs each layer's own cession, \
                     which the joint source does not carry; walk the occurrence tier instead"
                        .into(),
                ));
            }
            let flows = occ_flows.ok_or_else(|| Error::Data("no occurrence flows".into()))?;
            groups.push(cover_group(
                t.layers[idx[0]].name.clone(),
                flows,
                &mut taken,
            ));
            spans.push(("All occurrence", 1, 2));
        }
        if let (Some(t), Some(ceded)) = (agg, &agg_ceded) {
            let covers = agg_covers(t, face);
            if covers.is_empty() {
                return Err(Error::Data(
                    "every layer of the tower has a zero share; there is no cover to walk through"
                        .into(),
                ));
            }
            let first = groups.len();
            for (label, idx) in &covers {
                groups.push(cover_group(
                    label.clone(),
                    Flows::of(t, idx, ceded),
                    &mut taken,
                ));
            }
            spans.push(("All aggregate", first, first + covers.len()));
        }
        let mut layout = Layout::new(self.name.clone()).result_name("margin");
        for (label, lo, hi) in spans {
            layout = layout.span(label, lo, hi);
        }
        Ledger::new(source, groups, layout)
    }

    /// One group, the net position over the net loss `net` at each atom of
    /// `source`, after the covers' cash flows `covers`.
    fn consolidated(&self, source: &Source, net: Vec<f64>, covers: &[Flows]) -> Result<Ledger> {
        let n = net.len();
        let net_source = Source::new(net, source.probs().to_vec())?;
        let mut ceded_premium = vec![0.0; n];
        let mut commission = vec![0.0; n];
        let (mut premium_varies, mut commission_varies) = (false, false);
        for f in covers {
            premium_varies |= f.premium_varies;
            for k in 0..n {
                ceded_premium[k] += f.premium[k];
                commission[k] += f.flat;
            }
            for v in [&f.sliding, &f.profit].into_iter().flatten() {
                commission_varies = true;
                for k in 0..n {
                    commission[k] += v[k];
                }
            }
        }
        let prem = match &self.premium_label {
            Some(l) => format!("{l} (net)"),
            None => "net premium".into(),
        };
        let loss = format!("{} (net)", self.loss_key());
        let mut obligation = vec![Leg::of(loss.clone(), |x| x).kind(LegKind::Loss)];
        // Received: a negative obligation books positive.
        if commission_varies {
            obligation.push(
                Leg::values("commission", commission.iter().map(|c| -c).collect())
                    .kind(LegKind::Commission),
            );
        } else if commission[0] != 0.0 {
            obligation.push(Leg::constant("commission", -commission[0]).kind(LegKind::Commission));
        }
        // A loss-basis expense on each atom's gross loss (the consolidated
        // margin is then the walk's net margin atom by atom), or at its mean.
        let at_mean: Vec<f64>;
        let gross = if self.expense_at_mean {
            let m: f64 = source
                .values()
                .iter()
                .zip(source.probs())
                .map(|(x, p)| x * p)
                .sum();
            at_mean = vec![m; source.values().len()];
            &at_mean[..]
        } else {
            source.values()
        };
        obligation
            .extend(self.expense_legs(Some(gross), &[prem.clone(), loss, "commission".into()]));
        let net_premium = if premium_varies {
            Leg::values(
                prem,
                ceded_premium.iter().map(|c| self.premium - c).collect(),
            )
        } else {
            Leg::constant(prem, self.premium - ceded_premium[0])
        };
        Ledger::single(
            &net_source,
            self.name.clone(),
            Role::Sell,
            vec![net_premium.kind(LegKind::Premium)],
            obligation,
            "margin",
        )
    }
}

/// `(label, layers)` of each aggregate cover group: one per stage on a
/// walk (named after its first layer), one per layer on a peel, the highest
/// or the lowest first within each stage. Zero shares are left out.
fn agg_covers(tower: &Tower, face: Face) -> Vec<(String, Vec<usize>)> {
    let mut covers: Vec<(String, Vec<usize>)> = Vec::new();
    let mut start = 0;
    while start < tower.layers.len() {
        let stage = tower.stages[start];
        let end = start
            + tower.stages[start..]
                .iter()
                .take_while(|&&s| s == stage)
                .count();
        let idx: Vec<usize> = (start..end)
            .filter(|&i| tower.layers[i].share != 0.0)
            .collect();
        match face {
            Face::Walk if !idx.is_empty() => {
                covers.push((tower.layers[idx[0]].name.clone(), idx));
            }
            Face::PeelTopDown => {
                covers.extend(
                    idx.iter()
                        .rev()
                        .map(|&i| (tower.layers[i].name.clone(), vec![i])),
                );
            }
            Face::PeelBottomUp => {
                covers.extend(idx.iter().map(|&i| (tower.layers[i].name.clone(), vec![i])));
            }
            _ => {}
        }
        start = end;
    }
    covers
}

/// A cover's cash flows at each atom: its premium (swing rated, or fixed),
/// its recovery, its flat commission, and its sliding-scale and profit
/// commissions when it has them.
struct Flows {
    premium: Vec<f64>,
    premium_varies: bool,
    recovery: Vec<f64>,
    flat: f64,
    sliding: Option<Vec<f64>>,
    profit: Option<Vec<f64>>,
}

impl Flows {
    /// Fixed terms only, with the cover's total recovery at each atom.
    fn fixed(layers: &[&Layer], recovery: Vec<f64>) -> Self {
        let premium: f64 = layers.iter().map(|l| l.premium).sum();
        Self {
            premium: vec![premium; recovery.len()],
            premium_varies: false,
            recovery,
            flat: layers.iter().map(|l| flat_commission(l)).sum(),
            sliding: None,
            profit: None,
        }
    }

    /// The layers `idx` of `tower`, `ceded[k][i]` layer `i`'s ceded loss at
    /// atom `k`: each term at the layer's own annual ceded loss.
    fn of(tower: &Tower, idx: &[usize], ceded: &[Vec<f64>]) -> Self {
        let layers: Vec<&Layer> = idx.iter().map(|&i| &tower.layers[i]).collect();
        let at = |f: &dyn Fn(&Layer, f64) -> f64| -> Vec<f64> {
            ceded
                .iter()
                .map(|c| idx.iter().zip(&layers).map(|(&i, l)| f(l, c[i])).sum())
                .collect()
        };
        let sliding = layers
            .iter()
            .any(|l| matches!(l.commission, Some(Commission::SlidingScale(_))));
        let profit = layers.iter().any(|l| l.profit_commission.is_some());
        Self {
            premium: at(&|l, c| l.premium_for(c)),
            premium_varies: layers.iter().any(|l| l.swing.is_some()),
            recovery: at(&|_, c| c),
            flat: layers.iter().map(|l| flat_commission(l)).sum(),
            sliding: sliding.then(|| {
                at(&|l, c| match l.commission {
                    Some(Commission::SlidingScale(_)) => l.ceding_commission_for(c),
                    _ => 0.0,
                })
            }),
            profit: profit.then(|| at(&|l, c| l.profit_commission_for(c))),
        }
    }
}

/// `name`, or `<label> name` when another group took it.
fn free_label(name: &str, label: &str, taken: &mut Vec<String>) -> String {
    let key = if taken.iter().any(|t| t == name) {
        format!("{label} {name}")
    } else {
        name.to_string()
    };
    taken.push(key.clone());
    key
}

/// A cover's buy group: its premium, its recovery and its commissions at
/// each atom, labelled as `aggregate`'s (a sliding-scale or profit
/// commission is `sliding commission` or `profit commission`, prefixed by
/// the cover's label when another cover took that name).
fn cover_group(label: String, f: Flows, taken: &mut Vec<String>) -> Group {
    let premium = if f.premium_varies {
        Leg::values(format!("{label} premium"), f.premium)
    } else {
        Leg::constant(format!("{label} premium"), f.premium[0])
    };
    let mut g = Group::buy(label.clone())
        .consideration(premium.kind(LegKind::Premium))
        .obligation(Leg::values(format!("{label} recovery"), f.recovery).kind(LegKind::Recovery));
    if let Some(v) = f.sliding {
        let key = free_label("sliding commission", &label, taken);
        g = g.obligation(Leg::values(key, v).kind(LegKind::Commission));
    }
    if let Some(v) = f.profit {
        let key = free_label("profit commission", &label, taken);
        g = g.obligation(Leg::values(key, v).kind(LegKind::Commission));
    }
    if f.flat != 0.0 {
        g = g.obligation(
            Leg::constant(format!("{label} commission"), f.flat).kind(LegKind::Commission),
        );
    }
    g
}

/// Whether an occurrence tier is one layer with annual terms or paid
/// reinstatements, applied to the joint's unlimited annual cession.
fn annual_occurrence(tower: &Tower) -> Option<&Layer> {
    match tower.layers.as_slice() {
        [l] if l.has_annual_terms() || !l.reinstatement_rates.is_empty() => Some(l),
        _ => None,
    }
}

/// The occurrence tier's cash flows at each atom, `ceded` the joint's
/// unlimited annual cession (at the placed share): with one layer carrying
/// annual terms, its recovery `share × after_terms(R / share)` and its
/// premium plus the reinstatement premium on that layer loss
/// (`aggregate`'s `A(R)` and `D + h(R)`); otherwise the cession itself and
/// the fixed premium.
fn occurrence_flows(tower: &Tower, ceded: &[f64]) -> Flows {
    let Some(l) = annual_occurrence(tower) else {
        let live: Vec<&Layer> = tower.layers.iter().filter(|l| l.share != 0.0).collect();
        return Flows::fixed(&live, ceded.to_vec());
    };
    let loss: Vec<f64> = ceded
        .iter()
        .map(|&r| {
            if l.share > 0.0 {
                l.after_terms(r / l.share)
            } else {
                0.0
            }
        })
        .collect();
    let premium: Vec<f64> = loss
        .iter()
        .map(|&x| l.premium + l.reinstatement_premium_on(x))
        .collect();
    Flows {
        premium,
        premium_varies: !l.reinstatement_rates.is_empty(),
        recovery: loss.iter().map(|x| l.share * x).collect(),
        flat: flat_commission(l),
        sliding: None,
        profit: None,
    }
}

/// The tower each claim is ceded to: a single layer's annual terms and
/// reinstatements stripped, as they apply to the year's total.
fn per_claim(tower: &Tower) -> Result<Tower> {
    match annual_occurrence(tower) {
        Some(l) => {
            let mut l = l.clone();
            l.aggregate_deductible = 0.0;
            l.aggregate_limit = f64::INFINITY;
            l.corridor = None;
            l.reinstatement_rates = Vec::new();
            Tower::new(vec![l])
        }
        None => Ok(tower.clone()),
    }
}

/// Refuses an occurrence tier the joint cannot carry: annual terms or
/// reinstatements on a tier of several layers (the joint carries the
/// tier's total cession), or loss-sensitive terms.
fn check_occurrence(tower: &Tower) -> Result<()> {
    if annual_occurrence(tower).is_some() {
        let l = &tower.layers[0];
        if l.pro_rata_time {
            return Err(Error::Data(format!(
                "occurrence layer {:?} has reinstatements pro rata as to time; the joint carries \
                 no event times",
                l.name
            )));
        }
    }
    for l in &tower.layers {
        let annual = l.has_annual_terms() || !l.reinstatement_rates.is_empty();
        if annual && tower.layers.len() > 1 {
            return Err(Error::Data(format!(
                "occurrence layer {:?} has annual terms or reinstatements on a tier of several \
                 layers; the joint source carries the tier's total cession, so only a tier of \
                 one layer may have them",
                l.name
            )));
        }
        let variable = l.swing.is_some()
            || matches!(l.commission, Some(Commission::SlidingScale(_)))
            || l.profit_commission.is_some();
        if variable {
            return Err(Error::Data(format!(
                "occurrence layer {:?} has loss-sensitive terms; as in aggregate, they go on an \
                 aggregate cover",
                l.name
            )));
        }
    }
    Ok(())
}

/// The joint source of the annual gross loss and the annual loss ceded by
/// the per-occurrence covers of `occurrence`, `aggregate`'s
/// `occ_bivariate(views=('gross', 'ceded'))`: each claim `X` on the
/// severity's grid goes to `(X, c(X))`, `c` the tower's cession of one
/// event (inuring stages applied per event), split linearly between the
/// two grid points around it when it falls between them; the pairs are
/// compounded by [`fft_bivariate`](crate::bivariate::fft_bivariate) on
/// `points` points per axis, cells below `1e-15` of the total mass are
/// dropped as round-off (`aggregate`'s floor), and the rest are the atoms.
/// A tier of one layer with annual terms or reinstatements cedes each
/// claim without them: the second coordinate is then the unlimited annual
/// cession, which [`Book::occurrence_ledger`] caps.
///
/// ```
/// use prospicio_aggregate::pnl::occurrence_source;
/// use prospicio_aggregate::{Layer, Tower};
/// use prospicio_prob::{Grid, Poisson};
///
/// // Claims of 1, 2 or 4; a cover of 2 xs 2 on each.
/// let severity = Grid::new(1.0, vec![0.0, 0.5, 0.3, 0.0, 0.2]).unwrap();
/// let cover = Tower::new(vec![Layer::xol("occ 2 xs 2", 2.0, 2.0).unwrap()]).unwrap();
/// let (joint, _) =
///     occurrence_source(&Poisson::new(2.0).unwrap(), &severity, &cover, [64, 32]).unwrap();
/// let ceded = joint.second().unwrap();
/// let mean: f64 = ceded.iter().zip(joint.probs()).map(|(c, p)| c * p).sum();
/// assert!((mean - 0.8).abs() < 1e-12); // 2 claims × 0.2 × 2
/// ```
///
/// # Errors
///
/// Annual terms or reinstatements on a tier of several layers, a layer that
/// needs each event's sum insured or time, loss-sensitive terms, or what
/// [`fft_bivariate`](crate::bivariate::fft_bivariate) refuses.
pub fn occurrence_source<N: Counting + ?Sized>(
    frequency: &N,
    severity: &Grid,
    occurrence: &Tower,
    points: [usize; 2],
) -> Result<(Source, BivariateReport)> {
    check_events(occurrence)?;
    check_occurrence(occurrence)?;
    let claims = per_claim(occurrence)?;
    let h = severity.step();
    let cession: Vec<f64> = (0..severity.len())
        .map(|k| {
            claims
                .year(&[k as f64 * h], None, None)
                .iter()
                .map(|(c, _)| c)
                .sum()
        })
        .collect();
    let cols = cession
        .iter()
        .map(|c| (c / h).floor() as usize + 2)
        .max()
        .unwrap_or(1);
    let mut probs = vec![0.0; severity.len() * cols];
    for (k, (&p, &c)) in severity.probs().iter().zip(&cession).enumerate() {
        let u = c / h;
        let j = u.floor();
        let w = u - j;
        let j = j as usize;
        probs[k * cols + j] += p * (1.0 - w);
        if w > 0.0 {
            probs[k * cols + j + 1] += p * w;
        }
    }
    let pairs = JointGrid::new([h, h], [severity.len(), cols], probs)?;
    let (joint, report) = fft_bivariate(frequency, &pairs, points)?;
    let total: f64 = joint.probs().iter().map(|p| p.abs()).sum();
    let floor = 1e-15 * total;
    let [na, nb] = joint.shape();
    let (mut xs, mut ys, mut ps) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..na {
        for j in 0..nb {
            let p = joint.prob(i, j);
            if p >= floor && p > 0.0 {
                xs.push(i as f64 * h);
                ys.push(j as f64 * h);
                ps.push(p);
            }
        }
    }
    Ok((Source::joint(xs, ys, ps)?, report))
}

/// Each layer's ceded loss at each annual total.
fn ceded_at(tower: &Tower, totals: &[f64]) -> Vec<Vec<f64>> {
    totals
        .iter()
        .map(|&x| {
            tower
                .year(&[x], None, None)
                .into_iter()
                .map(|(c, _)| c)
                .collect()
        })
        .collect()
}

/// A layer's flat ceding commission on its premium.
fn flat_commission(layer: &Layer) -> f64 {
    match &layer.commission {
        Some(Commission::Flat(rate)) => rate * layer.premium,
        _ => 0.0,
    }
}

/// Refuses a layer that needs each event's sum insured or time.
fn check_events(tower: &Tower) -> Result<()> {
    for l in &tower.layers {
        if l.needs_sums_insured() || l.needs_times() {
            return Err(Error::Data(format!(
                "layer {:?} works event by event (a surplus treaty, or reinstatements pro rata \
                 as to time); an annual loss has no events",
                l.name
            )));
        }
    }
    Ok(())
}

/// Refuses an annual cover the builders cannot book: one that needs each
/// event, or paid reinstatements, which belong to an occurrence cover
/// ([`Book::occurrence_ledger`]).
fn check_tower(tower: &Tower) -> Result<()> {
    check_events(tower)?;
    if let Some(l) = tower
        .layers
        .iter()
        .find(|l| !l.reinstatement_rates.is_empty())
    {
        return Err(Error::Data(format!(
            "layer {:?} has paid reinstatements, which apply to an occurrence cover: book it \
             with occurrence_ledger over the joint source",
            l.name
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gross() -> Source {
        Source::new(
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
            vec![0.25, 0.3, 0.09, 0.2, 0.12, 0.04],
        )
        .unwrap()
    }

    #[test]
    fn expense_names_follow_aggregate() {
        let book = Book::new("B", 10.0)
            .expense(Expense::premium(0.1))
            .expense(Expense::loss(0.05))
            .expense(Expense::fixed(1.0))
            .expense(Expense::premium(0.02));
        let l = book.ledger(&gross(), None, Face::Consolidated).unwrap();
        for label in [
            "premium expense",
            "loss expense",
            "fixed expense",
            "premium expense (2)",
        ] {
            assert!(l.row(label).is_some(), "{label}");
        }
        let mixed = Book::new("B", 10.0)
            .expense(Expense::premium(0.1).and(ExpenseBasis::Loss, 0.05))
            .ledger(&gross(), None, Face::Consolidated)
            .unwrap();
        assert!((mixed.row("expense").unwrap().mean() + 1.19).abs() < 1e-12);
    }

    #[test]
    fn consolidated_loss_expense_at_each_atom() {
        let book = Book::new("B", 10.0).expense(Expense::loss(0.1));
        let cover = Tower::new(vec![
            Layer::stop_loss("SL", 3.0, 5.0)
                .unwrap()
                .deposit_premium(1.0)
                .unwrap(),
        ])
        .unwrap();
        let c = book
            .ledger(&gross(), Some(&cover), Face::Consolidated)
            .unwrap();
        let w = book.ledger(&gross(), Some(&cover), Face::Walk).unwrap();
        // The consolidated margin is the walk's net margin at every level.
        for q in [0.01, 0.25, 0.5, 0.9, 0.99] {
            let (a, b) = (
                c.result().quantile(q).unwrap(),
                w.result().quantile(q).unwrap(),
            );
            assert!((a - b).abs() < 1e-12, "{q}: {a} {b}");
        }
        assert!((c.result().sd() - w.result().sd()).abs() < 1e-12);
        // At the expected loss: the same mean, a constant leg.
        let m = book
            .clone()
            .expense_at_expected_loss()
            .ledger(&gross(), Some(&cover), Face::Consolidated)
            .unwrap();
        assert!((m.result().mean() - c.result().mean()).abs() < 1e-12);
        assert_eq!(m.row("expense").unwrap().sd(), 0.0);
        assert!(c.row("expense").unwrap().sd() > 0.0);
    }

    #[test]
    fn built_where_aggregate_refuses() {
        // A walk over a retro book: the one gross group (aggregate's xpnl
        // refuses retro; prospicio's walk has nothing to walk through).
        let retro = Book::new("R", 2.0)
            .retro(LossSensitivePremium::new(2.0, 1.1, Some(3.0), Some(9.0)).unwrap());
        let w = retro.ledger(&gross(), None, Face::Walk).unwrap();
        let c = retro.ledger(&gross(), None, Face::Consolidated).unwrap();
        assert!((w.result().mean() - c.result().mean()).abs() < 1e-12);
        // A peel of a single reinstated occurrence layer is its walk.
        let joint = Source::joint(
            vec![2.0, 4.0, 6.0],
            vec![1.0, 2.0, 4.0],
            vec![0.5, 0.3, 0.2],
        )
        .unwrap();
        let occ = Tower::new(vec![
            Layer::xol("XL", 2.0, 2.0)
                .unwrap()
                .paid_reinstatements(1.0, vec![1.0])
                .unwrap(),
        ])
        .unwrap();
        let b = Book::new("B", 10.0);
        let walk = b.occurrence_ledger(&joint, &occ, None, Face::Walk).unwrap();
        let peel = b
            .occurrence_ledger(&joint, &occ, None, Face::PeelTopDown)
            .unwrap();
        assert_eq!(walk.rows().len(), peel.rows().len());
        for (r, s) in walk.rows().iter().zip(peel.rows()) {
            assert_eq!(r.label, s.label);
            assert!((r.mean() - s.mean()).abs() < 1e-12);
        }
    }

    #[test]
    fn occurrence_refusals() {
        let book = Book::new("B", 10.0);
        let occ = Tower::new(vec![Layer::xol("a", 1.0, 1.0).unwrap()]).unwrap();
        // Not a joint source.
        assert!(
            book.occurrence_ledger(&gross(), &occ, None, Face::Walk)
                .is_err()
        );
        let joint = Source::joint(vec![2.0, 4.0], vec![1.0, 2.0], vec![0.5, 0.5]).unwrap();
        assert!(
            book.occurrence_ledger(&joint, &occ, None, Face::Walk)
                .is_ok()
        );
        // Annual terms on a single occurrence layer apply to the joint's
        // annual cession; on a tier of several they cannot.
        let capped = Layer::xol("a", 1.0, 1.0)
            .unwrap()
            .aggregate_limit(1.5)
            .unwrap();
        let one = Tower::new(vec![capped.clone()]).unwrap();
        let l = book
            .occurrence_ledger(&joint, &one, None, Face::Walk)
            .unwrap();
        // Ceded 1 and 2, capped at 1.5: a recovery of 1.25.
        assert!((l.row("a recovery").unwrap().mean() - 1.25).abs() < 1e-12);
        let annual = Tower::new(vec![capped, Layer::xol("b", 2.0, 2.0).unwrap()]).unwrap();
        assert!(
            book.occurrence_ledger(&joint, &annual, None, Face::Walk)
                .is_err()
        );
        // A peel through two occurrence layers.
        let two = Tower::new(vec![
            Layer::xol("a", 1.0, 1.0).unwrap(),
            Layer::xol("b", 2.0, 2.0).unwrap(),
        ])
        .unwrap();
        assert!(
            book.occurrence_ledger(&joint, &two, None, Face::Walk)
                .is_ok()
        );
        assert!(
            book.occurrence_ledger(&joint, &two, None, Face::PeelTopDown)
                .is_err()
        );
    }

    #[test]
    fn refusals() {
        let book = Book::new("B", 10.0);
        assert!(book.ledger(&gross(), None, Face::PeelTopDown).is_err());
        let pc = Layer::stop_loss("SL", 3.0, 5.0)
            .unwrap()
            .deposit_premium(1.0)
            .unwrap()
            .profit_commission(0.2, 0.1)
            .unwrap();
        let t = Tower::new(vec![pc]).unwrap();
        assert!(book.ledger(&gross(), Some(&t), Face::Walk).is_ok());
        // Reinstatement premiums, and retro with a cover.
        let reinstated = Layer::xol("XL", 2.0, 2.0)
            .unwrap()
            .paid_reinstatements(1.0, vec![1.0])
            .unwrap();
        let r = Tower::new(vec![reinstated]).unwrap();
        assert!(book.ledger(&gross(), Some(&r), Face::Walk).is_err());
        let retro = Book::new("R", 2.0)
            .retro(LossSensitivePremium::new(2.0, 1.1, Some(3.0), Some(9.0)).unwrap());
        assert!(retro.ledger(&gross(), Some(&t), Face::Walk).is_err());
        let plain = retro.ledger(&gross(), None, Face::Consolidated).unwrap();
        assert!((plain.row("Premium").unwrap().mean() - 6.108).abs() < 1e-12);
        assert!(derive_premium(1.0, &[Expense::loss(0.1)]).is_err());
        assert!(derive_premium(1.0, &[Expense::premium(1.0)]).is_err());
    }
}
