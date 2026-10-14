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
//!   over the net loss at each atom. The gross loss is not on that source,
//!   so a loss-basis expense books as its rate times the expected gross
//!   loss.
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
//! deductible, limit and corridor, at its share. The builders book a fixed
//! premium and a flat ceding commission; a layer with loss-sensitive terms
//! (swing rating, a sliding scale, a profit commission or reinstatement
//! premiums) is refused, and its legs can be written by hand with
//! [`Leg::of`].

use prospicio_core::{Error, Result};

use super::{Group, Layout, Ledger, Leg, LegKind, Role, Source};
use crate::reinsurance::{Commission, Layer, Tower};

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
        match tower {
            None => self.plain(gross, face),
            Some(t) => {
                check_tower(t)?;
                match face {
                    Face::Consolidated => self.consolidated(gross, t),
                    _ => self.walk(gross, t, face),
                }
            }
        }
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

    /// One obligation leg per expense, `aggregate`'s `_expense_legs`: on
    /// the source, a loss-basis term is `rate × x`; off it, `rate × E[gross
    /// loss]`. Zero legs are left out.
    fn expense_legs(&self, gross_mean: Option<f64>, taken: &[String]) -> Vec<Leg> {
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
            let leg = match gross_mean {
                _ if rate == 0.0 && scalar == 0.0 => continue,
                _ if rate == 0.0 => Leg::constant(key, scalar),
                None => Leg::of(key, move |x| rate * x + scalar),
                Some(m) => Leg::constant(key, rate * m + scalar),
            };
            legs.push(leg.kind(LegKind::Expense));
        }
        legs
    }

    /// The gross book's group.
    fn gross_group(&self, label: String) -> Group {
        let (prem, loss) = (self.prem_key(), self.loss_key());
        let mut g = Group::sell(label)
            .consideration(Leg::constant(prem.clone(), self.premium).kind(LegKind::Premium))
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
                    vec![Leg::constant(prem, self.premium).kind(LegKind::Premium)],
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

    fn consolidated(&self, gross: &Source, tower: &Tower) -> Result<Ledger> {
        let ceded = ceded_at(tower, gross.values());
        let net: Vec<f64> = gross
            .values()
            .iter()
            .zip(&ceded)
            .map(|(x, c)| x - c.iter().sum::<f64>())
            .collect();
        let source = Source::new(net, gross.probs().to_vec())?;
        let ceded_premium: f64 = tower.layers.iter().map(|l| l.premium).sum();
        let commission: f64 = tower.layers.iter().map(flat_commission).sum();
        let prem = match &self.premium_label {
            Some(l) => format!("{l} (net)"),
            None => "net premium".into(),
        };
        let loss = format!("{} (net)", self.loss_key());
        let mut obligation = vec![Leg::of(loss.clone(), |x| x).kind(LegKind::Loss)];
        if commission != 0.0 {
            // Received: a negative obligation books positive.
            obligation.push(Leg::constant("commission", -commission).kind(LegKind::Commission));
        }
        let gross_mean: f64 = gross
            .values()
            .iter()
            .zip(gross.probs())
            .map(|(x, p)| x * p)
            .sum();
        obligation.extend(
            self.expense_legs(Some(gross_mean), &[prem.clone(), loss, "commission".into()]),
        );
        Ledger::single(
            &source,
            self.name.clone(),
            Role::Sell,
            vec![Leg::constant(prem, self.premium - ceded_premium).kind(LegKind::Premium)],
            obligation,
            "margin",
        )
    }

    fn walk(&self, gross: &Source, tower: &Tower, face: Face) -> Result<Ledger> {
        let ceded = ceded_at(tower, gross.values());
        let step = self.step_label.clone().unwrap_or_else(|| "Gross".into());
        let mut groups = vec![self.gross_group(step)];
        // (label, layers) of each cover group.
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
        if covers.is_empty() {
            return Err(Error::Data(
                "every layer of the tower has a zero share; there is no cover to walk through"
                    .into(),
            ));
        }
        for (label, idx) in &covers {
            let recovery: Vec<f64> = ceded
                .iter()
                .map(|c| idx.iter().map(|&i| c[i]).sum())
                .collect();
            let premium: f64 = idx.iter().map(|&i| tower.layers[i].premium).sum();
            let commission: f64 = idx.iter().map(|&i| flat_commission(&tower.layers[i])).sum();
            let mut g = Group::buy(label.clone())
                .consideration(
                    Leg::constant(format!("{label} premium"), premium).kind(LegKind::Premium),
                )
                .obligation(
                    Leg::values(format!("{label} recovery"), recovery).kind(LegKind::Recovery),
                );
            if commission != 0.0 {
                g = g.obligation(
                    Leg::constant(format!("{label} commission"), commission)
                        .kind(LegKind::Commission),
                );
            }
            groups.push(g);
        }
        let n = covers.len();
        Ledger::new(
            gross,
            groups,
            Layout::new(self.name.clone())
                .result_name("margin")
                .span("All aggregate", 1, 1 + n),
        )
    }
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

/// Refuses a layer the builders cannot book.
fn check_tower(tower: &Tower) -> Result<()> {
    for l in &tower.layers {
        if l.needs_sums_insured() || l.needs_times() {
            return Err(Error::Data(format!(
                "layer {:?} works event by event (a surplus treaty, or reinstatements pro rata \
                 as to time); an annual loss has no events",
                l.name
            )));
        }
        let variable = l.swing.is_some()
            || matches!(l.commission, Some(Commission::SlidingScale(_)))
            || l.profit_commission.is_some()
            || !l.reinstatement_rates.is_empty();
        if variable {
            return Err(Error::Data(format!(
                "layer {:?} has loss-sensitive terms (swing rating, a sliding scale, a profit \
                 commission or reinstatement premiums); the builders book a fixed premium and \
                 a flat commission, so write its legs with Leg::of",
                l.name
            )));
        }
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
        assert!(book.ledger(&gross(), Some(&t), Face::Walk).is_err());
        assert!(derive_premium(1.0, &[Expense::loss(0.1)]).is_err());
        assert!(derive_premium(1.0, &[Expense::premium(1.0)]).is_err());
    }
}
