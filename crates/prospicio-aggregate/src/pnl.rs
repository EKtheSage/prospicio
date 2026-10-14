//! The P&L ledger: a treaty's profit and loss as distributions, as the
//! Python package `aggregate`'s `PnL` (`docs/design/pnl.md`).
//!
//! A ledger is a **source** (a distribution whose atoms carry the
//! probabilities: an annual loss on a grid, or any finite set of outcomes)
//! and an ordered list of **groups**, each a book that sells or buys cover:
//! its consideration and obligation **legs**, each a cash flow written as a
//! positive magnitude and evaluated at every atom of the source. The role
//! books the signs once: a [`Role::Sell`] group books `+consideration −
//! obligation`, a [`Role::Buy`] group the contra (ceded premium paid,
//! recoveries and commission received).
//!
//! Every other row is a sum of signed legs atom by atom: each side's total
//! (when it has more than one leg), each group's result, the running net
//! through each group after the first, tier subtotals and net-of blocks,
//! and, with several groups, the grand totals, the grand result and the
//! total impact of the covers (the grand result less the first group's).
//! Means add down the sheet and the dependence between rows comes for free:
//! no row is arithmetic on marginal distributions. Each row carries its
//! exact distribution, a [`Discrete`], and its moments taken off the atoms.
//!
//! The **scenario columns** (`κ01` to `κ99`) condition on the grand result:
//! at each level `q`, every row's conditional mean given that the grand
//! result equals its lower `q`-quantile, so each column is one state of the
//! world and foots exactly. Where the result is not monotone in the source,
//! the condition is a level set of several atoms. Values within `1e-12`
//! (relative, or absolute near 0) count as equal, so a sum taken in a
//! different order does not split an atom (`aggregate` compares exactly).
//!
//! The source's probabilities are rescaled to sum to 1, as `aggregate`
//! does, so a ledger over an FFT density that lost a little tail mass shows
//! rescaled means. Rows are in the payoff convention: profit is positive and
//! the adverse tail the low one, so `κ01` is the bad state.

use prospicio_core::{Error, Result};
use prospicio_prob::{Discrete, Distribution, Grid};

/// The levels of the scenario and marginal columns, `aggregate`'s
/// `PERCENTILE_LADDER`.
pub const LADDER: [f64; 9] = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];

/// Values at or below this in magnitude display as 0, `aggregate`'s
/// `validation.noise`.
pub const NOISE: f64 = 1e-12;

/// `v`, or 0 when it is within [`NOISE`] of 0, as `aggregate`'s
/// `_snap_noise`. NaN stays NaN.
pub fn snap(v: f64) -> f64 {
    if v.abs() <= NOISE { 0.0 } else { v }
}

/// Whether two values are the same atom.
fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
}

/// Whether a group sells cover or buys it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Books `+consideration − obligation`: the insurer of the subject
    /// business.
    Sell,
    /// Books `−consideration + obligation`: a cession, paying premium and
    /// receiving recoveries and commission.
    Buy,
}

/// What a leg is, for the ratio exhibit (PR 3); it does not change the
/// ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegKind {
    /// Premium, written or ceded.
    Premium,
    /// Loss.
    Loss,
    /// Expense.
    Expense,
    /// A recovery from a cover.
    Recovery,
    /// A commission.
    Commission,
}

impl LegKind {
    /// The name `aggregate` uses: `premium`, `loss`, `expense`, `recovery`
    /// or `commission`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Premium => "premium",
            Self::Loss => "loss",
            Self::Expense => "expense",
            Self::Recovery => "recovery",
            Self::Commission => "commission",
        }
    }
}

type Of = Box<dyn Fn(f64) -> f64 + Send + Sync>;

/// A side's legs once evaluated: label, kind and signed values.
type SignedLegs = Vec<(String, Option<LegKind>, Vec<f64>)>;

enum LegValue {
    Constant(f64),
    Of(Of),
    Values(Vec<f64>),
}

/// One cash flow of a group: a constant, a function of the source's value,
/// or its values at each atom; a positive magnitude, signed by the group's
/// role.
pub struct Leg {
    label: String,
    kind: Option<LegKind>,
    value: LegValue,
}

impl std::fmt::Debug for Leg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match &self.value {
            LegValue::Constant(c) => format!("constant {c}"),
            LegValue::Of(_) => "a function of the source".to_string(),
            LegValue::Values(v) => format!("{} values", v.len()),
        };
        f.debug_struct("Leg")
            .field("label", &self.label)
            .field("kind", &self.kind)
            .field("value", &value)
            .finish()
    }
}

impl Leg {
    /// A leg of the same amount at every atom: a fixed premium or expense.
    pub fn constant(label: impl Into<String>, amount: f64) -> Self {
        Self {
            label: label.into(),
            kind: None,
            value: LegValue::Constant(amount),
        }
    }

    /// A leg that is a function of the source's value: the loss, a
    /// recovery, a premium that depends on the loss.
    pub fn of(label: impl Into<String>, f: impl Fn(f64) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            label: label.into(),
            kind: None,
            value: LegValue::Of(Box::new(f)),
        }
    }

    /// A leg given by its value at each atom of the source, in the source's
    /// order (the bindings evaluate a callable once on all the atoms).
    pub fn values(label: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            label: label.into(),
            kind: None,
            value: LegValue::Values(values),
        }
    }

    /// The leg with its kind set.
    pub fn kind(mut self, kind: LegKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// The label.
    pub fn label(&self) -> &str {
        &self.label
    }

    fn evaluate(&self, xs: &[f64]) -> Result<Vec<f64>> {
        let v = match &self.value {
            LegValue::Constant(c) => vec![*c; xs.len()],
            LegValue::Of(f) => xs.iter().map(|&x| f(x)).collect(),
            LegValue::Values(v) => {
                if v.len() != xs.len() {
                    return Err(Error::Data(format!(
                        "leg {:?} has {} values for {} atoms",
                        self.label,
                        v.len(),
                        xs.len()
                    )));
                }
                v.clone()
            }
        };
        if let Some(bad) = v.iter().find(|x| !x.is_finite()) {
            return Err(Error::Data(format!(
                "leg {:?} is {bad} at an atom; every leg must be finite",
                self.label
            )));
        }
        Ok(v)
    }
}

/// A book in the ledger: its label, its role, and its consideration and
/// obligation legs.
#[derive(Debug)]
pub struct Group {
    label: String,
    role: Role,
    consideration: Vec<Leg>,
    obligation: Vec<Leg>,
    margin_label: Option<String>,
}

impl Group {
    /// A group with no legs yet.
    pub fn new(label: impl Into<String>, role: Role) -> Self {
        Self {
            label: label.into(),
            role,
            consideration: Vec::new(),
            obligation: Vec::new(),
            margin_label: None,
        }
    }

    /// A seller: the insurer of the subject business.
    pub fn sell(label: impl Into<String>) -> Self {
        Self::new(label, Role::Sell)
    }

    /// A buyer: a cession.
    pub fn buy(label: impl Into<String>) -> Self {
        Self::new(label, Role::Buy)
    }

    /// Adds a consideration leg (premium).
    pub fn consideration(mut self, leg: Leg) -> Self {
        self.consideration.push(leg);
        self
    }

    /// Adds an obligation leg (loss, expense, or for a buyer recovery and
    /// commission).
    pub fn obligation(mut self, leg: Leg) -> Self {
        self.obligation.push(leg);
        self
    }

    /// The label of a seller's own margin on a ledger that also buys
    /// cover, `Gross` by default.
    pub fn margin_label(mut self, label: impl Into<String>) -> Self {
        self.margin_label = Some(label.into());
        self
    }

    /// The label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The role.
    pub fn role(&self) -> Role {
        self.role
    }
}

/// A span of groups `lo..hi` with a subtotal block of its own (when it has
/// two or more groups) and, when a later group follows, a net-of block: the
/// position through the span. `aggregate`'s `tier_spans`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// The span's label, as `All occurrence`; its net-of block reads `Net
    /// of occurrence` (a leading `All ` replaced), or `Net of <label>`.
    pub label: String,
    /// The first group.
    pub lo: usize,
    /// One past the last group.
    pub hi: usize,
}

/// What a row of the ledger is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    /// A declared leg of group `group`.
    Leg {
        /// The group.
        group: usize,
        /// Consideration or obligation.
        side: Side,
        /// Its place among the side's legs.
        index: usize,
    },
    /// A group's side total, when the side has more than one leg.
    GroupTotal {
        /// The group.
        group: usize,
        /// The side.
        side: Side,
    },
    /// A group's result: its consideration and obligation together.
    GroupResult(usize),
    /// The running net through a group after the first.
    RunningNet(usize),
    /// A span's side total.
    SpanTotal {
        /// The span.
        span: usize,
        /// The side.
        side: Side,
    },
    /// A span's result.
    SpanResult(usize),
    /// A side of the position through a span (its net-of block).
    NetTotal {
        /// The span.
        span: usize,
        /// The side.
        side: Side,
    },
    /// The position through a span.
    NetResult(usize),
    /// A grand side total, with several groups.
    GrandTotal(Side),
    /// The grand result, with several groups.
    GrandResult,
    /// The grand result less the first group's.
    TotalImpact,
}

/// The side of the sheet a row sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Consideration: premium.
    Consideration,
    /// Obligation: loss and expenses, or recoveries and commission.
    Obligation,
    /// A result.
    Margin,
}

impl Side {
    /// `Consideration`, `Obligation` or `Margin`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Consideration => "Consideration",
            Self::Obligation => "Obligation",
            Self::Margin => "Margin",
        }
    }
}

/// One row of the ledger, signed (payoff convention).
#[derive(Debug, Clone)]
pub struct Row {
    /// The row's key, unique in the ledger: a leg's label, `total
    /// consideration`, `<group> result`, `net through <group>`, …, as
    /// `aggregate`'s ledger labels.
    pub label: String,
    /// What the row is.
    pub kind: RowKind,
    /// Where it sits on the sheet: step (a group, span, net-of block or
    /// `All`), side, and presentation label (`Total`, `Net`, `Impact`, a
    /// seller's margin label, or the leg's own), as `aggregate`'s
    /// `economic_df` index.
    pub step: String,
    /// The side.
    pub side: Side,
    /// The presentation label.
    pub display: String,
    /// The leg's kind, for a declared leg.
    pub leg_kind: Option<LegKind>,
    values: Vec<f64>,
    distribution: Discrete,
    mean: f64,
    sd: f64,
}

impl Row {
    /// The row's signed value at each atom of the source.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The row's distribution.
    pub fn distribution(&self) -> &Discrete {
        &self.distribution
    }

    /// The mean, off the atoms.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// The standard deviation, off the atoms (`E[X²] − E[X]²`, floored at
    /// 0; exactly 0 for a constant row).
    pub fn sd(&self) -> f64 {
        self.sd
    }

    /// The coefficient of variation; `None` when the mean is within `1e-12`
    /// of 0.
    pub fn cv(&self) -> Option<f64> {
        (self.mean.abs() > NOISE).then(|| self.sd / self.mean)
    }

    /// The skewness off the atoms; `None` when the standard deviation is 0.
    pub fn skew(&self) -> Option<f64> {
        if self.sd <= 0.0 {
            return None;
        }
        let d = &self.distribution;
        Some(
            d.values()
                .iter()
                .zip(d.probs())
                .map(|(x, p)| (x - self.mean).powi(3) * p)
                .sum::<f64>()
                / self.sd.powi(3),
        )
    }

    /// The row's own quantile at `p` (the marginal ladder).
    pub fn quantile(&self, p: f64) -> Result<f64> {
        self.distribution.quantile(p)
    }
}

/// One row of the card, [`Ledger::card`].
#[derive(Debug, Clone, PartialEq)]
pub struct CardRow {
    /// The step: a group, span, net-of block or `All`; empty on a ledger of
    /// one group.
    pub step: String,
    /// `Consideration`, `Obligation`, `Margin`, `Net` or `Impact`.
    pub side: String,
    /// The mean.
    pub ex: f64,
    /// The standard deviation, snapped to 0 within [`NOISE`].
    pub sd: f64,
    /// The coefficient of variation.
    pub cv: Option<f64>,
    /// The skewness, snapped.
    pub skew: Option<f64>,
    /// The 1% quantile of the row's own distribution.
    pub p01: f64,
    /// The median.
    pub median: f64,
    /// The 99% quantile.
    pub p99: f64,
}

/// A source: the atoms a ledger's legs are evaluated at, with their
/// probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    values: Vec<f64>,
    probs: Vec<f64>,
}

impl Source {
    /// Atoms `values` with probabilities `probs`, rescaled to sum to 1.
    pub fn new(values: Vec<f64>, probs: Vec<f64>) -> Result<Self> {
        if values.len() != probs.len() || values.is_empty() {
            return Err(Error::Data(
                "a source needs as many probabilities as values, and at least one".into(),
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Data("the source's values must be finite".into()));
        }
        if probs.iter().any(|p| !(p.is_finite() && *p >= 0.0)) {
            return Err(Error::Data(
                "the source's probabilities must be finite and non-negative".into(),
            ));
        }
        let total: f64 = probs.iter().sum();
        if total <= 0.0 {
            return Err(Error::Data(
                "the source needs a positive probability".into(),
            ));
        }
        Ok(Self {
            values,
            probs: probs.into_iter().map(|p| p / total).collect(),
        })
    }

    /// The points of a grid with mass.
    pub fn from_grid(grid: &Grid) -> Result<Self> {
        let (values, probs): (Vec<f64>, Vec<f64>) = grid
            .probs()
            .iter()
            .enumerate()
            .filter(|(_, p)| **p > 0.0)
            .map(|(k, p)| (grid.x(k), *p))
            .unzip();
        Self::new(values, probs)
    }

    /// The atoms' values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Their probabilities, summing to 1.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }
}

/// A P&L ledger; see the [module](self).
#[derive(Debug, Clone)]
pub struct Ledger {
    name: String,
    result_name: String,
    tower: bool,
    groups: Vec<(String, Role, usize, usize)>,
    spans: Vec<Span>,
    probs: Vec<f64>,
    rows: Vec<Row>,
    grand_consideration: Row,
    grand_obligation: Row,
    result: usize,
}

/// How a ledger is laid out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// The ledger's name (also the label of a single group made by
    /// [`Ledger::single`]).
    pub name: String,
    /// The label of the result row on a ledger of one group, and of the
    /// grand result on one of several; `result` by default.
    pub result_name: String,
    /// The spans with subtotal and net-of blocks.
    pub spans: Vec<Span>,
    /// Present one group as a tower (its block of the card under its own
    /// step), as `aggregate`'s one-step walk.
    pub force_tower: bool,
}

impl Layout {
    /// A layout named `name`, the result named `result`, no spans.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            result_name: "result".into(),
            spans: Vec::new(),
            force_tower: false,
        }
    }

    /// The result row's name.
    pub fn result_name(mut self, name: impl Into<String>) -> Self {
        self.result_name = name.into();
        self
    }

    /// Adds a span `lo..hi`.
    pub fn span(mut self, label: impl Into<String>, lo: usize, hi: usize) -> Self {
        self.spans.push(Span {
            label: label.into(),
            lo,
            hi,
        });
        self
    }

    /// Presents one group as a tower.
    pub fn force_tower(mut self) -> Self {
        self.force_tower = true;
        self
    }
}

fn net_of(label: &str) -> String {
    match label.strip_prefix("All ") {
        Some(rest) => format!("Net of {rest}"),
        None => format!("Net of {label}"),
    }
}

impl Ledger {
    /// The ledger of `groups` over `source`; see the [module](self).
    ///
    /// Fails without a group, with a group that has no leg, with a leg that
    /// is not finite or whose values do not match the source, with a span
    /// out of range, or when two rows would share a label.
    ///
    /// ```
    /// use prospicio_aggregate::pnl::{Group, Layout, Ledger, Leg, Source};
    ///
    /// // Premium 10 against a loss of 2 or 8, equally likely.
    /// let source = Source::new(vec![2.0, 8.0], vec![0.5, 0.5]).unwrap();
    /// let book = Group::sell("Book")
    ///     .consideration(Leg::constant("Premium", 10.0))
    ///     .obligation(Leg::of("Loss", |x| x));
    /// let pnl = Ledger::new(&source, vec![book], Layout::new("Book")).unwrap();
    /// assert_eq!(pnl.result().mean(), 5.0);
    /// // The bad year: the 1% scenario is the year of the loss of 8.
    /// let k = pnl.scenarios(0.01).unwrap();
    /// assert_eq!(k, vec![10.0, -8.0, 2.0]);
    /// ```
    pub fn new(source: &Source, groups: Vec<Group>, layout: Layout) -> Result<Self> {
        if groups.is_empty() {
            return Err(Error::Data("a ledger needs at least one group".into()));
        }
        if let Some(g) = groups
            .iter()
            .find(|g| g.consideration.is_empty() && g.obligation.is_empty())
        {
            return Err(Error::Data(format!("group {:?} has no leg", g.label)));
        }
        let n_groups = groups.len();
        for s in &layout.spans {
            if s.lo > s.hi || s.hi > n_groups {
                return Err(Error::Data(format!(
                    "span {:?} ({}..{}) is outside the {n_groups} groups",
                    s.label, s.lo, s.hi
                )));
            }
        }
        let xs = source.values();
        let probs = source.probs().to_vec();
        let n = xs.len();
        let multi = n_groups > 1;
        let tower = multi || layout.force_tower;
        let has_buy = groups.iter().any(|g| g.role == Role::Buy);
        let net_word = if has_buy { "Net" } else { "Total" };

        let add =
            |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
        let zeros = vec![0.0; n];
        // Each group's signed legs and totals.
        struct Evaluated {
            cons: SignedLegs,
            obl: SignedLegs,
            cons_total: Vec<f64>,
            obl_total: Vec<f64>,
            result: Vec<f64>,
        }
        let mut evaluated = Vec::with_capacity(n_groups);
        for g in &groups {
            let sign = match g.role {
                Role::Sell => 1.0,
                Role::Buy => -1.0,
            };
            let side = |legs: &[Leg], s: f64| -> Result<SignedLegs> {
                legs.iter()
                    .map(|leg| {
                        let v = leg.evaluate(xs)?;
                        Ok((
                            leg.label.clone(),
                            leg.kind,
                            v.into_iter().map(|x| s * x).collect(),
                        ))
                    })
                    .collect()
            };
            let cons = side(&g.consideration, sign)?;
            let obl = side(&g.obligation, -sign)?;
            let cons_total = cons
                .iter()
                .fold(zeros.clone(), |acc, (_, _, v)| add(&acc, v));
            let obl_total = obl
                .iter()
                .fold(zeros.clone(), |acc, (_, _, v)| add(&acc, v));
            let result = add(&cons_total, &obl_total);
            evaluated.push(Evaluated {
                cons,
                obl,
                cons_total,
                obl_total,
                result,
            });
        }
        let mut net_after = Vec::with_capacity(n_groups);
        let mut running = zeros.clone();
        for e in &evaluated {
            running = add(&running, &e.result);
            net_after.push(running.clone());
        }
        let sum_over = |range: std::ops::Range<usize>, f: &dyn Fn(&Evaluated) -> &Vec<f64>| {
            evaluated[range]
                .iter()
                .fold(zeros.clone(), |acc, e| add(&acc, f(e)))
        };

        let make = |label: String,
                    kind: RowKind,
                    step: String,
                    side: Side,
                    display: String,
                    leg_kind: Option<LegKind>,
                    values: Vec<f64>|
         -> Result<Row> {
            Self::build_row(label, kind, step, side, display, leg_kind, values, &probs)
        };

        let spans_after = |gi: usize| -> Vec<usize> {
            (0..layout.spans.len())
                .filter(|&k| {
                    let s = &layout.spans[k];
                    s.hi >= s.lo + 2 && s.hi - 1 == gi
                })
                .collect()
        };
        let nets_after = |gi: usize| -> Vec<usize> {
            (0..layout.spans.len())
                .filter(|&k| {
                    let s = &layout.spans[k];
                    s.lo < s.hi && s.hi < n_groups && s.hi >= 2 && s.hi - 1 == gi
                })
                .collect()
        };

        let mut rows: Vec<Row> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut push = |rows: &mut Vec<Row>, row: Row| -> Result<()> {
            if !seen.insert(row.label.clone()) {
                return Err(Error::Data(format!(
                    "two ledger rows are labelled {:?}; leg labels must be unique",
                    row.label
                )));
            }
            rows.push(row);
            Ok(())
        };
        for (gi, (g, e)) in groups.iter().zip(&evaluated).enumerate() {
            let qual = if multi {
                format!("{} ", g.label)
            } else {
                String::new()
            };
            for (li, (label, kind, v)) in e.cons.iter().enumerate() {
                push(
                    &mut rows,
                    make(
                        label.clone(),
                        RowKind::Leg {
                            group: gi,
                            side: Side::Consideration,
                            index: li,
                        },
                        g.label.clone(),
                        Side::Consideration,
                        label.clone(),
                        *kind,
                        v.clone(),
                    )?,
                )?;
            }
            if e.cons.len() > 1 {
                push(
                    &mut rows,
                    make(
                        format!("{qual}total consideration"),
                        RowKind::GroupTotal {
                            group: gi,
                            side: Side::Consideration,
                        },
                        g.label.clone(),
                        Side::Consideration,
                        "Total".into(),
                        None,
                        e.cons_total.clone(),
                    )?,
                )?;
            }
            for (li, (label, kind, v)) in e.obl.iter().enumerate() {
                push(
                    &mut rows,
                    make(
                        label.clone(),
                        RowKind::Leg {
                            group: gi,
                            side: Side::Obligation,
                            index: li,
                        },
                        g.label.clone(),
                        Side::Obligation,
                        label.clone(),
                        *kind,
                        v.clone(),
                    )?,
                )?;
            }
            if e.obl.len() > 1 {
                push(
                    &mut rows,
                    make(
                        format!("{qual}total obligation"),
                        RowKind::GroupTotal {
                            group: gi,
                            side: Side::Obligation,
                        },
                        g.label.clone(),
                        Side::Obligation,
                        "Total".into(),
                        None,
                        e.obl_total.clone(),
                    )?,
                )?;
            }
            let label = if multi {
                format!("{} result", g.label)
            } else {
                layout.result_name.clone()
            };
            let display = if has_buy && g.role == Role::Sell {
                g.margin_label.clone().unwrap_or_else(|| "Gross".into())
            } else {
                "Total".into()
            };
            push(
                &mut rows,
                make(
                    label,
                    RowKind::GroupResult(gi),
                    g.label.clone(),
                    Side::Margin,
                    display,
                    None,
                    e.result.clone(),
                )?,
            )?;
            if multi && gi > 0 {
                push(
                    &mut rows,
                    make(
                        format!("net through {}", g.label),
                        RowKind::RunningNet(gi),
                        g.label.clone(),
                        Side::Margin,
                        "Net".into(),
                        None,
                        net_after[gi].clone(),
                    )?,
                )?;
            }
            for k in spans_after(gi) {
                let s = &layout.spans[k];
                push(
                    &mut rows,
                    make(
                        format!("{} total consideration", s.label),
                        RowKind::SpanTotal {
                            span: k,
                            side: Side::Consideration,
                        },
                        s.label.clone(),
                        Side::Consideration,
                        "Total".into(),
                        None,
                        sum_over(s.lo..s.hi, &|e| &e.cons_total),
                    )?,
                )?;
                push(
                    &mut rows,
                    make(
                        format!("{} total obligation", s.label),
                        RowKind::SpanTotal {
                            span: k,
                            side: Side::Obligation,
                        },
                        s.label.clone(),
                        Side::Obligation,
                        "Total".into(),
                        None,
                        sum_over(s.lo..s.hi, &|e| &e.obl_total),
                    )?,
                )?;
                push(
                    &mut rows,
                    make(
                        format!("{} result", s.label),
                        RowKind::SpanResult(k),
                        s.label.clone(),
                        Side::Margin,
                        "Total".into(),
                        None,
                        sum_over(s.lo..s.hi, &|e| &e.result),
                    )?,
                )?;
            }
            for k in nets_after(gi) {
                let s = &layout.spans[k];
                let step = net_of(&s.label);
                let net = step.to_lowercase();
                push(
                    &mut rows,
                    make(
                        format!("{net} consideration"),
                        RowKind::NetTotal {
                            span: k,
                            side: Side::Consideration,
                        },
                        step.clone(),
                        Side::Consideration,
                        "Net".into(),
                        None,
                        sum_over(0..s.hi, &|e| &e.cons_total),
                    )?,
                )?;
                push(
                    &mut rows,
                    make(
                        format!("{net} obligation"),
                        RowKind::NetTotal {
                            span: k,
                            side: Side::Obligation,
                        },
                        step.clone(),
                        Side::Obligation,
                        "Net".into(),
                        None,
                        sum_over(0..s.hi, &|e| &e.obl_total),
                    )?,
                )?;
                push(
                    &mut rows,
                    make(
                        net,
                        RowKind::NetResult(k),
                        step,
                        Side::Margin,
                        "Net".into(),
                        None,
                        net_after[s.hi - 1].clone(),
                    )?,
                )?;
            }
        }
        let grand_cons = sum_over(0..n_groups, &|e| &e.cons_total);
        let grand_obl = sum_over(0..n_groups, &|e| &e.obl_total);
        let result_index;
        let (grand_consideration, grand_obligation);
        if multi {
            let gc = make(
                "total consideration".into(),
                RowKind::GrandTotal(Side::Consideration),
                "All".into(),
                Side::Consideration,
                net_word.into(),
                None,
                grand_cons,
            )?;
            let go = make(
                "total obligation".into(),
                RowKind::GrandTotal(Side::Obligation),
                "All".into(),
                Side::Obligation,
                net_word.into(),
                None,
                grand_obl,
            )?;
            push(&mut rows, gc.clone())?;
            push(&mut rows, go.clone())?;
            push(
                &mut rows,
                make(
                    layout.result_name.clone(),
                    RowKind::GrandResult,
                    "All".into(),
                    Side::Margin,
                    net_word.into(),
                    None,
                    net_after[n_groups - 1].clone(),
                )?,
            )?;
            result_index = rows.len() - 1;
            let impact: Vec<f64> = net_after[n_groups - 1]
                .iter()
                .zip(&evaluated[0].result)
                .map(|(a, b)| a - b)
                .collect();
            push(
                &mut rows,
                make(
                    "total impact".into(),
                    RowKind::TotalImpact,
                    "All".into(),
                    Side::Margin,
                    "Impact".into(),
                    None,
                    impact,
                )?,
            )?;
            grand_consideration = gc;
            grand_obligation = go;
        } else {
            result_index = rows
                .iter()
                .position(|r| r.kind == RowKind::GroupResult(0))
                .expect("the plan has the group's result");
            grand_consideration = make(
                "total consideration".into(),
                RowKind::GrandTotal(Side::Consideration),
                "All".into(),
                Side::Consideration,
                "Total".into(),
                None,
                grand_cons,
            )?;
            grand_obligation = make(
                "total obligation".into(),
                RowKind::GrandTotal(Side::Obligation),
                "All".into(),
                Side::Obligation,
                "Total".into(),
                None,
                grand_obl,
            )?;
        }
        Ok(Self {
            name: layout.name,
            result_name: layout.result_name,
            tower,
            groups: groups
                .iter()
                .map(|g| {
                    (
                        g.label.clone(),
                        g.role,
                        g.consideration.len(),
                        g.obligation.len(),
                    )
                })
                .collect(),
            spans: layout.spans,
            probs,
            rows,
            grand_consideration,
            grand_obligation,
            result: result_index,
        })
    }

    /// A ledger of one group named `name`, as `aggregate`'s single-group
    /// `PnL(role=…, consideration=…, obligation=…)`.
    pub fn single(
        source: &Source,
        name: impl Into<String>,
        role: Role,
        consideration: Vec<Leg>,
        obligation: Vec<Leg>,
        result_name: impl Into<String>,
    ) -> Result<Self> {
        let name = name.into();
        let mut g = Group::new(name.clone(), role);
        g.consideration = consideration;
        g.obligation = obligation;
        Self::new(source, vec![g], Layout::new(name).result_name(result_name))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_row(
        label: String,
        kind: RowKind,
        step: String,
        side: Side,
        display: String,
        leg_kind: Option<LegKind>,
        values: Vec<f64>,
        probs: &[f64],
    ) -> Result<Row> {
        let distribution = Discrete::new(values.clone(), probs.to_vec())?;
        let (lo, hi) = values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        let (mean, sd) = if lo == hi {
            (values[0], 0.0)
        } else {
            let m: f64 = values.iter().zip(probs).map(|(v, p)| v * p).sum();
            let m2: f64 = values.iter().zip(probs).map(|(v, p)| v * v * p).sum();
            (m, (m2 - m * m).max(0.0).sqrt())
        };
        Ok(Row {
            label,
            kind,
            step,
            side,
            display,
            leg_kind,
            values,
            distribution,
            mean,
            sd,
        })
    }

    /// The ledger's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The result row's name.
    pub fn result_name(&self) -> &str {
        &self.result_name
    }

    /// Whether the ledger is presented as a tower (several groups, or one
    /// forced): the sheet then has a step level.
    pub fn is_tower(&self) -> bool {
        self.tower
    }

    /// The rows, in sheet order.
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// The row labelled `label`.
    pub fn row(&self, label: &str) -> Option<&Row> {
        self.rows.iter().find(|r| r.label == label)
    }

    /// The (grand) result.
    pub fn result(&self) -> &Row {
        &self.rows[self.result]
    }

    /// The source's probabilities, summing to 1.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// The chance that the result is 0 (to `1e-12`).
    pub fn prob_eq_0(&self) -> f64 {
        self.result()
            .values
            .iter()
            .zip(&self.probs)
            .filter(|(v, _)| same(**v, 0.0))
            .map(|(_, p)| p)
            .sum()
    }

    /// The scenario column at level `q`: each row's conditional mean, in
    /// sheet order, given that the result equals its lower `q`-quantile
    /// (snapped within [`NOISE`] of 0). The result's own cell is that
    /// quantile.
    pub fn scenarios(&self, q: f64) -> Result<Vec<f64>> {
        let result = self.result();
        let x = result.quantile(q)?;
        let mask: Vec<bool> = result.values.iter().map(|&v| same(v, x)).collect();
        let mass: f64 = mask
            .iter()
            .zip(&self.probs)
            .filter(|(m, _)| **m)
            .map(|(_, p)| p)
            .sum();
        Ok(self
            .rows
            .iter()
            .map(|row| {
                snap(
                    row.values
                        .iter()
                        .zip(&self.probs)
                        .zip(&mask)
                        .filter(|(_, m)| **m)
                        .map(|((v, p), _)| v * p)
                        .sum::<f64>()
                        / mass,
                )
            })
            .collect())
    }

    /// The card: each block's consideration, obligation and margin (and on
    /// a tower each step's running net and the covers' total impact), with
    /// the mean, standard deviation, coefficient of variation, skewness and
    /// the 1%, 50% and 99% quantiles of each row's own distribution, as
    /// `aggregate`'s `summary_df`. Its quantiles are marginal and do not add
    /// down the card; the [`scenarios`](Ledger::scenarios) do.
    pub fn card(&self) -> Result<Vec<CardRow>> {
        let stat = |step: &str, side: &str, row: &Row| -> Result<CardRow> {
            Ok(CardRow {
                step: step.into(),
                side: side.into(),
                ex: row.mean,
                sd: snap(row.sd),
                cv: row.cv(),
                skew: row.skew().map(snap),
                p01: row.quantile(0.01)?,
                median: row.quantile(0.5)?,
                p99: row.quantile(0.99)?,
            })
        };
        if !self.tower {
            return Ok(vec![
                stat("", "Consideration", &self.grand_consideration)?,
                stat("", "Obligation", &self.grand_obligation)?,
                stat("", "Margin", self.result())?,
            ]);
        }
        let find = |kind: RowKind| self.rows.iter().find(|r| r.kind == kind);
        let mut out = Vec::new();
        for (gi, (label, _, n_cons, n_obl)) in self.groups.iter().enumerate() {
            for (side, count) in [(Side::Consideration, *n_cons), (Side::Obligation, *n_obl)] {
                let row = match count {
                    0 => None,
                    1 => find(RowKind::Leg {
                        group: gi,
                        side,
                        index: 0,
                    }),
                    _ => find(RowKind::GroupTotal { group: gi, side }),
                };
                match row {
                    Some(r) => out.push(stat(label, side.name(), r)?),
                    None => {
                        let zero = Self::build_row(
                            String::new(),
                            RowKind::GroupTotal { group: gi, side },
                            label.clone(),
                            side,
                            "Total".into(),
                            None,
                            vec![0.0; self.probs.len()],
                            &self.probs,
                        )?;
                        out.push(stat(label, side.name(), &zero)?);
                    }
                }
            }
            out.push(stat(
                label,
                "Margin",
                find(RowKind::GroupResult(gi)).expect("every group has a result"),
            )?);
            if gi > 0 {
                out.push(stat(
                    label,
                    "Net",
                    find(RowKind::RunningNet(gi)).expect("a running net follows each later group"),
                )?);
            }
            for (k, s) in self.spans.iter().enumerate() {
                if s.hi >= s.lo + 2 && s.hi - 1 == gi {
                    out.push(stat(
                        &s.label,
                        "Consideration",
                        find(RowKind::SpanTotal {
                            span: k,
                            side: Side::Consideration,
                        })
                        .expect("span row"),
                    )?);
                    out.push(stat(
                        &s.label,
                        "Obligation",
                        find(RowKind::SpanTotal {
                            span: k,
                            side: Side::Obligation,
                        })
                        .expect("span row"),
                    )?);
                    out.push(stat(
                        &s.label,
                        "Margin",
                        find(RowKind::SpanResult(k)).expect("span row"),
                    )?);
                }
            }
            for (k, s) in self.spans.iter().enumerate() {
                if s.lo < s.hi && s.hi < self.groups.len() && s.hi >= 2 && s.hi - 1 == gi {
                    let step = net_of(&s.label);
                    out.push(stat(
                        &step,
                        "Consideration",
                        find(RowKind::NetTotal {
                            span: k,
                            side: Side::Consideration,
                        })
                        .expect("net row"),
                    )?);
                    out.push(stat(
                        &step,
                        "Obligation",
                        find(RowKind::NetTotal {
                            span: k,
                            side: Side::Obligation,
                        })
                        .expect("net row"),
                    )?);
                    out.push(stat(
                        &step,
                        "Margin",
                        find(RowKind::NetResult(k)).expect("net row"),
                    )?);
                }
            }
        }
        if self.groups.len() > 1 {
            out.push(stat("All", "Consideration", &self.grand_consideration)?);
            out.push(stat("All", "Obligation", &self.grand_obligation)?);
            out.push(stat("All", "Margin", self.result())?);
            out.push(stat(
                "All",
                "Impact",
                find(RowKind::TotalImpact).expect("several groups have an impact row"),
            )?);
        }
        Ok(out)
    }

    /// The ledger of `self`'s groups followed by `other`'s, over the same
    /// source (`aggregate`'s `pnl_a + pnl_b`): every row is rebuilt from the
    /// legs' values. Fails if the sources differ.
    pub fn extend(&self, other: &Ledger, layout: Layout) -> Result<Ledger> {
        if self.probs.len() != other.probs.len()
            || self
                .probs
                .iter()
                .zip(&other.probs)
                .any(|(a, b)| !same(*a, *b))
        {
            return Err(Error::Data(
                "ledgers can be added only over the same source".into(),
            ));
        }
        let mut groups = Vec::new();
        for ledger in [self, other] {
            for (gi, (label, role, _, _)) in ledger.groups.iter().enumerate() {
                let mut g = Group::new(label.clone(), *role);
                for row in &ledger.rows {
                    if let RowKind::Leg { group, side, .. } = row.kind
                        && group == gi
                    {
                        // Back to the leg's magnitude.
                        let sign = match (role, side) {
                            (Role::Sell, Side::Consideration) | (Role::Buy, Side::Obligation) => {
                                1.0
                            }
                            _ => -1.0,
                        };
                        let mut leg = Leg::values(
                            row.label.clone(),
                            row.values.iter().map(|v| sign * v).collect(),
                        );
                        leg.kind = row.leg_kind;
                        g = if side == Side::Consideration {
                            g.consideration(leg)
                        } else {
                            g.obligation(leg)
                        };
                    }
                }
                if let Some(m) = ledger
                    .rows
                    .iter()
                    .find(|r| r.kind == RowKind::GroupResult(gi))
                    && m.display != "Total"
                    && m.display != "Gross"
                {
                    g = g.margin_label(m.display.clone());
                }
                groups.push(g);
            }
        }
        let n = self.probs.len();
        let source = Source {
            values: (0..n).map(|k| k as f64).collect(),
            probs: self.probs.clone(),
        };
        Ledger::new(&source, groups, layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_books() -> Ledger {
        let source = Source::new(vec![0.0, 10.0, 20.0, 50.0], vec![0.4, 0.3, 0.2, 0.1]).unwrap();
        let gross = Group::sell("Gross")
            .consideration(Leg::of("Premium", |x| (5.0 + 1.1 * x).clamp(12.0, 40.0)))
            .obligation(Leg::of("Loss", |x| x))
            .obligation(Leg::of("LAE", |x| 0.05 * x));
        let xl = Group::buy("XL")
            .consideration(Leg::constant("XL premium", 4.0))
            .obligation(Leg::of("XL recovery", |x| (x - 15.0).clamp(0.0, 25.0)))
            .obligation(Leg::constant("XL commission", 0.4));
        Ledger::new(
            &source,
            vec![gross, xl],
            Layout::new("Ex2").result_name("margin"),
        )
        .unwrap()
    }

    #[test]
    fn rows_foot_and_scenarios_add_down() {
        let pnl = two_books();
        let labels: Vec<&str> = pnl.rows().iter().map(|r| r.label.as_str()).collect();
        assert_eq!(
            labels,
            vec![
                "Premium",
                "Loss",
                "LAE",
                "Gross total obligation",
                "Gross result",
                "XL premium",
                "XL recovery",
                "XL commission",
                "XL total obligation",
                "XL result",
                "net through XL",
                "total consideration",
                "total obligation",
                "margin",
                "total impact",
            ]
        );
        assert!((pnl.result().mean() - 6.3).abs() < 1e-12);
        assert!((pnl.row("Gross result").unwrap().mean() - 6.4).abs() < 1e-12);
        for q in LADDER {
            let k = pnl.scenarios(q).unwrap();
            // margin = total consideration + total obligation, in every state.
            assert!((k[13] - (k[11] + k[12])).abs() < 1e-12);
            assert_eq!(k[13], pnl.result().quantile(q).unwrap());
        }
        let card = pnl.card().unwrap();
        assert_eq!(card.last().unwrap().side, "Impact");
        assert!((card.last().unwrap().ex - (6.3 - 6.4)).abs() < 1e-12);
    }

    #[test]
    fn a_level_set_and_a_constant() {
        let source = Source::new(vec![0.0, 10.0, 20.0, 50.0], vec![0.4, 0.3, 0.2, 0.1]).unwrap();
        let pnl = Ledger::single(
            &source,
            "Ex2c",
            Role::Sell,
            vec![Leg::of("Prem", |x| if x == 20.0 { 30.0 } else { 10.0 })],
            vec![Leg::of("Loss", |x| x)],
            "result",
        )
        .unwrap();
        // Results 10, 0, 10, -40: the median state is both atoms at 10.
        let k = pnl.scenarios(0.5).unwrap();
        assert!((k[0] - 50.0 / 3.0).abs() < 1e-12);
        assert!((k[2] - 10.0).abs() < 1e-12);
        let premium_sd = Ledger::single(
            &source,
            "c",
            Role::Sell,
            vec![Leg::constant("P", 3.0)],
            vec![Leg::of("L", |x| x)],
            "r",
        )
        .unwrap();
        assert_eq!(premium_sd.rows()[0].sd(), 0.0);
        assert_eq!(premium_sd.rows()[0].skew(), None);
    }

    #[test]
    fn bad_ledgers_are_refused() {
        let source = Source::new(vec![1.0, 2.0], vec![0.5, 0.5]).unwrap();
        assert!(Ledger::new(&source, vec![], Layout::new("x")).is_err());
        assert!(Ledger::new(&source, vec![Group::sell("A")], Layout::new("x")).is_err());
        let dup = vec![
            Group::sell("A").consideration(Leg::constant("P", 1.0)),
            Group::buy("B").consideration(Leg::constant("P", 1.0)),
        ];
        assert!(Ledger::new(&source, dup, Layout::new("x")).is_err());
        let bad = Group::sell("A").obligation(Leg::values("L", vec![1.0]));
        assert!(Ledger::new(&source, vec![bad], Layout::new("x")).is_err());
        let span = Layout::new("x").span("All", 0, 3);
        assert!(
            Ledger::new(
                &source,
                vec![Group::sell("A").consideration(Leg::constant("P", 1.0))],
                span
            )
            .is_err()
        );
        assert!(Source::new(vec![1.0], vec![0.0]).is_err());
    }

    #[test]
    fn extending_a_ledger_keeps_its_rows() {
        let pnl = two_books();
        let source = Source::new(vec![0.0, 10.0, 20.0, 50.0], vec![0.4, 0.3, 0.2, 0.1]).unwrap();
        let xl2 = Ledger::new(
            &source,
            vec![Group::buy("XL2").consideration(Leg::constant("XL2 premium", 1.0))],
            Layout::new("XL2"),
        )
        .unwrap();
        let both = pnl
            .extend(&xl2, Layout::new("Both").result_name("margin"))
            .unwrap();
        assert!((both.result().mean() - 5.3).abs() < 1e-12);
        assert!(
            (both.row("XL recovery").unwrap().mean() - pnl.row("XL recovery").unwrap().mean())
                .abs()
                < 1e-15
        );
    }
}
