//! A portfolio of deals rolled up simulation by simulation (#490): deals
//! chosen by name, each with an inception in whole months, and four tables
//! on the book. See `docs/design/time-value.md`, decisions 21 to 27.
//!
//! Every deal reads one shared simulation, so simulation `j` of one deal
//! and simulation `j` of another are the same year: the roll-up adds them
//! without reordering (decision 21). A [`DealRun`] holds one deal's
//! amounts (the premium, the loss and each deduction, as
//! [`Tower::deal_columns`] gives them), and optionally its monthly
//! patterns and its investor's flows from [`CashflowTerms::run`]. A
//! [`DealBook`] holds several and gives:
//!
//! - [`amounts`](DealBook::amounts) and [`summary`](DealBook::summary):
//!   every amount by deal and in total, with the ratios as sums over sums;
//! - [`distribution`](DealBook::distribution): the book's statistics of
//!   one item, each deal's value at the book's ranks and the marginal
//!   effect of each deal;
//! - [`cashflows`](DealBook::cashflows): the investors' flows on one
//!   calendar, with the book's IRR, NPV and MOIC on the summed flows;
//! - [`calendar`](DealBook::calendar): written and earned premium, paid
//!   loss and investor flows by calendar year.
//!
//! The inception moves only the cash flows and the patterns, not the
//! events: each deal's year is its own contract year (decision 22). Book
//! month 0 is 1 January of the first calendar year, so an inception of 6
//! is 1 July. Each deal keeps its own collateral trust (decision 23); the
//! book only adds the investors' flows.

use prospicio_aggregate::{DealColumns, Tower};
use prospicio_core::{Error, Result};
use prospicio_prob::statistics::{IndexRule, Statistics, StatisticsGrid};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance, Sampled};

use crate::cashflow::{CashflowTerms, Cashflows, MonthlyPatterns};
use crate::returns::{Irr, IrrOptions, TimeUnit, irr_rows, moic, npv_rows};

/// One deduction of a [`DealRun`]: a commission, the brokerage or another
/// deduction, an amount in each simulation.
#[derive(Debug, Clone, PartialEq)]
struct Line {
    name: String,
    outside_nmbf: bool,
    values: Vec<f64>,
}

/// The investor's flows of one deal: `months` a row, one row a simulation.
#[derive(Debug, Clone, PartialEq)]
struct Flows {
    months: usize,
    flows: Vec<f64>,
    funded: Vec<f64>,
    opening: Vec<f64>,
}

/// One deal's results in each simulation of the shared simulation, with
/// its inception in months from the start of the book.
///
/// ```
/// use prospicio_pricing::book::DealRun;
///
/// let run = DealRun::new("cat", vec![100.0, 100.0], vec![40.0, 150.0]).unwrap()
///     .deduction("brokerage", vec![2.5, 2.5], false).unwrap()
///     .inception(6);
/// assert_eq!(run.n_sims(), 2);
/// assert_eq!(run.inception_month(), 6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DealRun {
    name: String,
    premium: Vec<f64>,
    loss: Vec<f64>,
    lines: Vec<Line>,
    inception: usize,
    patterns: Option<MonthlyPatterns>,
    flows: Option<Flows>,
}

fn finite(values: &[f64], name: &'static str) -> Result<()> {
    if let Some(&bad) = values.iter().find(|x| !x.is_finite()) {
        return Err(Error::InvalidParameter {
            name,
            value: bad,
            reason: "must be finite",
        });
    }
    Ok(())
}

/// Names a deal's lines may not take: the summary's own lines.
const DERIVED: &[&str] = &["premium", "loss", "total_deductions", "nmbf", "margin"];

impl DealRun {
    /// A deal's premium and loss in each simulation, incepting at month 0.
    ///
    /// Fails if the name is empty, the two differ in length or are empty,
    /// or a value is not finite.
    pub fn new(name: &str, premium: Vec<f64>, loss: Vec<f64>) -> Result<Self> {
        if name.is_empty() {
            return Err(Error::Data("a deal needs a name".into()));
        }
        if premium.is_empty() || premium.len() != loss.len() {
            return Err(Error::Data(format!(
                "{name:?} has {} premiums and {} losses; it needs one of each a simulation",
                premium.len(),
                loss.len()
            )));
        }
        finite(&premium, "premium")?;
        finite(&loss, "loss")?;
        Ok(DealRun {
            name: name.into(),
            premium,
            loss,
            lines: Vec::new(),
            inception: 0,
            patterns: None,
            flows: None,
        })
    }

    /// A layer's or deal's run from a result of `tower`
    /// ([`Tower::deal_columns`]): its premium with the year's reinstatement
    /// premium, ceded loss, commissions, brokerage and deductions, under
    /// the name `name`. Each year's premium is the deposit (or swing-rated)
    /// premium plus what the year's losses reinstate, pro rata as to amount
    /// and, where the layer says so, as to time (#595). A layer without paid
    /// reinstatements keeps its deposit premium.
    ///
    /// ```
    /// use prospicio_aggregate::{EventSet, Layer, Tower};
    /// use prospicio_pricing::book::DealRun;
    ///
    /// // 10 xs 10, premium 2, one reinstatement at 100%: a year with a
    /// // layer loss of 5 pays 2 × 5 / 10 = 1 more.
    /// let layer = Layer::xol("cat", 10.0, 10.0).unwrap()
    ///     .paid_reinstatements(2.0, vec![1.0]).unwrap();
    /// let tower = Tower::new(vec![layer]).unwrap();
    /// let result = tower.apply(&EventSet::from_years(vec![vec![5.0], vec![15.0]], 1).unwrap()).unwrap();
    /// let run = DealRun::from_tower("cat", &tower, &result, "cat").unwrap();
    /// assert_eq!(run.premium(), [2.0, 3.0]);
    /// let deposit = DealRun::from_tower_with("cat", &tower, &result, "cat", false).unwrap();
    /// assert_eq!(deposit.premium(), [2.0, 2.0]);
    /// ```
    pub fn from_tower(
        name: &str,
        tower: &Tower,
        result: &PredictiveDistribution,
        layer: &str,
    ) -> Result<Self> {
        Self::from_tower_with(name, tower, result, layer, true)
    }

    /// [`from_tower`](Self::from_tower), with the reinstatement premium in
    /// the premium or, with `reinstatement_premium` false, left out, so that
    /// each year's premium is the deposit as [`Tower::deal_ratios`] reads
    /// it.
    pub fn from_tower_with(
        name: &str,
        tower: &Tower,
        result: &PredictiveDistribution,
        layer: &str,
        reinstatement_premium: bool,
    ) -> Result<Self> {
        let mut columns = tower.deal_columns(result, layer)?;
        if !reinstatement_premium {
            columns.reinstatement_premium.fill(0.0);
        }
        Self::from_columns(name, columns)
    }

    /// A run from the columns [`Tower::deal_columns`] gives, its premium
    /// the columns' premium plus their reinstatement premium.
    pub fn from_columns(name: &str, columns: DealColumns) -> Result<Self> {
        if columns.reinstatement_premium.len() != columns.premium.len() {
            return Err(Error::Data(format!(
                "{name:?} has {} premiums and {} reinstatement premiums",
                columns.premium.len(),
                columns.reinstatement_premium.len()
            )));
        }
        let premium = columns
            .premium
            .iter()
            .zip(&columns.reinstatement_premium)
            .map(|(p, r)| p + r)
            .collect();
        let mut run = Self::new(name, premium, columns.loss)?;
        for d in columns.deductions {
            run = run.deduction(&d.name, d.values, d.outside_nmbf)?;
        }
        Ok(run)
    }

    /// Adds a commission, the brokerage or another deduction, an amount in
    /// each simulation, inside NMBF or outside it (a manager's fee). Lines
    /// with the same name in different deals add up in the book.
    ///
    /// Fails if the name is empty, one of the summary's own lines
    /// (`premium`, `loss`, `total_deductions`, `nmbf`, `margin`) or already
    /// a line of this deal, or the values are not one finite amount a
    /// simulation.
    pub fn deduction(mut self, name: &str, values: Vec<f64>, outside_nmbf: bool) -> Result<Self> {
        if name.is_empty() || DERIVED.contains(&name) {
            return Err(Error::Data(format!(
                "a deduction may not be named {name:?}"
            )));
        }
        if self.lines.iter().any(|l| l.name == name) {
            return Err(Error::Data(format!(
                "{:?} already has a deduction {name:?}",
                self.name
            )));
        }
        if values.len() != self.premium.len() {
            return Err(Error::Data(format!(
                "{name:?} has {} values for {} simulations",
                values.len(),
                self.premium.len()
            )));
        }
        finite(&values, "values")?;
        self.lines.push(Line {
            name: name.into(),
            outside_nmbf,
            values,
        });
        Ok(self)
    }

    /// The month the deal incepts, counted from book month 0 (1 January
    /// of the book's first calendar year). It moves the deal's flows and
    /// patterns on the book's calendar, not its losses.
    pub fn inception(mut self, month: usize) -> Self {
        self.inception = month;
        self
    }

    /// The deal's monthly patterns, for [`DealBook::calendar`]: the written
    /// and earned premium and the paid loss, cumulative at the end of each
    /// month from inception.
    pub fn patterns(mut self, patterns: MonthlyPatterns) -> Self {
        self.patterns = Some(patterns);
        self
    }

    /// The investor's flows from [`CashflowTerms::run`], one simulation of
    /// `cashflows` for each of the deal's, and the terms' patterns.
    ///
    /// Fails if `cashflows` has another number of simulations.
    pub fn cashflows(self, terms: &CashflowTerms, cashflows: &Cashflows) -> Result<Self> {
        self.flows(
            cashflows.months(),
            cashflows.flows().to_vec(),
            cashflows.funded().to_vec(),
            cashflows.opening().to_vec(),
        )
        .map(|run| run.patterns(terms.patterns.clone()))
    }

    /// The investor's net flows in months `0..months` from inception, one
    /// row a simulation of a row-major matrix, with the capital it paid in
    /// during the term (`funded`) and at inception (`opening`), from any
    /// cash-flow model.
    ///
    /// Fails if the matrix is not `months` columns for each simulation, if
    /// `funded` or `opening` is not one value a simulation, or a value is
    /// not finite.
    pub fn flows(
        mut self,
        months: usize,
        flows: Vec<f64>,
        funded: Vec<f64>,
        opening: Vec<f64>,
    ) -> Result<Self> {
        let n = self.premium.len();
        if months == 0 || flows.len() != months * n || funded.len() != n || opening.len() != n {
            return Err(Error::Data(format!(
                "{:?}'s flows need {n} rows of {months} months, and capital funded and opening \
                 for each of {n} simulations",
                self.name
            )));
        }
        finite(&flows, "flows")?;
        finite(&funded, "funded")?;
        finite(&opening, "opening")?;
        self.flows = Some(Flows {
            months,
            flows,
            funded,
            opening,
        });
        Ok(self)
    }

    /// The deal's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The number of simulations.
    pub fn n_sims(&self) -> usize {
        self.premium.len()
    }

    /// The premium in each simulation.
    pub fn premium(&self) -> &[f64] {
        &self.premium
    }

    /// The loss in each simulation.
    pub fn loss(&self) -> &[f64] {
        &self.loss
    }

    /// The inception month.
    pub fn inception_month(&self) -> usize {
        self.inception
    }

    fn margin(&self, j: usize) -> f64 {
        self.premium[j] - self.loss[j] - self.lines.iter().map(|l| l.values[j]).sum::<f64>()
    }
}

/// The tail of an item [`DealBook::distribution`] reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tail {
    /// High values are the risk, as for a loss: percentiles and TVaRs as
    /// the grid reads them.
    #[default]
    Upper,
    /// Low values are the risk, as for a margin: the grid reads the item's
    /// negative and the values are turned back, so `"Percentile 99"` is the
    /// value the item falls below with probability 1% and `"TVaR 99"` the
    /// mean of the lowest 1%.
    Lower,
}

/// One deal's line in [`Distribution`].
#[derive(Debug, Clone, PartialEq)]
pub struct DealContribution {
    /// The deal's name.
    pub deal: String,
    /// The deal's item at the book's ranks ([`StatisticsGrid::co_statistics`]):
    /// its own mean (not its value at the book's mean), its value in the simulation at each percentile of the
    /// book's total, and its mean over each TVaR's tail of the book. Over
    /// the deals these add up to the book's total.
    pub contribution: Statistics,
    /// The book's statistic less the book's without this deal, for the same
    /// cells. They need not add up to the book's.
    pub marginal: Statistics,
}

/// The book's distribution of one item, from [`DealBook::distribution`].
#[derive(Debug, Clone, PartialEq)]
pub struct Distribution {
    /// The item.
    pub item: String,
    /// The book's statistics of the item, every cell of the grid.
    pub total: Statistics,
    /// Each deal's contribution and marginal effect, on the cells `"Mean"`,
    /// `"Percentile <q>"` and `"TVaR <q>"`.
    pub deals: Vec<DealContribution>,
}

/// The book's investor flows from [`DealBook::cashflows`].
#[derive(Debug, Clone, PartialEq)]
pub struct BookCashflows {
    /// The book's months `0..months`, month 0 at 1 January of its first
    /// year.
    pub months: usize,
    /// The deals' names, in the book's order.
    pub deals: Vec<String>,
    /// The book's net flow in each month, the deals' flows summed after
    /// their inceptions: one simulation a row of a row-major matrix.
    pub flows: Vec<f64>,
    /// Each deal's mean flow in each month of the book, in the order of
    /// [`deals`](Self::deals).
    pub mean_by_deal: Vec<Vec<f64>>,
    /// The book's IRR in each simulation, on its summed flows.
    pub irr: Vec<Irr>,
    /// The book's NPV in each simulation at the rate given, to month 0.
    pub npv: Vec<f64>,
    /// The book's MOIC in each simulation: `1 + Σ flows / C`, with `C` the
    /// capital funded plus the opening capital over every deal.
    pub moic: Vec<f64>,
    /// Each deal's own IRR in each simulation, in the order of
    /// [`deals`](Self::deals). The inception does not change it.
    pub irr_by_deal: Vec<Vec<Irr>>,
}

/// Several deals on one shared simulation.
///
/// Simulation `j` of every deal must be the same scenario: the book adds
/// the deals' flows index by index, so their dependence is the one their
/// inputs had at the same index. Build each deal's premium and losses from
/// one simulation in one order, for example a tower applied with
/// `Tower::apply_annual` to a reserving fit's `loss_ratio_draws` times the
/// premium (`prospicio_reserving::UltimateDraws`), and do not sort or
/// resample one deal's inputs alone.
#[derive(Debug, Clone, PartialEq)]
pub struct DealBook {
    deals: Vec<DealRun>,
}

impl DealBook {
    /// A book of deals on one shared simulation.
    ///
    /// Fails if there are no deals, two share a name, they have different
    /// numbers of simulations (the deals must come from one simulation, in
    /// one order), or two deals' deductions of the same name disagree on
    /// whether NMBF leaves them out.
    pub fn new(deals: Vec<DealRun>) -> Result<Self> {
        let Some(first) = deals.first() else {
            return Err(Error::Data("a book needs at least one deal".into()));
        };
        let n = first.n_sims();
        for (k, d) in deals.iter().enumerate() {
            if deals[..k].iter().any(|e| e.name == d.name) {
                return Err(Error::Data(format!("two deals are named {:?}", d.name)));
            }
            if d.n_sims() != n {
                return Err(Error::Data(format!(
                    "{:?} has {} simulations and {:?} {n}; a book's deals read one shared \
                     simulation",
                    d.name,
                    d.n_sims(),
                    first.name
                )));
            }
        }
        let book = DealBook { deals };
        book.line_names()?;
        Ok(book)
    }

    /// The deals named in `names`, in that order.
    ///
    /// Fails if a name is not a deal of the book or appears twice.
    pub fn select(&self, names: &[&str]) -> Result<Self> {
        let deals = names
            .iter()
            .map(|n| {
                self.deals
                    .iter()
                    .find(|d| d.name == *n)
                    .cloned()
                    .ok_or_else(|| Error::Data(format!("the book has no deal {n:?}")))
            })
            .collect::<Result<_>>()?;
        Self::new(deals)
    }

    /// The book with the deal `name` incepting at `month` instead.
    ///
    /// Fails if the book has no deal `name`.
    pub fn shifted(&self, name: &str, month: usize) -> Result<Self> {
        let mut book = self.clone();
        let d = book
            .deals
            .iter_mut()
            .find(|d| d.name == name)
            .ok_or_else(|| Error::Data(format!("the book has no deal {name:?}")))?;
        d.inception = month;
        Ok(book)
    }

    /// The deals, in order.
    pub fn deals(&self) -> &[DealRun] {
        &self.deals
    }

    /// The deals' names, in order.
    pub fn names(&self) -> Vec<&str> {
        self.deals.iter().map(|d| d.name.as_str()).collect()
    }

    /// The number of simulations.
    pub fn n_sims(&self) -> usize {
        self.deals[0].n_sims()
    }

    /// The deductions' names in the order each first appears, with whether
    /// NMBF leaves it out.
    fn line_names(&self) -> Result<Vec<(String, bool)>> {
        let mut names: Vec<(String, bool)> = Vec::new();
        for d in &self.deals {
            for l in &d.lines {
                match names.iter().find(|(n, _)| *n == l.name) {
                    Some((_, out)) if *out != l.outside_nmbf => {
                        return Err(Error::Data(format!(
                            "the deductions named {:?} disagree on whether NMBF leaves them out",
                            l.name
                        )));
                    }
                    Some(_) => {}
                    None => names.push((l.name.clone(), l.outside_nmbf)),
                }
            }
        }
        Ok(names)
    }

    /// The items of [`amounts`](Self::amounts), in order: `premium`,
    /// `loss`, each deduction, `total_deductions`, `nmbf` and `margin`.
    pub fn items(&self) -> Vec<String> {
        let mut items = vec!["premium".to_string(), "loss".to_string()];
        items.extend(
            self.line_names()
                .expect("checked when built")
                .into_iter()
                .map(|(n, _)| n),
        );
        items.extend(["total_deductions", "nmbf", "margin"].map(String::from));
        items
    }

    /// One item of one deal in each simulation.
    fn item(&self, deal: &DealRun, item: &str) -> Result<Vec<f64>> {
        let n = self.n_sims();
        Ok(match item {
            "premium" => deal.premium.clone(),
            "loss" => deal.loss.clone(),
            "total_deductions" => (0..n)
                .map(|j| deal.lines.iter().map(|l| l.values[j]).sum())
                .collect(),
            "margin" => (0..n).map(|j| deal.margin(j)).collect(),
            "nmbf" => (0..n)
                .map(|j| {
                    deal.margin(j)
                        + deal
                            .lines
                            .iter()
                            .filter(|l| l.outside_nmbf)
                            .map(|l| l.values[j])
                            .sum::<f64>()
                })
                .collect(),
            _ => {
                if !self.line_names()?.iter().any(|(name, _)| name == item) {
                    return Err(Error::Data(format!(
                        "no item {item:?}; the items are {:?}",
                        self.items()
                    )));
                }
                deal.lines
                    .iter()
                    .find(|l| l.name == item)
                    .map_or_else(|| vec![0.0; n], |l| l.values.clone())
            }
        })
    }

    /// One item of the book in each simulation, the deals summed.
    fn total(&self, item: &str) -> Result<Vec<f64>> {
        let mut total = vec![0.0; self.n_sims()];
        for d in &self.deals {
            for (t, x) in total.iter_mut().zip(self.item(d, item)?) {
                *t += x;
            }
        }
        Ok(total)
    }

    /// Every item of every deal and of the book in each simulation, with
    /// dimensions `["item", "deal"]`: the deals in order, then `"total"`,
    /// their sum. A deal without one of the book's deductions has 0 there.
    /// `margin` is the premium less the loss and every deduction, and
    /// `nmbf` adds back those NMBF leaves out.
    ///
    /// Fails if a deal is named `"total"`.
    pub fn amounts(&self) -> Result<PredictiveDistribution> {
        if self.deals.iter().any(|d| d.name == "total") {
            return Err(Error::Data(
                "a deal named \"total\" would clash with the book's total".into(),
            ));
        }
        let items = self.items();
        let mut columns: Vec<(ComponentKey, Vec<f64>)> = Vec::new();
        for item in &items {
            for d in &self.deals {
                columns.push((
                    vec![
                        KeyValue::from(item.as_str()),
                        KeyValue::from(d.name.as_str()),
                    ],
                    self.item(d, item)?,
                ));
            }
            columns.push((
                vec![KeyValue::from(item.as_str()), KeyValue::from("total")],
                self.total(item)?,
            ));
        }
        pd(vec!["item", "deal"], columns, self.n_sims(), "amounts")
    }

    /// The mean of every item by deal and in total, with its ratio to the
    /// mean premium: `(deal, item, mean, ratio)` rows, the deals in order
    /// then `"total"`, the items as [`items`](Self::items). The book's
    /// ratios are sums over sums, the total of an item over the total
    /// premium (equal to the ratio of the means).
    ///
    /// ```
    /// use prospicio_pricing::book::{DealBook, DealRun};
    ///
    /// let a = DealRun::new("a", vec![100.0, 100.0], vec![40.0, 80.0]).unwrap();
    /// let b = DealRun::new("b", vec![300.0, 300.0], vec![150.0, 330.0]).unwrap();
    /// let book = DealBook::new(vec![a, b]).unwrap();
    /// let rows = book.summary().unwrap();
    /// let lr = rows.iter().find(|r| r.0 == "total" && r.1 == "loss").unwrap();
    /// // (60 + 240) / (100 + 300).
    /// assert_eq!(lr.3, 0.75);
    /// ```
    pub fn summary(&self) -> Result<Vec<(String, String, f64, f64)>> {
        let n = self.n_sims() as f64;
        let mean = |x: &[f64]| x.iter().sum::<f64>() / n;
        let mut rows = Vec::new();
        let mut row = |deal: &str, values: &dyn Fn(&str) -> Result<Vec<f64>>| -> Result<()> {
            let premium = mean(&values("premium")?);
            for item in self.items() {
                let m = mean(&values(&item)?);
                rows.push((deal.to_string(), item, m, m / premium));
            }
            Ok(())
        };
        for d in &self.deals {
            row(&d.name, &|item| self.item(d, item))?;
        }
        row("total", &|item| self.total(item))?;
        Ok(rows)
    }

    /// The book's statistics of `item` on `grid`, each deal's value at the
    /// book's ranks and the marginal effect of each deal (the book's
    /// statistic less the book's without it), read on `tail`.
    ///
    /// ```
    /// use prospicio_pricing::book::{DealBook, DealRun, Tail};
    /// use prospicio_prob::statistics::StatisticsGrid;
    ///
    /// let a = DealRun::new("a", vec![10.0; 4], vec![1.0, 2.0, 3.0, 9.0]).unwrap();
    /// let b = DealRun::new("b", vec![10.0; 4], vec![4.0, 1.0, 2.0, 3.0]).unwrap();
    /// let book = DealBook::new(vec![a, b]).unwrap();
    /// let grid = StatisticsGrid::new(vec![]).unwrap().tvar_levels(vec![50.0]).unwrap();
    /// let d = book.distribution("loss", &grid, Tail::Upper).unwrap();
    /// // Book losses 5, 3, 5, 12: the TVaR 50 tail is o[2..] = sims 2 and 3.
    /// assert_eq!(d.total.get("TVaR 50"), Some(8.5));
    /// assert_eq!(d.deals[0].contribution.get("TVaR 50"), Some(6.0));
    /// assert_eq!(d.deals[1].contribution.get("TVaR 50"), Some(2.5));
    /// ```
    pub fn distribution(
        &self,
        item: &str,
        grid: &StatisticsGrid,
        tail: Tail,
    ) -> Result<Distribution> {
        let sign = match tail {
            Tail::Upper => 1.0,
            Tail::Lower => -1.0,
        };
        let flip = |x: Vec<f64>| -> Vec<f64> { x.into_iter().map(|v| sign * v).collect() };
        let back = |s: Statistics| -> Statistics { flip_statistics(s, sign) };
        let total = flip(self.total(item)?);
        let columns: Vec<Vec<f64>> = self
            .deals
            .iter()
            .map(|d| self.item(d, item).map(flip))
            .collect::<Result<_>>()?;
        let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
        let co = grid.co_statistics(&total, &refs, IndexRule::default())?;
        let of = |x: &[f64]| -> Result<Statistics> { Ok(grid.of(&Sampled::new(x.to_vec())?)) };
        let whole = of(&total)?;
        let names = grid.co_names();
        let deals = self
            .deals
            .iter()
            .zip(columns.iter())
            .zip(co)
            .map(|((d, x), mut c)| {
                // The deal's own mean, not its value at the book's mean, so
                // that the means add up too.
                c.values[0] = x.iter().sum::<f64>() / x.len() as f64;
                let without: Vec<f64> = total.iter().zip(x).map(|(t, v)| t - v).collect();
                let rest = of(&without)?;
                let marginal = Statistics {
                    names: names.clone(),
                    values: names
                        .iter()
                        .map(|k| {
                            whole.get(k).expect("a grid cell") - rest.get(k).expect("a grid cell")
                        })
                        .collect(),
                };
                Ok(DealContribution {
                    deal: d.name.clone(),
                    contribution: back(c),
                    marginal: back(marginal),
                })
            })
            .collect::<Result<_>>()?;
        Ok(Distribution {
            item: item.into(),
            total: back(whole),
            deals,
        })
    }

    /// The investors' flows on the book's calendar: each deal's flows
    /// moved to its inception and summed in each simulation, and the
    /// book's IRR, NPV at `rate` (to book month 0) and MOIC on the summed
    /// flows (decision 25), with each deal's mean flows and own IRR.
    ///
    /// Fails if a deal has no flows ([`DealRun::cashflows`] or
    /// [`DealRun::flows`]) or the capital is not positive in a simulation.
    pub fn cashflows(&self, rate: f64, options: &IrrOptions) -> Result<BookCashflows> {
        let flows: Vec<&Flows> = self
            .deals
            .iter()
            .map(|d| {
                d.flows
                    .as_ref()
                    .ok_or_else(|| Error::Data(format!("{:?} has no investor flows", d.name)))
            })
            .collect::<Result<_>>()?;
        let n = self.n_sims();
        let months = self
            .deals
            .iter()
            .zip(&flows)
            .map(|(d, f)| d.inception + f.months)
            .max()
            .expect("a book has a deal");
        let mut total = vec![0.0; n * months];
        let mut mean_by_deal = Vec::with_capacity(self.deals.len());
        let mut irr_by_deal = Vec::with_capacity(self.deals.len());
        for (d, f) in self.deals.iter().zip(&flows) {
            let mut mean = vec![0.0; months];
            for j in 0..n {
                let row = &f.flows[j * f.months..(j + 1) * f.months];
                for (t, &x) in row.iter().enumerate() {
                    total[j * months + d.inception + t] += x;
                    mean[d.inception + t] += x / n as f64;
                }
            }
            mean_by_deal.push(mean);
            let own: Vec<f64> = (0..f.months).map(|t| t as f64).collect();
            irr_by_deal.push(irr_rows(&f.flows, &TimeUnit::Months.years(&own), options)?);
        }
        let offsets: Vec<f64> = (0..months).map(|t| t as f64).collect();
        let years = TimeUnit::Months.years(&offsets);
        let irr = irr_rows(&total, &years, options)?;
        let npv = npv_rows(&total, &years, rate)?;
        let moic = (0..n)
            .map(|j| {
                let net: f64 = total[j * months..(j + 1) * months].iter().sum();
                let funded: f64 = flows.iter().map(|f| f.funded[j]).sum();
                let opening: f64 = flows.iter().map(|f| f.opening[j]).sum();
                moic(net, funded, opening)
            })
            .collect::<Result<_>>()?;
        Ok(BookCashflows {
            months,
            deals: self.deals.iter().map(|d| d.name.clone()).collect(),
            flows: total,
            mean_by_deal,
            irr,
            npv,
            moic,
            irr_by_deal,
        })
    }

    /// The book by calendar year, with dimensions `["item", "deal",
    /// "year"]`: `written_premium`, `earned_premium` and `paid_loss` from
    /// each deal's patterns (the premium and loss of each simulation spread
    /// by the month-by-month increments, placed at the deal's inception
    /// plus the month), and `investor_flow` when every deal has flows; the
    /// deals in order, then `"total"`; the years from `start_year`, book
    /// month `m` in year `start_year + m / 12`.
    ///
    /// A pattern that ends below 1, a commuted paid pattern say, leaves
    /// the rest of the premium or loss unplaced, so a year's total can
    /// fall short of the deal's amount.
    ///
    /// Fails if a deal has no patterns.
    pub fn calendar(&self, start_year: i64) -> Result<PredictiveDistribution> {
        let patterns: Vec<&MonthlyPatterns> = self
            .deals
            .iter()
            .map(|d| {
                d.patterns
                    .as_ref()
                    .ok_or_else(|| Error::Data(format!("{:?} has no monthly patterns", d.name)))
            })
            .collect::<Result<_>>()?;
        let with_flows = self.deals.iter().all(|d| d.flows.is_some());
        let end = self
            .deals
            .iter()
            .zip(&patterns)
            .map(|(d, p)| {
                let f = d.flows.as_ref().map_or(0, |f| f.months);
                d.inception + p.written.len().max(f)
            })
            .max()
            .expect("a book has a deal");
        let n_years = end.div_ceil(12);
        let n = self.n_sims();
        let mut items: Vec<&str> = vec!["written_premium", "earned_premium", "paid_loss"];
        if with_flows {
            items.push("investor_flow");
        }
        // By item, then deal (the total last), then year: n values each.
        let n_deals = self.deals.len() + 1;
        let mut cells = vec![vec![0.0; n]; items.len() * n_deals * n_years];
        let at = |item: usize, deal: usize, year: usize| (item * n_deals + deal) * n_years + year;
        for (k, (d, p)) in self.deals.iter().zip(&patterns).enumerate() {
            let mut place = |item: usize, cumulative: &[f64], amount: &[f64]| {
                let mut before = 0.0;
                for (t, &c) in cumulative.iter().enumerate() {
                    let step = c - before;
                    before = c;
                    let y = (d.inception + t) / 12;
                    for (j, &a) in amount.iter().enumerate() {
                        let v = step * a;
                        cells[at(item, k, y)][j] += v;
                        cells[at(item, n_deals - 1, y)][j] += v;
                    }
                }
            };
            place(0, &p.written, &d.premium);
            place(1, &p.earned, &d.premium);
            place(2, &p.paid, &d.loss);
            if let Some(f) = &d.flows {
                for (j, row) in f.flows.chunks(f.months).enumerate() {
                    for (t, &v) in row.iter().enumerate() {
                        let y = (d.inception + t) / 12;
                        cells[at(3, k, y)][j] += v;
                        cells[at(3, n_deals - 1, y)][j] += v;
                    }
                }
            }
        }
        let mut deal_names: Vec<&str> = self.names();
        deal_names.push("total");
        let mut columns = Vec::with_capacity(cells.len());
        for (i, item) in items.iter().enumerate() {
            for (k, deal) in deal_names.iter().enumerate() {
                for y in 0..n_years {
                    columns.push((
                        vec![
                            KeyValue::from(*item),
                            KeyValue::from(*deal),
                            KeyValue::from(start_year + y as i64),
                        ],
                        std::mem::take(&mut cells[at(i, k, y)]),
                    ));
                }
            }
        }
        pd(vec!["item", "deal", "year"], columns, n, "calendar")
    }
}

fn flip_statistics(s: Statistics, sign: f64) -> Statistics {
    if sign > 0.0 {
        return s;
    }
    // The grid read the item's negative: every value turns back but the
    // spreads, the negative's minimum is the item's maximum, and its
    // mean + k SD the item's mean - k SD.
    let names = s
        .names
        .iter()
        .map(|n| match n.as_str() {
            "Min" => "Max".to_string(),
            "Max" => "Min".to_string(),
            _ => match n.strip_prefix("Mean + ") {
                Some(rest) => format!("Mean - {rest}"),
                None => match n.strip_prefix("Mean - ") {
                    Some(rest) => format!("Mean + {rest}"),
                    None => n.clone(),
                },
            },
        })
        .collect();
    let values = s
        .names
        .iter()
        .zip(s.values)
        .map(|(n, v)| if n == "SD" || n == "Variance" { v } else { -v })
        .collect();
    Statistics { names, values }
}

/// A predictive distribution from columns, simulation-major.
fn pd(
    dims: Vec<&str>,
    columns: Vec<(ComponentKey, Vec<f64>)>,
    n: usize,
    what: &str,
) -> Result<PredictiveDistribution> {
    let mut draws = Vec::with_capacity(columns.len() * n);
    for j in 0..n {
        draws.extend(columns.iter().map(|(_, c)| c[j]));
    }
    PredictiveDistribution::from_draws(
        dims.into_iter().map(String::from).collect(),
        columns.into_iter().map(|(k, _)| k).collect(),
        draws,
        Provenance::new("prospicio_pricing::book").param("table", what),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cashflow::{Base, Requirement, Term};
    use prospicio_aggregate::{Deduction, Layer};
    use prospicio_prob::Empirical;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    /// Written over 12 months, earned over 24 and paid over 36, the whole
    /// premium and loss inside the 48 months.
    fn patterns() -> MonthlyPatterns {
        let ramp = |m: f64| (0..48).map(|t| ((t + 1) as f64 / m).min(1.0)).collect();
        MonthlyPatterns::new(ramp(12.0), ramp(24.0), ramp(36.0)).unwrap()
    }

    fn terms() -> CashflowTerms {
        let r = Requirement::new(vec![
            Term::new(Base::UnearnedPremium, 0.3),
            Term::new(Base::Reserve, 1.2),
        ]);
        let mut t = CashflowTerms::new(100.0, 40, patterns(), r).unwrap();
        t.operating.opening_fixed = 20.0;
        t
    }

    fn ulr() -> Vec<f64> {
        vec![0.4, 0.7, 1.3, 0.55, 0.9, 2.0]
    }

    /// A deal of premium 100 on the loss ratios of [`ulr`], with its
    /// investor's flows.
    fn run(name: &str, scale: f64) -> DealRun {
        let u: Vec<f64> = ulr().iter().map(|x| x * scale).collect();
        let t = terms();
        let cf = t.run(&u, &vec![0.25; u.len()]).unwrap();
        DealRun::new(
            name,
            vec![100.0; u.len()],
            u.iter().map(|x| 100.0 * x).collect(),
        )
        .unwrap()
        .deduction("ceding_commission", vec![25.0; u.len()], false)
        .unwrap()
        .cashflows(&t, &cf)
        .unwrap()
    }

    #[test]
    fn a_book_of_one_layer_reproduces_its_deal_ratios() {
        let layer = Layer::stop_loss("cover", 120.0, 0.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .sliding_scale(vec![(0.3, 0.5), (0.1, 0.9)])
            .unwrap()
            .brokerage(0.025)
            .unwrap()
            .deduction(
                Deduction::premium("manager_fee", 0.05)
                    .unwrap()
                    .outside_nmbf(),
            )
            .unwrap();
        let tower = Tower::new(vec![layer]).unwrap();
        let result = tower.apply_annual(&[40.0, 70.0, 150.0]).unwrap();
        let ratios = tower.deal_ratios(&result, "cover").unwrap();
        let book = DealBook::new(vec![
            DealRun::from_tower("cover", &tower, &result, "cover").unwrap(),
        ])
        .unwrap();
        let amounts = book.amounts().unwrap();
        for (ratio, item) in [
            ("loss_ratio", "loss"),
            ("ceding_commission", "ceding_commission"),
            ("brokerage", "brokerage"),
            ("manager_fee", "manager_fee"),
            ("total_deductions", "total_deductions"),
            ("nmbf", "nmbf"),
            ("margin", "margin"),
        ] {
            let r = ratios.marginal(&vec![KeyValue::from(ratio)]).unwrap();
            let a = amounts
                .marginal(&vec![KeyValue::from(item), KeyValue::from("total")])
                .unwrap();
            for (x, y) in r.draws().iter().zip(a.draws()) {
                assert!(close(x * 100.0, *y, 1e-14), "{item}: {x} {y}");
            }
        }
    }

    #[test]
    fn a_run_from_a_tower_takes_the_reinstatement_premium() {
        use prospicio_aggregate::EventSet;
        // 10 xs 10 at premium 2, the first reinstatement at 100% and the
        // second at 50%, beside a quota share with no reinstatements.
        let cat = Layer::xol("cat", 10.0, 10.0)
            .unwrap()
            .paid_reinstatements(2.0, vec![1.0, 0.5])
            .unwrap();
        let qs = Layer::quota_share("qs", 0.5)
            .unwrap()
            .premium_rate(1.0, 40.0)
            .unwrap();
        let years = vec![
            vec![5.0],
            vec![15.0],
            vec![22.0, 12.0],
            vec![40.0, 40.0, 40.0],
        ];
        let events = EventSet::from_years(years, 1).unwrap();
        let tower = Tower::new(vec![cat.clone(), qs]).unwrap();
        let result = tower.apply(&events).unwrap();
        let run = DealRun::from_tower("cat", &tower, &result, "cat").unwrap();
        let rip = result
            .aggregate(&["kind", "layer"])
            .unwrap()
            .marginal(&vec![
                KeyValue::from("reinstatement_premium"),
                KeyValue::from("cat"),
            ])
            .unwrap();
        // Each year's premium is the deposit plus the tower's own
        // reinstatement premium, so the mean is the expected one.
        for (j, (p, r)) in run.premium().iter().zip(rip.draws()).enumerate() {
            assert!(close(*p, 2.0 + r, 1e-15), "{j}: {p} {r}");
        }
        let by_layer: Vec<f64> = [&[5.0][..], &[15.0], &[22.0, 12.0], &[40.0, 40.0, 40.0]]
            .iter()
            .map(|x| 2.0 + cat.reinstatement_premium(x))
            .collect();
        assert_eq!(run.premium(), by_layer);
        assert_eq!(run.premium(), [2.0, 3.0, 4.2, 5.0]);
        let deposit = DealRun::from_tower_with("cat", &tower, &result, "cat", false).unwrap();
        assert_eq!(deposit.premium(), [2.0; 4]);
        assert_eq!(deposit.loss(), run.loss());
        // A layer without paid reinstatements is unchanged.
        let qs = DealRun::from_tower("qs", &tower, &result, "qs").unwrap();
        assert_eq!(qs.premium(), [20.0; 4]);
        assert_eq!(
            qs,
            DealRun::from_tower_with("qs", &tower, &result, "qs", false).unwrap()
        );
        // The book's margin counts it.
        let book = DealBook::new(vec![run]).unwrap();
        let margin = book
            .amounts()
            .unwrap()
            .marginal(&vec![KeyValue::from("margin"), KeyValue::from("cat")])
            .unwrap();
        let ceded = [0.0, 5.0, 12.0, 30.0];
        for ((m, p), c) in margin.draws().iter().zip([2.0, 3.0, 4.2, 5.0]).zip(ceded) {
            assert!(close(*m, p - c, 1e-15), "{m} {p} {c}");
        }
    }

    #[test]
    fn the_book_adds_its_deals_and_its_ratios_are_sums_over_sums() {
        let a = DealRun::new("a", vec![100.0, 120.0], vec![40.0, 150.0])
            .unwrap()
            .deduction("brokerage", vec![2.5, 3.0], false)
            .unwrap();
        let b = DealRun::new("b", vec![300.0, 280.0], vec![150.0, 100.0])
            .unwrap()
            .deduction("fee", vec![15.0, 14.0], true)
            .unwrap();
        let book = DealBook::new(vec![a, b]).unwrap();
        assert_eq!(
            book.items(),
            [
                "premium",
                "loss",
                "brokerage",
                "fee",
                "total_deductions",
                "nmbf",
                "margin"
            ]
        );
        let amounts = book.amounts().unwrap();
        let at = |item: &str, deal: &str| {
            amounts
                .marginal(&vec![KeyValue::from(item), KeyValue::from(deal)])
                .unwrap()
                .draws()
                .to_vec()
        };
        assert_eq!(at("brokerage", "b"), [0.0, 0.0]);
        assert_eq!(
            at("margin", "total"),
            [400.0 - 190.0 - 17.5, 400.0 - 250.0 - 17.0]
        );
        assert_eq!(
            at("nmbf", "total"),
            [400.0 - 190.0 - 2.5, 400.0 - 250.0 - 3.0]
        );
        let rows = book.summary().unwrap();
        let lr = rows
            .iter()
            .find(|r| r.0 == "total" && r.1 == "loss")
            .unwrap();
        assert!(close(lr.3, 440.0 / 800.0, 1e-15));
        assert_eq!(rows.len(), 3 * 7);
    }

    #[test]
    fn contributions_add_up_and_the_lower_tail_reads_the_margin() {
        let a = DealRun::new("a", vec![10.0; 5], vec![1.0, 2.0, 3.0, 9.0, 4.0]).unwrap();
        let b = DealRun::new("b", vec![10.0; 5], vec![4.0, 1.0, 2.0, 3.0, 8.0]).unwrap();
        let book = DealBook::new(vec![a, b]).unwrap();
        let grid = StatisticsGrid::new(vec![50.0, 80.0])
            .unwrap()
            .tvar_levels(vec![60.0])
            .unwrap();
        let d = book.distribution("loss", &grid, Tail::Upper).unwrap();
        for cell in grid.co_names() {
            let sum: f64 = d
                .deals
                .iter()
                .map(|c| c.contribution.get(&cell).unwrap())
                .sum();
            assert!(close(sum, d.total.get(&cell).unwrap(), 1e-14), "{cell}");
        }
        // Book losses 5, 3, 5, 12, 12; without b: 1, 2, 3, 9, 4.
        assert!(close(d.total.get("TVaR 60").unwrap(), 12.0, 1e-14));
        assert!(close(
            d.deals[1].marginal.get("TVaR 60").unwrap(),
            12.0 - 6.5,
            1e-14
        ));
        // The margin's lower tail is the loss's upper tail turned around.
        let m = book.distribution("margin", &grid, Tail::Lower).unwrap();
        assert!(close(m.total.get("TVaR 60").unwrap(), 20.0 - 12.0, 1e-14));
        assert_eq!(m.total.get("Max"), Some(17.0));
        assert_eq!(m.total.get("Min"), Some(8.0));
        assert!(close(
            m.deals[0].contribution.get("TVaR 60").unwrap(),
            10.0 - d.deals[0].contribution.get("TVaR 60").unwrap(),
            1e-15
        ));
    }

    #[test]
    fn one_deal_keeps_its_returns_and_a_shift_discounts_its_npv() {
        let deal = run("a", 1.0);
        let t = terms();
        let own = t.run(&ulr(), &[0.25; 6]).unwrap();
        let options = IrrOptions::default();
        let book = DealBook::new(vec![deal]).unwrap();
        let cf = book.cashflows(0.05, &options).unwrap();
        assert_eq!(cf.flows, own.flows());
        let same = |a: &[Irr], b: &[Irr]| {
            a.iter().zip(b).all(|(x, y)| {
                x.found == y.found
                    && (!x.found || x.rate == y.rate)
                    && x.sign_changes == y.sign_changes
            })
        };
        assert!(same(&cf.irr, &own.irr(&options).unwrap()));
        assert_eq!(cf.npv, own.npv(0.05).unwrap());
        for j in 0..6 {
            let net: f64 = own.flows()[j * own.months()..(j + 1) * own.months()]
                .iter()
                .sum();
            let want = 1.0 + net / (own.funded()[j] + own.opening()[j]);
            assert!(close(cf.moic[j], want, 1e-15));
        }
        let later = book
            .shifted("a", 7)
            .unwrap()
            .cashflows(0.05, &options)
            .unwrap();
        assert_eq!(later.months, cf.months + 7);
        for j in 0..6 {
            assert!(close(
                later.npv[j],
                cf.npv[j] * 1.05f64.powf(-7.0 / 12.0),
                1e-12
            ));
            if cf.irr[j].found {
                assert!(close(later.irr[j].rate, cf.irr[j].rate, 1e-9));
            }
        }
        assert!(same(&later.irr_by_deal[0], &cf.irr_by_deal[0]));
    }

    #[test]
    fn the_book_irr_is_on_the_summed_flows() {
        let book = DealBook::new(vec![run("a", 1.0), run("b", 0.8).inception(6)]).unwrap();
        let options = IrrOptions::default();
        let cf = book.cashflows(0.05, &options).unwrap();
        let a = run("a", 1.0);
        let b = run("b", 0.8);
        let (fa, fb) = (a.flows.as_ref().unwrap(), b.flows.as_ref().unwrap());
        assert_eq!(cf.months, fb.months + 6);
        for j in 0..6 {
            let mut sum = vec![0.0; cf.months];
            for (s, x) in sum
                .iter_mut()
                .zip(&fa.flows[j * fa.months..(j + 1) * fa.months])
            {
                *s += x;
            }
            for (s, x) in sum[6..]
                .iter_mut()
                .zip(&fb.flows[j * fb.months..(j + 1) * fb.months])
            {
                *s += x;
            }
            assert_eq!(
                &cf.flows[j * cf.months..(j + 1) * cf.months],
                sum.as_slice()
            );
        }
        let mean_total: f64 = cf.mean_by_deal.iter().flatten().sum();
        let all: f64 = cf.flows.iter().sum::<f64>() / 6.0;
        assert!(close(mean_total, all, 1e-12));
    }

    #[test]
    fn the_calendar_places_each_deal_from_its_inception() {
        let book = DealBook::new(vec![run("a", 1.0), run("b", 0.8).inception(6)]).unwrap();
        let cal = book.calendar(2026).unwrap();
        let at = |item: &str, deal: &str, year: i64| {
            cal.marginal(&vec![
                KeyValue::from(item),
                KeyValue::from(deal),
                KeyValue::from(year),
            ])
            .unwrap()
            .draws()
            .to_vec()
        };
        // a writes everything in 2026; b, from 1 July, half in each year.
        assert!(
            at("written_premium", "a", 2026)
                .iter()
                .all(|&x| close(x, 100.0, 1e-13))
        );
        assert!(
            at("written_premium", "b", 2026)
                .iter()
                .all(|&x| close(x, 50.0, 1e-13))
        );
        assert!(
            at("written_premium", "b", 2027)
                .iter()
                .all(|&x| close(x, 50.0, 1e-13))
        );
        // Each deal's paid loss over the years is its loss.
        let years = 2026..2026 + (6 + 48_i64).div_euclid(12) + 1;
        let paid: Vec<f64> = (0..6)
            .map(|j| years.clone().map(|y| at("paid_loss", "b", y)[j]).sum())
            .collect();
        for (p, u) in paid.iter().zip(ulr()) {
            assert!(close(*p, 80.0 * u, 1e-12));
        }
        // The investors' flows by year add to their flows.
        let cf = book.cashflows(0.0, &IrrOptions::default()).unwrap();
        for j in 0..6 {
            let by_year: f64 = years
                .clone()
                .map(|y| at("investor_flow", "total", y)[j])
                .sum();
            let all: f64 = cf.flows[j * cf.months..(j + 1) * cf.months].iter().sum();
            assert!(close(by_year, all, 1e-12));
        }
    }

    #[test]
    fn a_book_checks_its_deals() {
        let a = DealRun::new("a", vec![1.0; 3], vec![0.5; 3]).unwrap();
        let short = DealRun::new("b", vec![1.0; 2], vec![0.5; 2]).unwrap();
        assert!(DealBook::new(vec![]).is_err());
        assert!(DealBook::new(vec![a.clone(), short]).is_err());
        assert!(DealBook::new(vec![a.clone(), a.clone()]).is_err());
        let fee_in = a.clone().deduction("fee", vec![0.1; 3], false).unwrap();
        let fee_out = DealRun::new("c", vec![1.0; 3], vec![0.5; 3])
            .unwrap()
            .deduction("fee", vec![0.1; 3], true)
            .unwrap();
        assert!(DealBook::new(vec![fee_in, fee_out]).is_err());
        assert!(a.clone().deduction("margin", vec![0.0; 3], false).is_err());
        assert!(a.clone().deduction("x", vec![0.0; 2], false).is_err());
        assert!(DealRun::new("d", vec![1.0], vec![f64::NAN]).is_err());
        let b = DealRun::new("b", vec![2.0; 3], vec![1.0; 3]).unwrap();
        let book = DealBook::new(vec![a, b]).unwrap();
        assert_eq!(book.select(&["b"]).unwrap().names(), ["b"]);
        assert!(book.select(&["z"]).is_err());
        assert!(book.select(&["a", "a"]).is_err());
        assert!(book.shifted("z", 3).is_err());
        assert!(book.cashflows(0.0, &IrrOptions::default()).is_err());
        assert!(book.calendar(2026).is_err());
        assert!(
            book.distribution(
                "nothing",
                &StatisticsGrid::new(vec![]).unwrap(),
                Tail::Upper
            )
            .is_err()
        );
        let total = DealRun::new("total", vec![1.0; 3], vec![0.5; 3]).unwrap();
        assert!(DealBook::new(vec![total]).unwrap().amounts().is_err());
    }
}
