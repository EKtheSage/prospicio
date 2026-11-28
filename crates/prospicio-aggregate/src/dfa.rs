//! A multi-year projection of one company along economic scenarios, and
//! finite-time ruin and capital breach on its surplus paths
//! (`docs/design/esg-dfa.md`, PR 3).
//!
//! [`Projection`] rolls the company forward a year at a time on each path
//! of a [`Scenarios`] set: premium and expenses at the start of the year,
//! the year's new losses (simulated, in the projection's base money)
//! scaled by the path's claims index, their payout and the opening
//! reserves' run-off, investment income on cash at the short rate and on
//! bond buckets by duration, and the closing assets, reserves and surplus.
//! The result is a [`ProjectionPaths`]: each [`Line`] by path and year,
//! [`ruin`](ProjectionPaths::ruin) and [`breach`](ProjectionPaths::breach)
//! with their binomial standard errors.
//!
//! The accounts, at each year end `t` with claims index `I_t` (`I_0 = 1`):
//!
//! * the reserves are the payments still due, in base money, at `I_t`: the
//!   money of the year end, with no allowance for later inflation;
//! * incurred losses are the year's new losses at `I_t`, and reserve
//!   development is the prior reserves' revaluation, `R_(t-1) (I_t /
//!   I_(t-1) - 1)`;
//! * payments are made at the year end, at `I_t`;
//! * the assets after premium and expenses earn the year's return:
//!   `exp(∫ r) - 1` on cash, and `e^y - 1 - D Δy` on a bucket of duration
//!   `D`, its continuously compounded yield `y` read from the path's curve
//!   at the start and the end of the year (decision 8, first order), the
//!   buckets rebalanced to their shares each year;
//! * the result is premium less expenses, incurred losses and development,
//!   plus investment income, and the surplus, assets less reserves, moves by
//!   exactly the result (validation 6).
//!
//! Ruin and breach are checked at year ends only, which understates ruin
//! within a year. The underwriting cycle, tax and dividends or injections
//! come with the note's PR 4.

use prospicio_core::{Error, Result};
use prospicio_esg::{Index, Scenarios};
use prospicio_prob::{ComponentKey, PredictiveDistribution, Provenance};

fn data(why: impl Into<String>) -> Error {
    Error::Data(why.into())
}

/// A bond bucket: a share of the assets held at a duration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BondBucket {
    /// The share of the assets, after premium and expenses, at each year's
    /// start.
    pub share: f64,
    /// The duration in years, one of the scenarios' reported maturities.
    pub duration: f64,
}

/// A company's projection: its opening balance sheet, the new business of
/// each year and the investment rule.
///
/// ```
/// use prospicio_aggregate::dfa::{Line, Projection};
///
/// // Two paths of three years of losses, no economy: premium 100 a year,
/// // 30% expenses, losses paid half in their year and half the next.
/// let losses = vec![60.0, 70.0, 80.0, 50.0, 120.0, 40.0];
/// let projection = Projection::new(3, 200.0, losses, 2)?
///     .with_premiums(vec![100.0; 3])
///     .with_expense_ratio(0.3)
///     .with_payout(vec![0.5, 0.5]);
/// let paths = projection.run_without_economy()?;
/// // Path 2's year 2: 100 - 30 - 120 = -50.
/// assert!((paths.value(Line::Result, 1, 2)? + 50.0).abs() < 1e-12);
/// // Its surplus: 200 + 20 - 50 + 30.
/// assert!((paths.value(Line::Surplus, 1, 3)? - 200.0).abs() < 1e-12);
/// assert_eq!(paths.ruin(3)?.probability, 0.0);
/// # Ok::<(), prospicio_core::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Projection {
    /// The years projected.
    pub years: usize,
    /// The assets today.
    pub opening_assets: f64,
    /// The opening reserves' payments in base money, at the end of each
    /// year from the first; their sum is the opening reserve.
    pub opening_reserve_payments: Vec<f64>,
    /// The premium received at the start of each year.
    pub premiums: Vec<f64>,
    /// Expenses as a share of the year's premium, paid with it.
    pub expense_ratio: f64,
    /// The new losses of each year in base money, simulation-major:
    /// `losses[s * years + t]` is year `t + 1` of path `s`.
    pub losses: Vec<f64>,
    /// The number of paths.
    pub n_sims: usize,
    /// The share of a year's losses paid at the end of that year and of
    /// each later one; sums to 1.
    pub payout: Vec<f64>,
    /// The bond buckets; the rest of the assets is cash.
    pub bonds: Vec<BondBucket>,
    /// The index that scales the losses and reserves.
    pub index: Index,
}

impl Projection {
    /// `years` years from `opening_assets`, with `n_sims` paths of new
    /// losses (simulation-major, `years` a path); no premium, expenses,
    /// opening reserves or bonds, losses paid in their year, scaled by the
    /// claims index.
    ///
    /// Fails unless `years` and `n_sims` are positive, `losses` has
    /// `years · n_sims` finite values and `opening_assets` is finite.
    pub fn new(years: usize, opening_assets: f64, losses: Vec<f64>, n_sims: usize) -> Result<Self> {
        if years == 0 || n_sims == 0 {
            return Err(data(format!(
                "a projection needs years and paths, got {years} and {n_sims}"
            )));
        }
        if losses.len() != years * n_sims || losses.iter().any(|x| !x.is_finite()) {
            return Err(data(format!(
                "the losses must be {} finite values ({n_sims} paths of {years} years), got {}",
                years * n_sims,
                losses.len()
            )));
        }
        if !opening_assets.is_finite() {
            return Err(data("the opening assets are not finite"));
        }
        Ok(Self {
            years,
            opening_assets,
            opening_reserve_payments: Vec::new(),
            premiums: vec![0.0; years],
            expense_ratio: 0.0,
            losses,
            n_sims,
            payout: vec![1.0],
            bonds: Vec::new(),
            index: Index::Claims,
        })
    }

    /// The new losses from a distribution with one component per year, in
    /// order, as from an annual simulation.
    pub fn from_distribution(opening_assets: f64, losses: &PredictiveDistribution) -> Result<Self> {
        Self::new(
            losses.n_components(),
            opening_assets,
            losses.draw_matrix().to_vec(),
            losses.n_sims(),
        )
    }

    /// With the opening reserves' payments by year, in base money.
    pub fn with_opening_reserves(mut self, payments: Vec<f64>) -> Self {
        self.opening_reserve_payments = payments;
        self
    }

    /// With the premium of each year.
    pub fn with_premiums(mut self, premiums: Vec<f64>) -> Self {
        self.premiums = premiums;
        self
    }

    /// With expenses as a share of premium.
    pub fn with_expense_ratio(mut self, ratio: f64) -> Self {
        self.expense_ratio = ratio;
        self
    }

    /// With the losses' payout pattern.
    pub fn with_payout(mut self, payout: Vec<f64>) -> Self {
        self.payout = payout;
        self
    }

    /// With bond buckets.
    pub fn with_bonds(mut self, bonds: Vec<BondBucket>) -> Self {
        self.bonds = bonds;
        self
    }

    /// With the losses scaled by this index.
    pub fn with_index(mut self, index: Index) -> Self {
        self.index = index;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.premiums.len() != self.years || self.premiums.iter().any(|p| !p.is_finite()) {
            return Err(data(format!(
                "the premiums must be {} finite values, one a year",
                self.years
            )));
        }
        if !self.expense_ratio.is_finite() {
            return Err(data("the expense ratio is not finite"));
        }
        if self.opening_reserve_payments.iter().any(|p| !p.is_finite()) {
            return Err(data("the opening reserves' payments are not finite"));
        }
        let total: f64 = self.payout.iter().sum();
        if self.payout.is_empty()
            || self.payout.iter().any(|p| !(p.is_finite() && *p >= 0.0))
            || (total - 1.0).abs() > 1e-9
        {
            return Err(data("the payout must be non-negative shares that sum to 1"));
        }
        let shares: f64 = self.bonds.iter().map(|b| b.share).sum();
        if self.bonds.iter().any(|b| {
            !(b.share.is_finite() && b.share >= 0.0 && b.duration.is_finite() && b.duration > 0.0)
        }) || shares > 1.0 + 1e-12
        {
            return Err(data(
                "the bond buckets need non-negative shares summing to at most 1 and positive durations",
            ));
        }
        Ok(())
    }

    /// The projection along `scenarios`, path `s` on scenario path `s`.
    ///
    /// Fails when the scenarios have another number of paths, fewer years,
    /// a bucket's duration not among their maturities, or a step that does
    /// not fall on each year end; and on the checks of [`new`](Self::new)
    /// and the builders.
    pub fn run(&self, scenarios: &Scenarios) -> Result<ProjectionPaths> {
        self.validate()?;
        if scenarios.n_sims() != self.n_sims {
            return Err(data(format!(
                "the scenarios have {} paths, the losses {}",
                scenarios.n_sims(),
                self.n_sims
            )));
        }
        let per_year = scenarios.step().per_year();
        if scenarios.n_steps() < self.years * per_year {
            return Err(data(format!(
                "the scenarios run {} years, the projection {}",
                scenarios.n_steps() / per_year,
                self.years
            )));
        }
        let buckets: Vec<(usize, f64, f64)> = self
            .bonds
            .iter()
            .map(|b| {
                scenarios
                    .maturities()
                    .iter()
                    .position(|&m| (m - b.duration).abs() < 1e-12)
                    .map(|j| (j, b.share, b.duration))
                    .ok_or_else(|| {
                        data(format!(
                            "a bucket of duration {} is not among the scenarios' maturities",
                            b.duration
                        ))
                    })
            })
            .collect::<Result<_>>()?;
        // Without an inflation model in the scenarios the index stays at 1.
        let index_of = |s: usize, t: usize| {
            scenarios
                .index_level(self.index, s, t * per_year)
                .unwrap_or(1.0)
        };
        let mut economy = Economy::new(self.n_sims, self.years);
        for s in 0..self.n_sims {
            for t in 1..=self.years {
                let (k0, k1) = ((t - 1) * per_year, t * per_year);
                let cash = scenarios.deflator(s, k0)? / scenarios.deflator(s, k1)? - 1.0;
                let mut ret = cash * (1.0 - buckets.iter().map(|b| b.1).sum::<f64>());
                for &(j, share, d) in &buckets {
                    let y0 = -scenarios.bond_price(s, k0, j)?.ln() / d;
                    let y1 = -scenarios.bond_price(s, k1, j)?.ln() / d;
                    ret += share * (y0.exp_m1() - d * (y1 - y0));
                }
                economy.set(s, t, ret, index_of(s, t));
            }
        }
        Ok(self.roll(&economy))
    }

    /// The projection with no economy: no investment return and an index of
    /// 1 throughout (validation 5's setting).
    pub fn run_without_economy(&self) -> Result<ProjectionPaths> {
        self.validate()?;
        if !self.bonds.is_empty() {
            return Err(data("bond buckets need scenarios"));
        }
        Ok(self.roll(&Economy::new(self.n_sims, self.years)))
    }

    fn roll(&self, economy: &Economy) -> ProjectionPaths {
        let (n, years) = (self.n_sims, self.years);
        let mut lines = vec![vec![0.0; n * years]; Line::ALL.len()];
        let horizon = years + self.payout.len() + self.opening_reserve_payments.len();
        for s in 0..n {
            // Payments still due, in base money, by year from the first.
            let mut due = vec![0.0; horizon + 1];
            for (k, p) in self.opening_reserve_payments.iter().enumerate() {
                due[k + 1] += p;
            }
            let mut assets = self.opening_assets;
            let mut index = 1.0;
            for t in 1..=years {
                let premium = self.premiums[t - 1];
                let expenses = self.expense_ratio * premium;
                let invested = assets + premium - expenses;
                let investment = invested * economy.ret(s, t);
                let level = economy.index(s, t);
                let prior: f64 = due[t..].iter().sum();
                let development = prior * (level - index);
                let new = self.losses[s * years + t - 1];
                for (k, share) in self.payout.iter().enumerate() {
                    due[t + k] += new * share;
                }
                let incurred = new * level;
                let paid = due[t] * level;
                due[t] = 0.0;
                let reserves = due[t + 1..].iter().sum::<f64>() * level;
                assets = invested + investment - paid;
                let result = premium - expenses - incurred - development + investment;
                let i = s * years + t - 1;
                for (line, value) in [
                    (Line::Premium, premium),
                    (Line::Expenses, expenses),
                    (Line::IncurredLosses, incurred),
                    (Line::ReserveDevelopment, development),
                    (Line::InvestmentIncome, investment),
                    (Line::Result, result),
                    (Line::Paid, paid),
                    (Line::Assets, assets),
                    (Line::Reserves, reserves),
                    (Line::Surplus, assets - reserves),
                ] {
                    lines[line as usize][i] = value;
                }
                index = level;
            }
        }
        let opening_reserves: f64 = self.opening_reserve_payments.iter().sum();
        ProjectionPaths {
            n_sims: n,
            years,
            opening_surplus: self.opening_assets - opening_reserves,
            lines,
        }
    }
}

/// Each path's investment return and index by year.
struct Economy {
    years: usize,
    ret: Vec<f64>,
    index: Vec<f64>,
}

impl Economy {
    fn new(n: usize, years: usize) -> Self {
        Self {
            years,
            ret: vec![0.0; n * years],
            index: vec![1.0; n * years],
        }
    }

    fn set(&mut self, s: usize, t: usize, ret: f64, index: f64) {
        self.ret[s * self.years + t - 1] = ret;
        self.index[s * self.years + t - 1] = index;
    }

    fn ret(&self, s: usize, t: usize) -> f64 {
        self.ret[s * self.years + t - 1]
    }

    fn index(&self, s: usize, t: usize) -> f64 {
        self.index[s * self.years + t - 1]
    }
}

/// A line of the projection, by path and year.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Line {
    /// Premium received.
    Premium,
    /// Expenses paid.
    Expenses,
    /// The year's new losses at the year end's index.
    IncurredLosses,
    /// The prior reserves' revaluation by the index.
    ReserveDevelopment,
    /// Investment income.
    InvestmentIncome,
    /// The year's result: premium less expenses, incurred losses and
    /// development, plus investment income.
    Result,
    /// Losses paid.
    Paid,
    /// Assets at the year end.
    Assets,
    /// Reserves at the year end.
    Reserves,
    /// Surplus at the year end, assets less reserves.
    Surplus,
}

impl Line {
    /// Every line, in order.
    pub const ALL: [Line; 10] = [
        Line::Premium,
        Line::Expenses,
        Line::IncurredLosses,
        Line::ReserveDevelopment,
        Line::InvestmentIncome,
        Line::Result,
        Line::Paid,
        Line::Assets,
        Line::Reserves,
        Line::Surplus,
    ];

    fn model(self) -> &'static str {
        match self {
            Line::Premium => "dfa_premium",
            Line::Expenses => "dfa_expenses",
            Line::IncurredLosses => "dfa_incurred_losses",
            Line::ReserveDevelopment => "dfa_reserve_development",
            Line::InvestmentIncome => "dfa_investment_income",
            Line::Result => "dfa_result",
            Line::Paid => "dfa_paid",
            Line::Assets => "dfa_assets",
            Line::Reserves => "dfa_reserves",
            Line::Surplus => "dfa_surplus",
        }
    }
}

/// A probability estimated over the paths, with its binomial standard
/// error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// The share of paths.
    pub probability: f64,
    /// `√(p (1 - p) / n)`.
    pub standard_error: f64,
}

impl Estimate {
    fn of(hits: usize, n: usize) -> Self {
        let p = hits as f64 / n as f64;
        Self {
            probability: p,
            standard_error: (p * (1.0 - p) / n as f64).sqrt(),
        }
    }
}

/// The required capital in a breach test, at each year end.
#[derive(Debug, Clone, PartialEq)]
pub enum RequiredCapital {
    /// The same amount every year.
    Fixed(f64),
    /// An amount for each year, from the first.
    ByYear(Vec<f64>),
    /// A ratio of the year's premium.
    PremiumRatio(f64),
    /// A ratio of the year-end reserves.
    ReserveRatio(f64),
}

/// A capital breach: its probability, and the mean shortfall below the
/// required capital at the first breach over the paths that breach.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Breach {
    /// The probability and its standard error.
    pub estimate: Estimate,
    /// The mean of `K - U` at the first breach; `None` when no path breaches.
    pub shortfall: Option<f64>,
}

/// The projection's paths: every [`Line`] by path and year.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionPaths {
    n_sims: usize,
    years: usize,
    opening_surplus: f64,
    lines: Vec<Vec<f64>>,
}

impl ProjectionPaths {
    /// The number of paths.
    pub fn n_sims(&self) -> usize {
        self.n_sims
    }

    /// The years projected.
    pub fn years(&self) -> usize {
        self.years
    }

    /// The opening surplus, the assets less the opening reserves.
    pub fn opening_surplus(&self) -> f64 {
        self.opening_surplus
    }

    /// A line's values, simulation-major: `values[s * years + t - 1]`.
    pub fn values(&self, line: Line) -> &[f64] {
        &self.lines[line as usize]
    }

    /// A line on path `sim` in year `year`, 1 to the horizon.
    pub fn value(&self, line: Line, sim: usize, year: usize) -> Result<f64> {
        if sim >= self.n_sims || year == 0 || year > self.years {
            return Err(data(format!(
                "path {sim}, year {year} is outside {} paths of years 1 to {}",
                self.n_sims, self.years
            )));
        }
        Ok(self.lines[line as usize][sim * self.years + year - 1])
    }

    /// A line as a distribution with one component per year (dimension
    /// `year`, 1 to the horizon), so its quantiles, TVaR and capital
    /// allocation apply.
    pub fn distribution(&self, line: Line) -> Result<PredictiveDistribution> {
        let components: Vec<ComponentKey> =
            (1..=self.years).map(|t| vec![(t as i64).into()]).collect();
        let provenance =
            Provenance::new(line.model()).version("prospicio-aggregate", env!("CARGO_PKG_VERSION"));
        PredictiveDistribution::from_draws(
            vec!["year".into()],
            components,
            self.lines[line as usize].clone(),
            provenance,
        )
    }

    fn check_horizon(&self, horizon: usize) -> Result<()> {
        if horizon == 0 || horizon > self.years {
            return Err(data(format!(
                "a horizon of {horizon} years is outside 1 to {}",
                self.years
            )));
        }
        Ok(())
    }

    /// The probability that the surplus is negative at some year end up to
    /// `horizon`.
    pub fn ruin(&self, horizon: usize) -> Result<Estimate> {
        self.check_horizon(horizon)?;
        let surplus = self.values(Line::Surplus);
        let hits = (0..self.n_sims)
            .filter(|&s| {
                surplus[s * self.years..s * self.years + horizon]
                    .iter()
                    .any(|&u| u < 0.0)
            })
            .count();
        Ok(Estimate::of(hits, self.n_sims))
    }

    /// The probability that the surplus falls below the required capital at
    /// some year end up to `horizon`, and the mean shortfall at the first
    /// breach.
    pub fn breach(&self, horizon: usize, capital: &RequiredCapital) -> Result<Breach> {
        self.check_horizon(horizon)?;
        if let RequiredCapital::ByYear(k) = capital
            && k.len() < horizon
        {
            return Err(data(format!(
                "the required capital has {} years, the horizon {horizon}",
                k.len()
            )));
        }
        let (surplus, premium, reserves) = (
            self.values(Line::Surplus),
            self.values(Line::Premium),
            self.values(Line::Reserves),
        );
        let (mut hits, mut short) = (0, 0.0);
        for s in 0..self.n_sims {
            for t in 0..horizon {
                let i = s * self.years + t;
                let k = match capital {
                    RequiredCapital::Fixed(k) => *k,
                    RequiredCapital::ByYear(k) => k[t],
                    RequiredCapital::PremiumRatio(r) => r * premium[i],
                    RequiredCapital::ReserveRatio(r) => r * reserves[i],
                };
                if surplus[i] < k {
                    hits += 1;
                    short += k - surplus[i];
                    break;
                }
            }
        }
        Ok(Breach {
            estimate: Estimate::of(hits, self.n_sims),
            shortfall: (hits > 0).then(|| short / hits as f64),
        })
    }
}
