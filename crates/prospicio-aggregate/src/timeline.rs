//! One accident-period timeline shared by the underwriting years of a
//! term (`docs/design/aggregate.md`, "Underwriting years with a rank
//! correlation", item 4).
//!
//! A [`Timeline`] holds each simulated event once, at a time over the whole
//! term, measured in accident months. A cover basis splits it into contract
//! years for [`crate::Tower::apply_term`]: under
//! [`CoverBasis::LossesOccurring`] contract year `k` takes the events of
//! accident year `k` whole; under [`CoverBasis::RisksAttaching`]
//! underwriting year `u` takes its [`Exposure`] share of each event in the
//! months its policies are in force, so one event can fall to two years.
//! Each year's losses keep the index of the event they came from
//! ([`Timeline::years_with_sources`]). An aggregate part (attritional
//! losses with no event or time) is spread over its accident year's months
//! by a monthly pattern, uniform unless set
//! ([`Timeline::with_aggregate_pattern`]), and falls to the years as an
//! event in that month would. A tower's basis is set with
//! [`Tower::cover_basis`] and applied with [`Tower::apply_timeline`].

use prospicio_core::{Error, Result};

use prospicio_prob::PredictiveDistribution;

use crate::monte_carlo::EventSet;
use crate::reinsurance::Tower;

/// Accident months in a year.
pub const MONTHS_PER_YEAR: usize = 12;

/// How a cover attaches to the losses of a term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoverBasis {
    /// Losses occurring (LOD): contract year `k` covers the losses that
    /// happen in accident year `k`, whatever year their policies were
    /// written in.
    #[default]
    LossesOccurring,
    /// Risks attaching (RAD): underwriting year `u` covers the losses of
    /// the policies written in it, whenever they happen.
    RisksAttaching,
}

/// Each underwriting year's share of a full book's exposure in each
/// accident month from the start of the term: one row a year, one share a
/// month. A month's shares add to at most 1; the rest belongs to
/// underwriting years outside the term.
#[derive(Debug, Clone, PartialEq)]
pub struct Exposure {
    shares: Vec<Vec<f64>>,
}

impl Exposure {
    /// Shares given directly, one row per underwriting year, all of the
    /// same length. Fails on no rows or months, a share outside `[0, 1]`,
    /// or a month whose shares add to more than 1.
    ///
    /// ```
    /// use prospicio_aggregate::timeline::Exposure;
    ///
    /// let e = Exposure::new(vec![vec![1.0, 0.5, 0.0], vec![0.0, 0.5, 1.0]]).unwrap();
    /// assert_eq!((e.n_years(), e.months()), (2, 3));
    /// assert!(Exposure::new(vec![vec![0.7], vec![0.7]]).is_err());
    /// ```
    pub fn new(shares: Vec<Vec<f64>>) -> Result<Self> {
        let months = shares.first().map_or(0, Vec::len);
        if months == 0 {
            return Err(Error::Data(
                "exposure needs at least one underwriting year and one month".into(),
            ));
        }
        if shares.iter().any(|r| r.len() != months) {
            return Err(Error::Data(
                "every underwriting year needs a share for each month".into(),
            ));
        }
        if let Some(&s) = shares.iter().flatten().find(|s| !(0.0..=1.0).contains(*s)) {
            return Err(Error::InvalidParameter {
                name: "shares",
                value: s,
                reason: "must be in [0, 1]",
            });
        }
        for m in 0..months {
            let total: f64 = shares.iter().map(|r| r[m]).sum();
            if total > 1.0 + 1e-9 {
                return Err(Error::InvalidParameter {
                    name: "shares",
                    value: total,
                    reason: "a month's shares must add to at most 1",
                });
            }
        }
        Ok(Self { shares })
    }

    /// `n_years` underwriting years, each writing its policies evenly over
    /// its twelve months, each policy in force for `policy_months` from the
    /// middle of the month it is written in. Year `u`'s share of month `m`
    /// is its policies' time in force within `m` over `policy_months`, so
    /// the shares of a month in mid-term add to 1 and each year's shares
    /// add to 12. The exposure runs `12 n_years + policy_months` months.
    ///
    /// ```
    /// use prospicio_aggregate::timeline::Exposure;
    ///
    /// let e = Exposure::uniform(2, 12).unwrap();
    /// assert_eq!(e.months(), 36);
    /// // Month 12: the first year's policies run off as the second's come in.
    /// let m12 = e.share(0, 12) + e.share(1, 12);
    /// assert!((m12 - 1.0).abs() < 1e-12);
    /// assert!((e.share(0, 0) - 0.5 / 12.0).abs() < 1e-12);
    /// ```
    pub fn uniform(n_years: usize, policy_months: usize) -> Result<Self> {
        if n_years == 0 || policy_months == 0 {
            return Err(Error::Data(
                "uniform exposure needs at least one year and one month in force".into(),
            ));
        }
        let months = MONTHS_PER_YEAR * n_years + policy_months;
        let p = policy_months as f64;
        let shares = (0..n_years)
            .map(|u| {
                let mut row = vec![0.0; months];
                for w in MONTHS_PER_YEAR * u..MONTHS_PER_YEAR * (u + 1) {
                    let (start, end) = (w as f64 + 0.5, w as f64 + 0.5 + p);
                    for (m, share) in row.iter_mut().enumerate().skip(w) {
                        let overlap = end.min(m as f64 + 1.0) - start.max(m as f64);
                        if overlap <= 0.0 {
                            break;
                        }
                        *share += overlap / p;
                    }
                }
                row
            })
            .collect();
        Self::new(shares)
    }

    /// The number of underwriting years.
    pub fn n_years(&self) -> usize {
        self.shares.len()
    }

    /// The number of accident months the shares run over.
    pub fn months(&self) -> usize {
        self.shares[0].len()
    }

    /// Underwriting year `year`'s share of accident month `month`.
    pub fn share(&self, year: usize, month: usize) -> f64 {
        self.shares[year][month]
    }

    /// The shares, one row a year.
    pub fn shares(&self) -> &[Vec<f64>] {
        &self.shares
    }
}

/// Events over a whole term of accident months: simulation `j` is one
/// path, each event stored once with its time as a fraction of the term.
/// An event at time `x` is in accident month `⌊x·M⌋` (the last month for
/// `x = 1`) of the term's `M`.
///
/// An aggregate part of the events has no time. It is held by accident
/// period (each accident year's amounts from
/// [`from_years`](Self::from_years), the whole term's from
/// [`new`](Self::new)) and spread over the period's months by the
/// aggregate pattern, `pattern[i]` the weight of the `i`-th month of each
/// accident year, uniform unless set
/// ([`with_aggregate_pattern`](Self::with_aggregate_pattern)). Each month's
/// amount then falls to the contract years as an event in that month would:
/// whole to its accident year under [`CoverBasis::LossesOccurring`], by the
/// exposure shares under [`CoverBasis::RisksAttaching`].
#[derive(Debug, Clone)]
pub struct Timeline {
    /// The events, without their aggregate part.
    events: EventSet,
    months: usize,
    part: Option<TimelinePart>,
    pattern: [f64; MONTHS_PER_YEAR],
}

/// A timeline's aggregate part, by accident period.
#[derive(Debug, Clone)]
struct TimelinePart {
    /// Each column's combination, an index into the events' labels.
    of_column: Vec<usize>,
    /// Each period's first accident month and number of months.
    periods: Vec<(usize, usize)>,
    /// `amounts[(sim * periods + p) * columns + j]`.
    amounts: Vec<f64>,
}

impl Timeline {
    /// A timeline of `months` accident months from events whose times
    /// ([`EventSet::with_times`]) are fractions of the whole term. An
    /// aggregate part of the events is the whole term's, spread over all
    /// its months by the aggregate pattern. Fails without times or with no
    /// months.
    pub fn new(events: EventSet, months: usize) -> Result<Self> {
        if months == 0 {
            return Err(Error::Data("a timeline needs at least one month".into()));
        }
        if !events.has_times() {
            return Err(Error::Data(
                "a timeline needs each event's time: set times on the events".into(),
            ));
        }
        let (events, part) = events.without_aggregate_part();
        let part = part.map(|p| TimelinePart {
            of_column: p.of_column,
            periods: vec![(0, months)],
            amounts: p.amounts,
        });
        Ok(Self {
            events,
            months,
            part,
            pattern: [1.0; MONTHS_PER_YEAR],
        })
    }

    /// The same timeline with the aggregate part spread over each accident
    /// year's months by `pattern`, twelve non-negative weights (they need
    /// not add to 1), `pattern[i]` for the `i`-th month of each year: a
    /// seasonal book's attritional losses, say. The default is uniform.
    /// Fails on a weight that is negative or not finite, on all weights 0,
    /// or when an accident period's months all have weight 0.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    /// use prospicio_aggregate::timeline::{CoverBasis, Exposure, Timeline};
    /// use prospicio_prob::{PredictiveDistribution, Provenance};
    ///
    /// // One accident year with no events and attritional losses of 120.
    /// let part = PredictiveDistribution::from_draws(
    ///     vec![], vec![vec![]], vec![120.0], Provenance::new("attritional"),
    /// )
    /// .unwrap();
    /// let year = EventSet::from_years(vec![vec![]], 0)
    ///     .unwrap().with_times(vec![]).unwrap()
    ///     .with_aggregate_part(&part).unwrap();
    /// let timeline = Timeline::from_years(&[year]).unwrap();
    /// // The first underwriting year covers January to June, the second
    /// // July to December.
    /// let first_half: Vec<f64> = (0..12).map(|m| if m < 6 { 1.0 } else { 0.0 }).collect();
    /// let second_half = first_half.iter().map(|s| 1.0 - s).collect();
    /// let exposure = Exposure::new(vec![first_half, second_half]).unwrap();
    /// let uniform = timeline.years(CoverBasis::RisksAttaching, &exposure).unwrap();
    /// assert_eq!(uniform[0].aggregate_part(0), Some(&[60.0][..]));
    /// assert_eq!(uniform[1].aggregate_part(0), Some(&[60.0][..]));
    /// // All of it in December: it falls to the second year.
    /// let mut december = [0.0; 12];
    /// december[11] = 1.0;
    /// let seasonal = timeline.with_aggregate_pattern(&december).unwrap();
    /// let late = seasonal.years(CoverBasis::RisksAttaching, &exposure).unwrap();
    /// assert_eq!(late[0].aggregate_part(0), Some(&[0.0][..]));
    /// assert_eq!(late[1].aggregate_part(0), Some(&[120.0][..]));
    /// ```
    pub fn with_aggregate_pattern(mut self, pattern: &[f64]) -> Result<Self> {
        if pattern.len() != MONTHS_PER_YEAR {
            return Err(Error::Data(format!(
                "an aggregate pattern needs {MONTHS_PER_YEAR} monthly weights, got {}",
                pattern.len()
            )));
        }
        if let Some(&bad) = pattern.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
            return Err(Error::InvalidParameter {
                name: "pattern",
                value: bad,
                reason: "must be finite and non-negative",
            });
        }
        if pattern.iter().sum::<f64>() <= 0.0 {
            return Err(Error::Data(
                "an aggregate pattern needs a positive weight".into(),
            ));
        }
        let pattern: [f64; MONTHS_PER_YEAR] = pattern.try_into().expect("twelve weights");
        if let Some(part) = &self.part {
            for &(start, len) in &part.periods {
                if (start..start + len).all(|m| pattern[m % MONTHS_PER_YEAR] == 0.0) {
                    return Err(Error::Data(format!(
                        "the aggregate pattern gives accident months {} to {} no weight",
                        start + 1,
                        start + len
                    )));
                }
            }
        }
        self.pattern = pattern;
        Ok(self)
    }

    /// The aggregate pattern: the weight of each month of an accident year,
    /// as set (uniform by default).
    pub fn aggregate_pattern(&self) -> &[f64; MONTHS_PER_YEAR] {
        &self.pattern
    }

    /// Whether the timeline carries an aggregate part.
    pub fn has_aggregate_part(&self) -> bool {
        self.part.is_some()
    }

    /// A timeline stitched from one event set per accident year, each with
    /// times within its year and the same number of simulations: year
    /// `t`'s event at time `τ` is at `(t + τ) / T` of the term, in path
    /// `j` for its simulation `j`. The years can come from
    /// [`EventSet::term_years`] or [`EventSet::correlated_years`]. Labels
    /// and sums insured are kept as [`EventSet::merge`] keeps them. Each
    /// year's aggregate part stays its accident year's, spread over its
    /// twelve months by the aggregate pattern; a year without one has none.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    /// use prospicio_aggregate::timeline::{CoverBasis, Exposure, Timeline};
    ///
    /// let year1 = EventSet::from_years(vec![vec![10.0], vec![]], 0)
    ///     .unwrap().with_times(vec![0.5]).unwrap();
    /// let year2 = EventSet::from_years(vec![vec![20.0], vec![30.0]], 0)
    ///     .unwrap().with_times(vec![0.25, 0.75]).unwrap();
    /// let timeline = Timeline::from_years(&[year1, year2]).unwrap();
    /// assert_eq!(timeline.months(), 24);
    /// assert_eq!(timeline.events().events(0), &[10.0, 20.0]);
    ///
    /// // Losses occurring: each contract year is its accident year again.
    /// let lod = timeline
    ///     .years(CoverBasis::LossesOccurring, &Exposure::uniform(2, 12).unwrap())
    ///     .unwrap();
    /// assert_eq!(lod[1].events(0), &[20.0]);
    /// assert_eq!(lod[1].events(1), &[30.0]);
    /// ```
    pub fn from_years(years: &[EventSet]) -> Result<Self> {
        let Some(first) = years.first() else {
            return Err(Error::Data("a timeline needs at least one year".into()));
        };
        let t = years.len() as f64;
        let mut merged: Option<EventSet> = None;
        for (k, year) in years.iter().enumerate() {
            if !year.has_times() {
                return Err(Error::Data(format!(
                    "accident year {} has no times: set times on its events",
                    k + 1
                )));
            }
            if year.n_sims() != first.n_sims() {
                return Err(Error::Data(format!(
                    "accident year {} has {} simulations and year 1 has {}",
                    k + 1,
                    year.n_sims(),
                    first.n_sims()
                )));
            }
            let times = (0..year.n_sims())
                .flat_map(|s| year.times(s).unwrap_or_default().iter())
                .map(|&tau| ((k as f64 + tau) / t).min(1.0))
                .collect();
            let shifted = year.clone().with_times(times)?;
            merged = Some(match merged {
                None => shifted,
                Some(m) => m.merge(&shifted)?,
            });
        }
        let merged = merged.expect("at least one year");
        // Every year's part on the merged set's columns, matched by labels.
        let columns: Option<Vec<Vec<String>>> = merged
            .aggregate_part_labels()
            .map(|c| c.into_iter().map(<[String]>::to_vec).collect());
        let dims = merged.label_dims().to_vec();
        let (events, merged_part) = merged.without_aggregate_part();
        let part = match (columns, merged_part) {
            (Some(columns), Some(merged_part)) => {
                let of_column = merged_part.of_column;
                let (n, p, m) = (first.n_sims(), years.len(), columns.len());
                let mut amounts = vec![0.0; n * p * m];
                for (k, year) in years.iter().enumerate() {
                    let Some(labels) = year.aggregate_part_labels() else {
                        continue;
                    };
                    let own = year.label_dims();
                    let at: Vec<usize> = labels
                        .iter()
                        .map(|values| {
                            let in_merged: Vec<String> = dims
                                .iter()
                                .map(|d| {
                                    own.iter()
                                        .position(|x| x == d)
                                        .map_or_else(String::new, |i| values[i].clone())
                                })
                                .collect();
                            columns
                                .iter()
                                .position(|c| *c == in_merged)
                                .expect("merge keeps every year's columns")
                        })
                        .collect();
                    for sim in 0..n {
                        let row = year.aggregate_part(sim).expect("the year has a part");
                        for (j, &x) in row.iter().enumerate() {
                            amounts[(sim * p + k) * m + at[j]] += x;
                        }
                    }
                }
                Some(TimelinePart {
                    of_column,
                    periods: (0..p)
                        .map(|k| (MONTHS_PER_YEAR * k, MONTHS_PER_YEAR))
                        .collect(),
                    amounts,
                })
            }
            _ => None,
        };
        Ok(Self {
            events,
            months: MONTHS_PER_YEAR * years.len(),
            part,
            pattern: [1.0; MONTHS_PER_YEAR],
        })
    }

    /// The events, with times as fractions of the term, without the
    /// aggregate part.
    pub fn events(&self) -> &EventSet {
        &self.events
    }

    /// The number of accident months.
    pub fn months(&self) -> usize {
        self.months
    }

    /// The number of simulated paths.
    pub fn n_sims(&self) -> usize {
        self.events.n_sims()
    }

    /// The contract years under `basis`, one [`EventSet`] each with the
    /// timeline's simulations, for [`crate::Tower::apply_term`].
    ///
    /// Under [`CoverBasis::LossesOccurring`], contract year `k` takes every
    /// event in accident months `12k..12k + 12` whole, at its time within
    /// the year; `exposure` gives only the number of years, and the
    /// timeline must run `12 n_years` months. Under
    /// [`CoverBasis::RisksAttaching`], underwriting year `u` takes
    /// `exposure.share(u, m)` of each event in month `m` (its sum insured
    /// scaled alike), at its time from the first month in which the year
    /// has a share to the end of the last, as a fraction of that span; the
    /// timeline must run at least the exposure's months. Labels are kept.
    /// An aggregate part falls to the years month by month in the same
    /// way, after the aggregate pattern spreads each accident period's
    /// amount over its months, and each year carries its own.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    /// use prospicio_aggregate::timeline::{CoverBasis, Exposure, Timeline};
    ///
    /// // One path of 24 months: an event of 100 in month 13.
    /// let events = EventSet::from_years(vec![vec![100.0]], 0)
    ///     .unwrap().with_times(vec![13.5 / 24.0]).unwrap();
    /// let timeline = Timeline::new(events, 24).unwrap();
    /// // Month 13 is a quarter the first year's run-off, three quarters the second's.
    /// let mut shares = vec![vec![0.0; 24]; 2];
    /// shares[0][13] = 0.25;
    /// shares[1][13] = 0.75;
    /// let rad = timeline
    ///     .years(CoverBasis::RisksAttaching, &Exposure::new(shares).unwrap())
    ///     .unwrap();
    /// assert_eq!(rad[0].events(0), &[25.0]);
    /// assert_eq!(rad[1].events(0), &[75.0]);
    /// ```
    pub fn years(&self, basis: CoverBasis, exposure: &Exposure) -> Result<Vec<EventSet>> {
        Ok(self.years_with_sources(basis, exposure)?.0)
    }

    /// [`years`](Self::years), with each year's sources: for each of the
    /// year's losses, in order, its event's index among all the
    /// timeline's losses, so the same event can be found in every year it
    /// falls to.
    pub fn years_with_sources(
        &self,
        basis: CoverBasis,
        exposure: &Exposure,
    ) -> Result<(Vec<EventSet>, Vec<Vec<usize>>)> {
        let n_years = exposure.n_years();
        let needed = match basis {
            CoverBasis::LossesOccurring => MONTHS_PER_YEAR * n_years,
            CoverBasis::RisksAttaching => exposure.months(),
        };
        if self.months < needed {
            return Err(Error::Data(format!(
                "the timeline runs {} months and the term needs {needed}",
                self.months
            )));
        }
        // Each year's window of months, for its losses' times.
        let windows: Vec<(f64, f64)> = (0..n_years)
            .map(|u| match basis {
                CoverBasis::LossesOccurring => {
                    ((MONTHS_PER_YEAR * u) as f64, MONTHS_PER_YEAR as f64)
                }
                CoverBasis::RisksAttaching => {
                    let row = &exposure.shares[u];
                    let first = row.iter().position(|&s| s > 0.0);
                    let last = row.iter().rposition(|&s| s > 0.0);
                    match (first, last) {
                        (Some(f), Some(l)) => (f as f64, (l + 1 - f) as f64),
                        _ => (0.0, 1.0),
                    }
                }
            })
            .collect();
        let m_total = self.months as f64;
        let mut rows = vec![Vec::with_capacity(self.n_sims()); n_years];
        let mut sources = vec![Vec::new(); n_years];
        for sim in 0..self.n_sims() {
            let mut picks = vec![Vec::new(); n_years];
            let offset = self.events.offset(sim);
            let times = self.events.times(sim).expect("a timeline has times");
            for (i, &x) in times.iter().enumerate() {
                let at = x * m_total;
                let month = (at.floor() as usize).min(self.months - 1);
                let source = offset + i;
                let time_in = |u: usize| {
                    let (start, span) = windows[u];
                    ((at - start) / span).clamp(0.0, 1.0)
                };
                match basis {
                    CoverBasis::LossesOccurring => {
                        let k = month / MONTHS_PER_YEAR;
                        if k < n_years {
                            picks[k].push((source, 1.0, time_in(k)));
                            sources[k].push(source);
                        }
                    }
                    CoverBasis::RisksAttaching => {
                        if month >= exposure.months() {
                            continue;
                        }
                        for u in 0..n_years {
                            let share = exposure.shares[u][month];
                            if share > 0.0 {
                                picks[u].push((source, share, time_in(u)));
                                sources[u].push(source);
                            }
                        }
                    }
                }
            }
            for (row, pick) in rows.iter_mut().zip(picks) {
                row.push(pick);
            }
        }
        let mut years: Vec<EventSet> = rows.iter().map(|r| self.events.rebuilt(r)).collect();
        if let Some(part) = &self.part {
            // Year u's share of accident period p's amount: the pattern's
            // weight of each of p's months times the year's share of it.
            let share = |u: usize, month: usize| match basis {
                CoverBasis::LossesOccurring => {
                    if month / MONTHS_PER_YEAR == u {
                        1.0
                    } else {
                        0.0
                    }
                }
                CoverBasis::RisksAttaching => {
                    if month < exposure.months() {
                        exposure.shares[u][month]
                    } else {
                        0.0
                    }
                }
            };
            let coefficients: Vec<Vec<f64>> = (0..n_years)
                .map(|u| {
                    part.periods
                        .iter()
                        .map(|&(start, len)| {
                            let w = |m: usize| self.pattern[m % MONTHS_PER_YEAR];
                            let total: f64 = (start..start + len).map(w).sum();
                            (start..start + len)
                                .map(|m| w(m) * share(u, m))
                                .sum::<f64>()
                                / total
                        })
                        .collect()
                })
                .collect();
            let (n, p, m) = (self.n_sims(), part.periods.len(), part.of_column.len());
            years = years
                .into_iter()
                .zip(&coefficients)
                .map(|(year, c)| {
                    let mut amounts = vec![0.0; n * m];
                    for sim in 0..n {
                        for (k, &ck) in c.iter().enumerate() {
                            if ck == 0.0 {
                                continue;
                            }
                            let from = &part.amounts[(sim * p + k) * m..(sim * p + k + 1) * m];
                            for (to, &x) in amounts[sim * m..(sim + 1) * m].iter_mut().zip(from) {
                                *to += ck * x;
                            }
                        }
                    }
                    year.with_aggregate_columns(part.of_column.clone(), amounts)
                })
                .collect();
        }
        Ok((years, sources))
    }
}

impl Tower {
    /// The tower with its cover basis set: how the programme attaches to
    /// the losses of a term when applied to a [`Timeline`]. One basis holds
    /// for every layer of the tower, as for every contract of a
    /// programme; a deal of programmes on different bases is several
    /// towers.
    pub fn cover_basis(mut self, basis: CoverBasis) -> Self {
        self.cover_basis = Some(basis);
        self
    }

    /// Applies the tower over the contract years that its cover basis
    /// gives on `timeline` ([`Timeline::years`] with `exposure`), as
    /// [`apply_term`](Self::apply_term) does: the result's `year` is the
    /// contract year, `1..=exposure.n_years()`. Fails if the basis is not
    /// set: a programme's basis is required, not defaulted.
    ///
    /// ```
    /// use prospicio_aggregate::{EventSet, Layer, Tower};
    /// use prospicio_aggregate::timeline::{CoverBasis, Exposure, Timeline};
    ///
    /// // One path of 24 months: an event of 100 in month 13.
    /// let events = EventSet::from_years(vec![vec![100.0]], 0)
    ///     .unwrap().with_times(vec![13.5 / 24.0]).unwrap();
    /// let timeline = Timeline::new(events, 24).unwrap();
    /// let mut shares = vec![vec![0.0; 24]; 2];
    /// shares[0][13] = 0.25;
    /// shares[1][13] = 0.75;
    /// let exposure = Exposure::new(shares).unwrap();
    /// let xl = Layer::xol("XL", 50.0, 20.0).unwrap();
    /// let tower = Tower::new(vec![xl]).unwrap();
    /// assert!(tower.apply_timeline(&timeline, &exposure).is_err());
    /// let rad = tower
    ///     .cover_basis(CoverBasis::RisksAttaching)
    ///     .apply_timeline(&timeline, &exposure)
    ///     .unwrap();
    /// // Year 1 takes 25 of the event, 5 above 20; year 2 takes 75, 50 of it in the layer.
    /// let ceded = |year: i64| {
    ///     use prospicio_prob::KeyValue;
    ///     let key = vec![KeyValue::from("ceded"), KeyValue::from("XL"), KeyValue::Int(year)];
    ///     prospicio_prob::Empirical::draws(&rad.marginal(&key).unwrap())[0]
    /// };
    /// assert_eq!((ceded(1), ceded(2)), (5.0, 50.0));
    /// ```
    pub fn apply_timeline(
        &self,
        timeline: &Timeline,
        exposure: &Exposure,
    ) -> Result<PredictiveDistribution> {
        let basis = self.cover_basis.ok_or_else(|| {
            Error::Data(
                "the tower has no cover basis: set losses occurring or risks attaching".into(),
            )
        })?;
        if !(self.interlocking && basis == CoverBasis::RisksAttaching) {
            return self.apply_term(&timeline.years(basis, exposure)?);
        }
        let (years, sources) = timeline.years_with_sources(basis, exposure)?;
        let factors = interlocking_factors(&years, &sources);
        self.apply_term_with(&years, Some(&factors))
    }

    /// The tower with its contracts interlocking (`true`) or not, as the
    /// programme's interlocking clause says; off by default.
    ///
    /// Under risks attaching an event in a month that several underwriting
    /// years write falls to each of them by its share. Without
    /// interlocking each year's contracts apply their full limits to their
    /// share, so the event can use a limit in each year. With it each
    /// contract's limit is cut, for that event, by the factor (the loss
    /// the year's contract covers) / (the event's loss across the
    /// programme's underwriting years in the term): with shares `s_u` of
    /// a month, year `u`'s factor is `s_u / Σ_v s_v`. Retentions are not
    /// cut, and each year's annual terms (its aggregate deductible and
    /// limit, reinstatements and their premiums) apply to its recoveries
    /// as usual. Under losses occurring an event falls to one year, so
    /// interlocking changes nothing.
    ///
    /// ```
    /// use prospicio_aggregate::{EventSet, Layer, Tower};
    /// use prospicio_aggregate::timeline::{CoverBasis, Exposure, Timeline};
    ///
    /// // An event of 100 written 25/75 by two years, on a 50 xs 20 layer.
    /// let events = EventSet::from_years(vec![vec![100.0]], 0)
    ///     .unwrap().with_times(vec![13.5 / 24.0]).unwrap();
    /// let timeline = Timeline::new(events, 24).unwrap();
    /// let mut shares = vec![vec![0.0; 24]; 2];
    /// shares[0][13] = 0.25;
    /// shares[1][13] = 0.75;
    /// let exposure = Exposure::new(shares).unwrap();
    /// let tower = Tower::new(vec![Layer::xol("XL", 50.0, 20.0).unwrap()])
    ///     .unwrap()
    ///     .cover_basis(CoverBasis::RisksAttaching)
    ///     .interlocking(true);
    /// let rad = tower.apply_timeline(&timeline, &exposure).unwrap();
    /// let ceded = |year: i64| {
    ///     use prospicio_prob::KeyValue;
    ///     let key = vec![KeyValue::from("ceded"), KeyValue::from("XL"), KeyValue::Int(year)];
    ///     prospicio_prob::Empirical::draws(&rad.marginal(&key).unwrap())[0]
    /// };
    /// // Year 1: 25 - 20 = 5 within its limit 12.5; year 2: 75 - 20 = 55
    /// // cut to its limit 37.5. Without interlocking, 5 and 50.
    /// assert_eq!((ceded(1), ceded(2)), (5.0, 37.5));
    /// ```
    pub fn interlocking(mut self, on: bool) -> Self {
        self.interlocking = on;
        self
    }
}

/// Each loss's interlocking factor: its share of its event's losses across
/// all the years, in the order of each year's losses.
fn interlocking_factors(years: &[EventSet], sources: &[Vec<usize>]) -> Vec<Vec<f64>> {
    let flat = |e: &EventSet| -> Vec<f64> {
        (0..e.n_sims())
            .flat_map(|j| e.events(j).iter().copied())
            .collect()
    };
    let losses: Vec<Vec<f64>> = years.iter().map(flat).collect();
    let mut totals = std::collections::HashMap::<usize, f64>::new();
    for (l, s) in losses.iter().zip(sources) {
        for (&x, &k) in l.iter().zip(s) {
            *totals.entry(k).or_default() += x;
        }
    }
    losses
        .iter()
        .zip(sources)
        .map(|(l, s)| {
            l.iter()
                .zip(s)
                .map(|(&x, k)| {
                    let total = totals[k];
                    // A loss of zero takes no limit; keep it whole.
                    if total > 0.0 { x / total } else { 1.0 }
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Layer, Tower, simulate_events};
    use prospicio_prob::portfolio::CorrelationKind;
    use prospicio_prob::{Lognormal, Poisson};

    use crate::term::YearDraws;

    fn book(n: usize, seed: u64) -> EventSet {
        simulate_events(
            &Poisson::new(3.0).unwrap(),
            &Lognormal::from_mean_cv(10.0, 1.5).unwrap(),
            n,
            seed,
        )
        .unwrap()
        .with_uniform_times()
    }

    fn totals(e: &EventSet) -> Vec<f64> {
        (0..e.n_sims()).map(|s| e.events(s).iter().sum()).collect()
    }

    #[test]
    fn losses_occurring_on_one_year_is_the_year() {
        let year = book(500, 1);
        let timeline = Timeline::from_years(std::slice::from_ref(&year)).unwrap();
        let lod = timeline
            .years(
                CoverBasis::LossesOccurring,
                &Exposure::uniform(1, 12).unwrap(),
            )
            .unwrap();
        assert_eq!(lod.len(), 1);
        for s in 0..year.n_sims() {
            assert_eq!(lod[0].events(s), year.events(s));
            for (a, b) in lod[0].times(s).unwrap().iter().zip(year.times(s).unwrap()) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        let xl = Layer::xol("XL", 20.0, 10.0)
            .unwrap()
            .reinstatements(1)
            .unwrap()
            .paid_reinstatements(5.0, vec![1.0])
            .unwrap();
        let tower = Tower::new(vec![xl]).unwrap();
        let direct = tower.apply(&year).unwrap();
        let via = tower.apply_term(&lod).unwrap();
        assert_eq!(direct.draw_matrix().len(), via.draw_matrix().len());
        for (x, y) in direct.draw_matrix().iter().zip(via.draw_matrix()) {
            assert!((x - y).abs() < 1e-9);
        }
    }

    #[test]
    fn risks_attaching_with_one_year_of_exposure_is_losses_occurring() {
        let years = book(900, 2).term_years(3).unwrap();
        let timeline = Timeline::from_years(&years).unwrap();
        let mut shares = vec![vec![0.0; 36]; 3];
        for (u, row) in shares.iter_mut().enumerate() {
            for share in &mut row[12 * u..12 * (u + 1)] {
                *share = 1.0;
            }
        }
        let exposure = Exposure::new(shares).unwrap();
        let rad = timeline
            .years(CoverBasis::RisksAttaching, &exposure)
            .unwrap();
        let lod = timeline
            .years(CoverBasis::LossesOccurring, &exposure)
            .unwrap();
        for (r, l) in rad.iter().zip(&lod) {
            for s in 0..r.n_sims() {
                assert_eq!(r.events(s), l.events(s));
                assert_eq!(r.times(s), l.times(s));
            }
        }
        // And each is its accident year again.
        for (l, y) in lod.iter().zip(&years) {
            assert_eq!(totals(l), totals(y));
        }
    }

    #[test]
    fn shares_adding_to_one_conserve_every_event() {
        let timeline = Timeline::from_years(&book(800, 3).term_years(2).unwrap()).unwrap();
        // Two years sharing every month, the share moving from the first
        // to the second over the term.
        let first: Vec<f64> = (0..24).map(|m| 1.0 - m as f64 / 23.0).collect();
        let second: Vec<f64> = first.iter().map(|s| 1.0 - s).collect();
        let exposure = Exposure::new(vec![first, second]).unwrap();
        let (rad, sources) = timeline
            .years_with_sources(CoverBasis::RisksAttaching, &exposure)
            .unwrap();
        let whole = totals(timeline.events());
        let (a, b) = (totals(&rad[0]), totals(&rad[1]));
        for s in 0..whole.len() {
            assert!((a[s] + b[s] - whole[s]).abs() < 1e-9 * whole[s].max(1.0));
        }
        // An event in a month both years share is in both, by the same source.
        let both = sources[0].iter().filter(|i| sources[1].contains(i)).count();
        assert!(both > 0);
        for (year, src) in rad.iter().zip(&sources) {
            assert_eq!(year.counts().iter().sum::<usize>(), src.len());
            for s in 0..year.n_sims() {
                let t = year.times(s).unwrap();
                assert!(t.windows(2).all(|w| w[0] <= w[1]));
                assert!(t.iter().all(|x| (0.0..=1.0).contains(x)));
            }
        }
    }

    #[test]
    fn a_correlated_term_keeps_its_correlation() {
        let years = book(2000, 4)
            .correlated_years(3, &[0.6], CorrelationKind::Spearman, YearDraws::Same, 5)
            .unwrap();
        let timeline = Timeline::from_years(&years).unwrap();
        let lod = timeline
            .years(
                CoverBasis::LossesOccurring,
                &Exposure::uniform(3, 12).unwrap(),
            )
            .unwrap();
        for (l, y) in lod.iter().zip(&years) {
            for s in 0..l.n_sims() {
                assert_eq!(l.events(s), y.events(s));
            }
        }
    }

    #[test]
    fn uniform_exposure_ramps_up_and_runs_off() {
        let e = Exposure::uniform(3, 12).unwrap();
        assert_eq!((e.n_years(), e.months()), (3, 48));
        for u in 0..3 {
            let total: f64 = e.shares()[u].iter().sum();
            assert!((total - 12.0).abs() < 1e-9);
        }
        for m in 12..36 {
            let total: f64 = (0..3).map(|u| e.share(u, m)).sum();
            assert!((total - 1.0).abs() < 1e-12, "month {m}: {total}");
        }
        // The book builds up over the first year and runs off over the last.
        assert!(e.share(0, 0) < e.share(0, 6) && e.share(0, 6) < e.share(0, 11));
        assert!(e.share(2, 47) < e.share(2, 40));
        // Six-month policies: the last, written in the middle of month 11,
        // runs to the middle of month 17.
        let short = Exposure::uniform(1, 6).unwrap();
        assert_eq!(short.months(), 18);
        assert!((short.share(0, 17) - 0.5 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn risks_attaching_times_span_the_years_window() {
        let events = EventSet::from_years(vec![vec![1.0, 2.0, 3.0]], 0)
            .unwrap()
            .with_times(vec![0.5 / 36.0, 18.0 / 36.0, 35.5 / 36.0])
            .unwrap();
        let timeline = Timeline::new(events, 36).unwrap();
        let exposure = Exposure::uniform(1, 12).unwrap();
        let rad = timeline
            .years(CoverBasis::RisksAttaching, &exposure)
            .unwrap();
        // The year's policies are in force in months 0 to 23: the event in
        // month 35 is not its.
        assert_eq!(rad[0].events(0).len(), 2);
        let t = rad[0].times(0).unwrap();
        assert!((t[0] - 0.5 / 24.0).abs() < 1e-12 && (t[1] - 0.75).abs() < 1e-12);
        assert!((rad[0].events(0)[1] - 2.0 * exposure.share(0, 18)).abs() < 1e-12);
    }

    #[test]
    fn errors() {
        let bare = EventSet::from_years(vec![vec![1.0]], 0).unwrap();
        assert!(Timeline::new(bare.clone(), 12).is_err());
        assert!(Timeline::from_years(&[bare]).is_err());
        assert!(Timeline::from_years(&[]).is_err());
        let timed = book(10, 5);
        assert!(Timeline::new(timed.clone(), 0).is_err());
        let timeline = Timeline::new(timed, 12).unwrap();
        let two = Exposure::uniform(2, 12).unwrap();
        assert!(timeline.years(CoverBasis::LossesOccurring, &two).is_err());
        assert!(timeline.years(CoverBasis::RisksAttaching, &two).is_err());
        assert!(Exposure::new(vec![]).is_err());
        assert!(Exposure::new(vec![vec![0.5], vec![]]).is_err());
        assert!(Exposure::new(vec![vec![-0.1]]).is_err());
        assert!(Exposure::uniform(0, 12).is_err());
        assert!(Exposure::uniform(1, 0).is_err());
    }

    /// An aggregate part of `amounts`, one row a simulation, in the
    /// dimension `peril` with `perils` as its columns (no dimension when
    /// `perils` is empty).
    fn part(perils: &[&str], amounts: Vec<f64>) -> PredictiveDistribution {
        use prospicio_prob::{KeyValue, Provenance};
        let (dims, components): (Vec<String>, Vec<Vec<KeyValue>>) = if perils.is_empty() {
            (vec![], vec![vec![]])
        } else {
            (
                vec!["peril".into()],
                perils.iter().map(|p| vec![KeyValue::from(*p)]).collect(),
            )
        };
        PredictiveDistribution::from_draws(dims, components, amounts, Provenance::new("part"))
            .unwrap()
    }

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!(
                (x - y).abs() < 1e-9 * (1.0 + y.abs()),
                "{a:?} against {b:?}"
            );
        }
    }

    #[test]
    fn losses_occurring_gives_each_year_its_own_aggregate_part() {
        // Year 1 by peril, year 2 wind only, year 3 none.
        let one = book(4, 1)
            .with_aggregate_part(&part(&["flood", "wind"], (0..8).map(f64::from).collect()))
            .unwrap();
        let two = book(4, 2)
            .with_aggregate_part(&part(&["wind"], vec![10.0, 20.0, 30.0, 40.0]))
            .unwrap();
        let three = book(4, 3);
        let timeline = Timeline::from_years(&[one.clone(), two.clone(), three.clone()]).unwrap();
        assert!(timeline.has_aggregate_part());
        assert!(!timeline.events().has_aggregate_part());
        // Any pattern: under LOD an accident year's amount stays whole.
        let pattern: Vec<f64> = (1..=12).map(f64::from).collect();
        let timeline = timeline.with_aggregate_pattern(&pattern).unwrap();
        let years = timeline
            .years(
                CoverBasis::LossesOccurring,
                &Exposure::uniform(3, 12).unwrap(),
            )
            .unwrap();
        let labels: Vec<Vec<String>> = years[0]
            .aggregate_part_labels()
            .unwrap()
            .into_iter()
            .map(<[String]>::to_vec)
            .collect();
        assert_eq!(
            labels,
            [vec!["flood".to_string()], vec!["wind".to_string()]]
        );
        for sim in 0..4 {
            let s = sim as f64;
            close(
                years[0].aggregate_part(sim).unwrap(),
                &[2.0 * s, 2.0 * s + 1.0],
            );
            close(
                years[1].aggregate_part(sim).unwrap(),
                &[0.0, 10.0 * (s + 1.0)],
            );
            close(years[2].aggregate_part(sim).unwrap(), &[0.0, 0.0]);
        }
        // The events are the accident years' own, as before.
        for (year, own) in years.iter().zip([&one, &two, &three]) {
            close(&totals(year), &totals(own));
        }
    }

    #[test]
    fn risks_attaching_conserves_the_aggregate_part() {
        // Three accident years; two underwriting years of 12-month
        // policies, the rest of each month outside the term.
        let years: Vec<EventSet> = (0..3)
            .map(|k| {
                book(5, 10 + k)
                    .with_aggregate_part(&part(&[], (0..5).map(|s| 100.0 + f64::from(s)).collect()))
                    .unwrap()
            })
            .collect();
        let exposure = Exposure::uniform(2, 12).unwrap();
        for pattern in [
            [1.0; 12],
            [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 5.0, 1.0, 1.0, 0.0, 4.0],
        ] {
            let timeline = Timeline::from_years(&years)
                .unwrap()
                .with_aggregate_pattern(&pattern)
                .unwrap();
            let rad = timeline
                .years(CoverBasis::RisksAttaching, &exposure)
                .unwrap();
            // Each month's part falls to the years by their shares, as an
            // event in that month does.
            let w: f64 = pattern.iter().sum();
            for sim in 0..5 {
                let a = 100.0 + sim as f64;
                for (u, year) in rad.iter().enumerate() {
                    let expected: f64 = (0..exposure.months())
                        .map(|m| a * pattern[m % 12] / w * exposure.share(u, m))
                        .sum();
                    close(year.aggregate_part(sim).unwrap(), &[expected]);
                }
            }
            // Shares that add to 1 in every month and cover the term keep
            // every amount.
            let all = Exposure::new(vec![vec![0.4; 36], vec![0.6; 36]]).unwrap();
            let both = timeline.years(CoverBasis::RisksAttaching, &all).unwrap();
            for sim in 0..5 {
                let sum = both[0].aggregate_part(sim).unwrap()[0]
                    + both[1].aggregate_part(sim).unwrap()[0];
                close(&[sum], &[3.0 * (100.0 + sim as f64)]);
            }
        }
    }

    #[test]
    fn a_term_part_is_spread_over_every_month() {
        // A part on a term-level set: 240 over 24 months.
        let events = EventSet::from_years(vec![vec![]], 0)
            .unwrap()
            .with_times(vec![])
            .unwrap()
            .with_aggregate_part(&part(&[], vec![240.0]))
            .unwrap();
        let timeline = Timeline::new(events, 24).unwrap();
        let two = Exposure::uniform(2, 1).unwrap();
        let lod = timeline.years(CoverBasis::LossesOccurring, &two).unwrap();
        close(lod[0].aggregate_part(0).unwrap(), &[120.0]);
        close(lod[1].aggregate_part(0).unwrap(), &[120.0]);
        // Weight on the first month of each year alone.
        let mut january = [0.0; 12];
        january[0] = 1.0;
        let lod = timeline
            .with_aggregate_pattern(&january)
            .unwrap()
            .years(CoverBasis::LossesOccurring, &two)
            .unwrap();
        close(lod[0].aggregate_part(0).unwrap(), &[120.0]);
    }

    #[test]
    fn a_quota_share_cedes_the_part_and_an_xl_does_not() {
        let (one, two) = (
            book(50, 7)
                .with_aggregate_part(&part(&[], vec![30.0; 50]))
                .unwrap(),
            book(50, 8)
                .with_aggregate_part(&part(&[], vec![40.0; 50]))
                .unwrap(),
        );
        let timeline = Timeline::from_years(&[one.clone(), two.clone()]).unwrap();
        let exposure = Exposure::uniform(2, 12).unwrap();
        let tower = Tower::new(vec![
            Layer::xol("XL", 20.0, 10.0).unwrap(),
            Layer::quota_share("QS", 0.5).unwrap(),
        ])
        .unwrap()
        .cover_basis(CoverBasis::LossesOccurring);
        let r = tower.apply_timeline(&timeline, &exposure).unwrap();
        // Under LOD each contract year is its accident year, part and all.
        for (y, own) in [(1, &one), (2, &two)] {
            let alone = tower.apply(own).unwrap();
            let year = |kind: &str, layer: &str| column(&r, kind, layer, y);
            let plain = |kind: &str, layer: &str| {
                use prospicio_prob::KeyValue;
                let key = vec![KeyValue::from(kind), KeyValue::from(layer)];
                prospicio_prob::Empirical::draws(&alone.marginal(&key).unwrap()).to_vec()
            };
            close(&year("ceded", "QS"), &plain("ceded", "QS"));
            close(&year("ceded", "XL"), &plain("ceded", "XL"));
        }
        // Without the part the XL cedes the same and the QS less.
        let bare = Timeline::from_years(&[book(50, 7), book(50, 8)]).unwrap();
        let r0 = tower.apply_timeline(&bare, &exposure).unwrap();
        close(
            &column(&r0, "ceded", "XL", 1),
            &column(&r, "ceded", "XL", 1),
        );
        let qs: Vec<f64> = column(&r0, "ceded", "QS", 2)
            .iter()
            .map(|x| x + 20.0)
            .collect();
        close(&qs, &column(&r, "ceded", "QS", 2));
    }

    #[test]
    fn aggregate_pattern_errors() {
        let timeline = Timeline::from_years(&[book(3, 1)
            .with_aggregate_part(&part(&[], vec![1.0, 2.0, 3.0]))
            .unwrap()])
        .unwrap();
        assert!(timeline.clone().with_aggregate_pattern(&[1.0; 11]).is_err());
        assert!(timeline.clone().with_aggregate_pattern(&[0.0; 12]).is_err());
        let mut bad = [1.0; 12];
        bad[3] = -1.0;
        assert!(timeline.clone().with_aggregate_pattern(&bad).is_err());
        bad[3] = f64::NAN;
        assert!(timeline.clone().with_aggregate_pattern(&bad).is_err());
        // A one-month period whose month has no weight.
        let short = Timeline::new(
            EventSet::from_years(vec![vec![]], 0)
                .unwrap()
                .with_times(vec![])
                .unwrap()
                .with_aggregate_part(&part(&[], vec![5.0]))
                .unwrap(),
            1,
        )
        .unwrap();
        let mut late = [0.0; 12];
        late[11] = 1.0;
        assert!(short.with_aggregate_pattern(&late).is_err());
        assert_eq!(timeline.aggregate_pattern(), &[1.0; 12]);
    }

    fn column(result: &PredictiveDistribution, kind: &str, layer: &str, year: i64) -> Vec<f64> {
        use prospicio_prob::KeyValue;
        let key = vec![
            KeyValue::from(kind),
            KeyValue::from(layer),
            KeyValue::Int(year),
        ];
        prospicio_prob::Empirical::draws(&result.marginal(&key).unwrap()).to_vec()
    }

    fn straddling(shares: (f64, f64)) -> (Timeline, Exposure) {
        let events = EventSet::from_years(vec![vec![100.0]], 0)
            .unwrap()
            .with_times(vec![13.5 / 24.0])
            .unwrap();
        let mut rows = vec![vec![0.0; 24]; 2];
        rows[0][13] = shares.0;
        rows[1][13] = shares.1;
        (
            Timeline::new(events, 24).unwrap(),
            Exposure::new(rows).unwrap(),
        )
    }

    #[test]
    fn interlocking_cuts_each_years_limit_by_its_share() {
        let xl = Layer::xol("XL", 50.0, 20.0)
            .unwrap()
            .paid_reinstatements(10.0, vec![1.0])
            .unwrap();
        let tower = Tower::new(vec![xl])
            .unwrap()
            .cover_basis(CoverBasis::RisksAttaching);
        let (timeline, exposure) = straddling((0.25, 0.75));
        let at = |t: &Tower, kind: &str| -> Vec<f64> {
            let r = t.apply_timeline(&timeline, &exposure).unwrap();
            (1..=2).map(|y| column(&r, kind, "XL", y)[0]).collect()
        };
        assert_eq!(at(&tower, "ceded"), [5.0, 50.0]);
        let on = tower.clone().interlocking(true);
        // Limits 12.5 and 37.5; the retention of 20 stays whole.
        assert_eq!(at(&on, "ceded"), [5.0, 37.5]);
        // Each year's reinstatement premium is on its own recoveries.
        assert_eq!(at(&on, "reinstatement_premium"), [1.0, 7.5]);

        // Shares of 0.2 and 0.6: the rest is outside the term, so the
        // factors are still 0.25 and 0.75 of the programme's 80.
        let (timeline, exposure) = straddling((0.2, 0.6));
        let r = on.apply_timeline(&timeline, &exposure).unwrap();
        assert_eq!(column(&r, "ceded", "XL", 1)[0], 0.0);
        assert_eq!(column(&r, "ceded", "XL", 2)[0], 37.5);
    }

    #[test]
    fn interlocking_changes_nothing_when_one_year_takes_each_event() {
        let book = book(400, 9);
        let timeline = Timeline::new(book, 24).unwrap();
        let tower = Tower::new(vec![
            Layer::xol("XL", 15.0, 10.0)
                .unwrap()
                .paid_reinstatements(3.0, vec![1.0])
                .unwrap(),
        ])
        .unwrap();
        let same = |basis: CoverBasis, exposure: &Exposure| {
            let off = tower
                .clone()
                .cover_basis(basis)
                .apply_timeline(&timeline, exposure)
                .unwrap();
            let on = tower
                .clone()
                .cover_basis(basis)
                .interlocking(true)
                .apply_timeline(&timeline, exposure)
                .unwrap();
            assert_eq!(off.draw_matrix(), on.draw_matrix());
        };
        let uniform = Exposure::uniform(2, 12).unwrap();
        same(CoverBasis::LossesOccurring, &uniform);
        // Each month written by one year only, at 60%.
        let rows = (0..2)
            .map(|u| {
                (0..24)
                    .map(|m| if m / 12 == u { 0.6 } else { 0.0 })
                    .collect()
            })
            .collect();
        same(CoverBasis::RisksAttaching, &Exposure::new(rows).unwrap());
    }

    #[test]
    fn an_interlocked_event_uses_at_most_one_limit() {
        // One event a path, at a uniform time over 36 months.
        let n = 300;
        let losses: Vec<Vec<f64>> = (0..n).map(|j| vec![5.0 + j as f64]).collect();
        let times: Vec<f64> = (0..n).map(|j| (j as f64 + 0.5) / n as f64).collect();
        let events = EventSet::from_years(losses.clone(), 0)
            .unwrap()
            .with_times(times)
            .unwrap();
        let timeline = Timeline::new(events, 36).unwrap();
        let exposure = Exposure::uniform(2, 12).unwrap();
        let tower = Tower::new(vec![Layer::xol("XL", 50.0, 20.0).unwrap()])
            .unwrap()
            .cover_basis(CoverBasis::RisksAttaching);
        let off = tower.apply_timeline(&timeline, &exposure).unwrap();
        let on = tower
            .clone()
            .interlocking(true)
            .apply_timeline(&timeline, &exposure)
            .unwrap();
        let sum = |r: &PredictiveDistribution| -> Vec<f64> {
            let (a, b) = (column(r, "ceded", "XL", 1), column(r, "ceded", "XL", 2));
            a.iter().zip(&b).map(|(x, y)| x + y).collect()
        };
        let (off, on) = (sum(&off), sum(&on));
        assert!(on.iter().all(|&c| c <= 50.0 + 1e-9));
        assert!(on.iter().zip(&off).all(|(a, b)| a <= &(b + 1e-9)));
        // Some events do straddle the years and lose limit.
        assert!(off.iter().any(|&c| c > 50.0 + 1e-9));
    }
}
