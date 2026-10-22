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
//! ([`Timeline::years_with_sources`]). A tower's basis is set with
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
#[derive(Debug, Clone)]
pub struct Timeline {
    events: EventSet,
    months: usize,
}

impl Timeline {
    /// A timeline of `months` accident months from events whose times
    /// ([`EventSet::with_times`]) are fractions of the whole term. Fails
    /// without times, with an aggregate part (its split by accident month
    /// is not decided) or with no months.
    pub fn new(events: EventSet, months: usize) -> Result<Self> {
        if months == 0 {
            return Err(Error::Data("a timeline needs at least one month".into()));
        }
        if !events.has_times() {
            return Err(Error::Data(
                "a timeline needs each event's time: set times on the events".into(),
            ));
        }
        if events.has_aggregate_part() {
            return Err(Error::Data(
                "a timeline cannot carry an aggregate part: its split by accident month is not decided"
                    .into(),
            ));
        }
        Ok(Self { events, months })
    }

    /// A timeline stitched from one event set per accident year, each with
    /// times within its year and the same number of simulations: year
    /// `t`'s event at time `τ` is at `(t + τ) / T` of the term, in path
    /// `j` for its simulation `j`. The years can come from
    /// [`EventSet::term_years`] or [`EventSet::correlated_years`]. Labels
    /// and sums insured are kept as [`EventSet::merge`] keeps them.
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
        let events = merged.expect("at least one year");
        Self::new(events, MONTHS_PER_YEAR * years.len())
    }

    /// The events, with times as fractions of the term.
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
        let years = rows.iter().map(|r| self.events.rebuilt(r)).collect();
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
        self.apply_term(&timeline.years(basis, exposure)?)
    }
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
}
