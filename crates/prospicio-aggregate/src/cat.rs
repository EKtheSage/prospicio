//! Catastrophe model output: an event loss table (ELT) with secondary
//! uncertainty, its occurrence exceedance (OEP) curve, and simulated years
//! of events to apply a tower to. See `docs/design/cat-inputs.md`.
//!
//! Each row is an event with an annual `rate`, a mean loss, standard
//! deviations of its loss (independent and correlated) and an exposure,
//! the most it can lose. Events occur independently as Poisson processes,
//! so the year's events are a Poisson count with mean `Λ = Σ rate`, each
//! one the event `i` with probability `rate_i / Λ`; the ELT is a compound
//! Poisson model whose severity is the mixture of the events' losses.
//!
//! Secondary uncertainty: an event's loss, given that it occurs, is a beta
//! distribution on `[0, exposure]` with the row's mean and standard
//! deviation `σ = σ_independent + σ_correlated` (the two summed, not in
//! quadrature, as catastrophe models report them for one portfolio; the
//! split matters only when tables are combined by event, which is not done
//! here). A row with `σ = 0` loses its mean exactly.
//!
//! There is no file reading here: a table enters as columns, from a data
//! frame in Python or R.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_math::roots::bisect_log;
use prospicio_prob::{Beta, Counting, Distribution, Poisson};

use crate::EventSet;

/// The loss of one event given that it occurs.
#[derive(Debug, Clone, Copy, PartialEq)]
enum EventLoss {
    /// No secondary uncertainty: the mean, exactly.
    Point(f64),
    /// A beta distribution on `[0, exposure]`.
    Beta(Beta),
}

impl EventLoss {
    fn survival(&self, x: f64) -> f64 {
        match self {
            Self::Point(m) => {
                if *m > x {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Beta(b) => 1.0 - b.cdf(x),
        }
    }

    fn quantile(&self, u: f64) -> f64 {
        match self {
            Self::Point(m) => *m,
            Self::Beta(b) => b
                .quantile(u)
                .expect("a beta quantile at u in (0, 1) exists"),
        }
    }
}

/// An event loss table; see the [module](self).
#[derive(Debug, Clone, PartialEq)]
pub struct EventLossTable {
    rates: Vec<f64>,
    means: Vec<f64>,
    sds: Vec<f64>,
    exposures: Vec<f64>,
    losses: Vec<EventLoss>,
    /// Cumulative rates, for picking an event.
    cumulative: Vec<f64>,
}

impl EventLossTable {
    /// From one row per event: its annual rate, mean loss, independent and
    /// correlated standard deviations, and exposure.
    ///
    /// A row with a positive standard deviation needs a beta distribution
    /// on `[0, exposure]` with that mean and standard deviation, which
    /// exists only when `σ² < m (E − m)`; a row outside it is an error that
    /// names it, rather than being clamped.
    ///
    /// ```
    /// use prospicio_aggregate::cat::EventLossTable;
    ///
    /// let elt = EventLossTable::new(
    ///     &[0.01, 0.05],
    ///     &[800.0, 200.0],
    ///     &[200.0, 0.0],
    ///     &[100.0, 0.0],
    ///     &[2000.0, 500.0],
    /// )
    /// .unwrap();
    /// assert!((elt.rate() - 0.06).abs() < 1e-15);
    /// // The 200 event always exceeds 150; the other most of the time.
    /// let f = elt.exceedance_frequency(150.0);
    /// assert!(f > 0.05 && f < 0.06);
    /// assert!((elt.oep(150.0) - (1.0 - (-f).exp())).abs() < 1e-15);
    /// ```
    pub fn new(
        rates: &[f64],
        means: &[f64],
        sd_independent: &[f64],
        sd_correlated: &[f64],
        exposures: &[f64],
    ) -> Result<Self> {
        let n = rates.len();
        if n == 0 {
            return Err(Error::Data(
                "an event loss table needs at least one event".into(),
            ));
        }
        for (name, col) in [
            ("means", means),
            ("sd_independent", sd_independent),
            ("sd_correlated", sd_correlated),
            ("exposures", exposures),
        ] {
            if col.len() != n {
                return Err(Error::Data(format!("{} {name} for {n} rates", col.len())));
            }
        }
        let mut losses = Vec::with_capacity(n);
        let mut sds = Vec::with_capacity(n);
        for i in 0..n {
            let (rate, m, e) = (rates[i], means[i], exposures[i]);
            let sd = sd_independent[i] + sd_correlated[i];
            let row = |reason: &str| Error::Data(format!("event row {i}: {reason}"));
            if !(rate.is_finite() && rate >= 0.0) {
                return Err(row("the rate must be finite and non-negative"));
            }
            if !(e.is_finite() && e > 0.0) {
                return Err(row("the exposure must be finite and positive"));
            }
            if !(m.is_finite() && m >= 0.0 && m <= e) {
                return Err(row("the mean must be in [0, exposure]"));
            }
            if !(sd_independent[i] >= 0.0 && sd_correlated[i] >= 0.0 && sd.is_finite()) {
                return Err(row("standard deviations must be finite and non-negative"));
            }
            let loss = if sd == 0.0 {
                EventLoss::Point(m)
            } else {
                // m/E and (σ/E)² must allow a beta: v < m(1 − m).
                let (mu, v) = (m / e, (sd / e).powi(2));
                let room = mu * (1.0 - mu);
                if v >= room {
                    return Err(row(&format!(
                        "no beta on [0, {e}] has mean {m} and standard deviation {sd}; \
                         the standard deviation must be below {}",
                        e * room.sqrt()
                    )));
                }
                let k = room / v - 1.0;
                EventLoss::Beta(Beta::new(mu * k, (1.0 - mu) * k, e)?)
            };
            losses.push(loss);
            sds.push(sd);
        }
        let mut total = 0.0;
        let cumulative: Vec<f64> = rates
            .iter()
            .map(|r| {
                total += r;
                total
            })
            .collect();
        if total <= 0.0 {
            return Err(Error::Data("the rates sum to zero".into()));
        }
        Ok(Self {
            rates: rates.to_vec(),
            means: means.to_vec(),
            sds,
            exposures: exposures.to_vec(),
            losses,
            cumulative,
        })
    }

    /// The number of events.
    pub fn len(&self) -> usize {
        self.rates.len()
    }

    /// Whether the table has no events (never, once built).
    pub fn is_empty(&self) -> bool {
        self.rates.is_empty()
    }

    /// The first three raw moments of the loss of one event given that it
    /// occurs, the mixture of the rows' losses by rate:
    /// `Σ rate_e E[X_e^k] / Λ`, exact for the beta rows. The annual
    /// cumulants are `Λ` times them.
    pub fn severity_moments(&self) -> [f64; 3] {
        // A beta on [0, E] with mean m and sd σ: E[X²] = σ² + m², and
        // E[X³] = E³ a (a + 1)(a + 2) / ((a + b)(a + b + 1)(a + b + 2)).
        let (mut m1, mut m2, mut m3) = (0.0, 0.0, 0.0);
        let total = self.rate();
        for i in 0..self.rates.len() {
            let (w, m, sd, e) = (
                self.rates[i] / total,
                self.means[i],
                self.sds[i],
                self.exposures[i],
            );
            let third = if sd == 0.0 {
                m * m * m
            } else {
                let (mu, v) = (m / e, (sd / e).powi(2));
                let k = mu * (1.0 - mu) / v - 1.0;
                let (a, b) = (mu * k, (1.0 - mu) * k);
                e * e * e * a * (a + 1.0) * (a + 2.0) / ((a + b) * (a + b + 1.0) * (a + b + 2.0))
            };
            m1 += w * m;
            m2 += w * (sd * sd + m * m);
            m3 += w * third;
        }
        [m1, m2, m3]
    }

    /// `Λ = Σ rate`, the mean number of events a year.
    pub fn rate(&self) -> f64 {
        self.cumulative[self.cumulative.len() - 1]
    }

    /// The expected annual loss, `Σ rate × mean`.
    pub fn expected_annual_loss(&self) -> f64 {
        self.rates.iter().zip(&self.means).map(|(r, m)| r * m).sum()
    }

    /// Each event's total standard deviation, `σ_independent + σ_correlated`.
    pub fn standard_deviations(&self) -> &[f64] {
        &self.sds
    }

    /// The exposures.
    pub fn exposures(&self) -> &[f64] {
        &self.exposures
    }

    /// The annual frequency of events losing more than `x`,
    /// `Σ rate_i P(L_i > x)`.
    pub fn exceedance_frequency(&self, x: f64) -> f64 {
        self.rates
            .iter()
            .zip(&self.losses)
            .map(|(r, l)| r * l.survival(x))
            .sum()
    }

    /// The occurrence exceedance probability: the chance that some event in
    /// a year loses more than `x`, `1 − exp(−Σ rate_i P(L_i > x))`.
    pub fn oep(&self, x: f64) -> f64 {
        -(-self.exceedance_frequency(x)).exp_m1()
    }

    /// The occurrence PML at annual probability `p`: the loss `x` with
    /// `OEP(x) = p`, for `p` in `(0, 1 − exp(−Λ))`, found by bisection (to
    /// within the curve's steps where events have no secondary
    /// uncertainty: the smallest `x` with `OEP(x) <= p`).
    pub fn occurrence_pml(&self, p: f64) -> Result<f64> {
        let ceiling = -(-self.rate()).exp_m1();
        if !(p > 0.0 && p < ceiling) {
            return Err(Error::InvalidParameter {
                name: "p",
                value: p,
                reason: "must be in (0, 1 − exp(−Λ))",
            });
        }
        let top = self.exposures.iter().copied().fold(0.0, f64::max);
        Ok(bisect_log(f64::MIN_POSITIVE, top, |x| self.oep(x) > p))
    }

    /// Simulates `n_years` years of events, each loss with the index of
    /// its event (row) in the table, in the same order.
    ///
    /// Year `i` draws from `StreamRng::new(seed, i)`: the count from a
    /// Poisson with mean `Λ`, then for each event a uniform that picks the
    /// row by its rate and one for its loss by inverse transform, so any
    /// year can be replayed alone, whatever the table's size.
    ///
    /// ```
    /// use prospicio_aggregate::cat::EventLossTable;
    ///
    /// let elt = EventLossTable::new(&[0.5], &[100.0], &[0.0], &[0.0], &[100.0]).unwrap();
    /// let (events, ids) = elt.simulate(4, 7).unwrap();
    /// assert_eq!(events.n_sims(), 4);
    /// assert!(ids.iter().flatten().all(|&i| i == 0));
    /// assert!((0..4).all(|y| events.events(y).iter().all(|&x| x == 100.0)));
    /// ```
    pub fn simulate(&self, n_years: usize, seed: u64) -> Result<(EventSet, Vec<Vec<usize>>)> {
        if n_years == 0 {
            return Err(Error::InvalidParameter {
                name: "n_years",
                value: 0.0,
                reason: "must be positive",
            });
        }
        let count = Poisson::new(self.rate())?;
        let total = self.rate();
        let mut years = Vec::with_capacity(n_years);
        let mut ids = Vec::with_capacity(n_years);
        for i in 0..n_years as u64 {
            let mut rng = StreamRng::new(seed, i);
            let n = count
                .quantile(rng.next_open01())
                .expect("next_open01 is always in (0, 1)") as usize;
            let mut losses = Vec::with_capacity(n);
            let mut events = Vec::with_capacity(n);
            for _ in 0..n {
                let target = rng.next_open01() * total;
                let k = self
                    .cumulative
                    .partition_point(|&c| c <= target)
                    .min(self.len() - 1);
                losses.push(self.losses[k].quantile(rng.next_open01()));
                events.push(k);
            }
            years.push(losses);
            ids.push(events);
        }
        let set = EventSet::from_years(years, seed)?;
        Ok((set, ids))
    }
}

/// One row of [`UnitEventLossTables`]: event, unit, rate, mean, independent
/// sd, correlated sd, exposure.
type UnitRow = (usize, usize, f64, f64, f64, f64, f64);

/// Event loss tables of several units that share events (#777), `aggregate`
/// 2.1.0's `elt unit A [...] elt unit B [...]`: one row per event and
/// unit, the same event on several units being one physical occurrence.
///
/// [`rolled`](Self::rolled) is the portfolio's table, one row per event
/// with its units' rows rolled up as `aggregate` does by default
/// (`[ELT-Shared-Event-Rollup]`): the event's rate (its rows must agree to
/// `1e-12`), the means, exposures and correlated standard deviations
/// summed, the independent ones in quadrature. [`unit`](Self::unit) is one
/// unit's own table. A unit's rows for one event must not repeat.
///
/// ```
/// use prospicio_aggregate::cat::UnitEventLossTables;
///
/// let t = UnitEventLossTables::new(
///     &["1", "2", "1", "3"],
///     &["A", "A", "B", "B"],
///     &[0.01, 0.05, 0.01, 0.02],
///     &[800.0, 200.0, 300.0, 100.0],
///     &[200.0, 0.0, 50.0, 30.0],
///     &[100.0, 0.0, 40.0, 0.0],
///     &[2000.0, 500.0, 1000.0, 400.0],
/// )
/// .unwrap();
/// assert_eq!((t.units(), t.shared_events()), (&["A".to_string(), "B".into()][..], 1));
/// let all = t.rolled().unwrap();
/// // Event 1 is one occurrence: rate 0.01, not 0.02.
/// assert!((all.rate() - 0.08).abs() < 1e-15);
/// assert!((all.expected_annual_loss() - (0.01 * 1100.0 + 0.05 * 200.0 + 0.02 * 100.0)).abs() < 1e-12);
/// assert!((t.unit("B").unwrap().rate() - 0.03).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct UnitEventLossTables {
    units: Vec<String>,
    events: Vec<String>,
    /// Each row: (event index, unit index, rate, mean, sdi, sdc, exposure).
    rows: Vec<UnitRow>,
}

impl UnitEventLossTables {
    /// From a long table, one row per event and unit: the event's
    /// identifier, the unit's tag, and the row's rate, mean, independent and
    /// correlated standard deviations and exposure, as
    /// [`EventLossTable::new`] takes them. Units and events keep the order of
    /// their first appearance.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        event_ids: &[&str],
        units: &[&str],
        rates: &[f64],
        means: &[f64],
        sd_independent: &[f64],
        sd_correlated: &[f64],
        exposures: &[f64],
    ) -> Result<Self> {
        let n = event_ids.len();
        for (name, len) in [
            ("units", units.len()),
            ("rates", rates.len()),
            ("means", means.len()),
            ("sd_independent", sd_independent.len()),
            ("sd_correlated", sd_correlated.len()),
            ("exposures", exposures.len()),
        ] {
            if len != n {
                return Err(Error::Data(format!("{len} {name} for {n} event ids")));
            }
        }
        if n == 0 {
            return Err(Error::Data(
                "an event loss table needs at least one row".into(),
            ));
        }
        let index = |names: &mut Vec<String>, x: &str| match names.iter().position(|v| v == x) {
            Some(i) => i,
            None => {
                names.push(x.to_string());
                names.len() - 1
            }
        };
        let (mut unit_names, mut event_names) = (Vec::new(), Vec::new());
        let mut rows = Vec::with_capacity(n);
        let mut seen = std::collections::HashSet::new();
        let mut event_rate: Vec<f64> = Vec::new();
        for i in 0..n {
            let e = index(&mut event_names, event_ids[i]);
            let u = index(&mut unit_names, units[i]);
            if !seen.insert((e, u)) {
                return Err(Error::Data(format!(
                    "event {:?} has two rows for unit {:?}",
                    event_ids[i], units[i]
                )));
            }
            if e == event_rate.len() {
                event_rate.push(rates[i]);
            } else if (rates[i] - event_rate[e]).abs() > 1e-12 * event_rate[e].abs() {
                return Err(Error::Data(format!(
                    "event {:?} has rate {} on unit {:?} and {} on its first row; one event has one \
                     rate",
                    event_ids[i], rates[i], units[i], event_rate[e]
                )));
            }
            rows.push((
                e,
                u,
                event_rate[e],
                means[i],
                sd_independent[i],
                sd_correlated[i],
                exposures[i],
            ));
        }
        let t = Self {
            units: unit_names,
            events: event_names,
            rows,
        };
        // Each unit's table must be valid on its own (a beta for every row).
        for u in 0..t.units.len() {
            t.table_of(|r| r.1 == u)?;
        }
        Ok(t)
    }

    /// The units' tags, in order of first appearance.
    pub fn units(&self) -> &[String] {
        &self.units
    }

    /// The events' identifiers, in order of first appearance.
    pub fn events(&self) -> &[String] {
        &self.events
    }

    /// The number of events with rows on two or more units.
    pub fn shared_events(&self) -> usize {
        let mut count = vec![0usize; self.events.len()];
        for r in &self.rows {
            count[r.0] += 1;
        }
        count.iter().filter(|&&c| c > 1).count()
    }

    fn table_of(&self, keep: impl Fn(&UnitRow) -> bool) -> Result<EventLossTable> {
        let rows: Vec<_> = self.rows.iter().filter(|r| keep(r)).collect();
        EventLossTable::new(
            &rows.iter().map(|r| r.2).collect::<Vec<_>>(),
            &rows.iter().map(|r| r.3).collect::<Vec<_>>(),
            &rows.iter().map(|r| r.4).collect::<Vec<_>>(),
            &rows.iter().map(|r| r.5).collect::<Vec<_>>(),
            &rows.iter().map(|r| r.6).collect::<Vec<_>>(),
        )
    }

    /// One unit's table.
    pub fn unit(&self, unit: &str) -> Result<EventLossTable> {
        let u = self.units.iter().position(|x| x == unit).ok_or_else(|| {
            Error::Data(format!("no unit {unit:?}; the units are {:?}", self.units))
        })?;
        self.table_of(|r| r.1 == u)
    }

    /// The portfolio's table, one row per event, its units' rows rolled up:
    /// the event's rate, the means, exposures and correlated standard
    /// deviations summed, the independent ones in quadrature.
    pub fn rolled(&self) -> Result<EventLossTable> {
        let c = self.rolled_columns();
        EventLossTable::new(&c[0], &c[1], &c[2], &c[3], &c[4])
    }

    /// The rolled table's columns, one row per event in order of
    /// [`events`](Self::events): rate, mean, independent and correlated
    /// standard deviations, exposure.
    pub fn rolled_columns(&self) -> [Vec<f64>; 5] {
        let n = self.events.len();
        let (mut rate, mut mean, mut sdi2, mut sdc, mut exposure) = (
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
        );
        for &(e, _, r, m, si, sc, x) in &self.rows {
            rate[e] = r;
            mean[e] += m;
            sdi2[e] += si * si;
            sdc[e] += sc;
            exposure[e] += x;
        }
        let sdi: Vec<f64> = sdi2.iter().map(|v| v.sqrt()).collect();
        [rate, mean, sdi, sdc, exposure]
    }

    /// One unit's columns, as [`rolled_columns`](Self::rolled_columns), with
    /// its events' identifiers.
    pub fn unit_columns(&self, unit: &str) -> Result<(Vec<String>, [Vec<f64>; 5])> {
        let u = self.units.iter().position(|x| x == unit).ok_or_else(|| {
            Error::Data(format!("no unit {unit:?}; the units are {:?}", self.units))
        })?;
        let rows: Vec<_> = self.rows.iter().filter(|r| r.1 == u).collect();
        let col = |f: fn(&UnitRow) -> f64| rows.iter().map(|r| f(r)).collect::<Vec<_>>();
        Ok((
            rows.iter().map(|r| self.events[r.0].clone()).collect(),
            [
                col(|r| r.2),
                col(|r| r.3),
                col(|r| r.4),
                col(|r| r.5),
                col(|r| r.6),
            ],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> EventLossTable {
        EventLossTable::new(
            &[0.02, 0.1, 0.3],
            &[5000.0, 1000.0, 200.0],
            &[1500.0, 300.0, 0.0],
            &[500.0, 100.0, 0.0],
            &[20_000.0, 4000.0, 800.0],
        )
        .unwrap()
    }

    #[test]
    fn a_row_needs_a_feasible_beta() {
        let e = EventLossTable::new(&[0.1], &[50.0], &[60.0], &[0.0], &[100.0]);
        let msg = format!("{}", e.unwrap_err());
        assert!(msg.contains("row 0"), "{msg}");
        assert!(EventLossTable::new(&[0.1], &[150.0], &[0.0], &[0.0], &[100.0]).is_err());
        assert!(EventLossTable::new(&[0.0], &[50.0], &[0.0], &[0.0], &[100.0]).is_err());
        assert!(EventLossTable::new(&[], &[], &[], &[], &[]).is_err());
        assert!(EventLossTable::new(&[0.1], &[50.0], &[1.0], &[0.0], &[]).is_err());
    }

    #[test]
    fn each_event_s_beta_has_the_row_s_mean_and_deviation() {
        let elt = table();
        for i in 0..2 {
            let EventLoss::Beta(b) = elt.losses[i] else {
                panic!("row {i} has secondary uncertainty")
            };
            assert!((b.mean() - elt.means[i]).abs() < 1e-9 * elt.means[i]);
            let sd = b.variance().sqrt();
            assert!((sd - elt.sds[i]).abs() < 1e-9 * elt.sds[i], "{sd}");
        }
        assert_eq!(elt.losses[2], EventLoss::Point(200.0));
        assert!((elt.expected_annual_loss() - (100.0 + 100.0 + 60.0)).abs() < 1e-12);
    }

    #[test]
    fn the_pml_inverts_the_oep() {
        let elt = table();
        for p in [0.002, 0.01, 0.05] {
            let x = elt.occurrence_pml(p).unwrap();
            assert!((elt.oep(x) - p).abs() < 1e-9, "{p}: {}", elt.oep(x));
        }
        assert!(elt.occurrence_pml(0.5).is_err());
    }
}
