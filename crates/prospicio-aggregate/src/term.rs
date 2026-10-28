//! Multi-year contract terms: a tower applied to consecutive years of the
//! same simulated path, so that a profit commission or sliding scale can
//! carry deficits and credits from one year into the next
//! ([`crate::Layer::profit_commission_carry_forward`],
//! [`crate::Layer::sliding_scale_carry_forward`]).
//!
//! Simulation `j` of the result is one term: year 1 of it is simulation `j`
//! of the first year's losses, year 2 simulation `j` of the second's, and
//! so on. The years' losses are whatever the caller gives: independent
//! years ([`EventSet::term_years`] splits one simulation into consecutive
//! blocks), years whose totals have a rank correlation
//! ([`EventSet::correlated_years`], calibrated to a target spread of the
//! term's mean by [`EventSet::calibrate_year_correlation`]), or years that
//! share a trend or a parameter drawn once per path.

use prospicio_core::{Error, Result};
use prospicio_prob::portfolio::CorrelationKind;
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};

use crate::monte_carlo::EventSet;
use crate::reinsurance::{Columns, ScopeView, Tower};

impl Tower {
    /// Applies the tower over a contract term of `years.len()` years, each
    /// an [`EventSet`] with the same number of simulations; simulation `j`
    /// of every year belongs to the same path.
    ///
    /// The result has dimensions `["kind", "layer", "year"]`: each of
    /// [`apply`](Self::apply)'s components once per year, keyed by the
    /// year `1..=T` (year 1's components first). Every year's ceded and net
    /// loss, reinstatement and swing premium is what `apply` gives for that
    /// year alone; a profit commission or sliding scale that carries
    /// deficits or credits forward reads what the path's earlier years
    /// left. `select("year", ...)` takes one year, `aggregate(&["kind",
    /// "layer"])` totals each component over the term. The years' labels
    /// ([`EventSet::with_labels`]) are not split out here; `apply` on one
    /// year splits by them.
    ///
    /// ```
    /// use prospicio_aggregate::{EventSet, Layer, Tower};
    /// use prospicio_prob::KeyValue;
    ///
    /// // Two paths of two years each.
    /// let year1 = EventSet::from_years(vec![vec![120.0], vec![40.0]], 0).unwrap();
    /// let year2 = EventSet::from_years(vec![vec![40.0], vec![40.0]], 0).unwrap();
    /// let qs = Layer::quota_share("QS", 1.0).unwrap()
    ///     .premium_rate(1.0, 100.0).unwrap()
    ///     .profit_commission(0.25, 0.1).unwrap()
    ///     .profit_commission_carry_forward(None).unwrap();
    /// let term = Tower::new(vec![qs]).unwrap().apply_term(&[year1, year2]).unwrap();
    /// let pc = |year: i64| {
    ///     let key = vec![KeyValue::from("profit_commission"), KeyValue::from("QS"), KeyValue::Int(year)];
    ///     prospicio_prob::Empirical::draws(&term.marginal(&key).unwrap()).to_vec()
    /// };
    /// // Path 1 carries a deficit of 30 into year 2: 25% of (50 - 30).
    /// assert_eq!(pc(1), [0.0, 12.5]);
    /// assert_eq!(pc(2), [5.0, 12.5]);
    /// ```
    pub fn apply_term(&self, years: &[EventSet]) -> Result<PredictiveDistribution> {
        self.apply_term_with(years, None)
    }

    /// [`apply_term`](Self::apply_term) with `factors[t]` scaling the
    /// limit of each of year `t`'s losses, in the order of its simulations
    /// and their events (interlocking, [`Tower::interlocking`]).
    pub(crate) fn apply_term_with(
        &self,
        years: &[EventSet],
        factors: Option<&[Vec<f64>]>,
    ) -> Result<PredictiveDistribution> {
        let first = years
            .first()
            .ok_or_else(|| Error::Data("a term needs at least one year".into()))?;
        let n_sims = first.n_sims();
        for (t, events) in years.iter().enumerate() {
            if events.n_sims() != n_sims {
                return Err(Error::Data(format!(
                    "year {} has {} simulations, year 1 has {n_sims}",
                    t + 1,
                    events.n_sims()
                )));
            }
            if self.needs_sums_insured() && !events.has_sums_insured() {
                return Err(Error::Data(format!(
                    "a surplus treaty needs each event's sum insured (year {})",
                    t + 1
                )));
            }
            if self.needs_times() && !events.has_times() {
                return Err(Error::Data(format!(
                    "reinstatements pro rata as to time need each event's time (year {})",
                    t + 1
                )));
            }
        }
        // Each year's scoped layers over its own label combinations.
        let masks = years
            .iter()
            .map(|e| self.scope_masks(e))
            .collect::<Result<Vec<_>>>()?;
        let mut provenance = first.provenance("reinsurance_term");
        for (t, events) in years.iter().enumerate().skip(1) {
            provenance =
                provenance.param(format!("year:{}:seed", t + 1), events.seed().to_string());
        }
        self.apply_paths(
            years.len(),
            n_sims,
            |t, j| {
                let e = &years[t];
                let losses = e.events(j);
                let lf = factors.map(|f| &f[t][e.offset(j)..e.offset(j) + losses.len()]);
                let scope = masks[t].as_deref().map(|masks| ScopeView {
                    aggregate: e.aggregate_part_indices().unwrap_or(&[]),
                    events: e.label_indices(j).expect("a scope needs labels"),
                    masks,
                });
                (
                    e.aggregate_part(j).unwrap_or(&[]),
                    losses,
                    e.sums_insured(j),
                    e.times(j),
                    lf,
                    scope,
                )
            },
            provenance,
        )
    }

    /// [`apply_term`](Self::apply_term) on aggregate losses: year `t` of
    /// path `j` is simulation `j`'s total of `years[t]`, as in
    /// [`apply_aggregate`](Self::apply_aggregate).
    pub fn apply_aggregate_term(
        &self,
        years: &[PredictiveDistribution],
    ) -> Result<PredictiveDistribution> {
        if self.needs_sums_insured() || self.needs_times() {
            return Err(Error::Data(
                "an aggregate loss has no sums insured or event times".into(),
            ));
        }
        self.refuse_scope("an aggregate loss")?;
        let first = years
            .first()
            .ok_or_else(|| Error::Data("a term needs at least one year".into()))?;
        let totals: Vec<Vec<f64>> = years
            .iter()
            .map(|pd| prospicio_prob::Empirical::draws(pd.total()).to_vec())
            .collect();
        let n_sims = totals[0].len();
        if let Some(t) = totals.iter().position(|y| y.len() != n_sims) {
            return Err(Error::Data(format!(
                "year {} has {} simulations, year 1 has {n_sims}",
                t + 1,
                totals[t].len()
            )));
        }
        let source = first.provenance();
        let provenance = Provenance::new("reinsurance_term")
            .version("prospicio-aggregate", env!("CARGO_PKG_VERSION"))
            .param("loss", format!("total of {}", source.model))
            .draws_from(source);
        self.apply_paths(
            years.len(),
            n_sims,
            |t, j| {
                (
                    &[][..],
                    std::slice::from_ref(&totals[t][j]),
                    None,
                    None,
                    None,
                    None,
                )
            },
            provenance,
        )
    }

    fn apply_paths<'a>(
        &self,
        n_years: usize,
        n_sims: usize,
        year: impl Fn(usize, usize) -> YearInputs<'a>,
        base: Provenance,
    ) -> Result<PredictiveDistribution> {
        let columns = Columns::new(self);
        let width = columns.len(self);
        let mut draws = Vec::with_capacity(n_sims * n_years * width);
        for j in 0..n_sims {
            let mut carried = self.carries();
            for t in 0..n_years {
                let (aggregate, losses, si, times, lf, scope) = year(t, j);
                self.push_year(
                    &columns,
                    &mut draws,
                    aggregate,
                    losses,
                    si,
                    times,
                    lf,
                    scope,
                    t,
                    &mut carried,
                );
            }
        }
        let keys = columns.keys(self);
        let components: Vec<ComponentKey> = (0..n_years)
            .flat_map(|t| {
                keys.iter().map(move |k| {
                    let mut k = k.clone();
                    k.push(KeyValue::Int(t as i64 + 1));
                    k
                })
            })
            .collect();
        let provenance = self
            .terms_provenance(base)
            .param("term_years", n_years.to_string());
        PredictiveDistribution::from_draws(
            vec!["kind".into(), "layer".into(), "year".into()],
            components,
            draws,
            provenance,
        )
    }
}

/// One year of one path: the aggregate part, the losses, their sums
/// insured, times and limit factors.
type YearInputs<'a> = (
    &'a [f64],
    &'a [f64],
    Option<&'a [f64]>,
    Option<&'a [f64]>,
    Option<&'a [f64]>,
    Option<ScopeView<'a>>,
);

impl EventSet {
    /// Splits the simulations into a term of `n_years` consecutive blocks:
    /// year `t` (from 0) holds simulations `t × n` to `(t + 1) × n − 1`,
    /// with `n = n_sims / n_years`, so path `j`'s year `t` is simulation
    /// `t × n + j` and replays alone as before. Sums insured and times go
    /// with their losses. Fails unless `n_years` divides the simulations.
    ///
    /// ```
    /// use prospicio_aggregate::simulate_events;
    /// use prospicio_prob::{Distribution, Lognormal, Poisson};
    ///
    /// let events = simulate_events(
    ///     &Poisson::new(2.0).unwrap(),
    ///     &Lognormal::from_mean_cv(100.0, 1.0).unwrap(),
    ///     3_000,
    ///     7,
    /// )
    /// .unwrap();
    /// let term = events.term_years(3).unwrap();
    /// assert_eq!(term.len(), 3);
    /// assert_eq!(term[2].events(5), events.events(2_005));
    /// ```
    pub fn term_years(&self, n_years: usize) -> Result<Vec<EventSet>> {
        let n_sims = self.n_sims();
        if n_years == 0 || n_sims % n_years != 0 {
            return Err(Error::Data(format!(
                "{n_years} years do not divide {n_sims} simulations"
            )));
        }
        let n = n_sims / n_years;
        Ok((0..n_years)
            .map(|t| self.slice(t * n..(t + 1) * n))
            .collect())
    }
}

/// Where [`EventSet::correlated_years`] takes each year's simulations from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum YearDraws {
    /// Every year is a reordering of all the simulations: each year has
    /// exactly the one-year distribution, so a term's mean has exactly the
    /// one-year mean, and only the pairing of the years changes.
    #[default]
    Same,
    /// Year `t` is block `t` of [`EventSet::term_years`]: the years are
    /// different draws, each `n_sims / n_years` of them.
    Blocks,
}

/// A calibration from [`EventSet::calibrate_year_correlation`].
#[derive(Debug, Clone, PartialEq)]
pub struct YearCalibration {
    /// The correlation between every pair of years, of the kind asked for.
    pub correlation: f64,
    /// The ratio it achieves on the simulations: the standard deviation of
    /// the mean of a term's yearly totals over that of one year's total.
    pub ratio: f64,
    /// The term's years under it.
    pub years: Vec<EventSet>,
}

/// The correlation between every pair of `n_years` years whose totals have
/// the same spread for which the mean of a term's yearly totals has
/// `ratio` times one year's standard deviation:
/// `ρ = (T r² − 1) / (T − 1)`, from `sd(mean) / sd = √((1 + (T − 1) ρ) / T)`.
/// It is Pearson's correlation of the totals: a start for
/// [`EventSet::correlated_years`], which imposes a rank correlation, and
/// [`EventSet::calibrate_year_correlation`] hits the ratio on the
/// simulations themselves.
///
/// ```
/// use prospicio_aggregate::term::year_correlation_for_ratio;
///
/// // Independent years: sd(mean) = sd / √T.
/// assert!(year_correlation_for_ratio(3, 1.0 / 3f64.sqrt()).unwrap().abs() < 1e-15);
/// assert_eq!(year_correlation_for_ratio(3, 1.0).unwrap(), 1.0);
/// ```
pub fn year_correlation_for_ratio(n_years: usize, ratio: f64) -> Result<f64> {
    if n_years < 2 {
        return Err(Error::Data("a term needs at least two years".into()));
    }
    if !(0.0..=1.0).contains(&ratio) {
        return Err(Error::Data(format!(
            "the ratio of the term's spread to one year's must be in [0, 1], got {ratio}"
        )));
    }
    let t = n_years as f64;
    Ok((t * ratio * ratio - 1.0) / (t - 1.0))
}

impl EventSet {
    /// A term of `n_years` years whose yearly totals (each year's losses
    /// and aggregate part) have the correlation `correlation`, imposed by
    /// Iman–Conover as [`PredictiveDistribution::reorder_groups_indexed`]
    /// does: one value for every pair of years, or a full `n_years ×
    /// n_years` matrix, row-major, of the kind `kind` (`Spearman` for a
    /// rank correlation of the totals). Each year's simulations move as
    /// whole years ([`permute`](Self::permute)), so every year keeps its
    /// events, times, sums insured, labels and aggregate part, and the
    /// years go to [`Tower::apply_term`] as they come. `draws` says whether
    /// every year reorders all the simulations or takes its own block of
    /// them. The same `seed` gives the same years.
    ///
    /// ```
    /// use prospicio_aggregate::simulate_events;
    /// use prospicio_aggregate::term::YearDraws;
    /// use prospicio_prob::portfolio::CorrelationKind;
    /// use prospicio_prob::{Lognormal, Poisson};
    ///
    /// let events = simulate_events(
    ///     &Poisson::new(2.0).unwrap(),
    ///     &Lognormal::from_mean_cv(100.0, 1.0).unwrap(),
    ///     4_000,
    ///     7,
    /// )
    /// .unwrap();
    /// let years = events
    ///     .correlated_years(3, &[0.5], CorrelationKind::Spearman, YearDraws::Same, 11)
    ///     .unwrap();
    /// // Each year is a reordering of the same simulations.
    /// let total = |e: &prospicio_aggregate::EventSet| {
    ///     (0..e.n_sims()).map(|i| e.events(i).iter().sum::<f64>()).sum::<f64>()
    /// };
    /// assert!((total(&years[2]) - total(&events)).abs() < 1e-6 * total(&events));
    /// ```
    pub fn correlated_years(
        &self,
        n_years: usize,
        correlation: &[f64],
        kind: CorrelationKind,
        draws: YearDraws,
        seed: u64,
    ) -> Result<Vec<EventSet>> {
        if n_years < 2 {
            return Err(Error::Data(
                "correlated years need at least two years".into(),
            ));
        }
        let matrix = match correlation {
            [rho] => (0..n_years * n_years)
                .map(|k| {
                    if k / n_years == k % n_years {
                        1.0
                    } else {
                        *rho
                    }
                })
                .collect::<Vec<_>>(),
            m if m.len() == n_years * n_years => m.to_vec(),
            m => {
                return Err(Error::Data(format!(
                    "the correlation is one value or a {n_years} × {n_years} matrix, got {} values",
                    m.len()
                )));
            }
        };
        let sources = match draws {
            YearDraws::Same => vec![self.clone(); n_years],
            YearDraws::Blocks => self.term_years(n_years)?,
        };
        let totals: Vec<Vec<f64>> = sources.iter().map(year_totals).collect();
        let n = totals[0].len();
        let pd = PredictiveDistribution::from_draws(
            vec!["year".into()],
            (1..=n_years as i64)
                .map(|t| vec![KeyValue::Int(t)])
                .collect(),
            (0..n)
                .flat_map(|i| totals.iter().map(move |c| c[i]))
                .collect(),
            Provenance::new("correlated_years"),
        )?;
        let (_, index) = pd.reorder_groups_indexed("year", &matrix, kind, seed)?;
        sources
            .iter()
            .zip(&index)
            .map(|(events, index)| events.permute(index))
            .collect()
    }

    /// The correlation between every pair of `n_years` years for which
    /// [`correlated_years`](Self::correlated_years) gives a term whose mean
    /// yearly total has `ratio` times the standard deviation of one year's
    /// total, found by bisection on the simulations with the same `kind`,
    /// `draws` and `seed`; with the ratio it achieves and the years. A
    /// target between independent years (`1 / √T`) and identical ones (1)
    /// asks for positive dependence; one below `1 / √T`, negative.
    ///
    /// A loss ratio is a total over a premium, so with the same premium in
    /// each year the ratio of the term's loss ratio's spread to one year's
    /// is the same ratio.
    ///
    /// ```
    /// use prospicio_aggregate::simulate_events;
    /// use prospicio_aggregate::term::YearDraws;
    /// use prospicio_prob::portfolio::CorrelationKind;
    /// use prospicio_prob::{Lognormal, Poisson};
    ///
    /// let events = simulate_events(
    ///     &Poisson::new(2.0).unwrap(),
    ///     &Lognormal::from_mean_cv(100.0, 1.0).unwrap(),
    ///     4_000,
    ///     7,
    /// )
    /// .unwrap();
    /// let fit = events
    ///     .calibrate_year_correlation(3, 0.75, CorrelationKind::Spearman, YearDraws::Same, 11)
    ///     .unwrap();
    /// assert!((fit.ratio - 0.75).abs() < 1e-3);
    /// assert!(fit.correlation > 0.0 && fit.correlation < 1.0);
    /// ```
    pub fn calibrate_year_correlation(
        &self,
        n_years: usize,
        ratio: f64,
        kind: CorrelationKind,
        draws: YearDraws,
        seed: u64,
    ) -> Result<YearCalibration> {
        year_correlation_for_ratio(n_years, ratio)?;
        let achieved = |rho: f64| -> Result<(f64, Vec<EventSet>)> {
            let years = self.correlated_years(n_years, &[rho], kind, draws, seed)?;
            Ok((term_ratio(&years), years))
        };
        // The equicorrelation matrix of normal scores is positive definite
        // on (-1 / (T - 1), 1); stay a little inside, on the kind's scale.
        let edge = 1e-6;
        let floor = -1.0 / (n_years as f64 - 1.0) + edge;
        let mut lo = match kind {
            CorrelationKind::Pearson => floor,
            CorrelationKind::Spearman => 6.0 / std::f64::consts::PI * (floor / 2.0).asin(),
        };
        let mut hi = 1.0 - edge;
        let (r_lo, low) = achieved(lo)?;
        let (r_hi, high) = achieved(hi)?;
        if ratio <= r_lo {
            return Ok(YearCalibration {
                correlation: lo,
                ratio: r_lo,
                years: low,
            });
        }
        if ratio >= r_hi {
            return Ok(YearCalibration {
                correlation: hi,
                ratio: r_hi,
                years: high,
            });
        }
        let mut best = (hi, r_hi, high);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let (r, years) = achieved(mid)?;
            if (r - ratio).abs() < (best.1 - ratio).abs() {
                best = (mid, r, years);
            }
            if r < ratio {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < 1e-9 {
                break;
            }
        }
        Ok(YearCalibration {
            correlation: best.0,
            ratio: best.1,
            years: best.2,
        })
    }
}

/// Each year's total: its losses and its aggregate part.
fn year_totals(events: &EventSet) -> Vec<f64> {
    (0..events.n_sims())
        .map(|i| {
            events.events(i).iter().sum::<f64>()
                + events.aggregate_part(i).map_or(0.0, |a| a.iter().sum())
        })
        .collect()
}

fn sd(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt()
}

/// The standard deviation of the mean of a term's yearly totals over the
/// average of the years' own.
fn term_ratio(years: &[EventSet]) -> f64 {
    let totals: Vec<Vec<f64>> = years.iter().map(year_totals).collect();
    let t = totals.len() as f64;
    let mean: Vec<f64> = (0..totals[0].len())
        .map(|i| totals.iter().map(|c| c[i]).sum::<f64>() / t)
        .collect();
    let one = totals.iter().map(|c| sd(c)).sum::<f64>() / t;
    sd(&mean) / one
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Layer, simulate_events};
    use prospicio_prob::{Empirical, Lognormal, Poisson};

    fn pc_layer(years: Option<u32>) -> Layer {
        Layer::quota_share("QS", 1.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .profit_commission(0.5, 0.2)
            .unwrap()
            .profit_commission_carry_forward(years)
            .unwrap()
    }

    fn slide_layer(deficit: bool, credit: bool, years: Option<u32>) -> Layer {
        Layer::quota_share("QS", 1.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .sliding_scale(vec![(0.4, 0.5), (0.2, 0.7)])
            .unwrap()
            .sliding_scale_carry_forward(deficit, credit, years)
            .unwrap()
    }

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-12, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn profit_commission_deficits_by_hand() {
        // Threshold 80, share 50%. Results: -40 (deficit 40), +10, +50, -5, +30.
        let ceded = [120.0, 70.0, 30.0, 85.0, 50.0];
        // To extinction: 40 left after year 2's 10 is 30; year 3's 50
        // covers it and pays 50% of 20; year 4 a deficit of 5; year 5 pays
        // 50% of 25.
        close(
            &pc_layer(None).profit_commissions_over(&ceded),
            &[0.0, 0.0, 10.0, 0.0, 12.5],
        );
        // One year: year 1's 40 can be used in year 2 only (30 of it is
        // still left and then dropped); year 3 pays 50% of 50.
        close(
            &pc_layer(Some(1)).profit_commissions_over(&ceded),
            &[0.0, 0.0, 25.0, 0.0, 12.5],
        );
        // Without a carry, each year alone.
        let plain = Layer::quota_share("QS", 1.0)
            .unwrap()
            .premium_rate(1.0, 100.0)
            .unwrap()
            .profit_commission(0.5, 0.2)
            .unwrap();
        close(
            &plain.profit_commissions_over(&ceded),
            &[0.0, 5.0, 25.0, 0.0, 15.0],
        );
    }

    #[test]
    fn deficits_are_used_oldest_first() {
        // Deficits of 10 (year 1) and 20 (year 2); year 3 earns 15, which
        // uses year 1's 10 and 5 of year 2's. With a two-year limit, the
        // 15 left from year 2 is still usable in year 4, which earns 15.
        let ceded = [90.0, 100.0, 65.0, 65.0];
        close(
            &pc_layer(Some(2)).profit_commissions_over(&ceded),
            &[0.0, 0.0, 0.0, 0.0],
        );
        // A one-year limit: year 1's 10 is gone by year 3, so year 3's 15
        // all goes to year 2's 20; year 4 cannot use the 5 left of it.
        close(
            &pc_layer(Some(1)).profit_commissions_over(&ceded),
            &[0.0, 0.0, 0.0, 7.5],
        );
    }

    #[test]
    fn sliding_scale_deficits_and_credits_by_hand() {
        // 40% at a 50% loss ratio to 20% at 70%, premium 100.
        let ceded = [90.0, 40.0, 30.0, 60.0];
        // Deficits only: year 1 carries 20; year 2 reads 60 (30%) and
        // carries nothing; year 3 reads 30 (40%), credit dropped; year 4 60.
        close(
            &slide_layer(true, false, None).ceding_commissions_over(&ceded),
            &[20.0, 30.0, 40.0, 30.0],
        );
        // Credits only: year 1 is 20%, nothing carried; year 2 reads 40
        // (40%), carrying a credit of 10; year 3 reads 20 (40%), carrying
        // 30; year 4 reads 30: 40%.
        close(
            &slide_layer(false, true, None).ceding_commissions_over(&ceded),
            &[20.0, 40.0, 40.0, 40.0],
        );
        // Both, with a one-year limit: year 3's credit of 20 reaches year 4
        // alone (year 2 carried nothing), so year 4 reads 40: 40%.
        close(
            &slide_layer(true, true, Some(1)).ceding_commissions_over(&ceded),
            &[20.0, 30.0, 40.0, 40.0],
        );
    }

    fn term_events() -> Vec<EventSet> {
        simulate_events(
            &Poisson::new(3.0).unwrap(),
            &Lognormal::from_mean_cv(10.0, 1.5).unwrap(),
            4 * 2_000,
            11,
        )
        .unwrap()
        .term_years(4)
        .unwrap()
    }

    fn component(pd: &PredictiveDistribution, kind: &str, year: i64) -> Vec<f64> {
        let key = vec![
            KeyValue::from(kind),
            KeyValue::from("QS"),
            KeyValue::Int(year),
        ];
        Empirical::draws(&pd.marginal(&key).unwrap()).to_vec()
    }

    #[test]
    fn each_year_matches_apply_without_a_carry() {
        let years = term_events();
        let layer = Layer::quota_share("QS", 0.8)
            .unwrap()
            .premium_rate(1.0, 35.0)
            .unwrap()
            .sliding_scale(vec![(0.35, 0.55), (0.2, 0.75)])
            .unwrap()
            .profit_commission(0.2, 0.3)
            .unwrap();
        let tower = Tower::new(vec![layer]).unwrap();
        let term = tower.apply_term(&years).unwrap();
        assert_eq!(term.n_components(), 4 * 5);
        for (t, events) in years.iter().enumerate() {
            let alone = tower.apply(events).unwrap();
            let sel = term.select("year", &[KeyValue::Int(t as i64 + 1)]).unwrap();
            for key in alone.components() {
                let mut k = key.clone();
                k.push(KeyValue::Int(t as i64 + 1));
                let a = Empirical::draws(&alone.marginal(key).unwrap()).to_vec();
                let b = Empirical::draws(&sel.marginal(&k).unwrap()).to_vec();
                assert_eq!(a, b, "{key:?}");
            }
        }
    }

    #[test]
    fn a_scoped_layer_carries_what_it_carries_on_the_filtered_years() {
        let year = |shift: f64| {
            EventSet::from_years(
                vec![vec![20.0 + shift, 15.0], vec![8.0, 30.0, 12.0 + shift]],
                4,
            )
            .unwrap()
            .with_labels(vec![(
                "peril".into(),
                ["quake", "wind", "wind", "flood", "wind"]
                    .map(String::from)
                    .to_vec(),
            )])
            .unwrap()
        };
        let years = vec![year(0.0), year(5.0)];
        let layer = Layer::xol("XL", 10.0, 5.0)
            .unwrap()
            .premium_rate(1.0, 12.0)
            .unwrap()
            .profit_commission(0.3, 0.2)
            .unwrap()
            .profit_commission_carry_forward(None)
            .unwrap();
        let scoped = Tower::new(vec![layer.clone().scope("peril", &["wind"]).unwrap()])
            .unwrap()
            .apply_term(&years)
            .unwrap()
            .aggregate(&["kind", "layer", "year"])
            .unwrap();
        let wind: Vec<EventSet> = years
            .iter()
            .map(|y| y.filter("peril", &["wind"]).unwrap())
            .collect();
        let alone = Tower::new(vec![layer])
            .unwrap()
            .apply_term(&wind)
            .unwrap()
            .aggregate(&["kind", "layer", "year"])
            .unwrap();
        let mut compared = 0;
        for k in alone.components() {
            if matches!(k[0].to_string().as_str(), "gross" | "net") {
                continue;
            }
            assert_eq!(
                Empirical::draws(&scoped.marginal(k).unwrap()),
                Empirical::draws(&alone.marginal(k).unwrap()),
                "{k:?}"
            );
            compared += 1;
        }
        assert!(compared >= 4);
    }

    #[test]
    fn deficits_to_extinction_telescope_on_every_path() {
        let years = term_events();
        let (share, allowance, premium) = (0.3, 0.25, 30.0);
        let layer = Layer::quota_share("QS", 1.0)
            .unwrap()
            .premium_rate(1.0, premium)
            .unwrap()
            .profit_commission(share, allowance)
            .unwrap()
            .profit_commission_carry_forward(None)
            .unwrap();
        let term = Tower::new(vec![layer.clone()])
            .unwrap()
            .apply_term(&years)
            .unwrap();
        let ceded: Vec<Vec<f64>> = (1..=4).map(|t| component(&term, "ceded", t)).collect();
        let pc: Vec<Vec<f64>> = (1..=4)
            .map(|t| component(&term, "profit_commission", t))
            .collect();
        let threshold = premium * (1.0 - allowance);
        let mut carried_some = 0;
        for j in 0..ceded[0].len() {
            let path: Vec<f64> = ceded.iter().map(|c| c[j]).collect();
            // The deficit left at the end, by its own recursion.
            let d = path
                .iter()
                .fold(0.0_f64, |d, c| (d + c - threshold).max(0.0));
            let total: f64 = pc.iter().map(|p| p[j]).sum();
            let sum_result: f64 = path.iter().map(|c| threshold - c).sum();
            assert!((total - share * (sum_result + d)).abs() < 1e-9);
            assert_eq!(
                layer.profit_commissions_over(&path),
                pc.iter().map(|p| p[j]).collect::<Vec<_>>()
            );
            let alone: f64 = path.iter().map(|&c| layer.profit_commission_for(c)).sum();
            assert!(total <= alone + 1e-9);
            if total < alone - 1e-9 {
                carried_some += 1;
            }
        }
        // The carry bites on a good share of paths.
        assert!(carried_some > 200, "{carried_some}");
    }

    #[test]
    fn aggregate_term_and_checks() {
        let pd = |v: Vec<f64>| {
            PredictiveDistribution::from_draws(vec![], vec![vec![]], v, Provenance::new("test"))
                .unwrap()
        };
        let tower = Tower::new(vec![pc_layer(None)]).unwrap();
        let term = tower
            .apply_aggregate_term(&[pd(vec![120.0, 30.0]), pd(vec![30.0, 30.0])])
            .unwrap();
        assert_eq!(component(&term, "profit_commission", 1), [0.0, 25.0]);
        assert_eq!(component(&term, "profit_commission", 2), [5.0, 25.0]);
        assert!(tower.apply_aggregate_term(&[]).is_err());
        assert!(
            tower
                .apply_aggregate_term(&[pd(vec![1.0]), pd(vec![1.0, 2.0])])
                .is_err()
        );
        let years = term_events();
        assert!(
            tower
                .apply_term(&[years[0].clone(), years[1].slice(0..10)])
                .is_err()
        );
        assert!(years[0].term_years(7).is_err());
        assert!(years[0].term_years(0).is_err());
    }

    #[test]
    fn builders_refuse_what_they_cannot_carry() {
        let qs = || {
            Layer::quota_share("QS", 1.0)
                .unwrap()
                .premium_rate(1.0, 100.0)
                .unwrap()
        };
        assert!(qs().profit_commission_carry_forward(None).is_err());
        assert!(qs().sliding_scale_carry_forward(true, false, None).is_err());
        assert!(
            qs().ceding_commission(0.3)
                .unwrap()
                .sliding_scale_carry_forward(true, false, None)
                .is_err()
        );
        let slid = qs().sliding_scale(vec![(0.3, 0.6)]).unwrap();
        assert!(
            slid.clone()
                .sliding_scale_carry_forward(false, false, None)
                .is_err()
        );
        assert!(
            slid.clone()
                .sliding_scale_carry_forward(true, false, Some(0))
                .is_err()
        );
        // A flat commission replaces the scale and its carry.
        let flat = slid
            .sliding_scale_carry_forward(true, true, Some(2))
            .unwrap()
            .ceding_commission(0.3)
            .unwrap();
        assert_eq!(flat.sliding_scale_carry, None);
    }

    fn ranks(x: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
        let mut r = vec![0.0; x.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    }

    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let cov: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
        cov / (sd(x) * sd(y) * n)
    }

    fn book(n: usize) -> EventSet {
        simulate_events(
            &Poisson::new(3.0).unwrap(),
            &Lognormal::from_mean_cv(100.0, 1.5).unwrap(),
            n,
            5,
        )
        .unwrap()
    }

    fn sorted(mut x: Vec<f64>) -> Vec<f64> {
        x.sort_by(f64::total_cmp);
        x
    }

    #[test]
    fn correlated_years_reorder_the_same_draws_to_the_target_rank_correlation() {
        let events = book(5_000);
        let years = events
            .correlated_years(3, &[0.4], CorrelationKind::Spearman, YearDraws::Same, 3)
            .unwrap();
        let one = sorted(year_totals(&events));
        let totals: Vec<Vec<f64>> = years.iter().map(year_totals).collect();
        for c in &totals {
            assert_eq!(sorted(c.clone()), one);
        }
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let rho = pearson(&ranks(&totals[a]), &ranks(&totals[b]));
            assert!((rho - 0.4).abs() < 0.03, "{a} {b}: {rho}");
        }
        // Reproducible by seed.
        let again = events
            .correlated_years(3, &[0.4], CorrelationKind::Spearman, YearDraws::Same, 3)
            .unwrap();
        assert_eq!(again, years);
    }

    #[test]
    fn correlated_years_take_a_matrix_and_blocks() {
        let events = book(3_000);
        let m = [1.0, 0.6, 0.0, 0.6, 1.0, 0.3, 0.0, 0.3, 1.0];
        let years = events
            .correlated_years(3, &m, CorrelationKind::Spearman, YearDraws::Blocks, 9)
            .unwrap();
        let blocks = events.term_years(3).unwrap();
        let totals: Vec<Vec<f64>> = years.iter().map(year_totals).collect();
        for (c, b) in totals.iter().zip(&blocks) {
            assert_eq!(c.len(), 1_000);
            assert_eq!(sorted(c.clone()), sorted(year_totals(b)));
        }
        for (a, b, want) in [(0, 1, 0.6), (0, 2, 0.0), (1, 2, 0.3)] {
            let rho = pearson(&ranks(&totals[a]), &ranks(&totals[b]));
            assert!((rho - want).abs() < 0.05, "{a} {b}: {rho}");
        }
        // The years go to a term as they come.
        let term = Tower::new(vec![pc_layer(None)])
            .unwrap()
            .apply_term(&years)
            .unwrap();
        assert_eq!(term.n_sims(), 1_000);
    }

    #[test]
    fn calibration_hits_the_term_spread() {
        let events = book(4_000);
        for (ratio, kind) in [
            (0.8, CorrelationKind::Spearman),
            (0.65, CorrelationKind::Pearson),
            (0.5, CorrelationKind::Spearman),
        ] {
            let fit = events
                .calibrate_year_correlation(3, ratio, kind, YearDraws::Same, 2)
                .unwrap();
            assert!((fit.ratio - ratio).abs() < 1e-3, "{ratio}: {}", fit.ratio);
            assert!((term_ratio(&fit.years) - fit.ratio).abs() < 1e-15);
            // Below 1 / √3, the years are negatively dependent.
            assert_eq!(fit.correlation < 0.0, ratio < 1.0 / 3f64.sqrt());
        }
        // On a near-normal book the closed form is close.
        let near_normal = simulate_events(
            &Poisson::new(200.0).unwrap(),
            &Lognormal::from_mean_cv(100.0, 0.2).unwrap(),
            4_000,
            5,
        )
        .unwrap();
        let fit = near_normal
            .calibrate_year_correlation(3, 0.8, CorrelationKind::Pearson, YearDraws::Same, 2)
            .unwrap();
        let closed = year_correlation_for_ratio(3, 0.8).unwrap();
        assert!(
            (fit.correlation - closed).abs() < 0.03,
            "{} {closed}",
            fit.correlation
        );
    }

    #[test]
    fn correlated_years_errors() {
        let events = book(100);
        let k = CorrelationKind::Spearman;
        assert!(
            events
                .correlated_years(1, &[0.5], k, YearDraws::Same, 0)
                .is_err()
        );
        assert!(
            events
                .correlated_years(3, &[0.5, 0.5], k, YearDraws::Same, 0)
                .is_err()
        );
        assert!(
            events
                .correlated_years(3, &[0.5], k, YearDraws::Blocks, 0)
                .is_err()
        );
        assert!(year_correlation_for_ratio(3, 1.2).is_err());
        assert!(year_correlation_for_ratio(1, 0.5).is_err());
    }
}
