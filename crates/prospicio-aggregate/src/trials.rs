//! How often a layer is attached and used up, counted on simulated years
//! (#581).

use prospicio_core::{Error, Result};

use crate::monte_carlo::EventSet;
use crate::reinsurance::Layer;

/// A layer's attachment and exhaustion over simulated years, from
/// [`Layer::trial_statistics`], each probability with its binomial
/// standard error `sqrt(p (1 − p) / n)` so that it can be set beside the
/// event loss table's per-occurrence figures
/// ([`occurrence_layer_statistics`](crate::layer_statistics::occurrence_layer_statistics)).
#[derive(Debug, Clone, PartialEq)]
pub struct TrialStatistics {
    /// The number of simulated years.
    pub n_years: usize,
    /// The share of years with at least one occurrence above the
    /// attachment (a recovery before annual terms).
    pub p_attach: f64,
    pub p_attach_se: f64,
    /// The share of years with an occurrence at or above attachment plus
    /// limit, that is one that uses a whole limit; 0 for an unlimited
    /// layer.
    pub p_exhaust_occurrence: f64,
    pub p_exhaust_occurrence_se: f64,
    /// The share of years in which the annual limit (the aggregate limit,
    /// or the limit times one plus the reinstatements) is used up; 0
    /// without one.
    pub p_exhaust_annual: f64,
    pub p_exhaust_annual_se: f64,
    /// The mean number of occurrences a year that reach the layer, with
    /// its standard error.
    pub occurrences_mean: f64,
    pub occurrences_se: f64,
    /// The mean ceded loss a year (the placed share, after annual terms).
    pub ceded_mean: f64,
    /// The mean ceded loss over the attached years; NaN when no year is
    /// attached.
    pub ceded_given_attach: f64,
}

fn binomial_se(p: f64, n: usize) -> f64 {
    (p * (1.0 - p) / n as f64).sqrt()
}

impl Layer {
    /// How often the layer is attached and exhausted in `events`' years,
    /// seeing the gross losses as the first stage of a tower does: each
    /// year's events (and its aggregate part, for a layer that
    /// [sees it](Layer::sees_aggregate_part), which counts toward the annual
    /// terms but is no occurrence).
    ///
    /// An occurrence attaches when its recovery before annual terms is
    /// positive and exhausts the occurrence limit when that recovery is the
    /// whole limit; the year exhausts the annual limit when its layer loss
    /// after the annual deductible and corridor reaches it. The ceded loss
    /// is that of [`Tower::apply`](crate::Tower::apply) for a one-layer
    /// tower.
    ///
    /// ```
    /// use prospicio_aggregate::{EventSet, Layer};
    ///
    /// // 10 xs 10 with one reinstatement: 20 a year at most.
    /// let layer = Layer::xol("10x10", 10.0, 10.0).unwrap().reinstatements(1).unwrap();
    /// let years = vec![vec![5.0], vec![12.0], vec![25.0], vec![25.0, 30.0]];
    /// let s = layer.trial_statistics(&EventSet::from_years(years, 0).unwrap()).unwrap();
    /// assert_eq!(s.p_attach, 0.75);
    /// assert_eq!(s.p_exhaust_occurrence, 0.5);
    /// assert_eq!(s.p_exhaust_annual, 0.25);
    /// assert_eq!(s.occurrences_mean, 1.0);
    /// assert_eq!(s.ceded_given_attach, (2.0 + 10.0 + 20.0) / 3.0);
    /// ```
    pub fn trial_statistics(&self, events: &EventSet) -> Result<TrialStatistics> {
        if self.needs_sums_insured() && !events.has_sums_insured() {
            return Err(Error::Data(
                "a surplus treaty needs each event's sum insured: use \
                 EventSet::with_sums_insured or simulate from a risk profile"
                    .into(),
            ));
        }
        let n = events.n_sims();
        let sees_part = self.sees_aggregate_part();
        let (mut attached, mut occ_full, mut annual_full) = (0usize, 0usize, 0usize);
        let (mut count_sum, mut count_sq) = (0.0, 0.0);
        let (mut ceded_sum, mut ceded_attached) = (0.0, 0.0);
        for i in 0..n {
            let losses = events.events(i);
            let si = events.sums_insured(i);
            let mut recovery = 0.0;
            if sees_part && let Some(part) = events.aggregate_part(i) {
                recovery += part
                    .iter()
                    .map(|&x| self.recovery_at(x, None, 1.0))
                    .sum::<f64>();
            }
            let (mut count, mut full) = (0usize, false);
            for r in self.recoveries(losses, si, None) {
                recovery += r;
                if r > 0.0 {
                    count += 1;
                    full |= self.limit.is_finite() && r >= self.limit;
                }
            }
            let layer_loss = self.after_terms(recovery);
            let ceded = self.share * layer_loss;
            ceded_sum += ceded;
            if count > 0 {
                attached += 1;
                ceded_attached += ceded;
            }
            occ_full += usize::from(full);
            annual_full += usize::from(
                self.aggregate_limit.is_finite()
                    && layer_loss >= self.aggregate_limit * (1.0 - 1e-12),
            );
            count_sum += count as f64;
            count_sq += (count * count) as f64;
        }
        let nf = n as f64;
        let share = |k: usize| k as f64 / nf;
        let (p_attach, p_occ, p_annual) = (share(attached), share(occ_full), share(annual_full));
        let occurrences_mean = count_sum / nf;
        let var = if n > 1 {
            ((count_sq - nf * occurrences_mean * occurrences_mean) / (nf - 1.0)).max(0.0)
        } else {
            0.0
        };
        Ok(TrialStatistics {
            n_years: n,
            p_attach,
            p_attach_se: binomial_se(p_attach, n),
            p_exhaust_occurrence: p_occ,
            p_exhaust_occurrence_se: binomial_se(p_occ, n),
            p_exhaust_annual: p_annual,
            p_exhaust_annual_se: binomial_se(p_annual, n),
            occurrences_mean,
            occurrences_se: (var / nf).sqrt(),
            ceded_mean: ceded_sum / nf,
            ceded_given_attach: if attached > 0 {
                ceded_attached / attached as f64
            } else {
                f64::NAN
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Tower, simulate_events};
    use prospicio_prob::{Distribution, Empirical, KeyValue, Lognormal, Poisson};

    #[test]
    fn counts_by_hand_with_a_deductible_a_share_and_a_surplus_of_years() {
        // 10 xs 10, 50% placed, an annual deductible of 3 and limit 15.
        let layer = Layer::xol("L", 10.0, 10.0)
            .unwrap()
            .share(0.5)
            .unwrap()
            .aggregate_deductible(3.0)
            .unwrap()
            .aggregate_limit(15.0)
            .unwrap();
        let years = vec![
            vec![],
            vec![12.0],             // recovery 2, after the AAD 0
            vec![20.0],             // 10 -> 7
            vec![20.0, 15.0, 30.0], // 10 + 5 + 10 = 25 -> 22 -> 15, used up
            vec![9.0, 10.0],        // nothing above 10
        ];
        let s = layer
            .trial_statistics(&EventSet::from_years(years, 0).unwrap())
            .unwrap();
        assert_eq!(s.n_years, 5);
        assert_eq!(s.p_attach, 0.6);
        assert_eq!(s.p_exhaust_occurrence, 0.4);
        assert_eq!(s.p_exhaust_annual, 0.2);
        assert_eq!(s.occurrences_mean, 5.0 / 5.0);
        assert_eq!(s.ceded_mean, 0.5 * (7.0 + 15.0) / 5.0);
        assert_eq!(s.ceded_given_attach, 0.5 * (7.0 + 15.0) / 3.0);
        assert!((s.p_attach_se - (0.6f64 * 0.4 / 5.0).sqrt()).abs() < 1e-15);
        // Counts 0, 1, 1, 3, 0: sample variance 1.5.
        assert!((s.occurrences_se - (1.5f64 / 5.0).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn an_unlimited_layer_never_exhausts_and_no_attached_year_gives_nan() {
        let layer = Layer::xol("L", f64::INFINITY, 100.0).unwrap();
        let events = EventSet::from_years(vec![vec![5.0], vec![50.0]], 0).unwrap();
        let s = layer.trial_statistics(&events).unwrap();
        assert_eq!(
            (s.p_attach, s.p_exhaust_occurrence, s.p_exhaust_annual),
            (0.0, 0.0, 0.0)
        );
        assert!(s.ceded_given_attach.is_nan());
    }

    #[test]
    fn a_stop_loss_counts_the_aggregate_part_toward_its_terms_but_not_as_an_occurrence() {
        let part = prospicio_prob::PredictiveDistribution::from_draws(
            vec![],
            vec![vec![]],
            vec![30.0, 80.0],
            prospicio_prob::Provenance::new("test"),
        )
        .unwrap();
        let events = EventSet::from_years(vec![vec![], vec![40.0]], 0)
            .unwrap()
            .with_aggregate_part(&part)
            .unwrap();
        let sl = Layer::stop_loss("SL", 50.0, 60.0).unwrap();
        let s = sl.trial_statistics(&events).unwrap();
        // Year 0: 30 retained; year 1: 120 -> 60 -> 50, used up.
        assert_eq!(s.p_attach, 0.5);
        assert_eq!(s.p_exhaust_annual, 0.5);
        assert_eq!(s.ceded_mean, 25.0);
        let tower = Tower::new(vec![sl]).unwrap().apply(&events).unwrap();
        let ceded = tower
            .marginal(&vec![KeyValue::from("ceded"), KeyValue::from("SL")])
            .unwrap();
        assert_eq!(Empirical::draws(&ceded), [0.0, 50.0]);
    }

    #[test]
    fn agrees_with_the_tower_and_with_the_poisson_closed_forms() {
        let (lambda, n) = (2.0, 200_000);
        let sev = Lognormal::from_mean_cv(3e6, 1.5).unwrap();
        let events = simulate_events(&Poisson::new(lambda).unwrap(), &sev, n, 11).unwrap();
        let layer = Layer::xol("5x5", 5e6, 5e6)
            .unwrap()
            .reinstatements(1)
            .unwrap();
        let s = layer.trial_statistics(&events).unwrap();
        // The ceded mean is the tower's.
        let tower = Tower::new(vec![layer]).unwrap().apply(&events).unwrap();
        let ceded = tower
            .marginal(&vec![KeyValue::from("ceded"), KeyValue::from("5x5")])
            .unwrap();
        let mean = Empirical::draws(&ceded).iter().sum::<f64>() / n as f64;
        assert!((s.ceded_mean - mean).abs() < 1e-9 * mean);
        // Excess counts are Poisson with mean λ S(a), so a year attaches
        // with probability 1 − exp(−λ S(a)).
        let (sa, sd) = (sev.survival(5e6), sev.survival(10e6));
        let within = |got: f64, want: f64, se: f64| (got - want).abs() < 4.0 * se;
        assert!(within(
            s.p_attach,
            1.0 - (-lambda * sa).exp(),
            s.p_attach_se
        ));
        assert!(within(
            s.p_exhaust_occurrence,
            1.0 - (-lambda * sd).exp(),
            s.p_exhaust_occurrence_se
        ));
        assert!(within(s.occurrences_mean, lambda * sa, s.occurrences_se));
        // Two limits used up need at least two full losses, or more partial.
        assert!(s.p_exhaust_annual > 0.0 && s.p_exhaust_annual < s.p_exhaust_occurrence);
    }

    #[test]
    fn a_surplus_needs_sums_insured() {
        let layer = Layer::surplus("S", 1e6, 4.0).unwrap();
        let events = EventSet::from_years(vec![vec![1.0]], 0).unwrap();
        assert!(layer.trial_statistics(&events).is_err());
    }
}
