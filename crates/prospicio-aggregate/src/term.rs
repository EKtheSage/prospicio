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
//! blocks), or years that share a trend or a parameter drawn once per path.

use prospicio_core::{Error, Result};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};

use crate::monte_carlo::EventSet;
use crate::reinsurance::{Carried, Columns, Tower};

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
                (e.events(j), e.sums_insured(j), e.times(j))
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
            |t, j| (std::slice::from_ref(&totals[t][j]), None, None),
            provenance,
        )
    }

    fn apply_paths<'a>(
        &self,
        n_years: usize,
        n_sims: usize,
        year: impl Fn(usize, usize) -> (&'a [f64], Option<&'a [f64]>, Option<&'a [f64]>),
        base: Provenance,
    ) -> Result<PredictiveDistribution> {
        let columns = Columns::new(self);
        let width = columns.len(self);
        let mut draws = Vec::with_capacity(n_sims * n_years * width);
        for j in 0..n_sims {
            let mut carried = vec![Carried::default(); 2 * self.layers.len()];
            for t in 0..n_years {
                let (losses, si, times) = year(t, j);
                self.push_year(&columns, &mut draws, losses, si, times, t, &mut carried);
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
}
