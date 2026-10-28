//! Ultimate losses and loss ratios per simulation from a stochastic
//! reserving fit (`reserving-v02.md` decision 26, issue #602).
//!
//! A deal analysis turns each simulated year's ultimate loss into a loss
//! ratio on the program's subject premium, passes it through a tower
//! (`Tower::apply_annual` in `prospicio-aggregate`) and then a deal book or
//! the cash-flow engine (`DealBook`, `CashflowTerms::run` in
//! `prospicio-pricing`). The dependence between layers, and between the
//! reserve and the cash flows, survives only if every vector keeps the same
//! simulation at the same index. [`UltimateDraws`] gives those vectors from
//! any fit that returns per-simulation reserves, with that guarantee.
//!
//! # The order guarantee
//!
//! A fit's [`reserves`](UltimateDraws::reserves) are one
//! [`PredictiveDistribution`]: row `i` holds every origin's (and every
//! segment's) reserve in simulation `i`. Every method here reads row `i`
//! into index `i`:
//!
//! - index `i` of [`ultimate_draws`](UltimateDraws::ultimate_draws) is the
//!   same simulation for every origin, and the same simulation as index `i`
//!   of [`total_ultimate_draws`](UltimateDraws::total_ultimate_draws), the
//!   loss ratios, the fit's `reserves` and its other joint outputs (Meyers'
//!   `ultimates`, Compartmental's `paid` and `outstanding`);
//! - a fit is a function of its inputs and seed, so fitting again with the
//!   same seed gives the same draws in the same order (each simulation
//!   draws from its own stream, `StreamRng::new(seed, i)`, so the number of
//!   threads does not change it);
//! - [`ultimates`](UltimateDraws::ultimates) is a `PredictiveDistribution`
//!   with the reserves' dimensions and components, so `join`, `select`,
//!   `aggregate`, `reorder_groups` and the other operations on it move a
//!   row's components together: within a group the origins stay paired. A
//!   reorder changes which simulation row `i` of one group meets in another
//!   group; `reorder_groups_indexed` returns where each row came from.
//!
//! # The basis
//!
//! The ultimate is the latest observed value of the fitted column plus the
//! reserve draw, so it is on the column's basis: paid for a paid triangle,
//! incurred for an incurred one. Meyers' models simulate the cumulative
//! value at the oldest age directly and keep it as `ultimates`; their
//! ultimates are those draws (the reserves are those less the latest
//! value). Compartmental models simulate paid and outstanding together, and
//! their reserve is the paid at the square's last age less the latest paid,
//! so their ultimate is the paid at the square's last age (no tail past
//! it). An origin observed at its last age has a reserve of 0 and its
//! latest value as its ultimate.
//!
//! The one-year views (claims development results) are not ultimates and
//! do not implement the trait; their `fit` (the opening fit) does when it
//! is one of the fits listed below.
//!
//! ```
//! use prospicio_reserving::{
//!     DevelopmentColumn, Grain, Long, Month, OdpBootstrap, Triangle, UltimateDraws,
//! };
//! use prospicio_prob::KeyValue;
//!
//! let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
//! let tri = Triangle::from_long(&Long {
//!     keys: &[],
//!     origin: &origin,
//!     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
//!     values: &[("paid", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0])],
//!     origin_grain: Grain::Year,
//!     development_grain: Grain::Year,
//!     cumulative: true,
//! })?;
//! let fit = OdpBootstrap { n_sims: 1_000, seed: 1, ..Default::default() }.fit(&tri, "paid")?;
//! let origin_2022: Vec<KeyValue> = vec![2022.into()];
//! let ultimate = fit.ultimate_draws(&origin_2022)?;
//! let ratio = fit.loss_ratio_draws(&origin_2022, 200.0)?;
//! let reserve = fit.reserves.marginal(&origin_2022).unwrap();
//! for i in 0..ultimate.len() {
//!     // The same simulation at the same index.
//!     assert_eq!(ultimate[i], 120.0 + prospicio_prob::Empirical::draws(&reserve)[i]);
//!     assert_eq!(ratio[i], ultimate[i] / 200.0);
//! }
//! # Ok::<(), prospicio_reserving::Error>(())
//! ```

use prospicio_prob::{ComponentKey, Empirical, KeyValue, PredictiveDistribution};

use crate::bootstrap::{OdpBootstrapFit, OdpBootstrapFits};
use crate::error::{Error, Result};
use crate::mack_bootstrap::{MackBootstrapFit, MackBootstrapFits};
use crate::segments::{ReserveFit, SegmentFits};

/// Ultimate losses and loss ratios per simulation from a fit's reserve
/// draws, index `i` the same simulation across origins, segments and calls
/// (see the [module docs](self) for the guarantee and the basis).
///
/// Implemented by the fits that return per-simulation reserves:
/// [`OdpBootstrapFit`], [`OdpBootstrapFits`], [`MackBootstrapFit`],
/// [`MackBootstrapFits`] and, with the `bayes` feature, `BayesOdpFit`,
/// `BayesOdpFits`, `BayesBfFit`, `BayesBfNormalisedFit`, `MeyersFit`,
/// `MeyersFits`, `CompartmentalFit` and `CompartmentalFits`.
///
/// An origin is named by its component key in
/// [`reserves`](Self::reserves): `[origin]` for a fit of one segment,
/// `[key values..., origin]` over segments (`["Auto", 2021]`). A period
/// can be named by its label, as text or an integer (`"2021"`, `2021`,
/// `"2021Q3"`), as in [`PredictiveDistribution::component_index`].
pub trait UltimateDraws {
    /// The joint reserve draws, one component per (segment and) origin.
    fn reserves(&self) -> &PredictiveDistribution;

    /// Each component's latest observed value, in the order of
    /// [`reserves`](Self::reserves)' components.
    fn latest_values(&self) -> Vec<f64>;

    /// The joint ultimates: the reserves with each component's latest value
    /// added in every simulation, the same dimensions, components and rows,
    /// so the result can be joined, selected, aggregated or reordered with
    /// its rows kept whole.
    fn ultimates(&self) -> Result<PredictiveDistribution> {
        add_latest(self.reserves(), &self.latest_values())
    }

    /// One origin's ultimate in each simulation, in simulation order.
    ///
    /// Fails if no component of [`reserves`](Self::reserves) has the key.
    fn ultimate_draws(&self, origin: &[KeyValue]) -> Result<Vec<f64>> {
        let ultimates = self.ultimates()?;
        column(&ultimates, origin)
    }

    /// One origin's ultimate over `premium` in each simulation, in
    /// simulation order: the loss ratio a contract on the loss ratio reads.
    ///
    /// Fails if no component has the key or `premium` is not finite and
    /// positive.
    fn loss_ratio_draws(&self, origin: &[KeyValue], premium: f64) -> Result<Vec<f64>> {
        check_premium(premium)?;
        Ok(self
            .ultimate_draws(origin)?
            .into_iter()
            .map(|u| u / premium)
            .collect())
    }

    /// The ultimate summed over every component in each simulation (summed
    /// index by index), in simulation order. Use
    /// [`ultimates`](Self::ultimates) with `select` or `aggregate` for a
    /// total over some components only.
    fn total_ultimate_draws(&self) -> Result<Vec<f64>> {
        let ultimates = self.ultimates()?;
        Ok(Empirical::draws(ultimates.total()).to_vec())
    }

    /// [`total_ultimate_draws`](Self::total_ultimate_draws) over the total
    /// `premium`.
    ///
    /// Fails if `premium` is not finite and positive.
    fn total_loss_ratio_draws(&self, premium: f64) -> Result<Vec<f64>> {
        check_premium(premium)?;
        Ok(self
            .total_ultimate_draws()?
            .into_iter()
            .map(|u| u / premium)
            .collect())
    }
}

/// `reserves` with `latest[j]` added to component `j` in every simulation.
pub(crate) fn add_latest(
    reserves: &PredictiveDistribution,
    latest: &[f64],
) -> Result<PredictiveDistribution> {
    let n = reserves.n_components();
    if latest.len() != n {
        return Err(Error::LengthMismatch {
            column: "latest".into(),
            expected: n,
            found: latest.len(),
        });
    }
    let draws = reserves
        .draw_matrix()
        .chunks_exact(n)
        .flat_map(|row| row.iter().zip(latest).map(|(r, l)| r + l))
        .collect();
    Ok(PredictiveDistribution::from_draws(
        reserves.dims().to_vec(),
        reserves.components().to_vec(),
        draws,
        reserves.provenance().clone().param("output", "ultimates"),
    )?)
}

/// The column of `pd` keyed `key`, in simulation order.
pub(crate) fn column(pd: &PredictiveDistribution, key: &[KeyValue]) -> Result<Vec<f64>> {
    let key: ComponentKey = key.to_vec();
    let sampled = pd.marginal(&key).ok_or_else(|| Error::UnknownKeyValue {
        key: pd.dims().join(", "),
        value: key
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", "),
    })?;
    Ok(Empirical::draws(&sampled).to_vec())
}

fn check_premium(premium: f64) -> Result<()> {
    if premium.is_finite() && premium > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidSetting {
            name: "premium",
            value: premium,
            expected: "finite and positive",
        })
    }
}

/// The latest values of every segment's origins in turn, the component
/// order of a joint distribution over segments.
pub(crate) fn segment_latest<T: ReserveFit>(segments: &SegmentFits<T>) -> Vec<f64> {
    segments
        .fits
        .iter()
        .flat_map(|f| f.chain_ladder().latest.iter().copied())
        .collect()
}

impl UltimateDraws for OdpBootstrapFit {
    fn reserves(&self) -> &PredictiveDistribution {
        &self.reserves
    }

    fn latest_values(&self) -> Vec<f64> {
        self.chain_ladder.latest.clone()
    }
}

impl UltimateDraws for OdpBootstrapFits {
    fn reserves(&self) -> &PredictiveDistribution {
        &self.reserves
    }

    fn latest_values(&self) -> Vec<f64> {
        segment_latest(&self.segments)
    }
}

impl UltimateDraws for MackBootstrapFit {
    fn reserves(&self) -> &PredictiveDistribution {
        &self.reserves
    }

    fn latest_values(&self) -> Vec<f64> {
        self.mack.chain_ladder.latest.clone()
    }
}

impl UltimateDraws for MackBootstrapFits {
    fn reserves(&self) -> &PredictiveDistribution {
        &self.reserves
    }

    fn latest_values(&self) -> Vec<f64> {
        segment_latest(&self.segments)
    }
}

#[cfg(feature = "bayes")]
mod bayes_fits {
    use prospicio_prob::PredictiveDistribution;

    use super::{UltimateDraws, segment_latest};
    use crate::bayes::{
        BayesBfFit, BayesBfNormalisedFit, BayesOdpFit, BayesOdpFits, CompartmentalFit,
        CompartmentalFits, MeyersFit, MeyersFits,
    };
    use crate::error::Result;

    impl UltimateDraws for BayesOdpFit {
        fn reserves(&self) -> &PredictiveDistribution {
            &self.reserves
        }

        fn latest_values(&self) -> Vec<f64> {
            self.chain_ladder.latest.clone()
        }
    }

    impl UltimateDraws for BayesOdpFits {
        fn reserves(&self) -> &PredictiveDistribution {
            &self.reserves
        }

        fn latest_values(&self) -> Vec<f64> {
            segment_latest(&self.segments)
        }
    }

    impl UltimateDraws for BayesBfFit {
        fn reserves(&self) -> &PredictiveDistribution {
            &self.reserves
        }

        fn latest_values(&self) -> Vec<f64> {
            self.chain_ladder.latest.clone()
        }
    }

    impl UltimateDraws for BayesBfNormalisedFit {
        fn reserves(&self) -> &PredictiveDistribution {
            &self.reserves
        }

        fn latest_values(&self) -> Vec<f64> {
            self.chain_ladder.latest.clone()
        }
    }

    /// Meyers' models simulate the ultimate itself: its draws are the
    /// `ultimates` the fit keeps, not the reserves plus the latest values
    /// (which agree with them to rounding).
    impl UltimateDraws for MeyersFit {
        fn reserves(&self) -> &PredictiveDistribution {
            &self.reserves
        }

        fn latest_values(&self) -> Vec<f64> {
            self.chain_ladder.latest.clone()
        }

        fn ultimates(&self) -> Result<PredictiveDistribution> {
            Ok(self.ultimates.clone())
        }
    }

    impl UltimateDraws for MeyersFits {
        fn reserves(&self) -> &PredictiveDistribution {
            &self.reserves
        }

        fn latest_values(&self) -> Vec<f64> {
            segment_latest(&self.segments)
        }

        fn ultimates(&self) -> Result<PredictiveDistribution> {
            Ok(self.ultimates.clone())
        }
    }

    /// On the paid basis: the latest paid plus the reserve, the paid at the
    /// square's last age.
    impl UltimateDraws for CompartmentalFit {
        fn reserves(&self) -> &PredictiveDistribution {
            &self.reserves
        }

        fn latest_values(&self) -> Vec<f64> {
            self.latest_paid.clone()
        }
    }

    impl UltimateDraws for CompartmentalFits {
        fn reserves(&self) -> &PredictiveDistribution {
            &self.reserves
        }

        fn latest_values(&self) -> Vec<f64> {
            self.segments
                .fits
                .iter()
                .flat_map(|s| s.fit.latest_paid.iter().copied())
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use prospicio_prob::portfolio::{CorrelationKind, Pairing};
    use prospicio_prob::{Distribution, Empirical, KeyValue, PredictiveDistribution};

    use super::UltimateDraws;
    use crate::triangle::tests::raa;
    use crate::{DevelopmentColumn, Error, Grain, Long, MackBootstrap, OdpBootstrap, Triangle};

    fn odp(seed: u64, n_sims: usize) -> OdpBootstrap {
        OdpBootstrap {
            n_sims,
            seed,
            ..Default::default()
        }
    }

    /// RAA twice, under the `lob` values `A` and `B` (B scaled by 2).
    fn two_lines() -> Triangle {
        let segment = raa().segment("values").unwrap();
        let (mut keys, mut origin, mut ages, mut values) = (vec![], vec![], vec![], vec![]);
        for (name, scale) in [("A", 1.0), ("B", 2.0)] {
            for (o, period) in segment.origins.iter().enumerate() {
                for (d, &age) in segment.ages.iter().enumerate() {
                    if let Some(v) = segment.get(o, d) {
                        keys.push(name);
                        origin.push(period.start());
                        ages.push(age);
                        values.push(scale * v);
                    }
                }
            }
        }
        Triangle::from_long(&Long {
            keys: &[("lob", &keys)],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &values)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn year(y: i64) -> Vec<KeyValue> {
        vec![KeyValue::Int(y)]
    }

    #[test]
    fn each_origins_ultimate_is_its_latest_plus_its_reserve_in_the_same_simulation() {
        let fit = odp(3, 500).fit(&raa(), "values").unwrap();
        let cl = &fit.chain_ladder;
        for (o, origin) in cl.origins.iter().enumerate() {
            // Named by the origin's key, not by its position.
            let key = vec![KeyValue::Period(*origin)];
            let ultimate = fit.ultimate_draws(&key).unwrap();
            let reserve = fit.reserves.marginal(&key).unwrap();
            let reserve = Empirical::draws(&reserve);
            assert_eq!(ultimate.len(), 500);
            for (u, r) in ultimate.iter().zip(reserve) {
                assert_eq!(*u, cl.latest[o] + r);
            }
            let ratio = fit.loss_ratio_draws(&key, 25_000.0).unwrap();
            for (q, u) in ratio.iter().zip(&ultimate) {
                assert_eq!(*q, u / 25_000.0);
            }
        }
        // By label, as the bindings pass it.
        assert_eq!(
            fit.ultimate_draws(&year(1990)).unwrap(),
            fit.ultimate_draws(&[KeyValue::from("1990")]).unwrap()
        );
        // The total is the sum by index.
        let total = fit.total_ultimate_draws().unwrap();
        let latest: f64 = cl.latest.iter().sum();
        for (i, t) in total.iter().enumerate() {
            let row: f64 = fit.reserves.row(i).unwrap().iter().sum();
            assert!((t - (latest + row)).abs() < 1e-9 * t.abs());
        }
        let ratio = fit.total_loss_ratio_draws(400_000.0).unwrap();
        assert_eq!(ratio[7], total[7] / 400_000.0);
        // The joint ultimates keep the reserves' layout and seed.
        let ultimates = fit.ultimates().unwrap();
        assert_eq!(ultimates.components(), fit.reserves.components());
        assert_eq!(ultimates.provenance().seed, fit.reserves.provenance().seed);
    }

    #[test]
    fn the_mean_ultimate_is_the_bootstraps_expected_ultimate_to_monte_carlo_error() {
        let n = 4_000;
        let fit = odp(5, n).fit(&raa(), "values").unwrap();
        let cl = &fit.chain_ladder;
        let ultimates = fit.ultimates().unwrap();
        // Each origin's mean ultimate is the chain ladder's, on which the
        // bootstrap is centred, within 4 standard errors of the mean.
        for (o, origin) in cl.origins.iter().enumerate() {
            let m = ultimates
                .marginal(&vec![KeyValue::Period(*origin)])
                .unwrap();
            let se = m.variance().sqrt() / (n as f64).sqrt();
            let gap = (m.mean() - cl.ultimate[o]).abs();
            assert!(gap <= 4.0 * se + 1e-9, "{origin}: {gap} > 4 x {se}");
        }
        // The total's mean is the bootstrap's expected total ultimate (here
        // a run of ten times the simulations on another seed) within 4
        // standard errors of the difference. It is not the chain ladder's
        // to that error: the bootstrap's small upward bias, about 0.8% of
        // the total ultimate on RAA, adds up over the origins.
        let big = odp(6, 10 * n).fit(&raa(), "values").unwrap();
        let expected = big.ultimates().unwrap().total().clone();
        let total = ultimates.total();
        let se = (total.variance() / n as f64 + expected.variance() / (10 * n) as f64).sqrt();
        let gap = (total.mean() - expected.mean()).abs();
        assert!(gap <= 4.0 * se, "{gap} > 4 x {se}");
        let chain_ladder: f64 = cl.ultimate.iter().sum();
        assert!((expected.mean() / chain_ladder - 1.0).abs() < 0.02);
    }

    #[test]
    fn the_same_seed_gives_the_same_draws_in_the_same_order() {
        let a = odp(9, 300).fit(&raa(), "values").unwrap();
        let b = odp(9, 300).fit(&raa(), "values").unwrap();
        let c = odp(10, 300).fit(&raa(), "values").unwrap();
        let (ua, ub, uc) = (
            a.ultimates().unwrap(),
            b.ultimates().unwrap(),
            c.ultimates().unwrap(),
        );
        assert_eq!(ua.draw_matrix(), ub.draw_matrix());
        assert_ne!(ua.draw_matrix(), uc.draw_matrix());
        assert_eq!(
            a.loss_ratio_draws(&year(1985), 30_000.0).unwrap(),
            b.loss_ratio_draws(&year(1985), 30_000.0).unwrap()
        );
        let mack = |seed| {
            MackBootstrap {
                n_sims: 300,
                seed,
                ..Default::default()
            }
            .fit(&raa(), "values")
            .unwrap()
        };
        assert_eq!(
            mack(4).total_ultimate_draws().unwrap(),
            mack(4).total_ultimate_draws().unwrap()
        );
    }

    #[test]
    fn segments_take_the_full_key_and_join_and_reorder_keep_each_row_whole() {
        let tri = two_lines();
        let fits = odp(21, 400).fit_segments(&tri, "paid").unwrap();
        let a_1990 = vec![KeyValue::from("A"), KeyValue::Int(1990)];
        let b_1990 = vec![KeyValue::from("B"), KeyValue::Int(1990)];
        let latest = fits.latest_values();
        assert_eq!(latest.len(), 20);
        let ua = fits.ultimate_draws(&a_1990).unwrap();
        let ub = fits.ultimate_draws(&b_1990).unwrap();
        let (la, lb) = (latest[9], latest[19]);
        assert_eq!(lb, 2.0 * la);
        let reserves = &fits.reserves;
        for i in 0..400 {
            assert_eq!(ua[i], la + reserves.row(i).unwrap()[9]);
            assert_eq!(ub[i], lb + reserves.row(i).unwrap()[19]);
        }
        // A fit over segments needs the segment in the key.
        assert!(matches!(
            fits.ultimate_draws(&year(1990)),
            Err(Error::UnknownKeyValue { .. })
        ));
        assert!(matches!(
            fits.loss_ratio_draws(&a_1990, 0.0),
            Err(Error::InvalidSetting {
                name: "premium",
                ..
            })
        ));

        // Join the ODP ultimates with Mack's bootstrap of the same lines on
        // another seed, then reorder the two models to a rank correlation:
        // each model's row moves whole, so its origins and lines stay in
        // the same simulation (ultimate less latest is a reserve row of the
        // source simulation in every column).
        let mack = MackBootstrap {
            n_sims: 400,
            seed: 22,
            ..Default::default()
        }
        .fit_segments(&tri, "paid")
        .unwrap();
        let (odp_u, mack_u) = (fits.ultimates().unwrap(), mack.ultimates().unwrap());
        let joined = PredictiveDistribution::join(
            &[("odp", &odp_u), ("mack", &mack_u)],
            "model",
            Pairing::Independent,
        )
        .unwrap();
        assert_eq!(joined.dims(), ["model", "lob", "origin"]);
        // `join` keeps simulation i of each part at row i.
        for i in 0..400 {
            assert_eq!(&joined.row(i).unwrap()[..20], odp_u.row(i).unwrap());
            assert_eq!(&joined.row(i).unwrap()[20..], mack_u.row(i).unwrap());
        }
        let (reordered, index) = joined
            .reorder_groups_indexed("model", &[1.0, 0.7, 0.7, 1.0], CorrelationKind::Spearman, 3)
            .unwrap();
        let mack_latest = mack.latest_values();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(1.0);
        for (i, (&odp_src, &mack_src)) in index[0].iter().zip(&index[1]).enumerate() {
            let row = reordered.row(i).unwrap();
            for j in 0..20 {
                let r = fits.reserves.row(odp_src).unwrap()[j];
                assert!(close(row[j] - latest[j], r));
                let r = mack.reserves.row(mack_src).unwrap()[j];
                assert!(close(row[20 + j] - mack_latest[j], r));
            }
        }
    }
}
