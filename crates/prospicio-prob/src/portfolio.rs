//! Building a portfolio from predictive distributions of different models:
//! a reserve bootstrap by origin, premium risk by line, a reinsurance
//! tower's gross, ceded and net.
//!
//! [`PredictiveDistribution::join`] puts them side by side as segments of
//! one distribution, pairing simulation `i` of every part.
//! [`PredictiveDistribution::reorder_groups`] then sets the dependence
//! between segments, Iman–Conover on the segment totals, moving each
//! segment's simulations as whole rows so its own joint structure (across
//! origins, layers) is kept. The result goes to `aggregate`, the risk
//! measures and `capital`.

use std::collections::HashSet;

use prospicio_core::{Error, Result};

use crate::copula::target_ranks;
use crate::predictive::{ComponentKey, KeyValue, PredictiveDistribution};
use crate::provenance::Provenance;

/// How the simulations of joined parts relate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pairing {
    /// Simulated separately: two parts with the same seed and stream
    /// scheme would share random numbers, so they are refused, whatever
    /// samplers they were drawn with ([`Provenance::shares_streams`]).
    Independent,
    /// Derived from the same simulations (a cover applied to a reserve, a
    /// tower's ceded and net): simulation `i` of every part is the same
    /// scenario, and no seed check is made.
    SameSimulations,
}

impl PredictiveDistribution {
    /// Joins `parts` (label, distribution) into one distribution with a new
    /// leading dimension `dim` holding each part's label, followed by the
    /// union of the parts' dimensions in order of first appearance. A part
    /// without one of those dimensions has the empty text `""` there.
    ///
    /// Simulation `i` of the result is simulation `i` of every part. With
    /// [`Pairing::Independent`] the parts are independent only if they were
    /// simulated independently, so two parts with the same seed and stream
    /// scheme (which would draw the same random numbers in each
    /// simulation) are refused; set the dependence you want with
    /// [`reorder_groups`](Self::reorder_groups). With
    /// [`Pairing::SameSimulations`] the parts come from the same scenarios
    /// and keep their pairing.
    ///
    /// ```
    /// use prospicio_prob::portfolio::Pairing;
    /// use prospicio_prob::{Empirical, KeyValue, PredictiveDistribution, Provenance};
    ///
    /// let reserve = PredictiveDistribution::from_draws(
    ///     vec!["origin".into()],
    ///     vec![vec![KeyValue::Int(2023)], vec![KeyValue::Int(2024)]],
    ///     vec![10.0, 20.0, 12.0, 25.0],
    ///     Provenance::new("odp_bootstrap").seed(1, "chacha20/sim-index/v1"),
    /// )
    /// .unwrap();
    /// let premium = PredictiveDistribution::from_draws(
    ///     vec!["lob".into()],
    ///     vec![vec![KeyValue::from("motor")]],
    ///     vec![50.0, 40.0],
    ///     Provenance::new("collective").seed(2, "chacha20/sim-index/v1"),
    /// )
    /// .unwrap();
    /// let parts = [("reserve", &reserve), ("premium", &premium)];
    /// let p = PredictiveDistribution::join(&parts, "risk", Pairing::Independent).unwrap();
    /// assert_eq!(p.dims(), ["risk", "origin", "lob"]);
    /// assert_eq!(p.total().draws(), [80.0, 77.0]);
    /// let by_risk = p.aggregate(&["risk"]).unwrap();
    /// assert_eq!(by_risk.draw_matrix(), [30.0, 50.0, 37.0, 40.0]);
    /// ```
    pub fn join(
        parts: &[(&str, &PredictiveDistribution)],
        dim: &str,
        pairing: Pairing,
    ) -> Result<Self> {
        let (_, first) = parts
            .first()
            .ok_or_else(|| Error::Data("join needs at least one part".into()))?;
        let n = first.n_sims();
        let mut labels = HashSet::new();
        let mut seeded: Vec<&Provenance> = Vec::new();
        let mut dims: Vec<String> = vec![dim.to_string()];
        for (label, pd) in parts {
            if pd.n_sims() != n {
                return Err(Error::Data(format!(
                    "part {label:?} has {} simulations, the first has {n}",
                    pd.n_sims()
                )));
            }
            if !labels.insert(*label) {
                return Err(Error::Data(format!("part label {label:?} is repeated")));
            }
            if pd.dims().iter().any(|d| d == dim) {
                return Err(Error::Data(format!(
                    "part {label:?} already has a dimension {dim:?}"
                )));
            }
            // Seed and stream scheme only: parts drawn with different
            // samplers (before and after a sampler change) still share
            // their uniforms.
            let prov = pd.provenance();
            if pairing == Pairing::Independent {
                let shared = seeded.iter().any(|p| prov.shares_streams(p));
                if let (true, Some(seed)) = (shared, prov.seed) {
                    return Err(Error::Data(format!(
                        "part {label:?} reuses seed {seed} with the same stream scheme as an \
                         earlier part, so their simulations would share random numbers; \
                         simulate it with another seed"
                    )));
                }
                seeded.push(prov);
            }
            for d in pd.dims() {
                if !dims.contains(d) {
                    dims.push(d.clone());
                }
            }
        }
        let mut components: Vec<ComponentKey> = Vec::new();
        for (label, pd) in parts {
            for key in pd.components() {
                let mut k = vec![KeyValue::from(*label)];
                for d in &dims[1..] {
                    match pd.dims().iter().position(|x| x == d) {
                        Some(j) => k.push(key[j].clone()),
                        None => k.push(KeyValue::from("")),
                    }
                }
                components.push(k);
            }
        }
        let width = components.len();
        let mut draws = Vec::with_capacity(n * width);
        for i in 0..n {
            for (_, pd) in parts {
                draws.extend_from_slice(pd.row(i).expect("in range"));
            }
        }
        let mut provenance = Provenance::new("join").param("pairing", format!("{pairing:?}"));
        for (label, pd) in parts {
            let p = pd.provenance();
            let seed = p.seed.map_or_else(|| "none".to_string(), |s| s.to_string());
            provenance = provenance.param(
                format!("part:{label}"),
                format!("model {}, seed {seed}", p.model),
            );
        }
        Self::from_draws(dims, components, draws, provenance)
    }

    /// Sets the dependence between the groups of dimension `dim` (the
    /// segments of [`join`](Self::join), lines, layers) by Iman–Conover on
    /// the groups' totals: each group's simulations are reordered as whole
    /// rows, so every group keeps its own distribution and its internal
    /// joint structure, and the groups' totals take a rank correlation
    /// close to `correlation` (`G × G`, groups in order of first
    /// appearance; Spearman's rho near `(6/π) asin(r/2)`). Group `g > 0`
    /// shuffles with stream `g` of `seed`.
    ///
    /// ```
    /// use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
    ///
    /// let n = 2000;
    /// let pd = PredictiveDistribution::from_draws(
    ///     vec!["seg".into(), "item".into()],
    ///     vec![
    ///         vec![KeyValue::from("a"), KeyValue::Int(0)],
    ///         vec![KeyValue::from("a"), KeyValue::Int(1)],
    ///         vec![KeyValue::from("b"), KeyValue::Int(0)],
    ///     ],
    ///     (0..n).flat_map(|i| [i as f64, 2.0 * i as f64, ((i * 7919) % n) as f64]).collect(),
    ///     Provenance::new("example"),
    /// )
    /// .unwrap();
    /// let r = pd.reorder_groups("seg", &[1.0, 0.0, 0.0, 1.0], 5).unwrap();
    /// // Segment a's rows move together: item 1 is still twice item 0.
    /// assert!((0..n).all(|i| r.row(i).unwrap()[1] == 2.0 * r.row(i).unwrap()[0]));
    /// ```
    pub fn reorder_groups(&self, dim: &str, correlation: &[f64], seed: u64) -> Result<Self> {
        let d = self
            .dims()
            .iter()
            .position(|x| x == dim)
            .ok_or_else(|| Error::Data(format!("no dimension {dim:?}")))?;
        let mut groups: Vec<KeyValue> = Vec::new();
        let group_of: Vec<usize> = self
            .components()
            .iter()
            .map(|k| match groups.iter().position(|g| *g == k[d]) {
                Some(g) => g,
                None => {
                    groups.push(k[d].clone());
                    groups.len() - 1
                }
            })
            .collect();
        let (n, m, g) = (self.n_sims(), self.n_components(), groups.len());
        let ranks = target_ranks(n, g, correlation, seed)?;
        // Each group's rows sorted by the group's total.
        let order: Vec<Vec<usize>> = (0..g)
            .map(|gi| {
                let totals: Vec<f64> = (0..n)
                    .map(|i| {
                        let row = self.row(i).expect("in range");
                        (0..m).filter(|&j| group_of[j] == gi).map(|j| row[j]).sum()
                    })
                    .collect();
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| totals[a].total_cmp(&totals[b]));
                idx
            })
            .collect();
        let mut draws = vec![0.0; n * m];
        for i in 0..n {
            for (j, &gi) in group_of.iter().enumerate() {
                let source = order[gi][ranks[gi][i]];
                draws[i * m + j] = self.row(source).expect("in range")[j];
            }
        }
        let provenance = self.provenance().clone().param(
            "reorder_groups",
            format!("{dim}: {correlation:?}, seed {seed}"),
        );
        Self::from_draws(
            self.dims().to_vec(),
            self.components().to_vec(),
            draws,
            provenance,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Empirical;

    fn part(
        dim: &str,
        keys: &[i64],
        n: usize,
        seed: u64,
        f: impl Fn(usize, usize) -> f64,
    ) -> PredictiveDistribution {
        let draws = (0..n)
            .flat_map(|i| (0..keys.len()).map(move |j| (i, j)))
            .map(|(i, j)| f(i, j))
            .collect();
        PredictiveDistribution::from_draws(
            vec![dim.into()],
            keys.iter().map(|k| vec![KeyValue::Int(*k)]).collect(),
            draws,
            Provenance::new("test").seed(seed, "chacha20/sim-index/v1"),
        )
        .unwrap()
    }

    #[test]
    fn join_checks_shapes_labels_and_seeds() {
        let a = part("origin", &[1, 2], 10, 1, |i, j| (i + j) as f64);
        let b = part("lob", &[7], 10, 2, |i, _| i as f64);
        let same_seed = part("lob", &[7], 10, 1, |i, _| i as f64);
        let short = part("lob", &[7], 9, 3, |i, _| i as f64);
        assert!(
            PredictiveDistribution::join(
                &[("a", &a), ("b", &same_seed)],
                "risk",
                Pairing::Independent
            )
            .is_err()
        );
        assert!(
            PredictiveDistribution::join(
                &[("a", &a), ("b", &same_seed)],
                "risk",
                Pairing::SameSimulations
            )
            .is_ok()
        );
        assert!(
            PredictiveDistribution::join(&[("a", &a), ("b", &short)], "risk", Pairing::Independent)
                .is_err()
        );
        assert!(
            PredictiveDistribution::join(&[("a", &a), ("a", &b)], "risk", Pairing::Independent)
                .is_err()
        );
        assert!(
            PredictiveDistribution::join(&[("a", &a)], "origin", Pairing::Independent).is_err()
        );
        let j = PredictiveDistribution::join(&[("a", &a), ("b", &b)], "risk", Pairing::Independent)
            .unwrap();
        assert_eq!(
            j.components()[2],
            vec![KeyValue::from("b"), KeyValue::from(""), KeyValue::Int(7)]
        );
        assert_eq!(
            j.marginal(&j.components()[1].clone()).unwrap().draws(),
            a.marginal(&vec![KeyValue::Int(2)]).unwrap().draws()
        );
    }

    /// `part` with its provenance's samplers replaced.
    fn with_samplers(
        pd: &PredictiveDistribution,
        samplers: Option<Vec<(String, String)>>,
    ) -> PredictiveDistribution {
        let mut prov = pd.provenance().clone();
        prov.samplers = samplers;
        PredictiveDistribution::from_draws(
            pd.dims().to_vec(),
            pd.components().to_vec(),
            pd.draw_matrix().to_vec(),
            prov,
        )
        .unwrap()
    }

    #[test]
    fn join_refuses_a_shared_seed_across_a_sampler_change() {
        // The same seed drawn under today's samplers, under an earlier Gamma
        // sampler, and read from a file saved before samplers were recorded:
        // the uniforms are the same, so the parts are not independent.
        let new = part("origin", &[1], 10, 5, |i, _| i as f64);
        let before = with_samplers(&new, Some(vec![]));
        let unrecorded = with_samplers(&new, None);
        assert!(!new.provenance().replays_same_draws(before.provenance()));
        for old in [&before, &unrecorded] {
            for parts in [[("new", &new), ("old", old)], [("old", old), ("new", &new)]] {
                let err = PredictiveDistribution::join(&parts, "risk", Pairing::Independent)
                    .unwrap_err()
                    .to_string();
                assert!(err.contains("reuses seed 5"), "{err}");
                assert!(
                    PredictiveDistribution::join(&parts, "risk", Pairing::SameSimulations).is_ok()
                );
            }
        }
        let other_seed = with_samplers(&part("origin", &[1], 10, 6, |i, _| i as f64), None);
        assert!(
            PredictiveDistribution::join(
                &[("new", &new), ("other", &other_seed)],
                "risk",
                Pairing::Independent
            )
            .is_ok()
        );
    }

    #[test]
    fn reorder_groups_keeps_marginals_and_sets_rank_correlation() {
        let n = 4000;
        let a = part("origin", &[1, 2], n, 1, |i, j| {
            ((i * 7919 + j * 31) % n) as f64 + j as f64
        });
        let b = part("lob", &[7], n, 2, |i, _| ((i * 104_729) % n) as f64);
        let j = PredictiveDistribution::join(&[("a", &a), ("b", &b)], "risk", Pairing::Independent)
            .unwrap();
        let r = j.reorder_groups("risk", &[1.0, 0.8, 0.8, 1.0], 3).unwrap();
        for c in j.components() {
            let mut x = j.marginal(c).unwrap().draws().to_vec();
            let mut y = r.marginal(c).unwrap().draws().to_vec();
            x.sort_by(f64::total_cmp);
            y.sort_by(f64::total_cmp);
            assert_eq!(x, y);
        }
        // Rows of segment a move whole: each row of a appears in j.
        let rows_a: HashSet<(u64, u64)> = (0..n)
            .map(|i| {
                let row = j.row(i).unwrap();
                (row[0].to_bits(), row[1].to_bits())
            })
            .collect();
        assert!((0..n).all(|i| {
            let row = r.row(i).unwrap();
            rows_a.contains(&(row[0].to_bits(), row[1].to_bits()))
        }));
        // Spearman's rho of the segment totals near (6/π) asin(0.4).
        let by = r.aggregate(&["risk"]).unwrap();
        let (ta, tb): (Vec<f64>, Vec<f64>) = (0..n)
            .map(|i| (by.row(i).unwrap()[0], by.row(i).unwrap()[1]))
            .unzip();
        let rank = |v: &[f64]| {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&p, &q| v[p].total_cmp(&v[q]));
            let mut r = vec![0.0; v.len()];
            for (k, &i) in idx.iter().enumerate() {
                r[i] = k as f64;
            }
            r
        };
        let (ra, rb) = (rank(&ta), rank(&tb));
        let mean = (n as f64 - 1.0) / 2.0;
        let cov: f64 = ra
            .iter()
            .zip(&rb)
            .map(|(x, y)| (x - mean) * (y - mean))
            .sum();
        let var: f64 = ra.iter().map(|x| (x - mean).powi(2)).sum();
        let rho = cov / var;
        let want = 6.0 / std::f64::consts::PI * (0.4f64).asin();
        assert!((rho - want).abs() < 0.03, "{rho} vs {want}");
        assert!(r.reorder_groups("nope", &[1.0], 1).is_err());
    }
}
