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

use prospicio_core::{Error, Result, StreamRng};

use crate::copula::{Copula, target_ranks};
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

/// What a correlation matrix between groups measures, for
/// [`PredictiveDistribution::reorder_groups_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CorrelationKind {
    /// The correlation of the normal scores Iman–Conover rotates to, as
    /// [`PredictiveDistribution::reorder_groups`] takes it: Spearman's rho
    /// comes out near `(6/π) asin(r/2)`.
    #[default]
    Pearson,
    /// Spearman's rho of the group totals: converted to the normal-score
    /// correlation `r = 2 sin(π ρ / 6)` before reordering, so the achieved
    /// rank correlation is close to the target.
    Spearman,
}

impl CorrelationKind {
    /// The normal-score correlation for this kind of matrix: the matrix
    /// itself, or `2 sin(π ρ / 6)` entry by entry (1 stays 1).
    ///
    /// ```
    /// use prospicio_prob::portfolio::CorrelationKind;
    ///
    /// let r = CorrelationKind::Spearman.normal_scores(&[1.0, 0.5, 0.5, 1.0]);
    /// assert!((r[1] - 2.0 * (std::f64::consts::PI / 12.0).sin()).abs() < 1e-15);
    /// assert_eq!(r[0], 1.0);
    /// ```
    pub fn normal_scores(self, correlation: &[f64]) -> Vec<f64> {
        match self {
            Self::Pearson => correlation.to_vec(),
            Self::Spearman => correlation
                .iter()
                .map(|&rho| {
                    if rho == 1.0 {
                        1.0
                    } else {
                        2.0 * (std::f64::consts::PI * rho / 6.0).sin()
                    }
                })
                .collect(),
        }
    }
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
        Ok(self
            .reorder_groups_indexed(dim, correlation, CorrelationKind::Pearson, seed)?
            .0)
    }

    /// [`reorder_groups_with`](Self::reorder_groups_with), and where each
    /// simulation came from: `index[g][i]` is the simulation of the input
    /// whose row group `g` (in the order of [`groups`](Self::groups)) takes
    /// in simulation `i` of the result. Anything else simulated with the
    /// group, a class's events say, follows the reorder by the same index
    /// (`EventSet::permute` in `prospicio-aggregate`), with no matching of
    /// values, which tied draws would defeat.
    ///
    /// ```
    /// use prospicio_prob::portfolio::CorrelationKind;
    /// use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
    ///
    /// let n = 500;
    /// let pd = PredictiveDistribution::from_draws(
    ///     vec!["seg".into()],
    ///     vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
    ///     // Many ties: matching values could not tell the rows apart.
    ///     (0..n).flat_map(|i| [(i % 5) as f64, ((i * 7919) % 3) as f64]).collect(),
    ///     Provenance::new("example"),
    /// )
    /// .unwrap();
    /// let (r, index) = pd
    ///     .reorder_groups_indexed("seg", &[1.0, 0.5, 0.5, 1.0], CorrelationKind::Pearson, 5)
    ///     .unwrap();
    /// for i in 0..n {
    ///     for g in 0..2 {
    ///         assert_eq!(r.row(i).unwrap()[g], pd.row(index[g][i]).unwrap()[g]);
    ///     }
    /// }
    /// ```
    pub fn reorder_groups_indexed(
        &self,
        dim: &str,
        correlation: &[f64],
        kind: CorrelationKind,
        seed: u64,
    ) -> Result<(Self, Vec<Vec<usize>>)> {
        let correlation = kind.normal_scores(correlation);
        let (groups, group_of) = self.group_index(dim)?;
        let ranks = target_ranks(self.n_sims(), groups.len(), &correlation, seed)?;
        self.reorder_to_ranks(
            &group_of,
            &ranks,
            format!("{dim}: {correlation:?}, seed {seed}"),
        )
    }

    /// Sets the dependence between the groups of dimension `dim` by a
    /// copula: `n` draws of uniforms from `copula` (one dimension per group,
    /// groups in order of first appearance; draw `i` uses stream `i` of
    /// `seed`, as the bindings' `sample` does), and each group's
    /// simulations, whole rows sorted by the group's total, take the ranks
    /// of its column of uniforms. The groups' totals then have exactly the
    /// rank pattern of the copula's draws, so a t, Gumbel or Clayton copula
    /// gives the joint tails it would give simulated marginals, on results
    /// already simulated. Every group keeps its distribution and internal
    /// joint structure, as under [`reorder_groups`](Self::reorder_groups).
    ///
    /// ```
    /// use prospicio_prob::copula::{Archimedean, ArchimedeanCopula};
    /// use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
    ///
    /// let n = 4000;
    /// let pd = PredictiveDistribution::from_draws(
    ///     vec!["seg".into()],
    ///     vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
    ///     (0..n).flat_map(|i| [i as f64, ((i * 7919) % n) as f64]).collect(),
    ///     Provenance::new("example"),
    /// )
    /// .unwrap();
    /// // Clayton with theta = 2: Kendall's tau 0.5, Spearman's rho near 0.68.
    /// let clayton = ArchimedeanCopula::new(Archimedean::Clayton, 2.0, 2).unwrap();
    /// let r = pd.reorder_groups_by_copula("seg", &clayton, 5).unwrap();
    /// let (_, rho) = r.group_spearman("seg").unwrap();
    /// assert!((rho[1] - 0.68).abs() < 0.03);
    /// ```
    pub fn reorder_groups_by_copula(
        &self,
        dim: &str,
        copula: &dyn Copula,
        seed: u64,
    ) -> Result<Self> {
        Ok(self.reorder_groups_by_copula_indexed(dim, copula, seed)?.0)
    }

    /// [`reorder_groups_by_copula`](Self::reorder_groups_by_copula), and
    /// where each simulation came from, as
    /// [`reorder_groups_indexed`](Self::reorder_groups_indexed) gives it.
    pub fn reorder_groups_by_copula_indexed(
        &self,
        dim: &str,
        copula: &dyn Copula,
        seed: u64,
    ) -> Result<(Self, Vec<Vec<usize>>)> {
        let (groups, group_of) = self.group_index(dim)?;
        let (n, g) = (self.n_sims(), groups.len());
        if copula.dim() != g {
            return Err(Error::Data(format!(
                "a {}-dimensional copula for {g} groups of dimension {dim:?}",
                copula.dim()
            )));
        }
        let mut columns = vec![vec![0.0; n]; g];
        let mut u = vec![0.0; g];
        for i in 0..n {
            copula.sample(&mut StreamRng::new(seed, i as u64), &mut u);
            for (column, x) in columns.iter_mut().zip(&u) {
                column[i] = *x;
            }
        }
        let ranks: Vec<Vec<usize>> = columns
            .iter()
            .map(|column| {
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| column[a].total_cmp(&column[b]));
                let mut rank = vec![0; n];
                for (r, &i) in order.iter().enumerate() {
                    rank[i] = r;
                }
                rank
            })
            .collect();
        self.reorder_to_ranks(&group_of, &ranks, format!("{dim}: copula, seed {seed}"))
    }

    /// Simulation `i` of the result takes, for group `g`, the row of rank
    /// `ranks[g][i]` (0-based, ascending) among the group's totals; also
    /// returns that row's index, `index[g][i]`.
    fn reorder_to_ranks(
        &self,
        group_of: &[usize],
        ranks: &[Vec<usize>],
        note: String,
    ) -> Result<(Self, Vec<Vec<usize>>)> {
        let (n, m) = (self.n_sims(), self.n_components());
        let totals = self.group_totals(ranks.len(), group_of);
        // Each group's rows sorted by the group's total.
        let order: Vec<Vec<usize>> = totals
            .iter()
            .map(|totals| {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| totals[a].total_cmp(&totals[b]));
                idx
            })
            .collect();
        let index: Vec<Vec<usize>> = (0..ranks.len())
            .map(|g| (0..n).map(|i| order[g][ranks[g][i]]).collect())
            .collect();
        let mut draws = vec![0.0; n * m];
        for i in 0..n {
            for (j, &gi) in group_of.iter().enumerate() {
                draws[i * m + j] = self.row(index[gi][i]).expect("in range")[j];
            }
        }
        let provenance = self.provenance().clone().param("reorder_groups", note);
        let result = Self::from_draws(
            self.dims().to_vec(),
            self.components().to_vec(),
            draws,
            provenance,
        )?;
        Ok((result, index))
    }

    /// [`reorder_groups`](Self::reorder_groups) with a correlation of the
    /// given kind: a [`CorrelationKind::Spearman`] target is converted to
    /// the normal-score correlation `2 sin(π ρ / 6)` first, which must still
    /// be positive definite.
    ///
    /// ```
    /// use prospicio_prob::portfolio::CorrelationKind;
    /// use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
    ///
    /// let n = 4000;
    /// let pd = PredictiveDistribution::from_draws(
    ///     vec!["seg".into()],
    ///     vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
    ///     (0..n).flat_map(|i| [i as f64, ((i * 7919) % n) as f64]).collect(),
    ///     Provenance::new("example"),
    /// )
    /// .unwrap();
    /// let r = pd
    ///     .reorder_groups_with("seg", &[1.0, 0.6, 0.6, 1.0], CorrelationKind::Spearman, 5)
    ///     .unwrap();
    /// let (_, rho) = r.group_spearman("seg").unwrap();
    /// assert!((rho[1] - 0.6).abs() < 0.02);
    /// ```
    pub fn reorder_groups_with(
        &self,
        dim: &str,
        correlation: &[f64],
        kind: CorrelationKind,
        seed: u64,
    ) -> Result<Self> {
        Ok(self.reorder_groups_indexed(dim, correlation, kind, seed)?.0)
    }

    /// The groups of dimension `dim`, in order of first appearance: the
    /// order of the rows and columns of
    /// [`reorder_groups`](Self::reorder_groups)' matrix.
    pub fn groups(&self, dim: &str) -> Result<Vec<KeyValue>> {
        Ok(self.group_index(dim)?.0)
    }

    /// The `G × G` correlation matrix between the groups of `dim` from a
    /// long pairwise table `(group, group, value)`, in the order of
    /// [`groups`](Self::groups). Each pair of distinct groups must appear
    /// once, in either order (both orders only with the same value); a
    /// diagonal entry may be given and must be 1. Values lie in `[-1, 1]`.
    ///
    /// ```
    /// use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
    ///
    /// let pd = PredictiveDistribution::from_draws(
    ///     vec!["peril".into()],
    ///     ["wind", "quake", "flood"].iter().map(|p| vec![KeyValue::from(*p)]).collect(),
    ///     (0..30).map(|i| i as f64).collect(),
    ///     Provenance::new("example"),
    /// )
    /// .unwrap();
    /// let k = |s: &str| KeyValue::from(s);
    /// let m = pd
    ///     .correlation_from_pairs(
    ///         "peril",
    ///         &[(k("quake"), k("wind"), 0.2), (k("wind"), k("flood"), 0.5), (k("flood"), k("quake"), 0.1)],
    ///     )
    ///     .unwrap();
    /// assert_eq!(m, vec![1.0, 0.2, 0.5, 0.2, 1.0, 0.1, 0.5, 0.1, 1.0]);
    /// ```
    pub fn correlation_from_pairs(
        &self,
        dim: &str,
        pairs: &[(KeyValue, KeyValue, f64)],
    ) -> Result<Vec<f64>> {
        let groups = self.groups(dim)?;
        let g = groups.len();
        let position = |key: &KeyValue| {
            groups
                .iter()
                .position(|x| x == key)
                .ok_or_else(|| Error::Data(format!("no group {key:?} in dimension {dim:?}")))
        };
        let mut m = vec![f64::NAN; g * g];
        for i in 0..g {
            m[i * g + i] = 1.0;
        }
        for (a, b, value) in pairs {
            let (i, j) = (position(a)?, position(b)?);
            if !(-1.0..=1.0).contains(value) {
                return Err(Error::InvalidParameter {
                    name: "correlation",
                    value: *value,
                    reason: "must lie in [-1, 1]",
                });
            }
            if i == j {
                if *value != 1.0 {
                    return Err(Error::InvalidParameter {
                        name: "correlation",
                        value: *value,
                        reason: "must be 1 for a group with itself",
                    });
                }
                continue;
            }
            let current = m[i * g + j];
            if !current.is_nan() && current != *value {
                return Err(Error::Data(format!(
                    "the pair ({a:?}, {b:?}) is given twice, as {current} and {value}"
                )));
            }
            m[i * g + j] = *value;
            m[j * g + i] = *value;
        }
        if let Some(k) = m.iter().position(|v| v.is_nan()) {
            return Err(Error::Data(format!(
                "no correlation for the pair ({:?}, {:?})",
                groups[k / g],
                groups[k % g]
            )));
        }
        Ok(m)
    }

    /// Spearman's rho between the totals of the groups of `dim`: the
    /// groups (as [`groups`](Self::groups)) and the `G × G` matrix, ties
    /// given their average rank as `scipy.stats.spearmanr` and R's
    /// `cor(method = "spearman")` do. It checks what
    /// [`reorder_groups_with`](Self::reorder_groups_with) achieved.
    pub fn group_spearman(&self, dim: &str) -> Result<(Vec<KeyValue>, Vec<f64>)> {
        let (groups, group_of) = self.group_index(dim)?;
        let g = groups.len();
        let ranks: Vec<Vec<f64>> = self
            .group_totals(g, &group_of)
            .iter()
            .map(|t| average_ranks(t))
            .collect();
        let mut m = vec![1.0; g * g];
        for i in 0..g {
            for j in 0..i {
                let r = pearson(&ranks[i], &ranks[j]);
                m[i * g + j] = r;
                m[j * g + i] = r;
            }
        }
        Ok((groups, m))
    }

    /// The groups of `dim` in order of first appearance and each
    /// component's group.
    fn group_index(&self, dim: &str) -> Result<(Vec<KeyValue>, Vec<usize>)> {
        let d = self
            .dims()
            .iter()
            .position(|x| x == dim)
            .ok_or_else(|| Error::Data(format!("no dimension {dim:?}")))?;
        let mut groups: Vec<KeyValue> = Vec::new();
        let group_of = self
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
        Ok((groups, group_of))
    }

    /// Each group's total in every simulation.
    fn group_totals(&self, g: usize, group_of: &[usize]) -> Vec<Vec<f64>> {
        let n = self.n_sims();
        let mut totals = vec![vec![0.0; n]; g];
        for (i, row) in (0..n).map(|i| self.row(i).expect("in range")).enumerate() {
            for (x, &gi) in row.iter().zip(group_of) {
                totals[gi][i] += x;
            }
        }
        totals
    }
}

/// Ranks from 1, ties given their average rank.
fn average_ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut ranks = vec![0.0; x.len()];
    let mut start = 0;
    while start < idx.len() {
        let mut end = start + 1;
        while end < idx.len() && x[idx[end]] == x[idx[start]] {
            end += 1;
        }
        let rank = (start + end + 1) as f64 / 2.0;
        for &i in &idx[start..end] {
            ranks[i] = rank;
        }
        start = end;
    }
    ranks
}

/// Pearson's correlation, NaN when either vector is constant.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    sxy / (sxx * syy).sqrt()
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

    #[test]
    fn a_spearman_target_is_met_and_a_pearson_one_falls_short() {
        let n = 20_000;
        let pd = part("class", &[1, 2, 3], n, 1, |i, j| {
            ((i * [1, 7919, 104_729][j]) % n) as f64
        });
        let rho = [1.0, 0.5, 0.3, 0.5, 1.0, 0.7, 0.3, 0.7, 1.0];
        let s = pd
            .reorder_groups_with("class", &rho, CorrelationKind::Spearman, 9)
            .unwrap();
        let (groups, got) = s.group_spearman("class").unwrap();
        assert_eq!(
            groups,
            vec![KeyValue::Int(1), KeyValue::Int(2), KeyValue::Int(3)]
        );
        for (g, w) in got.iter().zip(&rho) {
            assert!((g - w).abs() < 0.01, "{g} vs {w}");
        }
        // The same matrix read as normal scores gives (6/π) asin(r/2).
        let p = pd.reorder_groups("class", &rho, 9).unwrap();
        let (_, got) = p.group_spearman("class").unwrap();
        let want = 6.0 / std::f64::consts::PI * (0.35f64).asin();
        assert!((got[5] - want).abs() < 0.01, "{} vs {want}", got[5]);
        assert!(got[5] < 0.7 - 0.005);
    }

    #[test]
    fn spearman_gives_ties_their_average_rank() {
        // scipy.stats.spearmanr([0, 0, 1, 2], [0, 1, 1, 3]) = 5 / 6.
        let pd = PredictiveDistribution::from_draws(
            vec!["g".into()],
            vec![vec![KeyValue::Int(0)], vec![KeyValue::Int(1)]],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0],
            Provenance::new("t"),
        )
        .unwrap();
        let (_, m) = pd.group_spearman("g").unwrap();
        assert!((m[1] - 5.0 / 6.0).abs() < 1e-15, "{}", m[1]);
    }

    #[test]
    fn a_pairwise_table_needs_every_pair_once() {
        let pd = part("class", &[1, 2, 3], 10, 1, |i, j| (i + j) as f64);
        let k = KeyValue::Int;
        let full = [
            (k(1), k(2), 0.5),
            (k(3), k(1), 0.2),
            (k(2), k(3), 0.1),
            (k(2), k(2), 1.0),
        ];
        assert_eq!(
            pd.correlation_from_pairs("class", &full).unwrap(),
            vec![1.0, 0.5, 0.2, 0.5, 1.0, 0.1, 0.2, 0.1, 1.0]
        );
        // Both orders with the same value are fine; different values are not.
        let mut twice = full.to_vec();
        twice.push((k(2), k(1), 0.5));
        assert!(pd.correlation_from_pairs("class", &twice).is_ok());
        twice.push((k(2), k(1), 0.4));
        assert!(pd.correlation_from_pairs("class", &twice).is_err());
        assert!(pd.correlation_from_pairs("class", &full[..2]).is_err());
        assert!(
            pd.correlation_from_pairs("class", &[(k(1), k(9), 0.5)])
                .is_err()
        );
        assert!(
            pd.correlation_from_pairs("class", &[(k(1), k(1), 0.5)])
                .is_err()
        );
        assert!(
            pd.correlation_from_pairs("class", &[(k(1), k(2), 1.5)])
                .is_err()
        );
        // A Spearman matrix that is not positive definite is refused.
        let bad = [1.0, 0.9, -0.9, 0.9, 1.0, 0.9, -0.9, 0.9, 1.0];
        assert!(
            pd.reorder_groups_with("class", &bad, CorrelationKind::Spearman, 1)
                .is_err()
        );
    }

    #[test]
    fn a_copula_reorder_gives_the_totals_the_copulas_ranks() {
        use crate::copula::{Archimedean, ArchimedeanCopula, StudentTCopula};
        let n = 3000;
        // Two segments of two items each; within a segment item 1 is 3x item 0.
        let draws: Vec<f64> = (0..n)
            .flat_map(|i| {
                let a = ((i * 7919) % n) as f64;
                let b = ((i * 104_729) % n) as f64;
                [a, 3.0 * a, b, b + 1.0]
            })
            .collect();
        let k = |s: &str, j: i64| vec![KeyValue::from(s), KeyValue::Int(j)];
        let pd = PredictiveDistribution::from_draws(
            vec!["seg".into(), "item".into()],
            vec![k("a", 0), k("a", 1), k("b", 0), k("b", 1)],
            draws,
            Provenance::new("test"),
        )
        .unwrap();
        let gumbel = ArchimedeanCopula::new(Archimedean::Gumbel, 2.0, 2).unwrap();
        let r = pd.reorder_groups_by_copula("seg", &gumbel, 9).unwrap();
        for i in 0..n {
            let row = r.row(i).unwrap();
            assert_eq!(row[1], 3.0 * row[0]);
            assert_eq!(row[3], row[2] + 1.0);
        }
        for key in pd.components() {
            assert_eq!(
                r.marginal(key).unwrap().sorted(),
                pd.marginal(key).unwrap().sorted()
            );
        }
        // The totals' ranks are the copula draws' ranks, draw i on stream i.
        let totals = r.group_totals(2, &[0, 0, 1, 1]);
        let mut u = [0.0; 2];
        let draws: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                gumbel.sample(&mut StreamRng::new(9, i as u64), &mut u);
                u
            })
            .collect();
        for g in 0..2 {
            let rank_of = |v: &[f64]| average_ranks(v);
            let column: Vec<f64> = draws.iter().map(|d| d[g]).collect();
            assert_eq!(rank_of(&totals[g]), rank_of(&column));
        }
        // Gumbel with theta = 2: Kendall's tau 0.5, Spearman's rho near 0.68.
        let (_, rho) = r.group_spearman("seg").unwrap();
        assert!((rho[1] - 0.68).abs() < 0.03, "{}", rho[1]);

        // A t copula with few degrees of freedom puts more joint extremes in
        // the upper corner than Iman-Conover's normal scores at the same
        // correlation.
        let joint_top = |r: &PredictiveDistribution| {
            let t = r.group_totals(2, &[0, 0, 1, 1]);
            let (a, b) = (average_ranks(&t[0]), average_ranks(&t[1]));
            let cut = 0.99 * n as f64;
            a.iter()
                .zip(&b)
                .filter(|(x, y)| **x > cut && **y > cut)
                .count()
        };
        let t = StudentTCopula::new(&[1.0, 0.5, 0.5, 1.0], 2, 2.0).unwrap();
        let by_t = pd.reorder_groups_by_copula("seg", &t, 3).unwrap();
        let normal = pd.reorder_groups("seg", &[1.0, 0.5, 0.5, 1.0], 3).unwrap();
        assert!(joint_top(&by_t) > joint_top(&normal));

        let clayton3 = ArchimedeanCopula::new(Archimedean::Clayton, 1.0, 3).unwrap();
        assert!(pd.reorder_groups_by_copula("seg", &clayton3, 1).is_err());
    }
}
