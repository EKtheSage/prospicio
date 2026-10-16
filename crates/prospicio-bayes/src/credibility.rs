//! Bayesian hierarchical credibility: Bühlmann–Straub's model, and
//! Jewell's hierarchy of it, as a normal random-effects model sampled with
//! NUTS (`docs/design/credibility.md`, "Bayesian hierarchical
//! credibility").
//!
//! Group `i` in period `j` has ratio `X_ij` with weight `w_ij`. Each group
//! has a risk premium `θ_i`, each node of a hierarchy above the groups
//! (sectors, then regions, ...) a mean of its own, and the top a collective
//! mean `μ`:
//!
//! ```text
//! X_ij | θ_i      ~ N(θ_i, σ² / w_ij)
//! θ_i  | ν_p(i)   ~ N(ν_p(i), τ_L²)          (the group level)
//! ν_k  | ν_p(k)   ~ N(ν_p(k), τ_l²)          (each level l above it)
//! ν_top           ~ N(μ, τ_0²),  μ ~ N(m, s²) or flat
//! ```
//!
//! with half-normal or half-Cauchy priors on the standard deviations `σ`
//! and `τ_l`. Given `σ` and the `τ_l`, every mean is Gaussian, so they are
//! integrated out exactly by one pass up the tree; NUTS samples only
//! `log σ` and the `log τ_l`, and the means are drawn afterwards from their
//! Gaussian conditionals. Given the structure parameters, each node's
//! posterior mean is the credibility formula `Z M + (1 − Z) E[parent]`,
//! with `M` the credibility-weighted mean of its children and `Z` its
//! credibility factor, so [`credibility_premiums`] at `actuar::cm`'s
//! estimates reproduces its premiums (Bühlmann–Straub with one level,
//! Bühlmann–Gisler's hierarchical model with more), and the Bayesian
//! premium is that formula averaged over the posterior of the structure
//! parameters.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_math::special::norm_quantile;
use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};

use crate::nuts::{LogDensity, sample};
pub use crate::nuts::{ParamSummary, PosteriorDraws, Sampler};

/// A prior on a standard deviation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalePrior {
    /// Half-normal with this scale.
    HalfNormal(f64),
    /// Half-Cauchy with this scale.
    HalfCauchy(f64),
}

impl ScalePrior {
    fn scale(self) -> f64 {
        match self {
            ScalePrior::HalfNormal(s) | ScalePrior::HalfCauchy(s) => s,
        }
    }

    /// Log density at `x > 0`, up to a constant.
    fn log_density<T: Real>(self, x: T) -> T {
        match self {
            ScalePrior::HalfNormal(s) => -(x * x) / T::from(2.0 * s * s),
            ScalePrior::HalfCauchy(s) => -(T::from(1.0) + (x * x) / T::from(s * s)).ln(),
        }
    }
}

/// Priors of [`HierarchicalCredibility`]. Fields left `None` (or `between`
/// empty) take weakly informative defaults from the data: half-normal
/// priors five times the spread the data show (`σ`: five times the pooled
/// within standard deviation; each `τ_l`: five times the standard
/// deviation of that level's weighted means, or of the groups' means when
/// a level has a single node), and a flat prior on `μ`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CredibilityPriors {
    /// `Some((mean, sd))` for a normal prior on the collective mean `μ`;
    /// `None` for a flat one.
    pub collective_mean: Option<(f64, f64)>,
    /// The prior on `σ`, the within standard deviation per unit of weight.
    pub within: Option<ScalePrior>,
    /// The priors on the between standard deviations, outermost level
    /// first and the groups' last; empty for the defaults.
    pub between: Vec<ScalePrior>,
}

/// Bayesian hierarchical credibility: priors and sampler.
///
/// ```
/// use prospicio_bayes::credibility::{HierarchicalCredibility, Sampler};
///
/// // Three groups of four years, equal weights.
/// let x = vec![
///     vec![1.0, 2.0, 3.0, 2.0],
///     vec![4.0, 5.0, 6.0, 5.0],
///     vec![2.0, 3.0, 4.0, 3.0],
/// ];
/// let w = vec![vec![1.0; 4]; 3];
/// let spec = HierarchicalCredibility::default()
///     .sampler(Sampler { chains: 2, tune: 300, draws: 300, ..Sampler::default() });
/// let fit = spec.fit(&x, &w, &[]).unwrap();
/// // Each premium lies between the group's own mean and the collective.
/// assert!(fit.premiums[1] < 5.0 && fit.premiums[1] > fit.collective_mean);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HierarchicalCredibility {
    pub priors: CredibilityPriors,
    pub sampler: Sampler,
}

/// A fitted [`HierarchicalCredibility`].
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalCredibilityFit {
    /// Posterior mean of each group's risk premium `θ_i`.
    pub premiums: Vec<f64>,
    /// Posterior mean of each group's credibility factor
    /// `Z_i = w_i τ_L² / (w_i τ_L² + σ²)`.
    pub credibility: Vec<f64>,
    /// Posterior mean of each node's mean, level by level (outermost
    /// first); empty without a hierarchy.
    pub level_premiums: Vec<Vec<f64>>,
    /// Posterior mean of the collective mean `μ`.
    pub collective_mean: f64,
    /// Each group's weighted mean `X̄_i` (0 for a group with no weight).
    pub group_means: Vec<f64>,
    /// Each group's total weight `w_i`.
    pub group_weights: Vec<f64>,
    /// The priors used, with the defaults filled in.
    pub priors: CredibilityPriors,
    /// Draws of `log σ` and the `log τ_l`, outermost level first.
    pub draws: PosteriorDraws,
    tree: Tree,
}

/// Credibility premiums at given structure parameters, from
/// [`credibility_premiums`].
#[derive(Debug, Clone, PartialEq)]
pub struct CredibilityPremiums {
    /// Each group's premium `Z_i X̄_i + (1 − Z_i) E[parent]`.
    pub premiums: Vec<f64>,
    /// Each group's credibility factor.
    pub credibility: Vec<f64>,
    /// Each node's premium, level by level (outermost first).
    pub level_premiums: Vec<Vec<f64>>,
    /// Each node's credibility factor, level by level.
    pub level_credibility: Vec<Vec<f64>>,
    /// The collective mean: the credibility-weighted mean of the top level
    /// with a flat prior on `μ`, shrunk to the prior's mean otherwise.
    pub collective_mean: f64,
}

/// The credibility premiums at structure parameters `within_variance`
/// (`σ²`, `actuar`'s `s²`) and `between_variances` (`τ_l²`, outermost level
/// first and the groups' last; one more than `levels`), with a flat prior
/// on `μ` unless `collective_mean` gives one. With one level these are
/// Bühlmann–Straub's premiums; with more, Bühlmann–Gisler's hierarchical
/// ones (as `actuar::cm`), which are the posterior means of the normal
/// model with those variances known.
///
/// `ratios` and `weights` are as for [`HierarchicalCredibility::fit`];
/// `levels[k][i]` is group `i`'s node at level `k` (outermost first).
///
/// ```
/// use prospicio_bayes::credibility::credibility_premiums;
///
/// let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 9.0]];
/// let w = vec![vec![1.0; 3]; 2];
/// // s² = 4 and a = 8 - 4/3, Bühlmann's estimates on these data.
/// let p = credibility_premiums(&x, &w, &[], 4.0, &[8.0 - 4.0 / 3.0], None).unwrap();
/// let z = 3.0 / (3.0 + 4.0 / (8.0 - 4.0 / 3.0));
/// assert!((p.credibility[0] - z).abs() < 1e-12);
/// assert!((p.collective_mean - 4.0).abs() < 1e-12);
/// assert!((p.premiums[0] - (z * 2.0 + (1.0 - z) * 4.0)).abs() < 1e-12);
/// ```
pub fn credibility_premiums(
    ratios: &[Vec<f64>],
    weights: &[Vec<f64>],
    levels: &[Vec<usize>],
    within_variance: f64,
    between_variances: &[f64],
    collective_mean: Option<(f64, f64)>,
) -> Result<CredibilityPremiums> {
    let tree = Tree::new(ratios, weights, levels)?;
    if between_variances.len() != tree.n_levels() {
        return Err(Error::Data(format!(
            "{} between variances for {} levels",
            between_variances.len(),
            tree.n_levels()
        )));
    }
    if !(within_variance.is_finite() && within_variance > 0.0) {
        return Err(Error::InvalidParameter {
            name: "within_variance",
            value: within_variance,
            reason: "must be positive",
        });
    }
    if let Some(&bad) = between_variances
        .iter()
        .find(|v| !(v.is_finite() && **v > 0.0))
    {
        return Err(Error::InvalidParameter {
            name: "between_variances",
            value: bad,
            reason: "must be positive",
        });
    }
    let sd: Vec<f64> = between_variances.iter().map(|v| v.sqrt()).collect();
    Ok(tree.conditional(within_variance.sqrt(), &sd, collective_mean))
}

impl HierarchicalCredibility {
    pub fn priors(mut self, priors: CredibilityPriors) -> Self {
        self.priors = priors;
        self
    }

    pub fn sampler(mut self, sampler: Sampler) -> Self {
        self.sampler = sampler;
        self
    }

    /// Fits to `ratios[i][j]` (group `i`, period `j`) with `weights` of the
    /// same shape; a missing cell is NaN in both. `levels` describes the
    /// hierarchy above the groups, outermost first: `levels[k][i]` is the
    /// node (from 0) group `i` belongs to at level `k`, and groups in the
    /// same node at a level must share their node at every level above.
    /// Empty `levels` is Bühlmann–Straub's model. Needs at least two groups
    /// with weight and more observations than such groups.
    pub fn fit(
        &self,
        ratios: &[Vec<f64>],
        weights: &[Vec<f64>],
        levels: &[Vec<usize>],
    ) -> Result<HierarchicalCredibilityFit> {
        let tree = Tree::new(ratios, weights, levels)?;
        let priors = self.filled_priors(&tree)?;
        let density = Posterior {
            tree: &tree,
            priors: &priors,
        };
        // Start at the classical spreads.
        let mut start = vec![0.5 * tree.pooled_within_variance().ln()];
        start.extend(
            priors
                .between
                .iter()
                .map(|p| (p.scale() / 5.0).max(1e-8).ln()),
        );
        let draws = sample(&density, &start, self.sampler)?;
        let names = std::iter::once("log_within_sd".to_string())
            .chain((0..tree.n_levels()).map(|l| format!("log_between_sd[{l}]")))
            .collect();
        let draws = draws.with_names(names)?;

        let n = draws.n_draws() as f64;
        let mut premiums = vec![0.0; tree.groups.len()];
        let mut credibility = vec![0.0; tree.groups.len()];
        let mut level_premiums: Vec<Vec<f64>> =
            tree.nodes.iter().map(|l| vec![0.0; l.len()]).collect();
        let mut collective_mean = 0.0;
        for d in 0..draws.n_draws() {
            let theta = draws.draw(d);
            let (sigma, tau) = hyper(theta);
            let c = tree.conditional(sigma, &tau, priors.collective_mean);
            for (acc, v) in premiums.iter_mut().zip(&c.premiums) {
                *acc += v / n;
            }
            for (acc, v) in credibility.iter_mut().zip(&c.credibility) {
                *acc += v / n;
            }
            for (acc, lv) in level_premiums.iter_mut().zip(&c.level_premiums) {
                for (a, v) in acc.iter_mut().zip(lv) {
                    *a += v / n;
                }
            }
            collective_mean += c.collective_mean / n;
        }
        Ok(HierarchicalCredibilityFit {
            premiums,
            credibility,
            level_premiums,
            collective_mean,
            group_means: tree.groups.iter().map(|g| g.mean).collect(),
            group_weights: tree.groups.iter().map(|g| g.weight).collect(),
            priors,
            draws,
            tree,
        })
    }

    fn filled_priors(&self, tree: &Tree) -> Result<CredibilityPriors> {
        let p = &self.priors;
        if let Some((_, s)) = p
            .collective_mean
            .filter(|&(m, s)| !(m.is_finite() && s.is_finite() && s > 0.0))
        {
            return Err(Error::InvalidParameter {
                name: "collective_mean",
                value: s,
                reason: "needs a finite mean and a positive sd",
            });
        }
        let check = |prior: ScalePrior| -> Result<ScalePrior> {
            let s = prior.scale();
            if s.is_finite() && s > 0.0 {
                Ok(prior)
            } else {
                Err(Error::InvalidParameter {
                    name: "scale",
                    value: s,
                    reason: "a scale prior needs a positive, finite scale",
                })
            }
        };
        let within = match p.within {
            Some(w) => check(w)?,
            None => ScalePrior::HalfNormal(5.0 * tree.pooled_within_variance().sqrt()),
        };
        let between = if p.between.is_empty() {
            tree.level_spreads()
                .into_iter()
                .map(|s| ScalePrior::HalfNormal(5.0 * s))
                .collect()
        } else if p.between.len() == tree.n_levels() {
            p.between
                .iter()
                .map(|&b| check(b))
                .collect::<Result<Vec<_>>>()?
        } else {
            return Err(Error::Data(format!(
                "{} between priors for {} levels",
                p.between.len(),
                tree.n_levels()
            )));
        };
        Ok(CredibilityPriors {
            collective_mean: p.collective_mean,
            within: Some(within),
            between,
        })
    }
}

impl HierarchicalCredibilityFit {
    /// Posterior summaries of the structure parameters on their natural
    /// scale: `within_variance` (`σ²`, `actuar`'s `s²`) and
    /// `between_variance[l]` (`τ_l²`, outermost level first, the groups'
    /// last: `actuar`'s `a` with one level, `b` then `a` with two).
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        let names = std::iter::once("within_variance".to_string())
            .chain((0..self.tree.n_levels()).map(|l| format!("between_variance[{l}]")))
            .collect();
        self.draws
            .transform(names, |x| x.iter().map(|v| (2.0 * v).exp()).collect())?
            .summary()
    }

    /// Divergent transitions over all chains.
    pub fn divergences(&self) -> usize {
        self.draws.divergences().iter().sum()
    }

    /// Posterior predictive draws by group, dimension `["group"]` with the
    /// group's index: each simulation picks a posterior draw of the
    /// structure parameters, draws every mean from its Gaussian
    /// conditional, then, with `weights`, a next-period ratio
    /// `N(θ_i, σ² / w_i)` for each group (`weights[i]` its weight next
    /// period); without, the risk premium `θ_i` itself. Simulation `s`
    /// uses stream `s` of `seed`.
    pub fn predictive(
        &self,
        weights: Option<&[f64]>,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        let k = self.tree.groups.len();
        if let Some(w) = weights {
            if w.len() != k {
                return Err(Error::Data(format!("{} weights for {k} groups", w.len())));
            }
            if let Some(&bad) = w.iter().find(|v| !(v.is_finite() && **v > 0.0)) {
                return Err(Error::InvalidParameter {
                    name: "weights",
                    value: bad,
                    reason: "must be positive",
                });
            }
        }
        let components = (0..k).map(|i| vec![KeyValue::Int(i as i64)]).collect();
        let provenance = Provenance::new("hierarchical_credibility")
            .version("prospicio-bayes", env!("CARGO_PKG_VERSION"))
            .param("groups", k)
            .param("levels", self.tree.n_levels())
            .param(
                "outcome",
                if weights.is_some() {
                    "next-period ratio"
                } else {
                    "risk premium"
                },
            );
        let cm = self.priors.collective_mean;
        self.draws.predictive(
            vec!["group".into()],
            components,
            n_sims,
            seed,
            provenance,
            |theta, rng, row| {
                let (sigma, tau) = hyper(theta);
                let means = self.tree.draw_means(sigma, &tau, cm, rng);
                for (i, out) in row.iter_mut().enumerate() {
                    *out = match weights {
                        Some(w) => {
                            means[i] + sigma / w[i].sqrt() * norm_quantile(rng.next_open01())
                        }
                        None => means[i],
                    };
                }
            },
        )
    }
}

/// `σ` and the `τ_l` from an unconstrained draw.
fn hyper(theta: &[f64]) -> (f64, Vec<f64>) {
    (theta[0].exp(), theta[1..].iter().map(|v| v.exp()).collect())
}

/// One group's sufficient statistics.
#[derive(Debug, Clone, PartialEq)]
struct Group {
    mean: f64,
    weight: f64,
    /// Observations with weight.
    n: usize,
    /// `Σ w_ij (X_ij − X̄_i)²`.
    ss: f64,
}

/// The groups and the hierarchy above them.
#[derive(Debug, Clone, PartialEq)]
struct Tree {
    groups: Vec<Group>,
    /// `group_parent[i]`: group `i`'s node at the innermost hierarchy
    /// level (unused without a hierarchy).
    group_parent: Vec<usize>,
    /// `nodes[l][k]`: node `k` at level `l`'s parent at level `l − 1`
    /// (0 at the top level, where the parent is `μ`).
    nodes: Vec<Vec<usize>>,
}

/// A Gaussian message about a node's value: its data say `N(mean; value,
/// 1 / precision)`. A node with no data has precision 0.
#[derive(Debug, Clone, Copy)]
struct Message<T> {
    mean: T,
    precision: T,
}

impl Tree {
    fn new(ratios: &[Vec<f64>], weights: &[Vec<f64>], levels: &[Vec<usize>]) -> Result<Self> {
        let k = ratios.len();
        if weights.len() != k || weights.iter().zip(ratios).any(|(w, r)| w.len() != r.len()) {
            return Err(Error::Data("weights must have the shape of ratios".into()));
        }
        let mut groups = Vec::with_capacity(k);
        for (r, w) in ratios.iter().zip(weights) {
            let (mut ws, mut xs, mut n) = (0.0, 0.0, 0);
            for (&x, &v) in r.iter().zip(w) {
                if x.is_nan() != v.is_nan() {
                    return Err(Error::Data(
                        "missing values must be in the same places in ratios and weights".into(),
                    ));
                }
                if x.is_nan() {
                    continue;
                }
                if !(x.is_finite() && v.is_finite() && v >= 0.0) {
                    return Err(Error::InvalidParameter {
                        name: "weights",
                        value: v,
                        reason: "ratios must be finite and weights finite and non-negative",
                    });
                }
                if v > 0.0 {
                    ws += v;
                    xs += v * x;
                    n += 1;
                }
            }
            let mean = if ws > 0.0 { xs / ws } else { 0.0 };
            let ss = r
                .iter()
                .zip(w)
                .filter(|(x, v)| !x.is_nan() && **v > 0.0)
                .map(|(x, v)| v * (x - mean).powi(2))
                .sum();
            groups.push(Group {
                mean,
                weight: ws,
                n,
                ss,
            });
        }
        let with_weight = groups.iter().filter(|g| g.weight > 0.0).count();
        let observations: usize = groups.iter().map(|g| g.n).sum();
        if with_weight < 2 || observations <= with_weight {
            return Err(Error::Data(
                "needs two groups with weight and more observations than groups".into(),
            ));
        }
        // The hierarchy: each level's nodes and their parents.
        let mut nodes = Vec::with_capacity(levels.len());
        for (l, level) in levels.iter().enumerate() {
            if level.len() != k {
                return Err(Error::Data(format!(
                    "level {l} names {} nodes for {k} groups",
                    level.len()
                )));
            }
            let count = level.iter().max().map_or(0, |m| m + 1);
            let mut parent = vec![usize::MAX; count];
            for (i, &node) in level.iter().enumerate() {
                let up = if l == 0 { 0 } else { levels[l - 1][i] };
                if parent[node] == usize::MAX {
                    parent[node] = up;
                } else if parent[node] != up {
                    return Err(Error::Data(format!(
                        "node {node} of level {l} lies in two nodes of level {}",
                        l - 1
                    )));
                }
            }
            if let Some(empty) = parent.iter().position(|&p| p == usize::MAX) {
                return Err(Error::Data(format!(
                    "node {empty} of level {l} has no group"
                )));
            }
            nodes.push(parent);
        }
        let group_parent = levels.last().cloned().unwrap_or_else(|| vec![0; k]);
        Ok(Self {
            groups,
            group_parent,
            nodes,
        })
    }

    /// Random-effect levels: the hierarchy's and the groups'.
    fn n_levels(&self) -> usize {
        self.nodes.len() + 1
    }

    /// `Σ w (X − X̄_i)² / (n − I)`, the classical within variance.
    fn pooled_within_variance(&self) -> f64 {
        let with_weight = self.groups.iter().filter(|g| g.weight > 0.0).count();
        let n: usize = self.groups.iter().map(|g| g.n).sum();
        let ss: f64 = self.groups.iter().map(|g| g.ss).sum();
        let v = ss / (n - with_weight) as f64;
        if v > 0.0 { v } else { 1.0 }
    }

    /// The spread of each level's means, outermost first and the groups'
    /// last: the standard deviation of the nodes' weighted means, or of
    /// the groups' means where a level has fewer than two nodes with
    /// weight.
    fn level_spreads(&self) -> Vec<f64> {
        let sd = |v: &[f64]| -> f64 {
            if v.len() < 2 {
                return f64::NAN;
            }
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
        };
        let group_means: Vec<f64> = self
            .groups
            .iter()
            .filter(|g| g.weight > 0.0)
            .map(|g| g.mean)
            .collect();
        let fallback = {
            let s = sd(&group_means);
            if s > 0.0 {
                s
            } else {
                self.pooled_within_variance().sqrt()
            }
        };
        let mut out = Vec::with_capacity(self.n_levels());
        for l in 0..self.nodes.len() {
            let count = self.nodes[l].len();
            let (mut ws, mut xs) = (vec![0.0; count], vec![0.0; count]);
            for (i, g) in self.groups.iter().enumerate() {
                let node = self.node_of(i, l);
                ws[node] += g.weight;
                xs[node] += g.weight * g.mean;
            }
            let means: Vec<f64> = ws
                .iter()
                .zip(&xs)
                .filter(|(w, _)| **w > 0.0)
                .map(|(w, x)| x / w)
                .collect();
            let s = sd(&means);
            out.push(if s > 0.0 { s } else { fallback });
        }
        out.push(fallback);
        out
    }

    /// Group `i`'s node at level `l`.
    fn node_of(&self, i: usize, l: usize) -> usize {
        let mut node = self.group_parent[i];
        for up in (l + 1..self.nodes.len()).rev() {
            node = self.nodes[up][node];
        }
        node
    }

    /// The messages up the tree at `σ` and the `τ_l`: each group's, then
    /// each level's nodes' (innermost level first in the computation,
    /// stored outermost first), then the top's about `μ`; and the log of
    /// the integrated likelihood, up to a constant.
    #[allow(clippy::type_complexity)]
    fn upward<T: Real>(
        &self,
        sigma: T,
        tau: &[T],
    ) -> (Vec<Message<T>>, Vec<Vec<Message<T>>>, Message<T>, T) {
        let zero = T::from(0.0);
        let two_pi = T::from(2.0 * std::f64::consts::PI);
        let s2 = sigma * sigma;
        let mut log_lik = zero;
        // The within part: σ^-(n_i - 1) exp(-SS_i / 2σ²).
        for g in &self.groups {
            if g.weight > 0.0 {
                log_lik = log_lik
                    - T::from((g.n - 1) as f64) * sigma.ln()
                    - T::from(g.ss) / (T::from(2.0) * s2);
            }
        }
        let leaves: Vec<Message<T>> = self
            .groups
            .iter()
            .map(|g| Message {
                mean: T::from(g.mean),
                precision: T::from(g.weight) / s2,
            })
            .collect();
        // Combines children's messages into one about their parent, whose
        // children have between sd `t`; adds the normalizing constant.
        let mut combine = |children: &mut dyn Iterator<Item = (usize, Message<T>)>,
                           count: usize,
                           t: T|
         -> Vec<Message<T>> {
            let t2 = t * t;
            let mut acc: Vec<(T, T, Vec<(T, T)>)> =
                (0..count).map(|_| (zero, zero, Vec::new())).collect();
            for (parent, m) in children {
                if m.precision.value() <= 0.0 {
                    continue;
                }
                let s = T::from(1.0) / m.precision + t2;
                let a = &mut acc[parent];
                a.0 = a.0 + T::from(1.0) / s;
                a.1 = a.1 + m.mean / s;
                a.2.push((m.mean, s));
            }
            acc.into_iter()
                .map(|(p, ms, kids)| {
                    if kids.is_empty() {
                        return Message {
                            mean: zero,
                            precision: zero,
                        };
                    }
                    let mean = ms / p;
                    for (m, s) in kids {
                        let d = m - mean;
                        log_lik =
                            log_lik - T::from(0.5) * (two_pi * s).ln() - d * d / (T::from(2.0) * s);
                    }
                    log_lik = log_lik + T::from(0.5) * (two_pi / p).ln();
                    Message { mean, precision: p }
                })
                .collect()
        };
        let depth = self.nodes.len();
        let mut levels: Vec<Vec<Message<T>>> = vec![Vec::new(); depth];
        let top = if depth == 0 {
            let mut it = leaves.iter().map(|&m| (0, m));
            combine(&mut it, 1, tau[0])[0]
        } else {
            let mut it = leaves.iter().zip(&self.group_parent).map(|(&m, &p)| (p, m));
            levels[depth - 1] = combine(&mut it, self.nodes[depth - 1].len(), tau[depth]);
            for l in (1..depth).rev() {
                let below = levels[l].clone();
                let mut it = below.into_iter().zip(&self.nodes[l]).map(|(m, &p)| (p, m));
                levels[l - 1] = combine(&mut it, self.nodes[l - 1].len(), tau[l]);
            }
            let below = levels[0].clone();
            let mut it = below.into_iter().map(|m| (0, m));
            combine(&mut it, 1, tau[0])[0]
        };
        (leaves, levels, top, log_lik)
    }

    /// Credibility factor of a node with message `m` and between sd `t`.
    fn factor(m: &Message<f64>, t: f64) -> f64 {
        let p = m.precision * t * t;
        p / (p + 1.0)
    }

    /// The posterior mean of `μ` given the top's message.
    fn mu_mean(top: &Message<f64>, prior: Option<(f64, f64)>) -> f64 {
        match prior {
            None => top.mean,
            Some((m0, s0)) => {
                let q = 1.0 / (s0 * s0);
                (top.mean * top.precision + m0 * q) / (top.precision + q)
            }
        }
    }

    /// The posterior means (and factors) given `σ` and the `τ_l`.
    fn conditional(
        &self,
        sigma: f64,
        tau: &[f64],
        prior: Option<(f64, f64)>,
    ) -> CredibilityPremiums {
        let (leaves, levels, top, _) = self.upward(sigma, tau);
        let mu = Self::mu_mean(&top, prior);
        let depth = self.nodes.len();
        let mut level_premiums: Vec<Vec<f64>> = Vec::with_capacity(depth);
        let mut level_credibility = Vec::with_capacity(depth);
        for l in 0..depth {
            let (mut prem, mut cred) = (Vec::new(), Vec::new());
            for (k, m) in levels[l].iter().enumerate() {
                let parent = if l == 0 {
                    mu
                } else {
                    level_premiums[l - 1][self.nodes[l][k]]
                };
                let z = Self::factor(m, tau[l]);
                prem.push(z * m.mean + (1.0 - z) * parent);
                cred.push(z);
            }
            level_premiums.push(prem);
            level_credibility.push(cred);
        }
        let (mut premiums, mut credibility) = (Vec::new(), Vec::new());
        for (i, m) in leaves.iter().enumerate() {
            let parent = if depth == 0 {
                mu
            } else {
                level_premiums[depth - 1][self.group_parent[i]]
            };
            let z = Self::factor(m, tau[depth]);
            premiums.push(z * m.mean + (1.0 - z) * parent);
            credibility.push(z);
        }
        CredibilityPremiums {
            premiums,
            credibility,
            level_premiums,
            level_credibility,
            collective_mean: mu,
        }
    }

    /// Every group's mean drawn from the joint posterior given `σ` and the
    /// `τ_l`: `μ` first, then down the tree, each node from its Gaussian
    /// conditional on its parent.
    fn draw_means(
        &self,
        sigma: f64,
        tau: &[f64],
        prior: Option<(f64, f64)>,
        rng: &mut StreamRng,
    ) -> Vec<f64> {
        let (leaves, levels, top, _) = self.upward(sigma, tau);
        let mut z = || norm_quantile(rng.next_open01());
        let mu = match prior {
            None => top.mean + z() / top.precision.sqrt(),
            Some((_, s0)) => {
                let p = top.precision + 1.0 / (s0 * s0);
                Self::mu_mean(&top, prior) + z() / p.sqrt()
            }
        };
        // A node given its parent: precision 1/τ² + P, mean weighted.
        let mut child = |m: &Message<f64>, parent: f64, t: f64| -> f64 {
            let q = 1.0 / (t * t);
            let p = q + m.precision;
            (parent * q + m.mean * m.precision) / p + z() / p.sqrt()
        };
        let depth = self.nodes.len();
        let mut values: Vec<Vec<f64>> = Vec::with_capacity(depth);
        for l in 0..depth {
            let row: Vec<f64> = levels[l]
                .iter()
                .enumerate()
                .map(|(k, m)| {
                    let parent = if l == 0 {
                        mu
                    } else {
                        values[l - 1][self.nodes[l][k]]
                    };
                    child(m, parent, tau[l])
                })
                .collect();
            values.push(row);
        }
        leaves
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let parent = if depth == 0 {
                    mu
                } else {
                    values[depth - 1][self.group_parent[i]]
                };
                child(m, parent, tau[depth])
            })
            .collect()
    }
}

/// The posterior of `log σ` and the `log τ_l`, the means integrated out.
struct Posterior<'a> {
    tree: &'a Tree,
    priors: &'a CredibilityPriors,
}

impl Posterior<'_> {
    fn value<T: Real>(&self, x: &[T]) -> T {
        let sigma = x[0].exp();
        let tau: Vec<T> = x[1..].iter().map(|v| v.exp()).collect();
        let (_, _, top, mut lp) = self.tree.upward(sigma, &tau);
        if let Some((m0, s0)) = self.priors.collective_mean {
            // ∫ N(M; μ, 1/P) N(μ; m0, s0²) dμ = N(M; m0, 1/P + s0²).
            let v = T::from(1.0) / top.precision + T::from(s0 * s0);
            let d = top.mean - T::from(m0);
            lp = lp - T::from(0.5) * v.ln() - d * d / (T::from(2.0) * v);
        }
        // Priors on the scales, with the log Jacobians of the log maps.
        let within = self.priors.within.expect("filled in");
        lp = lp + within.log_density(sigma) + x[0];
        for ((prior, t), &xl) in self.priors.between.iter().zip(&tau).zip(&x[1..]) {
            lp = lp + prior.log_density(*t) + xl;
        }
        lp
    }
}

impl LogDensity for Posterior<'_> {
    fn dim(&self) -> usize {
        1 + self.tree.n_levels()
    }

    fn log_density(&self, position: &[f64], gradient: &mut [f64]) -> Option<f64> {
        // Forward mode, one coordinate a pass: the dimension is small.
        let mut value = f64::NAN;
        for (j, g) in gradient.iter_mut().enumerate() {
            let x: Vec<Dual> = position
                .iter()
                .enumerate()
                .map(|(i, &v)| Dual {
                    v,
                    d: if i == j { 1.0 } else { 0.0 },
                })
                .collect();
            let lp = self.value(&x);
            value = lp.v;
            *g = lp.d;
        }
        value.is_finite().then_some(value)
    }
}

/// What the upward pass needs of a number: `f64` for values, [`Dual`] for
/// a derivative.
trait Real:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    fn from(v: f64) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn value(self) -> f64;
}

impl Real for f64 {
    fn from(v: f64) -> Self {
        v
    }
    fn ln(self) -> Self {
        f64::ln(self)
    }
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn value(self) -> f64 {
        self
    }
}

/// A number and its derivative along one direction.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Dual {
    v: f64,
    d: f64,
}

impl std::ops::Add for Dual {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Dual {
            v: self.v + o.v,
            d: self.d + o.d,
        }
    }
}

impl std::ops::Sub for Dual {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Dual {
            v: self.v - o.v,
            d: self.d - o.d,
        }
    }
}

impl std::ops::Mul for Dual {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Dual {
            v: self.v * o.v,
            d: self.d * o.v + self.v * o.d,
        }
    }
}

impl std::ops::Div for Dual {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        Dual {
            v: self.v / o.v,
            d: (self.d * o.v - self.v * o.d) / (o.v * o.v),
        }
    }
}

impl std::ops::Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Dual {
            v: -self.v,
            d: -self.d,
        }
    }
}

impl Real for Dual {
    fn from(v: f64) -> Self {
        Dual { v, d: 0.0 }
    }
    fn ln(self) -> Self {
        Dual {
            v: self.v.ln(),
            d: self.d / self.v,
        }
    }
    fn exp(self) -> Self {
        let e = self.v.exp();
        Dual {
            v: e,
            d: self.d * e,
        }
    }
    fn value(self) -> f64 {
        self.v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let x = vec![
            vec![1.0, 2.0, 3.0, 2.5],
            vec![4.0, 5.0, 9.0, 6.0],
            vec![2.0, 2.5, f64::NAN, 3.5],
            vec![7.0, 6.0, 8.0, 7.5],
        ];
        let w = vec![
            vec![1.0, 2.0, 1.5, 1.0],
            vec![2.0, 2.0, 1.0, 3.0],
            vec![1.0, 0.5, f64::NAN, 2.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        (x, w)
    }

    #[test]
    fn the_gradient_matches_finite_differences() {
        let (x, w) = data();
        for levels in [
            vec![],
            vec![vec![0, 0, 1, 1]],
            vec![vec![0, 0, 0, 0], vec![0, 0, 1, 1]],
        ] {
            let tree = Tree::new(&x, &w, &levels).unwrap();
            for cm in [None, Some((3.0, 2.0))] {
                let spec = HierarchicalCredibility::default().priors(CredibilityPriors {
                    collective_mean: cm,
                    within: Some(ScalePrior::HalfCauchy(2.0)),
                    between: Vec::new(),
                });
                let priors = spec.filled_priors(&tree).unwrap();
                let post = Posterior {
                    tree: &tree,
                    priors: &priors,
                };
                let p: Vec<f64> = (0..post.dim()).map(|j| 0.3 - 0.4 * j as f64).collect();
                let mut g = vec![0.0; p.len()];
                post.log_density(&p, &mut g).unwrap();
                for j in 0..p.len() {
                    let h = 1e-6;
                    let (mut a, mut b) = (p.clone(), p.clone());
                    a[j] += h;
                    b[j] -= h;
                    let fd = (post.value(&a) - post.value(&b)) / (2.0 * h);
                    assert!(
                        (fd - g[j]).abs() < 1e-6 * (1.0 + fd.abs()),
                        "{fd} vs {}",
                        g[j]
                    );
                }
            }
        }
    }

    #[test]
    fn the_integrated_likelihood_is_the_gaussian_marginal() {
        // One level, two groups of one observation each, flat μ: the
        // integrated likelihood in τ (σ fixed) is ∫ Π N(x_i; μ, σ²/w_i + τ²) dμ.
        let x = vec![vec![1.0, 2.0], vec![4.0, f64::NAN]];
        let w = vec![vec![1.0, 1.0], vec![2.0, f64::NAN]];
        let tree = Tree::new(&x, &w, &[]).unwrap();
        let (sigma, tau) = (0.7_f64, 1.3_f64);
        let (_, _, _, lp) = tree.upward(sigma, &[tau]);
        // By quadrature over μ.
        let v1 = sigma * sigma / 2.0 + tau * tau; // group 1: mean 1.5, weight 2
        let v2 = sigma * sigma / 2.0 + tau * tau;
        let norm = |x: f64, m: f64, v: f64| {
            (-(x - m).powi(2) / (2.0 * v)).exp() / (2.0 * std::f64::consts::PI * v).sqrt()
        };
        let (mut q, h) = (0.0, 1e-3);
        let mut mu = -30.0;
        while mu < 30.0 {
            q += norm(1.5, mu, v1) * norm(4.0, mu, v2) * h;
            mu += h;
        }
        // Within: group 1 has n = 2, SS = 0.5; σ^-(1) exp(-0.5/(2σ²)).
        let within = -sigma.ln() - 0.5 / (2.0 * sigma * sigma);
        assert!(
            (lp - (q.ln() + within)).abs() < 1e-8,
            "{lp} vs {}",
            q.ln() + within
        );
    }

    #[test]
    fn premiums_are_the_credibility_formula() {
        let (x, w) = data();
        let (s2, a) = (2.0, 3.0);
        let p = credibility_premiums(&x, &w, &[], s2, &[a], None).unwrap();
        let tree = Tree::new(&x, &w, &[]).unwrap();
        let z: Vec<f64> = tree
            .groups
            .iter()
            .map(|g| g.weight * a / (g.weight * a + s2))
            .collect();
        let mu = z
            .iter()
            .zip(&tree.groups)
            .map(|(z, g)| z * g.mean)
            .sum::<f64>()
            / z.iter().sum::<f64>();
        assert!((p.collective_mean - mu).abs() < 1e-12);
        for (i, g) in tree.groups.iter().enumerate() {
            assert!((p.credibility[i] - z[i]).abs() < 1e-12);
            assert!((p.premiums[i] - (z[i] * g.mean + (1.0 - z[i]) * mu)).abs() < 1e-12);
        }
        // A tight prior on μ pulls the collective to it.
        let tight = credibility_premiums(&x, &w, &[], s2, &[a], Some((100.0, 1e-6))).unwrap();
        assert!((tight.collective_mean - 100.0).abs() < 1e-6);
    }

    #[test]
    fn a_group_without_weight_takes_its_parents_premium() {
        let (mut x, mut w) = data();
        x.push(vec![f64::NAN; 4]);
        w.push(vec![f64::NAN; 4]);
        let p =
            credibility_premiums(&x, &w, &[vec![0, 0, 1, 1, 1]], 2.0, &[1.0, 3.0], None).unwrap();
        assert_eq!(p.credibility[4], 0.0);
        assert!((p.premiums[4] - p.level_premiums[0][1]).abs() < 1e-12);
    }

    #[test]
    fn refuses_a_broken_hierarchy() {
        let (x, w) = data();
        // Groups 1 and 2 share node 0 below but not above.
        assert!(Tree::new(&x, &w, &[vec![0, 0, 1, 1], vec![0, 1, 1, 2]]).is_err());
        assert!(Tree::new(&x, &w, &[vec![0, 0, 2, 2]]).is_err());
        assert!(Tree::new(&x, &w, &[vec![0, 0, 1]]).is_err());
        assert!(Tree::new(&x[..1], &w[..1], &[]).is_err());
        assert!(credibility_premiums(&x, &w, &[], 1.0, &[1.0, 1.0], None).is_err());
    }

    #[test]
    fn sampling_recovers_simulated_structure() {
        // 30 groups of 8 periods: μ = 10, τ = 2, σ = 3 at weight 1.
        let mut rng = StreamRng::new(5, 0);
        let mut z = || norm_quantile(rng.next_open01());
        let (mut x, mut w) = (Vec::new(), Vec::new());
        for _ in 0..30 {
            let theta = 10.0 + 2.0 * z();
            x.push((0..8).map(|_| theta + 3.0 * z()).collect());
            w.push(vec![1.0; 8]);
        }
        let spec = HierarchicalCredibility::default().sampler(Sampler {
            chains: 2,
            tune: 400,
            draws: 400,
            seed: 3,
            ..Sampler::default()
        });
        let fit = spec.fit(&x, &w, &[]).unwrap();
        let s = fit.summary().unwrap();
        assert!(s.iter().all(|p| p.rhat < 1.05), "{s:?}");
        assert!((s[0].mean.sqrt() - 3.0).abs() < 0.4, "{s:?}");
        assert!((s[1].mean.sqrt() - 2.0).abs() < 0.8, "{s:?}");
        assert!((fit.collective_mean - 10.0).abs() < 1.2);
        // The predictive's means are the premiums.
        let pd = fit.predictive(None, 20_000, 9).unwrap();
        for i in [0, 7, 29] {
            let key = vec![KeyValue::Int(i)];
            let m = pd.marginal(&key).unwrap();
            let mean = prospicio_prob::Empirical::draws(&m).iter().sum::<f64>() / 20_000.0;
            assert!((mean - fit.premiums[i as usize]).abs() < 0.05, "{i}");
        }
        assert!(fit.predictive(Some(&[1.0; 3]), 10, 1).is_err());
    }
}
