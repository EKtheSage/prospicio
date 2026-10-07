//! The joint predictive distribution every model returns.

use std::collections::HashSet;
use std::fmt;
use std::sync::OnceLock;

use prospicio_core::{Error, Period, Result, StreamRng};
use rayon::prelude::*;

use crate::distortion::Distortion;
use crate::distribution::{Distribution, check_probability};
use crate::provenance::{Provenance, SIM_INDEX_SCHEME};
use crate::risk::var_sorted;
use crate::sampled::{Empirical, Sampled};

/// One value of a component key: an integer (a layer number), text (a line
/// of business) or a [`Period`] (an origin).
///
/// A `Period` key compares equal to the same origin in a Triangle, so a
/// reserve component joins back to its triangle row without conversion.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KeyValue {
    Int(i64),
    Text(String),
    Period(Period),
}

impl From<i64> for KeyValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<i32> for KeyValue {
    fn from(v: i32) -> Self {
        Self::Int(v.into())
    }
}

impl From<&str> for KeyValue {
    fn from(v: &str) -> Self {
        Self::Text(v.into())
    }
}

impl From<String> for KeyValue {
    fn from(v: String) -> Self {
        Self::Text(v)
    }
}

impl From<Period> for KeyValue {
    fn from(v: Period) -> Self {
        Self::Period(v)
    }
}

impl fmt::Display for KeyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Text(v) => f.write_str(v),
            Self::Period(v) => write!(f, "{v}"),
        }
    }
}

/// The key of one component: one value per dimension of the distribution,
/// in the distribution's dimension order, e.g. `["Auto", 2019]` for the
/// dimensions `["lob", "origin"]`.
pub type ComponentKey = Vec<KeyValue>;

/// A joint distribution over components (origin years, lines, layers),
/// stored as simulated draws.
///
/// Draws are simulation-major: row `i` holds every component's value in
/// simulation `i`, so the components keep their dependence. The total's
/// 99.5% quantile is computed from row sums, not by adding the components'
/// 99.5% quantiles. See `docs/design/predictive-distribution.md`.
///
/// The [`Distribution`] and [`Empirical`] methods describe the **total**
/// over all components. Use [`marginal`](Self::marginal) for one
/// component, or [`aggregate`](Self::aggregate) to sum over dimensions.
///
/// # Example
///
/// ```
/// use prospicio_prob::{Distribution, Empirical, Lognormal, PredictiveDistribution, Provenance};
///
/// let sev = Lognormal::from_mean_cv(100.0, 0.5).unwrap();
/// let pd = PredictiveDistribution::simulate(
///     vec!["origin".into()],
///     vec![vec![2023.into()], vec![2024.into()]],
///     10_000,
///     42,
///     Provenance::new("example"),
///     |rng, row| {
///         for cell in row {
///             *cell = sev.quantile(rng.next_open01()).unwrap();
///         }
///     },
/// )
/// .unwrap();
/// assert!((pd.mean() - 200.0).abs() < 2.0);
/// let origin_2024 = pd.marginal(&vec![2024.into()]).unwrap();
/// assert!(pd.tvar(0.995).unwrap() > origin_2024.tvar(0.995).unwrap());
/// ```
#[derive(Debug, Clone)]
pub struct PredictiveDistribution {
    dims: Vec<String>,
    components: Vec<ComponentKey>,
    n_sims: usize,
    /// `n_sims × components.len()`, simulation-major.
    draws: Vec<f64>,
    provenance: Provenance,
    /// Row sums, computed on first use.
    total: OnceLock<Sampled>,
}

impl PredictiveDistribution {
    /// A distribution from draws already laid out simulation-major:
    /// `draws[i * n_components + j]` is component `j` in simulation `i`.
    ///
    /// Fails if `dims` repeats a name, a key does not have one value per
    /// dimension, keys repeat, there are no components, `draws` is empty
    /// or not a whole number of rows, or a draw is not finite.
    pub fn from_draws(
        dims: Vec<String>,
        components: Vec<ComponentKey>,
        draws: Vec<f64>,
        provenance: Provenance,
    ) -> Result<Self> {
        validate_keys(&dims, &components)?;
        let n_components = components.len();
        if draws.is_empty() || draws.len() % n_components != 0 {
            return Err(Error::InvalidParameter {
                name: "draws",
                value: draws.len() as f64,
                reason: "length must be a positive multiple of the number of components",
            });
        }
        if let Some(&bad) = draws.iter().find(|x| !x.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "draws",
                value: bad,
                reason: "must all be finite",
            });
        }
        Ok(Self {
            dims,
            components,
            n_sims: draws.len() / n_components,
            draws,
            provenance,
            total: OnceLock::new(),
        })
    }

    /// Runs `n_sims` simulations in parallel and collects them.
    ///
    /// Simulation `i` calls `simulate(rng, row)` with
    /// `rng = StreamRng::new(seed, i)` and fills `row`, one value per
    /// component. Each simulation owns its stream, so the result is
    /// identical for any number of threads and any row can be replayed
    /// alone. The seed and [`SIM_INDEX_SCHEME`] are recorded in the
    /// provenance.
    pub fn simulate<F>(
        dims: Vec<String>,
        components: Vec<ComponentKey>,
        n_sims: usize,
        seed: u64,
        provenance: Provenance,
        simulate: F,
    ) -> Result<Self>
    where
        F: Fn(&mut StreamRng, &mut [f64]) + Sync,
    {
        Self::simulate_with(true, dims, components, n_sims, seed, provenance, simulate)
    }

    /// [`PredictiveDistribution::simulate`], run in parallel only when
    /// `parallel` is true; otherwise every simulation runs in order on the
    /// calling thread (for a closure that calls a [`crate::Custom`] whose
    /// callbacks must stay there). The draws are the same either way.
    pub fn simulate_with<F>(
        parallel: bool,
        dims: Vec<String>,
        components: Vec<ComponentKey>,
        n_sims: usize,
        seed: u64,
        provenance: Provenance,
        simulate: F,
    ) -> Result<Self>
    where
        F: Fn(&mut StreamRng, &mut [f64]) + Sync,
    {
        validate_keys(&dims, &components)?;
        if n_sims == 0 {
            return Err(Error::InvalidParameter {
                name: "n_sims",
                value: 0.0,
                reason: "must be positive",
            });
        }
        let mut draws = vec![0.0; n_sims * components.len()];
        let run =
            |(i, row): (usize, &mut [f64])| simulate(&mut StreamRng::new(seed, i as u64), row);
        if parallel {
            draws
                .par_chunks_mut(components.len())
                .enumerate()
                .for_each(run);
        } else {
            draws.chunks_mut(components.len()).enumerate().for_each(run);
        }
        Self::from_draws(
            dims,
            components,
            draws,
            provenance.seed(seed, SIM_INDEX_SCHEME),
        )
    }

    /// Dimension names, e.g. `["lob", "origin"]`.
    pub fn dims(&self) -> &[String] {
        &self.dims
    }

    /// Component keys, in column order.
    pub fn components(&self) -> &[ComponentKey] {
        &self.components
    }

    /// Number of simulations (rows).
    pub fn n_sims(&self) -> usize {
        self.n_sims
    }

    /// Number of components (columns).
    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    /// Where this result came from.
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Every component's value in simulation `sim`, or `None` past the end.
    pub fn row(&self, sim: usize) -> Option<&[f64]> {
        let n = self.n_components();
        self.draws.get(sim * n..(sim + 1) * n)
    }

    /// All draws, simulation-major: `draw_matrix()[i * n_components() + j]`
    /// is component `j` in simulation `i`. (The [`Empirical::draws`] of a
    /// `PredictiveDistribution` are the per-simulation totals instead.)
    pub fn draw_matrix(&self) -> &[f64] {
        &self.draws
    }

    /// Column of the component with this key.
    ///
    /// A key that matches no component exactly may still name a
    /// [`Period`] by its label, as text or an integer (`"2021"` or `2021`
    /// for a year, `"2021Q3"` for a quarter): bindings and files carry
    /// periods as labels. The label match is used only when it names one
    /// component.
    ///
    /// ```
    /// use prospicio_core::{Grain, Month, Period};
    /// use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
    ///
    /// let origin = |y| KeyValue::Period(Period::containing(Month::january(y), Grain::Year));
    /// let pd = PredictiveDistribution::from_draws(
    ///     vec!["origin".into()],
    ///     vec![vec![origin(2020)], vec![origin(2021)]],
    ///     vec![1.0, 2.0],
    ///     Provenance::new("example"),
    /// )
    /// .unwrap();
    /// assert_eq!(pd.component_index(&vec![KeyValue::from(2021)]), Some(1));
    /// assert_eq!(pd.component_index(&vec![KeyValue::from("2020")]), Some(0));
    /// assert_eq!(pd.component_index(&vec![KeyValue::from("2022")]), None);
    /// ```
    pub fn component_index(&self, key: &ComponentKey) -> Option<usize> {
        if let Some(j) = self.components.iter().position(|k| k == key) {
            return Some(j);
        }
        let by_label = |k: &ComponentKey| {
            k.len() == key.len() && k.iter().zip(key).all(|(c, v)| {
                c == v
                    || matches!((c, v), (KeyValue::Period(p), KeyValue::Int(_) | KeyValue::Text(_))
                            if p.to_string() == v.to_string())
            })
        };
        let mut matches = self
            .components
            .iter()
            .enumerate()
            .filter(|(_, k)| by_label(k));
        match (matches.next(), matches.next()) {
            (Some((j, _)), None) => Some(j),
            _ => None,
        }
    }

    /// One component's draws, in simulation order, or `None` if no
    /// component has this key.
    pub fn marginal(&self, key: &ComponentKey) -> Option<Sampled> {
        let j = self.component_index(key)?;
        let column = self
            .draws
            .iter()
            .skip(j)
            .step_by(self.n_components())
            .copied()
            .collect();
        Some(Sampled::new(column).expect("draws were validated as finite and non-empty"))
    }

    /// Sums the components within each simulation over every dimension not
    /// in `keep`, keeping the joint structure.
    ///
    /// `aggregate(&["lob"])` gives one component per line of business, summed
    /// over origins. `aggregate(&[])` gives a single component, the total.
    /// Groups appear in the order their first component appears.
    pub fn aggregate(&self, keep: &[&str]) -> Result<Self> {
        let mut kept = Vec::with_capacity(keep.len());
        for (i, name) in keep.iter().enumerate() {
            let Some(d) = self.dims.iter().position(|dim| dim == name) else {
                return Err(Error::InvalidParameter {
                    name: "keep",
                    value: i as f64,
                    reason: "names a dimension this distribution does not have",
                });
            };
            if kept.contains(&d) {
                return Err(Error::InvalidParameter {
                    name: "keep",
                    value: i as f64,
                    reason: "repeats an earlier dimension",
                });
            }
            kept.push(d);
        }

        let mut groups: Vec<ComponentKey> = Vec::new();
        let group_of: Vec<usize> = self
            .components
            .iter()
            .map(|key| {
                let projected: ComponentKey = kept.iter().map(|&d| key[d].clone()).collect();
                match groups.iter().position(|g| *g == projected) {
                    Some(g) => g,
                    None => {
                        groups.push(projected);
                        groups.len() - 1
                    }
                }
            })
            .collect();

        let n_groups = groups.len();
        let mut draws = vec![0.0; self.n_sims * n_groups];
        for (row, out) in self
            .draws
            .chunks_exact(self.n_components())
            .zip(draws.chunks_exact_mut(n_groups))
        {
            for (value, &g) in row.iter().zip(&group_of) {
                out[g] += value;
            }
        }
        Self::from_draws(
            keep.iter().map(|s| s.to_string()).collect(),
            groups,
            draws,
            self.provenance.clone(),
        )
    }

    /// `n` whole rows drawn with replacement from stream `rng`, so the
    /// components keep their dependence.
    pub fn resample(&self, rng: &mut StreamRng, n: usize) -> Result<Self> {
        let mut draws = Vec::with_capacity(n * self.n_components());
        for _ in 0..n {
            let sim = ((rng.next_open01() * self.n_sims as f64) as usize).min(self.n_sims - 1);
            draws.extend_from_slice(self.row(sim).expect("sim < n_sims"));
        }
        Self::from_draws(
            self.dims.clone(),
            self.components.clone(),
            draws,
            self.provenance.clone().param("resampled_rows", n),
        )
    }

    /// The total over all components, one value per simulation.
    pub fn total(&self) -> &Sampled {
        self.total.get_or_init(|| {
            let sums = self
                .draws
                .chunks_exact(self.n_components())
                .map(|row| row.iter().sum())
                .collect();
            Sampled::new(sums).expect("sums of finite draws are finite")
        })
    }

    /// Allocates the distortion risk measure of the [`total`](Self::total)
    /// to the components by co-measure (Euler allocation): one
    /// contribution per component, in [`components`](Self::components)
    /// order, summing to `d` applied to the total.
    ///
    /// Simulations are ranked by their total, take the distortion's
    /// weights by rank ([`Distortion::weights`]), and each component's
    /// contribution is the weighted sum of its own draws. For
    /// `Distortion::Tvar(p)` the contributions are the CoTVaRs,
    /// `E[X_j | total in its top 1 - p]`. Simulations with equal totals
    /// share their weights equally, so the result does not depend on how
    /// ties are ordered.
    ///
    /// Components must add up to the portfolio being allocated: allocate
    /// a set of segments, not a tower result that holds gross, ceded and
    /// net side by side.
    ///
    /// ```
    /// use prospicio_prob::{Distortion, PredictiveDistribution, Provenance, KeyValue};
    ///
    /// // Two lines over four simulations; the totals are 3, 5, 7, 9.
    /// let pd = PredictiveDistribution::from_draws(
    ///     vec!["lob".into()],
    ///     vec![vec![KeyValue::from("motor")], vec![KeyValue::from("property")]],
    ///     vec![1.0, 2.0, 4.0, 1.0, 2.0, 5.0, 3.0, 6.0],
    ///     Provenance::new("example"),
    /// )
    /// .unwrap();
    /// // TVaR at 50%: the two worst years, 7 = 2 + 5 and 9 = 3 + 6.
    /// let co = pd.allocate(&Distortion::tvar(0.5).unwrap());
    /// assert_eq!(co, [2.5, 5.5]);
    /// ```
    pub fn allocate(&self, d: &Distortion) -> Vec<f64> {
        let n = self.n_sims;
        let m = self.n_components();
        let totals = self.total().draws();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| totals[a].total_cmp(&totals[b]));
        let mut weights = d.weights(n);
        // Simulations tied on the total share their weights.
        let mut start = 0;
        while start < n {
            let mut end = start + 1;
            while end < n && totals[order[end]] == totals[order[start]] {
                end += 1;
            }
            if end - start > 1 {
                let mean = weights[start..end].iter().sum::<f64>() / (end - start) as f64;
                weights[start..end].fill(mean);
            }
            start = end;
        }
        let mut contributions = vec![0.0; m];
        for (w, &sim) in weights.iter().zip(&order) {
            if *w == 0.0 {
                continue;
            }
            let row = &self.draws[sim * m..(sim + 1) * m];
            for (c, x) in contributions.iter_mut().zip(row) {
                *c += w * x;
            }
        }
        contributions
    }
}

impl PredictiveDistribution {
    /// Marginal expected shortfall of each component at level `p`:
    /// `E[X_j | total ≥ VaR_p(total)]` (Acharya et al., 2017), the
    /// components' expected losses in the portfolio's worst `1 - p`. It is
    /// the CoTVaR, the Euler allocation of the total's TVaR
    /// ([`allocate`](Self::allocate) with [`Distortion::Tvar`]), and sums
    /// to it.
    pub fn marginal_expected_shortfall(&self, p: f64) -> Result<Vec<f64>> {
        Ok(self.allocate(&Distortion::tvar(p)?))
    }

    /// CoVaR of a component (Adrian and Brunnermeier, 2016, in the form of
    /// Girardi and Ergün, 2013): the total's VaR at level `q` over the
    /// simulations where the component is in distress, at or above its own
    /// VaR at level `p`. Compare it with the total's unconditional VaR at
    /// `q` to see how much one segment's bad years drag the portfolio.
    ///
    /// ```
    /// use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
    ///
    /// // Two components over four simulations.
    /// let pd = PredictiveDistribution::from_draws(
    ///     vec!["lob".into()],
    ///     vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
    ///     vec![1.0, 0.0, 2.0, 1.0, 3.0, 5.0, 4.0, 1.0],
    ///     Provenance::new("example"),
    /// )
    /// .unwrap();
    /// // a is at or above its 75% VaR (3) in the last two simulations,
    /// // whose totals are 8 and 5; their median (q = 0.5) is 5.
    /// assert_eq!(pd.covar(&vec![KeyValue::from("a")], 0.75, 0.5).unwrap(), 5.0);
    /// ```
    pub fn covar(&self, component: &ComponentKey, p: f64, q: f64) -> Result<f64> {
        check_probability(q)?;
        let j = self.component_index(component).ok_or_else(|| {
            Error::Data(format!("no component {component:?} in this distribution"))
        })?;
        let m = self.n_components();
        let own: Vec<f64> = self.draws.iter().skip(j).step_by(m).copied().collect();
        let mut sorted = own.clone();
        sorted.sort_by(f64::total_cmp);
        let threshold = var_sorted(&sorted, p)?;
        let totals = self.total().draws();
        let mut stressed: Vec<f64> = own
            .iter()
            .zip(totals)
            .filter(|(x, _)| **x >= threshold)
            .map(|(_, t)| *t)
            .collect();
        stressed.sort_by(f64::total_cmp);
        var_sorted(&stressed, q)
    }

    /// Esscher allocation: `E[X_j e^(h·total)] / E[e^(h·total)]` per
    /// component, the components' means under the Esscher transform of the
    /// total. They sum to the total's Esscher premium
    /// ([`risk::esscher`](crate::risk::esscher)), and with `h = 0` they are
    /// the means.
    pub fn esscher_allocation(&self, h: f64) -> Result<Vec<f64>> {
        if !h.is_finite() {
            return Err(Error::InvalidParameter {
                name: "h",
                value: h,
                reason: "must be finite",
            });
        }
        let m = self.n_components();
        let w = crate::risk::esscher_weights(self.total().draws(), h);
        let mut out = vec![0.0; m];
        for (wi, row) in w.iter().zip(self.draws.chunks_exact(m)) {
            for (o, x) in out.iter_mut().zip(row) {
                *o += wi * x;
            }
        }
        Ok(out)
    }
}

impl PredictiveDistribution {
    /// Blends several models' predictive distributions with `weights`
    /// (from stacking or pseudo-BMA, say): simulation `i` is simulation `i`
    /// of model `k`, with `k` drawn with probability `weights[k]` from
    /// stream `i` of `seed`. Each row stays a joint draw of one model, so
    /// sums across components remain coherent.
    ///
    /// Every distribution must have the same dimensions, components (in
    /// the same order) and number of simulations. Weights are normalized;
    /// they must be non-negative and not all zero.
    ///
    /// ```
    /// use prospicio_prob::{Empirical, KeyValue, PredictiveDistribution, Provenance};
    ///
    /// let one = |v: f64| {
    ///     PredictiveDistribution::from_draws(
    ///         vec!["lob".into()],
    ///         vec![vec![KeyValue::from("a")]],
    ///         vec![v; 1000],
    ///         Provenance::new("example"),
    ///     )
    ///     .unwrap()
    /// };
    /// let blend = PredictiveDistribution::blend(&[&one(0.0), &one(1.0)], &[0.25, 0.75], 7).unwrap();
    /// let share = blend.total().draws().iter().sum::<f64>() / 1000.0;
    /// assert!((share - 0.75).abs() < 0.05);
    /// ```
    pub fn blend(models: &[&PredictiveDistribution], weights: &[f64], seed: u64) -> Result<Self> {
        let first = *models
            .first()
            .ok_or_else(|| Error::Data("blend needs at least one model".into()))?;
        if weights.len() != models.len() {
            return Err(Error::Data(format!(
                "{} weights for {} models",
                weights.len(),
                models.len()
            )));
        }
        if weights.iter().any(|w| !(w.is_finite() && *w >= 0.0)) {
            return Err(Error::Data(
                "weights must be finite and non-negative".into(),
            ));
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(Error::Data("weights must not all be zero".into()));
        }
        for m in &models[1..] {
            if m.dims != first.dims || m.components != first.components || m.n_sims != first.n_sims
            {
                return Err(Error::Data(
                    "blended distributions need the same dimensions, components and simulations"
                        .into(),
                ));
            }
        }
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for w in weights {
            acc += w / total;
            cumulative.push(acc);
        }
        let m = first.n_components();
        let mut draws = Vec::with_capacity(first.draws.len());
        for i in 0..first.n_sims {
            let u = StreamRng::new(seed, i as u64).next_open01();
            let k = cumulative
                .iter()
                .position(|&c| u < c)
                .unwrap_or(models.len() - 1);
            draws.extend_from_slice(&models[k].draws[i * m..(i + 1) * m]);
        }
        let mut provenance = Provenance::new("blend").seed(seed, SIM_INDEX_SCHEME);
        for (model, w) in models.iter().zip(weights) {
            provenance = provenance.param(model.provenance.model.clone(), w / total);
        }
        Self::from_draws(
            first.dims.clone(),
            first.components.clone(),
            draws,
            provenance,
        )
    }
}

impl PredictiveDistribution {
    /// Blends models with weights that differ by component, as
    /// hierarchical stacking gives them (`weights[j]` is component `j`'s
    /// weight vector, one entry per model). In simulation `i` every
    /// component draws its model from the same uniform (stream `i` of
    /// `seed`) against its own cumulative weights, so components with the
    /// same weights take the same model and dependence across components
    /// is kept as far as the weights allow. With equal weights everywhere
    /// it is [`blend`](Self::blend).
    pub fn blend_by_component(
        models: &[&PredictiveDistribution],
        weights: &[Vec<f64>],
        seed: u64,
    ) -> Result<Self> {
        let first = *models
            .first()
            .ok_or_else(|| Error::Data("blend needs at least one model".into()))?;
        for m in &models[1..] {
            if m.dims != first.dims || m.components != first.components || m.n_sims != first.n_sims
            {
                return Err(Error::Data(
                    "blended distributions need the same dimensions, components and simulations"
                        .into(),
                ));
            }
        }
        let c = first.n_components();
        if weights.len() != c {
            return Err(Error::Data(format!(
                "{} weight vectors for {c} components",
                weights.len()
            )));
        }
        let mut cumulative = Vec::with_capacity(c);
        for w in weights {
            if w.len() != models.len() || w.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
                return Err(Error::Data(format!(
                    "each component needs {} finite non-negative weights",
                    models.len()
                )));
            }
            let total: f64 = w.iter().sum();
            if total <= 0.0 {
                return Err(Error::Data("weights must not all be zero".into()));
            }
            let mut acc = 0.0;
            cumulative.push(
                w.iter()
                    .map(|v| {
                        acc += v / total;
                        acc
                    })
                    .collect::<Vec<f64>>(),
            );
        }
        let mut draws = Vec::with_capacity(first.draws.len());
        for i in 0..first.n_sims {
            let u = StreamRng::new(seed, i as u64).next_open01();
            for (j, cum) in cumulative.iter().enumerate() {
                let k = cum.iter().position(|&v| u < v).unwrap_or(models.len() - 1);
                draws.push(models[k].draws[i * c + j]);
            }
        }
        let mut provenance = Provenance::new("blend_by_component").seed(seed, SIM_INDEX_SCHEME);
        for model in models {
            provenance = provenance.param("model", model.provenance.model.clone());
        }
        Self::from_draws(
            first.dims.clone(),
            first.components.clone(),
            draws,
            provenance,
        )
    }
}

impl Distribution for PredictiveDistribution {
    fn mean(&self) -> f64 {
        self.total().mean()
    }

    fn variance(&self) -> f64 {
        self.total().variance()
    }

    fn cdf(&self, x: f64) -> f64 {
        self.total().cdf(x)
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        self.total().quantile(p)
    }

    fn sample(&self, rng: &mut StreamRng, n: usize) -> Vec<f64> {
        self.total().sample(rng, n)
    }
}

impl Empirical for PredictiveDistribution {
    fn draws(&self) -> &[f64] {
        self.total().draws()
    }

    fn sorted(&self) -> &[f64] {
        self.total().sorted()
    }
}

/// Checks the dimension names and component keys shared by every
/// constructor.
fn validate_keys(dims: &[String], components: &[ComponentKey]) -> Result<()> {
    if components.is_empty() {
        return Err(Error::InvalidParameter {
            name: "components",
            value: 0.0,
            reason: "must not be empty",
        });
    }
    let mut seen_dims = HashSet::new();
    for (i, dim) in dims.iter().enumerate() {
        if !seen_dims.insert(dim) {
            return Err(Error::InvalidParameter {
                name: "dims",
                value: i as f64,
                reason: "repeats an earlier dimension",
            });
        }
    }
    let mut seen_keys = HashSet::new();
    for (i, key) in components.iter().enumerate() {
        if key.len() != dims.len() {
            return Err(Error::InvalidParameter {
                name: "components",
                value: i as f64,
                reason: "key does not have one value per dimension",
            });
        }
        if !seen_keys.insert(key) {
            return Err(Error::InvalidParameter {
                name: "components",
                value: i as f64,
                reason: "repeats an earlier key",
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn period_components_match_their_labels() {
        use prospicio_core::{Grain, Month};
        let q = |y, m| {
            KeyValue::Period(Period::containing(
                Month::new(y, m).unwrap(),
                Grain::Quarter,
            ))
        };
        let pd = PredictiveDistribution::from_draws(
            vec!["lob".into(), "origin".into()],
            vec![
                vec![KeyValue::from("auto"), q(2021, 1)],
                vec![KeyValue::from("auto"), q(2021, 7)],
                vec![KeyValue::from("home"), q(2021, 7)],
            ],
            vec![1.0, 2.0, 3.0],
            Provenance::new("test"),
        )
        .unwrap();
        let key = |lob: &str, origin: &str| vec![KeyValue::from(lob), KeyValue::from(origin)];
        assert_eq!(pd.component_index(&key("auto", "2021Q3")), Some(1));
        assert_eq!(pd.marginal(&key("home", "2021Q3")).unwrap().mean(), 3.0);
        assert_eq!(pd.component_index(&key("home", "2021Q1")), None);
        // A label must still name the component's own grain.
        assert_eq!(pd.component_index(&key("auto", "2021")), None);
        assert_eq!(pd.component_index(&vec![KeyValue::from("auto")]), None);

        // An exact key wins over a label, and a label that names two
        // components names neither.
        let year = KeyValue::Period(Period::containing(Month::january(2021), Grain::Year));
        let both = PredictiveDistribution::from_draws(
            vec!["origin".into()],
            vec![vec![KeyValue::from("2021")], vec![year.clone()]],
            vec![1.0, 2.0],
            Provenance::new("test"),
        )
        .unwrap();
        assert_eq!(both.component_index(&vec![KeyValue::from("2021")]), Some(0));
        assert_eq!(both.component_index(&vec![year]), Some(1));
        assert_eq!(both.component_index(&vec![KeyValue::from(2021)]), Some(1));
    }

    #[test]
    fn mes_and_esscher_allocations_add_up() {
        let mut rng = StreamRng::new(4, 0);
        let n = 5000;
        let mut draws = Vec::with_capacity(2 * n);
        for _ in 0..n {
            let z = prospicio_math::special::norm_quantile(rng.next_open01());
            draws.push(10.0 + 2.0 * z);
            draws.push(5.0 + z + prospicio_math::special::norm_quantile(rng.next_open01()));
        }
        let pd = PredictiveDistribution::from_draws(
            vec!["lob".into()],
            vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
            draws,
            Provenance::new("test"),
        )
        .unwrap();
        let mes = pd.marginal_expected_shortfall(0.9).unwrap();
        let tvar = Distortion::tvar(0.9).unwrap().apply_sorted(&{
            let mut t = pd.total().draws().to_vec();
            t.sort_by(f64::total_cmp);
            t
        });
        assert!((mes.iter().sum::<f64>() - tvar).abs() < 1e-9);
        let es = pd.esscher_allocation(0.2).unwrap();
        let total = crate::risk::esscher(pd.total().draws(), 0.2).unwrap();
        assert!((es.iter().sum::<f64>() - total).abs() < 1e-9);
        let means = pd.esscher_allocation(0.0).unwrap();
        assert!((means[0] - pd.marginal(&vec![KeyValue::from("a")]).unwrap().mean()).abs() < 1e-9);
        // Distress in a, which drives the total, raises the total's VaR.
        let key = vec![KeyValue::from("a")];
        assert!(
            pd.covar(&key, 0.95, 0.5).unwrap() > pd.total().draws().iter().sum::<f64>() / n as f64
        );
        assert!(pd.covar(&vec![KeyValue::from("z")], 0.9, 0.5).is_err());
    }
    use crate::Lognormal;

    fn key(values: &[KeyValue]) -> ComponentKey {
        values.to_vec()
    }

    /// Two lines × two origins, three simulations.
    fn lob_origin() -> PredictiveDistribution {
        let components = vec![
            key(&["Auto".into(), 2023.into()]),
            key(&["Auto".into(), 2024.into()]),
            key(&["Home".into(), 2023.into()]),
            key(&["Home".into(), 2024.into()]),
        ];
        #[rustfmt::skip]
        let draws = vec![
            1.0, 2.0, 10.0, 20.0,
            3.0, 4.0, 30.0, 40.0,
            5.0, 6.0, 50.0, 60.0,
        ];
        PredictiveDistribution::from_draws(
            vec!["lob".into(), "origin".into()],
            components,
            draws,
            Provenance::new("test"),
        )
        .unwrap()
    }

    #[test]
    fn shape_and_rows() {
        let pd = lob_origin();
        assert_eq!((pd.n_sims(), pd.n_components()), (3, 4));
        assert_eq!(pd.row(1), Some(&[3.0, 4.0, 30.0, 40.0][..]));
        assert_eq!(pd.row(3), None);
    }

    #[test]
    fn marginal_is_one_column_in_simulation_order() {
        let pd = lob_origin();
        let m = pd.marginal(&key(&["Home".into(), 2023.into()])).unwrap();
        assert_eq!(m.draws(), [10.0, 30.0, 50.0]);
        assert!(pd.marginal(&key(&["Home".into(), 2025.into()])).is_none());
    }

    #[test]
    fn aggregate_sums_within_each_simulation() {
        let pd = lob_origin();
        let by_lob = pd.aggregate(&["lob"]).unwrap();
        assert_eq!(by_lob.dims(), ["lob"]);
        assert_eq!(
            by_lob.components(),
            [key(&["Auto".into()]), key(&["Home".into()])]
        );
        assert_eq!(by_lob.draw_matrix(), [3.0, 30.0, 7.0, 70.0, 11.0, 110.0]);

        let by_origin = pd.aggregate(&["origin"]).unwrap();
        assert_eq!(
            by_origin.draw_matrix(),
            [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
        );

        let total = pd.aggregate(&[]).unwrap();
        assert_eq!(total.components(), [ComponentKey::new()]);
        assert_eq!(total.draw_matrix(), pd.total().draws());
        assert_eq!(pd.total().draws(), [33.0, 77.0, 121.0]);
    }

    #[test]
    fn aggregate_can_reorder_dimensions() {
        let pd = lob_origin();
        let swapped = pd.aggregate(&["origin", "lob"]).unwrap();
        assert_eq!(swapped.components()[0], key(&[2023.into(), "Auto".into()]));
        assert_eq!(swapped.draw_matrix(), pd.draw_matrix());
    }

    #[test]
    fn aggregate_rejects_unknown_and_repeated_dimensions() {
        let pd = lob_origin();
        assert!(pd.aggregate(&["state"]).is_err());
        assert!(pd.aggregate(&["lob", "lob"]).is_err());
    }

    #[test]
    fn blend_keeps_rows_whole_and_checks_shapes() {
        let pd = |v: f64| {
            PredictiveDistribution::from_draws(
                vec!["lob".into()],
                vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
                (0..400)
                    .flat_map(|i| [v + i as f64, -(v + i as f64)])
                    .collect(),
                Provenance::new(format!("m{v}")),
            )
            .unwrap()
        };
        let (a, b) = (pd(0.0), pd(0.5));
        let blend = PredictiveDistribution::blend(&[&a, &b], &[1.0, 3.0], 3).unwrap();
        // Each row comes whole from one model: its components still cancel.
        assert!(blend.total().draws().iter().all(|t| t.abs() < 1e-12));
        let from_b = (0..400)
            .filter(|&i| blend.draws[2 * i].fract() != 0.0)
            .count() as f64;
        assert!((from_b / 400.0 - 0.75).abs() < 0.08);
        assert_eq!(
            blend.provenance().parameters[1],
            ("m0.5".into(), "0.75".into())
        );
        assert!(PredictiveDistribution::blend(&[&a, &b], &[1.0], 3).is_err());
        assert!(PredictiveDistribution::blend(&[&a, &b], &[0.0, 0.0], 3).is_err());
        let short = PredictiveDistribution::from_draws(
            vec!["lob".into()],
            vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
            vec![0.0; 4],
            Provenance::new("short"),
        )
        .unwrap();
        assert!(PredictiveDistribution::blend(&[&a, &short], &[1.0, 1.0], 3).is_err());
    }

    #[test]
    fn blend_by_component_matches_blend_with_equal_weights() {
        let pd = |v: f64| {
            PredictiveDistribution::from_draws(
                vec!["lob".into()],
                vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
                (0..300)
                    .flat_map(|i| [v + i as f64, v - i as f64])
                    .collect(),
                Provenance::new(format!("m{v}")),
            )
            .unwrap()
        };
        let (a, b) = (pd(0.0), pd(0.5));
        let same = PredictiveDistribution::blend_by_component(
            &[&a, &b],
            &[vec![1.0, 3.0], vec![1.0, 3.0]],
            4,
        )
        .unwrap();
        let plain = PredictiveDistribution::blend(&[&a, &b], &[1.0, 3.0], 4).unwrap();
        assert_eq!(same.draws, plain.draws);
        // Component b all from model a, component a all from model b.
        let split = PredictiveDistribution::blend_by_component(
            &[&a, &b],
            &[vec![0.0, 1.0], vec![1.0, 0.0]],
            4,
        )
        .unwrap();
        assert!((0..300).all(|i| split.draws[2 * i] == b.draws[2 * i]
            && split.draws[2 * i + 1] == a.draws[2 * i + 1]));
        assert!(
            PredictiveDistribution::blend_by_component(&[&a, &b], &[vec![1.0, 1.0]], 4).is_err()
        );
    }

    #[test]
    fn total_quantile_is_not_the_sum_of_marginal_quantiles() {
        // A pays in simulation 3, B in simulation 2: never together.
        let pd = PredictiveDistribution::from_draws(
            vec!["line".into()],
            vec![key(&["A".into()]), key(&["B".into()])],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 0.0],
            Provenance::new("test"),
        )
        .unwrap();
        let a = pd.marginal(&key(&["A".into()])).unwrap();
        let b = pd.marginal(&key(&["B".into()])).unwrap();
        assert_eq!(a.var(0.75).unwrap() + b.var(0.75).unwrap(), 0.0);
        assert_eq!(pd.var(0.75), Ok(100.0));
        assert_eq!(pd.tvar(0.5), Ok(100.0));
        assert_eq!(pd.mean(), 50.0);
    }

    #[test]
    fn rejects_bad_input() {
        let dims = || vec!["origin".to_string()];
        let comps = || vec![key(&[2023.into()]), key(&[2024.into()])];
        let p = || Provenance::new("test");
        let ok = PredictiveDistribution::from_draws(dims(), comps(), vec![1.0, 2.0], p());
        assert!(ok.is_ok());
        // Not a whole number of rows, empty, non-finite.
        assert!(PredictiveDistribution::from_draws(dims(), comps(), vec![1.0; 3], p()).is_err());
        assert!(PredictiveDistribution::from_draws(dims(), comps(), vec![], p()).is_err());
        assert!(
            PredictiveDistribution::from_draws(dims(), comps(), vec![1.0, f64::NAN], p()).is_err()
        );
        // No components, wrong key length, repeated key, repeated dimension.
        assert!(PredictiveDistribution::from_draws(dims(), vec![], vec![1.0], p()).is_err());
        let short = vec![key(&[])];
        assert!(PredictiveDistribution::from_draws(dims(), short, vec![1.0], p()).is_err());
        let repeated = vec![key(&[2023.into()]), key(&[2023.into()])];
        assert!(PredictiveDistribution::from_draws(dims(), repeated, vec![1.0; 2], p()).is_err());
        let two_dims = vec!["origin".to_string(), "origin".to_string()];
        let two_keys = vec![key(&[2023.into(), 1.into()])];
        assert!(PredictiveDistribution::from_draws(two_dims, two_keys, vec![1.0], p()).is_err());
    }

    fn uniform_rows(threads: usize) -> PredictiveDistribution {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            PredictiveDistribution::simulate(
                vec!["origin".into()],
                (2015..2025).map(|y| key(&[y.into()])).collect(),
                2_000,
                7,
                Provenance::new("test"),
                |rng, row| {
                    for cell in row {
                        *cell = rng.next_open01();
                    }
                },
            )
            .unwrap()
        })
    }

    #[test]
    fn simulate_is_identical_across_thread_counts() {
        let one = uniform_rows(1);
        assert_eq!(one.draw_matrix(), uniform_rows(4).draw_matrix());
        assert_eq!(one.draw_matrix(), uniform_rows(16).draw_matrix());
        assert_eq!(one.provenance().seed, Some(7));
        assert_eq!(
            one.provenance().stream_scheme.as_deref(),
            Some(SIM_INDEX_SCHEME)
        );
    }

    #[test]
    fn simulate_row_i_replays_from_stream_i() {
        let pd = uniform_rows(4);
        let mut rng = StreamRng::new(7, 1234);
        let replay: Vec<f64> = (0..10).map(|_| rng.next_open01()).collect();
        assert_eq!(pd.row(1234).unwrap(), replay);
    }

    #[test]
    fn simulate_rejects_zero_sims() {
        let r = PredictiveDistribution::simulate(
            vec![],
            vec![ComponentKey::new()],
            0,
            1,
            Provenance::new("test"),
            |_, _| {},
        );
        assert!(r.is_err());
    }

    #[test]
    fn resample_keeps_rows_whole() {
        let pd = lob_origin();
        let r = pd.resample(&mut StreamRng::new(3, 0), 50).unwrap();
        assert_eq!(r.n_sims(), 50);
        for sim in 0..r.n_sims() {
            let row = r.row(sim).unwrap();
            assert!((0..pd.n_sims()).any(|s| pd.row(s).unwrap() == row));
        }
        let again = pd.resample(&mut StreamRng::new(3, 0), 50).unwrap();
        assert_eq!(r.draw_matrix(), again.draw_matrix());
    }

    #[test]
    fn total_measures_come_from_row_sums() {
        let sev = Lognormal::new(0.0, 1.0).unwrap();
        let pd = PredictiveDistribution::simulate(
            vec!["origin".into()],
            (0..5).map(|y| key(&[y.into()])).collect(),
            5_000,
            11,
            Provenance::new("test"),
            |rng, row| {
                for cell in row {
                    *cell = sev.quantile(rng.next_open01()).unwrap();
                }
            },
        )
        .unwrap();
        let sums = Sampled::new(
            (0..pd.n_sims())
                .map(|s| pd.row(s).unwrap().iter().sum())
                .collect(),
        )
        .unwrap();
        assert_eq!(pd.mean(), sums.mean());
        assert_eq!(pd.quantile(0.99), sums.quantile(0.99));
        assert_eq!(pd.tvar(0.99), sums.tvar(0.99));
        assert_eq!(Empirical::draws(&pd), sums.draws());
    }

    #[test]
    fn key_values_display_and_order() {
        assert_eq!(KeyValue::from(2024).to_string(), "2024");
        assert_eq!(KeyValue::from("Auto").to_string(), "Auto");
        assert!(KeyValue::from(1) < KeyValue::from(2));
    }

    #[test]
    fn period_keys_display_order_and_join() {
        use prospicio_core::{Grain, Month};

        let q = |y, m| Period::containing(Month::new(y, m).unwrap(), Grain::Quarter);
        assert_eq!(KeyValue::from(q(2021, 8)).to_string(), "2021Q3");
        assert_eq!(KeyValue::from(Period::year(2019)).to_string(), "2019");
        assert!(KeyValue::from(q(2021, 3)) < KeyValue::from(q(2021, 4)));
        // Any month in the period gives the same key.
        assert_eq!(KeyValue::from(q(2021, 7)), KeyValue::from(q(2021, 9)));
        // A period key is not the integer year.
        assert_ne!(KeyValue::from(Period::year(2019)), KeyValue::from(2019));

        let origins = [Period::year(2019), Period::year(2020)];
        let pd = PredictiveDistribution::from_draws(
            vec!["lob".into(), "origin".into()],
            vec![
                vec!["Auto".into(), origins[0].into()],
                vec!["Auto".into(), origins[1].into()],
                vec!["Home".into(), origins[1].into()],
            ],
            vec![1.0, 2.0, 4.0, 10.0, 20.0, 40.0],
            Provenance::new("test"),
        )
        .unwrap();
        let by_origin = pd.aggregate(&["origin"]).unwrap();
        assert_eq!(
            by_origin.components(),
            &[vec![origins[0].into()], vec![origins[1].into()]]
        );
        let o2020 = by_origin.marginal(&vec![origins[1].into()]).unwrap();
        assert_eq!(o2020.draws(), &[6.0, 60.0]);
    }

    fn lines(draws: Vec<f64>, m: usize) -> PredictiveDistribution {
        let components = (0..m).map(|j| vec![KeyValue::Int(j as i64)]).collect();
        PredictiveDistribution::from_draws(
            vec!["lob".into()],
            components,
            draws,
            Provenance::new("test"),
        )
        .unwrap()
    }

    fn simulated_lines() -> PredictiveDistribution {
        use crate::Lognormal;
        let a = Lognormal::from_mean_cv(100.0, 0.3).unwrap();
        let b = Lognormal::from_mean_cv(50.0, 1.2).unwrap();
        PredictiveDistribution::simulate(
            vec!["lob".into()],
            vec![vec![KeyValue::from("a")], vec![KeyValue::from("b")]],
            20_000,
            5,
            Provenance::new("test"),
            |rng, row| {
                let x = a.sample(rng, 1)[0];
                let y = b.sample(rng, 1)[0];
                // Some dependence: the second line moves with the first.
                row.copy_from_slice(&[x, y + 0.5 * x]);
            },
        )
        .unwrap()
    }

    #[test]
    fn allocation_adds_up_to_the_measure_of_the_total() {
        let pd = simulated_lines();
        for d in [
            Distortion::tvar(0.99).unwrap(),
            Distortion::wang(0.5).unwrap(),
            Distortion::proportional_hazard(0.7).unwrap(),
            Distortion::dual_power(3.0).unwrap(),
        ] {
            let co = pd.allocate(&d);
            let whole = pd.distortion(&d);
            assert!(
                (co.iter().sum::<f64>() - whole).abs() <= 1e-9 * whole,
                "{d:?}"
            );
        }
        // The mean allocates to the component means.
        let co = pd.allocate(&Distortion::tvar(0.0).unwrap());
        for (j, c) in co.iter().enumerate() {
            let mean =
                (0..pd.n_sims()).map(|i| pd.row(i).unwrap()[j]).sum::<f64>() / pd.n_sims() as f64;
            assert!((c - mean).abs() <= 1e-9 * mean);
        }
    }

    #[test]
    fn cotvar_is_the_conditional_tail_mean() {
        let pd = simulated_lines();
        let p = 0.95;
        let co = pd.allocate(&Distortion::tvar(p).unwrap());
        // 20,000 × 0.05 = 1,000 whole simulations: the plain conditional mean.
        let mut rows: Vec<&[f64]> = (0..pd.n_sims()).map(|i| pd.row(i).unwrap()).collect();
        rows.sort_by(|a, b| a.iter().sum::<f64>().total_cmp(&b.iter().sum::<f64>()));
        let tail = &rows[19_000..];
        for (j, c) in co.iter().enumerate() {
            let want = tail.iter().map(|r| r[j]).sum::<f64>() / 1_000.0;
            assert!((c - want).abs() <= 1e-9 * want, "{c} vs {want}");
        }
    }

    #[test]
    fn comonotonic_parts_get_their_own_measure() {
        // The second line is twice the first: ρ is comonotonic additive, so
        // each line is allocated its own risk measure.
        let x = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0];
        let draws: Vec<f64> = x.iter().flat_map(|&v| [v, 2.0 * v]).collect();
        let pd = lines(draws, 2);
        let mut sorted = x.to_vec();
        sorted.sort_by(f64::total_cmp);
        let d = Distortion::wang(0.8).unwrap();
        let co = pd.allocate(&d);
        let own = d.apply_sorted(&sorted);
        assert!((co[0] - own).abs() < 1e-12);
        assert!((co[1] - 2.0 * own).abs() < 1e-12);
    }

    #[test]
    fn tied_totals_share_weights() {
        // Simulations 1 and 2 tie on the total (5) but split it differently;
        // TVaR at 2/3 takes the top third of the mass, which is all on that
        // tie, so each line gets the mean of its two values.
        let a = lines(vec![1.0, 1.0, 4.0, 1.0, 0.0, 5.0], 2);
        let b = lines(vec![1.0, 1.0, 0.0, 5.0, 4.0, 1.0], 2);
        let d = Distortion::tvar(2.0 / 3.0).unwrap();
        for pd in [a, b] {
            let co = pd.allocate(&d);
            assert!(
                (co[0] - 2.0).abs() < 1e-12 && (co[1] - 3.0).abs() < 1e-12,
                "{co:?}"
            );
        }
    }
}
