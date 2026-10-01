//! The joint predictive distribution every model returns.

use std::collections::HashSet;
use std::fmt;
use std::sync::OnceLock;

use act_core::{Error, Result, StreamRng};
use rayon::prelude::*;

use crate::distribution::Distribution;
use crate::provenance::{Provenance, SIM_INDEX_SCHEME};
use crate::sampled::{Empirical, Sampled};

/// One value of a component key: an integer (an origin year, a layer
/// number) or text (a line of business).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KeyValue {
    Int(i64),
    Text(String),
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

impl fmt::Display for KeyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Text(v) => f.write_str(v),
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
/// use act_prob::{Distribution, Empirical, Lognormal, PredictiveDistribution, Provenance};
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
        validate_keys(&dims, &components)?;
        if n_sims == 0 {
            return Err(Error::InvalidParameter {
                name: "n_sims",
                value: 0.0,
                reason: "must be positive",
            });
        }
        let mut draws = vec![0.0; n_sims * components.len()];
        draws
            .par_chunks_mut(components.len())
            .enumerate()
            .for_each(|(i, row)| simulate(&mut StreamRng::new(seed, i as u64), row));
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
    pub fn component_index(&self, key: &ComponentKey) -> Option<usize> {
        self.components.iter().position(|k| k == key)
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
}
