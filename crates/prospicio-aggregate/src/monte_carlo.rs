//! Monte Carlo frequency-severity: simulated years of individual losses.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_prob::provenance::{SIM_INDEX_SCHEME, current_samplers};
use prospicio_prob::{Counting, Distribution, PredictiveDistribution, Provenance};
use rayon::prelude::*;

/// Simulated years of individual losses (events), kept so that per-event
/// terms such as excess-of-loss layers can be applied to them.
///
/// Year `i` was drawn from `StreamRng::new(seed, i)` in a fixed order:
/// first the claim count, then each claim's severity, both by inverse
/// transform. So results are identical for any number of threads, and any
/// year can be replayed alone.
///
/// Each loss may carry the sum insured of the risk it hit
/// ([`EventSet::with_sums_insured`], or a risk profile), which a surplus
/// treaty needs, and the time in the year it happened
/// ([`EventSet::with_times`], [`EventSet::with_uniform_times`],
/// [`EventSet::with_seasonal_times`]), which reinstatement premiums pro
/// rata as to time need. Losses may carry labels too, a peril or a class
/// ([`EventSet::with_labels`]), by which [`crate::Tower::apply`] splits
/// its result.
#[derive(Debug, Clone, PartialEq)]
pub struct EventSet {
    /// `offsets[i]..offsets[i + 1]` indexes year `i`'s losses.
    offsets: Vec<usize>,
    losses: Vec<f64>,
    /// One per loss when known.
    sums_insured: Option<Vec<f64>>,
    /// One per loss when known: the fraction of the year elapsed, in
    /// `[0, 1]`, non-decreasing within each year.
    times: Option<Vec<f64>>,
    /// Each loss's labels, when given.
    labels: Option<Labels>,
    seed: u64,
    /// The samplers the losses were drawn with, for results' provenance:
    /// this build's when it drew them, `None` for years from elsewhere.
    samplers: Option<Vec<(String, String)>>,
    /// A hash of the years' source, a year loss table's file say, kept as
    /// results' input hash.
    input_hash: Option<String>,
}

/// Each loss's value in every label dimension, stored as an index into the
/// distinct combinations of values.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Labels {
    pub(crate) dims: Vec<String>,
    /// The distinct combinations, one value per dimension, sorted.
    pub(crate) combinations: Vec<Vec<String>>,
    /// Each loss's combination.
    pub(crate) of_loss: Vec<usize>,
}

impl Labels {
    /// From each loss's values, one per dimension.
    fn new(dims: Vec<String>, per_loss: Vec<Vec<String>>) -> Self {
        let mut combinations = per_loss.clone();
        combinations.sort();
        combinations.dedup();
        let of_loss = per_loss
            .iter()
            .map(|c| {
                combinations
                    .binary_search(c)
                    .expect("every loss's combination is listed")
            })
            .collect();
        Self {
            dims,
            combinations,
            of_loss,
        }
    }

    /// Loss `k`'s values, one per dimension.
    fn values(&self, k: usize) -> &[String] {
        &self.combinations[self.of_loss[k]]
    }
}

/// The hash of two merged sets' sources: either's alone, or one hash of
/// both in order.
fn merged_hash(a: &Option<String>, b: &Option<String>) -> Option<String> {
    match (a, b) {
        (None, None) => None,
        (Some(h), None) | (None, Some(h)) => Some(h.clone()),
        (Some(a), Some(b)) => Some(
            prospicio_prob::provenance::InputHasher::new()
                .str("merge")
                .str(a)
                .str(b)
                .finish(),
        ),
    }
}

/// Year `i`'s event times come from stream `TIME_STREAM + i`, apart from
/// the streams the years' losses were drawn from.
const TIME_STREAM: u64 = 1 << 63;

/// The rows of a long table in the order `from_long` lists them: by year,
/// in their own order within each year. Checks the lengths and years.
fn long_order(years: &[usize], n_losses: usize, n_years: usize) -> Result<Vec<usize>> {
    if years.len() != n_losses {
        return Err(Error::Data(format!(
            "{} years for {n_losses} losses",
            years.len()
        )));
    }
    if let Some(&y) = years.iter().find(|&&y| y >= n_years) {
        return Err(Error::Data(format!(
            "year {y} is not below n_years = {n_years}"
        )));
    }
    let mut order: Vec<usize> = (0..n_losses).collect();
    order.sort_by_key(|&k| years[k]);
    Ok(order)
}

/// Simulates `n_sims` years of claims: a count from `frequency`, then that
/// many independent losses from `severity`.
///
/// # Example
///
/// ```
/// use prospicio_aggregate::simulate_events;
/// use prospicio_prob::{Distribution, Lognormal, Poisson};
///
/// let freq = Poisson::new(5.0).unwrap();
/// let sev = Lognormal::from_mean_cv(1_000.0, 1.0).unwrap();
/// let events = simulate_events(&freq, &sev, 20_000, 42).unwrap();
/// let totals = events.totals().unwrap();
/// // E[S] = 5 × 1,000; the standard error is about 15.
/// assert!((totals.mean() - 5_000.0).abs() < 75.0);
/// ```
pub fn simulate_events<N, X>(
    frequency: &N,
    severity: &X,
    n_sims: usize,
    seed: u64,
) -> Result<EventSet>
where
    N: Counting + Sync + ?Sized,
    X: Distribution + Sync + ?Sized,
{
    if n_sims == 0 {
        return Err(Error::InvalidParameter {
            name: "n_sims",
            value: 0.0,
            reason: "must be positive",
        });
    }
    let year = |i: u64| -> Vec<f64> {
        let mut rng = StreamRng::new(seed, i);
        let count = frequency
            .quantile(rng.next_open01())
            .expect("next_open01 is always in (0, 1)");
        (0..count)
            .map(|_| {
                severity
                    .quantile(rng.next_open01())
                    .expect("next_open01 is always in (0, 1)")
            })
            .collect()
    };
    // A severity whose callbacks must stay on this thread (an R function)
    // runs the years in order; the draws are the same either way.
    let years: Vec<Vec<f64>> = if severity.is_parallel_safe() {
        (0..n_sims as u64).into_par_iter().map(year).collect()
    } else {
        (0..n_sims as u64).map(year).collect()
    };
    let mut offsets = Vec::with_capacity(n_sims + 1);
    offsets.push(0);
    let mut losses = Vec::with_capacity(years.iter().map(Vec::len).sum());
    for year in years {
        losses.extend(year);
        offsets.push(losses.len());
    }
    Ok(EventSet {
        offsets,
        losses,
        sums_insured: None,
        times: None,
        labels: None,
        seed,
        samplers: Some(current_samplers()),
        input_hash: None,
    })
}

impl EventSet {
    /// Years of losses from elsewhere (your own simulation, or a
    /// catastrophe model's event loss table by year), in order within each
    /// year. `seed` is recorded in results' provenance, and keys the draws
    /// of [`EventSet::with_uniform_times`]; give the one your simulation
    /// used, or 0. The samplers are not recorded (`None` in results'
    /// provenance), since the losses were not drawn here; a simulation in
    /// this build that builds its years by hand records them with
    /// [`EventSet::with_samplers`].
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// let events = EventSet::from_years(vec![vec![5.0, 2.0], vec![], vec![9.0]], 0).unwrap();
    /// assert_eq!(events.counts(), [2, 0, 1]);
    /// assert_eq!(events.events(2), [9.0]);
    /// ```
    pub fn from_years(years: Vec<Vec<f64>>, seed: u64) -> Result<Self> {
        if years.is_empty() {
            return Err(Error::Data("needs at least one year".into()));
        }
        let mut offsets = Vec::with_capacity(years.len() + 1);
        offsets.push(0);
        let mut losses = Vec::with_capacity(years.iter().map(Vec::len).sum());
        for year in years {
            losses.extend(year);
            offsets.push(losses.len());
        }
        if let Some(&bad) = losses.iter().find(|x| !(x.is_finite() && **x >= 0.0)) {
            return Err(Error::InvalidParameter {
                name: "losses",
                value: bad,
                reason: "must be finite and non-negative",
            });
        }
        Ok(Self {
            offsets,
            losses,
            sums_insured: None,
            times: None,
            labels: None,
            seed,
            samplers: None,
            input_hash: None,
        })
    }

    /// Years of losses in long form, as a year loss table lists them: one
    /// row per loss with its year, an index in `0..n_years`. Within a year
    /// the losses keep their order; years with no row are empty.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// let events = EventSet::from_long(&[2, 0, 2], &[5.0, 1.0, 7.0], 4, 0).unwrap();
    /// assert_eq!(events.events(0), &[1.0]);
    /// assert!(events.events(1).is_empty());
    /// assert_eq!(events.events(2), &[5.0, 7.0]);
    /// ```
    pub fn from_long(years: &[usize], losses: &[f64], n_years: usize, seed: u64) -> Result<Self> {
        let order = long_order(years, losses.len(), n_years)?;
        let mut grouped = vec![Vec::new(); n_years];
        for &k in &order {
            grouped[years[k]].push(losses[k]);
        }
        Self::from_years(grouped, seed)
    }

    /// [`EventSet::from_long`] with each row's labels: `labels` holds one
    /// `(dimension, values)` pair per label dimension, a value per row, as
    /// [`EventSet::with_labels`] takes them.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// let peril = vec!["wind".to_string(), "quake".into(), "wind".into()];
    /// let events =
    ///     EventSet::from_long_labeled(&[2, 0, 2], &[5.0, 1.0, 7.0], vec![("peril".into(), peril)], 3, 0)
    ///         .unwrap();
    /// assert_eq!(events.events(2), &[5.0, 7.0]);
    /// assert_eq!(events.labels(0).unwrap(), [["quake".to_string()]]);
    /// ```
    pub fn from_long_labeled(
        years: &[usize],
        losses: &[f64],
        labels: Vec<(String, Vec<String>)>,
        n_years: usize,
        seed: u64,
    ) -> Result<Self> {
        let order = long_order(years, losses.len(), n_years)?;
        for (dim, values) in &labels {
            if values.len() != losses.len() {
                return Err(Error::Data(format!(
                    "{} values of label {dim:?} for {} losses",
                    values.len(),
                    losses.len()
                )));
            }
        }
        let events = Self::from_long(years, losses, n_years, seed)?;
        let labels = labels
            .into_iter()
            .map(|(dim, values)| (dim, order.iter().map(|&k| values[k].clone()).collect()))
            .collect();
        events.with_labels(labels)
    }

    /// The same events with labels, a peril or a class say: one
    /// `(dimension, values)` pair per label dimension, a value per loss,
    /// years in order as [`EventSet::events`] lists them. Replaces any
    /// labels the set had. [`crate::Tower::apply`] then gives each of its
    /// gross, ceded and net components by label; `merge` keeps them.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// let events = EventSet::from_years(vec![vec![5.0, 2.0], vec![9.0]], 0)
    ///     .unwrap()
    ///     .with_labels(vec![("peril".into(), vec!["wind".into(), "flood".into(), "wind".into()])])
    ///     .unwrap();
    /// assert_eq!(events.label_dims(), ["peril"]);
    /// assert_eq!(events.label_combinations(), [["flood"], ["wind"]]);
    /// ```
    pub fn with_labels(mut self, labels: Vec<(String, Vec<String>)>) -> Result<Self> {
        if labels.is_empty() {
            return Err(Error::Data("give at least one label dimension".into()));
        }
        let n = self.losses.len();
        for (i, (dim, values)) in labels.iter().enumerate() {
            if dim.is_empty() || dim == "kind" || dim == "layer" {
                return Err(Error::Data(format!(
                    "label dimension {dim:?}: names must be non-empty and not \"kind\" or \"layer\""
                )));
            }
            if labels[..i].iter().any(|(d, _)| d == dim) {
                return Err(Error::Data(format!("label dimension {dim:?} is repeated")));
            }
            if values.len() != n {
                return Err(Error::Data(format!(
                    "{} values of label {dim:?} for {n} losses",
                    values.len()
                )));
            }
        }
        let per_loss = (0..n)
            .map(|k| labels.iter().map(|(_, v)| v[k].clone()).collect())
            .collect();
        let dims = labels.into_iter().map(|(d, _)| d).collect();
        self.labels = Some(Labels::new(dims, per_loss));
        Ok(self)
    }

    /// Whether the losses carry labels.
    pub fn has_labels(&self) -> bool {
        self.labels.is_some()
    }

    /// The label dimensions, empty without labels.
    pub fn label_dims(&self) -> &[String] {
        self.labels.as_ref().map_or(&[], |l| &l.dims)
    }

    /// The distinct combinations of label values among the losses, one
    /// value per dimension, sorted: the labels of [`crate::Tower::apply`]'s
    /// components. Empty without labels.
    pub fn label_combinations(&self) -> &[Vec<String>] {
        self.labels.as_ref().map_or(&[], |l| &l.combinations)
    }

    /// Year `sim`'s labels, when known: one value per dimension for each
    /// loss.
    pub fn labels(&self, sim: usize) -> Option<Vec<&[String]>> {
        self.labels.as_ref().map(|l| {
            (self.offsets[sim]..self.offsets[sim + 1])
                .map(|k| l.values(k))
                .collect()
        })
    }

    /// The labels, for the tower.
    pub(crate) fn label_set(&self) -> Option<&Labels> {
        self.labels.as_ref()
    }

    /// Year `sim`'s losses' label combinations, as indices into
    /// [`EventSet::label_combinations`].
    pub(crate) fn label_indices(&self, sim: usize) -> Option<&[usize]> {
        self.labels
            .as_ref()
            .map(|l| &l.of_loss[self.offsets[sim]..self.offsets[sim + 1]])
    }

    /// The same events, recorded as drawn with `samplers`
    /// ([`Provenance::samplers`]): [`current_samplers`] for years this
    /// build simulated, as a risk profile's simulation does, or `None` when
    /// unknown.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    /// use prospicio_prob::provenance::current_samplers;
    ///
    /// let events = EventSet::from_years(vec![vec![5.0]], 4).unwrap();
    /// assert_eq!(events.totals().unwrap().provenance().samplers, None);
    /// let drawn = events.with_samplers(Some(current_samplers()));
    /// assert_eq!(drawn.totals().unwrap().provenance().samplers, Some(current_samplers()));
    /// ```
    pub fn with_samplers(mut self, samplers: Option<Vec<(String, String)>>) -> Self {
        self.samplers = samplers;
        self
    }

    /// The samplers the losses were drawn with, if recorded.
    pub fn samplers(&self) -> Option<&[(String, String)]> {
        self.samplers.as_deref()
    }

    /// The same events with a hash of their source, which results record
    /// as their provenance's `input_hash`: a year loss table's file hashed
    /// by [`crate::ylt::hash_bytes`], say. `merge` combines the two sets'
    /// hashes.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// let events = EventSet::from_years(vec![vec![5.0]], 0)
    ///     .unwrap()
    ///     .with_input_hash(Some("blake3:ab".into()));
    /// assert_eq!(events.totals().unwrap().provenance().input_hash.as_deref(), Some("blake3:ab"));
    /// ```
    pub fn with_input_hash(mut self, hash: Option<String>) -> Self {
        self.input_hash = hash;
        self
    }

    /// The hash of the years' source, if recorded.
    pub fn input_hash(&self) -> Option<&str> {
        self.input_hash.as_deref()
    }

    /// Provenance for `model` on these years: the set's seed and stream
    /// scheme, and its samplers.
    pub(crate) fn provenance(&self, model: &str) -> Provenance {
        let mut p = Provenance::new(model)
            .version("prospicio-aggregate", env!("CARGO_PKG_VERSION"))
            .seed(self.seed, SIM_INDEX_SCHEME);
        p.samplers = self.samplers.clone();
        p.input_hash = self.input_hash.clone();
        p
    }

    /// The same events, each on a risk with the given sum insured: one
    /// value per loss, years in order, as [`EventSet::events`] lists them.
    /// A loss may not exceed its risk's sum insured.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// let events = EventSet::from_years(vec![vec![5.0, 2.0], vec![9.0]], 0)
    ///     .unwrap()
    ///     .with_sums_insured(vec![10.0, 2.0, 50.0])
    ///     .unwrap();
    /// assert_eq!(events.sums_insured(1), Some(&[50.0][..]));
    /// ```
    pub fn with_sums_insured(mut self, sums_insured: Vec<f64>) -> Result<Self> {
        if sums_insured.len() != self.losses.len() {
            return Err(Error::Data(format!(
                "{} sums insured for {} losses",
                sums_insured.len(),
                self.losses.len()
            )));
        }
        for (&si, &x) in sums_insured.iter().zip(&self.losses) {
            if !(si.is_finite() && si > 0.0) {
                return Err(Error::InvalidParameter {
                    name: "sums_insured",
                    value: si,
                    reason: "must be positive and finite",
                });
            }
            if x > si * (1.0 + 1e-12) {
                return Err(Error::InvalidParameter {
                    name: "sums_insured",
                    value: si,
                    reason: "is below its loss",
                });
            }
        }
        self.sums_insured = Some(sums_insured);
        Ok(self)
    }

    /// The same events at the given times: one value per loss, years in
    /// order, each the fraction of the year elapsed when the loss happened
    /// (in `[0, 1]`, non-decreasing within a year, since the losses are
    /// taken as chronological).
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// let events = EventSet::from_years(vec![vec![5.0, 2.0], vec![9.0]], 0)
    ///     .unwrap()
    ///     .with_times(vec![0.1, 0.6, 0.25])
    ///     .unwrap();
    /// assert_eq!(events.times(0), Some(&[0.1, 0.6][..]));
    /// ```
    pub fn with_times(mut self, times: Vec<f64>) -> Result<Self> {
        if times.len() != self.losses.len() {
            return Err(Error::Data(format!(
                "{} times for {} losses",
                times.len(),
                self.losses.len()
            )));
        }
        if let Some(&t) = times.iter().find(|t| !(0.0..=1.0).contains(*t)) {
            return Err(Error::InvalidParameter {
                name: "times",
                value: t,
                reason: "must be in [0, 1]",
            });
        }
        for w in self.offsets.windows(2) {
            if let Some(pair) = times[w[0]..w[1]].windows(2).find(|p| p[1] < p[0]) {
                return Err(Error::InvalidParameter {
                    name: "times",
                    value: pair[1],
                    reason: "must not decrease within a year",
                });
            }
        }
        self.times = Some(times);
        Ok(self)
    }

    /// The same events at times spread uniformly over the year: year `i`'s
    /// `n` losses take `n` sorted uniform draws from stream `2^63 + i` of
    /// the generator keyed by the set's seed, in the losses' order. The
    /// losses of a year are independent and identically distributed, so
    /// giving them sorted times in their drawn order is the same as dating
    /// each at random and sorting.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// let events = EventSet::from_years(vec![vec![5.0, 2.0, 7.0]], 3)
    ///     .unwrap()
    ///     .with_uniform_times();
    /// let t = events.times(0).unwrap();
    /// assert!(t[0] <= t[1] && t[1] <= t[2]);
    /// ```
    pub fn with_uniform_times(self) -> Self {
        self.with_drawn_times(|u| u)
    }

    /// The same events at times drawn from a seasonal density: the year is
    /// cut into `weights.len()` equal periods (12 for months, 52 for weeks),
    /// starting at the contract's inception, and a loss falls in period `k`
    /// with probability `weights[k] / Σ weights`, uniformly within it. A
    /// zero weight means no losses in that period (a hurricane season).
    ///
    /// Year `i` takes the same sorted uniform draws as
    /// [`EventSet::with_uniform_times`] and maps each through the season's
    /// quantile, which is increasing, so the times stay sorted and equal
    /// weights give the uniform times (to rounding).
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// // Losses only in the second half of the year.
    /// let events = EventSet::from_years(vec![vec![5.0, 2.0, 7.0]], 3)
    ///     .unwrap()
    ///     .with_seasonal_times(&[0.0, 1.0])
    ///     .unwrap();
    /// let t = events.times(0).unwrap();
    /// assert!(t.iter().all(|&x| x >= 0.5) && t.windows(2).all(|w| w[0] <= w[1]));
    /// ```
    pub fn with_seasonal_times(self, weights: &[f64]) -> Result<Self> {
        if weights.is_empty() {
            return Err(Error::Data("needs at least one period's weight".into()));
        }
        if let Some(&w) = weights.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
            return Err(Error::InvalidParameter {
                name: "weights",
                value: w,
                reason: "must be finite and non-negative",
            });
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "weights",
                value: total,
                reason: "must not all be zero",
            });
        }
        // cum[k] is the probability of the first k periods.
        let m = weights.len();
        let mut cum = Vec::with_capacity(m + 1);
        cum.push(0.0);
        let mut acc = 0.0;
        for w in weights {
            acc += w / total;
            cum.push(acc);
        }
        cum[m] = 1.0;
        Ok(self.with_drawn_times(|u| {
            // The period whose probability interval holds u; it has a
            // positive weight, since cum[k] <= u < cum[k + 1].
            let k = (cum[1..].partition_point(|&c| c <= u)).min(m - 1);
            let within = (u - cum[k]) / (cum[k + 1] - cum[k]);
            ((k as f64 + within.clamp(0.0, 1.0)) / m as f64).clamp(0.0, 1.0)
        }))
    }

    /// Year `i`'s `n` sorted uniform draws from stream `2^63 + i`, each
    /// mapped through `quantile` (increasing on `(0, 1)`).
    fn with_drawn_times(mut self, quantile: impl Fn(f64) -> f64) -> Self {
        let mut times = Vec::with_capacity(self.losses.len());
        for (i, w) in self.offsets.windows(2).enumerate() {
            let mut rng = StreamRng::new(self.seed, TIME_STREAM + i as u64);
            let start = times.len();
            times.extend((w[0]..w[1]).map(|_| rng.next_open01()));
            times[start..].sort_by(f64::total_cmp);
            for t in &mut times[start..] {
                *t = quantile(*t);
            }
        }
        self.times = Some(times);
        self
    }

    /// Both sets' losses in each year: a catastrophe event set with a
    /// non-catastrophe one, say. The sets need the same number of years.
    /// With times on both, each year's losses are merged in time order
    /// (this set's first on a tie); with times on neither, this set's
    /// losses come first. Sums insured are kept when both sets carry them.
    /// Labels are kept when either set carries them: the dimensions are
    /// this set's, then the other's not among them, and a loss without a
    /// value in a dimension gets `""` (so a labelled catastrophe set merged
    /// with an unlabelled one leaves the other's losses labelled `""`).
    /// The seed is this set's; the samplers are kept only when the two
    /// agree.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// let cat = EventSet::from_years(vec![vec![100.0], vec![]], 1)
    ///     .unwrap()
    ///     .with_times(vec![0.5])
    ///     .unwrap();
    /// let other = EventSet::from_years(vec![vec![3.0, 4.0], vec![5.0]], 2)
    ///     .unwrap()
    ///     .with_times(vec![0.2, 0.7, 0.1])
    ///     .unwrap();
    /// let all = cat.merge(&other).unwrap();
    /// assert_eq!(all.events(0), &[3.0, 100.0, 4.0]);
    /// assert_eq!(all.times(0), Some(&[0.2, 0.5, 0.7][..]));
    /// assert_eq!(all.events(1), &[5.0]);
    /// ```
    pub fn merge(&self, other: &EventSet) -> Result<Self> {
        if self.n_sims() != other.n_sims() {
            return Err(Error::Data(format!(
                "{} years and {} years: merged sets need the same number",
                self.n_sims(),
                other.n_sims()
            )));
        }
        if self.times.is_some() != other.times.is_some() {
            return Err(Error::Data("give both sets times, or neither".into()));
        }
        if self.sums_insured.is_some() != other.sums_insured.is_some() {
            return Err(Error::Data(
                "give both sets sums insured, or neither".into(),
            ));
        }
        let with_si = self.sums_insured.is_some();
        let with_times = self.times.is_some();
        let mut dims: Vec<String> = self.label_dims().to_vec();
        dims.extend(
            other
                .label_dims()
                .iter()
                .filter(|d| !self.label_dims().contains(d))
                .cloned(),
        );
        let with_labels = !dims.is_empty();
        // Each set's loss `k`'s values in the merged dimensions.
        let values_in = |src: &EventSet, k: usize| -> Vec<String> {
            let Some(l) = &src.labels else {
                return vec![String::new(); dims.len()];
            };
            let values = l.values(k);
            dims.iter()
                .map(|d| {
                    l.dims
                        .iter()
                        .position(|x| x == d)
                        .map_or_else(String::new, |i| values[i].clone())
                })
                .collect()
        };
        let mut per_loss = Vec::new();
        let mut offsets = Vec::with_capacity(self.offsets.len());
        offsets.push(0);
        let mut losses = Vec::with_capacity(self.losses.len() + other.losses.len());
        let mut sums = Vec::new();
        let mut times = Vec::new();
        for y in 0..self.n_sims() {
            // (time, which set, index within the year)
            let mut order: Vec<(f64, usize, usize)> = (0..self.events(y).len())
                .map(|k| (self.times(y).map_or(0.0, |t| t[k]), 0, k))
                .chain(
                    (0..other.events(y).len())
                        .map(|k| (other.times(y).map_or(0.0, |t| t[k]), 1, k)),
                )
                .collect();
            if with_times {
                order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            }
            for (t, set, k) in order {
                let src = if set == 0 { self } else { other };
                losses.push(src.events(y)[k]);
                if with_si {
                    sums.push(src.sums_insured(y).expect("both carry sums insured")[k]);
                }
                if with_times {
                    times.push(t);
                }
                if with_labels {
                    per_loss.push(values_in(src, src.offsets[y] + k));
                }
            }
            offsets.push(losses.len());
        }
        Ok(Self {
            offsets,
            losses,
            sums_insured: with_si.then_some(sums),
            times: with_times.then_some(times),
            labels: with_labels.then(|| Labels::new(dims.clone(), per_loss)),
            seed: self.seed,
            samplers: if self.samplers == other.samplers {
                self.samplers.clone()
            } else {
                None
            },
            input_hash: merged_hash(&self.input_hash, &other.input_hash),
        })
    }

    /// Each year's losses grouped into occurrences by an hours clause:
    /// losses within `window_hours` of the first loss of a window are one
    /// occurrence, whose loss is their sum and whose time is the window's
    /// start; the next loss after the window opens the next one. A year is
    /// 8,760 hours (times are fractions of the year). Needs times; sums
    /// insured are not carried, since an occurrence spans risks, and nor
    /// are labels: group each peril's set before merging them to keep
    /// those.
    ///
    /// Windows open at each first loss in turn, rather than where they
    /// would recover most, which a cedant may choose.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// let h = 1.0 / 8760.0; // an hour
    /// let per_risk = EventSet::from_years(vec![vec![10.0, 20.0, 5.0, 7.0]], 0)
    ///     .unwrap()
    ///     .with_times(vec![0.5, 0.5 + 30.0 * h, 0.5 + 80.0 * h, 0.9])
    ///     .unwrap();
    /// let occ = per_risk.grouped(72.0).unwrap();
    /// // 10 and 20 within 72 hours; 5 opens a new window, 7 another.
    /// assert_eq!(occ.events(0), &[30.0, 5.0, 7.0]);
    /// ```
    pub fn grouped(&self, window_hours: f64) -> Result<Self> {
        if !(window_hours.is_finite() && window_hours >= 0.0) {
            return Err(Error::InvalidParameter {
                name: "window_hours",
                value: window_hours,
                reason: "must be finite and non-negative",
            });
        }
        let Some(all_times) = self.times.as_deref() else {
            return Err(Error::Data("an hours clause needs each loss's time".into()));
        };
        const HOURS: f64 = 8760.0;
        let mut offsets = Vec::with_capacity(self.offsets.len());
        offsets.push(0);
        let mut losses = Vec::new();
        let mut times = Vec::new();
        for w in self.offsets.windows(2) {
            let mut start: Option<f64> = None;
            let year = w[0]..w[1];
            for (&x, &t) in self.losses[year.clone()].iter().zip(&all_times[year]) {
                match start {
                    Some(s) if (t - s) * HOURS <= window_hours => {
                        *losses.last_mut().expect("a window is open") += x;
                    }
                    _ => {
                        start = Some(t);
                        losses.push(x);
                        times.push(t);
                    }
                }
            }
            offsets.push(losses.len());
        }
        Ok(Self {
            offsets,
            losses,
            sums_insured: None,
            times: Some(times),
            labels: None,
            seed: self.seed,
            samplers: self.samplers.clone(),
            input_hash: self.input_hash.clone(),
        })
    }

    /// Whether the losses carry times.
    pub fn has_times(&self) -> bool {
        self.times.is_some()
    }

    /// Year `sim`'s times, one per loss, when known.
    pub fn times(&self, sim: usize) -> Option<&[f64]> {
        self.times
            .as_deref()
            .map(|t| &t[self.offsets[sim]..self.offsets[sim + 1]])
    }

    /// Simulations `range` alone, with their sums insured, times and
    /// labels.
    pub(crate) fn slice(&self, range: std::ops::Range<usize>) -> EventSet {
        let (lo, hi) = (self.offsets[range.start], self.offsets[range.end]);
        EventSet {
            offsets: self.offsets[range.start..=range.end]
                .iter()
                .map(|o| o - lo)
                .collect(),
            losses: self.losses[lo..hi].to_vec(),
            sums_insured: self.sums_insured.as_ref().map(|s| s[lo..hi].to_vec()),
            times: self.times.as_ref().map(|t| t[lo..hi].to_vec()),
            // Every combination stays, so each slice's results have the
            // same components.
            labels: self.labels.as_ref().map(|l| Labels {
                dims: l.dims.clone(),
                combinations: l.combinations.clone(),
                of_loss: l.of_loss[lo..hi].to_vec(),
            }),
            seed: self.seed,
            samplers: self.samplers.clone(),
            input_hash: self.input_hash.clone(),
        }
    }

    /// Whether the losses carry sums insured.
    pub fn has_sums_insured(&self) -> bool {
        self.sums_insured.is_some()
    }

    /// Year `sim`'s sums insured, one per loss, when known.
    pub fn sums_insured(&self, sim: usize) -> Option<&[f64]> {
        self.sums_insured
            .as_deref()
            .map(|s| &s[self.offsets[sim]..self.offsets[sim + 1]])
    }

    /// Number of simulated years.
    pub fn n_sims(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Year `sim`'s individual losses, in the order they were drawn.
    pub fn events(&self, sim: usize) -> &[f64] {
        &self.losses[self.offsets[sim]..self.offsets[sim + 1]]
    }

    /// Number of losses in each year.
    pub fn counts(&self) -> Vec<usize> {
        self.offsets.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// The seed the years were drawn from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Each year's total loss, as a one-component [`PredictiveDistribution`]
    /// (no dimensions), with the seed, stream scheme and samplers in its
    /// provenance.
    pub fn totals(&self) -> Result<PredictiveDistribution> {
        let totals = (0..self.n_sims())
            .map(|i| self.events(i).iter().sum())
            .collect();
        PredictiveDistribution::from_draws(
            vec![],
            vec![vec![]],
            totals,
            self.provenance("frequency_severity_monte_carlo"),
        )
    }

    /// Each year's largest loss (0 in a year with none), as a one-component
    /// [`PredictiveDistribution`]: its survival function is the simulated
    /// occurrence exceedance (OEP) curve, and its quantiles the occurrence
    /// PMLs ([`crate::pml`]).
    pub fn maxima(&self) -> Result<PredictiveDistribution> {
        let maxima = (0..self.n_sims())
            .map(|i| self.events(i).iter().copied().fold(0.0, f64::max))
            .collect();
        PredictiveDistribution::from_draws(
            vec![],
            vec![vec![]],
            maxima,
            self.provenance("frequency_severity_monte_carlo_maxima"),
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn merge_keeps_the_labels_of_either_set() {
        let cat = EventSet::from_years(vec![vec![100.0], vec![]], 1)
            .unwrap()
            .with_labels(vec![("peril".into(), vec!["wind".into()])])
            .unwrap();
        let other = EventSet::from_years(vec![vec![3.0], vec![5.0]], 2)
            .unwrap()
            .with_labels(vec![("class".into(), vec!["motor".into(), "home".into()])])
            .unwrap();
        let all = cat.merge(&other).unwrap();
        assert_eq!(all.label_dims(), ["peril", "class"]);
        assert_eq!(all.events(0), &[100.0, 3.0]);
        let year: Vec<Vec<String>> = all.labels(0).unwrap().iter().map(|v| v.to_vec()).collect();
        assert_eq!(
            year,
            [
                vec!["wind".to_string(), String::new()],
                vec![String::new(), "motor".into()]
            ]
        );
        // An unlabelled set's losses get "" in every dimension.
        let plain = EventSet::from_years(vec![vec![1.0], vec![2.0]], 0).unwrap();
        let mixed = plain.merge(&cat).unwrap();
        assert_eq!(
            mixed.label_combinations(),
            [vec![String::new()], vec!["wind".to_string()]]
        );
        assert!(!plain.merge(&plain).unwrap().has_labels());
        assert_eq!(
            all.slice(1..2).label_combinations(),
            all.label_combinations()
        );
        assert!(
            all.with_uniform_times()
                .grouped(72.0)
                .map(|g| !g.has_labels())
                .unwrap()
        );
    }

    #[test]
    fn labels_are_checked() {
        let events = EventSet::from_years(vec![vec![1.0, 2.0]], 0).unwrap();
        let two = || vec!["a".to_string(), "b".into()];
        assert!(events.clone().with_labels(vec![]).is_err());
        assert!(
            events
                .clone()
                .with_labels(vec![("kind".into(), two())])
                .is_err()
        );
        assert!(
            events
                .clone()
                .with_labels(vec![("".into(), two())])
                .is_err()
        );
        assert!(
            events
                .clone()
                .with_labels(vec![("p".into(), two()), ("p".into(), two())])
                .is_err()
        );
        assert!(
            events
                .clone()
                .with_labels(vec![("p".into(), vec!["a".into()])])
                .is_err()
        );
        assert!(
            EventSet::from_long_labeled(&[0], &[1.0], vec![("p".into(), two())], 1, 0).is_err()
        );
        assert!(
            EventSet::from_long_labeled(&[1], &[1.0], vec![("p".into(), vec!["a".into()])], 1, 0)
                .is_err()
        );
    }

    use super::*;
    use crate::fft;
    use prospicio_prob::{Grid, NegativeBinomial, Poisson};

    fn severity() -> Grid {
        Grid::new(1.0, vec![0.1, 0.3, 0.25, 0.2, 0.1, 0.05]).unwrap()
    }

    #[test]
    fn a_serial_severity_gives_the_same_draws() {
        use prospicio_prob::Custom;
        use std::sync::Arc;
        let grid = severity();
        let g = grid.clone();
        let serial = Custom::new(
            "grid",
            Arc::new(move |x| Ok(g.cdf(x))),
            Some(Arc::new(move |p| {
                grid.quantile(p).map_err(|e| e.to_string())
            })),
            false,
        )
        .unwrap();
        let freq = Poisson::new(4.0).unwrap();
        let a = simulate_events(&freq, &serial, 2_000, 3).unwrap();
        let b = simulate_events(&freq, &severity(), 2_000, 3).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn identical_across_thread_counts_and_replayable() {
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| simulate_events(&Poisson::new(4.0).unwrap(), &severity(), 5_000, 9))
                .unwrap()
        };
        let one = run(1);
        assert_eq!(one, run(4));
        assert_eq!(one, run(16));

        // Replay year 1234 by hand from its own stream.
        let mut rng = StreamRng::new(9, 1234);
        let n = Poisson::new(4.0)
            .unwrap()
            .quantile(rng.next_open01())
            .unwrap();
        let replay: Vec<f64> = (0..n)
            .map(|_| severity().quantile(rng.next_open01()).unwrap())
            .collect();
        assert_eq!(one.events(1234), replay);
    }

    #[test]
    fn totals_match_fft_by_kolmogorov_smirnov() {
        // A discrete severity makes the FFT result exact, so the simulated
        // totals must be a sample from it.
        let n_sims = 200_000;
        for freq in [
            &Poisson::new(3.0).unwrap() as &(dyn Counting + Sync),
            &NegativeBinomial::new(2.5, 1.5).unwrap(),
        ] {
            let (exact, report) = fft(freq, &severity(), 400).unwrap();
            assert!(report.tail_mass < 1e-12);
            let totals = simulate_events(freq, &severity(), n_sims, 1)
                .unwrap()
                .totals()
                .unwrap();
            let ks = (0..400)
                .map(|k| (totals.cdf(k as f64) - exact.cdf(k as f64)).abs())
                .fold(0.0, f64::max);
            // Critical value at the 0.1% level: 1.95 / sqrt(n).
            assert!(ks < 1.95 / (n_sims as f64).sqrt(), "KS {ks}");
        }
    }

    #[test]
    fn totals_carry_provenance() {
        let events = simulate_events(&Poisson::new(1.0).unwrap(), &severity(), 10, 77).unwrap();
        let totals = events.totals().unwrap();
        assert_eq!(totals.n_sims(), 10);
        assert_eq!(totals.provenance().seed, Some(77));
        assert_eq!(totals.provenance().model, "frequency_severity_monte_carlo");
        assert_eq!(totals.provenance().samplers, Some(current_samplers()));
        assert_eq!(
            events.counts().iter().sum::<usize>(),
            (0..10).map(|i| events.events(i).len()).sum::<usize>()
        );
    }

    #[test]
    fn years_from_elsewhere_record_no_samplers() {
        // Two event loss tables with the default seed: same streams, but
        // nothing says they are the same draws.
        let a = EventSet::from_years(vec![vec![1.0, 2.0], vec![3.0]], 0).unwrap();
        let b = EventSet::from_years(vec![vec![4.0], vec![]], 0).unwrap();
        assert_eq!(a.samplers(), None);
        let (pa, pb) = (a.totals().unwrap(), b.totals().unwrap());
        assert_eq!(pa.provenance().samplers, None);
        assert_eq!(pa.provenance().seed, Some(0));
        assert!(pa.provenance().shares_streams(pb.provenance()));
        assert!(!pa.provenance().replays_same_draws(pb.provenance()));
    }

    #[test]
    fn times_are_checked_and_replay() {
        let events = EventSet::from_years(vec![vec![1.0, 2.0], vec![3.0]], 0).unwrap();
        assert!(events.clone().with_times(vec![0.5, 0.4, 0.1]).is_err());
        assert!(events.clone().with_times(vec![0.1, 1.5, 0.1]).is_err());
        assert!(events.clone().with_times(vec![0.1, 0.2]).is_err());
        // A later year may start before an earlier one ended.
        assert!(events.clone().with_times(vec![0.4, 0.9, 0.1]).is_ok());
        assert!(!events.has_times() && events.times(0).is_none());

        let freq = Poisson::new(4.0).unwrap();
        let few = simulate_events(&freq, &severity(), 50, 8)
            .unwrap()
            .with_uniform_times();
        let many = simulate_events(&freq, &severity(), 500, 8)
            .unwrap()
            .with_uniform_times();
        let mut all = Vec::new();
        for i in 0..50 {
            let t = few.times(i).unwrap();
            assert_eq!(t, many.times(i).unwrap());
            assert_eq!(t.len(), few.events(i).len());
            assert!(t.windows(2).all(|w| w[0] <= w[1]));
            assert!(t.iter().all(|&x| x > 0.0 && x < 1.0));
            all.extend_from_slice(t);
        }
        // Spread over the year.
        let mean = all.iter().sum::<f64>() / all.len() as f64;
        assert!((mean - 0.5).abs() < 0.06, "{mean}");
    }

    #[test]
    fn seasonal_times_follow_the_weights() {
        let events = EventSet::from_years(vec![vec![1.0, 2.0], vec![3.0]], 0).unwrap();
        assert!(events.clone().with_seasonal_times(&[]).is_err());
        assert!(events.clone().with_seasonal_times(&[0.0, 0.0]).is_err());
        assert!(events.clone().with_seasonal_times(&[1.0, -1.0]).is_err());
        assert!(
            events
                .clone()
                .with_seasonal_times(&[1.0, f64::NAN])
                .is_err()
        );

        let freq = Poisson::new(4.0).unwrap();
        let sim = simulate_events(&freq, &severity(), 50_000, 8).unwrap();

        // Equal weights are the uniform times, to rounding.
        let uniform = sim.clone().with_uniform_times();
        let flat = sim.clone().with_seasonal_times(&[3.0; 12]).unwrap();
        for i in 0..1_000 {
            for (a, b) in uniform.times(i).unwrap().iter().zip(flat.times(i).unwrap()) {
                assert!((a - b).abs() < 1e-15, "{a} {b}");
            }
        }

        // A season: nothing in the first quarter, most in the third.
        let weights = [0.0, 1.0, 6.0, 1.0];
        let seasonal = sim.with_seasonal_times(&weights).unwrap();
        let mut all = Vec::new();
        for i in 0..seasonal.n_sims() {
            let t = seasonal.times(i).unwrap();
            assert_eq!(t.len(), seasonal.events(i).len());
            assert!(t.windows(2).all(|w| w[0] <= w[1]));
            assert!(t.iter().all(|&x| (0.25..=1.0).contains(&x)));
            all.extend_from_slice(t);
        }
        // The piecewise-linear cdf of the season.
        let cdf = |x: f64| {
            let mut c = 0.0;
            for (k, w) in weights.iter().enumerate() {
                let (lo, hi) = (k as f64 / 4.0, (k + 1) as f64 / 4.0);
                c += w / 8.0 * ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
            }
            c
        };
        all.sort_by(f64::total_cmp);
        let n = all.len() as f64;
        let ks = all
            .iter()
            .enumerate()
            .map(|(j, &x)| {
                let f = cdf(x);
                (f - j as f64 / n).abs().max(((j + 1) as f64 / n - f).abs())
            })
            .fold(0.0, f64::max);
        // Times within a year are not independent of its count, but pooled
        // over i.i.d. losses they are a sample of the season: the 0.1%
        // critical value 1.95 / sqrt(n).
        assert!(ks < 1.95 / n.sqrt(), "KS {ks}");
    }

    #[test]
    fn rejects_zero_sims() {
        assert!(simulate_events(&Poisson::new(1.0).unwrap(), &severity(), 0, 1).is_err());
    }

    #[test]
    fn merging_keeps_every_loss_and_orders_by_time() {
        let a = EventSet::from_years(vec![vec![1.0, 2.0], vec![3.0]], 5).unwrap();
        let b = EventSet::from_years(vec![vec![10.0], vec![]], 6).unwrap();
        let m = a.merge(&b).unwrap();
        assert_eq!(m.events(0), &[1.0, 2.0, 10.0]);
        assert_eq!(m.events(1), &[3.0]);
        assert_eq!(m.seed(), 5);
        assert!(!m.has_times());
        let si_a = a.clone().with_sums_insured(vec![5.0, 5.0, 5.0]).unwrap();
        let si_b = b.clone().with_sums_insured(vec![20.0]).unwrap();
        assert_eq!(
            si_a.merge(&si_b).unwrap().sums_insured(0),
            Some(&[5.0, 5.0, 20.0][..])
        );
        assert!(si_a.merge(&b).is_err());
        let timed = a.clone().with_times(vec![0.1, 0.2, 0.3]).unwrap();
        assert!(timed.merge(&b).is_err());
        let short = EventSet::from_years(vec![vec![1.0]], 0).unwrap();
        assert!(a.merge(&short).is_err());
        // A tie in time keeps this set's loss first.
        let t_a = EventSet::from_years(vec![vec![1.0]], 0)
            .unwrap()
            .with_times(vec![0.5])
            .unwrap();
        let t_b = EventSet::from_years(vec![vec![2.0]], 0)
            .unwrap()
            .with_times(vec![0.5])
            .unwrap();
        assert_eq!(t_a.merge(&t_b).unwrap().events(0), &[1.0, 2.0]);
        assert_eq!(t_b.merge(&t_a).unwrap().events(0), &[2.0, 1.0]);
    }

    #[test]
    fn an_hours_clause_groups_within_the_window_and_keeps_totals() {
        let h = 1.0 / 8760.0;
        let e = EventSet::from_years(vec![vec![1.0, 2.0, 4.0], vec![], vec![8.0, 16.0]], 0)
            .unwrap()
            .with_times(vec![0.0, 72.0 * h, 72.5 * h, 0.3, 0.3])
            .unwrap();
        let g = e.grouped(72.0).unwrap();
        // 72 hours after the first is still in its window; 72.5 is not.
        assert_eq!(g.events(0), &[3.0, 4.0]);
        assert_eq!(g.times(0), Some(&[0.0, 72.5 * h][..]));
        assert!(g.events(1).is_empty());
        assert_eq!(g.events(2), &[24.0]);
        let total = |s: &EventSet| -> f64 { (0..s.n_sims()).flat_map(|y| s.events(y)).sum() };
        assert_eq!(total(&g), total(&e));
        // A zero window still groups losses at the same time.
        assert_eq!(e.grouped(0.0).unwrap().events(2), &[24.0]);
        assert!(
            EventSet::from_years(vec![vec![1.0]], 0)
                .unwrap()
                .grouped(72.0)
                .is_err()
        );
        assert!(e.grouped(-1.0).is_err());
    }
}
