//! Monte Carlo frequency-severity: simulated years of individual losses.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_prob::provenance::{SIM_INDEX_SCHEME, current_samplers};
use prospicio_prob::{Counting, Distribution, PredictiveDistribution, Provenance};
use rayon::prelude::*;

/// What [`EventSet::contract_years`] does with the partial contract years
/// at the two ends of the simulated timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartialYears {
    /// Keep them: the first holds the losses of the first simulated year
    /// before the contract start, the last those of the last simulated
    /// year from it.
    Keep,
    /// Leave them out, with their losses: only whole contract years.
    #[default]
    Drop,
}

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
    /// Each year's aggregate part, when given: amounts with no event or
    /// time (attritional losses, say), which only covers without
    /// per-occurrence terms see.
    aggregate: Option<AggregatePart>,
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

/// Amounts in each year with no event or time: one row a year, one
/// column per combination of labels.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AggregatePart {
    /// Each column's combination, an index into the labels'
    /// combinations (0 without labels).
    pub(crate) of_column: Vec<usize>,
    /// Row-major, `of_column.len()` amounts a year.
    pub(crate) amounts: Vec<f64>,
}

impl AggregatePart {
    fn row(&self, sim: usize) -> &[f64] {
        let m = self.of_column.len();
        &self.amounts[sim * m..(sim + 1) * m]
    }
}

impl Labels {
    /// From each loss's values, one per dimension.
    fn new(dims: Vec<String>, per_loss: Vec<Vec<String>>) -> Self {
        Self::with_columns(dims, per_loss, &[]).0
    }

    /// From each loss's values and each aggregate column's, whose
    /// combinations are listed too; with each column's combination.
    fn with_columns(
        dims: Vec<String>,
        per_loss: Vec<Vec<String>>,
        columns: &[Vec<String>],
    ) -> (Self, Vec<usize>) {
        let mut combinations: Vec<Vec<String>> = per_loss.iter().chain(columns).cloned().collect();
        combinations.sort();
        combinations.dedup();
        let index = |c: &Vec<String>| {
            combinations
                .binary_search(c)
                .expect("every combination is listed")
        };
        let of_loss = per_loss.iter().map(index).collect();
        let of_column = columns.iter().map(index).collect();
        (
            Self {
                dims,
                combinations,
                of_loss,
            },
            of_column,
        )
    }

    /// Loss `k`'s values, one per dimension.
    fn values(&self, k: usize) -> &[String] {
        &self.combinations[self.of_loss[k]]
    }
}

/// What an hours clause does with the labels of the losses it groups
/// ([`EventSet::grouped_with`], [`EventSet::grouped_by`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupLabels {
    /// The occurrences carry no labels, as [`EventSet::grouped`] gives
    /// them.
    Drop,
    /// An occurrence takes its losses' labels, which must agree: an
    /// occurrence of losses with different labels is an error.
    Agree,
    /// An occurrence takes its largest loss's labels (the earliest of
    /// equal largest losses).
    Largest,
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
        aggregate: None,
        seed,
        samplers: Some(current_samplers()),
        input_hash: None,
    })
}

/// A defensive tail mixture for the losses of an importance sample
/// ([`simulate_events_weighted`]): each loss comes, with chance `share`,
/// from the severity conditioned above `threshold`, and otherwise from the
/// severity itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailMixture {
    /// The threshold above which the tail component draws.
    pub threshold: f64,
    /// The chance of the tail component, in `(0, 1)`.
    pub share: f64,
}

/// Simulated years drawn from a proposal, each with its likelihood ratio,
/// for importance sampling of the tail (#566).
///
/// Year `i` draws from stream `i` of `seed` as [`simulate_events`] does,
/// with two optional changes of measure:
///
/// - `count`, a proposal for the claim count (any family, a Poisson with a
///   higher mean say): the count is drawn from it, and the year's weight
///   carries `p(n) / q(n)` with `p` the count's pmf;
/// - `tail`, a [`TailMixture`]: each loss comes from the severity above the
///   threshold with chance `share`, and its weight is
///   `1 / ((1 − share) + share 1{x > u} / S(u))`.
///
/// The year's weight is the product. Every statistic of the years (a
/// tower's ceded loss, the net, the total) is then estimated by
/// [`WeightedSample`](prospicio_prob::weighted::WeightedSample) with these
/// weights. Without either change the years are [`simulate_events`]'s, and
/// every weight is 1. Weights average 1 in expectation; a proposal that
/// puts too little mass where the year matters shows as a small effective
/// sample size.
///
/// ```
/// use prospicio_aggregate::monte_carlo::{TailMixture, simulate_events_weighted};
/// use prospicio_prob::weighted::WeightedSample;
/// use prospicio_prob::{Empirical, Lognormal, Poisson};
///
/// let freq = Poisson::new(2.0).unwrap();
/// let sev = Lognormal::from_mean_cv(10.0, 2.0).unwrap();
/// let proposal = Poisson::new(4.0).unwrap();
/// let tail = TailMixture { threshold: 40.0, share: 0.3 };
/// let (events, w) =
///     simulate_events_weighted(&freq, &sev, 20_000, Some(&proposal), Some(tail), 7).unwrap();
/// let totals = Empirical::draws(&events.totals().unwrap()).to_vec();
/// let s = WeightedSample::new(totals, w).unwrap();
/// // The weighted mean estimates E[S] = 20.
/// assert!((s.mean() - 20.0).abs() < 4.0 * s.standard_error_mean(20).unwrap());
/// ```
pub fn simulate_events_weighted<N, X>(
    frequency: &N,
    severity: &X,
    n_sims: usize,
    count: Option<&(dyn Counting + Sync)>,
    tail: Option<TailMixture>,
    seed: u64,
) -> Result<(EventSet, Vec<f64>)>
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
    let tail_at = match tail {
        Some(t) => {
            if !(t.share > 0.0 && t.share < 1.0) {
                return Err(Error::InvalidParameter {
                    name: "share",
                    value: t.share,
                    reason: "must be in (0, 1)",
                });
            }
            let s = severity.survival(t.threshold);
            if !(t.threshold.is_finite() && s > 0.0) {
                return Err(Error::InvalidParameter {
                    name: "threshold",
                    value: t.threshold,
                    reason: "must be finite with the severity above it",
                });
            }
            Some((t, severity.cdf(t.threshold), s))
        }
        None => None,
    };
    let year = |i: u64| -> (Vec<f64>, f64) {
        let mut rng = StreamRng::new(seed, i);
        let u = rng.next_open01();
        let (n, mut w) = match count {
            Some(q) => {
                let n = q.quantile(u).expect("next_open01 is always in (0, 1)");
                let (pn, qn) = (frequency.pmf(n), q.pmf(n));
                (n, if qn > 0.0 { pn / qn } else { 0.0 })
            }
            None => (
                frequency
                    .quantile(u)
                    .expect("next_open01 is always in (0, 1)"),
                1.0,
            ),
        };
        let losses = (0..n)
            .map(|_| match tail_at {
                None => severity
                    .quantile(rng.next_open01())
                    .expect("next_open01 is always in (0, 1)"),
                Some((t, f, s)) => {
                    let pick = rng.next_open01();
                    let v = rng.next_open01();
                    let x = if pick < t.share {
                        severity
                            .quantile((f + v * s).min(1.0 - f64::EPSILON / 2.0))
                            .expect("a probability below 1")
                    } else {
                        severity
                            .quantile(v)
                            .expect("next_open01 is always in (0, 1)")
                    };
                    let above = if x > t.threshold { 1.0 / s } else { 0.0 };
                    w /= (1.0 - t.share) + t.share * above;
                    x
                }
            })
            .collect();
        (losses, w)
    };
    let years: Vec<(Vec<f64>, f64)> = if severity.is_parallel_safe() {
        (0..n_sims as u64).into_par_iter().map(year).collect()
    } else {
        (0..n_sims as u64).map(year).collect()
    };
    let mut offsets = Vec::with_capacity(n_sims + 1);
    offsets.push(0);
    let mut losses = Vec::new();
    let mut weights = Vec::with_capacity(n_sims);
    for (year, w) in years {
        losses.extend(year);
        offsets.push(losses.len());
        weights.push(w);
    }
    let events = EventSet {
        offsets,
        losses,
        sums_insured: None,
        times: None,
        labels: None,
        aggregate: None,
        seed,
        samplers: Some(current_samplers()),
        input_hash: None,
    };
    Ok((events, weights))
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
            aggregate: None,
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
        if self.aggregate.is_some() {
            return Err(Error::Data(
                "label the events before giving the aggregate part".into(),
            ));
        }
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

    /// The same events with an aggregate part: amounts in each year that
    /// are no event and have no time, attritional losses say, one column
    /// for each component of `part`, which needs as many simulations as
    /// the set has years. Simulation `i` of `part` goes with year `i`.
    /// Replaces any aggregate part the set had.
    ///
    /// The part's dimensions are label dimensions, so it can be given by
    /// peril or class: the set's labels get the part's dimensions not
    /// among them, with `""` for every loss, and each component's values
    /// (`""` in the set's dimensions it lacks) join
    /// [`label_combinations`](Self::label_combinations). A part with no
    /// dimensions is one unlabelled amount a year.
    ///
    /// [`crate::Tower::apply`] counts the part in gross and net, and only
    /// a layer that [sees the aggregate part](crate::Layer::sees_aggregate_part)
    /// (a quota share or a stop loss: no per-occurrence attachment or
    /// limit, on the loss) cedes from it; per-occurrence layers see the
    /// events alone. In each year the part comes before the events, so a
    /// binding annual deductible is used up on it first and a binding
    /// annual limit stops the last events (`LimitAllocation::EventOrder`),
    /// unless the layer shares them pro rata. [`totals`](Self::totals)
    /// includes it; [`maxima`](Self::maxima) and
    /// [`counts`](Self::counts), which count occurrences, do not.
    ///
    /// ```
    /// use prospicio_aggregate::{EventSet, Layer, Tower};
    /// use prospicio_prob::{Empirical, KeyValue, PredictiveDistribution, Provenance};
    ///
    /// // Attritional 10, 40, 30 beside large losses; a 30 xs 25 per risk
    /// // sees the large ones alone, a quota share sees everything.
    /// let attritional = PredictiveDistribution::from_draws(
    ///     vec![], vec![vec![]], vec![10.0, 40.0, 30.0], Provenance::new("attritional"),
    /// )
    /// .unwrap();
    /// let events = EventSet::from_years(vec![vec![], vec![50.0], vec![5.0, 60.0]], 0)
    ///     .unwrap()
    ///     .with_aggregate_part(&attritional)
    ///     .unwrap();
    /// let tower = Tower::new(vec![
    ///     Layer::xol("30x25", 30.0, 25.0).unwrap(),
    ///     Layer::quota_share("QS", 0.5).unwrap(),
    /// ])
    /// .unwrap();
    /// let result = tower.apply(&events).unwrap();
    /// let at = |kind: &str, layer: &str| {
    ///     let key: Vec<KeyValue> = vec![kind.into(), layer.into()];
    ///     Empirical::draws(&result.marginal(&key).unwrap()).to_vec()
    /// };
    /// assert_eq!(at("gross", "ground_up"), [10.0, 90.0, 95.0]);
    /// assert_eq!(at("ceded", "30x25"), [0.0, 25.0, 30.0]);
    /// assert_eq!(at("ceded", "QS"), [5.0, 45.0, 47.5]);
    /// ```
    pub fn with_aggregate_part(mut self, part: &PredictiveDistribution) -> Result<Self> {
        let n = self.n_sims();
        if part.n_sims() != n {
            return Err(Error::Data(format!(
                "an aggregate part of {} simulations for {n} years",
                part.n_sims()
            )));
        }
        if let Some(&bad) = part
            .draw_matrix()
            .iter()
            .find(|x| !(x.is_finite() && **x >= 0.0))
        {
            return Err(Error::InvalidParameter {
                name: "part",
                value: bad,
                reason: "must be finite and non-negative",
            });
        }
        if let Some(dim) = part.dims().iter().find(|d| *d == "kind" || *d == "layer") {
            return Err(Error::Data(format!(
                "label dimension {dim:?}: names must not be \"kind\" or \"layer\""
            )));
        }
        let mut dims = self.label_dims().to_vec();
        dims.extend(
            part.dims()
                .iter()
                .filter(|d| !self.label_dims().contains(d))
                .cloned(),
        );
        if dims.is_empty() {
            // No labels anywhere: one amount a year.
            self.aggregate = Some(AggregatePart {
                of_column: vec![0],
                amounts: (0..n)
                    .map(|i| part.row(i).expect("a row a year").iter().sum())
                    .collect(),
            });
            return Ok(self);
        }
        let per_loss: Vec<Vec<String>> = (0..self.losses.len())
            .map(|k| {
                let values = self.labels.as_ref().map(|l| l.values(k));
                dims.iter()
                    .enumerate()
                    .map(|(i, _)| values.and_then(|v| v.get(i)).cloned().unwrap_or_default())
                    .collect()
            })
            .collect();
        let columns: Vec<Vec<String>> = part
            .components()
            .iter()
            .map(|key| {
                dims.iter()
                    .map(|d| {
                        part.dims()
                            .iter()
                            .position(|x| x == d)
                            .map_or_else(String::new, |i| key[i].to_string())
                    })
                    .collect()
            })
            .collect();
        let (labels, of_column) = Labels::with_columns(dims, per_loss, &columns);
        self.labels = Some(labels);
        self.aggregate = Some(AggregatePart {
            of_column,
            amounts: part.draw_matrix().to_vec(),
        });
        Ok(self)
    }

    /// Whether the years carry an aggregate part.
    pub fn has_aggregate_part(&self) -> bool {
        self.aggregate.is_some()
    }

    /// Year `sim`'s aggregate part, when given: one amount per column, in
    /// the order of [`aggregate_part_labels`](Self::aggregate_part_labels).
    pub fn aggregate_part(&self, sim: usize) -> Option<&[f64]> {
        self.aggregate.as_ref().map(|a| a.row(sim))
    }

    /// Each column of the aggregate part's labels, one value per label
    /// dimension (none without labels), when the part is given.
    pub fn aggregate_part_labels(&self) -> Option<Vec<&[String]>> {
        self.aggregate.as_ref().map(|a| {
            a.of_column
                .iter()
                .map(|&c| self.labels.as_ref().map_or(&[][..], |l| &l.combinations[c]))
                .collect()
        })
    }

    /// Each column of the aggregate part's combination, as an index into
    /// [`EventSet::label_combinations`].
    pub(crate) fn aggregate_part_indices(&self) -> Option<&[usize]> {
        self.aggregate.as_ref().map(|a| &a.of_column[..])
    }

    /// The same events without their aggregate part, and the part. The labels keep every combination, the
    /// part's too, so [`with_aggregate_columns`](Self::with_aggregate_columns)
    /// can put amounts back on the same columns.
    pub(crate) fn without_aggregate_part(mut self) -> (Self, Option<AggregatePart>) {
        let part = self.aggregate.take();
        (self, part)
    }

    /// The same events with an aggregate part of `amounts` (row-major, one
    /// row a simulation) on the columns `of_column`, indices into the
    /// set's label combinations.
    pub(crate) fn with_aggregate_columns(
        mut self,
        of_column: Vec<usize>,
        amounts: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(amounts.len(), self.n_sims() * of_column.len());
        self.aggregate = Some(AggregatePart { of_column, amounts });
        self
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
        if let Some(a) = &self.aggregate {
            p = p.param("aggregate_part", format!("{} columns", a.of_column.len()));
        }
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

    /// The years in another order: year `i` of the result is year
    /// `index[i]` of this set, with its losses, sums insured, times and
    /// labels. Pass the index a reorder returns
    /// ([`PredictiveDistribution::reorder_groups_indexed`] gives one per
    /// group) to keep a class's events with its reordered totals; any
    /// index into the years works, so a resample with repeats does too.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    /// use prospicio_prob::portfolio::CorrelationKind;
    /// use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
    ///
    /// // A class's events and, from them, its totals beside another class.
    /// let large = EventSet::from_years(vec![vec![], vec![50.0], vec![5.0, 60.0]], 0).unwrap();
    /// let totals: Vec<f64> = (0..3).map(|i| large.events(i).iter().sum()).collect();
    /// let pd = PredictiveDistribution::from_draws(
    ///     vec!["class".into()],
    ///     vec![vec![KeyValue::from("large")], vec![KeyValue::from("other")]],
    ///     totals.iter().zip([3.0, 1.0, 2.0]).flat_map(|(&a, b)| [a, b]).collect(),
    ///     Provenance::new("example"),
    /// )
    /// .unwrap();
    /// let (r, index) = pd
    ///     .reorder_groups_indexed("class", &[1.0, 0.9, 0.9, 1.0], CorrelationKind::Pearson, 1)
    ///     .unwrap();
    /// let events = large.permute(&index[0]).unwrap();
    /// for i in 0..3 {
    ///     assert_eq!(events.events(i).iter().sum::<f64>(), r.row(i).unwrap()[0]);
    /// }
    /// ```
    ///
    /// [`PredictiveDistribution::reorder_groups_indexed`]: prospicio_prob::PredictiveDistribution::reorder_groups_indexed
    pub fn permute(&self, index: &[usize]) -> Result<Self> {
        if index.is_empty() {
            return Err(Error::Data("needs at least one year".into()));
        }
        let n = self.n_sims();
        if let Some(&bad) = index.iter().find(|&&i| i >= n) {
            return Err(Error::Data(format!(
                "year {bad} of an index into {n} years"
            )));
        }
        let mut offsets = Vec::with_capacity(index.len() + 1);
        offsets.push(0);
        let mut losses = Vec::new();
        let mut sums = Vec::new();
        let mut times = Vec::new();
        let mut of_loss = Vec::new();
        let mut aggregate = Vec::new();
        for &y in index {
            if let Some(a) = &self.aggregate {
                aggregate.extend_from_slice(a.row(y));
            }
            let range = self.offsets[y]..self.offsets[y + 1];
            losses.extend_from_slice(&self.losses[range.clone()]);
            if let Some(s) = &self.sums_insured {
                sums.extend_from_slice(&s[range.clone()]);
            }
            if let Some(t) = &self.times {
                times.extend_from_slice(&t[range.clone()]);
            }
            if let Some(l) = &self.labels {
                of_loss.extend_from_slice(&l.of_loss[range]);
            }
            offsets.push(losses.len());
        }
        Ok(Self {
            offsets,
            losses,
            sums_insured: self.sums_insured.as_ref().map(|_| sums),
            times: self.times.as_ref().map(|_| times),
            labels: self.labels.as_ref().map(|l| Labels {
                dims: l.dims.clone(),
                combinations: l.combinations.clone(),
                of_loss,
            }),
            aggregate: self.aggregate.as_ref().map(|a| AggregatePart {
                of_column: a.of_column.clone(),
                amounts: aggregate,
            }),
            seed: self.seed,
            samplers: self.samplers.clone(),
            input_hash: self.input_hash.clone(),
        })
    }

    /// The events regrouped into contract years that start at `start`, a
    /// fraction of the simulated year (1 July is about 0.4959 of a 365-day
    /// year), for a treaty whose year is not the simulation's (#586).
    ///
    /// The simulated years are read as consecutive years of one timeline:
    /// a loss of year `i` at time `t` happens at `i + t`, and contract year
    /// `j` runs from `j + start` to `j + 1 + start`. Each loss moves to its
    /// contract year with its time rebased to it, `i + t − start − j`, and
    /// keeps its sum insured and labels; within a contract year the losses
    /// of the earlier simulated year come first, in time order. Years
    /// simulated independently (a year loss table's) are not a timeline:
    /// regrouping them joins the end of one year to the start of another.
    ///
    /// For `start > 0` the first contract year (from `start − 1`) and the
    /// last (to `n + start`) are partial, holding only the losses of year
    /// 0 before `start` and of year `n − 1` from it. [`PartialYears::Keep`]
    /// keeps them, so the set has `n + 1` years, the first and the last
    /// partial; [`PartialYears::Drop`] leaves them out, so the set has the
    /// `n − 1` whole contract years, and their losses are in no contract
    /// year. `start = 0` returns the set as it is.
    ///
    /// Fails without times (a set from [`with_times`](Self::with_times),
    /// [`with_uniform_times`](Self::with_uniform_times) or a timed year
    /// loss table), for `start` outside `[0, 1)`, with an aggregate part,
    /// which has no times to cut by, and when dropping the partial years
    /// leaves no year.
    ///
    /// ```
    /// use prospicio_aggregate::{EventSet, PartialYears};
    ///
    /// let calendar = EventSet::from_years(vec![vec![1.0, 2.0], vec![3.0]], 0)
    ///     .unwrap()
    ///     .with_times(vec![0.25, 0.75, 0.25])
    ///     .unwrap();
    /// // Contract years from mid-year: the loss at 0.75 and the next
    /// // year's at 0.25 fall in the one whole contract year.
    /// let july = calendar.contract_years(0.5, PartialYears::Drop).unwrap();
    /// assert_eq!(july.n_sims(), 1);
    /// assert_eq!(july.events(0), &[2.0, 3.0]);
    /// assert_eq!(july.times(0), Some(&[0.25, 0.75][..]));
    /// let kept = calendar.contract_years(0.5, PartialYears::Keep).unwrap();
    /// assert_eq!((kept.n_sims(), kept.events(0), kept.events(2)), (3, &[1.0][..], &[][..]));
    /// ```
    pub fn contract_years(&self, start: f64, partial: PartialYears) -> Result<Self> {
        if !(0.0..1.0).contains(&start) {
            return Err(Error::InvalidParameter {
                name: "start",
                value: start,
                reason: "must be a fraction of the year in [0, 1)",
            });
        }
        let Some(times) = &self.times else {
            return Err(Error::Data(
                "contract years cut the losses by their times of loss, which these events do \
                 not carry"
                    .into(),
            ));
        };
        if self.aggregate.is_some() {
            return Err(Error::Data(
                "an aggregate part has no times to cut into contract years; regroup the events \
                 before adding it"
                    .into(),
            ));
        }
        if start == 0.0 {
            return Ok(self.clone());
        }
        let n = self.n_sims();
        // Contract year j + 1 (so the first, partial one, is 0) of each loss.
        let mut years: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n + 1];
        for i in 0..n {
            let (from, to) = (self.offsets[i], self.offsets[i + 1]);
            for (k, &tk) in times.iter().enumerate().take(to).skip(from) {
                let (j, rebased) = if tk >= start {
                    (i + 1, tk - start)
                } else {
                    (i, tk + 1.0 - start)
                };
                years[j].push((k, rebased));
            }
        }
        let keep: Vec<usize> = match partial {
            PartialYears::Keep => (0..=n).collect(),
            PartialYears::Drop => (1..n).collect(),
        };
        if keep.is_empty() {
            return Err(Error::Data(
                "one simulated year holds no whole contract year; keep the partial years".into(),
            ));
        }
        let mut offsets = vec![0];
        let mut losses = Vec::new();
        let mut sums = Vec::new();
        let mut new_times = Vec::new();
        let mut of_loss = Vec::new();
        for &j in &keep {
            for &(k, rebased) in &years[j] {
                losses.push(self.losses[k]);
                new_times.push(rebased.min(1.0));
                if let Some(s) = &self.sums_insured {
                    sums.push(s[k]);
                }
                if let Some(l) = &self.labels {
                    of_loss.push(l.of_loss[k]);
                }
            }
            offsets.push(losses.len());
        }
        Ok(Self {
            offsets,
            losses,
            sums_insured: self.sums_insured.as_ref().map(|_| sums),
            times: Some(new_times),
            labels: self.labels.as_ref().map(|l| Labels {
                dims: l.dims.clone(),
                combinations: l.combinations.clone(),
                of_loss,
            }),
            aggregate: None,
            seed: self.seed,
            samplers: self.samplers.clone(),
            input_hash: self.input_hash.clone(),
        })
    }

    /// The losses whose label `dim` takes one of `values`, with every year
    /// kept (a year with none of them is empty), each loss's sum insured,
    /// time and labels kept with it, and the aggregate part's columns with
    /// one of the values. The set a layer scoped to those values
    /// ([`crate::Layer::scope`]) sees: the scoped layer on the whole set
    /// cedes what the same layer unscoped cedes on this one.
    ///
    /// ```
    /// use prospicio_aggregate::EventSet;
    ///
    /// let events = EventSet::from_years(vec![vec![25.0, 30.0], vec![40.0]], 1)
    ///     .unwrap()
    ///     .with_labels(vec![(
    ///         "peril".into(),
    ///         vec!["ws".into(), "eq".into(), "eq".into()],
    ///     )])
    ///     .unwrap();
    /// let ws = events.filter("peril", &["ws"]).unwrap();
    /// assert_eq!(ws.n_sims(), 2);
    /// assert_eq!(ws.events(0), &[25.0]);
    /// assert!(ws.events(1).is_empty());
    /// ```
    pub fn filter(&self, dim: &str, values: &[&str]) -> Result<Self> {
        let labels = self
            .labels
            .as_ref()
            .ok_or_else(|| Error::Data("a filter needs labelled losses".into()))?;
        let d = labels.dims.iter().position(|x| x == dim).ok_or_else(|| {
            Error::Data(format!(
                "no label dimension {dim:?} among {:?}",
                labels.dims
            ))
        })?;
        if values.is_empty() {
            return Err(Error::Data("a filter needs at least one value".into()));
        }
        let keep_combination: Vec<bool> = labels
            .combinations
            .iter()
            .map(|c| values.contains(&c[d].as_str()))
            .collect();
        let mut offsets = Vec::with_capacity(self.offsets.len());
        offsets.push(0);
        let mut losses = Vec::new();
        let mut sums = Vec::new();
        let mut times = Vec::new();
        let mut per_loss = Vec::new();
        for y in 0..self.n_sims() {
            for k in self.offsets[y]..self.offsets[y + 1] {
                if !keep_combination[labels.of_loss[k]] {
                    continue;
                }
                losses.push(self.losses[k]);
                if let Some(s) = &self.sums_insured {
                    sums.push(s[k]);
                }
                if let Some(t) = &self.times {
                    times.push(t[k]);
                }
                per_loss.push(labels.values(k).to_vec());
            }
            offsets.push(losses.len());
        }
        let kept_columns: Vec<usize> = self.aggregate.as_ref().map_or_else(Vec::new, |a| {
            (0..a.of_column.len())
                .filter(|&j| keep_combination[a.of_column[j]])
                .collect()
        });
        let columns: Vec<Vec<String>> = self.aggregate.as_ref().map_or_else(Vec::new, |a| {
            kept_columns
                .iter()
                .map(|&j| labels.combinations[a.of_column[j]].clone())
                .collect()
        });
        let (new_labels, of_column) = Labels::with_columns(labels.dims.clone(), per_loss, &columns);
        let aggregate = self.aggregate.as_ref().and_then(|a| {
            (!kept_columns.is_empty()).then(|| AggregatePart {
                of_column,
                amounts: (0..self.n_sims())
                    .flat_map(|y| {
                        let row = a.row(y);
                        kept_columns.iter().map(move |&j| row[j])
                    })
                    .collect(),
            })
        });
        Ok(Self {
            offsets,
            losses,
            sums_insured: self.sums_insured.as_ref().map(|_| sums),
            times: self.times.as_ref().map(|_| times),
            labels: Some(new_labels),
            aggregate,
            seed: self.seed,
            samplers: self.samplers.clone(),
            input_hash: self.input_hash.clone(),
        })
    }

    /// This set with each entry's amount replaced: `by_year[i]` holds year
    /// `i`'s aggregate part's columns (when `aggregate` is set and the set
    /// has one), then its events. Events whose new amount is 0 are dropped
    /// with their times and labels; sums insured are dropped, since they
    /// belong to the original losses; the samplers are cleared.
    pub(crate) fn with_amounts(&self, by_year: &[Vec<f64>], aggregate: bool) -> Self {
        let m = self.aggregate.as_ref().map_or(0, |a| a.of_column.len());
        let keep_aggregate = aggregate && m > 0;
        let mut offsets = Vec::with_capacity(self.offsets.len());
        offsets.push(0);
        let mut losses = Vec::new();
        let mut times = Vec::new();
        let mut per_loss = Vec::new();
        let mut amounts = Vec::new();
        for (y, row) in by_year.iter().enumerate() {
            if keep_aggregate {
                amounts.extend_from_slice(&row[..m]);
            }
            for (j, k) in (self.offsets[y]..self.offsets[y + 1]).enumerate() {
                let x = row[m + j];
                if x == 0.0 {
                    continue;
                }
                losses.push(x);
                if let Some(t) = &self.times {
                    times.push(t[k]);
                }
                if let Some(l) = &self.labels {
                    per_loss.push(l.values(k).to_vec());
                }
            }
            offsets.push(losses.len());
        }
        let (labels, aggregate) = match &self.labels {
            Some(l) => {
                let columns: Vec<Vec<String>> = if keep_aggregate {
                    let a = self.aggregate.as_ref().expect("an aggregate part");
                    a.of_column
                        .iter()
                        .map(|&c| l.combinations[c].clone())
                        .collect()
                } else {
                    Vec::new()
                };
                let (labels, of_column) = Labels::with_columns(l.dims.clone(), per_loss, &columns);
                (
                    Some(labels),
                    keep_aggregate.then_some(AggregatePart { of_column, amounts }),
                )
            }
            None => (
                None,
                keep_aggregate.then(|| AggregatePart {
                    of_column: vec![0],
                    amounts,
                }),
            ),
        };
        Self {
            offsets,
            losses,
            sums_insured: None,
            times: self.times.as_ref().map(|_| times),
            labels,
            aggregate,
            seed: self.seed,
            samplers: None,
            input_hash: self.input_hash.clone(),
        }
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
    /// Either set's aggregate part is kept, and the two sets' columns
    /// with the same labels are added. The seed is this set's; the samplers are kept only when the two
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
        // Either set's aggregate columns in the merged dimensions, one
        // column per combination, the amounts of a shared one added.
        let mut columns: Vec<Vec<String>> = Vec::new();
        let mut column_of: Vec<(usize, usize, usize)> = Vec::new(); // (set, column, merged column)
        for (set, src) in [self, other].into_iter().enumerate() {
            let Some(a) = &src.aggregate else { continue };
            for (j, &c) in a.of_column.iter().enumerate() {
                let values: Vec<String> = match &src.labels {
                    Some(l) => dims
                        .iter()
                        .map(|d| {
                            l.dims
                                .iter()
                                .position(|x| x == d)
                                .map_or_else(String::new, |i| l.combinations[c][i].clone())
                        })
                        .collect(),
                    None => vec![String::new(); dims.len()],
                };
                let at = columns
                    .iter()
                    .position(|v| *v == values)
                    .unwrap_or_else(|| {
                        columns.push(values);
                        columns.len() - 1
                    });
                column_of.push((set, j, at));
            }
        }
        let n = self.n_sims();
        let aggregate_amounts = (!column_of.is_empty()).then(|| {
            let m = columns.len();
            let mut amounts = vec![0.0; n * m];
            for &(set, j, at) in &column_of {
                let src = if set == 0 { self } else { other };
                for i in 0..n {
                    amounts[i * m + at] += src.aggregate_part(i).expect("the set has the part")[j];
                }
            }
            amounts
        });
        let (labels, of_column) = if with_labels {
            let (l, of) = Labels::with_columns(dims.clone(), per_loss, &columns);
            (Some(l), of)
        } else {
            (None, vec![0; columns.len()])
        };
        Ok(Self {
            offsets,
            losses,
            sums_insured: with_si.then_some(sums),
            times: with_times.then_some(times),
            labels,
            aggregate: aggregate_amounts.map(|amounts| AggregatePart { of_column, amounts }),
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
    /// are labels ([`GroupLabels::Drop`]): [`EventSet::grouped_with`] keeps
    /// them, and [`EventSet::grouped_by`] applies the clause within each
    /// peril. An aggregate part stays, summed over its labels into one
    /// amount a year.
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
        self.group(window_hours, None, GroupLabels::Drop)
    }

    /// [`EventSet::grouped`] keeping the labels by `rule`: an occurrence
    /// takes the labels of its losses when they agree, and `rule` says what
    /// it takes when they do not. An aggregate part keeps its labels too.
    ///
    /// ```
    /// use prospicio_aggregate::{EventSet, GroupLabels};
    ///
    /// let h = 1.0 / 8760.0;
    /// let peril = |v: &[&str]| vec![("peril".to_string(), v.iter().map(|s| s.to_string()).collect())];
    /// let e = EventSet::from_years(vec![vec![10.0, 20.0, 5.0]], 0)
    ///     .unwrap()
    ///     .with_times(vec![0.5, 0.5 + 30.0 * h, 0.9])
    ///     .unwrap()
    ///     .with_labels(peril(&["wind", "flood", "wind"]))
    ///     .unwrap();
    /// // 10 (wind) and 20 (flood) fall in one window: they disagree.
    /// assert!(e.grouped_with(72.0, GroupLabels::Agree).is_err());
    /// let g = e.grouped_with(72.0, GroupLabels::Largest).unwrap();
    /// assert_eq!(g.events(0), &[30.0, 5.0]);
    /// assert_eq!(g.labels(0).unwrap(), [["flood".to_string()], ["wind".to_string()]]);
    /// ```
    pub fn grouped_with(&self, window_hours: f64, rule: GroupLabels) -> Result<Self> {
        self.group(window_hours, None, rule)
    }

    /// The hours clause within each value of label dimension `dim`: a
    /// window holds only losses of its own peril, say, so a windstorm and
    /// a flood a day apart stay two occurrences. Occurrences keep the
    /// labels, `rule` deciding the other dimensions' values when a window's
    /// losses disagree on them, and are listed by the time each window
    /// opens. Needs times and labels.
    ///
    /// ```
    /// use prospicio_aggregate::{EventSet, GroupLabels};
    ///
    /// let h = 1.0 / 8760.0;
    /// let e = EventSet::from_years(vec![vec![10.0, 20.0, 5.0]], 0)
    ///     .unwrap()
    ///     .with_times(vec![0.5, 0.5 + 30.0 * h, 0.5 + 40.0 * h])
    ///     .unwrap()
    ///     .with_labels(vec![("peril".into(), vec!["wind".into(), "flood".into(), "wind".into()])])
    ///     .unwrap();
    /// let g = e.grouped_by("peril", 72.0, GroupLabels::Agree).unwrap();
    /// // Wind's 10 and 5 are one occurrence, flood's 20 another.
    /// assert_eq!(g.events(0), &[15.0, 20.0]);
    /// ```
    pub fn grouped_by(&self, dim: &str, window_hours: f64, rule: GroupLabels) -> Result<Self> {
        let Some(labels) = &self.labels else {
            return Err(Error::Data(format!(
                "an hours clause by label {dim:?} needs labels"
            )));
        };
        let Some(d) = labels.dims.iter().position(|x| x == dim) else {
            return Err(Error::Data(format!("no label dimension {dim:?}")));
        };
        self.group(window_hours, Some(d), rule)
    }

    /// The hours clause, windows kept apart by the value of label dimension
    /// `by` when given, labels kept by `rule`.
    fn group(&self, window_hours: f64, by: Option<usize>, rule: GroupLabels) -> Result<Self> {
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
        let labels = self.labels.as_ref().filter(|_| rule != GroupLabels::Drop);
        let mut offsets = Vec::with_capacity(self.offsets.len());
        offsets.push(0);
        let mut losses = Vec::new();
        let mut times = Vec::new();
        // Each occurrence's combination, and its largest loss's.
        let mut of_loss: Vec<usize> = Vec::new();
        let mut largest: Vec<f64> = Vec::new();
        for (sim, w) in self.offsets.windows(2).enumerate() {
            // Each open window: its key, start and occurrence.
            let mut open: Vec<(usize, f64, usize)> = Vec::new();
            for (k, &t) in all_times.iter().enumerate().take(w[1]).skip(w[0]) {
                let x = self.losses[k];
                let key = match (by, &self.labels) {
                    (Some(d), Some(l)) => {
                        // The first combination with the value stands for it.
                        let v = &l.combinations[l.of_loss[k]][d];
                        l.combinations
                            .iter()
                            .position(|c| &c[d] == v)
                            .expect("listed")
                    }
                    _ => 0,
                };
                let found = open.iter_mut().find(|(g, _, _)| *g == key);
                match found {
                    Some((_, s, at)) if (t - *s) * HOURS <= window_hours => {
                        let at = *at;
                        losses[at] += x;
                        if let Some(l) = labels {
                            let c = l.of_loss[k];
                            if rule == GroupLabels::Agree && c != of_loss[at] {
                                return Err(Error::Data(format!(
                                    "year {}: the occurrence at time {} holds losses labelled \
                                     {:?} and {:?}; group by a label, or take the largest \
                                     loss's labels",
                                    sim + 1,
                                    times[at],
                                    l.combinations[of_loss[at]],
                                    l.combinations[c]
                                )));
                            }
                            // Ties keep the earlier loss's labels.
                            if x > largest[at] {
                                largest[at] = x;
                                of_loss[at] = c;
                            }
                        }
                    }
                    found => {
                        let at = losses.len();
                        match found {
                            Some(entry) => *entry = (key, t, at),
                            None => open.push((key, t, at)),
                        }
                        losses.push(x);
                        times.push(t);
                        if let Some(l) = labels {
                            of_loss.push(l.of_loss[k]);
                            largest.push(x);
                        }
                    }
                }
            }
            offsets.push(losses.len());
        }
        let (labels, aggregate) = match labels {
            Some(l) => {
                let per_loss = of_loss.iter().map(|&c| l.combinations[c].clone()).collect();
                let columns: Vec<Vec<String>> = self
                    .aggregate
                    .as_ref()
                    .map(|a| {
                        a.of_column
                            .iter()
                            .map(|&c| l.combinations[c].clone())
                            .collect()
                    })
                    .unwrap_or_default();
                let (labels, of_column) = Labels::with_columns(l.dims.clone(), per_loss, &columns);
                let aggregate = self.aggregate.as_ref().map(|a| AggregatePart {
                    of_column,
                    amounts: a.amounts.clone(),
                });
                (Some(labels), aggregate)
            }
            None => (
                None,
                self.aggregate.as_ref().map(|a| AggregatePart {
                    of_column: vec![0],
                    amounts: (0..self.n_sims()).map(|i| a.row(i).iter().sum()).collect(),
                }),
            ),
        };
        Ok(Self {
            offsets,
            losses,
            sums_insured: None,
            times: Some(times),
            labels,
            aggregate,
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

    /// The offset of simulation `sim`'s first loss among all the losses.
    pub(crate) fn offset(&self, sim: usize) -> usize {
        self.offsets[sim]
    }

    /// A set with one simulation per entry of `rows`, each a list of
    /// `(loss, scale, time)`: loss `loss` (an index among all this set's
    /// losses) times `scale`, with its sum insured scaled alike and its
    /// labels, at `time`. The times must lie in `[0, 1]` and not decrease
    /// within a simulation. The set must have no aggregate part.
    pub(crate) fn rebuilt(&self, rows: &[Vec<(usize, f64, f64)>]) -> EventSet {
        debug_assert!(self.aggregate.is_none());
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        offsets.push(0);
        let mut losses = Vec::new();
        let mut sums = Vec::new();
        let mut times = Vec::new();
        let mut of_loss = Vec::new();
        for row in rows {
            for &(i, scale, time) in row {
                losses.push(self.losses[i] * scale);
                if let Some(s) = &self.sums_insured {
                    sums.push(s[i] * scale);
                }
                times.push(time);
                if let Some(l) = &self.labels {
                    of_loss.push(l.of_loss[i]);
                }
            }
            offsets.push(losses.len());
        }
        EventSet {
            offsets,
            losses,
            sums_insured: self.sums_insured.as_ref().map(|_| sums),
            times: Some(times),
            labels: self.labels.as_ref().map(|l| Labels {
                dims: l.dims.clone(),
                combinations: l.combinations.clone(),
                of_loss,
            }),
            aggregate: None,
            seed: self.seed,
            samplers: self.samplers.clone(),
            input_hash: self.input_hash.clone(),
        }
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
            aggregate: self.aggregate.as_ref().map(|a| {
                let m = a.of_column.len();
                AggregatePart {
                    of_column: a.of_column.clone(),
                    amounts: a.amounts[range.start * m..range.end * m].to_vec(),
                }
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
            .map(|i| {
                self.events(i).iter().sum::<f64>()
                    + self.aggregate_part(i).map_or(0.0, |a| a.iter().sum())
            })
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
    fn part(dims: Vec<String>, keys: Vec<Vec<&str>>, draws: Vec<f64>) -> PredictiveDistribution {
        let keys = keys
            .into_iter()
            .map(|k| k.into_iter().map(prospicio_prob::KeyValue::from).collect())
            .collect();
        PredictiveDistribution::from_draws(dims, keys, draws, Provenance::new("part")).unwrap()
    }

    #[test]
    fn contract_years_from_the_simulated_start_are_the_set() {
        let e = EventSet::from_years(vec![vec![1.0, 2.0], vec![3.0]], 4)
            .unwrap()
            .with_times(vec![0.1, 0.9, 0.5])
            .unwrap();
        assert_eq!(e.contract_years(0.0, PartialYears::Drop).unwrap(), e);
        assert!(e.contract_years(1.0, PartialYears::Drop).is_err());
        assert!(e.contract_years(-0.1, PartialYears::Drop).is_err());
        let untimed = EventSet::from_years(vec![vec![1.0]], 0).unwrap();
        assert!(untimed.contract_years(0.5, PartialYears::Keep).is_err());
    }

    #[test]
    fn a_set_shifted_by_the_start_regroups_to_itself() {
        // Each loss of year i at time t moved to i + t + start on a
        // calendar timeline, one year longer, then cut at start.
        let mut rng = prospicio_core::StreamRng::new(11, 0);
        let n = 30;
        let mut years = Vec::new();
        let mut times = Vec::new();
        for _ in 0..n {
            let k = (rng.next_open01() * 4.0) as usize;
            let mut t: Vec<f64> = (0..k).map(|_| rng.next_open01()).collect();
            t.sort_by(f64::total_cmp);
            years.push(
                (0..k)
                    .map(|_| 1.0 + rng.next_open01() * 9.0)
                    .collect::<Vec<_>>(),
            );
            times.extend(t);
        }
        let labels: Vec<String> = (0..times.len()).map(|k| format!("c{}", k % 3)).collect();
        let original = EventSet::from_years(years.clone(), 2)
            .unwrap()
            .with_times(times.clone())
            .unwrap()
            .with_labels(vec![("class".into(), labels.clone())])
            .unwrap();
        for &start in &[0.25, 0.5, 0.875] {
            let mut cal: Vec<Vec<(f64, f64, String)>> = vec![Vec::new(); n + 1];
            let mut k = 0;
            for (i, y) in years.iter().enumerate() {
                for &x in y {
                    let at = i as f64 + times[k] + start;
                    let year = at.floor() as usize;
                    cal[year].push((at - year as f64, x, labels[k].clone()));
                    k += 1;
                }
            }
            for y in &mut cal {
                y.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
            let calendar = EventSet::from_years(
                cal.iter()
                    .map(|y| y.iter().map(|e| e.1).collect())
                    .collect(),
                2,
            )
            .unwrap()
            .with_times(cal.iter().flatten().map(|e| e.0).collect())
            .unwrap()
            .with_labels(vec![(
                "class".into(),
                cal.iter().flatten().map(|e| e.2.clone()).collect(),
            )])
            .unwrap();
            let back = calendar.contract_years(start, PartialYears::Drop).unwrap();
            assert_eq!(back.n_sims(), n);
            for i in 0..n {
                assert_eq!(
                    back.events(i),
                    original.events(i),
                    "start {start}, year {i}"
                );
                let (a, b) = (back.times(i).unwrap(), original.times(i).unwrap());
                assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12));
                assert_eq!(back.labels(i), original.labels(i));
            }
            // Keeping the partial years: the first is empty here, the last
            // holds what ran past the original's end.
            let kept = calendar.contract_years(start, PartialYears::Keep).unwrap();
            assert_eq!(kept.n_sims(), n + 2);
            assert!(kept.events(0).is_empty());
            let total = |e: &EventSet| {
                (0..e.n_sims())
                    .map(|i| e.events(i).iter().sum::<f64>())
                    .sum::<f64>()
            };
            assert!((total(&kept) - total(&calendar)).abs() < 1e-9);
        }
    }

    #[test]
    fn the_aggregate_part_follows_its_years() {
        let p = part(
            vec!["peril".into()],
            vec![vec!["fire"], vec!["wind"]],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let events = EventSet::from_years(vec![vec![7.0], vec![], vec![8.0, 9.0]], 0)
            .unwrap()
            .with_times(vec![0.5, 0.1, 0.2])
            .unwrap()
            .with_aggregate_part(&p)
            .unwrap();
        // The events get "" in the part's dimension.
        assert_eq!(events.label_combinations(), [[""], ["fire"], ["wind"]]);
        assert_eq!(events.aggregate_part_indices(), Some(&[1, 2][..]));
        let moved = events.permute(&[2, 0, 0]).unwrap();
        assert_eq!(moved.aggregate_part(0), Some(&[5.0, 6.0][..]));
        assert_eq!(moved.aggregate_part(2), Some(&[1.0, 2.0][..]));
        assert_eq!(moved.events(0), &[8.0, 9.0]);
        let years = events.term_years(3).unwrap();
        assert_eq!(years[1].aggregate_part(0), Some(&[3.0, 4.0][..]));
        let totals: Vec<f64> =
            prospicio_prob::Empirical::draws(events.totals().unwrap().total()).to_vec();
        assert_eq!(totals, [10.0, 7.0, 28.0]);
        let maxima: Vec<f64> =
            prospicio_prob::Empirical::draws(events.maxima().unwrap().total()).to_vec();
        assert_eq!(maxima, [7.0, 0.0, 9.0]);
        assert_eq!(events.counts(), [1, 0, 2]);
        // An hours clause keeps the part, summed over its labels.
        let grouped = events.grouped(0.0).unwrap();
        assert!(!grouped.has_labels());
        assert_eq!(grouped.aggregate_part(2), Some(&[11.0][..]));
        // Kept by label, the part keeps its columns.
        let kept = events.grouped_with(0.0, GroupLabels::Largest).unwrap();
        assert_eq!(kept.aggregate_part(2), Some(&[5.0, 6.0][..]));
        assert_eq!(kept.aggregate_part_indices(), Some(&[1, 2][..]));
        assert_eq!(kept.events(2), &[8.0, 9.0]);
        // Labels go before the part.
        assert!(
            events
                .clone()
                .with_labels(vec![("class".into(), vec!["a".into(); 3])])
                .is_err()
        );
        let short = part(vec![], vec![vec![]], vec![1.0, 2.0]);
        assert!(events.clone().with_aggregate_part(&short).is_err());
        let negative = part(vec![], vec![vec![]], vec![1.0, -2.0, 3.0]);
        assert!(events.clone().with_aggregate_part(&negative).is_err());
        let kind = part(vec!["kind".into()], vec![vec!["a"]], vec![1.0, 2.0, 3.0]);
        assert!(events.clone().with_aggregate_part(&kind).is_err());
    }

    #[test]
    fn merging_adds_aggregate_parts_with_the_same_labels() {
        let a = EventSet::from_years(vec![vec![7.0], vec![]], 0)
            .unwrap()
            .with_aggregate_part(&part(
                vec!["peril".into()],
                vec![vec!["fire"]],
                vec![1.0, 2.0],
            ))
            .unwrap();
        let b = EventSet::from_years(vec![vec![], vec![3.0]], 0)
            .unwrap()
            .with_aggregate_part(&part(
                vec!["peril".into()],
                vec![vec!["fire"], vec!["wind"]],
                vec![10.0, 20.0, 30.0, 40.0],
            ))
            .unwrap();
        let all = a.merge(&b).unwrap();
        let labels: Vec<Vec<String>> = all
            .aggregate_part_labels()
            .unwrap()
            .into_iter()
            .map(<[String]>::to_vec)
            .collect();
        assert_eq!(labels, [["fire"], ["wind"]]);
        assert_eq!(all.aggregate_part(0), Some(&[11.0, 20.0][..]));
        assert_eq!(all.aggregate_part(1), Some(&[32.0, 40.0][..]));
        // Unlabelled parts add into one amount a year; one set's alone is kept.
        let c = EventSet::from_years(vec![vec![1.0], vec![2.0]], 0).unwrap();
        let d = c
            .clone()
            .with_aggregate_part(&part(vec![], vec![vec![]], vec![5.0, 6.0]))
            .unwrap();
        let both = d.merge(&d).unwrap();
        assert_eq!(both.aggregate_part(1), Some(&[12.0][..]));
        assert!(!both.has_labels());
        assert_eq!(c.merge(&d).unwrap().aggregate_part(0), Some(&[5.0][..]));
        assert!(!c.merge(&c).unwrap().has_aggregate_part());
    }

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

    fn peril(values: &[&str]) -> Vec<(String, Vec<String>)> {
        vec![(
            "peril".into(),
            values.iter().map(|v| v.to_string()).collect(),
        )]
    }

    #[test]
    fn an_hours_clause_keeps_labels_that_agree() {
        let h = 1.0 / 8760.0;
        let e = EventSet::from_years(vec![vec![1.0, 2.0, 4.0], vec![8.0]], 0)
            .unwrap()
            .with_times(vec![0.0, 10.0 * h, 0.5, 0.2])
            .unwrap()
            .with_labels(peril(&["wind"; 4]))
            .unwrap();
        // One label value is kept by either rule; the default drops it.
        for rule in [GroupLabels::Agree, GroupLabels::Largest] {
            let g = e.grouped_with(72.0, rule).unwrap();
            assert_eq!(g.events(0), &[3.0, 4.0]);
            assert_eq!(g.label_combinations(), [["wind"]]);
            assert_eq!(g.labels(1).unwrap(), [["wind".to_string()]]);
        }
        let dropped = e.grouped_with(72.0, GroupLabels::Drop).unwrap();
        assert_eq!(dropped, e.grouped(72.0).unwrap());
        assert!(!dropped.has_labels());
        // Without labels, every rule groups as the plain clause does.
        let plain = EventSet::from_years(vec![vec![1.0, 2.0]], 0)
            .unwrap()
            .with_times(vec![0.0, 0.0])
            .unwrap();
        assert_eq!(
            plain.grouped_with(0.0, GroupLabels::Agree).unwrap(),
            plain.grouped(0.0).unwrap()
        );
        assert!(plain.grouped_by("peril", 0.0, GroupLabels::Agree).is_err());
    }

    #[test]
    fn an_occurrence_of_mixed_labels_takes_its_largest_loss_or_errs() {
        let h = 1.0 / 8760.0;
        let e = EventSet::from_years(vec![vec![5.0, 9.0, 9.0, 2.0]], 0)
            .unwrap()
            .with_times(vec![0.1, 0.1 + h, 0.1 + 2.0 * h, 0.6])
            .unwrap()
            .with_labels(peril(&["wind", "flood", "quake", "quake"]))
            .unwrap();
        let err = e
            .grouped_with(72.0, GroupLabels::Agree)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("year 1") && err.contains("wind") && err.contains("flood"),
            "{err}"
        );
        let g = e.grouped_with(72.0, GroupLabels::Largest).unwrap();
        assert_eq!(g.events(0), &[23.0, 2.0]);
        // The first of the two largest (flood's 9 before quake's 9).
        assert_eq!(
            g.labels(0).unwrap(),
            [["flood".to_string()], ["quake".to_string()]]
        );
    }

    #[test]
    fn an_hours_clause_by_peril_keeps_perils_apart() {
        let h = 1.0 / 8760.0;
        let class = |v: &[&str]| {
            (
                "class".to_string(),
                v.iter().map(|s| s.to_string()).collect(),
            )
        };
        let mut labels = peril(&["wind", "flood", "wind", "wind", "flood"]);
        labels.push(class(&["home", "home", "auto", "home", "home"]));
        let e = EventSet::from_years(vec![vec![1.0, 2.0, 4.0, 16.0, 8.0]], 0)
            .unwrap()
            .with_times(vec![
                0.1,
                0.1 + 10.0 * h,
                0.1 + 20.0 * h,
                0.1 + 80.0 * h,
                0.1 + 100.0 * h,
            ])
            .unwrap()
            .with_labels(labels)
            .unwrap();
        // Wind: 1 and 4 at 0 and 20 hours, 16 at 80 opens a new window;
        // flood: 2 at 10 hours, 8 at 100 (90 hours on) opens a new one.
        // Listed by the time each window opens.
        let g = e.grouped_by("peril", 72.0, GroupLabels::Largest).unwrap();
        assert_eq!(g.events(0), &[5.0, 2.0, 16.0, 8.0]);
        let t = g.times(0).unwrap();
        assert_eq!(t, &[0.1, 0.1 + 10.0 * h, 0.1 + 80.0 * h, 0.1 + 100.0 * h]);
        // Wind's first occurrence mixes classes and takes auto's (the 4).
        assert_eq!(
            g.labels(0).unwrap()[0],
            ["wind".to_string(), "auto".to_string()]
        );
        // The classes disagree within wind's first window.
        assert!(e.grouped_by("peril", 72.0, GroupLabels::Agree).is_err());
        assert!(e.grouped_by("region", 72.0, GroupLabels::Agree).is_err());
        // Grouping by peril on one peril is the plain clause.
        let wind = EventSet::from_years(vec![vec![1.0, 4.0, 16.0]], 0)
            .unwrap()
            .with_times(vec![0.1, 0.1 + 20.0 * h, 0.1 + 80.0 * h])
            .unwrap()
            .with_labels(peril(&["wind"; 3]))
            .unwrap();
        assert_eq!(
            wind.grouped_by("peril", 72.0, GroupLabels::Agree).unwrap(),
            wind.grouped_with(72.0, GroupLabels::Agree).unwrap()
        );
    }

    #[test]
    fn permute_carries_every_field_of_a_year() {
        let e = EventSet::from_years(vec![vec![1.0, 2.0], vec![], vec![3.0]], 9)
            .unwrap()
            .with_times(vec![0.1, 0.4, 0.8])
            .unwrap()
            .with_sums_insured(vec![10.0, 20.0, 30.0])
            .unwrap()
            .with_labels(vec![(
                "peril".into(),
                vec!["wind".into(), "quake".into(), "wind".into()],
            )])
            .unwrap()
            .with_input_hash(Some("h".into()));
        let p = e.permute(&[2, 0, 0, 1]).unwrap();
        assert_eq!(p.n_sims(), 4);
        assert_eq!(p.events(0), &[3.0]);
        assert_eq!(p.events(1), &[1.0, 2.0]);
        assert_eq!(p.events(2), &[1.0, 2.0]);
        assert!(p.events(3).is_empty());
        assert_eq!(p.times(0), Some(&[0.8][..]));
        assert_eq!(p.sums_insured(2), Some(&[10.0, 20.0][..]));
        let wind = ["wind".to_string()];
        let quake = ["quake".to_string()];
        assert_eq!(p.labels(1).unwrap(), vec![&wind[..], &quake[..]]);
        assert_eq!(p.label_combinations(), e.label_combinations());
        assert_eq!((p.seed(), p.input_hash()), (9, Some("h")));
        assert!(e.permute(&[3]).is_err());
        assert!(e.permute(&[]).is_err());
    }
}
