//! Monte Carlo frequency-severity: simulated years of individual losses.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_prob::provenance::SIM_INDEX_SCHEME;
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
/// rata as to time need.
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
    seed: u64,
}

/// Year `i`'s event times come from stream `TIME_STREAM + i`, apart from
/// the streams the years' losses were drawn from.
const TIME_STREAM: u64 = 1 << 63;

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
        seed,
    })
}

impl EventSet {
    /// Years of losses from elsewhere (your own simulation, or a
    /// catastrophe model's event loss table by year), in order within each
    /// year. `seed` is recorded in results' provenance; give the one your
    /// simulation used, or 0.
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
            seed,
        })
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
    /// (no dimensions), with the seed and stream scheme in its provenance.
    pub fn totals(&self) -> Result<PredictiveDistribution> {
        let totals = (0..self.n_sims())
            .map(|i| self.events(i).iter().sum())
            .collect();
        PredictiveDistribution::from_draws(
            vec![],
            vec![vec![]],
            totals,
            Provenance::new("frequency_severity_monte_carlo")
                .version("prospicio-aggregate", env!("CARGO_PKG_VERSION"))
                .seed(self.seed, SIM_INDEX_SCHEME),
        )
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(
            events.counts().iter().sum::<usize>(),
            (0..10).map(|i| events.events(i).len()).sum::<usize>()
        );
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
}
