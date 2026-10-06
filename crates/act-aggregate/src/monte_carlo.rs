//! Monte Carlo frequency-severity: simulated years of individual losses.

use act_core::{Error, Result, StreamRng};
use act_prob::provenance::SIM_INDEX_SCHEME;
use act_prob::{Counting, Distribution, PredictiveDistribution, Provenance};
use rayon::prelude::*;

/// Simulated years of individual losses (events), kept so that per-event
/// terms such as excess-of-loss layers can be applied to them.
///
/// Year `i` was drawn from `StreamRng::new(seed, i)` in a fixed order:
/// first the claim count, then each claim's severity, both by inverse
/// transform. So results are identical for any number of threads, and any
/// year can be replayed alone.
#[derive(Debug, Clone, PartialEq)]
pub struct EventSet {
    /// `offsets[i]..offsets[i + 1]` indexes year `i`'s losses.
    offsets: Vec<usize>,
    losses: Vec<f64>,
    seed: u64,
}

/// Simulates `n_sims` years of claims: a count from `frequency`, then that
/// many independent losses from `severity`.
///
/// # Example
///
/// ```
/// use act_aggregate::simulate_events;
/// use act_prob::{Distribution, Lognormal, Poisson};
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
        seed,
    })
}

impl EventSet {
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
                .version("act-aggregate", env!("CARGO_PKG_VERSION"))
                .seed(self.seed, SIM_INDEX_SCHEME),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft;
    use act_prob::{Grid, NegativeBinomial, Poisson};

    fn severity() -> Grid {
        Grid::new(1.0, vec![0.1, 0.3, 0.25, 0.2, 0.1, 0.05]).unwrap()
    }

    #[test]
    fn a_serial_severity_gives_the_same_draws() {
        use act_prob::Custom;
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
    fn rejects_zero_sims() {
        assert!(simulate_events(&Poisson::new(1.0).unwrap(), &severity(), 0, 1).is_err());
    }
}
