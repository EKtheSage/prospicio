//! Self-exciting (Hawkes) event counts for catastrophe years.
//!
//! A Hawkes process has the intensity
//!
//! ```text
//! λ(t) = μ + Σ_{t_i < t} α e^{−β (t − t_i)}
//! ```
//!
//! a baseline rate `μ` a year and an exponential excitation: each event
//! raises the rate by `α`, which decays at rate `β` a year. Each event has
//! on average `n = α / β` direct offspring (the branching ratio), so the
//! process is stationary for `n < 1`, with mean rate `μ / (1 − n)`. Events
//! cluster in time, which hours clauses, reinstatements pro rata as to time
//! and seasonal covers read, where a negative binomial count reproduces
//! the overdispersion but not the timing.
//!
//! Time is in years. [`Hawkes::simulate`] draws a path on `[0, T)` by the
//! branching construction (immigrants a Poisson process at `μ`, each event
//! a Poisson(`n`) number of children after exponential delays at `β`),
//! which is exact. [`Hawkes::event_set`] gives independent years of events
//! with their times and losses for a tower; each year starts from an empty
//! history, or from the events of `burn_in` years simulated before it and
//! dropped, which brings it near the stationary state. [`Hawkes::fit`] is
//! the maximum likelihood estimate from event times over one or more
//! periods, each starting empty.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_math::optimize::{NelderMead, nelder_mead};
use prospicio_prob::{Counting, Distribution, Poisson};

use crate::monte_carlo::EventSet;

/// A Hawkes process with an exponential kernel, time in years.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hawkes {
    baseline: f64,
    alpha: f64,
    beta: f64,
}

impl Hawkes {
    /// The process with baseline rate `baseline` a year, jump `alpha` and
    /// decay `beta` a year: `baseline` and `beta` positive, `alpha` at
    /// least 0 and the branching ratio `alpha / beta` below 1. `alpha = 0`
    /// is the Poisson process.
    ///
    /// ```
    /// use prospicio_aggregate::hawkes::Hawkes;
    ///
    /// // Two storms a year at base, each raising the rate by 12 a year,
    /// // decaying over about a month: a branching ratio of a half.
    /// let h = Hawkes::new(2.0, 12.0, 24.0).unwrap();
    /// assert_eq!(h.branching_ratio(), 0.5);
    /// assert_eq!(h.stationary_rate(), 4.0);
    /// assert!(Hawkes::new(2.0, 24.0, 24.0).is_err());
    /// ```
    pub fn new(baseline: f64, alpha: f64, beta: f64) -> Result<Self> {
        if !(baseline.is_finite() && baseline > 0.0) {
            return Err(Error::InvalidParameter {
                name: "baseline",
                value: baseline,
                reason: "must be finite and positive",
            });
        }
        if !(beta.is_finite() && beta > 0.0) {
            return Err(Error::InvalidParameter {
                name: "beta",
                value: beta,
                reason: "must be finite and positive",
            });
        }
        if !(alpha.is_finite() && alpha >= 0.0 && alpha < beta) {
            return Err(Error::InvalidParameter {
                name: "alpha",
                value: alpha,
                reason: "must be at least 0 and below beta (a branching ratio below 1)",
            });
        }
        Ok(Hawkes {
            baseline,
            alpha,
            beta,
        })
    }

    /// The baseline rate `μ`, a year.
    pub fn baseline(&self) -> f64 {
        self.baseline
    }

    /// The jump `α` in the rate at each event.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The decay rate `β`, a year.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// `n = α / β`, each event's mean number of direct offspring.
    pub fn branching_ratio(&self) -> f64 {
        self.alpha / self.beta
    }

    /// The stationary mean rate `μ / (1 − n)`, a year.
    pub fn stationary_rate(&self) -> f64 {
        self.baseline / (1.0 - self.branching_ratio())
    }

    /// The mean number of events on `[0, T)` from an empty history,
    /// `μT / (1 − n) − μ n (1 − e^{−β(1 − n)T}) / (β (1 − n)²)`; below the
    /// stationary `μT / (1 − n)` because the first events have no history
    /// to excite them.
    pub fn mean_count(&self, horizon: f64) -> f64 {
        let n = self.branching_ratio();
        let k = self.beta * (1.0 - n);
        self.stationary_rate() * horizon
            - self.baseline * n * (-(-k * horizon).exp_m1()) / (self.beta * (1.0 - n).powi(2))
    }

    /// The stationary variance of the count over a window of length `T`,
    /// `μ / (1 − n)³ · (T − n (2 − n) (1 − e^{−β(1 − n)T}) / (β (1 − n)))`
    /// — Hawkes and Oakes' result for the exponential kernel: about
    /// `μT / (1 − n)³` once `T` is many times `1 / β`.
    pub fn stationary_count_variance(&self, horizon: f64) -> f64 {
        let n = self.branching_ratio();
        let k = self.beta * (1.0 - n);
        self.baseline / (1.0 - n).powi(3)
            * (horizon - n * (2.0 - n) * (-(-k * horizon).exp_m1()) / k)
    }

    /// The event times of one path on `[0, horizon)`, sorted, from an empty
    /// history, by the branching construction.
    pub fn simulate(&self, rng: &mut StreamRng, horizon: f64) -> Result<Vec<f64>> {
        if !(horizon.is_finite() && horizon > 0.0) {
            return Err(Error::InvalidParameter {
                name: "horizon",
                value: horizon,
                reason: "must be finite and positive",
            });
        }
        let immigrants = poisson(self.baseline * horizon, rng)?;
        let mut times: Vec<f64> = (0..immigrants)
            .map(|_| horizon * rng.next_open01())
            .collect();
        let n = self.branching_ratio();
        let mut next = 0;
        while next < times.len() {
            let parent = times[next];
            next += 1;
            if n == 0.0 {
                continue;
            }
            for _ in 0..poisson(n, rng)? {
                let t = parent - rng.next_open01().ln() / self.beta;
                if t < horizon {
                    times.push(t);
                }
            }
        }
        times.sort_by(f64::total_cmp);
        Ok(times)
    }

    /// The times of one year, `[0, 1)`, after `burn_in` years whose events
    /// excite it and are then dropped.
    fn year(&self, rng: &mut StreamRng, burn_in: f64) -> Result<Vec<f64>> {
        let path = self.simulate(rng, 1.0 + burn_in)?;
        Ok(path
            .into_iter()
            .filter(|&t| t >= burn_in)
            .map(|t| t - burn_in)
            .collect())
    }

    /// The number of events in each of `n_years` independent years, year
    /// `i` from stream `i` of `seed`, after `burn_in` years (0 for an empty
    /// history); the same draws as [`event_set`](Self::event_set)'s years.
    pub fn annual_counts(&self, n_years: usize, burn_in: f64, seed: u64) -> Result<Vec<u64>> {
        check_burn_in(burn_in)?;
        (0..n_years as u64)
            .map(|i| Ok(self.year(&mut StreamRng::new(seed, i), burn_in)?.len() as u64))
            .collect()
    }

    /// `n_years` independent years of events, year `i` from stream `i` of
    /// `seed`, with their times in `[0, 1)` and a loss from `severity` each,
    /// drawn after the year's times from the same stream.
    ///
    /// ```
    /// use prospicio_aggregate::hawkes::Hawkes;
    /// use prospicio_prob::Lognormal;
    ///
    /// let h = Hawkes::new(2.0, 12.0, 24.0).unwrap();
    /// let sev = Lognormal::from_mean_cv(10.0, 1.0).unwrap();
    /// let events = h.event_set(&sev, 1_000, 0.0, 7).unwrap();
    /// assert_eq!(events.n_sims(), 1_000);
    /// let times = events.times(0).unwrap();
    /// assert!(times.windows(2).all(|w| w[0] <= w[1]));
    /// ```
    pub fn event_set<X>(
        &self,
        severity: &X,
        n_years: usize,
        burn_in: f64,
        seed: u64,
    ) -> Result<EventSet>
    where
        X: Distribution + ?Sized,
    {
        check_burn_in(burn_in)?;
        if n_years == 0 {
            return Err(Error::InvalidParameter {
                name: "n_years",
                value: 0.0,
                reason: "must be positive",
            });
        }
        let mut years = Vec::with_capacity(n_years);
        let mut times = Vec::new();
        for i in 0..n_years as u64 {
            let mut rng = StreamRng::new(seed, i);
            let t = self.year(&mut rng, burn_in)?;
            let losses = t
                .iter()
                .map(|_| severity.quantile(rng.next_open01()))
                .collect::<Result<Vec<f64>>>()?;
            times.extend(t);
            years.push(losses);
        }
        EventSet::from_years(years, seed)?.with_times(times)
    }

    /// The log-likelihood of event times, each period a sorted list of
    /// times on `[0, horizon)` from an empty history.
    pub fn log_likelihood(&self, periods: &[Vec<f64>], horizon: f64) -> f64 {
        log_likelihood(self.baseline, self.alpha, self.beta, periods, horizon)
    }

    /// The maximum likelihood estimate from event times over one or more
    /// periods of length `horizon` years, each sorted and on
    /// `[0, horizon)` from an empty history (for years, the year's times
    /// and `horizon` 1).
    ///
    /// The log-likelihood
    /// `Σ log λ(t_i) − μT − (α / β) Σ (1 − e^{−β(T − t_i)})` is computed by
    /// the usual recursion in `O(n)` a period and maximised over
    /// `(log μ, log α, log β)` by Nelder–Mead from the method of moments'
    /// start, restarted at the optimum until it stops improving. Standard
    /// errors are from the observed information, the Hessian of the
    /// log-likelihood in `(μ, α, β)` by central differences. A fit with a
    /// branching ratio at or above 1 is returned with
    /// [`HawkesFit::process`] an error.
    pub fn fit(periods: &[Vec<f64>], horizon: f64) -> Result<HawkesFit> {
        if !(horizon.is_finite() && horizon > 0.0) {
            return Err(Error::InvalidParameter {
                name: "horizon",
                value: horizon,
                reason: "must be finite and positive",
            });
        }
        let count: usize = periods.iter().map(Vec::len).sum();
        if count < 3 {
            return Err(Error::Data(format!(
                "a Hawkes fit needs at least 3 events, not {count}"
            )));
        }
        for p in periods {
            if let Some(&t) = p
                .iter()
                .find(|t| !(t.is_finite() && **t >= 0.0 && **t < horizon))
            {
                return Err(Error::InvalidParameter {
                    name: "times",
                    value: t,
                    reason: "must be in [0, horizon)",
                });
            }
            if p.windows(2).any(|w| w[1] < w[0]) {
                return Err(Error::Data("each period's times must be sorted".into()));
            }
        }
        let span = horizon * periods.len() as f64;
        let rate = count as f64 / span;
        // Start at half the events immigrants, decaying over a tenth of
        // the mean gap between events times ten.
        let beta0 = 10.0 * rate;
        let x0 = [(0.5 * rate).ln(), (0.5 * beta0).ln(), beta0.ln()];
        let f = |x: &[f64]| -> f64 {
            -log_likelihood(x[0].exp(), x[1].exp(), x[2].exp(), periods, horizon)
        };
        let options = NelderMead {
            tolerance: 1e-12,
            max_iterations: 20_000,
            ..NelderMead::default()
        };
        let mut best = nelder_mead(f, &x0, &options);
        for _ in 0..20 {
            let again = nelder_mead(f, &best.x, &options);
            let gain = best.value - again.value;
            best = again;
            if gain <= 1e-10 * (1.0 + best.value.abs()) {
                break;
            }
        }
        let theta = [best.x[0].exp(), best.x[1].exp(), best.x[2].exp()];
        let ll = -best.value;
        let std_errors = standard_errors(&theta, periods, horizon);
        Ok(HawkesFit {
            baseline: theta[0],
            alpha: theta[1],
            beta: theta[2],
            log_likelihood: ll,
            std_errors,
            n_events: count,
            converged: best.converged,
        })
    }
}

/// A fitted Hawkes process ([`Hawkes::fit`]).
#[derive(Clone, Debug, PartialEq)]
pub struct HawkesFit {
    /// `μ`, a year.
    pub baseline: f64,
    /// `α`.
    pub alpha: f64,
    /// `β`, a year.
    pub beta: f64,
    /// The maximised log-likelihood.
    pub log_likelihood: f64,
    /// Standard errors of `(μ, α, β)` from the observed information; NaN
    /// where the information is not positive definite.
    pub std_errors: [f64; 3],
    /// Events fitted.
    pub n_events: usize,
    /// Whether the last simplex pass met its tolerance.
    pub converged: bool,
}

impl HawkesFit {
    /// The branching ratio `α / β`.
    pub fn branching_ratio(&self) -> f64 {
        self.alpha / self.beta
    }

    /// `2k − 2ℓ` with `k = 3` parameters.
    pub fn aic(&self) -> f64 {
        6.0 - 2.0 * self.log_likelihood
    }

    /// The fitted process; an error when the branching ratio is at or
    /// above 1 (the data are not stationary at these parameters).
    pub fn process(&self) -> Result<Hawkes> {
        Hawkes::new(self.baseline, self.alpha, self.beta)
    }
}

fn check_burn_in(burn_in: f64) -> Result<()> {
    if burn_in.is_finite() && burn_in >= 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name: "burn_in",
            value: burn_in,
            reason: "must be finite and at least 0",
        })
    }
}

fn poisson(mean: f64, rng: &mut StreamRng) -> Result<u64> {
    if mean == 0.0 {
        return Ok(0);
    }
    Poisson::new(mean)?.quantile(rng.next_open01())
}

fn log_likelihood(mu: f64, alpha: f64, beta: f64, periods: &[Vec<f64>], horizon: f64) -> f64 {
    let mut ll = -mu * horizon * periods.len() as f64;
    for p in periods {
        let mut a = 0.0;
        let mut last = 0.0;
        for (i, &t) in p.iter().enumerate() {
            if i > 0 {
                a = (-beta * (t - last)).exp() * (1.0 + a);
            }
            last = t;
            ll += (mu + alpha * a).ln();
            ll -= alpha / beta * (-(-beta * (horizon - t)).exp_m1());
        }
    }
    ll
}

/// The square roots of the inverse observed information's diagonal.
fn standard_errors(theta: &[f64; 3], periods: &[Vec<f64>], horizon: f64) -> [f64; 3] {
    let ll = |x: &[f64; 3]| log_likelihood(x[0], x[1], x[2], periods, horizon);
    let h: [f64; 3] = theta.map(|v| 1e-4 * v);
    let mut info = [0.0; 9];
    for i in 0..3 {
        for j in i..3 {
            let at = |si: f64, sj: f64| {
                let mut x = *theta;
                x[i] += si * h[i];
                x[j] += sj * h[j];
                ll(&x)
            };
            let d = if i == j {
                (at(1.0, 0.0) - 2.0 * ll(theta) + at(-1.0, 0.0)) / (h[i] * h[i])
            } else {
                (at(1.0, 1.0) - at(1.0, -1.0) - at(-1.0, 1.0) + at(-1.0, -1.0))
                    / (4.0 * h[i] * h[j])
            };
            info[i * 3 + j] = -d;
            info[j * 3 + i] = -d;
        }
    }
    match prospicio_math::linalg::cholesky(&info, 3) {
        Some(l) => {
            let inv = prospicio_math::linalg::cholesky_inverse(&l, 3);
            [inv[0].sqrt(), inv[4].sqrt(), inv[8].sqrt()]
        }
        None => [f64::NAN; 3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::Lognormal;

    #[test]
    fn the_mean_and_variance_match_the_closed_forms() {
        let h = Hawkes::new(2.0, 12.0, 24.0).unwrap();
        let counts = h.annual_counts(40_000, 0.0, 3).unwrap();
        let m = counts.iter().sum::<u64>() as f64 / counts.len() as f64;
        let want = h.mean_count(1.0);
        let var =
            counts.iter().map(|&c| (c as f64 - m).powi(2)).sum::<f64>() / (counts.len() - 1) as f64;
        // Within four standard errors of the exact mean from an empty start.
        assert!(
            (m - want).abs() < 4.0 * (var / counts.len() as f64).sqrt(),
            "{m} {want}"
        );
        assert!(want < 4.0 && want > 3.8);
        // A burn-in of two years starts each year near stationarity.
        let warm = h.annual_counts(40_000, 2.0, 3).unwrap();
        let mw = warm.iter().sum::<u64>() as f64 / warm.len() as f64;
        let vw =
            warm.iter().map(|&c| (c as f64 - mw).powi(2)).sum::<f64>() / (warm.len() - 1) as f64;
        assert!(
            (mw - 4.0).abs() < 4.0 * (vw / warm.len() as f64).sqrt(),
            "{mw}"
        );
        let sv = h.stationary_count_variance(1.0);
        assert!((vw / sv - 1.0).abs() < 0.05, "{vw} {sv}");
        // Overdispersed against the Poisson.
        assert!(vw > 1.5 * mw);
        // A slow decay, where the window correction is about a third of
        // the variance, after a long burn-in.
        let slow = Hawkes::new(2.0, 2.0, 4.0).unwrap();
        let c = slow.annual_counts(60_000, 8.0, 4).unwrap();
        let m = c.iter().sum::<u64>() as f64 / c.len() as f64;
        let v = c.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / (c.len() - 1) as f64;
        let sv = slow.stationary_count_variance(1.0);
        assert!((sv / (2.0 / 0.125) - 1.0).abs() > 0.3);
        assert!((v / sv - 1.0).abs() < 0.03, "{v} {sv}");
    }

    #[test]
    fn no_excitation_is_the_poisson_process() {
        let h = Hawkes::new(3.0, 0.0, 1.0).unwrap();
        let mut rng = StreamRng::new(5, 0);
        let t = h.simulate(&mut rng, 1.0).unwrap();
        // The immigrants alone: the Poisson count and uniform times.
        let mut again = StreamRng::new(5, 0);
        let k = poisson(3.0, &mut again).unwrap() as usize;
        assert_eq!(t.len(), k);
        assert_eq!(h.mean_count(2.0), 6.0);
        let counts = h.annual_counts(20_000, 0.0, 1).unwrap();
        let m = counts.iter().sum::<u64>() as f64 / 20_000.0;
        assert!((m - 3.0).abs() < 0.05);
    }

    #[test]
    fn the_fit_recovers_simulated_parameters() {
        let h = Hawkes::new(1.5, 6.0, 10.0).unwrap();
        let periods: Vec<Vec<f64>> = (0..400)
            .map(|i| h.simulate(&mut StreamRng::new(11, i), 1.0).unwrap())
            .collect();
        let fit = Hawkes::fit(&periods, 1.0).unwrap();
        let truth = [1.5, 6.0, 10.0];
        let got = [fit.baseline, fit.alpha, fit.beta];
        for k in 0..3 {
            assert!(
                (got[k] - truth[k]).abs() < 3.0 * fit.std_errors[k],
                "{k}: {} ± {} for {}",
                got[k],
                fit.std_errors[k],
                truth[k]
            );
        }
        // The optimum: no nearby point does better.
        for d in [-1e-3, 1e-3] {
            for k in 0..3 {
                let mut x = got;
                x[k] *= 1.0 + d;
                assert!(log_likelihood(x[0], x[1], x[2], &periods, 1.0) <= fit.log_likelihood);
            }
        }
        assert!(fit.process().is_ok());
        assert_eq!(fit.aic(), 6.0 - 2.0 * fit.log_likelihood);
    }

    #[test]
    fn an_event_set_carries_the_times_and_draws_losses() {
        let h = Hawkes::new(2.0, 12.0, 24.0).unwrap();
        let sev = Lognormal::from_mean_cv(10.0, 1.0).unwrap();
        let ev = h.event_set(&sev, 500, 0.5, 9).unwrap();
        let counts = h.annual_counts(500, 0.5, 9).unwrap();
        for (i, &c) in counts.iter().enumerate() {
            assert_eq!(ev.events(i).len() as u64, c);
            assert!(ev.times(i).unwrap().iter().all(|t| (0.0..1.0).contains(t)));
        }
        assert_eq!(
            h.event_set(&sev, 500, 0.5, 9).unwrap().events(7),
            ev.events(7)
        );
        assert!(h.event_set(&sev, 0, 0.0, 9).is_err());
        assert!(h.annual_counts(5, -1.0, 9).is_err());
        assert!(Hawkes::fit(&[vec![0.5]], 1.0).is_err());
        assert!(Hawkes::fit(&[vec![0.5, 0.2, 0.7]], 1.0).is_err());
        assert!(Hawkes::fit(&[vec![0.2, 0.5, 1.5]], 1.0).is_err());
    }
}
