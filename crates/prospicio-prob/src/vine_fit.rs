//! Fitting pair copulas to data (#568): pseudo-observations, Kendall's
//! tau, the maximum likelihood fit of one family and rotation, and the
//! choice among families by an information criterion, with an
//! independence test first, as pyvinecopulib's `Bicop.select` does.
//!
//! Parameters are searched within pyvinecopulib's bounds: the Gaussian and
//! Student t correlation in `(−1, 1)`, Clayton in `(0, 28]`, Gumbel in
//! `[1, 50]`, Frank in `[−35, 35]`, Joe in `[1, 30]` and the Student t's
//! degrees of freedom in `[2, 50]`. Clayton, Gumbel and Joe are tried in
//! the rotations that match the sign of Kendall's tau (0 and 180 degrees
//! for a positive tau, 90 and 270 for a negative one).

use prospicio_core::{Error, Result};
use prospicio_math::optimize::minimize;
use prospicio_math::special::norm_cdf;

use crate::vine::{PairCopula, PairFamily};

/// Pseudo-observations of a sample: each value's average rank over
/// `n + 1`, so ties share a rank and every value lies in `(0, 1)`.
///
/// ```
/// use prospicio_prob::vine_fit::pseudo_observations;
///
/// assert_eq!(pseudo_observations(&[3.0, 1.0, 2.0, 2.0]).unwrap(), [0.8, 0.2, 0.5, 0.5]);
/// ```
pub fn pseudo_observations(x: &[f64]) -> Result<Vec<f64>> {
    if x.is_empty() {
        return Err(Error::Data("no observations".into()));
    }
    if let Some(&bad) = x.iter().find(|v| !v.is_finite()) {
        return Err(Error::InvalidParameter {
            name: "x",
            value: bad,
            reason: "must be finite",
        });
    }
    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut out = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[order[j + 1]] == x[order[i]] {
            j += 1;
        }
        // Ranks i + 1 to j + 1 share their average.
        let rank = 0.5 * ((i + 1) + (j + 1)) as f64;
        for &k in &order[i..=j] {
            out[k] = rank / (n as f64 + 1.0);
        }
        i = j + 1;
    }
    Ok(out)
}

/// Kendall's tau-b of two samples, by Knight's `O(n log n)` algorithm,
/// ties counted as tau-b counts them (`scipy.stats.kendalltau`).
///
/// ```
/// use prospicio_prob::vine_fit::kendall_tau;
///
/// // Pairs (1,1), (2,3), (3,2), (4,4): 5 concordant, 1 discordant.
/// let t = kendall_tau(&[1.0, 2.0, 3.0, 4.0], &[1.0, 3.0, 2.0, 4.0]).unwrap();
/// assert!((t - 4.0 / 6.0).abs() < 1e-15);
/// ```
pub fn kendall_tau(x: &[f64], y: &[f64]) -> Result<f64> {
    let n = x.len();
    if n < 2 || y.len() != n {
        return Err(Error::Data(format!(
            "Kendall's tau needs two samples of the same length, at least 2; got {} and {}",
            n,
            y.len()
        )));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(y[a].total_cmp(&y[b])));
    let pairs = |runs: &mut dyn Iterator<Item = usize>| -> f64 {
        runs.map(|k| (k * (k - 1) / 2) as f64).sum()
    };
    // Ties in x, and in both.
    let (mut tx, mut txy) = (0.0, 0.0);
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[order[j + 1]] == x[order[i]] {
            j += 1;
        }
        tx += pairs(&mut std::iter::once(j - i + 1));
        let mut k = i;
        while k <= j {
            let mut m = k;
            while m < j && y[order[m + 1]] == y[order[k]] {
                m += 1;
            }
            txy += pairs(&mut std::iter::once(m - k + 1));
            k = m + 1;
        }
        i = j + 1;
    }
    // Sort the y's in x order by merge sort, counting the swaps.
    let mut ys: Vec<f64> = order.iter().map(|&k| y[k]).collect();
    let mut buf = vec![0.0; n];
    let swaps = merge_count(&mut ys, &mut buf);
    let mut ty = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && ys[j + 1] == ys[i] {
            j += 1;
        }
        ty += pairs(&mut std::iter::once(j - i + 1));
        i = j + 1;
    }
    let n0 = (n * (n - 1) / 2) as f64;
    let concordant_minus_discordant = n0 - tx - ty + txy - 2.0 * swaps as f64;
    let denom = ((n0 - tx) * (n0 - ty)).sqrt();
    if denom == 0.0 {
        return Err(Error::Data(
            "a sample is constant; Kendall's tau is undefined".into(),
        ));
    }
    Ok(concordant_minus_discordant / denom)
}

/// Sorts `a` ascending and returns the number of inversions.
fn merge_count(a: &mut [f64], buf: &mut [f64]) -> u64 {
    let n = a.len();
    if n < 2 {
        return 0;
    }
    let m = n / 2;
    let (mut s, (left, right)) = (0, a.split_at_mut(m));
    s += merge_count(left, &mut buf[..m]);
    s += merge_count(right, &mut buf[m..]);
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            buf[k] = left[i];
            i += 1;
        } else {
            buf[k] = right[j];
            s += (left.len() - i) as u64;
            j += 1;
        }
        k += 1;
    }
    while i < left.len() {
        buf[k] = left[i];
        i += 1;
        k += 1;
    }
    while j < right.len() {
        buf[k] = right[j];
        j += 1;
        k += 1;
    }
    a.copy_from_slice(&buf[..n]);
    s
}

/// The criterion that chooses among families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// Akaike's, `2k − 2ℓ`.
    Aic,
    /// Schwarz's, `k ln n − 2ℓ`.
    Bic,
}

/// A fitted pair copula.
#[derive(Debug, Clone, PartialEq)]
pub struct PairFit {
    /// The pair copula at its estimate.
    pub pair: PairCopula,
    /// The maximised log-likelihood `Σ ln c(uᵢ, vᵢ)`.
    pub log_likelihood: f64,
    /// Its number of parameters: 0 for independence, 2 for the Student t,
    /// else 1.
    pub n_parameters: usize,
    /// The number of observations.
    pub n: usize,
}

impl PairFit {
    /// `2k − 2ℓ`.
    pub fn aic(&self) -> f64 {
        2.0 * self.n_parameters as f64 - 2.0 * self.log_likelihood
    }

    /// `k ln n − 2ℓ`.
    pub fn bic(&self) -> f64 {
        self.n_parameters as f64 * (self.n as f64).ln() - 2.0 * self.log_likelihood
    }

    fn score(&self, c: Criterion) -> f64 {
        match c {
            Criterion::Aic => self.aic(),
            Criterion::Bic => self.bic(),
        }
    }
}

/// `Σ ln c(uᵢ, vᵢ)`.
pub fn log_likelihood(pair: &PairCopula, u: &[f64], v: &[f64]) -> f64 {
    u.iter().zip(v).map(|(&a, &b)| pair.log_density(a, b)).sum()
}

fn check_data(u: &[f64], v: &[f64]) -> Result<()> {
    if u.len() != v.len() || u.len() < 2 {
        return Err(Error::Data(format!(
            "give two columns of the same length, at least 2; got {} and {}",
            u.len(),
            v.len()
        )));
    }
    if let Some(&bad) = u.iter().chain(v).find(|x| !(**x > 0.0 && **x < 1.0)) {
        return Err(Error::InvalidParameter {
            name: "u",
            value: bad,
            reason: "must be in (0, 1); take pseudo-observations",
        });
    }
    Ok(())
}

/// The maximum likelihood fit of `family` in `rotation` to the
/// observations `(u, v)`, each in `(0, 1)`.
///
/// ```
/// use prospicio_prob::vine::{PairCopula, PairFamily};
/// use prospicio_prob::vine_fit::fit_pair;
///
/// // Draws from a Clayton with θ = 2 by inverting h₂.
/// let c = PairCopula::new(PairFamily::Clayton, 2.0, 0).unwrap();
/// let n = 2_000;
/// let v: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
/// let u: Vec<f64> = (0..n).map(|i| c.h2_inverse(((i * 7919) % n) as f64 / n as f64 + 0.25 / n as f64, v[i])).collect();
/// let fit = fit_pair(PairFamily::Clayton, 0, &u, &v).unwrap();
/// assert!((fit.pair.parameter() - 2.0).abs() < 0.2);
/// ```
pub fn fit_pair(family: PairFamily, rotation: u16, u: &[f64], v: &[f64]) -> Result<PairFit> {
    check_data(u, v)?;
    let n = u.len();
    let one = |lo: f64, hi: f64| -> Result<(PairCopula, f64)> {
        let nll = |th: f64| -> f64 {
            match PairCopula::new(family, th, rotation) {
                Ok(c) => -log_likelihood(&c, u, v),
                Err(_) => f64::INFINITY,
            }
        };
        let th = minimize(lo, hi, nll);
        let c = PairCopula::new(family, th, rotation)?;
        let ll = log_likelihood(&c, u, v);
        Ok((c, ll))
    };
    let (pair, ll, k) = match family {
        PairFamily::Independence => (PairCopula::independence(), 0.0, 0),
        PairFamily::Gaussian => {
            let (c, ll) = one(-0.9999, 0.9999)?;
            (c, ll, 1)
        }
        PairFamily::Clayton => {
            let (c, ll) = one(1e-10, 28.0)?;
            (c, ll, 1)
        }
        PairFamily::Gumbel => {
            let (c, ll) = one(1.0, 50.0)?;
            (c, ll, 1)
        }
        PairFamily::Joe => {
            let (c, ll) = one(1.0, 30.0)?;
            (c, ll, 1)
        }
        PairFamily::Frank => {
            // Frank is not defined at 0; search the side of Kendall's tau.
            let tau = kendall_tau(u, v)?;
            let (lo, hi) = if tau >= 0.0 {
                (1e-10, 35.0)
            } else {
                (-35.0, -1e-10)
            };
            let (c, ll) = one(lo, hi)?;
            (c, ll, 1)
        }
        PairFamily::Student => {
            let (c, ll) = fit_student(u, v)?;
            (c, ll, 2)
        }
    };
    Ok(PairFit {
        pair,
        log_likelihood: ll,
        n_parameters: k,
        n,
    })
}

/// The Student t by its profile likelihood: for each `ν` the quantiles of
/// the data once and `ρ` maximised on them, `ν` searched by Brent's method
/// on `ln ν` in `[ln 2, ln 50]` after a short scan.
fn fit_student(u: &[f64], v: &[f64]) -> Result<(PairCopula, f64)> {
    use prospicio_math::special::{ln_gamma, student_t_quantile};
    // The profile at ν: the best ρ and its log-likelihood.
    let profile = |nu: f64| -> (f64, f64) {
        let x: Vec<f64> = u.iter().map(|&a| student_t_quantile(a, nu)).collect();
        let y: Vec<f64> = v.iter().map(|&b| student_t_quantile(b, nu)).collect();
        let k = ln_gamma(0.5 * (nu + 2.0)) + ln_gamma(0.5 * nu) - 2.0 * ln_gamma(0.5 * (nu + 1.0));
        let margins: f64 = x
            .iter()
            .zip(&y)
            .map(|(a, b)| 0.5 * (nu + 1.0) * ((a * a / nu).ln_1p() + (b * b / nu).ln_1p()))
            .sum();
        let ll = |rho: f64| -> f64 {
            let s = 1.0 - rho * rho;
            let core: f64 = x
                .iter()
                .zip(&y)
                .map(|(a, b)| {
                    (0.5 * (nu + 2.0)) * ((a * a - 2.0 * rho * a * b + b * b) / (nu * s)).ln_1p()
                })
                .sum();
            x.len() as f64 * (k - 0.5 * s.ln()) - core + margins
        };
        let rho = prospicio_math::optimize::brent(-0.9999, 0.9999, 1e-10, |r| -ll(r)).0;
        (rho, ll(rho))
    };
    let (lo, hi) = (2f64.ln(), 50f64.ln());
    // A scan of nine points picks the bracket.
    let grid: Vec<f64> = (0..9).map(|k| lo + (hi - lo) * k as f64 / 8.0).collect();
    let values: Vec<f64> = grid.iter().map(|&g| profile(g.exp()).1).collect();
    let best = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("nine points");
    let (a, b) = (grid[best.saturating_sub(1)], grid[(best + 1).min(8)]);
    let t = prospicio_math::optimize::brent(a, b, 1e-7, |g| -profile(g.exp()).1).0;
    let nu = t.exp().clamp(2.0, 50.0);
    let (rho, _) = profile(nu);
    let c = PairCopula::student(rho, nu)?;
    Ok((c, log_likelihood(&c, u, v)))
}

/// How [`select_pair`] chooses.
#[derive(Debug, Clone, PartialEq)]
pub struct PairSelect {
    /// The families to try; independence is always among them.
    pub families: Vec<PairFamily>,
    /// The criterion.
    pub criterion: Criterion,
    /// A level for Kendall's test of independence: when its p-value is
    /// above it, the pair is independence without a fit; `None` to fit
    /// every family.
    pub independence_test: Option<f64>,
}

impl Default for PairSelect {
    /// Every family, AIC, no independence test (pyvinecopulib's defaults
    /// over these families).
    fn default() -> Self {
        PairSelect {
            families: vec![
                PairFamily::Gaussian,
                PairFamily::Student,
                PairFamily::Clayton,
                PairFamily::Gumbel,
                PairFamily::Frank,
                PairFamily::Joe,
            ],
            criterion: Criterion::Aic,
            independence_test: None,
        }
    }
}

/// The p-value of Kendall's test of independence,
/// `z = 3τ √(n(n − 1)) / √(2(2n + 5))` against the normal.
pub fn independence_p_value(u: &[f64], v: &[f64]) -> Result<f64> {
    let tau = kendall_tau(u, v)?;
    let n = u.len() as f64;
    let z = 3.0 * tau * (n * (n - 1.0)).sqrt() / (2.0 * (2.0 * n + 5.0)).sqrt();
    Ok(2.0 * (1.0 - norm_cdf(z.abs())))
}

/// The best pair copula by `select.criterion` over its families (each
/// Clayton, Gumbel and Joe in the two rotations that match the sign of
/// Kendall's tau), and independence.
pub fn select_pair(u: &[f64], v: &[f64], select: &PairSelect) -> Result<PairFit> {
    check_data(u, v)?;
    let independence = PairFit {
        pair: PairCopula::independence(),
        log_likelihood: 0.0,
        n_parameters: 0,
        n: u.len(),
    };
    if let Some(level) = select.independence_test
        && independence_p_value(u, v)? > level
    {
        return Ok(independence);
    }
    let tau = kendall_tau(u, v)?;
    let rotations: [u16; 2] = if tau >= 0.0 { [0, 180] } else { [90, 270] };
    let mut best = independence;
    for &family in &select.families {
        let tries: Vec<u16> = match family {
            PairFamily::Clayton | PairFamily::Gumbel | PairFamily::Joe => rotations.to_vec(),
            PairFamily::Independence => continue,
            _ => vec![0],
        };
        for r in tries {
            let fit = fit_pair(family, r, u, v)?;
            if fit.score(select.criterion) < best.score(select.criterion) {
                best = fit;
            }
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kendall_tau_counts_ties_as_tau_b() {
        // Brute force over every pair with ties.
        let x: [f64; 8] = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 1.0];
        let y: [f64; 8] = [2.0, 1.0, 3.0, 3.0, 5.0, 4.0, 4.0, 2.0];
        // f64::signum gives 1 at 0.
        let sign = |d: f64| {
            if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            }
        };
        let (mut c, mut d, mut tx, mut ty) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in 0..x.len() {
            for j in i + 1..x.len() {
                let (a, b) = (sign(x[i] - x[j]), sign(y[i] - y[j]));
                if a == 0.0 && b == 0.0 {
                    continue;
                }
                if a == 0.0 {
                    tx += 1.0;
                } else if b == 0.0 {
                    ty += 1.0;
                } else if a == b {
                    c += 1.0;
                } else {
                    d += 1.0;
                }
            }
        }
        let want = (c - d) / ((c + d + tx) * (c + d + ty)).sqrt();
        assert!((kendall_tau(&x, &y).unwrap() - want).abs() < 1e-15);
        assert!(kendall_tau(&[1.0, 1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn the_fit_recovers_each_family() {
        let n = 3_000;
        let draws = |c: &PairCopula| -> (Vec<f64>, Vec<f64>) {
            let mut rng = prospicio_core::StreamRng::new(17, 0);
            let mut u = Vec::with_capacity(n);
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let b = rng.next_open01();
                let p = rng.next_open01();
                u.push(c.h2_inverse(p, b));
                v.push(b);
            }
            (u, v)
        };
        for (c, tol) in [
            (PairCopula::new(PairFamily::Gaussian, 0.6, 0).unwrap(), 0.03),
            (PairCopula::new(PairFamily::Clayton, 2.0, 0).unwrap(), 0.15),
            (PairCopula::new(PairFamily::Gumbel, 1.8, 180).unwrap(), 0.1),
            (PairCopula::new(PairFamily::Frank, -5.0, 0).unwrap(), 0.4),
            (PairCopula::new(PairFamily::Joe, 2.0, 90).unwrap(), 0.15),
        ] {
            let (u, v) = draws(&c);
            let fit = fit_pair(c.family(), c.rotation(), &u, &v).unwrap();
            assert!(
                (fit.pair.parameter() - c.parameter()).abs() < tol,
                "{c:?} {fit:?}"
            );
            // The optimum: nearby parameters do no better.
            for d in [-1e-3, 1e-3] {
                if let Ok(q) = PairCopula::new(c.family(), fit.pair.parameter() + d, c.rotation()) {
                    assert!(log_likelihood(&q, &u, &v) <= fit.log_likelihood + 1e-9);
                }
            }
            // Selection is at least as good as the true family by AIC (on
            // 3,000 Clayton draws pyvinecopulib, too, prefers a rotated Joe).
            let s = select_pair(&u, &v, &PairSelect::default()).unwrap();
            assert!(s.aic() <= fit.aic() + 1e-9, "{s:?}");
        }
        let t = PairCopula::student(0.5, 5.0).unwrap();
        let (u, v) = draws(&t);
        let fit = fit_pair(PairFamily::Student, 0, &u, &v).unwrap();
        assert!((fit.pair.parameter() - 0.5).abs() < 0.04, "{fit:?}");
        assert!((fit.pair.nu().unwrap() - 5.0).abs() < 2.0, "{fit:?}");
        let s = select_pair(&u, &v, &PairSelect::default()).unwrap();
        assert_eq!(s.pair.family(), PairFamily::Student, "{s:?}");
        // Independent draws pass the independence test.
        let mut rng = prospicio_core::StreamRng::new(3, 1);
        let (a, b): (Vec<f64>, Vec<f64>) = (0..500)
            .map(|_| (rng.next_open01(), rng.next_open01()))
            .unzip();
        let sel = PairSelect {
            independence_test: Some(0.05),
            ..PairSelect::default()
        };
        assert_eq!(
            select_pair(&a, &b, &sel).unwrap().pair.family(),
            PairFamily::Independence
        );
    }
}
