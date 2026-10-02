//! Panjer's recursion for compound distributions.

use act_core::{Error, Result};
use act_prob::{Counting, Distribution, Grid};

use crate::compound::{CompoundMethod, CompoundReport, lump_tail};

/// The distribution of `S = X_1 + … + X_N` by Panjer's recursion, on the
/// severity grid's step with `points` points.
///
/// ```text
/// g_0 = P_N(f_0)
/// g_k = 1 / (1 - a f_0) * Σ_{j=1..k} (a + b j / k) f_j g_{k-j}
/// ```
///
/// `N` must be in the `(a, b, 0)` class ([`Counting`]). The aggregate mass
/// above the last point is lumped onto it and reported. Cost is
/// `O(points × severity points)`.
///
/// Fails if `points` is 0, or if `g_0` underflows to 0 (a Poisson mean
/// above about 700 claims with no mass at zero severity); use FFT there.
///
/// # Example
///
/// ```
/// use act_aggregate::panjer;
/// use act_prob::{Distribution, Grid, Lognormal, Poisson};
///
/// let sev = Lognormal::from_mean_cv(1_000.0, 1.0).unwrap();
/// let (sev_grid, _) = Grid::local_moment(&sev, 100.0, 2_000).unwrap();
/// let (agg, report) = panjer(&Poisson::new(5.0).unwrap(), &sev_grid, 2_000).unwrap();
/// assert!((agg.mean() - 5.0 * sev_grid.mean()).abs() < 1e-6);
/// assert!(report.tail_mass < 1e-9);
/// ```
pub fn panjer<N: Counting + ?Sized>(
    frequency: &N,
    severity: &Grid,
    points: usize,
) -> Result<(Grid, CompoundReport)> {
    if points == 0 {
        return Err(Error::InvalidParameter {
            name: "points",
            value: 0.0,
            reason: "must be positive",
        });
    }
    let f = severity.probs();
    let (a, b) = frequency.panjer_ab();
    let g0 = frequency.pgf(f[0]);
    if g0 <= 0.0 || !g0.is_finite() {
        return Err(Error::InvalidParameter {
            name: "frequency",
            value: frequency.mean(),
            reason: "P(S = 0) underflows for this claim count; use FFT",
        });
    }
    let scale = 1.0 / (1.0 - a * f[0]);
    let mut g = Vec::with_capacity(points);
    g.push(g0);
    for k in 1..points {
        let kf = k as f64;
        let sum: f64 = (1..=k.min(f.len() - 1))
            .map(|j| (a + b * j as f64 / kf) * f[j] * g[k - j])
            .sum();
        g.push(sum * scale);
    }
    let (probs, tail_mass) = lump_tail(g);
    let grid = Grid::new(severity.step(), probs)?;
    let report = CompoundReport {
        method: CompoundMethod::Panjer,
        points,
        tail_mass,
        aliasing_error: 0.0,
        expected_mean: frequency.mean() * severity.mean(),
        grid_mean: grid.mean(),
    };
    Ok((grid, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use act_prob::{NegativeBinomial, Poisson};

    fn small_severity() -> Grid {
        Grid::new(1.0, vec![0.1, 0.3, 0.25, 0.2, 0.1, 0.05]).unwrap()
    }

    #[test]
    fn mean_and_variance_match_the_compound_formulas() {
        let sev = small_severity();
        let (ex, vx) = (sev.mean(), sev.variance());
        for n in [
            &Poisson::new(3.0).unwrap() as &dyn Counting,
            &NegativeBinomial::new(2.5, 1.5).unwrap(),
        ] {
            let (agg, report) = panjer(n, &sev, 400).unwrap();
            // E[S] = E[N] E[X]; Var[S] = E[N] Var[X] + Var[N] E[X]^2.
            let var = n.mean() * vx + n.variance() * ex * ex;
            assert!(report.tail_mass < 1e-14);
            assert!((agg.mean() - n.mean() * ex).abs() < 1e-10);
            assert!((agg.variance() - var).abs() < 1e-8);
            assert!(report.mean_error().abs() < 1e-10);
        }
    }

    #[test]
    fn single_point_severity_gives_the_count_distribution() {
        // X = 1 always, so S = N.
        let sev = Grid::new(1.0, vec![0.0, 1.0]).unwrap();
        let n = NegativeBinomial::new(2.0, 0.5).unwrap();
        let (agg, _) = panjer(&n, &sev, 30).unwrap();
        for k in 0..29 {
            assert!((agg.probs()[k] - n.pmf(k as u64)).abs() < 1e-15, "k {k}");
        }
    }

    #[test]
    fn short_grid_lumps_and_reports_the_tail() {
        let (agg, report) = panjer(&Poisson::new(3.0).unwrap(), &small_severity(), 8).unwrap();
        assert!((agg.probs().iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(report.tail_mass > 0.1);
        assert!(report.mean_error() < 0.0);
    }

    #[test]
    fn underflow_is_an_error_not_zeros() {
        let sev = Grid::new(1.0, vec![0.0, 1.0]).unwrap();
        assert!(panjer(&Poisson::new(800.0).unwrap(), &sev, 10).is_err());
        assert!(panjer(&Poisson::new(1.0).unwrap(), &sev, 0).is_err());
    }
}
