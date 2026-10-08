//! Panjer's recursion for compound distributions.

use prospicio_core::{Error, Result};
use prospicio_prob::{Counting, Distribution, Grid};

use crate::compound::{CompoundMethod, CompoundReport, lump_tail};

/// The distribution of `S = X_1 + … + X_N` by Panjer's recursion, on the
/// severity grid's step with `points` points.
///
/// ```text
/// g_0 = P_N(f_0)
/// g_k = 1 / (1 - a f_0) * [(p_1 - (a + b) p_0) f_k + Σ_{j=1..k} (a + b j / k) f_j g_{k-j}]
/// ```
///
/// `N` must be in the `(a, b, 1)` class ([`Counting::panjer_ab`]), which
/// holds the `(a, b, 0)` class (where `p_1 = (a + b) p_0`, so the first
/// term vanishes) and the zero-modified and zero-truncated counts. The
/// aggregate mass
/// above the last point is lumped onto it and reported. Cost is
/// `O(points × severity points)`.
///
/// Fails if `points` is 0, or if `g_0` underflows to 0 (a Poisson mean
/// above about 700 claims with no mass at zero severity); use FFT there.
///
/// # Example
///
/// ```
/// use prospicio_aggregate::panjer;
/// use prospicio_prob::{Distribution, Grid, Lognormal, Poisson};
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
    let (a, b) = frequency.panjer_ab().ok_or_else(|| {
        Error::Data(
            "Panjer's recursion needs a count in the (a, b, 1) class; use FFT for this one".into(),
        )
    })?;
    // (p_1 - (a + b) p_0), zero in the (a, b, 0) class.
    let extra = frequency.pmf(1) - (a + b) * frequency.pmf(0);
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
        let mut sum: f64 = (1..=k.min(f.len() - 1))
            .map(|j| (a + b * j as f64 / kf) * f[j] * g[k - j])
            .sum();
        if k < f.len() {
            sum += extra * f[k];
        }
        // With a binomial count (a < 0) the terms alternate in sign, so
        // where the true probability is 0 (beyond N_max times the largest
        // loss) the sum is rounding noise of either sign: clamp it.
        g.push((sum * scale).max(0.0));
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
    use prospicio_prob::{NegativeBinomial, Poisson};

    fn small_severity() -> Grid {
        Grid::new(1.0, vec![0.1, 0.3, 0.25, 0.2, 0.1, 0.05]).unwrap()
    }

    #[test]
    fn zero_modified_and_logarithmic_counts_agree_with_fft() {
        use prospicio_prob::count_families::{Logarithmic, MixedPoisson, Mixing, ZeroModified};
        let sev = small_severity();
        let zm = ZeroModified::new(Poisson::new(3.0).unwrap(), 0.4).unwrap();
        let zt = ZeroModified::truncated(NegativeBinomial::new(2.0, 1.5).unwrap()).unwrap();
        let log = Logarithmic::new(0.6).unwrap();
        let counts: [&dyn Counting; 3] = [&zm, &zt, &log];
        for n in counts {
            let (p, _) = panjer(n, &sev, 200).unwrap();
            let (f, _) = crate::fft(n, &sev, 200).unwrap();
            for (a, b) in p.probs().iter().zip(f.probs()) {
                assert!((a - b).abs() < 1e-13, "{a} vs {b}");
            }
        }
        // A count outside the (a, b, 1) class needs FFT.
        let pig = MixedPoisson::new(3.0, Mixing::InverseGaussian { cv: 0.5 }, 0.0).unwrap();
        assert!(panjer(&pig, &sev, 200).is_err());
        assert!(crate::fft(&pig, &sev, 200).is_ok());
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
    fn binomial_counts_give_no_negative_probabilities() {
        // Beyond n times the largest loss the aggregate has no mass; the
        // alternating recursion leaves rounding noise there, which used to
        // come out negative (down to -1.6e-20 for these) and fail Grid::new.
        let sev = small_severity();
        for (n, p) in [(3u64, 0.4), (20, 0.1), (7, 0.9), (50, 0.5)] {
            let b = prospicio_prob::Binomial::new(n, p).unwrap();
            let (agg, _) = panjer(&b, &sev, 2000).unwrap();
            let support = n as usize * 5;
            // The last point holds the lumped tail, 1 − Σ p: rounding only.
            let probs = agg.probs();
            let tail = probs[support + 1..probs.len() - 1]
                .iter()
                .cloned()
                .fold(0.0, f64::max);
            assert!(tail < 1e-15, "{n} {p} {tail:e}");
            assert!((agg.mean() - b.mean() * sev.mean()).abs() < 1e-10);
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
