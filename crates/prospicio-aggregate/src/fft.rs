//! Compound distributions by fast Fourier transform.

use prospicio_core::{Error, Result};
use prospicio_math::fft::{Complex64, real_forward, real_inverse, spectrum_len};
use prospicio_prob::{Counting, Distribution, Grid};

use crate::compound::{CompoundMethod, CompoundReport, lump_tail};

/// The distribution of `S = X_1 + … + X_N` by FFT, on the severity grid's
/// step with `points` points.
///
/// The severity grid is padded to a power-of-two buffer `L`, transformed,
/// passed through the claim-count pgf, and transformed back. A finite FFT
/// is circular: aggregate mass beyond `L` wraps around onto the start
/// (aliasing). The calculation runs at `2L`, and the result at `L` is that
/// buffer folded onto its first half, so one transform gives both; the
/// largest difference between the two over the returned points is reported
/// as [`CompoundReport::aliasing_error`], and the `2L` result is returned.
/// When that error is not negligible, wrapped mass sits inside the grid
/// and every probability, including `tail_mass`, is unreliable: request
/// more points.
///
/// Any [`Counting`] works (no `(a, b, 0)` restriction), and large claim
/// counts that make Panjer underflow are fine. Cost is `O(L log L)`.
///
/// # Example
///
/// ```
/// use prospicio_aggregate::{fft, panjer};
/// use prospicio_prob::{Grid, Poisson};
///
/// let sev = Grid::new(1.0, vec![0.1, 0.3, 0.25, 0.2, 0.1, 0.05]).unwrap();
/// let n = Poisson::new(3.0).unwrap();
/// let (by_fft, report) = fft(&n, &sev, 60).unwrap();
/// let (by_panjer, _) = panjer(&n, &sev, 60).unwrap();
/// for (a, b) in by_fft.probs().iter().zip(by_panjer.probs()) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// assert!(report.aliasing_error < 1e-12);
/// ```
pub fn fft<N: Counting + ?Sized>(
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
    let base = (2 * points.max(severity.len())).next_power_of_two();
    let (coarse, fine) = compound_pair(frequency, severity.probs(), base);
    let aliasing_error = aliasing_error(&coarse, &fine, points);

    let (probs, tail_mass) = lump_tail(fine[..points].to_vec());
    let grid = Grid::new(severity.step(), probs)?;
    let report = CompoundReport {
        method: CompoundMethod::Fft,
        points,
        tail_mass,
        aliasing_error,
        expected_mean: frequency.mean() * severity.mean(),
        grid_mean: grid.mean(),
    };
    Ok((grid, report))
}

/// The largest difference over the first `points` points between the
/// compound pmf on a buffer and on one twice as long.
pub(crate) fn aliasing_error(coarse: &[f64], fine: &[f64], points: usize) -> f64 {
    coarse[..points]
        .iter()
        .zip(&fine[..points])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// The compound pmf on a circular buffer of `len` points, round-off left
/// signed.
///
/// The severity is real, so the transform is a real one: half the
/// coefficients, and half the pgf evaluations. That takes the pgf of a
/// conjugate to be the conjugate of the pgf, which holds for any power
/// series with real coefficients.
fn compound_signed<N: Counting + ?Sized>(frequency: &N, severity: &[f64], len: usize) -> Vec<f64> {
    let mut signal = vec![0.0; len];
    // A severity longer than the buffer wraps around, like the aggregate.
    for (j, &p) in severity.iter().enumerate() {
        signal[j % len] += p;
    }
    let mut spectrum = vec![Complex64::new(0.0, 0.0); spectrum_len(len)];
    real_forward(&mut signal, &mut spectrum);
    for z in &mut spectrum {
        let (re, im) = frequency.pgf_complex((z.re, z.im));
        *z = Complex64::new(re, im);
    }
    real_inverse(&mut spectrum, &mut signal);
    // The transforms do not normalize.
    let scale = 1.0 / len as f64;
    signal.iter_mut().for_each(|p| *p *= scale);
    signal
}

/// The compound pmf on a circular buffer of `len` points.
pub(crate) fn compound_buffer<N: Counting + ?Sized>(
    frequency: &N,
    severity: &[f64],
    len: usize,
) -> Vec<f64> {
    let mut buffer = compound_signed(frequency, severity, len);
    // Round-off can leave tiny negatives.
    buffer.iter_mut().for_each(|p| *p = p.max(0.0));
    buffer
}

/// The compound pmf on circular buffers of `base` and of `2 * base`
/// points, from one transform at `2 * base`.
///
/// Mass that the long buffer holds at `k + base` is at `k` on the short
/// one, and nothing else differs: the short buffer's transform is every
/// second coefficient of the long one's, at which the pgf takes the same
/// values. So the short result is the long one folded onto its first
/// half, which [`compound_buffer`] at `base` reproduces to round-off.
pub(crate) fn compound_pair<N: Counting + ?Sized>(
    frequency: &N,
    severity: &[f64],
    base: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut fine = compound_signed(frequency, severity, 2 * base);
    let coarse = (0..base)
        .map(|k| (fine[k] + fine[k + base]).max(0.0))
        .collect();
    fine.iter_mut().for_each(|p| *p = p.max(0.0));
    (coarse, fine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panjer;
    use prospicio_prob::{Lognormal, NegativeBinomial, Poisson, Tweedie};

    #[test]
    fn poisson_gamma_matches_the_tweedie_series() {
        // A Poisson number of gamma losses is a Tweedie: FFT on a fine
        // local-moment grid against the closed series.
        let y = Tweedie::from_poisson_gamma(3.0, 2.0, 10.0).unwrap();
        let step = 0.01;
        let (sev, _) = Grid::local_moment(&y.severity(), step, 30_000).unwrap();
        let (agg, report) = fft(&Poisson::new(3.0).unwrap(), &sev, 100_000).unwrap();
        assert!(report.aliasing_error < 1e-12);
        assert!(report.tail_mass < 1e-12);
        // Local moment matching keeps every limited mean on the grid, so
        // the layer means agree closely; the distribution function within
        // a step's worth of mass.
        for x in [5.0, 40.0, 100.0, 250.0] {
            use prospicio_prob::{Distribution, Severity};
            assert!((agg.lev(x) / y.lev(x) - 1.0).abs() < 1e-6, "lev {x}");
            assert!((agg.cdf(x) - y.cdf(x)).abs() < 2e-4, "cdf {x}");
        }
    }

    #[test]
    fn one_transform_gives_both_buffers() {
        // The folded long buffer against a transform of its own at the
        // short length, with and without mass wrapping around, and with a
        // severity longer than the short buffer.
        let x = Lognormal::new(0.0, 1.0).unwrap();
        let (sev, _) = Grid::rounding(&x, 0.25, 100).unwrap();
        for (n, base) in [
            (&Poisson::new(3.0).unwrap() as &dyn Counting, 256),
            (&NegativeBinomial::new(2.5, 0.2).unwrap(), 256),
            (&Poisson::new(40.0).unwrap(), 64),
        ] {
            let (coarse, fine) = compound_pair(n, sev.probs(), base);
            let short = compound_buffer(n, sev.probs(), base);
            let long = compound_buffer(n, sev.probs(), 2 * base);
            assert_eq!(fine, long);
            for (a, b) in coarse.iter().zip(&short) {
                assert!((a - b).abs() < 1e-15, "base {base}: {a} against {b}");
            }
        }
        // Where the mass wraps, the two buffers differ by what wrapped.
        let (coarse, fine) = compound_pair(&Poisson::new(40.0).unwrap(), sev.probs(), 64);
        assert!(aliasing_error(&coarse, &fine, 64) > 1e-3);
    }

    #[test]
    fn the_real_transform_matches_the_complex_one() {
        // The buffer as it was computed before #973, by a complex
        // transform of the whole circle.
        use prospicio_math::fft::{forward, inverse};
        let x = Lognormal::new(0.0, 1.0).unwrap();
        let (sev, _) = Grid::rounding(&x, 0.25, 100).unwrap();
        for n in [
            &Poisson::new(3.0).unwrap() as &dyn Counting,
            &NegativeBinomial::new(2.5, 0.2).unwrap(),
        ] {
            for len in [64, 512] {
                let mut buffer = vec![Complex64::new(0.0, 0.0); len];
                for (j, &p) in sev.probs().iter().enumerate() {
                    buffer[j % len].re += p;
                }
                forward(&mut buffer);
                for z in &mut buffer {
                    let (re, im) = n.pgf_complex((z.re, z.im));
                    *z = Complex64::new(re, im);
                }
                inverse(&mut buffer);
                let real = compound_buffer(n, sev.probs(), len);
                for (z, p) in buffer.iter().zip(&real) {
                    assert!(((z.re / len as f64).max(0.0) - p).abs() < 1e-15);
                }
            }
        }
    }

    fn small_severity() -> Grid {
        Grid::new(1.0, vec![0.1, 0.3, 0.25, 0.2, 0.1, 0.05]).unwrap()
    }

    #[test]
    fn agrees_with_panjer() {
        let sev = small_severity();
        for n in [
            &Poisson::new(3.0).unwrap() as &dyn Counting,
            &NegativeBinomial::new(2.5, 1.5).unwrap(),
        ] {
            let (a, ra) = fft(n, &sev, 200).unwrap();
            let (b, rb) = panjer(n, &sev, 200).unwrap();
            for (x, y) in a.probs().iter().zip(b.probs()) {
                assert!((x - y).abs() < 1e-13);
            }
            assert!((ra.tail_mass - rb.tail_mass).abs() < 1e-13);
            assert_eq!(ra.method, CompoundMethod::Fft);
        }
    }

    #[test]
    fn handles_claim_counts_that_underflow_panjer() {
        let (sev, _) = Grid::local_moment(&Lognormal::new(0.0, 0.5).unwrap(), 0.25, 64).unwrap();
        let n = Poisson::new(2_000.0).unwrap();
        assert!(panjer(&n, &sev, 16_384).is_err());
        let (agg, report) = fft(&n, &sev, 16_384).unwrap();
        // E[S] = 2000 E[X], about 2266; the grid reaches 4096.
        assert!(report.tail_mass < 1e-12);
        assert!((agg.mean() - 2_000.0 * sev.mean()).abs() < 1e-6 * agg.mean());
        assert!(report.aliasing_error < 1e-12);
    }

    #[test]
    fn reports_aliasing_when_the_buffer_is_too_short() {
        // Mean 3000 but a 64-point request: the 128 / 256 buffers wrap.
        let sev = Grid::new(1.0, vec![0.0, 1.0]).unwrap();
        let (_, report) = fft(&Poisson::new(3_000.0).unwrap(), &sev, 64).unwrap();
        assert!(report.aliasing_error > 1e-6);
        // Nearly all of S lies above 64, yet wrapped mass lands back inside
        // the grid, so tail_mass understates it: only aliasing_error shows
        // the result is unusable.
        assert!(report.tail_mass < 0.99);
    }

    #[test]
    fn rejects_zero_points() {
        assert!(fft(&Poisson::new(1.0).unwrap(), &small_severity(), 0).is_err());
    }
}
