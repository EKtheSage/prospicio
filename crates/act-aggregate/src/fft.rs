//! Compound distributions by fast Fourier transform.

use act_core::{Error, Result};
use act_prob::{Counting, Distribution, Grid};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

use crate::compound::{CompoundMethod, CompoundReport, lump_tail};

/// The distribution of `S = X_1 + … + X_N` by FFT, on the severity grid's
/// step with `points` points.
///
/// The severity grid is padded to a power-of-two buffer `L`, transformed,
/// passed through the claim-count pgf, and transformed back. A finite FFT
/// is circular: aggregate mass beyond `L` wraps around onto the start
/// (aliasing). The calculation runs at `L` and `2L` and reports the largest
/// difference over the returned points as
/// [`CompoundReport::aliasing_error`]; the `2L` result is returned. When
/// that error is not negligible, wrapped mass sits inside the grid and
/// every probability, including `tail_mass`, is unreliable: request more
/// points.
///
/// Any [`Counting`] works (no `(a, b, 0)` restriction), and large claim
/// counts that make Panjer underflow are fine. Cost is `O(L log L)`.
///
/// # Example
///
/// ```
/// use act_aggregate::{fft, panjer};
/// use act_prob::{Grid, Poisson};
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
    let coarse = compound_buffer(frequency, severity.probs(), base);
    let fine = compound_buffer(frequency, severity.probs(), 2 * base);
    let aliasing_error = coarse[..points]
        .iter()
        .zip(&fine[..points])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

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

/// The compound pmf on a circular buffer of `len` points.
fn compound_buffer<N: Counting + ?Sized>(frequency: &N, severity: &[f64], len: usize) -> Vec<f64> {
    let mut planner = FftPlanner::<f64>::new();
    let mut buffer = vec![Complex64::new(0.0, 0.0); len];
    // A severity longer than the buffer wraps around, like the aggregate.
    for (j, &p) in severity.iter().enumerate() {
        buffer[j % len].re += p;
    }
    planner.plan_fft_forward(len).process(&mut buffer);
    for z in &mut buffer {
        let (re, im) = frequency.pgf_complex((z.re, z.im));
        *z = Complex64::new(re, im);
    }
    planner.plan_fft_inverse(len).process(&mut buffer);
    // rustfft does not normalize; round-off can leave tiny negatives.
    buffer
        .iter()
        .map(|z| (z.re / len as f64).max(0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panjer;
    use act_prob::{Lognormal, NegativeBinomial, Poisson};

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
