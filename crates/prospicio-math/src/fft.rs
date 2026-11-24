//! Fast Fourier transforms, with the plans and the scratch space kept per
//! thread.
//!
//! Every transform here is unnormalized, as `rustfft`'s are: a forward
//! transform followed by the inverse multiplies the signal by its length,
//! so the caller divides by the length once.
//!
//! A plan (the twiddle factors of one length) costs two to four times the
//! transform it serves, so it is worth keeping. The free functions build a
//! plan and the scratch buffer on a thread's first transform of a length
//! and reuse them in its later ones, for lengths up to [`CACHED_LEN`]; a
//! longer transform through them plans for itself and drops the plan, so
//! that no thread holds on to tens of megabytes after one large transform.
//! A caller that transforms one length several times, of any size, takes a
//! [`Plan`] or a [`RealPlan`] and keeps it for as long as it needs it.
//!
//! The real transforms take a real signal of length `n` to the first
//! `n / 2 + 1` coefficients of its transform (the rest are their complex
//! conjugates) and back, at about half the cost of a complex transform:
//! use them whenever the signal is real, as a probability vector is.
//!
//! ```
//! use prospicio_math::fft::{Complex64, real_forward, real_inverse, spectrum_len};
//!
//! // The convolution of a probability vector with itself.
//! let mut signal = vec![0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0];
//! let n = signal.len();
//! let mut spectrum = vec![Complex64::new(0.0, 0.0); spectrum_len(n)];
//! real_forward(&mut signal, &mut spectrum);
//! for z in &mut spectrum {
//!     *z = *z * *z;
//! }
//! real_inverse(&mut spectrum, &mut signal);
//! let twice: Vec<f64> = signal.iter().map(|v| v / n as f64).collect();
//! let exact = [0.25, 0.25, 0.3125, 0.125, 0.0625, 0.0, 0.0, 0.0];
//! for (a, b) in twice.iter().zip(exact) {
//!     assert!((a - b).abs() < 1e-15);
//! }
//! ```

use std::cell::RefCell;
use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};

pub use rustfft::num_complex::Complex64;

/// The longest transform whose plan a thread keeps: `2^16` points, a
/// megabyte of twiddle factors.
pub const CACHED_LEN: usize = 1 << 16;

/// The number of coefficients [`real_forward`] writes for a signal of
/// `len` points, `len / 2 + 1`.
pub fn spectrum_len(len: usize) -> usize {
    len / 2 + 1
}

/// One thread's plans and scratch space.
struct Plans {
    complex: FftPlanner<f64>,
    real: RealFftPlanner<f64>,
    scratch: Vec<Complex64>,
}

impl Plans {
    fn new() -> Self {
        Self {
            complex: FftPlanner::new(),
            real: RealFftPlanner::new(),
            scratch: Vec::new(),
        }
    }

    /// The scratch buffer, grown to `len`. Its contents are arbitrary.
    fn scratch(scratch: &mut Vec<Complex64>, len: usize) -> &mut [Complex64] {
        if scratch.len() < len {
            scratch.resize(len, Complex64::new(0.0, 0.0));
        }
        &mut scratch[..len]
    }
}

thread_local! {
    static PLANS: RefCell<Plans> = RefCell::new(Plans::new());
}

/// Runs `f` with the thread's plans when `len` is short enough to keep,
/// and with plans of its own otherwise.
fn with_plans<T>(len: usize, f: impl FnOnce(&mut Plans) -> T) -> T {
    if len <= CACHED_LEN {
        PLANS.with_borrow_mut(f)
    } else {
        f(&mut Plans::new())
    }
}

fn complex(buffer: &mut [Complex64], plan: impl FnOnce(&mut FftPlanner<f64>) -> Arc<dyn Fft<f64>>) {
    if buffer.is_empty() {
        return;
    }
    with_plans(buffer.len(), |plans| {
        let fft = plan(&mut plans.complex);
        let scratch = Plans::scratch(&mut plans.scratch, fft.get_inplace_scratch_len());
        fft.process_with_scratch(buffer, scratch);
    });
}

/// The forward transform of `buffer`, in place:
/// `X_k = Σ_j x_j exp(−2πi jk / n)`. Any length works; powers of two are
/// fastest.
///
/// ```
/// use prospicio_math::fft::{Complex64, forward, inverse};
///
/// let signal = [1.0, 2.0, 0.0, -1.0, 0.5];
/// let mut buffer: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
/// forward(&mut buffer);
/// assert!((buffer[0].re - 2.5).abs() < 1e-15);
/// inverse(&mut buffer);
/// for (z, x) in buffer.iter().zip(signal) {
///     assert!((z.re / 5.0 - x).abs() < 1e-15 && z.im.abs() < 1e-15);
/// }
/// ```
pub fn forward(buffer: &mut [Complex64]) {
    let len = buffer.len();
    complex(buffer, |planner| planner.plan_fft_forward(len));
}

/// The unnormalized inverse transform of `buffer`, in place:
/// `x_j = Σ_k X_k exp(2πi jk / n)`, which is `n` times the signal that
/// [`forward`] transformed.
pub fn inverse(buffer: &mut [Complex64]) {
    let len = buffer.len();
    complex(buffer, |planner| planner.plan_fft_inverse(len));
}

/// The forward transform of the real `signal`, of which `spectrum`
/// receives the coefficients `0..=n / 2`; coefficient `n − k` is the
/// complex conjugate of coefficient `k`. `signal` is used as working
/// space and holds arbitrary values afterwards.
///
/// # Panics
///
/// If `spectrum.len()` is not [`spectrum_len`]`(signal.len())`.
pub fn real_forward(signal: &mut [f64], spectrum: &mut [Complex64]) {
    let len = signal.len();
    assert_eq!(
        spectrum.len(),
        spectrum_len(len),
        "a real signal of {len} points has {} coefficients",
        spectrum_len(len)
    );
    if len == 0 {
        spectrum[0] = Complex64::new(0.0, 0.0);
        return;
    }
    with_plans(len, |plans| {
        let fft = plans.real.plan_fft_forward(len);
        let scratch = Plans::scratch(&mut plans.scratch, fft.get_scratch_len());
        fft.process_with_scratch(signal, spectrum, scratch)
            .expect("the lengths were checked");
    });
}

/// The unnormalized inverse of [`real_forward`]: `signal` receives `n`
/// times the real signal whose coefficients `0..=n / 2` are `spectrum`,
/// with `n = signal.len()`. `spectrum` is used as working space and holds
/// arbitrary values afterwards.
///
/// A real signal's coefficient 0 is real, and so is coefficient `n / 2`
/// when `n` is even: the imaginary parts of those two are ignored. That is
/// the real part of what [`inverse`] returns for the same coefficients
/// completed by conjugates.
///
/// # Panics
///
/// If `spectrum.len()` is not [`spectrum_len`]`(signal.len())`.
pub fn real_inverse(spectrum: &mut [Complex64], signal: &mut [f64]) {
    let len = signal.len();
    assert_eq!(
        spectrum.len(),
        spectrum_len(len),
        "a real signal of {len} points has {} coefficients",
        spectrum_len(len)
    );
    if len == 0 {
        return;
    }
    spectrum[0].im = 0.0;
    if len.is_multiple_of(2) {
        spectrum[len / 2].im = 0.0;
    }
    with_plans(len, |plans| {
        let fft = plans.real.plan_fft_inverse(len);
        let scratch = Plans::scratch(&mut plans.scratch, fft.get_scratch_len());
        fft.process_with_scratch(spectrum, signal, scratch)
            .expect("the lengths were checked, and the two real coefficients made real");
    });
}

/// The complex transforms of one length, planned once.
///
/// For a caller that transforms the same length several times: planning
/// costs more than a transform, and the free functions keep a plan only up
/// to [`CACHED_LEN`]. Each direction is planned at its first use.
///
/// ```
/// use prospicio_math::fft::{Complex64, Plan};
///
/// let mut plan = Plan::new(6);
/// let mut buffer = vec![Complex64::new(1.0, 0.0); 6];
/// plan.forward(&mut buffer);
/// assert!((buffer[0].re - 6.0).abs() < 1e-15 && buffer[1].norm() < 1e-15);
/// plan.inverse(&mut buffer);
/// assert!((buffer[3].re - 6.0).abs() < 1e-14);
/// ```
pub struct Plan {
    len: usize,
    forward: Option<Arc<dyn Fft<f64>>>,
    inverse: Option<Arc<dyn Fft<f64>>>,
    scratch: Vec<Complex64>,
}

impl Plan {
    /// A plan for buffers of `len` points.
    pub fn new(len: usize) -> Self {
        Self {
            len,
            forward: None,
            inverse: None,
            scratch: Vec::new(),
        }
    }

    /// The length of the buffers this plan transforms.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the plan is for empty buffers.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// [`forward`] on every `len` points of `buffer` in turn, so a matrix
    /// stored row by row has each row transformed.
    ///
    /// # Panics
    ///
    /// If `buffer.len()` is not a multiple of the plan's length.
    pub fn forward(&mut self, buffer: &mut [Complex64]) {
        let len = self.len;
        let fft = self
            .forward
            .get_or_insert_with(|| with_plans(len, |plans| plans.complex.plan_fft_forward(len)));
        Self::run(fft.as_ref(), &mut self.scratch, len, buffer);
    }

    /// [`inverse`] on every `len` points of `buffer` in turn.
    ///
    /// # Panics
    ///
    /// If `buffer.len()` is not a multiple of the plan's length.
    pub fn inverse(&mut self, buffer: &mut [Complex64]) {
        let len = self.len;
        let fft = self
            .inverse
            .get_or_insert_with(|| with_plans(len, |plans| plans.complex.plan_fft_inverse(len)));
        Self::run(fft.as_ref(), &mut self.scratch, len, buffer);
    }

    fn run(fft: &dyn Fft<f64>, scratch: &mut Vec<Complex64>, len: usize, buffer: &mut [Complex64]) {
        if len == 0 {
            assert!(
                buffer.is_empty(),
                "a plan of no points takes an empty buffer"
            );
            return;
        }
        assert!(
            buffer.len().is_multiple_of(len),
            "a buffer of {} points is not a whole number of transforms of {len}",
            buffer.len()
        );
        let scratch = Plans::scratch(scratch, fft.get_inplace_scratch_len());
        fft.process_with_scratch(buffer, scratch);
    }
}

/// The real transforms of one length, planned once: [`real_forward`] and
/// [`real_inverse`] for a caller that transforms the same length several
/// times. Each direction is planned at its first use.
///
/// ```
/// use prospicio_math::fft::{Complex64, RealPlan};
///
/// // The 3-fold convolution of a probability vector, one plan for both
/// // directions.
/// let mut plan = RealPlan::new(8);
/// let mut signal = vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let mut spectrum = vec![Complex64::new(0.0, 0.0); plan.spectrum_len()];
/// plan.forward(&mut signal, &mut spectrum);
/// for z in &mut spectrum {
///     *z = *z * *z * *z;
/// }
/// plan.inverse(&mut spectrum, &mut signal);
/// for (p, exact) in signal.iter().zip([0.125, 0.375, 0.375, 0.125]) {
///     assert!((p / 8.0 - exact).abs() < 1e-15);
/// }
/// ```
pub struct RealPlan {
    len: usize,
    forward: Option<Arc<dyn RealToComplex<f64>>>,
    inverse: Option<Arc<dyn ComplexToReal<f64>>>,
    scratch: Vec<Complex64>,
}

impl RealPlan {
    /// A plan for signals of `len` points.
    pub fn new(len: usize) -> Self {
        Self {
            len,
            forward: None,
            inverse: None,
            scratch: Vec::new(),
        }
    }

    /// The length of the signals this plan transforms.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the plan is for empty signals.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of coefficients of a signal of this length,
    /// [`spectrum_len`]`(len)`.
    pub fn spectrum_len(&self) -> usize {
        spectrum_len(self.len)
    }

    /// [`real_forward`] with this plan.
    ///
    /// # Panics
    ///
    /// If `signal.len()` is not the plan's length or `spectrum.len()` is
    /// not [`RealPlan::spectrum_len`].
    pub fn forward(&mut self, signal: &mut [f64], spectrum: &mut [Complex64]) {
        let len = self.len;
        check_real(len, signal.len(), spectrum.len());
        if len == 0 {
            spectrum[0] = Complex64::new(0.0, 0.0);
            return;
        }
        let fft = self
            .forward
            .get_or_insert_with(|| with_plans(len, |plans| plans.real.plan_fft_forward(len)));
        let scratch = Plans::scratch(&mut self.scratch, fft.get_scratch_len());
        fft.process_with_scratch(signal, spectrum, scratch)
            .expect("the lengths were checked");
    }

    /// [`real_inverse`] with this plan.
    ///
    /// # Panics
    ///
    /// If `signal.len()` is not the plan's length or `spectrum.len()` is
    /// not [`RealPlan::spectrum_len`].
    pub fn inverse(&mut self, spectrum: &mut [Complex64], signal: &mut [f64]) {
        let len = self.len;
        check_real(len, signal.len(), spectrum.len());
        if len == 0 {
            return;
        }
        spectrum[0].im = 0.0;
        if len.is_multiple_of(2) {
            spectrum[len / 2].im = 0.0;
        }
        let fft = self
            .inverse
            .get_or_insert_with(|| with_plans(len, |plans| plans.real.plan_fft_inverse(len)));
        let scratch = Plans::scratch(&mut self.scratch, fft.get_scratch_len());
        fft.process_with_scratch(spectrum, signal, scratch)
            .expect("the lengths were checked, and the two real coefficients made real");
    }
}

/// Panics unless a signal of `signal` points and `spectrum` coefficients
/// suit a real plan of `len` points.
fn check_real(len: usize, signal: usize, spectrum: usize) {
    assert_eq!(signal, len, "the plan is for signals of {len} points");
    assert_eq!(
        spectrum,
        spectrum_len(len),
        "a real signal of {len} points has {} coefficients",
        spectrum_len(len)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    /// A fixed signal with no symmetry.
    fn signal(len: usize) -> Vec<f64> {
        (0..len)
            .map(|j| ((j * j + 3 * j + 1) % 17) as f64 / 17.0 - 0.3)
            .collect()
    }

    /// The transform by its definition.
    fn slow(x: &[Complex64], sign: f64) -> Vec<Complex64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                x.iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        v * Complex64::from_polar(1.0, sign * TAU * (j * k % n) as f64 / n as f64)
                    })
                    .sum()
            })
            .collect()
    }

    fn lift(x: &[f64]) -> Vec<Complex64> {
        x.iter().map(|&v| Complex64::new(v, 0.0)).collect()
    }

    #[test]
    fn the_complex_transforms_match_the_definition() {
        for len in [1, 2, 3, 8, 12, 17, 64] {
            let x: Vec<Complex64> = signal(2 * len)
                .chunks(2)
                .map(|c| Complex64::new(c[0], c[1]))
                .collect();
            for (sign, transform) in [(-1.0, forward as fn(&mut [Complex64])), (1.0, inverse)] {
                let mut fast = x.clone();
                transform(&mut fast);
                for (a, b) in fast.iter().zip(slow(&x, sign)) {
                    assert!((a - b).norm() < 1e-12, "length {len}");
                }
            }
        }
    }

    #[test]
    fn the_real_transforms_match_the_complex_ones() {
        for len in [1, 2, 3, 8, 12, 17, 64, 1 << 10] {
            let x = signal(len);
            let mut full = lift(&x);
            forward(&mut full);
            let mut work = x.clone();
            let mut spectrum = vec![Complex64::new(0.0, 0.0); spectrum_len(len)];
            real_forward(&mut work, &mut spectrum);
            for (a, b) in spectrum.iter().zip(&full) {
                assert!((a - b).norm() < 1e-11, "length {len}");
            }
            let mut back = vec![0.0; len];
            real_inverse(&mut spectrum, &mut back);
            for (a, b) in back.iter().zip(&x) {
                assert!((a / len as f64 - b).abs() < 1e-13, "length {len}");
            }
        }
    }

    #[test]
    fn the_real_inverse_ignores_the_imaginary_part_of_the_real_coefficients() {
        // The real part of the complex inverse of the completed spectrum.
        for len in [8, 9] {
            let mut spectrum: Vec<Complex64> = signal(2 * spectrum_len(len))
                .chunks(2)
                .map(|c| Complex64::new(c[0], c[1]))
                .collect();
            let mut full = spectrum.clone();
            full.extend((1..len - len / 2).rev().map(|k| spectrum[k].conj()));
            inverse(&mut full);
            let mut back = vec![0.0; len];
            real_inverse(&mut spectrum, &mut back);
            for (a, b) in back.iter().zip(&full) {
                assert!((a - b.re).abs() < 1e-13, "length {len}");
            }
        }
    }

    #[test]
    fn a_length_above_the_cache_transforms_too() {
        // Too long for the definition's double loop: Parseval's identity
        // and the round trips instead.
        let len = 2 * CACHED_LEN;
        let x = signal(len);
        let mut work = x.clone();
        let mut spectrum = vec![Complex64::new(0.0, 0.0); spectrum_len(len)];
        real_forward(&mut work, &mut spectrum);
        let energy: f64 = x.iter().map(|v| v * v).sum();
        let transformed: f64 = spectrum[1..len / 2]
            .iter()
            .map(|z| 2.0 * z.norm_sqr())
            .sum::<f64>()
            + spectrum[0].norm_sqr()
            + spectrum[len / 2].norm_sqr();
        assert!((transformed / len as f64 / energy - 1.0).abs() < 1e-12);
        let mut back = vec![0.0; len];
        real_inverse(&mut spectrum, &mut back);
        for (a, b) in back.iter().zip(&x) {
            assert!((a / len as f64 - b).abs() < 1e-13);
        }
        let mut buffer = lift(&x);
        forward(&mut buffer);
        inverse(&mut buffer);
        for (a, b) in buffer.iter().zip(&x) {
            assert!((a.re / len as f64 - b).abs() < 1e-13);
        }
    }

    #[test]
    fn empty_buffers_are_left_alone() {
        forward(&mut []);
        inverse(&mut []);
        let mut spectrum = [Complex64::new(1.0, 1.0)];
        real_forward(&mut [], &mut spectrum);
        assert_eq!(spectrum[0], Complex64::new(0.0, 0.0));
        real_inverse(&mut spectrum, &mut []);
    }

    #[test]
    #[should_panic(expected = "has 5 coefficients")]
    fn a_spectrum_of_the_wrong_length_panics() {
        real_forward(&mut [0.0; 8], &mut [Complex64::new(0.0, 0.0); 8]);
    }

    #[test]
    fn the_plans_give_the_free_functions_transforms() {
        for len in [1, 5, 8, 64, 2 * CACHED_LEN] {
            let x = signal(len);
            let mut real = RealPlan::new(len);
            assert_eq!((real.len(), real.spectrum_len()), (len, spectrum_len(len)));
            let mut spectrum = vec![Complex64::new(0.0, 0.0); spectrum_len(len)];
            let mut free = spectrum.clone();
            // Twice, so the second use is of a kept plan.
            for _ in 0..2 {
                real.forward(&mut x.clone(), &mut spectrum);
                real_forward(&mut x.clone(), &mut free);
                assert_eq!(spectrum, free);
                let (mut back, mut free_back) = (vec![0.0; len], vec![0.0; len]);
                real.inverse(&mut spectrum.clone(), &mut back);
                real_inverse(&mut free.clone(), &mut free_back);
                assert_eq!(back, free_back);
            }
            let mut plan = Plan::new(len);
            assert_eq!(plan.len(), len);
            let (mut a, mut b) = (lift(&x), lift(&x));
            for _ in 0..2 {
                plan.forward(&mut a);
                forward(&mut b);
                assert_eq!(a, b);
                plan.inverse(&mut a);
                inverse(&mut b);
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn a_plan_transforms_each_row_of_a_matrix() {
        let (rows, cols) = (3, 8);
        let mut matrix: Vec<Complex64> = lift(&signal(rows * cols));
        let mut by_row = matrix.clone();
        Plan::new(cols).forward(&mut matrix);
        by_row.chunks_mut(cols).for_each(forward);
        assert_eq!(matrix, by_row);
    }

    #[test]
    fn empty_plans_do_nothing() {
        let mut plan = Plan::new(0);
        assert!(plan.is_empty() && RealPlan::new(0).is_empty() && !Plan::new(2).is_empty());
        plan.forward(&mut []);
        plan.inverse(&mut []);
        let mut spectrum = [Complex64::new(1.0, 1.0)];
        RealPlan::new(0).forward(&mut [], &mut spectrum);
        assert_eq!(spectrum[0], Complex64::new(0.0, 0.0));
        RealPlan::new(0).inverse(&mut spectrum, &mut []);
    }

    #[test]
    #[should_panic(expected = "not a whole number of transforms of 4")]
    fn a_buffer_that_is_no_multiple_of_the_plan_panics() {
        Plan::new(4).forward(&mut [Complex64::new(0.0, 0.0); 6]);
    }

    #[test]
    #[should_panic(expected = "the plan is for signals of 8 points")]
    fn a_signal_of_another_length_panics() {
        RealPlan::new(8).forward(&mut [0.0; 4], &mut [Complex64::new(0.0, 0.0); 5]);
    }
}
