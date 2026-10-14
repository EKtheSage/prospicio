//! Forward-mode dual numbers and the chunked gradient driver
//! (`docs/design/reserving-compartmental.md`, decision 4).
//!
//! A [`Dual<N>`] carries a value and `N` tangents. [`gradient`] runs a
//! log density written over [`Scalar`] `ceil(p / N)` times for `p`
//! parameters, each pass seeding the tangents of the next `N` of them, so
//! one fixed `N` serves any parameter count known only at run time. `N`
//! is [`CHUNK`] in the crate, chosen by the benchmark in
//! `knowledge/findings/autodiff-benchmark.md`.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use super::scalar::Scalar;

/// The number of tangents each pass of [`gradient`] seeds. Measured against
/// 4, 8 and 16 at 27, 29 and 90 parameters, 32 was fastest or within noise
/// of the fastest at each: one pass up to 32 parameters, and three, not
/// six, at 90 (`knowledge/findings/autodiff-benchmark.md`).
pub(crate) const CHUNK: usize = 32;

/// A value `v` and its derivatives `d` along `N` directions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Dual<const N: usize> {
    pub(crate) v: f64,
    pub(crate) d: [f64; N],
}

impl<const N: usize> Dual<N> {
    /// A constant: every tangent 0.
    #[inline]
    pub(crate) fn constant(v: f64) -> Self {
        Self { v, d: [0.0; N] }
    }

    /// The variable seeded in direction `k`: tangent `k` is 1 (the tests
    /// seed with it; [`gradient`] sets the tangents in place).
    #[cfg(test)]
    #[inline]
    pub(crate) fn variable(v: f64, k: usize) -> Self {
        let mut d = [0.0; N];
        d[k] = 1.0;
        Self { v, d }
    }

    /// `f(v)` with derivative `slope = f'(v)`: the chain rule.
    #[inline]
    fn chain(self, v: f64, slope: f64) -> Self {
        let mut d = self.d;
        for x in &mut d {
            *x *= slope;
        }
        Self { v, d }
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    #[inline]
    fn add(mut self, o: Self) -> Self {
        self.v += o.v;
        for k in 0..N {
            self.d[k] += o.d[k];
        }
        self
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    #[inline]
    fn sub(mut self, o: Self) -> Self {
        self.v -= o.v;
        for k in 0..N {
            self.d[k] -= o.d[k];
        }
        self
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    // The product rule.
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn mul(self, o: Self) -> Self {
        let d = std::array::from_fn(|k| self.d[k] * o.v + self.v * o.d[k]);
        Self { v: self.v * o.v, d }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    // The quotient rule.
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v;
        let v = self.v / o.v;
        let d = std::array::from_fn(|k| (self.d[k] - v * o.d[k]) * inv);
        Self { v, d }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.chain(-self.v, -1.0)
    }
}

impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn add(mut self, o: f64) -> Self {
        self.v += o;
        self
    }
}

impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn sub(mut self, o: f64) -> Self {
        self.v -= o;
        self
    }
}

impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn mul(self, o: f64) -> Self {
        self.chain(self.v * o, o)
    }
}

impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Self;
    #[inline]
    fn div(self, o: f64) -> Self {
        self.chain(self.v / o, 1.0 / o)
    }
}

impl<const N: usize> AddAssign for Dual<N> {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl<const N: usize> SubAssign for Dual<N> {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

impl<const N: usize> Scalar for Dual<N> {
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(v)
    }
    #[inline]
    fn value(self) -> f64 {
        self.v
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e)
    }
    #[inline]
    fn exp_m1(self) -> Self {
        self.chain(self.v.exp_m1(), self.v.exp())
    }
    #[inline]
    fn ln(self) -> Self {
        self.chain(self.v.ln(), 1.0 / self.v)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        self.chain(self.v.ln_1p(), 1.0 / (1.0 + self.v))
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.chain(s, 0.5 / s)
    }
    #[inline]
    fn tanh(self) -> Self {
        // 1 / cosh^2, not 1 - tanh^2, which cancels for large |x|.
        let c = self.v.cosh();
        self.chain(self.v.tanh(), 1.0 / (c * c))
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        match n {
            0 => Self::constant(1.0),
            _ => self.chain(self.v.powi(n), f64::from(n) * self.v.powi(n - 1)),
        }
    }
}

/// The value of `f` at `theta` and its gradient, written into `grad`, by
/// `ceil(p / N)` forward passes for `p = theta.len()`: pass `c` seeds the
/// tangents of parameters `c N .. c N + N`. The value is the first pass'
/// (every pass computes the same one).
pub(crate) fn gradient<const N: usize>(
    theta: &[f64],
    grad: &mut [f64],
    f: impl Fn(&[Dual<N>]) -> Dual<N>,
) -> f64 {
    let p = theta.len();
    assert_eq!(grad.len(), p, "the gradient has one entry per parameter");
    let mut x: Vec<Dual<N>> = theta.iter().map(|&v| Dual::constant(v)).collect();
    let mut value = f64::NAN;
    for start in (0..p.max(1)).step_by(N) {
        let end = (start + N).min(p);
        for (k, j) in (start..end).enumerate() {
            x[j].d[k] = 1.0;
        }
        let y = f(&x);
        if start == 0 {
            value = y.v;
        }
        grad[start..end].copy_from_slice(&y.d[..end - start]);
        for (k, j) in (start..end).enumerate() {
            x[j].d[k] = 0.0;
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A density of five parameters with every operation in it.
    fn f<S: Scalar>(x: &[S]) -> S {
        let a = (x[0] * x[1]).exp() + (x[2] / x[3]).tanh();
        let b = (x[4] * x[4] + 1.0).ln() - (x[0] - 0.3).powi(3) * 2.0;
        a * b + (x[1] * x[1] + 0.5).sqrt() - (-x[2]).exp_m1() + x[3].ln_1p() / 4.0
    }

    fn assert_driver<const N: usize>(theta: &[f64], expected: &[f64]) {
        let mut grad = vec![0.0; theta.len()];
        let v = gradient::<N>(theta, &mut grad, f);
        assert_eq!(v, f(theta));
        for (a, b) in grad.iter().zip(expected) {
            assert!(
                (a - b).abs() <= 1e-15 * b.abs().max(1.0),
                "N {N}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn every_chunk_size_gives_one_gradient() {
        let theta = [0.4, -0.7, 1.1, 0.9, -0.2];
        // One pass of five tangents, the reference.
        let mut expected = vec![0.0; 5];
        gradient::<5>(&theta, &mut expected, f);
        // Against central differences.
        for j in 0..5 {
            let h = 1e-6;
            let (mut up, mut down) = (theta, theta);
            up[j] += h;
            down[j] -= h;
            let fd = (f(&up) - f(&down)) / (2.0 * h);
            assert!(
                (fd - expected[j]).abs() <= 1e-8 * (1.0 + fd.abs()),
                "{j}: {fd} vs {}",
                expected[j]
            );
        }
        // Chunks that divide p, that do not, and that exceed it.
        assert_driver::<1>(&theta, &expected);
        assert_driver::<2>(&theta, &expected);
        assert_driver::<4>(&theta, &expected);
        assert_driver::<CHUNK>(&theta, &expected);
        assert_driver::<32>(&theta, &expected);
    }

    #[test]
    fn no_parameters_still_gives_the_value() {
        let mut grad = [];
        let v = gradient::<4>(&[], &mut grad, |_| Dual::constant(2.5));
        assert_eq!(v, 2.5);
    }
}
