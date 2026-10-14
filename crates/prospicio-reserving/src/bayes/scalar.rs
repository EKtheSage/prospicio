//! The number type a generic log density is written in
//! (`docs/design/reserving-compartmental.md`, decision 4;
//! `docs/design/reserving-bayes.md`, decision 15).
//!
//! A density written once over [`Scalar`] gives its value with `f64` and
//! its gradient with the forward-mode [`Dual`](super::dual::Dual) through
//! the chunked driver [`gradient`](super::dual::gradient). The trait holds
//! only what the reserving densities need: the arithmetic operators with
//! `Self` and with `f64`, and `exp`, `exp_m1`, `ln`, `ln_1p`, `sqrt`,
//! `tanh` and `powi`. A branch on a value (a series branch, an
//! overflow-safe form) reads [`Scalar::value`], so the dual follows the
//! same branch as the plain value.
//!
//! The existing models keep their hand-written gradients as the fast
//! path, with the generic densities beside them as test-only oracles;
//! [`Compartmental`](super::Compartmental) is written only generically.

// `softplus`, `logistic`, `powi` and `tanh` serve the test-only oracles
// (and `Compartmental` uses `tanh`), so some are unused outside tests.
#![cfg_attr(not(test), allow(dead_code))]

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// A real number a log density can be evaluated in: `f64` for the value,
/// [`Dual`](super::dual::Dual) for the value and its gradient.
pub(crate) trait Scalar:
    Copy
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + AddAssign
    + SubAssign
{
    /// A constant, with no derivative.
    fn from_f64(v: f64) -> Self;
    /// The value, without the derivatives.
    fn value(self) -> f64;
    fn exp(self) -> Self;
    /// `exp(x) - 1`, accurate near 0.
    fn exp_m1(self) -> Self;
    fn ln(self) -> Self;
    /// `ln(1 + x)`, accurate near 0.
    fn ln_1p(self) -> Self;
    fn sqrt(self) -> Self;
    fn tanh(self) -> Self;
    fn powi(self, n: i32) -> Self;
}

impl Scalar for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn value(self) -> f64 {
        self
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn exp_m1(self) -> Self {
        f64::exp_m1(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        f64::ln_1p(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
}

/// Below this `|x|`, [`exprel_neg`] takes its series.
pub(crate) const EXPREL_SERIES: f64 = 1e-2;

/// `(1 - e^(-x)) / x`, with the limit 1 at `x = 0`: the kernel that keeps
/// the one-compartment closed form finite and smooth where the reporting
/// and payment rates coincide (decision 2 of the compartmental note).
///
/// For `|x| < 1e-2` it takes the series `sum_k (-x)^k / (k + 1)!` to
/// `k = 6`, whose value and derivative are both exact to rounding there
/// (the first omitted terms are below `3e-19` and `2e-16`, a few eps relative). The threshold
/// is set by the derivative, not the value: the quotient's dual derivative
/// `(e^(-x) - f) / x` loses about `2 eps / |x|` relative, 2e-14 at 1e-2 but
/// 3e-12 at the `1e-5` of a value-only threshold.
#[inline]
pub(crate) fn exprel_neg<S: Scalar>(x: S) -> S {
    if x.value().abs() < EXPREL_SERIES {
        // Horner on the coefficients (-1)^k / (k + 1)!.
        const C: [f64; 7] = [
            1.0,
            -1.0 / 2.0,
            1.0 / 6.0,
            -1.0 / 24.0,
            1.0 / 120.0,
            -1.0 / 720.0,
            1.0 / 5040.0,
        ];
        let mut acc = S::from_f64(C[6]);
        for &c in C[..6].iter().rev() {
            acc = acc * x + c;
        }
        acc
    } else {
        -(-x).exp_m1() / x
    }
}

/// `ln(1 + e^x)` without overflow.
#[inline]
pub(crate) fn softplus<S: Scalar>(x: S) -> S {
    if x.value() >= 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// `1 / (1 + e^(-x))` without overflow.
#[inline]
pub(crate) fn logistic<S: Scalar>(x: S) -> S {
    if x.value() >= 0.0 {
        S::from_f64(1.0) / ((-x).exp() + 1.0)
    } else {
        let e = x.exp();
        e / (e + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::dual::Dual;
    use super::*;

    /// The dual derivative of `f` at `x` (one tangent).
    fn derivative(f: impl Fn(Dual<1>) -> Dual<1>, x: f64) -> (f64, f64) {
        let y = f(Dual::variable(x, 0));
        (y.v, y.d[0])
    }

    fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
        let err = (a - b).abs() / b.abs().max(f64::MIN_POSITIVE);
        assert!(
            err <= tol,
            "{what}: dual {a}, analytic {b}, relative {err:e}"
        );
    }

    #[test]
    fn each_elementary_function_has_its_analytic_derivative() {
        type F = fn(Dual<1>) -> Dual<1>;
        type G = fn(f64) -> f64;
        let cases: [(&str, F, G, G, &[f64]); 7] = [
            (
                "exp",
                |x| x.exp(),
                f64::exp,
                f64::exp,
                &[-3.0, -1e-8, 0.0, 0.7, 20.0],
            ),
            (
                "exp_m1",
                |x| x.exp_m1(),
                f64::exp_m1,
                f64::exp,
                &[-3.0, -1e-9, 1e-12, 0.7, 20.0],
            ),
            (
                "ln",
                |x| x.ln(),
                f64::ln,
                |x| 1.0 / x,
                &[1e-6, 0.3, 1.0, 7.5, 1e8],
            ),
            (
                "ln_1p",
                |x| x.ln_1p(),
                f64::ln_1p,
                |x| 1.0 / (1.0 + x),
                &[-0.9, -1e-10, 0.0, 2.0, 1e6],
            ),
            (
                "sqrt",
                |x| x.sqrt(),
                f64::sqrt,
                |x| 0.5 / x.sqrt(),
                &[1e-8, 0.25, 2.0, 1e10],
            ),
            (
                "tanh",
                |x| x.tanh(),
                f64::tanh,
                |x| 1.0 / (x.cosh() * x.cosh()),
                &[-4.0, -1e-7, 0.0, 0.6, 3.0],
            ),
            (
                "powi 5",
                |x| x.powi(5),
                |x| x.powi(5),
                |x| 5.0 * x.powi(4),
                &[-1.3, -0.2, 0.4, 0.9, 1.7],
            ),
        ];
        for (name, f, value, slope, xs) in cases {
            for &x in xs {
                let (v, d) = derivative(f, x);
                assert_eq!(v, value(x), "{name} value at {x}");
                assert_close(d, slope(x), 1e-14, &format!("{name}'({x})"));
            }
        }
        // powi at 0 and 1, and negative powers.
        assert_eq!(derivative(|x| x.powi(0), 2.0), (1.0, 0.0));
        assert_eq!(derivative(|x| x.powi(1), 2.0), (2.0, 1.0));
        let (_, d) = derivative(|x| x.powi(-2), 1.5);
        assert_close(d, -2.0 / 1.5f64.powi(3), 1e-15, "powi -2");
    }

    #[test]
    fn the_operators_follow_the_product_and_quotient_rules() {
        let x = Dual::<2>::variable(1.7, 0);
        let y = Dual::<2>::variable(-0.6, 1);
        let f = (x * y - y / x + x * 3.0 - 2.0) / (y + 4.0) + (-x);
        // d/dx and d/dy of (xy - y/x + 3x - 2) / (y + 4) - x.
        let (xv, yv) = (1.7f64, -0.6f64);
        let num = xv * yv - yv / xv + 3.0 * xv - 2.0;
        let dx = (yv + yv / (xv * xv) + 3.0) / (yv + 4.0) - 1.0;
        let dy = (xv - 1.0 / xv) / (yv + 4.0) - num / ((yv + 4.0) * (yv + 4.0));
        assert_close(f.v, num / (yv + 4.0) - xv, 1e-15, "value");
        assert_close(f.d[0], dx, 1e-15, "d/dx");
        assert_close(f.d[1], dy, 1e-15, "d/dy");
        let mut acc = Dual::<2>::constant(1.0);
        acc += x;
        acc -= y;
        assert_eq!((acc.v, acc.d), (1.0 + 1.7 + 0.6, [1.0, -1.0]));
    }

    /// `(1 - e^-x)/x` and its derivative `(e^-x (1 + x) - 1) / x^2`: in
    /// closed form for `|x| >= 0.5`, and below as their power series
    /// `sum (-x)^k / (k + 1)!` and `sum k (-1)^k x^(k-1) / (k + 1)!` to 25
    /// terms, where the closed forms cancel.
    fn exprel_neg_exact(x: f64) -> (f64, f64) {
        if x.abs() >= 0.5 {
            return (-(-x).exp_m1() / x, ((-x).exp() * (1.0 + x) - 1.0) / (x * x));
        }
        let (mut value, mut slope, mut fact) = (0.0, 0.0, 1.0);
        for k in 0..25 {
            fact *= (k + 1) as f64;
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            value += sign * x.powi(k) / fact;
            if k > 0 {
                slope += sign * k as f64 * x.powi(k - 1) / fact;
            }
        }
        (value, slope)
    }

    #[test]
    fn exprel_neg_and_its_derivative_on_both_branches() {
        // The series branch, its edge from both sides, and the closed form.
        for x in [
            0.0,
            1e-12,
            -3e-9,
            4e-7,
            1e-5,
            -9.99e-3,
            9.999_99e-3,
            1e-2,
            -1.0001e-2,
            0.03,
            -0.2,
            0.5,
            -2.0,
            7.0,
            40.0,
        ] {
            let (v, d) = derivative(exprel_neg, x);
            let (value, slope) = exprel_neg_exact(x);
            assert_close(v, value, 1e-15, &format!("exprel_neg({x})"));
            assert_close(d, slope, 1e-13, &format!("exprel_neg'({x})"));
        }
    }

    #[test]
    fn softplus_and_logistic_on_both_branches() {
        for x in [-750.0, -30.0, -1.2, -1e-9, 0.0, 1e-9, 0.8, 35.0, 750.0] {
            let (v, d) = derivative(softplus, x);
            let exact = x.max(0.0) + (-x.abs()).exp().ln_1p();
            assert_close(v.max(1e-300), exact.max(1e-300), 1e-15, "softplus");
            // softplus' = logistic; logistic' = l (1 - l).
            let l = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                x.exp() / (1.0 + x.exp())
            };
            assert_close(d.max(1e-300), l.max(1e-300), 1e-15, "softplus'");
            let (v, d) = derivative(logistic, x);
            assert_close(v.max(1e-300), l.max(1e-300), 1e-15, "logistic");
            let lc = 1.0 - l;
            let slope = if x >= 0.0 {
                // 1 - l loses digits for large x; use e^-x l.
                (-x).exp() * l * l
            } else {
                l * lc
            };
            assert_close(d.max(1e-300), slope.max(1e-300), 1e-14, "logistic'");
        }
    }
}
