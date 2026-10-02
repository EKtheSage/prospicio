//! Special functions.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// Standard normal density.
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal distribution function, accurate in both tails.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x * FRAC_1_SQRT_2)
}

/// Natural log of the gamma function, `ln Γ(x)` for `x > 0`.
///
/// ```
/// use act_math::special::ln_gamma;
///
/// // Γ(5) = 4! = 24.
/// assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-14);
/// ```
pub fn ln_gamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// Standard normal quantile (inverse of [`norm_cdf`]).
///
/// Returns `-inf` at 0, `+inf` at 1 and NaN outside `[0, 1]`. Starts from
/// Abramowitz & Stegun 26.2.23 (error below 4.5e-4) and polishes with Newton
/// steps on `ln Phi`, which keeps full relative accuracy deep in the tail.
///
/// ```
/// use act_math::special::norm_quantile;
///
/// assert!((norm_quantile(0.975) - 1.959963984540054).abs() < 1e-14);
/// ```
pub fn norm_quantile(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p > 0.5 {
        return -lower_quantile(1.0 - p);
    }
    lower_quantile(p)
}

/// Quantile for `0 < p <= 0.5`.
fn lower_quantile(p: f64) -> f64 {
    let t = (-2.0 * p.ln()).sqrt();
    let mut x = -(t
        - (2.515517 + 0.802853 * t + 0.010328 * t * t)
            / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t));
    let target = p.ln();
    for _ in 0..8 {
        let cdf = norm_cdf(x);
        let step = (cdf.ln() - target) * cdf / norm_pdf(x);
        x -= step;
        if step.abs() <= 1e-15 * x.abs().max(1.0) {
            break;
        }
    }
    x
}

/// Regularized incomplete beta function `I_x(a, b)` for `a, b > 0` and
/// `x` in `[0, 1]`; NaN outside that domain.
///
/// Evaluated by the continued fraction for `I_x(a, b)` (modified Lentz),
/// switching to `1 - I_{1-x}(b, a)` where that converges faster; relative
/// accuracy is about `1e-14` away from underflow.
///
/// ```
/// use act_math::special::beta_inc;
///
/// // I_x(1, 1) = x and I_x(a, 1) = x^a.
/// assert!((beta_inc(1.0, 1.0, 0.3) - 0.3).abs() < 1e-15);
/// assert!((beta_inc(2.5, 1.0, 0.3) - 0.3f64.powf(2.5)).abs() < 1e-15);
/// ```
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if !(a > 0.0 && b > 0.0) || x.is_nan() || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 || x == 1.0 {
        return x;
    }
    // x^a (1 - x)^b / (a B(a, b)), in logs.
    let ln_front = a * x.ln() + b * (-x).ln_1p() + ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    if x < (a + 1.0) / (a + b + 2.0) {
        (ln_front.exp() / a) * beta_cf(a, b, x)
    } else {
        1.0 - (ln_front.exp() / b) * beta_cf(b, a, 1.0 - x)
    }
}

/// Continued fraction for the incomplete beta (Numerical Recipes `betacf`),
/// by the modified Lentz method.
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=10_000 {
        let m = f64::from(m);
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Student's t distribution function with `nu > 0` degrees of freedom.
///
/// Uses `P(T > |t|) = I_{nu / (nu + t^2)}(nu / 2, 1 / 2) / 2`, so both
/// tails keep full relative accuracy.
///
/// ```
/// use act_math::special::student_t_cdf;
///
/// // With one degree of freedom, t is Cauchy: F(1) = 3/4.
/// assert!((student_t_cdf(1.0, 1.0) - 0.75).abs() < 1e-15);
/// ```
pub fn student_t_cdf(t: f64, nu: f64) -> f64 {
    if t.is_nan() || nu.is_nan() || nu <= 0.0 {
        return f64::NAN;
    }
    if t.is_infinite() {
        return if t > 0.0 { 1.0 } else { 0.0 };
    }
    let x = nu / (nu + t * t);
    let tail = 0.5 * beta_inc(0.5 * nu, 0.5, x);
    if t > 0.0 { 1.0 - tail } else { tail }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_inverts_cdf() {
        for &p in &[
            1e-300,
            1e-100,
            1e-20,
            1e-8,
            0.001,
            0.1,
            0.3,
            0.5,
            0.7,
            0.99,
            1.0 - 1e-12,
        ] {
            let x = norm_quantile(p);
            let back = norm_cdf(x);
            assert!(
                ((back - p) / p).abs() < 1e-13,
                "p = {p}, x = {x}, back = {back}"
            );
        }
    }

    #[test]
    fn quantile_known_values() {
        // SciPy 1.x: scipy.stats.norm.ppf.
        let cases = [
            (0.5, 0.0),
            (0.995, 2.5758293035489004),
            (0.001, -3.090232306167813),
            (1e-10, -6.361340902404056),
        ];
        for (p, want) in cases {
            let got = norm_quantile(p);
            assert!(
                (got - want).abs() < 1e-13,
                "p = {p}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn quantile_edges() {
        assert_eq!(norm_quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(norm_quantile(1.0), f64::INFINITY);
        assert!(norm_quantile(1.5).is_nan());
        assert!(norm_quantile(f64::NAN).is_nan());
    }

    #[test]
    fn beta_inc_closed_forms_and_symmetry() {
        for &x in &[1e-10, 0.01, 0.2, 0.5, 0.8, 0.999] {
            // I_x(a, b) = 1 - I_{1-x}(b, a); skipped where rounding 1 - x
            // alone moves the result by more than the tolerance.
            for &(a, b) in &[(0.5, 0.5), (2.0, 3.0), (10.0, 0.7), (50.0, 40.0)] {
                if x < 1e-3 {
                    continue;
                }
                let s = beta_inc(a, b, x) + beta_inc(b, a, 1.0 - x);
                assert!((s - 1.0).abs() < 1e-13, "a = {a}, b = {b}, x = {x}");
            }
            // I_x(1/2, 1/2) = (2 / pi) asin(sqrt(x)).
            let arcsine = 2.0 / PI * x.sqrt().asin();
            assert!((beta_inc(0.5, 0.5, x) - arcsine).abs() < 1e-14 * arcsine.max(1e-300));
        }
        assert!(beta_inc(0.0, 1.0, 0.5).is_nan());
        assert!(beta_inc(1.0, 1.0, 1.5).is_nan());
        assert_eq!(beta_inc(2.0, 3.0, 0.0), 0.0);
        assert_eq!(beta_inc(2.0, 3.0, 1.0), 1.0);
    }

    #[test]
    fn student_t_limits() {
        // nu = 2 has a closed form: F(t) = 1/2 + t / (2 sqrt(2 + t^2)).
        for &t in &[-30.0, -2.0, -0.1, 0.0, 0.7, 5.0] {
            let want = 0.5 + t / (2.0 * (2.0f64 + t * t).sqrt());
            assert!((student_t_cdf(t, 2.0) - want).abs() < 1e-15, "t = {t}");
        }
        // Large nu approaches the normal.
        assert!((student_t_cdf(1.3, 1e7) - norm_cdf(1.3)).abs() < 1e-7);
        assert_eq!(student_t_cdf(f64::INFINITY, 3.0), 1.0);
        assert_eq!(student_t_cdf(f64::NEG_INFINITY, 3.0), 0.0);
        assert!(student_t_cdf(1.0, 0.0).is_nan());
    }
}
