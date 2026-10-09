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
/// use prospicio_math::special::ln_gamma;
///
/// // Γ(5) = 4! = 24.
/// assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-14);
/// ```
pub fn ln_gamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// Digamma function `ψ(x) = d ln Γ(x) / dx` for `x > 0`: the recurrence
/// `ψ(x) = ψ(x + 1) - 1/x` up to `x ≥ 10`, then the asymptotic series.
///
/// ```
/// use prospicio_math::special::digamma;
///
/// // ψ(1) = -γ (Euler's constant).
/// assert!((digamma(1.0) + 0.5772156649015329).abs() < 1e-14);
/// ```
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let mut x = x;
    let mut acc = 0.0;
    while x < 10.0 {
        acc -= 1.0 / x;
        x += 1.0;
    }
    let f = 1.0 / (x * x);
    let series = f
        * (-1.0 / 12.0
            + f * (1.0 / 120.0
                + f * (-1.0 / 252.0
                    + f * (1.0 / 240.0 + f * (-1.0 / 132.0 + f * 691.0 / 32760.0)))));
    acc + x.ln() - 0.5 / x + series
}

/// Standard normal quantile (inverse of [`norm_cdf`]).
///
/// Returns `-inf` at 0, `+inf` at 1 and NaN outside `[0, 1]`. Starts from
/// Abramowitz & Stegun 26.2.23 (error below 4.5e-4) and polishes with Newton
/// steps on `ln Phi`, which keeps full relative accuracy deep in the tail.
///
/// ```
/// use prospicio_math::special::norm_quantile;
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
/// use prospicio_math::special::beta_inc;
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
/// use prospicio_math::special::student_t_cdf;
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

/// Regularized incomplete gamma functions `(P(a, x), Q(a, x))`, the lower
/// and upper parts, for shape `a > 0` and `x >= 0`: `P(a, x)` is the
/// distribution function of a unit-scale gamma with shape `a` at `x`, and
/// `Q = 1 - P`.
///
/// The smaller of the two is computed directly, by the power series below
/// `x = a + 1` and by Lentz's continued fraction above, so both tails keep
/// their relative precision; the other is `1` minus it. The common factor
/// `x^a e^(-x) / Γ(a)` is formed in Stirling form for `a >= 10`, which
/// keeps full precision for shapes in the thousands. NaN for invalid
/// arguments.
///
/// ```
/// use prospicio_math::special::gamma_inc;
///
/// // Shape 1 is the exponential: P(1, x) = 1 - e^(-x).
/// let (p, q) = gamma_inc(1.0, 2.0);
/// assert!((p - (1.0 - (-2f64).exp())).abs() < 1e-15);
/// assert!((q - (-2f64).exp()).abs() < 1e-16);
/// ```
pub fn gamma_inc(a: f64, x: f64) -> (f64, f64) {
    if a.is_nan() || x.is_nan() || a <= 0.0 || x < 0.0 || a == f64::INFINITY {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 1.0);
    }
    if x == f64::INFINITY {
        return (1.0, 0.0);
    }
    let factor = gamma_inc_factor(a, x).exp();
    if x < a + 1.0 {
        // P = factor × Σ_n x^n / (a (a+1) ⋯ (a+n)).
        let mut term = 1.0 / a;
        let mut sum = term;
        for n in 1..100_000 {
            term *= x / (a + f64::from(n));
            sum += term;
            if term <= sum * 1e-17 {
                break;
            }
        }
        let p = (factor * sum).min(1.0);
        (p, 1.0 - p)
    } else {
        // Q = factor / (x + 1 - a - 1(1 - a) / (x + 3 - a - 2(2 - a) / ⋯)).
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..100_000 {
            let an = -f64::from(i) * (f64::from(i) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() <= 1e-16 {
                break;
            }
        }
        let q = (factor * h).min(1.0);
        (1.0 - q, q)
    }
}

/// Euler's constant `γ`.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// The unnormalized lower incomplete beta
/// `B(a, b; x) = ∫_0^x t^(a-1) (1-t)^(b-1) dt` for `a > 0`, any `b`, and
/// `0 <= x < 1` (finite for every `b`, since `x < 1`).
///
/// For `b > 0` it is `I_x(a, b) B(a, b)`. Otherwise it steps `b` up to a
/// positive value (or zero) and back down by
/// `B(a, b; x) = ((a + b) B(a, b + 1; x) - x^a (1 - x)^b) / b`. These are
/// the limited moments of Pareto-tailed severities beyond the moments that
/// exist (loglogistic, Burr).
///
/// ```
/// use prospicio_math::special::beta_lower;
///
/// // B(1, 0; x) = -ln(1 - x) and B(1, -1; x) = x / (1 - x).
/// assert!((beta_lower(1.0, 0.0, 0.5) / 2f64.ln() - 1.0).abs() < 1e-14);
/// assert!((beta_lower(1.0, -1.0, 0.75) - 3.0).abs() < 1e-14);
/// ```
pub fn beta_lower(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if b > 0.0 {
        let ln_b = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
        return beta_inc(a, b, x) * ln_b.exp();
    }
    // The smallest c = b + n that is positive, or zero.
    let steps = (-b).floor();
    let c = b + steps;
    let (mut value, mut c) = if c == 0.0 {
        (beta_zero(a, x), 0.0)
    } else {
        let c = c + 1.0;
        let ln_b = ln_gamma(a) + ln_gamma(c) - ln_gamma(a + c);
        (beta_inc(a, c, x) * ln_b.exp(), c)
    };
    while c > b {
        c -= 1.0;
        value = ((a + c) * value - x.powf(a) * (c * (-x).ln_1p()).exp()) / c;
    }
    value
}

/// `B(a, 0; x) = ∫_0^x t^(a-1) / (1 - t) dt`: the series
/// `Σ x^(a+n) / (a+n)` below `x = 1/2`; above it, with `y = 1 - x`,
/// `-ln y - ψ(a) - γ - Σ_{k≥1} (-1)^k C(a-1, k) y^k / k`, from
/// `∫_0^1 (1 - t^(a-1)) / (1 - t) dt = ψ(a) + γ`. For integer `a` the sum
/// stops and this is `-ln(1 - x) - Σ_{k<a} x^k / k`.
fn beta_zero(a: f64, x: f64) -> f64 {
    if x < 0.5 {
        let mut sum = 0.0;
        let mut power = x.powf(a);
        for n in 0..4000 {
            let add = power / (a + f64::from(n));
            sum += add;
            if add < 1e-17 * sum {
                break;
            }
            power *= x;
        }
        sum
    } else {
        let y = 1.0 - x;
        let mut sum = 0.0;
        // c_k = (-1)^k C(a - 1, k) y^k.
        let mut c = 1.0;
        for k in 1..4000 {
            let kf = f64::from(k);
            c *= -(a - kf) / kf * y;
            let add = c / kf;
            sum += add;
            if c == 0.0 || add.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        -y.ln() - digamma(a) - EULER_GAMMA - sum
    }
}

/// The unnormalized upper incomplete gamma `Γ(s, z) = ∫_z^∞ t^(s-1) e^(-t) dt`
/// for any real `s` and `z > 0` (for `s <= 0` it is finite only because
/// `z > 0`). NaN for `z <= 0` with `s <= 0`.
///
/// For `s > 0` it is `Γ(s) Q(s, z)`. Otherwise it starts from `s + n` in
/// `(0, 1)`, or from the exponential integral `E_1(z) = Γ(0, z)` when `s`
/// is an integer, and steps down by `Γ(c, z) = (Γ(c + 1, z) - z^c e^(-z)) / c`.
/// These are the limited moments of the inverse gamma beyond the moments
/// that exist.
///
/// ```
/// use prospicio_math::special::gamma_upper;
///
/// // Γ(1, z) = e^(-z); Γ(-1, z) = e^(-z)/z - E_1(z).
/// assert!((gamma_upper(1.0, 2.0) - (-2f64).exp()).abs() < 1e-16);
/// let e1 = gamma_upper(0.0, 2.0);
/// assert!((gamma_upper(-1.0, 2.0) - ((-2f64).exp() / 2.0 - e1)).abs() < 1e-15);
/// ```
pub fn gamma_upper(s: f64, z: f64) -> f64 {
    if s.is_nan() || z.is_nan() || z < 0.0 {
        return f64::NAN;
    }
    if s > 0.0 {
        return ln_gamma(s).exp() * gamma_inc(s, z).1;
    }
    if z == 0.0 {
        return f64::NAN;
    }
    let steps = (-s).floor();
    let c = s + steps;
    let (mut value, mut c) = if c == 0.0 {
        (expint_e1(z), 0.0)
    } else {
        let c = c + 1.0;
        (ln_gamma(c).exp() * gamma_inc(c, z).1, c)
    };
    while c > s {
        c -= 1.0;
        value = (value - (c * z.ln() - z).exp()) / c;
    }
    value
}

/// The exponential integral `E_1(z) = ∫_z^∞ e^(-t) / t dt` for `z > 0`:
/// the power series up to `z = 1`, Lentz's continued fraction above.
///
/// ```
/// use prospicio_math::special::expint_e1;
///
/// // SciPy: scipy.special.exp1(1.0).
/// assert!((expint_e1(1.0) / 0.21938393439552029 - 1.0).abs() < 1e-14);
/// ```
pub fn expint_e1(z: f64) -> f64 {
    if z.is_nan() || z < 0.0 {
        return f64::NAN;
    }
    if z == 0.0 {
        return f64::INFINITY;
    }
    if z == f64::INFINITY {
        return 0.0;
    }
    if z <= 1.0 {
        // -γ - ln z - Σ_{k≥1} (-z)^k / (k k!).
        let mut sum = 0.0;
        let mut term = 1.0;
        for k in 1..200 {
            let kf = f64::from(k);
            term *= -z / kf;
            let add = term / kf;
            sum += add;
            if add.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        -EULER_GAMMA - z.ln() - sum
    } else {
        const TINY: f64 = 1e-300;
        let mut b = z + 1.0;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -f64::from(i) * f64::from(i);
            b += 2.0;
            d = 1.0 / (an * d + b);
            c = b + an / c;
            let delta = c * d;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        h * (-z).exp()
    }
}

/// Mills ratio `R(t) = Φ(-t) / φ(t)` for `t >= 0`, the normal tail
/// relative to the density, finite where both underflow: directly below
/// `t = 10`, by the continued fraction `1 / (t + 1 / (t + 2 / (t + ⋯)))`
/// above.
///
/// ```
/// use prospicio_math::special::{mills_ratio, norm_cdf, norm_pdf};
///
/// assert!((mills_ratio(1.0) - norm_cdf(-1.0) / norm_pdf(1.0)).abs() < 1e-15);
/// // R(t) ~ 1/t far in the tail.
/// assert!((mills_ratio(1e6) * 1e6 - 1.0).abs() < 1e-11);
/// ```
pub fn mills_ratio(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    if t < 10.0 {
        return norm_cdf(-t) / norm_pdf(t);
    }
    if t == f64::INFINITY {
        return 0.0;
    }
    // Backward evaluation of the continued fraction; 200 levels are far
    // more than t >= 10 needs.
    let mut v = 0.0;
    for k in (1..=200).rev() {
        v = f64::from(k) / (t + v);
    }
    1.0 / (t + v)
}

/// `ln(x^a e^(-x) / Γ(a))`. For `a >= 10`, as
/// `a (ln(1 + u) - u) + ln(a / 2π) / 2 - c(a)` with `u = (x - a) / a` and
/// `c` the Stirling series remainder of `ln Γ(a)`, so the large, nearly
/// cancelling terms `a ln x`, `x` and `ln Γ(a)` are never formed.
fn gamma_inc_factor(a: f64, x: f64) -> f64 {
    if a < 10.0 {
        return a * x.ln() - x - ln_gamma(a);
    }
    let u = (x - a) / a;
    let log1pmx = if u.abs() < 0.5 {
        // ln(1 + u) - u = Σ_{k≥2} (-1)^(k+1) u^k / k.
        let mut sum = 0.0;
        let mut power = u * u;
        for k in 2..200 {
            let term = power / f64::from(k);
            sum += if k % 2 == 0 { -term } else { term };
            if term.abs() <= 1e-17 * sum.abs() {
                break;
            }
            power *= u;
        }
        sum
    } else {
        // ln(x / a) directly: 1 + u would lose x's precision for x << a.
        (x / a).ln() - u
    };
    let inv = 1.0 / a;
    let inv2 = inv * inv;
    // ln Γ(a) - ((a - 1/2) ln a - a + ln(2π)/2), to about 1e-17 at a = 10.
    let stirling_remainder = inv
        * (1.0 / 12.0
            - inv2
                * (1.0 / 360.0
                    - inv2
                        * (1.0 / 1260.0
                            - inv2
                                * (1.0 / 1680.0
                                    - inv2
                                        * (1.0 / 1188.0
                                            - inv2 * (691.0 / 360_360.0 - inv2 / 156.0))))));
    a * log1pmx + 0.5 * (a / (2.0 * PI)).ln() - stirling_remainder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digamma_matches_the_derivative_of_ln_gamma() {
        // ψ(1/2) = -γ - 2 ln 2; ψ(n) = H_{n-1} - γ.
        let gamma = 0.5772156649015329;
        assert!((digamma(0.5) + gamma + 2.0 * 2f64.ln()).abs() < 1e-14);
        assert!((digamma(10.0) - (7129.0 / 2520.0 - gamma)).abs() < 1e-14);
        for x in [0.01, 0.3, 2.7, 45.0, 1e4] {
            let h = 1e-5 * x;
            let numeric = (ln_gamma(x + h) - ln_gamma(x - h)) / (2.0 * h);
            assert!(
                (digamma(x) - numeric).abs() < 1e-7 * numeric.abs().max(1.0),
                "{x}"
            );
        }
        assert!(digamma(0.0).is_nan());
    }

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

    /// mpmath 1.4.1 at 30 digits: `quad` of the integrand for `B(a, b; x)`,
    /// `gammainc(s, z)` for `Γ(s, z)`, `e1(z)` for `E_1(z)`.
    #[test]
    fn beta_lower_matches_mpmath() {
        for (a, b, x, want) in [
            (0.7, -2.0, 0.6, 3.598744771047808),
            (1.5, 0.5, 0.7, 0.5328990169356083),
            (2.0, -1.0, 0.9, 6.697414907005956),
            (2.25, -0.25, 0.4, 0.08655486451119004),
            (3.0, -1.0, 0.2, 0.003712897371580489),
            (2.5, 0.0, 0.3, 0.02525438280078752),
            (2.5, 0.0, 0.8, 0.6213887331578339),
            (3.0, 0.0, 0.9, 0.9975850929940459),
        ] {
            let got = beta_lower(a, b, x);
            assert!((got / want - 1.0).abs() < 1e-13, "{a} {b} {x}: {got}");
        }
    }

    #[test]
    fn gamma_upper_matches_mpmath() {
        assert!((expint_e1(0.1) / 1.8229239584193908 - 1.0).abs() < 1e-14);
        assert!((expint_e1(5.0) / 0.0011482955912753257 - 1.0).abs() < 1e-14);
        assert!((expint_e1(50.0) / 3.783264029550459e-24 - 1.0).abs() < 1e-13);
        for (s, z, want) in [
            (-0.5, 0.3, 1.1503670473551644),
            (-1.5, 2.0, 0.011832994103345996),
            (-2.0, 0.7, 0.3389003309406555),
            (0.4, 1.5, 0.13628632343383457),
            (-0.25, 4.0, 0.0025577114691076545),
        ] {
            let got = gamma_upper(s, z);
            assert!((got / want - 1.0).abs() < 1e-13, "{s} {z}: {got}");
        }
    }

    #[test]
    fn mills_ratio_is_continuous_at_the_switch() {
        let below = norm_cdf(-10.0) / norm_pdf(10.0);
        assert!((mills_ratio(10.0) / below - 1.0).abs() < 1e-13);
        assert!((mills_ratio(0.0) - (PI / 2.0).sqrt()).abs() < 1e-15);
    }
}
