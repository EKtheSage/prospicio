//! Numerical integration.

/// Nodes in `(0, 1)` and weights of the 8-point Gauss–Legendre rule on
/// `[-1, 1]`; the rule is symmetric, so each node `x` stands for `±x`.
const GL8_X: [f64; 4] = [
    0.183_434_642_495_649_8,
    0.525_532_409_916_329,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];
const GL8_W: [f64; 4] = [
    0.362_683_783_378_362,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_5,
    0.101_228_536_290_376_3,
];

/// `∫_lo^hi f` by the 8-point Gauss–Legendre rule, exact for polynomials
/// of degree up to 15. `f` may fail; the first error is returned.
///
/// One panel, no error estimate: callers split the range into panels
/// where `f` is smooth (between thresholds, for instance) and refine
/// until the result stops moving.
///
/// ```
/// use prospicio_math::integrate::gauss_legendre;
///
/// let v = gauss_legendre(|x| Ok::<_, ()>(x.powi(15)), 0.0, 1.0).unwrap();
/// assert!((v - 1.0 / 16.0).abs() < 1e-15);
/// ```
pub fn gauss_legendre<E>(
    mut f: impl FnMut(f64) -> Result<f64, E>,
    lo: f64,
    hi: f64,
) -> Result<f64, E> {
    let (mid, half) = (0.5 * (lo + hi), 0.5 * (hi - lo));
    let mut sum = 0.0;
    for (x, w) in GL8_X.iter().zip(GL8_W) {
        sum += w * (f(mid - half * x)? + f(mid + half * x)?);
    }
    Ok(sum * half)
}

/// The most panels [`adaptive_gauss_legendre`] splits, so that an
/// integrand whose rounding noise never meets the tolerance (values near
/// the subnormal range, say) costs a bounded number of evaluations.
pub const MAX_PANELS: usize = 1 << 14;

/// `∫_lo^hi f` by adaptive Gauss–Legendre: a panel whose 8-point value
/// differs from the sum over its two halves by more than its share of
/// `tol` (absolute) is split, down to `max_depth` halvings and at most
/// [`MAX_PANELS`] splits in all; the halves' sums are added. For
/// integrands that are smooth except for a sharp
/// transition somewhere inside, such as the stable law's integral
/// representation, whose integrand runs from 0 to 1 across a narrow band.
///
/// ```
/// use prospicio_math::integrate::adaptive_gauss_legendre;
///
/// let v = adaptive_gauss_legendre(|x| (-1e4 * (x - 0.3) * (x - 0.3)).exp(), 0.0, 1.0, 1e-14, 40);
/// assert!((v - (std::f64::consts::PI / 1e4).sqrt()).abs() < 1e-13);
/// ```
pub fn adaptive_gauss_legendre(
    mut f: impl FnMut(f64) -> f64,
    lo: f64,
    hi: f64,
    tol: f64,
    max_depth: u32,
) -> f64 {
    let mut rule = |a: f64, b: f64| {
        let Ok(v) = gauss_legendre(|x| Ok::<_, std::convert::Infallible>(f(x)), a, b);
        v
    };
    let width = hi - lo;
    if width <= 0.0 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut splits = 0usize;
    let mut stack = vec![(lo, hi, rule(lo, hi), 0u32)];
    while let Some((a, b, whole, depth)) = stack.pop() {
        let mid = 0.5 * (a + b);
        let (left, right) = (rule(a, mid), rule(mid, b));
        let share = tol * (b - a) / width;
        if depth >= max_depth || splits >= MAX_PANELS || (left + right - whole).abs() <= share {
            total += left + right;
        } else {
            splits += 1;
            stack.push((mid, b, right, depth + 1));
            stack.push((a, mid, left, depth + 1));
        }
    }
    total
}

/// The `n`-point Gauss–Hermite rule for `∫ e^(-x²) f(x) dx`: the nodes `x`,
/// in descending order, and `log w + x²` for each node's weight `w`, the
/// scale on which a caller adds the log of `e^(x²) f(x)` without
/// underflow. Found by Newton's method on the orthonormal Hermite
/// recurrence (Press et al., *Numerical Recipes*, 4.6); exact for
/// polynomials of degree up to `2n - 1`.
///
/// ```
/// use prospicio_math::integrate::gauss_hermite;
///
/// let (x, lw) = gauss_hermite(5);
/// // ∫ e^(-x²) x² dx = √π / 2.
/// let m2: f64 = x.iter().zip(&lw).map(|(x, l)| (l - x * x).exp() * x * x).sum();
/// assert!((m2 - std::f64::consts::PI.sqrt() / 2.0).abs() < 1e-13);
/// ```
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let pim4 = std::f64::consts::PI.powf(-0.25);
    let nf = n as f64;
    let mut z = 0.0f64;
    for i in 0..n.div_ceil(2) {
        z = match i {
            0 => (2.0 * nf + 1.0).sqrt() - 1.85575 * (2.0 * nf + 1.0).powf(-1.0 / 6.0),
            1 => z - 1.14 * nf.powf(0.426) / z,
            2 => 1.86 * z - 0.86 * x[0],
            3 => 1.91 * z - 0.91 * x[1],
            _ => 2.0 * z - x[i - 2],
        };
        let mut pp = 0.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (pim4, 0.0);
            for j in 1..=n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = z * (2.0 / jf).sqrt() * p2 - ((jf - 1.0) / jf).sqrt() * p3;
            }
            pp = (2.0 * nf).sqrt() * p2;
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() <= 1e-15 * z.abs().max(1.0) {
                break;
            }
        }
        x[i] = z;
        x[n - 1 - i] = -z;
        w[i] = 2.0 / (pp * pp);
        w[n - 1 - i] = w[i];
    }
    let lw = x.iter().zip(&w).map(|(xi, wi)| wi.ln() + xi * xi).collect();
    (x, lw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_hermite_integrates_polynomials() {
        let (x, lw) = gauss_hermite(7);
        let w: Vec<f64> = x
            .iter()
            .zip(&lw)
            .map(|(xi, l)| (l - xi * xi).exp())
            .collect();
        let pi = std::f64::consts::PI;
        let m = |k: i32| {
            x.iter()
                .zip(&w)
                .map(|(xi, wi)| wi * xi.powi(k))
                .sum::<f64>()
        };
        assert!((m(0) - pi.sqrt()).abs() < 1e-13);
        assert!((m(2) - pi.sqrt() / 2.0).abs() < 1e-13);
        assert!((m(4) - 3.0 * pi.sqrt() / 4.0).abs() < 1e-12);
        assert!((m(12) - 10395.0 * pi.sqrt() / 64.0).abs() < 1e-9);
    }

    #[test]
    fn integrates_smooth_functions() {
        let v = gauss_legendre(|x| Ok::<_, ()>(x.exp()), 0.0, 1.0).unwrap();
        assert!((v - (1f64.exp() - 1.0)).abs() < 1e-15);
    }

    #[test]
    fn adapts_to_a_sharp_step() {
        // A logistic step of width 1e-6 at 0.4: its integral over [0, 1]
        // is 0.6 to within 1e-12.
        let v = adaptive_gauss_legendre(
            |x| 1.0 / (1.0 + (-(x - 0.4) / 1e-6).exp()),
            0.0,
            1.0,
            1e-13,
            60,
        );
        assert!((v - 0.6).abs() < 1e-11, "{v}");
        assert_eq!(adaptive_gauss_legendre(|x| x, 1.0, 1.0, 1e-12, 10), 0.0);
    }

    #[test]
    fn passes_errors_through() {
        let v = gauss_legendre(|x| if x > 0.5 { Err("bad") } else { Ok(x) }, 0.0, 1.0);
        assert_eq!(v, Err("bad"));
    }
}
