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
/// use act_math::integrate::gauss_legendre;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrates_smooth_functions() {
        let v = gauss_legendre(|x| Ok::<_, ()>(x.exp()), 0.0, 1.0).unwrap();
        assert!((v - (1f64.exp() - 1.0)).abs() < 1e-15);
    }

    #[test]
    fn passes_errors_through() {
        let v = gauss_legendre(|x| if x > 0.5 { Err("bad") } else { Ok(x) }, 0.0, 1.0);
        assert_eq!(v, Err("bad"));
    }
}
