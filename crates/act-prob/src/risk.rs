//! Risk measures on sorted draws, and the exponential-utility measures
//! (entropic, Esscher) on any draws.
//!
//! These are the single implementation every domain uses: reserving,
//! aggregate and capital call them instead of computing their own
//! quantiles (see `docs/design/predictive-distribution.md`).
//!
//! Both measures treat the draws as an empirical distribution with mass
//! `1/n` on each draw, so ties and atoms are handled exactly.

use act_core::{Error, Result};

use crate::distribution::check_probability;

/// Value at risk: the smallest draw `x` with `P(X <= x) >= p`, the inverse
/// of the empirical distribution function (`numpy.quantile(...,
/// method="inverted_cdf")`, R `quantile(..., type = 1)`).
///
/// `p = 0` gives the smallest draw. `sorted` must be non-empty and sorted
/// ascending; this is checked only in debug builds.
///
/// ```
/// use act_prob::risk::var_sorted;
///
/// let x = [1.0, 2.0, 3.0, 4.0];
/// assert_eq!(var_sorted(&x, 0.5), Ok(2.0));
/// assert_eq!(var_sorted(&x, 0.51), Ok(3.0));
/// ```
pub fn var_sorted(sorted: &[f64], p: f64) -> Result<f64> {
    check_probability(p)?;
    debug_assert!(is_sorted(sorted), "draws must be sorted ascending");
    Ok(sorted[rank(sorted.len(), p) - 1])
}

/// Tail value at risk: the mean of the worst `1 - p` of the distribution,
/// `(1 / (1 - p)) * integral from p to 1 of VaR(u) du`.
///
/// When `p * n` is not a whole number the draw at the VaR is counted with
/// the fractional weight that falls above `p`, so the result is coherent
/// and moves continuously with `p`. `p = 0` gives the mean and `p = 1` the
/// largest draw. `sorted` must be non-empty and sorted ascending; this is
/// checked only in debug builds.
///
/// ```
/// use act_prob::risk::tvar_sorted;
///
/// let x = [1.0, 2.0, 3.0, 4.0];
/// assert_eq!(tvar_sorted(&x, 0.5), Ok(3.5));
/// // Half of the draw 2 lies above p = 0.375: (0.5 * 2 + 3 + 4) / 2.5.
/// assert_eq!(tvar_sorted(&x, 0.375), Ok(3.2));
/// ```
pub fn tvar_sorted(sorted: &[f64], p: f64) -> Result<f64> {
    check_probability(p)?;
    debug_assert!(is_sorted(sorted), "draws must be sorted ascending");
    let n = sorted.len();
    if p == 1.0 {
        return Ok(sorted[n - 1]);
    }
    let k = rank(n, p);
    let nf = n as f64;
    // Mass of draw k (1-based) that lies above p, then every draw after it.
    let partial = (k as f64 / nf - p) * sorted[k - 1];
    let tail: f64 = sorted[k..].iter().sum::<f64>() / nf;
    Ok((partial + tail) / (1.0 - p))
}

/// Entropic risk measure `(1/θ) log E[e^(θX)]` of equally likely draws
/// (losses positive), the certainty equivalent under exponential utility
/// with risk aversion `θ > 0`. It increases from the mean (`θ → 0`) to the
/// largest draw (`θ → ∞`); for a normal `X` it is `μ + θσ²/2`. Convex and
/// translation-invariant, but not positively homogeneous.
///
/// ```
/// use act_prob::risk::entropic;
///
/// // Two equally likely losses, 0 and 1, at θ = ln 2: log2((1 + 2)/2).
/// let r = entropic(&[0.0, 1.0], 2f64.ln()).unwrap();
/// assert!((r - 1.5f64.log2()).abs() < 1e-15);
/// ```
pub fn entropic(draws: &[f64], theta: f64) -> Result<f64> {
    check_draws(draws)?;
    if !(theta.is_finite() && theta > 0.0) {
        return Err(Error::InvalidParameter {
            name: "theta",
            value: theta,
            reason: "must be finite and positive",
        });
    }
    let m = draws.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_exp = draws.iter().map(|x| (theta * (x - m)).exp()).sum::<f64>() / draws.len() as f64;
    Ok(m + mean_exp.ln() / theta)
}

/// Esscher premium `E[X e^(hX)] / E[e^(hX)]` of equally likely draws: the
/// mean under the Esscher transform, which tilts probability towards large
/// losses for `h > 0` (the mean at `h = 0`; `μ + hσ²` for a normal `X`).
///
/// ```
/// use act_prob::risk::esscher;
///
/// // Draws 0 and 1 at h = ln 3: weights 1 and 3, so 3/4.
/// assert!((esscher(&[0.0, 1.0], 3f64.ln()).unwrap() - 0.75).abs() < 1e-15);
/// ```
pub fn esscher(draws: &[f64], h: f64) -> Result<f64> {
    check_draws(draws)?;
    if !h.is_finite() {
        return Err(Error::InvalidParameter {
            name: "h",
            value: h,
            reason: "must be finite",
        });
    }
    let w = esscher_weights(draws, h);
    Ok(w.iter().zip(draws).map(|(w, x)| w * x).sum())
}

/// Normalized Esscher weights `e^(h xᵢ) / Σ e^(h xⱼ)`, without overflow.
pub(crate) fn esscher_weights(draws: &[f64], h: f64) -> Vec<f64> {
    let m = draws
        .iter()
        .map(|x| h * x)
        .fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = draws.iter().map(|x| (h * x - m).exp()).collect();
    let total: f64 = w.iter().sum();
    w.into_iter().map(|v| v / total).collect()
}

fn check_draws(draws: &[f64]) -> Result<()> {
    if draws.is_empty() {
        return Err(Error::InvalidParameter {
            name: "draws",
            value: 0.0,
            reason: "must not be empty",
        });
    }
    Ok(())
}

/// Smallest `k` in `1..=n` with `k / n >= p`, using the same division as
/// the empirical distribution function so the two agree exactly.
fn rank(n: usize, p: f64) -> usize {
    let nf = n as f64;
    let mut k = ((p * nf).ceil() as usize).clamp(1, n);
    while k > 1 && (k - 1) as f64 / nf >= p {
        k -= 1;
    }
    while k < n && (k as f64 / nf) < p {
        k += 1;
    }
    k
}

fn is_sorted(x: &[f64]) -> bool {
    x.windows(2).all(|w| w[0] <= w[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Normal draws at plotting positions: entropic `μ + θσ²/2`, Esscher
    /// `μ + hσ²`.
    #[test]
    fn exponential_utility_measures_match_the_normal() {
        let n = 200_000;
        let (mu, sigma) = (3.0, 2.0);
        let x: Vec<f64> = (0..n)
            .map(|i| mu + sigma * act_math::special::norm_quantile((i as f64 + 0.5) / n as f64))
            .collect();
        let theta = 0.4;
        assert!((entropic(&x, theta).unwrap() - (mu + theta * sigma * sigma / 2.0)).abs() < 2e-3);
        let h = 0.3;
        assert!((esscher(&x, h).unwrap() - (mu + h * sigma * sigma)).abs() < 2e-3);
        assert!((esscher(&x, 0.0).unwrap() - mu).abs() < 1e-9);
        // Huge draws do not overflow.
        assert!((entropic(&[1e6, 1e6], 1.0).unwrap() - 1e6).abs() < 1e-6);
        assert!(entropic(&x, 0.0).is_err() && esscher(&[], 1.0).is_err());
    }
    use act_core::Error;

    const X: [f64; 5] = [10.0, 20.0, 30.0, 40.0, 50.0];

    #[test]
    fn var_steps_at_multiples_of_one_over_n() {
        assert_eq!(var_sorted(&X, 0.0), Ok(10.0));
        assert_eq!(var_sorted(&X, 0.2), Ok(10.0));
        assert_eq!(var_sorted(&X, 0.2000001), Ok(20.0));
        assert_eq!(var_sorted(&X, 0.6), Ok(30.0));
        assert_eq!(var_sorted(&X, 1.0), Ok(50.0));
    }

    #[test]
    fn rank_is_exact_where_p_times_n_rounds_up() {
        // 0.95 * 20 = 19.000000000000004 in floating point.
        assert_eq!(rank(20, 0.95), 19);
        // 0.7 * 10 = 7.000000000000001.
        assert_eq!(rank(10, 0.7), 7);
        assert_eq!(rank(3, 1.0 / 3.0), 1);
    }

    fn assert_close(got: f64, want: f64) {
        assert!((got - want).abs() <= 1e-12 * want.abs(), "{got} != {want}");
    }

    #[test]
    fn tvar_edges_and_tail_means() {
        assert_close(tvar_sorted(&X, 0.0).unwrap(), 30.0);
        assert_close(tvar_sorted(&X, 0.6).unwrap(), 45.0);
        assert_close(tvar_sorted(&X, 0.8).unwrap(), 50.0);
        assert_eq!(tvar_sorted(&X, 1.0), Ok(50.0));
        // (0.1 * 40 + 0.2 * 50) / 0.3.
        assert_close(tvar_sorted(&X, 0.7).unwrap(), 14.0 / 0.3);
    }

    #[test]
    fn tvar_is_at_least_var_and_continuous() {
        let mut prev = tvar_sorted(&X, 0.0).unwrap();
        for i in 1..=1000 {
            let p = i as f64 / 1000.0;
            let t = tvar_sorted(&X, p).unwrap();
            assert!(t >= var_sorted(&X, p).unwrap() - 1e-12);
            assert!(t >= prev - 1e-12, "not monotone at {p}");
            assert!(t - prev < 0.6, "jump at {p}");
            prev = t;
        }
    }

    #[test]
    fn ties_count_as_one_atom() {
        let x = [1.0, 5.0, 5.0, 5.0];
        assert_eq!(var_sorted(&x, 0.3), Ok(5.0));
        // 0.15 of the mass at 1 lies above p = 0.1, all of the mass at 5.
        assert_close(tvar_sorted(&x, 0.1).unwrap(), (0.15 + 3.75) / 0.9);
    }

    #[test]
    fn rejects_bad_probability() {
        assert_eq!(var_sorted(&X, -0.1), Err(Error::InvalidProbability(-0.1)));
        assert!(tvar_sorted(&X, f64::NAN).is_err());
    }
}
