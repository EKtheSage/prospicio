//! Risk measures on sorted draws, and the exponential-utility measures
//! (entropic, Esscher) on any draws.
//!
//! These are the single implementation every domain uses: reserving,
//! aggregate and capital call them instead of computing their own
//! quantiles (see `docs/design/predictive-distribution.md`).
//!
//! Both measures treat the draws as an empirical distribution with mass
//! `1/n` on each draw, so ties and atoms are handled exactly.

use prospicio_core::{Error, Result};

use crate::distribution::check_probability;

/// Value at risk: the smallest draw `x` with `P(X <= x) >= p`, the inverse
/// of the empirical distribution function (`numpy.quantile(...,
/// method="inverted_cdf")`, R `quantile(..., type = 1)`).
///
/// `p = 0` gives the smallest draw. `sorted` must be non-empty and sorted
/// ascending; this is checked only in debug builds.
///
/// ```
/// use prospicio_prob::risk::var_sorted;
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
/// use prospicio_prob::risk::tvar_sorted;
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

/// Value at risk by linear interpolation of the order statistics: with
/// `h = p (n - 1)`, `x[⌊h⌋] + (h - ⌊h⌋) (x[⌊h⌋ + 1] - x[⌊h⌋])`
/// (`numpy.quantile(..., method="linear")`, R `quantile(..., type = 7)`).
///
/// `p = 0` gives the smallest draw and `p = 1` the largest. `sorted` must
/// be non-empty and sorted ascending; this is checked only in debug builds.
///
/// ```
/// use prospicio_prob::risk::var_linear_sorted;
///
/// let x = [1.0, 2.0, 3.0, 4.0];
/// assert_eq!(var_linear_sorted(&x, 0.5), Ok(2.5));
/// assert_eq!(var_linear_sorted(&x, 1.0), Ok(4.0));
/// ```
pub fn var_linear_sorted(sorted: &[f64], p: f64) -> Result<f64> {
    check_probability(p)?;
    debug_assert!(is_sorted(sorted), "draws must be sorted ascending");
    let n = sorted.len();
    let h = p * (n - 1) as f64;
    let j = h.floor() as usize;
    if j >= n - 1 {
        return Ok(sorted[n - 1]);
    }
    Ok(sorted[j] + (h - j as f64) * (sorted[j + 1] - sorted[j]))
}

/// Tail value at risk as the mean of [`var_linear_sorted`]'s piecewise
/// linear quantile function over `[p, 1]`: trapezoids on the positions
/// `0, ..., n - 1`, from `h = p (n - 1)` to the largest draw.
///
/// This is the convention of simulation models that report numpy's linear
/// quantiles; [`tvar_sorted`] is the exact TVaR of the empirical
/// distribution. The two differ by a few tenths of a percent in thin tails
/// and by more on a tail of repeated (tiled) draws. `p = 1` gives the
/// largest draw. `sorted` must be non-empty and sorted ascending; this is
/// checked only in debug builds.
///
/// ```
/// use prospicio_prob::risk::tvar_linear_sorted;
///
/// let x = [1.0, 2.0, 3.0, 4.0];
/// // The quantile function runs from 2.5 to 4 over the top half.
/// assert_eq!(tvar_linear_sorted(&x, 0.5), Ok(3.25));
/// assert_eq!(tvar_linear_sorted(&x, 0.0), Ok(2.5));
/// ```
pub fn tvar_linear_sorted(sorted: &[f64], p: f64) -> Result<f64> {
    check_probability(p)?;
    debug_assert!(is_sorted(sorted), "draws must be sorted ascending");
    let n = sorted.len();
    let top = (n - 1) as f64;
    let h = p * top;
    let j = h.floor() as usize;
    if j >= n - 1 || h >= top {
        return Ok(sorted[n - 1]);
    }
    let v = sorted[j] + (h - j as f64) * (sorted[j + 1] - sorted[j]);
    let partial = (v + sorted[j + 1]) / 2.0 * ((j + 1) as f64 - h);
    let seg = &sorted[j + 1..];
    let full = (2.0 * seg.iter().sum::<f64>() - seg[0] - seg[seg.len() - 1]) / 2.0;
    Ok((partial + full) / (top - h))
}

/// The number `k` of the largest draws a whole-tail convention reads at
/// level `p` among `n`: the largest `k` with `k / n <= 1 - p`, at least 1
/// (`max(1, floor((1 - p) n))`, compared as `(n - k) / n >= p` so that
/// `1 - p`'s rounding cannot drop a draw).
pub(crate) fn tail_count(n: usize, p: f64) -> usize {
    let k = if p == 0.0 { n } else { n - rank(n, p) };
    k.max(1)
}

/// Value at risk as the `k`-th largest draw, `k = max(1, floor((1 - p) n))`:
/// with years as draws, the `k`-th worst of `n` simulated years, the
/// convention of many catastrophe pricing teams.
///
/// It is the upper quantile `inf {x : F(x) > p}` of the empirical
/// distribution, one order statistic above [`var_sorted`] (Hyndman and
/// Fan's type 1, the lower quantile) unless both are the largest draw:
/// at `p = 0.99` on 2,000 draws it is the 1,981st order statistic, the
/// 20th largest, where [`var_sorted`] gives the 1,980th. `p = 0` gives the
/// smallest draw and `p = 1` the largest. `sorted` must be non-empty and
/// sorted ascending; this is checked only in debug builds.
///
/// ```
/// use prospicio_prob::risk::{var_kth_largest_sorted, var_sorted};
///
/// let x: Vec<f64> = (1..=10).map(f64::from).collect();
/// // k = floor(0.2 * 10) = 2: the second largest.
/// assert_eq!(var_kth_largest_sorted(&x, 0.8), Ok(9.0));
/// assert_eq!(var_sorted(&x, 0.8), Ok(8.0));
/// // k = floor(0.25 * 10) = 2 again.
/// assert_eq!(var_kth_largest_sorted(&x, 0.75), Ok(9.0));
/// ```
pub fn var_kth_largest_sorted(sorted: &[f64], p: f64) -> Result<f64> {
    check_probability(p)?;
    debug_assert!(is_sorted(sorted), "draws must be sorted ascending");
    let n = sorted.len();
    Ok(sorted[n - tail_count(n, p)])
}

/// Tail value at risk as the mean of the `k` largest draws,
/// `k = max(1, floor((1 - p) n))`: with years as draws, the mean of the
/// `k` worst whole years, no year split at the level.
///
/// It equals [`tvar_sorted`], the exact TVaR, when `(1 - p) n` is a whole
/// number; otherwise it leaves out the fraction of a year [`tvar_sorted`]
/// counts, so it is the mean of a slightly thinner tail and at least as
/// large. `p = 0` gives the mean and `p = 1` the largest draw. `sorted`
/// must be non-empty and sorted ascending; this is checked only in debug
/// builds.
///
/// ```
/// use prospicio_prob::risk::{tvar_sorted, tvar_whole_years_sorted};
///
/// let x: Vec<f64> = (1..=10).map(f64::from).collect();
/// // (1 - 0.8) * 10 = 2 years: 9 and 10, as the exact TVaR.
/// assert_eq!(tvar_whole_years_sorted(&x, 0.8), Ok(9.5));
/// assert!((tvar_sorted(&x, 0.8).unwrap() - 9.5).abs() < 1e-12);
/// // 2.5 years: still the two worst, where the exact TVaR takes half of 8.
/// assert_eq!(tvar_whole_years_sorted(&x, 0.75), Ok(9.5));
/// assert!((tvar_sorted(&x, 0.75).unwrap() - 9.2).abs() < 1e-12);
/// ```
pub fn tvar_whole_years_sorted(sorted: &[f64], p: f64) -> Result<f64> {
    check_probability(p)?;
    debug_assert!(is_sorted(sorted), "draws must be sorted ascending");
    let n = sorted.len();
    let k = tail_count(n, p);
    Ok(sorted[n - k..].iter().sum::<f64>() / k as f64)
}

/// Value at risk of a discrete distribution: the smallest value `x` with
/// `P(X <= x) >= p`, the lower quantile. `values` are ascending, `probs`
/// their probabilities. `p = 0` gives the smallest value.
///
/// ```
/// use prospicio_prob::risk::var_weighted;
///
/// let (x, w) = ([0.0, 1.0, 2.0], [0.5, 0.3, 0.2]);
/// assert_eq!(var_weighted(&x, &w, 0.5), Ok(0.0));
/// assert_eq!(var_weighted(&x, &w, 0.6), Ok(1.0));
/// ```
pub fn var_weighted(values: &[f64], probs: &[f64], p: f64) -> Result<f64> {
    check_probability(p)?;
    check_weighted(values, probs)?;
    let mut cum = 0.0;
    for (&x, &w) in values.iter().zip(probs) {
        cum += w;
        if cum >= p {
            return Ok(x);
        }
    }
    // Rounding left the total a hair below p.
    Ok(values[values.len() - 1])
}

/// Tail value at risk of a discrete distribution,
/// `(1 / (1 - p)) * integral from p to 1 of VaR(u) du`: the value at the
/// VaR counted with the share of its mass above `p`, then every larger
/// value. `values` are ascending, `probs` their probabilities. `p = 0`
/// gives the mean; at or beyond the last value's cumulative start, the
/// largest value (as `aggregate`).
///
/// ```
/// use prospicio_prob::risk::tvar_weighted;
///
/// let (x, w) = ([0.0, 1.0, 2.0], [0.5, 0.3, 0.2]);
/// // Above p = 0.6: 0.2 of the value 1, then 0.2 of the value 2.
/// assert!((tvar_weighted(&x, &w, 0.6).unwrap() - 1.5).abs() < 1e-15);
/// ```
pub fn tvar_weighted(values: &[f64], probs: &[f64], p: f64) -> Result<f64> {
    check_probability(p)?;
    check_weighted(values, probs)?;
    Ok(tvar_unchecked(values, probs, p))
}

fn tvar_unchecked(values: &[f64], probs: &[f64], p: f64) -> f64 {
    let n = values.len();
    let mut cum = 0.0;
    for j in 0..n {
        cum += probs[j];
        if cum > p {
            if j == n - 1 {
                return values[n - 1];
            }
            let tail: f64 = values[j + 1..]
                .iter()
                .zip(&probs[j + 1..])
                .map(|(x, w)| x * w)
                .sum();
            return ((cum - p) * values[j] + tail) / (1.0 - p);
        }
    }
    values[n - 1]
}

/// The level `t` at which the tail value at risk reaches the value at
/// risk at `p`, `TVaR(t) = VaR(p)` (the lower VaR), as `aggregate`'s
/// `tvar_threshold`. TVaR rises continuously from the mean at `t = 0` to
/// the largest value, so `t <= p`; it is found by bisection to the last
/// bit. `p = 0`, or a VaR at or below the mean, gives 0 (`aggregate`
/// returns 0 at `p = 0` and fails below the mean); a VaR at the largest
/// value, where every `t` from the last value's cumulative start up
/// qualifies, gives 1, as `aggregate`'s bisection.
///
/// ```
/// use prospicio_prob::risk::{tvar_threshold, tvar_weighted, var_weighted};
///
/// let (x, w) = ([0.0, 1.0, 2.0, 5.0], [0.4, 0.3, 0.2, 0.1]);
/// let t = tvar_threshold(&x, &w, 0.85).unwrap();
/// let target = var_weighted(&x, &w, 0.85).unwrap();
/// assert!((tvar_weighted(&x, &w, t).unwrap() - target).abs() < 1e-12);
/// assert!(t < 0.85);
/// ```
pub fn tvar_threshold(values: &[f64], probs: &[f64], p: f64) -> Result<f64> {
    let target = var_weighted(values, probs, p)?;
    if p == 0.0 {
        return Ok(0.0);
    }
    if target >= values[values.len() - 1] {
        return Ok(1.0);
    }
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    if tvar_unchecked(values, probs, lo) >= target {
        return Ok(0.0);
    }
    while hi - lo > f64::EPSILON * hi {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if tvar_unchecked(values, probs, mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

fn check_weighted(values: &[f64], probs: &[f64]) -> Result<()> {
    if values.is_empty() || values.len() != probs.len() {
        return Err(Error::InvalidParameter {
            name: "values",
            value: values.len() as f64,
            reason: "must be non-empty and as long as probs",
        });
    }
    debug_assert!(is_sorted(values), "values must be sorted ascending");
    Ok(())
}

/// Entropic risk measure `(1/θ) log E[e^(θX)]` of equally likely draws
/// (losses positive), the certainty equivalent under exponential utility
/// with risk aversion `θ > 0`. It increases from the mean (`θ → 0`) to the
/// largest draw (`θ → ∞`); for a normal `X` it is `μ + θσ²/2`. Convex and
/// translation-invariant, but not positively homogeneous.
///
/// ```
/// use prospicio_prob::risk::entropic;
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
/// use prospicio_prob::risk::esscher;
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
pub(crate) fn rank(n: usize, p: f64) -> usize {
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
            .map(|i| {
                mu + sigma * prospicio_math::special::norm_quantile((i as f64 + 0.5) / n as f64)
            })
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
    use prospicio_core::Error;

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

    #[test]
    fn tail_count_is_exact_where_one_minus_p_times_n_rounds_down() {
        // 1 - 0.9 = 0.09999999999999998, times 10 is just below 1.
        assert_eq!(tail_count(10, 0.9), 1);
        assert_eq!(tail_count(20, 0.95), 1);
        assert_eq!(tail_count(2000, 0.99), 20);
        assert_eq!(tail_count(2000, 0.995), 10);
        assert_eq!(tail_count(2000, 0.996), 8);
        assert_eq!(tail_count(10, 0.75), 2);
        assert_eq!(tail_count(10, 0.0), 10);
        assert_eq!(tail_count(10, 1.0), 1);
        assert_eq!(tail_count(10, 0.99), 1);
    }

    #[test]
    fn the_kth_largest_and_whole_years_match_a_sort_by_hand() {
        // A scrambled sample with ties.
        let mut x: Vec<f64> = (0..2000u64)
            .map(|i| ((i * 7919) % 1013) as f64 / 7.0)
            .collect();
        x.sort_by(f64::total_cmp);
        let desc: Vec<f64> = x.iter().rev().copied().collect();
        for (p, k) in [
            (0.99, 20),
            (0.995, 10),
            (0.999, 2),
            (0.9995, 1),
            (0.5, 1000),
            (0.987, 26),
        ] {
            assert_eq!(tail_count(x.len(), p), k, "p = {p}");
            assert_eq!(var_kth_largest_sorted(&x, p).unwrap(), desc[k - 1]);
            let mean = desc[..k].iter().sum::<f64>() / k as f64;
            assert_close(tvar_whole_years_sorted(&x, p).unwrap(), mean);
        }
        // Where (1 - p) n is whole the TVaRs agree, and the k-th largest is
        // one order statistic above the inverted cdf.
        for p in [0.99, 0.995, 0.5] {
            assert_close(
                tvar_whole_years_sorted(&x, p).unwrap(),
                tvar_sorted(&x, p).unwrap(),
            );
            let k = rank(x.len(), p);
            assert_eq!(var_sorted(&x, p).unwrap(), x[k - 1]);
            assert_eq!(var_kth_largest_sorted(&x, p).unwrap(), x[k]);
        }
        // Elsewhere the whole years leave a fraction out, so are no smaller.
        assert!(tvar_whole_years_sorted(&x, 0.987).unwrap() >= tvar_sorted(&x, 0.987).unwrap());
        assert_eq!(var_kth_largest_sorted(&x, 1.0).unwrap(), x[1999]);
        assert_eq!(var_kth_largest_sorted(&x, 0.0).unwrap(), x[0]);
        assert_close(
            tvar_whole_years_sorted(&x, 0.0).unwrap(),
            x.iter().sum::<f64>() / 2000.0,
        );
        assert!(var_kth_largest_sorted(&x, 1.5).is_err());
        assert!(tvar_whole_years_sorted(&x, -0.1).is_err());
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
