//! The arithmetic of the model-agnostic explanation tools, on plain
//! vectors, so Python and R share it and keep only their predict loops
//! (#837): the accumulated local effects (Apley and Zhu, 2020, as R's
//! `ALEPlot`) and SHAP values (Lundberg and Lee, 2017).
//!
//! A front end asks a kernel where to predict, calls its model there, and
//! hands the predictions back. Indices are 0-based.

use prospicio_core::{Error, Result};
use prospicio_math::linalg::symmetric_eigen;

/// R's four-ulp fuzz in `quantile`.
const FUZZ: f64 = 4.0 * f64::EPSILON;

/// R's `quantile(type = 1)` of sorted values: the inverse of the empirical
/// cdf.
fn quantile_type1(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let nppm = n as f64 * p;
    let j = (nppm + FUZZ).floor();
    let k = if nppm > j + FUZZ { j + 1.0 } else { j };
    sorted[(k.max(1.0) as usize).min(n) - 1]
}

/// R's default `quantile(type = 7)` of sorted values.
fn quantile_type7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let h = (n - 1) as f64 * p;
    let lo = (h + FUZZ).floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// The grid `z_0 < ... < z_K` of a numeric feature's accumulated local
/// effects: its minimum and its type 1 quantiles at `1/bins, ..., 1` (the
/// probabilities as R's `seq(1/K, 1, length.out = K)` computes them),
/// duplicates merged. An error for no values, `bins` 0 or a feature with
/// one value.
///
/// ```
/// use prospicio_models::explain::ale_grid;
///
/// let x: Vec<f64> = (0..10).map(f64::from).collect();
/// assert_eq!(ale_grid(&x, 2).unwrap(), [0.0, 4.0, 9.0]);
/// ```
pub fn ale_grid(x: &[f64], bins: usize) -> Result<Vec<f64>> {
    if bins < 1 {
        return Err(Error::Data("bins must be at least 1".into()));
    }
    if x.is_empty() || x.iter().any(|v| !v.is_finite()) {
        return Err(Error::Data("the feature needs finite values".into()));
    }
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let k = bins as f64;
    let by = if bins > 1 {
        (1.0 - 1.0 / k) / (k - 1.0)
    } else {
        0.0
    };
    let mut z = vec![s[0]];
    z.extend((0..bins).map(|j| quantile_type1(&s, 1.0 / k + j as f64 * by)));
    z.sort_by(f64::total_cmp);
    z.dedup();
    if z.len() < 2 {
        return Err(Error::Data(
            "the feature takes one value: it has no local effect".into(),
        ));
    }
    Ok(z)
}

/// Each value's interval on the grid `z`, `1..=K` for `(z_{k-1}, z_k]`,
/// the first also closed below (R's `cut(include.lowest = TRUE)`): the
/// row's prediction moves from `z[k - 1]` to `z[k]`.
pub fn ale_intervals(x: &[f64], z: &[f64]) -> Vec<usize> {
    x.iter()
        .map(|v| z.partition_point(|g| g < v).max(1))
        .collect()
}

/// Accumulated local effects and the rows behind each value.
#[derive(Debug, Clone, PartialEq)]
pub struct AleEffects {
    /// The centred accumulated effect at each grid value or level.
    pub effects: Vec<f64>,
    /// A numeric feature's rows in each interval (one fewer than the
    /// effects), or a categorical feature's rows at each level.
    pub counts: Vec<usize>,
}

/// A numeric feature's effects on a grid of `K + 1` values from each row's
/// interval ([`ale_intervals`]) and its predictions at the interval's lower
/// and upper ends: each interval's mean move, accumulated from 0 and
/// centred so the mean over the rows is 0 (ALEPlot's trapezoid rule).
pub fn ale_numeric(interval: &[usize], lower: &[f64], upper: &[f64], k: usize) -> AleEffects {
    let mut total = vec![0.0; k + 1];
    let mut count = vec![0usize; k + 1];
    for ((&b, lo), hi) in interval.iter().zip(lower).zip(upper) {
        total[b] += hi - lo;
        count[b] += 1;
    }
    let mut f = vec![0.0];
    for b in 1..=k {
        f.push(f[b - 1] + total[b] / count[b] as f64);
    }
    let n = interval.len() as f64;
    let centre = (1..=k)
        .map(|b| (f[b - 1] + f[b]) / 2.0 * count[b] as f64)
        .sum::<f64>()
        / n;
    AleEffects {
        effects: f.iter().map(|v| v - centre).collect(),
        counts: count[1..].to_vec(),
    }
}

/// A categorical feature's effects over its `k` levels in order, from each
/// row's level (`at`) and its predictions at that level, the next and the
/// previous (the same level at either end): the moves between neighbouring
/// levels, averaged over the rows on both sides, accumulated from 0 and
/// centred on the levels' frequencies.
pub fn ale_levels(at: &[usize], k: usize, base: &[f64], up: &[f64], down: &[f64]) -> AleEffects {
    let mut total = vec![0.0; k.saturating_sub(1)];
    let mut count = vec![0usize; k.saturating_sub(1)];
    for (i, &a) in at.iter().enumerate() {
        if a + 1 < k {
            total[a] += up[i] - base[i];
            count[a] += 1;
        }
        if a > 0 {
            total[a - 1] += base[i] - down[i];
            count[a - 1] += 1;
        }
    }
    let mut f = vec![0.0];
    for j in 0..k.saturating_sub(1) {
        f.push(f[j] + total[j] / count[j] as f64);
    }
    let mut freq = vec![0usize; k];
    for &a in at {
        freq[a] += 1;
    }
    let n = at.len() as f64;
    let centre = f
        .iter()
        .zip(&freq)
        .map(|(fj, &c)| fj * c as f64)
        .sum::<f64>()
        / n;
    AleEffects {
        effects: f.iter().map(|v| v - centre).collect(),
        counts: freq,
    }
}

/// ALEPlot's distance between `k` levels from one numeric column: the
/// Kolmogorov-Smirnov distance between the levels' empirical cdfs of
/// `values` on the column's 100 type 7 quantiles. `level[i]` is row `i`'s
/// level; the result is `k × k`, row-major.
pub fn ale_numeric_distance(values: &[f64], level: &[usize], k: usize) -> Vec<f64> {
    let mut s = values.to_vec();
    s.sort_by(f64::total_cmp);
    let grid: Vec<f64> = (0..100)
        .map(|q| quantile_type7(&s, q as f64 / 99.0))
        .collect();
    let ecdf: Vec<Vec<f64>> = (0..k)
        .map(|l| {
            let mut v: Vec<f64> = values
                .iter()
                .zip(level)
                .filter(|&(_, &a)| a == l)
                .map(|(&x, _)| x)
                .collect();
            v.sort_by(f64::total_cmp);
            let m = v.len() as f64;
            grid.iter()
                .map(|g| v.partition_point(|x| x <= g) as f64 / m)
                .collect()
        })
        .collect();
    let mut d = vec![0.0; k * k];
    for a in 0..k {
        for b in a + 1..k {
            let v = ecdf[a]
                .iter()
                .zip(&ecdf[b])
                .map(|(u, w)| (u - w).abs())
                .fold(0.0, f64::max);
            d[a * k + b] = v;
            d[b * k + a] = v;
        }
    }
    d
}

/// ALEPlot's distance between `k` levels from one categorical column, its
/// categories coded `0..` in `codes`: half the L1 distance between the
/// levels' conditional category frequencies. `k × k`, row-major.
pub fn ale_categorical_distance(codes: &[usize], level: &[usize], k: usize) -> Vec<f64> {
    let c = codes.iter().max().map_or(0, |m| m + 1);
    let mut freq = vec![0.0_f64; k * c];
    let mut rows = vec![0.0_f64; k];
    for (&code, &a) in codes.iter().zip(level) {
        freq[a * c + code] += 1.0;
        rows[a] += 1.0;
    }
    let mut d = vec![0.0; k * k];
    for a in 0..k {
        for b in a + 1..k {
            let v = (0..c)
                .map(|j| (freq[a * c + j] / rows[a] - freq[b * c + j] / rows[b]).abs())
                .sum::<f64>()
                / 2.0;
            d[a * k + b] = v;
            d[b * k + a] = v;
        }
    }
    d
}

/// ALEPlot's order of `k` levels from their summed distances (`k × k`,
/// row-major, summed over the other columns): the first coordinate of
/// classical scaling (R's `cmdscale(d, k = 1)`, the leading eigenvector of
/// `-J D² J / 2`), read from the end with the lower index, so a front end
/// that passes its levels sorted gets the order that starts from the level
/// that sorts first. Returns the level indices in order.
///
/// ```
/// use prospicio_models::explain::ale_order;
///
/// // Three levels on a line, 0 -- 2 -- 1.
/// let d = [0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0];
/// assert_eq!(ale_order(&d, 3), [0, 2, 1]);
/// ```
pub fn ale_order(distance: &[f64], k: usize) -> Vec<usize> {
    if k < 2 {
        return (0..k).collect();
    }
    let d2: Vec<f64> = distance.iter().map(|v| v * v).collect();
    let row: Vec<f64> = (0..k)
        .map(|i| d2[i * k..(i + 1) * k].iter().sum::<f64>() / k as f64)
        .collect();
    let total = row.iter().sum::<f64>() / k as f64;
    let mut b = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            b[i * k + j] = -(d2[i * k + j] - row[i] - row[j] + total) / 2.0;
        }
    }
    let (_, v) = symmetric_eigen(&b, k);
    // The leading eigenvector is the last column.
    let coord: Vec<f64> = (0..k).map(|i| v[i * k + k - 1]).collect();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&i, &j| coord[i].total_cmp(&coord[j]).then(i.cmp(&j)));
    if order[0] > order[k - 1] {
        order.reverse();
    }
    order
}

/// The Shapley weight of adding one of `m` players to a coalition of each
/// size `s` in `0..m`: `s! (m - s - 1)! / m!`.
pub fn shapley_weights(m: usize) -> Vec<f64> {
    let fact = |n: usize| (1..=n).map(|i| i as f64).product::<f64>();
    (0..m)
        .map(|s| fact(s) * fact(m - s - 1) / fact(m))
        .collect()
}

/// Exact SHAP values from the game's value at every coalition: `values`
/// holds `2^m` blocks of `n` values, block `mask` the value for each row of
/// the coalition whose bit `g` is set when group `g` is in it (block 0 the
/// baseline, the last the outputs). Returns `n × m`, row-major: row `i`'s
/// contribution of each group, summing to its output less the baseline.
/// An error for more than 30 groups or the wrong length.
///
/// ```
/// use prospicio_models::explain::shap_exact;
///
/// // One row, v(S) = 1 + 2 [x in S] + 3 [z in S]: additive, so x gets 2.
/// let phi = shap_exact(&[1.0, 3.0, 4.0, 6.0], 1, 2).unwrap();
/// assert_eq!(phi, [2.0, 3.0]);
/// ```
pub fn shap_exact(values: &[f64], n: usize, m: usize) -> Result<Vec<f64>> {
    if m > 30 {
        return Err(Error::Data(format!("{m} groups are too many to enumerate")));
    }
    let masks = 1usize << m;
    if values.len() != masks * n {
        return Err(Error::Data(format!(
            "values must hold 2^{m} blocks of {n}, got {}",
            values.len()
        )));
    }
    let weight = shapley_weights(m);
    let mut phi = vec![0.0; n * m];
    for mask in 0..masks {
        let size = mask.count_ones() as usize;
        for g in 0..m {
            if mask >> g & 1 == 1 {
                continue;
            }
            let w = weight[size];
            let with_g = mask | 1 << g;
            for i in 0..n {
                phi[i * m + g] += w * (values[with_g * n + i] - values[mask * n + i]);
            }
        }
    }
    Ok(phi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_effects_of_a_line_are_the_line() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
        let z = ale_grid(&x, 5).unwrap();
        let at = ale_intervals(&x, &z);
        let f = |v: f64| 3.0 * v + 1.0;
        let lo: Vec<f64> = at.iter().map(|&b| f(z[b - 1])).collect();
        let hi: Vec<f64> = at.iter().map(|&b| f(z[b])).collect();
        let a = ale_numeric(&at, &lo, &hi, z.len() - 1);
        let slope = (a.effects[a.effects.len() - 1] - a.effects[0]) / (z[z.len() - 1] - z[0]);
        assert!((slope - 3.0).abs() < 1e-12);
        assert_eq!(a.counts.iter().sum::<usize>(), 50);
        let mean: f64 = (1..z.len())
            .map(|b| (a.effects[b - 1] + a.effects[b]) / 2.0 * a.counts[b - 1] as f64)
            .sum::<f64>()
            / 50.0;
        assert!(mean.abs() < 1e-12);
    }

    #[test]
    fn level_effects_centre_on_frequencies() {
        let at = [0, 0, 1, 2, 2, 2];
        let value = [0.0, 1.0, 5.0];
        let base: Vec<f64> = at.iter().map(|&a| value[a]).collect();
        let up: Vec<f64> = at.iter().map(|&a| value[(a + 1).min(2)]).collect();
        let down: Vec<f64> = at.iter().map(|&a| value[a.saturating_sub(1)]).collect();
        let e = ale_levels(&at, 3, &base, &up, &down);
        assert_eq!(e.counts, [2, 1, 3]);
        let mean = (e.effects[0] * 2.0 + e.effects[1] + e.effects[2] * 3.0) / 6.0;
        assert!(mean.abs() < 1e-15);
        assert!((e.effects[1] - e.effects[0] - 1.0).abs() < 1e-15);
        assert!((e.effects[2] - e.effects[1] - 4.0).abs() < 1e-15);
    }

    #[test]
    fn shapley_weights_sum_to_one_over_coalitions() {
        let m = 5;
        let w = shapley_weights(m);
        // Each player's weights over all coalitions without it sum to 1.
        let binom =
            |n: usize, k: usize| (0..k).fold(1.0, |a, i| a * (n - i) as f64 / (i + 1) as f64);
        let total: f64 = (0..m).map(|s| binom(m - 1, s) * w[s]).sum();
        assert!((total - 1.0).abs() < 1e-15);
    }

    #[test]
    fn exact_shap_is_additive_with_an_interaction() {
        // v(S) = x z + x for two players: phi_x = 1 + 1/2, phi_z = 1/2.
        let v = [0.0, 1.0, 0.0, 2.0];
        let phi = shap_exact(&v, 1, 2).unwrap();
        assert_eq!(phi, [1.5, 0.5]);
        assert!(shap_exact(&v, 2, 2).is_err());
    }

    #[test]
    fn distances_and_order() {
        let level = [0, 0, 1, 1, 2, 2];
        let codes = [0, 0, 0, 1, 1, 1];
        let d = ale_categorical_distance(&codes, &level, 3);
        assert_eq!(d[1], 0.5);
        assert_eq!(d[2], 1.0);
        let num = ale_numeric_distance(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &level, 3);
        // Fully separated levels: every Kolmogorov-Smirnov distance is 1.
        assert_eq!((num[1], num[2], num[5]), (1.0, 1.0, 1.0));
        assert_eq!(ale_order(&d, 3), [0, 1, 2]);
    }
}
