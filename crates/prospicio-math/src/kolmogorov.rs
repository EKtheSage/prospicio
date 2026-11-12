//! The two-sided Kolmogorov–Smirnov distribution: `P(D_n > d)` for the
//! largest distance `D_n` between the empirical distribution function of
//! `n` independent draws and their continuous distribution function, the
//! classical one-sample test's p-value.
//!
//! The method follows SciPy's `scipy.stats.kstwo` (Simard and L'Ecuyer,
//! "Computing the Two-Sided Kolmogorov–Smirnov Distribution", *Journal of
//! Statistical Software* 39(11), 2011): Ruben and Gambino's closed forms
//! at the two ends, twice the exact one-sided Smirnov tail (Birnbaum and
//! Tingey's sum) where that equals the two-sided tail or is within double
//! precision of it, Durbin's matrix (Marsaglia, Tsang and Wang, 2003) in
//! between, and Kolmogorov's limit `2 Σ (−1)^(k−1) e^(−2k²z²)` at
//! `z = √n d` when the matrix would be too large.

use crate::linalg::mat_mul;
use crate::special::ln_gamma;

/// The largest Durbin matrix side `2k − 1`, `k = ⌈n d⌉`, computed exactly;
/// beyond it the limiting distribution is used. SciPy switches to Pelz and
/// Good's asymptotic series at `n d^1.5 > 1.4`, which for `n ≤ 100,000` is
/// a side of at most about 2,300.
const MAX_SIDE: usize = 401;

/// `P(D_n > d)`, the two-sided Kolmogorov–Smirnov p-value of the distance
/// `d` between `n` draws and their hypothesized distribution: exact where
/// SciPy's `kstwo.sf` is exact (to about 1e-12 there), Kolmogorov's limit
/// for large `n` at small `d`. The test is classical: with the
/// distribution's parameters fitted to the same draws, or with the data
/// truncated, the true p-value is smaller, so this one is optimistic.
///
/// ```
/// use prospicio_math::kolmogorov::kolmogorov_sf;
///
/// // D_1 is uniform on [1/2, 1].
/// assert!((kolmogorov_sf(1, 0.8) - 0.4).abs() < 1e-15);
/// // scipy.stats.kstwo.sf(0.2, 20) = 0.352720173623415
/// assert!((kolmogorov_sf(20, 0.2) - 0.352720173623415).abs() < 1e-10);
/// ```
pub fn kolmogorov_sf(n: usize, d: f64) -> f64 {
    if n == 0 || d.is_nan() {
        return f64::NAN;
    }
    let nf = n as f64;
    if d >= 1.0 {
        return 0.0;
    }
    if d <= 0.0 {
        return 1.0;
    }
    let t = nf * d;
    if t <= 1.0 {
        // Ruben and Gambino: P(D_n ≤ d) = n!/n^n (2t − 1)^n for t ≤ 1.
        if t <= 0.5 {
            return 1.0;
        }
        let ln_cdf = ln_gamma(nf + 1.0) - nf * nf.ln() + nf * (2.0 * t - 1.0).ln();
        return clip(1.0 - ln_cdf.exp());
    }
    if t >= nf - 1.0 {
        return clip(2.0 * (1.0 - d).powf(nf));
    }
    if d >= 0.5 {
        // The two one-sided events cannot both happen: exactly twice
        // Smirnov's tail.
        return clip(2.0 * smirnov_sf(n, d));
    }
    let nd2 = t * d;
    // Where the tail is small, twice Smirnov's is the two-sided tail to
    // double precision (Miller); SciPy's cutoffs.
    if (n <= 140 && nd2 > 4.0) || (n > 140 && nd2 >= 2.2) {
        return clip(2.0 * smirnov_sf(n, d));
    }
    let k = t.ceil() as usize;
    if 2 * k - 1 <= MAX_SIDE {
        clip(1.0 - durbin_cdf(n, d))
    } else {
        kolmogorov_limit_sf(nf.sqrt() * d)
    }
}

/// Kolmogorov's limiting distribution's tail, `P(K > z) = 2 Σ_{k≥1}
/// (−1)^(k−1) e^(−2k²z²)`, the limit of `P(√n D_n > z)`. For small `z` the
/// equivalent `1 − √(2π)/z Σ e^(−(2k−1)²π²/(8z²))` converges faster.
///
/// ```
/// use prospicio_math::kolmogorov::kolmogorov_limit_sf;
///
/// // scipy.stats.kstwobign.sf(1.0) = 0.26999967167735456
/// assert!((kolmogorov_limit_sf(1.0) - 0.26999967167735456).abs() < 1e-15);
/// ```
pub fn kolmogorov_limit_sf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z <= 0.0 {
        return 1.0;
    }
    if z < 1.0 {
        let c = std::f64::consts::PI * std::f64::consts::PI / (8.0 * z * z);
        let s: f64 = (1..=20)
            .map(|k| {
                let m = (2 * k - 1) as f64;
                (-m * m * c).exp()
            })
            .sum();
        return clip(1.0 - (2.0 * std::f64::consts::PI).sqrt() / z * s);
    }
    let mut s = 0.0;
    for k in 1..=100 {
        let kf = k as f64;
        let term = (-2.0 * kf * kf * z * z).exp();
        s += if k % 2 == 1 { term } else { -term };
        if term < 1e-300 {
            break;
        }
    }
    clip(2.0 * s)
}

/// Smirnov's one-sided tail `P(D_n^+ ≥ d)`, exactly by Birnbaum and
/// Tingey's sum `d Σ_{j=0}^{⌊n(1−d)⌋} C(n, j) (1 − d − j/n)^(n−j)
/// (d + j/n)^(j−1)`, each term in logarithms.
fn smirnov_sf(n: usize, d: f64) -> f64 {
    let nf = n as f64;
    let last = (nf * (1.0 - d)).floor() as usize;
    let ln_n1 = ln_gamma(nf + 1.0);
    let mut s = 0.0;
    for j in 0..=last.min(n) {
        let jf = j as f64;
        let a = 1.0 - d - jf / nf;
        if a <= 0.0 {
            continue;
        }
        let ln_c = ln_n1 - ln_gamma(jf + 1.0) - ln_gamma(nf - jf + 1.0);
        let ln_term = ln_c + (nf - jf) * a.ln() + (jf - 1.0) * (d + jf / nf).ln();
        s += ln_term.exp();
    }
    d * s
}

/// `P(D_n ≤ d)` by Durbin's matrix as Marsaglia, Tsang and Wang compute
/// it: with `d = (k − h)/n`, the `k`-th diagonal element of `H^n` times
/// `n!/n^n`, scaled by powers of 2 so nothing overflows.
fn durbin_cdf(n: usize, d: f64) -> f64 {
    let nd = n as f64 * d;
    let k = nd.ceil() as usize;
    let h = k as f64 - nd;
    let m = 2 * k - 1;
    let mut hm = vec![0.0; m * m];
    // v: the first column (and, reversed, the last row); w[j] = 1/j!.
    let mut v = vec![0.0; m];
    let mut w = vec![0.0; m];
    let mut fac = 1.0;
    for j in 1..=m {
        w[j - 1] = fac;
        fac /= j as f64;
        v[j - 1] = (1.0 - h.powi(j as i32)) * fac;
    }
    let tt = (2.0 * h - 1.0).max(0.0).powi(m as i32) - 2.0 * h.powi(m as i32);
    v[m - 1] = (1.0 + tt) * fac;
    for i in 1..m {
        for r in (i - 1)..m {
            hm[r * m + i] = w[r - (i - 1)];
        }
    }
    for r in 0..m {
        hm[r * m] = v[r];
    }
    for c in 0..m {
        hm[(m - 1) * m + c] = v[m - 1 - c];
    }
    const E: i32 = 128;
    let big = 2f64.powi(E);
    let mut power = identity(m);
    let (mut expnt, mut h_expnt) = (0i32, 0i32);
    let mut nn = n;
    while nn > 0 {
        if nn % 2 == 1 {
            power = mat_mul(&power, &hm, m);
            expnt += h_expnt;
        }
        hm = mat_mul(&hm, &hm, m);
        h_expnt *= 2;
        if hm[(k - 1) * m + (k - 1)].abs() > big {
            for x in &mut hm {
                *x /= big;
            }
            h_expnt += E;
        }
        nn /= 2;
    }
    let mut p = power[(k - 1) * m + (k - 1)];
    for i in 1..=n {
        p = i as f64 * p / n as f64;
        if p.abs() < 1.0 / big {
            p *= big;
            expnt -= E;
        }
    }
    clip(p * 2f64.powi(expnt))
}

fn identity(m: usize) -> Vec<f64> {
    let mut a = vec![0.0; m * m];
    for i in 0..m {
        a[i * m + i] = 1.0;
    }
    a
}

fn clip(p: f64) -> f64 {
    p.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_ends_are_in_closed_form() {
        assert_eq!(kolmogorov_sf(10, 0.0), 1.0);
        assert_eq!(kolmogorov_sf(10, 0.04), 1.0);
        assert_eq!(kolmogorov_sf(10, 1.0), 0.0);
        // n = 2: P(D_2 ≤ d) = 2 (2d − 1/2)^2 ... at t = 2d ≤ 1, n!/n^n (2t-1)^n.
        assert!((kolmogorov_sf(2, 0.4) - (1.0 - 0.5 * 0.6f64.powi(2))).abs() < 1e-15);
        assert!((kolmogorov_sf(5, 0.9) - 2.0 * 0.1f64.powi(5)).abs() < 1e-18);
    }

    #[test]
    fn durbin_and_smirnov_agree_where_both_hold() {
        // Past n d² = 4 twice Smirnov's tail is the two-sided one to double
        // precision; Durbin's matrix is exact everywhere.
        for (n, d) in [(30, 0.38), (60, 0.27), (100, 0.21)] {
            let durbin = 1.0 - durbin_cdf(n, d);
            let smirnov = 2.0 * smirnov_sf(n, d);
            assert!(
                (durbin - smirnov).abs() < 1e-10,
                "{n} {d}: {durbin} {smirnov}"
            );
        }
    }

    #[test]
    fn large_samples_approach_the_limit() {
        let z = 0.9;
        let n = 1_000_000;
        let p = kolmogorov_sf(n, z / (n as f64).sqrt());
        assert!((p - kolmogorov_limit_sf(z)).abs() < 2e-3, "{p}");
        // Both forms of the limit agree around z = 1.
        let below = kolmogorov_limit_sf(0.999_999_9);
        let above = kolmogorov_limit_sf(1.000_000_1);
        assert!((below - above).abs() < 1e-6);
    }
}
