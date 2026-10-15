//! Classical credibility: limited fluctuation standards and the
//! Bühlmann–Straub model (`docs/design/credibility.md`).
//!
//! Hierarchical and Bayesian credibility belong with v0.7
//! (`docs/roadmap.md`).

use prospicio_core::{Error, Result};
use prospicio_math::special::norm_quantile;

/// Limited fluctuation (classical) credibility: the expected number of
/// claims for the observed value to lie within `k` of its mean with
/// probability `p`, under a Poisson claim count and the normal
/// approximation (Werner and Modlin, *Basic Ratemaking*, ch. 12).
///
/// With `z = Φ⁻¹((1 + p)/2)` and `n_F = (z/k)²`, the full credibility
/// standards in expected claims are `n_F` for the frequency, `n_F cv²` for
/// the severity and `n_F (1 + cv²)` for the pure premium, `cv` the
/// severity's coefficient of variation. With `n` claims, partial
/// credibility is the square root rule `Z = min(1, √(n / standard))`.
///
/// ```
/// use prospicio_pricing::credibility::LimitedFluctuation;
///
/// // 90% probability of being within 5%: 1,082 claims.
/// let lf = LimitedFluctuation::new(0.90, 0.05).unwrap();
/// assert!((lf.frequency_standard() - 1082.21).abs() < 0.01);
/// assert!((lf.credibility(270.55, lf.frequency_standard()) - 0.5).abs() < 1e-3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimitedFluctuation {
    p: f64,
    k: f64,
}

impl LimitedFluctuation {
    /// Probability `p` in `(0, 1)` of lying within `k > 0` of the mean.
    pub fn new(p: f64, k: f64) -> Result<Self> {
        if !(p > 0.0 && p < 1.0) {
            return Err(Error::InvalidParameter {
                name: "p",
                value: p,
                reason: "must be in (0, 1)",
            });
        }
        if !(k.is_finite() && k > 0.0) {
            return Err(Error::InvalidParameter {
                name: "k",
                value: k,
                reason: "must be finite and positive",
            });
        }
        Ok(Self { p, k })
    }

    /// The probability `p` of lying within `k` of the mean.
    pub fn p(&self) -> f64 {
        self.p
    }

    /// The tolerance `k`, relative to the mean.
    pub fn k(&self) -> f64 {
        self.k
    }

    /// `(Φ⁻¹((1 + p)/2) / k)²`, the full standard for the frequency.
    pub fn frequency_standard(&self) -> f64 {
        let z = norm_quantile(0.5 * (1.0 + self.p));
        (z / self.k).powi(2)
    }

    /// `n_F cv²`, the full standard for the severity in claims.
    pub fn severity_standard(&self, cv: f64) -> f64 {
        self.frequency_standard() * cv * cv
    }

    /// `n_F (1 + cv²)`, the full standard for the pure premium in claims.
    pub fn pure_premium_standard(&self, cv: f64) -> f64 {
        self.frequency_standard() * (1.0 + cv * cv)
    }

    /// `min(1, √(n / standard))`, the credibility of `n` claims against a
    /// full standard.
    pub fn credibility(&self, n: f64, standard: f64) -> f64 {
        if n.is_nan() || n <= 0.0 {
            return 0.0;
        }
        (n / standard).sqrt().min(1.0)
    }
}

/// How the between-group variance `a` is estimated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StructureEstimator {
    /// Bühlmann and Straub's unbiased estimator (`actuar::cm`'s
    /// `"Buhlmann-Gisler"` and `"Ohlsson"`, which coincide with one level).
    Unbiased,
    /// Bichsel and Straub's pseudo-estimator, iterated from the unbiased
    /// one until its relative change is below `tol` (`actuar::cm`'s
    /// `"iterative"`). With equal weights it is the unbiased estimator.
    Iterative { tol: f64, max_iter: usize },
}

impl StructureEstimator {
    /// The iterative estimator with `actuar`'s defaults, `tol = √ε` and 100
    /// iterations.
    pub fn iterative() -> Self {
        Self::Iterative {
            tol: f64::EPSILON.sqrt(),
            max_iter: 100,
        }
    }
}

/// The Bühlmann–Straub credibility model: group `i` in period `j` has
/// ratio `X_ij` (loss per unit of exposure) with weight `w_ij`, and
/// `E[X_ij | Θ_i] = μ(Θ_i)`, `Var[X_ij | Θ_i] = σ²(Θ_i) / w_ij`. Bühlmann's
/// model is the case of equal weights.
///
/// [`BuhlmannStraub::fit`] estimates the within variance
/// `s² = Σ w_ij (X_ij - X̄_i)² / (n - I)` (`n` observations, `I` groups
/// with weight), the between variance `a` by the chosen
/// [`StructureEstimator`], the credibility factors
/// `Z_i = w_i / (w_i + s²/a)`, the collective mean `Σ Z_i X̄_i / Σ Z_i`
/// and the premiums `μ̂ + Z_i (X̄_i - μ̂)`. When `a <= 0` every factor is 0
/// and the collective mean is the weighted mean, as `actuar::cm`.
///
/// ```
/// use prospicio_pricing::credibility::{BuhlmannStraub, StructureEstimator};
///
/// // Two groups of three years each, equal weights (Bühlmann).
/// let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 9.0]];
/// let fit = BuhlmannStraub::new(StructureEstimator::Unbiased).fit(&x, None).unwrap();
/// // s² = (2 + 14) / 4 = 4; a = Var(2, 6) - s²/3 = 8 - 4/3.
/// assert!((fit.within_variance - 4.0).abs() < 1e-12);
/// assert!((fit.between_variance - (8.0 - 4.0 / 3.0)).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuhlmannStraub {
    pub estimator: StructureEstimator,
}

/// The fitted Bühlmann–Straub model.
#[derive(Debug, Clone, PartialEq)]
pub struct BuhlmannStraubFit {
    /// The credibility-weighted collective mean `μ̂`.
    pub collective_mean: f64,
    /// The between-group variance `a` (may be negative for the unbiased
    /// estimator; the factors are then 0).
    pub between_variance: f64,
    /// The within-group variance `s²`.
    pub within_variance: f64,
    /// Each group's weighted mean `X̄_i` (0 for a group with no weight).
    pub group_means: Vec<f64>,
    /// Each group's total weight `w_i`.
    pub group_weights: Vec<f64>,
    /// Each group's credibility factor `Z_i`.
    pub credibility: Vec<f64>,
    /// Each group's credibility premium `μ̂ + Z_i (X̄_i - μ̂)`.
    pub premiums: Vec<f64>,
    /// Iterations of the iterative estimator (0 for the unbiased one).
    pub iterations: usize,
}

impl BuhlmannStraubFit {
    /// Bühlmann's `k = s² / a`, infinite when `a <= 0`.
    pub fn k(&self) -> f64 {
        if self.between_variance > 0.0 {
            self.within_variance / self.between_variance
        } else {
            f64::INFINITY
        }
    }
}

impl Default for BuhlmannStraub {
    fn default() -> Self {
        Self::new(StructureEstimator::Unbiased)
    }
}

impl BuhlmannStraub {
    pub fn new(estimator: StructureEstimator) -> Self {
        Self { estimator }
    }

    /// Fits to `ratios[i][j]` (group `i`, period `j`) with `weights` of the
    /// same shape, or equal weights (Bühlmann) when `None`. A missing cell
    /// is NaN in both, in the same places; missing ratios need weights.
    /// Needs at least two groups and more observations than groups with
    /// weight.
    pub fn fit(
        &self,
        ratios: &[Vec<f64>],
        weights: Option<&[Vec<f64>]>,
    ) -> Result<BuhlmannStraubFit> {
        let groups = ratios.len();
        if groups < 2 {
            return Err(Error::InvalidParameter {
                name: "ratios",
                value: groups as f64,
                reason: "needs at least two groups",
            });
        }
        let ones: Vec<Vec<f64>>;
        let weights = match weights {
            Some(w) => {
                if w.len() != groups || w.iter().zip(ratios).any(|(a, b)| a.len() != b.len()) {
                    return Err(Error::InvalidParameter {
                        name: "weights",
                        value: w.len() as f64,
                        reason: "must have the shape of ratios",
                    });
                }
                w
            }
            None => {
                if ratios.iter().flatten().any(|x| x.is_nan()) {
                    return Err(Error::InvalidParameter {
                        name: "ratios",
                        value: f64::NAN,
                        reason: "missing ratios need weights",
                    });
                }
                ones = ratios.iter().map(|r| vec![1.0; r.len()]).collect();
                &ones
            }
        };
        for (r, w) in ratios.iter().zip(weights) {
            for (&x, &v) in r.iter().zip(w) {
                if x.is_nan() != v.is_nan() {
                    return Err(Error::InvalidParameter {
                        name: "weights",
                        value: v,
                        reason: "missing values must be in the same places in ratios and weights",
                    });
                }
                if !x.is_nan() && !(x.is_finite() && v.is_finite() && v >= 0.0) {
                    return Err(Error::InvalidParameter {
                        name: "weights",
                        value: v,
                        reason: "ratios must be finite and weights finite and non-negative",
                    });
                }
            }
        }

        let mut group_weights = Vec::with_capacity(groups);
        let mut group_means = Vec::with_capacity(groups);
        for (r, w) in ratios.iter().zip(weights) {
            let (mut ws, mut xs) = (0.0, 0.0);
            for (&x, &v) in r.iter().zip(w) {
                if !x.is_nan() {
                    ws += v;
                    xs += v * x;
                }
            }
            group_weights.push(ws);
            group_means.push(if ws > 0.0 { xs / ws } else { 0.0 });
        }
        let contracts = group_weights.iter().filter(|&&w| w > 0.0).count();
        let observations = ratios.iter().flatten().filter(|x| !x.is_nan()).count();
        if contracts < 2 || observations <= contracts {
            return Err(Error::InvalidParameter {
                name: "ratios",
                value: observations as f64,
                reason: "needs two groups with weight and more observations than groups",
            });
        }
        let mut ss = 0.0;
        for ((r, w), m) in ratios.iter().zip(weights).zip(&group_means) {
            for (&x, &v) in r.iter().zip(w) {
                if !x.is_nan() {
                    ss += v * (x - m).powi(2);
                }
            }
        }
        let s2 = ss / (observations - contracts) as f64;

        let mut a = unbiased_between(&group_means, &group_weights, s2, contracts);
        let mut iterations = 0;
        let unequal = {
            let all: Vec<f64> = weights
                .iter()
                .flatten()
                .copied()
                .filter(|v| !v.is_nan())
                .collect();
            let lo = all.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = all.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            hi - lo > f64::EPSILON.sqrt()
        };
        if let (StructureEstimator::Iterative { tol, max_iter }, true) = (self.estimator, unequal) {
            if a > 0.0 {
                let (value, n) = iterative_between(
                    &group_means,
                    &group_weights,
                    s2,
                    contracts,
                    a,
                    tol,
                    max_iter,
                );
                a = value;
                iterations = n;
            } else {
                a = 0.0;
            }
        }

        let (credibility, collective_mean) = if a > 0.0 {
            let z: Vec<f64> = group_weights
                .iter()
                .map(|&w| 1.0 / (1.0 + s2 / (w * a)))
                .collect();
            let zs: f64 = z.iter().sum();
            let m = z.iter().zip(&group_means).map(|(z, x)| z * x).sum::<f64>() / zs;
            (z, m)
        } else {
            let ws: f64 = group_weights.iter().sum();
            let m = group_weights
                .iter()
                .zip(&group_means)
                .map(|(w, x)| w * x)
                .sum::<f64>()
                / ws;
            (vec![0.0; groups], m)
        };
        let premiums = credibility
            .iter()
            .zip(&group_means)
            .map(|(z, x)| collective_mean + z * (x - collective_mean))
            .collect();
        Ok(BuhlmannStraubFit {
            collective_mean,
            between_variance: a,
            within_variance: s2,
            group_means,
            group_weights,
            credibility,
            premiums,
            iterations,
        })
    }
}

/// `w (Σ w_i (X̄_i - X̄)² - (I - 1) s²) / (w² - Σ w_i²)`.
fn unbiased_between(means: &[f64], weights: &[f64], s2: f64, contracts: usize) -> f64 {
    let ws: f64 = weights.iter().sum();
    let xw = weights.iter().zip(means).map(|(w, x)| w * x).sum::<f64>() / ws;
    let dev: f64 = weights
        .iter()
        .zip(means)
        .map(|(w, x)| w * (x - xw).powi(2))
        .sum();
    let w2: f64 = weights.iter().map(|w| w * w).sum();
    ws * (dev - (contracts as f64 - 1.0) * s2) / (ws * ws - w2)
}

/// Bichsel–Straub: `a = Σ Z_i (X̄_i - X̄_Z)² / (I - 1)` with the factors of
/// the previous `a`, to a relative change below `tol`.
fn iterative_between(
    means: &[f64],
    weights: &[f64],
    s2: f64,
    contracts: usize,
    start: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, usize) {
    let mut a = start;
    for count in 1..=max_iter {
        let previous = a;
        let z: Vec<f64> = weights
            .iter()
            .map(|&w| 1.0 / (1.0 + s2 / (w * a)))
            .collect();
        let zs: f64 = z.iter().sum();
        let xz = z.iter().zip(means).map(|(z, x)| z * x).sum::<f64>() / zs;
        a = z
            .iter()
            .zip(means)
            .map(|(z, x)| z * (x - xz).powi(2))
            .sum::<f64>()
            / (contracts as f64 - 1.0);
        if ((a - previous) / previous).abs() < tol {
            return (a, count);
        }
    }
    (a, max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limited_fluctuation_standards() {
        let lf = LimitedFluctuation::new(0.90, 0.05).unwrap();
        let n = lf.frequency_standard();
        assert!((lf.pure_premium_standard(2.0) - 5.0 * n).abs() < 1e-9);
        assert!((lf.severity_standard(2.0) - 4.0 * n).abs() < 1e-9);
        assert_eq!(lf.credibility(2.0 * n, n), 1.0);
        assert_eq!(lf.credibility(0.0, n), 0.0);
        assert!(LimitedFluctuation::new(1.0, 0.05).is_err());
    }

    #[test]
    fn unequal_weights_and_missing_cells() {
        let nan = f64::NAN;
        let x = vec![
            vec![1.0, 2.0, nan],
            vec![4.0, 5.0, 9.0],
            vec![2.0, 2.5, 3.0],
        ];
        let w = vec![
            vec![10.0, 20.0, nan],
            vec![5.0, 5.0, 8.0],
            vec![30.0, 40.0, 50.0],
        ];
        let fit = BuhlmannStraub::default().fit(&x, Some(&w)).unwrap();
        assert!((fit.group_means[0] - 50.0 / 30.0).abs() < 1e-12);
        assert_eq!(fit.group_weights, vec![30.0, 18.0, 120.0]);
        let it = BuhlmannStraub::new(StructureEstimator::iterative())
            .fit(&x, Some(&w))
            .unwrap();
        assert!(it.iterations > 0);
        // Misplaced missing values and too little data are refused.
        let bad = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 9.0],
            vec![2.0, 2.5, 3.0],
        ];
        assert!(BuhlmannStraub::default().fit(&x, Some(&bad)).is_err());
        assert!(BuhlmannStraub::default().fit(&x, None).is_err());
        assert!(BuhlmannStraub::default().fit(&x[..1], None).is_err());
    }
}
