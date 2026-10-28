//! Classical credibility: limited fluctuation standards, the
//! Bühlmann–Straub model and Hachemeister's regression model
//! (`docs/design/credibility.md`).
//!
//! Hierarchical and Bayesian credibility belong with v0.7
//! (`docs/roadmap.md`).

use prospicio_core::{Error, Result};
use prospicio_math::linalg::{cholesky, cholesky_inverse, cholesky_solve, solve};
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

/// Hachemeister's regression credibility model: risk `i` in period `t` has
/// ratio `X_it` with weight `w_it` and
/// `E[X_it | Θ_i] = y_tᵀ β(Θ_i)`, `Var[X_it | Θ_i] = σ²(Θ_i) / w_it`, for
/// a design row `y_t` (a trend line `[1, t]`, say) shared by the risks.
/// Each risk's own weighted least squares fit `b_i` is shrunk towards the
/// collective `β̂` by a credibility matrix,
/// `β̂_i = β̂ + Z_i (b_i - β̂)` with `Z_i = A (A + s² W_i)⁻¹`, where `A` is
/// the covariance of `β(Θ)` across risks, `s²` the mean of the risks'
/// residual variances and `W_i = (Y_iᵀ diag(w_i) Y_i)⁻¹`. With one
/// column of ones it is the Bühlmann–Straub model.
///
/// It follows `actuar::cm(..., regformula = ...)` (Goulet's `hache`):
///
/// - at the origin (`adjust_intercept = false`) the matrices `Z_i` are
///   full and `A` is Hachemeister's iterative pseudo-estimator, started
///   from `Z_i = I` and iterated to a relative change of the collective
///   below `tol`, whatever the [`StructureEstimator`] (whose `tol` and
///   `max_iter` it takes when iterative, `actuar`'s defaults otherwise);
/// - with `adjust_intercept = true` (`actuar`'s `adj.intercept`) the
///   design is orthogonalised under the collective weights
///   `w_t = Σ_i w_it / Σ w`, which puts the intercept at the barycentre of
///   the periods: the `Z_i` and `A` are then diagonal and each diagonal is
///   estimated as Bühlmann–Straub's `a`, by the chosen estimator (the
///   design's first column must then be constant).
///
/// Coefficients are reported in the design's own basis; the credibility
/// and between-risk covariance matrices in the basis fitted, the
/// orthogonalised one when adjusted ([`HachemeisterFit::transition`] maps
/// it back).
///
/// ```
/// use prospicio_pricing::credibility::{Hachemeister, StructureEstimator};
///
/// // Three risks, five periods, a linear trend [1, t].
/// let x = vec![
///     vec![10.0, 12.0, 13.0, 15.0, 16.0],
///     vec![8.0, 8.5, 9.5, 9.0, 10.0],
///     vec![14.0, 13.0, 16.0, 17.0, 19.0],
/// ];
/// let design: Vec<Vec<f64>> = (1..=5).map(|t| vec![1.0, t as f64]).collect();
/// let fit = Hachemeister::new(StructureEstimator::Unbiased, true)
///     .fit(&x, None, &design)
///     .unwrap();
/// // Each risk's line, shrunk towards the collective one.
/// let next = fit.predict(&[1.0, 6.0]).unwrap();
/// assert_eq!(next.len(), 3);
/// assert!(next[1] < next[0] && next[0] < next[2]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hachemeister {
    pub estimator: StructureEstimator,
    pub adjust_intercept: bool,
}

/// The fitted Hachemeister model.
#[derive(Debug, Clone, PartialEq)]
pub struct HachemeisterFit {
    /// The collective coefficients `β̂`, in the design's basis.
    pub collective: Vec<f64>,
    /// Each risk's credibility coefficients `β̂ + Z_i (b_i - β̂)`, in the
    /// design's basis (the collective for a risk with no weight).
    pub coefficients: Vec<Vec<f64>>,
    /// Each risk's own weighted least squares coefficients `b_i`, in the
    /// design's basis (0 for a risk with no weight).
    pub individual: Vec<Vec<f64>>,
    /// Each risk's credibility matrix `Z_i` in the basis fitted (0 for a
    /// risk with no weight).
    pub credibility: Vec<Vec<Vec<f64>>>,
    /// The between-risk covariance `A` in the basis fitted (diagonal when
    /// adjusted; a diagonal may be negative for the unbiased estimator,
    /// whose factors are then 0).
    pub between: Vec<Vec<f64>>,
    /// The within-risk variance `s²`, the mean of the risks' residual
    /// variances `Σ_t w_it r_it² / (n_i - p)` (0 for a risk with exactly `p`
    /// observations).
    pub within_variance: f64,
    /// Each risk's total weight.
    pub risk_weights: Vec<f64>,
    /// When adjusted, the upper triangular `R` with the design `Y = Q R`,
    /// `Q` orthonormal under the collective weights: a coefficient `c` in
    /// the basis fitted is `R⁻¹ c` in the design's. `None` at the origin.
    pub transition: Option<Vec<Vec<f64>>>,
    /// Iterations: of the collective at the origin, the most of any
    /// coefficient's iterative estimator when adjusted (0 for unbiased).
    pub iterations: usize,
}

impl HachemeisterFit {
    /// Each risk's credibility premium at the design row `x`:
    /// `xᵀ β̂_i`.
    pub fn predict(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.collective.len() {
            return Err(Error::InvalidParameter {
                name: "x",
                value: x.len() as f64,
                reason: "must have one value per design column",
            });
        }
        Ok(self
            .coefficients
            .iter()
            .map(|b| b.iter().zip(x).map(|(b, x)| b * x).sum())
            .collect())
    }
}

impl Default for Hachemeister {
    fn default() -> Self {
        Self::new(StructureEstimator::Unbiased, false)
    }
}

/// One risk's weighted least squares fit.
struct RiskFit {
    coef: Vec<f64>,
    /// `(Yᵀ W Y)⁻¹`, row-major.
    w_inv: Vec<f64>,
    sigma2: f64,
}

/// The weighted least squares fit of `y` on `design` over the observed
/// cells with positive weight; `None` when they do not identify every
/// coefficient.
fn wls(y: &[f64], w: &[f64], design: &[Vec<f64>], p: usize) -> Option<RiskFit> {
    let mut info = vec![0.0; p * p];
    let mut rhs = vec![0.0; p];
    let mut n = 0usize;
    for ((&yt, &wt), row) in y.iter().zip(w).zip(design) {
        if yt.is_nan() || wt <= 0.0 {
            continue;
        }
        n += 1;
        for j in 0..p {
            rhs[j] += wt * row[j] * yt;
            for k in 0..p {
                info[j * p + k] += wt * row[j] * row[k];
            }
        }
    }
    let l = cholesky(&info, p)?;
    // A pivot lost to cancellation is a rank deficiency, not a fit.
    if (0..p).any(|j| l[j * p + j].powi(2) <= 1e-12 * info[j * p + j]) {
        return None;
    }
    let coef = cholesky_solve(&l, &rhs);
    let mut ss = 0.0;
    for ((&yt, &wt), row) in y.iter().zip(w).zip(design) {
        if yt.is_nan() || wt <= 0.0 {
            continue;
        }
        let fitted: f64 = row.iter().zip(&coef).map(|(a, b)| a * b).sum();
        ss += wt * (yt - fitted).powi(2);
    }
    let sigma2 = if n > p { ss / (n - p) as f64 } else { 0.0 };
    Some(RiskFit {
        coef,
        w_inv: cholesky_inverse(&l, p),
        sigma2,
    })
}

/// `a b` for row-major `p × p` matrices.
fn mat_mul(a: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut c = vec![0.0; p * p];
    for i in 0..p {
        for k in 0..p {
            for j in 0..p {
                c[i * p + j] += a[i * p + k] * b[k * p + j];
            }
        }
    }
    c
}

/// `a v` for a row-major `p × p` matrix.
fn mat_vec(a: &[f64], v: &[f64], p: usize) -> Vec<f64> {
    (0..p)
        .map(|i| (0..p).map(|j| a[i * p + j] * v[j]).sum())
        .collect()
}

/// `a m⁻¹` for a row-major `a` and an invertible symmetric `m`, column by
/// column of `(m⁻¹ aᵀ)`.
fn right_divide(a: &[f64], m: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut out = vec![0.0; p * p];
    for i in 0..p {
        // Row i of a m⁻¹ solves m xᵀ = (row i of a)ᵀ, m symmetric.
        let row = solve(m.to_vec(), a[i * p..(i + 1) * p].to_vec(), p)?;
        out[i * p..(i + 1) * p].copy_from_slice(&row);
    }
    Some(out)
}

fn nested(m: &[f64], p: usize) -> Vec<Vec<f64>> {
    m.chunks(p).map(<[f64]>::to_vec).collect()
}

/// `R⁻¹ c` for an upper triangular row-major `R`.
fn upper_solve(r: &[f64], c: &[f64], p: usize) -> Vec<f64> {
    let mut x = vec![0.0; p];
    for i in (0..p).rev() {
        let s: f64 = (i + 1..p).map(|j| r[i * p + j] * x[j]).sum();
        x[i] = (c[i] - s) / r[i * p + i];
    }
    x
}

impl Hachemeister {
    pub fn new(estimator: StructureEstimator, adjust_intercept: bool) -> Self {
        Self {
            estimator,
            adjust_intercept,
        }
    }

    /// Fits to `ratios[i][t]` (risk `i`, period `t`) with `weights` of the
    /// same shape, or equal weights when `None`, on `design[t]`, one row
    /// of `p` values per period. A missing cell is NaN in both ratios and
    /// weights. Needs at least two risks with weight, each with enough
    /// observed cells of positive weight to identify its `p` coefficients
    /// (`p` of them in general position).
    pub fn fit(
        &self,
        ratios: &[Vec<f64>],
        weights: Option<&[Vec<f64>]>,
        design: &[Vec<f64>],
    ) -> Result<HachemeisterFit> {
        let risks = ratios.len();
        let periods = design.len();
        let p = design.first().map_or(0, Vec::len);
        if p == 0
            || design
                .iter()
                .any(|r| r.len() != p || r.iter().any(|v| !v.is_finite()))
        {
            return Err(Error::InvalidParameter {
                name: "design",
                value: p as f64,
                reason: "must have one row of the same positive length per period, all finite",
            });
        }
        if risks < 2 {
            return Err(Error::InvalidParameter {
                name: "ratios",
                value: risks as f64,
                reason: "needs at least two risks",
            });
        }
        if ratios.iter().any(|r| r.len() != periods) {
            return Err(Error::InvalidParameter {
                name: "ratios",
                value: periods as f64,
                reason: "every risk needs one value per design row",
            });
        }
        let ones: Vec<Vec<f64>>;
        let weights = match weights {
            Some(w) => {
                if w.len() != risks || w.iter().any(|r| r.len() != periods) {
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
        let risk_weights: Vec<f64> = weights
            .iter()
            .map(|w| w.iter().filter(|v| !v.is_nan()).sum())
            .collect();
        let has_data: Vec<usize> = (0..risks).filter(|&i| risk_weights[i] > 0.0).collect();
        let contracts = has_data.len();
        if contracts < 2 {
            return Err(Error::InvalidParameter {
                name: "weights",
                value: contracts as f64,
                reason: "needs at least two risks with weight",
            });
        }
        let unequal = {
            let all = weights.iter().flatten().copied().filter(|v| !v.is_nan());
            let (lo, hi) = all.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(v), hi.max(v))
            });
            hi - lo > f64::EPSILON.sqrt()
        };

        // The basis fitted: the design, or its orthogonalisation under the
        // collective weights.
        let (basis, transition) = if self.adjust_intercept {
            let first = design[0][0];
            if first == 0.0 || design.iter().any(|r| r[0] != first) {
                return Err(Error::InvalidParameter {
                    name: "design",
                    value: first,
                    reason: "the intercept is adjusted only for a constant first column",
                });
            }
            let total: f64 = risk_weights.iter().sum();
            let wt: Vec<f64> = (0..periods)
                .map(|t| {
                    weights
                        .iter()
                        .map(|w| w[t])
                        .filter(|v| !v.is_nan())
                        .sum::<f64>()
                        / total
                })
                .collect();
            if wt.iter().any(|&v| v <= 0.0) {
                return Err(Error::InvalidParameter {
                    name: "weights",
                    value: 0.0,
                    reason: "the intercept is adjusted only when every period has weight",
                });
            }
            // Modified Gram–Schmidt on sqrt(w) Y: Q orthonormal, R upper
            // triangular with a positive diagonal.
            let mut q: Vec<Vec<f64>> = (0..p)
                .map(|j| (0..periods).map(|t| wt[t].sqrt() * design[t][j]).collect())
                .collect();
            let mut r = vec![0.0; p * p];
            for j in 0..p {
                for k in 0..j {
                    let d: f64 = q[k].iter().zip(&q[j]).map(|(a, b)| a * b).sum();
                    r[k * p + j] = d;
                    let qk = q[k].clone();
                    for (v, a) in q[j].iter_mut().zip(qk) {
                        *v -= d * a;
                    }
                }
                let norm: f64 = q[j].iter().map(|v| v * v).sum::<f64>().sqrt();
                let scale: f64 = (0..periods)
                    .map(|t| wt[t] * design[t][j] * design[t][j])
                    .sum::<f64>()
                    .sqrt();
                if norm.is_nan() || norm <= 1e-10 * scale {
                    return Err(Error::InvalidParameter {
                        name: "design",
                        value: j as f64,
                        reason: "the design's columns must be linearly independent",
                    });
                }
                r[j * p + j] = norm;
                for v in &mut q[j] {
                    *v /= norm;
                }
            }
            let basis: Vec<Vec<f64>> = (0..periods)
                .map(|t| (0..p).map(|j| q[j][t] / wt[t].sqrt()).collect())
                .collect();
            (basis, Some(r))
        } else {
            (design.to_vec(), None)
        };

        let mut fits: Vec<Option<RiskFit>> = Vec::with_capacity(risks);
        for i in 0..risks {
            if risk_weights[i] > 0.0 {
                let fit = wls(&ratios[i], &weights[i], &basis, p).ok_or_else(|| {
                    Error::Data(format!(
                        "risk {i}'s observed cells with weight do not identify the design's {p} \
                         coefficients"
                    ))
                })?;
                fits.push(Some(fit));
            } else {
                fits.push(None);
            }
        }
        let ind: Vec<Vec<f64>> = fits
            .iter()
            .map(|f| f.as_ref().map_or(vec![0.0; p], |f| f.coef.clone()))
            .collect();
        let s2 = has_data
            .iter()
            .map(|&i| fits[i].as_ref().map_or(0.0, |f| f.sigma2))
            .sum::<f64>()
            / contracts as f64;

        let (tol, max_iter) = match self.estimator {
            StructureEstimator::Iterative { tol, max_iter } => (tol, max_iter),
            StructureEstimator::Unbiased => (f64::EPSILON.sqrt(), 100),
        };
        let mut cred = vec![vec![0.0; p * p]; risks];
        let mut a_mat = vec![0.0; p * p];
        let mut coll = vec![0.0; p];
        let mut iterations = 0;
        let singular = || Error::Data("A + s² W_i is singular for a risk".into());

        if self.adjust_intercept {
            for k in 0..p {
                let x: Vec<f64> = has_data.iter().map(|&i| ind[i][k]).collect();
                let w: Vec<f64> = has_data
                    .iter()
                    .map(|&i| {
                        (0..periods)
                            .filter(|&t| !weights[i][t].is_nan())
                            .map(|t| weights[i][t] * basis[t][k].powi(2))
                            .sum()
                    })
                    .collect();
                let mut a = unbiased_between(&x, &w, s2, contracts);
                if matches!(self.estimator, StructureEstimator::Iterative { .. }) && unequal {
                    a = if a > 0.0 {
                        let (value, n) = iterative_between(&x, &w, s2, contracts, a, tol, max_iter);
                        iterations = iterations.max(n);
                        value
                    } else {
                        0.0
                    };
                }
                a_mat[k * p + k] = a;
                if a > 0.0 {
                    let z: Vec<f64> = w.iter().map(|&w| 1.0 / (1.0 + s2 / (w * a))).collect();
                    coll[k] =
                        z.iter().zip(&x).map(|(z, x)| z * x).sum::<f64>() / z.iter().sum::<f64>();
                    for (&i, z) in has_data.iter().zip(z) {
                        cred[i][k * p + k] = z;
                    }
                } else {
                    coll[k] =
                        w.iter().zip(&x).map(|(w, x)| w * x).sum::<f64>() / w.iter().sum::<f64>();
                }
            }
        } else {
            for &i in &has_data {
                for j in 0..p {
                    cred[i][j * p + j] = 1.0;
                }
            }
            for &i in &has_data {
                for (c, b) in coll.iter_mut().zip(&ind[i]) {
                    *c += b / contracts as f64;
                }
            }
            // A from the current factors, symmetrised, then each Z_i.
            let update = |coll: &[f64], cred: &mut Vec<Vec<f64>>| -> Result<Vec<f64>> {
                let mut a = vec![0.0; p * p];
                for &i in &has_data {
                    let d: Vec<f64> = ind[i].iter().zip(coll).map(|(b, c)| b - c).collect();
                    let outer: Vec<f64> = (0..p * p).map(|jk| d[jk / p] * d[jk % p]).collect();
                    for (a, v) in a.iter_mut().zip(mat_mul(&cred[i], &outer, p)) {
                        *a += v / (contracts as f64 - 1.0);
                    }
                }
                let a: Vec<f64> = (0..p * p)
                    .map(|jk| (a[jk] + a[(jk % p) * p + jk / p]) / 2.0)
                    .collect();
                for &i in &has_data {
                    let w_inv = &fits[i]
                        .as_ref()
                        .expect("a risk with weight is fitted")
                        .w_inv;
                    let m: Vec<f64> = a.iter().zip(w_inv).map(|(a, w)| a + s2 * w).collect();
                    cred[i] = right_divide(&a, &m, p).ok_or_else(singular)?;
                }
                Ok(a)
            };
            for count in 1..=max_iter {
                let previous = coll.clone();
                update(&coll, &mut cred)?;
                let mut cred_sum = vec![0.0; p * p];
                let mut rhs = vec![0.0; p];
                for &i in &has_data {
                    for (s, z) in cred_sum.iter_mut().zip(&cred[i]) {
                        *s += z;
                    }
                    for (r, v) in rhs.iter_mut().zip(mat_vec(&cred[i], &ind[i], p)) {
                        *r += v;
                    }
                }
                coll = solve(cred_sum, rhs, p).ok_or_else(|| {
                    Error::Data("the sum of the credibility matrices is singular".into())
                })?;
                iterations = count;
                let change = coll
                    .iter()
                    .zip(&previous)
                    .map(|(c, p)| ((c - p) / p).abs())
                    .fold(0.0, f64::max);
                if change < tol {
                    break;
                }
            }
            a_mat = update(&coll, &mut cred)?;
        }

        let coef_fitted: Vec<Vec<f64>> = (0..risks)
            .map(|i| {
                let d: Vec<f64> = ind[i].iter().zip(&coll).map(|(b, c)| b - c).collect();
                coll.iter()
                    .zip(mat_vec(&cred[i], &d, p))
                    .map(|(c, z)| c + z)
                    .collect()
            })
            .collect();
        let back = |c: &[f64]| match &transition {
            Some(r) => upper_solve(r, c, p),
            None => c.to_vec(),
        };
        Ok(HachemeisterFit {
            collective: back(&coll),
            coefficients: coef_fitted.iter().map(|c| back(c)).collect(),
            individual: (0..risks)
                .map(|i| {
                    if risk_weights[i] > 0.0 {
                        back(&ind[i])
                    } else {
                        vec![0.0; p]
                    }
                })
                .collect(),
            credibility: cred.iter().map(|z| nested(z, p)).collect(),
            between: nested(&a_mat, p),
            within_variance: s2,
            risk_weights,
            transition: transition.map(|r| nested(&r, p)),
            iterations,
        })
    }
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

    #[test]
    fn hachemeister_with_an_intercept_alone_is_buhlmann_straub() {
        let nan = f64::NAN;
        let x = vec![
            vec![1.0, 2.0, 1.5, 2.5],
            vec![4.0, 5.0, 9.0, 6.0],
            vec![2.0, 2.5, 3.0, 2.0],
        ];
        let w = vec![
            vec![10.0, 20.0, 15.0, 5.0],
            vec![5.0, 5.0, 8.0, 7.0],
            vec![30.0, 40.0, 50.0, 20.0],
        ];
        let ones = vec![vec![1.0]; 4];
        for estimator in [
            StructureEstimator::Unbiased,
            StructureEstimator::iterative(),
        ] {
            let bs = BuhlmannStraub::new(estimator).fit(&x, Some(&w)).unwrap();
            let h = Hachemeister::new(estimator, true)
                .fit(&x, Some(&w), &ones)
                .unwrap();
            assert!((h.within_variance - bs.within_variance).abs() < 1e-12 * bs.within_variance);
            assert!((h.between[0][0] - bs.between_variance).abs() < 1e-9 * bs.between_variance);
            assert!((h.collective[0] - bs.collective_mean).abs() < 1e-10);
            for i in 0..3 {
                assert!((h.coefficients[i][0] - bs.premiums[i]).abs() < 1e-10);
                assert!((h.credibility[i][0][0] - bs.credibility[i]).abs() < 1e-10);
            }
            // At the origin the factor is the iterative one in any case.
            let o = Hachemeister::new(estimator, false)
                .fit(&x, Some(&w), &ones)
                .unwrap();
            let it = BuhlmannStraub::new(StructureEstimator::iterative())
                .fit(&x, Some(&w))
                .unwrap();
            assert!((o.collective[0] - it.collective_mean).abs() < 1e-6);
        }
        // A risk with no weight takes the collective.
        let mut x0 = x.clone();
        let mut w0 = w.clone();
        x0.push(vec![nan; 4]);
        w0.push(vec![nan; 4]);
        let h = Hachemeister::default().fit(&x0, Some(&w0), &ones).unwrap();
        assert_eq!(h.coefficients[3], h.collective);
        assert_eq!(h.individual[3], vec![0.0]);
    }

    #[test]
    fn hachemeister_refuses_what_it_cannot_fit() {
        let x = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 2.0, 5.0],
            vec![1.0, 3.0, 4.0],
        ];
        let line: Vec<Vec<f64>> = (1..=3).map(|t| vec![1.0, t as f64]).collect();
        assert!(Hachemeister::default().fit(&x, None, &line).is_ok());
        // Two risks cannot identify a 2 × 2 between covariance at the origin.
        assert!(Hachemeister::default().fit(&x[..2], None, &line).is_err());
        // Collinear columns.
        let bad: Vec<Vec<f64>> = (1..=3)
            .map(|t| vec![1.0, 2.0 * t as f64, t as f64])
            .collect();
        assert!(Hachemeister::default().fit(&x, None, &bad).is_err());
        assert!(
            Hachemeister::new(StructureEstimator::Unbiased, true)
                .fit(&x, None, &bad)
                .is_err()
        );
        // A risk with one observation cannot fit a line.
        let nan = f64::NAN;
        let x1 = vec![
            vec![1.0, nan, nan],
            vec![2.0, 2.0, 5.0],
            vec![1.0, 3.0, 4.0],
        ];
        let w1 = vec![
            vec![1.0, nan, nan],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        let err = Hachemeister::default()
            .fit(&x1, Some(&w1), &line)
            .unwrap_err();
        assert!(err.to_string().contains("risk 0"));
        // Adjusting the intercept needs a constant first column.
        let flipped: Vec<Vec<f64>> = (1..=3).map(|t| vec![t as f64, 1.0]).collect();
        assert!(
            Hachemeister::new(StructureEstimator::Unbiased, true)
                .fit(&x, None, &flipped)
                .is_err()
        );
        assert!(Hachemeister::default().fit(&x[..1], None, &line).is_err());
        assert!(
            Hachemeister::default()
                .fit(&x, None, &line)
                .unwrap()
                .predict(&[1.0])
                .is_err()
        );
    }
}
