//! Multivariate normal and lognormal joints from a mean and a covariance.
//!
//! Many capital-model inputs arrive as a mean vector and a covariance
//! matrix, not as draws: a reserve by origin with Mack's covariance across
//! origins, lines aggregated by a correlation matrix. [`MultivariateNormal`]
//! and [`MultivariateLognormal`] hold such a joint with its exact marginals
//! and moments, and [`PredictiveDistribution::from_gaussian`] draws it on
//! the library's seeded streams.
//!
//! The covariance must be positive definite: it is factored by Cholesky,
//! and a singular matrix (a component with zero variance, two components
//! perfectly correlated) is refused (`docs/design/predictive-distribution.md`,
//! "Gaussian joints").
//!
//! Simulation `i` draws `z_j = Φ⁻¹(u_j)` from `StreamRng::new(seed, i)`,
//! one uniform per component in order, and returns `mean + L z` (the
//! normal) or its exponential (the lognormal), `L` the Cholesky factor:
//! inverse transform, the first sampler of every family.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_math::linalg::{cholesky, lower_mul};
use prospicio_math::special::norm_quantile;

use crate::lognormal::Lognormal;
use crate::normal::Normal;
use crate::predictive::{ComponentKey, PredictiveDistribution};
use crate::provenance::{InputHasher, Provenance};

/// A joint whose draws are a linear map of independent standard normals:
/// [`MultivariateNormal`] and [`MultivariateLognormal`], the joints
/// [`PredictiveDistribution::from_gaussian`] takes.
pub trait GaussianJoint: Sync {
    /// The number of components.
    fn dim(&self) -> usize;

    /// Fills `row` (length [`dim`](Self::dim)) with one draw of the joint,
    /// from `dim` uniforms of `rng`.
    fn sample(&self, rng: &mut StreamRng, row: &mut [f64]);

    /// The model's name and parameters, with its mean and covariance
    /// hashed as its input.
    fn provenance(&self) -> Provenance;
}

/// A multivariate normal with mean `mean` and covariance `cov`.
///
/// ```
/// use prospicio_prob::{Distribution, MultivariateNormal};
///
/// let mvn = MultivariateNormal::new(vec![100.0, 50.0], vec![400.0, 120.0, 120.0, 100.0]).unwrap();
/// let total = mvn.total();
/// assert_eq!(total.mean(), 150.0);
/// // 400 + 2 * 120 + 100
/// assert!((total.variance() - 740.0).abs() < 1e-9);
/// assert_eq!(mvn.marginal(1).unwrap().sigma(), 10.0);
/// assert!(MultivariateNormal::new(vec![0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]).is_err());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateNormal {
    mean: Vec<f64>,
    /// Row-major `d × d`, symmetric.
    cov: Vec<f64>,
    /// Lower Cholesky factor of `cov`, row-major.
    factor: Vec<f64>,
}

impl MultivariateNormal {
    /// The normal with mean `mean` (length `d >= 1`) and covariance `cov`,
    /// row-major `d × d`.
    ///
    /// Fails if a value is not finite, `cov` is not `d × d`, not
    /// symmetric (to `1e-10` of its largest variance) or not positive
    /// definite.
    pub fn new(mean: Vec<f64>, cov: Vec<f64>) -> Result<Self> {
        let factor = check_mean_cov(&mean, &cov, "cov")?;
        Ok(Self { mean, cov, factor })
    }

    /// The number of components.
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// The mean vector.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// The covariance matrix, row-major.
    pub fn cov(&self) -> &[f64] {
        &self.cov
    }

    /// The lower Cholesky factor `L` of the covariance (`L Lᵀ = cov`),
    /// row-major.
    pub fn cholesky_factor(&self) -> &[f64] {
        &self.factor
    }

    /// The correlation matrix, row-major.
    pub fn correlation(&self) -> Vec<f64> {
        correlation(&self.cov, self.dim())
    }

    /// Component `j`'s marginal, exact. Fails if `j` is out of range.
    pub fn marginal(&self, j: usize) -> Result<Normal> {
        let d = self.dim();
        check_index(j, d)?;
        Normal::new(self.mean[j], self.cov[j * d + j].sqrt())
    }

    /// The total over all components, exact: a sum of jointly normal
    /// components is normal, with mean `Σ μ_j` and variance `1ᵀ Σ 1`.
    pub fn total(&self) -> Normal {
        let mean = self.mean.iter().sum();
        let variance: f64 = self.cov.iter().sum();
        // Positive definite, so 1ᵀ Σ 1 > 0.
        Normal::new(mean, variance.sqrt()).expect("a positive definite covariance")
    }
}

impl GaussianJoint for MultivariateNormal {
    fn dim(&self) -> usize {
        self.mean.len()
    }

    fn sample(&self, rng: &mut StreamRng, row: &mut [f64]) {
        let z: Vec<f64> = (0..row.len())
            .map(|_| norm_quantile(rng.next_open01()))
            .collect();
        lower_mul(&self.factor, &z, row);
        for (x, m) in row.iter_mut().zip(&self.mean) {
            *x += m;
        }
    }

    fn provenance(&self) -> Provenance {
        Provenance::new("multivariate_normal")
            .param("dim", self.dim())
            .input_hash(
                InputHasher::new()
                    .str("mean")
                    .f64s(&self.mean)
                    .str("cov")
                    .f64s(&self.cov)
                    .finish(),
            )
    }
}

/// A multivariate lognormal: `exp(Y)` for `Y` multivariate normal with
/// mean `mu` and covariance `log_cov` on the log scale.
///
/// [`from_moments`](Self::from_moments) matches a mean and covariance on
/// the original scale exactly:
/// `log_cov_ij = ln(1 + cov_ij / (m_i m_j))`,
/// `mu_i = ln m_i - log_cov_ii / 2`, the conversion behind Mack's lognormal
/// reserve.
///
/// ```
/// use prospicio_prob::MultivariateLognormal;
///
/// let mean = vec![100.0, 50.0];
/// let cov = vec![400.0, 120.0, 120.0, 100.0];
/// let mvl = MultivariateLognormal::from_moments(mean.clone(), cov.clone()).unwrap();
/// for (got, want) in mvl.mean().iter().zip(&mean) {
///     assert!((got / want - 1.0).abs() < 1e-12);
/// }
/// for (got, want) in mvl.cov().iter().zip(&cov) {
///     assert!((got / want - 1.0).abs() < 1e-12);
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateLognormal {
    log: MultivariateNormal,
}

impl MultivariateLognormal {
    /// The lognormal `exp(Y)`, `Y` normal with mean `mu` and covariance
    /// `log_cov` (row-major `d × d`). Fails as [`MultivariateNormal::new`].
    pub fn from_log_scale(mu: Vec<f64>, log_cov: Vec<f64>) -> Result<Self> {
        let factor = check_mean_cov(&mu, &log_cov, "log_cov")?;
        Ok(Self {
            log: MultivariateNormal {
                mean: mu,
                cov: log_cov,
                factor,
            },
        })
    }

    /// The lognormal with mean `mean` and covariance `cov` on the original
    /// scale, matched exactly.
    ///
    /// Fails if a mean is not positive, `cov` is not a symmetric `d × d`
    /// matrix of finite values, `1 + cov_ij / (m_i m_j)` is not positive
    /// (a covariance too negative for a lognormal), or the covariance on
    /// the log scale is not positive definite (which can happen even when
    /// `cov` is).
    pub fn from_moments(mean: Vec<f64>, cov: Vec<f64>) -> Result<Self> {
        let d = mean.len();
        check_shape(&mean, &cov, "cov")?;
        if let Some(&m) = mean.iter().find(|m| !m.is_finite() || **m <= 0.0) {
            return Err(Error::InvalidParameter {
                name: "mean",
                value: m,
                reason: "must be positive for a lognormal",
            });
        }
        let mut log_cov = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                let ratio = 1.0 + cov[i * d + j] / (mean[i] * mean[j]);
                if ratio.is_nan() || ratio <= 0.0 {
                    return Err(Error::Data(format!(
                        "cov[{i}, {j}] = {} is at or below -mean[{i}] * mean[{j}] = {}: \
                         no lognormal has it",
                        cov[i * d + j],
                        -mean[i] * mean[j]
                    )));
                }
                log_cov[i * d + j] = ratio.ln();
            }
        }
        let mu = (0..d)
            .map(|i| mean[i].ln() - log_cov[i * d + i] / 2.0)
            .collect();
        let factor = cholesky(&log_cov, d).ok_or_else(|| {
            Error::Data(
                "the covariance on the log scale, ln(1 + cov_ij / (m_i m_j)), is not \
                 positive definite"
                    .into(),
            )
        })?;
        Ok(Self {
            log: MultivariateNormal {
                mean: mu,
                cov: log_cov,
                factor,
            },
        })
    }

    /// The number of components.
    pub fn dim(&self) -> usize {
        self.log.dim()
    }

    /// The normal on the log scale, `Y` with `exp(Y)` this joint.
    pub fn log_normal(&self) -> &MultivariateNormal {
        &self.log
    }

    /// The mean on the original scale, exact: `exp(mu_i + log_cov_ii / 2)`.
    pub fn mean(&self) -> Vec<f64> {
        let d = self.dim();
        (0..d)
            .map(|i| (self.log.mean[i] + self.log.cov[i * d + i] / 2.0).exp())
            .collect()
    }

    /// The covariance on the original scale, exact, row-major:
    /// `m_i m_j (exp(log_cov_ij) - 1)`.
    pub fn cov(&self) -> Vec<f64> {
        let d = self.dim();
        let m = self.mean();
        let mut cov = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                cov[i * d + j] = m[i] * m[j] * self.log.cov[i * d + j].exp_m1();
            }
        }
        cov
    }

    /// The correlation matrix on the original scale, row-major.
    pub fn correlation(&self) -> Vec<f64> {
        correlation(&self.cov(), self.dim())
    }

    /// Component `j`'s marginal, exact. Fails if `j` is out of range.
    pub fn marginal(&self, j: usize) -> Result<Lognormal> {
        let d = self.dim();
        check_index(j, d)?;
        Lognormal::new(self.log.mean[j], self.log.cov[j * d + j].sqrt())
    }

    /// The total's mean, exact. A sum of lognormals is not lognormal, so
    /// there is no exact total distribution; draw one with
    /// [`PredictiveDistribution::from_gaussian`].
    pub fn total_mean(&self) -> f64 {
        self.mean().iter().sum()
    }

    /// The total's variance, exact: the sum of the covariance's entries.
    pub fn total_variance(&self) -> f64 {
        self.cov().iter().sum()
    }
}

impl GaussianJoint for MultivariateLognormal {
    fn dim(&self) -> usize {
        self.log.dim()
    }

    fn sample(&self, rng: &mut StreamRng, row: &mut [f64]) {
        self.log.sample(rng, row);
        for x in row.iter_mut() {
            *x = x.exp();
        }
    }

    fn provenance(&self) -> Provenance {
        Provenance::new("multivariate_lognormal")
            .param("dim", self.dim())
            .input_hash(
                InputHasher::new()
                    .str("mu")
                    .f64s(&self.log.mean)
                    .str("log_cov")
                    .f64s(&self.log.cov)
                    .finish(),
            )
    }
}

impl PredictiveDistribution {
    /// `n_sims` draws of a multivariate normal or lognormal joint, one
    /// component per key, on the streams of `seed`
    /// ([`PredictiveDistribution::simulate`]): simulation `i` draws only
    /// from `StreamRng::new(seed, i)`, so the result reproduces by seed on
    /// any number of threads. The provenance records the model, its
    /// dimension, the seed and a hash of the mean and covariance.
    ///
    /// Fails if `components` does not have one key per component of the
    /// joint, or as [`PredictiveDistribution::simulate`].
    ///
    /// ```
    /// use prospicio_prob::{Distribution, MultivariateNormal, PredictiveDistribution};
    ///
    /// let mvn = MultivariateNormal::new(vec![100.0, 50.0], vec![400.0, 120.0, 120.0, 100.0]).unwrap();
    /// let keys = vec![vec!["auto".into()], vec!["property".into()]];
    /// let pd = PredictiveDistribution::from_gaussian(vec!["line".into()], keys, &mvn, 20_000, 1).unwrap();
    /// assert!((pd.mean() - 150.0).abs() < 1.0);
    /// ```
    pub fn from_gaussian(
        dims: Vec<String>,
        components: Vec<ComponentKey>,
        joint: &dyn GaussianJoint,
        n_sims: usize,
        seed: u64,
    ) -> Result<Self> {
        if components.len() != joint.dim() {
            return Err(Error::InvalidParameter {
                name: "components",
                value: components.len() as f64,
                reason: "must have one key per component of the joint",
            });
        }
        let provenance = joint.provenance().param("n_sims", n_sims);
        Self::simulate(dims, components, n_sims, seed, provenance, |rng, row| {
            joint.sample(rng, row)
        })
    }
}

/// Checks lengths and finiteness, then symmetry and positive definiteness,
/// and returns the Cholesky factor.
fn check_mean_cov(mean: &[f64], cov: &[f64], name: &'static str) -> Result<Vec<f64>> {
    check_shape(mean, cov, name)?;
    let d = mean.len();
    if let Some(&m) = mean.iter().find(|m| !m.is_finite()) {
        return Err(Error::InvalidParameter {
            name: "mean",
            value: m,
            reason: "must be finite",
        });
    }
    cholesky(cov, d).ok_or(Error::InvalidParameter {
        name,
        value: f64::NAN,
        reason: "must be positive definite (a zero variance or a perfect correlation is singular)",
    })
}

/// Checks that `cov` is a symmetric `d × d` matrix of finite values for
/// `d = mean.len() >= 1`.
fn check_shape(mean: &[f64], cov: &[f64], name: &'static str) -> Result<()> {
    let d = mean.len();
    if d == 0 {
        return Err(Error::InvalidParameter {
            name: "mean",
            value: 0.0,
            reason: "must have at least one component",
        });
    }
    if cov.len() != d * d {
        return Err(Error::InvalidParameter {
            name,
            value: cov.len() as f64,
            reason: "must have d * d entries for a mean of length d",
        });
    }
    if let Some(&c) = cov.iter().find(|c| !c.is_finite()) {
        return Err(Error::InvalidParameter {
            name,
            value: c,
            reason: "must be finite",
        });
    }
    let scale = (0..d).map(|i| cov[i * d + i].abs()).fold(0.0, f64::max);
    for i in 0..d {
        for j in 0..i {
            let (a, b) = (cov[i * d + j], cov[j * d + i]);
            if (a - b).abs() > 1e-10 * scale {
                return Err(Error::Data(format!(
                    "{name} is not symmetric: [{i}, {j}] = {a} but [{j}, {i}] = {b}"
                )));
            }
        }
    }
    Ok(())
}

fn check_index(j: usize, d: usize) -> Result<()> {
    if j >= d {
        return Err(Error::InvalidParameter {
            name: "j",
            value: j as f64,
            reason: "must be below the number of components",
        });
    }
    Ok(())
}

fn correlation(cov: &[f64], d: usize) -> Vec<f64> {
    let sd: Vec<f64> = (0..d).map(|i| cov[i * d + i].sqrt()).collect();
    let mut r = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            r[i * d + j] = if i == j {
                1.0
            } else {
                cov[i * d + j] / (sd[i] * sd[j])
            };
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Distribution;

    fn three() -> (Vec<f64>, Vec<f64>) {
        // Standard deviations 20, 10 and 30; correlations 0.6, -0.2, 0.3.
        let sd = [20.0, 10.0, 30.0];
        let r = [1.0, 0.6, -0.2, 0.6, 1.0, 0.3, -0.2, 0.3, 1.0];
        let cov = (0..9).map(|k| r[k] * sd[k / 3] * sd[k % 3]).collect();
        (vec![100.0, 50.0, 200.0], cov)
    }

    /// Sample mean and covariance of a draw matrix, `d` columns.
    fn moments(draws: &[f64], d: usize) -> (Vec<f64>, Vec<f64>) {
        let n = draws.len() / d;
        let mut mean = vec![0.0; d];
        for row in draws.chunks(d) {
            for (m, x) in mean.iter_mut().zip(row) {
                *m += x / n as f64;
            }
        }
        let mut cov = vec![0.0; d * d];
        for row in draws.chunks(d) {
            for i in 0..d {
                for j in 0..d {
                    cov[i * d + j] += (row[i] - mean[i]) * (row[j] - mean[j]) / n as f64;
                }
            }
        }
        (mean, cov)
    }

    fn keys(d: usize) -> Vec<ComponentKey> {
        (0..d).map(|j| vec![(j as i64).into()]).collect()
    }

    #[test]
    fn the_normal_draws_have_its_mean_and_covariance() {
        let (mean, cov) = three();
        let mvn = MultivariateNormal::new(mean.clone(), cov.clone()).unwrap();
        let n = 200_000;
        let pd =
            PredictiveDistribution::from_gaussian(vec!["j".into()], keys(3), &mvn, n, 7).unwrap();
        let (m, c) = moments(pd.draw_matrix(), 3);
        for j in 0..3 {
            // Within four standard errors of the mean.
            let se = (cov[j * 3 + j] / n as f64).sqrt();
            assert!((m[j] - mean[j]).abs() < 4.0 * se, "mean {j}: {}", m[j]);
        }
        for k in 0..9 {
            let (i, j) = (k / 3, k % 3);
            // The standard error of a sample covariance of normals.
            let se = ((cov[i * 3 + i] * cov[j * 3 + j] + cov[k] * cov[k]) / n as f64).sqrt();
            assert!(
                (c[k] - cov[k]).abs() < 4.0 * se,
                "cov {k}: {} vs {}",
                c[k],
                cov[k]
            );
        }
        // The total is exact: mean 350, variance the sum of the entries.
        let total = mvn.total();
        assert_eq!(total.mean(), 350.0);
        assert!((total.variance() - cov.iter().sum::<f64>()).abs() < 1e-9);
        assert!((pd.mean() - 350.0).abs() < 4.0 * total.sigma() / (n as f64).sqrt());
        assert_eq!(mvn.marginal(2).unwrap().sigma(), 30.0);
        assert!((mvn.correlation()[1] - 0.6).abs() < 1e-15);
        assert!(mvn.marginal(3).is_err());
    }

    #[test]
    fn draws_reproduce_by_seed_and_record_their_inputs() {
        let (mean, cov) = three();
        let mvn = MultivariateNormal::new(mean, cov).unwrap();
        let a = PredictiveDistribution::from_gaussian(vec!["j".into()], keys(3), &mvn, 500, 11)
            .unwrap();
        let b = PredictiveDistribution::from_gaussian(vec!["j".into()], keys(3), &mvn, 500, 11)
            .unwrap();
        let c = PredictiveDistribution::from_gaussian(vec!["j".into()], keys(3), &mvn, 500, 12)
            .unwrap();
        assert_eq!(a.draw_matrix(), b.draw_matrix());
        assert_ne!(a.draw_matrix(), c.draw_matrix());
        // Simulation 9 replays alone from its stream.
        let mut row = [0.0; 3];
        mvn.sample(&mut StreamRng::new(11, 9), &mut row);
        assert_eq!(&a.draw_matrix()[27..30], &row);
        let p = a.provenance();
        assert_eq!(p.model, "multivariate_normal");
        assert_eq!(p.seed, Some(11));
        assert!(p.parameters.contains(&("dim".into(), "3".into())));
        assert!(p.parameters.contains(&("n_sims".into(), "500".into())));
        let other = MultivariateNormal::new(vec![100.0, 50.0, 201.0], three().1).unwrap();
        assert_ne!(p.input_hash, other.provenance().input_hash);
        assert!(p.input_hash.is_some());
    }

    #[test]
    fn a_bad_covariance_is_refused() {
        let m = vec![0.0, 0.0];
        // Singular: perfect correlation, and a zero variance.
        assert!(MultivariateNormal::new(m.clone(), vec![1.0, 1.0, 1.0, 1.0]).is_err());
        assert!(MultivariateNormal::new(m.clone(), vec![1.0, 0.0, 0.0, 0.0]).is_err());
        // Indefinite.
        assert!(MultivariateNormal::new(m.clone(), vec![1.0, 2.0, 2.0, 1.0]).is_err());
        // Not symmetric, the wrong size, not finite, empty.
        assert!(matches!(
            MultivariateNormal::new(m.clone(), vec![1.0, 0.5, 0.4, 1.0]),
            Err(Error::Data(_))
        ));
        assert!(MultivariateNormal::new(m.clone(), vec![1.0, 0.0, 0.0]).is_err());
        assert!(MultivariateNormal::new(m.clone(), vec![1.0, f64::NAN, f64::NAN, 1.0]).is_err());
        assert!(
            MultivariateNormal::new(vec![f64::INFINITY, 0.0], vec![1.0, 0.0, 0.0, 1.0]).is_err()
        );
        assert!(MultivariateNormal::new(vec![], vec![]).is_err());
        // The wrong number of keys.
        let mvn = MultivariateNormal::new(m, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        assert!(
            PredictiveDistribution::from_gaussian(vec!["j".into()], keys(3), &mvn, 10, 1).is_err()
        );
    }

    #[test]
    fn the_lognormal_matches_its_moments_exactly() {
        let (mean, cov) = three();
        let mvl = MultivariateLognormal::from_moments(mean.clone(), cov.clone()).unwrap();
        for (got, want) in mvl.mean().iter().zip(&mean) {
            assert!((got / want - 1.0).abs() < 1e-12);
        }
        for (got, want) in mvl.cov().iter().zip(&cov) {
            assert!((got - want).abs() < 1e-10 * want.abs().max(1.0));
        }
        assert!((mvl.total_mean() - 350.0).abs() < 1e-10);
        assert!((mvl.total_variance() - cov.iter().sum::<f64>()).abs() < 1e-8);
        // A marginal is the univariate lognormal with the same moments.
        let m1 = mvl.marginal(1).unwrap();
        let direct = Lognormal::from_mean_cv(50.0, 10.0 / 50.0).unwrap();
        assert!((m1.meanlog() - direct.meanlog()).abs() < 1e-14);
        assert!((m1.sdlog() - direct.sdlog()).abs() < 1e-14);
        // And from the log scale it round-trips.
        let log = mvl.log_normal();
        let again =
            MultivariateLognormal::from_log_scale(log.mean().to_vec(), log.cov().to_vec()).unwrap();
        assert_eq!(again, mvl);
    }

    #[test]
    fn the_lognormal_draws_have_its_mean_and_covariance() {
        let (mean, cov) = three();
        let mvl = MultivariateLognormal::from_moments(mean.clone(), cov.clone()).unwrap();
        let n = 400_000;
        let pd =
            PredictiveDistribution::from_gaussian(vec!["j".into()], keys(3), &mvl, n, 3).unwrap();
        assert!(pd.draw_matrix().iter().all(|x| *x > 0.0));
        let (m, c) = moments(pd.draw_matrix(), 3);
        for j in 0..3 {
            let se = (cov[j * 3 + j] / n as f64).sqrt();
            assert!((m[j] - mean[j]).abs() < 4.0 * se, "mean {j}: {}", m[j]);
        }
        for k in 0..9 {
            // Lognormal tails make the covariance's error larger than a
            // normal's; 2% of the scale is many standard errors here.
            let scale = (cov[(k / 3) * 4] * cov[(k % 3) * 4]).sqrt();
            assert!(
                (c[k] - cov[k]).abs() < 0.02 * scale,
                "cov {k}: {} vs {}",
                c[k],
                cov[k]
            );
        }
        assert_eq!(pd.provenance().model, "multivariate_lognormal");
    }

    #[test]
    fn a_covariance_no_lognormal_has_is_refused() {
        // Non-positive means.
        assert!(
            MultivariateLognormal::from_moments(vec![0.0, 1.0], vec![1.0, 0.0, 0.0, 1.0]).is_err()
        );
        // cov_12 = -2 is below -m1 m2 = -1: no lognormal is that negatively
        // correlated.
        let err = MultivariateLognormal::from_moments(vec![1.0, 1.0], vec![4.0, -2.0, -2.0, 4.0])
            .unwrap_err();
        assert!(matches!(err, Error::Data(ref m) if m.contains("cov[0, 1]")));
        // Positive definite on the original scale but not on the log scale:
        // correlation -0.9 at cv 1, so ln(1 - 0.9) = -2.30 against log
        // variances ln 2 = 0.69.
        let cov = vec![1.0, -0.9, -0.9, 1.0];
        assert!(crate::multivariate::cholesky(&cov, 2).is_some());
        let err = MultivariateLognormal::from_moments(vec![1.0, 1.0], cov).unwrap_err();
        assert!(
            matches!(err, Error::Data(ref m) if m.contains("log scale")),
            "{err:?}"
        );
    }
}
