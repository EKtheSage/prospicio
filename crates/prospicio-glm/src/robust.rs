//! Heteroskedasticity- and cluster-robust (sandwich) covariance of a GLM's
//! coefficients, as statsmodels' `cov_type="HC0"` and `"cluster"`.
//!
//! The model covariance `φ (Xᵀ W X)⁻¹` is right only when the variance
//! function and dispersion are right. The sandwich `B M B` stays
//! consistent when they are not, as long as the mean is: `B` is the
//! inverse of the observed information and `M` the sum of outer products
//! of the per-row (or per-cluster) scores. Policies observed over several
//! years, or claims within one accident, are correlated; clustering on
//! the policy or event keeps their standard errors honest.
//!
//! The dispersion cancels, so the result does not depend on how `φ` was
//! estimated. With the canonical link the observed and expected
//! information agree; with another link (a log-link gamma, say) this uses
//! the observed information, as statsmodels does, where R's `sandwich`
//! uses the expected.

use prospicio_core::{Error, Result};
use prospicio_math::linalg::{cholesky, cholesky_inverse};
use prospicio_models::Design;

use crate::GlmFit;

/// Which sandwich estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Robust<'a> {
    /// White's estimator, from each row's score.
    Hc0,
    /// HC0 scaled by `n / (n - p)` (Stata's `robust`, R `sandwich`'s
    /// `"HC1"`; statsmodels' GLM reports HC0 for `"HC1"`).
    Hc1,
    /// Scores summed within each cluster, one label per row, scaled by
    /// `G / (G - 1) · (n - 1) / (n - p)` for `G` clusters (statsmodels'
    /// and Stata's default correction).
    Cluster(&'a [usize]),
}

impl GlmFit {
    /// Sandwich covariance of the coefficients, row-major `p × p`.
    /// `design` and `y` must be the data the model was fitted on.
    ///
    /// ```
    /// use prospicio_glm::{Glm, Robust};
    /// use prospicio_models::{Design, Family, Link, Model};
    ///
    /// let x = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    /// let design = Design::new(vec!["(Intercept)".into(), "x".into()], vec![vec![1.0; 6], x])
    ///     .unwrap();
    /// let y = [1.0, 2.0, 6.0, 1.0, 4.0, 2.0];
    /// let fit = Glm::new(Family::Poisson, Link::Log).fit(&design, &y).unwrap();
    /// let v = fit.robust_covariance(&design, &y, Robust::Hc0).unwrap();
    /// // Intercept only group (x = 0): Σ(y - ȳ)² / (n ȳ)², ȳ = 3.
    /// assert!((v[0] - 14.0 / 81.0).abs() < 1e-12);
    /// ```
    pub fn robust_covariance(&self, design: &Design, y: &[f64], kind: Robust) -> Result<Vec<f64>> {
        self.check_design(design)?;
        let n = self.n_obs;
        if design.n_rows() != n || y.len() != n {
            return Err(Error::Data(format!(
                "the model was fitted on {n} rows; got {} design rows and {} responses",
                design.n_rows(),
                y.len()
            )));
        }
        let (family, link) = (self.spec.family, self.spec.link);
        let p = self.coefficients.len();
        let w = design.weights();
        let mut info = vec![0.0; p * p];
        // Per-row scores (without the dispersion), row-major n × p.
        let mut scores = vec![0.0; n * p];
        for i in 0..n {
            let mu = self.fitted[i];
            let eta = link.link(mu);
            let (d1, d2) = (link.mu_eta(eta), link.mu_eta2(eta));
            let (v, dv) = (family.variance(mu), family.variance_deriv(mu));
            let r = y[i] - mu;
            let wi = w[i] * (d1 * d1 / v - r * (d2 / v - d1 * d1 * dv / (v * v)));
            let si = w[i] * r * d1 / v;
            for a in 0..p {
                let xa = design.column(a)[i];
                scores[i * p + a] = xa * si;
                for b in 0..=a {
                    info[a * p + b] += xa * wi * design.column(b)[i];
                }
            }
        }
        for a in 0..p {
            for b in a + 1..p {
                info[a * p + b] = info[b * p + a];
            }
        }
        let l = cholesky(&info, p).ok_or_else(|| {
            Error::Data("the observed information is not positive definite".into())
        })?;
        let bread = cholesky_inverse(&l, p);

        let (meat, scale) = match kind {
            Robust::Hc0 | Robust::Hc1 => {
                let scale = if kind == Robust::Hc1 {
                    n as f64 / (n - p) as f64
                } else {
                    1.0
                };
                (outer_sum(scores.chunks_exact(p), p), scale)
            }
            Robust::Cluster(groups) => {
                if groups.len() != n {
                    return Err(Error::Data(format!(
                        "{} cluster labels for {n} rows",
                        groups.len()
                    )));
                }
                let mut labels: Vec<usize> = groups.to_vec();
                labels.sort_unstable();
                labels.dedup();
                let g = labels.len();
                if g < 2 {
                    return Err(Error::Data("clustering needs at least 2 clusters".into()));
                }
                let mut sums = vec![0.0; g * p];
                for (i, label) in groups.iter().enumerate() {
                    let k = labels.binary_search(label).expect("label is in labels");
                    for a in 0..p {
                        sums[k * p + a] += scores[i * p + a];
                    }
                }
                let (n, p, g) = (n as f64, p as f64, g as f64);
                let scale = g / (g - 1.0) * (n - 1.0) / (n - p);
                (outer_sum(sums.chunks_exact(p as usize), p as usize), scale)
            }
        };
        let bm = matmul(&bread, &meat, p);
        Ok(matmul(&bm, &bread, p)
            .into_iter()
            .map(|c| c * scale)
            .collect())
    }

    /// Square roots of the diagonal of
    /// [`robust_covariance`](Self::robust_covariance).
    pub fn robust_std_errors(&self, design: &Design, y: &[f64], kind: Robust) -> Result<Vec<f64>> {
        let p = self.coefficients.len();
        let v = self.robust_covariance(design, y, kind)?;
        Ok((0..p).map(|j| v[j * p + j].sqrt()).collect())
    }
}

/// `Σ s sᵀ` over the rows `s`, row-major `p × p`.
fn outer_sum<'a>(rows: impl Iterator<Item = &'a [f64]>, p: usize) -> Vec<f64> {
    let mut m = vec![0.0; p * p];
    for s in rows {
        for a in 0..p {
            for b in 0..p {
                m[a * p + b] += s[a] * s[b];
            }
        }
    }
    m
}

/// Row-major `p × p` product.
fn matmul(a: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut c = vec![0.0; p * p];
    for i in 0..p {
        for k in 0..p {
            let aik = a[i * p + k];
            for j in 0..p {
                c[i * p + j] += aik * b[k * p + j];
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Glm;
    use prospicio_models::{Family, Link, Model};

    fn data() -> (Design, Vec<f64>) {
        let x: Vec<f64> = (0..40).map(|i| (i % 8) as f64 / 4.0).collect();
        let y: Vec<f64> = (0..40)
            .map(|i| 1.0 + x[i] * 2.0 + ((i * 7919) % 13) as f64 / 3.0)
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; 40], x],
        )
        .unwrap();
        (d, y)
    }

    #[test]
    fn hc1_and_singleton_clusters_rescale_hc0() {
        let (d, y) = data();
        let fit = Glm::new(Family::Gamma, Link::Log).fit(&d, &y).unwrap();
        let hc0 = fit.robust_covariance(&d, &y, Robust::Hc0).unwrap();
        let hc1 = fit.robust_covariance(&d, &y, Robust::Hc1).unwrap();
        let ids: Vec<usize> = (0..40).collect();
        let cl = fit
            .robust_covariance(&d, &y, Robust::Cluster(&ids))
            .unwrap();
        for k in 0..4 {
            assert!((hc1[k] - hc0[k] * 40.0 / 38.0).abs() < 1e-14 * hc0[k].abs().max(1e-300));
            // One row per cluster: G / (G - 1) (n - 1) / (n - p) = n / (n - p).
            assert!((cl[k] - hc1[k]).abs() < 1e-12 * hc1[k].abs());
        }
    }

    #[test]
    fn gaussian_identity_is_whites_estimator() {
        // OLS: (XᵀX)⁻¹ Σ e² x xᵀ (XᵀX)⁻¹, whatever the dispersion.
        let (d, y) = data();
        let fit = Glm::new(Family::Gaussian, Link::Identity)
            .fit(&d, &y)
            .unwrap();
        let v = fit.robust_covariance(&d, &y, Robust::Hc0).unwrap();
        let x = d.column(1);
        let (n, sx, sxx) = (
            40.0,
            x.iter().sum::<f64>(),
            x.iter().map(|v| v * v).sum::<f64>(),
        );
        let det = n * sxx - sx * sx;
        let inv = [sxx / det, -sx / det, -sx / det, n / det];
        let mut meat = [0.0; 4];
        for i in 0..40 {
            let e2 = (y[i] - fit.fitted()[i]).powi(2);
            let r = [1.0, x[i]];
            for a in 0..2 {
                for b in 0..2 {
                    meat[a * 2 + b] += e2 * r[a] * r[b];
                }
            }
        }
        let want = matmul(&matmul(&inv, &meat, 2), &inv, 2);
        for k in 0..4 {
            assert!(
                (v[k] - want[k]).abs() < 1e-10 * want[k].abs().max(1e-12),
                "{k}"
            );
        }
    }

    #[test]
    fn rejects_mismatched_data() {
        let (d, y) = data();
        let fit = Glm::new(Family::Poisson, Link::Log).fit(&d, &y).unwrap();
        assert!(fit.robust_covariance(&d, &y[1..], Robust::Hc0).is_err());
        assert!(
            fit.robust_covariance(&d, &y, Robust::Cluster(&[0; 40]))
                .is_err()
        );
        assert!(
            fit.robust_covariance(&d, &y, Robust::Cluster(&[0; 3]))
                .is_err()
        );
    }
}
