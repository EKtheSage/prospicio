//! Tests of a GLM against a larger one that contains it, as R's
//! `anova(small, big, test = "F")` and `test = "Chisq"`.
//!
//! Both rest on the drop in deviance `D_small − D_big` over the `q` extra
//! coefficients, scaled by the larger model's dispersion `φ`. The F test
//! (CAS Monograph 5, section 6.2.1) suits a family whose dispersion is
//! estimated; the likelihood ratio test suits one whose dispersion is known
//! (Poisson, binomial). Neither checks that the models are nested: the
//! smaller model's columns must lie in the span of the larger's, on the
//! same rows, family and weights.

use prospicio_core::{Error, Result};
use prospicio_math::special::{beta_inc, gamma_inc};

use crate::GlmFit;

/// The result of [`f_test`] or [`likelihood_ratio_test`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NestedTest {
    /// The test statistic: `F` or the scaled deviance drop.
    pub statistic: f64,
    /// Numerator degrees of freedom, `q = p_big − p_small`.
    pub df: f64,
    /// Denominator degrees of freedom of the F test (the larger model's
    /// residual degrees of freedom); infinite for the likelihood ratio
    /// test.
    pub df_resid: f64,
    /// The drop in deviance, `D_small − D_big`.
    pub deviance_drop: f64,
    /// Probability of a statistic at least as large under the smaller
    /// model.
    pub p_value: f64,
}

/// The F test of `small` within `big`: `F = (D_small − D_big) / (q φ_big)`
/// on `F(q, n − p_big)`, as R's `anova(small, big, test = "F")`.
///
/// Fails if the models differ in rows or `big` has no more coefficients
/// than `small`.
pub fn f_test(small: &GlmFit, big: &GlmFit) -> Result<NestedTest> {
    let (q, drop) = check(small, big)?;
    let d2 = big.df_resid();
    let f = drop / (q * big.dispersion());
    // P(F > f) = I_{d2 / (d2 + q f)}(d2 / 2, q / 2).
    let p_value = if f <= 0.0 {
        1.0
    } else {
        beta_inc(0.5 * d2, 0.5 * q, d2 / (d2 + q * f))
    };
    Ok(NestedTest {
        statistic: f,
        df: q,
        df_resid: d2,
        deviance_drop: drop,
        p_value,
    })
}

/// The likelihood ratio test of `small` within `big`: the scaled deviance
/// drop `(D_small − D_big) / φ_big` on `χ²(q)`, as R's
/// `anova(small, big, test = "Chisq")`.
///
/// Fails as [`f_test`].
pub fn likelihood_ratio_test(small: &GlmFit, big: &GlmFit) -> Result<NestedTest> {
    let (q, drop) = check(small, big)?;
    let chi2 = drop / big.dispersion();
    let p_value = if chi2 <= 0.0 {
        1.0
    } else {
        gamma_inc(0.5 * q, 0.5 * chi2).1
    };
    Ok(NestedTest {
        statistic: chi2,
        df: q,
        df_resid: f64::INFINITY,
        deviance_drop: drop,
        p_value,
    })
}

fn check(small: &GlmFit, big: &GlmFit) -> Result<(f64, f64)> {
    if small.n_obs() != big.n_obs() {
        return Err(Error::InvalidParameter {
            name: "small",
            value: small.n_obs() as f64,
            reason: "must be fitted on the same rows as big",
        });
    }
    let (ps, pb) = (small.coefficients().len(), big.coefficients().len());
    if ps >= pb {
        return Err(Error::InvalidParameter {
            name: "big",
            value: pb as f64,
            reason: "must have more coefficients than small",
        });
    }
    Ok(((pb - ps) as f64, small.deviance() - big.deviance()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Glm;
    use prospicio_models::{Design, Family, Link, Model};

    fn fits() -> (GlmFit, GlmFit) {
        let n = 40;
        let x: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| ((i * 7) % 3) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.5 * x[i] + 0.1 * z[i] + 0.3 * ((i * 13) % 7) as f64)
            .collect();
        let small = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; n], x.clone()],
        )
        .unwrap();
        let big = Design::new(
            vec!["(Intercept)".into(), "x".into(), "z".into()],
            vec![vec![1.0; n], x, z],
        )
        .unwrap();
        let glm = Glm::new(Family::Gaussian, Link::Identity);
        (glm.fit(&small, &y).unwrap(), glm.fit(&big, &y).unwrap())
    }

    #[test]
    fn one_extra_coefficient_is_the_squared_t_test() {
        // With q = 1 the F statistic is the square of the coefficient's t.
        let (small, big) = fits();
        let t = big.coefficients()[2] / big.std_errors()[2];
        let test = f_test(&small, &big).unwrap();
        assert!((test.statistic - t * t).abs() < 1e-9 * t * t);
        assert!((test.p_value - big.p_values()[2]).abs() < 1e-10);
        assert_eq!(test.df, 1.0);
        assert_eq!(test.df_resid, 37.0);
    }

    #[test]
    fn likelihood_ratio_and_errors() {
        let (small, big) = fits();
        let lr = likelihood_ratio_test(&small, &big).unwrap();
        assert!((lr.statistic - lr.deviance_drop / big.dispersion()).abs() < 1e-12);
        assert!(lr.p_value > 0.0 && lr.p_value < 1.0);
        assert!(f_test(&big, &small).is_err());
        assert!(big.bic() > big.aic());
    }
}
