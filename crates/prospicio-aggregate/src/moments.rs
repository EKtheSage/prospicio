//! Compound moments to third order and the approximations fitted to them
//! (`docs/design/aggregate.md`): the mean, cv and skewness of
//! `S = X_1 + … + X_N` from the count's and the severity's first three
//! moments, and the normal, gamma, lognormal, shifted gamma and shifted
//! lognormal with the same moments, as the Python package `aggregate`'s
//! `approximate`.

use prospicio_core::{Error, Result};
use prospicio_math::special::{norm_cdf, norm_quantile};
use prospicio_prob::{Counting, Distribution, Gamma, Lognormal, Severity};

/// The first three cumulants of a compound sum: its mean, variance and
/// third central moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompoundMoments {
    pub mean: f64,
    pub variance: f64,
    pub third_cumulant: f64,
}

impl CompoundMoments {
    /// The coefficient of variation `σ / μ`.
    pub fn cv(&self) -> f64 {
        self.variance.sqrt() / self.mean
    }

    /// The skewness `κ₃ / σ³`.
    pub fn skew(&self) -> f64 {
        self.third_cumulant / self.variance.powf(1.5)
    }

    /// The moment-matched approximation of the given kind.
    ///
    /// The gamma and the lognormal match the mean and cv and need a
    /// positive mean; the shifted gamma and the shifted lognormal also
    /// match the skewness, which must be positive. `aggregate` falls back
    /// to a normal for a symmetric book and reflects a left-skewed one;
    /// here both are errors.
    ///
    /// ```
    /// use prospicio_aggregate::moments::{Approximation, ApproximationKind, compound_moments};
    /// use prospicio_prob::{Gamma, Poisson};
    ///
    /// let m = compound_moments(&Poisson::new(10.0).unwrap(), &Gamma::from_mean_cv(50.0, 0.7).unwrap())
    ///     .unwrap();
    /// assert!((m.mean - 500.0).abs() < 1e-9);
    /// let Approximation::ShiftedGamma { shape, scale, shift } =
    ///     m.approximate(ApproximationKind::ShiftedGamma).unwrap()
    /// else {
    ///     unreachable!()
    /// };
    /// // The shifted gamma reproduces the skewness: 2 / sqrt(shape).
    /// assert!((2.0 / shape.sqrt() - m.skew()).abs() < 1e-12);
    /// assert!((shift + shape * scale - m.mean).abs() < 1e-9);
    /// ```
    pub fn approximate(&self, kind: ApproximationKind) -> Result<Approximation> {
        let (m, cv, skew) = (self.mean, self.cv(), self.skew());
        let sd = self.variance.sqrt();
        if !(m.is_finite() && sd.is_finite() && sd > 0.0) {
            return Err(Error::Data(
                "the moments must be finite with a positive variance".into(),
            ));
        }
        let positive_mean = || {
            if m > 0.0 {
                Ok(())
            } else {
                Err(Error::InvalidParameter {
                    name: "mean",
                    value: m,
                    reason: "must be positive for a gamma or lognormal fit",
                })
            }
        };
        let positive_skew = || {
            if skew.is_finite() && skew > 0.0 {
                Ok(())
            } else {
                Err(Error::InvalidParameter {
                    name: "skew",
                    value: skew,
                    reason: "must be finite and positive for a shifted fit",
                })
            }
        };
        Ok(match kind {
            ApproximationKind::Normal => Approximation::Normal { mean: m, sd },
            ApproximationKind::Gamma => {
                positive_mean()?;
                let shape = 1.0 / (cv * cv);
                Approximation::Gamma {
                    shape,
                    scale: m / shape,
                }
            }
            ApproximationKind::Lognormal => {
                positive_mean()?;
                let s2 = cv.mul_add(cv, 1.0).ln();
                Approximation::Lognormal {
                    meanlog: m.ln() - 0.5 * s2,
                    sdlog: s2.sqrt(),
                }
            }
            ApproximationKind::ShiftedGamma => {
                positive_skew()?;
                let shape = 4.0 / (skew * skew);
                let scale = sd * skew / 2.0;
                Approximation::ShiftedGamma {
                    shape,
                    scale,
                    shift: m - shape * scale,
                }
            }
            ApproximationKind::ShiftedLognormal => {
                positive_skew()?;
                // η = sqrt(exp(σ²) − 1) solves η³ + 3η = skew.
                let r = (skew * skew + 4.0).sqrt() / 2.0 + skew / 2.0;
                let eta = r.cbrt() - 1.0 / r.cbrt();
                let sdlog = eta.mul_add(eta, 1.0).ln().sqrt();
                let shift = m - sd / eta;
                Approximation::ShiftedLognormal {
                    meanlog: (m - shift).ln() - 0.5 * sdlog * sdlog,
                    sdlog,
                    shift,
                }
            }
        })
    }
}

/// The compound moments of `frequency` claims of `severity`, by cumulants
/// (no cancellation at large counts):
///
/// - `E[S] = E[N] E[X]`,
/// - `Var[S] = E[N] Var[X] + Var[N] E[X]²`,
/// - `κ₃[S] = E[N] κ₃[X] + 3 Var[N] E[X] Var[X] + κ₃[N] E[X]³`.
///
/// An error when the severity has no closed-form third moment, or when
/// its moments are infinite.
pub fn compound_moments<N, X>(frequency: &N, severity: &X) -> Result<CompoundMoments>
where
    N: Counting + ?Sized,
    X: Severity + ?Sized,
{
    let Some(m3) = severity.third_moment() else {
        return Err(Error::Data(
            "the severity has no closed-form third moment".into(),
        ));
    };
    let (x1, vx) = (severity.mean(), severity.variance());
    if !(x1.is_finite() && vx.is_finite() && m3.is_finite()) {
        return Err(Error::Data(
            "the severity's first three moments must be finite".into(),
        ));
    }
    // κ₃[X] = E[(X − μ)³] = m₃ − 3μ Var[X] − μ³.
    let k3x = m3 - 3.0 * x1 * vx - x1 * x1 * x1;
    let (n1, vn, k3n) = (
        frequency.mean(),
        frequency.variance(),
        frequency.third_cumulant(),
    );
    Ok(CompoundMoments {
        mean: n1 * x1,
        variance: n1 * vx + vn * x1 * x1,
        third_cumulant: n1 * k3x + 3.0 * vn * x1 * vx + k3n * x1 * x1 * x1,
    })
}

/// Which approximation to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApproximationKind {
    Normal,
    Gamma,
    Lognormal,
    ShiftedGamma,
    ShiftedLognormal,
}

/// A distribution fitted to compound moments. Shifted families are their
/// base family plus `shift`, which may be negative, so they are not
/// severities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Approximation {
    Normal {
        mean: f64,
        sd: f64,
    },
    Gamma {
        shape: f64,
        scale: f64,
    },
    Lognormal {
        meanlog: f64,
        sdlog: f64,
    },
    ShiftedGamma {
        shape: f64,
        scale: f64,
        shift: f64,
    },
    ShiftedLognormal {
        meanlog: f64,
        sdlog: f64,
        shift: f64,
    },
}

impl Approximation {
    fn base(&self) -> Result<(Option<Box<dyn Distribution>>, f64)> {
        Ok(match *self {
            Self::Normal { .. } => (None, 0.0),
            Self::Gamma { shape, scale } => (Some(Box::new(Gamma::new(shape, scale)?)), 0.0),
            Self::Lognormal { meanlog, sdlog } => {
                (Some(Box::new(Lognormal::new(meanlog, sdlog)?)), 0.0)
            }
            Self::ShiftedGamma {
                shape,
                scale,
                shift,
            } => (Some(Box::new(Gamma::new(shape, scale)?)), shift),
            Self::ShiftedLognormal {
                meanlog,
                sdlog,
                shift,
            } => (Some(Box::new(Lognormal::new(meanlog, sdlog)?)), shift),
        })
    }

    /// `P(S ≤ x)`.
    pub fn cdf(&self, x: f64) -> Result<f64> {
        if let Self::Normal { mean, sd } = *self {
            return Ok(norm_cdf((x - mean) / sd));
        }
        let (d, shift) = self.base()?;
        Ok(d.expect("non-normal").cdf(x - shift))
    }

    /// The `p` quantile.
    pub fn quantile(&self, p: f64) -> Result<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(Error::InvalidProbability(p));
        }
        if let Self::Normal { mean, sd } = *self {
            return Ok(mean + sd * norm_quantile(p));
        }
        let (d, shift) = self.base()?;
        Ok(shift + d.expect("non-normal").quantile(p)?)
    }

    /// The mean, which is the compound mean it was fitted to.
    pub fn mean(&self) -> f64 {
        match *self {
            Self::Normal { mean, .. } => mean,
            Self::Gamma { shape, scale } => shape * scale,
            Self::Lognormal { meanlog, sdlog } => (meanlog + 0.5 * sdlog * sdlog).exp(),
            Self::ShiftedGamma {
                shape,
                scale,
                shift,
            } => shift + shape * scale,
            Self::ShiftedLognormal {
                meanlog,
                sdlog,
                shift,
            } => shift + (meanlog + 0.5 * sdlog * sdlog).exp(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::{Lognormal, NegativeBinomial, Poisson};

    fn book() -> CompoundMoments {
        compound_moments(
            &Poisson::new(10.0).unwrap(),
            &Gamma::from_mean_cv(50.0, 0.7).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn every_fit_reproduces_the_moments_it_matches() {
        let m = book();
        for kind in [
            ApproximationKind::Normal,
            ApproximationKind::Gamma,
            ApproximationKind::Lognormal,
            ApproximationKind::ShiftedGamma,
            ApproximationKind::ShiftedLognormal,
        ] {
            let a = m.approximate(kind).unwrap();
            assert!((a.mean() / m.mean - 1.0).abs() < 1e-12, "{kind:?}");
            let q = a.quantile(0.9).unwrap();
            assert!((a.cdf(q).unwrap() - 0.9).abs() < 1e-10, "{kind:?}");
        }
        // The shifted lognormal matches the skewness: (e^σ² + 2) sqrt(e^σ² − 1).
        if let Approximation::ShiftedLognormal { sdlog, .. } =
            m.approximate(ApproximationKind::ShiftedLognormal).unwrap()
        {
            let w = (sdlog * sdlog).exp();
            assert!(((w + 2.0) * (w - 1.0).sqrt() - m.skew()).abs() < 1e-12);
        }
    }

    #[test]
    fn poisson_compound_cumulants_are_lambda_times_raw_moments() {
        let x = Lognormal::from_mean_cv(10.0, 0.5).unwrap();
        let n = Poisson::new(1000.0).unwrap();
        let m = compound_moments(&n, &x).unwrap();
        let m2 = x.variance() + 100.0;
        let m3 = x.third_moment().unwrap();
        assert!((m.variance / (1000.0 * m2) - 1.0).abs() < 1e-14);
        assert!((m.third_cumulant / (1000.0 * m3) - 1.0).abs() < 1e-13);
        // A negative binomial adds the count's own cumulants.
        let nb = NegativeBinomial::new(2.0, 3.0).unwrap();
        let c = compound_moments(&nb, &x).unwrap();
        assert!((c.mean - 60.0).abs() < 1e-12);
    }

    /// A count chosen by mean and dispersion gives the same exact third
    /// cumulant as its family (it was 1.2e-7 off at 1,000 claims while
    /// `PanjerClass` left the trait's default to convert the raw moment).
    #[test]
    fn panjer_class_compound_skewness_is_exact() {
        let x = Lognormal::from_mean_cv(10.0, 0.5).unwrap();
        let n = prospicio_prob::PanjerClass::from_mean_dispersion(1000.0, 1.0).unwrap();
        let m = compound_moments(&n, &x).unwrap();
        // 1000 E[X³] = 1000 × 10³ × (1 + 0.5²)³ = 1,953,125.
        assert!((m.third_cumulant / 1_953_125.0 - 1.0).abs() < 1e-13);
        let exact = compound_moments(&Poisson::new(1000.0).unwrap(), &x).unwrap();
        assert_eq!(m, exact);
    }

    #[test]
    fn refuses_what_it_cannot_fit() {
        use prospicio_prob::Pareto;
        let n = Poisson::new(2.0).unwrap();
        assert!(compound_moments(&n, &Pareto::new(100.0, 2.5).unwrap()).is_err());
        let symmetric = CompoundMoments {
            mean: 10.0,
            variance: 4.0,
            third_cumulant: 0.0,
        };
        assert!(
            symmetric
                .approximate(ApproximationKind::ShiftedGamma)
                .is_err()
        );
        assert!(symmetric.approximate(ApproximationKind::Normal).is_ok());
        assert!(
            book()
                .approximate(ApproximationKind::Gamma)
                .unwrap()
                .quantile(1.5)
                .is_err()
        );
    }
}
