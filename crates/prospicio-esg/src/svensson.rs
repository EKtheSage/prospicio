//! The Nelson-Siegel-Svensson curve (Svensson 1994): a zero-coupon yield
//! curve in six parameters, fitted to observed yields.

use prospicio_core::{Error, Result};
use prospicio_math::linalg::solve;
use prospicio_math::optimize::{NelderMead, nelder_mead};

/// Continuously compounded zero-coupon yields
/// `y(m) = β0 + β1 f1(m/τ1) + β2 f2(m/τ1) + β3 f2(m/τ2)`, with
/// `f1(x) = (1 - e^-x) / x` and `f2(x) = f1(x) - e^-x`, the form of the
/// Federal Reserve's published Treasury curve (Gürkaynak, Sack and Wright
/// 2007). Maturities are in years.
///
/// ```
/// use prospicio_esg::Svensson;
///
/// let curve = Svensson::new([0.04, -0.02, 0.01, 0.015], [1.5, 9.0])?;
/// let maturities = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0];
/// let yields: Vec<f64> = maturities.iter().map(|&m| curve.spot(m)).collect();
/// let fitted = Svensson::fit(&maturities, &yields)?;
/// for &m in &maturities {
///     assert!((fitted.spot(m) - curve.spot(m)).abs() < 1e-6);
/// }
/// # Ok::<(), prospicio_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Svensson {
    /// `β0` to `β3`: the long rate, and the loadings of the short and the
    /// two medium-term humps.
    pub beta: [f64; 4],
    /// `τ1` and `τ2`, the humps' decay times in years.
    pub tau: [f64; 2],
}

fn f1(x: f64) -> f64 {
    if x < 1e-8 {
        1.0 - x / 2.0
    } else {
        -(-x).exp_m1() / x
    }
}

fn f2(x: f64) -> f64 {
    f1(x) - (-x).exp()
}

/// The four loadings of the betas at maturity `m`.
fn loadings(m: f64, tau: [f64; 2]) -> [f64; 4] {
    [1.0, f1(m / tau[0]), f2(m / tau[0]), f2(m / tau[1])]
}

impl Svensson {
    /// A curve from its parameters. Fails unless every parameter is
    /// finite and both `τ` are positive.
    pub fn new(beta: [f64; 4], tau: [f64; 2]) -> Result<Self> {
        if let Some(&b) = beta.iter().find(|b| !b.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "beta",
                value: b,
                reason: "must be finite",
            });
        }
        if let Some(&t) = tau.iter().find(|t| !(t.is_finite() && **t > 0.0)) {
            return Err(Error::InvalidParameter {
                name: "tau",
                value: t,
                reason: "must be finite and positive",
            });
        }
        Ok(Self { beta, tau })
    }

    /// The continuously compounded zero-coupon yield at `m` years (the
    /// short rate `β0 + β1` at 0).
    pub fn spot(&self, m: f64) -> f64 {
        let l = loadings(m.max(0.0), self.tau);
        (0..4).map(|k| self.beta[k] * l[k]).sum()
    }

    /// The instantaneous forward rate at `m` years:
    /// `β0 + β1 e^(-m/τ1) + β2 (m/τ1) e^(-m/τ1) + β3 (m/τ2) e^(-m/τ2)`.
    pub fn forward(&self, m: f64) -> f64 {
        let m = m.max(0.0);
        let (x1, x2) = (m / self.tau[0], m / self.tau[1]);
        self.beta[0]
            + self.beta[1] * (-x1).exp()
            + self.beta[2] * x1 * (-x1).exp()
            + self.beta[3] * x2 * (-x2).exp()
    }

    /// The price of a zero-coupon bond paying 1 in `m` years, `e^(-m y(m))`.
    pub fn discount(&self, m: f64) -> f64 {
        (-m * self.spot(m)).exp()
    }

    /// Least squares fit to continuously compounded `yields` at
    /// `maturities` (years): for given `τ` the betas are linear, so the fit
    /// searches `τ1 < τ2` on a grid of 0.25 to 30 years, then refines both
    /// by Nelder-Mead on their logarithms, the usual robust fit.
    ///
    /// Fails unless the maturities are positive, finite and distinct, the
    /// yields finite and of the same number, and there are at least six.
    pub fn fit(maturities: &[f64], yields: &[f64]) -> Result<Self> {
        if maturities.len() != yields.len() {
            return Err(Error::Data(format!(
                "Svensson fit: {} maturities and {} yields",
                maturities.len(),
                yields.len()
            )));
        }
        if maturities.len() < 6 {
            return Err(Error::Data(format!(
                "Svensson fit: six parameters need at least six yields, got {}",
                maturities.len()
            )));
        }
        for (&m, &y) in maturities.iter().zip(yields) {
            if !(m.is_finite() && m > 0.0) {
                return Err(Error::InvalidParameter {
                    name: "maturity",
                    value: m,
                    reason: "must be finite and positive",
                });
            }
            if !y.is_finite() {
                return Err(Error::InvalidParameter {
                    name: "yield",
                    value: y,
                    reason: "must be finite",
                });
            }
        }
        let mut sorted = maturities.to_vec();
        sorted.sort_by(f64::total_cmp);
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::Data(
                "Svensson fit: a maturity is given twice".into(),
            ));
        }

        let sse = |tau: [f64; 2]| -> Option<(f64, [f64; 4])> {
            let beta = betas(maturities, yields, tau)?;
            let curve = Self { beta, tau };
            let s = maturities
                .iter()
                .zip(yields)
                .map(|(&m, &y)| (curve.spot(m) - y).powi(2))
                .sum();
            Some((s, beta))
        };
        let grid: Vec<f64> = (0..=40)
            .map(|k| 0.25 * (120f64).powf(k as f64 / 40.0))
            .collect();
        let mut best: Option<(f64, [f64; 2])> = None;
        for (i, &t1) in grid.iter().enumerate() {
            for &t2 in &grid[i + 1..] {
                if let Some((s, _)) = sse([t1, t2])
                    && best.is_none_or(|(b, _)| s < b)
                {
                    best = Some((s, [t1, t2]));
                }
            }
        }
        let Some((_, start)) = best else {
            return Err(Error::Data(
                "Svensson fit: the yields determine no betas for any decay times".into(),
            ));
        };
        let objective = |x: &[f64]| {
            let tau = [x[0].exp(), x[1].exp()];
            sse(tau).map_or(f64::INFINITY, |(s, _)| s)
        };
        let options = NelderMead {
            initial_step: 0.1,
            tolerance: 1e-12,
            ..NelderMead::default()
        };
        let m = nelder_mead(objective, &[start[0].ln(), start[1].ln()], &options);
        let tau = [m.x[0].exp(), m.x[1].exp()];
        let tau = if sse(tau).is_some() { tau } else { start };
        let (_, beta) = sse(tau).expect("checked above");
        Self::new(beta, tau)
    }
}

/// The least-squares betas for given decay times, by the normal equations;
/// `None` when they are singular (two equal decay times, say).
fn betas(maturities: &[f64], yields: &[f64], tau: [f64; 2]) -> Option<[f64; 4]> {
    let mut xtx = vec![0.0; 16];
    let mut xty = vec![0.0; 4];
    for (&m, &y) in maturities.iter().zip(yields) {
        let l = loadings(m, tau);
        for i in 0..4 {
            xty[i] += l[i] * y;
            for j in 0..4 {
                xtx[i * 4 + j] += l[i] * l[j];
            }
        }
    }
    let b = solve(xtx, xty, 4)?;
    if b.iter().all(|v| v.is_finite()) {
        Some([b[0], b[1], b[2], b[3]])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_forward_is_the_derivative_of_the_log_price() {
        let c = Svensson::new([0.045, -0.03, -0.02, 0.025], [1.2, 8.0]).unwrap();
        for m in [0.1, 1.0, 4.0, 12.0, 30.0] {
            let h = 1e-5;
            let numeric = -((c.discount(m + h)).ln() - (c.discount(m - h)).ln()) / (2.0 * h);
            assert!((numeric - c.forward(m)).abs() < 1e-8, "{m}");
        }
        assert!((c.spot(0.0) - 0.015).abs() < 1e-15);
        assert!((c.forward(0.0) - 0.015).abs() < 1e-15);
    }

    #[test]
    fn the_inputs_are_checked() {
        assert!(Svensson::new([0.0; 4], [0.0, 1.0]).is_err());
        assert!(Svensson::new([f64::NAN, 0.0, 0.0, 0.0], [1.0, 2.0]).is_err());
        let m = [1.0, 2.0, 3.0, 5.0, 7.0];
        assert!(Svensson::fit(&m, &[0.01; 5]).is_err());
        assert!(Svensson::fit(&[1.0, 1.0, 2.0, 3.0, 5.0, 7.0], &[0.01; 6]).is_err());
        assert!(Svensson::fit(&[1.0, 2.0, 3.0, 5.0, 7.0, 10.0], &[0.01; 5]).is_err());
    }
}
