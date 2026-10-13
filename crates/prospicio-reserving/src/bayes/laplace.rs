//! The whitened scales that the Bayesian reserving models sample on
//! (`docs/design/reserving-bayes.md`, decision 3, "Preconditioning", and
//! decision 7, "Sampling scale").
//! The mass matrix of `prospicio-bayes` is diagonal, and the corner
//! constraints make the log-linear parameters strongly correlated, so each
//! model samples `z` with `theta = centre + L z` for a dense `L`:
//!
//! * [`BayesOdp`](super::BayesOdp), whose posterior is strictly concave and
//!   close to normal: the posterior mode and `L L' = H^-1` for the negative
//!   Hessian `H` there (a Laplace approximation, [`Whitened::new`]);
//! * [`Meyers`](super::Meyers), whose joint mode sits deep in a funnel (the
//!   late `sigma_d` near zero) far from the posterior's bulk: a pilot run on
//!   a scale whitened at a start point ([`Whitened::regularised`]), then
//!   the pilot draws' mean and regularised covariance
//!   ([`Whitened::from_draws`]), as Stan's dense metric adaptation does.

use prospicio_bayes::nuts::{LogDensity, PosteriorDraws};
use prospicio_math::linalg::{cholesky, cholesky_solve, lower_solve};

use crate::error::{Error, Result};

/// A log posterior on an unconstrained `theta`, with its gradient.
pub(crate) trait Posterior: Sync {
    /// Number of parameters.
    fn dim(&self) -> usize;

    /// The log posterior at `theta` (up to a constant), its gradient
    /// written into `grad`. A value that is not finite marks a position
    /// outside the support.
    fn log_posterior(&self, theta: &[f64], grad: &mut [f64]) -> f64;

    /// The negative Hessian of the log posterior at `theta`, dense and
    /// row-major. By default, central differences of the gradient with
    /// step `1e-5 (1 + |theta_j|)`, symmetrised: about `2 p` gradient
    /// evaluations, cheap for the few dozen parameters of these models.
    fn negative_hessian(&self, theta: &[f64]) -> Vec<f64> {
        let p = self.dim();
        let mut h = vec![0.0; p * p];
        let (mut up, mut down) = (vec![0.0; p], vec![0.0; p]);
        let mut x = theta.to_vec();
        for j in 0..p {
            let step = 1e-5 * (1.0 + theta[j].abs());
            x[j] = theta[j] + step;
            self.log_posterior(&x, &mut up);
            x[j] = theta[j] - step;
            self.log_posterior(&x, &mut down);
            x[j] = theta[j];
            for i in 0..p {
                h[i * p + j] = -(up[i] - down[i]) / (2.0 * step);
            }
        }
        for i in 0..p {
            for j in 0..i {
                let v = 0.5 * (h[i * p + j] + h[j * p + i]);
                h[i * p + j] = v;
                h[j * p + i] = v;
            }
        }
        h
    }
}

/// Largest number of Newton iterations.
const MAX_ITER: usize = 200;

/// The posterior mode by Newton's method from `start`: each step solves
/// `H delta = grad` by Cholesky, with a ridge added when `H` is not
/// positive definite (a non-concave region, see [`positive_definite`]),
/// and halves the step until
/// the log posterior rises. It stops when `max |delta| < 1e-12`, or below
/// `1e-8` when a full step no longer raises it (the rounding floor). For a
/// strictly concave posterior the ridge is never needed and the mode is
/// unique.
pub(crate) fn mode<P: Posterior + ?Sized>(model: &P, start: Vec<f64>) -> Result<Vec<f64>> {
    let p = model.dim();
    let mut theta = start;
    let mut grad = vec![0.0; p];
    let mut f = model.log_posterior(&theta, &mut grad);
    if !f.is_finite() {
        return Err(Error::Bayes(
            "the log posterior is not finite at the start".into(),
        ));
    }
    let mut trial_grad = vec![0.0; p];
    for _ in 0..MAX_ITER {
        let l = positive_definite(model.negative_hessian(&theta), p)?;
        let delta = cholesky_solve(&l, &grad);
        let size = delta.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if size < 1e-12 {
            return Ok(theta);
        }
        let mut step = 1.0;
        loop {
            let trial: Vec<f64> = theta
                .iter()
                .zip(&delta)
                .map(|(t, d)| t + step * d)
                .collect();
            let f_trial = model.log_posterior(&trial, &mut trial_grad);
            if f_trial.is_finite() && f_trial >= f {
                theta = trial;
                f = f_trial;
                std::mem::swap(&mut grad, &mut trial_grad);
                break;
            }
            // At the mode a full step changes f by less than rounding:
            // stop there.
            if size < 1e-8 {
                return Ok(theta);
            }
            step *= 0.5;
            if step < 1e-10 {
                return Err(Error::Bayes(
                    "Newton's method could not raise the log posterior".into(),
                ));
            }
        }
    }
    Err(Error::Bayes(format!(
        "Newton's method did not converge to the posterior mode in {MAX_ITER} iterations"
    )))
}

/// The Cholesky factor of `h` when it is positive definite. Otherwise
/// that of `h + 2 lambda I`, for the first `lambda` of the ladder
/// `1e-4 c, 1e-3 c, ...` (`c` the largest absolute diagonal, at least 1)
/// with `h + lambda I` positive definite: doubling it keeps the smallest
/// eigenvalue above `lambda`, so the step stays bounded where `h` is
/// nearly singular, and the step halving does the rest.
fn positive_definite(mut h: Vec<f64>, p: usize) -> Result<Vec<f64>> {
    if let Some(l) = cholesky(&h, p) {
        return Ok(l);
    }
    let scale = (0..p).fold(1.0_f64, |m, j| m.max(h[j * p + j].abs()));
    let mut lambda = 1e-4 * scale;
    let mut added = 0.0;
    for _ in 0..30 {
        for j in 0..p {
            h[j * p + j] += lambda - added;
        }
        added = lambda;
        if cholesky(&h, p).is_some() {
            for j in 0..p {
                h[j * p + j] += lambda;
            }
            return cholesky(&h, p).ok_or_else(|| {
                Error::Bayes("the negative Hessian is not positive definite".into())
            });
        }
        lambda *= 10.0;
    }
    Err(Error::Bayes(
        "the negative Hessian is not positive definite".into(),
    ))
}

/// The log posterior on a whitened scale `z`, `theta = centre + L z` for a
/// dense `L`, so that the posterior in `z` is close to a standard normal.
/// The map is linear, so its Jacobian is a constant and left out; the
/// gradient is `L' grad_theta`.
pub(crate) struct Whitened<'a, P: ?Sized> {
    model: &'a P,
    centre: Vec<f64>,
    /// `L`, `p x p` row-major.
    l: Vec<f64>,
}

impl<'a, P: Posterior + ?Sized> Whitened<'a, P> {
    /// The Laplace scale at the posterior `mode`: `L L' = H^-1` for the
    /// negative Hessian `H` there, which must be positive definite.
    pub(crate) fn new(model: &'a P, mode: &[f64]) -> Result<Self> {
        let r = cholesky(&model.negative_hessian(mode), model.dim()).ok_or_else(|| {
            Error::Bayes("the negative Hessian at the mode is not positive definite".into())
        })?;
        Ok(Self::from_hessian_factor(model, mode, &r))
    }

    /// The scale at any `point`, with the negative Hessian there made
    /// positive definite by the ridge of [`mode`]'s steps: a rough scale
    /// for a pilot run.
    pub(crate) fn regularised(model: &'a P, point: &[f64]) -> Result<Self> {
        let r = positive_definite(model.negative_hessian(point), model.dim())?;
        Ok(Self::from_hessian_factor(model, point, &r))
    }

    /// The scale of pilot `draws` of `theta`: their mean, and `L` the
    /// Cholesky factor of their covariance regularised as Stan regularises
    /// its dense metric, `n / (n + 5) S + 1e-3 (5 / (n + 5)) I` for `n`
    /// draws.
    pub(crate) fn from_draws(model: &'a P, draws: &PosteriorDraws) -> Result<Self> {
        let p = model.dim();
        let n = draws.n_draws();
        let mean = draws.mean();
        let mut cov = vec![0.0; p * p];
        for theta in draws.values().chunks_exact(p) {
            for i in 0..p {
                let a = theta[i] - mean[i];
                for j in 0..=i {
                    cov[i * p + j] += a * (theta[j] - mean[j]);
                }
            }
        }
        let (n, w) = (n as f64, n as f64 / (n as f64 + 5.0));
        for i in 0..p {
            for j in 0..=i {
                let v = w * cov[i * p + j] / (n - 1.0);
                cov[i * p + j] = v;
                cov[j * p + i] = v;
            }
            cov[i * p + i] += 1e-3 * (1.0 - w);
        }
        let l = cholesky(&cov, p).ok_or_else(|| {
            Error::Bayes("the pilot draws' covariance is not positive definite".into())
        })?;
        Ok(Self {
            model,
            centre: mean,
            l,
        })
    }

    /// `L = (R')^-1` for `H = R R'`, so `L L' = H^-1`.
    fn from_hessian_factor(model: &'a P, centre: &[f64], r: &[f64]) -> Self {
        let p = model.dim();
        let mut l = vec![0.0; p * p];
        let mut e = vec![0.0; p];
        let mut column = vec![0.0; p];
        for j in 0..p {
            e.fill(0.0);
            e[j] = 1.0;
            lower_solve(r, &e, &mut column);
            // `column` is column j of R^-1, so row j of L.
            l[j * p..(j + 1) * p].copy_from_slice(&column);
        }
        Self {
            model,
            centre: centre.to_vec(),
            l,
        }
    }

    /// `centre + L z`.
    pub(crate) fn theta(&self, z: &[f64]) -> Vec<f64> {
        let p = z.len();
        // Index loops, not iterator chains: this runs at every leapfrog
        // step, and a debug build (CI's tests) runs them several times
        // faster.
        let mut theta = self.centre.clone();
        for (i, t) in theta.iter_mut().enumerate() {
            let row = &self.l[i * p..(i + 1) * p];
            let mut acc = 0.0;
            for j in 0..p {
                acc += row[j] * z[j];
            }
            *t += acc;
        }
        theta
    }
}

impl<P: Posterior + ?Sized> LogDensity for Whitened<'_, P> {
    fn dim(&self) -> usize {
        self.model.dim()
    }

    fn log_density(&self, z: &[f64], gradient: &mut [f64]) -> Option<f64> {
        let p = z.len();
        let theta = self.theta(z);
        let mut g = vec![0.0; p];
        let f = self.model.log_posterior(&theta, &mut g);
        gradient.fill(0.0);
        for (i, &gi) in g.iter().enumerate() {
            let row = &self.l[i * p..(i + 1) * p];
            for j in 0..p {
                gradient[j] += row[j] * gi;
            }
        }
        (f.is_finite() && gradient.iter().all(|v| v.is_finite())).then_some(f)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Central finite differences of `f` against `grad` at `x`.
    pub(crate) fn assert_gradient(f: impl Fn(&[f64], &mut [f64]) -> f64, x: &[f64]) {
        let mut grad = vec![0.0; x.len()];
        f(x, &mut grad);
        let mut scratch = vec![0.0; x.len()];
        for j in 0..x.len() {
            let h = 1e-5;
            let mut up = x.to_vec();
            up[j] += h;
            let mut down = x.to_vec();
            down[j] -= h;
            let fd = (f(&up, &mut scratch) - f(&down, &mut scratch)) / (2.0 * h);
            assert!(
                (fd - grad[j]).abs() <= 1e-5 * (1.0 + grad[j].abs()),
                "parameter {j}: finite difference {fd}, gradient {}",
                grad[j]
            );
        }
    }

    /// A correlated normal with a quartic term: not a quadratic, so
    /// Newton takes several steps; the default Hessian is a finite
    /// difference.
    struct Quartic;

    impl Posterior for Quartic {
        fn dim(&self) -> usize {
            2
        }
        fn log_posterior(&self, t: &[f64], g: &mut [f64]) -> f64 {
            let (x, y) = (t[0] - 1.0, t[1] + 2.0);
            g[0] = -(2.0 * x + y) - 4.0 * x.powi(3);
            g[1] = -(x + 2.0 * y);
            -(x * x + x * y + y * y) - x.powi(4)
        }
    }

    #[test]
    fn the_finite_difference_hessian_and_the_mode() {
        let h = Quartic.negative_hessian(&[1.5, 0.0]);
        // -d2f: [2 + 12 x^2, 1; 1, 2] at x = 0.5.
        for (a, b) in h.iter().zip([5.0, 1.0, 1.0, 2.0]) {
            assert!((a - b).abs() < 1e-6, "{h:?}");
        }
        let m = mode(&Quartic, vec![5.0, 5.0]).unwrap();
        assert!(
            (m[0] - 1.0).abs() < 1e-9 && (m[1] + 2.0).abs() < 1e-9,
            "{m:?}"
        );
        let w = Whitened::new(&Quartic, &m).unwrap();
        assert_gradient(|z, g| w.log_density(z, g).unwrap(), &[0.3, -0.7]);
    }

    /// Not concave at the start: the ridge carries Newton to the mode.
    struct DoubleWell;

    impl Posterior for DoubleWell {
        fn dim(&self) -> usize {
            1
        }
        fn log_posterior(&self, t: &[f64], g: &mut [f64]) -> f64 {
            // f = -(x^2 - 1)^2 - 0.1 (x - 1)^2: modes near +1, a saddle
            // region around 0.
            let x = t[0];
            g[0] = -4.0 * x * (x * x - 1.0) - 0.2 * (x - 1.0);
            -(x * x - 1.0).powi(2) - 0.1 * (x - 1.0).powi(2)
        }
    }

    #[test]
    fn the_ridge_handles_a_non_concave_start() {
        let m = mode(&DoubleWell, vec![0.1]).unwrap();
        let mut g = [0.0];
        DoubleWell.log_posterior(&m, &mut g);
        assert!(g[0].abs() < 1e-8, "{m:?} {g:?}");
        assert!(DoubleWell.negative_hessian(&m)[0] > 0.0);
    }
}
