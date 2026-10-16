//! Moment matching for the one-year view (`docs/design/reserving-bayes.md`,
//! decision 18): importance sampling with an adaptively transformed
//! proposal, as Paananen, Piironen, Bürkner and Vehtari (2021, "Implicitly
//! adaptive importance sampling", *Statistics and Computing* 31, 16) give
//! it and R's `loo::loo_moment_match` implements it.
//!
//! A simulation whose Pareto `k̂` is high reweights the posterior draws
//! towards a target (the posterior given the triangle extended by the
//! year) that lies far from them. Moment matching moves the draws instead:
//! an affine map takes them to the weighted mean (and marginal variances,
//! then covariance) that the importance weights estimate, the weights are
//! recomputed for the moved draws, and the steps repeat while `k̂`
//! improves. Each step is, in `loo`'s order:
//!
//! 1. a shift to the weighted mean;
//! 2. a shift and a scaling to the weighted mean and marginal variances;
//! 3. a shift and a linear map `L_w L⁻¹` to the weighted mean and
//!    covariance (`L`, `L_w` the Cholesky factors of the draws' plain and
//!    weighted covariance).
//!
//! A step is kept only if it lowers `k̂`; after a kept step the sequence
//! starts again at 1, and when none lowers it, or `k̂` is at most the
//! threshold, or [`MAX_STEPS`] steps are kept, matching stops. Every map is
//! affine, so the moved draws' proposal density is the posterior's at the
//! draw it came from over a constant Jacobian: the log ratio of a moved
//! draw `T(θ)` is `log p(T(θ) | y, year) − log p(θ | y)`, the constant
//! dropping out of the self-normalised weights.
//!
//! The estimate is then `loo`'s split one (its default, `split = TRUE`):
//! half of the draws moved by the whole map `T`, half left, weighted
//! against the mixture of the two proposals, `½ p(x | y) + ½ p(T⁻¹(x) |
//! y) / |det T|`, which keeps the estimate sound where `T` overshoots.
//! The closing reserve is the weighted mean of the model's closing reserve
//! at those draws.

use prospicio_bayes::nuts::PosteriorDraws;
use prospicio_math::linalg::{cholesky, lower_solve};
use rayon::prelude::*;

use super::one_year::YearModel;

/// The most steps kept per simulation, `loo`'s `max_iters`.
pub(crate) const MAX_STEPS: usize = 30;

/// What moment matching needs of the opening posterior, computed once per
/// segment: its draws, their log posterior on the observed triangle and
/// their means.
pub(crate) struct Opening<'a> {
    posterior: &'a PosteriorDraws,
    /// `log_posterior(θ_s, [])` per draw.
    log_prob: Vec<f64>,
    /// The draws' mean per parameter.
    mean: Vec<f64>,
}

impl<'a> Opening<'a> {
    pub(crate) fn new<M: YearModel>(model: &M, posterior: &'a PosteriorDraws) -> Self {
        let log_prob = (0..posterior.n_draws())
            .into_par_iter()
            .map(|d| model.log_posterior(posterior.draw(d), &[]))
            .collect();
        Self {
            posterior,
            log_prob,
            mean: posterior.mean(),
        }
    }
}

/// The kinds of step, in the order they are tried.
#[derive(Debug, Clone, Copy)]
enum Step {
    Shift,
    Scale,
    Covariance,
}

/// The affine map of all kept steps: `T(θ) = M (θ − m) + m + shift`, with
/// `m` the opening draws' mean and `M` lower triangular (each step's
/// linear part is diagonal or `L_w L⁻¹`, both lower triangular), row-major.
struct Affine {
    mapping: Vec<f64>,
    shift: Vec<f64>,
}

/// Moment matching of one simulation whose year is `year`, from its
/// smoothed log weights `log_w` and `k̂` `k` on the opening draws. The
/// closing reserve of each opening draw is `closing_by_draw` (row per
/// draw) when the model's closing reserve does not depend on the year.
///
/// Returns `k̂` after the last kept step (the simulation's own if none was
/// kept) and each origin's closing reserve when that `k̂` is at most
/// `threshold`; `None` otherwise (the simulation is then refitted).
pub(crate) fn moment_match<M: YearModel>(
    model: &M,
    opening: &Opening<'_>,
    closing_by_draw: Option<&[f64]>,
    year: &[f64],
    log_w: &[f64],
    k: f64,
    threshold: f64,
) -> (f64, Option<Vec<f64>>) {
    let posterior = opening.posterior;
    let (s, p) = (posterior.n_draws(), posterior.dim());
    let target = |x: &[f64]| model.log_posterior(x, &[]) + model.year_log_likelihood(x, year);
    // The log ratio of moved draws `u` against the opening draws they came
    // from; NaN, as where both terms are infinite, is a zero weight, as in
    // `loo`.
    let ratios = |u: &[f64]| -> Vec<f64> {
        u.par_chunks_exact(p)
            .zip(opening.log_prob.par_iter())
            .map(|(x, lp)| {
                let r = target(x) - lp;
                if r.is_nan() { f64::NEG_INFINITY } else { r }
            })
            .collect()
    };

    let mut u = posterior.values().to_vec();
    let mut w = normalised(log_w);
    let mut k = k;
    let mut total = Affine {
        mapping: identity(p),
        shift: vec![0.0; p],
    };
    let mut steps = 0;
    'outer: while steps < MAX_STEPS && k > threshold {
        for kind in [Step::Shift, Step::Scale, Step::Covariance] {
            let (moved, shift, linear) = step(kind, &u, &w, s, p);
            // A ratio PSIS cannot take (+inf) ends matching, as an
            // exception in the log density ends `loo`'s.
            let Some((lw, k_new)) = smooth(&ratios(&moved)) else {
                break 'outer;
            };
            if k_new < k {
                u = moved;
                w = normalised(&lw);
                k = k_new;
                steps += 1;
                for (t, v) in total.shift.iter_mut().zip(&shift) {
                    *t += v;
                }
                if let Some(l) = linear {
                    total.mapping = mat_mul(&l, &total.mapping, p);
                }
                continue 'outer;
            }
        }
        break;
    }
    let closing = (steps > 0 && k <= threshold)
        .then(|| split(model, opening, closing_by_draw, year, &total))
        .flatten();
    (k, closing)
}

/// One step of kind `kind` on the draws `u` (row per draw) under the
/// normalised weights `w`: the moved draws, the step's shift and its
/// linear part (`None` for a shift).
fn step(
    kind: Step,
    u: &[f64],
    w: &[f64],
    s: usize,
    p: usize,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    let mut mean = vec![0.0; p];
    let mut mean_w = vec![0.0; p];
    for (x, &wd) in u.chunks_exact(p).zip(w) {
        for j in 0..p {
            mean[j] += x[j] / s as f64;
            mean_w[j] += wd * x[j];
        }
    }
    let shift: Vec<f64> = mean_w.iter().zip(&mean).map(|(a, b)| a - b).collect();
    let linear = match kind {
        Step::Shift => None,
        Step::Scale => {
            // `loo`'s `shift_and_scale`: the weighted variance about the
            // weighted mean, times S / (S − 1), over the plain variance.
            let mut var = vec![0.0; p];
            let mut var_w = vec![0.0; p];
            for (x, &wd) in u.chunks_exact(p).zip(w) {
                for j in 0..p {
                    var[j] += (x[j] - mean[j]).powi(2) / (s - 1) as f64;
                    var_w[j] += wd * (x[j] - mean_w[j]).powi(2);
                }
            }
            let mut l = vec![0.0; p * p];
            for j in 0..p {
                let scaling = (var_w[j] * s as f64 / (s - 1) as f64 / var[j]).sqrt();
                l[j * p + j] = if scaling.is_finite() && scaling > 0.0 {
                    scaling
                } else {
                    1.0
                };
            }
            Some(l)
        }
        Step::Covariance => Some(covariance_map(u, w, &mean, &mean_w, s, p)),
    };
    let moved = match &linear {
        None => u
            .chunks_exact(p)
            .flat_map(|x| x.iter().zip(&shift).map(|(a, b)| a + b))
            .collect(),
        Some(l) => {
            let mut moved = Vec::with_capacity(u.len());
            let mut centred = vec![0.0; p];
            for x in u.chunks_exact(p) {
                for j in 0..p {
                    centred[j] = x[j] - mean[j];
                }
                for i in 0..p {
                    let lx: f64 = (0..=i).map(|j| l[i * p + j] * centred[j]).sum();
                    moved.push(lx + mean_w[i]);
                }
            }
            moved
        }
    };
    (moved, shift, linear)
}

/// `loo`'s `shift_and_cov`: `L_w L⁻¹`, with `L` the Cholesky factor of
/// the draws' covariance (divisor S − 1) and `L_w` of their weighted
/// covariance about the weighted mean (`cov.wt`'s unbiased divisor
/// `1 − Σ w²`); the identity when either is not positive definite.
fn covariance_map(
    u: &[f64],
    w: &[f64],
    mean: &[f64],
    mean_w: &[f64],
    s: usize,
    p: usize,
) -> Vec<f64> {
    let mut cov = vec![0.0; p * p];
    let mut cov_w = vec![0.0; p * p];
    for (x, &wd) in u.chunks_exact(p).zip(w) {
        for i in 0..p {
            let (a, aw) = (x[i] - mean[i], x[i] - mean_w[i]);
            for j in 0..=i {
                cov[i * p + j] += a * (x[j] - mean[j]);
                cov_w[i * p + j] += wd * aw * (x[j] - mean_w[j]);
            }
        }
    }
    let unbiased = 1.0 - w.iter().map(|v| v * v).sum::<f64>();
    for i in 0..p {
        for j in 0..=i {
            cov[i * p + j] /= (s - 1) as f64;
            cov_w[i * p + j] /= unbiased;
        }
    }
    let (Some(l), Some(l_w)) = (cholesky(&cov, p), cholesky(&cov_w, p)) else {
        return identity(p);
    };
    // L⁻¹, column by column, then L_w L⁻¹ (lower triangular).
    let mut inv = vec![0.0; p * p];
    let (mut e, mut col) = (vec![0.0; p], vec![0.0; p]);
    for j in 0..p {
        e.fill(0.0);
        e[j] = 1.0;
        lower_solve(&l, &e, &mut col);
        for i in 0..p {
            inv[i * p + j] = col[i];
        }
    }
    mat_mul(&l_w, &inv, p)
}

/// `loo`'s split estimate of the closing reserve under the whole map
/// `total`: the first half of the opening draws moved by `T`, the second
/// half left, each weighted by the target over the mixture of the two
/// proposals and Pareto-smoothed. `None` if PSIS cannot take the ratios.
fn split<M: YearModel>(
    model: &M,
    opening: &Opening<'_>,
    closing_by_draw: Option<&[f64]>,
    year: &[f64],
    total: &Affine,
) -> Option<Vec<f64>> {
    let posterior = opening.posterior;
    let (s, p, no) = (posterior.n_draws(), posterior.dim(), model.n_origins());
    let half = s / 2;
    let m = &opening.mean;
    let l = &total.mapping;
    let log_det: f64 = (0..p).map(|i| l[i * p + i].ln()).sum();
    // T(θ) and T⁻¹(θ).
    let forward = |theta: &[f64]| -> Vec<f64> {
        (0..p)
            .map(|i| {
                let lx: f64 = (0..=i).map(|j| l[i * p + j] * (theta[j] - m[j])).sum();
                lx + m[i] + total.shift[i]
            })
            .collect()
    };
    let inverse = |theta: &[f64]| -> Vec<f64> {
        let b: Vec<f64> = (0..p).map(|j| theta[j] - m[j] - total.shift[j]).collect();
        let mut y = vec![0.0; p];
        lower_solve(l, &b, &mut y);
        y.iter().zip(m).map(|(a, b)| a + b).collect()
    };
    // Per draw: the draw used (moved or not), its log ratio and its closing
    // reserve.
    let rows: Vec<(f64, Vec<f64>)> = (0..s)
        .into_par_iter()
        .map(|d| {
            let theta = posterior.draw(d);
            let (x, lp_x, lp_inv) = if d < half {
                let x = forward(theta);
                let lp = model.log_posterior(&x, &[]);
                (Some(x), lp, opening.log_prob[d])
            } else {
                let lp_inv = model.log_posterior(&inverse(theta), &[]);
                (None, opening.log_prob[d], lp_inv)
            };
            let x_ref = x.as_deref().unwrap_or(theta);
            let mixture = log_add_exp(lp_x, lp_inv - log_det);
            let mut r = lp_x + model.year_log_likelihood(x_ref, year) - mixture;
            if r.is_nan() {
                r = f64::NEG_INFINITY;
            }
            let mut row = vec![0.0; no];
            match (x.is_none(), closing_by_draw) {
                (true, Some(by_draw)) => row.copy_from_slice(&by_draw[d * no..(d + 1) * no]),
                _ => model.closing(x_ref, year, &mut row),
            }
            (r, row)
        })
        .collect();
    let lr: Vec<f64> = rows.iter().map(|(r, _)| *r).collect();
    let (lw, _) = smooth(&lr)?;
    let w = normalised(&lw);
    let mut closing = vec![0.0; no];
    for ((_, row), wd) in rows.iter().zip(&w) {
        for (c, v) in closing.iter_mut().zip(row) {
            *c += wd * v;
        }
    }
    Some(closing)
}

/// PSIS of `log_r`, relative efficiency 1 as the engine's; `None` when it
/// cannot take them (a NaN or `+inf`, or no finite ratio; `psis` rejects
/// each).
fn smooth(log_r: &[f64]) -> Option<(Vec<f64>, f64)> {
    prospicio_bayes::elpd::psis(log_r, 1.0).ok()
}

/// The weights `exp(log_w)` normalised to sum to 1.
fn normalised(log_w: &[f64]) -> Vec<f64> {
    let max = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = log_w.iter().map(|v| (v - max).exp()).collect();
    let total: f64 = w.iter().sum();
    w.into_iter().map(|v| v / total).collect()
}

fn log_add_exp(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    if max == f64::NEG_INFINITY {
        return max;
    }
    max + ((a - max).exp() + (b - max).exp()).ln()
}

fn identity(p: usize) -> Vec<f64> {
    let mut m = vec![0.0; p * p];
    for i in 0..p {
        m[i * p + i] = 1.0;
    }
    m
}

/// `a b` for row-major `p × p` matrices.
fn mat_mul(a: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut c = vec![0.0; p * p];
    for i in 0..p {
        for k in 0..p {
            let aik = a[i * p + k];
            if aik != 0.0 {
                for j in 0..p {
                    c[i * p + j] += aik * b[k * p + j];
                }
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{Error, Result};
    use prospicio_bayes::nuts::{LogDensity, Sampler, sample};
    use prospicio_core::StreamRng;

    /// The standard normal in `p` dimensions, sampled for the opening
    /// draws.
    struct Normal(usize);

    impl LogDensity for Normal {
        fn dim(&self) -> usize {
            self.0
        }
        fn log_density(&self, x: &[f64], grad: &mut [f64]) -> Option<f64> {
            for (g, v) in grad.iter_mut().zip(x) {
                *g = -v;
            }
            Some(-0.5 * x.iter().map(|v| v * v).sum::<f64>())
        }
    }

    /// A conjugate toy: the opening posterior `N(0, I)`, the year one
    /// observation of each `theta[j]` with sd `sigma`, and the closing
    /// reserve `theta` itself, so the target's mean is known. The log
    /// ratios then spread as a scaled chi-square, whose heavy right tail
    /// in the weights is what a far year gives the reserving models.
    struct Toy {
        p: usize,
        sigma: f64,
    }

    impl YearModel for Toy {
        fn n_origins(&self) -> usize {
            self.p
        }
        fn n_year(&self) -> usize {
            self.p
        }
        fn draw_year(&self, theta: &[f64], rng: &mut StreamRng, year: &mut [f64]) {
            for (y, t) in year.iter_mut().zip(theta) {
                *y = t + self.sigma * (rng.next_open01() - 0.5);
            }
        }
        fn year_log_likelihood(&self, theta: &[f64], year: &[f64]) -> f64 {
            theta
                .iter()
                .zip(year)
                .map(|(t, y)| -0.5 * ((y - t) / self.sigma).powi(2))
                .sum()
        }
        fn paid(&self, _year: &[f64], paid: &mut [f64]) {
            paid.fill(0.0);
        }
        fn opening(&self, theta: &[f64], reserve: &mut [f64]) {
            reserve.copy_from_slice(theta);
        }
        fn closing(&self, theta: &[f64], _year: &[f64], reserve: &mut [f64]) {
            reserve.copy_from_slice(theta);
        }
        fn closing_uses_year(&self) -> bool {
            false
        }
        fn refit(&self, _year: &[f64], _seed: u64) -> Result<(PosteriorDraws, bool)> {
            Err(Error::Bayes("no refit".into()))
        }
        fn log_posterior(&self, theta: &[f64], year: &[f64]) -> f64 {
            let prior = -0.5 * theta.iter().map(|v| v * v).sum::<f64>();
            if year.is_empty() {
                prior
            } else {
                prior + self.year_log_likelihood(theta, year)
            }
        }
    }

    fn opening_draws(p: usize) -> PosteriorDraws {
        let sampler = Sampler {
            chains: 4,
            tune: 500,
            draws: 1000,
            seed: 3,
            ..Sampler::default()
        };
        sample(&Normal(p), &vec![0.0; p], sampler).unwrap()
    }

    /// The opening posterior `N(0, I)` stored as `exp(theta)`, a
    /// constrained scale, with a log posterior that has the Jacobian
    /// (`-log x` per parameter) or leaves it out.
    struct ExpStored {
        jacobian: bool,
    }

    impl YearModel for ExpStored {
        fn n_origins(&self) -> usize {
            1
        }
        fn n_year(&self) -> usize {
            1
        }
        fn draw_year(&self, _theta: &[f64], _rng: &mut StreamRng, year: &mut [f64]) {
            year.fill(0.0);
        }
        fn year_log_likelihood(&self, _theta: &[f64], _year: &[f64]) -> f64 {
            0.0
        }
        fn paid(&self, _year: &[f64], paid: &mut [f64]) {
            paid.fill(0.0);
        }
        fn opening(&self, _theta: &[f64], reserve: &mut [f64]) {
            reserve.fill(0.0);
        }
        fn closing(&self, _theta: &[f64], _year: &[f64], reserve: &mut [f64]) {
            reserve.fill(0.0);
        }
        fn refit(&self, _year: &[f64], _seed: u64) -> Result<(PosteriorDraws, bool)> {
            Err(Error::Bayes("no refit".into()))
        }
        fn log_posterior(&self, theta: &[f64], _year: &[f64]) -> f64 {
            theta
                .iter()
                .map(|x| {
                    let v = x.ln();
                    -0.5 * v * v - if self.jacobian { v } else { 0.0 }
                })
                .sum()
        }
    }

    /// Conformance check 5 (the score identity) tells a log posterior
    /// that is the stored draws' density from one that leaves out the
    /// Jacobian of their scale, which check 1 cannot see.
    #[test]
    fn the_score_check_catches_a_missing_jacobian() {
        let p = 2;
        let names = vec!["x0".to_string(), "x1".to_string()];
        let stored = opening_draws(p)
            .transform(names, |v| v.iter().map(|t| t.exp()).collect())
            .unwrap();
        let worst = |jacobian| {
            crate::bayes::one_year::conformance::score_z(&ExpStored { jacobian }, &stored)
                .iter()
                .fold(0.0, |m: f64, v| m.max(v.abs()))
        };
        let (with, without) = (worst(true), worst(false));
        assert!(
            with < crate::bayes::one_year::conformance::SCORE_Z,
            "with the Jacobian: z {with}"
        );
        assert!(
            without > 2.0 * crate::bayes::one_year::conformance::SCORE_Z,
            "without the Jacobian: z {without}"
        );
    }

    /// A year far in the opening posterior's tail (`k̂` well above 0.7)
    /// is brought below the threshold, and the closing reserve is the
    /// conjugate posterior mean, where plain reweighting is far off.
    #[test]
    fn moment_matching_rescues_a_far_year_and_finds_the_posterior_mean() {
        let p = 6;
        let model = Toy { p, sigma: 0.3 };
        let posterior = opening_draws(p);
        let year = [1.0; 6];
        let log_r: Vec<f64> = (0..posterior.n_draws())
            .map(|d| model.year_log_likelihood(posterior.draw(d), &year))
            .collect();
        let (log_w, k) = prospicio_bayes::elpd::psis(&log_r, 1.0).unwrap();
        assert!(k > 0.7, "the year should be flagged, k {k}");
        let opening = Opening::new(&model, &posterior);
        let by_draw = posterior.values().to_vec();
        let (k_mm, closing) = moment_match(&model, &opening, Some(&by_draw), &year, &log_w, k, 0.7);
        let closing = closing.expect("rescued");
        assert!(k_mm <= 0.7, "k after matching {k_mm}");
        // Each theta[j]'s posterior: precision 1 + 1 / sigma^2, mean
        // (y / sigma^2) over it (0.92) and sd 0.29; matching lands within
        // 0.05 of it on every parameter (0.029 at this seed), reweighting
        // alone 0.25 off.
        let exact = (1.0 / 0.09) / (1.0 + 1.0 / 0.09);
        let w = normalised(&log_w);
        let gap = |c: &[f64]| c.iter().map(|v| (v - exact).abs()).fold(0.0, f64::max);
        assert!(gap(&closing) < 0.05, "{closing:?} vs {exact}");
        // Plain reweighting is much further off (k̂ about 5).
        let mut reweighted = vec![0.0; p];
        for (d, wd) in w.iter().enumerate() {
            for (r, t) in reweighted.iter_mut().zip(posterior.draw(d)) {
                *r += wd * t;
            }
        }
        assert!(
            gap(&reweighted) > 2.0 * gap(&closing),
            "reweighted {reweighted:?}, matched {closing:?}, exact {exact}"
        );
    }

    /// The scaling and covariance steps give moved draws whose plain mean
    /// and covariance are the weighted ones (`loo`'s definitions).
    #[test]
    fn the_steps_match_the_weighted_moments() {
        let posterior = opening_draws(3);
        let (s, p) = (posterior.n_draws(), 3);
        let u = posterior.values();
        // Weights tilted towards large theta[0] + theta[1].
        let log_w: Vec<f64> = u.chunks_exact(p).map(|x| 0.8 * (x[0] + x[1])).collect();
        let w = normalised(&log_w);
        let moments = |v: &[f64], w: &dyn Fn(usize) -> f64, divisor: f64| {
            let mut m = vec![0.0; p];
            for (d, x) in v.chunks_exact(p).enumerate() {
                for j in 0..p {
                    m[j] += w(d) * x[j];
                }
            }
            let mut c = vec![0.0; p * p];
            for (d, x) in v.chunks_exact(p).enumerate() {
                for i in 0..p {
                    for j in 0..p {
                        c[i * p + j] += w(d) * (x[i] - m[i]) * (x[j] - m[j]) / divisor;
                    }
                }
            }
            (m, c)
        };
        let unbiased = 1.0 - w.iter().map(|v| v * v).sum::<f64>();
        let (mean_w, cov_w) = moments(u, &|d| w[d], unbiased);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9 * (1.0 + b.abs());
        for kind in [Step::Shift, Step::Scale, Step::Covariance] {
            let (moved, shift, _) = step(kind, u, &w, s, p);
            let (mean, cov) = moments(&moved, &|_| 1.0 / s as f64, (s - 1) as f64 / s as f64);
            assert!(
                mean.iter().zip(&mean_w).all(|(a, b)| close(*a, *b)),
                "{kind:?}"
            );
            assert_eq!(shift.len(), p);
            match kind {
                Step::Shift => {}
                Step::Scale => {
                    // loo scales to the weighted variance times S / (S - 1)
                    // about the weighted mean.
                    let (_, cov_b) = moments(u, &|d| w[d], 1.0);
                    for j in 0..p {
                        let target = cov_b[j * p + j] * s as f64 / (s - 1) as f64;
                        assert!(close(cov[j * p + j], target), "{kind:?} {j}");
                    }
                }
                Step::Covariance => {
                    assert!(
                        cov.iter().zip(&cov_w).all(|(a, b)| close(*a, *b)),
                        "{cov:?}"
                    );
                }
            }
        }
    }
}
