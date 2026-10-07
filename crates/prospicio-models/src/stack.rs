//! Model weights for blending from pointwise out-of-sample log predictive
//! densities: stacking and pseudo-BMA (Yao, Vehtari, Simpson and Gelman,
//! 2018, "Using stacking to average Bayesian predictive distributions").
//!
//! The input is, for each model, the log predictive density of each
//! observation when that observation was held out: PSIS-LOO pointwise
//! values for a Bayesian fit (`prospicio_bayes::elpd`), or cross-validated
//! log densities ([`Family::log_density`](crate::Family::log_density) on
//! each test fold) for any other model. So the weights apply to every
//! engine, Bayesian or not. Blend the models' predictive distributions
//! with `prospicio_prob::PredictiveDistribution::blend`.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_math::linalg::{cholesky, cholesky_solve};

/// Stacking weights: the `w` on the simplex that maximizes
/// `Σ_i log Σ_k w_k exp(lpd_ik)`, the log score of the mixture of the
/// models' predictive distributions. `lpd[k][i]` is model `k`'s held-out
/// log density of observation `i`.
///
/// Unlike pseudo-BMA, stacking gives weight to a model that is worse on
/// its own when it predicts well where the others do not, and drops
/// models that add nothing (their weights are exactly 0). The objective
/// is concave; it is solved by EM, then Newton's method on the models
/// kept, checked against the optimality conditions.
///
/// ```
/// use prospicio_models::stack::stacking_weights;
///
/// // Model 0 predicts the first half well, model 1 the second half.
/// let a = [-0.1, -0.1, -3.0, -3.0];
/// let b = [-3.0, -3.0, -0.1, -0.1];
/// let w = stacking_weights(&[a.to_vec(), b.to_vec()]).unwrap();
/// assert!((w[0] - 0.5).abs() < 1e-9 && (w[1] - 0.5).abs() < 1e-9);
/// ```
pub fn stacking_weights(lpd: &[Vec<f64>]) -> Result<Vec<f64>> {
    let n = check(lpd)?;
    let k = lpd.len();
    // e[i * k + j] = exp(lpd_ij - max_j lpd_ij): rows scaled for range.
    let mut e = vec![0.0; n * k];
    for i in 0..n {
        let m = lpd.iter().map(|l| l[i]).fold(f64::NEG_INFINITY, f64::max);
        for j in 0..k {
            e[i * k + j] = (lpd[j][i] - m).exp();
        }
    }
    let mut w = vec![1.0 / k as f64; k];
    // A few EM steps (monotone: w_j <- w_j mean_i e_ij / (e_i . w)) give a
    // good start; then active-set Newton.
    for _ in 0..50 {
        let g = gradient(&e, &w, n, k);
        for (wj, gj) in w.iter_mut().zip(&g) {
            *wj *= gj / n as f64;
        }
    }
    let nf = n as f64;
    for _round in 0..100 {
        let mut support: Vec<usize> = (0..k).filter(|&j| w[j] > 0.0).collect();
        newton(&e, &mut w, &mut support, n, k);
        // Optimality: the gradient is n on the support and at most n off it.
        let g = gradient(&e, &w, n, k);
        let on = support.iter().all(|&j| (g[j] - nf).abs() <= 1e-8 * nf);
        let off = (0..k).all(|j| w[j] > 0.0 || g[j] <= nf * (1.0 + 1e-9));
        if on && off {
            return Ok(w);
        }
        // Bring back the best excluded model with a positive slope.
        if !off {
            let j = (0..k)
                .filter(|&j| w[j] == 0.0)
                .max_by(|&a, &b| g[a].total_cmp(&g[b]))
                .expect("a violated model is excluded");
            w.iter_mut().for_each(|x| *x *= 0.99);
            w[j] = 0.01;
        }
    }
    Err(Error::Data("stacking weights did not converge".into()))
}

/// Pseudo-BMA weights: `w_k ∝ exp(elpd_k)`, with `elpd_k = Σ_i lpd_ik`.
///
/// With `bootstrap = Some((draws, seed))` these are pseudo-BMA+ weights:
/// each of `draws` Bayesian-bootstrap replicates reweights the
/// observations by Dirichlet(1, ..., 1) (replicate `b` from stream `b` of
/// `seed`), and the weights are averaged over replicates. That accounts
/// for the uncertainty in the elpd differences, so a model that is only
/// slightly better does not take all the weight.
pub fn pseudo_bma_weights(lpd: &[Vec<f64>], bootstrap: Option<(usize, u64)>) -> Result<Vec<f64>> {
    let n = check(lpd)?;
    let k = lpd.len();
    let Some((draws, seed)) = bootstrap else {
        let elpd: Vec<f64> = lpd.iter().map(|l| l.iter().sum()).collect();
        return Ok(softmax(&elpd));
    };
    if draws == 0 {
        return Err(Error::InvalidParameter {
            name: "draws",
            value: 0.0,
            reason: "must be positive",
        });
    }
    let mut out = vec![0.0; k];
    let mut g = vec![0.0; n];
    for b in 0..draws {
        let mut rng = StreamRng::new(seed, b as u64);
        // Dirichlet(1, ..., 1) as normalized standard exponentials.
        for x in g.iter_mut() {
            *x = -rng.next_open01().ln();
        }
        let total: f64 = g.iter().sum();
        let z: Vec<f64> = lpd
            .iter()
            .map(|l| n as f64 * l.iter().zip(&g).map(|(a, gi)| a * gi).sum::<f64>() / total)
            .collect();
        for (o, wk) in out.iter_mut().zip(softmax(&z)) {
            *o += wk / draws as f64;
        }
    }
    Ok(out)
}

fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

fn check(lpd: &[Vec<f64>]) -> Result<usize> {
    if lpd.len() < 2 {
        return Err(Error::Data("weights need at least two models".into()));
    }
    let n = lpd[0].len();
    if n == 0 || lpd.iter().any(|l| l.len() != n) {
        return Err(Error::Data(
            "every model needs the same, non-zero number of pointwise log densities".into(),
        ));
    }
    if lpd.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Data("pointwise log densities must be finite".into()));
    }
    Ok(n)
}

/// `∂/∂w_j Σ_i log(e_i · w) = Σ_i e_ij / (e_i · w)`.
fn gradient(e: &[f64], w: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut g = vec![0.0; k];
    for row in e.chunks_exact(k).take(n) {
        let s: f64 = row.iter().zip(w).map(|(a, b)| a * b).sum();
        for (gj, a) in g.iter_mut().zip(row) {
            *gj += a / s;
        }
    }
    g
}

fn objective(e: &[f64], w: &[f64], k: usize) -> f64 {
    e.chunks_exact(k)
        .map(|row| row.iter().zip(w).map(|(a, b)| a * b).sum::<f64>().ln())
        .sum()
}

/// Damped Newton's method on the face of the simplex spanned by
/// `support`, eliminating its last weight. A step that would make a weight
/// negative stops at the boundary and drops that model from the support.
fn newton(e: &[f64], w: &mut [f64], support: &mut Vec<usize>, n: usize, k: usize) {
    for _ in 0..500 {
        let m = support.len();
        if m < 2 {
            if let [only] = support[..] {
                w.iter_mut().for_each(|x| *x = 0.0);
                w[only] = 1.0;
            }
            return;
        }
        let mut g = vec![0.0; m];
        let mut h = vec![0.0; m * m];
        for row in e.chunks_exact(k).take(n) {
            let s: f64 = row.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            for (a, &ja) in support.iter().enumerate() {
                let xa = row[ja] / s;
                g[a] += xa;
                for (b, &jb) in support.iter().enumerate() {
                    h[a * m + b] += xa * row[jb] / s;
                }
            }
        }
        // Reduced gradient and negated Hessian in the m - 1 free weights.
        let r = m - 1;
        let grad: Vec<f64> = (0..r).map(|a| g[a] - g[r]).collect();
        if grad.iter().all(|v| v.abs() <= 1e-12 * n as f64) {
            return;
        }
        let mut neg_h = vec![0.0; r * r];
        for a in 0..r {
            for b in 0..r {
                neg_h[a * r + b] = h[a * m + b] - h[a * m + r] - h[r * m + b] + h[r * m + r];
            }
        }
        // Levenberg damping when the models are nearly collinear.
        let scale = (0..r)
            .map(|a| neg_h[a * r + a])
            .fold(0.0, f64::max)
            .max(1e-300);
        let mut damping = 0.0;
        let step = loop {
            let mut a = neg_h.clone();
            for d in 0..r {
                a[d * r + d] += damping;
            }
            if let Some(l) = cholesky(&a, r) {
                break cholesky_solve(&l, &grad);
            }
            damping = if damping == 0.0 {
                1e-12 * scale
            } else {
                damping * 10.0
            };
        };
        let mut full = vec![0.0; m];
        full[..r].copy_from_slice(&step);
        full[r] = -step.iter().sum::<f64>();
        // Largest step that keeps every weight non-negative.
        let mut t_max = f64::INFINITY;
        let mut blocking = None;
        for (a, &ja) in support.iter().enumerate() {
            if full[a] < 0.0 {
                let t = -w[ja] / full[a];
                if t < t_max {
                    t_max = t;
                    blocking = Some(a);
                }
            }
        }
        let base = objective(e, w, k);
        let mut t = t_max.min(1.0);
        let mut moved = false;
        for _ in 0..60 {
            let mut trial = w.to_vec();
            for (a, &ja) in support.iter().enumerate() {
                trial[ja] = (w[ja] + t * full[a]).max(0.0);
            }
            if objective(e, &trial, k) >= base {
                w.copy_from_slice(&trial);
                moved = true;
                break;
            }
            t *= 0.5;
        }
        if !moved {
            return;
        }
        if t == t_max {
            if let Some(a) = blocking {
                w[support[a]] = 0.0;
                support.remove(a);
            }
        }
        let s: f64 = w.iter().sum();
        w.iter_mut().for_each(|x| *x /= s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dominated_model_gets_no_stacking_weight() {
        let a = vec![-1.0, -1.2, -0.8, -1.1];
        let worse: Vec<f64> = a.iter().map(|v| v - 0.5).collect();
        let w = stacking_weights(&[a, worse]).unwrap();
        assert_eq!(w, vec![1.0, 0.0]);
    }

    #[test]
    fn pseudo_bma_is_a_softmax_of_elpd_and_the_bootstrap_shrinks_it() {
        let a = vec![-1.0, -2.0, -1.5, -1.0, -0.5, -2.5];
        let b = vec![-1.2, -1.6, -1.4, -1.3, -0.9, -1.8];
        let w = pseudo_bma_weights(&[a.clone(), b.clone()], None).unwrap();
        let (ea, eb): (f64, f64) = (a.iter().sum(), b.iter().sum());
        assert!((w[0] - 1.0 / (1.0 + (eb - ea).exp())).abs() < 1e-15);
        let plus = pseudo_bma_weights(&[a.clone(), b.clone()], Some((4000, 1))).unwrap();
        assert!((plus[0] + plus[1] - 1.0).abs() < 1e-12);
        // The better model by elpd keeps more weight, but less than plain
        // pseudo-BMA gives it.
        assert!(plus[1] > 0.5 && plus[1] < w[1]);
        assert_eq!(plus, pseudo_bma_weights(&[a, b], Some((4000, 1))).unwrap());
    }

    #[test]
    fn rejects_bad_input() {
        assert!(stacking_weights(&[vec![-1.0]]).is_err());
        assert!(stacking_weights(&[vec![-1.0], vec![-1.0, -2.0]]).is_err());
        assert!(stacking_weights(&[vec![f64::NAN], vec![-1.0]]).is_err());
        assert!(pseudo_bma_weights(&[vec![-1.0], vec![-1.0]], Some((0, 1))).is_err());
    }
}
