//! Maximum likelihood fits of the body families (lognormal, gamma,
//! Weibull, Burr) to large losses with reporting thresholds and policy
//! limits: the same [`LargeLosses`] the Pareto fits take
//! (`docs/design/distributions.md`).
//!
//! Loss `i` with weight `w`, reporting threshold `r` and value `y`
//! contributes `w (ln f(y) − ln S(r))`, or `w (ln S(y) − ln S(r))` when a
//! policy limit capped it, so each loss is conditioned on having been
//! reported. The log-likelihood is maximized by Nelder–Mead over the
//! logarithms of the positive parameters, and the standard errors come
//! from a central-difference Hessian in the natural parameters.

use prospicio_core::{Error, Result};
use prospicio_math::linalg::solve;
use prospicio_math::optimize::{NelderMead, nelder_mead};
use prospicio_math::special::{gamma_inc, ln_gamma, mills_ratio, norm_cdf};

use crate::dist::Dist;
use crate::{Burr, Gamma, LargeLosses, Lognormal, Weibull};

/// A family [`fit_severity`] can fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// `Lognormal(meanlog, sdlog)`.
    Lognormal,
    /// `Gamma(shape, scale)`.
    Gamma,
    /// `Weibull(shape, scale)`.
    Weibull,
    /// `Burr(alpha, gamma, scale)`, `S(x) = (1 + (x/θ)^γ)^(−α)`.
    Burr,
}

impl Family {
    /// The family's name, as in [`Dist::family`].
    pub fn name(self) -> &'static str {
        match self {
            Self::Lognormal => "lognormal",
            Self::Gamma => "gamma",
            Self::Weibull => "weibull",
            Self::Burr => "burr",
        }
    }

    /// The family from its name.
    pub fn from_name(name: &str) -> Result<Self> {
        Ok(match name {
            "lognormal" => Self::Lognormal,
            "gamma" => Self::Gamma,
            "weibull" => Self::Weibull,
            "burr" => Self::Burr,
            other => {
                return Err(Error::Data(format!(
                    "family must be \"lognormal\", \"gamma\", \"weibull\" or \"burr\", not {other:?}"
                )));
            }
        })
    }

    /// The parameters' names, in the order of [`SeverityFit::parameters`].
    pub fn parameter_names(self) -> &'static [&'static str] {
        match self {
            Self::Lognormal => &["meanlog", "sdlog"],
            Self::Gamma | Self::Weibull => &["shape", "scale"],
            Self::Burr => &["alpha", "gamma", "scale"],
        }
    }

    /// `(ln f(x), ln S(x))` at the natural parameters `p`.
    fn ln_f_s(self, p: &[f64], x: f64) -> (f64, f64) {
        match self {
            Self::Lognormal => {
                let (mu, sd) = (p[0], p[1]);
                let z = (x.ln() - mu) / sd;
                let ln_f =
                    -x.ln() - sd.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5 * z * z;
                (ln_f, ln_norm_survival(z))
            }
            Self::Gamma => {
                let (k, theta) = (p[0], p[1]);
                let u = x / theta;
                let ln_f = (k - 1.0) * x.ln() - u - ln_gamma(k) - k * theta.ln();
                (ln_f, gamma_inc(k, u).1.ln())
            }
            Self::Weibull => {
                let (k, lambda) = (p[0], p[1]);
                let v = (x / lambda).powf(k);
                let ln_f = k.ln() - lambda.ln() + (k - 1.0) * (x / lambda).ln() - v;
                (ln_f, -v)
            }
            Self::Burr => {
                let (alpha, gamma, theta) = (p[0], p[1], p[2]);
                let ln_v = gamma * (x / theta).ln();
                let ln_1pv = ln_1p_exp(ln_v);
                let ln_f = alpha.ln() + gamma.ln() + (gamma - 1.0) * (x / theta).ln()
                    - theta.ln()
                    - (alpha + 1.0) * ln_1pv;
                (ln_f, -alpha * ln_1pv)
            }
        }
    }

    /// The natural parameters from the unconstrained ones: the logarithm
    /// of every positive parameter (the lognormal's `meanlog` as is).
    fn natural(self, u: &[f64]) -> Vec<f64> {
        match self {
            Self::Lognormal => vec![u[0], u[1].exp()],
            _ => u.iter().map(|v| v.exp()).collect(),
        }
    }

    fn unconstrained(self, p: &[f64]) -> Vec<f64> {
        match self {
            Self::Lognormal => vec![p[0], p[1].ln()],
            _ => p.iter().map(|v| v.ln()).collect(),
        }
    }

    fn dist(self, p: &[f64]) -> Result<Dist> {
        Ok(match self {
            Self::Lognormal => Lognormal::new(p[0], p[1])?.into(),
            Self::Gamma => Gamma::new(p[0], p[1])?.into(),
            Self::Weibull => Weibull::new(p[0], p[1])?.into(),
            Self::Burr => Burr::new(p[0], p[1], p[2])?.into(),
        })
    }
}

/// `ln(1 − Φ(z))`, through the Mills ratio far in the upper tail.
fn ln_norm_survival(z: f64) -> f64 {
    if z < 5.0 {
        norm_cdf(-z).ln()
    } else {
        -0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln() + mills_ratio(z).ln()
    }
}

/// `ln(1 + e^t)` without overflow.
fn ln_1p_exp(t: f64) -> f64 {
    if t > 35.0 {
        t + (-t).exp()
    } else {
        t.exp().ln_1p()
    }
}

/// A fitted severity: the distribution, its parameters with their standard
/// errors and covariance, and the fit's log-likelihood.
#[derive(Debug, Clone)]
pub struct SeverityFit {
    family: Family,
    dist: Dist,
    parameters: Vec<f64>,
    covariance: Vec<f64>,
    log_likelihood: f64,
    weight: f64,
    converged: bool,
}

impl SeverityFit {
    /// The fitted family.
    pub fn family(&self) -> Family {
        self.family
    }

    /// The fitted distribution.
    pub fn dist(&self) -> &Dist {
        &self.dist
    }

    /// The parameters, named by [`Family::parameter_names`].
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// The covariance of the parameters, row-major, from the inverse of
    /// the observed information; NaN where the information is singular.
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Each parameter's standard error, the root of the covariance's
    /// diagonal.
    pub fn standard_errors(&self) -> Vec<f64> {
        let n = self.parameters.len();
        (0..n).map(|i| self.covariance[i * n + i].sqrt()).collect()
    }

    /// The maximized log-likelihood, each loss conditioned on its
    /// reporting threshold.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Akaike's information criterion, `2 k − 2 ℓ`.
    pub fn aic(&self) -> f64 {
        2.0 * self.parameters.len() as f64 - 2.0 * self.log_likelihood
    }

    /// The Bayesian information criterion, `k ln n − 2 ℓ`, with `n` the
    /// total weight of the losses.
    pub fn bic(&self) -> f64 {
        self.parameters.len() as f64 * self.weight.ln() - 2.0 * self.log_likelihood
    }

    /// Whether the simplex met its tolerance.
    pub fn converged(&self) -> bool {
        self.converged
    }
}

/// The log-likelihood of `data` under `family` at the natural parameters
/// `p`: `Σ w (ln f(y) − ln S(r))`, with `ln S(y)` for a censored loss.
pub fn log_likelihood(family: Family, p: &[f64], data: &LargeLosses) -> f64 {
    data.rows()
        .map(|(y, r, censored, w)| {
            let (ln_f, ln_s) = family.ln_f_s(p, y);
            let own = if censored { ln_s } else { ln_f };
            let reach = if r > 0.0 { family.ln_f_s(p, r).1 } else { 0.0 };
            w * (own - reach)
        })
        .sum()
}

/// Maximum likelihood fit of `family` to `data`: each loss conditioned on
/// exceeding its reporting threshold, a censored loss known only to be at
/// least its value.
///
/// The simplex starts from moments of the uncensored losses (the Burr from
/// the Weibull fit and two heavier tails, keeping the best), and the
/// standard errors come from the observed information. A Burr fit on
/// little data can drift towards a boundary where it mimics a Weibull
/// (`α → ∞`) or a Pareto (`γ → 1` with a large scale): check
/// [`SeverityFit::converged`] and the standard errors.
///
/// ```
/// use prospicio_prob::severity_fit::{Family, fit_severity};
/// use prospicio_prob::LargeLosses;
///
/// let y = vec![120.0, 300.0, 450.0, 800.0, 1500.0, 2600.0, 4000.0];
/// let fit = fit_severity(Family::Lognormal, &LargeLosses::new(y.clone()).unwrap()).unwrap();
/// // Untruncated and uncensored: the mean and the n-denominator sd of the logs.
/// let mu = y.iter().map(|v| v.ln()).sum::<f64>() / 7.0;
/// assert!((fit.parameters()[0] - mu).abs() < 1e-6);
/// ```
pub fn fit_severity(family: Family, data: &LargeLosses) -> Result<SeverityFit> {
    let uncensored: Vec<(f64, f64)> = data
        .rows()
        .filter(|r| !r.2)
        .map(|(y, _, _, w)| (y, w))
        .collect();
    let weight: f64 = data.rows().map(|r| r.3).sum();
    let w_unc: f64 = uncensored.iter().map(|u| u.1).sum();
    if uncensored.len() < 2 {
        return Err(Error::Data(
            "need at least two uncensored losses to fit a severity".into(),
        ));
    }
    let moments = |g: &dyn Fn(f64) -> f64| {
        let m = uncensored.iter().map(|&(y, w)| w * g(y)).sum::<f64>() / w_unc;
        let v = uncensored
            .iter()
            .map(|&(y, w)| w * (g(y) - m).powi(2))
            .sum::<f64>()
            / w_unc;
        (m, v.max(1e-12))
    };
    let (lm, lv) = moments(&|y| y.ln());
    let (m, v) = moments(&|y| y);
    let sd = lv.sqrt().max(0.05);
    let weibull_start = {
        let k = (std::f64::consts::PI / (sd * 6f64.sqrt())).clamp(0.05, 20.0);
        vec![k, (lm + 0.577_215_664_901_532_9 / k).exp()]
    };
    let starts: Vec<Vec<f64>> = match family {
        Family::Lognormal => vec![vec![lm, sd]],
        Family::Gamma => vec![vec![(m * m / v).clamp(1e-3, 1e3), v / m]],
        Family::Weibull => vec![weibull_start],
        Family::Burr => {
            let wb = fit_severity(Family::Weibull, data)
                .map(|f| f.parameters)
                .unwrap_or(weibull_start);
            let median = lm.exp();
            vec![
                vec![5.0, wb[0], wb[1] * 5f64.powf(1.0 / wb[0])],
                vec![1.0, wb[0], median],
                vec![2.0, 1.0, median],
            ]
        }
    };
    let options = NelderMead {
        max_iterations: 40_000,
        ..NelderMead::default()
    };
    let objective = |u: &[f64]| {
        if u.iter().any(|v| v.abs() > 700.0) {
            return f64::NAN;
        }
        -log_likelihood(family, &family.natural(u), data)
    };
    let best = starts
        .iter()
        .map(|s| nelder_mead(objective, &family.unconstrained(s), &options))
        .min_by(|a, b| a.value.total_cmp(&b.value))
        .expect("at least one start");
    if !best.value.is_finite() {
        return Err(Error::Data(format!(
            "no {} fits these losses: the likelihood is not finite at any point tried",
            family.name()
        )));
    }
    let parameters = family.natural(&best.x);
    let dist = family.dist(&parameters)?;
    let covariance = covariance(family, &parameters, data);
    Ok(SeverityFit {
        family,
        dist,
        parameters,
        covariance,
        log_likelihood: -best.value,
        weight,
        converged: best.converged,
    })
}

/// The inverse of the observed information, `(−∇²ℓ)⁻¹`, by central
/// differences in the natural parameters; NaN if it is not invertible.
fn covariance(family: Family, p: &[f64], data: &LargeLosses) -> Vec<f64> {
    let n = p.len();
    let ll = |q: &[f64]| log_likelihood(family, q, data);
    let h: Vec<f64> = p.iter().map(|v| 1e-4 * v.abs().max(1e-3)).collect();
    let at = |di: &[(usize, f64)]| {
        let mut q = p.to_vec();
        for &(i, d) in di {
            q[i] += d;
        }
        ll(&q)
    };
    let f0 = ll(p);
    let mut info = vec![0.0; n * n];
    for i in 0..n {
        let d2 = (at(&[(i, h[i])]) - 2.0 * f0 + at(&[(i, -h[i])])) / (h[i] * h[i]);
        info[i * n + i] = -d2;
        for j in 0..i {
            let d = (at(&[(i, h[i]), (j, h[j])])
                - at(&[(i, h[i]), (j, -h[j])])
                - at(&[(i, -h[i]), (j, h[j])])
                + at(&[(i, -h[i]), (j, -h[j])]))
                / (4.0 * h[i] * h[j]);
            info[i * n + j] = -d;
            info[j * n + i] = -d;
        }
    }
    let mut cov = vec![f64::NAN; n * n];
    for c in 0..n {
        let mut e = vec![0.0; n];
        e[c] = 1.0;
        match solve(info.clone(), e, n) {
            Some(col) if col.iter().all(|v| v.is_finite()) => {
                for r in 0..n {
                    cov[r * n + c] = col[r];
                }
            }
            _ => return vec![f64::NAN; n * n],
        }
    }
    cov
}

/// One step of the [`product_limit`] estimator, at a value where an
/// uncensored loss lies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProductLimitStep {
    /// The value.
    pub x: f64,
    /// The weight at risk: losses reported below `x` and not below it,
    /// `Σ w` over `r < x ≤ y`.
    pub at_risk: f64,
    /// The uncensored weight at `x`.
    pub events: f64,
    /// `Ŝ(x)`, the estimate of `P(X > x | X > τ)` with `τ` the lowest
    /// reporting threshold.
    pub survival: f64,
}

/// The product-limit (Kaplan–Meier) estimator for left-truncated,
/// right-censored losses: at each value `x` where an uncensored loss lies,
/// `Ŝ(x) = Π (1 − d/n)` with `d` the uncensored weight at `x` and `n` the
/// weight at risk, the losses with `r < x ≤ y` (Tsai, Jewell and Wang,
/// 1987). It estimates the survival conditional on the lowest reporting
/// threshold `τ`, `S(x)/S(τ)`; without truncation or censoring it is the
/// empirical survival function. A step whose risk set is a single loss
/// can drop the estimate to 0 early; read the tail with `at_risk`.
///
/// ```
/// use prospicio_prob::LargeLosses;
/// use prospicio_prob::severity_fit::product_limit;
///
/// let data = LargeLosses::new(vec![100.0, 200.0, 300.0, 400.0]).unwrap();
/// let s: Vec<f64> = product_limit(&data).iter().map(|p| p.survival).collect();
/// assert_eq!(s, vec![0.75, 0.5, 0.25, 0.0]);
/// ```
pub fn product_limit(data: &LargeLosses) -> Vec<ProductLimitStep> {
    let rows: Vec<_> = data.rows().collect();
    let mut xs: Vec<f64> = rows.iter().filter(|r| !r.2).map(|r| r.0).collect();
    xs.sort_by(f64::total_cmp);
    xs.dedup();
    let mut survival = 1.0;
    xs.into_iter()
        .map(|x| {
            let (mut at_risk, mut events) = (0.0, 0.0);
            for &(y, r, censored, w) in &rows {
                if r < x && x <= y {
                    at_risk += w;
                }
                if !censored && y == x {
                    events += w;
                }
            }
            if at_risk > 0.0 {
                survival *= 1.0 - events / at_risk;
            }
            ProductLimitStep {
                x,
                at_risk,
                events,
                survival,
            }
        })
        .collect()
}

/// A fit against the [`product_limit`] estimate at each of its steps.
#[derive(Debug, Clone, PartialEq)]
pub struct GoodnessOfFit {
    /// The product-limit steps.
    pub steps: Vec<ProductLimitStep>,
    /// The model's survival conditional on the lowest reporting threshold
    /// `τ`, `S(x)/S(τ)`, at each step: the P-P comparison with
    /// `steps[i].survival`.
    pub model_survival: Vec<f64>,
    /// The model's conditional quantile at the middle of each step,
    /// `Q(1 − Ŝ_mid S(τ))` with `Ŝ_mid` halfway between the estimate before
    /// and after the step: the Q-Q comparison with `steps[i].x`.
    pub model_quantile: Vec<f64>,
    /// `max |Ŝ − S/S(τ)|` over the steps, on both sides of each jump: a
    /// Kolmogorov–Smirnov distance (its null distribution is not the
    /// classical one once the parameters are fitted or the data truncated).
    pub max_distance: f64,
}

/// Compares `dist` with the [`product_limit`] estimate of `data`, each loss
/// conditioned on its reporting threshold: P-P (survival), Q-Q (quantiles
/// at the steps' middles) and the largest distance between the two
/// survival functions. Works for any severity, a [`fit_severity`] result
/// or a Pareto fit.
///
/// ```
/// use prospicio_prob::severity_fit::{Family, fit_severity, goodness_of_fit};
/// use prospicio_prob::LargeLosses;
///
/// let data = LargeLosses::new(vec![120.0, 300.0, 450.0, 800.0, 1500.0, 2600.0]).unwrap();
/// let fit = fit_severity(Family::Lognormal, &data).unwrap();
/// let gof = goodness_of_fit(fit.dist(), &data).unwrap();
/// assert_eq!(gof.steps.len(), 6);
/// assert!(gof.max_distance < 0.3);
/// ```
///
/// # Errors
///
/// When `dist` gives the lowest reporting threshold no probability.
pub fn goodness_of_fit<D: crate::Distribution + ?Sized>(
    dist: &D,
    data: &LargeLosses,
) -> Result<GoodnessOfFit> {
    let tau = data.rows().map(|r| r.1).fold(f64::INFINITY, f64::min);
    let s_tau = if tau > 0.0 { dist.survival(tau) } else { 1.0 };
    if s_tau.is_nan() || s_tau <= 0.0 {
        return Err(Error::Data(format!(
            "the distribution gives no probability above the lowest reporting threshold {tau}"
        )));
    }
    let steps = product_limit(data);
    let (mut before, mut max_distance) = (1.0, 0.0f64);
    let mut model_survival = Vec::with_capacity(steps.len());
    let mut model_quantile = Vec::with_capacity(steps.len());
    for st in &steps {
        let m = dist.survival(st.x) / s_tau;
        max_distance = max_distance
            .max((st.survival - m).abs())
            .max((before - m).abs());
        model_survival.push(m);
        let mid = 0.5 * (before + st.survival);
        model_quantile.push(
            dist.quantile((1.0 - mid * s_tau).clamp(0.0, 1.0))
                .unwrap_or(f64::NAN),
        );
        before = st.survival;
    }
    Ok(GoodnessOfFit {
        steps,
        model_survival,
        model_quantile,
        max_distance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Distribution;
    use prospicio_core::StreamRng;
    use prospicio_math::special::digamma;

    #[test]
    fn untruncated_fits_match_their_closed_forms() {
        let y = vec![
            120.0, 300.0, 450.0, 800.0, 1500.0, 2600.0, 4000.0, 650.0, 90.0, 2100.0,
        ];
        let data = LargeLosses::new(y.clone()).unwrap();
        let n = y.len() as f64;
        // Lognormal: the mean and n-denominator sd of the logs, and
        // standard errors sd/sqrt(n), sd/sqrt(2n).
        let fit = fit_severity(Family::Lognormal, &data).unwrap();
        let mu = y.iter().map(|v| v.ln()).sum::<f64>() / n;
        let sd = (y.iter().map(|v| (v.ln() - mu).powi(2)).sum::<f64>() / n).sqrt();
        assert!(
            (fit.parameters()[0] - mu).abs() < 1e-7,
            "{:?}",
            fit.parameters()
        );
        assert!((fit.parameters()[1] / sd - 1.0).abs() < 1e-7);
        let se = fit.standard_errors();
        assert!((se[0] / (sd / n.sqrt()) - 1.0).abs() < 1e-4, "{se:?}");
        assert!(
            (se[1] / (sd / (2.0 * n).sqrt()) - 1.0).abs() < 1e-4,
            "{se:?}"
        );
        // Gamma: ln k − ψ(k) = ln ȳ − mean(ln y), and θ = ȳ / k.
        let fit = fit_severity(Family::Gamma, &data).unwrap();
        let (k, theta) = (fit.parameters()[0], fit.parameters()[1]);
        let ybar = y.iter().sum::<f64>() / n;
        assert!((k.ln() - digamma(k) - (ybar.ln() - mu)).abs() < 1e-8);
        assert!((theta * k / ybar - 1.0).abs() < 1e-7);
        // Weibull: the profile equation in k, and λ^k = mean(y^k).
        let fit = fit_severity(Family::Weibull, &data).unwrap();
        let (k, lambda) = (fit.parameters()[0], fit.parameters()[1]);
        let s0: f64 = y.iter().map(|v| v.powf(k)).sum();
        let s1: f64 = y.iter().map(|v| v.powf(k) * v.ln()).sum();
        assert!((s1 / s0 - 1.0 / k - mu).abs() < 1e-7);
        assert!((lambda.powf(k) / (s0 / n) - 1.0).abs() < 1e-7);
        assert!(fit.aic() > 0.0 && fit.converged());
    }

    #[test]
    fn truncated_and_censored_fits_recover_the_truth() {
        let mut rng = StreamRng::new(11, 0);
        for (family, truth) in [
            (Family::Lognormal, vec![7.0, 1.2]),
            (Family::Gamma, vec![1.5, 800.0]),
            (Family::Weibull, vec![0.8, 1000.0]),
            (Family::Burr, vec![1.5, 2.0, 1500.0]),
        ] {
            let d = family.dist(&truth).unwrap();
            let draws = d.sample(&mut rng, 10_000);
            // Reported above 300 for half the policies, capped at 5000.
            let (mut y, mut r, mut c) = (vec![], vec![], vec![]);
            for (i, x) in draws.into_iter().enumerate() {
                let threshold = if i % 2 == 0 { 300.0 } else { 0.0 };
                if x <= threshold {
                    continue;
                }
                y.push(x.min(5000.0));
                r.push(threshold);
                c.push(x > 5000.0);
            }
            let data = LargeLosses::new(y)
                .unwrap()
                .reporting_thresholds(r)
                .unwrap()
                .censored(c)
                .unwrap();
            let fit = fit_severity(family, &data).unwrap();
            let se = fit.standard_errors();
            for i in 0..truth.len() {
                let z = (fit.parameters()[i] - truth[i]) / se[i];
                assert!(
                    z.abs() < 4.5,
                    "{family:?} {i}: {:?} {se:?}",
                    fit.parameters()
                );
            }
            // The score is zero at the maximum.
            let p = fit.parameters().to_vec();
            for i in 0..p.len() {
                let h = 1e-6 * p[i].abs();
                let (mut a, mut b) = (p.clone(), p.clone());
                a[i] += h;
                b[i] -= h;
                let g = (log_likelihood(family, &a, &data) - log_likelihood(family, &b, &data))
                    / (2.0 * h);
                assert!((g * se[i]).abs() < 1e-2, "{family:?} score {i}: {g}");
            }
        }
    }

    #[test]
    fn product_limit_handles_truncation_and_censoring() {
        // Weighted, without truncation or censoring: the empirical survival.
        let data = LargeLosses::new(vec![100.0, 200.0, 200.0, 400.0])
            .unwrap()
            .weights(vec![1.0, 2.0, 1.0, 4.0])
            .unwrap();
        let s: Vec<f64> = product_limit(&data).iter().map(|p| p.survival).collect();
        assert_eq!(s, vec![7.0 / 8.0, 0.5, 0.0]);
        // Truncated and censored draws track the true conditional survival.
        let truth = Weibull::new(0.8, 1000.0).unwrap();
        let draws = truth.sample(&mut StreamRng::new(5, 0), 20_000);
        let (mut y, mut r, mut c) = (vec![], vec![], vec![]);
        for (i, x) in draws.into_iter().enumerate() {
            let threshold = [0.0, 200.0, 500.0][i % 3];
            if x <= threshold {
                continue;
            }
            y.push(x.min(4000.0));
            r.push(threshold);
            c.push(x > 4000.0);
        }
        let data = LargeLosses::new(y)
            .unwrap()
            .reporting_thresholds(r)
            .unwrap()
            .censored(c)
            .unwrap();
        let gof = goodness_of_fit(&truth, &data).unwrap();
        assert!(gof.max_distance < 0.02, "{}", gof.max_distance);
        // The Q-Q column runs close to the data where the risk set is large.
        for (st, q) in gof.steps.iter().zip(&gof.model_quantile) {
            if st.at_risk > 1000.0 && st.x > 50.0 {
                assert!((q / st.x - 1.0).abs() < 0.1, "{} {q}", st.x);
            }
        }
        // A fit is at least as close as the truth it came from, roughly.
        let fit = fit_severity(Family::Weibull, &data).unwrap();
        let gof_fit = goodness_of_fit(fit.dist(), &data).unwrap();
        assert!(gof_fit.max_distance < 0.02);
    }

    #[test]
    fn refuses_too_little_data() {
        let data = LargeLosses::new(vec![100.0, 200.0])
            .unwrap()
            .censored(vec![false, true])
            .unwrap();
        assert!(fit_severity(Family::Gamma, &data).is_err());
        assert!(Family::from_name("pareto").is_err());
    }
}
