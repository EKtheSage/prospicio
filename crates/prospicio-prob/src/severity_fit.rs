//! Maximum likelihood fits of the body families (lognormal, gamma,
//! Weibull, loglogistic, exponential, Burr, Lomax) to large losses with
//! reporting thresholds and policy limits: the same [`LargeLosses`] the
//! Pareto fits take (`docs/design/distributions.md`). The single Pareto and
//! Riegel's generalized Pareto, fitted from a threshold `t` by their own
//! estimators, come back as the same [`SeverityFit`] through
//! [`fit_severity_with_threshold`], scored on the same likelihood.
//!
//! Loss `i` with weight `w`, reporting threshold `r` and value `y`
//! contributes `w (ln f(y) − ln S(r))`, or `w (ln S(y) − ln S(r))` when a
//! policy limit capped it, so each loss is conditioned on having been
//! reported. A loss truncated from above at `u` is conditioned on
//! `r < X ≤ u` instead: `ln S(r)` becomes `ln(S(r) − S(u))`, and a censored
//! loss's `ln S(y)` becomes `ln(S(y) − S(u))`. The log-likelihood is maximized by Nelder–Mead over the
//! logarithms of the positive parameters, and the standard errors come
//! from a central-difference Hessian in the natural parameters.

use prospicio_core::{Error, Result};
use prospicio_math::linalg::solve;
use prospicio_math::optimize::{NelderMead, nelder_mead};
use prospicio_math::special::{gamma_inc, ln_gamma, mills_ratio, norm_cdf};

use crate::dist::Dist;
use crate::evt::Gpd;
use crate::{Burr, Exponential, Gamma, LargeLosses, Loglogistic, Lognormal, Pareto, Weibull};

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
    /// `Lomax(alpha, scale)`, `S(x) = (1 + x/θ)^(−α)`: the Burr with
    /// `γ = 1`, Bahnemann's (Monograph 2) Pareto.
    Lomax,
    /// `Loglogistic(shape, scale)`, `S(x) = 1 / (1 + (x/θ)^α)`: the Burr
    /// with its `α` pinned at 1 and `γ` the shape.
    Loglogistic,
    /// `Exponential(mean)`, `S(x) = exp(−x/θ)`.
    Exponential,
    /// The single Pareto from a threshold `t`, `S(x) = (x/t)^(−α)` for
    /// `x ≥ t`: parameters `t` (pinned, not counted) and `alpha`. Fitted
    /// by [`Pareto::fit`] through [`fit_severity_with_threshold`].
    Pareto,
    /// Riegel's generalized Pareto from a threshold `t`,
    /// `S(x) = (1 + (α_ini/α_tail)(x/t − 1))^(−α_tail)` for `x ≥ t`:
    /// parameters `t` (pinned, not counted), `alpha_ini` and `alpha_tail`.
    /// It is a GPD at location `t` with `ξ = 1/α_tail` and
    /// `β = t/α_ini`. Fitted by [`Gpd::fit_riegel`] through
    /// [`fit_severity_with_threshold`].
    GenPareto,
}

impl Family {
    /// Every family, in the order a comparison lists them: the body
    /// families, then the two fitted from a threshold.
    pub const ALL: [Family; 9] = [
        Family::Lognormal,
        Family::Gamma,
        Family::Weibull,
        Family::Loglogistic,
        Family::Exponential,
        Family::Burr,
        Family::Lomax,
        Family::Pareto,
        Family::GenPareto,
    ];

    /// The family's name. It differs from [`Dist::family`] for the Lomax
    /// (fitted as a Burr with `γ = 1`, so its distribution is a `"burr"`)
    /// and for `"gen_pareto"` (a `"generalized_pareto"`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Lognormal => "lognormal",
            Self::Gamma => "gamma",
            Self::Weibull => "weibull",
            Self::Burr => "burr",
            Self::Lomax => "lomax",
            Self::Loglogistic => "loglogistic",
            Self::Exponential => "exponential",
            Self::Pareto => "pareto",
            Self::GenPareto => "gen_pareto",
        }
    }

    /// The family from its name.
    pub fn from_name(name: &str) -> Result<Self> {
        Ok(match name {
            "lognormal" => Self::Lognormal,
            "gamma" => Self::Gamma,
            "weibull" => Self::Weibull,
            "burr" => Self::Burr,
            "lomax" => Self::Lomax,
            "loglogistic" => Self::Loglogistic,
            "exponential" => Self::Exponential,
            "pareto" => Self::Pareto,
            "gen_pareto" => Self::GenPareto,
            other => {
                return Err(Error::Data(format!(
                    "family must be \"lognormal\", \"gamma\", \"weibull\", \"loglogistic\", \
                     \"exponential\", \"burr\", \"lomax\", \"pareto\" or \"gen_pareto\", not {other:?}"
                )));
            }
        })
    }

    /// The parameters' names, in the order of [`SeverityFit::parameters`].
    /// The Pareto families start with their pinned threshold `t`.
    pub fn parameter_names(self) -> &'static [&'static str] {
        match self {
            Self::Lognormal => &["meanlog", "sdlog"],
            Self::Gamma | Self::Weibull | Self::Loglogistic => &["shape", "scale"],
            Self::Burr => &["alpha", "gamma", "scale"],
            Self::Lomax => &["alpha", "scale"],
            Self::Exponential => &["mean"],
            Self::Pareto => &["t", "alpha"],
            Self::GenPareto => &["t", "alpha_ini", "alpha_tail"],
        }
    }

    /// Whether the family is fitted from a threshold `t` (the Pareto and
    /// the generalized Pareto), which [`fit_severity_with_threshold`]
    /// takes.
    pub fn needs_threshold(self) -> bool {
        matches!(self, Self::Pareto | Self::GenPareto)
    }

    /// The number of estimated parameters, the `k` of AIC and BIC: a
    /// pinned threshold `t` is not counted.
    pub fn estimated_parameters(self) -> usize {
        self.parameter_names().len() - usize::from(self.needs_threshold())
    }

    /// `(ln f(x), ln S(x))` at the natural parameters `p`.
    pub(crate) fn ln_f_s(self, p: &[f64], x: f64) -> (f64, f64) {
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
            Self::Lomax => {
                let (alpha, theta) = (p[0], p[1]);
                let ln_1pv = (x / theta).ln_1p();
                (
                    alpha.ln() - theta.ln() - (alpha + 1.0) * ln_1pv,
                    -alpha * ln_1pv,
                )
            }
            Self::Loglogistic => {
                let (k, theta) = (p[0], p[1]);
                let ln_z = (x / theta).ln();
                let ln_1pv = ln_1p_exp(k * ln_z);
                (
                    k.ln() - theta.ln() + (k - 1.0) * ln_z - 2.0 * ln_1pv,
                    -ln_1pv,
                )
            }
            Self::Exponential => (-p[0].ln() - x / p[0], -x / p[0]),
            // Below the threshold the density is 0 and the survival 1, so
            // a reporting threshold under t conditions on t.
            Self::Pareto => {
                let (t, alpha) = (p[0], p[1]);
                if x < t {
                    return (f64::NEG_INFINITY, 0.0);
                }
                let ln_s = -alpha * (x / t).ln();
                (alpha.ln() - x.ln() + ln_s, ln_s)
            }
            Self::GenPareto => {
                let (t, a_ini, a_tail) = (p[0], p[1], p[2]);
                if x < t {
                    return (f64::NEG_INFINITY, 0.0);
                }
                let ln_1pv = (a_ini / a_tail * (x / t - 1.0)).ln_1p();
                ((a_ini / t).ln() - (a_tail + 1.0) * ln_1pv, -a_tail * ln_1pv)
            }
        }
    }

    /// The natural parameters from the unconstrained ones: the logarithm
    /// of every positive parameter (the lognormal's `meanlog` as is).
    /// The threshold of a Pareto family stays as it is.
    pub(crate) fn natural(self, u: &[f64]) -> Vec<f64> {
        let keep = usize::from(self == Self::Lognormal || self.needs_threshold());
        u.iter()
            .enumerate()
            .map(|(i, v)| if i < keep { *v } else { v.exp() })
            .collect()
    }

    pub(crate) fn unconstrained(self, p: &[f64]) -> Vec<f64> {
        let keep = usize::from(self == Self::Lognormal || self.needs_threshold());
        p.iter()
            .enumerate()
            .map(|(i, v)| if i < keep { *v } else { v.ln() })
            .collect()
    }

    pub(crate) fn dist(self, p: &[f64]) -> Result<Dist> {
        Ok(match self {
            Self::Lognormal => Lognormal::new(p[0], p[1])?.into(),
            Self::Gamma => Gamma::new(p[0], p[1])?.into(),
            Self::Weibull => Weibull::new(p[0], p[1])?.into(),
            Self::Burr => Burr::new(p[0], p[1], p[2])?.into(),
            Self::Lomax => Burr::new(p[0], 1.0, p[1])?.into(),
            Self::Loglogistic => Loglogistic::new(p[0], p[1])?.into(),
            Self::Exponential => Exponential::new(p[0])?.into(),
            Self::Pareto => Pareto::new(p[0], p[1])?.into(),
            Self::GenPareto => Gpd::riegel(p[0], p[1], p[2])?.into(),
        })
    }

    /// Refuses a Pareto family where only the body families make sense.
    pub(crate) fn body_only(self, what: &str) -> Result<()> {
        if self.needs_threshold() {
            return Err(Error::Data(format!(
                "{what} takes the body families, not {:?}, which needs a threshold",
                self.name()
            )));
        }
        Ok(())
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

/// `ln(S(a) − S(u))` from `ln S(a)` and `ln S(u)`, without cancelling
/// when both are tiny: `ln S(a) + ln(1 − S(u) / S(a))`.
fn ln_s_between(ln_s_a: f64, ln_s_u: f64) -> f64 {
    if ln_s_u == f64::NEG_INFINITY {
        ln_s_a
    } else {
        ln_s_a + (-(ln_s_u - ln_s_a).exp()).ln_1p()
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

    /// Akaike's information criterion, `2 k − 2 ℓ`, with `k` the
    /// [estimated parameters](Family::estimated_parameters): a Pareto
    /// family's pinned threshold is not counted.
    pub fn aic(&self) -> f64 {
        2.0 * self.family.estimated_parameters() as f64 - 2.0 * self.log_likelihood
    }

    /// The Bayesian information criterion, `k ln n − 2 ℓ`, with `n` the
    /// total weight of the losses.
    pub fn bic(&self) -> f64 {
        self.family.estimated_parameters() as f64 * self.weight.ln() - 2.0 * self.log_likelihood
    }

    /// The threshold `t` a Pareto family was fitted from, `None` for a
    /// body family.
    pub fn threshold(&self) -> Option<f64> {
        self.family.needs_threshold().then(|| self.parameters[0])
    }

    /// Whether the simplex met its tolerance.
    pub fn converged(&self) -> bool {
        self.converged
    }
}

/// The log-likelihood of `data` under `family` at the natural parameters
/// `p`: `Σ w (ln f(y) − ln S(r))`, with `ln S(y)` for a censored loss, and
/// `S(·) − S(u)` in place of each `S(·)` for a loss truncated from above
/// at `u`.
///
/// A Pareto family's survival is 1 below its threshold `t`, so each loss
/// is conditioned on `max(r, t)` where a body family conditions on `r`:
/// the two likelihoods, and so their AICs, are comparable only when every
/// reporting threshold is at least `t` (a loss below `t` makes the Pareto
/// likelihood `−∞`).
pub fn log_likelihood(family: Family, p: &[f64], data: &LargeLosses) -> f64 {
    data.rows()
        .zip(data.upper_bounds())
        .map(|((y, r, censored, w), &u)| {
            let ln_s_u = if u.is_finite() {
                family.ln_f_s(p, u).1
            } else {
                f64::NEG_INFINITY
            };
            let (ln_f, ln_s) = family.ln_f_s(p, y);
            let own = if censored {
                ln_s_between(ln_s, ln_s_u)
            } else {
                ln_f
            };
            let ln_s_r = if r > 0.0 { family.ln_f_s(p, r).1 } else { 0.0 };
            w * (own - ln_s_between(ln_s_r, ln_s_u))
        })
        .sum()
}

/// Maximum likelihood fit of `family` to `data`: each loss conditioned on
/// exceeding its reporting threshold (and on not exceeding its bound from
/// above, [`LargeLosses::truncated_above`]), a censored loss known only to
/// be at least its value.
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
    fit_severity_with_threshold(family, data, None)
}

/// [`fit_severity`] for every [`Family`]: a body family takes `t = None`,
/// a Pareto family ([`Family::needs_threshold`]) its threshold `t`, and the
/// fit comes back with the same log-likelihood, AIC and BIC as a body fit
/// ([`log_likelihood`]), `t` among the parameters but not counted in `k`.
///
/// The Pareto is [`Pareto::fit_with_standard_error`] untruncated; the
/// generalized Pareto is [`Gpd::fit_riegel`], its standard errors from the
/// observed information in `alpha_ini` and `alpha_tail`. Neither takes a
/// truncation from above: data with one is an error. Every loss must be at least `t`, and the
/// likelihood conditions each on `max(r, t)`: compare it with a body fit
/// only when every reporting threshold is at least `t`.
///
/// ```
/// use prospicio_prob::LargeLosses;
/// use prospicio_prob::severity_fit::{Family, fit_severity_with_threshold};
///
/// let y = vec![1500.0, 2500.0, 4000.0, 10_000.0];
/// let data = LargeLosses::new(y).unwrap().reporting_thresholds(vec![1000.0; 4]).unwrap();
/// let fit = fit_severity_with_threshold(Family::Pareto, &data, Some(1000.0)).unwrap();
/// // α = n / Σ ln(y/t), and ℓ = n ln α − (α + 1) Σ ln(y/t) − n ln t.
/// let s: f64 = [1.5f64, 2.5, 4.0, 10.0].iter().map(|v| v.ln()).sum();
/// let alpha = 4.0 / s;
/// assert!((fit.parameters()[1] - alpha).abs() < 1e-12);
/// let ll = 4.0 * alpha.ln() - s * (alpha + 1.0) - 4.0 * 1000f64.ln();
/// assert!((fit.log_likelihood() - ll).abs() < 1e-9);
/// assert_eq!(fit.aic(), 2.0 - 2.0 * fit.log_likelihood());
/// ```
pub fn fit_severity_with_threshold(
    family: Family,
    data: &LargeLosses,
    t: Option<f64>,
) -> Result<SeverityFit> {
    match (family.needs_threshold(), t) {
        (true, Some(_)) if data.is_truncated_above() => {
            return Err(Error::Data(format!(
                "{} takes no truncation from above",
                family.name()
            )));
        }
        (true, Some(t)) => return fit_from_threshold(family, data, t),
        (true, None) => {
            return Err(Error::Data(format!(
                "{} needs a threshold t",
                family.name()
            )));
        }
        (false, Some(_)) => {
            return Err(Error::Data(format!(
                "t is taken only by \"pareto\" and \"gen_pareto\", not {:?}",
                family.name()
            )));
        }
        (false, None) => {}
    }
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
        // The logistic on the logs: sd = π / (k √3), median e^mean.
        Family::Loglogistic => {
            let k = (std::f64::consts::PI / (sd * 3f64.sqrt())).clamp(0.05, 50.0);
            vec![
                vec![k, lm.exp()],
                vec![2.0 * k, lm.exp()],
                vec![0.5 * k, lm.exp()],
            ]
        }
        Family::Exponential => vec![vec![exponential_mean(data, w_unc).unwrap_or(m)]],
        Family::Pareto | Family::GenPareto => unreachable!("fitted from a threshold"),
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
        // The scale that puts the median of the uncensored losses at
        // their own, for a light, a middling and a heavy tail.
        Family::Lomax => [1.2, 2.5, 8.0]
            .iter()
            .map(|&a| vec![a, lm.exp() / (2f64.powf(1.0 / a) - 1.0)])
            .collect(),
    };
    // Without bounds from above the exponential's maximum is in closed form.
    if let (Family::Exponential, false) = (family, data.is_truncated_above()) {
        let parameters = starts[0].clone();
        return finish(family, parameters, data, weight, true, &[0]);
    }
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
    let free: Vec<usize> = (0..parameters.len()).collect();
    finish(family, parameters, data, weight, best.converged, &free)
}

/// The exponential's maximum likelihood mean with reporting thresholds and
/// censoring, `Σ w (y − r) / Σ_uncensored w` (memorylessness), or `None`
/// when no loss exceeds its threshold.
fn exponential_mean(data: &LargeLosses, w_unc: f64) -> Option<f64> {
    let exposure: f64 = data.rows().map(|(y, r, _, w)| w * (y - r)).sum();
    (exposure > 0.0 && w_unc > 0.0).then(|| exposure / w_unc)
}

/// A fit from its parameters: the distribution, the likelihood at them and
/// the covariance of the parameters `free` (NaN for the others).
fn finish(
    family: Family,
    parameters: Vec<f64>,
    data: &LargeLosses,
    weight: f64,
    converged: bool,
    free: &[usize],
) -> Result<SeverityFit> {
    let dist = family.dist(&parameters)?;
    let log_likelihood = log_likelihood(family, &parameters, data);
    if !log_likelihood.is_finite() {
        return Err(Error::Data(format!(
            "no {} fits these losses: the likelihood is not finite at the estimate",
            family.name()
        )));
    }
    let covariance = covariance(family, &parameters, data, free);
    Ok(SeverityFit {
        family,
        dist,
        parameters,
        covariance,
        log_likelihood,
        weight,
        converged,
    })
}

/// A Pareto family fitted from `t` by its own estimator.
fn fit_from_threshold(family: Family, data: &LargeLosses, t: f64) -> Result<SeverityFit> {
    let weight: f64 = data.rows().map(|r| r.3).sum();
    match family {
        Family::Pareto => {
            let (p, se) = Pareto::fit_with_standard_error(t, data, None)?;
            let mut fit = finish(family, vec![t, p.alpha()], data, weight, true, &[])?;
            fit.covariance[3] = se * se;
            Ok(fit)
        }
        Family::GenPareto => {
            let g = Gpd::fit_riegel(t, data)?;
            // Riegel's form: ξ = 1/α_tail, β = t/α_ini.
            let parameters = vec![t, t / g.beta(), 1.0 / g.xi()];
            finish(family, parameters, data, weight, true, &[1, 2])
        }
        _ => unreachable!("only the Pareto families need a threshold"),
    }
}

/// The inverse of the observed information in the parameters `free`,
/// `(−∇²ℓ)⁻¹`, by central differences in the natural parameters; NaN if it
/// is not invertible, and in the rows and columns of the other parameters.
fn covariance(family: Family, p: &[f64], data: &LargeLosses, free: &[usize]) -> Vec<f64> {
    let full = p.len();
    let n = free.len();
    let ll = |q: &[f64]| log_likelihood(family, q, data);
    let h: Vec<f64> = free.iter().map(|&i| 1e-4 * p[i].abs().max(1e-3)).collect();
    let at = |di: &[(usize, f64)]| {
        let mut q = p.to_vec();
        for &(i, d) in di {
            q[free[i]] += d;
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
    let mut cov = vec![f64::NAN; full * full];
    for c in 0..n {
        let mut e = vec![0.0; n];
        e[c] = 1.0;
        match solve(info.clone(), e, n) {
            Some(col) if col.iter().all(|v| v.is_finite()) => {
                for r in 0..n {
                    cov[free[r] * full + free[c]] = col[r];
                }
            }
            _ => return vec![f64::NAN; full * full],
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
/// When every loss is truncated from above at one common bound `U`, the
/// same estimator estimates `P(X > x | τ < X ≤ U)`, which
/// [`goodness_of_fit`] then compares with the model. Bounds that differ
/// from loss to loss bias it (their estimator is Efron and Petrosian's,
/// not built), so [`goodness_of_fit`] refuses them.
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
    /// `steps[i].survival`. With a bound `U` from above,
    /// `(S(x) − S(U)) / (S(τ) − S(U))`.
    pub model_survival: Vec<f64>,
    /// The model's conditional quantile at the middle of each step,
    /// `Q(1 − Ŝ_mid S(τ))` with `Ŝ_mid` halfway between the estimate before
    /// and after the step (`Q(1 − S(U) − Ŝ_mid (S(τ) − S(U)))` with a bound
    /// from above): the Q-Q comparison with `steps[i].x`.
    pub model_quantile: Vec<f64>,
    /// `max |Ŝ − S/S(τ)|` over the steps, on both sides of each jump: a
    /// Kolmogorov–Smirnov distance (its null distribution is not the
    /// classical one once the parameters are fitted or the data truncated).
    pub max_distance: f64,
    /// The classical two-sided Kolmogorov–Smirnov p-value of
    /// `max_distance` among `n` losses, `n` the total weight rounded
    /// (`prospicio_math::kolmogorov::kolmogorov_sf`, SciPy's `kstwo.sf`):
    /// exact where it holds, so with one threshold and no censoring it is
    /// `scipy.stats.kstest`'s. It is optimistic for a fit: with the
    /// parameters estimated from the same losses the distance is smaller
    /// than under a hypothesis fixed in advance, so the true p-value is
    /// smaller; with censoring the null distribution is not this one
    /// either.
    pub ks_p_value: f64,
}

/// Compares `dist` with the [`product_limit`] estimate of `data`, each loss
/// conditioned on its reporting threshold (and on a bound from above
/// common to every loss, if there is one): P-P (survival), Q-Q (quantiles
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
/// When `dist` gives no probability between the lowest reporting
/// threshold and the bound from above, or when the losses' bounds from
/// above differ.
pub fn goodness_of_fit<D: crate::Distribution + ?Sized>(
    dist: &D,
    data: &LargeLosses,
) -> Result<GoodnessOfFit> {
    let tau = data.rows().map(|r| r.1).fold(f64::INFINITY, f64::min);
    let bounds = data.upper_bounds();
    let u = bounds[0];
    if bounds.iter().any(|&b| b != u) {
        return Err(Error::Data(
            "goodness_of_fit needs one bound from above for every loss: the \
             product-limit estimate is biased when they differ"
                .into(),
        ));
    }
    let s_u = if u.is_finite() { dist.survival(u) } else { 0.0 };
    let s_tau = if tau > 0.0 { dist.survival(tau) } else { 1.0 };
    // The model's survival conditional on τ < X ≤ U is (S(x) − S(U)) / mass.
    let mass = s_tau - s_u;
    if mass.is_nan() || mass <= 0.0 {
        return Err(Error::Data(format!(
            "the distribution gives no probability above the lowest reporting threshold {tau}"
        )));
    }
    let steps = product_limit(data);
    let (mut before, mut max_distance) = (1.0, 0.0f64);
    let mut model_survival = Vec::with_capacity(steps.len());
    let mut model_quantile = Vec::with_capacity(steps.len());
    for st in &steps {
        let m = (dist.survival(st.x) - s_u) / mass;
        max_distance = max_distance
            .max((st.survival - m).abs())
            .max((before - m).abs());
        model_survival.push(m);
        let mid = 0.5 * (before + st.survival);
        model_quantile.push(
            dist.quantile((1.0 - s_u - mid * mass).clamp(0.0, 1.0))
                .unwrap_or(f64::NAN),
        );
        before = st.survival;
    }
    let n = data.rows().map(|r| r.3).sum::<f64>().round().max(1.0) as usize;
    Ok(GoodnessOfFit {
        steps,
        model_survival,
        model_quantile,
        max_distance,
        ks_p_value: prospicio_math::kolmogorov::kolmogorov_sf(n, max_distance),
    })
}

/// Every family in `families` fitted to `data`, in `families`' order, each
/// with its fit or the reason it failed. A Pareto family
/// ([`Family::needs_threshold`]) is fitted from `t`; the body families
/// ignore it. When a Pareto family is named, every reporting threshold must
/// be at least `t`, so that every likelihood conditions each loss on the
/// same threshold and the AICs compare ([`fit_severity_with_threshold`]),
/// and the losses must not be truncated from above.
/// [`rank_by_aic`] then orders the fits.
///
/// ```
/// use prospicio_prob::LargeLosses;
/// use prospicio_prob::severity_fit::{Family, compare_fits, rank_by_aic};
///
/// let y = vec![1500.0, 2500.0, 4000.0, 10_000.0, 1200.0, 3100.0];
/// let data = LargeLosses::new(y).unwrap().reporting_thresholds(vec![1000.0; 6]).unwrap();
/// let mut fits = compare_fits(&data, &Family::ALL, Some(1000.0)).unwrap();
/// rank_by_aic(&mut fits);
/// assert_eq!(fits.len(), 9);
/// let aic: Vec<f64> = fits.iter().filter_map(|(_, f)| f.as_ref().ok().map(|f| f.aic())).collect();
/// assert!(aic.windows(2).all(|w| w[0] <= w[1]));
/// ```
///
/// # Errors
///
/// When a Pareto family is named without `t`, a reporting threshold is
/// below `t`, or a Pareto family is named for losses truncated from above.
pub fn compare_fits(
    data: &LargeLosses,
    families: &[Family],
    t: Option<f64>,
) -> Result<Vec<(Family, Result<SeverityFit>)>> {
    let pareto = families.iter().find(|f| f.needs_threshold());
    match (t, pareto) {
        (None, Some(f)) => {
            return Err(Error::Data(format!(
                "{:?} is fitted only from a threshold t",
                f.name()
            )));
        }
        (Some(_), Some(f)) if data.is_truncated_above() => {
            return Err(Error::Data(format!(
                "{:?} takes no truncation from above",
                f.name()
            )));
        }
        _ => {}
    }
    if let (Some(t), Some(_)) = (t, pareto)
        && let Some((y, r, _, _)) = data.rows().find(|row| row.1 < t)
    {
        return Err(Error::Data(format!(
            "comparing fits from t needs every reporting threshold at least t ({t}), so that \
             every family conditions each loss on the same threshold; the loss {y} has {r}"
        )));
    }
    Ok(families
        .iter()
        .map(|&f| {
            let fit = fit_severity_with_threshold(f, data, t.filter(|_| f.needs_threshold()));
            (f, fit)
        })
        .collect())
}

/// Orders [`compare_fits`]' results by AIC, lowest first: a stable sort,
/// so a tie keeps the given order, and a family that failed to fit goes
/// last.
pub fn rank_by_aic(fits: &mut [(Family, Result<SeverityFit>)]) {
    let aic = |f: &Result<SeverityFit>| f.as_ref().map_or(f64::NAN, SeverityFit::aic);
    fits.sort_by(|a, b| {
        let (x, y) = (aic(&a.1), aic(&b.1));
        match (x.is_nan(), y.is_nan()) {
            (false, false) => x.total_cmp(&y),
            (p, q) => p.cmp(&q),
        }
    });
}

/// Each fitted family's survival beside the empirical one at the losses,
/// for a chart on a log axis ([`fit_curves`]).
#[derive(Debug, Clone)]
pub struct FitCurves {
    /// The losses kept, ascending.
    pub losses: Vec<f64>,
    /// The plotting position `(n − i + 0.5) / n` of each kept loss, its rank
    /// `i` among all `n` losses (1 the smallest), so that the last is not 0
    /// as the product-limit estimate is.
    pub empirical: Vec<f64>,
    /// The fits, ranked by AIC, each with its survival conditional on `t`
    /// (and on the bound from above, if any) at the kept losses. A family
    /// that failed to fit is left out.
    pub curves: Vec<(SeverityFit, Vec<f64>)>,
}

/// The rows [`fit_curves`] keeps of `n` sorted losses for `points` rows:
/// `unique(round(linspace(0, n − 1, points)))`, as numpy computes it
/// (`i · ((n − 1) / (points − 1))`, the last index `n − 1` exactly) and
/// rounds it (halves to even). The first and last losses are always kept;
/// with `points ≥ n` every loss is.
///
/// ```
/// use prospicio_prob::severity_fit::thinned_rows;
///
/// // 89 losses to 10 rows: 1, 11, 21, 30, 40, 50, 60, 69, 79, 89 counted from 1.
/// assert_eq!(thinned_rows(89, 10), [0, 10, 20, 29, 39, 49, 59, 68, 78, 88]);
/// assert_eq!(thinned_rows(3, 200), [0, 1, 2]);
/// ```
pub fn thinned_rows(n: usize, points: usize) -> Vec<usize> {
    if points >= n || n < 2 {
        return (0..n).collect();
    }
    let points = points.max(2);
    let step = (n - 1) as f64 / (points - 1) as f64;
    let mut rows: Vec<usize> = (0..points)
        .map(|i| {
            if i + 1 == points {
                n - 1
            } else {
                (i as f64 * step).round_ties_even() as usize
            }
        })
        .collect();
    rows.dedup();
    rows
}

/// The losses at or above `t`, ascending, with their plotting positions
/// `(n − i + 0.5) / n` and each family's survival conditional on `t`,
/// `S(x) / S(t)`, or `(S(x) − S(U)) / (S(t) − S(U))` with a bound from
/// above `U` common to every loss: the table behind a chart of every fit
/// against the data on a log axis. The families are fitted by
/// [`compare_fits`] from `t` and ranked by AIC; one that fails to fit is
/// left out. With more losses than `points` the table is thinned to
/// [`thinned_rows`], the plotting positions still counted among all `n`.
///
/// The plotting positions treat every loss as one exact observation, so
/// censored losses and weights other than 1 are refused.
///
/// ```
/// use prospicio_prob::LargeLosses;
/// use prospicio_prob::severity_fit::{Family, fit_curves};
///
/// let y = vec![1500.0, 2500.0, 4000.0, 10_000.0, 1200.0, 3100.0];
/// let data = LargeLosses::new(y).unwrap().reporting_thresholds(vec![1000.0; 6]).unwrap();
/// let c = fit_curves(&data, 1000.0, &[Family::Lognormal, Family::Pareto], 200).unwrap();
/// assert_eq!(c.losses[0], 1200.0);
/// assert!((c.empirical[0] - 5.5 / 6.0).abs() < 1e-15);
/// assert_eq!(c.curves.len(), 2);
/// ```
///
/// # Errors
///
/// As [`compare_fits`] (a reporting threshold below `t` with a Pareto
/// family named); when a loss is censored or weighted, the losses'
/// bounds from above differ, no loss is at or above `t`, or `points` is
/// below 2.
pub fn fit_curves(
    data: &LargeLosses,
    t: f64,
    families: &[Family],
    points: usize,
) -> Result<FitCurves> {
    use crate::Distribution;
    if points < 2 {
        return Err(Error::Data(format!(
            "points must be at least 2 (the smallest and largest loss), got {points}"
        )));
    }
    if data.rows().any(|r| r.2 || r.3 != 1.0) {
        return Err(Error::Data(
            "fit_curves takes exact, unweighted losses: the plotting positions count each \
             loss once"
                .into(),
        ));
    }
    let bounds = data.upper_bounds();
    let u = bounds[0];
    if bounds.iter().any(|&b| b != u) {
        return Err(Error::Data(
            "fit_curves needs one bound from above for every loss".into(),
        ));
    }
    let mut fits = compare_fits(data, families, Some(t))?;
    rank_by_aic(&mut fits);
    let mut y: Vec<f64> = data.values().iter().copied().filter(|&v| v >= t).collect();
    if y.is_empty() {
        return Err(Error::Data(format!("no loss is at or above t ({t})")));
    }
    y.sort_by(f64::total_cmp);
    let n = y.len();
    let rows = thinned_rows(n, points);
    let losses: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
    let empirical = rows
        .iter()
        .map(|&i| (n - i) as f64 - 0.5)
        .map(|k| k / n as f64)
        .collect();
    let curves = fits
        .into_iter()
        .filter_map(|(_, fit)| fit.ok())
        .filter_map(|fit| {
            let d = fit.dist();
            let s_u = if u.is_finite() { d.survival(u) } else { 0.0 };
            let mass = d.survival(t) - s_u;
            if mass.is_nan() || mass <= 0.0 {
                return None;
            }
            let s: Vec<f64> = losses
                .iter()
                .map(|&x| (d.survival(x) - s_u) / mass)
                .collect();
            Some((fit, s))
        })
        .collect();
    Ok(FitCurves {
        losses,
        empirical,
        curves,
    })
}

/// The mean and coefficient of variation of `X | X > t`, a fit's tail above
/// its threshold ([`tail_moments`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailMoments {
    /// `t`.
    pub threshold: f64,
    /// `E[X | X > t] = t + E[(X − t)₊] / S(t)`; infinite where the mean is.
    pub mean: f64,
    /// `sd(X | X > t) / E[X | X > t]`; infinite where the second moment is.
    pub cv: f64,
}

/// The mean and CV of `dist` above `t`, from its exact layer moments:
/// `E[X | X > t] = t + E[(X − t)₊] / S(t)` and
/// `Var(X | X > t) = E[(X − t)₊²] / S(t) − (E[(X − t)₊] / S(t))²`. A moment
/// that diverges (a Pareto's with `α ≤ 1` or `α ≤ 2`, a GPD's with
/// `ξ ≥ 1` or `ξ ≥ 1/2`, a loglogistic's with shape `≤ 1` or `≤ 2`) is
/// infinite. With a fit's `t` the lowest reporting threshold, these are
/// the moments of the losses the fit describes.
///
/// ```
/// use prospicio_prob::{Dist, Exponential};
/// use prospicio_prob::severity_fit::tail_moments;
///
/// // Memoryless: above t the exponential is t plus the same exponential.
/// let d: Dist = Exponential::new(1000.0).unwrap().into();
/// let m = tail_moments(&d, 500.0).unwrap();
/// assert!((m.mean - 1500.0).abs() < 1e-9);
/// assert!((m.cv - 1000.0 / 1500.0).abs() < 1e-12);
/// ```
///
/// # Errors
///
/// When `dist` is not a severity or gives no probability above `t`.
pub fn tail_moments(dist: &Dist, t: f64) -> Result<TailMoments> {
    let sev = dist.as_severity().ok_or_else(|| {
        Error::Data(format!(
            "tail_moments needs a severity, not a {}",
            dist.family()
        ))
    })?;
    let s = sev.survival(t);
    if s.is_nan() || s <= 0.0 {
        return Err(Error::Data(format!(
            "the distribution gives no probability above {t}"
        )));
    }
    let excess = sev.stop_loss(t) / s;
    let mean = t + excess;
    if !mean.is_finite() {
        return Ok(TailMoments {
            threshold: t,
            mean: f64::INFINITY,
            cv: f64::INFINITY,
        });
    }
    let second = sev.layer_second_moment(f64::INFINITY, t) / s;
    let cv = if second.is_finite() {
        (second - excess * excess).max(0.0).sqrt() / mean
    } else {
        f64::INFINITY
    };
    Ok(TailMoments {
        threshold: t,
        mean,
        cv,
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
    fn untruncated_lomax_solves_its_score_equations() {
        // Heavier than exponential (CV > 1), so the maximum is interior:
        // α = n / Σ ln(1 + y/θ) and (α + 1) Σ y/(θ + y) = n.
        let y = vec![
            50.0, 80.0, 120.0, 200.0, 300.0, 500.0, 900.0, 1500.0, 4000.0, 12000.0,
        ];
        let n = y.len() as f64;
        let fit = fit_severity(Family::Lomax, &LargeLosses::new(y.clone()).unwrap()).unwrap();
        let (alpha, theta) = (fit.parameters()[0], fit.parameters()[1]);
        let l: f64 = y.iter().map(|v| (v / theta).ln_1p()).sum();
        assert!((alpha * l / n - 1.0).abs() < 1e-7, "{:?}", fit.parameters());
        let q: f64 = y.iter().map(|v| v / (theta + v)).sum();
        assert!(((alpha + 1.0) * q / n - 1.0).abs() < 1e-7);
        assert!(fit.converged() && fit.standard_errors().iter().all(|s| s.is_finite()));
        // The fitted distribution is the Burr with γ = 1.
        assert_eq!(fit.dist().family(), "burr");
        let s = (1.0 + 1000.0 / theta).powf(-alpha);
        assert!((fit.dist().survival(1000.0) - s).abs() < 1e-14);
        // One fewer parameter than the Burr, which nests it.
        let burr = fit_severity(Family::Burr, &LargeLosses::new(y).unwrap()).unwrap();
        assert!(burr.log_likelihood() >= fit.log_likelihood() - 1e-6);
        assert_eq!(Family::from_name("lomax").unwrap(), Family::Lomax);
    }

    #[test]
    fn truncated_and_censored_fits_recover_the_truth() {
        let mut rng = StreamRng::new(11, 0);
        for (family, truth) in [
            (Family::Lognormal, vec![7.0, 1.2]),
            (Family::Gamma, vec![1.5, 800.0]),
            (Family::Weibull, vec![0.8, 1000.0]),
            (Family::Burr, vec![1.5, 2.0, 1500.0]),
            (Family::Lomax, vec![2.5, 1200.0]),
            (Family::Loglogistic, vec![2.2, 900.0]),
            (Family::Exponential, vec![1100.0]),
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
    fn fits_truncated_from_above_recover_the_truth() {
        let mut rng = StreamRng::new(17, 0);
        for (family, truth) in [
            (Family::Lognormal, vec![7.0, 1.2]),
            (Family::Gamma, vec![1.5, 800.0]),
            (Family::Weibull, vec![0.8, 1000.0]),
            (Family::Burr, vec![1.5, 2.0, 1500.0]),
            (Family::Lomax, vec![2.5, 1200.0]),
            (Family::Loglogistic, vec![2.2, 900.0]),
            (Family::Exponential, vec![1100.0]),
        ] {
            let d = family.dist(&truth).unwrap();
            // A third of the losses come from a study of claims up to 3000;
            // the rest are reported above 300 and capped at 5000.
            let (mut y, mut r, mut c, mut u) = (vec![], vec![], vec![], vec![]);
            for (i, x) in d.sample(&mut rng, 12_000).into_iter().enumerate() {
                let (lo, hi) = if i % 3 == 0 {
                    (0.0, 3000.0)
                } else {
                    (300.0, f64::INFINITY)
                };
                if x <= lo || x > hi {
                    continue;
                }
                y.push(x.min(5000.0));
                r.push(lo);
                c.push(x > 5000.0);
                u.push(hi);
            }
            let data = LargeLosses::new(y)
                .unwrap()
                .reporting_thresholds(r)
                .unwrap()
                .censored(c)
                .unwrap()
                .truncated_above(u)
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
            // The bounded likelihood, by hand at the truth, for the first
            // loss of the study (truncated, never censored).
            let (y0, u0) = (data.values()[0], data.upper_bounds()[0]);
            let one = LargeLosses::new(vec![y0])
                .unwrap()
                .truncated_above(vec![u0])
                .unwrap();
            let want = family.ln_f_s(&truth, y0).0 - d.cdf(u0).ln();
            assert!((log_likelihood(family, &truth, &one) - want).abs() < 1e-9);
        }
    }

    #[test]
    fn infinite_bounds_change_nothing() {
        let y = vec![120.0, 300.0, 450.0, 800.0, 1500.0, 2600.0, 4000.0];
        let plain = LargeLosses::new(y.clone()).unwrap();
        let bounded = plain
            .clone()
            .truncated_above(vec![f64::INFINITY; 7])
            .unwrap();
        assert!(!bounded.is_truncated_above());
        for family in [Family::Lognormal, Family::Gamma, Family::Lomax] {
            let a = fit_severity(family, &plain).unwrap();
            let b = fit_severity(family, &bounded).unwrap();
            assert_eq!(a.parameters(), b.parameters());
            assert_eq!(a.log_likelihood(), b.log_likelihood());
        }
    }

    #[test]
    fn truncation_from_above_is_checked_and_refused_where_unsupported() {
        let y = vec![100.0, 200.0, 400.0];
        let data = LargeLosses::new(y.clone()).unwrap();
        // Below a loss, or at a censored one.
        assert!(
            data.clone()
                .truncated_above(vec![150.0, 150.0, 500.0])
                .is_err()
        );
        assert!(data.clone().truncated_above(vec![f64::NAN; 3]).is_err());
        let censored = data.clone().censored(vec![false, false, true]).unwrap();
        assert!(censored.truncated_above(vec![500.0, 500.0, 400.0]).is_err());
        let bounded = data.truncated_above(vec![500.0, 500.0, 400.0]).unwrap();
        assert!(bounded.clone().censored(vec![false, false, true]).is_err());
        assert!(
            bounded
                .clone()
                .reporting_thresholds(vec![0.0, 0.0, 400.0])
                .is_err()
        );
        // The Pareto fits refuse it rather than ignore it.
        assert!(crate::Pareto::fit(50.0, &bounded, None).is_err());
        // goodness_of_fit wants one bound for every loss.
        let d = Lognormal::new(5.5, 1.0).unwrap();
        assert!(goodness_of_fit(&d, &bounded).is_err());
    }

    #[test]
    fn goodness_of_fit_conditions_on_a_common_bound() {
        let truth = Lognormal::new(7.0, 1.0).unwrap();
        let y: Vec<f64> = truth
            .sample(&mut StreamRng::new(3, 0), 20_000)
            .into_iter()
            .filter(|&x| x <= 2000.0)
            .collect();
        let n = y.len();
        let data = LargeLosses::new(y)
            .unwrap()
            .truncated_above(vec![2000.0; n])
            .unwrap();
        let gof = goodness_of_fit(&truth, &data).unwrap();
        assert!(gof.max_distance < 0.02, "{}", gof.max_distance);
        // Without the bound the comparison is far off.
        let plain = LargeLosses::new(data.values().to_vec()).unwrap();
        assert!(goodness_of_fit(&truth, &plain).unwrap().max_distance > 0.2);
        // The Q-Q points come back inside the bound.
        assert!(gof.model_quantile.iter().all(|&q| q <= 2000.0 + 1e-9));
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
        assert!(Family::from_name("frechet").is_err());
    }

    #[test]
    fn the_exponential_mean_is_in_closed_form() {
        // Σ w (y − r) / Σ_uncensored w, with thresholds and censoring.
        let data = LargeLosses::new(vec![300.0, 900.0, 1500.0, 4000.0, 2500.0])
            .unwrap()
            .reporting_thresholds(vec![0.0, 200.0, 200.0, 1000.0, 0.0])
            .unwrap()
            .censored(vec![false, false, false, false, true])
            .unwrap()
            .weights(vec![1.0, 2.0, 1.0, 1.0, 1.0])
            .unwrap();
        let fit = fit_severity(Family::Exponential, &data).unwrap();
        let want = (300.0 + 2.0 * 700.0 + 1300.0 + 3000.0 + 2500.0) / 5.0;
        assert!((fit.parameters()[0] / want - 1.0).abs() < 1e-14);
        // The standard error is θ / sqrt(n_uncensored).
        assert!((fit.standard_errors()[0] / (want / 5f64.sqrt()) - 1.0).abs() < 1e-4);
        assert_eq!(fit.dist().family(), "exponential");
        // One threshold, no censoring: mean(y) − t.
        let y = vec![1200.0, 1500.0, 2600.0, 5000.0];
        let one = LargeLosses::new(y)
            .unwrap()
            .reporting_thresholds(vec![1000.0; 4])
            .unwrap();
        let fit = fit_severity(Family::Exponential, &one).unwrap();
        assert!((fit.parameters()[0] - (2575.0 - 1000.0)).abs() < 1e-9);
        assert_eq!(fit.aic(), 2.0 - 2.0 * fit.log_likelihood());
    }

    #[test]
    fn the_loglogistic_is_the_burr_with_alpha_one() {
        let y = vec![
            50.0, 80.0, 120.0, 200.0, 300.0, 500.0, 900.0, 1500.0, 4000.0, 12000.0,
        ];
        let data = LargeLosses::new(y)
            .unwrap()
            .reporting_thresholds(vec![40.0; 10])
            .unwrap();
        let fit = fit_severity(Family::Loglogistic, &data).unwrap();
        let p = fit.parameters();
        let burr = log_likelihood(Family::Burr, &[1.0, p[0], p[1]], &data);
        assert!((burr - fit.log_likelihood()).abs() < 1e-10);
        // The score is zero at the maximum.
        for i in 0..2 {
            let h = 1e-6 * p[i];
            let (mut a, mut b) = (p.to_vec(), p.to_vec());
            a[i] += h;
            b[i] -= h;
            let g = (log_likelihood(Family::Loglogistic, &a, &data)
                - log_likelihood(Family::Loglogistic, &b, &data))
                / (2.0 * h);
            assert!((g * p[i]).abs() < 1e-4, "{i}: {g}");
        }
        assert_eq!(fit.dist().family(), "loglogistic");
        let s = 1.0 / (1.0 + (1000.0 / p[1]).powf(p[0]));
        assert!((fit.dist().survival(1000.0) - s).abs() < 1e-14);
    }

    #[test]
    fn the_pareto_families_need_a_threshold_and_only_they_take_one() {
        let data = LargeLosses::new(vec![1500.0, 2500.0, 4000.0, 10_000.0]).unwrap();
        assert!(fit_severity(Family::Pareto, &data).is_err());
        assert!(fit_severity_with_threshold(Family::Gamma, &data, Some(1000.0)).is_err());
        // A loss below t is refused.
        assert!(fit_severity_with_threshold(Family::Pareto, &data, Some(2000.0)).is_err());
        let fit = fit_severity_with_threshold(Family::Pareto, &data, Some(1000.0)).unwrap();
        assert_eq!(fit.family().estimated_parameters(), 1);
        assert_eq!(fit.parameters()[0], 1000.0);
        let se = fit.standard_errors();
        assert!(se[0].is_nan());
        assert!((se[1] - fit.parameters()[1] / 2.0).abs() < 1e-14);
        // A reporting threshold below t conditions on t: the same
        // likelihood as thresholds at t.
        let at_t = data.clone().reporting_thresholds(vec![1000.0; 4]).unwrap();
        let fit_t = fit_severity_with_threshold(Family::Pareto, &at_t, Some(1000.0)).unwrap();
        assert_eq!(fit.log_likelihood(), fit_t.log_likelihood());
        // The grouped fits take the body families only.
        assert!(Family::GenPareto.body_only("x").is_err());
        assert_eq!(Family::ALL.len(), 9);
        for f in Family::ALL {
            assert_eq!(Family::from_name(f.name()).unwrap(), f);
        }
        // Neither takes a truncation from above.
        let capped = at_t.truncated_above(vec![1e6; 4]).unwrap();
        for f in [Family::Pareto, Family::GenPareto] {
            assert!(fit_severity_with_threshold(f, &capped, Some(1000.0)).is_err());
        }
    }

    #[test]
    fn thinning_rounds_halves_to_even_as_numpy_does() {
        // numpy: unique(round(linspace(0, 5, 3))) = [0, 2, 5] (2.5 to 2),
        // unique(round(linspace(0, 3, 3))) = [0, 2, 3] (1.5 to 2).
        assert_eq!(thinned_rows(6, 3), [0, 2, 5]);
        assert_eq!(thinned_rows(4, 3), [0, 2, 3]);
        assert_eq!(thinned_rows(5, 2), [0, 4]);
        assert_eq!(thinned_rows(1, 2), [0]);
        assert_eq!(thinned_rows(10, 10), (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn fit_curves_refuse_what_the_plotting_positions_cannot_count() {
        let y = vec![1500.0, 2500.0, 4000.0, 10_000.0, 1200.0, 3100.0];
        let data = LargeLosses::new(y).unwrap();
        let at = data.clone().reporting_thresholds(vec![1000.0; 6]).unwrap();
        let fams = [Family::Lognormal, Family::Pareto];
        // A reporting threshold below t.
        assert!(fit_curves(&data, 1000.0, &fams, 200).is_err());
        assert!(fit_curves(&at, 1000.0, &fams, 1).is_err());
        let censored = at
            .clone()
            .censored(vec![false, false, true, false, false, false])
            .unwrap();
        assert!(fit_curves(&censored, 1000.0, &fams, 200).is_err());
        let weighted = at.clone().weights(vec![2.0; 6]).unwrap();
        assert!(fit_curves(&weighted, 1000.0, &fams, 200).is_err());
        // The body families alone need no thresholds at t.
        let body = fit_curves(&data, 1000.0, &[Family::Gamma, Family::Lognormal], 3).unwrap();
        // Six losses to three rows: indices 0, 2.5 (to 2, the even) and 5.
        assert_eq!(body.losses, [1200.0, 2500.0, 10_000.0]);
        assert_eq!(body.empirical, [5.5 / 6.0, 3.5 / 6.0, 0.5 / 6.0]);
        // Each curve is 1 at t and falls.
        let c = fit_curves(&at, 1000.0, &fams, 200).unwrap();
        assert_eq!(c.curves.len(), 2);
        for (_, s) in &c.curves {
            assert!(s.windows(2).all(|w| w[0] >= w[1]));
            assert!(s[0] < 1.0 && s[0] > 0.5);
        }
    }

    #[test]
    fn tail_moments_diverge_with_the_family() {
        use crate::{Burr, Pareto};
        // A Pareto from 1000 with alpha 1.5: mean 3000 above 1000, no CV.
        let p: Dist = Pareto::new(1000.0, 1.5).unwrap().into();
        let m = tail_moments(&p, 1000.0).unwrap();
        assert!((m.mean - 3000.0).abs() < 1e-9);
        assert_eq!(m.cv, f64::INFINITY);
        // Above 2000 a Pareto is a Pareto from 2000: mean 2000 α/(α − 1).
        let m = tail_moments(&Pareto::new(1000.0, 3.0).unwrap().into(), 2000.0).unwrap();
        assert!((m.mean - 3000.0).abs() < 1e-9);
        assert!((m.cv - (1.0f64 / 3.0).sqrt()).abs() < 1e-9);
        let heavy: Dist = Pareto::new(1000.0, 0.9).unwrap().into();
        let m = tail_moments(&heavy, 1000.0).unwrap();
        assert_eq!((m.mean, m.cv), (f64::INFINITY, f64::INFINITY));
        // A loglogistic with shape 1.5: a mean, no variance.
        let ll: Dist = Burr::new(1.0, 1.5, 1000.0).unwrap().into();
        let m = tail_moments(&ll, 500.0).unwrap();
        assert!(m.mean.is_finite() && m.cv == f64::INFINITY);
    }

    #[test]
    fn the_ks_p_value_is_scipys_kstest() {
        // An exponential with mean 500 against five losses:
        // scipy.stats.kstest(x, expon(scale=500).cdf).
        let x = vec![100.0, 300.0, 350.0, 700.0, 950.0];
        let data = LargeLosses::new(x).unwrap();
        let g = goodness_of_fit(&Exponential::new(500.0).unwrap(), &data).unwrap();
        assert!((g.max_distance - 0.2511883639059736).abs() < 1e-12);
        assert!(
            (g.ks_p_value - 0.8409019263900253).abs() < 1e-10,
            "{}",
            g.ks_p_value
        );
    }
}
