//! Merz and Wüthrich's paid-incurred chain (PIC, 2010): log-normal paid
//! and incurred development combined into one ultimate per origin, with a
//! closed-form posterior and mean squared error of prediction
//! (`docs/design/reserving-v02.md`, decision 32).
//!
//! The model (Merz and Wüthrich 2010, Model Assumptions 2.1 and 3.1), with
//! development periods `j = 0, ..., J` and as many origins:
//!
//! - paid: `log(P[i,j] / P[i,j-1]) ~ N(Phi_j, sigma_j^2)` (with
//!   `P[i,-1] = 1`), Hertig's model;
//! - incurred: `log(I[i,j+1] / I[i,j]) ~ N(Psi_j, tau_j^2)` backwards from
//!   `I[i,J] = P[i,J]`, Gogol's model;
//! - all of these independent given the parameters, and independent normal
//!   priors `Phi_j ~ N(phi_j, s_j^2)`, `Psi_j ~ N(psi_j, t_j^2)`, with the
//!   variances `sigma_j^2` and `tau_j^2` known.
//!
//! The posterior of `(Phi, Psi)` is then normal, and the ultimate of origin
//! `i`, latest at `k = J - i`, is predicted from both its latest paid and
//! latest incurred value:
//!
//! `E[P[i,J] | D] = P[i,k]^(1 - b_k) I[i,k]^b_k exp((1 - b_k) sum_{m>k}
//! Phi_m + b_k sum_{n>=k} Psi_n + (1 - b_k) sum_{m>k} sigma_m^2 / 2 + e'Te/2)`
//!
//! with the credibility weight `b_k = sum_{m>k} sigma_m^2 / (sum_{m>k}
//! sigma_m^2 + sum_{n>=k} tau_n^2)` given to incurred, the posterior means
//! in place of `Phi` and `Psi`, and `e'Te` the posterior variance of the
//! exponent. As the incurred variances grow, `b_k` falls to 0 and the
//! prediction becomes Hertig's paid chain ladder (log link ratios averaged);
//! as they shrink, it becomes the incurred chain ladder on geometric
//! averages.

use prospicio_core::{Lag, Period};
use prospicio_math::linalg::{cholesky, cholesky_inverse, cholesky_solve};

use crate::error::{Error, Result};
use crate::segments::FitTable;
use crate::triangle::{Segment, Triangle};

/// How the last paid and the last incurred variance, which have one
/// observation each, are filled in when the variances are estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VarianceExtrapolation {
    /// `exp` of a least-squares line through the logs of the other
    /// estimated variances against their development periods, as R
    /// ChainLadder's `PaidIncurredChain` does. R leaves a variance of zero
    /// out of the line; here an estimated variance of zero is an error
    /// instead, and the variances are then to be given.
    #[default]
    LogLinear,
    /// Mack's rule on the standard deviations, `min(s_{J-1}, s_{J-2},
    /// s_{J-1}^2 / s_{J-2})`, as Merz and Wüthrich's tail-factor paper
    /// (2010 preprint, section 4) does for `sigma_J`.
    Mack,
}

impl VarianceExtrapolation {
    /// Every name the Python, R and Excel front ends accept, with its
    /// rule, the canonical name of each first (#834).
    pub const NAMES: &'static [(&'static str, Self)] = &[
        ("loglinear", Self::LogLinear),
        ("log-linear", Self::LogLinear),
        ("log_linear", Self::LogLinear),
        ("mack", Self::Mack),
    ];

    /// The rule named `name` ([`NAMES`](Self::NAMES)), or `None`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|&(_, r)| r)
    }

    /// The canonical name, `"loglinear"` or `"mack"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::LogLinear => "loglinear",
            Self::Mack => "mack",
        }
    }
}

/// Normal priors on the paid and incurred log link parameters: `Phi_j ~
/// N(phi_mean[j], phi_sd[j]^2)` for `j = 0, ..., J` and `Psi_j ~
/// N(psi_mean[j], psi_sd[j]^2)` for `j = 0, ..., J - 1`, independent.
///
/// An empty pair of vectors leaves that side non-informative (the
/// default, and R ChainLadder's only case), and an infinite standard
/// deviation leaves that one parameter non-informative, whatever its
/// mean.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PicPriors {
    /// Prior means of `Phi_0, ..., Phi_J`, one per development period.
    pub phi_mean: Vec<f64>,
    /// Prior standard deviations of `Phi_0, ..., Phi_J`; positive, or
    /// infinite for none.
    pub phi_sd: Vec<f64>,
    /// Prior means of `Psi_0, ..., Psi_{J-1}`, one per development period
    /// but the last.
    pub psi_mean: Vec<f64>,
    /// Prior standard deviations of `Psi_0, ..., Psi_{J-1}`; positive, or
    /// infinite for none.
    pub psi_sd: Vec<f64>,
}

impl PicPriors {
    /// Non-informative priors on every parameter (`s_j, t_j -> infinity`),
    /// R ChainLadder's case.
    pub fn non_informative() -> Self {
        Self::default()
    }
}

/// Merz and Wüthrich's (2010) paid-incurred chain: one ultimate per origin
/// from a paid and an incurred column, with its mean squared error of
/// prediction in closed form. See the [module documentation](self).
///
/// The defaults reproduce R ChainLadder's `PaidIncurredChain`:
/// non-informative priors, and `sigma_j` and `tau_j` the sample standard
/// deviations of each period's log link ratios (of the first paid values
/// for `sigma_0`), the last of each extrapolated log-linearly.
///
/// ```
/// use prospicio_reserving::{PaidIncurredChain, Triangle, DevelopmentColumn, Grain, Long, Month};
///
/// let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
///     values: &[
///         ("paid", &[100.0, 150.0, 170.0, 180.0, 110.0, 168.0, 186.0, 125.0, 180.0, 130.0]),
///         ("incurred", &[200.0, 195.0, 185.0, 181.0, 215.0, 200.0, 192.0, 240.0, 230.0, 260.0]),
///     ],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let fit = PaidIncurredChain::default().fit(&tri, "paid", "incurred")?;
/// assert_eq!(fit.reserve[0], 0.0);
/// assert!(fit.total_reserve() > 0.0 && fit.total_std_err > 0.0);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PaidIncurredChain {
    /// Priors on the log link parameters; non-informative by default.
    pub priors: PicPriors,
    /// The paid standard deviations `sigma_0, ..., sigma_J`, one per
    /// development period; `None` estimates them.
    pub sigma: Option<Vec<f64>>,
    /// The incurred standard deviations `tau_0, ..., tau_{J-1}`, one per
    /// development period but the last; `None` estimates them.
    pub tau: Option<Vec<f64>>,
    /// How the last estimated `sigma` and `tau` are filled in.
    pub extrapolation: VarianceExtrapolation,
    /// Which posterior precision to use: the model's (the default), or R
    /// ChainLadder's, for reconciling with it.
    pub convention: PicConvention,
}

/// The posterior precision of the parameters.
///
/// R ChainLadder 0.2.21's `PaidIncurredChain` sets the entries of its
/// precision matrix `A` that couple `Psi_0` with `Phi_1, ..., Phi_J` to
/// zero (its loops test `n == 0 | m == 0` where only `n == 0` gives a
/// zero), although the newest origin's incurred-paid ratio
/// `log(I[J,0] / P[J,0])`, which its `A` and `b` count elsewhere, involves
/// all of them. That is not the posterior of the model; prospicio's default
/// is (checked in the unit tests against conditioning on every origin's
/// paid and incurred values directly). Whether the paper's printed formula
/// for `A` shares R's omission could not be checked, the paper not being
/// at hand. On R's USAA example the difference
/// raises the total reserve and its standard error by about 0.5% each
/// (`docs/design/reserving-v02.md`, decision 32).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PicConvention {
    /// The posterior of the model (Merz and Wüthrich 2010, Model
    /// Assumptions 3.1).
    #[default]
    Model,
    /// R ChainLadder's `PaidIncurredChain` exactly, with the coupling of
    /// `Psi_0` and `Phi_1, ..., Phi_J` left out.
    RChainLadder,
}

/// A fitted paid-incurred chain. Per-origin vectors run over the
/// triangle's origins; the oldest is taken as fully developed, its
/// ultimate its last paid value with no reserve and no error.
#[derive(Debug, Clone, PartialEq)]
pub struct PicFit {
    /// Origin periods.
    pub origins: Vec<Period>,
    /// Development ages, one per period `j = 0, ..., J`.
    pub ages: Vec<Lag>,
    /// Latest paid value per origin.
    pub latest_paid: Vec<f64>,
    /// Latest incurred value per origin.
    pub latest_incurred: Vec<f64>,
    /// Predicted ultimate per origin, `E[P[i,J] | D]`.
    pub ultimate: Vec<f64>,
    /// Reserve per origin: the ultimate less the latest paid value.
    pub reserve: Vec<f64>,
    /// Weight `b_k` of the incurred side in each origin's prediction (0 for
    /// the oldest origin).
    pub incurred_weight: Vec<f64>,
    /// Square root of the conditional mean squared error of prediction of
    /// each origin's ultimate (and so of its reserve).
    pub std_err: Vec<f64>,
    /// The same for the total, with the covariance between origins.
    pub total_std_err: f64,
    /// `sigma_0, ..., sigma_J` used, given or estimated.
    pub sigma: Vec<f64>,
    /// `tau_0, ..., tau_{J-1}` used, given or estimated.
    pub tau: Vec<f64>,
    /// Posterior mean of `(Phi_0, ..., Phi_J, Psi_0, ..., Psi_{J-1})`.
    pub posterior_mean: Vec<f64>,
    /// Posterior covariance of the same, row-major.
    pub posterior_covariance: Vec<f64>,
}

impl PaidIncurredChain {
    /// Fits the `paid` and `incurred` columns of a single-segment triangle.
    ///
    /// Both columns must cover the same square triangle: as many origins
    /// as development periods, origin `i` observed at exactly the first
    /// `J + 1 - i` periods, every value positive. The variances need at
    /// least four periods when estimated.
    ///
    /// # Errors
    ///
    /// [`Error::PaidIncurred`] for a triangle of another shape, a value
    /// that is not positive, settings of the wrong length or out of range,
    /// or an estimated variance of zero; [`Error::MultipleSegments`] and
    /// [`Error::UnknownColumn`] as for the other methods.
    pub fn fit(&self, triangle: &Triangle, paid: &str, incurred: &str) -> Result<PicFit> {
        let p = triangle.segment(paid)?;
        let i = triangle.segment(incurred)?;
        let paid_rows = square_rows(&p, paid)?;
        let incurred_rows = square_rows(&i, incurred)?;
        if p.origins != i.origins || p.ages != i.ages {
            return Err(Error::PaidIncurred(format!(
                "{paid:?} and {incurred:?} must cover the same origins and ages"
            )));
        }
        self.fit_rows(p.origins, p.ages, &paid_rows, &incurred_rows)
    }

    fn fit_rows(
        &self,
        origins: Vec<Period>,
        ages: Vec<Lag>,
        paid: &[Vec<f64>],
        incurred: &[Vec<f64>],
    ) -> Result<PicFit> {
        let n = ages.len(); // J + 1 periods
        let jj = n - 1; // the paper's J
        // Paid log increments per period: log P[i,0], then log link ratios.
        let xi: Vec<Vec<f64>> = (0..n)
            .map(|j| {
                (0..n - j)
                    .map(|o| {
                        let prev = if j == 0 { 0.0 } else { paid[o][j - 1].ln() };
                        paid[o][j].ln() - prev
                    })
                    .collect()
            })
            .collect();
        // Incurred log link ratios zeta[i,j] = log(I[i,j+1] / I[i,j]).
        let zeta: Vec<Vec<f64>> = (0..jj)
            .map(|j| {
                (0..jj - j)
                    .map(|o| (incurred[o][j + 1] / incurred[o][j]).ln())
                    .collect()
            })
            .collect();

        let sigma = match &self.sigma {
            Some(s) => given_sd(s, n, "sigma", "one per development period")?,
            None => estimated_sd(&xi, self.extrapolation, "sigma")?,
        };
        let tau = match &self.tau {
            Some(t) => given_sd(t, jj, "tau", "one per development period but the last")?,
            None => estimated_sd(&zeta, self.extrapolation, "tau")?,
        };
        let s2: Vec<f64> = sigma.iter().map(|s| s * s).collect();
        let t2: Vec<f64> = tau.iter().map(|t| t * t).collect();
        let (prior_precision, prior_shift) = self.priors.terms(n)?;

        // Variance of log(I[i,k] / P[i,k]) given the parameters: the paid
        // variance still to come and the incurred variance from k on,
        // `v_k^2 - w_k^2` in the paper.
        let paid_after = |k: usize| s2[k + 1..].iter().sum::<f64>();
        let gap = |k: usize| paid_after(k) + t2[k..].iter().sum::<f64>();

        // The posterior precision `A` and `A mean`, assembled from the
        // likelihood's terms (Merz and Wüthrich 2010, eq. 3.5): each one an
        // observation `y` of a linear form `a . theta` with variance `v`
        // adds `a a' / v` and `a y / v`.
        let dim = n + jj;
        let psi = |j: usize| n + j;
        let mut precision = vec![0.0; dim * dim];
        let mut shift = vec![0.0; dim];
        for (j, obs) in xi.iter().enumerate() {
            precision[j * dim + j] += obs.len() as f64 / s2[j];
            shift[j] += obs.iter().sum::<f64>() / s2[j];
        }
        for (j, obs) in zeta.iter().enumerate() {
            precision[psi(j) * dim + psi(j)] += obs.len() as f64 / t2[j];
            shift[psi(j)] += obs.iter().sum::<f64>() / t2[j];
        }
        for o in 1..n {
            let k = jj - o;
            let y = (incurred[o][k] / paid[o][k]).ln();
            let v = gap(k);
            let form: Vec<(usize, f64)> = (k + 1..n)
                .map(|m| (m, 1.0))
                .chain((k..jj).map(|m| (psi(m), -1.0)))
                .collect();
            for &(r, ar) in &form {
                shift[r] += ar * y / v;
                for &(c, ac) in &form {
                    let r_drops = |a: usize, b: usize| (1..n).contains(&a) && b == psi(0);
                    if self.convention == PicConvention::RChainLadder
                        && (r_drops(r, c) || r_drops(c, r))
                    {
                        continue;
                    }
                    precision[r * dim + c] += ar * ac / v;
                }
            }
        }
        for d in 0..dim {
            precision[d * dim + d] += prior_precision[d];
            shift[d] += prior_shift[d];
        }
        let factor = cholesky(&precision, dim).ok_or_else(|| {
            Error::PaidIncurred("the posterior precision is not positive definite".into())
        })?;
        let mean = cholesky_solve(&factor, &shift);
        let covariance = cholesky_inverse(&factor, dim);

        // Each origin's exponent as a linear form `e . theta`.
        let mut weight = vec![0.0; n];
        let mut forms: Vec<Vec<f64>> = vec![vec![0.0; dim]; n];
        let mut ultimate = vec![0.0; n];
        let mut own_var = vec![0.0; n];
        ultimate[0] = paid[0][jj];
        for o in 1..n {
            let k = jj - o;
            let b = paid_after(k) / gap(k);
            weight[o] = b;
            let e = &mut forms[o];
            e[k + 1..n].fill(1.0 - b);
            e[psi(k)..psi(jj)].fill(b);
            let quad = quadratic(&covariance, e, e);
            let mu: f64 = e.iter().zip(&mean).map(|(a, t)| a * t).sum();
            own_var[o] = (1.0 - b) * paid_after(k);
            ultimate[o] = ((1.0 - b) * paid[o][k].ln()
                + b * incurred[o][k].ln()
                + mu
                + own_var[o] / 2.0
                + quad / 2.0)
                .exp();
        }
        let mut std_err = vec![0.0; n];
        let mut total_msep = 0.0;
        for a in 1..n {
            for c in 1..n {
                let mut exponent = quadratic(&covariance, &forms[a], &forms[c]);
                if a == c {
                    exponent += own_var[a];
                }
                let term = exponent.exp_m1() * ultimate[a] * ultimate[c];
                if a == c {
                    std_err[a] = term.sqrt();
                }
                total_msep += term;
            }
        }
        let latest_paid: Vec<f64> = (0..n).map(|o| paid[o][jj - o]).collect();
        let latest_incurred: Vec<f64> = (0..n).map(|o| incurred[o][jj - o]).collect();
        let mut reserve: Vec<f64> = ultimate
            .iter()
            .zip(&latest_paid)
            .map(|(u, p)| u - p)
            .collect();
        reserve[0] = 0.0;
        Ok(PicFit {
            origins,
            ages,
            latest_paid,
            latest_incurred,
            ultimate,
            reserve,
            incurred_weight: weight,
            std_err,
            total_std_err: total_msep.sqrt(),
            sigma,
            tau,
            posterior_mean: mean,
            posterior_covariance: covariance,
        })
    }
}

impl PicPriors {
    /// The prior's precision and `precision * mean` for each of the `2n - 1`
    /// parameters.
    fn terms(&self, n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut precision = Vec::with_capacity(2 * n - 1);
        let mut shift = Vec::with_capacity(2 * n - 1);
        for (means, sds, len, name) in [
            (&self.phi_mean, &self.phi_sd, n, "phi"),
            (&self.psi_mean, &self.psi_sd, n - 1, "psi"),
        ] {
            if means.is_empty() && sds.is_empty() {
                precision.extend(std::iter::repeat_n(0.0, len));
                shift.extend(std::iter::repeat_n(0.0, len));
                continue;
            }
            if means.len() != len || sds.len() != len {
                return Err(Error::PaidIncurred(format!(
                    "the {name} prior needs {len} means and {len} standard deviations, \
                     got {} and {}",
                    means.len(),
                    sds.len()
                )));
            }
            for (&m, &s) in means.iter().zip(sds) {
                if s.is_nan() || s <= 0.0 {
                    return Err(Error::PaidIncurred(format!(
                        "a {name} prior standard deviation must be positive or infinite, got {s}"
                    )));
                }
                if s.is_infinite() {
                    precision.push(0.0);
                    shift.push(0.0);
                    continue;
                }
                if !m.is_finite() {
                    return Err(Error::PaidIncurred(format!(
                        "a {name} prior mean must be finite, got {m}"
                    )));
                }
                precision.push(1.0 / (s * s));
                shift.push(m / (s * s));
            }
        }
        Ok((precision, shift))
    }
}

/// `a' C b` for a row-major symmetric `C`.
fn quadratic(c: &[f64], a: &[f64], b: &[f64]) -> f64 {
    let dim = a.len();
    (0..dim)
        .filter(|&r| a[r] != 0.0)
        .map(|r| a[r] * (0..dim).map(|k| c[r * dim + k] * b[k]).sum::<f64>())
        .sum()
}

/// The rows of a square, fully observed upper triangle of positive values,
/// origin `o` holding its first `n - o` values.
fn square_rows(segment: &Segment, column: &str) -> Result<Vec<Vec<f64>>> {
    let n = segment.n_dev;
    if segment.n_origins != n {
        return Err(Error::PaidIncurred(format!(
            "{column:?} must be a square triangle, as many origins as development periods; \
             it has {} origins and {n} periods",
            segment.n_origins
        )));
    }
    if n < 3 {
        return Err(Error::TooFewAges {
            needed: 3,
            found: n,
        });
    }
    (0..n)
        .map(|o| {
            (0..n)
                .filter_map(|d| {
                    let value = segment.get(o, d);
                    match (d < n - o, value) {
                        (true, Some(v)) if v > 0.0 && v.is_finite() => Some(Ok(v)),
                        (true, Some(v)) => Some(Err(Error::PaidIncurred(format!(
                            "{column:?} must be positive, and origin {} has {v} at age {}",
                            segment.origins[o], segment.ages[d]
                        )))),
                        (true, None) | (false, Some(_)) => Some(Err(Error::PaidIncurred(format!(
                            "{column:?} must be observed exactly on and above the latest \
                             diagonal; origin {} at age {} is not",
                            segment.origins[o], segment.ages[d]
                        )))),
                        (false, None) => None,
                    }
                })
                .collect()
        })
        .collect()
}

/// Given standard deviations, checked.
fn given_sd(values: &[f64], len: usize, name: &str, what: &str) -> Result<Vec<f64>> {
    if values.len() != len {
        return Err(Error::PaidIncurred(format!(
            "{name} needs {len} values ({what}), got {}",
            values.len()
        )));
    }
    if let Some(v) = values.iter().find(|v| !(**v > 0.0 && v.is_finite())) {
        return Err(Error::PaidIncurred(format!(
            "{name} must be positive and finite, got {v}"
        )));
    }
    Ok(values.to_vec())
}

/// The sample standard deviation (divisor `n - 1`) of each period's
/// observations, the last period, with one observation, extrapolated.
fn estimated_sd(
    obs: &[Vec<f64>],
    extrapolation: VarianceExtrapolation,
    name: &str,
) -> Result<Vec<f64>> {
    let len = obs.len();
    if len < 3 {
        return Err(Error::TooFewAges {
            needed: 4,
            found: len + usize::from(name == "tau"),
        });
    }
    let mut var: Vec<f64> = obs[..len - 1]
        .iter()
        .map(|x| {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64
        })
        .collect();
    if let Some(j) = var.iter().position(|v| v.is_nan() || *v <= 0.0) {
        return Err(Error::PaidIncurred(format!(
            "the estimated {name} of development period {j} is zero; give {name}"
        )));
    }
    let last = match extrapolation {
        VarianceExtrapolation::LogLinear => {
            // R: lm(log(s2) ~ dev) over the estimated periods, numbered
            // from 1, predicted at the last one.
            let xs: Vec<f64> = (1..len).map(|d| d as f64).collect();
            let ys: Vec<f64> = var.iter().map(|v| v.ln()).collect();
            let k = xs.len() as f64;
            let mx = xs.iter().sum::<f64>() / k;
            let my = ys.iter().sum::<f64>() / k;
            let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
            let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
            let slope = sxy / sxx;
            (my + slope * (len as f64 - mx)).exp()
        }
        VarianceExtrapolation::Mack => {
            let (a, b) = (var[len - 2].sqrt(), var[len - 3].sqrt());
            a.min(b).min(a * a / b).powi(2)
        }
    };
    var.push(last);
    if !last.is_finite() || last <= 0.0 {
        return Err(Error::PaidIncurred(format!(
            "the last {name} cannot be extrapolated; give {name}"
        )));
    }
    Ok(var.into_iter().map(f64::sqrt).collect())
}

impl PicFit {
    /// Number of development periods, `J + 1`.
    fn n(&self) -> usize {
        self.ages.len()
    }

    /// Total predicted ultimate, the oldest origin's last paid value
    /// included.
    pub fn total_ultimate(&self) -> f64 {
        self.ultimate.iter().sum()
    }

    /// Total reserve.
    pub fn total_reserve(&self) -> f64 {
        self.reserve.iter().sum()
    }

    /// Posterior means of `Phi_0, ..., Phi_J`.
    pub fn phi(&self) -> &[f64] {
        &self.posterior_mean[..self.n()]
    }

    /// Posterior means of `Psi_0, ..., Psi_{J-1}`.
    pub fn psi(&self) -> &[f64] {
        &self.posterior_mean[self.n()..]
    }

    /// Posterior standard deviation of every parameter, in the order of
    /// [`posterior_mean`](Self::posterior_mean).
    pub fn posterior_sd(&self) -> Vec<f64> {
        let dim = self.posterior_mean.len();
        (0..dim)
            .map(|d| self.posterior_covariance[d * dim + d].sqrt())
            .collect()
    }

    /// One row per origin: `latest` (paid), `latest_incurred`, `ultimate`,
    /// `reserve`, `incurred_weight` and `std_err`.
    pub fn to_long(&self) -> FitTable {
        FitTable {
            keys: Vec::new(),
            origin: Some(self.origins.clone()),
            age: None,
            values: vec![
                ("latest".into(), self.latest_paid.clone()),
                ("latest_incurred".into(), self.latest_incurred.clone()),
                ("ultimate".into(), self.ultimate.clone()),
                ("reserve".into(), self.reserve.clone()),
                ("incurred_weight".into(), self.incurred_weight.clone()),
                ("std_err".into(), self.std_err.clone()),
            ],
        }
    }

    /// One row: total `latest`, `latest_incurred`, `ultimate`, `reserve`
    /// and the `std_err` of the total.
    pub fn totals(&self) -> FitTable {
        let one = |x: f64| vec![x];
        FitTable {
            keys: Vec::new(),
            origin: None,
            age: None,
            values: vec![
                ("latest".into(), one(self.latest_paid.iter().sum())),
                (
                    "latest_incurred".into(),
                    one(self.latest_incurred.iter().sum()),
                ),
                ("ultimate".into(), one(self.total_ultimate())),
                ("reserve".into(), one(self.total_reserve())),
                ("std_err".into(), one(self.total_std_err)),
            ],
        }
    }

    /// One row per development period `j`: `sigma`, the posterior `phi`
    /// and `phi_sd`, `tau`, the posterior `psi` and `psi_sd` (NaN at the
    /// last period, which has no incurred link).
    pub fn development_table(&self) -> FitTable {
        let n = self.n();
        let sd = self.posterior_sd();
        let incurred = |v: &[f64]| -> Vec<f64> { v.iter().copied().chain([f64::NAN]).collect() };
        FitTable {
            keys: Vec::new(),
            origin: None,
            age: Some(self.ages.clone()),
            values: vec![
                ("sigma".into(), self.sigma.clone()),
                ("phi".into(), self.phi().to_vec()),
                ("phi_sd".into(), sd[..n].to_vec()),
                ("tau".into(), incurred(&self.tau)),
                ("psi".into(), incurred(self.psi())),
                ("psi_sd".into(), incurred(&sd[n..])),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Average, ChainLadder, Development, DevelopmentColumn, Grain, Long, Month};

    /// Merz and Wüthrich's example data (Dahms 2008), as printed in their
    /// tail-factor paper (2010 preprint), Tables 1 and 2.
    const PAID: [&[f64]; 10] = [
        &[
            1216632.0, 1347072.0, 1786877.0, 2281606.0, 2656224.0, 2909307.0, 3283388.0, 3587549.0,
            3754403.0, 3821258.0,
        ],
        &[
            798924.0, 1051912.0, 1215785.0, 1349939.0, 1655312.0, 1926210.0, 2132833.0, 2287311.0,
            2567056.0,
        ],
        &[
            1115636.0, 1387387.0, 1930867.0, 2177002.0, 2513171.0, 2931930.0, 3047368.0, 3182511.0,
        ],
        &[
            1052161.0, 1321206.0, 1700132.0, 1971303.0, 2298349.0, 2645113.0, 3003425.0,
        ],
        &[
            808864.0, 1029523.0, 1229626.0, 1590338.0, 1842662.0, 2150351.0,
        ],
        &[1016862.0, 1251420.0, 1698052.0, 2105143.0, 2385339.0],
        &[948312.0, 1108791.0, 1315524.0, 1487577.0],
        &[917530.0, 1082426.0, 1484405.0],
        &[1001238.0, 1376124.0],
        &[841930.0],
    ];
    const INCURRED: [&[f64]; 10] = [
        &[
            3362115.0, 5217243.0, 4754900.0, 4381677.0, 4136883.0, 4094140.0, 4018736.0, 4001591.0,
            4001391.0, 4001258.0,
        ],
        &[
            2640443.0, 4643860.0, 3869954.0, 3248558.0, 3102002.0, 3019980.0, 2976064.0, 2966941.0,
            2959955.0,
        ],
        &[
            2879697.0, 4785531.0, 4045448.0, 3467822.0, 3377540.0, 3341934.0, 3283928.0, 3287827.0,
        ],
        &[
            2933345.0, 5299146.0, 4451963.0, 3700809.0, 3553391.0, 3469505.0, 3413921.0,
        ],
        &[
            2768181.0, 4658933.0, 3936455.0, 3512735.0, 3385129.0, 3298998.0,
        ],
        &[3228439.0, 5271304.0, 4484946.0, 3798384.0, 3702427.0],
        &[2927033.0, 5067768.0, 4066526.0, 3704113.0],
        &[3083429.0, 4790944.0, 4408097.0],
        &[2761163.0, 4132757.0],
        &[3045376.0],
    ];

    fn triangle_of(paid: &[&[f64]], incurred: &[&[f64]]) -> Triangle {
        let (mut origin, mut age, mut p, mut i) = (vec![], vec![], vec![], vec![]);
        for (o, (rp, ri)) in paid.iter().zip(incurred).enumerate() {
            for d in 0..rp.len() {
                origin.push(Month::january(2001 + o as i32));
                age.push(12 * (d as Lag + 1));
                p.push(rp[d]);
                i.push(ri[d]);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&age),
            values: &[("paid", &p), ("incurred", &i)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn dahms() -> Triangle {
        triangle_of(&PAID, &INCURRED)
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    #[test]
    fn mack_extrapolation_gives_the_papers_last_sigma() {
        // Table 4 of the tail-factor paper: sigma_0..sigma_8 estimated,
        // sigma_9 = min(sigma_8, sigma_7, sigma_8^2 / sigma_7) = 0.0227.
        let fit = PaidIncurredChain {
            extrapolation: VarianceExtrapolation::Mack,
            ..Default::default()
        }
        .fit(&dahms(), "paid", "incurred")
        .unwrap();
        let printed = [
            0.1393, 0.0650, 0.0731, 0.0640, 0.0264, 0.0271, 0.0405, 0.0227, 0.0494, 0.0227,
        ];
        for (s, p) in fit.sigma.iter().zip(printed) {
            assert!((s - p).abs() < 5e-5, "{s} against {p}");
        }
        assert_eq!(fit.sigma[9], fit.sigma[7]);
    }

    #[test]
    fn given_variances_equal_to_the_estimates_give_the_same_fit() {
        let tri = dahms();
        let est = PaidIncurredChain::default()
            .fit(&tri, "paid", "incurred")
            .unwrap();
        let given = PaidIncurredChain {
            sigma: Some(est.sigma.clone()),
            tau: Some(est.tau.clone()),
            ..Default::default()
        }
        .fit(&tri, "paid", "incurred")
        .unwrap();
        assert_eq!(est, given);
        let infinite = PaidIncurredChain {
            priors: PicPriors {
                phi_mean: vec![0.0; 10],
                phi_sd: vec![f64::INFINITY; 10],
                psi_mean: vec![0.0; 9],
                psi_sd: vec![f64::INFINITY; 9],
            },
            ..Default::default()
        }
        .fit(&tri, "paid", "incurred")
        .unwrap();
        assert_eq!(est, infinite);
    }

    /// The chain ladder on geometric averages of `column`'s link ratios.
    fn geometric_ultimates(tri: &Triangle, column: &str) -> Vec<f64> {
        ChainLadder {
            development: Development {
                average: Average::Geometric,
                ..Default::default()
            },
            ..Default::default()
        }
        .fit(tri, column)
        .unwrap()
        .ultimate
    }

    #[test]
    fn large_incurred_variances_give_hertigs_paid_chain_ladder() {
        // tau -> infinity: b_k -> 0 and the posterior of each Phi_j is
        // N(mean log link ratio, sigma_j^2 / n_j), so the ultimate is the
        // geometric paid chain ladder times exp(sum sigma^2 (1 + 1/n) / 2).
        let tri = dahms();
        let fit = PaidIncurredChain {
            tau: Some(vec![1e4; 9]),
            ..Default::default()
        }
        .fit(&tri, "paid", "incurred")
        .unwrap();
        let cl = geometric_ultimates(&tri, "paid");
        for (o, cl) in cl.iter().enumerate() {
            let k = 9 - o;
            let adj: f64 = (k + 1..10)
                .map(|m| fit.sigma[m].powi(2) * (1.0 + 1.0 / (10 - m) as f64) / 2.0)
                .sum();
            assert!(fit.incurred_weight[o] < 1e-9);
            assert!(close(fit.ultimate[o], cl * adj.exp(), 1e-8), "origin {o}");
        }
    }

    #[test]
    fn small_incurred_variances_give_the_geometric_incurred_chain_ladder() {
        let tri = dahms();
        let fit = PaidIncurredChain {
            tau: Some(vec![1e-7; 9]),
            ..Default::default()
        }
        .fit(&tri, "paid", "incurred")
        .unwrap();
        let cl = geometric_ultimates(&tri, "incurred");
        for (o, cl) in cl.iter().enumerate().skip(1) {
            assert!(fit.incurred_weight[o] > 1.0 - 1e-9);
            assert!(close(fit.ultimate[o], *cl, 1e-8), "origin {o}");
        }
    }

    #[test]
    fn a_tight_prior_fixes_the_parameters() {
        // With every parameter known, the ultimate is the closed form of
        // Merz and Wüthrich's Theorem 2.4 (no parameter uncertainty), and
        // the error is the process error alone.
        let tri = dahms();
        let phi: Vec<f64> = (0..10)
            .map(|j| if j == 0 { 13.8 } else { 0.1 / j as f64 })
            .collect();
        let psi: Vec<f64> = (0..9).map(|j| -0.05 / (j + 1) as f64).collect();
        let fit = PaidIncurredChain {
            priors: PicPriors {
                phi_mean: phi.clone(),
                phi_sd: vec![1e-9; 10],
                psi_mean: psi.clone(),
                psi_sd: vec![1e-9; 9],
            },
            ..Default::default()
        }
        .fit(&tri, "paid", "incurred")
        .unwrap();
        for (a, b) in fit.phi().iter().zip(&phi) {
            assert!((a - b).abs() < 1e-12);
        }
        let s2: Vec<f64> = fit.sigma.iter().map(|s| s * s).collect();
        let t2: Vec<f64> = fit.tau.iter().map(|t| t * t).collect();
        for o in 1..10 {
            let k = 9 - o;
            let after: f64 = s2[k + 1..].iter().sum();
            let b = after / (after + t2[k..].iter().sum::<f64>());
            let u = ((1.0 - b) * (PAID[o][k].ln() + phi[k + 1..].iter().sum::<f64>())
                + b * (INCURRED[o][k].ln() + psi[k..].iter().sum::<f64>())
                + (1.0 - b) * after / 2.0)
                .exp();
            assert!(close(fit.ultimate[o], u, 1e-9), "origin {o}");
            let process = u * ((1.0 - b) * after).exp_m1().sqrt();
            assert!(close(fit.std_err[o], process, 1e-6), "origin {o}");
        }
    }

    #[test]
    fn the_oldest_origin_is_settled() {
        let fit = PaidIncurredChain::default()
            .fit(&dahms(), "paid", "incurred")
            .unwrap();
        assert_eq!(fit.ultimate[0], PAID[0][9]);
        assert_eq!((fit.reserve[0], fit.std_err[0]), (0.0, 0.0));
        assert!(
            fit.incurred_weight[1..]
                .iter()
                .all(|b| *b > 0.0 && *b < 1.0)
        );
        let long = fit.to_long();
        assert_eq!(long.column("reserve").unwrap(), fit.reserve.as_slice());
        assert_eq!(fit.totals().column("std_err").unwrap(), [fit.total_std_err]);
        let dev = fit.development_table();
        assert_eq!(dev.n_rows(), 10);
        assert!(dev.column("tau").unwrap()[9].is_nan());
    }

    /// The posterior of `(Phi, Psi)` by conditioning on every origin's
    /// observed log paid and log incurred values directly: per origin, the
    /// vector of `xi` and `zeta` is normal with mean `theta` and diagonal
    /// variance, and each observation is a sum of its entries
    /// (`log P[i,j] = sum_{m<=j} xi_m`, `log I[i,j] = sum_m xi_m -
    /// sum_{n>=j} zeta_n`). No factorization of the likelihood is used.
    fn direct_posterior(
        paid: &[&[f64]],
        incurred: &[&[f64]],
        sigma: &[f64],
        tau: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        use prospicio_math::linalg::{cholesky, cholesky_inverse, cholesky_solve};
        let n = paid.len();
        let jj = n - 1;
        let dim = n + jj;
        let var: Vec<f64> = sigma.iter().chain(tau).map(|s| s * s).collect();
        let mut precision = vec![0.0; dim * dim];
        let mut shift = vec![0.0; dim];
        for o in 0..n {
            let k = jj - o;
            let mut rows: Vec<(Vec<f64>, f64)> = Vec::new();
            for j in 0..=k {
                let mut a = vec![0.0; dim];
                a[..=j].fill(1.0);
                rows.push((a, paid[o][j].ln()));
                // I[0,J] is P[0,J] in the model, not a separate value.
                if j < jj {
                    let mut a = vec![1.0; n];
                    a.extend((0..jj).map(|m| if m >= j { -1.0 } else { 0.0 }));
                    rows.push((a, incurred[o][j].ln()));
                }
            }
            let r = rows.len();
            let mut cov = vec![0.0; r * r];
            for (p, (ap, _)) in rows.iter().enumerate() {
                for (q, (aq, _)) in rows.iter().enumerate() {
                    cov[p * r + q] = (0..dim).map(|d| ap[d] * var[d] * aq[d]).sum();
                }
            }
            let inv = cholesky_inverse(&cholesky(&cov, r).unwrap(), r);
            for p in 0..r {
                for q in 0..r {
                    let w = inv[p * r + q];
                    for a in 0..dim {
                        shift[a] += rows[p].0[a] * w * rows[q].1;
                        for b in 0..dim {
                            precision[a * dim + b] += rows[p].0[a] * w * rows[q].0[b];
                        }
                    }
                }
            }
        }
        let l = cholesky(&precision, dim).unwrap();
        (cholesky_solve(&l, &shift), cholesky_inverse(&l, dim))
    }

    #[test]
    fn the_posterior_is_the_models_and_r_leaves_out_a_coupling() {
        // The model has I[0,J] = P[0,J]; give the data that.
        let mut row0 = INCURRED[0].to_vec();
        row0[9] = PAID[0][9];
        let mut incurred: Vec<&[f64]> = INCURRED.to_vec();
        incurred[0] = &row0;
        let tri = triangle_of(&PAID, &incurred);
        let fit = PaidIncurredChain::default()
            .fit(&tri, "paid", "incurred")
            .unwrap();
        let (mean, cov) = direct_posterior(&PAID, &incurred, &fit.sigma, &fit.tau);
        for (a, b) in fit.posterior_mean.iter().zip(&mean) {
            assert!((a - b).abs() < 1e-10, "{a} against {b}");
        }
        for (a, b) in fit.posterior_covariance.iter().zip(&cov) {
            assert!(
                (a - b).abs() < 1e-12 * cov[0].abs().max(1.0),
                "{a} against {b}"
            );
        }
        // R's convention differs from it in the posterior means of Phi_1,
        // ..., Phi_J and Psi, not in Phi_0 nor in the variances.
        let r = PaidIncurredChain {
            convention: PicConvention::RChainLadder,
            ..Default::default()
        }
        .fit(&tri, "paid", "incurred")
        .unwrap();
        assert_eq!(
            (r.sigma.clone(), r.tau.clone()),
            (fit.sigma.clone(), fit.tau.clone())
        );
        assert!((r.phi()[0] - fit.phi()[0]).abs() < 1e-12);
        assert!((r.phi()[1] - fit.phi()[1]).abs() > 1e-4);
        assert!((r.total_reserve() - fit.total_reserve()).abs() > 1.0);
    }

    #[test]
    fn bad_input_is_an_error() {
        let tri = dahms();
        let pic = PaidIncurredChain::default();
        assert!(matches!(
            pic.fit(&tri, "paid", "missing"),
            Err(Error::UnknownColumn(_))
        ));
        // A row short of the diagonal.
        let mut short: Vec<&[f64]> = PAID.to_vec();
        short[3] = &PAID[3][..6];
        assert!(matches!(
            pic.fit(&triangle_of(&short, &INCURRED), "paid", "incurred"),
            Err(Error::PaidIncurred(_))
        ));
        // Not square: one origin fewer.
        assert!(matches!(
            pic.fit(&triangle_of(&PAID[..9], &INCURRED[..9]), "paid", "incurred"),
            Err(Error::PaidIncurred(_))
        ));
        // Wrong lengths and values of the settings.
        for bad in [
            PaidIncurredChain {
                sigma: Some(vec![0.1; 9]),
                ..Default::default()
            },
            PaidIncurredChain {
                tau: Some(vec![0.0; 9]),
                ..Default::default()
            },
            PaidIncurredChain {
                priors: PicPriors {
                    phi_mean: vec![0.0; 10],
                    phi_sd: vec![-1.0; 10],
                    ..Default::default()
                },
                ..Default::default()
            },
            PaidIncurredChain {
                priors: PicPriors {
                    psi_mean: vec![0.0; 3],
                    psi_sd: vec![1.0; 3],
                    ..Default::default()
                },
                ..Default::default()
            },
        ] {
            assert!(matches!(
                bad.fit(&tri, "paid", "incurred"),
                Err(Error::PaidIncurred(_))
            ));
        }
    }
}
