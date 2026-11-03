//! Zero-inflated and hurdle count models, as R's `pscl::zeroinfl` and
//! `pscl::hurdle` (`docs/design/models.md`, "Zero-inflated and hurdle
//! counts").
//!
//! Both are two-part models for counts with more zeros than a Poisson or
//! negative binomial predicts. Each part has its own [`Design`]: the count
//! part a log link `μ = exp(xᵀβ + offset)`, the zero part a binary link
//! `h` (logit, probit or cloglog) on `zᵀγ + offset`.
//!
//! - **Zero-inflated** ([`ZeroInflated`]): a zero is structural with
//!   probability `π = h(zᵀγ)`, otherwise the count is drawn from `f(·; μ)`,
//!   so `P(0) = π + (1 − π) f(0)` and `P(k) = (1 − π) f(k)` for `k > 0`.
//!   The parameters are estimated jointly by maximum likelihood.
//! - **Hurdle** ([`Hurdle`]): the zero part gives `p = h(zᵀγ) = P(Y > 0)`,
//!   and a positive count is drawn from `f` truncated at zero, so
//!   `P(0) = 1 − p` and `P(k) = p f(k) / (1 − f(0))`. The two parts'
//!   likelihoods separate and are maximised one at a time.
//!
//! `f` is the Poisson or the negative binomial with variance `μ + μ²/θ`,
//! `θ` estimated on the log scale as pscl does. The maximum is found by
//! Newton's method on analytic gradients, the Hessian by central
//! differences of the gradient, each step damped until the Hessian is
//! negative definite and halved until the log-likelihood rises. Standard
//! errors come from the observed information at the maximum, as pscl's
//! (which uses optim's finite-difference Hessian), so they include the
//! parameters' correlation across the parts and with `log θ`.
//!
//! [`vuong`] is Vuong's non-nested test of two models from their pointwise
//! log-likelihoods, with pscl's AIC and BIC corrections.

use prospicio_core::{Error, Result};
use prospicio_math::linalg::{cholesky, cholesky_inverse, cholesky_solve, lower_mul};
use prospicio_math::special::{digamma, ln_gamma, norm_cdf, norm_quantile};
use prospicio_models::{Design, Family, Fitted, Link, Model};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution, Provenance};

use crate::Glm;

/// The count distribution of a two-part model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountDistribution {
    /// `f(k) = μᵏ e^{−μ} / k!`.
    Poisson,
    /// The negative binomial with variance `μ + μ²/θ`, `θ` estimated.
    NegativeBinomial,
}

impl CountDistribution {
    /// `"poisson"` or `"negative_binomial"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Poisson => "poisson",
            Self::NegativeBinomial => "negative_binomial",
        }
    }
}

/// Which two-part model a [`TwoPartFit`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoPart {
    /// Structural zeros mixed with a count model ([`ZeroInflated`]).
    ZeroInflated,
    /// A binary hurdle and a zero-truncated count model ([`Hurdle`]).
    Hurdle,
}

impl TwoPart {
    /// `"zero_inflated"` or `"hurdle"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::ZeroInflated => "zero_inflated",
            Self::Hurdle => "hurdle",
        }
    }
}

/// A zero-inflated Poisson or negative binomial model.
///
/// [`fit_parts`](Self::fit_parts) takes the count and zero parts' designs
/// (pscl's `y ~ x | z`); [`Model::fit`] uses one design for both, as
/// pscl's `y ~ x`, with the design's offset on the count part only.
///
/// ```
/// use prospicio_glm::zero::ZeroInflated;
/// use prospicio_models::{Design, Fitted, Model};
///
/// let x: Vec<f64> = (0..200).map(|i| f64::from(i % 4) / 3.0).collect();
/// // Counts with extra zeros: every third row is a structural zero.
/// let y: Vec<f64> = (0..200)
///     .map(|i| if i % 3 == 0 { 0.0 } else { f64::from((i * 7) % 5) })
///     .collect();
/// let d = Design::new(vec!["(Intercept)".into(), "x".into()], vec![vec![1.0; 200], x]).unwrap();
/// let fit = ZeroInflated::poisson().fit(&d, &y).unwrap();
/// assert!(fit.zero_coefficients()[0] > -2.0);
/// let mean = fit.predict(&d).unwrap();
/// assert!((mean.iter().sum::<f64>() - y.iter().sum::<f64>()).abs() < 1e-4 * 200.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroInflated {
    pub count: CountDistribution,
    /// The zero part's link: logit (the default), probit or cloglog.
    pub zero_link: Link,
    /// Newton stops when its next step moves no parameter by more than
    /// this times `max(1, |parameter|)`.
    pub tolerance: f64,
    pub max_iterations: usize,
}

/// A hurdle model: a binary part for `Y > 0` and a zero-truncated Poisson
/// or negative binomial for the positive counts.
///
/// The zero part's coefficients model `P(Y > 0)`, as pscl's. Designs are
/// taken as by [`ZeroInflated`].
///
/// ```
/// use prospicio_glm::zero::Hurdle;
/// use prospicio_models::{Design, Fitted, Model};
///
/// let x: Vec<f64> = (0..200).map(|i| f64::from(i % 4) / 3.0).collect();
/// let y: Vec<f64> = (0..200)
///     .map(|i| if i % 3 == 0 { 0.0 } else { f64::from(1 + (i * 7) % 5) })
///     .collect();
/// let d = Design::new(vec!["(Intercept)".into(), "x".into()], vec![vec![1.0; 200], x]).unwrap();
/// let fit = Hurdle::poisson().fit(&d, &y).unwrap();
/// // The hurdle reproduces the share of zeros exactly, as a logit GLM does.
/// let p0: f64 = fit.prob_zero(&d).unwrap().iter().sum();
/// assert!((p0 - y.iter().filter(|&&v| v == 0.0).count() as f64).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hurdle {
    pub count: CountDistribution,
    /// The binary part's link for `P(Y > 0)`: logit (the default), probit
    /// or cloglog.
    pub zero_link: Link,
    /// As [`ZeroInflated::tolerance`], for each part.
    pub tolerance: f64,
    pub max_iterations: usize,
}

macro_rules! spec_builders {
    ($t:ty) => {
        impl $t {
            /// With a Poisson count part and a logit zero part.
            pub fn poisson() -> Self {
                Self::new(CountDistribution::Poisson)
            }

            /// With a negative binomial count part (`θ` estimated) and a
            /// logit zero part.
            pub fn negative_binomial() -> Self {
                Self::new(CountDistribution::NegativeBinomial)
            }

            /// With the given count part, a logit zero part, tolerance
            /// `1e-10` and at most 200 Newton steps.
            pub fn new(count: CountDistribution) -> Self {
                Self {
                    count,
                    zero_link: Link::Logit,
                    tolerance: 1e-10,
                    max_iterations: 200,
                }
            }

            /// The same model with the zero part's link set to `link`.
            pub fn zero_link(mut self, link: Link) -> Self {
                self.zero_link = link;
                self
            }
        }

        impl Model for $t {
            type Fitted = TwoPartFit;

            /// Both parts on `design`'s columns, the offset on the count
            /// part only; see [`fit_parts`](Self::fit_parts).
            fn fit(&self, design: &Design, y: &[f64]) -> Result<TwoPartFit> {
                let zero = without_offset(design)?;
                self.fit_parts(design, &zero, y)
            }
        }
    };
}

spec_builders!(ZeroInflated);
spec_builders!(Hurdle);

/// `design`'s columns and weights with a zero offset.
fn without_offset(design: &Design) -> Result<Design> {
    let columns = (0..design.n_cols())
        .map(|j| design.column(j).to_vec())
        .collect();
    Design::new(design.names().to_vec(), columns)?.with_weights(design.weights().to_vec())
}

/// Adds to a failed negative binomial fit the likely reason.
fn explain(dist: CountDistribution, error: Error) -> Error {
    match (dist, error) {
        (CountDistribution::NegativeBinomial, Error::Data(message)) => Error::Data(format!(
            "{message}; with a negative binomial count part, theta grows without bound when the \
             counts are not overdispersed: use the Poisson"
        )),
        (_, error) => error,
    }
}

fn check_spec(zero_link: Link, max_iterations: usize) -> Result<()> {
    if !matches!(zero_link, Link::Logit | Link::Probit | Link::Cloglog) {
        return Err(Error::Data(format!(
            "the zero part needs a logit, probit or cloglog link, not {zero_link:?}"
        )));
    }
    if max_iterations == 0 {
        return Err(Error::Data("max_iterations must be at least 1".into()));
    }
    Ok(())
}

/// Checks the response and the two designs, and returns the weights.
fn check_data(count: &Design, zero: &Design, y: &[f64]) -> Result<Vec<f64>> {
    let n = y.len();
    if count.n_rows() != n || zero.n_rows() != n {
        return Err(Error::Data(format!(
            "the response has {n} rows, the count design {} and the zero design {}",
            count.n_rows(),
            zero.n_rows()
        )));
    }
    if let Some(&bad) = y.iter().find(|&&v| !(v >= 0.0 && v.fract() == 0.0)) {
        return Err(Error::InvalidParameter {
            name: "y",
            value: bad,
            reason: "a count model needs whole, non-negative counts",
        });
    }
    let w = count.weights();
    let wz = zero.weights();
    if wz.iter().any(|&v| v != 1.0) && wz != w {
        return Err(Error::Data(
            "the zero design's weights differ from the count design's; set them on the count \
             design"
                .into(),
        ));
    }
    if !y.iter().zip(w).any(|(&v, &wi)| v == 0.0 && wi > 0.0) {
        return Err(Error::Data(
            "the response has no zeros, so the zero part cannot be estimated".into(),
        ));
    }
    if !y.iter().zip(w).any(|(&v, &wi)| v > 0.0 && wi > 0.0) {
        return Err(Error::Data(
            "the response has no positive counts, so the count part cannot be estimated".into(),
        ));
    }
    Ok(w.to_vec())
}

/// `ln f(y)` and its derivatives in `η = ln μ` and `ln θ`.
#[derive(Clone, Copy)]
struct Count {
    dist: CountDistribution,
    theta: f64,
}

impl Count {
    fn ln_pmf(self, y: f64, mu: f64) -> f64 {
        match self.dist {
            CountDistribution::Poisson => {
                if y == 0.0 {
                    -mu
                } else {
                    y * mu.ln() - mu - ln_gamma(y + 1.0)
                }
            }
            CountDistribution::NegativeBinomial => {
                let t = self.theta;
                let base = -t * (mu / t).ln_1p();
                if y == 0.0 {
                    base
                } else {
                    ln_gamma(y + t) - ln_gamma(t) - ln_gamma(y + 1.0)
                        + base
                        + y * (mu.ln() - (t + mu).ln())
                }
            }
        }
    }

    /// `(∂ ln f(y)/∂η, ∂ ln f(y)/∂ ln θ)`.
    fn d_ln_pmf(self, y: f64, mu: f64) -> (f64, f64) {
        match self.dist {
            CountDistribution::Poisson => (y - mu, 0.0),
            CountDistribution::NegativeBinomial => {
                let t = self.theta;
                let d_eta = t * (y - mu) / (t + mu);
                let ratio = -(mu / t).ln_1p();
                let d_theta = if y == 0.0 {
                    ratio + mu / (t + mu)
                } else {
                    digamma(y + t) - digamma(t) + ratio + (mu - y) / (t + mu)
                };
                (d_eta, t * d_theta)
            }
        }
    }

    /// The count drawn at uniform `u`, by inversion.
    fn draw(self, mu: f64, u: f64) -> f64 {
        let family = match self.dist {
            CountDistribution::Poisson => Family::Poisson,
            CountDistribution::NegativeBinomial => Family::NegativeBinomial { theta: self.theta },
        };
        family.draw(mu, 1.0, 1.0, u).unwrap_or(f64::NAN)
    }
}

/// The parameter vector `(β, γ, ln θ)` split into the parts' predictors.
struct Layout<'a> {
    count: &'a Design,
    zero: &'a Design,
    dist: CountDistribution,
    zero_link: Link,
}

impl Layout<'_> {
    fn kx(&self) -> usize {
        self.count.n_cols()
    }
    fn kz(&self) -> usize {
        self.zero.n_cols()
    }
    fn n_par(&self) -> usize {
        self.kx() + self.kz() + usize::from(self.dist == CountDistribution::NegativeBinomial)
    }
    fn count_of(&self, par: &[f64]) -> Count {
        Count {
            dist: self.dist,
            theta: match self.dist {
                CountDistribution::Poisson => f64::NAN,
                CountDistribution::NegativeBinomial => par[self.kx() + self.kz()].exp(),
            },
        }
    }
    /// `(μ, η_z)` per row.
    fn predictors(&self, par: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (kx, kz) = (self.kx(), self.kz());
        let mu = self
            .count
            .linear_predictor(&par[..kx])
            .into_iter()
            .map(f64::exp)
            .collect();
        (mu, self.zero.linear_predictor(&par[kx..kx + kz]))
    }

    /// Adds `scale` times the derivatives in `η_c`, `η_z` and `ln θ` of row
    /// `i` to the gradient `g`.
    fn accumulate(&self, g: &mut [f64], i: usize, d_eta: f64, d_zeta: f64, d_theta: f64) {
        let (kx, kz) = (self.kx(), self.kz());
        for (j, gj) in g[..kx].iter_mut().enumerate() {
            *gj += d_eta * self.count.column(j)[i];
        }
        for (j, gj) in g[kx..kx + kz].iter_mut().enumerate() {
            *gj += d_zeta * self.zero.column(j)[i];
        }
        if self.dist == CountDistribution::NegativeBinomial {
            g[kx + kz] += d_theta;
        }
    }

    /// The zero-inflated log-likelihood's pointwise values and the
    /// weighted total's gradient.
    fn zero_inflated(&self, par: &[f64], y: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let count = self.count_of(par);
        let (mu, zeta) = self.predictors(par);
        let mut g = vec![0.0; self.n_par()];
        let mut ll = vec![0.0; y.len()];
        for i in 0..y.len() {
            let pi = self.zero_link.inverse(zeta[i]);
            let a = self.zero_link.mu_eta(zeta[i]);
            if y[i] == 0.0 {
                let lf0 = count.ln_pmf(0.0, mu[i]);
                let f0 = lf0.exp();
                let p0 = pi + (1.0 - pi) * f0;
                ll[i] = p0.ln();
                if w[i] != 0.0 {
                    let (de, dt) = count.d_ln_pmf(0.0, mu[i]);
                    let share = (1.0 - pi) * f0 / p0;
                    let dz = a * (1.0 - f0) / p0;
                    self.accumulate(&mut g, i, w[i] * share * de, w[i] * dz, w[i] * share * dt);
                }
            } else {
                ll[i] = (1.0 - pi).ln() + count.ln_pmf(y[i], mu[i]);
                if w[i] != 0.0 {
                    let (de, dt) = count.d_ln_pmf(y[i], mu[i]);
                    self.accumulate(&mut g, i, w[i] * de, -w[i] * a / (1.0 - pi), w[i] * dt);
                }
            }
        }
        (ll, g)
    }
}

/// The zero-truncated count part of a hurdle: `ln f(y) − ln(1 − f(0))` on
/// the positive rows, parameters `(β, ln θ)`.
fn truncated(
    design: &Design,
    dist: CountDistribution,
    par: &[f64],
    y: &[f64],
    w: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let kx = design.n_cols();
    let count = Count {
        dist,
        theta: match dist {
            CountDistribution::Poisson => f64::NAN,
            CountDistribution::NegativeBinomial => par[kx].exp(),
        },
    };
    let eta = design.linear_predictor(&par[..kx]);
    let mut g = vec![0.0; par.len()];
    let mut ll = vec![0.0; y.len()];
    for i in 0..y.len() {
        let mu = eta[i].exp();
        let lf0 = count.ln_pmf(0.0, mu);
        // f0 / (1 − f0) = 1 / (e^{−ln f0} − 1).
        let odds = 1.0 / (-lf0).exp_m1();
        ll[i] = count.ln_pmf(y[i], mu) - (-lf0.exp_m1()).ln();
        if w[i] != 0.0 {
            let (de, dt) = count.d_ln_pmf(y[i], mu);
            let (de0, dt0) = count.d_ln_pmf(0.0, mu);
            let d_eta = w[i] * (de + odds * de0);
            for (j, gj) in g[..kx].iter_mut().enumerate() {
                *gj += d_eta * design.column(j)[i];
            }
            if dist == CountDistribution::NegativeBinomial {
                g[kx] += w[i] * (dt + odds * dt0);
            }
        }
    }
    (ll, g)
}

/// The hurdle's binary part: `s ln p + (1 − s) ln(1 − p)` with `s = 1` for
/// a positive count.
fn binary(
    design: &Design,
    link: Link,
    par: &[f64],
    positive: &[bool],
    w: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let eta = design.linear_predictor(par);
    let mut g = vec![0.0; par.len()];
    let mut ll = vec![0.0; positive.len()];
    for i in 0..positive.len() {
        let p = link.inverse(eta[i]);
        let a = link.mu_eta(eta[i]);
        let (l, d) = if positive[i] {
            (p.ln(), a / p)
        } else {
            ((1.0 - p).ln(), -a / (1.0 - p))
        };
        ll[i] = l;
        if w[i] != 0.0 {
            for (j, gj) in g.iter_mut().enumerate() {
                *gj += w[i] * d * design.column(j)[i];
            }
        }
    }
    (ll, g)
}

/// The maximum of a log-likelihood.
struct Maximum {
    par: Vec<f64>,
    /// The pointwise log-likelihoods at `par`.
    pointwise: Vec<f64>,
    log_likelihood: f64,
    /// The inverse of the observed information, row-major.
    covariance: Vec<f64>,
    iterations: usize,
}

/// The observed information `−H` by central differences of the gradient,
/// symmetrised.
fn information(par: &[f64], grad: &impl Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
    let p = par.len();
    let mut h = vec![0.0; p * p];
    let mut x = par.to_vec();
    for j in 0..p {
        let step = 1e-5 * par[j].abs().max(1.0);
        x[j] = par[j] + step;
        let up = grad(&x);
        x[j] = par[j] - step;
        let down = grad(&x);
        x[j] = par[j];
        for k in 0..p {
            h[k * p + j] = -(up[k] - down[k]) / (2.0 * step);
        }
    }
    for j in 0..p {
        for k in 0..j {
            let m = 0.5 * (h[j * p + k] + h[k * p + j]);
            h[j * p + k] = m;
            h[k * p + j] = m;
        }
    }
    h
}

/// Newton's method on `ln L = Σ w ℓᵢ` from `start`: each step solves
/// `(−H + λ I) d = g` with `λ` raised until the matrix is positive
/// definite, and is halved until the log-likelihood does not fall.
fn maximize(
    start: Vec<f64>,
    w: &[f64],
    eval: impl Fn(&[f64]) -> (Vec<f64>, Vec<f64>),
    tolerance: f64,
    max_iterations: usize,
    what: &str,
) -> Result<Maximum> {
    let total = |ll: &[f64]| -> f64 {
        ll.iter()
            .zip(w)
            .filter(|(_, wi)| **wi != 0.0)
            .map(|(l, wi)| wi * l)
            .sum()
    };
    let grad = |x: &[f64]| eval(x).1;
    let p = start.len();
    let mut par = start;
    let (mut pointwise, mut g) = eval(&par);
    let mut ll = total(&pointwise);
    if !ll.is_finite() {
        return Err(Error::Data(format!(
            "the {what} log-likelihood is not finite at the start"
        )));
    }
    for iteration in 1..=max_iterations {
        let info = information(&par, &grad);
        let scale = (0..p)
            .map(|j| info[j * p + j].abs())
            .fold(0.0, f64::max)
            .max(1e-300);
        let mut lambda = 0.0;
        let factor = loop {
            let mut a = info.clone();
            for j in 0..p {
                a[j * p + j] += lambda;
            }
            if let Some(l) = cholesky(&a, p) {
                break l;
            }
            lambda = if lambda == 0.0 {
                1e-8 * scale
            } else {
                lambda * 10.0
            };
            if lambda > 1e12 * scale {
                return Err(Error::Data(format!(
                    "the {what} information cannot be made positive definite"
                )));
            }
        };
        let step = cholesky_solve(&factor, &g);
        let largest = step
            .iter()
            .zip(&par)
            .map(|(s, x)| s.abs() / x.abs().max(1.0))
            .fold(0.0, f64::max);
        if lambda == 0.0 && largest <= tolerance {
            let factor = cholesky(&info, p).ok_or_else(|| {
                Error::Data(format!("the {what} observed information is singular"))
            })?;
            return Ok(Maximum {
                par,
                pointwise,
                log_likelihood: ll,
                covariance: cholesky_inverse(&factor, p),
                iterations: iteration,
            });
        }
        let mut t = 1.0;
        let mut accepted = false;
        for _ in 0..60 {
            let next: Vec<f64> = par.iter().zip(&step).map(|(x, s)| x + t * s).collect();
            let (next_pw, next_g) = eval(&next);
            let next_ll = total(&next_pw);
            if next_ll.is_finite() && next_ll >= ll - 1e-12 * ll.abs().max(1.0) {
                par = next;
                pointwise = next_pw;
                g = next_g;
                ll = next_ll;
                accepted = true;
                break;
            }
            t /= 2.0;
        }
        if !accepted {
            return Err(Error::Data(format!(
                "the {what} Newton step found no rise in the log-likelihood"
            )));
        }
    }
    Err(Error::Data(format!(
        "the {what} maximum was not reached in {max_iterations} Newton steps"
    )))
}

impl ZeroInflated {
    /// Fits the count part on `count` (its offset the exposure's log) and
    /// the zero part on `zero` by joint maximum likelihood. Weights are the
    /// count design's.
    ///
    /// Starts from the Poisson GLM on `count`, the binary GLM of `y = 0` on
    /// `zero` with the zero link, and `θ = 1`. Fails if `y` is not a count,
    /// has no zeros or no positive counts, or the maximum is not found.
    pub fn fit_parts(&self, count: &Design, zero: &Design, y: &[f64]) -> Result<TwoPartFit> {
        check_spec(self.zero_link, self.max_iterations)?;
        let w = check_data(count, zero, y)?;
        let layout = Layout {
            count,
            zero,
            dist: self.count,
            zero_link: self.zero_link,
        };
        let poisson = Glm::new(Family::Poisson, Link::Log).fit(count, y)?;
        let is_zero: Vec<f64> = y.iter().map(|&v| f64::from(u8::from(v == 0.0))).collect();
        let zero_weighted = without_offset(zero)?
            .with_offset(zero.offset().to_vec())?
            .with_weights(w.clone())?;
        let binary = Glm::new(Family::Binomial, self.zero_link).fit(&zero_weighted, &is_zero)?;
        let mut start = poisson.coefficients().to_vec();
        start.extend_from_slice(binary.coefficients());
        if self.count == CountDistribution::NegativeBinomial {
            start.push(0.0);
        }
        let max = maximize(
            start,
            &w,
            |par| layout.zero_inflated(par, y, &w),
            self.tolerance,
            self.max_iterations,
            "zero-inflated",
        )
        .map_err(|e| explain(self.count, e))?;
        Ok(TwoPartFit::new(
            TwoPart::ZeroInflated,
            &layout,
            max.par,
            max.covariance,
            max.pointwise,
            max.log_likelihood,
            max.iterations,
            &w,
        ))
    }
}

impl Hurdle {
    /// Fits the binary part on `zero` (for `P(Y > 0)`) and the
    /// zero-truncated count part on the positive rows of `count`, each by
    /// maximum likelihood. Weights are the count design's.
    ///
    /// The binary part starts from its GLM; the count part from the Poisson
    /// GLM on the positive rows and `θ = 1`. Fails as
    /// [`ZeroInflated::fit_parts`].
    pub fn fit_parts(&self, count: &Design, zero: &Design, y: &[f64]) -> Result<TwoPartFit> {
        check_spec(self.zero_link, self.max_iterations)?;
        let w = check_data(count, zero, y)?;
        let layout = Layout {
            count,
            zero,
            dist: self.count,
            zero_link: self.zero_link,
        };
        let (kx, kz) = (count.n_cols(), zero.n_cols());
        let positive: Vec<bool> = y.iter().map(|&v| v > 0.0).collect();
        let zero_weighted = without_offset(zero)?
            .with_offset(zero.offset().to_vec())?
            .with_weights(w.clone())?;
        let indicator: Vec<f64> = positive.iter().map(|&s| f64::from(u8::from(s))).collect();
        let start = Glm::new(Family::Binomial, self.zero_link).fit(&zero_weighted, &indicator)?;
        let binary_max = maximize(
            start.coefficients().to_vec(),
            &w,
            |par| binary(zero, self.zero_link, par, &positive, &w),
            self.tolerance,
            self.max_iterations,
            "hurdle's binary",
        )?;
        let rows: Vec<usize> = (0..y.len()).filter(|&i| positive[i]).collect();
        let sub = count.select(&rows);
        let y_pos: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
        let w_pos: Vec<f64> = rows.iter().map(|&i| w[i]).collect();
        let mut start = Glm::new(Family::Poisson, Link::Log)
            .fit(&sub, &y_pos)?
            .coefficients()
            .to_vec();
        if self.count == CountDistribution::NegativeBinomial {
            start.push(0.0);
        }
        let count_max = maximize(
            start,
            &w_pos,
            |par| truncated(&sub, self.count, par, &y_pos, &w_pos),
            self.tolerance,
            self.max_iterations,
            "hurdle's truncated count",
        )
        .map_err(|e| explain(self.count, e))?;
        // (β, γ, ln θ), the two blocks independent.
        let mut par = count_max.par[..kx].to_vec();
        par.extend_from_slice(&binary_max.par);
        let nb = self.count == CountDistribution::NegativeBinomial;
        if nb {
            par.push(count_max.par[kx]);
        }
        let n_par = par.len();
        let count_index = |a: usize| if a < kx { a } else { kx + kz };
        let mut covariance = vec![0.0; n_par * n_par];
        let pc = count_max.par.len();
        for a in 0..pc {
            for b in 0..pc {
                covariance[count_index(a) * n_par + count_index(b)] =
                    count_max.covariance[a * pc + b];
            }
        }
        for a in 0..kz {
            for b in 0..kz {
                covariance[(kx + a) * n_par + kx + b] = binary_max.covariance[a * kz + b];
            }
        }
        let mut pointwise = binary_max.pointwise;
        for (k, &i) in rows.iter().enumerate() {
            pointwise[i] += count_max.pointwise[k];
        }
        Ok(TwoPartFit::new(
            TwoPart::Hurdle,
            &layout,
            par,
            covariance,
            pointwise,
            binary_max.log_likelihood + count_max.log_likelihood,
            binary_max.iterations + count_max.iterations,
            &w,
        ))
    }
}

/// A fitted zero-inflated or hurdle model, from [`ZeroInflated`] or
/// [`Hurdle`].
#[derive(Debug, Clone, PartialEq)]
pub struct TwoPartFit {
    kind: TwoPart,
    count: CountDistribution,
    zero_link: Link,
    count_names: Vec<String>,
    zero_names: Vec<String>,
    /// `(β, γ, ln θ)`.
    par: Vec<f64>,
    covariance: Vec<f64>,
    pointwise: Vec<f64>,
    log_likelihood: f64,
    n_obs: usize,
    iterations: usize,
    fitted: Vec<f64>,
}

impl TwoPartFit {
    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: TwoPart,
        layout: &Layout<'_>,
        par: Vec<f64>,
        covariance: Vec<f64>,
        pointwise: Vec<f64>,
        log_likelihood: f64,
        iterations: usize,
        w: &[f64],
    ) -> Self {
        let mut fit = Self {
            kind,
            count: layout.dist,
            zero_link: layout.zero_link,
            count_names: layout.count.names().to_vec(),
            zero_names: layout.zero.names().to_vec(),
            par,
            covariance,
            pointwise,
            log_likelihood,
            n_obs: w.iter().filter(|&&v| v > 0.0).count(),
            iterations,
            fitted: Vec::new(),
        };
        fit.fitted = fit
            .rows(layout.count, layout.zero)
            .map(|(mu, zeta)| fit.mean_at(&fit.par, mu, zeta))
            .collect();
        fit
    }

    /// Zero-inflated or hurdle.
    pub fn kind(&self) -> TwoPart {
        self.kind
    }

    /// The count distribution.
    pub fn count_distribution(&self) -> CountDistribution {
        self.count
    }

    /// The zero part's link.
    pub fn zero_link(&self) -> Link {
        self.zero_link
    }

    fn kx(&self) -> usize {
        self.count_names.len()
    }

    fn kz(&self) -> usize {
        self.zero_names.len()
    }

    /// The count design's column names.
    pub fn count_names(&self) -> &[String] {
        &self.count_names
    }

    /// The count part's coefficients `β`, on the log scale.
    pub fn count_coefficients(&self) -> &[f64] {
        &self.par[..self.kx()]
    }

    /// The zero design's column names.
    pub fn zero_names(&self) -> &[String] {
        &self.zero_names
    }

    /// The zero part's coefficients `γ`: of the structural zero's
    /// probability for a zero-inflated model, of `P(Y > 0)` for a hurdle.
    pub fn zero_coefficients(&self) -> &[f64] {
        &self.par[self.kx()..self.kx() + self.kz()]
    }

    /// The negative binomial's `θ`, `None` for the Poisson.
    pub fn theta(&self) -> Option<f64> {
        (self.count == CountDistribution::NegativeBinomial)
            .then(|| self.par[self.kx() + self.kz()].exp())
    }

    /// The standard error of `ln θ`, as pscl's `SE.logtheta`.
    pub fn se_log_theta(&self) -> Option<f64> {
        let p = self.par.len();
        self.theta()
            .map(|_| self.covariance[(p - 1) * p + p - 1].sqrt())
    }

    /// The covariance of `(β, γ, ln θ)` from the observed information,
    /// row-major; a hurdle's two parts are uncorrelated.
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    fn standard_errors(&self) -> Vec<f64> {
        let p = self.par.len();
        (0..p).map(|j| self.covariance[j * p + j].sqrt()).collect()
    }

    /// The standard errors of [`count_coefficients`](Self::count_coefficients).
    pub fn count_standard_errors(&self) -> Vec<f64> {
        self.standard_errors()[..self.kx()].to_vec()
    }

    /// The standard errors of [`zero_coefficients`](Self::zero_coefficients).
    pub fn zero_standard_errors(&self) -> Vec<f64> {
        self.standard_errors()[self.kx()..self.kx() + self.kz()].to_vec()
    }

    /// The maximised log-likelihood `Σ w ℓᵢ`.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Each row's log-likelihood `ℓᵢ`, without its weight: the input of
    /// [`vuong`].
    pub fn log_likelihoods(&self) -> &[f64] {
        &self.pointwise
    }

    /// The number of estimated parameters: both parts' coefficients and,
    /// for the negative binomial, `θ`.
    pub fn n_parameters(&self) -> usize {
        self.par.len()
    }

    /// `−2 ℓ + 2 k`, as R's `AIC` on a pscl fit.
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * self.par.len() as f64
    }

    /// `−2 ℓ + k ln n`, `n` the rows with a positive weight.
    pub fn bic(&self) -> f64 {
        -2.0 * self.log_likelihood + self.par.len() as f64 * (self.n_obs as f64).ln()
    }

    /// Newton steps used (both parts' for a hurdle).
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The fitted means on the data fitted.
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// `(μ, η_z)` for each row of the two designs at the estimates.
    fn rows<'a>(
        &'a self,
        count: &'a Design,
        zero: &'a Design,
    ) -> impl Iterator<Item = (f64, f64)> + 'a {
        let mu = count.linear_predictor(self.count_coefficients());
        let zeta = zero.linear_predictor(self.zero_coefficients());
        mu.into_iter().map(f64::exp).zip(zeta)
    }

    fn count_at(&self, par: &[f64]) -> Count {
        Count {
            dist: self.count,
            theta: match self.count {
                CountDistribution::Poisson => f64::NAN,
                CountDistribution::NegativeBinomial => par[self.kx() + self.kz()].exp(),
            },
        }
    }

    fn mean_at(&self, par: &[f64], mu: f64, zeta: f64) -> f64 {
        let h = self.zero_link.inverse(zeta);
        match self.kind {
            TwoPart::ZeroInflated => (1.0 - h) * mu,
            TwoPart::Hurdle => {
                let lf0 = self.count_at(par).ln_pmf(0.0, mu);
                h * mu / -lf0.exp_m1()
            }
        }
    }

    /// `P(Y = k)` for `k = 0..=max` at one row.
    fn probabilities_at(&self, mu: f64, zeta: f64, max: usize) -> Vec<f64> {
        let h = self.zero_link.inverse(zeta);
        let count = self.count_at(&self.par);
        let lf0 = count.ln_pmf(0.0, mu);
        (0..=max)
            .map(|k| {
                let f = count.ln_pmf(k as f64, mu).exp();
                match (self.kind, k) {
                    (TwoPart::ZeroInflated, 0) => h + (1.0 - h) * f,
                    (TwoPart::ZeroInflated, _) => (1.0 - h) * f,
                    (TwoPart::Hurdle, 0) => 1.0 - h,
                    (TwoPart::Hurdle, _) => h * f / -lf0.exp_m1(),
                }
            })
            .collect()
    }

    fn check_parts(&self, count: &Design, zero: &Design) -> Result<()> {
        if count.names() != self.count_names.as_slice() {
            return Err(Error::Data(format!(
                "the count design's columns {:?} are not the fit's {:?}",
                count.names(),
                self.count_names
            )));
        }
        if zero.names() != self.zero_names.as_slice() {
            return Err(Error::Data(format!(
                "the zero design's columns {:?} are not the fit's {:?}",
                zero.names(),
                self.zero_names
            )));
        }
        if count.n_rows() != zero.n_rows() {
            return Err(Error::Data(format!(
                "the count design has {} rows and the zero design {}",
                count.n_rows(),
                zero.n_rows()
            )));
        }
        Ok(())
    }

    /// The expected count of each row: `(1 − π) μ` zero-inflated,
    /// `p μ / (1 − f(0))` hurdle.
    pub fn predict_parts(&self, count: &Design, zero: &Design) -> Result<Vec<f64>> {
        self.check_parts(count, zero)?;
        Ok(self
            .rows(count, zero)
            .map(|(mu, zeta)| self.mean_at(&self.par, mu, zeta))
            .collect())
    }

    /// `P(Y = 0)` for each row.
    pub fn prob_zero_parts(&self, count: &Design, zero: &Design) -> Result<Vec<f64>> {
        self.check_parts(count, zero)?;
        Ok(self
            .rows(count, zero)
            .map(|(mu, zeta)| self.probabilities_at(mu, zeta, 0)[0])
            .collect())
    }

    /// `P(Y = k)` for `k = 0..=max`, one vector per row.
    pub fn probabilities_parts(
        &self,
        count: &Design,
        zero: &Design,
        max: usize,
    ) -> Result<Vec<Vec<f64>>> {
        self.check_parts(count, zero)?;
        Ok(self
            .rows(count, zero)
            .map(|(mu, zeta)| self.probabilities_at(mu, zeta, max))
            .collect())
    }

    /// The two parts' designs read by name from one design: each part takes
    /// the columns named as its coefficients, the count part the offset.
    pub fn split(&self, design: &Design) -> Result<(Design, Design)> {
        let pick = |names: &[String], what: &str| -> Result<Design> {
            let columns = names
                .iter()
                .map(|name| {
                    design
                        .names()
                        .iter()
                        .position(|n| n == name)
                        .map(|j| design.column(j).to_vec())
                        .ok_or_else(|| {
                            Error::Data(format!(
                                "the design has no column {name:?} of the {what} part"
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            Design::new(names.to_vec(), columns)
        };
        let count = pick(&self.count_names, "count")?.with_offset(design.offset().to_vec())?;
        Ok((count, pick(&self.zero_names, "zero")?))
    }

    /// [`prob_zero_parts`](Self::prob_zero_parts) on one design, read as
    /// [`split`](Self::split) reads it.
    pub fn prob_zero(&self, design: &Design) -> Result<Vec<f64>> {
        let (c, z) = self.split(design)?;
        self.prob_zero_parts(&c, &z)
    }

    /// [`probabilities_parts`](Self::probabilities_parts) on one design.
    pub fn probabilities(&self, design: &Design, max: usize) -> Result<Vec<Vec<f64>>> {
        let (c, z) = self.split(design)?;
        self.probabilities_parts(&c, &z, max)
    }

    /// Joint draws of the counts on the two designs: the parameters
    /// `(β, γ, ln θ)` drawn from the normal approximation at the maximum
    /// (a hurdle's two parts independently), then each row's count from the
    /// model at them.
    pub fn predict_distribution_parts(
        &self,
        count: &Design,
        zero: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        self.check_parts(count, zero)?;
        let p = self.par.len();
        let factor = cholesky(&self.covariance, p)
            .ok_or_else(|| Error::Data("the parameter covariance is singular".into()))?;
        let n = count.n_rows();
        let components: Vec<ComponentKey> =
            (0..n).map(|i| vec![KeyValue::from(i as i64)]).collect();
        let provenance = Provenance::new("two_part")
            .version("prospicio-glm", env!("CARGO_PKG_VERSION"))
            .param("kind", self.kind.name())
            .param("count", self.count.name())
            .param("zero_link", format!("{:?}", self.zero_link));
        let (kx, kz) = (self.kx(), self.kz());
        PredictiveDistribution::simulate(
            vec!["row".into()],
            components,
            n_sims,
            seed,
            provenance,
            |rng, row| {
                let z: Vec<f64> = (0..p).map(|_| norm_quantile(rng.next_open01())).collect();
                let mut shift = vec![0.0; p];
                lower_mul(&factor, &z, &mut shift);
                let par: Vec<f64> = self.par.iter().zip(&shift).map(|(a, b)| a + b).collect();
                let model = self.count_at(&par);
                let eta = count.linear_predictor(&par[..kx]);
                let zeta = zero.linear_predictor(&par[kx..kx + kz]);
                for (i, out) in row.iter_mut().enumerate() {
                    let mu = eta[i].exp();
                    let h = self.zero_link.inverse(zeta[i]);
                    let (u, v) = (rng.next_open01(), rng.next_open01());
                    *out = match self.kind {
                        TwoPart::ZeroInflated if u < h => 0.0,
                        TwoPart::ZeroInflated => model.draw(mu, v),
                        TwoPart::Hurdle if u >= h => 0.0,
                        TwoPart::Hurdle => {
                            let f0 = model.ln_pmf(0.0, mu).exp();
                            model.draw(mu, f0 + v * (1.0 - f0)).max(1.0)
                        }
                    };
                }
            },
        )
    }
}

impl Fitted for TwoPartFit {
    /// [`predict_parts`](Self::predict_parts) on one design, read as
    /// [`split`](Self::split) reads it.
    fn predict(&self, design: &Design) -> Result<Vec<f64>> {
        let (c, z) = self.split(design)?;
        self.predict_parts(&c, &z)
    }

    /// [`predict_distribution_parts`](Self::predict_distribution_parts) on
    /// one design.
    fn predict_distribution(
        &self,
        design: &Design,
        n_sims: usize,
        seed: u64,
    ) -> Result<PredictiveDistribution> {
        let (c, z) = self.split(design)?;
        self.predict_distribution_parts(&c, &z, n_sims, seed)
    }
}

/// Each row's log-likelihood under a plain Poisson (`theta` `None`) or
/// negative binomial count model with means `mu`: the other input of
/// [`vuong`] when a two-part model is tested against a GLM.
pub fn count_log_likelihoods(y: &[f64], mu: &[f64], theta: Option<f64>) -> Result<Vec<f64>> {
    if y.len() != mu.len() {
        return Err(Error::Data(format!(
            "{} responses and {} means",
            y.len(),
            mu.len()
        )));
    }
    let count = match theta {
        None => Count {
            dist: CountDistribution::Poisson,
            theta: f64::NAN,
        },
        Some(t) if t.is_finite() && t > 0.0 => Count {
            dist: CountDistribution::NegativeBinomial,
            theta: t,
        },
        Some(t) => {
            return Err(Error::InvalidParameter {
                name: "theta",
                value: t,
                reason: "must be finite and positive",
            });
        }
    };
    Ok(y.iter()
        .zip(mu)
        .map(|(&v, &m)| count.ln_pmf(v, m))
        .collect())
}

/// One of Vuong's statistics with its one-sided p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VuongStatistic {
    /// `z`: positive favours the first model.
    pub statistic: f64,
    /// `1 − Φ(z)` for a positive `z`, `Φ(z)` otherwise, as pscl's.
    pub p_value: f64,
}

/// Vuong's test of two non-nested models, as `pscl::vuong`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vuong {
    /// `Σ m / (s √n)`, `m` the pointwise log-likelihood differences and
    /// `s` their standard deviation.
    pub raw: VuongStatistic,
    /// With `Σ m` less `k_a − k_b`.
    pub aic: VuongStatistic,
    /// With `Σ m` less `(k_a − k_b) ln(n) / 2`.
    pub bic: VuongStatistic,
}

/// Vuong's test of model `a` against model `b` from each row's
/// log-likelihood under each and their parameter counts.
///
/// `pscl::vuong` counts `length(coef(m))`, the coefficients without the
/// negative binomial's `θ`; pass the same counts to reproduce it. Rows
/// where either log-likelihood is not finite are dropped, as pscl drops
/// them.
///
/// ```
/// use prospicio_glm::zero::vuong;
///
/// let a = [-1.0, -0.5, -2.0, -0.8];
/// let b = [-1.2, -0.9, -2.1, -0.7];
/// let v = vuong(&a, &b, 3, 2).unwrap();
/// assert!(v.raw.statistic > 0.0 && v.aic.statistic < v.raw.statistic);
/// ```
pub fn vuong(a: &[f64], b: &[f64], k_a: usize, k_b: usize) -> Result<Vuong> {
    if a.len() != b.len() {
        return Err(Error::Data(format!(
            "{} and {} pointwise log-likelihoods",
            a.len(),
            b.len()
        )));
    }
    let m: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| x - y)
        .filter(|d| d.is_finite())
        .collect();
    let n = m.len() as f64;
    if m.len() < 2 {
        return Err(Error::Data("Vuong's test needs at least two rows".into()));
    }
    let mean = m.iter().sum::<f64>() / n;
    let s = (m.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if s == 0.0 {
        return Err(Error::Data(
            "the two models' log-likelihoods differ by a constant: Vuong's statistic is undefined"
                .into(),
        ));
    }
    let dk = k_a as f64 - k_b as f64;
    let stat = |penalty: f64| {
        let z = (n * mean - penalty) / (s * n.sqrt());
        VuongStatistic {
            statistic: z,
            p_value: if z > 0.0 {
                1.0 - norm_cdf(z)
            } else {
                norm_cdf(z)
            },
        }
    };
    Ok(Vuong {
        raw: stat(0.0),
        aic: stat(dk),
        bic: stat(dk * n.ln() / 2.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A log-likelihood's pointwise values and gradient.
    type Eval<'a> = &'a dyn Fn(&[f64]) -> (Vec<f64>, Vec<f64>);

    fn book() -> (Design, Vec<f64>) {
        let n = 300;
        let x: Vec<f64> = (0..n).map(|i| f64::from(i % 7) / 6.0 - 0.5).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i % 4 == 0 {
                    0.0
                } else {
                    // Roughly uniform on 0..=12: variance 14 against mean 6,
                    // overdispersed for the negative binomial.
                    f64::from((i * i * 31 + 7) % 13)
                }
            })
            .collect();
        let d = Design::new(
            vec!["(Intercept)".into(), "x".into()],
            vec![vec![1.0; n as usize], x],
        )
        .unwrap();
        (d, y)
    }

    /// The analytic gradient against central differences of the
    /// log-likelihood, for every model and link.
    #[test]
    fn gradients_match_finite_differences() {
        let (d, y) = book();
        let w: Vec<f64> = (0..y.len()).map(|i| 1.0 + (i % 3) as f64).collect();
        let total = |ll: &[f64]| ll.iter().zip(&w).map(|(l, wi)| l * wi).sum::<f64>();
        let check = |par: &[f64], eval: Eval<'_>| {
            let g = eval(par).1;
            for j in 0..par.len() {
                let h = 1e-6;
                let mut up = par.to_vec();
                up[j] += h;
                let mut down = par.to_vec();
                down[j] -= h;
                let fd = (total(&eval(&up).0) - total(&eval(&down).0)) / (2.0 * h);
                assert!(
                    (fd - g[j]).abs() < 1e-5 * fd.abs().max(1.0),
                    "{j}: {fd} vs {}",
                    g[j]
                );
            }
        };
        for link in [Link::Logit, Link::Probit, Link::Cloglog] {
            for dist in [
                CountDistribution::Poisson,
                CountDistribution::NegativeBinomial,
            ] {
                let layout = Layout {
                    count: &d,
                    zero: &d,
                    dist,
                    zero_link: link,
                };
                let par = [0.3, -0.2, -0.7, 0.4, 0.5];
                let par = &par[..layout.n_par()];
                check(par, &|p| layout.zero_inflated(p, &y, &w));
                let pos: Vec<f64> = y.iter().map(|&v| v.max(1.0)).collect();
                check(
                    &par[..2 + usize::from(dist == CountDistribution::NegativeBinomial)],
                    &|p| truncated(&d, dist, p, &pos, &w),
                );
                let positive: Vec<bool> = y.iter().map(|&v| v > 0.0).collect();
                check(&[-0.4, 0.8], &|p| binary(&d, link, p, &positive, &w));
            }
        }
    }

    /// Probabilities sum to 1 over the support and their mean is the
    /// predicted mean.
    #[test]
    fn probabilities_sum_to_one_and_give_the_mean() {
        let (d, y) = book();
        for fit in [
            ZeroInflated::negative_binomial().fit(&d, &y).unwrap(),
            Hurdle::negative_binomial().fit(&d, &y).unwrap(),
            Hurdle::poisson()
                .zero_link(Link::Probit)
                .fit(&d, &y)
                .unwrap(),
        ] {
            let probs = fit.probabilities(&d, 200).unwrap();
            let mean = fit.predict(&d).unwrap();
            for (p, m) in probs.iter().zip(&mean) {
                let total: f64 = p.iter().sum();
                let first: f64 = p.iter().enumerate().map(|(k, q)| k as f64 * q).sum();
                assert!((total - 1.0).abs() < 1e-12, "{total}");
                assert!((first - m).abs() < 1e-10 * m, "{first} vs {m}");
            }
        }
    }

    /// A hurdle's binary part is the binary GLM's maximum, and its count
    /// part under the logit hurdle reproduces the positive rows' total.
    #[test]
    fn the_hurdle_binary_part_is_the_glm() {
        let (d, y) = book();
        let fit = Hurdle::poisson().fit(&d, &y).unwrap();
        let s: Vec<f64> = y.iter().map(|&v| f64::from(u8::from(v > 0.0))).collect();
        let glm = Glm::new(Family::Binomial, Link::Logit).fit(&d, &s).unwrap();
        for (a, b) in fit.zero_coefficients().iter().zip(glm.coefficients()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
        for (a, b) in fit.zero_standard_errors().iter().zip(glm.std_errors()) {
            assert!((a - b).abs() < 1e-6 * b, "{a} vs {b}");
        }
    }

    #[test]
    fn predictive_draws_centre_on_the_mean() {
        let (d, y) = book();
        for fit in [
            ZeroInflated::poisson().fit(&d, &y).unwrap(),
            Hurdle::negative_binomial().fit(&d, &y).unwrap(),
        ] {
            let pd = fit.predict_distribution(&d, 4000, 7).unwrap();
            let total: f64 = fit.predict(&d).unwrap().iter().sum();
            let draws = pd.total();
            use prospicio_prob::Distribution;
            let sd = draws.std_dev();
            assert!((draws.mean() - total).abs() < 4.0 * sd / 4000f64.sqrt() + 0.01 * total);
        }
    }

    #[test]
    fn it_rejects_what_it_cannot_fit() {
        let (d, y) = book();
        assert!(
            ZeroInflated::poisson()
                .zero_link(Link::Log)
                .fit(&d, &y)
                .is_err()
        );
        let mut bad = y.clone();
        bad[1] = 1.5;
        assert!(ZeroInflated::poisson().fit(&d, &bad).is_err());
        let positive: Vec<f64> = y.iter().map(|&v| v + 1.0).collect();
        assert!(Hurdle::poisson().fit(&d, &positive).is_err());
        let zeros = vec![0.0; y.len()];
        assert!(Hurdle::poisson().fit(&d, &zeros).is_err());
        let fit = ZeroInflated::poisson().fit(&d, &y).unwrap();
        let other = Design::new(vec!["(Intercept)".into()], vec![vec![1.0; 3]]).unwrap();
        assert!(fit.predict(&other).is_err());
    }

    #[test]
    fn vuong_agrees_with_its_definition() {
        let a = [-1.0, -0.5, -2.0, -0.8, -1.1];
        let b = [-1.2, -0.9, -2.1, -0.7, -1.0];
        let v = vuong(&a, &b, 3, 2).unwrap();
        let m: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        let mean = m.iter().sum::<f64>() / 5.0;
        let s = (m.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!((v.raw.statistic - mean * 5f64.sqrt() / s).abs() < 1e-14);
        assert!((v.aic.statistic - (5.0 * mean - 1.0) / (s * 5f64.sqrt())).abs() < 1e-14);
        assert!(vuong(&a, &a, 1, 1).is_err());
    }
}
