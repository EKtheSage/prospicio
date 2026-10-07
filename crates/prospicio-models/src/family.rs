//! Exponential dispersion families: what a GLM, a neural network's loss, a
//! booster's objective and a Bayesian likelihood all share.

use prospicio_core::{Error, Result};
use prospicio_math::roots::bisect;
use prospicio_math::special::{ln_gamma, norm_cdf, norm_quantile};
use prospicio_prob::{Binomial, Counting, Distribution, Gamma, NegativeBinomial, Poisson, Tweedie};

use crate::link::Link;

/// A response family: variance function `V(μ)`, unit deviance and
/// log-likelihood, for a response with mean `μ`, dispersion `φ` and prior
/// weight `w` (variance `φ V(μ) / w`).
///
/// A closed enum, like [`Link`], so a fitted model serializes as data. Each
/// family names the `prospicio-prob` distribution it predicts, which
/// [`Family::draw`] samples from.
///
/// | Family | `V(μ)` | Response | Dispersion |
/// |---|---|---|---|
/// | `Gaussian` | 1 | Normal | estimated |
/// | `Poisson` | `μ` | counts or rates; `φ ≠ 1` is the over-dispersed (quasi-) Poisson | 1, or estimated |
/// | `Gamma` | `μ²` | [`Gamma`] severities | estimated |
/// | `InverseGaussian` | `μ³` | inverse Gaussian | estimated |
/// | `Binomial` | `μ(1 - μ)` | proportions of `w` trials | 1 |
/// | `NegativeBinomial { theta }` | `μ + μ²/θ` | [`NegativeBinomial`] counts | 1 |
/// | `Tweedie { power }` | `μ^p`, `1 < p < 2` | [`Tweedie`] pure premium | estimated |
///
/// ```
/// use prospicio_models::Family;
///
/// let f = Family::Poisson;
/// assert_eq!(f.variance(3.0), 3.0);
/// // The unit deviance is 0 at a perfect fit.
/// assert_eq!(f.unit_deviance(3.0, 3.0), 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Family {
    Gaussian,
    Poisson,
    Gamma,
    InverseGaussian,
    Binomial,
    NegativeBinomial { theta: f64 },
    Tweedie { power: f64 },
}

impl Family {
    /// Short name: `"gaussian"`, `"poisson"`, `"gamma"`,
    /// `"inverse_gaussian"`, `"binomial"`, `"negative_binomial"`,
    /// `"tweedie"`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Poisson => "poisson",
            Self::Gamma => "gamma",
            Self::InverseGaussian => "inverse_gaussian",
            Self::Binomial => "binomial",
            Self::NegativeBinomial { .. } => "negative_binomial",
            Self::Tweedie { .. } => "tweedie",
        }
    }

    /// Checks the family's own parameter.
    pub fn validate(&self) -> Result<()> {
        match *self {
            Self::NegativeBinomial { theta } if !(theta.is_finite() && theta > 0.0) => {
                Err(invalid("theta", theta, "must be finite and positive"))
            }
            Self::Tweedie { power } if !(power > 1.0 && power < 2.0) => {
                Err(invalid("power", power, "must be in (1, 2)"))
            }
            _ => Ok(()),
        }
    }

    /// The canonical link, where the family's natural parameter is linear.
    pub fn canonical_link(&self) -> Link {
        match *self {
            Self::Gaussian => Link::Identity,
            Self::Poisson | Self::NegativeBinomial { .. } => Link::Log,
            Self::Gamma => Link::Inverse,
            Self::InverseGaussian => Link::InverseSquared,
            Self::Binomial => Link::Logit,
            Self::Tweedie { power } => Link::Power(1.0 - power),
        }
    }

    /// Whether the dispersion is 1 by definition (Binomial, negative
    /// binomial; the Poisson unless over-dispersion is estimated).
    pub fn unit_dispersion(&self) -> bool {
        matches!(
            self,
            Self::Poisson | Self::Binomial | Self::NegativeBinomial { .. }
        )
    }

    /// Variance function `V(μ)`.
    pub fn variance(&self, mu: f64) -> f64 {
        match *self {
            Self::Gaussian => 1.0,
            Self::Poisson => mu,
            Self::Gamma => mu * mu,
            Self::InverseGaussian => mu * mu * mu,
            Self::Binomial => mu * (1.0 - mu),
            Self::NegativeBinomial { theta } => mu + mu * mu / theta,
            Self::Tweedie { power } => mu.powf(power),
        }
    }

    /// Derivative of the variance function, `V'(μ)`.
    pub fn variance_deriv(&self, mu: f64) -> f64 {
        match *self {
            Self::Gaussian => 0.0,
            Self::Poisson => 1.0,
            Self::Gamma => 2.0 * mu,
            Self::InverseGaussian => 3.0 * mu * mu,
            Self::Binomial => 1.0 - 2.0 * mu,
            Self::NegativeBinomial { theta } => 1.0 + 2.0 * mu / theta,
            Self::Tweedie { power } => power * mu.powf(power - 1.0),
        }
    }

    /// Whether `y` is a possible response.
    pub fn valid_y(&self, y: f64) -> bool {
        y.is_finite()
            && match self {
                Self::Gaussian => true,
                Self::Poisson | Self::NegativeBinomial { .. } | Self::Tweedie { .. } => y >= 0.0,
                Self::Gamma | Self::InverseGaussian => y > 0.0,
                Self::Binomial => (0.0..=1.0).contains(&y),
            }
    }

    /// Whether `μ` is a possible mean.
    pub fn valid_mu(&self, mu: f64) -> bool {
        mu.is_finite()
            && match self {
                Self::Gaussian => true,
                Self::Binomial => mu > 0.0 && mu < 1.0,
                _ => mu > 0.0,
            }
    }

    /// Starting mean for IRLS: `(y + ȳ) / 2`, and for the binomial
    /// `(w y + 1/2) / (w + 1)`, as statsmodels and R start.
    pub fn initial_mu(&self, y: f64, weight: f64, y_mean: f64) -> f64 {
        match self {
            Self::Binomial => (weight * y + 0.5) / (weight + 1.0),
            _ => 0.5 * (y + y_mean),
        }
    }

    /// Unit deviance `d(y, μ)`; the deviance is `Σ w d(y, μ)`.
    ///
    /// For the over-dispersed Poisson a response may be negative (a
    /// negative incremental loss): there `d = 2 (y ln(|y|/μ) - (y - μ))`,
    /// the quasi-deviance. It has the Poisson's derivative in `μ`, so it
    /// gives the same quasi-likelihood estimates, but no saturated model
    /// exists and it can be negative; it is not a distance.
    pub fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        if matches!(self, Self::Poisson) && y < 0.0 {
            return 2.0 * (y * (-y / mu).ln() - (y - mu));
        }
        // y ln(y / μ), 0 at y = 0.
        let ylog = |y: f64, m: f64| if y == 0.0 { 0.0 } else { y * (y / m).ln() };
        let d = match *self {
            Self::Gaussian => (y - mu) * (y - mu),
            Self::Poisson => 2.0 * (ylog(y, mu) - (y - mu)),
            Self::Gamma => 2.0 * ((y - mu) / mu - (y / mu).ln()),
            Self::InverseGaussian => (y - mu) * (y - mu) / (mu * mu * y),
            Self::Binomial => 2.0 * (ylog(y, mu) + ylog(1.0 - y, 1.0 - mu)),
            Self::NegativeBinomial { theta } => {
                2.0 * (ylog(y, mu) - (y + theta) * ((y + theta) / (mu + theta)).ln())
            }
            Self::Tweedie { power: p } => {
                let first = if y == 0.0 {
                    0.0
                } else {
                    y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p))
                };
                2.0 * (first - y * mu.powf(1.0 - p) / (1.0 - p) + mu.powf(2.0 - p) / (2.0 - p))
            }
        };
        d.max(0.0)
    }

    /// Log-likelihood of one observation with prior weight `w` and
    /// dispersion `φ`, as statsmodels defines it with `var_weights`
    /// (for the binomial, `w` is the number of trials and `y` the observed
    /// proportion).
    pub fn log_likelihood(&self, y: f64, mu: f64, weight: f64, dispersion: f64) -> f64 {
        let (w, phi) = (weight, dispersion);
        match *self {
            Self::Gaussian => {
                -0.5 * (w * (y - mu) * (y - mu) / phi + (2.0 * std::f64::consts::PI * phi / w).ln())
            }
            // No likelihood exists for a negative (quasi-Poisson) response.
            Self::Poisson if y < 0.0 => f64::NAN,
            Self::Poisson => w / phi * (y * mu.ln() - mu - ln_gamma(y + 1.0)),
            Self::Gamma => {
                let s = w / phi;
                let r = y / mu;
                s * (s * r).ln() - s * r - y.ln() - ln_gamma(s)
            }
            Self::InverseGaussian => {
                -0.5 * (w * (y - mu) * (y - mu) / (phi * y * mu * mu)
                    + (phi * y * y * y / w).ln()
                    + (2.0 * std::f64::consts::PI).ln())
            }
            Self::Binomial => {
                let k = w * y;
                let ln_choose = ln_gamma(w + 1.0) - ln_gamma(k + 1.0) - ln_gamma(w - k + 1.0);
                let term = |x: f64, p: f64| if x == 0.0 { 0.0 } else { x * p.ln() };
                ln_choose + term(k, mu) + term(w - k, 1.0 - mu)
            }
            Self::NegativeBinomial { theta } => {
                let ll = ln_gamma(y + theta) - ln_gamma(theta) - ln_gamma(y + 1.0)
                    + theta * (theta / (theta + mu)).ln()
                    + if y == 0.0 {
                        0.0
                    } else {
                        y * (mu / (theta + mu)).ln()
                    };
                w / phi * ll
            }
            Self::Tweedie { power } => match Tweedie::new(mu, phi / w, power) {
                Ok(t) => t.ln_pdf(y),
                Err(_) => f64::NAN,
            },
        }
    }

    /// A draw of the response with mean `μ`, dispersion `φ` and weight
    /// `w`, by inverse transform of the uniform `u` in `(0, 1)`: the
    /// family's process noise, for predictive distributions.
    ///
    /// - Poisson: `(φ/w) N` with `N ~ Poisson(μ w / φ)`, so variance
    ///   `φ μ / w` (the over-dispersed Poisson when `φ ≠ 1`);
    /// - Binomial: `N / w` with `N ~ Binomial(w, μ)` (`w` rounded);
    /// - negative binomial: `NegativeBinomial(θ, μ/θ)`;
    /// - gamma: shape `w/φ`, scale `μφ/w`;
    /// - Tweedie: `Tweedie(μ, φ/w, p)`;
    /// - Gaussian: `μ + √(φ/w) Φ⁻¹(u)`;
    /// - inverse Gaussian: shape `λ = w/φ`, by bisection on its
    ///   distribution function.
    pub fn draw(&self, mu: f64, dispersion: f64, weight: f64, u: f64) -> Result<f64> {
        let (w, phi) = (weight, dispersion);
        match *self {
            Self::Gaussian => Ok(mu + (phi / w).sqrt() * norm_quantile(u)),
            Self::Poisson => {
                let n = Poisson::new(mu * w / phi)?.quantile(u)?;
                Ok(phi / w * n as f64)
            }
            Self::Binomial => {
                let trials = w.round().max(1.0);
                let n = Binomial::new(trials as u64, mu)?.quantile(u)?;
                Ok(n as f64 / trials)
            }
            Self::NegativeBinomial { theta } => {
                Ok(NegativeBinomial::new(theta, mu / theta)?.quantile(u)? as f64)
            }
            Self::Gamma => Gamma::new(w / phi, mu * phi / w)?.quantile(u),
            Self::Tweedie { power } => Tweedie::new(mu, phi / w, power)?.quantile(u),
            Self::InverseGaussian => {
                let lambda = w / phi;
                let cdf = |x: f64| inverse_gaussian_cdf(x, mu, lambda);
                let mut hi = mu * 2.0;
                while cdf(hi) < u {
                    hi *= 2.0;
                }
                Ok(bisect(0.0, hi, |x| cdf(x) < u))
            }
        }
    }
}

impl Family {
    /// `(P(Y < y), P(Y ≤ y))` for the response with mean `μ`, dispersion
    /// `φ` and weight `w`, under the distribution [`draw`](Self::draw)
    /// samples. The two differ only where `Y` has an atom (the counts, and
    /// the Tweedie at 0); a randomized PIT draws between them.
    ///
    /// ```
    /// use prospicio_models::Family;
    ///
    /// // Poisson(2): P(Y < 1) = e^-2, P(Y ≤ 1) = 3 e^-2.
    /// let (lo, hi) = Family::Poisson.cdf_bounds(1.0, 2.0, 1.0, 1.0).unwrap();
    /// assert!((lo - (-2f64).exp()).abs() < 1e-15 && (hi - 3.0 * (-2f64).exp()).abs() < 1e-15);
    /// ```
    pub fn cdf_bounds(&self, y: f64, mu: f64, dispersion: f64, weight: f64) -> Result<(f64, f64)> {
        let (w, phi) = (weight, dispersion);
        // For a count `k` on a scaled lattice: below and at it.
        fn lattice(c: &dyn Counting, k: f64) -> (f64, f64) {
            if k < 0.0 {
                return (0.0, 0.0);
            }
            let (lo, hi) = (k.ceil(), k.floor());
            let below = if lo >= 1.0 { c.cdf(lo as u64 - 1) } else { 0.0 };
            (below, c.cdf(hi as u64))
        }
        let continuous = |f: f64| (f, f);
        Ok(match *self {
            Self::Gaussian => continuous(norm_cdf((y - mu) / (phi / w).sqrt())),
            Self::Poisson => lattice(&Poisson::new(mu * w / phi)?, snap(y * w / phi)),
            Self::Binomial => {
                let trials = w.round().max(1.0);
                lattice(&Binomial::new(trials as u64, mu)?, snap(y * trials))
            }
            Self::NegativeBinomial { theta } => {
                lattice(&NegativeBinomial::new(theta, mu / theta)?, snap(y))
            }
            Self::Gamma => continuous(Gamma::new(w / phi, mu * phi / w)?.cdf(y)),
            Self::InverseGaussian => continuous(inverse_gaussian_cdf(y, mu, w / phi)),
            Self::Tweedie { power } => {
                let t = Tweedie::new(mu, phi / w, power)?;
                if y < 0.0 {
                    (0.0, 0.0)
                } else if y == 0.0 {
                    (0.0, t.cdf(0.0))
                } else {
                    continuous(t.cdf(y))
                }
            }
        })
    }

    /// The log density (log probability for the counts) of `y` under the
    /// distribution [`draw`](Self::draw) samples: the log score is its
    /// negative. It equals [`log_likelihood`](Self::log_likelihood) for the
    /// continuous families, and for the counts when `φ = 1`. A count off
    /// its lattice has log density `-∞`.
    ///
    /// The over-dispersed Poisson (`φ ≠ 1`) is scored by a density in `y`,
    /// so that it compares with continuous models (stacking, log scores
    /// on claim amounts): the Poisson probability of `N = w y / φ`,
    /// continued to real `N` through `Γ`, normalized to integrate to 1, and
    /// scaled by `w / φ`:
    /// `f(y) = (w/φ) λᴺ e^(-λ) / (Γ(N + 1) C(λ))` with `λ = w μ / φ` and
    /// `C(λ) = ∫₀^∞ λˣ e^(-λ) / Γ(x + 1) dx` by quadrature (`C ≈ 1` unless
    /// `λ` is small). On the lattice it is the count's probability times
    /// `w / C φ`; [`draw`](Self::draw) still samples the lattice. A
    /// negative `y` has log density `-∞`.
    pub fn log_density(&self, y: f64, mu: f64, dispersion: f64, weight: f64) -> Result<f64> {
        let (w, phi) = (weight, dispersion);
        let ln_pmf = |c: &dyn Counting, k: f64| -> f64 {
            if k < 0.0 || k.fract() != 0.0 {
                f64::NEG_INFINITY
            } else {
                c.pmf(k as u64).ln()
            }
        };
        Ok(match *self {
            Self::Poisson if phi != 1.0 => {
                let lambda = mu * w / phi;
                Poisson::new(lambda)?;
                if y < 0.0 {
                    f64::NEG_INFINITY
                } else {
                    let n = y * w / phi;
                    n * lambda.ln()
                        - lambda
                        - ln_gamma(n + 1.0)
                        - continuous_poisson_mass(lambda).ln()
                        + (w / phi).ln()
                }
            }
            Self::Poisson => ln_pmf(&Poisson::new(mu * w / phi)?, snap(y * w / phi)),
            Self::Binomial => {
                let trials = w.round().max(1.0);
                ln_pmf(&Binomial::new(trials as u64, mu)?, snap(y * trials))
            }
            Self::NegativeBinomial { theta } => {
                ln_pmf(&NegativeBinomial::new(theta, mu / theta)?, snap(y))
            }
            _ => self.log_likelihood(y, mu, w, phi),
        })
    }
}

/// `C(λ) = ∫₀^∞ λˣ e^(-λ) / Γ(x + 1) dx`, the mass of the Poisson
/// probability continued to real counts.
///
/// By Euler–Maclaurin `C` differs from the lattice's mass 1 only through
/// terms in `e^(-λ)` at 0 (the periodic remainder is of order
/// `e^(-2π²λ)`), so from `λ = 40` on `C = 1` to within `4e-15` and is
/// taken as 1. Below, 8-point Gauss–Legendre panels half a unit or a
/// quarter of a standard deviation wide cover `λ ± 40 √λ + 40`, beyond
/// which the integrand is below `1e-300` of its peak.
pub(crate) fn continuous_poisson_mass(lambda: f64) -> f64 {
    if lambda >= 40.0 {
        return 1.0;
    }
    let sd = lambda.sqrt();
    let (lo, hi) = (
        (lambda - 40.0 * sd - 40.0).max(0.0),
        lambda + 40.0 * sd + 40.0,
    );
    let width = (0.25 * sd).max(0.5);
    let panels = ((hi - lo) / width).ceil().max(1.0) as usize;
    let h = (hi - lo) / panels as f64;
    let ln_lambda = lambda.ln();
    let g = |x: f64| Ok::<_, ()>((x * ln_lambda - lambda - ln_gamma(x + 1.0)).exp());
    (0..panels)
        .map(|k| {
            let a = lo + k as f64 * h;
            prospicio_math::integrate::gauss_legendre(g, a, a + h).unwrap_or(0.0)
        })
        .sum()
}

/// `x` rounded when within `1e-9` (relative) of a whole number, so that
/// counts recovered from scaled responses land on the lattice.
fn snap(x: f64) -> f64 {
    let r = x.round();
    if (x - r).abs() <= 1e-9 * r.abs().max(1.0) {
        r
    } else {
        x
    }
}

/// `P(X <= x)` for an inverse Gaussian with mean `μ` and shape `λ`.
fn inverse_gaussian_cdf(x: f64, mu: f64, lambda: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let r = (lambda / x).sqrt();
    let a = norm_cdf(r * (x / mu - 1.0));
    // e^(2λ/μ) Φ(-r(x/μ + 1)), in logs so neither factor overflows.
    let b = norm_cdf(-r * (x / mu + 1.0));
    let second = if b > 0.0 {
        (2.0 * lambda / mu + b.ln()).exp()
    } else {
        0.0
    };
    (a + second).min(1.0)
}

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_over_dispersed_poisson_density_integrates_to_one() {
        // Independent check: the density integrated over y on a fine grid.
        for (mu, phi, w) in [
            (0.3, 2.0, 1.0),
            (2.5, 3.0, 1.0),
            (30.0, 4.0, 2.0),
            (400.0, 7.0, 1.0),
        ] {
            let f = |y: f64| Family::Poisson.log_density(y, mu, phi, w).unwrap().exp();
            let hi = mu * 3.0 + 60.0 * phi;
            let panels = 4000;
            let h = hi / panels as f64;
            let total: f64 = (0..panels)
                .map(|k| {
                    prospicio_math::integrate::gauss_legendre(
                        |y| Ok::<_, ()>(f(y)),
                        k as f64 * h,
                        (k + 1) as f64 * h,
                    )
                    .unwrap()
                })
                .sum();
            assert!((total - 1.0).abs() < 1e-9, "{mu} {phi} {w}: {total}");
        }
    }

    #[test]
    fn the_over_dispersed_poisson_density_matches_the_lattice() {
        // On the lattice y = kφ/w, for λ ≥ 40, the density is the count's
        // probability times w/φ.
        let (mu, phi, w) = (300.0, 5.0, 1.0);
        for k in [40u64, 60, 75] {
            let y = k as f64 * phi / w;
            let got = Family::Poisson.log_density(y, mu, phi, w).unwrap();
            let pmf = Poisson::new(mu * w / phi).unwrap().pmf(k).ln();
            assert!((got - (pmf + (w / phi).ln())).abs() < 1e-10, "{k}");
        }
        // Off the lattice it is finite; negative responses are not in it.
        assert!(
            Family::Poisson
                .log_density(301.7, mu, phi, w)
                .unwrap()
                .is_finite()
        );
        assert_eq!(
            Family::Poisson.log_density(-3.0, mu, phi, w).unwrap(),
            f64::NEG_INFINITY
        );
        // A Poisson with φ = 1 keeps its probabilities.
        assert_eq!(
            Family::Poisson.log_density(1.5, 2.0, 1.0, 1.0).unwrap(),
            f64::NEG_INFINITY
        );
        assert!((continuous_poisson_mass(1.0) - 0.833_811_448_088_410_6).abs() < 1e-12);
    }

    use prospicio_core::StreamRng;

    const FAMILIES: [Family; 7] = [
        Family::Gaussian,
        Family::Poisson,
        Family::Gamma,
        Family::InverseGaussian,
        Family::Binomial,
        Family::NegativeBinomial { theta: 2.5 },
        Family::Tweedie { power: 1.5 },
    ];

    fn sample_point(f: Family) -> (f64, f64) {
        match f {
            Family::Binomial => (0.3, 0.4),
            _ => (2.0, 1.4),
        }
    }

    #[test]
    fn deviance_is_twice_the_log_likelihood_gap() {
        // d(y, μ) = 2 φ (ℓ(y; y) - ℓ(y; μ)) / w for every family.
        for f in FAMILIES {
            let (y, mu) = sample_point(f);
            let (w, phi) = (if f == Family::Binomial { 1.0 } else { 3.0 }, 0.7);
            let phi = if f.unit_dispersion() { 1.0 } else { phi };
            if matches!(f, Family::Tweedie { .. }) {
                // Its likelihood is a series that the deviance does not
                // reduce to; the slope test below covers it.
                continue;
            }
            let gap = f.log_likelihood(y, y, w, phi) - f.log_likelihood(y, mu, w, phi);
            let want = 2.0 * phi * gap / w;
            assert!((f.unit_deviance(y, mu) - want).abs() < 1e-12, "{f:?}");
        }
    }

    #[test]
    fn variance_derivative_matches_a_difference() {
        for f in FAMILIES {
            let (_, mu) = sample_point(f);
            let h = 1e-6;
            let numeric = (f.variance(mu + h) - f.variance(mu - h)) / (2.0 * h);
            assert!((f.variance_deriv(mu) - numeric).abs() < 1e-8, "{f:?}");
        }
    }

    #[test]
    fn deviance_slope_is_the_score() {
        // ∂d/∂μ = -2 (y - μ) / V(μ) for every exponential dispersion family.
        for f in FAMILIES {
            let (y, mu) = sample_point(f);
            let h = 1e-6;
            let slope = (f.unit_deviance(y, mu + h) - f.unit_deviance(y, mu - h)) / (2.0 * h);
            let want = -2.0 * (y - mu) / f.variance(mu);
            assert!((slope - want).abs() < 1e-6, "{f:?} {slope} {want}");
        }
    }

    #[test]
    fn draws_have_the_family_mean_and_variance() {
        for f in FAMILIES {
            let (_, mu) = sample_point(f);
            let (w, phi) = (2.0, if f.unit_dispersion() { 1.0 } else { 0.5 });
            let mut rng = StreamRng::new(7, 0);
            let n = 10_000;
            let x: Vec<f64> = (0..n)
                .map(|_| f.draw(mu, phi, w, rng.next_open01()).unwrap())
                .collect();
            let m = x.iter().sum::<f64>() / n as f64;
            let v = x.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / n as f64;
            let want_v = match f {
                // The negative binomial's variance ignores the weight.
                Family::NegativeBinomial { .. } => f.variance(mu),
                _ => phi * f.variance(mu) / w,
            };
            let se = (want_v / n as f64).sqrt();
            assert!((m - mu).abs() < 5.0 * se, "{f:?} mean {m}");
            assert!(
                (v / want_v - 1.0).abs() < 0.1,
                "{f:?} var {v} want {want_v}"
            );
        }
    }

    #[test]
    fn inverse_gaussian_cdf_is_a_distribution() {
        let (mu, lambda) = (2.0, 3.0);
        assert!(inverse_gaussian_cdf(1e-9, mu, lambda) < 1e-12);
        assert!((inverse_gaussian_cdf(1e4, mu, lambda) - 1.0).abs() < 1e-12);
        // Large λ/μ: no overflow.
        let c = inverse_gaussian_cdf(1.0, 1.0, 1e4);
        assert!((0.0..=1.0).contains(&c) && (c - 0.5).abs() < 0.01, "{c}");
    }
}
