//! The gamma distribution.

use prospicio_core::{Error, Result, StreamRng};
use prospicio_math::roots::bisect;
use prospicio_math::special::{gamma_inc, ln_gamma, norm_quantile};

use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// Gamma distribution with shape `α` and scale `θ`: density
/// `x^(α-1) e^(-x/θ) / (Γ(α) θ^α)`, mean `αθ`, variance `αθ²`.
///
/// Parameterized as in SciPy (`a = shape`, `scale`) and R (`shape`,
/// `scale = 1/rate`). The gamma GLM family is this distribution with
/// shape `1/φ` and mean `μ`; see [`Gamma::from_mean_dispersion`].
///
/// # Example
///
/// ```
/// use prospicio_prob::{Distribution, Gamma, Severity};
///
/// let d = Gamma::from_mean_cv(1000.0, 0.5).unwrap();
/// assert!((d.shape() - 4.0).abs() < 1e-12);
/// assert!((d.std_dev() - 500.0).abs() < 1e-9);
/// // LEV + stop-loss = mean.
/// assert!((d.lev(1500.0) + d.stop_loss(1500.0) - 1000.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma {
    shape: f64,
    scale: f64,
}

impl Gamma {
    /// Gamma with shape `α > 0` and scale `θ > 0`.
    pub fn new(shape: f64, scale: f64) -> Result<Self> {
        positive("shape", shape)?;
        positive("scale", scale)?;
        Ok(Self { shape, scale })
    }

    /// Gamma with the given mean and coefficient of variation: shape
    /// `1/cv²`, scale `mean cv²`.
    pub fn from_mean_cv(mean: f64, cv: f64) -> Result<Self> {
        positive("mean", mean)?;
        positive("cv", cv)?;
        Self::new(1.0 / (cv * cv), mean * cv * cv)
    }

    /// Gamma with mean `μ` and GLM dispersion `φ` (variance `φμ²`): shape
    /// `1/φ`, scale `φμ`.
    pub fn from_mean_dispersion(mean: f64, dispersion: f64) -> Result<Self> {
        positive("mean", mean)?;
        positive("dispersion", dispersion)?;
        Self::new(1.0 / dispersion, dispersion * mean)
    }

    /// Shape `α`.
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Scale `θ`.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Log density at `x`; `-inf` outside the support (and at 0 for
    /// shape above 1).
    ///
    /// ```
    /// use prospicio_prob::Gamma;
    ///
    /// // Shape 1 is the exponential: f(x) = e^(-x/θ) / θ.
    /// let d = Gamma::new(1.0, 2.0).unwrap();
    /// assert!((d.ln_pdf(3.0) - (-1.5 - 2f64.ln())).abs() < 1e-15);
    /// ```
    pub fn ln_pdf(&self, x: f64) -> f64 {
        if x < 0.0 || x.is_nan() {
            return f64::NEG_INFINITY;
        }
        let (a, t) = (self.shape, self.scale);
        if x == 0.0 {
            return if a < 1.0 {
                f64::INFINITY
            } else if a == 1.0 {
                -t.ln()
            } else {
                f64::NEG_INFINITY
            };
        }
        (a - 1.0) * x.ln() - x / t - ln_gamma(a) - a * t.ln()
    }

    /// `E[X^k; X > u] - u^k S(u)`, the tail part of `E[X^k] -
    /// E[min(X, u)^k]`, for `k` in 1 and 2. Small in the tail, so layer
    /// moments built from it keep their precision.
    fn tail_moment(&self, k: i32, u: f64) -> f64 {
        if u <= 0.0 {
            return self.raw_moment(k) - u.powi(k);
        }
        if u == f64::INFINITY {
            return 0.0;
        }
        let z = u / self.scale;
        let (_, q_shifted) = gamma_inc(self.shape + f64::from(k), z);
        let (_, q) = gamma_inc(self.shape, z);
        (self.raw_moment(k) * q_shifted - u.powi(k) * q).max(0.0)
    }

    /// `E[X^k]` for `k` in 1 and 2.
    fn raw_moment(&self, k: i32) -> f64 {
        let (a, t) = (self.shape, self.scale);
        match k {
            1 => a * t,
            _ => a * (a + 1.0) * t * t,
        }
    }
}

impl Distribution for Gamma {
    fn mean(&self) -> f64 {
        self.shape * self.scale
    }

    fn variance(&self) -> f64 {
        self.shape * self.scale * self.scale
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        gamma_inc(self.shape, x / self.scale).0
    }

    fn survival(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        gamma_inc(self.shape, x / self.scale).1
    }

    /// By bisection to full precision, on the distribution function below
    /// the median and on the survival function above it, so both tails
    /// keep their relative precision.
    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        let below = |x: f64| {
            if p <= 0.5 {
                self.cdf(x) < p
            } else {
                self.survival(x) > 1.0 - p
            }
        };
        let mut hi = self.mean() + self.std_dev();
        while below(hi) {
            hi *= 2.0;
        }
        Ok(bisect(0.0, hi, below))
    }

    /// `n` draws by Marsaglia and Tsang (2000), not inverse transform:
    /// each is `θ` times a Gamma(`α`, 1) draw from `rng` (for `α < 1`, a
    /// draw at `α + 1` times `U^(1/α)`), so draws stay a pure function of
    /// `(seed, stream)` but are not monotone in one uniform. The quantile
    /// function costs a bisection on the incomplete gamma function, whose
    /// series grows with the shape, where this costs about one normal and
    /// one uniform at any shape.
    fn sample(&self, rng: &mut StreamRng, n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| self.scale * standard_gamma(rng, self.shape))
            .collect()
    }
}

/// A Gamma(`shape`, 1) draw by Marsaglia and Tsang (2000), "A simple
/// method for generating gamma variables", ACM Transactions on
/// Mathematical Software 26(3), for `shape >= 1`; below 1, their boost: a
/// draw at `shape + 1` times `U^(1/shape)`.
///
/// Every uniform comes from `rng` in order: for `shape >= 1`, each attempt
/// takes a normal (by inverse transform) and, unless `1 + c x <= 0`, a
/// uniform; above shape 1 at least 95% of attempts are accepted. The boost
/// takes its uniform after the draw at `shape + 1`. A draw below
/// `f64::MIN_POSITIVE` comes out subnormal, with fewer significant bits,
/// and one below the smallest subnormal (about `5e-324`) rounds to 0: at
/// shape `1e-3`, 49% and about 47.5% of the mass.
pub(crate) fn standard_gamma(rng: &mut StreamRng, shape: f64) -> f64 {
    if shape < 1.0 {
        let g = standard_gamma(rng, shape + 1.0);
        return g * rng.next_open01().powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = norm_quantile(rng.next_open01());
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.next_open01();
        // The squeeze accepts most draws without the logarithms.
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 || u.ln() < 0.5 * x2 + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

impl Severity for Gamma {
    /// `E[min(X, u)] = αθ P(α + 1, u/θ) + u Q(α, u/θ)`.
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        if limit == f64::INFINITY {
            return self.mean();
        }
        let z = limit / self.scale;
        self.mean() * gamma_inc(self.shape + 1.0, z).0 + limit * gamma_inc(self.shape, z).1
    }

    /// `E[(X - d)+] = αθ Q(α + 1, d/θ) - d Q(α, d/θ)`, from the tail, so
    /// it does not cancel against the mean.
    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean() - retention;
        }
        self.tail_moment(1, retention)
    }

    /// `LEV(a + limit) - LEV(a)` for a layer attaching below the mean,
    /// where both are small, and `stop_loss(a) - stop_loss(a + limit)`
    /// above it, where those are: neither difference cancels.
    fn layer(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        if a <= self.mean() {
            self.lev(a + limit) - self.lev(a)
        } else {
            self.stop_loss(a) - self.stop_loss(a + limit)
        }
    }

    /// With `b = a + limit` and `t_k(u) = E[X^k; X > u] - u^k S(u)`,
    /// `E[Y²] = t_2(a) - t_2(b) - 2a (t_1(a) - t_1(b))`, every term taken
    /// from the tail.
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        let b = a + limit;
        let t = |k: i32, u: f64| self.tail_moment(k, u);
        (t(2, a) - t(2, b) - 2.0 * a * (t(1, a) - t(1, b))).max(0.0)
    }
}

fn positive(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name,
            value,
            reason: "must be finite and positive",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_core::StreamRng;

    #[test]
    fn severity_identities_and_edges() {
        let d = Gamma::new(2.5, 400.0).unwrap();
        for limit in [100.0, 1_000.0, 5_000.0] {
            let sum = d.lev(limit) + d.stop_loss(limit);
            assert!((sum - d.mean()).abs() < 1e-12 * d.mean());
        }
        assert_eq!(d.lev(0.0), 0.0);
        assert_eq!(d.lev(f64::INFINITY), d.mean());
        assert_eq!(d.stop_loss(f64::INFINITY), 0.0);
        assert_eq!(d.stop_loss(0.0), d.mean());
        assert!((d.layer(f64::INFINITY, 0.0) - d.mean()).abs() < 1e-9);
        let stacked = d.layer(1_000.0, 0.0) + d.layer(1_000.0, 1_000.0);
        assert!((stacked - d.layer(2_000.0, 0.0)).abs() < 1e-9);
        // The whole distribution as one layer: E[X²].
        let m2 = d.layer_second_moment(f64::INFINITY, 0.0);
        assert!((m2 - (d.variance() + d.mean() * d.mean())).abs() < 1e-9 * m2);
    }

    #[test]
    fn layer_variance_matches_simulation() {
        let d = Gamma::new(1.5, 1000.0).unwrap();
        let x = d.sample(&mut StreamRng::new(4, 0), 100_000);
        let (l, a) = (2000.0, 1000.0);
        let y: Vec<f64> = x.iter().map(|v| (v - a).clamp(0.0, l)).collect();
        let n = y.len() as f64;
        let m = y.iter().sum::<f64>() / n;
        let v = y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
        assert!((m / d.layer(l, a) - 1.0).abs() < 0.02, "{m}");
        assert!((v / d.layer_variance(l, a) - 1.0).abs() < 0.04, "{v}");
    }

    #[test]
    fn quantile_inverts_both_tails() {
        let d = Gamma::new(0.3, 2.0).unwrap();
        for p in [1e-12, 0.01, 0.5, 0.99] {
            let x = d.quantile(p).unwrap();
            assert!((d.cdf(x) / p - 1.0).abs() < 1e-12, "{p}");
        }
        let x = d.quantile(1.0 - 1e-12).unwrap();
        assert!((d.survival(x) / 1e-12 - 1.0).abs() < 1e-3, "{x}");
        assert_eq!(d.quantile(0.0), Ok(0.0));
        assert_eq!(d.quantile(1.0), Ok(f64::INFINITY));
    }

    /// Central moments `mu_2` to `mu_6` of Gamma(`a`, 1) (index `k` holds
    /// `mu_k`), from its cumulants `kappa_n = a (n - 1)!`.
    fn central_moments(a: f64) -> [f64; 7] {
        let mut mu = [0.0; 7];
        mu[2] = a;
        mu[3] = 2.0 * a;
        mu[4] = 3.0 * a * a + 6.0 * a;
        mu[5] = 20.0 * a * a + 24.0 * a;
        mu[6] = 15.0 * a.powi(3) + 130.0 * a * a + 120.0 * a;
        mu
    }

    /// The upper `z`-sigma point of a chi-square with `df` degrees of
    /// freedom, by Wilson and Hilferty (1931).
    fn chi_square_upper(df: f64, z: f64) -> f64 {
        let h = 2.0 / (9.0 * df);
        df * (1.0 - h + z * h.sqrt()).powi(3)
    }

    /// Mean, second and third central moments (about the true mean, so each
    /// is an i.i.d. average with an exact standard error) as z-scores, and
    /// the chi-square of the draws over 100 equiprobable bins cut at the
    /// distribution's own quantiles, against its `1e-6` upper point, at
    /// shapes from `1e-3` to `1e6`. Below `f64::MIN_POSITIVE` the quantiles
    /// (and the draws, subnormal or, below about `5e-324`, 0) cannot be
    /// told apart, so those bins merge into one: at shape `1e-3` that is
    /// the lower 49% of the mass.
    #[test]
    fn sampler_matches_moments_and_quantiles() {
        let n = 100_000;
        let mut failures = Vec::new();
        let shapes = [
            1e-3, 0.01, 0.1, 0.5, 0.999, 1.0, 1.5, 3.0, 10.0, 100.0, 1e4, 1e6,
        ];
        for (stream, &a) in shapes.iter().enumerate() {
            let d = Gamma::new(a, 1.0).unwrap();
            let mut x = d.sample(&mut StreamRng::new(2026, stream as u64), n);
            let nf = n as f64;
            let mu = central_moments(a);
            let average = |f: &dyn Fn(f64) -> f64| x.iter().map(|&v| f(v)).sum::<f64>() / nf;
            let z = [
                (average(&|v| v) - a) / (mu[2] / nf).sqrt(),
                (average(&|v| (v - a).powi(2)) - mu[2]) / ((mu[4] - mu[2] * mu[2]) / nf).sqrt(),
                (average(&|v| (v - a).powi(3)) - mu[3]) / ((mu[6] - mu[3] * mu[3]) / nf).sqrt(),
            ];
            for (name, z) in ["mean", "variance", "third moment"].iter().zip(z) {
                if z.abs() > 5.0 {
                    failures.push(format!("shape {a}: {name} z = {z:.2}"));
                }
            }

            x.sort_by(f64::total_cmp);
            let mut edges: Vec<f64> = (1..100)
                .map(|k| d.quantile(f64::from(k) / 100.0).unwrap())
                .filter(|&q| q >= f64::MIN_POSITIVE)
                .collect();
            edges.dedup();
            let mut chi2 = 0.0;
            let (mut below, mut cdf_below) = (0, 0.0);
            for i in 0..=edges.len() {
                let (count, cdf) = match edges.get(i) {
                    Some(&e) => (x.partition_point(|&v| v <= e), d.cdf(e)),
                    None => (n, 1.0),
                };
                let expected = (cdf - cdf_below) * nf;
                chi2 += ((count - below) as f64 - expected).powi(2) / expected;
                (below, cdf_below) = (count, cdf);
            }
            let critical = chi_square_upper(edges.len() as f64, 4.75);
            if chi2 > critical {
                failures.push(format!(
                    "shape {a}: chi-square {chi2:.1} over {} bins, above {critical:.1}",
                    edges.len() + 1
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// A regression pin of the sampler's output (Marsaglia–Tsang since
    /// 2026-10-08), so a change to the draws is deliberate: one shape below
    /// 1 (the boost) and one above, from one stream.
    /// `validation/scripts/gamma_sampler.py` reproduces them independently
    /// (ChaCha20 from `cryptography`, SciPy's `ndtri`) to within 2e-15.
    #[test]
    fn sample_is_pinned() {
        // These draws are the sampler `GAMMA_SAMPLER` names. A change that
        // moves them gives the sampler a new id (docs/design/rng.md,
        // stability policy), so the two are pinned together.
        assert_eq!(crate::provenance::GAMMA_SAMPLER, "marsaglia-tsang/2026-10");
        let small = Gamma::new(0.3, 2.0).unwrap();
        let large = Gamma::new(2.5, 400.0).unwrap();
        let mut rng = StreamRng::new(42, 3);
        let draws = [small.sample(&mut rng, 2), large.sample(&mut rng, 2)].concat();
        assert_eq!(
            draws,
            [
                0.23214754851650782,
                0.7755693217108093,
                110.9058975015963,
                560.609408990688
            ]
        );
    }

    #[test]
    fn exponential_and_parameterizations() {
        let d = Gamma::new(1.0, 3.0).unwrap();
        assert!((d.survival(6.0) - (-2f64).exp()).abs() < 1e-16);
        let g = Gamma::from_mean_dispersion(200.0, 0.25).unwrap();
        assert!((g.mean() - 200.0).abs() < 1e-12);
        assert!((g.variance() - 0.25 * 200.0 * 200.0).abs() < 1e-8);
        assert!(Gamma::new(0.0, 1.0).is_err());
        assert!(Gamma::from_mean_cv(1.0, f64::NAN).is_err());
    }
}
