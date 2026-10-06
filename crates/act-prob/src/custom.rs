//! [`Custom`], a user-defined loss severity given by its cdf: the "slow
//! path" of `docs/design/distributions.md`, through which a Python or R
//! function enters the native calculations.
//!
//! The cdf is the only function a user must supply; a quantile function
//! is optional and makes sampling much faster. Everything else (the mean,
//! the variance, limited expected values, layer moments) is computed by
//! Gauss–Legendre quadrature of the survival function on panels between
//! the distribution's own quantiles, so it is as accurate as the cdf is
//! smooth. The integrals stop at the `1 - 1e-12` quantile: the probability
//! above it is ignored, which is negligible unless the tail is so heavy
//! that the mean barely exists.
//!
//! A callback may come from a language that must not be entered from
//! several threads at once (R must only ever be called on its main thread),
//! so a `Custom` built with `parallel_safe = false` reports so through
//! [`Distribution::is_parallel_safe`], and the parallel simulations run
//! single-threaded on the calling thread when they meet one.

use std::fmt;
use std::sync::{Arc, Mutex};

use act_core::{Error, Result};
use act_math::integrate::gauss_legendre;
use act_math::roots::bisect_log;

use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// A user function of one variable. An `Err` carries the message of the
/// failure (a Python exception, an R error).
pub type Callback = Arc<dyn Fn(f64) -> std::result::Result<f64, String> + Send + Sync>;

/// Probabilities whose quantiles bound the integration panels.
const PANEL_PROBS: [f64; 16] = [
    0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12,
];

/// Gauss–Legendre pieces per panel.
const PIECES: usize = 8;

/// A non-negative loss severity defined by a user's cdf (and optionally
/// quantile) function.
///
/// ```
/// use std::sync::Arc;
/// use act_prob::{Custom, Distribution, Severity};
///
/// // An exponential with mean 100, given only by its cdf.
/// let cdf = Arc::new(|x: f64| Ok((1.0 - (-x / 100.0).exp()).max(0.0)));
/// let d = Custom::new("exponential", cdf, None, true).unwrap();
/// assert!((d.mean() - 100.0).abs() < 1e-6);
/// // E[min(X, 50)] = 100 (1 - e^-0.5).
/// assert!((d.lev(50.0) - 100.0 * (1.0 - (-0.5f64).exp())).abs() < 1e-9);
/// assert!((d.quantile(0.5).unwrap() - 100.0 * 2f64.ln()).abs() < 1e-9);
/// ```
#[derive(Clone)]
pub struct Custom {
    name: String,
    cdf: Callback,
    quantile: Option<Callback>,
    parallel_safe: bool,
    /// Panel boundaries: 0, then quantiles up to the `1 - 1e-12` one.
    breaks: Vec<f64>,
    mean: f64,
    second_moment: f64,
    /// The first callback failure after construction.
    error: Arc<Mutex<Option<String>>>,
}

impl fmt::Debug for Custom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Custom")
            .field("name", &self.name)
            .field("has_quantile", &self.quantile.is_some())
            .field("parallel_safe", &self.parallel_safe)
            .field("mean", &self.mean)
            .field("upper", &self.upper())
            .finish()
    }
}

fn callback_error(name: &str, what: &str, x: f64, msg: &str) -> Error {
    Error::Data(format!(
        "custom distribution {name:?}: {what}({x}) failed: {msg}"
    ))
}

impl Custom {
    /// A severity from its `cdf` and, optionally, its `quantile` function.
    ///
    /// `parallel_safe` says whether the callbacks may run on several
    /// threads at once (true for a pure Rust closure; false for an R
    /// function). Construction evaluates the cdf at the panel quantiles and
    /// computes the mean and second moment, so a cdf that fails, leaves
    /// `[0, 1]`, decreases or never reaches `1 - 1e-12` is reported here.
    pub fn new(
        name: impl Into<String>,
        cdf: Callback,
        quantile: Option<Callback>,
        parallel_safe: bool,
    ) -> Result<Self> {
        let mut d = Self {
            name: name.into(),
            cdf,
            quantile,
            parallel_safe,
            breaks: vec![0.0],
            mean: f64::NAN,
            second_moment: f64::NAN,
            error: Arc::new(Mutex::new(None)),
        };
        d.try_cdf(0.0)?;
        let mut breaks = vec![0.0];
        for p in PANEL_PROBS {
            let p = if p < 0.01 { 1.0 - p } else { p };
            let q = d.try_quantile(p)?;
            if !q.is_finite() {
                return Err(Error::Data(format!(
                    "custom distribution {:?}: its cdf does not reach {p} at any finite loss",
                    d.name
                )));
            }
            if q > *breaks.last().unwrap() {
                breaks.push(q);
            } else if q < *breaks.last().unwrap() {
                return Err(Error::Data(format!(
                    "custom distribution {:?}: quantiles must increase with p, but q({p}) = {q} \
                     is below an earlier quantile",
                    d.name
                )));
            }
        }
        d.breaks = breaks;
        // Both moments in one pass: E[X] = ∫ S, E[X²] = ∫ 2x S.
        let upper = d.upper();
        d.mean = d.integrate(0.0, upper, |_, s| s)?;
        d.second_moment = d.integrate(0.0, upper, |x, s| 2.0 * x * s)?;
        Ok(d)
    }

    /// The name given at construction.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether a quantile function was supplied (otherwise quantiles invert
    /// the cdf by bisection, about a hundred cdf calls each).
    pub fn has_quantile(&self) -> bool {
        self.quantile.is_some()
    }

    /// The `1 - 1e-12` quantile, where the integrals stop.
    pub fn upper(&self) -> f64 {
        *self.breaks.last().unwrap()
    }

    /// The first callback failure since construction, if any. A failure
    /// inside a calculation makes that value NaN; this says why.
    pub fn error(&self) -> Option<String> {
        self.error.lock().map(|e| e.clone()).unwrap_or(None)
    }

    fn record(&self, e: &Error) {
        if let Ok(mut slot) = self.error.lock()
            && slot.is_none()
        {
            *slot = Some(e.to_string());
        }
    }

    fn try_cdf(&self, x: f64) -> Result<f64> {
        let v = (self.cdf)(x).map_err(|m| callback_error(&self.name, "cdf", x, &m))?;
        if !(0.0..=1.0).contains(&v) {
            return Err(callback_error(
                &self.name,
                "cdf",
                x,
                &format!("returned {v}, outside [0, 1]"),
            ));
        }
        Ok(v)
    }

    fn try_quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        if let Some(q) = &self.quantile {
            let v = q(p).map_err(|m| callback_error(&self.name, "quantile", p, &m))?;
            if v.is_nan() || v < 0.0 {
                return Err(callback_error(
                    &self.name,
                    "quantile",
                    p,
                    &format!("returned {v}; losses are non-negative"),
                ));
            }
            return Ok(v);
        }
        if self.try_cdf(0.0)? >= p {
            return Ok(0.0);
        }
        // Bracket by doubling, then bisect on a log scale.
        let mut hi = 1.0;
        while self.try_cdf(hi)? < p {
            hi *= 2.0;
            if !hi.is_finite() {
                return Ok(f64::INFINITY);
            }
        }
        let mut lo = hi;
        loop {
            lo *= 0.5;
            if lo == 0.0 {
                return Ok(0.0);
            }
            if self.try_cdf(lo)? < p {
                break;
            }
        }
        let mut failure = None;
        let x = bisect_log(lo, hi, |x| match self.try_cdf(x) {
            Ok(c) => c < p,
            Err(e) => {
                failure.get_or_insert(e);
                false
            }
        });
        match failure {
            Some(e) => Err(e),
            None => Ok(x),
        }
    }

    /// `∫_lo^hi g(x, S(x)) dx` over the panels, clipped to `[0, upper]`.
    fn integrate(&self, lo: f64, hi: f64, g: impl Fn(f64, f64) -> f64) -> Result<f64> {
        let (lo, hi) = (lo.max(0.0), hi.min(self.upper()));
        let mut total = 0.0;
        for w in self.breaks.windows(2) {
            let (a, b) = (w[0].max(lo), w[1].min(hi));
            if a >= b {
                continue;
            }
            let step = (b - a) / PIECES as f64;
            for k in 0..PIECES {
                let x0 = a + step * k as f64;
                let x1 = if k + 1 == PIECES { b } else { x0 + step };
                total += gauss_legendre(|x| self.try_cdf(x).map(|c| g(x, 1.0 - c)), x0, x1)?;
            }
        }
        Ok(total)
    }

    fn or_nan(&self, r: Result<f64>) -> f64 {
        r.unwrap_or_else(|e| {
            self.record(&e);
            f64::NAN
        })
    }
}

impl Distribution for Custom {
    fn mean(&self) -> f64 {
        self.mean
    }

    fn variance(&self) -> f64 {
        self.second_moment - self.mean * self.mean
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        self.or_nan(self.try_cdf(x))
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        self.try_quantile(p).inspect_err(|e| self.record(e))
    }

    fn is_parallel_safe(&self) -> bool {
        self.parallel_safe
    }
}

impl Severity for Custom {
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        if limit == f64::INFINITY {
            return self.mean;
        }
        self.or_nan(self.integrate(0.0, limit, |_, s| s))
    }

    /// `∫_r^∞ S`, integrated directly so far retentions keep their
    /// precision.
    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean - retention;
        }
        self.or_nan(self.integrate(retention, f64::INFINITY, |_, s| s))
    }

    /// `∫_a^{a+l} 2 (x - a) S(x) dx`.
    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        let a = attachment.max(0.0);
        self.or_nan(self.integrate(a, a + limit, |x, s| 2.0 * (x - a) * s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lognormal;

    fn from<D: Distribution + Send + Sync + 'static>(d: D, with_quantile: bool) -> Custom {
        let d = Arc::new(d);
        let c = d.clone();
        let cdf: Callback = Arc::new(move |x| Ok(c.cdf(x)));
        let quantile: Option<Callback> = with_quantile.then(|| {
            let q = d.clone();
            Arc::new(move |p| q.quantile(p).map_err(|e| e.to_string())) as Callback
        });
        Custom::new("test", cdf, quantile, true).unwrap()
    }

    #[test]
    fn moments_and_layers_match_a_closed_form() {
        let ln = Lognormal::from_mean_cv(1000.0, 1.0).unwrap();
        for with_quantile in [true, false] {
            let c = from(ln, with_quantile);
            assert!((c.mean() / ln.mean() - 1.0).abs() < 1e-7, "{with_quantile}");
            assert!((c.variance() / ln.variance() - 1.0).abs() < 1e-4);
            for l in [100.0, 1000.0, 5000.0] {
                assert!((c.lev(l) / ln.lev(l) - 1.0).abs() < 1e-9, "lev {l}");
                assert!(
                    (c.stop_loss(l) / ln.stop_loss(l) - 1.0).abs() < 1e-6,
                    "sl {l}"
                );
            }
            let (m, v) = (c.layer(2000.0, 1000.0), c.layer_variance(2000.0, 1000.0));
            assert!((m / ln.layer(2000.0, 1000.0) - 1.0).abs() < 1e-8);
            assert!((v / ln.layer_variance(2000.0, 1000.0) - 1.0).abs() < 1e-8);
            for p in [0.001, 0.5, 0.99] {
                let q = c.quantile(p).unwrap();
                assert!((q / ln.quantile(p).unwrap() - 1.0).abs() < 1e-12, "q {p}");
            }
        }
    }

    #[test]
    fn a_point_mass_at_zero_is_allowed() {
        // 30% no-claim, else exponential(mean 10).
        let cdf: Callback = Arc::new(|x| {
            Ok(if x < 0.0 {
                0.0
            } else {
                0.3 + 0.7 * (1.0 - (-x / 10.0).exp())
            })
        });
        let c = Custom::new("zero-inflated", cdf, None, true).unwrap();
        assert!((c.mean() - 7.0).abs() < 1e-7);
        assert_eq!(c.quantile(0.2).unwrap(), 0.0);
        assert_eq!(c.cdf(-1.0), 0.0);
    }

    #[test]
    fn bad_callbacks_are_reported() {
        let failing: Callback = Arc::new(|x| {
            if x > 50.0 {
                Err("boom".into())
            } else {
                Ok(x / 100.0)
            }
        });
        let e = Custom::new("bad", failing, None, true)
            .unwrap_err()
            .to_string();
        assert!(e.contains("boom"), "{e}");

        let outside: Callback = Arc::new(|_| Ok(1.5));
        assert!(Custom::new("bad", outside, None, true).is_err());

        let never: Callback = Arc::new(|x| Ok(0.5 * (1.0 - (-x).exp())));
        let e = Custom::new("bad", never, None, true)
            .unwrap_err()
            .to_string();
        assert!(e.contains("does not reach"), "{e}");

        // A failure after construction gives NaN and is kept.
        let late: Callback = Arc::new(|x| {
            if x == 12345.0 {
                Err("late".into())
            } else {
                Ok(1.0 - (-x).exp())
            }
        });
        let c = Custom::new("late", late, None, false).unwrap();
        assert!(c.error().is_none());
        assert!(c.cdf(12345.0).is_nan());
        assert!(c.error().unwrap().contains("late"));
        assert!(!c.is_parallel_safe());
    }
}
