//! The occurrence exceedance probability (OEP) curve: the chance that some
//! occurrence in a year exceeds a loss, and its inverse, the occurrence PML
//! (`docs/design/aggregate.md`). The aggregate exceedance (AEP) curve is the
//! compound distribution's survival function.
//!
//! With `N` occurrences a year, independent of their severities `X` with
//! cdf `F`, the year's largest occurrence is at most `x` with probability
//! `E[F(x)^N] = G_N(F(x))`, the count's probability generating function, so
//!
//! `OEP(x) = 1 − G_N(F(x))`.
//!
//! For a Poisson count with mean `λ` this is `1 − exp(−λ S(x))`, `S = 1 − F`,
//! and the PML at an annual probability `p` is the severity's quantile at
//! `1 + ln(1 − p) / λ`, as the Python package `aggregate`'s `oep`. Any count
//! works here: a negative binomial's clustering lowers the OEP at a given
//! mean.

use prospicio_core::{Error, Result};
use prospicio_math::roots::{Stop, bisect_log, try_bisect};
use prospicio_prob::{Counting, Distribution};

use crate::reinsurance::Tower;

/// `1 − G_N(F(x))`: the chance that the year's largest occurrence exceeds
/// `x`.
///
/// ```
/// use prospicio_aggregate::pml::occurrence_exceedance;
/// use prospicio_prob::{Lognormal, Poisson};
///
/// let n = Poisson::new(2.0).unwrap();
/// let x = Lognormal::new(4.0, 1.0).unwrap();
/// // Poisson: 1 − exp(−λ S(x)).
/// let s = 1.0 - prospicio_prob::Distribution::cdf(&x, 500.0);
/// assert!((occurrence_exceedance(&n, &x, 500.0) - (1.0 - (-2.0 * s).exp())).abs() < 1e-12);
/// ```
pub fn occurrence_exceedance<N, X>(frequency: &N, severity: &X, x: f64) -> f64
where
    N: Counting + ?Sized,
    X: Distribution + ?Sized,
{
    1.0 - frequency.pgf(severity.cdf(x))
}

/// One point of the occurrence PML curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OccurrencePml {
    /// The annual probability `p` asked for.
    pub probability: f64,
    /// The loss `x` with `OEP(x) = p`.
    pub loss: f64,
    /// `S(x) = P(X > x)`, the severity's exceedance at the loss.
    pub severity_survival: f64,
    /// `1 / (E[N] S(x))`, the average time between occurrences above the
    /// loss; it can be shorter than a year.
    pub occurrence_return_period: f64,
    /// `1 / p`, the average time between years with such an occurrence.
    pub annual_return_period: f64,
}

/// The occurrence PML at an annual probability `p`: the loss `x` with
/// `OEP(x) = p`, for `p` in `(0, 1 − P(N = 0))`.
///
/// The severity's exceedance `s` solving `1 − G_N(1 − s) = p` is found by
/// bisection on a log scale, and the loss is the severity's quantile at
/// `1 − s`. Far in the tail (`s` below about `1e-12`) the quantile at
/// `1 − s` loses relative precision.
///
/// ```
/// use prospicio_aggregate::pml::occurrence_pml;
/// use prospicio_prob::{Distribution, Lognormal, Poisson};
///
/// let n = Poisson::new(2.0).unwrap();
/// let x = Lognormal::new(4.0, 1.0).unwrap();
/// let pml = occurrence_pml(&n, &x, 0.01).unwrap();
/// // Poisson: the severity quantile at 1 + ln(0.99) / 2.
/// let q = x.quantile(1.0 + 0.99f64.ln() / 2.0).unwrap();
/// assert!((pml.loss / q - 1.0).abs() < 1e-12);
/// assert_eq!(pml.annual_return_period, 100.0);
/// ```
pub fn occurrence_pml<N, X>(frequency: &N, severity: &X, p: f64) -> Result<OccurrencePml>
where
    N: Counting + ?Sized,
    X: Distribution + ?Sized,
{
    let ceiling = 1.0 - frequency.pgf(0.0);
    if !(p > 0.0 && p < ceiling) {
        return Err(Error::InvalidParameter {
            name: "p",
            value: p,
            reason: "must be in (0, 1 − P(N = 0))",
        });
    }
    // 1 − G(1 − s) increases from 0 at s = 0 to the ceiling at s = 1.
    let oep = |s: f64| 1.0 - frequency.pgf(1.0 - s);
    let s = bisect_log(f64::MIN_POSITIVE, 1.0, |s| oep(s) < p);
    let loss = severity.quantile(1.0 - s)?;
    let severity_survival = 1.0 - severity.cdf(loss);
    Ok(OccurrencePml {
        probability: p,
        loss,
        severity_survival,
        occurrence_return_period: 1.0 / (frequency.mean() * severity_survival),
        annual_return_period: 1.0 / p,
    })
}

/// Which loss of an occurrence a program's OEP curve is of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccurrenceView {
    /// The ground-up loss.
    Gross,
    /// The tower's total ceded loss, [`Tower::occurrence_ceded`].
    Ceded,
    /// The gross loss less the ceded loss.
    Net,
}

impl OccurrenceView {
    /// The view by name: `"gross"`, `"ceded"` or `"net"`.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "gross" => Ok(Self::Gross),
            "ceded" => Ok(Self::Ceded),
            "net" => Ok(Self::Net),
            other => Err(Error::Data(format!(
                "view must be \"gross\", \"ceded\" or \"net\", not {other:?}"
            ))),
        }
    }

    fn of(self, tower: &Tower, x: f64) -> Result<f64> {
        Ok(match self {
            Self::Gross => x,
            Self::Ceded => tower.occurrence_ceded(x)?,
            Self::Net => x - tower.occurrence_ceded(x)?,
        })
    }
}

/// The OEP of a program at a loss `y` of the view: the chance that some
/// occurrence's ceded (or net) loss exceeds `y`, `1 − G_N(1 − S)` with `S`
/// the gross survival at the largest gross loss whose view is at most `y`.
/// The program's ceded and net functions of an occurrence are
/// non-decreasing (shares at most 1, layers in a stage not overlapping), so
/// that loss is found by bisection.
pub fn program_occurrence_exceedance<N, X>(
    frequency: &N,
    severity: &X,
    tower: &Tower,
    view: OccurrenceView,
    y: f64,
) -> Result<f64>
where
    N: Counting + ?Sized,
    X: Distribution + ?Sized,
{
    let x = largest_at_most(severity, tower, view, y)?;
    Ok(1.0 - frequency.pgf(severity.cdf(x)))
}

/// The largest gross loss whose view is at most `y` (`+inf` when every
/// loss's is).
fn largest_at_most<X: Distribution + ?Sized>(
    severity: &X,
    tower: &Tower,
    view: OccurrenceView,
    y: f64,
) -> Result<f64> {
    let g = |x: f64| view.of(tower, x);
    if g(0.0)? > y {
        return Ok(-f64::MIN_POSITIVE);
    }
    // Double past y, then bisect.
    let mut hi = severity.quantile(0.5)?.max(1.0);
    while g(hi)? <= y {
        hi *= 2.0;
        if hi > 1e300 {
            return Ok(f64::INFINITY);
        }
    }
    let (lo, _) = try_bisect(0.0, hi, Stop::steps(200), |mid| g(mid).map(|v| v <= y))?;
    Ok(lo)
}

/// The occurrence PML of a program at an annual probability `p`: the ceded
/// (or net) loss of the year's largest occurrence, `g(x_p)` with `x_p` the
/// gross [`occurrence_pml`] and `g` the program's ceded or net function of
/// an occurrence ([`Tower::occurrence_ceded`]), `aggregate` 2.1.0's `oep` on
/// an aggregate with an occurrence program. Annual terms do not enter.
///
/// Where `g` is flat (a layer placed in full, for the net; below the first
/// attachment, across a gap or above a finite program, for the ceded) the
/// view has an atom, and `severity_survival` is the gross survival at the
/// right end of that stretch, so the achieved OEP, `1 − G_N(1 −
/// severity_survival)`, falls below `p`, as at a policy limit.
///
/// ```
/// use prospicio_aggregate::pml::{OccurrenceView, occurrence_pml, program_occurrence_pml};
/// use prospicio_aggregate::{Layer, Tower};
/// use prospicio_prob::{Lognormal, Poisson};
///
/// let (n, x) = (Poisson::new(2.0).unwrap(), Lognormal::from_mean_cv(100.0, 2.0).unwrap());
/// let t = Tower::new(vec![Layer::xol("L", 400.0, 100.0).unwrap()]).unwrap();
/// let gross = occurrence_pml(&n, &x, 0.05).unwrap().loss;
/// let net = program_occurrence_pml(&n, &x, &t, OccurrenceView::Net, 0.05).unwrap();
/// assert!((net.loss - (gross - (gross - 100.0).clamp(0.0, 400.0))).abs() < 1e-9);
/// ```
pub fn program_occurrence_pml<N, X>(
    frequency: &N,
    severity: &X,
    tower: &Tower,
    view: OccurrenceView,
    p: f64,
) -> Result<OccurrencePml>
where
    N: Counting + ?Sized,
    X: Distribution + ?Sized,
{
    let gross = occurrence_pml(frequency, severity, p)?;
    if view == OccurrenceView::Gross {
        return Ok(gross);
    }
    let loss = view.of(tower, gross.loss)?;
    let right = largest_at_most(severity, tower, view, loss)?.max(gross.loss);
    let severity_survival = if right.is_infinite() {
        0.0
    } else {
        1.0 - severity.cdf(right)
    };
    Ok(OccurrencePml {
        probability: p,
        loss,
        severity_survival,
        occurrence_return_period: 1.0 / (frequency.mean() * severity_survival),
        annual_return_period: 1.0 / p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::{Lognormal, NegativeBinomial, Poisson};

    #[test]
    fn inverts_the_exceedance() {
        let x = Lognormal::new(4.0, 1.2).unwrap();
        let poisson = Poisson::new(3.0).unwrap();
        let negbin = NegativeBinomial::new(2.0, 1.5).unwrap();
        for p in [0.5, 0.1, 0.01, 1e-4, 1e-6] {
            let a = occurrence_pml(&poisson, &x, p).unwrap();
            assert!((occurrence_exceedance(&poisson, &x, a.loss) / p - 1.0).abs() < 1e-9);
            let b = occurrence_pml(&negbin, &x, p).unwrap();
            assert!((occurrence_exceedance(&negbin, &x, b.loss) / p - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn rejects_probabilities_past_the_ceiling() {
        let n = Poisson::new(0.1).unwrap();
        let x = Lognormal::new(0.0, 1.0).unwrap();
        // P(N ≥ 1) = 1 − exp(−0.1) ≈ 0.0952.
        assert!(occurrence_pml(&n, &x, 0.1).is_err());
        assert!(occurrence_pml(&n, &x, 0.0).is_err());
        assert!(occurrence_pml(&n, &x, 0.09).is_ok());
    }
}
