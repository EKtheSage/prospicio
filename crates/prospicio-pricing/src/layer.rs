//! Rating limits and layers from a severity.
//!
//! [`ilf`] and [`loss_elimination_ratio`] are ratios of limited expected
//! values, so they apply to any [`Severity`]. The Pareto helpers rate one
//! layer from another, or infer the Pareto alpha that two market or
//! experience figures imply: the alpha between two layers, between a
//! frequency and a layer, or between two frequencies. All of them serve
//! primary pricing (ILF tables, deductible credits, large-loss loads) and
//! reinsurance pricing alike. See `docs/design/pareto.md`.
//!
//! [`censored_las`] is the layer-by-layer limited average severity of
//! claims censored at several policy limits, from one size-of-loss table
//! per limit (Werner and Modlin, *Basic Ratemaking*, Table 11.6).

use prospicio_core::{Error, Result};
use prospicio_math::roots::bisect;
use prospicio_prob::{Distribution, Pareto, Severity, SizeOfLoss};

/// The layer `limit` xs `attachment` (per loss), for the Pareto helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XsLayer {
    limit: f64,
    attachment: f64,
}

impl XsLayer {
    /// A layer with positive `limit` (`+inf` for unlimited) and finite,
    /// positive `attachment`. (A Pareto has no mass at 0, so a ground-up
    /// layer has no Pareto rate.)
    pub fn new(limit: f64, attachment: f64) -> Result<Self> {
        positive("limit", limit)?;
        positive("attachment", attachment)?;
        if !attachment.is_finite() {
            return Err(invalid("attachment", attachment, "must be finite"));
        }
        Ok(Self { limit, attachment })
    }

    pub fn limit(&self) -> f64 {
        self.limit
    }

    pub fn attachment(&self) -> f64 {
        self.attachment
    }

    /// The exhaustion point, `attachment + limit`.
    pub fn top(&self) -> f64 {
        self.attachment + self.limit
    }

    fn expected(&self, p: &Pareto) -> f64 {
        p.layer(self.limit, self.attachment)
    }
}

/// Upper bound on the alpha the solvers search, as in the R package
/// Pareto.
pub const MAX_ALPHA: f64 = 100.0;

/// The expected loss of layer `to` per unit of expected loss of layer
/// `from`, under a Pareto with this `alpha` (and truncation, if any):
///
/// ```text
/// E[to] / E[from] = ∫_to S(x) dx / ∫_from S(x) dx,   S(x) ∝ x^(−α)
/// ```
///
/// The Pareto's threshold cancels, so only the alpha matters. Multiply by
/// the expected loss of `from` to rate `to`.
///
/// ```
/// use prospicio_pricing::layer::{XsLayer, pareto_extrapolation};
///
/// // At alpha 2, 2m xs 2m expects half the loss of 1m xs 1m:
/// // ∫_2^4 x^-2 dx / ∫_1^2 x^-2 dx = (1/4) / (1/2).
/// let from = XsLayer::new(1.0e6, 1.0e6).unwrap();
/// let to = XsLayer::new(2.0e6, 2.0e6).unwrap();
/// let r = pareto_extrapolation(from, to, 2.0, None).unwrap();
/// assert!((r - 0.5).abs() < 1e-14);
/// ```
pub fn pareto_extrapolation(
    from: XsLayer,
    to: XsLayer,
    alpha: f64,
    truncation: Option<f64>,
) -> Result<f64> {
    positive("alpha", alpha)?;
    if !alpha.is_finite() {
        return Err(invalid("alpha", alpha, "must be finite"));
    }
    let p = pareto(from.attachment.min(to.attachment), alpha, truncation)?;
    let base = from.expected(&p);
    if base <= 0.0 {
        return Err(invalid(
            "truncation",
            truncation.unwrap_or(f64::NAN),
            "leaves the from layer with no expected loss",
        ));
    }
    Ok(to.expected(&p) / base)
}

/// The Pareto alpha at which two layers have the given expected losses.
///
/// The layers may come in either order; the higher one (attachment and
/// exhaustion point no lower, not both equal) must have the lower rate,
/// below its rate at `alpha → 0` (where the survival function is flat).
/// Solved by bisection on `(0, MAX_ALPHA]`, on which the ratio of the two
/// layers' losses decreases strictly.
///
/// ```
/// use prospicio_pricing::layer::{XsLayer, alpha_between_layers, pareto_extrapolation};
///
/// let low = XsLayer::new(1.0e6, 1.0e6).unwrap();
/// let high = XsLayer::new(3.0e6, 2.0e6).unwrap();
/// let r = pareto_extrapolation(low, high, 1.7, None).unwrap();
/// let alpha = alpha_between_layers((low, 1.0), (high, r), None).unwrap();
/// assert!((alpha - 1.7).abs() < 1e-12);
/// ```
pub fn alpha_between_layers(
    a: (XsLayer, f64),
    b: (XsLayer, f64),
    truncation: Option<f64>,
) -> Result<f64> {
    positive("expected loss", a.1)?;
    positive("expected loss", b.1)?;
    let (low, high) = if (b.0.attachment, b.0.top()) >= (a.0.attachment, a.0.top()) {
        (a, b)
    } else {
        (b, a)
    };
    if high.0.attachment < low.0.attachment || high.0.top() < low.0.top() || high.0 == low.0 {
        return Err(invalid(
            "layers",
            high.0.attachment,
            "one layer must lie above the other (attachment and exhaustion point no lower)",
        ));
    }
    let target = high.1 / low.1;
    solve_alpha(target, |alpha| {
        pareto_extrapolation(low.0, high.0, alpha, truncation)
    })
}

/// The Pareto alpha at which `frequency` losses a year above `threshold`
/// give the layer an expected loss of `expected_loss`, with the frequency
/// of losses above `x` taken as `frequency · (threshold / x)^α` (truncated,
/// if set) on both sides of the threshold.
///
/// The threshold must not lie strictly inside the layer, since the layer
/// would then collect losses the frequency does not describe. Above the
/// threshold the layer's loss falls as alpha grows; below it, it rises.
///
/// ```
/// use prospicio_pricing::layer::{XsLayer, alpha_between_frequency_and_layer};
///
/// // 2 losses above 1m a year; at alpha 2, 1m xs 1m expects 2 · 0.5m.
/// let layer = XsLayer::new(1.0e6, 1.0e6).unwrap();
/// let alpha = alpha_between_frequency_and_layer(1.0e6, 2.0, layer, 1.0e6, None).unwrap();
/// assert!((alpha - 2.0).abs() < 1e-12);
/// ```
pub fn alpha_between_frequency_and_layer(
    threshold: f64,
    frequency: f64,
    layer: XsLayer,
    expected_loss: f64,
    truncation: Option<f64>,
) -> Result<f64> {
    positive("threshold", threshold)?;
    positive("frequency", frequency)?;
    positive("expected loss", expected_loss)?;
    if !threshold.is_finite() || (layer.attachment < threshold && threshold < layer.top()) {
        return Err(invalid(
            "threshold",
            threshold,
            "must be at or below the attachment, or at or above the exhaustion point",
        ));
    }
    // Per loss above the threshold: ∫_layer S(x) dx / S(threshold) for a
    // Pareto starting below both.
    let per_loss = |alpha: f64| -> Result<f64> {
        let p = pareto(threshold.min(layer.attachment), alpha, truncation)?;
        Ok(layer.expected(&p) / p.survival(threshold))
    };
    let target = expected_loss / frequency;
    if layer.attachment >= threshold {
        solve_alpha(target, per_loss)
    } else {
        // Below the threshold the loss rises with alpha: solve on its inverse.
        solve_alpha(1.0 / target, |alpha| Ok(1.0 / per_loss(alpha)?))
    }
}

/// The Pareto alpha between `frequency_1` losses a year above
/// `threshold_1` and `frequency_2` above `threshold_2`:
/// `ln(f_1 / f_2) / ln(t_2 / t_1)` untruncated, solved numerically with a
/// truncation. The higher threshold must have the lower frequency.
///
/// ```
/// use prospicio_pricing::layer::alpha_between_frequencies;
///
/// let alpha = alpha_between_frequencies(1.0e6, 4.0, 2.0e6, 1.0, None).unwrap();
/// assert!((alpha - 2.0).abs() < 1e-14);
/// ```
pub fn alpha_between_frequencies(
    threshold_1: f64,
    frequency_1: f64,
    threshold_2: f64,
    frequency_2: f64,
    truncation: Option<f64>,
) -> Result<f64> {
    for (name, v) in [
        ("threshold", threshold_1),
        ("threshold", threshold_2),
        ("frequency", frequency_1),
        ("frequency", frequency_2),
    ] {
        positive(name, v)?;
        if !v.is_finite() {
            return Err(invalid(name, v, "must be finite"));
        }
    }
    let ((t1, f1), (t2, f2)) = if threshold_1 <= threshold_2 {
        ((threshold_1, frequency_1), (threshold_2, frequency_2))
    } else {
        ((threshold_2, frequency_2), (threshold_1, frequency_1))
    };
    if t1 == t2 || f2 >= f1 {
        return Err(invalid(
            "frequency",
            f2,
            "the higher threshold must have the lower frequency",
        ));
    }
    match truncation {
        None => Ok((f1 / f2).ln() / (t2 / t1).ln()),
        Some(_) => solve_alpha(f2 / f1, |alpha| {
            Ok(pareto(t1, alpha, truncation)?.survival(t2))
        }),
    }
}

/// A Pareto from `t` with optional truncation.
fn pareto(t: f64, alpha: f64, truncation: Option<f64>) -> Result<Pareto> {
    let p = Pareto::new(t, alpha)?;
    match truncation {
        None => Ok(p),
        Some(tr) => p.truncated(tr),
    }
}

/// The alpha in `(0, MAX_ALPHA]` with `ratio(alpha) = target`, for a ratio
/// that decreases strictly in alpha.
fn solve_alpha(target: f64, ratio: impl Fn(f64) -> Result<f64>) -> Result<f64> {
    // Alpha this small leaves the survival function flat to rounding.
    let (lo, hi) = (1e-12, MAX_ALPHA);
    if target >= ratio(lo)? {
        return Err(invalid(
            "expected loss",
            target,
            "is too high for any positive alpha",
        ));
    }
    if target < ratio(hi)? {
        return Err(invalid(
            "expected loss",
            target,
            "implies an alpha above MAX_ALPHA",
        ));
    }
    // The ratio is finite on the bracket, so an error inside it is a bug in
    // the caller's ratio; keep the first one rather than lose it.
    let mut error = None;
    let alpha = bisect(lo, hi, |a| match ratio(a) {
        Ok(r) => r > target,
        Err(e) => {
            error.get_or_insert(e);
            false
        }
    });
    match error {
        Some(e) => Err(e),
        None => Ok(alpha),
    }
}

/// The coefficient of variation of the loss to `limit` xs `attachment`
/// per loss that reaches the layer: `sd / mean` of `min(max(X − a, 0), l)`
/// given `X > a`. It is what [`LimitedFluctuation`]'s pure premium
/// standard needs for a layer (`crate::credibility`), from the severity's
/// [`layer`](Severity::layer) and
/// [`layer_second_moment`](Severity::layer_second_moment).
///
/// [`LimitedFluctuation`]: crate::credibility::LimitedFluctuation
///
/// ```
/// use prospicio_pricing::layer::layer_cv;
/// use prospicio_prob::Gamma;
///
/// // Memoryless: an unlimited layer's loss given a loss above it is the
/// // exponential severity again, whose CV is 1.
/// let e = Gamma::new(1.0, 100.0).unwrap();
/// assert!((layer_cv(&e, f64::INFINITY, 50.0).unwrap() - 1.0).abs() < 1e-9);
/// ```
///
/// # Errors
///
/// An invalid layer, or no chance of a loss above the attachment.
pub fn layer_cv<S: Severity + ?Sized>(severity: &S, limit: f64, attachment: f64) -> Result<f64> {
    if limit.is_nan() || limit <= 0.0 || !(attachment.is_finite() && attachment >= 0.0) {
        return Err(Error::Data(format!(
            "the layer {limit} xs {attachment} needs a positive limit and a finite, \
             non-negative attachment"
        )));
    }
    let reach = severity.survival(attachment);
    if reach.is_nan() || reach <= 0.0 {
        return Err(Error::Data(format!(
            "no loss reaches the layer's attachment {attachment}"
        )));
    }
    let m1 = severity.layer(limit, attachment) / reach;
    let m2 = severity.layer_second_moment(limit, attachment) / reach;
    Ok((m2 - m1 * m1).max(0.0).sqrt() / m1)
}

/// Increased limit factor: the expected loss capped at `limit` relative
/// to the expected loss capped at `basic_limit`,
/// `ILF(limit) = LEV(limit) / LEV(basic_limit)`.
///
/// A limit of `+inf` gives the factor for unlimited cover (the mean over
/// `LEV(basic_limit)`).
///
/// ```
/// use prospicio_prob::Lognormal;
/// use prospicio_pricing::layer::ilf;
///
/// let sev = Lognormal::from_mean_cv(50_000.0, 3.0).unwrap();
/// let f = ilf(&sev, 1e6, 1e5).unwrap();
/// assert!(f > 1.0);
/// assert_eq!(ilf(&sev, 1e5, 1e5).unwrap(), 1.0);
/// ```
pub fn ilf<S: Severity + ?Sized>(severity: &S, limit: f64, basic_limit: f64) -> Result<f64> {
    positive("limit", limit)?;
    positive("basic_limit", basic_limit)?;
    if !basic_limit.is_finite() {
        return Err(Error::InvalidParameter {
            name: "basic_limit",
            value: basic_limit,
            reason: "must be finite",
        });
    }
    Ok(severity.lev(limit) / severity.lev(basic_limit))
}

/// Loss elimination ratio of a deductible: the share of expected ground-up
/// loss below `deductible`, `LEV(deductible) / E[X]`. This is the credit
/// for a straight deductible.
///
/// ```
/// use prospicio_prob::Lognormal;
/// use prospicio_pricing::layer::loss_elimination_ratio;
///
/// let sev = Lognormal::from_mean_cv(10_000.0, 1.0).unwrap();
/// let ler = loss_elimination_ratio(&sev, 1_000.0).unwrap();
/// assert!(ler > 0.0 && ler < 0.1);
/// ```
pub fn loss_elimination_ratio<S: Severity + ?Sized>(severity: &S, deductible: f64) -> Result<f64> {
    if deductible.is_nan() || deductible < 0.0 {
        return Err(Error::InvalidParameter {
            name: "deductible",
            value: deductible,
            reason: "must be non-negative",
        });
    }
    Ok(severity.lev(deductible) / severity.mean())
}

/// One layer of [`censored_las`]: from `bottom` to `top`, estimated from
/// the policies whose limit is at least `top`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CensoredLayer {
    /// The bottom of the layer (the next lower policy limit, or 0).
    pub bottom: f64,
    /// The top of the layer, a policy limit.
    pub top: f64,
    /// The claims of the policies with a limit at least `top`.
    pub claims: f64,
    /// Their average loss in the layer, `E[min(X, top) − min(X, bottom)]`.
    pub severity: f64,
    /// The limited average severity at `top`, the layers' severities up to
    /// this one added.
    pub las: f64,
}

/// The limited average severity at each policy limit from claims censored
/// at their policies' limits, layer by layer (Werner and Modlin, *Basic
/// Ratemaking*, section 11.3 and Table 11.6).
///
/// `tables` holds one size-of-loss table per policy limit, with the claims
/// capped at the limit (in the top band), so each table's last bound is its
/// limit. Between consecutive limits `l₋ < l`, the layer's average
/// severity uses only the policies with a limit at least `l`, whose claims
/// are uncensored up to `l`; `LAS(l)` adds the layers up to `l`. A claim
/// on a lower limit says nothing about the layers above it, so this is
/// below the LAS of all claims pooled when the losses are censored.
///
/// Each layer's bounds must be band edges of the tables it uses (or below
/// their first band), where a table's limited average severity is exact.
///
/// ```
/// use prospicio_prob::SizeOfLoss;
/// use prospicio_pricing::layer::censored_las;
///
/// // Werner and Modlin, Table 11.6: policies with limits of 100K, 250K and 500K.
/// let tables = [
///     SizeOfLoss::new(&[0.0, 100e3], &[2_019.0], &[156_657_898.0]).unwrap(),
///     SizeOfLoss::new(&[0.0, 100e3, 250e3], &[690.0, 773.0], &[34_903_214.0, 142_767_479.0])
///         .unwrap(),
///     SizeOfLoss::new(
///         &[0.0, 100e3, 250e3, 500e3],
///         &[712.0, 574.0, 232.0],
///         &[35_768_111.0, 90_009_422.0, 81_092_725.0],
///     )
///     .unwrap(),
/// ];
/// let layers = censored_las(&tables).unwrap();
/// assert!((layers[0].las - 77_045.8446).abs() < 1e-3);
/// assert!((layers[1].severity - 44_574.6).abs() < 0.1);
/// assert!((layers[2].las - 136_833.0).abs() < 0.1);
/// ```
pub fn censored_las(tables: &[SizeOfLoss]) -> Result<Vec<CensoredLayer>> {
    if tables.is_empty() {
        return Err(Error::Data("censored_las needs at least one table".into()));
    }
    let limit = |t: &SizeOfLoss| *t.bounds().last().expect("a table has bounds");
    if let Some(t) = tables.iter().find(|t| !limit(t).is_finite()) {
        return Err(Error::Data(format!(
            "each table's last bound is its policy limit and must be finite, not {}",
            limit(t)
        )));
    }
    let mut tops: Vec<f64> = tables.iter().map(limit).collect();
    tops.sort_by(f64::total_cmp);
    tops.dedup();
    let exact = |t: &SizeOfLoss, x: f64| x <= t.bounds()[0] || t.bounds().contains(&x);
    let (mut bottom, mut las) = (0.0, 0.0);
    let mut layers = Vec::with_capacity(tops.len());
    for top in tops {
        let (mut claims, mut total) = (0.0, 0.0);
        for t in tables.iter().filter(|t| limit(t) >= top) {
            if let Some(&x) = [bottom, top].iter().find(|&&x| !exact(t, x)) {
                return Err(Error::Data(format!(
                    "{x} is a layer bound but falls inside a band of the table with limit {}",
                    limit(t)
                )));
            }
            claims += t.claims();
            total += t.claims() * (t.lev(top) - t.lev(bottom));
        }
        let severity = total / claims;
        las += severity;
        layers.push(CensoredLayer {
            bottom,
            top,
            claims,
            severity,
            las,
        });
        bottom = top;
    }
    Ok(layers)
}

fn invalid(name: &'static str, value: f64, reason: &'static str) -> Error {
    Error::InvalidParameter {
        name,
        value,
        reason,
    }
}

fn positive(name: &'static str, value: f64) -> Result<()> {
    if value.is_nan() || value <= 0.0 {
        return Err(Error::InvalidParameter {
            name,
            value,
            reason: "must be positive",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::{Distribution, Lognormal};

    #[test]
    fn ilf_is_a_ratio_of_limited_means() {
        let sev = Lognormal::new(9.0, 1.5).unwrap();
        let f = ilf(&sev, 250_000.0, 100_000.0).unwrap();
        assert!((f - sev.lev(250_000.0) / sev.lev(100_000.0)).abs() < 1e-15);
        // Increasing in the limit, unlimited cover the largest.
        let g = ilf(&sev, 1e6, 100_000.0).unwrap();
        let unlimited = ilf(&sev, f64::INFINITY, 100_000.0).unwrap();
        assert!(1.0 < f && f < g && g < unlimited);
        assert!((unlimited - sev.mean() / sev.lev(100_000.0)).abs() < 1e-12);
        assert!(ilf(&sev, 0.0, 1.0).is_err());
        assert!(ilf(&sev, 1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn loss_elimination_ratio_bounds() {
        let sev = Lognormal::new(9.0, 1.5).unwrap();
        assert_eq!(loss_elimination_ratio(&sev, 0.0).unwrap(), 0.0);
        let full = loss_elimination_ratio(&sev, f64::INFINITY).unwrap();
        assert!((full - 1.0).abs() < 1e-15);
        let ler = loss_elimination_ratio(&sev, 5_000.0).unwrap();
        assert!((ler - sev.lev(5_000.0) / sev.mean()).abs() < 1e-15);
        assert!(loss_elimination_ratio(&sev, -1.0).is_err());
    }

    fn xs(limit: f64, attachment: f64) -> XsLayer {
        XsLayer::new(limit, attachment).unwrap()
    }

    #[test]
    fn extrapolation_matches_pareto_layers() {
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0] {
            let p = Pareto::new(500.0, alpha).unwrap();
            let (from, to) = (xs(1000.0, 1000.0), xs(f64::INFINITY, 5000.0));
            if alpha <= 1.0 {
                assert_eq!(
                    pareto_extrapolation(from, to, alpha, None).unwrap(),
                    f64::INFINITY
                );
                continue;
            }
            let want = p.layer(f64::INFINITY, 5000.0) / p.layer(1000.0, 1000.0);
            let got = pareto_extrapolation(from, to, alpha, None).unwrap();
            assert!((got / want - 1.0).abs() < 1e-13, "{alpha}");
        }
        // Rating down works too, and truncation changes the answer.
        let (lo, hi) = (xs(500.0, 500.0), xs(1000.0, 2000.0));
        let up = pareto_extrapolation(lo, hi, 1.5, None).unwrap();
        let down = pareto_extrapolation(hi, lo, 1.5, None).unwrap();
        assert!((up * down - 1.0).abs() < 1e-14);
        let t = pareto_extrapolation(lo, hi, 1.5, Some(2500.0)).unwrap();
        assert!(t < up);
        assert!(pareto_extrapolation(lo, hi, 1.5, Some(400.0)).is_err());
        assert!(pareto_extrapolation(lo, hi, 0.0, None).is_err());
    }

    #[test]
    fn implied_alphas_round_trip() {
        let (low, high) = (xs(1000.0, 1000.0), xs(3000.0, 2000.0));
        for alpha in [0.3, 1.0, 1.999, 2.0, 4.5, 40.0] {
            for tr in [None, Some(20_000.0)] {
                let r = pareto_extrapolation(low, high, alpha, tr).unwrap();
                let back = alpha_between_layers((high, 5.0 * r), (low, 5.0), tr).unwrap();
                assert!((back / alpha - 1.0).abs() < 1e-11, "{alpha} {tr:?} {back}");

                let p = pareto(800.0, alpha, tr).unwrap();
                let e = 3.0 * high.expected(&p);
                let a = alpha_between_frequency_and_layer(800.0, 3.0, high, e, tr).unwrap();
                assert!((a / alpha - 1.0).abs() < 1e-11, "{alpha} {tr:?} {a}");
                // A threshold above the layer: 3 · S(5000) losses above 5000.
                let f = 3.0 * p.survival(5000.0);
                let e = 3.0 * low.expected(&p);
                if alpha < 20.0 {
                    let a = alpha_between_frequency_and_layer(5000.0, f, low, e, tr).unwrap();
                    assert!((a / alpha - 1.0).abs() < 1e-10, "{alpha} {tr:?} {a}");
                }

                let f2 = 3.0 * p.survival(5000.0);
                let a = alpha_between_frequencies(5000.0, f2, 800.0, 3.0, tr).unwrap();
                assert!((a / alpha - 1.0).abs() < 1e-11, "{alpha} {tr:?} {a}");
            }
        }
    }

    #[test]
    fn implied_alpha_rejects_impossible_inputs() {
        let (low, high) = (xs(1000.0, 1000.0), xs(1000.0, 2000.0));
        // Flat survival gives equal losses for equal widths: no alpha above.
        assert!(alpha_between_layers((low, 1.0), (high, 1.0), None).is_err());
        // Far too steep.
        assert!(alpha_between_layers((low, 1.0), (high, 1e-40), None).is_err());
        // Overlapping layers with neither above the other.
        let wide = xs(5000.0, 500.0);
        assert!(alpha_between_layers((low, 1.0), (wide, 0.5), None).is_err());
        assert!(alpha_between_layers((low, 1.0), (low, 0.5), None).is_err());
        // Threshold inside the layer, or a loss above frequency · limit
        // (or below it, under the threshold).
        assert!(alpha_between_frequency_and_layer(1500.0, 1.0, low, 0.5, None).is_err());
        assert!(alpha_between_frequency_and_layer(500.0, 1.0, low, 1001.0, None).is_err());
        assert!(alpha_between_frequency_and_layer(5000.0, 1.0, low, 999.0, None).is_err());
        assert!(alpha_between_frequencies(1.0, 1.0, 2.0, 1.0, None).is_err());
        assert!(XsLayer::new(1.0, 0.0).is_err());
    }
}
