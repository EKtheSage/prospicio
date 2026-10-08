//! Choosing the FFT grid: the bucket size and the number of points.
//!
//! [`recommend_grid`] sizes a 0-based grid for a compound distribution from
//! the frequency and severity alone, as `aggregate`'s `Aggregate.update`
//! does when no `bs` is given (`docs/design/aggregate.md`, "Grid sizing").
//! [`fft_auto`] sizes, discretizes the severity by rounding, and runs the
//! FFT.

use prospicio_core::{Error, Result};
use prospicio_prob::{Counting, Distribution, Gamma, Grid, Lognormal, Severity};

use crate::compound::CompoundReport;
use crate::fft::fft;

/// Rounds a bucket size *up* to a "nice" value, as `aggregate`'s
/// `round_bucket`: at 1 and above, the smallest of `{1, 2, 4, 5, 8} × 10^k`
/// at least `bs`; below 1, the smallest power of two at least `bs` (exact in
/// floating point). Every rung is at most twice the one below, so the
/// result is less than `2 bs`.
///
/// ```
/// use prospicio_aggregate::round_bucket;
///
/// assert_eq!(round_bucket(3.4).unwrap(), 4.0);
/// assert_eq!(round_bucket(5.5).unwrap(), 8.0);
/// assert_eq!(round_bucket(2412.0).unwrap(), 4000.0);
/// assert_eq!(round_bucket(0.3).unwrap(), 0.5);
/// ```
pub fn round_bucket(bs: f64) -> Result<f64> {
    if !(bs.is_finite() && bs > 0.0) {
        return Err(Error::InvalidParameter {
            name: "bs",
            value: bs,
            reason: "must be finite and positive",
        });
    }
    if bs >= 1.0 {
        let base = 10f64.powf(bs.log10().floor());
        let m = bs / base;
        for rung in [1.0, 2.0, 4.0, 5.0, 8.0, 10.0] {
            if m <= rung * (1.0 + 1e-9) {
                return Ok(rung * base);
            }
        }
        return Ok(10.0 * base);
    }
    // The largest power of two at most floor(1/bs), inverted.
    let n = (1.0 / bs).floor();
    Ok(2f64.powi(-(n.log2().floor() as i32)))
}

/// What sized the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizingMethod {
    /// A lognormal or gamma fitted to the aggregate's mean and variance, at
    /// probability `p`.
    Moments,
    /// One large claim on a typical bulk, `E[S] - E[X] + q_X(p**)`: at
    /// probability `p` when it reaches past the moment extent (or the
    /// variance is infinite), or at `p_star` when that fits at the same
    /// bucket.
    SingleBigJump,
}

/// The settings [`recommend_grid`] sizes with; the defaults are
/// `aggregate` 1.0.1's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sizing {
    /// The grid has at most `2^log2` points. Default 16.
    pub log2: u32,
    /// Probability of the moment extent. Default `1 - 1e-5`
    /// (`aggregate`'s `bucket_sizing_p`).
    pub p: f64,
    /// Aggregate probability the single big jump covers. Default
    /// `1 - 1e-12` (`aggregate`'s `window_nines`).
    pub p_star: f64,
    /// Floor on the severity tail probability `(1 - p_star) / E[N]` the
    /// single big jump probes, so its quantile stays finite. Default `1e-14`.
    pub tail_floor: f64,
}

impl Default for Sizing {
    fn default() -> Self {
        Self {
            log2: 16,
            p: 1.0 - 1e-5,
            p_star: 1.0 - 1e-12,
            tail_floor: 1e-14,
        }
    }
}

/// A recommended FFT grid and how it was found.
#[derive(Debug, Clone, PartialEq)]
pub struct GridSize {
    /// Bucket size, a [`round_bucket`] rung.
    pub step: f64,
    /// Number of points, a power of two at most `2^log2`.
    pub points: usize,
    /// The extent the grid was sized to cover.
    pub extent: f64,
    pub method: SizingMethod,
    /// The moment extent, when the variance is finite.
    pub moment_extent: Option<f64>,
    /// The single-big-jump extent, when the severity quantile it needs is
    /// finite.
    pub jump_extent: Option<f64>,
    /// `min(1, E[N] S_X(top))` at the grid's top `points × step`: roughly
    /// the aggregate probability beyond the grid, from one claim alone. Large
    /// when the single big jump did not fit; raise `log2` then.
    pub tail_estimate: f64,
}

/// A 0-based grid for the compound distribution of `frequency` claims of
/// `severity`, as `aggregate` 1.0.1 sizes one when no bucket is given.
///
/// 1. The aggregate mean `E[N] E[X]` and variance
///    `E[N] Var X + Var N E[X]²`; the moment extent is the larger of the
///    lognormal's and the gamma's `p` quantile at that mean and coefficient
///    of variation (`aggregate` fits shifted lognormal and gamma to three
///    moments; [`Severity`] stops at two).
/// 2. The single big jump at the same `p`,
///    `E[S] - E[X] + q_X(1 - (1 - p)/E[N])`, when it is larger: one claim
///    alone then leaves about `1 - p` beyond the grid. (`aggregate` sizes by
///    its three-moment fits only, which reach further on skewed books.)
/// 3. `step = round_bucket(extent / 2^log2)`, and the fewest points
///    `2^k`, `k <= log2`, that cover `extent / step + 1` buckets; when even
///    `2^log2` do not, `step = round_bucket(extent / (2^log2 - 1))`.
/// 4. The single big jump `E[S] - E[X] + q_X(1 - max((1 - p*)/E[N], floor))`,
///    capped at the severity's top: when it reaches past the moment extent
///    and fits at the same step within `2^log2` points, the grid grows to
///    cover it. A far tail never coarsens the step.
/// 5. With an infinite variance, the single big jump at `p` alone sizes
///    the bulk (`aggregate` refuses such books without an explicit step).
///
/// Fails when the mean is not finite and positive (an infinite mean needs
/// a step chosen by hand) or `log2` is not in `1..=30`.
///
/// ```
/// use prospicio_aggregate::{Sizing, SizingMethod, recommend_grid};
/// use prospicio_prob::{Gamma, Poisson};
///
/// let g = recommend_grid(&Poisson::new(10.0).unwrap(), &Gamma::new(2.0, 50.0).unwrap(),
///                        &Sizing::default()).unwrap();
/// assert_eq!(g.method, SizingMethod::Moments);
/// assert!(g.points.is_power_of_two() && g.points <= 1 << 16);
/// assert!(g.step * g.points as f64 > g.extent);
/// ```
pub fn recommend_grid<N, X>(frequency: &N, severity: &X, sizing: &Sizing) -> Result<GridSize>
where
    N: Counting + ?Sized,
    X: Severity + ?Sized,
{
    if !(1..=30).contains(&sizing.log2) {
        return Err(Error::InvalidParameter {
            name: "log2",
            value: f64::from(sizing.log2),
            reason: "must be in 1..=30",
        });
    }
    for (name, v) in [("p", sizing.p), ("p_star", sizing.p_star)] {
        if !(v > 0.0 && v < 1.0) {
            return Err(Error::InvalidParameter {
                name,
                value: v,
                reason: "must be in (0, 1)",
            });
        }
    }
    let en = frequency.mean();
    let mx = severity.mean();
    let mean = en * mx;
    if !(mean.is_finite() && mean > 0.0) {
        return Err(Error::InvalidParameter {
            name: "severity",
            value: mean,
            reason: "the aggregate mean must be finite and positive; choose the step by hand",
        });
    }
    let variance = en * severity.variance() + frequency.variance() * mx * mx;
    let moment_extent = if variance.is_finite() {
        Some(moment_extent(mean, variance, sizing.p)?)
    } else {
        None
    };
    // One big claim at the bulk's own probability: the grid must hold it
    // too, so one claim leaves about 1 - p beyond the top.
    let bulk_jump = jump_extent(en, mx, severity, sizing.p, sizing.tail_floor);
    let jump_extent = jump_extent(en, mx, severity, sizing.p_star, sizing.tail_floor);
    let cap = 1usize << sizing.log2;

    let (base, mut method) = match (moment_extent, bulk_jump) {
        (Some(m), Some(j)) if j > m => (j, SizingMethod::SingleBigJump),
        (Some(m), _) => (m, SizingMethod::Moments),
        (None, Some(j)) => (j, SizingMethod::SingleBigJump),
        (None, None) => {
            return Err(Error::InvalidParameter {
                name: "severity",
                value: variance,
                reason: "infinite variance and no finite severity quantile to size the grid by",
            });
        }
    };
    let (step, mut need) = size(base, cap, sizing.log2)?;
    let mut extent = base;
    if let Some(jump) = jump_extent
        && jump > base
    {
        let k = need_log2(jump, step);
        if k <= sizing.log2 {
            need = need.max(k);
            extent = jump;
            method = SizingMethod::SingleBigJump;
        }
    }
    let points = 1usize << need.max(1);
    let top = points as f64 * step;
    let tail_estimate = (en * severity.survival(top)).min(1.0);
    Ok(GridSize {
        step,
        points,
        extent,
        method,
        moment_extent,
        jump_extent,
        tail_estimate,
    })
}

/// The larger of the lognormal's and the gamma's `p` quantile with the
/// given mean and variance; the mean itself for a point mass.
fn moment_extent(mean: f64, variance: f64, p: f64) -> Result<f64> {
    if variance <= 0.0 {
        return Ok(mean);
    }
    let cv = variance.sqrt() / mean;
    let ln = Lognormal::from_mean_cv(mean, cv)?.quantile(p)?;
    let ga = Gamma::from_mean_cv(mean, cv)?.quantile(p)?;
    Ok(ln.max(ga))
}

/// `E[S] - E[X] + q_X(p**)` with `1 - p** = max((1 - p)/E[N], floor)`,
/// the quantile capped at the severity's top; `None` without a finite
/// quantile or with fewer than one expected claim.
fn jump_extent<X: Severity + ?Sized>(
    en: f64,
    mx: f64,
    severity: &X,
    p: f64,
    floor: f64,
) -> Option<f64> {
    if !(en.is_finite() && en >= 1.0) {
        return None;
    }
    let tail = ((1.0 - p) / en).max(floor);
    let q = severity.quantile(1.0 - tail).ok()?;
    let top = severity.quantile(1.0).ok().unwrap_or(f64::INFINITY);
    let q = q.min(top);
    q.is_finite().then_some(en * mx - mx + q)
}

/// `(step, log2)` for a 0-based window of width `extent`, as `aggregate`'s
/// `_size`: the resolution `round_bucket(extent / 2^cap)`, then the fewest
/// points that cover it, coarsening to `extent / (2^cap - 1)` when the cap
/// is not enough.
fn size(extent: f64, cap: usize, log2: u32) -> Result<(f64, u32)> {
    if extent <= 0.0 {
        return Ok((1.0, 1));
    }
    let step = round_bucket(extent / cap as f64)?;
    let need = need_log2(extent, step);
    if need <= log2 {
        return Ok((step, need.max(1)));
    }
    Ok((round_bucket(extent / (cap - 1) as f64)?, log2))
}

/// `ceil(log2(span / step + 1))`: the power of two a closed window of
/// width `span` needs at `step`.
fn need_log2(span: f64, step: f64) -> u32 {
    let ratio = (span / step + 1.0).max(1.0);
    if !ratio.is_finite() {
        return u32::MAX;
    }
    ratio.log2().ceil() as u32
}

/// The compound distribution on a grid [`recommend_grid`] chooses: the
/// severity discretized by rounding (`aggregate`'s default) on that step
/// and number of points, then [`fft`].
///
/// ```
/// use prospicio_aggregate::{Sizing, fft_auto};
/// use prospicio_prob::{Distribution, Lognormal, Poisson};
///
/// let sev = Lognormal::from_mean_cv(100.0, 1.0).unwrap();
/// let (agg, report, size) = fft_auto(&Poisson::new(5.0).unwrap(), &sev, &Sizing::default()).unwrap();
/// assert!(report.aliasing_error < 1e-12);
/// assert!((agg.mean() / 500.0 - 1.0).abs() < 1e-3);
/// assert_eq!(agg.probs().len(), size.points);
/// ```
pub fn fft_auto<N, X>(
    frequency: &N,
    severity: &X,
    sizing: &Sizing,
) -> Result<(Grid, CompoundReport, GridSize)>
where
    N: Counting + ?Sized,
    X: Severity,
{
    let size = recommend_grid(frequency, severity, sizing)?;
    let (sev, _) = Grid::rounding(severity, size.step, size.points)?;
    let (agg, report) = fft(frequency, &sev, size.points)?;
    Ok((agg, report, size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::{NegativeBinomial, Pareto, Poisson};

    #[test]
    fn round_bucket_matches_aggregate_rungs() {
        // aggregate 1.0.1's docstring cases.
        for (x, want) in [
            (1.0, 1.0),
            (1.1, 2.0),
            (2.5, 4.0),
            (9.9, 10.0),
            (13.0, 20.0),
            (457.0, 500.0),
            (57_000.0, 80_000.0),
            (0.1, 0.125),
            (0.5, 0.5),
            (1.0 / 457.0, 1.0 / 256.0),
        ] {
            assert_eq!(round_bucket(x).unwrap(), want, "{x}");
        }
        assert!(round_bucket(0.0).is_err());
    }

    #[test]
    fn heavy_tails_grow_by_the_single_big_jump() {
        // Pareto alpha 1.5: infinite variance, so the jump sizes alone.
        let p = Pareto::new(100.0, 1.5).unwrap();
        let g = recommend_grid(&Poisson::new(20.0).unwrap(), &p, &Sizing::default()).unwrap();
        assert_eq!(g.method, SizingMethod::SingleBigJump);
        assert!(g.moment_extent.is_none());
        assert!(g.step * g.points as f64 > g.extent);
        // A light severity stays on its moments.
        let ga = Gamma::new(5.0, 20.0).unwrap();
        let n = NegativeBinomial::new(4.0, 2.0).unwrap();
        let g = recommend_grid(&n, &ga, &Sizing::default()).unwrap();
        assert_eq!(g.method, SizingMethod::Moments);
        assert!(g.tail_estimate < 1e-10);
    }

    #[test]
    fn fewer_points_when_the_cap_is_not_needed() {
        let g = recommend_grid(
            &Poisson::new(2.0).unwrap(),
            &Gamma::new(2.0, 1.0).unwrap(),
            &Sizing {
                log2: 10,
                ..Sizing::default()
            },
        )
        .unwrap();
        assert!(g.points <= 1 << 10);
        assert!(g.step < 1.0);
        let bad = Sizing {
            log2: 0,
            ..Sizing::default()
        };
        assert!(
            recommend_grid(
                &Poisson::new(2.0).unwrap(),
                &Gamma::new(2.0, 1.0).unwrap(),
                &bad
            )
            .is_err()
        );
    }
}
