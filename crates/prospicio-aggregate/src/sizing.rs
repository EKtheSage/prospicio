//! Choosing the FFT grid: the bucket size and the number of points.
//!
//! [`recommend_grid`] sizes a 0-based grid for a compound distribution from
//! the frequency and severity alone, as `aggregate`'s `Aggregate.update`
//! does when no `bs` is given (`docs/design/aggregate.md`, "Grid sizing").
//! [`fft_auto`] sizes, discretizes the severity by rounding, and runs the
//! FFT. [`recommend_portfolio_grid`] sizes one grid for several independent
//! units. [`audit_grid`] and [`audit_portfolio_grid`] audit a grid after the
//! fact, as `aggregate`'s `sharpen`: they rebuild the distribution on
//! neighbouring steps and sizes, score each by how well it reproduces the
//! exact moments, and pick the best one that costs no more points (#317).

use prospicio_core::{Error, Result};
use prospicio_prob::{Counting, Distribution, Gamma, Grid, Lognormal, Severity};

use crate::compound::CompoundReport;
use crate::fft::{compound_buffer, fft};

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
    check_sizing(sizing)?;
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
    finish(
        moment_extent,
        bulk_jump,
        jump_extent,
        variance,
        sizing,
        |top| (en * severity.survival(top)).min(1.0),
    )
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

/// A grid for the sum of independent compound units, as `aggregate`
/// sizes a `Portfolio`: [`recommend_grid`]'s steps on the total. The
/// moment extent is the larger of the lognormal's and the gamma's `p`
/// quantile at the total's mean and variance (the units' added); the
/// single big jump is the largest over the units of the total mean less
/// that unit's severity mean plus its severity quantile, so one large claim
/// on any unit, on a typical rest of the book, fits.
///
/// Each unit is `(frequency, severity)`. Fails as [`recommend_grid`] does,
/// or when there are no units.
///
/// ```
/// use prospicio_aggregate::{Sizing, recommend_portfolio_grid};
/// use prospicio_prob::{Counting, Gamma, Lognormal, Poisson, Severity};
///
/// let (n1, x1) = (Poisson::new(10.0).unwrap(), Gamma::new(2.0, 50.0).unwrap());
/// let (n2, x2) = (Poisson::new(2.0).unwrap(), Lognormal::from_mean_cv(500.0, 2.0).unwrap());
/// let units: [(&dyn Counting, &dyn Severity); 2] = [(&n1, &x1), (&n2, &x2)];
/// let g = recommend_portfolio_grid(&units, &Sizing::default()).unwrap();
/// assert!(g.points.is_power_of_two() && g.step * g.points as f64 > g.extent);
/// ```
pub fn recommend_portfolio_grid(
    units: &[(&dyn Counting, &dyn Severity)],
    sizing: &Sizing,
) -> Result<GridSize> {
    if units.is_empty() {
        return Err(Error::Data("a portfolio needs at least one unit".into()));
    }
    check_sizing(sizing)?;
    let (mut mean, mut variance) = (0.0, 0.0);
    for (n, x) in units {
        let (en, mx) = (n.mean(), x.mean());
        mean += en * mx;
        variance += en * x.variance() + n.variance() * mx * mx;
    }
    if !(mean.is_finite() && mean > 0.0) {
        return Err(Error::InvalidParameter {
            name: "severity",
            value: mean,
            reason: "the total mean must be finite and positive; choose the step by hand",
        });
    }
    let moment_extent = if variance.is_finite() {
        Some(moment_extent(mean, variance, sizing.p)?)
    } else {
        None
    };
    // One large claim on one unit, the rest of the book at its mean.
    let jump = |p: f64| {
        units
            .iter()
            .filter_map(|(n, x)| {
                let (en, mx) = (n.mean(), x.mean());
                jump_extent(en.max(1.0), mx, *x, p, sizing.tail_floor)
                    .map(|j| j - en.max(1.0) * mx + mean)
            })
            .reduce(f64::max)
    };
    let tail_at = |top: f64| {
        units
            .iter()
            .map(|(n, x)| n.mean() * x.survival(top))
            .sum::<f64>()
            .min(1.0)
    };
    finish(
        moment_extent,
        jump(sizing.p),
        jump(sizing.p_star),
        variance,
        sizing,
        tail_at,
    )
}

fn check_sizing(sizing: &Sizing) -> Result<()> {
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
    Ok(())
}

/// Steps 3 to 5 of [`recommend_grid`], from the extents.
fn finish(
    moment_extent: Option<f64>,
    bulk_jump: Option<f64>,
    jump_extent: Option<f64>,
    variance: f64,
    sizing: &Sizing,
    tail_at: impl Fn(f64) -> f64,
) -> Result<GridSize> {
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
    if let Some(jump) = jump_extent.filter(|&j| j > base) {
        let k = need_log2(jump, step);
        if k <= sizing.log2 {
            need = need.max(k);
            extent = jump;
            method = SizingMethod::SingleBigJump;
        }
    }
    let points = 1usize << need.max(1);
    let tail_estimate = tail_at(points as f64 * step);
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

/// The settings of [`audit_grid`]; the defaults are `aggregate` 1.2.0's
/// `sharpen`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuditSettings {
    /// The validation tolerance: a relative moment error of `eps` on a
    /// mean, `10 eps` on a cv and `100 eps` on a skewness scores 1.
    /// Default `1e-4`.
    pub eps: f64,
    /// The power mean that combines the terms. Default 2.
    pub power: f64,
    /// The target score. A center at or below it (and sound) is not
    /// probed; the number of points grows only when no cell at the
    /// center's size or smaller reaches it. Default 0.5.
    pub good_enough: f64,
    /// The line search doubles or halves the step at most this many times
    /// each way. Default 4 (`aggregate`'s `bs_limit = 16`).
    pub max_doublings: u32,
    /// The number of points never grows past `2^log2_cap`. Default 20.
    pub log2_cap: u32,
    /// The row with half the points is searched only when the center has
    /// more than `2^log2_floor`. Default 8.
    pub log2_floor: u32,
}

impl Default for AuditSettings {
    fn default() -> Self {
        Self {
            eps: 1e-4,
            power: 2.0,
            good_enough: 0.5,
            max_doublings: 4,
            log2_cap: 20,
            log2_floor: 8,
        }
    }
}

/// Moment errors in units of their tolerance, `|error| / (k eps)`; `None`
/// for a term whose exact value is not finite or is zero (a symmetric
/// severity's skewness), which no grid can be scored on.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScoreTerms {
    pub severity_mean: Option<f64>,
    pub severity_cv: Option<f64>,
    pub severity_skew: Option<f64>,
    pub aggregate_mean: Option<f64>,
    pub aggregate_cv: Option<f64>,
    pub aggregate_skew: Option<f64>,
}

impl ScoreTerms {
    fn all(&self) -> [Option<f64>; 6] {
        [
            self.severity_mean,
            self.severity_cv,
            self.severity_skew,
            self.aggregate_mean,
            self.aggregate_cv,
            self.aggregate_skew,
        ]
    }

    /// `(mean u^power)^(1 / power)` over the live terms, the largest for an
    /// infinite power; NaN with none.
    pub fn score(&self, power: f64) -> f64 {
        let live: Vec<f64> = self.all().into_iter().flatten().collect();
        if live.is_empty() {
            return f64::NAN;
        }
        if live.iter().any(|u| u.is_infinite()) {
            return f64::INFINITY;
        }
        if power.is_infinite() {
            return live.iter().copied().fold(0.0, f64::max);
        }
        (live.iter().map(|u| u.powf(power)).sum::<f64>() / live.len() as f64).powf(1.0 / power)
    }
}

/// One grid the audit built.
#[derive(Debug, Clone, PartialEq)]
pub struct GridCell {
    /// Doublings of the step from the center (negative halves it).
    pub d_step: i32,
    /// Change of `log2` from the center.
    pub d_log2: i32,
    pub step: f64,
    pub log2: u32,
    /// The combined score, NaN when the cell failed to build.
    pub score: f64,
    pub terms: ScoreTerms,
    /// The aggregate mass beyond the grid, lumped onto its last point.
    pub tail_mass: f64,
    pub aliasing_error: f64,
    /// Mass beyond the grid above `1e-12`: such a cell is chosen only when
    /// no sound one scored.
    pub defective: bool,
    pub selected: bool,
    /// Why the line search stopped here, or why the cell failed.
    pub note: String,
}

/// The cells an audit built and the one it chose.
#[derive(Debug, Clone, PartialEq)]
pub struct GridAudit {
    /// Every cell, the center first.
    pub cells: Vec<GridCell>,
    /// The index of the chosen cell.
    pub selected: usize,
    /// False when the center met the target and nothing was probed.
    pub probed: bool,
    /// True when the chosen cell has more points than the center.
    pub grew: bool,
    /// Why that cell.
    pub reason: String,
}

impl GridAudit {
    /// The chosen cell.
    pub fn best(&self) -> &GridCell {
        &self.cells[self.selected]
    }
}

/// What one cell's build reports.
struct CellBuild {
    terms: ScoreTerms,
    tail_mass: f64,
    aliasing_error: f64,
}

/// Errors below this are arithmetic dust, as `aggregate`'s
/// `VALIDATION_NOISE`.
const NOISE: f64 = 1e-12;
/// Scores within this factor of the best are a tie, broken by fewer
/// points and then a smaller move (`aggregate`'s `SHARPEN_FALLBACK_SLACK`).
const SLACK: f64 = 1.25;

/// `|error| / (k eps)`: the relative error, absolute when the exact value is
/// within the noise of 0; `None` for an exact value that cannot be scored.
fn term(empirical: f64, exact: f64, k: f64, eps: f64) -> Option<f64> {
    if !(exact.is_finite() && exact.abs() > NOISE) {
        return None;
    }
    let e = empirical / exact - 1.0;
    Some(if e.is_finite() {
        e.abs() / (k * eps)
    } else {
        f64::INFINITY
    })
}

/// Mean, cv and skewness from the first three raw moments.
fn shape(m1: f64, m2: f64, m3: f64) -> (f64, f64, f64) {
    let var = (m2 - m1 * m1).max(0.0);
    let k3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 * m1 * m1;
    (m1, var.sqrt() / m1, k3 / var.powf(1.5))
}

fn raw_moments(g: &Grid) -> (f64, f64, f64) {
    let (mut m1, mut m2, mut m3) = (0.0, 0.0, 0.0);
    for (j, &p) in g.probs().iter().enumerate() {
        let x = j as f64 * g.step();
        m1 += x * p;
        m2 += x * x * p;
        m3 += x * x * x * p;
    }
    (m1, m2, m3)
}

/// The exact first three raw moments of a severity: its closed-form third
/// moment, or the quadrature of the unlimited layer's.
fn exact_raw<X: Severity + ?Sized>(x: &X) -> (f64, f64, f64) {
    let m1 = x.mean();
    let m2 = x.variance() + m1 * m1;
    let m3 = x
        .third_moment()
        .unwrap_or_else(|| x.layer_third_moment(f64::INFINITY, 0.0));
    (m1, m2, m3)
}

fn severity_terms(t: &mut ScoreTerms, grid: (f64, f64, f64), exact: (f64, f64, f64), eps: f64) {
    let (gm, gc, gs) = shape(grid.0, grid.1, grid.2);
    let (xm, xc, xs) = shape(exact.0, exact.1, exact.2);
    t.severity_mean = term(gm, xm, 1.0, eps);
    t.severity_cv = term(gc, xc, 10.0, eps);
    t.severity_skew = term(gs, xs, 100.0, eps);
}

fn aggregate_terms(t: &mut ScoreTerms, agg: &Grid, exact: (f64, f64, f64), eps: f64) {
    let (m1, m2, m3) = raw_moments(agg);
    let (gm, gc, gs) = shape(m1, m2, m3);
    let (mean, var, k3) = exact;
    t.aggregate_mean = term(gm, mean, 1.0, eps);
    t.aggregate_cv = term(gc, var.sqrt() / mean, 10.0, eps);
    t.aggregate_skew = term(gs, k3 / var.powf(1.5), 100.0, eps);
}

/// Audits the grid `(step, 2^log2)` for `frequency` claims of `severity`,
/// as `aggregate` 1.2.0's `sharpen` does (#317).
///
/// Each cell discretizes the severity by rounding on its step and number
/// of points, runs [`fft`], and is scored by six moment errors: the
/// severity's and the aggregate's mean, cv and skewness against their exact
/// values, each divided by its tolerance (`eps`, `10 eps`, `100 eps`) and
/// combined by a power mean ([`ScoreTerms::score`]). A score of 1 is
/// `aggregate`'s validation boundary. Small is good.
///
/// 1. A center that scores at or below `good_enough`, with no mass beyond
///    the grid, is kept and nothing else is built.
/// 2. Otherwise the step is doubled until the score stops improving, then
///    halved until it stops improving, at most `max_doublings` times each
///    way, at the center's size and at half of it (above
///    `2^log2_floor`). At a fixed size the score is single-troughed in the
///    step (a larger step buys extent and loses resolution), so the first
///    worse cell ends each walk.
/// 3. The choice is the best sound score among those cells, scores within
///    a factor 1.25 counting as tied and ties going to fewer points, then
///    to a smaller move. Only when that misses the target is the row with
///    twice the points searched (up to `2^log2_cap`), and taken if it
///    reaches it; failing that, the best cell anywhere.
///
/// The exact moments need the severity's third moment: its closed form,
/// or else the quadrature of `E[X³]` (finite for any limited severity).
///
/// ```
/// use prospicio_aggregate::{AuditSettings, audit_grid};
/// use prospicio_prob::{Dist, Lognormal, PolicyLimits, Poisson, SeverityDist};
///
/// // aggregate's `100 claims 1000 xs 0 sev lognorm 100 cv 1.5`, from a step of 1
/// // on 2^13 points: the grid ends at 8,192, short of the book.
/// let ln = SeverityDist::try_from(Dist::from(Lognormal::from_mean_cv(100.0, 1.5).unwrap())).unwrap();
/// let capped = PolicyLimits::new(ln, vec![1000.0], vec![1.0]).unwrap();
/// let audit = audit_grid(&Poisson::new(100.0).unwrap(), &capped, 1.0, 13, &AuditSettings::default())
///     .unwrap();
/// assert_eq!((audit.best().step, audit.best().log2), (4.0, 13));
/// assert!(audit.best().score < 0.5 && audit.cells[0].score > 500.0);
/// ```
pub fn audit_grid<N, X>(
    frequency: &N,
    severity: &X,
    step: f64,
    log2: u32,
    settings: &AuditSettings,
) -> Result<GridAudit>
where
    N: Counting + ?Sized,
    X: Severity,
{
    let exact_sev = exact_raw(severity);
    let m = compound_moments_from(frequency, exact_sev);
    let eps = settings.eps;
    audit(step, log2, settings, |h, points| {
        let (sev, _) = Grid::rounding(severity, h, points)?;
        let (agg, report) = fft(frequency, &sev, points)?;
        let mut terms = ScoreTerms::default();
        severity_terms(&mut terms, raw_moments(&sev), exact_sev, eps);
        aggregate_terms(&mut terms, &agg, m, eps);
        Ok(CellBuild {
            terms,
            tail_mass: report.tail_mass,
            aliasing_error: report.aliasing_error,
        })
    })
}

/// The exact compound mean, variance and third cumulant from the
/// severity's raw moments.
fn compound_moments_from<N: Counting + ?Sized>(
    n: &N,
    (m1, m2, m3): (f64, f64, f64),
) -> (f64, f64, f64) {
    let vx = m2 - m1 * m1;
    let k3x = m3 - 3.0 * m1 * vx - m1 * m1 * m1;
    let (n1, vn, k3n) = (n.mean(), n.variance(), n.third_cumulant());
    (
        n1 * m1,
        n1 * vx + vn * m1 * m1,
        n1 * k3x + 3.0 * vn * m1 * vx + k3n * m1 * m1 * m1,
    )
}

/// [`audit_grid`] for the sum of independent units, each
/// `(frequency, severity)`, on one grid. The severity terms score the
/// claim-weighted mixture of the units' severities (`aggregate`'s total
/// severity), the aggregate terms the total.
///
/// ```
/// use prospicio_aggregate::{AuditSettings, Sizing, audit_portfolio_grid, recommend_portfolio_grid};
/// use prospicio_prob::{Counting, Gamma, Poisson, Severity};
///
/// let (n1, x1) = (Poisson::new(10.0).unwrap(), Gamma::new(2.0, 50.0).unwrap());
/// let (n2, x2) = (Poisson::new(2.0).unwrap(), Gamma::new(0.5, 1000.0).unwrap());
/// let units: [(&dyn Counting, &dyn Severity); 2] = [(&n1, &x1), (&n2, &x2)];
/// let g = recommend_portfolio_grid(&units, &Sizing { log2: 12, ..Sizing::default() }).unwrap();
/// let audit = audit_portfolio_grid(&units, g.step, g.points.trailing_zeros(), &AuditSettings::default())
///     .unwrap();
/// assert!(audit.best().score < 0.5);
/// ```
pub fn audit_portfolio_grid(
    units: &[(&dyn Counting, &dyn Severity)],
    step: f64,
    log2: u32,
    settings: &AuditSettings,
) -> Result<GridAudit> {
    if units.is_empty() {
        return Err(Error::Data("a portfolio needs at least one unit".into()));
    }
    let eps = settings.eps;
    let exact: Vec<(f64, f64, f64)> = units.iter().map(|(_, x)| exact_raw(*x)).collect();
    let claims: f64 = units.iter().map(|(n, _)| n.mean()).sum();
    let mix = |raw: &[(f64, f64, f64)]| {
        let w = |i: usize| units[i].0.mean() / claims;
        raw.iter().enumerate().fold((0.0, 0.0, 0.0), |a, (i, r)| {
            (a.0 + w(i) * r.0, a.1 + w(i) * r.1, a.2 + w(i) * r.2)
        })
    };
    let exact_mix = mix(&exact);
    let total = units
        .iter()
        .zip(&exact)
        .map(|((n, _), &r)| compound_moments_from(*n, r))
        .fold((0.0, 0.0, 0.0), |a, m| (a.0 + m.0, a.1 + m.1, a.2 + m.2));
    audit(step, log2, settings, |h, points| {
        let base = (2 * points).next_power_of_two();
        let mut sev_raw = Vec::with_capacity(units.len());
        let (mut coarse, mut fine) = (vec![1.0; 0], vec![1.0; 0]);
        for (n, x) in units {
            let (sev, _) = Grid::rounding(*x, h, points)?;
            sev_raw.push(raw_moments(&sev));
            let c = compound_buffer(*n, sev.probs(), base);
            let f = compound_buffer(*n, sev.probs(), 2 * base);
            coarse = if coarse.is_empty() {
                c
            } else {
                circular(&coarse, &c)
            };
            fine = if fine.is_empty() {
                f
            } else {
                circular(&fine, &f)
            };
        }
        let aliasing_error = coarse[..points]
            .iter()
            .zip(&fine[..points])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let (probs, tail_mass) = crate::compound::lump_tail(fine[..points].to_vec());
        let agg = Grid::new(h, probs)?;
        let mut terms = ScoreTerms::default();
        severity_terms(&mut terms, mix(&sev_raw), exact_mix, eps);
        aggregate_terms(&mut terms, &agg, total, eps);
        Ok(CellBuild {
            terms,
            tail_mass,
            aliasing_error,
        })
    })
}

/// The circular convolution of two buffers of the same length, by FFT.
fn circular(a: &[f64], b: &[f64]) -> Vec<f64> {
    use rustfft::FftPlanner;
    use rustfft::num_complex::Complex64;
    let len = a.len();
    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(len);
    let mut x: Vec<Complex64> = a.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let mut y: Vec<Complex64> = b.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    forward.process(&mut x);
    forward.process(&mut y);
    for (u, v) in x.iter_mut().zip(&y) {
        *u *= v;
    }
    planner.plan_fft_inverse(len).process(&mut x);
    x.iter().map(|z| (z.re / len as f64).max(0.0)).collect()
}

/// The search of [`audit_grid`], on any build.
fn audit(
    step: f64,
    log2: u32,
    s: &AuditSettings,
    mut build: impl FnMut(f64, usize) -> Result<CellBuild>,
) -> Result<GridAudit> {
    if !(step.is_finite() && step > 0.0) {
        return Err(Error::InvalidParameter {
            name: "step",
            value: step,
            reason: "must be finite and positive",
        });
    }
    if !(1..=30).contains(&log2) || s.log2_cap > 30 {
        return Err(Error::InvalidParameter {
            name: "log2",
            value: f64::from(log2),
            reason: "log2 and log2_cap must be in 1..=30",
        });
    }
    let mut cells: Vec<GridCell> = Vec::new();
    let mut cell = |k: i32, j: i32, cells: &mut Vec<GridCell>| -> f64 {
        let h = step * 2f64.powi(k);
        let l = (log2 as i32 + j) as u32;
        let c = match build(h, 1usize << l) {
            Ok(b) => {
                let score = b.terms.score(s.power);
                GridCell {
                    d_step: k,
                    d_log2: j,
                    step: h,
                    log2: l,
                    score,
                    terms: b.terms,
                    tail_mass: b.tail_mass,
                    aliasing_error: b.aliasing_error,
                    defective: b.tail_mass > NOISE,
                    selected: false,
                    note: String::new(),
                }
            }
            Err(e) => GridCell {
                d_step: k,
                d_log2: j,
                step: h,
                log2: l,
                score: f64::NAN,
                terms: ScoreTerms::default(),
                tail_mass: f64::NAN,
                aliasing_error: f64::NAN,
                defective: false,
                selected: false,
                note: format!("failed: {e}"),
            },
        };
        let score = c.score;
        cells.push(c);
        score
    };
    let center = cell(0, 0, &mut cells);
    cells[0].note = "center".into();
    if center.is_finite() && center <= s.good_enough && !cells[0].defective {
        cells[0].selected = true;
        cells[0].note = "center; meets the target and is sound, so nothing was probed".into();
        return Ok(GridAudit {
            cells,
            selected: 0,
            probed: false,
            grew: false,
            reason: "the center meets the target".into(),
        });
    }
    let max_k = s.max_doublings as i32;
    let search_row = |cell: &mut dyn FnMut(i32, i32, &mut Vec<GridCell>) -> f64,
                      j: i32,
                      anchor: f64,
                      cells: &mut Vec<GridCell>| {
        for dir in [1i32, -1] {
            let (mut prev, mut k) = (anchor, dir);
            let mut stopped = false;
            while k.abs() <= max_k {
                let score = cell(k, j, cells);
                // A failed cell or anchor (a NaN score) stops the walk too.
                if score.partial_cmp(&prev) != Some(std::cmp::Ordering::Less) {
                    let last = cells.last_mut().expect("a cell was just pushed");
                    if last.note.is_empty() {
                        last.note = "stopped: no better than the previous step".into();
                    }
                    stopped = true;
                    break;
                }
                prev = score;
                k += dir;
            }
            if !stopped {
                cells.last_mut().expect("a cell was just pushed").note =
                    "stopped: step limit reached, still improving".into();
            }
        }
    };
    search_row(&mut cell, 0, center, &mut cells);
    if log2 > s.log2_floor && log2 > 1 {
        let anchor = cell(0, -1, &mut cells);
        search_row(&mut cell, -1, anchor, &mut cells);
    }
    let thrifty = pick(&cells, |_| true);
    let (selected, grew, reason) = match thrifty {
        None => (
            0,
            false,
            "every cell failed, so the center stays".to_string(),
        ),
        Some(i) if cells[i].score <= s.good_enough => (
            i,
            false,
            "the best sound score at the center's size or smaller meets the target".into(),
        ),
        Some(i) if log2 >= s.log2_cap => (
            i,
            false,
            "nothing at the center's size or smaller meets the target, and log2 is at its cap"
                .into(),
        ),
        Some(_) => {
            let anchor = cell(0, 1, &mut cells);
            search_row(&mut cell, 1, anchor, &mut cells);
            match pick(&cells, |c| c.d_log2 == 1 && c.score <= s.good_enough && !c.defective) {
                Some(i) => (
                    i,
                    true,
                    "nothing sound at the center's size or smaller meets the target, so log2 grew by one"
                        .into(),
                ),
                None => {
                    let i = pick(&cells, |_| true).unwrap_or(0);
                    (
                        i,
                        cells[i].d_log2 == 1,
                        "no cell meets the target, so the best was taken; audit again from it to go on"
                            .into(),
                    )
                }
            }
        }
    };
    cells[selected].selected = true;
    Ok(GridAudit {
        cells,
        selected,
        probed: true,
        grew,
        reason,
    })
}

/// The best cell among those `keep` accepts: sound before defective, then
/// within [`SLACK`] of the best score, fewest points, smallest move, best
/// score.
fn pick(cells: &[GridCell], keep: impl Fn(&GridCell) -> bool) -> Option<usize> {
    let finite: Vec<usize> = (0..cells.len())
        .filter(|&i| cells[i].score.is_finite() && keep(&cells[i]))
        .collect();
    let clean: Vec<usize> = finite
        .iter()
        .copied()
        .filter(|&i| !cells[i].defective)
        .collect();
    let pool = if clean.is_empty() { finite } else { clean };
    let best = pool
        .iter()
        .map(|&i| cells[i].score)
        .fold(f64::INFINITY, f64::min);
    pool.into_iter()
        .filter(|&i| cells[i].score <= best * SLACK)
        .min_by(|&a, &b| {
            let (x, y) = (&cells[a], &cells[b]);
            x.log2
                .cmp(&y.log2)
                .then(x.d_step.abs().cmp(&y.d_step.abs()))
                .then(x.score.total_cmp(&y.score))
        })
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

    #[test]
    fn a_portfolio_of_one_unit_audits_as_the_unit() {
        let (n, x) = (Poisson::new(20.0).unwrap(), Pareto::new(10.0, 4.5).unwrap());
        let s = AuditSettings::default();
        let one = audit_grid(&n, &x, 0.5, 9, &s).unwrap();
        let units: [(&dyn Counting, &dyn Severity); 1] = [(&n, &x)];
        let book = audit_portfolio_grid(&units, 0.5, 9, &s).unwrap();
        assert_eq!(one.cells.len(), book.cells.len());
        assert_eq!(one.selected, book.selected);
        for (a, b) in one.cells.iter().zip(&book.cells) {
            assert_eq!((a.step, a.log2), (b.step, b.log2));
            assert!(
                (a.score - b.score).abs() <= 1e-6 * a.score.abs().max(1.0),
                "{a:?} {b:?}"
            );
        }
    }

    #[test]
    fn a_sound_center_on_target_is_kept_without_probing() {
        let (n, x) = (Poisson::new(5.0).unwrap(), Pareto::new(10.0, 4.0).unwrap());
        let g = recommend_grid(&n, &x, &Sizing::default()).unwrap();
        let a = audit_grid(
            &n,
            &x,
            g.step,
            g.points.trailing_zeros(),
            &AuditSettings::default(),
        )
        .unwrap();
        if a.cells[0].score <= 0.5 && !a.cells[0].defective {
            assert!(!a.probed && a.cells.len() == 1 && a.selected == 0);
        } else {
            assert!(a.probed && a.cells.len() > 1);
        }
        assert!(audit_grid(&n, &x, -1.0, 10, &AuditSettings::default()).is_err());
        assert!(audit_grid(&n, &x, 1.0, 31, &AuditSettings::default()).is_err());
    }
}
