//! Renewal claim counts: the number of claims by a horizon `T` when the
//! waits between claims are independent draws from any law, `aggregate`'s
//! `years ... wait ...` frequency (Sparre Andersen).
//!
//! With `S_k` the time of the `k`-th claim, `{N(T) >= k} = {S_k <= T}`, so
//! `P(N = k) = F^{*k}(T) - F^{*(k+1)}(T)`. The wait is put on a grid of
//! step `h` and each `F^{*k}(T)` read off in the frequency domain: by
//! Plancherel it is the inner product of the transform of the wait's
//! `k`-th power with the transform of the indicator of `[0, T]`. Both are
//! tilted by `e^{-θx}` (and the indicator by `e^{θx}`), which damps the
//! circular wrap of mass past the grid by `e^{-θL}` with `θL = 20`, so the
//! grid need only reach a third past `T`. One forward transform, then a
//! vector product and a dot product for each `k`.
//!
//! Waits of zero (claims that arrive together) are taken out exactly: with
//! `p0 = P(W <= 0)` the count is `N = G_1 + … + G_{M+1} - 1`, where `M` is
//! the count for the positive waits `W | W > 0` and the `G_i` are
//! geometric on `1, 2, …` with `P(G = g) = (1 - p0) p0^(g-1)`. A defective
//! wait, whose probabilities sum to less than 1, ends the process: the
//! count stays a proper distribution.
//!
//! The grid follows `aggregate`'s `wait_grid`: for a continuous wait the
//! step resolves the wait's shape (`h <= μ/64`, and `σ/64` unless the wait
//! is nearly fixed) and keeps the count's error small (`k_max h² <=
//! 1e-9`), on `2^16` to `2^24` points with `T` at three quarters of the
//! grid; for waits on a lattice that `T` is a multiple of, the step is the
//! lattice's and the reading at `T` takes the whole last cell. With that
//! grid the counts match `aggregate` 1.0.1 (`validation/tests/aggregate.rs`).

use prospicio_core::{Error, Result};
use prospicio_prob::Distribution;
use prospicio_prob::count_families::EmpiricalCount;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

/// Contributions below this are dropped when the count's support is sized.
const TAIL_EPS: f64 = 1e-17;
/// `θL`: wrap-around is damped by `e^{-20}`.
const TILT: f64 = 20.0;
/// Standard deviations of headroom on the count, by the renewal central
/// limit theorem.
const Z: f64 = 10.0;
const LOG2_FLOOR: i32 = 16;
const LOG2_CAP: i32 = 24;
/// The count's error budget, `k_max h²`.
const TOL: f64 = 1e-9;
/// The step resolves the wait's shape: `h <= μ / KAPPA`.
const KAPPA: f64 = 64.0;

/// The law of the wait between claims.
#[derive(Clone, Copy)]
pub enum Wait<'a> {
    /// A distribution, put on the grid by rounding.
    Law(&'a dyn Distribution),
    /// Waits `values` with probabilities `probs`, `aggregate`'s `dwait`.
    /// The probabilities may sum to less than 1: the rest is the chance
    /// that the process ends (a defective wait). Waits at or below zero
    /// are claims that arrive together.
    Atoms {
        /// The waits.
        values: &'a [f64],
        /// Their probabilities, summing to at most 1.
        probs: &'a [f64],
    },
}

impl std::fmt::Debug for Wait<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Law(d) => f
                .debug_struct("Law")
                .field("mean", &d.mean())
                .field("variance", &d.variance())
                .finish(),
            Self::Atoms { values, probs } => f
                .debug_struct("Atoms")
                .field("values", values)
                .field("probs", probs)
                .finish(),
        }
    }
}

impl Wait<'_> {
    fn check(&self) -> Result<()> {
        if let Self::Atoms { values, probs } = self {
            if values.is_empty() || values.len() != probs.len() {
                return Err(Error::Data(
                    "a discrete wait needs as many probabilities as values, and at least one"
                        .into(),
                ));
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error::Data("the waits must be finite".into()));
            }
            if probs.iter().any(|p| !(p.is_finite() && *p >= 0.0)) {
                return Err(Error::Data(
                    "the wait probabilities must be finite and non-negative".into(),
                ));
            }
            let total: f64 = probs.iter().sum();
            if total > 1.0 + 1e-12 {
                return Err(Error::Data(format!(
                    "the wait probabilities sum to {total} > 1"
                )));
            }
        }
        Ok(())
    }

    /// `P(W <= 0)`.
    fn p0(&self) -> f64 {
        match self {
            Self::Law(d) => d.cdf(0.0),
            Self::Atoms { values, probs } => values
                .iter()
                .zip(*probs)
                .filter(|(v, _)| **v <= 0.0)
                .map(|(_, p)| p)
                .sum(),
        }
    }

    /// The chance the process ends: `1 -` the total probability.
    fn defect(&self) -> f64 {
        match self {
            Self::Law(_) => 0.0,
            Self::Atoms { probs, .. } => (1.0 - probs.iter().sum::<f64>()).max(0.0),
        }
    }

    /// `E[W]` and `E[W²]` over the whole law, the defect counting as 0.
    fn raw_moments(&self) -> (f64, f64) {
        match self {
            Self::Law(d) => {
                let m = d.mean();
                (m, d.variance() + m * m)
            }
            Self::Atoms { values, probs } => values
                .iter()
                .zip(*probs)
                .fold((0.0, 0.0), |(m1, m2), (v, p)| (m1 + p * v, m2 + p * v * v)),
        }
    }

    /// `P(W > x)` on the whole law (atoms not rescaled), with `P(W > -∞)`
    /// the total mass.
    fn survival(&self, x: f64) -> f64 {
        match self {
            Self::Law(_) if x == f64::NEG_INFINITY => 1.0,
            Self::Law(d) => d.survival(x),
            Self::Atoms { values, probs } => values
                .iter()
                .zip(*probs)
                .filter(|(v, _)| **v > x)
                .map(|(_, p)| p)
                .sum(),
        }
    }
}

/// The grid a wait is put on: `points` points of step `step` (in units of
/// time), and whether the reading at the horizon takes the whole last cell
/// (`closed`, for waits on the grid's lattice) or half of it (continuous
/// waits).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaitGrid {
    /// The step `h`.
    pub step: f64,
    /// The number of points, a power of 2 by default.
    pub points: usize,
    /// Whether a sum of waits landing on the horizon counts in full.
    pub closed: bool,
}

impl WaitGrid {
    /// The grid `aggregate`'s `wait_grid` picks for `wait` and horizon
    /// `horizon` (see the [module](self)).
    pub fn sized(wait: &Wait<'_>, horizon: f64) -> Result<Self> {
        wait.check()?;
        check_horizon(horizon)?;
        let p0 = wait.p0();
        let defect = wait.defect();
        let q_pos = (1.0 - p0 - defect).max(1e-15);
        let (m1, m2) = wait.raw_moments();
        let (mu, sigma) = (
            m1 / q_pos,
            (m2 / q_pos - (m1 / q_pos).powi(2)).max(0.0).sqrt(),
        );
        if mu.is_nan() || mu <= 0.0 {
            return Err(Error::Data(format!(
                "the positive waits' mean {mu} must be positive"
            )));
        }
        let var = sigma * sigma;
        let kmax_est = (horizon / mu + Z * (horizon * var / mu.powi(3)).sqrt() + Z).ceil();
        let implied = |h: f64| {
            let raw = (horizon * 4.0 / 3.0 / h).log2().ceil() as i32;
            let l2 = raw.clamp(LOG2_FLOOR, LOG2_CAP);
            let n1 = (0.75 * (1u64 << l2) as f64) as u64;
            (l2, horizon / n1 as f64)
        };
        let mut h_shape = mu / KAPPA;
        if sigma >= mu / 10.0 {
            h_shape = h_shape.min(sigma / KAPPA);
        }
        let h_acc = (TOL / kmax_est).sqrt();
        let (log2, step) = implied(h_shape.min(h_acc));
        if let Wait::Atoms { values, .. } = wait
            && let Some(lattice) = lattice_step(values, horizon)
        {
            let n1 = (horizon / lattice).round_ties_even();
            let l2 = ((n1 * 4.0 / 3.0).log2().ceil() as i32).max(3);
            if l2 <= LOG2_CAP {
                return Ok(Self {
                    step: horizon / n1,
                    points: 1 << l2,
                    closed: true,
                });
            }
        }
        Ok(Self {
            step,
            points: 1 << log2,
            closed: false,
        })
    }
}

fn check_horizon(horizon: f64) -> Result<()> {
    if horizon.is_finite() && horizon > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name: "horizon",
            value: horizon,
            reason: "must be finite and positive",
        })
    }
}

/// The exact common step of the positive `values` and `horizon`, or `None`
/// when they have none (or it is finer than `horizon / (3 · 2^22)`). Each
/// value is first replaced by its closest fraction with a denominator of
/// at most `10^9`, as `aggregate`'s `_lattice_step`.
fn lattice_step(values: &[f64], horizon: f64) -> Option<f64> {
    let all: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| *v > 0.0)
        .chain(std::iter::once(horizon))
        .collect();
    let mut g = (0i128, 1i128);
    for &v in &all {
        let f = limit_denominator(v, 1_000_000_000)?;
        // gcd(a/b, c/d) = gcd(a d, c b) / (b d)
        let num = gcd(g.0.checked_mul(f.1)?, f.0.checked_mul(g.1)?);
        let den = g.1.checked_mul(f.1)?;
        let k = gcd(num, den);
        g = (num / k, den / k);
    }
    if g.0 <= 0 {
        return None;
    }
    let step = g.0 as f64 / g.1 as f64;
    for &v in &all {
        let k = (v / step).round_ties_even();
        if k < 0.0 || (v - k * step).abs() > 1e-9 * v.abs().max(1.0) {
            return None;
        }
    }
    let n1 = (horizon / step).round_ties_even();
    (n1 >= 1.0 && n1 <= (3u64 << 22) as f64).then_some(step)
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The closest fraction to `x > 0` with a denominator of at most `max`,
/// Python's `Fraction(x).limit_denominator(max)`: the exact binary value
/// of `x`, then its continued fraction. `None` when `x` is too large or
/// too small for the arithmetic.
fn limit_denominator(x: f64, max: i128) -> Option<(i128, i128)> {
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = (bits & ((1 << 52) - 1)) as i128;
    let (mant, e) = if exp == 0 {
        (frac, -1074)
    } else {
        (frac | (1 << 52), exp - 1075)
    };
    let (mut n, mut d) = if e >= 0 {
        if e > 60 {
            return None;
        }
        (mant << e, 1i128)
    } else {
        if -e > 80 {
            return None;
        }
        (mant, 1i128 << -e)
    };
    let k = gcd(n, d);
    (n, d) = (n / k, d / k);
    if d <= max {
        return Some((n, d));
    }
    let denominator = d;
    let (mut p0, mut q0, mut p1, mut q1) = (0i128, 1i128, 1i128, 0i128);
    loop {
        let a = n / d;
        let q2 = q0 + a * q1;
        if q2 > max {
            break;
        }
        (p0, q0, p1, q1) = (p1, q1, p0 + a * p1, q2);
        (n, d) = (d, n - a * d);
    }
    let k = (max - q0) / q1;
    if 2 * d * (q0 + k * q1) <= denominator {
        Some((p1, q1))
    } else {
        Some((p0 + k * p1, q0 + k * q1))
    }
}

/// The count's length: `k_max = ⌈T/μ + z √(T σ²/μ³) + z⌉` from the
/// renewal central limit theorem with the grid's own moments, and for a
/// wait that sums to `q < 1`, at most `⌈ln 10⁻¹⁷ / ln q⌉`.
fn kmax_from_pmf(pm: &[f64], step: f64, horizon: f64) -> Result<usize> {
    let q: f64 = pm.iter().sum();
    if q.is_nan() || q <= 0.0 {
        return Err(Error::Data(
            "the wait has no positive mass on the grid".into(),
        ));
    }
    let (s1, s2) = pm.iter().enumerate().fold((0.0, 0.0), |(s1, s2), (j, p)| {
        let x = j as f64 * step;
        (s1 + x * p, s2 + x * x * p)
    });
    let mu = s1 / q;
    if mu.is_nan() || mu <= 0.0 {
        return Err(Error::Data(
            "all the wait's mass is at zero, so the count is unbounded".into(),
        ));
    }
    let var = (s2 / q - mu * mu).max(0.0);
    let mut kmax = (horizon / mu + Z * (horizon * var / mu.powi(3)).sqrt() + Z).ceil() as usize;
    if q < 1.0 {
        kmax = kmax.min((TAIL_EPS.ln() / q.ln()).ceil() as usize);
    }
    Ok(kmax.max(1))
}

/// `P(N(T) = k)` for `k = 0, …, k_max`, for a wait already on a grid:
/// `pm[j]` is the chance of a wait of `j · step`, summing to at most 1
/// (mass missing, or cut at the horizon, ends the process; for the count
/// by `T` the two are the same). Mass at 0 counts as waits of zero; take
/// it out first with [`zero_wait_batches`] for the exact clusters. The
/// grid, `pm.len() · step`, must reach past `horizon`; `closed` reads the
/// whole last cell at `T` (waits on the lattice), otherwise half of it,
/// which centres the reading on `T` for continuous waits. `kmax` is the
/// [module](self)'s by default.
///
/// When `pm[0]` is 0 the count is cut at `⌊T / w⌋`, with `w` the smallest
/// wait: the round-off of the tilted sums would otherwise put specks of
/// probability on counts that cannot happen.
pub fn renewal_count_pmf(
    pm: &[f64],
    step: f64,
    horizon: f64,
    closed: bool,
    kmax: Option<usize>,
) -> Result<Vec<f64>> {
    check_horizon(horizon)?;
    if !(step.is_finite() && step > 0.0) {
        return Err(Error::InvalidParameter {
            name: "step",
            value: step,
            reason: "must be finite and positive",
        });
    }
    if pm.iter().any(|p| !(p.is_finite() && *p >= 0.0)) {
        return Err(Error::Data(
            "the wait probabilities must be finite and non-negative".into(),
        ));
    }
    let m = pm.len();
    let length = m as f64 * step;
    if length <= horizon {
        return Err(Error::Data(format!(
            "the grid, {length}, must reach past the horizon {horizon}"
        )));
    }
    let n1 = (horizon / step).round_ties_even() as usize;
    if n1 < 1 {
        return Err(Error::Data(format!(
            "the horizon {horizon} must be at least one step {step}"
        )));
    }
    let kmax = match kmax {
        Some(k) => k.max(1),
        None => kmax_from_pmf(pm, step, horizon)?,
    };
    let theta = TILT / length;
    let tilted: Vec<f64> = pm
        .iter()
        .enumerate()
        .map(|(j, p)| p * (-theta * j as f64 * step).exp())
        .collect();
    let mut indicator = vec![0.0; m];
    for (j, v) in indicator.iter_mut().enumerate().take(n1 + 1) {
        *v = (theta * j as f64 * step).exp();
    }
    if !closed {
        // The partial sum through cell j reads F at j h + h/2: weighting
        // the last cell by 1/2 + (T - n1 h)/h centres the reading on T.
        indicator[n1] *= 0.5 + (horizon - n1 as f64 * step) / step;
    }
    // Two transforms, not one packed: the tilted indicator is up to e^15
    // times the wait, and packing them would share its round-off.
    let fft = FftPlanner::<f64>::new().plan_fft_forward(m);
    let half = m / 2 + 1;
    let transform = |x: &[f64]| {
        let mut z: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        fft.process(&mut z);
        z.truncate(half);
        z
    };
    let phat = transform(&tilted);
    // Plancherel over the half spectrum of real sequences:
    // sum a_j b_j = Re(sum_w c_w A_w conj B_w) / m, c = 1, 2, …, 2, 1.
    let weight: Vec<Complex64> = transform(&indicator)
        .into_iter()
        .enumerate()
        .map(|(w, b)| {
            let c = if w == 0 || (m % 2 == 0 && w == m / 2) {
                1.0
            } else {
                2.0
            };
            b.conj() * (c / m as f64)
        })
        .collect();
    let mut f = Vec::with_capacity(kmax + 2);
    f.push(1.0);
    let mut v = vec![Complex64::new(1.0, 0.0); half];
    for _ in 1..kmax + 2 {
        let mut dot = 0.0;
        for ((vw, pw), cw) in v.iter_mut().zip(&phat).zip(&weight) {
            *vw *= pw;
            dot += (*vw * cw).re;
        }
        f.push(dot);
    }
    let mut pn: Vec<f64> = f.windows(2).map(|w| (w[0] - w[1]).max(0.0)).collect();
    if let Some(first) = pm.iter().position(|p| *p != 0.0)
        && first >= 1
    {
        let bound = n1 / first + 1;
        if bound < pn.len() {
            pn[bound..].iter_mut().for_each(|p| *p = 0.0);
        }
    }
    Ok(pn)
}

/// The count `N` from the count `M` of the positive waits, when a wait is
/// zero with probability `p0` in `[0, 1)`: `N = G_1 + … + G_{M+1} - 1`
/// with `G` geometric on `1, 2, …`. Given `M = m`, `N - m` is negative
/// binomial with `m + 1` successes of probability `1 - p0`. The result
/// runs until the geometric tail is below `10⁻¹⁷`. `aggregate`'s
/// `geometric_batch_compose`.
///
/// ```
/// use prospicio_aggregate::renewal::zero_wait_batches;
///
/// // Waits of 0 or 1 with even odds over one period: the one positive wait
/// // comes at 1, so M = 1, and P(N = n) = n / 2^(n+1).
/// let m = [0.0, 1.0];
/// let n = zero_wait_batches(&m, 0.5).unwrap();
/// assert!((n[3] - 3.0 / 16.0).abs() < 1e-15);
/// ```
pub fn zero_wait_batches(pmf_m: &[f64], p0: f64) -> Result<Vec<f64>> {
    if !(0.0..1.0).contains(&p0) {
        return Err(Error::InvalidParameter {
            name: "p0",
            value: p0,
            reason: "must be in [0, 1)",
        });
    }
    if p0 == 0.0 {
        return Ok(pmf_m.to_vec());
    }
    let extra = (TAIL_EPS.ln() / p0.ln()).ceil() as usize;
    let nmax = pmf_m.len() - 1 + extra;
    let mut pn = vec![0.0; nmax + 1];
    for (m, &w) in pmf_m.iter().enumerate() {
        if w == 0.0 {
            continue;
        }
        // nbinom(j; m + 1, 1 - p0) = C(j + m, m) (1 - p0)^(m+1) p0^j
        let mut term = (1.0 - p0).powi(m as i32 + 1);
        for (j, slot) in pn[m..].iter_mut().enumerate() {
            if j > 0 {
                term *= (j + m) as f64 / j as f64 * p0;
            }
            *slot += w * term;
        }
    }
    Ok(pn)
}

/// A renewal count, with the grid it was read on.
#[derive(Debug, Clone, PartialEq)]
pub struct RenewalCount {
    /// The count `N(T)`.
    pub count: EmpiricalCount,
    /// The wait's grid.
    pub grid: WaitGrid,
    /// `P(W <= 0)`, the chance a claim arrives with the one before.
    pub p0: f64,
    /// The chance the process ends at a wait: 1 less the wait's total
    /// probability.
    pub defect: f64,
    /// The length of the count of positive waits before the zero waits
    /// were put back.
    pub kmax: usize,
}

/// The number of claims by `horizon` when the waits between them are
/// independent draws from `wait`, `aggregate`'s `wait_count_pmf`; on
/// `grid`, or by default the grid [`WaitGrid::sized`] picks. See the
/// [module](self).
///
/// The wait is put on the grid by rounding (each point takes the mass
/// within half a step of it, from survival differences); the mass at or
/// below 0 is taken out as clusters, the mass past the horizon dropped
/// (it cannot add a claim), and the rest rescaled to the positive waits.
///
/// ```
/// use prospicio_aggregate::renewal::{Wait, renewal_count};
/// use prospicio_prob::{Counting, Gamma};
///
/// // Exponential waits with mean 1 over 10 years: Poisson with mean 10.
/// let wait = Gamma::new(1.0, 1.0).unwrap();
/// let n = renewal_count(&Wait::Law(&wait), 10.0, None).unwrap();
/// assert!((n.count.mean() - 10.0).abs() < 1e-6);
/// assert!((n.count.variance() - 10.0).abs() < 1e-5);
/// ```
pub fn renewal_count(
    wait: &Wait<'_>,
    horizon: f64,
    grid: Option<WaitGrid>,
) -> Result<RenewalCount> {
    wait.check()?;
    check_horizon(horizon)?;
    let p0 = wait.p0();
    let defect = wait.defect();
    if p0 >= 1.0 - 1e-15 || p0 + defect >= 1.0 - 1e-15 {
        return Err(Error::Data(format!(
            "all the wait's mass is at or below zero (P(W <= 0) = {p0}, defect = {defect}), \
             so the count is unbounded"
        )));
    }
    let grid = match grid {
        Some(g) => g,
        None => WaitGrid::sized(wait, horizon)?,
    };
    let (h, m) = (grid.step, grid.points);
    if !(h.is_finite() && h > 0.0) || m < 2 {
        return Err(Error::Data(
            "the wait grid needs a positive step and at least 2 points".into(),
        ));
    }
    let n1 = (horizon / h).round_ties_even() as usize;
    let mut pm = vec![0.0; m];
    let mut upper = wait.survival(f64::NEG_INFINITY);
    for (j, slot) in pm.iter_mut().enumerate().take((n1 + 1).min(m)) {
        let next = wait.survival((j as f64 + 0.5) * h);
        *slot = upper - next;
        upper = next;
    }
    // Cell 0 holds P(W <= h/2): keep only the rounding mass in (0, h/2).
    pm[0] = (pm[0] - p0).max(0.0);
    if pm.iter().sum::<f64>() <= 0.0 {
        return Err(Error::Data(
            "the wait has no positive mass on the grid".into(),
        ));
    }
    pm.iter_mut().for_each(|p| *p /= 1.0 - p0);
    let kmax = kmax_from_pmf(&pm, h, horizon)?;
    let pmf_m = renewal_count_pmf(&pm, h, horizon, grid.closed, Some(kmax))?;
    let pn = zero_wait_batches(&pmf_m, p0)?;
    Ok(RenewalCount {
        count: EmpiricalCount::new(pn)?,
        grid,
        p0,
        defect,
        kmax,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospicio_prob::{Counting, Gamma};

    #[test]
    fn fixed_waits_give_a_fixed_count() {
        let w = Wait::Atoms {
            values: &[1.0],
            probs: &[1.0],
        };
        let n = renewal_count(&w, 3.0, None).unwrap();
        assert!(n.grid.closed);
        assert_eq!(n.grid.step, 1.0);
        assert!((n.count.pmf(3) - 1.0).abs() < 1e-12);
        assert!(n.count.variance() < 1e-10);
    }

    #[test]
    fn clusters_of_zero_waits_are_geometric() {
        // P(N = n) = n / 2^(n+1) over one period of waits 0 or 1.
        let w = Wait::Atoms {
            values: &[0.0, 1.0],
            probs: &[0.5, 0.5],
        };
        let n = renewal_count(&w, 1.0, None).unwrap();
        for k in 0..20u64 {
            let exact = k as f64 / 2f64.powi(k as i32 + 1);
            assert!((n.count.pmf(k) - exact).abs() < 1e-12, "{k}");
        }
    }

    #[test]
    fn a_defective_wait_ends_the_process() {
        // A wait of 1 with probability 0.9: N(2) is 0, 1 or 2 with
        // P(N >= k) = 0.9^k.
        let w = Wait::Atoms {
            values: &[1.0],
            probs: &[0.9],
        };
        let n = renewal_count(&w, 2.0, None).unwrap();
        assert!((n.defect - 0.1).abs() < 1e-15);
        assert!((n.count.pmf(0) - 0.1).abs() < 1e-12);
        assert!((n.count.pmf(1) - 0.09).abs() < 1e-12);
        assert!((n.count.pmf(2) - 0.81).abs() < 1e-12);
    }

    #[test]
    fn exponential_waits_are_poisson() {
        let wait = Gamma::new(1.0, 0.5).unwrap();
        let n = renewal_count(&Wait::Law(&wait), 2.0, None).unwrap();
        let mut p = (-4.0f64).exp();
        for k in 0..30u64 {
            assert!((n.count.pmf(k) - p).abs() < 1e-8, "{k}");
            p *= 4.0 / (k + 1) as f64;
        }
    }

    #[test]
    fn fractions_and_lattices() {
        assert_eq!(limit_denominator(0.5, 1_000_000_000), Some((1, 2)));
        assert_eq!(limit_denominator(1.1, 1_000_000_000), Some((11, 10)));
        assert_eq!(limit_denominator(3.0, 1_000_000_000), Some((3, 1)));
        assert_eq!(lattice_step(&[0.5, 1.5], 3.0), Some(0.5));
        assert_eq!(lattice_step(&[0.4, 1.1], 2.0), Some(0.1));
        assert_eq!(lattice_step(&[std::f64::consts::PI], 2.0), None);
    }

    #[test]
    fn bad_inputs_are_refused() {
        let w = Wait::Atoms {
            values: &[0.0],
            probs: &[1.0],
        };
        assert!(renewal_count(&w, 1.0, None).is_err());
        let w = Wait::Atoms {
            values: &[1.0],
            probs: &[0.6, 0.6],
        };
        assert!(renewal_count(&w, 1.0, None).is_err());
        assert!(zero_wait_batches(&[1.0], 1.0).is_err());
        assert!(renewal_count_pmf(&[0.0, 1.0], 1.0, 2.5, true, None).is_err());
    }
}
