//! The discretized representation: probabilities on an evenly spaced grid.

use act_core::{Error, Result};

use crate::distribution::{Distribution, check_probability};
use crate::severity::Severity;

/// A distribution on the points `0, h, 2h, …, (n - 1)h`.
///
/// Grids are what FFT and Panjer aggregation work on. They are exact for
/// sums, layers and stop-loss on the grid, but a grid made from a continuous
/// distribution is an approximation: see [`Grid::local_moment`],
/// [`Grid::rounding`] and [`Grid::lower`], which return a
/// [`DiscretizationReport`] with the error introduced.
///
/// The probabilities always sum to 1. Discretization lumps whatever lies
/// beyond the last point onto it and reports that mass.
///
/// # Example
///
/// ```
/// use act_prob::{Distribution, Grid, Lognormal, Severity};
///
/// let sev = Lognormal::new(7.0, 0.5).unwrap();
/// let (grid, report) = Grid::local_moment(&sev, 100.0, 200).unwrap();
/// // Local moment matching preserves the limited mean up to the last point.
/// assert!((grid.mean() - sev.lev(199.0 * 100.0)).abs() < 1e-9);
/// // About 3e-9 of the mass lies beyond the last point (19,900).
/// assert!(report.tail_mass < 1e-8);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    step: f64,
    probs: Vec<f64>,
}

/// How a continuous distribution was turned into a [`Grid`], and the error
/// that introduced.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscretizationReport {
    pub method: Discretization,
    pub step: f64,
    pub points: usize,
    /// Probability the source puts above the last grid point, `S((n - 1)h)`.
    /// Discretization lumps it onto the last point.
    pub tail_mass: f64,
    /// Mean of the source distribution.
    pub source_mean: f64,
    /// Mean of the grid.
    pub grid_mean: f64,
}

impl DiscretizationReport {
    /// `grid_mean - source_mean`.
    pub fn mean_error(&self) -> f64 {
        self.grid_mean - self.source_mean
    }
}

/// Discretization methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Discretization {
    /// Local moment matching on the mean (mass dispersal with matched
    /// first moment, Klugman, Panjer & Willmot): every cell's mass is split
    /// between its two end points so its mean is kept.
    LocalMoment,
    /// Each point takes the mass within half a step of it.
    Rounding,
    /// Each cell's mass moves to its left end, so the grid is a stochastic
    /// lower bound of the source.
    Lower,
}

impl Grid {
    /// A grid with step `step` and probabilities `probs` at `0, step, …`.
    ///
    /// Fails if `step` is not finite and positive, `probs` is empty or holds a
    /// negative or non-finite value, or the probabilities do not sum to 1
    /// within `1e-9`.
    pub fn new(step: f64, probs: Vec<f64>) -> Result<Self> {
        if !step.is_finite() || step <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "step",
                value: step,
                reason: "must be finite and positive",
            });
        }
        if probs.is_empty() {
            return Err(Error::InvalidParameter {
                name: "probs",
                value: 0.0,
                reason: "must not be empty",
            });
        }
        if let Some(&bad) = probs.iter().find(|p| !p.is_finite() || **p < 0.0) {
            return Err(Error::InvalidParameter {
                name: "probs",
                value: bad,
                reason: "must all be finite and non-negative",
            });
        }
        let total: f64 = probs.iter().sum();
        if (total - 1.0).abs() > 1e-9 {
            return Err(Error::InvalidParameter {
                name: "probs",
                value: total,
                reason: "must sum to 1",
            });
        }
        Ok(Self { step, probs })
    }

    /// Discretizes `source` by local moment matching on the mean, using its
    /// limited expected values:
    ///
    /// ```text
    /// f_0 = 1 - LEV(h) / h
    /// f_j = (2 LEV(jh) - LEV((j-1)h) - LEV((j+1)h)) / h,  0 < j < n - 1
    /// f_{n-1} = (LEV((n-1)h) - LEV((n-2)h)) / h            (the rest)
    /// ```
    ///
    /// The grid's mean is exactly `LEV((n - 1)h)`.
    pub fn local_moment<D: Severity>(
        source: &D,
        step: f64,
        points: usize,
    ) -> Result<(Self, DiscretizationReport)> {
        check_shape(step, points)?;
        let lev: Vec<f64> = (0..points).map(|j| source.lev(j as f64 * step)).collect();
        let probs = if points == 1 {
            vec![1.0]
        } else {
            let mut probs = Vec::with_capacity(points);
            probs.push(1.0 - lev[1] / step);
            for j in 1..points - 1 {
                // LEV is concave, so this is non-negative up to rounding.
                probs.push(((2.0 * lev[j] - lev[j - 1] - lev[j + 1]) / step).max(0.0));
            }
            probs.push((lev[points - 1] - lev[points - 2]) / step);
            probs
        };
        Self::finish(source, step, probs, Discretization::LocalMoment)
    }

    /// Discretizes `source` by rounding each loss to the nearest point:
    /// `f_0 = F(h/2)`, `f_j = F((j + 1/2)h) - F((j - 1/2)h)`, and the last
    /// point takes everything above `(n - 3/2)h`.
    pub fn rounding<D: Distribution>(
        source: &D,
        step: f64,
        points: usize,
    ) -> Result<(Self, DiscretizationReport)> {
        check_shape(step, points)?;
        let edges: Vec<f64> = (0..points.saturating_sub(1))
            .map(|j| source.cdf((j as f64 + 0.5) * step))
            .collect();
        Self::finish(source, step, cells(&edges), Discretization::Rounding)
    }

    /// Discretizes `source` by moving each cell's mass to its left end:
    /// `f_j = F((j + 1)h) - F(jh)`, with the last point taking everything
    /// above `(n - 1)h`. The result is a stochastic lower bound of `source`.
    pub fn lower<D: Distribution>(
        source: &D,
        step: f64,
        points: usize,
    ) -> Result<(Self, DiscretizationReport)> {
        check_shape(step, points)?;
        let edges: Vec<f64> = (1..points).map(|j| source.cdf(j as f64 * step)).collect();
        Self::finish(source, step, cells(&edges), Discretization::Lower)
    }

    fn finish<D: Distribution>(
        source: &D,
        step: f64,
        probs: Vec<f64>,
        method: Discretization,
    ) -> Result<(Self, DiscretizationReport)> {
        let points = probs.len();
        let grid = Self::new(step, probs)?;
        let report = DiscretizationReport {
            method,
            step,
            points,
            tail_mass: 1.0 - source.cdf((points - 1) as f64 * step),
            source_mean: source.mean(),
            grid_mean: grid.mean(),
        };
        Ok((grid, report))
    }

    /// Grid step `h`.
    pub fn step(&self) -> f64 {
        self.step
    }

    /// Probabilities at `0, h, 2h, …`.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// Number of grid points.
    pub fn len(&self) -> usize {
        self.probs.len()
    }

    /// Always `false`: a grid has at least one point.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Loss at point `j`, `j * h`.
    pub fn x(&self, j: usize) -> f64 {
        j as f64 * self.step
    }

    /// Distortion risk measure of the grid, exact for the grid; see
    /// [`Distortion::apply_discrete`](crate::Distortion::apply_discrete).
    ///
    /// ```
    /// use act_prob::{Distortion, Grid};
    ///
    /// let g = Grid::new(1.0, vec![0.5, 0.25, 0.25]).unwrap();
    /// // TVaR at 50%: the top half of the mass, at 1 and 2.
    /// assert_eq!(g.distortion(&Distortion::tvar(0.5).unwrap()), 1.5);
    /// ```
    pub fn distortion(&self, d: &crate::Distortion) -> f64 {
        let values: Vec<f64> = (0..self.len()).map(|j| self.x(j)).collect();
        d.apply_discrete(&values, &self.probs)
    }

    /// The distribution of `f(X)` on the same step, and whether it is
    /// exact.
    ///
    /// Each point's mass moves to `f(x_j)`. A value on a grid point (within
    /// `1e-9` of a step) keeps its mass there; one between points `k` and
    /// `k + 1` is split between them so its mean is kept, as local moment
    /// matching does. The mean is therefore always exact, and the whole
    /// distribution is exact when the flag is `true`. The result is as long
    /// as the largest value needs.
    ///
    /// For a layer with boundaries on grid points the map is exact:
    ///
    /// ```
    /// use act_prob::{Distribution, Grid};
    ///
    /// let x = Grid::new(1.0, vec![0.2, 0.3, 0.3, 0.2]).unwrap();
    /// // 1 xs 1: values 0, 0, 1, 1.
    /// let (layer, exact) = x.map(|v| (v - 1.0).clamp(0.0, 1.0)).unwrap();
    /// assert!(exact);
    /// assert_eq!(layer.probs(), [0.5, 0.5]);
    /// // 1.5 xs 0.5 puts 0.5 and 1.5 between points: the mean is kept.
    /// let (layer, exact) = x.map(|v| (v - 0.5).clamp(0.0, 1.5)).unwrap();
    /// assert!(!exact);
    /// assert!((layer.mean() - (0.3 * 0.5 + 0.5 * 1.5)).abs() < 1e-15);
    /// ```
    ///
    /// Fails if `f` returns a negative or non-finite value at a point with
    /// mass.
    pub fn map(&self, mut f: impl FnMut(f64) -> f64) -> Result<(Self, bool)> {
        let mut probs = vec![0.0; 1];
        let mut exact = true;
        let add = |probs: &mut Vec<f64>, k: usize, p: f64| {
            if probs.len() <= k {
                probs.resize(k + 1, 0.0);
            }
            probs[k] += p;
        };
        for (x, p) in self.points() {
            if p == 0.0 {
                continue;
            }
            let y = f(x);
            if !y.is_finite() || y < 0.0 {
                return Err(Error::InvalidParameter {
                    name: "f",
                    value: y,
                    reason: "must map every point with mass to a finite, non-negative value",
                });
            }
            let at = y / self.step;
            let nearest = at.round();
            if (at - nearest).abs() <= 1e-9 * nearest.max(1.0) {
                add(&mut probs, nearest as usize, p);
            } else {
                exact = false;
                let k = at.floor();
                let upper = at - k;
                add(&mut probs, k as usize, p * (1.0 - upper));
                add(&mut probs, k as usize + 1, p * upper);
            }
        }
        // The masses only moved, so they still sum to 1 up to rounding.
        Ok((Self::new(self.step, probs)?, exact))
    }

    fn points(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.probs.iter().enumerate().map(|(j, &p)| (self.x(j), p))
    }
}

/// Cell masses from interior cdf edges `F(e_0) ≤ … ≤ F(e_{n-2})`: the first
/// cell is `F(e_0)`, the last is `1 - F(e_{n-2})`.
fn cells(edges: &[f64]) -> Vec<f64> {
    let mut probs = Vec::with_capacity(edges.len() + 1);
    let mut below = 0.0;
    for &f in edges {
        probs.push((f - below).max(0.0));
        below = f;
    }
    probs.push(1.0 - below);
    probs
}

fn check_shape(step: f64, points: usize) -> Result<()> {
    if !step.is_finite() || step <= 0.0 {
        return Err(Error::InvalidParameter {
            name: "step",
            value: step,
            reason: "must be finite and positive",
        });
    }
    if points == 0 {
        return Err(Error::InvalidParameter {
            name: "points",
            value: 0.0,
            reason: "must be positive",
        });
    }
    Ok(())
}

impl Distribution for Grid {
    fn mean(&self) -> f64 {
        self.points().map(|(x, p)| x * p).sum()
    }

    fn variance(&self) -> f64 {
        let mean = self.mean();
        self.points()
            .map(|(x, p)| (x - mean) * (x - mean) * p)
            .sum()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let last = ((x / self.step).floor() as usize).min(self.len() - 1);
        self.probs[..=last].iter().sum::<f64>().min(1.0)
    }

    /// Summed from the top, so small tail probabilities keep their
    /// precision.
    fn survival(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 1.0;
        }
        let last = ((x / self.step).floor() as usize).min(self.len() - 1);
        self.probs[last + 1..].iter().sum::<f64>().min(1.0)
    }

    fn quantile(&self, p: f64) -> Result<f64> {
        check_probability(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        let mut total = 0.0;
        for (j, &q) in self.probs.iter().enumerate() {
            total += q;
            if q > 0.0 && total >= p {
                return Ok(self.x(j));
            }
        }
        // Rounding left the total a hair below p: the last point with mass.
        let last = self.probs.iter().rposition(|&q| q > 0.0).unwrap_or(0);
        Ok(self.x(last))
    }
}

impl Severity for Grid {
    fn lev(&self, limit: f64) -> f64 {
        if limit <= 0.0 {
            return limit;
        }
        self.points().map(|(x, p)| x.min(limit) * p).sum()
    }

    fn stop_loss(&self, retention: f64) -> f64 {
        if retention <= 0.0 {
            return self.mean() - retention;
        }
        self.points()
            .map(|(x, p)| (x - retention).max(0.0) * p)
            .sum()
    }

    fn layer_second_moment(&self, limit: f64, attachment: f64) -> f64 {
        self.points()
            .map(|(x, p)| {
                let y = (x - attachment).max(0.0).min(limit);
                y * y * p
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lognormal;

    #[test]
    fn map_keeps_mass_and_mean_off_the_points() {
        let x = Grid::new(2.0, vec![0.1, 0.2, 0.3, 0.25, 0.15]).unwrap();
        let f = |v: f64| 0.7 * v + 0.3;
        let (y, exact) = x.map(f).unwrap();
        assert!(!exact);
        assert!((y.probs().iter().sum::<f64>() - 1.0).abs() < 1e-15);
        let want: f64 = (0..x.len()).map(|j| f(x.x(j)) * x.probs()[j]).sum();
        assert!((y.mean() - want).abs() < 1e-14);
        assert_eq!(y.step(), 2.0);
    }

    #[test]
    fn map_can_grow_the_grid_and_rejects_negative_values() {
        let x = Grid::new(1.0, vec![0.5, 0.5]).unwrap();
        let (y, exact) = x.map(|v| 3.0 * v).unwrap();
        assert!(exact);
        assert_eq!(y.probs(), [0.5, 0.0, 0.0, 0.5]);
        assert!(x.map(|v| v - 0.5).is_err());
    }

    fn sev() -> Lognormal {
        Lognormal::new(7.0, 0.5).unwrap()
    }

    #[test]
    fn local_moment_preserves_the_limited_mean() {
        let s = sev();
        for (step, n) in [(100.0, 200), (250.0, 20), (50.0, 30)] {
            let (g, r) = Grid::local_moment(&s, step, n).unwrap();
            let lev = s.lev((n - 1) as f64 * step);
            assert!((g.mean() - lev).abs() < 1e-9 * lev, "h {step}, n {n}");
            assert!(g.probs().iter().all(|&p| p >= 0.0));
            assert_eq!(r.grid_mean, g.mean());
            assert!((r.mean_error() + s.stop_loss((n - 1) as f64 * step)).abs() < 1e-9);
        }
    }

    #[test]
    fn probabilities_sum_to_one_and_tail_is_reported() {
        let s = sev();
        for method in [Grid::rounding::<Lognormal>, Grid::lower::<Lognormal>] {
            let (g, r) = method(&s, 100.0, 20).unwrap();
            assert!((g.probs().iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert_eq!(r.points, 20);
            assert!((r.tail_mass - (1.0 - s.cdf(1900.0))).abs() < 1e-15);
        }
        let (_, r) = Grid::local_moment(&s, 100.0, 20).unwrap();
        assert!(r.tail_mass > 0.01, "a short grid truncates");
    }

    #[test]
    fn rounding_and_lower_cells() {
        let s = sev();
        let (g, _) = Grid::rounding(&s, 100.0, 50).unwrap();
        assert!((g.probs()[0] - s.cdf(50.0)).abs() < 1e-15);
        assert!((g.probs()[10] - (s.cdf(1050.0) - s.cdf(950.0))).abs() < 1e-15);
        let (g, _) = Grid::lower(&s, 100.0, 50).unwrap();
        assert!((g.probs()[0] - s.cdf(100.0)).abs() < 1e-15);
        assert!((g.probs()[10] - (s.cdf(1100.0) - s.cdf(1000.0))).abs() < 1e-15);
    }

    #[test]
    fn lower_is_a_stochastic_lower_bound() {
        let s = sev();
        let (g, _) = Grid::lower(&s, 100.0, 60).unwrap();
        for x in (0..60).map(|j| j as f64 * 100.0 + 50.0) {
            assert!(g.cdf(x) >= s.cdf(x) - 1e-15, "at {x}");
        }
        assert!(g.mean() <= s.mean());
    }

    #[test]
    fn methods_converge_as_the_step_shrinks() {
        let s = sev();
        let mut prev = f64::INFINITY;
        for step in [400.0, 100.0, 25.0] {
            let n = (40_000.0 / step) as usize;
            let (g, _) = Grid::rounding(&s, step, n).unwrap();
            let err = (g.mean() - s.mean()).abs();
            assert!(err < prev);
            prev = err;
        }
        let (g, _) = Grid::local_moment(&s, 25.0, 1600).unwrap();
        assert!((g.quantile(0.99).unwrap() - s.quantile(0.99).unwrap()).abs() <= 25.0);
    }

    #[test]
    fn grid_distribution_methods() {
        let g = Grid::new(10.0, vec![0.5, 0.25, 0.25]).unwrap();
        assert_eq!(g.mean(), 7.5);
        assert_eq!(g.variance(), 0.5 * 56.25 + 0.25 * 6.25 + 0.25 * 156.25);
        assert_eq!(g.cdf(-1.0), 0.0);
        assert_eq!(g.cdf(0.0), 0.5);
        assert_eq!(g.cdf(15.0), 0.75);
        assert_eq!(g.cdf(1e9), 1.0);
        assert_eq!(g.quantile(0.0), Ok(0.0));
        assert_eq!(g.quantile(0.5), Ok(0.0));
        assert_eq!(g.quantile(0.6), Ok(10.0));
        assert_eq!(g.quantile(1.0), Ok(20.0));
        assert_eq!(g.lev(15.0), 0.25 * 10.0 + 0.25 * 15.0);
        assert_eq!(g.stop_loss(15.0), 0.25 * 5.0);
        assert_eq!(g.layer(5.0, 10.0), 0.25 * 5.0);
    }

    #[test]
    fn quantile_skips_points_without_mass() {
        let g = Grid::new(1.0, vec![0.5, 0.0, 0.5]).unwrap();
        assert_eq!(g.quantile(0.5), Ok(0.0));
        assert_eq!(g.quantile(0.50001), Ok(2.0));
    }

    #[test]
    fn rejects_bad_input() {
        assert!(Grid::new(0.0, vec![1.0]).is_err());
        assert!(Grid::new(1.0, vec![]).is_err());
        assert!(Grid::new(1.0, vec![0.5, 0.4]).is_err());
        assert!(Grid::new(1.0, vec![1.5, -0.5]).is_err());
        assert!(Grid::local_moment(&sev(), 100.0, 0).is_err());
        assert!(Grid::rounding(&sev(), f64::NAN, 10).is_err());
        let (g, _) = Grid::local_moment(&sev(), 100.0, 1).unwrap();
        assert_eq!(g.probs(), [1.0]);
    }
}
