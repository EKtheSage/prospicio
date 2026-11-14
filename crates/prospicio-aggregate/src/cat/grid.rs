//! An event loss table as the severity of a compound Poisson distribution
//! on a grid (#781), as `aggregate` 2.1.0 lowers `agg NAME elt [...]`: the
//! year's events a Poisson count with mean `Λ = Σ rate`, one event's loss
//! the mixture of the rows' losses by rate, bucketed on the grid, and the
//! annual total by FFT. See `docs/design/cat-inputs.md`, "An event loss
//! table on a grid".

use prospicio_core::{Error, Result};
use prospicio_prob::{Distribution, Grid, Poisson};

use super::{EventLoss, EventLossTable};
use crate::{CompoundReport, fft};

/// How an event loss table's severity is put on a grid of step `h` and
/// `n` points, `0, h, …, (n − 1) h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EltBucketing {
    /// `aggregate`'s rule for a table with secondary uncertainty (an ELT6,
    /// quantized by `eltq`'s survival method): each row's loss law is
    /// bucketed by the nearest node, node `k` taking the mass in
    /// `((k − ½) h, (k + ½) h]` and node 0 the mass in `[0, h / 2]`. A row
    /// without secondary uncertainty whose mean lies exactly on a bucket
    /// edge is split evenly between the two nodes, as `eltq` does. The
    /// mass above `(n − ½) h` moves onto the top node, so the rate is
    /// exact and the mean falls.
    #[default]
    Nearest,
    /// `aggregate`'s rule for a table of means alone (an ELT3 or ELT4, or
    /// an ELT6 under `no2u`): the secondary uncertainty is dropped, and each
    /// row's mean is an atom split between its two neighbouring nodes so
    /// that the mean is kept exactly (`aggregate`'s `dsev_bucket='linear'`).
    /// A mean within `1e-12` times the largest mean of a node lands on it
    /// whole; a mean at or above the top node lands on it.
    MeansLinear,
}

/// What putting an event loss table's severity on a grid lost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EltGridReport {
    /// The bucketing rule.
    pub bucketing: EltBucketing,
    /// The annual rate of events whose loss lies above the top bucket's
    /// edge, `(n − ½) h` (`Nearest`) or the top node (`MeansLinear`), moved
    /// onto the top node.
    pub rate_above: f64,
    /// One event's mean loss in the table under the rule: the rows' means
    /// by rate.
    pub severity_mean: f64,
    /// One event's mean loss on the grid; below `severity_mean` by the
    /// mass moved onto the top node and, under `Nearest`, by the rounding.
    pub grid_mean: f64,
}

impl EventLossTable {
    /// The year's number of events: a Poisson count with mean
    /// [`rate`](Self::rate).
    pub fn count(&self) -> Poisson {
        Poisson::new(self.rate()).expect("a table's rate is finite and positive")
    }

    /// One event's loss given that it occurs on the grid `0, step, …,
    /// (points − 1) step`: each row's loss bucketed by `bucketing`,
    /// weighted by `rate / Λ`. With [`count`](Self::count) it is the
    /// severity of the compound Poisson model of the table, for
    /// [`fft`](crate::fft) and [`Tower::on_grid`](crate::Tower::on_grid).
    ///
    /// ```
    /// use prospicio_aggregate::cat::{EltBucketing, EventLossTable};
    ///
    /// let elt = EventLossTable::new(
    ///     &[0.01, 0.05],
    ///     &[800.0, 230.0],
    ///     &[200.0, 0.0],
    ///     &[100.0, 0.0],
    ///     &[2000.0, 500.0],
    /// )
    /// .unwrap();
    /// // Means alone, each split so its mean is kept.
    /// let (sev, report) = elt.severity_grid(100.0, 32, EltBucketing::MeansLinear).unwrap();
    /// assert!((sev.probs()[2] - 0.7 * 0.05 / 0.06).abs() < 1e-15);
    /// assert!((report.grid_mean - report.severity_mean).abs() < 1e-12);
    /// // The beta row bucketed by the nearest node.
    /// let (sev, report) = elt.severity_grid(100.0, 32, EltBucketing::Nearest).unwrap();
    /// assert!((sev.probs()[2] - 0.05 / 0.06).abs() < 1e-3);
    /// assert_eq!(report.rate_above, 0.0);
    /// ```
    pub fn severity_grid(
        &self,
        step: f64,
        points: usize,
        bucketing: EltBucketing,
    ) -> Result<(Grid, EltGridReport)> {
        if !(step.is_finite() && step > 0.0) {
            return Err(Error::InvalidParameter {
                name: "step",
                value: step,
                reason: "must be finite and positive",
            });
        }
        if points < 2 {
            return Err(Error::InvalidParameter {
                name: "points",
                value: points as f64,
                reason: "must be at least 2",
            });
        }
        let total = self.rate();
        let mut probs = vec![0.0; points];
        let mut rate_above = 0.0;
        match bucketing {
            EltBucketing::Nearest => {
                let top_edge = (points as f64 - 0.5) * step;
                for (&rate, loss) in self.rates.iter().zip(&self.losses) {
                    if rate == 0.0 {
                        continue;
                    }
                    let w = rate / total;
                    match loss {
                        EventLoss::Point(m) => {
                            if *m > top_edge {
                                rate_above += rate;
                                probs[points - 1] += w;
                                continue;
                            }
                            let s = m / step + 0.5;
                            let k = s.floor();
                            if s == k && k > 0.0 {
                                // On the edge between nodes k − 1 and k.
                                let k = k as usize;
                                probs[k - 1] += 0.5 * w;
                                // On the top edge, the upper half is above
                                // the grid and moves onto the top node.
                                probs[k.min(points - 1)] += 0.5 * w;
                                if k == points {
                                    rate_above += 0.5 * rate;
                                }
                            } else {
                                probs[(k as usize).min(points - 1)] += w;
                            }
                        }
                        EventLoss::Beta(b) => {
                            // Survival at the upper edges, ½ h, 3/2 h, …
                            let mut below = 1.0;
                            for (k, p) in probs.iter_mut().enumerate() {
                                let above = b.survival((k as f64 + 0.5) * step);
                                *p += w * (below - above);
                                below = above;
                                if above == 0.0 {
                                    break;
                                }
                            }
                            if below > 0.0 {
                                rate_above += rate * below;
                                probs[points - 1] += w * below;
                            }
                        }
                    }
                }
            }
            EltBucketing::MeansLinear => {
                // aggregate's snap is relative to its largest atom, among
                // the events with a positive rate.
                let largest = self
                    .rates
                    .iter()
                    .zip(&self.means)
                    .filter(|(r, _)| **r > 0.0)
                    .map(|(_, m)| *m)
                    .fold(0.0, f64::max);
                let tol = 1e-12 * largest / step;
                let top = (points - 1) as f64;
                for (&rate, &m) in self.rates.iter().zip(&self.means) {
                    if rate == 0.0 {
                        continue;
                    }
                    let w = rate / total;
                    let mut s = m / step;
                    let near = s.round();
                    if (s - near).abs() <= tol {
                        s = near;
                    }
                    if s >= top {
                        if s > top {
                            rate_above += rate;
                        }
                        probs[points - 1] += w;
                        continue;
                    }
                    let k = s.floor();
                    let f = s - k;
                    let k = k as usize;
                    probs[k] += w * (1.0 - f);
                    probs[k + 1] += w * f;
                }
            }
        }
        let severity_mean = self.expected_annual_loss() / total;
        let grid = Grid::normalized(step, probs)?;
        let grid_mean = grid.mean();
        Ok((
            grid,
            EltGridReport {
                bucketing,
                rate_above,
                severity_mean,
                grid_mean,
            },
        ))
    }

    /// The annual total loss on the grid `0, step, …, (points − 1) step`:
    /// the Poisson count of [`count`](Self::count) compounded with
    /// [`severity_grid`](Self::severity_grid) by [`fft`](crate::fft), the
    /// severity on the same `points`. Check the compound report's
    /// `aliasing_error` and `tail_mass`, and the severity report's
    /// `rate_above`.
    ///
    /// ```
    /// use prospicio_aggregate::cat::{EltBucketing, EventLossTable};
    /// use prospicio_prob::Distribution;
    ///
    /// let elt = EventLossTable::new(&[0.5, 0.25], &[100.0, 300.0], &[0.0; 2], &[0.0; 2], &[400.0; 2])
    ///     .unwrap();
    /// let (annual, report, _) = elt.aggregate_grid(100.0, 64, EltBucketing::Nearest).unwrap();
    /// assert!((annual.mean() - elt.expected_annual_loss()).abs() < 1e-9);
    /// assert!((annual.probs()[0] - (-0.75f64).exp()).abs() < 1e-15);
    /// assert!(report.aliasing_error < 1e-12);
    /// ```
    pub fn aggregate_grid(
        &self,
        step: f64,
        points: usize,
        bucketing: EltBucketing,
    ) -> Result<(Grid, CompoundReport, EltGridReport)> {
        let (severity, sev_report) = self.severity_grid(step, points, bucketing)?;
        let (annual, report) = fft(&self.count(), &severity, points)?;
        Ok((annual, report, sev_report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> EventLossTable {
        EventLossTable::new(
            &[0.01, 0.05, 0.002, 0.0],
            &[800.0, 250.0, 1500.0, 40.0],
            &[200.0, 0.0, 300.0, 0.0],
            &[100.0, 0.0, 300.0, 0.0],
            &[2000.0, 500.0, 5000.0, 100.0],
        )
        .unwrap()
    }

    #[test]
    fn nearest_keeps_every_rows_mass_and_reports_the_overflow() {
        let elt = table();
        let (g, r) = elt.severity_grid(50.0, 64, EltBucketing::Nearest).unwrap();
        assert!((g.probs().iter().sum::<f64>() - 1.0).abs() < 1e-14);
        // The atom at 250 sits on node 5 whole.
        let beta_at_5: f64 = {
            let (g2, _) = EventLossTable::new(
                &[0.01, 0.002],
                &[800.0, 1500.0],
                &[200.0, 300.0],
                &[100.0, 300.0],
                &[2000.0, 5000.0],
            )
            .unwrap()
            .severity_grid(50.0, 64, EltBucketing::Nearest)
            .unwrap();
            g2.probs()[5] * 0.012
        };
        assert!((g.probs()[5] * 0.062 - 0.05 - beta_at_5).abs() < 1e-15);
        // Above 63.5 × 50 only the 5000-exposure row reaches.
        let b = match elt.losses[2] {
            EventLoss::Beta(b) => b,
            EventLoss::Point(_) => unreachable!(),
        };
        assert!((r.rate_above - 0.002 * b.survival(63.5 * 50.0)).abs() < 1e-18);
        assert!(r.grid_mean < r.severity_mean);
        // On a wide grid nothing is above it and the mean is close.
        let (_, wide) = elt.severity_grid(50.0, 128, EltBucketing::Nearest).unwrap();
        assert_eq!(wide.rate_above, 0.0);
        assert!((wide.grid_mean - wide.severity_mean).abs() < 1e-2);
    }

    #[test]
    fn an_atom_on_an_edge_splits_evenly() {
        let elt = EventLossTable::new(&[1.0], &[25.0], &[0.0], &[0.0], &[100.0]).unwrap();
        let (g, _) = elt.severity_grid(10.0, 12, EltBucketing::Nearest).unwrap();
        assert_eq!(&g.probs()[..4], &[0.0, 0.0, 0.5, 0.5]);
        // Just inside the top edge stays; just above moves and is reported.
        let elt = EventLossTable::new(&[1.0], &[114.9], &[0.0], &[0.0], &[200.0]).unwrap();
        let (g, r) = elt.severity_grid(10.0, 12, EltBucketing::Nearest).unwrap();
        assert_eq!(g.probs()[11], 1.0);
        assert_eq!(r.rate_above, 0.0);
        let elt = EventLossTable::new(&[1.0], &[115.1], &[0.0], &[0.0], &[200.0]).unwrap();
        let (_, r) = elt.severity_grid(10.0, 12, EltBucketing::Nearest).unwrap();
        assert_eq!(r.rate_above, 1.0);
    }

    #[test]
    fn means_linear_keeps_the_mean() {
        let elt = table();
        let (g, r) = elt
            .severity_grid(30.0, 128, EltBucketing::MeansLinear)
            .unwrap();
        assert_eq!(r.rate_above, 0.0);
        assert!((r.grid_mean - r.severity_mean).abs() < 1e-12 * r.severity_mean);
        // An atom on the lattice up to round-off lands whole.
        let elt = EventLossTable::new(&[1.0], &[0.3 * 7.0], &[0.0], &[0.0], &[10.0]).unwrap();
        let (g1, _) = elt
            .severity_grid(0.3, 16, EltBucketing::MeansLinear)
            .unwrap();
        assert_eq!(g1.probs()[7], 1.0);
        // Above the top node it lands there and is reported.
        let (g2, r2) = elt
            .severity_grid(0.3, 4, EltBucketing::MeansLinear)
            .unwrap();
        assert_eq!(g2.probs()[3], 1.0);
        assert_eq!(r2.rate_above, 1.0);
        assert!(g.probs().iter().all(|p| *p >= 0.0));
    }

    #[test]
    fn the_annual_total_is_the_compound_poisson() {
        let elt = table();
        let (annual, report, _) = elt
            .aggregate_grid(50.0, 256, EltBucketing::MeansLinear)
            .unwrap();
        assert!(report.aliasing_error < 1e-14);
        assert!((annual.probs()[0] - (-elt.rate()).exp()).abs() < 1e-15);
        assert!((annual.mean() - elt.expected_annual_loss()).abs() < 1e-10);
    }

    #[test]
    fn bad_grids_are_refused() {
        let elt = table();
        assert!(elt.severity_grid(0.0, 16, EltBucketing::Nearest).is_err());
        assert!(
            elt.severity_grid(f64::NAN, 16, EltBucketing::Nearest)
                .is_err()
        );
        assert!(elt.severity_grid(1.0, 1, EltBucketing::Nearest).is_err());
    }
}
