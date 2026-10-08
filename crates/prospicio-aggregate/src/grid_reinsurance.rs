//! Reinsurance on the aggregate grid: exact gross, ceded and net
//! distributions without simulation.
//!
//! A per-occurrence layer is a function of each loss, so its recoveries
//! have a severity grid of their own, and their annual total is a compound
//! distribution like the gross. Annual terms are a function of that total.
//! Every step maps one grid through a function ([`Grid::map`]) or
//! compounds it ([`fft`]), so the results carry no sampling error.

use prospicio_core::{Error, Result};
use prospicio_prob::{Counting, Grid};

use crate::compound::CompoundReport;
use crate::fft::fft;
use crate::reinsurance::{Layer, Tower};

/// A tower's annual distributions on the grid; see [`Tower::on_grid`].
#[derive(Debug, Clone, PartialEq)]
pub struct TowerGrids {
    /// Annual gross loss.
    pub gross: Grid,
    pub gross_report: CompoundReport,
    /// Annual ceded loss of each layer, in tower order, at the placed share
    /// and after annual terms. A layer with share `c` has step `c h`.
    pub ceded: Vec<Grid>,
    /// The compound calculation behind each layer: the annual recovery at
    /// 100%, before annual terms.
    pub ceded_reports: Vec<CompoundReport>,
    /// Annual net loss, gross less every layer's ceded loss, when it is a
    /// function of a single compound total; `None` otherwise (see
    /// [`Tower::on_grid`]).
    pub net: Option<Grid>,
    /// Expected reinstatement premium of each layer; zero without paid
    /// reinstatements.
    pub expected_reinstatement_premium: Vec<f64>,
    /// `true` when every layer boundary, annual term and net loss fell on
    /// a grid point, so the grids are exact up to the reports' truncation
    /// and aliasing error. When `false`, losses between points were split
    /// between their neighbours: means stay exact, the shape is smeared
    /// by up to one step.
    pub on_points: bool,
}

impl Tower {
    /// Gross, ceded and net annual distributions for `frequency` claims a
    /// year with severity `severity`, each on `points` points, by FFT.
    ///
    /// Each layer's per-occurrence recoveries form a severity grid; their
    /// annual total is compounded with the same claim count (a claim that
    /// misses the layer recovers 0), and annual terms and the share are
    /// applied to that total. With attachments, limits and annual terms on
    /// multiples of the step, the result is the exact distribution of the
    /// discretized problem, which Monte Carlo only samples.
    ///
    /// Grids are marginal: each holds one quantity's distribution, not the
    /// joint distribution across layers (use [`Tower::apply`] on simulated
    /// events for that). Net is a single compound total, and so returned,
    /// when either
    ///
    /// - no layer has annual terms: net is a function of each loss; or
    /// - the last stage is one aggregate cover (attachment 0, unlimited per
    ///   occurrence, such as a stop-loss): net is a function of the annual
    ///   total net of earlier stages.
    ///
    /// Fails if a layer with annual terms inures to a later stage: how much
    /// of each event such a layer takes depends on event order, which a
    /// compound distribution does not have.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
    /// use prospicio_prob::{Distribution, Grid, Poisson};
    ///
    /// let sev = Grid::new(1.0, vec![0.0, 0.4, 0.3, 0.2, 0.1]).unwrap();
    /// let tower = Tower::new(vec![Layer::xol("2x2", 2.0, 2.0).unwrap()]).unwrap();
    /// let r = tower.on_grid(&Poisson::new(3.0).unwrap(), &sev, 200).unwrap();
    /// // E[ceded] = E[N] E[min((X - 2)+, 2)] = 3 × (0.2 × 1 + 0.1 × 2).
    /// assert!((r.ceded[0].mean() - 1.2).abs() < 1e-12);
    /// let net = r.net.unwrap();
    /// assert!((r.gross.mean() - r.ceded[0].mean() - net.mean()).abs() < 1e-12);
    /// assert!(r.on_points);
    /// ```
    pub fn on_grid<N: Counting + ?Sized>(
        &self,
        frequency: &N,
        severity: &Grid,
        points: usize,
    ) -> Result<TowerGrids> {
        let stages = self.stage_ranges();
        let (last_start, last_end) = *stages.last().expect("a tower has a layer");
        if let Some(l) = self.layers.iter().find(|l| l.needs_sums_insured()) {
            return Err(Error::Data(format!(
                "layer {:?} is a surplus treaty, which works risk by risk; a compound \
                 distribution has no sums insured: use Tower::apply on events from a risk profile",
                l.name
            )));
        }
        if let Some(l) = self.layers.iter().find(|l| l.needs_times()) {
            return Err(Error::Data(format!(
                "layer {:?} has reinstatements pro rata as to time; a compound distribution \
                 has no event times: use Tower::apply on events with times",
                l.name
            )));
        }
        for &(start, end) in &stages[..stages.len() - 1] {
            if let Some(l) = self.layers[start..end]
                .iter()
                .find(|l| l.has_annual_terms())
            {
                return Err(Error::InvalidParameter {
                    name: "layers",
                    value: l.aggregate_deductible,
                    reason: "annual terms in a stage that inures to a later one depend on \
                             event order; use Tower::apply on simulated events",
                });
            }
        }

        // The loss each stage sees from a gross loss x: x net of every
        // earlier stage, event by event.
        let seen = |x: f64, stage_start: usize| -> f64 {
            let mut x = x;
            for &(start, end) in stages.iter().take_while(|(s, _)| *s < stage_start) {
                x -= self.layers[start..end]
                    .iter()
                    .map(|l| l.share * l.recovery(x))
                    .sum::<f64>();
            }
            x
        };

        let mut on_points = true;
        let (gross, gross_report) = fft(frequency, severity, points)?;
        let mut ceded = Vec::with_capacity(self.layers.len());
        let mut ceded_reports = Vec::with_capacity(self.layers.len());
        let mut premiums = Vec::with_capacity(self.layers.len());
        for &(start, end) in &stages {
            for layer in &self.layers[start..end] {
                let (recoveries, exact) = severity.map(|x| layer.recovery(seen(x, start)))?;
                on_points &= exact;
                let (annual, report) = fft(frequency, &recoveries, points)?;
                let (after_terms, exact) = annual.map(|r| layer.after_terms(r))?;
                on_points &= exact;
                premiums.push(expected_reinstatement_premium(layer, &after_terms));
                ceded.push(Grid::new(
                    layer.share * after_terms.step(),
                    after_terms.probs().to_vec(),
                )?);
                ceded_reports.push(report);
            }
        }

        let net = if !self.layers.iter().any(Layer::has_annual_terms) {
            let all = self.layers.len();
            let (net_severity, exact) = severity.map(|x| seen(x, all))?;
            on_points &= exact;
            Some(fft(frequency, &net_severity, points)?.0)
        } else if last_end - last_start == 1 && is_aggregate_cover(&self.layers[last_start]) {
            let cover = &self.layers[last_start];
            let (before, exact) = severity.map(|x| seen(x, last_start))?;
            on_points &= exact;
            let (total, _) = fft(frequency, &before, points)?;
            let (net, exact) = total.map(|s| s - cover.share * cover.after_terms(s))?;
            on_points &= exact;
            Some(net)
        } else {
            None
        };

        Ok(TowerGrids {
            gross,
            gross_report,
            ceded,
            ceded_reports,
            net,
            expected_reinstatement_premium: premiums,
            on_points,
        })
    }

    /// `(start, end)` of each stage's layers.
    fn stage_ranges(&self) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let mut start = 0;
        while start < self.layers.len() {
            let stage = self.stages[start];
            let end = start
                + self.stages[start..]
                    .iter()
                    .take_while(|&&s| s == stage)
                    .count();
            ranges.push((start, end));
            start = end;
        }
        ranges
    }
}

/// Unlimited per-occurrence cover from 0: the layer sees the annual total.
fn is_aggregate_cover(layer: &Layer) -> bool {
    layer.attachment == 0.0 && layer.limit == f64::INFINITY
}

/// `E[premium × Σ_k rates[k] × min(max(L - k l, 0), l) / l]` over the
/// annual layer loss `L` at 100%.
fn expected_reinstatement_premium(layer: &Layer, loss: &Grid) -> f64 {
    if layer.reinstatement_rates.is_empty() {
        return 0.0;
    }
    let l = layer.limit;
    loss.probs()
        .iter()
        .enumerate()
        .map(|(j, &p)| {
            let x = loss.x(j);
            let used: f64 = layer
                .reinstatement_rates
                .iter()
                .enumerate()
                .map(|(k, rate)| rate * (x - k as f64 * l).clamp(0.0, l))
                .sum();
            p * used
        })
        .sum::<f64>()
        * layer.premium
        / l
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulate_events;
    use prospicio_prob::{Distribution, NegativeBinomial, Poisson, Severity};

    fn severity() -> Grid {
        // Losses 0..=11 on a unit step.
        Grid::new(
            1.0,
            vec![
                0.0, 0.15, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02,
            ],
        )
        .unwrap()
    }

    /// Largest CDF difference between a grid and a sample on `0, h, …`.
    fn ks(grid: &Grid, sample: &prospicio_prob::Sampled, points: usize) -> f64 {
        (0..points)
            .map(|k| {
                let x = grid.x(k);
                (grid.cdf(x) - sample.cdf(x)).abs()
            })
            .fold(0.0, f64::max)
    }

    /// Every grid of `tower` against 200,000 simulated years, by a
    /// Kolmogorov–Smirnov statistic at the 0.1% level.
    fn matches_simulation(tower: &Tower, freq: &(dyn Counting + Sync)) {
        let points = 400;
        let r = tower.on_grid(freq, &severity(), points).unwrap();
        assert!(r.on_points);
        assert!(r.gross_report.tail_mass < 1e-12);
        let n_sims = 200_000;
        let sim = tower
            .apply(&simulate_events(freq, &severity(), n_sims, 3).unwrap())
            .unwrap();
        let critical = 1.95 / (n_sims as f64).sqrt();
        let component =
            |kind: &str, layer: &str| sim.marginal(&vec![kind.into(), layer.into()]).unwrap();
        let d = ks(&r.gross, &component("gross", "ground_up"), points);
        assert!(d < critical, "gross KS {d}");
        for (layer, grid) in tower.layers.iter().zip(&r.ceded) {
            let d = ks(grid, &component("ceded", &layer.name), points);
            assert!(d < critical, "{} KS {d}", layer.name);
        }
        if let Some(net) = &r.net {
            let d = ks(net, &component("net", "retained"), points);
            assert!(d < critical, "net KS {d}");
        }
    }

    #[test]
    fn per_occurrence_layers_match_simulation() {
        let tower = Tower::new(vec![
            Layer::xol("2x2", 2.0, 2.0).unwrap(),
            Layer::xol("4x4", 4.0, 4.0).unwrap(),
        ])
        .unwrap();
        matches_simulation(&tower, &Poisson::new(3.0).unwrap());
        matches_simulation(&tower, &NegativeBinomial::new(2.5, 1.5).unwrap());
    }

    #[test]
    fn annual_terms_match_simulation() {
        let tower = Tower::new(vec![
            Layer::xol("3x3", 3.0, 3.0)
                .unwrap()
                .aggregate_deductible(2.0)
                .unwrap()
                .reinstatements(1)
                .unwrap(),
        ])
        .unwrap();
        let r = tower
            .on_grid(&Poisson::new(3.0).unwrap(), &severity(), 400)
            .unwrap();
        // One layer with annual terms: no net grid.
        assert!(r.net.is_none());
        matches_simulation(&tower, &Poisson::new(3.0).unwrap());
    }

    #[test]
    fn loss_corridors_match_simulation() {
        // Corridor bounds on the unit points, so the grids are exact.
        let tower = Tower::new(vec![
            Layer::xol("3x3", 3.0, 3.0)
                .unwrap()
                .loss_corridor(1.0, 4.0, 1.0)
                .unwrap()
                .aggregate_limit(6.0)
                .unwrap(),
        ])
        .unwrap();
        matches_simulation(&tower, &Poisson::new(3.0).unwrap());
        // A stop-loss with a corridor is still one compound total: net has
        // a grid.
        let tower = Tower::inuring(vec![
            vec![Layer::xol("4x4", 4.0, 4.0).unwrap()],
            vec![
                Layer::stop_loss("SL", 10.0, 12.0)
                    .unwrap()
                    .loss_corridor(2.0, 5.0, 1.0)
                    .unwrap(),
            ],
        ])
        .unwrap();
        let r = tower
            .on_grid(&Poisson::new(3.0).unwrap(), &severity(), 400)
            .unwrap();
        let net = r.net.as_ref().unwrap();
        let ceded: f64 = r.ceded.iter().map(|g| g.mean()).sum();
        assert!((r.gross.mean() - ceded - net.mean()).abs() < 1e-10);
        matches_simulation(&tower, &Poisson::new(3.0).unwrap());
        // A corridor is an annual term: it may not inure on the grid.
        let tower = Tower::inuring(vec![
            vec![
                Layer::quota_share("QS", 0.5)
                    .unwrap()
                    .loss_corridor(1.0, 2.0, 1.0)
                    .unwrap(),
            ],
            vec![Layer::xol("4x4", 4.0, 4.0).unwrap()],
        ])
        .unwrap();
        assert!(
            tower
                .on_grid(&Poisson::new(3.0).unwrap(), &severity(), 100)
                .is_err()
        );
    }

    #[test]
    fn inuring_stop_loss_has_a_net_grid() {
        let tower = Tower::inuring(vec![
            vec![Layer::xol("4x4", 4.0, 4.0).unwrap()],
            vec![Layer::stop_loss("SL", 10.0, 12.0).unwrap()],
        ])
        .unwrap();
        let r = tower
            .on_grid(&Poisson::new(3.0).unwrap(), &severity(), 400)
            .unwrap();
        let net = r.net.as_ref().unwrap();
        let ceded: f64 = r.ceded.iter().map(|g| g.mean()).sum();
        assert!((r.gross.mean() - ceded - net.mean()).abs() < 1e-10);
        matches_simulation(&tower, &Poisson::new(3.0).unwrap());
    }

    #[test]
    fn quota_share_scales_the_step() {
        let tower = Tower::inuring(vec![
            vec![Layer::quota_share("QS", 0.25).unwrap()],
            vec![Layer::xol("4x4", 4.0, 4.0).unwrap()],
        ])
        .unwrap();
        let freq = Poisson::new(3.0).unwrap();
        let r = tower.on_grid(&freq, &severity(), 400).unwrap();
        assert_eq!(r.ceded[0].step(), 0.25);
        assert!((r.ceded[0].mean() - 0.25 * r.gross.mean()).abs() < 1e-10);
        // The cover sees 75% of each loss, off the unit points: the mean is
        // still exact.
        assert!(!r.on_points);
        let want = 3.0
            * (0..12)
                .map(|j| severity().probs()[j] * (0.75 * j as f64 - 4.0).clamp(0.0, 4.0))
                .sum::<f64>();
        assert!((r.ceded[1].mean() - want).abs() < 1e-10);
        let net = r.net.unwrap();
        assert!(
            (r.gross.mean() - r.ceded[0].mean() - r.ceded[1].mean() - net.mean()).abs() < 1e-10
        );
    }

    #[test]
    fn expected_reinstatement_premium_matches_simulation() {
        let tower = Tower::new(vec![
            Layer::xol("3x3", 3.0, 3.0)
                .unwrap()
                .paid_reinstatements(1.0, vec![1.0, 0.5])
                .unwrap(),
        ])
        .unwrap();
        let freq = Poisson::new(3.0).unwrap();
        let r = tower.on_grid(&freq, &severity(), 400).unwrap();
        let sim = tower
            .apply(&simulate_events(&freq, &severity(), 200_000, 5).unwrap())
            .unwrap();
        let premiums = sim
            .marginal(&vec!["reinstatement_premium".into(), "3x3".into()])
            .unwrap();
        let se = (premiums.variance() / 200_000.0).sqrt();
        assert!((r.expected_reinstatement_premium[0] - premiums.mean()).abs() < 4.0 * se);
    }

    #[test]
    fn layer_means_are_exact_off_the_points() {
        // Boundaries between points: means stay exact.
        let tower = Tower::new(vec![Layer::xol("2.5x1.5", 2.5, 1.5).unwrap()]).unwrap();
        let r = tower
            .on_grid(&Poisson::new(3.0).unwrap(), &severity(), 400)
            .unwrap();
        assert!(!r.on_points);
        let want = 3.0 * severity().layer(2.5, 1.5);
        assert!((r.ceded[0].mean() - want).abs() < 1e-10);
    }

    #[test]
    fn rejects_annual_terms_that_inure() {
        let tower = Tower::inuring(vec![
            vec![
                Layer::xol("2x2", 2.0, 2.0)
                    .unwrap()
                    .reinstatements(1)
                    .unwrap(),
            ],
            vec![Layer::xol("4x4", 4.0, 4.0).unwrap()],
        ])
        .unwrap();
        assert!(
            tower
                .on_grid(&Poisson::new(3.0).unwrap(), &severity(), 100)
                .is_err()
        );
    }
}
