//! Reinsurance on the aggregate grid: exact gross, ceded and net
//! distributions without simulation.
//!
//! A per-occurrence layer is a function of each loss, so its recoveries
//! have a severity grid of their own, and their annual total is a compound
//! distribution like the gross. Annual terms are a function of that total.
//! Every step maps one grid through a function ([`Grid::map`]) or
//! compounds it ([`fft`]), so the results carry no sampling error.

use prospicio_core::{Error, Result};
use prospicio_prob::{Counting, Distortion, Distribution, Grid};

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
    /// Annual ceded loss of each stage, all its layers together at their
    /// shares: the total cession `aggregate` prices and reports as `Ceded`.
    /// `None` for a stage with several layers of which one has annual
    /// terms: the layers' annual totals are then dependent, and their sum
    /// is not one compound. A one-layer stage repeats its layer's grid.
    pub stage_ceded: Vec<Option<Grid>>,
    /// Annual net loss, gross less every layer's ceded loss, when it is a
    /// function of a single compound total; `None` otherwise (see
    /// [`Tower::on_grid`]).
    pub net: Option<Grid>,
    /// Expected reinstatement premium of each layer; zero without paid
    /// reinstatements.
    pub expected_reinstatement_premium: Vec<f64>,
    /// Expected premium of each layer for the year: the swing-rated
    /// premium's mean, or else the fixed premium (zero when none is set).
    pub expected_premium: Vec<f64>,
    /// Expected ceding commission of each layer (flat or sliding); zero
    /// without one.
    pub expected_ceding_commission: Vec<f64>,
    /// Expected profit commission of each layer; zero without one.
    pub expected_profit_commission: Vec<f64>,
    /// `true` when every layer boundary, annual term and net loss fell on
    /// a grid point, so the grids are exact up to the reports' truncation
    /// and aliasing error. When `false`, losses between points were split
    /// between their neighbours: means stay exact, the shape is smeared
    /// by up to one step.
    pub on_points: bool,
}

/// The price of one view of a tower under a distortion: expected loss `L`,
/// premium `P = ρ_g(X)`, margin `M = P − L` and loss ratio `L / P`, as the
/// Python package `aggregate`'s `reins_price_df`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewPrice {
    pub loss: f64,
    pub premium: f64,
    pub margin: f64,
    pub loss_ratio: f64,
}

impl ViewPrice {
    fn of(grid: &Grid, distortion: &Distortion) -> Self {
        let loss = grid.mean();
        let premium = grid.distortion(distortion);
        Self {
            loss,
            premium,
            margin: premium - loss,
            loss_ratio: loss / premium,
        }
    }
}

/// [`ViewPrice`] of the gross, each layer's ceded and the net annual loss;
/// see [`TowerGrids::price`].
#[derive(Debug, Clone, PartialEq)]
pub struct TowerPrice {
    pub gross: ViewPrice,
    /// One per layer, in tower order.
    pub ceded: Vec<ViewPrice>,
    /// One per stage, the total cession; `None` where
    /// [`TowerGrids::stage_ceded`] is.
    pub stages: Vec<Option<ViewPrice>>,
    /// `None` when the tower has no net grid.
    pub net: Option<ViewPrice>,
}

impl TowerGrids {
    /// Each view priced on its own by the distortion `distortion`: the
    /// stand-alone price of the gross, of every cession and of the
    /// retention. A concave distortion's prices are subadditive, so then the
    /// ceded and net premiums together are at least the gross premium.
    pub fn price(&self, distortion: &Distortion) -> TowerPrice {
        TowerPrice {
            gross: ViewPrice::of(&self.gross, distortion),
            ceded: self
                .ceded
                .iter()
                .map(|g| ViewPrice::of(g, distortion))
                .collect(),
            stages: self
                .stage_ceded
                .iter()
                .map(|g| g.as_ref().map(|g| ViewPrice::of(g, distortion)))
                .collect(),
            net: self.net.as_ref().map(|g| ViewPrice::of(g, distortion)),
        }
    }
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
    /// Probabilities below machine epsilon are FFT round-off and are set
    /// to 0 in the ceded and net grids ([`Grid::remove_fuzz`], as
    /// `aggregate`'s `remove_fuzz`); a premium that weights the tail
    /// heavily, such as the exponential principle, would otherwise read the
    /// round-off at the top of the grid.
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
        let mut premium = Vec::with_capacity(self.layers.len());
        let mut commission = Vec::with_capacity(self.layers.len());
        let mut profit = Vec::with_capacity(self.layers.len());
        let mut stage_ceded = Vec::with_capacity(stages.len());
        for &(start, end) in &stages {
            for layer in &self.layers[start..end] {
                let (recoveries, exact) = severity.map(|x| layer.recovery(seen(x, start)))?;
                on_points &= exact;
                let (annual, report) = fft(frequency, &recoveries, points)?;
                let (after_terms, exact) = annual.map(|r| layer.after_terms(r))?;
                on_points &= exact;
                premiums.push(expected_reinstatement_premium(layer, &after_terms));
                let layer_ceded = Grid::new(
                    layer.share * after_terms.step(),
                    after_terms.probs().to_vec(),
                )?;
                premium.push(expectation(&layer_ceded, |c| layer.premium_for(c)));
                commission.push(expectation(&layer_ceded, |c| {
                    layer.ceding_commission_for(c)
                }));
                profit.push(expectation(&layer_ceded, |c| {
                    layer.profit_commission_for(c)
                }));
                ceded.push(layer_ceded.remove_fuzz(f64::EPSILON));
                ceded_reports.push(report);
            }
            let layers = &self.layers[start..end];
            stage_ceded.push(if layers.len() == 1 {
                ceded.last().cloned()
            } else if layers.iter().any(Layer::has_annual_terms) {
                None
            } else {
                // Without annual terms the stage's cession is a function of
                // each loss: compound it once.
                let (recoveries, exact) = severity.map(|x| {
                    let seen = seen(x, start);
                    layers.iter().map(|l| l.share * l.recovery(seen)).sum()
                })?;
                on_points &= exact;
                Some(
                    fft(frequency, &recoveries, points)?
                        .0
                        .remove_fuzz(f64::EPSILON),
                )
            });
        }

        let net = if !self.layers.iter().any(Layer::has_annual_terms) {
            let all = self.layers.len();
            let (net_severity, exact) = severity.map(|x| seen(x, all))?;
            on_points &= exact;
            Some(
                fft(frequency, &net_severity, points)?
                    .0
                    .remove_fuzz(f64::EPSILON),
            )
        } else if last_end - last_start == 1 && is_aggregate_cover(&self.layers[last_start]) {
            let cover = &self.layers[last_start];
            let (before, exact) = severity.map(|x| seen(x, last_start))?;
            on_points &= exact;
            let (total, _) = fft(frequency, &before, points)?;
            let (net, exact) = total.map(|s| s - cover.share * cover.after_terms(s))?;
            on_points &= exact;
            Some(net.remove_fuzz(f64::EPSILON))
        } else {
            None
        };

        Ok(TowerGrids {
            gross,
            gross_report,
            ceded,
            ceded_reports,
            stage_ceded,
            net,
            expected_reinstatement_premium: premiums,
            expected_premium: premium,
            expected_ceding_commission: commission,
            expected_profit_commission: profit,
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

/// A tower applied to an annual loss distribution, from
/// [`Tower::on_annual_grid`].
#[derive(Debug, Clone, PartialEq)]
pub struct AnnualGrids {
    /// The annual loss given.
    pub gross: Grid,
    /// Annual ceded loss of each layer, in tower order, at the placed share
    /// and after annual terms. A layer with share `c` has step `c h`.
    pub ceded: Vec<Grid>,
    /// Annual net loss, gross less every layer's ceded loss.
    pub net: Grid,
    /// Expected reinstatement premium of each layer.
    pub expected_reinstatement_premium: Vec<f64>,
    /// Expected premium of each layer: the swing-rated premium's mean, or
    /// else the fixed premium.
    pub expected_premium: Vec<f64>,
    /// Expected ceding commission (flat or sliding) of each layer.
    pub expected_ceding_commission: Vec<f64>,
    /// Expected profit commission of each layer.
    pub expected_profit_commission: Vec<f64>,
    /// `true` when every ceded and net loss fell on a grid point; when
    /// `false`, means are exact and shapes smeared by up to a step. The
    /// expectations above are exact either way: they are taken at each
    /// annual loss.
    pub on_points: bool,
}

impl Tower {
    /// The tower applied to an annual loss on a grid, each year's total
    /// treated as one loss, as [`Tower::apply_aggregate`] does for draws:
    /// a quota share or a stop-loss priced on a loss-ratio or annual-loss
    /// distribution, with its sliding-scale or profit commission. Every
    /// layer's terms are a function of the annual total, so inuring
    /// stages and annual terms are all allowed, and the net exists.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
    /// use prospicio_prob::{Distribution, Grid};
    ///
    /// // A loss ratio of 50%, 60%, 70% or 80% on premium 100, equally likely.
    /// let mut probs = vec![0.0; 81];
    /// for k in [50, 60, 70, 80] {
    ///     probs[k] = 0.25;
    /// }
    /// let annual = Grid::new(1.0, probs).unwrap();
    /// // A 40% quota share with a 30% flat ceding commission on 40 of premium.
    /// let qs = Layer::quota_share("QS", 0.4).unwrap()
    ///     .deposit_premium(100.0).unwrap()
    ///     .ceding_commission(0.3).unwrap();
    /// let r = Tower::new(vec![qs]).unwrap().on_annual_grid(&annual).unwrap();
    /// assert!((r.ceded[0].mean() - 26.0).abs() < 1e-12); // 0.4 × 65
    /// assert!((r.expected_ceding_commission[0] - 12.0).abs() < 1e-12); // 0.3 × 40
    /// assert!((r.net.mean() - 39.0).abs() < 1e-12);
    /// ```
    pub fn on_annual_grid(&self, annual: &Grid) -> Result<AnnualGrids> {
        if let Some(l) = self.layers.iter().find(|l| l.needs_sums_insured()) {
            return Err(Error::Data(format!(
                "layer {:?} is a surplus treaty, which works risk by risk; an annual loss \
                 has no sums insured",
                l.name
            )));
        }
        if let Some(l) = self.layers.iter().find(|l| l.needs_times()) {
            return Err(Error::Data(format!(
                "layer {:?} has reinstatements pro rata as to time; an annual loss has no \
                 event times",
                l.name
            )));
        }
        let h = annual.step();
        let m = self.layers.len();
        // Each layer's (ceded, reinstatement premium) at every point.
        let years: Vec<Vec<(f64, f64)>> = (0..annual.len())
            .map(|k| self.year(&[k as f64 * h], None, None))
            .collect();
        let at = |x: f64| &years[((x / h).round() as usize).min(years.len() - 1)];
        let mut on_points = true;
        let mut ceded = Vec::with_capacity(m);
        let (mut reinstatement, mut premium, mut commission, mut profit) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, layer) in self.layers.iter().enumerate() {
            // At 100%, then the share as the step, as `on_grid`.
            let (full, exact) = annual.map(|x| {
                let c = at(x)[i].0;
                if layer.share > 0.0 {
                    c / layer.share
                } else {
                    0.0
                }
            })?;
            on_points &= exact;
            ceded.push(Grid::new(layer.share * full.step(), full.probs().to_vec())?);
            let mean = |f: &dyn Fn(f64) -> f64| -> f64 {
                annual
                    .probs()
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| **p > 0.0)
                    .map(|(k, p)| p * f(years[k][i].0))
                    .sum()
            };
            reinstatement.push(
                annual
                    .probs()
                    .iter()
                    .zip(&years)
                    .map(|(p, y)| p * y[i].1)
                    .sum(),
            );
            premium.push(mean(&|c| layer.premium_for(c)));
            commission.push(mean(&|c| layer.ceding_commission_for(c)));
            profit.push(mean(&|c| layer.profit_commission_for(c)));
        }
        let (net, exact) = annual.map(|x| x - at(x).iter().map(|(c, _)| c).sum::<f64>())?;
        on_points &= exact;
        Ok(AnnualGrids {
            gross: annual.clone(),
            ceded,
            net,
            expected_reinstatement_premium: reinstatement,
            expected_premium: premium,
            expected_ceding_commission: commission,
            expected_profit_commission: profit,
            on_points,
        })
    }
}

/// `E[f(X)]` for `X` distributed as the grid: exact for the discretized
/// problem, as the grids are.
fn expectation(grid: &Grid, f: impl Fn(f64) -> f64) -> f64 {
    let h = grid.step();
    grid.probs()
        .iter()
        .enumerate()
        .filter(|(_, p)| **p > 0.0)
        .map(|(k, p)| p * f(k as f64 * h))
        .sum()
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

    /// The annual grid against `apply_aggregate` on the same years drawn
    /// with their probabilities' weights: a quota share with a sliding
    /// scale and a profit commission, and a stop-loss inuring to it.
    #[test]
    fn annual_grid_matches_apply_aggregate() {
        use prospicio_prob::{Empirical, KeyValue, PredictiveDistribution, Provenance};
        // Losses 40, 55, 70, 85, 120 with weights 3, 4, 3, 2, 1 of 13.
        let (xs, ws) = ([40usize, 55, 70, 85, 120], [3usize, 4, 3, 2, 1]);
        let mut probs = vec![0.0; 121];
        let mut draws = Vec::new();
        for (&x, &w) in xs.iter().zip(&ws) {
            probs[x] = w as f64 / 13.0;
            draws.extend(std::iter::repeat_n(x as f64, w));
        }
        let annual = Grid::new(1.0, probs).unwrap();
        let qs = Layer::quota_share("QS", 0.5)
            .unwrap()
            .premium_rate(1.0, 200.0)
            .unwrap()
            .sliding_scale(vec![(0.45, 0.60), (0.25, 0.70), (0.19, 0.80)])
            .unwrap();
        let pc = Layer::quota_share("PC", 0.2)
            .unwrap()
            .premium_rate(1.0, 200.0)
            .unwrap()
            .profit_commission(0.25, 0.1)
            .unwrap();
        let sl = Layer::stop_loss("SL", 30.0, 20.0).unwrap();
        let tower = Tower::inuring(vec![vec![qs, pc], vec![sl]]).unwrap();
        let r = tower.on_annual_grid(&annual).unwrap();
        let pd = PredictiveDistribution::from_draws(
            vec!["line".into()],
            vec![vec![KeyValue::from("all")]],
            draws,
            Provenance::new("test"),
        )
        .unwrap();
        let sims = tower.apply_aggregate(&pd).unwrap();
        let mean = |kind: &str, layer: &str| {
            let m = sims
                .marginal(&vec![KeyValue::from(kind), KeyValue::from(layer)])
                .unwrap();
            m.draws().iter().sum::<f64>() / m.draws().len() as f64
        };
        for (i, name) in ["QS", "PC", "SL"].iter().enumerate() {
            assert!(
                (r.ceded[i].mean() - mean("ceded", name)).abs() < 1e-12,
                "{name}"
            );
        }
        assert!((r.net.mean() - mean("net", "retained")).abs() < 1e-12);
        assert!((r.expected_ceding_commission[0] - mean("ceding_commission", "QS")).abs() < 1e-12);
        assert!((r.expected_profit_commission[1] - mean("profit_commission", "PC")).abs() < 1e-12);
        assert!(
            (r.gross.mean() - r.ceded.iter().map(|g| g.mean()).sum::<f64>() - r.net.mean()).abs()
                < 1e-12
        );
    }

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
    fn a_stage_with_several_layers_has_a_total_cession() {
        let freq = Poisson::new(3.0).unwrap();
        // 2 xs 2 and 4 xs 4 side by side are 6 xs 2.
        let split = Tower::new(vec![
            Layer::xol("2x2", 2.0, 2.0).unwrap(),
            Layer::xol("4x4", 4.0, 4.0).unwrap(),
        ])
        .unwrap()
        .on_grid(&freq, &severity(), 200)
        .unwrap();
        let whole = Tower::new(vec![Layer::xol("6x2", 6.0, 2.0).unwrap()])
            .unwrap()
            .on_grid(&freq, &severity(), 200)
            .unwrap();
        let total = split.stage_ceded[0].as_ref().unwrap();
        for (a, b) in total.probs().iter().zip(whole.ceded[0].probs()) {
            assert!((a - b).abs() < 1e-14);
        }
        assert_eq!(whole.stage_ceded[0].as_ref(), Some(&whole.ceded[0]));
        let price = split.price(&Distortion::proportional_hazard(0.7).unwrap());
        let stage = price.stages[0].unwrap();
        assert!((stage.loss - price.ceded.iter().map(|c| c.loss).sum::<f64>()).abs() < 1e-12);
        // Subadditive: the layers priced alone cost at least the stage.
        assert!(price.ceded.iter().map(|c| c.premium).sum::<f64>() >= stage.premium - 1e-12);
        // With annual terms in a stage of two the total is not one compound.
        let terms = Tower::new(vec![
            Layer::xol("2x2", 2.0, 2.0)
                .unwrap()
                .aggregate_limit(4.0)
                .unwrap(),
            Layer::xol("4x4", 4.0, 4.0).unwrap(),
        ])
        .unwrap()
        .on_grid(&freq, &severity(), 200)
        .unwrap();
        assert!(terms.stage_ceded[0].is_none());
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
    fn expected_contract_terms_match_simulation() {
        // A quota share with a sliding scale and a profit commission, and a
        // swing-rated excess layer with a flat commission, each in its own
        // stage-one tower (no inuring, so every grid is marginal and exact).
        let qs = Layer::quota_share("QS", 0.5)
            .unwrap()
            .premium_rate(1.0, 30.0)
            .unwrap()
            .sliding_scale(vec![(0.4, 0.5), (0.2, 0.9)])
            .unwrap()
            .profit_commission(0.3, 0.25)
            .unwrap();
        let swing = crate::LossSensitivePremium::new(1.0, 1.2, Some(2.0), Some(6.0)).unwrap();
        let xl = Layer::xol("3x3", 3.0, 3.0)
            .unwrap()
            .share(0.5)
            .unwrap()
            .swing_rated(swing)
            .unwrap()
            .ceding_commission(0.1)
            .unwrap();
        let tower = Tower::new(vec![qs, xl]).unwrap();
        let freq = Poisson::new(3.0).unwrap();
        let r = tower.on_grid(&freq, &severity(), 400).unwrap();
        let n = 200_000;
        let sim = tower
            .apply(&simulate_events(&freq, &severity(), n, 9).unwrap())
            .unwrap();
        let check = |kind: &str, layer: &str, expected: f64| {
            let draws = sim.marginal(&vec![kind.into(), layer.into()]).unwrap();
            let se = (draws.variance() / n as f64).sqrt();
            assert!(
                (expected - draws.mean()).abs() < 4.0 * se,
                "{kind} {layer}: {expected} vs {} (se {se})",
                draws.mean()
            );
        };
        check("ceding_commission", "QS", r.expected_ceding_commission[0]);
        check("profit_commission", "QS", r.expected_profit_commission[0]);
        check("swing_premium", "3x3", r.expected_premium[1]);
        check("ceding_commission", "3x3", r.expected_ceding_commission[1]);
        assert!((r.expected_premium[0] - 15.0).abs() < 1e-12);
        assert_eq!(r.expected_profit_commission[1], 0.0);
        // The terms are not constant: the scale slides and the swing moves.
        assert!(r.expected_ceding_commission[0] < 0.4 * 15.0);
        assert!(r.expected_premium[1] > 0.5 * 2.0);
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
