//! Over-dispersed Poisson (ODP) bootstrap of the chain ladder, after England
//! and Verrall (2002), as implemented by R ChainLadder's `BootChainLadder`.
//!
//! 1. Fit the volume-weighted chain ladder and back out fitted incremental
//!    values from each origin's latest diagonal.
//! 2. Pearson residuals `(X - m) / sqrt(|m|)` on every observed cell, the
//!    scale `phi = sum(r^2) / (n - p)` and residuals adjusted by
//!    `sqrt(n / (n - p))`, with `n` observed cells and `p = origins + ages - 1`
//!    parameters.
//! 3. Per simulation: resample adjusted residuals with replacement into a
//!    pseudo triangle `m + r* sqrt(|m|)`, re-estimate the factors, project
//!    the expected future incrementals and add process error to each.
//!
//! Every simulation draws from its own [`StreamRng`] stream
//! (`PredictiveDistribution::simulate`), so results do not depend on the
//! number of threads.

use act_core::{Lag, Period, StreamRng};
use act_prob::{
    Distribution, Gamma, InputHasher, KeyValue, PredictiveDistribution, Provenance, Sampled,
};

use crate::chain_ladder::{ChainLadder, ChainLadderFit};
use crate::error::{Error, Result};
use crate::segments::{FitTable, ReserveFit, SegmentFits, fit_each};
use crate::triangle::{Label, Segment, Triangle};

/// Process error added to each simulated future incremental value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProcessDistribution {
    /// Gamma with the expected value as mean and variance `phi |m|`, signed
    /// like the mean (R's `process.distr = "gamma"`).
    #[default]
    Gamma,
    /// No process error: the expected future values of each resampled
    /// triangle, i.e. parameter error only (R's `ParamDist`).
    None,
}

/// ODP bootstrap of the chain ladder.
///
/// ```
/// use act_reserving::{DevelopmentColumn, Grain, Long, Month, OdpBootstrap, Triangle};
/// use act_prob::Distribution;
///
/// let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
///     values: &[(
///         "paid",
///         &[100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
///     )],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let boot = OdpBootstrap { n_sims: 2_000, seed: 42, ..Default::default() }.fit(&tri, "paid")?;
/// assert_eq!(boot.reserves.n_components(), 4);
/// assert!(boot.reserves.mean() > 0.0);
/// # Ok::<(), act_reserving::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OdpBootstrap {
    /// Number of simulations.
    pub n_sims: usize,
    /// Seed of the simulation streams; simulation `i` uses stream `i`.
    pub seed: u64,
    /// Process error on the simulated future values.
    pub process: ProcessDistribution,
}

impl Default for OdpBootstrap {
    fn default() -> Self {
        Self {
            n_sims: 10_000,
            seed: 0,
            process: ProcessDistribution::Gamma,
        }
    }
}

/// A fitted ODP bootstrap.
#[derive(Debug, Clone)]
pub struct OdpBootstrapFit {
    /// The deterministic volume-weighted chain ladder the bootstrap is
    /// centred on.
    pub chain_ladder: ChainLadderFit,
    /// Fitted incremental values, row-major over origin × development; NaN
    /// where the triangle is not observed.
    pub fitted: Vec<f64>,
    /// Adjusted Pearson residuals, laid out like `fitted`; NaN where not
    /// observed or where the fitted value is zero.
    pub residuals: Vec<f64>,
    /// The scale parameter `phi`.
    pub scale: f64,
    /// Joint distribution of the reserve (sum of future incremental values)
    /// by origin: dimension `origin`, one component per origin period.
    pub reserves: PredictiveDistribution,
}

/// What the bootstrap estimates in one segment before simulating: the
/// fields of [`OdpBootstrapFit`] but the reserves.
#[derive(Debug, Clone, PartialEq)]
pub struct OdpBootstrapSegment {
    /// The volume-weighted chain ladder of the segment.
    pub chain_ladder: ChainLadderFit,
    /// Fitted incremental values, row-major over origin × development.
    pub fitted: Vec<f64>,
    /// Adjusted Pearson residuals, laid out like `fitted`.
    pub residuals: Vec<f64>,
    /// The segment's scale parameter `phi`.
    pub scale: f64,
}

impl ReserveFit for OdpBootstrapSegment {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("scale", self.scale)]
    }
}

/// An ODP bootstrap of every segment of a triangle column.
///
/// Segments are bootstrapped independently, each with its own residuals
/// and scale. Simulation `i` uses stream `i` for every segment, in index
/// order, so the result is reproducible and does not depend on the number
/// of threads; a segment's draws differ from bootstrapping it alone.
///
/// ```
/// use act_reserving::{DevelopmentColumn, Grain, Long, Month, OdpBootstrap, Triangle};
/// use act_prob::Distribution;
///
/// let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
/// let ages = [12, 24, 36, 12, 24, 12];
/// let paid = [100.0, 150.0, 165.0, 110.0, 170.0, 120.0];
/// let tri = Triangle::from_long(&Long {
///     keys: &[("lob", &[["Auto"; 6], ["Home"; 6]].concat())],
///     origin: &[origin, origin].concat(),
///     development: DevelopmentColumn::Age(&[ages, ages].concat()),
///     values: &[("paid", &[paid, paid.map(|v| v * 2.0)].concat())],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let boot = OdpBootstrap { n_sims: 1_000, seed: 1, ..Default::default() }
///     .fit_segments(&tri, "paid")?;
/// assert_eq!(boot.reserves.dims(), ["lob", "origin"]);
/// assert_eq!(boot.reserves.n_components(), 6);
/// let by_lob = boot.reserves.aggregate(&["lob"])?;
/// assert_eq!(by_lob.n_components(), 2);
/// assert_eq!(boot.totals().column("scale").unwrap().len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct OdpBootstrapFits {
    /// Each segment's chain ladder, fitted values, residuals and scale.
    pub segments: SegmentFits<OdpBootstrapSegment>,
    /// Joint distribution of the reserve by segment and origin: the key
    /// names and `origin` are its dimensions, and its components run over
    /// the origins of each segment in turn, like the rows of
    /// [`to_long`](Self::to_long).
    pub reserves: PredictiveDistribution,
}

impl OdpBootstrapFits {
    /// One row per segment × origin: the chain ladder's `latest`,
    /// `ultimate` and `reserve`, and the `mean` and `std_dev` of the
    /// bootstrapped reserve.
    pub fn to_long(&self) -> FitTable {
        let mut table = self.segments.to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        table
    }

    /// One row per segment: the chain ladder's totals, the `scale`, and the
    /// `mean` and `std_dev` of the segment's bootstrapped total reserve.
    pub fn totals(&self) -> FitTable {
        let mut table = self.segments.totals();
        let totals = segment_sums(&self.segments, &self.reserves);
        push_moments(&mut table, "", &totals);
        table
    }

    /// The chain ladders' development factors, one row per segment × age.
    pub fn development_table(&self) -> FitTable {
        self.segments.development_table()
    }

    /// The one segment chosen as in [`SegmentFits::position`], with its
    /// part of the joint reserves (same dimensions).
    pub fn segment(&self, keys: &[(&str, &str)]) -> Result<Self> {
        let (segments, reserves) = pick_segment(&self.segments, &self.reserves, keys)?;
        Ok(Self { segments, reserves })
    }
}

/// Positions of segment `s`'s components in a joint distribution whose
/// components run over the origins of each segment of `fits` in turn.
fn components_of<T: ReserveFit>(fits: &SegmentFits<T>, s: usize) -> std::ops::Range<usize> {
    let n_origins = |f: &T| f.chain_ladder().origins.len();
    let start: usize = fits.fits[..s].iter().map(n_origins).sum();
    start..start + n_origins(&fits.fits[s])
}

/// Per simulation, the sum of the components of `draws` in `range`.
fn sums(draws: &PredictiveDistribution, range: std::ops::Range<usize>) -> Sampled {
    let n = draws.n_components();
    let sums = draws
        .draw_matrix()
        .chunks_exact(n)
        .map(|row| row[range.clone()].iter().sum())
        .collect();
    Sampled::new(sums).expect("draws are finite and non-empty")
}

/// Each component of `draws` on its own, in component order.
pub(crate) fn component_sums(draws: &PredictiveDistribution) -> Vec<Sampled> {
    (0..draws.n_components())
        .map(|j| sums(draws, j..j + 1))
        .collect()
}

/// Each segment's total of `draws`, whose components run over the origins
/// of each segment of `fits` in turn.
pub(crate) fn segment_sums<T: ReserveFit>(
    fits: &SegmentFits<T>,
    draws: &PredictiveDistribution,
) -> Vec<Sampled> {
    (0..fits.len())
        .map(|s| sums(draws, components_of(fits, s)))
        .collect()
}

/// The one segment of `fits` chosen as in [`SegmentFits::position`], with
/// its part of the joint `draws` (same dimensions).
pub(crate) fn pick_segment<T: ReserveFit + Clone>(
    fits: &SegmentFits<T>,
    draws: &PredictiveDistribution,
    keys: &[(&str, &str)],
) -> Result<(SegmentFits<T>, PredictiveDistribution)> {
    let s = fits.position(keys)?;
    let range = components_of(fits, s);
    let n = draws.n_components();
    let part = draws
        .draw_matrix()
        .chunks_exact(n)
        .flat_map(|row| row[range.clone()].iter().copied())
        .collect();
    let part = PredictiveDistribution::from_draws(
        draws.dims().to_vec(),
        draws.components()[range].to_vec(),
        part,
        draws.provenance().clone(),
    )?;
    let one = SegmentFits {
        key_names: fits.key_names.clone(),
        labels: vec![fits.labels[s].clone()],
        fits: vec![fits.fits[s].clone()],
    };
    Ok((one, part))
}

/// Appends the `mean` and `std_dev` of each row's draws to `table`, their
/// names led by `prefix`.
pub(crate) fn push_moments(table: &mut FitTable, prefix: &str, draws: &[Sampled]) {
    table.values.push((
        format!("{prefix}mean"),
        draws.iter().map(|d| d.mean()).collect(),
    ));
    table.values.push((
        format!("{prefix}std_dev"),
        draws.iter().map(|d| d.std_dev()).collect(),
    ));
}

/// The key of each origin of a segment labelled `label`: the label's parts,
/// then the origin, as the components of a joint distribution over segments.
pub(crate) fn origin_keys(label: &Label, origins: &[Period]) -> Vec<Vec<KeyValue>> {
    origins
        .iter()
        .map(|&origin| {
            let mut key: Vec<KeyValue> = label.parts().iter().map(|p| p.as_str().into()).collect();
            key.push(origin.into());
            key
        })
        .collect()
}

impl OdpBootstrap {
    /// Bootstraps `column` of a single-segment cumulative triangle. Every
    /// origin must be observed at every age from the first up to its latest.
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<OdpBootstrapFit> {
        if self.n_sims == 0 {
            return Err(Error::Bootstrap("n_sims must be positive"));
        }
        let segment = triangle.segment(column)?;
        let ages = &segment.ages;
        let (fit, pool) = prepare(&segment, ages)?;

        let mut hasher = InputHasher::new();
        hasher.str(column);
        hash_segment(&mut hasher, &segment, &fit.chain_ladder, ages);
        let sim = Simulation {
            segment: &segment,
            latest: &fit.chain_ladder.latest_position,
            fitted: &fit.fitted,
            pool: &pool,
            scale: fit.scale,
            process: self.process,
        };
        let reserves = PredictiveDistribution::simulate(
            vec!["origin".into()],
            segment.origins.iter().map(|&p| vec![p.into()]).collect(),
            self.n_sims,
            self.seed,
            self.provenance(column, hasher),
            |rng, row| sim.run(rng, row),
        )?;
        let OdpBootstrapSegment {
            chain_ladder,
            fitted,
            residuals,
            scale,
        } = fit;
        Ok(OdpBootstrapFit {
            chain_ladder,
            fitted,
            residuals,
            scale,
            reserves,
        })
    }

    /// Bootstraps `column` in every segment of a cumulative triangle, each
    /// with its own residuals and scale, into one joint distribution of the
    /// reserves; see [`OdpBootstrapFits`]. A failure names its segment.
    pub fn fit_segments(&self, triangle: &Triangle, column: &str) -> Result<OdpBootstrapFits> {
        if self.n_sims == 0 {
            return Err(Error::Bootstrap("n_sims must be positive"));
        }
        let prepared = fit_each(triangle, column, |s| {
            let (fit, pool) = prepare(s, &s.ages)?;
            Ok((fit, pool, s.clone()))
        })?;

        let mut hasher = InputHasher::new();
        hasher.str(column);
        let mut dims = prepared.key_names.clone();
        dims.push("origin".into());
        let mut components = Vec::new();
        let mut sims = Vec::with_capacity(prepared.len());
        for (label, (fit, pool, segment)) in prepared.iter() {
            hasher.str(&label.to_string());
            hash_segment(&mut hasher, segment, &fit.chain_ladder, &segment.ages);
            components.extend(origin_keys(label, &segment.origins));
            sims.push(Simulation {
                segment,
                latest: &fit.chain_ladder.latest_position,
                fitted: &fit.fitted,
                pool,
                scale: fit.scale,
                process: self.process,
            });
        }
        let reserves = PredictiveDistribution::simulate(
            dims,
            components,
            self.n_sims,
            self.seed,
            self.provenance(column, hasher)
                .param("segments", prepared.len()),
            |rng, row| {
                let mut start = 0;
                for sim in &sims {
                    let end = start + sim.segment.n_origins;
                    sim.run(rng, &mut row[start..end]);
                    start = end;
                }
            },
        )?;
        Ok(OdpBootstrapFits {
            segments: prepared.map(|(fit, _, _)| fit.clone()),
            reserves,
        })
    }

    fn provenance(&self, column: &str, hasher: InputHasher) -> Provenance {
        Provenance::new("odp_bootstrap")
            .param("n_sims", self.n_sims)
            .param("process", format!("{:?}", self.process))
            .param("column", column)
            .version("act-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(hasher.finish())
    }
}

/// Hashes the observed cells of `segment` that the bootstrap uses.
pub(crate) fn hash_segment(
    hasher: &mut InputHasher,
    segment: &Segment,
    cl: &ChainLadderFit,
    ages: &[Lag],
) {
    for (o, &last) in cl.latest_position.iter().enumerate() {
        hasher.str(&segment.origins[o].to_string());
        for (d, &age) in ages[..=last].iter().enumerate() {
            hasher.i64(age as i64);
            hasher.f64s(&[segment.get(o, d).expect("checked when prepared")]);
        }
    }
}

/// The chain ladder, fitted values, residuals and scale of one segment,
/// and the pool of residuals to resample.
pub(crate) fn prepare(segment: &Segment, ages: &[Lag]) -> Result<(OdpBootstrapSegment, Vec<f64>)> {
    let (no, nd) = (segment.n_origins, segment.n_dev);
    let chain_ladder = ChainLadder::default().fit_segment(segment, ages)?;
    let ldf = &chain_ladder.development.ldf;
    let latest = &chain_ladder.latest_position;
    for (o, &last) in latest.iter().enumerate() {
        if (0..=last).any(|d| segment.get(o, d).is_none()) {
            return Err(Error::Bootstrap(
                "every origin must be observed from the first age to its latest",
            ));
        }
    }

    // Fitted cumulative values back from the latest diagonal, then
    // incrementals.
    let mut fitted = vec![f64::NAN; no * nd];
    let mut observed = vec![f64::NAN; no * nd];
    for o in 0..no {
        let mut cum = vec![0.0; latest[o] + 1];
        cum[latest[o]] = chain_ladder.latest[o];
        for d in (0..latest[o]).rev() {
            cum[d] = cum[d + 1] / ldf[d];
        }
        let mut previous_fit = 0.0;
        let mut previous_obs = 0.0;
        for d in 0..=latest[o] {
            let obs = segment.get(o, d).expect("checked above");
            fitted[o * nd + d] = cum[d] - previous_fit;
            observed[o * nd + d] = obs - previous_obs;
            previous_fit = cum[d];
            previous_obs = obs;
        }
    }

    let n_obs = fitted.iter().filter(|m| !m.is_nan()).count();
    let n_params = no + nd - 1;
    if n_obs <= n_params {
        return Err(Error::Bootstrap(
            "too few observed cells: degrees of freedom must be positive",
        ));
    }
    let unscaled: Vec<f64> = fitted
        .iter()
        .zip(&observed)
        .map(|(&m, &x)| {
            if m.is_nan() || m == 0.0 {
                f64::NAN
            } else {
                (x - m) / m.abs().sqrt()
            }
        })
        .collect();
    let dof = (n_obs - n_params) as f64;
    let scale = unscaled
        .iter()
        .filter(|r| !r.is_nan())
        .map(|r| r * r)
        .sum::<f64>()
        / dof;
    let adjust = (n_obs as f64 / dof).sqrt();
    let residuals: Vec<f64> = unscaled.iter().map(|r| r * adjust).collect();
    let pool: Vec<f64> = residuals.iter().copied().filter(|r| !r.is_nan()).collect();
    if pool.is_empty() {
        return Err(Error::Bootstrap("no residuals to resample"));
    }
    Ok((
        OdpBootstrapSegment {
            chain_ladder,
            fitted,
            residuals,
            scale,
        },
        pool,
    ))
}

/// Inputs shared by every simulation.
pub(crate) struct Simulation<'a> {
    pub(crate) segment: &'a Segment,
    pub(crate) latest: &'a [usize],
    pub(crate) fitted: &'a [f64],
    pub(crate) pool: &'a [f64],
    pub(crate) scale: f64,
    pub(crate) process: ProcessDistribution,
}

impl Simulation<'_> {
    /// One bootstrap replicate: fills `reserves` with each origin's
    /// simulated reserve.
    fn run(&self, rng: &mut StreamRng, reserves: &mut [f64]) {
        let nd = self.segment.n_dev;
        let (pseudo, factors) = self.resample(rng);
        for (o, reserve) in reserves.iter_mut().enumerate() {
            let mut cum = pseudo[o * nd + self.latest[o]];
            let mut total = 0.0;
            for f in &factors[self.latest[o]..] {
                let next = cum * f;
                total += self.with_process(next - cum, rng);
                cum = next;
            }
            *reserve = total;
        }
    }

    /// A pseudo cumulative triangle from resampled residuals, row-major
    /// over origin × development (zero where not observed), and its
    /// volume-weighted factors: the parameter error of one replicate.
    pub(crate) fn resample(&self, rng: &mut StreamRng) -> (Vec<f64>, Vec<f64>) {
        let (no, nd) = (self.segment.n_origins, self.segment.n_dev);

        // Pseudo cumulative triangle from resampled residuals.
        let mut pseudo = vec![0.0; no * nd];
        for o in 0..no {
            let mut cum = 0.0;
            for d in 0..=self.latest[o] {
                let m = self.fitted[o * nd + d];
                let k = ((rng.next_open01() * self.pool.len() as f64) as usize)
                    .min(self.pool.len() - 1);
                cum += m + self.pool[k] * m.abs().sqrt();
                pseudo[o * nd + d] = cum;
            }
        }

        // Volume-weighted factors of the pseudo triangle; 1 where the
        // earlier values sum to zero, as R does.
        let factors: Vec<f64> = (0..nd.saturating_sub(1))
            .map(|k| {
                let (mut num, mut den) = (0.0, 0.0);
                for o in (0..no).filter(|&o| self.latest[o] > k) {
                    num += pseudo[o * nd + k + 1];
                    den += pseudo[o * nd + k];
                }
                if den == 0.0 { 1.0 } else { num / den }
            })
            .collect();
        (pseudo, factors)
    }

    /// An incremental value with expected value `mean` and the process
    /// error of [`ProcessDistribution`].
    pub(crate) fn with_process(&self, mean: f64, rng: &mut StreamRng) -> f64 {
        match self.process {
            ProcessDistribution::None => mean,
            ProcessDistribution::Gamma => {
                if mean == 0.0 || self.scale == 0.0 {
                    return mean;
                }
                let gamma = Gamma::new(mean.abs() / self.scale, self.scale)
                    .expect("shape and scale are finite and positive");
                mean.signum() * gamma.sample(rng, 1)[0]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::{annual, raa};
    use act_core::Period;

    fn boot(process: ProcessDistribution, n_sims: usize) -> OdpBootstrapFit {
        OdpBootstrap {
            n_sims,
            seed: 7,
            process,
        }
        .fit(&raa(), "values")
        .unwrap()
    }

    #[test]
    fn raa_residuals_and_scale() {
        // R ChainLadder BootChainLadder(RAA): sum of squared unscaled
        // residuals / 36 degrees of freedom.
        let fit = boot(ProcessDistribution::None, 10);
        assert_eq!(fit.residuals.iter().filter(|r| !r.is_nan()).count(), 55);
        // The fitted values reproduce each origin's latest cumulative.
        for o in 0..10 {
            let sum: f64 = fit.fitted[o * 10..o * 10 + 10 - o].iter().sum();
            assert!((sum - fit.chain_ladder.latest[o]).abs() < 1e-6);
        }
        // Corners fit exactly: the oldest origin's last cell, the newest's
        // first.
        assert!(fit.residuals[9].abs() < 1e-9);
        assert!(fit.residuals[90].abs() < 1e-9);
        assert!(fit.scale > 0.0);
    }

    #[test]
    fn parameter_only_mean_is_near_chain_ladder() {
        let fit = boot(ProcessDistribution::None, 4_000);
        let cl = fit.chain_ladder.total_reserve();
        let mean = fit.reserves.total().mean();
        assert!((mean / cl - 1.0).abs() < 0.05, "{mean} vs {cl}");
        assert_eq!(
            fit.reserves
                .marginal(&vec![Period::year(1981).into()])
                .unwrap()
                .mean(),
            0.0
        );
    }

    #[test]
    fn reproducible_by_seed() {
        let a = boot(ProcessDistribution::Gamma, 300);
        let b = boot(ProcessDistribution::Gamma, 300);
        assert_eq!(a.reserves.draw_matrix(), b.reserves.draw_matrix());
        let c = OdpBootstrap {
            n_sims: 300,
            seed: 8,
            process: ProcessDistribution::Gamma,
        }
        .fit(&raa(), "values")
        .unwrap();
        assert_ne!(a.reserves.draw_matrix(), c.reserves.draw_matrix());
        assert_eq!(a.reserves.provenance().model, "odp_bootstrap");
    }

    #[test]
    fn process_error_widens_the_distribution() {
        let param = boot(ProcessDistribution::None, 2_000);
        let full = boot(ProcessDistribution::Gamma, 2_000);
        assert!(full.reserves.total().std_dev() > param.reserves.total().std_dev());
    }

    #[test]
    fn rejects_holes_and_tiny_triangles() {
        use crate::{DevelopmentColumn, Grain, Long, Month};
        let origin = [2019, 2019, 2019, 2020, 2020, 2021].map(Month::january);
        let holes = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 36, 12, 36, 12]),
            values: &[("paid", &[1.0, 2.0, 3.0, 1.0, 3.0, 1.0])],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let b = OdpBootstrap::default();
        assert!(matches!(b.fit(&holes, "paid"), Err(Error::Bootstrap(_))));
        let tiny = annual(2020, &[&[1.0, 2.0], &[1.0]]);
        assert!(matches!(b.fit(&tiny, "values"), Err(Error::Bootstrap(_))));
        let none = OdpBootstrap {
            n_sims: 0,
            ..Default::default()
        };
        assert!(matches!(
            none.fit(&raa(), "values"),
            Err(Error::Bootstrap(_))
        ));
    }
}
