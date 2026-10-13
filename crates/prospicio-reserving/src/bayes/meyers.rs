//! Meyers' correlated chain ladder (CRC), changing settlement rate (CSR) and
//! correlated accident year (CAY) models on cumulative values with premium
//! (`docs/design/reserving-bayes.md`, decision 7).

use prospicio_bayes::elpd::Loo;
use prospicio_bayes::nuts::{ParamSummary, PosteriorDraws, Sampler, sample};
use prospicio_core::StreamRng;
use prospicio_math::linalg::{cholesky, cholesky_solve};
use prospicio_math::special::norm_quantile;
use prospicio_prob::{InputHasher, PredictiveDistribution, Provenance};

use super::laplace::{Posterior, Whitened};
use super::{Diagnostics, Stream, derive_seed, pilot_seed};
use crate::bootstrap::{component_sums, hash_segment, push_moments, segment_sums};
use crate::chain_ladder::{ChainLadder, ChainLadderFit};
use crate::error::{Error, Result};
use crate::expected_loss::origin_exposure;
use crate::segments::{FitTable, ReserveFit, SegmentFits};
use crate::triangle::{Label, Segment, Triangle};

/// Which of Meyers' models [`Meyers`] fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeyersModel {
    /// The correlated chain ladder on cumulative values (Meyers 2019,
    /// sec. 5), without its correlation between origins: CRC.
    #[default]
    Crc,
    /// The changing settlement rate model (Meyers 2019, sec. 7): CRC with
    /// each development effect scaled by `(1 - gamma)^(w - 1)`.
    Csr,
    /// The correlated accident year model (Meyers 2019, sec. 8): CRC with
    /// `rho` times the previous origin's residual at the same age added to
    /// each mean, which makes neighbouring origins correlated. Meyers fits
    /// it to incurred data.
    Cay,
}

impl MeyersModel {
    /// `crc`, `csr` or `cay`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Crc => "crc",
            Self::Csr => "csr",
            Self::Cay => "cay",
        }
    }
}

/// Meyers' (2019, CAS Monograph 8) log-normal models of cumulative values
/// `C_wd` with premium `P_w`, for origins `w = 1..n` and ages `d = 1..m`:
///
/// ```text
/// mu_wd = log P_w + logelr + alpha_w + beta_d g_w + rho r_(w-1)d,   alpha_1 = 0,  beta_m = 0
/// g_w   = 1 (CRC, CAY)  or  (1 - gamma)^(w - 1) (CSR)
/// r_wd  = log C_wd - mu_wd,   rho = 0 (CRC, CSR),   r_0d = 0
/// sigma_d^2 = s_d + s_(d+1) + ... + s_m,   s_i ~ U(1 - exp(-1e-5), 1)
/// log C_wd ~ N(mu_wd, sigma_d^2)
/// logelr ~ N(-0.4, 3.162^2),  alpha_w, beta_d ~ N(0, 3.162^2),  gamma ~ N(0, 0.05^2)
/// (1 - rho) / 2 ~ Beta(2, 2)
/// ```
///
/// with the priors of Meyers' Stan code (`CRC.R`, `CSR.R` and `CAY.R` of
/// the CAS appendix; 3.162 is his rounding of `sqrt(10)`). In CAY each
/// mean takes `rho` times the previous origin's residual at the same age,
/// itself a residual from its own (recursive) mean, so `log C_wd` is a
/// moving average of order 1 down each age's column: the correlation of
/// neighbouring origins is `rho / (1 + rho^2)`, of origins further apart 0
/// (Meyers 2019, eq. 8.3). He draws
/// `a_i ~ InvGamma(1, 1)` bounded by `1e5` and sets
/// `s_i = 1 - exp(-1 / a_i)`, which is uniform on `(1 - exp(-1e-5), 1)`:
/// the floor of about `1e-5` keeps each `sigma_d` at least
/// `sqrt((m - d + 1) 1e-5)` and matters, since without it the late
/// `sigma_d` fall about 15% lower on incurred data. The ultimate is the
/// value at the oldest age `m` (`beta_m = 0`; no tail): given a posterior
/// draw, each origin's ultimate is a log-normal draw with mean
/// `log P_w + logelr + alpha_w` (CAY: plus `rho` times the previous
/// origin's residual at age `m`, observed or drawn, origin by origin) and
/// sd `sigma_m`, and an origin already observed at age `m` keeps its
/// observed value, as Meyers computes it. Ages are independent given the
/// parameters (in CRC and CSR every cell is), so the ultimate is drawn
/// from age `m` alone, and the reserve (ultimate less the latest value of
/// the fitted column: an outstanding reserve on paid data, IBNR on
/// incurred data) can be negative.
///
/// The model samples `s_i` through a logit,
/// `s_i = s0 + (1 - s0) / (1 + exp(-u_i))` with `s0 = 1 - exp(-1e-5)`, and
/// the log Jacobian so that the prior on `s_i` is exactly Meyers'; the
/// posterior is the same as his. CAY samples `rho = tanh(v)`, with the log
/// Jacobian, so that `(1 - rho) / 2` has Meyers' `Beta(2, 2)` prior. NUTS
/// samples a dense whitened scale: the
/// joint mode lies deep in the funnel of the late `sigma_d`, far from the
/// posterior's bulk, so the scale is not a Laplace approximation (as for
/// [`BayesOdp`](super::BayesOdp)) but the mean and covariance of a pilot
/// run of `tune` warm-up and `max(tune, 100)` kept draws per chain, itself
/// on a scale whitened at a least-squares start (as Stan's dense metric
/// adapts). The pilot's draws are discarded.
///
/// Every observed cell must be positive (Meyers' code silently raises
/// values below 1 to 1; here a non-positive cell is an error). CAY also
/// needs, for each observed cell after the first origin, the previous
/// origin's cell at the same age.
///
/// ```
/// use prospicio_reserving::bayes::{Meyers, MeyersModel};
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// use prospicio_bayes::nuts::Sampler;
/// use prospicio_prob::Distribution;
///
/// // Six origins of cumulative paid, each with its premium.
/// let paid: [&[f64]; 6] = [
///     &[952.0, 1529.0, 2813.0, 3647.0, 3724.0, 3832.0],
///     &[849.0, 1564.0, 2202.0, 2432.0, 2468.0],
///     &[983.0, 2211.0, 2830.0, 3832.0],
///     &[1657.0, 2685.0, 3169.0],
///     &[932.0, 1940.0],
///     &[1162.0],
/// ];
/// let premium = [5812.0, 4908.0, 5454.0, 5165.0, 5214.0, 5230.0];
/// let (mut origin, mut age, mut values, mut prem) = (vec![], vec![], vec![], vec![]);
/// for (w, row) in paid.iter().enumerate() {
///     for (d, &v) in row.iter().enumerate() {
///         origin.push(Month::january(1988 + w as i32));
///         age.push(12 * (d as u32 + 1));
///         values.push(v);
///         prem.push(premium[w]);
///     }
/// }
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&age),
///     values: &[("paid", &values), ("premium", &prem)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let model = Meyers {
///     model: MeyersModel::Csr,
///     sampler: Sampler { tune: 300, draws: 300, ..Meyers::default().sampler },
///     n_sims: 2_000,
///     seed: 1,
///     ..Meyers::default()
/// };
/// let fit = model.fit(&tri, "paid", "premium")?;
/// assert_eq!(fit.reserves.n_components(), 6);
/// assert!(fit.ultimates.total().mean() > fit.chain_ladder.latest.iter().sum::<f64>());
/// assert_eq!(fit.posterior.names()[0], "logelr");
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Meyers {
    /// CRC, CSR or CAY.
    pub model: MeyersModel,
    /// Mean of the normal prior on `logelr`, the log expected loss ratio.
    pub logelr_mean: f64,
    /// Standard deviation of the normal prior on `logelr`.
    pub logelr_sd: f64,
    /// Standard deviation of the normal prior on each origin effect
    /// `alpha_w`, `w >= 2`, centred on 0.
    pub origin_sd: f64,
    /// Standard deviation of the normal prior on each development effect
    /// `beta_d`, `d < m`, centred on 0.
    pub development_sd: f64,
    /// Standard deviation of the normal prior on `gamma`, centred on 0
    /// (CSR only).
    pub gamma_sd: f64,
    /// Both shape parameters of the symmetric beta prior on
    /// `(1 - rho) / 2` (CAY only; Meyers' 2 makes the prior density of
    /// `rho` proportional to `1 - rho^2`).
    pub rho_shape: f64,
    /// NUTS settings. Its `seed` is not used: the chains take a seed
    /// derived from [`seed`](Self::seed) (module docs, "Seeds").
    pub sampler: Sampler,
    /// Number of predictive simulations, independent of the number of
    /// posterior draws.
    pub n_sims: usize,
    /// Seed of the chains and of the simulations, each through its own
    /// derived seed (module docs, "Seeds").
    pub seed: u64,
}

impl Default for Meyers {
    /// CRC with Meyers' priors (`logelr ~ N(-0.4, 3.162^2)`, origin and
    /// development effects `N(0, 3.162^2)`, `gamma ~ N(0, 0.05^2)`,
    /// `(1 - rho) / 2 ~ Beta(2, 2)`), four
    /// chains of 1,000 warm-up and 2,000 kept draws with target acceptance
    /// 0.95, 10,000 simulations, seed 0. On Meyers' group 353 data, at
    /// target acceptance 0.8 the paid fits gave 1 to 56 divergent
    /// transitions at nearly every one of 30 seeds (the late `sigma_d`,
    /// near 0.02, make a strongly curved posterior), and with 1,000 kept
    /// draws the late `sigma_d` had an R-hat above 1.01 (at most 1.016) in
    /// 5 of 120 fits; with these defaults all 120 pass every threshold.
    fn default() -> Self {
        Self {
            model: MeyersModel::Crc,
            logelr_mean: -0.4,
            logelr_sd: 3.162,
            origin_sd: 3.162,
            development_sd: 3.162,
            gamma_sd: 0.05,
            rho_shape: 2.0,
            sampler: Sampler {
                target_accept: 0.95,
                draws: 2000,
                ..Sampler::default()
            },
            n_sims: 10_000,
            seed: 0,
        }
    }
}

/// Fewest kept draws per chain of the pilot run.
const MIN_PILOT_DRAWS: usize = 100;

/// A fitted [`Meyers`].
#[derive(Debug, Clone)]
pub struct MeyersFit {
    /// The volume-weighted chain ladder of the fitted column: latest values
    /// and the long tables' `ultimate` and `reserve` come from it.
    pub chain_ladder: ChainLadderFit,
    /// The premium of each origin, the latest value of the premium column.
    pub premium: Vec<f64>,
    /// Posterior draws of `logelr`, `alpha[2]..alpha[n]`,
    /// `beta[1]..beta[m-1]`, `gamma` (CSR only) or `rho` (CAY only) and
    /// `sigma[1]..sigma[m]`,
    /// by chain.
    pub posterior: PosteriorDraws,
    /// R-hat, ESS and divergences, with warnings for failed thresholds.
    pub diagnostics: Diagnostics,
    /// Joint posterior predictive distribution of the ultimate (the value
    /// at the oldest age) by origin: dimension `origin`, one component per
    /// origin; an origin observed at the oldest age keeps its value.
    pub ultimates: PredictiveDistribution,
    /// The ultimates less the latest values, simulation by simulation:
    /// the same layout as [`OdpBootstrapFit::reserves`](crate::OdpBootstrapFit::reserves).
    pub reserves: PredictiveDistribution,
    /// The model, for the pointwise log likelihood of [`loo`](Self::loo).
    model: Model,
}

impl Meyers {
    /// Fits `column` of a single-segment cumulative triangle with premium
    /// from its `premium` column (the latest value per origin, as
    /// [`CapeCod`](crate::CapeCod) reads exposure). Every origin must be
    /// observed from the first age to its latest, every observed cell must
    /// be positive, and there must be more observed cells than parameters.
    /// Failed diagnostics do not fail the fit: they are in
    /// [`MeyersFit::diagnostics`].
    pub fn fit(&self, triangle: &Triangle, column: &str, premium: &str) -> Result<MeyersFit> {
        self.validate()?;
        let segment = triangle.segment(column)?;
        let premium_segment = triangle.segment(premium)?;
        let chain_ladder = ChainLadder::default().fit_segment(&segment, &segment.ages)?;
        let premiums = origin_exposure(&segment, &premium_segment, premium)?;
        let model = self.model_of(&segment, &chain_ladder, &premiums)?;
        let p = model.dim();
        // The joint mode lies deep in the funnel of the late sigma_d, far
        // from the posterior's bulk, so the scale comes from a pilot run
        // (decision 7, "Sampling scale"): whitened roughly at the start
        // point, then by the pilot draws' mean and covariance.
        let start = model.start();
        let rough = Whitened::regularised(&model, &start)?;
        let pilot = sample(
            &rough,
            &vec![0.0; p],
            Sampler {
                seed: pilot_seed(self.seed),
                draws: self.sampler.tune.max(MIN_PILOT_DRAWS),
                ..self.sampler
            },
        )?;
        let pilot = pilot.transform(model.names(), |z| rough.theta(z))?;
        let whitened = Whitened::from_draws(&model, &pilot)?;
        let draws = sample(
            &whitened,
            &vec![0.0; p],
            Sampler {
                seed: derive_seed(self.seed, Stream::Chains),
                ..self.sampler
            },
        )?;
        let posterior = draws.transform(model.names(), |z| model.natural(&whitened.theta(z)))?;
        let summary = posterior.summary()?;
        let diagnostics = Diagnostics::new(
            &summary,
            posterior.chains(),
            posterior.divergences().iter().sum(),
        );

        let mut hasher = InputHasher::new();
        hasher.str(column);
        hasher.str(premium);
        hash_segment(&mut hasher, &segment, &chain_ladder, &segment.ages);
        hasher.f64s(&premiums);
        let provenance = self.provenance(column, premium, hasher);
        let nd = segment.n_dev;
        let closed: Vec<bool> = chain_ladder
            .latest_position
            .iter()
            .map(|&d| d + 1 == nd)
            .collect();
        let latest = &chain_ladder.latest;
        let no = segment.n_origins;
        let ultimates = posterior.predictive(
            vec!["origin".into()],
            segment.origins.iter().map(|&o| vec![o.into()]).collect(),
            self.n_sims,
            derive_seed(self.seed, Stream::Predictive),
            provenance.clone().param("output", "ultimates"),
            |nat, rng: &mut StreamRng, row| {
                model.draw_ultimates(nat, &premiums, &closed, latest, rng, row);
            },
        )?;
        let reserve_draws: Vec<f64> = ultimates
            .draw_matrix()
            .chunks_exact(no)
            .flat_map(|row| row.iter().zip(latest).map(|(u, l)| u - l))
            .collect();
        // The ultimates' provenance (seed, posterior draws) with the output
        // renamed.
        let mut provenance = ultimates.provenance().clone();
        for (key, value) in &mut provenance.parameters {
            if key == "output" {
                *value = "reserves".into();
            }
        }
        let reserves = PredictiveDistribution::from_draws(
            ultimates.dims().to_vec(),
            ultimates.components().to_vec(),
            reserve_draws,
            provenance,
        )?;
        Ok(MeyersFit {
            chain_ladder,
            premium: premiums,
            posterior,
            diagnostics,
            ultimates,
            reserves,
            model,
        })
    }

    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("logelr_sd", self.logelr_sd),
            ("origin_sd", self.origin_sd),
            ("development_sd", self.development_sd),
            ("gamma_sd", self.gamma_sd),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::InvalidSetting {
                    name,
                    value,
                    expected: "a finite, positive prior standard deviation",
                });
            }
        }
        if !(self.rho_shape.is_finite() && self.rho_shape > 0.0) {
            return Err(Error::InvalidSetting {
                name: "rho_shape",
                value: self.rho_shape,
                expected: "a finite, positive beta shape",
            });
        }
        if !self.logelr_mean.is_finite() {
            return Err(Error::InvalidSetting {
                name: "logelr_mean",
                value: self.logelr_mean,
                expected: "a finite prior mean",
            });
        }
        if self.n_sims == 0 {
            return Err(Error::InvalidSetting {
                name: "n_sims",
                value: 0.0,
                expected: "a positive number of simulations",
            });
        }
        Ok(())
    }

    /// The log posterior of the observed cells of `segment`.
    fn model_of(
        &self,
        segment: &Segment,
        chain_ladder: &ChainLadderFit,
        premiums: &[f64],
    ) -> Result<Model> {
        let mut cells = Vec::new();
        for (w, &last) in chain_ladder.latest_position.iter().enumerate() {
            for d in 0..=last {
                let value = segment
                    .get(w, d)
                    .expect("the chain ladder checked the gaps");
                if !(value.is_finite() && value > 0.0) {
                    return Err(Error::Bayes(format!(
                        "origin {} at development age {} months is {value}; the log-normal \
                         model needs positive cumulative values",
                        segment.origins[w], segment.ages[d]
                    )));
                }
                cells.push(Cell {
                    w,
                    d,
                    y: value.ln(),
                    offset: premiums[w].ln(),
                    prev: None,
                });
            }
        }
        if self.model == MeyersModel::Cay {
            // Each cell's mean takes the previous origin's residual at the
            // same age; cells run origin by origin, so it comes first.
            let mut at = vec![vec![None; segment.n_dev]; segment.n_origins];
            for (i, c) in cells.iter().enumerate() {
                at[c.w][c.d] = Some(i);
            }
            for c in &mut cells {
                if c.w == 0 {
                    continue;
                }
                c.prev = at[c.w - 1][c.d];
                if c.prev.is_none() {
                    return Err(Error::Bayes(format!(
                        "origin {} is observed at development age {} months but origin {} is \
                         not; CAY needs the previous origin's cell at each age",
                        segment.origins[c.w],
                        segment.ages[c.d],
                        segment.origins[c.w - 1]
                    )));
                }
            }
        }
        let model = Model {
            n_origins: segment.n_origins,
            n_dev: segment.n_dev,
            kind: self.model,
            cells,
            logelr_mean: self.logelr_mean,
            logelr_sd: self.logelr_sd,
            origin_sd: self.origin_sd,
            development_sd: self.development_sd,
            gamma_sd: self.gamma_sd,
            rho_shape: self.rho_shape,
            priors: true,
        };
        if model.cells.len() <= model.dim() {
            return Err(Error::Bayes(format!(
                "{} observed cells for {} parameters; the model needs more cells than \
                 parameters",
                model.cells.len(),
                model.dim()
            )));
        }
        Ok(model)
    }

    fn provenance(&self, column: &str, premium: &str, hasher: InputHasher) -> Provenance {
        let s = &self.sampler;
        let mut provenance = Provenance::new(format!("meyers_{}", self.model.name()))
            .param("logelr_mean", self.logelr_mean)
            .param("logelr_sd", self.logelr_sd)
            .param("origin_sd", self.origin_sd)
            .param("development_sd", self.development_sd);
        match self.model {
            MeyersModel::Crc => {}
            MeyersModel::Csr => provenance = provenance.param("gamma_sd", self.gamma_sd),
            MeyersModel::Cay => provenance = provenance.param("rho_shape", self.rho_shape),
        }
        provenance
            .param("chains", s.chains)
            .param("tune", s.tune)
            .param("draws", s.draws)
            .param("target_accept", s.target_accept)
            .param("max_depth", s.max_depth)
            .param("n_sims", self.n_sims)
            .param("seed", self.seed)
            .param("column", column)
            .param("premium", premium)
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(hasher.finish())
    }
}

impl MeyersFit {
    /// The model that produced this fit, CRC, CSR or CAY.
    pub fn model(&self) -> MeyersModel {
        self.model.kind
    }

    /// Posterior summary of each parameter: mean, sd, 5/50/95%
    /// quantiles, R-hat, bulk and tail ESS.
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        Ok(self.posterior.summary()?)
    }

    /// PSIS-LOO of the observed cells, with each cell's relative
    /// efficiency from the chains. The pointwise term is Meyers', the
    /// normal density of `log C_wd` (`normal_lpdf` in his Stan code), so
    /// the result is on the log scale and comparable only between CRC, CSR
    /// and CAY fits of the same column (Meyers 2019, Tables 6.4, 7.3 and
    /// 8.3). In CAY a cell's mean holds the previous origin's observed
    /// residual, so its term is conditional on that cell, as in Meyers'
    /// code.
    pub fn loo(&self) -> Result<Loo> {
        let model = &self.model;
        let n = model.cells.len();
        let s = self.posterior.n_draws();
        let mut ll = Vec::with_capacity(s * n);
        let mut residuals = Vec::with_capacity(n);
        for nat in self.posterior.values().chunks_exact(self.posterior.dim()) {
            model.log_densities(nat, &mut residuals, &mut ll);
        }
        let chains = self.posterior.chains();
        let per_chain = s / chains;
        let r_eff: Vec<f64> = (0..n)
            .map(|i| {
                let by_chain: Vec<Vec<f64>> = (0..chains)
                    .map(|c| {
                        (0..per_chain)
                            .map(|k| ll[(c * per_chain + k) * n + i].exp())
                            .collect()
                    })
                    .collect();
                let refs: Vec<&[f64]> = by_chain.iter().map(Vec::as_slice).collect();
                let ess = prospicio_bayes::ess_mean(&refs).unwrap_or(s as f64);
                // A constant likelihood: as independent draws.
                if ess.is_finite() && ess > 0.0 {
                    ess / s as f64
                } else {
                    1.0
                }
            })
            .collect();
        Ok(prospicio_bayes::elpd::loo(&ll, n, Some(&r_eff))?)
    }

    /// One row per origin: the chain ladder's `latest`, `ultimate` and
    /// `reserve`, the `premium`, the `mean` and `std_dev` of the posterior
    /// predictive reserve, and the `ultimate_mean` and `ultimate_std_dev`
    /// of the predictive ultimate.
    pub fn to_long(&self) -> FitTable {
        let mut table = self.plug_in().to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        push_moments(&mut table, "ultimate_", &component_sums(&self.ultimates));
        table
    }

    /// One row: the chain ladder's totals, the total `premium`, the `mean`
    /// and `std_dev` of the posterior predictive total reserve, and the
    /// `ultimate_mean` and `ultimate_std_dev` of the total ultimate.
    pub fn totals(&self) -> FitTable {
        let fits = self.plug_in();
        let mut table = fits.totals();
        push_moments(&mut table, "", &segment_sums(&fits, &self.reserves));
        push_moments(
            &mut table,
            "ultimate_",
            &segment_sums(&fits, &self.ultimates),
        );
        table
    }

    /// The chain ladder's development factors, one row per age.
    pub fn development_table(&self) -> FitTable {
        self.plug_in().development_table()
    }

    /// The fit as the single segment of the long tables.
    fn plug_in(&self) -> SegmentFits<PlugIn> {
        SegmentFits {
            key_names: Vec::new(),
            labels: vec![Label::new(Vec::<String>::new())],
            fits: vec![PlugIn {
                chain_ladder: self.chain_ladder.clone(),
                premium: self.premium.clone(),
            }],
        }
    }
}

impl ReserveFit for MeyersFit {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        vec![("premium", self.premium.clone())]
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("premium", self.premium.iter().sum())]
    }
}

/// The chain ladder and premium of a fit, for its long tables.
#[derive(Debug, Clone)]
struct PlugIn {
    chain_ladder: ChainLadderFit,
    premium: Vec<f64>,
}

impl ReserveFit for PlugIn {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        vec![("premium", self.premium.clone())]
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        vec![("premium", self.premium.iter().sum())]
    }
}

/// An observed cell: origin and development positions, `log C_wd` and
/// `log P_w`, and (CAY only, after the first origin) the index of the
/// previous origin's cell at the same age, which comes earlier.
#[derive(Debug, Clone, Copy)]
struct Cell {
    w: usize,
    d: usize,
    y: f64,
    offset: f64,
    prev: Option<usize>,
}

/// Smallest step `s_i` of `sigma_d^2`, `1 - exp(-1e-5)`: Meyers bounds
/// `a_ig` above by `1e5` in his Stan code, and `s_i = 1 - exp(-1 / a_ig)`.
pub(crate) const S_FLOOR: f64 = 9.999_950_000_166_667e-6;

/// The log posterior on the unconstrained
/// `theta = (logelr, alpha_2..alpha_n, beta_1..beta_(m-1), [gamma | v], u_1..u_m)`
/// with `s_i = S_FLOOR + (1 - S_FLOOR) logistic(u_i)` and (CAY)
/// `rho = tanh(v)`.
#[derive(Debug, Clone)]
struct Model {
    n_origins: usize,
    n_dev: usize,
    kind: MeyersModel,
    cells: Vec<Cell>,
    logelr_mean: f64,
    logelr_sd: f64,
    origin_sd: f64,
    development_sd: f64,
    gamma_sd: f64,
    rho_shape: f64,
    /// Without priors, the likelihood alone (for the tests).
    priors: bool,
}

/// `log(1 + exp(x))` without overflow.
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// `1 / (1 + exp(-u))` without overflow.
fn logistic(u: f64) -> f64 {
    if u >= 0.0 {
        1.0 / (1.0 + (-u).exp())
    } else {
        let e = u.exp();
        e / (1.0 + e)
    }
}

impl Model {
    fn csr(&self) -> bool {
        self.kind == MeyersModel::Csr
    }

    fn cay(&self) -> bool {
        self.kind == MeyersModel::Cay
    }

    /// Position of `beta_1` in `theta`.
    fn beta0(&self) -> usize {
        self.n_origins
    }

    /// Position of `gamma` (CSR), of `v` or `rho` (CAY) or of `u_1` (CRC).
    fn after_beta(&self) -> usize {
        self.n_origins + self.n_dev - 1
    }

    /// Position of `u_1`.
    fn u0(&self) -> usize {
        self.after_beta() + usize::from(self.kind != MeyersModel::Crc)
    }

    fn names(&self) -> Vec<String> {
        let mut names = vec!["logelr".to_string()];
        names.extend((2..=self.n_origins).map(|w| format!("alpha[{w}]")));
        names.extend((1..self.n_dev).map(|d| format!("beta[{d}]")));
        match self.kind {
            MeyersModel::Crc => {}
            MeyersModel::Csr => names.push("gamma".into()),
            MeyersModel::Cay => names.push("rho".into()),
        }
        names.extend((1..=self.n_dev).map(|d| format!("sigma[{d}]")));
        names
    }

    /// `sigma_d^2` for each age from `theta`'s `u`, and each
    /// `l_i = logistic(u_i)` and `1 - l_i`, with
    /// `s_i = S_FLOOR + (1 - S_FLOOR) l_i`.
    fn variances(&self, theta: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let m = self.n_dev;
        let u = &theta[self.u0()..self.u0() + m];
        let l: Vec<f64> = u.iter().map(|&u| logistic(u)).collect();
        let lc: Vec<f64> = u.iter().map(|&u| logistic(-u)).collect();
        let mut var = vec![0.0; m];
        let mut acc = 0.0;
        for d in (0..m).rev() {
            acc += S_FLOOR + (1.0 - S_FLOOR) * l[d];
            var[d] = acc;
        }
        (var, l, lc)
    }

    /// `theta` on the natural scale: `logelr`, `alpha`, `beta`, `gamma`
    /// or `rho = tanh(v)`, and `sigma_d = sqrt(sigma_d^2)`.
    fn natural(&self, theta: &[f64]) -> Vec<f64> {
        let (var, _, _) = self.variances(theta);
        let mut out = theta[..self.u0()].to_vec();
        if self.cay() {
            let k = self.after_beta();
            out[k] = out[k].tanh();
        }
        out.extend(var.iter().map(|v| v.sqrt()));
        out
    }

    /// `g_w` and its derivative in `gamma`, for 0-based origin `w`.
    fn speedup(&self, gamma: f64, w: usize) -> (f64, f64) {
        if !self.csr() || w == 0 {
            return (1.0, 0.0);
        }
        let k = w as i32;
        let base = 1.0 - gamma;
        (base.powi(k), -(w as f64) * base.powi(k - 1))
    }

    /// `mu_wd` without CAY's `rho` term, from natural-scale or
    /// unconstrained `theta` (both hold `logelr`, `alpha`, `beta` and
    /// `gamma` at the same positions).
    fn mu(&self, theta: &[f64], cell: &Cell) -> f64 {
        let alpha = if cell.w > 0 { theta[cell.w] } else { 0.0 };
        let beta = if cell.d + 1 < self.n_dev {
            theta[self.beta0() + cell.d]
        } else {
            0.0
        };
        let gamma = if self.csr() {
            theta[self.after_beta()]
        } else {
            0.0
        };
        cell.offset + theta[0] + alpha + beta * self.speedup(gamma, cell.w).0
    }

    /// The residual `log C_wd - mu_wd` of each cell, in `residuals`, with
    /// CAY's `rho` (0 otherwise): each cell's mean takes `rho` times the
    /// previous origin's residual, which comes earlier.
    fn residuals(&self, theta: &[f64], rho: f64, residuals: &mut Vec<f64>) {
        residuals.clear();
        for c in &self.cells {
            let mut r = c.y - self.mu(theta, c);
            if let Some(p) = c.prev {
                r -= rho * residuals[p];
            }
            residuals.push(r);
        }
    }

    /// Meyers' pointwise log likelihood, `normal_lpdf(log C | mu, sigma_d)`,
    /// of every cell from a natural-scale draw, appended to `out`
    /// (`residuals` is scratch).
    fn log_densities(&self, nat: &[f64], residuals: &mut Vec<f64>, out: &mut Vec<f64>) {
        let rho = if self.cay() {
            nat[self.after_beta()]
        } else {
            0.0
        };
        self.residuals(nat, rho, residuals);
        for (c, r) in self.cells.iter().zip(residuals.iter()) {
            let sigma = nat[self.u0() + c.d];
            let z = r / sigma;
            out.push(-0.5 * (std::f64::consts::TAU.ln() + z * z) - sigma.ln());
        }
    }

    /// One simulation of the ultimate (the value at age `m`) of each origin
    /// into `row`, given a natural-scale draw: a closed origin keeps its
    /// latest value, any other is log-normal with mean
    /// `log P_w + logelr + alpha_w` (`beta_m = 0`, so neither beta nor
    /// gamma enters) and sd `sigma_m`. In CAY each mean also takes `rho`
    /// times the previous origin's residual at age `m`, observed for a
    /// closed origin and drawn otherwise, origin by origin (Meyers 2019,
    /// sec. 8).
    fn draw_ultimates(
        &self,
        nat: &[f64],
        premiums: &[f64],
        closed: &[bool],
        latest: &[f64],
        rng: &mut StreamRng,
        row: &mut [f64],
    ) {
        // The natural-scale draw ends with sigma[m].
        let sigma_m = nat[self.dim() - 1];
        if !self.cay() {
            for (w, ultimate) in row.iter_mut().enumerate() {
                *ultimate = if closed[w] {
                    latest[w]
                } else {
                    let alpha = if w > 0 { nat[w] } else { 0.0 };
                    let z = norm_quantile(rng.next_open01());
                    (premiums[w].ln() + nat[0] + alpha + sigma_m * z).exp()
                };
            }
            return;
        }
        let rho = nat[self.after_beta()];
        let mut residual = 0.0;
        for (w, ultimate) in row.iter_mut().enumerate() {
            let alpha = if w > 0 { nat[w] } else { 0.0 };
            let mean = premiums[w].ln() + nat[0] + alpha + rho * residual;
            if closed[w] {
                *ultimate = latest[w];
                residual = latest[w].ln() - mean;
            } else {
                residual = sigma_m * norm_quantile(rng.next_open01());
                *ultimate = (mean + residual).exp();
            }
        }
    }

    /// The Newton start: least squares of `log C - log P` on `logelr`,
    /// `alpha` and `beta` (CRC, `gamma = 0`), then each age's mean squared
    /// residual made non-increasing from the oldest age, its steps as
    /// `s_i` within `[0.001, 0.5]`.
    fn start(&self) -> Vec<f64> {
        let k = self.after_beta();
        let (n, m) = (self.n_origins, self.n_dev);
        let columns = |c: &Cell| {
            let mut x = vec![0];
            if c.w > 0 {
                x.push(c.w);
            }
            if c.d + 1 < m {
                x.push(n + c.d);
            }
            x
        };
        let mut xtx = vec![0.0; k * k];
        let mut xty = vec![0.0; k];
        for c in &self.cells {
            let x = columns(c);
            for &i in &x {
                xty[i] += c.y - c.offset;
                for &j in &x {
                    xtx[i * k + j] += 1.0;
                }
            }
        }
        for j in 0..k {
            xtx[j * k + j] += 1e-8;
        }
        let mut theta = vec![0.0; self.dim()];
        if let Some(l) = cholesky(&xtx, k) {
            theta[..k].copy_from_slice(&cholesky_solve(&l, &xty));
        } else {
            theta[0] = self.logelr_mean;
        }
        let mut sum = vec![0.0; m];
        let mut count = vec![0.0; m];
        for c in &self.cells {
            let r = c.y - self.mu(&theta, c);
            sum[c.d] += r * r;
            count[c.d] += 1.0;
        }
        let mut var: Vec<f64> = sum
            .iter()
            .zip(&count)
            .map(|(s, n)| if *n > 0.0 { s / n } else { 0.0 })
            .collect();
        for d in (0..m.saturating_sub(1)).rev() {
            var[d] = var[d].max(var[d + 1]);
        }
        let u0 = self.u0();
        for d in 0..m {
            let next = if d + 1 < m { var[d + 1] } else { 0.0 };
            let s = (var[d] - next).clamp(1e-3, 0.5);
            let l = (s - S_FLOOR) / (1.0 - S_FLOOR);
            theta[u0 + d] = (l / (1.0 - l)).ln();
        }
        theta
    }
}

impl Posterior for Model {
    fn dim(&self) -> usize {
        self.u0() + self.n_dev
    }

    fn log_posterior(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        grad.fill(0.0);
        let (n, m) = (self.n_origins, self.n_dev);
        let mut f = 0.0;
        if self.priors {
            let mut prior = |j: usize, mean: f64, sd: f64| {
                let r = theta[j] - mean;
                f -= 0.5 * r * r / (sd * sd);
                grad[j] -= r / (sd * sd);
            };
            prior(0, self.logelr_mean, self.logelr_sd);
            for j in 1..n {
                prior(j, 0.0, self.origin_sd);
            }
            for j in n..n + m - 1 {
                prior(j, 0.0, self.development_sd);
            }
            if self.csr() {
                prior(self.after_beta(), 0.0, self.gamma_sd);
            }
            if self.cay() {
                // (1 - rho) / 2 = logistic(-2 v) ~ Beta(a, a), with its
                // Jacobian: a (log logistic(2 v) + log logistic(-2 v)).
                let (k, a) = (self.after_beta(), self.rho_shape);
                let v = theta[k];
                f -= a * (softplus(2.0 * v) + softplus(-2.0 * v));
                grad[k] -= 2.0 * a * v.tanh();
            }
        }
        let (var, s, sc) = self.variances(theta);
        let gamma = if self.csr() {
            theta[self.after_beta()]
        } else {
            0.0
        };
        let rho = if self.cay() {
            theta[self.after_beta()].tanh()
        } else {
            0.0
        };
        let mut r = Vec::with_capacity(self.cells.len());
        self.residuals(theta, rho, &mut r);
        // CAY: h_i = d f / d mu_i at fixed rho, through every later
        // residual of its age's column: h_i = r_i / v_i - rho h_(next i),
        // collected backwards.
        let mut h = Vec::new();
        if self.cay() {
            h = vec![0.0; self.cells.len()];
            for (i, c) in self.cells.iter().enumerate().rev() {
                h[i] += r[i] / var[c.d];
                if let Some(p) = c.prev {
                    h[p] -= rho * h[i];
                }
            }
        }
        let mut drho = 0.0;
        // d f / d sigma_d^2.
        let mut dvar = vec![0.0; m];
        for (i, c) in self.cells.iter().enumerate() {
            let v = var[c.d];
            let r_i = r[i];
            f -= 0.5 * (v.ln() + r_i * r_i / v);
            dvar[c.d] += 0.5 * (r_i * r_i / v - 1.0) / v;
            let e = if self.cay() { h[i] } else { r_i / v };
            if let Some(p) = c.prev {
                drho += e * r[p];
            }
            grad[0] += e;
            if c.w > 0 {
                grad[c.w] += e;
            }
            if c.d + 1 < m {
                let (g, dg) = self.speedup(gamma, c.w);
                let beta = theta[self.beta0() + c.d];
                grad[self.beta0() + c.d] += e * g;
                if self.csr() {
                    grad[self.after_beta()] += e * beta * dg;
                }
            }
        }
        if self.cay() {
            grad[self.after_beta()] += drho * (1.0 - rho * rho);
        }
        // sigma_d^2 = sum over i >= d of s_i, so d f / d s_i sums dvar
        // over d <= i, and ds_i / du_i = (1 - S_FLOOR) l_i (1 - l_i); the
        // U(S_FLOOR, 1) prior on s_i is l_i (1 - l_i) in u_i.
        let u0 = self.u0();
        let mut acc = 0.0;
        for i in 0..m {
            acc += dvar[i];
            let u = theta[u0 + i];
            f -= softplus(-u) + softplus(u);
            grad[u0 + i] = (1.0 - S_FLOOR) * s[i] * sc[i] * acc + (sc[i] - s[i]);
        }
        if f.is_finite() { f } else { f64::NAN }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bayes::laplace::tests::assert_gradient;
    use crate::triangle::Long;
    use crate::{DevelopmentColumn, Grain, Month};
    use prospicio_bayes::nuts::LogDensity;
    use prospicio_core::Period;
    use prospicio_prob::Distribution;

    /// Meyers' illustrative insurer, group 353 Commercial Auto paid
    /// (Meyers 2019, Table 2.2; validation/data/clrd_comauto_353.csv), the
    /// cells at calendar years up to 1997.
    const PAID: [[f64; 10]; 10] = [
        [
            952., 1529., 2813., 3647., 3724., 3832., 3899., 3907., 3911., 3912.,
        ],
        [
            849., 1564., 2202., 2432., 2468., 2487., 2513., 2526., 2531., 2527.,
        ],
        [
            983., 2211., 2830., 3832., 4039., 4065., 4102., 4155., 4268., 4274.,
        ],
        [
            1657., 2685., 3169., 3600., 3900., 4320., 4332., 4338., 4341., 4341.,
        ],
        [
            932., 1940., 2626., 3332., 3368., 3491., 3531., 3540., 3540., 3583.,
        ],
        [
            1162., 2402., 2799., 2996., 3034., 3042., 3230., 3238., 3241., 3268.,
        ],
        [
            1478., 2980., 3945., 4714., 5462., 5680., 5682., 5683., 5684., 5684.,
        ],
        [
            1240., 2080., 2607., 3080., 3678., 4116., 4117., 4125., 4128., 4128.,
        ],
        [
            1326., 2412., 3367., 3843., 3965., 4127., 4133., 4141., 4142., 4144.,
        ],
        [
            1413., 2683., 3173., 3674., 3805., 4005., 4020., 4095., 4132., 4139.,
        ],
    ];
    const PREMIUM: [f64; 10] = [
        5812., 4908., 5454., 5165., 5214., 5230., 4992., 5466., 5226., 4962.,
    ];

    /// The training triangle of `rows` (the first `10 - w` cells of row
    /// `w`), columns `paid` and `premium`, origins from 1988.
    fn triangle(rows: &[[f64; 10]], premium: &[f64]) -> Triangle {
        let n = rows.len();
        let (mut origin, mut ages, mut paid, mut prem) = (vec![], vec![], vec![], vec![]);
        for (w, row) in rows.iter().enumerate() {
            for (d, &v) in row.iter().enumerate().take(n - w) {
                origin.push(Month::january(1988 + w as i32));
                ages.push(12 * (d as crate::Lag + 1));
                paid.push(v);
                prem.push(premium[w]);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    fn comauto() -> Triangle {
        triangle(&PAID, &PREMIUM)
    }

    fn quick(model: MeyersModel, seed: u64) -> Meyers {
        Meyers {
            model,
            sampler: Sampler {
                tune: 300,
                draws: 300,
                ..Meyers::default().sampler
            },
            n_sims: 2_000,
            seed,
            ..Meyers::default()
        }
    }

    fn prepared(model: MeyersModel) -> Model {
        let tri = comauto();
        let segment = tri.segment("paid").unwrap();
        let cl = ChainLadder::default()
            .fit_segment(&segment, &segment.ages)
            .unwrap();
        let premium =
            origin_exposure(&segment, &tri.segment("premium").unwrap(), "premium").unwrap();
        Meyers {
            model,
            ..Meyers::default()
        }
        .model_of(&segment, &cl, &premium)
        .unwrap()
    }

    /// A point off the mode with `gamma` or `rho` away from 0.
    fn off_mode(model: &Model) -> Vec<f64> {
        let mut x = model.start();
        for (j, v) in x.iter_mut().enumerate() {
            *v += 0.01 * (j as f64 % 7.0 - 3.0);
        }
        match model.kind {
            MeyersModel::Crc => {}
            MeyersModel::Csr => x[model.after_beta()] = 0.04,
            MeyersModel::Cay => x[model.after_beta()] = 0.6,
        }
        x
    }

    #[test]
    fn the_gradient_matches_a_finite_difference() {
        for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
            let model = prepared(kind);
            assert_eq!(model.dim(), if kind == MeyersModel::Crc { 29 } else { 30 });
            assert_eq!(model.cells.len(), 55);
            let x = off_mode(&model);
            assert_gradient(|t, g| model.log_posterior(t, g), &x);
            let whitened = Whitened::regularised(&model, &model.start()).unwrap();
            let z: Vec<f64> = (0..model.dim())
                .map(|j| 0.3 * ((j % 3) as f64 - 1.0))
                .collect();
            assert_gradient(|z, g| whitened.log_density(z, g).unwrap(), &z);
        }
    }

    #[test]
    fn csr_at_gamma_zero_is_crc_plus_the_gamma_prior() {
        let (crc, csr) = (prepared(MeyersModel::Crc), prepared(MeyersModel::Csr));
        let theta = off_mode(&crc);
        // The same point with gamma = 0 inserted before u.
        let mut with_gamma = theta[..crc.after_beta()].to_vec();
        with_gamma.push(0.0);
        with_gamma.extend(&theta[crc.after_beta()..]);
        let (mut g1, mut g2) = (vec![0.0; crc.dim()], vec![0.0; csr.dim()]);
        let a = crc.log_posterior(&theta, &mut g1);
        let b = csr.log_posterior(&with_gamma, &mut g2);
        // The gamma prior at 0 is 0 (up to its constant, left out).
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        let mut g2_without = g2.clone();
        g2_without.remove(csr.after_beta());
        for (x, y) in g1.iter().zip(&g2_without) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn cay_at_rho_zero_is_crc_plus_the_rho_prior() {
        let (crc, cay) = (prepared(MeyersModel::Crc), prepared(MeyersModel::Cay));
        let theta = off_mode(&crc);
        let k = crc.after_beta();
        let mut with_v = theta[..k].to_vec();
        with_v.push(0.0);
        with_v.extend(&theta[k..]);
        let (mut g1, mut g2) = (vec![0.0; crc.dim()], vec![0.0; cay.dim()]);
        let a = crc.log_posterior(&theta, &mut g1);
        let b = cay.log_posterior(&with_v, &mut g2);
        // The Beta(2, 2) prior with its Jacobian at v = 0: 2 log(1/4).
        assert!((b - a - 2.0 * 0.25_f64.ln()).abs() < 1e-10, "{a} vs {b}");
        let mut g2_without = g2.clone();
        g2_without.remove(k);
        for (x, y) in g1.iter().zip(&g2_without) {
            assert!((x - y).abs() < 1e-10, "{x} vs {y}");
        }
        // At rho = 0 the rho gradient is the sum over cells after the first
        // origin of (r_i / v_i) r_(w-1)d: not zero on these data.
        assert!(g2[k].abs() > 1e-3, "{}", g2[k]);
    }

    #[test]
    fn cay_follows_meyers_recursion() {
        // Meyers' CAY.R: cells sorted by (age, origin), and
        // mu[i] = base[i] + rho (y[i - 1] - mu[i - 1]) for origins after
        // the first; here recomputed that way from a natural-scale point.
        let model = prepared(MeyersModel::Cay);
        let nat = model.natural(&off_mode(&model));
        let rho = nat[model.after_beta()];
        assert!((rho - 0.6_f64.tanh()).abs() < 1e-15);
        let mut order: Vec<usize> = (0..model.cells.len()).collect();
        order.sort_by_key(|&i| (model.cells[i].d, model.cells[i].w));
        let mut expected = vec![0.0; model.cells.len()];
        let (mut last_y, mut last_mu) = (0.0, 0.0);
        for &i in &order {
            let c = &model.cells[i];
            let mut mu = model.mu(&nat, c);
            if c.w > 0 {
                mu += rho * (last_y - last_mu);
            }
            let sigma = nat[model.u0() + c.d];
            expected[i] =
                -0.5 * ((c.y - mu) / sigma).powi(2) - sigma.ln() - 0.5 * std::f64::consts::TAU.ln();
            (last_y, last_mu) = (c.y, mu);
        }
        let mut got = Vec::new();
        model.log_densities(&nat, &mut Vec::new(), &mut got);
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-12, "{g} vs {e}");
        }
        // The log posterior without priors is their sum less the
        // constant, plus the log Jacobian of each u_i.
        let unpriored = Model {
            priors: false,
            ..model.clone()
        };
        let theta = off_mode(&model);
        let mut grad = vec![0.0; model.dim()];
        let f = unpriored.log_posterior(&theta, &mut grad);
        let jacobian: f64 = theta[model.u0()..]
            .iter()
            .map(|&u| -(softplus(-u) + softplus(u)))
            .sum();
        let n = model.cells.len() as f64;
        let sum: f64 = expected.iter().sum();
        assert!(
            (f - (sum + 0.5 * n * std::f64::consts::TAU.ln() + jacobian)).abs() < 1e-9,
            "{f} vs {sum}"
        );
    }

    #[test]
    fn cay_draws_the_oldest_age_as_a_moving_average() {
        // Given one natural-scale draw with a large sigma_m and rho, the
        // log ultimates down age m follow Meyers' recursion: origin 2's
        // mean takes rho times origin 1's observed residual, and the open
        // origins after it are a moving average of order 1, with lag-one
        // correlation rho / (1 + rho^2) and none further (Meyers 2019,
        // eq. 8.3). On group 353 sigma_m is near 0.013, so this barely
        // moves the predictive there; this test is what checks it.
        let model = prepared(MeyersModel::Cay);
        let (n, k) = (model.n_origins, model.after_beta());
        let (rho, sigma) = (0.8, 0.5);
        let mut nat = vec![0.0; model.dim()];
        nat[0] = -0.4;
        for (w, alpha) in nat.iter_mut().enumerate().take(n).skip(1) {
            *alpha = 0.05 * w as f64;
        }
        nat[k] = rho;
        nat[model.dim() - 1] = sigma;
        let premiums = PREMIUM;
        let mut closed = vec![false; n];
        closed[0] = true;
        let mut latest = vec![0.0_f64; n];
        latest[0] = 3912.0;
        let base = |w: usize| premiums[w].ln() + nat[0] + if w > 0 { nat[w] } else { 0.0 };
        let r1 = latest[0].ln() - base(0);
        let sims = 200_000;
        let mut rng = StreamRng::new(7, 0);
        let mut row = vec![0.0; n];
        let mut logs = vec![Vec::with_capacity(sims); n];
        for _ in 0..sims {
            model.draw_ultimates(&nat, &premiums, &closed, &latest, &mut rng, &mut row);
            assert_eq!(row[0], latest[0]);
            for (w, u) in row.iter().enumerate() {
                logs[w].push(u.ln());
            }
        }
        let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
        let cov = |x: &[f64], y: &[f64]| {
            let (mx, my) = (mean(x), mean(y));
            x.iter()
                .zip(y)
                .map(|(a, b)| (a - mx) * (b - my))
                .sum::<f64>()
                / (x.len() as f64 - 1.0)
        };
        let se = sigma / (sims as f64).sqrt();
        assert!((mean(&logs[1]) - (base(1) + rho * r1)).abs() < 5.0 * se);
        assert!((cov(&logs[1], &logs[1]).sqrt() / sigma - 1.0).abs() < 0.01);
        for (w, log) in logs.iter().enumerate().skip(2) {
            assert!((mean(log) - base(w)).abs() < 5.0 * se * 1.3);
            let sd = cov(log, log).sqrt();
            assert!((sd / (sigma * (1.0 + rho * rho).sqrt()) - 1.0).abs() < 0.01);
        }
        let corr = |a: usize, b: usize| {
            cov(&logs[a], &logs[b]) / (cov(&logs[a], &logs[a]) * cov(&logs[b], &logs[b])).sqrt()
        };
        let ma1 = rho / (1.0 + rho * rho);
        for w in 2..n - 1 {
            assert!((corr(w, w + 1) - ma1).abs() < 0.01, "{}", corr(w, w + 1));
        }
        assert!(corr(3, 5).abs() < 0.01);
        // Origin 2 has no earlier random term: its correlation with origin
        // 3 is rho / sqrt(1 + rho^2).
        assert!((corr(1, 2) - rho / (1.0 + rho * rho).sqrt()).abs() < 0.01);
        // CRC draws the same origins independently.
        let crc = prepared(MeyersModel::Crc);
        let mut crc_nat = nat.clone();
        crc_nat.remove(k);
        let mut a = Vec::with_capacity(sims);
        let mut b = Vec::with_capacity(sims);
        for _ in 0..sims {
            crc.draw_ultimates(&crc_nat, &premiums, &closed, &latest, &mut rng, &mut row);
            a.push(row[3].ln());
            b.push(row[4].ln());
        }
        assert!((cov(&a, &b) / (cov(&a, &a) * cov(&b, &b)).sqrt()).abs() < 0.01);
    }

    #[test]
    fn the_rho_prior_is_meyers_beta() {
        // With no cells and only the rho prior, the density of v is
        // Beta(a, a) of r = (1 - tanh v) / 2 times |dr/dv| = 2 r (1 - r),
        // up to its constant, so differences from v = 0 match exactly.
        for a in [2.0, 0.7, 5.0] {
            let model = Model {
                cells: Vec::new(),
                rho_shape: a,
                ..prepared(MeyersModel::Cay)
            };
            let k = model.after_beta();
            let mut theta = vec![0.0; model.dim()];
            let mut grad = vec![0.0; model.dim()];
            let base = model.log_posterior(&theta, &mut grad);
            for v in [-3.0, -0.4, 0.9, 2.5_f64] {
                theta[k] = v;
                let f = model.log_posterior(&theta, &mut grad) - base;
                let r = (1.0 - v.tanh()) / 2.0;
                let density =
                    |r: f64| (a - 1.0) * (r * (1.0 - r)).ln() + (2.0 * r * (1.0 - r)).ln();
                let expected = density(r) - density(0.5);
                assert!(
                    (f - expected).abs() < 1e-10,
                    "a {a} v {v}: {f} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn the_logit_gives_meyers_uniform_prior_on_s() {
        // Meyers' floor: a_ig < 1e5, s = 1 - exp(-1 / a_ig).
        assert!((S_FLOOR + (-1e-5_f64).exp_m1()).abs() < 1e-20);
        // With no cells and no priors, the density in u is
        // log l + log(1 - l) for l = logistic(u): the U(S_FLOOR, 1) density
        // of s = S_FLOOR + (1 - S_FLOOR) l times |ds/du|.
        let model = Model {
            cells: Vec::new(),
            priors: false,
            ..prepared(MeyersModel::Crc)
        };
        let mut theta = vec![0.0; model.dim()];
        let mut grad = vec![0.0; model.dim()];
        for u in [-6.0, -1.0, 0.0, 0.5, 3.0] {
            theta[model.u0()..].fill(u);
            let f = model.log_posterior(&theta, &mut grad);
            let s = logistic(u);
            let expected = model.n_dev as f64 * (s * (1.0 - s)).ln();
            assert!((f - expected).abs() < 1e-12, "u {u}: {f} vs {expected}");
        }
        // Far down, every step is the floor.
        theta[model.u0()..].fill(-60.0);
        let (var, _, _) = model.variances(&theta);
        assert!((var[model.n_dev - 1] / S_FLOOR - 1.0).abs() < 1e-12);
        assert!((var[0] / (10.0 * S_FLOOR) - 1.0).abs() < 1e-12);
        // Its integral over u is 1 (trapezoid on a wide grid).
        let h = 1e-3;
        let total: f64 = (-40_000..=40_000)
            .map(|k| {
                let s = logistic(k as f64 * h);
                s * (1.0 - s) * h
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-9, "{total}");
    }

    #[test]
    fn the_start_is_least_squares_with_falling_sigmas() {
        for kind in [MeyersModel::Crc, MeyersModel::Csr, MeyersModel::Cay] {
            let model = prepared(kind);
            let start = model.start();
            // The location part is least squares: its gradient without
            // priors, at equal sigmas, is zero.
            let ols = Model {
                priors: false,
                ..model.clone()
            };
            // All sigma_d^2 equal (to 0.5, up to 1e-22): only s_m is not
            // negligible.
            let mut theta = start.clone();
            theta[model.u0()..].fill(-50.0);
            theta[model.dim() - 1] = 0.0;
            let mut grad = vec![0.0; model.dim()];
            ols.log_posterior(&theta, &mut grad);
            let worst = grad[..model.after_beta()]
                .iter()
                .fold(0.0_f64, |m, g| m.max(g.abs()));
            // Up to the floor's 1e-4 relative difference between the sigmas.
            assert!(worst < 1e-3, "{kind:?}: gradient {worst}");
            if kind != MeyersModel::Crc {
                assert_eq!(start[model.after_beta()], 0.0);
            }
            let sigma = &model.natural(&start)[model.u0()..];
            assert!(sigma.windows(2).all(|p| p[0] > p[1]), "{sigma:?}");
            assert!(sigma.iter().all(|&s| s > 0.0 && s < 1.0), "{sigma:?}");
        }
    }

    #[test]
    fn the_joint_mode_is_deep_in_the_funnel() {
        // Why Meyers samples on a pilot scale and not at the mode as
        // BayesOdp does: the joint mode puts sigma[10] far below its
        // posterior mean (0.020 in Stan's reference), where a Laplace scale
        // says little about the bulk.
        let model = prepared(MeyersModel::Crc);
        let theta = match crate::bayes::laplace::mode(&model, model.start()) {
            Ok(theta) => theta,
            // Newton creeps down the funnel and may not converge.
            Err(Error::Bayes(_)) => return,
            Err(e) => panic!("{e}"),
        };
        let sigma = model.natural(&theta)[model.dim() - 1];
        assert!(sigma < 0.005, "sigma[10] at the mode {sigma}");
    }

    #[test]
    fn a_seed_replays_whatever_the_threads() {
        let tri = comauto();
        let a = quick(MeyersModel::Csr, 3)
            .fit(&tri, "paid", "premium")
            .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let b = pool.install(|| {
            quick(MeyersModel::Csr, 3)
                .fit(&tri, "paid", "premium")
                .unwrap()
        });
        assert_eq!(a.posterior, b.posterior);
        assert_eq!(a.reserves.draw_matrix(), b.reserves.draw_matrix());
        let c = quick(MeyersModel::Csr, 4)
            .fit(&tri, "paid", "premium")
            .unwrap();
        assert_ne!(a.posterior.values(), c.posterior.values());
        assert_ne!(a.reserves.draw_matrix(), c.reserves.draw_matrix());
        // The sampler's own seed is not used.
        let mut d = quick(MeyersModel::Csr, 3);
        d.sampler.seed = 99;
        assert_eq!(
            d.fit(&tri, "paid", "premium").unwrap().posterior,
            a.posterior
        );
        // CAY's sequential prediction replays too.
        let e = quick(MeyersModel::Cay, 3)
            .fit(&tri, "paid", "premium")
            .unwrap();
        let f = pool.install(|| {
            quick(MeyersModel::Cay, 3)
                .fit(&tri, "paid", "premium")
                .unwrap()
        });
        assert_eq!(e.posterior, f.posterior);
        assert_eq!(e.reserves.draw_matrix(), f.reserves.draw_matrix());
    }

    #[test]
    fn a_short_run_returns_the_fit_with_warnings() {
        let model = Meyers {
            sampler: Sampler {
                tune: 50,
                draws: 20,
                ..Meyers::default().sampler
            },
            n_sims: 100,
            ..Meyers::default()
        };
        let fit = model.fit(&comauto(), "paid", "premium").unwrap();
        assert!(fit.diagnostics.min_ess_bulk < 400.0);
        assert!(
            fit.diagnostics
                .warnings
                .iter()
                .any(|w| w.contains("bulk ESS")),
            "{:?}",
            fit.diagnostics.warnings
        );
    }

    #[test]
    fn ultimates_and_reserves_by_origin() {
        let fit = quick(MeyersModel::Crc, 5)
            .fit(&comauto(), "paid", "premium")
            .unwrap();
        for pd in [&fit.ultimates, &fit.reserves] {
            assert_eq!(pd.dims(), ["origin"]);
            assert_eq!(pd.n_components(), 10);
            assert_eq!(pd.n_sims(), 2_000);
        }
        let oldest = vec![Period::year(1988).into()];
        let reserve = fit.reserves.marginal(&oldest).unwrap();
        assert_eq!((reserve.mean(), reserve.std_dev()), (0.0, 0.0));
        assert_eq!(fit.ultimates.marginal(&oldest).unwrap().mean(), 3912.0);
        // Reserves are the ultimates less the latest values, draw by draw.
        for (row_u, row_r) in fit
            .ultimates
            .draw_matrix()
            .chunks_exact(10)
            .zip(fit.reserves.draw_matrix().chunks_exact(10))
        {
            for ((u, r), l) in row_u.iter().zip(row_r).zip(&fit.chain_ladder.latest) {
                assert_eq!(*r, u - l);
            }
        }
        let names = fit.posterior.names();
        assert_eq!(names.len(), 29);
        assert_eq!(names[0], "logelr");
        assert_eq!(names[9], "alpha[10]");
        assert_eq!(names[10], "beta[1]");
        assert_eq!(names[19], "sigma[1]");
        assert_eq!(names[28], "sigma[10]");
        assert_eq!(fit.premium, PREMIUM);
        let long = fit.to_long();
        assert_eq!(long.n_rows(), 10);
        for column in [
            "premium",
            "mean",
            "std_dev",
            "ultimate_mean",
            "ultimate_std_dev",
        ] {
            assert!(long.column(column).is_some(), "{column}");
        }
        let totals = fit.totals();
        assert_eq!(
            totals.column("premium").unwrap(),
            [PREMIUM.iter().sum::<f64>()]
        );
        assert!((totals.column("mean").unwrap()[0] - fit.reserves.total().mean()).abs() < 1e-6);
        assert!(
            (totals.column("ultimate_mean").unwrap()[0] - fit.ultimates.total().mean()).abs()
                < 1e-6
        );
        assert_eq!(fit.development_table().n_rows(), 10);
        assert_eq!(fit.summary().unwrap().len(), 29);
        let loo = fit.loo().unwrap();
        assert_eq!(loo.pareto_k.len(), 55);
        assert!(loo.estimate.elpd.is_finite());
    }

    #[test]
    fn provenance_records_the_settings() {
        let fit = quick(MeyersModel::Csr, 5)
            .fit(&comauto(), "paid", "premium")
            .unwrap();
        assert_eq!(fit.posterior.names()[19], "gamma");
        assert_eq!(fit.model(), MeyersModel::Csr);
        for (pd, output) in [(&fit.ultimates, "ultimates"), (&fit.reserves, "reserves")] {
            let provenance = pd.provenance();
            assert_eq!(provenance.model, "meyers_csr");
            let keys: Vec<&str> = provenance
                .parameters
                .iter()
                .map(|(k, _)| k.as_str())
                .collect();
            for key in [
                "logelr_mean",
                "logelr_sd",
                "origin_sd",
                "development_sd",
                "gamma_sd",
                "chains",
                "tune",
                "draws",
                "target_accept",
                "max_depth",
                "n_sims",
                "seed",
                "column",
                "premium",
                "output",
                "posterior_draws",
            ] {
                assert!(keys.contains(&key), "{key} missing from {keys:?}");
            }
            assert!(
                provenance
                    .parameters
                    .iter()
                    .any(|(k, v)| k == "output" && v == output)
            );
        }
        let crc = quick(MeyersModel::Crc, 5)
            .fit(&comauto(), "paid", "premium")
            .unwrap();
        assert_eq!(crc.reserves.provenance().model, "meyers_crc");
        let cay = quick(MeyersModel::Cay, 5)
            .fit(&comauto(), "paid", "premium")
            .unwrap();
        assert_eq!(cay.model(), MeyersModel::Cay);
        assert_eq!(cay.posterior.names()[19], "rho");
        let rho = cay.posterior.parameter(19).concat();
        assert!(rho.iter().all(|r| r.abs() < 1.0));
        let provenance = cay.reserves.provenance();
        assert_eq!(provenance.model, "meyers_cay");
        assert!(
            provenance
                .parameters
                .iter()
                .any(|(k, v)| k == "rho_shape" && v == "2")
        );
        assert!(
            !provenance.parameters.iter().any(|(k, _)| k == "gamma_sd"),
            "{:?}",
            provenance.parameters
        );
        // The oldest origin keeps its value; the CAY chain starts from its
        // observed residual.
        let oldest = vec![Period::year(1988).into()];
        assert_eq!(cay.ultimates.marginal(&oldest).unwrap().mean(), 3912.0);
        assert_eq!(cay.loo().unwrap().pareto_k.len(), 55);
    }

    #[test]
    fn errors() {
        let tri = comauto();
        for (name, model) in [
            (
                "logelr_sd",
                Meyers {
                    logelr_sd: 0.0,
                    ..Meyers::default()
                },
            ),
            (
                "origin_sd",
                Meyers {
                    origin_sd: f64::NAN,
                    ..Meyers::default()
                },
            ),
            (
                "development_sd",
                Meyers {
                    development_sd: -1.0,
                    ..Meyers::default()
                },
            ),
            (
                "gamma_sd",
                Meyers {
                    gamma_sd: f64::INFINITY,
                    ..Meyers::default()
                },
            ),
            (
                "logelr_mean",
                Meyers {
                    logelr_mean: f64::NAN,
                    ..Meyers::default()
                },
            ),
            (
                "n_sims",
                Meyers {
                    n_sims: 0,
                    ..Meyers::default()
                },
            ),
            (
                "rho_shape",
                Meyers {
                    model: MeyersModel::Cay,
                    rho_shape: 0.0,
                    ..Meyers::default()
                },
            ),
        ] {
            let e = model.fit(&tri, "paid", "premium").unwrap_err();
            assert!(
                matches!(e, Error::InvalidSetting { name: n, .. } if n == name),
                "{name}: {e}"
            );
        }
        assert!(matches!(
            Meyers::default().fit(&tri, "incurred", "premium"),
            Err(Error::UnknownColumn(_))
        ));

        // A non-positive cell: 1990 at 24 months.
        let mut rows = PAID;
        rows[2][1] = 0.0;
        let e = quick(MeyersModel::Crc, 0)
            .fit(&triangle(&rows, &PREMIUM), "paid", "premium")
            .unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("origin 1990 at development age 24 months is 0")),
            "{e}"
        );
        // A missing or non-positive premium.
        let mut premium = PREMIUM;
        premium[4] = 0.0;
        let e = quick(MeyersModel::Crc, 0)
            .fit(&triangle(&PAID, &premium), "paid", "premium")
            .unwrap_err();
        assert!(
            matches!(&e, Error::InvalidExposure { origin, .. } if origin == "1992"),
            "{e}"
        );
        // Too few cells: 3 origins, 6 cells, 3 + 2 + 3 = 8 parameters.
        let small = triangle(&PAID[..3], &PREMIUM[..3]);
        let e = quick(MeyersModel::Crc, 0)
            .fit(&small, "paid", "premium")
            .unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("6 observed cells for 8 parameters")),
            "{e}"
        );
        // CAY without the previous origin's cell: 1992 stops at 36 months,
        // 1993 runs to 60.
        let (mut origin, mut ages, mut paid, mut prem) = (vec![], vec![], vec![], vec![]);
        for (w, row) in PAID.iter().enumerate() {
            let cells = if w == 4 { 3 } else { 10 - w };
            for (d, &v) in row.iter().enumerate().take(cells) {
                origin.push(Month::january(1988 + w as i32));
                ages.push(12 * (d as crate::Lag + 1));
                paid.push(v);
                prem.push(PREMIUM[w]);
            }
        }
        let ragged = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let e = quick(MeyersModel::Cay, 0)
            .fit(&ragged, "paid", "premium")
            .unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with(
                "origin 1993 is observed at development age 48 months but origin 1992 is not"
            )),
            "{e}"
        );
        // CRC takes the same triangle.
        quick(MeyersModel::Crc, 0)
            .fit(&ragged, "paid", "premium")
            .unwrap();

        let origin = [2020, 2020, 2021].map(Month::january);
        let two = Triangle::from_long(&Long {
            keys: &[("lob", &["a", "a", "a", "b", "b", "b"])],
            origin: &[origin, origin].concat(),
            development: DevelopmentColumn::Age(&[12, 24, 12, 12, 24, 12]),
            values: &[
                ("paid", &[1.0, 2.0, 1.0, 1.0, 2.0, 1.0]),
                ("premium", &[3.0; 6]),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        assert!(matches!(
            quick(MeyersModel::Crc, 0).fit(&two, "paid", "premium"),
            Err(Error::MultipleSegments(2))
        ));
    }
}
