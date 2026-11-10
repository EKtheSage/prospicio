//! The hierarchical Bornhuetter–Ferguson: several programs pooled under an
//! industry hyperprior on the loss ratio, the development pattern partially
//! pooled, each program's cells the over-dispersed Poisson of
//! [`BayesBfNormalised`](super::BayesBfNormalised)
//! (`docs/design/reserving-bayes.md`, decision 30).

use prospicio_bayes::nuts::{ParamSummary, PosteriorDraws, Sampler, sample};
use prospicio_core::StreamRng;
use prospicio_prob::{ComponentKey, InputHasher, KeyValue, PredictiveDistribution, Provenance};

use super::bf_normalised::{column_error, origin_error};
use super::dual::{CHUNK, gradient};
use super::laplace::{Posterior, Whitened};
use super::one_year::{self, BayesOneYear, BayesOneYearFit, LightRefit, OneYearRefit, YearModel};
use super::scalar::Scalar;
use super::{Diagnostics, Stream, derive_seed, one_year_seed, pilot_seed};
use crate::bootstrap::{ProcessDistribution, prepare, process_draw};
use crate::error::{Error, Result};
use crate::expected_loss::origin_exposure;
use crate::industry_prior::PatternPrior;
use crate::tail::Tail;
use crate::triangle::{Label, Segment, Triangle};
use crate::{Lag, Period};

/// The median of a half-normal over its scale, `Phi^-1(0.75)`: the start
/// of `tau` and `omega`.
const HALF_NORMAL_MEDIAN: f64 = 0.674_489_750_196_081_7;

/// The factor a start uses at an age whose pooled column gives none above
/// 1: only a start, the column rule has already passed.
const FLAT_FACTOR: f64 = 1.001;

/// The pilot run has at most this many warm-up draws and a fifth as many
/// kept draws (at least `MIN_PILOT_DRAWS`): only their variances are used.
const MAX_PILOT_TUNE: usize = 500;

/// From this many programs the pilot's full covariance scales the main run;
/// below it only the pilot's variances do (the hyperparameters of a few
/// programs have a funnel that a dense scale fitted on a pilot misreads).
const DENSE_FROM_PROGRAMS: usize = 8;

/// The fewest draws of a pilot run, as [`Meyers`](super::Meyers).
const MIN_PILOT_DRAWS: usize = 100;

/// The R-hat above which, and the effective draws below which, a light
/// refit's closing reserve warns (the bounds of `Compartmental`'s, decision
/// 24).
const LIGHT_MAX_RHAT: f64 = 1.05;
const LIGHT_MIN_ESS: f64 = 100.0;

/// The hierarchical Bornhuetter–Ferguson over programs, one segment of a
/// [`Triangle`] each, all on the triangle's development ages:
///
/// ```text
/// L ~ LogNormal(mean apriori, cv level_cv)        flat on log L when level_cv is None
/// tau ~ HalfNormal(tau_scale)                     the spread of programs' loss ratios
/// log s_c = tau z_c - tau^2 / 2,  z_c ~ N(0, 1)   so E[s_c] = 1
/// U_cw ~ Gamma(shape 1 / cv^2, mean L s_c P_cw)   the origin's expected ultimate
/// beta_d ~ N(0, development_sd^2),  d = 2, ..., m,  beta_1 = 0
/// omega ~ HalfNormal(omega_scale)                 the spread of programs' patterns
/// beta_cd = beta_d + omega delta_cd,  delta_cd ~ N(0, 1),  beta_c1 = 0
/// p_cd = exp(beta_cd) / sum_k exp(beta_ck)
/// X_cwd ~ quasi-ODP(mean U_cw p_cd, scale phi_c)
/// ```
///
/// A program's loss ratio is `L s_c`, shrunk towards `L` by as much as the
/// spread between the programs supports, and a thin program's young ages
/// move with its own data while its later ages follow the shared pattern.
/// One program under a tight `level_cv` with `tau` and `omega` near 0 is
/// [`BayesBfNormalised`](super::BayesBfNormalised).
///
/// The scale is plugged in, pooled per unit of premium: `phi_c = kappa
/// Pbar_c` with `Pbar_c` the program's mean premium per origin and `kappa
/// = sum_c (X2_c / Pbar_c) / sum_c df_c` from each program's chain ladder,
/// so a program of two origins, which has no degree of freedom of its own,
/// still has a scale. Design note `reserving-bayes.md`, decision 30, the
/// user's decision of 2026-11-04 (open decision 26).
///
/// ```
/// use prospicio_bayes::nuts::Sampler;
/// use prospicio_reserving::bayes::HierarchicalBf;
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// use prospicio_prob::Distribution;
///
/// let program = ["a", "a", "a", "a", "a", "a", "b", "b", "b"];
/// let origin = [2021, 2021, 2021, 2022, 2022, 2023, 2022, 2022, 2023].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[("program", &program)],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12, 12, 24, 12]),
///     values: &[
///         ("paid", &[100.0, 150.0, 170.0, 110.0, 172.0, 130.0, 55.0, 80.0, 60.0]),
///         ("premium", &[250.0, 250.0, 250.0, 270.0, 270.0, 290.0, 120.0, 120.0, 130.0]),
///     ],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let model = HierarchicalBf {
///     apriori: 0.7,
///     level_cv: Some(0.2),
///     sampler: Sampler { tune: 200, draws: 200, chains: 2, ..Sampler::default() },
///     n_sims: 500,
///     ..HierarchicalBf::default()
/// };
/// let fit = model.fit(&tri, "paid", "premium")?;
/// assert_eq!(fit.programs.len(), 2);
/// // The thin program has a reserve past its own last age.
/// assert!(fit.programs[1].reserves.mean() > 0.0);
/// assert_eq!(fit.reserves.dims(), ["program", "origin"]);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalBf {
    /// The industry loss ratio's prior mean `m_L`: the mean of the
    /// lognormal hyperprior, and the centre of the sampling scale under a
    /// flat one.
    pub apriori: f64,
    /// The coefficient of variation of the hyperprior on the industry loss
    /// ratio `L`; `None` (the default) is flat on `log L`, so the programs
    /// alone set the level.
    pub level_cv: Option<f64>,
    /// The coefficient of variation of each origin's expected ultimate
    /// about its program's level: the spread within a program, fixed as in
    /// [`BayesBfNormalised`](super::BayesBfNormalised), since two or three
    /// origins say almost nothing about it. It must be finite.
    pub cv: f64,
    /// Standard deviation of the normal prior on each shared `beta_d`,
    /// centred on 0; unused by
    /// [`fit_with_pattern_prior`](Self::fit_with_pattern_prior).
    pub development_sd: f64,
    /// Scale of the half-normal prior on `tau`, on the log loss ratio.
    pub tau_scale: f64,
    /// Scale of the half-normal prior on `omega`, on the log share ratio.
    pub omega_scale: f64,
    /// Process error on each simulated future increment.
    pub process: ProcessDistribution,
    /// NUTS settings. Its `seed` is not used: the chains take a seed
    /// derived from [`seed`](Self::seed) (module docs, "Seeds").
    pub sampler: Sampler,
    /// Number of predictive simulations.
    pub n_sims: usize,
    /// Seed of the chains and of the simulations, each through its own
    /// derived seed.
    pub seed: u64,
}

impl Default for HierarchicalBf {
    /// `apriori` 1, a flat `level_cv`, `cv` 0.1, `development_sd` 10,
    /// `tau_scale` 0.3 (a median spread of about 20% between programs'
    /// loss ratios), `omega_scale` 0.5, Gamma process error, four chains
    /// of 1,000 warm-up and 1,000 kept draws with target acceptance 0.99,
    /// 10,000 simulations, seed 0.
    fn default() -> Self {
        Self {
            apriori: 1.0,
            level_cv: None,
            cv: 0.1,
            development_sd: 10.0,
            tau_scale: 0.3,
            omega_scale: 0.5,
            process: ProcessDistribution::Gamma,
            sampler: Sampler {
                target_accept: 0.99,
                ..Sampler::default()
            },
            n_sims: 10_000,
            seed: 0,
        }
    }
}

/// One program of a [`HierarchicalBfFit`].
#[derive(Debug, Clone)]
pub struct HierarchicalBfProgram {
    /// The program's segment label.
    pub label: Label,
    /// The program's origins.
    pub origins: Vec<Period>,
    /// Exposure per origin: the exposure column's latest observed value.
    pub exposure: Vec<f64>,
    /// The latest observed cumulative value of each origin.
    pub latest: Vec<f64>,
    /// The hyperprior's mean of each origin's expected ultimate, `apriori
    /// * exposure`.
    pub prior_ultimate: Vec<f64>,
    /// The plugged-in scale `phi_c = kappa Pbar_c`.
    pub scale: f64,
    /// The posterior mean of each origin's expected ultimate `U_cw`.
    pub expected_ultimate: Vec<f64>,
    /// The posterior mean of the share `p_cd` at each of the fit's
    /// [`ages`](HierarchicalBfFit::ages); the shares sum to 1.
    pub pattern: Vec<f64>,
    /// The posterior mean of the program's loss ratio `L s_c`.
    pub loss_ratio: f64,
    /// The loss ratio of the program's own chain ladder: its ultimates
    /// over its premium, with its own volume-weighted factors and, past
    /// the ages it observes, the factors pooled over every program.
    pub own_loss_ratio: f64,
    /// How far the program stays from the collective, on the log scale:
    /// `(post - mu) / (own - mu)`, with `post` the posterior mean of `log
    /// (L s_c)`, `own` the log of [`own_loss_ratio`](Self::own_loss_ratio)
    /// and `mu` the posterior mean of `log L - tau^2 / 2`. 1 keeps the
    /// program's own level and 0 takes the collective's; NaN when its own
    /// loss ratio is not positive. It is a ratio of two estimates, not a
    /// credibility factor, so it can fall outside 0 to 1, as it does for a
    /// program near the collective or moved by its pattern's deviation.
    pub shrinkage: f64,
    /// The program's part of [`HierarchicalBfFit::reserves`], by origin,
    /// paired with it simulation by simulation.
    pub reserves: PredictiveDistribution,
}

/// A fitted [`HierarchicalBf`].
#[derive(Debug, Clone)]
pub struct HierarchicalBfFit {
    /// The programs, in the triangle's index order.
    pub programs: Vec<HierarchicalBfProgram>,
    /// The development ages of every program's pattern.
    pub ages: Vec<Lag>,
    /// The pooled scale per unit of premium `kappa`.
    pub scale_per_premium: f64,
    /// Posterior draws, by chain: `log_level`, `log_tau`, `z[c]`,
    /// `alpha[c,w]` (the log expected ultimates), `beta[d]`, `log_omega`
    /// and `delta[c,d]`, including the deviations of the ages a program
    /// never observes, which each draw carries.
    pub posterior: PosteriorDraws,
    /// R-hat, ESS and divergences, with warnings for failed thresholds.
    pub diagnostics: Diagnostics,
    /// The joint posterior predictive distribution of every program's
    /// reserve by origin: the segment's keys and `origin` as dimensions.
    /// The draws are paired by posterior draw, since the programs share
    /// parameters.
    pub reserves: PredictiveDistribution,
}

impl HierarchicalBf {
    /// Fits loss `column` of a cumulative triangle with a segment per
    /// program and exposure from its `exposure` column. A triangle with
    /// one segment is one program. Every origin must be observed from the
    /// triangle's first age to its latest, and the programs together must
    /// have a degree of freedom. Failed diagnostics do not fail the fit:
    /// they are in [`HierarchicalBfFit::diagnostics`].
    pub fn fit(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
    ) -> Result<HierarchicalBfFit> {
        self.fit_with(triangle, column, exposure, None)
    }

    /// [`fit`](Self::fit) with an informative prior on the shared pattern:
    /// each `beta_d ~ N(prior.mean_d, prior.sd_d^2)` (decision 29), for
    /// example the industry pattern of an
    /// [`IndustryPrior`](crate::IndustryPrior). The prior's ages must be
    /// the programs' ages from the first to the last. `development_sd` is
    /// then unused.
    pub fn fit_with_pattern_prior(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
        prior: &PatternPrior,
    ) -> Result<HierarchicalBfFit> {
        self.fit_with(triangle, column, exposure, Some(prior))
    }

    fn fit_with(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
        pattern: Option<&PatternPrior>,
    ) -> Result<HierarchicalBfFit> {
        self.validate()?;
        let panel = self.prepare(triangle, column, exposure, pattern)?;
        self.fit_panel(triangle, column, exposure, &panel)
    }

    /// The fit of a prepared `panel`.
    fn fit_panel(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
        panel: &Panel,
    ) -> Result<HierarchicalBfFit> {
        let model = &panel.model;
        let (posterior, diagnostics) = self.sample(
            model,
            panel.start.clone(),
            derive_seed(self.seed, Stream::Chains),
        )?;

        let dims = Self::dims(triangle);
        let components = panel.components();
        let provenance = self.provenance(column, exposure, panel);
        let process = self.process;
        let reserves = posterior.predictive(
            dims,
            components,
            self.n_sims,
            derive_seed(self.seed, Stream::Predictive),
            provenance.clone(),
            |theta, rng: &mut StreamRng, row| model.simulate(theta, process, rng, row),
        )?;

        let programs = self.programs(panel, &posterior, &reserves, &provenance)?;
        Ok(HierarchicalBfFit {
            programs,
            ages: panel.ages.clone(),
            scale_per_premium: panel.kappa,
            posterior,
            diagnostics,
            reserves,
        })
    }

    /// The dimensions of a joint distribution over every program's
    /// origins: the triangle's keys and `origin`.
    fn dims(triangle: &Triangle) -> Vec<String> {
        let mut dims = triangle.key_names().to_vec();
        dims.push("origin".to_string());
        dims
    }

    /// Each program's posterior summaries and its slice of the joint
    /// `reserves`.
    fn programs(
        &self,
        panel: &Panel,
        posterior: &PosteriorDraws,
        reserves: &PredictiveDistribution,
        provenance: &Provenance,
    ) -> Result<Vec<HierarchicalBfProgram>> {
        let model = &panel.model;
        let n_programs = model.programs.len();
        let dim = posterior.dim();
        let n = posterior.n_draws() as f64;
        let (mut mu, mut log_ratio, mut ratio) =
            (0.0, vec![0.0; n_programs], vec![0.0; n_programs]);
        let mut ultimate: Vec<Vec<f64>> = model
            .programs
            .iter()
            .map(|p| vec![0.0; p.n_origins()])
            .collect();
        let mut pattern = vec![vec![0.0; model.n_dev]; n_programs];
        for theta in posterior.values().chunks_exact(dim) {
            let tau = theta[1].exp();
            mu += (theta[0] - 0.5 * tau * tau) / n;
            for (c, program) in model.programs.iter().enumerate() {
                let log_level = theta[0] + tau * theta[model.z(c)] - 0.5 * tau * tau;
                log_ratio[c] += log_level / n;
                ratio[c] += log_level.exp() / n;
                for (u, a) in ultimate[c].iter_mut().zip(&theta[model.u(program, 0)..]) {
                    *u += a.exp() / n;
                }
                for (p, l) in pattern[c].iter_mut().zip(model.log_shares(theta, c)) {
                    *p += l.exp() / n;
                }
            }
        }
        let width = reserves.n_components();
        let mut programs = Vec::with_capacity(n_programs);
        for (c, program) in model.programs.iter().enumerate() {
            let columns = program.first..program.first + program.n_origins();
            let draws: Vec<f64> = reserves
                .draw_matrix()
                .chunks_exact(width)
                .flat_map(|row| row[columns.clone()].iter().copied())
                .collect();
            let components = panel.origins[c].iter().map(|&o| vec![o.into()]).collect();
            let own = panel.own_loss_ratio[c];
            let shrinkage = if own > 0.0 {
                (log_ratio[c] - mu) / (own.ln() - mu)
            } else {
                f64::NAN
            };
            programs.push(HierarchicalBfProgram {
                label: panel.labels[c].clone(),
                origins: panel.origins[c].clone(),
                exposure: panel.exposure[c].clone(),
                latest: panel.latest[c].clone(),
                prior_ultimate: panel.exposure[c].iter().map(|e| self.apriori * e).collect(),
                scale: program.phi,
                expected_ultimate: std::mem::take(&mut ultimate[c]),
                pattern: std::mem::take(&mut pattern[c]),
                loss_ratio: ratio[c],
                own_loss_ratio: own,
                shrinkage,
                reserves: PredictiveDistribution::from_draws(
                    vec!["origin".to_string()],
                    components,
                    draws,
                    provenance.clone(),
                )?,
            });
        }
        Ok(programs)
    }

    /// NUTS on `model` after a pilot run from `start`, as
    /// [`Meyers`](super::Meyers): the mode of a hierarchy sits at `tau =
    /// 0`, so there is no mode to whiten at. The draws on the stored scale
    /// and their diagnostics.
    fn sample(
        &self,
        model: &Model,
        start: Vec<f64>,
        chains: u64,
    ) -> Result<(PosteriorDraws, Diagnostics)> {
        let p = model.dim();
        let rough = Whitened::regularised(model, &start)?;
        let pilot = sample(
            &rough,
            &vec![0.0; p],
            Sampler {
                seed: pilot_seed(chains),
                tune: self.sampler.tune.min(MAX_PILOT_TUNE),
                draws: (self.sampler.tune / 5).max(MIN_PILOT_DRAWS),
                ..self.sampler
            },
        )?;
        let pilot = pilot.transform(model.names(), |z| rough.theta(z))?;
        let whitened = if model.programs.len() < DENSE_FROM_PROGRAMS {
            Whitened::from_draws_diagonal(model, &pilot)?
        } else {
            Whitened::from_draws(model, &pilot)?
        };
        let draws = sample(
            &whitened,
            &vec![0.0; p],
            Sampler {
                seed: chains,
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
        Ok((posterior, diagnostics))
    }

    fn validate(&self) -> Result<()> {
        let positive = |name: &'static str, value: f64, expected: &'static str| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(Error::InvalidSetting {
                    name,
                    value,
                    expected,
                })
            }
        };
        positive("apriori", self.apriori, "a finite, positive loss ratio")?;
        if let Some(c) = self.level_cv {
            positive(
                "level_cv",
                c,
                "a finite, positive coefficient of variation (None for a flat hyperprior)",
            )?;
        }
        positive(
            "cv",
            self.cv,
            "a finite, positive coefficient of variation (the programs' levels are not \
             identified under a flat prior on the origins)",
        )?;
        positive(
            "development_sd",
            self.development_sd,
            "a finite, positive prior standard deviation",
        )?;
        positive("tau_scale", self.tau_scale, "a finite, positive scale")?;
        positive("omega_scale", self.omega_scale, "a finite, positive scale")?;
        if self.n_sims == 0 {
            return Err(Error::InvalidSetting {
                name: "n_sims",
                value: 0.0,
                expected: "a positive number of simulations",
            });
        }
        Ok(())
    }

    /// The panel's cells, scales, model and start.
    fn prepare(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
        pattern: Option<&PatternPrior>,
    ) -> Result<Panel> {
        let segments = triangle.segments(column)?;
        let exposures = triangle.segments(exposure)?;
        let labels = triangle.index().to_vec();
        let name = |c: usize| {
            let label = labels[c].parts().join("/");
            if label.is_empty() {
                "the program".to_string()
            } else {
                format!("program {label}")
            }
        };

        // The grid: every program's ages, from the first to the last.
        let first = segments.iter().map(|s| s.dev_offset).min();
        let last = segments.iter().map(|s| s.dev_offset + s.n_dev).max();
        let (Some(first), Some(last)) = (first, last) else {
            return Err(Error::Bayes("the triangle has no programs".to_string()));
        };
        let ages = triangle.development()[first..last].to_vec();
        let nd = ages.len();
        if nd < 2 {
            return Err(Error::Bayes(
                "the programs observe one development age: there is no pattern to estimate"
                    .to_string(),
            ));
        }
        let (development_mean, development_precision) = match pattern {
            None => (
                vec![0.0; nd - 1],
                vec![self.development_sd.powi(-2); nd - 1],
            ),
            Some(prior) => {
                if prior.ages != ages {
                    return Err(Error::Bayes(format!(
                        "the pattern prior's ages {:?} are not the programs' {ages:?}: a prior \
                         with more ages is cut to the programs' last age with \
                         PatternPrior::truncated or IndustryPrior::at_age",
                        prior.ages
                    )));
                }
                if prior.mean.len() != nd - 1 || prior.sd.len() != nd - 1 {
                    return Err(Error::Bayes(format!(
                        "the pattern prior has {} means and {} sds for {} log share ratios",
                        prior.mean.len(),
                        prior.sd.len(),
                        nd - 1
                    )));
                }
                (
                    prior.mean.clone(),
                    prior.sd.iter().map(|s| s.powi(-2)).collect(),
                )
            }
        };

        // Each program's cells on the grid, its Pearson statistic and its
        // degrees of freedom.
        let mut data = Vec::with_capacity(segments.len());
        for (c, (segment, exposure_segment)) in segments.iter().zip(&exposures).enumerate() {
            if segment.dev_offset != first {
                return Err(Error::Bayes(format!(
                    "{}: every origin must be observed from the first age, {} months, and its \
                     first age is {} months",
                    name(c),
                    ages[0],
                    segment.ages[0]
                )));
            }
            let premium = origin_exposure(segment, exposure_segment, exposure)?;
            let observed = observed_cells(segment)
                .map_err(|why| Error::Bayes(format!("{}: {why}", name(c))))?;
            let parameters = segment.n_origins + segment.n_dev - 1;
            let (pearson, df) = if observed.cells.len() > parameters {
                let df = (observed.cells.len() - parameters) as f64;
                let (fit, _) = prepare(segment, &segment.ages, Tail::default())?;
                (fit.scale * df, df)
            } else {
                (0.0, 0.0)
            };
            data.push(ProgramData {
                premium,
                observed,
                pearson,
                df,
            });
        }

        // The scale pooled per unit of premium (open decision 26, D2).
        let mean_premium: Vec<f64> = data
            .iter()
            .map(|p| p.premium.iter().sum::<f64>() / p.premium.len() as f64)
            .collect();
        let total_df: f64 = data.iter().map(|p| p.df).sum();
        if total_df == 0.0 {
            let names: Vec<String> = (0..data.len()).map(name).collect();
            return Err(Error::Bayes(format!(
                "no degree of freedom to estimate the scale: the chain ladder fits every cell \
                 of {} exactly, as it does for a program of two origins or fewer. Add a \
                 program of three origins or more",
                names.join(", ")
            )));
        }
        let kappa = data
            .iter()
            .zip(&mean_premium)
            .map(|(p, m)| p.pearson / m)
            .sum::<f64>()
            / total_df;
        if !(kappa.is_finite() && kappa > 0.0) {
            return Err(Error::Bayes(format!(
                "the pooled scale per unit of premium is {kappa}: the chain ladders leave no \
                 scatter to estimate it from"
            )));
        }

        // The origin rule, with the program's own scale.
        let shape = self.cv.powi(-2);
        for (p, m) in data.iter().zip(&mean_premium) {
            let phi = kappa * m;
            for (w, origin) in p.observed.origins.iter().enumerate() {
                let row = p.observed.cells.iter().filter(|cell| cell.0 == w);
                let (n, sum) = row.fold((0, 0.0), |(n, s), cell| (n + 1, s + cell.2));
                if sum.is_nan() || shape + sum / phi <= 0.0 {
                    return Err(origin_error(origin, n, sum, shape, phi));
                }
            }
        }

        // The column rule, on the pooled column: a single program's column
        // that sums to zero or less is held by its deviation's prior, up to
        // the bound below. The longest program observes every age of the
        // grid, so every pooled column has cells.
        let mut column_sum = vec![0.0; nd];
        let mut column_n = vec![0; nd];
        for p in &data {
            for &(_, d, x) in &p.observed.cells {
                column_sum[d] += x;
                column_n[d] += 1;
            }
        }
        for d in 0..nd {
            if column_sum[d].is_nan() || column_sum[d] <= 0.0 {
                let alone = data.len() == 1;
                return Err(
                    match column_error(&ages, d, column_n[d], column_sum[d], None, &[], alone) {
                        Error::Bayes(why) => {
                            Error::Bayes(format!("summed over every program, {why}"))
                        }
                        other => other,
                    },
                );
            }
        }

        // A falling column gains `|S_cd| / phi_c` per unit that `omega
        // delta_cd` falls, and the deviations' priors charge at most `1 /
        // omega_scale` per unit of `|omega delta|` over every program
        // together, so past that bound the posterior has no finite mass
        // (`omega` runs away; decision 30, "Columns and origins").
        let mut falling = Vec::new();
        let mut squared = 0.0;
        for (c, (p, m)) in data.iter().zip(&mean_premium).enumerate() {
            let phi = kappa * m;
            let mut column = vec![0.0; nd];
            for &(_, d, x) in &p.observed.cells {
                column[d] += x;
            }
            let fall: f64 = column[1..].iter().filter(|s| **s < 0.0).sum();
            if column[0] + fall < 0.0 {
                return Err(Error::Bayes(format!(
                    "{}: its falling columns sum to {fall} and its first age to {}, so its \
                     pattern has no finite posterior. Correct the program's data or leave it out",
                    name(c),
                    column[0]
                )));
            }
            for (d, s) in column.iter().enumerate() {
                if *s < 0.0 {
                    squared += (s / phi).powi(2);
                    falling.push(format!("{} at {} months: {s}", name(c), ages[d]));
                }
            }
        }
        if squared.sqrt() * self.omega_scale >= 1.0 {
            return Err(Error::Bayes(format!(
                "the columns that fall ({}) come to {} in units of each program's scale, at or \
                 past 1 / omega_scale = {}: the posterior has no finite mass, as the programs' \
                 deviations would grow without bound. Correct those columns, or reduce \
                 omega_scale",
                falling.join("; "),
                squared.sqrt(),
                1.0 / self.omega_scale
            )));
        }

        let pooled = pooled_factors(&data, nd);
        let log_apriori = self.apriori.ln();
        let (level_mean, level_sd, level_free) = match self.level_cv {
            Some(c) => {
                let variance = (c * c).ln_1p();
                (log_apriori - 0.5 * variance, variance.sqrt(), false)
            }
            None => (log_apriori, 1.0, true),
        };
        let mut programs = Vec::with_capacity(data.len());
        let mut next = 0;
        for (p, m) in data.iter().zip(&mean_premium) {
            programs.push(Program {
                first: next,
                latest: p.observed.last.clone(),
                phi: kappa * m,
                log_prior: p.premium.iter().map(|e| log_apriori + e.ln()).collect(),
                cells: p.observed.cells.clone(),
            });
            next += p.premium.len();
        }
        let model = Model {
            programs,
            n_dev: nd,
            n_origins: next,
            shape,
            log_apriori,
            level_mean,
            level_sd,
            level_free,
            tau_precision: self.tau_scale.powi(-2),
            omega_precision: self.omega_scale.powi(-2),
            development_mean,
            development_precision,
        };
        let own_ultimate: Vec<Vec<f64>> = data.iter().map(|p| own_ultimates(p, &pooled)).collect();
        let own_loss_ratio: Vec<f64> = own_ultimate
            .iter()
            .zip(&data)
            .map(|(u, p)| u.iter().sum::<f64>() / p.premium.iter().sum::<f64>())
            .collect();
        let start = self.start(&model, &data, &own_ultimate, &own_loss_ratio, &pooled);
        let mut hasher = InputHasher::new();
        hasher.str(column);
        hasher.str(exposure);
        for (p, label) in data.iter().zip(&labels) {
            hasher.str(&label.parts().join("/"));
            hasher.f64s(&p.premium);
            for &(w, d, x) in &p.observed.cells {
                hasher.f64s(&[w as f64, d as f64, x]);
            }
        }
        Ok(Panel {
            labels,
            origins: data.iter().map(|p| p.observed.origins.clone()).collect(),
            latest: data.iter().map(|p| p.observed.latest.clone()).collect(),
            exposure: data.into_iter().map(|p| p.premium).collect(),
            ages,
            kappa,
            own_loss_ratio,
            model,
            start,
            input_hash: hasher.finish(),
            pattern: pattern.cloned(),
        })
    }

    /// The start from the data, on the sampling scale: the pooled pattern,
    /// `tau` and `omega` at their prior medians, the level between its
    /// hyperprior and the pooled loss ratio, each program at its own loss
    /// ratio and each origin between its level and its chain ladder
    /// ultimate, by the weights a gamma's update gives them.
    fn start(
        &self,
        model: &Model,
        data: &[ProgramData],
        own_ultimate: &[Vec<f64>],
        own_loss_ratio: &[f64],
        pooled: &[f64],
    ) -> Vec<f64> {
        let tau = HALF_NORMAL_MEDIAN * self.tau_scale;
        let omega = HALF_NORMAL_MEDIAN * self.omega_scale;
        let premium: f64 = data.iter().flat_map(|p| &p.premium).sum();
        let total: f64 = own_ultimate.iter().flatten().sum();
        let observed = if total > 0.0 {
            (total / premium).ln()
        } else {
            model.log_apriori
        };
        let log_level = if model.level_free {
            observed
        } else {
            let between = (self.cv * self.cv + self.tau_scale * self.tau_scale) / data.len() as f64;
            let prior = model.level_sd * model.level_sd;
            (between * model.level_mean + prior * observed) / (between + prior)
        };
        let mut theta = vec![0.0; model.dim()];
        theta[0] = (log_level - model.level_mean) / model.level_sd;
        theta[1] = tau.ln();
        // The cumulative shares from the last age back, then the increments.
        let mut cumulative = vec![1.0; model.n_dev];
        for d in (0..model.n_dev - 1).rev() {
            cumulative[d] = cumulative[d + 1] / pooled[d];
        }
        for d in 1..model.n_dev {
            let share = cumulative[d] - cumulative[d - 1];
            theta[model.beta(d)] = (share / cumulative[0]).ln();
        }
        theta[model.log_omega()] = omega.ln();
        for (c, program) in model.programs.iter().enumerate() {
            let ratio = own_loss_ratio[c];
            let z = if ratio > 0.0 {
                (ratio.ln() - log_level + 0.5 * tau * tau) / tau
            } else {
                0.0
            };
            let z = z.clamp(-2.0, 2.0);
            theta[model.z(c)] = z;
            let log_program = log_level + tau * z - 0.5 * tau * tau;
            for (w, &log_prior) in program.log_prior.iter().enumerate() {
                let sum: f64 = program
                    .cells
                    .iter()
                    .filter(|cell| cell.0 == w)
                    .map(|cell| cell.2)
                    .sum();
                let information = (sum / program.phi).max(0.0);
                let weight = information / (information + model.shape);
                let centre = log_program + log_prior - model.log_apriori;
                let own = own_ultimate[c][w];
                let alpha = if own > 0.0 {
                    centre + weight * (own.ln() - centre)
                } else {
                    centre
                };
                theta[model.u(program, w)] = alpha - log_prior;
            }
        }
        theta
    }

    fn provenance(&self, column: &str, exposure: &str, panel: &Panel) -> Provenance {
        let s = &self.sampler;
        let provenance = Provenance::new("hierarchical_bf")
            .param("apriori", self.apriori)
            .param("cv", self.cv);
        let provenance = match self.level_cv {
            Some(c) => provenance.param("level_cv", c),
            None => provenance,
        };
        let provenance = match &panel.pattern {
            Some(prior) => provenance
                .param("pattern_prior_mean", format!("{:?}", prior.mean))
                .param("pattern_prior_sd", format!("{:?}", prior.sd)),
            None => provenance,
        };
        provenance
            .param("development_sd", self.development_sd)
            .param("tau_scale", self.tau_scale)
            .param("omega_scale", self.omega_scale)
            .param("scale_per_premium", panel.kappa)
            .param("process", format!("{:?}", self.process))
            .param("chains", s.chains)
            .param("tune", s.tune)
            .param("draws", s.draws)
            .param("target_accept", s.target_accept)
            .param("max_depth", s.max_depth)
            .param("n_sims", self.n_sims)
            .param("seed", self.seed)
            .param("column", column)
            .param("exposure", exposure)
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(panel.input_hash.clone())
    }
}

impl HierarchicalBfFit {
    /// R-hat, ESS and quantiles of every sampled parameter.
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        Ok(self.posterior.summary()?)
    }

    /// The posterior of the hyperparameters on their natural scale: the
    /// industry loss ratio `level`, the spread `tau` of the programs' loss
    /// ratios and the spread `omega` of their patterns.
    pub fn hyperparameters(&self) -> Result<Vec<ParamSummary>> {
        let omega = self.posterior.index("log_omega").expect("a sampled name");
        let names = ["level", "tau", "omega"].map(String::from).to_vec();
        let draws = self
            .posterior
            .transform(names, |t| vec![t[0].exp(), t[1].exp(), t[omega].exp()])?;
        Ok(draws.summary()?)
    }
}

/// A segment's observed increments on its own positions, which are the
/// grid's: each program starts at the grid's first age.
#[derive(Debug, Clone)]
struct Observed {
    origins: Vec<Period>,
    /// (origin, development position, increment).
    cells: Vec<(usize, usize, f64)>,
    /// Each origin's last observed position.
    last: Vec<usize>,
    /// Each origin's cumulative values from the first age to its latest.
    cumulative: Vec<Vec<f64>>,
    /// Each origin's latest cumulative value.
    latest: Vec<f64>,
}

fn observed_cells(segment: &Segment) -> std::result::Result<Observed, String> {
    let mut observed = Observed {
        origins: segment.origins.clone(),
        cells: Vec::new(),
        last: Vec::with_capacity(segment.n_origins),
        cumulative: Vec::with_capacity(segment.n_origins),
        latest: Vec::with_capacity(segment.n_origins),
    };
    for w in 0..segment.n_origins {
        let Some(last) = (0..segment.n_dev)
            .rev()
            .find(|&d| segment.get(w, d).is_some())
        else {
            return Err(format!(
                "origin {} has no observed cell",
                segment.origins[w]
            ));
        };
        let mut row = Vec::with_capacity(last + 1);
        let mut previous = 0.0;
        for d in 0..=last {
            let Some(cum) = segment.get(w, d) else {
                return Err(format!(
                    "every origin must be observed from the first age to its latest, and \
                     origin {} has no cell at {} months",
                    segment.origins[w], segment.ages[d]
                ));
            };
            observed.cells.push((w, d, cum - previous));
            row.push(cum);
            previous = cum;
        }
        observed.last.push(last);
        observed.latest.push(previous);
        observed.cumulative.push(row);
    }
    Ok(observed)
}

/// One program's data before the pooled scale is known.
#[derive(Debug, Clone)]
struct ProgramData {
    premium: Vec<f64>,
    observed: Observed,
    /// The Pearson statistic of the program's chain ladder, `X2_c`.
    pearson: f64,
    /// Its degrees of freedom, `df_c`.
    df: f64,
}

/// The volume-weighted factor out of each position `0..nd - 1` over `rows`
/// of cumulative values; `None` where no row has the link or a sum is not
/// positive.
fn factors<'a>(rows: impl Iterator<Item = &'a Vec<f64>> + Clone, nd: usize) -> Vec<Option<f64>> {
    (0..nd - 1)
        .map(|d| {
            let linked = rows.clone().filter(|row| row.len() > d + 1);
            let (up, down) = linked.fold((0.0, 0.0), |(u, l), row| (u + row[d + 1], l + row[d]));
            (down > 0.0 && up > 0.0).then(|| up / down)
        })
        .collect()
}

/// The factors pooled over every program, each above 1 for a start's
/// positive shares.
fn pooled_factors(data: &[ProgramData], nd: usize) -> Vec<f64> {
    factors(data.iter().flat_map(|p| &p.observed.cumulative), nd)
        .into_iter()
        .map(|f| f.filter(|&f| f > 1.0).unwrap_or(FLAT_FACTOR))
        .collect()
}

/// Each origin's chain ladder ultimate: the program's own factors where it
/// has them, the pooled ones past them.
fn own_ultimates(program: &ProgramData, pooled: &[f64]) -> Vec<f64> {
    let own = factors(program.observed.cumulative.iter(), pooled.len() + 1);
    program
        .observed
        .latest
        .iter()
        .zip(&program.observed.last)
        .map(|(&latest, &last)| {
            let to_ultimate: f64 = (last..pooled.len())
                .map(|d| own[d].unwrap_or(pooled[d]))
                .product();
            latest * to_ultimate
        })
        .collect()
}

/// What the fit reads from the data, with the model.
struct Panel {
    labels: Vec<Label>,
    origins: Vec<Vec<Period>>,
    exposure: Vec<Vec<f64>>,
    latest: Vec<Vec<f64>>,
    ages: Vec<Lag>,
    kappa: f64,
    own_loss_ratio: Vec<f64>,
    model: Model,
    start: Vec<f64>,
    input_hash: String,
    pattern: Option<PatternPrior>,
}

impl Panel {
    /// The key of each program's origins, program by program: the label's
    /// parts and the origin.
    fn components(&self) -> Vec<ComponentKey> {
        self.labels
            .iter()
            .zip(&self.origins)
            .flat_map(|(label, origins)| {
                origins.iter().map(move |&origin| {
                    let mut key: ComponentKey = label
                        .parts()
                        .iter()
                        .map(|part| KeyValue::from(part.as_str()))
                        .collect();
                    key.push(origin.into());
                    key
                })
            })
            .collect()
    }
}

/// One program in the model.
#[derive(Debug, Clone)]
struct Program {
    /// The position of its first origin among every program's origins.
    first: usize,
    /// Each origin's last observed position.
    latest: Vec<usize>,
    /// The plugged-in `phi_c`.
    phi: f64,
    /// `log(apriori P_cw)` of each origin: the shift of `u_cw`.
    log_prior: Vec<f64>,
    /// Observed increments: (origin, development position, value).
    cells: Vec<(usize, usize, f64)>,
}

impl Program {
    fn n_origins(&self) -> usize {
        self.log_prior.len()
    }
}

/// The log posterior on `theta = (e, log tau, z_1..z_J, u_cw, beta_2..beta_m,
/// log omega, delta_cd)`, with `log L = level_mean + level_sd e` and `u_cw =
/// alpha_cw - log(apriori P_cw)`: a shift and a scale, so that a tight
/// hyperprior keeps full precision and a pilot's covariance, whose floor
/// is near 1e-3 of a unit, can resolve it. The draws are stored as
/// `(log L, log tau, z, alpha, beta, log omega, delta)`, an affine map of
/// this scale, so moment matching needs no Jacobian (decision 17).
#[derive(Debug, Clone)]
struct Model {
    programs: Vec<Program>,
    n_dev: usize,
    /// Every program's origins.
    n_origins: usize,
    /// Shape `1 / cv^2` of the gamma prior on each `U_cw`.
    shape: f64,
    /// `log apriori`.
    log_apriori: f64,
    /// The mean of `log L`, or `log apriori` under a flat hyperprior.
    level_mean: f64,
    /// The sd of `log L`, or 1 under a flat hyperprior.
    level_sd: f64,
    /// Whether the hyperprior is flat on `log L`.
    level_free: bool,
    tau_precision: f64,
    omega_precision: f64,
    /// The prior mean of each shared `beta_d`, `d = 2..m`.
    development_mean: Vec<f64>,
    /// The prior precision of each shared `beta_d`.
    development_precision: Vec<f64>,
}

impl Model {
    /// The position of `z_c`.
    fn z(&self, c: usize) -> usize {
        2 + c
    }

    /// The position of `u_cw`, or of `alpha_cw` on the stored scale.
    fn u(&self, program: &Program, w: usize) -> usize {
        2 + self.programs.len() + program.first + w
    }

    /// The position of `beta_d` at development position `d >= 1`.
    fn beta(&self, d: usize) -> usize {
        2 + self.programs.len() + self.n_origins + d - 1
    }

    fn log_omega(&self) -> usize {
        self.beta(self.n_dev)
    }

    /// The position of `delta_cd` at development position `d >= 1`.
    fn delta(&self, c: usize, d: usize) -> usize {
        self.log_omega() + 1 + c * (self.n_dev - 1) + d - 1
    }

    fn names(&self) -> Vec<String> {
        let mut names = vec!["log_level".to_string(), "log_tau".to_string()];
        names.extend((1..=self.programs.len()).map(|c| format!("z[{c}]")));
        for (c, program) in self.programs.iter().enumerate() {
            names.extend((1..=program.n_origins()).map(|w| format!("alpha[{},{w}]", c + 1)));
        }
        names.extend((2..=self.n_dev).map(|d| format!("beta[{d}]")));
        names.push("log_omega".to_string());
        for c in 1..=self.programs.len() {
            names.extend((2..=self.n_dev).map(|d| format!("delta[{c},{d}]")));
        }
        names
    }

    /// `(log L, ..., alpha, ...)` from the sampling scale.
    fn natural(&self, theta: &[f64]) -> Vec<f64> {
        let mut natural = theta.to_vec();
        natural[0] = self.level_mean + self.level_sd * theta[0];
        for program in &self.programs {
            for (w, l) in program.log_prior.iter().enumerate() {
                natural[self.u(program, w)] += l;
            }
        }
        natural
    }

    /// The sampling scale from a draw as stored: the inverse of
    /// [`natural`](Self::natural).
    fn sampling(&self, theta: &[f64]) -> Vec<f64> {
        let mut sampling = theta.to_vec();
        sampling[0] = (theta[0] - self.level_mean) / self.level_sd;
        for program in &self.programs {
            for (w, l) in program.log_prior.iter().enumerate() {
                sampling[self.u(program, w)] -= l;
            }
        }
        sampling
    }

    /// `log p_cd` at every position, from the stored or the sampling scale
    /// (the pattern block is the same).
    fn log_shares(&self, theta: &[f64], c: usize) -> Vec<f64> {
        let omega = theta[self.log_omega()].exp();
        let mut beta = vec![0.0; self.n_dev];
        for d in 1..self.n_dev {
            beta[d] = theta[self.beta(d)] + omega * theta[self.delta(c, d)];
        }
        let log_sum = beta.iter().map(|b| b.exp()).sum::<f64>().ln();
        for b in &mut beta {
            *b -= log_sum;
        }
        beta
    }

    /// Each origin's simulated reserve under the stored draw `theta`.
    fn simulate(
        &self,
        theta: &[f64],
        process: ProcessDistribution,
        rng: &mut StreamRng,
        row: &mut [f64],
    ) {
        for (c, program) in self.programs.iter().enumerate() {
            let log_share = self.log_shares(theta, c);
            for (w, &last) in program.latest.iter().enumerate() {
                let alpha = theta[self.u(program, w)];
                row[program.first + w] = (last + 1..self.n_dev)
                    .map(|d| process_draw(process, program.phi, (alpha + log_share[d]).exp(), rng))
                    .sum();
            }
        }
    }
}

impl Model {
    /// The log posterior, written once over the number type (decision 15).
    fn log_density<S: Scalar>(&self, theta: &[S]) -> S {
        self.log_density_with(theta, None)
    }

    /// [`log_density`](Self::log_density) with a year's cells appended to
    /// the observed ones, in the order [`HierarchicalYear::extended`] puts
    /// them, so no copy of the model is needed.
    fn log_density_with<S: Scalar>(
        &self,
        theta: &[S],
        year: Option<(&[ProgramYear], &[f64])>,
    ) -> S {
        let mut f = S::from_f64(0.0);
        let e = theta[0];
        if !self.level_free {
            f -= e * e * 0.5;
        }
        let shift = e * self.level_sd + S::from_f64(self.level_mean - self.log_apriori);
        // The half-normals on log tau and log omega, with their Jacobians.
        let log_tau = theta[1];
        let tau = log_tau.exp();
        f += log_tau - tau * tau * (0.5 * self.tau_precision);
        let log_omega = theta[self.log_omega()];
        let omega = log_omega.exp();
        f += log_omega - omega * omega * (0.5 * self.omega_precision);
        for d in 1..self.n_dev {
            let centred = theta[self.beta(d)] - S::from_f64(self.development_mean[d - 1]);
            f -= centred * centred * (0.5 * self.development_precision[d - 1]);
        }
        let half_tau_squared = tau * tau * 0.5;
        let mut beta = vec![S::from_f64(0.0); self.n_dev];
        for (c, program) in self.programs.iter().enumerate() {
            let z = theta[self.z(c)];
            f -= z * z * 0.5;
            let log_s = tau * z - half_tau_squared;
            let mut sum = S::from_f64(1.0);
            for d in 1..self.n_dev {
                let delta = theta[self.delta(c, d)];
                f -= delta * delta * 0.5;
                beta[d] = theta[self.beta(d)] + omega * delta;
                sum += beta[d].exp();
            }
            let log_sum = sum.ln();
            // As BayesBfNormalised: the gamma on U with its Jacobian, less
            // its value at the mean, in v = alpha - log(L s P).
            for w in 0..program.n_origins() {
                let v = theta[self.u(program, w)] - shift - log_s;
                f += (v - v.exp_m1()) * self.shape;
            }
            for &(w, d, x) in &program.cells {
                let eta = theta[self.u(program, w)] + program.log_prior[w] + beta[d] - log_sum;
                f += (eta * x - eta.exp()) / program.phi;
            }
            if let Some((programs, values)) = year {
                let py = &programs[c];
                for (k, &(w, d)) in py.year.iter().enumerate() {
                    let x = values[py.first_cell + k];
                    let eta = theta[self.u(program, w)] + program.log_prior[w] + beta[d] - log_sum;
                    f += (eta * x - eta.exp()) / program.phi;
                }
            }
        }
        f
    }
}

impl Posterior for Model {
    fn dim(&self) -> usize {
        2 + self.programs.len()
            + self.n_origins
            + self.n_dev
            + self.programs.len() * (self.n_dev - 1)
    }

    fn log_posterior(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        gradient::<CHUNK>(theta, grad, |x| self.log_density(x))
    }
}

impl HierarchicalBf {
    /// The most one-year simulations [`one_year`](Self::one_year) runs by
    /// default (`BayesOneYear::n_sims` `None`): 2,000, against the 10,000 of
    /// the lifetime view, as Meyers' and the compartmental models', since
    /// every simulation can cost a refit of the whole panel (on the
    /// three-program reference panel, release, 16 threads: about 1.5 s each
    /// with the default light refit, 13 s with the fit's own sampler at its
    /// defaults and 1.2 s at 2 chains of 200 warm-up and 200 kept draws).
    pub const ONE_YEAR_SIMS: usize = 2_000;

    /// Whether [`one_year`](Self::one_year) refits with the light refit of
    /// [`LightRefit::default`] when `settings.refit` and
    /// `settings.light_refit` are `None` and `settings.no_light_refit` is
    /// `false`: `true`, by the user's decision of 2026-11-09 (issue #762),
    /// once it was measured within about 0.05 sd of full refits on the
    /// reference panel. On EVW, GenIns and RAA, at 40 simulations a seed
    /// with every one refitted, its total claims development result is
    /// within 0.004 sd (rms) of the full refit's at seeds 0 and 1, at about
    /// 1.5 s a refit against 13 s for the fit's own sampler (release, 16
    /// threads); none of its 80 refits warned.
    pub const LIGHT_REFIT_BY_DEFAULT: bool = true;

    /// `settings` with this model's one-year defaults: the refits' sampler,
    /// as [`light_refit_of`](Self::light_refit_of) resolves it at
    /// [`LIGHT_REFIT_BY_DEFAULT`](Self::LIGHT_REFIT_BY_DEFAULT); and the
    /// number of one-year simulations, the settings' or this model's
    /// `n_sims` capped at [`ONE_YEAR_SIMS`](Self::ONE_YEAR_SIMS).
    fn one_year_defaults(&self, settings: &BayesOneYear) -> (BayesOneYear, usize) {
        let mut resolved = *settings;
        resolved.light_refit = Self::light_refit_of(settings, Self::LIGHT_REFIT_BY_DEFAULT);
        (
            resolved,
            settings.sims(self.n_sims.min(Self::ONE_YEAR_SIMS)),
        )
    }

    /// The refits' sampler under `settings`, as `Compartmental`'s view
    /// resolves it (decision 24): the `light_refit` given; else, when the
    /// settings choose no refit mode (`refit` `None`), the light refit of
    /// [`LightRefit::default`] if `by_default` holds and `no_light_refit` is
    /// not set; else none, the fit's own sampler. A refit mode the settings
    /// choose keeps the fit's own sampler unless a light refit is given.
    /// None when the view reweights only, so that the provenance records
    /// only a sampler that runs.
    fn light_refit_of(settings: &BayesOneYear, by_default: bool) -> Option<LightRefit> {
        if settings.mode() == OneYearRefit::ReweightOnly {
            return None;
        }
        let default = (by_default && settings.refit.is_none() && !settings.no_light_refit)
            .then(LightRefit::default);
        settings.light_refit.or(default)
    }

    /// The one-year view of the fit over every program at once
    /// (`reserving-bayes.md`, decisions 17 and 30): the claims development
    /// result of each program and origin over the coming year, by
    /// reweighting the posterior by the year's cells as
    /// [`BayesOdp::one_year`](super::BayesOdp::one_year) does, with the
    /// flagged simulations moment-matched and then refitted as `settings`
    /// says. It is one view over the stacked programs and not one per
    /// program, because a program's next diagonal moves the industry level,
    /// the spreads and the shared pattern, and through them the other
    /// programs' closing reserves. A refit samples the whole panel: as
    /// [`fit`](Self::fit) does (a pilot run, then the fit's sampler), or,
    /// with `settings.light_refit`, or by default (the settings choose no
    /// refit mode and `settings.no_light_refit` is not set; see
    /// [`LIGHT_REFIT_BY_DEFAULT`](Self::LIGHT_REFIT_BY_DEFAULT)), warm-started
    /// as `Compartmental`'s light refit (decision 24): no pilot run, the
    /// sampler scaled by the opening posterior's draws (by their variances
    /// below 8 programs, their covariance from 8, as the fit's main run) and
    /// started at their mean, with the light refit's chains, warm-up and
    /// draws, and a warning judged on the closing reserve's R-hat (above
    /// 1.05), effective draws (under 100) and divergences.
    ///
    /// The view uses the fit's seed. It runs at most
    /// [`ONE_YEAR_SIMS`](Self::ONE_YEAR_SIMS) simulations unless
    /// `settings.n_sims` sets their number. A program whose ages stop short
    /// of the panel's grid (EVW at 6 of 10 on the reference panel) has its
    /// next diagonal at ages it has never observed, where only the prior
    /// speaks. On the reference panel every simulation has a Pareto k above
    /// 6 and is refitted (matching rescues none); a control with copies of
    /// GenIns and RAA shows that a program stopping short raises k, though
    /// not to that size. On a panel whose programs share their ages, k is
    /// mostly below 0.7 and the reweighting is close to the refits. Each
    /// refit is a NUTS run over the panel. The view's
    /// [`warnings`](BayesOneYearFit::warnings) name the programs whose next
    /// diagonal falls at ages they have never observed (the user's decision
    /// of 2026-11-09, issue #762): the view still runs, but is expensive (or,
    /// reweighting only, approximate).
    /// Failed diagnostics of the fit do not fail the view; they are in
    /// `fit.diagnostics`.
    ///
    /// ```
    /// use prospicio_bayes::nuts::Sampler;
    /// use prospicio_reserving::bayes::{BayesOneYear, HierarchicalBf};
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
    /// use prospicio_prob::Distribution;
    ///
    /// let program = ["a", "a", "a", "a", "a", "a", "b", "b", "b"];
    /// let origin = [2021, 2021, 2021, 2022, 2022, 2023, 2022, 2022, 2023].map(Month::january);
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[("program", &program)],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12, 12, 24, 12]),
    ///     values: &[
    ///         ("paid", &[100.0, 150.0, 170.0, 110.0, 172.0, 130.0, 55.0, 80.0, 60.0]),
    ///         ("premium", &[250.0, 250.0, 250.0, 270.0, 270.0, 290.0, 120.0, 120.0, 130.0]),
    ///     ],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })?;
    /// let model = HierarchicalBf {
    ///     apriori: 0.7,
    ///     level_cv: Some(0.2),
    ///     sampler: Sampler { tune: 200, draws: 200, chains: 2, ..Sampler::default() },
    ///     n_sims: 500,
    ///     ..HierarchicalBf::default()
    /// };
    /// let view = model.one_year(&tri, "paid", "premium", &BayesOneYear::default())?;
    /// // One component per program and origin, 5 in all.
    /// assert_eq!(view.cdr.dims(), ["program", "origin"]);
    /// assert_eq!(view.cdr.n_components(), 5);
    /// // A program's total is the sum of its origins.
    /// let totals = view.cdr.aggregate(&["program"])?;
    /// assert_eq!(totals.n_components(), 2);
    /// assert!(view.cdr.std_dev() > 0.0);
    /// # Ok::<(), prospicio_reserving::Error>(())
    /// ```
    pub fn one_year(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
        settings: &BayesOneYear,
    ) -> Result<BayesOneYearFit<HierarchicalBfFit>> {
        self.one_year_with(triangle, column, exposure, settings, None)
    }

    /// [`one_year`](Self::one_year) under a pattern prior, as
    /// [`fit_with_pattern_prior`](Self::fit_with_pattern_prior): the
    /// reweighting and any refit keep the prior.
    pub fn one_year_with_pattern_prior(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
        settings: &BayesOneYear,
        prior: &PatternPrior,
    ) -> Result<BayesOneYearFit<HierarchicalBfFit>> {
        self.one_year_with(triangle, column, exposure, settings, Some(prior))
    }

    fn one_year_with(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
        settings: &BayesOneYear,
        pattern: Option<&PatternPrior>,
    ) -> Result<BayesOneYearFit<HierarchicalBfFit>> {
        self.validate()?;
        settings.validate()?;
        let panel = self.prepare(triangle, column, exposure, pattern)?;
        let fit = self.fit_panel(triangle, column, exposure, &panel)?;
        let segments = triangle.segments(column)?;
        let start = panel.model.sampling(&fit.posterior.mean());
        let mut year = HierarchicalYear::new(self, &panel.model, triangle, &segments, start);
        let (settings, n_sims) = self.one_year_defaults(settings);
        let settings = &settings;
        year.warm_start(settings.light_refit, &fit.posterior)?;
        let warnings: Vec<String> = year
            .unobserved_ages(&panel.labels, &panel.ages, settings.mode())
            .into_iter()
            .collect();
        let draws = one_year::simulate(
            &year,
            &fit.posterior,
            settings,
            n_sims,
            one_year_seed(derive_seed(self.seed, Stream::Predictive)),
        )?;
        let mut provenance = self.provenance(column, exposure, &panel);
        provenance.model = "hierarchical_bf_one_year".into();
        let provenance = one_year::provenance_with_light_refit(provenance, settings, n_sims);
        let cdr = PredictiveDistribution::from_draws(
            Self::dims(triangle),
            panel.components(),
            draws.cdr,
            provenance,
        )?;
        let latest: Vec<f64> = panel.latest.iter().flatten().copied().collect();
        Ok(BayesOneYearFit {
            opening_ultimate: latest
                .iter()
                .zip(&draws.opening)
                .map(|(l, r)| l + r)
                .collect(),
            opening_reserve: draws.opening,
            fit,
            cdr,
            pareto_k: draws.pareto_k,
            refitted: draws.refitted,
            moment_matched: draws.moment_matched,
            unrefitted: draws.unrefitted,
            refit_warnings: draws.refit_warnings,
            k_threshold: settings.k_threshold,
            approximate: None,
            warnings,
        })
    }
}

/// The one-year view of a [`HierarchicalBf`] over every program
/// ([`HierarchicalBf::one_year`]): the lifetime fit, the opening reserve,
/// the claims development result and how reliable each simulation's
/// reweighting was, as for the other Bayesian models. Per-origin lists
/// (`opening_reserve`, `opening_ultimate`) run over every program's
/// origins in turn, program by program, as the components of
/// [`cdr`](BayesOneYearFit::cdr) and of the fit's joint
/// [`reserves`](HierarchicalBfFit::reserves) do. A program's claims
/// development result is the sum of its origins' components,
/// `cdr.aggregate(&["program"])`.
impl BayesOneYearFit<HierarchicalBfFit> {
    /// The share of simulations whose Pareto `k` is above the threshold
    /// (or NaN), whether or not they were refitted.
    pub fn flagged_share(&self) -> f64 {
        one_year::flagged_share(&self.pareto_k, self.k_threshold)
    }
}

/// The coming year of one program: its cells by position among the year's
/// values, and the cells that remain before and after it.
#[derive(Debug, Clone)]
struct ProgramYear {
    /// The position of the program's first cell among the year's values.
    first_cell: usize,
    /// The year's cells, `(origin, position)`, in origin and then
    /// development order.
    year: Vec<(usize, usize)>,
    /// The cells after the latest diagonal: the opening reserve.
    future: Vec<(usize, usize)>,
    /// The cells after the year: the closing reserve.
    after: Vec<(usize, usize)>,
}

/// A fitted [`HierarchicalBf`] as the one-year view sees it
/// ([`YearModel`], decisions 17 and 30): one model over every program's
/// origins, program by program. The draws are stored as `(log L, log tau,
/// z, alpha, beta, log omega, delta)`, an affine map of the sampling scale
/// (`log L` is `level_mean + level_sd e` and `alpha` is `u` shifted by
/// `log(apriori P)`), so the log posterior needs a constant Jacobian only,
/// which a ratio of densities cancels, and moment matching moves
/// unconstrained values. Every shared effect and every program's
/// deviations (including those of ages it never observes) are in the draw,
/// so a cell's mean is read from it alone; `L`, `tau` and `z` enter the
/// prior only and travel with the draw. A refit starts at the opening
/// posterior mean moved to the sampling scale, and the plugged-in scales
/// `phi_c` stay at their opening values.
struct HierarchicalYear<'a> {
    bf: &'a HierarchicalBf,
    model: &'a Model,
    /// The opening posterior mean on the sampling scale.
    start: Vec<f64>,
    programs: Vec<ProgramYear>,
    n_year: usize,
    /// A warm-started refit's sampler and the opening draws on the sampling
    /// scale that scale it ([`warm_start`](Self::warm_start)).
    light: Option<(LightRefit, PosteriorDraws)>,
}

impl<'a> HierarchicalYear<'a> {
    fn new(
        bf: &'a HierarchicalBf,
        model: &'a Model,
        triangle: &Triangle,
        segments: &[Segment],
        start: Vec<f64>,
    ) -> Self {
        let nd = model.n_dev;
        let mut n_year = 0;
        let programs = model
            .programs
            .iter()
            .zip(segments)
            .map(|(program, segment)| {
                let valuation = |w: usize, d: usize| {
                    triangle.valuation_of(w + segment.origin_offset, d + segment.dev_offset)
                };
                let latest = &program.latest;
                let opening = latest
                    .iter()
                    .enumerate()
                    .map(|(w, &d)| valuation(w, d))
                    .max()
                    .expect("a program has an origin");
                let closing = opening.add_months(12);
                // Where each origin stands after the year: its cells
                // valued by then, if any is valued after the opening (no
                // tail, as `OdpYear`). The grid, not the segment, bounds
                // the ages: a thin program carries the later ones too.
                let last: Vec<usize> = latest
                    .iter()
                    .enumerate()
                    .map(|(w, &d)| {
                        let last = (d + 1..nd)
                            .take_while(|&e| valuation(w, e) <= closing)
                            .last()
                            .unwrap_or(d);
                        if (d + 1..=last).any(|e| valuation(w, e) > opening) {
                            last
                        } else {
                            d
                        }
                    })
                    .collect();
                let between = |from: &dyn Fn(usize) -> usize, to: &dyn Fn(usize) -> usize| {
                    (0..latest.len())
                        .flat_map(|w| (from(w) + 1..=to(w)).map(move |d| (w, d)))
                        .collect::<Vec<_>>()
                };
                let year = between(&|w| latest[w], &|w| last[w]);
                let first_cell = n_year;
                n_year += year.len();
                ProgramYear {
                    first_cell,
                    year,
                    future: between(&|w| latest[w], &|_| nd - 1),
                    after: between(&|w| last[w], &|_| nd - 1),
                }
            })
            .collect();
        Self {
            bf,
            model,
            start,
            programs,
            n_year,
            light: None,
        }
    }

    /// Refit as `light` says, when it is `Some`, scaled by `opening`, the
    /// opening posterior as stored.
    fn warm_start(&mut self, light: Option<LightRefit>, opening: &PosteriorDraws) -> Result<()> {
        if let Some(l) = light {
            let sampled = opening.transform(self.model.names(), |x| self.model.sampling(x))?;
            self.light = Some((l, sampled));
        }
        Ok(())
    }

    /// The warning of a panel whose programs stop at different ages, or
    /// `None` (the user's decision of 2026-11-09, issue #762): the programs
    /// (by `labels`) whose year has a cell at a development age (of `ages`)
    /// past the last they observe. Only the prior speaks there, so nearly
    /// every simulation is flagged. Under a refit `mode` the warning says
    /// that they are refitted over the whole panel and the view is
    /// expensive; under [`OneYearRefit::ReweightOnly`] that the weights are
    /// unreliable and the view approximate.
    fn unobserved_ages(
        &self,
        labels: &[Label],
        ages: &[Lag],
        mode: OneYearRefit,
    ) -> Option<String> {
        let short: Vec<String> = self
            .model
            .programs
            .iter()
            .zip(&self.programs)
            .zip(labels)
            .filter_map(|((program, year), label)| {
                let last = program.latest.iter().copied().max()?;
                let next = year.year.iter().map(|&(_, d)| d).max()?;
                (next > last).then(|| {
                    format!(
                        "{label} (observed to age {}, its next diagonal to {})",
                        ages[last], ages[next]
                    )
                })
            })
            .collect();
        (!short.is_empty()).then(|| {
            let consequence = if mode == OneYearRefit::ReweightOnly {
                "nearly every one-year simulation is flagged (Pareto k above the threshold), and \
                 since the view reweights only, its weights are unreliable and the view is \
                 approximate (a refit mode refits them, at a cost)"
            } else {
                "nearly every one-year simulation is flagged and refitted over the whole panel, \
                 and the view is expensive"
            };
            format!(
                "programs whose next diagonal falls at development ages they have never \
                 observed: {}. Only the prior speaks for those ages, so {consequence}",
                short.join("; ")
            )
        })
    }

    /// The sum of `exp(eta)` over each program's `cells`, by origin.
    fn expected(
        &self,
        theta: &[f64],
        cells: impl Fn(&ProgramYear) -> &[(usize, usize)],
        reserve: &mut [f64],
    ) {
        reserve.fill(0.0);
        for (c, (program, year)) in self.model.programs.iter().zip(&self.programs).enumerate() {
            let log_share = self.model.log_shares(theta, c);
            for &(w, d) in cells(year) {
                reserve[program.first + w] +=
                    (theta[self.model.u(program, w)] + log_share[d]).exp();
            }
        }
    }

    /// The model with the year's cells added to the observed ones.
    fn extended(&self, year: &[f64]) -> Model {
        let mut extended = self.model.clone();
        for (program, py) in extended.programs.iter_mut().zip(&self.programs) {
            for (k, &(w, d)) in py.year.iter().enumerate() {
                program.cells.push((w, d, year[py.first_cell + k]));
            }
        }
        extended
    }
}

impl YearModel for HierarchicalYear<'_> {
    fn n_origins(&self) -> usize {
        self.model.n_origins
    }

    fn n_year(&self) -> usize {
        self.n_year
    }

    fn draw_year(&self, theta: &[f64], rng: &mut StreamRng, year: &mut [f64]) {
        for (c, (program, py)) in self.model.programs.iter().zip(&self.programs).enumerate() {
            let log_share = self.model.log_shares(theta, c);
            for (k, &(w, d)) in py.year.iter().enumerate() {
                let mu = (theta[self.model.u(program, w)] + log_share[d]).exp();
                year[py.first_cell + k] = process_draw(self.bf.process, program.phi, mu, rng);
            }
        }
    }

    fn year_log_likelihood(&self, theta: &[f64], year: &[f64]) -> f64 {
        let mut total = 0.0;
        for (c, (program, py)) in self.model.programs.iter().zip(&self.programs).enumerate() {
            let log_share = self.model.log_shares(theta, c);
            let mut part = 0.0;
            for (k, &(w, d)) in py.year.iter().enumerate() {
                let eta = theta[self.model.u(program, w)] + log_share[d];
                part += year[py.first_cell + k] * eta - eta.exp();
            }
            total += part / program.phi;
        }
        total
    }

    fn paid(&self, year: &[f64], paid: &mut [f64]) {
        paid.fill(0.0);
        for (program, py) in self.model.programs.iter().zip(&self.programs) {
            for (k, &(w, _)) in py.year.iter().enumerate() {
                paid[program.first + w] += year[py.first_cell + k];
            }
        }
    }

    fn opening(&self, theta: &[f64], reserve: &mut [f64]) {
        self.expected(theta, |p| &p.future, reserve);
    }

    fn closing(&self, theta: &[f64], _year: &[f64], reserve: &mut [f64]) {
        self.expected(theta, |p| &p.after, reserve);
    }

    fn closing_uses_year(&self) -> bool {
        false
    }

    /// Sampled as [`HierarchicalBf::fit`] samples (the pilot, the diagonal
    /// scale below 8 programs and the fit's `target_accept`), or with a
    /// light refit warm-started from the opening posterior.
    fn refit(&self, year: &[f64], seed: u64) -> Result<(PosteriorDraws, bool)> {
        let model = self.extended(year);
        let Some((light, opening)) = &self.light else {
            let (posterior, diagnostics) = self.bf.sample(&model, self.start.clone(), seed)?;
            return Ok((posterior, !diagnostics.warnings.is_empty()));
        };
        // Warm start: the opening posterior's draws scale the sampler (no
        // pilot run), as the fit's main run is scaled by its pilot's, and it
        // starts at their mean with a shorter adaptation and fewer draws.
        let whitened = if model.programs.len() < DENSE_FROM_PROGRAMS {
            Whitened::from_draws_diagonal(&model, opening)?
        } else {
            Whitened::from_draws(&model, opening)?
        };
        let draws = sample(
            &whitened,
            &vec![0.0; model.dim()],
            Sampler {
                chains: light.chains,
                tune: light.tune,
                draws: light.draws,
                seed,
                ..self.bf.sampler
            },
        )?;
        let posterior = draws.transform(model.names(), |z| model.natural(&whitened.theta(z)))?;
        // As `Compartmental`'s light refit: the warning is about what the
        // view uses of a refit, its closing reserve (the total over every
        // program's origins), and the divergences.
        let n = self.model.n_origins;
        let reserve = posterior.transform(vec!["closing_reserve".into()], |x| {
            let mut row = vec![0.0; n];
            self.closing(x, year, &mut row);
            vec![row.iter().sum()]
        })?;
        let divergent = posterior.divergences().iter().sum::<usize>() > 0;
        // A run too short to diagnose has no summary, and that warns too.
        let warned = divergent
            || reserve.summary().map_or(true, |summary| {
                summary.iter().any(|s| {
                    s.rhat.is_nan()
                        || s.rhat > LIGHT_MAX_RHAT
                        || s.ess_bulk.is_nan()
                        || s.ess_bulk < LIGHT_MIN_ESS
                        || s.ess_tail.is_nan()
                        || s.ess_tail < LIGHT_MIN_ESS
                })
            });
        Ok((posterior, warned))
    }

    fn log_posterior(&self, theta: &[f64], year: &[f64]) -> f64 {
        let sampling = self.model.sampling(theta);
        if year.is_empty() {
            // Moment matching's many calls on the observed triangle.
            self.model.log_density(&sampling)
        } else {
            self.model
                .log_density_with(&sampling, Some((&self.programs, year)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bayes::BayesBfNormalised;
    use crate::bayes::laplace::tests::assert_gradient;
    use crate::bootstrap::component_sums;
    use crate::triangle::tests::GENINS;
    use crate::{DevelopmentColumn, Grain, Long};
    use prospicio_bayes::nuts::LogDensity;
    use prospicio_core::Month;
    use prospicio_prob::Distribution;

    /// One program: its name, first origin year, cumulative rows and
    /// premium per origin.
    type Book<'a> = (&'a str, i32, &'a [&'a [f64]], &'a [f64]);

    /// A cumulative annual triangle with a segment per program of `books`.
    fn panel(books: &[Book]) -> Triangle {
        let (mut key, mut origin, mut ages) = (vec![], vec![], vec![]);
        let (mut values, mut premium) = (vec![], vec![]);
        for &(name, first_year, rows, premiums) in books {
            for (k, row) in rows.iter().enumerate() {
                for (d, &v) in row.iter().enumerate() {
                    key.push(name);
                    origin.push(Month::january(first_year + k as i32));
                    ages.push(12 * (d as Lag + 1));
                    values.push(v);
                    premium.push(premiums[k]);
                }
            }
        }
        Triangle::from_long(&Long {
            keys: &[("program", &key)],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("values", &values), ("premium", &premium)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    /// A program of four origins, one of three and a thin one of two.
    fn three_programs() -> Triangle {
        panel(&[
            (
                "a",
                2020,
                &[
                    &[100.0, 160.0, 185.0, 196.0],
                    &[110.0, 170.0, 201.0],
                    &[95.0, 158.0],
                    &[120.0],
                ],
                &[250.0, 260.0, 240.0, 280.0],
            ),
            (
                "b",
                2021,
                &[&[210.0, 330.0, 395.0], &[190.0, 320.0], &[240.0]],
                &[400.0, 420.0, 450.0],
            ),
            ("c", 2022, &[&[42.0, 70.0], &[50.0]], &[90.0, 100.0]),
        ])
    }

    fn genins() -> Triangle {
        let rows: Vec<&[f64]> = GENINS.iter().map(|row| &row[..]).collect();
        let premium: Vec<f64> = (0..10).map(|k| 1e7 + 4e5 * k as f64).collect();
        panel(&[("genins", 2001, &rows, &premium)])
    }

    fn quick(seed: u64) -> HierarchicalBf {
        HierarchicalBf {
            apriori: 0.7,
            level_cv: Some(0.3),
            sampler: Sampler {
                tune: 300,
                draws: 300,
                target_accept: 0.95,
                ..Sampler::default()
            },
            n_sims: 1_000,
            seed,
            ..HierarchicalBf::default()
        }
    }

    fn prepared(tri: &Triangle, model: &HierarchicalBf) -> Panel {
        model
            .prepare(tri, "values", "premium", None)
            .map_err(|e| e.to_string())
            .unwrap()
    }

    /// Deterministic points about `centre`.
    fn near(centre: &[f64], k: usize, spread: f64) -> Vec<f64> {
        let wave = |j: usize| ((j * 7 + k * 13 + 1) as f64 * 0.731).sin();
        centre
            .iter()
            .enumerate()
            .map(|(j, x)| x + spread * wave(j))
            .collect()
    }

    #[test]
    fn the_layout_counts_every_parameter() {
        let p = prepared(&three_programs(), &quick(1));
        let m = &p.model;
        // 2 + J + origins + (m - 1) + 1 + J (m - 1), design note decision 30.
        assert_eq!(m.dim(), 2 + 3 + 9 + 3 + 1 + 3 * 3);
        let names = m.names();
        assert_eq!(names.len(), m.dim());
        assert_eq!(names[m.z(2)], "z[3]");
        assert_eq!(names[m.u(&m.programs[1], 2)], "alpha[2,3]");
        assert_eq!(names[m.beta(3)], "beta[4]");
        assert_eq!(names[m.log_omega()], "log_omega");
        assert_eq!(names[m.delta(2, 1)], "delta[3,2]");
        assert_eq!(names[m.dim() - 1], "delta[3,4]");
    }

    #[test]
    fn the_gradient_matches_finite_differences() {
        for model in [
            quick(1),
            HierarchicalBf {
                level_cv: None,
                ..quick(1)
            },
        ] {
            let p = prepared(&three_programs(), &model);
            for k in 0..3 {
                let theta = near(&p.start, k, 0.2);
                assert_gradient(|t, g| p.model.log_posterior(t, g), &theta);
            }
        }
    }

    #[test]
    fn one_program_at_zero_spreads_is_the_normalised_model() {
        // Validation item 1: tau and omega at 0 (a log of -800 underflows
        // to exactly 0), L at apriori (the flat hyperprior's e = 0), z and
        // delta held: their terms are a constant.
        let tri = genins();
        let hierarchical = HierarchicalBf {
            apriori: 0.6,
            cv: 0.2,
            development_sd: 7.0,
            ..HierarchicalBf::default()
        };
        let flat = BayesBfNormalised {
            apriori: 0.6,
            cv: 0.2,
            development_sd: 7.0,
            ..BayesBfNormalised::default()
        };
        let p = prepared(&tri, &hierarchical);
        let m = &p.model;
        let segment = tri.segments("values").unwrap().pop().unwrap();
        let exposure = tri.segments("premium").unwrap().pop().unwrap();
        let premium = origin_exposure(&segment, &exposure, "premium").unwrap();
        let (_, normalised, start) = flat.prepare(&segment, &premium).unwrap();

        let (no, nd) = (10, 10);
        let mut difference = None;
        for k in 0..20 {
            let point = near(&start, k, 0.3);
            let mut theta = vec![0.0; m.dim()];
            theta[1] = -800.0;
            theta[m.log_omega()] = -800.0;
            theta[m.z(0)] = 0.3;
            for d in 1..nd {
                theta[m.delta(0, d)] = 0.1 * d as f64;
                theta[m.beta(d)] = point[no + d - 1];
            }
            for w in 0..no {
                theta[m.u(&m.programs[0], w)] = point[w];
            }
            let mut grad = vec![0.0; m.dim()];
            let ours = m.log_posterior(&theta, &mut grad);
            let mut grad = vec![0.0; point.len()];
            let theirs = normalised.log_posterior(&point, &mut grad);
            let gap = ours - theirs;
            let first = *difference.get_or_insert(gap);
            assert!(
                (gap - first).abs() <= 1e-10 * theirs.abs().max(1.0),
                "point {k}: {gap} against {first}"
            );
        }
    }

    /// The prior of one origin with no cells, with `log tau` held: `(z, u)`.
    struct HeldTau {
        model: Model,
        log_tau: f64,
    }

    impl HeldTau {
        fn full(&self, x: &[f64]) -> Vec<f64> {
            let m = &self.model;
            let mut theta = vec![0.0; m.dim()];
            theta[1] = self.log_tau;
            theta[m.z(0)] = x[0];
            theta[m.u(&m.programs[0], 0)] = x[1];
            theta
        }
    }

    impl LogDensity for HeldTau {
        fn dim(&self) -> usize {
            2
        }

        fn log_density(&self, x: &[f64], gradient: &mut [f64]) -> Option<f64> {
            let m = &self.model;
            let mut grad = vec![0.0; m.dim()];
            let f = m.log_posterior(&self.full(x), &mut grad);
            gradient[0] = grad[m.z(0)];
            gradient[1] = grad[m.u(&m.programs[0], 0)];
            Some(f)
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_prior_ultimate_is_the_lognormal_gamma_mixture() {
        // Validation item 1, under (A1) with tau above 0: E[U / m] = 1 and
        // E[(U / m)^2] = (1 + cv^2) exp(tau^2).
        let (cv, tau) = (0.2_f64, 0.4_f64);
        let held = HeldTau {
            model: Model {
                programs: vec![Program {
                    first: 0,
                    latest: vec![0],
                    phi: 1.0,
                    log_prior: vec![0.0],
                    cells: vec![],
                }],
                n_dev: 2,
                n_origins: 1,
                shape: cv.powi(-2),
                log_apriori: 0.0,
                level_mean: 0.0,
                level_sd: 1.0,
                level_free: true,
                tau_precision: 1.0,
                omega_precision: 1.0,
                development_mean: vec![0.0],
                development_precision: vec![1.0],
            },
            log_tau: tau.ln(),
        };
        let draws = sample(
            &held,
            &[0.0, 0.0],
            Sampler {
                tune: 1000,
                draws: 4000,
                seed: 5,
                ..Sampler::default()
            },
        )
        .unwrap();
        let u: Vec<f64> = draws.values().chunks_exact(2).map(|x| x[1].exp()).collect();
        let n = u.len() as f64;
        let mean = u.iter().sum::<f64>() / n;
        let second = u.iter().map(|x| x * x).sum::<f64>() / n;
        assert!((mean - 1.0).abs() < 0.02, "{mean}");
        let exact = (1.0 + cv * cv) * (tau * tau).exp();
        assert!(
            (second - exact).abs() < 0.06 * exact,
            "{second} against {exact}"
        );
    }

    #[test]
    fn the_scale_is_pooled_per_unit_of_premium() {
        let tri = three_programs();
        let p = prepared(&tri, &quick(1));
        let segments = tri.segments("values").unwrap();
        // Program a has 10 cells and 7 parameters, b has 6 and 5, c none.
        let pearson = |s: &Segment, df: f64| {
            let (fit, _) = prepare(s, &s.ages, Tail::default()).unwrap();
            fit.scale * df
        };
        let (a, b) = (pearson(&segments[0], 3.0), pearson(&segments[1], 1.0));
        let (pa, pb, pc) = (257.5, 1270.0 / 3.0, 95.0);
        let kappa = (a / pa + b / pb) / 4.0;
        assert!((p.kappa - kappa).abs() < 1e-12 * kappa);
        for (program, mean) in p.model.programs.iter().zip([pa, pb, pc]) {
            assert!((program.phi - kappa * mean).abs() < 1e-10 * program.phi);
        }
    }

    #[test]
    fn one_program_has_its_own_chain_ladder_scale() {
        let tri = genins();
        let p = prepared(&tri, &quick(1));
        let segment = tri.segments("values").unwrap().pop().unwrap();
        let (fit, _) = prepare(&segment, &segment.ages, Tail::default()).unwrap();
        assert!((p.model.programs[0].phi - fit.scale).abs() < 1e-9 * fit.scale);
    }

    #[test]
    fn a_panel_with_no_degree_of_freedom_names_its_programs() {
        let tri = panel(&[
            ("thin", 2020, &[&[10.0, 16.0], &[12.0]], &[30.0, 30.0]),
            ("slim", 2020, &[&[20.0, 33.0], &[21.0]], &[60.0, 60.0]),
        ]);
        let error = quick(1).fit(&tri, "values", "premium").unwrap_err();
        let message = error.to_string();
        assert!(message.contains("program slim, program thin"), "{message}");
        assert!(message.contains("no degree of freedom"), "{message}");
    }

    #[test]
    fn a_falling_column_fails_pooled_and_not_alone() {
        // Program c's second column falls and the pooled one does not.
        let rows: &[&[f64]] = &[&[42.0, 30.0], &[50.0]];
        let tri = panel(&[
            (
                "a",
                2020,
                &[&[100.0, 160.0, 185.0], &[110.0, 170.0], &[95.0]],
                &[250.0, 260.0, 240.0],
            ),
            ("c", 2021, rows, &[90.0, 100.0]),
        ]);
        // These chain ladders fit almost exactly, so the fall is many scales.
        let held = HierarchicalBf {
            omega_scale: 1e-3,
            ..quick(1)
        };
        prepared(&tri, &held);
        // The last column falls in the only program that reaches it.
        let tri = panel(&[
            (
                "a",
                2020,
                &[&[100.0, 160.0, 150.0], &[110.0, 170.0], &[95.0]],
                &[250.0, 260.0, 240.0],
            ),
            ("c", 2021, rows, &[90.0, 100.0]),
        ]);
        let error = quick(1).fit(&tri, "values", "premium").unwrap_err();
        let message = error.to_string();
        assert!(message.contains("summed over every program"), "{message}");
        assert!(message.contains("36 months"), "{message}");
    }

    #[test]
    fn falling_columns_past_the_bound_have_no_posterior() {
        // Two programs of two origins add nothing to the scale, so `a` sets
        // kappa and each fall can be placed in units of its program's scale.
        let healthy: Book = (
            "a",
            2020,
            &[
                &[100.0, 160.0, 185.0, 196.0],
                &[110.0, 170.0, 201.0],
                &[95.0, 158.0],
                &[120.0],
            ],
            &[250.0, 260.0, 240.0, 280.0],
        );
        let model = quick(1);
        let kappa = prepared(&panel(&[healthy]), &model).kappa;
        let premium = [100.0, 100.0];
        let fall = |units: f64| [100.0, 100.0 - units * kappa * 100.0];
        let at = |units: f64| fall(units);

        // 1.5 scales alone is inside 1 / omega_scale = 2.
        let one = at(1.5);
        let rows: &[&[f64]] = &[&one, &[100.0]];
        let inside = panel(&[healthy, ("b", 2021, rows, &premium)]);
        prepared(&inside, &model);

        // 2.5 scales alone is past it, and names the program and the age.
        let one = at(2.5);
        let rows: &[&[f64]] = &[&one, &[100.0]];
        let past = panel(&[healthy, ("b", 2021, rows, &premium)]);
        let message = model
            .fit(&past, "values", "premium")
            .unwrap_err()
            .to_string();
        assert!(message.contains("program b at 24 months"), "{message}");
        assert!(message.contains("omega_scale"), "{message}");

        // Falls inside the bound one at a time are past it together: one
        // omega scales every program's deviations.
        let one = at(1.5);
        let rows: &[&[f64]] = &[&one, &[100.0]];
        let together = panel(&[
            healthy,
            ("b", 2021, rows, &premium),
            ("c", 2021, rows, &premium),
        ]);
        let message = model
            .fit(&together, "values", "premium")
            .unwrap_err()
            .to_string();
        assert!(message.contains("program c at 24 months"), "{message}");

        // A tighter prior on the deviations holds them.
        let tight = HierarchicalBf {
            omega_scale: 0.1,
            ..quick(1)
        };
        prepared(&together, &tight);
    }

    #[test]
    fn falls_larger_than_the_first_age_are_an_error() {
        let healthy: Book = (
            "a",
            2020,
            &[&[1000.0, 1600.0, 1850.0], &[1100.0, 1700.0], &[950.0]],
            &[2500.0, 2600.0, 2400.0],
        );
        let rows: &[&[f64]] = &[&[10.0, -10.0, -30.0], &[5.0]];
        let tri = panel(&[healthy, ("b", 2021, rows, &[100.0, 100.0])]);
        let wide = HierarchicalBf {
            cv: 1e-3,
            omega_scale: 1e-6,
            ..quick(1)
        };
        let message = wide.fit(&tri, "values", "premium").unwrap_err().to_string();
        assert!(message.contains("program b"), "{message}");
        assert!(message.contains("first age"), "{message}");
    }

    #[test]
    fn a_pattern_prior_of_the_wrong_length_is_an_error() {
        let tri = three_programs();
        let prior = PatternPrior {
            ages: vec![12, 24, 36, 48],
            mean: vec![0.0],
            sd: vec![1.0, 1.0, 1.0],
        };
        let error = quick(1).fit_with_pattern_prior(&tri, "values", "premium", &prior);
        let message = error.unwrap_err().to_string();
        assert!(message.contains("1 means and 3 sds"), "{message}");
    }

    /// The log posterior from the note, on the natural scale, with the
    /// gamma's density in full.
    fn by_hand(m: &Model, theta: &[f64]) -> f64 {
        let natural = m.natural(theta);
        let (log_level, log_tau) = (natural[0], theta[1]);
        let log_omega = theta[m.log_omega()];
        let (tau, omega) = (log_tau.exp(), log_omega.exp());
        let mut f = log_tau - 0.5 * m.tau_precision * tau * tau;
        f += log_omega - 0.5 * m.omega_precision * omega * omega;
        if !m.level_free {
            f -= 0.5 * ((log_level - m.level_mean) / m.level_sd).powi(2);
        }
        for d in 1..m.n_dev {
            let gap = theta[m.beta(d)] - m.development_mean[d - 1];
            f -= 0.5 * m.development_precision[d - 1] * gap * gap;
        }
        let k = m.shape;
        for (c, program) in m.programs.iter().enumerate() {
            let z = theta[m.z(c)];
            f -= 0.5 * z * z;
            let s = (tau * z - 0.5 * tau * tau).exp();
            let mut weight = vec![1.0; m.n_dev];
            for d in 1..m.n_dev {
                let delta = theta[m.delta(c, d)];
                f -= 0.5 * delta * delta;
                weight[d] = (theta[m.beta(d)] + omega * delta).exp();
            }
            let total: f64 = weight.iter().sum();
            for (w, log_prior) in program.log_prior.iter().enumerate() {
                let premium = (log_prior - m.log_apriori).exp();
                let mean = log_level.exp() * s * premium;
                let ultimate = natural[m.u(program, w)].exp();
                // Gamma(k, mean k^-1) on U, with the Jacobian dU / d alpha.
                f += k * (k / mean).ln() + (k - 1.0) * ultimate.ln() - k * ultimate / mean
                    + ultimate.ln();
            }
            for &(w, d, x) in &program.cells {
                let cell = natural[m.u(program, w)].exp() * weight[d] / total;
                f += (x * cell.ln() - cell) / program.phi;
            }
        }
        f
    }

    #[test]
    fn the_density_is_the_notes_on_the_natural_scale() {
        // Every term away from the corners, flat and informative: differences
        // between points, since each form drops its own constants.
        let tri = three_programs();
        for level_cv in [None, Some(0.3)] {
            let model = HierarchicalBf {
                level_cv,
                ..quick(1)
            };
            let panel = prepared(&tri, &model);
            let m = &panel.model;
            let base = near(&panel.start, 0, 0.3);
            let mut gradient = vec![0.0; m.dim()];
            let at_base = m.log_posterior(&base, &mut gradient);
            for k in 1..6 {
                let point = near(&panel.start, k, 0.3);
                let f = m.log_posterior(&point, &mut gradient) - at_base;
                let g = by_hand(m, &point) - by_hand(m, &base);
                assert!((f - g).abs() < 1e-8 * (1.0 + g.abs()), "{f} against {g}");
            }
        }
    }

    #[test]
    fn a_program_starting_late_is_an_error() {
        let origin = [2020, 2020, 2021, 2020, 2020].map(Month::january);
        let tri = Triangle::from_long(&Long {
            keys: &[("program", &["a", "a", "a", "b", "b"])],
            origin: &origin,
            development: DevelopmentColumn::Age(&[12, 24, 12, 24, 36]),
            values: &[
                ("values", &[10.0, 16.0, 12.0, 30.0, 40.0]),
                ("premium", &[30.0, 30.0, 30.0, 60.0, 60.0]),
            ],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        let error = quick(1).fit(&tri, "values", "premium").unwrap_err();
        let message = error.to_string();
        assert!(message.contains("program b"), "{message}");
        assert!(message.contains("first age"), "{message}");
    }

    #[test]
    fn invalid_settings() {
        let tri = three_programs();
        let bad = [
            HierarchicalBf {
                apriori: 0.0,
                ..quick(1)
            },
            HierarchicalBf {
                level_cv: Some(0.0),
                ..quick(1)
            },
            HierarchicalBf {
                cv: f64::INFINITY,
                ..quick(1)
            },
            HierarchicalBf {
                development_sd: -1.0,
                ..quick(1)
            },
            HierarchicalBf {
                tau_scale: 0.0,
                ..quick(1)
            },
            HierarchicalBf {
                omega_scale: f64::NAN,
                ..quick(1)
            },
            HierarchicalBf {
                n_sims: 0,
                ..quick(1)
            },
        ];
        for model in bad {
            let error = model.fit(&tri, "values", "premium").unwrap_err();
            assert!(matches!(error, Error::InvalidSetting { .. }), "{error}");
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_seed_replays_whatever_the_threads() {
        let tri = three_programs();
        let a = quick(3).fit(&tri, "values", "premium").unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let b = pool.install(|| quick(3).fit(&tri, "values", "premium").unwrap());
        assert_eq!(a.posterior, b.posterior);
        assert_eq!(a.reserves.draw_matrix(), b.reserves.draw_matrix());
        let c = quick(4).fit(&tri, "values", "premium").unwrap();
        assert_ne!(a.posterior.values(), c.posterior.values());
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_fit_and_its_programs() {
        let tri = three_programs();
        let fit = quick(2).fit(&tri, "values", "premium").unwrap();
        assert_eq!(fit.ages, [12, 24, 36, 48]);
        assert_eq!(fit.reserves.dims(), ["program", "origin"]);
        assert_eq!(fit.reserves.n_components(), 9);
        let hyper = fit.hyperparameters().unwrap();
        assert_eq!(hyper[0].name, "level");
        assert!(hyper.iter().all(|h| h.mean > 0.0));
        let summed: f64 = fit.programs.iter().map(|p| p.reserves.mean()).sum();
        assert!((summed - fit.reserves.mean()).abs() < 1e-6 * summed);
        // Each program's draws are the joint draws' own columns, simulation
        // by simulation, which sums of means alone would not show.
        let width = fit.reserves.n_components();
        let joint = fit.reserves.draw_matrix();
        let mut first = 0;
        for program in &fit.programs {
            let own = program.reserves.n_components();
            for (i, row) in program.reserves.draw_matrix().chunks_exact(own).enumerate() {
                assert_eq!(row, &joint[i * width + first..i * width + first + own]);
            }
            first += own;
        }
        assert_eq!(first, width);
        for program in &fit.programs {
            let total: f64 = program.pattern.iter().sum();
            assert!((total - 1.0).abs() < 1e-9);
            assert!(program.loss_ratio > 0.0);
            assert_eq!(program.expected_ultimate.len(), program.origins.len());
            assert!(program.reserves.mean() > 0.0);
        }
        // The thin program's deviations past its ages are carried, and
        // held by their prior.
        let summary = fit.summary().unwrap();
        let late = summary.iter().find(|s| s.name == "delta[3,4]").unwrap();
        assert!((late.sd - 1.0).abs() < 0.15, "{}", late.sd);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_very_tight_hyperprior_keeps_its_variance() {
        // cv^2 is below the rounding of 1 + cv^2, so the variance needs ln_1p.
        let model = HierarchicalBf {
            level_cv: Some(1e-9),
            ..quick(3)
        };
        let panel = prepared(&three_programs(), &model);
        assert!((panel.model.level_sd - 1e-9).abs() < 1e-15);
        let fit = model.fit(&three_programs(), "values", "premium").unwrap();
        let level = &fit.hyperparameters().unwrap()[0];
        assert!((level.mean - 0.7).abs() < 1e-6, "{}", level.mean);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_tight_hyperprior_with_no_spread_reproduces_the_normalised_fit() {
        // Validation item 1's fits, on GenIns with premium (RAA is in the
        // validation file).
        let tri = genins();
        let sampler = Sampler {
            tune: 500,
            draws: 1000,
            target_accept: 0.95,
            ..Sampler::default()
        };
        let hierarchical = HierarchicalBf {
            apriori: 0.6,
            level_cv: Some(1e-6),
            cv: 0.2,
            tau_scale: 1e-6,
            omega_scale: 1e-6,
            sampler,
            n_sims: 8_000,
            seed: 11,
            ..HierarchicalBf::default()
        }
        .fit(&tri, "values", "premium")
        .unwrap();
        let normalised = BayesBfNormalised {
            apriori: 0.6,
            cv: 0.2,
            sampler,
            n_sims: 8_000,
            seed: 12,
            ..BayesBfNormalised::default()
        }
        .fit(&tri, "values", "premium")
        .unwrap();
        let (a, b) = (&hierarchical.reserves, &normalised.reserves);
        let sd = b.std_dev();
        assert!(
            (a.mean() - b.mean()).abs() < 0.1 * sd,
            "{} {}",
            a.mean(),
            b.mean()
        );
        assert!((a.std_dev() / sd - 1.0).abs() < 0.1, "{} {sd}", a.std_dev());
        let (per_a, per_b) = (component_sums(a), component_sums(b));
        for w in 1..10 {
            let (x, y) = (&per_a[w], &per_b[w]);
            assert!(
                (x.mean() - y.mean()).abs() < 0.1 * y.std_dev(),
                "origin {w}: {} {}",
                x.mean(),
                y.mean()
            );
            assert!((x.std_dev() / y.std_dev() - 1.0).abs() < 0.1, "origin {w}");
        }
    }

    /// The year of a prepared panel and its fit.
    fn year_of<'a>(
        model: &'a HierarchicalBf,
        tri: &Triangle,
        panel: &'a Panel,
        fit: &HierarchicalBfFit,
    ) -> HierarchicalYear<'a> {
        let segments = tri.segments("values").unwrap();
        HierarchicalYear::new(
            model,
            &panel.model,
            tri,
            &segments,
            panel.model.sampling(&fit.posterior.mean()),
        )
    }

    /// Three programs of three origins each, every origin at the last age:
    /// a year without cells.
    fn closed_programs() -> Triangle {
        panel(&[
            (
                "a",
                2020,
                &[
                    &[100.0, 160.0, 185.0],
                    &[110.0, 170.0, 201.0],
                    &[95.0, 158.0, 186.0],
                ],
                &[250.0, 260.0, 240.0],
            ),
            (
                "b",
                2020,
                &[
                    &[210.0, 330.0, 395.0],
                    &[190.0, 320.0, 380.0],
                    &[240.0, 372.0, 430.0],
                ],
                &[400.0, 420.0, 450.0],
            ),
            (
                "c",
                2020,
                &[
                    &[42.0, 70.0, 83.0],
                    &[50.0, 77.0, 91.0],
                    &[39.0, 66.0, 80.0],
                ],
                &[90.0, 100.0, 95.0],
            ),
        ])
    }

    #[test]
    fn the_stored_scale_is_the_sampling_scale_moved() {
        for level_cv in [Some(0.3), None] {
            let p = prepared(
                &three_programs(),
                &HierarchicalBf {
                    level_cv,
                    ..quick(1)
                },
            );
            let theta = near(&p.start, 3, 0.4);
            let stored = p.model.natural(&theta);
            let back = p.model.sampling(&stored);
            for (a, b) in theta.iter().zip(&back) {
                assert!((a - b).abs() < 1e-12, "{a} {b}");
            }
            // The density of the stored draw is the sampling scale's.
            let mut grad = vec![0.0; theta.len()];
            let direct = p.model.log_posterior(&theta, &mut grad);
            assert!((p.model.log_density(&back) - direct).abs() < 1e-9 * direct.abs().max(1.0));
        }
    }

    #[test]
    fn the_year_is_each_programs_next_diagonal() {
        let tri = three_programs();
        let model = quick(1);
        let panel = prepared(&tri, &model);
        let segments = tri.segments("values").unwrap();
        let year =
            HierarchicalYear::new(&model, &panel.model, &tri, &segments, panel.start.clone());
        // Program a (four origins, ages to 48), b (to 36), c (to 24), on a
        // grid of four ages: the next age of each origin that has one.
        let years: Vec<_> = year.programs.iter().map(|p| p.year.clone()).collect();
        assert_eq!(years[0], [(1, 3), (2, 2), (3, 1)]);
        assert_eq!(years[1], [(0, 3), (1, 2), (2, 1)]);
        assert_eq!(years[2], [(0, 2), (1, 1)]);
        assert_eq!(year.n_year(), 8);
        // The cells to come: a thin program carries the grid's late ages.
        assert_eq!(year.programs[2].future.len(), 2 + 3);
        assert_eq!(year.programs[2].after.len(), 3);
        assert_eq!(year.programs[1].first_cell, 3);
        assert_eq!(year.programs[2].first_cell, 6);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_one_year_view_conforms() {
        let model = HierarchicalBf {
            n_sims: 400,
            ..quick(6)
        };
        let open = three_programs();
        let closed = closed_programs();
        let open_panel = prepared(&open, &model);
        let closed_panel = prepared(&closed, &model);
        let open_fit = model.fit(&open, "values", "premium").unwrap();
        let closed_fit = model.fit(&closed, "values", "premium").unwrap();
        crate::bayes::one_year::conformance::check(
            "HierarchicalBf",
            &year_of(&model, &open, &open_panel, &open_fit),
            &open_fit.posterior,
            &year_of(&model, &closed, &closed_panel, &closed_fit),
            &closed_fit.posterior,
        );
    }

    /// The year's likelihood is the model's stated one for the year's cells:
    /// each program's own `phi_c`, the mean `exp(alpha) p_d` with `p` the
    /// program's softmax of `beta + omega delta`, and no other term that
    /// depends on the draw (computed here from the formulas alone, and as
    /// the difference of the posterior with and without the year's cells).
    #[test]
    fn the_year_likelihood_is_the_stated_likelihood_of_the_year() {
        let tri = three_programs();
        let model = quick(1);
        let panel = prepared(&tri, &model);
        let year = year_of_start(&model, &tri, &panel);
        let m = &panel.model;
        let y: Vec<f64> = (0..year.n_year())
            .map(|k| 20.0 + 7.0 * (k % 4) as f64 + k as f64)
            .collect();
        for k in 0..6 {
            let sampling = near(&panel.start, k, 0.4);
            let stored = m.natural(&sampling);
            let omega = stored[m.log_omega()].exp();
            let mut direct = 0.0;
            for (c, (program, py)) in m.programs.iter().zip(&year.programs).enumerate() {
                let mut b = vec![0.0; m.n_dev];
                for d in 1..m.n_dev {
                    b[d] = stored[m.beta(d)] + omega * stored[m.delta(c, d)];
                }
                let sum: f64 = b.iter().map(|x| x.exp()).sum();
                for (j, &(w, d)) in py.year.iter().enumerate() {
                    let mu = stored[m.u(program, w)].exp() * b[d].exp() / sum;
                    direct += (y[py.first_cell + j] * mu.ln() - mu) / program.phi;
                }
            }
            let got = year.year_log_likelihood(&stored, &y);
            assert!(
                (got - direct).abs() < 1e-10 * direct.abs().max(1.0),
                "{got} {direct}"
            );
            // The posterior with the year's cells less the posterior without.
            let extended = year.extended(&y).log_density(&sampling);
            // The density with the year added in place is the density of
            // the extended model, bit for bit.
            assert_eq!(
                m.log_density_with(&sampling, Some((&year.programs, &y))),
                extended
            );
            assert_eq!(
                year.log_posterior(&stored, &y),
                year.extended(&y).log_density(&m.sampling(&stored))
            );
            let diff = extended - m.log_density(&sampling);
            assert!(
                (diff - direct).abs() < 1e-9 * direct.abs().max(1.0),
                "{diff} {direct}"
            );
        }
    }

    /// A refit's data are the opening data plus exactly the simulated year,
    /// and the year is drawn from the draw's own mean and its program's
    /// `phi_c` (mean `mu`, variance `phi mu`).
    #[test]
    fn the_extended_data_are_the_opening_cells_and_the_year_and_the_year_is_drawn_from_the_draw() {
        let tri = three_programs();
        let model = quick(1);
        let panel = prepared(&tri, &model);
        let year = year_of_start(&model, &tri, &panel);
        let m = &panel.model;
        let y: Vec<f64> = (0..year.n_year()).map(|k| 10.0 + k as f64).collect();
        let extended = year.extended(&y);
        for ((base, ext), py) in m
            .programs
            .iter()
            .zip(&extended.programs)
            .zip(&year.programs)
        {
            assert_eq!(ext.cells.len(), base.cells.len() + py.year.len());
            assert_eq!(ext.cells[..base.cells.len()], base.cells[..]);
            for (j, &(w, d)) in py.year.iter().enumerate() {
                assert_eq!(
                    ext.cells[base.cells.len() + j],
                    (w, d, y[py.first_cell + j])
                );
                assert!(!base.cells.iter().any(|&(bw, bd, _)| (bw, bd) == (w, d)));
            }
            assert_eq!(ext.phi, base.phi);
        }
        let stored = m.natural(&near(&panel.start, 2, 0.3));
        let omega = stored[m.log_omega()].exp();
        let n = 40_000;
        let mut rng = StreamRng::new(3, 0);
        let mut sum = vec![0.0; year.n_year()];
        let mut square = vec![0.0; year.n_year()];
        let mut row = vec![0.0; year.n_year()];
        for _ in 0..n {
            year.draw_year(&stored, &mut rng, &mut row);
            for (j, x) in row.iter().enumerate() {
                sum[j] += x;
                square[j] += x * x;
            }
        }
        for (c, (program, py)) in m.programs.iter().zip(&year.programs).enumerate() {
            let mut b = vec![0.0; m.n_dev];
            for d in 1..m.n_dev {
                b[d] = stored[m.beta(d)] + omega * stored[m.delta(c, d)];
            }
            let total: f64 = b.iter().map(|x| x.exp()).sum();
            for (j, &(w, d)) in py.year.iter().enumerate() {
                let mu = stored[m.u(program, w)].exp() * b[d].exp() / total;
                let k = py.first_cell + j;
                let mean = sum[k] / n as f64;
                let var = square[k] / n as f64 - mean * mean;
                assert!(
                    (mean / mu - 1.0).abs() < 0.03,
                    "program {c} cell {j}: {mean} {mu}"
                );
                assert!(
                    (var / (program.phi * mu) - 1.0).abs() < 0.08,
                    "program {c} cell {j}: {var} {}",
                    program.phi * mu
                );
            }
        }
    }

    /// With no `n_sims` in the settings the view runs the model's `n_sims`
    /// at most `ONE_YEAR_SIMS` times, 2,000, whatever the lifetime count.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_default_number_of_simulations_is_capped_at_two_thousand() {
        use crate::bayes::one_year::{BayesOneYear, OneYearRefit};
        let tri = three_programs();
        let model = HierarchicalBf {
            sampler: Sampler {
                tune: 100,
                draws: 100,
                chains: 2,
                ..quick(2).sampler
            },
            n_sims: 2_500,
            ..quick(2)
        };
        let settings = BayesOneYear {
            refit: Some(OneYearRefit::ReweightOnly),
            ..BayesOneYear::default()
        };
        let view = model
            .one_year(&tri, "values", "premium", &settings)
            .unwrap();
        assert_eq!(HierarchicalBf::ONE_YEAR_SIMS, 2_000);
        assert_eq!(view.cdr.n_sims(), 2_000);
        assert_eq!(view.pareto_k.len(), 2_000);
        // The lifetime reserves keep the model's count.
        assert_eq!(view.fit.reserves.n_sims(), 2_500);
        // A count in the settings is used as it is, above the cap too.
        let more = BayesOneYear {
            n_sims: Some(2_100),
            ..settings
        };
        let view = model.one_year(&tri, "values", "premium", &more).unwrap();
        assert_eq!(view.cdr.n_sims(), 2_100);
    }

    fn year_of_start<'a>(
        model: &'a HierarchicalBf,
        tri: &Triangle,
        panel: &'a Panel,
    ) -> HierarchicalYear<'a> {
        let segments = tri.segments("values").unwrap();
        HierarchicalYear::new(model, &panel.model, tri, &segments, panel.start.clone())
    }

    /// A program's next diagonal moves the other programs' closing
    /// reserves: the pooling reaches the view, which a view per program
    /// would miss (decision 30).
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn one_programs_diagonal_moves_anothers_closing_reserve() {
        let tri = three_programs();
        let model = quick(5);
        let panel = prepared(&tri, &model);
        let fit = model.fit(&tri, "values", "premium").unwrap();
        let year = year_of(&model, &tri, &panel, &fit);
        let posterior = &fit.posterior;
        let s = posterior.n_draws();
        // The year at the posterior mean cells, then program a's cells
        // scaled by `factor`.
        let mean = posterior.mean();
        let expected = |theta: &[f64], scale_a: f64| {
            let mut y = vec![0.0; year.n_year()];
            for (c, (program, py)) in panel.model.programs.iter().zip(&year.programs).enumerate() {
                let log_share = panel.model.log_shares(theta, c);
                for (k, &(w, d)) in py.year.iter().enumerate() {
                    let mu = (theta[panel.model.u(program, w)] + log_share[d]).exp();
                    y[py.first_cell + k] = mu * if c == 0 { scale_a } else { 1.0 };
                }
            }
            y
        };
        let closing_bc = |scale_a: f64| {
            let y = expected(&mean, scale_a);
            let log_w: Vec<f64> = (0..s)
                .map(|d| year.year_log_likelihood(posterior.draw(d), &y))
                .collect();
            let top = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = log_w.iter().map(|l| (l - top).exp()).collect();
            let total: f64 = w.iter().sum();
            let ess = total * total / w.iter().map(|x| x * x).sum::<f64>();
            println!("ess {ess}");
            assert!(ess > 20.0, "ess {ess}");
            let mut row = vec![0.0; year.n_origins()];
            let mut closing = vec![0.0; 2];
            for (d, &wd) in w.iter().enumerate() {
                year.closing(posterior.draw(d), &y, &mut row);
                // Program b's origins are positions 4 to 6.
                closing[0] += wd * row[4..7].iter().sum::<f64>() / total;
                closing[1] += wd * row[7..9].iter().sum::<f64>() / total;
            }
            closing
        };
        let (low, high) = (closing_bc(0.8), closing_bc(1.25));
        println!("closing reserve of b and c, a's year at 0.8 and 1.25 times: {low:?} {high:?}");
        // Higher losses in a raise the industry level and so b's and c's
        // reserves; independent programs would leave them unchanged. Measured
        // (seed 5, release): the ratio high / low is 1.083 for b and 1.131
        // for c, so the bound of 1.04 is under half of the smaller excess.
        for k in 0..2 {
            assert!(
                high[k] > 1.04 * low[k],
                "program {}: {low:?} {high:?}",
                k + 1
            );
        }
    }

    /// A panel whose programs stop at different ages warns (the user's
    /// decision of 2026-11-09, issue #762): on the three-program panel,
    /// programs b and c stop short of the grid's 48 months, so their next
    /// diagonals fall at ages they never observed, and a does not; a panel
    /// whose programs share their ages, and one with no year, do not warn.
    #[test]
    fn programs_stopping_short_of_the_panels_ages_warn() {
        let model = quick(1);
        let warning_in = |tri: &Triangle, mode: OneYearRefit| {
            let panel = prepared(tri, &model);
            let segments = tri.segments("values").unwrap();
            HierarchicalYear::new(&model, &panel.model, tri, &segments, panel.start.clone())
                .unobserved_ages(&panel.labels, &panel.ages, mode)
        };
        let warning = |tri: &Triangle| warning_in(tri, OneYearRefit::MomentMatchThenRefit);
        let reweighted =
            warning_in(&three_programs(), OneYearRefit::ReweightOnly).expect("b and c stop short");
        assert!(reweighted.contains("b (observed to age 36"), "{reweighted}");
        assert!(reweighted.contains("unreliable"), "{reweighted}");
        assert!(!reweighted.contains("expensive"), "{reweighted}");
        let ragged = warning(&three_programs()).expect("b and c stop short");
        assert!(
            ragged.contains("b (observed to age 36, its next diagonal to 48)"),
            "{ragged}"
        );
        assert!(
            ragged.contains("c (observed to age 24, its next diagonal to 36)"),
            "{ragged}"
        );
        assert!(!ragged.contains("a ("), "{ragged}");
        assert!(ragged.contains("expensive"), "{ragged}");
        let shared = panel(&[
            (
                "a",
                2020,
                &[&[100.0, 160.0, 185.0], &[110.0, 170.0], &[95.0]],
                &[250.0, 260.0, 240.0],
            ),
            (
                "b",
                2020,
                &[&[210.0, 330.0, 395.0], &[190.0, 320.0], &[240.0]],
                &[400.0, 420.0, 450.0],
            ),
        ]);
        assert_eq!(warning(&shared), None);
        assert_eq!(warning(&closed_programs()), None);
    }

    /// The refits' sampler: a light refit given is used whatever the refit
    /// mode; the default light refit only with no refit mode chosen and
    /// when the model's default holds (both values of the default checked);
    /// none when `no_light_refit` turns the default off, and none when the
    /// view reweights only; the number of simulations is capped as before.
    #[test]
    fn the_light_refit_setting_resolves() {
        let model = quick(1);
        let light = LightRefit {
            chains: 2,
            tune: 60,
            draws: 60,
        };
        let resolve = |settings: BayesOneYear| model.one_year_defaults(&settings);
        let default = HierarchicalBf::LIGHT_REFIT_BY_DEFAULT.then(LightRefit::default);
        assert_eq!(resolve(BayesOneYear::default()).0.light_refit, default);
        assert_eq!(resolve(BayesOneYear::default()).1, 1_000);
        for refit in [None, Some(OneYearRefit::RefitAll)] {
            let given = BayesOneYear {
                refit,
                light_refit: Some(light),
                ..BayesOneYear::default()
            };
            assert_eq!(resolve(given).0.light_refit, Some(light));
        }
        let off = BayesOneYear {
            no_light_refit: true,
            ..BayesOneYear::default()
        };
        assert_eq!(resolve(off).0.light_refit, None);
        let reweight = BayesOneYear {
            refit: Some(OneYearRefit::ReweightOnly),
            light_refit: Some(light),
            ..BayesOneYear::default()
        };
        assert_eq!(resolve(reweight).0.light_refit, None);
        // The default light refit, whatever the constant says.
        let of = HierarchicalBf::light_refit_of;
        assert_eq!(
            of(&BayesOneYear::default(), true),
            Some(LightRefit::default())
        );
        assert_eq!(of(&BayesOneYear::default(), false), None);
        for refit in [
            OneYearRefit::RefitAll,
            OneYearRefit::RefitFlagged,
            OneYearRefit::MomentMatchThenRefit,
            OneYearRefit::ReweightOnly,
        ] {
            let chosen = BayesOneYear {
                refit: Some(refit),
                ..BayesOneYear::default()
            };
            assert_eq!(of(&chosen, true), None, "{refit:?}");
        }
        let off = BayesOneYear {
            no_light_refit: true,
            ..BayesOneYear::default()
        };
        assert_eq!(of(&off, true), None);
    }

    /// A bad light refit is an error before any sampling.
    #[test]
    fn a_bad_light_refit_is_an_error() {
        for bad in [(0, 60, 60), (2, 0, 60), (2, 60, 1)] {
            let settings = BayesOneYear {
                light_refit: Some(LightRefit {
                    chains: bad.0,
                    tune: bad.1,
                    draws: bad.2,
                }),
                ..BayesOneYear::default()
            };
            assert!(matches!(
                quick(1).one_year(&three_programs(), "values", "premium", &settings),
                Err(Error::InvalidSetting {
                    name: "light_refit",
                    ..
                })
            ));
        }
    }

    /// A light refit samples the posterior a full refit samples, for one
    /// year drawn from the fit: the closing reserve's mean within 0.3 sd of
    /// a long full refit's and its sd within 25%; its draws as stored, with
    /// a score of mean zero under the year's `log_posterior`; and a light
    /// refit of 2 draws warns.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_light_refit_samples_the_full_refits_posterior() {
        use crate::bayes::one_year::conformance::{SCORE_Z, score_z_at};
        let model = quick(5);
        let tri = three_programs();
        let panel = prepared(&tri, &model);
        let fit = model.fit(&tri, "values", "premium").unwrap();
        let mut year = year_of(&model, &tri, &panel, &fit);
        let mut cells = vec![0.0; year.n_year()];
        year.draw_year(
            fit.posterior.draw(0),
            &mut StreamRng::new(13, 0),
            &mut cells,
        );
        let reference = HierarchicalBf {
            sampler: Sampler {
                chains: 4,
                tune: 500,
                draws: 1_000,
                ..model.sampler
            },
            ..model.clone()
        };
        let (full, _) = year_of(&reference, &tri, &panel, &fit)
            .refit(&cells, 21)
            .unwrap();
        year.warm_start(Some(LightRefit::default()), &fit.posterior)
            .unwrap();
        let (light, _) = year.refit(&cells, 21).unwrap();
        assert_eq!(light.names(), fit.posterior.names());
        let reserves = |draws: &PosteriorDraws| -> (f64, f64) {
            let mut row = vec![0.0; year.n_origins()];
            let t: Vec<f64> = (0..draws.n_draws())
                .map(|d| {
                    year.closing(draws.draw(d), &cells, &mut row);
                    row.iter().sum()
                })
                .collect();
            let n = t.len() as f64;
            let mean = t.iter().sum::<f64>() / n;
            let sd = (t.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
            (mean, sd)
        };
        let ((mf, sf), (ml, sl)) = (reserves(&full), reserves(&light));
        assert!(
            (ml - mf).abs() < 0.3 * sf && (sl / sf - 1.0).abs() < 0.25,
            "closing reserve {ml} (sd {sl}) light, {mf} (sd {sf}) full"
        );
        let z = score_z_at(&year, &light, &cells);
        let worst = z.iter().copied().fold(0.0, |w: f64, v| w.max(v.abs()));
        assert!(worst < SCORE_Z, "the light refit's score has mean z {z:?}");
        let tiny = LightRefit {
            chains: 1,
            tune: 1,
            draws: 2,
        };
        year.warm_start(Some(tiny), &fit.posterior).unwrap();
        let (_, warned) = year.refit(&cells, 21).unwrap();
        assert!(warned, "a light refit of 2 draws did not warn");
    }

    /// A light refit in the view refits every simulation `RefitAll` sends
    /// it, on the same years and `k̂` as the fit's own sampler,
    /// reproducibly, and is recorded in the provenance; the view's warning
    /// names the programs that stop short.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_light_refit_in_the_view_refits_every_simulation() {
        let model = HierarchicalBf {
            n_sims: 4,
            ..quick(4)
        };
        let light = LightRefit {
            chains: 2,
            tune: 60,
            draws: 60,
        };
        let with = |light_refit: Option<LightRefit>, no_light_refit: bool| {
            model
                .one_year(
                    &three_programs(),
                    "values",
                    "premium",
                    &BayesOneYear {
                        refit: Some(OneYearRefit::RefitAll),
                        light_refit,
                        no_light_refit,
                        ..BayesOneYear::default()
                    },
                )
                .unwrap()
        };
        let (full, a, b) = (
            with(None, true),
            with(Some(light), false),
            with(Some(light), false),
        );
        assert_eq!(a.refitted, [0, 1, 2, 3]);
        assert_eq!(a.refitted, full.refitted);
        assert_eq!(a.pareto_k, full.pareto_k);
        assert_eq!(a.cdr.draw_matrix(), b.cdr.draw_matrix());
        assert_ne!(a.cdr.draw_matrix(), full.cdr.draw_matrix());
        let sd = full.cdr.std_dev();
        let n = full.cdr.n_components();
        for (x, y) in a
            .cdr
            .draw_matrix()
            .chunks(n)
            .zip(full.cdr.draw_matrix().chunks(n))
        {
            let (x, y): (f64, f64) = (x.iter().sum(), y.iter().sum());
            assert!(x.is_finite() && (x - y).abs() < 3.0 * sd, "{x} against {y}");
        }
        let recorded = |v: &BayesOneYearFit<HierarchicalBfFit>| {
            v.cdr
                .provenance()
                .parameters
                .iter()
                .find(|(k, _)| k == "light_refit")
                .map(|(_, v)| v.clone())
        };
        assert_eq!(recorded(&a), Some("2x60+60".into()));
        assert_eq!(recorded(&full), None);
        assert_eq!(a.warnings.len(), 1);
        assert!(
            a.warnings[0].contains("b (observed to age 36"),
            "{:?}",
            a.warnings
        );
    }

    /// Bit-exact on CI's Linux runner (#841; the doc of `bayes::pins` says what
    /// to do when the hash moves): the three small programs of
    /// `three_programs`, two chains of 100 warm-up and 50 kept draws, 100
    /// simulations.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_fit_is_pinned_on_linux() {
        let model = HierarchicalBf {
            sampler: Sampler {
                chains: 2,
                tune: 100,
                draws: 50,
                ..quick(841).sampler
            },
            n_sims: 100,
            ..quick(841)
        };
        crate::bayes::pins::check("HierarchicalBf", 0, || {
            let fit = model.fit(&three_programs(), "values", "premium").unwrap();
            crate::bayes::pins::fit_hash(&fit.posterior, &fit.reserves)
        });
    }
}
