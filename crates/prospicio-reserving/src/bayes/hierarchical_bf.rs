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
use super::scalar::Scalar;
use super::{Diagnostics, Stream, derive_seed, pilot_seed};
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

/// The fewest draws of a pilot run, as [`Meyers`](super::Meyers).
const MIN_PILOT_DRAWS: usize = 100;

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
    /// of 1,000 warm-up and 1,000 kept draws with target acceptance 0.95,
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
                target_accept: 0.95,
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
    /// loss ratio is not positive.
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
    /// the programs' ages from the first to the last, and it may hold an
    /// age that no program observes. `development_sd` is then unused.
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
        let model = &panel.model;
        let (posterior, diagnostics) = self.sample(
            model,
            panel.start.clone(),
            derive_seed(self.seed, Stream::Chains),
        )?;

        let mut dims = triangle.key_names().to_vec();
        dims.push("origin".to_string());
        let components: Vec<ComponentKey> = panel
            .labels
            .iter()
            .zip(&panel.origins)
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
            .collect();
        let provenance = self.provenance(column, exposure, &panel);
        let process = self.process;
        let reserves = posterior.predictive(
            dims,
            components,
            self.n_sims,
            derive_seed(self.seed, Stream::Predictive),
            provenance.clone(),
            |theta, rng: &mut StreamRng, row| model.simulate(theta, process, rng, row),
        )?;

        let programs = self.programs(&panel, &posterior, &reserves, &provenance)?;
        Ok(HierarchicalBfFit {
            programs,
            ages: panel.ages,
            scale_per_premium: panel.kappa,
            posterior,
            diagnostics,
            reserves,
        })
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
                draws: self.sampler.tune.max(MIN_PILOT_DRAWS),
                ..self.sampler
            },
        )?;
        let pilot = pilot.transform(model.names(), |z| rough.theta(z))?;
        let whitened = Whitened::from_draws(model, &pilot)?;
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
        // that sums to zero or less is held by its deviation's prior.
        let mut column_sum = vec![0.0; nd];
        let mut column_n = vec![0; nd];
        for p in &data {
            for &(_, d, x) in &p.observed.cells {
                column_sum[d] += x;
                column_n[d] += 1;
            }
        }
        for d in 0..nd {
            if column_n[d] == 0 {
                if pattern.is_none() {
                    return Err(Error::Bayes(format!(
                        "development age {} months: no program observes it, and the vague prior \
                         on the shared pattern cannot hold it. Give an informative pattern \
                         prior with fit_with_pattern_prior",
                        ages[d]
                    )));
                }
            } else if column_sum[d].is_nan() || column_sum[d] <= 0.0 {
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

        let pooled = pooled_factors(&data, nd);
        let log_apriori = self.apriori.ln();
        let (level_mean, level_sd, level_free) = match self.level_cv {
            Some(c) => {
                let variance = c.mul_add(c, 1.0).ln();
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
        prepared(&tri, &quick(1));
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
    fn a_tight_hyperprior_with_no_spread_reproduces_the_normalised_fit() {
        // Validation item 1's fits, on GenIns with premium.
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
}
