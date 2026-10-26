//! The Bayesian Bornhuetter–Ferguson: the over-dispersed Poisson model with
//! informative gamma priors on the origin parameters, after England,
//! Verrall and Wüthrich (2012) (`docs/design/reserving-bayes.md`,
//! decision 9).

use prospicio_bayes::elpd::Loo;
use prospicio_bayes::nuts::{ParamSummary, PosteriorDraws, Sampler, sample};
use prospicio_core::StreamRng;
use prospicio_models::Family;
use prospicio_prob::{InputHasher, PredictiveDistribution, Provenance};
use rayon::prelude::*;

use super::laplace::{Posterior, Whitened, mode};
use super::one_year::{self, BayesOneYear, BayesOneYearFit, OdpCells, OdpYear};
use super::{Diagnostics, Stream, derive_seed, one_year_seed};
use crate::bootstrap::{
    ProcessDistribution, component_sums, hash_segment, prepare, process_draw, push_moments,
    segment_sums,
};
use crate::chain_ladder::ChainLadderFit;
use crate::error::{Error, Result};
use crate::expected_loss::origin_exposure;
use crate::segments::{FitTable, ReserveFit, SegmentFits};
use crate::tail::Tail;
use crate::triangle::{Label, Segment, Triangle};

/// The Bayesian Bornhuetter–Ferguson of England, Verrall and Wüthrich
/// (2012, Model 3.1): the over-dispersed Poisson model of the incremental
/// values `X_wd` with cross-classified means `mu_w gamma_d`,
///
/// ```text
/// log p(X | theta) = sum over observed cells of (X_wd eta_wd - exp(eta_wd)) / phi,
/// eta_wd = alpha_w + beta_d,   mu_w = exp(alpha_w),   gamma_d = exp(beta_d)
/// mu_w ~ Gamma(shape 1 / cv^2, mean m_w),   m_w = apriori * exposure_w
/// beta_d ~ N(0, development_sd^2)
/// ```
///
/// with `phi` the chain ladder's Pearson scale, plugged in as in
/// [`BayesOdp`](super::BayesOdp). The prior on each origin's row parameter
/// `mu_w` has mean `m_w`, `apriori` times the origin's exposure (the
/// latest value of the exposure column, as for
/// [`BornhuetterFerguson`](crate::BornhuetterFerguson)), and coefficient of
/// variation `cv`; the development pattern `gamma_d` is not normalised and
/// has a vague prior. So `mu_w` is EVW's row parameter, not the origin's
/// expected ultimate: that is `mu_w` times the sum of the `gamma_d`, which
/// under a tight prior is `m_w` times the sum of `S_d / M_d` over all
/// columns (below). On EVW's triangle that sum is 0.885, so the implied
/// ultimate loss ratio is about 11.5% below `apriori`; `apriori` is the
/// classical method's loss ratio only when the sum is 1. For a prior on the
/// ultimate itself, so that `apriori` is the expected ultimate loss ratio,
/// use [`BayesBfNormalised`](super::BayesBfNormalised) (decision 23). The log
/// posterior is strictly concave; NUTS samples it on a scale whitened at the
/// mode, as for `BayesOdp`.
///
/// The prior's strength moves the reserve between two limits:
///
/// * `cv = f64::INFINITY` (a flat prior on `log mu_w`): the posterior is
///   the Bayesian chain ladder's, and the reserves agree with
///   [`BayesOdp`](super::BayesOdp)'s within Monte Carlo error (the two
///   vague priors differ slightly);
/// * `cv -> 0`: `mu_w` is fixed at `m_w`, and each origin's mean reserve
///   tends to `m_w` times the sum of the future `S_d / M_d`, with `S_d`
///   the observed increments of column `d` and `M_d` the prior ultimates
///   of the origins observed there (EVW's `R^MMSE*(infinity)`, Mack's
///   (2008) Bornhuetter–Ferguson with the raw pattern). That is the
///   classical Bornhuetter–Ferguson reserve `(1 - 1 / cdf_w) m_w` only
///   when the prior ultimates are the chain ladder's: the rows' prior
///   reshapes the development pattern, which the classical method takes
///   from the chain ladder.
///
/// Every development column's observed increments must sum to more than
/// zero, and every origin's to more than `-phi / cv^2`, the bound below
/// which its posterior is improper ([`Error::Bayes`]). Negative cells are
/// allowed.
///
/// ```
/// use prospicio_reserving::bayes::BayesBf;
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// use prospicio_bayes::nuts::Sampler;
/// use prospicio_prob::Distribution;
///
/// let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
/// let premium = [200.0, 200.0, 200.0, 200.0, 220.0, 220.0, 220.0, 240.0, 240.0, 260.0];
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
///     values: &[
///         ("paid", &[100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]),
///         ("premium", &premium),
///     ],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let model = BayesBf {
///     apriori: 0.85,
///     cv: 0.1,
///     sampler: Sampler { tune: 300, draws: 300, ..Sampler::default() },
///     n_sims: 2_000,
///     seed: 1,
///     ..BayesBf::default()
/// };
/// let fit = model.fit(&tri, "paid", "premium")?;
/// assert_eq!(fit.reserves.n_components(), 4);
/// assert!(fit.reserves.mean() > 0.0);
/// assert_eq!(fit.prior_ultimate[3], 0.85 * 260.0);
/// assert_eq!(fit.posterior.names()[0], "alpha[1]");
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BayesBf {
    /// Prior mean of each origin's row parameter `mu_w` per unit of
    /// exposure. It is the expected loss ratio only when the development
    /// pattern sums to 1 (type docs): the model's expected ultimate is
    /// `mu_w` times the sum of the `gamma_d`.
    pub apriori: f64,
    /// Coefficient of variation of the gamma prior on each `mu_w`, so its
    /// shape is `1 / cv^2` (EVW's `a_i`). `f64::INFINITY` is a flat prior
    /// on `log mu_w`, the Bayesian chain ladder.
    pub cv: f64,
    /// Standard deviation of the normal prior on each `beta_d`, centred
    /// on 0.
    pub development_sd: f64,
    /// Process error on each simulated future increment, as in the ODP
    /// bootstrap and [`BayesOdp`](super::BayesOdp).
    pub process: ProcessDistribution,
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

impl Default for BayesBf {
    /// `apriori` 1 (as [`BornhuetterFerguson`](crate::BornhuetterFerguson)),
    /// `cv` 0.1 (EVW's example, shape 100), `development_sd` 10, Gamma
    /// process error, four chains of 1,000 warm-up and 1,000 kept draws
    /// with target acceptance 0.95 (as `BayesOdp`), 10,000 simulations,
    /// seed 0.
    fn default() -> Self {
        Self {
            apriori: 1.0,
            cv: 0.1,
            development_sd: 10.0,
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

/// A fitted [`BayesBf`].
#[derive(Debug, Clone)]
pub struct BayesBfFit {
    /// The volume-weighted chain ladder of the triangle: latest values and
    /// the plug-in `phi` come from it, and the long tables' `ultimate` and
    /// `reserve` are its.
    pub chain_ladder: ChainLadderFit,
    /// The plugged-in scale `phi`, the chain ladder's Pearson scale, as
    /// [`BayesOdpFit::scale`](super::BayesOdpFit::scale).
    pub scale: f64,
    /// Exposure per origin: the exposure column's latest observed value.
    pub exposure: Vec<f64>,
    /// The prior mean `m_w` of each origin's row parameter `mu_w`,
    /// `apriori * exposure` (EVW's prior ultimate). The model's expected
    /// ultimate is `mu_w` times the sum of the `gamma_d`, not `mu_w`.
    pub prior_ultimate: Vec<f64>,
    /// Posterior draws of `alpha[1]..alpha[n]` and `beta[1]..beta[m]`, by
    /// chain.
    pub posterior: PosteriorDraws,
    /// R-hat, ESS and divergences, with warnings for failed thresholds.
    pub diagnostics: Diagnostics,
    /// Joint posterior predictive distribution of the reserve (sum of
    /// future incremental values) by origin: dimension `origin`, one
    /// component per origin, as [`BayesOdpFit::reserves`](super::BayesOdpFit::reserves).
    pub reserves: PredictiveDistribution,
    /// The model, for the pointwise log likelihood of [`loo`](Self::loo).
    model: Model,
}

impl BayesBf {
    /// Fits loss `column` of a single-segment cumulative triangle with
    /// exposure from its `exposure` column. Every origin must be observed
    /// from the first age to its latest and there must be more observed
    /// cells than chain-ladder parameters, as for `BayesOdp`. Failed
    /// diagnostics do not fail the fit: they are in
    /// [`BayesBfFit::diagnostics`].
    pub fn fit(&self, triangle: &Triangle, column: &str, exposure: &str) -> Result<BayesBfFit> {
        self.validate()?;
        let segment = triangle.segment(column)?;
        let exposures = origin_exposure(&segment, &triangle.segment(exposure)?, exposure)?;
        self.fit_segment(&segment, column, exposure, exposures)
    }

    /// The one-year view of loss `column` of a single-segment cumulative
    /// triangle with exposure from its `exposure` column (decisions 16 and
    /// 17): the claims development result over the twelve months after the
    /// valuation, by reweighting [`fit`](Self::fit)'s posterior, draw for
    /// draw, with Pareto smoothed importance sampling.
    ///
    /// The method is [`BayesOdp::one_year`](super::BayesOdp::one_year)'s
    /// with this model's means `exp(alpha_w + beta_d)`: each of
    /// [`n_sims`](Self::n_sims) simulations draws the increments of the
    /// year's cells from a posterior draw with [`process`](Self::process)
    /// error, weights every posterior draw by the year's quasi likelihood
    /// (`phi`, the priors and the prior ultimates unchanged), and takes
    /// origin `w`'s claims development result as its posterior mean reserve
    /// at the opening less its payment in the year and its weighted
    /// posterior mean reserve after it. A simulation whose Pareto `k̂` is
    /// above [`BayesOneYear::k_threshold`] is handled as
    /// [`BayesOneYear::refit`] says: by default moment-matched (decision 18)
    /// and, only if that cannot bring its `k̂` to the threshold, refitted
    /// with NUTS on the triangle extended by its year. A triangle with no cells in the year leaves the posterior unchanged
    /// (`k̂` 0, no refit). The prior's strength moves the view between the
    /// Bayesian chain ladder's (`cv = f64::INFINITY`) and a narrower one
    /// that tends, as `cv -> 0`, to the strong-prior limit, in which only
    /// the development pattern learns from the year. There is no view over
    /// segments: `BayesBf` fits one segment.
    ///
    /// ```
    /// use prospicio_bayes::nuts::Sampler;
    /// use prospicio_reserving::bayes::{BayesBf, BayesOneYear};
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
    /// use prospicio_prob::Distribution;
    ///
    /// let origin = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023].map(Month::january);
    /// let premium = [200.0, 200.0, 200.0, 200.0, 220.0, 220.0, 220.0, 240.0, 240.0, 260.0];
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&[12, 24, 36, 48, 12, 24, 36, 12, 24, 12]),
    ///     values: &[
    ///         ("paid", &[100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]),
    ///         ("premium", &premium),
    ///     ],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })?;
    /// let model = BayesBf {
    ///     apriori: 0.85,
    ///     cv: 0.1,
    ///     sampler: Sampler { tune: 300, draws: 300, ..Sampler::default() },
    ///     n_sims: 500,
    ///     ..BayesBf::default()
    /// };
    /// let view = model.one_year(&tri, "paid", "premium", &BayesOneYear::default())?;
    /// assert_eq!(view.cdr.n_components(), 4);
    /// // 2020 is fully developed: no cell in the year, no change.
    /// assert!(view.cdr.draw_matrix().chunks(4).all(|row| row[0] == 0.0));
    /// assert!(view.cdr.std_dev() < view.fit.reserves.std_dev());
    /// # Ok::<(), prospicio_reserving::Error>(())
    /// ```
    pub fn one_year(
        &self,
        triangle: &Triangle,
        column: &str,
        exposure: &str,
        settings: &BayesOneYear,
    ) -> Result<BayesOneYearFit<BayesBfFit>> {
        self.validate()?;
        settings.validate()?;
        let segment = triangle.segment(column)?;
        let exposures = origin_exposure(&segment, &triangle.segment(exposure)?, exposure)?;
        let fit = self.fit_segment(&segment, column, exposure, exposures)?;
        let cells = BfRefit::new(self, &fit);
        let year = OdpYear::new(
            &cells,
            self.process,
            triangle,
            &segment,
            &fit.chain_ladder.latest_position,
        );
        let draws = one_year::simulate(
            &year,
            &fit.posterior,
            settings,
            settings.sims(self.n_sims),
            one_year_seed(derive_seed(self.seed, Stream::Predictive)),
        )?;
        let hasher = input_hasher(column, exposure, &segment, &fit.chain_ladder, &fit.exposure);
        let mut provenance = self.provenance(column, exposure, &fit.model, hasher);
        provenance.model = "bayes_bf_one_year".into();
        let provenance = one_year::provenance(provenance, settings, settings.sims(self.n_sims));
        one_year::assemble(fit, draws, &segment, settings, provenance)
    }

    /// The fit of one segment, with exposure `exposures` from the column
    /// named `exposure`.
    fn fit_segment(
        &self,
        segment: &Segment,
        column: &str,
        exposure: &str,
        exposures: Vec<f64>,
    ) -> Result<BayesBfFit> {
        let (chain_ladder, model, start) = self.prepare(segment, &exposures)?;
        let (posterior, diagnostics) =
            self.sample(&model, start, derive_seed(self.seed, Stream::Chains))?;

        let hasher = input_hasher(column, exposure, segment, &chain_ladder, &exposures);
        let latest = &chain_ladder.latest_position;
        let (no, nd) = (segment.n_origins, segment.n_dev);
        let (process, phi) = (self.process, model.phi);
        let reserves = posterior.predictive(
            vec!["origin".into()],
            segment.origins.iter().map(|&o| vec![o.into()]).collect(),
            self.n_sims,
            derive_seed(self.seed, Stream::Predictive),
            self.provenance(column, exposure, &model, hasher),
            |theta, rng: &mut StreamRng, row| {
                for (w, reserve) in row.iter_mut().enumerate() {
                    *reserve = (latest[w] + 1..nd)
                        .map(|d| {
                            process_draw(process, phi, natural_eta(no, theta, w, d).exp(), rng)
                        })
                        .sum();
                }
            },
        )?;
        Ok(BayesBfFit {
            scale: model.phi,
            prior_ultimate: model.prior_ultimate.clone(),
            exposure: exposures,
            chain_ladder,
            posterior,
            diagnostics,
            reserves,
            model,
        })
    }

    /// NUTS on `model` whitened at its mode (found from `start`, on the
    /// sampling scale `(u, beta)`), with the chains' `seed`; the draws on
    /// the natural scale `(alpha, beta)` and their diagnostics.
    fn sample(
        &self,
        model: &Model,
        start: Vec<f64>,
        seed: u64,
    ) -> Result<(PosteriorDraws, Diagnostics)> {
        let p = model.dim();
        let mode = mode(model, start)?;
        let whitened = Whitened::new(model, &mode)?;
        let draws = sample(
            &whitened,
            &vec![0.0; p],
            Sampler {
                seed,
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
        if !(self.apriori.is_finite() && self.apriori > 0.0) {
            return Err(Error::InvalidSetting {
                name: "apriori",
                value: self.apriori,
                expected: "a finite, positive loss ratio",
            });
        }
        // Infinity is the flat prior; NaN, zero and negatives are not.
        if self.cv.is_nan() || self.cv <= 0.0 {
            return Err(Error::InvalidSetting {
                name: "cv",
                value: self.cv,
                expected: "a positive coefficient of variation (infinity for a flat prior)",
            });
        }
        if !(self.development_sd.is_finite() && self.development_sd > 0.0) {
            return Err(Error::InvalidSetting {
                name: "development_sd",
                value: self.development_sd,
                expected: "a finite, positive prior standard deviation",
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

    /// The chain ladder, the model and the Newton start of one segment.
    fn prepare(
        &self,
        segment: &Segment,
        exposure: &[f64],
    ) -> Result<(ChainLadderFit, Model, Vec<f64>)> {
        let (fit, _) = prepare(segment, &segment.ages, Tail::default())?;
        let (no, nd) = (segment.n_origins, segment.n_dev);
        let mut cells = Vec::new();
        for (w, &last) in fit.chain_ladder.latest_position.iter().enumerate() {
            let mut previous = 0.0;
            for d in 0..=last {
                let cum = segment.get(w, d).expect("checked when prepared");
                cells.push((w, d, cum - previous));
                previous = cum;
            }
        }
        let phi = fit.scale;
        let shape = self.cv.powi(-2);
        let prior_ultimate: Vec<f64> = exposure.iter().map(|e| self.apriori * e).collect();
        // The posterior of mu_w given the pattern is a gamma of shape
        // shape + S_w / phi: proper only when that is positive.
        for w in 0..no {
            let sum: f64 = cells.iter().filter(|c| c.0 == w).map(|c| c.2).sum();
            if sum.is_nan() || shape + sum / phi <= 0.0 {
                let bound = if shape == 0.0 {
                    "so no log-linear mean exists".to_string()
                } else {
                    format!(
                        "at most -phi / cv^2 = {}, so the posterior of its mu is improper",
                        -shape * phi
                    )
                };
                return Err(Error::Bayes(format!(
                    "origin {}: the observed increments sum to {sum}, {bound}",
                    segment.origins[w]
                )));
            }
        }
        let mut column_sum = vec![0.0; nd];
        let mut column_prior = vec![0.0; nd];
        for &(_, d, x) in &cells {
            column_sum[d] += x;
        }
        for (w, &last) in fit.chain_ladder.latest_position.iter().enumerate() {
            for prior in &mut column_prior[..=last] {
                *prior += prior_ultimate[w];
            }
        }
        for (d, &sum) in column_sum.iter().enumerate() {
            if sum.is_nan() || sum <= 0.0 {
                return Err(Error::Bayes(format!(
                    "development age {} months: the observed increments sum to {sum}, so no \
                     log-linear mean exists",
                    segment.ages[d]
                )));
            }
        }
        // Start at the strong-prior mode: mu_w = m_w (u_w = 0) and the raw
        // pattern S_d / M_d (EVW, sec. 3.3). The posterior is concave, so
        // Newton reaches the mode from there.
        let mut start = vec![0.0; no];
        start.extend((0..nd).map(|d| (column_sum[d] / column_prior[d]).ln()));
        let model = Model {
            n_origins: no,
            n_dev: nd,
            cells,
            phi,
            shape,
            log_prior: prior_ultimate.iter().map(|m| m.ln()).collect(),
            prior_ultimate,
            development_precision: self.development_sd.powi(-2),
        };
        Ok((fit.chain_ladder, model, start))
    }

    fn provenance(
        &self,
        column: &str,
        exposure: &str,
        model: &Model,
        hasher: InputHasher,
    ) -> Provenance {
        let s = &self.sampler;
        Provenance::new("bayes_bf")
            .param("apriori", self.apriori)
            .param("cv", self.cv)
            .param("development_sd", self.development_sd)
            .param("scale", model.phi)
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
            .input_hash(hasher.finish())
    }
}

impl BayesBfFit {
    /// Posterior summary of each parameter: mean, sd, 5/50/95%
    /// quantiles, R-hat, bulk and tail ESS.
    pub fn summary(&self) -> Result<Vec<ParamSummary>> {
        Ok(self.posterior.summary()?)
    }

    /// PSIS-LOO of the observed increments, as
    /// [`BayesOdpFit::loo`](super::BayesOdpFit::loo): the pointwise term
    /// is the normalised density of the over-dispersed Poisson at
    /// `mu_w gamma_d` and `phi`, so the result compares with `BayesOdp`'s
    /// on the same triangle. A triangle with a negative increment has no
    /// such density and is an error.
    pub fn loo(&self) -> Result<Loo> {
        let model = &self.model;
        if let Some(&(w, d, x)) = model.cells.iter().find(|c| c.2 < 0.0) {
            return Err(Error::Bayes(format!(
                "loo needs non-negative increments, and origin {} at development age {} \
                 months is {x}",
                self.chain_ladder.origins[w], self.chain_ladder.development.development[d]
            )));
        }
        let n = model.cells.len();
        let s = self.posterior.n_draws();
        let rows: Vec<Vec<f64>> = self
            .posterior
            .values()
            .par_chunks_exact(self.posterior.dim())
            .map(|theta| {
                model
                    .cells
                    .iter()
                    .map(|&(w, d, x)| {
                        let mu = natural_eta(model.n_origins, theta, w, d).exp();
                        Family::Poisson.log_density(x, mu, model.phi, 1.0)
                    })
                    .collect::<prospicio_core::Result<Vec<f64>>>()
            })
            .collect::<prospicio_core::Result<_>>()?;
        let ll = rows.concat();
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
    /// `reserve`, the `exposure` and `prior_ultimate`, and the `mean` and
    /// `std_dev` of the posterior predictive reserve.
    pub fn to_long(&self) -> FitTable {
        let mut table = self.plug_in().to_long();
        push_moments(&mut table, "", &component_sums(&self.reserves));
        table
    }

    /// One row: the chain ladder's totals, the total `exposure` and
    /// `prior_ultimate`, the `scale`, and the `mean` and `std_dev` of the
    /// posterior predictive total reserve.
    pub fn totals(&self) -> FitTable {
        let fits = self.plug_in();
        let mut table = fits.totals();
        push_moments(&mut table, "", &segment_sums(&fits, &self.reserves));
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
                scale: self.scale,
                exposure: self.exposure.clone(),
                prior_ultimate: self.prior_ultimate.clone(),
            }],
        }
    }
}

impl ReserveFit for BayesBfFit {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        origin_columns(&self.exposure, &self.prior_ultimate)
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        total_columns(self.scale, &self.exposure, &self.prior_ultimate)
    }
}

pub(super) fn origin_columns(exposure: &[f64], prior: &[f64]) -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("exposure", exposure.to_vec()),
        ("prior_ultimate", prior.to_vec()),
    ]
}

pub(super) fn total_columns(
    scale: f64,
    exposure: &[f64],
    prior: &[f64],
) -> Vec<(&'static str, f64)> {
    vec![
        ("exposure", exposure.iter().sum()),
        ("prior_ultimate", prior.iter().sum()),
        ("scale", scale),
    ]
}

/// The chain ladder, scale and priors of a fit, for its long tables.
#[derive(Debug, Clone)]
pub(super) struct PlugIn {
    pub(super) chain_ladder: ChainLadderFit,
    pub(super) scale: f64,
    pub(super) exposure: Vec<f64>,
    pub(super) prior_ultimate: Vec<f64>,
}

impl ReserveFit for PlugIn {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        origin_columns(&self.exposure, &self.prior_ultimate)
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        total_columns(self.scale, &self.exposure, &self.prior_ultimate)
    }
}

/// The log posterior on `theta = (u_1..u_n, beta_1..beta_m)`, sampled with
/// `u_w = alpha_w - log m_w`.
#[derive(Debug, Clone)]
struct Model {
    n_origins: usize,
    n_dev: usize,
    /// Observed increments: (origin, development position, value).
    cells: Vec<(usize, usize, f64)>,
    phi: f64,
    /// Shape `1 / cv^2` of the gamma prior on each `mu_w`; 0 is flat on
    /// `alpha_w`.
    shape: f64,
    /// Prior mean `m_w` of each `mu_w`.
    prior_ultimate: Vec<f64>,
    /// `log m_w`.
    log_prior: Vec<f64>,
    /// `1 / development_sd^2`.
    development_precision: f64,
}

impl Model {
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = (1..=self.n_origins)
            .map(|w| format!("alpha[{w}]"))
            .collect();
        names.extend((1..=self.n_dev).map(|d| format!("beta[{d}]")));
        names
    }

    /// `eta_wd` on the sampling scale `theta = (u, beta)`, with
    /// `u_w = alpha_w - log m_w`: centred on the prior, `u` keeps full
    /// precision under a tight prior, where `alpha_w` near `log m_w` (about
    /// 15) would round its steps to 2e-15.
    fn eta(&self, theta: &[f64], w: usize, d: usize) -> f64 {
        theta[w] + self.log_prior[w] + theta[self.n_origins + d]
    }

    /// `(alpha, beta)` from the sampling scale's `(u, beta)`.
    fn natural(&self, theta: &[f64]) -> Vec<f64> {
        let mut natural = theta.to_vec();
        for (a, l) in natural.iter_mut().zip(&self.log_prior) {
            *a += l;
        }
        natural
    }

    /// `(u, beta)` from the natural scale's `(alpha, beta)`.
    fn sampling(&self, natural: &[f64]) -> Vec<f64> {
        let mut theta = natural.to_vec();
        for (u, l) in theta.iter_mut().zip(&self.log_prior) {
            *u -= l;
        }
        theta
    }
}

/// `eta_wd` from a draw of `(alpha, beta)` with `n_origins` origins.
fn natural_eta(n_origins: usize, theta: &[f64], w: usize, d: usize) -> f64 {
    theta[w] + theta[n_origins + d]
}

/// The input hash of a fit: the column names, the segment and the
/// exposures.
pub(super) fn input_hasher(
    column: &str,
    exposure: &str,
    segment: &Segment,
    chain_ladder: &ChainLadderFit,
    exposures: &[f64],
) -> InputHasher {
    let mut hasher = InputHasher::new();
    hasher.str(column);
    hasher.str(exposure);
    hash_segment(&mut hasher, segment, chain_ladder, &segment.ages);
    hasher.f64s(exposures);
    hasher
}

/// A fitted [`BayesBf`] as the one-year view sees it ([`OdpCells`],
/// decision 17): the draws are stored as `(alpha, beta)`, the model
/// samples `(u, beta)` with `u_w = alpha_w - log m_w`, a shift, so the log
/// posterior moves to `u` and needs no Jacobian, and a refit starts Newton
/// at the opening posterior mean moved to `u`. Every origin and age has an
/// observed cell, so the draws carry every effect a future cell needs; the
/// prior ultimates and `phi` stay at their opening values.
struct BfRefit<'a> {
    bf: &'a BayesBf,
    model: &'a Model,
    /// The opening posterior mean on the sampling scale, the refits'
    /// Newton start.
    start: Vec<f64>,
}

impl<'a> BfRefit<'a> {
    fn new(bf: &'a BayesBf, fit: &'a BayesBfFit) -> Self {
        let model = &fit.model;
        Self {
            bf,
            model,
            start: model.sampling(&fit.posterior.mean()),
        }
    }

    fn extended(&self, extra: &[(usize, usize, f64)]) -> Model {
        let mut extended = self.model.clone();
        extended.cells.extend_from_slice(extra);
        extended
    }
}

impl OdpCells for BfRefit<'_> {
    fn eta(&self, theta: &[f64], w: usize, d: usize) -> f64 {
        natural_eta(self.model.n_origins, theta, w, d)
    }

    fn phi(&self) -> f64 {
        self.model.phi
    }

    fn log_posterior(&self, theta: &[f64], extra: &[(usize, usize, f64)]) -> f64 {
        let mut grad = vec![0.0; theta.len()];
        if extra.is_empty() {
            // Moment matching's many calls on the observed triangle.
            return self
                .model
                .log_posterior(&self.model.sampling(theta), &mut grad);
        }
        self.extended(extra)
            .log_posterior(&self.model.sampling(theta), &mut grad)
    }

    /// Sampled as [`BayesBf::fit`] samples.
    fn refit(&self, extra: &[(usize, usize, f64)], seed: u64) -> Result<(PosteriorDraws, bool)> {
        let (posterior, diagnostics) =
            self.bf
                .sample(&self.extended(extra), self.start.clone(), seed)?;
        Ok((posterior, !diagnostics.warnings.is_empty()))
    }
}

/// The log posterior of [`BayesBf`], on `(u, beta)`. Each term is concave and the normal
/// prior on `beta` fixes the one direction the likelihood leaves flat
/// (`alpha + k`, `beta - k`), so it is strictly concave and [`mode`]
/// never needs a ridge.
impl Posterior for Model {
    fn dim(&self) -> usize {
        self.n_origins + self.n_dev
    }

    fn log_posterior(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        let no = self.n_origins;
        let mut f = 0.0;
        // Gamma(shape, rate shape / m) on mu = exp(alpha), with the
        // Jacobian: shape alpha - shape exp(alpha) / m, here less its value
        // at alpha = log m, in u = alpha - log m. Centred, a tight prior
        // (shape 1e6, say) adds terms near 0 rather than near
        // shape log m, which would swamp the rest of f in rounding.
        for w in 0..no {
            let u = theta[w];
            let em1 = u.exp_m1();
            f += self.shape * (u - em1);
            grad[w] = -self.shape * em1;
        }
        for j in no..self.dim() {
            f -= 0.5 * self.development_precision * theta[j] * theta[j];
            grad[j] = -self.development_precision * theta[j];
        }
        for &(w, d, x) in &self.cells {
            let eta = self.eta(theta, w, d);
            let mu = eta.exp();
            f += (x * eta - mu) / self.phi;
            let g = (x - mu) / self.phi;
            grad[w] += g;
            grad[no + d] += g;
        }
        f
    }

    fn negative_hessian(&self, theta: &[f64]) -> Vec<f64> {
        let (no, p) = (self.n_origins, self.dim());
        let mut h = vec![0.0; p * p];
        for w in 0..no {
            h[w * p + w] = self.shape * theta[w].exp();
        }
        for j in no..p {
            h[j * p + j] = self.development_precision;
        }
        for &(w, d, _) in &self.cells {
            let weight = self.eta(theta, w, d).exp() / self.phi;
            let b = no + d;
            h[w * p + w] += weight;
            h[b * p + b] += weight;
            h[w * p + b] += weight;
            h[b * p + w] += weight;
        }
        h
    }
}

/// The log posterior written once over [`Scalar`](super::scalar::Scalar),
/// the oracle of the compartmental note's decision 4: test-only, beside
/// the hand-written [`Posterior::log_posterior`], which stays the fast
/// path.
#[cfg(test)]
impl super::autodiff::GenericDensity for Model {
    fn n_params(&self) -> usize {
        self.dim()
    }

    fn log_density<S: super::scalar::Scalar>(&self, theta: &[S]) -> S {
        let no = self.n_origins;
        let mut f = S::from_f64(0.0);
        for &u in &theta[..no] {
            f += (u - u.exp_m1()) * self.shape;
        }
        for &b in &theta[no..] {
            f -= b * b * (0.5 * self.development_precision);
        }
        for &(w, d, x) in &self.cells {
            let eta = theta[w] + self.log_prior[w] + theta[no + d];
            f += (eta * x - eta.exp()) / self.phi;
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bayes::BayesOdp;
    use crate::bayes::laplace::tests::assert_gradient;
    use crate::triangle::tests::{GENINS, annual};
    use crate::{BornhuetterFerguson, ChainLadder, Lag};
    use crate::{DevelopmentColumn, Grain, Long};
    use prospicio_bayes::nuts::LogDensity;
    use prospicio_core::{Month, Period};
    use prospicio_prob::Distribution;

    /// A cumulative annual triangle with `values` and a `premium` column,
    /// one premium per origin repeated on each age.
    fn with_premium(first_year: i32, rows: &[&[f64]], premium: &[f64]) -> Triangle {
        let (mut origin, mut ages, mut values, mut prem) = (vec![], vec![], vec![], vec![]);
        for (k, row) in rows.iter().enumerate() {
            for (d, &v) in row.iter().enumerate() {
                origin.push(Month::january(first_year + k as i32));
                ages.push(12 * (d as Lag + 1));
                values.push(v);
                prem.push(premium[k]);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("values", &values), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    /// GenIns with R ChainLadder's `ClarkCapeCod` premium,
    /// `10,000,000 + 400,000 k`.
    fn genins() -> Triangle {
        let premium: Vec<f64> = (0..10).map(|k| 1e7 + 4e5 * k as f64).collect();
        with_premium(2001, &GENINS, &premium)
    }

    fn quick(seed: u64) -> BayesBf {
        BayesBf {
            apriori: 0.5,
            cv: 0.2,
            sampler: Sampler {
                tune: 300,
                draws: 300,
                target_accept: 0.95,
                ..Sampler::default()
            },
            n_sims: 2_000,
            seed,
            ..BayesBf::default()
        }
    }

    fn prepared(tri: &Triangle, model: &BayesBf) -> (ChainLadderFit, Model, Vec<f64>) {
        let segment = tri.segment("values").unwrap();
        let exposure =
            origin_exposure(&segment, &tri.segment("premium").unwrap(), "premium").unwrap();
        model.prepare(&segment, &exposure).unwrap()
    }

    /// EVW's strong-prior limit per origin (their sec. 3.3, with a flat
    /// prior on `beta`): `mu_w = m_w` and independent
    /// `gamma_d ~ Gamma(S_d / phi, M_d / phi)`, so the reserve given the
    /// parameters, `m_w` times the future `gamma_d`, has mean
    /// `m_w sum S_d / M_d` and variance `m_w^2 sum phi S_d / M_d^2`. Returns
    /// the means and sds by origin, and those of the total.
    fn strong_prior_limit(fit: &BayesBfFit) -> (Vec<(f64, f64)>, (f64, f64)) {
        let model = &fit.model;
        let (no, nd, phi) = (model.n_origins, model.n_dev, model.phi);
        let latest = &fit.chain_ladder.latest_position;
        let (mut s, mut m) = (vec![0.0; nd], vec![0.0; nd]);
        for &(w, d, x) in &model.cells {
            s[d] += x;
            m[d] += model.prior_ultimate[w];
        }
        let future = |w: usize| latest[w] + 1..nd;
        let by_origin: Vec<(f64, f64)> = (0..no)
            .map(|w| {
                let mw = model.prior_ultimate[w];
                let mean: f64 = future(w).map(|d| mw * s[d] / m[d]).sum();
                let var: f64 = future(w)
                    .map(|d| mw * mw * phi * s[d] / (m[d] * m[d]))
                    .sum();
                (mean, var.sqrt())
            })
            .collect();
        let total_mean = by_origin.iter().map(|r| r.0).sum();
        let total_var: f64 = (0..nd)
            .map(|d| {
                let open: f64 = (0..no)
                    .filter(|&w| d > latest[w])
                    .map(|w| model.prior_ultimate[w])
                    .sum();
                open * open * phi * s[d] / (m[d] * m[d])
            })
            .sum();
        (by_origin, (total_mean, total_var.sqrt()))
    }

    fn sampler(draws: usize) -> Sampler {
        Sampler {
            tune: 300,
            draws,
            target_accept: 0.95,
            ..Sampler::default()
        }
    }

    #[test]
    fn the_gradient_and_hessian_match_finite_differences() {
        for cv in [0.2, f64::INFINITY] {
            let (_, model, start) = prepared(&genins(), &BayesBf { cv, ..quick(0) });
            let x: Vec<f64> = start
                .iter()
                .enumerate()
                .map(|(j, v)| v + 0.01 * (j as f64 - 5.0))
                .collect();
            assert_gradient(|t, g| model.log_posterior(t, g), &x);
            // The analytic negative Hessian against the trait's default
            // central differences of the gradient.
            struct Numeric<'a>(&'a Model);
            impl Posterior for Numeric<'_> {
                fn dim(&self) -> usize {
                    self.0.dim()
                }
                fn log_posterior(&self, t: &[f64], g: &mut [f64]) -> f64 {
                    self.0.log_posterior(t, g)
                }
            }
            let (exact, numeric) = (
                model.negative_hessian(&x),
                Numeric(&model).negative_hessian(&x),
            );
            for (a, b) in exact.iter().zip(&numeric) {
                assert!((a - b).abs() <= 1e-5 * (1.0 + a.abs()), "{a} vs {b}");
            }
            let mode = mode(&model, start).unwrap();
            let whitened = Whitened::new(&model, &mode).unwrap();
            let z: Vec<f64> = (0..model.dim())
                .map(|j| 0.3 * ((j % 3) as f64 - 1.0))
                .collect();
            assert_gradient(|z, g| whitened.log_density(z, g).unwrap(), &z);
        }
    }

    #[test]
    fn the_generic_density_matches_the_hand_written_one() {
        use crate::bayes::autodiff::{assert_oracle, posterior_points};
        for (label, cv) in [("BayesBf cv 0.2", 0.2), ("BayesBf flat", f64::INFINITY)] {
            let (_, model, start) = prepared(&genins(), &BayesBf { cv, ..quick(0) });
            let mode = mode(&model, start).unwrap();
            let whitened = Whitened::new(&model, &mode).unwrap();
            let points = posterior_points(&whitened, model.dim(), 20);
            assert_oracle(label, &model, &points);
        }
    }

    /// With a flat prior on `alpha` the mode's fitted means are the chain
    /// ladder's (EVW, eq. 2.1-2.2 and sec. 2.3), up to the vague prior on
    /// `beta`, which at sd 1,000 moves them by less than 1e-4.
    #[test]
    fn a_flat_prior_has_the_chain_ladder_at_its_mode() {
        let model = BayesBf {
            cv: f64::INFINITY,
            development_sd: 1e3,
            ..quick(0)
        };
        let tri = genins();
        let (_, m, start) = prepared(&tri, &model);
        let theta = mode(&m, start).unwrap();
        let segment = tri.segment("values").unwrap();
        let (fit, _) = prepare(&segment, &segment.ages, Tail::default()).unwrap();
        for &(w, d, _) in &m.cells {
            let mu = m.eta(&theta, w, d).exp();
            let expected = fit.fitted[w * segment.n_dev + d];
            assert!(
                (mu / expected - 1.0).abs() < 1e-4,
                "({w}, {d}): {mu} vs {expected}"
            );
        }
    }

    /// A strong prior fixes `mu_w` near `m_w`, and the mode of `gamma_d`
    /// near the raw pattern `S_d / M_d` (the Newton start), which on the
    /// log scale with a flat prior on `beta_d` is also EVW's posterior
    /// mean of `gamma_d` (their eq. 3.15). At `cv = 1e-3` the data move
    /// `mu_w` by less than 1e-3 of `m_w`.
    #[test]
    fn a_strong_prior_fixes_mu_at_the_prior() {
        let model = BayesBf {
            cv: 1e-3,
            development_sd: 100.0,
            ..quick(0)
        };
        let (_, m, start) = prepared(&genins(), &model);
        let theta = mode(&m, start.clone()).unwrap();
        for (w, u) in theta.iter().take(m.n_origins).enumerate() {
            let ratio = u.exp();
            assert!((ratio - 1.0).abs() < 1e-3, "{w}: mu / m = {ratio}");
        }
        for d in 0..m.n_dev {
            let raw = start[m.n_origins + d].exp();
            let gamma = theta[m.n_origins + d].exp();
            assert!((gamma / raw - 1.0).abs() < 2e-3, "{d}: {gamma} vs {raw}");
        }
    }

    /// The flat prior gives `BayesOdp`'s reserves within Monte Carlo
    /// error: the means within `0.1 s` and the sds within 10% by origin
    /// and in total, `s` the sd of `BayesOdp`'s reserve.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_flat_prior_reproduces_bayes_odp() {
        let tri = genins();
        let bf = BayesBf {
            cv: f64::INFINITY,
            sampler: sampler(1000),
            n_sims: 10_000,
            ..quick(1)
        }
        .fit(&tri, "values", "premium")
        .unwrap();
        let odp = BayesOdp {
            sampler: sampler(1000),
            n_sims: 10_000,
            seed: 2,
            ..BayesOdp::default()
        }
        .fit(&tri, "values")
        .unwrap();
        let mut keys: Vec<Option<Period>> = (2002..=2010).map(|y| Some(Period::year(y))).collect();
        keys.push(None);
        for key in keys {
            let pick = |r: &PredictiveDistribution| match key {
                Some(o) => r.marginal(&vec![o.into()]).unwrap(),
                None => r.total().clone(),
            };
            let (a, b) = (pick(&bf.reserves), pick(&odp.reserves));
            let s = b.std_dev();
            assert!(
                (a.mean() - b.mean()).abs() < 0.1 * s,
                "{key:?}: mean {} vs {} (sd {s})",
                a.mean(),
                b.mean()
            );
            assert!(
                (a.std_dev() / s - 1.0).abs() < 0.10,
                "{key:?}: sd {} vs {s}",
                a.std_dev()
            );
        }
    }

    /// A very tight prior gives EVW's strong-prior limit: with no process
    /// error, each origin's and the total reserve's mean within `0.1 s`
    /// and sd within 10% of the closed form's, `s` its sd. When the prior
    /// ultimates are the chain ladder's, the limit's mean is the classical
    /// Bornhuetter–Ferguson reserve, which is then the chain ladder's.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_tight_prior_reproduces_the_bornhuetter_ferguson_limit() {
        let tri = genins();
        let tight = BayesBf {
            cv: 1e-3,
            development_sd: 100.0,
            process: ProcessDistribution::None,
            sampler: sampler(1000),
            ..quick(3)
        };
        let check = |fit: &BayesBfFit| {
            let (by_origin, total) = strong_prior_limit(fit);
            for (k, &(mean, sd)) in by_origin.iter().enumerate().skip(1) {
                let o = fit.chain_ladder.origins[k];
                let got = fit.reserves.marginal(&vec![o.into()]).unwrap();
                assert!(
                    (got.mean() - mean).abs() < 0.1 * sd,
                    "{o}: {} vs {mean} (sd {sd})",
                    got.mean()
                );
                assert!(
                    (got.std_dev() / sd - 1.0).abs() < 0.1,
                    "{o}: sd {} vs {sd}",
                    got.std_dev()
                );
            }
            let got = fit.reserves.total();
            assert!(
                (got.mean() - total.0).abs() < 0.1 * total.1,
                "total {} vs {total:?}",
                got.mean()
            );
            assert!(
                (got.std_dev() / total.1 - 1.0).abs() < 0.1,
                "total sd {} vs {total:?}",
                got.std_dev()
            );
            total
        };
        check(&tight.fit(&tri, "values", "premium").unwrap());

        let cl = ChainLadder::default().fit(&tri, "values").unwrap();
        let rows: Vec<&[f64]> = GENINS.to_vec();
        let at_cl = with_premium(2001, &rows, &cl.ultimate);
        let classical = BornhuetterFerguson {
            apriori: 1.0.into(),
            ..Default::default()
        }
        .fit(&at_cl, "values", "premium")
        .unwrap()
        .total_reserve();
        let fit = BayesBf {
            apriori: 1.0,
            ..tight
        }
        .fit(&at_cl, "values", "premium")
        .unwrap();
        let (mean, _) = check(&fit);
        assert!(
            (mean / classical - 1.0).abs() < 1e-9,
            "{mean} vs {classical}"
        );
        assert!((classical / cl.total_reserve() - 1.0).abs() < 1e-9);
    }

    /// Between the limits, the total mean reserve moves monotonically from
    /// the flat prior's to the tight prior's as `cv` falls.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn an_intermediate_prior_sits_between_the_limits() {
        let tri = genins();
        let total = |cv: f64| {
            BayesBf {
                cv,
                process: ProcessDistribution::None,
                ..quick(4)
            }
            .fit(&tri, "values", "premium")
            .unwrap()
            .reserves
            .total()
            .mean()
        };
        let means: Vec<f64> = [f64::INFINITY, 0.5, 0.2, 0.05, 1e-3]
            .into_iter()
            .map(total)
            .collect();
        // apriori 0.5 is above GenIns' chain-ladder loss ratio (about 0.45),
        // so the reserve rises with the prior's weight.
        for pair in means.windows(2) {
            assert!(pair[0] < pair[1], "{means:?}");
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn a_seed_replays_whatever_the_threads() {
        let tri = genins();
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
        let mut d = quick(3);
        d.sampler.seed = 99;
        assert_eq!(
            d.fit(&tri, "values", "premium").unwrap().posterior,
            a.posterior
        );
    }

    #[test]
    fn errors() {
        let tri = genins();
        let fit = |m: BayesBf| m.fit(&tri, "values", "premium");
        for (model, name) in [
            (
                BayesBf {
                    apriori: 0.0,
                    ..quick(0)
                },
                "apriori",
            ),
            (
                BayesBf {
                    cv: 0.0,
                    ..quick(0)
                },
                "cv",
            ),
            (
                BayesBf {
                    cv: f64::NAN,
                    ..quick(0)
                },
                "cv",
            ),
            (
                BayesBf {
                    development_sd: f64::INFINITY,
                    ..quick(0)
                },
                "development_sd",
            ),
            (
                BayesBf {
                    n_sims: 0,
                    ..quick(0)
                },
                "n_sims",
            ),
        ] {
            assert!(
                matches!(fit(model), Err(Error::InvalidSetting { name: n, .. }) if n == name),
                "{name}"
            );
        }
        assert!(matches!(
            quick(0).fit(&tri, "values", "earned"),
            Err(Error::UnknownColumn(_))
        ));
        let zero = with_premium(
            2020,
            &[&[1.0, 2.0, 3.0], &[1.0, 2.0], &[1.0]],
            &[1.0, 0.0, 1.0],
        );
        assert!(matches!(
            quick(0).fit(&zero, "values", "premium"),
            Err(Error::InvalidExposure { .. })
        ));

        // The newest origin's only cell is negative: improper under a flat
        // prior, proper under an informative one.
        let rows: [&[f64]; 4] = [
            &[100.0, 150.0, 160.0, 165.0],
            &[110.0, 170.0, 180.0],
            &[120.0, 175.0],
            &[-1.0],
        ];
        let negative = with_premium(2020, &rows, &[200.0; 4]);
        let flat = BayesBf {
            cv: f64::INFINITY,
            ..quick(0)
        };
        let e = flat.fit(&negative, "values", "premium").unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("origin 2023")
                && m.contains("no log-linear mean")),
            "{e}"
        );
        // Only a weak prior (shape 1 / cv^2 at most 1 / phi, the cell over
        // phi) leaves the origin improper.
        let fit = quick(0).fit(&negative, "values", "premium").unwrap();
        assert!(fit.reserves.total().mean().is_finite());
        let weak = BayesBf {
            cv: 1e3,
            ..quick(0)
        };
        let e = weak.fit(&negative, "values", "premium").unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("origin 2023") && m.contains("improper")),
            "{e}"
        );
        // The last development column falls.
        let falling = with_premium(
            2020,
            &[
                &[100.0, 150.0, 160.0, 155.0],
                &[110.0, 170.0, 180.0],
                &[120.0, 175.0],
                &[130.0],
            ],
            &[200.0; 4],
        );
        let e = quick(0).fit(&falling, "values", "premium").unwrap_err();
        assert!(
            matches!(&e, Error::Bayes(m) if m.starts_with("development age 48 months")),
            "{e}"
        );
        assert!(matches!(
            quick(0).fit(&annual(2020, &[&[1.0, 2.0], &[1.0]]), "values", "values"),
            Err(Error::Bootstrap(_))
        ));
    }

    #[test]
    fn the_fit_and_its_tables() {
        let fit = quick(5).fit(&genins(), "values", "premium").unwrap();
        assert_eq!(fit.reserves.dims(), ["origin"]);
        assert_eq!(fit.reserves.n_components(), 10);
        let oldest = fit
            .reserves
            .marginal(&vec![Period::year(2001).into()])
            .unwrap();
        assert_eq!((oldest.mean(), oldest.std_dev()), (0.0, 0.0));
        assert_eq!(fit.posterior.names().len(), 20);
        assert_eq!(fit.posterior.names()[10], "beta[1]");
        assert_eq!(fit.prior_ultimate[9], 0.5 * 13.6e6);
        assert_eq!(fit.exposure[0], 1e7);
        let long = fit.to_long();
        assert_eq!(long.n_rows(), 10);
        assert_eq!(long.column("prior_ultimate").unwrap(), fit.prior_ultimate);
        assert!(long.column("mean").is_some() && long.column("std_dev").is_some());
        let totals = fit.totals();
        assert_eq!(totals.column("scale").unwrap(), [fit.scale]);
        assert_eq!(totals.column("exposure").unwrap(), [118e6]);
        assert_eq!(fit.development_table().n_rows(), 10);
        assert_eq!(fit.summary().unwrap().len(), 20);
        let provenance = fit.reserves.provenance();
        assert_eq!(provenance.model, "bayes_bf");
        for key in [
            "apriori",
            "cv",
            "development_sd",
            "scale",
            "exposure",
            "seed",
        ] {
            assert!(
                provenance.parameters.iter().any(|(k, _)| k == key),
                "{key} missing"
            );
        }
        // The scale is the ODP bootstrap's (and BayesOdp's), the chain
        // ladder's Pearson scale.
        let boot = crate::OdpBootstrap {
            n_sims: 10,
            ..Default::default()
        }
        .fit(&genins(), "values")
        .unwrap();
        assert_eq!(fit.scale, boot.scale);
        let short = BayesBf {
            sampler: Sampler {
                tune: 100,
                draws: 25,
                ..Sampler::default()
            },
            n_sims: 100,
            ..quick(5)
        };
        let loo = short
            .fit(&genins(), "values", "premium")
            .unwrap()
            .loo()
            .unwrap();
        assert_eq!(loo.pareto_k.len(), 55);
        assert!(loo.estimate.elpd.is_finite());
    }

    /// The one-year conformance suite (decision 17) on GenIns with an
    /// informative and a flat prior, with a fully developed square as the
    /// triangle without a year.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_one_year_view_conforms() {
        let square = with_premium(
            2001,
            &[
                &[357.0, 1124.0, 1735.0, 2218.0],
                &[352.0, 1236.0, 2170.0, 3353.0],
                &[290.0, 1292.0, 2219.0, 3235.0],
                &[310.0, 1418.0, 2195.0, 3757.0],
            ],
            &[5000.0, 6000.0, 6000.0, 7000.0],
        );
        for (name, cv) in [("BayesBf cv 0.2", 0.2), ("BayesBf flat", f64::INFINITY)] {
            let model = BayesBf { cv, ..quick(6) };
            let fitted: Vec<(Triangle, Segment, BayesBfFit)> = [genins(), square.clone()]
                .into_iter()
                .map(|tri| {
                    let segment = tri.segment("values").unwrap();
                    let exposure =
                        origin_exposure(&segment, &tri.segment("premium").unwrap(), "premium")
                            .unwrap();
                    let fit = model
                        .fit_segment(&segment, "values", "premium", exposure)
                        .unwrap();
                    (tri, segment, fit)
                })
                .collect();
            let cells: Vec<BfRefit> = fitted
                .iter()
                .map(|(_, _, fit)| BfRefit::new(&model, fit))
                .collect();
            let years: Vec<_> = fitted
                .iter()
                .zip(&cells)
                .map(|((tri, segment, fit), cells)| {
                    OdpYear::new(
                        cells,
                        model.process,
                        tri,
                        segment,
                        &fit.chain_ladder.latest_position,
                    )
                })
                .collect();
            crate::bayes::one_year::conformance::check(
                name,
                &years[0],
                &fitted[0].2.posterior,
                &years[1],
                &fitted[1].2.posterior,
            );
        }
    }

    #[test]
    fn the_one_year_view_reweights_the_fits_posterior() {
        let model = BayesBf {
            n_sims: 1_000,
            ..quick(7)
        };
        let settings = BayesOneYear {
            refit: Some(crate::bayes::OneYearRefit::ReweightOnly),
            ..BayesOneYear::default()
        };
        let view = model
            .one_year(&genins(), "values", "premium", &settings)
            .unwrap();
        // The posterior and lifetime reserves are fit()'s, draw for draw.
        let fit = model.fit(&genins(), "values", "premium").unwrap();
        assert_eq!(view.fit.posterior, fit.posterior);
        assert_eq!(view.fit.reserves.draw_matrix(), fit.reserves.draw_matrix());
        assert_eq!(view.cdr.dims(), ["origin"]);
        assert_eq!(view.pareto_k.len(), 1_000);
        assert!(view.refitted.is_empty());
        // 2001 is at the last age: no cell, no change.
        assert!(view.cdr.draw_matrix().chunks(10).all(|r| r[0] == 0.0));
        assert_eq!(view.opening_reserve[0], 0.0);
        // The tables carry the prior's columns, as the lifetime fit's do.
        let long = view.to_long();
        assert_eq!(long.n_rows(), 10);
        assert_eq!(long.column("prior_ultimate").unwrap(), fit.prior_ultimate);
        assert!(long.column("opening_reserve").is_some());
        let totals = view.totals();
        assert_eq!(totals.column("exposure").unwrap(), [118e6]);
        assert_eq!(
            totals.column("flagged_share").unwrap(),
            [view.flagged_share()]
        );
        let provenance = view.cdr.provenance();
        assert_eq!(provenance.model, "bayes_bf_one_year");
        for key in ["apriori", "cv", "k_threshold", "refit", "exposure"] {
            assert!(
                provenance.parameters.iter().any(|(k, _)| k == key),
                "{key} missing"
            );
        }
        // The one-year view is narrower than the lifetime view.
        assert!(view.cdr.total().std_dev() < fit.reserves.total().std_dev());
        let e = model
            .one_year(
                &genins(),
                "values",
                "premium",
                &BayesOneYear {
                    k_threshold: f64::NAN,
                    ..settings
                },
            )
            .unwrap_err();
        assert!(
            matches!(
                e,
                Error::InvalidSetting {
                    name: "k_threshold",
                    ..
                }
            ),
            "{e}"
        );
    }
}
