//! The `Compartmental` model: its builder, fits, one-year views and prior predictive check.

use prospicio_bayes::nuts::{PosteriorDraws, Sampler, sample};
use prospicio_core::{Lag, StreamRng};
use prospicio_math::special::{norm_cdf, norm_quantile};
use prospicio_prob::{
    Distribution, Gamma, InputHasher, KeyValue, PredictiveDistribution, Provenance,
};

use super::super::laplace::{Posterior, Whitened};
use super::super::one_year::{
    self, BayesOneYear, BayesOneYearFit, BayesOneYearSegment, LightRefit, OneYearRefit,
};
use super::super::{Diagnostics, Stream, derive_seed, one_year_seed, pilot_seed};
use super::data::{Data, Hierarchy, Response, SquareGrain, premium_of};
use super::model::Model;
use super::one_os;
#[cfg(doc)]
use super::priors::PositivePrior;
use super::priors::{
    CompartmentalPriors, MAX_INCURRED_ONLY_KP_SD, MIN_ORDERED_MASS, PRIOR_SHARE_WARNING, draw_half,
    lognormal, ordered_rates, payment_rate_prior_share, rate_order_bound,
};
use super::results::{
    CompartmentalFit, CompartmentalFits, CompartmentalOneYearFits, CompartmentalPaths,
    CompartmentalSegment,
};
use super::settings::{
    CompartmentalLikelihood, CompartmentalStart, DevelopmentLevels, Fitted, Outstanding,
    PaidResponse, RateOrder, Scale,
};
use crate::bootstrap::origin_keys;
use crate::dependence::SegmentDependence;
use crate::error::{Error, Result};
use crate::pattern_draws::predictive_keyed_with_shares;
use crate::segments::fit_each;
use crate::triangle::Triangle;

/// The warning of a one-year view that reweights only because the default
/// light refit of development effects (model 2) was turned off
/// ([`BayesOneYear::no_light_refit`]).
const APPROXIMATE: &str = "the one-year view reweights its flagged simulations only, as the light refit of development effects (model 2) was turned off: moment matching rescues none of them. The view is approximate, about 0.4 sd of the total claims development result off a refit per simulation on company 337; leave light_refit at its default to refit them with a shorter sampler (within about 0.03 sd there, at about 20 times the run time of reweighting only), or refit (with max_refits as a budget) for the fit's own sampler";

/// Gesmann and Morris' (2020) hierarchical compartmental reserving model
/// with one outstanding compartment (`docs/design/reserving-compartmental.md`).
/// Claims flow from exposure into outstanding as they are reported, at rate
/// `k_er`, and from outstanding into paid as they settle, at rate `k_p`:
///
/// ```text
/// dEX/dt = -k_er EX                      EX(0) = Π_i
/// dOS/dt =  k_er RLR_i EX - k_p OS       OS(0) = 0
/// dPD/dt =  k_p RRF_i OS                 PD(0) = 0
/// ```
///
/// for origin `i` with premium `Π_i` and development time `t` in years. In
/// closed form (the paper's Eq. 2), with `g(t) = (e^(-k_p t) - e^(-k_er t))
/// / (k_er - k_p)` computed stably through `k_er = k_p`:
///
/// ```text
/// OS_i(t) = Π_i RLR_i k_er g(t)
/// PD_i(t) = Π_i RLR_i RRF_i [(1 - e^(-k_p t)) - k_p g(t)]
/// ```
///
/// so `RLR_i` is the reported loss ratio, `RRF_i` the reserve robustness
/// factor (paid over reported) and `RLR_i RRF_i` the ultimate loss ratio.
/// The rates are shared; `RLR` and `RRF` vary by origin with correlated
/// effects on the paper's standardised `o` scale (model 1, as its brms
/// code):
///
/// ```text
/// RLR_i = m_RLR e^(s_RLR (b_RLR + u_i1)),   RRF_i = m_RRF e^(s_RRF (b_RRF + u_i2))
/// k_er = m_ker e^(s_ker b_ker),             k_p = m_kp e^(s_kp b_kp)
/// u_i = diag(τ) L z_i,  z_i ~ N(0, I),  L L' = [[1, ρ], [ρ, 1]],  ρ ~ LKJ(η)
/// b ~ N(0, 1),  τ_1, τ_2 ~ half priors,  log σ_OS, log σ_PD ~ real priors
/// y_OS ~ N(OS_i(t), σ_OS²),  y_PD ~ N(PD_i(t), σ_PD²)   (cumulative paid)
/// ```
///
/// with each `m e^(s o)` a [`PositivePrior::LogNormal`] of
/// [`CompartmentalPriors`]. The priors are a required argument;
/// [`CompartmentalPriors::gesmann_morris`] is the paper's case study.
///
/// NUTS samples the intercepts `b`, `log τ`, `atanh ρ`, the `z_i` of each
/// origin with an observed cell and the two `log σ`, on a dense scale from a
/// pilot run as [`Meyers`](super::super::Meyers) does: a scale whitened at a start
/// point ([`CompartmentalStart`]), a pilot run of `tune` warm-up and
/// `max(tune, 100)` kept draws per chain, then the pilot draws' mean and
/// covariance. The gradient comes from the density written once over
/// dual numbers (decision 4) for priors without effects by cell (model 1).
/// With effects by cell (model 2, or any origin effect on a rate or
/// development effect), it is per-cell dual numbers over the cell's four
/// parameters and `log σ`, then a hand-written chain rule to the sampled
/// parameters, checked against the single dual pass over the generic
/// density by a unit test. A non-finite value counts as a divergence.
///
/// An origin with a positive premium and no observed cell (the papers'
/// training triangle has none for its youngest accident year) has no `z`:
/// each predictive simulation draws its effect from the population with the
/// draw's `τ` and `ρ` (decision 2); the fit lists such origins in
/// [`CompartmentalFit::population_origins`]. The reserve is the paid to the
/// square's last age less the latest observed paid (decision 5): the square
/// runs for as many origin periods as the premium column has origins, in
/// development periods, and an origin observed at that age has a reserve of
/// 0. The origin grain must be a whole multiple of the development grain
/// (decision 13: annual origins with quarterly development give 40 ages for
/// ten origins); each cell's development time is its age in years on every
/// grain, so the rates are per year, and an incremental paid cell is the
/// increment over one development period.
///
/// The likelihood, the response scale and the paid response are settings
/// (decision 1), each combination allowed: [`likelihood`](Self::likelihood)
/// normal or log-normal, [`scale`](Self::scale) amounts or loss ratios to
/// premium, [`paid_response`](Self::paid_response) cumulative or
/// incremental. The hierarchy is in the priors: origin effects on `k_er`
/// and `k_p` and development-year effects on any of the four parameters
/// (`ker_effect_sd`, `kp_effect_sd` and `development_effect_sd` of
/// [`CompartmentalPriors`]), so each cell has its own parameters, and the
/// mean of an incremental paid cell is the closed form at its age less the
/// same at the age before, both with that cell's parameters. Gesmann and
/// Morris' model 2 is
/// `Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())`
/// ([`model_2`](Self::model_2) sets the log-normal likelihood on loss
/// ratios with incremental paid and a longer, finer sampler).
///
/// [`fitted`](Self::fitted) chooses the responses: paid and outstanding
/// (the default), paid and incurred, or one of them alone, which needs
/// [`RateOrder::Ordered`] (decision 12).
///
/// # Fitting sub-annual data
///
/// Quarterly or monthly development cells are allowed (decision 13). What a
/// study on simulated books found (decision 14; no real sub-annual triangle
/// is in the repository, so every number below is from simulation) is in
/// `knowledge/findings/compartmental-subannual.md`.
///
/// * **Rate priors.** The rates are per year on every grain, so the paper's
///   presets are kept as they are. Finer cells pull the preset towards the
///   truth; a line whose first year runs much faster or slower than the
///   preset's `k_er` of about 3 needs priors of its own.
/// * **Development levels** ([`development_levels`](Self::development_levels),
///   model 2). [`DevelopmentLevels::PerAge`] gives each age its own level of
///   the development effects, [`DevelopmentLevels::ByYear`] each development
///   year. Choose with `loo` on the data: where effects move by year, by-year
///   levels agree with per-age ones on the reserve (0.10 sd apart on average
///   in the study), win elpd (30 of 30 quarterly books) and on 10 monthly
///   books under a short sampler took 89 s against 126 s (rough times) with a
///   worst R-hat of 1.07 against 1.62, though a median bulk ESS of 101 against
///   114; where effects move within a year they lose 361 of elpd in 30 of 30
///   books and widen the reserve's sd by 59%. At annual grain the two are the
///   same fit.
/// * **Seasonality.** There is no seasonal term. Model 1 reads a season as
///   noise (the interval widens, by up to 54% in the sd for a 30% season, and
///   the mean can sit nearly 1 sd off the outcome), and model 2's per-age
///   levels do not absorb a season unless every origin starts in the same
///   calendar quarter. A calendar-quarter factor is not built; it comes first
///   among the builds if your lines are seasonal.
/// * **Cumulative paid.** Under [`PaidResponse::Cumulative`] the interval
///   narrows as the cells get finer: right where the noise of the cells is
///   independent, too narrow where it persists. In the study's world with
///   persistent payment noise, at one persistence chosen by hand and fitted
///   to no book (`PERSIST_PHI` 25, `PERSIST_RHO` 0.95; a smaller fall is
///   expected with less, untested), the 90% interval covered 88%, 70% and 56%
///   at annual, quarterly and monthly cells and the RMSE rose from 1,623 to
///   2,033. A fit
///   with cumulative paid on cells finer than a year therefore carries a
///   warning that asks you to check the residuals; it also fires on a fit
///   that is calibrated, since the data do not say which world they are in.
///   [`PaidResponse::Incremental`] keeps the interval but, in that world, ran
///   1.3 to 2 times too wide, with a worse RMSE at quarterly and monthly
///   grain and a failing Meyers' D at annual grain. Neither is chosen for you.
/// * **Earned exposure.** All of an origin's exposure starts at age 0, so
///   sub-annual development of annual origins fits an exposure the model
///   does not have; use quarterly origins where the data allow. On simulated
///   books with earned premium annual origins failed Meyers' D at annual,
///   quarterly and monthly development, and quarterly origins of the same
///   books passed narrowly (17.8 against a critical 19.2, on the short
///   sampler, with 3 books at the default sampler); that comparison also
///   changes the number of origins and the effects' variance. A setting that
///   spreads the exposure over the origin's period is not built.
///
/// ```
/// use prospicio_bayes::nuts::Sampler;
/// use prospicio_reserving::bayes::{Compartmental, CompartmentalPriors, Outstanding};
/// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
/// use prospicio_prob::{Distribution, KeyValue};
///
/// // Five origins of cumulative paid and incurred with premium.
/// let paid: [&[f64]; 5] = [
///     &[9.6, 22.8, 33.3, 40.3, 45.1],
///     &[7.9, 19.5, 29.6, 36.8],
///     &[8.7, 24.3, 35.4],
///     &[13.3, 33.0],
///     &[11.4],
/// ];
/// let incurred: [&[f64]; 5] = [
///     &[62.7, 64.0, 65.6, 65.3, 62.3],
///     &[62.1, 56.7, 56.6, 56.8],
///     &[64.0, 61.5, 63.2],
///     &[73.9, 75.1],
///     &[77.1],
/// ];
/// let premium = [104.4, 88.9, 86.0, 99.3, 104.9];
/// let (mut origin, mut age, mut pd, mut inc, mut prem) = (vec![], vec![], vec![], vec![], vec![]);
/// for w in 0..5 {
///     for d in 0..paid[w].len() {
///         origin.push(Month::january(1988 + w as i32));
///         age.push(12 * (d as u32 + 1));
///         pd.push(paid[w][d]);
///         inc.push(incurred[w][d]);
///         prem.push(premium[w]);
///     }
/// }
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&age),
///     values: &[("paid", &pd), ("incurred", &inc), ("premium", &prem)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// // The paper's case study priors; a short run for the example.
/// let preset = Compartmental::new(CompartmentalPriors::gesmann_morris());
/// let model = Compartmental {
///     sampler: Sampler { chains: 2, tune: 150, draws: 150, target_accept: 0.9, ..preset.sampler },
///     n_sims: 1_000,
///     ..preset
/// };
/// // Check the priors first: paths of the mean paid and outstanding they
/// // imply, here the paid at age 5 of the oldest origin over its premium.
/// let paths = model.prior_predictive(&tri, "premium", 1_000)?;
/// let key = vec![tri.origins()[0].into(), KeyValue::Int(60)];
/// let paid_5 = paths.paid.marginal(&key).unwrap();
/// let (lo, hi) = (paid_5.quantile(0.05)?, paid_5.quantile(0.95)?);
/// assert!(lo / 104.4 > 0.2 && hi / 104.4 < 1.2);
/// let fit = model.fit(&tri, "paid", Outstanding::FromIncurred("incurred"), "premium")?;
/// assert_eq!(fit.reserves.n_components(), 5);
/// assert_eq!(fit.posterior.names()[0], "k_er");
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Compartmental {
    /// The priors (decision 2; a required argument of [`new`](Self::new)).
    pub priors: CompartmentalPriors,
    /// The likelihood.
    pub likelihood: CompartmentalLikelihood,
    /// The response scale.
    pub scale: Scale,
    /// Cumulative or incremental paid.
    pub paid_response: PaidResponse,
    /// The responses fitted ([`Fitted`]).
    pub fitted: Fitted,
    /// How the rates are kept apart.
    pub rate_order: RateOrder,
    /// How wide a development level is (model 2's development effects; an
    /// [`Error::NotSupported`] under priors with none,
    /// [`check_supported`](Self::check_supported)). A modelling choice to
    /// make with `loo`: see [`DevelopmentLevels`].
    pub development_levels: DevelopmentLevels,
    /// Where the pilot run starts.
    pub start: CompartmentalStart,
    /// NUTS settings. Its `seed` is not used: the chains take a seed derived
    /// from [`seed`](Self::seed) (module docs, "Seeds").
    pub sampler: Sampler,
    /// Number of predictive simulations. The one-year view runs at most
    /// [`ONE_YEAR_SIMS`](Self::ONE_YEAR_SIMS) of them unless its settings'
    /// [`n_sims`](BayesOneYear::n_sims) says otherwise.
    pub n_sims: usize,
    /// Seed of the chains (and the pilot run) and of the simulations, each
    /// through its own derived seed (module docs, "Seeds").
    pub seed: u64,
    /// How segments depend on each other in a fit of several segments
    /// ([`fit_segments`](Self::fit_segments), decision 8; `Synchronized` is
    /// an error there); [`fit`](Self::fit) has one segment and does not use
    /// it.
    pub dependence: SegmentDependence,
}

impl Compartmental {
    /// The most one-year simulations [`one_year`](Self::one_year) runs by
    /// default (`BayesOneYear::n_sims` `None`): 2,000, against the 10,000
    /// of the lifetime view, since nearly every simulation of these models
    /// is flagged and costs a moment match, a NUTS refit or (model 2) the
    /// reweighting of 95 parameters (the user's decision of 2026-10-16,
    /// `docs/design/reserving-bayes.md` decision 20). A model whose
    /// [`n_sims`](Self::n_sims) is smaller runs that many.
    pub const ONE_YEAR_SIMS: usize = 2_000;

    /// The model with `priors` and the defaults: model 1 (normal on amounts,
    /// cumulative paid, paid and outstanding), free rate order, the
    /// data-based start, four chains of 1,000 warm-up and 2,000 kept draws
    /// with target acceptance 0.99 (the paper's `adapt_delta`) and tree depth
    /// at most 10, 10,000 simulations, seed 0, independent segments.
    pub fn new(priors: CompartmentalPriors) -> Self {
        Self {
            priors,
            likelihood: CompartmentalLikelihood::Normal,
            scale: Scale::Amount,
            paid_response: PaidResponse::Cumulative,
            fitted: Fitted::PaidAndOutstanding,
            rate_order: RateOrder::Free,
            development_levels: DevelopmentLevels::PerAge,
            start: CompartmentalStart::Data,
            sampler: Sampler {
                target_accept: 0.99,
                draws: 2000,
                ..Sampler::default()
            },
            n_sims: 10_000,
            seed: 0,
            dependence: SegmentDependence::Independent,
        }
    }

    /// Gesmann and Morris' model 2 settings with `priors` (the paper's are
    /// [`CompartmentalPriors::gesmann_morris_2`]): the log-normal
    /// likelihood on loss ratios with incremental paid, and a longer, finer
    /// sampler than [`new`](Self::new)'s, four chains of 2,000 warm-up and
    /// 8,000 kept draws with target acceptance 0.999 (decision 5: with 87
    /// parameters, effect sds whose posteriors pile up near 0 and a slowly
    /// mixing `k_p`, model 1's sampler left divergent transitions or an
    /// R-hat above 1.01 on most seeds); the rest as [`new`](Self::new).
    pub fn model_2(priors: CompartmentalPriors) -> Self {
        let model_1 = Self::new(priors);
        Self {
            likelihood: CompartmentalLikelihood::LogNormal,
            scale: Scale::LossRatio,
            paid_response: PaidResponse::Incremental,
            sampler: Sampler {
                tune: 2000,
                draws: 8000,
                target_accept: 0.999,
                ..model_1.sampler
            },
            ..model_1
        }
    }

    /// Fits paid and outstanding (or the responses [`fitted`](Self::fitted)
    /// names) of a single-segment cumulative triangle
    /// with premium from its `premium` column (the latest value per origin).
    /// The fit's origins are the premium column's; an origin with no
    /// fitted cell is drawn from the population. Under
    /// [`Fitted::PaidOnly`] `outstanding` is not read (any value will do).
    /// Failed diagnostics
    /// do not fail the fit: they are warnings in
    /// [`CompartmentalFit::diagnostics`], as is any posterior draw with
    /// `k_er <= k_p` under [`RateOrder::Free`] and cumulative paid on
    /// development cells finer than a year (a prompt to check the residuals,
    /// see "Fitting sub-annual data" on [`Compartmental`]).
    pub fn fit(
        &self,
        triangle: &Triangle,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
    ) -> Result<CompartmentalFit> {
        self.fit_streams(
            triangle,
            paid,
            outstanding,
            premium,
            Stream::Chains,
            Stream::Predictive,
        )
    }

    /// [`fit`](Self::fit) of a single-segment triangle, its chains (and
    /// pilot run) and its simulations taking the seeds that `chains` and
    /// `predictive` derive (module docs, "Seeds").
    pub(super) fn fit_streams(
        &self,
        triangle: &Triangle,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
        chains: Stream,
        predictive: Stream,
    ) -> Result<CompartmentalFit> {
        self.validate()?;
        let data = Data::new(triangle, paid, outstanding, premium, self.fitted)?;
        let model = Model::new(
            &data,
            self.priors,
            self.rate_order == RateOrder::Ordered,
            self.response(),
        )?;
        if model.cells.len() <= model.dim() {
            return Err(Error::Bayes(format!(
                "{} observed cells for {} parameters; the model needs more cells than parameters",
                model.cells.len(),
                model.dim()
            )));
        }
        let (posterior, sampled, mut diagnostics) =
            self.sample(&model, derive_seed(self.seed, chains))?;
        let rate_swaps = {
            let (j_er, j_p) = (0, 1);
            posterior
                .values()
                .chunks_exact(posterior.dim())
                .filter(|d| d[j_er] <= d[j_p])
                .count()
        };
        if rate_swaps > 0 {
            diagnostics.warnings.push(format!(
                "{rate_swaps} of {} posterior draws have k_er <= k_p, the mirror image of the \
                 rate swap; check the rate priors, or set rate_order to Ordered",
                posterior.n_draws()
            ));
        }
        if let Some(warning) = cumulative_sub_annual_warning(model.response, data.grain) {
            diagnostics.warnings.push(warning);
        }
        if self.fitted == Fitted::IncurredOnly {
            if let Some(share) = payment_rate_prior_share(&posterior, self.priors) {
                if share > PRIOR_SHARE_WARNING {
                    diagnostics.warnings.push(format!(
                        "the posterior sd of log k_p is {:.0}% of its prior's: the incurred \
                         does not identify the payment rate, so the paid run-off and the \
                         reserve come from the prior on k_p (no paid cell is fitted); use \
                         a prior you believe, or fit paid as well",
                        100.0 * share
                    ));
                }
            }
        }
        let cell_rate_swaps = model.cell_rate_swaps(&posterior);

        let mut hasher = InputHasher::new();
        hasher.str(paid);
        if self.fitted != Fitted::PaidOnly {
            hasher.str(outstanding_name(outstanding));
        }
        hasher.str(premium);
        hasher.f64s(&data.premium);
        for c in &model.cells {
            hasher.f64s(&[c.origin as f64, c.t, c.y, f64::from(u8::from(c.paid))]);
        }
        if !model.response.paid {
            // The paid column gives each origin's latest paid only.
            for l in &data.latest_paid {
                hasher.f64s(&l.map_or([-1.0, 0.0], |(k, v)| [k as f64, v]));
            }
        }
        let provenance = self.provenance(paid, outstanding, premium, hasher);

        let future = data.future_cells();
        let n_paid = future.paid.len();
        let n_future = n_paid + future.outstanding.len();
        // The pattern columns after the cells, written from the draw alone:
        // the cells are what they were without them (decision 28 item 3).
        let (joint, patterns) = predictive_keyed_with_shares(
            &posterior,
            "cell",
            (0..n_future)
                .map(|j| vec![KeyValue::Int(j as i64)])
                .collect(),
            data.grain.origin,
            (0..data.n_ages).map(|k| data.age(k)).collect(),
            self.n_sims,
            derive_seed(self.seed, predictive),
            provenance.clone(),
            |nat, rng, cells, shares| {
                model.simulate(nat, &future, rng, cells);
                model.pattern(nat, shares);
            },
        )?;
        let provenance = joint.provenance().clone();
        let n_origins = data.origins.len();
        let mut paid_draws = Vec::with_capacity(self.n_sims * n_paid);
        let mut os_draws = Vec::with_capacity(self.n_sims * (n_future - n_paid));
        let mut reserve_draws = Vec::with_capacity(self.n_sims * n_origins);
        for row in joint.draw_matrix().chunks_exact(n_future) {
            paid_draws.extend_from_slice(&row[..n_paid]);
            os_draws.extend_from_slice(&row[n_paid..]);
            for (i, last) in future.last_paid.iter().enumerate() {
                reserve_draws.push(match last {
                    Some(j) => row[*j] - data.latest_paid[i].map_or(0.0, |(_, v)| v),
                    None => 0.0,
                });
            }
        }
        let cell_keys = |cells: &[(usize, usize)]| -> Vec<Vec<KeyValue>> {
            cells
                .iter()
                .map(|&(i, k)| {
                    vec![
                        data.origins[i].into(),
                        KeyValue::Int(i64::from(data.age(k))),
                    ]
                })
                .collect()
        };
        let with_output = |name: &str| {
            let mut p = provenance.clone();
            p.parameters.retain(|(k, _)| k != "output");
            p.param("output", name)
        };
        let reserves = PredictiveDistribution::from_draws(
            vec!["origin".into()],
            data.origins.iter().map(|&o| vec![o.into()]).collect(),
            reserve_draws,
            with_output("reserves"),
        )?;
        let paid_pd = PredictiveDistribution::from_draws(
            vec!["origin".into(), "age".into()],
            cell_keys(&future.paid),
            paid_draws,
            with_output("paid"),
        )?;
        let os_pd = if model.response.second {
            Some(PredictiveDistribution::from_draws(
                vec!["origin".into(), "age".into()],
                cell_keys(&future.outstanding),
                os_draws,
                with_output("outstanding"),
            )?)
        } else {
            None
        };
        let ultimate_loss_ratio = model.mean_ultimate_loss_ratios(&posterior);
        Ok(CompartmentalFit {
            origins: data.origins.clone(),
            premium: data.premium.clone(),
            latest_paid: data
                .latest_paid
                .iter()
                .map(|l| l.map_or(0.0, |(_, v)| v))
                .collect(),
            population_origins: (0..n_origins)
                .filter(|&i| model.level[i].is_none())
                .map(|i| data.origins[i])
                .collect(),
            posterior,
            diagnostics,
            rate_swaps,
            cell_rate_swaps,
            reserves,
            paid: paid_pd,
            outstanding: os_pd,
            ultimate_loss_ratio,
            patterns,
            model,
            sampled,
            data,
        })
    }

    /// Fits paid and outstanding in every segment of a cumulative triangle,
    /// each segment with its own premium column and posterior (the same
    /// model and settings, sampled on their own), into joint distributions
    /// of the reserves and the future paid and outstanding cells, the
    /// segments depending on each other as [`dependence`](Self::dependence)
    /// says (decision 8), as in
    /// [`Meyers::fit_segments`](super::super::Meyers::fit_segments):
    ///
    /// * [`Independent`](SegmentDependence::Independent): each segment's
    ///   simulations as drawn. Segment `k` (in index order) takes its own
    ///   derived seeds for its chains (and its pilot run) and its
    ///   simulations (module docs, "Seeds"), so the segments share no
    ///   random numbers; a segment's draws therefore differ from fitting it
    ///   alone with [`fit`](Self::fit), which a one-segment triangle's
    ///   segment 0 streams reproduce.
    /// * [`RankCorrelation`](SegmentDependence::RankCorrelation): the
    ///   segments' simulations reordered as whole rows by Iman–Conover on the
    ///   segment totals of the reserves; each segment's reserves, future
    ///   paid cells and future outstanding cells move together, so every
    ///   simulation's reserve is still its paid at the last age less the
    ///   latest paid.
    /// * [`Synchronized`](SegmentDependence::Synchronized) is an
    ///   [`Error::NotSupported`]: it draws the residuals of the same positions in
    ///   every segment, and a posterior has no residuals to share.
    ///
    /// A bad rank-correlation matrix fails before any sampling, and a
    /// failure names its segment. Each segment is read as
    /// [`fit`](Self::fit) reads a single-segment triangle.
    pub fn fit_segments(
        &self,
        triangle: &Triangle,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
    ) -> Result<CompartmentalFits> {
        self.validate()?;
        if self.dependence == SegmentDependence::Synchronized {
            return Err(super::super::synchronized_error());
        }
        self.dependence.validate(triangle.index().len())?;
        let mut k = 0;
        let mut fitted = fit_each(triangle, paid, |_| {
            let label = &triangle.index()[k];
            let values: Vec<[&str; 1]> = label.parts().iter().map(|p| [p.as_str()]).collect();
            let conditions: Vec<(&str, &[&str])> = triangle
                .key_names()
                .iter()
                .zip(&values)
                .map(|(name, value)| (name.as_str(), value.as_slice()))
                .collect();
            let fit = triangle.select(&conditions).and_then(|one| {
                self.fit_streams(
                    &one,
                    paid,
                    outstanding,
                    premium,
                    Stream::SegmentChains(k),
                    Stream::SegmentPredictive(k),
                )
            });
            k += 1;
            fit.map(|fit| CompartmentalSegment { fit })
        })?;

        // One row per simulation: every segment's reserves, then its future
        // paid cells, then its future outstanding cells. Iman–Conover ranks
        // a segment by its reserves alone and moves the whole of its row,
        // and its patterns with it.
        let (mut ranges, mut ranked, mut width) = (Vec::new(), Vec::new(), 0);
        for s in &fitted.fits {
            let f = &s.fit;
            let m = f.reserves.n_components();
            let len: usize = f.parts().iter().map(|p| p.n_components()).sum();
            ranges.push(width..width + len);
            ranked.push(width..width + m);
            width += len;
        }
        let mut rows = Vec::with_capacity(self.n_sims * width);
        for i in 0..self.n_sims {
            for s in &fitted.fits {
                for part in s.fit.parts() {
                    let n = part.n_components();
                    rows.extend_from_slice(&part.draw_matrix()[i * n..(i + 1) * n]);
                }
            }
        }
        let rows = self.dependence.reorder_rows_with(
            rows,
            width,
            &ranges,
            &ranked,
            self.seed,
            |s, index| fitted.fits[s].fit.patterns.permute(0, index),
        )?;
        for (s, range) in fitted.fits.iter_mut().zip(&ranges) {
            let f = &mut s.fit;
            let mut start = range.start;
            let mut rebuilt = Vec::with_capacity(3);
            for part in f.parts() {
                let n = part.n_components();
                let draws = rows
                    .chunks_exact(width)
                    .flat_map(|row| row[start..start + n].iter().copied())
                    .collect();
                rebuilt.push(PredictiveDistribution::from_draws(
                    part.dims().to_vec(),
                    part.components().to_vec(),
                    draws,
                    part.provenance().clone(),
                )?);
                start += n;
            }
            let mut rebuilt = rebuilt.into_iter();
            f.reserves = rebuilt.next().expect("the reserves");
            f.paid = rebuilt.next().expect("the paid cells");
            // The future outstanding (or incurred) cells, when fitted.
            f.outstanding = rebuilt.next();
        }

        let mut hasher = InputHasher::new();
        hasher.str(paid);
        if self.fitted != Fitted::PaidOnly {
            hasher.str(outstanding_name(outstanding));
        }
        hasher.str(premium);
        for (label, s) in fitted.iter() {
            hasher.str(&label.to_string());
            hasher.f64s(&s.fit.premium);
            for c in s.fit.data.paid.iter().chain(&s.fit.data.outstanding) {
                hasher.f64s(&[c.0 as f64, c.1 as f64, c.2]);
            }
        }
        let premiums: Vec<f64> = fitted
            .fits
            .iter()
            .map(|s| s.fit.premium.iter().sum())
            .collect();
        let provenance = self
            .provenance(paid, outstanding, premium, hasher)
            .param("premium_total", format!("{premiums:?}"))
            .param("segments", fitted.len())
            .param("dependence", format!("{:?}", self.dependence))
            .param(
                "posterior_draws",
                fitted.fits.first().map_or(0, |s| s.fit.posterior.n_draws()),
            );
        CompartmentalFits::assemble(fitted, provenance)
    }

    /// The one-year view (`docs/design/reserving-bayes.md`, decisions 16
    /// to 18 and 20): fits as [`fit`](Self::fit), then draws
    /// [`n_sims`](Self::n_sims) years of the paid and outstanding cells
    /// valued in the twelve months after the valuation (each from a
    /// posterior draw, with the likelihood's noise), reweights the
    /// posterior draws by each year's likelihood with Pareto smoothed
    /// importance sampling, and takes the claims development result of
    /// each origin as the opening reserve less the year's payment and the
    /// reserve after the year, each reserve the (weighted) posterior mean
    /// of the expected paid to the square's last age less the paid so
    /// far. A simulation whose Pareto `k̂` is above the settings' threshold
    /// is moment-matched, and refitted with NUTS (this model's sampler, on
    /// the triangle extended by its year) only when matching cannot bring
    /// its `k̂` to the threshold, or as [`BayesOneYear::refit`] says.
    ///
    /// The defaults (the user's decision of 2026-10-16, decision 20): at
    /// most [`ONE_YEAR_SIMS`](Self::ONE_YEAR_SIMS) simulations, and the
    /// refit mode of [`one_year_refit`](Self::one_year_refit) when the
    /// settings' `refit` is `None`. With development effects (model 2) that
    /// is a refit of the flagged simulations (moment matching rescues none
    /// of them on company 337) by the light refit, [`LightRefit::default`],
    /// a warm-started sampler of 4 chains, 150 warm-up and 250 kept draws
    /// (the user's decision of 2026-10-21, decision 24; within about 0.03 sd
    /// of the total claims development result of a refit per simulation, at
    /// about 20 times the run time of reweighting only). Set
    /// [`light_refit`](BayesOneYear::light_refit) for another sampler,
    /// [`no_light_refit`](BayesOneYear::no_light_refit) to reweight only
    /// (approximate, about 0.4 sd off, and
    /// [`approximate`](BayesOneYearFit::approximate) says so), or `refit` to
    /// choose the mode (with [`max_refits`](BayesOneYear::max_refits) as a
    /// budget).
    ///
    /// An origin with no observed cell (the papers' 1997) and, with
    /// development effects (model 2), an age with none take their effects
    /// from the population in [`fit`](Self::fit)'s simulations; here they
    /// are part of each posterior draw (standard normal `z` added to the
    /// fit's draws, which is their posterior), so the year's cells update
    /// them. A refit takes about as long as the fit: on the papers'
    /// triangle about 5.5 s for model 1 and about a minute for model 2 in a
    /// release build. The view's seed is the simulations' derived seed of
    /// [`seed`](Self::seed); the fit is the same as `fit`'s.
    ///
    /// ```
    /// use prospicio_bayes::nuts::Sampler;
    /// use prospicio_reserving::bayes::{
    ///     BayesOneYear, Compartmental, CompartmentalPriors, OneYearRefit, Outstanding,
    /// };
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, Triangle};
    /// use prospicio_prob::Distribution;
    ///
    /// let paid: [&[f64]; 5] = [
    ///     &[9.6, 22.8, 33.3, 40.3, 45.1],
    ///     &[7.9, 19.5, 29.6, 36.8],
    ///     &[8.7, 24.3, 35.4],
    ///     &[13.3, 33.0],
    ///     &[11.4],
    /// ];
    /// let incurred: [&[f64]; 5] = [
    ///     &[62.7, 64.0, 65.6, 65.3, 62.3],
    ///     &[62.1, 56.7, 56.6, 56.8],
    ///     &[64.0, 61.5, 63.2],
    ///     &[73.9, 75.1],
    ///     &[77.1],
    /// ];
    /// let premium = [104.4, 88.9, 86.0, 99.3, 104.9];
    /// let (mut origin, mut age, mut pd, mut inc, mut prem) = (vec![], vec![], vec![], vec![], vec![]);
    /// for w in 0..5 {
    ///     for d in 0..paid[w].len() {
    ///         origin.push(Month::january(1988 + w as i32));
    ///         age.push(12 * (d as u32 + 1));
    ///         pd.push(paid[w][d]);
    ///         inc.push(incurred[w][d]);
    ///         prem.push(premium[w]);
    ///     }
    /// }
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&age),
    ///     values: &[("paid", &pd), ("incurred", &inc), ("premium", &prem)],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })?;
    /// let preset = Compartmental::new(CompartmentalPriors::gesmann_morris());
    /// let model = Compartmental {
    ///     sampler: Sampler { chains: 2, tune: 150, draws: 150, target_accept: 0.9, ..preset.sampler },
    ///     n_sims: 500,
    ///     ..preset
    /// };
    /// // Reweighting only, for speed: model 1's default moment-matches the
    /// // flagged simulations and refits those it cannot rescue.
    /// assert_eq!(model.one_year_refit(), OneYearRefit::MomentMatchThenRefit);
    /// let only = BayesOneYear { refit: Some(OneYearRefit::ReweightOnly), ..BayesOneYear::default() };
    /// let view = model.one_year(&tri, "paid", Outstanding::FromIncurred("incurred"), "premium", &only)?;
    /// assert_eq!(view.cdr.n_components(), 5);
    /// // 1988 is at the square's last age: no cell in the year, no change.
    /// assert!(view.cdr.draw_matrix().chunks(5).all(|row| row[0] == 0.0));
    /// // A refit mode the settings choose is used as it is: not approximate.
    /// assert_eq!(view.approximate, None);
    /// # Ok::<(), prospicio_reserving::Error>(())
    /// ```
    pub fn one_year(
        &self,
        triangle: &Triangle,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
        settings: &BayesOneYear,
    ) -> Result<BayesOneYearFit<CompartmentalFit>> {
        settings.validate()?;
        let (settings, n_sims, approximate) = self.one_year_defaults(settings);
        let settings = &settings;
        let (fit, draws) = self.year_streams(
            triangle,
            paid,
            outstanding,
            premium,
            settings,
            n_sims,
            Stream::Chains,
            Stream::Predictive,
        )?;
        let mut provenance = fit.reserves.provenance().clone();
        provenance.parameters.retain(|(k, _)| k != "output");
        provenance.model = format!("{}_one_year", provenance.model);
        let provenance = one_year::provenance_with_light_refit(provenance, settings, n_sims);
        let cdr = PredictiveDistribution::from_draws(
            vec!["origin".into()],
            fit.origins.iter().map(|&o| vec![o.into()]).collect(),
            draws.cdr,
            provenance,
        )?;
        Ok(BayesOneYearFit {
            opening_ultimate: fit
                .latest_paid
                .iter()
                .zip(&draws.opening)
                .map(|(l, r)| l + r)
                .collect(),
            fit,
            opening_reserve: draws.opening,
            cdr,
            pareto_k: draws.pareto_k,
            refitted: draws.refitted,
            moment_matched: draws.moment_matched,
            unrefitted: draws.unrefitted,
            refit_warnings: draws.refit_warnings,
            k_threshold: settings.k_threshold,
            approximate,
            warnings: Vec::new(),
        })
    }

    /// [`fit_streams`](Self::fit_streams) of a single-segment triangle and
    /// its one-year simulations (`settings` with the model's defaults
    /// applied, `n_sims` of them), the chains and the simulations taking the
    /// seeds that `chains` and `predictive` derive. The view's seed is the
    /// simulations' derived seed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn year_streams(
        &self,
        triangle: &Triangle,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
        settings: &BayesOneYear,
        n_sims: usize,
        chains: Stream,
        predictive: Stream,
    ) -> Result<(CompartmentalFit, one_year::YearDraws)> {
        let fit = self.fit_streams(triangle, paid, outstanding, premium, chains, predictive)?;
        let view_seed = one_year_seed(derive_seed(self.seed, predictive));
        let mut year = super::year::CompartmentalYear::new(self, &fit.data)?;
        // The added `z` of each draw: stream `d` of a seed that the
        // simulations' streams (`0..n_sims` of `view_seed`) do not reach.
        let opening = year.opening_draws(
            &fit.model,
            &fit.sampled,
            StreamRng::new(view_seed, u64::MAX).next_u64(),
        )?;
        year.warm_start(settings.light_refit, &opening)?;
        let draws = one_year::simulate(&year, &opening, settings, n_sims, view_seed)?;
        Ok((fit, draws))
    }

    /// The one-year view of every segment of a triangle, each segment as
    /// [`one_year`](Self::one_year) views a single-segment triangle (its own
    /// premium column and outstanding cells, the refit mode default and the
    /// [`approximate`](CompartmentalOneYearFits::approximate) warning of
    /// decision 20), on the seeds of [`fit_segments`](Self::fit_segments):
    /// segment `k` (in index order) takes `Stream::SegmentChains(k)` for its
    /// chains and `Stream::SegmentPredictive(k)` for its simulations, so a
    /// one-segment triangle's view is `one_year` on segment 0's streams. The
    /// segments' claims development results are joined by row, as
    /// [`dependence`](Self::dependence) says: `Independent` as drawn,
    /// `RankCorrelation` by Iman-Conover on the segments' total claims
    /// development results (as `BayesOdp` and `Meyers` do);
    /// `Synchronized` is an [`Error::NotSupported`] and a bad matrix an
    /// [`Error::InvalidSetting`], both before any sampling. The refit budget
    /// [`max_refits`](BayesOneYear::max_refits) applies to each segment. A
    /// failure names its segment.
    pub fn one_year_segments(
        &self,
        triangle: &Triangle,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
        settings: &BayesOneYear,
    ) -> Result<CompartmentalOneYearFits> {
        self.validate()?;
        settings.validate()?;
        if self.dependence == SegmentDependence::Synchronized {
            return Err(super::super::synchronized_error());
        }
        self.dependence.validate(triangle.index().len())?;
        let (settings, n_sims, approximate) = self.one_year_defaults(settings);
        let settings = &settings;
        let mut k = 0;
        let fitted = fit_each(triangle, paid, |_| {
            let label = &triangle.index()[k];
            let values: Vec<[&str; 1]> = label.parts().iter().map(|p| [p.as_str()]).collect();
            let conditions: Vec<(&str, &[&str])> = triangle
                .key_names()
                .iter()
                .zip(&values)
                .map(|(name, value)| (name.as_str(), value.as_slice()))
                .collect();
            let view = triangle.select(&conditions).and_then(|one| {
                self.year_streams(
                    &one,
                    paid,
                    outstanding,
                    premium,
                    settings,
                    n_sims,
                    Stream::SegmentChains(k),
                    Stream::SegmentPredictive(k),
                )
            });
            k += 1;
            view
        })?;

        let mut hasher = InputHasher::new();
        hasher.str(paid);
        hasher.str(outstanding_name(outstanding));
        hasher.str(premium);
        let mut dims = fitted.key_names.clone();
        dims.push("origin".into());
        let mut components = Vec::new();
        let mut ranges = Vec::new();
        for (label, (fit, _)) in fitted.iter() {
            hasher.str(&label.to_string());
            hasher.f64s(&fit.premium);
            for c in fit.data.paid.iter().chain(&fit.data.outstanding) {
                hasher.f64s(&[c.0 as f64, c.1 as f64, c.2]);
            }
            let start = components.len();
            components.extend(origin_keys(label, &fit.origins));
            ranges.push(start..components.len());
        }
        let mut draws = Vec::with_capacity(n_sims * components.len());
        for i in 0..n_sims {
            for (_, year) in &fitted.fits {
                let m = year.opening.len();
                draws.extend_from_slice(&year.cdr[i * m..(i + 1) * m]);
            }
        }
        let premiums: Vec<f64> = fitted
            .fits
            .iter()
            .map(|(fit, _)| fit.premium.iter().sum())
            .collect();
        let mut provenance = self
            .provenance(paid, outstanding, premium, hasher)
            .param("premium_total", format!("{premiums:?}"))
            .param("segments", fitted.len())
            .param("dependence", format!("{:?}", self.dependence));
        provenance.model = format!("{}_one_year", provenance.model);
        let provenance = one_year::provenance_with_light_refit(provenance, settings, n_sims);
        let cdr = PredictiveDistribution::from_draws(dims, components, draws, provenance)?;
        let cdr = self.dependence.reorder_ranges(cdr, &ranges, self.seed)?;
        let segments = fitted.map(|(fit, year)| {
            let opening_ultimate = fit
                .latest_paid
                .iter()
                .zip(&year.opening)
                .map(|(l, r)| l + r)
                .collect();
            BayesOneYearSegment {
                fit: CompartmentalSegment { fit: fit.clone() },
                opening_ultimate,
                opening_reserve: year.opening.clone(),
                pareto_k: year.pareto_k.clone(),
                refitted: year.refitted.clone(),
                moment_matched: year.moment_matched.clone(),
                unrefitted: year.unrefitted.clone(),
                refit_warnings: year.refit_warnings,
                k_threshold: settings.k_threshold,
            }
        });
        Ok(CompartmentalOneYearFits {
            segments,
            cdr,
            approximate,
        })
    }

    /// The one-year view's refit mode when its settings' `refit` is `None`
    /// (the user's decision of 2026-10-16, `docs/design/reserving-bayes.md`
    /// decision 20, changed by the user's decision of 2026-10-21, decision
    /// 24): [`OneYearRefit::RefitFlagged`] with development effects in the
    /// priors (model 2), whose flagged simulations moment matching does not
    /// rescue, refitted by the light refit of [`LightRefit`] unless the
    /// settings turn it off ([`BayesOneYear::no_light_refit`], then the view
    /// reweights only and is approximate);
    /// [`OneYearRefit::MomentMatchThenRefit`] without (model 1).
    pub fn one_year_refit(&self) -> OneYearRefit {
        if self
            .priors
            .development_effect_sd
            .iter()
            .any(Option::is_some)
        {
            OneYearRefit::RefitFlagged
        } else {
            OneYearRefit::MomentMatchThenRefit
        }
    }

    /// `settings` with this model's one-year defaults in place of a `None`
    /// refit mode ([`one_year_refit`](Self::one_year_refit)), the number of
    /// one-year simulations (the settings' `n_sims`, or this model's capped
    /// at [`ONE_YEAR_SIMS`](Self::ONE_YEAR_SIMS)), and the view's
    /// [`approximate`](BayesOneYearFit::approximate) warning: set when the
    /// default leaves the flagged simulations reweighted only.
    pub(super) fn one_year_defaults(
        &self,
        settings: &BayesOneYear,
    ) -> (BayesOneYear, usize, Option<String>) {
        let model_2 = self.one_year_refit() == OneYearRefit::RefitFlagged;
        // Model 2's default (a `None` refit mode) refits its flagged
        // simulations with the light refit (the user's decision of
        // 2026-10-21, decision 24), unless the settings turn it off, when
        // the view reweights only and is approximate. A refit mode the
        // settings choose is used as it is, with the fit's own sampler
        // unless a light refit is given; a light refit given with no mode
        // is a choice to refit.
        let default_light = (model_2 && !settings.no_light_refit).then(LightRefit::default);
        let light = settings.light_refit.or(if settings.refit.is_none() {
            default_light
        } else {
            None
        });
        let default_refit = match (model_2, light) {
            (true, Some(_)) => OneYearRefit::RefitFlagged,
            (true, None) => OneYearRefit::ReweightOnly,
            (false, _) => OneYearRefit::MomentMatchThenRefit,
        };
        let approximate = (settings.refit.is_none() && default_refit == OneYearRefit::ReweightOnly)
            .then(|| APPROXIMATE.to_string());
        let n_sims = settings.sims(self.n_sims.min(Self::ONE_YEAR_SIMS));
        let mut resolved = settings.with_refit_default(default_refit);
        // A view that reweights only refits nothing: no sampler to record.
        resolved.light_refit = if resolved.mode() == OneYearRefit::ReweightOnly {
            None
        } else {
            light
        };
        (resolved, n_sims, approximate)
    }

    /// Paths of the mean outstanding and cumulative paid of each origin of
    /// the premium column at each age of the square, drawn from the priors
    /// alone (Gesmann and Morris, sec. 3.4): a check of a prior set before
    /// fitting. Each simulation draws the intercepts, the effect sds, the
    /// correlation and each origin's effects, and with development effects
    /// in the priors the effects of each level of
    /// [`development_levels`](Self::development_levels), as the fit has them
    /// (shared by the origins; each age its own under
    /// [`DevelopmentLevels::PerAge`], the ages of a development year one
    /// under [`DevelopmentLevels::ByYear`]); with
    /// effects by cell and incremental paid, the cumulative paid path is
    /// the sum of the cells' mean increments. The paths are means, without
    /// the likelihood's noise: the preset's prior on `log σ` is effectively
    /// flat, so its draws of `σ` would swamp them. Under
    /// [`RateOrder::Ordered`] the rate intercepts are drawn from their priors
    /// restricted to `k_er > k_p`, exactly; if those priors put less than
    /// `1e-6` of their mass on `k_er > k_p` the check is an
    /// [`Error::Bayes`]. The seed is the simulations' derived
    /// seed of [`seed`](Self::seed).
    pub fn prior_predictive(
        &self,
        triangle: &Triangle,
        premium: &str,
        n_sims: usize,
    ) -> Result<CompartmentalPaths> {
        self.validate_priors()?;
        if n_sims == 0 {
            return Err(Error::InvalidSetting {
                name: "n_sims",
                value: 0.0,
                expected: "a positive number of simulations",
            });
        }
        if self.rate_order == RateOrder::Ordered {
            let (a, _) = rate_order_bound(self.priors);
            let mass = norm_cdf(-a);
            if mass < MIN_ORDERED_MASS {
                return Err(Error::Bayes(format!(
                    "the rate priors give almost no mass to k_er > k_p ({mass:.1e}, \
                     below {MIN_ORDERED_MASS:.0e}), so the prior check under \
                     RateOrder::Ordered cannot draw from them; centre the rate \
                     priors with k_er above k_p, or use RateOrder::Free"
                )));
            }
        }
        let (origins, premiums, grain) = premium_of(triangle, premium)?;
        let n = origins.len();
        // The square: `n` origin periods of development, in development
        // periods (decision 13).
        let na = grain.square_ages(n);
        let ages: Vec<Lag> = (0..na).map(|k| grain.age(k)).collect();
        let times: Vec<f64> = (0..na).map(|k| grain.time(k)).collect();
        // Each age's development level, as in the fit. Groups rise with the
        // age and skip none, since the square's ages are consecutive.
        let levels: Vec<usize> = ages
            .iter()
            .enumerate()
            .map(|(k, &a)| self.development_levels.group(a, k))
            .collect();
        let n_levels = levels.last().map_or(0, |l| l + 1);
        let pr = self.priors;
        let ordered = self.rate_order == RateOrder::Ordered;
        let cellwise = pr.rates_vary() || pr.development_effect_sd.iter().any(Option::is_some);
        let incremental = self.paid_response == PaidResponse::Incremental;
        let mut keys = Vec::with_capacity(n * na);
        for &o in &origins {
            for &a in &ages {
                keys.push(vec![o.into(), KeyValue::Int(i64::from(a))]);
            }
        }
        let provenance = Provenance::new("compartmental_prior_predictive")
            .param("n_sims", n_sims)
            .param("seed", self.seed)
            .param("rate_order", format!("{:?}", self.rate_order))
            .param("priors", format!("{pr:?}"))
            .param("paid_response", format!("{:?}", self.paid_response))
            .param(
                "development_levels",
                format!("{:?}", self.development_levels),
            )
            .param("premium", premium)
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"));
        let both = PredictiveDistribution::simulate(
            vec!["cell".into()],
            (0..2 * n * na)
                .map(|j| vec![KeyValue::Int(j as i64)])
                .collect(),
            n_sims,
            derive_seed(self.seed, Stream::Predictive),
            provenance.clone(),
            |rng, row| {
                let normal = |rng: &mut StreamRng| norm_quantile(rng.next_open01());
                let (m_er, s_er) = lognormal(pr.k_er);
                let (m_p, s_p) = lognormal(pr.k_p);
                let (m_rlr, s_rlr) = lognormal(pr.rlr);
                let (m_rrf, s_rrf) = lognormal(pr.rrf);
                let (b_ker, b_kp) = if ordered {
                    ordered_rates(pr, rng)
                } else {
                    (normal(rng), normal(rng))
                };
                let k_er = m_er * (s_er * b_ker).exp();
                let k_p = m_p * (s_p * b_kp).exp();
                let b_rlr = normal(rng);
                let b_rrf = normal(rng);
                let tau1 = draw_half(pr.rlr_effect_sd, rng);
                let tau2 = draw_half(pr.rrf_effect_sd, rng);
                let gamma = Gamma::new(pr.lkj_eta, 1.0).expect("lkj_eta was validated");
                let (g1, g2) = (gamma.sample(rng, 1)[0], gamma.sample(rng, 1)[0]);
                let rho = 2.0 * g1 / (g1 + g2) - 1.0;
                let rho_c = (1.0 - rho * rho).max(0.0).sqrt();
                if !cellwise {
                    // Model 1's hierarchy: shared rates, one curve per age.
                    let curves: Vec<(f64, f64)> =
                        times.iter().map(|&t| one_os(k_er, k_p, t)).collect();
                    for i in 0..n {
                        let (z1, z2) = (normal(rng), normal(rng));
                        let u1 = tau1 * z1;
                        let u2 = tau2 * (rho * z1 + rho_c * z2);
                        let rlr = m_rlr * (s_rlr * (b_rlr + u1)).exp() * premiums[i];
                        let rrf = m_rrf * (s_rrf * (b_rrf + u2)).exp();
                        for (k, (os, pd)) in curves.iter().enumerate() {
                            row[i * na + k] = rlr * os;
                            row[n * na + i * na + k] = rlr * rrf * pd;
                        }
                    }
                    return;
                }
                // Effects by cell: the origin effects on the rates and the
                // development effects, each drawn from its population.
                let tau_rate =
                    [pr.ker_effect_sd, pr.kp_effect_sd].map(|p| p.map(|h| draw_half(h, rng)));
                let tau_dev = pr
                    .development_effect_sd
                    .map(|p| p.map(|h| draw_half(h, rng)));
                let dev: Vec<[f64; 4]> = (0..n_levels)
                    .map(|_| tau_dev.map(|t| t.map_or(0.0, |t| t * normal(rng))))
                    .collect();
                let medians = [(m_er, s_er), (m_p, s_p), (m_rlr, s_rlr), (m_rrf, s_rrf)];
                for i in 0..n {
                    let (z1, z2) = (normal(rng), normal(rng));
                    let u = [
                        tau_rate[0].map_or(0.0, |t| t * normal(rng)),
                        tau_rate[1].map_or(0.0, |t| t * normal(rng)),
                        tau1 * z1,
                        tau2 * (rho * z1 + rho_c * z2),
                    ];
                    let b = [b_ker, b_kp, b_rlr, b_rrf];
                    let mut cum = 0.0;
                    for (k, &t) in times.iter().enumerate() {
                        let [ker, kp, rlr, rrf] = [0, 1, 2, 3].map(|p| {
                            medians[p].0 * (medians[p].1 * (b[p] + u[p] + dev[levels[k]][p])).exp()
                        });
                        let (os, pd) = one_os(ker, kp, t);
                        let base = rlr * premiums[i];
                        row[i * na + k] = base * os;
                        cum = if incremental {
                            let before = if k == 0 {
                                0.0
                            } else {
                                one_os(ker, kp, times[k - 1]).1
                            };
                            cum + base * rrf * (pd - before)
                        } else {
                            base * rrf * pd
                        };
                        row[n * na + i * na + k] = cum;
                    }
                }
            },
        )?;
        let split = |half: usize, name: &str| -> Result<PredictiveDistribution> {
            let draws: Vec<f64> = both
                .draw_matrix()
                .chunks_exact(2 * n * na)
                .flat_map(|r| r[half * n * na..(half + 1) * n * na].to_vec())
                .collect();
            Ok(PredictiveDistribution::from_draws(
                vec!["origin".into(), "age".into()],
                keys.clone(),
                draws,
                provenance.clone().param("output", name),
            )?)
        };
        Ok(CompartmentalPaths {
            outstanding: split(0, "outstanding")?,
            paid: split(1, "paid")?,
        })
    }

    /// NUTS on `model` with the chains' seed `chains`: a pilot run on a
    /// scale whitened at the start, then the main run on the pilot draws'
    /// scale, as [`Meyers`](super::super::Meyers) samples. The draws on the
    /// natural scale, the same draws on the unconstrained scale of `theta`
    /// (an affine map of the whitened scale NUTS samples on, for the
    /// one-year view) and their diagnostics.
    pub(super) fn sample(
        &self,
        model: &Model,
        chains: u64,
    ) -> Result<(PosteriorDraws, PosteriorDraws, Diagnostics)> {
        let p = model.dim();
        let start = model.start(self.start);
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
        let pilot = pilot.transform(model.sampled_names(), |z| rough.theta(z))?;
        let whitened = Whitened::from_draws(model, &pilot)?;
        let draws = sample(
            &whitened,
            &vec![0.0; p],
            Sampler {
                seed: chains,
                ..self.sampler
            },
        )?;
        let sampled = draws.transform(model.sampled_names(), |z| whitened.theta(z))?;
        let posterior = sampled.transform(model.names(), |theta| model.natural(theta))?;
        let summary = posterior.summary()?;
        let diagnostics = Diagnostics::new(
            &summary,
            posterior.chains(),
            posterior.divergences().iter().sum(),
        );
        Ok((posterior, sampled, diagnostics))
    }

    fn validate_priors(&self) -> Result<()> {
        self.priors.validate()
    }

    pub(super) fn response(&self) -> Response {
        Response {
            likelihood: self.likelihood,
            loss_ratio: self.scale == Scale::LossRatio,
            incremental: self.paid_response == PaidResponse::Incremental,
            incurred: matches!(self.fitted, Fitted::PaidAndIncurred | Fitted::IncurredOnly),
            paid: self.fitted != Fitted::IncurredOnly,
            second: self.fitted != Fitted::PaidOnly,
            levels: self.development_levels,
        }
    }

    /// Refuses a combination of settings the model does not support,
    /// without reading any data (`reserving-v02.md` decision 29): one
    /// response alone ([`Fitted::PaidOnly`] or [`Fitted::IncurredOnly`])
    /// under [`RateOrder::Free`] is an [`Error::NotSupported`]. Every fit,
    /// the one-year view and the backtest's candidate check it before
    /// anything else, with the range checks of the settings (a prior on
    /// `k_p` wider than [`MAX_INCURRED_ONLY_KP_SD`] under
    /// [`Fitted::IncurredOnly`], a bad prior, `n_sims` of 0, each an
    /// [`Error::InvalidSetting`]). The fields are public, so a struct
    /// literal can hold any setting; call this when the settings are chosen
    /// to refuse an unsupported one at once, as the Python and R bindings
    /// do when they build a model. [`prior_predictive`](Self::prior_predictive)
    /// does not need it, so the prior check runs on any settings.
    pub fn check_supported(&self) -> Result<()> {
        let one = match self.fitted {
            Fitted::PaidAndOutstanding | Fitted::PaidAndIncurred => None,
            Fitted::PaidOnly => Some("fitting paid alone (Fitted::PaidOnly)"),
            Fitted::IncurredOnly => Some("fitting incurred alone (Fitted::IncurredOnly)"),
        };
        if self.development_levels == DevelopmentLevels::ByYear
            && self
                .priors
                .development_effect_sd
                .iter()
                .all(Option::is_none)
        {
            return Err(Error::NotSupported {
                setting: "development_levels ByYear",
                method: "priors with no development effects (model 1)",
                reason: "levels by development year group the development effects, and these \
                         priors have none: model 1's rates are shared by every origin and \
                         age, so there is no level to widen. Use a model with development \
                         effects (model 2, CompartmentalPriors::gesmann_morris_2), or keep \
                         development_levels PerAge",
            });
        }
        if let Some(what) = one
            && self.rate_order == RateOrder::Free
        {
            return Err(Error::NotSupported {
                setting: "rate_order Free",
                method: what,
                reason: "from one response the means are unchanged by the rate swap (k_er, k_p, \
                         RLR, RRF) -> (k_p, k_er, RLR k_er / k_p, RRF k_p / k_er), and from paid \
                         alone only RLR x RRF is identified, so under free rates the posterior has \
                         two modes the data cannot tell apart (design note decisions 2 and 12); it \
                         needs rate_order Ordered",
            });
        }
        Ok(())
    }

    /// [`check_supported`](Self::check_supported) and the range checks of
    /// the settings, before any data are read.
    pub(super) fn validate(&self) -> Result<()> {
        self.check_supported()?;
        self.validate_priors()?;
        if self.fitted == Fitted::IncurredOnly {
            let (_, sd) = lognormal(self.priors.k_p);
            if sd > MAX_INCURRED_ONLY_KP_SD {
                return Err(Error::InvalidSetting {
                    name: "the log-scale sd of the prior on k_p, fitting incurred alone (Fitted::IncurredOnly)",
                    value: sd,
                    expected: "at most 0.5: the incurred does not identify the payment rate, so \
                               the paid run-off and the reserve follow the prior on k_p, and a \
                               wide prior leaves them unidentified (on company 337 the reserve \
                               runs from -51,000 at a log-scale sd of 1 to 203,000 at 0.3). Give a \
                               prior you believe, for example from the book's paid history or a \
                               paid-and-incurred fit of a similar book, or fit paid as well \
                               (decision 12)",
                });
            }
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

    /// The model's name: model 1 and model 2 when the settings and the
    /// hierarchy are the paper's, else `compartmental`.
    pub(super) fn model_name(&self) -> &'static str {
        let response = self.response();
        let hierarchy = self.priors.hierarchy();
        // The width of a level is a setting of the model, not a different
        // model: model 2 with levels by year is still model 2.
        let response = Response {
            levels: DevelopmentLevels::PerAge,
            ..response
        };
        if response == Response::MODEL_1 && hierarchy == Hierarchy::Model1 {
            "compartmental_gm1"
        } else if response == Response::MODEL_2 && hierarchy == Hierarchy::Model2 {
            "compartmental_gm2"
        } else {
            "compartmental"
        }
    }

    pub(super) fn provenance(
        &self,
        paid: &str,
        outstanding: Outstanding<'_>,
        premium: &str,
        hasher: InputHasher,
    ) -> Provenance {
        let s = &self.sampler;
        let pr = &self.priors;
        Provenance::new(self.model_name())
            .param("priors", format!("{pr:?}"))
            .param("likelihood", format!("{:?}", self.likelihood))
            .param("scale", format!("{:?}", self.scale))
            .param("paid_response", format!("{:?}", self.paid_response))
            .param("fitted", format!("{:?}", self.fitted))
            .param("rate_order", format!("{:?}", self.rate_order))
            .param(
                "development_levels",
                format!("{:?}", self.development_levels),
            )
            .param("start", format!("{:?}", self.start))
            .param("chains", s.chains)
            .param("tune", s.tune)
            .param("draws", s.draws)
            .param("target_accept", s.target_accept)
            .param("max_depth", s.max_depth)
            .param("n_sims", self.n_sims)
            .param("seed", self.seed)
            .param("paid", paid)
            .param(
                "outstanding",
                if self.fitted == Fitted::PaidOnly {
                    "not read".to_string()
                } else {
                    format!("{outstanding:?}")
                },
            )
            .param("premium", premium)
            .version("prospicio-reserving", env!("CARGO_PKG_VERSION"))
            .input_hash(hasher.finish())
    }
}

/// How the warning of cumulative paid on cells finer than a year starts. It
/// sits among a fit's [`Diagnostics::warnings`] but is no failed diagnostic,
/// so the bindings use this to word it apart from the sampler's: a segment's
/// label can come before it.
pub const CUMULATIVE_SUB_ANNUAL_WARNING: &str = "cumulative paid is fitted on ";

/// The warning of a fit with cumulative paid on cells finer than a year
/// (decision 14, open decision 12 (b)), or `None`. The data do not say
/// whether payment noise persists from cell to cell, so it is a prompt to
/// look at the residuals and not a finding: it fires on a calibrated fit too.
/// It starts with [`CUMULATIVE_SUB_ANNUAL_WARNING`].
pub(super) fn cumulative_sub_annual_warning(
    response: Response,
    grain: SquareGrain,
) -> Option<String> {
    (response.paid && !response.incremental && grain.development.months() < 12).then(|| {
        format!(
            "{CUMULATIVE_SUB_ANNUAL_WARNING}{} development cells, whose noise the likelihood counts as independent between cells; where payment noise persists from cell to cell the interval narrows as the cells get finer (on simulated books, at one persistence chosen by hand, PERSIST_PHI 25 and PERSIST_RHO 0.95, the 90% interval covered 88%, 70% and 56% at annual, quarterly and monthly cells and the RMSE rose from 1,623 to 2,033; a smaller fall is expected with less persistence, untested), and where it does not the fit is as calibrated as at any grain. Check the residuals; incremental paid keeps the interval but ran 1.3 to 2 times too wide there (decision 14)",
            grain.development
        )
    })
}

pub(super) fn outstanding_name<'a>(outstanding: Outstanding<'a>) -> &'a str {
    match outstanding {
        Outstanding::Column(c) | Outstanding::FromIncurred(c) => c,
    }
}

/// Fewest kept draws per chain of the pilot run.
const MIN_PILOT_DRAWS: usize = 100;
