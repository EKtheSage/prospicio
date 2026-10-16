//! The one-year view of a Bayesian reserving model by Pareto smoothed
//! importance sampling (`docs/design/reserving-bayes.md`, decisions 16 and
//! 17).
//!
//! Each simulation draws the coming year's cells from the posterior
//! predictive (a posterior draw, then process error), reweights every
//! posterior draw by the likelihood of those cells, Pareto-smooths the
//! weights (`prospicio_bayes::elpd::psis`, Vehtari et al. 2024) and takes
//! the time-1 reserve as the weighted posterior mean of the reserve that
//! remains after the year. A simulation whose weights are unreliable
//! (Pareto `k̂` above a threshold) is refitted with NUTS on the extended
//! triangle instead, or first moment-matched (decision 18,
//! [`moment_match`](super::moment_match)), as [`BayesOneYear::refit`]
//! says.
//!
//! The engine knows nothing of a particular model (decision 17): a model
//! plugs in through [`YearModel`], the one interface every Bayesian
//! model's one-year view implements, and [`simulate`] and [`assemble`] are
//! written once over it. A model whose cells are over-dispersed Poisson
//! increments with a log-linear mean ([`BayesOdp`](super::BayesOdp),
//! [`BayesBf`](super::BayesBf)) gets its [`YearModel`] from [`OdpYear`] by
//! implementing the smaller [`OdpCells`]. The conformance suite in this
//! module's tests (and its release-build twin,
//! `validation/tests/reserving_bayes_one_year_conformance.rs`) runs on every
//! implementor.

use std::sync::Mutex;

use prospicio_bayes::nuts::PosteriorDraws;
use prospicio_core::StreamRng;
use prospicio_prob::{PredictiveDistribution, Provenance};
use rayon::prelude::*;

use super::moment_match;
use crate::bootstrap::{
    ProcessDistribution, component_sums, pick_segment, process_draw, push_moments, segment_sums,
};
use crate::chain_ladder::ChainLadderFit;
use crate::error::{Error, Result};
use crate::one_year_bootstrap::{Failures, YearCells, opening_names};
use crate::segments::{FitTable, ReserveFit, SegmentFits};
use crate::triangle::{Segment, Triangle};

/// What the one-year view does with a simulation whose Pareto `k̂` is
/// above [`BayesOneYear::k_threshold`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OneYearRefit {
    /// Reweight every simulation, flagged or not: the fastest, and biased
    /// where `k̂` is high.
    ReweightOnly,
    /// Refit the flagged simulations with NUTS on the triangle extended by
    /// their year, reweight the others (the default).
    #[default]
    RefitFlagged,
    /// Refit every simulation: exact up to the refits' Monte Carlo error,
    /// one NUTS run per simulation; for checking the reweighting.
    RefitAll,
    /// Moment-match the flagged simulations (Paananen et al. 2021, as R's
    /// `loo::loo_moment_match`: the posterior draws moved by affine maps
    /// to the weighted moments until `k̂` is at most the threshold), and
    /// refit with NUTS only those it cannot bring there; reweight the
    /// others. Opt-in (decision 18).
    MomentMatchThenRefit,
}

/// Settings of the one-year view of a Bayesian model
/// (`BayesOdp::one_year`, `BayesBf::one_year`): the Pareto `k̂` above
/// which a simulation's importance weights are not trusted, and what is
/// done then.
///
/// ```
/// use prospicio_reserving::bayes::{BayesOneYear, OneYearRefit};
///
/// let settings = BayesOneYear::default();
/// assert_eq!(settings.k_threshold, 0.7);
/// assert_eq!(settings.refit, OneYearRefit::RefitFlagged);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BayesOneYear {
    /// A simulation whose Pareto `k̂` is above this is flagged (0.7, the
    /// threshold of Vehtari et al. 2024 for a few thousand draws).
    pub k_threshold: f64,
    /// What happens to a flagged simulation.
    pub refit: OneYearRefit,
}

impl Default for BayesOneYear {
    fn default() -> Self {
        Self {
            k_threshold: 0.7,
            refit: OneYearRefit::RefitFlagged,
        }
    }
}

impl BayesOneYear {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.k_threshold.is_nan() {
            return Err(Error::InvalidSetting {
                name: "k_threshold",
                value: self.k_threshold,
                expected: "a number, the Pareto k above which a simulation is flagged",
            });
        }
        Ok(())
    }

    /// Whether a simulation with Pareto `k` is not simply reweighted:
    /// refitted, or (`MomentMatchThenRefit`) moment-matched first.
    fn refits(&self, k: f64) -> bool {
        match self.refit {
            OneYearRefit::ReweightOnly => false,
            // A NaN k is flagged, and an infinite one (too few draws to fit the
            // tail) is above any finite threshold.
            OneYearRefit::RefitFlagged | OneYearRefit::MomentMatchThenRefit => {
                flagged(k, self.k_threshold)
            }
            OneYearRefit::RefitAll => true,
        }
    }
}

/// A Bayesian model's coming year, as the one-year view needs it: the one
/// interface every Bayesian model's one-year view implements (decision
/// 17). The year is a fixed list of values per simulation (for the ODP
/// likelihood the increments of the year's cells, in a fixed order), and
/// `theta` is always one posterior draw on the scale
/// [`PosteriorDraws`] stores it.
///
/// That scale must be unconstrained: every point of `R^p` a valid
/// parameter, as the sampling scale is. Moment matching (decision 18)
/// moves the draws by affine maps on it, which on a constrained scale (a
/// standard deviation, `tanh` of a correlation) can leave the support;
/// a Jacobian term does not mend that. `BayesOdp` and `BayesBf` store
/// such draws. `Meyers` and `Compartmental` store natural-scale values
/// (`tanh`, square roots of variances, exponentials) and must store or
/// expose their unconstrained sampling scale before they implement this
/// trait.
///
/// Everything a simulation's predictive step would draw afresh that the
/// year's cells inform (an effect of an origin or an age with no observed
/// cell, say) must be part of `theta`, carried with each posterior draw,
/// so that reweighting the draws updates it; only process error that the
/// year does not inform may be drawn in [`draw_year`](Self::draw_year).
/// A model whose remaining cells depend on the year's cells given `theta`
/// (Meyers' cumulative values) says so through
/// [`closing_uses_year`](Self::closing_uses_year).
pub(crate) trait YearModel: Sync {
    /// Number of origins, the components of the claims development result.
    fn n_origins(&self) -> usize;
    /// Number of values in a year.
    fn n_year(&self) -> usize;
    /// Draws the year given the parameters `theta`, with process error.
    fn draw_year(&self, theta: &[f64], rng: &mut StreamRng, year: &mut [f64]);
    /// The log likelihood of `year` given `theta`, up to a constant that
    /// does not depend on `theta`: the log importance ratio. It must be
    /// the change in [`log_posterior`](Self::log_posterior) that the year
    /// makes, so that the reweighted draws target the posterior a
    /// [`refit`](Self::refit) samples.
    fn year_log_likelihood(&self, theta: &[f64], year: &[f64]) -> f64;
    /// Each origin's payment in `year`, written into `paid`.
    fn paid(&self, year: &[f64], paid: &mut [f64]);
    /// Each origin's expected reserve given `theta` at the opening.
    fn opening(&self, theta: &[f64], reserve: &mut [f64]);
    /// Each origin's expected reserve after the year, given `theta` and
    /// `year`.
    fn closing(&self, theta: &[f64], year: &[f64], reserve: &mut [f64]);
    /// Whether [`closing`](Self::closing) depends on `year`; when it does
    /// not (the ODP likelihood), it is computed once per posterior draw.
    fn closing_uses_year(&self) -> bool {
        true
    }
    /// The posterior given the observed triangle extended by `year`,
    /// sampled with NUTS from `seed`, and whether its diagnostics warn.
    fn refit(&self, year: &[f64], seed: u64) -> Result<(PosteriorDraws, bool)>;
    /// The log posterior density of `theta` up to a constant, on the scale
    /// the draws are stored on, given the observed triangle extended by
    /// `year` (an empty `year` is the observed triangle alone). Moment
    /// matching (decision 18) evaluates it at moved draws with an empty
    /// year, where it should be cheap, and adds
    /// [`year_log_likelihood`](Self::year_log_likelihood), which the
    /// conformance suite checks is the year's change to it. It is the
    /// density of the draws as stored: on a scale that is an affine map
    /// of the sampling scale (every model so far) no Jacobian term is
    /// needed, on any other unconstrained scale the model adds its own.
    /// Conformance check 5 (the score identity) fails a log posterior
    /// that is not the stored draws' density, which check 1 cannot see.
    fn log_posterior(&self, theta: &[f64], year: &[f64]) -> f64;
}

/// What the engine returns for one segment.
#[derive(Debug, Clone)]
pub(crate) struct YearDraws {
    /// Each origin's posterior mean reserve at the opening.
    pub(crate) opening: Vec<f64>,
    /// The claims development result, `n_sims` rows of one value per origin.
    pub(crate) cdr: Vec<f64>,
    /// Each simulation's Pareto `k̂`.
    pub(crate) pareto_k: Vec<f64>,
    /// The simulations refitted with NUTS, in order.
    pub(crate) refitted: Vec<usize>,
    /// The simulations moment matching brought to `k̂` at most the
    /// threshold, in order.
    pub(crate) moment_matched: Vec<usize>,
    /// How many refits had a diagnostic warning.
    pub(crate) refit_warnings: usize,
}

/// One simulation's outcome.
struct Sim {
    cdr: Vec<f64>,
    k: f64,
    refitted: bool,
    matched: bool,
    warned: bool,
}

/// The one-year view of `model` under `posterior`: `n_sims` simulations,
/// simulation `i` on stream `i` of `seed`.
pub(crate) fn simulate<M: YearModel>(
    model: &M,
    posterior: &PosteriorDraws,
    settings: &BayesOneYear,
    n_sims: usize,
    seed: u64,
) -> Result<YearDraws> {
    let s = posterior.n_draws();
    let (no, ny) = (model.n_origins(), model.n_year());
    let per_draw = |f: &(dyn Fn(&[f64], &mut [f64]) + Sync)| -> Vec<f64> {
        let mut out = vec![0.0; s * no];
        out.par_chunks_mut(no)
            .enumerate()
            .for_each(|(d, row)| f(posterior.draw(d), row));
        out
    };
    let mean_over = |by_draw: &[f64], weight: &dyn Fn(usize) -> f64, out: &mut [f64]| {
        out.fill(0.0);
        for (d, row) in by_draw.chunks_exact(no).enumerate() {
            let w = weight(d);
            for (o, v) in out.iter_mut().zip(row) {
                *o += w * v;
            }
        }
    };
    let opening_by_draw = per_draw(&|theta, row| model.opening(theta, row));
    let mut opening = vec![0.0; no];
    mean_over(&opening_by_draw, &|_| 1.0 / s as f64, &mut opening);
    let closing_by_draw = (!model.closing_uses_year())
        .then(|| per_draw(&|theta, row| model.closing(theta, &[], row)));
    // The opening draws' log posterior, once, for moment matching.
    let matching = (settings.refit == OneYearRefit::MomentMatchThenRefit)
        .then(|| moment_match::Opening::new(model, posterior));

    let failures = Mutex::new(Failures::default());
    let fail = |e: Error| {
        failures
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record(e);
    };
    let sims: Vec<Option<Sim>> = (0..n_sims)
        .into_par_iter()
        .map(|i| {
            let mut rng = StreamRng::new(seed, i as u64);
            let j = ((rng.next_open01() * s as f64) as usize).min(s - 1);
            let mut year = vec![0.0; ny];
            model.draw_year(posterior.draw(j), &mut rng, &mut year);
            // Drawn whether or not the simulation is refitted, so every
            // setting of `refit` sees the same years.
            let refit_seed = rng.next_u64();
            let log_r: Vec<f64> = (0..s)
                .map(|d| model.year_log_likelihood(posterior.draw(d), &year))
                .collect();
            // A year that tells nothing about the parameters (no cells, as
            // in a fully developed segment, or equal ratios) leaves the
            // posterior as it is: equal weights, k̂ = 0 and no refit, where
            // `psis` would give an infinite k̂ for want of a tail.
            let uninformative = ny == 0 || log_r.iter().all(|&v| v == log_r[0]);
            let (log_w, k) = if uninformative {
                (vec![0.0; s], 0.0)
            } else {
                // Relative efficiency 1: the simulations' draws are taken
                // as independent, as before decision 17.
                match prospicio_bayes::elpd::psis(&log_r, 1.0) {
                    Ok(smoothed) => smoothed,
                    Err(e) => {
                        fail(Error::Bayes(format!(
                            "the year's log likelihood is not a number in some posterior \
                             draw ({e})"
                        )));
                        return None;
                    }
                }
            };
            let mut closing = vec![0.0; no];
            let flagged = !uninformative && settings.refits(k);
            // Under `MomentMatchThenRefit` a flagged simulation is
            // moment-matched first; one it cannot bring to the threshold
            // (or with a NaN k̂) is refitted on the seed the other settings
            // use.
            let rescued = match &matching {
                Some(opening) if flagged && !k.is_nan() => {
                    moment_match::moment_match(
                        model,
                        opening,
                        closing_by_draw.as_deref(),
                        &year,
                        &log_w,
                        k,
                        settings.k_threshold,
                    )
                    .1
                }
                _ => None,
            };
            let matched = rescued.is_some();
            let (refitted, mut warned) = (flagged && !matched, false);
            if let Some(c) = rescued {
                closing = c;
            } else if refitted {
                match model.refit(&year, refit_seed) {
                    Ok((draws, warn)) => {
                        warned = warn;
                        let n = draws.n_draws();
                        let mut row = vec![0.0; no];
                        for d in 0..n {
                            model.closing(draws.draw(d), &year, &mut row);
                            for (c, v) in closing.iter_mut().zip(&row) {
                                *c += v / n as f64;
                            }
                        }
                    }
                    Err(e) => {
                        fail(e);
                        return None;
                    }
                }
            } else {
                let max = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = log_w.iter().map(|v| (v - max).exp()).collect();
                let total: f64 = w.iter().sum();
                match &closing_by_draw {
                    Some(by_draw) => mean_over(by_draw, &|d| w[d] / total, &mut closing),
                    None => {
                        let mut row = vec![0.0; no];
                        for (d, wd) in w.iter().enumerate() {
                            model.closing(posterior.draw(d), &year, &mut row);
                            for (c, v) in closing.iter_mut().zip(&row) {
                                *c += wd / total * v;
                            }
                        }
                    }
                }
            }
            let mut paid = vec![0.0; no];
            model.paid(&year, &mut paid);
            let cdr = (0..no).map(|o| opening[o] - paid[o] - closing[o]).collect();
            Some(Sim {
                cdr,
                k,
                refitted,
                matched,
                warned,
            })
        })
        .collect();
    let failures = failures
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((message, _)) = failures.first {
        return Err(Error::Bayes(format!(
            "the one-year view failed in {} of {n_sims} simulations, for example: {message}",
            failures.count
        )));
    }
    let mut out = YearDraws {
        opening,
        cdr: Vec::with_capacity(n_sims * no),
        pareto_k: Vec::with_capacity(n_sims),
        refitted: Vec::new(),
        moment_matched: Vec::new(),
        refit_warnings: 0,
    };
    for (i, sim) in sims.into_iter().enumerate() {
        let sim = sim.expect("a failed simulation is reported above");
        out.cdr.extend(sim.cdr);
        out.pareto_k.push(sim.k);
        if sim.refitted {
            out.refitted.push(i);
        }
        if sim.matched {
            out.moment_matched.push(i);
        }
        out.refit_warnings += usize::from(sim.warned);
    }
    Ok(out)
}

/// The one-year view of one segment from the engine's `year`, with the
/// lifetime `fit` it reweights: the claims development result by origin
/// (dimension `origin`, `provenance` on it), the opening reserve and
/// ultimate, and the reliability of each simulation. Written once for
/// every Bayesian model (decision 17).
pub(crate) fn assemble<F: ReserveFit>(
    fit: F,
    year: YearDraws,
    segment: &Segment,
    settings: &BayesOneYear,
    provenance: Provenance,
) -> Result<BayesOneYearFit<F>> {
    let cdr = PredictiveDistribution::from_draws(
        vec!["origin".into()],
        segment.origins.iter().map(|&o| vec![o.into()]).collect(),
        year.cdr,
        provenance,
    )?;
    Ok(BayesOneYearFit {
        opening_ultimate: opening_ultimate(fit.chain_ladder(), &year.opening),
        fit,
        opening_reserve: year.opening,
        cdr,
        pareto_k: year.pareto_k,
        refitted: year.refitted,
        moment_matched: year.moment_matched,
        refit_warnings: year.refit_warnings,
        k_threshold: settings.k_threshold,
    })
}

/// The latest value plus `reserve`, per origin.
pub(crate) fn opening_ultimate(chain_ladder: &ChainLadderFit, reserve: &[f64]) -> Vec<f64> {
    chain_ladder
        .latest
        .iter()
        .zip(reserve)
        .map(|(l, r)| l + r)
        .collect()
}

/// A model of over-dispersed Poisson increments with a log-linear mean
/// `exp(eta(theta, w, d))` and a plugged-in scale `phi`, which
/// [`OdpYear`] turns into a [`YearModel`]: [`BayesOdp`](super::BayesOdp)
/// and [`BayesBf`](super::BayesBf). Every origin and every age of such a
/// model has an observed cell, so `theta` holds every effect a future cell
/// needs and nothing in the predictive step but the process error is drawn
/// afresh (decision 17).
pub(crate) trait OdpCells: Sync {
    /// The log mean of cell `(w, d)` given a posterior draw `theta`.
    fn eta(&self, theta: &[f64], w: usize, d: usize) -> f64;
    /// The plugged-in scale.
    fn phi(&self) -> f64;
    /// The log posterior of `theta` (a posterior draw) with the
    /// observed increments `extra` (origin, development, value) added, as
    /// [`YearModel::log_posterior`].
    fn log_posterior(&self, theta: &[f64], extra: &[(usize, usize, f64)]) -> f64;
    /// The posterior with the increments `extra` added, as
    /// [`YearModel::refit`].
    fn refit(&self, extra: &[(usize, usize, f64)], seed: u64) -> Result<(PosteriorDraws, bool)>;
}

/// The coming year of a fitted model of [`OdpCells`], for the one-year
/// view (decision 16): the year's increments, one per cell valued in the
/// twelve months after the valuation, in origin and then development order.
pub(crate) struct OdpYear<'a, C> {
    cells: &'a C,
    n_origins: usize,
    process: ProcessDistribution,
    /// The year's cells, `(origin, position)`.
    year: Vec<(usize, usize)>,
    /// The cells after the latest diagonal, the opening reserve.
    future: Vec<(usize, usize)>,
    /// The cells after the year, the closing reserve.
    after: Vec<(usize, usize)>,
}

impl<'a, C: OdpCells> OdpYear<'a, C> {
    /// The year of `segment` of `triangle`, whose origins' latest observed
    /// positions are `latest`, with `process` error on each drawn cell.
    pub(crate) fn new(
        cells: &'a C,
        process: ProcessDistribution,
        triangle: &Triangle,
        segment: &Segment,
        latest: &[usize],
    ) -> Self {
        let nd = segment.n_dev;
        let mut last = latest.to_vec();
        // No tail (decision 12): the year stops at the last age.
        for y in YearCells::of(
            segment,
            latest,
            |o, d| triangle.valuation_of(o + segment.origin_offset, d + segment.dev_offset),
            false,
        ) {
            last[y.origin] = y.last;
        }
        let between = |from: &dyn Fn(usize) -> usize, to: &dyn Fn(usize) -> usize| {
            (0..segment.n_origins)
                .flat_map(|w| (from(w) + 1..=to(w)).map(move |d| (w, d)))
                .collect::<Vec<_>>()
        };
        Self {
            cells,
            n_origins: segment.n_origins,
            process,
            year: between(&|w| latest[w], &|w| last[w]),
            future: between(&|w| latest[w], &|_| nd - 1),
            after: between(&|w| last[w], &|_| nd - 1),
        }
    }

    /// The sum of `exp(eta)` over `cells` by origin, into `reserve`.
    fn expected(&self, theta: &[f64], cells: &[(usize, usize)], reserve: &mut [f64]) {
        reserve.fill(0.0);
        for &(w, d) in cells {
            reserve[w] += self.cells.eta(theta, w, d).exp();
        }
    }

    /// The year's cells with their values.
    fn extra(&self, year: &[f64]) -> Vec<(usize, usize, f64)> {
        self.year
            .iter()
            .zip(year)
            .map(|(&(w, d), &x)| (w, d, x))
            .collect()
    }
}

impl<C: OdpCells> YearModel for OdpYear<'_, C> {
    fn n_origins(&self) -> usize {
        self.n_origins
    }

    fn n_year(&self) -> usize {
        self.year.len()
    }

    fn draw_year(&self, theta: &[f64], rng: &mut StreamRng, year: &mut [f64]) {
        let phi = self.cells.phi();
        for (x, &(w, d)) in year.iter_mut().zip(&self.year) {
            let mu = self.cells.eta(theta, w, d).exp();
            *x = process_draw(self.process, phi, mu, rng);
        }
    }

    fn year_log_likelihood(&self, theta: &[f64], year: &[f64]) -> f64 {
        self.year
            .iter()
            .zip(year)
            .map(|(&(w, d), &x)| {
                let eta = self.cells.eta(theta, w, d);
                x * eta - eta.exp()
            })
            .sum::<f64>()
            / self.cells.phi()
    }

    fn paid(&self, year: &[f64], paid: &mut [f64]) {
        paid.fill(0.0);
        for (&(w, _), x) in self.year.iter().zip(year) {
            paid[w] += x;
        }
    }

    fn opening(&self, theta: &[f64], reserve: &mut [f64]) {
        self.expected(theta, &self.future, reserve);
    }

    fn closing(&self, theta: &[f64], _year: &[f64], reserve: &mut [f64]) {
        self.expected(theta, &self.after, reserve);
    }

    fn closing_uses_year(&self) -> bool {
        false
    }

    fn refit(&self, year: &[f64], seed: u64) -> Result<(PosteriorDraws, bool)> {
        self.cells.refit(&self.extra(year), seed)
    }

    fn log_posterior(&self, theta: &[f64], year: &[f64]) -> f64 {
        self.cells.log_posterior(theta, &self.extra(year))
    }
}

/// The one-year view of a Bayesian model on one segment
/// ([`BayesOdp::one_year`](super::BayesOdp::one_year),
/// [`BayesBf::one_year`](super::BayesBf::one_year)): the lifetime fit, the
/// opening reserve, the claims development result and how reliable each
/// simulation's reweighting was.
///
/// `F` is the model's lifetime fit: [`BayesOdpFit`](super::BayesOdpFit)
/// for the Bayesian ODP, [`BayesBfFit`](super::BayesBfFit) for the
/// Bayesian Bornhuetter–Ferguson.
#[derive(Debug, Clone)]
pub struct BayesOneYearFit<F = super::BayesOdpFit> {
    /// The model fitted to the observed triangle, with its lifetime
    /// reserves.
    pub fit: F,
    /// The latest value plus the opening reserve, per origin.
    pub opening_ultimate: Vec<f64>,
    /// Each origin's posterior mean reserve on the observed triangle: the
    /// mean over the posterior draws of the expected future increments.
    pub opening_reserve: Vec<f64>,
    /// Joint distribution of the claims development result by origin
    /// (dimension `origin`): the opening reserve less the year's payment and
    /// the posterior mean reserve after the year. A negative value is an
    /// adverse development.
    pub cdr: PredictiveDistribution,
    /// Each simulation's Pareto `k̂`, in the order of the simulations; 0
    /// when the year has no cells (a fully developed triangle), whose
    /// simulations are never refitted.
    pub pareto_k: Vec<f64>,
    /// The simulations refitted with NUTS, in order.
    pub refitted: Vec<usize>,
    /// The simulations that moment matching rescued (their `k̂` brought to
    /// at most the threshold), in order; empty unless the settings'
    /// `refit` is [`OneYearRefit::MomentMatchThenRefit`]. Each flagged
    /// simulation is then in this or in `refitted`.
    pub moment_matched: Vec<usize>,
    /// How many of the refits had a diagnostic warning (R-hat, ESS or a
    /// divergence).
    pub refit_warnings: usize,
    /// The settings' threshold, for [`flagged_share`](Self::flagged_share).
    pub k_threshold: f64,
}

impl<F: ReserveFit> BayesOneYearFit<F> {
    /// The share of simulations whose `k̂` is above the threshold (or
    /// NaN), whether or not they were refitted.
    pub fn flagged_share(&self) -> f64 {
        flagged_share(&self.pareto_k, self.k_threshold)
    }

    /// One row per origin: the `latest` value, the `opening_ultimate` and
    /// `opening_reserve`, and the `cdr_mean` and `cdr_std_dev` of the
    /// claims development result.
    pub fn to_long(&self) -> FitTable {
        let mut table = opening_names(self.opened().to_long());
        push_moments(&mut table, "cdr_", &component_sums(&self.cdr));
        table
    }

    /// One row: the totals of [`to_long`](Self::to_long)'s columns, the
    /// model's own totals (the ODP's `scale`), the `flagged_share`, and the
    /// `cdr_mean` and `cdr_std_dev` of the total claims development result.
    pub fn totals(&self) -> FitTable {
        let fits = self.opened();
        let mut table = opening_names(fits.totals());
        table
            .values
            .push(("flagged_share".into(), vec![self.flagged_share()]));
        push_moments(&mut table, "cdr_", &segment_sums(&fits, &self.cdr));
        table
    }

    fn opened(&self) -> SegmentFits<Opened> {
        SegmentFits {
            key_names: Vec::new(),
            labels: vec![crate::triangle::Label::new(Vec::<String>::new())],
            fits: vec![Opened {
                chain_ladder: self.fit.chain_ladder().clone(),
                opening_ultimate: self.opening_ultimate.clone(),
                origins: self.fit.origin_columns(),
                totals: self.fit.total_columns(),
            }],
        }
    }
}

/// Whether a simulation with Pareto `k` is flagged: above `threshold`, or
/// NaN. An infinite `k` (too few draws to fit the tail) is above any
/// finite threshold.
fn flagged(k: f64, threshold: f64) -> bool {
    k.is_nan() || k > threshold
}

/// The share of `pareto_k` above `threshold` or NaN.
fn flagged_share(pareto_k: &[f64], threshold: f64) -> f64 {
    let n = pareto_k.iter().filter(|&&k| flagged(k, threshold)).count();
    n as f64 / pareto_k.len().max(1) as f64
}

/// A fit's chain ladder with the one-year view's opening ultimate, for the
/// long tables.
#[derive(Debug, Clone)]
struct Opened {
    chain_ladder: ChainLadderFit,
    opening_ultimate: Vec<f64>,
    origins: Vec<(&'static str, Vec<f64>)>,
    totals: Vec<(&'static str, f64)>,
}

impl ReserveFit for Opened {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn ultimate(&self) -> &[f64] {
        &self.opening_ultimate
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        self.origins.clone()
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        self.totals.clone()
    }
}

/// One segment of a [`BayesOneYearFits`]: the segment's posterior, its
/// opening reserve and the reliability of its reweighting. Its claims
/// development result is in the joint [`BayesOneYearFits::cdr`].
#[derive(Debug, Clone)]
pub struct BayesOneYearSegment<S = super::BayesOdpSegment> {
    /// The segment's posterior, as in a fit over segments.
    pub fit: S,
    /// The latest value plus the opening reserve, per origin.
    pub opening_ultimate: Vec<f64>,
    /// Each origin's posterior mean reserve on the observed triangle.
    pub opening_reserve: Vec<f64>,
    /// Each simulation's Pareto `k̂` for this segment.
    pub pareto_k: Vec<f64>,
    /// The simulations of this segment refitted with NUTS.
    pub refitted: Vec<usize>,
    /// The simulations of this segment that moment matching rescued.
    pub moment_matched: Vec<usize>,
    /// How many of this segment's refits had a diagnostic warning.
    pub refit_warnings: usize,
    /// The settings' threshold.
    pub k_threshold: f64,
}

impl<S> BayesOneYearSegment<S> {
    /// The share of this segment's simulations whose `k̂` is above the
    /// threshold (or NaN).
    pub fn flagged_share(&self) -> f64 {
        flagged_share(&self.pareto_k, self.k_threshold)
    }
}

impl<S: ReserveFit> ReserveFit for BayesOneYearSegment<S> {
    fn chain_ladder(&self) -> &ChainLadderFit {
        self.fit.chain_ladder()
    }

    fn ultimate(&self) -> &[f64] {
        &self.opening_ultimate
    }

    fn origin_columns(&self) -> Vec<(&'static str, Vec<f64>)> {
        self.fit.origin_columns()
    }

    fn total_columns(&self) -> Vec<(&'static str, f64)> {
        let mut columns = self.fit.total_columns();
        columns.push(("flagged_share", self.flagged_share()));
        columns
    }
}

/// The one-year view of a Bayesian model in every segment of a triangle
/// ([`BayesOdp::one_year_segments`](super::BayesOdp::one_year_segments)),
/// with the layout of [`OneYearFits`](crate::OneYearFits).
#[derive(Debug, Clone)]
pub struct BayesOneYearFits<S = super::BayesOdpSegment> {
    /// Each segment's posterior, opening reserve and Pareto `k̂`.
    pub segments: SegmentFits<BayesOneYearSegment<S>>,
    /// Joint distribution of the claims development result by segment and
    /// origin: the key names and `origin` are its dimensions.
    pub cdr: PredictiveDistribution,
}

impl<S: ReserveFit + Clone> BayesOneYearFits<S> {
    /// One row per segment × origin: the `latest` value, the
    /// `opening_ultimate` and `opening_reserve`, and the `cdr_mean` and
    /// `cdr_std_dev` of the claims development result.
    pub fn to_long(&self) -> FitTable {
        let mut table = opening_names(self.segments.to_long());
        push_moments(&mut table, "cdr_", &component_sums(&self.cdr));
        table
    }

    /// One row per segment: the totals of `to_long`'s columns, the model's
    /// own totals, the `flagged_share`, and the `cdr_mean` and
    /// `cdr_std_dev` of the segment's total claims development result.
    pub fn totals(&self) -> FitTable {
        let mut table = opening_names(self.segments.totals());
        push_moments(&mut table, "cdr_", &segment_sums(&self.segments, &self.cdr));
        table
    }

    /// The one segment chosen as in [`SegmentFits::position`], with its
    /// part of the joint claims development result.
    pub fn segment(&self, keys: &[(&str, &str)]) -> Result<Self> {
        let (segments, cdr) = pick_segment(&self.segments, &self.cdr, keys)?;
        Ok(Self { segments, cdr })
    }
}

/// The provenance fields of the one-year view's settings.
pub(crate) fn provenance(provenance: Provenance, settings: &BayesOneYear) -> Provenance {
    provenance
        .param("k_threshold", settings.k_threshold)
        .param("refit", format!("{:?}", settings.refit))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_threshold_decides_the_refits() {
        let s = BayesOneYear::default();
        assert!(!s.refits(0.69) && s.refits(0.71) && s.refits(f64::INFINITY));
        assert!(s.refits(f64::NAN));
        let only = BayesOneYear {
            refit: OneYearRefit::ReweightOnly,
            ..s
        };
        assert!(!only.refits(f64::INFINITY));
        let all = BayesOneYear {
            refit: OneYearRefit::RefitAll,
            ..s
        };
        assert!(all.refits(-1.0));
        let matching = BayesOneYear {
            refit: OneYearRefit::MomentMatchThenRefit,
            ..s
        };
        assert!(!matching.refits(0.69) && matching.refits(0.71) && matching.refits(f64::NAN));
        assert_eq!(flagged_share(&[0.1, 0.8, f64::INFINITY, 0.7], 0.7), 0.5);
        assert!(
            BayesOneYear {
                k_threshold: f64::NAN,
                ..s
            }
            .validate()
            .is_err()
        );
    }
}

/// The conformance suite of the one-year view (decision 17): checks that
/// need no external reference, run on every [`YearModel`] in its model's
/// unit tests. The checks that need hundreds of NUTS refits run in a
/// release build in `validation/tests/reserving_bayes_one_year_conformance.rs`.
/// A new Bayesian model's one-year view passes both before it ships
/// (AGENTS.md, "Checks").
#[cfg(test)]
pub(crate) mod conformance {
    use super::*;

    /// Runs the suite on `model` under its opening `posterior`, and on
    /// `closed`, the same model fitted to a triangle whose every origin is
    /// at the last age, under its posterior `closed_posterior`:
    ///
    /// 1. the year's log likelihood is the change the year makes to the
    ///    log posterior (the importance ratio targets the posterior a
    ///    refit samples), for years drawn from several posterior draws;
    /// 2. the total claims development result has mean zero within 4.5
    ///    Monte Carlo standard errors over 2,000 reweighted simulations;
    /// 3. a seed replays the view bit for bit, whatever the threads, and
    ///    another seed gives another view;
    /// 4. a year with no cells gives a zero claims development result,
    ///    `k̂` 0 and no refit or moment matching, even under `RefitAll`;
    /// 5. `log_posterior` is the density of the draws as stored, not only
    ///    up to a term in `theta` that check 1 cancels: its score has
    ///    mean zero over the posterior draws ([`score_z`], every
    ///    parameter within [`SCORE_Z`] standard errors). A model that
    ///    stores a constrained transform of its sampling scale and leaves
    ///    out the Jacobian fails it.
    pub(crate) fn check<M: YearModel, C: YearModel>(
        name: &str,
        model: &M,
        posterior: &PosteriorDraws,
        closed: &C,
        closed_posterior: &PosteriorDraws,
    ) {
        let s = posterior.n_draws();
        let ny = model.n_year();
        assert!(ny > 0, "{name}: the open triangle has no year");

        // 1. The importance ratio is the posterior's change.
        let mut rng = StreamRng::new(11, 0);
        let mut year = vec![0.0; ny];
        for j in (0..s).step_by(s / 8) {
            model.draw_year(posterior.draw(j), &mut rng, &mut year);
            let change = |d: usize| {
                let theta = posterior.draw(d);
                let (with, without) = (
                    model.log_posterior(theta, &year),
                    model.log_posterior(theta, &[]),
                );
                (with - without, with.abs() + without.abs())
            };
            let (base, scale0) = change(0);
            let ll0 = model.year_log_likelihood(posterior.draw(0), &year);
            for d in (1..s).step_by(s / 16) {
                let (delta, scale) = change(d);
                let ll = model.year_log_likelihood(posterior.draw(d), &year);
                let gap = (delta - base) - (ll - ll0);
                assert!(
                    gap.abs() <= 1e-10 * (1.0 + scale + scale0),
                    "{name}: draw {d}, year from draw {j}: the log posterior moves by {} and \
                     the year's log likelihood by {}",
                    delta - base,
                    ll - ll0
                );
            }
        }

        // 2. The claims development result has mean zero.
        let only = BayesOneYear {
            refit: OneYearRefit::ReweightOnly,
            ..BayesOneYear::default()
        };
        let n = 2_000;
        let view = simulate(model, posterior, &only, n, 5).unwrap();
        let no = model.n_origins();
        let totals: Vec<f64> = view.cdr.chunks_exact(no).map(|r| r.iter().sum()).collect();
        let mean = totals.iter().sum::<f64>() / n as f64;
        let sd = (totals.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        assert!(
            sd > 0.0,
            "{name}: the claims development result does not vary"
        );
        assert!(
            mean.abs() < 4.5 * sd / (n as f64).sqrt(),
            "{name}: mean total CDR {mean}, sd {sd}"
        );
        assert!(view.refitted.is_empty() && view.pareto_k.len() == n);

        // 3. A seed replays the view, whatever the threads.
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| simulate(model, posterior, &only, 200, 5).unwrap());
        assert_eq!(one.cdr, view.cdr[..200 * no], "{name}: replay");
        assert_eq!(one.pareto_k, view.pareto_k[..200], "{name}: replay");
        let other = simulate(model, posterior, &only, 200, 6).unwrap();
        assert_ne!(other.cdr, one.cdr, "{name}: another seed");

        // 4. A year with no cells changes nothing and refits nothing.
        assert_eq!(closed.n_year(), 0, "{name}: the closed triangle has a year");
        for refit in [
            OneYearRefit::ReweightOnly,
            OneYearRefit::RefitFlagged,
            OneYearRefit::RefitAll,
            OneYearRefit::MomentMatchThenRefit,
        ] {
            let settings = BayesOneYear {
                refit,
                ..BayesOneYear::default()
            };
            let view = simulate(closed, closed_posterior, &settings, 50, 7).unwrap();
            assert!(view.refitted.is_empty(), "{name}: {refit:?} refitted");
            assert!(view.moment_matched.is_empty(), "{name}: {refit:?} matched");
            assert!(view.pareto_k.iter().all(|&k| k == 0.0), "{name}: {refit:?}");
            assert!(
                view.cdr.iter().all(|v| v.abs() <= 1e-9 * (1.0 + v.abs())),
                "{name}: {refit:?} gives a CDR without a year"
            );
        }

        // 5. The log posterior is the stored draws' density.
        let z = score_z(model, posterior);
        let worst = z.iter().copied().fold(0.0, |m: f64, v| m.max(v.abs()));
        assert!(
            worst < SCORE_Z,
            "{name}: the score of log_posterior has mean z {z:?} over the posterior draws"
        );
    }

    /// The bound on [`score_z`] in [`check`]: generous, as NUTS draws
    /// are autocorrelated and the standard error assumes they are not.
    pub(crate) const SCORE_Z: f64 = 6.0;

    /// The Stein (score) identity: for draws from a density `p`, the mean
    /// of `d log p / d theta_i` is zero. For each parameter, the mean over
    /// `posterior`'s draws of the central finite difference of
    /// `log_posterior(theta, [])` (step `1e-5 (1 + |theta_i|)`), over its
    /// standard error taking the draws as independent. A parameter whose
    /// score does not vary gives 0 if its mean is 0 and infinity if not.
    pub(crate) fn score_z<M: YearModel>(model: &M, posterior: &PosteriorDraws) -> Vec<f64> {
        let (s, p) = (posterior.n_draws(), posterior.draw(0).len());
        let mut scores = vec![Vec::with_capacity(s); p];
        let mut theta = vec![0.0; p];
        for d in 0..s {
            theta.copy_from_slice(posterior.draw(d));
            for (i, score) in scores.iter_mut().enumerate() {
                let t = theta[i];
                let h = 1e-5 * (1.0 + t.abs());
                theta[i] = t + h;
                let up = model.log_posterior(&theta, &[]);
                theta[i] = t - h;
                let down = model.log_posterior(&theta, &[]);
                theta[i] = t;
                score.push((up - down) / (2.0 * h));
            }
        }
        scores
            .iter()
            .map(|g| {
                let n = g.len() as f64;
                let mean = g.iter().sum::<f64>() / n;
                let sd = (g.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
                if sd > 0.0 {
                    mean / (sd / n.sqrt())
                } else if mean == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                }
            })
            .collect()
    }
}
