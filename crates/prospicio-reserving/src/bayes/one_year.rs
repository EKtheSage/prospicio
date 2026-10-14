//! The one-year view of a Bayesian reserving model by Pareto smoothed
//! importance sampling (`docs/design/reserving-bayes.md`, decision 16).
//!
//! Each simulation draws the coming year's cells from the posterior
//! predictive (a posterior draw, then process error), reweights every
//! posterior draw by the likelihood of those cells, Pareto-smooths the
//! weights (Vehtari et al. 2024) and takes the time-1 reserve as the
//! weighted posterior mean of the reserve that remains after the year. A
//! simulation whose weights are unreliable (Pareto `k̂` above a threshold)
//! is refitted with NUTS on the extended triangle instead, as
//! [`BayesOneYear::refit`] says.
//!
//! The engine here knows nothing of a particular model: a model plugs in
//! through [`YearModel`], which draws the year, scores it and gives the
//! expected reserve before and after it. [`BayesOdp`](super::BayesOdp)
//! implements it; Meyers' models, the Bayesian Bornhuetter–Ferguson and the
//! compartmental model follow.

use std::sync::Mutex;

use prospicio_bayes::nuts::PosteriorDraws;
use prospicio_core::StreamRng;
use prospicio_prob::{PredictiveDistribution, Provenance};
use rayon::prelude::*;

use crate::bootstrap::{component_sums, pick_segment, push_moments, segment_sums};
use crate::chain_ladder::ChainLadderFit;
use crate::error::{Error, Result};
use crate::one_year_bootstrap::{Failures, opening_names};
use crate::segments::{FitTable, ReserveFit, SegmentFits};

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
}

/// Settings of the one-year view of a Bayesian model
/// (`BayesOdp::one_year`): the Pareto `k̂` above which a simulation's
/// importance weights are not trusted, and what is done then.
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

    /// Whether a simulation with Pareto `k` is refitted.
    fn refits(&self, k: f64) -> bool {
        match self.refit {
            OneYearRefit::ReweightOnly => false,
            // A NaN k is flagged, and an infinite one (too few draws to fit the
            // tail) is above any finite threshold.
            OneYearRefit::RefitFlagged => flagged(k, self.k_threshold),
            OneYearRefit::RefitAll => true,
        }
    }
}

/// A Bayesian model's coming year, as the one-year view needs it. The year
/// is a fixed list of values per simulation (for the ODP model the
/// increments of the year's cells, in a fixed order).
pub(crate) trait YearModel: Sync {
    /// Number of origins, the components of the claims development result.
    fn n_origins(&self) -> usize;
    /// Number of values in a year.
    fn n_year(&self) -> usize;
    /// Draws the year given the parameters `theta`, with process error.
    fn draw_year(&self, theta: &[f64], rng: &mut StreamRng, year: &mut [f64]);
    /// The log likelihood of `year` given `theta`, up to a constant that
    /// does not depend on `theta`: the log importance ratio.
    fn year_log_likelihood(&self, theta: &[f64], year: &[f64]) -> f64;
    /// Each origin's payment in `year`, written into `paid`.
    fn paid(&self, year: &[f64], paid: &mut [f64]);
    /// Each origin's expected reserve given `theta` at the opening.
    fn opening(&self, theta: &[f64], reserve: &mut [f64]);
    /// Each origin's expected reserve after the year, given `theta` and
    /// `year`.
    fn closing(&self, theta: &[f64], year: &[f64], reserve: &mut [f64]);
    /// Whether [`closing`](Self::closing) depends on `year`; when it does
    /// not (the ODP model), it is computed once per posterior draw.
    fn closing_uses_year(&self) -> bool {
        true
    }
    /// The posterior given the observed triangle extended by `year`,
    /// sampled with NUTS from `seed`, and whether its diagnostics warn.
    fn refit(&self, year: &[f64], seed: u64) -> Result<(PosteriorDraws, bool)>;
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
    /// How many refits had a diagnostic warning.
    pub(crate) refit_warnings: usize,
}

/// One simulation's outcome.
struct Sim {
    cdr: Vec<f64>,
    k: f64,
    refitted: bool,
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

    let failures = Mutex::new(Failures::default());
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
                psis(&log_r)
            };
            let mut closing = vec![0.0; no];
            let (refitted, mut warned) = (!uninformative && settings.refits(k), false);
            if refitted {
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
                        failures
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .record(e);
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
                warned,
            })
        })
        .collect();
    let failures = failures
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((message, _)) = failures.first {
        return Err(Error::Bayes(format!(
            "the one-year view's refit failed in {} of {n_sims} simulations, for example: {message}",
            failures.count
        )));
    }
    let mut out = YearDraws {
        opening,
        cdr: Vec::with_capacity(n_sims * no),
        pareto_k: Vec::with_capacity(n_sims),
        refitted: Vec::new(),
        refit_warnings: 0,
    };
    for (i, sim) in sims.into_iter().enumerate() {
        let sim = sim.expect("a failed simulation is reported above");
        out.cdr.extend(sim.cdr);
        out.pareto_k.push(sim.k);
        if sim.refitted {
            out.refitted.push(i);
        }
        out.refit_warnings += usize::from(sim.warned);
    }
    Ok(out)
}

/// Pareto smoothed log importance weights for the log ratios `log_r` and
/// the fitted shape `k̂` (infinite when the tail is too short to fit): the
/// algorithm of `prospicio_bayes::elpd`'s PSIS-LOO with relative
/// efficiency 1, the tail `M = min(0.2 S, 3 sqrt(S))` of the `S` ratios
/// replaced by the quantiles of a generalized Pareto fitted by Zhang and
/// Stephens' method and shrunk towards 0.5 (Vehtari et al. 2024). A test
/// checks that `k̂` is `prospicio_bayes::elpd::loo`'s on the same ratios.
pub(crate) fn psis(log_r: &[f64]) -> (Vec<f64>, f64) {
    let s = log_r.len();
    let max = log_r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut lw: Vec<f64> = log_r.iter().map(|v| v - max).collect();
    let m = ((0.2 * s as f64).ceil()).min((3.0 * (s as f64).sqrt()).ceil()) as usize;
    if m < 5 || m >= s {
        return (lw, f64::INFINITY);
    }
    let mut order: Vec<usize> = (0..s).collect();
    order.sort_by(|&a, &b| lw[a].total_cmp(&lw[b]));
    let tail = &order[s - m..];
    let cutoff = lw[order[s - m - 1]];
    let exceed: Vec<f64> = tail.iter().map(|&j| lw[j].exp() - cutoff.exp()).collect();
    if exceed.iter().all(|&x| x <= 0.0) {
        return (lw, f64::INFINITY);
    }
    let (k, sigma) = gpd_fit(&exceed);
    for (z, &j) in tail.iter().enumerate() {
        let p = (z as f64 + 0.5) / m as f64;
        let q = gpd_quantile(p, k, sigma);
        lw[j] = (cutoff.exp() + q).ln().min(0.0);
    }
    (lw, k)
}

/// The generalized Pareto quantile `σ/k ((1 - p)^(-k) - 1)`.
fn gpd_quantile(p: f64, k: f64, sigma: f64) -> f64 {
    if k.abs() < 1e-12 {
        -sigma * (-p).ln_1p()
    } else {
        sigma / k * ((-k * (-p).ln_1p()).exp() - 1.0)
    }
}

/// Zhang and Stephens' (2009) estimate of the generalized Pareto shape and
/// scale of the exceedances `x`, shrunk towards `k = 0.5` as a prior worth
/// 10 observations would.
fn gpd_fit(x: &[f64]) -> (f64, f64) {
    let mut x = x.to_vec();
    x.sort_by(f64::total_cmp);
    let n = x.len();
    let nf = n as f64;
    let m = 30 + nf.sqrt().floor() as usize;
    let x_max = x[n - 1];
    let x_q = x[((nf / 4.0 + 0.5).floor() as usize).max(1) - 1];
    let thetas: Vec<f64> = (1..=m)
        .map(|j| 1.0 / x_max + (1.0 - (m as f64 / (j as f64 - 0.5)).sqrt()) / (3.0 * x_q))
        .collect();
    let k_of = |theta: f64| x.iter().map(|v| (-theta * v).ln_1p()).sum::<f64>() / nf;
    let ll: Vec<f64> = thetas
        .iter()
        .map(|&t| {
            let k = k_of(t);
            nf * ((-t / k).ln() - k - 1.0)
        })
        .collect();
    let weights: Vec<f64> = ll
        .iter()
        .map(|&lj| {
            let total: f64 = ll.iter().map(|&li| (li - lj).exp()).sum();
            if total.is_finite() { 1.0 / total } else { 0.0 }
        })
        .collect();
    let wsum: f64 = weights.iter().sum();
    let theta: f64 = thetas.iter().zip(&weights).map(|(t, w)| t * w).sum::<f64>() / wsum;
    let k = k_of(theta);
    let sigma = -k / theta;
    let k = (nf * k + 10.0 * 0.5) / (nf + 10.0);
    (k, sigma)
}

/// The one-year view of a Bayesian model on one segment
/// ([`BayesOdp::one_year`](super::BayesOdp::one_year)): the lifetime fit,
/// the opening reserve, the claims development result and how reliable
/// each simulation's reweighting was.
///
/// `F` is the model's lifetime fit, [`BayesOdpFit`](super::BayesOdpFit)
/// for the Bayesian ODP.
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
    totals: Vec<(&'static str, f64)>,
}

impl ReserveFit for Opened {
    fn chain_ladder(&self) -> &ChainLadderFit {
        &self.chain_ladder
    }

    fn ultimate(&self) -> &[f64] {
        &self.opening_ultimate
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
    fn the_shape_is_prospicio_bayes_loo_s() {
        // Log ratios with a heavy right tail and a light one: loo smooths
        // the ratios of minus its log likelihood, so it gets -log_r.
        let mut rng = StreamRng::new(3, 0);
        for scale in [0.3, 1.0, 3.0] {
            let log_r: Vec<f64> = (0..4000)
                .map(|_| scale * prospicio_math::special::norm_quantile(rng.next_open01()))
                .collect();
            let (lw, k) = psis(&log_r);
            let minus: Vec<f64> = log_r.iter().map(|v| -v).collect();
            let loo = prospicio_bayes::elpd::loo(&minus, 1, None).unwrap();
            assert_eq!(k, loo.pareto_k[0], "scale {scale}");
            assert!(lw.iter().all(|v| v.is_finite() && *v <= 0.0));
        }
    }

    #[test]
    fn equal_ratios_give_equal_weights() {
        let (lw, k) = psis(&[0.5; 1000]);
        assert!(lw.iter().all(|&v| v == 0.0));
        assert!(k.is_infinite());
        let (_, k) = psis(&[0.0, 1.0, 2.0]);
        assert!(k.is_infinite(), "too few ratios for a tail");
    }

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
