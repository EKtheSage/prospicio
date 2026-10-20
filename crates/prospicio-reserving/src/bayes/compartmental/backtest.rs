//! The compartmental model in the diagonal backtest
//! (`docs/design/reserving-compartmental.md`, decision 8):
//! [`CompartmentalCandidate`] implements [`TriangleModel`] for the paid
//! increments, and scores the held-out outstanding diagonal as a second
//! table ([`CompartmentalCandidate::outstanding_backtest`]).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use prospicio_bayes::nuts::PosteriorDraws;
use prospicio_core::StreamRng;
use prospicio_prob::{InputHasher, KeyValue};

use super::{
    Compartmental, CompartmentalLikelihood, Data, Future, KER, KP, Model, Outstanding,
    PaidResponse, RLR, RRF, lognormal, one_os,
};
use crate::backtest::{
    Backtest, CellForecast, METRICS, ScoredSplits, TriangleModel, check_settings, score,
    scored_splits,
};
use crate::bayes::laplace::Posterior;
use crate::bayes::odp::log_mean_exp;
use crate::bayes::{Stream, derive_seed};
use crate::diagnostics::Diagnostics;
use crate::error::{Error, Result};
use crate::frame::TriangleFrame;
use crate::triangle::Triangle;

/// How many fitted splits a candidate keeps, so that the paid table and the
/// outstanding table of the same backtest share their fits.
/// The triangle's outstanding by data cell `(origin index, age index)`.
type Outstandings = BTreeMap<(usize, usize), f64>;

const CACHED_FITS: usize = 4;

/// A [`Compartmental`] for the diagonal backtest: the model's settings with
/// the triangle that holds the paid, outstanding (or incurred) and premium
/// columns. The backtest's [`TriangleFrame`] holds paid increments only, so
/// the candidate takes the other columns from its own triangle at the
/// training rows' cells, and the premium of each origin from the premium
/// column as [`Compartmental::fit`] does (it is known, so reading it
/// for the origins of the held-out diagonal is no leak). It must be built on
/// the triangle the frame was built from.
///
/// As a [`TriangleModel`] (its name is the model's: `compartmental_gm1`,
/// `compartmental_gm2` or `compartmental`):
///
/// * **Training cells.** The train rows of each origin must run from the
///   first age without a gap (an [`Error::Bayes`] otherwise); their running
///   sums are its cumulative paid. [`fit_rows`](TriangleModel::fit_rows)
///   drops the rows the likelihood rejects, as the Poisson GLM drops
///   negative increments: under the log-normal likelihood the first row of
///   an origin whose response is not positive (the increment, or the
///   cumulative under cumulative paid) and every later row of the origin
///   (an incremental cell needs the paid before it); a row of the normal
///   likelihood is never rejected. The outstanding of a training cell is
///   its triangle's value at that origin and age, so no outstanding of a
///   held-out diagonal reaches the fit, and an outstanding cell the
///   log-normal likelihood rejects (not positive) is left out.
/// * **Forecast.** One fit on the training cells, with the chains seeded
///   from the backtest's `seed` as [`Compartmental::fit`] seeds them from
///   [`Compartmental::seed`]; the simulations take the predictive seed
///   likewise, and the backtest's `n_sims` replaces
///   [`Compartmental::n_sims`]. Each origin's paid is simulated along one
///   path from the age after its latest training row to its oldest test
///   row, one posterior draw per simulation, and a test row's increment is
///   the difference of consecutive values on it (the first from the latest
///   training paid), so an origin's increments sum to its last simulated
///   paid less its latest, which is how the fit's reserves are made. The
///   fit's diagnostics come back in [`CellForecast::diagnostics`] and
///   [`Backtest::diagnostics`], and a fit with warnings is still scored.
///   A held-out triangle is smaller than the whole: check a backtest's
///   fits with [`Compartmental::fit`] on the same training cells before
///   relying on its scores.
/// * **Mean** is the mean of the predictive draws of the row (a Monte
///   Carlo mean over [`n_sims`](Compartmental::n_sims) draws, which for the
///   log-normal likelihood includes the factor `exp(sigma^2 / 2)` over the
///   median the cell is drawn about).
/// * **Log density.** When every test row follows its origin's latest
///   training row (always so in
///   [`diagonal_backtest`](crate::diagonal_backtest)) and its origin and
///   age have an observed cell, a test increment `y` is a known amount
///   plus a normal (model 1) or a log-normal (model 2) response given the
///   parameters, so its density is closed: the cumulative paid at the cell
///   is normal about its mean at `y` plus the latest paid, or the increment
///   itself is, as the paid response says. The log density is the log of
///   the mean of that density over every posterior draw (the log
///   predictive density, as [`BayesOdp`](super::super::BayesOdp) gives it);
///   `None` when some test row is further out or its origin or age has no
///   observed cell.
///
/// ```
/// use prospicio_reserving::bayes::{
///     Compartmental, CompartmentalCandidate, CompartmentalPriors, Outstanding,
/// };
/// use prospicio_reserving::{
///     DevelopmentColumn, Grain, Long, Month, Triangle, TriangleFrame, TriangleModel,
/// };
///
/// let (mut origin, mut age, mut paid, mut incurred, mut premium) =
///     (vec![], vec![], vec![], vec![], vec![]);
/// for (w, p) in [100.0, 110.0, 120.0].into_iter().enumerate() {
///     for d in 0..(3 - w) {
///         origin.push(Month::january(2020 + w as i32));
///         age.push(12 * (d as u32 + 1));
///         paid.push(p * [0.4, 0.7, 0.8][d]);
///         incurred.push(p * 0.9);
///         premium.push(p * 1.5);
///     }
/// }
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&age),
///     values: &[("paid", &paid), ("incurred", &incurred), ("premium", &premium)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let candidate = CompartmentalCandidate::new(
///     Compartmental::new(CompartmentalPriors::gesmann_morris()),
///     &tri,
///     "paid",
///     Outstanding::FromIncurred("incurred"),
///     "premium",
/// );
/// assert_eq!(candidate.name(), "compartmental_gm1");
/// let cells = TriangleFrame::new(&tri, "paid", None)?;
/// // Every observed row of a gap-free triangle is fitted by the normal model.
/// assert_eq!(candidate.fit_rows(&cells, cells.observed()), cells.observed());
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
///
/// # Settings are fixed
///
/// A candidate keeps the fits of its last few splits, shared between the
/// paid and the outstanding backtests of the same split. Its model, triangle
/// and columns are therefore private and read through accessors: to try other
/// settings, build a new candidate, so a cached fit is never reused under
/// settings other than the ones it was made with.
///
/// ```compile_fail,E0616
/// use prospicio_reserving::bayes::{
///     Compartmental, CompartmentalCandidate, CompartmentalPriors, Outstanding,
/// };
/// use prospicio_reserving::{
///     DevelopmentColumn, Grain, Long, Month, Triangle, TriangleFrame, TriangleModel,
/// };
///
/// let (mut origin, mut age, mut paid, mut incurred, mut premium) =
///     (vec![], vec![], vec![], vec![], vec![]);
/// for (w, p) in [100.0, 110.0, 120.0].into_iter().enumerate() {
///     for d in 0..(3 - w) {
///         origin.push(Month::january(2020 + w as i32));
///         age.push(12 * (d as u32 + 1));
///         paid.push(p * [0.4, 0.7, 0.8][d]);
///         incurred.push(p * 0.9);
///         premium.push(p * 1.5);
///     }
/// }
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&age),
///     values: &[("paid", &paid), ("incurred", &incurred), ("premium", &premium)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let mut candidate = CompartmentalCandidate::new(
///     Compartmental::new(CompartmentalPriors::gesmann_morris()),
///     &tri,
///     "paid",
///     Outstanding::FromIncurred("incurred"),
///     "premium",
/// );
/// candidate.paid = "incurred";
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
#[derive(Debug)]
pub struct CompartmentalCandidate<'a> {
    model: Compartmental,
    triangle: &'a Triangle,
    paid: &'a str,
    outstanding: Outstanding<'a>,
    premium: &'a str,
    fits: Mutex<Vec<Arc<TrainFit>>>,
}

impl<'a> CompartmentalCandidate<'a> {
    /// A candidate for `model` on `triangle`.
    pub fn new(
        model: Compartmental,
        triangle: &'a Triangle,
        paid: &'a str,
        outstanding: Outstanding<'a>,
        premium: &'a str,
    ) -> Self {
        Self {
            model,
            triangle,
            paid,
            outstanding,
            premium,
            fits: Mutex::new(Vec::new()),
        }
    }

    /// The model and its settings.
    pub fn model(&self) -> &Compartmental {
        &self.model
    }

    /// The triangle the frame was built on.
    pub fn triangle(&self) -> &'a Triangle {
        self.triangle
    }

    /// The cumulative paid column (the frame's column).
    pub fn paid(&self) -> &'a str {
        self.paid
    }

    /// The outstanding column, or the incurred column it comes from.
    pub fn outstanding(&self) -> Outstanding<'a> {
        self.outstanding
    }

    /// The premium column.
    pub fn premium(&self) -> &'a str {
        self.premium
    }
}

/// A fit on the training cells of one split, shared between the paid and
/// the outstanding forecasts of the same backtest.
#[derive(Debug)]
struct TrainFit {
    train: Vec<usize>,
    seed: u64,
    data: Data,
    model: Model,
    posterior: PosteriorDraws,
    /// The sampler diagnostics of the fit.
    diagnostics: Diagnostics,
    /// The triangle's outstanding at every cell it has one, held out or not.
    full_outstanding: Outstandings,
}

impl CompartmentalCandidate<'_> {
    /// The training rows the likelihood accepts, each with its origin's
    /// cumulative paid there, in row order within origins.
    fn kept(&self, cells: &TriangleFrame, train: &[usize]) -> Result<Vec<(usize, f64)>> {
        let mut by_origin: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for &r in train {
            by_origin
                .entry(cells.origin_of(r))
                .or_default()
                .push((cells.development_of(r), r));
        }
        let log_normal = self.model.likelihood == CompartmentalLikelihood::LogNormal;
        let incremental = self.model.paid_response == PaidResponse::Incremental;
        let response = cells.response();
        let mut kept = Vec::with_capacity(train.len());
        for (&o, rows) in &mut by_origin {
            rows.sort_unstable();
            let mut cumulative = 0.0;
            let mut rejected = false;
            for (k, &(d, r)) in rows.iter().enumerate() {
                if d != k {
                    return Err(Error::Bayes(format!(
                        "origin {} has no train row at development age {} months but has one \
                         later; the compartmental model rebuilds each origin's cumulative \
                         paid from its first age",
                        cells.origins()[o],
                        cells.ages()[k]
                    )));
                }
                let increment = response[r];
                cumulative += increment;
                let positive = if incremental { increment } else { cumulative } > 0.0;
                let accepted =
                    increment.is_finite() && cumulative.is_finite() && (!log_normal || positive);
                rejected |= !accepted;
                if !rejected {
                    kept.push((r, cumulative));
                }
            }
        }
        Ok(kept)
    }

    /// The data cell `(origin index, age index)` of frame row `r`.
    fn locate(&self, cells: &TriangleFrame, data: &Data, r: usize) -> Result<(usize, usize)> {
        let period = cells.origins()[cells.origin_of(r)];
        let age = cells.ages()[cells.development_of(r)];
        let months = data.grain.months();
        let i = data
            .origins
            .iter()
            .position(|&p| p == period)
            .ok_or_else(|| {
                Error::Bayes(format!(
                    "origin {period} of the frame is not an origin of the premium column {}",
                    self.premium
                ))
            })?;
        if age == 0 || age % months != 0 {
            return Err(Error::Bayes(format!(
                "development age {age} months is not a whole number of {} periods",
                data.grain
            )));
        }
        Ok((i, (age / months - 1) as usize))
    }

    /// The data of a fit on the `train` rows (their rebuilt paid and the
    /// outstanding at their cells), and the triangle's outstanding at every
    /// cell.
    fn prepare(&self, cells: &TriangleFrame, train: &[usize]) -> Result<(Data, Outstandings)> {
        let kept = self.kept(cells, train)?;
        let mut data = Data::new(self.triangle, self.paid, self.outstanding, self.premium)?;
        let full_outstanding: Outstandings = data
            .outstanding
            .iter()
            .map(|&(i, k, v)| ((i, k), v))
            .collect();
        let positions = train
            .iter()
            .map(|&r| self.locate(cells, &data, r))
            .collect::<Result<BTreeSet<_>>>()?;
        data.paid = kept
            .iter()
            .map(|&(r, cumulative)| {
                self.locate(cells, &data, r)
                    .map(|(i, k)| (i, k, cumulative))
            })
            .collect::<Result<Vec<_>>>()?;
        let log_normal = self.model.likelihood == CompartmentalLikelihood::LogNormal;
        data.outstanding = full_outstanding
            .iter()
            .filter(|&(&(i, k), &v)| {
                positions.contains(&(i, k)) && (!log_normal || (v > 0.0 && v.is_finite()))
            })
            .map(|(&(i, k), &v)| (i, k, v))
            .collect();
        if data.outstanding.is_empty() {
            return Err(Error::Bayes(
                "no training cell has an outstanding the likelihood accepts".into(),
            ));
        }
        data.n_ages = data.origins.len().max(
            data.paid
                .iter()
                .chain(&data.outstanding)
                .map(|c| c.1 + 1)
                .max()
                .unwrap_or(0),
        );
        data.set_latest();
        Ok((data, full_outstanding))
    }

    /// The fit on the `train` rows with chains seeded from `seed`, or the
    /// one a forecast of the same split made.
    fn train_fit(
        &self,
        cells: &TriangleFrame,
        train: &[usize],
        seed: u64,
    ) -> Result<Arc<TrainFit>> {
        let cached = {
            let fits = self.fits.lock().unwrap_or_else(|e| e.into_inner());
            fits.iter()
                .find(|f| f.seed == seed && f.train == train)
                .cloned()
        };
        if let Some(fit) = cached {
            return Ok(fit);
        }
        self.model.validate()?;
        let (data, full_outstanding) = self.prepare(cells, train)?;
        let model = Model::new(
            &data,
            self.model.priors,
            self.model.rate_order == super::RateOrder::Ordered,
            self.model.response(),
        )?;
        if model.cells.len() <= model.dim() {
            return Err(Error::Bayes(format!(
                "{} training cells for {} parameters; the model needs more cells than parameters",
                model.cells.len(),
                model.dim()
            )));
        }
        let (posterior, _, diagnostics) = self
            .model
            .sample(&model, derive_seed(seed, Stream::Chains))?;
        let fit = Arc::new(TrainFit {
            train: train.to_vec(),
            seed,
            data,
            model,
            posterior,
            diagnostics,
            full_outstanding,
        });
        let mut fits = self.fits.lock().unwrap_or_else(|e| e.into_inner());
        if fits.len() >= CACHED_FITS {
            fits.remove(0);
        }
        fits.push(Arc::clone(&fit));
        Ok(fit)
    }

    /// The paid increments of the `test` rows from `fit`.
    fn paid_forecast(
        &self,
        fit: &TrainFit,
        cells: &TriangleFrame,
        test: &[usize],
        n_sims: usize,
        seed: u64,
    ) -> Result<CellForecast> {
        let data = &fit.data;
        let n = data.origins.len();
        let at = test
            .iter()
            .map(|&r| self.locate(cells, data, r))
            .collect::<Result<Vec<_>>>()?;
        // Each origin's path runs from the age after its latest training
        // row to its oldest test row.
        let first = |i: usize| data.latest_paid[i].map_or(0, |(k, _)| k + 1);
        let latest = |i: usize| data.latest_paid[i].map_or(0.0, |(_, v)| v);
        let mut end: Vec<Option<usize>> = vec![None; n];
        for (&r, &(i, k)) in test.iter().zip(&at) {
            if k < first(i) {
                return Err(Error::Bayes(format!(
                    "test row {r} is not after its origin's latest training row; the \
                     compartmental model forecasts each origin's paid past it"
                )));
            }
            end[i] = Some(end[i].map_or(k, |e| e.max(k)));
        }
        let mut path: Vec<(usize, usize)> = Vec::new();
        let mut position: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (i, e) in end.iter().enumerate() {
            if let Some(e) = *e {
                for k in first(i)..=e {
                    position.insert((i, k), path.len());
                    path.push((i, k));
                }
            }
        }
        let future = Future {
            paid: path.clone(),
            outstanding: Vec::new(),
            last_paid: vec![None; n],
        };
        let n_path = path.len();
        let increment = |values: &[f64], (i, k): (usize, usize)| -> f64 {
            let before = k
                .checked_sub(1)
                .and_then(|b| position.get(&(i, b)))
                .map_or_else(|| latest(i), |&j| values[j]);
            values[position[&(i, k)]] - before
        };
        let distribution = fit.posterior.predictive(
            vec!["row".into()],
            test.iter()
                .map(|&r| vec![KeyValue::Int(r as i64)])
                .collect(),
            n_sims,
            derive_seed(seed, Stream::Predictive),
            self.model
                .provenance(
                    self.paid,
                    self.outstanding,
                    self.premium,
                    InputHasher::new(),
                )
                .param("output", "test increments"),
            |nat, rng: &mut StreamRng, row| {
                let mut values = vec![0.0; n_path];
                fit.model.simulate(nat, &future, rng, &mut values);
                for (x, &cell) in row.iter_mut().zip(&at) {
                    *x = increment(&values, cell);
                }
            },
        )?;
        let mean = column_means(&distribution);
        let response = cells.response();
        let closed = at.iter().all(|&(i, k)| k == first(i));
        let log_density = closed
            .then(|| {
                at.iter()
                    .zip(test)
                    .map(|(&(i, k), &r)| fit.log_density_at(i, k, response[r], true))
                    .collect::<Option<Vec<f64>>>()
            })
            .flatten();
        Ok(CellForecast {
            mean,
            distribution,
            log_density,
            diagnostics: Some(fit.diagnostics.clone()),
        })
    }

    /// The outstanding amounts of the `test` rows from `fit`: the cells the
    /// triangle holds at their origin and age, which the fit never saw.
    fn outstanding_forecast(
        &self,
        fit: &TrainFit,
        cells: &TriangleFrame,
        test: &[usize],
        actual: &[f64],
        n_sims: usize,
        seed: u64,
    ) -> Result<CellForecast> {
        let data = &fit.data;
        let n = data.origins.len();
        let at = test
            .iter()
            .map(|&r| self.locate(cells, data, r))
            .collect::<Result<Vec<_>>>()?;
        for (&r, &(i, k)) in test.iter().zip(&at) {
            if data.latest_outstanding[i].is_some_and(|l| k <= l) {
                return Err(Error::Bayes(format!(
                    "test row {r} is not after its origin's latest training outstanding"
                )));
            }
        }
        let future = Future {
            paid: Vec::new(),
            outstanding: at.clone(),
            last_paid: vec![None; n],
        };
        let distribution = fit.posterior.predictive(
            vec!["row".into()],
            test.iter()
                .map(|&r| vec![KeyValue::Int(r as i64)])
                .collect(),
            n_sims,
            derive_seed(seed, Stream::Predictive),
            self.model
                .provenance(
                    self.paid,
                    self.outstanding,
                    self.premium,
                    InputHasher::new(),
                )
                .param("output", "test outstanding"),
            |nat, rng: &mut StreamRng, row| {
                fit.model.simulate(nat, &future, rng, row);
            },
        )?;
        let mean = column_means(&distribution);
        let log_density = at
            .iter()
            .zip(actual)
            .map(|(&(i, k), &y)| fit.log_density_at(i, k, y, false))
            .collect::<Option<Vec<f64>>>();
        Ok(CellForecast {
            mean,
            distribution,
            log_density,
            diagnostics: Some(fit.diagnostics.clone()),
        })
    }

    /// The scores of the held-out outstanding diagonals, a second table
    /// beside [`diagonal_backtest`](crate::diagonal_backtest)'s paid one
    /// (decision 8): for each of the latest `n_diagonals` calendar
    /// diagonals of `cells` the candidate is fitted on the earlier
    /// diagonals (the outstanding of the held-out diagonal is always
    /// withheld) and forecasts the outstanding of the diagonal's cells,
    /// scored by the backtest's metrics ([`METRICS`]): the mean CRPS of the
    /// cells, the coverage of the central `interval`, actual over expected
    /// on the diagonal total, the CRPS of the total from the joint draws,
    /// the mean held-out log density and the percentile of the actual
    /// total among the simulated ones. Each split reports the diagnostics
    /// of its fit ([`Backtest::diagnostics`]). The splits, `n_sims`, `seed` and `interval` are
    /// [`diagonal_backtest`](crate::diagonal_backtest)'s, so the fits of a
    /// backtest of this candidate on the same arguments are shared with this
    /// one (the candidate keeps its last four).
    ///
    /// The cells scored are the held-out diagonal's rows that the candidate
    /// fits an origin and age of (as [`diagonal_backtest`](crate::diagonal_backtest)
    /// with this candidate alone) and whose outstanding the triangle holds;
    /// [`Backtest::excluded`] counts the others, and
    /// [`scored_rows`](Backtest::scored_rows) are frame rows. The log
    /// density of a cell is that of its closed form (normal for model 1,
    /// log-normal for model 2), the log of its mean over the posterior
    /// draws. The one model of the table is named after the candidate with
    /// `_outstanding`.
    pub fn outstanding_backtest(
        &self,
        cells: &TriangleFrame,
        n_diagonals: usize,
        n_sims: usize,
        seed: u64,
        interval: f64,
    ) -> Result<Backtest> {
        check_settings(n_sims, interval)?;
        let ScoredSplits {
            splits,
            mut excluded,
            mut scored,
        } = scored_splits(cells, &[self as &dyn TriangleModel], n_diagonals)?;
        let n_splits = splits.len();
        let n_metrics = METRICS.len();
        let mut scores = vec![f64::NAN; n_metrics * n_splits];
        let mut log_densities = Some(Vec::new());
        let mut diagnostics = Vec::with_capacity(n_splits);
        for (k, (split, rows)) in splits.iter().zip(scored.iter_mut()).enumerate() {
            let fit = self.train_fit(cells, &split.train, seed)?;
            let before = rows.len();
            let mut actual = Vec::with_capacity(before);
            rows.retain(|&r| {
                let value = self
                    .locate(cells, &fit.data, r)
                    .ok()
                    .and_then(|c| fit.full_outstanding.get(&c).copied());
                actual.extend(value);
                value.is_some()
            });
            if rows.is_empty() {
                return Err(Error::Bayes(format!(
                    "no held-out row of diagonal {} has an outstanding",
                    cells.calendar_of(split.test[0])
                )));
            }
            excluded[k] += before - rows.len();
            let f = self.outstanding_forecast(&fit, cells, rows, &actual, n_sims, seed)?;
            for (metric, value) in score(&f, &actual, interval)?.into_iter().enumerate() {
                scores[metric * n_splits + k] = value;
            }
            diagnostics.push(f.diagnostics);
            log_densities = match (log_densities.take(), f.log_density) {
                (Some(mut all), Some(lpd)) => {
                    all.extend(lpd);
                    Some(all)
                }
                _ => None,
            };
        }
        Ok(Backtest {
            models: vec![format!("{}_outstanding", self.name())],
            n_splits,
            scores,
            excluded,
            scored,
            log_densities: vec![log_densities],
            diagnostics,
            diagonals: splits
                .iter()
                .map(|s| cells.calendar_of(s.test[0]))
                .collect(),
        })
    }
}

/// The mean of each component's draws.
fn column_means(distribution: &prospicio_prob::PredictiveDistribution) -> Vec<f64> {
    let (n, n_sims) = (distribution.n_components(), distribution.n_sims());
    let mut mean = vec![0.0; n];
    for row in distribution.draw_matrix().chunks_exact(n) {
        for (m, x) in mean.iter_mut().zip(row) {
            *m += x / n_sims as f64;
        }
    }
    mean
}

impl TrainFit {
    /// The log of the mean over the posterior draws of the density of
    /// `y` at cell `(i, k)`: a paid increment after the origin's latest
    /// paid (`paid`) or an outstanding amount. `None` when the origin or
    /// the age has no observed cell, so its effect would have to be drawn.
    fn log_density_at(&self, i: usize, k: usize, y: f64, paid: bool) -> Option<f64> {
        let latest = self.data.latest_paid[i].map_or(0.0, |(_, v)| v);
        let n = self.posterior.n_draws();
        let mut ln_f = Vec::with_capacity(n);
        for s in 0..n {
            let nat = self.posterior.draw(s);
            ln_f.push(if paid {
                self.model.paid_log_density(nat, i, k, latest, y)?
            } else {
                self.model.outstanding_log_density(nat, i, k, y)?
            });
        }
        Some(log_mean_exp(&ln_f))
    }
}

impl TriangleModel for CompartmentalCandidate<'_> {
    fn name(&self) -> &str {
        self.model.model_name()
    }

    fn forecast(
        &self,
        cells: &TriangleFrame,
        train: &[usize],
        test: &[usize],
        n_sims: usize,
        seed: u64,
    ) -> Result<CellForecast> {
        let fit = self.train_fit(cells, train, seed)?;
        self.paid_forecast(&fit, cells, test, n_sims, seed)
    }

    fn fit_rows(&self, cells: &TriangleFrame, train: &[usize]) -> Vec<usize> {
        // A gap in an origin is an error the forecast reports.
        match self.kept(cells, train) {
            Ok(kept) => kept.into_iter().map(|(r, _)| r).collect(),
            Err(_) => train.to_vec(),
        }
    }
}

const HALF_LN_TAU: f64 = 0.918_938_533_204_672_7;

impl Model {
    /// The development multipliers of the parameters at age index `k` from
    /// a natural-scale draw, `None` for an age with no observed cell (a
    /// new effect would have to be drawn).
    fn dev_multiplier_at(&self, nat: &[f64], k: usize) -> Option<[f64; 4]> {
        if !self.has_dev() {
            return Some([1.0; 4]);
        }
        let lay = &self.layout;
        let ms = self.priors.positive().map(lognormal);
        let d = self.dev_level[k]?;
        let mut out = [1.0; 4];
        for (p, x) in out.iter_mut().enumerate() {
            if let (Some(first), Some(_)) = (lay.nat_dev[p], lay.nat_tau_dev[p]) {
                *x = (ms[p].1 * nat[first + d]).exp();
            }
        }
        Some(out)
    }

    /// The cell's parameters and closed forms `(OS, PD)` at age `k` and
    /// its PD at the age before, as [`simulate`](Self::simulate) forms
    /// them, for an origin and age with an observed cell; `None` otherwise.
    fn closed_cell(&self, nat: &[f64], i: usize, k: usize) -> Option<([f64; 4], (f64, f64), f64)> {
        self.level[i]?;
        // The origin has a level, so no new effect is drawn.
        let mut unused = StreamRng::new(0, 0);
        let th = self.origin_values(nat, i, &mut unused);
        if !self.cellwise {
            let (k_er, k_p) = (nat[KER], nat[KP]);
            let before = if k == 0 {
                0.0
            } else {
                one_os(k_er, k_p, self.times[k - 1]).1
            };
            return Some((th, one_os(k_er, k_p, self.times[k]), before));
        }
        let multiplier = self.dev_multiplier_at(nat, k)?;
        let th: [f64; 4] = std::array::from_fn(|p| th[p] * multiplier[p]);
        let now = one_os(th[KER], th[KP], self.times[k]);
        let before = if k == 0 {
            0.0
        } else {
            one_os(th[KER], th[KP], self.times[k - 1]).1
        };
        Some((th, now, before))
    }

    /// The log density of a response `value` about `mean` (a median under
    /// the log-normal likelihood) with scale `sigma`, for origin `i`.
    fn response_log_density(&self, value: f64, mean: f64, sigma: f64, i: usize) -> f64 {
        if self.response.log_normal() {
            if value > 0.0 && mean > 0.0 {
                let z = (value / mean).ln() / sigma;
                -value.ln() - sigma.ln() - HALF_LN_TAU - 0.5 * z * z
            } else {
                f64::NEG_INFINITY
            }
        } else {
            let s = if self.response.loss_ratio {
                sigma * self.premium[i]
            } else {
                sigma
            };
            let z = (value - mean) / s;
            -s.ln() - HALF_LN_TAU - 0.5 * z * z
        }
    }

    /// The log density, per unit of the amount, of a paid increment `y`
    /// after the cumulative paid `latest` at the age before cell `(i, k)`,
    /// given a natural-scale draw.
    fn paid_log_density(
        &self,
        nat: &[f64],
        i: usize,
        k: usize,
        latest: f64,
        y: f64,
    ) -> Option<f64> {
        let (th, (_, pd), before) = self.closed_cell(nat, i, k)?;
        let scale = self.premium[i] * th[RLR] * th[RRF];
        let (value, mean) = if self.response.incremental {
            (y, scale * (pd - before))
        } else {
            (latest + y, scale * pd)
        };
        Some(self.response_log_density(value, mean, nat[5], i))
    }

    /// The log density of an outstanding amount `y` at cell `(i, k)`,
    /// given a natural-scale draw.
    fn outstanding_log_density(&self, nat: &[f64], i: usize, k: usize, y: f64) -> Option<f64> {
        let (th, (os, _), _) = self.closed_cell(nat, i, k)?;
        let mean = self.premium[i] * th[RLR] * os;
        Some(self.response_log_density(y, mean, nat[4], i))
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{
        INCURRED, PAID, PREMIUM, incurred, model, model_2, points, quick, triangle,
    };
    use super::super::{Compartmental, CompartmentalPriors};
    use super::*;
    use crate::backtest::diagonal_backtest;
    use crate::triangle::Long;
    use crate::{DevelopmentColumn, Month};
    use prospicio_core::{Grain, Lag, Period};
    use prospicio_prob::Distribution;

    fn candidate(model: Compartmental, tri: &Triangle) -> CompartmentalCandidate<'_> {
        CompartmentalCandidate::new(model, tri, "paid", incurred(), "premium")
    }

    /// The mass, mean and sd of a density on a grid (trapezoid rule).
    fn moments(density: impl Fn(f64) -> f64, grid: &[f64]) -> (f64, f64, f64) {
        let f: Vec<f64> = grid.iter().map(|&y| density(y)).collect();
        let integral = |g: &dyn Fn(usize) -> f64| -> f64 {
            (1..grid.len())
                .map(|j| 0.5 * (g(j - 1) + g(j)) * (grid[j] - grid[j - 1]))
                .sum()
        };
        let mass = integral(&|j| f[j]);
        let mean = integral(&|j| grid[j] * f[j]) / mass;
        let var = integral(&|j| (grid[j] - mean).powi(2) * f[j]) / mass;
        (mass, mean, var.sqrt())
    }

    /// `n` simulations of one future cell of `m` given a draw.
    fn simulated(m: &Model, nat: &[f64], future: &Future, n: usize) -> Vec<f64> {
        let mut rng = StreamRng::new(11, 0);
        let width = future.paid.len() + future.outstanding.len();
        (0..n)
            .map(|_| {
                let mut row = vec![0.0; width];
                m.simulate(nat, future, &mut rng, &mut row);
                row[0]
            })
            .collect()
    }

    fn sample_moments(x: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var.sqrt())
    }

    /// The closed forms are the densities of what `simulate` draws, for a
    /// paid increment after an origin's latest paid and for an outstanding,
    /// in model 1 (normal, cumulative paid) and model 2 (log-normal,
    /// incremental paid, development effects by cell).
    #[test]
    fn the_closed_forms_are_the_densities_of_the_simulation() {
        for (m, log_normal) in [(model(false), false), (model_2(false), true)] {
            let nat = m.natural(&points(&m)[0]);
            // Origin 1991 (index 3) has paid to age 6; its next age is 7.
            let (i, k) = (3, 6);
            let latest = m.latest_paid[i];
            assert_eq!(latest, PAID[3][5]);
            for paid in [true, false] {
                let future = Future {
                    paid: if paid { vec![(i, k)] } else { vec![] },
                    outstanding: if paid { vec![] } else { vec![(i, k)] },
                    last_paid: vec![None; 10],
                };
                let draws = simulated(&m, &nat, &future, 40_000);
                let draws: Vec<f64> = if paid {
                    draws.iter().map(|c| c - latest).collect()
                } else {
                    draws
                };
                let (mean, sd) = sample_moments(&draws);
                let density = |y: f64| {
                    if paid {
                        m.paid_log_density(&nat, i, k, latest, y)
                    } else {
                        m.outstanding_log_density(&nat, i, k, y)
                    }
                    .unwrap()
                    .exp()
                };
                let grid: Vec<f64> = if log_normal {
                    let (lo, hi) = (
                        draws.iter().copied().fold(f64::INFINITY, f64::min) / 4.0,
                        draws.iter().copied().fold(f64::NEG_INFINITY, f64::max) * 4.0,
                    );
                    (0..=40_000)
                        .map(|j| lo * (hi / lo).powf(j as f64 / 40_000.0))
                        .collect()
                } else {
                    (0..=20_000)
                        .map(|j| mean - 10.0 * sd + 20.0 * sd * j as f64 / 20_000.0)
                        .collect()
                };
                let (mass, got_mean, got_sd) = moments(density, &grid);
                let what = format!("log_normal {log_normal}, paid {paid}");
                assert!((mass - 1.0).abs() < 2e-3, "{what}: mass {mass}");
                let se = sd / (draws.len() as f64).sqrt();
                assert!(
                    (got_mean - mean).abs() < 4.0 * se,
                    "{what}: {got_mean} vs {mean}"
                );
                assert!((got_sd / sd - 1.0).abs() < 0.03, "{what}: {got_sd} vs {sd}");
            }
        }
    }

    #[test]
    fn a_cell_without_an_observed_origin_has_no_closed_form() {
        let m = model(false);
        let nat = m.natural(&points(&m)[0]);
        // 1997 (index 9) has no observed cell.
        assert!(m.outstanding_log_density(&nat, 9, 0, 100.0).is_none());
        assert!(m.paid_log_density(&nat, 9, 0, 0.0, 100.0).is_none());
        assert!(m.paid_log_density(&nat, 3, 6, PAID[3][5], 100.0).is_some());
    }

    fn rebuilt_triangle(change: impl Fn(usize, usize, f64) -> f64) -> Triangle {
        let (mut origin, mut ages) = (vec![], vec![]);
        let (mut paid, mut inc, mut prem) = (vec![], vec![], vec![]);
        for (w, row) in PAID.iter().enumerate() {
            for (d, &p) in row.iter().enumerate() {
                origin.push(Month::january(1988 + w as i32));
                ages.push(12 * (d as Lag + 1));
                paid.push(change(w, d, p));
                inc.push(INCURRED[w][d]);
                prem.push(PREMIUM[w]);
            }
        }
        Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&ages),
            values: &[("paid", &paid), ("incurred", &inc), ("premium", &prem)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap()
    }

    /// The log-normal likelihood drops the first row whose increment is not
    /// positive and every later row of its origin; the normal keeps all.
    #[test]
    fn rows_the_likelihood_rejects_are_not_fitted() {
        // Origin 1990 (index 2) falls from 35406 to 30000 at age 48.
        let tri = rebuilt_triangle(|w, d, p| if (w, d) == (2, 3) { 30_000.0 } else { p });
        let cells = TriangleFrame::new(&tri, "paid", None).unwrap();
        let train = cells.observed();
        let one = candidate(quick(1), &tri);
        assert_eq!(one.fit_rows(&cells, train), train);
        let two = candidate(
            Compartmental::model_2(CompartmentalPriors::gesmann_morris_2()),
            &tri,
        );
        let kept = two.fit_rows(&cells, train);
        // 1990 has 7 observed ages; it keeps its first 3 rows.
        assert_eq!(kept.len(), train.len() - 4);
        let dropped: Vec<usize> = train
            .iter()
            .copied()
            .filter(|r| !kept.contains(r))
            .collect();
        assert!(
            dropped
                .iter()
                .all(|&r| cells.origin_of(r) == 2 && cells.development_of(r) >= 3)
        );
        let (data, _) = two.prepare(&cells, train).unwrap();
        assert_eq!(data.latest_paid[2], Some((2, PAID[2][2])));
        assert_eq!(data.paid.len(), kept.len());
        assert_eq!(data.latest_paid[0], Some((8, PAID[0][8])));
        assert_eq!(data.outstanding.len(), 45);
    }

    /// The settings are read through accessors and fixed at construction, so
    /// the cache (keyed on seed and rows) cannot return a fit made under
    /// other settings: a second candidate has its own, empty cache.
    #[test]
    fn a_new_candidate_starts_with_no_cached_fit() {
        let tri = triangle(false);
        let a = candidate(quick(1), &tri);
        assert_eq!(a.paid(), "paid");
        assert_eq!(a.premium(), "premium");
        assert!(std::ptr::eq(a.triangle(), &tri));
        let b = candidate(quick(1), &tri);
        assert_eq!(a.fits.lock().unwrap().len(), 0);
        assert_eq!(b.fits.lock().unwrap().len(), 0);
    }

    /// The fit never sees the held-out diagonal: its paid is rebuilt from
    /// the training rows and its outstanding is that of the training cells.
    #[test]
    fn the_held_out_diagonal_is_withheld() {
        let tri = triangle(true);
        let cells = TriangleFrame::new(&tri, "paid", None).unwrap();
        let split = cells.diagonal_splits(1).unwrap().remove(0);
        let c = candidate(quick(1), &tri);
        let (data, full) = c.prepare(&cells, &split.train).unwrap();
        // 1988 to 1995 have training rows, 1996 only its held-out age 12.
        assert_eq!(data.paid.len(), 36);
        assert_eq!(data.outstanding.len(), 36);
        assert_eq!(full.len(), 45);
        assert_eq!(data.latest_paid[0], Some((7, PAID[0][7])));
        assert_eq!(data.latest_paid[8], None);
        for &r in &split.test {
            let cell = c.locate(&cells, &data, r).unwrap();
            assert!(!data.outstanding.iter().any(|&(i, k, _)| (i, k) == cell));
            assert!(!data.paid.iter().any(|&(i, k, _)| (i, k) == cell));
        }
        // A gap in an origin is an error.
        let gap: Vec<usize> = split.train.iter().copied().filter(|&r| r != 1).collect();
        assert!(c.prepare(&cells, &gap).is_err());
    }

    /// Fitting on every observed row and forecasting the rest of the frame
    /// gives the fit's reserves: each origin's increments sum to its paid
    /// at the frame's last age less its latest paid, on the same posterior.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_increments_sum_to_the_fits_reserve() {
        let tri = triangle(false);
        let cells = TriangleFrame::new(&tri, "paid", None).unwrap();
        let m = quick(3);
        let fit = m.fit(&tri, "paid", incurred(), "premium").unwrap();
        let c = candidate(m, &tri);
        let test: Vec<usize> = cells
            .future()
            .iter()
            .copied()
            .filter(|&r| (1..9).contains(&cells.origin_of(r)))
            .collect();
        let f = c.forecast(&cells, cells.observed(), &test, 500, 3).unwrap();
        let (n, n_sims) = (test.len(), f.distribution.n_sims());
        let draws = f.distribution.draw_matrix();
        for o in 1..9 {
            let sums: Vec<f64> = (0..n_sims)
                .map(|s| {
                    test.iter()
                        .enumerate()
                        .filter(|&(_, &r)| cells.origin_of(r) == o)
                        .map(|(j, _)| draws[s * n + j])
                        .sum()
                })
                .collect();
            // Paid at the frame's last age (108 months) less the latest.
            let key = vec![Period::year(1988 + o as i32).into(), KeyValue::Int(108)];
            let at_last = fit.paid.marginal(&key).unwrap();
            let want = at_last.mean() - fit.latest_paid[o];
            let (mean, sd) = sample_moments(&sums);
            let tol = 5.0 * (sd * sd + at_last.variance()).sqrt() / (n_sims as f64).sqrt();
            assert!(
                (mean - want).abs() < tol,
                "origin {o}: {mean} vs {want} +- {tol}"
            );
        }
    }

    /// The closed-form log density of the backtest's rows is that of the
    /// simulation they are drawn from, and the paid and outstanding tables
    /// share one fit.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "samples with NUTS; needs --features slow-tests"
    )]
    fn the_backtest_scores_paid_and_outstanding_on_one_fit() {
        let tri = triangle(false);
        let cells = TriangleFrame::new(&tri, "paid", None).unwrap();
        let c = candidate(quick(5), &tri);
        let bt = diagonal_backtest(&cells, &[&c], 1, 500, 5, 0.9).unwrap();
        assert_eq!(bt.models(), ["compartmental_gm1"]);
        // 1996 has no training origin and 1988 no training row at 108 months.
        assert_eq!(bt.excluded(), [2]);
        assert_eq!(bt.scored_rows()[0].len(), 7);
        let os = c.outstanding_backtest(&cells, 1, 500, 5, 0.9).unwrap();
        assert_eq!(os.models(), ["compartmental_gm1_outstanding"]);
        assert_eq!(os.scored_rows(), bt.scored_rows());
        assert_eq!(os.excluded(), bt.excluded());
        // One NUTS run served both tables.
        assert_eq!(c.fits.lock().unwrap().len(), 1);
        assert_eq!(os.diagonals(), bt.diagonals());
        assert_eq!(os.diagnostics(0), bt.diagnostics(0));
        for t in [&bt, &os] {
            let d = t.diagnostics(0)[0]
                .as_ref()
                .expect("a compartmental fit reports diagnostics");
            assert!(d.min_ess_bulk > 0.0);
            for metric in 0..METRICS.len() {
                assert!(t.mean(0, metric).is_finite(), "{metric}");
            }
            let lpd = t.log_densities()[0].as_ref().unwrap();
            assert_eq!(lpd.len(), 7);
            assert!(lpd.iter().all(|v| v.is_finite()));
            assert!(t.mean(0, 1) >= 0.0 && t.mean(0, 1) <= 1.0);
        }

        // The log density of a paid row against its simulation: the
        // predictive density integrates to one and has the draws' mean.
        let split = cells.diagonal_splits(1).unwrap().remove(0);
        let fit = c.train_fit(&cells, &split.train, 5).unwrap();
        let rows = bt.scored_rows()[0].clone();
        let f = c.forecast(&cells, &split.train, &rows, 500, 5).unwrap();
        let n = rows.len();
        let y = cells.response_of(&rows);
        let lpd = f.log_density.as_ref().unwrap();
        for j in [0, 3, 6] {
            let (i, k) = c.locate(&cells, &fit.data, rows[j]).unwrap();
            let draws: Vec<f64> = f
                .distribution
                .draw_matrix()
                .chunks(n)
                .map(|r| r[j])
                .collect();
            let (mean, sd) = sample_moments(&draws);
            let density = |x: f64| fit.log_density_at(i, k, x, true).unwrap().exp();
            let grid: Vec<f64> = (0..=4_000)
                .map(|g| mean - 12.0 * sd + 24.0 * sd * g as f64 / 4_000.0)
                .collect();
            let (mass, got_mean, _) = moments(density, &grid);
            assert!((mass - 1.0).abs() < 2e-2, "row {j}: mass {mass}");
            let se = sd / (draws.len() as f64).sqrt();
            assert!(
                (got_mean - mean).abs() < 5.0 * se,
                "row {j}: {got_mean} vs {mean}"
            );
            let want = fit.log_density_at(i, k, y[j], true).unwrap();
            assert!((lpd[j] - want).abs() < 1e-9);
        }
    }
}
