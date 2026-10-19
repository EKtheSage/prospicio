//! Calendar-diagonal backtest for any model of the cells of a triangle.
//!
//! Reserving forecasts the next calendar period, so a model is scored by
//! refitting it with the latest diagonals held out, one at a time, and
//! forecasting each held-out diagonal (`docs/design/models.md`, "Fitting a
//! triangle with several models"). Any model takes part through
//! [`TriangleModel`]; [`GlmCandidate`] wraps an `prospicio-glm` GLM.
//!
//! The scores ([`METRICS`]) are the mean cell CRPS, the coverage of a central
//! interval, actual vs expected on the diagonal total, the CRPS of the
//! diagonal total from the joint draws, the mean held-out log density and
//! the percentile of the actual total among the simulated totals. Each
//! split also reports the sampler diagnostics of the fits behind it
//! ([`Backtest::diagnostics`]); a fit with warnings is still scored. The pointwise held-out log densities feed
//! [`prospicio_models::stack::stacking_weights`].

use prospicio_core::{Error as CoreError, StreamRng};
use prospicio_glm::{Glm, GlmFit, ParameterDraws};
use prospicio_math::linalg::{cholesky, lower_mul};
use prospicio_math::special::norm_quantile;
use prospicio_models::metrics::{coverage, crps};
use prospicio_models::{Design, Fitted, Link, Model, Terms};
use prospicio_prob::PredictiveDistribution;

use crate::diagnostics::Diagnostics;
use crate::error::{Error, Result};
use crate::frame::TriangleFrame;

/// A model's forecast of some cells: what [`diagonal_backtest`] scores.
#[derive(Debug, Clone)]
pub struct CellForecast {
    /// Predicted mean of each test row, in the order given.
    pub mean: Vec<f64>,
    /// Joint predictive distribution over the test rows: component `j` is
    /// test row `j`.
    pub distribution: PredictiveDistribution,
    /// Log predictive density of each test row's observed response, per
    /// unit of the response, if the model gives one.
    pub log_density: Option<Vec<f64>>,
    /// The sampler diagnostics of the fit behind the forecast: `None` for a
    /// model with no sampler (the GLM). A fit with warnings is still
    /// forecast and scored; [`Backtest::warnings`] lists them.
    pub diagnostics: Option<Diagnostics>,
}

/// A model of the cells of a triangle that can be refitted and forecast:
/// a GLM, a GAM, a Bayesian model, a Chain Ladder wrapper.
pub trait TriangleModel: Sync {
    /// The model's name in the backtest.
    fn name(&self) -> &str;

    /// Fits on the `train` rows of `cells` and forecasts the `test` rows
    /// (row numbers index `cells`) with `n_sims` joint draws from `seed`.
    fn forecast(
        &self,
        cells: &TriangleFrame,
        train: &[usize],
        test: &[usize],
        n_sims: usize,
        seed: u64,
    ) -> Result<CellForecast>;

    /// The `train` rows the model actually fits on: all of them unless the
    /// model drops some (a Poisson GLM drops negative increments).
    /// [`diagonal_backtest`] scores only held-out rows whose origin and
    /// development levels appear among every model's fit rows.
    fn fit_rows(&self, cells: &TriangleFrame, train: &[usize]) -> Vec<usize> {
        let _ = cells;
        train.to_vec()
    }
}

/// A GLM on the features of [`TriangleFrame`]: the ODP model is a
/// quasi-Poisson GLM with intercept, origin and development factors.
///
/// The frame is coded with factor levels learned on the train and test
/// rows together, the GLM is fitted on the train rows it accepts
/// ([`Glm::accepts`]: the quasi-Poisson takes negative increments) and the
/// test rows get [`GlmFit::predict_distribution_with`], with mean-preserving
/// parameter draws under a log or identity link so each cell's simulated
/// mean is its fitted mean.
///
/// The log density of each test response is a log predictive density with
/// parameter uncertainty: the log of the mean, over `n_sims` draws of the
/// coefficients from their normal approximation, of the response density
/// at the fitted dispersion ([`prospicio_models::Family::log_density`], which
/// scores the over-dispersed Poisson by a normalized density in `y`).
/// Without the averaging a model with many poorly identified parameters
/// (an origin factor for the newest origins) is scored as if they were
/// known.
///
/// ```
/// use prospicio_glm::Glm;
/// use prospicio_models::Terms;
/// use prospicio_reserving::{GlmCandidate, TriangleModel};
///
/// let odp = GlmCandidate {
///     name: "odp".into(),
///     terms: Terms::new().intercept().factor("origin").factor("development"),
///     glm: Glm::over_dispersed_poisson(),
/// };
/// assert_eq!(odp.name(), "odp");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GlmCandidate {
    pub name: String,
    pub terms: Terms,
    pub glm: Glm,
}

impl TriangleModel for GlmCandidate {
    fn name(&self) -> &str {
        &self.name
    }

    fn forecast(
        &self,
        cells: &TriangleFrame,
        train: &[usize],
        test: &[usize],
        n_sims: usize,
        seed: u64,
    ) -> Result<CellForecast> {
        let train = self.fit_rows(cells, train);
        let both: Vec<usize> = train.iter().chain(test).copied().collect();
        let coding = self.terms.fit(&cells.select(&both)?)?;
        let train_design = coding.design(&cells.select(&train)?)?;
        let test_design = coding.design(&cells.select(test)?)?;
        let fit = self.glm.fit(&train_design, &cells.response_of(&train))?;
        let mean = fit.predict(&test_design)?;
        // Mean-preserving draws keep each cell's simulated mean at its fitted
        // mean under a log or identity link (plain normal draws through a log
        // link overstate it by exp(x'Σx/2)).
        let parameters = match self.glm.link {
            Link::Log | Link::Identity => ParameterDraws::MeanPreserving,
            _ => ParameterDraws::Normal,
        };
        let distribution = fit.predict_distribution_with(&test_design, n_sims, seed, parameters)?;
        let log_density = parameter_averaged_log_density(
            &fit,
            &test_design,
            &cells.response_of(test),
            n_sims,
            seed,
        )?;
        Ok(CellForecast {
            mean,
            distribution,
            log_density: Some(log_density),
            diagnostics: None,
        })
    }

    /// The `train` rows whose response the GLM accepts ([`Glm::accepts`]):
    /// the family's range, and any finite value for the quasi-Poisson.
    fn fit_rows(&self, cells: &TriangleFrame, train: &[usize]) -> Vec<usize> {
        let response = cells.response();
        train
            .iter()
            .copied()
            .filter(|&r| self.glm.accepts(response[r]))
            .collect()
    }
}

/// Log of the mean, over `n_sims` draws of `β ~ N(β̂, Cov β̂)` (stream `s`
/// of `seed` for draw `s`), of each row's response density at the fitted
/// dispersion: the log predictive density with parameter uncertainty.
fn parameter_averaged_log_density(
    fit: &GlmFit,
    design: &Design,
    y: &[f64],
    n_sims: usize,
    seed: u64,
) -> Result<Vec<f64>> {
    let (family, link) = (fit.spec().family, fit.spec().link);
    let phi = fit.dispersion();
    let p = fit.coefficients().len();
    let factor = cholesky(&fit.covariance(), p)
        .ok_or_else(|| data("the coefficient covariance is singular".into()))?;
    let density = |y: f64, mu: f64| -> prospicio_core::Result<f64> {
        if !family.valid_mu(mu) {
            Ok(f64::NEG_INFINITY)
        } else {
            family.log_density(y, mu, phi, 1.0)
        }
    };
    // Row-major: row j's log densities over the draws.
    let mut ln_f = vec![0.0; y.len() * n_sims];
    let (mut z, mut shift) = (vec![0.0; p], vec![0.0; p]);
    for s in 0..n_sims {
        let mut rng = StreamRng::new(seed, s as u64);
        for zi in &mut z {
            *zi = norm_quantile(rng.next_open01());
        }
        lower_mul(&factor, &z, &mut shift);
        let beta: Vec<f64> = fit
            .coefficients()
            .iter()
            .zip(&shift)
            .map(|(b, d)| b + d)
            .collect();
        let eta = design.linear_predictor(&beta);
        for (j, (&yj, &e)) in y.iter().zip(&eta).enumerate() {
            ln_f[j * n_sims + s] = density(yj, link.inverse(e))?;
        }
    }
    Ok(ln_f
        .chunks_exact(n_sims)
        .map(|l| {
            let m = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if m == f64::NEG_INFINITY {
                return m;
            }
            m + (l.iter().map(|v| (v - m).exp()).sum::<f64>() / n_sims as f64).ln()
        })
        .collect())
}

/// A model under a name of the caller's: [`diagonal_backtest`] tells models
/// apart by [`TriangleModel::name`], so two settings of the same kind of
/// model (two Bayesian ODP priors, say) are told apart by wrapping each in
/// a `NamedModel`.
///
/// The model is owned and boxed (`Send`; [`TriangleModel`] requires
/// `Sync`) so that a binding can hold a list of models of different kinds
/// without a lifetime. [`forecast`](TriangleModel::forecast) and
/// [`fit_rows`](TriangleModel::fit_rows) are the wrapped model's.
///
/// ```
/// use prospicio_glm::Glm;
/// use prospicio_models::Terms;
/// use prospicio_reserving::{GlmCandidate, NamedModel, TriangleModel};
///
/// let odp = GlmCandidate {
///     name: "odp".into(),
///     terms: Terms::new().intercept().factor("origin").factor("development"),
///     glm: Glm::over_dispersed_poisson(),
/// };
/// let renamed = NamedModel::new("odp, wide", odp);
/// assert_eq!(renamed.name(), "odp, wide");
/// ```
pub struct NamedModel {
    /// The name in the backtest.
    pub name: String,
    /// The wrapped model.
    pub model: Box<dyn TriangleModel + Send>,
}

impl NamedModel {
    /// Wraps `model` under `name`.
    pub fn new<M: TriangleModel + Send + 'static>(name: impl Into<String>, model: M) -> Self {
        Self {
            name: name.into(),
            model: Box::new(model),
        }
    }
}

impl std::fmt::Debug for NamedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedModel")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl TriangleModel for NamedModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn forecast(
        &self,
        cells: &TriangleFrame,
        train: &[usize],
        test: &[usize],
        n_sims: usize,
        seed: u64,
    ) -> Result<CellForecast> {
        self.model.forecast(cells, train, test, n_sims, seed)
    }

    fn fit_rows(&self, cells: &TriangleFrame, train: &[usize]) -> Vec<usize> {
        self.model.fit_rows(cells, train)
    }
}

/// Metric names of a [`Backtest`], in the order of its metric indices:
/// `crps`, `coverage`, `actual_vs_expected`, `total_crps`, then
/// `mean_log_density` (the mean over the split's scored rows of the model's
/// held-out log density, NaN for a model that gives none) and
/// `total_percentile` (the share of the simulated diagonal totals, the
/// joint draws summed per simulation, at or below the actual total: the
/// probability integral transform of the diagonal, uniform over many
/// triangles for a calibrated model).
pub const METRICS: [&str; 6] = [
    "crps",
    "coverage",
    "actual_vs_expected",
    "total_crps",
    "mean_log_density",
    "total_percentile",
];

/// The scores from [`diagonal_backtest`]: every model on every metric and
/// held-out diagonal, and the pointwise held-out log densities.
#[derive(Debug, Clone, PartialEq)]
pub struct Backtest {
    models: Vec<String>,
    n_splits: usize,
    /// Model-major, then metric, then split.
    scores: Vec<f64>,
    excluded: Vec<usize>,
    scored: Vec<Vec<usize>>,
    log_densities: Vec<Option<Vec<f64>>>,
    /// Model-major, then split.
    diagnostics: Vec<Option<Diagnostics>>,
    diagonals: Vec<i64>,
}

impl Backtest {
    /// Model names, in the order given.
    pub fn models(&self) -> &[String] {
        &self.models
    }

    /// Metric names: [`METRICS`].
    pub fn metrics(&self) -> &[&'static str] {
        &METRICS
    }

    /// Number of splits (held-out diagonals), oldest first.
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Scores of `model` on `metric`, one per split (indices into
    /// [`models`](Self::models) and [`metrics`](Self::metrics)).
    pub fn split_scores(&self, model: usize, metric: usize) -> &[f64] {
        let k = self.n_splits;
        let start = (model * METRICS.len() + metric) * k;
        &self.scores[start..start + k]
    }

    /// Mean score over the splits.
    pub fn mean(&self, model: usize, metric: usize) -> f64 {
        let s = self.split_scores(model, metric);
        s.iter().sum::<f64>() / s.len() as f64
    }

    /// Number of held-out rows left out of scoring in each split because
    /// their origin or development level has no training row that every
    /// model fits on.
    pub fn excluded(&self) -> &[usize] {
        &self.excluded
    }

    /// The rows scored in each split, in the order of the log densities.
    pub fn scored_rows(&self) -> &[Vec<usize>] {
        &self.scored
    }

    /// Each model's held-out log density of every scored row, splits in
    /// order (`None` for a model that gives none). Every model's values are
    /// for the same rows in the same order, so the models that give one are
    /// the input of [`prospicio_models::stack::stacking_weights`].
    pub fn log_densities(&self) -> &[Option<Vec<f64>>] {
        &self.log_densities
    }

    /// The sampler diagnostics of `model`'s fit in each split, oldest
    /// split first (`None` for a model with no sampler). A fit with
    /// warnings is scored like any other.
    pub fn diagnostics(&self, model: usize) -> &[Option<Diagnostics>] {
        let k = self.n_splits;
        &self.diagnostics[model * k..(model + 1) * k]
    }

    /// Every diagnostics warning as `(model, split, message)`, indices into
    /// [`models`](Self::models) and the splits, in model then split order.
    pub fn warnings(&self) -> Vec<(usize, usize, String)> {
        let k = self.n_splits;
        self.diagnostics
            .iter()
            .enumerate()
            .flat_map(|(i, d)| {
                d.iter()
                    .flat_map(|d| d.warnings.iter())
                    .map(move |w| (i / k, i % k, w.clone()))
            })
            .collect()
    }

    /// The calendar period index of each split's held-out diagonal
    /// ([`TriangleFrame::calendar_of`]), oldest split first: a label for
    /// the split.
    pub fn diagonals(&self) -> &[i64] {
        &self.diagonals
    }
}

/// Backtests `models` on the latest `n_diagonals` calendar diagonals of
/// `cells` ([`TriangleFrame::diagonal_splits`]): for each, every model is
/// fitted on the earlier diagonals and forecasts the held-out one with
/// `n_sims` joint draws from `seed` (the same seed for every model and
/// split, so the models share their random numbers).
///
/// A held-out row whose origin or development level has no training row
/// that every model fits on ([`TriangleModel::fit_rows`]) cannot be
/// forecast by a model with origin and development effects: the newest
/// origin and the oldest age on the diagonal, or an age whose only
/// training cell is a negative increment a Poisson GLM drops. It is left
/// out for every model, and [`Backtest::excluded`] counts it.
///
/// Scores per model and split ([`METRICS`]): the mean CRPS of the cells,
/// the share of cells inside the central `interval` of their draws, actual
/// over expected on the diagonal total `Σy / Σμ`, the CRPS of the
/// diagonal total from the draws summed per simulation, the mean log density
/// of the scored rows and the share of the simulated totals at or below the
/// actual one.
///
/// Two models with the same [`TriangleModel::name`] are an error, checked
/// before any fit ([`NamedModel`] renames one). A model that fails to fit is
/// an error naming it; a fit that merely has diagnostics warnings is scored
/// and reported in [`Backtest::warnings`].
///
/// ```
/// use prospicio_glm::Glm;
/// use prospicio_models::Terms;
/// use prospicio_reserving::{
///     DevelopmentColumn, GlmCandidate, Grain, Long, Month, Triangle, TriangleFrame,
///     diagonal_backtest,
/// };
///
/// let origin = [2019, 2019, 2019, 2019, 2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
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
/// let cells = TriangleFrame::new(&tri, "paid", None)?;
/// let odp = GlmCandidate {
///     name: "odp".into(),
///     terms: Terms::new().intercept().factor("origin").factor("development"),
///     glm: Glm::over_dispersed_poisson(),
/// };
/// let bt = diagonal_backtest(&cells, &[&odp], 1, 1000, 7, 0.9)?;
/// // The 2022 diagonal: 2022 at 12 and 2019 at 48 have no training level.
/// assert_eq!(bt.excluded(), [2]);
/// assert_eq!(bt.scored_rows()[0].len(), 2);
/// # Ok::<(), prospicio_reserving::Error>(())
/// ```
pub fn diagonal_backtest(
    cells: &TriangleFrame,
    models: &[&dyn TriangleModel],
    n_diagonals: usize,
    n_sims: usize,
    seed: u64,
    interval: f64,
) -> Result<Backtest> {
    if models.is_empty() {
        return Err(data("the backtest needs at least one model".into()));
    }
    for (i, m) in models.iter().enumerate() {
        if models[..i].iter().any(|o| o.name() == m.name()) {
            return Err(data(format!(
                "two models are named {:?}: name each model once (NamedModel renames one)",
                m.name()
            )));
        }
    }
    if n_sims == 0 {
        return Err(data("the backtest needs at least one simulation".into()));
    }
    if !(interval > 0.0 && interval < 1.0) {
        return Err(Error::Core(CoreError::InvalidParameter {
            name: "interval",
            value: interval,
            reason: "must be in (0, 1)",
        }));
    }
    let splits = cells.diagonal_splits(n_diagonals)?;
    let n_splits = splits.len();
    let mut excluded = Vec::with_capacity(n_splits);
    let mut scored = Vec::with_capacity(n_splits);
    for s in &splits {
        let fit_rows: Vec<Vec<usize>> =
            models.iter().map(|m| m.fit_rows(cells, &s.train)).collect();
        let seen = |rows: &[usize], r: usize| {
            rows.iter()
                .any(|&t| cells.origin_of(t) == cells.origin_of(r))
                && rows
                    .iter()
                    .any(|&t| cells.development_of(t) == cells.development_of(r))
        };
        let keep: Vec<usize> = s
            .test
            .iter()
            .copied()
            .filter(|&r| fit_rows.iter().all(|rows| seen(rows, r)))
            .collect();
        if keep.is_empty() {
            return Err(data(format!(
                "no held-out row of diagonal {} has a training origin and development",
                cells.calendar_of(s.test[0])
            )));
        }
        excluded.push(s.test.len() - keep.len());
        scored.push(keep);
    }

    let n_metrics = METRICS.len();
    let mut scores = vec![f64::NAN; models.len() * n_metrics * n_splits];
    let mut log_densities: Vec<Option<Vec<f64>>> = vec![Some(Vec::new()); models.len()];
    let mut diagnostics: Vec<Option<Diagnostics>> = vec![None; models.len() * n_splits];
    for (k, (s, test)) in splits.iter().zip(&scored).enumerate() {
        let y = cells.response_of(test);
        for (m, model) in models.iter().enumerate() {
            let named = |e: Error| data(format!("{}: {e}", model.name()));
            let f = model
                .forecast(cells, &s.train, test, n_sims, seed)
                .map_err(named)?;
            let split_scores = score(&f, &y, interval).map_err(named)?;
            for (metric, value) in split_scores.into_iter().enumerate() {
                scores[(m * n_metrics + metric) * n_splits + k] = value;
            }
            diagnostics[m * n_splits + k] = f.diagnostics;
            log_densities[m] = match (log_densities[m].take(), f.log_density) {
                (Some(mut all), Some(lpd)) => {
                    if lpd.len() != test.len() {
                        return Err(named(data(format!(
                            "{} log densities for {} test rows",
                            lpd.len(),
                            test.len()
                        ))));
                    }
                    all.extend(lpd);
                    Some(all)
                }
                _ => None,
            };
        }
    }
    Ok(Backtest {
        models: models.iter().map(|m| m.name().to_string()).collect(),
        n_splits,
        scores,
        excluded,
        scored,
        log_densities,
        diagnostics,
        diagonals: splits
            .iter()
            .map(|s| cells.calendar_of(s.test[0]))
            .collect(),
    })
}

/// The [`METRICS`] of one forecast against the outcomes `y`.
fn score(f: &CellForecast, y: &[f64], interval: f64) -> Result<[f64; 6]> {
    let n = y.len();
    let dist = &f.distribution;
    if f.mean.len() != n || dist.n_components() != n {
        return Err(data(format!(
            "a forecast of {n} test rows has {} means and {} components",
            f.mean.len(),
            dist.n_components()
        )));
    }
    let n_sims = dist.n_sims();
    let draws = dist.draw_matrix();
    let column = |j: usize| -> Vec<f64> { (0..n_sims).map(|i| draws[i * n + j]).collect() };
    let (p_lo, p_hi) = ((1.0 - interval) / 2.0, (1.0 + interval) / 2.0);
    let (mut cell_crps, mut lo, mut hi) = (0.0, Vec::with_capacity(n), Vec::with_capacity(n));
    for (j, &yj) in y.iter().enumerate() {
        let mut d = column(j);
        cell_crps += crps(&d, yj)?;
        d.sort_by(f64::total_cmp);
        lo.push(quantile(&d, p_lo));
        hi.push(quantile(&d, p_hi));
    }
    let totals: Vec<f64> = (0..n_sims)
        .map(|i| draws[i * n..(i + 1) * n].iter().sum())
        .collect();
    let actual: f64 = y.iter().sum();
    let mean_log_density = match &f.log_density {
        Some(l) if l.len() == n => l.iter().sum::<f64>() / n as f64,
        _ => f64::NAN,
    };
    let total_percentile = totals.iter().filter(|&&t| t <= actual).count() as f64 / n_sims as f64;
    Ok([
        cell_crps / n as f64,
        coverage(y, &lo, &hi)?,
        actual / f.mean.iter().sum::<f64>(),
        crps(&totals, actual)?,
        mean_log_density,
        total_percentile,
    ])
}

/// Quantile `p` of sorted draws, interpolating between order statistics
/// (R's type 7).
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * p;
    let (i, frac) = (h.floor() as usize, h - h.floor());
    match sorted.get(i + 1) {
        Some(next) => sorted[i] + frac * (next - sorted[i]),
        None => sorted[i],
    }
}

fn data(message: String) -> Error {
    Error::Core(CoreError::Data(message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangle::tests::raa;
    use prospicio_models::Link;

    #[test]
    fn quantile_type_7() {
        let d = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile(&d, 0.0), 1.0);
        assert_eq!(quantile(&d, 1.0), 4.0);
        assert!((quantile(&d, 0.5) - 2.5).abs() < 1e-15);
        assert!((quantile(&d, 0.25) - 1.75).abs() < 1e-15);
    }

    struct NoDensity;

    impl TriangleModel for NoDensity {
        fn name(&self) -> &str {
            "no density"
        }

        fn forecast(
            &self,
            cells: &TriangleFrame,
            train: &[usize],
            test: &[usize],
            n_sims: usize,
            seed: u64,
        ) -> Result<CellForecast> {
            let mut f = GlmCandidate {
                name: String::new(),
                terms: Terms::new().intercept().factor("development"),
                glm: Glm::new(prospicio_models::Family::Poisson, Link::Log)
                    .dispersion(prospicio_glm::Dispersion::Pearson),
            }
            .forecast(cells, train, test, n_sims, seed)?;
            f.log_density = None;
            Ok(f)
        }
    }

    #[test]
    fn a_model_without_densities_gives_none() {
        let cells = TriangleFrame::new(&raa(), "values", None).unwrap();
        let odp = GlmCandidate {
            name: "odp".into(),
            terms: Terms::new()
                .intercept()
                .factor("origin")
                .factor("development"),
            glm: Glm::over_dispersed_poisson(),
        };
        let bt = diagonal_backtest(&cells, &[&odp, &NoDensity], 2, 200, 3, 0.8).unwrap();
        assert_eq!(bt.log_densities()[0].as_ref().unwrap().len(), 15);
        assert!(bt.log_densities()[1].is_none());
        assert_eq!(bt.models(), ["odp", "no density"]);
    }

    /// A triangle of yearly cumulative rows, oldest origin first.
    fn small(rows: &[&[f64]]) -> TriangleFrame {
        use crate::{DevelopmentColumn, Grain, Long, Month, Triangle};
        let (mut origin, mut age, mut value) = (vec![], vec![], vec![]);
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                origin.push(Month::january(2018 + i as i32));
                age.push(12 * (j as u32 + 1));
                value.push(v);
            }
        }
        let tri = Triangle::from_long(&Long {
            keys: &[],
            origin: &origin,
            development: DevelopmentColumn::Age(&age),
            values: &[("paid", &value)],
            origin_grain: Grain::Year,
            development_grain: Grain::Year,
            cumulative: true,
        })
        .unwrap();
        TriangleFrame::new(&tri, "paid", None).unwrap()
    }

    /// Four draws per test row, off-centre from the outcome and alternating in
    /// direction, so that the draws of neighbouring rows cancel in the
    /// total; the mean is the outcome plus one, and the log density is the
    /// row number.
    struct Fixed;

    impl TriangleModel for Fixed {
        fn name(&self) -> &str {
            "fixed"
        }

        fn forecast(
            &self,
            cells: &TriangleFrame,
            _: &[usize],
            test: &[usize],
            _: usize,
            _: u64,
        ) -> Result<CellForecast> {
            let y = cells.response_of(test);
            let n = y.len();
            let mut draws = vec![0.0; 4 * n];
            for i in 0..4 {
                for (j, yj) in y.iter().enumerate() {
                    let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
                    draws[i * n + j] = yj + sign * (2.0 * i as f64 - 5.0);
                }
            }
            let keys = (0..n).map(|j| vec![(j as i64).into()]).collect();
            let distribution = PredictiveDistribution::from_draws(
                vec!["row".into()],
                keys,
                draws,
                prospicio_prob::Provenance::new("fixed"),
            )?;
            Ok(CellForecast {
                mean: y.iter().map(|v| v + 1.0).collect(),
                distribution,
                log_density: Some(test.iter().map(|&r| r as f64).collect()),
                diagnostics: None,
            })
        }
    }

    #[test]
    fn scores_of_known_draws() {
        // Latest diagonal: 2019 at 36 (increment 10) and 2020 at 24 (55)
        // are scored; 2021 at 12 and 2018 at 48 have no training level.
        let cells = small(&[
            &[100.0, 150.0, 165.0, 170.0],
            &[110.0, 170.0, 180.0],
            &[120.0, 175.0],
            &[130.0],
        ]);
        let bt = diagonal_backtest(&cells, &[&Fixed], 1, 4, 1, 0.9).unwrap();
        assert_eq!(bt.excluded(), [2]);
        let cell = |y: f64, s: f64| {
            let d: Vec<f64> = (0..4).map(|i| y + s * (2.0 * i as f64 - 5.0)).collect();
            crps(&d, y).unwrap()
        };
        let expected = (cell(10.0, 1.0) + cell(55.0, -1.0)) / 2.0;
        assert!((bt.split_scores(0, 0)[0] - expected).abs() < 1e-12);
        // Central 90% of {y-5, y-3, y-1, y+1}: [y - 4.7, y + 0.7], which
        // holds y (and its mirror image for the second cell).
        assert_eq!(bt.split_scores(0, 1), [1.0]);
        assert!((bt.split_scores(0, 2)[0] - 65.0 / 67.0).abs() < 1e-15);
        // The joint draws' totals are all 65, the outcome: CRPS zero.
        assert!(bt.split_scores(0, 3)[0].abs() < 1e-12);

        // The central 50%, [y - 3.5, y - 0.75], misses both cells.
        let half = diagonal_backtest(&cells, &[&Fixed], 1, 4, 1, 0.5).unwrap();
        assert_eq!(half.split_scores(0, 1), [0.0]);
    }

    #[test]
    fn log_densities_follow_the_scored_rows() {
        let cells = TriangleFrame::new(&raa(), "values", None).unwrap();
        let twin = NamedModel::new("twin", Fixed);
        let bt = diagonal_backtest(&cells, &[&Fixed, &twin], 3, 4, 1, 0.9).unwrap();
        let rows: Vec<f64> = bt
            .scored_rows()
            .concat()
            .iter()
            .map(|&r| r as f64)
            .collect();
        assert_eq!(rows.len(), 6 + 7 + 8);
        for l in bt.log_densities() {
            assert_eq!(l.as_deref(), Some(&rows[..]));
        }
    }

    #[test]
    fn an_age_seen_only_in_a_dropped_increment_is_excluded() {
        // 2018 at 48 is the only training cell at 48 and a negative
        // increment. A Poisson GLM with fixed dispersion does not accept it
        // (the quasi-Poisson would, and then fails: no positive mean fits a
        // level seen only in a negative cell), so 2019 at 48 cannot be
        // forecast, and is excluded with 2022 at 12 and 2018 at 60.
        let cells = small(&[
            &[100.0, 150.0, 165.0, 160.0, 162.0],
            &[110.0, 160.0, 180.0, 185.0],
            &[120.0, 175.0, 190.0],
            &[125.0, 180.0],
            &[130.0],
        ]);
        let odp = GlmCandidate {
            name: "odp".into(),
            terms: Terms::new()
                .intercept()
                .factor("origin")
                .factor("development"),
            glm: Glm::new(prospicio_models::Family::Poisson, Link::Log),
        };
        let bt = diagonal_backtest(&cells, &[&odp], 1, 100, 7, 0.9).unwrap();
        assert_eq!(bt.excluded(), [3]);
        assert_eq!(bt.scored_rows()[0].len(), 2);
        // A model that fits every row still scores only the shared rows.
        let both = diagonal_backtest(&cells, &[&odp, &Fixed], 1, 100, 7, 0.9).unwrap();
        assert_eq!(both.scored_rows(), bt.scored_rows());
    }

    #[test]
    fn rejects_bad_settings() {
        let cells = TriangleFrame::new(&raa(), "values", None).unwrap();
        let odp = GlmCandidate {
            name: "odp".into(),
            terms: Terms::new()
                .intercept()
                .factor("origin")
                .factor("development"),
            glm: Glm::over_dispersed_poisson(),
        };
        assert!(diagonal_backtest(&cells, &[], 2, 10, 1, 0.9).is_err());
        assert!(diagonal_backtest(&cells, &[&odp], 2, 0, 1, 0.9).is_err());
        assert!(diagonal_backtest(&cells, &[&odp], 2, 10, 1, 1.0).is_err());
        assert!(diagonal_backtest(&cells, &[&odp], 10, 10, 1, 0.9).is_err());
    }

    fn forecast_of(draws: Vec<f64>, n: usize, log_density: Option<Vec<f64>>) -> CellForecast {
        let keys = (0..n).map(|j| vec![(j as i64).into()]).collect();
        CellForecast {
            mean: vec![1.0; n],
            distribution: PredictiveDistribution::from_draws(
                vec!["row".into()],
                keys,
                draws,
                prospicio_prob::Provenance::new("known"),
            )
            .unwrap(),
            log_density,
            diagnostics: None,
        }
    }

    #[test]
    fn mean_log_density_and_total_percentile_follow_their_definitions() {
        // Two rows, four simulations: row sums 3, 5, 7, 9 (rows 1+2, 2+3,
        // 3+4, 4+5). The actual total 7 has 3 of 4 totals at or below it.
        let draws = vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        let y = [3.0, 4.0];
        let f = forecast_of(draws.clone(), 2, Some(vec![-1.0, -2.5]));
        let s = score(&f, &y, 0.9).unwrap();
        assert!((s[4] - (-1.75)).abs() < 1e-15);
        assert_eq!(s[5], 0.75);
        // A total below every draw is the 0 percentile, above every one 1.
        assert_eq!(score(&f, &[0.0, 0.0], 0.9).unwrap()[5], 0.0);
        assert_eq!(score(&f, &[9.0, 9.0], 0.9).unwrap()[5], 1.0);
        // A model without densities has no mean log density.
        let none = forecast_of(draws, 2, None);
        assert!(score(&none, &y, 0.9).unwrap()[4].is_nan());
    }

    #[test]
    fn the_new_metrics_come_after_the_old_four() {
        assert_eq!(
            METRICS,
            [
                "crps",
                "coverage",
                "actual_vs_expected",
                "total_crps",
                "mean_log_density",
                "total_percentile"
            ]
        );
    }

    #[test]
    fn the_backtest_reports_the_new_metrics_of_a_glm_and_no_diagnostics() {
        let cells = small(&[
            &[100.0, 150.0, 165.0, 170.0],
            &[110.0, 170.0, 180.0],
            &[120.0, 175.0],
            &[130.0],
        ]);
        let bt = diagonal_backtest(&cells, &[&Fixed], 1, 4, 1, 0.9).unwrap();
        // Fixed's log density is the row number, the draws' totals are all
        // the actual total.
        let rows = &bt.scored_rows()[0];
        let expected = rows.iter().map(|&r| r as f64).sum::<f64>() / rows.len() as f64;
        assert!((bt.split_scores(0, 4)[0] - expected).abs() < 1e-12);
        assert_eq!(bt.split_scores(0, 5), [1.0]);
        assert_eq!(bt.diagonals(), [cells.calendar_of(rows[0])]);

        let odp = GlmCandidate {
            name: "odp".into(),
            terms: Terms::new()
                .intercept()
                .factor("origin")
                .factor("development"),
            glm: Glm::over_dispersed_poisson(),
        };
        let cells = TriangleFrame::new(&raa(), "values", None).unwrap();
        let bt = diagonal_backtest(&cells, &[&odp], 2, 300, 3, 0.9).unwrap();
        assert_eq!(bt.diagnostics(0).len(), 2);
        assert!(bt.diagnostics(0).iter().all(Option::is_none));
        assert!(bt.warnings().is_empty());
        let (lpd, k) = (bt.log_densities()[0].as_ref().unwrap(), 0);
        let first = bt.scored_rows()[k].len();
        let mean = lpd[..first].iter().sum::<f64>() / first as f64;
        assert!((bt.split_scores(0, 4)[k] - mean).abs() < 1e-12);
        let p = bt.split_scores(0, 5);
        assert!(p.iter().all(|v| (0.0..=1.0).contains(v)));
        let d = bt.diagonals();
        assert_eq!(d.len(), 2);
        assert_eq!(d[1], d[0] + 1);
    }

    /// `Fixed` with warnings in its diagnostics.
    struct Warned;

    impl TriangleModel for Warned {
        fn name(&self) -> &str {
            "warned"
        }

        fn forecast(
            &self,
            cells: &TriangleFrame,
            train: &[usize],
            test: &[usize],
            n_sims: usize,
            seed: u64,
        ) -> Result<CellForecast> {
            let mut f = Fixed.forecast(cells, train, test, n_sims, seed)?;
            f.diagnostics = Some(Diagnostics {
                max_rhat: 1.2,
                min_ess_bulk: 40.0,
                min_ess_tail: 50.0,
                divergences: 3,
                warnings: vec![format!("{} test rows", test.len()), "second".into()],
            });
            Ok(f)
        }
    }

    #[test]
    fn warnings_are_listed_by_model_and_split_and_the_fit_is_scored() {
        let cells = TriangleFrame::new(&raa(), "values", None).unwrap();
        let bt = diagonal_backtest(&cells, &[&Fixed, &Warned], 2, 4, 1, 0.9).unwrap();
        assert!(bt.diagnostics(0).iter().all(Option::is_none));
        assert_eq!(bt.diagnostics(1).len(), 2);
        assert_eq!(bt.diagnostics(1)[1].as_ref().unwrap().divergences, 3);
        let w = bt.warnings();
        assert_eq!(
            w,
            [
                (1, 0, "7 test rows".to_string()),
                (1, 0, "second".to_string()),
                (1, 1, "8 test rows".to_string()),
                (1, 1, "second".to_string()),
            ]
        );
        // Scored the same as the model without warnings.
        for metric in 0..METRICS.len() {
            assert_eq!(bt.split_scores(0, metric), bt.split_scores(1, metric));
        }
    }

    #[test]
    fn duplicate_names_are_an_error_before_any_fit() {
        struct Panics;
        impl TriangleModel for Panics {
            fn name(&self) -> &str {
                "fixed"
            }
            fn forecast(
                &self,
                _: &TriangleFrame,
                _: &[usize],
                _: &[usize],
                _: usize,
                _: u64,
            ) -> Result<CellForecast> {
                panic!("fitted before the names were checked");
            }
        }
        let cells = TriangleFrame::new(&raa(), "values", None).unwrap();
        let e = diagonal_backtest(&cells, &[&Fixed, &Panics], 1, 4, 1, 0.9).unwrap_err();
        assert!(e.to_string().contains("\"fixed\""), "{e}");
    }

    #[test]
    fn a_named_model_renames_and_delegates() {
        let cells = TriangleFrame::new(&raa(), "values", None).unwrap();
        let a = NamedModel::new("first", Fixed);
        let b = NamedModel::new(String::from("second"), Fixed);
        assert_eq!(a.name(), "first");
        let bt = diagonal_backtest(&cells, &[&a, &b], 2, 4, 1, 0.9).unwrap();
        assert_eq!(bt.models(), ["first", "second"]);
        let plain = diagonal_backtest(&cells, &[&Fixed], 2, 4, 1, 0.9).unwrap();
        for metric in 0..METRICS.len() {
            assert_eq!(bt.split_scores(1, metric), plain.split_scores(0, metric));
        }
        // Boxed models of different kinds make a list.
        let list: Vec<NamedModel> = vec![a, NamedModel::new("glm", Warned)];
        let refs: Vec<&dyn TriangleModel> = list.iter().map(|m| m as &dyn TriangleModel).collect();
        assert_eq!(
            diagonal_backtest(&cells, &refs, 1, 4, 1, 0.9)
                .unwrap()
                .models(),
            ["first", "glm"]
        );
    }
}
