//! Calendar-diagonal backtest for any model of the cells of a triangle.
//!
//! Reserving forecasts the next calendar period, so a model is scored by
//! refitting it with the latest diagonals held out, one at a time, and
//! forecasting each held-out diagonal (`docs/design/models.md`, "Fitting a
//! triangle with several models"). Any model takes part through
//! [`TriangleModel`]; [`GlmCandidate`] wraps an `act-glm` GLM.
//!
//! The scores are the mean cell CRPS, the coverage of a central interval,
//! actual vs expected on the diagonal total, and the CRPS of the diagonal
//! total from the joint draws. The pointwise held-out log densities feed
//! [`act_models::stack::stacking_weights`].

use act_core::{Error as CoreError, StreamRng};
use act_glm::{Glm, GlmFit, ParameterDraws};
use act_math::linalg::{cholesky, lower_mul};
use act_math::special::norm_quantile;
use act_models::metrics::{coverage, crps};
use act_models::{Design, Fitted, Link, Model, Terms};
use act_prob::PredictiveDistribution;

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
/// at the fitted dispersion ([`act_models::Family::log_density`], which
/// scores the over-dispersed Poisson by a normalized density in `y`).
/// Without the averaging a model with many poorly identified parameters
/// (an origin factor for the newest origins) is scored as if they were
/// known.
///
/// ```
/// use act_glm::Glm;
/// use act_models::Terms;
/// use act_reserving::{GlmCandidate, TriangleModel};
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
    let density = |y: f64, mu: f64| -> act_core::Result<f64> {
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

/// Metric names of a [`Backtest`], in the order of its metric indices.
pub const METRICS: [&str; 4] = ["crps", "coverage", "actual_vs_expected", "total_crps"];

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
    /// the input of [`act_models::stack::stacking_weights`].
    pub fn log_densities(&self) -> &[Option<Vec<f64>>] {
        &self.log_densities
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
/// over expected on the diagonal total `Σy / Σμ`, and the CRPS of the
/// diagonal total from the draws summed per simulation.
///
/// ```
/// use act_glm::Glm;
/// use act_models::Terms;
/// use act_reserving::{
///     DevelopmentColumn, GlmCandidate, Grain, Long, Month, Triangle, TriangleFrame,
///     diagonal_backtest,
/// };
///
/// let origin = [2019, 2019, 2019, 2019, 2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     index: None,
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
/// # Ok::<(), act_reserving::Error>(())
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
    })
}

/// The [`METRICS`] of one forecast against the outcomes `y`.
fn score(f: &CellForecast, y: &[f64], interval: f64) -> Result<[f64; 4]> {
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
    Ok([
        cell_crps / n as f64,
        coverage(y, &lo, &hi)?,
        actual / f.mean.iter().sum::<f64>(),
        crps(&totals, actual)?,
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
    use act_models::Link;

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
                glm: Glm::new(act_models::Family::Poisson, Link::Log)
                    .dispersion(act_glm::Dispersion::Pearson),
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
            index: None,
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
                act_prob::Provenance::new("fixed"),
            )?;
            Ok(CellForecast {
                mean: y.iter().map(|v| v + 1.0).collect(),
                distribution,
                log_density: Some(test.iter().map(|&r| r as f64).collect()),
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
        let bt = diagonal_backtest(&cells, &[&Fixed, &Fixed], 3, 4, 1, 0.9).unwrap();
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
            glm: Glm::new(act_models::Family::Poisson, Link::Log),
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
}
