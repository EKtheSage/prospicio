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
use act_glm::{Glm, GlmFit};
use act_math::linalg::{cholesky, lower_mul};
use act_math::special::{ln_gamma, norm_quantile};
use act_models::metrics::{coverage, crps};
use act_models::{Design, Family, Fitted, Model, Terms};
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
}

/// A GLM on the features of [`TriangleFrame`]: the ODP model is a
/// quasi-Poisson GLM with intercept, origin and development factors.
///
/// The frame is coded with factor levels learned on the train and test
/// rows together, the GLM is fitted on the train rows and the test rows
/// get [`Fitted::predict_distribution`]. Training rows whose response is
/// outside the family's range (a negative increment for the Poisson) are
/// left out of the fit.
///
/// The log density of each test response is a log predictive density with
/// parameter uncertainty: the log of the mean, over `n_sims` draws of the
/// coefficients from their normal approximation, of the response density
/// at the fitted dispersion. Without the averaging a model with many
/// poorly identified parameters (an origin factor for the newest origins)
/// is scored as if they were known. The response density is
/// [`Family::log_density`], except for the Poisson: the over-dispersed
/// Poisson's response `φ N` lives on a lattice of spacing `φ`, which
/// observed increments are not on. Its log density is the count's log
/// probability at `k = y/φ`, continued to every `k ≥ 0` by `ln Γ`, less
/// `ln φ`: a density per unit of `y`, comparable with other models'. (At
/// `φ = 1` and a whole `y` it is the Poisson log probability.)
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
        let family = self.glm.family;
        let response = cells.response();
        let train: Vec<usize> = train
            .iter()
            .copied()
            .filter(|&r| family.valid_y(response[r]))
            .collect();
        let both: Vec<usize> = train.iter().chain(test).copied().collect();
        let coding = self.terms.fit(&cells.select(&both)?)?;
        let train_design = coding.design(&cells.select(&train)?)?;
        let test_design = coding.design(&cells.select(test)?)?;
        let fit = self.glm.fit(&train_design, &cells.response_of(&train))?;
        let mean = fit.predict(&test_design)?;
        let distribution = fit.predict_distribution(&test_design, n_sims, seed)?;
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
        } else if family == Family::Poisson {
            Ok(continuous_poisson_log_density(y, mu, phi))
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

/// `ln[λ^k e^{-λ} / Γ(k + 1)] - ln φ` with `k = y/φ`, `λ = μ/φ`: the
/// over-dispersed Poisson's log probability continued off its lattice, per
/// unit of `y`. `-∞` for a negative `y`.
fn continuous_poisson_log_density(y: f64, mu: f64, phi: f64) -> f64 {
    if y < 0.0 || !y.is_finite() {
        return f64::NEG_INFINITY;
    }
    let (k, lambda) = (y / phi, mu / phi);
    let ln_term = if k == 0.0 { 0.0 } else { k * lambda.ln() };
    ln_term - lambda - ln_gamma(k + 1.0) - phi.ln()
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
    /// their origin or development level has no training row.
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
/// (the newest origin and the oldest age on the diagonal) cannot be
/// forecast by a model with origin and development effects; it is left
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
        let has = |level: &dyn Fn(usize) -> usize, r: usize| {
            s.train.iter().any(|&t| level(t) == level(r))
        };
        let keep: Vec<usize> = s
            .test
            .iter()
            .copied()
            .filter(|&r| has(&|x| cells.origin_of(x), r) && has(&|x| cells.development_of(x), r))
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

    #[test]
    fn continuous_poisson_matches_the_lattice() {
        // At φ = 1 and a whole y it is the Poisson log probability.
        let exact = Family::Poisson.log_density(3.0, 2.5, 1.0, 1.0).unwrap();
        assert!((continuous_poisson_log_density(3.0, 2.5, 1.0) - exact).abs() < 1e-12);
        // On the lattice of spacing φ: the count's probability over φ.
        let lattice = Family::Poisson.log_density(40.0, 25.0, 10.0, 1.0).unwrap();
        let continued = continuous_poisson_log_density(40.0, 25.0, 10.0);
        assert!((continued - (lattice - 10f64.ln())).abs() < 1e-12);
        assert_eq!(
            continuous_poisson_log_density(-1.0, 2.0, 1.0),
            f64::NEG_INFINITY
        );
        assert!(continuous_poisson_log_density(0.0, 2.0, 3.0).is_finite());
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
                glm: Glm::new(Family::Poisson, Link::Log).dispersion(act_glm::Dispersion::Pearson),
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
