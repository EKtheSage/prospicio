//! The over-dispersed Poisson (ODP) reserving model as a quasi-Poisson GLM.
//!
//! The incremental value of each observed cell is modelled with a log link,
//! an intercept and origin and development factors:
//! `ln E[X_od] = c + a_o + b_d`, `Var X_od = φ E[X_od]`. The fit runs
//! [`prospicio_glm::Glm::over_dispersed_poisson`] on the observed rows of a
//! [`TriangleFrame`] and predicts the future rows. With these factors the
//! fitted future cells summed by origin are the volume-weighted Chain
//! Ladder reserves (Renshaw and Verrall, 1998), and Pearson's dispersion is
//! the ODP bootstrap's scale. [`OdpGlmFit::prediction_error`] gives the
//! reserves' analytic prediction error by the delta method (Taylor and
//! McGuire, CAS Monograph 3, Table 5-4), and [`OdpGlmFit::residuals`] the
//! fit's residuals by cell, summarized by origin, development or calendar
//! period (their Chapter 6). [`OdpGlm::fit_design`] fits another design on
//! the triangle's cells: fewer parameters, calendar trends, interactions
//! (their Chapter 7). See `docs/design/triangle.md`, "ODP GLM".

use prospicio_core::{Lag, Period};
pub use prospicio_glm::diagnostics::SumPredictionError;
use prospicio_glm::{Glm, GlmFit, ParameterDraws};
use prospicio_models::Column;
use prospicio_models::{Coding, Design, Fitted, Model, Terms};
use prospicio_prob::{ComponentKey, KeyValue, PredictiveDistribution};

use crate::error::{Error, Result};
use crate::frame::TriangleFrame;
use crate::triangle::Triangle;

/// The ODP model as a quasi-Poisson GLM with origin and development
/// factors.
///
/// ```
/// use prospicio_reserving::{ChainLadder, DevelopmentColumn, Grain, Long, Month, OdpGlm, Triangle};
///
/// let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
///     values: &[("paid", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0])],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: true,
/// })?;
/// let odp = OdpGlm::default().fit(&tri, "paid")?;
/// let cl = ChainLadder::default().fit(&tri, "paid")?;
/// assert!((odp.total_reserve() - cl.total_reserve()).abs() < 1e-8);
/// assert_eq!(odp.reserves[0], 0.0);
///
/// // Draws over the future cells, summed to reserves by origin.
/// let by_origin = odp.predict_distribution(1_000, 42)?.aggregate(&["origin"])?;
/// assert_eq!(by_origin.n_components(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdpGlm {
    /// IRLS convergence tolerance (see [`Glm::tolerance`]).
    pub tolerance: f64,
    /// Maximum IRLS iterations.
    pub max_iterations: usize,
}

impl Default for OdpGlm {
    /// [`Glm::over_dispersed_poisson`]'s tolerance (`1e-10`) and iteration
    /// limit (100).
    fn default() -> Self {
        let glm = Glm::over_dispersed_poisson();
        Self {
            tolerance: glm.tolerance,
            max_iterations: glm.max_iterations,
        }
    }
}

/// A fitted ODP GLM: the GLM, and its predictions for the future cells.
#[derive(Debug, Clone)]
pub struct OdpGlmFit {
    /// The quasi-Poisson GLM on the observed increments: coefficients,
    /// standard errors and Pearson's dispersion.
    pub glm: GlmFit,
    /// The design coding of [`OdpGlm::fit`]: `(Intercept)`,
    /// `origin[<period>]` and `development[<age>]` columns. The references
    /// are the first origin and the first age; the other levels are in
    /// string order (so `development[108]` comes before `development[24]`).
    /// `None` for a fit of [`OdpGlm::fit_design`], whose column names are
    /// [`GlmFit::names`].
    pub coding: Option<Coding>,
    /// Origin periods of the triangle.
    pub origins: Vec<Period>,
    /// The future cells, as (origin, age in months), origin-major.
    pub future: Vec<(Period, Lag)>,
    /// Fitted mean of each future cell, aligned with `future`.
    pub future_means: Vec<f64>,
    /// Reserve of each origin, aligned with `origins`: its future means
    /// summed, 0 for a fully developed origin.
    pub reserves: Vec<f64>,
    /// The observed cells the GLM was fitted to, as (origin, age in
    /// months), origin-major.
    pub observed: Vec<(Period, Lag)>,
    /// Calendar period of each observed cell, aligned with `observed`:
    /// its 0-based index from the earliest valuation and its label at the
    /// development grain.
    pub observed_calendar: Vec<(i64, String)>,
    /// Incremental value of each observed cell, aligned with `observed`.
    pub observed_values: Vec<f64>,
    /// Design of the future cells, for prediction.
    future_design: Design,
    /// Design of the observed cells.
    observed_design: Design,
}

/// The residuals of an ODP GLM, one per observed cell in the order of
/// [`OdpGlmFit::observed`], from [`OdpGlmFit::residuals`].
#[derive(Debug, Clone, PartialEq)]
pub struct OdpResiduals {
    /// Origin of each cell.
    pub origin: Vec<Period>,
    /// Age in months of each cell.
    pub development: Vec<Lag>,
    /// Calendar period of each cell: its 0-based index from the earliest
    /// valuation and its label.
    pub calendar: Vec<(i64, String)>,
    /// The incremental value.
    pub actual: Vec<f64>,
    /// The fitted mean `μ`.
    pub fitted: Vec<f64>,
    /// `actual / fitted`, the cell of the actual-to-fitted heat map.
    pub actual_to_fitted: Vec<f64>,
    /// Pearson residual `(y − μ) / √μ`.
    pub pearson: Vec<f64>,
    /// Deviance residual `sign(y − μ) √(2 (y ln(y / μ) − (y − μ)))`.
    pub deviance: Vec<f64>,
    /// The hat value `h`, the diagonal of the hat matrix.
    pub leverage: Vec<f64>,
    /// `deviance / √(φ (1 − h))`, R's `rstandard`; NaN for a cell with
    /// leverage 1 (the only cell of its origin or age, fitted exactly).
    pub standardized_deviance: Vec<f64>,
    /// `pearson / √(φ (1 − h))`; NaN where `standardized_deviance` is.
    pub standardized_pearson: Vec<f64>,
}

/// How [`OdpResiduals::summary`] groups the cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualGroup {
    /// By origin period, in the triangle's order.
    Origin,
    /// By development age.
    Development,
    /// By calendar period, earliest first.
    Calendar,
}

/// The standardized deviance residuals of one group of cells, from
/// [`OdpResiduals::summary`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualSummary {
    /// The origin, age in months, or calendar period.
    pub label: String,
    /// Cells in the group with a defined residual (leverage below 1).
    pub n: usize,
    /// Their mean; NaN when `n` is 0.
    pub mean: f64,
    /// Their sample standard deviation (divisor `n − 1`); NaN when `n` is
    /// below 2.
    pub sd: f64,
}

/// Leverage at or above which a cell is taken as fitted exactly.
const EXACT_FIT: f64 = 1.0 - 1e-9;

impl OdpResiduals {
    /// The standardized deviance residuals grouped by origin, development
    /// age or calendar period: a row per group, with the count, mean and
    /// standard deviation of its defined residuals. A trend in the means
    /// over calendar periods points to a calendar effect the model lacks;
    /// a spread that falls with age, to a variance that is not `φ μ`
    /// (Taylor and McGuire, CAS Monograph 3, Chapter 6).
    pub fn summary(&self, group: ResidualGroup) -> Vec<ResidualSummary> {
        let mut keys: Vec<(i64, String)> = Vec::new();
        let mut members: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.origin.len() {
            let key = match group {
                ResidualGroup::Origin => (0, self.origin[i].to_string()),
                ResidualGroup::Development => (
                    i64::from(self.development[i]),
                    self.development[i].to_string(),
                ),
                ResidualGroup::Calendar => self.calendar[i].clone(),
            };
            let at = match keys.iter().position(|k| k.1 == key.1) {
                Some(at) => at,
                None => {
                    keys.push(key);
                    members.push(Vec::new());
                    keys.len() - 1
                }
            };
            let r = self.standardized_deviance[i];
            if !r.is_nan() {
                members[at].push(r);
            }
        }
        let mut rows: Vec<(i64, ResidualSummary)> = keys
            .into_iter()
            .zip(members)
            .map(|((order, label), r)| {
                let n = r.len();
                let mean = r.iter().sum::<f64>() / n as f64;
                let sd = if n < 2 {
                    f64::NAN
                } else {
                    (r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
                };
                (order, ResidualSummary { label, n, mean, sd })
            })
            .collect();
        // Origins keep the triangle's order; ages and calendar periods sort
        // by their number.
        rows.sort_by_key(|(order, _)| *order);
        rows.into_iter().map(|(_, row)| row).collect()
    }
}

/// The analytic prediction error of an ODP GLM's reserves: for each origin
/// and in total, the reserve, its process variance `φ Σ μ` and its
/// parameter variance `g' Σ g` by the delta method, `g = Σ μ x` over the
/// future cells (Taylor and McGuire, CAS Monograph 3, (5-11) to (5-13)).
#[derive(Debug, Clone, PartialEq)]
pub struct OdpPredictionError {
    /// One per origin, aligned with [`OdpGlmFit::origins`]; zero for a
    /// fully developed origin.
    pub by_origin: Vec<SumPredictionError>,
    /// The total reserve. Its variance exceeds the sum of the origins'
    /// because their parameter errors are correlated: every origin shares
    /// the development factors.
    pub total: SumPredictionError,
}

impl OdpGlm {
    /// The GLM this model fits.
    pub fn glm(&self) -> Glm {
        let mut glm = Glm::over_dispersed_poisson();
        glm.tolerance = self.tolerance;
        glm.max_iterations = self.max_iterations;
        glm
    }

    /// Fits the GLM to the incremental values of `column` in a
    /// single-segment triangle and predicts the cells below the latest
    /// diagonal.
    ///
    /// Fails if an origin or development age has no observed increment
    /// (its factor level would appear only in future cells), if a past
    /// cell has no increment (a hole: a missing value, or a value after
    /// one), or if the GLM does not converge (for example when a level's
    /// only observed increments are negative, so no positive mean fits).
    /// Negative increments are otherwise fitted: the quasi-likelihood needs
    /// only `V(μ) = μ` and `μ > 0` (`prospicio_glm::Glm::over_dispersed_poisson`).
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<OdpGlmFit> {
        let cells = TriangleFrame::new(triangle, column, None)?;
        check_holes(&cells)?;
        let origins = cells.origins();
        let ages = cells.ages();
        // Every factor level needs an observed increment.
        let mut origin_seen = vec![false; origins.len()];
        let mut age_seen = vec![false; ages.len()];
        for &r in cells.observed() {
            origin_seen[cells.origin_of(r)] = true;
            age_seen[cells.development_of(r)] = true;
        }
        if let Some(o) = origin_seen.iter().position(|seen| !seen) {
            return Err(Error::OdpGlm(format!(
                "origin {} has no observed increment, so its factor level appears only \
                 in future cells and cannot be estimated",
                origins[o]
            )));
        }
        if let Some(d) = age_seen.iter().position(|seen| !seen) {
            return Err(Error::OdpGlm(format!(
                "development age {} months has no observed increment, so its factor level \
                 appears only in future cells and cannot be estimated",
                ages[d]
            )));
        }

        // Levels are learned on every cell so the future rows code with the
        // same columns as the observed ones.
        let coding = Terms::new()
            .intercept()
            .factor_with_reference("origin", &origins[0].to_string())
            .factor_with_reference("development", &ages[0].to_string())
            .fit(cells.frame())?;
        let observed_design = coding.design(&cells.select(cells.observed())?)?;
        let future_design = coding.design(&cells.select(cells.future())?)?;
        self.finish(&cells, Some(coding), observed_design, future_design)
    }

    /// Fits the GLM with another design: `design` has one row per row of
    /// [`TriangleFrame::new(triangle, column, None)`](TriangleFrame::new),
    /// in its order (every origin × development cell, origin-major), and
    /// its columns are the model's, intercept included if wanted. Its
    /// offset and weights, if set, are used too. The fit, reserves,
    /// prediction error, residuals and draws are as for [`fit`](Self::fit).
    ///
    /// This is how the parameters are cut down or the model widened (Taylor
    /// and McGuire, CAS Monograph 3, Chapter 7): an origin effect as a
    /// polynomial in `origin_index`, development effects as a line with a
    /// kink (a hinge on `development_index`), a trend in `calendar_index`,
    /// a shift for some origins at some ages. A `Terms` coding on the
    /// frame's columns gives such a design (`coding.design(cells.frame())`),
    /// or the columns can be built from the frame's indices directly. A
    /// numeric term extrapolates to future cells; a factor level that only
    /// future cells take (a `calendar` factor, for one) cannot be estimated.
    ///
    /// Fails if `design` does not have a row per cell, if a column is zero
    /// on every observed cell (its coefficient cannot be estimated), on a
    /// hole as [`fit`](Self::fit) does, or if the GLM does not converge (a
    /// collinear design among them).
    ///
    /// ```
    /// use prospicio_models::Terms;
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, OdpGlm, Triangle, TriangleFrame};
    ///
    /// let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
    ///     values: &[("paid", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0])],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })?;
    /// // One level for every origin, development factors as usual.
    /// let cells = TriangleFrame::new(&tri, "paid", None)?;
    /// let design = Terms::new()
    ///     .intercept()
    ///     .factor_with_reference("development", "12")
    ///     .fit(cells.frame())?
    ///     .design(cells.frame())?;
    /// let fit = OdpGlm::default().fit_design(&tri, "paid", &design)?;
    /// assert_eq!(fit.glm.names(), ["(Intercept)", "development[24]", "development[36]"]);
    /// assert!(fit.coding.is_none());
    /// assert_eq!(fit.reserves.len(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn fit_design(
        &self,
        triangle: &Triangle,
        column: &str,
        design: &Design,
    ) -> Result<OdpGlmFit> {
        let cells = TriangleFrame::new(triangle, column, None)?;
        if design.n_rows() != cells.n_rows() {
            return Err(Error::OdpGlm(format!(
                "the design has {} rows; it needs one per cell of the triangle, {} \
                 (the rows of its TriangleFrame, in order)",
                design.n_rows(),
                cells.n_rows()
            )));
        }
        check_holes(&cells)?;
        let observed_design = design.select(cells.observed());
        if let Some(j) = (0..observed_design.n_cols())
            .find(|&j| observed_design.column(j).iter().all(|&x| x == 0.0))
        {
            return Err(Error::OdpGlm(format!(
                "column {} is zero on every observed cell, so its coefficient cannot be \
                 estimated",
                design.names()[j]
            )));
        }
        let future_design = design.select(cells.future());
        self.finish(&cells, None, observed_design, future_design)
    }

    /// Fits the GLM on the observed rows and predicts the future ones.
    fn finish(
        &self,
        cells: &TriangleFrame,
        coding: Option<Coding>,
        observed_design: Design,
        future_design: Design,
    ) -> Result<OdpGlmFit> {
        let origins = cells.origins().to_vec();
        let ages = cells.ages();
        let observed = cells.observed();
        let future = cells.future();
        let y = cells.response_of(observed);
        let glm = self.glm().fit(&observed_design, &y)?;
        let future_means = glm.predict(&future_design)?;

        let mut reserves = vec![0.0; origins.len()];
        for (&r, m) in future.iter().zip(&future_means) {
            reserves[cells.origin_of(r)] += m;
        }
        let cell = |r: usize| (origins[cells.origin_of(r)], ages[cells.development_of(r)]);
        let future = future.iter().map(|&r| cell(r)).collect();
        let Some(Column::Categorical(calendar)) = cells.frame().column("calendar") else {
            unreachable!("a triangle frame has a calendar factor")
        };
        let observed_calendar = observed
            .iter()
            .map(|&r| (cells.calendar_of(r), calendar[r].clone()))
            .collect();
        let observed_cells = observed.iter().map(|&r| cell(r)).collect();
        Ok(OdpGlmFit {
            glm,
            coding,
            origins,
            future,
            future_means,
            reserves,
            observed: observed_cells,
            observed_calendar,
            observed_values: y,
            future_design,
            observed_design,
        })
    }
}

/// Fails on a past cell with no increment (a missing value, or a value
/// after one): the GLM needs every past increment.
fn check_holes(cells: &TriangleFrame) -> Result<()> {
    let mut has_role = vec![false; cells.n_rows()];
    for &r in cells.observed().iter().chain(cells.future()) {
        has_role[r] = true;
    }
    match has_role.iter().position(|known| !known) {
        Some(hole) => Err(Error::OdpGlm(format!(
            "origin {} at {} months is a past cell with no increment (missing, or after a \
             missing value); the GLM needs every past increment",
            cells.origins()[cells.origin_of(hole)],
            cells.ages()[cells.development_of(hole)]
        ))),
        None => Ok(()),
    }
}

impl OdpGlmFit {
    /// Sum of the reserves.
    pub fn total_reserve(&self) -> f64 {
        self.reserves.iter().sum()
    }

    /// Design of the future cells, rows aligned with
    /// [`future`](Self::future): what [`glm`](Self::glm) predicts from.
    pub fn future_design(&self) -> &Design {
        &self.future_design
    }

    /// The reserves' prediction error by origin and in total: the process
    /// variance `φ Σ μ` of the future cells and the parameter variance
    /// `g' Σ g` by the delta method, with `Σ` the coefficients' covariance
    /// and `g = Σ μ x` over the cells (Taylor and McGuire, CAS Monograph 3,
    /// (5-11) to (5-13), Table 5-4). The root mean square error of
    /// prediction is [`SumPredictionError::rmsep`].
    ///
    /// ```
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, OdpGlm, Triangle};
    ///
    /// let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
    ///     values: &[("paid", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0])],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })?;
    /// let fit = OdpGlm::default().fit(&tri, "paid")?;
    /// let error = fit.prediction_error()?;
    /// assert_eq!(error.by_origin[0].rmsep(), 0.0);
    /// assert!((error.total.total - fit.total_reserve()).abs() < 1e-8);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn prediction_error(&self) -> Result<OdpPredictionError> {
        let mut rows = vec![Vec::new(); self.origins.len()];
        for (i, (origin, _)) in self.future.iter().enumerate() {
            let o = self
                .origins
                .iter()
                .position(|p| p == origin)
                .expect("every future cell's origin is an origin of the fit");
            rows[o].push(i);
        }
        let by_origin = rows
            .iter()
            .map(|r| {
                self.glm
                    .sum_prediction_error(&self.future_design.select(r))
                    .map_err(Error::from)
            })
            .collect::<Result<_>>()?;
        let total = self.glm.sum_prediction_error(&self.future_design)?;
        Ok(OdpPredictionError { by_origin, total })
    }

    /// The residuals of every observed cell: Pearson, deviance and
    /// standardized deviance residuals, leverages and the ratio of actual
    /// to fitted, with [`OdpResiduals::summary`] grouping them by origin,
    /// development or calendar period (Taylor and McGuire, CAS Monograph 3,
    /// Chapter 6: Figures 6-19, 6-20 and the heat map of Figure 6-22). A
    /// cell with leverage 1, the only observation of its origin or age, is
    /// fitted exactly and has no standardized residual (NaN).
    ///
    /// ```
    /// use prospicio_reserving::{DevelopmentColumn, Grain, Long, Month, OdpGlm, ResidualGroup, Triangle};
    ///
    /// let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
    /// let tri = Triangle::from_long(&Long {
    ///     keys: &[],
    ///     origin: &origin,
    ///     development: DevelopmentColumn::Age(&[12, 24, 36, 12, 24, 12]),
    ///     values: &[("paid", &[100.0, 150.0, 165.0, 110.0, 170.0, 120.0])],
    ///     origin_grain: Grain::Year,
    ///     development_grain: Grain::Year,
    ///     cumulative: true,
    /// })?;
    /// let residuals = OdpGlm::default().fit(&tri, "paid")?.residuals()?;
    /// assert_eq!(residuals.actual.len(), 6);
    /// // 2022's only cell is fitted exactly.
    /// assert!(residuals.standardized_deviance[5].is_nan());
    /// let by_age = residuals.summary(ResidualGroup::Development);
    /// assert_eq!(by_age.iter().map(|g| g.n).collect::<Vec<_>>(), [2, 2, 0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn residuals(&self) -> Result<OdpResiduals> {
        let d = self
            .glm
            .diagnostics(&self.observed_design, &self.observed_values)?;
        let exact = |i: usize, r: f64| {
            if d.leverage[i] >= EXACT_FIT {
                f64::NAN
            } else {
                r
            }
        };
        let n = self.observed.len();
        Ok(OdpResiduals {
            origin: self.observed.iter().map(|c| c.0).collect(),
            development: self.observed.iter().map(|c| c.1).collect(),
            calendar: self.observed_calendar.clone(),
            actual: self.observed_values.clone(),
            actual_to_fitted: self
                .observed_values
                .iter()
                .zip(&d.fitted)
                .map(|(y, m)| y / m)
                .collect(),
            standardized_deviance: (0..n)
                .map(|i| exact(i, d.standardized_deviance[i]))
                .collect(),
            standardized_pearson: (0..n)
                .map(|i| exact(i, d.standardized_pearson[i]))
                .collect(),
            fitted: d.fitted,
            pearson: d.pearson,
            deviance: d.deviance,
            leverage: d.leverage,
        })
    }

    /// Mean of each future cell under normal parameter draws
    /// ([`ParameterDraws::Normal`] in
    /// [`predict_distribution_with`](Self::predict_distribution_with)):
    /// `exp(η + v / 2)`,
    /// with `η` the linear predictor and `v = xᵀ Σ x` its variance under the
    /// coefficients' normal approximation. It exceeds the fitted mean
    /// `exp(η)` (the Chain Ladder's) by the factor `exp(v / 2)`, the
    /// lognormal bias of parameter uncertainty on the log scale.
    pub fn predictive_means(&self) -> Vec<f64> {
        let p = self.glm.coefficients().len();
        let cov = self.glm.covariance();
        let x = &self.future_design;
        (0..x.n_rows())
            .zip(&self.future_means)
            .map(|(i, &mean)| {
                let v: f64 = (0..p)
                    .flat_map(|a| (0..p).map(move |b| (a, b)))
                    .map(|(a, b)| x.column(a)[i] * cov[a * p + b] * x.column(b)[i])
                    .sum();
                mean * (v / 2.0).exp()
            })
            .collect()
    }

    /// Joint draws of the future cells' increments, with dimensions
    /// `["origin", "development"]` and keys `(origin period, age in
    /// months)` in the order of [`future`](Self::future).
    ///
    /// Simulation `i` (stream `i` of `seed`) draws the coefficients from
    /// their normal approximation (parameter uncertainty, shared by every
    /// cell) and then each cell as `φ · Poisson(μ / φ)` (process
    /// uncertainty). The parameter draws are mean-preserving
    /// ([`ParameterDraws::MeanPreserving`]): each cell's linear predictor is
    /// shifted by `-v / 2`, so its draws average its fitted mean and the
    /// total averages the Chain Ladder reserve. `aggregate(&["origin"])`
    /// gives the reserves by origin with a future cell.
    ///
    /// Fails if `n_sims` is 0 or the triangle has no future cells.
    pub fn predict_distribution(&self, n_sims: usize, seed: u64) -> Result<PredictiveDistribution> {
        self.predict_distribution_with(n_sims, seed, ParameterDraws::MeanPreserving)
    }

    /// [`predict_distribution`](Self::predict_distribution) with another
    /// treatment of parameter uncertainty: [`ParameterDraws::Normal`]
    /// (unshifted, so each cell's mean is
    /// [`predictive_means`](Self::predictive_means)) or
    /// [`ParameterDraws::Fixed`] (process uncertainty only).
    pub fn predict_distribution_with(
        &self,
        n_sims: usize,
        seed: u64,
        parameters: ParameterDraws,
    ) -> Result<PredictiveDistribution> {
        if self.future.is_empty() {
            return Err(Error::OdpGlm(
                "the triangle is fully developed: there are no future cells to simulate".into(),
            ));
        }
        let draws =
            self.glm
                .predict_distribution_with(&self.future_design, n_sims, seed, parameters)?;
        let components: Vec<ComponentKey> = self
            .future
            .iter()
            .map(|&(origin, age)| vec![KeyValue::Period(origin), KeyValue::Int(age.into())])
            .collect();
        Ok(PredictiveDistribution::from_draws(
            vec!["origin".into(), "development".into()],
            components,
            draws.draw_matrix().to_vec(),
            draws.provenance().clone().param("model", "odp_glm"),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain_ladder::ChainLadder;
    use crate::triangle::tests::annual;
    use prospicio_prob::Distribution;

    fn small() -> Triangle {
        annual(
            2020,
            &[
                &[100.0, 150.0, 165.0, 170.0],
                &[110.0, 170.0, 180.0],
                &[120.0, 175.0],
                &[130.0],
            ],
        )
    }

    #[test]
    fn future_cells_reproduce_chain_ladder() {
        let tri = small();
        let fit = OdpGlm::default().fit(&tri, "values").unwrap();
        let cl = ChainLadder::default().fit(&tri, "values").unwrap();
        assert_eq!(fit.origins, cl.origins);
        for (a, b) in fit.reserves.iter().zip(cl.reserves()) {
            assert!((a - b).abs() <= 1e-9 * b.abs().max(1.0), "{a} vs {b}");
        }
        assert_eq!(fit.reserves[0], 0.0);
        assert_eq!(fit.future.len(), 6);
        assert_eq!(fit.future[0], (Period::year(2021), 48));
        assert_eq!(fit.future_means.len(), 6);
        assert_eq!(
            fit.glm.names(),
            [
                "(Intercept)",
                "origin[2021]",
                "origin[2022]",
                "origin[2023]",
                "development[24]",
                "development[36]",
                "development[48]",
            ]
        );
        assert!(fit.glm.dispersion() > 0.0);
    }

    #[test]
    fn predictive_distribution_is_keyed_by_cell() {
        let fit = OdpGlm::default().fit(&small(), "values").unwrap();
        let pd = fit.predict_distribution(4_000, 7).unwrap();
        assert_eq!(pd.dims(), ["origin", "development"]);
        assert_eq!(pd.n_components(), 6);
        assert_eq!(
            pd.components()[0],
            vec![KeyValue::Period(Period::year(2021)), KeyValue::Int(48)]
        );
        let by_origin = pd.aggregate(&["origin"]).unwrap();
        assert_eq!(by_origin.n_components(), 3);
        // Same seed, same draws.
        let again = fit.predict_distribution(4_000, 7).unwrap();
        assert_eq!(pd.draw_matrix(), again.draw_matrix());
        // Mean-preserving draws average the fitted (Chain Ladder) reserve;
        // normal draws average the lognormal-corrected mean above it.
        let se = pd.std_dev() / 4_000f64.sqrt();
        assert!(
            (pd.mean() - fit.total_reserve()).abs() < 4.0 * se,
            "{} vs {}",
            pd.mean(),
            fit.total_reserve()
        );
        let normal = fit
            .predict_distribution_with(4_000, 7, ParameterDraws::Normal)
            .unwrap();
        let want: f64 = fit.predictive_means().iter().sum();
        assert!(
            (normal.mean() - want).abs() < 4.0 * normal.std_dev() / 4_000f64.sqrt(),
            "{} vs {want}",
            normal.mean()
        );
        assert!(want > fit.total_reserve());
        assert!(fit.predict_distribution(0, 7).is_err());
    }

    #[test]
    fn origin_with_no_increment_is_rejected() {
        // 2021 is empty inside the triangle's range: no increment to fit
        // its origin factor on.
        let tri = annual(2020, &[&[100.0, 150.0, 160.0], &[f64::NAN], &[120.0]]);
        let msg = OdpGlm::default()
            .fit(&tri, "values")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("2021"), "{msg}");
    }

    #[test]
    fn empty_trailing_origins_and_ages_are_dropped() {
        // No value at 48 months or for 2023: the segment is trimmed to the
        // origins and ages it observes, and fits as without them.
        let full = annual(2020, &[&[100.0, 150.0, 160.0], &[110.0, 165.0], &[120.0]]);
        let padded = annual(
            2020,
            &[
                &[100.0, 150.0, 160.0, f64::NAN],
                &[110.0, 165.0],
                &[120.0],
                &[f64::NAN],
            ],
        );
        let a = OdpGlm::default().fit(&full, "values").unwrap();
        let b = OdpGlm::default().fit(&padded, "values").unwrap();
        assert_eq!(a.origins, b.origins);
        assert_eq!(a.future, b.future);
        assert_eq!(a.reserves, b.reserves);
    }

    #[test]
    fn holes_are_rejected() {
        let tri = annual(
            2020,
            &[&[100.0, f64::NAN, 165.0], &[110.0, 170.0], &[120.0]],
        );
        let msg = OdpGlm::default()
            .fit(&tri, "values")
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("origin 2020 at 24 months") && msg.contains("past cell"),
            "{msg}"
        );
    }

    #[test]
    fn negative_increments_are_fitted() {
        // 2020 falls by 10 at 36 months; the level still has a positive
        // mean, and the Chain Ladder identity holds.
        let tri = annual(
            2020,
            &[
                &[100.0, 150.0, 140.0],
                &[110.0, 170.0, 190.0],
                &[120.0, 180.0],
                &[130.0],
            ],
        );
        let fit = OdpGlm::default().fit(&tri, "values").unwrap();
        let cl = ChainLadder::default().fit(&tri, "values").unwrap();
        for (a, b) in fit.reserves.iter().zip(cl.reserves()) {
            assert!((a - b).abs() <= 1e-8 * b.abs().max(1.0), "{a} vs {b}");
        }
        // A level seen only in a negative increment has no positive mean.
        let only_negative = annual(2020, &[&[100.0, 90.0], &[110.0]]);
        assert!(OdpGlm::default().fit(&only_negative, "values").is_err());
    }

    #[test]
    fn fully_developed_triangle_has_no_distribution() {
        let tri = annual(2020, &[&[100.0, 150.0, 160.0], &[110.0, 160.0, 175.0]]);
        let fit = OdpGlm::default().fit(&tri, "values").unwrap();
        assert!(fit.future.is_empty());
        assert_eq!(fit.reserves, [0.0, 0.0]);
        let msg = fit.predict_distribution(100, 1).unwrap_err().to_string();
        assert!(msg.contains("fully developed"), "{msg}");
        let error = fit.prediction_error().unwrap();
        assert_eq!(error.total.rmsep(), 0.0);
        assert!(error.by_origin.iter().all(|e| e.total == 0.0));
    }

    #[test]
    fn prediction_error_sums_over_origins() {
        let fit = OdpGlm::default().fit(&small(), "values").unwrap();
        let error = fit.prediction_error().unwrap();
        assert_eq!(error.by_origin.len(), 4);
        assert_eq!(error.by_origin[0].rmsep(), 0.0);
        for (e, r) in error.by_origin.iter().zip(&fit.reserves) {
            assert!((e.total - r).abs() < 1e-9);
        }
        // Process variances add; parameter variances do not, because the
        // origins share the development factors.
        let process: f64 = error.by_origin.iter().map(|e| e.process_variance).sum();
        let parameter: f64 = error.by_origin.iter().map(|e| e.parameter_variance).sum();
        assert!((error.total.process_variance - process).abs() < 1e-9 * process);
        assert!(error.total.parameter_variance > parameter);
        let phi = fit.glm.dispersion();
        assert!((error.total.process_variance - phi * fit.total_reserve()).abs() < 1e-6);
    }
}
