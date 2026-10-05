//! The over-dispersed Poisson (ODP) reserving model as a quasi-Poisson GLM.
//!
//! The incremental value of each observed cell is modelled with a log link,
//! an intercept and origin and development factors:
//! `ln E[X_od] = c + a_o + b_d`, `Var X_od = φ E[X_od]`. The fit runs
//! [`act_glm::Glm::over_dispersed_poisson`] on the observed rows of a
//! [`TriangleFrame`] and predicts the future rows. With these factors the
//! fitted future cells summed by origin are the volume-weighted Chain
//! Ladder reserves (Renshaw and Verrall, 1998), and Pearson's dispersion is
//! the ODP bootstrap's scale. See `docs/design/triangle.md`, "ODP GLM".

use act_core::{Lag, Period};
use act_glm::{Glm, GlmFit};
use act_models::{Coding, Design, Fitted, Model, Terms};
use act_prob::{ComponentKey, KeyValue, PredictiveDistribution};

use crate::error::{Error, Result};
use crate::frame::TriangleFrame;
use crate::triangle::Triangle;

/// The ODP model as a quasi-Poisson GLM with origin and development
/// factors.
///
/// ```
/// use act_reserving::{ChainLadder, DevelopmentColumn, Grain, Long, Month, OdpGlm, Triangle};
///
/// let origin = [2020, 2020, 2020, 2021, 2021, 2022].map(Month::january);
/// let tri = Triangle::from_long(&Long {
///     index: None,
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
    /// The design coding: `(Intercept)`, `origin[<period>]` and
    /// `development[<age>]` columns. The references are the first origin
    /// and the first age; the other levels are in string order (so
    /// `development[108]` comes before `development[24]`).
    pub coding: Coding,
    /// Origin periods of the triangle.
    pub origins: Vec<Period>,
    /// The future cells, as (origin, age in months), origin-major.
    pub future: Vec<(Period, Lag)>,
    /// Fitted mean of each future cell, aligned with `future`.
    pub future_means: Vec<f64>,
    /// Reserve of each origin, aligned with `origins`: its future means
    /// summed, 0 for a fully developed origin.
    pub reserves: Vec<f64>,
    /// Design of the future cells, for prediction.
    future_design: Design,
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
    /// one), if an increment is negative (the quasi-Poisson family needs
    /// non-negative responses), or if the GLM does not converge.
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<OdpGlmFit> {
        let cells = TriangleFrame::new(triangle, column, None)?;
        let origins = cells.origins().to_vec();
        let ages = cells.ages().to_vec();
        let observed = cells.observed();
        let label = |row: usize| {
            format!(
                "origin {} at {} months",
                origins[cells.origin_of(row)],
                ages[cells.development_of(row)]
            )
        };

        // Every past cell needs an increment.
        let future = cells.future();
        let mut has_role = vec![false; cells.n_rows()];
        for &r in observed.iter().chain(future) {
            has_role[r] = true;
        }
        if let Some(hole) = has_role.iter().position(|known| !known) {
            return Err(Error::OdpGlm(format!(
                "{} is a past cell with no increment (missing, or after a missing value); \
                 the GLM needs every past increment",
                label(hole)
            )));
        }
        // Every factor level needs an observed increment.
        let mut origin_seen = vec![false; origins.len()];
        let mut age_seen = vec![false; ages.len()];
        for &r in observed {
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
        let y = cells.response_of(observed);
        if let Some((k, &v)) = y.iter().enumerate().find(|(_, v)| **v < 0.0) {
            return Err(Error::OdpGlm(format!(
                "{} has a negative increment ({v}); the quasi-Poisson GLM needs \
                 non-negative increments",
                label(observed[k])
            )));
        }

        // Levels are learned on every cell so the future rows code with the
        // same columns as the observed ones.
        let coding = Terms::new()
            .intercept()
            .factor_with_reference("origin", &origins[0].to_string())
            .factor_with_reference("development", &ages[0].to_string())
            .fit(cells.frame())?;
        let design = coding.design(&cells.select(observed)?)?;
        let glm = self.glm().fit(&design, &y)?;
        let future_design = coding.design(&cells.select(future)?)?;
        let future_means = glm.predict(&future_design)?;

        let mut reserves = vec![0.0; origins.len()];
        for (&r, m) in future.iter().zip(&future_means) {
            reserves[cells.origin_of(r)] += m;
        }
        let future = future
            .iter()
            .map(|&r| (origins[cells.origin_of(r)], ages[cells.development_of(r)]))
            .collect();
        Ok(OdpGlmFit {
            glm,
            coding,
            origins,
            future,
            future_means,
            reserves,
            future_design,
        })
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

    /// Mean of each future cell under
    /// [`predict_distribution`](Self::predict_distribution): `exp(η + v / 2)`,
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
    /// These are the GLM's predictive draws
    /// ([`GlmFit::predict_distribution`](act_models::Fitted::predict_distribution)):
    /// simulation `i` (stream `i` of `seed`) draws the coefficients from
    /// their normal approximation (parameter uncertainty, shared by every
    /// cell) and then each cell as `φ · Poisson(μ / φ)` (process
    /// uncertainty). `aggregate(&["origin"])` gives the reserves by origin
    /// with a future cell, and the total is the reserve.
    ///
    /// Fails if `n_sims` is 0 or the triangle has no future cells.
    pub fn predict_distribution(&self, n_sims: usize, seed: u64) -> Result<PredictiveDistribution> {
        if self.future.is_empty() {
            return Err(Error::OdpGlm(
                "the triangle is fully developed: there are no future cells to simulate".into(),
            ));
        }
        let draws = self
            .glm
            .predict_distribution(&self.future_design, n_sims, seed)?;
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
    use act_prob::Distribution;

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
        // The draws' mean is the lognormal-corrected mean, above the fitted
        // (Chain Ladder) reserve.
        let se = pd.std_dev() / 4_000f64.sqrt();
        let want: f64 = fit.predictive_means().iter().sum();
        assert!(
            (pd.mean() - want).abs() < 4.0 * se,
            "{} vs {want}",
            pd.mean()
        );
        assert!(want > fit.total_reserve());
        assert!(fit.predict_distribution(0, 7).is_err());
    }

    #[test]
    fn origin_with_only_future_cells_is_rejected() {
        let tri = annual(2020, &[&[100.0, 150.0], &[110.0], &[f64::NAN]]);
        let err = OdpGlm::default().fit(&tri, "values").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("origin 2022") && msg.contains("only in future"),
            "{msg}"
        );
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
    fn negative_increments_are_rejected() {
        let tri = annual(2020, &[&[100.0, 90.0], &[110.0]]);
        let msg = OdpGlm::default()
            .fit(&tri, "values")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("negative increment (-10)"), "{msg}");
    }

    #[test]
    fn fully_developed_triangle_has_no_distribution() {
        let tri = annual(2020, &[&[100.0, 150.0, 160.0], &[110.0, 160.0, 175.0]]);
        let fit = OdpGlm::default().fit(&tri, "values").unwrap();
        assert!(fit.future.is_empty());
        assert_eq!(fit.reserves, [0.0, 0.0]);
        let msg = fit.predict_distribution(100, 1).unwrap_err().to_string();
        assert!(msg.contains("fully developed"), "{msg}");
    }
}
