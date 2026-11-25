//! The age-period-cohort (APC) reserving GLM (`reserving-inflation.md`,
//! PR 6, issue #979).
//!
//! `log mu_ij = delta + alpha_i + beta_j + gamma_{i+j}` on the increments of
//! a full triangle, with the over-dispersed Poisson (or the gamma and
//! Tweedie errors of [`OdpGlm`]). The calendar index is the sum of the
//! accident and development indices, so the three effects are identified
//! only up to linear trends. The model is fitted in the canonical
//! parameterisation of Kuang, Nielsen and Nielsen (2008a), which has no
//! redundancy: a level and two slopes (the log means at accident 0 and age
//! 0, and the steps to accident 1 and to age 1), then the second
//! differences of each effect. On a `k` by `k` triangle that is `3k - 3`
//! parameters, and the design is the hinge functions
//! `(i - l + 1)+`, `(j - l + 1)+` and `(i + j - l + 1)+` for `l = 2..k - 1`
//! (the design of Python `apc`). Fitted values and the deviance do not
//! depend on the identification.
//!
//! The future cells need the period effect beyond the latest diagonal, the
//! second differences `DDgamma_l` for `l >= k`. Three rules extrapolate it
//! without depending on the identification (Kuang, Nielsen and Nielsen
//! 2008b; R `apc`'s `apc.forecast.apc`), with `n = k - 2` fitted second
//! differences `DD`:
//!
//! * [`Extrapolation::I2`]: the future second differences are 0, so the last
//!   linear trend of the period effect continues.
//! * [`Extrapolation::I1`]: with `D = cumsum(DD)`, the first future second
//!   difference is `mean(D) - D[n]`, then 0: the future first difference of
//!   the period effect is the mean of the fitted ones, with the
//!   first difference left out of the mean. That is R `apc`'s convention
//!   (`apc_forecast.R`, line 673), not an identification requirement: a
//!   change of identification moves every first difference by the same
//!   constant, so either mean would be invariant.
//! * [`Extrapolation::I0`] (R `apc`'s default): with `y = cumsum(cumsum(DD))`
//!   a straight line is fitted to `y` by least squares (an intercept and
//!   a demeaned trend), extrapolated two steps to `f1` and `f2`, and the
//!   first two future second differences are the second differences of
//!   `y, f1, f2` at their ends, then 0: the future period effects lie on
//!   the line fitted to the effects of the periods from the third on.
//!
//! All three are linear in the fitted second differences, so the future
//! rows of the design carry them as fixed weights on the period columns and
//! the fit goes through [`OdpGlm::fit_design`]: reserves, the delta-method
//! [`prediction_error`](ApcGlmFit::prediction_error) and
//! [`predict_distribution`](ApcGlmFit::predict_distribution) then cover the
//! extrapolated period effect as a function of the fitted parameters.
//! The sub-models drop a block of second differences: [`ApcEffects`].
//!
//! Like the separation method, the model needs a full triangle (decision 7).

use prospicio_core::{Lag, Period};
use prospicio_models::{Design, Family};
use prospicio_prob::PredictiveDistribution;

use crate::error::{Error, Result};
use crate::frame::TriangleFrame;
use crate::inflation::FutureCells;
use crate::odp_glm::{OdpGlm, OdpGlmFit, OdpPredictionError};
use crate::triangle::{Segment, Triangle};

fn apc_err<T>(why: impl Into<String>) -> Result<T> {
    Err(Error::Inflation(format!("apc: {}", why.into())))
}

/// Which of the three effects the model has: the full model or one of
/// its sub-models, as Python and R `apc` name them. The level and the two
/// slopes are always there; a sub-model drops one effect's second
/// differences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApcEffects {
    /// Age (development), period (calendar) and cohort (accident) effects.
    #[default]
    Apc,
    /// Accident and calendar effects (`PC`): no development effect.
    AccidentPeriod,
    /// Development and calendar effects (`AP`): Taylor's separation model
    /// as a GLM, with no accident effect.
    AgePeriod,
    /// Accident and development effects (`AC`): the chain ladder (the
    /// ODP GLM), with no calendar effect, so the extrapolation does not
    /// apply.
    AccidentAge,
}

impl ApcEffects {
    fn age(self) -> bool {
        matches!(self, Self::Apc | Self::AgePeriod | Self::AccidentAge)
    }
    fn accident(self) -> bool {
        matches!(self, Self::Apc | Self::AccidentPeriod | Self::AccidentAge)
    }
    fn calendar(self) -> bool {
        matches!(self, Self::Apc | Self::AccidentPeriod | Self::AgePeriod)
    }
}

/// The rule that extrapolates the period effect past the latest diagonal.
/// Each is invariant to the identification of the model (see the module
/// documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Extrapolation {
    /// Future second differences of the period effect are 0.
    I2,
    /// The first future second difference makes the next first difference
    /// the mean of the fitted ones, the first difference left out of the
    /// mean (R `apc`'s convention), then 0.
    I1,
    /// A line through the double cumulated second differences, extrapolated
    /// two steps, then 0 (R `apc`'s default). Needs at least 4 origins.
    #[default]
    I0,
}

/// The age-period-cohort reserving GLM.
///
/// ```no_run
/// use prospicio_reserving::{
///     ApcGlm, DevelopmentColumn, Extrapolation, Grain, Long, Month, Triangle,
/// };
///
/// let mut origin = vec![];
/// let mut age = vec![];
/// let mut value = vec![];
/// for i in 0..5u32 {
///     for j in 0..(5 - i) {
///         origin.push(Month::january(2001 + i as i32));
///         age.push(12 * (j + 1));
///         value.push(100.0 * (1.0 + 0.05 * f64::from(i + j)) * [0.5, 0.25, 0.12, 0.08, 0.05][j as usize]);
///     }
/// }
/// let tri = Triangle::from_long(&Long {
///     keys: &[],
///     origin: &origin,
///     development: DevelopmentColumn::Age(&age),
///     values: &[("paid", &value)],
///     origin_grain: Grain::Year,
///     development_grain: Grain::Year,
///     cumulative: false,
/// })?;
/// let fit = ApcGlm { extrapolation: Extrapolation::I1, ..ApcGlm::default() }.fit(&tri, "paid")?;
/// assert_eq!(fit.reserves.len(), 5);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApcGlm {
    /// The error distribution, as [`OdpGlm::family`]: Poisson (the
    /// default), gamma or Tweedie with `1 < p < 2`.
    pub family: Family,
    /// The effects in the model.
    pub effects: ApcEffects,
    /// The period effect's extrapolation (ignored by
    /// [`ApcEffects::AccidentAge`], which has no period effect).
    pub extrapolation: Extrapolation,
    /// IRLS convergence tolerance (see [`OdpGlm::tolerance`]); finite and
    /// positive.
    pub tolerance: f64,
    /// Maximum IRLS iterations, at least 1.
    pub max_iterations: usize,
}

impl Default for ApcGlm {
    /// The Poisson family, the full model, `I0`, and [`OdpGlm`]'s
    /// tolerance and iteration limit.
    fn default() -> Self {
        let odp = OdpGlm::default();
        Self {
            family: odp.family,
            effects: ApcEffects::Apc,
            extrapolation: Extrapolation::I0,
            tolerance: odp.tolerance,
            max_iterations: odp.max_iterations,
        }
    }
}

/// A fitted APC GLM.
#[derive(Debug, Clone)]
pub struct ApcGlmFit {
    /// The effects of the model.
    pub effects: ApcEffects,
    /// The extrapolation used for the future cells.
    pub extrapolation: Extrapolation,
    /// The GLM fit on the canonical design: coefficients named `level`,
    /// `slope_age`, `slope_accident`, then `dd_age[l]`, `dd_accident[l]`
    /// and `dd_calendar[l]` for `l = 2..k - 1` (the 0-based index of the
    /// effect's period), with their covariance, the future cells' means,
    /// reserves, residuals, prediction error and draws.
    pub fit: OdpGlmFit,
    /// The origins.
    pub origins: Vec<Period>,
    /// The development ages in months.
    pub ages: Vec<Lag>,
    /// The log mean of the first origin's first age.
    pub level: f64,
    /// The step in the log mean from age 0 to age 1 (the development
    /// slope), at origin 0.
    pub age_slope: f64,
    /// The step in the log mean from accident 0 to accident 1 (the
    /// accident slope), at age 0.
    pub accident_slope: f64,
    /// The development effect's second differences, `l = 2..k - 1`; empty
    /// when the model has no development effect.
    pub age_second_differences: Vec<f64>,
    /// The accident effect's second differences; empty without it.
    pub accident_second_differences: Vec<f64>,
    /// The calendar effect's fitted second differences; empty without it.
    pub calendar_second_differences: Vec<f64>,
    /// The two extrapolated calendar second differences, at `l = k` and
    /// `l = k + 1` (all later ones are 0); zeros for `I2` and without a
    /// calendar effect.
    pub future_calendar_second_differences: [f64; 2],
    /// The fitted mean of each observed cell, one row per origin and one
    /// value per age, `None` below the latest diagonal.
    pub fitted: Vec<Vec<Option<f64>>>,
    /// The GLM's deviance on the observed cells.
    pub deviance: f64,
    /// The sum of each origin's future means. The first origin's is 0:
    /// the model has no tail past the oldest age.
    pub reserves: Vec<f64>,
}

impl ApcGlmFit {
    /// The total reserve.
    pub fn total_reserve(&self) -> f64 {
        self.reserves.iter().sum()
    }

    /// The canonical coefficients, in the order of [`names`](Self::names).
    pub fn coefficients(&self) -> &[f64] {
        self.fit.glm.coefficients()
    }

    /// The canonical coefficients' names.
    pub fn names(&self) -> &[String] {
        self.fit.glm.names()
    }

    /// The coefficients' covariance, row-major (`p * p`).
    pub fn covariance(&self) -> Vec<f64> {
        self.fit.glm.covariance()
    }

    /// The reserves' prediction error by origin and in total (delta
    /// method and process variance; see [`OdpGlmFit::prediction_error`]).
    /// The extrapolated period effect is a fixed linear function of the
    /// fitted second differences, so its uncertainty is included.
    pub fn prediction_error(&self) -> Result<OdpPredictionError> {
        self.fit.prediction_error()
    }

    /// Joint draws of the future cells, as
    /// [`OdpGlmFit::predict_distribution`].
    pub fn predict_distribution(&self, n_sims: usize, seed: u64) -> Result<PredictiveDistribution> {
        self.fit.predict_distribution(n_sims, seed)
    }
}

impl FutureCells {
    /// An APC GLM fit's future means.
    ///
    /// Fails as [`FutureCells::from_odp_glm`] does.
    pub fn from_apc_glm(fit: &ApcGlmFit) -> Result<Self> {
        Self::from_odp_glm(&fit.fit)
    }
}

/// `(x - l + 1)+`: the hinge of the second difference at index `l`.
fn hinge(x: usize, l: usize) -> f64 {
    if x + 1 > l { (x + 1 - l) as f64 } else { 0.0 }
}

/// The first two future second differences from the `n` fitted ones (the
/// module documentation's rules).
fn extrapolate(rule: Extrapolation, dd: &[f64]) -> [f64; 2] {
    let n = dd.len();
    match rule {
        Extrapolation::I2 => [0.0, 0.0],
        Extrapolation::I1 => {
            let mut sum = 0.0;
            let mut d = 0.0;
            for v in dd {
                d += v;
                sum += d;
            }
            [sum / n as f64 - d, 0.0]
        }
        Extrapolation::I0 => {
            let mut y = Vec::with_capacity(n);
            let (mut d, mut b) = (0.0, 0.0);
            for v in dd {
                d += v;
                b += d;
                y.push(b);
            }
            let nf = n as f64;
            let mean = y.iter().sum::<f64>() / nf;
            let (mut sxy, mut sxx) = (0.0, 0.0);
            for (t, v) in y.iter().enumerate() {
                let x = (t + 1) as f64 - (nf + 1.0) / 2.0;
                sxy += x * v;
                sxx += x * x;
            }
            let slope = sxy / sxx;
            let f1 = mean + slope * (nf + 1.0) / 2.0;
            let f2 = mean + slope * (nf + 3.0) / 2.0;
            [f1 - 2.0 * y[n - 1] + y[n - 2], f2 - 2.0 * f1 + y[n - 1]]
        }
    }
}

/// The weights of the `n` fitted second differences in the two future
/// ones: `extrapolate` is linear, so a unit vector gives each column.
fn weights(rule: Extrapolation, n: usize) -> [Vec<f64>; 2] {
    let mut w = [vec![0.0; n], vec![0.0; n]];
    for m in 0..n {
        let mut unit = vec![0.0; n];
        unit[m] = 1.0;
        let f = extrapolate(rule, &unit);
        w[0][m] = f[0];
        w[1][m] = f[1];
    }
    w
}

/// The shape conditions of a full triangle (decision 7), as the separation
/// method's: as many ages as origins, both at one grain, the first age one
/// development period, every cell up to the latest diagonal and none past
/// it. Returns the number of origins.
fn check_full(seg: &Segment) -> Result<usize> {
    let k = seg.n_origins;
    let step = seg.origins[0].grain().months();
    if seg.n_dev != k {
        return apc_err(format!(
            "a full triangle has as many ages as origins; this has {} ages and {k} origins with a value (an origin with no value yet is left out)",
            seg.n_dev
        ));
    }
    if k < 3 {
        return Err(Error::TooFewAges {
            needed: 3,
            found: k,
        });
    }
    if seg.ages[0] != step {
        return apc_err(format!(
            "the first age must be one development period ({step} months), not {}",
            seg.ages[0]
        ));
    }
    if seg.ages[1] - seg.ages[0] != step {
        return apc_err(format!(
            "the development grain ({} months) must be the origin grain ({step} months)",
            seg.ages[1] - seg.ages[0]
        ));
    }
    for o in 0..k {
        for d in 0..k {
            match (seg.get(o, d).is_some(), o + d < k) {
                (true, false) => {
                    return apc_err(format!(
                        "origin {} has a cell past the latest diagonal",
                        seg.origins[o]
                    ));
                }
                (false, true) => {
                    return apc_err(format!(
                        "origin {} has no value at age {}: the model needs every cell up to the latest diagonal",
                        seg.origins[o], seg.ages[d]
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(k)
}

impl ApcGlm {
    /// Fits `column` of a single-segment triangle and projects the cells
    /// below the latest diagonal.
    ///
    /// Fails with [`Error::MultipleSegments`] on a triangle with several
    /// segments (fit one at a time), with [`Error::Inflation`] if it is not
    /// a full triangle (decision 7), with [`Error::TooFewAges`] on fewer
    /// than 3 origins (4 for `I0` with a period effect), with [`Error::InvalidSetting`] on a tolerance that is not
    /// finite and positive, no iterations, or a Tweedie power outside
    /// `(1, 2)`, with [`Error::NotSupported`] on a family other than the
    /// Poisson, gamma or Tweedie, and as [`OdpGlm::fit_design`] does on an
    /// increment the family cannot fit or an IRLS that does not converge.
    pub fn fit(&self, triangle: &Triangle, column: &str) -> Result<ApcGlmFit> {
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(Error::InvalidSetting {
                name: "tolerance",
                value: self.tolerance,
                expected: "finite and positive",
            });
        }
        if self.max_iterations == 0 {
            return Err(Error::InvalidSetting {
                name: "max_iterations",
                value: 0.0,
                expected: "at least 1",
            });
        }
        match self.family {
            Family::Poisson | Family::Gamma => {}
            Family::Tweedie { power } if power > 1.0 && power < 2.0 => {}
            Family::Tweedie { power } => {
                return Err(Error::InvalidSetting {
                    name: "tweedie_power",
                    value: power,
                    expected: "in (1, 2)",
                });
            }
            _ => {
                return Err(Error::NotSupported {
                    setting: "family",
                    method: "apc_glm",
                    reason: "a reserving error is the Poisson (ODP), gamma or Tweedie",
                });
            }
        }
        let seg = triangle.segment(column)?;
        let k = check_full(&seg)?;
        let (age, accident, calendar) = (
            self.effects.age(),
            self.effects.accident(),
            self.effects.calendar(),
        );
        if calendar && self.extrapolation == Extrapolation::I0 && k < 4 {
            return Err(Error::TooFewAges {
                needed: 4,
                found: k,
            });
        }
        let cells = TriangleFrame::new(triangle, column, None)?;
        if cells.origins().len() != k || cells.ages().len() != k {
            return apc_err(format!(
                "the triangle has {} origins and {} ages, but a full {k} by {k} triangle was expected",
                cells.origins().len(),
                cells.ages().len()
            ));
        }

        let n_dd = k - 2;
        let n_params = 3 + n_dd * [age, accident, calendar].iter().filter(|b| **b).count();
        if n_params >= k * (k + 1) / 2 {
            return apc_err(format!(
                "{n_params} parameters for {} observed cells leave no residual degrees of freedom;                  this model needs a larger triangle (4 origins for the full model)",
                k * (k + 1) / 2
            ));
        }
        let w = if calendar {
            weights(self.extrapolation, n_dd)
        } else {
            [vec![0.0; n_dd], vec![0.0; n_dd]]
        };
        let rows = cells.n_rows();
        let at = |r: usize| (cells.origin_of(r), cells.development_of(r));
        let mut names = vec![
            "level".to_string(),
            "slope_age".to_string(),
            "slope_accident".to_string(),
        ];
        let mut columns: Vec<Vec<f64>> = vec![
            vec![1.0; rows],
            (0..rows).map(|r| at(r).1 as f64).collect(),
            (0..rows).map(|r| at(r).0 as f64).collect(),
        ];
        if self.effects.age() {
            for l in 2..k {
                names.push(format!("dd_age[{l}]"));
                columns.push((0..rows).map(|r| hinge(at(r).1, l)).collect());
            }
        }
        if self.effects.accident() {
            for l in 2..k {
                names.push(format!("dd_accident[{l}]"));
                columns.push((0..rows).map(|r| hinge(at(r).0, l)).collect());
            }
        }
        if calendar {
            for l in 2..k {
                names.push(format!("dd_calendar[{l}]"));
                columns.push(
                    (0..rows)
                        .map(|r| {
                            let (i, j) = at(r);
                            let h = i + j;
                            hinge(h, l) + w[0][l - 2] * hinge(h, k) + w[1][l - 2] * hinge(h, k + 1)
                        })
                        .collect(),
                );
            }
        }
        let design = Design::new(names, columns).map_err(Error::from)?;
        let odp = OdpGlm {
            family: self.family,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
        };
        let fit = odp.fit_design(triangle, column, &design)?;

        let coef = fit.glm.coefficients();
        let mut at_coef = 3;
        let mut block = |present: bool| -> Vec<f64> {
            if present {
                let v = coef[at_coef..at_coef + n_dd].to_vec();
                at_coef += n_dd;
                v
            } else {
                Vec::new()
            }
        };
        let age_dd = block(self.effects.age());
        let accident_dd = block(self.effects.accident());
        let calendar_dd = block(calendar);
        let future_dd = if calendar {
            [
                w[0].iter().zip(&calendar_dd).map(|(a, b)| a * b).sum(),
                w[1].iter().zip(&calendar_dd).map(|(a, b)| a * b).sum(),
            ]
        } else {
            [0.0; 2]
        };

        let mut fitted = vec![vec![None; k]; k];
        for (&(origin, age), &mu) in fit.observed.iter().zip(fit.glm.fitted()) {
            let o = fit.origins.iter().position(|&p| p == origin);
            let d = cells.ages().iter().position(|&a| a == age);
            if let (Some(o), Some(d)) = (o, d) {
                fitted[o][d] = Some(mu);
            }
        }
        Ok(ApcGlmFit {
            effects: self.effects,
            extrapolation: self.extrapolation,
            origins: fit.origins.clone(),
            ages: cells.ages().to_vec(),
            level: coef[0],
            age_slope: coef[1],
            accident_slope: coef[2],
            age_second_differences: age_dd,
            accident_second_differences: accident_dd,
            calendar_second_differences: calendar_dd,
            future_calendar_second_differences: future_dd,
            fitted,
            deviance: fit.glm.deviance(),
            reserves: fit.reserves.clone(),
            fit,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain_ladder::ChainLadder;
    use crate::separation::{FutureInflation, Separation, SeparationKind};
    use crate::triangle::tests::{annual, genins};

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-300)
    }

    fn apc(effects: ApcEffects, extrapolation: Extrapolation) -> ApcGlm {
        ApcGlm {
            effects,
            extrapolation,
            ..ApcGlm::default()
        }
    }

    #[test]
    fn the_accident_age_model_is_the_chain_ladder() {
        let tri = genins();
        let fit = apc(ApcEffects::AccidentAge, Extrapolation::I0)
            .fit(&tri, "values")
            .unwrap();
        let cl = ChainLadder::default().fit(&tri, "values").unwrap();
        for (got, want) in fit.reserves.iter().zip(cl.reserves()) {
            assert!(close(*got, want, 1e-8), "{got} {want}");
        }
        assert_eq!(fit.coefficients().len(), 2 * 10 - 1);
        assert!(fit.calendar_second_differences.is_empty());
        // The extrapolation is not used.
        let i2 = apc(ApcEffects::AccidentAge, Extrapolation::I2)
            .fit(&tri, "values")
            .unwrap();
        assert!(close(i2.total_reserve(), fit.total_reserve(), 1e-12));
    }

    #[test]
    fn the_age_period_model_is_the_separation_model() {
        let tri = genins();
        let fit = apc(ApcEffects::AgePeriod, Extrapolation::I2)
            .fit(&tri, "values")
            .unwrap();
        let sep = Separation {
            kind: SeparationKind::Arithmetic,
            normaliser: None,
            future: FutureInflation::Rate(0.0),
        }
        .fit(&tri, "values")
        .unwrap();
        assert_eq!(fit.coefficients().len(), 2 * 10 - 1);
        for (a, b) in fit.fitted.iter().zip(&sep.fitted) {
            for (x, y) in a.iter().zip(b) {
                match (x, y) {
                    (Some(x), Some(y)) => assert!(close(*x, *y, 1e-8), "{x} {y}"),
                    (None, None) => {}
                    other => panic!("{other:?}"),
                }
            }
        }
    }

    /// A second identification of the full model: the slopes of the
    /// accident and the calendar index (instead of the age and accident
    /// indices), the same hinges. Fitted through `fit_design`; its
    /// calendar effect `gamma_h` is extrapolated by the identification-free
    /// forms of the rules: `I2` continues the last two effects linearly,
    /// `I1` continues with the mean of the first differences from the
    /// second on, `I0` puts every future effect on the least-squares line
    /// through the effects of the periods from the third on.
    #[test]
    fn the_reserves_do_not_depend_on_the_identification() {
        let tri = genins();
        let k = 10;
        let cells = TriangleFrame::new(&tri, "values", None).unwrap();
        let rows = cells.n_rows();
        let at = |r: usize| (cells.origin_of(r), cells.development_of(r));
        let mut names = vec!["c".to_string(), "i".into(), "h".into()];
        let mut columns: Vec<Vec<f64>> = vec![
            vec![1.0; rows],
            (0..rows).map(|r| at(r).0 as f64).collect(),
            (0..rows).map(|r| (at(r).0 + at(r).1) as f64).collect(),
        ];
        for l in 2..k {
            names.push(format!("a{l}"));
            columns.push((0..rows).map(|r| hinge(at(r).1, l)).collect());
        }
        for l in 2..k {
            names.push(format!("c{l}"));
            columns.push((0..rows).map(|r| hinge(at(r).0, l)).collect());
        }
        for l in 2..k {
            names.push(format!("g{l}"));
            columns.push((0..rows).map(|r| hinge(at(r).0 + at(r).1, l)).collect());
        }
        let design = Design::new(names, columns).unwrap();
        let other = OdpGlm::default()
            .fit_design(&tri, "values", &design)
            .unwrap();
        let b = other.glm.coefficients();
        // gamma_h = p h + sum_l (h - l + 1)+ DD_l on the observed periods.
        let gamma = |h: usize| {
            b[2] * h as f64
                + (2..k)
                    .map(|l| hinge(h, l) * b[3 + 2 * (k - 2) + l - 2])
                    .sum::<f64>()
        };
        let observed: Vec<f64> = (0..k).map(gamma).collect();
        let future_gamma = |rule: Extrapolation| -> Vec<f64> {
            let mut g = observed.clone();
            match rule {
                Extrapolation::I2 => {
                    let d = g[k - 1] - g[k - 2];
                    for h in k..2 * k - 1 {
                        g.push(g[h - 1] + d);
                    }
                }
                Extrapolation::I1 => {
                    let d: f64 = (2..k).map(|h| g[h] - g[h - 1]).sum::<f64>() / (k - 2) as f64;
                    for h in k..2 * k - 1 {
                        g.push(g[h - 1] + d);
                    }
                }
                Extrapolation::I0 => {
                    let xs: Vec<f64> = (2..k).map(|h| h as f64).collect();
                    let ys: Vec<f64> = (2..k).map(|h| g[h]).collect();
                    let n = xs.len() as f64;
                    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
                    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
                    let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
                    let slope = sxy / sxx;
                    for h in k..2 * k - 1 {
                        g.push(my + slope * (h as f64 - mx));
                    }
                }
            }
            g
        };
        for rule in [Extrapolation::I2, Extrapolation::I1, Extrapolation::I0] {
            let fit = apc(ApcEffects::Apc, rule).fit(&tri, "values").unwrap();
            // Fitted values.
            let mut observed_fitted = Vec::new();
            for row in &fit.fitted {
                observed_fitted.extend(row.iter().flatten());
            }
            for (got, want) in observed_fitted.iter().zip(other.glm.fitted()) {
                assert!(close(*got, *want, 1e-8), "{rule:?} fitted {got} {want}");
            }
            assert!(close(fit.deviance, other.glm.deviance(), 1e-8));
            // Reserves: the other identification's cells, with the future
            // effect from its own estimates.
            let g = future_gamma(rule);
            let mut reserves = vec![0.0; k];
            for i in 1..k {
                for j in (k - i)..k {
                    let eta = b[0]
                        + b[1] * i as f64
                        + (2..k).map(|l| hinge(j, l) * b[3 + l - 2]).sum::<f64>()
                        + (2..k)
                            .map(|l| hinge(i, l) * b[3 + (k - 2) + l - 2])
                            .sum::<f64>()
                        + g[i + j];
                    reserves[i] += eta.exp();
                }
            }
            for (got, want) in fit.reserves.iter().zip(&reserves) {
                assert!(close(*got, *want, 1e-7), "{rule:?} reserve {got} {want}");
            }
        }
    }

    #[test]
    fn the_rules_are_ordered_as_documented_on_taylor_ashe() {
        let tri = genins();
        let total = |r| {
            apc(ApcEffects::Apc, r)
                .fit(&tri, "values")
                .unwrap()
                .total_reserve()
        };
        // R apc documents 19,061,428 for I0.
        assert!((total(Extrapolation::I0) - 19_061_428.0).abs() < 1.0);
        assert!(total(Extrapolation::I2) > total(Extrapolation::I1));
        assert!(total(Extrapolation::I1) > total(Extrapolation::I0));
        let fit = apc(ApcEffects::Apc, Extrapolation::I0)
            .fit(&tri, "values")
            .unwrap();
        assert_eq!(fit.coefficients().len(), 3 * 10 - 3);
        assert_eq!(fit.names()[0], "level");
        assert_eq!(fit.reserves[0], 0.0);
        // The future cells and the prediction error and draws that cover them.
        let cells = FutureCells::from_apc_glm(&fit).unwrap();
        assert_eq!(cells.cells.len(), 45);
        for (got, want) in cells.reserves().iter().zip(&fit.reserves) {
            assert!(close(*got, *want, 1e-12));
        }
        let error = fit.prediction_error().unwrap();
        assert!(close(error.total.total, fit.total_reserve(), 1e-9));
        assert!(error.total.rmsep() > 0.0);
        let draws = fit.predict_distribution(200, 7).unwrap();
        assert_eq!(draws.n_components(), 45);
    }

    #[test]
    fn the_extrapolation_widens_the_uncertainty() {
        // I0 and I1 depend on all fitted second differences, I2 on the
        // last, so their parameter variances differ.
        let tri = genins();
        let rmsep = |r| {
            apc(ApcEffects::Apc, r)
                .fit(&tri, "values")
                .unwrap()
                .prediction_error()
                .unwrap()
                .total
                .rmsep()
        };
        assert!(rmsep(Extrapolation::I2) != rmsep(Extrapolation::I0));
    }

    #[test]
    fn settings_and_shapes_are_checked() {
        let tri = genins();
        let bad = |f: ApcGlm| f.fit(&tri, "values").unwrap_err();
        assert!(matches!(
            bad(ApcGlm {
                tolerance: 0.0,
                ..ApcGlm::default()
            }),
            Error::InvalidSetting { .. }
        ));
        assert!(matches!(
            bad(ApcGlm {
                max_iterations: 0,
                ..ApcGlm::default()
            }),
            Error::InvalidSetting { .. }
        ));
        assert!(matches!(
            bad(ApcGlm {
                family: Family::Tweedie { power: 2.5 },
                ..ApcGlm::default()
            }),
            Error::InvalidSetting { .. }
        ));
        assert!(matches!(
            bad(ApcGlm {
                family: Family::Gaussian,
                ..ApcGlm::default()
            }),
            Error::NotSupported { .. }
        ));
        assert!(matches!(
            ApcGlm::default().fit(&tri, "nope"),
            Err(Error::UnknownColumn(_))
        ));
        // Not a full triangle.
        let ragged = annual(2001, &[&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &[1.0]]);
        assert!(matches!(
            ApcGlm::default().fit(&ragged, "values"),
            Err(Error::Inflation(_))
        ));
        // A hole.
        let quarterly = Triangle::from_long(&crate::triangle::Long {
            keys: &[],
            origin: &[2001, 2001, 2002].map(crate::Month::january),
            development: crate::triangle::DevelopmentColumn::Age(&[3, 6, 3]),
            values: &[("values", &[1.0, 2.0, 1.0])],
            origin_grain: crate::Grain::Year,
            development_grain: crate::Grain::Quarter,
            cumulative: true,
        })
        .unwrap();
        assert!(ApcGlm::default().fit(&quarterly, "values").is_err());
        // Too small: 3 origins, and I0 wants 4.
        let small = annual(2001, &[&[10.0, 15.0, 17.0], &[11.0, 16.0], &[12.0]]);
        assert!(matches!(
            ApcGlm::default().fit(&small, "values"),
            Err(Error::TooFewAges { needed: 4, .. })
        ));
        let two = annual(2001, &[&[10.0, 15.0], &[11.0]]);
        assert!(matches!(
            apc(ApcEffects::Apc, Extrapolation::I2).fit(&two, "values"),
            Err(Error::TooFewAges { needed: 3, .. })
        ));
    }

    #[test]
    fn a_saturated_model_is_refused_and_the_others_have_a_prediction_error() {
        let tri = annual(2001, &[&[10.0, 15.0, 17.0], &[11.0, 16.0], &[12.0]]);
        // The full model on 3 origins has 6 parameters for 6 cells.
        assert!(matches!(
            apc(ApcEffects::Apc, Extrapolation::I1).fit(&tri, "values"),
            Err(Error::Inflation(_))
        ));
        for effects in [
            ApcEffects::AgePeriod,
            ApcEffects::AccidentPeriod,
            ApcEffects::AccidentAge,
        ] {
            let fit = apc(effects, Extrapolation::I1).fit(&tri, "values").unwrap();
            let e = fit.prediction_error().unwrap().total;
            assert!(e.rmsep().is_finite() && e.rmsep() > 0.0, "{effects:?}");
        }
    }

    #[test]
    fn a_zero_increment_is_fitted_by_the_poisson_and_tweedie_not_the_gamma() {
        let zero = genins_with_zero();
        let fit = |family| {
            ApcGlm {
                family,
                ..ApcGlm::default()
            }
            .fit(&zero, "values")
        };
        assert!(fit(Family::Poisson).unwrap().total_reserve().is_finite());
        assert!(
            fit(Family::Tweedie { power: 1.5 })
                .unwrap()
                .total_reserve()
                .is_finite()
        );
        assert!(matches!(fit(Family::Gamma), Err(Error::OdpGlm(_))));
    }

    fn genins_with_zero() -> Triangle {
        // Cumulative with a zero increment at origin 2001, age 24.
        let mut rows: Vec<Vec<f64>> = crate::triangle::tests::GENINS
            .iter()
            .map(|r| r.to_vec())
            .collect();
        let first = rows[0][0];
        let tail: Vec<f64> = rows[0][1..].to_vec();
        let delta = rows[0][1] - first;
        rows[0][1] = first;
        for (v, t) in rows[0][2..].iter_mut().zip(&tail[1..]) {
            *v = t - delta;
        }
        let refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        annual(2001, &refs)
    }
}
